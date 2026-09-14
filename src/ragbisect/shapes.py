"""The three structured question shapes (M4): comparison, aggregation, table.

Conceptual and exact-lookup questions come from one chunk each and need a
model to phrase them. These three are different: their ground truth is
computed from corpus structure, and two of them need no model at all.

* comparison  — two chunks from different documents with high lexical overlap
                but differing values (versions of the same text). The model
                phrases a question that needs both; gold is both ids.
* aggregation — an identifier family (TX-4401, TX-4419, TX-4420 …) spread over
                several chunks. The question asks how many distinct members
                exist; the answer is counted programmatically; gold is every
                chunk that contributes a member. No model call.
* table       — a pipe table row and column chosen programmatically; the
                question asks for that cell; gold is the chunk with the row
                plus the chunk with the header if the chunker separated them.
                No model call.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass

from .corpus import Chunk
from .generate import IDENTIFIER_PATTERNS, Question, _clean_q
from .llm import LLM, LLMError
from .retrievers import tokenize

_IDENT_RE = re.compile("|".join(IDENTIFIER_PATTERNS))
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


# ------------------------------------------------------------------ comparison

COMPARISON_SYSTEM = """You write evaluation questions for a document search system. You are given two passages from two different documents that are near-duplicates: the same topic, with some values or rules changed (typically two versions or two variants).

Write ONE question a real user would ask that can only be answered by reading BOTH passages and comparing them: what changed, which differs, how they contrast. Name the two variants the way the passages do (years, versions, product names, regions). Never say "the passages" or "the documents".

Return JSON: {"usable": true, "q": "...", "a": "short answer naming both values"} or {"usable": false} if the passages are not genuinely comparable variants."""


def _num_set(text: str) -> set[str]:
    return set(_NUM_RE.findall(text))


def comparison_pairs(chunks: list[Chunk], max_pairs: int, seed: int, lo: float = 0.45, hi: float = 0.95) -> list[tuple[Chunk, Chunk, float]]:
    """Cross-document pairs with token Jaccard in [lo, hi] and differing numbers,
    best overlap first. O(n²) on token sets; fine for tens of thousands of chunks
    when bucketed by shared rare tokens, but v1 caps the candidate set."""
    toks = {c.id: set(tokenize(c.text)) for c in chunks}
    by_source: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        by_source[c.source].append(c)
    sources = sorted(by_source)
    pairs: list[tuple[Chunk, Chunk, float]] = []
    # Candidate generation: inverted index on rarer tokens to avoid the full n².
    df: dict[str, int] = defaultdict(int)
    for s in toks.values():
        for t in s:
            df[t] += 1
    index: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        for t in toks[c.id]:
            if 2 <= df[t] <= max(20, len(chunks) // 20):
                index[t].append(c)
    seen: set[tuple[str, str]] = set()
    for t, cs in index.items():
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                a, b = cs[i], cs[j]
                if a.source == b.source:
                    continue
                key = (a.id, b.id) if a.id < b.id else (b.id, a.id)
                if key in seen:
                    continue
                seen.add(key)
                sa, sb = toks[a.id], toks[b.id]
                if not sa or not sb:
                    continue
                jac = len(sa & sb) / len(sa | sb)
                if lo <= jac <= hi and _num_set(a.text) != _num_set(b.text):
                    pairs.append((a, b, jac) if a.id < b.id else (b, a, jac))
    pairs.sort(key=lambda p: (-p[2], p[0].id, p[1].id))
    # One pair per (source_a, source_b) chunk: avoid ten questions about the same two documents' identical structure.
    out, used = [], set()
    for a, b, jac in pairs:
        if a.id in used or b.id in used:
            continue
        used.update((a.id, b.id))
        out.append((a, b, jac))
        if len(out) >= max_pairs:
            break
    return out


def generate_comparisons(chunks: list[Chunk], llm: LLM, max_pairs: int, seed: int) -> tuple[list[Question], str | None]:
    pairs = comparison_pairs(chunks, max_pairs, seed)
    if not pairs:
        return [], "no cross-document near-duplicate chunks with differing values (no versioned or variant documents)"
    out: list[Question] = []
    for a, b, jac in pairs:
        prompt = f"Passage A ({a.source}):\n{a.text}\n\nPassage B ({b.source}):\n{b.text}"
        try:
            obj = llm.complete_json(prompt, system=COMPARISON_SYSTEM)
        except LLMError:
            continue
        if not obj.get("usable", False):
            continue
        q = _clean_q(obj.get("q"))
        if q:
            out.append(Question("", q, "comparison", [a.id, b.id], str(obj.get("a", "")), f"{a.source} | {b.source}"))
    return out, None


# ----------------------------------------------------------------- aggregation


def _family_key(ident: str) -> str:
    return re.sub(r"\d", "#", ident.strip("`"))


def identifier_families(chunks: list[Chunk], min_members: int = 3, min_chunks: int = 2) -> list[tuple[str, dict[str, set[str]]]]:
    """Families of identifiers that share a shape (TX-#### , R-##, v#.#) with at
    least ``min_members`` distinct members spread over ``min_chunks`` chunks.
    Returns (family_key, {member: {chunk ids}})."""
    fam: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for c in chunks:
        for m in _IDENT_RE.finditer(c.text):
            ident = m.group(0).strip("`")
            if not re.search(r"\d", ident) or len(ident) < 3 or "." in ident and ident.replace(".", "").isdigit():
                continue  # needs digits to form a family; bare version numbers are too generic
            if re.search(r"[\s=\"'(),:]", ident) or len(ident) > 24:
                continue  # a family member is a compact token (TX-4419, R-40, ERR_203), not an expression
            if ident.startswith("-"):
                continue
            fam[_family_key(ident)][ident].add(c.id)
    out = []
    for key, members in fam.items():
        chunk_ids = set().union(*members.values())
        if len(members) >= min_members and len(chunk_ids) >= min_chunks and "#" in key and len(key.replace("#", "")) >= 1:
            out.append((key, members))
    out.sort(key=lambda kv: (-len(kv[1]), kv[0]))
    return out


def generate_aggregations(chunks: list[Chunk], max_questions: int, seed: int, max_gold: int = 8) -> tuple[list[Question], str | None]:
    fams = [f for f in identifier_families(chunks) if len(set().union(*f[1].values())) <= max_gold]
    if not fams:
        return [], "no identifier families (codes, part numbers, versions) spread across several chunks"
    rng = random.Random(seed)
    out: list[Question] = []
    for key, members in fams[:max_questions]:
        examples = sorted(members)[:2]
        gold = sorted(set().union(*members.values()))
        pattern = key.replace("#", "N")
        q = f"How many distinct identifiers of the form {pattern} (for example {', '.join(examples)}) are documented, and what are they?"
        a = f"{len(members)}: {', '.join(sorted(members))}"
        out.append(Question("", q, "aggregation", gold, a, ", ".join(sorted({cid.rsplit('#', 1)[0] for cid in gold}))))
    return out, None


# ----------------------------------------------------------------------- table


@dataclass
class TableSpan:
    chunk: Chunk
    header: list[str]
    rows: list[list[str]]
    header_chunk_id: str | None  # a different chunk when the chunker split the header away


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def find_tables(chunks: list[Chunk]) -> list[TableSpan]:
    by_source: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        by_source[c.source].append(c)
    spans: list[TableSpan] = []
    for source, cs in by_source.items():
        for idx, c in enumerate(cs):
            lines = c.text.splitlines()
            i = 0
            while i < len(lines):
                if "|" in lines[i] and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1]) and "|" in lines[i + 1]:
                    header = _cells(lines[i])
                    j = i + 2
                    rows = []
                    while j < len(lines) and "|" in lines[j] and lines[j].strip():
                        cells = _cells(lines[j])
                        if len(cells) == len(header):
                            rows.append(cells)
                        j += 1
                    if len(header) >= 2 and len(rows) >= 2:
                        spans.append(TableSpan(c, header, rows, None))
                    i = j
                    continue
                if "|" in lines[i] and idx > 0 and i == 0:
                    # rows with no header in this chunk: look back for the header
                    prev = cs[idx - 1]
                    plines = prev.text.splitlines()
                    for k in range(len(plines) - 1, 0, -1):
                        if _TABLE_SEP.match(plines[k]) and "|" in plines[k - 1]:
                            header = _cells(plines[k - 1])
                            rows = []
                            j = i
                            while j < len(lines) and "|" in lines[j] and lines[j].strip():
                                cells = _cells(lines[j])
                                if len(cells) == len(header):
                                    rows.append(cells)
                                j += 1
                            if len(header) >= 2 and len(rows) >= 1:
                                spans.append(TableSpan(c, header, rows, prev.id))
                            i = j
                            break
                    else:
                        i += 1
                    continue
                i += 1
    return spans


def generate_table_questions(chunks: list[Chunk], max_questions: int, seed: int) -> tuple[list[Question], str | None]:
    spans = find_tables(chunks)
    if not spans:
        return [], "no pipe tables with a header and at least two rows"
    
    rng = random.Random(seed)
    out: list[Question] = []
    # A row key that heads rows in more than one table (the same constraints table
    # repeated per type, say) makes the cell question answerable from several
    # sections; the single gold would be arbitrary. Only unique keys are asked about.
    key_tables: dict[str, set[str]] = defaultdict(set)
    for sp in spans:
        for r in sp.rows:
            key_tables[" ".join(r[0].replace("`", "").replace("**", "").split()).lower()].add(sp.chunk.id)
    per_table = max(1, max_questions // len(spans))
    for span in spans:
        unique_rows = [r for r in span.rows if len(key_tables[" ".join(r[0].replace("`", "").replace("**", "").split()).lower()]) == 1]
        if not unique_rows:
            continue
        picks = rng.sample(unique_rows, min(per_table, len(unique_rows)))
        for row in picks:
            cols = [i for i in range(1, len(span.header)) if row[i] and span.header[i]]
            if not cols:
                continue
            col = rng.choice(cols)
            def clean(x: str) -> str:
                x = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", x)   # [text][ref]  -> text
                x = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", x)   # [text](url)  -> text
                return " ".join(x.replace("`", "").replace("**", "").split())
            key, col_name, value = clean(row[0]), clean(span.header[col]), clean(row[col])
            if not key or not value or len(key) > 60:
                continue
            q = f"What is the {col_name} for {key}?"
            gold = [span.chunk.id] + ([span.header_chunk_id] if span.header_chunk_id else [])
            out.append(Question("", q, "table", gold, value, span.chunk.source))
            if len(out) >= max_questions:
                return out, None
    if not out:
        return [], "tables found, but every row key repeats across tables, so no cell has a single answering section"
    return out, None
