"""Eval-set generator (M1): builds questions from the corpus, no hand labelling.

Shapes implemented here: ``conceptual`` and ``exact_lookup``. For each sampled
chunk we make one LLM call that returns a conceptual question and, when the
chunk contains an identifier (flag, error code, env var, version, command),
an exact-lookup question too. Ground truth is the chunk ID.

Comparison, aggregation and table shapes need structured extraction and
belong to M4.
"""

from __future__ import annotations

import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .corpus import Chunk, corpus_fingerprint
from .llm import LLM, LLMError

SHAPES = ("conceptual", "exact_lookup", "comparison", "aggregation", "table")
MIN_SOURCE_CHARS = 200

# Things that look like identifiers a user would type verbatim.
IDENTIFIER_PATTERNS = (
    r"(?<![\w-])--?[a-z][a-z0-9]+(?:-[a-z0-9]+)+\b",  # --no-cache, -p
    r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b",             # UV_CACHE_DIR, ERR_CODE
    r"\b[A-Z]{1,6}-\d{2,}\b",                          # TX-4419
    r"\b\d+\.\d+(?:\.\d+)?\b",                         # 3.12, 0.4.1
    r"`[^`\n]{3,40}`",                                 # `uv pip sync`
    r"\b[a-z]+\.[a-z]+(?:\.[a-z]+)*\b",                # tool.uv.sources
)
_IDENT_RE = re.compile("|".join(IDENTIFIER_PATTERNS))


@dataclass
class Question:
    id: str
    q: str
    shape: str
    gold_chunk_ids: list[str]
    answer: str = ""
    source: str = ""
    identifier: str = ""  # exact_lookup only: the token the question hinges on

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GenerationReport:
    corpus_fingerprint: str
    model: str
    n_chunks: int
    n_eligible_chunks: int
    n_chunks_used: int
    n_unusable_chunks: int
    counts: dict[str, int] = field(default_factory=dict)
    skipped_shapes: dict[str, str] = field(default_factory=dict)
    errors: int = 0
    first_error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


SYSTEM = """You write evaluation questions for a document search system. You are given one chunk of text from a larger corpus.

Write questions the way a real user of this knowledge base would type them into a search box: self-contained, specific, and natural. The person asking has NOT seen the chunk.

Rules:
- Never refer to "this section", "the text", "the document", "above", or "the following". The question must stand alone.
- The question must be answerable from this chunk alone, and specific enough that a generic chunk on a nearby topic would NOT answer it.
- Do not copy a whole sentence from the chunk as the question.
- Keep questions under 25 words.

Two shapes:
1. "conceptual": asks about a behaviour, approach, reason, or how-to. Paraphrase; avoid quoting rare identifiers verbatim. Example: "How does the tool decide which Python version to use when none is specified?"
2. "exact_lookup": the question MUST contain one specific identifier from the chunk verbatim — a command-line flag, error code, environment variable, version number, setting name, or command — and ask what it does or means. Example: "What does the UV_CACHE_DIR environment variable control?" Only produce this if the chunk actually explains such an identifier; otherwise return null.

If the chunk is navigation, a table of contents, boilerplate, a bare code block with no explanation, or too fragmentary to support a good question, set "usable" to false.

Return JSON exactly like:
{"usable": true, "conceptual": {"q": "...", "a": "short answer"}, "exact_lookup": {"q": "...", "a": "short answer", "identifier": "..."} or null}"""


def has_identifier(text: str) -> bool:
    return _IDENT_RE.search(text) is not None


def _prompt(chunk: Chunk, want_exact: bool) -> str:
    hint = (
        "This chunk appears to contain identifiers; produce an exact_lookup question if one is genuinely explained."
        if want_exact
        else "This chunk has no obvious identifiers; set exact_lookup to null."
    )
    return f"Source file: {chunk.source}\n{hint}\n\n<chunk>\n{chunk.text}\n</chunk>"


def _clean_q(q: object) -> str | None:
    if not isinstance(q, str):
        return None
    q = " ".join(q.split()).strip()
    if len(q) < 12 or len(q) > 300:
        return None
    lowered = q.lower()
    if any(bad in lowered for bad in ("this section", "this document", "the text", "this chunk", "the passage", "this text")):
        return None
    return q


def generate_for_chunk(llm: LLM, chunk: Chunk) -> tuple[list[Question], bool]:
    """Returns (questions, usable). IDs are assigned by the caller."""
    want_exact = has_identifier(chunk.text)
    obj = llm.complete_json(_prompt(chunk, want_exact), system=SYSTEM)
    if not obj.get("usable", True):
        return [], False
    out: list[Question] = []
    c = obj.get("conceptual")
    if isinstance(c, dict):
        q = _clean_q(c.get("q"))
        if q:
            out.append(Question("", q, "conceptual", [chunk.id], str(c.get("a", "")), chunk.source))
    e = obj.get("exact_lookup")
    if want_exact and isinstance(e, dict):
        q = _clean_q(e.get("q"))
        ident = str(e.get("identifier", "")).strip()
        # The identifier must really be in the chunk and in the question; otherwise it's a fabricated lookup.
        if q and ident and ident in chunk.text and ident.lower() in q.lower():
            out.append(Question("", q, "exact_lookup", [chunk.id], str(e.get("a", "")), chunk.source, ident))
    return out, True


def generate(
    chunks: list[Chunk],
    llm: LLM,
    n: int = 200,
    seed: int = 0,
    workers: int = 8,
    progress: Callable[[int, int], None] | None = None,
    common_identifier_max_fraction: float = 0.05,
) -> tuple[list[Question], GenerationReport]:
    """Generate at least ``n`` questions (or as many as the corpus supports)."""
    eligible = [c for c in chunks if len(c.text) >= MIN_SOURCE_CHARS]
    rng = random.Random(seed)
    order = list(eligible)
    rng.shuffle(order)

    report = GenerationReport(
        corpus_fingerprint=corpus_fingerprint(chunks),
        model=f"{llm.provider}:{llm.model}",
        n_chunks=len(chunks),
        n_eligible_chunks=len(eligible),
        n_chunks_used=0,
        n_unusable_chunks=0,
    )
    for shape in ("comparison", "aggregation", "table"):
        report.skipped_shapes[shape] = "not implemented in this version (planned: M4)"

    results: dict[int, tuple[list[Question], bool] | None] = {}
    questions: list[Question] = []
    cursor = 0
    done = 0
    # Work in batches so we stop once we have enough, keeping cost proportional to n.
    batch = max(workers * 4, 16)
    while len(questions) < n and cursor < len(order):
        idx = list(range(cursor, min(cursor + batch, len(order))))
        cursor += len(idx)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(generate_for_chunk, llm, order[i]): i for i in idx}
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    results[i] = fut.result()
                except LLMError as e:
                    results[i] = None
                    report.errors += 1
                    if report.first_error is None:
                        report.first_error = str(e)[:300]
                done += 1
                if progress:
                    progress(done, len(order))
        if report.n_chunks_used == 0 and all(results.get(i) is None for i in idx):
            raise LLMError(f"every call in the first batch failed; first error: {report.first_error}")
        # Consume in deterministic order regardless of completion order.
        for i in idx:
            res = results.get(i)
            if res is None:
                continue
            qs, usable = res
            report.n_chunks_used += 1
            if not usable:
                report.n_unusable_chunks += 1
            questions.extend(qs)

    questions = _drop_common_identifiers(questions, chunks, max_fraction=common_identifier_max_fraction)
    questions = _dedupe(questions)
    for i, q in enumerate(questions, 1):
        q.id = f"q{i:04d}"
    for shape in ("conceptual", "exact_lookup"):
        report.counts[shape] = sum(1 for q in questions if q.shape == shape)
    if report.counts["exact_lookup"] == 0:
        report.skipped_shapes["exact_lookup"] = "corpus has no identifier-bearing chunks the model could turn into lookups"
    return questions, report


def _drop_common_identifiers(questions: list[Question], chunks: list[Chunk], max_fraction: float) -> list[Question]:
    """An exact-lookup question only works if its identifier is rare. Drop
    lookups whose identifier is very short or appears in more than
    ``max_fraction`` of chunks (``uv``, ``pyproject.toml``, ...): those are
    not lookups, they are noise with a single arbitrary gold chunk."""
    out: list[Question] = []
    n = max(len(chunks), 1)
    for q in questions:
        if q.shape == "exact_lookup":
            ident = q.identifier.strip("`").strip()
            if len(ident) < 3:
                continue
            df = sum(1 for c in chunks if ident in c.text) / n
            if df > max_fraction:
                continue
        out.append(q)
    return out


def _dedupe(questions: list[Question]) -> list[Question]:
    """Drop repeated question text, and repeated exact-lookup identifiers.

    Doc corpora explain the same flag or setting in several places. Two
    lookups on the same identifier from different chunks would each have a
    single gold chunk while being answerable from either, so keep only the
    first (chunk order is seeded, so this is deterministic)."""
    seen: set[str] = set()
    seen_ident: set[str] = set()
    out: list[Question] = []
    for q in questions:
        key = re.sub(r"[^a-z0-9 ]", "", q.q.lower()).strip()
        if key in seen:
            continue
        if q.shape == "exact_lookup":
            ident = q.identifier.lower()
            if ident in seen_ident:
                continue
            seen_ident.add(ident)
        seen.add(key)
        out.append(q)
    return out


# ------------------------------------------------------------------ persistence


def write_questions(questions: list[Question], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(json.dumps(q.to_dict(), ensure_ascii=False) + "\n")


def read_questions(path: Path) -> list[Question]:
    out: list[Question] = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out.append(
                Question(
                    id=str(rec.get("id", f"q{i:04d}")),
                    q=rec["q"],
                    shape=rec.get("shape", "conceptual"),
                    gold_chunk_ids=list(rec["gold_chunk_ids"]),
                    answer=rec.get("answer", ""),
                    source=rec.get("source", ""),
                    identifier=rec.get("identifier", ""),
                )
            )
    return out
