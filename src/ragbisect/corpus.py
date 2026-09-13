"""Corpus loading and the baseline chunker.

Two input forms are accepted:

* a directory of ``.md`` / ``.txt`` files, which ragbisect parses and chunks
  itself with a plain recursive splitter (this is the *baseline* being
  measured, not something we try to make good); or
* a ``.jsonl`` file of ``{"id": ..., "text": ..., "source": ...}`` records, if
  the user has already chunked and wants their own IDs to be the ground truth.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

TEXT_SUFFIXES = (".md", ".markdown", ".txt")

# Ordered from coarsest to finest. Headings are split *before* the marker so a
# heading stays attached to the section it introduces.
SEPARATORS = ("\n# ", "\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ")

MIN_CHUNK_CHARS = 40


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str

    def to_dict(self) -> dict:
        return asdict(self)


class CorpusError(Exception):
    pass


# --------------------------------------------------------------------------- #
# chunking
# --------------------------------------------------------------------------- #


def split_text(text: str, chunk_size: int, separators: tuple[str, ...] = SEPARATORS) -> list[str]:
    """Recursive character splitter. Deterministic, no overlap.

    Overlap is deliberately avoided: with overlap, "the chunk this question is
    answerable from" becomes ambiguous, which corrupts ground truth.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    if not separators:
        # Nothing left to split on: hard cut.
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, rest = separators[0], separators[1:]
    if sep not in text:
        return split_text(text, chunk_size, rest)

    parts = re.split(f"(?={re.escape(sep)})", text)
    out: list[str] = []
    buf = ""
    for part in parts:
        if not part:
            continue
        if len(buf) + len(part) <= chunk_size:
            buf += part
            continue
        if buf:
            out.append(buf)
            buf = ""
        if len(part) > chunk_size:
            out.extend(split_text(part, chunk_size, rest))
        else:
            buf = part
    if buf:
        out.append(buf)
    return out


def chunk_document(text: str, source: str, chunk_size: int) -> list[Chunk]:
    pieces = [p.strip() for p in split_text(text, chunk_size)]
    pieces = [p for p in pieces if len(p) >= MIN_CHUNK_CHARS]
    return [Chunk(id=f"{source}#{i}", text=p, source=source) for i, p in enumerate(pieces)]


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def iter_documents(root: Path) -> Iterable[tuple[str, str]]:
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES)
    for p in files:
        try:
            yield p.relative_to(root).as_posix(), p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue


def chunk_directory(root: Path, chunk_size: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for source, text in iter_documents(root):
        chunks.extend(chunk_document(text, source, chunk_size))
    return chunks


def read_chunks_jsonl(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise CorpusError(f"{path}:{lineno}: invalid JSON ({e})") from e
            if "id" not in rec or "text" not in rec:
                raise CorpusError(f"{path}:{lineno}: record needs 'id' and 'text'")
            cid = str(rec["id"])
            if cid in seen:
                raise CorpusError(f"{path}:{lineno}: duplicate chunk id {cid!r}")
            seen.add(cid)
            chunks.append(Chunk(id=cid, text=str(rec["text"]), source=str(rec.get("source", ""))))
    return chunks


def write_chunks_jsonl(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def load_corpus(path: str | Path, chunk_size: int = 800) -> list[Chunk]:
    p = Path(path)
    if p.is_file():
        if p.suffix.lower() == ".jsonl":
            chunks = read_chunks_jsonl(p)
        else:
            raise CorpusError(f"{p}: expected a directory or a .jsonl file of pre-chunked records")
    elif p.is_dir():
        chunks = chunk_directory(p, chunk_size)
    else:
        raise CorpusError(f"{p}: not found")
    if not chunks:
        raise CorpusError(f"{p}: no chunks produced (supported files: {', '.join(TEXT_SUFFIXES)})")
    return chunks


def corpus_fingerprint(chunks: list[Chunk]) -> str:
    """Stable hash of the chunk inventory. Used to key the eval-set cache so
    questions are only regenerated when the corpus (or chunking) changes."""
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.id.encode())
        h.update(b"\0")
        h.update(c.text.encode())
        h.update(b"\0")
    return h.hexdigest()[:16]
