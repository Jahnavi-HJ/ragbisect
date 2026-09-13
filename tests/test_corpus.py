from pathlib import Path

from stagewise.corpus import Chunk, chunk_document, corpus_fingerprint, load_corpus, read_chunks_jsonl, split_text


def test_split_respects_size_and_prefers_headings():
    text = "# Title\n\nIntro paragraph. " + ("word " * 100) + "\n\n## Section two\n\nMore text here. " + ("blah " * 100)
    pieces = split_text(text, 300)
    assert all(len(p) <= 300 for p in pieces)
    assert any(p.lstrip().startswith("## Section two") for p in pieces)
    assert "".join(pieces) == text  # lossless, no overlap


def test_split_is_deterministic_and_hard_cuts_when_needed():
    text = "x" * 1000
    assert split_text(text, 300) == [text[i : i + 300] for i in range(0, 1000, 300)]


def test_chunk_ids_are_stable(tmp_path: Path):
    (tmp_path / "a.md").write_text("# A\n\n" + "alpha " * 200 + "\n\n## B\n\n" + "beta " * 200)
    (tmp_path / "ignored.py").write_text("print('no')")
    c1 = load_corpus(tmp_path, chunk_size=500)
    c2 = load_corpus(tmp_path, chunk_size=500)
    assert [c.id for c in c1] == [c.id for c in c2]
    assert c1[0].id == "a.md#0"
    assert corpus_fingerprint(c1) == corpus_fingerprint(c2)
    assert all(c.source == "a.md" for c in c1)


def test_chunk_document_drops_tiny_pieces():
    chunks = chunk_document("# H\n\n" + "content " * 50, "doc.md", 200)
    assert all(len(c.text) >= 40 for c in chunks)


def test_read_prechunked_jsonl(tmp_path: Path):
    p = tmp_path / "chunks.jsonl"
    p.write_text('{"id": "c1", "text": "hello", "source": "s"}\n{"id": "c2", "text": "world"}\n')
    chunks = read_chunks_jsonl(p)
    assert chunks == [Chunk("c1", "hello", "s"), Chunk("c2", "world", "")]
