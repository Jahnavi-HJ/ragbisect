import json
from pathlib import Path

from stagewise.corpus import Chunk
from stagewise.generate import Question, _clean_q, generate, has_identifier, read_questions, write_questions
from stagewise.llm import LLM


def test_identifier_detection():
    assert has_identifier("Run `uv sync` to install.")
    assert has_identifier("Set UV_CACHE_DIR to override.")
    assert has_identifier("Pass --no-cache to skip.")
    assert has_identifier("Error TX-4419 occurred.")
    assert not has_identifier("Our approach to refunds is generous and simple.")


def test_clean_q_rejects_context_references():
    assert _clean_q("What does this section say about caching?") is None
    assert _clean_q("short") is None
    assert _clean_q("  How does   uv pick a Python version?  ") == "How does uv pick a Python version?"


def test_generate_end_to_end_with_fake_llm(tmp_path: Path, monkeypatch):
    def fake_post(self, url, headers, body):
        prompt = body["messages"][-1]["content"]
        if "boilerplate" in prompt:
            content = {"usable": False}
        else:
            content = {
                "usable": True,
                "conceptual": {"q": "How does the cache directory get chosen?", "a": "via env var"},
                "exact_lookup": {"q": "What does UV_CACHE_DIR control?", "a": "cache location", "identifier": "UV_CACHE_DIR"},
            }
        return {"choices": [{"message": {"content": json.dumps(content)}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    monkeypatch.setattr(LLM, "_post", fake_post)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    llm = LLM(cache_path=tmp_path / "c.sqlite")
    chunks = [
        Chunk("a.md#0", "The UV_CACHE_DIR environment variable controls where the cache lives. " * 4, "a.md"),
        Chunk("a.md#1", "boilerplate navigation footer " * 10, "a.md"),
        Chunk("a.md#2", "too short", "a.md"),
    ]
    qs, report = generate(chunks, llm, n=10, seed=1, workers=2, common_identifier_max_fraction=1.0)
    assert {q.shape for q in qs} == {"conceptual", "exact_lookup"}
    assert all(q.gold_chunk_ids == ["a.md#0"] for q in qs)
    assert report.n_unusable_chunks == 1
    assert report.n_eligible_chunks == 2
    assert [q.id for q in qs] == ["q0001", "q0002"]

    p = tmp_path / "q.jsonl"
    write_questions(qs, p)
    assert read_questions(p) == qs


def test_dedupe_drops_repeated_identifiers_but_keeps_conceptual():
    from stagewise.generate import _dedupe

    qs = [
        Question("", "What does --foo do?", "exact_lookup", ["a"], identifier="--foo"),
        Question("", "What does the --foo flag control?", "exact_lookup", ["b"], identifier="--foo"),
        Question("", "What does --bar do?", "exact_lookup", ["c"], identifier="--bar"),
        Question("", "How does foo work?", "conceptual", ["a"]),
        Question("", "How does foo work?", "conceptual", ["b"]),
    ]
    out = _dedupe(qs)
    assert [q.gold_chunk_ids[0] for q in out] == ["a", "c", "a"]


def test_common_identifiers_are_dropped():
    from stagewise.generate import _drop_common_identifiers

    chunks = [Chunk(f"c{i}", "uv is great. " + ("UV_RARE_FLAG " if i == 0 else ""), "s") for i in range(20)]
    qs = [
        Question("", "What does uv do?", "exact_lookup", ["c0"], identifier="uv"),
        Question("", "What does UV_RARE_FLAG do?", "exact_lookup", ["c0"], identifier="UV_RARE_FLAG"),
        Question("", "How does uv work?", "conceptual", ["c0"]),
    ]
    out = _drop_common_identifiers(qs, chunks, max_fraction=0.05)
    assert [q.q for q in out] == ["What does UV_RARE_FLAG do?", "How does uv work?"]
