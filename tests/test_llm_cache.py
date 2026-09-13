import json
from pathlib import Path

import pytest

import stagewise.llm as llm_mod
from stagewise.llm import LLM, Usage, parse_model


def test_parse_model():
    assert parse_model("openai:gpt-5.4-mini") == ("openai", "gpt-5.4-mini")
    assert parse_model("claude-haiku-4-5-20251001") == ("anthropic", "claude-haiku-4-5-20251001")
    assert parse_model("gpt-4.1-mini") == ("openai", "gpt-4.1-mini")


def test_usage_cost_known_and_unknown():
    u = Usage()
    u.add("gpt-5-mini", 1_000_000, 0)
    assert u.cost_usd() == pytest.approx(0.25)
    u.add("mystery-model", 10, 10)
    assert u.cost_usd() is None
    assert "cost: unknown" in u.summary()


def test_complete_hits_cache_second_time(tmp_path: Path, monkeypatch):
    calls = []

    def fake_post(self, url, headers, body):
        calls.append(body)
        return {"choices": [{"message": {"content": json.dumps({"ok": True})}}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}

    monkeypatch.setattr(LLM, "_post", fake_post)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    llm = LLM(model="openai:gpt-test", cache_path=tmp_path / "c.sqlite")
    assert llm.complete_json("hi") == {"ok": True}
    assert llm.complete_json("hi") == {"ok": True}
    assert len(calls) == 1
    assert llm.usage.calls == 1 and llm.usage.cache_hits == 1
    assert llm.usage.prompt_tokens == 5

    # A fresh instance sharing the cache file also hits.
    llm2 = LLM(model="openai:gpt-test", cache_path=tmp_path / "c.sqlite")
    llm2.complete_json("hi")
    assert len(calls) == 1


def test_embed_caches_per_text(tmp_path: Path, monkeypatch):
    seen = []

    def fake_post(self, url, headers, body):
        seen.append(list(body["input"]))
        return {"data": [{"index": i, "embedding": [float(len(t)), 1.0]} for i, t in enumerate(body["input"])], "usage": {"prompt_tokens": 3}}

    monkeypatch.setattr(LLM, "_post", fake_post)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    llm = LLM(cache_path=tmp_path / "c.sqlite")
    assert llm.embed(["a", "bb"]) == [[1.0, 1.0], [2.0, 1.0]]
    assert llm.embed(["bb", "ccc"]) == [[2.0, 1.0], [3.0, 1.0]]
    assert seen == [["a", "bb"], ["ccc"]]
