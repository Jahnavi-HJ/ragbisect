from stagewise.corpus import Chunk
from stagewise.retrievers import BM25Retriever, HybridRRFRetriever, tokenize


def test_tokenize_keeps_identifiers():
    toks = tokenize("Set UV_CACHE_DIR or pass --no-cache; error TX-4419 on 3.12.")
    assert "uv_cache_dir" in toks
    assert "no-cache" in toks
    assert "tx-4419" in toks
    assert "3.12" in toks


def test_bm25_finds_exact_identifier():
    chunks = [
        Chunk("a", "Refund policy: customers may request a refund within 30 days.", "p"),
        Chunk("b", "Error TX-4419 means the transaction was declined by the issuer.", "e"),
        Chunk("c", "Shipping takes three to five business days.", "s"),
    ]
    r = BM25Retriever(chunks)
    assert r.retrieve("What does error TX-4419 mean?", 2)[0] == "b"
    assert r.retrieve("refund approach", 1) == ["a"]
    assert r.retrieve("zzz unknown", 3) == []


class _Fixed:
    def __init__(self, ranking):
        self.ranking = ranking

    def retrieve(self, query, k):
        return self.ranking[:k]


def test_rrf_fuses_rankings():
    a = _Fixed(["x", "y", "z"])
    b = _Fixed(["y", "x", "w"])
    fused = HybridRRFRetriever([a, b]).retrieve("q", 3)
    assert set(fused[:2]) == {"x", "y"}
    assert fused[2] in {"z", "w"}


def test_rrf_is_deterministic_on_ties():
    a = _Fixed(["x", "y"])
    b = _Fixed(["y", "x"])
    assert HybridRRFRetriever([a, b]).retrieve("q", 2) == HybridRRFRetriever([a, b]).retrieve("q", 2)
