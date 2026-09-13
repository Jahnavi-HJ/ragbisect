"""Built-in candidate retrievers used as reference configs.

These exist so ragbisect can *measure* alternatives against the user's
pipeline (dense-only vs dense+BM25 with reciprocal rank fusion). They are not
meant for production use.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from .corpus import Chunk
from .llm import LLM

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9._\-]*")


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens that keep identifiers intact (``--no-cache``,
    ``tx-4419``, ``3.12``, ``tool.uv.sources``)."""
    return [t.strip("._-") for t in _TOKEN_RE.findall(text.lower()) if t.strip("._-")]


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids = [c.id for c in chunks]
        self.tf: list[Counter[str]] = []
        self.doc_len: list[int] = []
        self.postings: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(chunks):
            toks = tokenize(c.text)
            tf = Counter(toks)
            self.tf.append(tf)
            self.doc_len.append(len(toks))
            for term in tf:
                self.postings[term].append(i)
        n = max(len(chunks), 1)
        self.avgdl = sum(self.doc_len) / n
        self.idf = {t: math.log((n - len(p) + 0.5) / (len(p) + 0.5) + 1.0) for t, p in self.postings.items()}

    def scores(self, query: str) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in self.postings[term]:
                f = self.tf[i][term]
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
                out[i] += idf * f * (self.k1 + 1) / denom
        return out

    def retrieve(self, query: str, k: int) -> list[str]:
        s = self.scores(query)
        ranked = sorted(s.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [self.ids[i] for i, _ in ranked]


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(math.fsum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class DenseRetriever:
    name = "dense"

    def __init__(self, chunks: list[Chunk], llm: LLM):
        self.llm = llm
        self.ids = [c.id for c in chunks]
        self.vectors = [_normalize(v) for v in llm.embed([c.text for c in chunks])]

    def retrieve(self, query: str, k: int) -> list[str]:
        q = _normalize(self.llm.embed([query])[0])
        sims = [(math.sumprod(q, v), i) for i, v in enumerate(self.vectors)]
        sims.sort(key=lambda t: (-t[0], t[1]))
        return [self.ids[i] for _, i in sims[:k]]


class HybridRRFRetriever:
    """Reciprocal rank fusion over several retrievers."""

    name = "hybrid (dense+bm25, rrf)"

    def __init__(self, retrievers: list, rrf_k: int = 60, depth_multiplier: int = 4):
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.depth_multiplier = depth_multiplier

    def retrieve(self, query: str, k: int) -> list[str]:
        depth = max(k * self.depth_multiplier, 20)
        fused: dict[str, float] = defaultdict(float)
        first_seen: dict[str, int] = {}
        for r in self.retrievers:
            for rank, cid in enumerate(r.retrieve(query, depth)):
                fused[cid] += 1.0 / (self.rrf_k + rank + 1)
                first_seen.setdefault(cid, len(first_seen))
        ranked = sorted(fused.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
        return [cid for cid, _ in ranked[:k]]
