"""Stage scorers and fault localization.

* Retrieval: recall@k — did the gold chunk(s) appear anywhere in the top k?
* Ranking:   NDCG@k, conditioned on a hit — given the chunk was retrieved, how
             close to the top was it? Conditioning isolates ranking from recall.
* Generation: faithfulness via an LLM judge, only when the adapter exposes
             ``generate``. Thinnest stage by design.

Fault localization rule: the stage with the largest gap between its score and
the best achievable score (1.0) is the bottleneck. Report the top one.
"""

from __future__ import annotations

import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .corpus import Chunk
from .generate import Question
from .llm import LLM, LLMError


# ------------------------------------------------------------------ metrics


def recall_at_k(gold: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    if not gold:
        return 0.0
    top = set(retrieved[:k])
    return sum(1 for g in gold if g in top) / len(gold)


def mrr_at_k(gold: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    gold_set = set(gold)
    for i, cid in enumerate(retrieved[:k]):
        if cid in gold_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(gold: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    gold_set = set(gold)
    dcg = sum(1.0 / math.log2(i + 2) for i, cid in enumerate(retrieved[:k]) if cid in gold_set)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_set), k)))
    return dcg / ideal if ideal else 0.0


# ------------------------------------------------------------------ results


@dataclass
class QueryResult:
    qid: str
    shape: str
    gold: list[str]
    retrieved: list[str]
    k: int

    @property
    def recall(self) -> float:
        return recall_at_k(self.gold, self.retrieved, self.k)

    @property
    def mrr(self) -> float:
        return mrr_at_k(self.gold, self.retrieved, self.k)

    @property
    def ndcg(self) -> float:
        return ndcg_at_k(self.gold, self.retrieved, self.k)

    @property
    def hit(self) -> bool:
        return self.recall > 0


@dataclass
class StageScores:
    n: int
    recall: float
    mrr: float
    ndcg_given_hit: float | None  # None when nothing was retrieved at all
    faithfulness: float | None = None
    n_generated: int = 0


@dataclass
class ConfigResult:
    name: str
    k: int
    results: list[QueryResult]
    faithfulness: dict[str, bool] = field(default_factory=dict)  # qid -> supported?
    errors: int = 0

    def _agg(self, rs: list[QueryResult]) -> StageScores:
        if not rs:
            return StageScores(0, 0.0, 0.0, None)
        hits = [r for r in rs if r.hit]
        faith = [self.faithfulness[r.qid] for r in rs if r.qid in self.faithfulness]
        return StageScores(
            n=len(rs),
            recall=sum(r.recall for r in rs) / len(rs),
            mrr=sum(r.mrr for r in rs) / len(rs),
            ndcg_given_hit=(sum(r.ndcg for r in hits) / len(hits)) if hits else None,
            faithfulness=(sum(faith) / len(faith)) if faith else None,
            n_generated=len(faith),
        )

    @property
    def overall(self) -> StageScores:
        return self._agg(self.results)

    def by_shape(self) -> dict[str, StageScores]:
        groups: dict[str, list[QueryResult]] = defaultdict(list)
        for r in self.results:
            groups[r.shape].append(r)
        return {shape: self._agg(rs) for shape, rs in sorted(groups.items())}

    def misses(self) -> list[QueryResult]:
        return [r for r in self.results if not r.hit]


# ------------------------------------------------------------------ running


def run_config(
    name: str,
    retriever,
    questions: list[Question],
    k: int,
    workers: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> ConfigResult:
    """Run every question through a retriever. Sequential by default: user
    adapters are not guaranteed thread-safe."""
    out: list[QueryResult | None] = [None] * len(questions)
    errors = 0

    def one(i: int) -> None:
        q = questions[i]
        retrieved = [str(c) for c in retriever.retrieve(q.q, k)]
        out[i] = QueryResult(q.id, q.shape, list(q.gold_chunk_ids), retrieved, k)

    done = 0
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in pool.map(one, range(len(questions))):
                done += 1
                if progress:
                    progress(done, len(questions))
    else:
        for i in range(len(questions)):
            try:
                one(i)
            except Exception:
                errors += 1
            done += 1
            if progress:
                progress(done, len(questions))
    return ConfigResult(name, k, [r for r in out if r is not None], errors=errors)


JUDGE_SYSTEM = """You are a strict grader. You will see a question, some context passages, and an answer written from those passages.
Decide whether EVERY factual claim in the answer is directly supported by the passages. Ignore style. If the answer says it cannot answer, count that as supported.
Return JSON: {"supported": true|false, "unsupported_claim": "quote the first unsupported claim, or empty string"}"""


def score_generation(
    config: ConfigResult,
    adapter,
    questions: list[Question],
    chunk_text: dict[str, str],
    llm: LLM,
    sample: int = 50,
    workers: int = 4,
) -> None:
    """Fill ``config.faithfulness`` for up to ``sample`` questions, using
    ``adapter.generate`` and an LLM judge. Mutates ``config``."""
    by_id = {q.id: q for q in questions}
    subset = config.results[:sample]

    def one(r: QueryResult) -> tuple[str, bool | None]:
        q = by_id[r.qid]
        try:
            answer = adapter.generate(q.q, list(r.retrieved))
        except Exception:
            return r.qid, None
        context = "\n\n".join(f"[{cid}]\n{chunk_text.get(cid, '')}" for cid in r.retrieved)
        prompt = f"Question: {q.q}\n\nPassages:\n{context}\n\nAnswer:\n{answer}"
        try:
            verdict = llm.complete_json(prompt, system=JUDGE_SYSTEM)
        except LLMError:
            return r.qid, None
        return r.qid, bool(verdict.get("supported", False))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for qid, ok in pool.map(one, subset):
            if ok is None:
                config.errors += 1
            else:
                config.faithfulness[qid] = ok


# ------------------------------------------------------------ localization


@dataclass
class Verdict:
    stage: str
    gap: float
    detail: str


def localize(config: ConfigResult) -> Verdict:
    s = config.overall
    gaps: list[tuple[str, float, str]] = []
    miss_pct = 100 * (1 - s.recall)
    gaps.append(
        (
            "retrieval",
            1 - s.recall,
            f"recall@{config.k} is {s.recall:.2f}; {miss_pct:.0f}% of questions never see their gold chunk in the top {config.k}",
        )
    )
    if s.ndcg_given_hit is not None:
        gaps.append(
            (
                "ranking",
                1 - s.ndcg_given_hit,
                f"when the gold chunk is retrieved it ranks at NDCG {s.ndcg_given_hit:.2f}; MRR@{config.k} is {s.mrr:.2f}",
            )
        )
    if s.faithfulness is not None:
        gaps.append(
            (
                "generation",
                1 - s.faithfulness,
                f"faithfulness is {s.faithfulness:.2f} over {s.n_generated} judged answers",
            )
        )
    stage, gap, detail = max(gaps, key=lambda g: g[1])
    shapes = config.by_shape()
    if len(shapes) > 1:
        worst_shape, worst = min(shapes.items(), key=lambda kv: kv[1].recall)
        detail += f". Weakest shape: {worst_shape} (recall {worst.recall:.2f}, n={worst.n})"
    return Verdict(stage, gap, detail)
