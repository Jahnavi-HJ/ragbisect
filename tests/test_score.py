import math

from ragbisect.score import ConfigResult, QueryResult, localize, mrr_at_k, ndcg_at_k, recall_at_k


def test_recall_binary_single_gold():
    assert recall_at_k(["g"], ["a", "g", "b"], 5) == 1.0
    assert recall_at_k(["g"], ["a", "b"], 5) == 0.0
    assert recall_at_k(["g"], ["a", "b", "c", "d", "e", "g"], 5) == 0.0  # outside k


def test_recall_fraction_multi_gold():
    assert recall_at_k(["g1", "g2"], ["g1", "x"], 5) == 0.5


def test_mrr():
    assert mrr_at_k(["g"], ["g"], 5) == 1.0
    assert mrr_at_k(["g"], ["a", "b", "g"], 5) == 1 / 3
    assert mrr_at_k(["g"], [], 5) == 0.0


def test_ndcg():
    assert ndcg_at_k(["g"], ["g", "a"], 5) == 1.0
    assert math.isclose(ndcg_at_k(["g"], ["a", "g"], 5), 1 / math.log2(3))
    assert ndcg_at_k(["g1", "g2"], ["g1", "g2"], 5) == 1.0


def _cfg(results):
    return ConfigResult("t", 5, results)


def test_localize_blames_retrieval_when_recall_is_low():
    rs = [QueryResult(f"q{i}", "conceptual", ["g"], ["x", "y"], 5) for i in range(8)]
    rs += [QueryResult("h1", "conceptual", ["g"], ["g"], 5), QueryResult("h2", "exact_lookup", ["g"], ["g"], 5)]
    v = localize(_cfg(rs))
    assert v.stage == "retrieval"
    assert "exact_lookup" in v.detail or "conceptual" in v.detail


def test_localize_blames_ranking_when_hits_are_buried():
    rs = [QueryResult(f"q{i}", "conceptual", ["g"], ["a", "b", "c", "d", "g"], 5) for i in range(10)]
    c = _cfg(rs)
    assert c.overall.recall == 1.0
    assert localize(c).stage == "ranking"


def test_by_shape_and_conditional_ndcg():
    rs = [
        QueryResult("a", "conceptual", ["g"], ["g"], 5),
        QueryResult("b", "conceptual", ["g"], ["x"], 5),
        QueryResult("c", "exact_lookup", ["g"], ["x", "g"], 5),
    ]
    c = _cfg(rs)
    shapes = c.by_shape()
    assert shapes["conceptual"].recall == 0.5
    assert shapes["conceptual"].ndcg_given_hit == 1.0  # the miss is excluded from ranking
    assert math.isclose(shapes["exact_lookup"].ndcg_given_hit, 1 / math.log2(3))


def test_run_config_records_ms_stats_and_concurrency():
    from ragbisect.generate import Question
    from ragbisect.report import stage_table
    from ragbisect.score import run_config

    class Agent:
        concurrency = 4

        def __init__(self):
            self.calls = 0

        def retrieve(self, q, k):
            self.calls += 1
            return ["g"]

        def stats(self):
            return {"tokens_in": 1000, "tokens_out": 100, "tool_calls": 30, "queries": self.calls}

    qs = [Question(f"q{i}", "what?", "conceptual", ["g"]) for i in range(10)]
    c = run_config("agent", Agent(), qs, 5)
    assert c.overall.recall == 1.0 and c.overall.ms_per_query >= 0
    assert c.tokens_per_query() == 110.0 and c.calls_per_query() == 3.0
    table = stage_table([c], 5, show_shapes=False)
    assert "tok/q" in table.splitlines()[0] and "110" in table.splitlines()[2] and "3.0" in table.splitlines()[2]
