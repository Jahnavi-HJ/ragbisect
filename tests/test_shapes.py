import json

from ragbisect.corpus import Chunk
from ragbisect.generate import generate
from ragbisect.llm import LLM
from ragbisect.shapes import comparison_pairs, find_tables, generate_aggregations, generate_table_questions, identifier_families

POLICY_2024 = "## Refund window\n\nCustomers may request a full refund within 30 days of delivery. After 30 days no refund is available."
POLICY_2025 = "## Refund window\n\nCustomers may request a full refund within 14 days of delivery. From day 15 to day 30 a store credit is issued instead. After 30 days no refund or credit is available."
ERRORS_A = "## TX-44xx: issuer declines\n\n| Code | Meaning | Retry |\n|---|---|---|\n| TX-4401 | Insufficient funds | yes, after 24h |\n| TX-4419 | Declined by issuer; retry is not permitted | no |\n| TX-4420 | Card reported lost or stolen | no |"
ERRORS_B = "## TX-45xx: gateway errors\n\n- TX-4501: gateway timeout. Retry with the same idempotency key.\n- TX-4502: gateway unavailable. Retry after the interval in the Retry-After header."
SPECS = "## Storage\n\n| Model | Storage temp | Voltage |\n|---|---|---|\n| R-10 | -10 to 40 °C | 110–240 V |\n| R-40 | -20 to 50 °C | 100–240 V |\n| R-90 | -35 to 65 °C | 100–240 V |"

CHUNKS = [
    Chunk("policy-2024.md#2", POLICY_2024, "policy-2024.md"),
    Chunk("policy-2025.md#2", POLICY_2025, "policy-2025.md"),
    Chunk("errors.md#1", ERRORS_A, "errors.md"),
    Chunk("errors.md#2", ERRORS_B, "errors.md"),
    Chunk("specs.md#1", SPECS, "specs.md"),
    Chunk("guide.md#0", "Welcome. This guide walks through installation and first use of the tool, step by step.", "guide.md"),
]


def test_comparison_pairs_find_cross_document_variants():
    pairs = comparison_pairs(CHUNKS, max_pairs=5, seed=0)
    assert [(a.id, b.id) for a, b, _ in pairs] == [("policy-2024.md#2", "policy-2025.md#2")]
    assert 0.45 <= pairs[0][2] <= 0.95


def test_identifier_families_and_aggregation_questions():
    noisy = CHUNKS + [Chunk("n.md#0", 'Set requires-python = ">=3.10" or requires-python = ">=3.12" or requires-python = ">=3.13".', "n.md")]
    fams = dict(identifier_families(noisy))
    assert not any("requires-python" in k for k in fams)  # expressions with spaces/quotes are not identifier families
    assert set(fams["TX-####"]) == {"TX-4401", "TX-4419", "TX-4420", "TX-4501", "TX-4502"}
    assert fams["TX-####"]["TX-4501"] == {"errors.md#2"}
    qs, why = generate_aggregations(CHUNKS, max_questions=4, seed=0)
    assert why is None
    tx = next(q for q in qs if "TX-NNNN" in q.q)
    assert sorted(tx.gold_chunk_ids) == ["errors.md#1", "errors.md#2"] and tx.answer.startswith("5:")
    assert tx.shape == "aggregation"


def test_table_questions_target_a_cell():
    spans = find_tables(CHUNKS)
    assert {s.chunk.id for s in spans} == {"errors.md#1", "specs.md#1"}
    qs, why = generate_table_questions(CHUNKS, max_questions=4, seed=1)
    assert why is None and qs
    spec = next(q for q in qs if q.source == "specs.md")
    assert spec.q.startswith("What is the ") and " for R-" in spec.q and spec.gold_chunk_ids == ["specs.md#1"]
    assert "`" not in spec.q and ("Storage temp" in spec.q or "Voltage" in spec.q)  # header case kept, markup stripped
    assert spec.answer in ("-10 to 40 °C", "-20 to 50 °C", "-35 to 65 °C", "110–240 V", "100–240 V")


def test_table_header_in_previous_chunk_becomes_second_gold():
    header = Chunk("t.md#0", "## Specs\n\n| Model | Temp |\n|---|---|\n| R-10 | 40 |", "t.md")
    rows = Chunk("t.md#1", "| R-40 | 50 |\n| R-90 | 65 |", "t.md")
    spans = find_tables([header, rows])
    cont = next(s for s in spans if s.chunk.id == "t.md#1")
    assert cont.header == ["Model", "Temp"] and cont.header_chunk_id == "t.md#0"
    qs, _ = generate_table_questions([header, rows], max_questions=4, seed=0)
    assert any(q.gold_chunk_ids == ["t.md#1", "t.md#0"] for q in qs)


def test_generate_produces_all_five_shapes_with_mock_model(tmp_path, monkeypatch):
    def fake_post(self, url, headers, body):
        prompt = body["messages"][-1]["content"]
        if prompt.startswith("Passage A"):
            content = {"usable": True, "q": "How did the refund window change between the 2024 and 2025 policy?", "a": "30 days -> 14 days plus store credit"}
        else:
            content = {"usable": True, "conceptual": {"q": "How does the refund process handle late requests?", "a": "no refund after 30 days"}, "exact_lookup": None}
        return {"choices": [{"message": {"content": json.dumps(content)}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    monkeypatch.setattr(LLM, "_post", fake_post)
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    llm = LLM(cache_path=tmp_path / "c.sqlite")
    big = [Chunk(c.id, c.text + " " + "Additional explanatory text to pass the minimum length filter. " * 3, c.source) for c in CHUNKS]
    qs, report = generate(big, llm, n=8, seed=0, workers=2)
    shapes = {q.shape for q in qs}
    assert {"conceptual", "comparison", "aggregation", "table"} <= shapes
    comp = next(q for q in qs if q.shape == "comparison")
    assert sorted(comp.gold_chunk_ids) == ["policy-2024.md#2", "policy-2025.md#2"]
    assert report.counts["comparison"] == 1 and report.counts["aggregation"] >= 1 and report.counts["table"] >= 1
    assert "comparison" not in report.skipped_shapes
    # shapes selection
    qs2, report2 = generate(big, llm, n=8, seed=0, workers=2, shapes=("table",))
    assert {q.shape for q in qs2} == {"table"} and report2.skipped_shapes["conceptual"] == "not requested"


def test_table_questions_skip_keys_repeated_across_tables():
    a = Chunk("t.md#0", "## Dates\n\n| Constraint | JSON Schema |\n|---|---|\n| gt | exclusiveMinimum |\n| lt | exclusiveMaximum |\n| unique_a | onlyhere |", "t.md")
    b = Chunk("t.md#5", "## Times\n\n| Constraint | JSON Schema |\n|---|---|\n| gt | exclusiveMinimum |\n| lt | exclusiveMaximum |", "t.md")
    qs, why = generate_table_questions([a, b], max_questions=6, seed=0)
    assert why is None and [q.q for q in qs] == ["What is the JSON Schema for unique_a?"]
    qs, why = generate_table_questions([a, Chunk("u.md#0", b.text.replace("Times", "Other"), "u.md")], max_questions=6, seed=0)
    assert [q.answer for q in qs] == ["onlyhere"]
    only_dupes = [Chunk("x.md#0", b.text, "x.md"), Chunk("y.md#0", b.text, "y.md")]
    assert generate_table_questions(only_dupes, 4, 0)[1].startswith("tables found, but every row key repeats")


def test_table_keys_drop_markdown_link_syntax():
    c = Chunk("e.md#0", "## Errors\n\n| Field | Description |\n|---|---|\n| [loc][pydantic_core.ErrorDetails.loc] | The error location |\n| [msg](https://x) | The message |\n| `type` | The type |", "e.md")
    qs, _ = generate_table_questions([c], max_questions=6, seed=0)
    assert {q.q for q in qs} <= {"What is the Description for loc?", "What is the Description for msg?", "What is the Description for type?"}
