# ragbisect — build spec (formerly stagewise; renamed 2026-09-13 to avoid the stagewise-io IDE)

Drop this in the repo root as `CLAUDE.md` (or keep it as `SPEC.md` and reference it). It is the source of truth for what we're building and, more importantly, what we're not.

---

## What this is

A diagnostic tool for RAG pipelines. The name is the thesis: other tools score the pipeline end to end, ragbisect scores it stage by stage so it can tell you which one is broken. Point it at a document corpus and an existing retrieval pipeline; it generates its own evaluation set, measures each stage of the pipeline independently, and reports which stage is losing the most quality.

It is a **measuring instrument**, not a RAG framework.

## The problem it solves

Teams build a RAG pipeline, quality is poor, and they have no way to tell *where* it's poor. When retrieval misses, nothing errors — the LLM writes a fluent answer from the wrong chunks. So teams guess: swap the embedding model, change chunk size, try another vector DB, blame the LLM. Months go by.

Existing eval tools (Ragas, DeepEval, promptfoo) score the **final answer**. That tells you the pipeline is bad, not which part. And they assume you already have a labelled test set, which is why most teams never evaluate at all.

## What makes it different

1. **The eval set builds itself** from the user's corpus. No hand-labelling.
2. **Each stage is scored in isolation** — retrieval, ranking, generation — so the report can localize the fault.
3. **The output is a recommended configuration**, not a number. It runs an ablation and says which change buys the most on *their* data.
4. **Framework-agnostic.** Works on whatever they already built. No rewrite, no migration.

## Non-goals

Be strict about these. Scope creep kills solo-maintained projects.

- Not a retrieval framework. We don't build chunkers, retrievers, or agents for production use.
- Not a serving layer. No API server, no daemon.
- Not a UI. v1 is a CLI that prints a table.
- Not an LLM-output quality tool. Generation scoring is the thinnest stage; retrieval is the focus.
- No plugin system, no config DSL, no extension registry in v1.

---

## The five query shapes

The eval generator must cover all of these. Most pipelines only handle the first one, and most eval sets only test the first one — which is precisely why teams don't know they're broken.

| Shape | Example | Why naive RAG fails |
|---|---|---|
| Conceptual | "What's our approach to refunds?" | Works. This is the only shape dense retrieval handles well. |
| Exact lookup | "What does error TX-4419 mean?" | Embeddings blur identifiers. Needs BM25. |
| Comparison | "What changed between the 2024 and 2025 policy?" | One blended query returns a mush of both docs. |
| Aggregation | "How many contracts expire in Q3?" | The answer is in no chunk. No value of k fixes it. |
| Table / structured | "Storage temp range for the R-40?" | Chunking cut the data row from its header. |

Generation strategy per shape:
- **Conceptual / exact lookup** — pick a chunk, generate a question answerable only from it, record its ID as ground truth.
- **Comparison** — find two chunks with high lexical overlap but differing values (likely near-duplicates across document versions); generate a question requiring both.
- **Aggregation** — extract structured values (dates, amounts, counts) across the corpus; compute the true answer programmatically; ground truth is the set of contributing chunks.
- **Table** — detect table structures at parse time; generate questions targeting a specific cell; ground truth is the row plus its header.

If a corpus can't support a shape (no tables, no versioned docs), skip it and say so in the report rather than generating junk questions.

---

## Architecture

```
corpus/ ──► parse ──► chunk inventory
                          │
                          ▼
                  eval set generator ──► questions.jsonl
                          │
                          ▼
        ┌─────────────────┴─────────────────┐
        │                                   │
   user pipeline                    candidate configs
   (their adapter)                  (hybrid, rerank, etc.)
        │                                   │
        └─────────────────┬─────────────────┘
                          ▼
                  stage scorers
              (recall, ranking, faithfulness)
                          │
                          ▼
                     report
```

### The adapter interface

This is the main design risk. If it's awkward, nobody integrates, and the tool dies. Keep it as small as physically possible.

```python
from typing import Protocol

class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[str]:
        """Return chunk IDs, most relevant first."""
```

That's the whole required surface for v1. A user wraps their existing pipeline in one function.

Optional second method, only if they want generation scored:

```python
    def generate(self, query: str, chunk_ids: list[str]) -> str: ...
```

**Resist adding to this.** No config objects, no lifecycle hooks, no async variants until someone actually asks.

The corpus side needs a mapping from chunk ID to text. Accept either a directory the tool parses itself, or a JSONL file of `{"id": ..., "text": ..., "source": ...}` if the user already chunked.

### Metrics per stage

- **Retrieval** — recall@k. Did the ground-truth chunk appear anywhere in the top k? This is the headline number.
- **Ranking** — NDCG@k, or MRR. Given that the chunk was retrieved, was it ranked near the top?
- **Generation** — faithfulness: is the answer supported by the retrieved chunks? Use an LLM judge. Lowest priority; do it last.

Fault localization rule: the stage with the largest gap between its score and the best achievable score for that stage is the bottleneck. Report the top one, not all four.

---

## Milestones

### M1 — eval set generator (the hard part, do it first)

- Parse a corpus directory (start with `.md` and `.txt`; add PDF later).
- Chunk it with a simple recursive splitter — this is the *baseline* we measure, not something we're trying to make good.
- Generate questions for conceptual and exact-lookup shapes via an LLM call, one per chunk, recording the source chunk ID.
- Output `questions.jsonl`: `{"q": ..., "shape": ..., "gold_chunk_ids": [...]}`

**Acceptance:** generate 200+ questions on a real corpus, hand-check 30. If the questions are bad, everything downstream is worthless. Do not proceed until this passes.

### M2 — retrieval scoring

- Take an adapter, run every question, compute recall@k.
- Compare two configs: dense-only vs dense+BM25 with reciprocal rank fusion.
- Print a two-row table.

**Acceptance:** show a measurable gap between the two configs on a real corpus. If there's no gap, either the corpus is unusually easy or the measurement is wrong — investigate before moving on.

### M3 — CLI

```
ragbisect run --corpus ./docs --adapter ./my_pipeline.py:MyRetriever
```

Prints the stage table and a one-line verdict. Nothing else.

### M4 — remaining query shapes

Comparison, aggregation, table. These need the structured extraction work described above, so they're meaningfully harder than M1's shapes.

### M5 — ablation and recommendation

Run candidate configs, report which change buys the most. This is the thing the tool is actually *for* — everything before it is plumbing.

### Later, only if M1–M5 land and people use it

Reranking as a tested config. Domain-tuned embeddings. Cached-full-context mode for small corpora (skip retrieval entirely — genuinely better when the corpus fits in a context window, and almost nobody does it). HTML report output.

---

## Engineering notes

- **Python.** That's where the ecosystem is.
- **Few dependencies.** Every dependency is a version conflict in someone's existing project, and this tool installs *alongside* their pipeline. Be ruthless.
- **LLM calls go through one provider-agnostic wrapper.** Users will have different providers and will not switch for us.
- **Cache aggressively.** Eval generation is expensive; regenerating on every run will stop people from running it.
- **Deterministic where possible.** Seed everything. A diagnostic tool that gives different answers on re-run is not a diagnostic tool.
- **Print costs.** Show the user what an eval run spent in tokens. Trust, and it stops surprise bills.

## Positioning

Differentiate on honesty, not feature count. Publish benchmarks including cases where the tool's own recommendations don't help. In a space this full of hype, that alone earns trust.

Before writing code: search GitHub and PyPI for the chosen name and for close prior art. This ecosystem moves fast and something similar may have shipped recently.
