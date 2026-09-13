# ragbisect

Bisect a RAG pipeline to find the broken stage. Stage-by-stage diagnostics for RAG pipelines, and the benchmark harness for [ContextPull](https://github.com/mi2arun/contextpull).

ragbisect (formerly stagewise) and ContextPull are one project in two packages, kept apart on purpose. ContextPull is the retrieval server; ragbisect is the neutral instrument that measures it against bm25, dense and hybrid pipelines on the same self-built eval set. ragbisect has no dependency on ContextPull and works on any pipeline you already have.

Other tools score your pipeline end to end and tell you it is bad. ragbisect
scores **retrieval**, **ranking** and **generation** separately, on an eval set
it **builds itself from your corpus**, and tells you **which stage is losing the
most quality**, and which built-in alternative would recover it.

It is a measuring instrument, not a RAG framework. Zero runtime dependencies.

## Install

```sh
uv add --dev ragbisect      # or: pip install ragbisect
export OPENAI_API_KEY=...   # used for question generation, judging and the built-in dense config
```

Any OpenAI-compatible endpoint works (`OPENAI_BASE_URL`). Anthropic models work
for generation and judging (`--model anthropic:claude-...`); embeddings still
need an OpenAI-compatible endpoint.

## Use

Wrap your existing pipeline in one method:

```python
class MyRetriever:
    def retrieve(self, query: str, k: int) -> list[str]:
        """Return chunk IDs, most relevant first."""
```

Then point ragbisect at your corpus and your adapter:

```sh
ragbisect run --corpus ./docs --adapter ./my_pipeline.py:MyRetriever
```

```
config                             recall@5   mrr@5  ndcg@5|hit    faith     n
---------------------------------------------------------------------------------
your adapter                          0.812    0.667        0.885      n/a   214
  conceptual                          0.842    0.702        0.899      n/a   171
  exact_lookup                        0.698    0.528        0.831      n/a    43
bm25 (built-in)                       0.771    0.611        0.862      n/a   214
dense (built-in)                      0.836    0.688        0.893      n/a   214
hybrid dense+bm25 rrf (built-in)      0.897    0.741        0.912      n/a   214

Bottleneck for 'your adapter': retrieval — recall@5 is 0.81; 19% of questions never see their gold chunk in the top 5. Weakest shape: exact_lookup (recall 0.70, n=43).
'hybrid dense+bm25 rrf (built-in)' would raise recall@5 from 0.81 to 0.90 (+0.09).
```

### What it does

1. **Parses and chunks** your corpus (`.md`, `.txt`), or takes your own chunks
   as a JSONL of `{"id", "text", "source"}` so your IDs are the ground truth.
2. **Generates an eval set**: for each sampled chunk, a *conceptual* question
   and, where the chunk explains an identifier (flag, error code, env var,
   version), an *exact-lookup* question. Ground truth is the chunk ID.
   Cached; re-runs are free until the corpus changes.
3. **Scores each stage**: recall@k (retrieval), NDCG@k conditioned on a hit
   (ranking), and faithfulness via an LLM judge if your adapter also has
   `generate(query, chunk_ids) -> str` (generation).
4. **Localizes the fault**: the stage furthest from its ceiling is the
   bottleneck. Built-in bm25 / dense / hybrid-RRF configs run on the same
   questions so you can see what a change would buy on *your* data.
5. **Prints what it spent** in tokens.

### Commands

```sh
ragbisect generate --corpus ./docs --n 200        # just build (and inspect) the eval set
ragbisect run --corpus ./docs                     # built-in configs only, no adapter
ragbisect run --corpus ./chunks.jsonl --adapter ./p.py:R --k 10 --dump-misses misses.jsonl
```

Everything lands in `.ragbisect/`: the cache, `chunks.jsonl`, and
`questions.jsonl`. Hand-check the questions. If they are bad, everything
downstream is worthless.

## Status

Milestones from `CLAUDE.md`:

- [x] M1 eval set generator (conceptual, exact lookup)
- [x] M2 retrieval scoring; dense vs dense+BM25 RRF comparison
- [x] M3 CLI with stage table and one-line verdict
- [ ] M4 comparison, aggregation and table query shapes
- [ ] M5 ablation across candidate configs with a recommendation

## Known limits (read before trusting a number)

- **Single gold chunk.** Doc corpora explain the same thing in several places.
  When the retriever returns a chunk that also answers the question but is not
  the one it was generated from, that counts as a miss. On the `uv` docs about
  half of the exact-lookup misses looked like this, so recall@k here is a
  **lower bound**. Repeated identifiers are deduped and very common ones
  (present in >5% of chunks) are rejected, which reduces but does not remove
  the effect. `--dump-misses` lets you see for yourself.
- **Two of five query shapes.** Comparison, aggregation and table questions
  are not generated yet, and those are the shapes naive RAG fails on hardest.
- **No PDF parsing.** `.md` and `.txt` only.
- **Generation scoring is thin** by design: one LLM judge on a sample.
- **Baseline chunker is the baseline.** If you already chunk, pass your own
  `.jsonl` so ground truth uses your IDs.
