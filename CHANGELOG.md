# Changelog

## 0.2.0 — 2026-09-14

### Added
- Three structured question shapes: comparison (cross-document near-duplicate pairs, model-phrased, gold is both chunks), aggregation (identifier families counted programmatically) and table (one cell of a pipe table, only for row keys unique across tables). Aggregation and table need no model. `--shapes` selects.
- Cost columns: wall time per query for every config; model tokens and tool calls per query when an adapter exposes `stats()`.
- Adapters may declare `concurrency = N`; `--sample N` evaluates a seeded subset.
- Configs that produce no results are dropped from the table with a note instead of printing zero rows.

### Changed
- Renamed from stagewise to ragbisect. Working directory is `.ragbisect/`.

## 0.1.0 — 2026-09-14
First release: self-built eval set (conceptual, exact lookup), per-stage scoring, built-in bm25 / dense / hybrid configs, CLI.
