"""Plain-text report: a stage table and a one-line verdict. Nothing else."""

from __future__ import annotations

from .generate import GenerationReport
from .score import ConfigResult, StageScores, localize


def _fmt(x: float | None) -> str:
    return "  —  " if x is None else f"{x:.3f}"


def _row(label: str, s: StageScores, k: int, indent: int = 0) -> str:
    faith = _fmt(s.faithfulness) if s.faithfulness is not None else "  n/a"
    return f"{' ' * indent}{label:<34.34}{_fmt(s.recall):>10}{_fmt(s.mrr):>9}{_fmt(s.ndcg_given_hit):>13}{faith:>9}{s.n:>6}"


def stage_table(configs: list[ConfigResult], k: int, show_shapes: bool = True) -> str:
    lines = [f"{'config':<34}{f'recall@{k}':>10}{f'mrr@{k}':>9}{f'ndcg@{k}|hit':>13}{'faith':>9}{'n':>6}"]
    lines.append("-" * len(lines[0]))
    for c in configs:
        lines.append(_row(c.name, c.overall, k))
        if show_shapes:
            for shape, s in c.by_shape().items():
                lines.append(_row(shape, s, k, indent=2))
    return "\n".join(lines)


def verdict_line(config: ConfigResult) -> str:
    v = localize(config)
    return f"Bottleneck for '{config.name}': {v.stage} — {v.detail}."


def comparison_line(baseline: ConfigResult, others: list[ConfigResult]) -> str | None:
    """If a candidate config beats the primary one on recall, say by how much."""
    b = baseline.overall.recall
    best = max(others, key=lambda c: c.overall.recall, default=None)
    if best is None:
        return None
    delta = best.overall.recall - b
    if delta > 0.005:
        return f"'{best.name}' would raise recall@{baseline.k} from {b:.2f} to {best.overall.recall:.2f} (+{delta:.2f})."
    return f"No built-in config beats '{baseline.name}' on recall@{baseline.k}."


def generation_notes(report: GenerationReport) -> str:
    counts = ", ".join(f"{k}={v}" for k, v in report.counts.items())
    lines = [
        f"eval set: {sum(report.counts.values())} questions ({counts}) from {report.n_chunks_used} chunks "
        f"({report.n_unusable_chunks} judged unusable, {report.errors} errors); corpus {report.n_chunks} chunks, "
        f"fingerprint {report.corpus_fingerprint}, model {report.model}"
    ]
    for shape, why in report.skipped_shapes.items():
        lines.append(f"  skipped shape '{shape}': {why}")
    if report.errors and report.first_error:
        lines.append(f"  first error: {report.first_error}")
    return "\n".join(lines)
