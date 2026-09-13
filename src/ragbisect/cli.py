"""``ragbisect`` command line.

    ragbisect generate --corpus ./docs [--n 200]
    ragbisect run --corpus ./docs --adapter ./my_pipeline.py:MyRetriever [--k 5]

``run`` prints the stage table and a one-line verdict. Nothing else.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import sys
import time
from pathlib import Path

from . import __version__
from .corpus import Chunk, CorpusError, corpus_fingerprint, load_corpus, write_chunks_jsonl
from .generate import GenerationReport, generate, read_questions, write_questions
from .llm import DEFAULT_EMBED_MODEL, DEFAULT_MODEL, LLM, LLMError
from .report import comparison_line, generation_notes, stage_table, verdict_line
from .retrievers import BM25Retriever, DenseRetriever, HybridRRFRetriever
from .score import ConfigResult, run_config, score_generation


def _eprint(*a: object, **kw) -> None:
    print(*a, file=sys.stderr, flush=True, **kw)


class _Progress:
    def __init__(self, label: str):
        self.label = label
        self.t0 = time.time()
        self.last = 0.0

    def __call__(self, done: int, total: int) -> None:
        now = time.time()
        if done == total or now - self.last > 1.0:
            self.last = now
            _eprint(f"\r  {self.label}: {done}/{total} ({now - self.t0:.0f}s)", end="" if done < total else "\n")


# ------------------------------------------------------------------ helpers


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--corpus", required=True, help="directory of .md/.txt files, or a .jsonl of pre-chunked {id,text,source}")
    p.add_argument("--workdir", default=".ragbisect", help="where cache, chunks and questions live (default: .ragbisect)")
    p.add_argument("--chunk-size", type=int, default=800, help="baseline chunker size in characters (default: 800)")
    p.add_argument("--n", type=int, default=200, help="target number of questions (default: 200)")
    p.add_argument("--seed", type=int, default=0, help="sampling seed (default: 0)")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"provider:model for generation and judging (default: {DEFAULT_MODEL})")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help=f"embedding model for the built-in dense config (default: {DEFAULT_EMBED_MODEL})")
    p.add_argument("--workers", type=int, default=8, help="parallel LLM calls during generation (default: 8)")


def _load_chunks(args: argparse.Namespace) -> list[Chunk]:
    chunks = load_corpus(args.corpus, chunk_size=args.chunk_size)
    workdir = Path(args.workdir)
    write_chunks_jsonl(chunks, workdir / "chunks.jsonl")
    return chunks


def _questions_path(args: argparse.Namespace, fingerprint: str) -> Path:
    return Path(args.workdir) / f"questions-{fingerprint}-n{args.n}-s{args.seed}.jsonl"


def _ensure_questions(args: argparse.Namespace, chunks: list[Chunk], llm: LLM):
    fp = corpus_fingerprint(chunks)
    qpath = _questions_path(args, fp)
    mpath = qpath.with_suffix(".meta.json")
    if getattr(args, "questions", None):
        qpath = Path(args.questions)
        mpath = qpath.with_suffix(".meta.json")
    if qpath.exists() and not getattr(args, "regenerate", False):
        questions = read_questions(qpath)
        report = None
        if mpath.exists():
            report = GenerationReport(**json.loads(mpath.read_text()))
        _eprint(f"using cached eval set {qpath} ({len(questions)} questions)")
        return questions, report, qpath
    _eprint(f"generating eval set from {len(chunks)} chunks (target {args.n} questions)")
    questions, report = generate(chunks, llm, n=args.n, seed=args.seed, workers=args.workers, progress=_Progress("generate"))
    if not questions:
        _eprint(generation_notes(report))
        raise LLMError("no questions could be generated; not caching an empty eval set")
    write_questions(questions, qpath)
    mpath.write_text(json.dumps(report.to_dict(), indent=2))
    # Stable alias for humans.
    latest = Path(args.workdir) / "questions.jsonl"
    write_questions(questions, latest)
    return questions, report, qpath


def load_adapter(spec: str, chunks: list[Chunk]):
    """``path/to/file.py:Name`` or ``package.module:Name``.

    ``Name`` may be an instance or a class. A class is instantiated with the
    chunk list if its constructor takes a required positional argument,
    otherwise with no arguments.
    """
    if ":" not in spec:
        raise SystemExit(f"--adapter must look like path/to/file.py:ClassName (got {spec!r})")
    target, name = spec.rsplit(":", 1)
    if target.endswith(".py") or "/" in target:
        path = Path(target)
        if not path.exists():
            raise SystemExit(f"adapter file not found: {path}")
        modspec = importlib.util.spec_from_file_location(f"_ragbisect_adapter_{path.stem}", path)
        assert modspec and modspec.loader
        module = importlib.util.module_from_spec(modspec)
        sys.path.insert(0, str(path.parent.resolve()))
        modspec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    try:
        obj = getattr(module, name)
    except AttributeError:
        raise SystemExit(f"{target} has no attribute {name!r}")
    if inspect.isclass(obj):
        params = [
            p
            for p in inspect.signature(obj).parameters.values()
            if p.default is inspect.Parameter.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        obj = obj(chunks) if params else obj()
    if not hasattr(obj, "retrieve"):
        raise SystemExit(f"adapter {spec} has no retrieve(query, k) method")
    return obj


# ------------------------------------------------------------------ commands


def cmd_generate(args: argparse.Namespace) -> int:
    llm = LLM(model=args.model, embed_model=args.embed_model, cache_path=Path(args.workdir) / "cache.sqlite")
    chunks = _load_chunks(args)
    questions, report, qpath = _ensure_questions(args, chunks, llm)
    if report:
        print(generation_notes(report))
    print(f"wrote {len(questions)} questions to {qpath}")
    print(f"spend: {llm.usage.summary()}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    llm = LLM(model=args.model, embed_model=args.embed_model, cache_path=Path(args.workdir) / "cache.sqlite")
    chunks = _load_chunks(args)
    chunk_text = {c.id: c.text for c in chunks}
    questions, report, _ = _ensure_questions(args, chunks, llm)
    if not questions:
        _eprint("no questions generated; nothing to score")
        return 2
    if args.sample and args.sample < len(questions):
        import random

        rng = random.Random(args.seed)
        questions = sorted(rng.sample(questions, args.sample), key=lambda q: q.id)
        _eprint(f"evaluating a seeded sample of {len(questions)} questions")

    known = set(chunk_text)
    unknown_gold = sum(1 for q in questions for g in q.gold_chunk_ids if g not in known)
    if unknown_gold:
        _eprint(f"warning: {unknown_gold} gold chunk ids are not in the corpus (stale eval set?)")

    configs: list[ConfigResult] = []
    primary: ConfigResult | None = None
    adapter = None
    if args.adapter:
        adapter = load_adapter(args.adapter, chunks)
        primary = run_config("your adapter", adapter, questions, args.k, progress=_Progress("adapter"))
        if primary.errors:
            _eprint(f"warning: adapter raised on {primary.errors} queries")
        configs.append(primary)

    if not args.no_baselines:
        _eprint("indexing built-in configs")
        bm25 = BM25Retriever(chunks)
        try:
            dense = DenseRetriever(chunks, llm)
        except LLMError as e:
            _eprint(f"dense config skipped: {e}")
            dense = None
        configs.append(run_config("bm25 (built-in)", bm25, questions, args.k, workers=4))
        if dense is not None:
            configs.append(run_config("dense (built-in)", dense, questions, args.k, workers=4, progress=_Progress("dense")))
            configs.append(
                run_config("hybrid dense+bm25 rrf (built-in)", HybridRRFRetriever([dense, bm25]), questions, args.k, workers=4, progress=_Progress("hybrid"))
            )

    if primary is not None and adapter is not None and hasattr(adapter, "generate") and args.gen_sample > 0:
        _eprint(f"judging generation on {min(args.gen_sample, len(primary.results))} answers")
        score_generation(primary, adapter, questions, chunk_text, llm, sample=args.gen_sample)

    if not configs:
        _eprint("nothing to run: pass --adapter or drop --no-baselines")
        return 2

    print()
    print(stage_table(configs, args.k, show_shapes=not args.no_shapes))
    print()
    focus = primary or configs[0]
    print(verdict_line(focus))
    if primary is not None and len(configs) > 1:
        line = comparison_line(primary, configs[1:])
        if line:
            print(line)
    if primary is not None and primary.stats:
        st = primary.stats
        extra = f"; ≈ ${st['usd']:.2f}" if st.get("usd") else ""
        print(f"adapter spend: {st.get('tokens_in', 0):,} in / {st.get('tokens_out', 0):,} out tokens, {st.get('tool_calls', 0)} tool calls over {st.get('queries', 0)} queries{extra}")
    if report:
        print()
        print(generation_notes(report))
    print(f"spend: {llm.usage.summary()}")

    if args.dump_misses:
        _dump_misses(focus, questions, chunk_text, Path(args.dump_misses))
        _eprint(f"wrote misses to {args.dump_misses}")
    return 0


def _dump_misses(config: ConfigResult, questions, chunk_text: dict[str, str], path: Path) -> None:
    by_id = {q.id: q for q in questions}
    with path.open("w", encoding="utf-8") as fh:
        for r in config.misses():
            q = by_id[r.qid]
            fh.write(
                json.dumps(
                    {
                        "id": q.id,
                        "q": q.q,
                        "shape": q.shape,
                        "gold": q.gold_chunk_ids,
                        "gold_text": [chunk_text.get(g, "")[:400] for g in q.gold_chunk_ids],
                        "retrieved": r.retrieved,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


# ------------------------------------------------------------------ main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ragbisect", description="Stage-by-stage diagnostics for RAG pipelines.")
    p.add_argument("--version", action="version", version=f"ragbisect {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="build the eval set from the corpus and stop")
    _add_common(g)
    g.add_argument("--regenerate", action="store_true", help="ignore any cached eval set for this corpus")
    g.set_defaults(func=cmd_generate)

    r = sub.add_parser("run", help="score a pipeline stage by stage")
    _add_common(r)
    r.add_argument("--adapter", help="path/to/file.py:ClassName implementing retrieve(query, k) -> list[str]")
    r.add_argument("--k", type=int, default=5, help="cutoff for recall@k / ndcg@k (default: 5)")
    r.add_argument("--questions", help="use this questions.jsonl instead of the auto-generated set")
    r.add_argument("--regenerate", action="store_true", help="ignore any cached eval set for this corpus")
    r.add_argument("--no-baselines", action="store_true", help="skip the built-in bm25/dense/hybrid configs")
    r.add_argument("--no-shapes", action="store_true", help="hide the per-shape breakdown rows")
    r.add_argument("--gen-sample", type=int, default=50, help="answers to judge for faithfulness if the adapter has generate() (default: 50; 0 disables)")
    r.add_argument("--dump-misses", metavar="FILE", help="write questions whose gold chunk was not retrieved, as JSONL")
    r.add_argument("--sample", type=int, default=0, help="evaluate only a seeded random sample of N questions (for expensive adapters)")
    r.set_defaults(func=cmd_run)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (CorpusError, LLMError) as e:
        _eprint(f"error: {e}")
        return 1
    except KeyboardInterrupt:
        _eprint("\ninterrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
