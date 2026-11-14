"""Pipeline entry point.

    python -m src.run ingest      # load corpus, chunk A/B, embed changed chunks, store
    python -m src.run evaluate    # (after the benchmark is frozen) run dev/test, save traces

Strategies (same corpus, embedder and store):
    A  fixed 500 chars, overlap 50           -> index "A", no filter
    B  paragraph/heading packing, max 1000   -> index "B", no filter
    C  same chunks/vectors as B              -> index "B", + metadata filter when the
                                                question supplies one (only difference B-C)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import OpenAIEmbedder
from .ingestion import DEFAULT_MANIFEST, load_corpus
from .store import PgVectorStore

STARTER = Path(__file__).resolve().parents[1]
RESULTS = STARTER / "results"

CHUNKERS = {
    "A": {"fn": lambda page: fixed_size(page, size=500, overlap=50),
          "params": {"chunking": "fixed_size", "size": 500, "overlap": 50}},
    "B": {"fn": lambda page: by_paragraph(page, max_size=1000, min_size=250),
          "params": {"chunking": "by_paragraph", "max_size": 1000, "min_size": 250}},
}
STRATEGIES = {
    "A": {"index": "A", "use_filter": False},
    "B": {"index": "B", "use_filter": False},
    "C": {"index": "B", "use_filter": True},
}


def _length_stats(lengths: list[int]) -> dict:
    ordered = sorted(lengths)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]
    return {"count": len(ordered), "total_chars": sum(ordered), "min": ordered[0],
            "p10": pick(0.1), "median": statistics.median(ordered), "p90": pick(0.9),
            "max": ordered[-1], "mean": round(statistics.mean(ordered), 1)}


def ingest() -> dict:
    t0 = time.perf_counter()
    pages, errors = load_corpus(DEFAULT_MANIFEST)
    if len(pages) < 30:
        raise SystemExit(f"only {len(pages)} content pages loaded; need >= 30")
    load_seconds = time.perf_counter() - t0

    embedder = OpenAIEmbedder()
    store = PgVectorStore(os.environ["DATABASE_URL"], embedder.dimensions)
    store.create_schema()

    report = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "pages": len(pages), "load_errors": errors,
              "load_seconds": round(load_seconds, 2),
              "embedding": {"provider": "openai", "model": embedder.model_name,
                            "dimensions": embedder.dimensions},
              "store": {"backend": "postgresql+pgvector", "table": "chunks",
                        "metric": "cosine", "search": "exact (sequential scan, no ANN index)"},
              "indexes": {}}

    for index, cfg in CHUNKERS.items():
        t1 = time.perf_counter()
        chunks = [c for page in pages for c in cfg["fn"](page)]
        stored = store.existing_hashes(index)
        changed = [c for c in chunks if stored.get(c.chunk_id) != c.metadata["content_hash"]]
        tokens_before = embedder.total_tokens
        vectors = embedder.embed_documents([c.text for c in changed]) if changed else []
        written = store.upsert(index, changed, vectors)["written"]
        deleted = store.delete_obsolete(index, sorted({c.document_id for c in chunks}),
                                        [c.chunk_id for c in chunks])
        rows = store.count(index)
        if rows != len(chunks):
            raise RuntimeError(f"index {index}: {rows} rows stored, {len(chunks)} chunks expected")
        report["indexes"][index] = {
            **cfg["params"], "chunks": _length_stats([len(c.text) for c in chunks]),
            "embedded_this_run": len(changed), "skipped_unchanged": len(chunks) - len(changed),
            "written": written, "deleted_obsolete": deleted, "rows_in_store": rows,
            "tokens_this_run": embedder.total_tokens - tokens_before,
            "seconds": round(time.perf_counter() - t1, 2)}
        print(f"[{index}] chunks={len(chunks)} embedded={len(changed)} "
              f"skipped={len(chunks) - len(changed)} deleted={deleted} rows={rows}")

    report["embedding_usage"] = embedder.usage()
    report["total_seconds"] = round(time.perf_counter() - t0, 2)
    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for path in (RESULTS / "ingest_report.json", RESULTS / "logs" / f"ingest_{stamp}.json"):
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"tokens={embedder.total_tokens} cost~${embedder.estimated_cost_usd:.6f} "
          f"time={report['total_seconds']}s -> results/ingest_report.json")
    return report


def _markdown_table(summary: dict, split: str) -> str:
    lines = [f"| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Gold page@3 | MRR@3 "
             f"| Filter Q Hit@3 | Filter compliance | Misses@3 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for strategy, by_split in summary.items():
        m = by_split.get(split)
        if m:
            lines.append(f"| {strategy} | {split} | {m['hit_at_1']} | {m['evidence_hit_at_3']} | "
                         f"{m['gold_page_at_3']} | {m['mrr_at_3']} | {m['filter_questions_hit_at_3']} "
                         f"| {m['filter_compliance']} | {', '.join(m['misses_at_3']) or '-'} |")
    return "\n".join(lines)


def evaluate(split: str, top_k: int = 3) -> dict:
    """Run every question of ``split`` through A/B/C and save full traces + metrics."""
    from .evaluation import evaluate_query, summarize

    bench = json.loads((STARTER.parent / "benchmark" / "questions.json").read_text(encoding="utf-8"))
    frozen = str(bench["meta"].get("status", "")).upper().startswith("FROZEN")
    if split in ("test", "all") and not frozen:
        raise SystemExit("Benchmark chưa FROZEN: không chạy test. Đối chiếu evidence, đặt "
                         "meta.status = 'FROZEN' và chọn chiến lược trên dev trước.")
    questions = [q for q in bench["questions"] if split == "all" or q["split"] == split]

    embedder = OpenAIEmbedder()
    store = PgVectorStore(os.environ["DATABASE_URL"], embedder.dimensions)
    traces, t0 = [], time.perf_counter()
    for q in questions:
        vector = embedder.embed_query(q["question"])
        for name, cfg in STRATEGIES.items():
            applied = q.get("filter") if cfg["use_filter"] and q.get("filter") else None
            t1 = time.perf_counter()
            hits = store.search(cfg["index"], vector, top_k=top_k, filters=applied)
            trace = evaluate_query(q, hits, applied)
            trace.update({"strategy": name, "index": cfg["index"],
                          "search_ms": round((time.perf_counter() - t1) * 1000, 1)})
            traces.append(trace)

    summary = summarize(traces)
    run_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta = {"run_at": run_at, "split": split, "top_k": top_k,
            "benchmark_version": bench["meta"].get("version"),
            "benchmark_status": bench["meta"].get("status"),
            "strategies": {k: {**v, **CHUNKERS[v["index"]]["params"]} for k, v in STRATEGIES.items()},
            "embedding_usage": embedder.usage(),
            "seconds": round(time.perf_counter() - t0, 2)}
    for strategy in summary:
        for s in summary[strategy]:
            summary[strategy][s]["mean_search_ms"] = round(statistics.mean(
                t["search_ms"] for t in traces if t["strategy"] == strategy and t["split"] == s), 1)

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    trace_path = RESULTS / f"traces_{split}.jsonl"
    with trace_path.open("w", encoding="utf-8") as fh:
        for t in traces:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    metrics = {"meta": meta, "summary": summary}
    for path in (RESULTS / f"metrics_{split}.json", RESULTS / "logs" / f"metrics_{split}_{stamp}.json"):
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    splits = ["dev", "test"] if split == "all" else [split]
    table = "\n\n".join(_markdown_table(summary, s) for s in splits)
    (RESULTS / f"metrics_{split}.md").write_text(
        f"# Retrieval metrics ({split})\n\nRun at {run_at} · top-{top_k} · benchmark "
        f"{meta['benchmark_version']} ({meta['benchmark_status'][:5]})\n\n{table}\n", encoding="utf-8")
    print(table)
    print(f"\ntraces -> {trace_path.relative_to(STARTER)} | tokens={embedder.total_tokens} "
          f"cost~${embedder.estimated_cost_usd:.6f}")
    return metrics


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    load_dotenv(STARTER / ".env")
    parser = argparse.ArgumentParser(description="Day 06 ingestion pipeline")
    parser.add_argument("command", choices=["ingest", "evaluate"])
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    if args.command == "ingest":
        ingest()
    else:
        evaluate(args.split, args.top_k)


if __name__ == "__main__":
    main()
