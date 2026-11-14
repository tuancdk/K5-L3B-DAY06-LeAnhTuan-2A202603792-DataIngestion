"""Aggregate the manually filled audit sheets into report numbers.

Reads whichever of results/audit/{coherence_labels,metadata_labels_test,
grounding_labels_test}.csv exist, rejects missing/invalid labels, and writes
results/audit/audit_summary.{json,md}. It never invents labels.
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.run import RESULTS  # noqa: E402

AUDIT = RESULTS / "audit"


def _read(name: str) -> list[dict] | None:
    path = AUDIT / name
    if not path.exists():
        return None
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _label(value: str, allowed: set[str], where: str, problems: list[str]) -> int | None:
    value = (value or "").strip()
    if value not in allowed:
        problems.append(f"{where}: label '{value}' not in {sorted(allowed)}")
        return None
    return int(value)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    summary, problems, md = {}, [], ["# Audit summary (nhãn thủ công)\n"]

    rows = _read("coherence_labels.csv")
    if rows is not None:
        by = defaultdict(list)
        for r in rows:
            lab = _label(r["label_0_2"], {"0", "1", "2"}, f"coherence {r['position_id']}/{r['strategy']}", problems)
            if lab is not None:
                by[r["strategy"]].append(lab)
        if "B" in by:
            by["C"] = list(by["B"])  # C shares B's chunks
        summary["coherence"] = {s: {"n": len(v), "mean_of_2": round(sum(v) / len(v), 2),
                                    "counts_0_1_2": [v.count(0), v.count(1), v.count(2)]}
                                for s, v in sorted(by.items()) if v}
        md.append("| Strategy | Coherence trung bình /2 (n) | Số nhãn 0/1/2 |\n|---|---|---|")
        md += [f"| {s} | {m['mean_of_2']} ({m['n']}) | {'/'.join(map(str, m['counts_0_1_2']))} |"
               for s, m in summary["coherence"].items()]
        md.append("\n_C dùng chung chunk với B nên nhãn coherence C = B._\n")

    rows = _read("metadata_labels_test.csv")
    if rows is not None:
        by = defaultdict(lambda: {"ok": 0, "n": 0, "errors": []})
        for r in rows:
            lab = _label(r["metadata_ok_1_0"], {"0", "1"}, f"metadata {r['question_id']}/{r['strategy']}#{r['rank']}", problems)
            if lab is None:
                continue
            b = by[r["strategy"]]
            b["n"] += 1
            b["ok"] += lab
            if lab == 0:
                b["errors"].append(f"{r['question_id']}#{r['rank']}: {r['error_note']}")
        summary["metadata"] = {s: {"ok": f"{v['ok']}/{v['n']}", "errors": v["errors"]}
                               for s, v in sorted(by.items())}
        md.append("| Strategy | Metadata đủ và đúng (n/N) |\n|---|---|")
        md += [f"| {s} | {m['ok']} |" for s, m in summary["metadata"].items()]
        md.append("")

    rows = _read("grounding_labels_test.csv")
    if rows is not None:
        by = defaultdict(list)
        for r in rows:
            lab = _label(r["label_0_2"], {"0", "1", "2"}, f"grounding {r['question_id']}/{r['strategy']}", problems)
            if lab is not None:
                by[r["strategy"]].append(lab)
        summary["grounding"] = {s: {"n": len(v), "counts_0_1_2": [v.count(0), v.count(1), v.count(2)],
                                    "mean_of_2": round(sum(v) / len(v), 2)}
                                for s, v in sorted(by.items()) if v}
        md.append("| Strategy | Grounding 0/1/2 (số câu) | Trung bình /2 |\n|---|---|---|")
        md += [f"| {s} | {'/'.join(map(str, m['counts_0_1_2']))} | {m['mean_of_2']} |"
               for s, m in summary["grounding"].items()]

    if not summary:
        sys.exit("No audit sheets found in results/audit/.")
    summary["problems"] = problems
    (AUDIT / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    (AUDIT / "audit_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    for p in problems:
        print(f"[problem] {p}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
