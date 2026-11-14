"""Add AI-suggested labels to the audit sheets in SEPARATE columns.

Writes ``ai_suggestion`` / ``ai_reason`` next to the official label columns, which
are left untouched for the student. Coherence and grounding suggestions come from
results/audit/ai_suggestions.json (written after reading every chunk); metadata
suggestions are computed: the chunk text must occur on the claimed OCR page, the
section must match the manifest page range, and the source must be the gold document.
"""
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ingestion import DEFAULT_MANIFEST, load_corpus, section_lookup  # noqa: E402
from src.run import RESULTS  # noqa: E402

AUDIT = RESULTS / "audit"


def canon(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip().casefold()


def rewrite(name: str, fill) -> None:
    path = AUDIT / name
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    header = list(rows[0].keys())
    for col in ("ai_suggestion", "ai_reason"):
        if col not in header:
            header.append(col)
    for row in rows:
        row["ai_suggestion"], row["ai_reason"] = fill(row)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{name}: {len(rows)} rows annotated")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sugg = json.loads((AUDIT / "ai_suggestions.json").read_text(encoding="utf-8"))
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    source = manifest["sources"][0]
    sections = section_lookup(source["sections"])
    pages = {p.page: canon(p.text) for p in load_corpus(DEFAULT_MANIFEST)[0]}
    traces = {(t["id"], t["strategy"]): t for t in
              map(json.loads, (RESULTS / "traces_test.jsonl").open(encoding="utf-8"))}

    rewrite("coherence_labels.csv",
            lambda r: tuple(sugg["coherence"][f"{r['position_id']}|{r['strategy']}"]))
    rewrite("grounding_labels_test.csv",
            lambda r: tuple(sugg["grounding"][f"{r['question_id']}|{r['strategy']}"]))

    def metadata(r):
        hit = traces[(r["question_id"], r["strategy"])]["hits"][int(r["rank"]) - 1]
        page = int(r["page"])
        problems = []
        if r["source"] != source["source_id"]:
            problems.append("source khác tài liệu gold")
        if canon(hit["text"]) not in pages.get(page, ""):
            problems.append(f"text chunk không nằm trên trang {page}")
        if sections.get(page, {}).get("section") != r["section"]:
            problems.append(f"section không khớp dải trang manifest ({sections.get(page)})")
        if r["applied_filter"] and r["section"] != r["applied_filter"]:
            problems.append("vi phạm filter")
        if problems:
            return "0", "; ".join(problems)
        return "1", (f"text chunk khớp trang OCR {page}; section '{r['section']}' đúng dải trang "
                     "manifest; source đúng" + ("; tuân thủ filter" if r["applied_filter"] else ""))

    rewrite("metadata_labels_test.csv", metadata)


if __name__ == "__main__":
    main()
