"""Create blank manual-audit sheets (the labels themselves are filled in by the student).

    python scripts/make_audit.py coherence   # 10 fixed source positions x strategies A, B
    python scripts/make_audit.py test        # metadata + grounding sheets from traces_test.jsonl

Coherence positions are fixed by a rule decided before looking at any score: 10 content
pages evenly spaced over the sorted page list; on each page the anchor is the line that
contains the middle character. C reuses B's chunks, so its coherence label equals B's.
CSV files are UTF-8 with BOM so Excel shows Vietnamese correctly.
"""
import csv
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ingestion import DEFAULT_MANIFEST, load_corpus  # noqa: E402
from src.run import CHUNKERS, RESULTS  # noqa: E402

AUDIT = RESULTS / "audit"


def _write(path: Path, header: list[str], rows: list[list]) -> None:
    if path.exists():
        sys.exit(f"{path.name} already exists; refusing to overwrite labels. Delete it first.")
    AUDIT.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"wrote {path.relative_to(RESULTS.parent)} ({len(rows)} rows)")


def coherence() -> None:
    pages = sorted(load_corpus(DEFAULT_MANIFEST)[0], key=lambda p: p.page)
    picks = [pages[round(i * (len(pages) - 1) / 9)] for i in range(10)]
    positions, rows = [], []
    for n, page in enumerate(picks, start=1):
        mid = len(page.text) // 2
        line_start = page.text.rfind("\n", 0, mid) + 1
        line_end = page.text.find("\n", mid)
        line_end = len(page.text) if line_end == -1 else line_end
        anchor = page.text[line_start:line_end].strip()
        positions.append({"position_id": f"P{n:02d}", "page": page.page,
                          "section": page.metadata.get("section"), "char_offset": mid,
                          "anchor_line": anchor})
        for strategy in ("A", "B"):
            chunks = CHUNKERS[strategy]["fn"](page)
            chunk = next(c for c in chunks if c.metadata["char_start"] <= mid < c.metadata["char_end"])
            rows.append([f"P{n:02d}", page.page, page.metadata.get("section"), anchor, strategy,
                         chunk.chunk_id, len(chunk.text), chunk.text, "", ""])
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / "coherence_positions.json").write_text(json.dumps({
        "created_at": date.today().isoformat(),
        "rule": "10 trang nội dung cách đều trong danh sách trang đã sắp xếp; vị trí = dòng chứa "
                "ký tự giữa trang; chunk = chunk đầu tiên của chiến lược chứa vị trí đó. "
                "C dùng chung chunk với B nên nhãn C = nhãn B.",
        "scale": "0 = bị cắt giữa ý/thiếu chủ thể, không hiểu; 1 = hiểu một phần, thiếu ngữ cảnh; "
                 "2 = tự hiểu được và giữ trọn ý",
        "positions": positions}, ensure_ascii=False, indent=2), encoding="utf-8")
    _write(AUDIT / "coherence_labels.csv",
           ["position_id", "page", "section", "anchor_line", "strategy", "chunk_id",
            "chunk_chars", "chunk_text", "label_0_2", "reason"], rows)


def test_sheets() -> None:
    path = RESULTS / "traces_test.jsonl"
    if not path.exists():
        sys.exit("results/traces_test.jsonl not found: freeze the benchmark and run "
                 "`python -m src.run evaluate --split test` first.")
    traces = [json.loads(line) for line in path.open(encoding="utf-8")]
    meta_rows, ground_rows = [], []
    for t in traces:
        filt = (t["applied_filter"] or {}).get("section", "")
        for h in t["hits"][:3]:
            meta_rows.append([t["id"], t["strategy"], h["rank"], h["chunk_id"],
                              t["gold_source_id"], h["page"], h["section"], filt,
                              h["text"][:300], "", "", "", "", ""])
        top3 = "\n\n".join(f"[#{h['rank']} tr.{h['page']} {h['section']}]\n{h['text']}"
                           for h in t["hits"][:3])
        ground_rows.append([t["id"], t["strategy"], t["question"], t["expected_answer"],
                            filt, top3, "", ""])
    _write(AUDIT / "metadata_labels_test.csv",
           ["question_id", "strategy", "rank", "chunk_id", "source", "page", "section",
            "applied_filter", "chunk_text_preview", "source_ok_1_0", "page_ok_1_0",
            "section_ok_1_0", "metadata_ok_1_0", "error_note"], meta_rows)
    _write(AUDIT / "grounding_labels_test.csv",
           ["question_id", "strategy", "question", "expected_answer", "applied_filter",
            "top3_text", "label_0_2", "reason"], ground_rows)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    {"coherence": coherence, "test": test_sheets}.get(
        command, lambda: sys.exit("usage: make_audit.py coherence|test"))()
