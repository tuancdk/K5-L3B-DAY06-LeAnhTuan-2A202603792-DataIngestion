"""Validate benchmark/questions.json against the corpus WITHOUT running retrieval.

Checks structure, split sizes, filter rules (BENCHMARK.md) and that every
evidence_quote occurs verbatim (whitespace/case-insensitive) on every gold page.
Run from starter/:  python scripts/validate_benchmark.py
"""
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ingestion import DEFAULT_MANIFEST, REPO_ROOT, load_corpus  # noqa: E402

QUESTIONS = REPO_ROOT / "benchmark" / "questions.json"
REQUIRED = ("id", "split", "question", "expected_answer", "gold_source_id",
            "gold_pages", "evidence_quote", "filter")


def canon(text: str) -> str:
    """Same normalization as the hit definition: NFC, collapse whitespace, casefold."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip().casefold()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    bench = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    questions = bench["questions"]
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    sections = {s["id"] for src in manifest["sources"] for s in src["sections"]}
    pages, _ = load_corpus(DEFAULT_MANIFEST)
    page_text = {(p.document_id, p.page): canon(p.text) for p in pages}
    page_section = {(p.document_id, p.page): p.metadata.get("section") for p in pages}
    allowed = set(bench["meta"]["allowed_filter_fields"])
    problems: list[str] = []

    ids = [q.get("id") for q in questions]
    if len(set(ids)) != len(ids):
        problems.append("duplicate ids")
    for q in questions:
        missing = [k for k in REQUIRED if k not in q]
        if missing:
            problems.append(f"{q.get('id')}: missing {missing}")
            continue
        quote = canon(q["evidence_quote"])
        found = []
        for page in q["gold_pages"]:
            key = (q["gold_source_id"], page)
            if key not in page_text:
                problems.append(f"{q['id']}: gold page {page} not in corpus")
            elif quote in page_text[key]:
                found.append(page)
            else:
                problems.append(f"{q['id']}: evidence NOT found on page {page}")
        if q["filter"]:
            for field, value in q["filter"].items():
                if field not in allowed:
                    problems.append(f"{q['id']}: filter field '{field}' not allowed")
                elif field == "section":
                    if value not in sections:
                        problems.append(f"{q['id']}: unknown section '{value}'")
                    bad = [p for p in q["gold_pages"]
                           if page_section.get((q["gold_source_id"], p)) != value]
                    if bad:
                        problems.append(f"{q['id']}: gold pages {bad} outside filter section")
        print(f"{q['id']:4s} {q['split']:4s} {q.get('type', ''):20s} pages={q['gold_pages']} "
              f"quote_len={len(q['evidence_quote']):3d} found_on={found} "
              f"filter={q['filter']['section'] if q['filter'] else '-'}")

    splits = Counter(q["split"] for q in questions)
    filtered = Counter(q["split"] for q in questions if q.get("filter"))
    types = Counter(q.get("type") for q in questions)
    print(f"\nTotal {len(questions)} | splits {dict(splits)} | with filter {dict(filtered)}")
    print(f"Types {dict(types)}")
    if len(questions) < 20 or splits["dev"] < 15 or splits["test"] < 5:
        problems.append("need >= 20 questions with >= 15 dev and >= 5 test")
    if sum(filtered.values()) < 3 or filtered["test"] < 1:
        problems.append("need >= 3 filter questions with >= 1 in test")
    print(f"Status: {bench['meta']['status']}")
    for p in problems:
        print(f"[problem] {p}")
    print("VALID" if not problems else f"{len(problems)} problem(s)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
