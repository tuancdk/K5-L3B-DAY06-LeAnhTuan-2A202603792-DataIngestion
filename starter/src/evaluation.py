"""Score retrieved evidence per query and aggregate Hit@1 / Evidence Hit@3.

A retrieved chunk is a *hit* when it comes from the gold document, from one of the
gold pages, and contains the WHOLE evidence quote after canonicalization (NFC,
whitespace collapsed, casefolded). Partial evidence is not a hit; such cases are
flagged so they can be analysed as chunk-boundary errors. Human labels (coherence,
grounding) are NOT inferred here -- they are collected separately.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Any


def canon(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip().casefold()


def _evidence_overlap(evidence: str, text: str, min_words: int = 4) -> bool:
    """True if the chunk holds a run of >= ``min_words`` consecutive evidence words."""
    words = canon(evidence).split()
    chunk = canon(text)
    n = min(min_words, len(words))
    return any(" ".join(words[i:i + n]) in chunk for i in range(len(words) - n + 1))


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]],
                   applied_filter: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a full trace for one (question, strategy) run."""
    evidence = canon(question["evidence_quote"])
    gold_pages = set(question["gold_pages"])
    rows = []
    for rank, hit in enumerate(hits, start=1):
        on_gold_page = (hit["document_id"] == question["gold_source_id"]
                        and hit["page"] in gold_pages)
        full = evidence in canon(hit["text"])
        rows.append({
            "rank": rank, "chunk_id": hit["chunk_id"], "page": hit["page"],
            "section": hit.get("section"), "score": hit["score"],
            "gold_page": on_gold_page, "full_evidence": full, "hit": on_gold_page and full,
            "partial_evidence": on_gold_page and not full
                                and _evidence_overlap(question["evidence_quote"], hit["text"]),
            "heading": hit.get("metadata", {}).get("heading"),
            "text": hit["text"],
        })
    top3 = rows[:3]
    first_hit = next((r["rank"] for r in rows if r["hit"]), None)
    compliant = None
    if applied_filter:
        compliant = all(r["section"] == applied_filter.get("section") for r in top3)
    return {
        "id": question["id"], "split": question["split"], "type": question.get("type"),
        "question": question["question"], "expected_answer": question["expected_answer"],
        "gold_source_id": question["gold_source_id"], "gold_pages": question["gold_pages"],
        "evidence_quote": question["evidence_quote"],
        "question_filter": question.get("filter"), "applied_filter": applied_filter,
        "hit_at_1": bool(rows) and rows[0]["hit"],
        "hit_at_3": any(r["hit"] for r in top3),
        "gold_page_at_3": any(r["gold_page"] for r in top3),
        "evidence_split_suspect": (not any(r["hit"] for r in top3)
                                   and any(r["partial_evidence"] for r in top3)),
        "first_hit_rank": first_hit,
        "filter_compliant": compliant,
        "hits": rows,
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate by strategy and split; always report numerator/denominator."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for t in traces:
        groups[(t["strategy"], t["split"])].append(t)
    summary: dict[str, Any] = {}
    for (strategy, split), items in sorted(groups.items()):
        n = len(items)
        hit1 = sum(t["hit_at_1"] for t in items)
        hit3 = sum(t["hit_at_3"] for t in items)
        page3 = sum(t["gold_page_at_3"] for t in items)
        filtered = [t for t in items if t["applied_filter"]]
        with_filter_q = [t for t in items if t["question_filter"]]
        rr = [1 / t["first_hit_rank"] if t["first_hit_rank"] and t["first_hit_rank"] <= 3 else 0
              for t in items]
        summary.setdefault(strategy, {})[split] = {
            "n": n,
            "hit_at_1": f"{hit1}/{n}", "hit_at_1_rate": round(hit1 / n, 3),
            "evidence_hit_at_3": f"{hit3}/{n}", "evidence_hit_at_3_rate": round(hit3 / n, 3),
            "gold_page_at_3": f"{page3}/{n}",
            "mrr_at_3": round(sum(rr) / n, 3),
            "evidence_split_suspects": [t["id"] for t in items if t["evidence_split_suspect"]],
            "misses_at_3": [t["id"] for t in items if not t["hit_at_3"]],
            "filter_questions_hit_at_3": (f"{sum(t['hit_at_3'] for t in with_filter_q)}/"
                                          f"{len(with_filter_q)}") if with_filter_q else "N/A",
            "filter_compliance": (f"{sum(bool(t['filter_compliant']) for t in filtered)}/"
                                  f"{len(filtered)}") if filtered else "N/A",
        }
    return summary
