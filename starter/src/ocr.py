"""OCR scanned PDFs page by page with Google Cloud Vision (DOCUMENT_TEXT_DETECTION).

Each physical PDF page is rendered to JPEG, sent to the Vision REST API and cached as
``<out_dir>/page_XXX.json`` so page numbers stay aligned with ``gold_page`` and re-runs
cost nothing. A summary with per-page confidence and errors goes to ``ocr_log.json``.

Usage (from starter/):
    python -m src.ocr data/raw_mbb/MBB_Baocaotaichinh_Q3_2026_Hopnhat.pdf data/ocr/mbb_q3_2026
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pymupdf as fitz
import requests
from dotenv import load_dotenv

VISION_URL = "https://vision.googleapis.com/v1/images:annotate"
LOW_CONFIDENCE = 0.80  # words below this are counted as suspicious


def render_page(doc: fitz.Document, index: int, dpi: int) -> bytes:
    """Render one page to JPEG bytes (JPEG keeps requests well under the 10 MB limit)."""
    pix = doc[index].get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    return pix.tobytes("jpeg", jpg_quality=90)


def call_vision(image: bytes, api_key: str, language_hints: list[str],
                retries: int = 3, timeout: int = 120) -> dict:
    """Send one image to Vision; raise on HTTP or API-level errors (no silent fallback)."""
    body = {"requests": [{
        "image": {"content": base64.b64encode(image).decode("ascii")},
        "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
        "imageContext": {"languageHints": language_hints},
    }]}
    # Key goes in a header, not the URL, so it never shows up in logs or tracebacks.
    headers = {"X-Goog-Api-Key": api_key, "Content-Type": "application/json"}
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(VISION_URL, headers=headers, json=body, timeout=timeout)
            if resp.status_code in (429, 500, 503) and attempt < retries:
                time.sleep(2 ** attempt)
                continue
            if resp.status_code != 200:
                message = resp.json().get("error", {}).get("message", resp.text[:300])
                raise RuntimeError(f"HTTP {resp.status_code}: {message}")
            result = resp.json()["responses"][0]
            if "error" in result:
                raise RuntimeError(f"Vision error: {result['error'].get('message')}")
            return result
        except requests.RequestException as exc:  # network problems: retry
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Vision request failed after {retries} attempts: {last_error}")


def summarize_annotation(annotation: dict) -> dict:
    """Extract text plus confidence statistics from a fullTextAnnotation."""
    confidences: list[float] = []
    for page in annotation.get("pages", []):
        for block in page.get("blocks", []):
            for paragraph in block.get("paragraphs", []):
                for word in paragraph.get("words", []):
                    if "confidence" in word:
                        confidences.append(word["confidence"])
    text = unicodedata.normalize("NFC", annotation.get("text", ""))
    return {
        "text": text,
        "chars": len(text.strip()),
        "word_count": len(confidences),
        "mean_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
        "low_confidence_words": sum(c < LOW_CONFIDENCE for c in confidences),
    }


def ocr_pdf(pdf_path: Path, out_dir: Path, api_key: str, dpi: int = 300,
            language_hints: list[str] | None = None, pages: list[int] | None = None,
            force: bool = False) -> dict:
    """OCR every (or selected, 1-based) page; reuse cached pages unless ``force``."""
    language_hints = language_hints or ["vi"]
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    targets = pages or list(range(1, doc.page_count + 1))
    log_pages, errors, api_calls = [], [], 0

    for page_no in targets:
        cache = out_dir / f"page_{page_no:03d}.json"
        if cache.exists() and not force:
            record = json.loads(cache.read_text(encoding="utf-8"))
            status = "cached"
        else:
            try:
                result = call_vision(render_page(doc, page_no - 1, dpi), api_key, language_hints)
                api_calls += 1
            except RuntimeError as exc:
                errors.append({"page": page_no, "error": str(exc)})
                print(f"page {page_no:3d}  ERROR  {exc}", file=sys.stderr)
                continue
            annotation = result.get("fullTextAnnotation", {})
            record = {"page": page_no, "source": pdf_path.name, "dpi": dpi,
                      **summarize_annotation(annotation), "full_text_annotation": annotation}
            cache.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            status = "ocr"
        entry = {k: record[k] for k in ("page", "chars", "word_count",
                                         "mean_confidence", "low_confidence_words")}
        entry["empty"] = record["chars"] == 0
        log_pages.append(entry)
        print(f"page {page_no:3d}  {status:6s}  chars={entry['chars']:5d}  "
              f"conf={entry['mean_confidence']}  low={entry['low_confidence_words']}")

    log = {
        "source": str(pdf_path).replace("\\", "/"),
        "engine": "Google Cloud Vision API v1 images:annotate DOCUMENT_TEXT_DETECTION",
        "language_hints": language_hints, "dpi": dpi,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "physical_pages": doc.page_count, "api_calls_this_run": api_calls,
        "low_confidence_threshold": LOW_CONFIDENCE,
        "pages": sorted(log_pages, key=lambda p: p["page"]), "errors": errors,
    }
    (out_dir / "ocr_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    return log


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pdf", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--pages", type=str, help="e.g. 1-3,7 (1-based); default all")
    parser.add_argument("--force", action="store_true", help="ignore cache and re-OCR")
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    api_key = os.getenv("GOOGLE_VISION_API_KEY")
    if not api_key:
        sys.exit("GOOGLE_VISION_API_KEY missing in starter/.env")

    pages = None
    if args.pages:
        pages = []
        for part in args.pages.split(","):
            lo, _, hi = part.partition("-")
            pages.extend(range(int(lo), int(hi or lo) + 1))

    log = ocr_pdf(args.pdf, args.out_dir, api_key, args.dpi, pages=pages, force=args.force)
    done = [p for p in log["pages"] if not p["empty"]]
    print(f"\nPages with text: {len(done)}/{len(log['pages'])} | "
          f"API calls: {log['api_calls_this_run']} | errors: {len(log['errors'])}")
    if log["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
