"""Read and normalize source data, keeping every page traceable to its source.

Two readers share one output contract (``DocumentPage``):
- ``load_pdf``: text-based PDFs via pypdf.
- ``load_ocr_pages``: scanned PDFs already OCR'd by ``src.ocr`` (one JSON per page).
``load_corpus`` drives either reader from ``data/manifest.json`` so page selection,
exclusions and section labels live in one documented place.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from .layout import rebuild_text
from .models import DocumentPage

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifest.json"

_INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
_SPACES = re.compile(r"[ \t  -  　]+")
_NOISE_LINE = re.compile(r"^[\s\-–—_|.•·:;,'\"`~]*$")  # OCR specks: "-", "||", "." ...

# Running header printed at the top of every MB page; it carries no page content and
# would otherwise make every chunk look similar to the embedder.
_RUNNING_HEADER = [
    re.compile(r"^(Ngân hàng Thương mại Cổ phần Quân đội|Đơn vị báo cáo: Ngân hàng TMCP Quân Đội)$", re.I),
    re.compile(r"^Địa chỉ: Số 18 Lê Văn Lương", re.I),
]
_COLUMN_SEP = " | "


def normalize_text(text: str) -> str:
    """NFC-normalize and tidy whitespace while keeping line structure.

    Line breaks are kept because headings, numbered notes and table rows are
    the structure that paragraph chunking relies on.
    """
    text = unicodedata.normalize("NFC", text).translate(_INVISIBLE)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    lines = ["" if _NOISE_LINE.match(line) else line for line in lines]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def strip_running_header(text: str, max_lines: int = 3) -> str:
    """Remove the bank's name/address header if it appears in the first lines.

    With row layout the header can share a row with a form code
    ("Ngân hàng ... | B03a/TCTD-HN"): only the header cell is dropped.
    """
    lines = text.split("\n")
    head, rest = lines[:max_lines], lines[max_lines:]
    kept = []
    for line in head:
        cells = [c for c in line.split(_COLUMN_SEP)
                 if not any(p.match(c.strip()) for p in _RUNNING_HEADER)]
        if cells:
            kept.append(_COLUMN_SEP.join(cells))
    return "\n".join(kept + rest).strip()


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract each physical page of a text-based PDF; flag pages with no text.

    Pages that come back empty are kept with ``metadata['empty']=True`` so the
    caller can log them; OCR for scanned pages is a separate step (``src.ocr``).
    """
    reader = PdfReader(path)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")
        pages.append(DocumentPage(document_id, source, number, text,
                                  {"extraction": "pypdf", "empty": not text}))
    return pages


def load_ocr_pages(ocr_dir: Path, document_id: str, source: str, pages: list[int],
                   metadata_for_page: dict[int, dict[str, Any]] | None = None,
                   text_layout: str = "vision",
                   ) -> tuple[list[DocumentPage], list[dict[str, Any]]]:
    """Load cached OCR pages written by ``src.ocr``; return (pages, errors).

    ``text_layout="vision"`` uses Vision's own text order; ``"rows"`` rebuilds visual
    rows from word boxes (``src.layout``) so table labels stay next to their values.
    Missing cache files and pages that are empty after normalization are
    reported as errors instead of being silently skipped.
    """
    if text_layout not in ("vision", "rows"):
        raise ValueError(f"unknown text_layout: {text_layout}")
    metadata_for_page = metadata_for_page or {}
    loaded, errors = [], []
    for number in pages:
        path = ocr_dir / f"page_{number:03d}.json"
        if not path.exists():
            errors.append({"document_id": document_id, "page": number,
                           "error": f"OCR cache missing: {path.name}"})
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if text_layout == "rows":
            raw = rebuild_text(record.get("full_text_annotation", {}))
        else:
            raw = record.get("text", "")
        text = strip_running_header(normalize_text(raw))
        if not text:
            errors.append({"document_id": document_id, "page": number,
                           "error": "empty after normalization"})
            continue
        metadata = {"extraction": "ocr", "text_layout": text_layout,
                    "ocr_mean_confidence": record.get("mean_confidence"),
                    "ocr_low_confidence_words": record.get("low_confidence_words"),
                    **metadata_for_page.get(number, {})}
        loaded.append(DocumentPage(document_id, source, number, text, metadata))
    return loaded, errors


def section_lookup(sections: list[dict[str, Any]]) -> dict[int, dict[str, str]]:
    """Map page number -> {section, section_title} from inclusive page ranges."""
    lookup: dict[int, dict[str, str]] = {}
    for sec in sections:
        first, last = sec["pages"]
        for number in range(first, last + 1):
            if number in lookup:
                raise ValueError(f"page {number} assigned to two sections")
            lookup[number] = {"section": sec["id"], "section_title": sec["title"]}
    return lookup


def counted_pages(source: dict[str, Any]) -> list[int]:
    """Physical pages minus documented exclusions."""
    excluded = {item["page"] for item in source.get("excluded_pages", [])}
    return [p for p in range(1, source["physical_pages"] + 1) if p not in excluded]


def load_corpus(manifest_path: Path = DEFAULT_MANIFEST
                ) -> tuple[list[DocumentPage], list[dict[str, Any]]]:
    """Load every counted page of every manifest source; return (pages, errors)."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_pages, all_errors = [], []
    for src in manifest["sources"]:
        sections = section_lookup(src.get("sections", []))
        shared = {key: src[key] for key in ("report_period", "report_type", "language")
                  if key in src}
        per_page = {p: {**shared, **sections.get(p, {})} for p in counted_pages(src)}
        extraction = src["extraction"]
        if extraction["method"] == "ocr":
            pages, errors = load_ocr_pages(REPO_ROOT / extraction["ocr_dir"],
                                           src["source_id"], src["source_path_or_url"],
                                           counted_pages(src), per_page,
                                           extraction.get("text_layout", "vision"))
        elif extraction["method"] == "pypdf":
            wanted = set(counted_pages(src))
            pages, errors = [], []
            for page in load_pdf(REPO_ROOT / src["local_path"], src["source_id"],
                                 src["source_path_or_url"]):
                if page.page not in wanted:
                    continue
                if page.metadata["empty"]:
                    errors.append({"document_id": src["source_id"], "page": page.page,
                                   "error": "no extractable text"})
                    continue
                page.metadata.update(per_page.get(page.page, {}))
                pages.append(page)
        else:
            raise ValueError(f"unknown extraction method: {extraction['method']}")
        all_pages.extend(pages)
        all_errors.extend(errors)
    return all_pages, all_errors
