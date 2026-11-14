"""Fixed-size (strategy A) and structure-aware paragraph chunking (strategies B/C).

Chunks never cross a page boundary, so every chunk keeps an exact page for tracing.
Chunk IDs are deterministic (document, page, chunking config, index) so re-ingesting
the same data overwrites rows instead of duplicating them; ``content_hash`` lets the
store skip re-embedding unchanged chunks and detect edited ones.
"""
from __future__ import annotations

import hashlib
import re

from .models import Chunk, DocumentPage

# Numbered headings used in the MB notes: "14.", "5.2.1 Rủi ro lãi suất", "III- THÔNG TIN",
# "IV-" -- plus short all-caps title lines. List items "(a)", "a." are NOT headings: they
# continue the clause above them and should stay in the same chunk.
_HEADING = re.compile(r"^((\d{1,2}(\.\d{1,2}){0,3}\.?)|([IVX]{1,4}\s*[-–.]))(\s|$)")
_SENTENCE_END = re.compile(r"[.:;!?)]$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.;!?])\s+")


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _make_chunk(page: DocumentPage, config: str, index: int, text: str,
                start: int, end: int, extra: dict | None = None) -> Chunk:
    metadata = {**page.metadata, "chunking": config, "chunk_index": index,
                "char_start": start, "char_end": end, "content_hash": _hash(text),
                **(extra or {})}
    return Chunk(chunk_id=f"{page.document_id}:p{page.page:03d}:{config}:{index:03d}",
                 document_id=page.document_id, source=page.source, page=page.page,
                 text=text, metadata=metadata)


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Character windows of ``size`` with ``overlap``; ignores structure on purpose (baseline)."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    text, config, step = page.text, f"fixed{size}o{overlap}", size - overlap
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(_make_chunk(page, config, len(chunks), piece, start, end))
        if end == len(text):
            break
        start += step
    return chunks


def is_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 120:
        return False
    if _HEADING.match(line):
        return True
    letters = [c for c in line if c.isalpha()]
    return len(letters) >= 8 and all(c.isupper() for c in letters)


def _paragraphs(text: str) -> list[tuple[int, int, bool]]:
    """Split OCR text into (start, end, is_heading) spans.

    OCR keeps visual line breaks but rarely blank lines, so a paragraph ends at a blank
    line, before a heading, or after a line ending in sentence punctuation when the next
    line starts a new sentence (uppercase, digit or bullet).
    """
    spans: list[tuple[int, int, bool]] = []
    lines, offset = [], 0
    for raw in text.split("\n"):
        lines.append((offset, offset + len(raw), raw))
        offset += len(raw) + 1

    para_start: int | None = None
    para_end = 0
    for i, (start, end, line) in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            if para_start is not None:
                spans.append((para_start, para_end, False))
                para_start = None
            continue
        if is_heading(stripped):
            if para_start is not None:
                spans.append((para_start, para_end, False))
                para_start = None
            spans.append((start, end, True))
            continue
        if para_start is None:
            para_start = start
        para_end = end
        nxt = lines[i + 1][2].strip() if i + 1 < len(lines) else ""
        if _SENTENCE_END.search(stripped) and nxt and (nxt[0].isupper() or nxt[0].isdigit()
                                                       or nxt[0] in "-+•"):
            spans.append((para_start, para_end, False))
            para_start = None
    if para_start is not None:
        spans.append((para_start, para_end, False))
    return spans


def _split_long(text: str, start: int, max_size: int) -> list[tuple[int, int]]:
    """Split an oversized paragraph at sentence ends, else at the last whitespace."""
    pieces, cursor = [], start
    end_all = start + len(text)
    while end_all - cursor > max_size:
        window = text[cursor - start: cursor - start + max_size]
        cut = max((m.end() for m in _SENTENCE_SPLIT.finditer(window)), default=0)
        if cut < max_size // 3:  # no usable sentence break: fall back to whitespace
            cut = window.rfind(" ")
            cut = cut if cut > max_size // 3 else max_size
        pieces.append((cursor, cursor + cut))
        cursor += cut
    pieces.append((cursor, end_all))
    return pieces


def by_paragraph(page: DocumentPage, max_size: int, min_size: int = 250) -> list[Chunk]:
    """Pack whole paragraphs into chunks of at most ~``max_size`` characters.

    A heading opens a new chunk once the current one holds at least ``min_size``
    characters (OCR turns table cells such as "1" into heading-like lines, so breaking
    on every heading would leave fragments). A chunk that is only headings is never
    closed, so titles stay attached to the text after them. A short tail is merged
    into the previous chunk. The nearest heading is kept as metadata.
    """
    text, config = page.text, f"para{max_size}"
    units: list[tuple[int, int, bool]] = []
    for start, end, heading in _paragraphs(text):
        if not heading and end - start > max_size:
            units.extend((s, e, False) for s, e in _split_long(text[start:end], start, max_size))
        else:
            units.append((start, end, heading))

    spans: list[list] = []  # [start, end, heading_text]
    current: list[tuple[int, int, bool]] = []
    last_heading: str | None = None
    chunk_heading: str | None = None

    for unit in units:
        start, end, heading = unit
        if current:
            cur_len = current[-1][1] - current[0][0]
            only_headings = all(u[2] for u in current)
            exceeds = end - current[0][0] > max_size
            if not only_headings and (exceeds or (heading and cur_len >= min_size)):
                spans.append([current[0][0], current[-1][1], chunk_heading])
                current = []
        if heading:
            last_heading = text[start:end].strip()
        if not current:
            chunk_heading = last_heading
        current.append(unit)
    if current:
        spans.append([current[0][0], current[-1][1], chunk_heading])

    if len(spans) > 1 and spans[-1][1] - spans[-1][0] < min_size \
            and spans[-1][1] - spans[-2][0] <= max_size:
        tail = spans.pop()
        spans[-1][1] = tail[1]

    chunks: list[Chunk] = []
    for start, end, heading_text in spans:
        piece = text[start:end].strip()
        if piece:
            chunks.append(_make_chunk(page, config, len(chunks), piece, start, end,
                                      {"heading": heading_text}))
    return chunks
