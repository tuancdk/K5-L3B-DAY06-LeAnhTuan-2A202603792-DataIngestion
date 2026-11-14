"""Rebuild reading order from Google Vision word boxes so table rows stay together.

Vision's plain ``text`` follows detected blocks, which on financial tables puts all row
labels first and each numeric column later. Here every word is placed by its box:
1. deskew: rotate coordinates by the median word baseline angle (handles scan tilt and
   rotated pages);
2. rows: words whose vertical centres lie within ~0.6 x median word height join a row;
3. columns: inside a row, a horizontal gap wider than ``col_gap`` x median word height
   becomes " | ", otherwise Vision's own break (space / no space) is kept.
Prose comes out as ordinary lines; tables come out as "label | note | value | value".
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

_SPACE_BREAKS = {"SPACE", "SURE_SPACE", "EOL_SURE_SPACE", "LINE_BREAK"}


@dataclass
class _Word:
    text: str
    x0: float
    x1: float
    yc: float
    h: float
    space_after: bool


def _rotate(x: float, y: float, cos_a: float, sin_a: float) -> tuple[float, float]:
    return x * cos_a + y * sin_a, -x * sin_a + y * cos_a


def _words(annotation: dict) -> list[_Word]:
    raw = []
    for page in annotation.get("pages", []):
        for block in page.get("blocks", []):
            for paragraph in block.get("paragraphs", []):
                for word in paragraph.get("words", []):
                    symbols = word.get("symbols", [])
                    text = "".join(s.get("text", "") for s in symbols)
                    vertices = word.get("boundingBox", {}).get("vertices", [])
                    if not text or len(vertices) != 4:
                        continue
                    pts = [(v.get("x", 0), v.get("y", 0)) for v in vertices]
                    brk = (symbols[-1].get("property", {}).get("detectedBreak", {}) or {}).get("type")
                    raw.append((text, pts, brk in _SPACE_BREAKS))
    if not raw:
        return []

    # Baseline angle from vertex 0 -> 1 (Vision orders vertices in reading direction).
    angles = [math.atan2(p[1][1] - p[0][1], p[1][0] - p[0][0]) for _, p, _ in raw
              if (p[1][0] - p[0][0]) ** 2 + (p[1][1] - p[0][1]) ** 2 > 25]
    angle = statistics.median(angles) if angles else 0.0
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    words = []
    for text, pts, space in raw:
        rot = [_rotate(x, y, cos_a, sin_a) for x, y in pts]
        xs, ys = [r[0] for r in rot], [r[1] for r in rot]
        words.append(_Word(text, min(xs), max(xs), (min(ys) + max(ys)) / 2,
                           max(ys) - min(ys), space))
    return words


def rebuild_text(annotation: dict, row_tol: float = 0.6, col_gap: float = 1.6) -> str:
    """Return page text with one visual row per line and ' | ' between table columns."""
    words = _words(annotation)
    if not words:
        return ""
    med_h = statistics.median(w.h for w in words) or 1.0

    rows: list[list[_Word]] = []
    centres: list[float] = []
    for word in sorted(words, key=lambda w: w.yc):
        if rows and abs(word.yc - centres[-1]) <= row_tol * med_h:
            rows[-1].append(word)
            centres[-1] = statistics.fmean(w.yc for w in rows[-1])
        else:
            rows.append([word])
            centres.append(word.yc)

    lines = []
    for row in rows:
        row.sort(key=lambda w: w.x0)
        parts = [row[0].text]
        for prev, cur in zip(row, row[1:]):
            gap = cur.x0 - prev.x1
            if gap > col_gap * med_h:
                parts.append(" | ")
            elif prev.space_after or gap > 0.6 * med_h:  # trust Vision's break first
                parts.append(" ")
            parts.append(cur.text)
        lines.append("".join(parts))
    return "\n".join(lines)
