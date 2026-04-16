"""
Normalize PyMuPDF geometry: Rect objects vs 4-float sequences across versions.
"""
from __future__ import annotations

from typing import Any


def normalize_quad(rect: Any) -> tuple[float, float, float, float]:
    """
    Convert fitz.Rect or a sequence of 4 numbers to (x0, y0, x1, y1) floats.

    Some PyMuPDF builds return tuples for tab.bbox and drawing['rect'].
    """
    if rect is None:
        raise ValueError("rect is None")
    if hasattr(rect, "x0"):
        return (
            float(rect.x0),
            float(rect.y0),
            float(rect.x1),
            float(rect.y1),
        )
    seq = tuple(rect)
    if len(seq) < 4:
        raise ValueError("rect sequence must have at least 4 elements")
    return (float(seq[0]), float(seq[1]), float(seq[2]), float(seq[3]))
