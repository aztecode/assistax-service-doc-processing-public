"""
Diagnostic script: inspect a single page of a PDF to understand
what the extraction pipeline sees (tables, visual rects, text blocks, bboxes).

Usage:
    python -m scripts.diagnose_page path/to/file.pdf --page 306
    python -m scripts.diagnose_page path/to/file.pdf --page 306 --json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from typing import Any

import fitz  # PyMuPDF

from pipeline.pdf_extractor import (
    _extract_tables_from_page,
    _extract_text_excluding_bboxes,
    _is_editorial_boxed_note,
    _normalize_text,
    _rects_overlap,
)
from settings import settings


@dataclass
class DiagnosticResult:
    page_number: int
    tables_detected: int
    table_details: list[dict[str, Any]]
    visual_rects: list[dict[str, Any]]
    text_blocks_total: int
    text_blocks_excluded: int
    text_blocks_kept: int
    final_text_preview: str
    raw_text_preview: str


def _extract_visual_rects_diagnostic(
    page: Any,
) -> list[dict[str, Any]]:
    """Extract drawn rectangles/paths from the page for diagnostic purposes."""
    results: list[dict[str, Any]] = []
    try:
        drawings = page.get_drawings()
    except Exception:
        return results

    for d in drawings:
        items = d.get("items", [])
        rect = d.get("rect")
        if rect is None:
            continue
        bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
        width = rect.x1 - rect.x0
        height = rect.y1 - rect.y0
        if width < 50 or height < 20:
            continue
        has_stroke = d.get("color") is not None
        has_fill = d.get("fill") is not None
        stroke_width = d.get("width", 0)
        item_types = [item[0] for item in items]
        has_rect_item = "re" in item_types
        has_line_items = item_types.count("l") >= 4

        results.append({
            "bbox": bbox,
            "width": round(width, 1),
            "height": round(height, 1),
            "has_stroke": has_stroke,
            "has_fill": has_fill,
            "stroke_width": round(stroke_width, 2),
            "item_types": item_types[:10],
            "has_rect_item": has_rect_item,
            "has_line_items": has_line_items,
        })
    return results


def diagnose_page(
    pdf_path: str,
    page_number: int,
) -> DiagnosticResult:
    """Analyze a single page and return diagnostic info."""
    with open(pdf_path, "rb") as f:
        source = f.read()

    doc = fitz.open(stream=source, filetype="pdf")
    if page_number < 1 or page_number > len(doc):
        doc.close()
        raise ValueError(f"Page {page_number} out of range (1-{len(doc)})")

    page = doc[page_number - 1]

    tables, _, _ = _extract_tables_from_page(
        page,
        page_number,
        0,
        relax_prose_table_filter=settings.RELAX_PROSE_TABLE_FILTER,
    )
    table_details: list[dict[str, Any]] = []
    for t in tables:
        table_details.append({
            "table_index": t.table_index,
            "bbox": t.bbox,
            "is_boxed_note": t.is_boxed_note,
            "rows_count": len(t.rows),
            "cols_count": max((len(r) for r in t.rows), default=0),
            "first_cell_preview": (
                t.rows[0][0][:120] if t.rows and t.rows[0] else ""
            ),
        })

    visual_rects = _extract_visual_rects_diagnostic(page)

    raw_text = page.get_text("text")
    all_blocks = page.get_text("blocks")
    text_blocks = [b for b in all_blocks if b[6] == 0]
    exclude_bboxes = [t.bbox for t in tables]

    excluded_count = 0
    for block in text_blocks:
        bbox = (block[0], block[1], block[2], block[3])
        if any(_rects_overlap(bbox, eb, 0.5) for eb in exclude_bboxes):
            excluded_count += 1

    final_text = _extract_text_excluding_bboxes(page, exclude_bboxes)
    final_text = _normalize_text(final_text)

    doc.close()

    return DiagnosticResult(
        page_number=page_number,
        tables_detected=len(tables),
        table_details=table_details,
        visual_rects=visual_rects,
        text_blocks_total=len(text_blocks),
        text_blocks_excluded=excluded_count,
        text_blocks_kept=len(text_blocks) - excluded_count,
        final_text_preview=final_text[:500],
        raw_text_preview=raw_text[:500],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose PDF page extraction")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--page", type=int, required=True, help="1-indexed page number")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    result = diagnose_page(args.pdf_path, args.page)

    if args.json:
        print(json.dumps(asdict(result), indent=2, ensure_ascii=False, default=str))
    else:
        print(f"\n{'='*70}")
        print(f"  DIAGNOSTIC: Page {result.page_number}")
        print(f"{'='*70}")
        print(f"\nTables detected: {result.tables_detected}")
        for td in result.table_details:
            print(f"  - Table {td['table_index']}: bbox={td['bbox']}, "
                  f"boxed_note={td['is_boxed_note']}, "
                  f"rows={td['rows_count']}, cols={td['cols_count']}")
            print(f"    first_cell: {td['first_cell_preview'][:80]}")

        print(f"\nVisual rects found: {len(result.visual_rects)}")
        for vr in result.visual_rects:
            print(f"  - bbox={vr['bbox']}, "
                  f"stroke={vr['has_stroke']}, fill={vr['has_fill']}, "
                  f"w={vr['width']}, h={vr['height']}, "
                  f"rect_item={vr['has_rect_item']}, "
                  f"line_items={vr['has_line_items']}")

        print(f"\nText blocks: {result.text_blocks_total} total, "
              f"{result.text_blocks_excluded} excluded, "
              f"{result.text_blocks_kept} kept")

        print(f"\n--- RAW TEXT (first 500 chars) ---")
        print(result.raw_text_preview)
        print(f"\n--- FINAL TEXT (first 500 chars) ---")
        print(result.final_text_preview)


if __name__ == "__main__":
    main()
