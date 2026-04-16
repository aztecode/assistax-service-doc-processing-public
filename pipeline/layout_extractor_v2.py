"""
Layout extractor v2: rich per-span extraction from PDFs using PyMuPDF dict mode.

Isolated from the production pipeline — does not modify any shared state.
No classification or filtering is applied here; that is done in later phases.
"""
import logging
import re

import fitz  # PyMuPDF

from pipeline.exceptions import PDFExtractionError
from pipeline.layout_models import (
    DocumentLayout,
    ExtractedSpan,
    LayoutBlock,
    PageLayout,
)

logger = logging.getLogger(__name__)

# PyMuPDF span flags bitmask (defined in MuPDF source; not exported as constants)
_FLAG_ITALIC: int = 2
_FLAG_BOLD: int = 16


def _extract_text_blocks_from_page(page: fitz.Page, page_number: int) -> list[LayoutBlock]:
    """Extract text blocks with per-span style info using PyMuPDF dict mode.

    'dict' mode gives font, size, and flags per span with the full text string,
    unlike 'blocks' mode which only returns concatenated plain text.
    reading_order is set to 0 here; it is assigned by _compute_reading_order.
    """
    raw: dict = page.get_text("dict")  # type: ignore[assignment]
    blocks_out: list[LayoutBlock] = []

    for block_no, block in enumerate(raw.get("blocks", [])):
        if block.get("type") != 0:
            # Type 1 = image block; skip — no text to extract
            continue

        block_bbox: tuple[float, float, float, float] = tuple(block["bbox"])  # type: ignore[assignment]
        spans_out: list[ExtractedSpan] = []
        text_parts: list[str] = []

        for line_no, line in enumerate(block.get("lines", [])):
            for span_no, span in enumerate(line.get("spans", [])):
                span_text: str = span.get("text", "")
                if not span_text.strip():
                    continue

                flags: int = span.get("flags", 0)
                extracted = ExtractedSpan(
                    text=span_text,
                    bbox=tuple(span["bbox"]),  # type: ignore[arg-type]
                    font_size=span.get("size"),
                    font_name=span.get("font"),
                    is_bold=bool(flags & _FLAG_BOLD),
                    is_italic=bool(flags & _FLAG_ITALIC),
                    page_number=page_number,
                    block_no=block_no,
                    line_no=line_no,
                    span_no=span_no,
                )
                spans_out.append(extracted)
                text_parts.append(span_text)

        full_text: str = "".join(text_parts).strip()
        if not full_text:
            # Discard whitespace-only blocks before they enter the pipeline
            continue

        # Collapse runs of 3+ newlines but preserve intentional double-breaks
        full_text = re.sub(r"\n{3,}", "\n\n", full_text)

        blocks_out.append(
            LayoutBlock(
                block_id=f"p{page_number}_b{block_no}",
                page_number=page_number,
                bbox=block_bbox,
                text=full_text,
                kind="text",
                reading_order=0,
                spans=spans_out,
                source="pymupdf_text",
                metadata={},
            )
        )

    return blocks_out


def _extract_candidate_tables_from_page(
    page: fitz.Page,
    page_number: int,
) -> list[dict[str, object]]:
    """Capture candidate table positions from PyMuPDF find_tables().

    These are unfiltered candidates — prose-in-grid detection and actual table
    validation happen in later normalization and classification phases.
    """
    candidates: list[dict[str, object]] = []
    try:
        finder = page.find_tables()
    except Exception as exc:
        logger.debug("find_tables failed on page %d: %s", page_number, exc)
        return candidates

    if not finder or not finder.tables:
        return candidates

    for idx, tab in enumerate(finder.tables):
        try:
            rows_raw: list = tab.extract() or []
            row_count: int = len(rows_raw)
            col_count: int = max((len(r) for r in rows_raw if r), default=0)
            candidates.append(
                {
                    "table_index": idx,
                    "page_number": page_number,
                    "bbox": tuple(tab.bbox),
                    "row_count": row_count,
                    "col_count": col_count,
                }
            )
        except Exception as exc:
            logger.debug(
                "Candidate table extraction failed page=%d index=%d: %s",
                page_number,
                idx,
                exc,
            )

    return candidates


def _extract_visual_frames_from_page(
    page: fitz.Page,
    page_number: int,
) -> list[dict[str, object]]:
    """Capture drawn rectangles as visual signals for bordered notes and boxes.

    Only rectangles large enough to be meaningful editorial frames are kept.
    Smaller decorative lines and thin borders are discarded.
    """
    _MIN_FRAME_WIDTH: float = 50.0
    _MIN_FRAME_HEIGHT: float = 20.0

    frames: list[dict[str, object]] = []
    try:
        drawings: list[dict] = page.get_drawings()
    except Exception as exc:
        logger.debug("get_drawings failed on page %d: %s", page_number, exc)
        return frames

    for drawing in drawings:
        rect = drawing.get("rect")
        if rect is None:
            continue
        try:
            x0 = float(rect.x0)
            y0 = float(rect.y0)
            x1 = float(rect.x1)
            y1 = float(rect.y1)
        except (AttributeError, TypeError, ValueError):
            continue

        if (x1 - x0) < _MIN_FRAME_WIDTH or (y1 - y0) < _MIN_FRAME_HEIGHT:
            continue

        frames.append(
            {
                "bbox": (x0, y0, x1, y1),
                "page_number": page_number,
                "color": drawing.get("color"),
                "fill": drawing.get("fill"),
                "line_width": drawing.get("width"),
            }
        )

    return frames


def _compute_reading_order(
    blocks: list[LayoutBlock],
    page_width: float,
    page_height: float,
) -> list[LayoutBlock]:
    """Assign reading_order by sorting blocks top-to-bottom, left-to-right.

    Legal documents are almost always single-column, so Y-then-X ordering is
    reliable. The Y-band tolerance groups blocks at the same visual line despite
    minor bbox misalignments introduced by PyMuPDF's coordinate rounding.

    page_width and page_height are accepted for future multi-column detection
    (column boundary inference requires knowing the page dimensions).
    """
    _Y_BAND_PTS: float = 5.0

    def _sort_key(block: LayoutBlock) -> tuple[float, float]:
        y_band = (block.bbox[1] // _Y_BAND_PTS) * _Y_BAND_PTS
        return (y_band, block.bbox[0])

    sorted_blocks = sorted(blocks, key=_sort_key)
    return [
        block.model_copy(update={"reading_order": idx})
        for idx, block in enumerate(sorted_blocks)
    ]


def extract_document_layout(pdf_bytes: bytes) -> DocumentLayout:
    """Extract full structural layout from PDF bytes.

    Returns a DocumentLayout with per-span blocks, candidate tables, visual
    drawings, and the native TOC if one exists. No classification or filtering
    is applied — all blocks are returned with kind='text'.

    Raises PDFExtractionError for corrupt or unreadable PDFs.
    """
    try:
        doc: fitz.Document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise PDFExtractionError(f"PDF corrupto o inválido: {exc}") from exc

    native_toc: list[dict[str, object]] = []
    try:
        raw_toc: list = doc.get_toc(simple=True) or []
        for item in raw_toc:
            if not item or len(item) < 3:
                continue
            # get_toc returns page as 1-based; we normalise to 1-based consistently
            # with pdf_extractor.py (adds 1 to match its historical convention)
            native_toc.append(
                {
                    "level": int(item[0]) if item[0] is not None else 1,
                    "title": str(item[1]).strip() if item[1] is not None else "",
                    "page": max(1, int(item[2])) if item[2] is not None else 1,
                }
            )
    except Exception as exc:
        logger.debug("Native TOC extraction failed: %s", exc)

    pages: list[PageLayout] = []
    page_idx: int = 0

    try:
        for page_idx in range(len(doc)):
            page: fitz.Page = doc[page_idx]
            page_number: int = page_idx + 1
            page_rect = page.rect
            page_width: float = float(page_rect.width)
            page_height: float = float(page_rect.height)

            raw_blocks = _extract_text_blocks_from_page(page, page_number)
            ordered_blocks = _compute_reading_order(raw_blocks, page_width, page_height)
            candidate_tables = _extract_candidate_tables_from_page(page, page_number)
            visual_frames = _extract_visual_frames_from_page(page, page_number)

            logger.info(
                "layout_v2.extraction page=%d blocks=%d tables_candidate=%d drawings=%d",
                page_number,
                len(ordered_blocks),
                len(candidate_tables),
                len(visual_frames),
            )

            pages.append(
                PageLayout(
                    page_number=page_number,
                    width=page_width,
                    height=page_height,
                    blocks=ordered_blocks,
                    raw_tables=candidate_tables,
                    raw_drawings=visual_frames,
                )
            )
    except PDFExtractionError:
        raise
    except Exception as exc:
        raise PDFExtractionError(
            f"Error al extraer página {page_idx + 1}: {exc}"
        ) from exc
    finally:
        doc.close()

    total_blocks: int = sum(len(p.blocks) for p in pages)
    logger.info(
        "layout_v2.extraction.completed pages=%d total_blocks=%d toc_entries=%d",
        len(pages),
        total_blocks,
        len(native_toc),
    )

    return DocumentLayout(
        pages=pages,
        native_toc=native_toc,
        metadata={"total_pages": len(pages), "total_blocks": total_blocks},
    )
