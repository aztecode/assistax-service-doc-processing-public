"""
Extracción de texto y tablas desde PDFs con PyMuPDF.
Alineado con assistax-fn: get_text("text"), normalización interna, retorna (pages, toc).
"""
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import fitz  # PyMuPDF

from pipeline.exceptions import PDFExtractionError
from pipeline.pymupdf_bbox import normalize_quad
from settings import settings

logger = logging.getLogger(__name__)

# Palabras clave en headers que indican tabla tarifaria (nunca filtrar).
# Alineado con assistax-fn.
TARIFF_HEADER_KEYWORDS = ("cuota", "rango", "superior", "inferior", "límite", "limite")


@dataclass
class TableBlock:
    """Bloque de tabla extraída."""
    table_index: int
    page_number: int
    markdown: str
    rows: List[List[str]]
    bbox: tuple[float, float, float, float]
    is_boxed_note: bool


@dataclass
class PageContent:
    """Contenido de una página: texto sin tablas + tablas."""
    page_number: int
    text: str
    tables: List[TableBlock]


def sanitize_cell(value: str) -> str:
    """Strip cell content and escape pipe characters for Markdown. Alineado con assistax-fn."""
    if value is None:
        return ""
    s = str(value).strip()
    s = s.replace("|", "\\|")
    return s


def table_rows_to_markdown(rows: List[List[str]], table_index: int) -> str:
    """Convert table rows to Markdown with [TABLE_N]...[/TABLE_N] format. Alineado con assistax-fn."""
    if not rows:
        return f"[TABLE_{table_index}]\n| Col 1 |\n|---|---|\n|  |\n[/TABLE_{table_index}]"

    max_cols = max(len(r) for r in rows) if rows else 0
    if max_cols == 0:
        return f"[TABLE_{table_index}]\n| Col 1 |\n|---|---|\n|  |\n[/TABLE_{table_index}]"

    normalized: List[List[str]] = []
    for row in rows:
        cells = [sanitize_cell(c) for c in row]
        while len(cells) < max_cols:
            cells.append("")
        normalized.append(cells[:max_cols])

    if len(normalized) > 1:
        header_row = normalized[0]
        data_rows = normalized[1:]
    else:
        header_row = [f"Col {i + 1}" for i in range(max_cols)]
        data_rows = normalized

    header_line = "| " + " | ".join(header_row) + " |"
    separator = "|" + "|".join(["---" for _ in range(max_cols)]) + "|"
    lines = [f"[TABLE_{table_index}]", header_line, separator]

    for row in data_rows:
        line = "| " + " | ".join(row) + " |"
        lines.append(line)

    lines.append(f"[/TABLE_{table_index}]")
    return "\n".join(lines)


def _is_tariff_like_table(rows: List[List[str]]) -> bool:
    """
    Return True if table looks like a tariff table (keep it, do not filter).
    Alineado con assistax-fn: keywords + pct_numeric >= 0.4 y len(rows) >= 3.
    """
    if not rows or len(rows) < 2:
        return False
    header_cells = " ".join(rows[0]).lower()
    if any(kw in header_cells for kw in TARIFF_HEADER_KEYWORDS):
        return True
    all_cells = [c for r in rows for c in r if c.strip()]
    if all_cells:
        pct_numeric = sum(1 for c in all_cells if re.search(r"\d", c)) / len(all_cells)
        if pct_numeric >= 0.4 and len(rows) >= 3:
            return True
    return False


def _is_likely_prose_not_table(
    rows: List[List[str]],
    *,
    relax_prose_table_filter: bool,
) -> bool:
    """
    Return True if detected 'table' is likely prose text in a grid layout.
    Legal docs often have invisible lines that make sentences look like tables.
    Alineado con assistax-fn.
    """
    if _is_tariff_like_table(rows):
        return False

    non_empty = [r for r in rows if any(c.strip() for c in r)]
    if not non_empty:
        return True
    max_cols = max(len(r) for r in rows)
    num_rows = len(non_empty)
    all_cells = [c.strip() for r in rows for c in r if c.strip()]

    if num_rows <= 2 and max_cols >= 8:
        # Wide numeric reference grids (e.g. errata cross-refs): keep when relax flag set.
        if relax_prose_table_filter and all_cells:
            cells_with_digit = sum(1 for c in all_cells if re.search(r"\d", c))
            pct_numeric = cells_with_digit / len(all_cells)
            if pct_numeric >= 0.5:
                return False
        return True
    if all_cells and max_cols >= 6:
        avg_cell_len = sum(len(c) for c in all_cells) / len(all_cells)
        cells_with_digit = sum(1 for c in all_cells if re.search(r"\d", c))
        pct_numeric = cells_with_digit / len(all_cells)
        if avg_cell_len < 15 and pct_numeric < 0.35:
            return True
    if all_cells and 4 <= max_cols <= 5:
        avg_cell_len = sum(len(c) for c in all_cells) / len(all_cells)
        cells_with_digit = sum(1 for c in all_cells if re.search(r"\d", c))
        pct_numeric = cells_with_digit / len(all_cells)
        if avg_cell_len < 10 and pct_numeric < 0.25:
            return True
    return False


_EDITORIAL_NOTE_PREFIXES = re.compile(
    r"(?:ACLARACI[OÓ]N|Nota\s*de\s+erratas|Fe\s+de\s+erratas|N\.\s*de\s+E\."
    r"|NOTA\s+DEL\s+EDITOR|Nota\s*:\s|Art[íi]culo\s+(?:reformad[oa]|adicionad[oa]|derogad[oa]))",
    re.IGNORECASE,
)


def _is_editorial_boxed_note(rows: List[List[str]]) -> bool:
    """
    Return True if table is a single-column editorial note (boxed note)
    rather than a real data table. Tariff tables are never classified here.
    """
    if _is_tariff_like_table(rows):
        return False
    non_empty = [r for r in rows if any(c.strip() for c in r)]
    if not non_empty:
        return False
    max_cols = max(len(r) for r in rows)
    if max_cols > 2:
        return False
    all_text = " ".join(c.strip() for r in non_empty for c in r if c.strip())
    if not all_text:
        return False
    if _EDITORIAL_NOTE_PREFIXES.search(all_text):
        return True
    if max_cols == 1 and len(all_text) > 100 and len(non_empty) <= 5:
        words = all_text.split()
        if len(words) >= 10:
            return True
    return False


def _rects_overlap(
    block_bbox: tuple[float, float, float, float],
    exclude_bbox: tuple[float, float, float, float],
    min_ratio: float,
) -> bool:
    """True if at least *min_ratio* of block_bbox's area overlaps with exclude_bbox."""
    bx0, by0, bx1, by1 = block_bbox
    ex0, ey0, ex1, ey1 = exclude_bbox
    ix0 = max(bx0, ex0)
    iy0 = max(by0, ey0)
    ix1 = min(bx1, ex1)
    iy1 = min(by1, ey1)
    if ix0 >= ix1 or iy0 >= iy1:
        return False
    inter_area = (ix1 - ix0) * (iy1 - iy0)
    block_area = (bx1 - bx0) * (by1 - by0)
    if block_area <= 0:
        return False
    return (inter_area / block_area) >= min_ratio


_MIN_RECT_WIDTH = 100.0
_MIN_RECT_HEIGHT = 30.0
_BBOX_MERGE_GAP = 5.0


def _extract_visual_rects(
    page,
    *,
    relaxed_visual_frame_detection: bool,
) -> List[tuple[float, float, float, float]]:
    """
    Detect drawn rectangles (borders / frames) via page.get_drawings().
    Returns bboxes of rectangles large enough to be editorial boxes.
    """
    rects: List[tuple[float, float, float, float]] = []
    try:
        drawings = page.get_drawings()
    except Exception:
        return rects

    relaxed = relaxed_visual_frame_detection
    for d in drawings:
        rect = d.get("rect")
        if rect is None:
            continue
        try:
            x0, y0, x1, y1 = normalize_quad(rect)
        except (ValueError, TypeError):
            continue
        w = x1 - x0
        h = y1 - y0
        if w < _MIN_RECT_WIDTH or h < _MIN_RECT_HEIGHT:
            continue
        if not relaxed and d.get("color") is None:
            continue
        items = d.get("items", [])
        item_types = [item[0] for item in items]
        is_rect = "re" in item_types or item_types.count("l") >= 4
        if not is_rect:
            continue
        rects.append((x0, y0, x1, y1))
    return rects


def _merge_overlapping_bboxes(
    bboxes: List[tuple[float, float, float, float]],
    gap: float,
) -> List[tuple[float, float, float, float]]:
    """
    Merge bboxes that overlap or are within *gap* pixels of each other.
    Iterates until no further merges are possible.
    """
    if not bboxes:
        return []
    merged = list(bboxes)
    changed = True
    while changed:
        changed = False
        new_merged: List[tuple[float, float, float, float]] = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            ax0, ay0, ax1, ay1 = merged[i]
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                bx0, by0, bx1, by1 = merged[j]
                if (ax0 - gap <= bx1 and bx0 - gap <= ax1
                        and ay0 - gap <= by1 and by0 - gap <= ay1):
                    ax0 = min(ax0, bx0)
                    ay0 = min(ay0, by0)
                    ax1 = max(ax1, bx1)
                    ay1 = max(ay1, by1)
                    used[j] = True
                    changed = True
            new_merged.append((ax0, ay0, ax1, ay1))
            used[i] = True
        merged = new_merged
    return merged


def _build_exclude_bboxes(
    table_bboxes: List[tuple[float, float, float, float]],
    visual_rects: List[tuple[float, float, float, float]],
) -> List[tuple[float, float, float, float]]:
    """
    Combine table bboxes with visual rectangles that overlap tables.
    This captures the full visual extent of boxed notes even when
    find_tables() only detects part of the visual frame.
    """
    if not visual_rects:
        return table_bboxes
    if not table_bboxes:
        return table_bboxes

    extended: List[tuple[float, float, float, float]] = list(table_bboxes)
    for vr in visual_rects:
        for tb in table_bboxes:
            if _rects_overlap(tb, vr, 0.3) or _rects_overlap(vr, tb, 0.3):
                extended.append(vr)
                break
    return _merge_overlapping_bboxes(extended, _BBOX_MERGE_GAP)


def _extract_text_excluding_bboxes(
    page,
    exclude_bboxes: List[tuple[float, float, float, float]],
) -> str:
    """Extract page text, omitting text blocks that overlap with excluded bboxes."""
    if not exclude_bboxes:
        return page.get_text("text")
    blocks = page.get_text("blocks")
    kept: List[str] = []
    for block in blocks:
        if block[6] != 0:
            continue
        bbox: tuple[float, float, float, float] = (
            block[0], block[1], block[2], block[3],
        )
        if any(_rects_overlap(bbox, eb, 0.5) for eb in exclude_bboxes):
            continue
        kept.append(block[4])
    return "".join(kept)


def _merge_header_rows(rows: List[List[str]]) -> List[List[str]]:
    """Fusiona filas de encabezado multi-línea que no contienen datos numéricos."""
    if len(rows) <= 1:
        return rows
    merged = [str(c).strip() if c else "" for c in rows[0]]
    i = 1
    while i < len(rows):
        row = rows[i]
        if len(row) != len(merged):
            break
        if any(re.search(r"\d", str(c) or "") for c in row if c):
            break
        for j in range(len(merged)):
            base = merged[j]
            add = str(row[j]).strip() if row[j] else ""
            if base and add:
                merged[j] = base + " " + add
            elif add:
                merged[j] = add
        i += 1
    return [merged] + [[str(c).strip() if c else "" for c in r] for r in rows[i:]]


def _extract_tables_from_page(
    page,
    page_number: int,
    table_index_offset: int,
    visual_rects: List[tuple[float, float, float, float]] | None = None,
    *,
    relax_prose_table_filter: bool,
):
    """Extrae tablas de la página. Filtra prosa en grid; nunca filtra tarifarias. Alineado con assistax-fn."""
    from pipeline.boxed_note_classifier import (
        AmbiguousBlock,
        classify_and_route_block,
    )

    try:
        finder = page.find_tables()
    except Exception:
        return [], table_index_offset, []
    if not finder or not finder.tables:
        return [], table_index_offset, []
    tables: List[TableBlock] = []
    arbiter_log: List[dict] = []
    idx = table_index_offset
    vr_list = visual_rects or []

    for tab in finder.tables:
        try:
            rows_raw = tab.extract()
            if rows_raw is None:
                rows_raw = []
            rows: List[List[str]] = []
            for row in rows_raw:
                if row is None:
                    rows.append([])
                else:
                    cells = ["" if c is None else str(c).strip() for c in row]
                    rows.append(cells)

            rows = _merge_header_rows(rows)
            if _is_likely_prose_not_table(
                rows, relax_prose_table_filter=relax_prose_table_filter
            ):
                continue
            markdown = table_rows_to_markdown(rows, idx + 1)
            try:
                bbox = normalize_quad(tab.bbox)
            except (ValueError, TypeError, AttributeError):
                logger.debug(
                    "Skipping table on page %s: invalid bbox %r", page_number, tab.bbox
                )
                continue
            is_boxed = _is_editorial_boxed_note(rows)

            if not is_boxed:
                max_cols = max((len(r) for r in rows), default=0)
                non_empty = [r for r in rows if any(c.strip() for c in r)]
                inside_visual = any(
                    _rects_overlap(bbox, vr, 0.3) for vr in vr_list
                )
                if inside_visual and max_cols <= 2:
                    all_text = " ".join(
                        c.strip() for r in non_empty for c in r if c.strip()
                    )
                    amb = AmbiguousBlock(
                        text=all_text,
                        bbox=bbox,
                        page=page_number,
                        source="table_in_visual_rect",
                        nearby_text_before="",
                        nearby_text_after="",
                        is_inside_visual_box=True,
                        is_table_like=True,
                        rows_count=len(non_empty),
                        cols_count=max_cols,
                    )
                    kind, decision, reason = classify_and_route_block(amb)
                    arbiter_log.append({
                        "page": page_number,
                        "bbox": bbox,
                        "kind": kind,
                        "decision": decision,
                        "reason": reason,
                    })
                    if kind == "editorial_note":
                        is_boxed = True

            block = TableBlock(
                table_index=idx + 1,
                page_number=page_number,
                markdown=markdown,
                rows=rows,
                bbox=bbox,
                is_boxed_note=is_boxed,
            )
            tables.append(block)
            idx += 1
        except Exception as e:
            logger.debug(
                "Table extraction failed for table on page %s: %s", page_number, e
            )
            continue

    return tables, idx, arbiter_log


def _normalize_text(text: str) -> str:
    """Normaliza texto extraído: máx 2 newlines consecutivos, strip. Alineado con assistax-fn."""
    if not text:
        return ""
    normalized = re.sub(r"\n{3,}", "\n\n", text)
    return normalized.strip()


def extract_pdf(
    source: bytes,
    *,
    relax_prose_table_filter: Optional[bool] = None,
    relaxed_visual_frame_detection: Optional[bool] = None,
) -> tuple[List[PageContent], List[dict]]:
    """
    Extrae texto, tablas y TOC del PDF.
    Alineado con assistax-fn: get_text("text"), normalización interna, retorna (pages, toc).
    Per-run overrides: None → use settings.RELAX_*.
    """
    eff_relax_prose = (
        relax_prose_table_filter
        if relax_prose_table_filter is not None
        else settings.RELAX_PROSE_TABLE_FILTER
    )
    eff_relaxed_visual = (
        relaxed_visual_frame_detection
        if relaxed_visual_frame_detection is not None
        else settings.RELAXED_VISUAL_FRAME_DETECTION
    )
    try:
        doc = fitz.open(stream=source, filetype="pdf")
    except Exception as e:
        raise PDFExtractionError(f"PDF corrupto o inválido: {e}") from e

    toc: List[dict] = []
    try:
        raw = doc.get_toc(simple=True)
        for item in raw or []:  # noqa: SIM110
            if not item or len(item) < 3:
                continue
            level = int(item[0]) if item[0] is not None else 1
            title = str(item[1]).strip() if item[1] is not None else ""
            page_raw = item[2]
            page = int(page_raw) + 1 if page_raw is not None else 1
            if page < 1:
                page = 1
            toc.append({"level": level, "title": title, "page": page})
    except Exception:
        pass

    pages: List[PageContent] = []
    global_table_index = 0
    all_arbiter_logs: List[dict] = []

    do_normalize = settings.ENABLE_PDF_TEXT_NORMALIZATION
    seen_headers: set[str] = set()

    try:
        for i in range(len(doc)):
            page = doc[i]
            page_number = i + 1

            visual_rects = _extract_visual_rects(
                page, relaxed_visual_frame_detection=eff_relaxed_visual
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "extract_pdf page=%s visual_rects=%d relaxed_visual=%s",
                    page_number,
                    len(visual_rects),
                    eff_relaxed_visual,
                )
            tables, global_table_index, arbiter_log = _extract_tables_from_page(
                page,
                page_number,
                global_table_index,
                visual_rects,
                relax_prose_table_filter=eff_relax_prose,
            )
            if arbiter_log:
                all_arbiter_logs.extend(arbiter_log)

            table_bboxes = [t.bbox for t in tables]
            exclude_bboxes = _build_exclude_bboxes(table_bboxes, visual_rects)
            
            # --- Rescate heurístico de notas editoriales que PyMuPDF no vio como tabla/dibujo ---
            blocks = page.get_text("blocks")
            for block in blocks:
                # blocks = (x0, y0, x1, y1, "lines in block", block_no, block_type)
                # block_type 0 = texto
                if block[6] != 0:
                    continue
                bbox = (block[0], block[1], block[2], block[3])
                if any(_rects_overlap(bbox, eb, 0.5) for eb in exclude_bboxes):
                    continue
                    
                text_b = block[4].strip()
                # Considerar como boxed_note si coincide con prefijos editoriales fuertes
                if _EDITORIAL_NOTE_PREFIXES.search(text_b):
                    tb = TableBlock(
                        table_index=global_table_index + 1,
                        page_number=page_number,
                        markdown=f"[TABLE_{global_table_index + 1}]\n| Col 1 |\n|---|\n| {sanitize_cell(text_b)} |\n[/TABLE_{global_table_index + 1}]",
                        rows=[[text_b]],
                        bbox=bbox,
                        is_boxed_note=True
                    )
                    tables.append(tb)
                    exclude_bboxes.append(bbox)
                    global_table_index += 1
            
            text = _extract_text_excluding_bboxes(page, exclude_bboxes)
            text = _normalize_text(text)

            if do_normalize:
                try:
                    from pipeline.pdf_text_normalization import normalize_pdf_text
                    text, _ = normalize_pdf_text(
                        text,
                        seen_headers,
                        settings.ENABLE_DECRETO_PROSE_BLANK_COLLAPSE,
                    )
                except Exception:
                    pass

            pages.append(
                PageContent(
                    page_number=page_number,
                    text=text,
                    tables=tables,
                )
            )
    except Exception as e:
        pn = i + 1 if "i" in dir() else "?"
        raise PDFExtractionError(f"Error al extraer página {pn}: {e}") from e
    finally:
        doc.close()

    if all_arbiter_logs:
        logger.info(
            "extract_pdf.arbiter_summary: %d blocks arbitrated",
            len(all_arbiter_logs),
        )

    return pages, toc


def extract_toc(source: bytes) -> List[dict]:
    """
    Extrae TOC del PDF.
    Retorna lista de {level, title, page} con page 1-indexed.
    """
    try:
        doc = fitz.open(stream=source, filetype="pdf")
    except Exception as e:
        raise PDFExtractionError(f"PDF corrupto o inválido: {e}") from e
    try:
        raw = doc.get_toc(simple=True)
        return [
            {"level": item[0], "title": item[1], "page": item[2] + 1}
            for item in raw
        ]
    finally:
        doc.close()
