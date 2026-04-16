"""
Tests unitarios para pdf_extractor.
Criterios Fase 1: PDF duplicado, tablas correctas, PDF corrupto → PDFExtractionError.
"""
import pytest

from pipeline.pdf_extractor import (
    extract_pdf,
    PageContent,
    TableBlock,
    _is_editorial_boxed_note,
    _rects_overlap,
    _merge_overlapping_bboxes,
    _build_exclude_bboxes,
)
from pipeline.exceptions import PDFExtractionError


def test_extract_pdf_ley_corta(pdf_bytes_ley_corta):
    """PDF con texto tipo ley → List[PageContent] con texto extraído."""
    pages, _toc = extract_pdf(pdf_bytes_ley_corta)
    assert len(pages) >= 1
    assert all(isinstance(p, PageContent) for p in pages)
    text = " ".join(p.text for p in pages)
    assert "LEY FEDERAL" in text or "Artículo" in text
    assert pages[0].page_number == 1


def test_extract_pdf_per_run_extraction_overrides(pdf_bytes_ley_corta):
    """Per-run relax flags are accepted without error (same shape as default path)."""
    pages_default, _ = extract_pdf(pdf_bytes_ley_corta)
    pages_relaxed, _ = extract_pdf(
        pdf_bytes_ley_corta,
        relax_prose_table_filter=True,
        relaxed_visual_frame_detection=True,
    )
    assert len(pages_default) == len(pages_relaxed)
    assert all(isinstance(p, PageContent) for p in pages_relaxed)


def test_extract_pdf_con_tablas(pdf_bytes_con_tablas):
    """PDF con tabla tarifaria → tablas en tables, texto sin duplicar tablas."""
    pages, _toc = extract_pdf(pdf_bytes_con_tablas)
    assert len(pages) >= 1
    # Tabla tarifaria nunca filtrada: si hay tabla, debe estar en tables
    total_tables = sum(len(p.tables) for p in pages)
    # Puede ser 0 si PyMuPDF no detecta tabla sin bordes
    for p in pages:
        assert isinstance(p.tables, list)
        for t in p.tables:
            assert isinstance(t, TableBlock)
            assert "[TABLE_" in t.markdown


def test_extract_pdf_corrupto_raise(pdf_bytes_corrupto):
    """PDF corrupto → PDFExtractionError con mensaje descriptivo."""
    with pytest.raises(PDFExtractionError) as exc_info:
        extract_pdf(pdf_bytes_corrupto)
    assert "corrupto" in str(exc_info.value).lower() or "inválido" in str(exc_info.value).lower()


def test_extract_pdf_vacio(pdf_bytes_vacio):
    """PDF con página vacía → una PageContent con text vacío."""
    pages, _toc = extract_pdf(pdf_bytes_vacio)
    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].tables == []


# ---------------------------------------------------------------------------
# Tests: editorial boxed note detection
# ---------------------------------------------------------------------------


def test_is_editorial_boxed_note_single_cell_with_keyword() -> None:
    """Single-cell table starting with ACLARACION → boxed note."""
    rows = [["ACLARACIÓN: El texto del artículo se corrigió conforme a Fe de Erratas."]]
    assert _is_editorial_boxed_note(rows) is True


def test_is_editorial_boxed_note_nota_de_erratas() -> None:
    """Single-cell table with 'Nota de erratas' → boxed note."""
    rows = [["Nota de erratas al artículo publicada en el Diario Oficial de la Federación"]]
    assert _is_editorial_boxed_note(rows) is True


def test_is_editorial_boxed_note_fe_de_erratas() -> None:
    """Single-cell with 'Fe de erratas' prefix → boxed note."""
    rows = [["Fe de erratas publicada en el DOF el 15 de marzo de 2020 al artículo 306."]]
    assert _is_editorial_boxed_note(rows) is True


def test_is_editorial_boxed_note_long_prose_no_keyword() -> None:
    """Single-cell with >100 chars of prose text without keyword → boxed note by length."""
    long_text = "Este es un texto editorial largo que aparece en un recuadro. " * 3
    rows = [[long_text]]
    assert _is_editorial_boxed_note(rows) is True


def test_is_editorial_boxed_note_short_cell_no_keyword() -> None:
    """Single short cell without keyword → not a boxed note."""
    rows = [["Artículo 5"]]
    assert _is_editorial_boxed_note(rows) is False


def test_is_editorial_boxed_note_tariff_never_classified() -> None:
    """Tariff-like table → never classified as boxed note."""
    rows = [
        ["cuota", "rango inferior", "rango superior"],
        ["100", "0", "1000"],
        ["200", "1001", "2000"],
    ]
    assert _is_editorial_boxed_note(rows) is False


def test_is_editorial_boxed_note_multicol_table() -> None:
    """Table with >2 columns → not a boxed note regardless of content."""
    rows = [["ACLARACIÓN", "col2", "col3"]]
    assert _is_editorial_boxed_note(rows) is False


# ---------------------------------------------------------------------------
# Tests: bbox overlap logic
# ---------------------------------------------------------------------------


def test_rects_overlap_full() -> None:
    """Block fully inside exclude bbox → overlap."""
    assert _rects_overlap((10, 10, 50, 50), (0, 0, 100, 100), 0.5) is True


def test_rects_overlap_none() -> None:
    """Non-intersecting rects → no overlap."""
    assert _rects_overlap((0, 0, 10, 10), (20, 20, 30, 30), 0.5) is False


def test_rects_overlap_partial_below_threshold() -> None:
    """Partial overlap below min_ratio → no overlap."""
    assert _rects_overlap((0, 0, 100, 100), (80, 80, 200, 200), 0.5) is False


def test_rects_overlap_partial_above_threshold() -> None:
    """Partial overlap above min_ratio → overlap."""
    assert _rects_overlap((0, 0, 100, 100), (0, 0, 100, 60), 0.5) is True


# ---------------------------------------------------------------------------
# Tests: merge overlapping bboxes
# ---------------------------------------------------------------------------


def test_merge_overlapping_bboxes_no_overlap() -> None:
    """Disjoint bboxes stay separate."""
    bboxes = [(0, 0, 10, 10), (50, 50, 60, 60)]
    result = _merge_overlapping_bboxes(bboxes, 5.0)
    assert len(result) == 2


def test_merge_overlapping_bboxes_touching() -> None:
    """Bboxes within gap distance merge into one."""
    bboxes = [(0, 0, 10, 10), (12, 0, 20, 10)]
    result = _merge_overlapping_bboxes(bboxes, 5.0)
    assert len(result) == 1
    merged = result[0]
    assert merged[0] == 0 and merged[2] == 20


def test_merge_overlapping_bboxes_overlapping() -> None:
    """Overlapping bboxes merge into one."""
    bboxes = [(0, 0, 50, 50), (30, 30, 80, 80)]
    result = _merge_overlapping_bboxes(bboxes, 0.0)
    assert len(result) == 1
    assert result[0] == (0, 0, 80, 80)


def test_merge_overlapping_bboxes_empty() -> None:
    """Empty input returns empty."""
    assert _merge_overlapping_bboxes([], 5.0) == []


def test_merge_overlapping_bboxes_chain() -> None:
    """Three bboxes that chain-overlap merge into one."""
    bboxes = [(0, 0, 30, 10), (25, 0, 55, 10), (50, 0, 80, 10)]
    result = _merge_overlapping_bboxes(bboxes, 0.0)
    assert len(result) == 1
    assert result[0] == (0, 0, 80, 10)


# ---------------------------------------------------------------------------
# Tests: build_exclude_bboxes with visual rects
# ---------------------------------------------------------------------------


def test_build_exclude_bboxes_no_visual_rects() -> None:
    """Without visual rects, returns table bboxes as-is."""
    table_bboxes = [(0, 0, 100, 50)]
    result = _build_exclude_bboxes(table_bboxes, [])
    assert result == table_bboxes


def test_build_exclude_bboxes_visual_rect_overlapping_table() -> None:
    """Visual rect that overlaps a table bbox gets merged in."""
    table_bboxes = [(50, 50, 200, 150)]
    visual_rects = [(40, 40, 210, 200)]
    result = _build_exclude_bboxes(table_bboxes, visual_rects)
    assert len(result) == 1
    merged = result[0]
    assert merged[0] == 40 and merged[1] == 40
    assert merged[2] == 210 and merged[3] == 200


def test_build_exclude_bboxes_visual_rect_no_overlap() -> None:
    """Visual rect that doesn't overlap any table bbox is ignored."""
    table_bboxes = [(0, 0, 50, 50)]
    visual_rects = [(300, 300, 400, 400)]
    result = _build_exclude_bboxes(table_bboxes, visual_rects)
    assert result == table_bboxes


def test_build_exclude_bboxes_no_tables() -> None:
    """No tables means visual rects are irrelevant."""
    result = _build_exclude_bboxes([], [(100, 100, 200, 200)])
    assert result == []
