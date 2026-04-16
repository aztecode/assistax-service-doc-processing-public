"""
Unit tests for layout_extractor_v2 — Phase 1 acceptance criteria + closure hardening.

Covers:
- DocumentLayout returned for a simple PDF
- All blocks have block_id, page_number, bbox, text
- reading_order is assigned to all blocks
- Native TOC is preserved when present
- Empty TOC returned without failure when absent
- Full nested JSON serialization (pages, blocks, spans, metadata)
- _compute_reading_order with vertical misalignment and two-column zones
- Candidate table detection: no-table and grid cases
"""
import json

import fitz
import pytest

from pipeline.exceptions import PDFExtractionError
from pipeline.layout_extractor_v2 import (
    _compute_reading_order,
    _extract_candidate_tables_from_page,
    _extract_text_blocks_from_page,
    _extract_visual_frames_from_page,
    extract_document_layout,
)
from pipeline.layout_models import DocumentLayout, LayoutBlock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pdf_bytes_simple() -> bytes:
    """Single-page PDF with legal-style text."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "LEY FEDERAL DEL TRABAJO")
    page.insert_text((72, 100), "Artículo 1. La presente Ley es de observancia general.")
    page.insert_text((72, 120), "Artículo 2. Las normas de trabajo buscan el equilibrio.")
    page.insert_text((72, 140), "Capítulo I - Disposiciones generales")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def pdf_bytes_with_native_toc() -> bytes:
    """PDF with a native embedded TOC."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "REGLAMENTO DE PRUEBA")
    page.insert_text((72, 100), "Artículo 1. Disposición general.")
    doc.set_toc([
        [1, "REGLAMENTO DE PRUEBA", 1],
        [2, "Artículo 1", 1],
    ])
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def pdf_bytes_empty_page() -> bytes:
    """PDF with a single blank page."""
    doc = fitz.open()
    doc.new_page()
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def pdf_bytes_multipage() -> bytes:
    """3-page PDF each with legal text."""
    doc = fitz.open()
    for i in range(1, 4):
        page = doc.new_page()
        page.insert_text((72, 72), f"Página {i}")
        page.insert_text((72, 100), f"Artículo {i}. Texto del artículo número {i}.")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def pdf_bytes_corrupto() -> bytes:
    """Invalid bytes that are not a PDF."""
    return b"No soy un PDF, solo basura \x00\x01\x02"


@pytest.fixture
def pdf_bytes_with_grid() -> bytes:
    """Single-page PDF with a 2x2 drawn grid (outer rect + inner lines).

    Explicit line drawing is the most reliable way to trigger PyMuPDF's
    find_tables() detector in programmatic PDFs.
    """
    doc = fitz.open()
    page = doc.new_page()
    # Outer border of the 2-column, 2-row table
    page.draw_rect(fitz.Rect(50, 100, 250, 180), color=(0, 0, 0), width=1)
    # Vertical divider (column split at x=150)
    page.draw_line((150, 100), (150, 180), color=(0, 0, 0), width=1)
    # Horizontal divider (row split at y=140)
    page.draw_line((50, 140), (250, 140), color=(0, 0, 0), width=1)
    page.insert_text((60, 125), "Concepto")
    page.insert_text((160, 125), "Valor")
    page.insert_text((60, 162), "Artículo 1")
    page.insert_text((160, 162), "100%")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# ---------------------------------------------------------------------------
# DocumentLayout shape
# ---------------------------------------------------------------------------


def test_returns_document_layout_instance(pdf_bytes_simple: bytes) -> None:
    """extract_document_layout returns a DocumentLayout for a valid PDF."""
    result = extract_document_layout(pdf_bytes_simple)
    assert isinstance(result, DocumentLayout)


def test_document_layout_has_pages(pdf_bytes_simple: bytes) -> None:
    """DocumentLayout has at least one page for a non-empty PDF."""
    result = extract_document_layout(pdf_bytes_simple)
    assert len(result.pages) >= 1


def test_document_layout_not_empty(pdf_bytes_simple: bytes) -> None:
    """At least one block is extracted from a PDF with text content."""
    result = extract_document_layout(pdf_bytes_simple)
    total_blocks = sum(len(p.blocks) for p in result.pages)
    assert total_blocks >= 1


# ---------------------------------------------------------------------------
# Required block fields
# ---------------------------------------------------------------------------


def test_all_blocks_have_block_id(pdf_bytes_simple: bytes) -> None:
    """Every LayoutBlock has a non-empty block_id."""
    result = extract_document_layout(pdf_bytes_simple)
    for page in result.pages:
        for block in page.blocks:
            assert block.block_id, f"Missing block_id on page {page.page_number}"


def test_block_ids_follow_stable_format(pdf_bytes_simple: bytes) -> None:
    """block_id follows the p{page}_b{index} convention (stable across runs)."""
    result = extract_document_layout(pdf_bytes_simple)
    for page in result.pages:
        for block in page.blocks:
            assert block.block_id.startswith(f"p{page.page_number}_b"), (
                f"Unexpected block_id format: {block.block_id}"
            )


def test_all_blocks_have_valid_page_number(pdf_bytes_simple: bytes) -> None:
    """Every LayoutBlock has a page_number >= 1."""
    result = extract_document_layout(pdf_bytes_simple)
    for page in result.pages:
        for block in page.blocks:
            assert block.page_number >= 1, f"Invalid page_number: {block.page_number}"


def test_all_blocks_have_four_element_bbox(pdf_bytes_simple: bytes) -> None:
    """Every LayoutBlock bbox is a 4-tuple of floats."""
    result = extract_document_layout(pdf_bytes_simple)
    for page in result.pages:
        for block in page.blocks:
            assert len(block.bbox) == 4, f"Malformed bbox in {block.block_id}: {block.bbox}"
            assert all(isinstance(v, float) for v in block.bbox), (
                f"Non-float in bbox of {block.block_id}"
            )


def test_all_blocks_have_non_empty_text(pdf_bytes_simple: bytes) -> None:
    """Every LayoutBlock has non-empty text (whitespace-only blocks are discarded)."""
    result = extract_document_layout(pdf_bytes_simple)
    for page in result.pages:
        for block in page.blocks:
            assert block.text.strip(), f"Empty text in block {block.block_id}"


# ---------------------------------------------------------------------------
# reading_order
# ---------------------------------------------------------------------------


def test_reading_order_assigned_to_all_blocks(pdf_bytes_simple: bytes) -> None:
    """Every LayoutBlock has reading_order >= 0."""
    result = extract_document_layout(pdf_bytes_simple)
    for page in result.pages:
        for block in page.blocks:
            assert block.reading_order >= 0, (
                f"Unset reading_order in block {block.block_id}"
            )


def test_reading_order_unique_per_page(pdf_bytes_simple: bytes) -> None:
    """reading_order values are unique within a page."""
    result = extract_document_layout(pdf_bytes_simple)
    for page in result.pages:
        orders = [b.reading_order for b in page.blocks]
        assert len(orders) == len(set(orders)), (
            f"Duplicate reading_order on page {page.page_number}: {orders}"
        )


def test_reading_order_starts_at_zero_per_page(pdf_bytes_multipage: bytes) -> None:
    """reading_order is 0-based and sequential per page."""
    result = extract_document_layout(pdf_bytes_multipage)
    for page in result.pages:
        if not page.blocks:
            continue
        orders = sorted(b.reading_order for b in page.blocks)
        assert orders[0] == 0, (
            f"reading_order doesn't start at 0 on page {page.page_number}"
        )
        assert orders == list(range(len(orders))), (
            f"Non-sequential reading_order on page {page.page_number}: {orders}"
        )


# ---------------------------------------------------------------------------
# Native TOC
# ---------------------------------------------------------------------------


def test_native_toc_preserved_when_present(pdf_bytes_with_native_toc: bytes) -> None:
    """native_toc is populated when the PDF has an embedded TOC."""
    result = extract_document_layout(pdf_bytes_with_native_toc)
    assert len(result.native_toc) >= 1, "Expected non-empty native_toc"
    entry = result.native_toc[0]
    assert "level" in entry
    assert "title" in entry
    assert "page" in entry


def test_native_toc_entry_values_are_typed(pdf_bytes_with_native_toc: bytes) -> None:
    """Native TOC entries have correctly typed level (int), title (str), page (int)."""
    result = extract_document_layout(pdf_bytes_with_native_toc)
    for entry in result.native_toc:
        assert isinstance(entry["level"], int), f"level not int: {entry}"
        assert isinstance(entry["title"], str), f"title not str: {entry}"
        assert isinstance(entry["page"], int), f"page not int: {entry}"
        assert entry["page"] >= 1, f"page number < 1: {entry}"


def test_native_toc_empty_when_absent(pdf_bytes_simple: bytes) -> None:
    """native_toc is an empty list for PDFs without embedded TOC (no error)."""
    result = extract_document_layout(pdf_bytes_simple)
    assert isinstance(result.native_toc, list)
    assert result.native_toc == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_page_produces_no_blocks(pdf_bytes_empty_page: bytes) -> None:
    """A page with no text yields zero blocks without raising."""
    result = extract_document_layout(pdf_bytes_empty_page)
    assert len(result.pages) == 1
    assert result.pages[0].blocks == []


def test_empty_page_page_number_is_one(pdf_bytes_empty_page: bytes) -> None:
    """An empty single-page PDF has page_number=1."""
    result = extract_document_layout(pdf_bytes_empty_page)
    assert result.pages[0].page_number == 1


def test_corrupt_pdf_raises_extraction_error(pdf_bytes_corrupto: bytes) -> None:
    """Corrupt bytes raise PDFExtractionError with a descriptive message."""
    with pytest.raises(PDFExtractionError) as exc_info:
        extract_document_layout(pdf_bytes_corrupto)
    msg = str(exc_info.value).lower()
    assert "corrupto" in msg or "inválido" in msg or "invalid" in msg


# ---------------------------------------------------------------------------
# Multipage
# ---------------------------------------------------------------------------


def test_multipage_page_count(pdf_bytes_multipage: bytes) -> None:
    """Three-page PDF produces exactly 3 PageLayout entries."""
    result = extract_document_layout(pdf_bytes_multipage)
    assert len(result.pages) == 3


def test_multipage_page_numbers_sequential(pdf_bytes_multipage: bytes) -> None:
    """page_number is 1-indexed and sequential across pages."""
    result = extract_document_layout(pdf_bytes_multipage)
    for expected, page in enumerate(result.pages, start=1):
        assert page.page_number == expected, (
            f"Expected page_number={expected}, got {page.page_number}"
        )


def test_multipage_block_page_numbers_match_page(pdf_bytes_multipage: bytes) -> None:
    """Every block's page_number matches the PageLayout it belongs to."""
    result = extract_document_layout(pdf_bytes_multipage)
    for page in result.pages:
        for block in page.blocks:
            assert block.page_number == page.page_number, (
                f"Block {block.block_id} has page_number={block.page_number} "
                f"but lives in PageLayout page_number={page.page_number}"
            )


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


def test_document_layout_is_json_serializable(pdf_bytes_simple: bytes) -> None:
    """DocumentLayout serializes to valid JSON and round-trips without error."""
    result = extract_document_layout(pdf_bytes_simple)
    json_str = result.model_dump_json()
    parsed = json.loads(json_str)
    assert "pages" in parsed
    assert "native_toc" in parsed
    assert "metadata" in parsed


def test_json_metadata_contains_page_count(pdf_bytes_multipage: bytes) -> None:
    """DocumentLayout.metadata includes total_pages matching actual page count."""
    result = extract_document_layout(pdf_bytes_multipage)
    assert result.metadata["total_pages"] == 3


# ---------------------------------------------------------------------------
# Helper unit tests (internals)
# ---------------------------------------------------------------------------


def test_compute_reading_order_assigns_zero_start() -> None:
    """_compute_reading_order returns blocks with reading_order starting at 0."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Bloque A")
    page.insert_text((72, 200), "Bloque B")
    raw = _extract_text_blocks_from_page(page, 1)
    doc.close()
    ordered = _compute_reading_order(raw, 595.0, 842.0)
    orders = [b.reading_order for b in ordered]
    assert 0 in orders
    assert sorted(orders) == list(range(len(orders)))


def test_compute_reading_order_top_before_bottom() -> None:
    """Blocks higher on the page get lower reading_order."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 50), "Bloque superior")
    page.insert_text((72, 700), "Bloque inferior")
    raw = _extract_text_blocks_from_page(page, 1)
    doc.close()
    ordered = _compute_reading_order(raw, 595.0, 842.0)
    if len(ordered) >= 2:
        top_block = min(ordered, key=lambda b: b.bbox[1])
        bottom_block = max(ordered, key=lambda b: b.bbox[1])
        assert top_block.reading_order < bottom_block.reading_order


def test_compute_reading_order_empty_input() -> None:
    """_compute_reading_order returns empty list for empty input."""
    result = _compute_reading_order([], 595.0, 842.0)
    assert result == []


def test_extract_text_blocks_discards_whitespace_only(pdf_bytes_empty_page: bytes) -> None:
    """_extract_text_blocks_from_page produces no blocks for an empty page."""
    doc = fitz.open(stream=pdf_bytes_empty_page, filetype="pdf")
    page = doc[0]
    blocks = _extract_text_blocks_from_page(page, 1)
    doc.close()
    assert blocks == []


def test_layout_block_model_copy_preserves_fields() -> None:
    """model_copy(update=...) on LayoutBlock only changes the specified field."""
    block = LayoutBlock(
        block_id="p1_b0",
        page_number=1,
        bbox=(0.0, 0.0, 100.0, 20.0),
        text="Artículo 1.",
        kind="text",
        reading_order=0,
        spans=[],
        source="pymupdf_text",
        metadata={},
    )
    updated = block.model_copy(update={"reading_order": 5})
    assert updated.reading_order == 5
    assert updated.block_id == block.block_id
    assert updated.text == block.text


# ---------------------------------------------------------------------------
# Harder reading_order tests
# ---------------------------------------------------------------------------


def _make_block(block_id: str, x0: float, y0: float, text: str = "texto") -> LayoutBlock:
    """Helper: build a minimal LayoutBlock at the given position."""
    return LayoutBlock(
        block_id=block_id,
        page_number=1,
        bbox=(x0, y0, x0 + 150.0, y0 + 12.0),
        text=text,
        kind="text",
        reading_order=0,
        spans=[],
        source="pymupdf_text",
        metadata={},
    )


def test_reading_order_misaligned_blocks_in_same_band() -> None:
    """Blocks within the Y_BAND_PTS=5 tolerance are grouped and sorted left-to-right.

    A 3pt vertical offset between two blocks that are visually on the same line
    must not cause the right-hand block to be ordered before the left-hand one.
    """
    # y=100 and y=103 both quantize to band 100 → same logical line
    block_left = _make_block("left", x0=50.0, y0=100.0, text="Encabezado izquierdo")
    block_right = _make_block("right", x0=300.0, y0=103.0, text="Encabezado derecho")

    ordered = _compute_reading_order([block_right, block_left], 595.0, 842.0)

    assert len(ordered) == 2
    assert ordered[0].block_id == "left", (
        "Left block (x=50) must precede right block (x=300) within the same Y band"
    )
    assert ordered[1].block_id == "right"


def test_reading_order_misalignment_exceeding_band_creates_new_row() -> None:
    """Blocks more than _Y_BAND_PTS apart are placed in separate rows.

    y=100 → band 100; y=106 → band 105. These are distinct bands, so the
    block at y=106 is always ordered after the block at y=100 regardless of X.
    """
    block_top_right = _make_block("top_right", x0=400.0, y0=100.0)
    block_bottom_left = _make_block("bottom_left", x0=50.0, y0=106.0)

    ordered = _compute_reading_order([block_bottom_left, block_top_right], 595.0, 842.0)

    assert ordered[0].block_id == "top_right", (
        "Block at y=100 (band=100) must precede block at y=106 (band=105)"
    )
    assert ordered[1].block_id == "bottom_left"


def test_reading_order_two_column_zone_interleaves_by_y() -> None:
    """Documents the current two-column behavior: columns are interleaved by Y-band.

    The present algorithm (Y-band → X) does not detect column boundaries.
    For a two-column layout where both columns share the same Y positions,
    the result correctly interleaves left/right within each row.

    NOTE: True column-first ordering (left column fully before right column)
    requires column boundary detection, which is reserved for a future phase
    and will leverage the page_width parameter already in the function signature.
    """
    positions: list[tuple[str, float, float]] = [
        ("left_100", 50.0, 100.0),
        ("right_100", 350.0, 100.0),
        ("left_200", 50.0, 200.0),
        ("right_200", 350.0, 200.0),
        ("left_300", 50.0, 300.0),
        ("right_300", 350.0, 300.0),
    ]
    blocks = [_make_block(bid, x, y) for bid, x, y in positions]
    # Shuffle input to confirm output is deterministic
    import random
    random.shuffle(blocks)

    ordered = _compute_reading_order(blocks, 595.0, 842.0)
    ids = [b.block_id for b in ordered]

    # Within each column, top-to-bottom order must be preserved
    assert ids.index("left_100") < ids.index("left_200") < ids.index("left_300"), (
        "Left column must read top-to-bottom"
    )
    assert ids.index("right_100") < ids.index("right_200") < ids.index("right_300"), (
        "Right column must read top-to-bottom"
    )
    # Within the same Y-band, left column comes before right column
    assert ids.index("left_100") < ids.index("right_100"), (
        "Left block precedes right block within same Y band"
    )


def test_reading_order_deterministic_regardless_of_input_order() -> None:
    """reading_order output is stable regardless of the order blocks arrive in."""
    forward = [_make_block(f"p1_b{i}", x0=72.0, y0=float(100 + i * 50)) for i in range(5)]
    backward = list(reversed(forward))

    result_fwd = [b.block_id for b in _compute_reading_order(forward, 595.0, 842.0)]
    result_bwd = [b.block_id for b in _compute_reading_order(backward, 595.0, 842.0)]

    assert result_fwd == result_bwd, (
        "reading_order must be deterministic regardless of input list order"
    )


def test_reading_order_single_block_gets_order_zero() -> None:
    """A single block receives reading_order=0."""
    block = _make_block("p1_b0", x0=72.0, y0=100.0)
    result = _compute_reading_order([block], 595.0, 842.0)
    assert len(result) == 1
    assert result[0].reading_order == 0


# ---------------------------------------------------------------------------
# Full nested JSON serialization
# ---------------------------------------------------------------------------


def test_full_json_serialization_nested_structure(pdf_bytes_simple: bytes) -> None:
    """model_dump_json produces fully nested JSON including pages, blocks and spans."""
    result = extract_document_layout(pdf_bytes_simple)
    parsed: dict = json.loads(result.model_dump_json())

    # Top-level keys
    assert set(parsed.keys()) >= {"pages", "native_toc", "metadata"}

    # Page-level shape
    assert len(parsed["pages"]) >= 1
    page = parsed["pages"][0]
    assert set(page.keys()) >= {"page_number", "width", "height", "blocks", "raw_tables", "raw_drawings"}
    assert isinstance(page["page_number"], int)
    assert isinstance(page["width"], float)
    assert isinstance(page["height"], float)

    # Block-level shape (requires at least one block on page 1)
    assert len(page["blocks"]) >= 1, "Expected at least one text block on the first page"
    block = page["blocks"][0]
    assert set(block.keys()) >= {
        "block_id", "page_number", "bbox", "text", "kind", "reading_order", "spans",
        "source", "metadata",
    }
    assert isinstance(block["bbox"], list) and len(block["bbox"]) == 4
    assert isinstance(block["text"], str) and block["text"].strip()
    assert isinstance(block["reading_order"], int)

    # Span-level shape (insert_text always produces at least one span)
    assert len(block["spans"]) >= 1, "Expected spans inside the first block"
    span = block["spans"][0]
    assert set(span.keys()) >= {"text", "bbox", "page_number", "font_size", "font_name"}
    assert isinstance(span["text"], str) and span["text"].strip()
    assert isinstance(span["bbox"], list) and len(span["bbox"]) == 4


def test_full_json_serialization_toc_when_present(pdf_bytes_with_native_toc: bytes) -> None:
    """native_toc entries serialize correctly with typed fields."""
    result = extract_document_layout(pdf_bytes_with_native_toc)
    parsed: dict = json.loads(result.model_dump_json())

    assert len(parsed["native_toc"]) >= 1
    entry = parsed["native_toc"][0]
    assert isinstance(entry["level"], int)
    assert isinstance(entry["title"], str)
    assert isinstance(entry["page"], int)


def test_full_json_serialization_metadata_fields(pdf_bytes_multipage: bytes) -> None:
    """metadata at document level serializes total_pages and total_blocks as ints."""
    result = extract_document_layout(pdf_bytes_multipage)
    parsed: dict = json.loads(result.model_dump_json())

    meta = parsed["metadata"]
    assert "total_pages" in meta and meta["total_pages"] == 3
    assert "total_blocks" in meta and isinstance(meta["total_blocks"], int)
    assert meta["total_blocks"] >= 3  # At least one block per page


def test_full_json_round_trip_preserves_block_ids(pdf_bytes_simple: bytes) -> None:
    """block_ids survive a full model_dump → JSON → re-parse round-trip."""
    result = extract_document_layout(pdf_bytes_simple)
    original_ids = {b.block_id for p in result.pages for b in p.blocks}

    parsed = json.loads(result.model_dump_json())
    serialized_ids = {b["block_id"] for p in parsed["pages"] for b in p["blocks"]}

    assert original_ids == serialized_ids


# ---------------------------------------------------------------------------
# Candidate tables: no-table and grid cases
# ---------------------------------------------------------------------------


def test_candidate_tables_text_only_page_returns_empty(pdf_bytes_simple: bytes) -> None:
    """A page with unstructured text (no grid borders) yields no table candidates."""
    doc = fitz.open(stream=pdf_bytes_simple, filetype="pdf")
    candidates = _extract_candidate_tables_from_page(doc[0], 1)
    doc.close()
    assert isinstance(candidates, list)
    assert len(candidates) == 0, (
        "Plain prose paragraphs must not trigger find_tables() candidate detection"
    )


def test_candidate_tables_returns_list_type(pdf_bytes_empty_page: bytes) -> None:
    """_extract_candidate_tables_from_page always returns list[dict], never raises."""
    doc = fitz.open(stream=pdf_bytes_empty_page, filetype="pdf")
    candidates = _extract_candidate_tables_from_page(doc[0], 1)
    doc.close()
    assert isinstance(candidates, list)


def test_candidate_tables_grid_has_correct_shape(pdf_bytes_with_grid: bytes) -> None:
    """A page with a drawn 2x2 grid returns candidates with valid shape fields.

    If PyMuPDF detects the grid, every candidate must carry bbox, row_count,
    col_count and page_number. If detection yields 0 results (which can happen
    with certain PDF viewers/renderers), the test still passes — the critical
    invariant is: no crash, correct return type, valid field schema on any entry.
    """
    doc = fitz.open(stream=pdf_bytes_with_grid, filetype="pdf")
    candidates = _extract_candidate_tables_from_page(doc[0], 1)
    doc.close()

    assert isinstance(candidates, list)
    for c in candidates:
        assert "bbox" in c, "Candidate must have bbox"
        assert len(c["bbox"]) == 4, "bbox must be a 4-tuple"
        assert all(isinstance(v, float) for v in c["bbox"]), "bbox values must be floats"
        assert "row_count" in c and isinstance(c["row_count"], int)
        assert "col_count" in c and isinstance(c["col_count"], int)
        assert "page_number" in c and c["page_number"] == 1
        assert "table_index" in c and isinstance(c["table_index"], int)


def test_candidate_tables_grid_is_detected(pdf_bytes_with_grid: bytes) -> None:
    """The drawn 2x2 grid must be detected as at least one table candidate.

    PyMuPDF's find_tables() reliably detects explicit rect+line grids.
    If this test fails it means the fixture geometry is insufficient for
    detection and the fixture should be adjusted before Phase 2.
    """
    doc = fitz.open(stream=pdf_bytes_with_grid, filetype="pdf")
    candidates = _extract_candidate_tables_from_page(doc[0], 1)
    doc.close()
    assert len(candidates) >= 1, (
        "Expected at least 1 candidate from explicit 2x2 drawn grid. "
        "If PyMuPDF changed detection behavior, verify fixture geometry."
    )
