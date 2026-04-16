"""
Unit tests for layout_normalizer_v2 — Phase 2 acceptance + closure hardening.

Covers:
 1. Header repeated across pages → marked as kind="header"
 2. Footer repeated across pages → marked as kind="footer"
 3. Slight numbering/date variations don't prevent header/footer detection
 4. Unique block on a single page is NOT falsely marked header/footer
 5. Index-like blocks with many article/chapter references → possible_index_zone=True
 6. Normal legal prose is NOT marked as index zone
 7. Two contiguous paragraph blocks are merged
 8. Merge blocked when right block looks like a legal heading
 9. Merge blocked when either block is marked header/footer
10. Merge blocked when a candidate table sits between blocks
11. Merged block preserves traceability (source_block_ids, merged_from)
12. After merging, reading_order is consistent (sequential, 0-based)
13. Empty pages and tiny documents don't crash
14. Output is serializable with Pydantic v2 (model_dump_json round-trip)

Closure hardening (Ajustes 1-3):
15. 6-page doc, signature in 2 pages → NOT marked (ceil threshold)
16. 6-page doc, signature in 3 pages → marked (ceil threshold)
17. 2-page doc, signature in 2 pages → marked (minimum-pages rule)
18. Uppercase continuation after mid-sentence split merges
19. Quote-starting continuation merges
20. Parenthesis / dash-starting continuation merges
21. Legal heading still blocks merge
22. Ambiguous uppercase after full stop does NOT merge
23. reading_order is stable and deterministic after merge
24. reading_order uses Y-band quantisation consistent with extractor v2
25. Existing merge cases unaffected by reading_order change
"""
import json

import pytest

from pipeline.layout_models import (
    DocumentLayout,
    ExtractedSpan,
    LayoutBlock,
    PageLayout,
)
from pipeline.layout_normalizer_v2 import (
    _is_bottom_zone,
    _is_top_zone,
    _looks_like_paragraph_continuation,
    _merge_blocks,
    _normalize_signature,
    _should_merge_blocks,
    normalize_document_layout,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_span(
    text: str,
    page_number: int,
    font_size: float = 10.0,
    y0: float = 0.0,
) -> ExtractedSpan:
    return ExtractedSpan(
        text=text,
        bbox=(72.0, y0, 300.0, y0 + 12.0),
        font_size=font_size,
        font_name="Helvetica",
        is_bold=False,
        is_italic=False,
        page_number=page_number,
        block_no=0,
        line_no=0,
        span_no=0,
    )


def _make_block(
    block_id: str,
    page_number: int,
    text: str,
    y0: float,
    kind: str = "text",
    reading_order: int = 0,
    font_size: float = 10.0,
    metadata: dict[str, object] | None = None,
    source: str = "pymupdf_text",
) -> LayoutBlock:
    return LayoutBlock(
        block_id=block_id,
        page_number=page_number,
        bbox=(72.0, y0, 500.0, y0 + 14.0),
        text=text,
        kind=kind,
        reading_order=reading_order,
        spans=[_make_span(text, page_number, font_size=font_size, y0=y0)],
        source=source,
        metadata=metadata if metadata is not None else {},
    )


def _make_page(
    page_number: int,
    blocks: list[LayoutBlock],
    height: float = 842.0,
    raw_tables: list[dict[str, object]] | None = None,
) -> PageLayout:
    return PageLayout(
        page_number=page_number,
        width=595.0,
        height=height,
        blocks=blocks,
        raw_tables=raw_tables if raw_tables is not None else [],
        raw_drawings=[],
    )


def _make_doc(pages: list[PageLayout]) -> DocumentLayout:
    return DocumentLayout(
        pages=pages,
        native_toc=[],
        metadata={"total_pages": len(pages)},
    )


# ---------------------------------------------------------------------------
# Header / footer detection
# ---------------------------------------------------------------------------


class TestHeaderFooterDetection:
    """Tests 1-4: header/footer detection by similarity."""

    def test_repeated_header_marked(self) -> None:
        """#1 — Header repeated on 3 pages is marked kind='header'."""
        pages: list[PageLayout] = []
        for pn in range(1, 4):
            header_block = _make_block(
                f"p{pn}_b0", pn, "DIARIO OFICIAL DE LA FEDERACIÓN", y0=20.0,
                reading_order=0,
            )
            body_block = _make_block(
                f"p{pn}_b1", pn, f"Artículo {pn}. Texto del artículo.", y0=200.0,
                reading_order=1,
            )
            pages.append(_make_page(pn, [header_block, body_block]))

        result = normalize_document_layout(_make_doc(pages))

        for page in result.pages:
            top_blocks = [b for b in page.blocks if b.bbox[1] < 50.0]
            assert len(top_blocks) >= 1
            for b in top_blocks:
                assert b.kind == "header", (
                    f"Expected header on page {page.page_number}, got kind={b.kind}"
                )
                assert "normalized_signature" in b.metadata
                assert "header_footer_score" in b.metadata
                assert "header_footer_reason" in b.metadata

    def test_repeated_footer_marked(self) -> None:
        """#2 — Footer repeated on 3 pages is marked kind='footer'."""
        pages: list[PageLayout] = []
        for pn in range(1, 4):
            body_block = _make_block(
                f"p{pn}_b0", pn, f"Artículo {pn}. Texto del artículo.", y0=200.0,
                reading_order=0,
            )
            footer_block = _make_block(
                f"p{pn}_b1", pn, "Secretaría de Gobernación", y0=810.0,
                reading_order=1,
            )
            pages.append(_make_page(pn, [body_block, footer_block]))

        result = normalize_document_layout(_make_doc(pages))

        for page in result.pages:
            bottom_blocks = [b for b in page.blocks if b.bbox[1] > 800.0]
            assert len(bottom_blocks) >= 1
            for b in bottom_blocks:
                assert b.kind == "footer", (
                    f"Expected footer on page {page.page_number}, got kind={b.kind}"
                )

    def test_date_variation_does_not_prevent_detection(self) -> None:
        """#3 — Headers with different dates/page numbers still match."""
        pages: list[PageLayout] = []
        dates = [
            "DOF 15 de enero de 2024 - Página 1",
            "DOF 15 de febrero de 2024 - Página 2",
            "DOF 15 de marzo de 2024 - Página 3",
        ]
        for pn, date_text in enumerate(dates, start=1):
            header_block = _make_block(
                f"p{pn}_b0", pn, date_text, y0=15.0, reading_order=0,
            )
            body_block = _make_block(
                f"p{pn}_b1", pn, f"Artículo {pn}. Contenido.", y0=200.0,
                reading_order=1,
            )
            pages.append(_make_page(pn, [header_block, body_block]))

        result = normalize_document_layout(_make_doc(pages))

        header_count: int = sum(
            1 for p in result.pages for b in p.blocks if b.kind == "header"
        )
        assert header_count == 3, (
            f"Expected 3 headers despite date variations, got {header_count}"
        )

    def test_unique_block_not_falsely_marked(self) -> None:
        """#4 — A block appearing only on one page is NOT marked header/footer."""
        pages: list[PageLayout] = []
        for pn in range(1, 4):
            blocks: list[LayoutBlock] = []
            if pn == 1:
                blocks.append(
                    _make_block(
                        f"p{pn}_b0", pn, "Encabezado único de página 1",
                        y0=20.0, reading_order=0,
                    )
                )
            blocks.append(
                _make_block(
                    f"p{pn}_b1", pn, f"Artículo {pn}. Texto normal.", y0=200.0,
                    reading_order=len(blocks),
                )
            )
            pages.append(_make_page(pn, blocks))

        result = normalize_document_layout(_make_doc(pages))

        page1 = result.pages[0]
        top_blocks = [b for b in page1.blocks if b.bbox[1] < 50.0]
        for b in top_blocks:
            assert b.kind != "header", (
                "Unique block should NOT be marked as header"
            )


# ---------------------------------------------------------------------------
# Index zone detection
# ---------------------------------------------------------------------------


class TestIndexZoneDetection:
    """Tests 5-6: possible_index_zone marking."""

    def test_index_like_block_marked(self) -> None:
        """#5 — Block with many short article/chapter references → possible_index_zone."""
        index_text = (
            "Artículo 1\n"
            "Artículo 2\n"
            "Artículo 3\n"
            "Capítulo I\n"
            "Sección Primera\n"
            "Título Segundo\n"
        )
        block = _make_block("p1_b0", 1, index_text, y0=200.0, reading_order=0)
        page = _make_page(1, [block])
        doc = _make_doc([page])

        result = normalize_document_layout(doc)

        result_block = result.pages[0].blocks[0]
        assert result_block.metadata.get("possible_index_zone") is True, (
            "Block with dense legal references should be marked as possible_index_zone"
        )

    def test_normal_prose_not_marked_as_index(self) -> None:
        """#6 — Normal legal prose is NOT marked as index zone."""
        prose = (
            "La presente Ley es de observancia general en toda la República y rige "
            "las relaciones de trabajo comprendidas en el Artículo 123, apartado A, "
            "de la Constitución Política de los Estados Unidos Mexicanos. Esta ley "
            "establece las condiciones mínimas de trabajo que deben observarse en "
            "beneficio de los trabajadores."
        )
        block = _make_block("p1_b0", 1, prose, y0=200.0, reading_order=0)
        page = _make_page(1, [block])
        doc = _make_doc([page])

        result = normalize_document_layout(doc)

        result_block = result.pages[0].blocks[0]
        assert result_block.metadata.get("possible_index_zone") is not True, (
            "Regular prose should not be marked as possible_index_zone"
        )


# ---------------------------------------------------------------------------
# Block merging
# ---------------------------------------------------------------------------


class TestBlockMerging:
    """Tests 7-12: controlled fusion of artificially-split blocks."""

    def test_contiguous_paragraphs_merged(self) -> None:
        """#7 — Two contiguous paragraph blocks are merged into one."""
        left = _make_block(
            "p1_b0", 1, "La presente Ley establece las condiciones", y0=200.0,
            reading_order=0,
        )
        right = _make_block(
            "p1_b1", 1, "mínimas de trabajo en beneficio de los trabajadores.",
            y0=214.0, reading_order=1,
        )
        page = _make_page(1, [left, right])
        doc = _make_doc([page])

        result = normalize_document_layout(doc)

        assert len(result.pages[0].blocks) == 1, (
            "Two contiguous paragraph blocks should merge into one"
        )
        merged = result.pages[0].blocks[0]
        assert "condiciones" in merged.text
        assert "mínimas" in merged.text

    def test_no_merge_when_right_is_legal_heading(self) -> None:
        """#8 — Merge blocked when right block looks like a legal heading."""
        left = _make_block(
            "p1_b0", 1, "Fin del artículo anterior.", y0=200.0, reading_order=0,
        )
        right = _make_block(
            "p1_b1", 1, "Artículo 5. Siguiente disposición.", y0=214.0,
            reading_order=1,
        )
        page = _make_page(1, [left, right])
        doc = _make_doc([page])

        result = normalize_document_layout(doc)

        assert len(result.pages[0].blocks) == 2, (
            "Should NOT merge when right block starts with a legal heading pattern"
        )

    def test_no_merge_when_header_footer(self) -> None:
        """#9 — Merge blocked when either block is marked as header/footer."""
        header = _make_block(
            "p1_b0", 1, "DIARIO OFICIAL", y0=200.0, reading_order=0,
            kind="header",
        )
        body = _make_block(
            "p1_b1", 1, "continuación del texto.", y0=214.0, reading_order=1,
        )
        page = _make_page(1, [header, body])

        assert not _should_merge_blocks(header, body, page)

    def test_no_merge_when_table_between(self) -> None:
        """#10 — Merge blocked when a candidate table sits between blocks."""
        left = _make_block(
            "p1_b0", 1, "Texto antes de tabla", y0=200.0, reading_order=0,
        )
        right = _make_block(
            "p1_b1", 1, "texto después de tabla.", y0=300.0, reading_order=1,
        )
        table_candidate: dict[str, object] = {
            "bbox": (72.0, 215.0, 500.0, 290.0),
            "page_number": 1,
            "row_count": 3,
            "col_count": 2,
            "table_index": 0,
        }
        page = _make_page(1, [left, right], raw_tables=[table_candidate])

        assert not _should_merge_blocks(left, right, page)

    def test_merged_block_has_traceability(self) -> None:
        """#11 — Merged block preserves source_block_ids and merged_from."""
        left = _make_block(
            "p1_b0", 1, "Primera parte del párrafo", y0=200.0, reading_order=0,
        )
        right = _make_block(
            "p1_b1", 1, "segunda parte continúa aquí.", y0=214.0, reading_order=1,
        )
        merged = _merge_blocks(left, right)

        assert merged.source == "merged"
        assert merged.metadata.get("merged") is True
        assert "p1_b0" in merged.metadata.get("source_block_ids", [])  # type: ignore[operator]
        assert "p1_b1" in merged.metadata.get("source_block_ids", [])  # type: ignore[operator]
        assert merged.metadata.get("merged_from") == ["p1_b0", "p1_b1"]

    def test_reading_order_consistent_after_merge(self) -> None:
        """#12 — After merging, reading_order is 0-based and sequential."""
        blocks: list[LayoutBlock] = [
            _make_block("p1_b0", 1, "Primera parte del texto", y0=200.0, reading_order=0),
            _make_block("p1_b1", 1, "segunda parte continúa.", y0=214.0, reading_order=1),
            _make_block("p1_b2", 1, "Artículo 5. Otro bloque.", y0=300.0, reading_order=2),
            _make_block("p1_b3", 1, "Artículo 6. Último bloque.", y0=400.0, reading_order=3),
        ]
        page = _make_page(1, blocks)
        doc = _make_doc([page])

        result = normalize_document_layout(doc)

        orders = [b.reading_order for b in result.pages[0].blocks]
        assert orders == list(range(len(orders))), (
            f"reading_order should be 0-based sequential, got {orders}"
        )


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


class TestRobustness:
    """Tests 13-14: edge cases and serialization."""

    def test_empty_pages_and_small_documents(self) -> None:
        """#13 — Empty pages and single-page docs don't crash."""
        empty_page = _make_page(1, [])
        doc_empty = _make_doc([empty_page])

        result_empty = normalize_document_layout(doc_empty)
        assert len(result_empty.pages) == 1
        assert result_empty.pages[0].blocks == []

        single_block = _make_block("p1_b0", 1, "Artículo único.", y0=200.0)
        doc_single = _make_doc([_make_page(1, [single_block])])
        result_single = normalize_document_layout(doc_single)
        assert len(result_single.pages[0].blocks) == 1

        doc_no_pages = _make_doc([])
        result_no_pages = normalize_document_layout(doc_no_pages)
        assert len(result_no_pages.pages) == 0

    def test_output_serializable_pydantic_v2(self) -> None:
        """#14 — Output is valid JSON via Pydantic v2 model_dump_json."""
        blocks: list[LayoutBlock] = [
            _make_block("p1_b0", 1, "Primera parte", y0=200.0, reading_order=0),
            _make_block("p1_b1", 1, "segunda parte continúa.", y0=214.0, reading_order=1),
        ]
        page = _make_page(1, blocks)
        doc = _make_doc([page])

        result = normalize_document_layout(doc)

        json_str: str = result.model_dump_json()
        parsed: dict = json.loads(json_str)
        assert "pages" in parsed
        assert "native_toc" in parsed
        assert "metadata" in parsed

        # Verify blocks inside pages are properly serialized
        for p in parsed["pages"]:
            assert "blocks" in p
            for b in p["blocks"]:
                assert "block_id" in b
                assert "metadata" in b

        # Round-trip: re-validate
        reconstructed = DocumentLayout.model_validate(parsed)
        assert len(reconstructed.pages) == len(result.pages)


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


class TestHelpers:
    """Direct tests of internal helpers for edge-case coverage."""

    def test_normalize_signature_collapses_whitespace(self) -> None:
        raw = "  DIARIO   OFICIAL   DE   LA   FEDERACIÓN  "
        sig = _normalize_signature(raw)
        assert "  " not in sig
        assert sig == sig.strip()

    def test_normalize_signature_replaces_dates(self) -> None:
        raw = "DOF 15 de enero de 2024"
        sig = _normalize_signature(raw)
        assert "ENERO" not in sig
        assert "<<DATE>>" in sig

    def test_normalize_signature_replaces_page_numbers(self) -> None:
        raw = "Página 42"
        sig = _normalize_signature(raw)
        assert "42" not in sig

    def test_is_top_zone(self) -> None:
        page = _make_page(1, [], height=842.0)
        block_top = _make_block("b1", 1, "header", y0=10.0)
        block_mid = _make_block("b2", 1, "body", y0=400.0)
        assert _is_top_zone(block_top, page) is True
        assert _is_top_zone(block_mid, page) is False

    def test_is_bottom_zone(self) -> None:
        page = _make_page(1, [], height=842.0)
        block_bottom = _make_block("b1", 1, "footer", y0=810.0)
        block_mid = _make_block("b2", 1, "body", y0=400.0)
        assert _is_bottom_zone(block_bottom, page) is True
        assert _is_bottom_zone(block_mid, page) is False


# ---------------------------------------------------------------------------
# Ajuste 1 — Repetition threshold with ceil
# ---------------------------------------------------------------------------


class TestRepetitionThreshold:
    """Tests 15-17: ceil-based threshold for header/footer detection."""

    def test_6_pages_2_appearances_not_marked(self) -> None:
        """#15 — In a 6-page doc, a signature in only 2 pages does NOT reach 40%.

        ceil(6 * 0.40) = ceil(2.4) = 3, so 2 appearances < 3 → not marked.
        """
        pages: list[PageLayout] = []
        for pn in range(1, 7):
            blocks: list[LayoutBlock] = [
                _make_block(
                    f"p{pn}_b0", pn, f"Artículo {pn}. Contenido.", y0=200.0,
                    reading_order=0,
                ),
            ]
            if pn <= 2:
                blocks.insert(
                    0,
                    _make_block(
                        f"p{pn}_hdr", pn, "ENCABEZADO REPETIDO", y0=15.0,
                        reading_order=0,
                    ),
                )
            pages.append(_make_page(pn, blocks))

        result = normalize_document_layout(_make_doc(pages))

        header_count: int = sum(
            1 for p in result.pages for b in p.blocks if b.kind == "header"
        )
        assert header_count == 0, (
            f"2 appearances in 6 pages should NOT meet ceil threshold, got {header_count} headers"
        )

    def test_6_pages_3_appearances_marked(self) -> None:
        """#16 — In a 6-page doc, a signature in 3 pages meets the 40% ceil threshold.

        ceil(6 * 0.40) = ceil(2.4) = 3, so 3 appearances >= 3 → marked.
        """
        pages: list[PageLayout] = []
        for pn in range(1, 7):
            blocks: list[LayoutBlock] = [
                _make_block(
                    f"p{pn}_b0", pn, f"Artículo {pn}. Contenido.", y0=200.0,
                    reading_order=0,
                ),
            ]
            if pn <= 3:
                blocks.insert(
                    0,
                    _make_block(
                        f"p{pn}_hdr", pn, "ENCABEZADO REPETIDO", y0=15.0,
                        reading_order=0,
                    ),
                )
            pages.append(_make_page(pn, blocks))

        result = normalize_document_layout(_make_doc(pages))

        header_count: int = sum(
            1 for p in result.pages for b in p.blocks if b.kind == "header"
        )
        assert header_count == 3, (
            f"3 appearances in 6 pages should meet ceil threshold, got {header_count} headers"
        )

    def test_2_pages_2_appearances_marked_by_minimum(self) -> None:
        """#17 — In a 2-page doc, signature in both pages is marked (min-pages rule).

        max(2, ceil(2 * 0.40)) = max(2, 1) = 2, and 2 >= 2 → marked.
        """
        pages: list[PageLayout] = []
        for pn in range(1, 3):
            header = _make_block(
                f"p{pn}_hdr", pn, "DIARIO OFICIAL", y0=15.0, reading_order=0,
            )
            body = _make_block(
                f"p{pn}_b0", pn, f"Artículo {pn}.", y0=200.0, reading_order=1,
            )
            pages.append(_make_page(pn, [header, body]))

        result = normalize_document_layout(_make_doc(pages))

        header_count: int = sum(
            1 for p in result.pages for b in p.blocks if b.kind == "header"
        )
        assert header_count == 2, (
            f"2-page doc with 2 appearances should be marked by min-pages rule, got {header_count}"
        )


# ---------------------------------------------------------------------------
# Ajuste 2 — Robust merge continuation heuristic
# ---------------------------------------------------------------------------


class TestMergeContinuationHeuristic:
    """Tests 18-22: _looks_like_paragraph_continuation coverage."""

    def test_uppercase_continuation_after_mid_sentence(self) -> None:
        """#18 — Right starts uppercase but left is mid-sentence → merge."""
        left = _make_block(
            "p1_b0", 1, "de la República y rige las relaciones de", y0=200.0,
            reading_order=0,
        )
        right = _make_block(
            "p1_b1", 1, "México y demás entidades federativas.", y0=214.0,
            reading_order=1,
        )
        page = _make_page(1, [left, right])
        doc = _make_doc([page])

        result = normalize_document_layout(doc)

        assert len(result.pages[0].blocks) == 1, (
            "Uppercase continuation after mid-sentence split should merge"
        )
        assert "México" in result.pages[0].blocks[0].text

    def test_quote_starting_continuation_merges(self) -> None:
        """#19 — Right starts with opening quote → merge."""
        left = _make_block(
            "p1_b0", 1, "conforme establece la ley.", y0=200.0, reading_order=0,
        )
        right = _make_block(
            "p1_b1", 1, '"en los términos del presente reglamento"', y0=214.0,
            reading_order=1,
        )
        page = _make_page(1, [left, right])

        assert _should_merge_blocks(left, right, page) is True

    def test_paren_and_dash_continuation_merges(self) -> None:
        """#20 — Right starts with parenthesis or dash → merge."""
        base = _make_block(
            "p1_b0", 1, "las siguientes condiciones.", y0=200.0, reading_order=0,
        )
        paren = _make_block(
            "p1_b1", 1, "(incluidas las del artículo anterior)", y0=214.0,
            reading_order=1,
        )
        dash = _make_block(
            "p1_b2", 1, "—según lo establecido—", y0=214.0, reading_order=1,
        )
        page_paren = _make_page(1, [base, paren])
        page_dash = _make_page(1, [base, dash])

        assert _should_merge_blocks(base, paren, page_paren) is True, (
            "Parenthesis-starting block should merge"
        )
        assert _should_merge_blocks(base, dash, page_dash) is True, (
            "Dash-starting block should merge"
        )

    def test_legal_heading_still_blocks_merge(self) -> None:
        """#21 — Legal heading pattern in right block prevents merge regardless."""
        left = _make_block(
            "p1_b0", 1, "contenido del artículo anterior", y0=200.0,
            reading_order=0,
        )
        right = _make_block(
            "p1_b1", 1, "Capítulo IV De las obligaciones", y0=214.0,
            reading_order=1,
        )
        page = _make_page(1, [left, right])

        assert _should_merge_blocks(left, right, page) is False

    def test_ambiguous_uppercase_after_full_stop_no_merge(self) -> None:
        """#22 — Left ends with '.' and right starts uppercase non-heading → no merge."""
        left = _make_block(
            "p1_b0", 1, "establecidas por la autoridad competente.", y0=200.0,
            reading_order=0,
        )
        right = _make_block(
            "p1_b1", 1, "Los trabajadores podrán solicitar revisión.", y0=214.0,
            reading_order=1,
        )
        page = _make_page(1, [left, right])

        assert _should_merge_blocks(left, right, page) is False, (
            "Ambiguous uppercase after full stop should NOT merge (conservative)"
        )

    def test_looks_like_paragraph_continuation_unit(self) -> None:
        """Direct unit tests for _looks_like_paragraph_continuation helper."""
        assert _looks_like_paragraph_continuation("foo", "bar") is True
        assert _looks_like_paragraph_continuation("foo", "Bar") is True
        assert _looks_like_paragraph_continuation("foo.", "bar") is True
        assert _looks_like_paragraph_continuation("foo.", "Bar") is False
        assert _looks_like_paragraph_continuation("foo.", '"Bar"') is True
        assert _looks_like_paragraph_continuation("foo.", "(bar)") is True
        assert _looks_like_paragraph_continuation("foo.", "—bar") is True
        assert _looks_like_paragraph_continuation("foo,", "Bar") is True
        assert _looks_like_paragraph_continuation("foo;", "Bar") is True
        assert _looks_like_paragraph_continuation("", "bar") is False
        assert _looks_like_paragraph_continuation("foo", "") is False


# ---------------------------------------------------------------------------
# Ajuste 3 — reading_order aligned with extractor v2
# ---------------------------------------------------------------------------


class TestReadingOrderPostMerge:
    """Tests 23-25: reading_order consistency after merge."""

    def test_reading_order_stable_and_deterministic(self) -> None:
        """#23 — Same input produces same reading_order regardless of block list order."""
        blocks_fwd: list[LayoutBlock] = [
            _make_block("p1_b0", 1, "Primer bloque.", y0=100.0, reading_order=0),
            _make_block("p1_b1", 1, "Segundo bloque.", y0=200.0, reading_order=1),
            _make_block("p1_b2", 1, "Tercer bloque.", y0=300.0, reading_order=2),
        ]
        blocks_rev: list[LayoutBlock] = list(reversed(blocks_fwd))

        doc_fwd = _make_doc([_make_page(1, blocks_fwd)])
        doc_rev = _make_doc([_make_page(1, blocks_rev)])

        result_fwd = normalize_document_layout(doc_fwd)
        result_rev = normalize_document_layout(doc_rev)

        ids_fwd = [(b.block_id, b.reading_order) for b in result_fwd.pages[0].blocks]
        ids_rev = [(b.block_id, b.reading_order) for b in result_rev.pages[0].blocks]
        assert ids_fwd == ids_rev, (
            "reading_order must be deterministic regardless of input order"
        )

    def test_reading_order_uses_y_band_quantisation(self) -> None:
        """#24 — Blocks within 5pt Y tolerance are grouped and ordered by X.

        This matches the Y-band logic in layout_extractor_v2._compute_reading_order.
        """
        left = _make_block("p1_b0", 1, "Bloque izquierdo.", y0=100.0, reading_order=0)
        right_block = LayoutBlock(
            block_id="p1_b1",
            page_number=1,
            bbox=(350.0, 103.0, 500.0, 115.0),
            text="Bloque derecho.",
            kind="text",
            reading_order=1,
            spans=[_make_span("Bloque derecho.", 1, y0=103.0)],
            source="pymupdf_text",
            metadata={},
        )
        page = _make_page(1, [right_block, left])
        doc = _make_doc([page])

        result = normalize_document_layout(doc)
        blocks = result.pages[0].blocks

        assert blocks[0].block_id == "p1_b0", (
            "Left block (x=72) should come first within the same Y-band"
        )
        assert blocks[1].block_id == "p1_b1"

    def test_existing_merge_unaffected_by_reading_order_change(self) -> None:
        """#25 — Merge + reading_order reassignment still produces correct results."""
        blocks: list[LayoutBlock] = [
            _make_block("p1_b0", 1, "Primera parte del texto", y0=200.0, reading_order=0),
            _make_block("p1_b1", 1, "segunda parte continúa.", y0=214.0, reading_order=1),
            _make_block("p1_b2", 1, "Artículo 5. Otro bloque.", y0=300.0, reading_order=2),
        ]
        page = _make_page(1, blocks)
        doc = _make_doc([page])

        result = normalize_document_layout(doc)
        result_blocks = result.pages[0].blocks

        assert len(result_blocks) == 2, "First two blocks should merge"
        assert result_blocks[0].reading_order == 0
        assert result_blocks[1].reading_order == 1
        assert "segunda" in result_blocks[0].text
        assert "Artículo 5" in result_blocks[1].text
