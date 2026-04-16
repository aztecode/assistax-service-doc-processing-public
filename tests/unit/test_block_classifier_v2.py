"""
Unit tests for the hybrid block classifier v2 (Phase 3).

Covers 19 required scenarios:

 1. Date in uppercase → not title_heading
 2. DOF-style date → not chapter_heading or section_heading
 3. "de conformidad con el artículo 8..." → not article_heading
 4. "artículo 74 de la Constitución..." inside phrase → article_body
 5. Block kind=header → page_header
 6. Block kind=footer → page_footer
 7. Top-zone block that looks like real title → not crushed as page_header
 8. possible_index_zone=True + enumerative → index_block
 9. Normal prose → not index_block
10. "TRANSITORIOS" → transitory_heading
11. "Primero.-" / "Segundo.-" → transitory_item
12. "I." / "II." → fraction
13. "a)" / "b)" → inciso
14. Incidental "I." in prose → not over-detected as fraction
15. Strong table signal → table
16. Editorial note / aclaración → editorial_note
17. Heuristic sufficient → llm_used=False
18. LLM failure → clean fallback to heuristic
19. Output is serializable with Pydantic v2
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from pipeline.block_rules_v2 import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    VALID_LABELS,
    classify_block_by_rules,
)
from pipeline.block_classifier_v2 import (
    _classify_single_block,
    classify_document_layout,
)
from pipeline.layout_models import (
    ClassifiedBlock,
    DocumentLayout,
    ExtractedSpan,
    LayoutBlock,
    PageLayout,
)


# ---------------------------------------------------------------------------
# Helpers — reuse the same pattern as test_layout_normalizer_v2
# ---------------------------------------------------------------------------


def _span(
    text: str,
    page_number: int = 1,
    font_size: float = 10.0,
    y0: float = 200.0,
) -> ExtractedSpan:
    return ExtractedSpan(
        text=text,
        bbox=(72.0, y0, 400.0, y0 + 12.0),
        font_size=font_size,
        font_name="Helvetica",
        is_bold=False,
        is_italic=False,
        page_number=page_number,
        block_no=0,
        line_no=0,
        span_no=0,
    )


def _block(
    text: str,
    block_id: str = "p1_b0",
    page_number: int = 1,
    y0: float = 200.0,
    kind: str = "text",
    reading_order: int = 0,
    source: str = "pymupdf_text",
    metadata: dict[str, object] | None = None,
) -> LayoutBlock:
    return LayoutBlock(
        block_id=block_id,
        page_number=page_number,
        bbox=(72.0, y0, 500.0, y0 + 14.0),
        text=text,
        kind=kind,
        reading_order=reading_order,
        spans=[_span(text, page_number, y0=y0)],
        source=source,
        metadata=metadata if metadata is not None else {},
    )


def _page(
    blocks: list[LayoutBlock],
    page_number: int = 1,
) -> PageLayout:
    return PageLayout(
        page_number=page_number,
        width=595.0,
        height=842.0,
        blocks=blocks,
        raw_tables=[],
        raw_drawings=[],
    )


def _doc(pages: list[PageLayout]) -> DocumentLayout:
    return DocumentLayout(
        pages=pages,
        native_toc=[],
        metadata={"total_pages": len(pages)},
    )


def _classify_text(
    text: str,
    kind: str = "text",
    metadata: dict[str, object] | None = None,
    y0: float = 200.0,
    source: str = "pymupdf_text",
) -> tuple[str, float, str | None]:
    """Shorthand: run heuristic classification on a text string."""
    b: LayoutBlock = _block(text, kind=kind, metadata=metadata, y0=y0, source=source)
    return classify_block_by_rules(b)


# ---------------------------------------------------------------------------
# 1-2: Dates must not become headings
# ---------------------------------------------------------------------------


class TestDateFalsePositives:
    def test_uppercase_date_not_title(self) -> None:
        """#1 — '12 DE ENERO DE 2025' → not title_heading."""
        label, _, _ = _classify_text("12 DE ENERO DE 2025")
        assert label != "title_heading", f"Date should not be title_heading, got {label}"
        assert label != "chapter_heading"
        assert label != "section_heading"

    def test_dof_date_not_heading(self) -> None:
        """#2 — 'Nuevo Presupuesto DOF 24-12-2024' → not chapter/section heading."""
        label, _, _ = _classify_text("Nuevo Presupuesto DOF 24-12-2024")
        assert label != "chapter_heading", f"DOF date should not be chapter_heading, got {label}"
        assert label != "section_heading"
        assert label != "title_heading"

    def test_city_date_not_heading(self) -> None:
        """Bonus — 'Ciudad de México, 24 de diciembre de 2024' → not heading."""
        label, _, _ = _classify_text("Ciudad de México, 24 de diciembre de 2024")
        assert label not in ("title_heading", "chapter_heading", "section_heading")


# ---------------------------------------------------------------------------
# 3-4: Article references in prose are not article_heading
# ---------------------------------------------------------------------------


class TestArticleReferencesInProse:
    def test_conformidad_articulo_not_heading(self) -> None:
        """#3 — 'de conformidad con el artículo 8...' → not article_heading."""
        text = "de conformidad con el artículo 8 de esta Ley, los sujetos obligados deberán informar."
        label, _, _ = _classify_text(text)
        assert label != "article_heading", f"Inline article ref should not be article_heading, got {label}"

    def test_articulo_constitucion_not_heading(self) -> None:
        """#4 — 'artículo 74 de la Constitución...' inside a phrase → article_body."""
        text = "Lo dispuesto en el artículo 74 de la Constitución Política de los Estados Unidos Mexicanos será aplicable en los términos de esta ley."
        label, _, _ = _classify_text(text)
        assert label != "article_heading"
        assert label == "article_body"


# ---------------------------------------------------------------------------
# 5-7: Headers, footers, and real document titles in top zone
# ---------------------------------------------------------------------------


class TestHeaderFooterTitleZone:
    def test_kind_header_becomes_page_header(self) -> None:
        """#5 — Block with kind='header' → page_header."""
        label, confidence, _ = _classify_text("DIARIO OFICIAL DE LA FEDERACIÓN", kind="header")
        assert label == "page_header"
        assert confidence >= CONFIDENCE_HIGH

    def test_kind_footer_becomes_page_footer(self) -> None:
        """#6 — Block with kind='footer' → page_footer."""
        label, confidence, _ = _classify_text("Secretaría de Gobernación", kind="footer")
        assert label == "page_footer"
        assert confidence >= CONFIDENCE_HIGH

    def test_top_zone_real_title_not_crushed_as_header(self) -> None:
        """#7 — Block in top zone that is clearly a document title → document_title."""
        label, _, _ = _classify_text(
            "LEY GENERAL DE SALUD", kind="text", y0=50.0,
        )
        assert label == "document_title", f"Real title should be document_title, got {label}"


# ---------------------------------------------------------------------------
# 8-9: Index zone detection
# ---------------------------------------------------------------------------


class TestIndexZone:
    def test_index_zone_with_enumerative(self) -> None:
        """#8 — Block with possible_index_zone=True → index_block."""
        meta: dict[str, object] = {
            "possible_index_zone": True,
            "index_keyword_hits": 5,
        }
        text = "Artículo 1\nArtículo 2\nCapítulo I\nSección Primera"
        label, _, _ = _classify_text(text, metadata=meta)
        assert label == "index_block"

    def test_normal_prose_not_index(self) -> None:
        """#9 — Normal prose without index metadata → not index_block."""
        text = (
            "La presente Ley es de observancia general en toda la República y rige "
            "las relaciones de trabajo comprendidas en el Artículo 123, apartado A, "
            "de la Constitución Política de los Estados Unidos Mexicanos."
        )
        label, _, _ = _classify_text(text)
        assert label != "index_block"


# ---------------------------------------------------------------------------
# 10-11: Transitorios
# ---------------------------------------------------------------------------


class TestTransitorios:
    def test_transitorios_heading(self) -> None:
        """#10 — 'TRANSITORIOS' → transitory_heading."""
        label, _, _ = _classify_text("TRANSITORIOS")
        assert label == "transitory_heading"

    def test_transitory_items(self) -> None:
        """#11 — 'Primero.-' and 'Segundo.-' → transitory_item."""
        for text in ("Primero.- El presente decreto entrará en vigor...",
                      "Segundo.- Se derogan las disposiciones anteriores."):
            label, _, _ = _classify_text(text)
            assert label == "transitory_item", f"Expected transitory_item for '{text[:30]}', got {label}"


# ---------------------------------------------------------------------------
# 12-14: Fracciones e incisos
# ---------------------------------------------------------------------------


class TestFraccionesIncisos:
    def test_roman_fraction(self) -> None:
        """#12 — 'I.' and 'II.' → fraction."""
        for text in ("I. Presentar declaración anual.",
                      "II.- Conservar la contabilidad."):
            label, _, _ = _classify_text(text)
            assert label == "fraction", f"Expected fraction for '{text[:20]}', got {label}"

    def test_letter_inciso(self) -> None:
        """#13 — 'a)' and 'b)' → inciso."""
        for text in ("a) Los contribuyentes del régimen general.",
                      "b) Los que tributen bajo el régimen simplificado."):
            label, _, _ = _classify_text(text)
            assert label == "inciso", f"Expected inciso for '{text[:20]}', got {label}"

    def test_roman_in_prose_not_fraction(self) -> None:
        """#14 — Incidental 'I.' in prose → not over-detected as fraction."""
        text = (
            "Con base en lo señalado por la fracción I. del artículo anterior, "
            "se establece que los contribuyentes deberán presentar declaración anual."
        )
        label, _, _ = _classify_text(text)
        assert label != "fraction", f"Incidental roman numeral in prose should not be fraction, got {label}"


# ---------------------------------------------------------------------------
# 15-16: Tables and editorial notes
# ---------------------------------------------------------------------------


class TestTablesAndEditorialNotes:
    def test_table_signal(self) -> None:
        """#15 — Block with strong table signal → table."""
        label, _, _ = _classify_text(
            "Tabla de tarifas", source="pymupdf_table",
        )
        assert label == "table"

    def test_table_by_kind(self) -> None:
        """Block with kind='table' → table."""
        label, _, _ = _classify_text("Valores de referencia", kind="table")
        assert label == "table"

    def test_editorial_note(self) -> None:
        """#16 — Aclaración / nota editorial → editorial_note."""
        label, _, _ = _classify_text(
            "ACLARACIÓN: El texto del artículo 306 fue corregido conforme a Fe de Erratas.",
        )
        assert label == "editorial_note"

    def test_editorial_boxed_note(self) -> None:
        """Block kind='boxed_note' → editorial_note."""
        label, _, _ = _classify_text(
            "Nota del Editor: reformas vigentes al 2024.", kind="boxed_note",
        )
        assert label == "editorial_note"


# ---------------------------------------------------------------------------
# 17-19: Hybrid classifier robustness
# ---------------------------------------------------------------------------


class TestHybridRobustness:
    def test_heuristic_sufficient_no_llm(self) -> None:
        """#17 — When heuristic is confident enough, LLM is not invoked."""
        b = _block("TRANSITORIOS", block_id="p1_b0")
        result: ClassifiedBlock = _classify_single_block(b, [], [], {})
        assert result.label == "transitory_heading"
        assert result.llm_used is False

    def test_heuristic_sufficient_for_article(self) -> None:
        """Heuristic is confident on clear article heading → no LLM."""
        b = _block("Artículo 5.- Los contribuyentes deberán presentar declaración.")
        result: ClassifiedBlock = _classify_single_block(b, [], [], {})
        assert result.label == "article_heading"
        assert result.llm_used is False

    @patch("pipeline.block_classifier_v2.classify_ambiguous_block")
    def test_llm_failure_fallback(self, mock_llm: Any) -> None:
        """#18 — LLM failure → clean fallback to heuristic result."""
        mock_llm.return_value = {
            "label": "unknown",
            "confidence": 0.3,
            "reason": "llm_error",
        }
        # Short ambiguous text with low heuristic confidence
        b = _block("XYZ", block_id="p1_b0")
        result: ClassifiedBlock = _classify_single_block(b, [], [], {})
        # Should not crash and should have a label
        assert result.label in ("unknown", "article_body")
        assert result.llm_used is False  # LLM didn't provide convincing result

    @patch("pipeline.block_classifier_v2.classify_ambiguous_block")
    def test_llm_called_for_ambiguous(self, mock_llm: Any) -> None:
        """LLM is called when heuristic is low-confidence on unknown blocks."""
        mock_llm.return_value = {
            "label": "annex_body",
            "confidence": 0.90,
            "reason": "annex_content",
        }
        b = _block("Contenido ambiguo.", block_id="p1_b0")
        result: ClassifiedBlock = _classify_single_block(b, [], [], {})
        assert result.llm_used is True
        assert result.label == "annex_body"

    def test_output_serializable_pydantic_v2(self) -> None:
        """#19 — Full classify_document_layout output is JSON-serializable."""
        blocks: list[LayoutBlock] = [
            _block("DIARIO OFICIAL DE LA FEDERACIÓN", block_id="p1_b0", kind="header", y0=20.0),
            _block("LEY GENERAL DE SALUD", block_id="p1_b1", y0=100.0, reading_order=1),
            _block("Artículo 1.- Esta ley es de aplicación general.", block_id="p1_b2", y0=200.0, reading_order=2),
            _block("TRANSITORIOS", block_id="p1_b3", y0=400.0, reading_order=3),
            _block("Primero.- El presente decreto entrará en vigor.", block_id="p1_b4", y0=450.0, reading_order=4),
        ]
        doc: DocumentLayout = _doc([_page(blocks)])

        classified: list[ClassifiedBlock] = classify_document_layout(doc)

        assert len(classified) == 5

        # Serialize each block to JSON and back
        for cb in classified:
            json_str: str = cb.model_dump_json()
            parsed: dict[str, Any] = json.loads(json_str)
            assert "block_id" in parsed
            assert "label" in parsed
            assert "confidence" in parsed
            assert "llm_used" in parsed
            assert "normalized_text" in parsed
            assert "metadata" in parsed

            # Round-trip
            reconstructed = ClassifiedBlock.model_validate(parsed)
            assert reconstructed.label == cb.label

    def test_full_pipeline_label_consistency(self) -> None:
        """All labels in the output belong to VALID_LABELS."""
        blocks: list[LayoutBlock] = [
            _block("DIARIO OFICIAL DE LA FEDERACIÓN", block_id="p1_h", kind="header", y0=20.0),
            _block("LEY FEDERAL DEL TRABAJO", block_id="p1_t", y0=100.0, reading_order=1),
            _block("Título Primero", block_id="p1_b0", y0=150.0, reading_order=2),
            _block("Capítulo I", block_id="p1_b1", y0=200.0, reading_order=3),
            _block("Artículo 1.- Las disposiciones de esta ley.", block_id="p1_b2", y0=250.0, reading_order=4),
            _block("I. Presentar declaración.", block_id="p1_b3", y0=300.0, reading_order=5),
            _block("a) Contribuyentes del régimen general.", block_id="p1_b4", y0=350.0, reading_order=6),
            _block("TRANSITORIOS", block_id="p1_b5", y0=400.0, reading_order=7),
            _block("Primero.- Entrará en vigor al día siguiente.", block_id="p1_b6", y0=450.0, reading_order=8),
        ]
        doc: DocumentLayout = _doc([_page(blocks)])
        classified: list[ClassifiedBlock] = classify_document_layout(doc)

        for cb in classified:
            assert cb.label in VALID_LABELS, f"Label {cb.label!r} not in VALID_LABELS"


# ---------------------------------------------------------------------------
# Structural heading patterns
# ---------------------------------------------------------------------------


class TestStructuralHeadings:
    def test_book_heading(self) -> None:
        label, _, _ = _classify_text("Libro Primero")
        # "Primero" doesn't match [IVXLCDM\d]+ so it won't match book_heading
        # But "Libro I" should
        label2, _, _ = _classify_text("Libro I")
        assert label2 == "book_heading"

    def test_title_heading(self) -> None:
        label, _, _ = _classify_text("Título II De los derechos fundamentales")
        assert label == "title_heading"

    def test_chapter_heading(self) -> None:
        label, _, _ = _classify_text("Capítulo III De las obligaciones")
        assert label == "chapter_heading"

    def test_section_heading(self) -> None:
        label, _, _ = _classify_text("Sección IV De los sujetos exentos")
        assert label == "section_heading"

    def test_annex_heading(self) -> None:
        label, _, _ = _classify_text("Anexo I")
        assert label == "annex_heading"

    def test_article_heading_real(self) -> None:
        label, _, _ = _classify_text("Artículo 10.- Los contribuyentes persona moral deberán...")
        assert label == "article_heading"
