"""
Unit tests for the hierarchical structure builder v2 (Phase 4).

Covers 20 required scenarios:

 1. title -> chapter -> article -> article_body produces correct tree
 2. article_heading without chapter/section produces valid article under root
 3. Table between two article_body blocks stays in correct relative position
 4. Table outside article is not lost
 5. page_header does not enter the main tree
 6. page_footer and index_block are preserved in metadata
 7. TRANSITORIOS creates a container node
 8. Primero.-, Segundo.- create individual transitory nodes
 9. Paragraphs after a transitory item hang from the correct transitory
10. Fraction hangs from the current article
11. Inciso hangs from the current fraction
12. Inciso without fraction hangs from article or nearest structural parent
13. editorial_note is preserved as a note node
14. editorial_note does not appear in the TOC
15. TOC is derived from tree and contains navigable nodes
16. TOC does not include paragraph, table, or note
17. document_title is captured in metadata
18. source_block_ids are preserved
19. page_start/page_end are updated correctly
20. Full output is serializable with Pydantic v2
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from pipeline.layout_models import ClassifiedBlock, DocumentStructure, StructuralNode
from pipeline.structure_builder_v2 import build_document_structure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cb(
    block_id: str,
    label: str,
    text: str,
    page_number: int = 1,
    confidence: float = 0.95,
) -> ClassifiedBlock:
    """Create a minimal ClassifiedBlock for testing."""
    return ClassifiedBlock(
        block_id=block_id,
        page_number=page_number,
        label=label,
        confidence=confidence,
        reason="test",
        llm_used=False,
        normalized_text=text,
        metadata={},
    )


def _find_node(root: StructuralNode, node_id: str) -> StructuralNode | None:
    """DFS search for a node by id."""
    if root.node_id == node_id:
        return root
    for child in root.children:
        found: StructuralNode | None = _find_node(child, node_id)
        if found is not None:
            return found
    return None


def _collect_node_types(root: StructuralNode) -> list[str]:
    """Collect all node_types in DFS pre-order."""
    result: list[str] = [root.node_type]
    for child in root.children:
        result.extend(_collect_node_types(child))
    return result


def _collect_node_ids(root: StructuralNode) -> list[str]:
    """Collect all node_ids in DFS pre-order."""
    result: list[str] = [root.node_id]
    for child in root.children:
        result.extend(_collect_node_ids(child))
    return result


# ---------------------------------------------------------------------------
# 1. Hierarchy: title -> chapter -> article -> article_body
# ---------------------------------------------------------------------------


class TestBasicHierarchy:
    def test_title_chapter_article_body(self) -> None:
        """#1 — Full structural hierarchy produces correct tree."""
        blocks: list[ClassifiedBlock] = [
            _cb("b1", "title_heading", "Título I De las disposiciones generales"),
            _cb("b2", "chapter_heading", "Capítulo I Principios"),
            _cb("b3", "article_heading", "Artículo 1.- Ámbito de aplicación."),
            _cb("b4", "article_body", "Esta ley es de observancia general en toda la República."),
        ]
        result: DocumentStructure = build_document_structure(blocks, None)

        assert result.root.node_type == "document"
        assert len(result.root.children) == 1

        title_node: StructuralNode = result.root.children[0]
        assert title_node.node_type == "title"
        assert len(title_node.children) == 1

        chapter_node: StructuralNode = title_node.children[0]
        assert chapter_node.node_type == "chapter"
        assert len(chapter_node.children) == 1

        article_node: StructuralNode = chapter_node.children[0]
        assert article_node.node_type == "article"
        assert article_node.article_ref == "1"
        assert len(article_node.children) == 1

        paragraph_node: StructuralNode = article_node.children[0]
        assert paragraph_node.node_type == "paragraph"

    def test_article_without_chapter(self) -> None:
        """#2 — article_heading without chapter/section produces valid article under root."""
        blocks: list[ClassifiedBlock] = [
            _cb("b1", "article_heading", "Artículo 5.- Objeto de la ley."),
            _cb("b2", "article_body", "El objeto de esta ley es regular..."),
        ]
        result: DocumentStructure = build_document_structure(blocks, None)

        assert len(result.root.children) == 1
        article_node: StructuralNode = result.root.children[0]
        assert article_node.node_type == "article"
        assert article_node.article_ref == "5"
        assert len(article_node.children) == 1
        assert article_node.children[0].node_type == "paragraph"


# ---------------------------------------------------------------------------
# 3-4. Tables: position preservation
# ---------------------------------------------------------------------------


class TestTables:
    def test_table_between_article_body_blocks(self) -> None:
        """#3 — Table between two article_body blocks is in correct relative position."""
        blocks: list[ClassifiedBlock] = [
            _cb("b1", "article_heading", "Artículo 10.- Tarifas aplicables."),
            _cb("b2", "article_body", "Las tarifas serán las siguientes:"),
            _cb("b3", "table", "Concepto | Tasa | Monto"),
            _cb("b4", "article_body", "Dichas tarifas se actualizarán anualmente."),
        ]
        result: DocumentStructure = build_document_structure(blocks, None)

        article: StructuralNode = result.root.children[0]
        assert article.node_type == "article"
        assert len(article.children) == 3

        child_types: list[str] = [c.node_type for c in article.children]
        assert child_types == ["paragraph", "table", "paragraph"]

    def test_table_outside_article_not_lost(self) -> None:
        """#4 — Table outside article is not lost."""
        blocks: list[ClassifiedBlock] = [
            _cb("b1", "table", "Tabla general de valores"),
        ]
        result: DocumentStructure = build_document_structure(blocks, None)

        assert len(result.root.children) == 1
        assert result.root.children[0].node_type == "table"


# ---------------------------------------------------------------------------
# 5-6. Excluded blocks: headers, footers, index
# ---------------------------------------------------------------------------


class TestExcludedBlocks:
    def test_page_header_excluded_from_tree(self) -> None:
        """#5 — page_header does not enter the main tree."""
        blocks: list[ClassifiedBlock] = [
            _cb("h1", "page_header", "DIARIO OFICIAL DE LA FEDERACIÓN"),
            _cb("b1", "article_heading", "Artículo 1.- Texto."),
        ]
        result: DocumentStructure = build_document_structure(blocks, None)

        all_types: list[str] = _collect_node_types(result.root)
        assert "page_header" not in all_types
        assert len(result.root.children) == 1
        assert result.root.children[0].node_type == "article"

    def test_footer_and_index_preserved_in_metadata(self) -> None:
        """#6 — page_footer and index_block are preserved in metadata."""
        blocks: list[ClassifiedBlock] = [
            _cb("f1", "page_footer", "Página 1"),
            _cb("i1", "index_block", "Artículo 1...pág 3"),
            _cb("b1", "article_heading", "Artículo 1.- Texto."),
        ]
        result: DocumentStructure = build_document_structure(blocks, None)

        excluded: list[dict[str, object]] = result.metadata["excluded_blocks"]  # type: ignore[assignment]
        excluded_ids: list[str] = [e["block_id"] for e in excluded]  # type: ignore[index]
        assert "f1" in excluded_ids
        assert "i1" in excluded_ids
        assert len(result.root.children) == 1


# ---------------------------------------------------------------------------
# 7-9. Transitorios
# ---------------------------------------------------------------------------


class TestTransitorios:
    def test_transitorios_creates_container(self) -> None:
        """#7 — TRANSITORIOS creates a container node."""
        blocks: list[ClassifiedBlock] = [
            _cb("b1", "transitory_heading", "TRANSITORIOS"),
        ]
        result: DocumentStructure = build_document_structure(blocks, None)

        assert len(result.root.children) == 1
        container: StructuralNode = result.root.children[0]
        assert container.node_type == "transitory"
        assert container.heading is not None
        assert "TRANSITORIOS" in container.heading

    def test_transitory_items_individual_nodes(self) -> None:
        """#8 — Primero.-, Segundo.- create individual transitory nodes."""
        blocks: list[ClassifiedBlock] = [
            _cb("b1", "transitory_heading", "TRANSITORIOS"),
            _cb("b2", "transitory_item", "Primero.- El presente decreto entrará en vigor."),
            _cb("b3", "transitory_item", "Segundo.- Se derogan las disposiciones anteriores."),
        ]
        result: DocumentStructure = build_document_structure(blocks, None)

        container: StructuralNode = result.root.children[0]
        assert container.node_type == "transitory"
        assert len(container.children) == 2
        assert container.children[0].node_id == "transitory-primero"
        assert container.children[1].node_id == "transitory-segundo"

    def test_paragraphs_after_transitory_hang_from_it(self) -> None:
        """#9 — Paragraphs after a transitory item hang from the correct transitory."""
        blocks: list[ClassifiedBlock] = [
            _cb("b1", "transitory_heading", "TRANSITORIOS"),
            _cb("b2", "transitory_item", "Primero.- Vigencia."),
            _cb("b3", "article_body", "Este decreto entrará en vigor al día siguiente."),
            _cb("b4", "transitory_item", "Segundo.- Derogaciones."),
            _cb("b5", "article_body", "Se derogan las disposiciones contrarias."),
        ]
        result: DocumentStructure = build_document_structure(blocks, None)

        container: StructuralNode = result.root.children[0]
        primero: StructuralNode = container.children[0]
        segundo: StructuralNode = container.children[1]

        assert primero.node_id == "transitory-primero"
        assert len(primero.children) == 1
        assert primero.children[0].node_type == "paragraph"
        assert "entrará en vigor" in primero.children[0].text  # type: ignore[operator]

        assert segundo.node_id == "transitory-segundo"
        assert len(segundo.children) == 1
        assert segundo.children[0].node_type == "paragraph"


# ---------------------------------------------------------------------------
# 10-12. Fracciones e incisos
# ---------------------------------------------------------------------------


class TestFraccionesIncisos:
    def test_fraction_hangs_from_article(self) -> None:
        """#10 — Fraction hangs from the current article."""
        blocks: list[ClassifiedBlock] = [
            _cb("b1", "article_heading", "Artículo 3.- Obligaciones."),
            _cb("b2", "fraction", "I. Presentar declaración anual."),
            _cb("b3", "fraction", "II. Conservar la contabilidad."),
        ]
        result: DocumentStructure = build_document_structure(blocks, None)

        article: StructuralNode = result.root.children[0]
        assert article.node_type == "article"
        assert len(article.children) == 2
        assert all(c.node_type == "fraction" for c in article.children)

    def test_inciso_hangs_from_fraction(self) -> None:
        """#11 — Inciso hangs from the current fraction."""
        blocks: list[ClassifiedBlock] = [
            _cb("b1", "article_heading", "Artículo 4.- Requisitos."),
            _cb("b2", "fraction", "I. Los contribuyentes deberán:"),
            _cb("b3", "inciso", "a) Presentar identificación oficial."),
            _cb("b4", "inciso", "b) Comprobante de domicilio."),
        ]
        result: DocumentStructure = build_document_structure(blocks, None)

        article: StructuralNode = result.root.children[0]
        fraction: StructuralNode = article.children[0]
        assert fraction.node_type == "fraction"
        assert len(fraction.children) == 2
        assert all(c.node_type == "inciso" for c in fraction.children)

    def test_inciso_without_fraction_hangs_from_article(self) -> None:
        """#12 — Inciso without fraction hangs from article or nearest parent."""
        blocks: list[ClassifiedBlock] = [
            _cb("b1", "article_heading", "Artículo 7.- Sanciones."),
            _cb("b2", "inciso", "a) Multa equivalente al 50%."),
            _cb("b3", "inciso", "b) Clausura temporal del establecimiento."),
        ]
        result: DocumentStructure = build_document_structure(blocks, None)

        article: StructuralNode = result.root.children[0]
        assert article.node_type == "article"
        assert len(article.children) == 2
        assert all(c.node_type == "inciso" for c in article.children)


# ---------------------------------------------------------------------------
# 13-14. Editorial notes
# ---------------------------------------------------------------------------


class TestEditorialNotes:
    def test_editorial_note_preserved_as_note(self) -> None:
        """#13 — editorial_note is preserved as a note node."""
        blocks: list[ClassifiedBlock] = [
            _cb("b1", "article_heading", "Artículo 2.- Definiciones."),
            _cb("b2", "editorial_note", "ACLARACIÓN: Texto corregido conforme a Fe de Erratas."),
        ]
        result: DocumentStructure = build_document_structure(blocks, None)

        article: StructuralNode = result.root.children[0]
        assert len(article.children) == 1
        note: StructuralNode = article.children[0]
        assert note.node_type == "note"

    def test_editorial_note_not_in_toc(self) -> None:
        """#14 — editorial_note does not appear in the TOC."""
        blocks: list[ClassifiedBlock] = [
            _cb("b1", "article_heading", "Artículo 2.- Definiciones."),
            _cb("b2", "editorial_note", "Nota del Editor: reformas vigentes."),
        ]
        result: DocumentStructure = build_document_structure(blocks, None)

        toc_types: list[str] = [entry["node_type"] for entry in result.toc]  # type: ignore[index]
        assert "note" not in toc_types


# ---------------------------------------------------------------------------
# 15-16. TOC
# ---------------------------------------------------------------------------


class TestTOC:
    def test_toc_derived_from_tree(self) -> None:
        """#15 — TOC is derived from tree and contains navigable nodes."""
        blocks: list[ClassifiedBlock] = [
            _cb("b1", "title_heading", "Título I Generalidades"),
            _cb("b2", "chapter_heading", "Capítulo I Principios"),
            _cb("b3", "article_heading", "Artículo 1.- Ámbito."),
            _cb("b4", "article_body", "Cuerpo del artículo."),
        ]
        result: DocumentStructure = build_document_structure(blocks, None)

        assert len(result.toc) == 3
        toc_types: list[str] = [entry["node_type"] for entry in result.toc]  # type: ignore[index]
        assert "title" in toc_types
        assert "chapter" in toc_types
        assert "article" in toc_types

        for entry in result.toc:
            assert "node_id" in entry
            assert "node_type" in entry
            assert "heading" in entry
            assert "page_start" in entry

    def test_toc_excludes_non_navigable(self) -> None:
        """#16 — TOC does not include paragraph, table, note."""
        blocks: list[ClassifiedBlock] = [
            _cb("b1", "article_heading", "Artículo 1.- Texto."),
            _cb("b2", "article_body", "Cuerpo del artículo."),
            _cb("b3", "table", "Tabla de valores"),
            _cb("b4", "editorial_note", "Nota editorial."),
        ]
        result: DocumentStructure = build_document_structure(blocks, None)

        toc_types: list[str] = [entry["node_type"] for entry in result.toc]  # type: ignore[index]
        assert "paragraph" not in toc_types
        assert "table" not in toc_types
        assert "note" not in toc_types


# ---------------------------------------------------------------------------
# 17-20. Metadata and traceability
# ---------------------------------------------------------------------------


class TestMetadataTraceability:
    def test_document_title_in_metadata(self) -> None:
        """#17 — document_title is captured in metadata."""
        blocks: list[ClassifiedBlock] = [
            _cb("b1", "document_title", "LEY GENERAL DE SALUD"),
            _cb("b2", "article_heading", "Artículo 1.- Objeto."),
        ]
        result: DocumentStructure = build_document_structure(blocks, None)

        assert result.metadata["document_title"] == "LEY GENERAL DE SALUD"
        assert result.root.heading == "LEY GENERAL DE SALUD"

    def test_source_block_ids_preserved(self) -> None:
        """#18 — source_block_ids are preserved in nodes."""
        blocks: list[ClassifiedBlock] = [
            _cb("b1", "article_heading", "Artículo 1.- Texto."),
            _cb("b2", "article_body", "Cuerpo del artículo."),
        ]
        result: DocumentStructure = build_document_structure(blocks, None)

        article: StructuralNode = result.root.children[0]
        assert "b1" in article.source_block_ids

        paragraph: StructuralNode = article.children[0]
        assert "b2" in paragraph.source_block_ids

    def test_page_range_updated(self) -> None:
        """#19 — page_start/page_end are updated correctly across children."""
        blocks: list[ClassifiedBlock] = [
            _cb("b1", "title_heading", "Título I", page_number=1),
            _cb("b2", "article_heading", "Artículo 1.- Texto.", page_number=1),
            _cb("b3", "article_body", "Cuerpo.", page_number=2),
            _cb("b4", "article_heading", "Artículo 2.- Más texto.", page_number=3),
        ]
        result: DocumentStructure = build_document_structure(blocks, None)

        title_node: StructuralNode = result.root.children[0]
        assert title_node.page_start == 1
        assert title_node.page_end == 3

        assert result.root.page_start == 1
        assert result.root.page_end == 3

    def test_output_serializable_pydantic_v2(self) -> None:
        """#20 — Full output is JSON-serializable with Pydantic v2."""
        blocks: list[ClassifiedBlock] = [
            _cb("h1", "page_header", "DIARIO OFICIAL"),
            _cb("b0", "document_title", "LEY FEDERAL DEL TRABAJO"),
            _cb("b1", "title_heading", "Título I Generalidades"),
            _cb("b2", "chapter_heading", "Capítulo I Principios"),
            _cb("b3", "article_heading", "Artículo 1.- Ámbito."),
            _cb("b4", "article_body", "Cuerpo del artículo."),
            _cb("b5", "fraction", "I. Primera fracción."),
            _cb("b6", "inciso", "a) Primer inciso."),
            _cb("b7", "table", "Tabla de tarifas"),
            _cb("b8", "editorial_note", "Nota del editor."),
            _cb("b9", "transitory_heading", "TRANSITORIOS"),
            _cb("b10", "transitory_item", "Primero.- Vigencia."),
            _cb("b11", "article_body", "Entrará en vigor mañana."),
            _cb("f1", "page_footer", "Página 1"),
        ]
        result: DocumentStructure = build_document_structure(blocks, {"source": "test"})

        json_str: str = result.model_dump_json()
        parsed: dict[str, Any] = json.loads(json_str)

        assert "root" in parsed
        assert "toc" in parsed
        assert "sections" in parsed
        assert "metadata" in parsed

        reconstructed: DocumentStructure = DocumentStructure.model_validate(parsed)
        assert reconstructed.root.node_id == "doc"
        assert reconstructed.root.node_type == "document"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_blocks_list(self) -> None:
        """Empty input produces a valid empty DocumentStructure."""
        result: DocumentStructure = build_document_structure([], None)
        assert result.root.node_type == "document"
        assert len(result.root.children) == 0
        assert len(result.toc) == 0

    def test_chapter_resets_article_context(self) -> None:
        """New chapter resets article context — articles attach to new chapter."""
        blocks: list[ClassifiedBlock] = [
            _cb("b1", "chapter_heading", "Capítulo I Primero"),
            _cb("b2", "article_heading", "Artículo 1.- Texto."),
            _cb("b3", "chapter_heading", "Capítulo II Segundo"),
            _cb("b4", "article_heading", "Artículo 2.- Otro."),
        ]
        result: DocumentStructure = build_document_structure(blocks, None)

        ch1: StructuralNode = result.root.children[0]
        ch2: StructuralNode = result.root.children[1]
        assert ch1.node_type == "chapter"
        assert ch2.node_type == "chapter"
        assert len(ch1.children) == 1
        assert ch1.children[0].article_ref == "1"
        assert len(ch2.children) == 1
        assert ch2.children[0].article_ref == "2"

    def test_annex_heading_creates_section_with_metadata(self) -> None:
        """Annex heading creates a section node with is_annex=True."""
        blocks: list[ClassifiedBlock] = [
            _cb("b1", "annex_heading", "Anexo I"),
            _cb("b2", "annex_body", "Contenido del anexo."),
        ]
        result: DocumentStructure = build_document_structure(blocks, None)

        annex: StructuralNode = result.root.children[0]
        assert annex.node_type == "section"
        assert annex.metadata.get("is_annex") is True
        assert len(annex.children) == 1
        assert annex.children[0].node_type == "paragraph"

    def test_document_direct_to_article(self) -> None:
        """Document that goes directly to articles without any structural heading."""
        blocks: list[ClassifiedBlock] = [
            _cb("b1", "article_heading", "Artículo 1.- Primero."),
            _cb("b2", "article_heading", "Artículo 2.- Segundo."),
            _cb("b3", "article_heading", "Artículo 3.- Tercero."),
        ]
        result: DocumentStructure = build_document_structure(blocks, None)

        assert len(result.root.children) == 3
        assert all(c.node_type == "article" for c in result.root.children)

    def test_unknown_label_becomes_paragraph(self) -> None:
        """Blocks with 'unknown' label are treated as paragraphs."""
        blocks: list[ClassifiedBlock] = [
            _cb("b1", "unknown", "Texto ambiguo no clasificado."),
        ]
        result: DocumentStructure = build_document_structure(blocks, None)

        assert len(result.root.children) == 1
        assert result.root.children[0].node_type == "paragraph"

    def test_sections_summary_coherent(self) -> None:
        """Sections summary includes structural nodes with article counts."""
        blocks: list[ClassifiedBlock] = [
            _cb("b1", "title_heading", "Título I Generalidades"),
            _cb("b2", "chapter_heading", "Capítulo I Principios"),
            _cb("b3", "article_heading", "Artículo 1.- Primero."),
            _cb("b4", "article_heading", "Artículo 2.- Segundo."),
        ]
        result: DocumentStructure = build_document_structure(blocks, None)

        assert len(result.sections) >= 1
        chapter_sections: list[dict[str, object]] = [
            s for s in result.sections if s["node_type"] == "chapter"
        ]
        assert len(chapter_sections) == 1
        assert chapter_sections[0]["article_count"] == 2

    def test_has_transitories_flag(self) -> None:
        """Metadata correctly flags has_transitories."""
        blocks: list[ClassifiedBlock] = [
            _cb("b1", "transitory_heading", "TRANSITORIOS"),
        ]
        result: DocumentStructure = build_document_structure(blocks, None)
        assert result.metadata["has_transitories"] is True

    def test_has_annexes_flag(self) -> None:
        """Metadata correctly flags has_annexes."""
        blocks: list[ClassifiedBlock] = [
            _cb("b1", "annex_heading", "Anexo I"),
        ]
        result: DocumentStructure = build_document_structure(blocks, None)
        assert result.metadata["has_annexes"] is True
