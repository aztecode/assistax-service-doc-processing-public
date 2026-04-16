"""
Unit tests for the chunk projector v2 (Phase 6).

Covers 20 required scenarios:

 1. Article with body produces chunk with article_ref
 2. Article with paragraphs aggregates text correctly
 3. Article with fraction produces coherent chunks
 4. Table inside article produces independent atomic chunk
 5. Table outside article is not lost
 6. Table preserves source_block_ids and page range
 7. editorial_note is projected separately from normative text
 8. note does not contaminate article's main chunk
 9. Primero.- / Segundo.- produce useful transitory chunks
10. Transitory items preserve transitory context in metadata
11. Annex is projected and preserves is_annex=True in metadata
12. Chunks come out in correct document order
13. All chunks preserve source_block_ids
14. All chunks have coherent page_start/page_end
15. quality_score and quality_severity propagate to metadata
16. Empty document produces empty list without crash
17. Full output is JSON-serializable
18. Headers/footers/index_blocks are not projected
19. Chunks do not have empty text except justified exceptions
20. Edge case: unknown->paragraph does not lose content
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from pipeline.layout_models import DocumentStructure, StructuralNode
from pipeline.chunk_projector_v2 import project_structure_to_chunks


# ---------------------------------------------------------------------------
# Helpers — reused patterns from test_quality_validator_v2
# ---------------------------------------------------------------------------


def _node(
    node_id: str,
    node_type: str,
    heading: str | None,
    text: str | None,
    article_ref: str | None,
    page_start: int | None,
    page_end: int | None,
    children: list[StructuralNode] | None,
    source_block_ids: list[str] | None,
    metadata: dict[str, object] | None = None,
) -> StructuralNode:
    return StructuralNode(
        node_id=node_id,
        node_type=node_type,
        heading=heading,
        text=text,
        article_ref=article_ref,
        page_start=page_start,
        page_end=page_end,
        children=children if children is not None else [],
        source_block_ids=source_block_ids if source_block_ids is not None else [],
        metadata=metadata if metadata is not None else {},
    )


def _make_structure(
    root: StructuralNode,
    quality_report: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
) -> DocumentStructure:
    return DocumentStructure(
        root=root,
        toc=[],
        sections=[],
        quality_report=quality_report if quality_report is not None else {},
        metadata=metadata if metadata is not None else {},
    )


def _doc_root(
    children: list[StructuralNode],
    doc_title: str | None = None,
) -> StructuralNode:
    meta: dict[str, object] = {}
    if doc_title is not None:
        meta["document_title"] = doc_title
    return _node(
        node_id="doc",
        node_type="document",
        heading=doc_title,
        text=None,
        article_ref=None,
        page_start=1,
        page_end=10,
        children=children,
        source_block_ids=[],
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# 1. Article with body produces chunk with article_ref
# ---------------------------------------------------------------------------


class TestArticles:
    def test_article_with_body_has_article_ref(self) -> None:
        """#1 — Article with body produces chunk with article_ref."""
        article: StructuralNode = _node(
            node_id="article-1",
            node_type="article",
            heading="Artículo 1.- Ámbito",
            text="Esta ley es de observancia general.",
            article_ref="1",
            page_start=1,
            page_end=1,
            children=None,
            source_block_ids=["b1"],
        )
        structure: DocumentStructure = _make_structure(_doc_root([article]))
        chunks: list[dict[str, object]] = project_structure_to_chunks(structure)

        assert len(chunks) == 1
        assert chunks[0]["chunk_type"] == "article"
        assert chunks[0]["article_ref"] == "1"
        assert chunks[0]["heading"] == "Artículo 1.- Ámbito"
        assert "observancia general" in str(chunks[0]["text"])

    def test_article_with_paragraphs_aggregates_text(self) -> None:
        """#2 — Article with paragraphs aggregates text correctly."""
        para1: StructuralNode = _node(
            "p1", "paragraph", None, "Primer párrafo del artículo.",
            None, 1, 1, None, ["b2"],
        )
        para2: StructuralNode = _node(
            "p2", "paragraph", None, "Segundo párrafo del artículo.",
            None, 2, 2, None, ["b3"],
        )
        article: StructuralNode = _node(
            "article-2", "article", "Artículo 2.- Definiciones",
            "Texto propio del artículo.",
            "2", 1, 2, [para1, para2], ["b1"],
        )
        structure: DocumentStructure = _make_structure(_doc_root([article]))
        chunks: list[dict[str, object]] = project_structure_to_chunks(structure)

        article_chunks: list[dict[str, object]] = [
            c for c in chunks if c["chunk_type"] == "article"
        ]
        assert len(article_chunks) == 1
        text: str = str(article_chunks[0]["text"])
        assert "Texto propio del artículo" in text
        assert "Primer párrafo" in text
        assert "Segundo párrafo" in text

    def test_article_with_fraction_produces_coherent_chunks(self) -> None:
        """#3 — Article with fraction produces coherent chunks."""
        fraction: StructuralNode = _node(
            "f1", "fraction", None, "I. Presentar declaración anual.",
            None, 1, 1, None, ["b3"],
        )
        article: StructuralNode = _node(
            "article-3", "article", "Artículo 3.- Obligaciones",
            "Son obligaciones de los contribuyentes:",
            "3", 1, 1, [fraction], ["b1"],
        )
        structure: DocumentStructure = _make_structure(_doc_root([article]))
        chunks: list[dict[str, object]] = project_structure_to_chunks(structure)

        types: list[str] = [str(c["chunk_type"]) for c in chunks]
        assert "article" in types
        assert "fraction" in types

        fraction_chunk: dict[str, object] = next(
            c for c in chunks if c["chunk_type"] == "fraction"
        )
        assert "declaración anual" in str(fraction_chunk["text"])
        assert fraction_chunk["article_ref"] == "3"


# ---------------------------------------------------------------------------
# 4-6. Tables
# ---------------------------------------------------------------------------


class TestTables:
    def test_table_inside_article_atomic(self) -> None:
        """#4 — Table inside article produces independent atomic chunk."""
        table: StructuralNode = _node(
            "table-1", "table", None, "Concepto | Tasa | Monto",
            None, 2, 2, None, ["b3"],
        )
        article: StructuralNode = _node(
            "article-10", "article", "Artículo 10.- Tarifas",
            "Las tarifas son las siguientes:",
            "10", 1, 2, [table], ["b1"],
        )
        structure: DocumentStructure = _make_structure(_doc_root([article]))
        chunks: list[dict[str, object]] = project_structure_to_chunks(structure)

        table_chunks: list[dict[str, object]] = [
            c for c in chunks if c["chunk_type"] == "table"
        ]
        assert len(table_chunks) == 1
        assert "Concepto" in str(table_chunks[0]["text"])
        assert table_chunks[0]["article_ref"] == "10"

    def test_table_outside_article_not_lost(self) -> None:
        """#5 — Table outside article is not lost."""
        table: StructuralNode = _node(
            "table-1", "table", None, "Tabla general de valores",
            None, 1, 1, None, ["b1"],
        )
        structure: DocumentStructure = _make_structure(_doc_root([table]))
        chunks: list[dict[str, object]] = project_structure_to_chunks(structure)

        assert len(chunks) == 1
        assert chunks[0]["chunk_type"] == "table"

    def test_table_preserves_source_and_pages(self) -> None:
        """#6 — Table preserves source_block_ids and page range."""
        table: StructuralNode = _node(
            "table-1", "table", None, "Datos tabulares",
            None, 3, 5, None, ["b10", "b11"],
        )
        structure: DocumentStructure = _make_structure(_doc_root([table]))
        chunks: list[dict[str, object]] = project_structure_to_chunks(structure)

        assert chunks[0]["page_start"] == 3
        assert chunks[0]["page_end"] == 5
        assert "b10" in chunks[0]["source_block_ids"]
        assert "b11" in chunks[0]["source_block_ids"]


# ---------------------------------------------------------------------------
# 7-8. Notes
# ---------------------------------------------------------------------------


class TestNotes:
    def test_editorial_note_projected_separately(self) -> None:
        """#7 — editorial_note is projected separately from normative text."""
        note: StructuralNode = _node(
            "note-1", "note", None,
            "ACLARACIÓN: Texto corregido conforme a Fe de Erratas.",
            None, 1, 1, None, ["b5"],
        )
        article: StructuralNode = _node(
            "article-2", "article", "Artículo 2.- Definiciones",
            "Texto del artículo.",
            "2", 1, 1, [note], ["b1"],
        )
        structure: DocumentStructure = _make_structure(_doc_root([article]))
        chunks: list[dict[str, object]] = project_structure_to_chunks(structure)

        note_chunks: list[dict[str, object]] = [
            c for c in chunks if c["chunk_type"] == "boxed_note"
        ]
        assert len(note_chunks) == 1
        assert "Fe de Erratas" in str(note_chunks[0]["text"])

    def test_note_does_not_contaminate_article(self) -> None:
        """#8 — note does not contaminate article's main chunk text."""
        note: StructuralNode = _node(
            "note-1", "note", None, "Nota editorial interna.",
            None, 1, 1, None, ["b5"],
        )
        article: StructuralNode = _node(
            "article-5", "article", "Artículo 5.- Objeto",
            "Texto normativo puro.",
            "5", 1, 1, [note], ["b1"],
        )
        structure: DocumentStructure = _make_structure(_doc_root([article]))
        chunks: list[dict[str, object]] = project_structure_to_chunks(structure)

        article_chunk: dict[str, object] = next(
            c for c in chunks if c["chunk_type"] == "article"
        )
        assert "Nota editorial" not in str(article_chunk["text"])


# ---------------------------------------------------------------------------
# 9-10. Transitorios
# ---------------------------------------------------------------------------


class TestTransitorios:
    def test_transitory_items_produce_chunks(self) -> None:
        """#9 — Primero.- / Segundo.- produce useful transitory chunks."""
        item1: StructuralNode = _node(
            "transitory-primero", "transitory", "Primero.- Vigencia",
            "Primero.- El presente decreto entrará en vigor.",
            "primero", 5, 5, None, ["b2"],
        )
        item2: StructuralNode = _node(
            "transitory-segundo", "transitory", "Segundo.- Derogaciones",
            "Segundo.- Se derogan las disposiciones anteriores.",
            "segundo", 5, 5, None, ["b3"],
        )
        container: StructuralNode = _node(
            "transitory-1", "transitory", "TRANSITORIOS", "TRANSITORIOS",
            None, 5, 5, [item1, item2], ["b1"],
        )
        structure: DocumentStructure = _make_structure(_doc_root([container]))
        chunks: list[dict[str, object]] = project_structure_to_chunks(structure)

        trans_chunks: list[dict[str, object]] = [
            c for c in chunks if c["chunk_type"] == "transitory"
        ]
        assert len(trans_chunks) == 2
        assert "entrará en vigor" in str(trans_chunks[0]["text"])
        assert "derogan" in str(trans_chunks[1]["text"])

    def test_transitory_items_preserve_context(self) -> None:
        """#10 — Transitory items preserve transitory context in metadata."""
        item: StructuralNode = _node(
            "transitory-primero", "transitory", "Primero.- Vigencia",
            "Primero.- Este decreto entra en vigor mañana.",
            "primero", 5, 5, None, ["b2"],
        )
        container: StructuralNode = _node(
            "transitory-1", "transitory", "TRANSITORIOS", "TRANSITORIOS",
            None, 5, 5, [item], ["b1"],
        )
        structure: DocumentStructure = _make_structure(_doc_root([container]))
        chunks: list[dict[str, object]] = project_structure_to_chunks(structure)

        assert len(chunks) == 1
        meta: dict[str, object] = chunks[0]["metadata"]  # type: ignore[assignment]
        assert meta.get("is_transitory") is True
        assert chunks[0]["article_ref"] == "primero"


# ---------------------------------------------------------------------------
# 11. Anexos
# ---------------------------------------------------------------------------


class TestAnexos:
    def test_annex_preserves_is_annex(self) -> None:
        """#11 — Annex is projected and preserves is_annex=True in metadata."""
        annex_body: StructuralNode = _node(
            "paragraph-1", "paragraph", None, "Contenido del anexo I.",
            None, 8, 8, None, ["b2"], {"is_annex": True},
        )
        annex_section: StructuralNode = _node(
            "annex-i", "section", "Anexo I", "Anexo I",
            None, 8, 8, [annex_body], ["b1"], {"is_annex": True},
        )
        structure: DocumentStructure = _make_structure(_doc_root([annex_section]))
        chunks: list[dict[str, object]] = project_structure_to_chunks(structure)

        assert len(chunks) >= 1
        annex_chunks: list[dict[str, object]] = [
            c for c in chunks
            if c["metadata"].get("is_annex") is True  # type: ignore[union-attr]
        ]
        assert len(annex_chunks) >= 1


# ---------------------------------------------------------------------------
# 12-14. Order and traceability
# ---------------------------------------------------------------------------


class TestOrderTraceability:
    def test_chunks_in_document_order(self) -> None:
        """#12 — Chunks come out in correct document order."""
        art1: StructuralNode = _node(
            "article-1", "article", "Artículo 1", "Primero.",
            "1", 1, 1, None, ["b1"],
        )
        art2: StructuralNode = _node(
            "article-2", "article", "Artículo 2", "Segundo.",
            "2", 2, 2, None, ["b2"],
        )
        trans_item: StructuralNode = _node(
            "transitory-primero", "transitory", "Primero.-",
            "Primero.- Vigencia.", "primero", 3, 3, None, ["b4"],
        )
        trans_container: StructuralNode = _node(
            "transitory-1", "transitory", "TRANSITORIOS", "TRANSITORIOS",
            None, 3, 3, [trans_item], ["b3"],
        )

        structure: DocumentStructure = _make_structure(
            _doc_root([art1, art2, trans_container]),
        )
        chunks: list[dict[str, object]] = project_structure_to_chunks(structure)

        types: list[str] = [str(c["chunk_type"]) for c in chunks]
        assert types == ["article", "article", "transitory"]

    def test_all_chunks_have_source_block_ids(self) -> None:
        """#13 — All chunks preserve source_block_ids."""
        note: StructuralNode = _node(
            "note-1", "note", None, "Nota.",
            None, 1, 1, None, ["b2"],
        )
        article: StructuralNode = _node(
            "article-1", "article", "Artículo 1", "Texto.",
            "1", 1, 1, [note], ["b1"],
        )
        structure: DocumentStructure = _make_structure(_doc_root([article]))
        chunks: list[dict[str, object]] = project_structure_to_chunks(structure)

        for chunk in chunks:
            assert isinstance(chunk["source_block_ids"], list)
            assert len(chunk["source_block_ids"]) > 0  # type: ignore[arg-type]

    def test_all_chunks_have_coherent_pages(self) -> None:
        """#14 — All chunks have coherent page_start/page_end."""
        para: StructuralNode = _node(
            "p1", "paragraph", None, "Texto.",
            None, 2, 3, None, ["b2"],
        )
        article: StructuralNode = _node(
            "article-1", "article", "Artículo 1", "Texto art.",
            "1", 1, 3, [para], ["b1"],
        )
        structure: DocumentStructure = _make_structure(_doc_root([article]))
        chunks: list[dict[str, object]] = project_structure_to_chunks(structure)

        for chunk in chunks:
            ps: int | None = chunk.get("page_start")  # type: ignore[assignment]
            pe: int | None = chunk.get("page_end")  # type: ignore[assignment]
            if ps is not None and pe is not None:
                assert ps <= pe, f"page_start={ps} > page_end={pe}"


# ---------------------------------------------------------------------------
# 15. Quality propagation
# ---------------------------------------------------------------------------


class TestQualityPropagation:
    def test_quality_score_propagated(self) -> None:
        """#15 — quality_score and quality_severity propagate to metadata."""
        article: StructuralNode = _node(
            "article-1", "article", "Artículo 1", "Texto.",
            "1", 1, 1, None, ["b1"],
        )
        quality_report: dict[str, object] = {
            "quality_score": 0.92,
            "summary": {"severity": "low", "passed": True, "reasons": []},
        }
        structure: DocumentStructure = _make_structure(
            _doc_root([article]),
            quality_report=quality_report,
        )
        chunks: list[dict[str, object]] = project_structure_to_chunks(structure)

        assert len(chunks) == 1
        meta: dict[str, object] = chunks[0]["metadata"]  # type: ignore[assignment]
        assert meta["quality_score"] == 0.92
        assert meta["quality_severity"] == "low"


# ---------------------------------------------------------------------------
# 16-20. Compatibility and robustness
# ---------------------------------------------------------------------------


class TestCompatibilityRobustness:
    def test_empty_document_empty_list(self) -> None:
        """#16 — Empty document produces empty list without crash."""
        root: StructuralNode = _node(
            "doc", "document", None, None,
            None, None, None, [], [],
        )
        structure: DocumentStructure = _make_structure(root)
        chunks: list[dict[str, object]] = project_structure_to_chunks(structure)

        assert chunks == []

    def test_output_json_serializable(self) -> None:
        """#17 — Full output is JSON-serializable."""
        fraction: StructuralNode = _node(
            "f1", "fraction", None, "I. Fracción.",
            None, 1, 1, None, ["b3"],
        )
        note: StructuralNode = _node(
            "note-1", "note", None, "Nota editorial.",
            None, 1, 1, None, ["b4"],
        )
        table: StructuralNode = _node(
            "table-1", "table", None, "Datos.",
            None, 1, 1, None, ["b5"],
        )
        article: StructuralNode = _node(
            "article-1", "article", "Artículo 1", "Texto.",
            "1", 1, 1, [fraction, note, table], ["b1"],
        )
        trans_item: StructuralNode = _node(
            "transitory-primero", "transitory", "Primero.-",
            "Primero.- Vigencia.", "primero", 2, 2, None, ["b6"],
        )
        trans_container: StructuralNode = _node(
            "transitory-1", "transitory", "TRANSITORIOS", "TRANSITORIOS",
            None, 2, 2, [trans_item], ["b7"],
        )
        annex_body: StructuralNode = _node(
            "p-annex", "paragraph", None, "Texto anexo.",
            None, 3, 3, None, ["b9"], {"is_annex": True},
        )
        annex: StructuralNode = _node(
            "annex-i", "section", "Anexo I", "Anexo I",
            None, 3, 3, [annex_body], ["b8"], {"is_annex": True},
        )
        root: StructuralNode = _doc_root(
            [article, trans_container, annex], doc_title="LEY DE PRUEBA",
        )
        structure: DocumentStructure = _make_structure(
            root,
            quality_report={"quality_score": 0.85, "summary": {"severity": "low"}},
        )
        chunks: list[dict[str, object]] = project_structure_to_chunks(structure)

        json_str: str = json.dumps(chunks, ensure_ascii=False)
        parsed: list[dict[str, Any]] = json.loads(json_str)

        assert len(parsed) == len(chunks)
        for item in parsed:
            assert "chunk_type" in item
            assert "text" in item
            assert "source_block_ids" in item
            assert "metadata" in item

    def test_no_headers_footers_index_projected(self) -> None:
        """#18 — Headers, footers, and index_blocks are not projected."""
        header: StructuralNode = _node(
            "h1", "page_header", None, "DIARIO OFICIAL",
            None, 1, 1, None, ["bh"],
        )
        footer: StructuralNode = _node(
            "f1", "page_footer", None, "Página 1",
            None, 1, 1, None, ["bf"],
        )
        index: StructuralNode = _node(
            "idx1", "index_block", None, "Índice general",
            None, 1, 1, None, ["bi"],
        )
        article: StructuralNode = _node(
            "article-1", "article", "Artículo 1", "Texto real.",
            "1", 1, 1, None, ["b1"],
        )
        # These excluded types would not normally be in the tree,
        # but we verify the projector defends against them
        root: StructuralNode = _doc_root([header, footer, index, article])
        structure: DocumentStructure = _make_structure(root)
        chunks: list[dict[str, object]] = project_structure_to_chunks(structure)

        chunk_types: set[str] = {str(c["chunk_type"]) for c in chunks}
        assert "page_header" not in chunk_types
        assert "page_footer" not in chunk_types
        assert "index_block" not in chunk_types
        assert "article" in chunk_types

    def test_no_empty_text_chunks(self) -> None:
        """#19 — Chunks do not have empty text except justified exceptions."""
        article: StructuralNode = _node(
            "article-1", "article", "Artículo 1", "Texto completo.",
            "1", 1, 1, None, ["b1"],
        )
        # Table with empty text still gets projected (justified exception)
        table: StructuralNode = _node(
            "table-1", "table", None, "",
            None, 1, 1, None, ["b2"],
        )
        structure: DocumentStructure = _make_structure(
            _doc_root([article, table]),
        )
        chunks: list[dict[str, object]] = project_structure_to_chunks(structure)

        non_table_chunks: list[dict[str, object]] = [
            c for c in chunks if c["chunk_type"] != "table"
        ]
        for chunk in non_table_chunks:
            text: str = str(chunk["text"])
            assert len(text.strip()) > 0, (
                f"Non-table chunk has empty text: {chunk['chunk_type']}"
            )

    def test_unknown_paragraph_preserves_content(self) -> None:
        """#20 — unknown->paragraph at root does not lose content."""
        para: StructuralNode = _node(
            "paragraph-1", "paragraph", None,
            "Texto ambiguo proveniente de bloque desconocido.",
            None, 1, 1, None, ["b1"],
        )
        structure: DocumentStructure = _make_structure(_doc_root([para]))
        chunks: list[dict[str, object]] = project_structure_to_chunks(structure)

        assert len(chunks) == 1
        assert "ambiguo" in str(chunks[0]["text"])
        assert chunks[0]["chunk_type"] == "paragraph"


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_article_with_inciso_no_fraction(self) -> None:
        """Incisos directly under article are folded into article chunk."""
        inciso1: StructuralNode = _node(
            "i1", "inciso", None, "a) Primer inciso.",
            None, 1, 1, None, ["b2"],
        )
        inciso2: StructuralNode = _node(
            "i2", "inciso", None, "b) Segundo inciso.",
            None, 1, 1, None, ["b3"],
        )
        article: StructuralNode = _node(
            "article-7", "article", "Artículo 7.- Sanciones",
            "Las sanciones serán:",
            "7", 1, 1, [inciso1, inciso2], ["b1"],
        )
        structure: DocumentStructure = _make_structure(_doc_root([article]))
        chunks: list[dict[str, object]] = project_structure_to_chunks(structure)

        article_chunks: list[dict[str, object]] = [
            c for c in chunks if c["chunk_type"] == "article"
        ]
        assert len(article_chunks) == 1
        text: str = str(article_chunks[0]["text"])
        assert "Primer inciso" in text
        assert "Segundo inciso" in text

    def test_fraction_with_incisos_folded(self) -> None:
        """Incisos inside a fraction are folded into the fraction chunk."""
        inciso: StructuralNode = _node(
            "i1", "inciso", None, "a) Identificación oficial.",
            None, 1, 1, None, ["b4"],
        )
        fraction: StructuralNode = _node(
            "f1", "fraction", None, "I. Los contribuyentes deberán:",
            None, 1, 1, [inciso], ["b3"],
        )
        article: StructuralNode = _node(
            "article-4", "article", "Artículo 4.- Requisitos",
            "Requisitos:",
            "4", 1, 1, [fraction], ["b1"],
        )
        structure: DocumentStructure = _make_structure(_doc_root([article]))
        chunks: list[dict[str, object]] = project_structure_to_chunks(structure)

        fraction_chunks: list[dict[str, object]] = [
            c for c in chunks if c["chunk_type"] == "fraction"
        ]
        assert len(fraction_chunks) == 1
        text = str(fraction_chunks[0]["text"])
        assert "contribuyentes" in text
        assert "Identificación oficial" in text

    def test_hierarchy_context_propagated(self) -> None:
        """Structural containers propagate hierarchy context to chunks."""
        article: StructuralNode = _node(
            "article-1", "article", "Artículo 1", "Texto.",
            "1", 1, 1, None, ["b4"],
        )
        chapter: StructuralNode = _node(
            "chapter-i", "chapter", "Capítulo I Principios", None,
            None, 1, 1, [article], ["b3"],
        )
        title: StructuralNode = _node(
            "title-i", "title", "Título I Generalidades", None,
            None, 1, 1, [chapter], ["b2"],
        )
        root: StructuralNode = _doc_root([title], doc_title="LEY DE PRUEBA")
        structure: DocumentStructure = _make_structure(root)
        chunks: list[dict[str, object]] = project_structure_to_chunks(structure)

        assert len(chunks) == 1
        meta: dict[str, object] = chunks[0]["metadata"]  # type: ignore[assignment]
        assert meta["document_title"] == "LEY DE PRUEBA"
        assert meta["title_heading"] == "Título I Generalidades"
        assert meta["chapter_heading"] == "Capítulo I Principios"

    def test_transitory_with_paragraphs(self) -> None:
        """Transitory item with paragraph children aggregates text."""
        para: StructuralNode = _node(
            "p1", "paragraph", None,
            "Este decreto entrará en vigor al día siguiente.",
            None, 5, 5, None, ["b3"],
        )
        item: StructuralNode = _node(
            "transitory-primero", "transitory", "Primero.- Vigencia",
            "Primero.- Vigencia.",
            "primero", 5, 5, [para], ["b2"],
        )
        container: StructuralNode = _node(
            "transitory-1", "transitory", "TRANSITORIOS", "TRANSITORIOS",
            None, 5, 5, [item], ["b1"],
        )
        structure: DocumentStructure = _make_structure(_doc_root([container]))
        chunks: list[dict[str, object]] = project_structure_to_chunks(structure)

        assert len(chunks) == 1
        text: str = str(chunks[0]["text"])
        assert "Vigencia" in text
        assert "entrará en vigor" in text

    def test_annex_paragraph_typed_as_annex(self) -> None:
        """Paragraphs inside annex section get chunk_type='annex'."""
        para: StructuralNode = _node(
            "p1", "paragraph", None, "Tabla de conversión.",
            None, 9, 9, None, ["b2"], {"is_annex": True},
        )
        annex: StructuralNode = _node(
            "annex-i", "section", "Anexo I", None,
            None, 9, 9, [para], ["b1"], {"is_annex": True},
        )
        structure: DocumentStructure = _make_structure(_doc_root([annex]))
        chunks: list[dict[str, object]] = project_structure_to_chunks(structure)

        assert len(chunks) == 1
        assert chunks[0]["chunk_type"] == "annex"
        assert chunks[0]["metadata"]["is_annex"] is True  # type: ignore[index]

    def test_multiple_chunk_types_required_keys(self) -> None:
        """Every projected chunk has the minimum required keys."""
        fraction: StructuralNode = _node(
            "f1", "fraction", None, "I. Fracción.", None, 1, 1, None, ["b2"],
        )
        note: StructuralNode = _node(
            "note-1", "note", None, "Nota.", None, 1, 1, None, ["b3"],
        )
        table: StructuralNode = _node(
            "table-1", "table", None, "Tabla.", None, 1, 1, None, ["b4"],
        )
        article: StructuralNode = _node(
            "article-1", "article", "Art 1", "Texto.",
            "1", 1, 1, [fraction, note, table], ["b1"],
        )
        trans_item: StructuralNode = _node(
            "transitory-primero", "transitory", "Primero.-",
            "Vigencia.", "primero", 2, 2, None, ["b5"],
        )
        trans_cont: StructuralNode = _node(
            "transitory-1", "transitory", "TRANSITORIOS", "TRANSITORIOS",
            None, 2, 2, [trans_item], ["b6"],
        )

        structure: DocumentStructure = _make_structure(
            _doc_root([article, trans_cont]),
        )
        chunks: list[dict[str, object]] = project_structure_to_chunks(structure)

        required_keys: set[str] = {
            "chunk_type", "heading", "text", "article_ref",
            "page_start", "page_end", "source_block_ids", "metadata",
        }
        for chunk in chunks:
            missing: set[str] = required_keys - set(chunk.keys())
            assert not missing, f"Chunk missing keys: {missing}"
