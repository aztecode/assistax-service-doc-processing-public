"""
Unit tests for the quality validator v2 (Phase 5).

Covers 20 required scenarios:

 1. heading with TABLE or [TABLE_1] lowers quality strongly
 2. clean document does not trigger visible table token check
 3. reasonable article sequence 1,2,3,4 is healthy
 4. absurd jumps like 62,47,59,63 are penalized
 5. articles 14,14-A,14 Bis,15 not penalized as severe break
 6. TOC with DIARIO OFICIAL or CÁMARA DE DIPUTADOS is penalized
 7. clean TOC does not trigger header/footer bleed
 8. navigable heading like '24 de diciembre de 2024' counts as false positive
 9. document without date headings does not produce false positives
10. table without pages or source blocks counts as orphan
11. well-located table does not count as orphan
12. structure with many degraded/fallback nodes lowers quality
13. clean structure does not penalize unknown ratio
14. duplicates in TOC raise the ratio
15. clean TOC has zero or near-zero duplicate ratio
16. articles with complete refs have good coverage
17. many articles without article_ref lower coverage score
18. clearly bad document falls below 0.70
19. good document scores >= 0.85
20. full output is JSON-serializable
"""
from __future__ import annotations

import json

import pytest

from pipeline.layout_models import DocumentStructure, StructuralNode
from pipeline.quality_validator_v2 import (
    compute_quality_score,
    validate_document_structure,
)


# ---------------------------------------------------------------------------
# Helpers
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
        metadata={},
    )


def _article(
    ref: str,
    idx: int,
    page: int = 1,
) -> StructuralNode:
    return _node(
        node_id=f"article-{ref}",
        node_type="article",
        heading=f"Artículo {ref}",
        text=f"Contenido del artículo {ref}.",
        article_ref=ref,
        page_start=page,
        page_end=page,
        children=None,
        source_block_ids=[f"b{idx}"],
    )


def _article_no_ref(idx: int, page: int = 1) -> StructuralNode:
    return _node(
        node_id=f"article-{idx}",
        node_type="article",
        heading=f"Artículo sin referencia {idx}",
        text="Texto genérico.",
        article_ref=None,
        page_start=page,
        page_end=page,
        children=None,
        source_block_ids=[f"b{idx}"],
    )


def _make_structure(
    root: StructuralNode,
    toc: list[dict[str, object]] | None,
) -> DocumentStructure:
    if toc is None:
        toc = _build_simple_toc(root)
    return DocumentStructure(
        root=root,
        toc=toc,
        sections=[],
        quality_report={},
        metadata={},
    )


def _build_simple_toc(root: StructuralNode) -> list[dict[str, object]]:
    navigable_types: frozenset[str] = frozenset({
        "book", "title", "chapter", "section", "article", "transitory",
    })
    toc: list[dict[str, object]] = []
    _walk_toc(root, navigable_types, toc)
    return toc


def _walk_toc(
    node: StructuralNode,
    nav_types: frozenset[str],
    toc: list[dict[str, object]],
) -> None:
    if node.node_type != "document" and node.node_type in nav_types:
        toc.append({
            "node_id": node.node_id,
            "node_type": node.node_type,
            "heading": node.heading,
            "page_start": node.page_start,
        })
    for child in node.children:
        _walk_toc(child, nav_types, toc)


def _clean_structure() -> DocumentStructure:
    """Builds a well-formed structure for 'clean' test cases."""
    articles: list[StructuralNode] = [
        _article("1", 1, page=1),
        _article("2", 2, page=2),
        _article("3", 3, page=3),
        _article("4", 4, page=4),
    ]
    chapter: StructuralNode = _node(
        node_id="chapter-i",
        node_type="chapter",
        heading="Capítulo I Disposiciones generales",
        text=None,
        article_ref=None,
        page_start=1,
        page_end=4,
        children=articles,
        source_block_ids=["b0"],
    )
    root: StructuralNode = _node(
        node_id="doc",
        node_type="document",
        heading=None,
        text=None,
        article_ref=None,
        page_start=1,
        page_end=4,
        children=[chapter],
        source_block_ids=[],
    )
    return _make_structure(root, None)


# ---------------------------------------------------------------------------
# 1-2. Visible table tokens
# ---------------------------------------------------------------------------


class TestVisibleTableTokens:
    def test_heading_with_table_token_lowers_quality(self) -> None:
        """#1 — heading with TABLE or [TABLE_1] lowers quality strongly."""
        article: StructuralNode = _node(
            node_id="article-1",
            node_type="article",
            heading="Artículo 1 [TABLE_1] Disposiciones",
            text="Contenido.",
            article_ref="1",
            page_start=1,
            page_end=1,
            children=None,
            source_block_ids=["b1"],
        )
        root: StructuralNode = _node(
            node_id="doc", node_type="document", heading=None, text=None,
            article_ref=None, page_start=1, page_end=1,
            children=[article], source_block_ids=[],
        )
        structure: DocumentStructure = _make_structure(root, None)
        report: dict[str, object] = validate_document_structure(structure)
        checks: dict[str, object] = report["checks"]  # type: ignore[assignment]
        table_check: dict[str, object] = checks["has_visible_table_tokens"]  # type: ignore[assignment]

        assert table_check["passed"] is False
        assert int(table_check["count"]) >= 1  # type: ignore[arg-unpack]
        assert float(report["quality_score"]) < 0.85  # type: ignore[arg-unpack]

    def test_clean_document_no_table_tokens(self) -> None:
        """#2 — clean document does not trigger visible table token check."""
        structure: DocumentStructure = _clean_structure()
        report: dict[str, object] = validate_document_structure(structure)
        checks: dict[str, object] = report["checks"]  # type: ignore[assignment]
        table_check: dict[str, object] = checks["has_visible_table_tokens"]  # type: ignore[assignment]

        assert table_check["passed"] is True
        assert int(table_check["count"]) == 0  # type: ignore[arg-unpack]


# ---------------------------------------------------------------------------
# 3-5. Article sequence health
# ---------------------------------------------------------------------------


class TestArticleSequenceHealth:
    def test_reasonable_sequence_is_healthy(self) -> None:
        """#3 — sequence 1,2,3,4 is healthy."""
        structure: DocumentStructure = _clean_structure()
        report: dict[str, object] = validate_document_structure(structure)
        checks: dict[str, object] = report["checks"]  # type: ignore[assignment]
        seq_check: dict[str, object] = checks["article_sequence_health"]  # type: ignore[assignment]

        assert seq_check["passed"] is True
        assert float(seq_check["disorder_ratio"]) == 0.0  # type: ignore[arg-unpack]

    def test_absurd_jumps_penalized(self) -> None:
        """#4 — absurd jumps like 62,47,59,63 are penalized."""
        articles: list[StructuralNode] = [
            _article("62", 1),
            _article("47", 2),
            _article("59", 3),
            _article("63", 4),
        ]
        root: StructuralNode = _node(
            node_id="doc", node_type="document", heading=None, text=None,
            article_ref=None, page_start=1, page_end=1,
            children=articles, source_block_ids=[],
        )
        structure: DocumentStructure = _make_structure(root, None)
        report: dict[str, object] = validate_document_structure(structure)
        checks: dict[str, object] = report["checks"]  # type: ignore[assignment]
        seq_check: dict[str, object] = checks["article_sequence_health"]  # type: ignore[assignment]

        assert seq_check["passed"] is False
        assert float(seq_check["disorder_ratio"]) > 0.0  # type: ignore[arg-unpack]

    def test_bis_ter_not_penalized(self) -> None:
        """#5 — articles 14,14-A,14 Bis,15 not penalized as severe break."""
        articles: list[StructuralNode] = [
            _article("14", 1),
            _article("14-A", 2),
            _article("14Bis", 3),
            _article("15", 4),
        ]
        root: StructuralNode = _node(
            node_id="doc", node_type="document", heading=None, text=None,
            article_ref=None, page_start=1, page_end=1,
            children=articles, source_block_ids=[],
        )
        structure: DocumentStructure = _make_structure(root, None)
        report: dict[str, object] = validate_document_structure(structure)
        checks: dict[str, object] = report["checks"]  # type: ignore[assignment]
        seq_check: dict[str, object] = checks["article_sequence_health"]  # type: ignore[assignment]

        assert seq_check["passed"] is True


# ---------------------------------------------------------------------------
# 6-7. Header/footer bleed
# ---------------------------------------------------------------------------


class TestHeaderFooterBleed:
    def test_toc_with_diario_oficial_penalized(self) -> None:
        """#6 — TOC with DIARIO OFICIAL or CÁMARA DE DIPUTADOS is penalized."""
        article: StructuralNode = _article("1", 1)
        bleed_node: StructuralNode = _node(
            node_id="section-bleed",
            node_type="section",
            heading="DIARIO OFICIAL DE LA FEDERACIÓN",
            text=None,
            article_ref=None,
            page_start=1,
            page_end=1,
            children=None,
            source_block_ids=["bx"],
        )
        root: StructuralNode = _node(
            node_id="doc", node_type="document", heading=None, text=None,
            article_ref=None, page_start=1, page_end=1,
            children=[bleed_node, article], source_block_ids=[],
        )
        structure: DocumentStructure = _make_structure(root, None)
        report: dict[str, object] = validate_document_structure(structure)
        checks: dict[str, object] = report["checks"]  # type: ignore[assignment]
        bleed_check: dict[str, object] = checks["header_footer_bleed"]  # type: ignore[assignment]

        assert bleed_check["passed"] is False
        assert int(bleed_check["count"]) >= 1  # type: ignore[arg-unpack]

    def test_clean_toc_no_bleed(self) -> None:
        """#7 — clean TOC does not trigger header/footer bleed."""
        structure: DocumentStructure = _clean_structure()
        report: dict[str, object] = validate_document_structure(structure)
        checks: dict[str, object] = report["checks"]  # type: ignore[assignment]
        bleed_check: dict[str, object] = checks["header_footer_bleed"]  # type: ignore[assignment]

        assert bleed_check["passed"] is True
        assert int(bleed_check["count"]) == 0  # type: ignore[arg-unpack]


# ---------------------------------------------------------------------------
# 8-9. Date heading false positives
# ---------------------------------------------------------------------------


class TestDateHeadingFalsePositives:
    def test_date_heading_counted_as_false_positive(self) -> None:
        """#8 — navigable heading like '24 de diciembre de 2024' counts as false positive."""
        date_node: StructuralNode = _node(
            node_id="section-date",
            node_type="section",
            heading="24 de diciembre de 2024",
            text=None,
            article_ref=None,
            page_start=1,
            page_end=1,
            children=None,
            source_block_ids=["b1"],
        )
        root: StructuralNode = _node(
            node_id="doc", node_type="document", heading=None, text=None,
            article_ref=None, page_start=1, page_end=1,
            children=[date_node], source_block_ids=[],
        )
        structure: DocumentStructure = _make_structure(root, None)
        report: dict[str, object] = validate_document_structure(structure)
        checks: dict[str, object] = report["checks"]  # type: ignore[assignment]
        date_check: dict[str, object] = checks["date_heading_false_positive_count"]  # type: ignore[assignment]

        assert date_check["passed"] is False
        assert int(date_check["count"]) >= 1  # type: ignore[arg-unpack]

    def test_no_date_headings_no_false_positives(self) -> None:
        """#9 — document without date headings does not produce false positives."""
        structure: DocumentStructure = _clean_structure()
        report: dict[str, object] = validate_document_structure(structure)
        checks: dict[str, object] = report["checks"]  # type: ignore[assignment]
        date_check: dict[str, object] = checks["date_heading_false_positive_count"]  # type: ignore[assignment]

        assert date_check["passed"] is True
        assert int(date_check["count"]) == 0  # type: ignore[arg-unpack]


# ---------------------------------------------------------------------------
# 10-11. Orphan tables
# ---------------------------------------------------------------------------


class TestOrphanTables:
    def test_table_without_pages_is_orphan(self) -> None:
        """#10 — table without pages or source blocks counts as orphan."""
        orphan_table: StructuralNode = _node(
            node_id="table-1",
            node_type="table",
            heading=None,
            text="Tabla de valores",
            article_ref=None,
            page_start=None,
            page_end=None,
            children=None,
            source_block_ids=[],
        )
        root: StructuralNode = _node(
            node_id="doc", node_type="document", heading=None, text=None,
            article_ref=None, page_start=1, page_end=1,
            children=[orphan_table], source_block_ids=[],
        )
        structure: DocumentStructure = _make_structure(root, None)
        report: dict[str, object] = validate_document_structure(structure)
        checks: dict[str, object] = report["checks"]  # type: ignore[assignment]
        orphan_check: dict[str, object] = checks["orphan_tables_count"]  # type: ignore[assignment]

        assert orphan_check["passed"] is False
        assert int(orphan_check["orphan_count"]) == 1  # type: ignore[arg-unpack]

    def test_well_located_table_not_orphan(self) -> None:
        """#11 — well-located table does not count as orphan."""
        good_table: StructuralNode = _node(
            node_id="table-1",
            node_type="table",
            heading=None,
            text="Tabla de tarifas",
            article_ref=None,
            page_start=2,
            page_end=3,
            children=None,
            source_block_ids=["b5"],
        )
        article: StructuralNode = _node(
            node_id="article-1",
            node_type="article",
            heading="Artículo 1",
            text="Texto.",
            article_ref="1",
            page_start=1,
            page_end=3,
            children=[good_table],
            source_block_ids=["b1"],
        )
        root: StructuralNode = _node(
            node_id="doc", node_type="document", heading=None, text=None,
            article_ref=None, page_start=1, page_end=3,
            children=[article], source_block_ids=[],
        )
        structure: DocumentStructure = _make_structure(root, None)
        report: dict[str, object] = validate_document_structure(structure)
        checks: dict[str, object] = report["checks"]  # type: ignore[assignment]
        orphan_check: dict[str, object] = checks["orphan_tables_count"]  # type: ignore[assignment]

        assert orphan_check["passed"] is True
        assert int(orphan_check["orphan_count"]) == 0  # type: ignore[arg-unpack]


# ---------------------------------------------------------------------------
# 12-13. Unknown/generic ratio
# ---------------------------------------------------------------------------


class TestUnknownBlockRatio:
    def test_many_degraded_nodes_lower_quality(self) -> None:
        """#12 — structure with many degraded/fallback nodes lowers quality."""
        paragraphs: list[StructuralNode] = [
            _node(
                node_id=f"paragraph-{i}",
                node_type="paragraph",
                heading=None,
                text=f"Texto genérico {i}",
                article_ref=None,
                page_start=1,
                page_end=1,
                children=None,
                source_block_ids=[f"b{i}"],
            )
            for i in range(10)
        ]
        root: StructuralNode = _node(
            node_id="doc", node_type="document", heading=None, text=None,
            article_ref=None, page_start=1, page_end=1,
            children=paragraphs, source_block_ids=[],
        )
        structure: DocumentStructure = _make_structure(root, None)
        report: dict[str, object] = validate_document_structure(structure)
        checks: dict[str, object] = report["checks"]  # type: ignore[assignment]
        unknown_check: dict[str, object] = checks["unknown_block_ratio"]  # type: ignore[assignment]

        assert unknown_check["passed"] is False
        assert float(unknown_check["ratio"]) > 0.0  # type: ignore[arg-unpack]

    def test_clean_structure_no_unknown_penalty(self) -> None:
        """#13 — clean structure does not penalize unknown ratio."""
        structure: DocumentStructure = _clean_structure()
        report: dict[str, object] = validate_document_structure(structure)
        checks: dict[str, object] = report["checks"]  # type: ignore[assignment]
        unknown_check: dict[str, object] = checks["unknown_block_ratio"]  # type: ignore[assignment]

        assert unknown_check["passed"] is True


# ---------------------------------------------------------------------------
# 14-15. TOC duplicate ratio
# ---------------------------------------------------------------------------


class TestTocDuplicateRatio:
    def test_duplicates_in_toc_raise_ratio(self) -> None:
        """#14 — duplicates in TOC raise the ratio."""
        toc: list[dict[str, object]] = [
            {"node_id": "art-1", "node_type": "article", "heading": "Artículo 1", "page_start": 1},
            {"node_id": "art-1", "node_type": "article", "heading": "Artículo 1", "page_start": 1},
            {"node_id": "art-1", "node_type": "article", "heading": "Artículo 1", "page_start": 1},
            {"node_id": "art-2", "node_type": "article", "heading": "Artículo 2", "page_start": 2},
        ]
        root: StructuralNode = _node(
            node_id="doc", node_type="document", heading=None, text=None,
            article_ref=None, page_start=1, page_end=2,
            children=[], source_block_ids=[],
        )
        structure: DocumentStructure = _make_structure(root, toc)
        report: dict[str, object] = validate_document_structure(structure)
        checks: dict[str, object] = report["checks"]  # type: ignore[assignment]
        dup_check: dict[str, object] = checks["toc_duplicate_ratio"]  # type: ignore[assignment]

        assert dup_check["passed"] is False
        assert float(dup_check["ratio"]) > 0.0  # type: ignore[arg-unpack]

    def test_clean_toc_no_duplicates(self) -> None:
        """#15 — clean TOC has zero or near-zero duplicate ratio."""
        structure: DocumentStructure = _clean_structure()
        report: dict[str, object] = validate_document_structure(structure)
        checks: dict[str, object] = report["checks"]  # type: ignore[assignment]
        dup_check: dict[str, object] = checks["toc_duplicate_ratio"]  # type: ignore[assignment]

        assert dup_check["passed"] is True
        assert float(dup_check["ratio"]) == 0.0  # type: ignore[arg-unpack]


# ---------------------------------------------------------------------------
# 16-17. Article ref coverage
# ---------------------------------------------------------------------------


class TestArticleRefCoverage:
    def test_articles_with_complete_refs(self) -> None:
        """#16 — articles with complete refs have good coverage."""
        structure: DocumentStructure = _clean_structure()
        report: dict[str, object] = validate_document_structure(structure)
        checks: dict[str, object] = report["checks"]  # type: ignore[assignment]
        cov_check: dict[str, object] = checks["article_ref_coverage"]  # type: ignore[assignment]

        assert cov_check["passed"] is True
        assert float(cov_check["coverage"]) == 1.0  # type: ignore[arg-unpack]

    def test_many_articles_without_ref_lower_coverage(self) -> None:
        """#17 — many articles without article_ref lower coverage score."""
        articles: list[StructuralNode] = [
            _article_no_ref(1),
            _article_no_ref(2),
            _article_no_ref(3),
            _article_no_ref(4),
            _article_no_ref(5),
            _article("6", 6),
        ]
        root: StructuralNode = _node(
            node_id="doc", node_type="document", heading=None, text=None,
            article_ref=None, page_start=1, page_end=1,
            children=articles, source_block_ids=[],
        )
        structure: DocumentStructure = _make_structure(root, None)
        report: dict[str, object] = validate_document_structure(structure)
        checks: dict[str, object] = report["checks"]  # type: ignore[assignment]
        cov_check: dict[str, object] = checks["article_ref_coverage"]  # type: ignore[assignment]

        assert cov_check["passed"] is False
        assert float(cov_check["coverage"]) < 0.5  # type: ignore[arg-unpack]


# ---------------------------------------------------------------------------
# 18-19. Final score integration
# ---------------------------------------------------------------------------


class TestFinalScore:
    def test_bad_document_below_070(self) -> None:
        """#18 — clearly bad document falls below 0.70."""
        bleed_node: StructuralNode = _node(
            node_id="section-bleed",
            node_type="section",
            heading="DIARIO OFICIAL DE LA FEDERACIÓN",
            text=None,
            article_ref=None,
            page_start=1,
            page_end=1,
            children=None,
            source_block_ids=["bx"],
        )
        table_heading: StructuralNode = _node(
            node_id="article-table",
            node_type="article",
            heading="Artículo [TABLE_1] texto",
            text="[TABLE_2] más basura",
            article_ref=None,
            page_start=1,
            page_end=1,
            children=None,
            source_block_ids=["b2"],
        )
        orphan_table: StructuralNode = _node(
            node_id="table-orphan",
            node_type="table",
            heading=None,
            text="Tabla suelta",
            article_ref=None,
            page_start=None,
            page_end=None,
            children=None,
            source_block_ids=[],
        )
        paragraphs: list[StructuralNode] = [
            _node(
                node_id=f"paragraph-{i}",
                node_type="paragraph",
                heading=None,
                text=f"Texto degradado {i}",
                article_ref=None,
                page_start=1,
                page_end=1,
                children=None,
                source_block_ids=[f"b{i}"],
            )
            for i in range(8)
        ]
        root: StructuralNode = _node(
            node_id="doc", node_type="document", heading=None, text=None,
            article_ref=None, page_start=1, page_end=1,
            children=[bleed_node, table_heading, orphan_table] + paragraphs,
            source_block_ids=[],
        )
        structure: DocumentStructure = _make_structure(root, None)
        report: dict[str, object] = validate_document_structure(structure)

        score: float = float(report["quality_score"])
        assert score < 0.70, f"Expected score < 0.70, got {score}"

        summary: dict[str, object] = report["summary"]  # type: ignore[assignment]
        assert summary["passed"] is False
        assert summary["severity"] == "high"

    def test_good_document_above_085(self) -> None:
        """#19 — good document scores >= 0.85."""
        structure: DocumentStructure = _clean_structure()
        report: dict[str, object] = validate_document_structure(structure)

        score: float = float(report["quality_score"])
        assert score >= 0.85, f"Expected score >= 0.85, got {score}"

        summary: dict[str, object] = report["summary"]  # type: ignore[assignment]
        assert summary["passed"] is True
        assert summary["severity"] == "low"


# ---------------------------------------------------------------------------
# 20. Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_output_serializable_json(self) -> None:
        """#20 — full output is JSON-serializable."""
        structure: DocumentStructure = _clean_structure()
        report: dict[str, object] = validate_document_structure(structure)

        json_str: str = json.dumps(report, ensure_ascii=False)
        parsed: dict[str, object] = json.loads(json_str)

        assert "quality_score" in parsed
        assert "checks" in parsed
        assert "summary" in parsed

        checks: dict[str, object] = parsed["checks"]  # type: ignore[assignment]
        assert "has_visible_table_tokens" in checks
        assert "article_sequence_health" in checks
        assert "header_footer_bleed" in checks
        assert "date_heading_false_positive_count" in checks
        assert "orphan_tables_count" in checks
        assert "unknown_block_ratio" in checks
        assert "toc_duplicate_ratio" in checks
        assert "article_ref_coverage" in checks

        summary: dict[str, object] = parsed["summary"]  # type: ignore[assignment]
        assert "passed" in summary
        assert "severity" in summary
        assert "reasons" in summary


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_document_valid(self) -> None:
        """Empty document returns a valid report with high score."""
        root: StructuralNode = _node(
            node_id="doc", node_type="document", heading=None, text=None,
            article_ref=None, page_start=None, page_end=None,
            children=[], source_block_ids=[],
        )
        structure: DocumentStructure = _make_structure(root, [])
        report: dict[str, object] = validate_document_structure(structure)

        score: float = float(report["quality_score"])
        assert score >= 0.85

    def test_compute_quality_score_standalone(self) -> None:
        """compute_quality_score works with a well-formed report dict."""
        report: dict[str, object] = {
            "checks": {
                "has_visible_table_tokens": {"passed": True, "count": 0},
                "article_sequence_health": {"passed": True, "disorder_ratio": 0.0, "duplicate_ratio": 0.0},
                "header_footer_bleed": {"passed": True, "count": 0},
                "date_heading_false_positive_count": {"passed": True, "count": 0},
                "orphan_tables_count": {"passed": True, "orphan_count": 0, "total_tables": 0},
                "unknown_block_ratio": {"passed": True, "ratio": 0.0},
                "toc_duplicate_ratio": {"passed": True, "ratio": 0.0},
                "article_ref_coverage": {"passed": True, "coverage": 1.0},
            }
        }
        score: float = compute_quality_score(report)
        assert score == 1.0

    def test_camara_de_diputados_detected(self) -> None:
        """CÁMARA DE DIPUTADOS in heading triggers header/footer bleed."""
        camara_node: StructuralNode = _node(
            node_id="section-camara",
            node_type="section",
            heading="CÁMARA DE DIPUTADOS DEL H. CONGRESO",
            text=None,
            article_ref=None,
            page_start=1,
            page_end=1,
            children=None,
            source_block_ids=["b1"],
        )
        root: StructuralNode = _node(
            node_id="doc", node_type="document", heading=None, text=None,
            article_ref=None, page_start=1, page_end=1,
            children=[camara_node], source_block_ids=[],
        )
        structure: DocumentStructure = _make_structure(root, None)
        report: dict[str, object] = validate_document_structure(structure)
        checks: dict[str, object] = report["checks"]  # type: ignore[assignment]
        bleed_check: dict[str, object] = checks["header_footer_bleed"]  # type: ignore[assignment]

        assert bleed_check["passed"] is False
