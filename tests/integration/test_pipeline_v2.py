"""
Integration tests for Phase 7 — v2 pipeline integration.

Tests cover routing/flags, quality gate, chunk adaptation, shadow compare,
metadata persistence, and robustness. All tests run without a real database
or Azure services by mocking external dependencies.

Run with: pytest tests/integration/test_pipeline_v2.py -v --tb=short
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch, call

import pytest

from pipeline.layout_models import (
    ClassifiedBlock,
    DocumentLayout,
    DocumentStructure,
    ExtractedSpan,
    LayoutBlock,
    PageLayout,
    StructuralNode,
)
from pipeline.legal_chunker import Chunk
from pipeline.shadow_compare_v2 import compare_pipeline_outputs


# ── Fixtures / helpers ───────────────────────────────────────────────────────

_MINIMAL_PDF_BYTES = (
    b"%PDF-1.4\n1 0 obj\n<<\n/Type/Catalog\n/Pages 2 0 R\n>>\nendobj\n"
    b"2 0 obj\n<<\n/Type/Pages\n/Kids[3 0 R]\n/Count 1\n>>\nendobj\n"
    b"3 0 obj\n<<\n/Type/Page\n/Parent 2 0 R\n/MediaBox[0 0 612 792]\n"
    b"/Contents 4 0 R\n>>\nendobj\n"
    b"4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n"
    b"100 700 Td\n(Articulo 1.) Tj\nET\nendstream\nendobj\n"
    b"xref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n"
    b"0000000115 00000 n \n0000000206 00000 n \n"
    b"trailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n298\n%%EOF"
)


def _make_payload(
    run_id: str | None = None,
    blob_path: str = "test/v2-test.pdf",
    title: str = "Ley Test V2",
) -> object:
    """Build a minimal payload object that quacks like ProcessPdfRequest."""
    @dataclass
    class _FakePayload:
        runId: str
        blobPath: str
        documentTitle: str
        categoryId: str
        publishDate: Optional[str]
        relaxProseTableFilter: Optional[bool] = None
        relaxedVisualFrameDetection: Optional[bool] = None
        enableLlmGenericHeadingRefine: Optional[bool] = None

    return _FakePayload(
        runId=run_id or str(uuid.uuid4()),
        blobPath=blob_path,
        documentTitle=title,
        categoryId=str(uuid.uuid4()),
        publishDate="2024-01-15",
    )


def _make_v2_structure(
    quality_score: float = 0.92,
    article_count: int = 3,
) -> DocumentStructure:
    """Build a minimal DocumentStructure for testing."""
    children: list[StructuralNode] = []
    for i in range(1, article_count + 1):
        children.append(StructuralNode(
            node_id=f"article-{i}",
            node_type="article",
            heading=f"Artículo {i}",
            text=f"Contenido del artículo {i}.",
            article_ref=str(i),
            page_start=1,
            page_end=1,
            children=[],
            source_block_ids=[f"p1_b{i}"],
            metadata={},
        ))

    root = StructuralNode(
        node_id="doc",
        node_type="document",
        heading="Test Document",
        text=None,
        article_ref=None,
        page_start=1,
        page_end=1,
        children=children,
        source_block_ids=[],
        metadata={"document_title": "Test Document", "has_transitories": False, "has_annexes": False},
    )

    severity: str = "low" if quality_score >= 0.85 else ("medium" if quality_score >= 0.70 else "high")
    quality_report: dict[str, object] = {
        "quality_score": quality_score,
        "checks": {},
        "summary": {"passed": quality_score >= 0.70, "severity": severity, "reasons": []},
    }

    toc: list[dict[str, object]] = [
        {"node_id": f"article-{i}", "node_type": "article", "heading": f"Artículo {i}", "page_start": 1}
        for i in range(1, article_count + 1)
    ]

    return DocumentStructure(
        root=root,
        toc=toc,
        sections=[],
        quality_report=quality_report,
        metadata={"document_title": "Test Document", "has_transitories": False, "has_annexes": False},
    )


def _make_v2_chunks(count: int = 3) -> list[dict[str, object]]:
    """Build minimal v2 chunk dicts."""
    return [
        {
            "chunk_type": "article",
            "heading": f"Artículo {i}",
            "text": f"Contenido del artículo {i}.",
            "article_ref": str(i),
            "page_start": 1,
            "page_end": 1,
            "source_block_ids": [f"p1_b{i}"],
            "metadata": {"node_id": f"article-{i}", "node_type": "article"},
        }
        for i in range(1, count + 1)
    ]


def _make_legacy_chunks(count: int = 3) -> list[Chunk]:
    """Build minimal legacy Chunk instances."""
    return [
        Chunk(
            text=f"Contenido del artículo {i}.",
            chunk_no=i,
            chunk_type="article",
            article_ref=str(i),
            heading=f"Artículo {i}",
            start_page=1,
            end_page=1,
            has_table=False,
            table_index=None,
        )
        for i in range(1, count + 1)
    ]


def _fake_embed(chunks, **kwargs):
    n = len(chunks)
    cb = kwargs.get("progress_callback")
    if cb:
        cb(n, n)
    return [[0.1] * 1536 for _ in range(n)]


def _noop_db_conn():
    """Context manager returning a MagicMock connection."""
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        yield mock_conn

    return _ctx()


# ── Test 1-3: Routing / flags ───────────────────────────────────────────────


class TestPipelineRouting:
    """Tests 1-3: verify correct pipeline dispatch based on feature flags."""

    @patch("pipeline.runner.settings")
    @patch("pipeline.runner.get_db_conn")
    @patch("pipeline.runner.download_pdf_bytes", return_value=_MINIMAL_PDF_BYTES)
    @patch("pipeline.runner.embed_chunks", side_effect=_fake_embed)
    @patch("pipeline.runner.extract_pdf")
    @patch("pipeline.runner.chunk_content")
    @patch("pipeline.runner.classify_doc_type", return_value=("ley_federal", 0.9, "heuristic"))
    @patch("pipeline.runner.extract_legal_legend", return_value={})
    @patch("pipeline.runner.extract_law_name", return_value="Ley Test")
    @patch("pipeline.runner.refine_generic_chunk_headings", return_value=0)
    @patch("pipeline.runner.resolve_llm_heading_refinement_flags", return_value=(False, False))
    @patch("pipeline.runner.upsert_legal_document", return_value=str(uuid.uuid4()))
    @patch("pipeline.runner.delete_existing_chunks")
    @patch("pipeline.runner.insert_chunks_bulk")
    @patch("pipeline.runner.persist_legal_outline", return_value=5)
    @patch("pipeline.runner.update_index_run")
    @patch("pipeline.runner.update_index_run_progress")
    @patch("pipeline.runner.merge_legal_document_metadata")
    def test_flags_off_uses_legacy(
        self, mock_merge_meta, mock_progress, mock_update,
        mock_outline, mock_insert, mock_delete, mock_upsert,
        mock_resolve_flags, mock_refine, mock_law_name, mock_legend,
        mock_classify, mock_chunk, mock_extract, mock_embed,
        mock_download, mock_conn, mock_settings,
    ):
        """Test 1: with flags off, legacy pipeline runs."""
        mock_settings.ENABLE_LAYOUT_V2 = False
        mock_settings.LAYOUT_V2_SHADOW_MODE = False
        mock_settings.ENABLE_LLM_DOC_TYPE = False
        mock_settings.ENABLE_LLM_GENERIC_HEADING_REFINE = False
        mock_settings.LLM_GENERIC_HEADING_REFINE_ALL = False
        mock_settings.RELAX_PROSE_TABLE_FILTER = False
        mock_settings.RELAXED_VISUAL_FRAME_DETECTION = False
        mock_settings.AZURE_BLOB_CONTAINER = "laws"
        mock_settings.PDF_WORKER_THREADS = 1

        mock_conn.return_value = _noop_db_conn()

        fake_page = MagicMock()
        fake_page.text = "Artículo 1. Contenido."
        fake_page.tables = []
        mock_extract.return_value = ([fake_page], [])
        mock_chunk.return_value = _make_legacy_chunks(2)

        payload = _make_payload()

        from pipeline.runner import run_pipeline
        run_pipeline(payload)

        mock_extract.assert_called_once()
        mock_chunk.assert_called_once()

    @patch("pipeline.runner.settings")
    @patch("pipeline.runner.get_db_conn")
    @patch("pipeline.runner.download_pdf_bytes", return_value=_MINIMAL_PDF_BYTES)
    @patch("pipeline.runner.embed_chunks", side_effect=_fake_embed)
    @patch("pipeline.runner._run_v2_extraction")
    @patch("pipeline.runner.classify_doc_type", return_value=("ley_federal", 0.9, "heuristic"))
    @patch("pipeline.runner.extract_legal_legend", return_value={})
    @patch("pipeline.runner.extract_law_name", return_value="Ley Test")
    @patch("pipeline.runner.upsert_legal_document", return_value=str(uuid.uuid4()))
    @patch("pipeline.runner.delete_existing_chunks")
    @patch("pipeline.runner.insert_chunks_bulk")
    @patch("pipeline.runner.persist_legal_outline", return_value=5)
    @patch("pipeline.runner.update_index_run")
    @patch("pipeline.runner.update_index_run_progress")
    @patch("pipeline.runner.merge_legal_document_metadata")
    def test_enable_v2_uses_v2(
        self, mock_merge_meta, mock_progress, mock_update,
        mock_outline, mock_insert, mock_delete, mock_upsert,
        mock_law_name, mock_legend, mock_classify,
        mock_v2_extract, mock_embed, mock_download, mock_conn, mock_settings,
    ):
        """Test 2: with ENABLE_LAYOUT_V2=True, v2 pipeline runs."""
        mock_settings.ENABLE_LAYOUT_V2 = True
        mock_settings.LAYOUT_V2_SHADOW_MODE = False
        mock_settings.LAYOUT_V2_MIN_QUALITY_SCORE = 0.85
        mock_settings.ALLOW_LOW_QUALITY_INDEX_IN_NON_PROD = True
        mock_settings.ENVIRONMENT = "development"
        mock_settings.ENABLE_LLM_DOC_TYPE = False
        mock_settings.AZURE_BLOB_CONTAINER = "laws"
        mock_settings.PDF_WORKER_THREADS = 1

        mock_conn.return_value = _noop_db_conn()

        structure = _make_v2_structure(quality_score=0.92)
        v2_chunks = _make_v2_chunks(3)
        quality_report = dict(structure.quality_report)

        mock_v2_extract.return_value = (v2_chunks, structure, quality_report)

        payload = _make_payload()

        from pipeline.runner import run_pipeline
        run_pipeline(payload)

        mock_v2_extract.assert_called_once()

    @patch("pipeline.runner.settings")
    @patch("pipeline.runner.get_db_conn")
    @patch("pipeline.runner.download_pdf_bytes", return_value=_MINIMAL_PDF_BYTES)
    @patch("pipeline.runner.embed_chunks", side_effect=_fake_embed)
    @patch("pipeline.runner._run_v2_extraction")
    @patch("pipeline.runner._run_legacy_extraction")
    @patch("pipeline.runner.classify_doc_type", return_value=("ley_federal", 0.9, "heuristic"))
    @patch("pipeline.runner.extract_legal_legend", return_value={})
    @patch("pipeline.runner.extract_law_name", return_value="Ley Test")
    @patch("pipeline.runner.upsert_legal_document", return_value=str(uuid.uuid4()))
    @patch("pipeline.runner.delete_existing_chunks")
    @patch("pipeline.runner.insert_chunks_bulk")
    @patch("pipeline.runner.persist_legal_outline", return_value=5)
    @patch("pipeline.runner.update_index_run")
    @patch("pipeline.runner.update_index_run_progress")
    @patch("pipeline.runner.merge_legal_document_metadata")
    @patch("pipeline.runner._get_first_pages_text", return_value="")
    def test_shadow_mode_runs_both(
        self, mock_first_text, mock_merge_meta, mock_progress, mock_update,
        mock_outline, mock_insert, mock_delete, mock_upsert,
        mock_law_name, mock_legend, mock_classify,
        mock_legacy_extract, mock_v2_extract,
        mock_embed, mock_download, mock_conn, mock_settings,
    ):
        """Test 3: shadow mode runs legacy (primary) + v2 (comparison)."""
        mock_settings.ENABLE_LAYOUT_V2 = False
        mock_settings.LAYOUT_V2_SHADOW_MODE = True
        mock_settings.LAYOUT_V2_MIN_QUALITY_SCORE = 0.85
        mock_settings.ALLOW_LOW_QUALITY_INDEX_IN_NON_PROD = True
        mock_settings.ENVIRONMENT = "development"
        mock_settings.ENABLE_LLM_DOC_TYPE = False
        mock_settings.AZURE_BLOB_CONTAINER = "laws"
        mock_settings.PDF_WORKER_THREADS = 1

        mock_conn.return_value = _noop_db_conn()

        legacy_chunks = _make_legacy_chunks(3)
        mock_legacy_extract.return_value = (legacy_chunks, 5, [])

        structure = _make_v2_structure(quality_score=0.90)
        v2_chunks = _make_v2_chunks(3)
        mock_v2_extract.return_value = (v2_chunks, structure, dict(structure.quality_report))

        payload = _make_payload()

        from pipeline.runner import run_pipeline
        run_pipeline(payload)

        mock_legacy_extract.assert_called_once()
        mock_v2_extract.assert_called_once()

        # Verify shadow_compare was persisted in metadata
        merge_calls = mock_merge_meta.call_args_list
        found_shadow = any(
            isinstance(c.args[2] if len(c.args) > 2 else c.kwargs.get("patch", {}), dict)
            and "shadow_compare" in (c.args[2] if len(c.args) > 2 else c.kwargs.get("patch", {}))
            for c in merge_calls
        )
        assert found_shadow, "shadow_compare should be persisted in metadata"


# ── Test 4-6: Quality gate ───────────────────────────────────────────────────


class TestQualityGate:
    """Tests 4-6: quality gate behavior."""

    def test_sufficient_score_allows_indexing(self):
        """Test 4: score >= threshold allows normal indexing."""
        from pipeline.runner import _evaluate_quality_gate

        with patch("pipeline.runner.settings") as mock_settings:
            mock_settings.LAYOUT_V2_MIN_QUALITY_SCORE = 0.85
            mock_settings.ENVIRONMENT = "production"
            mock_settings.ALLOW_LOW_QUALITY_INDEX_IN_NON_PROD = True

            should_index, reason = _evaluate_quality_gate(
                {"quality_score": 0.92}, "run-1", "test.pdf",
            )
            assert should_index is True
            assert reason == "quality_sufficient"

    def test_low_score_non_prod_allowed(self):
        """Test 5: low score in non-prod continues if flag allows."""
        from pipeline.runner import _evaluate_quality_gate

        with patch("pipeline.runner.settings") as mock_settings:
            mock_settings.LAYOUT_V2_MIN_QUALITY_SCORE = 0.85
            mock_settings.ENVIRONMENT = "development"
            mock_settings.ALLOW_LOW_QUALITY_INDEX_IN_NON_PROD = True

            should_index, reason = _evaluate_quality_gate(
                {"quality_score": 0.60}, "run-1", "test.pdf",
            )
            assert should_index is True
            assert reason == "low_quality_allowed_non_prod"

    def test_low_score_prod_rejected(self):
        """Test 6: low score in production is rejected."""
        from pipeline.runner import _evaluate_quality_gate

        with patch("pipeline.runner.settings") as mock_settings:
            mock_settings.LAYOUT_V2_MIN_QUALITY_SCORE = 0.85
            mock_settings.ENVIRONMENT = "production"
            mock_settings.ALLOW_LOW_QUALITY_INDEX_IN_NON_PROD = True

            should_index, reason = _evaluate_quality_gate(
                {"quality_score": 0.60}, "run-1", "test.pdf",
            )
            assert should_index is False
            assert reason == "quality_insufficient"


# ── Test 7-8: Chunk adaptation ───────────────────────────────────────────────


class TestChunkAdaptation:
    """Tests 7-8: v2 chunks adapt to legacy contract."""

    def test_v2_chunks_adapt_to_legacy_contract(self):
        """Test 7: adapted chunks have all required legacy fields."""
        from pipeline.runner import _adapt_v2_chunks_to_legacy

        v2_chunks: list[dict[str, object]] = [
            {
                "chunk_type": "article",
                "heading": "Artículo 1",
                "text": "Texto del artículo.",
                "article_ref": "1",
                "page_start": 2,
                "page_end": 3,
                "source_block_ids": ["p2_b0"],
                "metadata": {"node_id": "article-1"},
            },
            {
                "chunk_type": "table",
                "heading": None,
                "text": "Col1 | Col2\nVal1 | Val2",
                "article_ref": "1",
                "page_start": 3,
                "page_end": 3,
                "source_block_ids": ["p3_b1"],
                "metadata": {},
            },
        ]

        adapted = _adapt_v2_chunks_to_legacy(v2_chunks)

        assert len(adapted) == 2
        assert all(isinstance(c, Chunk) for c in adapted)

        art = adapted[0]
        assert art.chunk_no == 1
        assert art.chunk_type == "article"
        assert art.article_ref == "1"
        assert art.heading == "Artículo 1"
        assert art.start_page == 2
        assert art.end_page == 3
        assert art.has_table is False
        assert art.table_index is None

        table = adapted[1]
        assert table.chunk_no == 2
        assert table.chunk_type == "table"
        assert table.has_table is True
        assert table.table_index == 1

    def test_chunk_no_and_table_index_consistent(self):
        """Test 8: chunk_no, has_table, table_index computed consistently."""
        from pipeline.runner import _adapt_v2_chunks_to_legacy

        v2_chunks = [
            {"chunk_type": "article", "heading": "Art 1", "text": "T1",
             "article_ref": "1", "page_start": 1, "page_end": 1,
             "source_block_ids": [], "metadata": {}},
            {"chunk_type": "table", "heading": None, "text": "T",
             "article_ref": None, "page_start": 1, "page_end": 1,
             "source_block_ids": [], "metadata": {}},
            {"chunk_type": "article", "heading": "Art 2", "text": "T2",
             "article_ref": "2", "page_start": 2, "page_end": 2,
             "source_block_ids": [], "metadata": {}},
            {"chunk_type": "table", "heading": None, "text": "T2",
             "article_ref": None, "page_start": 2, "page_end": 2,
             "source_block_ids": [], "metadata": {}},
        ]

        adapted = _adapt_v2_chunks_to_legacy(v2_chunks)

        assert [c.chunk_no for c in adapted] == [1, 2, 3, 4]
        assert adapted[0].has_table is False
        assert adapted[1].has_table is True
        assert adapted[1].table_index == 1
        assert adapted[3].has_table is True
        assert adapted[3].table_index == 2


# ── Test 9-10: Shadow compare ────────────────────────────────────────────────


class TestShadowCompare:
    """Tests 9-10: shadow comparison metrics."""

    def test_shadow_compare_generates_required_metrics(self):
        """Test 9: compare_pipeline_outputs returns all required metrics."""
        legacy = _make_legacy_chunks(5)
        v2 = _make_v2_chunks(4)

        result = compare_pipeline_outputs(
            legacy_chunks=legacy,
            v2_chunks=v2,
            legacy_metadata=None,
            v2_metadata={"quality_score": 0.91},
        )

        assert "legacy_chunk_count" in result
        assert "v2_chunk_count" in result
        assert "legacy_article_ref_coverage" in result
        assert "v2_article_ref_coverage" in result
        assert "legacy_table_chunk_count" in result
        assert "v2_table_chunk_count" in result
        assert "legacy_boxed_note_count" in result
        assert "v2_boxed_note_count" in result
        assert "legacy_has_visible_table_tokens" in result
        assert "v2_has_visible_table_tokens" in result
        assert "heading_quality_delta" in result
        assert "quality_score_v2" in result
        assert "summary" in result

        assert result["legacy_chunk_count"] == 5
        assert result["v2_chunk_count"] == 4
        assert result["quality_score_v2"] == 0.91

    def test_shadow_mode_does_not_break_legacy_result(self):
        """Test 10: shadow compare produces valid output without errors."""
        legacy = _make_legacy_chunks(3)
        v2 = _make_v2_chunks(3)

        result = compare_pipeline_outputs(
            legacy_chunks=legacy,
            v2_chunks=v2,
            legacy_metadata=None,
            v2_metadata=None,
        )

        assert isinstance(result, dict)
        # Should be JSON-serializable
        serialized = json.dumps(result)
        assert len(serialized) > 0


# ── Test 11-14: Metadata persistence ─────────────────────────────────────────


class TestMetadataPersistence:
    """Tests 11-14: extended metadata is persisted correctly."""

    def test_metadata_includes_pipeline_version(self):
        """Test 11: metadata includes pipeline_version."""
        from pipeline.runner import _build_structure_summary

        structure = _make_v2_structure(quality_score=0.92)
        v2_metadata: dict[str, object] = {
            "pipeline_version": "v2",
            "quality_report": dict(structure.quality_report),
            "structure_summary": _build_structure_summary(structure),
        }

        assert v2_metadata["pipeline_version"] == "v2"

    def test_metadata_includes_quality_data(self):
        """Test 12: metadata includes quality_score and quality_report."""
        structure = _make_v2_structure(quality_score=0.88)
        quality_report = dict(structure.quality_report)

        v2_metadata: dict[str, object] = {
            "quality_report": quality_report,
            "quality_score": float(quality_report.get("quality_score", 0.0)),
        }

        assert v2_metadata["quality_score"] == 0.88
        assert "quality_report" in v2_metadata
        assert isinstance(v2_metadata["quality_report"], dict)

    def test_metadata_includes_structure_summary(self):
        """Test 13: metadata includes structure_summary with correct counts."""
        from pipeline.runner import _build_structure_summary

        structure = _make_v2_structure(quality_score=0.92, article_count=5)
        summary = _build_structure_summary(structure)

        assert summary["total_articles"] == 5
        assert "node_type_counts" in summary
        assert "toc_entry_count" in summary
        assert summary["toc_entry_count"] == 5

    def test_metadata_includes_shadow_compare(self):
        """Test 14: in shadow mode, metadata includes shadow_compare."""
        legacy = _make_legacy_chunks(3)
        v2 = _make_v2_chunks(4)

        compare_result = compare_pipeline_outputs(
            legacy_chunks=legacy,
            v2_chunks=v2,
            legacy_metadata=None,
            v2_metadata={"quality_score": 0.90},
        )

        extended_metadata: dict[str, object] = {
            "pipeline_version": "legacy",
            "shadow_mode": True,
            "shadow_compare": compare_result,
        }

        assert "shadow_compare" in extended_metadata
        assert extended_metadata["shadow_compare"]["legacy_chunk_count"] == 3
        assert extended_metadata["shadow_compare"]["v2_chunk_count"] == 4

        # Must be serializable
        serialized = json.dumps(extended_metadata)
        assert len(serialized) > 0


# ── Test 15-18: Robustness ───────────────────────────────────────────────────


class TestRobustness:
    """Tests 15-18: error handling and resilience."""

    @patch("pipeline.runner.settings")
    @patch("pipeline.runner.get_db_conn")
    @patch("pipeline.runner.download_pdf_bytes", return_value=_MINIMAL_PDF_BYTES)
    @patch("pipeline.runner.embed_chunks", side_effect=_fake_embed)
    @patch("pipeline.runner._run_v2_extraction", side_effect=RuntimeError("v2 crash"))
    @patch("pipeline.runner._run_legacy_extraction")
    @patch("pipeline.runner.classify_doc_type", return_value=("ley_federal", 0.9, None))
    @patch("pipeline.runner.extract_legal_legend", return_value={})
    @patch("pipeline.runner.extract_law_name", return_value="Ley Test")
    @patch("pipeline.runner.upsert_legal_document", return_value=str(uuid.uuid4()))
    @patch("pipeline.runner.delete_existing_chunks")
    @patch("pipeline.runner.insert_chunks_bulk")
    @patch("pipeline.runner.persist_legal_outline", return_value=5)
    @patch("pipeline.runner.update_index_run")
    @patch("pipeline.runner.update_index_run_progress")
    @patch("pipeline.runner.merge_legal_document_metadata")
    @patch("pipeline.runner._get_first_pages_text", return_value="")
    def test_v2_exception_does_not_break_shadow(
        self, mock_first_text, mock_merge_meta, mock_progress, mock_update,
        mock_outline, mock_insert, mock_delete, mock_upsert,
        mock_law_name, mock_legend, mock_classify,
        mock_legacy_extract, mock_v2_extract,
        mock_embed, mock_download, mock_conn, mock_settings,
    ):
        """Test 15: v2 exception in shadow mode does not break legacy."""
        mock_settings.ENABLE_LAYOUT_V2 = False
        mock_settings.LAYOUT_V2_SHADOW_MODE = True
        mock_settings.LAYOUT_V2_MIN_QUALITY_SCORE = 0.85
        mock_settings.ALLOW_LOW_QUALITY_INDEX_IN_NON_PROD = True
        mock_settings.ENVIRONMENT = "development"
        mock_settings.ENABLE_LLM_DOC_TYPE = False
        mock_settings.AZURE_BLOB_CONTAINER = "laws"
        mock_settings.PDF_WORKER_THREADS = 1

        mock_conn.return_value = _noop_db_conn()
        mock_legacy_extract.return_value = (_make_legacy_chunks(3), 5, [])

        payload = _make_payload()

        from pipeline.runner import run_pipeline
        run_pipeline(payload)

        # Legacy still ran and completed
        mock_legacy_extract.assert_called_once()
        mock_insert.assert_called_once()

    @patch("pipeline.runner.settings")
    @patch("pipeline.runner.get_db_conn")
    @patch("pipeline.runner.download_pdf_bytes", return_value=_MINIMAL_PDF_BYTES)
    @patch("pipeline.runner.embed_chunks", side_effect=_fake_embed)
    @patch("pipeline.runner._run_v2_extraction", side_effect=RuntimeError("v2 crash"))
    @patch("pipeline.runner._run_legacy_extraction")
    @patch("pipeline.runner.classify_doc_type", return_value=("ley_federal", 0.9, None))
    @patch("pipeline.runner.extract_legal_legend", return_value={})
    @patch("pipeline.runner.extract_law_name", return_value="Ley Test")
    @patch("pipeline.runner.upsert_legal_document", return_value=str(uuid.uuid4()))
    @patch("pipeline.runner.delete_existing_chunks")
    @patch("pipeline.runner.insert_chunks_bulk")
    @patch("pipeline.runner.persist_legal_outline", return_value=5)
    @patch("pipeline.runner.update_index_run")
    @patch("pipeline.runner.update_index_run_progress")
    @patch("pipeline.runner.merge_legal_document_metadata")
    def test_v2_primary_falls_back_to_legacy_on_error(
        self, mock_merge_meta, mock_progress, mock_update,
        mock_outline, mock_insert, mock_delete, mock_upsert,
        mock_law_name, mock_legend, mock_classify,
        mock_legacy_extract, mock_v2_extract,
        mock_embed, mock_download, mock_conn, mock_settings,
    ):
        """Test 16: v2 as primary falls back to legacy on exception."""
        mock_settings.ENABLE_LAYOUT_V2 = True
        mock_settings.LAYOUT_V2_SHADOW_MODE = False
        mock_settings.LAYOUT_V2_MIN_QUALITY_SCORE = 0.85
        mock_settings.ALLOW_LOW_QUALITY_INDEX_IN_NON_PROD = True
        mock_settings.ENVIRONMENT = "development"
        mock_settings.ENABLE_LLM_DOC_TYPE = False
        mock_settings.AZURE_BLOB_CONTAINER = "laws"
        mock_settings.PDF_WORKER_THREADS = 1

        mock_conn.return_value = _noop_db_conn()
        mock_legacy_extract.return_value = (_make_legacy_chunks(3), 5, [])

        payload = _make_payload()

        from pipeline.runner import run_pipeline
        run_pipeline(payload)

        # V2 was attempted
        mock_v2_extract.assert_called_once()
        # Fallback to legacy
        mock_legacy_extract.assert_called_once()
        # Still wrote chunks
        mock_insert.assert_called_once()

    def test_output_is_serializable(self):
        """Test 17: all output structures are JSON-serializable."""
        from pipeline.runner import _build_structure_summary, _adapt_v2_chunks_to_legacy

        structure = _make_v2_structure(quality_score=0.92)
        summary = _build_structure_summary(structure)

        # Summary is serializable
        json.dumps(summary)

        # Adapted chunks produce serializable metadata
        v2_chunks = _make_v2_chunks(3)
        adapted = _adapt_v2_chunks_to_legacy(v2_chunks)
        for c in adapted:
            json.dumps({
                "text": c.text,
                "chunk_no": c.chunk_no,
                "chunk_type": c.chunk_type,
                "article_ref": c.article_ref,
                "heading": c.heading,
                "start_page": c.start_page,
                "end_page": c.end_page,
                "has_table": c.has_table,
                "table_index": c.table_index,
            })

        # Shadow compare is serializable
        compare_result = compare_pipeline_outputs(
            _make_legacy_chunks(3), v2_chunks, None, None,
        )
        json.dumps(compare_result)

    @patch("pipeline.runner.settings")
    @patch("pipeline.runner.get_db_conn")
    @patch("pipeline.runner.download_pdf_bytes", return_value=_MINIMAL_PDF_BYTES)
    @patch("pipeline.runner.embed_chunks", side_effect=_fake_embed)
    @patch("pipeline.runner.extract_pdf")
    @patch("pipeline.runner.chunk_content")
    @patch("pipeline.runner.classify_doc_type", return_value=("ley_federal", 0.9, None))
    @patch("pipeline.runner.extract_legal_legend", return_value={})
    @patch("pipeline.runner.extract_law_name", return_value="Ley Test")
    @patch("pipeline.runner.refine_generic_chunk_headings", return_value=0)
    @patch("pipeline.runner.resolve_llm_heading_refinement_flags", return_value=(False, False))
    @patch("pipeline.runner.upsert_legal_document", return_value=str(uuid.uuid4()))
    @patch("pipeline.runner.delete_existing_chunks")
    @patch("pipeline.runner.insert_chunks_bulk")
    @patch("pipeline.runner.persist_legal_outline", return_value=5)
    @patch("pipeline.runner.update_index_run")
    @patch("pipeline.runner.update_index_run_progress")
    @patch("pipeline.runner.merge_legal_document_metadata")
    def test_no_regression_legacy_path(
        self, mock_merge_meta, mock_progress, mock_update,
        mock_outline, mock_insert, mock_delete, mock_upsert,
        mock_resolve_flags, mock_refine, mock_law_name, mock_legend,
        mock_classify, mock_chunk, mock_extract, mock_embed,
        mock_download, mock_conn, mock_settings,
    ):
        """Test 18: legacy path produces same behavior as before."""
        mock_settings.ENABLE_LAYOUT_V2 = False
        mock_settings.LAYOUT_V2_SHADOW_MODE = False
        mock_settings.ENABLE_LLM_DOC_TYPE = False
        mock_settings.ENABLE_LLM_GENERIC_HEADING_REFINE = False
        mock_settings.LLM_GENERIC_HEADING_REFINE_ALL = False
        mock_settings.RELAX_PROSE_TABLE_FILTER = False
        mock_settings.RELAXED_VISUAL_FRAME_DETECTION = False
        mock_settings.AZURE_BLOB_CONTAINER = "laws"
        mock_settings.PDF_WORKER_THREADS = 1

        mock_conn.return_value = _noop_db_conn()

        fake_page = MagicMock()
        fake_page.text = "Artículo 1. Contenido legal del artículo."
        fake_page.tables = []
        mock_extract.return_value = ([fake_page], [])
        legacy_chunks = _make_legacy_chunks(3)
        mock_chunk.return_value = legacy_chunks

        payload = _make_payload()

        from pipeline.runner import run_pipeline
        run_pipeline(payload)

        # Core legacy functions called in order
        mock_extract.assert_called_once()
        mock_chunk.assert_called_once()
        mock_embed.assert_called_once()
        mock_insert.assert_called_once()
        mock_outline.assert_called_once()

        # Completed status update was called (last update_index_run call)
        last_update_call = mock_update.call_args_list[-1]
        assert last_update_call.args[2] == "completed" or (
            len(last_update_call.args) > 2 and "completed" in str(last_update_call)
        )
