"""
Tests for scripts/_v2_eval_helpers.py — shared utilities for Phase 8 operational scripts.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from scripts._v2_eval_helpers import (
    _check_passed,
    _median,
    _score_buckets,
    build_result_record,
    build_structure_summary,
    compute_aggregate_metrics,
    discover_local_pdfs,
    quality_report_to_flat_row,
    read_local_pdf,
    run_v2_pipeline_eval,
    title_from_path,
)


# ── discover_local_pdfs ──────────────────────────────────────────────────────


class TestDiscoverLocalPdfs:
    def test_finds_pdfs_in_flat_dir(self, tmp_path: Path) -> None:
        (tmp_path / "a.pdf").write_bytes(b"fake")
        (tmp_path / "b.pdf").write_bytes(b"fake")
        (tmp_path / "c.txt").write_bytes(b"not a pdf")

        result: list[str] = discover_local_pdfs(str(tmp_path), "*.pdf")
        assert len(result) == 2
        assert all(p.endswith(".pdf") for p in result)

    def test_finds_pdfs_recursively(self, tmp_path: Path) -> None:
        sub: Path = tmp_path / "sub" / "deep"
        sub.mkdir(parents=True)
        (sub / "nested.pdf").write_bytes(b"fake")
        (tmp_path / "top.pdf").write_bytes(b"fake")

        result: list[str] = discover_local_pdfs(str(tmp_path), "*.pdf")
        assert len(result) == 2

    def test_respects_glob_pattern(self, tmp_path: Path) -> None:
        (tmp_path / "a.pdf").write_bytes(b"fake")
        (tmp_path / "b.xlsx").write_bytes(b"fake")

        result: list[str] = discover_local_pdfs(str(tmp_path), "*.xlsx")
        assert len(result) == 1
        assert result[0].endswith(".xlsx")

    def test_raises_on_missing_dir(self) -> None:
        with pytest.raises(FileNotFoundError):
            discover_local_pdfs("/nonexistent/path/xyz", "*.pdf")

    def test_returns_empty_for_no_matches(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_bytes(b"text")
        result: list[str] = discover_local_pdfs(str(tmp_path), "*.pdf")
        assert result == []


# ── read_local_pdf ───────────────────────────────────────────────────────────


class TestReadLocalPdf:
    def test_reads_file(self, tmp_path: Path) -> None:
        pdf_file: Path = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-fake-content")
        result: bytes = read_local_pdf(str(pdf_file))
        assert result == b"%PDF-fake-content"

    def test_raises_on_missing(self) -> None:
        with pytest.raises(FileNotFoundError):
            read_local_pdf("/nonexistent/file.pdf")


# ── title_from_path ──────────────────────────────────────────────────────────


class TestTitleFromPath:
    def test_extracts_stem(self) -> None:
        assert title_from_path("/path/to/ley_federal.pdf") == "ley_federal"

    def test_handles_nested_path(self) -> None:
        assert title_from_path("tmp/pdf_legales/2025/LEYES/doc.pdf") == "doc"


# ── _median ──────────────────────────────────────────────────────────────────


class TestMedian:
    def test_empty(self) -> None:
        assert _median([]) == 0.0

    def test_single(self) -> None:
        assert _median([0.5]) == 0.5

    def test_odd_count(self) -> None:
        assert _median([0.1, 0.5, 0.9]) == 0.5

    def test_even_count(self) -> None:
        assert _median([0.2, 0.4, 0.6, 0.8]) == 0.5


# ── _score_buckets ───────────────────────────────────────────────────────────


class TestScoreBuckets:
    def test_empty(self) -> None:
        result: dict[str, int] = _score_buckets([])
        assert all(v == 0 for v in result.values())

    def test_distribution(self) -> None:
        scores: list[float] = [0.3, 0.6, 0.75, 0.9, 0.95]
        result: dict[str, int] = _score_buckets(scores)
        assert result["0.0-0.5"] == 1
        assert result["0.5-0.7"] == 1
        assert result["0.7-0.85"] == 1
        assert result["0.85-1.0"] == 2


# ── _check_passed ────────────────────────────────────────────────────────────


class TestCheckPassed:
    def test_passed_true(self) -> None:
        assert _check_passed({"passed": True}) is True

    def test_passed_false(self) -> None:
        assert _check_passed({"passed": False}) is False

    def test_none(self) -> None:
        assert _check_passed(None) is True

    def test_not_dict(self) -> None:
        assert _check_passed("something") is True


# ── build_result_record ──────────────────────────────────────────────────────


class TestBuildResultRecord:
    def test_builds_failed_record(self) -> None:
        record: dict[str, object] = build_result_record(
            document_id="abc",
            blob_path=None,
            file_path="/path/to/file.pdf",
            input_source="file",
            pipeline_result=None,
            shadow_compare=None,
            status="failed",
            error="test error",
            mode="v2+dry_run",
        )
        assert record["status"] == "failed"
        assert record["error"] == "test error"
        assert record["quality_score"] == 0.0
        assert record["chunk_count"] == 0

    def test_builds_ok_record_with_pipeline_result(self) -> None:
        from pipeline.layout_models import DocumentStructure, StructuralNode

        root: StructuralNode = StructuralNode(
            node_id="root",
            node_type="document",
            heading=None,
            text=None,
            article_ref=None,
            page_start=1,
            page_end=1,
            children=[],
            source_block_ids=[],
            metadata={},
        )
        structure: DocumentStructure = DocumentStructure(
            root=root,
            toc=[],
            sections=[],
            quality_report={},
            metadata={},
        )

        pipeline_result: dict[str, object] = {
            "quality_report": {
                "quality_score": 0.92,
                "summary": {"severity": "low", "reasons": []},
            },
            "structure": structure,
            "v2_chunks": [{"text": "chunk1"}, {"text": "chunk2"}],
            "duration_ms": 500,
        }

        record: dict[str, object] = build_result_record(
            document_id=None,
            blob_path=None,
            file_path="/file.pdf",
            input_source="file",
            pipeline_result=pipeline_result,
            shadow_compare=None,
            status="ok",
            error=None,
            mode="v2",
        )
        assert record["status"] == "ok"
        assert record["quality_score"] == 0.92
        assert record["quality_severity"] == "low"
        assert record["chunk_count"] == 2


# ── quality_report_to_flat_row ───────────────────────────────────────────────


class TestQualityReportToFlatRow:
    def test_produces_all_expected_keys(self) -> None:
        record: dict[str, object] = {
            "document_id": "abc",
            "blob_path": "laws/doc.pdf",
            "file_path": "/local/doc.pdf",
            "input_source": "file",
            "quality_score": 0.85,
            "quality_severity": "low",
            "chunk_count": 10,
            "structure_summary": {
                "total_articles": 5,
                "total_tables": 2,
            },
            "status": "ok",
        }
        qr: dict[str, object] = {
            "quality_score": 0.85,
            "checks": {
                "has_visible_table_tokens": {"passed": True},
                "header_footer_bleed": {"passed": False, "count": 2},
                "article_ref_coverage": {"passed": True, "coverage": 0.9},
                "orphan_tables_count": {"passed": True},
                "toc_duplicate_ratio": {"passed": True},
            },
            "summary": {"severity": "low", "reasons": ["header_footer_bleed"]},
        }

        row: dict[str, object] = quality_report_to_flat_row(record, qr)

        assert row["document_id"] == "abc"
        assert row["has_header_footer_bleed"] is True
        assert row["has_visible_table_tokens"] is False
        assert row["top_reason"] == "header_footer_bleed"
        assert "article_count" in row
        assert "table_count" in row


# ── compute_aggregate_metrics ────────────────────────────────────────────────


class TestComputeAggregateMetrics:
    def test_basic_aggregation(self) -> None:
        records: list[dict[str, object]] = [
            {"status": "ok", "quality_score": 0.9, "quality_severity": "low"},
            {"status": "ok", "quality_score": 0.75, "quality_severity": "medium"},
            {"status": "failed", "quality_score": 0.0, "quality_severity": "unknown"},
        ]
        qrs: list[dict[str, object] | None] = [
            {
                "quality_score": 0.9,
                "checks": {
                    "has_visible_table_tokens": {"passed": True},
                    "header_footer_bleed": {"passed": True},
                },
                "summary": {"severity": "low", "reasons": []},
            },
            {
                "quality_score": 0.75,
                "checks": {
                    "has_visible_table_tokens": {"passed": False, "count": 1},
                    "header_footer_bleed": {"passed": True},
                },
                "summary": {"severity": "medium", "reasons": ["has_visible_table_tokens"]},
            },
            None,
        ]

        agg: dict[str, object] = compute_aggregate_metrics(records, qrs)

        assert agg["total_documents"] == 3
        assert agg["processed_documents"] == 2
        assert agg["failed_documents"] == 1
        assert agg["passed_documents"] == 1
        assert agg["warning_documents"] == 1
        assert agg["documents_with_visible_table_tokens"] == 1
        assert float(agg["avg_quality_score"]) == pytest.approx(0.825, abs=0.001)

    def test_empty_records(self) -> None:
        agg: dict[str, object] = compute_aggregate_metrics([], [])
        assert agg["total_documents"] == 0
        assert agg["avg_quality_score"] == 0.0


# ── run_v2_pipeline_eval (integration-like, uses real v2 modules) ────────────


class TestRunV2PipelineEval:
    def test_runs_on_simple_pdf(self, pdf_bytes_ley_corta: bytes) -> None:
        result: dict[str, object] = run_v2_pipeline_eval(
            pdf_bytes_ley_corta, "LEY FEDERAL DEL TRABAJO",
        )
        assert "layout" in result
        assert "classified_blocks" in result
        assert "structure" in result
        assert "quality_report" in result
        assert "v2_chunks" in result
        assert "duration_ms" in result
        assert isinstance(result["duration_ms"], int)
        assert result["duration_ms"] >= 0
