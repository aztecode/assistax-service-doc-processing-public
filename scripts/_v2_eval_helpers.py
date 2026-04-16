"""
Shared helpers for Phase 8 operational scripts (backfill, evaluation, debug export).

Encapsulates v2 pipeline execution in eval mode, local PDF discovery, and
serialization utilities. All functions are pure or side-effect-free unless
explicitly noted.
"""
from __future__ import annotations

import glob as glob_mod
import os
import time
from pathlib import Path
from typing import Literal

import structlog

_logger = structlog.get_logger()

# Re-export types from layout_models for convenience
from pipeline.layout_models import (  # noqa: E402
    ClassifiedBlock,
    DocumentLayout,
    DocumentStructure,
    LayoutBlock,
    PageLayout,
    StructuralNode,
)

InputSource = Literal["doc_id", "blob_path", "file", "input_dir"]


# ── Local PDF discovery ──────────────────────────────────────────────────────


def discover_local_pdfs(
    input_dir: str,
    glob_pattern: str,
) -> list[str]:
    """Return sorted list of absolute paths matching *glob_pattern* inside *input_dir*.

    Walks subdirectories recursively when the pattern contains '**'.
    Falls back to recursive walk if glob_pattern is a simple extension filter.
    """
    base: Path = Path(input_dir).resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {base}")

    if "**" in glob_pattern or os.sep in glob_pattern:
        full_pattern: str = str(base / glob_pattern)
    else:
        full_pattern = str(base / "**" / glob_pattern)

    matches: list[str] = sorted(glob_mod.glob(full_pattern, recursive=True))
    return [m for m in matches if os.path.isfile(m)]


def read_local_pdf(file_path: str) -> bytes:
    """Read PDF bytes from a local file path."""
    resolved: Path = Path(file_path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"PDF file not found: {resolved}")
    return resolved.read_bytes()


# ── V2 pipeline evaluation runner ────────────────────────────────────────────


def run_v2_pipeline_eval(
    pdf_bytes: bytes,
    document_title: str,
) -> dict[str, object]:
    """Execute the full v2 pipeline in evaluation mode (no DB writes).

    Returns a dict with keys: layout, classified_blocks, structure,
    quality_report, v2_chunks, duration_ms.
    Raises on pipeline failure — callers should catch and record the error.
    """
    from pipeline.layout_extractor_v2 import extract_document_layout
    from pipeline.layout_normalizer_v2 import normalize_document_layout
    from pipeline.block_classifier_v2 import classify_document_layout
    from pipeline.structure_builder_v2 import build_document_structure
    from pipeline.quality_validator_v2 import validate_document_structure
    from pipeline.chunk_projector_v2 import project_structure_to_chunks

    t0: float = time.perf_counter()

    layout: DocumentLayout = extract_document_layout(pdf_bytes)
    normalized: DocumentLayout = normalize_document_layout(layout)
    classified: list[ClassifiedBlock] = classify_document_layout(normalized)
    structure: DocumentStructure = build_document_structure(
        classified,
        {"document_title": document_title},
    )
    quality_report: dict[str, object] = validate_document_structure(structure)
    structure = structure.model_copy(update={"quality_report": quality_report})
    v2_chunks: list[dict[str, object]] = project_structure_to_chunks(structure)

    duration_ms: int = round((time.perf_counter() - t0) * 1000)

    return {
        "layout": layout,
        "normalized_layout": normalized,
        "classified_blocks": classified,
        "structure": structure,
        "quality_report": quality_report,
        "v2_chunks": v2_chunks,
        "duration_ms": duration_ms,
    }


# ── Structure summary builder ────────────────────────────────────────────────


def build_structure_summary(structure: DocumentStructure) -> dict[str, object]:
    """Build a serializable summary of the v2 document structure tree."""
    counts: dict[str, int] = {}

    def _count(node: StructuralNode) -> None:
        counts[node.node_type] = counts.get(node.node_type, 0) + 1
        for child in node.children:
            _count(child)

    _count(structure.root)

    meta: dict[str, object] = structure.metadata or {}
    return {
        "total_articles": counts.get("article", 0),
        "total_transitories": counts.get("transitory", 0),
        "total_tables": counts.get("table", 0),
        "total_notes": counts.get("note", 0),
        "total_paragraphs": counts.get("paragraph", 0),
        "total_fractions": counts.get("fraction", 0),
        "total_incisos": counts.get("inciso", 0),
        "has_transitories": bool(meta.get("has_transitories", False)),
        "has_annexes": bool(meta.get("has_annexes", False)),
        "node_type_counts": counts,
        "toc_entry_count": len(structure.toc),
        "sections_count": len(structure.sections),
    }


# ── Result record builder ────────────────────────────────────────────────────


def build_result_record(
    document_id: str | None,
    blob_path: str | None,
    file_path: str | None,
    input_source: InputSource,
    pipeline_result: dict[str, object] | None,
    shadow_compare: dict[str, object] | None,
    status: str,
    error: str | None,
    mode: str,
) -> dict[str, object]:
    """Build a serializable result record for a single document processing run."""
    if pipeline_result is not None:
        quality_report: dict[str, object] = pipeline_result.get("quality_report", {})  # type: ignore[assignment]
        quality_score: float = float(quality_report.get("quality_score", 0.0))
        summary_raw: object = quality_report.get("summary")
        quality_severity: str = "unknown"
        if isinstance(summary_raw, dict):
            sev: object = summary_raw.get("severity")
            if isinstance(sev, str):
                quality_severity = sev

        structure: object = pipeline_result.get("structure")
        structure_summary: dict[str, object] = {}
        if isinstance(structure, DocumentStructure):
            structure_summary = build_structure_summary(structure)

        v2_chunks: list[dict[str, object]] = pipeline_result.get("v2_chunks", [])  # type: ignore[assignment]
        chunk_count: int = len(v2_chunks)
        duration_ms: int = int(pipeline_result.get("duration_ms", 0))
    else:
        quality_score = 0.0
        quality_severity = "unknown"
        structure_summary = {}
        chunk_count = 0
        duration_ms = 0

    return {
        "document_id": document_id,
        "blob_path": blob_path,
        "file_path": file_path,
        "input_source": input_source,
        "mode": mode,
        "pipeline_version": "v2",
        "quality_score": quality_score,
        "quality_severity": quality_severity,
        "chunk_count": chunk_count,
        "shadow_compare": shadow_compare,
        "structure_summary": structure_summary,
        "status": status,
        "error": error,
        "duration_ms": duration_ms,
    }


# ── Quality report to CSV row ────────────────────────────────────────────────


def quality_report_to_flat_row(
    record: dict[str, object],
    quality_report: dict[str, object] | None,
) -> dict[str, object]:
    """Flatten a result record + quality report into a single dict suitable for CSV."""
    checks: dict[str, object] = {}
    if quality_report is not None:
        raw_checks: object = quality_report.get("checks", {})
        if isinstance(raw_checks, dict):
            checks = raw_checks

    struct_summary: object = record.get("structure_summary", {})
    if not isinstance(struct_summary, dict):
        struct_summary = {}

    has_visible_table_tokens: bool = not _check_passed(
        checks.get("has_visible_table_tokens"),
    )
    has_header_footer_bleed: bool = not _check_passed(
        checks.get("header_footer_bleed"),
    )
    has_toc_duplicates: bool = not _check_passed(
        checks.get("toc_duplicate_ratio"),
    )
    has_orphan_tables: bool = not _check_passed(
        checks.get("orphan_tables_count"),
    )

    coverage_raw: object = checks.get("article_ref_coverage", {})
    low_article_ref_coverage: bool = False
    if isinstance(coverage_raw, dict):
        low_article_ref_coverage = not bool(coverage_raw.get("passed", True))

    reasons: list[str] = []
    if quality_report is not None:
        summary: object = quality_report.get("summary", {})
        if isinstance(summary, dict):
            raw_reasons: object = summary.get("reasons", [])
            if isinstance(raw_reasons, list):
                reasons = [str(r) for r in raw_reasons]

    return {
        "document_id": record.get("document_id", ""),
        "blob_path": record.get("blob_path", ""),
        "file_path": record.get("file_path", ""),
        "input_source": record.get("input_source", ""),
        "quality_score": record.get("quality_score", 0.0),
        "quality_severity": record.get("quality_severity", "unknown"),
        "chunk_count": record.get("chunk_count", 0),
        "article_count": struct_summary.get("total_articles", 0),
        "table_count": struct_summary.get("total_tables", 0),
        "has_visible_table_tokens": has_visible_table_tokens,
        "has_header_footer_bleed": has_header_footer_bleed,
        "low_article_ref_coverage": low_article_ref_coverage,
        "has_orphan_tables": has_orphan_tables,
        "has_toc_duplicates": has_toc_duplicates,
        "status": record.get("status", ""),
        "top_reason": reasons[0] if reasons else "",
        "all_reasons": ";".join(reasons),
        "duration_ms": record.get("duration_ms", 0),
    }


def _check_passed(check_result: object) -> bool:
    if isinstance(check_result, dict):
        return bool(check_result.get("passed", True))
    return True


# ── Aggregate metrics builder ────────────────────────────────────────────────


def compute_aggregate_metrics(
    records: list[dict[str, object]],
    quality_reports: list[dict[str, object] | None],
) -> dict[str, object]:
    """Compute aggregate metrics from a list of processing result records."""
    total: int = len(records)
    processed: int = sum(1 for r in records if r.get("status") == "ok")
    failed: int = sum(1 for r in records if r.get("status") == "failed")
    skipped: int = sum(1 for r in records if r.get("status") == "skipped")

    scores: list[float] = [
        float(r.get("quality_score", 0.0))
        for r in records
        if r.get("status") == "ok"
    ]

    severities: dict[str, int] = {}
    for r in records:
        if r.get("status") != "ok":
            continue
        sev: str = str(r.get("quality_severity", "unknown"))
        severities[sev] = severities.get(sev, 0) + 1

    passed: int = severities.get("low", 0)
    warning: int = severities.get("medium", 0)
    rejected: int = severities.get("high", 0)

    avg_score: float = sum(scores) / len(scores) if scores else 0.0
    median_score: float = _median(scores)

    score_buckets: dict[str, int] = _score_buckets(scores)

    failure_reasons: dict[str, int] = {}
    warning_reasons: dict[str, int] = {}
    doc_visible_table_tokens: int = 0
    doc_header_footer_bleed: int = 0
    doc_low_article_ref: int = 0
    doc_orphan_tables: int = 0
    doc_toc_duplicates: int = 0

    for qr in quality_reports:
        if qr is None:
            continue
        checks_raw: object = qr.get("checks", {})
        if not isinstance(checks_raw, dict):
            continue
        checks: dict[str, object] = checks_raw

        for check_name, check_result in checks.items():
            if not isinstance(check_result, dict):
                continue
            if not check_result.get("passed", True):
                failure_reasons[check_name] = failure_reasons.get(check_name, 0) + 1

        summary_raw: object = qr.get("summary", {})
        if isinstance(summary_raw, dict):
            sev_val: object = summary_raw.get("severity")
            if sev_val == "medium":
                raw_reasons: object = summary_raw.get("reasons", [])
                if isinstance(raw_reasons, list):
                    for reason in raw_reasons:
                        warning_reasons[str(reason)] = warning_reasons.get(str(reason), 0) + 1

        if not _check_passed(checks.get("has_visible_table_tokens")):
            doc_visible_table_tokens += 1
        if not _check_passed(checks.get("header_footer_bleed")):
            doc_header_footer_bleed += 1
        if not _check_passed(checks.get("article_ref_coverage")):
            doc_low_article_ref += 1
        if not _check_passed(checks.get("orphan_tables_count")):
            doc_orphan_tables += 1
        if not _check_passed(checks.get("toc_duplicate_ratio")):
            doc_toc_duplicates += 1

    return {
        "total_documents": total,
        "processed_documents": processed,
        "failed_documents": failed,
        "skipped_documents": skipped,
        "passed_documents": passed,
        "warning_documents": warning,
        "rejected_documents": rejected,
        "avg_quality_score": round(avg_score, 4),
        "median_quality_score": round(median_score, 4),
        "severity_distribution": severities,
        "quality_score_buckets": score_buckets,
        "top_failure_reasons": dict(
            sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True)
        ),
        "top_warning_reasons": dict(
            sorted(warning_reasons.items(), key=lambda x: x[1], reverse=True)
        ),
        "documents_with_visible_table_tokens": doc_visible_table_tokens,
        "documents_with_header_footer_bleed": doc_header_footer_bleed,
        "documents_with_low_article_ref_coverage": doc_low_article_ref,
        "documents_with_orphan_tables": doc_orphan_tables,
        "documents_with_toc_duplicates": doc_toc_duplicates,
    }


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_vals: list[float] = sorted(values)
    n: int = len(sorted_vals)
    mid: int = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
    return sorted_vals[mid]


def _score_buckets(scores: list[float]) -> dict[str, int]:
    buckets: dict[str, int] = {
        "0.0-0.5": 0,
        "0.5-0.7": 0,
        "0.7-0.85": 0,
        "0.85-1.0": 0,
    }
    for s in scores:
        if s < 0.5:
            buckets["0.0-0.5"] += 1
        elif s < 0.7:
            buckets["0.5-0.7"] += 1
        elif s < 0.85:
            buckets["0.7-0.85"] += 1
        else:
            buckets["0.85-1.0"] += 1
    return buckets


# ── Title inference from file path ───────────────────────────────────────────


def title_from_path(file_path: str) -> str:
    """Derive a document title from a file path (stem without extension)."""
    return Path(file_path).stem
