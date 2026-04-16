#!/usr/bin/env python3
"""
Evaluate a corpus of documents with the v2 layout pipeline and produce
aggregate quality metrics for promotion decisions.

Supports local PDFs (--input-dir, --file), DB-based discovery (--doc-id,
--blob-prefix), and hybrid modes. Never writes production data.

Usage examples:
  python scripts/evaluate_layout_v2_corpus.py --input-dir tmp/pdf_legales --limit 20 --output-csv eval.csv
  python scripts/evaluate_layout_v2_corpus.py --input-dir tmp/pdf_legales --output-json eval.json --shadow-mode
  python scripts/evaluate_layout_v2_corpus.py --doc-id <uuid> --output-json single.json
  python scripts/evaluate_layout_v2_corpus.py --input-dir tmp/pdf_legales --min-quality-score 0.85 --sample-errors 5
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import structlog  # noqa: E402

from scripts._v2_eval_helpers import (  # noqa: E402
    InputSource,
    build_result_record,
    compute_aggregate_metrics,
    discover_local_pdfs,
    quality_report_to_flat_row,
    read_local_pdf,
    run_v2_pipeline_eval,
    title_from_path,
)

_logger = structlog.get_logger()

# Threshold below which a document is considered "problematic"
_DEFAULT_MIN_QUALITY_SCORE: float = 0.70


# ── Input resolution ─────────────────────────────────────────────────────────


def _resolve_inputs(args: argparse.Namespace) -> list[dict[str, object]]:
    """Build input descriptors from CLI arguments."""
    inputs: list[dict[str, object]] = []

    if args.doc_id is not None:
        inputs.append({
            "source": "doc_id",
            "doc_id": args.doc_id,
            "blob_path": None,
            "file_path": None,
            "title": "",
        })

    if args.file is not None:
        inputs.append({
            "source": "file",
            "doc_id": None,
            "blob_path": None,
            "file_path": args.file,
            "title": title_from_path(args.file),
        })

    if args.input_dir is not None:
        glob_pattern: str = args.glob or "*.pdf"
        paths: list[str] = discover_local_pdfs(args.input_dir, glob_pattern)
        for p in paths:
            inputs.append({
                "source": "input_dir",
                "doc_id": None,
                "blob_path": None,
                "file_path": p,
                "title": title_from_path(p),
            })

    if args.blob_prefix is not None:
        db_inputs: list[dict[str, object]] = _discover_from_db(
            blob_prefix=args.blob_prefix,
            category_id=args.category_id,
        )
        inputs.extend(db_inputs)

    if not inputs:
        print("ERROR: No input specified. Use --doc-id, --file, --input-dir, or --blob-prefix.")
        sys.exit(1)

    if args.limit is not None and args.limit > 0:
        inputs = inputs[: args.limit]

    return inputs


def _discover_from_db(
    blob_prefix: str | None,
    category_id: str | None,
) -> list[dict[str, object]]:
    """Discover documents from DB by blob prefix and/or category."""
    import psycopg2
    from psycopg2.extras import RealDictCursor

    dsn: str | None = os.environ.get("DATABASE_URL")
    if not dsn:
        print("WARNING: DATABASE_URL not set — skipping DB discovery")
        return []

    conditions: list[str] = []
    params: list[object] = []

    if blob_prefix:
        conditions.append('"blobPath" LIKE %s')
        params.append(f"{blob_prefix}%")
    if category_id:
        conditions.append('"categoryId" = %s::uuid')
        params.append(category_id)

    where: str = " AND ".join(conditions) if conditions else "TRUE"

    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f'SELECT id, title, "blobPath" FROM legal_documents WHERE {where} ORDER BY "updatedAt" DESC',
                params,
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    result: list[dict[str, object]] = []
    for row in rows:
        result.append({
            "source": "doc_id",
            "doc_id": str(row["id"]),
            "blob_path": str(row["blobPath"]),
            "file_path": None,
            "title": str(row.get("title") or ""),
        })
    return result


# ── PDF loader ───────────────────────────────────────────────────────────────


def _load_pdf_bytes(descriptor: dict[str, object]) -> bytes:
    """Load PDF bytes from the appropriate source."""
    source: str = str(descriptor["source"])

    if source in ("file", "input_dir"):
        return read_local_pdf(str(descriptor["file_path"]))

    if source == "doc_id":
        blob_path: str | None = descriptor.get("blob_path")  # type: ignore[assignment]
        if not blob_path:
            blob_path = _resolve_blob_path(str(descriptor["doc_id"]))
            descriptor["blob_path"] = blob_path
        if not blob_path:
            raise ValueError(f"No blob_path for doc_id={descriptor['doc_id']}")
        from pipeline.blob_download import download_pdf_bytes
        return download_pdf_bytes(blob_path)

    raise ValueError(f"Unknown input source: {source}")


def _resolve_blob_path(doc_id: str) -> str | None:
    import psycopg2
    dsn: str | None = os.environ.get("DATABASE_URL")
    if not dsn:
        return None
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT "blobPath" FROM legal_documents WHERE id = %s::uuid', (doc_id,))
            row = cur.fetchone()
            return str(row[0]) if row else None
    finally:
        conn.close()


# ── Shadow comparison ────────────────────────────────────────────────────────


def _run_shadow_compare(
    pdf_bytes: bytes,
    v2_chunks: list[dict[str, object]],
    quality_score: float,
    title: str,
) -> dict[str, object] | None:
    try:
        from pipeline.pdf_extractor import extract_pdf
        from pipeline.legal_chunker import chunk_content

        pages, _ = extract_pdf(pdf_bytes)
        head_limit: int = min(3, len(pages))
        text_head_sample: str = "\n".join(p.text for p in pages[:head_limit])
        legacy_chunks = chunk_content(pages, 2000, title, text_head_sample)

        from pipeline.shadow_compare_v2 import compare_pipeline_outputs
        return compare_pipeline_outputs(
            legacy_chunks=legacy_chunks,
            v2_chunks=v2_chunks,
            legacy_metadata=None,
            v2_metadata={"quality_score": quality_score},
        )
    except Exception as e:
        _logger.warning("shadow_compare.failed", error=str(e))
        return {"error": str(e)}


# ── Evaluation loop ─────────────────────────────────────────────────────────


def _evaluate_document(
    descriptor: dict[str, object],
    shadow_mode: bool,
) -> tuple[dict[str, object], dict[str, object] | None]:
    """Evaluate a single document. Returns (result_record, quality_report)."""
    source: InputSource = descriptor["source"]  # type: ignore[assignment]
    doc_id: str | None = descriptor.get("doc_id")  # type: ignore[assignment]
    blob_path: str | None = descriptor.get("blob_path")  # type: ignore[assignment]
    file_path: str | None = descriptor.get("file_path")  # type: ignore[assignment]
    title: str = str(descriptor.get("title", ""))

    mode: str = "v2+eval"
    if shadow_mode:
        mode += "+shadow"

    try:
        pdf_bytes: bytes = _load_pdf_bytes(descriptor)
    except Exception as e:
        record: dict[str, object] = build_result_record(
            document_id=str(doc_id) if doc_id else None,
            blob_path=str(blob_path) if blob_path else None,
            file_path=str(file_path) if file_path else None,
            input_source=source,
            pipeline_result=None,
            shadow_compare=None,
            status="failed",
            error=f"load_failed: {e}",
            mode=mode,
        )
        return (record, None)

    try:
        pipeline_result: dict[str, object] = run_v2_pipeline_eval(pdf_bytes, title)
    except Exception as e:
        record = build_result_record(
            document_id=str(doc_id) if doc_id else None,
            blob_path=str(blob_path) if blob_path else None,
            file_path=str(file_path) if file_path else None,
            input_source=source,
            pipeline_result=None,
            shadow_compare=None,
            status="failed",
            error=f"pipeline_failed: {e}",
            mode=mode,
        )
        return (record, None)

    shadow_compare: dict[str, object] | None = None
    if shadow_mode:
        v2_chunks: list[dict[str, object]] = pipeline_result.get("v2_chunks", [])  # type: ignore[assignment]
        q_score: float = float(
            pipeline_result.get("quality_report", {}).get("quality_score", 0.0)  # type: ignore[union-attr]
        )
        shadow_compare = _run_shadow_compare(pdf_bytes, v2_chunks, q_score, title)

    quality_report: dict[str, object] = pipeline_result.get("quality_report", {})  # type: ignore[assignment]

    record = build_result_record(
        document_id=str(doc_id) if doc_id else None,
        blob_path=str(blob_path) if blob_path else None,
        file_path=str(file_path) if file_path else None,
        input_source=source,
        pipeline_result=pipeline_result,
        shadow_compare=shadow_compare,
        status="ok",
        error=None,
        mode=mode,
    )
    return (record, quality_report)


# ── Output writers ───────────────────────────────────────────────────────────


def _write_json(
    output_path: str,
    records: list[dict[str, object]],
    quality_reports: list[dict[str, object] | None],
    aggregates: dict[str, object],
    problematic: list[dict[str, object]],
) -> None:
    payload: dict[str, object] = {
        "summary": aggregates,
        "documents": records,
        "problematic_documents": problematic,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    print(f"JSON report written to: {output_path}")


def _write_csv(
    output_path: str,
    records: list[dict[str, object]],
    quality_reports: list[dict[str, object] | None],
) -> None:
    if not records:
        print("No records to write to CSV.")
        return

    rows: list[dict[str, object]] = []
    for record, qr in zip(records, quality_reports):
        rows.append(quality_report_to_flat_row(record, qr))

    fieldnames: list[str] = list(rows[0].keys())
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"CSV report written to: {output_path}")


# ── Console summary ─────────────────────────────────────────────────────────


def _print_console_summary(
    aggregates: dict[str, object],
    problematic: list[dict[str, object]],
    sample_errors: int,
) -> None:
    print("\n" + "=" * 70)
    print("CORPUS EVALUATION SUMMARY")
    print("=" * 70)

    print(f"  Total documents:        {aggregates['total_documents']}")
    print(f"  Processed (ok):         {aggregates['processed_documents']}")
    print(f"  Failed:                 {aggregates['failed_documents']}")
    print(f"  Passed (low severity):  {aggregates['passed_documents']}")
    print(f"  Warning (medium):       {aggregates['warning_documents']}")
    print(f"  Rejected (high):        {aggregates['rejected_documents']}")
    print(f"  Avg quality score:      {aggregates['avg_quality_score']}")
    print(f"  Median quality score:   {aggregates['median_quality_score']}")

    print("\n  Score distribution:")
    buckets: object = aggregates.get("quality_score_buckets", {})
    if isinstance(buckets, dict):
        for bucket, count in buckets.items():
            print(f"    {bucket}: {count}")

    print("\n  Severity distribution:")
    severities: object = aggregates.get("severity_distribution", {})
    if isinstance(severities, dict):
        for sev, count in severities.items():
            print(f"    {sev}: {count}")

    print("\n  Quality checks failing:")
    failure_reasons: object = aggregates.get("top_failure_reasons", {})
    if isinstance(failure_reasons, dict):
        for reason, count in list(failure_reasons.items())[:10]:
            print(f"    {reason}: {count}")

    print(f"\n  Docs with visible table tokens:     {aggregates.get('documents_with_visible_table_tokens', 0)}")
    print(f"  Docs with header/footer bleed:      {aggregates.get('documents_with_header_footer_bleed', 0)}")
    print(f"  Docs with low article ref coverage: {aggregates.get('documents_with_low_article_ref_coverage', 0)}")
    print(f"  Docs with orphan tables:            {aggregates.get('documents_with_orphan_tables', 0)}")
    print(f"  Docs with TOC duplicates:           {aggregates.get('documents_with_toc_duplicates', 0)}")

    if problematic:
        limit: int = min(sample_errors, len(problematic))
        print(f"\n  Top {limit} worst documents:")
        for doc in problematic[:limit]:
            label: str = str(doc.get("file_path") or doc.get("blob_path") or doc.get("document_id") or "unknown")
            score: float = float(doc.get("quality_score", 0.0))
            severity: str = str(doc.get("quality_severity", ""))
            error: str = str(doc.get("error") or "")
            status: str = str(doc.get("status", ""))
            if status == "failed":
                print(f"    - {label}: FAILED ({error})")
            else:
                print(f"    - {label}: score={score:.4f} severity={severity}")

    print("=" * 70)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate document corpus with v2 layout pipeline",
    )

    input_group = parser.add_argument_group("Input sources")
    input_group.add_argument("--doc-id", type=str, help="Evaluate a single document by UUID")
    input_group.add_argument("--file", type=str, help="Evaluate a single local PDF")
    input_group.add_argument("--input-dir", type=str, help="Evaluate PDFs from local directory")
    input_group.add_argument("--glob", type=str, default="*.pdf", help="Glob filter for --input-dir (default: *.pdf)")
    input_group.add_argument("--blob-prefix", type=str, help="Discover documents by blob path prefix (requires DB)")
    input_group.add_argument("--category-id", type=str, help="Filter by category ID (with --blob-prefix)")
    input_group.add_argument("--limit", type=int, help="Max documents to evaluate")

    mode_group = parser.add_argument_group("Evaluation options")
    mode_group.add_argument("--shadow-mode", action="store_true", help="Also run legacy and compare")
    mode_group.add_argument("--dry-run", action="store_true", help="Alias for default eval mode (no DB writes)")
    mode_group.add_argument("--min-quality-score", type=float, default=_DEFAULT_MIN_QUALITY_SCORE,
                            help=f"Threshold for 'problematic' (default: {_DEFAULT_MIN_QUALITY_SCORE})")
    mode_group.add_argument("--sample-errors", type=int, default=10,
                            help="Number of worst documents to show in summary (default: 10)")
    mode_group.add_argument("--include-passed", action="store_true",
                            help="Include passed documents in problematic list output")

    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--output-json", type=str, help="Write full JSON report")
    output_group.add_argument("--output-csv", type=str, help="Write CSV per-document report")

    args = parser.parse_args()

    print(f"Corpus evaluation v2 — min_quality_score={args.min_quality_score}"
          f"{' + shadow' if args.shadow_mode else ''}")

    inputs: list[dict[str, object]] = _resolve_inputs(args)
    print(f"Documents to evaluate: {len(inputs)}")

    t_start: float = time.perf_counter()
    records: list[dict[str, object]] = []
    quality_reports: list[dict[str, object] | None] = []

    for i, descriptor in enumerate(inputs):
        label: str = str(
            descriptor.get("file_path")
            or descriptor.get("blob_path")
            or descriptor.get("doc_id")
            or "unknown"
        )
        print(f"[{i + 1}/{len(inputs)}] {label}", end=" ", flush=True)

        record, qr = _evaluate_document(descriptor, shadow_mode=args.shadow_mode)
        records.append(record)
        quality_reports.append(qr)

        status: str = str(record["status"])
        score: float = float(record.get("quality_score", 0.0))
        print(f"→ {status} score={score:.4f}")

    total_duration_ms: int = round((time.perf_counter() - t_start) * 1000)
    print(f"\nTotal evaluation time: {total_duration_ms}ms")

    aggregates: dict[str, object] = compute_aggregate_metrics(records, quality_reports)

    problematic: list[dict[str, object]] = _identify_problematic(
        records, args.min_quality_score, args.include_passed,
    )

    _print_console_summary(aggregates, problematic, args.sample_errors)

    if args.output_json:
        _write_json(args.output_json, records, quality_reports, aggregates, problematic)

    if args.output_csv:
        _write_csv(args.output_csv, records, quality_reports)


def _identify_problematic(
    records: list[dict[str, object]],
    min_quality_score: float,
    include_passed: bool,
) -> list[dict[str, object]]:
    """Identify documents that are problematic based on quality criteria."""
    problematic: list[dict[str, object]] = []
    for r in records:
        status: str = str(r.get("status", ""))
        if status == "failed":
            problematic.append(r)
            continue
        if status != "ok":
            continue

        score: float = float(r.get("quality_score", 0.0))
        severity: str = str(r.get("quality_severity", ""))

        is_problem: bool = (
            score < min_quality_score
            or severity == "high"
        )

        if is_problem or include_passed:
            problematic.append(r)

    problematic.sort(key=lambda x: float(x.get("quality_score", 0.0)))
    return problematic


if __name__ == "__main__":
    main()
