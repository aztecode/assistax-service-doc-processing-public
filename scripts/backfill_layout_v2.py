#!/usr/bin/env python3
"""
Backfill / reprocess documents with the v2 layout pipeline.

Supports multiple input sources (doc_id, blob_path, local file, local directory)
and multiple execution modes (dry-run, shadow, force). Never writes production
data by default — use explicit flags to opt in.

Usage examples:
  python scripts/backfill_layout_v2.py --file tmp/pdf_legales/2025/LEYES/ley.pdf --dry-run
  python scripts/backfill_layout_v2.py --input-dir tmp/pdf_legales --glob "*.pdf" --limit 5
  python scripts/backfill_layout_v2.py --doc-id <uuid> --shadow-mode
  python scripts/backfill_layout_v2.py --blob-path laws/2025/doc.pdf --force
  python scripts/backfill_layout_v2.py --input-dir tmp/pdf_legales --output-json results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import structlog  # noqa: E402

from scripts._v2_eval_helpers import (  # noqa: E402
    InputSource,
    build_result_record,
    discover_local_pdfs,
    read_local_pdf,
    run_v2_pipeline_eval,
    title_from_path,
)

_logger = structlog.get_logger()


# ── Input resolution ─────────────────────────────────────────────────────────


def _resolve_inputs(args: argparse.Namespace) -> list[dict[str, object]]:
    """Build a list of input descriptors from CLI arguments.

    Each descriptor has: source, pdf_bytes | blob_path | doc_id, title, file_path.
    """
    inputs: list[dict[str, object]] = []

    if args.doc_id is not None:
        inputs.append({
            "source": "doc_id",
            "doc_id": args.doc_id,
            "title": args.document_title or "",
            "blob_path": None,
            "file_path": None,
        })

    if args.blob_path is not None:
        inputs.append({
            "source": "blob_path",
            "doc_id": None,
            "blob_path": args.blob_path,
            "title": args.document_title or title_from_path(args.blob_path),
            "file_path": None,
        })

    if args.file is not None:
        inputs.append({
            "source": "file",
            "doc_id": None,
            "blob_path": None,
            "file_path": args.file,
            "title": args.document_title or title_from_path(args.file),
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

    if not inputs:
        print("ERROR: No input specified. Use --doc-id, --blob-path, --file, or --input-dir.")
        sys.exit(1)

    if args.limit is not None and args.limit > 0:
        inputs = inputs[: args.limit]

    return inputs


# ── PDF bytes loader ─────────────────────────────────────────────────────────


def _load_pdf_bytes(descriptor: dict[str, object]) -> bytes:
    """Load PDF bytes from the appropriate source."""
    source: str = str(descriptor["source"])

    if source in ("file", "input_dir"):
        file_path: str = str(descriptor["file_path"])
        return read_local_pdf(file_path)

    if source == "blob_path":
        from pipeline.blob_download import download_pdf_bytes
        blob_path: str = str(descriptor["blob_path"])
        return download_pdf_bytes(blob_path)

    if source == "doc_id":
        doc_id: str = str(descriptor["doc_id"])
        blob_path_resolved: str | None = _resolve_blob_path_from_db(doc_id)
        if blob_path_resolved is None:
            raise ValueError(f"No blob_path found for doc_id={doc_id}")
        descriptor["blob_path"] = blob_path_resolved
        from pipeline.blob_download import download_pdf_bytes
        return download_pdf_bytes(blob_path_resolved)

    raise ValueError(f"Unknown input source: {source}")


def _resolve_blob_path_from_db(doc_id: str) -> str | None:
    """Look up blob_path for a document by its UUID."""
    import psycopg2
    dsn: str | None = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL required to resolve doc_id")
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT "blobPath" FROM legal_documents WHERE id = %s::uuid',
                (doc_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return str(row[0])
    finally:
        conn.close()


def _resolve_title_from_db(doc_id: str) -> str:
    """Look up document title by UUID, returning empty string if not found."""
    import psycopg2
    dsn: str | None = os.environ.get("DATABASE_URL")
    if not dsn:
        return ""
    try:
        conn = psycopg2.connect(dsn)
        with conn.cursor() as cur:
            cur.execute(
                'SELECT title FROM legal_documents WHERE id = %s::uuid',
                (doc_id,),
            )
            row = cur.fetchone()
            return str(row[0]) if row else ""
    except Exception:
        return ""


# ── Shadow comparison ────────────────────────────────────────────────────────


def _run_shadow_compare(
    pdf_bytes: bytes,
    v2_chunks: list[dict[str, object]],
    quality_score: float,
    title: str,
) -> dict[str, object] | None:
    """Run legacy pipeline and compare with v2 results."""
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


# ── Main processing loop ────────────────────────────────────────────────────


def _process_document(
    descriptor: dict[str, object],
    dry_run: bool,
    shadow_mode: bool,
    force: bool,
    category_id: str | None,
    publish_date: str | None,
) -> dict[str, object]:
    """Process a single document and return a result record."""
    source: InputSource = descriptor["source"]  # type: ignore[assignment]
    doc_id: str | None = descriptor.get("doc_id")  # type: ignore[assignment]
    blob_path: str | None = descriptor.get("blob_path")  # type: ignore[assignment]
    file_path: str | None = descriptor.get("file_path")  # type: ignore[assignment]
    title: str = str(descriptor.get("title", ""))

    if source == "doc_id" and doc_id and not title:
        title = _resolve_title_from_db(str(doc_id))

    mode_parts: list[str] = ["v2"]
    if dry_run:
        mode_parts.append("dry_run")
    if shadow_mode:
        mode_parts.append("shadow")
    if force:
        mode_parts.append("force")
    mode: str = "+".join(mode_parts)

    _logger.info(
        "backfill.document.start",
        doc_id=doc_id,
        blob_path=blob_path,
        file_path=file_path,
        input_source=source,
        mode=mode,
        title=title[:60] if title else "",
    )

    try:
        pdf_bytes: bytes = _load_pdf_bytes(descriptor)
    except Exception as e:
        _logger.error("backfill.document.load_failed", error=str(e))
        return build_result_record(
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

    try:
        pipeline_result: dict[str, object] = run_v2_pipeline_eval(pdf_bytes, title)
    except Exception as e:
        _logger.error("backfill.document.pipeline_failed", error=str(e))
        return build_result_record(
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

    shadow_compare: dict[str, object] | None = None
    if shadow_mode:
        v2_chunks: list[dict[str, object]] = pipeline_result.get("v2_chunks", [])  # type: ignore[assignment]
        quality_score: float = float(
            pipeline_result.get("quality_report", {}).get("quality_score", 0.0)  # type: ignore[union-attr]
        )
        shadow_compare = _run_shadow_compare(pdf_bytes, v2_chunks, quality_score, title)

    quality_report: dict[str, object] = pipeline_result.get("quality_report", {})  # type: ignore[assignment]
    q_score: float = float(quality_report.get("quality_score", 0.0))
    q_severity: str = "unknown"
    summary_raw: object = quality_report.get("summary")
    if isinstance(summary_raw, dict):
        sev: object = summary_raw.get("severity")
        if isinstance(sev, str):
            q_severity = sev

    _logger.info(
        "backfill.document.completed",
        doc_id=doc_id,
        file_path=file_path,
        quality_score=q_score,
        quality_severity=q_severity,
        chunk_count=len(pipeline_result.get("v2_chunks", [])),  # type: ignore[arg-type]
        mode=mode,
        duration_ms=pipeline_result.get("duration_ms"),
    )

    return build_result_record(
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


# ── CLI ──────────────────────────────────────────────────────────────────────


def _print_summary(results: list[dict[str, object]]) -> None:
    total: int = len(results)
    ok: int = sum(1 for r in results if r["status"] == "ok")
    failed: int = sum(1 for r in results if r["status"] == "failed")
    scores: list[float] = [
        float(r["quality_score"]) for r in results if r["status"] == "ok"
    ]
    avg: float = sum(scores) / len(scores) if scores else 0.0

    print("\n" + "=" * 60)
    print("BACKFILL SUMMARY")
    print("=" * 60)
    print(f"  Total documents:    {total}")
    print(f"  Processed (ok):     {ok}")
    print(f"  Failed:             {failed}")
    if scores:
        print(f"  Avg quality score:  {avg:.4f}")
        print(f"  Min quality score:  {min(scores):.4f}")
        print(f"  Max quality score:  {max(scores):.4f}")
    print("=" * 60)

    if failed > 0:
        print("\nFailed documents:")
        for r in results:
            if r["status"] == "failed":
                label: str = str(r.get("file_path") or r.get("blob_path") or r.get("document_id") or "unknown")
                print(f"  - {label}: {r.get('error', 'unknown error')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill / reprocess documents with v2 layout pipeline",
    )

    input_group = parser.add_argument_group("Input sources")
    input_group.add_argument("--doc-id", type=str, help="Process by document UUID (requires DATABASE_URL)")
    input_group.add_argument("--blob-path", type=str, help="Process by Azure Blob path")
    input_group.add_argument("--file", type=str, help="Process a local PDF file")
    input_group.add_argument("--input-dir", type=str, help="Process PDFs from a local directory")
    input_group.add_argument("--glob", type=str, default="*.pdf", help="Glob filter for --input-dir (default: *.pdf)")
    input_group.add_argument("--limit", type=int, help="Max documents to process")

    mode_group = parser.add_argument_group("Execution modes")
    mode_group.add_argument("--dry-run", action="store_true", help="Run v2 pipeline but do not write to DB")
    mode_group.add_argument("--shadow-mode", action="store_true", help="Compare v2 vs legacy (no production changes)")
    mode_group.add_argument("--force", action="store_true", help="Reprocess even if already processed")

    meta_group = parser.add_argument_group("Document metadata (for manual input)")
    meta_group.add_argument("--category-id", type=str, help="Category ID override")
    meta_group.add_argument("--publish-date", type=str, help="Publish date override (ISO format)")
    meta_group.add_argument("--document-title", type=str, help="Document title override")

    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--output-json", type=str, help="Write results to JSON file")

    args = parser.parse_args()

    print(f"Backfill v2 — mode: {'dry-run' if args.dry_run else 'live'}"
          f"{' + shadow' if args.shadow_mode else ''}"
          f"{' + force' if args.force else ''}")

    inputs: list[dict[str, object]] = _resolve_inputs(args)
    print(f"Documents to process: {len(inputs)}")

    results: list[dict[str, object]] = []
    for i, descriptor in enumerate(inputs):
        label: str = str(
            descriptor.get("file_path")
            or descriptor.get("blob_path")
            or descriptor.get("doc_id")
            or "unknown"
        )
        print(f"\n[{i + 1}/{len(inputs)}] Processing: {label}")

        result: dict[str, object] = _process_document(
            descriptor,
            dry_run=args.dry_run,
            shadow_mode=args.shadow_mode,
            force=args.force,
            category_id=args.category_id,
            publish_date=args.publish_date,
        )
        results.append(result)

        status: str = str(result["status"])
        score: float = float(result.get("quality_score", 0.0))
        severity: str = str(result.get("quality_severity", ""))
        print(f"  → status={status}  score={score:.4f}  severity={severity}")

    _print_summary(results)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nResults written to: {args.output_json}")


if __name__ == "__main__":
    main()
