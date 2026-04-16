#!/usr/bin/env python3
"""
Backfill: reconstruir toc/sections desde legal_chunks y guardar en legal_documents.metadata.

Supports two strategies:
  --strategy chunk_based   (default) pure chunk-based tree
  --strategy hybrid        native_toc_hybrid_v1 when sourcePdfToc exists in metadata

Uso (con venv y DATABASE_URL):
  python scripts/backfill_legal_toc.py [--limit N] [--strategy hybrid]
  python scripts/backfill_legal_toc.py --force --limit 5    # reprocess docs that already have TOC
  python scripts/backfill_legal_toc.py --doc-id <uuid>      # reprocess a specific document
  python scripts/backfill_legal_toc.py --enable-arbiter     # enable LLM boxed note arbiter

Requiere migración con columnas startPage/endPage (opcional; si faltan, target.page=1).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import psycopg2
from psycopg2.extras import RealDictCursor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.toc_builder import (  # noqa: E402
    STRATEGY_CHUNK_BASED,
    STRATEGY_NATIVE_HYBRID,
    build_native_toc_hybrid_tree,
    build_toc_tree,
    manifest_version_from_toc,
    merge_top_level_metadata,
    native_toc_stats,
    sections_from_toc_tree,
)
from datetime import datetime, timezone  # noqa: E402


def _build_chunk_rows(chunk_rows: list[dict]) -> list[dict]:
    py_rows: list[dict] = []
    for c in chunk_rows:
        py_rows.append(
            {
                "chunkNo": int(c["chunkNo"]),
                "chunkType": str(c["chunkType"] or ""),
                "heading": c.get("heading"),
                "articleRef": c.get("articleRef"),
                "text": str(c.get("text") or ""),
                "startPage": c.get("startPage"),
                "endPage": c.get("endPage"),
            }
        )
    return py_rows


def _max_page(chunk_rows: list[dict]) -> int:
    page_count = 1
    for c in chunk_rows:
        for key in ("endPage", "startPage"):
            v = c.get(key)
            if isinstance(v, int) and v >= page_count:
                page_count = v
    return page_count


def _run(
    limit: int | None,
    strategy: str,
    force: bool,
    doc_id: str | None,
) -> None:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL is required")

    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if doc_id is not None:
                cur.execute(
                    """
                    SELECT d.id AS doc_id, d."contentHash" AS content_hash,
                           d.metadata AS metadata
                    FROM legal_documents d
                    WHERE d.id = %s::uuid
                      AND EXISTS (
                          SELECT 1 FROM legal_chunks c WHERE c."documentId" = d.id
                      )
                    """,
                    (doc_id,),
                )
            elif force:
                cur.execute(
                    """
                    SELECT d.id AS doc_id, d."contentHash" AS content_hash,
                           d.metadata AS metadata
                    FROM legal_documents d
                    WHERE EXISTS (
                        SELECT 1 FROM legal_chunks c WHERE c."documentId" = d.id
                    )
                    ORDER BY d."updatedAt" ASC
                    """
                    + (f" LIMIT {int(limit)}" if limit is not None else "")
                )
            else:
                cur.execute(
                    """
                    SELECT d.id AS doc_id, d."contentHash" AS content_hash,
                           d.metadata AS metadata
                    FROM legal_documents d
                    WHERE EXISTS (
                        SELECT 1 FROM legal_chunks c WHERE c."documentId" = d.id
                    )
                    AND (
                        d.metadata IS NULL
                        OR (d.metadata::text NOT LIKE '%%"toc"%%')
                        OR (d.metadata->>'toc' IS NULL)
                        OR (d.metadata->'toc' = '[]'::jsonb)
                    )
                    ORDER BY d."updatedAt" ASC
                    """
                    + (f" LIMIT {int(limit)}" if limit is not None else "")
                )
            candidates = list(cur.fetchall())

        for row in candidates:
            doc_id = str(row["doc_id"])
            content_hash = str(row["content_hash"] or "")
            existing_meta = row.get("metadata")
            if isinstance(existing_meta, str):
                existing_meta = json.loads(existing_meta)
            if not isinstance(existing_meta, dict):
                existing_meta = {}

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT "chunkNo", "chunkType", heading, "articleRef", text,
                           "startPage", "endPage"
                    FROM legal_chunks
                    WHERE "documentId" = %s::uuid
                    ORDER BY "chunkNo" ASC
                    """,
                    (doc_id,),
                )
                chunk_rows = list(cur.fetchall())

            py_rows = _build_chunk_rows(chunk_rows)

            pdf_native_toc: list[dict] | None = None
            effective_strategy = STRATEGY_CHUNK_BASED

            if strategy == "hybrid":
                stored_toc = existing_meta.get("sourcePdfToc")
                if isinstance(stored_toc, list) and len(stored_toc) > 0:
                    pdf_native_toc = stored_toc
                    effective_strategy = STRATEGY_NATIVE_HYBRID

            if pdf_native_toc is not None:
                tree, stats = build_native_toc_hybrid_tree(pdf_native_toc, py_rows)
            else:
                tree, stats = build_toc_tree(py_rows)

            sections = sections_from_toc_tree(tree)
            mv = manifest_version_from_toc(content_hash, tree, effective_strategy)
            page_count = _max_page(chunk_rows)
            generated = (
                datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )
            patch: dict = {
                "toc": tree,
                "sections": sections,
                "manifestVersion": mv,
                "pageCount": page_count,
                "outlineStrategy": effective_strategy,
                "generatedAt": generated,
                "outlineStats": stats,
                "outlineError": None,
            }
            if pdf_native_toc is not None:
                patch["nativeTocStats"] = native_toc_stats(pdf_native_toc)

            merged = merge_top_level_metadata(existing_meta, patch)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE legal_documents
                    SET metadata = %s::jsonb, "updatedAt" = NOW()
                    WHERE id = %s::uuid
                    """,
                    (json.dumps(merged), doc_id),
                )
            conn.commit()
            print(
                f"OK doc_id={doc_id} strategy={effective_strategy} "
                f"nodes={stats.get('totalNodes', 0)}"
            )
    finally:
        conn.close()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Backfill TOC into legal_documents.metadata"
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--strategy",
        choices=["chunk_based", "hybrid"],
        default="chunk_based",
        help="chunk_based (default) or hybrid (uses sourcePdfToc from metadata)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Reprocess documents that already have a TOC",
    )
    p.add_argument(
        "--doc-id",
        type=str,
        default=None,
        help="Reprocess a specific document by UUID",
    )
    p.add_argument(
        "--enable-arbiter",
        action="store_true",
        help="Enable LLM boxed note arbiter for this run (overrides env var)",
    )
    args = p.parse_args()

    if args.enable_arbiter:
        os.environ["ENABLE_LLM_BOXED_NOTE_ARBITER"] = "true"
        print("LLM boxed note arbiter ENABLED for this run")

    _run(args.limit, args.strategy, args.force, args.doc_id)


if __name__ == "__main__":
    main()
