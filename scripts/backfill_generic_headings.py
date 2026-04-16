#!/usr/bin/env python3
"""
Backfill: corregir headings de chunks generic (prosa mal etiquetada como título) vía LLM.

No recalcula embeddings (solo se usa chunk.text para vectores).

Uso (venv + DATABASE_URL + credenciales Azure OpenAI si --enable-llm):
  python scripts/backfill_generic_headings.py --doc-id <uuid> --enable-llm --dry-run
  python scripts/backfill_generic_headings.py --limit 5 --enable-llm
  python scripts/backfill_generic_headings.py --doc-id <uuid> --enable-llm --force-all-generic

Tras actualizar headings, reconstruye toc/sections en metadata (mismo criterio que backfill_legal_toc).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import psycopg2
from psycopg2.extras import RealDictCursor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.heading_refinement import compute_refined_heading_for_generic_row  # noqa: E402
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
from settings import settings  # noqa: E402


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


def _refresh_outline_for_doc(
    conn,
    doc_id: str,
    content_hash: str,
    existing_meta: dict,
    chunk_rows: list[dict],
) -> None:
    py_rows = _build_chunk_rows(chunk_rows)
    pdf_native_toc: list[dict] | None = None
    effective_strategy = STRATEGY_CHUNK_BASED
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


def _run(
    doc_id: str | None,
    limit: int | None,
    dry_run: bool,
    enable_llm: bool,
    force_all_generic: bool,
) -> None:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL is required")

    force_llm = enable_llm or settings.ENABLE_LLM_GENERIC_HEADING_REFINE
    if not force_llm:
        raise RuntimeError(
            "LLM is disabled. Set ENABLE_LLM_GENERIC_HEADING_REFINE=true in .env "
            "or pass --enable-llm (requires Azure OpenAI env vars)."
        )

    refine_all = force_all_generic or settings.LLM_GENERIC_HEADING_REFINE_ALL

    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if doc_id is not None:
                cur.execute(
                    """
                    SELECT d.id AS doc_id, d.title AS doc_title, d."contentHash" AS content_hash,
                           d.metadata AS metadata
                    FROM legal_documents d
                    WHERE d.id = %s::uuid
                      AND EXISTS (SELECT 1 FROM legal_chunks c WHERE c."documentId" = d.id)
                    """,
                    (doc_id,),
                )
            else:
                cur.execute(
                    """
                    SELECT d.id AS doc_id, d.title AS doc_title, d."contentHash" AS content_hash,
                           d.metadata AS metadata
                    FROM legal_documents d
                    WHERE EXISTS (SELECT 1 FROM legal_chunks c WHERE c."documentId" = d.id)
                    ORDER BY d."updatedAt" ASC
                    """
                    + (f" LIMIT {int(limit)}" if limit is not None else "")
                )
            candidates = list(cur.fetchall())

        for row in candidates:
            did = str(row["doc_id"])
            doc_title = str(row["doc_title"] or "")
            content_hash = str(row["content_hash"] or "")
            existing_meta = row.get("metadata")
            if isinstance(existing_meta, str):
                existing_meta = json.loads(existing_meta)
            if not isinstance(existing_meta, dict):
                existing_meta = {}

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, "chunkNo", "chunkType", heading, "articleRef", text,
                           "startPage", "endPage"
                    FROM legal_chunks
                    WHERE "documentId" = %s::uuid
                    ORDER BY "chunkNo" ASC
                    """,
                    (did,),
                )
                chunk_rows = list(cur.fetchall())

            text_head_sample = ""
            for c in chunk_rows[:3]:
                text_head_sample += str(c.get("text") or "") + "\n"
            text_head_sample = text_head_sample[:4000]

            updates: list[tuple[str, str]] = []
            for c in chunk_rows:
                if str(c.get("chunkType") or "") != "generic":
                    continue
                ar = c.get("articleRef")
                if ar is not None and str(ar).strip():
                    continue
                h = c.get("heading")
                if not h or not str(h).strip():
                    continue
                new_h = compute_refined_heading_for_generic_row(
                    str(h),
                    str(c.get("text") or ""),
                    doc_title,
                    text_head_sample,
                    refine_all=refine_all,
                    force_llm=force_llm,
                )
                if new_h is not None:
                    updates.append((new_h, str(c["id"])))

            if dry_run:
                print(
                    f"DRY-RUN doc_id={did} would_update={len(updates)} "
                    f"refine_all={refine_all}"
                )
                continue

            if not updates:
                print(f"SKIP doc_id={did} (no heading changes)")
                continue

            with conn.cursor() as cur:
                for new_heading, chunk_uuid in updates:
                    cur.execute(
                        """
                        UPDATE legal_chunks
                        SET heading = %s
                        WHERE id = %s::uuid
                        """,
                        (new_heading, chunk_uuid),
                    )

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT "chunkNo", "chunkType", heading, "articleRef", text,
                           "startPage", "endPage"
                    FROM legal_chunks
                    WHERE "documentId" = %s::uuid
                    ORDER BY "chunkNo" ASC
                    """,
                    (did,),
                )
                refreshed = list(cur.fetchall())

            _refresh_outline_for_doc(conn, did, content_hash, existing_meta, refreshed)
            conn.commit()
            print(f"OK doc_id={did} chunks_updated={len(updates)} outline_refreshed")
    finally:
        conn.close()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Backfill generic chunk headings via LLM and refresh TOC metadata"
    )
    p.add_argument("--doc-id", type=str, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print how many rows would change per document; no DB writes",
    )
    p.add_argument(
        "--enable-llm",
        action="store_true",
        help="Run Azure OpenAI even if ENABLE_LLM_GENERIC_HEADING_REFINE is false in .env",
    )
    p.add_argument(
        "--force-all-generic",
        action="store_true",
        help="Evaluate every generic chunk without articleRef (not only suspicious headings)",
    )
    args = p.parse_args()
    _run(args.doc_id, args.limit, args.dry_run, args.enable_llm, args.force_all_generic)


if __name__ == "__main__":
    main()
