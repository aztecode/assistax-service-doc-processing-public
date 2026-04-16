"""
Pool de conexiones PostgreSQL para el pipeline.
Usado por health check, cleanup job y pipeline completo.
"""
import json
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Optional

import psycopg2
from psycopg2 import pool
from psycopg2.extras import execute_values

from pipeline.toc_builder import (
    STRATEGY_CHUNK_BASED,
    STRATEGY_NATIVE_HYBRID,
    build_native_toc_hybrid_tree,
    build_toc_tree,
    manifest_version_from_toc,
    merge_top_level_metadata,
    native_toc_stats,
    sections_from_toc_tree,
)
from settings import settings

_pool: Optional[pool.ThreadedConnectionPool] = None


def init_pool() -> None:
    """Inicializa el pool de conexiones. Falla si DATABASE_URL es inválida."""
    global _pool
    _pool = pool.ThreadedConnectionPool(
        minconn=settings.DB_POOL_MIN_CONN,
        maxconn=settings.DB_POOL_MAX_CONN,
        dsn=settings.DATABASE_URL,
    )


def close_pool() -> None:
    """Cierra todas las conexiones del pool."""
    global _pool
    if _pool:
        _pool.closeall()
        _pool = None


@contextmanager
def get_db_conn():
    """Context manager que entrega una conexión del pool."""
    if _pool is None:
        raise RuntimeError("Pool no inicializado. Llamar a init_pool() antes.")
    conn = _pool.getconn()
    try:
        yield conn
    finally:
        _pool.putconn(conn)


def check_database() -> str:
    """
    Verifica conectividad con PostgreSQL.
    Retorna 'ok' si la conexión funciona, o mensaje de error descriptivo.
    """
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return "ok"
    except Exception as e:
        return f"error: {e}"


def check_duplicate_by_hash(conn, content_hash: str) -> str | None:
    """
    Retorna document_id si el PDF ya fue indexado, None si es nuevo.
    Requiere que legal_documents tenga columna contentHash.
    """
    with conn.cursor() as cursor:
        cursor.execute(
            'SELECT id FROM legal_documents WHERE "contentHash" = %s',
            (content_hash,),
        )
        row = cursor.fetchone()
        return str(row[0]) if row else None


def update_index_run(
    conn,
    run_id: str,
    status: str,
    docs_indexed: int | None = None,
    chunks_total: int | None = None,
    error_log: str | None = None,
) -> None:
    """
    Actualiza el estado de un index_run.
    Alineado con assistax-fn: siempre setea endedAt.
    """
    set_clauses = ["status = %s", '"endedAt" = NOW()']
    params: list = [status]

    if docs_indexed is not None:
        set_clauses.append('"docsIndexed" = %s')
        params.append(docs_indexed)
    if chunks_total is not None:
        set_clauses.append('"chunksTotal" = %s')
        params.append(chunks_total)
    if error_log is not None:
        set_clauses.append('error = %s')
        params.append(error_log)

    params.append(run_id)
    with conn.cursor() as cursor:
        cursor.execute(
            f"UPDATE index_runs SET {', '.join(set_clauses)} WHERE id = %s",
            params,
        )
        conn.commit()


def update_index_run_progress(
    conn,
    run_id: str,
    processed_chunks: int,
    chunks_total: int,
) -> None:
    """
    Actualiza progreso de embeddings durante el pipeline.
    Alineado con assistax-fn: usa "docsIndexed" para chunks. El caller hace commit.
    """
    with conn.cursor() as cursor:
        cursor.execute(
            """
            UPDATE index_runs
            SET "docsIndexed" = %s, "chunksTotal" = %s
            WHERE id = %s
            """,
            (processed_chunks, chunks_total, run_id),
        )


def upsert_legal_document(
    conn,
    *,
    blob_path: str,
    document_title: str,
    category_id: str,
    publish_date: str | None,
    blob_container: str,
    content_hash: str,
    doc_type: str,
    law_name: str,
    metadata: dict[str, Any] | None,
    jurisdiction: str = "MX-FEDERAL",
    source_type: str = "official_gazette",
) -> str:
    """
    Upsert de legal_document por blobPath.
    Si existe: UPDATE. Si no: INSERT.
    Retorna document_id. No hace commit (caller controla la transacción).
    Requiere extensión unaccent para metadata JSON en algunos setups.
    """
    title_truncated = (document_title or "")[:500]
    law_name_truncated = (law_name or "")[:255] if law_name else None

    # Parsear publish_date: ISO string o None
    publish_date_val = None
    if publish_date:
        try:
            publish_date_val = datetime.strptime(publish_date[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            pass

    with conn.cursor() as cursor:
        cursor.execute(
            'SELECT id FROM legal_documents WHERE "blobPath" = %s LIMIT 1',
            (blob_path,),
        )
        row = cursor.fetchone()

        metadata_json = json.dumps(metadata) if metadata else None

        if row:
            doc_id = str(row[0])
            cursor.execute(
                "SELECT metadata FROM legal_documents WHERE id = %s",
                (doc_id,),
            )
            meta_row = cursor.fetchone()
            existing_raw = meta_row[0] if meta_row else None
            existing_dict: dict[str, Any] = {}
            if isinstance(existing_raw, dict):
                existing_dict = dict(existing_raw)
            elif isinstance(existing_raw, str):
                try:
                    parsed = json.loads(existing_raw)
                    if isinstance(parsed, dict):
                        existing_dict = parsed
                except (json.JSONDecodeError, TypeError):
                    existing_dict = {}
            incoming_meta = metadata if isinstance(metadata, dict) else {}
            merged_doc_meta = merge_top_level_metadata(existing_dict, incoming_meta)
            metadata_json = json.dumps(merged_doc_meta)
            cursor.execute(
                """
                UPDATE legal_documents SET
                    title = %s, category_id = %s, "publishDate" = %s,
                    "contentHash" = %s, "docType" = %s, "lawName" = %s,
                    metadata = %s::jsonb, jurisdiction = %s, "updatedAt" = NOW()
                WHERE id = %s
                """,
                (
                    title_truncated,
                    category_id,
                    publish_date_val,
                    content_hash,
                    doc_type,
                    law_name_truncated,
                    metadata_json,
                    jurisdiction,
                    doc_id,
                ),
            )
        else:
            doc_id = str(uuid.uuid4())
            cursor.execute(
                """
                INSERT INTO legal_documents (
                    id, "blobContainer", "blobPath", title, category_id,
                    "publishDate", "contentHash", "sourceType", "sourceUri",
                    jurisdiction, "docType", "lawName", metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    doc_id,
                    blob_container,
                    blob_path,
                    title_truncated,
                    category_id,
                    publish_date_val,
                    content_hash,
                    source_type,
                    blob_path,  # sourceUri = blobPath como identificador
                    jurisdiction,
                    doc_type,
                    law_name_truncated,
                    metadata_json,
                ),
            )
    return doc_id


def delete_existing_chunks(conn, document_id: str) -> int:
    """
    Elimina chunks existentes del documento. Evita duplicados al re-indexar.
    Retorna número de filas eliminadas.
    """
    with conn.cursor() as cursor:
        cursor.execute('DELETE FROM legal_chunks WHERE "documentId" = %s', (document_id,))
        deleted = cursor.rowcount
    return deleted


def insert_chunks_bulk(
    conn,
    document_id: str,
    chunks: list,
    embeddings: list[list[float]],
    batch_size: int = 500,
) -> int:
    """
    Inserta chunks en lotes usando execute_values.
    tsv = to_tsvector('spanish', unaccent(text)).
    Requiere extensión unaccent en PostgreSQL.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks ({len(chunks)}) y embeddings ({len(embeddings)}) deben tener la misma longitud"
        )

    def _truncate(s: str | None, max_len: int) -> str | None:
        if s is None:
            return None
        return s[:max_len] if len(s) > max_len else s

    def _vec_str(vec: list[float]) -> str:
        return "[" + ",".join(str(x) for x in vec) + "]"

    # Construir filas para execute_values
    template = """(
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s::vector,
        to_tsvector('spanish', unaccent(%s))
    )"""
    columns = (
        'id, "documentId", "chunkNo", "chunkType", "articleRef", heading, text, '
        '"tokenCount", "startPage", "endPage", embedding, tsv'
    )

    total_inserted = 0
    for batch_start in range(0, len(chunks), batch_size):
        batch_chunks = chunks[batch_start : batch_start + batch_size]
        batch_embeddings = embeddings[batch_start : batch_start + batch_size]
        values = []
        for i, (chunk, emb) in enumerate(zip(batch_chunks, batch_embeddings)):
            chunk_no = batch_start + i + 1
            token_count = len(chunk.text.split()) if chunk.text else 0
            heading = _truncate(chunk.heading, 500)
            article_ref = _truncate(chunk.article_ref, 100)
            values.append(
                (
                    str(uuid.uuid4()),
                    document_id,
                    chunk_no,
                    chunk.chunk_type,
                    article_ref,
                    heading or "",
                    chunk.text or "",
                    token_count,
                    chunk.start_page,
                    chunk.end_page,
                    _vec_str(emb),
                    chunk.text or "",
                )
            )
        execute_values(
            conn.cursor(),
            f'INSERT INTO legal_chunks ({columns}) VALUES %s',
            values,
            template=template,
        )
        total_inserted += len(values)
    conn.commit()
    return total_inserted


def merge_legal_document_metadata(
    conn,
    document_id: str,
    patch: dict[str, Any],
) -> None:
    """
    Shallow-merge patch into legal_documents.metadata (preserves unrelated keys).
    Caller commits.
    """
    with conn.cursor() as cursor:
        cursor.execute(
            'SELECT metadata FROM legal_documents WHERE id = %s::uuid',
            (document_id,),
        )
        row = cursor.fetchone()
        existing_raw = row[0] if row else None
        existing_dict: dict[str, Any] = {}
        if isinstance(existing_raw, dict):
            existing_dict = dict(existing_raw)
        elif isinstance(existing_raw, str):
            try:
                parsed = json.loads(existing_raw)
                if isinstance(parsed, dict):
                    existing_dict = parsed
            except (json.JSONDecodeError, TypeError):
                existing_dict = {}
        merged = merge_top_level_metadata(existing_dict, patch)
        cursor.execute(
            """
            UPDATE legal_documents
            SET metadata = %s::jsonb, "updatedAt" = NOW()
            WHERE id = %s::uuid
            """,
            (json.dumps(merged), document_id),
        )


def _chunks_to_rows(chunks: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        rows.append(
            {
                "chunkNo": i + 1,
                "chunkType": chunk.chunk_type,
                "heading": chunk.heading if chunk.heading else None,
                "articleRef": chunk.article_ref,
                "text": chunk.text or "",
                "startPage": chunk.start_page,
                "endPage": chunk.end_page,
            }
        )
    return rows


def persist_legal_outline(
    conn,
    document_id: str,
    content_hash: str,
    page_count: int,
    chunks: list[Any],
    pdf_native_toc: list[dict[str, Any]] | None,
) -> int:
    """
    Build TOC (hybrid when native TOC is available, chunk-based otherwise)
    and persist into legal_documents.metadata.
    Returns total TOC node count.
    """
    rows = _chunks_to_rows(chunks)

    use_hybrid = pdf_native_toc is not None and len(pdf_native_toc) > 0
    if use_hybrid:
        tree, stats = build_native_toc_hybrid_tree(pdf_native_toc, rows)
        strategy = STRATEGY_NATIVE_HYBRID
    else:
        tree, stats = build_toc_tree(rows)
        strategy = STRATEGY_CHUNK_BASED

    sections = sections_from_toc_tree(tree)
    manifest_version = manifest_version_from_toc(content_hash, tree, strategy)
    generated = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    patch: dict[str, Any] = {
        "toc": tree,
        "sections": sections,
        "manifestVersion": manifest_version,
        "pageCount": page_count,
        "outlineStrategy": strategy,
        "generatedAt": generated,
        "outlineStats": stats,
        "outlineError": None,
    }
    if pdf_native_toc is not None:
        patch["nativeTocStats"] = native_toc_stats(pdf_native_toc)
        patch["sourcePdfToc"] = pdf_native_toc

    merge_legal_document_metadata(conn, document_id, patch)
    return int(stats.get("totalNodes", 0))


def persist_legal_outline_from_chunks(
    conn,
    document_id: str,
    content_hash: str,
    page_count: int,
    chunks: list[Any],
) -> int:
    """Backward-compatible wrapper: chunk-based only (no native TOC)."""
    return persist_legal_outline(
        conn, document_id, content_hash, page_count, chunks, None
    )
