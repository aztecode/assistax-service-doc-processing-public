"""
Orquestador del pipeline de procesamiento de PDFs.
Alineado con assistax-fn: sin dedup temprana, hash al final, metadata_extra, source_type.

Phase 7: supports legacy, v2, and shadow execution modes via feature flags.
"""
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import structlog
from settings import settings
from pipeline.db_writer import (
    get_db_conn,
    insert_chunks_bulk,
    merge_legal_document_metadata,
    persist_legal_outline,
    update_index_run,
    update_index_run_progress,
    upsert_legal_document,
    delete_existing_chunks,
)
from pipeline.blob_download import download_pdf_bytes
from pipeline.pdf_extractor import extract_pdf
from pipeline.heading_refinement import (
    refine_generic_chunk_headings,
    resolve_llm_heading_refinement_flags,
)
from pipeline.legal_chunker import Chunk, chunk_content
from pipeline.metadata_extractor import extract_legal_legend, extract_law_name
from pipeline.doc_type_classifier import classify_doc_type
from pipeline.embeddings import embed_chunks
from pipeline.exceptions import PDFExtractionError, EmbeddingError

_executor = ThreadPoolExecutor(max_workers=settings.PDF_WORKER_THREADS)
_pipeline_semaphore = threading.Semaphore(settings.PDF_WORKER_THREADS)
_logger = structlog.get_logger()


def _stage_from_exception(e: Exception) -> str:
    """Infiere la etapa donde falló según el tipo de excepción."""
    if isinstance(e, PDFExtractionError):
        return "pdf_extraction"
    if isinstance(e, EmbeddingError):
        return "embeddings"
    return "unknown"


# ── V2 pipeline helpers ─────────────────────────────────────────────────────


def _run_v2_extraction(
    pdf_bytes: bytes,
    document_title: str,
    run_id: str,
    blob_path: str,
) -> tuple[list[dict[str, object]], object, dict[str, object]]:
    """Run v2 extraction pipeline stages 1–6.

    Returns (v2_chunk_dicts, DocumentStructure, quality_report).
    Imports are deferred to avoid loading v2 modules when not needed.
    """
    from pipeline.layout_extractor_v2 import extract_document_layout
    from pipeline.layout_normalizer_v2 import normalize_document_layout
    from pipeline.block_classifier_v2 import classify_document_layout
    from pipeline.structure_builder_v2 import build_document_structure
    from pipeline.quality_validator_v2 import validate_document_structure
    from pipeline.chunk_projector_v2 import project_structure_to_chunks

    t_v2 = time.perf_counter()
    _logger.info(
        "layout_v2.pipeline.started",
        run_id=run_id,
        blob_path=blob_path,
        pipeline_version="v2",
    )

    layout = extract_document_layout(pdf_bytes)
    normalized = normalize_document_layout(layout)
    classified = classify_document_layout(normalized)
    structure = build_document_structure(
        classified, {"document_title": document_title},
    )
    quality_report: dict[str, object] = validate_document_structure(structure)
    structure = structure.model_copy(update={"quality_report": quality_report})
    v2_chunks: list[dict[str, object]] = project_structure_to_chunks(structure)

    quality_score: float = float(quality_report.get("quality_score", 0.0))
    summary_raw: object = quality_report.get("summary")
    quality_severity: str = "unknown"
    if isinstance(summary_raw, dict):
        sev: object = summary_raw.get("severity")
        if isinstance(sev, str):
            quality_severity = sev

    duration_ms: int = round((time.perf_counter() - t_v2) * 1000)
    _logger.info(
        "layout_v2.pipeline.completed",
        run_id=run_id,
        blob_path=blob_path,
        pipeline_version="v2",
        quality_score=quality_score,
        quality_severity=quality_severity,
        chunk_count=len(v2_chunks),
        duration_ms=duration_ms,
    )
    _logger.info(
        "layout_v2.quality.completed",
        run_id=run_id,
        blob_path=blob_path,
        quality_score=quality_score,
        quality_severity=quality_severity,
    )

    return (v2_chunks, structure, quality_report)


def _adapt_v2_chunks_to_legacy(
    v2_chunks: list[dict[str, object]],
) -> list[Chunk]:
    """Convert v2 projected chunk dicts into legacy Chunk dataclass instances.

    Computes fields required by the downstream contract (db_writer, embeddings):
    chunk_no, has_table, table_index. Original v2 metadata is preserved inside
    each Chunk's heading field is used as-is; extended v2 metadata travels
    separately through merge_legal_document_metadata.
    """
    adapted: list[Chunk] = []
    table_counter: int = 0

    for idx, v2c in enumerate(v2_chunks):
        chunk_type: str = str(v2c.get("chunk_type", "unknown"))
        is_table: bool = chunk_type == "table"
        table_index: Optional[int] = None
        if is_table:
            table_counter += 1
            table_index = table_counter

        raw_ref: object = v2c.get("article_ref")
        article_ref: Optional[str] = str(raw_ref) if raw_ref is not None else None

        raw_heading: object = v2c.get("heading")
        heading: str = str(raw_heading) if raw_heading else ""

        page_start: int = int(v2c.get("page_start") or 1)
        page_end: int = int(v2c.get("page_end") or page_start)

        adapted.append(Chunk(
            text=str(v2c.get("text", "")),
            chunk_no=idx + 1,
            chunk_type=chunk_type,
            article_ref=article_ref,
            heading=heading,
            start_page=page_start,
            end_page=page_end,
            has_table=is_table,
            table_index=table_index,
        ))

    _logger.info(
        "layout_v2.adapter.completed",
        chunk_count=len(adapted),
        table_count=table_counter,
    )
    return adapted


def _build_structure_summary(structure: object) -> dict[str, object]:
    """Build a serializable summary of the v2 document structure tree."""
    from pipeline.layout_models import StructuralNode

    counts: dict[str, int] = {}

    def _count(node: StructuralNode) -> None:
        counts[node.node_type] = counts.get(node.node_type, 0) + 1
        for child in node.children:
            _count(child)

    _count(structure.root)  # type: ignore[union-attr]

    meta: dict[str, object] = getattr(structure, "metadata", {}) or {}
    toc: list[object] = getattr(structure, "toc", []) or []
    sections: list[object] = getattr(structure, "sections", []) or []

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
        "toc_entry_count": len(toc),
        "sections_count": len(sections),
    }


def _evaluate_quality_gate(
    quality_report: dict[str, object],
    run_id: str,
    blob_path: str,
) -> tuple[bool, str]:
    """Check if v2 quality is sufficient for indexing.

    Returns (should_index, reason).
    """
    score: float = float(quality_report.get("quality_score", 0.0))
    min_score: float = settings.LAYOUT_V2_MIN_QUALITY_SCORE
    is_prod: bool = settings.ENVIRONMENT == "production"

    if score >= min_score:
        return (True, "quality_sufficient")

    if not is_prod and settings.ALLOW_LOW_QUALITY_INDEX_IN_NON_PROD:
        _logger.warning(
            "layout_v2.quality.low_score_allowed",
            run_id=run_id,
            blob_path=blob_path,
            quality_score=score,
            min_score=min_score,
            environment=settings.ENVIRONMENT,
        )
        return (True, "low_quality_allowed_non_prod")

    _logger.warning(
        "layout_v2.quality.rejected",
        run_id=run_id,
        blob_path=blob_path,
        quality_score=score,
        min_score=min_score,
        environment=settings.ENVIRONMENT,
    )
    return (False, "quality_insufficient")


def _extract_text_from_v2_pages(structure: object) -> str:
    """Extract concatenated text from first 8 pages of a v2 structure for metadata.

    TECH DEBT: This reconstruction is fragile — the v2 tree may have excluded
    visual DOF fragments (headers, footers, editorial notes) that carry the
    legal legend.  Monitor legend quality in shadow mode before promoting v2
    to primary in staging.  A more robust approach would extract the legend
    directly from the raw DocumentLayout pages before classification.
    """
    from pipeline.layout_models import DocumentStructure
    if not isinstance(structure, DocumentStructure):
        return ""

    from pipeline.layout_models import StructuralNode
    texts: list[str] = []

    def _collect(node: StructuralNode) -> None:
        if node.text and node.page_start is not None and node.page_start <= 8:
            texts.append(node.text)
        for child in node.children:
            _collect(child)

    _collect(structure.root)
    return "\n".join(texts)


# ── Legacy pipeline ──────────────────────────────────────────────────────────


def run_pipeline_legacy(payload) -> None:
    """
    Pipeline end-to-end (legacy): download → extract → chunk → embed → write.
    Usa una conexión del pool para todo el flujo.
    Garantiza que index_run quede en estado terminal (completed/failed/skipped).
    """
    run_id = payload.runId
    blob_path = payload.blobPath
    title = payload.documentTitle[:60]
    t0 = time.perf_counter()

    _logger.info(
        "pipeline.start",
        run_id=run_id,
        title=title,
        blob_path=blob_path,
        pipeline_version="legacy",
        shadow_mode=False,
    )

    try:
        with get_db_conn() as conn:
            update_index_run(
                conn, payload.runId, "processing",
                docs_indexed=0, chunks_total=0,
            )

            # 1. Descarga
            t1 = time.perf_counter()
            pdf_bytes = download_pdf_bytes(payload.blobPath)
            _logger.info(
                "step.download.ok",
                run_id=run_id,
                title=title,
                size_kb=round(len(pdf_bytes) / 1024),
                duration_ms=round((time.perf_counter() - t1) * 1000),
            )

            # 2. Extracción (incluye normalización interna)
            t2 = time.perf_counter()
            relax_prose = getattr(payload, "relaxProseTableFilter", None)
            relaxed_visual = getattr(payload, "relaxedVisualFrameDetection", None)
            eff_relax_prose = (
                relax_prose
                if relax_prose is not None
                else settings.RELAX_PROSE_TABLE_FILTER
            )
            eff_relaxed_visual = (
                relaxed_visual
                if relaxed_visual is not None
                else settings.RELAXED_VISUAL_FRAME_DETECTION
            )
            pages, native_toc = extract_pdf(
                pdf_bytes,
                relax_prose_table_filter=relax_prose,
                relaxed_visual_frame_detection=relaxed_visual,
            )
            total_tables = sum(len(p.tables) for p in pages)
            _logger.info(
                "step.extract.ok",
                run_id=run_id,
                title=title,
                pages=len(pages),
                tables=total_tables,
                native_toc_entries=len(native_toc),
                relax_prose_table_filter=eff_relax_prose,
                relaxed_visual_frame_detection=eff_relaxed_visual,
                duration_ms=round((time.perf_counter() - t2) * 1000),
            )

            # 5. Leyenda legal (primeras 8 páginas)
            first_8_text = "\n".join(p.text for p in pages[:8])
            legend = extract_legal_legend(first_8_text)

            # 6. Nombre de ley
            law_name = extract_law_name(payload.documentTitle)
            _logger.info(
                "step.metadata.ok",
                run_id=run_id,
                title=title,
                law_name=law_name,
                legend_fields=len(legend) if legend else 0,
            )

            # 7. Chunking
            t3 = time.perf_counter()
            head_limit = min(3, len(pages))
            text_head_sample = "\n".join(p.text for p in pages[:head_limit])
            chunks = chunk_content(
                pages,
                2000,
                payload.documentTitle,
                text_head_sample,
            )
            chunk_type_counts: dict[str, int] = {}
            for c in chunks:
                chunk_type_counts[c.chunk_type] = chunk_type_counts.get(c.chunk_type, 0) + 1
            _logger.info(
                "step.chunk.ok",
                run_id=run_id,
                title=title,
                total=len(chunks),
                articles=chunk_type_counts.get("article", 0),
                transitorios=chunk_type_counts.get("transitorio", 0),
                tables=chunk_type_counts.get("table", 0),
                duration_ms=round((time.perf_counter() - t3) * 1000),
            )

            t_refine = time.perf_counter()
            raw_heading_refine = getattr(
                payload, "enableLlmGenericHeadingRefine", None
            )
            enable_refinement, classify_force_llm = resolve_llm_heading_refinement_flags(
                raw_heading_refine,
                settings.ENABLE_LLM_GENERIC_HEADING_REFINE,
            )
            refine_all_for_run = settings.LLM_GENERIC_HEADING_REFINE_ALL or (
                raw_heading_refine is True
            )
            heading_refine_changed = refine_generic_chunk_headings(
                chunks,
                payload.documentTitle,
                text_head_sample,
                enable_refinement=enable_refinement,
                classify_force_llm=classify_force_llm,
                refine_all=refine_all_for_run,
                run_id=run_id,
                blob_path=blob_path,
            )
            if enable_refinement:
                _logger.info(
                    "step.heading_refine.ok",
                    run_id=run_id,
                    title=title,
                    blob_path=blob_path,
                    refine_all=refine_all_for_run,
                    headings_changed=heading_refine_changed,
                    duration_ms=round((time.perf_counter() - t_refine) * 1000),
                )

            if not chunks:
                update_index_run(
                    conn,
                    payload.runId,
                    "completed",
                    docs_indexed=1,
                    chunks_total=0,
                )
                _logger.warning(
                    "pipeline.done",
                    run_id=run_id,
                    title=title,
                    chunks_total=0,
                    reason="pdf_sin_contenido_extraible",
                    duration_ms=round((time.perf_counter() - t0) * 1000),
                )
                return

            # 8. Tipo de documento
            headings = [c.heading for c in chunks if c.heading]
            doc_type, doc_type_confidence, doc_type_source = classify_doc_type(
                payload.documentTitle, headings
            )
            _logger.info(
                "step.classify.ok",
                run_id=run_id,
                title=title,
                doc_type=doc_type,
                confidence=doc_type_confidence,
                source=doc_type_source if doc_type_source else "inferred",
            )
            metadata_extra = {}
            if settings.ENABLE_LLM_DOC_TYPE and doc_type_source:
                metadata_extra = {
                    "docTypeSource": "azure-openai",
                    "docTypeConfidence": str(doc_type_confidence),
                }
            metadata_final = dict(legend or {})
            metadata_final.update(metadata_extra)

            # 9. Embeddings con progreso
            def progress_cb(processed: int, total: int) -> None:
                update_index_run_progress(conn, payload.runId, processed, total)
                conn.commit()

            t4 = time.perf_counter()
            embeddings = embed_chunks(chunks, progress_callback=progress_cb)
            _logger.info(
                "step.embed.ok",
                run_id=run_id,
                title=title,
                chunks=len(chunks),
                duration_ms=round((time.perf_counter() - t4) * 1000),
            )

            # 10. Hash y upsert documento
            content_hash = hashlib.sha256(pdf_bytes).hexdigest()
            document_id = upsert_legal_document(
                conn,
                blob_path=payload.blobPath,
                document_title=payload.documentTitle,
                category_id=payload.categoryId,
                publish_date=payload.publishDate,
                blob_container=settings.AZURE_BLOB_CONTAINER,
                content_hash=content_hash,
                doc_type=doc_type,
                law_name=law_name,
                metadata=metadata_final,
            )
            _logger.info(
                "step.upsert.ok",
                run_id=run_id,
                title=title,
                document_id=str(document_id),
                content_hash=content_hash[:12] + "...",
            )

            # 11. Borrar chunks previos
            delete_existing_chunks(conn, document_id)

            # 12. Insertar chunks (hace commit internamente)
            t5 = time.perf_counter()
            insert_chunks_bulk(conn, document_id, chunks, embeddings)
            _logger.info(
                "step.db_write.ok",
                run_id=run_id,
                title=title,
                chunks=len(chunks),
                duration_ms=round((time.perf_counter() - t5) * 1000),
            )

            try:
                node_count = persist_legal_outline(
                    conn,
                    document_id,
                    content_hash,
                    len(pages),
                    chunks,
                    native_toc,
                )
                conn.commit()
                _logger.info(
                    "outline.persisted",
                    run_id=run_id,
                    document_id=str(document_id),
                    node_count=node_count,
                )
            except Exception as outline_exc:
                _logger.error(
                    "outline.persist_failed",
                    run_id=run_id,
                    document_id=str(document_id),
                    error=str(outline_exc),
                )
                merge_legal_document_metadata(
                    conn,
                    document_id,
                    {"outlineError": str(outline_exc)},
                )
                conn.commit()

            # Persist pipeline_version in metadata
            merge_legal_document_metadata(
                conn, document_id, {"pipeline_version": "legacy"},
            )
            conn.commit()

            # 13. Completar run
            update_index_run(
                conn,
                payload.runId,
                "completed",
                docs_indexed=1,
                chunks_total=len(chunks),
            )

            _logger.info(
                "pipeline.done",
                run_id=run_id,
                title=title,
                doc_type=doc_type,
                chunks_total=len(chunks),
                pipeline_version="legacy",
                duration_ms=round((time.perf_counter() - t0) * 1000),
            )

    except (PDFExtractionError, EmbeddingError) as e:
        _logger.error(
            "pipeline.error",
            run_id=run_id,
            title=title,
            stage=_stage_from_exception(e),
            error=str(e),
        )
        _mark_run_failed(payload.runId, str(e))
    except Exception as e:
        _logger.error(
            "pipeline.error",
            run_id=run_id,
            title=title,
            stage="unknown",
            error=str(e),
        )
        _mark_run_failed(payload.runId, str(e))


# ── V2 pipeline ──────────────────────────────────────────────────────────────


def run_pipeline_v2(payload) -> None:
    """
    Pipeline end-to-end (v2): download → v2 extract → quality gate → embed → write.
    Falls back to legacy if v2 fails or quality is insufficient in production.
    """
    run_id = payload.runId
    blob_path = payload.blobPath
    title = payload.documentTitle[:60]
    t0 = time.perf_counter()
    pipeline_version: str = "v2"

    _logger.info(
        "pipeline.start",
        run_id=run_id,
        title=title,
        blob_path=blob_path,
        pipeline_version="v2",
        shadow_mode=False,
    )

    try:
        with get_db_conn() as conn:
            update_index_run(
                conn, payload.runId, "processing",
                docs_indexed=0, chunks_total=0,
            )

            # 1. Download (shared)
            t1 = time.perf_counter()
            pdf_bytes = download_pdf_bytes(payload.blobPath)
            _logger.info(
                "step.download.ok",
                run_id=run_id,
                title=title,
                size_kb=round(len(pdf_bytes) / 1024),
                duration_ms=round((time.perf_counter() - t1) * 1000),
            )

            # 2. V2 extraction with quality gate
            v2_metadata: dict[str, object] = {}
            chunks: list[Chunk] = []
            native_toc: list[dict[str, object]] = []
            pages_count: int = 0
            v2_structure: object = None
            fallback_used: bool = False

            try:
                v2_chunks_raw, v2_structure, quality_report = _run_v2_extraction(
                    pdf_bytes, payload.documentTitle, run_id, blob_path,
                )

                should_index, gate_reason = _evaluate_quality_gate(
                    quality_report, run_id, blob_path,
                )

                quality_score: float = float(quality_report.get("quality_score", 0.0))
                summary_raw: object = quality_report.get("summary")
                quality_severity: str = "unknown"
                if isinstance(summary_raw, dict):
                    sev: object = summary_raw.get("severity")
                    if isinstance(sev, str):
                        quality_severity = sev

                v2_metadata = {
                    "pipeline_version": "v2",
                    "quality_report": quality_report,
                    "quality_score": quality_score,
                    "quality_severity": quality_severity,
                    "quality_gate_reason": gate_reason,
                    "structure_summary": _build_structure_summary(v2_structure),
                }

                if should_index:
                    chunks = _adapt_v2_chunks_to_legacy(v2_chunks_raw)
                    # Extract pages_count from structure metadata
                    struct_meta: dict[str, object] = getattr(v2_structure, "metadata", {}) or {}
                    node_counts: object = struct_meta.get("node_counts", {})
                    pages_count = int(struct_meta.get("total_pages", 0) or 0)
                    if pages_count == 0:
                        pages_count = _estimate_page_count_from_chunks(chunks)
                    # Get native_toc from the layout for outline compatibility
                    native_toc = []
                    pipeline_version = "v2"
                else:
                    # Quality insufficient → fallback to legacy
                    _logger.warning(
                        "layout_v2.quality.fallback",
                        run_id=run_id,
                        blob_path=blob_path,
                        quality_score=quality_score,
                        gate_reason=gate_reason,
                    )
                    fallback_used = True
                    pipeline_version = "legacy_fallback_quality"
                    v2_metadata["pipeline_version"] = pipeline_version

            except Exception as v2_exc:
                _logger.error(
                    "layout_v2.pipeline.failed",
                    run_id=run_id,
                    blob_path=blob_path,
                    error=str(v2_exc),
                    pipeline_version="v2",
                )
                fallback_used = True
                pipeline_version = "legacy_fallback_error"
                v2_metadata = {
                    "pipeline_version": pipeline_version,
                    "v2_error": str(v2_exc),
                }

            # If v2 failed or quality was insufficient, run legacy extraction
            if fallback_used:
                chunks, pages_count, native_toc = _run_legacy_extraction(
                    payload, pdf_bytes, run_id, title,
                )

            if not chunks:
                update_index_run(
                    conn, payload.runId, "completed",
                    docs_indexed=1, chunks_total=0,
                )
                _logger.warning(
                    "pipeline.done",
                    run_id=run_id,
                    title=title,
                    chunks_total=0,
                    reason="pdf_sin_contenido_extraible",
                    pipeline_version=pipeline_version,
                    duration_ms=round((time.perf_counter() - t0) * 1000),
                )
                return

            # 3. Metadata (legend + law_name)
            if v2_structure is not None and not fallback_used:
                first_text = _extract_text_from_v2_pages(v2_structure)
            else:
                first_text = ""
            legend = extract_legal_legend(first_text) if first_text else None
            law_name = extract_law_name(payload.documentTitle)

            # 4. Doc type classification
            headings = [c.heading for c in chunks if c.heading]
            doc_type, doc_type_confidence, doc_type_source = classify_doc_type(
                payload.documentTitle, headings
            )
            metadata_extra: dict[str, object] = {}
            if settings.ENABLE_LLM_DOC_TYPE and doc_type_source:
                metadata_extra = {
                    "docTypeSource": "azure-openai",
                    "docTypeConfidence": str(doc_type_confidence),
                }
            metadata_final: dict[str, object] = dict(legend or {})
            metadata_final.update(metadata_extra)

            # 5. Embeddings
            def progress_cb(processed: int, total: int) -> None:
                update_index_run_progress(conn, payload.runId, processed, total)
                conn.commit()

            t4 = time.perf_counter()
            embeddings = embed_chunks(chunks, progress_callback=progress_cb)
            _logger.info(
                "step.embed.ok",
                run_id=run_id,
                title=title,
                chunks=len(chunks),
                pipeline_version=pipeline_version,
                duration_ms=round((time.perf_counter() - t4) * 1000),
            )

            # 6. Hash + upsert document
            content_hash = hashlib.sha256(pdf_bytes).hexdigest()
            document_id = upsert_legal_document(
                conn,
                blob_path=payload.blobPath,
                document_title=payload.documentTitle,
                category_id=payload.categoryId,
                publish_date=payload.publishDate,
                blob_container=settings.AZURE_BLOB_CONTAINER,
                content_hash=content_hash,
                doc_type=doc_type,
                law_name=law_name,
                metadata=metadata_final,
            )

            # 7. Delete + insert chunks
            delete_existing_chunks(conn, document_id)
            t5 = time.perf_counter()
            insert_chunks_bulk(conn, document_id, chunks, embeddings)
            _logger.info(
                "step.db_write.ok",
                run_id=run_id,
                title=title,
                chunks=len(chunks),
                pipeline_version=pipeline_version,
                duration_ms=round((time.perf_counter() - t5) * 1000),
            )

            # 8. Outline
            # TECH DEBT: v2 outline still goes through legacy persist_legal_outline
            # using adapted chunks, instead of persisting structure.toc directly.
            # Valid for transition; should be replaced with a v2-native outline
            # builder once shadow-mode data confirms structural parity.
            try:
                node_count = persist_legal_outline(
                    conn, document_id, content_hash,
                    pages_count, chunks, native_toc or None,
                )
                conn.commit()
                _logger.info(
                    "outline.persisted",
                    run_id=run_id,
                    document_id=str(document_id),
                    node_count=node_count,
                    pipeline_version=pipeline_version,
                )
            except Exception as outline_exc:
                _logger.error(
                    "outline.persist_failed",
                    run_id=run_id,
                    document_id=str(document_id),
                    error=str(outline_exc),
                )
                merge_legal_document_metadata(
                    conn, document_id, {"outlineError": str(outline_exc)},
                )
                conn.commit()

            # 9. Extended metadata persistence
            merge_legal_document_metadata(conn, document_id, v2_metadata)
            conn.commit()

            # 10. Complete
            update_index_run(
                conn, payload.runId, "completed",
                docs_indexed=1, chunks_total=len(chunks),
            )
            _logger.info(
                "pipeline.done",
                run_id=run_id,
                title=title,
                doc_type=doc_type,
                chunks_total=len(chunks),
                pipeline_version=pipeline_version,
                duration_ms=round((time.perf_counter() - t0) * 1000),
            )

    except (PDFExtractionError, EmbeddingError) as e:
        _logger.error(
            "pipeline.error",
            run_id=run_id,
            title=title,
            stage=_stage_from_exception(e),
            error=str(e),
            pipeline_version=pipeline_version,
        )
        _mark_run_failed(payload.runId, str(e))
    except Exception as e:
        _logger.error(
            "pipeline.error",
            run_id=run_id,
            title=title,
            stage="unknown",
            error=str(e),
            pipeline_version=pipeline_version,
        )
        _mark_run_failed(payload.runId, str(e))


# ── Shadow mode pipeline ─────────────────────────────────────────────────────


def _run_pipeline_with_shadow(payload) -> None:
    """Shadow mode: legacy as primary + v2 comparison (v2 cannot break legacy)."""
    run_id = payload.runId
    blob_path = payload.blobPath
    title = payload.documentTitle[:60]
    t0 = time.perf_counter()

    _logger.info(
        "pipeline.start",
        run_id=run_id,
        title=title,
        blob_path=blob_path,
        pipeline_version="legacy",
        shadow_mode=True,
    )

    try:
        with get_db_conn() as conn:
            update_index_run(
                conn, payload.runId, "processing",
                docs_indexed=0, chunks_total=0,
            )

            # 1. Download
            t1 = time.perf_counter()
            pdf_bytes = download_pdf_bytes(payload.blobPath)
            _logger.info(
                "step.download.ok",
                run_id=run_id,
                title=title,
                size_kb=round(len(pdf_bytes) / 1024),
                duration_ms=round((time.perf_counter() - t1) * 1000),
            )

            # 2. Legacy extraction (primary)
            chunks, pages_count, native_toc = _run_legacy_extraction(
                payload, pdf_bytes, run_id, title,
            )

            if not chunks:
                update_index_run(
                    conn, payload.runId, "completed",
                    docs_indexed=1, chunks_total=0,
                )
                _logger.warning(
                    "pipeline.done",
                    run_id=run_id,
                    title=title,
                    chunks_total=0,
                    reason="pdf_sin_contenido_extraible",
                    pipeline_version="legacy",
                    shadow_mode=True,
                    duration_ms=round((time.perf_counter() - t0) * 1000),
                )
                return

            # 3. V2 shadow extraction (cannot break legacy)
            v2_metadata: dict[str, object] = {}
            shadow_compare_result: dict[str, object] | None = None
            try:
                v2_chunks_raw, v2_structure, quality_report = _run_v2_extraction(
                    pdf_bytes, payload.documentTitle, run_id, blob_path,
                )
                quality_score: float = float(quality_report.get("quality_score", 0.0))
                summary_raw: object = quality_report.get("summary")
                quality_severity: str = "unknown"
                if isinstance(summary_raw, dict):
                    sev: object = summary_raw.get("severity")
                    if isinstance(sev, str):
                        quality_severity = sev

                v2_metadata = {
                    "quality_report": quality_report,
                    "quality_score": quality_score,
                    "quality_severity": quality_severity,
                    "structure_summary": _build_structure_summary(v2_structure),
                }

                from pipeline.shadow_compare_v2 import compare_pipeline_outputs
                shadow_compare_result = compare_pipeline_outputs(
                    legacy_chunks=chunks,
                    v2_chunks=v2_chunks_raw,
                    legacy_metadata=None,
                    v2_metadata={"quality_score": quality_score},
                )
                _logger.info(
                    "layout_v2.shadow.compare",
                    run_id=run_id,
                    blob_path=blob_path,
                    pipeline_version="legacy",
                    shadow_mode=True,
                    legacy_chunk_count=len(chunks),
                    v2_chunk_count=len(v2_chunks_raw),
                    quality_score=quality_score,
                    quality_severity=quality_severity,
                )
            except Exception as v2_exc:
                _logger.error(
                    "layout_v2.pipeline.failed",
                    run_id=run_id,
                    blob_path=blob_path,
                    error=str(v2_exc),
                    shadow_mode=True,
                )
                v2_metadata["v2_error"] = str(v2_exc)

            # 4. Continue with legacy flow: metadata, doc type, embeddings
            first_8_text = _get_first_pages_text(pdf_bytes, payload)
            legend = extract_legal_legend(first_8_text) if first_8_text else None
            law_name = extract_law_name(payload.documentTitle)

            headings = [c.heading for c in chunks if c.heading]
            doc_type, doc_type_confidence, doc_type_source = classify_doc_type(
                payload.documentTitle, headings
            )
            metadata_extra: dict[str, object] = {}
            if settings.ENABLE_LLM_DOC_TYPE and doc_type_source:
                metadata_extra = {
                    "docTypeSource": "azure-openai",
                    "docTypeConfidence": str(doc_type_confidence),
                }
            metadata_final: dict[str, object] = dict(legend or {})
            metadata_final.update(metadata_extra)

            def progress_cb(processed: int, total: int) -> None:
                update_index_run_progress(conn, payload.runId, processed, total)
                conn.commit()

            t4 = time.perf_counter()
            embeddings = embed_chunks(chunks, progress_callback=progress_cb)
            _logger.info(
                "step.embed.ok",
                run_id=run_id,
                title=title,
                chunks=len(chunks),
                pipeline_version="legacy",
                shadow_mode=True,
                duration_ms=round((time.perf_counter() - t4) * 1000),
            )

            # 5. Hash + upsert + write
            content_hash = hashlib.sha256(pdf_bytes).hexdigest()
            document_id = upsert_legal_document(
                conn,
                blob_path=payload.blobPath,
                document_title=payload.documentTitle,
                category_id=payload.categoryId,
                publish_date=payload.publishDate,
                blob_container=settings.AZURE_BLOB_CONTAINER,
                content_hash=content_hash,
                doc_type=doc_type,
                law_name=law_name,
                metadata=metadata_final,
            )
            delete_existing_chunks(conn, document_id)
            t5 = time.perf_counter()
            insert_chunks_bulk(conn, document_id, chunks, embeddings)
            _logger.info(
                "step.db_write.ok",
                run_id=run_id,
                title=title,
                chunks=len(chunks),
                pipeline_version="legacy",
                shadow_mode=True,
                duration_ms=round((time.perf_counter() - t5) * 1000),
            )

            # 6. Outline (legacy)
            try:
                node_count = persist_legal_outline(
                    conn, document_id, content_hash,
                    pages_count, chunks, native_toc,
                )
                conn.commit()
            except Exception as outline_exc:
                _logger.error(
                    "outline.persist_failed",
                    run_id=run_id,
                    document_id=str(document_id),
                    error=str(outline_exc),
                )
                merge_legal_document_metadata(
                    conn, document_id, {"outlineError": str(outline_exc)},
                )
                conn.commit()

            # 7. Persist extended metadata (pipeline_version + v2 data + shadow compare)
            extended_metadata: dict[str, object] = {
                "pipeline_version": "legacy",
                "shadow_mode": True,
            }
            extended_metadata.update(v2_metadata)
            if shadow_compare_result is not None:
                extended_metadata["shadow_compare"] = shadow_compare_result
            merge_legal_document_metadata(conn, document_id, extended_metadata)
            conn.commit()

            # 8. Complete
            update_index_run(
                conn, payload.runId, "completed",
                docs_indexed=1, chunks_total=len(chunks),
            )
            _logger.info(
                "pipeline.done",
                run_id=run_id,
                title=title,
                doc_type=doc_type,
                chunks_total=len(chunks),
                pipeline_version="legacy",
                shadow_mode=True,
                duration_ms=round((time.perf_counter() - t0) * 1000),
            )

    except (PDFExtractionError, EmbeddingError) as e:
        _logger.error(
            "pipeline.error",
            run_id=run_id,
            title=title,
            stage=_stage_from_exception(e),
            error=str(e),
            shadow_mode=True,
        )
        _mark_run_failed(payload.runId, str(e))
    except Exception as e:
        _logger.error(
            "pipeline.error",
            run_id=run_id,
            title=title,
            stage="unknown",
            error=str(e),
            shadow_mode=True,
        )
        _mark_run_failed(payload.runId, str(e))


# ── Shared extraction helpers ────────────────────────────────────────────────


def _run_legacy_extraction(
    payload: object,
    pdf_bytes: bytes,
    run_id: str,
    title: str,
) -> tuple[list[Chunk], int, list[dict[str, object]]]:
    """Run legacy extraction + chunking. Returns (chunks, page_count, native_toc).

    Encapsulates steps 2-7 of the legacy pipeline so they can be reused
    by both v2-fallback and shadow-mode paths without duplicating code.
    """
    blob_path: str = getattr(payload, "blobPath", "")

    t2 = time.perf_counter()
    relax_prose = getattr(payload, "relaxProseTableFilter", None)
    relaxed_visual = getattr(payload, "relaxedVisualFrameDetection", None)
    pages, native_toc = extract_pdf(
        pdf_bytes,
        relax_prose_table_filter=relax_prose,
        relaxed_visual_frame_detection=relaxed_visual,
    )
    _logger.info(
        "step.extract.ok",
        run_id=run_id,
        title=title,
        pages=len(pages),
        tables=sum(len(p.tables) for p in pages),
        duration_ms=round((time.perf_counter() - t2) * 1000),
    )

    t3 = time.perf_counter()
    head_limit = min(3, len(pages))
    text_head_sample = "\n".join(p.text for p in pages[:head_limit])
    chunks: list[Chunk] = chunk_content(
        pages, 2000, getattr(payload, "documentTitle", ""), text_head_sample,
    )

    raw_heading_refine = getattr(payload, "enableLlmGenericHeadingRefine", None)
    enable_refinement, classify_force_llm = resolve_llm_heading_refinement_flags(
        raw_heading_refine, settings.ENABLE_LLM_GENERIC_HEADING_REFINE,
    )
    refine_all_for_run: bool = settings.LLM_GENERIC_HEADING_REFINE_ALL or (
        raw_heading_refine is True
    )
    refine_generic_chunk_headings(
        chunks,
        getattr(payload, "documentTitle", ""),
        text_head_sample,
        enable_refinement=enable_refinement,
        classify_force_llm=classify_force_llm,
        refine_all=refine_all_for_run,
        run_id=run_id,
        blob_path=blob_path,
    )
    _logger.info(
        "step.chunk.ok",
        run_id=run_id,
        title=title,
        total=len(chunks),
        duration_ms=round((time.perf_counter() - t3) * 1000),
    )

    return (chunks, len(pages), native_toc)


def _get_first_pages_text(pdf_bytes: bytes, payload: object) -> str:
    """Re-extract first 8 pages text for legend extraction (lightweight)."""
    try:
        pages, _ = extract_pdf(pdf_bytes)
        return "\n".join(p.text for p in pages[:8])
    except Exception:
        return ""


def _estimate_page_count_from_chunks(chunks: list[Chunk]) -> int:
    """Estimate page count from chunk page ranges when exact count is unavailable."""
    if not chunks:
        return 0
    return max(c.end_page for c in chunks)


# ── Orchestrator ─────────────────────────────────────────────────────────────


def run_pipeline(payload) -> None:
    """
    Pipeline end-to-end: dispatches to legacy, v2, or shadow mode based on
    feature flags ENABLE_LAYOUT_V2 and LAYOUT_V2_SHADOW_MODE.

    Case 1 (default): ENABLE_LAYOUT_V2=False, SHADOW=False → legacy
    Case 2: ENABLE_LAYOUT_V2=True, SHADOW=False → v2 (with quality gate + fallback)
    Case 3: LAYOUT_V2_SHADOW_MODE=True → legacy primary + v2 comparison
    """
    if settings.LAYOUT_V2_SHADOW_MODE:
        _run_pipeline_with_shadow(payload)
    elif settings.ENABLE_LAYOUT_V2:
        run_pipeline_v2(payload)
    else:
        run_pipeline_legacy(payload)


# ── Infrastructure ───────────────────────────────────────────────────────────


def _mark_run_failed(run_id: str, error_message: str) -> None:
    """
    Marca index_run como failed.
    Garantiza estado terminal incluso si falla el pipeline.
    """
    try:
        with get_db_conn() as conn:
            update_index_run(conn, run_id, "failed", error_log=error_message)
    except Exception:
        pass


def _run_pipeline_safe(payload) -> None:
    """
    Ejecuta el pipeline dentro del semáforo.
    Garantiza que index_run SIEMPRE quede en estado terminal.
    """
    with _pipeline_semaphore:
        try:
            run_pipeline(payload)
        except Exception as e:
            _mark_run_failed(payload.runId, str(e))


def submit_pipeline(payload) -> None:
    """
    Encola el procesamiento del PDF.
    No bloqueante. El endpoint responde 202 antes de que termine.
    """
    _executor.submit(_run_pipeline_safe, payload)
