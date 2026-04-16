"""
Optional LLM pass to fix generic chunk headings that are body prose, not section titles.
Mutates Chunk.heading in place when the classifier disagrees with the heuristic heading.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import structlog

from pipeline.decreto_heading import (
    _looks_like_sentence_continuation_heading,
    heading_for_generic_chunk,
    is_decreto_context,
)
from pipeline.generic_heading_classifier import classify_generic_heading_is_section_title
from pipeline.metadata_extractor import extract_law_name
from settings import settings

if TYPE_CHECKING:
    from pipeline.legal_chunker import Chunk

logger = structlog.get_logger()

_TEXT_PREFIX_LIMIT: int = 600
_CHUNK_SLICE_FOR_HEADING: int = 4000

_RE_CASOS_SE_START = re.compile(r"^casos,\s*se\s+", re.IGNORECASE)
_RE_BODY_DE_LEY_START = re.compile(
    r"^(de esta ley|de la presente ley|de este código|de este codigo)\b",
    re.IGNORECASE,
)


def resolve_llm_heading_refinement_flags(
    payload_value: bool | None,
    env_llm_heading_enabled: bool,
) -> tuple[bool, bool]:
    """
    Maps per-request enableLlmGenericHeadingRefine to (enable_refinement, classify_force_llm).
    When the payload forces refinement on but env is off, classify_force_llm is True so the
    classifier runs despite ENABLE_LLM_GENERIC_HEADING_REFINE=false.
    """
    if payload_value is False:
        return False, False
    if payload_value is True:
        return True, not env_llm_heading_enabled
    return env_llm_heading_enabled, False


def heading_looks_suspicious_for_llm(heading: str) -> bool:
    """
    True when the stored heading matches high-risk patterns (narrow gate before LLM).
    """
    s = heading.strip()
    if not s:
        return False
    if _looks_like_sentence_continuation_heading(s):
        return True
    if _RE_CASOS_SE_START.match(s):
        return True
    if _RE_BODY_DE_LEY_START.match(s):
        return True
    return False


def replacement_heading_for_misassigned_generic(
    chunk_text: str,
    document_title: str,
    text_head_sample: str,
) -> str:
    """
    Deterministic fallback when the LLM says the heading is not a section title.
    Prefer law name from document title; else re-run heading_for_generic_chunk on text
    after the first line.
    """
    law = extract_law_name(document_title)
    if law:
        return law[:500]

    is_dec = is_decreto_context(document_title, text_head_sample)
    stripped = chunk_text.strip()
    parts = stripped.split("\n", 1)
    if len(parts) == 2 and parts[1].strip():
        h = heading_for_generic_chunk(
            parts[1][:_CHUNK_SLICE_FOR_HEADING],
            document_title,
            is_dec,
        )
        if h:
            return h[:500]

    h2 = heading_for_generic_chunk(
        stripped[:_CHUNK_SLICE_FOR_HEADING],
        document_title,
        is_dec,
    )
    return (h2 or "")[:500]


def compute_refined_heading_for_generic_row(
    heading: str,
    chunk_text: str,
    document_title: str,
    text_head_sample: str,
    *,
    refine_all: bool,
    force_llm: bool,
) -> str | None:
    """
    If the heading should be replaced, returns the new heading; otherwise None.
    Used by backfill scripts (force_llm=True to run without ENABLE_LLM_* in settings).
    """
    current = (heading or "").strip()
    if not current:
        return None
    if not refine_all and not heading_looks_suspicious_for_llm(current):
        return None

    prefix = (chunk_text or "")[:_TEXT_PREFIX_LIMIT]
    verdict, _invoked = classify_generic_heading_is_section_title(
        current,
        prefix,
        document_title,
        force=force_llm,
    )
    if verdict is not False:
        return None

    new_heading = replacement_heading_for_misassigned_generic(
        chunk_text,
        document_title,
        text_head_sample,
    )
    if new_heading and new_heading != current:
        return new_heading
    if not new_heading and document_title.strip():
        fallback = document_title.strip()[:500]
        if fallback != current:
            return fallback
    return None


def refine_generic_chunk_headings(
    chunks: list[Chunk],
    document_title: str,
    text_head_sample: str,
    *,
    enable_refinement: bool,
    classify_force_llm: bool,
    refine_all: bool,
    run_id: str,
    blob_path: str,
) -> int:
    """
    For generic chunks without articleRef, optionally call the LLM to detect prose headings.

    refine_all: when True, every eligible generic chunk is sent to the classifier; when False,
    only chunks whose heading matches heading_looks_suspicious_for_llm.

    Returns the number of chunks whose heading was replaced.
    """
    if not enable_refinement:
        logger.info(
            "heading_refinement.skipped",
            reason="enable_refinement_false",
            run_id=run_id,
            blob_path=blob_path,
        )
        return 0

    changed = 0
    evaluated = 0
    skipped_not_generic = 0
    skipped_has_article_ref = 0
    skipped_empty_heading = 0
    skipped_not_suspicious = 0
    llm_invocations = 0
    llm_verdict_section_title = 0
    llm_verdict_body = 0
    llm_verdict_none = 0

    for c in chunks:
        if c.chunk_type != "generic":
            skipped_not_generic += 1
            continue
        if c.article_ref is not None and str(c.article_ref).strip():
            skipped_has_article_ref += 1
            continue
        current = (c.heading or "").strip()
        if not current:
            skipped_empty_heading += 1
            continue
        if not refine_all and not heading_looks_suspicious_for_llm(current):
            skipped_not_suspicious += 1
            continue

        evaluated += 1
        prefix = (c.text or "")[:_TEXT_PREFIX_LIMIT]
        verdict, llm_called = classify_generic_heading_is_section_title(
            current,
            prefix,
            document_title,
            force=classify_force_llm,
            run_id=run_id,
            chunk_no=c.chunk_no,
            blob_path=blob_path,
        )
        if llm_called:
            llm_invocations += 1

        ctx_chunk = {
            "run_id": run_id,
            "blob_path": blob_path,
            "chunk_no": c.chunk_no,
            "heading_preview": current[:80],
        }

        if verdict is True:
            llm_verdict_section_title += 1
            logger.debug(
                "heading_refinement.keep_heading",
                reason="llm_section_title",
                **ctx_chunk,
            )
            continue

        if verdict is None:
            if llm_called:
                llm_verdict_none += 1
                logger.debug(
                    "heading_refinement.keep_heading",
                    reason="llm_failed_or_invalid_response",
                    **ctx_chunk,
                )
            else:
                logger.warning(
                    "heading_refinement.classifier_no_llm_unexpected",
                    message="eligible chunk but classifier did not invoke LLM",
                    **ctx_chunk,
                )
            continue

        llm_verdict_body += 1

        new_heading = replacement_heading_for_misassigned_generic(
            c.text,
            document_title,
            text_head_sample,
        )
        if new_heading and new_heading != current:
            c.heading = new_heading
            changed += 1
            logger.debug(
                "heading_refinement.heading_replaced",
                **ctx_chunk,
                new_heading_preview=new_heading[:80],
            )
        elif not new_heading and document_title.strip():
            c.heading = document_title.strip()[:500]
            if c.heading != current:
                changed += 1
                logger.debug(
                    "heading_refinement.heading_replaced",
                    **ctx_chunk,
                    new_heading_preview=c.heading[:80],
                )

    if evaluated == 0:
        logger.warning(
            "heading_refinement.no_eligible_chunks",
            run_id=run_id,
            blob_path=blob_path,
            refine_all=refine_all,
            total_chunks=len(chunks),
            skipped_not_generic=skipped_not_generic,
            skipped_has_article_ref=skipped_has_article_ref,
            skipped_empty_heading=skipped_empty_heading,
            skipped_not_suspicious=skipped_not_suspicious,
        )

    logger.info(
        "heading_refinement.done",
        run_id=run_id,
        blob_path=blob_path,
        evaluated=evaluated,
        changed=changed,
        refine_all=refine_all,
        llm_invocations=llm_invocations,
        llm_verdict_section_title=llm_verdict_section_title,
        llm_verdict_body=llm_verdict_body,
        llm_verdict_none=llm_verdict_none,
        skipped_not_generic=skipped_not_generic,
        skipped_has_article_ref=skipped_has_article_ref,
        skipped_empty_heading=skipped_empty_heading,
        skipped_not_suspicious=skipped_not_suspicious,
    )
    return changed
