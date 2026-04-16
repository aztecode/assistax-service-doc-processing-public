"""
Hybrid block classifier v2: heuristic-first, LLM-only-for-ambiguous (Phase 3).

Orchestrates block_rules_v2 (fast, deterministic) with
block_classifier_llm_v2 (slow, high-quality) to classify every block in
a DocumentLayout into a ClassifiedBlock with a legal label.

Not yet wired into runner.py — standalone for testing and validation.
"""
from __future__ import annotations

import logging
import re

from pipeline.block_classifier_llm_v2 import classify_ambiguous_block
from pipeline.block_rules_v2 import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    VALID_LABELS,
    classify_block_by_rules,
)
from pipeline.layout_models import ClassifiedBlock, DocumentLayout, LayoutBlock

logger = logging.getLogger(__name__)

# Heuristic confidence at or above this threshold skips LLM entirely
_SKIP_LLM_THRESHOLD: float = CONFIDENCE_MEDIUM


def _needs_llm_escalation(
    label: str,
    confidence: float,
    block: LayoutBlock,
) -> bool:
    """Decide whether the heuristic result is uncertain enough for LLM."""
    if confidence >= _SKIP_LLM_THRESHOLD:
        return False

    # Ambiguous top/bottom-zone blocks (header vs title, footer vs body)
    if block.bbox[1] < 100.0 and label not in ("page_header", "page_footer"):
        return True
    if block.bbox[3] > 750.0 and label not in ("page_header", "page_footer"):
        return True

    if label == "unknown":
        return True

    if confidence < CONFIDENCE_LOW:
        return True

    return False


def _normalize_text(text: str) -> str:
    """Light normalization preserving content for downstream phases."""
    normalized: str = text.strip()
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    return normalized


def _classify_single_block(
    block: LayoutBlock,
    prev_blocks: list[LayoutBlock],
    next_blocks: list[LayoutBlock],
    document_metadata: dict[str, object],
) -> ClassifiedBlock:
    """Classify one block: heuristic first, LLM if ambiguous, fallback if LLM fails."""
    label: str
    confidence: float
    reason: str | None
    llm_used: bool = False

    label, confidence, reason = classify_block_by_rules(block)

    if _needs_llm_escalation(label, confidence, block):
        llm_result: dict[str, object] = classify_ambiguous_block(
            block, prev_blocks, next_blocks, document_metadata,
        )

        llm_label: str = str(llm_result.get("label", "unknown"))
        llm_confidence: float = float(llm_result.get("confidence", 0.0))
        llm_reason: str = str(llm_result.get("reason", ""))

        # Accept LLM only if it produced a valid label with decent confidence
        if llm_label in VALID_LABELS and llm_confidence >= 0.70:
            label = llm_label
            confidence = llm_confidence
            reason = f"llm:{llm_reason}"
            llm_used = True
        else:
            # LLM not convincing enough — keep heuristic decision
            reason = f"llm_low_confidence_fallback:{reason}"

    normalized_text: str = _normalize_text(block.text)

    return ClassifiedBlock(
        block_id=block.block_id,
        page_number=block.page_number,
        label=label,
        confidence=confidence,
        reason=reason,
        llm_used=llm_used,
        normalized_text=normalized_text,
        metadata={
            **block.metadata,
            "kind": block.kind,
            "source": block.source,
            "bbox": block.bbox,
            "reading_order": block.reading_order,
        },
    )


def classify_document_layout(
    layout: DocumentLayout,
) -> list[ClassifiedBlock]:
    """Classify all blocks in a DocumentLayout using the hybrid strategy.

    Returns a flat list of ClassifiedBlock ordered by page_number then
    reading_order.  Does not modify the input layout.
    """
    all_blocks: list[LayoutBlock] = []
    for page in layout.pages:
        sorted_blocks: list[LayoutBlock] = sorted(
            page.blocks, key=lambda b: b.reading_order,
        )
        all_blocks.extend(sorted_blocks)

    document_metadata: dict[str, object] = dict(layout.metadata)
    classified: list[ClassifiedBlock] = []

    for idx, block in enumerate(all_blocks):
        prev_blocks: list[LayoutBlock] = all_blocks[max(0, idx - 2) : idx]
        next_blocks: list[LayoutBlock] = all_blocks[idx + 1 : idx + 3]

        result: ClassifiedBlock = _classify_single_block(
            block, prev_blocks, next_blocks, document_metadata,
        )
        classified.append(result)

    # Summary log
    label_counts: dict[str, int] = {}
    llm_count: int = 0
    for cb in classified:
        label_counts[cb.label] = label_counts.get(cb.label, 0) + 1
        if cb.llm_used:
            llm_count += 1

    logger.info(
        "block_classifier_v2.completed total=%d llm_used=%d labels=%s",
        len(classified),
        llm_count,
        label_counts,
    )

    return classified
