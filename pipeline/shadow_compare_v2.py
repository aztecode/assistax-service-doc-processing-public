"""
Shadow comparison between legacy and v2 pipeline outputs (Phase 7).

Compares chunk lists produced by the legacy and v2 pipelines, generating
metrics useful for observability and validation before promoting v2 to
primary. Metrics are approximate and optimised for operator insight rather
than academic precision.
"""
from __future__ import annotations

import re
from collections import Counter

_ARTICLE_REF_RE: re.Pattern[str] = re.compile(
    r"art[ií]culo\s+\d+",
    re.IGNORECASE,
)

_TABLE_TOKEN_RE: re.Pattern[str] = re.compile(
    r"\[?/?TABLE(?:_\d+)?\]?",
    re.IGNORECASE,
)


def _safe_str(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _chunk_attr(chunk: object, attr: str, default: object = None) -> object:
    """Read attribute from Chunk dataclass or dict transparently."""
    if isinstance(chunk, dict):
        return chunk.get(attr, default)
    return getattr(chunk, attr, default)


def _count_article_refs(chunks: list[object], attr: str) -> int:
    count: int = 0
    for chunk in chunks:
        ref: object = _chunk_attr(chunk, attr)
        if ref is not None and _safe_str(ref).strip():
            count += 1
    return count


def _count_by_type(chunks: list[object], chunk_type: str, type_attr: str) -> int:
    return sum(
        1 for c in chunks
        if _safe_str(_chunk_attr(c, type_attr)) == chunk_type
    )


def _has_visible_table_tokens(chunks: list[object], heading_attr: str) -> bool:
    for chunk in chunks:
        heading: str = _safe_str(_chunk_attr(chunk, heading_attr))
        text: str = _safe_str(_chunk_attr(chunk, "text"))
        combined: str = f"{heading} {text}"
        if _TABLE_TOKEN_RE.search(combined):
            return True
    return False


def _compute_article_ref_coverage(
    chunks: list[object],
    ref_attr: str,
    type_attr: str,
) -> float:
    article_chunks: list[object] = [
        c for c in chunks
        if _safe_str(_chunk_attr(c, type_attr)) in ("article", "transitorio", "transitory")
    ]
    if not article_chunks:
        return 1.0
    with_ref: int = sum(
        1 for c in article_chunks
        if _chunk_attr(c, ref_attr) is not None and _safe_str(_chunk_attr(c, ref_attr)).strip()
    )
    return with_ref / len(article_chunks)


def _heading_quality_score(chunks: list[object], heading_attr: str) -> float:
    """Estimate heading quality: ratio of non-empty, non-garbage headings."""
    total: int = len(chunks)
    if total == 0:
        return 1.0
    good: int = 0
    for chunk in chunks:
        heading: str = _safe_str(_chunk_attr(chunk, heading_attr)).strip()
        if not heading:
            continue
        if _TABLE_TOKEN_RE.search(heading):
            continue
        if len(heading) > 300:
            continue
        good += 1
    return good / total


def compare_pipeline_outputs(
    legacy_chunks: list[object],
    v2_chunks: list[dict[str, object]],
    legacy_metadata: dict[str, object] | None,
    v2_metadata: dict[str, object] | None,
) -> dict[str, object]:
    """Compare legacy and v2 pipeline outputs for shadow-mode observability.

    Returns a JSON-serializable dict with comparative metrics.
    Metrics marked as 'inferred' rely on heuristics; those marked
    'exact' are direct counts.
    """
    legacy_count: int = len(legacy_chunks)
    v2_count: int = len(v2_chunks)

    legacy_article_coverage: float = _compute_article_ref_coverage(
        legacy_chunks, "article_ref", "chunk_type",
    )
    v2_article_coverage: float = _compute_article_ref_coverage(
        v2_chunks, "article_ref", "chunk_type",
    )

    legacy_table_count: int = _count_by_type(legacy_chunks, "table", "chunk_type")
    v2_table_count: int = _count_by_type(v2_chunks, "table", "chunk_type")

    legacy_boxed_note_count: int = _count_by_type(legacy_chunks, "boxed_note", "chunk_type")
    v2_boxed_note_count: int = _count_by_type(v2_chunks, "boxed_note", "chunk_type")

    legacy_has_table_tokens: bool = _has_visible_table_tokens(legacy_chunks, "heading")
    v2_has_table_tokens: bool = _has_visible_table_tokens(v2_chunks, "heading")

    legacy_heading_quality: float = _heading_quality_score(legacy_chunks, "heading")
    v2_heading_quality: float = _heading_quality_score(v2_chunks, "heading")
    heading_quality_delta: float = round(v2_heading_quality - legacy_heading_quality, 4)

    quality_score_v2: float | None = None
    if v2_metadata is not None:
        raw_score: object = v2_metadata.get("quality_score")
        if raw_score is not None:
            quality_score_v2 = float(raw_score)

    # Type distribution comparison
    legacy_types: Counter[str] = Counter(
        _safe_str(_chunk_attr(c, "chunk_type")) for c in legacy_chunks
    )
    v2_types: Counter[str] = Counter(
        _safe_str(_chunk_attr(c, "chunk_type")) for c in v2_chunks
    )

    summary_parts: list[str] = []
    chunk_diff: int = v2_count - legacy_count
    if abs(chunk_diff) > 0:
        summary_parts.append(f"chunk_count_delta={chunk_diff:+d}")
    if heading_quality_delta != 0.0:
        summary_parts.append(f"heading_quality_delta={heading_quality_delta:+.4f}")
    if legacy_has_table_tokens and not v2_has_table_tokens:
        summary_parts.append("v2_removed_table_tokens")
    if not legacy_has_table_tokens and v2_has_table_tokens:
        summary_parts.append("v2_introduced_table_tokens")
    if abs(legacy_table_count - v2_table_count) > 0:
        summary_parts.append(f"table_count_delta={v2_table_count - legacy_table_count:+d}")

    return {
        "legacy_chunk_count": legacy_count,
        "v2_chunk_count": v2_count,
        "legacy_article_ref_coverage": round(legacy_article_coverage, 4),
        "v2_article_ref_coverage": round(v2_article_coverage, 4),
        "legacy_table_chunk_count": legacy_table_count,
        "v2_table_chunk_count": v2_table_count,
        "legacy_boxed_note_count": legacy_boxed_note_count,
        "v2_boxed_note_count": v2_boxed_note_count,
        "legacy_has_visible_table_tokens": legacy_has_table_tokens,
        "v2_has_visible_table_tokens": v2_has_table_tokens,
        "heading_quality_delta": heading_quality_delta,
        "quality_score_v2": quality_score_v2,
        "legacy_type_distribution": dict(legacy_types),
        "v2_type_distribution": dict(v2_types),
        "summary": "; ".join(summary_parts) if summary_parts else "no_significant_differences",
        "_metric_notes": {
            "article_ref_coverage": "exact",
            "table_chunk_count": "exact",
            "boxed_note_count": "exact",
            "has_visible_table_tokens": "inferred (regex-based)",
            "heading_quality_delta": "inferred (heuristic ratio)",
        },
    }
