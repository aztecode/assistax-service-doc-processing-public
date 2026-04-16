"""
Layout normalizer v2: structural pre-classification cleanup.

Detects header/footer repetition, marks possible index zones, and merges
artificially-split paragraph blocks — all without losing traceability.
No legal classification or LLM calls happen here.
"""
import logging
import math
import re
from collections import Counter

from pipeline.layout_extractor_v2 import _compute_reading_order
from pipeline.layout_models import DocumentLayout, LayoutBlock, PageLayout

logger = logging.getLogger(__name__)

# Zone thresholds as fraction of page height
_TOP_ZONE_FRACTION: float = 0.10
_BOTTOM_ZONE_FRACTION: float = 0.10

# Minimum pages a signature must appear in to qualify as repeated header/footer
_MIN_REPETITION_PAGES: int = 2

# Fraction of total pages a signature must appear in (for docs with many pages)
_MIN_REPETITION_RATIO: float = 0.40

# Maximum vertical gap (in points) between blocks to consider them contiguous
_MAX_MERGE_GAP_PTS: float = 18.0

# Font-size tolerance for style compatibility when merging blocks
_FONT_SIZE_TOLERANCE: float = 1.5

# Patterns for index-zone detection
_INDEX_KEYWORDS_RE: re.Pattern[str] = re.compile(
    r"\b(art[ií]culo|cap[ií]tulo|secci[oó]n|t[ií]tulo|transitorio|libro|anexo)\b",
    re.IGNORECASE,
)

# Legal heading patterns that block merging
_LEGAL_HEADING_RE: re.Pattern[str] = re.compile(
    r"^\s*(art[ií]culo\s+\d|cap[ií]tulo\s+[IVXLCDM\d]|secci[oó]n\s+[IVXLCDM\d]"
    r"|t[ií]tulo\s+[IVXLCDM\d]|libro\s+[IVXLCDM\d]|transitorio"
    r"|primero|segundo|tercero|cuarto|quinto|sexto|s[eé]ptimo|octavo|noveno|d[eé]cimo"
    r"|disposicion|anexo\s)",
    re.IGNORECASE,
)

# Date normalization: replace common date-like fragments with placeholder
_DATE_RE: re.Pattern[str] = re.compile(
    r"\b\d{1,2}\s*(?:de\s+)?"
    r"(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
    r"septiembre|octubre|noviembre|diciembre)"
    r"(?:\s*(?:de|del)\s*\d{2,4})?\b",
    re.IGNORECASE,
)

# Isolated page numbers: standalone digits on a line (e.g. "  3  " or "Pág. 12")
_PAGE_NUMBER_RE: re.Pattern[str] = re.compile(
    r"(?:p[aá]g(?:ina)?\.?\s*)?\b\d{1,5}\b",
    re.IGNORECASE,
)

# Folio / numbering patterns
_FOLIO_RE: re.Pattern[str] = re.compile(
    r"\b(?:folio|n[uú]m(?:ero)?|no\.)?\s*\d+\b",
    re.IGNORECASE,
)


def _normalize_signature(text: str) -> str:
    """Produce a canonical signature for repetition comparison.

    Uppercases, collapses whitespace, strips dates, page numbers, and folios
    so that minor per-page variations do not prevent matching.
    """
    sig: str = text.upper()
    sig = _DATE_RE.sub("<<DATE>>", sig)
    sig = _FOLIO_RE.sub("<<NUM>>", sig)
    sig = _PAGE_NUMBER_RE.sub("<<NUM>>", sig)
    sig = re.sub(r"\s+", " ", sig).strip()
    return sig


def _is_top_zone(block: LayoutBlock, page: PageLayout) -> bool:
    """True if the block sits within the top zone of the page."""
    threshold: float = page.height * _TOP_ZONE_FRACTION
    return block.bbox[1] < threshold


def _is_bottom_zone(block: LayoutBlock, page: PageLayout) -> bool:
    """True if the block sits within the bottom zone of the page."""
    threshold: float = page.height * (1.0 - _BOTTOM_ZONE_FRACTION)
    return block.bbox[3] > threshold


def _detect_repeated_headers_and_footers(
    pages: list[PageLayout],
) -> list[PageLayout]:
    """Scan top/bottom zones across pages and mark repeated blocks as header/footer.

    A signature is considered repeated if it appears in at least _MIN_REPETITION_PAGES
    pages AND in at least _MIN_REPETITION_RATIO of total pages.
    """
    total_pages: int = len(pages)
    if total_pages < _MIN_REPETITION_PAGES:
        return pages

    # ceil ensures we never round *down* and accept a signature that barely
    # misses the ratio (e.g. 6 pages × 0.40 = 2.4 → ceil = 3, not int = 2).
    min_appearances: int = max(
        _MIN_REPETITION_PAGES,
        math.ceil(total_pages * _MIN_REPETITION_RATIO),
    )

    top_signatures: Counter[str] = Counter()
    bottom_signatures: Counter[str] = Counter()

    for page in pages:
        seen_top: set[str] = set()
        seen_bottom: set[str] = set()
        for block in page.blocks:
            sig: str = _normalize_signature(block.text)
            if not sig:
                continue
            if _is_top_zone(block, page) and sig not in seen_top:
                top_signatures[sig] += 1
                seen_top.add(sig)
            if _is_bottom_zone(block, page) and sig not in seen_bottom:
                bottom_signatures[sig] += 1
                seen_bottom.add(sig)

    repeated_top: set[str] = {
        sig for sig, count in top_signatures.items() if count >= min_appearances
    }
    repeated_bottom: set[str] = {
        sig for sig, count in bottom_signatures.items() if count >= min_appearances
    }

    updated_pages: list[PageLayout] = []
    for page in pages:
        updated_blocks: list[LayoutBlock] = []
        for block in page.blocks:
            sig = _normalize_signature(block.text)
            new_kind: str = block.kind
            reason: str | None = None
            score: float = 0.0

            if sig in repeated_top and _is_top_zone(block, page):
                new_kind = "header"
                score = top_signatures[sig] / total_pages
                reason = f"repeated_top_signature({top_signatures[sig]}/{total_pages})"
            elif sig in repeated_bottom and _is_bottom_zone(block, page):
                new_kind = "footer"
                score = bottom_signatures[sig] / total_pages
                reason = f"repeated_bottom_signature({bottom_signatures[sig]}/{total_pages})"

            if reason is not None:
                merged_meta: dict[str, object] = {
                    **block.metadata,
                    "normalized_signature": sig,
                    "header_footer_score": score,
                    "header_footer_reason": reason,
                }
                updated_blocks.append(
                    block.model_copy(update={"kind": new_kind, "metadata": merged_meta})
                )
            else:
                updated_blocks.append(block)

        updated_pages.append(page.model_copy(update={"blocks": updated_blocks}))

    return updated_pages


def _mark_possible_index_blocks(pages: list[PageLayout]) -> list[PageLayout]:
    """Mark blocks that look like an initial table-of-contents / index zone.

    Heuristic signals:
    - high density of legal reference keywords relative to text length
    - many short lines (avg < 80 chars per line)
    - enumerative structure without substantial body text
    """
    _SHORT_LINE_THRESHOLD: int = 80
    _MIN_KEYWORD_DENSITY: float = 0.02
    _MIN_SHORT_LINE_RATIO: float = 0.60
    _MIN_KEYWORD_HITS: int = 2

    updated_pages: list[PageLayout] = []
    for page in pages:
        updated_blocks: list[LayoutBlock] = []
        for block in page.blocks:
            if block.kind in ("header", "footer"):
                updated_blocks.append(block)
                continue

            text: str = block.text
            lines: list[str] = text.split("\n")
            non_empty_lines: list[str] = [ln for ln in lines if ln.strip()]

            if not non_empty_lines:
                updated_blocks.append(block)
                continue

            keyword_hits: int = len(_INDEX_KEYWORDS_RE.findall(text))
            text_len: int = max(len(text), 1)
            keyword_density: float = keyword_hits / text_len

            short_lines: int = sum(
                1 for ln in non_empty_lines if len(ln.strip()) < _SHORT_LINE_THRESHOLD
            )
            short_line_ratio: float = short_lines / len(non_empty_lines)

            is_index: bool = (
                keyword_hits >= _MIN_KEYWORD_HITS
                and keyword_density >= _MIN_KEYWORD_DENSITY
                and short_line_ratio >= _MIN_SHORT_LINE_RATIO
            )

            if is_index:
                merged_meta: dict[str, object] = {
                    **block.metadata,
                    "possible_index_zone": True,
                    "index_keyword_hits": keyword_hits,
                    "index_keyword_density": round(keyword_density, 4),
                    "index_short_line_ratio": round(short_line_ratio, 4),
                }
                updated_blocks.append(block.model_copy(update={"metadata": merged_meta}))
            else:
                updated_blocks.append(block)

        updated_pages.append(page.model_copy(update={"blocks": updated_blocks}))

    return updated_pages


def _dominant_font_size(block: LayoutBlock) -> float | None:
    """Return the most common font_size across a block's spans, or None."""
    sizes: list[float] = [
        s.font_size for s in block.spans if s.font_size is not None
    ]
    if not sizes:
        return None
    counter: Counter[float] = Counter(sizes)
    return counter.most_common(1)[0][0]


# Strong sentence-ending punctuation — signals a complete thought
_STRONG_SENTENCE_END_CHARS: frozenset[str] = frozenset(".!?")

# Characters that, when they open the right block, signal continuation rather
# than a new structural unit: quotes, parentheses, dashes.
_CONTINUATION_START_CHARS: frozenset[str] = frozenset(
    "\"'«\u201c\u2018([\u2014\u2013-"
)


def _looks_like_paragraph_continuation(left_text: str, right_text: str) -> bool:
    """Determine if right_text is a natural continuation of left_text.

    Positive signals (return True):
    - right starts with quotes, parentheses, or dashes
    - right starts with a lowercase letter
    - right starts with a digit (legal heading regex already ran upstream)
    - left does NOT end with strong sentence-ending punctuation (.!?)

    Conservative rejection (return False):
    - left ends with strong sentence-ending punctuation AND right starts
      with an uppercase letter that is not a known continuation char
    """
    left_end: str = left_text.rstrip()
    right_start: str = right_text.lstrip()

    if not right_start or not left_end:
        return False

    first_char: str = right_start[0]

    if first_char in _CONTINUATION_START_CHARS:
        return True

    if first_char.islower():
        return True

    if first_char.isdigit():
        return True

    # Left not ending with strong punctuation → mid-sentence split
    if left_end[-1] not in _STRONG_SENTENCE_END_CHARS:
        return True

    return False


def _blocks_overlap_table(
    left: LayoutBlock,
    right: LayoutBlock,
    raw_tables: list[dict[str, object]],
) -> bool:
    """True if a candidate table bbox sits between left and right vertically."""
    gap_top: float = left.bbox[3]
    gap_bottom: float = right.bbox[1]
    for table in raw_tables:
        bbox = table.get("bbox")
        if bbox is None or len(bbox) < 4:  # type: ignore[arg-type]
            continue
        table_top: float = float(bbox[1])  # type: ignore[index]
        table_bottom: float = float(bbox[3])  # type: ignore[index]
        if table_top < gap_bottom and table_bottom > gap_top:
            return True
    return False


def _should_merge_blocks(
    left: LayoutBlock,
    right: LayoutBlock,
    page: PageLayout,
) -> bool:
    """Decide if two consecutive blocks should be merged.

    Conditions (all must hold):
    - Same page
    - Vertically contiguous (gap <= threshold)
    - Compatible base font size
    - Right block starts with lowercase or continuation punctuation
    - Right block does not look like a legal heading
    - Neither block is marked header/footer
    - Neither block is in a possible index zone
    - No candidate table sits between them
    """
    if left.page_number != right.page_number:
        return False

    if left.kind in ("header", "footer") or right.kind in ("header", "footer"):
        return False

    if left.metadata.get("possible_index_zone") or right.metadata.get("possible_index_zone"):
        return False

    vertical_gap: float = right.bbox[1] - left.bbox[3]
    if vertical_gap > _MAX_MERGE_GAP_PTS or vertical_gap < -2.0:
        return False

    left_size: float | None = _dominant_font_size(left)
    right_size: float | None = _dominant_font_size(right)
    if left_size is not None and right_size is not None:
        if abs(left_size - right_size) > _FONT_SIZE_TOLERANCE:
            return False

    right_stripped: str = right.text.lstrip()
    if not right_stripped:
        return False

    if _LEGAL_HEADING_RE.match(right_stripped):
        return False

    if not _looks_like_paragraph_continuation(left.text, right.text):
        return False

    if _blocks_overlap_table(left, right, page.raw_tables):
        return False

    return True


def _merge_blocks(left: LayoutBlock, right: LayoutBlock) -> LayoutBlock:
    """Merge two blocks into one, preserving traceability."""
    merged_text: str = left.text.rstrip() + " " + right.text.lstrip()
    merged_text = re.sub(r"\n{3,}", "\n\n", merged_text)

    x0: float = min(left.bbox[0], right.bbox[0])
    y0: float = min(left.bbox[1], right.bbox[1])
    x1: float = max(left.bbox[2], right.bbox[2])
    y1: float = max(left.bbox[3], right.bbox[3])

    left_sources: list[str] = left.metadata.get("source_block_ids", [left.block_id])  # type: ignore[assignment]
    right_sources: list[str] = right.metadata.get("source_block_ids", [right.block_id])  # type: ignore[assignment]

    merged_meta: dict[str, object] = {
        **left.metadata,
        "source_block_ids": list(left_sources) + list(right_sources),
        "merged": True,
        "merged_from": [left.block_id, right.block_id],
    }

    return LayoutBlock(
        block_id=left.block_id,
        page_number=left.page_number,
        bbox=(x0, y0, x1, y1),
        text=merged_text,
        kind=left.kind,
        reading_order=left.reading_order,
        spans=list(left.spans) + list(right.spans),
        source="merged",
        metadata=merged_meta,
    )


def _merge_page_blocks(page: PageLayout) -> PageLayout:
    """Apply conservative merging to contiguous blocks within a single page."""
    blocks: list[LayoutBlock] = sorted(page.blocks, key=lambda b: b.reading_order)
    if len(blocks) <= 1:
        return page

    merged: list[LayoutBlock] = [blocks[0]]
    for i in range(1, len(blocks)):
        current: LayoutBlock = blocks[i]
        last: LayoutBlock = merged[-1]

        if _should_merge_blocks(last, current, page):
            merged[-1] = _merge_blocks(last, current)
        else:
            merged.append(current)

    return page.model_copy(update={"blocks": merged})


def _reassign_reading_order(page: PageLayout) -> PageLayout:
    """Re-number reading_order sequentially after merges or removals.

    Delegates to _compute_reading_order from layout_extractor_v2 so the
    Y-band quantisation logic stays in one place and both extraction and
    normalisation produce a consistent block order.
    """
    reordered: list[LayoutBlock] = _compute_reading_order(
        page.blocks, page.width, page.height,
    )
    return page.model_copy(update={"blocks": reordered})


def normalize_document_layout(layout: DocumentLayout) -> DocumentLayout:
    """Apply structural normalization to a raw DocumentLayout.

    Pipeline:
    1. Detect and mark repeated headers/footers across pages.
    2. Mark blocks that look like an initial index / table of contents.
    3. Merge artificially-split paragraph blocks (conservative).
    4. Reassign reading_order per page.
    """
    pages: list[PageLayout] = list(layout.pages)

    pages = _detect_repeated_headers_and_footers(pages)
    pages = _mark_possible_index_blocks(pages)

    pages = [_merge_page_blocks(p) for p in pages]
    pages = [_reassign_reading_order(p) for p in pages]

    headers_found: int = sum(
        1 for p in pages for b in p.blocks if b.kind == "header"
    )
    footers_found: int = sum(
        1 for p in pages for b in p.blocks if b.kind == "footer"
    )
    index_zones: int = sum(
        1 for p in pages for b in p.blocks if b.metadata.get("possible_index_zone")
    )
    merged_blocks: int = sum(
        1 for p in pages for b in p.blocks if b.metadata.get("merged")
    )

    logger.info(
        "layout_v2.normalization.completed pages=%d headers=%d footers=%d "
        "index_zones=%d merged_blocks=%d",
        len(pages),
        headers_found,
        footers_found,
        index_zones,
        merged_blocks,
    )

    return layout.model_copy(update={"pages": pages})
