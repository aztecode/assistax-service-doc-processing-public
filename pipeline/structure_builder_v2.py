"""
Hierarchical legal-document structure builder (Phase 4).

Converts a flat sequence of ClassifiedBlock (Phase 3 output) into a
DocumentStructure tree that preserves legal hierarchy, relative positions
of tables and notes, transitories, and full traceability to source blocks.

Not yet wired into runner.py — standalone for testing and validation.
"""
from __future__ import annotations

import logging
import re

from pipeline.layout_models import ClassifiedBlock, DocumentStructure, StructuralNode

logger = logging.getLogger(__name__)

# ── Hierarchy definition ────────────────────────────────────────────────────

_HIERARCHY_ORDER: list[str] = [
    "document",
    "book",
    "title",
    "chapter",
    "section",
    "article",
]

_HIERARCHY_RANK: dict[str, int] = {
    level: idx for idx, level in enumerate(_HIERARCHY_ORDER)
}

_LABEL_TO_NODE_TYPE: dict[str, str] = {
    "book_heading": "book",
    "title_heading": "title",
    "chapter_heading": "chapter",
    "section_heading": "section",
    "article_heading": "article",
    "article_body": "paragraph",
    "fraction": "fraction",
    "inciso": "inciso",
    "table": "table",
    "editorial_note": "note",
    "transitory_heading": "transitory",
    "transitory_item": "transitory",
    "annex_heading": "section",
    "annex_body": "paragraph",
}

# Labels that must NOT enter the main body tree
_EXCLUDED_LABELS: frozenset[str] = frozenset({
    "page_header",
    "page_footer",
    "index_block",
})

# Node types that appear in the navigable TOC
_TOC_NODE_TYPES: frozenset[str] = frozenset({
    "book",
    "title",
    "chapter",
    "section",
    "article",
    "transitory",
})

# ── Regex helpers ───────────────────────────────────────────────────────────

_ARTICLE_REF_RE: re.Pattern[str] = re.compile(
    r"Art[ií]culo\s+(\d+[\w]*)",
    re.IGNORECASE,
)

_TRANSITORY_REF_RE: re.Pattern[str] = re.compile(
    r"^\s*(Primero|Segundo|Tercero|Cuarto|Quinto|Sexto|S[eé]ptimo|"
    r"Octavo|Noveno|D[eé]cimo|Und[eé]cimo|Duod[eé]cimo|"
    r"D[eé]cimo\s*(?:primer|segund|tercer|cuart|quint|sext|s[eé]ptim|octav|noven)o?"
    r"|Vig[eé]simo|Trig[eé]simo|Cuadrag[eé]simo|Quincuag[eé]simo"
    r")\s*[.\-–—]",
    re.IGNORECASE,
)

_HEADING_NUMBER_RE: re.Pattern[str] = re.compile(
    r"(?:Libro|T[ií]tulo|Cap[ií]tulo|Secci[oó]n|Anexo)\s+([IVXLCDM\d]+)",
    re.IGNORECASE,
)


# ── Pure helper functions ───────────────────────────────────────────────────

def _extract_article_ref(text: str) -> str | None:
    match: re.Match[str] | None = _ARTICLE_REF_RE.search(text)
    if match:
        return match.group(1)
    return None


def _extract_transitory_ref(text: str) -> str | None:
    match: re.Match[str] | None = _TRANSITORY_REF_RE.match(text.strip())
    if match:
        return match.group(1).lower()
    return None


def _extract_heading_number(text: str) -> str | None:
    match: re.Match[str] | None = _HEADING_NUMBER_RE.search(text)
    if match:
        return match.group(1)
    return None


def _is_navigable_node(node: StructuralNode) -> bool:
    return node.node_type in _TOC_NODE_TYPES


def _new_node(
    node_id: str,
    node_type: str,
    block: ClassifiedBlock,
    heading: str | None,
    article_ref: str | None,
) -> StructuralNode:
    return StructuralNode(
        node_id=node_id,
        node_type=node_type,
        heading=heading,
        text=block.normalized_text,
        article_ref=article_ref,
        page_start=block.page_number,
        page_end=block.page_number,
        children=[],
        source_block_ids=[block.block_id],
        metadata={},
    )


def _append_child(parent: StructuralNode, child: StructuralNode) -> None:
    parent.children.append(child)
    _update_ancestor_pages(parent, child.page_start, child.page_end)


def _update_ancestor_pages(
    node: StructuralNode,
    page_start: int | None,
    page_end: int | None,
) -> None:
    if page_start is not None:
        if node.page_start is None or page_start < node.page_start:
            node.page_start = page_start
    if page_end is not None:
        if node.page_end is None or page_end > node.page_end:
            node.page_end = page_end


def _recompute_page_ranges(node: StructuralNode) -> None:
    """Bottom-up pass to ensure page_start/page_end reflect all descendants."""
    for child in node.children:
        _recompute_page_ranges(child)
        _update_ancestor_pages(node, child.page_start, child.page_end)


def _build_toc_from_tree(root: StructuralNode) -> list[dict[str, object]]:
    toc: list[dict[str, object]] = []
    _collect_toc_entries(root, toc)
    return toc


def _collect_toc_entries(
    node: StructuralNode,
    toc: list[dict[str, object]],
) -> None:
    if node.node_type != "document" and _is_navigable_node(node):
        toc.append({
            "node_id": node.node_id,
            "node_type": node.node_type,
            "heading": node.heading,
            "page_start": node.page_start,
        })
    for child in node.children:
        _collect_toc_entries(child, toc)


def _build_sections_summary(
    root: StructuralNode,
) -> list[dict[str, object]]:
    sections: list[dict[str, object]] = []
    _collect_sections(root, sections)
    return sections


def _collect_sections(
    node: StructuralNode,
    sections: list[dict[str, object]],
) -> None:
    if node.node_type in ("book", "title", "chapter", "section"):
        child_articles: int = sum(
            1 for c in node.children if c.node_type == "article"
        )
        sections.append({
            "node_id": node.node_id,
            "node_type": node.node_type,
            "heading": node.heading,
            "page_start": node.page_start,
            "page_end": node.page_end,
            "article_count": child_articles,
            "child_count": len(node.children),
        })
    for child in node.children:
        _collect_sections(child, sections)


# ── Active context tracker ──────────────────────────────────────────────────

class _BuildContext:
    """Mutable cursor tracking the active position in the document tree."""

    def __init__(self, root: StructuralNode) -> None:
        self.root: StructuralNode = root
        self.current_book: StructuralNode | None = None
        self.current_title: StructuralNode | None = None
        self.current_chapter: StructuralNode | None = None
        self.current_section: StructuralNode | None = None
        self.current_article: StructuralNode | None = None
        self.current_fraction: StructuralNode | None = None
        self.current_transitory_container: StructuralNode | None = None
        self.current_transitory_item: StructuralNode | None = None
        self.in_annex: bool = False

        self._counters: dict[str, int] = {}

    def next_id(self, prefix: str) -> str:
        count: int = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = count
        return f"{prefix}-{count}"

    def reset_below(self, level: str) -> None:
        """Reset all context levels strictly below the given level."""
        rank: int = _HIERARCHY_RANK.get(level, -1)
        if rank < _HIERARCHY_RANK["book"]:
            self.current_book = None
        if rank <= _HIERARCHY_RANK["book"]:
            self.current_title = None
        if rank <= _HIERARCHY_RANK["title"]:
            self.current_chapter = None
        if rank <= _HIERARCHY_RANK["chapter"]:
            self.current_section = None
        if rank <= _HIERARCHY_RANK["section"]:
            self.current_article = None
        if rank <= _HIERARCHY_RANK["article"]:
            self.current_fraction = None

    def nearest_structural_parent(self) -> StructuralNode:
        """Return the deepest active structural node for attaching content."""
        if self.current_article is not None:
            return self.current_article
        if self.current_section is not None:
            return self.current_section
        if self.current_chapter is not None:
            return self.current_chapter
        if self.current_title is not None:
            return self.current_title
        if self.current_book is not None:
            return self.current_book
        return self.root

    def set_structural(self, level: str, node: StructuralNode) -> None:
        if level == "book":
            self.current_book = node
        elif level == "title":
            self.current_title = node
        elif level == "chapter":
            self.current_chapter = node
        elif level == "section":
            self.current_section = node
        elif level == "article":
            self.current_article = node


# ── Block processing dispatch ──────────────────────────────────────────────

def _process_structural_heading(
    block: ClassifiedBlock,
    node_type: str,
    ctx: _BuildContext,
) -> None:
    """Handle book/title/chapter/section headings."""
    ctx.reset_below(node_type)
    heading_number: str | None = _extract_heading_number(block.normalized_text)
    node_id: str
    if heading_number:
        node_id = f"{node_type}-{heading_number}".lower()
    else:
        node_id = ctx.next_id(node_type)

    node: StructuralNode = _new_node(
        node_id=node_id,
        node_type=node_type,
        block=block,
        heading=block.normalized_text.strip(),
        article_ref=None,
    )

    parent: StructuralNode = _parent_for_structural(node_type, ctx)
    _append_child(parent, node)
    ctx.set_structural(node_type, node)
    ctx.in_annex = False


def _parent_for_structural(
    node_type: str,
    ctx: _BuildContext,
) -> StructuralNode:
    """Determine the correct parent for a structural heading based on hierarchy."""
    if node_type == "book":
        return ctx.root
    if node_type == "title":
        return ctx.current_book if ctx.current_book is not None else ctx.root
    if node_type == "chapter":
        if ctx.current_title is not None:
            return ctx.current_title
        if ctx.current_book is not None:
            return ctx.current_book
        return ctx.root
    if node_type == "section":
        if ctx.current_chapter is not None:
            return ctx.current_chapter
        if ctx.current_title is not None:
            return ctx.current_title
        if ctx.current_book is not None:
            return ctx.current_book
        return ctx.root
    return ctx.root


def _process_article_heading(
    block: ClassifiedBlock,
    ctx: _BuildContext,
) -> None:
    ctx.current_article = None
    ctx.current_fraction = None

    article_ref: str | None = _extract_article_ref(block.normalized_text)
    node_id: str
    if article_ref:
        node_id = f"article-{article_ref}".lower()
    else:
        node_id = ctx.next_id("article")

    node: StructuralNode = _new_node(
        node_id=node_id,
        node_type="article",
        block=block,
        heading=block.normalized_text.strip(),
        article_ref=article_ref,
    )

    parent: StructuralNode
    if ctx.current_section is not None:
        parent = ctx.current_section
    elif ctx.current_chapter is not None:
        parent = ctx.current_chapter
    elif ctx.current_title is not None:
        parent = ctx.current_title
    elif ctx.current_book is not None:
        parent = ctx.current_book
    else:
        parent = ctx.root

    _append_child(parent, node)
    ctx.current_article = node


def _process_article_body(
    block: ClassifiedBlock,
    ctx: _BuildContext,
) -> None:
    node_id: str = ctx.next_id("paragraph")
    node: StructuralNode = _new_node(
        node_id=node_id,
        node_type="paragraph",
        block=block,
        heading=None,
        article_ref=None,
    )
    parent: StructuralNode
    if ctx.current_transitory_item is not None:
        parent = ctx.current_transitory_item
    elif ctx.current_transitory_container is not None:
        parent = ctx.current_transitory_container
    else:
        parent = ctx.nearest_structural_parent()
    _append_child(parent, node)


def _process_fraction(
    block: ClassifiedBlock,
    ctx: _BuildContext,
) -> None:
    ctx.current_fraction = None
    node_id: str = ctx.next_id("fraction")
    node: StructuralNode = _new_node(
        node_id=node_id,
        node_type="fraction",
        block=block,
        heading=None,
        article_ref=None,
    )
    parent: StructuralNode
    if ctx.current_article is not None:
        parent = ctx.current_article
    else:
        parent = ctx.nearest_structural_parent()
    _append_child(parent, node)
    ctx.current_fraction = node


def _process_inciso(
    block: ClassifiedBlock,
    ctx: _BuildContext,
) -> None:
    node_id: str = ctx.next_id("inciso")
    node: StructuralNode = _new_node(
        node_id=node_id,
        node_type="inciso",
        block=block,
        heading=None,
        article_ref=None,
    )
    parent: StructuralNode
    if ctx.current_fraction is not None:
        parent = ctx.current_fraction
    elif ctx.current_article is not None:
        parent = ctx.current_article
    else:
        parent = ctx.nearest_structural_parent()
    _append_child(parent, node)


def _process_table(
    block: ClassifiedBlock,
    ctx: _BuildContext,
) -> None:
    node_id: str = ctx.next_id("table")
    node: StructuralNode = _new_node(
        node_id=node_id,
        node_type="table",
        block=block,
        heading=None,
        article_ref=None,
    )

    parent: StructuralNode
    if ctx.current_transitory_item is not None:
        parent = ctx.current_transitory_item
    elif ctx.current_transitory_container is not None:
        parent = ctx.current_transitory_container
    elif ctx.current_article is not None:
        parent = ctx.current_article
    else:
        parent = ctx.nearest_structural_parent()
    _append_child(parent, node)


def _process_editorial_note(
    block: ClassifiedBlock,
    ctx: _BuildContext,
) -> None:
    node_id: str = ctx.next_id("note")
    node: StructuralNode = _new_node(
        node_id=node_id,
        node_type="note",
        block=block,
        heading=None,
        article_ref=None,
    )

    parent: StructuralNode
    if ctx.current_transitory_item is not None:
        parent = ctx.current_transitory_item
    elif ctx.current_article is not None:
        parent = ctx.current_article
    else:
        parent = ctx.nearest_structural_parent()
    _append_child(parent, node)


def _process_transitory_heading(
    block: ClassifiedBlock,
    ctx: _BuildContext,
) -> None:
    # Transitories break out of normal structural context
    ctx.current_article = None
    ctx.current_fraction = None
    ctx.current_transitory_item = None

    node_id: str = ctx.next_id("transitory")
    node: StructuralNode = _new_node(
        node_id=node_id,
        node_type="transitory",
        block=block,
        heading=block.normalized_text.strip(),
        article_ref=None,
    )
    _append_child(ctx.root, node)
    ctx.current_transitory_container = node


def _process_transitory_item(
    block: ClassifiedBlock,
    ctx: _BuildContext,
) -> None:
    ref: str | None = _extract_transitory_ref(block.normalized_text)
    node_id: str
    if ref:
        node_id = f"transitory-{ref}"
    else:
        node_id = ctx.next_id("transitory-item")

    node: StructuralNode = _new_node(
        node_id=node_id,
        node_type="transitory",
        block=block,
        heading=block.normalized_text.strip(),
        article_ref=ref,
    )
    parent: StructuralNode
    if ctx.current_transitory_container is not None:
        parent = ctx.current_transitory_container
    else:
        parent = ctx.root
    _append_child(parent, node)
    ctx.current_transitory_item = node
    ctx.current_article = None
    ctx.current_fraction = None


def _process_annex_heading(
    block: ClassifiedBlock,
    ctx: _BuildContext,
) -> None:
    # Annexes treated as section nodes with is_annex metadata
    ctx.current_article = None
    ctx.current_fraction = None
    ctx.current_transitory_container = None
    ctx.current_transitory_item = None

    heading_number: str | None = _extract_heading_number(block.normalized_text)
    node_id: str
    if heading_number:
        node_id = f"annex-{heading_number}".lower()
    else:
        node_id = ctx.next_id("annex")

    node: StructuralNode = _new_node(
        node_id=node_id,
        node_type="section",
        block=block,
        heading=block.normalized_text.strip(),
        article_ref=None,
    )
    node.metadata["is_annex"] = True
    _append_child(ctx.root, node)
    ctx.in_annex = True
    # Re-use section context so articles inside annex attach correctly
    ctx.current_section = node


def _process_annex_body(
    block: ClassifiedBlock,
    ctx: _BuildContext,
) -> None:
    node_id: str = ctx.next_id("paragraph")
    node: StructuralNode = _new_node(
        node_id=node_id,
        node_type="paragraph",
        block=block,
        heading=None,
        article_ref=None,
    )
    node.metadata["is_annex"] = True
    parent: StructuralNode = ctx.nearest_structural_parent()
    _append_child(parent, node)


def _process_document_title(
    block: ClassifiedBlock,
    ctx: _BuildContext,
) -> None:
    """Enrich root metadata with the document title; optionally set root heading."""
    ctx.root.metadata["document_title"] = block.normalized_text.strip()
    if ctx.root.heading is None:
        ctx.root.heading = block.normalized_text.strip()
    ctx.root.source_block_ids.append(block.block_id)
    _update_ancestor_pages(ctx.root, block.page_number, block.page_number)


# ── Main dispatch ───────────────────────────────────────────────────────────

_STRUCTURAL_HEADING_LABELS: dict[str, str] = {
    "book_heading": "book",
    "title_heading": "title",
    "chapter_heading": "chapter",
    "section_heading": "section",
}


def _process_block(
    block: ClassifiedBlock,
    ctx: _BuildContext,
) -> None:
    label: str = block.label

    if label in _STRUCTURAL_HEADING_LABELS:
        _process_structural_heading(block, _STRUCTURAL_HEADING_LABELS[label], ctx)
        return

    if label == "article_heading":
        _process_article_heading(block, ctx)
        return

    if label == "article_body":
        # If inside transitory context, attach to transitory
        if ctx.current_transitory_item is not None:
            _process_article_body(block, ctx)
        elif ctx.current_transitory_container is not None and ctx.current_transitory_item is None:
            # Loose body after TRANSITORIOS heading but before first item
            _process_article_body(block, ctx)
        else:
            _process_article_body(block, ctx)
        return

    if label == "fraction":
        _process_fraction(block, ctx)
        return

    if label == "inciso":
        _process_inciso(block, ctx)
        return

    if label == "table":
        _process_table(block, ctx)
        return

    if label == "editorial_note":
        _process_editorial_note(block, ctx)
        return

    if label == "transitory_heading":
        _process_transitory_heading(block, ctx)
        return

    if label == "transitory_item":
        _process_transitory_item(block, ctx)
        return

    if label == "annex_heading":
        _process_annex_heading(block, ctx)
        return

    if label == "annex_body":
        _process_annex_body(block, ctx)
        return

    if label == "document_title":
        _process_document_title(block, ctx)
        return

    if label == "unknown":
        _process_article_body(block, ctx)
        return


# ── Document metadata builder ──────────────────────────────────────────────

def _build_document_metadata(
    root: StructuralNode,
    excluded_blocks: list[dict[str, object]],
    has_transitories: bool,
    has_annexes: bool,
    document_metadata: dict[str, object] | None,
) -> dict[str, object]:
    counts: dict[str, int] = {}
    _count_node_types(root, counts)

    meta: dict[str, object] = {}
    if document_metadata is not None:
        meta.update(document_metadata)

    doc_title: object = root.metadata.get("document_title")
    if doc_title is not None:
        meta["document_title"] = doc_title

    meta["node_counts"] = counts
    meta["has_transitories"] = has_transitories
    meta["has_annexes"] = has_annexes
    meta["excluded_blocks"] = excluded_blocks
    return meta


def _count_node_types(
    node: StructuralNode,
    counts: dict[str, int],
) -> None:
    counts[node.node_type] = counts.get(node.node_type, 0) + 1
    for child in node.children:
        _count_node_types(child, counts)


# ── Public API ──────────────────────────────────────────────────────────────

def build_document_structure(
    classified_blocks: list[ClassifiedBlock],
    document_metadata: dict[str, object] | None,
) -> DocumentStructure:
    """Build a hierarchical legal-document tree from classified blocks.

    This is the Phase 4 entry point.  It receives the flat sequence produced
    by Phase 3 and returns a fully navigable DocumentStructure with TOC,
    sections summary, and traceability metadata.
    """
    root: StructuralNode = StructuralNode(
        node_id="doc",
        node_type="document",
        heading=None,
        text=None,
        article_ref=None,
        page_start=None,
        page_end=None,
        children=[],
        source_block_ids=[],
        metadata={},
    )

    ctx: _BuildContext = _BuildContext(root)
    excluded_blocks: list[dict[str, object]] = []
    has_transitories: bool = False
    has_annexes: bool = False

    for block in classified_blocks:
        if block.label in _EXCLUDED_LABELS:
            excluded_blocks.append({
                "block_id": block.block_id,
                "label": block.label,
                "page_number": block.page_number,
                "text_preview": block.normalized_text[:120],
            })
            continue

        if block.label == "transitory_heading" or block.label == "transitory_item":
            has_transitories = True
        if block.label in ("annex_heading", "annex_body"):
            has_annexes = True

        _process_block(block, ctx)

    _recompute_page_ranges(root)

    toc: list[dict[str, object]] = _build_toc_from_tree(root)
    sections: list[dict[str, object]] = _build_sections_summary(root)
    metadata: dict[str, object] = _build_document_metadata(
        root, excluded_blocks, has_transitories, has_annexes, document_metadata,
    )

    logger.info(
        "structure_builder_v2.completed total_blocks=%d excluded=%d toc_entries=%d",
        len(classified_blocks),
        len(excluded_blocks),
        len(toc),
    )

    return DocumentStructure(
        root=root,
        toc=toc,
        sections=sections,
        quality_report={},
        metadata=metadata,
    )
