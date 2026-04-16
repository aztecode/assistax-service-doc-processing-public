"""
Chunk projector for DocumentStructure (Phase 6).

Projects the validated legal-document tree (Phase 4/5 output) into a flat
list of chunks compatible with the existing embeddings/search pipeline.

Each chunk preserves legal semantics, traceability, and structural context
while staying compatible with the legacy Chunk schema used by db_writer
and embeddings modules.

Not yet wired into runner.py — standalone for testing and validation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from pipeline.layout_models import DocumentStructure, StructuralNode

logger = logging.getLogger(__name__)

# Node types that must never be projected as chunks
_EXCLUDED_NODE_TYPES: frozenset[str] = frozenset({
    "page_header",
    "page_footer",
    "index_block",
})

# Node types that produce their own standalone chunk
_ATOMIC_TYPES: frozenset[str] = frozenset({
    "table",
    "note",
})

# Hierarchy levels that carry navigational context but are not chunks themselves
# unless they contain direct text worth projecting
_STRUCTURAL_CONTAINERS: frozenset[str] = frozenset({
    "document",
    "book",
    "title",
    "chapter",
    "section",
})


@dataclass(frozen=True)
class _HierarchyContext:
    """Immutable snapshot of the current position in the document tree."""

    document_title: str | None
    book_heading: str | None
    title_heading: str | None
    chapter_heading: str | None
    section_heading: str | None
    is_transitory: bool
    is_annex: bool
    quality_score: float | None
    quality_severity: str | None
    parent_node_id: str | None
    parent_article_ref: str | None

    def with_updates(
        self,
        document_title: str | None = None,
        book_heading: str | None = None,
        title_heading: str | None = None,
        chapter_heading: str | None = None,
        section_heading: str | None = None,
        is_transitory: bool | None = None,
        is_annex: bool | None = None,
        parent_node_id: str | None = None,
        parent_article_ref: str | None = None,
    ) -> _HierarchyContext:
        return _HierarchyContext(
            document_title=document_title if document_title is not None else self.document_title,
            book_heading=book_heading if book_heading is not None else self.book_heading,
            title_heading=title_heading if title_heading is not None else self.title_heading,
            chapter_heading=chapter_heading if chapter_heading is not None else self.chapter_heading,
            section_heading=section_heading if section_heading is not None else self.section_heading,
            is_transitory=is_transitory if is_transitory is not None else self.is_transitory,
            is_annex=is_annex if is_annex is not None else self.is_annex,
            quality_score=self.quality_score,
            quality_severity=self.quality_severity,
            parent_node_id=parent_node_id if parent_node_id is not None else self.parent_node_id,
            parent_article_ref=parent_article_ref if parent_article_ref is not None else self.parent_article_ref,
        )


# ── Chunk dict builder ──────────────────────────────────────────────────────


def _build_chunk_metadata(
    node: StructuralNode,
    ctx: _HierarchyContext,
) -> dict[str, object]:
    meta: dict[str, object] = {
        "node_id": node.node_id,
        "node_type": node.node_type,
    }
    if ctx.parent_node_id is not None:
        meta["parent_node_id"] = ctx.parent_node_id
    if ctx.document_title is not None:
        meta["document_title"] = ctx.document_title
    if ctx.book_heading is not None:
        meta["book_heading"] = ctx.book_heading
    if ctx.title_heading is not None:
        meta["title_heading"] = ctx.title_heading
    if ctx.chapter_heading is not None:
        meta["chapter_heading"] = ctx.chapter_heading
    if ctx.section_heading is not None:
        meta["section_heading"] = ctx.section_heading
    if ctx.is_transitory:
        meta["is_transitory"] = True
    if ctx.is_annex:
        meta["is_annex"] = True
    if ctx.quality_score is not None:
        meta["quality_score"] = ctx.quality_score
    if ctx.quality_severity is not None:
        meta["quality_severity"] = ctx.quality_severity
    return meta


def _make_chunk(
    chunk_type: str,
    heading: str | None,
    text: str,
    article_ref: str | None,
    page_start: int | None,
    page_end: int | None,
    source_block_ids: list[str],
    metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "chunk_type": chunk_type,
        "heading": heading,
        "text": text,
        "article_ref": article_ref,
        "page_start": page_start,
        "page_end": page_end,
        "source_block_ids": list(source_block_ids),
        "metadata": metadata,
    }


# ── Text aggregation ────────────────────────────────────────────────────────


def _node_own_text(node: StructuralNode) -> str:
    return (node.text or "").strip()


def _aggregate_text_from_children(
    node: StructuralNode,
    include_types: frozenset[str],
) -> str:
    """Collect text from direct children matching include_types."""
    parts: list[str] = []
    for child in node.children:
        if child.node_type in include_types:
            child_text: str = _node_own_text(child)
            if child_text:
                parts.append(child_text)
    return "\n".join(parts)


def _collect_all_source_block_ids(node: StructuralNode) -> list[str]:
    ids: list[str] = list(node.source_block_ids)
    for child in node.children:
        ids.extend(_collect_all_source_block_ids(child))
    return ids


def _resolve_page_range(
    node: StructuralNode,
    children: list[StructuralNode],
) -> tuple[int | None, int | None]:
    """Compute a merged page range from a node and optional children."""
    pages: list[int] = []
    if node.page_start is not None:
        pages.append(node.page_start)
    if node.page_end is not None:
        pages.append(node.page_end)
    for child in children:
        if child.page_start is not None:
            pages.append(child.page_start)
        if child.page_end is not None:
            pages.append(child.page_end)
    if not pages:
        return (None, None)
    return (min(pages), max(pages))


def _resolve_heading(node: StructuralNode) -> str | None:
    if node.heading:
        return node.heading
    return None


# ── Node-type specific projectors ───────────────────────────────────────────


def _project_article_node(
    node: StructuralNode,
    ctx: _HierarchyContext,
) -> list[dict[str, object]]:
    """Project an article node.

    Strategy:
    - Article produces a main chunk with its own text + paragraph children text.
    - Fractions produce separate chunks (they carry independent legal content).
    - Incisos are folded into their parent fraction chunk.
    - Tables and notes within the article are projected as atomic chunks.
    """
    chunks: list[dict[str, object]] = []

    article_ctx: _HierarchyContext = ctx.with_updates(
        parent_node_id=node.node_id,
        parent_article_ref=node.article_ref,
    )

    # Gather the article's own text + paragraph children
    text_parts: list[str] = []
    own_text: str = _node_own_text(node)
    if own_text:
        text_parts.append(own_text)

    paragraph_text: str = _aggregate_text_from_children(
        node, frozenset({"paragraph"}),
    )
    if paragraph_text:
        text_parts.append(paragraph_text)

    # Inline incisos that are direct children of the article (no fraction parent)
    inline_inciso_parts: list[str] = []
    inline_inciso_block_ids: list[str] = []
    inline_inciso_children: list[StructuralNode] = []
    for child in node.children:
        if child.node_type == "inciso":
            inciso_text: str = _node_own_text(child)
            if inciso_text:
                inline_inciso_parts.append(inciso_text)
            inline_inciso_block_ids.extend(child.source_block_ids)
            inline_inciso_children.append(child)

    if inline_inciso_parts:
        text_parts.append("\n".join(inline_inciso_parts))

    article_text: str = "\n".join(text_parts)

    # Collect source_block_ids from the article itself + paragraphs + inline incisos
    source_ids: list[str] = list(node.source_block_ids)
    for child in node.children:
        if child.node_type == "paragraph":
            source_ids.extend(child.source_block_ids)
    source_ids.extend(inline_inciso_block_ids)

    page_children: list[StructuralNode] = [
        c for c in node.children
        if c.node_type in ("paragraph", "inciso")
    ]
    page_start, page_end = _resolve_page_range(node, page_children)

    if article_text:
        chunks.append(_make_chunk(
            chunk_type="article",
            heading=_resolve_heading(node),
            text=article_text,
            article_ref=node.article_ref,
            page_start=page_start,
            page_end=page_end,
            source_block_ids=source_ids,
            metadata=_build_chunk_metadata(node, article_ctx),
        ))

    # Project fractions as separate chunks
    for child in node.children:
        if child.node_type == "fraction":
            chunks.extend(_project_fraction_node(child, article_ctx))
        elif child.node_type in _ATOMIC_TYPES:
            chunks.extend(_project_atomic_node(child, article_ctx))

    return chunks


def _project_fraction_node(
    node: StructuralNode,
    ctx: _HierarchyContext,
) -> list[dict[str, object]]:
    """Project a fraction node, folding in its incisos."""
    chunks: list[dict[str, object]] = []

    text_parts: list[str] = []
    own_text: str = _node_own_text(node)
    if own_text:
        text_parts.append(own_text)

    source_ids: list[str] = list(node.source_block_ids)
    page_children: list[StructuralNode] = []

    for child in node.children:
        if child.node_type == "inciso":
            child_text: str = _node_own_text(child)
            if child_text:
                text_parts.append(child_text)
            source_ids.extend(child.source_block_ids)
            page_children.append(child)
        elif child.node_type == "paragraph":
            child_text = _node_own_text(child)
            if child_text:
                text_parts.append(child_text)
            source_ids.extend(child.source_block_ids)
            page_children.append(child)
        elif child.node_type in _ATOMIC_TYPES:
            chunks.extend(_project_atomic_node(child, ctx))

    fraction_text: str = "\n".join(text_parts)
    page_start, page_end = _resolve_page_range(node, page_children)

    if fraction_text:
        chunks.insert(0, _make_chunk(
            chunk_type="fraction",
            heading=_resolve_heading(node),
            text=fraction_text,
            article_ref=ctx.parent_article_ref,
            page_start=page_start,
            page_end=page_end,
            source_block_ids=source_ids,
            metadata=_build_chunk_metadata(node, ctx),
        ))

    return chunks


def _project_transitory_node(
    node: StructuralNode,
    ctx: _HierarchyContext,
) -> list[dict[str, object]]:
    """Project a transitory node (container or item).

    Container nodes (e.g. 'TRANSITORIOS') only produce a chunk if they
    have meaningful text. Individual items always produce a chunk.
    """
    chunks: list[dict[str, object]] = []
    trans_ctx: _HierarchyContext = ctx.with_updates(
        is_transitory=True,
        parent_node_id=node.node_id,
    )

    is_container: bool = _is_transitory_container(node)

    if is_container:
        # Container: recurse into children; skip chunk for the container itself
        for child in node.children:
            if child.node_type == "transitory":
                chunks.extend(_project_transitory_node(child, trans_ctx))
            elif child.node_type in _ATOMIC_TYPES:
                chunks.extend(_project_atomic_node(child, trans_ctx))
            elif child.node_type == "paragraph":
                # Loose paragraphs inside transitory container
                child_text: str = _node_own_text(child)
                if child_text:
                    chunks.append(_make_chunk(
                        chunk_type="transitory",
                        heading=_resolve_heading(child),
                        text=child_text,
                        article_ref=None,
                        page_start=child.page_start,
                        page_end=child.page_end,
                        source_block_ids=list(child.source_block_ids),
                        metadata=_build_chunk_metadata(child, trans_ctx),
                    ))
        return chunks

    # Individual transitory item
    text_parts: list[str] = []
    own_text: str = _node_own_text(node)
    if own_text:
        text_parts.append(own_text)

    source_ids: list[str] = list(node.source_block_ids)
    page_children: list[StructuralNode] = []

    for child in node.children:
        if child.node_type == "paragraph":
            child_text = _node_own_text(child)
            if child_text:
                text_parts.append(child_text)
            source_ids.extend(child.source_block_ids)
            page_children.append(child)
        elif child.node_type in _ATOMIC_TYPES:
            chunks.extend(_project_atomic_node(child, trans_ctx))
        elif child.node_type == "fraction":
            chunks.extend(_project_fraction_node(child, trans_ctx))

    trans_text: str = "\n".join(text_parts)
    page_start, page_end = _resolve_page_range(node, page_children)

    article_ref: str | None = node.article_ref

    if trans_text:
        chunks.insert(0, _make_chunk(
            chunk_type="transitory",
            heading=_resolve_heading(node),
            text=trans_text,
            article_ref=article_ref,
            page_start=page_start,
            page_end=page_end,
            source_block_ids=source_ids,
            metadata=_build_chunk_metadata(node, trans_ctx),
        ))

    return chunks


def _is_transitory_container(node: StructuralNode) -> bool:
    """A transitory container has transitory children (it's a heading-level node)."""
    return any(c.node_type == "transitory" for c in node.children)


def _project_atomic_node(
    node: StructuralNode,
    ctx: _HierarchyContext,
) -> list[dict[str, object]]:
    """Project a table or note as an independent atomic chunk."""
    chunk_type: str
    if node.node_type == "table":
        chunk_type = "table"
    elif node.node_type == "note":
        chunk_type = "boxed_note"
    else:
        chunk_type = node.node_type

    text: str = _node_own_text(node)

    return [_make_chunk(
        chunk_type=chunk_type,
        heading=_resolve_heading(node),
        text=text,
        article_ref=ctx.parent_article_ref,
        page_start=node.page_start,
        page_end=node.page_end,
        source_block_ids=list(node.source_block_ids),
        metadata=_build_chunk_metadata(node, ctx),
    )]


def _project_section_node(
    node: StructuralNode,
    ctx: _HierarchyContext,
) -> list[dict[str, object]]:
    """Project an annex section or a regular section with direct content."""
    chunks: list[dict[str, object]] = []
    is_annex: bool = bool(node.metadata.get("is_annex", False))

    section_ctx: _HierarchyContext = ctx.with_updates(
        section_heading=node.heading,
        is_annex=is_annex or ctx.is_annex,
        parent_node_id=node.node_id,
    )

    # Sections themselves produce a chunk only if they have their own text
    own_text: str = _node_own_text(node)
    if own_text and is_annex:
        chunks.append(_make_chunk(
            chunk_type="annex",
            heading=_resolve_heading(node),
            text=own_text,
            article_ref=node.article_ref,
            page_start=node.page_start,
            page_end=node.page_end,
            source_block_ids=list(node.source_block_ids),
            metadata=_build_chunk_metadata(node, section_ctx),
        ))

    # Recurse into children
    for child in node.children:
        chunks.extend(_project_node(child, section_ctx))

    return chunks


# ── Tree traversal dispatcher ───────────────────────────────────────────────


def _update_context_for_structural_node(
    node: StructuralNode,
    ctx: _HierarchyContext,
) -> _HierarchyContext:
    """Update hierarchy context when entering a structural container."""
    if node.node_type == "book":
        return ctx.with_updates(
            book_heading=node.heading,
            parent_node_id=node.node_id,
        )
    if node.node_type == "title":
        return ctx.with_updates(
            title_heading=node.heading,
            parent_node_id=node.node_id,
        )
    if node.node_type == "chapter":
        return ctx.with_updates(
            chapter_heading=node.heading,
            parent_node_id=node.node_id,
        )
    if node.node_type == "section":
        return ctx.with_updates(
            section_heading=node.heading,
            is_annex=bool(node.metadata.get("is_annex", False)) or ctx.is_annex,
            parent_node_id=node.node_id,
        )
    if node.node_type == "document":
        doc_title: str | None = None
        raw_title: object = node.metadata.get("document_title")
        if isinstance(raw_title, str):
            doc_title = raw_title
        elif node.heading:
            doc_title = node.heading
        return ctx.with_updates(
            document_title=doc_title,
            parent_node_id=node.node_id,
        )
    return ctx


def _project_node(
    node: StructuralNode,
    ctx: _HierarchyContext,
) -> list[dict[str, object]]:
    """Dispatch projection for a single node based on its type."""
    if node.node_type in _EXCLUDED_NODE_TYPES:
        return []

    if node.node_type == "article":
        return _project_article_node(node, ctx)

    if node.node_type == "transitory":
        return _project_transitory_node(node, ctx)

    if node.node_type in _ATOMIC_TYPES:
        return _project_atomic_node(node, ctx)

    if node.node_type == "section":
        return _project_section_node(node, ctx)

    if node.node_type in _STRUCTURAL_CONTAINERS:
        updated_ctx: _HierarchyContext = _update_context_for_structural_node(node, ctx)
        chunks: list[dict[str, object]] = []
        for child in node.children:
            chunks.extend(_project_node(child, updated_ctx))
        return chunks

    # Paragraph or unknown at root level — preserve content
    if node.node_type == "paragraph":
        own_text: str = _node_own_text(node)
        if own_text:
            chunk_type: str = "annex" if ctx.is_annex else "paragraph"
            return [_make_chunk(
                chunk_type=chunk_type,
                heading=_resolve_heading(node),
                text=own_text,
                article_ref=node.article_ref,
                page_start=node.page_start,
                page_end=node.page_end,
                source_block_ids=list(node.source_block_ids),
                metadata=_build_chunk_metadata(node, ctx),
            )]
        return []

    # Fraction or inciso at unexpected positions — still project
    if node.node_type in ("fraction", "inciso"):
        own_text = _node_own_text(node)
        if own_text:
            return [_make_chunk(
                chunk_type=node.node_type,
                heading=_resolve_heading(node),
                text=own_text,
                article_ref=ctx.parent_article_ref,
                page_start=node.page_start,
                page_end=node.page_end,
                source_block_ids=list(node.source_block_ids),
                metadata=_build_chunk_metadata(node, ctx),
            )]
        return []

    # Fallback: unknown node types with text
    own_text = _node_own_text(node)
    if own_text:
        return [_make_chunk(
            chunk_type=node.node_type,
            heading=_resolve_heading(node),
            text=own_text,
            article_ref=node.article_ref,
            page_start=node.page_start,
            page_end=node.page_end,
            source_block_ids=list(node.source_block_ids),
            metadata=_build_chunk_metadata(node, ctx),
        )]
    return []


# ── Quality report extraction ───────────────────────────────────────────────


def _extract_quality_info(
    quality_report: dict[str, object],
) -> tuple[float | None, str | None]:
    raw_score: object = quality_report.get("quality_score")
    score: float | None = float(raw_score) if raw_score is not None else None

    severity: str | None = None
    summary: object = quality_report.get("summary")
    if isinstance(summary, dict):
        raw_severity: object = summary.get("severity")
        if isinstance(raw_severity, str):
            severity = raw_severity

    return (score, severity)


# ── Public API ──────────────────────────────────────────────────────────────


def project_structure_to_chunks(
    structure: DocumentStructure,
) -> list[dict[str, object]]:
    """Project a DocumentStructure tree into a flat list of chunks.

    This is the Phase 6 entry point. It receives the validated
    DocumentStructure from Phases 4+5 and produces chunks compatible
    with the legacy embeddings/search pipeline.

    Chunks are returned in document order and include full traceability
    metadata (source_block_ids, page ranges, hierarchy context).
    """
    quality_score: float | None
    quality_severity: str | None
    quality_score, quality_severity = _extract_quality_info(
        structure.quality_report,
    )

    root_ctx: _HierarchyContext = _HierarchyContext(
        document_title=None,
        book_heading=None,
        title_heading=None,
        chapter_heading=None,
        section_heading=None,
        is_transitory=False,
        is_annex=False,
        quality_score=quality_score,
        quality_severity=quality_severity,
        parent_node_id=None,
        parent_article_ref=None,
    )

    chunks: list[dict[str, object]] = _project_node(structure.root, root_ctx)

    logger.info(
        "chunk_projector_v2.completed total_chunks=%d",
        len(chunks),
    )

    return chunks
