"""
Structural data models for the v2 layout extraction pipeline.
All models are Pydantic-based for strict typing and JSON serialization.
Phases 2-10 of the pipeline consume and produce these types.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractedSpan(BaseModel):
    """A single text run within a line, with per-span style and position data."""

    text: str
    bbox: tuple[float, float, float, float]
    font_size: float | None
    font_name: str | None
    is_bold: bool | None
    is_italic: bool | None
    page_number: int
    block_no: int | None
    line_no: int | None
    span_no: int | None


class LayoutBlock(BaseModel):
    """Content block on a page with bbox, kind label, reading order, and span detail."""

    block_id: str
    page_number: int
    bbox: tuple[float, float, float, float]
    text: str
    # Possible values: text | table | boxed_note | header | footer | unknown
    kind: str
    reading_order: int
    spans: list[ExtractedSpan]
    # Possible values: pymupdf_text | pymupdf_table | inferred | merged
    source: str
    metadata: dict[str, object]


class PageLayout(BaseModel):
    """Full structural layout of a single page."""

    page_number: int
    width: float
    height: float
    blocks: list[LayoutBlock]
    # Candidate table positions from find_tables(); unfiltered at this stage
    raw_tables: list[dict[str, object]]
    # Visual rectangle signals from get_drawings() for editorial boxes/borders
    raw_drawings: list[dict[str, object]]


class DocumentLayout(BaseModel):
    """Complete layout representation of a document, prior to classification."""

    pages: list[PageLayout]
    native_toc: list[dict[str, object]]
    metadata: dict[str, object]


class ClassifiedBlock(BaseModel):
    """A LayoutBlock after heuristic or LLM classification."""

    block_id: str
    page_number: int
    label: str
    confidence: float
    reason: str | None
    llm_used: bool
    normalized_text: str
    metadata: dict[str, object]


class StructuralNode(BaseModel):
    """A node in the legal document tree (article, section, chapter, etc.)."""

    node_id: str
    # Possible values: document | book | title | chapter | section | article
    #                  fraction | inciso | transitory | table | note | paragraph
    node_type: str
    heading: str | None
    text: str | None
    article_ref: str | None
    page_start: int | None
    page_end: int | None
    # Forward reference resolved via model_rebuild() below
    children: list[StructuralNode] = Field(default_factory=list)
    source_block_ids: list[str]
    metadata: dict[str, object]


# Required after the class body when children references the same class
StructuralNode.model_rebuild()


class DocumentStructure(BaseModel):
    """Final hierarchical legal structure derived from classified layout blocks."""

    root: StructuralNode
    toc: list[dict[str, object]]
    sections: list[dict[str, object]]
    quality_report: dict[str, object]
    metadata: dict[str, object]
