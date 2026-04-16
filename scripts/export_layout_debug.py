#!/usr/bin/env python3
"""
Export detailed debug artifacts for manual inspection of v2 pipeline output.

Generates JSON and/or Markdown reports with layout, classification, structure,
quality, and chunk projection details for a single document.

Usage examples:
  python scripts/export_layout_debug.py --file tmp/pdf_legales/2025/LEYES/ley.pdf --output-dir tmp/debug
  python scripts/export_layout_debug.py --doc-id <uuid> --output-dir tmp/debug --format md
  python scripts/export_layout_debug.py --file doc.pdf --output-dir tmp/debug --format json --include-chunks
  python scripts/export_layout_debug.py --blob-path laws/doc.pdf --output-dir tmp/debug --include-layout --include-quality
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import structlog  # noqa: E402

from scripts._v2_eval_helpers import (  # noqa: E402
    ClassifiedBlock,
    DocumentLayout,
    DocumentStructure,
    LayoutBlock,
    PageLayout,
    StructuralNode,
    build_structure_summary,
    read_local_pdf,
    run_v2_pipeline_eval,
    title_from_path,
)

_logger = structlog.get_logger()


# ── PDF loader ───────────────────────────────────────────────────────────────


def _load_pdf_bytes(args: argparse.Namespace) -> tuple[bytes, str, str]:
    """Load PDF and return (bytes, identifier_label, title)."""
    if args.file is not None:
        pdf_bytes: bytes = read_local_pdf(args.file)
        title: str = args.document_title or title_from_path(args.file)
        return (pdf_bytes, args.file, title)

    if args.blob_path is not None:
        from pipeline.blob_download import download_pdf_bytes
        pdf_bytes = download_pdf_bytes(args.blob_path)
        title = args.document_title or title_from_path(args.blob_path)
        return (pdf_bytes, args.blob_path, title)

    if args.doc_id is not None:
        blob_path: str | None = _resolve_blob_path(args.doc_id)
        if blob_path is None:
            raise ValueError(f"No blob_path found for doc_id={args.doc_id}")
        from pipeline.blob_download import download_pdf_bytes
        pdf_bytes = download_pdf_bytes(blob_path)
        title = args.document_title or _resolve_title(args.doc_id) or title_from_path(blob_path)
        return (pdf_bytes, f"doc_id:{args.doc_id}", title)

    raise ValueError("No input specified. Use --file, --blob-path, or --doc-id.")


def _resolve_blob_path(doc_id: str) -> str | None:
    import psycopg2
    dsn: str | None = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL required for --doc-id")
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT "blobPath" FROM legal_documents WHERE id = %s::uuid', (doc_id,))
            row = cur.fetchone()
            return str(row[0]) if row else None
    finally:
        conn.close()


def _resolve_title(doc_id: str) -> str | None:
    import psycopg2
    dsn: str | None = os.environ.get("DATABASE_URL")
    if not dsn:
        return None
    try:
        conn = psycopg2.connect(dsn)
        with conn.cursor() as cur:
            cur.execute("SELECT title FROM legal_documents WHERE id = %s::uuid", (doc_id,))
            row = cur.fetchone()
            return str(row[0]) if row else None
    except Exception:
        return None


# ── JSON export builders ─────────────────────────────────────────────────────


def _build_layout_json(layout: DocumentLayout) -> list[dict[str, object]]:
    """Serialize layout to a list of per-page dicts."""
    pages_out: list[dict[str, object]] = []
    for page in layout.pages:
        blocks_out: list[dict[str, object]] = []
        for block in page.blocks:
            blocks_out.append({
                "block_id": block.block_id,
                "bbox": list(block.bbox),
                "kind": block.kind,
                "reading_order": block.reading_order,
                "source": block.source,
                "text_preview": block.text[:200] if block.text else "",
                "text_length": len(block.text),
                "span_count": len(block.spans),
                "metadata": block.metadata,
            })
        pages_out.append({
            "page_number": page.page_number,
            "width": page.width,
            "height": page.height,
            "block_count": len(page.blocks),
            "blocks": blocks_out,
            "raw_table_count": len(page.raw_tables),
            "raw_drawing_count": len(page.raw_drawings),
        })
    return pages_out


def _build_classification_json(
    classified: list[ClassifiedBlock],
) -> list[dict[str, object]]:
    """Serialize classified blocks to a list of dicts."""
    return [
        {
            "block_id": cb.block_id,
            "page_number": cb.page_number,
            "label": cb.label,
            "confidence": cb.confidence,
            "reason": cb.reason,
            "llm_used": cb.llm_used,
            "normalized_text_preview": cb.normalized_text[:200] if cb.normalized_text else "",
            "metadata": cb.metadata,
        }
        for cb in classified
    ]


def _build_structure_json(structure: DocumentStructure) -> dict[str, object]:
    """Serialize the document structure tree."""
    def _node_to_dict(node: StructuralNode) -> dict[str, object]:
        return {
            "node_id": node.node_id,
            "node_type": node.node_type,
            "heading": node.heading,
            "text_preview": node.text[:200] if node.text else None,
            "text_length": len(node.text) if node.text else 0,
            "article_ref": node.article_ref,
            "page_start": node.page_start,
            "page_end": node.page_end,
            "source_block_ids": node.source_block_ids,
            "children": [_node_to_dict(c) for c in node.children],
        }

    return {
        "tree": _node_to_dict(structure.root),
        "toc": structure.toc,
        "sections": structure.sections,
        "summary": build_structure_summary(structure),
        "metadata": structure.metadata,
    }


def _build_chunks_json(
    v2_chunks: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Serialize projected chunks with text preview."""
    return [
        {
            "chunk_index": i,
            "chunk_type": chunk.get("chunk_type", ""),
            "heading": chunk.get("heading", ""),
            "article_ref": chunk.get("article_ref"),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "text_preview": str(chunk.get("text", ""))[:300],
            "text_length": len(str(chunk.get("text", ""))),
            "source_block_ids": chunk.get("source_block_ids", []),
            "metadata": {
                k: v for k, v in chunk.items()
                if k not in ("text", "heading", "article_ref", "page_start",
                             "page_end", "chunk_type", "source_block_ids")
            },
        }
        for i, chunk in enumerate(v2_chunks)
    ]


# ── Markdown export builders ─────────────────────────────────────────────────


def _build_markdown(
    label: str,
    title: str,
    pipeline_result: dict[str, object],
    include_layout: bool,
    include_classification: bool,
    include_structure: bool,
    include_quality: bool,
    include_chunks: bool,
) -> str:
    """Build a human-readable Markdown debug report."""
    lines: list[str] = []
    lines.append(f"# Debug Report: {title}")
    lines.append(f"\n**Source:** {label}")
    lines.append(f"**Pipeline duration:** {pipeline_result.get('duration_ms', 0)}ms")
    lines.append("")

    quality_report: dict[str, object] = pipeline_result.get("quality_report", {})  # type: ignore[assignment]
    lines.append(f"**Quality score:** {quality_report.get('quality_score', 'N/A')}")
    summary: object = quality_report.get("summary", {})
    if isinstance(summary, dict):
        lines.append(f"**Severity:** {summary.get('severity', 'N/A')}")
        reasons: object = summary.get("reasons", [])
        if isinstance(reasons, list) and reasons:
            lines.append(f"**Failing checks:** {', '.join(str(r) for r in reasons)}")
    lines.append("")

    layout: DocumentLayout = pipeline_result["layout"]  # type: ignore[assignment]
    classified: list[ClassifiedBlock] = pipeline_result["classified_blocks"]  # type: ignore[assignment]
    structure: DocumentStructure = pipeline_result["structure"]  # type: ignore[assignment]
    v2_chunks: list[dict[str, object]] = pipeline_result["v2_chunks"]  # type: ignore[assignment]

    lines.append(f"**Pages:** {len(layout.pages)}")
    lines.append(f"**Total blocks (raw):** {sum(len(p.blocks) for p in layout.pages)}")
    lines.append(f"**Classified blocks:** {len(classified)}")
    lines.append(f"**Chunks projected:** {len(v2_chunks)}")
    lines.append("")

    if include_layout:
        lines.append("---")
        lines.append("## Layout by Page")
        lines.append("")
        for page in layout.pages:
            lines.append(f"### Page {page.page_number} ({page.width:.0f}×{page.height:.0f})")
            lines.append(f"Blocks: {len(page.blocks)} | Tables: {len(page.raw_tables)} | Drawings: {len(page.raw_drawings)}")
            lines.append("")
            for block in page.blocks:
                bbox_str: str = f"[{block.bbox[0]:.1f}, {block.bbox[1]:.1f}, {block.bbox[2]:.1f}, {block.bbox[3]:.1f}]"
                lines.append(f"- **{block.block_id}** (order={block.reading_order}, kind={block.kind}, source={block.source})")
                lines.append(f"  bbox: {bbox_str}")
                text_preview: str = block.text[:150].replace("\n", " ") if block.text else ""
                lines.append(f"  text: `{text_preview}`")
                lines.append("")

    if include_classification:
        lines.append("---")
        lines.append("## Block Classification")
        lines.append("")
        lines.append("| Block ID | Page | Label | Confidence | LLM | Reason | Text Preview |")
        lines.append("|----------|------|-------|------------|-----|--------|-------------|")
        for cb in classified:
            text_preview = cb.normalized_text[:80].replace("\n", " ").replace("|", "\\|") if cb.normalized_text else ""
            reason: str = (cb.reason or "").replace("|", "\\|")
            lines.append(
                f"| {cb.block_id} | {cb.page_number} | {cb.label} | {cb.confidence:.2f} "
                f"| {'✓' if cb.llm_used else '—'} | {reason[:50]} | {text_preview} |"
            )
        lines.append("")

    if include_structure:
        lines.append("---")
        lines.append("## Structure Tree")
        lines.append("")
        _render_tree_md(lines, structure.root, depth=0)
        lines.append("")

        lines.append("### TOC")
        lines.append("")
        if structure.toc:
            for entry in structure.toc:
                heading_val: str = str(entry.get("heading", ""))
                ntype: str = str(entry.get("node_type", ""))
                lines.append(f"- [{ntype}] {heading_val}")
        else:
            lines.append("*(empty)*")
        lines.append("")

        struct_summary: dict[str, object] = build_structure_summary(structure)
        lines.append("### Structure Summary")
        lines.append("")
        for key, val in struct_summary.items():
            lines.append(f"- **{key}:** {val}")
        lines.append("")

    if include_quality:
        lines.append("---")
        lines.append("## Quality Report")
        lines.append("")
        lines.append(f"**Score:** {quality_report.get('quality_score', 'N/A')}")
        lines.append("")

        checks: object = quality_report.get("checks", {})
        if isinstance(checks, dict):
            lines.append("| Check | Passed | Details |")
            lines.append("|-------|--------|---------|")
            for check_name, check_result in checks.items():
                if not isinstance(check_result, dict):
                    continue
                passed: str = "✓" if check_result.get("passed") else "✗"
                detail_parts: list[str] = []
                for k, v in check_result.items():
                    if k == "passed":
                        continue
                    if isinstance(v, list) and len(v) > 3:
                        detail_parts.append(f"{k}=[{len(v)} items]")
                    else:
                        detail_parts.append(f"{k}={v}")
                details: str = ", ".join(detail_parts[:5]).replace("|", "\\|")
                lines.append(f"| {check_name} | {passed} | {details} |")
        lines.append("")

    if include_chunks:
        lines.append("---")
        lines.append("## Projected Chunks")
        lines.append("")
        for i, chunk in enumerate(v2_chunks):
            heading: str = str(chunk.get("heading", ""))
            article_ref: str = str(chunk.get("article_ref", ""))
            chunk_type: str = str(chunk.get("chunk_type", ""))
            text: str = str(chunk.get("text", ""))
            page_start: str = str(chunk.get("page_start", ""))
            page_end: str = str(chunk.get("page_end", ""))
            lines.append(f"### Chunk {i + 1}: {chunk_type}")
            lines.append(f"- **Heading:** {heading}")
            lines.append(f"- **Article ref:** {article_ref or '—'}")
            lines.append(f"- **Pages:** {page_start}–{page_end}")
            lines.append(f"- **Length:** {len(text)} chars")
            lines.append(f"- **Source blocks:** {chunk.get('source_block_ids', [])}")
            text_preview = text[:300].replace("\n", "\n> ")
            lines.append(f"\n> {text_preview}")
            if len(text) > 300:
                lines.append(f"> *...({len(text) - 300} more chars)*")
            lines.append("")

    return "\n".join(lines)


def _render_tree_md(
    lines: list[str],
    node: StructuralNode,
    depth: int,
) -> None:
    """Recursively render a StructuralNode as an indented Markdown tree."""
    indent: str = "  " * depth
    heading: str = node.heading or ""
    ref: str = f" [{node.article_ref}]" if node.article_ref else ""
    pages: str = ""
    if node.page_start is not None:
        pages = f" (p.{node.page_start}"
        if node.page_end is not None and node.page_end != node.page_start:
            pages += f"–{node.page_end}"
        pages += ")"

    text_hint: str = ""
    if node.text and not node.children:
        text_hint = f" — {len(node.text)} chars"

    lines.append(f"{indent}- **{node.node_type}**{ref}: {heading}{pages}{text_hint}")

    for child in node.children:
        _render_tree_md(lines, child, depth + 1)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export debug artifacts for v2 pipeline inspection",
    )

    input_group = parser.add_argument_group("Input (exactly one required)")
    input_group.add_argument("--doc-id", type=str, help="Document UUID (requires DATABASE_URL)")
    input_group.add_argument("--blob-path", type=str, help="Azure Blob path")
    input_group.add_argument("--file", type=str, help="Local PDF file path")
    input_group.add_argument("--document-title", type=str, help="Override document title")

    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--output-dir", type=str, required=True, help="Directory for debug artifacts")
    output_group.add_argument("--format", type=str, choices=["json", "md", "both"], default="both",
                              help="Output format (default: both)")

    section_group = parser.add_argument_group("Sections to include (default: all)")
    section_group.add_argument("--include-layout", action="store_true", help="Include raw layout per page")
    section_group.add_argument("--include-classification", action="store_true", help="Include block classification")
    section_group.add_argument("--include-structure", action="store_true", help="Include structure tree")
    section_group.add_argument("--include-quality", action="store_true", help="Include quality report")
    section_group.add_argument("--include-chunks", action="store_true", help="Include projected chunks")

    args = parser.parse_args()

    has_any_section: bool = (
        args.include_layout
        or args.include_classification
        or args.include_structure
        or args.include_quality
        or args.include_chunks
    )
    if not has_any_section:
        args.include_layout = True
        args.include_classification = True
        args.include_structure = True
        args.include_quality = True
        args.include_chunks = True

    out_dir: Path = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading PDF...")
    pdf_bytes, label, title = _load_pdf_bytes(args)
    print(f"  Source: {label}")
    print(f"  Title: {title}")
    print(f"  Size: {len(pdf_bytes) / 1024:.1f} KB")

    print("Running v2 pipeline...")
    pipeline_result: dict[str, object] = run_v2_pipeline_eval(pdf_bytes, title)

    quality_report: dict[str, object] = pipeline_result.get("quality_report", {})  # type: ignore[assignment]
    print(f"  Quality score: {quality_report.get('quality_score', 'N/A')}")
    print(f"  Duration: {pipeline_result.get('duration_ms', 0)}ms")

    safe_name: str = "".join(c if c.isalnum() or c in "-_" else "_" for c in title[:60])

    if args.format in ("json", "both"):
        json_path: Path = out_dir / f"{safe_name}_debug.json"
        _write_json_export(json_path, label, title, pipeline_result, args)
        print(f"  JSON written: {json_path}")

    if args.format in ("md", "both"):
        md_path: Path = out_dir / f"{safe_name}_debug.md"
        _write_md_export(md_path, label, title, pipeline_result, args)
        print(f"  Markdown written: {md_path}")

    print("\nDone.")


def _write_json_export(
    path: Path,
    label: str,
    title: str,
    pipeline_result: dict[str, object],
    args: argparse.Namespace,
) -> None:
    """Write a comprehensive JSON debug export."""
    export: dict[str, object] = {
        "document": {
            "source": label,
            "title": title,
            "duration_ms": pipeline_result.get("duration_ms", 0),
        },
    }

    if args.include_layout:
        layout: DocumentLayout = pipeline_result["layout"]  # type: ignore[assignment]
        export["layout"] = _build_layout_json(layout)

    if args.include_classification:
        classified: list[ClassifiedBlock] = pipeline_result["classified_blocks"]  # type: ignore[assignment]
        export["classification"] = _build_classification_json(classified)

    if args.include_structure:
        structure: DocumentStructure = pipeline_result["structure"]  # type: ignore[assignment]
        export["structure"] = _build_structure_json(structure)

    if args.include_quality:
        export["quality_report"] = pipeline_result.get("quality_report", {})

    if args.include_chunks:
        v2_chunks: list[dict[str, object]] = pipeline_result["v2_chunks"]  # type: ignore[assignment]
        export["chunks"] = _build_chunks_json(v2_chunks)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, ensure_ascii=False, default=str)


def _write_md_export(
    path: Path,
    label: str,
    title: str,
    pipeline_result: dict[str, object],
    args: argparse.Namespace,
) -> None:
    """Write a human-readable Markdown debug report."""
    md_content: str = _build_markdown(
        label, title, pipeline_result,
        include_layout=args.include_layout,
        include_classification=args.include_classification,
        include_structure=args.include_structure,
        include_quality=args.include_quality,
        include_chunks=args.include_chunks,
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(md_content)


if __name__ == "__main__":
    main()
