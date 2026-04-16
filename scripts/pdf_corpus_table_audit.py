#!/usr/bin/env python3
"""
Audit PDF corpus: PyMuPDF find_tables vs prose filter, visual rects without tables,
and boxed-note arbiter LLM candidates (deterministic ambiguous count).

Uses the same settings as the API: set RELAXED_VISUAL_FRAME_DETECTION / RELAX_PROSE_TABLE_FILTER
in .env so metrics match production tuning. For a corpus of ~500+ laws, point --root at the
unified tree (multiple years/folders), not a single subfolder.

Usage (from assistax-service-doc-processing):
  .venv/bin/python scripts/pdf_corpus_table_audit.py --root /path/to/pdfs
  .venv/bin/python scripts/pdf_corpus_table_audit.py --root /path --export-discards tmp/discards.csv

Sampling discards: filter the CSV by path (e.g. fiscal vs civil) and review `preview` column.

Does not call Azure OpenAI; uses classify_block_deterministic only.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fitz  # noqa: E402

from pipeline.boxed_note_classifier import AmbiguousBlock, classify_block_deterministic  # noqa: E402
from pipeline.pdf_extractor import (  # noqa: E402
    _extract_visual_rects,
    _is_editorial_boxed_note,
    _is_likely_prose_not_table,
    _merge_header_rows,
    _rects_overlap,
)
from pipeline.pymupdf_bbox import normalize_quad  # noqa: E402
from settings import settings  # noqa: E402


def _drawing_path_count(page: fitz.Page) -> int:
    try:
        return len(page.get_drawings())
    except Exception:
        return 0


def _discarded_table_row_stats(rows: List[List[str]]) -> tuple[int, int, float, float, str]:
    non_empty = [r for r in rows if any(c.strip() for c in r)]
    max_cols = max((len(r) for r in rows), default=0)
    num_rows = len(non_empty)
    all_cells = [c.strip() for r in rows for c in r if c.strip()]
    if not all_cells:
        return max_cols, num_rows, 0.0, 0.0, ""
    avg_cell_len = sum(len(c) for c in all_cells) / len(all_cells)
    cells_with_digit = sum(1 for c in all_cells if re.search(r"\d", c))
    pct_numeric = cells_with_digit / len(all_cells)
    preview = " | ".join(
        " ".join(str(c) for c in r[:8]) for r in rows[:2]
    )
    if len(preview) > 600:
        preview = preview[:600] + "..."
    return max_cols, num_rows, pct_numeric, avg_cell_len, preview


@dataclass(frozen=True)
class PageMetrics:
    page_number: int
    drawing_path_count: int
    visual_rect_count: int
    tables_raw: int
    tables_kept: int
    tables_discarded: int
    visual_rect_no_raw_table: bool
    drawings_but_no_raw_table: bool
    arbiter_eligible_blocks: int
    arbiter_ambiguous_blocks: int


@dataclass
class PdfMetrics:
    path: str
    pages: int
    tables_raw: int
    tables_kept: int
    tables_discarded: int
    pages_with_drawings: int
    pages_drawings_no_raw_table: int
    pages_with_visual_rect: int
    pages_visual_rect_no_raw_table: int
    pages_with_any_discard: int
    arbiter_eligible_blocks: int
    arbiter_ambiguous_blocks: int
    seconds: float
    error: str


def export_discarded_tables(
    pdfs: List[Path],
    root_for_rel: Path,
    out_csv: Path,
) -> int:
    """
    Write one CSV row per table candidate dropped by _is_likely_prose_not_table.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    fieldnames = [
        "pdf",
        "page",
        "max_cols",
        "num_rows",
        "pct_numeric",
        "avg_cell_len",
        "preview",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pdf_path in pdfs:
            try:
                rel = str(pdf_path.relative_to(root_for_rel))
            except ValueError:
                rel = str(pdf_path)
            try:
                doc = fitz.open(pdf_path)
            except Exception:
                continue
            for i in range(len(doc)):
                page = doc[i]
                page_no = i + 1
                try:
                    finder = page.find_tables()
                except Exception:
                    continue
                if not finder or not finder.tables:
                    continue
                for tab in finder.tables:
                    try:
                        rows_raw = tab.extract()
                        if rows_raw is None:
                            rows_raw = []
                        rows: List[List[str]] = []
                        for row in rows_raw:
                            if row is None:
                                rows.append([])
                            else:
                                cells = ["" if c is None else str(c).strip() for c in row]
                                rows.append(cells)
                        rows = _merge_header_rows(rows)
                        if not _is_likely_prose_not_table(
                            rows,
                            relax_prose_table_filter=settings.RELAX_PROSE_TABLE_FILTER,
                        ):
                            continue
                        mc, nr, pct, avg_len, preview = _discarded_table_row_stats(rows)
                        writer.writerow({
                            "pdf": rel,
                            "page": page_no,
                            "max_cols": mc,
                            "num_rows": nr,
                            "pct_numeric": round(pct, 4),
                            "avg_cell_len": round(avg_len, 2),
                            "preview": preview,
                        })
                        written += 1
                    except Exception:
                        continue
            doc.close()
    return written


def iter_pdfs(root: Path) -> Iterator[Path]:
    """Yield PDF paths under root, sorted for stable reports."""
    if root.is_file() and root.suffix.lower() == ".pdf":
        yield root
        return
    found = sorted(root.glob("**/*.pdf"))
    for p in found:
        if p.is_file():
            yield p


def analyze_page(page: fitz.Page, page_number: int) -> PageMetrics:
    drawing_path_count = _drawing_path_count(page)
    visual_rects = _extract_visual_rects(
        page,
        relaxed_visual_frame_detection=settings.RELAXED_VISUAL_FRAME_DETECTION,
    )
    vr_count = len(visual_rects)

    tables_raw = 0
    tables_kept = 0
    tables_discarded = 0
    arbiter_eligible = 0
    arbiter_ambiguous = 0

    try:
        finder = page.find_tables()
    except Exception:
        finder = None

    raw_list: List[Any] = []
    if finder and finder.tables:
        raw_list = list(finder.tables)
    tables_raw = len(raw_list)

    for tab in raw_list:
        try:
            rows_raw = tab.extract()
            if rows_raw is None:
                rows_raw = []
            rows: List[List[str]] = []
            for row in rows_raw:
                if row is None:
                    rows.append([])
                else:
                    cells = ["" if c is None else str(c).strip() for c in row]
                    rows.append(cells)
            rows = _merge_header_rows(rows)
        except Exception:
            continue

        if _is_likely_prose_not_table(
            rows,
            relax_prose_table_filter=settings.RELAX_PROSE_TABLE_FILTER,
        ):
            tables_discarded += 1
            continue

        try:
            bbox = normalize_quad(tab.bbox)
        except (ValueError, TypeError, AttributeError):
            tables_discarded += 1
            continue

        tables_kept += 1
        is_boxed = _is_editorial_boxed_note(rows)
        non_empty = [r for r in rows if any(c.strip() for c in r)]
        max_cols = max((len(r) for r in rows), default=0)

        if not is_boxed and vr_count > 0:
            inside_visual = any(
                _rects_overlap(bbox, vr, 0.3) for vr in visual_rects
            )
            if inside_visual and max_cols <= 2:
                arbiter_eligible += 1
                all_text = " ".join(
                    c.strip() for r in non_empty for c in r if c.strip()
                )
                amb = AmbiguousBlock(
                    text=all_text,
                    bbox=bbox,
                    page=page_number,
                    source="table_in_visual_rect",
                    nearby_text_before="",
                    nearby_text_after="",
                    is_inside_visual_box=True,
                    is_table_like=True,
                    rows_count=len(non_empty),
                    cols_count=max_cols,
                )
                if classify_block_deterministic(amb) == "ambiguous":
                    arbiter_ambiguous += 1

    visual_no_raw = vr_count > 0 and tables_raw == 0
    drawings_no_raw = drawing_path_count > 0 and tables_raw == 0

    return PageMetrics(
        page_number=page_number,
        drawing_path_count=drawing_path_count,
        visual_rect_count=vr_count,
        tables_raw=tables_raw,
        tables_kept=tables_kept,
        tables_discarded=tables_discarded,
        visual_rect_no_raw_table=visual_no_raw,
        drawings_but_no_raw_table=drawings_no_raw,
        arbiter_eligible_blocks=arbiter_eligible,
        arbiter_ambiguous_blocks=arbiter_ambiguous,
    )


def analyze_pdf(path: Path, root_for_rel: Path) -> PdfMetrics:
    try:
        rel = str(path.relative_to(root_for_rel))
    except ValueError:
        rel = str(path)
    t0 = time.perf_counter()
    err = ""
    pages_n = 0
    tot_raw = tot_kept = tot_disc = 0
    pages_draw = 0
    pages_draw_no_raw = 0
    pages_vr = 0
    pages_vr_no_raw = 0
    pages_disc = 0
    tot_arb_elig = tot_arb_amb = 0

    try:
        doc = fitz.open(path)
        pages_n = len(doc)
        for i in range(pages_n):
            pm = analyze_page(doc[i], i + 1)
            tot_raw += pm.tables_raw
            tot_kept += pm.tables_kept
            tot_disc += pm.tables_discarded
            if pm.drawing_path_count > 0:
                pages_draw += 1
            if pm.drawings_but_no_raw_table:
                pages_draw_no_raw += 1
            if pm.visual_rect_count > 0:
                pages_vr += 1
            if pm.visual_rect_no_raw_table:
                pages_vr_no_raw += 1
            if pm.tables_discarded > 0:
                pages_disc += 1
            tot_arb_elig += pm.arbiter_eligible_blocks
            tot_arb_amb += pm.arbiter_ambiguous_blocks
        doc.close()
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"

    return PdfMetrics(
        path=rel,
        pages=pages_n,
        tables_raw=tot_raw,
        tables_kept=tot_kept,
        tables_discarded=tot_disc,
        pages_with_drawings=pages_draw,
        pages_drawings_no_raw_table=pages_draw_no_raw,
        pages_with_visual_rect=pages_vr,
        pages_visual_rect_no_raw_table=pages_vr_no_raw,
        pages_with_any_discard=pages_disc,
        arbiter_eligible_blocks=tot_arb_elig,
        arbiter_ambiguous_blocks=tot_arb_amb,
        seconds=round(time.perf_counter() - t0, 3),
        error=err,
    )


def write_csv(rows: List[PdfMetrics], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "pages",
        "tables_raw",
        "tables_kept",
        "tables_discarded",
        "pages_with_drawings",
        "pages_drawings_no_raw_table",
        "pages_with_visual_rect",
        "pages_visual_rect_no_raw_table",
        "pages_with_any_discard",
        "arbiter_eligible_blocks",
        "arbiter_ambiguous_blocks",
        "seconds",
        "error",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: getattr(r, k) for k in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description="PDF corpus table / layout audit.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/Users/rodrigojacome/Documents/Proyectos/rbv/procesar-archivos"),
        help="Root directory or single PDF file",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "tmp" / "pdf_table_audit",
        help="Output directory for report artifacts",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="If > 0, stop after this many PDFs (debug)",
    )
    parser.add_argument(
        "--export-discards",
        type=Path,
        default=None,
        help="If set, write CSV of prose-filter-discarded tables (same PDF list as audit)",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = list(iter_pdfs(root))
    if args.max_files > 0:
        pdfs = pdfs[: args.max_files]

    root_for_rel = root if root.is_dir() else root.parent

    from settings import settings as app_settings

    if args.export_discards is not None:
        n_disc = export_discarded_tables(pdfs, root_for_rel, args.export_discards.resolve())
        print(f"export-discards: {n_disc} rows -> {args.export_discards.resolve()}", flush=True)

    results: List[PdfMetrics] = []
    t_all = time.perf_counter()
    for idx, pdf_path in enumerate(pdfs, start=1):
        m = analyze_pdf(pdf_path, root_for_rel)
        results.append(m)
        if idx % 25 == 0:
            print(f"Processed {idx}/{len(pdfs)} …", flush=True)

    elapsed_all = round(time.perf_counter() - t_all, 2)

    ok = [r for r in results if not r.error]
    failed = [r for r in results if r.error]

    def sum_attr(name: str) -> int:
        return sum(getattr(r, name) for r in ok)

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "pdf_count_requested": len(pdfs),
        "pdf_count_ok": len(ok),
        "pdf_count_failed": len(failed),
        "total_pages": sum_attr("pages"),
        "total_tables_raw": sum_attr("tables_raw"),
        "total_tables_kept": sum_attr("tables_kept"),
        "total_tables_discarded": sum_attr("tables_discarded"),
        "total_pages_with_drawings": sum_attr("pages_with_drawings"),
        "total_pages_drawings_no_raw_table": sum_attr("pages_drawings_no_raw_table"),
        "total_pages_with_visual_rect": sum_attr("pages_with_visual_rect"),
        "total_pages_visual_rect_no_raw_table": sum_attr("pages_visual_rect_no_raw_table"),
        "total_pages_with_any_discard": sum_attr("pages_with_any_discard"),
        "total_arbiter_eligible_blocks": sum_attr("arbiter_eligible_blocks"),
        "total_arbiter_ambiguous_blocks": sum_attr("arbiter_ambiguous_blocks"),
        "wall_seconds": elapsed_all,
        "settings_snapshot": {
            "RELAXED_VISUAL_FRAME_DETECTION": app_settings.RELAXED_VISUAL_FRAME_DETECTION,
            "RELAX_PROSE_TABLE_FILTER": app_settings.RELAX_PROSE_TABLE_FILTER,
        },
        "notes": {
            "visual_rect": "pdf_extractor._extract_visual_rects; respects RELAXED_VISUAL_FRAME_DETECTION.",
            "tables": "PyMuPDF find_tables + _merge_header_rows + _is_likely_prose_not_table (honors RELAX_PROSE_TABLE_FILTER).",
            "llm_arbiter_estimate": "If ENABLE_LLM_BOXED_NOTE_ARBITER=true, LLM calls are up to arbiter_ambiguous per block (deterministic ambiguous only).",
            "drawings_no_table_proxy": "pages_drawings_no_raw_table = get_drawings() non-empty on page but find_tables returned 0 tables on that page (broad LLM/vision candidate ceiling).",
            "visual_rect_strict": "pages_visual_rect_* uses pdf_extractor rules (stroked rect min 100x30); many PDFs use lines without meeting this filter, so counts can be 0 while drawings_no_table is high.",
        },
    }

    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = out_dir / "by_pdf.csv"
    write_csv(results, csv_path)

    fail_path = out_dir / "failures.txt"
    if failed:
        lines = [f"{r.path}\t{r.error}" for r in failed]
        fail_path.write_text("\n".join(lines), encoding="utf-8")
    else:
        fail_path.write_text("", encoding="utf-8")

    # Markdown report
    md_lines = [
        "# Auditoría de tablas y layout (corpus PDF)",
        "",
        f"> PDFs encontrados bajo `--root`: **{summary['pdf_count_requested']}** (cambia la ruta o une carpetas si el corpus objetivo es mayor, p. ej. ~500).",
        "",
        f"- **Generado (UTC):** {summary['generated_at_utc']}",
        f"- **Raíz escaneada:** `{summary['root']}`",
        f"- **PDFs analizados:** {summary['pdf_count_ok']} ok / {summary['pdf_count_failed']} fallidos (total listados: {summary['pdf_count_requested']})",
        f"- **Tiempo total:** {summary['wall_seconds']} s",
        "",
        "## Totales",
        "",
        "| Métrica | Valor |",
        "|--------|-------|",
        f"| Páginas | {summary['total_pages']} |",
        f"| Tablas detectadas (raw `find_tables`) | {summary['total_tables_raw']} |",
        f"| Tablas conservadas (tras filtro prosa) | {summary['total_tables_kept']} |",
        f"| Tablas descartadas (filtro prosa) | {summary['total_tables_discarded']} |",
        f"| Páginas con ≥1 trazo vectorial (`get_drawings`) | {summary['total_pages_with_drawings']} |",
        f"| Páginas con trazos y **0** tablas raw | {summary['total_pages_drawings_no_raw_table']} |",
        f"| Páginas con ≥1 rect. “editorial” (filtro estricto) | {summary['total_pages_with_visual_rect']} |",
        f"| Páginas rect. estricto y **0** tablas raw | {summary['total_pages_visual_rect_no_raw_table']} |",
        f"| Páginas con ≥1 tabla descartada | {summary['total_pages_with_any_discard']} |",
        f"| Bloques elegibles arbiter (tabla en marco, ≤2 cols, no nota editorial) | {summary['total_arbiter_eligible_blocks']} |",
        f"| De ellos, **ambigüos** deterministas → **máx. llamadas LLM** arbiter | {summary['total_arbiter_ambiguous_blocks']} |",
        "",
        "## Dimensionamiento LLM (orientativo)",
        "",
        "1. **Arbiter de notas enmarcadas** (`ENABLE_LLM_BOXED_NOTE_ARBITER`): como máximo **una llamada por bloque ambiguo** listado arriba (solo si el flag está activo). Los bloques estructurales o editoriales claros no llaman al modelo.",
        "2. **Recuperación de tablas / layout**: **páginas con `get_drawings` no vacío y 0 tablas raw** es un *techo muy alto*: casi toda página legal con marco tipográfico tiene trazos vectoriales (líneas de encabezado, bordes decorativos). **No** uses ese número como “llamadas LLM = páginas” sin filtrar; sirve para ver que el problema no es solo `find_tables` sino la señal. Refina con umbral de cantidad de paths, texto cercano (“Fe de erratas”, “TABLA”), o `pymupdf_layout`.",
        "3. **Descartes por prosa**: el total de tablas descartadas indica dónde el pipeline **rechaza** candidatos de `find_tables`; no implica automáticamente LLM, pero sirve para priorizar revisión manual o reglas adicionales.",
        "",
        "## Artefactos",
        "",
        f"- `{json_path.name}` — resumen JSON",
        f"- `{csv_path.name}` — una fila por PDF",
        f"- `{fail_path.name}` — errores de apertura/lectura",
        "",
    ]
    if failed:
        md_lines.extend(
            [
                "## PDFs con error",
                "",
                "```",
                *[f"{r.path}: {r.error}" for r in failed[:50]],
                ("..." if len(failed) > 50 else ""),
                "```",
                "",
            ]
        )

    (out_dir / "REPORT.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote: {out_dir}/REPORT.md, summary.json, by_pdf.csv")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
