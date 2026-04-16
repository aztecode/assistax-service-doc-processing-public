#!/usr/bin/env python3
"""
Spike: compare PyMuPDF find_tables vs pymupdf-layout (optional dependency).

LICENSE WARNING: pymupdf-layout is distributed under Polyform Noncommercial (or
commercial license from Artifex). Do not add it to production requirements until
legal/compliance approves.

Install spike deps only:
  pip install -r requirements-layout-spike.txt

Usage:
  .venv/bin/python scripts/spike_pymupdf_layout.py /path/to/file.pdf [max_pages]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__.strip())
        return 2
    pdf_path = Path(sys.argv[1]).resolve()
    max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    if not pdf_path.is_file():
        print(f"File not found: {pdf_path}")
        return 1

    try:
        import fitz  # noqa: WPS433
    except ImportError:
        print("PyMuPDF (fitz) required")
        return 1

    try:
        import pymupdf_layout  # noqa: F401,WPS433
    except ImportError:
        print(
            "pymupdf-layout not installed. Run:\n"
            f"  pip install -r {ROOT / 'requirements-layout-spike.txt'}\n"
            "See script docstring for license warning.",
        )
        return 1

    doc = fitz.open(pdf_path)
    n = min(len(doc), max_pages)
    print(f"File: {pdf_path.name}  pages_scanned={n}  (find_tables count per page)")
    for i in range(n):
        page = doc[i]
        try:
            finder = page.find_tables()
            n_tab = len(finder.tables) if finder and finder.tables else 0
        except Exception as exc:
            n_tab = f"err:{exc}"
        print(f"  page {i + 1}: find_tables -> {n_tab}")
    doc.close()

    print(
        "\nNext step: use pymupdf_layout API per "
        "https://pymupdf.readthedocs.io/en/latest/pymupdf-layout/index.html "
        "to extract structured layout and compare table detection on the same pages.",
    )
    print("This script intentionally does not call layout extraction without a stable API contract.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
