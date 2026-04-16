"""Tests for _is_likely_prose_not_table relax branch (RELAX_PROSE_TABLE_FILTER)."""

from __future__ import annotations

from pipeline.pdf_extractor import _is_likely_prose_not_table


def _numeric_grid_2x10() -> list[list[str]]:
    """2 rows, 10 columns of short numeric cells (reference grid pattern)."""
    row_a = [str(1500 + i) for i in range(10)]
    row_b = [str(2400 + i) for i in range(10)]
    return [row_a, row_b]


def test_wide_numeric_grid_discarded_by_default() -> None:
    rows = _numeric_grid_2x10()
    assert (
        _is_likely_prose_not_table(rows, relax_prose_table_filter=False) is True
    )


def test_wide_numeric_grid_kept_when_relax_enabled() -> None:
    rows = _numeric_grid_2x10()
    assert (
        _is_likely_prose_not_table(rows, relax_prose_table_filter=True) is False
    )


def test_wide_low_numeric_still_discarded_when_relax_enabled() -> None:
    """Mostly words, few digits: should not pass the numeric escape hatch."""
    rows = [["foo", "bar", "baz", "qux", "nope", "x", "y", "z", "w", "v"]] * 2
    assert (
        _is_likely_prose_not_table(rows, relax_prose_table_filter=True) is True
    )
