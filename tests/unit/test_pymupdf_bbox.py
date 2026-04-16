"""Tests for pipeline.pymupdf_bbox.normalize_quad."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipeline.pymupdf_bbox import normalize_quad


class TestNormalizeQuad:
    def test_rect_like_object(self) -> None:
        r = SimpleNamespace(x0=1.0, y0=2.0, x1=10.0, y1=20.0)
        assert normalize_quad(r) == (1.0, 2.0, 10.0, 20.0)

    def test_four_tuple(self) -> None:
        assert normalize_quad((0.0, 0.0, 100.5, 200.25)) == (0.0, 0.0, 100.5, 200.25)

    def test_four_list(self) -> None:
        assert normalize_quad([1, 2, 3, 4]) == (1.0, 2.0, 3.0, 4.0)

    def test_none_raises(self) -> None:
        with pytest.raises(ValueError, match="None"):
            normalize_quad(None)

    def test_short_sequence_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 4"):
            normalize_quad((1, 2, 3))
