"""Unit tests for heading_refinement (mocked LLM)."""
from __future__ import annotations

from unittest.mock import patch

from pipeline.heading_refinement import (
    compute_refined_heading_for_generic_row,
    heading_looks_suspicious_for_llm,
    refine_generic_chunk_headings,
    replacement_heading_for_misassigned_generic,
    resolve_llm_heading_refinement_flags,
)
from pipeline.legal_chunker import Chunk


def test_heading_looks_suspicious_casos_se() -> None:
    assert heading_looks_suspicious_for_llm(
        "casos, se considera adecuado otorgar un estímulo"
    )


def test_heading_looks_suspicious_considera() -> None:
    assert heading_looks_suspicious_for_llm("considera procedente la medida")


def test_heading_looks_suspicious_otorga() -> None:
    assert heading_looks_suspicious_for_llm(
        "otorga el subsidio para el empleo, publicado en el Diario Oficial."
    )


def test_heading_looks_suspicious_fundamento() -> None:
    assert heading_looks_suspicious_for_llm(
        "fundamento en los artículos 31 de la Ley Orgánica de la Administración Pública Federal."
    )


def test_heading_not_suspicious_rubric() -> None:
    assert not heading_looks_suspicious_for_llm("DECRETO por el que se reforman disposiciones")


def test_resolve_llm_heading_refinement_flags_false() -> None:
    assert resolve_llm_heading_refinement_flags(False, True) == (False, False)
    assert resolve_llm_heading_refinement_flags(False, False) == (False, False)


def test_resolve_llm_heading_refinement_flags_true() -> None:
    assert resolve_llm_heading_refinement_flags(True, False) == (True, True)
    assert resolve_llm_heading_refinement_flags(True, True) == (True, False)


def test_resolve_llm_heading_refinement_flags_none() -> None:
    assert resolve_llm_heading_refinement_flags(None, True) == (True, False)
    assert resolve_llm_heading_refinement_flags(None, False) == (False, False)


def test_replacement_uses_law_name_from_title() -> None:
    h = replacement_heading_for_misassigned_generic(
        "casos, se considera…\nMás texto",
        "Ley Federal del Impuesto sobre la Renta (LISR)",
        "",
    )
    assert "Impuesto" in h or "Ley Federal" in h


def test_refine_changes_heading_when_llm_says_not_title() -> None:
    chunks = [
        Chunk(
            text="casos, se considera adecuado otorgar estímulo fiscal.",
            chunk_no=1,
            chunk_type="generic",
            article_ref=None,
            heading="casos, se considera adecuado otorgar estímulo fiscal.",
            start_page=1,
            end_page=1,
            has_table=False,
            table_index=None,
        )
    ]
    with (
        patch("pipeline.heading_refinement.settings") as s,
        patch(
            "pipeline.heading_refinement.classify_generic_heading_is_section_title",
            return_value=(False, True),
        ),
    ):
        s.ENABLE_LLM_GENERIC_HEADING_REFINE = True
        s.LLM_GENERIC_HEADING_REFINE_ALL = False

        n = refine_generic_chunk_headings(
            chunks,
            "Decreto estímulos derechos servicios migratorios DOF 30062025",
            "DECRETO por el que",
            enable_refinement=True,
            classify_force_llm=False,
            refine_all=False,
            run_id="run-test-1",
            blob_path="path/doc.pdf",
        )

    assert n == 1
    assert chunks[0].heading != "casos, se considera adecuado otorgar estímulo fiscal."


def test_refine_skips_when_llm_says_title() -> None:
    original = "Disposiciones generales"
    chunks = [
        Chunk(
            text="Disposiciones generales. Para efectos de la Ley…",
            chunk_no=1,
            chunk_type="generic",
            article_ref=None,
            heading=original,
            start_page=1,
            end_page=1,
            has_table=False,
            table_index=None,
        )
    ]
    with (
        patch("pipeline.heading_refinement.settings") as s,
        patch(
            "pipeline.heading_refinement.classify_generic_heading_is_section_title",
            return_value=(True, True),
        ),
    ):
        s.ENABLE_LLM_GENERIC_HEADING_REFINE = True
        s.LLM_GENERIC_HEADING_REFINE_ALL = True

        n = refine_generic_chunk_headings(
            chunks,
            "Ley X",
            "",
            enable_refinement=True,
            classify_force_llm=False,
            refine_all=True,
            run_id="run-test-2",
            blob_path="path/ley.pdf",
        )

    assert n == 0
    assert chunks[0].heading == original


def test_refine_non_suspicious_heading_when_refine_all_true_calls_llm() -> None:
    """Matches per-request enableLlmGenericHeadingRefine=true (full generic sweep)."""
    original = "de servicios proporcionados en territorio nacional"
    chunks = [
        Chunk(
            text=original + " más texto del cuerpo del fragmento.",
            chunk_no=1,
            chunk_type="generic",
            article_ref=None,
            heading=original,
            start_page=1,
            end_page=1,
            has_table=False,
            table_index=None,
        )
    ]
    with (
        patch("pipeline.heading_refinement.settings") as s,
        patch(
            "pipeline.heading_refinement.classify_generic_heading_is_section_title",
            return_value=(False, True),
        ) as mock_llm,
    ):
        s.ENABLE_LLM_GENERIC_HEADING_REFINE = True
        s.LLM_GENERIC_HEADING_REFINE_ALL = False

        n = refine_generic_chunk_headings(
            chunks,
            "Ley Federal del Impuesto sobre la Renta (LISR)",
            "",
            enable_refinement=True,
            classify_force_llm=False,
            refine_all=True,
            run_id="run-test-3",
            blob_path="path/dec.pdf",
        )

    mock_llm.assert_called_once()
    assert n == 1
    assert chunks[0].heading != original


def test_compute_refined_heading_with_force_llm() -> None:
    with patch(
        "pipeline.heading_refinement.classify_generic_heading_is_section_title",
        return_value=(False, True),
    ):
        out = compute_refined_heading_for_generic_row(
            "casos, se considera adecuado",
            "casos, se considera adecuado otorgar estímulo.",
            "Decreto estímulos DOF 30062025",
            "DECRETO",
            refine_all=False,
            force_llm=True,
        )
    assert out is not None
    assert out != "casos, se considera adecuado"


def test_refine_skips_non_suspicious_when_not_refine_all() -> None:
    chunks = [
        Chunk(
            text="Texto",
            chunk_no=1,
            chunk_type="generic",
            article_ref=None,
            heading="Rúbrica editorial sin patrón de riesgo larga",
            start_page=1,
            end_page=1,
            has_table=False,
            table_index=None,
        )
    ]
    with (
        patch("pipeline.heading_refinement.settings") as s,
        patch(
            "pipeline.heading_refinement.classify_generic_heading_is_section_title",
        ) as mock_llm,
    ):
        s.ENABLE_LLM_GENERIC_HEADING_REFINE = True
        s.LLM_GENERIC_HEADING_REFINE_ALL = False

        n = refine_generic_chunk_headings(
            chunks,
            "Doc",
            "",
            enable_refinement=True,
            classify_force_llm=False,
            refine_all=False,
            run_id="run-test-4",
            blob_path="path/x.pdf",
        )

    assert n == 0
    mock_llm.assert_not_called()


def test_refine_returns_zero_when_enable_refinement_false() -> None:
    chunks = [
        Chunk(
            text="casos, se considera adecuado",
            chunk_no=1,
            chunk_type="generic",
            article_ref=None,
            heading="casos, se considera adecuado",
            start_page=1,
            end_page=1,
            has_table=False,
            table_index=None,
        )
    ]
    with patch(
        "pipeline.heading_refinement.classify_generic_heading_is_section_title",
    ) as mock_llm:
        n = refine_generic_chunk_headings(
            chunks,
            "Doc",
            "",
            enable_refinement=False,
            classify_force_llm=False,
            refine_all=False,
            run_id="run-test-5",
            blob_path="path/y.pdf",
        )
    assert n == 0
    mock_llm.assert_not_called()
