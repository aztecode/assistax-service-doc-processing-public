"""
Tests for pipeline.boxed_note_classifier — deterministic classification,
LLM arbiter (mocked), and orchestrator.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pipeline.boxed_note_classifier import (
    AmbiguousBlock,
    BlockClassification,
    classify_and_route_block,
    classify_block_deterministic,
    classify_boxed_note_candidate,
    summarize_arbiter_logs,
    _parse_arbiter_response,
)


def _make_block(
    text: str,
    is_inside_visual_box: bool = False,
    is_table_like: bool = False,
    rows_count: int = 1,
    cols_count: int = 1,
) -> AmbiguousBlock:
    return AmbiguousBlock(
        text=text,
        bbox=(0, 0, 100, 50),
        page=306,
        source="test",
        nearby_text_before="",
        nearby_text_after="",
        is_inside_visual_box=is_inside_visual_box,
        is_table_like=is_table_like,
        rows_count=rows_count,
        cols_count=cols_count,
    )


# ---------------------------------------------------------------------------
# classify_block_deterministic
# ---------------------------------------------------------------------------


class TestClassifyBlockDeterministic:
    def test_empty_text_returns_editorial(self) -> None:
        block = _make_block("")
        assert classify_block_deterministic(block) == "editorial_note"

    def test_aclaracion_prefix_returns_editorial(self) -> None:
        block = _make_block("ACLARACIÓN: El texto se ajustó conforme a fe de erratas.")
        assert classify_block_deterministic(block) == "editorial_note"

    def test_fe_erratas_prefix_returns_editorial(self) -> None:
        block = _make_block("Fe de erratas publicada en el DOF el 15-03-2020")
        assert classify_block_deterministic(block) == "editorial_note"

    def test_nota_editorial_prefix_returns_editorial(self) -> None:
        block = _make_block("Nota editorial: se incluye texto corregido.")
        assert classify_block_deterministic(block) == "editorial_note"

    def test_article_returns_structural(self) -> None:
        block = _make_block("Artículo 307.- Los responsables de publicidad.")
        assert classify_block_deterministic(block) == "structural"

    def test_capitulo_returns_structural(self) -> None:
        block = _make_block("Capítulo III\nDe las obligaciones")
        assert classify_block_deterministic(block) == "structural"

    def test_transitorios_returns_structural(self) -> None:
        block = _make_block("TRANSITORIOS\nPrimero.- El decreto entrará en vigor...")
        assert classify_block_deterministic(block) == "structural"

    def test_inside_box_with_dof_dates_returns_editorial(self) -> None:
        text = (
            "Este artículo fue modificado. Reforma DOF 15-03-2020. "
            "Publicada en el DOF el 01-06-2021."
        )
        block = _make_block(text, is_inside_visual_box=True, cols_count=1)
        assert classify_block_deterministic(block) == "editorial_note"

    def test_inside_box_long_prose_single_col_returns_editorial(self) -> None:
        words = " ".join(["palabra"] * 20)
        block = _make_block(words, is_inside_visual_box=True, cols_count=1)
        assert classify_block_deterministic(block) == "editorial_note"

    def test_inside_box_prose_no_table_returns_ambiguous(self) -> None:
        words = " ".join(["texto"] * 12)
        block = _make_block(
            words,
            is_inside_visual_box=True,
            is_table_like=False,
            cols_count=3,
        )
        assert classify_block_deterministic(block) == "ambiguous"

    def test_single_col_long_text_no_structure_returns_ambiguous(self) -> None:
        words = " ".join(["contenido"] * 25)
        block = _make_block(words, cols_count=1, rows_count=2)
        assert classify_block_deterministic(block) == "ambiguous"

    def test_nota_aclaratoria_returns_editorial(self) -> None:
        block = _make_block("NOTA ACLARATORIA: Se corrige el texto del artículo.")
        assert classify_block_deterministic(block) == "editorial_note"

    def test_nota_del_editor_returns_editorial(self) -> None:
        block = _make_block("NOTA DEL EDITOR: Se incluye reforma vigente.")
        assert classify_block_deterministic(block) == "editorial_note"

    def test_regular_short_text_returns_structural(self) -> None:
        block = _make_block("Los montos señalados.")
        assert classify_block_deterministic(block) == "structural"


# ---------------------------------------------------------------------------
# _parse_arbiter_response
# ---------------------------------------------------------------------------


class TestParseArbiterResponse:
    def test_valid_editorial_note(self) -> None:
        content = '{"kind": "editorial_note", "confidence": "alta", "reasonCode": "aclaracion"}'
        kind, conf, reason = _parse_arbiter_response(content)
        assert kind == "editorial_note"
        assert conf == "alta"
        assert reason == "aclaracion"

    def test_valid_structural(self) -> None:
        content = '{"kind": "structural", "confidence": "media", "reasonCode": "articulo"}'
        kind, conf, reason = _parse_arbiter_response(content)
        assert kind == "structural"
        assert conf == "media"

    def test_invalid_kind_raises(self) -> None:
        content = '{"kind": "unknown", "confidence": "alta", "reasonCode": "x"}'
        with pytest.raises(ValueError, match="kind inválido"):
            _parse_arbiter_response(content)

    def test_invalid_confidence_raises(self) -> None:
        content = '{"kind": "structural", "confidence": "xxx", "reasonCode": "x"}'
        with pytest.raises(ValueError, match="confidence inválido"):
            _parse_arbiter_response(content)

    def test_not_json_raises(self) -> None:
        with pytest.raises(ValueError, match="no es JSON"):
            _parse_arbiter_response("not json at all")


# ---------------------------------------------------------------------------
# classify_boxed_note_candidate (LLM mocked)
# ---------------------------------------------------------------------------


class TestClassifyBoxedNoteCandidate:
    def test_arbiter_disabled_returns_structural_fallback(self) -> None:
        block = _make_block("Texto ambiguo.")
        with patch("pipeline.boxed_note_classifier.settings") as mock_settings:
            mock_settings.ENABLE_LLM_BOXED_NOTE_ARBITER = False
            kind, conf, reason = classify_boxed_note_candidate(block)
        assert kind == "structural"
        assert reason == "arbiter_disabled"

    def test_arbiter_llm_returns_editorial(self) -> None:
        block = _make_block("Texto ambiguo dentro de recuadro.")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            '{"kind": "editorial_note", "confidence": "alta", "reasonCode": "nota_recuadro"}'
        )
        with patch("pipeline.boxed_note_classifier.settings") as mock_settings, \
             patch("pipeline.boxed_note_classifier.AzureOpenAI") as mock_client_cls:
            mock_settings.ENABLE_LLM_BOXED_NOTE_ARBITER = True
            mock_settings.AZURE_OPENAI_ENDPOINT = "https://test"
            mock_settings.AZURE_OPENAI_API_KEY = "key"
            mock_settings.AZURE_OPENAI_API_VERSION = "v1"
            mock_settings.AZURE_OPENAI_CHAT_DEPLOYMENT = "gpt-4o-mini"
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_client_cls.return_value = mock_client

            kind, conf, reason = classify_boxed_note_candidate(block)

        assert kind == "editorial_note"
        assert conf == "alta"
        assert reason == "nota_recuadro"

    def test_arbiter_llm_low_confidence_falls_back_to_structural(self) -> None:
        block = _make_block("Texto ambiguo.")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            '{"kind": "editorial_note", "confidence": "baja", "reasonCode": "uncertain"}'
        )
        with patch("pipeline.boxed_note_classifier.settings") as mock_settings, \
             patch("pipeline.boxed_note_classifier.AzureOpenAI") as mock_client_cls:
            mock_settings.ENABLE_LLM_BOXED_NOTE_ARBITER = True
            mock_settings.AZURE_OPENAI_ENDPOINT = "https://test"
            mock_settings.AZURE_OPENAI_API_KEY = "key"
            mock_settings.AZURE_OPENAI_API_VERSION = "v1"
            mock_settings.AZURE_OPENAI_CHAT_DEPLOYMENT = "gpt-4o-mini"
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_client_cls.return_value = mock_client

            kind, conf, reason = classify_boxed_note_candidate(block)

        assert kind == "structural"
        assert "low_confidence" in reason

    def test_arbiter_llm_error_falls_back_safely(self) -> None:
        block = _make_block("Texto ambiguo.")
        with patch("pipeline.boxed_note_classifier.settings") as mock_settings, \
             patch("pipeline.boxed_note_classifier.AzureOpenAI") as mock_client_cls:
            mock_settings.ENABLE_LLM_BOXED_NOTE_ARBITER = True
            mock_settings.AZURE_OPENAI_ENDPOINT = "https://test"
            mock_settings.AZURE_OPENAI_API_KEY = "key"
            mock_settings.AZURE_OPENAI_API_VERSION = "v1"
            mock_settings.AZURE_OPENAI_CHAT_DEPLOYMENT = "gpt-4o-mini"
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = RuntimeError("API error")
            mock_client_cls.return_value = mock_client

            kind, conf, reason = classify_boxed_note_candidate(block)

        assert kind == "structural"
        assert reason == "llm_error"


# ---------------------------------------------------------------------------
# classify_and_route_block (orchestrator)
# ---------------------------------------------------------------------------


class TestClassifyAndRouteBlock:
    def test_deterministic_editorial_no_llm_call(self) -> None:
        block = _make_block("ACLARACIÓN: texto corregido.")
        kind, decision, reason = classify_and_route_block(block)
        assert kind == "editorial_note"
        assert decision == "deterministic_editorial"

    def test_deterministic_structural_no_llm_call(self) -> None:
        block = _make_block("Artículo 307.- Disposiciones.")
        kind, decision, reason = classify_and_route_block(block)
        assert kind == "structural"
        assert decision == "deterministic_structural"

    def test_ambiguous_escalates_to_llm(self) -> None:
        words = " ".join(["contenido"] * 25)
        block = _make_block(words, cols_count=1, rows_count=2)
        assert classify_block_deterministic(block) == "ambiguous"

        with patch("pipeline.boxed_note_classifier.classify_boxed_note_candidate") as mock_llm:
            mock_llm.return_value = ("editorial_note", "alta", "nota_test")
            kind, decision, reason = classify_and_route_block(block)

        assert kind == "editorial_note"
        assert decision == "llm_editorial"
        mock_llm.assert_called_once()


# ---------------------------------------------------------------------------
# summarize_arbiter_logs
# ---------------------------------------------------------------------------


class TestSummarizeArbiterLogs:
    def test_empty_logs(self) -> None:
        result = summarize_arbiter_logs([])
        assert result["totalBlocks"] == 0
        assert result["llmEditorial"] == 0

    def test_mixed_decisions(self) -> None:
        logs = [
            {"page": 1, "decision": "deterministic_editorial", "kind": "editorial_note"},
            {"page": 2, "decision": "deterministic_structural", "kind": "structural"},
            {"page": 3, "decision": "llm_editorial", "kind": "editorial_note"},
            {"page": 4, "decision": "llm_fallback", "kind": "structural"},
        ]
        result = summarize_arbiter_logs(logs)
        assert result["totalBlocks"] == 4
        assert result["deterministicEditorial"] == 1
        assert result["deterministicStructural"] == 1
        assert result["llmEditorial"] == 1
        assert result["llmFallback"] == 1
        assert result["llmStructural"] == 0
