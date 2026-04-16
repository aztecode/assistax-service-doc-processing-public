"""Unit tests for generic_heading_classifier (mocked Azure OpenAI)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from pipeline.generic_heading_classifier import classify_generic_heading_is_section_title


def test_flag_off_returns_none() -> None:
    with patch("pipeline.generic_heading_classifier.settings") as s:
        s.ENABLE_LLM_GENERIC_HEADING_REFINE = False
        verdict, invoked = classify_generic_heading_is_section_title("any", "text", "title")
    assert verdict is None
    assert invoked is False


def test_llm_returns_false_for_body_heading() -> None:
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content='{"isSectionTitle": false}'))]

    with (
        patch("pipeline.generic_heading_classifier.settings") as s,
        patch("pipeline.generic_heading_classifier.AzureOpenAI") as mock_client_cls,
    ):
        s.ENABLE_LLM_GENERIC_HEADING_REFINE = True
        s.AZURE_OPENAI_ENDPOINT = "https://x.openai.azure.com"
        s.AZURE_OPENAI_API_KEY = "k"
        s.AZURE_OPENAI_API_VERSION = "2024-02-01"
        s.AZURE_OPENAI_CHAT_DEPLOYMENT = "gpt-4o-mini"
        s.OPENAI_MAX_CONCURRENT = 2
        inst = mock_client_cls.return_value
        inst.chat.completions.create.return_value = mock_resp

        verdict, invoked = classify_generic_heading_is_section_title(
            "casos, se considera adecuado",
            "casos, se considera adecuado otorgar estímulo",
            "Decreto estímulos DOF",
        )

    assert verdict is False
    assert invoked is True
    inst.chat.completions.create.assert_called_once()


def test_llm_returns_true_keeps_section_title() -> None:
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content='{"isSectionTitle": true}'))]

    with (
        patch("pipeline.generic_heading_classifier.settings") as s,
        patch("pipeline.generic_heading_classifier.AzureOpenAI") as mock_client_cls,
    ):
        s.ENABLE_LLM_GENERIC_HEADING_REFINE = True
        s.AZURE_OPENAI_ENDPOINT = "https://x.openai.azure.com"
        s.AZURE_OPENAI_API_KEY = "k"
        s.AZURE_OPENAI_API_VERSION = "2024-02-01"
        s.AZURE_OPENAI_CHAT_DEPLOYMENT = "gpt-4o-mini"
        s.OPENAI_MAX_CONCURRENT = 2
        inst = mock_client_cls.return_value
        inst.chat.completions.create.return_value = mock_resp

        verdict, invoked = classify_generic_heading_is_section_title(
            "CONSIDERANDO",
            "CONSIDERANDO Que la Constitución",
            "Decreto",
        )

    assert verdict is True
    assert invoked is True


def test_llm_failure_returns_none() -> None:
    with (
        patch("pipeline.generic_heading_classifier.settings") as s,
        patch("pipeline.generic_heading_classifier.AzureOpenAI") as mock_client_cls,
    ):
        s.ENABLE_LLM_GENERIC_HEADING_REFINE = True
        s.AZURE_OPENAI_ENDPOINT = "https://x.openai.azure.com"
        s.AZURE_OPENAI_API_KEY = "k"
        s.AZURE_OPENAI_API_VERSION = "2024-02-01"
        s.AZURE_OPENAI_CHAT_DEPLOYMENT = "gpt-4o-mini"
        s.OPENAI_MAX_CONCURRENT = 2
        mock_client_cls.return_value.chat.completions.create.side_effect = RuntimeError("api down")

        verdict, invoked = classify_generic_heading_is_section_title("h", "t", "doc")

    assert verdict is None
    assert invoked is True
