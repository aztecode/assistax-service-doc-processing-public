"""
LLM-based classifier for ambiguous legal blocks (Phase 3).

Invoked only when heuristic rules in block_rules_v2 cannot confidently
classify a block.  Follows the same Azure OpenAI pattern used by
heading_classifier.py and boxed_note_classifier.py: temperature=0,
structured JSON output, safe fallback on any failure.
"""
from __future__ import annotations

import json
import logging
import re

from openai import AzureOpenAI

from pipeline.block_rules_v2 import VALID_LABELS
from pipeline.layout_models import LayoutBlock
from settings import settings

logger = logging.getLogger(__name__)

# ── Prompt ──────────────────────────────────────────────────────────────────

_VALID_LABELS_STR: str = ", ".join(sorted(VALID_LABELS))

_SYSTEM_PROMPT: str = f"""\
Eres un clasificador de bloques de texto en documentos legales mexicanos.
No expliques nada. No inventes información. No salgas del formato solicitado.

Dado un bloque de texto extraído de un PDF legal, clasifícalo con UNA de \
las siguientes etiquetas:
{_VALID_LABELS_STR}

Reglas:
- Una fecha sola (e.g. "12 DE ENERO DE 2025") NO es un título ni heading.
- Una mención a un artículo dentro de prosa ("conforme al artículo 8...") \
es article_body, NO article_heading.
- article_heading es solo cuando el bloque ABRE un artículo nuevo \
(e.g. "Artículo 5.- ...").
- Si el bloque parece un encabezado/pie de página repetido, usa \
page_header o page_footer.
- No inventes etiquetas fuera de la lista.

Devuelve exclusivamente un JSON:
{{"label": "...", "confidence": 0.0-1.0, "reason": "string_corto"}}"""

_FEW_SHOT_MESSAGES: list[dict[str, str]] = [
    {
        "role": "user",
        "content": (
            'bloque: "DIARIO OFICIAL DE LA FEDERACIÓN"\n'
            "kind: header\nzona: top\npossible_index_zone: false"
        ),
    },
    {
        "role": "assistant",
        "content": '{"label": "page_header", "confidence": 0.95, "reason": "dof_header"}',
    },
    {
        "role": "user",
        "content": (
            'bloque: "Artículo 5.- Los contribuyentes deberán presentar..."\n'
            "kind: text\nzona: middle\npossible_index_zone: false"
        ),
    },
    {
        "role": "assistant",
        "content": '{"label": "article_heading", "confidence": 0.95, "reason": "article_start"}',
    },
    {
        "role": "user",
        "content": (
            'bloque: "12 DE ENERO DE 2025"\n'
            "kind: text\nzona: top\npossible_index_zone: false"
        ),
    },
    {
        "role": "assistant",
        "content": '{"label": "article_body", "confidence": 0.90, "reason": "date_not_heading"}',
    },
    {
        "role": "user",
        "content": (
            'bloque: "TRANSITORIOS"\n'
            "kind: text\nzona: middle\npossible_index_zone: false"
        ),
    },
    {
        "role": "assistant",
        "content": '{"label": "transitory_heading", "confidence": 0.95, "reason": "transitorios_heading"}',
    },
]

# ── Response parsing ────────────────────────────────────────────────────────


def _extract_json(content: str) -> dict[str, object] | None:
    match: re.Match[str] | None = re.search(r"\{[\s\S]*\}", content)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _parse_llm_response(content: str) -> dict[str, object]:
    """Parse and validate the LLM JSON response.

    Raises ValueError on invalid output so the caller can fall back.
    """
    payload: dict[str, object] | None = _extract_json(content)
    if not payload or not isinstance(payload, dict):
        raise ValueError("LLM response is not valid JSON")

    label: str = str(payload.get("label", "")).strip()
    if label not in VALID_LABELS:
        raise ValueError(f"Invalid label from LLM: {label!r}")

    confidence_raw = payload.get("confidence", 0.0)
    try:
        confidence: float = float(confidence_raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid confidence: {confidence_raw!r}") from exc

    reason: str = str(payload.get("reason", "llm"))

    return {"label": label, "confidence": confidence, "reason": reason}


# ── User message builder ───────────────────────────────────────────────────


def _block_zone(block: LayoutBlock) -> str:
    """Coarse vertical zone label for prompt context."""
    if block.bbox[1] < 100.0:
        return "top"
    if block.bbox[3] > 750.0:
        return "bottom"
    return "middle"


def _build_user_message(
    block: LayoutBlock,
    prev_blocks: list[LayoutBlock],
    next_blocks: list[LayoutBlock],
    document_metadata: dict[str, object],
) -> str:
    text_sample: str = re.sub(r"\s+", " ", block.text.strip())[:400]
    zone: str = _block_zone(block)
    index_zone: bool = block.metadata.get("possible_index_zone") is True
    is_merged: bool = block.metadata.get("merged") is True
    source: str = block.source

    parts: list[str] = [
        f'bloque: "{text_sample}"',
        f"kind: {block.kind}",
        f"zona: {zone}",
        f"possible_index_zone: {str(index_zone).lower()}",
        f"source: {source}",
        f"merged: {str(is_merged).lower()}",
    ]

    if prev_blocks:
        prev_text: str = re.sub(r"\s+", " ", prev_blocks[-1].text.strip())[:150]
        parts.append(f'bloque_anterior: "{prev_text}"')
    if next_blocks:
        next_text: str = re.sub(r"\s+", " ", next_blocks[0].text.strip())[:150]
        parts.append(f'bloque_siguiente: "{next_text}"')

    return "\n".join(parts)


# ── Public API ──────────────────────────────────────────────────────────────


def classify_ambiguous_block(
    block: LayoutBlock,
    prev_blocks: list[LayoutBlock],
    next_blocks: list[LayoutBlock],
    document_metadata: dict[str, object],
) -> dict[str, object]:
    """Classify a single ambiguous block via Azure OpenAI.

    Returns dict with keys: label, confidence, reason.

    Falls back to {"label": "unknown", "confidence": 0.3, "reason": "llm_error"}
    on any failure so the pipeline is never broken.
    """
    fallback: dict[str, object] = {
        "label": "unknown",
        "confidence": 0.3,
        "reason": "llm_error",
    }

    if not settings.ENABLE_BLOCK_LLM_CLASSIFIER_V2:
        return {
            "label": "unknown",
            "confidence": 0.3,
            "reason": "llm_classifier_disabled",
        }

    try:
        client = AzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )

        user_message: str = _build_user_message(
            block, prev_blocks, next_blocks, document_metadata,
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *_FEW_SHOT_MESSAGES,
            {"role": "user", "content": user_message},
        ]

        response = client.chat.completions.create(
            model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=messages,  # type: ignore[arg-type]
            temperature=0,
        )

        content: str = (response.choices[0].message.content or "").strip()
        if not content:
            raise ValueError("Empty LLM response")

        result: dict[str, object] = _parse_llm_response(content)

        logger.info(
            "block_classifier_llm_v2.classified block=%s label=%s confidence=%.2f",
            block.block_id,
            result["label"],
            result["confidence"],
        )
        return result

    except Exception as exc:
        logger.warning(
            "block_classifier_llm_v2.fallback block=%s error=%s",
            block.block_id,
            str(exc),
        )
        return fallback
