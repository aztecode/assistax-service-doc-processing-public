"""
LLM-based heading classifier for legal document TOC nodes.
Only invoked when heuristics would discard a structural candidate (rescue mode).
Falls back silently to heuristic decision on LLM failure or when flag is off.
"""
from __future__ import annotations

import json
import re

import structlog
from openai import AzureOpenAI

from settings import settings

logger = structlog.get_logger()

# Only structural types where false positives / negatives cause real damage to the index.
# Articles and rules are reliably detected by regex and are excluded.
LLM_ELIGIBLE_TYPES: frozenset[str] = frozenset({"title", "chapter", "section"})

ALLOWED_CONFIDENCES: tuple[str, ...] = ("alta", "media", "baja")

_SYSTEM_PROMPT = """\
Eres un clasificador de encabezados en documentos legales mexicanos.
No expliques nada.
No inventes información.
No salgas del formato solicitado.

Dado el texto inicial de un fragmento de ley y el tipo estructural detectado,
determina si la primera línea es:

A) Un ENCABEZADO ESTRUCTURAL real que introduce una sección del documento
   (por ejemplo: "Capítulo III\\nDe las obligaciones...").

B) Una REFERENCIA al cuerpo del texto que simplemente MENCIONA ese término
   (por ejemplo: "...conforme al Capítulo III de esta Ley, se considerará...").

Devuelve exclusivamente un JSON con esta estructura exacta:

{
  "isStructural": true | false,
  "confidence": "alta | media | baja"
}"""

# Few-shot conversation: 6 examples (3 structural / 3 body references)
_FEW_SHOT_MESSAGES: list[dict[str, str]] = [
    # Example 1 — structural chapter
    {
        "role": "user",
        "content": 'tipo: chapter\ntexto: "Capítulo III\\nDe las obligaciones de los contribuyentes"',
    },
    {"role": "assistant", "content": '{"isStructural": true, "confidence": "alta"}'},
    # Example 2 — body reference to chapter
    {
        "role": "user",
        "content": 'tipo: chapter\ntexto: "...conforme al Capítulo III de esta Ley, se considerará ingreso gravable..."',
    },
    {"role": "assistant", "content": '{"isStructural": false, "confidence": "alta"}'},
    # Example 3 — structural title
    {
        "role": "user",
        "content": 'tipo: title\ntexto: "Título Segundo\\nDe la capacidad y representación de las personas"',
    },
    {"role": "assistant", "content": '{"isStructural": true, "confidence": "alta"}'},
    # Example 4 — body reference to title
    {
        "role": "user",
        "content": 'tipo: title\ntexto: "...señalado en el Título Segundo del Código Fiscal de la Federación, se aplicará..."',
    },
    {"role": "assistant", "content": '{"isStructural": false, "confidence": "alta"}'},
    # Example 5 — structural section
    {
        "role": "user",
        "content": 'tipo: section\ntexto: "Sección II\\nDe los sujetos exentos del impuesto"',
    },
    {"role": "assistant", "content": '{"isStructural": true, "confidence": "alta"}'},
    # Example 6 — body reference to section
    {
        "role": "user",
        "content": 'tipo: section\ntexto: "...de la Sección II que incluye los artículos 10 al 15 de la presente Ley..."',
    },
    {"role": "assistant", "content": '{"isStructural": false, "confidence": "alta"}'},
]


def _build_user_message(heading: str, chunk_type: str, chunk_text_prefix: str) -> str:
    """Format the classification request for the model."""
    text_sample = re.sub(r"\s+", " ", chunk_text_prefix.strip())[:300]
    return f'tipo: {chunk_type}\ntexto: "{text_sample}"'


def _extract_json(content: str) -> dict | None:
    match = re.search(r"\{[\s\S]*\}", content)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _parse_response(content: str) -> tuple[bool, str]:
    payload = _extract_json(content)
    if not payload or not isinstance(payload, dict):
        raise ValueError("Respuesta inválida: no es JSON")

    is_structural_raw = payload.get("isStructural")
    confidence_raw = str(payload.get("confidence", "")).strip().lower()

    if not isinstance(is_structural_raw, bool):
        raise ValueError(f"isStructural inválido: {is_structural_raw!r}")
    if confidence_raw not in ALLOWED_CONFIDENCES:
        raise ValueError(f"confidence inválido: {confidence_raw!r}")

    return (is_structural_raw, confidence_raw)


def classify_heading_node(
    heading: str,
    chunk_type: str,
    chunk_text_prefix: str,
) -> tuple[bool, str]:
    """
    Decide whether a structural heading candidate is a real TOC node.

    Returns:
        (is_structural, confidence)
        - is_structural: True when the heading is a genuine structural node.
        - confidence: "alta" | "media" | "baja"

    Falls back to (False, "baja") on any LLM error so the caller keeps the
    heuristic decision unchanged.

    Only called when ENABLE_LLM_HEADING_CLASSIFIER=true and chunk_type is
    in LLM_ELIGIBLE_TYPES.
    """
    if not settings.ENABLE_LLM_HEADING_CLASSIFIER:
        return (False, "baja")

    try:
        client = AzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )
        user_message = _build_user_message(heading, chunk_type, chunk_text_prefix)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *_FEW_SHOT_MESSAGES,
            {"role": "user", "content": user_message},
        ]

        response = client.chat.completions.create(
            model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=messages,
            temperature=0,
        )

        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise ValueError("Respuesta vacía del modelo")

        is_structural, confidence = _parse_response(content)
        logger.info(
            "heading_classifier.classified",
            heading=heading[:80],
            chunk_type=chunk_type,
            is_structural=is_structural,
            confidence=confidence,
        )
        return (is_structural, confidence)

    except Exception as exc:
        logger.warning(
            "heading_classifier.llm_failed_fallback",
            error=str(exc),
            heading=heading[:80],
            chunk_type=chunk_type,
        )
        return (False, "baja")
