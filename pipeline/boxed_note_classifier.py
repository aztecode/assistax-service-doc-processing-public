"""
Deterministic + LLM-arbitrated classifier for ambiguous blocks in legal PDFs.

Architecture:
  1. classify_block_deterministic() — ternary output: structural / editorial_note / ambiguous.
  2. classify_boxed_note_candidate() — LLM arbiter invoked only for 'ambiguous' blocks.
  3. classify_and_route_block() — orchestrator that chains both steps.

The LLM arbiter follows the same pattern as heading_classifier.py:
  temperature=0, structured JSON output, safe fallback on any failure.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

import structlog
from openai import AzureOpenAI

from settings import settings

logger = structlog.get_logger()

BlockClassification = Literal["structural", "editorial_note", "ambiguous"]

ARBITER_DECISION = Literal[
    "deterministic_structural",
    "deterministic_editorial",
    "llm_structural",
    "llm_editorial",
    "llm_fallback",
]


@dataclass
class AmbiguousBlock:
    """Rich representation of a block whose nature is uncertain."""
    text: str
    bbox: tuple[float, float, float, float]
    page: int
    source: str
    nearby_text_before: str
    nearby_text_after: str
    is_inside_visual_box: bool
    is_table_like: bool
    rows_count: int
    cols_count: int


_EDITORIAL_PREFIXES = re.compile(
    r"(?:ACLARACI[OÓ]N|Nota\s*de\s+erratas|Fe\s+de\s+erratas|N\.\s*de\s+E\."
    r"|NOTA\s+DEL\s+EDITOR|Nota\s*:\s|Art[íi]culo\s+(?:reformad[oa]|adicionad[oa]|derogad[oa])"
    r"|NOTA\s+ACLARATORIA|Correcci[oó]n|AVISO|Nota\s+editorial)",
    re.IGNORECASE,
)

_STRUCTURAL_PATTERNS = re.compile(
    r"^(?:Art[íi]culo\s+\d|Cap[íi]tulo\s+[IVX\d]|T[íi]tulo\s+[IVX\d]"
    r"|Secci[oó]n\s+[IVX\d]|Libro\s+[IVX\d]|Regla\s+\d|TRANSITORIOS?)\b",
    re.IGNORECASE,
)

_DATE_DOF_PATTERN = re.compile(
    r"(?:DOF|Diario\s+Oficial|publicad[ao]|vigente|vigor)\s",
    re.IGNORECASE,
)

_NUMERIC_REF_DENSE = re.compile(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}")


def classify_block_deterministic(block: AmbiguousBlock) -> BlockClassification:
    """
    Ternary deterministic classifier for a text block.

    Returns:
      - 'structural': clearly part of the law's normative structure.
      - 'editorial_note': clearly an editorial / errata / clarification note.
      - 'ambiguous': cannot decide with confidence; escalate to LLM arbiter.
    """
    text = block.text.strip()
    if not text:
        return "editorial_note"

    first_line = text.split("\n", 1)[0].strip()

    if _EDITORIAL_PREFIXES.search(first_line):
        return "editorial_note"

    if _STRUCTURAL_PATTERNS.match(first_line):
        return "structural"

    if block.is_inside_visual_box and block.cols_count <= 2:
        date_refs = len(_NUMERIC_REF_DENSE.findall(text))
        dof_mentions = len(_DATE_DOF_PATTERN.findall(text))
        if date_refs >= 2 or dof_mentions >= 2:
            return "editorial_note"

        words = text.split()
        if len(words) >= 15 and block.cols_count == 1:
            return "editorial_note"

    if block.is_inside_visual_box and not block.is_table_like:
        words = text.split()
        if len(words) >= 10:
            return "ambiguous"

    if block.cols_count == 1 and block.rows_count <= 3:
        words = text.split()
        if len(words) >= 20 and not _STRUCTURAL_PATTERNS.match(first_line):
            return "ambiguous"

    return "structural"


# ---------------------------------------------------------------------------
# LLM arbiter (Etapa 4)
# ---------------------------------------------------------------------------

_ARBITER_SYSTEM_PROMPT = """\
Eres un clasificador de bloques de texto en documentos legales mexicanos.
No expliques nada. No inventes información.

Dado un bloque de texto extraído de un PDF, determina si es:

A) CONTENIDO ESTRUCTURAL: texto normativo que forma parte del cuerpo de la ley
   (artículos, capítulos, títulos, disposiciones, transitorios).

B) NOTA EDITORIAL: notas de aclaración, fe de erratas, correcciones del editor,
   avisos sobre reformas, texto informativo dentro de recuadros que NO forma
   parte del contenido normativo.

Devuelve exclusivamente un JSON con esta estructura:

{
  "kind": "structural" | "editorial_note",
  "confidence": "alta" | "media" | "baja",
  "reasonCode": "string_corto_sin_espacios"
}"""

_ARBITER_FEW_SHOT: list[dict[str, str]] = [
    {
        "role": "user",
        "content": (
            "bloque: \"ACLARACIÓN: El texto del artículo 306 Bis se corrigió "
            "conforme a Fe de Erratas publicada en el DOF el 15-03-2020.\"\n"
            "dentro_de_recuadro: sí\ncolumnas: 1"
        ),
    },
    {
        "role": "assistant",
        "content": '{"kind": "editorial_note", "confidence": "alta", "reasonCode": "aclaracion_fe_erratas"}',
    },
    {
        "role": "user",
        "content": (
            "bloque: \"Artículo 307.- Los responsables de la publicidad de "
            "productos y servicios deberán cumplir...\"\n"
            "dentro_de_recuadro: no\ncolumnas: 0"
        ),
    },
    {
        "role": "assistant",
        "content": '{"kind": "structural", "confidence": "alta", "reasonCode": "articulo_normativo"}',
    },
    {
        "role": "user",
        "content": (
            "bloque: \"Nota del Editor: Se incluye el texto vigente con las reformas "
            "publicadas en el DOF 01-06-2021.\"\n"
            "dentro_de_recuadro: sí\ncolumnas: 1"
        ),
    },
    {
        "role": "assistant",
        "content": '{"kind": "editorial_note", "confidence": "alta", "reasonCode": "nota_editor"}',
    },
    {
        "role": "user",
        "content": (
            "bloque: \"TRANSITORIOS\\nPrimero.- El presente decreto entrará en vigor...\"\n"
            "dentro_de_recuadro: no\ncolumnas: 0"
        ),
    },
    {
        "role": "assistant",
        "content": '{"kind": "structural", "confidence": "alta", "reasonCode": "transitorios_decreto"}',
    },
]

ALLOWED_KINDS: frozenset[str] = frozenset({"structural", "editorial_note"})
ALLOWED_CONFIDENCES: tuple[str, ...] = ("alta", "media", "baja")


def _build_arbiter_user_message(block: AmbiguousBlock) -> str:
    text_sample = re.sub(r"\s+", " ", block.text.strip())[:400]
    inside = "sí" if block.is_inside_visual_box else "no"
    return (
        f'bloque: "{text_sample}"\n'
        f"dentro_de_recuadro: {inside}\n"
        f"columnas: {block.cols_count}"
    )


def _extract_json(content: str) -> dict | None:
    match = re.search(r"\{[\s\S]*\}", content)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _parse_arbiter_response(
    content: str,
) -> tuple[BlockClassification, str, str]:
    """Parse LLM response into (kind, confidence, reasonCode)."""
    payload = _extract_json(content)
    if not payload or not isinstance(payload, dict):
        raise ValueError("Respuesta inválida: no es JSON")

    kind_raw = str(payload.get("kind", "")).strip().lower()
    confidence_raw = str(payload.get("confidence", "")).strip().lower()
    reason_code = str(payload.get("reasonCode", "unknown")).strip()

    if kind_raw not in ALLOWED_KINDS:
        raise ValueError(f"kind inválido: {kind_raw!r}")
    if confidence_raw not in ALLOWED_CONFIDENCES:
        raise ValueError(f"confidence inválido: {confidence_raw!r}")

    kind: BlockClassification = kind_raw  # type: ignore[assignment]
    return kind, confidence_raw, reason_code


def classify_boxed_note_candidate(
    block: AmbiguousBlock,
) -> tuple[BlockClassification, str, str]:
    """
    LLM arbiter for ambiguous blocks. Only called when deterministic
    classifier returns 'ambiguous'.

    Returns (kind, confidence, reasonCode).
    Falls back to ('structural', 'baja', 'llm_error') on any failure,
    preserving the block in the semantic flow as a safe default.
    """
    if not settings.ENABLE_LLM_BOXED_NOTE_ARBITER:
        return ("structural", "baja", "arbiter_disabled")

    try:
        client = AzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )
        user_message = _build_arbiter_user_message(block)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": _ARBITER_SYSTEM_PROMPT},
            *_ARBITER_FEW_SHOT,
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

        kind, confidence, reason_code = _parse_arbiter_response(content)

        if confidence == "baja":
            logger.info(
                "boxed_note_arbiter.low_confidence_fallback",
                page=block.page,
                kind=kind,
                reason_code=reason_code,
            )
            return ("structural", "baja", f"low_confidence_{reason_code}")

        logger.info(
            "boxed_note_arbiter.classified",
            page=block.page,
            kind=kind,
            confidence=confidence,
            reason_code=reason_code,
        )
        return kind, confidence, reason_code

    except Exception as exc:
        logger.warning(
            "boxed_note_arbiter.llm_failed_fallback",
            error=str(exc),
            page=block.page,
        )
        return ("structural", "baja", "llm_error")


# ---------------------------------------------------------------------------
# Orchestrator (Etapa 5 integration point)
# ---------------------------------------------------------------------------


def classify_and_route_block(
    block: AmbiguousBlock,
) -> tuple[BlockClassification, ARBITER_DECISION, str]:
    """
    Full classification pipeline: deterministic first, LLM only if ambiguous.

    Returns (final_kind, decision_source, reason_code).
    """
    det_kind = classify_block_deterministic(block)

    if det_kind == "structural":
        return "structural", "deterministic_structural", "heuristic"

    if det_kind == "editorial_note":
        return "editorial_note", "deterministic_editorial", "heuristic"

    kind, confidence, reason_code = classify_boxed_note_candidate(block)

    if kind == "editorial_note" and confidence != "baja":
        return "editorial_note", "llm_editorial", reason_code

    if kind == "structural" and confidence != "baja":
        return "structural", "llm_structural", reason_code

    return "structural", "llm_fallback", reason_code


def summarize_arbiter_logs(
    logs: list[dict],
) -> dict:
    """
    Build a summary of arbiter decisions for metadata persistence.
    Input: list of dicts with keys page, bbox, kind, decision, reason.
    """
    if not logs:
        return {
            "totalBlocks": 0,
            "deterministicStructural": 0,
            "deterministicEditorial": 0,
            "llmStructural": 0,
            "llmEditorial": 0,
            "llmFallback": 0,
        }

    counts: dict[str, int] = {}
    for log_entry in logs:
        decision = log_entry.get("decision", "unknown")
        counts[decision] = counts.get(decision, 0) + 1

    return {
        "totalBlocks": len(logs),
        "deterministicStructural": counts.get("deterministic_structural", 0),
        "deterministicEditorial": counts.get("deterministic_editorial", 0),
        "llmStructural": counts.get("llm_structural", 0),
        "llmEditorial": counts.get("llm_editorial", 0),
        "llmFallback": counts.get("llm_fallback", 0),
    }
