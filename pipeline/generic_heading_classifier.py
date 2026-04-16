"""
LLM classifier: whether a generic chunk's stored heading is a real section title
vs mid-paragraph prose (e.g. DOF bold blocks). Used only when enabled via settings.
"""
from __future__ import annotations

import json
import re
import threading
from typing import Optional

import structlog
from openai import AzureOpenAI

from settings import settings

logger = structlog.get_logger()

_llm_semaphore = threading.Semaphore(settings.OPENAI_MAX_CONCURRENT)

_SYSTEM_PROMPT = """\
Eres un clasificador de fragmentos de documentos legales mexicanos (leyes, decretos, DOF).
No expliques nada. No inventes información.

Se te da el "heading" guardado para un fragmento genérico (sin número de artículo) y un
extracto del texto del fragmento.

Determina si el heading es un TÍTULO DE SECCIÓN real (introduce un bloque temático nuevo)
o es TEXTO DEL CUERPO (continuación de párrafo, a menudo por maquetado en negrita/corte).

Ejemplos de NO título: frases que continúan una oración ("casos, se considera adecuado...",
"considera procedente...", "de esta Ley, fuere necesario...").
Ejemplos de SÍ título: rúbricas cortas que encabezan un bloque ("CONSIDERANDO", "ÚNICO",
un epígrafe claro de decreto, nombre abreviado del acto).

Devuelve exclusivamente un JSON con esta estructura exacta:

{
  "isSectionTitle": true | false
}"""

_FEW_SHOT_MESSAGES: list[dict[str, str]] = [
    {
        "role": "user",
        "content": (
            'titulo_documento: "Decreto por estímulos fiscales DOF"\n'
            'heading: "casos, se considera adecuado otorgar un estímulo fiscal a los pasajeros"\n'
            'texto: "casos, se considera adecuado otorgar un estímulo fiscal a los pasajeros '
            'que reingresen a territorio nacional a bordo de buques de crucero, consistente en..."'
        ),
    },
    {"role": "assistant", "content": '{"isSectionTitle": false}'},
    {
        "role": "user",
        "content": (
            'titulo_documento: "Ley del Impuesto"\n'
            'heading: "Disposiciones generales aplicables"\n'
            'texto: "Disposiciones generales aplicables. Para efectos de la presente Ley se '
            'entenderá por contribuyente..."'
        ),
    },
    {"role": "assistant", "content": '{"isSectionTitle": true}'},
    {
        "role": "user",
        "content": (
            'titulo_documento: "Código Fiscal"\n'
            'heading: "de esta Ley, fuere necesario acreditar el domicilio"\n'
            'texto: "de esta Ley, fuere necesario acreditar el domicilio fiscal ante las '
            'autoridades competentes conforme a las reglas..."'
        ),
    },
    {"role": "assistant", "content": '{"isSectionTitle": false}'},
    {
        "role": "user",
        "content": (
            'titulo_documento: "Decreto de reformas"\n'
            'heading: "CONSIDERANDO"\n'
            'texto: "CONSIDERANDO Que el artículo 123 de la Constitución Política..."'
        ),
    },
    {"role": "assistant", "content": '{"isSectionTitle": true}'},
    {
        "role": "user",
        "content": (
            'titulo_documento: "Ley aduanera"\n'
            'heading: "considera procedente autorizar la medida temporal"\n'
            'texto: "considera procedente autorizar la medida temporal prevista en el '
            'artículo anterior para los sujetos siguientes..."'
        ),
    },
    {"role": "assistant", "content": '{"isSectionTitle": false}'},
]


def _build_user_message(heading: str, text_prefix: str, document_title: str) -> str:
    h = re.sub(r"\s+", " ", heading.strip())[:400]
    t = re.sub(r"\s+", " ", text_prefix.strip())[:500]
    d = re.sub(r"\s+", " ", document_title.strip())[:200]
    return f'titulo_documento: "{d}"\nheading: "{h}"\ntexto: "{t}"'


def _extract_json(content: str) -> dict | None:
    match = re.search(r"\{[\s\S]*\}", content)
    if match is None:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _classify_log_bindings(
    run_id: str | None,
    chunk_no: int | None,
    blob_path: str | None,
) -> dict[str, str | int]:
    """Bindings for structlog (omit empty / unset)."""
    out: dict[str, str | int] = {}
    if run_id is not None and run_id != "":
        out["run_id"] = run_id
    if chunk_no is not None:
        out["chunk_no"] = chunk_no
    if blob_path is not None and blob_path != "":
        out["blob_path"] = blob_path
    return out


def classify_generic_heading_is_section_title(
    heading: str,
    text_prefix: str,
    document_title: str,
    *,
    force: bool = False,
    run_id: str | None = None,
    chunk_no: int | None = None,
    blob_path: str | None = None,
) -> tuple[Optional[bool], bool]:
    """
    Returns (verdict, llm_was_called). verdict is True/False from the model, or None if the LLM
    was not called or the call failed. llm_was_called is True only when chat.completions.create ran.

    When force=True, calls the LLM even if ENABLE_LLM_GENERIC_HEADING_REFINE is false
    (for offline backfill scripts that pass credentials explicitly).
    """
    ctx = _classify_log_bindings(run_id, chunk_no, blob_path)

    if not force and not settings.ENABLE_LLM_GENERIC_HEADING_REFINE:
        return None, False

    try:
        client = AzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )
        user_message = _build_user_message(heading, text_prefix, document_title)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *_FEW_SHOT_MESSAGES,
            {"role": "user", "content": user_message},
        ]

        model = settings.AZURE_OPENAI_CHAT_DEPLOYMENT
        logger.info(
            "generic_heading_classifier.llm_invoke",
            model=model,
            heading_preview=heading[:120],
            **ctx,
        )

        with _llm_semaphore:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )

        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise ValueError("empty model response")

        payload = _extract_json(content)
        if payload is None or not isinstance(payload, dict):
            raise ValueError("invalid json")

        raw = payload.get("isSectionTitle")
        if not isinstance(raw, bool):
            raise ValueError(f"isSectionTitle not bool: {raw!r}")

        logger.info(
            "generic_heading_classifier.classified",
            heading=heading[:80],
            is_section_title=raw,
            **ctx,
        )
        return raw, True

    except Exception as exc:
        logger.warning(
            "generic_heading_classifier.llm_failed",
            error=str(exc),
            heading=(heading[:80] if heading else ""),
            **ctx,
        )
        return None, True
