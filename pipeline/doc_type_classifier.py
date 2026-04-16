"""
Clasificación de tipo de documento vía LLM (Azure OpenAI).
Alineado con assistax-fn: mismo prompt, JSON docType/confidence (alta|media|baja).
Solo se usa cuando ENABLE_LLM_DOC_TYPE=true.
En fallo (timeout, parse error) → fallback silencioso a infer_doc_type.
"""
import json
import re

import structlog
from openai import AzureOpenAI

from pipeline.metadata_extractor import infer_doc_type
from settings import settings

logger = structlog.get_logger()

ALLOWED_DOC_TYPES = (
    "ley", "codigo", "reglamento", "presupuesto", "resolucion", "estatuto",
)
ALLOWED_CONFIDENCES = ("alta", "media", "baja")

CLASSIFICATION_PROMPT = """Eres un sistema de clasificación jurídica.
No expliques nada.
No inventes información.
No salgas del formato solicitado.

A partir del título y encabezados de un documento legal mexicano,
determina su tipo jurídico.

Tipos permitidos (elige SOLO uno):
- ley
- codigo
- reglamento
- presupuesto
- resolucion
- estatuto

Devuelve exclusivamente un JSON con esta estructura exacta:

{
  "docType": "<tipo>",
  "confidence": "alta | media | baja"
}"""


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def _build_summary(title: str, headings: list[str]) -> str:
    """Formato alineado con assistax-fn."""
    normalized_title = _normalize_text(title)
    seen: set[str] = set()
    unique_headings: list[str] = []
    for h in headings:
        normalized = _normalize_text(h)
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        unique_headings.append(normalized)
    heading_text = " | ".join(unique_headings) if unique_headings else "Sin encabezados disponibles"
    return f"Titulo: {normalized_title}\nEncabezados: {heading_text}"


def _extract_json(content: str) -> dict | None:
    match = re.search(r"\{[\s\S]*\}", content)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _parse_classification(content: str) -> tuple[str, str]:
    payload = _extract_json(content)
    if not payload or not isinstance(payload, dict):
        raise ValueError("Respuesta inválida: no es JSON")

    doc_type_raw = str(payload.get("docType", "")).strip().lower()
    confidence_raw = str(payload.get("confidence", "")).strip().lower()

    if doc_type_raw not in ALLOWED_DOC_TYPES:
        raise ValueError(f"docType inválido: {doc_type_raw}")
    if confidence_raw not in ALLOWED_CONFIDENCES:
        raise ValueError(f"confidence inválido: {confidence_raw}")

    return (doc_type_raw, confidence_raw)


def classify_doc_type(title: str, headings: list[str]) -> tuple[str, str, str]:
    """
    Clasifica el tipo de documento usando el LLM.
    Alineado con assistax-fn.

    Returns:
        (doc_type, confidence, source)
        - doc_type: uno de ley, codigo, reglamento, presupuesto, resolucion, estatuto
        - confidence: "alta" | "media" | "baja"
        - source: "azure-openai" cuando LLM usado, "" cuando fallback
    """
    if not settings.ENABLE_LLM_DOC_TYPE:
        doc_type = infer_doc_type(title)
        return (doc_type, "media", "")

    try:
        client = AzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )
        summary = _build_summary(title, (headings or [])[:40])

        response = client.chat.completions.create(
            model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": CLASSIFICATION_PROMPT},
                {"role": "user", "content": summary},
            ],
            temperature=0,
        )

        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise ValueError("Respuesta vacía del modelo")

        doc_type, confidence = _parse_classification(content)
        logger.info(
            "doc_type.classified",
            doc_type=doc_type,
            confidence=confidence,
            title=title[:80],
        )
        return (doc_type, confidence, "azure-openai")
    except Exception as e:
        doc_type = infer_doc_type(title)
        logger.warning(
            "doc_type.llm_failed_fallback",
            error=str(e),
            fallback=doc_type,
        )
        return (doc_type, "media", "")
