"""Extract metadata from legal documents using regex and stdlib."""

from __future__ import annotations

import re
from datetime import date
from typing import Any

SEARCH_LIMIT_CHARS = 8000
SPANISH_MONTHS: dict[str, int] = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}

ULTIMA_REFORMA_REGEX = re.compile(
    r"(?:última|ultima)s?\s+reformas?\s+(?:publicadas?\s+)?dof\s+(\d{2})-(\d{2})-(\d{4})",
    re.IGNORECASE,
)
NUEVA_LEY_REGEX = re.compile(
    r"nueva\s+ley\s+publicada\s+en\s+el\s+diario\s+oficial\s+de\s+la\s+federaci[oó]n\s+el\s+(\d{1,2})\s+de\s+"
    r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+de\s+(\d{4})",
    re.IGNORECASE,
)
TEXTO_VIGENTE_REGEX = re.compile(r"\bTEXTO\s+VIGENTE\b", re.IGNORECASE)
CANTIDADES_REGEX = re.compile(
    r"cantidades\s+actualizadas\s+por\s+[^.]*dof\s+(\d{2})-(\d{2})-(\d{4})",
    re.IGNORECASE,
)


def _is_valid_date(day: int, month: int, year: int) -> bool:
    if day < 1 or day > 31:
        return False
    if month < 1 or month > 12:
        return False
    if year < 1900 or year > 2100:
        return False
    try:
        d = date(year, month, day)
        return d.day == day and d.month == month and d.year == year
    except ValueError:
        return False


def _parse_dd_mm_yyyy(match: re.Match[str]) -> str | None:
    day = int(match.group(1), 10)
    month = int(match.group(2), 10)
    year = int(match.group(3), 10)
    if not _is_valid_date(day, month, year):
        return None
    return date(year, month, day).isoformat()


def _parse_spanish_date(text: str) -> str | None:
    pattern = re.compile(
        r"(\d{1,2})\s+de\s+(enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
        r"septiembre|octubre|noviembre|diciembre)\s+de\s+(\d{4})",
        re.IGNORECASE,
    )
    m = pattern.search(text)
    if not m:
        return None
    day = int(m.group(1), 10)
    month_name = m.group(2).lower()
    month = SPANISH_MONTHS.get(month_name)
    year = int(m.group(3), 10)
    if month is None or not _is_valid_date(day, month, year):
        return None
    return date(year, month, day).isoformat()


def _clean_legend_text(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\n+", " ", s)).strip()


def extract_legal_legend(content: str | None) -> dict[str, Any]:
    """
    Extract legal legend from PDF content. Aligned with assistax-back legalLegendExtractor.
    Returns dict with: legalLegend (str), publicationDate (ISO), lastReformDate (ISO).
    """
    result: dict[str, Any] = {}
    if not content or not isinstance(content, str):
        return result

    search_text = content[:SEARCH_LIMIT_CHARS]
    lines = [l.strip() for l in search_text.splitlines() if l.strip()]

    collected_lines: list[str] = []
    last_reform_date: str | None = None
    publication_date: str | None = None
    seen: set[str] = set()

    for line in lines:
        if len(line) < 10:
            continue

        ultima_match = ULTIMA_REFORMA_REGEX.search(line)
        if ultima_match:
            iso_date = _parse_dd_mm_yyyy(ultima_match)
            if iso_date:
                last_reform_date = iso_date
                phrase = ultima_match.group(0).strip()
                if phrase and "ultima" not in seen:
                    seen.add("ultima")
                    collected_lines.append(phrase)
            continue

        nueva_match = NUEVA_LEY_REGEX.search(line)
        if nueva_match:
            iso_date = _parse_spanish_date(line)
            if iso_date:
                publication_date = iso_date
                if "nuevaley" not in seen:
                    seen.add("nuevaley")
                    collected_lines.append(nueva_match.group(0).strip())
            continue

        if TEXTO_VIGENTE_REGEX.search(line) and "vigente" not in seen:
            seen.add("vigente")
            collected_lines.append(line)
            continue

        cant_match = CANTIDADES_REGEX.search(line)
        if cant_match:
            iso_date = _parse_dd_mm_yyyy(cant_match)
            if iso_date and "cantidades" not in seen:
                seen.add("cantidades")
                collected_lines.append(cant_match.group(0).strip())

    if collected_lines:
        result["legalLegend"] = _clean_legend_text(" ".join(collected_lines))
    if publication_date:
        result["publicationDate"] = publication_date
    if last_reform_date:
        result["lastReformDate"] = last_reform_date

    return result


def infer_doc_type(title: str) -> str:
    """
    Detecta el tipo de documento por palabras clave en el título (case-insensitive).
    Orden de evaluación:
    "reglamento" → "reglamento"
    "código" o "codigo" → "codigo"
    "presupuesto" → "presupuesto"
    "decreto" → "decreto"
    "acuerdo" → "acuerdo"
    "norma" o "nom-" → "norma"
    "ley" → "ley"
    default → "otro"
    Retorna minúsculas.
    """
    if not title or not isinstance(title, str):
        return "otro"
    lower = title.lower()
    if "reglamento" in lower:
        return "reglamento"
    if "código" in lower or "codigo" in lower:
        return "codigo"
    if "presupuesto" in lower:
        return "presupuesto"
    if "decreto" in lower:
        return "decreto"
    if "acuerdo" in lower:
        return "acuerdo"
    if "norma" in lower or "nom-" in lower:
        return "norma"
    if "ley" in lower:
        return "ley"
    return "otro"


_STOP_WORDS = frozenset(
    {
        "de", "del", "la", "los", "las", "el",
        "sobre", "en", "para", "por", "al", "con", "que", "y", "e", "o", "u",
        "federal", "general", "nacional", "nueva",
    }
)


def extract_law_name(title: str) -> str:
    """
    Extract law name from document title. Keeps full name, removes trailing parentheses.
    Aligned with assistax-back legalNormalizationService.extractLawName.
    """
    if not title or not isinstance(title, str):
        return ""
    normalized = re.sub(r"\s+", " ", title).strip()
    cleaned = re.sub(r"\s*\(.*?\)\s*$", "", normalized).strip()
    return cleaned[:255] if len(cleaned) > 255 else cleaned


def normalize_law_name(title: str) -> str:
    """
    Genera siglas del nombre de la ley.
    1. Palabras a ignorar para las siglas: {"de", "del", "la", "los", "las", "el",
       "sobre", "en", "para", "por", "al", "con", "que", "y", "e", "o", "u",
       "federal", "general", "nacional", "nueva"}
    2. Tomar la primera letra mayúscula de cada palabra NO ignorada.
    3. Si el resultado tiene menos de 2 chars, retornar title[:50].strip()
    Ejemplo: "Ley Federal del Impuesto sobre Automóviles Nuevos" → "LIAN"
    Retorna string, nunca None.
    """
    if not title or not isinstance(title, str):
        return ""
    words = title.split()
    acronym_chars: list[str] = []
    for w in words:
        if not w:
            continue
        lower = w.lower()
        if lower in _STOP_WORDS:
            continue
        first = w[0]
        if first.isalpha():
            acronym_chars.append(first.upper())
    result = "".join(acronym_chars)
    if len(result) < 2:
        return title[:50].strip()
    return result
