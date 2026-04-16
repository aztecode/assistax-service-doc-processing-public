"""Deterministic legal chunking with table integrity rules."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

from pipeline.legal_ordinal_patterns import (
    ARTICLE_ORDINAL_WORDS_PATTERN,
    ROMAN_WITH_LOOKAHEAD,
)
from pipeline.decreto_heading import heading_for_generic_chunk, is_decreto_context
from pipeline.pdf_extractor import PageContent, TableBlock

logger = logging.getLogger(__name__)

MIN_CHUNK_SIZE = 25
MAX_HEADING_LENGTH = 500  # legal_chunks.heading VARCHAR(500)
# Synthetic article_ref for generic preamble between TRANSITORIOS header and first ordinal.
TRANSITORIOS_PREAMBLE_ARTICLE_REF: str = "Transitorios"


@dataclass
class Chunk:
    """Single chunk of document content for indexing."""

    text: str
    chunk_no: int
    chunk_type: str
    article_ref: Optional[str]
    heading: str
    start_page: int
    end_page: int
    has_table: bool
    table_index: Optional[int]


@dataclass
class TextSegment:
    """Internal segment for legal structure tagging.

    article_ref is set for chunk_type 'article' (e.g. 'Artículo 5 Transitorio')
    and for chunk_type 'transitorio' when the segment is an ordinal transitorio
    (e.g. 'Artículo Primero Transitorio'). It is None for all other types.
    """

    text: str
    chunk_type: str
    article_ref: Optional[str]


# Regex patterns for legal structure (order = hierarchy)
# Use ROMAN_WITH_LOOKAHEAD to avoid "VI" matching in "VIGESIMO"
_STRICT_NUM = f"(?:{ROMAN_WITH_LOOKAHEAD}|\\d+[º°o]?)"
_FLEX_NUM = f"(?:{ARTICLE_ORDINAL_WORDS_PATTERN}|{ROMAN_WITH_LOOKAHEAD}|\\d+[º°o]?)"

_RE_BOOK = re.compile(rf"^[Ll]ibro\s+{_STRICT_NUM}[\.\sº°o\-]*", re.MULTILINE)
_RE_TITLE = re.compile(rf"^([Tt][íi]tulo|[Tt]ema)\s+({_STRICT_NUM})[\.\sº°o\-]*", re.MULTILINE | re.IGNORECASE)
_RE_CHAPTER = re.compile(rf"^[Cc]ap[íi]tulo\s+({_STRICT_NUM})[\.\sº°o\-]*", re.MULTILINE | re.IGNORECASE)
_RE_SECTION = re.compile(
    rf"^[Ss]ecci[óo]n\s+((?:{ROMAN_WITH_LOOKAHEAD}|\\d+(?:\\.\\d+)+))[\.\sº°o\-]*",
    re.MULTILINE | re.IGNORECASE,
)
_RE_ANNEX = re.compile(
    rf"^[Aa]nexo\s+(?:{_STRICT_NUM}|[A-Z])[\.\sº°o\-]*",
    re.MULTILINE | re.IGNORECASE,
)
# Order: (1) digit+suffix (194-N, 194-N-1) (2) 9o.-A/9o.A (3) digit only (4) ordinals/roman
# Trailing class excludes dash so "-" in 194-N stays in capture group
# Use explicit pattern build to avoid rf-string backslash escaping issues
# 9o.-A: letra debe ir seguida de . o espacio (evitar capturar "5.L" de "5. Los")
_ARTICLE_NUM = (
    r"(?:\d+[º°o]?(?:\.\s*Bis)?(?:\s*-\s*[A-Za-z0-9]+)+"  # digit + suffix (194-N-1)
    r"|\d+[º°o]?(?:\.[\s\-]*[A-Za-z](?=[.\s]|$))"  # 9o.-A, 9o.A (letra suelta A./B./C.)
    r"|\d+[º°o]?(?:\.\s*Bis)?"  # digit only
    rf"|{_FLEX_NUM})"
)
_RE_ARTICLE = re.compile(
    rf"^Art[íi]culo\s+({_ARTICLE_NUM})[\.\sº°o]*",
    re.MULTILINE | re.IGNORECASE,
)
_RE_RULE = re.compile(r"^[Rr]egla\s+\d+(?:\.\d+)+[\.\sº°o\-]*", re.MULTILINE)
_RE_NUMERAL = re.compile(r"^[Nn]umeral\s+\d+[\.\sº°o\-]*", re.MULTILINE)
_RE_FRACTION = re.compile(r"^[IVXivx]+[\.\-]+[\s]*", re.MULTILINE)
_RE_INCISO = re.compile(r"^[a-zA-Z]\)\s", re.MULTILINE)
# Header variants: "TRANSITORIOS", "TRANSITORIO", "Transitorio", "ARTÍCULOS TRANSITORIOS", etc.
_RE_TRANSITORIO = re.compile(r"^(TRANSITORIOS?|Transitorio)\s*$", re.MULTILINE)
_RE_TRANSITORIO_HEADER = re.compile(
    r"^(?:ART[ÍI]CULOS?\s+TRANSITORIOS?|TRANSITORIOS?|Transitorio)\s*$",
    re.IGNORECASE,
)
_RE_TRANSITORIO_ORDINAL = re.compile(
    r"^(PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|SÉPTIMO|OCTAVO|NOVENO|DÉCIMO"
    r"|Primero|Segundo|Tercero|Cuarto|Quinto|Sexto|Séptimo|Octavo|Noveno|Décimo)"
    r"[\.\-]+",
    re.MULTILINE,
)
_RE_TABLE = re.compile(r"^\[TABLE_\d+\]", re.MULTILINE)

# Artículo multi-línea: "Artículo" en línea previa, "5o.-" o "5o. " en línea actual
_RE_ARTICLE_NUM_CONTINUATION = re.compile(
    r"^(\d+[º°o]?)(?:\.-\s*|\.\s+)[A-Za-z]",
    re.MULTILINE | re.IGNORECASE,
)
_RE_PREV_IS_ARTICULO_LABEL = re.compile(r"^Art[íi]culo\s*$", re.IGNORECASE)

# Número de artículo solo (tras nota de reforma): "5o.- Tratándose..."
_RE_ARTICLE_NUM_AFTER_REFORM = re.compile(
    r"^(\d+[º°o]?)(?:\.-\s*|\.\s*)",
    re.MULTILINE | re.IGNORECASE,
)

# Filters: reform notes, pagination
_RE_REFORM_NOTE = re.compile(
    r"^(Art[íi]culo|P[áa]rrafo|Fracc[ií][óo]n|Secci[óo]n|Cap[íi]tulo|T[íi]tulo)\s+"
    r"(reformad[oa]|adicionad[oa]|derogad[oa])\s*\.?\s*DOF",
    re.IGNORECASE,
)
_RE_REFORMA_DOF = re.compile(r"^Reforma\s+DOF", re.IGNORECASE)
_RE_PAGINATION = re.compile(r"^\d+\s+de\s+\d+$")
_RE_ARTICLE_INLINE_REF = re.compile(
    r"^de (esta|la presente)\s+(Ley|ley)|^del presente|^de este (Código|código|ordenamiento|cuerpo)",
    re.IGNORECASE,
)
_RE_TRANSITORIO_CITATION = re.compile(r"^Transitori[oa]\s+(del|de la|de los|de las)\s", re.IGNORECASE)
_RE_EDITORIAL_NOTE = re.compile(
    r"^(?:ACLARACI[OÓ]N|Nota\s*de\s+erratas|Fe\s+de\s+erratas|N\.\s*de\s+E\."
    r"|NOTA\s+DEL\s+EDITOR|Nota\s*:\s)",
    re.IGNORECASE,
)
_RE_BODY_PHRASE = re.compile(
    r"^(con el |con la |con el factor |de este |de la presente |de la |de los |de las |"
    r"de la ley |se considera|se señalan|se refiere|partir del |partir de la |partir de los |"
    r"partir de |del cap[íi]tulo |del t[íi]tulo |del c[oó]digo |primera del |segunda del |"
    r"tercera del |cuarta del |quinta del |sexta del |s[eé]ptima del |octava del |novena del |"
    r"d[eé]cima del |que incluye |incluye los art[íi]culos |, del |, de la |, de los |, de las |"
    r"del T[íi]tulo |de esta Ley[,.]?\s*|de la presente Ley[,.]?\s*|de este C[oó]digo[,.]?\s*|del\s)",
    re.IGNORECASE,
)
_RE_TEXT_BEFORE_INDICATES_BODY = re.compile(
    r"(?:La denominación del|denominación del|señalados en el|señaladas en el|señalado en el|"
    r"señalada en el|referido en el|referida en el|mencionado en el|mencionada en el|"
    r"indicado en el|indicada en el|establecido en el|establecida en el|previsto en el|"
    r"prevista en el|del cap[íi]tulo|del t[íi]tulo|del art[íi]culo|del libro|de la secci[óo]n|"
    r"permisibles\s+señalado|permisibles\s+señalada|establecidos|previstos|mencionados|referidos)",
    re.IGNORECASE,
)
_RE_PREV_ENDS_WITH_CONTINUATION = re.compile(r"\b(?:del|en el|de la|de los|de las|en la)\s*$", re.IGNORECASE)
_RE_PREV_IS_KEYWORD = re.compile(
    r"^(?:permisibles|establecidos|previstos|mencionados|referidos)\s*$",
    re.IGNORECASE,
)


def _is_valid_structural_number(num: str) -> bool:
    """Validate number format for title/chapter/section/book: roman or digits with optional ordinal."""
    if not num or not num.strip():
        return False
    return bool(re.match(r"^[IVX\d]+[º°o]?$", num.strip(), re.IGNORECASE))


def _is_valid_section_number(num: str) -> bool:
    """Validate section number: roman or dotted format (e.g. 1.1.1)."""
    if not num or not num.strip():
        return False
    s = num.strip()
    return bool(
        re.match(r"^[IVX]+[º°o]?$", s, re.IGNORECASE)
        or re.match(r"^\d+(?:\.\d+)+$", s)
    )


def normalize_article_number(raw: str) -> str:
    """Normalize article number: 1o/1º->1, preserve 14-A, 3 Bis, ordinal words."""
    if not raw or not isinstance(raw, str):
        return ""
    trimmed = raw.strip()
    if not trimmed:
        return ""
    bis_match = re.match(r"^(\d+)[º°o]?\.?\s*Bis$", trimmed, re.IGNORECASE)
    if bis_match:
        return f"{bis_match.group(1)} Bis"
    ordinal_dot_letter = re.match(r"^(\d+)[º°o]?\.?\s*-\s*([A-Za-z])", trimmed, re.IGNORECASE)
    if ordinal_dot_letter:
        return f"{ordinal_dot_letter.group(1)}-{ordinal_dot_letter.group(2)}"
    if re.search(r"\d+(-\s*[A-Za-z0-9]+)+$", trimmed):
        return trimmed
    digit_ordinal = re.sub(r"^(\d+)[º°o]\s*$", r"\1", trimmed, flags=re.IGNORECASE).strip()
    return digit_ordinal if digit_ordinal != trimmed else trimmed


def _extract_article_ref(match: re.Match[str]) -> str:
    """Build article_ref from Article regex match: Artículo 5o. or Artículo 10."""
    raw_num = match.group(1).strip()
    if not raw_num:
        return "Artículo"
    normalized = normalize_article_number(raw_num)
    return f"Artículo {normalized}"


def _is_transitorio_header(line: str) -> bool:
    """Return True if line is a transitorio block header (e.g. TRANSITORIOS, ARTÍCULOS TRANSITORIOS)."""
    return bool(_RE_TRANSITORIO_HEADER.match(line.strip()))


def _normalize_transitorio_ordinal(raw: str) -> str:
    """Return ordinal word in Title Case for display in headings."""
    return raw.strip().capitalize()



def _is_reform_note(line: str) -> bool:
    return bool(_RE_REFORM_NOTE.search(line) or _RE_REFORMA_DOF.search(line) or _RE_PAGINATION.search(line))


def _is_reform_note_context(line: str) -> bool:
    """True si la línea parece nota de reforma (para contexto de línea previa)."""
    if not line or len(line.strip()) < 10:
        return False
    return bool(
        re.search(r"(?:Reforma|derog[oó]|adicion[oó]|reformad[oa])\s+.*DOF", line, re.I)
        or re.search(r"\d{4}\)\s+Reforma\s+DOF", line, re.I)
        or re.search(r"\d{2}-\d{2}-\d{4}\s*:?\s*(?:Derogó|Adicionó|Reformó)", line, re.I)
    )


def _classify_line(
    line: str, prev_line: str = "", next_line: str = ""
) -> tuple[str, Optional[str]]:
    """Return (chunk_type, article_ref) for a line. article_ref only for article type."""
    trimmed = line.strip()
    if len(trimmed) < 5:
        return ("generic", None)

    if _is_reform_note(trimmed):
        return ("generic", None)

    if _RE_EDITORIAL_NOTE.match(trimmed):
        return ("generic", None)

    if _RE_BOOK.match(trimmed):
        m = _RE_BOOK.match(trimmed)
        if m:
            after = trimmed[m.end() :].strip()
            num_part = after.split()[0] if after else ""
            if not _is_valid_structural_number(num_part):
                return ("generic", None)
        return ("book", None)
    if _RE_TITLE.match(trimmed):
        m = _RE_TITLE.match(trimmed)
        if m:
            num_part = m.group(2).strip() if m.lastindex >= 2 else ""
            if not _is_valid_structural_number(num_part):
                return ("generic", None)
        rest = trimmed[m.end() :].strip() if m else ""
        if _RE_BODY_PHRASE.search(rest) or (rest and re.search(r"^del\s", rest, re.I)):
            return ("generic", None)
        if _RE_TEXT_BEFORE_INDICATES_BODY.search(prev_line):
            return ("generic", None)
        if _RE_PREV_ENDS_WITH_CONTINUATION.search(prev_line):
            return ("generic", None)
        if _RE_PREV_IS_KEYWORD.search(prev_line) and rest and re.search(r"señalado en el|señalada en el", rest, re.I):
            return ("generic", None)
        if not rest and next_line:
            if _RE_BODY_PHRASE.search(next_line) or re.search(r"^del\s|^de esta Ley|^de la presente Ley", next_line, re.I):
                return ("generic", None)
        return ("title", None)
    if _RE_CHAPTER.match(trimmed):
        m = _RE_CHAPTER.match(trimmed)
        if m:
            num_part = m.group(1).strip() if m.lastindex >= 1 else ""
            if not _is_valid_structural_number(num_part):
                return ("generic", None)
        rest = trimmed[m.end() :].strip() if m else ""
        if _RE_BODY_PHRASE.search(rest) or (rest and re.search(r"^del\s", rest, re.I)):
            return ("generic", None)
        if _RE_TEXT_BEFORE_INDICATES_BODY.search(prev_line):
            return ("generic", None)
        if _RE_PREV_ENDS_WITH_CONTINUATION.search(prev_line):
            return ("generic", None)
        if _RE_PREV_IS_KEYWORD.search(prev_line) and rest and re.search(r"señalado en el|señalada en el", rest, re.I):
            return ("generic", None)
        if not rest and next_line:
            if _RE_BODY_PHRASE.search(next_line) or re.search(r"^del\s|^de esta Ley|^de la presente Ley", next_line, re.I):
                return ("generic", None)
        return ("chapter", None)
    if _RE_SECTION.match(trimmed):
        m = _RE_SECTION.match(trimmed)
        if m:
            num_part = m.group(1).strip() if m.lastindex >= 1 else ""
            if not _is_valid_section_number(num_part):
                return ("generic", None)
        return ("section", None)
    if _RE_ANNEX.match(trimmed):
        return ("annex", None)
    m = _RE_ARTICLE.match(trimmed)
    if m:
        raw_num = m.group(1).strip()
        if not raw_num or re.search(r"^[uú]nico$", raw_num, re.I):
            return ("generic", None)
        rest = trimmed[m.end() :].strip() if m else ""
        starts_lower = trimmed.startswith("artículo")
        prev_ends_sentence = bool(re.search(r"[.;:]$", prev_line)) or prev_line == ""
        prev_is_reform_note = bool(
            re.search(r"(?:Reforma|derog[oó]|adicion[oó]|reformad[oa])\s+.*DOF", prev_line, re.I)
            or re.search(r"\d{2}-\d{2}-\d{4}\s*$", prev_line)
        )
        if starts_lower and not prev_ends_sentence and not prev_is_reform_note:
            return ("generic", None)
        if rest and _RE_ARTICLE_INLINE_REF.search(rest):
            return ("generic", None)
        if rest and _RE_TRANSITORIO_CITATION.search(rest):
            return ("generic", None)
        return ("article", _extract_article_ref(m))
    if _RE_RULE.match(trimmed):
        return ("rule", None)
    if _RE_NUMERAL.match(trimmed):
        return ("numeral", None)
    if _RE_FRACTION.match(trimmed):
        return ("fraction", None)
    if _RE_INCISO.match(trimmed):
        return ("inciso", None)
    if _is_transitorio_header(trimmed):
        return ("transitorio", None)
    if _RE_TRANSITORIO_ORDINAL.match(trimmed):
        if prev_line and not re.search(r"[\.:;]\s*$", prev_line):
            return ("generic", None)
        return ("transitorio", None)
    if _RE_TABLE.match(trimmed):
        return ("table", None)

    # Artículo multi-línea: prev="Artículo", curr="5o.- Tratándose..."
    m_cont = _RE_ARTICLE_NUM_CONTINUATION.match(trimmed)
    if m_cont and _RE_PREV_IS_ARTICULO_LABEL.match(prev_line.strip()):
        raw_num = m_cont.group(1).strip()
        if raw_num and not re.search(r"^[uú]nico$", raw_num, re.I):
            normalized = normalize_article_number(raw_num)
            return ("article", f"Art. {normalized}.")

    # Tras nota de reforma: curr="5o.- Tratándose..." cuando prev es nota de reforma
    m_reform = _RE_ARTICLE_NUM_AFTER_REFORM.match(trimmed)
    if m_reform and _is_reform_note_context(prev_line):
        raw_num = m_reform.group(1).strip()
        if raw_num and not re.search(r"^[uú]nico$", raw_num, re.I):
            normalized = normalize_article_number(raw_num)
            return ("article", f"Art. {normalized}.")

    return ("generic", None)


# Línea combinada: reform note + "Artículo N" en la misma línea
_RE_ARTICLE_IN_LINE = re.compile(
    rf"(Art[íi]culo\s+{_ARTICLE_NUM}[\.\sº°o\-]*)",
    re.IGNORECASE,
)


def _split_combined_reform_article_lines(lines: list[str]) -> list[str]:
    """
    Divide líneas que contienen nota de reforma + Artículo N en la misma línea.
    Ej: "1989) Reforma DOF... Artículo 5o.- Tratándose..." -> ["1989) Reforma...", "Artículo 5o.- Tratándose..."]
    """
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if len(stripped) < 30:
            result.append(line)
            continue
        if not _is_reform_note_context(stripped):
            result.append(line)
            continue
        match = _RE_ARTICLE_IN_LINE.search(stripped)
        if not match:
            result.append(line)
            continue
        # Dividir en la posición del artículo
        start = match.start()
        if start > 20:  # Solo si hay contenido de reforma significativo antes
            reform_part = stripped[:start].rstrip()
            article_part = stripped[start:].strip()
            if reform_part and article_part:
                result.append(reform_part)
                result.append(article_part)
                continue
        result.append(line)
    return result



# Line with transitorio header + "Artículo N" fused (PDF layout): split so chunker detects article.
# Handles "TRANSITORIOS Artículo 1º.-" and "ARTÍCULOS TRANSITORIOS Artículo 1º.-".
_RE_TRANSITORIOS_THEN_ARTICLE = re.compile(
    rf"^(ART[ÍI]CULOS?\s+TRANSITORIOS?|TRANSITORIOS?|Transitorio)\s+(Art[íi]culo\s+{_ARTICLE_NUM}[\.\sº°o\-]*)",
    re.IGNORECASE,
)


def _split_transitorios_article_lines(lines: list[str]) -> list[str]:
    """
    Split lines where "TRANSITORIOS" and "Artículo N" are fused on the same line.
    E.g. "TRANSITORIOS Artículo 1º.- Texto..." -> ["TRANSITORIOS", "Artículo 1º.- Texto..."]
    Ensures articles inside transitorios are detected as structural markers.
    """
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if len(stripped) < 15:
            result.append(line)
            continue
        match = _RE_TRANSITORIOS_THEN_ARTICLE.match(stripped)
        if not match:
            result.append(line)
            continue
        transitorios_part = match.group(1).strip()
        article_part = match.group(2).strip()
        rest = stripped[match.end() :].strip()
        result.append(transitorios_part)
        result.append(article_part + (" " + rest if rest else ""))
    return result


def split_by_legal_structure(
    text: str,
    leading_context_line: str = "",
    carry_inside_transitorios: bool = False,
) -> tuple[list[TextSegment], bool]:
    """
    Split text into segments tagged by legal structure.

    Each segment gets chunk_type and article_ref from its first structural line.
    Consecutive lines without structure inherit from previous segment (generic).

    leading_context_line: last line from previous page; used as prev_line for
    the first line when article header is split across pages (e.g. "Artículo"
    at end of page, "5o.-" at start of next).

    carry_inside_transitorios: transitorio block state carried from the previous
    page so that numeric articles on subsequent pages still receive the " Transitorio"
    suffix even when the TRANSITORIOS header appeared on an earlier page.

    Returns (segments, inside_transitorios_at_end) so the caller can propagate
    the state to the next invocation.
    """
    if not text or not text.strip():
        return [], carry_inside_transitorios

    segments: list[TextSegment] = []
    raw_lines = text.splitlines()
    lines = _split_combined_reform_article_lines(raw_lines)
    lines = _split_transitorios_article_lines(lines)
    current_text: list[str] = []
    current_type = "generic"
    current_ref: Optional[str] = None
    inside_transitorios = carry_inside_transitorios

    for i, line in enumerate(lines):
        prev_line = (
            lines[i - 1].strip()
            if i > 0
            else (leading_context_line.strip() if leading_context_line else "")
        )
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
        chunk_type, article_ref = _classify_line(line, prev_line, next_line)

        # Reset transitorio block when a major structural section begins (book/title/chapter/section).
        # This handles codes with multiple sets of transitory articles at different points.
        if inside_transitorios and chunk_type in ("book", "title", "chapter", "section"):
            inside_transitorios = False

        # Transitorio header line (e.g. "TRANSITORIOS", "ARTÍCULOS TRANSITORIOS"): activate block
        # state and emit a container segment so DB has a chunkType='transitorio' entry without
        # articleRef. That container chunk allows legalOutlineService to group children under it.
        if chunk_type == "transitorio" and _is_transitorio_header(line.strip()):
            inside_transitorios = True
            if current_text:
                seg_text = "\n".join(current_text).strip()
                if seg_text:
                    segments.append(
                        TextSegment(
                            text=seg_text,
                            chunk_type=current_type,
                            article_ref=current_ref,
                        )
                    )
            # Emit container segment (article_ref=None) so the TOC can show a group header.
            segments.append(
                TextSegment(
                    text=line.strip(),
                    chunk_type="transitorio",
                    article_ref=None,
                )
            )
            current_text = []
            current_type = "generic"
            current_ref = None
            continue

        # Ordinal transitorio (Primero.-, Segundo.-, ...): build synthetic heading ref.
        if chunk_type == "transitorio":
            m_ord = _RE_TRANSITORIO_ORDINAL.match(line.strip())
            if m_ord:
                inside_transitorios = True
                ordinal = _normalize_transitorio_ordinal(m_ord.group(1))
                article_ref = f"Artículo {ordinal} Transitorio"

        # Articles inside the transitorio section: append suffix to distinguish from body.
        if inside_transitorios and chunk_type == "article" and article_ref:
            article_ref = f"{article_ref} Transitorio"

        if chunk_type != "generic":
            if current_text:
                seg_text = "\n".join(current_text).strip()
                if seg_text:
                    segments.append(
                        TextSegment(
                            text=seg_text,
                            chunk_type=current_type,
                            article_ref=current_ref,
                        )
                    )
            current_text = [line]
            current_type = chunk_type
            current_ref = article_ref
        else:
            if current_text:
                current_text.append(line)
            else:
                current_text = [line]

    if current_text:
        seg_text = "\n".join(current_text).strip()
        if seg_text:
            segments.append(
                TextSegment(
                    text=seg_text,
                    chunk_type=current_type,
                    article_ref=current_ref,
                )
            )

    return segments, inside_transitorios


def _first_nonempty_line(text: str) -> str:
    """Return first non-empty line of text, stripped."""
    for line in text.splitlines():
        s = line.strip()
        if s:
            return s
    return ""


def _last_nonempty_line(text: str) -> str:
    """Return last non-empty line of text, stripped."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else ""


def _safe_heading(raw: str, max_len: int = MAX_HEADING_LENGTH) -> str:
    """Return raw trimmed and truncated to max_len. Empty string if raw is empty."""
    s = (raw or "").strip()
    return s[:max_len] if s else ""


_SPLIT_AVOID_TAIL_TOKENS = frozenset(
    {"se", "le", "les", "nos", "me", "te", "lo", "los", "las"}
)


def _adjust_split_for_clitic_tail(
    remaining: str,
    split_at: int,
    max_chars: int,
) -> int:
    """
    If the chunk would end on a lone clitic/pronoun, move split to the previous space
    in the window so the token stays with the following part.
    """
    min_space_pos = max_chars // 2
    part = remaining[:split_at].rstrip()
    last_space_in_part = part.rfind(" ")
    if last_space_in_part < 0:
        return split_at
    tail = part[last_space_in_part + 1 :]
    token = re.sub(r"^[^\wáéíóúñÁÉÍÓÚÑ]+|[^\wáéíóúñÁÉÍÓÚÑ]+$", "", tail, flags=re.IGNORECASE).lower()
    if token not in _SPLIT_AVOID_TAIL_TOKENS:
        return split_at
    prev_space = part.rfind(" ", 0, last_space_in_part)
    if prev_space < min_space_pos:
        return split_at
    return prev_space + 1


def _split_by_size(text: str, max_chars: int) -> list[str]:
    """
    Split text into parts not exceeding max_chars.
    Prefer split by double newline, then single newline, then length.
    Parts smaller than MIN_CHUNK_SIZE are merged with the previous part.
    """
    if len(text) <= max_chars:
        return [text] if text.strip() else []

    parts: list[str] = []
    remaining = text

    while remaining and len(remaining) > max_chars:
        chunk = remaining[: max_chars + 1]
        last_double = chunk.rfind("\n\n")
        last_single = chunk.rfind("\n")

        split_by_newline = False
        if last_double >= 0:
            split_at = last_double + 2
            split_by_newline = True
        elif last_single >= 0:
            split_at = last_single + 1
            split_by_newline = True
        else:
            window = remaining[:max_chars]
            last_space = window.rfind(" ")
            # Prefer ". " + uppercase (sentence end) when the window has no newline,
            # to avoid splitting mid-clause in long merged paragraphs (e.g. DOF decrees).
            split_at_sentence = -1
            search_hi = len(window)
            min_pos = max_chars // 2
            while search_hi > min_pos:
                dot_idx = window.rfind(". ", min_pos, search_hi)
                if dot_idx < 0:
                    break
                after = dot_idx + 2
                if after < len(remaining) and remaining[after].isupper():
                    split_at_sentence = after
                    break
                search_hi = dot_idx
            if split_at_sentence > min_pos:
                split_at = split_at_sentence
                split_by_newline = True
            elif last_space > max_chars // 2:
                split_at = last_space + 1
            else:
                split_at = max_chars

        if not split_by_newline:
            split_at = _adjust_split_for_clitic_tail(remaining, split_at, max_chars)

        part = remaining[:split_at].rstrip()
        if part:
            if len(part.strip()) >= MIN_CHUNK_SIZE:
                parts.append(part)
            elif parts:
                parts[-1] = parts[-1] + "\n" + part
            else:
                parts.append(part)
        remaining = remaining[split_at:].lstrip()

    if remaining.strip():
        last_part = remaining.strip()
        if len(last_part) >= MIN_CHUNK_SIZE:
            parts.append(last_part)
        elif parts:
            parts[-1] = parts[-1] + "\n" + last_part
        else:
            parts.append(last_part)

    return parts


def chunk_content(
    pages: list[PageContent],
    max_chunk_chars: int,
    document_title: str,
    text_head_sample: str,
) -> list[Chunk]:
    """
    Convert pages into chunks with deterministic legal-aware chunking.

    Tables are never split; each table is a single atomic chunk.
    Text is split by max_chunk_chars, preferring paragraph boundaries.
    document_title and text_head_sample enable decree-specific generic headings.
    """
    t0 = time.perf_counter()
    is_decreto = is_decreto_context(document_title, text_head_sample)

    def resolved_generic_heading(part: str) -> str:
        return heading_for_generic_chunk(part, document_title, is_decreto)

    chunks: list[Chunk] = []
    buffer_text = ""
    buffer_type = "generic"
    buffer_ref: Optional[str] = None
    buffer_start_page = 1
    buffer_end_page = 1

    prev_page_last_line = ""
    carry_inside_transitorios = False
    transitorios_preamble_ref: Optional[str] = None
    for page in pages:
        page_num = page.page_number

        segments, carry_inside_transitorios = split_by_legal_structure(
            page.text,
            leading_context_line=prev_page_last_line,
            carry_inside_transitorios=carry_inside_transitorios,
        )
        for segment in segments:
            to_add = segment.text
            seg_type = segment.chunk_type
            seg_ref = segment.article_ref

            # Flush buffer when new article starts (different article_ref) so each article
            # gets its own chunk for the index/TOC
            if (
                seg_type == "article"
                and seg_ref is not None
                and seg_ref != buffer_ref
                and buffer_text.strip()
            ):
                sub_parts = _split_by_size(buffer_text, max_chunk_chars)
                for part in sub_parts:
                    chunks.append(
                        Chunk(
                            text=part,
                            chunk_no=len(chunks) + 1,
                            chunk_type=buffer_type,
                            article_ref=buffer_ref,
                            heading=_safe_heading(buffer_ref or resolved_generic_heading(part)),
                            start_page=buffer_start_page,
                            end_page=buffer_end_page,
                            has_table=False,
                            table_index=None,
                        )
                    )
                buffer_text = ""
                buffer_type = "generic"
                buffer_ref = None

            # Flush when new transitorio segment (Primero.-, Segundo.-, or TRANSITORIOS header)
            # so each transitorio gets its own chunk instead of merging into one
            if (
                seg_type == "transitorio"
                and buffer_text.strip()
            ):
                sub_parts = _split_by_size(buffer_text, max_chunk_chars)
                for part in sub_parts:
                    chunks.append(
                        Chunk(
                            text=part,
                            chunk_no=len(chunks) + 1,
                            chunk_type=buffer_type,
                            article_ref=buffer_ref,
                            heading=_safe_heading(buffer_ref or resolved_generic_heading(part)),
                            start_page=buffer_start_page,
                            end_page=buffer_end_page,
                            has_table=False,
                            table_index=None,
                        )
                    )
                buffer_text = ""
                buffer_type = "generic"
                buffer_ref = None

            if len(buffer_text) + len(to_add) + 1 > max_chunk_chars:
                if buffer_text.strip():
                    sub_parts = _split_by_size(buffer_text, max_chunk_chars)
                    for part in sub_parts:
                        chunks.append(
                            Chunk(
                                text=part,
                                chunk_no=len(chunks) + 1,
                                chunk_type=buffer_type,
                                article_ref=buffer_ref,
                                heading=_safe_heading(buffer_ref or resolved_generic_heading(part)),
                                start_page=buffer_start_page,
                                end_page=buffer_end_page,
                                has_table=False,
                                table_index=None,
                            )
                        )
                buffer_text = to_add
                buffer_type = seg_type
                if seg_ref is not None:
                    buffer_ref = seg_ref
                elif (
                    seg_type == "generic"
                    and transitorios_preamble_ref is not None
                ):
                    buffer_ref = transitorios_preamble_ref
                buffer_start_page = page_num
                buffer_end_page = page_num
            else:
                if buffer_text:
                    buffer_text += "\n" + to_add
                    if (
                        buffer_ref is None
                        and seg_type == "generic"
                        and transitorios_preamble_ref is not None
                    ):
                        buffer_ref = transitorios_preamble_ref
                else:
                    buffer_text = to_add
                    buffer_type = seg_type
                    if seg_ref is not None:
                        buffer_ref = seg_ref
                    elif (
                        seg_type == "generic"
                        and transitorios_preamble_ref is not None
                    ):
                        buffer_ref = transitorios_preamble_ref
                    buffer_start_page = page_num
                buffer_end_page = page_num

            header_stripped = to_add.strip()
            if (
                seg_type == "transitorio"
                and seg_ref is None
                and _is_transitorio_header(header_stripped)
            ):
                transitorios_preamble_ref = TRANSITORIOS_PREAMBLE_ARTICLE_REF
            elif seg_type == "transitorio" and seg_ref is not None:
                transitorios_preamble_ref = None
            elif (
                seg_type == "article"
                and seg_ref is not None
                and seg_ref.endswith(" Transitorio")
            ):
                transitorios_preamble_ref = None
            elif seg_type in ("book", "title", "chapter", "section"):
                transitorios_preamble_ref = None

        for table in page.tables:
            if buffer_text.strip():
                sub_parts = _split_by_size(buffer_text, max_chunk_chars)
                for part in sub_parts:
                    chunks.append(
                        Chunk(
                            text=part,
                            chunk_no=len(chunks) + 1,
                            chunk_type=buffer_type,
                            article_ref=buffer_ref,
                            heading=_safe_heading(buffer_ref or resolved_generic_heading(part)),
                            start_page=buffer_start_page,
                            end_page=buffer_end_page,
                            has_table=False,
                            table_index=None,
                        )
                    )
                buffer_text = ""
                buffer_type = "generic"
                buffer_ref = None

            table_article_ref = detect_article_context(chunks)
            table_chunk_type = "boxed_note" if table.is_boxed_note else "table"
            chunks.append(
                Chunk(
                    text=table.markdown,
                    chunk_no=len(chunks) + 1,
                    chunk_type=table_chunk_type,
                    article_ref=table_article_ref,
                    heading=_safe_heading(table_article_ref or f"[TABLE_{table.table_index}]"),
                    start_page=table.page_number,
                    end_page=table.page_number,
                    has_table=True,
                    table_index=table.table_index,
                )
            )

        prev_page_last_line = _last_nonempty_line(page.text)

    if buffer_text.strip():
        sub_parts = _split_by_size(buffer_text, max_chunk_chars)
        for part in sub_parts:
            chunks.append(
                Chunk(
                    text=part,
                    chunk_no=len(chunks) + 1,
                    chunk_type=buffer_type,
                    article_ref=buffer_ref,
                    heading=_safe_heading(buffer_ref or resolved_generic_heading(part)),
                    start_page=buffer_start_page,
                    end_page=buffer_end_page,
                    has_table=False,
                    table_index=None,
                )
            )

    elapsed = time.perf_counter() - t0
    type_counts: dict[str, int] = {}
    for c in chunks:
        type_counts[c.chunk_type] = type_counts.get(c.chunk_type, 0) + 1
    logger.debug(
        "Chunking done: %d chunks en %.2fs — articles=%d transitorios=%d tables=%d generic=%d (max_chars=%d)",
        len(chunks),
        elapsed,
        type_counts.get("article", 0),
        type_counts.get("transitorio", 0),
        type_counts.get("table", 0),
        type_counts.get("generic", 0),
        max_chunk_chars,
    )
    return chunks


def detect_article_context(chunks: list[Chunk]) -> Optional[str]:
    """
    Find the last chunk with article_ref not None, considering both 'article'
    and 'transitorio' chunk types (ordinal transitorios also carry article_ref).
    Return that article_ref, or None if none found.
    """
    for c in reversed(chunks):
        if c.chunk_type in ("article", "transitorio") and c.article_ref is not None:
            return c.article_ref
    return None
