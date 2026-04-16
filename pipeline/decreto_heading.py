"""
Heading helpers for Mexican DOF-style decrees: skip masthead lines and long rubrics.
Pure functions; no I/O.
"""

from __future__ import annotations

import re

from pipeline.metadata_extractor import extract_law_name, infer_doc_type

TEXT_HEAD_SAMPLE_LIMIT: int = 4000
RUBRIC_LONG_MIN_CHARS: int = 120
RUBRIC_TRUNCATE_CHARS: int = 120
HEADING_FIRST_LINE_MAX_CHARS: int = 220

# First line of chunk that looks like mid-sentence continuation (not a section title).
# Keep aligned with assistax-front DocumentChunksContent isSentenceContinuationHeading.
_RE_SENTENCE_CONTINUATION_HEADING = re.compile(
    r"^(considera|consideró|consideran|deberá|deberán|podrá|podrán|tendrá|tendrán|"
    r"habrá|habrán|resulta|resultan|establece|establecen|determina|determinan|"
    r"dispone|disponen|autoriza|autorizan|ordena|ordenan|precisa|precisan|"
    r"señala|señalan|define|definen|otorga|otorgan|fundamento)\b\s+",
    re.IGNORECASE,
)

_RE_SPANISH_WEEKDAY_START = re.compile(
    r"^(?:Lunes|Martes|Miércoles|Miercoles|Jueves|Viernes|Sábado|Sabado|Domingo)\b",
    re.IGNORECASE,
)
_RE_SPANISH_DATE_INLINE = re.compile(
    r"\d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
    r"septiembre|octubre|noviembre|diciembre)\s+de\s+\d{4}",
    re.IGNORECASE,
)

_RE_HEAD_DECRETO = re.compile(r"\bDECRETO\s+por\s+el\s+que", re.IGNORECASE)
# First sentence of the decree rubric (ends at first period; avoids swallowing body text).
_RE_DECRETO_RUBRIC_SENTENCE = re.compile(
    r"DECRETO\s+por\s+el\s+que\s+[\s\S]*?\.",
    re.IGNORECASE,
)
_RE_PRESIDENCIA_DECRETO = re.compile(
    r"PRESIDENCIA\s+DE\s+LA\s+REP[ÚU]BLICA[\s\S]{0,1200}?\bDECRETO\b",
    re.IGNORECASE,
)
_RE_EDICION_VESPERTINA = re.compile(
    r"^\(?\s*Edici[oó]n\s+Vespertina\s*\)?\s*$",
    re.IGNORECASE,
)


def is_decreto_context(document_title: str, combined_text_head: str) -> bool:
    """
    True if the document should use decree heading heuristics.
    Uses upload title keywords and/or leading extracted text (rubric patterns).
    """
    if infer_doc_type(document_title) == "decreto":
        return True
    head = combined_text_head[:TEXT_HEAD_SAMPLE_LIMIT]
    if _RE_HEAD_DECRETO.search(head):
        return True
    if _RE_PRESIDENCIA_DECRETO.search(head):
        return True
    return False


def extract_decreto_por_el_que_rubric(text: str) -> str:
    """
    Return the decree rubric sentence (DECRETO por el que ….) if present in text.
    Normalizes internal whitespace for use as chunk heading.
    """
    if not text or not text.strip():
        return ""
    m = _RE_DECRETO_RUBRIC_SENTENCE.search(text)
    if not m:
        return ""
    s = re.sub(r"\s+", " ", m.group(0).strip())
    return s[:500] if len(s) > 500 else s


def _is_dof_masthead_line(line: str) -> bool:
    t = line.strip()
    if not t:
        return True
    if _RE_EDICION_VESPERTINA.match(t):
        return True
    if re.match(r"^DIARIO\s+OFICIAL\b", t, re.IGNORECASE):
        return True
    if re.match(r"^PODER\s+EJECUTIVO\s*$", t, re.IGNORECASE):
        return True
    return False


def _is_dof_date_que_preamble_line(line_stripped: str) -> bool:
    """
    DOF-style line: weekday + Spanish date + (optional) Edición Vespertina + considerando (Que …).
    Not a document section title; skip for chunk headings.
    """
    t = line_stripped.strip()
    if len(t) < 40:
        return False
    if not _RE_SPANISH_WEEKDAY_START.match(t):
        return False
    if not _RE_SPANISH_DATE_INLINE.search(t):
        return False
    if re.search(r"\bQue\s+", t):
        return True
    if re.search(r"Edici[oó]n\s+Vespertina", t, re.IGNORECASE):
        return True
    return False


def first_substantive_line_for_heading(text: str) -> str:
    """
    First non-empty line that is not a typical DOF masthead line.
    Falls back to first non-empty line if every line looks like masthead (defensive).
    """
    lines = text.splitlines()
    first_any = ""
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if not first_any:
            first_any = s
        if _is_dof_masthead_line(line):
            continue
        if _is_dof_date_que_preamble_line(s):
            continue
        return s
    if first_any and (
        _is_dof_masthead_line(first_any)
        or _is_dof_date_que_preamble_line(first_any.strip())
    ):
        return ""
    return first_any


def _looks_like_long_rubric(line: str) -> bool:
    if re.search(r"\bDECRETO\s+por\s+el\s+que", line, re.IGNORECASE):
        return len(line.strip()) >= RUBRIC_LONG_MIN_CHARS
    if re.search(r"\bDECRETO\b", line, re.IGNORECASE) and len(line.strip()) >= RUBRIC_LONG_MIN_CHARS:
        return True
    return False


def _truncate_at_word_boundary(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text.strip()
    cut = text[: max_len + 1]
    last_space = cut.rfind(" ")
    if last_space > max_len // 2:
        return cut[:last_space].strip() + "…"
    return text[:max_len].strip() + "…"


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if s:
            return s
    return ""


def _looks_like_sentence_continuation_heading(line: str) -> bool:
    s = line.strip()
    if not s or not s[0].islower():
        return False
    return bool(_RE_SENTENCE_CONTINUATION_HEADING.match(s))


def _heading_candidate_lines(part: str, is_decreto: bool) -> list[str]:
    """Non-empty lines, skipping DOF masthead/preamble when decree context."""
    out: list[str] = []
    for line in part.splitlines():
        s = line.strip()
        if not s:
            continue
        if is_decreto:
            if _is_dof_masthead_line(line):
                continue
            if _is_dof_date_que_preamble_line(s):
                continue
        out.append(s)
    return out


def _first_noncontinuation_heading_line(candidates: list[str]) -> str:
    for s in candidates:
        if not _looks_like_sentence_continuation_heading(s):
            return s
    return ""


def heading_for_generic_chunk(part: str, document_title: str, is_decreto: bool) -> str:
    """
    Resolve chunk heading for generic segments.
    Non-decreto: first non-empty line (parity with legal_chunker).
    Decreto: if the chunk contains a DECRETO por el que … sentence, use it as heading;
    else skip masthead and apply long-rubric / law-name fallbacks.
    Skips sentence-continuation first lines (e.g. 'considera adecuado…').
    """
    if is_decreto:
        rubric = extract_decreto_por_el_que_rubric(part)
        if rubric:
            return rubric
    candidates = _heading_candidate_lines(part, is_decreto)
    if not candidates:
        law = extract_law_name(document_title)
        return law if law else ""

    line = _first_noncontinuation_heading_line(candidates)
    if not line:
        law = extract_law_name(document_title)
        if law:
            return law
        return _truncate_at_word_boundary(candidates[0], RUBRIC_TRUNCATE_CHARS)

    if not is_decreto:
        return line

    if _is_dof_masthead_line(line):
        law = extract_law_name(document_title)
        return law if law else line
    if _looks_like_long_rubric(line):
        law = extract_law_name(document_title)
        if law:
            return law
        return _truncate_at_word_boundary(line, RUBRIC_TRUNCATE_CHARS)
    if len(line) > HEADING_FIRST_LINE_MAX_CHARS:
        law = extract_law_name(document_title)
        if law:
            return law
        return _truncate_at_word_boundary(line, RUBRIC_TRUNCATE_CHARS)
    return line
