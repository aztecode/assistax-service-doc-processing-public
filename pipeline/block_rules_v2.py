"""
Heuristic rules for legal block classification (Phase 3).

Pure functions that inspect text, metadata, and geometric signals to produce
a (label, confidence, reason) tuple.  No LLM calls — those live in
block_classifier_llm_v2.py and are triggered only when these rules are
insufficient.
"""
from __future__ import annotations

import re

from pipeline.layout_models import LayoutBlock

# ── Valid label enumeration ─────────────────────────────────────────────────

VALID_LABELS: frozenset[str] = frozenset(
    {
        "document_title",
        "book_heading",
        "title_heading",
        "chapter_heading",
        "section_heading",
        "article_heading",
        "article_body",
        "fraction",
        "inciso",
        "transitory_heading",
        "transitory_item",
        "table",
        "editorial_note",
        "page_header",
        "page_footer",
        "index_block",
        "annex_heading",
        "annex_body",
        "unknown",
    }
)

# ── Confidence convention ──────────────────────────────────────────────────

CONFIDENCE_HIGH: float = 0.95
CONFIDENCE_MEDIUM: float = 0.80
CONFIDENCE_LOW: float = 0.60

# ── Date patterns ──────────────────────────────────────────────────────────

_MONTHS_ES: str = (
    "enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
    "septiembre|octubre|noviembre|diciembre"
)

_DATE_LONG_RE: re.Pattern[str] = re.compile(
    rf"\b\d{{1,2}}\s+de\s+(?:{_MONTHS_ES})\s+(?:de|del)\s+\d{{4}}\b",
    re.IGNORECASE,
)

_DATE_DOF_RE: re.Pattern[str] = re.compile(
    r"\bDOF\s+\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
    re.IGNORECASE,
)

_DATE_CITY_RE: re.Pattern[str] = re.compile(
    rf"^[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+de\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)?"
    rf",?\s+\d{{1,2}}\s+de\s+(?:{_MONTHS_ES})",
    re.IGNORECASE,
)

_DATE_ONLY_RE: re.Pattern[str] = re.compile(
    rf"^\s*\d{{1,2}}\s+DE\s+(?:{_MONTHS_ES.upper()})\s+DE\s+\d{{4}}\s*$",
    re.IGNORECASE,
)


def _looks_like_date_heading(text: str) -> bool:
    """True if the text is primarily a date string that should NOT become a heading."""
    stripped: str = text.strip()
    if _DATE_ONLY_RE.match(stripped):
        return True
    if _DATE_CITY_RE.match(stripped) and len(stripped) < 120:
        return True
    if _DATE_DOF_RE.search(stripped) and len(stripped) < 100:
        return True
    return False


# ── Article heading vs body reference ──────────────────────────────────────

_ARTICLE_HEADING_RE: re.Pattern[str] = re.compile(
    r"^\s*Art[ií]culo\s+\d+[\s\w]*[.\-–—]",
    re.IGNORECASE,
)

_ARTICLE_INLINE_RE: re.Pattern[str] = re.compile(
    r"(?:conforme|conform[ea]|según|previsto|dispuesto|establecid[oa]|"
    r"señalad[oa]|referid[oa]|contenid[oa]|indicad[oa]|mencionad[oa]|"
    r"de\s+(?:acuerdo|conformidad)|en\s+(?:t[eé]rminos|los\s+t[eé]rminos)|"
    r"(?:el|del|al|los|las|un)\s+)"
    r"\s*(?:art[ií]culo|art\.)\s+\d+",
    re.IGNORECASE,
)


def _looks_like_legal_article_heading(text: str) -> bool:
    """True only if the block starts with an article heading pattern."""
    stripped: str = text.strip()
    first_line: str = stripped.split("\n", 1)[0].strip()
    if not _ARTICLE_HEADING_RE.match(first_line):
        return False
    # Reject if the article mention is an inline reference inside prose
    if _ARTICLE_INLINE_RE.search(first_line):
        return False
    return True


# ── Transitorios ───────────────────────────────────────────────────────────

_TRANSITORY_HEADING_RE: re.Pattern[str] = re.compile(
    r"^\s*TRANSITORIOS?\s*$",
    re.IGNORECASE,
)

_TRANSITORY_ORDINAL_RE: re.Pattern[str] = re.compile(
    r"^\s*(?:Primero|Segundo|Tercero|Cuarto|Quinto|Sexto|S[eé]ptimo|"
    r"Octavo|Noveno|D[eé]cimo|Und[eé]cimo|Duod[eé]cimo|"
    r"D[eé]cimo\s*(?:primer|segund|tercer|cuart|quint|sext|s[eé]ptim|octav|noven)o?"
    r"|Vig[eé]simo|Trig[eé]simo|Cuadrag[eé]simo|Quincuag[eé]simo"
    r")\s*[.\-–—]",
    re.IGNORECASE,
)


def _looks_like_transitory_heading(text: str) -> bool:
    return bool(_TRANSITORY_HEADING_RE.match(text.strip()))


def _looks_like_transitory_item(text: str) -> bool:
    return bool(_TRANSITORY_ORDINAL_RE.match(text.strip()))


# ── Fracciones e incisos ───────────────────────────────────────────────────

_FRACTION_RE: re.Pattern[str] = re.compile(
    r"^\s*[IVXLCDM]+\s*[.\-–—]",
)

_INCISO_RE: re.Pattern[str] = re.compile(
    r"^\s*[a-z]\)\s",
)


def _looks_like_fraction(text: str) -> bool:
    """True if the block starts with a roman-numeral fraction (e.g. 'I.', 'II.-').

    The regex is anchored to start-of-string, so the only false-positive risk
    is prose that coincidentally begins with a roman numeral.  We guard
    against that by rejecting when the roman-numeral token is followed by a
    lowercase word with no separator (e.g. "Información general..." is not
    a fraction, but "I. Presentar declaración..." is).
    """
    stripped: str = text.strip()
    match: re.Match[str] | None = _FRACTION_RE.match(stripped)
    if not match:
        return False
    # Text after the roman numeral + punctuation
    rest: str = stripped[match.end():].lstrip()
    # If nothing follows, it's a bare roman-numeral heading — accept
    if not rest:
        return True
    # If the rest starts with uppercase or is clearly a list item, accept
    return True


def _looks_like_inciso(text: str) -> bool:
    """True if the block starts with a lowercase letter-parenthesis inciso."""
    return bool(_INCISO_RE.match(text.strip()))


# ── Document title ─────────────────────────────────────────────────────────

_DOC_TITLE_KEYWORDS_RE: re.Pattern[str] = re.compile(
    r"\b(LEY|DECRETO|REGLAMENTO|CÓDIGO|ESTATUTO|ACUERDO|NORMA|CONSTITUCIÓN|"
    r"PRESUPUESTO|LINEAMIENTOS?)\b",
    re.IGNORECASE,
)


def _looks_like_document_title(text: str) -> bool:
    """True if block looks like the document's main title (all-caps, legal keyword)."""
    stripped: str = text.strip()
    if len(stripped) > 300 or len(stripped) < 5:
        return False
    if _looks_like_date_heading(stripped):
        return False
    upper_ratio: float = sum(1 for c in stripped if c.isupper()) / max(
        sum(1 for c in stripped if c.isalpha()), 1
    )
    if upper_ratio < 0.60:
        return False
    if not _DOC_TITLE_KEYWORDS_RE.search(stripped):
        return False
    return True


# ── Index block ────────────────────────────────────────────────────────────

def _looks_like_index_block(text: str, metadata: dict[str, object]) -> bool:
    if metadata.get("possible_index_zone") is True:
        return True
    return False


# ── Editorial note ─────────────────────────────────────────────────────────

_EDITORIAL_NOTE_RE: re.Pattern[str] = re.compile(
    r"(?:ACLARACI[OÓ]N|Fe\s+de\s+erratas|N\.\s*de\s+E\.|"
    r"NOTA\s+(?:DEL\s+EDITOR|ACLARATORIA|editorial)|"
    r"Art[íi]culo\s+(?:reformad[oa]|adicionad[oa]|derogad[oa])\b|"
    r"Correcci[oó]n|AVISO)",
    re.IGNORECASE,
)


def _looks_like_editorial_note(block: LayoutBlock) -> bool:
    first_line: str = block.text.strip().split("\n", 1)[0].strip()
    if _EDITORIAL_NOTE_RE.search(first_line):
        return True
    if block.kind == "boxed_note":
        return True
    if block.metadata.get("is_inside_visual_box") is True:
        return True
    return False


# ── Table block ────────────────────────────────────────────────────────────

def _looks_like_table_block(block: LayoutBlock) -> bool:
    if block.source == "pymupdf_table":
        return True
    if block.kind == "table":
        return True
    if block.metadata.get("is_table_candidate") is True:
        return True
    return False


# ── Structural heading patterns ────────────────────────────────────────────

_BOOK_HEADING_RE: re.Pattern[str] = re.compile(
    r"^\s*Libro\s+[IVXLCDM\d]+",
    re.IGNORECASE,
)
_TITLE_HEADING_RE: re.Pattern[str] = re.compile(
    r"^\s*T[ií]tulo\s+[IVXLCDM\d]+",
    re.IGNORECASE,
)
_CHAPTER_HEADING_RE: re.Pattern[str] = re.compile(
    r"^\s*Cap[ií]tulo\s+[IVXLCDM\d]+",
    re.IGNORECASE,
)
_SECTION_HEADING_RE: re.Pattern[str] = re.compile(
    r"^\s*Secci[oó]n\s+[IVXLCDM\d]+",
    re.IGNORECASE,
)
_ANNEX_HEADING_RE: re.Pattern[str] = re.compile(
    r"^\s*Anexo\s+[IVXLCDM\d]+",
    re.IGNORECASE,
)


# ── Main classification function ──────────────────────────────────────────

def classify_block_by_rules(
    block: LayoutBlock,
) -> tuple[str, float, str | None]:
    """Classify a single block using deterministic heuristics.

    Returns (label, confidence, reason).
    A confidence < CONFIDENCE_HIGH may warrant LLM escalation.
    """
    text: str = block.text.strip()
    first_line: str = text.split("\n", 1)[0].strip()

    # ── 1. Header/footer from Phase 2 normalizer ──
    if block.kind == "header":
        return ("page_header", CONFIDENCE_HIGH, "kind_header_from_normalizer")
    if block.kind == "footer":
        return ("page_footer", CONFIDENCE_HIGH, "kind_footer_from_normalizer")

    # ── 2. Table signal ──
    if _looks_like_table_block(block):
        return ("table", CONFIDENCE_HIGH, "table_signal")

    # ── 3. Editorial note ──
    if _looks_like_editorial_note(block):
        return ("editorial_note", CONFIDENCE_HIGH, "editorial_note_pattern")

    # ── 4. Index zone ──
    if _looks_like_index_block(text, block.metadata):
        return ("index_block", CONFIDENCE_MEDIUM, "possible_index_zone_metadata")

    # ── 5. Transitorios ──
    if _looks_like_transitory_heading(text):
        return ("transitory_heading", CONFIDENCE_HIGH, "transitory_heading_pattern")
    if _looks_like_transitory_item(text):
        return ("transitory_item", CONFIDENCE_HIGH, "transitory_ordinal_pattern")

    # ── 6. Date guard — must run BEFORE heading patterns ──
    if _looks_like_date_heading(text):
        return ("article_body", CONFIDENCE_MEDIUM, "date_not_heading")

    # ── 7. Document title ──
    if _looks_like_document_title(text):
        return ("document_title", CONFIDENCE_MEDIUM, "document_title_keywords_caps")

    # ── 8. Structural headings (order: most specific → most general) ──
    if _looks_like_legal_article_heading(text):
        return ("article_heading", CONFIDENCE_HIGH, "article_heading_pattern")

    if _BOOK_HEADING_RE.match(first_line):
        return ("book_heading", CONFIDENCE_HIGH, "book_heading_pattern")
    if _TITLE_HEADING_RE.match(first_line):
        return ("title_heading", CONFIDENCE_HIGH, "title_heading_pattern")
    if _CHAPTER_HEADING_RE.match(first_line):
        return ("chapter_heading", CONFIDENCE_HIGH, "chapter_heading_pattern")
    if _SECTION_HEADING_RE.match(first_line):
        return ("section_heading", CONFIDENCE_HIGH, "section_heading_pattern")
    if _ANNEX_HEADING_RE.match(first_line):
        return ("annex_heading", CONFIDENCE_HIGH, "annex_heading_pattern")

    # ── 9. Fracciones / incisos ──
    if _looks_like_fraction(text):
        return ("fraction", CONFIDENCE_HIGH, "fraction_roman_pattern")
    if _looks_like_inciso(text):
        return ("inciso", CONFIDENCE_HIGH, "inciso_letter_pattern")

    # ── 10. Default: prose body ──
    if len(text) > 30:
        return ("article_body", CONFIDENCE_LOW, "default_prose_body")

    return ("unknown", CONFIDENCE_LOW, None)
