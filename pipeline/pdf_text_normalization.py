"""
PDF text normalization: merges layout-induced line breaks while preserving
structural breaks (articles, chapters, lists, incisos) and repairing hyphenated word splits.
Ported from assistax-back/src/services/pdfTextNormalization.ts
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from pipeline.legal_ordinal_patterns import ARTICLE_ORDINAL_WORDS_PATTERN

MAX_EXAMPLES = 10


@dataclass
class NormalizationExample:
    prev_line: str
    next_line: str
    action: str  # 'merge' | 'preserve' | 'hyphen_join'
    rule: str


@dataclass
class NormalizationAudit:
    total_lines_raw: int
    total_lines_after: int
    merges_applied: int
    merge_ratio: float
    hyphen_joins: int
    preserved_breaks_by_rule: dict[str, int]
    headers_removed: int
    examples: list[NormalizationExample]


# Hard-stop: never merge before lines matching these (structural markers)
_PRESERVE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            rf"^(Artículo|ART[IÍ]CULO)\s+(\d+[º°o]?(?:\.\s*Bis)?(?:\s*-\s*[A-Za-z0-9]+)*\b|{ARTICLE_ORDINAL_WORDS_PATTERN})\b",
            re.IGNORECASE,
        ),
        "articulo",
    ),
    (
        re.compile(
            r"^Art\.\s*\d+(?:\.\s*Bis)?(?:\s*-\s*[A-Za-z0-9]+)*\b",
            re.IGNORECASE,
        ),
        "articulo_abbrev",
    ),
    (re.compile(r"^(Capítulo|Título|Sección)\b", re.IGNORECASE), "titulo_capitulo_seccion"),
    (re.compile(r"^(I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII)\.\s*", re.IGNORECASE), "romanos"),
    (re.compile(r"^[a-z]\)\s*(?:\.-|\.)?\s*", re.IGNORECASE), "inciso"),
    (re.compile(r"^\d+\.\s+"), "lista_numerica"),
    (re.compile(r"^(Fracción|Inciso)\s+", re.IGNORECASE), "fraccion_inciso"),
    (
        re.compile(r"^\$\s*[\d,]+\.?\d*\s+.*\$\s*[\d,]+"),
        "tabla_fila",
    ),
    (
        re.compile(r"^\d[\d,]*\.?\d*\s+[\d,]+\.?\d*\s+[\d,]+\.?\d*\s+[\d.]+"),
        "tabla_fila",
    ),
    (
        re.compile(r"^\d[\d,]*\.?\d*\s+En adelante\s+[\d,]+\.?\d*\s+[\d.]+", re.IGNORECASE),
        "tabla_fila",
    ),
    # Notas de reforma: no unir con la siguiente línea (evita fusionar con "Artículo N")
    (
        re.compile(
            r"(?:Reforma|derog[oó]|adicion[oó]|reformad[oa])\s+.*DOF|\d{4}\)\s+Reforma\s+DOF",
            re.IGNORECASE,
        ),
        "reform_note",
    ),
    # TRANSITORIOS: preserve break so "Artículo 1º.-" on next line is not merged.
    # Covers variants: "TRANSITORIOS", "TRANSITORIO", "Transitorio", "ARTÍCULOS TRANSITORIOS".
    (re.compile(r"^(?:ART[ÍI]CULOS?\s+TRANSITORIOS?|TRANSITORIOS?|Transitorio)\s*$", re.IGNORECASE), "transitorios"),
    # Transitorio ordinals (Primero.-, Segundo.-, etc.) so articles inside transitorios are detected
    (
        re.compile(
            r"^(PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|SÉPTIMO|OCTAVO|NOVENO|DÉCIMO"
            r"|Primero|Segundo|Tercero|Cuarto|Quinto|Sexto|Séptimo|Octavo|Noveno|Décimo)\s*[\.\-]+",
            re.IGNORECASE,
        ),
        "transitorio_ordinal",
    ),
]


def _matches_preservation_rule(line: str) -> Optional[str]:
    """Returns rule name if line should preserve break (not merge), None otherwise."""
    trimmed = line.strip()
    for pattern, rule in _PRESERVE_PATTERNS:
        if pattern.search(trimmed):
            return rule
    return None


_ALWAYS_REMOVE_PATTERNS = [
    re.compile(r"^\d+\s+de\s+\d+$"),
    re.compile(r"^Página\s+\d+", re.IGNORECASE),
]


def _should_always_remove(line: str) -> bool:
    trimmed = line.strip()
    return any(p.search(trimmed) for p in _ALWAYS_REMOVE_PATTERNS)


_HEADER_CANDIDATE_PATTERNS = [
    re.compile(r"^\d+\s+de\s+\d+$"),
    re.compile(r"^Página\s+\d+", re.IGNORECASE),
    re.compile(r"^[A-ZÁÉÍÓÚÑ\d\s(),.\-]{8,250}$"),
    re.compile(r"CÁMARA DE DIPUTADOS", re.IGNORECASE),
    re.compile(r"Secretaría General", re.IGNORECASE),
    re.compile(r"Secretaría de Servicios Parlamentarios", re.IGNORECASE),
    re.compile(r"Última Reforma DOF \d{2}-\d{2}-\d{4}", re.IGNORECASE),
]
# Leyendas de reforma: nunca eliminar por repetición (conservar al menos una)
_RE_REFORM_LEGEND = re.compile(
    r"Art[íi]culo\s+(reformad[oa]|adicionad[oa]|derogad[oa])\s*\.?\s*DOF",
    re.IGNORECASE,
)

HEADER_MAX_LENGTH = 250


def _is_header_candidate(line: str) -> bool:
    trimmed = line.strip()
    if len(trimmed) > HEADER_MAX_LENGTH:
        return False
    return any(p.search(trimmed) for p in _HEADER_CANDIDATE_PATTERNS)


def _normalize_header_line_for_counting(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


# B starts a new normative block; do not merge across blank line onto these.
_NORMATIVE_BLOCK_START = re.compile(
    r"^(CONSIDERANDO|EXPIDIENDO|EXHORTANDO|ART[ÍI]CULO|ART\.\s*\d|T[ÍI]TULO|CAP[ÍI]TULO|"
    r"SECCI[ÓO]N|TRANSITORIOS?|ÚNICO|VISTO|PUNTOS?\s+DE\s+ACUERDO|CÁMARA\s+DE\s+DIPUTADOS|"
    r"DECRETO\s+por\s+el\s+que)\b",
    re.IGNORECASE,
)

_RE_AL_MARGEN_LINE = re.compile(r"^Al\s+margen\b", re.IGNORECASE)
# Merge breaks for DOF layout only — excludes ^Artículo so "Artículo 14-" + "A." still merge.
_DOF_LINE_START_BREAK_MERGE = re.compile(
    r"^(CONSIDERANDO|EXPIDIENDO|EXHORTANDO|DECRETO\s+por\s+el\s+que|T[ÍI]TULO|CAP[ÍI]TULO|"
    r"SECCI[ÓO]N|TRANSITORIOS?|ÚNICO|VISTO|PUNTOS?\s+DE\s+ACUERDO|CÁMARA\s+DE\s+DIPUTADOS)\b",
    re.IGNORECASE,
)


def _is_dof_structural_line_start(s: str) -> bool:
    """
    Lines that must stay on their own row so DOF masthead, decree rubric, and
    marginal note are not merged into one string (breaks headings and chunking).
    Does not include Artículo lines (those use separate merge rules).
    """
    t = s.strip()
    if not t:
        return False
    if _DOF_LINE_START_BREAK_MERGE.match(t):
        return True
    if _RE_AL_MARGEN_LINE.match(t):
        return True
    return False

_RE_SINGLE_LETTER_TAIL = re.compile(r" [a-záéíóúñ]$", re.IGNORECASE)
# Comma + clitic/aux at line end: layout often splits before "considera…" on next line.
_RE_CLITIC_AFTER_COMMA_END = re.compile(
    r",\s*(se|le|les|nos|me|te|lo|los|las)\s*$",
    re.IGNORECASE,
)
# Verbal start of line = sentence continuation (lowercase first char enforced separately).
_RE_SENTENCE_VERB_CONTINUATION_START = re.compile(
    r"^(considera|consideró|consideran|deberá|deberán|podrá|podrán|tendrá|tendrán|"
    r"habrá|habrán|resulta|resultan|establece|establecen|determina|determinan|"
    r"dispone|disponen|autoriza|autorizan|ordena|ordenan|precisa|precisan|"
    r"señala|señalan|define|definen|otorga|otorgan|fundamento)\b\s+",
    re.IGNORECASE,
)


def _line_looks_mid_word_continuation_start(b_stripped: str) -> bool:
    """True if line starts as second half of a split word."""
    bl = b_stripped.lower()
    return bool(
        re.match(r"^culos\b", bl)
        or re.match(r"^on\b", bl)
        or re.match(r"^e\b", bl)
    )


def _line_looks_sentence_verb_continuation_start(b_stripped: str) -> bool:
    """Lowercase-first line starting with typical normative verb (continuation, not a heading)."""
    if not b_stripped or not b_stripped[0].islower():
        return False
    return bool(_RE_SENTENCE_VERB_CONTINUATION_START.match(b_stripped))


def _line_looks_cut_at_end(a_stripped: str) -> bool:
    """True if line likely ends mid-word or with syntactic cliff before a wrapped continuation."""
    low = a_stripped.lower().rstrip()
    if low.endswith("artí") or low.endswith("durant"):
        return True
    if _RE_CLITIC_AFTER_COMMA_END.search(a_stripped.rstrip()):
        return True
    return bool(_RE_SINGLE_LETTER_TAIL.search(a_stripped))


def _line_looks_cut_at_start(b_stripped: str) -> bool:
    """True if next line continues previous (split word or sentence continuation)."""
    return _line_looks_mid_word_continuation_start(
        b_stripped
    ) or _line_looks_sentence_verb_continuation_start(b_stripped)


def _should_collapse_blank_between(a: str, b: str) -> bool:
    """
    True if non-empty A, blank, non-empty B should be merged (DOF-style stray blank).
    Conservative: only when A or B shows known split-artifact patterns and merge_score fits.
    """
    a_st = a.strip()
    b_st = b.strip()
    if not a_st or not b_st:
        return False
    if _matches_preservation_rule(b_st) is not None:
        return False
    if _NORMATIVE_BLOCK_START.match(b_st):
        return False
    score = _compute_merge_score(a_st, b_st)
    if score < 2:
        return False
    if not (_line_looks_cut_at_end(a_st) or _line_looks_cut_at_start(b_st)):
        return False
    return True


def _join_collapsed_lines(a: str, b: str) -> str:
    """Join two lines collapsed across a stray blank; omit space for mid-word glue."""
    a_st = a.rstrip()
    b_st = b.strip()
    if not a_st:
        return b_st
    if not b_st:
        return a_st
    # Glue without space only for mid-word splits, not "…, se" + "considera …".
    if (
        a_st[-1].isalpha()
        and b_st[0].isalpha()
        and b_st[0].islower()
        and (
            a_st.lower().endswith("artí")
            or a_st.lower().endswith("durant")
            or bool(_RE_SINGLE_LETTER_TAIL.search(a_st))
            or _line_looks_mid_word_continuation_start(b_st)
        )
    ):
        return a_st + b_st
    return a_st + " " + b_st


def _collapse_stray_blank_lines(lines: list[str]) -> list[str]:
    """
    Remove a single empty line between A and B when B continues A (layout split).
    Does not collapse multiple consecutive blank lines (likely real paragraph gap).
    """
    if len(lines) < 3:
        return lines
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        if i + 2 < n:
            a, mid, b = lines[i], lines[i + 1], lines[i + 2]
            if a.strip() != "" and mid.strip() == "" and b.strip() != "":
                if _should_collapse_blank_between(a, b):
                    out.append(_join_collapsed_lines(a, b))
                    i += 3
                    continue
        out.append(lines[i])
        i += 1
    return out


def _compute_merge_score(prev_line: str, next_line: str) -> int:
    prev = prev_line.strip()
    next_ = next_line.strip()
    if not prev or not next_:
        return 0

    score = 0

    if not re.search(r"[.?!;:]$", prev):
        score += 2
    if re.search(r'[,("\']$', prev):
        score += 1
    first_char = next_[0] if next_ else ""
    if re.search(r"[a-záéíóúñü0-9]", first_char):
        score += 1
    if len(next_) > 20:
        score += 1
    is_all_caps = bool(re.search(r"^[^a-záéíóúñü]*$", next_) and re.search(r"[A-Z]", next_))
    if is_all_caps and (len(next_) < 60 or len(next_) < 30):
        score -= 2
    if len(next_) < 10 and not re.search(r"[.,;:?!-]", next_):
        score -= 2
    if re.search(r"[)\]]$", prev):
        score -= 1

    return score


def _pre_normalize(raw: str) -> list[str]:
    normalized = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    return [re.sub(r"\s{2,}", " ", line).rstrip() for line in lines]


# Encabezado de artículo: Artículo N, 9o, 9o., 14-A, etc.; opcionalmente termina en .- o .
_RE_ARTICLE_HEADING = re.compile(
    r"^Art[íi]culo\s+\d+[º°o]?(?:\.\s*Bis)?(?:\s*-\s*[A-Za-z0-9]+)*(?:\s*[.\-]*\s*)?$",
    re.IGNORECASE,
)
# Línea de una sola letra mayúscula + punto: "A.", "B."
_RE_ARTICLE_LETTER_LINE = re.compile(r"^[A-Z]\.\s*$")


def _merge_article_letter_lines(lines: list[str]) -> list[str]:
    """
    Une líneas "Artículo N.-" + "A." en una sola.
    Evita que la letra del artículo quede como párrafo suelto.
    """
    if len(lines) < 2:
        return lines
    result: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            prev_trimmed = line.strip()
            if (
                _RE_ARTICLE_LETTER_LINE.match(next_line)
                and _RE_ARTICLE_HEADING.search(prev_trimmed)
            ):
                merged = prev_trimmed.rstrip() + next_line.strip()
                result.append(merged)
                i += 2
                continue
            if (
                _RE_ARTICLE_LETTER_LINE.match(next_line)
                and re.search(r"Art[íi]culo\s+\d+[º°o]?-\s*$", prev_trimmed, re.I)
            ):
                merged = prev_trimmed.rstrip().rstrip("-").rstrip() + "-" + next_line.strip()
                result.append(merged)
                i += 2
                continue
        result.append(line)
        i += 1
    return result


def _post_normalize(text: str) -> str:
    out = re.sub(r"[ \t]+", " ", text)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    out = re.sub(r"\s*\d+\s+de\s+\d+\s*", " ", out)
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", out)).strip()


def normalize_pdf_text(
    raw_text: str,
    seen_headers: Optional[set[str]],
    enable_blank_line_collapse: bool,
) -> tuple[str, NormalizationAudit]:
    """
    Normalizes PDF-extracted text: merges layout line breaks, preserves structural
    breaks, repairs hyphenated word splits. Returns normalized text and audit.

    When seen_headers is provided, header lines are removed on subsequent
    occurrences (first occurrence kept). When None, uses threshold-based removal
    (lines appearing >= 10 times in the same text).

    When enable_blank_line_collapse is True, collapses single blank lines between
    prose fragments (DOF-style bold/regular block boundaries).
    """
    audit = NormalizationAudit(
        total_lines_raw=0,
        total_lines_after=0,
        merges_applied=0,
        merge_ratio=0.0,
        hyphen_joins=0,
        preserved_breaks_by_rule={},
        headers_removed=0,
        examples=[],
    )

    if not raw_text or not isinstance(raw_text, str):
        return ("", audit)

    lines = _pre_normalize(raw_text)
    lines = _merge_article_letter_lines(lines)

    # Compute headers_to_remove only when NOT using seen_headers (backward compat)
    headers_to_remove: set[str] = set()
    if seen_headers is None:
        line_counts: dict[str, int] = {}
        for line in lines:
            if _is_header_candidate(line) and len(line) > 0:
                key = _normalize_header_line_for_counting(line)
                line_counts[key] = line_counts.get(key, 0) + 1
        header_threshold = 10
        headers_to_remove = {
            k for k, c in line_counts.items()
            if c >= header_threshold and not _RE_REFORM_LEGEND.search(k)
        }

    filtered_lines: list[str] = []
    for line in lines:
        if _should_always_remove(line):
            audit.headers_removed += 1
            continue
        if _is_header_candidate(line):
            key = _normalize_header_line_for_counting(line)
            if _RE_REFORM_LEGEND.search(key):
                filtered_lines.append(line)
                continue
            if seen_headers is not None:
                if key in seen_headers:
                    audit.headers_removed += 1
                    continue
                seen_headers.add(key)
            else:
                if key in headers_to_remove:
                    audit.headers_removed += 1
                    continue
        filtered_lines.append(line)

    lines = filtered_lines
    if enable_blank_line_collapse:
        lines = _collapse_stray_blank_lines(lines)
    audit.total_lines_raw = len(lines)

    result: list[str] = []
    i = 0
    merges_applied = 0
    hyphen_joins = 0
    preserved_by_rule: dict[str, int] = {}
    current_block = ""

    def flush_block() -> None:
        nonlocal current_block
        if current_block:
            result.append(current_block)
            current_block = ""

    while i < len(lines):
        line = lines[i]
        if line == "":
            flush_block()
            result.append("")
            i += 1
            continue

        if line.strip() and _is_dof_structural_line_start(line):
            flush_block()
            result.append(line)
            i += 1
            continue

        next_line = lines[i + 1] if i + 1 < len(lines) else None

        ends_with_hyphen = bool(re.search(r"-\s*$", line))
        next_starts_with_letter = (
            next_line is not None
            and bool(re.search(r"^[a-zA-ZáéíóúñüÁÉÍÓÚÑÜ]", next_line.strip()))
        )

        if ends_with_hyphen and next_line is not None and next_starts_with_letter:
            # Skip hyphen_join when line ends with article number + hyphen (e.g. "Artículo 194-")
            # Otherwise we'd get "194N-1" instead of "194-N-1". Merge without space to preserve "194-N-1"
            if re.search(
                r"Art[íi]culo\s+\d+(?:[º°o]?(?:\.\s*Bis)?(?:\s*-\s*[A-Za-z0-9]+)*)?-\s*$",
                line,
                re.IGNORECASE,
            ):
                current_block = (
                    (current_block + " " if current_block else "")
                    + line.rstrip()
                    + next_line.strip()
                )
                merges_applied += 1
                i += 2
                continue
            merged = (current_block + " " if current_block else "") + re.sub(
                r"\s*-\s*$", "", line
            ) + next_line.strip()
            current_block = merged
            hyphen_joins += 1
            if len(audit.examples) < MAX_EXAMPLES:
                audit.examples.append(
                    NormalizationExample(
                        prev_line=line[-30:],
                        next_line=next_line[:30],
                        action="hyphen_join",
                        rule="hyphen_word_split",
                    )
                )
            i += 2
            continue

        # No unir línea actual con la siguiente si la siguiente es artículo o la actual es nota de reforma
        preserve_rule = _matches_preservation_rule(next_line) if next_line else None
        if preserve_rule is not None and next_line is not None:
            flush_block()
            result.append(line)
            preserved_by_rule[preserve_rule] = preserved_by_rule.get(preserve_rule, 0) + 1
            if len(audit.examples) < MAX_EXAMPLES:
                audit.examples.append(
                    NormalizationExample(
                        prev_line=line[-40:],
                        next_line=next_line[:40],
                        action="preserve",
                        rule=preserve_rule,
                    )
                )
            i += 1
            continue

        current_line_preserve = _matches_preservation_rule(line)
        if current_line_preserve is not None:
            flush_block()
            result.append(line)
            preserved_by_rule[current_line_preserve] = preserved_by_rule.get(
                current_line_preserve, 0
            ) + 1
            i += 1
            continue

        # Merge across a single blank line when collapse did not run or missed (same rules as collapse).
        if (
            next_line is not None
            and next_line.strip() == ""
            and i + 2 < len(lines)
        ):
            beyond_line = lines[i + 2]
            if beyond_line.strip() != "" and _should_collapse_blank_between(line, beyond_line):
                merged_pair = _join_collapsed_lines(line, beyond_line)
                current_block = (
                    (current_block + " " if current_block else "") + merged_pair
                )
                merges_applied += 1
                if len(audit.examples) < MAX_EXAMPLES:
                    audit.examples.append(
                        NormalizationExample(
                            prev_line=line[-40:],
                            next_line=beyond_line[:40],
                            action="merge",
                            rule="skip_blank_merge",
                        )
                    )
                i += 3
                continue

        if next_line is not None and next_line != "":
            if next_line.strip() and _is_dof_structural_line_start(next_line.strip()):
                if current_block:
                    merged_pre = (
                        (current_block + " " + line.strip()).strip()
                        if line.strip()
                        else current_block.strip()
                    )
                    result.append(merged_pre)
                    current_block = ""
                elif line.strip():
                    result.append(line)
                i += 1
                continue
            prev_for_score = current_block or line
            score = _compute_merge_score(prev_for_score, next_line)
            if score >= 2:
                current_block = (
                    (current_block + " " if current_block else "") + line + " " + next_line.strip()
                )
                merges_applied += 1
                if len(audit.examples) < MAX_EXAMPLES:
                    audit.examples.append(
                        NormalizationExample(
                            prev_line=line[-40:],
                            next_line=next_line[:40],
                            action="merge",
                            rule=f"score={score}",
                        )
                    )
                i += 2
                continue

        if current_block and _compute_merge_score(current_block, line) >= 2:
            current_block = current_block + " " + line.strip()
            merges_applied += 1
            i += 1
            continue

        flush_block()
        result.append(line)
        i += 1

    flush_block()

    text = _post_normalize("\n".join(result))
    total_lines_after = len(text.split("\n"))

    audit.total_lines_after = total_lines_after
    audit.merges_applied = merges_applied
    audit.merge_ratio = merges_applied / audit.total_lines_raw if audit.total_lines_raw > 0 else 0
    audit.hyphen_joins = hyphen_joins
    audit.preserved_breaks_by_rule = preserved_by_rule

    return (text, audit)
