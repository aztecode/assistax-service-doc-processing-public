"""
Quality gate for DocumentStructure (Phase 5).

Evaluates structural quality of the legal-document tree built in Phase 4,
producing a detailed quality_report with per-check results and a composite
quality_score in [0.0, 1.0].

Not yet wired into runner.py — standalone for testing and validation.
"""
from __future__ import annotations

import re
from collections import Counter

from pipeline.layout_models import DocumentStructure, StructuralNode

# ── Regex patterns ──────────────────────────────────────────────────────────

_TABLE_TOKEN_RE: re.Pattern[str] = re.compile(
    r"\[?/?TABLE(?:_\d+)?\]?",
    re.IGNORECASE,
)

_ARTICLE_NUM_RE: re.Pattern[str] = re.compile(
    r"^(\d+)\s*[-]?\s*([A-Za-z]*)$",
)

_BIS_SUFFIXES: dict[str, int] = {
    "bis": 1,
    "ter": 2,
    "quater": 3,
    "quinquies": 4,
    "sexies": 5,
    "septies": 6,
    "octies": 7,
    "nonies": 8,
    "decies": 9,
    "a": 1,
    "b": 2,
    "c": 3,
    "d": 4,
    "e": 5,
    "f": 6,
}

_HEADER_FOOTER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"DIARIO\s+OFICIAL", re.IGNORECASE),
    re.compile(r"C[AÁ]MARA\s+DE\s+DIPUTADOS", re.IGNORECASE),
    re.compile(r"C[AÁ]MARA\s+DE\s+SENADORES", re.IGNORECASE),
    re.compile(r"GACETA\s+PARLAMENTARIA", re.IGNORECASE),
    re.compile(r"SECRETAR[IÍ]A\s+DE\s+GOBERNACI[OÓ]N", re.IGNORECASE),
    re.compile(r"^\s*\d{1,4}\s*$"),  # bare page numbers
    re.compile(r"P[aá]gina\s+\d+", re.IGNORECASE),
]

_DATE_HEADING_RE: re.Pattern[str] = re.compile(
    r"^\s*\d{1,2}\s+de\s+"
    r"(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
    r"septiembre|octubre|noviembre|diciembre)"
    r"\s+de\s+\d{4}\s*$",
    re.IGNORECASE,
)

_DOF_DATE_RE: re.Pattern[str] = re.compile(
    r"(?:DOF|Diario\s+Oficial)\s+\d{1,2}[\s/\-]",
    re.IGNORECASE,
)

# ── Tree traversal helpers ──────────────────────────────────────────────────


def _collect_all_nodes(root: StructuralNode) -> list[StructuralNode]:
    result: list[StructuralNode] = [root]
    for child in root.children:
        result.extend(_collect_all_nodes(child))
    return result


def _collect_nodes_by_type(
    root: StructuralNode,
    node_type: str,
) -> list[StructuralNode]:
    return [n for n in _collect_all_nodes(root) if n.node_type == node_type]


def _collect_article_nodes(root: StructuralNode) -> list[StructuralNode]:
    return _collect_nodes_by_type(root, "article")


def _collect_navigable_nodes(root: StructuralNode) -> list[StructuralNode]:
    navigable_types: frozenset[str] = frozenset({
        "book", "title", "chapter", "section", "article", "transitory",
    })
    return [
        n for n in _collect_all_nodes(root)
        if n.node_type in navigable_types and n.node_type != "document"
    ]


# ── Check 1: visible table tokens ──────────────────────────────────────────


def _check_visible_table_tokens(
    root: StructuralNode,
    toc: list[dict[str, object]],
) -> dict[str, object]:
    navigable: list[StructuralNode] = _collect_navigable_nodes(root)
    contaminated_headings: list[str] = []
    for node in navigable:
        text: str = node.heading or ""
        if node.text:
            text = f"{text} {node.text}"
        if _TABLE_TOKEN_RE.search(text):
            contaminated_headings.append(node.node_id)

    for entry in toc:
        heading_val: object = entry.get("heading")
        if isinstance(heading_val, str) and _TABLE_TOKEN_RE.search(heading_val):
            node_id: object = entry.get("node_id", "unknown")
            if str(node_id) not in contaminated_headings:
                contaminated_headings.append(str(node_id))

    count: int = len(contaminated_headings)
    return {
        "passed": count == 0,
        "count": count,
        "contaminated_node_ids": contaminated_headings,
    }


# ── Check 2: article sequence health ──────────────────────────────────────


def _collect_article_refs(root: StructuralNode) -> list[str]:
    articles: list[StructuralNode] = _collect_article_nodes(root)
    refs: list[str] = []
    for node in articles:
        if node.article_ref is not None:
            refs.append(node.article_ref)
    return refs


def _normalize_article_ref_for_sequence(
    ref: str,
) -> tuple[int | None, str | None]:
    """Extract (base_number, suffix) from an article ref.

    Handles formats like '14', '14A', '14-A', '14 Bis', '14Bis'.
    Returns (None, None) for unparseable refs (e.g. ordinal transitories).
    """
    cleaned: str = ref.strip().replace("-", "").replace(" ", "")
    match: re.Match[str] | None = _ARTICLE_NUM_RE.match(cleaned)
    if match is None:
        return (None, None)
    base: int = int(match.group(1))
    suffix: str = match.group(2).lower() if match.group(2) else ""
    return (base, suffix if suffix else None)


def _score_article_sequence(refs: list[str]) -> dict[str, object]:
    if len(refs) < 2:
        return {
            "passed": True,
            "total_refs": len(refs),
            "disorder_ratio": 0.0,
            "max_gap": 0,
            "duplicate_ratio": 0.0,
            "detail": "too_few_refs_to_evaluate",
        }

    parsed: list[tuple[int | None, str | None]] = [
        _normalize_article_ref_for_sequence(r) for r in refs
    ]
    bases: list[int] = [p[0] for p in parsed if p[0] is not None]

    if len(bases) < 2:
        return {
            "passed": True,
            "total_refs": len(refs),
            "disorder_ratio": 0.0,
            "max_gap": 0,
            "duplicate_ratio": 0.0,
            "detail": "non_numeric_refs",
        }

    # Disorder: count pairs where next base < previous base (ignoring equal for bis)
    disorder_count: int = 0
    max_gap: int = 0
    for i in range(1, len(bases)):
        gap: int = bases[i] - bases[i - 1]
        if gap < 0:
            disorder_count += 1
        if gap > 0:
            max_gap = max(max_gap, gap)

    disorder_ratio: float = disorder_count / (len(bases) - 1)

    # Duplicates: same full ref appearing too many times
    ref_counts: Counter[str] = Counter(refs)
    total: int = len(refs)
    duplicates: int = sum(c - 1 for c in ref_counts.values() if c > 1)
    duplicate_ratio: float = duplicates / total if total > 0 else 0.0

    # A sequence is unhealthy if heavily disordered or has absurd gaps
    # Tolerant: gaps up to 20 are normal in Mexican law compilations
    passed: bool = disorder_ratio < 0.3 and max_gap <= 50 and duplicate_ratio < 0.4

    return {
        "passed": passed,
        "total_refs": len(refs),
        "numeric_bases": len(bases),
        "disorder_ratio": round(disorder_ratio, 4),
        "max_gap": max_gap,
        "duplicate_ratio": round(duplicate_ratio, 4),
    }


def _check_article_sequence_health(
    root: StructuralNode,
) -> dict[str, object]:
    refs: list[str] = _collect_article_refs(root)
    return _score_article_sequence(refs)


# ── Check 3: header/footer bleed ──────────────────────────────────────────


def _looks_like_header_footer_text(text: str) -> bool:
    for pattern in _HEADER_FOOTER_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _check_header_footer_bleed(
    root: StructuralNode,
    toc: list[dict[str, object]],
) -> dict[str, object]:
    contaminated: list[str] = []

    navigable: list[StructuralNode] = _collect_navigable_nodes(root)
    for node in navigable:
        heading: str = node.heading or ""
        if heading and _looks_like_header_footer_text(heading):
            contaminated.append(node.node_id)

    for entry in toc:
        heading_val: object = entry.get("heading")
        if isinstance(heading_val, str) and _looks_like_header_footer_text(heading_val):
            nid: str = str(entry.get("node_id", "unknown"))
            if nid not in contaminated:
                contaminated.append(nid)

    count: int = len(contaminated)
    return {
        "passed": count == 0,
        "count": count,
        "contaminated_node_ids": contaminated,
    }


# ── Check 4: date heading false positives ─────────────────────────────────


def _is_date_heading(text: str) -> bool:
    if _DATE_HEADING_RE.match(text):
        return True
    if _DOF_DATE_RE.search(text):
        return True
    return False


def _check_date_heading_false_positives(
    root: StructuralNode,
    toc: list[dict[str, object]],
) -> dict[str, object]:
    count: int = 0
    examples: list[str] = []

    navigable: list[StructuralNode] = _collect_navigable_nodes(root)
    for node in navigable:
        heading: str = node.heading or ""
        if heading and _is_date_heading(heading):
            count += 1
            if len(examples) < 5:
                examples.append(heading)

    for entry in toc:
        heading_val: object = entry.get("heading")
        if isinstance(heading_val, str) and _is_date_heading(heading_val):
            already_counted: bool = heading_val in examples
            if not already_counted:
                count += 1
                if len(examples) < 5:
                    examples.append(heading_val)

    return {
        "passed": count == 0,
        "count": count,
        "examples": examples,
    }


# ── Check 5: orphan tables ────────────────────────────────────────────────


def _is_orphan_table(node: StructuralNode) -> bool:
    # Conservative: only missing pages/source_block_ids.
    # Does not yet detect structurally misplaced tables (e.g. table
    # hanging from an unexpected parent). To be refined when chunk
    # projection exposes more context about table placement.
    if node.node_type != "table":
        return False
    has_pages: bool = node.page_start is not None and node.page_end is not None
    has_source: bool = len(node.source_block_ids) > 0
    if not has_pages or not has_source:
        return True
    return False


def _check_orphan_tables(root: StructuralNode) -> dict[str, object]:
    tables: list[StructuralNode] = _collect_nodes_by_type(root, "table")
    orphans: list[str] = [t.node_id for t in tables if _is_orphan_table(t)]
    total: int = len(tables)
    orphan_count: int = len(orphans)
    return {
        "passed": orphan_count == 0,
        "total_tables": total,
        "orphan_count": orphan_count,
        "orphan_node_ids": orphans,
    }


# ── Check 6: unknown/generic block ratio ─────────────────────────────────


def _check_unknown_block_ratio(root: StructuralNode) -> dict[str, object]:
    all_nodes: list[StructuralNode] = _collect_all_nodes(root)
    # Exclude the root document node from counting
    content_nodes: list[StructuralNode] = [
        n for n in all_nodes if n.node_type != "document"
    ]
    total: int = len(content_nodes)
    if total == 0:
        return {
            "passed": True,
            "total_content_nodes": 0,
            "unknown_count": 0,
            "ratio": 0.0,
        }

    # Nodes with "paragraph" type that came from "unknown" label can be detected
    # via metadata or the node_type itself being a generic fallback.
    # Since structure_builder maps unknown->paragraph, we count paragraphs
    # not attached to any article as potentially degraded. However, the most
    # reliable signal is nodes that have node_type not in the known structural set.
    known_types: frozenset[str] = frozenset({
        "book", "title", "chapter", "section", "article",
        "fraction", "inciso", "transitory", "table", "note", "paragraph",
    })
    unknown_nodes: list[StructuralNode] = [
        n for n in content_nodes if n.node_type not in known_types
    ]

    # Also count paragraphs that are direct children of root (no structural parent)
    # as degraded content — they likely came from unclassified blocks
    root_paragraphs: list[StructuralNode] = [
        n for n in root.children if n.node_type == "paragraph"
    ]
    degraded_count: int = len(unknown_nodes) + len(root_paragraphs)
    ratio: float = degraded_count / total if total > 0 else 0.0

    return {
        "passed": ratio < 0.3,
        "total_content_nodes": total,
        "unknown_count": len(unknown_nodes),
        "root_paragraph_count": len(root_paragraphs),
        "degraded_count": degraded_count,
        "ratio": round(ratio, 4),
    }


# ── Check 7: TOC duplicate ratio ─────────────────────────────────────────


def _check_toc_duplicate_ratio(toc: list[dict[str, object]]) -> dict[str, object]:
    total: int = len(toc)
    if total == 0:
        return {
            "passed": True,
            "total_entries": 0,
            "duplicate_count": 0,
            "ratio": 0.0,
        }

    # Build composite keys for duplicate detection
    keys: list[str] = []
    for entry in toc:
        nid: str = str(entry.get("node_id", ""))
        ntype: str = str(entry.get("node_type", ""))
        heading: str = str(entry.get("heading", ""))
        pstart: str = str(entry.get("page_start", ""))
        key: str = f"{ntype}|{heading}|{pstart}"
        keys.append(key)

    # Also check for repeated node_ids
    id_counts: Counter[str] = Counter(
        str(entry.get("node_id", "")) for entry in toc
    )
    key_counts: Counter[str] = Counter(keys)

    duplicates_by_key: int = sum(c - 1 for c in key_counts.values() if c > 1)
    duplicates_by_id: int = sum(c - 1 for c in id_counts.values() if c > 1)
    duplicate_count: int = max(duplicates_by_key, duplicates_by_id)
    ratio: float = duplicate_count / total if total > 0 else 0.0

    return {
        "passed": ratio < 0.15,
        "total_entries": total,
        "duplicate_count": duplicate_count,
        "ratio": round(ratio, 4),
    }


# ── Check 8: article ref coverage ────────────────────────────────────────


def _compute_article_ref_coverage(
    nodes: list[StructuralNode],
) -> dict[str, object]:
    total: int = len(nodes)
    if total == 0:
        return {
            "passed": True,
            "total_articles": 0,
            "with_ref": 0,
            "without_ref": 0,
            "coverage": 1.0,
        }

    with_ref: int = sum(1 for n in nodes if n.article_ref is not None)
    without_ref: int = total - with_ref
    coverage: float = with_ref / total

    # Tolerant: only fail if a normative document has very poor coverage
    passed: bool = coverage >= 0.5 or total <= 2

    return {
        "passed": passed,
        "total_articles": total,
        "with_ref": with_ref,
        "without_ref": without_ref,
        "coverage": round(coverage, 4),
    }


def _check_article_ref_coverage(
    root: StructuralNode,
) -> dict[str, object]:
    articles: list[StructuralNode] = _collect_article_nodes(root)
    return _compute_article_ref_coverage(articles)


# ── Score computation ────────────────────────────────────────────────────

# Weight distribution for composite score.
# Higher weight = more impact on final score.
# Severe structural issues (table tokens, bleed, sequence) weigh more.
_CHECK_WEIGHTS: dict[str, float] = {
    "has_visible_table_tokens": 0.20,
    "article_sequence_health": 0.15,
    "header_footer_bleed": 0.20,
    "date_heading_false_positive_count": 0.05,
    "orphan_tables_count": 0.05,
    "unknown_block_ratio": 0.10,
    "toc_duplicate_ratio": 0.10,
    "article_ref_coverage": 0.15,
}


def _check_score(check_result: dict[str, object], check_name: str) -> float:
    """Convert a single check result into a 0.0–1.0 sub-score.

    Each check has its own degradation curve depending on the severity
    metrics it reports.
    """
    passed: bool = bool(check_result.get("passed", True))
    if passed:
        return 1.0

    if check_name == "has_visible_table_tokens":
        count: int = int(check_result.get("count", 0))
        # Even 1 contaminated heading is severe; strong per-token penalty
        return max(0.0, 1.0 - count * 0.8)

    if check_name == "article_sequence_health":
        disorder: float = float(check_result.get("disorder_ratio", 0.0))
        dup: float = float(check_result.get("duplicate_ratio", 0.0))
        max_gap: int = int(check_result.get("max_gap", 0))
        # Gaps beyond 50 are suspicious; scale penalty linearly up to 100
        gap_penalty: float = min(1.0, max(0.0, max_gap - 50) / 50) * 0.5
        # Blend disorder, duplicate, and gap penalties
        penalty: float = min(1.0, disorder * 1.5 + dup * 0.8 + gap_penalty)
        return max(0.0, 1.0 - penalty)

    if check_name == "header_footer_bleed":
        count = int(check_result.get("count", 0))
        return max(0.0, 1.0 - count * 0.4)

    if check_name == "date_heading_false_positive_count":
        count = int(check_result.get("count", 0))
        return max(0.0, 1.0 - count * 0.15)

    if check_name == "orphan_tables_count":
        orphan_count: int = int(check_result.get("orphan_count", 0))
        total_tables: int = int(check_result.get("total_tables", 1))
        ratio: float = orphan_count / total_tables if total_tables > 0 else 0.0
        return max(0.0, 1.0 - ratio)

    if check_name == "unknown_block_ratio":
        r: float = float(check_result.get("ratio", 0.0))
        return max(0.0, 1.0 - r * 2.0)

    if check_name == "toc_duplicate_ratio":
        r = float(check_result.get("ratio", 0.0))
        return max(0.0, 1.0 - r * 2.5)

    if check_name == "article_ref_coverage":
        coverage: float = float(check_result.get("coverage", 1.0))
        return coverage

    return 1.0 if passed else 0.0


def compute_quality_score(report: dict[str, object]) -> float:
    """Compute a weighted quality score from the checks in a quality report.

    Formula: score = sum(weight_i * sub_score_i) for each check.
    Sub-scores are derived from per-check metrics, not just pass/fail.
    """
    checks: object = report.get("checks", {})
    if not isinstance(checks, dict):
        return 0.0

    total_weight: float = 0.0
    weighted_sum: float = 0.0

    for check_name, weight in _CHECK_WEIGHTS.items():
        check_result: object = checks.get(check_name)
        if check_result is None or not isinstance(check_result, dict):
            sub_score: float = 1.0
        else:
            sub_score = _check_score(check_result, check_name)
        weighted_sum += weight * sub_score
        total_weight += weight

    if total_weight == 0.0:
        return 1.0

    return round(weighted_sum / total_weight, 4)


# ── Summary builder ─────────────────────────────────────────────────────


def _build_summary(
    score: float,
    checks: dict[str, dict[str, object]],
) -> dict[str, object]:
    reasons: list[str] = []
    for name, result in checks.items():
        if not result.get("passed", True):
            reasons.append(name)

    severity: str
    if score >= 0.85:
        severity = "low"
    elif score >= 0.70:
        severity = "medium"
    else:
        severity = "high"

    return {
        "passed": score >= 0.70,
        "severity": severity,
        "reasons": reasons,
    }


# ── Public API ──────────────────────────────────────────────────────────


def validate_document_structure(
    structure: DocumentStructure,
) -> dict[str, object]:
    """Run all quality checks on a DocumentStructure and return a quality report.

    Returns a dict with quality_score, checks, and summary.
    The report is JSON-serializable and can be stored in
    DocumentStructure.quality_report in a later phase.
    """
    checks: dict[str, dict[str, object]] = {
        "has_visible_table_tokens": _check_visible_table_tokens(
            structure.root, structure.toc,
        ),
        "article_sequence_health": _check_article_sequence_health(
            structure.root,
        ),
        "header_footer_bleed": _check_header_footer_bleed(
            structure.root, structure.toc,
        ),
        "date_heading_false_positive_count": _check_date_heading_false_positives(
            structure.root, structure.toc,
        ),
        "orphan_tables_count": _check_orphan_tables(structure.root),
        "unknown_block_ratio": _check_unknown_block_ratio(structure.root),
        "toc_duplicate_ratio": _check_toc_duplicate_ratio(structure.toc),
        "article_ref_coverage": _check_article_ref_coverage(structure.root),
    }

    partial_report: dict[str, object] = {
        "quality_score": 0.0,
        "checks": checks,
    }

    score: float = compute_quality_score(partial_report)
    summary: dict[str, object] = _build_summary(score, checks)

    return {
        "quality_score": score,
        "checks": checks,
        "summary": summary,
    }
