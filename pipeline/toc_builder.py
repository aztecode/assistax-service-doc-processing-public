"""
Build hierarchical TOC from legal chunk rows (parity with assistax-back buildHierarchicalOutline).
Pure functions: input rows are plain dicts / mappings; output is JSON-serializable dict trees.

Strategies:
  - STRATEGY_CHUNK_BASED:  tree built exclusively from chunk rows (original behavior).
  - STRATEGY_NATIVE_HYBRID: native PDF bookmarks as structural base, enriched with chunks.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, NotRequired, TypedDict

from pipeline.heading_classifier import LLM_ELIGIBLE_TYPES, classify_heading_node
from settings import settings

STRATEGY_CHUNK_BASED: str = "hierarchical_chunk_based_v1"
STRATEGY_NATIVE_HYBRID: str = "native_toc_hybrid_v1"


class TocTargetDict(TypedDict):
    page: int
    offset: NotRequired[int]


class TocNodeDict(TypedDict, total=False):
    id: str
    title: str
    level: int
    target: TocTargetDict
    children: list[TocNodeDict]


_TRANSITORIO_CONTAINER = re.compile(
    r"^(?:ART[ÍI]CULOS?\s+TRANSITORIOS?|TRANSITORIOS?|Transitorio)\s*$",
    re.IGNORECASE,
)

_EDITORIAL_NOTE_HEADING = re.compile(
    r"(?:^|\b)(?:ACLARACI[OÓ]N|Nota\s*de\s+erratas|Fe\s+de\s+erratas|N\.\s*de\s+E\."
    r"|NOTA\s+DEL\s+EDITOR|Nota\s*:\s|NOTA\s+ACLARATORIA|Correcci[oó]n\s+al\s"
    r"|AVISO\b|Nota\s+editorial)",
    re.IGNORECASE,
)

_OUTLINE_BODY_PHRASE = re.compile(
    r"(?:^|\s)(con el |con la |de la |de la ley |del cap[íi]tulo |del t[íi]tulo |del c[oó]digo |"
    r"de esta Ley[,.]?\s*|de la presente Ley[,.]?\s*|de este C[oó]digo[,.]?\s*|primera del |"
    r"segunda del |tercera del |partir del |partir de la |se considera|se señalan|"
    r"que incluye |incluye los art[íi]culos )\b",
    re.IGNORECASE,
)

_LEVEL_MAP: dict[str, int] = {
    "book": 1,
    "title": 2,
    "chapter": 3,
    "section": 4,
    "article": 5,
    "rule": 5,
    "transitorio": 4,
    "numeral": 6,
}

_STRUCTURAL_BLOCK_TYPES = frozenset({"book", "title", "chapter", "section"})


def _is_transitorio_container_heading(heading: str) -> bool:
    return bool(_TRANSITORIO_CONTAINER.match(heading.strip()))


def _normalize_transitorio_container_title(title: str) -> str:
    """Match JS double replace on \\w-boundaries (ASCII word chars)."""

    def up_first(m: re.Match[str]) -> str:
        return m.group(1).upper()

    s = re.sub(r"\b(\w)", up_first, title)

    def title_word(m: re.Match[str]) -> str:
        w = m.group(1)
        if len(w) <= 1:
            return w.upper()
        return w[0].upper() + w[1:].lower()

    return re.sub(r"\b(\w+)\b", title_word, s)


def _page_for_row(row: Mapping[str, Any]) -> int:
    sp = row.get("startPage")
    if isinstance(sp, int) and sp >= 1:
        return sp
    return 1


def _build_transitorio_block_index(
    chunks: list[Mapping[str, Any]],
) -> dict[int, int]:
    block_by_chunk_no: dict[int, int] = {}
    block_index = 0
    in_block = False
    for chunk in chunks:
        chunk_no = int(chunk["chunkNo"])
        ct = str(chunk.get("chunkType") or "")
        ar = (chunk.get("articleRef") or "") or ""
        ar = ar.strip() if isinstance(ar, str) else ""
        heading = (chunk.get("heading") or "") or ""
        heading = heading if isinstance(heading, str) else ""
        is_container = (
            ct == "transitorio"
            and not ar
            and _is_transitorio_container_heading(heading)
        )
        if is_container:
            block_index += 1
            in_block = True
            block_by_chunk_no[chunk_no] = block_index
        elif ct in _STRUCTURAL_BLOCK_TYPES:
            in_block = False
        elif in_block and (ct == "article" or ct == "transitorio"):
            block_by_chunk_no[chunk_no] = block_index
    return block_by_chunk_no


def build_toc_tree(
    rows: list[Mapping[str, Any]],
) -> tuple[list[TocNodeDict], dict[str, Any]]:
    """
    Returns (root_nodes, stats) matching legalOutlineService.buildHierarchicalOutline tree/stats.
    """
    chunks: list[Mapping[str, Any]] = list(rows)
    if not chunks:
        return [], {"totalNodes": 0, "byLevel": {}, "maxDepth": 0}

    transitorio_block_by_chunk_no = _build_transitorio_block_index(chunks)
    nodes: list[TocNodeDict] = []
    stack: list[TocNodeDict] = []
    seen_articles: set[str] = set()

    for chunk in chunks:
        chunk_no = int(chunk["chunkNo"])
        chunk_type = str(chunk.get("chunkType") or "")
        block_index = transitorio_block_by_chunk_no.get(chunk_no, 0)
        is_article_in_transitorio = chunk_type == "article" and block_index > 0
        effective_type = "transitorio" if is_article_in_transitorio else chunk_type

        if chunk_type == "boxed_note":
            continue

        if effective_type not in _LEVEL_MAP:
            continue

        ar_raw = chunk.get("articleRef")
        ar = (ar_raw.strip() if isinstance(ar_raw, str) else "") or ""
        heading_raw = chunk.get("heading")
        heading = (heading_raw.strip() if isinstance(heading_raw, str) else "") or ""

        is_transitorio_container = (
            chunk_type == "transitorio"
            and not ar
            and _is_transitorio_container_heading(heading)
        )
        if is_transitorio_container:
            level = 4
        elif chunk_type == "transitorio" or is_article_in_transitorio:
            level = 5
        else:
            level = _LEVEL_MAP[effective_type]

        if not heading and not ar:
            continue

        title = heading or ar or f"Chunk {chunk_no}"
        if len(title) > 150 or "\n" in title:
            title = title.split("\n", 1)[0][:150]

        if chunk_type == "article" or chunk_type == "rule":
            title = ar or title
        elif is_transitorio_container:
            title = _normalize_transitorio_container_title(title)
        elif chunk_type == "transitorio" and ar:
            title = ar or title
        elif is_article_in_transitorio:
            title = ar or title
        elif chunk_type == "fraction" or chunk_type == "numeral":
            base_title = ar or ""
            detail = title[len(base_title) :].strip()
            if len(detail) > 50:
                title = base_title + f": {detail[:50]}..."
            elif detail:
                title = base_title + f": {detail}"
            else:
                title = base_title

        if _EDITORIAL_NOTE_HEADING.search(title):
            continue

        if _OUTLINE_BODY_PHRASE.search(title):
            if (
                chunk_type in LLM_ELIGIBLE_TYPES
                and settings.ENABLE_LLM_HEADING_CLASSIFIER
            ):
                is_structural, confidence = classify_heading_node(
                    title, chunk_type, str(chunk.get("text") or "")[:300]
                )
                if not is_structural or confidence == "baja":
                    continue
                # rescued by LLM — fall through to include the node
            else:
                continue

        if (
            chunk_type in ("article", "title", "chapter")
            and not is_article_in_transitorio
            and re.match(r"^(Artículo|Título|Capítulo)\s*$", title, re.IGNORECASE)
        ):
            continue

        if level == 2 and len(title) > 100 and not re.match(
            r"^Título\s+[IVX\d]+", title, re.IGNORECASE
        ):
            if (
                chunk_type in LLM_ELIGIBLE_TYPES
                and settings.ENABLE_LLM_HEADING_CLASSIFIER
            ):
                is_structural, confidence = classify_heading_node(
                    title, chunk_type, str(chunk.get("text") or "")[:300]
                )
                if not is_structural or confidence == "baja":
                    continue
                # rescued by LLM — fall through to include the node
            else:
                continue

        norm_title = title.strip().lower()
        if chunk_type == "rule":
            dedupe_key = f"rule:{norm_title}"
        elif is_article_in_transitorio or chunk_type == "transitorio":
            block_id = str(block_index) if block_index > 0 else "orphan"
            suffix = (
                "container"
                if chunk_type == "transitorio" and not ar
                else norm_title
            )
            dedupe_key = f"transitorio:{block_id}:{suffix}"
        elif chunk_type == "article":
            dedupe_key = f"article:{norm_title}"
        else:
            dedupe_key = f"{effective_type}:{norm_title}"

        if chunk_type in ("article", "rule", "transitorio") or is_article_in_transitorio:
            if dedupe_key in seen_articles:
                continue
            seen_articles.add(dedupe_key)

        page = _page_for_row(chunk)
        node: TocNodeDict = {
            "id": f"chunk-{chunk_no}",
            "title": title,
            "level": level,
            "target": {"page": page},
        }

        while stack:
            top = stack[-1]
            if top["level"] >= level:
                stack.pop()
            else:
                break

        if stack:
            parent = stack[-1]
            ch = parent.get("children")
            if ch is None:
                parent["children"] = []
                ch = parent["children"]
            ch.append(node)
        else:
            nodes.append(node)

        stack.append(node)

    by_level = _count_by_level(nodes)
    max_depth = _max_depth(nodes)
    total_nodes = _count_total(nodes)

    stats: dict[str, Any] = {
        "totalNodes": total_nodes,
        "byLevel": by_level,
        "maxDepth": max_depth,
    }
    return nodes, stats


def _count_by_level(
    node_list: list[TocNodeDict], counts: dict[int, int] | None = None
) -> dict[int, int]:
    if counts is None:
        counts = {}
    for node in node_list:
        lvl = int(node["level"])
        counts[lvl] = counts.get(lvl, 0) + 1
        children = node.get("children")
        if children:
            _count_by_level(children, counts)
    return counts


def _max_depth(node_list: list[TocNodeDict], current_depth: int = 1) -> int:
    best = current_depth
    for node in node_list:
        children = node.get("children")
        if children:
            d = _max_depth(children, current_depth + 1)
            best = max(best, d)
    return best


def _count_total(node_list: list[TocNodeDict]) -> int:
    n = len(node_list)
    for node in node_list:
        children = node.get("children")
        if children:
            n += _count_total(children)
    return n


def sections_from_toc_tree(tree: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Flatten tree to manifest sections (same walk as documentController)."""

    def walk(n: Mapping[str, Any]) -> list[dict[str, Any]]:
        tgt = n.get("target") or {}
        page = int(tgt.get("page") or 1)
        out: list[dict[str, Any]] = [
            {
                "id": str(n["id"]),
                "title": str(n["title"]),
                "pageStart": page,
                "pageEnd": page,
            }
        ]
        for c in n.get("children") or []:
            if isinstance(c, Mapping):
                out.extend(walk(c))
        return out

    flat: list[dict[str, Any]] = []
    for root in tree:
        if isinstance(root, Mapping):
            flat.extend(walk(root))
    return flat


def manifest_version_from_toc(
    content_hash: str,
    tree: list[Mapping[str, Any]],
    strategy: str = STRATEGY_CHUNK_BASED,
) -> str:
    canonical = json.dumps(tree, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    toc_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    ch = (content_hash or "")[:12]
    return f"{strategy}-{ch}-{toc_digest}"


def merge_top_level_metadata(
    existing: Mapping[str, Any] | None,
    patch: Mapping[str, Any],
) -> dict[str, Any]:
    """Shallow merge: patch keys overwrite; nested dicts are not deep-merged."""
    base: dict[str, Any] = {}
    if existing and isinstance(existing, dict):
        base = dict(existing)
    for key, value in patch.items():
        base[key] = value
    return base


# ---------------------------------------------------------------------------
# Native TOC hybrid builder (native_toc_hybrid_v1)
# ---------------------------------------------------------------------------

_NATIVE_TRANSITORIO = re.compile(
    r"^(?:art[íi]culos?\s+transitorios?|transitorios?)\b",
    re.IGNORECASE,
)


def _normalize_native_title(title: str) -> str:
    """Replace underscores and collapse whitespace for bookmark comparison."""
    return re.sub(r"[_\s]+", " ", title).strip()


def _is_native_transitorio_entry(entry: Mapping[str, Any]) -> bool:
    title = _normalize_native_title(str(entry.get("title") or ""))
    if not title:
        return False
    return bool(_NATIVE_TRANSITORIO.match(title))


def _find_native_transitorio_entries(
    native_toc: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [e for e in native_toc if _is_native_transitorio_entry(e)]


def _is_native_toc_usable(native_toc: list[dict[str, Any]]) -> bool:
    """Native TOC is usable if it has at least 3 titled entries."""
    if not native_toc or len(native_toc) < 3:
        return False
    titled = sum(1 for e in native_toc if (str(e.get("title") or "")).strip())
    return titled >= 3


def _count_transitorio_containers_in_tree(nodes: list[TocNodeDict]) -> int:
    count = 0
    for node in nodes:
        if (
            node.get("level") == 4
            and _is_transitorio_container_heading(str(node.get("title", "")))
        ):
            count += 1
        children = node.get("children")
        if children:
            count += _count_transitorio_containers_in_tree(children)
    return count


def _collapse_transitorio_containers(
    tree: list[TocNodeDict],
    native_trans: list[dict[str, Any]],
) -> list[TocNodeDict]:
    """
    Collapse chunk-based transitorio containers to match native TOC count.
    Single native entry -> one merged container.
    Multiple native entries -> distribute children by page-range proximity.
    """
    all_children: list[TocNodeDict] = []
    container_ids: list[str] = []
    seen_child_ids: set[str] = set()

    def _strip(nodes: list[TocNodeDict]) -> list[TocNodeDict]:
        result: list[TocNodeDict] = []
        for node in nodes:
            is_container = (
                node.get("level") == 4
                and _is_transitorio_container_heading(str(node.get("title", "")))
            )
            if is_container:
                container_ids.append(str(node["id"]))
                for child in node.get("children") or []:
                    cid = str(child.get("id", ""))
                    if cid not in seen_child_ids:
                        seen_child_ids.add(cid)
                        all_children.append(child)
            else:
                new_node = dict(node)
                children = node.get("children")
                if children:
                    new_children = _strip(list(children))
                    if new_children:
                        new_node["children"] = new_children
                    elif "children" in new_node:
                        del new_node["children"]
                result.append(new_node)
        return result

    cleaned = _strip(tree)

    if not container_ids:
        return tree

    sorted_native = sorted(native_trans, key=lambda e: int(e.get("page") or 1))

    if len(sorted_native) == 1:
        native_page = int(sorted_native[0].get("page") or 1)
        merged: TocNodeDict = {
            "id": container_ids[0],
            "title": "Transitorios",
            "level": 4,
            "target": {"page": native_page},
        }
        if all_children:
            merged["children"] = all_children
        cleaned.append(merged)
    else:
        page_starts = [int(ne.get("page") or 1) for ne in sorted_native]
        containers: list[TocNodeDict] = []
        for i, ne in enumerate(sorted_native):
            cid = container_ids[i] if i < len(container_ids) else f"native-trans-{i}"
            raw_title = _normalize_native_title(str(ne.get("title") or "Transitorios"))
            title = _normalize_transitorio_container_title(raw_title)
            c: TocNodeDict = {
                "id": cid,
                "title": title,
                "level": 4,
                "target": {"page": page_starts[i]},
            }
            containers.append(c)

        for child in all_children:
            child_page = int((child.get("target") or {}).get("page") or 1)
            best_idx = 0
            for j in range(1, len(page_starts)):
                if child_page >= page_starts[j]:
                    best_idx = j
            ch_list = containers[best_idx].get("children")
            if ch_list is None:
                containers[best_idx]["children"] = []
            containers[best_idx]["children"].append(child)

        for c in containers:
            cleaned.append(c)

    return cleaned


def build_native_toc_hybrid_tree(
    native_toc: list[dict[str, Any]],
    chunk_rows: list[Mapping[str, Any]],
) -> tuple[list[TocNodeDict], dict[str, Any]]:
    """
    Build TOC using native PDF bookmarks as structural base, enriched with chunks.

    1. Build chunk-based tree as baseline (preserves article-level granularity and chunk IDs).
    2. Use native TOC to correct transitorio structure:
       - Collapse multiple transitorio containers to match native TOC count.
       - Apply native page numbers to merged containers.
    3. Fallback to pure chunk-based if native TOC is unusable.

    Returns (root_nodes, stats) with same shape as build_toc_tree.
    """
    if not _is_native_toc_usable(native_toc):
        return build_toc_tree(chunk_rows)

    tree, _ = build_toc_tree(chunk_rows)
    if not tree:
        return tree, {"totalNodes": 0, "byLevel": {}, "maxDepth": 0}

    native_trans = _find_native_transitorio_entries(native_toc)
    chunk_trans_count = _count_transitorio_containers_in_tree(tree)

    if native_trans and chunk_trans_count > len(native_trans):
        tree = _collapse_transitorio_containers(tree, native_trans)

    # Extension point: LLM arbitration for conflicts between native bookmarks
    # and chunk-based headings is handled inside build_toc_tree when
    # ENABLE_LLM_HEADING_CLASSIFIER is active. Future per-node conflict
    # resolution (e.g. page mismatch, title mismatch) can be added here.

    by_level = _count_by_level(tree)
    max_depth = _max_depth(tree)
    total_nodes = _count_total(tree)
    stats: dict[str, Any] = {
        "totalNodes": total_nodes,
        "byLevel": by_level,
        "maxDepth": max_depth,
    }
    return tree, stats


def native_toc_stats(native_toc: list[dict[str, Any]]) -> dict[str, Any]:
    """Summary stats for persisting alongside the outline for traceability."""
    if not native_toc:
        return {"entries": 0, "transitorioSections": 0}
    trans = _find_native_transitorio_entries(native_toc)
    return {
        "entries": len(native_toc),
        "transitorioSections": len(trans),
    }
