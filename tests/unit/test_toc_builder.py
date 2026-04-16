"""Tests for hierarchical TOC builder (parity with Node legalOutlineService)."""

from __future__ import annotations

from unittest.mock import patch

from pipeline.toc_builder import (
    STRATEGY_CHUNK_BASED,
    STRATEGY_NATIVE_HYBRID,
    build_native_toc_hybrid_tree,
    build_toc_tree,
    manifest_version_from_toc,
    merge_top_level_metadata,
    native_toc_stats,
    sections_from_toc_tree,
)


def test_build_toc_tree_title_and_articles() -> None:
    rows = [
        {
            "chunkNo": 1,
            "chunkType": "title",
            "heading": "Título Primero",
            "articleRef": None,
            "text": "Título Primero\nfoo",
            "startPage": 1,
            "endPage": 1,
        },
        {
            "chunkNo": 2,
            "chunkType": "article",
            "heading": "Artículo 1",
            "articleRef": "Artículo 1",
            "text": "Artículo 1. Texto",
            "startPage": 2,
            "endPage": 2,
        },
    ]
    tree, stats = build_toc_tree(rows)
    assert stats["totalNodes"] == 2
    assert len(tree) == 1
    assert tree[0]["id"] == "chunk-1"
    assert tree[0]["level"] == 2
    assert tree[0]["children"] is not None
    assert len(tree[0]["children"]) == 1
    child = tree[0]["children"][0]
    assert child["id"] == "chunk-2"
    assert child["target"]["page"] == 2


def test_transitorio_container_and_child() -> None:
    rows = [
        {
            "chunkNo": 1,
            "chunkType": "title",
            "heading": "Título Cuarto",
            "articleRef": None,
            "text": "Título Cuarto",
            "startPage": 1,
            "endPage": 1,
        },
        {
            "chunkNo": 2,
            "chunkType": "transitorio",
            "heading": "TRANSITORIOS",
            "articleRef": None,
            "text": "TRANSITORIOS",
            "startPage": 5,
            "endPage": 5,
        },
        {
            "chunkNo": 3,
            "chunkType": "article",
            "heading": "Artículo 1 Transitorio",
            "articleRef": "Artículo 1 Transitorio",
            "text": "Artículo 1 Transitorio.- Texto",
            "startPage": 5,
            "endPage": 5,
        },
    ]
    tree, _stats = build_toc_tree(rows)
    assert len(tree) == 1
    root_children = tree[0].get("children") or []
    trans_nodes = [n for n in root_children if n.get("level") == 4]
    assert len(trans_nodes) == 1
    assert trans_nodes[0]["children"]
    assert trans_nodes[0]["children"][0]["level"] == 5


def test_sections_from_toc_tree_order() -> None:
    tree = [
        {
            "id": "chunk-1",
            "title": "A",
            "level": 1,
            "target": {"page": 1},
            "children": [
                {"id": "chunk-2", "title": "B", "level": 2, "target": {"page": 3}},
            ],
        }
    ]
    sections = sections_from_toc_tree(tree)
    assert [s["id"] for s in sections] == ["chunk-1", "chunk-2"]
    assert sections[1]["pageStart"] == 3


def test_manifest_version_stable() -> None:
    tree = [{"id": "chunk-1", "title": "X", "level": 1, "target": {"page": 1}}]
    v1 = manifest_version_from_toc("abc" * 20, tree)
    v2 = manifest_version_from_toc("abc" * 20, tree)
    assert v1 == v2
    assert v1.startswith("hierarchical_chunk_based_v1-")


def test_merge_top_level_metadata_preserves_keys() -> None:
    merged = merge_top_level_metadata(
        {"a": 1, "toc": [{"old": True}]},
        {"toc": [{"new": True}], "pageCount": 10},
    )
    assert merged["a"] == 1
    assert merged["pageCount"] == 10
    assert merged["toc"] == [{"new": True}]


def test_outline_error_cleared_on_merge() -> None:
    merged = merge_top_level_metadata(
        {"outlineError": "failed"},
        {"outlineError": None},
    )
    assert merged["outlineError"] is None


# ---------------------------------------------------------------------------
# LLM heading classifier integration tests
# ---------------------------------------------------------------------------

def _chapter_body_ref_rows() -> list[dict]:
    """Chunk whose heading matches _OUTLINE_BODY_PHRASE (would normally be discarded)."""
    return [
        {
            "chunkNo": 1,
            "chunkType": "title",
            "heading": "Título Primero",
            "articleRef": None,
            "text": "Título Primero\nDisposiciones generales",
            "startPage": 1,
            "endPage": 1,
        },
        {
            "chunkNo": 2,
            "chunkType": "chapter",
            # "de esta Ley" triggers _OUTLINE_BODY_PHRASE — normally discarded
            "heading": "Capítulo III de esta Ley",
            "articleRef": None,
            "text": "Capítulo III de esta Ley\nDe las obligaciones de los contribuyentes",
            "startPage": 2,
            "endPage": 2,
        },
        {
            "chunkNo": 3,
            "chunkType": "article",
            "heading": "Artículo 10",
            "articleRef": "Artículo 10",
            "text": "Artículo 10. Obligación de...",
            "startPage": 3,
            "endPage": 3,
        },
    ]


def test_body_phrase_node_discarded_without_llm_flag() -> None:
    """With flag off the heuristic still discards the body-phrase chapter."""
    rows = _chapter_body_ref_rows()
    tree, stats = build_toc_tree(rows)
    # Only title (level 2) and article (level 5) survive; chapter is discarded
    all_titles = [n["title"] for n in tree] + [
        c["title"] for n in tree for c in (n.get("children") or [])
    ]
    assert not any("Capítulo III" in t for t in all_titles)


def test_body_phrase_node_rescued_by_llm() -> None:
    """LLM returning is_structural=True with alta confidence rescues the node."""
    rows = _chapter_body_ref_rows()

    with (
        patch("pipeline.toc_builder.settings") as mock_settings,
        patch(
            "pipeline.toc_builder.classify_heading_node",
            return_value=(True, "alta"),
        ) as mock_classify,
    ):
        mock_settings.ENABLE_LLM_HEADING_CLASSIFIER = True
        tree, stats = build_toc_tree(rows)

    # Classifier must have been called for the chapter candidate
    mock_classify.assert_called_once()
    call_kwargs = mock_classify.call_args
    assert call_kwargs[0][1] == "chapter"  # chunk_type arg

    # Rescued chapter should appear in the tree
    title_node = tree[0]
    children_titles = [c["title"] for c in (title_node.get("children") or [])]
    assert any("Capítulo III" in t for t in children_titles)


def test_body_phrase_node_stays_discarded_when_llm_returns_false() -> None:
    """LLM returning is_structural=False keeps the heuristic discard."""
    rows = _chapter_body_ref_rows()

    with (
        patch("pipeline.toc_builder.settings") as mock_settings,
        patch(
            "pipeline.toc_builder.classify_heading_node",
            return_value=(False, "alta"),
        ),
    ):
        mock_settings.ENABLE_LLM_HEADING_CLASSIFIER = True
        tree, stats = build_toc_tree(rows)

    all_titles = [n["title"] for n in tree] + [
        c["title"] for n in tree for c in (n.get("children") or [])
    ]
    assert not any("Capítulo III" in t for t in all_titles)


def test_body_phrase_node_stays_discarded_when_llm_baja_confidence() -> None:
    """Low confidence defers to heuristic even when LLM says is_structural=True."""
    rows = _chapter_body_ref_rows()

    with (
        patch("pipeline.toc_builder.settings") as mock_settings,
        patch(
            "pipeline.toc_builder.classify_heading_node",
            return_value=(True, "baja"),
        ),
    ):
        mock_settings.ENABLE_LLM_HEADING_CLASSIFIER = True
        tree, stats = build_toc_tree(rows)

    all_titles = [n["title"] for n in tree] + [
        c["title"] for n in tree for c in (n.get("children") or [])
    ]
    assert not any("Capítulo III" in t for t in all_titles)


def test_long_title_level2_rescued_by_llm() -> None:
    """A level-2 title longer than 100 chars is rescued when LLM is confident."""
    long_title = "Título I " + "De las disposiciones generales aplicables a " * 3
    rows = [
        {
            "chunkNo": 1,
            "chunkType": "title",
            "heading": long_title,
            "articleRef": None,
            "text": long_title,
            "startPage": 1,
            "endPage": 1,
        }
    ]

    with (
        patch("pipeline.toc_builder.settings") as mock_settings,
        patch(
            "pipeline.toc_builder.classify_heading_node",
            return_value=(True, "alta"),
        ),
    ):
        mock_settings.ENABLE_LLM_HEADING_CLASSIFIER = True
        tree, _stats = build_toc_tree(rows)

    assert len(tree) == 1
    assert tree[0]["level"] == 2


def test_llm_not_called_for_non_eligible_types() -> None:
    """Article and rule types never trigger LLM calls even if they match body phrase."""
    rows = [
        {
            "chunkNo": 1,
            "chunkType": "article",
            # trigger body phrase pattern
            "heading": "Artículo 5 de esta Ley",
            "articleRef": "Artículo 5 de esta Ley",
            "text": "Artículo 5 de esta Ley. Texto del artículo.",
            "startPage": 1,
            "endPage": 1,
        }
    ]

    with (
        patch("pipeline.toc_builder.settings") as mock_settings,
        patch(
            "pipeline.toc_builder.classify_heading_node",
        ) as mock_classify,
    ):
        mock_settings.ENABLE_LLM_HEADING_CLASSIFIER = True
        build_toc_tree(rows)

    mock_classify.assert_not_called()


# ---------------------------------------------------------------------------
# Native TOC hybrid builder tests (native_toc_hybrid_v1)
# ---------------------------------------------------------------------------

def _multiple_transitorio_containers_rows() -> list[dict]:
    """Chunks with multiple transitorio container headings (the problem scenario)."""
    return [
        {"chunkNo": 1, "chunkType": "title", "heading": "Título Primero",
         "articleRef": None, "text": "Título Primero", "startPage": 1, "endPage": 1},
        {"chunkNo": 2, "chunkType": "article", "heading": "Artículo 1",
         "articleRef": "Artículo 1", "text": "Artículo 1. Texto", "startPage": 2, "endPage": 2},
        {"chunkNo": 10, "chunkType": "transitorio", "heading": "TRANSITORIOS",
         "articleRef": None, "text": "TRANSITORIOS", "startPage": 50, "endPage": 50},
        {"chunkNo": 11, "chunkType": "article", "heading": "Artículo 1 Transitorio",
         "articleRef": "Artículo 1 Transitorio", "text": "Art 1 Trans", "startPage": 50, "endPage": 50},
        {"chunkNo": 12, "chunkType": "article", "heading": "Artículo 2 Transitorio",
         "articleRef": "Artículo 2 Transitorio", "text": "Art 2 Trans", "startPage": 51, "endPage": 51},
        {"chunkNo": 20, "chunkType": "transitorio", "heading": "ARTÍCULOS TRANSITORIOS",
         "articleRef": None, "text": "ARTÍCULOS TRANSITORIOS", "startPage": 100, "endPage": 100},
        {"chunkNo": 21, "chunkType": "article", "heading": "Artículo 3 Transitorio",
         "articleRef": "Artículo 3 Transitorio", "text": "Art 3 Trans", "startPage": 100, "endPage": 100},
        {"chunkNo": 30, "chunkType": "transitorio", "heading": "Transitorio",
         "articleRef": None, "text": "Transitorio", "startPage": 150, "endPage": 150},
        {"chunkNo": 31, "chunkType": "article", "heading": "Artículo 4 Transitorio",
         "articleRef": "Artículo 4 Transitorio", "text": "Art 4 Trans", "startPage": 150, "endPage": 150},
    ]


def test_hybrid_empty_native_toc_falls_back() -> None:
    """Empty native TOC should produce same result as pure chunk-based builder."""
    rows = _multiple_transitorio_containers_rows()
    tree_chunk, stats_chunk = build_toc_tree(rows)
    tree_hybrid, stats_hybrid = build_native_toc_hybrid_tree([], rows)
    assert tree_hybrid == tree_chunk
    assert stats_hybrid == stats_chunk


def test_hybrid_unusable_native_toc_falls_back() -> None:
    """Native TOC with too few entries falls back to chunk-based."""
    rows = _multiple_transitorio_containers_rows()
    native_toc = [{"level": 1, "title": "X", "page": 1}]
    tree_chunk, _ = build_toc_tree(rows)
    tree_hybrid, _ = build_native_toc_hybrid_tree(native_toc, rows)
    assert tree_hybrid == tree_chunk


def test_hybrid_single_native_transitorio_collapses_containers() -> None:
    """When native TOC has 1 'TRANSITORIOS' but chunks have 3 containers, collapse to 1."""
    rows = _multiple_transitorio_containers_rows()
    native_toc = [
        {"level": 1, "title": "Artículo_1", "page": 2},
        {"level": 1, "title": "Artículo_2", "page": 3},
        {"level": 1, "title": "Artículo_3", "page": 4},
        {"level": 1, "title": "TRANSITORIOS", "page": 304},
        {"level": 2, "title": "Artículo_1ro", "page": 305},
    ]
    tree, stats = build_native_toc_hybrid_tree(native_toc, rows)

    def _find_transitorio_containers(nodes):
        found = []
        for n in nodes:
            if n.get("level") == 4 and "transitorio" in n.get("title", "").lower():
                found.append(n)
            for c in n.get("children") or []:
                found.extend(_find_transitorio_containers([c]))
        return found

    containers = _find_transitorio_containers(tree)
    assert len(containers) == 1, f"Expected 1 container, got {len(containers)}"
    container = containers[0]
    assert container["title"] == "Transitorios"
    assert container["target"]["page"] == 304

    children = container.get("children") or []
    child_titles = [c["title"] for c in children]
    assert "Artículo 1 Transitorio" in child_titles
    assert "Artículo 2 Transitorio" in child_titles
    assert "Artículo 3 Transitorio" in child_titles
    assert "Artículo 4 Transitorio" in child_titles


def test_hybrid_multiple_native_transitorios_distributes_children() -> None:
    """When native TOC has 2 transitorio entries, children are distributed by page."""
    rows = _multiple_transitorio_containers_rows()
    native_toc = [
        {"level": 1, "title": "Art 1", "page": 2},
        {"level": 1, "title": "Art 2", "page": 3},
        {"level": 1, "title": "Art 3", "page": 4},
        {"level": 1, "title": "TRANSITORIOS", "page": 50},
        {"level": 1, "title": "TRANSITORIOS_DE_DECRETOS_DE_REFORMA", "page": 100},
    ]
    tree, _ = build_native_toc_hybrid_tree(native_toc, rows)

    def _find_containers(nodes):
        found = []
        for n in nodes:
            if n.get("level") == 4 and "transitorio" in n.get("title", "").lower():
                found.append(n)
            for c in n.get("children") or []:
                found.extend(_find_containers([c]))
        return found

    containers = _find_containers(tree)
    assert len(containers) == 2

    first_children = [c["title"] for c in (containers[0].get("children") or [])]
    second_children = [c["title"] for c in (containers[1].get("children") or [])]

    assert "Artículo 1 Transitorio" in first_children
    assert "Artículo 2 Transitorio" in first_children
    assert "Artículo 3 Transitorio" in second_children
    assert "Artículo 4 Transitorio" in second_children


def test_hybrid_preserves_structural_nodes() -> None:
    """Hybrid builder preserves non-transitorio structural nodes from chunks."""
    rows = _multiple_transitorio_containers_rows()
    native_toc = [
        {"level": 1, "title": "A1", "page": 1},
        {"level": 1, "title": "A2", "page": 2},
        {"level": 1, "title": "A3", "page": 3},
        {"level": 1, "title": "TRANSITORIOS", "page": 304},
    ]
    tree, _ = build_native_toc_hybrid_tree(native_toc, rows)

    root_titles = [n["title"] for n in tree]
    assert "Título Primero" in root_titles


def test_hybrid_manifest_version_uses_hybrid_prefix() -> None:
    """Hybrid strategy should produce manifestVersion with native_toc_hybrid_v1 prefix."""
    tree = [{"id": "chunk-1", "title": "X", "level": 1, "target": {"page": 1}}]
    v = manifest_version_from_toc("abc" * 20, tree, STRATEGY_NATIVE_HYBRID)
    assert v.startswith("native_toc_hybrid_v1-")


def test_manifest_version_different_per_strategy() -> None:
    """Same tree with different strategies produces different versions."""
    tree = [{"id": "chunk-1", "title": "X", "level": 1, "target": {"page": 1}}]
    v_chunk = manifest_version_from_toc("abc" * 20, tree, STRATEGY_CHUNK_BASED)
    v_hybrid = manifest_version_from_toc("abc" * 20, tree, STRATEGY_NATIVE_HYBRID)
    assert v_chunk != v_hybrid
    assert v_chunk.startswith("hierarchical_chunk_based_v1-")
    assert v_hybrid.startswith("native_toc_hybrid_v1-")


def test_manifest_version_hybrid_stable() -> None:
    """Hybrid manifest version is stable for the same tree."""
    tree = [{"id": "chunk-1", "title": "X", "level": 1, "target": {"page": 1}}]
    v1 = manifest_version_from_toc("abc" * 20, tree, STRATEGY_NATIVE_HYBRID)
    v2 = manifest_version_from_toc("abc" * 20, tree, STRATEGY_NATIVE_HYBRID)
    assert v1 == v2


def test_native_toc_stats_empty() -> None:
    assert native_toc_stats([]) == {"entries": 0, "transitorioSections": 0}


def test_native_toc_stats_with_transitorios() -> None:
    toc = [
        {"level": 1, "title": "Art 1", "page": 1},
        {"level": 1, "title": "TRANSITORIOS", "page": 50},
        {"level": 1, "title": "Artículos Transitorios", "page": 100},
    ]
    stats = native_toc_stats(toc)
    assert stats["entries"] == 3
    assert stats["transitorioSections"] == 2


def test_hybrid_no_collapse_when_chunk_count_matches_native() -> None:
    """No collapsing when chunk tree has same or fewer containers than native."""
    rows = [
        {"chunkNo": 1, "chunkType": "transitorio", "heading": "TRANSITORIOS",
         "articleRef": None, "text": "TRANSITORIOS", "startPage": 50, "endPage": 50},
        {"chunkNo": 2, "chunkType": "article", "heading": "Artículo 1 Transitorio",
         "articleRef": "Artículo 1 Transitorio", "text": "Art 1 Trans", "startPage": 50, "endPage": 50},
    ]
    native_toc = [
        {"level": 1, "title": "A1", "page": 1},
        {"level": 1, "title": "A2", "page": 2},
        {"level": 1, "title": "TRANSITORIOS", "page": 50},
    ]
    tree_chunk, _ = build_toc_tree(rows)
    tree_hybrid, _ = build_native_toc_hybrid_tree(native_toc, rows)

    def _count_containers(nodes):
        count = 0
        for n in nodes:
            if n.get("level") == 4 and "transitorio" in n.get("title", "").lower():
                count += 1
            for c in n.get("children") or []:
                count += _count_containers([c])
        return count

    assert _count_containers(tree_chunk) == _count_containers(tree_hybrid)


# ---------------------------------------------------------------------------
# Tests: editorial note heading defense
# ---------------------------------------------------------------------------


def test_editorial_note_heading_excluded_from_toc() -> None:
    """Heading that looks like an editorial note → not included in TOC."""
    rows = [
        {
            "chunkNo": 1,
            "chunkType": "title",
            "heading": "Título Primero",
            "articleRef": None,
            "text": "Título Primero\nfoo",
            "startPage": 1,
            "endPage": 1,
        },
        {
            "chunkNo": 2,
            "chunkType": "section",
            "heading": "ACLARACIÓN: texto corregido conforme a Fe de Erratas",
            "articleRef": None,
            "text": "ACLARACIÓN: ...",
            "startPage": 306,
            "endPage": 306,
        },
        {
            "chunkNo": 3,
            "chunkType": "article",
            "heading": "Artículo 307",
            "articleRef": "Artículo 307",
            "text": "Artículo 307. Texto.",
            "startPage": 307,
            "endPage": 307,
        },
    ]
    tree, stats = build_toc_tree(rows)
    all_titles = _collect_titles(tree)
    assert "ACLARACIÓN" not in " ".join(all_titles)
    assert "Artículo 307" in all_titles


def test_fe_de_erratas_heading_excluded() -> None:
    """Heading starting with 'Fe de erratas' → excluded from TOC."""
    rows = [
        {
            "chunkNo": 1,
            "chunkType": "chapter",
            "heading": "Fe de erratas al artículo publicada en el DOF",
            "articleRef": None,
            "text": "Fe de erratas...",
            "startPage": 100,
            "endPage": 100,
        },
    ]
    tree, stats = build_toc_tree(rows)
    assert stats["totalNodes"] == 0


def test_nota_de_erratas_heading_excluded() -> None:
    """Heading with 'Nota de erratas' → excluded from TOC."""
    rows = [
        {
            "chunkNo": 1,
            "chunkType": "title",
            "heading": "Nota de erratas publicada en el DOF el 15 de marzo",
            "articleRef": None,
            "text": "Nota de erratas...",
            "startPage": 200,
            "endPage": 200,
        },
    ]
    tree, stats = build_toc_tree(rows)
    assert stats["totalNodes"] == 0


def test_real_structural_heading_not_excluded() -> None:
    """Regular structural headings are NOT excluded by editorial note filter."""
    rows = [
        {
            "chunkNo": 1,
            "chunkType": "title",
            "heading": "Título Segundo",
            "articleRef": None,
            "text": "Título Segundo\nDe los derechos",
            "startPage": 10,
            "endPage": 10,
        },
        {
            "chunkNo": 2,
            "chunkType": "chapter",
            "heading": "Capítulo I",
            "articleRef": None,
            "text": "Capítulo I\nDisposiciones generales",
            "startPage": 11,
            "endPage": 11,
        },
    ]
    tree, stats = build_toc_tree(rows)
    assert stats["totalNodes"] == 2


def _collect_titles(nodes: list) -> list[str]:
    """Flatten all titles in a TOC tree."""
    result: list[str] = []
    for n in nodes:
        result.append(n.get("title", ""))
        for c in n.get("children") or []:
            result.extend(_collect_titles([c]))
    return result


# ---------------------------------------------------------------------------
# Tests: boxed_note chunk type excluded from TOC
# ---------------------------------------------------------------------------


def test_boxed_note_chunk_type_excluded_from_toc() -> None:
    """Chunks with chunkType='boxed_note' never enter the TOC tree."""
    rows = [
        {
            "chunkNo": 1,
            "chunkType": "article",
            "heading": "Artículo 306",
            "articleRef": "Artículo 306",
            "text": "Artículo 306. Disposiciones.",
            "startPage": 305,
            "endPage": 305,
        },
        {
            "chunkNo": 2,
            "chunkType": "boxed_note",
            "heading": "ACLARACIÓN: nota editorial",
            "articleRef": "Artículo 306",
            "text": "ACLARACIÓN: ...",
            "startPage": 306,
            "endPage": 306,
        },
        {
            "chunkNo": 3,
            "chunkType": "article",
            "heading": "Artículo 307",
            "articleRef": "Artículo 307",
            "text": "Artículo 307. Texto.",
            "startPage": 307,
            "endPage": 307,
        },
    ]
    tree, stats = build_toc_tree(rows)
    all_titles = _collect_titles(tree)
    assert "ACLARACIÓN" not in " ".join(all_titles)
    assert "Artículo 306" in all_titles
    assert "Artículo 307" in all_titles
    assert stats["totalNodes"] == 2


def test_nota_aclaratoria_heading_excluded() -> None:
    """Heading with 'NOTA ACLARATORIA' → excluded from TOC."""
    rows = [
        {
            "chunkNo": 1,
            "chunkType": "section",
            "heading": "NOTA ACLARATORIA: Se corrige el artículo 5",
            "articleRef": None,
            "text": "NOTA ACLARATORIA...",
            "startPage": 50,
            "endPage": 50,
        },
    ]
    tree, stats = build_toc_tree(rows)
    assert stats["totalNodes"] == 0


def test_aviso_heading_excluded() -> None:
    """Heading starting with 'AVISO' → excluded from TOC."""
    rows = [
        {
            "chunkNo": 1,
            "chunkType": "chapter",
            "heading": "AVISO de modificación al artículo",
            "articleRef": None,
            "text": "AVISO...",
            "startPage": 75,
            "endPage": 75,
        },
    ]
    tree, stats = build_toc_tree(rows)
    assert stats["totalNodes"] == 0
