"""
Tests unitarios para legal_chunker.
Casos edge: 194-N-1, artículo con tabla, Bis/Ter, max_chunk_chars, transitorio.
"""
import re

import pytest

from pipeline.legal_chunker import (
    Chunk,
    TRANSITORIOS_PREAMBLE_ARTICLE_REF,
    _adjust_split_for_clitic_tail,
    chunk_content,
    detect_article_context,
    split_by_legal_structure,
    _split_by_size,
)
from pipeline.pdf_extractor import PageContent, TableBlock


def test_article_194_n_1_no_romper_numero():
    """Artículo 194-N-1 no debe romperse en el número."""
    text = "Artículo 194-N-1. Las disposiciones sobre jornada aplican al trabajo nocturno."
    segments, _ = split_by_legal_structure(text)
    assert len(segments) >= 1
    seg = segments[0]
    assert "194-N-1" in seg.text
    assert seg.chunk_type == "article"


def test_article_con_fraccion_y_tabla_incrustada():
    """Artículo con fracción que contiene tabla → chunk atómico de tabla."""
    pages = [
        PageContent(
            page_number=1,
            text="Artículo 5. Los montos se calcularán conforme a la siguiente tabla:\n\na) La base gravable.",
            tables=[
                TableBlock(
                    table_index=1,
                    page_number=1,
                    markdown="[TABLE_1]\n|Concepto|Monto|\n|---|---|\n|Base|1000|\n[/TABLE_1]",
                    rows=[["Concepto", "Monto"], ["Base", "1000"]],
                    bbox=(0, 0, 100, 50),
                    is_boxed_note=False,
                )
            ],
        )
    ]
    chunks = chunk_content(pages, 2000, "", "")
    assert len(chunks) >= 1
    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert len(table_chunks) == 1
    assert table_chunks[0].article_ref == "Artículo 5"
    assert table_chunks[0].has_table is True


def test_articulo_bis_seguido_de_ter():
    """Artículo Bis seguido de Artículo Ter."""
    text = (
        "Artículo 10 Bis. Texto del artículo bis que establece criterios adicionales.\n\n"
        "Artículo 10 Ter. Texto del artículo ter con disposiciones complementarias."
    )
    segments, _ = split_by_legal_structure(text)
    article_segments = [s for s in segments if s.chunk_type == "article"]
    assert len(article_segments) >= 2
    texts = [s.text for s in article_segments]
    assert any("Bis" in t for t in texts)
    assert any("Ter" in t for t in texts)


def test_texto_excede_max_chunk_chars_con_separadores():
    """Texto largo con \\n\\n se divide respetando separadores."""
    long_text = "A" * 500 + "\n\n" + "B" * 500 + "\n\n" + "C" * 500
    parts = _split_by_size(long_text, max_chars=600)
    assert len(parts) >= 2
    for p in parts:
        assert len(p) >= 25


def test_texto_excede_max_chunk_sin_separadores():
    """Texto largo sin espacios se divide por max_chars (corte duro)."""
    long_text = "X" * 3000
    parts = _split_by_size(long_text, max_chars=500)
    assert len(parts) >= 4
    for p in parts:
        assert len(p.strip()) >= 25


def test_texto_excede_max_chunk_prefiere_limite_de_palabra():
    """Sin saltos de línea, partir en el último espacio del tramo (no a mitad de token)."""
    tokens = [f"w{i:04d}" for i in range(200)]
    text = " ".join(tokens)
    parts = _split_by_size(text, max_chars=80)
    assert len(parts) >= 2
    roundtrip: list[str] = []
    for p in parts:
        roundtrip.extend(p.split())
    assert roundtrip == tokens


def test_adjust_split_moves_back_when_chunk_would_end_on_clitic_se():
    """Evita que el primer trozo termine en pronombre/clítico suelto (p. ej. …, se)."""
    filler = " ".join([f"w{i}" for i in range(25)])
    remaining = filler + " casos, se considera largo"
    split_at = remaining.index("considera")
    max_chars = 120
    adjusted = _adjust_split_for_clitic_tail(remaining, split_at, max_chars)
    assert adjusted < split_at
    first = remaining[:adjusted].rstrip()
    assert not re.search(r"\bse\s*$", first)


def test_split_by_size_no_chunk_ends_on_lone_clitic():
    """Integración: partes intermedias no deben terminar en clítico de la lista."""
    filler = " ".join([f"p{i}" for i in range(12)])
    text = filler + " casos, se considera el texto " + "z " * 200
    parts = _split_by_size(text, max_chars=55)
    assert len(parts) >= 2
    for p in parts[:-1]:
        assert not re.search(r"\bse\s*$", p.rstrip()), p[-50:]


def test_transitorios_preamble_generic_gets_transitorios_ref():
    """Texto genérico entre cabecera TRANSITORIOS y primer ordinal lleva article_ref sintético."""
    text = (
        "Artículo 3. Disposiciones finales del decreto.\n\n"
        "TRANSITORIOS\n\n"
        "Lunes 30 de junio de 2025 (Edición Vespertina) Que se otorgan estímulos fiscales.\n\n"
        "PRIMERO.- La Secretaría publicará el acuerdo en el DOF.\n"
    )
    pages = [PageContent(page_number=1, text=text, tables=[])]
    chunks = chunk_content(pages, 2000, "Decreto prueba DOF", "")
    preamble = [c for c in chunks if "Que se otorgan" in c.text]
    assert len(preamble) >= 1
    assert preamble[0].article_ref == TRANSITORIOS_PREAMBLE_ARTICLE_REF


def test_transitorio_con_referencia_articulo():
    """Transitorio con referencia a artículo previo."""
    text = (
        "Artículo 99. Disposiciones finales.\n\n"
        "Primero.- Lo dispuesto en el artículo 5 aplicará a partir de 2025."
    )
    segments, _ = split_by_legal_structure(text)
    transitorio = [s for s in segments if s.chunk_type == "transitorio"]
    assert len(transitorio) >= 1
    full_content = " ".join(s.text for s in segments)
    assert "artículo" in full_content.lower() or "5" in full_content


def test_ningun_chunk_menor_25_chars():
    """Ningún chunk debe tener len(text) < 25."""
    pages = [
        PageContent(
            page_number=1,
            text="Artículo 1. La presente Ley es de observancia en toda la República Mexicana.",
            tables=[],
        )
    ]
    chunks = chunk_content(pages, 2000, "", "")
    for c in chunks:
        assert len(c.text) >= 25, f"Chunk con texto < 25: {c.text!r}"


def test_detect_article_context():
    """detect_article_context retorna el article_ref más reciente."""
    chunks = [
        Chunk("x" * 30, 1, "generic", None, "", 1, 1, False, None),
        Chunk("x" * 30, 2, "article", "Art. 5.", "", 1, 1, False, None),
        Chunk("x" * 30, 3, "fraction", None, "", 1, 1, False, None),
    ]
    assert detect_article_context(chunks) == "Art. 5."


def test_detect_article_context_sin_articulos():
    """detect_article_context retorna None si no hay artículos."""
    chunks = [
        Chunk("x" * 30, 1, "generic", None, "", 1, 1, False, None),
    ]
    assert detect_article_context(chunks) is None


def test_tabla_dentro_articulo_tiene_article_ref():
    """Tabla dentro de un artículo tiene article_ref del contenedor."""
    pages = [
        PageContent(
            page_number=1,
            text="Artículo 7. Los montos son los siguientes:",
            tables=[
                TableBlock(
                    table_index=1,
                    page_number=1,
                    markdown="[TABLE_1]\n|A|B|\n|---|---|\n|1|2|\n[/TABLE_1]",
                    rows=[["A", "B"], ["1", "2"]],
                    bbox=(0, 0, 50, 30),
                    is_boxed_note=False,
                )
            ],
        )
    ]
    chunks = chunk_content(pages, 2000, "", "")
    table_chunk = next((c for c in chunks if c.chunk_type == "table"), None)
    assert table_chunk is not None
    assert table_chunk.article_ref == "Artículo 7"


def test_articulo_5o_con_nota_reforma_previa():
    """Artículo 5o.- con línea previa de nota de reforma → clasificado como article."""
    text = (
        "Reforma DOF 24-12-2007: Derogó del artículo los entonces párrafos décimo tercero y décimo cuarto\n"
        "Artículo 5o.- Tratándose de los servicios que a continuación se enumeran."
    )
    segments, _ = split_by_legal_structure(text)
    article_segments = [s for s in segments if s.chunk_type == "article"]
    assert len(article_segments) >= 1, "Artículo 5o debe identificarse tras nota de reforma"
    assert article_segments[0].article_ref == "Artículo 5"


def test_articulo_9o_punto_guion_a_article_ref():
    """Artículo 9o.-A. La... → article_ref = Art. 9-A."""
    text = "Artículo 9o.-A. La disposición aplica a todos los casos."
    segments, _ = split_by_legal_structure(text)
    article_segments = [s for s in segments if s.chunk_type == "article"]
    assert len(article_segments) >= 1
    assert article_segments[0].article_ref == "Artículo 9-A"


def test_articulo_5o_numero_solo_tras_reforma():
    """Línea '5o.- Tratándose...' tras nota de reforma → clasificado como article."""
    text = (
        "Reforma DOF 24-12-2007: Derogó del artículo los entonces párrafos décimo tercero y décimo cuarto\n"
        "5o.- Tratándose de los servicios que a continuación se enumeran."
    )
    segments, _ = split_by_legal_structure(text)
    article_segments = [s for s in segments if s.chunk_type == "article"]
    assert len(article_segments) >= 1, "5o.- tras reforma debe identificarse como artículo"
    assert article_segments[0].article_ref == "Art. 5."


def test_articulo_multilinea_artículo_en_linea_previa():
    """'Artículo' en línea previa, '5o.- Tratándose...' en actual → article."""
    text = (
        "Artículo\n"
        "5o.- Tratándose de los servicios que a continuación se enumeran."
    )
    segments, _ = split_by_legal_structure(text)
    article_segments = [s for s in segments if s.chunk_type == "article"]
    assert len(article_segments) >= 1, "Artículo multi-línea debe identificarse"
    assert article_segments[0].article_ref == "Art. 5."


def test_linea_combinada_reforma_y_articulo_5o():
    """Línea con reform note + Artículo 5o en la misma línea → se divide y detecta artículo."""
    text = (
        "1989) Reforma DOF 24-12-2007: Derogó del artículo los entonces párrafos décimo tercero y décimo cuarto "
        "Artículo 5o.- Tratándose de los servicios que a continuación se enumeran."
    )
    segments, _ = split_by_legal_structure(text)
    article_segments = [s for s in segments if s.chunk_type == "article"]
    assert len(article_segments) >= 1, "Línea combinada debe dividirse y detectar Art. 5"
    assert article_segments[0].article_ref == "Artículo 5"


def test_split_by_legal_structure_con_leading_context():
    """leading_context_line='Artículo' permite detectar 5o.- como artículo."""
    text = "5o.- Tratándose de los servicios."
    segments, _ = split_by_legal_structure(text, leading_context_line="Artículo")
    assert len(segments) >= 1
    assert segments[0].chunk_type == "article"
    assert segments[0].article_ref == "Art. 5."


def test_articulo_5o_dividido_entre_paginas():
    """Artículo al final de página, 5o.- al inicio de siguiente → detectado."""
    pages = [
        PageContent(
            page_number=1,
            text="Artículo 4. Texto del artículo cuatro.\n\nArtículo",
            tables=[],
        ),
        PageContent(
            page_number=2,
            text="5o.- Tratándose de los servicios que a continuación se enumeran.",
            tables=[],
        ),
    ]
    chunks = chunk_content(pages, 2000, "", "")
    article_refs = [c.article_ref for c in chunks if c.chunk_type == "article" and c.article_ref]
    assert "Art. 5." in article_refs


# ---------------------------------------------------------------------------
# Tests: transitorios con sufijo en encabezado
# ---------------------------------------------------------------------------

def test_articulo_cuerpo_no_lleva_sufijo_transitorio():
    """Artículo antes de TRANSITORIOS no debe llevar sufijo 'Transitorio'."""
    text = (
        "Artículo 10. Disposiciones generales de la ley.\n\n"
        "El presente artículo aplica en todo el territorio nacional."
    )
    segments, _ = split_by_legal_structure(text)
    article_segs = [s for s in segments if s.chunk_type == "article"]
    assert len(article_segs) >= 1
    assert article_segs[0].article_ref == "Artículo 10"
    assert "Transitorio" not in (article_segs[0].article_ref or "")


def test_articulo_en_zona_transitoria_lleva_sufijo():
    """Artículo N dentro del bloque TRANSITORIOS → article_ref contiene 'Transitorio'."""
    text = (
        "TRANSITORIOS\n\n"
        "Artículo 1. El presente decreto entrará en vigor al día siguiente."
    )
    segments, _ = split_by_legal_structure(text)
    article_segs = [s for s in segments if s.chunk_type == "article"]
    assert len(article_segs) >= 1
    assert article_segs[0].article_ref == "Artículo 1 Transitorio"


def test_ordinal_transitorio_primero_heading():
    """'Primero.-' tras TRANSITORIOS → article_ref 'Artículo Primero Transitorio'."""
    text = (
        "TRANSITORIOS\n\n"
        "Primero.- El presente decreto entrará en vigor al día siguiente de su publicación."
    )
    segments, _ = split_by_legal_structure(text)
    # Container segment (article_ref=None) + ordinal segment (article_ref set)
    ordinal_segs = [s for s in segments if s.chunk_type == "transitorio" and s.article_ref]
    assert len(ordinal_segs) >= 1
    assert ordinal_segs[0].article_ref == "Artículo Primero Transitorio"


def test_multiples_ordinales_transitorios():
    """Primero.-, Segundo.-, Tercero.- → refs individuales con sufijo Transitorio."""
    text = (
        "TRANSITORIOS\n\n"
        "Primero.- El presente decreto entrará en vigor al día siguiente.\n\n"
        "Segundo.- Se derogan las disposiciones que se opongan al presente decreto.\n\n"
        "Tercero.- Las entidades federativas deberán adecuar su legislación."
    )
    segments, _ = split_by_legal_structure(text)
    transitorio_segs = [s for s in segments if s.chunk_type == "transitorio"]
    refs = [s.article_ref for s in transitorio_segs]
    assert "Artículo Primero Transitorio" in refs
    assert "Artículo Segundo Transitorio" in refs
    assert "Artículo Tercero Transitorio" in refs


def test_cabecera_transitorios_emite_segmento_contenedor():
    """La línea 'TRANSITORIOS' genera un segmento contenedor con chunk_type='transitorio' y article_ref=None."""
    text = (
        "TRANSITORIOS\n\n"
        "Primero.- El presente decreto entrará en vigor al día siguiente."
    )
    segments, _ = split_by_legal_structure(text)
    container_segs = [
        s for s in segments
        if s.chunk_type == "transitorio" and s.article_ref is None and s.text.strip() == "TRANSITORIOS"
    ]
    assert len(container_segs) == 1


def test_chunk_heading_ordinal_transitorio():
    """chunk_content: heading de transitorio ordinal usa article_ref sintético."""
    pages = [
        PageContent(
            page_number=1,
            text=(
                "Artículo 5. Disposiciones generales aplicables al territorio nacional.\n\n"
                "TRANSITORIOS\n\n"
                "Primero.- El presente decreto entrará en vigor al día siguiente de su publicación."
            ),
            tables=[],
        )
    ]
    chunks = chunk_content(pages, 2000, "", "")
    # Filter out the container chunk (article_ref=None); ordinal chunk has article_ref set
    ordinal_chunks = [c for c in chunks if c.chunk_type == "transitorio" and c.article_ref]
    assert len(ordinal_chunks) >= 1
    assert ordinal_chunks[0].heading == "Artículo Primero Transitorio"


def test_chunk_heading_articulo_en_transitorios():
    """chunk_content: artículo en zona transitoria tiene heading con sufijo Transitorio."""
    pages = [
        PageContent(
            page_number=1,
            text=(
                "Artículo 3. Texto del cuerpo principal de la ley que aplica en general.\n\n"
                "TRANSITORIOS\n\n"
                "Artículo 1. Vigencia a partir del día siguiente de publicación."
            ),
            tables=[],
        )
    ]
    chunks = chunk_content(pages, 2000, "", "")
    transitorio_art = [c for c in chunks if c.article_ref and "Transitorio" in c.article_ref]
    assert len(transitorio_art) >= 1
    assert transitorio_art[0].heading == "Artículo 1 Transitorio"


def test_detect_article_context_transitorio_ordinal():
    """detect_article_context reconoce transitorio ordinal con article_ref."""
    chunks = [
        Chunk("x" * 30, 1, "article", "Artículo 5", "", 1, 1, False, None),
        Chunk("x" * 30, 2, "transitorio", "Artículo Primero Transitorio", "", 1, 1, False, None),
    ]
    assert detect_article_context(chunks) == "Artículo Primero Transitorio"


def test_tabla_bajo_transitorio_ordinal_hereda_ref():
    """Tabla inmediatamente tras un transitorio ordinal hereda su article_ref."""
    pages = [
        PageContent(
            page_number=1,
            text=(
                "TRANSITORIOS\n\n"
                "Primero.- Las tasas aplicables se señalan en la siguiente tabla:"
            ),
            tables=[
                TableBlock(
                    table_index=1,
                    page_number=1,
                    markdown="[TABLE_1]\n|Tasa|%|\n|---|---|\n|Base|10|\n[/TABLE_1]",
                    rows=[["Tasa", "%"], ["Base", "10"]],
                    bbox=(0, 0, 100, 50),
                    is_boxed_note=False,
                )
            ],
        )
    ]
    chunks = chunk_content(pages, 2000, "", "")
    table_chunk = next((c for c in chunks if c.chunk_type == "table"), None)
    assert table_chunk is not None
    assert table_chunk.article_ref == "Artículo Primero Transitorio"


# ---------------------------------------------------------------------------
# Tests: variante "ARTÍCULOS TRANSITORIOS" y casos nuevos
# ---------------------------------------------------------------------------

def test_cabecera_articulos_transitorios_activa_bloque():
    """'ARTÍCULOS TRANSITORIOS' activa el bloque: artículo siguiente lleva sufijo."""
    text = (
        "Artículo 3074. Último artículo del cuerpo principal.\n\n"
        "ARTÍCULOS TRANSITORIOS\n\n"
        "Artículo 1. El presente Código comenzará a regir el 1o. de octubre de 1932."
    )
    segments, _ = split_by_legal_structure(text)
    transitorio_art = [
        s for s in segments
        if s.chunk_type == "article" and s.article_ref and "Transitorio" in s.article_ref
    ]
    assert len(transitorio_art) >= 1
    assert transitorio_art[0].article_ref == "Artículo 1 Transitorio"


def test_cabecera_articulos_transitorios_emite_contenedor():
    """'ARTÍCULOS TRANSITORIOS' emite un segmento contenedor con article_ref=None."""
    text = (
        "ARTÍCULOS TRANSITORIOS\n\n"
        "Artículo 1. Vigencia a partir de la publicación."
    )
    segments, _ = split_by_legal_structure(text)
    container_segs = [
        s for s in segments
        if s.chunk_type == "transitorio" and s.article_ref is None
        and "TRANSITORIOS" in s.text.upper()
    ]
    assert len(container_segs) == 1


def test_linea_fusionada_articulos_transitorios_se_divide():
    """'ARTÍCULOS TRANSITORIOS Artículo 1º.-' se divide y el artículo lleva sufijo."""
    text = "ARTÍCULOS TRANSITORIOS Artículo 1. El decreto entra en vigor al día siguiente."
    segments, _ = split_by_legal_structure(text)
    transitorio_art = [
        s for s in segments
        if s.article_ref and "Transitorio" in s.article_ref
    ]
    assert len(transitorio_art) >= 1
    assert "1" in transitorio_art[0].article_ref


def test_reset_bloque_transitorios_en_titulo():
    """Un Título/Capítulo/Sección después de los transitorios resetea el bloque:
    los artículos posteriores no deben llevar sufijo 'Transitorio'."""
    text = (
        "ARTÍCULOS TRANSITORIOS\n\n"
        "Artículo 1. El decreto entra en vigor al día siguiente.\n\n"
        "TÍTULO SEGUNDO\n\n"
        "Capítulo I\n\n"
        "Artículo 10. Este artículo pertenece al cuerpo principal."
    )
    segments, _ = split_by_legal_structure(text)
    article_10 = next(
        (s for s in segments if s.chunk_type == "article" and s.article_ref == "Artículo 10"),
        None,
    )
    assert article_10 is not None, "Artículo 10 no encontrado"
    assert "Transitorio" not in (article_10.article_ref or "")


# ---------------------------------------------------------------------------
# Tests: carry de estado entre páginas (regresión del bug cross-page)
# ---------------------------------------------------------------------------


def test_carry_transitorio_entre_paginas_numerico():
    """Artículo numérico en página 2 hereda sufijo cuando TRANSITORIOS estaba en página 1."""
    pages = [
        PageContent(
            page_number=1,
            text=(
                "Artículo 100. Último artículo del cuerpo principal.\n\n"
                "TRANSITORIOS\n\n"
                "Artículo 1. El presente decreto entrará en vigor al día siguiente."
            ),
            tables=[],
        ),
        PageContent(
            page_number=2,
            text="Artículo 2. Se derogan todas las disposiciones que se opongan al presente decreto.",
            tables=[],
        ),
    ]
    chunks = chunk_content(pages, 2000, "", "")
    art2 = next(
        (c for c in chunks if c.article_ref and "Artículo 2" in c.article_ref),
        None,
    )
    assert art2 is not None, "No se encontró chunk con 'Artículo 2'"
    assert "Transitorio" in (art2.article_ref or ""), (
        f"Artículo 2 en página 2 debería llevar sufijo 'Transitorio', ref: {art2.article_ref!r}"
    )


def test_carry_transitorio_return_state_true():
    """split_by_legal_structure retorna inside_transitorios=True al terminar en un bloque."""
    text = "TRANSITORIOS\n\nArtículo 1. Vigencia a partir del día siguiente."
    _, carry = split_by_legal_structure(text)
    assert carry is True


def test_carry_transitorio_return_state_false_sin_bloque():
    """split_by_legal_structure retorna inside_transitorios=False cuando no hay bloque."""
    text = "Artículo 5. Disposiciones generales aplicables en todo el territorio."
    _, carry = split_by_legal_structure(text)
    assert carry is False


def test_carry_transitorio_con_reset_por_titulo():
    """Carry es False si el bloque se cerró por un Capítulo/Título (con numeral) antes de fin de página."""
    text = (
        "TRANSITORIOS\n\n"
        "Artículo 1. Vigencia.\n\n"
        "Capítulo I\n\n"
        "Artículo 10. Contenido posterior."
    )
    _, carry = split_by_legal_structure(text)
    assert carry is False


def test_carry_multiple_paginas_reset_por_capitulo():
    """El carry se apaga en página 2 si aparece un Capítulo (numeral válido), y la página 3 queda limpia."""
    pages = [
        PageContent(
            page_number=1,
            text="TRANSITORIOS\n\nArtículo 1. El presente decreto entrará en vigor.",
            tables=[],
        ),
        PageContent(
            page_number=2,
            text=(
                "Artículo 2. Se derogan disposiciones anteriores.\n\n"
                "Capítulo III\n\n"
                "Artículo 50. Inicio del nuevo capítulo."
            ),
            tables=[],
        ),
        PageContent(
            page_number=3,
            text="Artículo 51. Continuación del Capítulo III.",
            tables=[],
        ),
    ]
    chunks = chunk_content(pages, 2000, "", "")
    art51 = next(
        (c for c in chunks if c.article_ref and "Artículo 51" in c.article_ref),
        None,
    )
    assert art51 is not None, "No se encontró chunk con 'Artículo 51'"
    assert "Transitorio" not in (art51.article_ref or ""), (
        f"Artículo 51 no debe llevar sufijo 'Transitorio', ref: {art51.article_ref!r}"
    )


# ---------------------------------------------------------------------------
# Tests: boxed note classification in chunker
# ---------------------------------------------------------------------------


def test_boxed_note_table_gets_boxed_note_chunk_type():
    """TableBlock with is_boxed_note=True → chunk_type='boxed_note'."""
    pages = [
        PageContent(
            page_number=1,
            text="Artículo 306. Disposiciones sobre publicidad farmacéutica.",
            tables=[
                TableBlock(
                    table_index=1,
                    page_number=1,
                    markdown="[TABLE_1]\n| Col 1 |\n|---|\n| ACLARACIÓN: nota editorial |\n[/TABLE_1]",
                    rows=[["ACLARACIÓN: nota editorial"]],
                    bbox=(0, 200, 500, 300),
                    is_boxed_note=True,
                )
            ],
        )
    ]
    chunks = chunk_content(pages, 2000, "", "")
    boxed = [c for c in chunks if c.chunk_type == "boxed_note"]
    assert len(boxed) == 1
    tables = [c for c in chunks if c.chunk_type == "table"]
    assert len(tables) == 0


def test_regular_table_keeps_table_chunk_type():
    """TableBlock with is_boxed_note=False → chunk_type='table'."""
    pages = [
        PageContent(
            page_number=1,
            text="Artículo 5. Los montos son los siguientes:",
            tables=[
                TableBlock(
                    table_index=1,
                    page_number=1,
                    markdown="[TABLE_1]\n|A|B|\n|---|---|\n|1|2|\n[/TABLE_1]",
                    rows=[["A", "B"], ["1", "2"]],
                    bbox=(0, 0, 100, 50),
                    is_boxed_note=False,
                )
            ],
        )
    ]
    chunks = chunk_content(pages, 2000, "", "")
    tables = [c for c in chunks if c.chunk_type == "table"]
    assert len(tables) == 1
    boxed = [c for c in chunks if c.chunk_type == "boxed_note"]
    assert len(boxed) == 0


def test_editorial_note_line_classified_as_generic():
    """Lines starting with editorial note prefixes → generic, not structural."""
    text = (
        "ACLARACIÓN: El texto se ajustó conforme a fe de erratas publicada en el DOF.\n\n"
        "Artículo 307. Disposiciones siguientes."
    )
    segments, _ = split_by_legal_structure(text)
    first_seg = segments[0]
    assert first_seg.chunk_type == "generic"
    assert "ACLARACIÓN" in first_seg.text


def test_fe_de_erratas_line_classified_as_generic():
    """'Fe de erratas' at start of text → generic, never structural."""
    text = "Fe de erratas publicada en el DOF el 15 de marzo de 2020 al artículo 306."
    segments, _ = split_by_legal_structure(text)
    assert len(segments) == 1
    assert segments[0].chunk_type == "generic"
