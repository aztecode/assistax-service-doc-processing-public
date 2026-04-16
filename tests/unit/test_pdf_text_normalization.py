"""
Tests unitarios para pdf_text_normalization.
Plan: merge artículo+letra, exclusión leyendas reforma, header_threshold.
"""
import pytest

from pipeline.pdf_text_normalization import normalize_pdf_text


def test_merge_articulo_9o_con_letra_a():
    """ARTÍCULO 9o.- + A. (línea sola) → una línea ARTÍCULO 9o.-A."""
    raw = "ARTÍCULO 9o.-\nA.\nLa disposición aplica a todos los casos."
    text, _ = normalize_pdf_text(raw, None, True)
    assert "9o.-A." in text or "9o.-A" in text
    assert text.count("\n") < raw.count("\n")


def test_merge_articulo_14_con_letra_a():
    """Artículo 14- + A. → Artículo 14-A."""
    raw = "Artículo 14-\nA. Texto del inciso A."
    text, _ = normalize_pdf_text(raw, None, True)
    assert "Artículo 14-A." in text or "14-A" in text


def test_no_merge_letra_a_sin_encabezado_articulo():
    """A. no precedida por encabezado de artículo → no unir."""
    raw = "Alguno.\nA. Siguiente párrafo."
    text, _ = normalize_pdf_text(raw, None, True)
    assert "Alguno.A." not in text
    assert "A." in text


def test_leyenda_reforma_no_eliminada_por_repeticion():
    """Leyenda 'Artículo adicionado DOF...' repetida no se elimina por repetición."""
    legend = "Artículo adicionado DOF 09-12-2013"
    raw = "\n".join([legend] * 5 + ["\nArtículo 5. Contenido del artículo."])
    text, audit = normalize_pdf_text(raw, None, True)
    assert legend in text
    assert audit.headers_removed == 0 or legend in text


def test_articulo_9o_ya_unido_sin_cambio():
    """ARTÍCULO 9o.-A. ya unido → sin cambio."""
    raw = "ARTÍCULO 9o.-A. La disposición aplica."
    text, _ = normalize_pdf_text(raw, None, True)
    assert "ARTÍCULO 9o.-A." in text


def test_nota_reforma_no_se_une_con_articulo_siguiente():
    """Nota de reforma + Artículo 5o en líneas separadas → no unir."""
    raw = (
        "Reforma DOF 24-12-2007: Derogó del artículo los entonces párrafos décimo tercero y décimo cuarto\n"
        "Artículo 5o.- Tratándose de los servicios que a continuación se enumeran."
    )
    text, _ = normalize_pdf_text(raw, None, True)
    # Debe conservar el salto: Artículo 5o en línea propia
    assert "Artículo 5o" in text
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    # Debe haber al menos 2 líneas (reforma y artículo)
    assert len(lines) >= 2
    # La línea del artículo debe empezar con Artículo 5o
    article_line = next((l for l in lines if "Artículo 5o" in l or "5o.-" in l), None)
    assert article_line is not None


def test_header_deduplication_first_occurrence_kept():
    """Header on first 'page' kept, on second 'page' removed."""
    header = "CÁMARA DE DIPUTADOS DEL H. CONGRESO DE LA UNIÓN"
    page1 = f"{header}\nSecretaría General\nArtículo 1.- Contenido."
    page2 = f"{header}\nSecretaría General\nArtículo 2.- Más contenido."
    seen: set[str] = set()
    text1, _ = normalize_pdf_text(page1, seen, True)
    text2, _ = normalize_pdf_text(page2, seen, True)
    assert header in text1
    assert "Artículo 1" in text1
    assert header not in text2
    assert "Artículo 2" in text2


def test_reform_legend_never_removed_with_seen_headers():
    """Reform legend kept even when using seen_headers mode."""
    legend = "Artículo reformado DOF 09-12-2013"
    seen: set[str] = set()
    text1, _ = normalize_pdf_text(legend + "\nArt. 1.", seen, True)
    text2, _ = normalize_pdf_text(legend + "\nArt. 2.", seen, True)
    assert legend in text1
    assert legend in text2


def test_collapse_blank_articulos_split():
    """DOF-style blank between 'artí' and 'culos' → single flow."""
    raw = "fundamento en los artí\n\nculos 31 de la Ley Orgánica de la Administración Pública Federal."
    text, _ = normalize_pdf_text(raw, None, True)
    assert "artículos" in text.lower()


def test_collapse_blank_con_on_split():
    """Blank between '… c' and 'on fines…' → merged."""
    raw = (
        "la mayoría de los turistas extranjeros que arriban a territorio mexicano c\n\n"
        "on fines recreativos en embarcaciones que prestan servicios de cruceros."
    )
    text, _ = normalize_pdf_text(raw, None, True)
    assert "con fines" in text.lower()
    assert "mexicano c\n\non" not in text


def test_no_collapse_blank_before_considerando():
    """Paragraph break before CONSIDERANDO must stay."""
    raw = "así lo estimó necesario.\n\nCONSIDERANDO Que la Ley Federal establece."
    text, _ = normalize_pdf_text(raw, None, True)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    assert any(l.upper().startswith("CONSIDERANDO") for l in lines)
    assert "así lo estimó necesario" in text


def test_blank_collapse_disabled_keeps_split():
    raw = "fundamento en los artí\n\nculos 31 de la Ley."
    text_on, _ = normalize_pdf_text(raw, None, True)
    text_off, _ = normalize_pdf_text(raw, None, False)
    assert "culos" in text_off
    assert "artículos" in text_on.lower()


def test_collapse_blank_comma_se_then_considera():
    """Blank between '…, se' and 'considera…' → one paragraph (DOF bold/body split)."""
    raw = (
        "En estos casos, se\n\n"
        "considera adecuado que la autoridad administrativa proceda conforme a la ley."
    )
    text, _ = normalize_pdf_text(raw, None, True)
    flat = " ".join(text.split())
    assert "se considera" in flat.lower()


def test_collapse_blank_decreto_que_otorga_continuation():
    """Blank between '… diverso que' and 'otorga…' → collapse (verb continuation, DOF split)."""
    raw = (
        "DECRETO por el que se modifica el diverso que\n\n"
        "otorga el subsidio para el empleo, publicado en el Diario Oficial de la Federación con texto suficiente."
    )
    text, _ = normalize_pdf_text(raw, None, True)
    flat = " ".join(text.split())
    assert "que otorga" in flat.lower()


def test_dof_preserves_newlines_before_decreto_and_al_margen():
    """Score-merge must not glue masthead + rubric + marginal note into one line."""
    raw = (
        "DIARIO OFICIAL \n"
        "Miércoles 31 de diciembre de 2025 \n"
        "DECRETO por el que se modifica el diverso que otorga el subsidio para el empleo. \n"
        "Al margen un sello con el Escudo Nacional, que dice: Estados Unidos Mexicanos.- "
        "Presidencia de la República. \n"
        "CLAUDIA SHEINBAUM PARDO, Presidenta de los Estados Unidos Mexicanos, en ejercicio "
        "de la facultad \n"
        "que me confiere el artículo 89, fracción I, de la Constitución Política de los "
        "Estados Unidos Mexicanos, con \n"
        "fundamento en los artículos 31 de la Ley Orgánica de la Administración Pública "
        "Federal y 39, primer párrafo, \n"
        "fracción III, del Código Fiscal de la Federación, y \n"
        "CONSIDERANDO \n"
        "Que el Plan Nacional de Desarrollo 2025-2030 establece."
    )
    text, _ = normalize_pdf_text(raw, None, True)
    assert "DIARIO OFICIAL" in text
    assert "DECRETO por el que se modifica el diverso que otorga el subsidio para el empleo." in text
    assert "Al margen un sello" in text
    # Rubric and marginal note must not sit on the same line (was one string in broken UI).
    assert "empleo.\nAl margen" in text or "empleo.\r\nAl margen" in text
    assert "República.\nCLAUDIA" in text or "República.\r\nCLAUDIA" in text


def test_skip_blank_merge_when_collapse_disabled_se_considera():
    """Main-loop skip-blank merge still joins when blank collapse flag is off."""
    raw = (
        "En estos casos, se\n\n"
        "considera adecuado que la autoridad administrativa proceda conforme a la ley."
    )
    text, _ = normalize_pdf_text(raw, None, False)
    flat = " ".join(text.split())
    assert "se considera" in flat.lower()
