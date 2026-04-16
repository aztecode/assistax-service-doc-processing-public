"""Unit tests for decree masthead/rubric heading resolution."""

from pipeline.decreto_heading import (
    extract_decreto_por_el_que_rubric,
    first_substantive_line_for_heading,
    heading_for_generic_chunk,
    is_decreto_context,
)
from pipeline.metadata_extractor import extract_law_name


def test_is_decreto_from_title() -> None:
    assert is_decreto_context("DECRETO por estímulos fiscales", "") is True
    assert is_decreto_context("Ley del ISR", "") is False


def test_is_decreto_from_text_head_without_title_keyword() -> None:
    head = "PRESIDENCIA DE LA REPÚBLICA\nDECRETO por el que se reforman diversas disposiciones."
    assert is_decreto_context("Normativa fiscal 2025", head) is True


def test_first_substantive_skips_masthead() -> None:
    raw = (
        "(Edición Vespertina)\n"
        "DIARIO OFICIAL Lunes 30 de junio de 2025\n"
        "PODER EJECUTIVO\n"
        "PRESIDENCIA DE LA REPÚBLICA DECRETO por el que se otorgan estímulos."
    )
    line = first_substantive_line_for_heading(raw)
    assert "PRESIDENCIA" in line
    assert "DECRETO" in line
    assert "Edición Vespertina" not in line


def test_heading_generic_decreto_uses_rubric_sentence_even_when_long() -> None:
    part = (
        "(Edición Vespertina)\n"
        "PRESIDENCIA DE LA REPÚBLICA DECRETO por el que se otorgan estímulos fiscales "
        "a los contribuyentes que se indican en materia de derechos por servicios migratorios. "
        "Al margen un sello con el Escudo Nacional."
    )
    title = "Decreto estímulos derechos servicios migratorios DOF 30062025"
    h = heading_for_generic_chunk(part, title, True)
    assert h.lower().startswith("decreto por el que")
    assert "migratorios." in h or "migratorios" in h


def test_extract_rubric_from_merged_dof_header_line() -> None:
    merged = (
        "DIARIO OFICIAL Miércoles 31 de diciembre de 2025 "
        "DECRETO por el que se modifica el diverso que otorga el subsidio para el empleo. "
        "Al margen un sello con el Escudo Nacional, que dice: Estados Unidos Mexicanos."
    )
    r = extract_decreto_por_el_que_rubric(merged)
    assert r.lower().startswith("decreto por el que")
    assert "subsidio para el empleo." in r


def test_heading_non_decreto_unchanged() -> None:
    part = "Primera línea del capítulo.\nSegunda."
    h = heading_for_generic_chunk(part, "Ley de aduanas", False)
    assert h == "Primera línea del capítulo."


def test_heading_mastil_only_chunk_uses_document_title() -> None:
    part = "(Edición Vespertina)\n"
    title = "Decreto estímulos DOF 30062025"
    h = heading_for_generic_chunk(part, title, True)
    assert h == title


def test_first_substantive_skips_date_que_preamble() -> None:
    raw = (
        "Lunes 30 de junio de 2025 (Edición Vespertina) Que la participación del sector.\n"
        "Segunda línea con el detalle operativo del decreto."
    )
    line = first_substantive_line_for_heading(raw)
    assert line.startswith("Segunda línea")


def test_heading_skips_dof_date_que_single_line() -> None:
    part = (
        "Lunes 30 de junio de 2025 (Edición Vespertina) Que la participación de las empresas "
        "resulta de suma importancia en la promoción del país como destino turístico."
    )
    title = "Decreto estímulos derechos servicios migratorios DOF 30062025"
    h = heading_for_generic_chunk(part, title, True)
    assert h == extract_law_name(title)


def test_heading_decreto_long_first_line_uses_law_name() -> None:
    part = "Z" * 250 + "\nLínea corta siguiente."
    title = "Decreto corto DOF 30062025"
    h = heading_for_generic_chunk(part, title, True)
    assert h == extract_law_name(title)


def test_heading_skips_sentence_continuation_otorga_decreto() -> None:
    part = (
        "otorga el subsidio para el empleo, publicado en el Diario Oficial de la Federación.\n"
        "Párrafo siguiente con más texto operativo."
    )
    title = "Decreto modifica subsidio empleo DOF 31122025"
    h = heading_for_generic_chunk(part, title, True)
    assert h.startswith("Párrafo siguiente")


def test_heading_skips_sentence_continuation_fundamento_decreto() -> None:
    part = (
        "fundamento en los artículos 31 de la Ley Orgánica de la Administración Pública Federal.\n"
        "Texto del cuerpo que sigue al fundamento."
    )
    title = "Decreto ejemplo DOF 31122025"
    h = heading_for_generic_chunk(part, title, True)
    assert h.startswith("Texto del cuerpo")


def test_heading_skips_sentence_continuation_considera_decreto() -> None:
    part = (
        "considera adecuado que la autoridad proceda conforme a la ley.\n"
        "Párrafo siguiente con más texto operativo."
    )
    title = "Decreto ejemplo DOF 01012026"
    h = heading_for_generic_chunk(part, title, True)
    assert h.startswith("Párrafo siguiente")


def test_heading_skips_sentence_continuation_non_decreto() -> None:
    part = "considera procedente la medida.\nSegunda línea del fragmento."
    title = "Ley de aduanas"
    h = heading_for_generic_chunk(part, title, False)
    assert h.startswith("Segunda línea")


def test_heading_only_considera_uses_law_name() -> None:
    part = "considera adecuado el procedimiento administrativo."
    title = "Ley Orgánica de Ejemplo"
    h = heading_for_generic_chunk(part, title, False)
    assert h == extract_law_name(title)
