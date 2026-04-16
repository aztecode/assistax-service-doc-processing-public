"""
Tests unitarios para metadata_extractor.
Verifica infer_doc_type, extract_law_name y extract_legal_legend.
Alineado con assistax-fn: claves legalLegend, publicationDate, lastReformDate.
"""
import pytest

from pipeline.metadata_extractor import (
    infer_doc_type,
    extract_law_name,
    extract_legal_legend,
)


def test_infer_doc_type_reglamento():
    """Reglamento de la Ley del ISR → reglamento (no ley)."""
    assert infer_doc_type("Reglamento de la Ley del ISR") == "reglamento"


def test_infer_doc_type_codigo():
    """Código Fiscal de la Federación → codigo."""
    assert infer_doc_type("Código Fiscal de la Federación") == "codigo"


def test_infer_doc_type_codigo_sin_tilde():
    """Código sin tilde también debe detectarse."""
    assert infer_doc_type("Codigo de Comercio") == "codigo"


def test_infer_doc_type_presupuesto():
    """Presupuesto de Egresos 2024 → presupuesto."""
    assert infer_doc_type("Presupuesto de Egresos 2024") == "presupuesto"


def test_infer_doc_type_decreto():
    """Decreto → decreto (alineado con fn)."""
    assert infer_doc_type("Decreto por el que se reforma la Ley") == "decreto"


def test_infer_doc_type_acuerdo():
    """Acuerdo → acuerdo (alineado con fn)."""
    assert infer_doc_type("Acuerdo de coordinación fiscal") == "acuerdo"


def test_infer_doc_type_norma():
    """Norma o nom- → norma (alineado con fn)."""
    assert infer_doc_type("Norma Oficial Mexicana") == "norma"
    assert infer_doc_type("NOM-001-SEMARNAT") == "norma"


def test_infer_doc_type_ley():
    """Ley sin otros keywords → ley."""
    assert infer_doc_type("Ley Federal del Trabajo") == "ley"


def test_infer_doc_type_otro():
    """Sin match → otro (alineado con fn)."""
    assert infer_doc_type("Resolución por la que se emiten reglas") == "otro"
    assert infer_doc_type("Estatuto Orgánico del Tribunal") == "otro"


def test_extract_law_name_sin_parentesis():
    """Nombre sin paréntesis se retorna igual."""
    title = "Ley Federal del Trabajo"
    assert extract_law_name(title) == title


def test_extract_law_name_con_parentesis():
    """Paréntesis final y contenido se eliminan."""
    title = "Ley del ISR (reformada en 2024)"
    assert extract_law_name(title) == "Ley del ISR"


def test_extract_law_name_max_255_chars():
    """Máximo 255 caracteres."""
    long_title = "A" * 300
    result = extract_law_name(long_title)
    assert len(result) == 255


def test_extract_legal_legend_last_reform_date():
    """Extrae última reforma DOF; retorna lastReformDate en ISO."""
    content = "Última reforma DOF 15-03-2024 publicada."
    result = extract_legal_legend(content)
    assert result["lastReformDate"] == "2024-03-15"
    assert "legalLegend" in result


def test_extract_legal_legend_publication_date():
    """Extrae nueva ley publicada en DOF; retorna publicationDate en ISO."""
    content = (
        "Nueva ley publicada en el Diario Oficial de la Federación "
        "el 1 de abril de 2024."
    )
    result = extract_legal_legend(content)
    assert result["publicationDate"] == "2024-04-01"
    assert "legalLegend" in result


def test_extract_legal_legend_texto_vigente():
    """TEXTO VIGENTE se incluye en legalLegend."""
    content = "TEXTO VIGENTE. Última reforma DOF 01-01-2024."
    result = extract_legal_legend(content)
    assert "TEXTO VIGENTE" in result.get("legalLegend", "")
    assert result["lastReformDate"] == "2024-01-01"


def test_extract_legal_legend_cantidades_in_legend():
    """Cantidades actualizadas se incluyen en legalLegend."""
    content = "Cantidades actualizadas por DOF 10-05-2024."
    result = extract_legal_legend(content)
    assert "legalLegend" in result
    legend = result.get("legalLegend", "").lower()
    assert "cantidades" in legend or "10-05-2024" in result.get("legalLegend", "")
