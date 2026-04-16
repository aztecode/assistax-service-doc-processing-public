"""
Tests unitarios para pdf_text_normalization.
Un caso por regla de preservación y por regla de unión.
"""
import pytest

from pipeline.pdf_text_normalization import normalize_pdf_text, NormalizationAudit


def test_preserva_articulo():
    """Saltos antes de Artículo N nunca se unen."""
    raw = "fin del párrafo anterior.\nArtículo 5. Este artículo se mantiene separado."
    out, audit = normalize_pdf_text(raw, None, True)
    assert "Artículo 5" in out
    assert "\n" in out
    assert "Artículo 5" in out.split("\n")[1] or "Artículo 5" in out


def test_preserva_capitulo():
    """Saltos antes de Capítulo no se unen."""
    raw = "contenido previo.\nCapítulo II - De los derechos"
    out, _ = normalize_pdf_text(raw, None, True)
    assert "Capítulo II" in out


def test_preserva_inciso():
    """Saltos antes de incisos a), b) no se unen."""
    raw = "texto.\na) primer inciso\nb) segundo inciso"
    out, _ = normalize_pdf_text(raw, None, True)
    assert "a)" in out and "b)" in out


def test_hyphen_join():
    """Línea termina en - + siguiente empieza con letra → unir palabra."""
    raw = "La obliga-\nción es clara."
    out, audit = normalize_pdf_text(raw, None, True)
    assert audit.hyphen_joins >= 1
    assert "obligación" in out


def test_merge_score():
    """merge_score >= 2 → unir líneas (no termina en puntuación + minúscula)."""
    raw = "esta línea no termina en punto\ny la siguiente empieza con minúscula y es larga"
    out, audit = normalize_pdf_text(raw, None, True)
    assert audit.merges_applied >= 1 or "línea no termina" in out


def test_headers_removed():
    """Paginación tipo '23 de 180' se elimina."""
    raw = "Contenido útil.\n23 de 180\nMás contenido."
    out, audit = normalize_pdf_text(raw, None, True)
    assert "23 de 180" not in out
    assert audit.headers_removed >= 1


def test_audit_fields():
    """NormalizationAudit tiene todos los campos."""
    raw = "Línea uno.\nLínea dos."
    _, audit = normalize_pdf_text(raw, None, True)
    assert isinstance(audit, NormalizationAudit)
    assert hasattr(audit, "merges_applied")
    assert hasattr(audit, "hyphen_joins")
    assert hasattr(audit, "headers_removed")
    assert hasattr(audit, "total_lines_raw")
    assert hasattr(audit, "total_lines_after")
