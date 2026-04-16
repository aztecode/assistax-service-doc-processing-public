"""
Fixtures compartidas para tests.
Crea PDFs mínimos en memoria para tests del extractor.
"""
import fitz
import pytest


@pytest.fixture
def pdf_bytes_ley_corta() -> bytes:
    """PDF mínimo con texto tipo ley federal."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "LEY FEDERAL DEL TRABAJO")
    page.insert_text((72, 100), "Artículo 1. La presente Ley es de observancia en toda la República.")
    page.insert_text((72, 120), "Artículo 2. Las normas de trabajo tienden a conseguir el equilibrio.")
    page.insert_text((72, 140), "Capítulo I - Disposiciones generales")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def pdf_bytes_con_tablas() -> bytes:
    """PDF con tabla tarifaria (tasa, cuota, monto)."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "TARIFA DE CUOTAS")
    # Tabla simple: PyMuPDF find_tables puede o no detectarla según estructura
    page.insert_text((72, 100), "Concepto")
    page.insert_text((200, 100), "Tasa")
    page.insert_text((280, 100), "Monto")
    page.insert_text((72, 120), "1. Base general")
    page.insert_text((200, 120), "16%")
    page.insert_text((280, 120), "1,500.00")
    page.insert_text((72, 140), "Artículo 5. Lo no previsto en esta ley...")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def pdf_bytes_corrupto() -> bytes:
    """Bytes que no son un PDF válido."""
    return b"No soy un PDF, solo texto basura \x00\x01\x02"


@pytest.fixture
def pdf_bytes_vacio() -> bytes:
    """PDF con una página vacía (documento mínimo válido)."""
    doc = fitz.open()
    doc.new_page()
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes
