"""
Tests de integración para el pipeline Fase 4.
Requieren PostgreSQL con pgvector y unaccent. Ejecutar schema_test.sql antes.

Ejecutar con: pytest tests/integration/ -v --tb=short
Variables: DATABASE_URL (default: postgresql://postgres:postgres@localhost:5432/assistax)
"""
import os
import uuid
from unittest.mock import patch, MagicMock

import pytest

# PDF mínimo válido (1 página con texto)
_MINIMAL_PDF_BYTES = (
    b"%PDF-1.4\n1 0 obj\n<<\n/Type/Catalog\n/Pages 2 0 R\n>>\nendobj\n"
    b"2 0 obj\n<<\n/Type/Pages\n/Kids[3 0 R]\n/Count 1\n>>\nendobj\n"
    b"3 0 obj\n<<\n/Type/Page\n/Parent 2 0 R\n/MediaBox[0 0 612 792]\n"
    b"/Contents 4 0 R\n>>\nendobj\n"
    b"4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n"
    b"100 700 Td\n(Articulo 1.) Tj\nET\nendstream\nendobj\n"
    b"xref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n"
    b"0000000115 00000 n \n0000000206 00000 n \n"
    b"trailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n298\n%%EOF"
)


@pytest.fixture(scope="module")
def db_url():
    return os.environ.get(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/assistax",
    )


@pytest.fixture(scope="module")
def setup_schema(db_url):
    """Crea schema si no existe. Ejecutar schema_test.sql manualmente una vez."""
    import psycopg2
    schema_path = os.path.join(
        os.path.dirname(__file__), "schema_test.sql"
    )
    if os.path.exists(schema_path):
        with open(schema_path) as f:
            sql = f.read()
        conn = psycopg2.connect(db_url)
        try:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(sql)
        finally:
            conn.close()
    yield


@pytest.fixture
def clean_db(db_url):
    """Limpia tablas antes de cada test."""
    import psycopg2
    conn = psycopg2.connect(db_url)
    try:
        conn.autocommit = False
        with conn.cursor() as cur:
            cur.execute('DELETE FROM legal_chunks')
            cur.execute('DELETE FROM legal_documents')
            cur.execute('DELETE FROM index_runs')
        conn.commit()
    finally:
        conn.close()
    yield
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor() as cur:
            cur.execute('DELETE FROM legal_chunks')
            cur.execute('DELETE FROM legal_documents')
            cur.execute('DELETE FROM index_runs')
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def mock_blob_download():
    """Mock de download_pdf_bytes retornando PDF mínimo. Alineado con assistax-fn: solo bytes."""
    with patch(
        "pipeline.runner.download_pdf_bytes",
        return_value=_MINIMAL_PDF_BYTES,
    ):
        yield


@pytest.fixture
def mock_embeddings():
    """Mock de embed_chunks retornando vectores 1536-dim falsos."""
    def _fake_embed(chunks, batch_size=100, progress_callback=None):
        n = len(chunks)
        result = [[0.1] * 1536 for _ in range(n)]
        if progress_callback:
            progress_callback(n, n)
        return result

    with patch("pipeline.runner.embed_chunks", side_effect=_fake_embed):
        yield


@pytest.mark.integration
def test_pipeline_new_pdf_completed(
    db_url, setup_schema, clean_db, mock_blob_download, mock_embeddings
):
    """
    Test 1: PDF nuevo → index_run en completed, chunks en BD con embeddings válidos.
    """
    os.environ["DATABASE_URL"] = db_url
    # Recrear settings para coger nueva DATABASE_URL
    import importlib
    import settings as st
    importlib.reload(st)

    from pipeline.db_writer import init_pool, close_pool, get_db_conn
    from pipeline.runner import run_pipeline
    from models import ProcessPdfRequest

    init_pool()
    try:
        run_id = str(uuid.uuid4())
        # Crear index_run (el backend Node lo crea antes de enviar el payload)
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'INSERT INTO index_runs (id, status) VALUES (%s, %s)',
                    (run_id, "running"),
                )
                conn.commit()

        payload = ProcessPdfRequest(
            runId=run_id,
            blobPath="test/fase4-doc-nuevo.pdf",
            documentTitle="Ley Test Fase 4",
            categoryId=str(uuid.uuid4()),
            publishDate="2024-01-15",
        )
        run_pipeline(payload)

        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT status, "docsIndexed", "chunksTotal" FROM index_runs WHERE id = %s',
                    (run_id,),
                )
                row = cur.fetchone()
        assert row is not None
        assert row[0] == "completed"
        assert row[1] == 1
        assert row[2] >= 1

        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT COUNT(*) FROM legal_chunks c JOIN legal_documents d ON c."documentId" = d.id WHERE d."blobPath" = %s',
                    ("test/fase4-doc-nuevo.pdf",),
                )
                count = cur.fetchone()[0]
        assert count >= 1
    finally:
        close_pool()


@pytest.mark.integration
def test_pipeline_duplicate_skipped(
    db_url, setup_schema, clean_db, mock_blob_download, mock_embeddings
):
    """
    Test 2: PDF duplicado (mismo SHA256) → index_run en skipped, BD sin cambios.
    """
    os.environ["DATABASE_URL"] = db_url
    import importlib
    import settings as st
    importlib.reload(st)

    from pipeline.db_writer import init_pool, close_pool, get_db_conn
    from pipeline.runner import run_pipeline
    from models import ProcessPdfRequest

    init_pool()
    try:
        # Hash fijo para simular duplicado
        fixed_hash = "b" * 64
        with patch(
            "pipeline.runner.download_pdf_bytes",
            return_value=(_MINIMAL_PDF_BYTES, fixed_hash),
        ):
            run_id1 = str(uuid.uuid4())
            with get_db_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        'INSERT INTO index_runs (id, status) VALUES (%s, %s)',
                        (run_id1, "running"),
                    )
                    conn.commit()
            payload1 = ProcessPdfRequest(
                runId=run_id1,
                blobPath="test/primero.pdf",
                documentTitle="Primero",
                categoryId=str(uuid.uuid4()),
                publishDate=None,
            )
            run_pipeline(payload1)

        # Segundo con mismo hash (diferente blobPath para simular mismo contenido)
        with patch(
            "pipeline.runner.download_pdf_bytes",
            return_value=(_MINIMAL_PDF_BYTES, fixed_hash),
        ):
            run_id2 = str(uuid.uuid4())
            with get_db_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        'INSERT INTO index_runs (id, status) VALUES (%s, %s)',
                        (run_id2, "running"),
                    )
                    conn.commit()
            payload2 = ProcessPdfRequest(
                runId=run_id2,
                blobPath="test/segundo-mismo-hash.pdf",
                documentTitle="Segundo",
                categoryId=str(uuid.uuid4()),
                publishDate=None,
            )
            run_pipeline(payload2)

        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT status FROM index_runs WHERE id = %s',
                    (run_id2,),
                )
                row = cur.fetchone()
        assert row is not None
        assert row[0] == "skipped"
    finally:
        close_pool()


@pytest.mark.integration
def test_pipeline_insert_failure_marks_failed(db_url, setup_schema, clean_db, mock_blob_download):
    """
    Test 4: Fallo en insert_chunks_bulk → index_run en failed, conexión devuelta al pool.
    """
    os.environ["DATABASE_URL"] = db_url
    import importlib
    import settings as st
    importlib.reload(st)

    from pipeline.db_writer import init_pool, close_pool, get_db_conn
    from pipeline.runner import run_pipeline
    from models import ProcessPdfRequest

    init_pool()
    try:
        run_id = str(uuid.uuid4())
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'INSERT INTO index_runs (id, status) VALUES (%s, %s)',
                    (run_id, "running"),
                )
                conn.commit()

        payload = ProcessPdfRequest(
            runId=run_id,
            blobPath="test/fail-insert.pdf",
            documentTitle="Fail Insert",
            categoryId=str(uuid.uuid4()),
            publishDate=None,
        )

        def _fail_insert(*args, **kwargs):
            raise RuntimeError("Simulated insert failure")

        with patch("pipeline.runner.embed_chunks", side_effect=lambda c, **kw: [[0.1] * 1536] * len(c)):
            with patch("pipeline.runner.insert_chunks_bulk", side_effect=_fail_insert):
                run_pipeline(payload)

        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT status, error FROM index_runs WHERE id = %s',
                    (run_id,),
                )
                row = cur.fetchone()
        assert row is not None
        assert row[0] == "failed"
        assert "Simulated insert failure" in (row[1] or "")
    finally:
        close_pool()
