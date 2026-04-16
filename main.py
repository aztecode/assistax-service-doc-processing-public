"""
Assistax PDF Processor API.
API que sustituye Azure Functions para procesamiento de PDFs legales.
"""
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from settings import settings
from logging_config import configure_structlog
from middleware.auth import verify_api_key
from pipeline.db_writer import init_pool, close_pool, get_db_conn
from jobs.cleanup import start_cleanup_job, stop_cleanup_job
from models import ProcessPdfRequest
from pipeline.runner import submit_pipeline

_logger = structlog.get_logger()

limiter = Limiter(key_func=get_remote_address)


def _check_database() -> str:
    """Verifica conectividad con PostgreSQL. Retorna 'ok' o mensaje de error."""
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return "ok"
    except Exception as e:
        return f"error: {e}"


def _check_blob_storage() -> str:
    """Verifica conectividad con Azure Blob Storage. Retorna 'ok' o mensaje de error."""
    try:
        from azure.storage.blob import BlobServiceClient
        client = BlobServiceClient.from_connection_string(
            settings.AZURE_STORAGE_CONNECTION_STRING
        )
        container_client = client.get_container_client(settings.AZURE_BLOB_CONTAINER)
        container_client.get_container_properties()
        return "ok"
    except Exception as e:
        return f"error: {e}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan: startup y shutdown de recursos (pool DB, cleanup job)."""
    configure_structlog()
    init_pool()
    start_cleanup_job()
    _logger.info(
        "app.started",
        environment=settings.ENVIRONMENT,
        port=settings.WEBSITES_PORT,
        llm_doc_type=settings.ENABLE_LLM_DOC_TYPE,
        pdf_normalization=settings.ENABLE_PDF_TEXT_NORMALIZATION,
    )
    yield
    _logger.info("app.shutting_down")
    stop_cleanup_job()
    close_pool()


app = FastAPI(
    title="Assistax PDF Processor",
    description="API que sustituye Azure Functions para procesamiento de PDFs legales.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None if settings.ENVIRONMENT == "production" else "/docs",
    redoc_url=None if settings.ENVIRONMENT == "production" else "/redoc",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.middleware("http")(verify_api_key)


@app.get("/health")
async def health() -> dict:
    """
    Health check para Azure Web App probes.
    Responde 200 si todo ok, 503 si algún check falla.
    Debe responder en menos de 5 segundos.
    """
    db_status = _check_database()
    blob_status = _check_blob_storage()
    all_ok = db_status == "ok" and blob_status == "ok"

    response = {
        "status": "ok" if all_ok else "degraded",
        "checks": {
            "database": db_status,
            "blob_storage": blob_status,
        },
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
    }

    if all_ok:
        return response
    return JSONResponse(status_code=503, content=response)


@app.post("/api/process-pdf")
@limiter.limit("20/minute")
async def process_pdf(request: Request, payload: ProcessPdfRequest):
    """
    Fase 1: acepta payload y ejecuta pipeline en background.
    Respuesta 202 inmediata; el pipeline actualiza index_run (processing/skipped/failed/completed).
    """
    _logger.info(
        "request.process_pdf",
        run_id=payload.runId,
        title=payload.documentTitle[:60],
        blob_path=payload.blobPath,
    )
    submit_pipeline(payload)
    return JSONResponse(
        status_code=202,
        content={"runId": payload.runId, "status": "processing"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.WEBSITES_PORT,
        reload=settings.ENVIRONMENT != "production",
    )
