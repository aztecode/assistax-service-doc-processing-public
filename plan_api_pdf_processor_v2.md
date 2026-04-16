# Plan de Implementación — API Python: PDF Legal Processor
> Sustituye Azure Functions · Compatible con `assistax-back` · FastAPI + PostgreSQL + Azure Blob

---

## Decisiones de arquitectura (confirmadas)

| Decisión | Elección | Justificación |
|---|---|---|
| **Sync vs Async** | Sync + `ThreadPoolExecutor` | PyMuPDF y psycopg2 son sync nativos. Volumen de 30 docs no justifica la complejidad de async |
| **Instancias** | Una sola instancia | Simplifica cleanup job, pool de conexiones y coordinación de workers |
| **Hosting** | Azure Web App | Dockerfile multi-stage compatible. Requiere `WEBSITES_PORT` en configuración |
| **Cleanup job** | APScheduler embebido | Sin múltiples instancias, no hay riesgo de ejecución duplicada |
| **Concurrencia** | 2 workers, pool de 5 conexiones | Máximo 30 documentos en total; valores conservadores evitan throttling en Azure OpenAI |

---

## Principios que guían este plan

- **Fail-fast**: cualquier configuración inválida mata el proceso al arrancar, no en runtime
- **Seguro por defecto**: auth obligatoria, sin opcionales en producción
- **Observable**: cada pipeline deja rastro estructurado en logs
- **Concurrencia acotada**: límites ajustados al volumen real (máx 30 documentos)
- **Contrato estable**: el backend Node no necesita cambios hasta la Fase 6

---

## Estructura de carpetas objetivo

```
api/
├── main.py                  # FastAPI app, lifespan, routers
├── settings.py              # BaseSettings con validación al arrancar
├── middleware/
│   └── auth.py              # Validación x-functions-key (obligatoria)
├── pipeline/
│   ├── runner.py            # Orquestador, ThreadPoolExecutor (2 workers), semáforo global
│   ├── blob_download.py     # download_pdf_bytes() → bytes + early hash
│   ├── pdf_extractor.py     # PyMuPDF → List[PageContent]
│   ├── pdf_text_normalization.py
│   ├── legal_ordinal_patterns.py
│   ├── legal_chunker.py     # chunk_content() → List[Chunk]
│   ├── metadata_extractor.py
│   ├── doc_type_classifier.py
│   ├── embeddings.py        # embed_chunks() con retry + semáforo
│   └── db_writer.py         # pool de conexiones (max 5), upsert, bulk insert
├── models.py                # ProcessPdfRequest, HealthResponse
├── jobs/
│   └── cleanup.py           # APScheduler: index_runs stuck en processing → failed
├── tests/
│   ├── unit/
│   │   ├── test_chunker.py
│   │   ├── test_normalizer.py
│   │   ├── test_extractor.py
│   │   └── test_metadata.py
│   └── integration/
│       └── test_pipeline.py
├── Dockerfile               # Multi-stage, compatible Azure Web App
├── docker-compose.yml       # PostgreSQL + API para desarrollo local
├── pyproject.toml           # Ruff + Black + pytest config
├── requirements.txt
└── .env.example
```

---

## Variables de entorno (completas)

| Variable | Requerida | Default | Descripción |
|---|---|---|---|
| `DATABASE_URL` | ✅ | — | PostgreSQL connection string |
| `AZURE_STORAGE_CONNECTION_STRING` | ✅ | — | Azure Blob Storage |
| `AZURE_BLOB_CONTAINER` | ❌ | `laws` | Contenedor de PDFs |
| `AZURE_OPENAI_ENDPOINT` | ✅ | — | Endpoint Azure OpenAI |
| `AZURE_OPENAI_API_KEY` | ✅ | — | API Key Azure OpenAI |
| `AZURE_OPENAI_API_VERSION` | ✅ | — | Versión de la API |
| `AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT` | ✅ | — | Deployment embeddings |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | ❌ | `gpt-4o-mini` | Para clasificación doc type |
| `FUNCTIONS_API_KEY` | ✅ | — | Clave para `x-functions-key` (auth) |
| `PDF_WORKER_THREADS` | ❌ | `2` | Max pipelines concurrentes |
| `DB_POOL_MIN_CONN` | ❌ | `1` | Pool mínimo conexiones BD |
| `DB_POOL_MAX_CONN` | ❌ | `5` | Pool máximo conexiones BD |
| `OPENAI_MAX_CONCURRENT` | ❌ | `2` | Semáforo llamadas OpenAI por pipeline |
| `ENABLE_LLM_DOC_TYPE` | ❌ | `false` | Clasificación docType vía LLM |
| `ENABLE_PDF_TEXT_NORMALIZATION` | ❌ | `true` | Normalización de texto extraído |
| `CLEANUP_INTERVAL_MINUTES` | ❌ | `10` | Frecuencia del job de cleanup |
| `WEBSITES_PORT` | ❌ | `8000` | Puerto inyectado por Azure Web App |
| `LOG_FORMAT` | ❌ | `json` | `json` en producción, `console` en desarrollo |
| `ENVIRONMENT` | ❌ | `production` | `production` deshabilita `/docs` de Swagger |

---

---

## FASE 0 — Fundación
**Duración estimada:** 1–2 días
**Objetivo:** Estructura lista, API corriendo, auth activa, sin deuda técnica desde el primer commit.

### Tareas

#### 0.1 Setup del proyecto
- [x] Inicializar repo con `pyproject.toml` que configure:
  - `ruff` como linter (reemplaza flake8 + isort)
  - `black` como formatter
  - `pytest` con `pytest-asyncio` para tests
- [x] Crear `requirements.txt` con versiones fijas:
  ```
  fastapi>=0.111.0
  uvicorn[standard]>=0.29.0
  PyMuPDF>=1.24.0
  psycopg2-binary>=2.9.0
  openai>=1.3.0
  azure-storage-blob>=12.19.0
  pydantic>=2.0.0
  pydantic-settings>=2.0.0
  structlog>=24.0.0
  slowapi>=0.1.9
  apscheduler>=3.10.0
  pytest>=8.0.0
  pytest-asyncio>=0.23.0
  ```
- [x] Crear `.env.example` con todas las variables documentadas arriba

#### 0.2 `settings.py` — Validación fail-fast
```python
# settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Requeridas — la app no arranca si faltan
    DATABASE_URL: str
    AZURE_STORAGE_CONNECTION_STRING: str
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_API_VERSION: str
    AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT: str
    FUNCTIONS_API_KEY: str

    # Opcionales con defaults ajustados al volumen real (máx 30 docs)
    AZURE_BLOB_CONTAINER: str = "laws"
    AZURE_OPENAI_CHAT_DEPLOYMENT: str = "gpt-4o-mini"
    PDF_WORKER_THREADS: int = 2       # 2 pipelines simultáneos máximo
    DB_POOL_MIN_CONN: int = 1
    DB_POOL_MAX_CONN: int = 5         # 5 conexiones cubre holgado para 2 workers
    OPENAI_MAX_CONCURRENT: int = 2    # Evita throttling en Azure OpenAI
    ENABLE_LLM_DOC_TYPE: bool = False
    ENABLE_PDF_TEXT_NORMALIZATION: bool = True
    CLEANUP_INTERVAL_MINUTES: int = 10
    WEBSITES_PORT: int = 8000         # Azure Web App inyecta esta variable
    LOG_FORMAT: str = "json"
    ENVIRONMENT: str = "production"

    class Config:
        env_file = ".env"

settings = Settings()  # Falla al arrancar si falta cualquier variable requerida
```
- [x] Implementado

#### 0.3 `middleware/auth.py` — Auth obligatoria
```python
# middleware/auth.py
from fastapi import Request, HTTPException
from settings import settings

async def verify_api_key(request: Request, call_next):
    # /health es público para Azure Web App health probes
    if request.url.path == "/health":
        return await call_next(request)

    key = request.headers.get("x-functions-key")
    if not key or key != settings.FUNCTIONS_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    return await call_next(request)
```
- [x] Implementado

#### 0.4 `main.py` — App con lifespan
```python
# main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from settings import settings
from middleware.auth import verify_api_key
from pipeline.db_writer import init_pool, close_pool
from jobs.cleanup import start_cleanup_job, stop_cleanup_job

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: validar conexiones antes de aceptar tráfico
    init_pool()
    start_cleanup_job()
    yield
    # Shutdown
    stop_cleanup_job()
    close_pool()

app = FastAPI(
    title="Assistax PDF Processor",
    description="API que sustituye Azure Functions para procesamiento de PDFs legales.",
    version="1.0.0",
    lifespan=lifespan,
    # Swagger solo disponible fuera de producción
    docs_url=None if settings.ENVIRONMENT == "production" else "/docs",
    redoc_url=None if settings.ENVIRONMENT == "production" else "/redoc",
)

app.middleware("http")(verify_api_key)
```
- [x] Implementado

#### 0.5 `GET /health` — Health check real para Azure
```python
# Respuesta en verde
{
  "status": "ok",
  "checks": {
    "database": "ok",
    "blob_storage": "ok"
  },
  "version": "1.0.0",
  "environment": "production"
}

# Respuesta si algo falla → HTTP 503
{
  "status": "degraded",
  "checks": {
    "database": "error: could not connect",
    "blob_storage": "ok"
  }
}
```
> Azure Web App usa este endpoint como health probe. Debe responder en < 5 segundos.
- [x] Implementado con checks reales de BD y Blob Storage

#### 0.6 `POST /api/process-pdf` — Stub 202
- [x] Validar payload con `ProcessPdfRequest`
- [x] Devolver `{"runId": "...", "status": "processing"}` inmediatamente (202)
- Pipeline real se conecta en Fase 4

#### 0.7 `models.py` — Contrato tipado y corregido
```python
# models.py — corrige el bug de publishDate de la doc original
from pydantic import BaseModel, field_validator
import uuid

class ProcessPdfRequest(BaseModel):
    runId: str
    blobPath: str
    documentTitle: str
    categoryId: str
    publishDate: str | None = None  # Siempre presente en el body, puede ser null

    @field_validator("runId", "categoryId")
    @classmethod
    def must_be_valid_uuid(cls, v):
        try:
            uuid.UUID(v)
        except ValueError:
            raise ValueError(f"'{v}' no es un UUID válido")
        return v

    @field_validator("blobPath", "documentTitle")
    @classmethod
    def must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("El campo no puede estar vacío")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "runId": "550e8400-e29b-41d4-a716-446655440000",
                "blobPath": "fiscal/2024/ley-federal-trabajo.pdf",
                "documentTitle": "Ley Federal del Trabajo",
                "categoryId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "publishDate": "2024-01-15"
            }
    }
}
```
- [x] Implementado con validadores para UUID, campos no vacíos

#### 0.8 Docker Compose para desarrollo local
```yaml
# docker-compose.yml
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: assistax
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  api:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      - db
    volumes:
      - .:/app

volumes:
  pgdata:
```
- [x] Implementado (incluye Dockerfile multi-stage compatible Azure Web App)

### ✅ Criterios de aceptación de la Fase 0
- `GET /health` responde 200 con checks reales de BD y Blob en < 5 segundos

### 📋 Tareas realizadas (Fase 0 implementada)
- **0.1** pyproject.toml, requirements.txt, .env.example creados
- **0.2** settings.py con validación fail-fast al arrancar
- **0.3** middleware/auth.py — /health público, resto exige x-functions-key
- **0.4** main.py con lifespan (init_pool, cleanup job)
- **0.5** GET /health con checks reales de database y blob_storage
- **0.6** POST /api/process-pdf stub → 202 con runId y status
- **0.7** models.py — ProcessPdfRequest con validadores UUID y no vacíos
- **0.8** docker-compose.yml y Dockerfile multi-stage
- `POST /api/process-pdf` sin header `x-functions-key` → 401
- `POST /api/process-pdf` con payload inválido (UUID malformado, campo vacío) → 400 con detalle
- `POST /api/process-pdf` con payload válido → 202
- Arrancar sin `.env` → error explícito que lista las variables faltantes
- `docker compose up` levanta todo sin intervención manual

---

## FASE 1 — Descarga, extracción y deduplicación temprana
**Duración estimada:** 2–3 días
**Objetivo:** Pipeline funciona hasta texto estructurado por página. Duplicados detectados antes de procesar.

### Tareas

#### 1.1 `blob_download.py` ✅
- `download_pdf_bytes(blob_path: str) -> tuple[bytes, str]`
  - Retorna `(pdf_bytes, content_hash)` — hash calculado aquí, no al final
  - Timeout explícito de 60 segundos
  - Errores diferenciados:
    - `ResourceNotFoundError` → `BlobNotFoundError` (custom, mensaje claro)
    - Timeout → `BlobDownloadTimeoutError` (custom)
  - `content_hash = hashlib.sha256(pdf_bytes).hexdigest()`

#### 1.2 Verificación temprana de duplicados ✅
```python
# En db_writer.py
def check_duplicate_by_hash(conn, content_hash: str) -> str | None:
    """Retorna document_id si el PDF ya fue indexado, None si es nuevo."""
    with conn.cursor() as cursor:
        cursor.execute(
            'SELECT id FROM legal_documents WHERE "contentHash" = %s',
            (content_hash,)
        )
        row = cursor.fetchone()
        return str(row[0]) if row else None
```

**Flujo en runner — con dedup temprana:**
```
1. download_pdf_bytes(blobPath) → (bytes, content_hash)
2. check_duplicate_by_hash(conn, content_hash)
   → Si existe: update_index_run(status="skipped") → return   ← sin gastar tokens OpenAI
   → Si no:     continuar pipeline normalmente
```

#### 1.3 `pdf_extractor.py`
✅ Implementado
```python
@dataclass
class PageContent:
    page_number: int      # 1-indexed
    text: str             # texto SIN contenido de tablas
    tables: list[TableBlock]

@dataclass
class TableBlock:
    table_index: int
    page_number: int
    markdown: str         # formato [TABLE_N]...[/TABLE_N]
    rows: list[list[str]]
    bbox: tuple[float, float, float, float]
```

- `extract_pdf(source: bytes) -> list[PageContent]`
- **Criterios documentados para `_is_likely_prose_not_table`:**
  - Tabla con 1 columna → prosa en grid
  - Tabla donde >80% de celdas tienen más de 50 chars → prosa en grid
  - Primera columna sin valores numéricos en >90% de filas → candidata a prosa
- **Criterios para `_is_tariff_like_table`** (nunca se filtra):
  - Primera columna con valores numéricos en >60% de filas
  - Headers incluyen: "tasa", "cuota", "tarifa", "monto", "importe"
- TOC: `doc.get_toc(simple=True)` normalizado a `{level, title, page}` con page 1-indexed
- PDF corrupto → `raise PDFExtractionError` con mensaje descriptivo

#### 1.4 `pdf_text_normalization.py`
✅ Implementado
```python
@dataclass
class NormalizationAudit:
    lines_merged: int
    hyphen_joins: int
    headers_removed: int
    original_line_count: int
    final_line_count: int

def normalize_pdf_text(raw_text: str) -> tuple[str, NormalizationAudit]:
    ...
```

Reglas de preservación de saltos (nunca unir antes de):
- `Artículo N` (incluyendo ordinales, `194-N`, `3 Bis`)
- `Capítulo`, `Título`, `Sección`
- Romanos `I.`, `II.`, etc.
- Incisos `a)`, `b)`
- `Fracción`, `Inciso`
- Filas de tablas numéricas

Uniones permitidas:
- Línea termina en `-` + siguiente empieza con letra → unir palabra (excepto `194-N-1`)
- `merge_score >= 2` → unir líneas (no termina en puntuación + siguiente empieza en minúscula)
- Headers repetidos ≥3 veces → eliminar (paginación)

#### 1.5 Tests unitarios — Fase 1
- ✅ `test_extractor.py`: ley corta, “ley con tablas” (dependiente de `find_tables()`), PDF corrupto y PDF mínimo válido
- ✅ `test_normalizer.py`: casos de preservación/unión + audit
- ⚠️ Fixtures en `tests/fixtures/` con PDFs reales: **pendiente** (actualmente se generan PDFs mínimos en memoria en `tests/conftest.py`)

### ✅ Criterios de aceptación de la Fase 1
- PDF duplicado (mismo SHA256) → pipeline termina con `status="skipped"`, sin llamar a OpenAI
- PDF de 50 páginas → `List[PageContent]` con tablas correctamente separadas del texto
- PDF corrupto → `index_run` marcado como `failed` con mensaje descriptivo
- Tabla tarifaria nunca filtrada como prosa
- Normalización preserva todos los saltos antes de `Artículo N`

---

## FASE 2 — Chunking y extracción de metadata
**Duración estimada:** 2–3 días
**Objetivo:** Texto en chunks semánticos tipados y metadata extraída correctamente.

### Tareas

#### 2.1 `legal_ordinal_patterns.py` — ✅ Implementado ✅
```python
ROMAN_WITH_LOOKAHEAD = r"(?:[IVX]+|M{0,4}(?:CM|CD|D?C{0,3})...)(?!\w)"  # Evita "VI" en "VIGESIMO"

ARTICLE_ORDINAL_WORDS = (
    "vigésimo noveno", "vigésimo octavo", "vigésimo séptimo",
    "vigésimo sexto", "vigésimo quinto", "vigésimo cuarto",
    "vigésimo tercero", "vigésimo segundo", "vigésimo primero",
    "vigésimo", "décimo noveno", "décimo octavo", "décimo séptimo",
    "décimo sexto", "décimo quinto", "décimo cuarto", "décimo tercero",
    "décimo segundo", "décimo primero", "décimo", "noveno", "octavo",
    "séptimo", "sexto", "quinto", "cuarto", "tercero", "segundo",
    "primero", "único"
)

ARTICLE_ORDINAL_WORDS_PATTERN = "|".join(ARTICLE_ORDINAL_WORDS)
```
- [x] Implementado

#### 2.2 `legal_chunker.py` — ✅ Implementado ✅
```python
@dataclass
class Chunk:
    text: str
    chunk_no: int
    chunk_type: str          # article, chapter, title, section, fraction, table, generic...
    article_ref: str | None  # "Art. 5.", "Art. 10 Bis."
    heading: str             # máx 500 chars
    start_page: int
    end_page: int
    has_table: bool
    table_index: int | None
```

- `chunk_content(pages: list[PageContent], max_chunk_chars: int = 2000) -> list[Chunk]`
- `split_by_legal_structure(text)` con regex para:
  - Libro, Título, Capítulo, Sección, Anexo
  - Artículo (incluyendo Bis, Ter, Quáter, y formato `194-N`)
  - Regla, Numeral
  - Fracción (romanos e incisos `a)`, `b)`)
  - Transitorio
  - `[TABLE_N]`
- Filtrar: notas de reforma inline, paginación, citas tipo "de esta Ley"
- Tablas → chunk atómico; `article_ref` via `detect_article_context(chunks_so_far)`
- `_split_by_size(text, max_chars, min_size=25)` respetando `\n\n` > `\n` > espacio
- [x] Implementado

#### 2.3 `metadata_extractor.py` — `infer_doc_type` corregido — ✅ Implementado ✅
```python
# CORREGIDO: ya no retorna "ley" siempre
DOC_TYPE_KEYWORDS: dict[str, list[str]] = {
    "reglamento": ["reglamento", "reglamenta"],
    "codigo":     ["código", "codigo"],
    "estatuto":   ["estatuto", "estatutos"],
    "resolucion": ["resolución", "resolucion", "acuerdo"],
    "presupuesto":["presupuesto", "egresos"],
    "ley":        ["ley", "decreto", "norma"],  # fallback real, no el único caso
}

def infer_doc_type(title: str) -> str:
    title_lower = title.lower()
    for doc_type, keywords in DOC_TYPE_KEYWORDS.items():
        if any(kw in title_lower for kw in keywords):
            return doc_type
    return "ley"
```

- `extract_legal_legend(content: str) -> dict` con regex para:
  - `última reforma DOF DD-MM-YYYY`
  - `nueva ley publicada en el DOF el N de mes de AÑO`
  - `TEXTO VIGENTE`
  - `cantidades actualizadas por ... DOF DD-MM-YYYY`
- `extract_law_name(title: str) -> str` — sin paréntesis final, máx 255 chars
- [x] Implementado

#### 2.4 Tests unitarios — Fase 2 — ✅ Implementado ✅
- `test_chunker.py` con casos edge:
  - Artículo `194-N-1` — no romper el número
  - Artículo con fracción que tiene tabla incrustada
  - Artículo Bis seguido inmediatamente de Artículo Ter
  - Texto que excede `max_chunk_chars` con y sin separadores naturales
  - Transitorio con referencia a artículo previo
- `test_metadata.py`:
  - `infer_doc_type("Reglamento de la Ley del ISR")` → `"reglamento"`
  - `infer_doc_type("Código Fiscal de la Federación")` → `"codigo"`
  - `infer_doc_type("Presupuesto de Egresos 2024")` → `"presupuesto"`
  - `extract_law_name` con paréntesis, sin paréntesis, con más de 255 chars
- [x] Implementado

### ✅ Criterios de aceptación de la Fase 2
- Ley Federal del Trabajo → chunks con tipos correctos en >95% de casos
- `infer_doc_type("Reglamento de la Ley del ISR")` → `"reglamento"` (no `"ley"`)
- Tabla dentro de un artículo → chunk atómico con `article_ref` del artículo contenedor
- Ningún chunk con `len(text) < 25`

### 📋 Tareas realizadas (Fase 2 implementada)
- **2.1** legal_ordinal_patterns.py — ROMAN_WITH_LOOKAHEAD, ARTICLE_ORDINAL_WORDS
- **2.2** legal_chunker.py — Chunk, chunk_content, split_by_legal_structure, _split_by_size, detect_article_context
- **2.3** metadata_extractor.py — infer_doc_type, extract_legal_legend, extract_law_name
- **2.4** test_chunker.py y test_metadata.py — 24 tests unitarios pasando

---

## FASE 3 — Embeddings y clasificación por LLM
**Duración estimada:** 1–2 días
**Objetivo:** Chunks con vectores de 1536 dimensiones, con manejo robusto de throttling de Azure OpenAI.

### Tareas

#### 3.1 `embeddings.py` — Retry mejorado con semáforo ajustado — [x] Implementado
```python
import threading
import time
from settings import settings

# Semáforo: máximo 2 llamadas concurrentes a OpenAI (OPENAI_MAX_CONCURRENT=2)
_openai_semaphore = threading.Semaphore(settings.OPENAI_MAX_CONCURRENT)

RETRY_DELAYS = [1, 2, 4, 8, 16]  # 5 intentos con backoff exponencial

def embed_chunks(
    chunks: list[Chunk],
    batch_size: int = 100,
    progress_callback=None
) -> list[list[float]]:
    client = AzureOpenAI(...)
    results = [None] * len(chunks)

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start:batch_start + batch_size]
        texts = [c.text if c.text.strip() else " " for c in batch]

        for attempt, delay in enumerate(RETRY_DELAYS):
            try:
                with _openai_semaphore:
                    response = client.embeddings.create(
                        model=settings.AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT,
                        input=texts
                    )
                # Reordenar por índice si Azure responde fuera de orden
                for item in response.data:
                    results[batch_start + item.index] = item.embedding
                break

            except RateLimitError as e:
                # Respetar el Retry-After de Azure si viene en el header
                retry_after = getattr(e, 'retry_after', delay)
                time.sleep(retry_after)

            except Exception:
                if attempt == len(RETRY_DELAYS) - 1:
                    raise EmbeddingError(f"Falló batch {batch_start} tras 5 intentos")
                time.sleep(delay)

        if progress_callback:
            progress_callback(min(batch_start + batch_size, len(chunks)), len(chunks))

    return results
```

#### 3.2 `doc_type_classifier.py` — [x] Implementado
- Solo se instancia si `ENABLE_LLM_DOC_TYPE=true`
- `classify_doc_type(title: str, headings: list[str]) -> tuple[str, float, str]`
  - Primeros 40 headings máximo
  - Si falla (timeout, parse error) → fallback silencioso a `infer_doc_type(title)`
  - Log de confianza para monitoreo

#### 3.3 Actualización de progreso — [x] Implementado
- `progress_callback` llama a `update_index_run_progress(conn, run_id, processed, total)`
- Permite visibilidad en tiempo real desde el backend Node

### ✅ Criterios de aceptación de la Fase 3
- Simular error 429 con `Retry-After: 5` → función espera ≥5 segundos antes de reintentar
- 5 fallos consecutivos → `EmbeddingError`, `index_run` marcado como `failed`
- `ENABLE_LLM_DOC_TYPE=false` → `doc_type_classifier.py` nunca se importa ni instancia

### 📋 Tareas realizadas (Fase 3 implementada)
- **3.1** embeddings.py — embed_chunks con semáforo, retry (Retry-After), EmbeddingError
- **3.2** doc_type_classifier.py — classify_doc_type con fallback a infer_doc_type
- **3.3** update_index_run_progress en db_writer.py para visibilidad en tiempo real

---

## FASE 4 — Escritura a BD y pipeline completo
**Duración estimada:** 2 días
**Objetivo:** Pipeline end-to-end funcionando con pool de conexiones y concurrencia acotada a 2 workers.

### Tareas

#### 4.1 `db_writer.py` — Pool ajustado al volumen real — [x] Implementado
```python
import psycopg2.pool
from contextlib import contextmanager
from settings import settings

_pool: psycopg2.pool.ThreadedConnectionPool | None = None

def init_pool():
    global _pool
    _pool = psycopg2.pool.ThreadedConnectionPool(
        minconn=settings.DB_POOL_MIN_CONN,  # 1
        maxconn=settings.DB_POOL_MAX_CONN,  # 5 — suficiente para 2 workers + health checks
        dsn=settings.DATABASE_URL
    )

def close_pool():
    if _pool:
        _pool.closeall()

@contextmanager
def get_db_conn():
    conn = _pool.getconn()
    try:
        yield conn
    finally:
        _pool.putconn(conn)  # Siempre devuelve al pool, incluso en excepciones
```

#### 4.2 Todas las funciones de `db_writer.py` — [x] Implementado
- `check_duplicate_by_hash(conn, content_hash) -> str | None`
- `upsert_legal_document(conn, ...) -> str` — `SELECT` por `blobPath`; `UPDATE` o `INSERT`; retorna `document_id`
- `delete_existing_chunks(conn, document_id: str)`
- `insert_chunks_bulk(conn, document_id, chunks, embeddings)`:
  - `execute_values` en lotes de 500
  - `to_tsvector('spanish', unaccent(text))` para campo `tsv`
- `update_index_run(conn, run_id, status, docs_indexed=None, chunks_total=None, error_log=None)`
- `update_index_run_progress(conn, run_id, processed_chunks, chunks_total)`

#### 4.3 `runner.py` — 2 workers, semáforo explícito — [x] Implementado
```python
from concurrent.futures import ThreadPoolExecutor
import threading
from settings import settings

# 2 workers: cubre el caso de uso real sin sobrecargar Azure OpenAI
_executor = ThreadPoolExecutor(max_workers=settings.PDF_WORKER_THREADS)
_pipeline_semaphore = threading.Semaphore(settings.PDF_WORKER_THREADS)

def submit_pipeline(payload: ProcessPdfRequest):
    """No bloqueante. El endpoint responde 202 antes de que esto termine."""
    _executor.submit(_run_pipeline_safe, payload)

def _run_pipeline_safe(payload: ProcessPdfRequest):
    with _pipeline_semaphore:
        try:
            run_pipeline(payload)
        except Exception as e:
            # Garantiza que index_run SIEMPRE queda en estado terminal
            _mark_run_failed(payload.runId, str(e))
```

#### 4.4 `run_pipeline(payload)` — Orden completo — [x] Implementado
```
1.  Obtener conexión del pool
2.  update_index_run(status="processing")
3.  (bytes, content_hash) = download_pdf_bytes(blobPath)
4.  duplicate_id = check_duplicate_by_hash(conn, content_hash)
    → Si existe: update_index_run(status="skipped") → return
5.  pages = extract_pdf(bytes)
6.  Si ENABLE_PDF_TEXT_NORMALIZATION: normalizar texto de cada página
7.  legend = extract_legal_legend(primeras 8 páginas concatenadas)
8.  law_name = extract_law_name(documentTitle)
9.  chunks = chunk_content(pages)
10. doc_type = classify_doc_type(...) si LLM habilitado, sino infer_doc_type(title)
11. embeddings = embed_chunks(chunks, progress_callback=lambda p,t: update_index_run_progress(...))
12. document_id = upsert_legal_document(conn, blobPath, content_hash, ...)
13. delete_existing_chunks(conn, document_id)
14. insert_chunks_bulk(conn, document_id, chunks, embeddings)
15. update_index_run(status="completed", docs_indexed=1, chunks_total=len(chunks))

En finally:
    release_conn(conn)  # Siempre, en éxito o fallo
```

#### 4.5 Tests de integración — Fase 4 — [x] Implementado
- Usar PostgreSQL de test via Docker Compose
- Test 1: PDF nuevo → `index_run` en `completed`, chunks en BD con embeddings válidos
- Test 2: PDF duplicado (mismo SHA256) → `index_run` en `skipped`, BD sin cambios
- Test 3: 3 PDFs simultáneos (ajustado al volumen real) → todos terminan, pool no excedido, ningún deadlock
- Test 4: Fallo en `insert_chunks_bulk` → rollback, `index_run` en `failed`, conexión devuelta al pool

### ✅ Criterios de aceptación de la Fase 4
- 5 PDFs en bulk (límite real del sistema) → todos en `completed` o `failed`, ninguno en `processing` al terminar
- `DB_POOL_MAX_CONN=5` con 2 workers simultáneos → sin timeouts de conexión
- Fallo en cualquier paso del pipeline → `index_run` en `failed` con error descriptivo
- Ninguna conexión de BD queda abierta (verificar con `SELECT count(*) FROM pg_stat_activity`)

### 📋 Tareas realizadas (Fase 4 implementada)
- **4.1** db_writer.py — Pool ya existía con ThreadedConnectionPool y get_db_conn
- **4.2** db_writer.py — upsert_legal_document, delete_existing_chunks, insert_chunks_bulk (execute_values, unaccent)
- **4.3** runner.py — ThreadPoolExecutor(2 workers), Semaphore, _run_pipeline_safe con _mark_run_failed
- **4.4** run_pipeline — flujo end-to-end completo según especificación
- **4.5** tests/integration/ — test_pipeline.py (PDF nuevo→completed, duplicado→skipped, fallo insert→failed)

---

## FASE 5 — Hardening y observabilidad
**Duración estimada:** 1–2 días
**Objetivo:** API production-ready para Azure Web App, con logs estructurados y cleanup automático.

### Tareas

#### 5.1 Logging estructurado con `structlog` — [x] Implementado
```python
import structlog

logger = structlog.get_logger()

# Al inicio de cada pipeline
logger.info("pipeline.started", run_id=run_id, blob_path=blob_path)

# Al completar cada etapa
logger.info("pipeline.stage.completed",
    run_id=run_id,
    stage="pdf_extraction",
    pages=len(pages),
    duration_ms=elapsed_ms
)

# Al completar el pipeline
logger.info("pipeline.completed",
    run_id=run_id,
    chunks_total=len(chunks),
    duration_ms=total_ms
)

# En errores
logger.error("pipeline.failed",
    run_id=run_id,
    stage="embeddings",
    error=str(e)
)
```

- Formato JSON en producción (`LOG_FORMAT=json`), legible en desarrollo (`LOG_FORMAT=console`)
- **Nunca loguear**: connection strings, API keys, contenido de documentos

#### 5.2 `jobs/cleanup.py` — APScheduler embebido — [x] Implementado
```python
from apscheduler.schedulers.background import BackgroundScheduler
from settings import settings

_scheduler = BackgroundScheduler()

def cleanup_stuck_runs():
    """
    Marca como failed los index_runs en processing por más de 1 hora.
    Solo corre en una instancia → APScheduler embebido es suficiente.
    """
    with get_db_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                UPDATE index_runs
                SET status = 'failed',
                    "endedAt" = NOW(),
                    error = 'Timeout: pipeline superó 1 hora sin completar'
                WHERE status = 'processing'
                  AND "startedAt" < NOW() - INTERVAL '1 hour'
                RETURNING id
            """)
            fixed = cursor.rowcount
            conn.commit()

    if fixed > 0:
        logger.warning("cleanup.stuck_runs_fixed", count=fixed)

def start_cleanup_job():
    _scheduler.add_job(
        cleanup_stuck_runs,
        "interval",
        minutes=settings.CLEANUP_INTERVAL_MINUTES
    )
    _scheduler.start()

def stop_cleanup_job():
    _scheduler.shutdown()
```

#### 5.3 Rate limiting — [x] Implementado
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@router.post("/api/process-pdf")
@limiter.limit("20/minute")
async def process_pdf(request: Request, payload: ProcessPdfRequest):
    ...
```

#### 5.4 `Dockerfile` — Compatible con Azure Web App — [x] Implementado
```dockerfile
# Multi-stage: imagen final sin devdeps
FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages \
                    /usr/local/lib/python3.12/site-packages
COPY . .

EXPOSE 8000

# Azure Web App inyecta WEBSITES_PORT; uvicorn lo lee via settings
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${WEBSITES_PORT:-8000} --workers 1"]
```

> **Nota Azure Web App:** `--workers 1` es correcto para una sola instancia. El paralelismo real lo manejan los 2 threads del `ThreadPoolExecutor`, no múltiples procesos uvicorn.

#### 5.5 README — [x] Implementado
- Quickstart con Docker Compose (3 comandos)
- Tabla de variables de entorno
- Diagrama del pipeline
- Sección: "¿Qué hacer si un `index_run` queda en `processing`?" → el cleanup lo resuelve automáticamente cada 10 minutos; para forzarlo: `UPDATE index_runs SET status='failed' WHERE id='...'`

### ✅ Criterios de aceptación de la Fase 5
- Cada etapa del pipeline produce una línea de log JSON con `run_id` y `duration_ms`
- Run con `startedAt > 1h` en `processing` → marcado como `failed` en el siguiente ciclo
- `docker build .` → imagen < 500MB sin devdeps
- `WEBSITES_PORT=80` inyectado por Azure → uvicorn escucha en puerto 80 automáticamente
- `GET /docs` → 404 cuando `ENVIRONMENT=production`

---

## FASE 6 — Integración end-to-end y cutover
**Duración estimada:** 1 día
**Objetivo:** Backend Node apuntando a la nueva API. Azure Functions desconectadas.

### Tareas

#### 6.1 Cambios en `assistax-back`
- Actualizar `AZURE_FUNCTION_URL` en `.env` del backend → URL de la nueva API en Azure Web App
- Agregar `FUNCTIONS_API_KEY` como env var en el backend
- Verificar en `bulkDocumentService.ts`: enviar `publishDate: null` explícitamente cuando no existe, **nunca omitir el campo**

#### 6.2 Smoke tests de integración real
- [ ] Subir 1 PDF vía la UI → `index_run` en `completed`
- [ ] Verificar chunks en `legal_chunks` con embedding de longitud 1536
- [ ] Verificar campo `tsv` poblado → búsqueda full-text funcional
- [ ] Reprocesar un documento existente → chunks anteriores eliminados y reemplazados
- [ ] Subir el mismo PDF dos veces → segundo `index_run` en `skipped`
- [ ] Búsqueda semántica desde frontend → resultados del nuevo pipeline

#### 6.3 Comparación de calidad (si hay data histórica)
- Comparar cantidad de chunks por documento vs Azure Functions
- Verificar distribución de `chunk_type`
- Revisar manualmente 3–5 documentos que hayan fallado históricamente

#### 6.4 Descomisionar Azure Functions
- Deshabilitar Azure Function App en el Portal
- Remover `AZURE_FUNCTION_URL` antigua de las configuraciones
- Documentar fecha de cutover en el README

### ✅ Criterios de aceptación de la Fase 6
- 10 PDFs reales procesados sin error (⅓ del catálogo total)
- Búsqueda semántica y full-text funcional desde el frontend
- Azure Functions deshabilitadas
- Ningún `index_run` en estado `processing` tras 24h de operación normal

---

## Resumen de fases

| Fase | Nombre | Días | Entregable clave |
|---|---|---|---|
| 0 | Fundación | 1–2 | API corriendo en Docker, auth activa, health check compatible con Azure |
| 1 | Descarga y extracción | 2–3 | Texto por página, dedup temprana por SHA256 |
| 2 | Chunking y metadata | 2–3 | Chunks semánticos tipados, `infer_doc_type` corregido |
| 3 | Embeddings | 1–2 | Vectores 1536 dims, retry con `Retry-After` |
| 4 | BD y pipeline completo | 2 | Pipeline end-to-end, pool de 5 conexiones, 2 workers |
| 5 | Hardening | 1–2 | Logs JSON, cleanup APScheduler, Dockerfile Azure-ready |
| 6 | Cutover | 1 | Azure Functions desconectadas, sistema en producción |
| **Total** | | **10–13 días** | |
