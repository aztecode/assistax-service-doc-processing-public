# Documentación del Repositorio Assistax (Procesamiento de PDFs Legales)

> **Nota:** Este repositorio se llama `assistax-service-doc-processing` y sustituye la implementación original `assistax-fn` (Azure Functions). El código mantiene compatibilidad con el backend `assistax-back`.

---

## 1. Estructura de directorios y archivos principales

```
assistax-service-doc-processing/
├── main.py                    # Punto de entrada FastAPI: app, lifespan, endpoints
├── settings.py                # BaseSettings con validación fail-fast de variables de entorno
├── models.py                  # ProcessPdfRequest, ProcessPdfResponse, HealthResponse
├── logging_config.py          # Configuración de structlog (json/console)
│
├── middleware/
│   ├── __init__.py
│   └── auth.py                # Validación x-functions-key (obligatoria excepto /health)
│
├── jobs/
│   └── cleanup.py             # APScheduler: index_runs stuck en processing → failed
│
├── pipeline/                  # Lógica de procesamiento de documentos
│   ├── __init__.py
│   ├── runner.py              # Orquestador principal del pipeline
│   ├── blob_download.py       # Descarga PDF desde Azure Blob Storage
│   ├── pdf_extractor.py       # Extracción texto/tablas/TOC con PyMuPDF
│   ├── pdf_text_normalization.py  # Normalización de texto (fusionar líneas, guiones)
│   ├── legal_ordinal_patterns.py  # Patrones regex para ordinales legales
│   ├── legal_chunker.py        # Chunking semántico por estructura legal
│   ├── metadata_extractor.py  # Leyenda legal, nombre de ley, tipo de documento
│   ├── doc_type_classifier.py # Clasificación docType (LLM o heurística)
│   ├── embeddings.py         # Generación de embeddings vía Azure OpenAI
│   ├── db_writer.py           # Pool PostgreSQL, upsert documentos, bulk insert chunks
│   ├── toc_builder.py        # TOC jerárquico + sections (paridad con Node buildHierarchicalOutline)
│   └── exceptions.py         # Excepciones custom (BlobNotFound, PDFExtraction, Embedding)
│
├── tests/
│   ├── conftest.py            # Fixtures: PDFs mínimos para tests
│   ├── unit/
│   │   ├── test_chunker.py
│   │   ├── test_extractor.py
│   │   ├── test_metadata.py
│   │   └── test_normalizer.py
│   └── integration/
│       ├── test_pipeline.py
│       └── schema_test.sql    # Schema mínimo para tests
│
├── Dockerfile                 # Multi-stage, compatible Azure Web App
├── requirements.txt           # Dependencias Python
├── pyproject.toml             # Ruff, Black, pytest
├── .env.example               # Plantilla de variables de entorno
└── README.md
```

---

## 2. Lógica de procesamiento de documentos

### 2.1 Chunking (`legal_chunker.py`)

- **Objetivo:** Dividir el contenido en chunks semánticos respetando la estructura legal mexicana.
- **Tipos de chunk:** `article`, `chapter`, `title`, `section`, `rule`, `fraction`, `transitorio`, `table`, `generic`.
- **Patrones detectados:**
  - Artículos: `Artículo N`, `Art. N Bis/Ter/Quáter`, formato `194-N`, `194-N-1`.
  - Libro, Título, Capítulo, Sección, Anexo.
  - Reglas, numerales, fracciones romanas (I., II.), incisos (a), b)).
  - Transitorios.
  - Tablas: marcadores `[TABLE_N]...[/TABLE_N]`.
- **Tamaño:** Máx 2000 chars por chunk; mínimo 25 chars. División por `\n\n` > `\n` > espacio.
- **Filtrado:** Elimina notas de reforma inline, paginación y citas vacías.
- **Contexto:** Cada chunk mantiene `article_ref` del artículo más reciente para contexto.
- **Tablas:** Chunks atómicos con `has_table=True`, `table_index` opcional.
- **Transitorios en el índice:**
  - **Cabecera de bloque**: se reconocen `TRANSITORIOS`, `TRANSITORIO`, `Transitorio` y `ARTÍCULOS TRANSITORIOS` (y variantes de acento/mayúsculas). Al detectar la cabecera, `inside_transitorios` pasa a `True` y se emite un **chunk contenedor** con `chunk_type="transitorio"` y `article_ref=None`; este chunk es el nodo padre en el TOC del visor.
  - Artículos dentro de la sección transitoria reciben el sufijo ` Transitorio` en `article_ref` y `heading` (ej. `"Artículo 1 Transitorio"`).
  - Ordinales transitorios (`Primero.-`, `Segundo.-`, …) generan un `article_ref` sintético: `"Artículo Primero Transitorio"`, `"Artículo Segundo Transitorio"`, etc.
  - **Reset del bloque**: si aparece un marcador estructural (`Libro`, `Título`, `Capítulo`, `Sección`) después de los transitorios, `inside_transitorios` vuelve a `False` para evitar falsos positivos en el resto del documento.
  - **Líneas fusionadas** (ej. `ARTÍCULOS TRANSITORIOS Artículo 1º.-`) se dividen automáticamente antes del análisis estructural.
  - Para re-indexar documentos ya procesados con el nuevo formato de títulos es necesario reprocesar los PDFs afectados.
  - **Estado transitorio entre páginas**: el flag `inside_transitorios` se propaga entre páginas mediante `carry_inside_transitorios` en `chunk_content`; así, los artículos numéricos en páginas posteriores al encabezado `TRANSITORIOS` reciben el sufijo ` Transitorio` en `article_ref` aunque el encabezado esté en una página anterior. El estado se apaga si aparece un marcador estructural (`Libro`, `Título`, `Capítulo`, `Sección`) en cualquier página subsiguiente.

### 2.1b Índice (TOC) en `legal_documents.metadata` (`toc_builder.py` + `db_writer.py`)

Tras insertar chunks, el pipeline fusiona en `legal_documents.metadata` (sin borrar otras claves del JSON):

| Clave | Descripción |
|-------|-------------|
| `toc` | Árbol `TocNode[]` (`id` = `chunk-{chunkNo}`, `title`, `level`, `target.page`, `children?`). |
| `sections` | Lista aplanada: `id`, `title`, `pageStart`, `pageEnd`. |
| `manifestVersion` | Versión estable: `hierarchical_chunk_based_v1-{contentHash[:12]}-{sha256(toc)[:16]}`. |
| `pageCount` | Número real de páginas del PDF (`len(pages)`). |
| `outlineStrategy` | `hierarchical_chunk_based_v1`. |
| `generatedAt` | ISO-8601 UTC. |
| `outlineStats` | `totalNodes`, `byLevel`, `maxDepth`. |
| `outlineError` | Mensaje si el builder falló (el backend Node puede usar fallback). |

**Ejemplo mínimo** (fragmento):

```json
{
  "toc": [
    {
      "id": "chunk-1",
      "title": "Título Primero",
      "level": 2,
      "target": { "page": 1 },
      "children": [
        { "id": "chunk-2", "title": "Artículo 1", "level": 5, "target": { "page": 2 } }
      ]
    }
  ],
  "manifestVersion": "hierarchical_chunk_based_v1-deadbeef1234-a1b2c3d4e5f67890",
  "pageCount": 120,
  "outlineStrategy": "hierarchical_chunk_based_v1",
  "generatedAt": "2026-03-20T12:00:00Z"
}
```

Los chunks persisten `startPage` / `endPage` (cuando la migración Prisma está aplicada) para rellenar `target.page` en el TOC.

### 2.2 Embeddings (`embeddings.py`)

- **Modelo:** Azure OpenAI (text-embedding-3-small, 1536 dims).
- **Batch size:** 100 chunks por request (máx 2048 para Azure).
- **Retry:** 5 intentos con backoff exponencial (1, 2, 4, 8, 16 s).
- **Concurrencia:** Semáforo `OPENAI_MAX_CONCURRENT` (default 2) para evitar throttling.
- **Input vacío:** Sustituido por `" "` para evitar rechazo de Azure.

### 2.3 Clasificación de tipo de documento (`doc_type_classifier.py`)

- **Modos:**
  - **LLM (ENABLE_LLM_DOC_TYPE=true):** Azure OpenAI GPT-4o-mini con prompt estructurado.
  - **Heurística (default):** `infer_doc_type()` por keywords en el título.
- **Tipos permitidos:** `ley`, `codigo`, `reglamento`, `presupuesto`, `resolucion`, `estatuto`.
- **Confianza:** `alta`, `media`, `baja` (solo cuando usa LLM).
- **Fallback:** En error de LLM → heurística silenciosa.

### 2.4 Extracción de metadatos (`metadata_extractor.py`)

- **Leyenda legal (`extract_legal_legend`):** Primeras 8 páginas.
  - `ultima_reforma`: fecha DOF.
  - `nueva_ley_dof`: día, mes, año.
  - `texto_vigente`: booleano.
  - `cantidades_actualizadas`: fecha si aplica.
- **Nombre de ley (`extract_law_name`):** Título sin paréntesis final, máx 255 chars.
- **Tipo de documento (`infer_doc_type`):** Keywords por prioridad (reglamento, código, estatuto, resolución, presupuesto, ley).

### 2.5 Extracción de PDF (`pdf_extractor.py`)

- **Librería:** PyMuPDF (`fitz`).
- **Salida:** `List[PageContent]` con `page_number`, `text`, `tables`.
- **Tablas:**
  - Extracción con `page.find_tables()`.
  - Filtro: prosa en grid (1 columna, celdas largas, sin numéricos) vs tablas tarifarias (cuota, rango, tasa, etc.).
  - Marcadores: `[TABLE_N]...[/TABLE_N]` en Markdown.
- **Texto:** Excluye regiones que se solapan con tablas reales.
- **Normalización:** `ENABLE_PDF_TEXT_NORMALIZATION` activa `normalize_pdf_text` (fusionar líneas cortadas, guiones, eliminar headers repetidos).
- **TOC:** `doc.get_toc(simple=True)` → `{level, title, page}`.

### 2.6 Normalización de texto (`pdf_text_normalization.py`)

- **Reglas:**
  - Preservar saltos antes de Artículo, Capítulo, Título, incisos, romanos, etc.
  - Unir palabras con guión (`obliga-` + `ción` → `obligación`).
  - Unir líneas con `_merge_score >= 2` (terminación en coma, siguiente con minúscula, etc.).
  - Eliminar headers repetidos ≥ 3 veces.
  - Eliminar paginación (`1 de 5`, `Página 3`).

---

## 3. Flujo principal del pipeline

```
POST /api/process-pdf (payload: runId, blobPath, documentTitle, categoryId, publishDate)
         │
         ▼
   ┌─────────────────┐
   │ 202 Accepted    │  ← Respuesta inmediata; pipeline en background
   └────────┬────────┘
            │
            ▼ (ThreadPoolExecutor, semáforo PDF_WORKER_THREADS)
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │ 1. update_index_run(conn, runId, "processing")                               │
   │ 2. blob_download(payload.blobPath) → pdf_bytes                               │
   │ 3. extract_pdf(pdf_bytes) → pages, toc                                       │
   │ 4. extract_legal_legend(first_8_pages) → metadata                            │
   │ 5. extract_law_name(documentTitle)                                           │
   │ 6. chunk_content(pages) → chunks                                              │
   │ 7. classify_doc_type(documentTitle, headings) → doc_type, confidence         │
   │ 8. embed_chunks(chunks, progress_callback) → embeddings                       │
   │ 9. content_hash = SHA256(pdf_bytes)                                          │
   │ 10. upsert_legal_document(...) → document_id                                  │
   │ 11. delete_existing_chunks(document_id)                                       │
   │ 12. insert_chunks_bulk(document_id, chunks, embeddings)                        │
   │ 12b. persist_legal_outline_from_chunks → merge metadata (toc, sections, …)   │
   │ 13. update_index_run(runId, "completed", docs_indexed=1, chunks_total=N)     │
   └─────────────────────────────────────────────────────────────────────────────┘
            │
            ▼ (on error)
   update_index_run(runId, "failed", error_log=...)
```

**Nota:** Todo el pipeline se ejecuta dentro de una única conexión del pool, con commits puntuales en el progreso de embeddings.

---

## 4. Configuración y variables de entorno

Definidas en `settings.py` (BaseSettings con validación al arrancar):

| Variable | Requerida | Default | Descripción |
|----------|------------|---------|-------------|
| `DATABASE_URL` | ✅ | — | PostgreSQL connection string |
| `AZURE_STORAGE_CONNECTION_STRING` | ✅ | — | Azure Blob Storage |
| `AZURE_BLOB_CONTAINER` | ❌ | `laws` | Contenedor de PDFs |
| `AZURE_OPENAI_ENDPOINT` | ✅ | — | Endpoint Azure OpenAI |
| `AZURE_OPENAI_API_KEY` | ✅ | — | API Key Azure OpenAI |
| `AZURE_OPENAI_API_VERSION` | ✅ | — | Versión de la API |
| `AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT` | ✅ | — | Deployment de embeddings |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | ❌ | `gpt-4o-mini` | Para clasificación docType vía LLM |
| `FUNCTIONS_API_KEY` | ✅ | — | Header `x-functions-key` (auth) |
| `PDF_WORKER_THREADS` | ❌ | `2` | Máx pipelines concurrentes |
| `DB_POOL_MIN_CONN` | ❌ | `1` | Pool mínimo conexiones BD |
| `DB_POOL_MAX_CONN` | ❌ | `5` | Pool máximo conexiones BD |
| `OPENAI_MAX_CONCURRENT` | ❌ | `2` | Semáforo llamadas OpenAI |
| `ENABLE_LLM_DOC_TYPE` | ❌ | `false` | Clasificación docType vía LLM |
| `ENABLE_PDF_TEXT_NORMALIZATION` | ❌ | `true` | Normalización texto extraído |
| `CLEANUP_INTERVAL_MINUTES` | ❌ | `10` | Frecuencia job de cleanup |
| `WEBSITES_PORT` | ❌ | `8000` | Puerto (Azure Web App inyecta) |
| `LOG_FORMAT` | ❌ | `json` | `json` (prod) / `console` (dev) |
| `ENVIRONMENT` | ❌ | `production` | `production` oculta `/docs` |

---

## 5. Dependencias (`requirements.txt`)

```txt
# API
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
gunicorn>=21.2.0

# PDF y procesamiento
PyMuPDF>=1.24.0
psycopg2-binary>=2.9.0

# Azure
openai>=1.3.0
azure-storage-blob>=12.19.0

# Configuración y validación
pydantic>=2.0.0
pydantic-settings>=2.0.0

# Observabilidad
structlog>=24.0.0
slowapi>=0.1.9

# Jobs
apscheduler>=3.10.0

# Tests
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

---

## Esquema de base de datos (referencia)

- **legal_documents:** Documento legal (blobPath, contentHash, docType, lawName, metadata).
- **legal_chunks:** Chunks con embedding (vector 1536), tsv (to_tsvector español + unaccent).
- **index_runs:** Seguimiento de ejecución (status: processing/completed/failed/skipped).

Requiere extensiones: `vector` (pgvector), `unaccent`.
