# Stack tecnológico — Assistax PDF Processor API

API que sustituye Azure Functions para procesamiento de PDFs legales. Compatible con `assistax-back`.

---

## Lenguaje y runtime

| Tecnología | Versión | Uso |
|------------|---------|-----|
| **Python** | 3.12 | Lenguaje base |
| **Uvicorn** | ≥ 0.29.0 | Servidor ASGI para FastAPI |

---

## Framework web y API

| Tecnología | Versión | Uso |
|------------|---------|-----|
| **FastAPI** | ≥ 0.111.0 | Framework web, API REST |
| **Pydantic** | ≥ 2.0.0 | Validación de modelos y request/response |
| **Pydantic Settings** | ≥ 2.0.0 | Configuración desde variables de entorno |

---

## Base de datos

| Tecnología | Versión | Uso |
|------------|---------|-----|
| **PostgreSQL** | 16 | Base de datos relacional |
| **pgvector** | — | Extensión para vectores (embeddings 1536 dims) |
| **unaccent** | — | Extensión para búsqueda full-text sin acentos |
| **psycopg2-binary** | ≥ 2.9.0 | Driver PostgreSQL (pool de conexiones) |

---

## Azure

| Servicio | Uso |
|----------|-----|
| **Azure Blob Storage** | Almacenamiento de PDFs (`azure-storage-blob` ≥ 12.19.0) |
| **Azure OpenAI** | Embeddings y clasificación de documentos (`openai` ≥ 1.3.0) |
| **Azure Web App** | Hosting de la API (contenedor Docker) |

---

## Procesamiento de documentos

| Tecnología | Versión | Uso |
|------------|---------|-----|
| **PyMuPDF** | ≥ 1.24.0 | Extracción de texto e imágenes desde PDFs |

---

## Observabilidad y hardening

| Tecnología | Versión | Uso |
|------------|---------|-----|
| **structlog** | ≥ 24.0.0 | Logging estructurado (JSON en prod, console en dev) |
| **SlowAPI** | ≥ 0.1.9 | Rate limiting (20 req/min en `/api/process-pdf`) |
| **APScheduler** | ≥ 3.10.0 | Job de cleanup (index_runs stuck → failed) |

---

## Contenedores y orquestación

| Tecnología | Uso |
|------------|-----|
| **Docker** | Imagen multi-stage, compatible Azure Web App |
| **Docker Compose** | Desarrollo local: API + PostgreSQL |
| **pgvector/pgvector:pg16** | Imagen oficial PostgreSQL con extensión vector |

---

## Tests y calidad de código

| Tecnología | Versión | Uso |
|------------|---------|-----|
| **pytest** | ≥ 8.0.0 | Framework de tests |
| **pytest-asyncio** | ≥ 0.23.0 | Tests async |
| **Ruff** | — | Linter (reemplaza flake8, isort) |
| **Black** | — | Formateador de código |

---

## Arquitectura resumida

```
┌─────────────────────────────────────────────────────────────┐
│                    Azure Web App (Docker)                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  FastAPI + Uvicorn (1 worker)                           │ │
│  │  ThreadPoolExecutor (2 workers pipeline)                 │ │
│  │  APScheduler (cleanup cada 10 min)                     │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────────┐
│ Azure Blob     │  │ Azure OpenAI   │  │ PostgreSQL 16       │
│ Storage (PDFs) │  │ (embeddings)   │  │ pgvector + unaccent  │
└────────────────┘  └────────────────┘  └────────────────────┘
```

---

## Dependencias directas (requirements.txt)

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
