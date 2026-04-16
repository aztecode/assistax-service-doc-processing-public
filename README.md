# Assistax PDF Processor API

API que sustituye Azure Functions para procesamiento de PDFs legales. Compatible con `assistax-back`. Stack: FastAPI + PostgreSQL + Azure Blob + Azure OpenAI.

## Quickstart (Docker Compose)

```bash
cp .env.example .env          # Completar variables requeridas
docker compose up -d db       # Esperar que PostgreSQL arranque (~5 s)
docker compose up api         # API en http://localhost:8000
```

3 comandos. La API escucha en el puerto configurado por `WEBSITES_PORT` (default 8000).

---

## Variables de entorno

| Variable | Requerida | Default | Descripción |
|----------|------------|---------|-------------|
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
| `OPENAI_MAX_CONCURRENT` | ❌ | `2` | Semáforo llamadas OpenAI |
| `ENABLE_LLM_DOC_TYPE` | ❌ | `false` | Clasificación docType vía LLM |
| `ENABLE_LLM_GENERIC_HEADING_REFINE` | ❌ | `false` | Tras el chunking, LLM corrige headings de chunks `generic` que son prosa (p. ej. negritas DOF), no título de sección. Coste: 1 llamada chat por chunk candidato (por defecto solo headings “sospechosos” salvo `LLM_GENERIC_HEADING_REFINE_ALL` o body `enableLlmGenericHeadingRefine`). |
| `LLM_GENERIC_HEADING_REFINE_ALL` | ❌ | `false` | Si `true` con el flag anterior, evalúa **todos** los genéricos sin `articleRef` en **cada** corrida (más costoso). |
| `ENABLE_PDF_TEXT_NORMALIZATION` | ❌ | `true` | Normalización texto extraído |
| `RELAXED_VISUAL_FRAME_DETECTION` | ❌ | `false` | Marcos vectoriales sin exigir `color` en trazo |
| `RELAX_PROSE_TABLE_FILTER` | ❌ | `false` | Conservar rejillas 2×N muy numéricas (filtro prosa) |
| `CLEANUP_INTERVAL_MINUTES` | ❌ | `10` | Frecuencia del job de cleanup |
| `WEBSITES_PORT` | ❌ | `8000` | Puerto inyectado por Azure Web App |
| `LOG_FORMAT` | ❌ | `json` | `json` en prod, `console` en dev |
| `ENVIRONMENT` | ❌ | `production` | `production` deshabilita `/docs` |

En el body JSON de `POST /api/process-pdf`, `enableLlmGenericHeadingRefine` (boolean opcional) controla el refinamiento LLM de headings genéricos para **esa corrida**: `true` fuerza el paso aunque `ENABLE_LLM_GENERIC_HEADING_REFINE=false` y además aplica **amplitud completa** (evalúa con el clasificador **todos** los chunks `generic` elegibles sin `articleRef`, igual que `LLM_GENERIC_HEADING_REFINE_ALL=true`, solo en esa petición). `false` desactiva el refinamiento aunque el `.env` esté en `true`. Omitido o `null` usa `ENABLE_LLM_GENERIC_HEADING_REFINE` y `LLM_GENERIC_HEADING_REFINE_ALL` del entorno (sin amplitud completa salvo que el `.env` la active).

**Logs (refinamiento headings):** `generic_heading_classifier.llm_invoke` (inmediatamente antes de cada `chat.completions.create`, con `run_id`, `chunk_no`, `blob_path`, `heading_preview`); `generic_heading_classifier.classified` / `generic_heading_classifier.llm_failed`; `heading_refinement.done` (agrega `llm_invocations`, veredictos, skips por tipo); `heading_refinement.no_eligible_chunks` (warning si ningún chunk entró al clasificador); `heading_refinement.skipped` si el paso está desactivado. Motivos por chunk (`keep_heading`, `heading_replaced`) en nivel **debug**.

---

## Diagrama del pipeline

```
POST /api/process-pdf
         │
         ▼
   ┌─────────────┐
   │ 202 Accepted │
   └─────────────┘
         │
         ▼ (background, 2 workers máx)
   ┌──────────────────┐
   │ blob_download    │  ← Download PDF + SHA256 hash
   └────────┬─────────┘
            │
   ┌────────▼─────────┐
   │ dedup temprana   │  ← Si hash existe → skipped
   └────────┬─────────┘
            │
   ┌────────▼─────────┐
   │ pdf_extraction   │  ← PyMuPDF → páginas
   └────────┬─────────┘
            │
   ┌────────▼─────────┐
   │ chunking         │  ← Chunks semánticos tipados
   └────────┬─────────┘
            │
   ┌────────▼─────────┐
   │ heading_refine   │  ← Opcional: LLM (ENABLE_LLM_GENERIC_HEADING_REFINE)
   └────────┬─────────┘
            │
   ┌────────▼─────────┐
   │ embeddings       │  ← Azure OpenAI 1536 dims (solo usa `text`, no `heading`)
   └────────┬─────────┘
            │
   ┌────────▼─────────┐
   │ db_write         │  ← legal_documents + legal_chunks
   └────────┬─────────┘
            │
   ┌────────▼─────────┐
   │ index_run =      │
   │ completed        │
   └──────────────────┘
```

---

## ¿Qué hacer si un `index_run` queda en `processing`?

El job de cleanup lo resuelve **automáticamente cada 10 minutos** (o el valor de `CLEANUP_INTERVAL_MINUTES`). Marca como `failed` los runs en `processing` con `startedAt` mayor a 1 hora.

Si necesitas forzarlo manualmente:

```sql
UPDATE index_runs SET status = 'failed', error = 'Manual reset' WHERE id = 'uuid-del-run';
```

---

## Auditoría de corpus PDF (tablas / layout)

Desde el directorio del servicio (con `.env` cargado, mismas variables que la API):

```bash
.venv/bin/python scripts/pdf_corpus_table_audit.py --root /ruta/al/corpus/pdfs
```

- Usa **`--root`** apuntando al árbol completo de leyes (p. ej. varias carpetas o ~500 PDFs), no solo un subconjunto.
- Respeta `RELAXED_VISUAL_FRAME_DETECTION` y `RELAX_PROSE_TABLE_FILTER` del `.env` para alinear métricas con extracción real.
- **`--export-discards ruta.csv`**: exporta cada tabla descartada por el filtro de prosa (`max_cols`, `pct_numeric`, `preview`, etc.) para muestreo manual.

Spike opcional `pymupdf-layout` (solo investigación; ver licencia en el script):

```bash
pip install -r requirements-layout-spike.txt
.venv/bin/python scripts/spike_pymupdf_layout.py /ruta/archivo.pdf 5
```

---

## Backfill: headings genéricos (LLM)

Para documentos ya indexados, sin re-embeber (los vectores usan solo el campo `text`):

```bash
uv run python scripts/backfill_generic_headings.py --doc-id <uuid> --enable-llm --dry-run
uv run python scripts/backfill_generic_headings.py --doc-id <uuid> --enable-llm
```

- **`--enable-llm`**: ejecuta Azure OpenAI aunque `ENABLE_LLM_GENERIC_HEADING_REFINE` esté en `false` en `.env`.
- **`--force-all-generic`**: envía al modelo todos los chunks `generic` sin `articleRef` (coste alto).
- Tras aplicar cambios, el script **regenera** `toc` / `sections` en `legal_documents.metadata` (misma lógica que `backfill_legal_toc.py`).

---

## Endpoints

| Método | Ruta | Auth | Descripción |
|--------|------|------|-------------|
| GET | `/health` | No | Health check para Azure probes |
| POST | `/api/process-pdf` | `x-functions-key` | Inicia procesamiento (202, background) |
| GET | `/docs` | No | Swagger UI (solo si `ENVIRONMENT != production`) |

---

## Docker en producción (Azure Web App)

```bash
docker build -t assistax-pdf-processor .
```

Azure Web App inyecta `WEBSITES_PORT` (típicamente 80). Uvicorn escucha en ese puerto automáticamente.

---

## Experimental: Layout V2

`extract_document_layout(pdf_bytes)` es el punto de entrada del nuevo pipeline de extracción estructural (Fase 1 del plan de reescritura). Extrae cada bloque de texto con detalle por *span* (nombre de fuente, tamaño, flags negrita/cursiva y bbox), captura tablas candidatas como señales sin filtrar, y conserva el TOC nativo del PDF. Todo el resultado es un `DocumentLayout` serializable a JSON.

**Estado actual:** aislado del pipeline productivo. No afecta `POST /api/process-pdf` ni `pipeline/runner.py`. Las fases de normalización, clasificación de bloques e integración al runner son trabajo futuro.

**Uso local (debug / exploración):**

```python
from pipeline.layout_extractor_v2 import extract_document_layout

with open("mi_ley.pdf", "rb") as f:
    layout = extract_document_layout(f.read())

print(f"Páginas: {len(layout.pages)}")
for page in layout.pages:
    print(f"  Página {page.page_number}: {len(page.blocks)} bloques, "
          f"{len(page.raw_tables)} tablas candidatas")

# Serializar a JSON para inspección
import json
print(json.dumps(json.loads(layout.model_dump_json())["metadata"], indent=2))
```

Los modelos de datos viven en `pipeline/layout_models.py` (`DocumentLayout`, `PageLayout`, `LayoutBlock`, `ExtractedSpan`, `ClassifiedBlock`, `StructuralNode`, `DocumentStructure`). Ninguno de estos módulos importa settings ni conecta a Azure/DB.

---

## Operaciones v2: backfill, evaluación y debug

Scripts CLI para validar el pipeline v2 sobre corpus real sin afectar producción.

### Backfill / reproceso puntual

Reprocesar documentos con v2 y opcionalmente comparar contra legacy:

```bash
# Un archivo local en dry-run (no toca DB)
python scripts/backfill_layout_v2.py --file tmp/pdf_legales/2025/LEYES/ley.pdf --dry-run

# Directorio local, primeros 10 PDFs
python scripts/backfill_layout_v2.py --input-dir tmp/pdf_legales --limit 10 --output-json results.json

# Por doc_id con comparación shadow (requiere DATABASE_URL)
python scripts/backfill_layout_v2.py --doc-id <uuid> --shadow-mode --output-json shadow.json

# Por blob_path forzando reproceso
python scripts/backfill_layout_v2.py --blob-path laws/2025/doc.pdf --force
```

Parámetros principales: `--doc-id`, `--blob-path`, `--file`, `--input-dir`, `--glob`, `--limit`, `--dry-run`, `--shadow-mode`, `--force`, `--output-json`, `--document-title`, `--category-id`, `--publish-date`.

### Evaluación de corpus

Evaluar un lote y obtener métricas agregadas de calidad:

```bash
# Evaluar 20 PDFs locales y exportar CSV
python scripts/evaluate_layout_v2_corpus.py --input-dir tmp/pdf_legales --limit 20 --output-csv eval.csv

# Evaluación completa con JSON y shadow
python scripts/evaluate_layout_v2_corpus.py --input-dir tmp/pdf_legales --output-json eval.json --shadow-mode

# Filtrar documentos problemáticos (score < 0.85)
python scripts/evaluate_layout_v2_corpus.py --input-dir tmp/pdf_legales --min-quality-score 0.85 --sample-errors 5

# Por blob prefix desde DB
python scripts/evaluate_layout_v2_corpus.py --blob-prefix laws/2025 --limit 50 --output-csv db_eval.csv
```

Produce: quality score por documento, distribución de severidades, conteo de errores por tipo, top documentos problemáticos, resumen agregado en JSON y/o CSV.

### Debug visual / estructural

Exportar artefactos detallados para inspección manual:

```bash
# Exportar todo (JSON + Markdown)
python scripts/export_layout_debug.py --file tmp/pdf_legales/2025/LEYES/ley.pdf --output-dir tmp/debug

# Solo calidad y chunks en Markdown
python scripts/export_layout_debug.py --file doc.pdf --output-dir tmp/debug --format md --include-quality --include-chunks

# Por doc_id, solo JSON
python scripts/export_layout_debug.py --doc-id <uuid> --output-dir tmp/debug --format json
```

Secciones exportables: `--include-layout`, `--include-classification`, `--include-structure`, `--include-quality`, `--include-chunks`. Sin flags, exporta todas.
