# Plan detallado para Cursor — Reescritura del procesamiento legal en `assistax-service-doc-processing-public`

## Objetivo
Reconvertir el pipeline actual de extracción heurística en un pipeline de **estructura legal asistida por LLM + validación automática**, sin romper el contrato actual del servicio FastAPI.

## Resultado esperado
Al terminar, el sistema debe:
- preservar el orden real del documento;
- distinguir encabezados/footers DOF del contenido normativo;
- anclar tablas y notas en su posición correcta;
- generar un árbol legal confiable (`toc`, `sections`, `article_ref`);
- rechazar o marcar documentos con calidad insuficiente antes de indexarlos.

---

# Reglas de ejecución para Cursor

1. **No reescribir todo de golpe.** Trabajar por fases pequeñas con commits atómicos.
2. **No romper endpoints existentes**: mantener `POST /api/process-pdf` y `GET /health`.
3. **No eliminar el pipeline actual al inicio**; introducir un pipeline nuevo detrás de feature flags.
4. **Cada fase debe cerrar con tests y un checklist de aceptación.**
5. **Toda salida estructural nueva debe ser serializable a JSON.**
6. **Todo comportamiento ambiguo debe ir a logs estructurados.**
7. **No usar el LLM para embeddings ni para lógica trivial.** Usarlo solo para clasificación y reconstrucción estructural.

---

# Contexto técnico que Cursor debe asumir

## Estado actual del repo
- FastAPI como entrypoint en `main.py`
- configuración en `settings.py`
- procesamiento principal en `pipeline/runner.py`
- extracción PDF en `pipeline/pdf_extractor.py`
- normalización en `pipeline/pdf_text_normalization.py`
- chunking legal en `pipeline/legal_chunker.py`
- TOC en `pipeline/toc_builder.py`
- embeddings en `pipeline/embeddings.py`
- escritura DB en `pipeline/db_writer.py`
- tests bajo `tests/`

## Problemas reales que se deben resolver
- encabezados y pies DOF mezclados con contenido;
- fechas interpretadas como títulos;
- referencias a artículos dentro de prosa interpretadas como encabezados reales;
- tablas detectadas donde no existen;
- tablas reales colocadas al final de la página y no en su posición lógica;
- índices incompletos o desordenados;
- texto subrayado o contenido visualmente especial que se pierde o se salta.

---

# Arquitectura objetivo

## Nuevo flujo lógico
1. `blob_download`
2. `layout_extraction_v2`
3. `layout_normalization_v2`
4. `block_classification_v2` (heurística + LLM)
5. `structure_reconstruction_v2`
6. `quality_validation_v2`
7. `chunk_projection_v2`
8. `embeddings`
9. `db_write`

## Principio clave
El sistema ya no debe construir el índice directamente desde texto plano o chunks lineales.
Debe construir primero una **representación estructural intermedia** y de ahí derivar TOC, chunks y metadata.

---

# Modelo intermedio obligatorio

Cursor debe introducir un nuevo módulo de modelos estructurales.

## Crear archivo
`pipeline/layout_models.py`

## Definir dataclasses o pydantic models

### `ExtractedSpan`
Campos mínimos:
- `text: str`
- `bbox: tuple[float, float, float, float]`
- `font_size: float | None`
- `font_name: str | None`
- `is_bold: bool | None`
- `is_italic: bool | None`
- `page_number: int`
- `block_no: int | None`
- `line_no: int | None`
- `span_no: int | None`

### `LayoutBlock`
Campos mínimos:
- `block_id: str`
- `page_number: int`
- `bbox: tuple[float, float, float, float]`
- `text: str`
- `kind: str`  # text, table, boxed_note, header, footer, unknown
- `reading_order: int`
- `spans: list[ExtractedSpan]`
- `source: str`  # pymupdf_text, pymupdf_table, inferred, merged
- `metadata: dict`

### `PageLayout`
Campos mínimos:
- `page_number: int`
- `width: float`
- `height: float`
- `blocks: list[LayoutBlock]`
- `raw_tables: list[dict]`
- `raw_drawings: list[dict]`

### `DocumentLayout`
Campos mínimos:
- `pages: list[PageLayout]`
- `native_toc: list[dict]`
- `metadata: dict`

### `ClassifiedBlock`
Campos mínimos:
- `block_id: str`
- `page_number: int`
- `label: str`
- `confidence: float`
- `reason: str | None`
- `llm_used: bool`
- `normalized_text: str`
- `metadata: dict`

### `StructuralNode`
Campos mínimos:
- `node_id: str`
- `node_type: str`  # document, book, title, chapter, section, article, fraction, transitory, table, note, paragraph
- `heading: str | None`
- `text: str | None`
- `article_ref: str | None`
- `page_start: int | None`
- `page_end: int | None`
- `children: list[StructuralNode]`
- `source_block_ids: list[str]`
- `metadata: dict`

### `DocumentStructure`
Campos mínimos:
- `root: StructuralNode`
- `toc: list[dict]`
- `sections: list[dict]`
- `quality_report: dict`
- `metadata: dict`

---

# Fase 1 — Base técnica del layout v2

## Objetivo
Agregar extracción rica por bloques sin tocar todavía la lógica actual de chunking productiva.

## Tareas

### 1.1 Crear módulo nuevo
Crear `pipeline/layout_extractor_v2.py`

### 1.2 Implementar función principal
```python
extract_document_layout(pdf_bytes: bytes) -> DocumentLayout
```

### 1.3 Requerimientos
- usar PyMuPDF;
- extraer texto por `dict` o `rawdict`, no solo por `blocks` simples;
- conservar `bbox`, spans, estilo y orden por página;
- capturar tablas detectadas como candidatos, pero sin asumir que todas son reales;
- capturar drawings/rectangles como señales visuales;
- almacenar TOC nativo si existe;
- no filtrar aún headers/footers en esta fase.

### 1.4 Crear helpers internos
- `_extract_text_blocks_from_page(page)`
- `_extract_candidate_tables_from_page(page)`
- `_extract_visual_frames_from_page(page)`
- `_compute_reading_order(blocks, page_width, page_height)`

### 1.5 Reglas de implementación
- cada bloque debe tener `block_id` estable: `p{page}_b{index}`;
- eliminar bloques vacíos o whitespace-only;
- normalizar saltos múltiples, pero no colapsar estructura legal todavía;
- registrar en logs cuántos bloques y tablas candidatas salieron por página.

## Tests de esta fase
Crear `tests/unit/test_layout_extractor_v2.py`

Cubrir:
- devuelve `DocumentLayout` no vacío para PDF simple;
- todos los bloques tienen `block_id`, `page_number`, `bbox`, `text`;
- `reading_order` se asigna a todos los bloques;
- si existe TOC nativo, se conserva;
- si no existe, retorna lista vacía sin fallar.

## Criterio de aceptación
- extracción estructural funciona sin romper el pipeline actual;
- test unitario verde;
- logs muestran bloques por página.

---

# Fase 2 — Normalización estructural previa

## Objetivo
Separar header/footer, bloques repetidos, y limpiar basura visual antes de clasificar estructura.

## Crear módulo
`pipeline/layout_normalizer_v2.py`

## Función principal
```python
normalize_document_layout(layout: DocumentLayout) -> DocumentLayout
```

## Tareas

### 2.1 Detectar repetición de bloques por páginas
Implementar detector de header/footer por similitud:
- comparar bloques en zona superior e inferior de cada página;
- permitir variaciones leves de fecha, folio y numeración;
- usar normalización textual previa (`uppercase`, sin números de página, sin fechas exactas cuando aplique).

### 2.2 Marcar bloques como `header` o `footer`
No eliminarlos físicamente al inicio; solo etiquetarlos en `kind` y `metadata`.

### 2.3 Detectar bloques sospechosos de índice inicial
Señales:
- alta densidad de referencias `Artículo X`, `Capítulo`, `Sección`;
- secuencia corta sin cuerpo normativo;
- muchas líneas cortas consecutivas.

Marcar `metadata["possible_index_zone"] = True`.

### 2.4 Fusionar bloques partidos artificialmente
Solo cuando:
- son contiguos verticalmente;
- mismo estilo base;
- siguiente bloque inicia en minúscula o continuación natural;
- no parece encabezado legal;
- no cruza tabla.

## Tests
Crear `tests/unit/test_layout_normalizer_v2.py`

Casos:
- header repetido en varias páginas queda marcado;
- footer DOF queda marcado;
- índice inicial se marca como `possible_index_zone`;
- dos bloques de párrafo contiguos se fusionan.

## Criterio de aceptación
- headers/footers ya no dependen de igualdad exacta;
- el layout normalizado conserva trazabilidad a bloques originales.

---

# Fase 3 — Clasificación de bloques

## Objetivo
Clasificar cada bloque según su rol jurídico real.

## Crear módulo
`pipeline/block_classifier_v2.py`

## Etiquetas obligatorias
- `document_title`
- `book_heading`
- `title_heading`
- `chapter_heading`
- `section_heading`
- `article_heading`
- `article_body`
- `fraction`
- `inciso`
- `transitory_heading`
- `transitory_item`
- `table`
- `editorial_note`
- `page_header`
- `page_footer`
- `index_block`
- `annex_heading`
- `annex_body`
- `unknown`

## Estrategia

### 3.1 Clasificador híbrido
Primero heurístico, luego LLM solo en ambiguos.

### 3.2 Crear reglas heurísticas fuertes
En `pipeline/block_rules_v2.py`:
- regex de artículos reales;
- detección de fechas para bloquear falsos títulos;
- detección de nombres largos de decreto que aparecen como prosa;
- patrones de transitorios;
- detección de tablas textuales vs tablas reales;
- detección de encabezado DOF.

### 3.3 Integrar LLM para ambiguos
Crear `pipeline/block_classifier_llm_v2.py`

Función:
```python
classify_ambiguous_block(block, prev_blocks, next_blocks, document_metadata) -> dict
```

#### Reglas LLM
- salida obligatoria JSON;
- temperature 0;
- labels restringidos a enumeración cerrada;
- incluir contexto de vecinos, página, estilo y flags visuales;
- timeout y retry;
- fallback seguro a heurística si falla.

### 3.4 Settings nuevos
Agregar en `settings.py`:
- `ENABLE_LAYOUT_V2`
- `ENABLE_BLOCK_LLM_CLASSIFIER_V2`
- `BLOCK_LLM_MAX_CONCURRENT`
- `BLOCK_LLM_BATCH_SIZE`
- `LAYOUT_V2_SHADOW_MODE`

## Tests
Crear `tests/unit/test_block_classifier_v2.py`

Cubrir:
- fecha en mayúsculas no se clasifica como title/chapter/section;
- referencia a “artículo 8” dentro de prosa no se vuelve `article_heading`;
- header DOF se clasifica como `page_header`;
- ordinal transitorio se clasifica como `transitory_item`.

## Criterio de aceptación
- falsos positivos de títulos y artículos caen de forma evidente en muestras del reporte.

---

# Fase 4 — Reconstrucción del árbol jurídico

## Objetivo
Construir una representación legal jerárquica robusta.

## Crear módulo
`pipeline/structure_builder_v2.py`

## Función principal
```python
build_document_structure(classified_layout: DocumentLayout) -> DocumentStructure
```

## Reglas de reconstrucción

### 4.1 Jerarquía permitida
`document > book > title > chapter > section > article > paragraph/fraction/inciso/table/note`

### 4.2 Tablas
- las tablas deben insertarse exactamente donde aparecen en el orden de lectura;
- si una tabla cae entre bloques del mismo artículo, pertenece a ese artículo;
- nunca se emiten al final de la página por default.

### 4.3 Transitorios
- `TRANSITORIOS` crea nodo padre;
- `Primero.-`, `Segundo.-` etc. generan nodos hijos estructurados;
- `article_ref` sintético obligatorio para cada transitorio ordinal.

### 4.4 Índice inicial
- los bloques `index_block` no deben entrar al cuerpo normativo principal;
- pueden guardarse en metadata si se quiere trazabilidad.

### 4.5 Notas editoriales
- si son `editorial_note`, excluir del TOC principal;
- mantener nodo si aportan visibilidad, pero no como heading navegable.

## Tests
Crear `tests/unit/test_structure_builder_v2.py`

Cubrir:
- tabla queda anclada entre artículo primero y segundo;
- índice inicial no contamina el árbol;
- transitorios crean nodos correctos;
- headings repetidos no duplican artículos.

## Criterio de aceptación
- el árbol final es la nueva fuente de verdad para TOC y sections.

---

# Fase 5 — Validación automática de calidad

## Objetivo
No indexar documentos estructuralmente malos sin dejar evidencia.

## Crear módulo
`pipeline/quality_validator_v2.py`

## Función principal
```python
validate_document_structure(structure: DocumentStructure) -> dict
```

## Checks obligatorios
- `has_visible_table_tokens`: no debe existir `TABLE`, `[TABLE_1]`, etc. como heading visible;
- `article_sequence_health`: no saltos absurdos por contaminación del índice;
- `header_footer_bleed`: porcentaje de bloques DOF filtrados del cuerpo;
- `date_heading_false_positive_count`;
- `orphan_tables_count`;
- `unknown_block_ratio`;
- `toc_duplicate_ratio`;
- `article_ref_coverage`.

## Política inicial
- score >= 0.85 → indexar
- 0.70 a 0.84 → indexar con warning y log fuerte
- < 0.70 → marcar `failed_quality_gate`

## Requerimientos
- guardar `quality_report` completo;
- loggear razones exactas de rechazo;
- permitir override por flag en dev, no en prod.

## Tests
Crear `tests/unit/test_quality_validator_v2.py`

## Criterio de aceptación
- documentos con índice roto o headings `TABLE` fallan el quality gate.

---

# Fase 6 — Proyección del árbol a chunks compatibles

## Objetivo
Mantener compatibilidad con búsqueda y embeddings sin perder la nueva estructura.

## Crear módulo
`pipeline/chunk_projector_v2.py`

## Función principal
```python
project_structure_to_chunks(structure: DocumentStructure) -> list[Chunk]
```

## Reglas
- cada chunk debe derivar de nodos estructurales reales;
- `heading`, `article_ref`, `chunk_type`, `page_start`, `page_end` salen del árbol;
- tablas y notas se mantienen atómicas;
- el texto del chunk no debe incluir headers/footers;
- `metadata` debe incluir `source_block_ids`.

## Importante
No cambiar el contrato downstream de embeddings o DB si no es necesario.
Agregar solo campos compatibles o metadata extendida.

## Tests
Crear `tests/unit/test_chunk_projector_v2.py`

Casos:
- artículo con tabla produce chunks coherentes;
- transitorio ordinal produce `article_ref` estable;
- chunk no contiene encabezado de página.

## Criterio de aceptación
- `legal_chunks` sigue siendo usable por el backend actual.

---

# Fase 7 — Integración gradual en runner

## Objetivo
Conectar todo al pipeline real sin romper producción.

## Tareas

### 7.1 Modificar `pipeline/runner.py`
Introducir dos rutas:
- `run_pipeline_legacy(...)`
- `run_pipeline_v2(...)`

### 7.2 Selección por flag
Si `ENABLE_LAYOUT_V2=true`:
- ejecutar pipeline nuevo.
Si además `LAYOUT_V2_SHADOW_MODE=true`:
- ejecutar ambos y comparar outputs sin usar aún el v2 como fuente final.

### 7.3 Crear comparador de resultados
Nuevo módulo:
`pipeline/shadow_compare_v2.py`

Medir:
- número de chunks;
- cobertura de `article_ref`;
- headings basura;
- tablas huérfanas;
- secuencia de TOC.

### 7.4 Persistencia de metadata extendida
En `db_writer.py`, guardar en `legal_documents.metadata`:
- `pipeline_version`
- `quality_report`
- `shadow_compare`
- `structure_summary`

## Tests de integración
Crear o ampliar `tests/integration/test_pipeline_v2.py`

Casos:
- documento nuevo con v2 → `completed`;
- documento de mala calidad → `failed_quality_gate` o equivalente;
- shadow mode produce comparación sin romper indexación.

## Criterio de aceptación
- se puede activar por env sin tocar el backend consumidor.

---

# Fase 8 — Backfill y utilidades operativas

## Objetivo
Poder reprocesar corpus existente sin dolor humano innecesario.

## Tareas

### 8.1 Script de backfill v2
Crear `scripts/backfill_layout_v2.py`

Parámetros:
- `--doc-id`
- `--blob-path`
- `--limit`
- `--dry-run`
- `--shadow-mode`
- `--force`

### 8.2 Script de evaluación de corpus
Crear `scripts/evaluate_layout_v2_corpus.py`

Salida:
- CSV/JSON con quality score por documento;
- top fallas por categoría;
- distribución de tipos de error;
- lista de documentos que requieren revisión manual.

### 8.3 Export de debug visual
Crear `scripts/export_layout_debug.py`

Debe permitir exportar por página:
- bloques con bbox;
- labels clasificados;
- tablas candidatas;
- headers/footers detectados.

Esto ayuda cuando PyMuPDF se pone creativo. Y ya sabemos que eso pasa.

---

# Fase 9 — Observabilidad y métricas

## Objetivo
Saber exactamente cuándo el documento sale bien, sale mal, o sale “legalmente sospechoso”.

## Eventos de log obligatorios
- `layout_v2.extraction.started`
- `layout_v2.extraction.completed`
- `layout_v2.normalization.completed`
- `layout_v2.classification.completed`
- `layout_v2.structure.completed`
- `layout_v2.quality.completed`
- `layout_v2.quality.failed`
- `layout_v2.shadow.compare`

## Campos mínimos de log
- `run_id`
- `blob_path`
- `pages`
- `blocks_total`
- `classified_blocks_total`
- `unknown_blocks_total`
- `tables_total`
- `headers_detected`
- `footers_detected`
- `quality_score`
- `pipeline_version`
- `duration_ms`

## Métricas de negocio técnicas
- `% documentos aprobados por quality gate`
- `% documentos con índice válido`
- `% chunks con article_ref`
- `% tablas huérfanas`
- `% headings basura`

---

# Fase 10 — Limpieza y promoción final

## Objetivo
Promover v2 como pipeline por defecto cuando la evidencia lo justifique.

## Requisitos previos
- corpus piloto de al menos 20–30 PDFs problemáticos;
- comparación legacy vs v2;
- score estable;
- sin regresiones críticas en integración.

## Tareas
- hacer `ENABLE_LAYOUT_V2=true` en staging;
- mantener legacy detrás de flag por una ventana corta;
- documentar rollback;
- actualizar README y documentación del repo.

## No hacer todavía
- eliminar completamente el pipeline legacy en el mismo PR de promoción.

---

# Cambios específicos por archivo

## Archivos nuevos
- `pipeline/layout_models.py`
- `pipeline/layout_extractor_v2.py`
- `pipeline/layout_normalizer_v2.py`
- `pipeline/block_rules_v2.py`
- `pipeline/block_classifier_llm_v2.py`
- `pipeline/block_classifier_v2.py`
- `pipeline/structure_builder_v2.py`
- `pipeline/quality_validator_v2.py`
- `pipeline/chunk_projector_v2.py`
- `pipeline/shadow_compare_v2.py`
- `scripts/backfill_layout_v2.py`
- `scripts/evaluate_layout_v2_corpus.py`
- `scripts/export_layout_debug.py`
- nuevos tests unitarios e integración por módulo

## Archivos a modificar
- `settings.py`
- `pipeline/runner.py`
- `pipeline/db_writer.py`
- `README.md`
- `DOCUMENTACION_REPOSITORIO_ASSISTAX.md`

---

# Feature flags requeridos

Agregar en `settings.py`:

```python
ENABLE_LAYOUT_V2: bool = False
LAYOUT_V2_SHADOW_MODE: bool = False
ENABLE_BLOCK_LLM_CLASSIFIER_V2: bool = True
BLOCK_LLM_MAX_CONCURRENT: int = 2
BLOCK_LLM_BATCH_SIZE: int = 8
LAYOUT_V2_MIN_QUALITY_SCORE: float = 0.85
ALLOW_LOW_QUALITY_INDEX_IN_NON_PROD: bool = True
```

---

