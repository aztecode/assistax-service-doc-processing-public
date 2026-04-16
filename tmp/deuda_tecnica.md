# Deuda técnica — Pipeline v2 de procesamiento documental

Registro acumulativo de deuda técnica identificada durante la reescritura.
Cada entrada indica la fase donde se originó, el impacto y la fase sugerida para resolverla.

---

## DT-001 · Mapeo de `dict[str, object]` hacia `Chunk` legacy

| Campo | Valor |
|---|---|
| **Origen** | Fase 6 — `chunk_projector_v2.py` |
| **Resolver en** | Fase 7 (integración al runner) |
| **Severidad** | Media |

### Descripción

`project_structure_to_chunks()` devuelve `list[dict[str, object]]` con campos enriquecidos (`source_block_ids`, `metadata`), pero el pipeline productivo actual (`runner.py` → `db_writer.py` → `embeddings.py`) espera instancias del dataclass `Chunk` definido en `pipeline/legal_chunker.py`.

### Campos pendientes de generar

| Campo legacy | Fuente esperada |
|---|---|
| `chunk_no` | Índice secuencial 1-based; se asigna al construir la lista final |
| `has_table` | `chunk_type == "table"` |
| `table_index` | Requiere correlación con la extracción original de tablas |

### Acción requerida

En Fase 7, crear una función de mapeo `_chunk_dicts_to_legacy(chunks: list[dict]) -> list[Chunk]` o adaptar `db_writer.py` para aceptar el nuevo formato directamente. Evaluar cuál opción minimiza cambios downstream.

---

## DT-002 · Sin estrategia de partición por tamaño/tokens

| Campo | Valor |
|---|---|
| **Origen** | Fase 6 — `chunk_projector_v2.py` |
| **Resolver en** | Fase 7 o Fase 8 |
| **Severidad** | Media |

### Descripción

No se implementa truncamiento ni split de chunks por cantidad de tokens. Un artículo extenso (o un transitorio con muchos párrafos) puede producir un chunk legalmente correcto pero que exceda el contexto del modelo de embeddings.

### Impacto

- Embeddings truncados silenciosamente por el modelo → pérdida de semántica al final del chunk.
- Posible degradación de retrieval para artículos largos.

### Acción requerida

Definir un umbral de tokens (e.g. 512 o 1024 según el modelo de embeddings usado) y una estrategia de split que preserve la coherencia legal:
- Partir por fracciones si existen.
- Partir por párrafos como fallback.
- Mantener `source_block_ids` y `page_start/page_end` coherentes en cada sub-chunk.

---

## DT-003 · Tablas con texto vacío producen chunks con embedding pobre

| Campo | Valor |
|---|---|
| **Origen** | Fase 6 — `chunk_projector_v2.py` |
| **Resolver en** | Fase 7+ |
| **Severidad** | Baja |

### Descripción

Las tablas se proyectan como chunks atómicos independientes. Si el nodo `table` tiene `text=""` (e.g. tabla extraída solo como imagen o coordenadas sin OCR), el chunk resultante tendrá texto vacío.

### Impacto

- El embedding será un vector de baja información.
- La tabla no será recuperable por búsqueda semántica.

### Acción requerida

Evaluar si enriquecer el texto de tabla con contexto del nodo padre (heading del artículo, por ejemplo) o marcar el chunk como `non_embeddable` para excluirlo del índice vectorial sin perderlo en el outline.

---

## DT-004 · Taxonomía de `chunk_type` entre v2 y legacy

| Campo | Valor |
|---|---|
| **Origen** | Fase 6 — `chunk_projector_v2.py` |
| **Resolver en** | Fase 7 |
| **Severidad** | Baja |

### Descripción

El projector v2 usa tipos como `article`, `fraction`, `inciso`, `transitory`, `annex`, `paragraph`, `table`, `boxed_note`. El legacy usa una taxonomía parcialmente superpuesta: `generic`, `book`, `title`, `chapter`, `section`, `annex`, `article`, `rule`, `numeral`, `fraction`, `inciso`, `transitorio`, `table`, `boxed_note`.

### Diferencias notables

| v2 | Legacy | Nota |
|---|---|---|
| `transitory` | `transitorio` | Nombre diferente |
| `paragraph` | `generic` | Equivalente funcional |
| — | `rule`, `numeral` | No usados en v2 |

### Acción requerida

En la función de mapeo de Fase 7, normalizar los tipos v2 al vocabulario que el frontend y los índices de búsqueda esperan. Verificar con el equipo de frontend qué valores de `chunkType` se usan en filtros y UI.

---

## DT-005 · Orden de chunks: fracciones después del artículo

| Campo | Valor |
|---|---|
| **Origen** | Fase 6 — `chunk_projector_v2.py` |
| **Resolver en** | Fase 7 (validar) |
| **Severidad** | Baja |

### Descripción

Los chunks de fracción se emiten inmediatamente después del chunk principal del artículo, respetando el orden DFS. Esto es semánticamente correcto, pero difiere del legacy donde todo el contenido de un artículo (fracciones incluidas) podía estar en un solo chunk.

### Impacto

- La reconstrucción de outline en el frontend podría necesitar agrupar chunks por `article_ref`.
- El conteo de `chunk_no` cambiará respecto al pipeline legacy para el mismo documento.

### Acción requerida

Validar en Fase 7 que el outline builder y el frontend manejan correctamente múltiples chunks por artículo. Si no, considerar colapsar fracciones de vuelta al artículo como opción configurable.

---

## DT-006 · Leyenda legal frágil en v2 puro

| Campo | Valor |
|---|---|
| **Origen** | Fase 7 — `pipeline/runner.py` (`_extract_text_from_v2_pages`) |
| **Resolver en** | Fase 8 o antes de promover v2 en staging |
| **Severidad** | Media |

### Descripción

Cuando v2 es primary, el runner reconstruye texto de las primeras 8 páginas caminando el árbol estructural con `_extract_text_from_v2_pages(...)`. El árbol v2 excluye headers, footers y bloques DOF marcados durante la normalización, pero esos fragmentos pueden contener la leyenda legal (fecha de publicación, última reforma, texto vigente).

### Impacto

- La leyenda legal podría extraerse incompleta o vacía si el contenido relevante fue filtrado como header/footer DOF.
- Campos como `legalLegend`, `lastReformDate` y `publicationDate` podrían no persistirse en `legal_documents.metadata`.

### Acción requerida

Extraer la leyenda directamente desde las páginas crudas del `DocumentLayout` (antes de normalización/clasificación), en vez de reconstruir texto desde el árbol post-clasificación. Monitorear la calidad de leyenda en shadow mode antes de promover v2 a staging.

---

## DT-007 · Outline de v2 pasa por el camino legacy

| Campo | Valor |
|---|---|
| **Origen** | Fase 7 — `pipeline/runner.py` (`run_pipeline_v2`) |
| **Resolver en** | Fase 8 o staging |
| **Severidad** | Baja |

### Descripción

En `run_pipeline_v2`, el outline se persiste con `persist_legal_outline(...)` usando chunks adaptados al formato legacy y `native_toc` vacío, en vez de derivarse directamente de `structure.toc` y `structure.sections` que ya contienen el TOC completo construido por el árbol v2.

### Impacto

- El TOC persistido es una reconstrucción desde chunks (via `toc_builder`) que puede diferir del árbol v2 original.
- Se pierde granularidad estructural que `structure.toc` ya tiene (por ejemplo, `node_id`, relaciones jerárquicas exactas).
- Duplicación de trabajo: el árbol ya fue construido y validado, pero se recalcula el outline desde los chunks.

### Acción requerida

Crear un builder de outline nativo para v2 que persista `structure.toc` y `structure.sections` directamente en `legal_documents.metadata`, sin pasar por `toc_builder`. Activar una vez que shadow mode confirme paridad estructural entre ambos caminos.
