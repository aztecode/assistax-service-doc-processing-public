# Auditoría de tablas y layout (corpus PDF)

> PDFs encontrados bajo `--root`: **230** (cambia la ruta o une carpetas si el corpus objetivo es mayor, p. ej. ~500).

- **Generado (UTC):** 2026-03-25T05:30:56.198985+00:00
- **Raíz escaneada:** `/Users/rodrigojacome/Documents/Proyectos/rbv/procesar-archivos`
- **PDFs analizados:** 230 ok / 0 fallidos (total listados: 230)
- **Tiempo total:** 917.6 s

## Totales

| Métrica | Valor |
|--------|-------|
| Páginas | 20958 |
| Tablas detectadas (raw `find_tables`) | 3576 |
| Tablas conservadas (tras filtro prosa) | 3155 |
| Tablas descartadas (filtro prosa) | 421 |
| Páginas con ≥1 trazo vectorial (`get_drawings`) | 15853 |
| Páginas con trazos y **0** tablas raw | 12962 |
| Páginas con ≥1 rect. “editorial” (filtro estricto) | 0 |
| Páginas rect. estricto y **0** tablas raw | 0 |
| Páginas con ≥1 tabla descartada | 419 |
| Bloques elegibles arbiter (tabla en marco, ≤2 cols, no nota editorial) | 0 |
| De ellos, **ambigüos** deterministas → **máx. llamadas LLM** arbiter | 0 |

## Proyección a ~500 PDFs (lineal)

Si el resto del corpus se parece a esta muestra, multiplica por **500/230 ≈ 2,17**: del orden de **45k páginas**, **~7800 tablas raw**, **~910 tablas descartadas** por filtro de prosa. El tiempo de auditoría escala de forma similar (~33 min por 500 PDFs en hardware comparable).

## Dimensionamiento LLM (orientativo)

1. **Arbiter de notas enmarcadas** (`ENABLE_LLM_BOXED_NOTE_ARBITER`): como máximo **una llamada por bloque ambiguo** listado arriba (solo si el flag está activo). Los bloques estructurales o editoriales claros no llaman al modelo.
2. **Recuperación de tablas / layout**: **páginas con `get_drawings` no vacío y 0 tablas raw** es un *techo muy alto*: casi toda página con maquetación formal tiene trazos vectoriales (reglas, marcos decorativos). **No** interpretes ese conteo como “llamadas LLM necesarias = páginas” sin filtrar. Úsalo como señal de que `find_tables` deja muchas páginas solo con texto + gráficos; refinad con umbrales de paths, palabras clave (“Fe de erratas”, “TABLA”), o `pymupdf_layout`. El filtro **rectángulo editorial** del `pdf_extractor` dio **0** en este corpus: muchos marcos no cumplen `color`/forma esperada.
3. **Descartes por prosa**: el total de tablas descartadas indica dónde el pipeline **rechaza** candidatos de `find_tables`; no implica automáticamente LLM, pero sirve para priorizar revisión manual o reglas adicionales.

## Artefactos

- `summary.json` — resumen JSON
- `by_pdf.csv` — una fila por PDF
- `failures.txt` — errores de apertura/lectura
