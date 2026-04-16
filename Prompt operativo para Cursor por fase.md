# Prompt operativo para Cursor por fase

## Prompt base
Usar esto antes de cada fase:

> Implementa únicamente la fase indicada del plan. No adelantes fases futuras. Mantén compatibilidad con el pipeline actual. Agrega tests unitarios o de integración según corresponda. Explica brevemente decisiones no obvias en comentarios de código. Al final, entrega resumen de archivos creados/modificados, riesgos y cómo validar localmente.

## Prompt ejemplo para Fase 1

> Implementa la Fase 1 del plan detallado. Crea `pipeline/layout_models.py` y `pipeline/layout_extractor_v2.py`. No modifiques todavía el flujo productivo del runner. Agrega `tests/unit/test_layout_extractor_v2.py`. Usa PyMuPDF con extracción por spans/bbox. Devuelve `DocumentLayout`. Al final, muestra diff resumido y pasos para ejecutar tests.

## Prompt ejemplo para Fase 3

> Implementa la Fase 3 del plan detallado. Crea `block_rules_v2.py`, `block_classifier_llm_v2.py` y `block_classifier_v2.py`. Usa heurística primero y LLM solo en ambiguos. Agrega flags nuevos en `settings.py`. No conectes todavía al runner principal si no es necesario. Añade tests que cubran fechas mal clasificadas, referencias a artículos dentro de prosa y headers DOF.

---

# Checklist de aceptación final

El proyecto puede considerarse exitoso cuando:
- ya no aparecen headings visibles como `TABLE`;
- las fechas DOF no salen como títulos;
- los encabezados/footers DOF no contaminan chunks;
- las tablas se insertan donde van;
- el índice no se alimenta del índice inicial del documento;
- los transitorios quedan estructurados correctamente;
- el árbol legal es la fuente de verdad para `toc` y `sections`;
- el pipeline puede operar en shadow mode y comparar con legacy;
- existe quality gate automático antes de indexar.

---

# Orden recomendado de PRs

1. PR1 — modelos y extractor v2
2. PR2 — normalizador v2
3. PR3 — reglas + clasificador v2
4. PR4 — builder de estructura
5. PR5 — quality validator
6. PR6 — projector a chunks
7. PR7 — integración runner + shadow mode
8. PR8 — scripts operativos + documentación
9. PR9 — promoción a staging

---

# Nota final para Cursor

No intentes “arreglar” el pipeline actual con más regex como estrategia principal.
La meta es que las regex ayuden, pero que la **fuente de verdad** sea una reconstrucción estructural trazable y validable.

