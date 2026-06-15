## Nota: correcciones a las funciones de evaluación

Al revisar las funciones de métricas se detectaron **dos errores** que distorsionaban la evaluación del recomendador. Se corrigieron y se re-calcularon las métricas.

### Bug 1 — `ndcg` recibía distancias en vez de similitudes

`scores` contiene **distancias** (coseno), donde *menor = más relevante*. Pero `ndcg_score(y_true, y_score)` ordena poniendo **primero los `y_score` más altos** (asume *mayor = mejor*). Al pasarle las distancias directamente, el ranking quedaba **invertido**: la métrica medía qué tan bien se ordenaba de peor a mejor.

**Corrección:** pasar `-scores` para convertir la distancia en un puntaje donde mayor = mejor.

`return ndcg_score(relevance, -scores, k=k)`   *(antes: `ndcg_score(relevance, scores, k=k)`)*

### Bug 2 — `pr_at_k` no promediaba sobre los usuarios

La función acumulaba precision y recall por usuario en las listas `precision` y `recall`, pero retornaba `np.mean(p), np.mean(r)`, donde `p` y `r` son **escalares** del último usuario del loop. El resultado era la precision/recall de **un solo usuario** (el último), no el promedio.

**Corrección:** promediar sobre las listas completas.

`return np.mean(precision), np.mean(recall)`   *(antes: `np.mean(p), np.mean(r)`)*

### Impacto de las correcciones

| Métrica (k=400) | Antes (con bugs) | Después (corregido) |
|---|---|---|
| nDCG | 0.022 | **0.857** |
| Precision | 0.1725 (1 usuario) | **0.227** (promedio 10) |
| Recall | 0.69 (1 usuario) | **0.908** (promedio 10) |

El nDCG pasó de 0.022 (prácticamente el peor caso) a 0.857, confirmando que el sistema **siempre funcionó bien**: el problema estaba en la implementación de la métrica, no en el recomendador. Esto ilustra que una métrica mal implementada puede hacer ver un sistema correcto como defectuoso — medir bien es tan importante como modelar bien.

### Observación adicional

En la curva de nDCG vs k aparece un mínimo local exactamente en **k=100**, que coincide con el número de ítems relevantes por usuario (100 de 1000). Es el punto más exigente para la métrica: el ideal sería tener los 100 relevantes perfectamente ordenados en las primeras 100 posiciones. Para k>100 el IDCG se satura (solo existen 100 relevantes) y el nDCG se recupera y sube.
