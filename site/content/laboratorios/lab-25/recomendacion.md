---
title: "Recomendación por similitud de descriptores"
weight: 4
math: true
---

> **Celdas 53-69 del notebook.** El modelo que entrenamos para *clasificar usuarios* nunca fue el verdadero objetivo: era un **proxy task**. Aquí cobramos el premio — reciclamos esa red como **extractor de descriptores** y construimos un recomendador por vecino más cercano, sin entrenar nada nuevo.

## El pago del proxy task

Entrenamos un clasificador, pero lo que de verdad queríamos era un **espacio de representación** donde items parecidos queden cerca. La clasificación de usuarios fue solo la excusa (el *proxy*) para forzar a la red a aprender ese espacio. Ahora cambiamos el flag `features=True` y, en vez de devolver las 10 probabilidades de usuario, el modelo entrega el **descriptor intermedio de 32 dimensiones** que vive justo antes del clasificador final.

La premisa de la recomendación es directa, y el notebook la enuncia así:

> Usuarios prefieren contenido similar al que ya han interactuado antes.

Si eso es cierto, recomendar se reduce a un problema geométrico: **buscar los vecinos más cercanos** en el espacio de descriptores. Esta idea — que una red entrenada para una tarea de clasificación produce embeddings reutilizables por distancia — es el corazón del [metric learning](/fundamentos/triplet-loss) y la base conceptual de las arquitecturas [two-tower](/fundamentos/two-tower-retrieval) que vimos en la clase.

## Extracción de descriptores (celdas 56-57)

Se recorre cada split y se pasa cada muestra por `model(..., features=True)`, guardando el vector de 32-d:

```python
test_f  = np.zeros((len(testset), 32),  dtype=np.float32)   # 1000 × 32
train_f = np.zeros((len(trainset), 32), dtype=np.float16)   # 4000 × 32
```

| Matriz | Forma | dtype | Por qué |
|---|---|---|---|
| `test_f` | 1000 × 32 | `float32` | Items nuevos a recomendar |
| `train_f` | 4000 × 32 | `float16` | Items ya vistos por los usuarios; **mitad de memoria** |

`train_f` usa `float16` (`.half()`) porque es 4× más grande que test y la precisión exacta no importa para distancias relativas — un truco de ahorro de memoria perfectamente válido aquí.

> **Gotcha real — el `device` que nunca usa GPU.** La línea de configuración es:
>
> ```python
> device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
> ```
>
> Las **dos ramas dicen `"cpu"`**. Sin importar si hay GPU disponible, siempre corre en CPU. Es un typo: la primera rama debería ser `"cuda"`. La consecuencia es concreta — como el forward incluye un **BERT** completo para el texto, extraer los 5000 descriptores en CPU es **lento**. Fix:
>
> ```python
> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
> ```

## Encontrar similares (celdas 59-61)

`find_similar_images` calcula la distancia de una *query* contra **todos** los demás descriptores y se queda con el top-k usando un *heap*:

```python
distances = pairwise_distances(embedding[query_id].reshape(1,-1), embedding, metric=metric)
# heapq mantiene los topk más cercanos sin ordenar todo
```

Se prueban dos métricas:

| Métrica | Qué mide | Cuándo |
|---|---|---|
| `cosine` | Ángulo entre vectores (ignora la norma) | **La usual para embeddings** — captura "dirección semántica" |
| `euclidean` | Distancia recta en el espacio | Sensible a la magnitud del vector |

> **Gotcha de escalabilidad.** `pairwise_distances(query, todos)` es **O(N) por consulta**: compara contra cada item del catálogo. Con 4000 items va bien, pero **no escala a millones**. Ese es exactamente el problema que resuelve el *retrieval* aproximado (MIPS / ANN) detrás del [two-tower](/fundamentos/two-tower-retrieval): se precomputan los embeddings de items y se indexan para que cada consulta sea sublineal. Aquí hacemos la versión "de fuerza bruta" que es didáctica pero no productiva.

## La recomendación (celdas 63-66)

El salto de "items similares" a "recomendar a usuarios" pasa por agrupar los descriptores de train **por usuario**:

```python
user_repr = np.array([user_dict[user_id] for user_id in range(n_targets)])
# forma: (10, 400, 32)  →  10 usuarios, 400 items cada uno, 32-d
```

> **Nota — esto solo funciona porque el dataset está balanceado.** Cada usuario tiene exactamente **400 items**, así que `np.array(...)` produce un tensor rectangular limpio `(10, 400, 32)`. Un dataset real con conteos distintos por usuario **rompería** ese `np.array` (filas de largo desigual); habría que recurrir a listas o *padding*.

Luego se calculan los scores usuario × item-de-test:

```python
user_repr = user_repr.reshape(-1, 32)          # (4000, 32)
dists = pairwise_distances(user_repr, test_f, metric='cosine')
dists = dists.reshape(n_users, n_images, -1)   # (10, 400, 1000)
scores = dists.min(axis=1)                      # (10, 1000)
```

El paso clave es `.min(axis=1)`: para cada usuario y cada item de test, se queda con la distancia al item **más cercano** de ese usuario.

**¿Por qué `min` (nearest-neighbor) y no `mean` (centroide)?** Un item es una buena recomendación si se parece a **algo** que al usuario le gustó, no al "promedio" de todo lo que le gustó. Si un usuario tiene gustos diversos (fotos de comida *y* de montañas), su centroide cae en un punto intermedio que no representa ni una cosa ni la otra, y penaliza recomendaciones que encajan perfecto con una de sus facetas. El `min` **captura gustos múltiples**: basta con acercarse a una de sus preferencias.

## Top-k con `argpartition` (celdas 67-69)

```python
k = 10
recommendation_list = np.argpartition(scores, k)[:, :k]
```

`np.argpartition` reordena para dejar los `k` menores (las distancias más chicas = items más relevantes) en las primeras posiciones, **sin ordenar todo el arreglo**. Es más rápido que `np.argsort`: O(N) en vez de O(N log N), y aquí solo necesitamos *quiénes* son los top-k, no toda la clasificación.

> **Gotcha — `argpartition` no ordena dentro del top-k.** Los `k` items quedan seleccionados pero **en orden arbitrario** entre sí. Para una lista de recomendación esto importa: el mejor item podría aparecer en la posición 7 y el séptimo mejor en la 1. Si más adelante se evalúa con **nDCG** — que premia poner lo más relevante arriba (ver [métricas de ranking](/fundamentos/ranking-metrics)) — habría que hacer un `argsort` final sobre el top-k para ordenarlo de mejor a peor.

La celda 69 imprime el **texto** de los items recomendados a un usuario. No es una métrica formal: es una verificación cualitativa de **coherencia temática** — ¿las recomendaciones se parecen, en tema, a lo que el usuario ya comentó? Es el "test de olfato" antes de pasar a la evaluación cuantitativa.

---

**Anterior:** [Entrenamiento: multimodal vs baseline](entrenamiento) · **Siguiente:** [Evaluación y métricas](evaluacion-y-metricas)
