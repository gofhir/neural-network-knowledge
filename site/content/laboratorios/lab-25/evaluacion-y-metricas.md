---
title: "Evaluación: nDCG, precision/recall y dos bugs de métrica"
weight: 5
math: true
---

> **Celdas 70-80 del notebook.** El clímax del laboratorio. Definir las métricas de recomendación, encontrar **dos bugs reales** al implementarlas, y descubrir que el sistema siempre funcionó bien: el problema nunca fue el recomendador, sino cómo lo estábamos midiendo. Cierra con la Actividad 3.

## Las métricas (celda 70)

Para evaluar un sistema de recomendación con relevancia binaria usamos tres métricas. Un ítem se considera **relevante (1)** si pertenece al mismo usuario que la consulta, y **0** en caso contrario. Como el test tiene 1000 ítems repartidos entre 10 usuarios, hay exactamente **100 ítems relevantes por usuario**.

- **Precision@k** — de lo que recomendé (los $k$ primeros), ¿qué fracción es relevante?

$$P@k = \frac{|\text{relevantes} \cap \text{recomendados}_k|}{k}$$

- **Recall@k** — de todo lo relevante que existe, ¿qué fracción logré recomendar dentro del top-$k$?

$$R@k = \frac{|\text{relevantes} \cap \text{recomendados}_k|}{|\text{relevantes}|}$$

- **nDCG@k** — a diferencia de las dos anteriores, es **sensible al ORDEN**: premia que los ítems relevantes aparezcan **primero** en la lista. Se calcula como el DCG (Discounted Cumulative Gain) normalizado por el IDCG (el DCG del ranking ideal):

$$\text{nDCG}@k = \frac{\text{DCG}@k}{\text{IDCG}@k}, \qquad \text{DCG}@k = \sum_{i=1}^{k} \frac{rel_i}{\log_2(i+1)}$$

El descuento logarítmico $\log_2(i+1)$ hace que un acierto en la posición 1 valga mucho más que uno en la posición 50. El IDCG es el DCG que se obtendría si todos los relevantes estuvieran perfectamente ordenados al frente; dividir por él deja el nDCG entre 0 y 1.

Profundización en [/fundamentos/ranking-metrics](/fundamentos/ranking-metrics) y en el paper que introdujo el nDCG, [Järvelin & Kekäläinen 2002](/papers/ndcg-jarvelin-2002).

## Bug 1 — nDCG recibe distancias en vez de similitudes

El recomendador ordena ítems por **distancia coseno**: menor distancia = más parecido = mejor candidato. Pero `sklearn.metrics.ndcg_score(y_true, y_score)` asume que `y_score` es una **puntuación de relevancia donde mayor = mejor**: rankea poniendo primero los valores de `y_score` MÁS ALTOS.

```python
# BUG: scores son distancias coseno (menor = mejor)
ndcg_score(y_true, scores)   # nDCG mide ordenar de PEOR a MEJOR → invertido
```

Al pasarle distancias directamente, el ranking que evalúa el nDCG queda **completamente invertido**: la métrica premia poner primero los ítems más lejanos (los peores). Estábamos midiendo, literalmente, qué tan bien el sistema ordenaba de peor a mejor.

```python
# FIX: negar las distancias para convertirlas en "similitud" (mayor = mejor)
ndcg_score(y_true, -scores)
```

## Bug 2 — pr_at_k no promedia sobre los usuarios

La función `pr_at_k` recorre los 10 usuarios, acumula la precision y el recall de cada uno en listas, pero al retornar no usa esas listas:

```python
def pr_at_k(...):
    precision, recall = [], []
    for user in users:
        ...
        p = ...   # escalar del usuario actual
        r = ...
        precision.append(p)
        recall.append(r)
    return np.mean(p), np.mean(r)   # BUG: p y r son los del ÚLTIMO usuario
```

`np.mean(p)` y `np.mean(r)` operan sobre `p` y `r`, que tras el bucle son los **escalares del último usuario** (el usuario 9), no las listas acumuladas. `np.mean` de un escalar devuelve ese mismo escalar. Resultado: la función reporta solo al usuario 9, no el promedio de los 10.

```python
# FIX: promediar las listas acumuladas
return np.mean(precision), np.mean(recall)
```

## Impacto medido (antes vs. después, en k=400)

Lo notable es que **ambos bugs eran de la métrica, no del recomendador**. Al corregirlos, los números dan un salto enorme con el mismo sistema y la misma data:

| Métrica | Antes (con bug) | **Después (corregido)** | Factor |
|---|---|---|---|
| nDCG@400 | 0.022 | **0.857** | **~40×** |
| Precision@400 | 0.1725 *(1 usuario)* | **0.227** *(promedio 10)* | — |
| Recall@400 | 0.69 | **0.908** | — |

El nDCG pasó de **0.022** (parecía un sistema pésimo) a **0.857** (ranking casi ideal): un factor de **~40×** producido únicamente por negar las distancias. El sistema **siempre funcionó bien** — lo que estaba roto era la regla de medición.

## Interpretación de los números corregidos

- **Precision@400 = 0.227** → es **2.3× sobre el azar**. El baseline aleatorio es $100/1000 = 0.10$ (probabilidad de que un ítem cualquiera sea relevante). Recomendar acertando 22.7% de las veces más que duplica esa línea base.
- **Recall@400 = 0.908** → en el top-400 capturamos el **91% de los 100 ítems relevantes** del usuario.
- **nDCG@400 = 0.857** → el ranking está **cerca del ideal**: los relevantes no solo aparecen, sino que aparecen temprano.

La curva precision-recall corregida tiene la **forma clásica**: con $k$ chico la precision llega a **~0.73** (lo poco que recomendamos es muy preciso) y va cayendo a medida que aumentamos $k$ y subimos el recall. Es la típica curva de compromiso precision-recall.

## El valle del nDCG en k=100 (detalle fino)

La curva de nDCG vs. $k$ tiene un **mínimo local exactamente en $k=100$**, que no es casualidad: **100 es el número de ítems relevantes por usuario**.

- En $k=100$ estamos en el punto **más exigente**: el ideal sería tener los 100 relevantes perfectamente ordenados en las primeras 100 posiciones. Cualquier ítem irrelevante que se cuele entre los primeros 100 castiga fuerte el nDCG, porque el IDCG en ese punto asume 100 aciertos perfectos.
- Para $k > 100$ el **IDCG se satura**: solo existen 100 relevantes, así que el ranking ideal no puede mejorar más, mientras el DCG del modelo sigue recogiendo aciertos rezagados. El cociente se **recupera y sube hacia ~0.90**.

El valle en $k=100$ es, por tanto, una firma estructural del dataset (10 usuarios, 100 ítems c/u), no un defecto del modelo.

## La lección

> La misma data, el mismo sistema, dos implementaciones de métrica → **nDCG 0.02 vs. 0.86**. Una métrica mal implementada puede hacer que un sistema perfectamente bueno parezca un desastre. **Medir bien es tan importante como modelar bien.** Antes de tirar un modelo a la basura por sus números, hay que auditar la métrica que produjo esos números.

## Actividad 3 (resuelta): ¿qué más medir además del rendimiento?

Optimizar solo precision/recall/nDCG deja fuera dimensiones que importan para un buen recomendador en producción:

- **Diversidad y novedad** — un sistema que solo recomienda lo más parecido cae en *filter bubbles*: el usuario ve siempre lo mismo. Conviene medir qué tan variadas son las recomendaciones.
- **Cobertura de catálogo (long tail)** — los sistemas suelen sufrir *popularity bias* y recomendar siempre los mismos pocos ítems populares, dejando el resto del catálogo sin exposición.
- **Validez de la propia métrica** — los dos bugs de este lab son el ejemplo perfecto: hay que auditar que la métrica mida lo que creemos que mide.

**Dos métricas concretas distintas a las de clase**, aprovechando que cada ítem tiene un descriptor de 32 dimensiones:

- **Intra-list diversity** — disimilitud promedio (p. ej. 1 − similitud coseno) entre los ítems recomendados de una misma lista, medible directamente con los **descriptores de 32-d**. Mayor disimilitud = lista más diversa.
- **Catalog coverage** — fracción del catálogo total que el sistema llega a recomendar a lo largo de todos los usuarios. Detecta el sesgo de popularidad.

Más contexto en [/fundamentos/ranking-metrics](/fundamentos/ranking-metrics) y [/fundamentos/recommender-systems](/fundamentos/recommender-systems).

---

**Anterior:** [Recomendación por similitud](recomendacion) · **Volver al** [índice del lab](../)
