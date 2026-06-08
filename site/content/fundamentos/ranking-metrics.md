---
title: "Métricas de Ranking"
weight: 141
math: true
---

Cuando un sistema **recomienda** o **recupera** información, no entrega una etiqueta binaria sino una **lista ordenada**: los primeros resultados pesan mucho más que los últimos. Por eso las métricas clásicas de clasificación —accuracy, precision, recall— **no alcanzan**: ignoran el orden, asumen clases balanceadas y tratan por igual un acierto en la posición 1 y otro en la posición 50. Este fundamento recorre el camino completo, desde la **matriz de confusión** y los errores Tipo I / Tipo II hasta las métricas propias de ranking —**Precision@k, Recall@k, MAP, MRR y nDCG**—, reproduciendo los ejemplos numéricos exactos de la [Clase 25](/clases/clase-25) sobre sistemas de recomendación multimodal. El objetivo es que sepas **qué mide cada métrica, cuándo usarla y cómo se calcula a mano**.

---

## 1. Por qué las métricas de clasificación no bastan en recomendación

Un clasificador binario produce un sí/no por ejemplo y se evalúa con accuracy, precision o recall sobre el conjunto completo. Un recomendador (o un buscador) produce algo distinto: una **lista ordenada de candidatos** de la que el usuario solo verá los primeros. Tres propiedades hacen insuficientes las métricas planas:

- **El orden importa.** Acertar un ítem relevante en la posición 1 vale mucho más que acertarlo en la posición 20. Accuracy y F1 no distinguen entre ambos casos: solo cuentan aciertos.
- **El feedback es sparse.** De millones de ítems, un usuario interactúa con un puñado. La mayoría de las celdas de la "verdad" están vacías: no sabemos si un ítem no consumido es irrelevante o simplemente no fue visto.
- **Las clases están desbalanceadas.** Los ítems relevantes son una fracción minúscula del catálogo. Un modelo que predice "irrelevante" para todo tendría accuracy altísima y utilidad nula.

{{< concept-alert type="clave" >}}
En recomendación y recuperación de información lo que se evalúa es la **calidad del top-k**: qué tan buenos son los primeros $k$ resultados que el usuario realmente verá. Las métricas de ranking incorporan **posición** y **relevancia graduada**, dos dimensiones que las métricas de clasificación ignoran.
{{< /concept-alert >}}

Esta distinción conecta con el diseño mismo de los [sistemas de recomendación](/fundamentos/recommender-systems), donde el objetivo no es clasificar sino **ordenar bien**.

---

## 2. Matriz de confusión y errores Tipo I / Tipo II

Toda métrica de clasificación nace de la **matriz de confusión**, que cruza la predicción del modelo contra la verdad:

|                       | Relevante (real +) | No relevante (real −) |
|-----------------------|--------------------|------------------------|
| **Predicho +**        | TP (verdadero positivo) | FP (falso positivo) |
| **Predicho −**        | FN (falso negativo) | TN (verdadero negativo) |

Los dos tipos de error tienen nombres clásicos de la estadística:

- **Error Tipo I (falso positivo, FP):** el sistema recomienda algo irrelevante. Equivale a rechazar la hipótesis nula siendo verdadera. En un buscador, es "spam" en los resultados.
- **Error Tipo II (falso negativo, FN):** el sistema omite algo relevante. Equivale a no rechazar la hipótesis nula siendo falsa. En recomendación, es la película que te habría encantado y nunca te mostró.

El costo relativo de cada error depende del dominio. En un filtro de detección de fraude o en triage clínico, un FN (no detectar el caso peligroso) suele ser mucho más caro que un FP. En recomendación de contenido, el balance es más simétrico, pero los FP en el top de la lista dañan la confianza del usuario.

---

## 3. Precision, Recall y F1

Sobre la matriz de confusión se definen las tres métricas base:

$$\text{Precision} = \frac{TP}{TP + FP} \qquad \text{Recall} = \frac{TP}{TP + FN}$$

- **Precision** responde: *de lo que recomendé, ¿qué fracción era relevante?* Penaliza los FP.
- **Recall** responde: *de todo lo relevante que existía, ¿qué fracción recuperé?* Penaliza los FN.

Hay un **trade-off** inherente: subir el recall (recomendar más) suele bajar la precision (entran irrelevantes), y viceversa. El **F1-score** resume ambos con su media armónica, que castiga los valores extremos:

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

La media armónica (y no la aritmética) se usa porque un sistema con precision 1.0 y recall 0.0 debe puntuar cerca de 0, no de 0.5. La generalización $F_\beta$ permite ponderar recall ($\beta > 1$) o precision ($\beta < 1$) según el costo de cada error.

{{< concept-alert type="clave" >}}
Precision, recall y F1 son **agnósticas al orden**: tratan el conjunto recomendado como una bolsa sin posiciones. Por eso, para ranking, las adaptamos a un corte $k$ (las métricas @k de la siguiente sección).
{{< /concept-alert >}}

---

## 4. Precision@k y Recall@k

La adaptación más simple al ranking consiste en **truncar la lista en los primeros $k$ resultados** y calcular precision y recall solo sobre ese corte:

$$\text{Precision@}k = \frac{\#\{\text{relevantes en el top-}k\}}{k} \qquad \text{Recall@}k = \frac{\#\{\text{relevantes en el top-}k\}}{\#\{\text{relevantes totales}\}}$$

El ejemplo numérico exacto de la clase: un recomendador devuelve **10 ítems**, de los cuales **5 son relevantes**, y en el catálogo existen **20 ítems relevantes en total**. Entonces:

$$\text{Precision@}10 = \frac{5}{10} = 50\% \qquad \text{Recall@}10 = \frac{5}{20} = 20\%$$

La lectura: de los 10 ítems mostrados, la mitad acertó (precision alta), pero solo capturamos una quinta parte de todo lo que le habría gustado al usuario (recall bajo, porque $k=10 \ll 20$). Esta tensión es típica: con un $k$ pequeño es imposible tener recall alto si hay muchos relevantes.

El **corte $k$** se elige según la interfaz: si el usuario ve 10 resultados por página, evaluamos @10; si una app móvil muestra 3 tarjetas, evaluamos @3. Pero ni Precision@k ni Recall@k miran **dónde** dentro del top-k cayeron los aciertos: un relevante en la posición 1 cuenta igual que uno en la posición $k$. Para corregir eso aparecen MAP, MRR y nDCG.

---

## 5. Average Precision (AP) y MAP

**Average Precision** introduce la sensibilidad al orden. La idea: recorrer la lista de arriba hacia abajo y, **cada vez que aparece un ítem relevante**, calcular la Precision@k en esa posición; luego promediar esas precisiones sobre el número de relevantes.

$$\text{AP} = \frac{1}{\#\text{relevantes}} \sum_{k=1}^{n} \text{Precision@}k \cdot \mathbb{1}[\text{ítem }k\text{ es relevante}]$$

Donde $\mathbb{1}[\cdot]$ vale 1 si el ítem en la posición $k$ es relevante y 0 si no. Como solo suma en las posiciones relevantes, **premia que los relevantes estén arriba**: un relevante en la posición 1 contribuye con $1/1$, mientras que uno en la posición 10 contribuye a lo sumo con una precisión pequeña.

Ejemplo: lista de 5 ítems con relevantes en las posiciones 1 y 3.
- Posición 1 (relevante): Precision@1 $= 1/1 = 1.0$
- Posición 3 (relevante): Precision@3 $= 2/3 \approx 0.667$
- $\text{AP} = \tfrac{1}{2}(1.0 + 0.667) = 0.833$

El **MAP (Mean Average Precision)** simplemente promedia el AP sobre **todas las queries o usuarios** del conjunto de evaluación:

$$\text{MAP} = \frac{1}{Q} \sum_{q=1}^{Q} \text{AP}(q)$$

MAP es la métrica reina en recuperación de información con relevancia **binaria** (relevante / no relevante) y múltiples relevantes por query.

---

## 6. MRR (Mean Reciprocal Rank)

Cuando lo único que importa es **dónde aparece el primer acierto** —típico en búsqueda de respuesta única, QA, o "feeling lucky"— se usa el **Reciprocal Rank**: el inverso de la posición del primer ítem relevante.

$$\text{RR} = \frac{1}{\text{rank}_{\text{primer relevante}}}$$

Si el primer relevante está en la posición 1, $RR = 1$; en la posición 2, $RR = 0.5$; en la posición 5, $RR = 0.2$. El **MRR** promedia el RR sobre todas las queries:

$$\text{MRR} = \frac{1}{Q} \sum_{q=1}^{Q} \frac{1}{\text{rank}_q}$$

| Posición del 1er relevante | Reciprocal Rank |
|----------------------------|-----------------|
| 1                          | 1.000           |
| 2                          | 0.500           |
| 3                          | 0.333           |
| 4                          | 0.250           |
| 5                          | 0.200           |

{{< concept-alert type="clave" >}}
**MRR ignora todo lo que viene después del primer acierto.** Es ideal cuando existe esencialmente una respuesta correcta (autocompletar, navegación a un sitio), pero inadecuado cuando hay muchos relevantes y queremos medir la calidad de toda la lista (ahí preferimos MAP o nDCG).
{{< /concept-alert >}}

---

## 7. DCG, iDCG y nDCG

MAP y MRR asumen relevancia **binaria**. Pero muchas veces la relevancia es **graduada**: una película puede ser "imperdible" (rel=3), "buena" (rel=2), "pasable" (rel=1) o "irrelevante" (rel=0). El **DCG (Discounted Cumulative Gain)** captura esto con dos ideas: acumular la **ganancia** (relevancia) de cada ítem y **descontarla logarítmicamente** según su posición, de modo que un acierto profundo aporta menos.

$$\text{DCG@}k = \sum_{i=1}^{k} \frac{rel_i}{\log_2(i+1)}$$

El **descuento logarítmico** modela que la atención del usuario decae suavemente: el ítem 2 vale $1/\log_2(3)$, el ítem 3 vale $1/\log_2(4)$, etc. El problema es que el DCG crudo no es comparable entre queries (depende de cuántos relevantes haya). Por eso se **normaliza** dividiendo por el **iDCG (ideal DCG)**: el DCG que se obtendría con el **ordenamiento perfecto** (todos los relevantes arriba, ordenados por relevancia decreciente):

$$\text{nDCG@}k = \frac{\text{DCG@}k}{\text{iDCG@}k} \in [0, 1]$$

**Ejemplo exacto de la clase.** Lista de 5 ítems con **relevancia binaria**, donde los relevantes están en las posiciones **1 y 3**. El DCG suma la contribución de cada posición relevante (usando la convención de la clase $1/\log_2(\text{pos}+1)$ para las posiciones que aportan), y el iDCG corresponde al caso ideal en que todas las posiciones aportan ganancia:

$$\text{DCG} = \frac{1}{\log_2 3} + \frac{1}{\log_2 5} + \frac{1}{\log_2 6} = 0.6309 + 0.4307 + 0.3869 = 1.4485$$

$$\text{iDCG} = \frac{1}{\log_2 2} + \frac{1}{\log_2 3} + \frac{1}{\log_2 4} + \frac{1}{\log_2 5} + \frac{1}{\log_2 6} = 1.0 + 0.6309 + 0.5 + 0.4307 + 0.3869 = 2.9485$$

$$\text{nDCG} = \frac{\text{DCG}}{\text{iDCG}} = \frac{1.4485}{2.9485} = 0.4912$$

El resultado **0.4912** dice que este ranking logra cerca de la mitad de la ganancia del ranking ideal: hay aciertos, pero mal posicionados. El nDCG es hoy la métrica estándar de facto en ranking con relevancia graduada y fue formalizado por Järvelin y Kekäläinen en [nDCG (Järvelin 2002)](/papers/ndcg-jarvelin-2002).

{{< concept-alert type="clave" >}}
**nDCG es la única de estas métricas que aprovecha relevancia graduada y normaliza por el ideal.** Por eso es comparable entre queries con distinto número de relevantes y se ha vuelto el reporte obligatorio en *learning-to-rank* y motores de búsqueda.
{{< /concept-alert >}}

---

## 8. Métricas online vs offline

Todas las anteriores son métricas **offline**: se calculan sobre un conjunto de evaluación con relevancia conocida, sin usuarios reales. Son rápidas, reproducibles y baratas, pero miden un **proxy** del comportamiento real, sujeto al sesgo del feedback histórico (solo conocemos relevancia de lo que el sistema anterior ya mostró).

Las métricas **online** miden el comportamiento de usuarios reales en producción:

- **CTR (Click-Through Rate):** fracción de impresiones que reciben clic. Proxy inmediato de relevancia percibida.
- **Conversion rate, dwell time, retención, revenue:** métricas de negocio más cercanas al valor real.
- **A/B testing:** se compara una variante del modelo (B) contra la actual (A) repartiendo tráfico aleatoriamente y midiendo la diferencia en la métrica de negocio con significancia estadística.

| Aspecto         | Offline (MAP, nDCG, MRR) | Online (CTR, A/B test) |
|-----------------|--------------------------|-------------------------|
| Costo           | Bajo                     | Alto (tráfico real)     |
| Velocidad       | Segundos                 | Días/semanas            |
| Riesgo          | Nulo                     | Afecta usuarios reales  |
| Sesgo           | Feedback histórico       | Más cercano al negocio  |
| Uso típico      | Iteración de modelos     | Decisión de despliegue  |

La práctica estándar: iterar rápido con métricas offline para filtrar candidatos, y validar las apuestas finales con **A/B testing**, que es lo único que mide el impacto real en el negocio.

---

## 9. Cómo elegir la métrica correcta

No existe una métrica universal; la elección depende de la **tarea** y de la **naturaleza del feedback**:

| Situación                                          | Métrica recomendada |
|----------------------------------------------------|---------------------|
| Una sola respuesta correcta (QA, autocompletar)    | **MRR**             |
| Varios relevantes, relevancia binaria              | **MAP**             |
| Relevancia graduada (ratings, niveles)             | **nDCG**            |
| Solo importa el top-k visible (interfaz fija)      | **Precision@k / Recall@k** |
| Catálogo enorme, recuperar el máximo de relevantes | **Recall@k**        |
| Validar impacto real en producción                 | **A/B test (CTR, conversión)** |

Reglas prácticas:

- Si el usuario **busca una cosa**, optimiza MRR. Si **explora muchas**, MAP o nDCG.
- Si tienes **señales de relevancia graduada** (estrellas, tiempo de visionado), úsalas: nDCG las aprovecha y MAP las desperdicia.
- El **corte $k$** debe coincidir con lo que el usuario realmente ve, no con un número arbitrario.
- Ninguna métrica offline reemplaza el **A/B test**: son brújulas para iterar, no el veredicto final.

---

## Para profundizar

- [Clase 25 — Sistemas de recomendación multimodal](/clases/clase-25): la clase donde se presentan estas métricas con los ejemplos numéricos reproducidos aquí.
- [Fundamento: Sistemas de recomendación](/fundamentos/recommender-systems): cómo se construyen los recomendadores cuya calidad miden estas métricas.
- [nDCG (Järvelin 2002)](/papers/ndcg-jarvelin-2002): el paper seminal que formaliza el discounted cumulative gain y su normalización.
- [Fundamento: ROUGE](/fundamentos/rouge-metric) y [BLEU](/fundamentos/bleu-metric): métricas de evaluación para generación de texto, complementarias a las de ranking.
