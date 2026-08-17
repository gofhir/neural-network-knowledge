---
title: "HOTA: Higher Order Tracking Accuracy (2020)"
weight: 451
math: true
---

{{< paper-card
    title="HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking"
    authors="Jonathon Luiten, Aljoša Ošep, Patrick Dendorfer, Philip Torr, Andreas Geiger, Laura Leal-Taixé, Bastian Leibe"
    year="2020"
    venue="International Journal of Computer Vision (IJCV) / arXiv:2009.07736"
    arxiv="2009.07736"
    pdf="/papers/hota-luiten-2020.pdf" >}}
El paper que reorganizó cómo se evalúa el seguimiento multi-objeto, con un diagnóstico simple: **MOTA e IDF1 no son métricas rivales sino dos proyecciones sesgadas de un espacio de dos dimensiones** —detección y asociación—, y ninguna de las dos mide localización. HOTA mide los tres ejes explícitamente y los combina en una **media geométrica** que castiga el desbalance, además de descomponerse en una familia de sub-métricas que permiten diagnosticar qué falla. Un estudio con usuarios muestra que su ranking se alinea mejor con el juicio visual humano que el de las métricas previas. Es hoy la métrica principal de MOTChallenge y KITTI.
{{< /paper-card >}}

---

## El diagnóstico

El paper abre con un ejemplo de tres trackers construidos sobre el mismo *ground truth*, con detección creciente y asociación decreciente:

| Tracker | DetA | AssA | **MOTA** | **IDF1** | **HOTA** |
|---|---|---|---|---|---|
| A | 50 % | 50 % | 50 % | **67 %** | 50 % |
| B | 70 % | 35 % | 69 % | 52 % | 50 % |
| C | 100 % | 25 % | **97 %** | 25 % | 50 % |

MOTA los ordena C > B > A. IDF1 los ordena exactamente al revés, A > B > C. HOTA dice que están empatados, porque su producto DetA·AssA es el mismo en los tres.

El punto no es que MOTA o IDF1 estén mal calculadas. Es que **cada una colapsa un espacio bidimensional sobre un eje distinto**, y presentar cualquiera de las dos como "el" resultado es una decisión editorial disfrazada de medición.

El sesgo de MOTA tiene una explicación aritmética directa: sus tres términos —FN, FP e IDSW— se suman con el mismo peso, pero en un benchmark típico hay decenas de miles de los dos primeros y **cientos** del tercero. Los errores de asociación aportan menos del 2 % del numerador.

## La construcción

La novedad técnica son los **conjuntos de asociación**, definidos por cada verdadero positivo $c$:

- **TPA(c)**: los TP que comparten con $c$ tanto el ID verdadero como el ID predicho.
- **FNA(c)**: los que comparten el ID verdadero pero llevan otro ID predicho (o se perdieron).
- **FPA(c)**: los que llevan el mismo ID predicho pero corresponden a otro objeto (o a ninguno).

Con ellos se define una puntuación de asociación por detección y se agrega:

$$A(c) = \frac{|\mathrm{TPA}(c)|}{|\mathrm{TPA}(c)| + |\mathrm{FNA}(c)| + |\mathrm{FPA}(c)|}$$

$$\mathrm{HOTA}_\alpha = \sqrt{\frac{\sum_{c \in \mathrm{TP}} A(c)}{|\mathrm{TP}|+|\mathrm{FN}|+|\mathrm{FP}|}} \;=\; \sqrt{\mathrm{DetA}_\alpha \cdot \mathrm{AssA}_\alpha}$$

Los autores la llaman **doble Jaccard**: un índice de Jaccard sobre los conjuntos de detección, donde cada TP del numerador entra pesado por otro índice de Jaccard sobre los conjuntos de asociación.

Y la métrica final integra sobre el umbral de localización, lo que introduce el tercer eje:

$$\mathrm{HOTA} = \int_0^1 \mathrm{HOTA}_\alpha \, d\alpha \;\approx\; \frac{1}{19}\sum_{\alpha \in \{0{,}05,\; 0{,}10,\; \dots,\; 0{,}95\}} \mathrm{HOTA}_\alpha$$

{{< concept-alert type="clave" >}}
**Por qué media geométrica y no aritmética.** $\sqrt{\mathrm{DetA}\cdot\mathrm{AssA}}$ penaliza el desbalance: un sistema con DetA = 1,00 y AssA = 0,25 obtiene 0,50, lo mismo que uno con 0,50 y 0,50. Con media aritmética el primero habría sacado 0,625 y el segundo 0,50, reintroduciendo por la puerta trasera el sesgo que se quería eliminar.
{{< /concept-alert >}}

## La descomposición

HOTA se abre en un árbol de sub-métricas que separan los cinco tipos básicos de error:

$$\mathrm{HOTA} \;\to\; \begin{cases} \mathrm{LocA} & \text{localización} \\ \mathrm{DetA} \to \{\mathrm{DetRe},\, \mathrm{DetPr}\} & \text{recall y precisión de detección} \\ \mathrm{AssA} \to \{\mathrm{AssRe},\, \mathrm{AssPr}\} & \text{recall y precisión de asociación} \end{cases}$$

con una interpretación operativa clara: los errores de **recall de asociación** (medidos por FNA) son **fragmentaciones** —una trayectoria verdadera partida en varias predichas—; los de **precisión de asociación** (FPA) son **fusiones** —una trayectoria predicha que abarca dos objetos distintos—. Son fallas cualitativamente distintas y para ciertas aplicaciones una es tolerable y la otra no.

Esto resuelve, según los autores, la vieja discusión sobre si conviene una métrica única o varias: HOTA da **ambas cosas** — un escalar para rankear y una descomposición para entender.

## Validación

Además del análisis teórico (monotonía respecto de los cinco tipos de error, y las propiedades de simetría y subaditividad que hacen de HOTA la única de las tres que es literalmente una métrica en el sentido matemático), el paper hace un **estudio con usuarios**: se muestran pares de salidas de trackers y se pregunta cuál es mejor. El ranking de HOTA se alinea con el juicio humano más que el de MOTA o IDF1.

## Por qué importa para la Clase 42

La [Clase 42](/clases/clase-42) no discute métricas, y esa omisión es justamente lo que HOTA vuelve visible. La afirmación de la clase de que *"en la práctica DeepSORT es más robusto que SORT para mantener identidades"* es correcta, pero solo se puede sostener mirando los ID switches — no MOTA, que apenas se mueve.

Con la descomposición de HOTA la comparación se vuelve legible: [DeepSORT](/papers/deepsort-wojke-2017) mejora **AssA** y empeora ligeramente **DetPr** (por sus falsos positivos adicionales). MOTA promedia ambas cosas y no muestra nada; DetA y AssA muestran el intercambio exacto. Ver el fundamento [Métricas de Tracking](/fundamentos/metricas-de-tracking) y la [práctica](/clases/clase-42/practica) de la clase, donde el ejemplo de los tres trackers se reproduce numéricamente.

---

**Ver también:** [MOT16 (2016)](/papers/mot16-milan-2016) · [IDF1 (2016)](/papers/idf1-ristani-2016) · [Métricas de Tracking](/fundamentos/metricas-de-tracking) · [Seguimiento de Objetos](/fundamentos/seguimiento-de-objetos)
