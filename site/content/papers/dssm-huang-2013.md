---
title: "DSSM: Learning Deep Structured Semantic Models for Web Search"
weight: 258
math: true
---

{{< paper-card
    title="Learning Deep Structured Semantic Models for Web Search using Clickthrough Data"
    authors="Huang, He, Gao, Deng, Acero, Heck"
    year="2013"
    venue="CIKM 2013"
    pdf="/papers/dssm-huang-2013.pdf" >}}
DSSM es el **ancestro directo del two-tower / dual-encoder**. Propone dos redes neuronales profundas que proyectan **consulta y documento a un espacio semántico común** de baja dimensión, midiendo la relevancia como **similaridad coseno**. Se entrena de forma **discriminativa sobre clickthrough data** (logs de clics), maximizando la verosimilitud del documento clicado vía softmax. Introduce el **word hashing por n-gramas de letras** para escalar a vocabularios de cientos de miles de palabras. Es la semilla conceptual de la arquitectura del case study de la [Clase 25](/clases/clase-25).
{{< /paper-card >}}

---

## Contexto

A principios de los 2010, la búsqueda web recuperaba documentos por **coincidencia de palabras clave**. El problema: un mismo concepto se expresa con vocabularios distintos en consulta y documento (la *discrepancia lingüística*). Alguien busca "ccra" y el documento relevante se titula "canada revenue agency website": la coincidencia léxica falla aunque la relación semántica sea clara.

Los **modelos semánticos latentes** atacaban esto. El clásico es **LSA** (Deerwester et al., 1990): vía SVD de la matriz documento-término, mapea documento y consulta a vectores concepto de baja dimensión y compara por coseno. Sus extensiones probabilísticas **PLSA** (Hofmann, 1999) y **LDA** (Blei et al., 2003) se entrenan de forma **no supervisada**, con un objetivo solo débilmente acoplado a la métrica de retrieval, por lo que su rendimiento real quedaba corto.

Dos líneas previas prepararon el terreno. Primero, el uso de **clickthrough data**: si un usuario clica un documento tras una consulta, ese documento es al menos parcialmente relevante; es una señal de supervisión barata y masiva (Gao et al., 2010, 2011, con los modelos BLTM y DPM). Segundo, el **deep learning para semántica**: Salakhutdinov y Hinton (2007) usaron autoencoders profundos (semantic hashing), pero seguían siendo **no supervisados** (optimizaban reconstrucción, no relevancia) y limitaban el vocabulario a 2000 palabras por escalabilidad. DSSM nace cruzando ambas líneas. Ver [/fundamentos/two-tower-retrieval](/fundamentos/two-tower-retrieval).

## Ideas principales

DSSM se sostiene sobre tres pilares que los autores resumen así:

1. **Entrenamiento discriminativo sobre clickthrough orientado a ranking.** Todos los parámetros se optimizan apuntando directo al objetivo de ranking: maximizar la verosimilitud condicional del documento clicado dada la consulta. Es la diferencia clave frente a LSA/PLSA/autoencoders no supervisados.
2. **Proyección profunda no lineal.** En vez de una sola proyección lineal (LSA), una **DNN multicapa** mapea consulta y documento al espacio común, capturando estructura semántica más rica.
3. **Word hashing por n-gramas de letras.** Una reducción de dimensionalidad que permite escalar a vocabularios de cientos de miles de palabras, el cuello de botella de los métodos previos.

La arquitectura resultante es lo que hoy llamamos **two-tower / dual-encoder**: dos torres (red de la consulta, red del documento) proyectan a un **espacio semántico común** y la relevancia es una **distancia** en ese espacio.

### Word hashing por n-gramas de letras

El vector de términos crudo tiene el tamaño del vocabulario (millones en web), inmanejable como entrada. El word hashing lo comprime: dada la palabra *good*, se le agregan marcas `#good#` y se descompone en **trigramas de letras** `#go, goo, ood, od#`; la palabra es el vector de esos n-gramas. Esto es la primera capa de la DNN, una **transformación lineal fija que no se entrena** ($W_1$).

El beneficio es grande: para un vocabulario de 500k palabras, los trigramas de letras dan un vector de **30 621 dimensiones** (reducción de 16×) con una **tasa de colisión de 0,0044 %** (22 de 500 000). Además es **robusto a out-of-vocabulary**: una palabra nunca vista igual se representa por n-gramas conocidos, y las variaciones morfológicas quedan cerca en el espacio de n-gramas.

### DNN, similaridad coseno y softmax

Con $x$ el vector de términos de entrada, $W_i$ y $b_i$ los pesos y sesgos, y $\tanh$ como activación:

$$l_1 = W_1 x, \qquad l_i = f(W_i l_{i-1} + b_i), \qquad y = f(W_N l_{N-1} + b_N)$$

$$f(x) = \frac{1 - e^{-2x}}{1 + e^{-2x}}$$

La relevancia entre consulta $Q$ y documento $D$ es la **similaridad coseno** de sus vectores concepto $y_Q, y_D$:

$$R(Q, D) = \text{cosine}(y_Q, y_D) = \frac{y_Q^{T} y_D}{\lVert y_Q \rVert \, \lVert y_D \rVert}$$

El puntaje se convierte en **probabilidad posterior** vía softmax con un factor de suavizado $\gamma$:

$$P(D \mid Q) = \frac{\exp(\gamma \, R(Q, D))}{\sum_{D' \in \mathbf{D}} \exp(\gamma \, R(Q, D'))}$$

En la práctica, para cada par $(Q, D^+)$ con documento clicado $D^+$, se aproxima $\mathbf{D}$ con $D^+$ y **cuatro documentos no clicados al azar** $\{D_j^-\}$. El entrenamiento minimiza la verosimilitud negativa:

$$L(\Lambda) = -\log \prod_{(Q, D^+)} P(D^+ \mid Q)$$

Como $L(\Lambda)$ es diferenciable, se entrena con SGD por mini-batches (1024 muestras, ~20 épocas). La arquitectura: capa de word hashing (~30k nodos) → dos capas ocultas de 300 → salida de **128 dimensiones** (el espacio semántico común).

## Resultados experimentales

Evaluado sobre 16 510 consultas en inglés de un motor comercial (~15 URLs por consulta, etiquetas humanas 0-4), entrenado con ~100 millones de pares consulta-título, métrica **NDCG@{1,3,10}** con validación cruzada 2-fold y test de significancia (paired t-test, p < 0,05).

| # | Modelo | NDCG@1 | NDCG@3 | NDCG@10 |
|---|---|---|---|---|
| 1 | TF-IDF | 0.319 | 0.382 | 0.462 |
| 2 | BM25 | 0.308 | 0.373 | 0.455 |
| 4 | LSA | 0.298 | 0.372 | 0.455 |
| 5 | PLSA | 0.295 | 0.371 | 0.456 |
| 6 | DAE (autoencoder profundo) | 0.310 | 0.377 | 0.459 |
| 7 | BLTM-PR | 0.337 | 0.403 | 0.480 |
| 8 | DPM | 0.329 | 0.401 | 0.479 |
| 9 | DNN (sin word hashing) | 0.342 | 0.410 | 0.486 |
| **12** | **L-WH DNN (mejor DSSM)** | **0.362** | **0.425** | **0.498** |

Hallazgos del paper:

- El mejor DSSM (L-WH DNN) supera a todos los competidores por margen estadísticamente significativo: **+2,5-4,3 % en NDCG@1** sobre el mejor baseline previo.
- **La supervisión sobre clickthrough es esencial:** DNN (fila 9) y DAE (fila 6) comparten vocabulario (40k) y arquitectura, pero DNN es supervisado y DAE no; DNN gana 3,2 puntos de NDCG@1.
- **El word hashing habilita vocabularios grandes:** el modelo de 500k palabras supera al de 40k, aun teniendo menos parámetros libres.
- **La profundidad ayuda:** pasar de 1 a 3 capas no lineales sube 0,4-0,5 puntos de NDCG.

El análisis de errores muestra que las victorias del DSSM vienen de coincidencias semánticas (no léxicas), y la visualización de nodos de salida revela clusters semánticamente coherentes (cars/auto/vehicle; chevrolet/toyota/chevy).

## Limitaciones reconocibles

- **Bolsa de palabras:** la entrada es un vector de recuentos sin orden; no modela secuencia ni sintaxis (eso llega después con CNN-DSSM, LSTM-DSSM y luego Transformers).
- **Solo el título** del documento se usa para ranking (por el diseño con URLs de cola sin clics).
- **Negativos aleatorios:** solo 4 negativos muestreados al azar, sin hard-negative mining.
- **Colisiones de word hashing:** mínimas, pero existen.
- **Datos propietarios:** logs de un motor comercial no público; sin arxiv ni reproducción abierta directa. Evaluación monolingüe (inglés).

## Por qué importa hoy

DSSM es la **raíz del paradigma two-tower / dual-encoder**. Su patrón —dos codificadores que proyectan a un espacio común, comparan por coseno y se entrenan con softmax sobre un positivo y negativos— se convirtió en el esqueleto de los embeddings densos, el dense retrieval, los recomendadores two-tower a escala industrial (como [/papers/two-tower-yi-2019](/papers/two-tower-yi-2019)) y, conceptualmente, del aprendizaje contrastivo multimodal (CLIP y los recomendadores multimodales actuales). El softmax con negativos de DSSM es precursor de la pérdida contrastiva / sampled softmax que domina hoy, y el word hashing anticipa las representaciones subword (BPE, WordPiece, fastText) por su robustez a OOV. Ver [/fundamentos/recommender-systems](/fundamentos/recommender-systems).

## Conexión con la Clase 25

La [Clase 25](/clases/clase-25) (recomendación multimodal) usa **dos torres** que proyectan usuario/contexto e ítem a un **espacio común** y comparan por **distancia**. DSSM es su antecesor directo:

- **Dos torres → espacio común:** en DSSM, la torre de la consulta y la del documento mapean a un mismo espacio de 128 dimensiones; en la Clase 25, las torres mapean usuario e ítem (con features multimodales) al mismo espacio.
- **Similaridad como score:** $R(Q,D)=\text{cosine}(y_Q,y_D)$ es exactamente el scoring por producto interno/coseno que permite retrieval eficiente (vecinos aproximados) a escala.
- **Softmax sobre negativos:** el $P(D\mid Q)$ con un positivo y negativos muestreados es la semilla de la pérdida contrastiva / sampled softmax de los two-tower modernos.
- **Feedback implícito:** clickthrough en DSSM ≈ interacciones (clics, compras, vistas) en recomendación.

La Clase 25 reemplaza el bag-of-words por encoders multimodales y escala con muestreo de negativos y ANN, pero la columna vertebral —dos torres, un espacio, una similaridad— es la de DSSM. Ver [/fundamentos/two-tower-retrieval](/fundamentos/two-tower-retrieval).

## Notas y enlaces

- **Paper:** Huang, He, Gao, Deng, Acero, Heck. "Learning Deep Structured Semantic Models for Web Search using Clickthrough Data." CIKM 2013.
- **PDF:** [/papers/dssm-huang-2013.pdf](/papers/dssm-huang-2013.pdf)
- **Relacionados en el sitio:** [/papers/two-tower-yi-2019](/papers/two-tower-yi-2019) · [/fundamentos/two-tower-retrieval](/fundamentos/two-tower-retrieval) · [/fundamentos/recommender-systems](/fundamentos/recommender-systems) · [/clases/clase-25](/clases/clase-25)
