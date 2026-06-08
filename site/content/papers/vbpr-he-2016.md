---
title: "VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback"
weight: 260
math: true
---

{{< paper-card
    title="VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback"
    authors="He, McAuley"
    year="2016"
    venue="AAAI 2016"
    pdf="/papers/vbpr-he-2016.pdf"
    arxiv="1510.01784" >}}
**VBPR** incorpora la **apariencia visual** de los productos al problema de recomendación. Toma **features de una Deep CNN (4096-d, salida FC7 de AlexNet pre-entrenada en ImageNet)**, las proyecta con una **matriz de embedding compartida** $E$ a un espacio de "rating visual" de baja dimensión, y suma esos **factores visuales** a la factorización matricial clásica. Se entrena con **Bayesian Personalized Ranking (BPR)** sobre **feedback implícito**. Resultado: mejora **>12% en AUC global y >28% en cold start** sobre BPR-MF, y hasta **+44,9%** en el dataset de segunda mano Tradesy.com. Es el trabajo seminal de recomendación visual personalizada y el más cercano al case study de la [Clase 25](/clases/clase-25).
{{< /paper-card >}}

---

## Contexto

Los sistemas de recomendación descubren las **dimensiones latentes** que explican las preferencias de los usuarios a partir de su feedback histórico. Ese feedback suele ser **implícito** (historiales de compra, clics, navegación) más que explícito (calificaciones con estrellas). Para modelarlo a gran escala se usa **Matrix Factorization (MF)**, que sin embargo sufre de **cold start**: un ítem con muy pocas (o cero) interacciones no tiene señal suficiente para estimar sus factores latentes. Ver [/fundamentos/recommender-systems](/fundamentos/recommender-systems).

El gancho del paper es intuitivo: **uno no compraría una polera en Amazon sin ver la imagen**. La apariencia visual es decisiva al comprar ropa, calzado o accesorios, y es justamente la señal que los sistemas tradicionales ignoran. Trabajos previos habían incorporado texto, ubicación o estación del año, pero ninguno la apariencia visual directamente en el predictor de preferencias.

El momento lo permite: hacia 2015 las Deep CNN ([/fundamentos/redes-convolucionales](/fundamentos/redes-convolucionales)) ya dominaban tareas de visión, y el [transfer learning](/fundamentos/transfer-learning) había demostrado que una CNN entrenada en ImageNet produce features genéricas reutilizables "out-of-the-box" en otras tareas. VBPR explota exactamente esa propiedad: no entrena una CNN, usa features pre-computadas y aprende una capa encima.

## Ideas principales

El predictor de MF clásico modela la compatibilidad usuario-ítem con un producto interno de factores latentes. VBPR lo **extiende particionando las dimensiones** en factores latentes (no visuales) y **factores visuales** derivados de las features CNN.

### El predictor con factores visuales y embedding compartido

Partiendo del MF clásico $\hat{x}_{u,i} = \alpha + \beta_u + \beta_i + \gamma_u^T \gamma_i$ (offset global, sesgos, y factores latentes $\gamma_u, \gamma_i \in \mathbb{R}^K$), VBPR agrega **factores visuales** $\theta_u, \theta_i \in \mathbb{R}^D$. El problema es que las features CNN $f_i$ tienen $F = 4096$ dimensiones: usarlas directamente sería intratable. La solución es aprender una **matriz de embedding compartida** $E$ de tamaño $D \times F$ que proyecta las features al espacio de rating visual de baja dimensión (~20):

$$\theta_i = E f_i$$

Como **todos los ítems comparten la misma** $E$, la cantidad de parámetros se reduce drásticamente. Sumando además un **sesgo visual global** $\beta'$, el predictor final es:

$$\hat{x}_{u,i} = \alpha + \beta_u + \beta_i + \gamma_u^T \gamma_i + \theta_u^T (E f_i) + \beta'^T f_i$$

El término $\theta_u^T(E f_i)$ es la afinidad visual **personalizada** (cuánto le atrae al usuario cada faceta visual); $\beta'^T f_i$ es la opinión visual **global** sobre el ítem.

### Entrenamiento con BPR (pairwise, feedback implícito)

VBPR se entrena con **Bayesian Personalized Ranking** (ver [/papers/bpr-rendle-2009](/papers/bpr-rendle-2009)), un criterio **pairwise**. El conjunto de entrenamiento son triples $(u, i, j)$ donde $i$ es un ítem positivo del usuario y $j$ uno no observado. Se maximiza:

$$\sum_{(u,i,j)} \ln \sigma(\hat{x}_{u,i} - \hat{x}_{u,j}) - \lambda_\Theta \|\Theta\|^2$$

La suposición pairwise es débil pero realista: no se asume que lo no observado sea negativo, solo que el positivo debe rankearse **por encima**. El entrenamiento es por ascenso de gradiente estocástico; junto a los parámetros no visuales se actualizan los visuales $\theta_u$, $\beta'$ y la matriz $E$ (con un regularizador propio $\lambda_E$). La complejidad por triple es $O(K + D)$, **lineal en las dimensiones**, lo que mantiene el método escalable a millones de interacciones.

## Resultados experimentales

Cuatro datasets reales: **Amazon Women/Men Clothing**, **Amazon Cell Phones** y **Tradesy.com** (comunidad de ropa de segunda mano, inherentemente cold start por su naturaleza "one-off"). En total ~267 mil usuarios, ~790 mil ítems y ~2,49 millones de interacciones. Las features se extraen con Caffe usando AlexNet (Krizhevsky et al. 2012) pre-entrenada en ImageNet, tomando la salida **FC7 de 4096-d**. La métrica es **AUC**.

Con 20 factores totales (split 50/50 visual/latente), VBPR mejora sobre **BPR-MF** en promedio **más de 12% en all items y más de 28% en cold start**. Algunos números puntuales (AUC):

- **Amazon Women**: 0,7834 (all) vs 0,7020 de BPR-MF (+11,6%); cold start 0,6813 vs 0,5281 (+29,0%).
- **Amazon Men**: 0,7841 (all) vs 0,7100 (+10,4%); cold start 0,6898 vs 0,5512 (+25,1%).
- **Tradesy.com**: 0,7829 (all) vs 0,6198 (+26,3%); cold start **0,7594 vs 0,5241 (+44,9%)**.

Hallazgos clave: el baseline content-based **IBR** gana a MF en cold start pero pierde en warm start (no usa feedback); VBPR combina ambas fortalezas y supera a todos en casi todos los casos. Las features visuales aportan **más en ropa que en celulares** (en Phones cold start VBPR queda −4,2% bajo IBR, único caso negativo). VBPR le gana al método point-wise WRMF en 14,3% (all) y 20,3% (cold start). Entrena en ~3,5 horas en el dataset más grande sobre un desktop estándar. Una visualización t-SNE del espacio visual de 10-D muestra que el embedding aprende una transición coherente entre subcategorías de ropa.

## Limitaciones reconocibles

- **Una sola imagen por ítem** y **features CNN congeladas** (sin fine-tuning de la red): el espacio visual queda atado a lo que ImageNet considera relevante, que no siempre coincide con lo estéticamente importante para comprar.
- **Lo visual no domina en todas las categorías** (poco aporte en celulares).
- **Split 50/50 fijo** entre dimensiones visuales y latentes; los autores indican que afinarlo podría mejorar resultados.
- **Sin dinámica temporal** (no modela la deriva de las modas) y limitado a **feedback implícito** — ambos quedan como trabajo futuro.
- **AUC** valora el orden global pero es poco sensible al top-k que el usuario realmente ve.

## Por qué importa hoy

VBPR es el trabajo seminal de la **recomendación visual personalizada**. Su receta — features CNN pre-entrenadas + matriz de embedding compartida a un espacio visual de baja dimensión + entrenamiento BPR — se volvió plantilla replicada y extendida (Sherlock, Monomer, modelos de dinámica visual de moda de los mismos autores). Demostró que features visuales genéricas "off-the-shelf" tienen poder predictivo real sobre el comportamiento de compra, no solo sobre clasificación de objetos, y popularizó la idea de mitigar cold start con la señal de la imagen. Es un puente temprano entre visión por computador y sistemas de recomendación.

## Conexión con la Clase 25

La [Clase 25](/clases/clase-25) trata exactamente esto: **recomendación multimodal con features visuales de CNN + feedback implícito**. Es el paper más cercano al case study.

- **Features CNN 4096-d == el dataset Pinterest de la clase.** Igual que VBPR usa FC7 de AlexNet (4096-d), el case study trabaja con features visuales pre-extraídas de la misma dimensionalidad. Se **usan features pre-computadas y se aprende un embedding encima** ([transfer learning](/fundamentos/transfer-learning)), no se entrena la CNN — por eso las imágenes llegan como vectores y no como píxeles.
- **Ranking pairwise sobre feedback implícito.** La clase recomienda por ranking a partir de pins/likes, no de calificaciones; VBPR formaliza por qué [BPR](/papers/bpr-rendle-2009) (preferir lo positivo sobre lo no observado) es lo adecuado.
- **Cold start vía la señal visual.** El resultado estrella de VBPR (+44,9% en cold start) es justamente el argumento de la clase: un ítem nuevo sin historial igual tiene imagen, y por lo tanto factores visuales utilizables de inmediato.

## Notas y enlaces

- **PDF:** [/papers/vbpr-he-2016.pdf](/papers/vbpr-he-2016.pdf) · **arXiv:** [1510.01784](https://arxiv.org/abs/1510.01784)
- **Clase:** [/clases/clase-25](/clases/clase-25)
- **Paper relacionado:** [BPR — Bayesian Personalized Ranking (Rendle et al. 2009)](/papers/bpr-rendle-2009)
- **Fundamentos:** [recommender-systems](/fundamentos/recommender-systems) · [redes-convolucionales](/fundamentos/redes-convolucionales) · [transfer-learning](/fundamentos/transfer-learning)
