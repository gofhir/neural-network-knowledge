---
title: "Profundizacion - Recomendación Multimodal"
weight: 20
math: true
---

> Math riguroso que sustenta la [Clase 25](/clases/clase-25) — *Recomendación usando Imágenes y Texto*. Cinco partes: (I) **formalización del problema de recomendación** como composición $r_{ij} = h(g(u_i), f(x_j, c_j))$, (II) **metric learning** — por qué clasificar por usuario induce un espacio de co-preferencia, con triplet loss y BPR, (III) **two-tower y sampled softmax** con la corrección log-Q de Yi et al. 2019, (IV) **representación de datos heterogéneos** — embeddings, proyecciones, transformer sin/con positional y demostración de invarianza a permutación, (V) **métricas de ranking** — Precision@k, Recall@k, MAP, MRR, nDCG con descuento logarítmico y reproducción del ejemplo numérico de la clase.

---

## Parte I — Formalización del problema de recomendación

### I.1 Espacio de pins y embedding multimodal

Sea $\mathcal{P}$ el catálogo de **pins** (items). Cada pin $p_j \in \mathcal{P}$ se describe por una imagen $x_j \in \mathcal{X}$ (tensor de píxeles) y un conjunto de atributos de contexto $c_j \in \mathcal{C}$ (texto, categorías, tags, metadata numérica). El núcleo del sistema es un **embedding multimodal**

$$
f : \mathcal{X} \times \mathcal{C} \longrightarrow \mathbb{R}^d, \qquad f(x_j, c_j) = \mathbf{p}_j \in \mathbb{R}^d,
$$

que mapea cada pin a un vector denso de dimensión $d$. La idea de fondo (ver [/fundamentos/recommender-systems](/fundamentos/recommender-systems)) es que la **geometría** de $\mathbb{R}^d$ codifique afinidad: pins semánticamente parecidos o co-preferidos por los mismos usuarios quedan cerca.

### I.2 Representación del usuario como agregación de sus pins

A diferencia de la factorización matricial clásica, donde el embedding del usuario es una fila aprendida de una tabla $U \in \mathbb{R}^{m \times d}$, en el enfoque *content-based* multimodal el usuario $u_i$ se representa **agregando los embeddings de los pins con los que interactuó**. Sea $\mathcal{H}_i \subseteq \mathcal{P}$ su historial. Entonces

$$
g(u_i) \;=\; \mathrm{Agg}\big(\{\, f(x_j, c_j) : p_j \in \mathcal{H}_i \,\}\big) \in \mathbb{R}^d.
$$

La agregación $\mathrm{Agg}$ puede ser un promedio simple

$$
g(u_i) = \frac{1}{|\mathcal{H}_i|} \sum_{p_j \in \mathcal{H}_i} f(x_j, c_j),
$$

una suma pesada por recencia/importancia, o una atención aprendida (Parte IV). La ventaja decisiva es que $g$ generaliza a usuarios y pins **nunca vistos** (cold-start): basta su contenido, sin necesidad de un id en una tabla.

### I.3 Función de relevancia y la matriz usuario-item

La relevancia del pin $j$ para el usuario $i$ es una composición

$$
\boxed{\; r_{ij} \;=\; h\big(g(u_i),\, f(x_j, c_j)\big) \;}
$$

donde $h : \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$ es una **función de relevancia**. Las dos elecciones canónicas:

- **Producto interno** $h(\mathbf{u}, \mathbf{p}) = \langle \mathbf{u}, \mathbf{p}\rangle = \mathbf{u}^\top \mathbf{p}$.
- **Similitud coseno** $h(\mathbf{u}, \mathbf{p}) = \dfrac{\mathbf{u}^\top \mathbf{p}}{\lVert \mathbf{u}\rVert\,\lVert \mathbf{p}\rVert}$, equivalente al producto interno si los vectores se normalizan a la esfera unitaria.

A veces se usa una **distancia** $h(\mathbf{u},\mathbf{p}) = -\lVert \mathbf{u} - \mathbf{p}\rVert_2^2$ (relevancia decreciente con distancia). Notar la identidad que conecta ambas vistas: para vectores normalizados $\lVert\mathbf{u}\rVert=\lVert\mathbf{p}\rVert=1$,

$$
\lVert \mathbf{u} - \mathbf{p}\rVert_2^2 = \lVert\mathbf{u}\rVert^2 + \lVert\mathbf{p}\rVert^2 - 2\,\mathbf{u}^\top\mathbf{p} = 2 - 2\,\langle\mathbf{u},\mathbf{p}\rangle,
$$

de modo que **minimizar distancia euclídea $\Leftrightarrow$ maximizar producto interno** en la esfera. Esto justifica usar índices de vecinos más cercanos (ANN) tanto para coseno como para $\ell_2$.

### I.4 Feedback implícito

En recomendación a escala no hay ratings explícitos; observamos **feedback implícito**: clics, guardados, dwell time. Modelamos la interacción observada como una matriz binaria $Y \in \{0,1\}^{m\times n}$ con $y_{ij}=1$ si $u_i$ interactuó con $p_j$. La señal positiva es confiable, pero los $y_{ij}=0$ son ambiguos (no-observado $\ne$ no-relevante). El objetivo no es reconstruir $Y$ entrada por entrada (eso sería regresión sobre ceros ruidosos), sino **ordenar** los pins de manera que los positivos queden arriba — un problema de *ranking*, no de regresión. Esto motiva las pérdidas pairwise (BPR) y de metric learning de la Parte II y las métricas de la Parte V.

---

## Parte II — Metric learning y el espacio de embeddings

### II.1 El objetivo geométrico

La afirmación operativa de la clase es: *"los pins del mismo usuario quedan cerca en el espacio de embeddings"*. Formalicemos. Queremos que $f$ satisfaga, para usuarios $u_i$ y pins $p_a, p_b \in \mathcal{H}_i$, $p_c \notin \mathcal{H}_i$:

$$
d\big(f(p_a), f(p_b)\big) \;<\; d\big(f(p_a), f(p_c)\big),
$$

es decir, dos pins co-preferidos por el mismo usuario estén más cerca que un pin no relacionado. Hay dos rutas equivalentes para inducir esta geometría: la **clasificación de usuario** (softmax) y las **pérdidas pairwise/triplet**.

### II.2 Clasificación de usuario → embedding (proxy-based)

Tratamos a cada usuario como una **clase**. Dado un pin $p_j$ del historial de $u_i$, entrenamos un clasificador que prediga el usuario $i$ a partir de la representación del pin $\mathbf{p}_j = f(x_j,c_j)$. Con $N$ usuarios y un vector de pesos $\mathbf{w}_i \in \mathbb{R}^d$ por usuario (el "proxy" del usuario), el softmax sobre usuarios es

$$
P(i \mid p_j) = \frac{\exp\!\big(\mathbf{w}_i^\top \mathbf{p}_j\big)}{\sum_{k=1}^{N} \exp\!\big(\mathbf{w}_k^\top \mathbf{p}_j\big)},
$$

y la **cross-entropy** a minimizar sobre los pares observados $(i,j)$ es

$$
\mathcal{L}_{\text{cls}} = -\sum_{(i,j)} \log P(i \mid p_j) = -\sum_{(i,j)} \Big[ \mathbf{w}_i^\top \mathbf{p}_j - \log \sum_{k=1}^{N} \exp\!\big(\mathbf{w}_k^\top \mathbf{p}_j\big) \Big].
$$

**El embedding es el penúltimo layer.** La red tiene la forma $\mathbf{p}_j = f(x_j, c_j)$ (backbone multimodal) seguida de una capa lineal de logits $\mathbf{w}_k^\top \mathbf{p}_j$. Una vez entrenada, **descartamos la capa de clasificación** $\{\mathbf{w}_k\}$ y usamos $\mathbf{p}_j$ como representación. ¿Por qué esto induce co-preferencia? El gradiente respecto de $\mathbf{p}_j$ es

$$
\frac{\partial \mathcal{L}_{\text{cls}}}{\partial \mathbf{p}_j} = -\mathbf{w}_i + \sum_{k=1}^N P(k\mid p_j)\,\mathbf{w}_k,
$$

cuyo paso de descenso **acerca $\mathbf{p}_j$ a su proxy $\mathbf{w}_i$** y lo **aleja** de los proxies de otros usuarios (ponderados por su probabilidad). Como todos los pins de $u_i$ son empujados hacia el mismo $\mathbf{w}_i$, terminan agrupados — exactamente la propiedad deseada. Los proxies $\mathbf{w}_i$ actúan como centroides aprendidos.

### II.3 Triplet loss (sample-based)

La alternativa directa es el **triplet loss** (ver [/fundamentos/triplet-loss](/fundamentos/triplet-loss)). Dado un ancla $a$, un positivo $p$ (mismo usuario) y un negativo $n$ (distinto usuario):

$$
\mathcal{L}_{\text{trip}}(a,p,n) = \Big[\, \lVert f(a) - f(p)\rVert_2^2 - \lVert f(a) - f(n)\rVert_2^2 + \alpha \,\Big]_+,
$$

con $[\,z\,]_+ = \max(0, z)$ y margen $\alpha > 0$. La pérdida es cero solo si el negativo está al menos $\alpha$ más lejos que el positivo. El subgradiente respecto del ancla, cuando el margen se viola, es

$$
\frac{\partial \mathcal{L}_{\text{trip}}}{\partial f(a)} = 2\big(f(n) - f(p)\big),
$$

que empuja el ancla en la dirección $f(p) - f(n)$: hacia el positivo, lejos del negativo. La diferencia con II.2 es que aquí no hay proxy global por usuario; las restricciones son **relativas** entre tripletas muestreadas, lo que evita mantener $N$ vectores $\mathbf{w}_k$ pero exige *hard-negative mining* para no colapsar.

### II.4 BPR — Bayesian Personalized Ranking (pairwise)

[BPR (Rendle et al. 2009)](/papers/bpr-rendle-2009) optimiza el orden con una pérdida **pairwise** sobre feedback implícito (ver [/fundamentos/recommender-systems](/fundamentos/recommender-systems)). Para el usuario $i$, un positivo $p_a \in \mathcal{H}_i$ y un negativo $p_b \notin \mathcal{H}_i$, se desea $r_{ia} > r_{ib}$. BPR modela la probabilidad de ordenarlos correctamente con la sigmoide $\sigma(z) = 1/(1+e^{-z})$:

$$
\mathcal{L}_{\text{BPR}} = -\sum_{(i,a,b)} \log \sigma\big(r_{ia} - r_{ib}\big) + \lambda \lVert\Theta\rVert^2,
$$

donde $r_{ij} = h(g(u_i), f(p_j))$ y $\Theta$ son los parámetros. El gradiente del término principal respecto del *score gap* $\delta_{iab} = r_{ia} - r_{ib}$ es

$$
\frac{\partial}{\partial \delta_{iab}} \big[-\log\sigma(\delta_{iab})\big] = -\big(1 - \sigma(\delta_{iab})\big) = \sigma(\delta_{iab}) - 1 < 0,
$$

de modo que el descenso **aumenta** la brecha $\delta_{iab}$ — empuja el positivo arriba del negativo. Las tres pérdidas (softmax, triplet, BPR) comparten el mismo norte: **distancia/score que refleja co-preferencia**; difieren en si la señal es absoluta (softmax con proxies), de triplete con margen, o pairwise probabilística.

---

## Parte III — Two-tower y sampled softmax

### III.1 Arquitectura two-tower

El sistema se implementa como un modelo de **dos torres** (ver [/fundamentos/two-tower-retrieval](/fundamentos/two-tower-retrieval) y [Yi et al. 2019](/papers/two-tower-yi-2019)): una torre de **query/usuario** $g(\cdot)$ y una torre de **item/pin** $f(\cdot)$, ambas mapeando a $\mathbb{R}^d$, con score por producto interno

$$
s(u, p) = \langle g(u), f(p)\rangle.
$$

La virtud es que los embeddings de pins $f(p)$ se **precomputan** y se indexan con ANN; en *serving* solo se calcula $g(u)$ y se hace un nearest-neighbor search. Las torres no comparten pesos pero comparten el espacio de salida.

### III.2 Full softmax y por qué no escala

El entrenamiento de retrieval se plantea como clasificación multiclase sobre **todo el corpus** $\mathcal{P}$: dado un usuario $u$, predecir el item positivo $p$ entre los $|\mathcal{P}|$ candidatos:

$$
P(p \mid u) = \frac{\exp\!\big(s(u,p)\big)}{\sum_{p' \in \mathcal{P}} \exp\!\big(s(u,p')\big)}.
$$

El denominador suma sobre **millones** de items: intratable por paso de gradiente. La solución es estimar el softmax con un subconjunto de negativos.

### III.3 In-batch negatives y sampled softmax

Con un batch de $B$ pares positivos $\{(u_i, p_i)\}_{i=1}^B$, usamos los **otros items del mismo batch** como negativos (in-batch negatives). El softmax aproximado para el par $i$ es

$$
\hat{P}(p_i \mid u_i) = \frac{\exp\!\big(s(u_i, p_i)\big)}{\sum_{j=1}^{B} \exp\!\big(s(u_i, p_j)\big)}.
$$

Esto es eficiente — reutiliza la matriz de scores $B\times B$ ya calculada. Pero introduce un **sesgo**: los items populares aparecen en muchos batches y se sobre-penalizan como negativos, distorsionando el estimador del softmax completo.

### III.4 La corrección log-Q (Yi et al. 2019)

Sea $p_j$ la probabilidad de que el item $j$ sea **muestreado** en un batch (proporcional a su popularidad). El estimador insesgado del softmax con muestreo por importancia requiere restar $\log p_j$ del logit de cada candidato:

$$
\boxed{\; s^{\text{corr}}(u_i, p_j) = s(u_i, p_j) - \log p_j \;}
$$

y el softmax corregido es

$$
\hat{P}^{\text{corr}}(p_i \mid u_i) = \frac{\exp\!\big(s(u_i,p_i) - \log p_i\big)}{\sum_{j=1}^{B} \exp\!\big(s(u_i,p_j) - \log p_j\big)}.
$$

**Derivación del sesgo.** El softmax completo que queremos estimar es $P(p\mid u) = \exp(s(u,p))/Z$ con $Z = \sum_{p'} \exp(s(u,p'))$. Al muestrear candidatos según una propuesta $Q$ con probabilidad $p_j = Q(j)$, el estimador de Monte Carlo del denominador con corrección de importancia es

$$
\hat{Z} = \sum_{j \in \text{batch}} \frac{\exp(s(u,p_j))}{p_j} = \sum_{j} \exp\!\big(s(u,p_j) - \log p_j\big),
$$

porque $\exp(s)/p_j = \exp(s - \log p_j)$. Es decir, dividir por la probabilidad de muestreo (importance weighting) es **exactamente** restar $\log p_j$ en el dominio de los logits. Sin la corrección, un item popular (alto $p_j$) tiene su contribución al denominador inflada, lo que sub-estima $P(p_i\mid u_i)$ y empuja al modelo a **degradar** items populares. Con la corrección, $\mathbb{E}_Q[\hat{Z}] = Z$: el estimador es insesgado y la popularidad deja de contaminar el ranking. En la práctica $p_j$ se estima online con un *streaming frequency estimator* (count-min sketch sobre los ids vistos).

---

## Parte IV — Representación de tipos de datos heterogéneos

Un pin combina **features discretos** (categoría, tags, id), **continuos** (precio, dimensiones, popularidad) y **conjuntos/secuencias** (lista de tags, historial). Cada tipo requiere un tratamiento distinto antes de fusionarse en $\mathbb{R}^d$.

### IV.1 Discretos → embedding lookup

Un feature categórico con vocabulario de tamaño $V$ se representa con una tabla aprendida $E \in \mathbb{R}^{V \times d_e}$. El token con índice $v$ se mapea por **lookup**: $\mathbf{e}_v = E_{v,:} \in \mathbb{R}^{d_e}$. Equivale a $\mathbf{e}_v = \mathbf{1}_v^\top E$ con $\mathbf{1}_v$ el one-hot — una multiplicación matriz-vector que la implementación reemplaza por indexación.

### IV.2 Continuos → proyección lineal

Un feature numérico $x \in \mathbb{R}$ (tras normalización, p.ej. estandarización $z = (x-\mu)/\sigma$ o *log1p*) se proyecta a $\mathbb{R}^{d_e}$ con una capa afín:

$$
\mathbf{e} = \mathbf{w}\, z + \mathbf{b}, \qquad \mathbf{w}, \mathbf{b} \in \mathbb{R}^{d_e}.
$$

Para un vector de $m$ continuos, $\mathbf{e} = W \mathbf{z} + \mathbf{b}$ con $W \in \mathbb{R}^{d_e \times m}$. La normalización previa es crítica: sin ella, features con escalas dispares dominan los gradientes.

### IV.3 Conjuntos → transformer encoder sin positional (invarianza a permutación)

Un conjunto de tags $\{t_1, \dots, t_L\}$ no tiene orden. Lo procesamos con un **transformer encoder sin positional encoding**, que es invariante a permutación. **Demostración.** Sea $X = [\mathbf{x}_1; \dots; \mathbf{x}_L] \in \mathbb{R}^{L\times d}$ la matriz de embeddings de entrada y $\pi$ una permutación con matriz $P_\pi \in \{0,1\}^{L\times L}$, de modo que $(P_\pi X)$ reordena las filas. La self-attention (una cabeza) calcula

$$
\mathrm{Attn}(X) = \mathrm{softmax}\!\Big(\tfrac{(XW_Q)(XW_K)^\top}{\sqrt{d_k}}\Big)(XW_V).
$$

Definamos $S(X) = \mathrm{softmax}\big(\tfrac{1}{\sqrt{d_k}}(XW_Q)(XW_K)^\top\big)$ (matriz $L\times L$ de pesos, softmax por filas). Bajo la permutación $X \mapsto P_\pi X$:

$$
S(P_\pi X) = \mathrm{softmax}\!\Big(\tfrac{1}{\sqrt{d_k}}\,P_\pi (XW_Q)(XW_K)^\top P_\pi^\top\Big) = P_\pi\, S(X)\, P_\pi^\top,
$$

donde la última igualdad usa que el softmax actúa fila a fila y $P_\pi$ permuta filas y $P_\pi^\top$ columnas de forma consistente. Entonces

$$
\mathrm{Attn}(P_\pi X) = S(P_\pi X)(P_\pi X W_V) = P_\pi S(X) P_\pi^\top P_\pi X W_V = P_\pi\, S(X)\, X W_V = P_\pi\,\mathrm{Attn}(X),
$$

usando $P_\pi^\top P_\pi = I$. Es decir, la salida se **permuta de la misma manera** que la entrada: la función es **equivariante** a permutación. Si después agregamos con una operación simétrica — suma, media o un token `[CLS]` que atiende a todos por igual — obtenemos **invarianza**: $\mathrm{Pool}(\mathrm{Attn}(P_\pi X)) = \mathrm{Pool}(\mathrm{Attn}(X))$. Por eso, para conjuntos, **se omite el positional encoding**: incluirlo rompería esta invarianza e impondría un orden artificial.

### IV.4 Secuencias → transformer encoder con positional

Para datos con orden real (historial temporal de interacciones), **sí** añadimos positional encoding $\mathbf{x}_t \leftarrow \mathbf{x}_t + \mathbf{pos}_t$. Esto rompe deliberadamente la equivariancia: $\mathrm{Attn}(P_\pi(X + \mathrm{Pos})) \ne P_\pi\,\mathrm{Attn}(X+\mathrm{Pos})$ en general, porque $\mathrm{Pos}$ no se permuta con los datos. Así el modelo distingue "vio A luego B" de "vio B luego A".

### IV.5 Combinación de modalidades y la capa lineal final

Tras obtener un embedding por feature/modalidad $\{\mathbf{e}^{(1)}, \dots, \mathbf{e}^{(M)}\}$, hay tres estrategias de fusión:

- **Concatenación**: $\mathbf{z} = [\mathbf{e}^{(1)}; \dots; \mathbf{e}^{(M)}] \in \mathbb{R}^{\sum_k d_k}$. Conserva toda la información pero la dimensión crece y los features no quedan en un espacio común.
- **Suma**: $\mathbf{z} = \sum_k \mathbf{e}^{(k)}$ (requiere $d_k$ iguales). Compacta pero asume aditividad y pierde de qué modalidad vino cada componente.
- **Atención**: $\mathbf{z} = \sum_k \alpha_k \mathbf{e}^{(k)}$ con pesos $\alpha_k = \mathrm{softmax}(a_k)$ aprendidos o dependientes del input. Pondera modalidades dinámicamente.

En todos los casos se aplica una **capa lineal final** $\mathbf{p} = W_o \mathbf{z} + \mathbf{b}_o$ con $W_o \in \mathbb{R}^{d \times \dim(\mathbf{z})}$. ¿Por qué? Tres razones: (1) **proyecta a la dimensión común $d$** del espacio de matching, indispensable cuando se concatena; (2) **mezcla las modalidades** — permite que el modelo aprenda interacciones cruzadas que la mera concatenación deja desacopladas; (3) **alinea ambas torres** al mismo $\mathbb{R}^d$ donde el producto interno $s(u,p)$ tiene sentido. Sin ella, las modalidades quedarían en subespacios disjuntos y el score no las combinaría.

---

## Parte V — Métricas de ranking: derivaciones

Para evaluar el orden producido (no la predicción puntual) usamos métricas de ranking (ver [/fundamentos/ranking-metrics](/fundamentos/ranking-metrics)). Sea, para un usuario, la lista ordenada de items recomendados, $\mathrm{rel}(i) \in \{0,1\}$ (binaria) o $\mathrm{rel}_i \in \{0,1,2,\dots\}$ (graduada) la relevancia del item en la posición $i$, y $|\mathrm{rel}|$ el número total de items relevantes.

### V.1 Precision@k y Recall@k

$$
P@k = \frac{1}{k}\sum_{i=1}^{k} \mathrm{rel}(i), \qquad R@k = \frac{1}{|\mathrm{rel}|}\sum_{i=1}^{k} \mathrm{rel}(i).
$$

$P@k$ mide la fracción de los top-$k$ que son relevantes; $R@k$ la fracción de todos los relevantes que aparecen en el top-$k$. Ninguna es sensible al **orden dentro** del top-$k$ — una limitación que MAP y nDCG corrigen.

### V.2 Average Precision y MAP

La **Average Precision** de una consulta promedia $P@k$ solo en las posiciones donde hay un acierto:

$$
\mathrm{AP} = \frac{1}{|\mathrm{rel}|}\sum_{k=1}^{n} P@k \cdot \mathrm{rel}(k),
$$

donde $\mathrm{rel}(k)\in\{0,1\}$ activa el término únicamente cuando la posición $k$ es relevante. Premia colocar los relevantes **temprano**: un acierto en posición 1 contribuye con $P@1=1$, mientras que el mismo acierto en posición 10 contribuye con a lo más $P@10=0.1$. La **Mean Average Precision** promedia sobre el conjunto de consultas $\mathcal{Q}$:

$$
\mathrm{MAP} = \frac{1}{|\mathcal{Q}|}\sum_{q\in\mathcal{Q}} \mathrm{AP}_q.
$$

### V.3 MRR — Mean Reciprocal Rank

Cuando interesa solo el **primer** acierto (p.ej. respuesta correcta única), se usa el rango recíproco. Si $\mathrm{rank}_q$ es la posición del primer item relevante para la consulta $q$:

$$
\mathrm{MRR} = \frac{1}{|\mathcal{Q}|}\sum_{q\in\mathcal{Q}} \frac{1}{\mathrm{rank}_q}.
$$

Primer acierto en posición 1 aporta $1$; en posición 2 aporta $0.5$; en posición 5 aporta $0.2$ — un decaimiento armónico que castiga fuerte cualquier demora.

### V.4 DCG, iDCG y nDCG

El **Discounted Cumulative Gain** introduce dos ideas clave (ver [Järvelin & Kekäläinen 2002](/papers/ndcg-jarvelin-2002) y [/fundamentos/ranking-metrics](/fundamentos/ranking-metrics)): **relevancia graduada** (no solo 0/1) y **descuento por posición**:

$$
DCG@k = \sum_{i=1}^{k} \frac{\mathrm{rel}_i}{\log_2(i+1)}.
$$

El divisor $\log_2(i+1)$ penaliza posiciones tardías: posición 1 tiene descuento $\log_2 2 = 1$ (sin penalización), posición 2 tiene $\log_2 3 \approx 1.585$, etc. **¿Por qué logarítmico y no lineal o exponencial?** El descuento logarítmico modela un decaimiento **suave y de cola pesada** de la atención del usuario: la diferencia de utilidad entre las posiciones 1 y 2 debe ser mayor que entre las posiciones 9 y 10, pero no infinitamente — el usuario aún puede mirar resultados profundos. Un descuento lineal ($1/i$) cae demasiado rápido; uno constante ignoraría el orden. El log captura el punto medio empírico observado en logs de clics.

El **iDCG** (ideal DCG) es el $DCG@k$ del **orden perfecto** — los items ordenados por relevancia descendente. El **normalized DCG** es

$$
nDCG@k = \frac{DCG@k}{iDCG@k} \in [0,1],
$$

acotado a $[0,1]$ para poder promediar entre consultas con distinto número de relevantes.

### V.5 Reproducción del ejemplo numérico de la clase

La clase presenta un ejemplo con cinco resultados relevantes (relevancia binaria, $\mathrm{rel}=1$ por acierto). El **ranking producido** acierta en las posiciones 2, 4 y 5 y falla en 1 y 3, es decir, el vector de relevancia ordenado por el modelo es $[\,0,1,0,1,1\,]$. El **orden ideal** colocaría todos los aciertos primero: $[\,1,1,1,1,1\,]$.

**DCG del ranking producido:**

$$
DCG@5 = \frac{0}{\log_2 2} + \frac{1}{\log_2 3} + \frac{0}{\log_2 4} + \frac{1}{\log_2 5} + \frac{1}{\log_2 6}
$$

$$
= 0 + \frac{1}{1.585} + 0 + \frac{1}{2.322} + \frac{1}{2.585} = 0.6309 + 0.4307 + 0.3869 = \mathbf{1.4485}.
$$

**iDCG (orden ideal $[1,1,1,1,1]$):**

$$
iDCG@5 = \frac{1}{\log_2 2} + \frac{1}{\log_2 3} + \frac{1}{\log_2 4} + \frac{1}{\log_2 5} + \frac{1}{\log_2 6}
$$

$$
= 1 + 0.6309 + 0.5 + 0.4307 + 0.3869 = \mathbf{2.9485}.
$$

**nDCG:**

$$
nDCG@5 = \frac{DCG@5}{iDCG@5} = \frac{1.4485}{2.9485} = \mathbf{0.4912}.
$$

Los tres valores coinciden exactamente con los de la clase: $DCG=1.4485$, $iDCG=2.9485$, $nDCG=0.4912$. La lectura: el ranking captura menos del 50% del *gain* ideal disponible, principalmente por haber fallado en la posición 1 (la de mayor peso, descuento $=1$) — exactamente el tipo de error que el descuento logarítmico castiga con más severidad.

---

> **Síntesis.** El sistema es la composición $r_{ij} = h(g(u_i), f(x_j,c_j))$: un embedding multimodal $f$ y una agregación de usuario $g$, fusionados por una relevancia $h$ (Parte I). Entrenar como clasificación de usuario o con triplet/BPR induce un espacio donde la distancia refleja co-preferencia (Parte II). La implementación es two-tower con sampled softmax corregido por log-Q para insesgar el muestreo de negativos (Parte III). Los datos heterogéneos se representan con embeddings, proyecciones y transformers — sin positional para conjuntos (invariancia a permutación demostrada), con positional para secuencias (Parte IV). Y la evaluación usa métricas de ranking sensibles al orden, culminando en nDCG con descuento logarítmico (Parte V).

**Cross-links.** [Clase 25](/clases/clase-25) · [Recommender systems](/fundamentos/recommender-systems) · [Ranking metrics](/fundamentos/ranking-metrics) · [Two-tower retrieval](/fundamentos/two-tower-retrieval) · [Triplet loss](/fundamentos/triplet-loss) · [Yi et al. 2019](/papers/two-tower-yi-2019) · [BPR — Rendle 2009](/papers/bpr-rendle-2009) · [nDCG — Järvelin 2002](/papers/ndcg-jarvelin-2002).
