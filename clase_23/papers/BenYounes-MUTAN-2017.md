---
título: "MUTAN: Multimodal Tucker Fusion for Visual Question Answering"
autores: "Hedi Ben-younes, Rémi Cadene, Matthieu Cord, Nicolas Thome"
venue: "ICCV 2017 (IEEE International Conference on Computer Vision)"
año: 2017
arxiv: "1705.06676"
link: "https://arxiv.org/abs/1705.06676"
afiliaciones: "Sorbonne Universités, UPMC Univ Paris 06, CNRS, LIP6 UMR 7606; Heuritech; CNAM"
codigo: "https://github.com/cadene/vqa.pytorch"
clase: 23
tema: "Visual Question Answering — Fusión multimodal bilineal"
---

# MUTAN: Multimodal Tucker Fusion for Visual Question Answering

> **Cita.** H. Ben-younes, R. Cadene, M. Cord, N. Thome. *MUTAN: Multimodal Tucker Fusion for Visual Question Answering.* IEEE International Conference on Computer Vision (ICCV), 2017. arXiv:1705.06676. Hedi Ben-younes y Rémi Cadene contribuyeron por igual.

---

## 1. Resumen y posicionamiento

MUTAN (**MU**ltimodal **Tu**cker fusio**N**) es el paper canónico que cierra la línea de **fusión bilineal** en Visual Question Answering (VQA). Su tesis es simple y profunda: la fusión ideal entre la representación de una imagen $v$ y la de una pregunta $q$ es una **interacción bilineal completa** —es decir, todas las correlaciones de segundo orden entre cada dimensión de $v$ y cada dimensión de $q$—, pero esa interacción completa está parametrizada por un **tensor de tercer orden** $\mathcal{T} \in \mathbb{R}^{d_q \times d_v \times |\mathcal{A}|}$ cuyo número de parámetros es prohibitivo (del orden de $10^{10}$ con dimensiones realistas). El paper propone factorizar ese tensor mediante una **descomposición de Tucker**, lo que permite controlar explícitamente la complejidad del modelo sin renunciar a la riqueza de la interacción de segundo orden.

La contribución no es solo un modelo concreto con buen rendimiento (state-of-the-art en el dataset VQA en 2017), sino un **marco unificador**: MUTAN demuestra algebraicamente que dos de los modelos más fuertes de la época —MCB (Fukui et al. 2016) y MLB (Kim et al. 2017)— son **casos particulares** de una descomposición de Tucker con restricciones específicas sobre sus componentes. Es decir, MUTAN no compite con MCB y MLB; los subsume. Esa es la razón por la que es un paper "de pizarra": lo importante es la lección algebraica de que la **estructura tensorial** (Tucker + restricción de rango) es la palanca correcta para gobernar el trade-off entre expresividad y número de parámetros.

Para Roberto, que viene del mundo de los sistemas estructurados (FHIR, validadores, motores de reglas): MUTAN es a la fusión multimodal lo que un esquema normalizado es a una base de datos. En vez de almacenar la tabla cartesiana completa de interacciones (intratable), se factoriza en componentes de baja dimensión con un "núcleo" que codifica solo las correlaciones que importan, y se imponen restricciones estructurales (sparsity de rango) que actúan como regularizador.

---

## 2. Contexto — la línea de fusión bilineal en VQA

El problema de VQA, planteado por Antol et al. (2015), pide responder en lenguaje natural una pregunta sobre una imagen. La arquitectura genérica es siempre la misma: (1) un extractor visual produce $v \in \mathbb{R}^{d_v}$ (en MUTAN, ResNet-152); (2) un codificador de texto produce $q \in \mathbb{R}^{d_q}$ (en MUTAN, una GRU inicializada con Skip-thoughts); (3) un **módulo de fusión** combina $v$ y $q$ en un vector que pasa por un softmax sobre el vocabulario de respuestas $\mathcal{A}$ (las 2000 respuestas más frecuentes). El cuello de botella científico de la era 2015–2017 fue precisamente el paso (3).

La progresión histórica de las estrategias de fusión:

**Fusión de primer orden (lineal).** Los primeros modelos —IMG+BOW (Ren et al. 2015)— simplemente **concatenan** $v$ y $q$, o los suman elemento a elemento. Esto solo captura interacciones de primer orden: el modelo nunca aprende correlaciones explícitas del tipo "la dimensión $i$ de la imagen es relevante *cuando* la dimensión $j$ de la pregunta está activa". En MUTAN esto aparece como la línea base **Concat**, con 58.91 % de accuracy en *test-dev*.

**Fusión de segundo orden (bilineal completa).** La forma de capturar todas las interacciones cruzadas es el producto bilineal completo:

$$y = (\mathcal{T} \times_1 q) \times_2 v$$

donde $\mathcal{T} \in \mathbb{R}^{d_q \times d_v \times |\mathcal{A}|}$ y $\times_i$ es el producto modo-$i$ entre tensor y vector. Esto codifica **todas** las correlaciones de segundo orden. El problema, citado textualmente por los autores: con $d_v \approx d_q \approx 2048$ y $|\mathcal{A}| \approx 2000$, el tensor tiene $\sim 10^{10}$ parámetros. Un tensor de 8 mil millones de escalares en float32 ocupa $\sim 32$ GB, mientras que las GPUs de gama alta de la época tenían $\sim 24$ GB. Intratable tanto para entrenar como para almacenar.

**MCB — Multimodal Compact Bilinear pooling (Fukui et al. 2016).** Primera aproximación práctica. Calcula el producto externo $q \otimes v$ (que es la base de la interacción bilineal) pero lo proyecta a un espacio de menor dimensión usando **count-sketch** (Charikar et al. 2002) y la propiedad de que el count-sketch del producto externo es la **convolución** de los count-sketches individuales, computable eficientemente vía **FFT**. Ganó el VQA Challenge 2016. Su debilidad, identificada por MUTAN: los parámetros de interacción en MCB están **fijos** por la proyección count-sketch (vectores aleatorios en $\{0, -1, 1\}$ muestreados una vez y congelados), lo que limita su poder expresivo. Para compensar, MCB necesita una dimensión de salida enorme ($t_o \approx 16000$).

**MLB — Multimodal Low-rank Bilinear pooling (Kim et al. 2017).** Parametriza la interacción bilineal con un tensor restringido a ser de **rango bajo** $R$. La descomposición de rango bajo es un caso especial de Tucker en el que las tres dimensiones del core se igualan ($t_q = t_v = t_o = R$) y el core es la identidad. MLB alcanza state-of-the-art en VQA con muchos menos parámetros que MCB (7.7 M vs 32 M). Su debilidad, según MUTAN: la estructura de rango bajo equivale a proyectar ambas modalidades a un espacio común de dimensión $r$ y luego hacer un **simple producto de Hadamard** (elemento a elemento) en ese espacio. Es decir, MLB aprende excelentes proyecciones monomodales pero la fusión en sí es pobre (Hadamard simple, sin tensor de interacción aprendido).

El **problema persistente** que MUTAN ataca: ¿cómo controlar la complejidad de la interacción bilineal manteniendo —e incluso aumentando— la expresividad de la fusión, en vez de sacrificar una por la otra?

---

## 3. El tensor de interacción bilineal — formalización

Formalmente, en VQA buscamos predecir la respuesta más probable:

$$\hat{a} = \arg\max_{a \in \mathcal{A}} \, p_\Theta(a \mid v, q)$$

donde $\Theta$ es el conjunto de parámetros. Tras embeber imagen y pregunta en $v \in \mathbb{R}^{d_v}$ y $q \in \mathbb{R}^{d_q}$, la fusión bilineal completa produce el vector de logits $y \in \mathbb{R}^{|\mathcal{A}|}$ mediante:

$$y = (\mathcal{T} \times_1 q) \times_2 v \qquad \text{con} \quad \mathcal{T} \in \mathbb{R}^{d_q \times d_v \times |\mathcal{A}|}$$

Componente a componente, el logit de la respuesta $k$ es:

$$y[k] = \sum_{i=1}^{d_q} \sum_{j=1}^{d_v} \mathcal{T}[i, j, k] \, q[i] \, v[j]$$

Esto es una **forma bilineal** distinta para cada respuesta $k$: la matriz $\mathcal{T}[:, :, k] \in \mathbb{R}^{d_q \times d_v}$ pondera todos los pares $(q[i], v[j])$. Es la fusión más expresiva posible a segundo orden: ninguna correlación entre una dimensión de la pregunta y una de la imagen queda fuera.

El **operador modo-$n$** $\times_n$ contrae un tensor con un vector (o matriz) a lo largo del eje $n$. Si $\mathcal{X} \in \mathbb{R}^{I_1 \times \cdots \times I_N}$ y $u \in \mathbb{R}^{I_n}$, entonces $(\mathcal{X} \times_n u)$ suma sobre el índice $n$, reduciendo el orden del tensor en uno. Aquí $\mathcal{T} \times_1 q$ contrae la dimensión de la pregunta (queda un tensor $\mathbb{R}^{d_v \times |\mathcal{A}|}$), y $\times_2 v$ contrae la de la imagen (queda el vector $y \in \mathbb{R}^{|\mathcal{A}|}$).

El problema, ya señalado, es el tamaño: $|\Theta| = d_q \cdot d_v \cdot |\mathcal{A}| \approx 10^{10}$. Imposible. La pregunta de diseño es: ¿cómo aproximar $\mathcal{T}$ con un número manejable de parámetros sin perder la capacidad de modelar interacciones ricas?

---

## 4. Descomposición de Tucker — la idea central

La respuesta de MUTAN es la **descomposición de Tucker** (Tucker, 1966; ver Kolda & Bader 2009 para el tratamiento moderno). Tucker expresa un tensor de 3 modos como un **tensor núcleo (core)** más pequeño, multiplicado modo a modo por tres **matrices de factores**, una por cada eje:

$$\mathcal{T} = \big( (\mathcal{T}_c \times_1 W_q) \times_2 W_v \big) \times_3 W_o$$

con las dimensiones:

$$W_q \in \mathbb{R}^{d_q \times t_q}, \quad W_v \in \mathbb{R}^{d_v \times t_v}, \quad W_o \in \mathbb{R}^{|\mathcal{A}| \times t_o}, \quad \mathcal{T}_c \in \mathbb{R}^{t_q \times t_v \times t_o}$$

Se resume habitualmente con la notación de Tucker:

$$\mathcal{T} = [\![\, \mathcal{T}_c \,;\, W_q,\, W_v,\, W_o \,]\!]$$

Componente a componente, cada peso del tensor original se reconstruye como:

$$\mathcal{T}[i, j, k] = \sum_{l=1}^{t_q} \sum_{m=1}^{t_v} \sum_{n=1}^{t_o} \mathcal{T}_c[l, m, n] \, W_q[i, l] \, W_v[j, m] \, W_o[k, n]$$

para todo $i \in [1, d_q]$, $j \in [1, d_v]$, $k \in [1, d_o]$ (con $d_o = |\mathcal{A}|$).

La interpretación geométrica: $W_q$, $W_v$ y $W_o$ son **proyecciones** que llevan cada modo (pregunta, imagen, salida) desde su dimensión original ($d_q, d_v, |\mathcal{A}|$) a una dimensión latente más pequeña ($t_q, t_v, t_o$). El tensor núcleo $\mathcal{T}_c$ vive enteramente en ese **espacio latente comprimido** y codifica las interacciones entre las versiones proyectadas de las modalidades.

**Control de complejidad.** El número de parámetros pasa de $d_q d_v |\mathcal{A}|$ (intratable) a:

$$\underbrace{d_q t_q}_{W_q} + \underbrace{d_v t_v}_{W_v} + \underbrace{|\mathcal{A}| t_o}_{W_o} + \underbrace{t_q t_v t_o}_{\mathcal{T}_c}$$

Las tres dimensiones latentes $t_q, t_v, t_o$ son **perillas independientes**: cada una gobierna cuánta complejidad se permite a cada modalidad. La intuición clave de los autores es que el espacio de imagen y el de lenguaje son distintos y pueden requerir niveles de complejidad diferentes —por eso se permite $t_q \neq t_v$, algo que MLB prohíbe (obliga $t_q = t_v = t_o = R$). El término dominante en costo es el core $t_q t_v t_o$, que crece cúbicamente; controlarlo motiva la segunda contribución (la sparsity de rango, sección 5).

---

## 5. La estructura de MUTAN — encadenamiento del modelo

Sustituyendo la descomposición de Tucker (sección 4) en la fusión bilineal completa (sección 3), MUTAN reescribe el cómputo de forma que las proyecciones se aplican **antes** de la interacción:

$$y = \Big( \big( \mathcal{T}_c \times_1 (q^\top W_q) \big) \times_2 (v^\top W_v) \Big) \times_3 W_o$$

Esto es **estrictamente equivalente** a codificar una interacción bilineal completa entre las proyecciones de $q$ y $v$. Definiendo las **proyecciones monomodales latentes**:

$$\tilde{q} = q^\top W_q \in \mathbb{R}^{t_q}, \qquad \tilde{v} = v^\top W_v \in \mathbb{R}^{t_v}$$

el modelo se separa en tres etapas conceptuales limpias:

1. **Proyección de cada modalidad** a su espacio latente vía $W_q$ y $W_v$. En la práctica MUTAN añade no linealidades: $\tilde{q} = \tanh(q^\top W_q)$ y $\tilde{v} = \tanh(v^\top W_v)$, igual que MLB.

2. **Interacción vía el core tensor** $\mathcal{T}_c$, que produce el vector de pares latentes:

$$z = (\mathcal{T}_c \times_1 \tilde{q}) \times_2 \tilde{v} \in \mathbb{R}^{t_o}$$

Aquí $z$ es la representación fusionada del par (imagen, pregunta) en el espacio latente de salida. El tensor $\mathcal{T}_c$ aprende **qué proyecciones de $\tilde{q}$ interactúan con qué proyecciones de $\tilde{v}$**, y la dimensión $t_o$ controla la complejidad de esa interacción.

3. **Proyección a salida** vía $W_o$:

$$y = z^\top W_o \in \mathbb{R}^{|\mathcal{A}|}, \qquad p = \mathrm{softmax}(y)$$

$W_o$ puntúa el par fusionado $z$ contra cada clase de respuesta.

Cada uno de los cuatro componentes tiene un rol interpretable: $W_q$ y $W_v$ definen *qué tan compleja* es la representación de cada modalidad ($t_q$, $t_v$); $\mathcal{T}_c$ modela las *interacciones* entre modalidades (controlado por $t_o$); $W_o$ es el *clasificador* que mapea el par fusionado a respuestas.

### 5.1 La restricción de rango sobre el core (parámetro $R$)

El core $\mathcal{T}_c$ sigue costando $t_q t_v t_o$ parámetros. Para equilibrar mejor expresividad y complejidad, MUTAN impone una **restricción de sparsity estructurada** sobre las **slices** (rebanadas) del core. Cada elemento de salida $z[k]$ es una forma bilineal sobre la slice $\mathcal{T}_c[:, :, k]$:

$$z[k] = \tilde{q}^\top \, \mathcal{T}_c[:, :, k] \, \tilde{v}$$

La restricción consiste en obligar a que **cada slice tenga rango a lo más $R$**, expresándola como una suma de $R$ matrices de rango 1:

$$\mathcal{T}_c[:, :, k] = \sum_{r=1}^{R} m_r^k \otimes n_r^{k\top}, \qquad m_r^k \in \mathbb{R}^{t_q}, \; n_r^k \in \mathbb{R}^{t_v}$$

Sustituyendo, el elemento $z[k]$ se reescribe como una suma de $R$ productos escalares:

$$z[k] = \sum_{r=1}^{R} \big( \tilde{q}^\top m_r^k \big) \big( \tilde{v}^\top n_r^k \big)$$

Reagrupando los vectores $m_r^k$ y $n_r^k$ en matrices $M_r \in \mathbb{R}^{t_q \times t_o}$ y $N_r \in \mathbb{R}^{t_v \times t_o}$ (con $M_r[:, k] = m_r^k$, $N_r[:, k] = n_r^k$), la salida completa se vuelve una **suma de $R$ vectores**:

$$z = \sum_{r=1}^{R} z_r, \qquad z_r = (\tilde{q}^\top M_r) * (\tilde{v}^\top N_r)$$

donde $*$ es el producto de Hadamard (elemento a elemento).

**Interpretación lógica.** Esta forma final tiene una lectura preciosa. Cada $z_r$ se obtiene proyectando $\tilde{q}$ y $\tilde{v}$ a un espacio común y fusionándolos con un producto Hadamard —exactamente la fusión de MLB. Pero MUTAN suma $R$ de esas fusiones. Los autores lo describen como una estructura de **compuertas lógicas**: cada componente $z_r[k]$ se comporta como un AND entre "($\tilde{q}$ es similar a $m_r^k$) Y ($\tilde{v}$ es similar a $n_r^k$)", y la suma sobre $r$ actúa como un OR:

$$z_r[k] = \big( \tilde{q} \text{ similar a } m_r^k \big) \;\text{AND}\; \big( \tilde{v} \text{ similar a } n_r^k \big)$$
$$z[k] = z_1[k] \;\text{OR}\; \cdots \;\text{OR}\; z_R[k]$$

Es decir, MUTAN modela cada interacción de salida como una **disyunción de $R$ conjunciones** entre patrones de pregunta e imagen. Esta es justamente la intuición que conectaría más tarde con la atención multi-cabeza: múltiples "modos" de interacción combinados.

---

## 6. Interpretación y unificación — relación con MCB y MLB

El aporte teórico más citado de MUTAN es que **MCB y MLB son casos particulares** de la fusión de Tucker, cada uno imponiendo restricciones distintas sobre $\{\mathcal{T}_c, W_q, W_v, W_o\}$. La Figura 3 del paper lo visualiza: en cada modelo se colorean los componentes *aprendidos* y se dejan en gris los *fijos*.

**MCB como Tucker restringido.** El tensor de pesos de MCB, $\mathcal{T}^{mcb}$, se descompone en Tucker con estructuras muy rígidas:
- $W_q^{mcb} = \mathrm{Diag}(s_q)$ y $W_v^{mcb} = \mathrm{Diag}(s_v)$ son **matrices diagonales** cuyos coeficientes no nulos toman valores en $\{-1, 1\}$, con $s_q \in \mathbb{R}^{d_q}$ y $s_v \in \mathbb{R}^{d_v}$ vectores **aleatorios muestreados al instanciar el modelo y congelados** (no se aprenden).
- El core es **disperso y fijo**: $\mathcal{T}_c^{mcb}[i, j, k] = 1$ si $h(i, j) = k$ (y 0 en otro caso), donde $h$ es una función hash $[1, d_q] \times [1, d_v] \to [1, d_o]$ muestreada aleatoriamente y congelada.
- Solo se aprende la matriz de salida $W_o$, y eso **después** de la fusión.

Como las combinaciones de dimensiones que interactúan están fijadas de antemano por el azar (vía $s_q$, $s_v$, $h$), MCB necesita una $t_o$ muy grande (típicamente 16000) para que, por fuerza bruta, alguna de esas combinaciones aleatorias sea útil para la clasificación.

**MLB como Tucker restringido.** MLB corresponde a una **descomposición canónica (CP)** de rango $R$, que es el caso de Tucker en que:
- Las tres dimensiones latentes son **iguales**: $t_q = t_v = t_o = R$.
- El core es la **identidad**: $\mathcal{T}_c^{mlb} = \mathcal{I}_R$ (tensor identidad de tercer orden).
- Las tres matrices de factores $W_q, W_v, W_o$ **sí se aprenden**.

Con el core fijado a la identidad, la interacción se reduce a un producto de Hadamard en el espacio común de dimensión $R$: MLB aprende buenas proyecciones pero no aprende *la interacción en sí*. Además, forzar $t_q = t_v$ impide modelar imagen y lenguaje con complejidades distintas, y un elemento $k$ de $\tilde{q}^{mlb}$ solo puede interactuar con el mismo elemento $k$ de $\tilde{v}^{mlb}$ —una restricción que MUTAN argumenta es innecesariamente limitante.

**MUTAN.** Aprende **los cuatro componentes** $W_q, W_v, W_o, \mathcal{T}_c$, con dimensiones latentes independientes y la interacción bilineal completa $\mathcal{T}_c$ estructurada (no eliminada) mediante la sparsity de rango $R$ sobre las slices. Es el caso general que contiene a los otros dos.

**Diferencia técnica MUTAN vs MLB en la sparsity.** Los autores subrayan dos diferencias: (1) la reducción de rango de MUTAN actúa sobre el **core tensor** $\mathcal{T}_c$, mientras que MLB restringe el rango del **tensor global** $\mathcal{T}$; (2) MUTAN no reduce el tercer modo (salida), solo los dos primeros (imagen y pregunta), de modo que los parámetros implícitos en $\mathcal{T}_c$ están **correlacionados dentro de cada slice de modo-3 pero son independientes entre slices**.

### 6.1 Tabla comparativa

| Modelo | $W_q, W_v$ | Core $\mathcal{T}_c$ | $W_o$ | Dims latentes | Interacción | Params $\Theta$ (test-dev) |
|---|---|---|---|---|---|---|
| **Concat** | — (concatenación) | — | aprendido | — | primer orden | 8.9 M |
| **MCB** | diagonales fijas $\{-1,1\}$ | disperso fijo (hash) | aprendido | $t_o \approx 16000$ | bilineal aproximada (count-sketch), params fijos | 32 M |
| **MLB** | aprendidos | identidad (fijo) | aprendido | $t_q{=}t_v{=}t_o{=}R$ | Hadamard en espacio común | 7.7 M |
| **MUTAN** | aprendidos | aprendido, low-rank $R$ por slice | aprendido | $t_q, t_v, t_o$ independientes | bilineal completa estructurada | 4.9 M |
| **MUTAN+MLB** | ambos | ambos | ambos | — | fusión tardía complementaria | 17.5 M |

Lo notable: MUTAN logra la interacción **más expresiva** (bilineal completa, aprendida) con el **menor número de parámetros** entre los modelos bilineales (4.9 M, frente a 7.7 M de MLB y 32 M de MCB). La estructura algebraica es el origen de esa eficiencia.

---

## 7. Attention con MUTAN

La fusión descrita hasta aquí es **global**: un solo vector $v$ que resume toda la imagen (el promedio de los $14 \times 14$ vectores de región del mapa de features de ResNet-152). Para la arquitectura final, MUTAN se integra en un **mecanismo de atención visual** del tipo introducido por MCB y MLB.

La idea: en vez de promediar las regiones a ciegas, se usa la propia fusión MUTAN para **puntuar la relevancia** de cada región respecto de la pregunta. Concretamente, para cada uno de los $14 \times 14 = 196$ vectores de región $v_{\text{reg}}$, se computa una fusión MUTAN con la pregunta para obtener un escalar de atención; estos escalares pasan por softmax y producen pesos que ponderan las regiones, generando un vector visual global como **suma ponderada**. Ese vector atendido se vuelve a fundir con la pregunta (de nuevo vía MUTAN) para producir los logits finales. MUTAN usa atención **multi-glimpse** (varias cabezas de atención, "glimpses"), igual que MCB y MLB.

La estructura de rango $R$ del core adquiere aquí una lectura semántica especialmente clara: el análisis cualitativo (Figura 7) muestra que, al **apagar todas las proyecciones latentes salvo una**, cada proyección se especializa en un concepto distinto necesario para responder. Por ejemplo, en "¿Dónde está la mujer?" una proyección atiende al elefante y otra a la mujer —ambas informaciones son necesarias para la respuesta "sobre el elefante". Es decir, las $R$ componentes de la descomposición de rango se reparten roles interpretables.

---

## 8. Experimentos

**Dataset VQA (Antol et al. 2015).** Construido sobre imágenes de MS-COCO. Cada imagen tiene 3 preguntas anotadas; cada pregunta es respondida por 10 anotadores, dando 10 respuestas ground-truth. El dataset tiene 248 349 pares (imagen, pregunta) de entrenamiento, 121 512 de validación y 244 302 de test. Las respuestas ground-truth solo se publican para *train* y *val*; el test se evalúa contra un servidor (con *test-dev* como subconjunto y *test-std* limitado a 5 submissions totales). El foco es la tarea **open-ended** (respuesta en lenguaje libre).

**Métrica VQA.** Una respuesta predicha se considera correcta de forma graduada según el consenso de anotadores:

$$\mathrm{Acc}(\hat{a}) = \min\left(1, \; \frac{\#\{\text{humanos que dieron } \hat{a}\}}{3}\right)$$

Si la respuesta aparece al menos 3 veces entre las 10 humanas, cuenta como acierto pleno (1.0). La métrica premia el acuerdo entre anotadores.

**Setup de MUTAN.** Imágenes redimensionadas a $448 \times 448$, ResNet-152 produce mapas de features de $14 \times 14 \times 2048$. Con atención se conserva el tiling $14 \times 14$; sin atención se promedian los 196 vectores. Preguntas codificadas con una GRU inicializada con parámetros de un modelo Skip-thoughts preentrenado. $|\mathcal{A}|$ fijado a las **2000 respuestas más frecuentes**. Optimizador **Adam** (learning rate $10^{-4}$, sin learning-rate decay), early stopping como regularizador. Las imágenes se recortan al centro tras escalar el borde menor a 448 y se normalizan con la normalización de ImageNet; las preguntas se truncan/rellenan (zero-padding con TrimZero) a longitud máxima de 26 palabras, con palabras fuera de vocabulario mapeadas a "UNK".

### 8.1 Comparación de esquemas de fusión (ablation principal)

Todos los modelos se entrenan bajo el mismo marco, **sin atención**, para aislar el efecto de la fusión. Dimensiones elegidas: para MCB $t_o = 16000$, para MLB $R = 1200$. Dos variantes de MUTAN:
- **MUTAN_noR**: descomposición de Tucker **sin** la restricción de rango, con todas las dimensiones iguales $t_q = t_v = t_o = 160$ (elegidas para igualar el número de parámetros de MUTAN, 4.9 M).
- **MUTAN** (completo): Tucker **con** sparsity de rango, $t_q = t_v = t_o = 360$ y rango $R = 10$.

### 8.2 Análisis adicionales

**Impacto del tensor pleno (plain tensor).** Para medir cuánto aportan los parámetros del core, se entrenan varios MUTAN_noR fijando $t_q = t_v = t_o = t$ con $t$ de 20 a 220, y se comparan contra un modelo donde $\mathcal{T}_c$ se reemplaza por el **tensor identidad** (equivalente a MLB sin atención). Resultado (Figura 4): MUTAN_noR supera ampliamente al tensor identidad incluso para dimensiones de core muy pequeñas, demostrando que el core **aprende correlaciones reales** entre modalidades, no redundantes con las proyecciones. Nota: para cada $t$, MUTAN_noR añade $t^3$ parámetros sobre la identidad (para $t = 220$, son 10.6 M extra).

**Impacto de la sparsity de rango.** Fijando $t_q = t_v = 210$ y variando $t_o$ para distintos rangos $R$ (Figura 5): controlar el rango de las slices permite modelar mejor las interacciones. Comparando $R = 60$ y $R = 20$, un rango **menor** permite alcanzar valores de $t_o$ más altos **sin sobreajuste**, con menos parámetros en la fusión y mayor accuracy en *val*. La sparsity de rango actúa como regularizador.

**Observaciones cualitativas.** Entrenando con $R = 20$ y luego apagando todas las proyecciones latentes $z_r$ salvo una (Figura 6), se mide la contribución de cada rango por tipo de pregunta. Aparecen tres comportamientos: (a) preguntas cuya respuesta es casi siempre "sí"/"no" ("Is there") —cada rango por sí solo alcanza casi el rendimiento global; (b) preguntas que requieren información de **todas** las proyecciones ("What is the man") —cada proyección aislada rinde mucho peor que el conjunto; (c) proyecciones **especializadas** por tipo de pregunta —la variable latente 16 rinde bien en "What room is", la 17 en "what sport is", con comportamiento opuesto entre ambas.

---

## 9. Resultados numéricos (grounded en el PDF)

### 9.1 Comparación de fusiones sin atención (Tabla 1, *test-dev* y *val*)

| Modelo | $\Theta$ (M) | Y/N | No. | Other | **All (test-dev)** | All (val) |
|---|---|---|---|---|---|---|
| Concat | 8.9 | 79.25 | 36.18 | 46.69 | 58.91 | 56.92 |
| MCB | 32 | 80.81 | 35.91 | 46.43 | 59.40 | 57.39 |
| MLB | 7.7 | **82.02** | 36.61 | 46.65 | 60.08 | 57.91 |
| MUTAN_noR | 4.9 | 81.44 | 36.42 | 46.86 | 59.92 | 57.94 |
| **MUTAN** | 4.9 | 81.45 | **37.32** | **47.17** | **60.17** | **58.16** |
| MUTAN+MLB | 17.5 | 82.29 | 37.27 | 48.23 | **61.02** | 58.76 |

Lecturas clave:
- **MUTAN_noR > MLB en complejidad equivalente.** Con dimensiones bajas (160) y mismos parámetros que MUTAN, MUTAN_noR (59.92) supera a la *baja-dimensión + Hadamard* implícita de MLB en igualdad de condiciones, validando que **modelar la interacción bilineal completa sobre proyecciones de baja dimensión es más potente** que tener proyecciones de alta dimensión con una fusión simple.
- **MUTAN > MUTAN_noR.** La sparsity de rango mejora el resultado (60.17 vs 59.92) **con el mismo número de parámetros**, confirmando que actúa como regularizador útil.
- **MUTAN es el mejor modelo individual** en *test-dev* (60.17) con **menos parámetros que todos** (4.9 M).
- **MUTAN+MLB (fusión tardía)** suma $\sim$1 punto (61.02), validando la **complementariedad** entre ambos esquemas tensoriales.

### 9.2 Comparación con el state-of-the-art (Tabla 2, con atención y ensamble)

| Modelo | Y/N | No. | Other | **All (test-dev)** | All (test-std) |
|---|---|---|---|---|---|
| SMem 2-hop | 80.87 | 37.32 | 43.12 | 57.99 | 58.24 |
| Ask Your Neurons | 78.39 | 36.45 | 46.28 | 58.39 | 58.43 |
| SAN | 79.3 | 36.6 | 46.1 | 58.7 | 58.9 |
| D-NMN | 81.1 | 38.6 | 45.5 | 59.4 | 59.4 |
| ACK | 81.01 | 38.42 | 45.23 | 59.17 | 59.44 |
| MRN | 82.28 | 38.82 | 49.25 | 61.68 | 61.84 |
| HieCoAtt | 79.7 | 38.7 | 51.7 | 61.8 | 62.1 |
| MCB (7) | 83.4 | 39.8 | 58.5 | 66.7 | 66.5 |
| MLB (7) | 84.54 | 39.21 | 57.81 | 66.77 | 66.89 |
| MUTAN (3) | 84.57 | 39.32 | 57.36 | 67.03 | 66.96 |
| **MUTAN (5)** | **85.14** | **39.81** | **58.52** | **67.42** | **67.36** |

Donde $(n)$ indica un ensamble de $n$ modelos. **MUTAN (5)** —ensamble de 3 MUTAN con atención + 2 MLB (que son casos especiales de MUTAN) entrenados con data augmentation de Visual Genome— alcanza el **state-of-the-art** de la época: **67.42 % en *test-dev* y 67.36 % en *test-std***. Notablemente, **MUTAN (3)** (solo 3 modelos, sin MLB) ya supera a MCB (7) y MLB (7), que usaban ensambles de 7 modelos. MUTAN logra mejores resultados con ensambles más pequeños.

---

## 10. Limitaciones

1. **Sigue siendo fusión de dos vectores.** A pesar de la atención multi-glimpse, MUTAN colapsa la imagen en un (o pocos) vector(es) global(es) atendido(s) antes de la fusión final. No hay razonamiento composicional ni interacción región-palabra a nivel fino sostenida; toda la riqueza vive en la forma bilineal entre dos vectores resumen.

2. **Complejidad de implementación de la descomposición.** Aunque conceptualmente elegante, manejar tres dimensiones latentes ($t_q, t_v, t_o$) más el rango $R$ implica un espacio de hiperparámetros mayor que el de fusiones simples. La forma final con $R$ matrices $M_r, N_r$ y suma de Hadamards requiere implementación cuidadosa, y la elección de $t_q, t_v, t_o, R$ depende del tipo de pregunta (como muestra el propio análisis de ablations).

3. **Interacción de segundo orden, no de orden superior.** La fusión bilineal captura correlaciones de segundo orden, pero el razonamiento sobre relaciones complejas (conteo, comparación espacial, lógica multi-paso) excede lo que una forma bilineal puede expresar de manera natural.

4. **Superado por Transformers cross-modales.** A partir de 2019 (ViLBERT, LXMERT, VisualBERT, UNITER) la **cross-attention** sustituyó la fusión bilineal: en vez de comprimir todo en una forma bilineal, se dejan interactuar libremente *todos* los tokens de texto con *todas* las regiones de imagen a través de múltiples capas de atención. La interacción bilineal de MUTAN es, en retrospectiva, una única "capa de interacción" comprimida algebraicamente; los Transformers apilan muchas, con mucha más capacidad.

5. **Costo del extractor visual.** Depende de features de ResNet-152 de $14 \times 14 \times 2048$ precomputados; la calidad de la fusión está acotada por la calidad de esas features (a diferencia de los enfoques bottom-up de Anderson et al. que usan detecciones de objetos).

---

## 11. Impacto y legado

MUTAN es, en la práctica, el **broche de la era de la fusión bilineal** en VQA (2015–2017). Su contribución perdurable no es el número de accuracy —rápidamente superado por los Transformers cross-modales— sino dos lecciones de diseño:

**Lección 1 — la estructura algebraica gobierna el trade-off expresividad/parámetros.** MUTAN demuestra que la pregunta correcta no es "¿interacción rica o pocos parámetros?" sino "¿qué *estructura* impongo al tensor de interacción?". Tucker + restricción de rango permite tener la interacción más rica con los menos parámetros. Esta idea —factorizar una operación cara con descomposiciones tensoriales de bajo rango— reaparece por todas partes en deep learning moderno: LoRA (adaptación de rango bajo en LLMs), atención lineal, factorización de matrices de proyección en Transformers.

**Lección 2 — el marco unificador.** Al mostrar que MCB y MLB son casos particulares de Tucker con distintas restricciones, MUTAN ofrece un **lenguaje común** para razonar sobre familias enteras de modelos de fusión. Esto es valioso pedagógicamente: en vez de memorizar tres arquitecturas, se entiende un marco (Tucker) y tres puntos en su espacio de restricciones.

El propio grupo (Cadene, Ben-younes, Cord, Thome) continuó esta línea con **BLOCK** (AAAI 2019, descomposición block-superdiagonal, que generaliza aún más) y **MUREL**, antes de que el campo migrara masivamente a la pre-entrenamiento + cross-attention. La librería `vqa.pytorch` de Cadene fue una referencia de implementación importante en la comunidad.

La **transición a cross-attention** puede leerse como el reconocimiento de que, en lugar de comprimir la interacción multimodal en una sola forma bilineal estructurada, conviene apilar muchas capas de atención que dejan interactuar todos los elementos —pagando el costo computacional con la ganancia de capacidad que permite el pre-entrenamiento a gran escala.

---

## 12. Conexión con la clase 23

La clase 23 (VQA) presenta el **espectro de la fusión multimodal**, y MUTAN ocupa uno de sus extremos. El eje conceptual es: **¿cuánta maquinaria algebraica dedicas a combinar imagen y texto?**

- **Extremo simple (eficiente):** la fusión por **producto punto / Hadamard** o concatenación, que es lo que usa **Pythia** (el modelo VQA eficiente que la clase usa como caballo de batalla práctico, descendiente de bottom-up attention de Anderson et al.). La fusión simple es barata, rápida de entrenar y escala bien con buenos features de objetos (bottom-up).

- **Extremo expresivo (algebraico):** **MCB** (aproximación bilineal por count-sketch + FFT) y **MUTAN** (interacción bilineal completa vía Tucker + rango bajo). Maximizan la expresividad de la fusión a costa de complejidad.

La lección de la clase es que, dada la calidad de los features de objetos modernos (bottom-up attention), la fusión simple **basta** para gran parte del rendimiento, y por eso Pythia y los sistemas de producción la prefieren por eficiencia. MUTAN muestra **el otro extremo**: cuánto puede exprimirse de la fusión cuando se modela la interacción bilineal completa de forma algebraicamente cuidada. Juntos, MCB y MUTAN delimitan el techo de lo que la fusión pura (sin cross-attention profunda) puede lograr, y explican por qué el campo terminó migrando a Transformers cross-modales cuando se buscó superar ese techo.

Para Roberto: el paralelo de ingeniería es el clásico trade-off **"índice/estructura sofisticada vs. operación simple sobre datos bien preparados"**. MUTAN invierte en la estructura del operador de fusión; Pythia invierte en la calidad de la entrada (features de objetos) y usa un operador trivial. Ambas son válidas; la elección depende de dónde está el cuello de botella.

---

## 13. Notas y enlaces

- **Relación con MCB** (Fukui et al. 2016, mismo curso, clase 23): MCB es el predecesor directo; MUTAN demuestra que MCB es un caso particular de Tucker con factores diagonales fijos y core hash disperso. Comparar las dos arquitecturas es el ejercicio canónico de la unidad de fusión bilineal.
- **Relación con MLB** (Kim et al. 2017): MLB es el otro predecesor; MUTAN demuestra que MLB es Tucker con core identidad y dimensiones latentes iguales. El ensamble MUTAN+MLB y MUTAN(5) explotan su complementariedad.
- **Pythia / bottom-up attention** (Anderson et al. 2018; Jiang et al. 2018): el contrapunto eficiente de la clase. Usa features de objetos de un detector (Faster R-CNN) y fusión simple. Muestra que la inversión en buenos features puede sustituir la inversión en fusión sofisticada.
- **Descomposición de Tucker**: referencia matemática en Tucker (1966) y, para el tratamiento moderno de descomposiciones tensoriales, Kolda & Bader, *Tensor Decompositions and Applications*, SIAM Review 2009.
- **Código**: implementación oficial en PyTorch, https://github.com/cadene/vqa.pytorch
- **Trabajo posterior del grupo**: BLOCK (Ben-younes et al., AAAI 2019) generaliza MUTAN con una descomposición block-superdiagonal; es la continuación natural de la línea.
- **Detalles reproducibles**: Adam con lr $10^{-4}$ sin decay; batch 512 sin atención, 100 con atención; early stopping; mejores épocas asociadas a accuracies de entrenamiento de 63–70 %.
