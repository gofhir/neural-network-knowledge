---
título: "Pythia v0.1: the Winning Entry to the VQA Challenge 2018"
autores: "Yu Jiang*, Vivek Natarajan*, Xinlei Chen*, Marcus Rohrbach, Dhruv Batra, Devi Parikh"
afiliación: "Facebook AI Research (FAIR) — equipo A-STAR"
venue: "Tech report del equipo ganador del VQA Challenge 2018 (arXiv)"
año: 2018
arxiv: "1807.09956"
arxiv_version: "v2 (27 Jul 2018)"
link: "https://arxiv.org/abs/1807.09956"
código: "https://github.com/facebookresearch/pythia"
clase: "Clase 23 — Visual Question Answering"
rol_en_clase: "Modelo central (slides 9-19); el diagrama de arquitectura es la slide 13"
---

# Pythia v0.1: la Entrada Ganadora del VQA Challenge 2018

> "Chaerephon: *Pythia — Is there any man alive wiser than Socrates?*
> Pythia: *None.*"
> — Epígrafe del paper. El nombre *Pythia* es un homenaje al oráculo de Apolo en Delfos, que respondía preguntas en la Antigua Grecia. Es un guiño perfecto: un sistema que responde preguntas.

## 1. Ficha bibliográfica y resumen ejecutivo

- **Título:** *Pythia v0.1: the Winning Entry to the VQA Challenge 2018*.
- **Autores:** Yu Jiang\*, Vivek Natarajan\*, Xinlei Chen\* (los tres con contribución igual), Marcus Rohrbach, Dhruv Batra, Devi Parikh.
- **Afiliación:** Facebook AI Research (FAIR), equipo A-STAR ("Agents that See, Talk, Act, and Reason").
- **Venue:** Tech report (3 páginas) del equipo ganador del *VQA Challenge 2018*, publicado en arXiv.
- **arXiv:** 1807.09956 (versión v2, 27 de julio de 2018).
- **Código:** `https://github.com/facebookresearch/pythia` (entrenamiento, evaluación, *data augmentation*, *ensembling* y modelos preentrenados liberados).

Pythia v0.1 es, en palabras del propio reporte, una **reimplementación modular del modelo *bottom-up top-down* (up-down)** de Anderson et al. (2018). La tesis central del documento es contraintuitivamente humilde: **no introduce una arquitectura nueva**, sino que demuestra que un conjunto de cambios "sutiles pero importantes" sobre el modelo up-down —cambios en la arquitectura, en el *learning rate schedule*, en el *fine-tuning* de las características de imagen, y la adición de *data augmentation*— elevan la *accuracy* en VQA v2.0 **de 65.67% a 70.24%** en el conjunto *test-std*. Con un *ensemble* diverso de 30 modelos se alcanza **72.27%** en *test-std*, la cifra que ganó el desafío (la entrada oficial del equipo A-STAR fue 72.25%; el reporte describe el código liberado que llega a 72.27%).

La importancia pedagógica de este paper es doble. Primero, es un **manual de ingeniería de VQA**: muestra qué decisiones de implementación realmente mueven la aguja en un sistema VQA de la era pre-Transformers. Segundo, su columna vertebral —el modelo up-down con *bottom-up attention*— es la arquitectura canónica que la Clase 23 usa para explicar cómo funciona un sistema VQA "clásico". El diagrama de la slide 13 *es* la arquitectura de Pythia.

> **Aviso metodológico.** El reporte de Pythia es deliberadamente conciso (3 páginas) y **asume conocido el modelo BUTD de Anderson et al. (2018)**. Por eso, las secciones 3 y 4 de este análisis reconstruyen BUTD con detalle a partir del paper base, y señalan explícitamente qué es aporte de BUTD y qué es contribución original de Pythia. Donde el texto diga "el reporte de Pythia dice...", la afirmación está literalmente en el PDF; donde diga "en BUTD...", proviene del paper base que Pythia extiende.

## 2. Contexto — el VQA Challenge y la línea base BUTD

### 2.1. ¿Qué es VQA y qué es ganar el desafío?

**Visual Question Answering (VQA)** es la tarea de responder, en lenguaje natural, una pregunta abierta sobre una imagen. El *benchmark* dominante es **VQA v2.0** (Goyal et al., 2017, referencia [6] del paper), una versión "balanceada" del dataset original de Antol et al. (2015). El balanceo es clave: para cada pregunta existen pares de imágenes que dan respuestas distintas, lo que **fuerza al modelo a mirar la imagen** y no a explotar atajos puramente lingüísticos (los famosos *language priors*, p. ej. responder "tennis" a "what sport...?" sin mirar nada).

El **VQA Challenge** es la competencia anual asociada (se celebra junto a CVPR). La métrica oficial es una *accuracy* suavizada: cada pregunta tiene 10 respuestas humanas, y una respuesta del modelo recibe

$$\text{Acc}(a) = \min\left(\frac{\#\,\text{humanos que dieron } a}{3},\ 1\right),$$

es decir, se obtiene crédito completo si al menos 3 de 10 anotadores coincidieron con la respuesta del modelo. Esto convierte VQA en un problema de **clasificación multi-etiqueta con etiquetas blandas** (*soft scores*), un detalle que reaparece en la función de pérdida (Sección 7). "Ganar el desafío" significa lograr la *accuracy* más alta en el conjunto privado *test-std* sobre las cerca de 65 categorías de tipos de pregunta.

### 2.2. La línea base BUTD (Anderson et al., 2018)

El punto de partida de Pythia es el modelo **bottom-up top-down (up-down)** de Peter Anderson et al., *"Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering"* (CVPR 2018, referencia [1]). Este modelo había sido la base de la **entrada ganadora del VQA Challenge 2017**, alcanzando 65.32% en *test-dev* y 65.67% en *test-std* (fila "up-down" de la Tabla 1 del paper).

La idea central de up-down —y la razón por la que la Clase 23 lo llama "modelo up-down"— es combinar **dos clases de atención visual**:

- **Atención *bottom-up* (de abajo hacia arriba, *data-driven*):** un detector de objetos —Faster R-CNN (Ren et al., 2015, ref. [12]) preentrenado en Visual Genome (ref. [9])— propone un conjunto de **K regiones salientes** de la imagen, cada una representada por un vector de **2048 dimensiones**. Esta es la atención "natural" del sistema visual humano: ciertas regiones (objetos, partes) saltan a la vista independientemente de la tarea.
- **Atención *top-down* (de arriba hacia abajo, *task-driven*):** la pregunta guía *cuáles* de esas K regiones merecen atención. Es la atención "voluntaria", modulada por el objetivo cognitivo actual (la pregunta).

Antes de BUTD, la práctica estándar era aplicar atención sobre una grilla uniforme de características convolucionales (p. ej. un mapa $14\times14\times2048$ de un ResNet). La innovación de BUTD fue **reemplazar esa grilla por las regiones propuestas por un detector**: en lugar de atender a celdas arbitrarias, el modelo atiende a objetos semánticamente coherentes. Esto mejoró sustancialmente VQA y *image captioning* simultáneamente.

## 3. Arquitectura base BUTD en detalle

Reconstruyo aquí el modelo up-down completo, porque es el andamiaje sobre el que se montan todas las mejoras de Pythia.

### 3.1. Bottom-up attention: el detector como extractor de características

El componente *bottom-up* es un **Faster R-CNN con backbone ResNet-101**, preentrenado en Visual Genome con anotaciones tanto de objetos como de atributos. Su salida no son cajas para mostrar al usuario, sino **características de región**:

1. La red propone regiones candidatas (RPN) y se quedan las K mejores (en BUTD, K es adaptativo entre 10 y 100 según un umbral de confianza).
2. Para cada región $i$, se extrae un vector $v_i \in \mathbb{R}^{2048}$ por *average pooling* del mapa de características espacial de esa región. El reporte de Pythia precisa el detalle: en up-down, "cada región queda representada por la característica de 2048-D tras el *average pooling* de una grilla $7\times7$" (esto es la salida del bloque Res-5 usado como clasificador de región de segunda etapa).

El resultado es una matriz de características visuales

$$V = [v_1, v_2, \dots, v_K], \qquad V \in \mathbb{R}^{K \times 2048}.$$

Esta es exactamente la caja "imagen → Mask R-CNN/ResNet101 → K×2048 features de regiones" del diagrama de la clase. (En BUTD original el detector es Faster R-CNN/ResNet-101; Pythia migra a Detectron/FPN, ver Sección 5.)

### 3.2. Codificación de la pregunta

La pregunta se tokeniza (en el diagrama de la clase, hasta **14 tokens** para "What is this?"), cada token se mapea a un *embedding* **GloVe de 300 dimensiones** (Pennington et al., 2014, ref. [11]) —matriz $14 \times 300$ en el diagrama— y la secuencia se procesa con una **GRU** que produce un vector de pregunta $q \in \mathbb{R}^{512}$. (BUTD usa una GRU estándar; Pythia añade además una atención sobre los tokens de la pregunta, ver Sección 4a.)

### 3.3. Top-down attention: la pregunta pesa las regiones

Dado el vector de pregunta $q$ y las K características de región $\{v_i\}$, la atención *top-down* calcula un escalar de relevancia $a_i$ por región. En la formulación de BUTD:

$$a_i = w_a^{\top}\, f_a\!\left(W_v v_i \,\circ\, W_q q\right),$$

donde $f_a$ es la no linealidad (en BUTD, *gated tanh*; en Pythia, *weight normalization* + ReLU). Los escalares se normalizan con **softmax sobre las K regiones**:

$$\alpha_i = \frac{\exp(a_i)}{\sum_{j=1}^{K}\exp(a_j)}, \qquad \sum_{i=1}^{K}\alpha_i = 1.$$

La imagen atendida es la **suma ponderada** de las características de región:

$$\hat{v} = \sum_{i=1}^{K}\alpha_i\, v_i, \qquad \hat{v} \in \mathbb{R}^{2048}.$$

Este $\hat{v}$ de 2048-D se proyecta luego a 512-D mediante una capa lineal $W$. Esto corresponde exactamente al bloque del diagrama de la clase: "combina GRU 512 + features → W → W → Softmax (K pesos) → weighted sum (2048) → W → 512".

### 3.4. Fusión multimodal y clasificador

Una vez que se tiene la representación visual atendida proyectada a 512-D y el vector de pregunta de 512-D, se **fusionan** (en BUTD, por producto de Hadamard tras proyecciones), y el vector fusionado pasa por un **clasificador multi-etiqueta con activación sigmoide** que produce un *score* por cada respuesta candidata del vocabulario fijo de respuestas. Esto es el final del diagrama: "fusión por dot-product de los dos 512 → W → W → Sigmoid → scores de respuesta".

## 4. La arquitectura de Pythia paso a paso

Ahora recorro el grafo exactamente como lo dibuja la slide 13 de la clase, anotando dimensiones. Pythia conserva el esqueleto de BUTD y refina cada bloque.

**(a) Embedding de la pregunta — GloVe + GRU + atención sobre tokens.**
La pregunta de hasta 14 tokens se embebe con GloVe 300-D (matriz $14\times300$) y se procesa con una **GRU** que entrega un estado de **512 dimensiones**. El reporte de Pythia añade explícitamente un **módulo de atención sobre la pregunta** (cita la *question attention* de Yu et al., ref. [16]) para extraer características textuales atentivas, en lugar de tomar solo el último estado de la GRU. Salida: $q \in \mathbb{R}^{512}$.

**(b) Características de imagen — K×2048 vía detector.**
La imagen pasa por el detector (en Pythia, Detectron con FPN; ver Sección 5) y produce $V \in \mathbb{R}^{K \times 2048}$, una característica de 2048-D por cada una de las K regiones. En la mejor configuración, $K=100$ cajas fijas por imagen.

**(c) Top-down attention — softmax sobre K regiones + weighted sum.**
Combina $q$ (512) con cada $v_i$ (2048), calcula los logits de atención, aplica **softmax sobre las K regiones** para obtener pesos $\alpha_i$, y produce la suma ponderada $\hat{v}=\sum_i \alpha_i v_i \in \mathbb{R}^{2048}$, que se proyecta a 512-D. Ecuaciones idénticas a las de la Sección 3.3, pero con ReLU + *weight normalization* en lugar de *gated tanh*.

**(d) Fusión multimodal — producto de Hadamard.**
Las dos representaciones de 512-D (texto y visión atendida) se combinan por **multiplicación elemento a elemento (producto de Hadamard)**:

$$h = \big(W_q' q\big) \,\circ\, \big(W_v' \hat{v}\big), \qquad h \in \mathbb{R}^{d_h}.$$

El reporte señala que el mejor tamaño de capa oculta es $d_h = 5000$ ("we found the best-performing hidden size to be 5000"). El símbolo $\circ$ denota el producto componente a componente. Ver Sección 6 para por qué esta operación es tan efectiva.

**(e) Clasificador sigmoide — multi-etiqueta sobre respuestas.**
El vector fusionado pasa por capas lineales y una **activación sigmoide** que emite un *score* independiente en $[0,1]$ por cada respuesta del vocabulario:

$$\hat{y} = \sigma\!\big(W_2\, g(W_1 h)\big) \in [0,1]^{|\mathcal{A}|},$$

donde $|\mathcal{A}|$ es el tamaño del vocabulario de respuestas (típicamente las ~3129 respuestas más frecuentes en VQA v2.0). Que sea sigmoide y no softmax es deliberado: VQA es clasificación **multi-etiqueta** porque varias respuestas pueden ser parcialmente correctas (Sección 7).

## 5. Las mejoras de Pythia sobre BUTD

El corazón del paper es esta secuencia de cambios incrementales. La Tabla 1 los presenta como una escalera, cada peldaño sumando *accuracy*. Reproduzco la tabla y luego explico cada mejora.

| Modelo | test-dev | test-std |
|---|---|---|
| up-down [1] (BUTD, baseline 2017) | 65.32 | 65.67 |
| + Adaptación del modelo up-down (§2.1) | 66.91 | — |
| + Learning Schedule (§2.2) | 68.05 | — |
| + Detectron & Fine-tuning (§2.3) | 68.49 | — |
| + Data Augmentation* (§2.4) | 69.24 | — |
| + Grid Feature* (§2.5) | 69.81 | — |
| + 100 bboxes* (§2.5) | 70.01 | 70.24 |
| Ensemble, 30× mismo modelo (§2.6) | 70.96 | — |
| Ensemble, 30× modelos diversos (§2.6) | 72.18 | **72.27** |

(El asterisco \* marca configuraciones que **no** se incluyeron en el *ensemble* enviado al desafío.)

### 5.1. Adaptación de la arquitectura: weight normalization + ReLU + Hadamard (65.32 → 66.91)

Tres cambios agrupados, la mejora individual más grande del *single model* (+1.59):

- **Weight normalization + ReLU en vez de *gated tanh*.** BUTD usaba *gated hyperbolic tangent* (una activación con compuerta, costosa). Pythia la reemplaza por **weight normalization** (Salimans & Kingma, 2016, ref. [13]) seguida de **ReLU**, "para reducir el cómputo". *Weight normalization* reparametriza cada peso como $w = g\,\frac{v}{\lVert v\rVert}$, separando magnitud y dirección, lo que acelera y estabiliza el entrenamiento sin el costo de *batch normalization*.
- **Fusión por producto de Hadamard (elemento a elemento) en vez de concatenación.** El reporte dice: "replaced feature concatenation with element-wise multiplication to combine the features from text and visual modalities". Esto es central (Sección 6).
- **GloVe 300-D + atención sobre la pregunta + hidden size 5000.** Inicialización de los *embeddings* con vectores GloVe de 300-D, módulo de atención textual (ref. [16]), y la capa oculta de fusión dimensionada a 5000.

### 5.2. Learning rate schedule con *warmup* (66.91 → 68.05)

Pythia optimiza con **Adamax** (variante de Adam con norma infinita, Kingma & Ba, ref. [8]). El equipo observó que **reducir el *batch size* mejoraba el desempeño**, lo que sugería que el *learning rate* era demasiado bajo; pero **subir el learning rate ingenuamente causaba divergencia**. La solución fue importar la estrategia de *warmup* de Goyal et al. ("Accurate, Large Minibatch SGD", ref. [5], el famoso "ImageNet en 1 hora"):

- Empezar con learning rate **0.002**, subiéndolo linealmente cada iteración hasta **0.01** en la iteración **1000** (fase de *warmup*).
- Luego reducirlo por un factor de **0.1** en la iteración **5K**, y otra vez cada **2K** iteraciones.
- Detener el entrenamiento en **12K** iteraciones.

Esta sola mejora del *schedule* aportó +1.14 (66.91 → 68.05), un recordatorio de que el calendario de aprendizaje es tan importante como la arquitectura.

### 5.3. Detectron + fine-tuning de las características bottom-up (68.05 → 68.49)

Pythia cambia el detector. En lugar del Faster R-CNN/ResNet-101 de BUTD, usa **detectores Detectron basados en Feature Pyramid Networks (FPN)** (Lin et al., ref. [10]) con backbone **ResNeXt** (ref. [15]) y dos capas *fully connected* (fc6 y fc7) para clasificar regiones. La ventaja práctica: las características de 2048-D salen de **fc6** y se hace **fine-tuning solo de fc7**, en lugar de afinar capas convolucionales sobre mapas $7\times7\times2048$ (mucho más costoso en almacenamiento y cómputo, como en up-down original). Igual que BUTD, el detector se entrena en Visual Genome con anotaciones de objetos y atributos. El *fine-tune learning rate* se fija en **0.1 veces** el learning rate global. Resultado: 68.49% en test-dev.

### 5.4. Data augmentation con Visual Genome y Visual Dialog (68.49 → 69.24)

Pythia agranda los datos por dos vías:

- **Datos adicionales de Visual Genome (VG, ref. [9]) y Visual Dialog (VisDial v0.9, ref. [3]).** Para VisDial, convirtieron los 10 turnos de cada diálogo en 10 pares pregunta-respuesta independientes. Como VG y VisDial traen una sola respuesta *ground-truth* (mientras VQA trae 10), simplemente **replicaron la respuesta 10 veces** para que el formato fuera compatible con el protocolo de evaluación de VQA.
- **Mirroring (espejado) de imágenes de VQA.** Al voltear horizontalmente la imagen, se procesan preguntas y respuestas intercambiando los tokens **"left" ↔ "right"** (un *flip* ingenuo confundiría la izquierda con la derecha). Solo se hace en las muestras que contienen esas palabras.

Al añadir estos datos se baja el learning rate antes (a 15K y 22K iteraciones) y se entrena hasta 22K. Ganancia: +0.75 (→ 69.24).

### 5.5. Grid features + 100 bounding boxes (mejoras post-desafío) (69.24 → 70.01 → 70.24)

Esta sección (§2.5) son **mejoras posteriores al desafío** (post-challenge), no incluidas en la entrada oficial:

- **Grid features.** La hipótesis: las características *bottom-up* (solo regiones de objetos propuestas) **no capturan información espacial holística** ni las zonas de la imagen no cubiertas por ninguna propuesta (cielo, fondo, texturas). Para remediarlo, Pythia combina las características de regiones con **características de grilla** (la atención sobre el mapa convolucional uniforme, al estilo clásico, extraídas de ResNet-152, refs. [4] y [7]). Las características a nivel de objeto y a nivel de grilla se fusionan **por separado** con la pregunta y luego se **concatenan** antes del clasificador. Aporte: +0.57 (→ 69.81).
- **100 bounding boxes fijos.** En lugar del protocolo adaptativo de up-down (entre 10 y 100 propuestas por imagen según confianza), Pythia prueba una estrategia más simple (aunque más lenta): usar **100 propuestas fijas** para *todas* las imágenes. Con 100 cajas se llega a 70.01% en test-dev y **70.24% en test-std**. Aporte: +0.20.

### 5.6. Ensembling (70.96 / 72.27)

Dos estrategias de *ensemble*, todas con modelos entrenados *antes* de la fecha límite del desafío (no incluyen las mejoras post-challenge de §2.5):

- ***Ensemble* ingenuo: 30× el mismo modelo con distintas semillas.** Promediar predicciones de 30 redes idénticas entrenadas con *random seeds* distintos. El desempeño **se aplana en 70.96%** (Figura 1): añadir más copias del mismo modelo da rendimientos decrecientes rápidos.
- ***Ensemble* diverso: 30 modelos diferentes.** Mezclar modelos entrenados con configuraciones distintas: el up-down con/sin *data augmentation*, y modelos con características extraídas de distintos modelos Detectron con/sin *data augmentation*. Esta estrategia es **mucho más efectiva**: con 30 modelos diversos se llega a **72.18% test-dev y 72.27% test-std**, una mejora de **+1.31** sobre el *ensemble* ingenuo. La lección: en *ensembling*, la **diversidad** importa más que la cantidad.

## 6. La fusión multimodal: por qué Hadamard

El reporte de Pythia hace explícito que reemplazó la **concatenación** de características por la **multiplicación elemento a elemento (producto de Hadamard)**. La slide 13 de la clase resume la intuición: el producto punto/Hadamard "mezcla información multimodal sin aumentar la dimensión del modelo".

¿Por qué? Comparemos las dos opciones para fusionar un vector de texto $q'\in\mathbb{R}^{d}$ y uno de visión $\hat{v}'\in\mathbb{R}^{d}$:

- **Concatenación:** $[q'; \hat{v}'] \in \mathbb{R}^{2d}$. Duplica la dimensión; la capa siguiente necesita el doble de parámetros. Y, sobre todo, la concatenación seguida de una capa lineal solo permite **interacciones aditivas** entre modalidades ($W_1 q' + W_2 \hat{v}'$): el modelo ve texto y visión "sumados", no "cruzados".
- **Producto de Hadamard:** $q' \circ \hat{v}' \in \mathbb{R}^{d}$. Mantiene la dimensión $d$ (no la aumenta), y cada componente $i$ es $q'_i\cdot\hat{v}'_i$, una **interacción multiplicativa** entre la dimensión $i$ del texto y la dimensión $i$ de la visión. Multiplicar es la forma más barata de "gating" cruzado: la activación textual modula (abre/cierra) la activación visual y viceversa.

En términos de capacidad de representación, la multiplicación captura **correlaciones de segundo orden** entre las modalidades que la suma no puede. Es una versión barata y diagonal de las técnicas de *bilinear pooling* (como MCB, ref. [4], o el *factorized high-order pooling* de ref. [16]), que modelan la interacción completa $q'^{\top} W \hat{v}'$ con un tensor $W$ enorme. Hadamard equivale a forzar $W$ a ser diagonal: mucho menos parámetros, casi todo el beneficio. Por eso "mezcla información multimodal sin aumentar la dimensión del modelo".

## 7. Entrenamiento

Aunque el reporte es breve en este punto, los detalles de entrenamiento se infieren del paper y del marco BUTD:

- **Función de pérdida: binary cross-entropy multi-etiqueta sobre *soft scores*.** Como VQA v2.0 da 10 respuestas humanas por pregunta, la etiqueta de cada respuesta candidata $a$ es un *score* blando $s_a = \min(\#\text{votos}/3,\ 1)\in[0,1]$. El clasificador sigmoide emite $\hat{y}_a$ por respuesta, y se minimiza la **entropía cruzada binaria** sumada sobre el vocabulario:

$$\mathcal{L} = -\sum_{a\in\mathcal{A}} \Big[\, s_a \log \hat{y}_a + (1-s_a)\log(1-\hat{y}_a)\,\Big].$$

La sigmoide (en lugar de softmax) permite que **varias respuestas reciban crédito parcial** simultáneamente, alineándose con la naturaleza multi-anotador del dataset. Esta elección —tomada de Teney et al. (ref. [14], "Tips and Tricks for VQA")— es uno de los trucos clave de la era up-down.

- **Optimizador:** Adamax (ref. [8]), con el *schedule* de *warmup* + *step decay* descrito en §5.2.
- **Datos:** VQA v2.0 (ref. [6]) como base, aumentado con Visual Genome (ref. [9]) y Visual Dialog v0.9 (ref. [3]), más *mirroring* de imágenes VQA (§5.4). El detector se preentrena en Visual Genome con objetos y atributos.

## 8. Resultados

La cifra titular: **72.27% en test-std de VQA v2.0** (la entrada oficial A-STAR fue 72.25%), suficiente para **ganar el VQA Challenge 2018**. La progresión completa está en la Tabla 1 de la Sección 5.

Lecturas clave de los resultados:

- **El *single model* pasó de 65.32% a 70.01%** en test-dev (+4.69 puntos) **solo con ingeniería incremental**, sin arquitectura nueva. Esto subraya el mensaje del paper: la diferencia entre un sistema VQA competente y uno ganador estaba, en 2018, en los detalles de implementación.
- **El mayor salto individual** vino de la adaptación de arquitectura (+1.59: weight norm + ReLU + Hadamard + GloVe), seguido del *learning schedule* (+1.14).
- **El *ensemble* aportó otros ~2 puntos** (70.01 → 72.27), pero solo cuando fue **diverso**; el *ensemble* de copias idénticas se estanca en 70.96% (Figura 1).
- Frente al **segundo lugar** del desafío, el margen fue estrecho (las entradas top de 2018 rondaban 71-72%), lo que hace que cada décima de las mejoras descritas haya sido decisiva.

## 9. Limitaciones y problemas conocidos

El reporte de Pythia, por su brevedad, no dedica una sección a limitaciones; pero estas son centrales para la Clase 23 (slides 14-19), que usa a Pythia precisamente para mostrar **cómo y por qué fallan los sistemas VQA clásicos**. Conecto cada falla con su causa arquitectónica:

- **Persistencia de *language priors* / sesgo lingüístico (slide 15-16).** A pesar de que VQA v2.0 está balanceado para mitigarlos, el modelo sigue explotando correlaciones del lenguaje. El ejemplo de la clase: ante "is it a cat?" tiende a responder **"yes"** casi siempre, porque las preguntas binarias de presencia tienen una fuerte prior hacia "yes" en los datos. La fusión Hadamard y el clasificador sigmoide no imponen ninguna restricción que obligue a verificar visualmente la presencia del objeto.

- **Falta de composicionalidad / fallo en conteo (slide 17).** El modelo falla en preguntas que requieren **razonamiento composicional o cuantificación**, como "are there two cats?". La atención *top-down* produce una **suma ponderada** $\hat{v}=\sum_i\alpha_i v_i$ que **colapsa** las K regiones en un único vector de 2048-D: ese promedio destruye la información de **cardinalidad** (cuántas regiones distintas se activaron). Contar requiere preservar identidades de instancias, algo que un *soft attention* + suma ponderada no hace. Es una limitación arquitectónica intrínseca del esquema up-down.

- **Confusión en composiciones de color/atributos (slide 18).** Preguntas que combinan atributos ("the red cup next to the blue plate") confunden al modelo, porque el *binding* atributo-objeto no está modelado explícitamente: las características de región mezclan objeto y atributo, y la atención difusa no garantiza asociar el color correcto con el objeto correcto.

- **Vocabulario de respuestas cerrado y limitado (slide 19).** El clasificador sigmoide opera sobre un **vocabulario fijo** (las ~3129 respuestas más frecuentes). Toda respuesta fuera de ese conjunto es **inalcanzable**: el modelo solo puede emitir clases que vio en entrenamiento. No genera lenguaje libre; clasifica. Una pregunta cuya respuesta correcta no esté en el vocabulario tiene *accuracy* 0 garantizada, sin importar cuán bien "entienda" la imagen.

Estas cuatro fallas no son accidentes de Pythia, sino **consecuencias del paradigma up-down**: detección + atención blanda + suma ponderada + clasificación de vocabulario cerrado. Justamente por eso la Clase 23 las usa como motivación para la siguiente generación de modelos.

## 10. Impacto y legado

- **De Pythia a MMF.** El framework Pythia v0.1 evolucionó hasta convertirse en **MMF (Multimodal Framework)** de Facebook AI, una de las plataformas de investigación multimodal más usadas. La filosofía "modular" anunciada en la introducción del paper —módulos intercambiables para codificación de pregunta, extracción de características, fusión y clasificación— se materializó en esa biblioteca.

- **El estado del arte de la era up-down.** Pythia consolidó la receta canónica de VQA clásico: características *bottom-up* de un detector + atención *top-down* + fusión Hadamard + clasificación multi-etiqueta. Durante 2017-2019 esta fue *la* arquitectura de referencia.

- **El relevo de los Transformers vision-language.** Casi inmediatamente después, los VLMs basados en Transformers **destronaron** a Pythia. **ViLBERT** y **LXMERT** (ambos 2019) reemplazaron la atención *top-down* de una sola pasada por **co-atención cruzada multicapa** entre tokens de texto y regiones de imagen, preentrenada con objetivos tipo BERT (masked language/region modeling) sobre grandes corpus imagen-texto. Estos modelos superaron a Pythia en VQA y se generalizaron a muchas tareas multimodales. La transición es exactamente el arco narrativo de la Clase 23: del up-down/Pythia a los VLMs Transformer.

- **Lección metodológica perdurable.** Más allá de la arquitectura, Pythia dejó una enseñanza que sigue vigente: en *deep learning* aplicado, el *learning rate schedule*, la elección de fusión, la *data augmentation* y el *ensembling* diverso pueden valer tantos puntos como una arquitectura "nueva". Es un paper de ingeniería rigurosa más que de invención.

## 11. Conexión con la Clase 23

Pythia es el **modelo que estructura la primera mitad de la Clase 23 (slides 9-19)**:

- **Slides 9-12:** introducción al up-down/BUTD como "modelo up-down", con el detector que produce K×2048 y la atención *top-down*.
- **Slide 13:** el **diagrama de arquitectura completo** que este análisis recorre en la Sección 4: pregunta → 14 tokens → GloVe (14×300) → GRU → 512; imagen → detector → K×2048; *top-down attention* (softmax sobre K, *weighted sum* a 2048, proyección a 512); fusión por producto de los dos vectores de 512; → sigmoide → *scores* de respuesta. La frase "mezcla información multimodal sin aumentar la dimensión del modelo" es de esta slide (ver Sección 6).
- **Slides 14-19:** los **fallos** de Pythia (Sección 9) — "yes" automático, fallo en conteo, confusión de color/composición, vocabulario cerrado — que motivan la transición hacia modelos más capaces.

Estudiar este paper junto al de Anderson et al. (BUTD) da la base completa para entender qué hacía un sistema VQA de 2018 y por qué la comunidad migró a los Transformers vision-language.

## 12. Notas y enlaces

- **Paper (Pythia):** arXiv:1807.09956 — `https://arxiv.org/abs/1807.09956`
- **Código:** `https://github.com/facebookresearch/pythia` (luego absorbido en MMF: `https://github.com/facebookresearch/mmf`).
- **Paper base imprescindible:** P. Anderson et al., *Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering*, CVPR 2018 (ref. [1]). Sin él, el reporte de Pythia no es autocontenido.
- **Referencias clave citadas en el paper:**
  - [4] Fukui et al., *Multimodal Compact Bilinear Pooling* (MCB), 2016 — fusión bilineal.
  - [5] Goyal et al., *Accurate, Large Minibatch SGD* (ImageNet en 1 hora), 2017 — el *warmup* del learning rate.
  - [6] Goyal et al., *Making the V in VQA Matter* (VQA v2.0), CVPR 2017 — el dataset balanceado.
  - [8] Kingma & Ba, *Adam* (Adamax), 2014 — el optimizador.
  - [10] Lin et al., *Feature Pyramid Networks*, CVPR 2017 — el detector de Detectron.
  - [11] Pennington et al., *GloVe*, EMNLP 2014 — los *embeddings* de palabra.
  - [12] Ren et al., *Faster R-CNN*, NIPS 2015 — el detector de BUTD.
  - [13] Salimans & Kingma, *Weight Normalization*, NIPS 2016 — la normalización de pesos.
  - [14] Teney et al., *Tips and Tricks for VQA*, 2017 — el manual de buenas prácticas (BCE multi-etiqueta, sigmoide).
  - [16] Yu et al., *Beyond Bilinear: Generalized Multimodal Factorized High-order Pooling*, TNNLS 2018 — la atención sobre la pregunta.

- **Una nota sobre el nombre del detector en el diagrama de la clase.** La slide menciona "Mask R-CNN + ResNet101"; el reporte de Pythia describe detectores **Detectron/FPN con backbone ResNeXt** (post-mejora) sobre la base original de **Faster R-CNN/ResNet-101** de BUTD. Mask R-CNN, Faster R-CNN y FPN pertenecen a la misma familia de detectores de dos etapas de Detectron; la diferencia es de configuración, no de paradigma. Lo esencial —un detector de objetos que entrega K regiones con un vector de 2048-D cada una— se mantiene idéntico.
