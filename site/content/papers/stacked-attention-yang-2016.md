---
title: "Stacked Attention Networks (SAN)"
weight: 245
math: true
---

{{< paper-card
    title="Stacked Attention Networks for Image Question Answering"
    authors="Yang, He, Gao, Deng, Smola"
    year="2016"
    venue="CVPR 2016"
    pdf="/papers/stacked-attention-yang-2016.pdf"
    arxiv="1511.02274" >}}
Introduce la idea de que responder una pregunta visual no es un acto único de "mirar", sino un proceso de **razonamiento en varios pasos** (*multi-hop*). SAN aplica **atención espacial guiada por la pregunta** sobre una grilla de features de CNN y **apila** dos o más capas de atención: la primera localiza de forma difusa todos los conceptos referidos por la pregunta, las siguientes filtran el ruido y apuntan a la región que contiene la respuesta. Es el antecedente conceptual directo de la *top-down attention* sobre objetos detectados que popularizarían BUTD y Pythia.
{{< /paper-card >}}

---

## Contexto

El Visual Question Answering (VQA) emerge alrededor de 2014–2015 como una de las tareas multimodales más activas, en la intersección de visión y NLP: dada una imagen y una pregunta en lenguaje natural, producir la respuesta correcta. Es una prueba de comprensión integrada que exige razonar conjuntamente sobre contenido visual y texto. Ver [fundamento Visual Question Answering](/fundamentos/visual-question-answering) y [dominio Multimodal](/dominios/multimodal).

Los modelos previos a SAN compartían un patrón: extraer **un único vector de imagen global** con una CNN (la activación de la última capa *fully-connected* de GoogLeNet o VGGNet) y codificar la pregunta con una LSTM, para luego combinar ambos vectores por concatenación, suma o producto elemento a elemento. Así operaban Ask-Your-Neurons (Malinowski et al.), VSE (Ren et al.), el modelo VQA de Antol et al. y el IMG-CNN (Ma, Lu, Li).

El defecto es estructural: **al colapsar la imagen en un solo vector se pierde la información espacial**. Si la respuesta depende de un objeto pequeño y específico, el vector global lo mezcla con el resto de la escena e introduce ruido que arrastra la predicción a respuestas subóptimas. El ejemplo canónico del paper es la pregunta *"what are sitting in the basket on a bicycle?"*: para responder `dogs`, el modelo debe localizar los conceptos referidos (`basket`, `bicycle`, la relación `sitting in`), descartar lo irrelevante y apuntar finalmente a la región indicativa. Hacerlo con un vector global es prácticamente imposible.

---

## Ideas principales

La contribución de SAN tiene dos componentes separables: **atención espacial guiada por la pregunta** y el hecho de **apilarla** en varias capas.

### Atención espacial sobre la grilla CNN

En lugar de un vector global, SAN extrae un **mapa de features espacial**. Se reescala la imagen a **448×448 píxeles** y se toma la salida de la **última capa de pooling** de VGGNet, de dimensión **512 × 14 × 14**. Esto produce **14 × 14 = 196 regiones**, cada una un vector de 512 dimensiones correspondiente a un parche de 32×32 píxeles de la imagen original:

$$f_I = \mathrm{CNN}_{\text{vgg}}(I)$$

Cada vector de región se proyecta con un perceptrón de una capa a la misma dimensión que el vector de pregunta, formando la matriz $v_I$ (columna $i$ = región $i$):

$$v_I = \tanh(W_I f_I + b_I)$$

La pregunta se codifica como vector $v_Q$ con una **LSTM** (último estado oculto $h_T$) o, alternativamente, con una **CNN de oraciones** estilo Kim (2014) con filtros unigrama/bigrama/trigrama y max-pooling temporal. Ambos codificadores rinden de forma similar.

Con $v_I$ y $v_Q$, una capa de atención calcula una **distribución de probabilidad sobre las regiones**. Este es el [mecanismo de atención](/fundamentos/mecanismo-atencion) de Bahdanau (2014) trasladado de "palabras de origen" a "regiones de imagen" —la misma idea que *Show, Attend and Tell* (Xu et al., 2015) aplicó a captioning; la novedad de SAN es traerla a VQA:

$$h_A = \tanh(W_{I,A}\, v_I \oplus (W_{Q,A}\, v_Q + b_A))$$
$$p_I = \mathrm{softmax}(W_P\, h_A + b_P)$$

donde $v_I \in \mathbb{R}^{d \times m}$ con $m = 196$ regiones, y el símbolo $\oplus$ denota sumar el vector $(W_{Q,A}\, v_Q + b_A) \in \mathbb{R}^{k}$ a **cada columna** de la matriz $W_{I,A}\, v_I \in \mathbb{R}^{k \times m}$ (broadcasting). El vector $p_I \in \mathbb{R}^m$ da la probabilidad de atención de cada región. Con él se calcula la suma ponderada de regiones y la **consulta refinada**:

$$\tilde{v}_I = \sum_i p_i\, v_i \qquad\qquad u = \tilde{v}_I + v_Q$$

$u$ es más informativo que la simple fusión global porque codifica simultáneamente la pregunta **y** la evidencia visual relevante.

### Stacked attention: razonamiento multi-paso

Aquí está el aporte distintivo. Los autores argumentan que **una sola capa de atención no basta para preguntas complejas**: cuando hay varios objetos y relaciones sutiles, la atención de un solo paso reparte la masa de probabilidad de forma difusa sin lograr decidir cuál candidato es la respuesta. La solución es **iterar el proceso $K$ veces**, donde cada capa toma como consulta el resultado refinado de la anterior. Para la $k$-ésima capa, con $u^0 = v_Q$:

$$h_A^k = \tanh(W_{I,A}^k\, v_I \oplus (W_{Q,A}^k\, u^{k-1} + b_A^k))$$
$$p_I^k = \mathrm{softmax}(W_P^k\, h_A^k + b_P^k)$$
$$\tilde{v}_I^k = \sum_i p_i^k\, v_i \qquad\qquad u^k = \tilde{v}_I^k + u^{k-1}$$

La consulta final $u^K$ infiere la respuesta como **clasificación** sobre un vocabulario fijo (casi todas las respuestas son de una palabra; en VQA se usan las 1000 más frecuentes):

$$p_{\text{ans}} = \mathrm{softmax}(W_u\, u^K + b_u)$$

Un detalle clave de diseño: la actualización $u^k = \tilde{v}_I^k + u^{k-1}$ **acumula** sobre la consulta vieja en lugar de reemplazarla. Esta conexión residual implícita preserva la información de la pregunta a través de las capas y estabiliza el entrenamiento.

La metáfora operativa es *"locate roughly, then sharpen"*. En el ejemplo del canasto: la **capa 1** dispersa la atención sobre `bicycle`, `basket` y los objetos dentro del canasto (localización amplia); la **capa 2**, con $u^1$ como consulta —que ya es **multimodal**, pues incorpora evidencia de la primera pasada—, concentra la atención en los perros y produce `dogs`. Cada hop condiciona el siguiente sobre lo ya observado: es razonamiento iterativo y compositivo, no fusión en un paso.

La **visualización de los mapas de atención** (sobremuestreados con filtro gaussiano de 14×14 a 448×448) confirma empíricamente el patrón en todos los ejemplos: en la primera capa la atención está dispersa sobre muchos objetos; en la segunda está mucho más enfocada en la región que lleva a la respuesta. Esto aporta además explicabilidad: se puede inspeccionar literalmente dónde mira el modelo en cada paso.

---

## Resultados experimentales

SAN se evalúa en **cuatro benchmarks** con la nomenclatura **SAN(k, modelo)**, donde $k \in \{1, 2\}$ es el número de capas de atención y el modelo es LSTM o CNN. La VGGNet se mantiene **congelada** (sin fine-tuning). Métricas: accuracy de clasificación, WUPS (umbrales 0.9 y 0.0) en DAQUAR/COCO-QA, y la métrica de Antol et al. en VQA.

**DAQUAR-ALL** (accuracy en %): las SAN de dos capas (29.3) superan al IMG-CNN (23.4) y a Ask-Your-Neurons (21.7) por **5.9** y **7.6 puntos** absolutos. **DAQUAR-REDUCED**: SAN(2, LSTM) alcanza **46.2**, mejorando sobre IMG-CNN (39.7) por 6.5 puntos.

**COCO-QA** (el dataset más grande de los cuatro tras VQA, ~78k muestras de entrenamiento):

| Método | Accuracy | WUPS0.9 | WUPS0.0 |
|---|---|---|---|
| IMG+BOW (Ren et al.) | 55.9 | 66.8 | 89.0 |
| 2-VIS+BLSTM | 55.1 | 65.3 | 88.6 |
| IMG-CNN | 55.0 | 65.4 | 88.6 |
| SAN(1, CNN) | 60.7 | 70.6 | 90.5 |
| **SAN(2, CNN)** | **61.6** | **71.6** | **90.9** |

SAN(2, CNN) supera a los mejores baselines por **5.1–6.6 puntos**. Por clase mejora especialmente en *Color* (+7.2) y *Objects* (+6.1).

**VQA — servidor de prueba oficial** (accuracy en %):

| Método | test-dev All | Yes/No | Number | Other |
|---|---|---|---|---|
| LSTM Q+I (Antol et al.) | 53.7 | 78.9 | 35.2 | 36.4 |
| **SAN(2, CNN)** | **58.7** | 79.3 | 36.6 | **46.1** |

SAN(2, CNN) supera al mejor baseline de Antol et al. por **4.8 puntos**, con la mayor ganancia en *Other* (+9.7).

**1 vs 2 capas.** El mensaje central: en los cuatro datasets, **las SAN de dos capas siempre superan a las de una capa**, aunque por márgenes modestos (en COCO-QA, +2.2 en *Color*, +1.0 en *Objects*; en VQA, +1.4 en *Other*). Los autores reportan que **tres o más capas no mejoran**: el beneficio del razonamiento iterativo se satura rápido. El tipo *Yes/No* casi no se beneficia, porque su respuesta depende fuertemente de la pregunta y poco del modelado visual.

**Análisis de errores** (100 imágenes de COCO-QA mal respondidas): solo el **22%** son atención en región equivocada; un **42%** ocurre con **atención correcta pero respuesta equivocada** —el cuello de botella ya no es *dónde mirar* sino el razonamiento posterior—; y un 36% combinado son casos ambiguos/aceptables o etiquetas de oro erróneas, lo que sugiere que la accuracy reportada subestima el rendimiento real.

---

## Limitaciones

- **Grilla uniforme, no objetos.** SAN atiende sobre una grilla regular de 14×14 celdas, parches fijos de 32×32 píxeles sin noción de objetos ni de sus límites. Un objeto pequeño cae en una sola celda (poca resolución) y uno grande se reparte en muchas. Esta es exactamente la limitación que [Bottom-Up Attention (Anderson 2018)](/papers/bottom-up-attention-anderson-2018) corrige atendiendo sobre **regiones propuestas por un detector de objetos** (Faster R-CNN), donde cada región es un objeto saliente con semántica.
- **Número fijo de hops.** $K$ es un hiperparámetro fijo (2 en el mejor caso); el modelo no decide adaptativamente cuántos pasos de razonamiento necesita cada pregunta.
- **El cuello de botella se desplazó al razonamiento.** El 42% de los fallos con atención correcta muestra que SAN mejora *dónde mirar*, pero la composición y el razonamiento relacional siguen limitados.
- **CNN congelada.** No se hace fine-tuning de VGGNet, lo que limita la adaptación de las features visuales a VQA.
- **Sesgos de lenguaje.** El modesto beneficio en *Yes/No* evidencia que VQA tiene fuertes sesgos lingüísticos que la atención visual no resuelve por sí sola.

---

## Por qué importa hoy

SAN es el paper que **popularizó la atención visual guiada por pregunta como componente estándar de VQA**: después de él, prácticamente todo modelo competitivo incorpora algún mecanismo de atención sobre la imagen. Introdujo además el **razonamiento multi-hop** ("localizar grueso, luego refinar"), que inspiró la co-atención (Lu et al. 2016) y los módulos de razonamiento posteriores.

Su legado más directo es ser el **antecedente conceptual de la *top-down attention*** que materializan [Bottom-Up Attention (Anderson 2018)](/papers/bottom-up-attention-anderson-2018) y [Pythia (Jiang 2018)](/papers/pythia-jiang-2018). En ambos casos la pregunta actúa como consulta que pondera con un softmax un conjunto de regiones visuales, y la suma ponderada de features alimenta la predicción —SAN comparte las ecuaciones de atención con la *top-down attention* de Pythia. Lo que cambia es **qué son las regiones**:

| Aspecto | SAN (Yang 2016) | Pythia / BUTD (Anderson 2018) |
|---|---|---|
| Unidad de atención | Grilla 14×14 = 196 celdas (parches 32×32 px) | $K$ regiones de objetos detectados |
| Origen de las regiones | Mapa de pooling de VGGNet (sin semántica) | Faster R-CNN sobre Visual Genome (semántica explícita) |
| Multi-hop | Sí (apilar 2 capas) | Típicamente una capa top-down + señal bottom-up |
| Interpretabilidad | Mapa de calor sobre la imagen | Bounding boxes de objetos atendidos |

El paso de "grilla" a "objetos" es lo que dio el salto de calidad que Pythia capitaliza al ganar el VQA Challenge 2018. La noción de refinar progresivamente una consulta multimodal a través de varias capas reaparece además en los transformers multimodales modernos (LXMERT, ViLBERT), donde múltiples capas de cross-attention cumplen un rol análogo al apilamiento de SAN. En la cronología del módulo: Bahdanau (2014) → Xu et al. (2015, *Show, Attend and Tell*) → **Yang et al. (2016, SAN)** → Anderson et al. (2018, BUTD) → Pythia (2018) → transformers multimodales (2019+). Ver [Clase 23](/clases/clase-23).

---

## Notas y enlaces

- **Truco de implementación clave.** Tomar la **última pooling layer** (512×14×14) en vez de la última capa fully-connected es lo que preserva la estructura espacial; sin ese mapa no hay sobre qué atender.
- **VQA como clasificación.** SAN trata la tarea como clasificación sobre un vocabulario fijo (top-1000 en VQA, que cubre el 82.67% de las respuestas), no como generación. Funciona porque casi todas las respuestas son de una palabra, pero limita la expresividad —restricción común en la era 2016.
- **Relación con fusión bilineal.** [MCB (Fukui 2016)](/papers/mcb-fukui-2016) explora una vía complementaria: en lugar de refinar la consulta por iteración, mejora la **fusión** entre pregunta e imagen con un producto exterior compacto. Ambos atacan la misma limitación de la fusión global ingenua desde ángulos distintos.
- **Reproducibilidad.** CNN congelada, SGD con momentum 0.9, batch 100, gradient clipping, dropout, $K = 2$. Tres o más capas no mejoran.
- **Referencias germinales:** Bahdanau et al. (2014, arXiv:1409.0473, origen de la atención); Xu et al. (2015, arXiv:1502.03044, *Show, Attend and Tell*, predecesor directo); Antol et al. (2015, arXiv:1505.00468, dataset VQA); Ren et al. (2015, arXiv:1505.02074, COCO-QA); Simonyan-Zisserman (2014, arXiv:1409.1556, VGGNet).
