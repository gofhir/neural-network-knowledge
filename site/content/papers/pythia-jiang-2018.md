---
title: "Pythia v0.1 (Winning Entry VQA Challenge 2018)"
weight: 242
math: true
---

{{< paper-card
    title="Pythia v0.1: the Winning Entry to the VQA Challenge 2018"
    authors="Jiang, Natarajan, Chen, Rohrbach, Batra, Parikh"
    year="2018"
    venue="VQA Challenge 2018 / arXiv"
    pdf="/papers/pythia-jiang-2018.pdf"
    arxiv="1807.09956" >}}
Pythia v0.1 es una **reimplementación modular** del modelo *bottom-up top-down* (up-down) de [Anderson et al. 2018](/papers/bottom-up-attention-anderson-2018). No introduce una arquitectura nueva: demuestra que un conjunto de mejoras incrementales "sutiles pero importantes" —*weight normalization* + ReLU, fusión por Hadamard, *learning rate schedule* con *warmup*, *fine-tuning* del detector, *data augmentation* y *ensembling* diverso— eleva la *accuracy* en VQA v2.0 de 65.67% a 70.24% (modelo único) y a **72.27% (ensemble), ganando el VQA Challenge 2018**. Es, ante todo, un manual de ingeniería de VQA de la era pre-Transformers, y el modelo central de la [Clase 23](/clases/clase-23).
{{< /paper-card >}}

---

## Contexto: el VQA Challenge y la base BUTD

**Visual Question Answering (VQA)** es la tarea de responder, en lenguaje natural, una pregunta abierta sobre una imagen (ver [fundamento Visual Question Answering](/fundamentos/visual-question-answering)). El *benchmark* dominante es [VQAv2 (Goyal 2017)](/papers/vqav2-goyal-2017), una versión "balanceada" del dataset original de Antol et al. (2015): para cada pregunta existen pares de imágenes con respuestas distintas, lo que **fuerza al modelo a mirar la imagen** y no a explotar atajos lingüísticos (los *language priors*).

El **VQA Challenge** es la competencia anual asociada a CVPR. Su métrica oficial es una *accuracy* suavizada: cada pregunta tiene 10 respuestas humanas y una respuesta del modelo recibe

$$\text{Acc}(a) = \min\left(\frac{\#\,\text{humanos que dieron } a}{3},\ 1\right),$$

es decir, crédito completo si al menos 3 de 10 anotadores coincidieron. Esto convierte VQA en un problema de **clasificación multi-etiqueta con etiquetas blandas** (*soft scores*), detalle que reaparece en la función de pérdida.

El punto de partida de Pythia es el modelo **bottom-up top-down** de [Bottom-Up Attention (Anderson 2018)](/papers/bottom-up-attention-anderson-2018), que había ganado el VQA Challenge **2017** (65.67% en *test-std*). Su idea central es combinar dos clases de [atención visual](/fundamentos/mecanismo-atencion):

- **Atención *bottom-up* (*data-driven*):** un detector de objetos (Faster R-CNN preentrenado en Visual Genome) propone **K regiones salientes**, cada una representada por un vector de **2048 dimensiones**. Reemplaza la grilla convolucional uniforme que usaba [Stacked Attention (Yang 2016)](/papers/stacked-attention-yang-2016) por regiones alineadas a objetos.
- **Atención *top-down* (*task-driven*):** la pregunta guía *cuáles* de esas K regiones merecen atención.

El nombre *Pythia* homenajea al oráculo de Apolo en Delfos, que respondía preguntas: un guiño perfecto para un sistema que responde preguntas.

---

## La arquitectura paso a paso

Pythia conserva el esqueleto de BUTD y refina cada bloque. Recorro el grafo tal como lo dibuja el diagrama de la clase, anotando dimensiones.

**(a) Embedding de la pregunta — GloVe + GRU.**
La pregunta (hasta 14 tokens) se embebe con [GloVe (Pennington 2014)](/papers/glove-pennington-2014) de **300 dimensiones** (matriz $14\times300$) y se procesa con una **GRU**. Pythia añade un módulo de **atención sobre los tokens de la pregunta** en lugar de tomar solo el último estado. Salida: $q \in \mathbb{R}^{512}$.

**(b) Características de imagen — K×2048 vía detector.**
La imagen pasa por el detector (Mask R-CNN / ResNet-101 en el diagrama; Detectron+FPN en la mejor configuración de Pythia) y produce una matriz de características visuales

$$V = [v_1, v_2, \dots, v_K], \qquad V \in \mathbb{R}^{K \times 2048}.$$

En la mejor configuración, $K=100$ cajas fijas por imagen.

**(c) Top-down attention — softmax sobre K regiones + weighted sum.**
Dado $q$ y cada $v_i$, se calcula un escalar de relevancia por región:

$$a_i = w_a^{\top}\, f_a\!\left(W_v v_i \,\circ\, W_q q\right),$$

donde $f_a$ es la no linealidad (*weight normalization* + ReLU en Pythia; *gated tanh* en BUTD). Los escalares se normalizan con **softmax sobre las K regiones**:

$$\alpha_i = \frac{\exp(a_i)}{\sum_{j=1}^{K}\exp(a_j)}, \qquad \sum_{i=1}^{K}\alpha_i = 1,$$

y la imagen atendida es la **suma ponderada** de las características de región, luego proyectada de 2048-D a 512-D:

$$\hat{v} = \sum_{i=1}^{K}\alpha_i\, v_i, \qquad \hat{v} \in \mathbb{R}^{2048}.$$

**(d) Fusión multimodal — producto de Hadamard.**
Las dos representaciones de 512-D (texto y visión atendida) se combinan por **multiplicación elemento a elemento**:

$$h = \big(W_q' q\big) \,\circ\, \big(W_v' \hat{v}\big), \qquad h \in \mathbb{R}^{d_h},$$

con el mejor tamaño de capa oculta en $d_h = 5000$. El producto de Hadamard "mezcla información multimodal sin aumentar la dimensión del modelo".

**(e) Clasificador sigmoide — multi-etiqueta sobre respuestas.**
El vector fusionado pasa por capas lineales y una **activación sigmoide** que emite un *score* independiente en $[0,1]$ por cada respuesta del vocabulario fijo (las ~3129 respuestas más frecuentes):

$$\hat{y} = \sigma\!\big(W_2\, g(W_1 h)\big) \in [0,1]^{|\mathcal{A}|}.$$

Que sea sigmoide y no softmax es deliberado: VQA es clasificación **multi-etiqueta** porque varias respuestas pueden ser parcialmente correctas. La pérdida es **binary cross-entropy** sobre los *soft scores* $s_a = \min(\#\text{votos}/3, 1)$:

$$\mathcal{L} = -\sum_{a\in\mathcal{A}} \Big[\, s_a \log \hat{y}_a + (1-s_a)\log(1-\hat{y}_a)\,\Big].$$

---

## Las mejoras sobre BUTD

El corazón del paper es esta escalera de cambios incrementales, cada peldaño sumando *accuracy*:

| Modelo | test-dev | test-std |
|---|---|---|
| up-down (BUTD, baseline 2017) | 65.32 | 65.67 |
| + Adaptación de arquitectura | 66.91 | — |
| + Learning rate schedule | 68.05 | — |
| + Detectron & Fine-tuning | 68.49 | — |
| + Data Augmentation* | 69.24 | — |
| + Grid Features* | 69.81 | — |
| + 100 bboxes* | 70.01 | 70.24 |
| Ensemble, 30× mismo modelo | 70.96 | — |
| Ensemble, 30× modelos diversos | 72.18 | **72.27** |

(\* = mejoras post-desafío, no incluidas en la entrada oficial.)

**1. Adaptación de arquitectura (+1.59, el salto individual más grande).** Tres cambios agrupados: (a) **weight normalization + ReLU** en vez de *gated tanh* —*weight normalization* reparametriza cada peso como $w = g\,\frac{v}{\lVert v\rVert}$, separando magnitud y dirección, estabilizando el entrenamiento sin el costo de *batch norm*—; (b) **fusión por producto de Hadamard** en vez de concatenación; (c) GloVe 300-D + atención sobre la pregunta + capa oculta de 5000.

**2. Learning rate schedule con *warmup* (+1.14).** Optimizador **Adamax**. Subir el *learning rate* ingenuamente causaba divergencia; la solución fue el *warmup* de Goyal et al. ("ImageNet en 1 hora"): empezar en 0.002, subir linealmente hasta 0.01 en la iteración 1000, reducir por 0.1 en la 5K y luego cada 2K, deteniendo en 12K iteraciones. El calendario de aprendizaje valió tanto como un cambio de arquitectura.

**3. Detectron + fine-tuning (+0.44).** Reemplaza el Faster R-CNN/ResNet-101 por detectores **Detectron con Feature Pyramid Networks (FPN)** y backbone ResNeXt, con dos capas *fully connected* (fc6, fc7). Las características de 2048-D salen de fc6 y se hace *fine-tuning* solo de fc7 (con *learning rate* 0.1× el global), mucho más barato que afinar capas convolucionales.

**4. Data augmentation (+0.75).** Datos adicionales de **Visual Genome** y **Visual Dialog** (los 10 turnos de cada diálogo convertidos en 10 pares pregunta-respuesta), replicando su única respuesta 10 veces para encajar en el protocolo VQA. Además, ***mirroring*** de imágenes VQA: al voltear horizontalmente, se intercambian los tokens **"left" ↔ "right"** para no confundir la orientación.

**5. Grid + region features (+0.57) y 100 bboxes (+0.20).** Las características *bottom-up* no capturan zonas no cubiertas por ninguna propuesta (cielo, fondo, texturas). Pythia las combina con **características de grilla** (atención sobre el mapa convolucional uniforme de ResNet-152), fusionadas por separado con la pregunta y luego concatenadas. Y usa **100 cajas fijas** por imagen en lugar del protocolo adaptativo (10–100) de up-down.

**6. Ensembling (+0.95 a +2.26).** El *ensemble* de **30 copias del mismo modelo** (distintas semillas) se aplana en 70.96%: rendimientos decrecientes rápidos. El *ensemble* de **30 modelos diversos** (con/sin *data augmentation*, distintos Detectron) alcanza **72.27%**, una mejora de +1.31 sobre el ingenuo. La lección: en *ensembling*, **la diversidad importa más que la cantidad**.

### Por qué Hadamard

La concatenación $[q'; \hat{v}'] \in \mathbb{R}^{2d}$ duplica la dimensión y, seguida de una capa lineal, solo permite **interacciones aditivas** ($W_1 q' + W_2 \hat{v}'$). El producto de Hadamard $q' \circ \hat{v}' \in \mathbb{R}^{d}$ mantiene la dimensión y crea **interacciones multiplicativas**: cada componente $q'_i\cdot\hat{v}'_i$ es un "gating" cruzado donde la activación textual modula la visual y viceversa. Captura correlaciones de segundo orden que la suma no puede; es una versión barata y diagonal del *bilinear pooling* (MCB), equivalente a forzar el tensor de interacción $W$ a ser diagonal: muchos menos parámetros, casi todo el beneficio.

---

## Resultados

La cifra titular: **72.27% en test-std de VQA v2.0** (entrada oficial A-STAR: 72.25%), suficiente para **ganar el VQA Challenge 2018**. Lecturas clave:

- El **modelo único pasó de 65.32% a 70.01%** en *test-dev* (+4.69 puntos) **solo con ingeniería incremental**, sin arquitectura nueva.
- El **mayor salto individual** vino de la adaptación de arquitectura (+1.59), seguido del *learning schedule* (+1.14).
- El ***ensemble* aportó ~2 puntos** (70.01 → 72.27), pero solo cuando fue **diverso**.
- Frente al segundo lugar, el margen fue estrecho (las entradas top de 2018 rondaban 71–72%), lo que hizo decisiva cada décima de las mejoras.

El mensaje del paper: la diferencia entre un sistema VQA competente y uno ganador estaba, en 2018, en los **detalles de implementación**.

---

## Limitaciones

El reporte no dedica sección a limitaciones, pero estas son centrales para la [Clase 23](/clases/clase-23) (slides 14-19), que usa a Pythia para mostrar **cómo y por qué fallan los sistemas VQA clásicos**. Cada falla tiene una causa arquitectónica:

- **Persistencia de *language priors* (slides 15-16).** Pese al balanceo de VQA v2.0, el modelo sigue explotando correlaciones del lenguaje: ante "is it a cat?" tiende a responder "yes" casi siempre, porque las preguntas binarias de presencia tienen una fuerte prior hacia "yes". Nada en la fusión Hadamard ni en el clasificador sigmoide obliga a verificar visualmente la presencia del objeto.

- **Falta de composicionalidad / fallo en conteo (slide 17).** La atención *top-down* produce una suma ponderada $\hat{v}=\sum_i\alpha_i v_i$ que **colapsa** las K regiones en un único vector de 2048-D: ese promedio destruye la información de **cardinalidad**. Contar ("are there two cats?") requiere preservar identidades de instancias, algo que un *soft attention* + suma ponderada no hace. Es una limitación arquitectónica intrínseca del esquema up-down.

- **Confusión en composiciones de color/atributos (slide 18).** Preguntas que combinan atributos ("the red cup next to the blue plate") confunden al modelo porque el *binding* atributo-objeto no está modelado: la atención difusa no garantiza asociar el color correcto con el objeto correcto.

- **Vocabulario de respuestas cerrado (slide 19).** El clasificador sigmoide opera sobre un **vocabulario fijo** (~3129 respuestas). Toda respuesta fuera de ese conjunto es **inalcanzable**: el modelo clasifica, no genera lenguaje libre. Una pregunta cuya respuesta correcta no esté en el vocabulario tiene *accuracy* 0 garantizada.

Estas fallas no son accidentes de Pythia, sino **consecuencias del paradigma up-down**: detección + atención blanda + suma ponderada + clasificación de vocabulario cerrado. Por eso la Clase 23 las usa como motivación para la siguiente generación de modelos.

---

## Por qué importa hoy

- **De Pythia a MMF.** El framework Pythia v0.1 evolucionó hasta convertirse en **MMF (Multimodal Framework)** de Facebook AI, una de las plataformas de investigación multimodal más usadas. La filosofía modular del paper —módulos intercambiables para codificación de pregunta, extracción de características, fusión y clasificación— se materializó en esa biblioteca.

- **El estado del arte de la era up-down.** Pythia consolidó la receta canónica de VQA clásico: características *bottom-up* de un detector + atención *top-down* + fusión Hadamard + clasificación multi-etiqueta. Durante 2017-2019 fue *la* arquitectura de referencia.

- **El relevo de los Transformers vision-language.** Casi inmediatamente, **ViLBERT** y **LXMERT** (2019) reemplazaron la atención *top-down* de una sola pasada por **co-atención cruzada multicapa** entre tokens de texto y regiones, preentrenada con objetivos tipo BERT sobre grandes corpus imagen-texto, y destronaron a Pythia. Esta transición es el arco narrativo de la Clase 23: del up-down/Pythia a los VLMs Transformer (ver [dominio Multimodal](/dominios/multimodal)).

- **Lección metodológica perdurable.** En *deep learning* aplicado, el *learning rate schedule*, la elección de fusión, la *data augmentation* y el *ensembling* diverso pueden valer tantos puntos como una arquitectura "nueva". Es un paper de ingeniería rigurosa más que de invención.

---

## Notas y enlaces

- **Paper:** arXiv:1807.09956 — `https://arxiv.org/abs/1807.09956`
- **Código:** `https://github.com/facebookresearch/pythia` (luego absorbido en MMF: `https://github.com/facebookresearch/mmf`).
- **Paper base imprescindible:** [Bottom-Up Attention (Anderson 2018)](/papers/bottom-up-attention-anderson-2018), CVPR 2018. Sin él, el reporte de Pythia no es autocontenido.
- **Sobre el detector en el diagrama de la clase.** La slide menciona "Mask R-CNN + ResNet101"; el reporte describe Detectron/FPN con backbone ResNeXt sobre la base original de Faster R-CNN/ResNet-101 de BUTD. Mask R-CNN, Faster R-CNN y FPN pertenecen a la misma familia de detectores de dos etapas; la diferencia es de configuración, no de paradigma. Lo esencial —un detector que entrega K regiones con un vector de 2048-D cada una— se mantiene idéntico.
- **Referencias clave:** Goyal et al. (VQA v2.0, *warmup*), Pennington et al. (GloVe), Salimans & Kingma (*weight normalization*), Teney et al. (*Tips and Tricks for VQA*: BCE multi-etiqueta), Fukui et al. (MCB, *bilinear pooling*).

Ver fundamentos: [Visual Question Answering](/fundamentos/visual-question-answering) · [Mecanismo de Atención](/fundamentos/mecanismo-atencion). Dominio: [Multimodal](/dominios/multimodal). Clase: [Clase 23](/clases/clase-23).
