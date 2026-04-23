---
title: "Bottom-Up/Top-Down Attention"
weight: 230
math: true
---

{{< paper-card
    title="Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering"
    authors="Anderson, He, Buehler, Teney, Johnson, Gould, Zhang"
    year="2018"
    venue="CVPR 2018"
    pdf="/papers/bottom-up-attention-anderson-2018.pdf"
    arxiv="1707.07998" >}}
Propone un **mecanismo combinado** de atencion bottom-up (regiones salientes propuestas por Faster R-CNN) y top-down (pesos adaptativos segun la tarea). Reemplaza el grid CNN uniforme que usaban modelos previos (Xu 2015) con **regiones alineadas a objetos**, obteniendo un sustrato mas natural para atencion visual. Establece nuevo state-of-art en image captioning MSCOCO (CIDEr 117.9, SPICE 21.5) y **gano el VQA Challenge 2017** con 70.3% de accuracy.
{{< /paper-card >}}

---

## Contexto

Los modelos previos de visual attention ([Xu et al. 2015 Show-Attend-Tell](/papers/show-attend-tell-xu-2015), y los modelos estandar de VQA) atendian sobre un **grid uniforme** de features de CNN -- por ejemplo $14 \times 14$ regiones del 4th conv layer de VGG. El problema:

- Las regiones del grid **no se alinean con objetos reales**. Un objeto pequeno puede caer entre celdas; un objeto grande puede ocupar varias.
- La atencion debe "reconstruir" el objeto a partir de partes del grid, lo que es dificil.
- Para VQA, preguntas sobre objetos especificos ("de que color es la pelota?") requieren selectivamente enfocarse en esa pelota -- mas natural con regiones que con grids.

Anderson et al. (Adelaide + Microsoft + ANU) propusieron en CVPR 2018 usar **regiones propuestas por Faster R-CNN** como base de atencion. Resultado: un salto cualitativo en calidad y state-of-art en dos tareas simultaneas.

---

## Ideas principales

### 1. Bottom-up: Faster R-CNN como proposer de regiones

En vez de un grid uniforme, usan **Faster R-CNN** (Ren et al. 2015) -- un detector de objetos -- para proponer **k regiones salientes** en la imagen, cada una con:

- **Bounding box** (localizacion).
- **Feature vector 2048-dim** (mean-pooled sobre la region).

El detector es preentrenado en:

1. **ImageNet** para clasificacion (ResNet-101 backbone).
2. **Visual Genome** dataset con anotaciones de objetos + atributos (1600 clases de objetos, 400 atributos).

Se quedan con regiones donde la probabilidad de deteccion exceda un threshold → tipicamente 10-100 regiones por imagen.

Notablemente, esto **ya es una forma de hard attention** -- las regiones son selecciones discretas de las salientes.

### 2. Top-down: atencion sobre las regiones

Dadas las regiones $V = \{v_1, \ldots, v_k\}$ con $v_i \in \mathbb{R}^{2048}$, el modelo aplica atencion soft:

$$a_{i,t} = w_a^T \tanh(W_{va} v_i + W_{ha} h_t^1)$$
$$\alpha_t = \text{softmax}(a_t)$$
$$\hat{v}_t = \sum_i \alpha_{i,t} v_i$$

donde $h_t^1$ es el estado de la "Top-Down Attention LSTM" que provee el query dependiente del contexto de la tarea.

### 3. Modelo de Image Captioning

Dos-LSTM stack:

- **LSTM1 (Top-Down Attention LSTM)**: recibe $[h_{t-1}^2, \bar{v}, W_e \Pi_t]$ donde $\bar{v}$ es el mean de regiones, $\Pi_t$ es el token previo. Produce query para atencion.
- **LSTM2 (Language LSTM)**: recibe $[\hat{v}_t, h_t^1]$ y genera logits sobre vocabulario.

Loss: cross-entropy durante training, fine-tuneable con **CIDEr optimization** via Self-Critical Sequence Training (SCST, Rennie 2017) para mejorar metrica especifica.

### 4. Modelo de VQA

Pregunta + imagen → respuesta.

- Pregunta → GRU sobre word embeddings → query vector $q$.
- Imagen → features de regiones $V$ (bottom-up).
- Atencion: $\alpha_i = \text{softmax}(W^T f_a([v_i, q]))$.
- Attended feature $\hat{v} = \sum \alpha_i v_i$.
- Joint representation: $h = f_q(q) \odot f_v(\hat{v})$.
- Output: clasificador multi-label sobre candidate answers.

Implementa **gated tanh** activations (Dauphin 2016) para mejor performance.

---

## Resultados experimentales

### Image Captioning (MSCOCO)

Resultados en Karpathy test split:

| Modelo | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|---|---|---|---|---|---|---|
| ResNet baseline (grid 7×7) | 74.5 | 33.4 | 26.1 | 54.4 | 105.4 | 19.2 |
| **Up-Down (este)** | **77.2** | **36.2** | **27.0** | **56.4** | **113.5** | **20.3** |
| Up-Down + CIDEr opt | 79.8 | 36.3 | 27.7 | 56.9 | **120.1** | 21.4 |

Mejora absoluta de **~2-3% en BLEU-4 y 8% en CIDEr** sobre el baseline grid. Breakdown por subcategorias (Tabla 2): **Up-Down mejora especialmente en objetos y atributos** -- exactamente lo que Faster R-CNN captura bien.

### VQA (VQA v2.0)

| Metric | Up-Down |
|---|---|
| Yes/No | 86.6% |
| Number | 48.6% |
| Other | 61.1% |
| **Overall** | **70.3%** |

**Gano el VQA Challenge 2017**, superando todos los demas submissions.

### Test server (MSCOCO)

- CIDEr: 117.9
- SPICE: 21.5
- BLEU-4: 36.9

Nuevo state-of-art en 2018.

---

## Por que importa hoy

- **Pre-computadas features**: el paper publica features pre-computadas de Faster R-CNN para MSCOCO y VQA. Esto se volvio un **recurso estandar** que muchos papers usaron para anos (2018-2021) sin tener que re-ejecutar Faster R-CNN.
- **Object-level attention**: el patron de atender sobre regiones detectadas es usado en **OSCAR**, **VinVL** (Microsoft), y modelos multimodales pre-Transformer.
- **Inspiro VinVL** (Zhang 2021): mejora el detector subyacente para obtener mejores features, CIDEr 141+.
- **Puente a vision-language moderna**: aunque los Transformers multimodales (CLIP, BLIP, LLaVA) usan patches uniformes mas que regiones, la intuicion de "objeto-como-unidad-de-atencion" persiste en arquitecturas mixtas.
- **Standard en 2017-2020 para VQA**: practicamente todo paper VQA de esa era usa Up-Down features + variantes de atencion.

---

## Limitaciones

- **Dos-fase training**: requiere Faster R-CNN preentrenado (en Visual Genome ideal, que es mas grande que ImageNet para este proposito). Entrenamiento no end-to-end.
- **Dependencia del detector**: si Faster R-CNN falla al detectar un objeto crucial, todo el pipeline sufre. Transformers multimodales que procesan pixels directos evitan esto.
- **Costo de pre-compute**: ejecutar Faster R-CNN en millones de imagenes es costoso; features deben almacenarse (cientos de GB).
- **Regiones fijas en inferencia**: para cada imagen, las regiones son computadas una vez; el modelo no puede re-detectar dinamicamente si la pregunta lo requiere.
- **Superado por Transformers**: Vision Transformers + atencion multi-modal (ViLT, BLIP, CLIP-based modelos) alcanzan performance similar o superior sin necesidad de detector separado.

---

## Notas y enlaces

- La **Figura 1** contrasta visualmente grid uniforme vs regiones de Faster R-CNN -- la imagen mas citada del paper.
- La **Figura 2** muestra ejemplos de regiones con atributos detectados ("blue sky", "green grass", "black jacket").
- Codigo y features pre-computadas: [peteanderson80/bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention).
- Follow-ups:
  - **Teney et al. 2018** (mismo grupo) "Tips and Tricks for Visual Question Answering" -- optimiza hyperparameters del modelo VQA.
  - **Zhang et al. 2021** "VinVL" -- mejora el detector base, CIDEr 141+.
  - **ViLT** (Kim 2021) -- vision-language Transformer sin detector separado.

Ver fundamentos: [Mecanismo de Atencion](/fundamentos/mecanismo-atencion) · [Redes Convolucionales](/fundamentos/redes-convolucionales) · [Transfer Learning](/fundamentos/transfer-learning) · [Sequence to Sequence](/fundamentos/seq2seq).
