---
title: "FaceNet: A Unified Embedding for Face Recognition and Clustering"
weight: 83
math: true
---

{{< paper-card
    title="FaceNet: A Unified Embedding for Face Recognition and Clustering"
    authors="Schroff, Kalenichenko, Philbin"
    year="2015"
    venue="CVPR 2015"
    pdf="/papers/facenet-schroff-2015.pdf"
    arxiv="1503.03832" >}}
Entrena un embedding compacto **128-D** (96 bytes si se cuantiza) en la esfera unitaria $L_2$-normalizada, optimizando **triplet loss** con online semi-hard mining. Reduce **30% el error** sobre el SOTA anterior en LFW (99.63% acc) y YouTube Faces DB (95.12%). El paper canónico de *metric learning* para faces y la base conceptual de cualquier pipeline contrastiva moderna (SimCLR, MoCo, ArcFace).
{{< /paper-card >}}

---

## Contexto

Antes de FaceNet, el SOTA en face recognition (DeepFace, DeepID2+) usaba **softmax sobre miles de identidades** + extraer el bottleneck como embedding. Esto era **indirecto** (el embedding era un subproducto, no el objetivo), **no generalizaba** a identidades nuevas y producía vectores de alta dimensión (~1000-4000). FaceNet propone entrenar **directamente la métrica de similaridad** en el espacio de embedding.

## Ideas principales

### Triplet Loss (Ec. 3)

$$
\mathcal{L} = \sum_{i=1}^N \left[ \|f(x^a_i) - f(x^p_i)\|_2^2 - \|f(x^a_i) - f(x^n_i)\|_2^2 + \alpha \right]_+
$$

Anchor $x^a$, positive $x^p$ (misma identidad), negative $x^n$ (otra identidad). Margen $\alpha = 0.2$. El operador hinge $[\cdot]_+$ pena solo los triplets que violan el margen — los fáciles no contribuyen gradiente.

### Online semi-hard negative mining

Hard negatives globales → colapso del modelo. Random negatives → muchos triviales. Solución:

$$
\|f(x^a_i) - f(x^p_i)\|^2 < \|f(x^a_i) - f(x^n_i)\|^2 < \|f(x^a_i) - f(x^p_i)\|^2 + \alpha
$$

Negative *semi-hard*: viola el margen pero está más lejos del anchor que el positive. Estable y efectivo.

Mini-batches estructurados: **~40 caras × 45 identidades = 1800 ejemplares**.

### L2 normalization

El embedding final $f(x) \in \mathbb{R}^{128}$ se proyecta a la esfera unitaria: $\|f(x)\|_2 = 1$. Hace que la métrica Euclidiana sea equivalente a la cosine similarity y estabiliza el entrenamiento.

### Arquitectura

Dos backbones experimentados:

- **NN1** — Zeiler & Fergus style, 140M params.
- **NN2** — Inception/GoogLeNet style, **7.5M params** (20× menos).

Ambos terminan con un FC layer → 128-D + $L_2$ norm. El backbone es intercambiable; lo importante es el loss.

### Downstream tasks gratis

Una vez aprendido el embedding:
- **Verification**: threshold sobre $\|f(x_1) - f(x_2)\|_2^2$.
- **Recognition**: k-NN.
- **Clustering**: k-means o agglomerative.

No hace falta re-entrenar para nuevas identidades.

## Resultados experimentales

### LFW (Labeled Faces in the Wild)

| Sistema | Acc % |
|---|---|
| DeepFace (Taigman 2014) | 97.35 |
| DeepID2+ (Sun 2015) | 99.47 |
| **FaceNet (fixed crop)** | 98.87 |
| **FaceNet + alignment** | **99.63** |

**Reducción de error del 30%** sobre DeepID2+.

### YouTube Faces DB

| Sistema | Acc % |
|---|---|
| DeepFace | 91.4 |
| DeepID2+ | 93.2 |
| **FaceNet** | **95.12** |

### Embedding dimensionality (Tabla 5)

| Dim | val rate |
|---|---|
| 64 | 86.8 |
| **128** | **87.9** |
| 256 | 87.7 |
| 512 | 85.6 |

128 es óptimo. Cuantizar a uint8 reduce el costo a **96 bytes/cara** — clave para indexar bases de datos a escala.

### Robustez

- JPEG quality ≥ 20: 81%+ val rate (vs 67% a Q10).
- Downscale hasta 80×80 px: 79%+ val rate.
- Datos: diminishing returns después de ~26M caras.

## Limitaciones reconocibles

1. **Datos privados**: 200M caras de Google, no público. Replicar requiere construir dataset comparable.
2. **Sesgo demográfico**: estudios posteriores (Buolamwini & Gebru 2018 — *Gender Shades*) muestran sesgos sistemáticos.
3. **Triplet selection costosa**: requiere PK samplers, no es training estándar.
4. **Sensible al margen**: superado por **angular margin losses** (SphereFace, ArcFace, CosFace).
5. **Solo cara**: aunque el método es agnóstico.

## Por qué importa hoy

**El paper más citado en face recognition** de la última década.

- **Triplet loss** se generalizó a Person Re-ID (Hermans 2017), image retrieval, Sentence-BERT.
- Es ancestro conceptual de **SimCLR, MoCo, BYOL** — self-supervised contrastive learning.
- **Sucesores en face**: CenterLoss, SphereFace, **ArcFace** (Deng 2019, SOTA actual con 99.83% LFW), CosFace.
- **Implementaciones**: OpenFace (CMU), face_recognition (Geitgey), InsightFace.
- **Aplicaciones industriales**: Google Photos, Facebook tagging, Apple Face ID, Azure, AWS Rekognition.

## Conexión con la clase 17

FaceNet aparece en los slides 55-56 (sección *Facial recognition*) como el ejemplo canónico de **triplet network**. El profesor introduce literalmente la ecuación

$$
L(f(I_1), f(I_2), f(I_3)) := \max\{0, m - |f(I_1) - f(I_3)| + |f(I_1) - f(I_2)|\}
$$

y muestra el diagrama anchor/positive/negative. Conecta pose con otras técnicas de la vida real (mezclar pose + tracking + face) y abre la discusión ética sobre vigilancia (slides 57-58).

## Notas y enlaces

- OpenFace (CMU): https://cmusatyalab.github.io/openface/
- face_recognition: https://github.com/ageitgey/face_recognition
- InsightFace (ArcFace, sucesor moderno): https://github.com/deepinsight/insightface
- Análisis interno con código PyTorch/TF/JAX en el repositorio del curso.
