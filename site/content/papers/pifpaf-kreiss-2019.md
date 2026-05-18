---
title: "PifPaf: Composite Fields for Human Pose Estimation"
weight: 81
math: true
---

{{< paper-card
    title="PifPaf: Composite Fields for Human Pose Estimation"
    authors="Kreiss, Bertoni, Alahi"
    year="2019"
    venue="CVPR 2019"
    pdf="/papers/pifpaf-kreiss-2019.pdf"
    arxiv="1903.06593" >}}
Método **bottom-up** para multi-person 2D pose estimation pensado para self-driving y robots sociales: opera bien a **baja resolución (30-90 px de altura)** y escenas multitudinarias con oclusiones. Introduce **Part Intensity Field (PIF)** + **Part Association Field (PAF)** como *composite fields* con regresión sub-píxel, y la **Laplace loss** para incertidumbre aprendida. +18% AP sobre OpenPose a 321 px y on-par con SOTA a alta resolución, 32% más rápido.
{{< /paper-card >}}

---

## Contexto

En 2019, top-down (Mask R-CNN, CPN, SimpleBaseline) dominaba **alta resolución**, mientras bottom-up (OpenPose, Associative Embedding, PersonLab) ofrecía mejor recall en escenas crowded. **Ningún método** funcionaba bien en el régimen crítico para **autonomous driving**: peatones de 30-90 px de altura, con oclusiones constantes entre cuerpos cercanos. OpenPose perdía recall por su PAF discreto anclado a píxeles enteros del feature map; Mask R-CNN se confundía cuando dos bboxes de peatones se intersectaban.

## Ideas principales

### Part Intensity Field (PIF)

Para cada keypoint $k$ y posición $(i, j)$ del feature map, predice:

$$
\mathbf{p}_k^{ij} = \{p_c^{ij},\ p_x^{ij},\ p_y^{ij},\ p_b^{ij},\ p_\sigma^{ij}\}
$$

- $p_c$ — confianza (sigmoid).
- $(p_x, p_y)$ — offset vectorial a la posición precisa del keypoint.
- $p_b$ — spread Laplace para la regresión.
- $p_\sigma$ — escala del keypoint en píxeles.

**Mapa de confianza fusionado** (Ec. 1) que recupera precisión sub-píxel a partir del feature map cuantizado:

$$
f(x, y) = \sum_{ij} p_c^{ij} \cdot \mathcal{N}(x, y \mid p_x^{ij}, p_y^{ij}, p_\sigma^{ij})
$$

con $\mathcal{N}$ una Gaussiana no normalizada.

### Part Association Field (PAF)

Generaliza los PAFs de OpenPose. Para cada conexión $(k_1, k_2)$ y posición $(i, j)$:

$$
\mathbf{a}^{ij} = \{a_c^{ij},\ a_{x_1}^{ij}, a_{y_1}^{ij},\ a_{x_2}^{ij}, a_{y_2}^{ij},\ a_{b_1}^{ij},\ a_{b_2}^{ij}\}
$$

**Clave**: el origen del vector es **flotante** (mid-range offset learned), no anclado al centro de la celda. Esto resuelve dos personas adyacentes sin colisión de anotaciones.

### Adaptive Regression — Laplace loss (Ec. 2)

$$
L = \frac{|x - \mu|}{b} + \log(2b)
$$

Log-verosimilitud negativa de una Laplace con escala $b$ aprendida. El modelo aprende a relajar $b$ para predicciones en cuerpos grandes y a apretarla para cuerpos pequeños — *learned uncertainty*.

Ablation (Tabla 3):

| Loss | AP | AP^M | AP^L |
|---|---|---|---|
| vanilla $L_1$ | 41.7 | 26.5 | 62.5 |
| **Laplace** | **45.1** | **31.4** | 64.0 |
| **Laplace (b en decoder)** | **45.5** | 31.4 | **64.9** |

+3.5 AP con solo cambiar el loss.

### Greedy decoder

1. **Seed**: argmax del mapa de confianza fusionado.
2. **BFS por el esqueleto**: desde el seed, sigue los PAFs computando score:

$$
s(\mathbf{a}, \vec{x}) = a_c \cdot \exp\!\left(-\frac{\|\vec{x} - \vec{a}_1\|}{b_1}\right) \cdot f_2(a_{x_2}, a_{y_2})
$$

3. **Reverse-match check**: verifica que el PAF reverso sea consistente.
4. **NMS dinámico** con radio = función de $p_\sigma$.

## Resultados experimentales

### Low-resolution COCO (321 px lado mayor — emula self-driving)

| Método | AP | AP^M | AP^L | AR |
|---|---|---|---|---|
| Mask R-CNN (re-entrenado) | 41.6 | 28.2 | 59.8 | 49.0 |
| OpenPose | 37.6 | 25.0 | 55.3 | 43.9 |
| **PifPaf** | **50.0** | **35.9** | **69.7** | **55.0** |

+8 AP sobre Mask R-CNN, +12 AP sobre OpenPose en el régimen donde más importa.

### High-resolution COCO

| Método | AP |
|---|---|
| Mask R-CNN | 63.1 |
| OpenPose | 61.8 |
| PersonLab (single-scale) | 66.5 |
| **PifPaf (single-scale)** | **66.7** |

On-par con SOTA bottom-up, **32% más rápido** que PersonLab (240ms vs 355ms).

### Market-1501 Re-ID cross-domain

Sin re-entrenar, sobre crops 64×128:
- Mask R-CNN: 43% poses correctas.
- **PifPaf: 96%**.

## Limitaciones reconocibles

1. **No optimizado para alta resolución**: solo *a la par* con PersonLab, no significativamente mejor.
2. **Decoder secuencial**: difícil de batchear en GPU; ~70% del tiempo total.
3. **17 keypoints COCO fijos**: misma limitación topológica que el resto.
4. **NMS dinámico depende de $\sigma$ aprendido**: si es ruidoso, puede fusionar poses cercanas.
5. **Sin temporal**: tracking requiere postprocesamiento separado.

## Por qué importa hoy

- **openpifpaf** (https://github.com/vita-epfl/openpifpaf) sigue activo desde 2019, con soporte para animales, vehículos, hands, tracking.
- La idea **Laplace loss + uncertainty aprendida** se generalizó: DETR-uncertainty, object pose 6D, ...
- *Composite Fields* aparece en MoveNet (Google MediaPipe Pose), Lightweight OpenPose, BlazePose.
- Inspiró **HigherHRNet** (CVPR 2020), **DEKR** (CVPR 2021), **CID** (CVPR 2022).
- Pose de peatones para autonomous driving es ahora estándar en stacks de conducción.

## Conexión con la clase 17

PifPaf es el ejemplo **canónico bottom-up** que el profesor presenta en los slides 37-46. Concretiza las dos componentes (PIF + PAF) y justifica la sección *"¿Por qué Bottom-up > Top-down?"* — el caso del baseball con oclusión (slide 48) es exactamente donde PifPaf brilla.

## Notas y enlaces

- Código: https://github.com/vita-epfl/openpifpaf
- Demo: https://openpifpaf.github.io/
- Análisis interno con código PyTorch/TF/JAX en el repositorio del curso.
