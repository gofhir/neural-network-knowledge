---
title: "DensePose: Dense Human Pose Estimation In The Wild"
weight: 80
math: true
---

{{< paper-card
    title="DensePose: Dense Human Pose Estimation In The Wild"
    authors="Güler, Neverova, Kokkinos"
    year="2018"
    venue="CVPR 2018"
    pdf="/papers/densepose-guler-2018.pdf"
    arxiv="1802.00434" >}}
Mapea cada píxel humano de una imagen RGB a la superficie 3D del cuerpo (modelo SMPL). Introduce **COCO-DensePose** (50K personas con ~5M correspondencias manuales), una pipeline de anotación en dos etapas y **DensePose-RCNN** que predice $(c, U, V)$ por píxel a 20-26 fps. Define el estándar moderno de *dense human pose estimation* y es la representación canónica de aproximación **top-down** discutida en la clase.
{{< /paper-card >}}

---

## Contexto

Antes de 2018 la estimación de pose se reducía a **17 keypoints** discretos (COCO Keypoints, MPII). El cuerpo humano es realmente una **superficie continua de ~6890 vértices** (la malla SMPL). DensePose es el primer trabajo que cierra esa brecha *en imágenes reales* (no sintéticas como SURREAL), entregando un mapa pixel-a-superficie aprendido por una CNN region-based.

## Ideas principales

### Parametrización del cuerpo

El cuerpo se divide en **24 partes** semánticas (cabeza, torso frontal/dorsal, brazos sup/inf, etc.). Cada parte se desenvuelve a un plano 2D vía:

- **Cabeza, manos, pies**: UV fields del modelo SMPL.
- **Resto**: *multidimensional scaling (MDS)* sobre distancias geodésicas del mesh, partiendo extremidades en frontal/dorsal y sup/inf.

Cada vértice del cuerpo queda etiquetado con $(c, u, v)$ donde $c \in \{1..24\}$ es la parte y $(u, v) \in [0,1]^2$ es la coordenada intra-parte.

### Pipeline de anotación en dos etapas

1. **Task 1 — segmentación**: el anotador delinea 14 regiones del cuerpo.
2. **Task 2 — correspondencia**: para ~14 puntos sorteados por k-means en cada parte, se le muestran 6 vistas pre-renderizadas del cuerpo SMPL y el anotador hace clic en el punto correspondiente.

Esta indirección clave evita pedir "rotar superficie 3D para cazar vértices". Total: **50K personas × ~100-150 puntos = ~5M correspondencias**.

### Métrica GPS (Geodesic Point Similarity)

Análoga a OKS para keypoints:

$$
\text{GPS}_j = \frac{1}{|P_j|} \sum_{p \in P_j} \exp\!\left(\frac{-g(i_p, \hat{i}_p)^2}{2\kappa^2}\right)
$$

con $g(\cdot, \cdot)$ la distancia geodésica sobre el mesh SMPL y $\kappa = 0.255$ (un error de ~30cm da GPS = 0.5).

### Arquitectura DensePose-RCNN

Es **Mask R-CNN** con una rama adicional. Sobre los features RoIAlign:

- 8 capas Conv 3×3 + ReLU (512 canales).
- **Cabeza 1 (clasificación)**: $H \times W \times 25$ logits (background + 24 partes), cross-entropy.
- **Cabeza 2 (regresión UV)**: $H \times W \times 48$ (24 partes × 2 coords), smooth L1 — **solo aplica donde la parte es correcta**.

Backbone: ResNet-50/101-FPN.

### Distillation-based ground-truth interpolation

Solo ~150 píxeles/persona están anotados. Solución:

1. Entrenar una *teacher network* FCN con la supervisión escasa.
2. Inferir predicciones de la teacher sobre **toda la región humana**.
3. Usar esas predicciones inferidas como supervisión densa para entrenar el modelo final (restringido a foreground).

Ganancia: +5-7 puntos AUC consistentes.

### Cascading multi-task

Una segunda pasada recibe los features RoIAlign + outputs de las ramas de máscara y keypoints. Esto inyecta contexto multi-tarea y suma ~+3 AP.

## Resultados experimentales

### Single-person (vs SMPLify, Bogo et al. 2016)

| Método | AUC₁₀ (all) | AUC₃₀ (all) | Tiempo |
|---|---|---|---|
| SMPLify-14 | 0.099 | 0.19 | 60-200s |
| **DensePose (FCN)** | **0.378** | **0.614** | **0.04s** |
| Human (upper bound) | 0.563 | 0.835 | — |

~4× la precisión, **1500× más rápido**.

### Multi-person COCO `minival` (Tabla 1, ResNet-50)

| Variante | AP | AP₅₀ | AR |
|---|---|---|---|
| DensePose (base) | 51.0 | 83.5 | 60.1 |
| + masks (multi-task) | 51.9 | 85.5 | 61.1 |
| + keypoints (multi-task) | 52.8 | 85.6 | 62.6 |
| + cascading + keypoints | **55.8** | 87.5 | **63.9** |

## Limitaciones reconocibles

1. **Brecha con humano**: 0.390 AUC₁₀ vs 0.563 humano. Margen para mejorar.
2. **Cuerpo único**: SMPL tiene una sola topología — no captura particularidades anatómicas (amputaciones, malformaciones).
3. **Top-down**: hereda problemas de Mask R-CNN cuando bboxes se intersectan — exactamente el punto que critica [PifPaf](/papers/pifpaf-kreiss-2019).
4. **Anotación behind clothes**: asume el cuerpo desnudo subyacente — funciona razonable con vestidos y abrigos pero no es matemáticamente perfecto.
5. **Costo de anotación**: 5M correspondencias manuales no escalan a otras categorías.

## Por qué importa hoy

DensePose definió el estándar de *dense human correspondence* en imágenes reales y abrió múltiples líneas:

- **Virtual try-on**: la representación UV permite mapear texturas de ropa al cuerpo.
- **DensePose-Track** (Neverova 2019): extensión temporal.
- **HMR/VIBE/4DHumans**: pose 3D end-to-end fitting SMPL, usan DensePose como supervisión auxiliar.
- **Continuous Surface Embeddings** (Neverova 2020): generaliza la idea a animales.

Es ahora un baseline de referencia en Detectron2 y la representación $(c, U, V)$ + SMPL es la *lingua franca* del cuerpo humano en visión por computador.

## Conexión con la clase 17

DensePose es el ejemplo **canónico top-down** que el profesor Vergara presenta en los slides 26-36. Muestra que se puede ir más allá de 17 keypoints — el cuerpo es una superficie. Su debilidad bajo oclusión (bboxes intersectados) motiva la siguiente sección de la clase: **PifPaf** y los métodos bottom-up.

## Notas y enlaces

- Project page: http://densepose.org
- Código (Detectron2): https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose
- Análisis interno (5500 palabras + código PyTorch/TF/JAX) en el repositorio del curso.
