---
title: "Pose Estimation (humana 2D)"
weight: 130
math: true
---

La **estimación de pose humana** consiste en identificar puntos o superficies de un cuerpo humano en una imagen o video. Es un problema fundacional de visión por computador con aplicaciones en deportes, salud, vigilancia, VR/AR, robótica y conducción autónoma. El campo ha pasado por tres paradigmas en la última década: **keypoints discretos**, **dense correspondence** y **vision transformers**, todos compartiendo conceptos transversales — top-down vs bottom-up, heatmaps Gaussianos, OKS/GPS como métricas, SMPL como modelo subyacente — que conviene aislar como referencia reutilizable.

---

## 1. Definición y taxonomía

Hay tres niveles de detalle en pose recognition, en orden creciente de información extraída:

1. **Keypoint detection** — predecir las coordenadas de un conjunto fijo de articulaciones (típicamente 17 en COCO: nariz, ojos, orejas, hombros, codos, muñecas, caderas, rodillas, tobillos). Es la representación más usada.

2. **Dense human pose** — mapear cada píxel del cuerpo humano a su correspondencia en la **superficie 3D** del modelo SMPL ([DensePose](/papers/densepose-guler-2018), Güler 2018). Representación mucho más rica, captura orientación, deformación, partes intermedias.

3. **3D body recovery** — recuperar shape + pose en el espacio paramétrico de [SMPL](/papers/smpl-loper-2015) o variantes: $(\vec\beta, \vec\theta)$. Reconstruye una malla completa del cuerpo, usable en gráficos y simulación física (HMR, VIBE, 4DHumans).

Esta sección se enfoca en los **dos primeros** — keypoints y dense correspondence en 2D — que es lo cubierto por la [Clase 17](/clases/clase-17).

## 2. La pregunta arbitraria de los 17 keypoints

¿Por qué 17? No hay razón biofísica. La elección viene de **MPII** (Andriluka 2014) y **COCO Keypoints** (Lin 2014), que se popularizaron como benchmarks. Otras elecciones existen:

| Dataset | Keypoints | Caso típico |
|---|---|---|
| COCO | 17 | Pose 2D general |
| MPII | 16 | Articulaciones tradicionales |
| AI Challenger | 14 | Pose 2D, dataset chino |
| Halpe | 136 | Cuerpo + cara + manos |
| LSP | 14 | Sports poses |
| OpenPose body+foot | 25 | Cuerpo + pies (para análisis biomecánico) |

El profesor Vergara enfatiza en la Clase 17 que **"no hay razón particular" para 17** — es una elección de diseño que parece razonable. Esto motiva alternativas como DensePose (cubre todo el cuerpo).

{{< concept-alert type="clave" >}}
La elección de keypoints es **arbitraria** y **lossy** — toda parametrización discreta del cuerpo pierde información. Para tareas donde la superficie importa (try-on virtual, body interaction, surface contact), se usan representaciones densas (DensePose) o paramétricas (SMPL).
{{< /concept-alert >}}

## 3. Top-down vs Bottom-up

Dos paradigmas dominantes para escenas multi-persona:

### 3.1 Top-down

```
Imagen → Person Detector (Faster R-CNN, etc.) → bbox por persona
                                              → crop por persona
                                              → Single-person pose estimator → keypoints
```

**Ejemplos**: Mask R-CNN keypoint head (He 2017), Simple Baselines (Xiao 2018), HRNet (Sun 2019), [ViTPose](/papers/vitpose-xu-2022) (Xu 2022), [DensePose](/papers/densepose-guler-2018) (Güler 2018).

**Pros**:
- Cada persona se procesa en su propia ventana — el modelo ve gran contexto local.
- Generalmente más preciso en alta resolución.
- Pipeline modular (detector y pose estimator se entrenan separados).

**Cons** (ver slide 48 de la Clase 17):
- Falla con **oclusiones**: si el detector no detecta a la persona, no hay pose.
- Falla con **bboxes intersectados**: dos personas adyacentes generan dos crops que contienen ambos cuerpos, confundiendo al estimator.
- Costo lineal en número de personas.

### 3.2 Bottom-up

```
Imagen → Detect body parts in image  → keypoints (todos, sin asignar a personas)
       → Associate parts             → poses agrupadas
```

**Ejemplos**: OpenPose / PAF (Cao 2017), Associative Embedding (Newell 2017), PersonLab (Papandreou 2018), [PifPaf](/papers/pifpaf-kreiss-2019) (Kreiss 2019), HigherHRNet (Cheng 2020).

**Pros**:
- **Robusto a oclusiones**: detecta keypoints individuales aunque la persona esté parcialmente oculta.
- **No depende del person detector** — no hay punto único de falla.
- **Costo constante** en número de personas (el inference time no depende del crowd).
- Mejor en **escenas crowded** y **baja resolución** (ver Tabla 1 del paper PifPaf: +18% AP a 321px).

**Cons**:
- La **asociación de partes es el cuello de botella** — pierde precisión cuando dos personas idénticas (uniformes deportivos) están muy cerca.
- Decoder secuencial (greedy decoding) difícil de batchear en GPU.
- Conceptualmente más complejo.

### 3.3 Comparación lado a lado

| Aspecto | Top-down | Bottom-up |
|---|---|---|
| Ejemplo canónico (Clase 17) | DensePose | PifPaf |
| Person detector | Requerido | No |
| Costo en N personas | $O(N)$ | $O(1)$ |
| Robusto a oclusión | No | Sí |
| Robusto a bbox intersectado | No | Sí |
| Precisión alta-res | Mejor | A la par |
| Precisión baja-res | Peor | Mejor |
| Caso de uso típico | Pose en escenarios curados (deportes, retratos) | Crowd analysis, self-driving |

Hoy ambos paradigmas siguen vigentes; la elección depende del dominio (ver discusión en slide 50).

## 4. Heatmaps Gaussianos: la representación dominante

Casi todos los métodos modernos predicen **heatmaps Gaussianos** en vez de regresar coordenadas directamente. Para cada keypoint $k$ y ground-truth $(x_k, y_k)$, el target es:

$$
H_k(i, j) = \exp\!\left( -\frac{(i - x_k)^2 + (j - y_k)^2}{2 \sigma^2} \right)
$$

con $\sigma$ típicamente 2-3 píxeles en la resolución del heatmap.

**Pérdida**: MSE pixel-wise. El modelo aprende a producir un blob Gaussiano en la ubicación correcta.

**Decoding**: argmax sobre el heatmap (o argmax + refinamiento sub-pixel vía DARK o UDP).

**Por qué no regresión directa de $(x, y)$**:
- La regresión directa colapsa el problema a un único número por keypoint — pierde la noción de incertidumbre espacial.
- Los heatmaps mantienen la distribución de probabilidad — útil para keypoints ambiguos.
- Empíricamente mejor convergencia y mejor AP en COCO.

Excepciones notables: DeepPose original (Toshev 2014, regresión directa) y métodos basados en transformers con cabezas tipo coord-conv.

## 5. Métricas

### 5.1 Object Keypoint Similarity (OKS) — COCO

Para una persona predicha $j$ y los keypoints ground-truth $\{(x_k, y_k)\}$:

$$
\text{OKS}_j = \frac{\sum_k \exp\!\left( -d_k^2 / (2 s^2 \kappa_k^2) \right) \delta(v_k > 0)}{\sum_k \delta(v_k > 0)}
$$

donde $d_k$ es la distancia euclidiana entre predicción y ground-truth para el keypoint $k$, $s$ es el área del bbox del cuerpo, $\kappa_k$ es una **constante per-keypoint** (más alta para ojos, más baja para caderas — refleja la dificultad típica), y $\delta(v_k > 0)$ indica visibilidad.

A partir de OKS se computan **AP y AR** a múltiples thresholds (0.5 a 0.95 en pasos de 0.05) — análogo a mAP en detección.

### 5.2 Geodesic Point Similarity (GPS) — DensePose

Análogo de OKS para correspondencia densa:

$$
\text{GPS}_j = \frac{1}{|P_j|} \sum_{p \in P_j} \exp\!\left( -\frac{g(i_p, \hat i_p)^2}{2 \kappa^2} \right)
$$

con $g(\cdot, \cdot)$ la **distancia geodésica** sobre la superficie SMPL (no Euclidiana en imagen). $\kappa = 0.255$ calibrado para que un error de ~30cm dé GPS = 0.5.

### 5.3 PCK / PCKh — métricas legacy

**Percentage of Correct Keypoints**: predicción correcta si está dentro de un threshold de píxeles (típicamente 0.5 × altura de la cabeza en PCKh) del ground-truth. Más simple pero menos informativa que OKS. Aún se usa en MPII.

## 6. Arquitecturas dominantes

### 6.1 Era CNN (2014-2021)

| Año | Modelo | Idea clave |
|---|---|---|
| 2014 | DeepPose | Regresión directa (x, y) — superado rápidamente |
| 2016 | Convolutional Pose Machines | Cascada de stages con context fusion |
| 2016 | Stacked Hourglass | Encoder-decoder repetido, refinamiento iterativo |
| 2017 | OpenPose / PAF | Bottom-up con Part Affinity Fields |
| 2017 | Mask R-CNN keypoints | Mask R-CNN con cabeza de 17 mascaras one-hot |
| 2018 | Simple Baselines | ResNet + 2-3 deconvs — sorprendentemente fuerte |
| 2018 | CPN, RMPE | Top-down con global+refine |
| 2019 | HRNet | Mantener alta-res en paralelo, fusiones |
| 2019 | PifPaf | Bottom-up con PIF + PAF + Laplace loss |

### 6.2 Era Transformer (2021-presente)

| Año | Modelo | Idea clave |
|---|---|---|
| 2021 | TokenPose, TransPose | CNN backbone + transformer decoder |
| 2021 | HRFormer | HRNet + transformer modules |
| 2022 | ViTPose | **ViT plain como backbone + decoder lightweight** — SOTA |
| 2023 | DWPose | Distillation de ViTPose-X |
| 2023 | ViTPose+ | Multi-task / multi-dataset / animal pose |

La frontera actual es **vision-language pose**: modelos que reciben imagen + descripción textual de la pose esperada, útiles para zero-shot category-agnostic pose.

## 7. Datasets de referencia

| Dataset | Imágenes | Keypoints | Caso | Año |
|---|---|---|---|---|
| MPII | 25K | 16 | Single-person, actividades diversas | 2014 |
| **COCO Keypoints** | 200K | 17 | Multi-person in-the-wild | 2014 |
| AI Challenger | 700K | 14 | Multi-person, gran escala | 2017 |
| CrowdPose | 20K | 14 | Específicamente crowded scenes | 2019 |
| DensePose-COCO | 50K (anotadas) | UV + 24 parts | Dense correspondence | 2018 |
| Halpe | 50K | 136 | Full-body (cuerpo + cara + manos) | 2020 |
| AGORA | sintético | SMPL params | 3D body recovery | 2021 |
| AMASS | mocap unificado | SMPL params | 3D body, ~14M frames | 2019 |

## 8. Conexiones con la Clase 17

Esta página complementa la [Clase 17 — Pose Recognition](/clases/clase-17). En particular:

- **Slide 18**: la idea de 17 keypoints estándar — Sección 2 acá.
- **Slide 19-20**: cabeza Mask R-CNN para keypoints, heatmaps — Sección 4 acá.
- **Slide 25**: la crítica de los 17 keypoints arbitrarios — abordada con DensePose (slide 26+) y revisitada en Sección 2 acá.
- **Slides 37-50**: top-down vs bottom-up — Sección 3 acá.
- **Slides 51-52**: ViTPose como SOTA moderno — Sección 6.2 acá.

## 9. Recursos relacionados

- [DensePose (Güler 2018)](/papers/densepose-guler-2018) — dense correspondence top-down.
- [PifPaf (Kreiss 2019)](/papers/pifpaf-kreiss-2019) — bottom-up moderno.
- [ViTPose (Xu 2022)](/papers/vitpose-xu-2022) — SOTA transformer.
- [SMPL (Loper 2015)](/papers/smpl-loper-2015) — modelo paramétrico subyacente.
- [FaceNet (Schroff 2015)](/papers/facenet-schroff-2015) — face recognition relacionado (slide 55).
- [Dense correspondence](/fundamentos/dense-correspondence) — fundamento de UV mapping.
- [Triplet loss](/fundamentos/triplet-loss) — base de la sección de facial recognition.
- [Mask R-CNN (He 2017)](/papers/mask-rcnn-he-2017) — backbone arquitectural de DensePose.
- [Faster R-CNN (Ren 2015)](/papers/faster-rcnn-ren-2015) — recap en la Clase 17 slides 12-17.
