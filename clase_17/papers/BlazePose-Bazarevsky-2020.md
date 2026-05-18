# BlazePose: On-device Real-time Body Pose Tracking

**Autores:** Valentin Bazarevsky, Ivan Grishchenko, Karthik Raveendran, Tyler Zhu, Fan Zhang, Matthias Grundmann (Google Research, Mountain View)
**Año:** 2020 (CVPR Workshop on Computer Vision for Augmented and Virtual Reality)
**arXiv:** 2006.10204
**Producto:** integrado en **MediaPipe Pose** (https://google.github.io/mediapipe/solutions/pose), librería open-source de Google para perception en dispositivos móviles.

---

## 1. Contexto histórico

Para 2020 el paisaje de pose estimation estaba dividido entre dos extremos:

- **Modelos de servidor** (OpenPose, PifPaf, HRNet, PersonLab): alta precisión, pero **inviables en mobile**. OpenPose en una CPU desktop de 20 cores corre a 0.4 fps. En un teléfono, ni siquiera intentable.
- **Modelos mobile genéricos** (PoseNet de Google, primer modelo en TensorFlow.js): rápidos pero con precisión muy baja, sin tracking entre frames.

El nicho que Google necesita resolver no es ranking en COCO — es **fitness, danza, sign language y AR en teléfonos**, donde:
- Hay **una sola persona** centrada en el frame.
- Se necesita >30 fps **en CPU de teléfono**.
- Hay **continuidad temporal** (video, no foto suelta).
- Las poses son **completas y desafiantes** (yoga, atletismo) — no peatones en una calle.

BlazePose nace como el **modelo de pose del stack MediaPipe**, integrado al detector-tracker pattern que Google ya usaba en BlazeFace (rostros) y BlazePalm (manos). Su valor no está en superar el estado del arte académico, sino en **redefinir el regime práctico** donde la pose se vuelve consumer-grade.

## 2. Contribución central

Cuatro aportes distintivos:

1. **Pipeline detector-tracker** (Fig. 1 del paper) que evita correr un detector pesado cada frame. Un detector ligero corre solo cuando se pierde la persona; mientras esté en frame, un tracker keypoint-based produce el ROI para el siguiente frame.

2. **Face-as-pose-proxy detection**: la cara es el ROI más estable y consistente del cuerpo, así que en vez de detectar la persona completa (donde NMS falla por la articulación), BlazePose **detecta la cara** y de ella infiere mid-hip, escala y rotación del torso. **Asume cabeza visible** — válido para AR/fitness, no para vigilancia.

3. **Heatmap-supervised regression híbrido** (Fig. 4 del paper): durante entrenamiento se usan heatmaps + offsets + regresión coordinada, pero **los heatmaps se descartan en inferencia**, reduciendo drásticamente el costo. La rama heatmap supervisa el embedding que alimenta a la rama regresión, con **stop-gradient** para que no se interfieran.

4. **Topología propia de 33 keypoints** (Fig. 3 del paper, vs. 17 de COCO o 18 de OpenPose) — añade puntos en cara (10), manos (12 nudillos) y pies (4) para soportar control gestual fino, fitness pose detection y handoff a BlazePalm/BlazeFace.

Resultado headline:
- **BlazePose Full** corre a **102 fps en una Pixel 2** (mid-tier 2020).
- **BlazePose Lite** corre a **312 fps en la misma Pixel 2**.
- OpenPose corre a **0.4 fps** en una **20-core desktop CPU**.
- Sobre el dataset Yoga in-house, BlazePose Full **supera a OpenPose** (84.5 vs. 83.4 PCK@0.2).

Es decir: **25-75× más rápido**, en un teléfono, con paridad o mejor precisión en el dominio target.

## 3. Arquitectura

### 3.1 Pipeline detector-tracker (Sección 2.1)

Inspirado en BlazePalm (hand tracking 2019) y BlazeFace facial geometry (2019):

```
Frame 1: Face detector (with pose alignment) ─► Pose Landmarks ─►
                                                                 │
                                                                 ▼ (region passed via alignment)
Frame 2:                                       Pose Landmarks ─► …
Frame 3:                                       Pose Landmarks ─► …
…
Frame N: Pose Landmarks says "no human" ─► back to Face detector
```

**Razón**: la mayoría del costo computacional está en localizar la persona. Si en frame $t$ ya sé dónde estaba y los keypoints predichos son consistentes, no necesito re-detectar — uso los keypoints anteriores para definir el ROI del frame $t+1$.

### 3.2 Person detector — la cara como proxy (Sección 2.2)

El problema con detectar bboxes de personas:
- **NMS falla con humanos**: bboxes muy articuladas (alguien con los brazos abiertos) tienen IoU ambiguo. Dos personas abrazándose se fusionan o se duplican.
- Los **detectores rígidos** (SSD, YOLO) están diseñados para objetos con pocos grados de libertad.

Solución: **detectar la cara** con **BlazeFace** (sub-millisecond detector mobile). De la bounding box facial, predicen además:
- **Mid-hip point** (punto medio entre caderas).
- **Radio del círculo circunscrito** al cuerpo completo.
- **Incline angle** (ángulo entre la línea mid-shoulder ↔ mid-hip).

Estos tres parámetros bastan para **alinear al hombre vitruviano** (Fig. 2): poner mid-hip en el centro, escalar para que el cuerpo entre en el cuadrado, rotar para verticalizar el eje hombros-caderas. Esa imagen alineada es lo que recibe el pose estimator.

**Asunción clave**: la cabeza es siempre visible. Para AR, fitness y sign language esto es razonable. Para CCTV o deportes con casco, no.

### 3.3 Topología de 33 keypoints (Sección 2.3)

| Rango | Keypoints | Razón |
|---|---|---|
| 0-10 | nose, eyes (inner/outer/middle), ears, mouth | Mismo schema que **BlazeFace** — handoff directo a face geometry. |
| 11-16 | shoulders, elbows, wrists | Estructura general — superset de COCO. |
| 17-22 | pinky, index, thumb knuckles (L+R) | Suficiente para detectar "mano abierta/cerrada" y handoff a **BlazePalm**. |
| 23-32 | hips, knees, ankles, heels, foot indices | Soporte para **yoga/fitness** (necesitas saber dónde apunta el pie). |

**Comparación**:
- COCO 17 = no incluye pies ni manos detalladas.
- OpenPose 18 = COCO + neck.
- Kinect = 25 keypoints, muy enfocado en torso.
- **BlazePose 33** = superset diseñado para handoff con otros modelos del stack MediaPipe.

### 3.4 Red neuronal (Fig. 4 del paper, Sección 2.5)

Es un **encoder-decoder híbrido** con tres salidas conjuntas durante entrenamiento:

```
Input RGB image 256×256×3
       │
       ▼ encoder (CNN convencional, downsampling)
       │
       ▼ 8×8×192 ← feature map más profundo
       │
       ▼ decoder (upsampling con skip connections)
       │
       ▼ 64×64×32
       │
       ├─► Heatmap branch:    64×64×99 (33 keypoints × 3: heatmap + offset_x + offset_y)
       │                       Solo usado en TRAIN, DESCARTADO en inferencia
       │
       └─► Regression branch: encoder adicional sobre el feature compartido
                              ↓
                              33 × 3 final layer → (x, y, visibility) por keypoint
```

**Detalles cruciales**:

- **Skip connections** desde el encoder al decoder en todas las resoluciones — preserva información de alta frecuencia.
- **Stop-gradient en la rama heatmap → regression**: los gradientes de la regresión **no** se propagan hacia atrás vía las features supervisadas por heatmap. Esto evita que la regresión "corrompa" el embedding optimizado para heatmap (que es más estable de aprender).
- **Pesos compartidos**: la regresión consume el feature map del decoder más una serie de convolutionales extra.
- **Visibilidad per-point**: salida adicional `visibility ∈ [0,1]` por keypoint que predice si está ocluido. Permite mostrar solo keypoints visibles en AR.

**Inspiración**: la estructura es una versión miniaturizada del **Stacked Hourglass** de Newell et al. (2016), pero apilando un encoder-decoder pequeño para heatmaps + un encoder de regresión, en vez de stacks completos de hourglass.

### 3.5 Alignment y augmentación de oclusiones (Sección 2.6)

Como el modelo **espera la entrada alineada**, hay augmentation agresiva pero controlada:

- **Rango de rotación limitado** — porque el detector ya entrega una pose verticalizada, no necesitas entrenar para rotaciones extremas.
- **±10% scale + shift augmentation** — para tolerar imprecisión del detector y movimientos entre frames.
- **Random colored rectangles** sobrepuestos en training — simulan oclusiones (manos, objetos, otra persona pasando enfrente). Esto entrena al modelo para predecir keypoints **invisibles** consistentemente con la pose.

**Resultado**: el modelo predice puntos ocluidos basándose en el resto de la pose. Fig. 5 muestra detección razonable de un cuerpo parcialmente fuera de frame (solo torso visible).

### 3.6 Dataset (Sección 2.4)

- **60K imágenes** con 1-2 personas en poses comunes.
- **25K imágenes** con una sola persona en **fitness/yoga**.
- Todas anotadas por humanos.
- Restricción: solo se anotan casos donde hombros + caderas son inferibles con confianza (porque alignment depende de eso).

**Sesgo**: el dataset está dominado por escenas con cabeza visible y una persona prominente. No generaliza bien a crowds, vigilancia o multi-persona.

## 4. Ejemplo de uso en el laboratorio (lab 17 IA UC)

El lab usa el wrapper Python oficial `mediapipe`:

```python
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Inferencia (espera RGB numpy)
results = pose.process(image_rgb)
# results.pose_landmarks.landmark : iterable de 33 NormalizedLandmark
#   cada uno con .x, .y, .z, .visibility, todos normalizados a [0, 1]
```

**Diferencia importante con OpenPose/PifPaf en el lab**: BlazePose es **single-person only**. Si hay 4 personas en la imagen del dataset Stanford 40 (lo que ocurre en `playing_guitar` con audiencia), MediaPipe devuelve solo una pose — la más prominente según el detector facial.

Por esta razón, **el lab 17 NO usa BlazePose para entrenar el clasificador MLP**. Aparece solo como demo en la celda 14 para mostrar la diversidad de outputs entre las 3 librerías. El profesor lo descarta para la comparación porque:
1. No detecta múltiples personas — incompatibles con muchas imágenes del dataset.
2. La topología de 33 puntos no es comparable head-to-head con los 17-18 de OpenPose/PifPaf sin reducción manual.

**Modo `static_image_mode=True`** (usado en el lab): activa el detector facial en **cada frame**, **desactivando el tracker**. Esto es lo que quieres para fotos sueltas pero pierde 5-10× de velocidad respecto al modo video.

## 5. Experimentos clave

### 5.1 Comparación con OpenPose (Tabla 1 del paper)

| Modelo | FPS | AR dataset PCK@0.2 | Yoga dataset PCK@0.2 |
|---|---|---|---|
| OpenPose (body only)¹ | 0.4 | **87.8** | 83.4 |
| **BlazePose Full**² | **102** | 84.1 | **84.5** |
| **BlazePose Lite**² | **312** | 79.6 | 77.6 |

¹ Intel i9-7900X, 20 cores.
² Pixel 2, single core, XNNPACK backend.

Lecturas:
- En **escenas AR genéricas**, OpenPose todavía gana en precisión (87.8 vs. 84.1) — su modelo es más grande, entrena en datasets más diversos.
- En **yoga/fitness**, BlazePose Full **supera** a OpenPose (84.5 vs. 83.4). Es el regime donde más se entrena.
- BlazePose Full es **255× más rápido** que OpenPose en hardware mucho más débil.

### 5.2 Tamaños de modelo

| Modelo | MFlops | Parámetros |
|---|---|---|
| BlazePose Full | 6.9 | 3.5M |
| BlazePose Lite | 2.7 | 1.3M |
| OpenPose | ~160 | 53M+ |

BlazePose Full es **15× más pequeño en parámetros** y **23× más liviano en flops** que OpenPose body-only.

### 5.3 Baseline humano

Dos anotadores re-anotaron el AR dataset independientemente — **PCK@0.2 humano vs. humano = 97.2**. Es decir, hay un ceiling humano del que BlazePose Full está a ~13 puntos. Espacio para mejora futura.

## 6. Limitaciones reconocidas

1. **Single-person**: la cara como proxy de detección **funciona solo para una persona** dominante. Multi-persona requiere correr el pipeline N veces (no escalable).
2. **Cabeza visible**: asunción dura. Falla cuando la persona da la espalda, está acostada boca abajo, o lleva casco/máscara.
3. **Sin 3D nativo** en el paper original (aunque MediaPipe v2 añadió un keypoint Z aproximado).
4. **Sin tracking de identidad**: si hay 2 personas y la dominante cambia, BlazePose "salta" sin avisar.
5. **Dataset privado**: los autores reportan resultados sobre datasets in-house de 1000 imágenes, no sobre benchmarks reproducibles públicos (COCO, MPII). Comparación con OpenPose se limita a métricas que ellos eligen.
6. **PCK@0.2 sobre torso size** es métrica menos estricta que la OKS de COCO — los números de PCK no son comparables 1:1 con los de papers en COCO.

## 7. Impacto y legado

- **MediaPipe Pose** es **la** API de pose estimation default en mobile. Está integrada en Google Fit, YouTube Shorts, Snapchat AR Effects, TikTok efectos, Instagram Reels, miles de fitness apps.
- **Inspiró BlazePose GHUM 3D** (2021) — extensión a 3D usando el modelo paramétrico GHUM.
- **MoveNet** (Google TF.js, 2021) toma muchas ideas del pipeline detector-tracker pero las re-diseña para web.
- **Patron del pipeline**: el "detector ligero + tracker keypoints + face-as-anchor" se generalizó a otras tareas (hand tracking, animal pose, sign language).
- **Crítica académica**: el paper tiene **4 páginas** y muy pocas comparaciones rigurosas — es deliberadamente un **engineering report**, no un paper SOTA. Sin embargo su impacto en producto excede a la mayoría de SOTA papers.

## 8. Conexión con la clase 17

BlazePose representa **la cara mobile/producción de la pose estimation**, complementaria al espíritu académico de OpenPose y PifPaf. En el lab 17:

- Aparece como **demo de la celda 14** para mostrar que MediaPipe es la opción más accesible (un solo `pip install` y dos líneas de código vs. la pesadilla de OpenPose).
- **No participa en la comparación de modelos** porque su limitación a single-person la deja fuera del scope del experimento.
- Pero es el modelo que el alumno **más probablemente usará en producción** si hace una app de fitness o AR.

Pedagógicamente, BlazePose es el contraste perfecto:
- OpenPose: investigación, multi-persona, top precisión, costo enorme.
- PifPaf: investigación, bottom-up moderno, balance precisión/velocidad.
- **BlazePose: producto, single-person, precisión razonable, velocidad obscena**.

Cross-links:
- [[OpenPose-Cao-2017]] — el baseline que BlazePose deliberadamente compara y supera en velocidad.
- [[PifPaf-Kreiss-2019]] — la alternativa bottom-up moderna usada como tercer competidor del lab.
- [[DensePose-Guler-2018]] — top-down con mesh denso, otro extremo del trade-off.
- [[ViTPose-Xu-2022]] — la nueva generación SOTA con backbones Transformer.
- [[fundamentos/pose-estimation.md]] — bottom-up vs. top-down vs. detector-tracker.

## 9. Enlaces

- Paper: https://arxiv.org/abs/2006.10204
- MediaPipe Pose docs: https://google.github.io/mediapipe/solutions/pose
- Modelo TensorFlow.js: https://github.com/tensorflow/tfjs-models/tree/master/pose-detection
- Demo web interactiva: https://mediapipe-studio.webapps.google.com/demo/pose_landmarker
