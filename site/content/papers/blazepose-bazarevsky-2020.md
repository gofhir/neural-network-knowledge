---
title: "BlazePose: On-device Real-time Body Pose Tracking"
weight: 82
math: true
---

{{< paper-card
    title="BlazePose: On-device Real-time Body Pose Tracking"
    authors="Bazarevsky, Grishchenko, Raveendran, Zhu, Zhang, Grundmann (Google Research)"
    year="2020"
    venue="CVPR Workshop on Computer Vision for Augmented and Virtual Reality"
    pdf="/papers/blazepose-bazarevsky-2020.pdf"
    arxiv="2006.10204" >}}
Modelo de pose **single-person tailored para mobile**: 33 keypoints corriendo a **>30 fps en Pixel 2**. Introduce el **detector-tracker pattern** (cara como proxy de detección + tracker keypoint-based) y la **arquitectura heatmap-supervised regression** con stop-gradient. Producto distribuido como **MediaPipe Pose**, hoy ubicuo en fitness apps, AR y sign language tracking.
{{< /paper-card >}}

---

## Contexto

Para 2020 el paisaje de pose estimation estaba dividido entre dos extremos sin punto medio práctico:

- **Modelos de servidor** (OpenPose, PifPaf, HRNet, PersonLab): alta precisión pero **inviables en mobile**. OpenPose corre a 0.4 fps en CPU desktop de 20 cores.
- **Modelos mobile genéricos** (PoseNet de Google, primer modelo TF.js): rápidos pero **precisión muy baja**, sin tracking temporal.

El nicho de Google: **fitness, danza, sign language y AR en teléfonos**, donde:

- Hay **una sola persona** centrada en el frame.
- Se necesita **>30 fps en CPU de teléfono**.
- Hay **continuidad temporal** (video, no foto suelta).
- Las poses son **desafiantes pero predecibles** (yoga, atletismo).

BlazePose nace como el **modelo de pose de MediaPipe**, integrado al stack detector-tracker que Google ya usaba en BlazeFace y BlazePalm. **No compite por SOTA académico** — redefine el regime práctico para producto.

## Ideas principales

### Pipeline detector-tracker

Evita correr un detector pesado cada frame:

```
Frame 1: Face detector ─► Pose Landmarks ─► (alignment)
                                            │
                                            ▼
Frame 2:                  Pose Landmarks ─► …
…
Frame N: Pose Landmarks says "no human" ─► back to Face detector
```

**Razón**: la mayoría del costo está en localizar la persona. Si el frame previo dio keypoints consistentes, el ROI del frame actual se deriva de allí sin re-detectar.

### Face-as-pose-proxy detection

El problema con detectar bboxes de personas:

- **NMS falla con humanos**: bboxes muy articuladas tienen IoU ambiguo.
- Los **detectores rígidos** (SSD, YOLO) asumen pocos grados de libertad.

Solución: **detectar la cara** con BlazeFace (sub-millisecond mobile detector). De la bbox facial predicen:

- **Mid-hip point** (punto medio entre caderas).
- **Radio del círculo circunscrito** al cuerpo completo.
- **Incline angle** (ángulo línea mid-shoulder ↔ mid-hip).

Estos tres parámetros alinean al "hombre vitruviano": mid-hip al centro, escala correcta, rotación verticalizada. La imagen alineada es lo que recibe el pose estimator.

**Asunción dura**: cabeza siempre visible. Válida para AR/fitness, no para vigilancia o personas dándole la espalda a la cámara.

### Topología de 33 keypoints

| Rango | Keypoints | Función |
|---|---|---|
| 0-10 | nose, eyes (inner/outer/middle), ears, mouth | Mismo schema BlazeFace — handoff a face geometry |
| 11-16 | shoulders, elbows, wrists | Superset COCO |
| 17-22 | pinky, index, thumb knuckles (L+R) | Suficiente para "mano abierta/cerrada", handoff a BlazePalm |
| 23-32 | hips, knees, ankles, heels, foot indices | Soporte para yoga/fitness |

**Diseñado para handoff entre modelos del stack MediaPipe** — no para benchmark COCO. Por eso elige 33 puntos: superset que conecta con BlazeFace y BlazePalm.

### Arquitectura híbrida — heatmap supervisa, regression infiere

Encoder-decoder con dos cabezas durante entrenamiento:

```
Input RGB 256×256×3
   │
   ▼ encoder (downsampling)
   │
   ▼ 8×8×192
   │
   ▼ decoder (upsampling + skip connections)
   │
   ▼ 64×64×32
   │
   ├─► Heatmap branch:    64×64×99 (33 KP × 3: heatmap + offset_x + offset_y)
   │                       Solo usada en TRAIN, DESCARTADA en inferencia
   │
   └─► Regression branch: encoder adicional → 33×3 final (x, y, visibility)
```

**Stop-gradient** clave: los gradientes de la regresión **no** se propagan hacia las features supervisadas por heatmap. Esto evita corromper el embedding heatmap-trained (más estable de aprender) con el ruido de la regresión.

**Visibilidad per-point**: el modelo predice explícitamente $\text{visibility} \in [0, 1]$ por keypoint — un classifier auxiliar que indica si el punto está ocluido. Permite mostrar solo keypoints confiables en AR.

### Augmentation de oclusiones

Como el modelo **espera entrada alineada** (gracias al detector), el rango de augmentation es estrecho:

- **Rotación limitada** — la entrada ya está verticalizada.
- **±10% scale + shift** para tolerar imprecisión del detector entre frames.
- **Random colored rectangles** simulando oclusiones (manos, objetos, otra persona pasando).

Esto entrena al modelo a **inferir keypoints invisibles consistentes con la pose**. Fig. 5 muestra detección razonable de upper-body con piernas fuera de frame.

## Resultados clave

### Comparación con OpenPose (Tabla 1 del paper)

| Modelo | FPS | AR dataset PCK@0.2 | Yoga dataset PCK@0.2 |
|---|---|---|---|
| OpenPose (body only)¹ | 0.4 | **87.8** | 83.4 |
| **BlazePose Full**² | **102** | 84.1 | **84.5** |
| **BlazePose Lite**² | **312** | 79.6 | 77.6 |

¹ Intel i9-7900X, 20 cores
² Pixel 2, single core, XNNPACK backend

Lecturas:

- En **AR genérico** OpenPose gana (87.8 vs. 84.1) — su modelo es más grande, entrena con datasets más diversos.
- En **yoga/fitness**, BlazePose Full **supera** a OpenPose (84.5 vs. 83.4) — su dominio target.
- BlazePose Full es **255× más rápido** sobre hardware mucho más débil.

### Tamaños de modelo

| Modelo | MFlops | Parámetros |
|---|---|---|
| BlazePose Full | 6.9 | 3.5M |
| BlazePose Lite | 2.7 | 1.3M |
| OpenPose | ~160 | 53M+ |

**15× menos parámetros, 23× menos flops** que OpenPose body-only.

### Baseline humano

Dos anotadores re-anotaron el AR dataset independientemente: **PCK@0.2 humano vs. humano = 97.2**. BlazePose Full está a ~13 puntos del ceiling humano — espacio para mejora futura.

## Limitaciones reconocibles

1. **Single-person**: la cara como proxy **solo funciona para una persona** dominante. Multi-persona requiere correr el pipeline N veces (no escalable).
2. **Cabeza visible**: asunción dura. Falla cuando la persona da la espalda, está acostada boca abajo, o lleva casco/máscara.
3. **Sin 3D nativo** en el paper original (MediaPipe v2 añadió Z aproximado).
4. **Sin tracking de identidad**: si hay 2 personas y la dominante cambia, BlazePose "salta" sin avisar.
5. **Dataset privado**: comparación con OpenPose se limita a datasets in-house de 1000 imágenes. **No reproducible públicamente**.
6. **PCK@0.2 sobre torso size** es métrica menos estricta que OKS de COCO — números no comparables 1:1 con SOTA académico.

## Impacto y legado

- **MediaPipe Pose** es **la** API default de pose mobile. Integrada en Google Fit, YouTube Shorts, Snapchat AR Effects, TikTok efectos, Instagram Reels, miles de fitness apps.
- **BlazePose GHUM 3D** (2021) extiende a 3D usando el modelo paramétrico GHUM.
- **MoveNet** (Google TF.js, 2021) toma ideas del detector-tracker pattern y las re-diseña para web.
- **Patrón "detector ligero + tracker keypoint + face-as-anchor"** se generalizó a hand tracking, animal pose, sign language.
- **Crítica académica**: el paper tiene 4 páginas y pocas comparaciones rigurosas — deliberadamente un **engineering report**, no un paper SOTA. Pero su impacto en producto excede a la mayoría de papers académicos.

## Conexión con el laboratorio

En el [Lab 17](/laboratorios/lab-17), MediaPipe/BlazePose aparece como **demo del paradigma single-person mobile** (celda 14):

```python
import mediapipe as mp
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
results = pose.process(image_rgb)
```

Tres líneas vs. cuatro objetos de OpenPifPaf. La diferencia de API refleja la filosofía:

- **OpenPifPaf**: research-grade, expone componentes (model, decoder, processor).
- **MediaPipe**: producto, monolítico, opaco.
- **OpenPose**: research-grade legacy, fricción de instalación máxima.

BlazePose **no participa en la comparación cuantitativa** del lab porque su limitación a single-person la deja fuera del scope de Stanford 40 (con escenas multi-persona en background). Pero es el modelo que el alumno más probablemente usaría en producción mobile.

Cross-links:

{{< cards >}}
  {{< card link="/laboratorios/lab-17" title="Lab 17 - Pose Recognition" subtitle="A/B test PifPaf vs. OpenPose + demo de MediaPipe" icon="academic-cap" >}}
  {{< card link="/papers/openpose-cao-2017" title="OpenPose (Cao 2017)" subtitle="El baseline que BlazePose deliberadamente compara y supera en velocidad" icon="document-text" >}}
  {{< card link="/papers/pifpaf-kreiss-2019" title="PifPaf (Kreiss 2019)" subtitle="La alternativa bottom-up moderna" icon="document-text" >}}
  {{< card link="/fundamentos/pose-estimation" title="Fundamento: Pose Estimation 2D" subtitle="Bottom-up vs. top-down vs. detector-tracker" icon="book-open" >}}
  {{< card link="/clases/clase-17" title="Clase 17 - Pose Recognition" subtitle="Recorrido teórico" icon="academic-cap" >}}
{{< /cards >}}
