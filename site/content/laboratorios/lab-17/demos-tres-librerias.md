---
title: "Demos sobre la misma imagen"
weight: 20
---

Tras instalar las tres librerías, el lab corre **inferencia sobre una sola imagen base** para construir intuición visual sobre cómo dibujan los keypoints. El compromiso metodológico explícito del profesor:

> *"Para cada una de las librerías/modelos vamos a hacer la detección sobre la misma imagen."*

Sin esa restricción, comparar outputs sería trampa. **Misma imagen = primera regla de comparación justa**.

## La imagen base

```python
import io, PIL, requests
from IPython.display import display

image_response = requests.get('https://raw.githubusercontent.com/vita-epfl/openpifpaf/master/docs/coco/000000081988.jpg')
pil_image = PIL.Image.open(io.BytesIO(image_response.content)).convert('RGB')
display(pil_image)
```

Una foto del dataset COCO 2017, alojada en el repo oficial de OpenPifPaf. Multi-persona, con personas en distintas escalas y posibles oclusiones — escenario ideal para probar capacidades multi-persona.

El `.convert('RGB')` es **defensa contra modos de color**: PIL.Image puede venir como `L` (grayscale), `RGBA` (con alpha) o `CMYK`. Las librerías de pose esperan estrictamente RGB de 3 canales. **Patrón a internalizar**: cualquier pipeline de visión empieza con `Image.open(...).convert('RGB')`.

![Imagen base COCO antes de inferencia](/laboratorios/lab-17/coco-base.jpg)

## Demo 1 — OpenPifPaf

```python
data = openpifpaf.datasets.PilImageList([pil_image])
loader = torch.utils.data.DataLoader(data, batch_size=1, pin_memory=True)
keypoint_painter = openpifpaf.show.KeypointPainter(color_connections=True, linewidth=6)

for images_batch, _, __ in loader:
    images_batch = images_batch.cuda()
    fields_batch = processor.fields(images_batch)
    predictions = processor.annotations(fields_batch[0])
    with openpifpaf.show.image_canvas(pil_image) as ax:
        keypoint_painter.annotations(ax, predictions)
```

**Dos pasos conceptuales** que reflejan la arquitectura del paper:

| Línea de código | Concepto del paper [PifPaf](/papers/pifpaf-kreiss-2019) |
|---|---|
| `processor.fields(images_batch)` | Forward pass: $\mathbf{F} \to (\mathbf{PIF}, \mathbf{PAF})$ |
| `processor.annotations(fields_batch[0])` | Greedy decoding: $(\mathbf{PIF}, \mathbf{PAF}) \to \{\text{poses}\}$ |

La API legacy expone los componentes a propósito. Si usaras la API moderna (`openpifpaf.Predictor(...)`), todo se colapsa en una sola llamada y pierdes visibilidad de **dónde termina el deep learning y dónde empieza el algoritmo clásico**.

![Inferencia OpenPifPaf sobre la imagen base](/laboratorios/lab-17/demo-pifpaf-coco.jpg)

> Múltiples esqueletos detectados con colores diferenciados por **tipo de limb** (brazo izquierdo siempre rosa, pierna derecha siempre naranja, etc.). Cada `Annotation` tiene `.data` de shape `(17, 3)`.

## Demo 2 — MediaPipe

```python
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

image_rgb = np.array(pil_image)
results = pose.process(image_rgb)

mp_drawing = mp.solutions.drawing_utils
annotated_image = image_rgb.copy()
if results.pose_landmarks:
    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=results.pose_landmarks,
        connections=mp_pose.POSE_CONNECTIONS
    )
```

**Toda la inferencia en una sola llamada**: `pose.process(image_rgb)` orquesta internamente:

1. BlazeFace detecta la cara → bounding box.
2. De la cara extrae mid-hip, escala, rotation angle.
3. Alinea la imagen al "hombre vitruviano".
4. BlazePose procesa la imagen alineada.
5. Predice **33 landmarks** normalizados.
6. Deshace la transformación.

Detalle clave: `static_image_mode=True` **fuerza correr el detector facial en cada llamada**. Para video real-time usarías `False` (que reusa el tracker entre frames y es 5-10× más rápido).

![Inferencia MediaPipe sobre la imagen base](/laboratorios/lab-17/demo-mediapipe-coco.jpg)

> Un solo esqueleto detectado, con 33 keypoints incluyendo detalles de cara (10 puntos), manos (6 nudillos) y pies (4 puntos). **No detecta a las otras personas** porque BlazePose es single-person por design.

**Esto es la lección operacional**: MediaPipe es óptimo para **selfie de fitness**, inadecuado para **escena multitudinaria**. Para tu app de yoga personal, perfecto. Para CCTV en un parque, useless.

## Demo 3 — OpenPose

```python
import sys
sys.path.append("pytorch-openpose")

from src.body import Body
openpose_model = Body('pytorch-openpose/model/body_pose_model.pth')

opencvImage = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
annotated_body = annotate_image(opencvImage)
annotated_body_pil = PIL.Image.fromarray(annotated_body)
display(annotated_body_pil)
```

**Tres detalles únicos de OpenPose** vs. los anteriores:

### 1. El `sys.path.append("pytorch-openpose")` — code smell consciente

El repo clonado en celda 16 **no es un paquete Python instalable** (sin `setup.py`). El hack `sys.path.append` permite importar de allí como si fuera un módulo. Es **anti-patrón en código de producción** pero es **la única opción** dado el estado del fork.

Cuando veas `sys.path.append("...")` en un notebook, sospecha: estás trabajando con un **artefacto histórico que nunca fue empaquetado profesionalmente**.

### 2. La conversión BGR — legado de OpenCV

```python
opencvImage = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
```

El modelo OpenPose fue entrenado con **datos en formato BGR** (default de OpenCV desde los 90s). Si pasaras RGB:

- Las features de color están **invertidas** respecto al entrenamiento.
- Los keypoints se predicen en lugares raros.
- **No falla con error explícito** — solo produce peores predicciones silenciosamente.

Es **uno de los bugs más insidiosos** de pipelines de visión. Internalizar el patrón "OpenCV = BGR, todo lo demás = RGB" es supervivencia.

### 3. Formato `(candidate, subset)` — herencia académica

`openpose_model(opencvImage)` retorna una tupla `(candidate, subset)`:

- **`candidate`**: array `(N_candidates, 4)` con **todos los keypoints detectados sin agruparse**. Cada fila: `(x, y, confidence, id_global)`.
- **`subset`**: array `(N_persons, 20)` con **agrupamientos por persona**. Cada fila tiene 18 índices a `candidate` (o `-1` si el keypoint no fue detectado).

Esta separación refleja el algoritmo bottom-up del paper: detectar todos los keypoints primero, luego agruparlos. Pero **es incómodo en Python**. Por eso el lab define `openpose_extract_keypoints(subset, candidate)` que normaliza a una **lista de arrays NumPy `(18, 3)`** — el formato manejable que el MLP downstream va a consumir.

![Inferencia OpenPose sobre la imagen base](/laboratorios/lab-17/demo-openpose-coco.jpg)

> Múltiples esqueletos detectados con colores **fijos por tipo de limb** (no por persona). 18 keypoints (17 COCO + neck inferido). Estilo visual característico del paper original CMU.

## Comparación visual side-by-side

| Aspecto | OpenPifPaf | MediaPipe | OpenPose |
|---|---|---|---|
| Personas detectadas | Multi (3-5 visibles) | 1 | Multi (3-5 visibles) |
| Keypoints | 17 COCO | 33 BlazePose | 18 (COCO + neck) |
| Conexiones dibujadas | 19 limbs | 35 connections | 17 limbs CMU |
| Tiempo (Colab GPU) | ~5-15s | ~2-5s | ~5-15s |
| Multi-persona robusto | ✅ | ❌ | ✅ |
| Mobile-friendly | ❌ | ✅ | ❌ |
| Maneja oclusión | ✅ (PifPaf brilla aquí) | ⚠️ (depende cabeza visible) | ⚠️ (greedy errors) |

**Esta tabla es el take-away pedagógico real** del bloque de demos. Para una imagen multi-persona del dataset COCO:

- **OpenPifPaf** intenta detectar a las 4 (con éxito mixto).
- **MediaPipe** detecta **solo una**, la más visible/cercana al detector facial. **Las otras 3 son invisibles para MediaPipe**.
- **OpenPose** intenta detectar a las 4 con esqueletos clásicos estilo paper.

Esta diferencia **NO es "MediaPipe es peor"** — es **"MediaPipe está diseñado para otro caso de uso"**. La distinción es clave para elegir bien en producción.

## Cross-links

{{< cards >}}
  {{< card link="../instalacion-tres-modelos" title="Instalación de las 3 librerías" subtitle="Setup previo" icon="academic-cap" >}}
  {{< card link="../dataset-stanford40" title="Stanford 40 Dataset" subtitle="Siguiente paso: dataset cuantitativo" icon="academic-cap" >}}
  {{< card link="/papers/openpose-cao-2017" title="Paper OpenPose" subtitle="Detalles teóricos del modelo" icon="document-text" >}}
  {{< card link="/papers/pifpaf-kreiss-2019" title="Paper PifPaf" subtitle="Detalles teóricos del modelo" icon="document-text" >}}
  {{< card link="/papers/blazepose-bazarevsky-2020" title="Paper BlazePose" subtitle="Detalles teóricos del modelo" icon="document-text" >}}
{{< /cards >}}
