---
title: "Demo end-to-end"
weight: 2
---

> **Celdas 9-18 del notebook.** Descargar el dataset y los pesos preentrenados, correr el script `demo.py` sobre las 300 imágenes de test de TotalText, y analizar el log de inferencia.

## TotalText: el dataset de texto curvo (celda 10)

```python
!gdown 18KmxrawLTWm-0PZX_gLKZQnwbWQE8TRH
!mkdir datasets
!unzip -q totaltext.zip -d datasets/
```

[TotalText](/papers/total-text-chng-2017) (Ch'ng & Chan, ICDAR 2017) tiene **1555 imágenes y 11459 instancias** de texto (≈7.4 palabras/imagen). Su aporte histórico fue incluir masivamente **texto curvo/circular** anotado con **polígonos** (no cajas), justo lo que motivó la representación Bézier de ABCNet: las cajas no capturan texto curvo, y los polígonos densos son caros de predecir; la curva Bézier es el punto medio. Aquí se usa como dataset de referencia (el modelo viene fine-tuneado sobre él), no para entrenar.

> `gdown <id>` baja de Google Drive por id (maneja la página de confirmación de virus-scan que rompe a `curl`/`wget`).

## Pesos preentrenados (celda 12)

```python
!wget -O tt_attn_R_50.pth https://huggingface.co/ZjuCv/AdelaiDet/resolve/main/tt_e2e_attn_R_50.pth?download=true
```

El nombre del checkpoint codifica su receta:

| Token | Significado |
|---|---|
| `tt` | fine-tuneado en **T**otal**T**ext |
| `e2e` | **end-to-end**: detección y reconocimiento en una sola red |
| `attn` | recognizer basado en **atención** (no CTC) |
| `R_50` | backbone **ResNet-50** |

> ABCNet ofrece **dos variantes de lector**: [CTC](/fundamentos/ctc-loss) (rápido, alineación implícita) y **atención** (decoder secuencial tipo seq2seq, mejor accuracy). El profe eligió el checkpoint `attn`, que prioriza calidad de lectura. Esto matiza el paper original, que destaca el head CTC por velocidad.

El modelo fue **preentrenado sobre 150K imágenes sintéticas** y luego fine-tuneado con TotalText real. Ese patrón "sintético masivo → real escaso" es la razón de que el modelo lea texto que nunca vio: aprendió **formas de glifos latinos**, no idiomas (clave para la Actividad 2).

## El demo (celda 16)

```python
!python AdelaiDet/demo/demo.py \
    --config-file AdelaiDet/configs/BAText/TotalText/attn_R_50.yaml \
    --input datasets/totaltext/test_images/ \
    --output predictions \
    --opts MODEL.WEIGHTS tt_attn_R_50.pth
```

Detectron2 separa **arquitectura** (el `.yaml`) de **parámetros aprendidos** (el `.pth`). El config construye el esqueleto vacío; los pesos lo rellenan. `--opts MODEL.WEIGHTS …` sobreescribe la ruta de pesos desde la línea de comandos (mecanismo de override de Detectron2). Hay que mantener coherencia `attn` ↔ `attn`: cargar pesos `attn` en un config `ctc` daría `size mismatch`.

El pipeline completo por imagen: backbone ResNet-50 + FPN → detección Bézier (16 valores de control por palabra) → BezierAlign (rectifica la región curva a un rectángulo) → recognizer de atención (lee carácter a carácter) → dibuja curva + texto sobre la imagen.

## Análisis del log de inferencia

El demo procesó las **300 imágenes en ~1 min 58 s**. El log revela varias cosas:

### Rendimiento
- **~2.5 img/s** de media.
- La **primera imagen tardó 2.25 s**; el resto ~0.17–0.5 s. Ese pico es el *warmup*: PyTorch compila/inicializa los kernels CUDA de forma perezosa en la primera pasada, carga cuDNN y reserva VRAM. **Nunca midas latencia con la primera muestra.**
- Las imágenes lentas (0.4–0.8 s) son las densas: 0000056 con **60 detecciones**, 0000060 con 26. El reconocimiento es secuencial por instancia → más texto, más tiempo.

### Distribución de detecciones
De **0** instancias (0000022, sin texto legible o bajo el `confidence_threshold=0.3`) a **60** (0000056). La mayoría cae en 3–15, coherente con el promedio de TotalText.

### Los warnings — qué ignorar y qué no

| Warning | Veredicto |
|---|---|
| `FutureWarning … @torch.no_grad()` en `MaskEncoding.py` | Ignorable. Viene de `MEInst` (otro modelo de AdelaiDet que ni usamos). |
| `Config … has no VERSION. Assuming v2` | Ignorable. El `.yaml` de 2021 no trae campo de versión. |
| `torch.meshgrid: … pass the indexing argument` | Ignorable. Otra deprecación de API; el default sigue correcto. |
| 🔴 `Glyph 21475 (CJK 53E3 / 口) missing from font DejaVu Sans` | **Relevante.** El modelo emitió el carácter 口. No "sabe chino": el índice 95 del charset está hardcodeado a 口 como placeholder de "desconocido". Es la frontera del reconocimiento, y la pista de la Actividad 2. Se explica en detalle en la [disección del output](diseccion-output). |

> **Lo que te llevas:** ABCNet hace detección + reconocimiento end-to-end a ~2.5 img/s en una GPU de Colab, con variabilidad enorme (0–60 textos/imagen). El glifo 口 fantasma ya adelanta el límite del reconocimiento: **está acotado por el charset de salida fijo**.

---

**Anterior:** [instalación y stack](instalacion-stack) · **Siguiente:** [disección del output](diseccion-output)
