---
title: "Lab 36 - Introducción al Análisis de Video"
weight: 360
sidebar:
  open: true
---

**Profesor:** Vladimir Araujo (Senior AI Researcher)
**Módulo:** Audio y Video (parte de video)
**Notebook origen:** `clase_36/material/Laboratorio/Lab_36_Intro_Video.ipynb`
**Notebook ejecutado:** [lab36.ipynb](/notebooks/lab36.ipynb) · [HTML](/notebooks-html/lab36.html)

## Encuadre

La contraparte práctica de la [clase 36](/clases/clase-36): **clasificación de acciones** en video con el dataset UCF11. El lab construye un pipeline completo de video understanding y, al hacerlo, te hace *construir y luego criticar* el enfoque más simple: tratar el video como un **"bag of frames" promediados**.

La estrategia: muestrear 8 frames de cada video, pasar **cada frame por separado** por un ResNet-34 (CNN 2D pre-entrenada en ImageNet), **promediar** sus features (temporal pooling), y clasificar. Simple y barato — pero con un techo: **pierde el orden temporal**.

## Resultados consolidados (medidos en el notebook)

| Experimento | Val Acc | Tiempo | Lectura |
|---|---|---|---|
| **8 frames** | 84.6% | 4m30s | El baseline |
| **4 frames** | **85.9%** | **2m12s** | ¡No empeoró, en la mitad del tiempo! |

### Las lecciones del lab

1. **Video = secuencia de imágenes.** El preprocesamiento descompone cada video en frames JPG; se muestrean 8 uniformemente (`np.linspace`). Descomprimir una vez, muestrear muchas.
2. **Transfer learning hace viable el video.** Un ResNet-34 de ImageNet aplicado frame por frame — un frame *es* una imagen. Con ~1200 videos y 3 épocas se llega a 84%.
3. **El average pooling es invariante al orden.** `torch.mean` sobre los frames descarta el orden temporal: el modelo no distingue "sentarse" de "pararse". Trata el video como un "bag of frames".
4. **El hallazgo contraintuitivo (4 vs 8 frames).** Reducir a 4 frames **no empeoró** (85.9% ≥ 84.6%) a la mitad del costo. Esto **prueba empíricamente** que el modelo no usa la información temporal — si la usara, quitar frames dolería. La pregunta 6 confirma la crítica de la pregunta 5.
5. **La augmentation de video preserva la coherencia temporal.** Las transformaciones `Group` se aplican igual a los 8 frames — voltear frame por frame destruiría el movimiento.

## Bloques del lab

{{< cards >}}
  {{< card link="01-pipeline-de-video" title="El pipeline de video" subtitle="UCF11, preprocesamiento video→frames, el VideoDataset con muestreo temporal uniforme (linspace), y la augmentation Group que preserva la coherencia temporal" icon="variable" >}}
  {{< card link="02-modelo-y-temporal-pooling" title="El modelo y el temporal pooling" subtitle="VideoNet: ResNet-34 + transfer learning, el viaje de las dimensiones, el average pooling que pierde el orden temporal, y el entrenamiento (84.6%)" icon="adjustments" >}}
  {{< card link="03-actividad-y-hallazgo" title="Actividad y el hallazgo de los frames" subtitle="Las 5 preguntas conceptuales y el experimento 4 vs 8 frames: 85.9% ≥ 84.6% en la mitad del tiempo — la prueba empírica de que el modelo ignora el orden temporal" icon="academic-cap" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/analisis-de-video" title="Análisis de video" subtitle="El fundamento transversal: la dimensión temporal, muestreo, representaciones de video" icon="book-open" >}}
  {{< card link="/fundamentos/reconocimiento-de-acciones" title="Reconocimiento de acciones" subtitle="Action classification, bag-of-frames vs métodos temporales (C3D, LSTM, two-stream)" icon="book-open" >}}
  {{< card link="/fundamentos/flujo-optico" title="Flujo óptico" subtitle="Cómo capturar el movimiento explícitamente — lo que el average pooling no hace" icon="book-open" >}}
{{< /cards >}}

## Cross-links

{{< cards >}}
  {{< card link="/clases/clase-36" title="Clase 36 - Teoría" subtitle="Introducción al análisis de video: datasets, two-stream, C3D, I3D, flujo óptico, TSN" icon="academic-cap" >}}
  {{< card link="/clases/clase-36/practica" title="Práctica de clase" subtitle="Demuestra la invarianza al orden del 2D CNN en triple framework" icon="code" >}}
  {{< card link="/laboratorios/lab-35" title="Lab 35 - Análisis de Audio (anterior)" subtitle="FFT, STFT, MFCC — el otro medio del módulo Audio y Video" icon="arrow-left" >}}
  {{< card link="/laboratorios/lab-38" title="Lab 38 - Action Recognition con I3D" subtitle="El mismo sesgo temporal, una arquitectura más arriba: invertir el video no cambia la predicción de una CNN 3D. Aquí el pooling 2D no puede usar el orden; allá la 3D puede y no lo usa" icon="film" >}}
  {{< card link="/laboratorios/lab-40" title="Lab 40 - Reconocimiento de acciones con TSM" subtitle="El tercer punto de la serie: una CNN 2D que sí modela el tiempo, y una ablación que mide el aporte video por video — 82.76 puntos en un salto alto, 0.42 en una guitarra" icon="film" >}}
{{< /cards >}}

---

> **Estado:** Lab completo. Recorrido celda a celda de las 84 celdas + las 6 preguntas de la actividad resueltas. Notebook ejecutado en Colab (GPU): entrenamiento con 8 frames (84.6% val) y el experimento de la pregunta 6 con 4 frames (85.9% val, 2× más rápido). Hallazgo central: reducir frames no perjudica porque el average pooling no aprovecha la info temporal. Sin papers ni fundamentos nuevos (todos de la clase 36).
