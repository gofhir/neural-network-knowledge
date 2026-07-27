---
title: "Actividad y el hallazgo de los frames"
weight: 3
---

La actividad son 6 preguntas: 5 conceptuales y un experimento (4 vs 8 frames) cuyo resultado es **contraintuitivo** y confirma empíricamente la limitación del modelo.

## Las 5 preguntas conceptuales

**1. Desafíos del análisis de video.** Hereda los de las imágenes y agrega el tiempo: costo computacional (decenas/cientos de frames), modelado del movimiento (muchas acciones se definen por su dinámica), redundancia temporal (frames consecutivos casi idénticos), condiciones "in the wild" (cámara temblorosa, iluminación, oclusiones), y variabilidad de duración/velocidad.

**2. Por qué importa.** Gran parte de la información real es dinámica y solo se entiende en el tiempo. Aplicaciones: reconocimiento de acciones/gestos, vigilancia, deportes, conducción autónoma, y en salud el análisis de movimiento (marcha, detección de caídas). Distinguir "sentarse" de "pararse" o detectar una caída es imposible con una sola imagen.

**3. Qué es action classification.** Asignar a un video completo una etiqueta de la acción realizada, de un conjunto de clases. Análogo a clasificar imágenes, pero la entrada es una secuencia de frames que el modelo debe integrar. Distinto de la *detección* (localizar cuándo/dónde) — la clasificación solo responde *qué* acción es.

**4. El muestreo temporal.** **Uniforme**: `np.linspace(0, total-1, num=num_frames)` selecciona `num_frames` (8) frames equiespaciados a lo largo del video. Reduce costo, elimina redundancia y garantiza cubrir todo el video.

**5. El temporal pooling y sus problemas.** **Average pooling** (`torch.mean(out, dim=1)`): promedia los features de los 8 frames. Problemas: es **invariante al orden** (no distingue acciones inversas temporales), no captura dinámica ni dirección de movimiento, y trata el video como un "bag of frames". Ver [reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones).

## Pregunta 6: el hallazgo contraintuitivo (4 vs 8 frames)

El experimento: re-entrenar con 4 frames en vez de 8 (cambiar `num_frames=4`, re-instanciar dataset y modelo — sin re-preprocesar). El resultado sorprende:

| | **8 frames** | **4 frames** |
|---|---|---|
| Val Acc (best) | 84.6% | **85.9%** |
| Tiempo | 4m30s | **2m12s** |

![Curva de accuracy con 4 frames: train y val ascendentes, alcanzando ~86% de val](/laboratorios/lab-36/curva-4-frames.png)

**Con 4 frames NO empeoró — incluso mejoró marginalmente (85.9% vs 84.6%), en la mitad del tiempo.** Esto es mejor que la expectativa ingenua de "más frames = mejor", y no es casualidad:

{{< callout type="warning" >}}
**El experimento de la pregunta 6 confirma la crítica de la pregunta 5.** Como el modelo usa average pooling, que **descarta el orden temporal** y solo modela la apariencia promedio, la información temporal fina de tener 8 frames en vez de 4 **no aporta nada esencial — el modelo no la aprovecha de todos modos**. Si el modelo *usara* la dinámica, quitar frames debería dolerle; como no le dolió (85.9% ≥ 84.6%), se demuestra empíricamente que el modelo *no está usando* la información temporal. Las dos preguntas se conectan: la 5 dice "el pooling ignora el orden", la 6 lo prueba con datos.
{{< /callout >}}

**La conclusión:** para este modelo y este dataset, reducir de 8 a 4 frames es un **trade-off favorable** —casi el mismo accuracy a la mitad del costo—. Y el mensaje profundo: el cuello de botella no es la cantidad de frames sino la **incapacidad de la arquitectura de modelar el tiempo**. Un modelo que sí usara el orden (C3D, LSTM, two-stream) probablemente sí se beneficiaría de más frames — y ahí es donde apunta el resto del módulo de video.

## Síntesis del lab

El lab construye el pipeline completo de video understanding (UCF11 → frames → muestreo → ResNet + pooling → entrenamiento → evaluación) y, al hacerlo, te hace *construir y luego criticar* el enfoque más simple: el video como "bag of frames" promediados. Funciona sorprendentemente bien (84–86%) porque UCF11 se resuelve por apariencia, pero su límite —la ceguera al orden temporal— queda demostrado tanto teóricamente (pregunta 5) como empíricamente (pregunta 6). Es el baseline honesto sobre el que se construyen los métodos temporales de la clase 36.
