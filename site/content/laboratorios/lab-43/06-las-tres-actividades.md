---
title: "Las tres actividades"
weight: 6
---

Las tres son preguntas de alternativa, y las tres se pueden resolver **sin salir del código del laboratorio**: la aritmética de una capa, el contenido de un `state_dict` y la tabla de resultados del paper alcanzan para descartar todo lo demás.

## Actividad 1

> **¿Por qué puede ser necesario utilizar los frames de video para la tarea de speech recognition?**

### ✅ *Para dar robustez al modelo cuando el audio viene con ruido ambiente*

La Tabla 1 del paper muestra que con audio limpio el aporte del video es marginal — 97,7 % solo audio contra 98,0 % audiovisual, **+0,3 puntos**. La Figura 3 mide el desempeño en función del ruido y ahí el panorama cambia por completo:

| Condición | Ganancia del audiovisual sobre el de solo audio |
|---|---|
| 5 dB SNR | +1,3 |
| 0 dB SNR | +3,9 |
| **−5 dB SNR** | **+14,1** |

> *"the contribution of the visual modality is usually marginal in clean audio conditions"* … *"it significantly outperforms both of them under high noise levels"*

**El video no está para mejorar el caso fácil, sino para sostener el caso degradado.** La razón es estructural: el ruido acústico no toca el canal visual, así que la curva del stream de video en la Figura 3 es **una línea horizontal**. Ese gráfico es el argumento entero del paper.

Hay evidencia de esto dentro del propio código: la constante huérfana `self.clean = 1/7.` de `MyDataset` corresponde a las **siete condiciones equiprobables** con que se entrena según la sección 4.3 — audio limpio más *babble noise* a 20, 15, 10, 5, 0 y −5 dB. El entrenamiento fue diseñado alrededor del ruido.

### Por qué las otras son falsas

- ❌ *"Leer los labios y generar el audio del habla"* — el modelo **clasifica** entre 500 palabras; no sintetiza ninguna señal. No hay decoder ni vocoder en ninguna parte de la arquitectura.
- ❌ *"Localizar la persona que está hablando"* — la ROI de la boca es un recorte **fijo** `[115:211, 79:175]`, sin detección facial ni tracking. El paper: *"a fixed bounding box of 96 by 96 is used for all videos"*.
- ❌ *"Determinar el intervalo de tiempo donde se produce el habla"* — no hay detección de actividad vocal. Los clips vienen pre-segmentados en 1,16 s y el audio se recorta ciegamente a los últimos 19.456 samples.
- ❌ *"Los frames de video son imprescindibles"* — refutado por el propio paper: el modelo de **solo audio alcanza 97,7 %**. Son útiles, no imprescindibles.

## Actividad 2

> **¿Por qué el entrenamiento del modelo se hace por etapas (primero ResNets, luego BiGRUs, luego todo junto)?**

### ✅ *Para aumentar la estabilidad del entrenamiento y obtener un mayor rendimiento*

Cita textual de la sección 4.3.1: ***"Directly training end-to-end each stream leads to suboptimal performance** so we follow the same 3-step procedure."*

El procedimiento:

1. **Fase 1** — cada ResNet se entrena con un *backend temporal convolucional* y una capa softmax, hasta que la validación deja de mejorar durante 5 épocas.
2. **Fase 2** — se remueve ese backend, se conecta el BiGRU de 2 capas y se entrena solo el BiGRU por 5 épocas, con la ResNet congelada.
3. **Fase 3** — se destraba todo y se entrena end-to-end con Adam (lr 3e-4, batch 36 por stream; lr 1e-4, batch 18 para el modelo audiovisual completo), con *early stopping* de 5 épocas.

La razón de fondo es de optimización: **una recurrente montada sobre una ResNet sin entrenar converge mal**, porque el gradiente que llega a las capas convolucionales debe atravesar 29 pasos temporales de compuertas. Un backend convolucional es mucho más fácil de optimizar y estabiliza primero el extractor de features.

### La evidencia física

Este laboratorio permite responderla **sin citar el paper**, mostrando el `state_dict`. Los checkpoints contienen 36 claves que el modelo actual nunca usa:

```
backend_conv1.0.weight   (1024, 512, 5)   ← backend temporal-convolucional (fase 1)
backend_conv2.3.weight   (500, 512)       ← su capa softmax de 500 clases
lstm.forwardModule1...   (2048, 512)      ← BiLSTM de 2 capas de una versión anterior
```

**23,3 M de parámetros inertes en el archivo de audio —el 65 % del checkpoint— y 11,5 M en el de video.** El backend fue "removido" del grafo de cómputo pero quedó grabado en el `.pt`: es el registro material del entrenamiento por etapas. El detalle completo está en [la arqueología del checkpoint](03-la-arqueologia-del-checkpoint).

### Por qué las otras son falsas

- ❌ *"El modelo es muy grande y no cabe el gradiente en una GPU"* — son **54,6 M de parámetros** y la fase 3 entrena todo junto con mini-batch de 18. Si no cupiera, esa fase sería imposible.
- ❌ *"No es posible combinar feedforward con recurrentes en la propagación de gradientes"* — falso de plano; es exactamente lo que hace la fase 3, y es práctica estándar en todo el campo.
- ❌ *"Para separar el entrenamiento del stream de video del de audio"* — confunde el medio con el fin y describe solo la fase 1, ignorando que **cada stream individual también se entrena en tres pasos internos**.
- ❌ *"Para separar los 29 instantes de tiempo"* — no guarda relación con el procedimiento.

## Actividad 3

> **¿Cuál es el principal objetivo de la capa convolucional 3D en la rama de video?**

### ✅ *Capturar las dinámicas producidas en pequeños intervalos de tiempo*

La configuración de la capa lo demuestra sin necesidad de interpretación:

```python
nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
```

| Eje | Kernel | Stride | Padding | Transformación |
|---|---|---|---|---|
| **Tiempo** | 5 | **1** | **2** | **29 → 29 (sin cambio)** |
| Alto | 7 | 2 | 3 | 88 → 44 |
| Ancho | 7 | 2 | 3 | 88 → 44 |

Con stride temporal 1 y padding 2, entran 29 frames y salen 29. El `MaxPool3d` que sigue tiene kernel y stride temporal 1, de modo que tampoco altera el eje del tiempo. **La reducción es exclusivamente espacial**, y se verifica en una línea: `frontend3D(x).shape`.

Lo que la capa sí hace es dar a cada unidad de salida un campo receptivo de **5 frames consecutivos** = **200 ms** a 25 fps, aproximadamente la duración de una sílaba en habla continua. No aprende cómo *se ve* una boca, sino cómo **se mueve** una boca en un intervalo corto.

> *"A spatiotemporal convolutional layer is capable of **capturing the short-term dynamics of the mouth region** and is proven to be advantageous, **even when recurrent networks are deployed for back-end**."*

Ese "even when" es central. No basta con poner un BiGRU al final: la recurrente modela la dinámica a escala de la palabra completa (1,16 s), pero opera sobre vectores de 256 números por frame, cuando la información del movimiento rápido ya se perdió.

Conviene notar además que la ResNet-34 posterior procesa los 29 frames de forma **independiente** — el `view(-1, 64, 22, 22)` colapsa el eje temporal dentro del batch. Toda la modelación temporal del stream recae, por tanto, en dos lugares: esta Conv3D para el corto plazo y el BiGRU para el largo.

### Por qué las otras son falsas

- ❌ *"Reducir la dimensión temporal de 29 frames a 1"* — el stride temporal es **1** y la salida conserva los 29 frames.
- ❌ *"Reducir los 3 canales RGB a una matriz bidimensional"* — la entrada tiene **1 canal**. La conversión a escala de grises ocurre en `load_video_file`, antes de tocar la red.
- ❌ *"Realizar un downsampling de algunos frames"* — no se descarta ningún frame.

---

**Anterior:** [Los defectos del notebook](05-los-defectos-del-notebook) · **Volver al** [índice del lab](/laboratorios/lab-43)
