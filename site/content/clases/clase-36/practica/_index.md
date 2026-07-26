---
title: "Practica desde 0 - Análisis de Video"
weight: 30
sidebar:
  open: true
---

La Clase 36 tiene una tesis central: **el 2D CNN por frame ignora el orden temporal**, y las arquitecturas de video existen para recuperarlo. Esta práctica lo demuestra **desde cero** con una tarea de juguete diseñada para exponer exactamente esa limitación. El primer camino construye el **clasificador 2D CNN + fusión temporal** y prueba, empíricamente, que **no puede** distinguir dos acciones que solo difieren en el orden de sus frames. El segundo camino agrega el **modelado temporal** (CNN + LSTM y convolución 3D) y muestra que **sí** lo logra. Cada uno en **triple framework** (PyTorch, TensorFlow, JAX).

## Caminos

{{< cards >}}
  {{< card link="01-clasificador-2d-cnn-y-fusion-temporal" title="01 - Clasificador 2D CNN + fusión temporal" subtitle="El enfoque base y la demostración de su invarianza al orden (las 3 representaciones)" icon="code" >}}
  {{< card link="02-modelar-el-tiempo-cnn-lstm-y-conv3d" title="02 - Modelar el tiempo: CNN+LSTM y Conv3D" subtitle="Recuperar el orden temporal con LSTM y con convolución 3D, en PyTorch, TensorFlow y JAX" icon="code" >}}
{{< /cards >}}

## Requisitos previos

- [Clase 36 - Teoría](/clases/clase-36/teoria) y [Profundización](/clases/clase-36/profundizacion).
- [Clase 11 - Redes Recurrentes](/clases/clase-11) (RNN/LSTM) para el camino 02.
- Nociones de CNN y clasificación de imágenes.
- Python intermedio y NumPy; PyTorch básico. Útil: TensorFlow/Keras y JAX.
- GPU **no necesaria**: los "videos" de juguete son diminutos y corren en CPU en segundos.

## Tecnologias usadas

| Camino | Stack principal | Frameworks secundarios |
|--------|------------------|------------------------|
| 01 - 2D CNN + fusión | PyTorch | TensorFlow / JAX |
| 02 - CNN+LSTM / Conv3D | PyTorch 2.x | TensorFlow 2.x, JAX |

## El hilo conductor

1. **2D CNN + fusión temporal**: procesamos cada frame con una CNN y **promediamos** las predicciones. Sobre una tarea donde la etiqueta depende del **orden** de los frames (un punto que se mueve izquierda→derecha vs. derecha→izquierda), este modelo alcanza ~50% —**azar**—, porque el promedio es **invariante a permutaciones**. Es la limitación que denuncia la clase, hecha visible.
2. **Modelar el tiempo**: reemplazamos el promedio por una **LSTM** (que procesa los frames en orden) o por una **convolución 3D** (cuyo kernel se extiende en el tiempo). Ambos resuelven la tarea —la evidencia de por qué el video necesita arquitecturas especializadas.

---

**Ver tambien:** [Clase 36 - Teoria](/clases/clase-36/teoria) · [Clase 36 - Profundizacion](/clases/clase-36/profundizacion) · Fundamentos: [Reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones) · [Análisis de Video](/fundamentos/analisis-de-video).
