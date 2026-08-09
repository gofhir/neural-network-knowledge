---
title: "Practica desde 0 - Modelos pre-entrenados en video"
weight: 30
sidebar:
  open: true
---

El subtítulo de la Clase 38 es **"modelos pre-entrenados"**, y su momento central es el truco de I3D: **inflar** una CNN 2D entrenada en ImageNet para que opere sobre volúmenes de video, heredando sus pesos. Esta práctica implementa ese mecanismo **desde cero** y lo verifica numéricamente, en lugar de aceptarlo como afirmación. El primer camino construye la función de inflado y comprueba la **condición de punto fijo del video aburrido** —que la red inflada, alimentada con una imagen repetida, produzca exactamente el mismo logit que la red 2D original—. El segundo camino ataca la contracara: las tres desventajas que la clase le atribuye a I3D (muchos parámetros, costoso, inferencia lenta) y la respuesta de la literatura posterior, la **factorización $(2+1)$D**. Cada uno en **triple framework** (PyTorch, TensorFlow, JAX).

## Caminos

{{< cards >}}
  {{< card link="01-inflar-una-cnn-2d-a-3d" title="01 - Inflar una CNN 2D a 3D" subtitle="La función de inflado, la verificación del punto fijo end-to-end y los tres gotchas (max-pool, BatchNorm, padding)" icon="code" >}}
  {{< card link="02-factorizar-el-tiempo-2plus1d" title="02 - Factorizar el tiempo: bloques (2+1)D" subtitle="La cuenta de canales que iguala parámetros, el costo real medido y el inflado de kernels separables" icon="code" >}}
{{< /cards >}}

## Requisitos previos

- [Clase 38 - Teoría](/clases/clase-38/teoria) y sobre todo la [Profundización](/clases/clase-38/profundizacion): el camino 01 implementa su Parte I y el camino 02 su Parte III.
- [Clase 12 - Transfer Learning](/clases/clase-12) para el contexto de heredar pesos pre-entrenados.
- [Clase 36](/clases/clase-36) si se quiere el panorama de arquitecturas de video antes de entrar al detalle.
- Python intermedio y NumPy; PyTorch a nivel de `nn.Module`. Útil: TensorFlow/Keras y JAX/Flax.
- GPU **no necesaria** para el camino 01 (verificación numérica sobre tensores pequeños). Para las mediciones de latencia del camino 02, una GPU cambia los números pero no las conclusiones.

## Tecnologias usadas

| Camino | Stack principal | Frameworks secundarios |
|--------|------------------|------------------------|
| 01 - Inflado 2D→3D | NumPy + PyTorch (torchvision) | TensorFlow 2.x / JAX + Flax |
| 02 - Bloques (2+1)D | PyTorch 2.x | TensorFlow 2.x, JAX + Flax |

## El hilo conductor

1. **Inflar y verificar.** La condición que debe cumplir el inflado es que los pesos repartidos a lo largo del eje temporal **sumen** el peso 2D original: $\sum_{\tau} \widetilde{W}[\cdot,\tau,\cdot] = W$. Implementamos las dos soluciones válidas (reparto uniforme y delta central), inflamos una ResNet completa y comprobamos que el logit sobre un video aburrido coincide con el de la red 2D. En el camino aparecen los errores que la fórmula del slide no advierte: el max-pooling **no** se divide, BatchNorm hay que ponerlo en `eval()` para que la igualdad se cumpla, y el padding temporal rompe el punto fijo en los bordes del clip.

2. **Pagar menos por el tiempo.** Inflar resuelve el pre-entrenamiento pero no el costo: los pesos convolucionales se multiplican por $N$. La factorización $(2+1)$D descompone el kernel cúbico en uno espacial y uno temporal, con un número de canales intermedios $M$ elegido para **igualar exactamente los parámetros** del bloque 3D — de modo que cualquier diferencia de precisión no se pueda atribuir a capacidad. Medimos el costo real y confrontamos una expectativa común: tener los mismos parámetros no garantiza ser más rápido en wall-clock.

---

**Ver tambien:** [Clase 38 - Teoria](/clases/clase-38/teoria) · [Clase 38 - Profundizacion](/clases/clase-38/profundizacion) · Fundamentos: [Inflado de Convoluciones](/fundamentos/inflado-de-convoluciones) · [Transfer Learning](/fundamentos/transfer-learning) · [Reconocimiento de Acciones](/fundamentos/reconocimiento-de-acciones).
