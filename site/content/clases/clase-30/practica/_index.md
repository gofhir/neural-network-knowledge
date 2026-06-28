---
title: "Practica desde 0 - Modelos con memoria externa"
weight: 30
sidebar:
  open: true
---

La clase 30 cubre las **redes con memoria externa**: arquitecturas que guardan información en **slots explícitos** (interpretables, editables) en vez de solo en los pesos. Esta práctica implementa los **dos mecanismos centrales** en **mínima escala** y sin librerías especializadas: una **End-to-End Memory Network** (la línea Weston, lectura por atención suave para QA) y el **núcleo diferenciable de una Neural Turing Machine** (la línea Graves, lectura/escritura sobre una matriz de memoria). Cada camino se replica en **triple framework** (PyTorch, TensorFlow y JAX).

## Caminos

{{< cards >}}
  {{< card link="01-end-to-end-memnn-desde-cero" title="01 - End-to-End MemNN desde cero" subtitle="Embeddings A/B/C, atención softmax sobre memorias, hops, sobre una tarea bAbI de juguete" icon="code" >}}
  {{< card link="02-memoria-diferenciable-desde-cero" title="02 - Memoria diferenciable (NTM) desde cero" subtitle="Matriz de memoria, lectura por contenido, escritura erase+add, todo diferenciable" icon="code" >}}
{{< /cards >}}

## Requisitos previos

- [Clase 14 - Transformer desde 0](../../clase-14/practica): la self-attention es lectura de memoria (query/key/value); la End-to-End MemNN es su precursor.
- [Clase 15 - Mecanismo de atención](../../clase-15): la base de la lectura por atención suave.
- Python intermedio y NumPy; PyTorch básico. Útil: nociones de TensorFlow/Keras y JAX.
- GPU **no necesaria**: las tareas de juguete (bAbI sintético, copy) corren en CPU en segundos.

## Tecnologias usadas

| Camino | Stack principal | Frameworks secundarios |
|--------|------------------|------------------------|
| 01 - End-to-End MemNN | PyTorch 2.x | TensorFlow 2.x, JAX |
| 02 - Memoria diferenciable (NTM) | PyTorch 2.x | TensorFlow 2.x, JAX |

## El hilo conductor

Las dos grandes estirpes de la memoria externa:

1. **Memory Networks (Weston)**: memoria como un conjunto de slots de contenido; se lee por **atención suave** (softmax de similitud). Pensada para QA y razonamiento sobre texto — y es el **precursor directo de la self-attention** de los Transformers.
2. **Memoria diferenciable (Graves, NTM/DNC)**: memoria como una matriz direccionable tipo computador, con cabezas de lectura/escritura diferenciables. Pensada para aprender **algoritmos**.

Ambas comparten la idea clave: una memoria **explícita y editable**, separada del "programa" (los pesos).

---

**Ver tambien:** [Clase 30 - Teoria](../teoria) · [Clase 30 - Profundizacion](../profundizacion) · Fundamentos: [Redes con Memoria Externa](/fundamentos/redes-de-memoria) · [Memory-Augmented Networks](/fundamentos/memory-augmented-networks).
