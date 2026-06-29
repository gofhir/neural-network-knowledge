---
title: "Practica desde 0 - Olvido Catastrófico"
weight: 30
sidebar:
  open: true
---

La clase 32 cubre el **olvido catastrófico** y el **aprendizaje continuo**: cómo entrenar un modelo en tareas que llegan en el tiempo sin que olvide lo aprendido. Esta práctica implementa las **dos familias principales** de soluciones en **mínima escala**, sobre benchmarks de juguete (MNIST permutado / split): **EWC** (regularización — anclar los pesos importantes con la matriz de Fisher) y **Experience Replay** (memoria — guardar y repasar ejemplos pasados). El contraste entre ambas es la lección central: la regularización funciona en *task-incremental* pero **colapsa en class-incremental**, donde el replay es necesario. Cada camino se replica en **triple framework** (PyTorch, TensorFlow y JAX).

## Caminos

{{< cards >}}
  {{< card link="01-ewc-desde-cero" title="01 - EWC desde cero" subtitle="Regularización con la matriz de Fisher sobre Permuted MNIST: naive olvida, EWC preserva" icon="code" >}}
  {{< card link="02-experience-replay-desde-cero" title="02 - Experience Replay desde cero" subtitle="Buffer de memoria + reservoir sampling sobre Split MNIST class-incremental, en triple framework" icon="code" >}}
{{< /cards >}}

## Requisitos previos

- [Clase 31 - Aprendizaje Reforzado](../../clase-31) (clase anterior del bloque).
- Nociones de redes neuronales (MLP, backprop) y entrenamiento.
- Python intermedio y NumPy; PyTorch básico. Útil: nociones de TensorFlow/Keras y JAX.
- GPU **no necesaria**: MNIST permutado/split corre en CPU en minutos.

## Tecnologias usadas

| Camino | Stack principal | Frameworks secundarios |
|--------|------------------|------------------------|
| 01 - EWC | PyTorch 2.x | TensorFlow 2.x, JAX |
| 02 - Experience Replay | PyTorch 2.x | TensorFlow 2.x, JAX |

## El hilo conductor

Las dos grandes familias contra el olvido (la tercera, arquitectura, se ve en la teoría):

1. **Regularización (EWC)**: no se guardan datos viejos; se penaliza mover los pesos **importantes** para las tareas anteriores (importancia = matriz de Fisher). Barato en memoria, pero **falla en class-incremental**.
2. **Memoria / Replay**: se guarda un pequeño **buffer** de ejemplos pasados y se mezclan con los nuevos. Simple y efectivo; es la familia que **escala a class-incremental** (la lección de van de Ven).

---

**Ver tambien:** [Clase 32 - Teoria](../teoria) · [Clase 32 - Profundizacion](../profundizacion) · Fundamentos: [Aprendizaje Continuo](/fundamentos/aprendizaje-continuo) · [Transfer Learning](/fundamentos/transfer-learning).
