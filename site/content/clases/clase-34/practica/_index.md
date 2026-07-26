---
title: "Practica desde 0 - Razonamiento"
weight: 30
sidebar:
  open: true
---

La Clase 34 muestra que buena parte del razonamiento moderno de los LLMs **no vive en la arquitectura sino en cómo se muestrea y agrega la inferencia**: muchas cadenas de razonamiento, un voto por mayoría, una recompensa verificable. Esta práctica construye esos mecanismos **desde cero**, sin necesidad de un LLM real, sobre tareas de juguete **verificables**. El primer camino implementa **Self-Consistency y Pass@k** (muestrear y agregar); el segundo implementa **GRPO** (el RL con ventaja relativa al grupo que entrena a DeepSeek-R1). Cada uno se muestra en **triple framework** (PyTorch, TensorFlow y JAX).

## Caminos

{{< cards >}}
  {{< card link="01-self-consistency-y-pass-at-k-desde-cero" title="01 - Self-Consistency y Pass@k desde cero" subtitle="Muestrear cadenas, votar por mayoría, estimar cobertura; el rol del verificador (NumPy + las 3 representaciones)" icon="code" >}}
  {{< card link="02-grpo-desde-cero" title="02 - GRPO desde cero" subtitle="Ventaja relativa al grupo + policy gradient con recompensa verificable, en PyTorch, TensorFlow y JAX" icon="code" >}}
{{< /cards >}}

## Requisitos previos

- [Clase 34 - Teoría](/clases/clase-34/teoria) y [Profundización](/clases/clase-34/profundizacion) (Pass@k, self-consistency, GRPO).
- [Clase 31 - Aprendizaje Reforzado](/clases/clase-31) (política, recompensa, policy gradient) para el camino 02.
- Python intermedio y NumPy; PyTorch básico. Útil: TensorFlow/Keras y JAX.
- GPU **no necesaria**: todo corre en CPU en segundos.

## Tecnologias usadas

| Camino | Stack principal | Frameworks secundarios |
|--------|------------------|------------------------|
| 01 - Self-Consistency / Pass@k | NumPy | PyTorch / TensorFlow / JAX (agregación tensorial) |
| 02 - GRPO | PyTorch 2.x | TensorFlow 2.x, JAX + optax |

## El hilo conductor

1. **Self-Consistency y Pass@k**: dado un "razonador" que muestrea respuestas con cierta probabilidad de acierto, ¿cuánto ayuda muestrear muchas veces? Implementamos el **voto por mayoría** (self-consistency) y el estimador insesgado de **Pass@k**, y vemos la diferencia crítica entre tener o no un **verificador**.
2. **GRPO**: reemplazamos el "razonador de juguete" por una **política entrenable**. Con una **recompensa verificable** (¿la respuesta es correcta?) y la **ventaja normalizada dentro de un grupo** de muestras, entrenamos la política por policy gradient —la receta de DeepSeek-R1 en miniatura.

---

**Ver tambien:** [Clase 34 - Teoria](/clases/clase-34/teoria) · [Clase 34 - Profundizacion](/clases/clase-34/profundizacion) · Fundamentos: [Chain-of-Thought](/fundamentos/chain-of-thought) · [Test-time compute](/fundamentos/test-time-compute).
