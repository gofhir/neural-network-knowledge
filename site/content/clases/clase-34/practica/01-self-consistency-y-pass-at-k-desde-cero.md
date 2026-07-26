---
title: "Self-Consistency y Pass@k desde cero"
weight: 1
math: true
---

La [teoría de la Clase 34](/clases/clase-34/teoria) muestra que gran parte del razonamiento moderno se gana **en tiempo de inferencia**: en vez de confiar en una sola respuesta, se **muestrean muchas** y se **agregan**. Este capítulo construye desde cero los dos mecanismos centrales —**Self-Consistency** (voto por mayoría, Wang et al. 2022) y **Pass@k** (cobertura, Brown et al. 2024)— sin necesidad de un LLM real. Modelamos un "razonador" como un proceso estocástico sobre un problema **verificable** de juguete, y descubrimos la lección más importante de la clase: **muestrear mucho solo sirve si puedes verificar**.

> **Lecturas de apoyo:** los fundamentos [Chain-of-Thought](/fundamentos/chain-of-thought) y [Test-time compute](/fundamentos/test-time-compute); los papers [Self-Consistency](/papers/self-consistency-wang-2022) y [Large Language Monkeys](/papers/large-language-monkeys-brown-2024).

---

## 1. El montaje: un razonador de juguete verificable

No necesitamos un LLM para estudiar la agregación. Basta un **razonador estocástico**: dado un problema, muestrea una respuesta que es la correcta con probabilidad $p$, y si no, cae en alguna respuesta incorrecta según una distribución de "distractores". Este modelo captura lo esencial de un LLM razonando: hay una respuesta correcta a la que **converge por varios caminos**, y errores que **se dispersan**.

```python
import numpy as np
rng = np.random.default_rng(0)

def toy_reasoner(p_correct, n_distractors=4, n_samples=1, rng=rng):
    """Muestrea n respuestas. La correcta es la etiqueta 0.
    Con prob. p_correct devuelve 0; si no, un distractor 1..n_distractors.
    Los distractores NO son uniformes: uno es 'tentador' (error común)."""
    out = []
    # el distractor 1 es un 'error sistemático' atractivo: se lleva la mitad de la masa de error
    distract_p = np.array([0.5] + [0.5/(n_distractors-1)]*(n_distractors-1))
    for _ in range(n_samples):
        if rng.random() < p_correct:
            out.append(0)                                   # respuesta correcta
        else:
            out.append(1 + rng.choice(n_distractors, p=distract_p))
    return np.array(out)
```

La respuesta correcta es la etiqueta `0`. El detalle clave —un **distractor sistemático** que concentra la mitad de los errores— es lo que hará interesante (y peligrosa) la agregación: es el análogo del error que un LLM comete de forma consistente.

{{< concept-alert type="clave" >}}
El problema es **verificable** si podemos comprobar cuál respuesta es la correcta (aquí, "¿es igual a `0`?"). Un problema de matemáticas con checker o código con tests **lo es**; una recomendación clínica sin ground-truth **no lo es**. Toda la diferencia entre Pass@k y Self-Consistency depende de esta distinción.
{{< /concept-alert >}}

---

## 2. Pass@k: cobertura con un verificador

**Pass@k** es la probabilidad de que **al menos una** de $k$ muestras sea correcta —*asumiendo que un verificador puede identificarla*. Con $n$ muestras totales de las que $c$ son correctas, el **estimador insesgado** (Chen et al., 2021) es:

$$
\text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}.
$$

```python
from math import comb

def pass_at_k(n, c, k):
    """Estimador insesgado de pass@k: n muestras, c correctas."""
    if n - c < k:            # imposible NO tener una correcta entre k
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

# Estima la cobertura empírica en función de k
def coverage_curve(p_correct, ks, n=200, n_problems=500):
    cov = []
    for k in ks:
        solved = 0
        for _ in range(n_problems):
            samples = toy_reasoner(p_correct, n_samples=n)
            c = int((samples == 0).sum())
            solved += pass_at_k(n, c, k)      # prob. de resolverlo con k muestras
        cov.append(solved / n_problems)
    return np.array(cov)

ks = [1, 2, 5, 10, 20, 50, 100]
print("cobertura:", np.round(coverage_curve(p_correct=0.2, ks=ks), 3))
```

Con $p=0.2$, una sola muestra acierta el 20%, pero la cobertura **crece rápido** con $k$: con suficientes intentos, casi todos los problemas tienen alguna muestra correcta. Es el hallazgo de *Large Language Monkeys*: la cobertura escala de forma predecible con el cómputo de inferencia.

{{< concept-alert type="advertencia" >}}
Pass@k mide un **límite superior**: presupone que puedes **elegir** la muestra correcta. Sin verificador, ese `== 0` no existe —solo tienes 100 respuestas sin etiqueta. La cobertura es una promesa que **solo el verificador puede cobrar**.
{{< /concept-alert >}}

---

## 3. Self-Consistency: agregar sin verificador

¿Y si no hay verificador? Self-Consistency propone quedarse con la respuesta **más frecuente** (voto por mayoría), marginalizando sobre los caminos de razonamiento:

$$
\hat y = \arg\max_{y} \sum_{i=1}^{m} \mathbb{1}\!\left[\,y_i = y\,\right].
$$

```python
def self_consistency(samples):
    """Voto por mayoría: devuelve la respuesta más frecuente."""
    vals, counts = np.unique(samples, return_counts=True)
    return vals[counts.argmax()]

def accuracy_sc(p_correct, m, n_problems=2000):
    hits = 0
    for _ in range(n_problems):
        samples = toy_reasoner(p_correct, n_samples=m)
        hits += (self_consistency(samples) == 0)     # ¿el voto acertó?
    return hits / n_problems

for m in [1, 5, 11, 41]:
    print(f"m={m:3d}  self-consistency acc={accuracy_sc(0.35, m):.3f}")
```

Con $p=0.35$ (la correcta es la moda de la distribución), self-consistency **mejora** sobre una sola muestra: al votar, la respuesta correcta —a la que se llega por varios caminos— gana a los errores dispersos. Esto reproduce el resultado de Wang et al. sin ningún modelo entrenado.

### 3.1 Cuándo self-consistency FALLA

Aquí es donde el distractor sistemático cobra protagonismo. Si el **error sistemático es más probable que la respuesta correcta**, el voto por mayoría converge... **a la respuesta equivocada**:

```python
# p_correct=0.2, pero el distractor 1 se lleva 0.5*0.8 = 0.40 de la masa: ¡supera a la correcta!
print("acc correcta:", accuracy_sc(0.2, m=41))          # baja: la mayoría vota el distractor
# Compara con pass@k, que SÍ resolvería estos casos si hubiera verificador:
samples = toy_reasoner(0.2, n_samples=100)
print("pass@100 (con verificador):", pass_at_k(100, int((samples==0).sum()), 100))
```

El contraste es la moraleja del capítulo: **con verificador** (Pass@k) el muestreo masivo resuelve el problema; **sin verificador** (Self-Consistency), si el modelo tiene un sesgo sistemático, muestrear más solo **refuerza el error** —la mayoría vota confiada por la respuesta incorrecta.

---

## 4. Las mismas cuentas en triple framework

La agregación —conteo, argmax, marginalización— es una operación tensorial idéntica en cualquier framework. Aquí el voto por mayoría sobre un lote de problemas (`[n_problemas, m]` muestras).

### PyTorch

```python
import torch

def majority_vote_torch(samples, n_classes):
    # samples: LongTensor [P, m] con etiquetas en [0, n_classes)
    onehot = torch.nn.functional.one_hot(samples, n_classes)   # [P, m, C]
    counts = onehot.sum(dim=1)                                 # [P, C]
    return counts.argmax(dim=1)                                # [P]
```

### TensorFlow

```python
import tensorflow as tf

def majority_vote_tf(samples, n_classes):
    onehot = tf.one_hot(samples, n_classes)      # [P, m, C]
    counts = tf.reduce_sum(onehot, axis=1)       # [P, C]
    return tf.argmax(counts, axis=1)             # [P]
```

### JAX

```python
import jax.numpy as jnp

def majority_vote_jax(samples, n_classes):
    onehot = jax.nn.one_hot(samples, n_classes)  # [P, m, C]
    counts = onehot.sum(axis=1)                  # [P, C]
    return counts.argmax(axis=1)                 # [P]
```

Las tres son la misma receta: **one-hot → sumar sobre las muestras → argmax**. Marginalizar sobre caminos de razonamiento es, en el fondo, contar votos.

---

## 5. Qué nos llevamos

- **Pass@k** mide la cobertura alcanzable con $k$ muestras; crece rápido con el cómputo de inferencia, pero **presupone un verificador** que elija la muestra correcta.
- **Self-Consistency** agrega por voto mayoritario **sin verificador**; funciona cuando la respuesta correcta es la moda, pero **falla y refuerza el error** cuando el modelo tiene un sesgo sistemático.
- La distinción **verificable / no verificable** decide cuál sirve —la lección central del test-time compute, y la razón por la que el RL de DeepSeek-R1 solo aplica a dominios con verificador.

En el [camino 02](/clases/clase-34/practica/02-grpo-desde-cero) damos el siguiente paso: en vez de un razonador fijo, **entrenamos** una política con recompensa verificable usando **GRPO**.

---

**Ver también:** [Clase 34 - Teoría](/clases/clase-34/teoria) · [Clase 34 - Profundización](/clases/clase-34/profundizacion) · [Camino 02: GRPO](/clases/clase-34/practica/02-grpo-desde-cero) · [Laboratorio](/laboratorios/lab-34).
