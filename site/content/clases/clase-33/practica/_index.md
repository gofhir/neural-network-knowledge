---
title: "Practica desde 0 - Imitación e IRL"
weight: 30
sidebar:
  open: true
---

La Clase 33 muestra que se puede aprender una política **imitando a un experto** en vez de por ensayo y error. Esta práctica construye los **dos pilares** de la imitación en **mínima escala**, sobre un mismo entorno de juguete: el **Behavioral Cloning** (imitación supervisada directa, que expone el problema del *distribution shift*) y **DAgger** (Dataset Aggregation, que lo resuelve consultando al experto sobre los estados visitados). El salto entre ambos es el corazón de la clase: por qué copiar al experto sobre *sus* estados falla, y cómo cerrar el bucle lo arregla. Cada camino se muestra en **triple framework** (PyTorch, TensorFlow y JAX).

## Caminos

{{< cards >}}
  {{< card link="01-behavioral-cloning-desde-cero" title="01 - Behavioral Cloning desde cero" subtitle="Imitación supervisada (estado→acción) y la demostración del compounding error, en las 3 representaciones" icon="code" >}}
  {{< card link="02-dagger-desde-cero" title="02 - DAgger desde cero" subtitle="El bucle rodar→consultar experto→agregar→reentrenar que elimina el distribution shift (PyTorch, TensorFlow, JAX)" icon="code" >}}
{{< /cards >}}

## Requisitos previos

- [Clase 31 - Aprendizaje Reforzado](/clases/clase-31) (MDP, política, la noción de experto/agente).
- Nociones de aprendizaje supervisado (clasificación, entropía cruzada, descenso de gradiente).
- Python intermedio y NumPy; PyTorch básico. Útil: nociones de TensorFlow/Keras y JAX.
- GPU **no necesaria**: el entorno de juguete corre en CPU en segundos.

## Tecnologias usadas

| Camino | Stack principal | Frameworks secundarios |
|--------|------------------|------------------------|
| 01 - Behavioral Cloning | NumPy + PyTorch | TensorFlow / JAX (la red de política) |
| 02 - DAgger | PyTorch 2.x | TensorFlow 2.x, JAX + Flax/optax |

## El hilo conductor

1. **Behavioral Cloning**: entrena $\pi_\theta(a\mid s)$ como un **clasificador** sobre pares (estado, acción) del experto. Simple y rápido, pero al desplegarse la política visita estados **fuera de su distribución de entrenamiento** y los errores se **acumulan** (compounding error, cota $\mathcal{O}(T^2\epsilon)$).
2. **DAgger**: cierra el bucle. Rueda la política actual, **consulta al experto** sobre los estados que ella misma visita, **agrega** esos pares y **reentrena**. La garantía pasa a ser **lineal** $\mathcal{O}(T\epsilon)$.

Ambos ilustran la advertencia central de la clase: la imitación funciona **mientras la política se mantenga dentro de su zona de entrenamiento**. El [laboratorio](/laboratorios/lab-33) lleva DAgger a un caso real (Atari Breakout con experto DQN).

---

**Ver tambien:** [Clase 33 - Teoria](/clases/clase-33/teoria) · [Clase 33 - Profundizacion](/clases/clase-33/profundizacion) · Fundamentos: [Aprendizaje por Imitación](/fundamentos/aprendizaje-por-imitacion) · [Aprendizaje Reforzado Inverso](/fundamentos/aprendizaje-reforzado-inverso).
