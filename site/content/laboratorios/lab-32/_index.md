---
title: "Lab 32 - Aprendizaje Incremental y Olvido Catastrófico"
weight: 320
sidebar:
  open: true
---

**Profesor:** Alain Raymond
**Fecha:** Junio 2026
**Notebook origen:** `clase_32/material/Laboratorio/Practico_32.ipynb`
**Notebook ejecutado:** [lab32.ipynb](/notebooks/lab32.ipynb) · [HTML](/notebooks-html/lab32.html)

## Encuadre

La contraparte práctica de la [clase 32](/clases/clase-32) —la **última clase del curso**—: medir el **olvido catastrófico** y comparar tres estrategias de [aprendizaje continuo](/fundamentos/aprendizaje-continuo) sobre el benchmark canónico **Permuted MNIST**. El notebook se basa en los tutoriales de [ContinualAI](https://www.continualai.org/), la comunidad autora de la librería [Avalanche](https://avalanche.continualai.org/).

El hilo conductor es un solo número que hay que subir: tras entrenar 3 tareas en secuencia, un modelo ingenuo **olvida el ~70% de la primera tarea**. Las tres estrategias son tres formas distintas de comprarle **estabilidad** a la red sin matar su **plasticidad**:

| Estrategia | Mecanismo | Qué cuesta | Familia |
|---|---|---|---|
| **Naive** (baseline) | SGD puro, sin protección | 0 | — (control negativo) |
| **Rehearsal** | Buffer de ejemplos viejos re-mezclados en cada batch | Memoria de **datos** (crece con el dataset) | Replay |
| **EWC** | Penaliza mover los pesos importantes (matriz de Fisher) | Memoria del **modelo** (2× params por tarea) | Regularización |

El escenario del lab es **Domain-Incremental** (Van de Ven & Tolias): cambia la distribución de entrada (la permutación de píxeles) pero las 10 clases y su significado se mantienen — por eso basta una red single-head.

## Resultados consolidados (medidos en el notebook)

Matriz de accuracy tras entrenar las 3 tareas (T0 = MNIST original, T1/T2 = permutaciones):

| Estrategia | Test T0 | Test T1 | Test T2 | **Avg ACC final** |
|---|---|---|---|---|
| **Naive** | 24% | 28% | 80% | **~44%** |
| **EWC** (λ=0.7, mal calibrado) | 20% | 37% | 81% | ~46% |
| **EWC** (λ=10.000, óptimo) | — | — | — | **58.98%** |
| **Rehearsal** (1.000/tarea) | 80% | 44% | 86% | **70.02%** |

- **Naive:** T0 se desploma de 94% → 24% (–70 pts). La red solo recuerda la última tarea.
- **EWC:** con λ bien calibrado protege sin guardar datos; **0.52 MB** de memoria extra.
- **Rehearsal:** el mejor accuracy, a costa de **3-6× más memoria** que EWC y de retener datos crudos.

### Las lecciones del lab

1. **El olvido catastrófico es abrupto, no gradual.** T0 cae 70 puntos tras solo 2 épocas de una tarea nueva. Y es **acumulativo**: cada tarea nueva erosiona *todas* las anteriores (T1 pasó de 83% → 44% al llegar T2).
2. **El promedio engaña; miran las columnas.** El Avg ACC de Naive parece "subir" (38→44%) porque la última tarea recién aprendida infla el promedio. La señal real de olvido está en la caída por columna de cada tarea vieja.
3. **Rehearsal protege más a la tarea más antigua** (contraintuitivo): T0 se rehearsó 2 veces (en T1 y T2) y T1 solo 1 vez (en T2), así que T0 (80%) > T1 (44%) pese al mismo buffer.
4. **EWC tiene un λ óptimo con forma de U invertida.** λ bajo (0.7) ≈ Naive; λ=10.000 da el pico (59%); λ≥100.000 **colapsa a 9.8% (azar)** por divergencia numérica del gradiente de la penalización — exactamente lo que el parámetro `fisher_clip` (declarado pero no implementado) debía prevenir.
5. **Ninguno gana siempre.** Es un trade-off de tres vías entre accuracy, memoria y plasticidad futura. EWC brilla cuando no se pueden guardar datos (privacidad, datos clínicos); Rehearsal cuando la memoria no es limitante.

## Bloques del lab

{{< cards >}}
  {{< card link="01-olvido-catastrofico" title="Olvido catastrófico y Permuted MNIST" subtitle="El benchmark, la CNN LeNet, cómo se permutan los píxeles, y la demostración cruda: T0 de 96% a 34% tras una tarea nueva" icon="trending-down" >}}
  {{< card link="02-tres-estrategias" title="Las tres estrategias" subtitle="Naive (baseline), Rehearsal (buffer + shuffle_in_unison), EWC (matriz de Fisher = g², penalización elástica). Los mecanismos y su código" icon="adjustments" >}}
  {{< card link="03-actividades-y-resultados" title="Las 4 actividades resueltas" subtitle="Orden y escalabilidad (Act 1), trade-off buffer↔memoria (Act 2), curva de λ y comparación de memoria (Act 3), síntesis comparativa (Act 4)" icon="academic-cap" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/aprendizaje-continuo" title="Aprendizaje Continuo" subtitle="El fundamento transversal: olvido catastrófico, estabilidad-plasticidad, las tres familias de métodos, los escenarios de Van de Ven" icon="book-open" >}}
{{< /cards >}}

## Cross-links

{{< cards >}}
  {{< card link="/clases/clase-32" title="Clase 32 - Teoría" subtitle="Olvido catastrófico, EWC/LwF/SI/GEM/iCaRL, arquitecturas dinámicas, los tres escenarios. La última clase del curso" icon="academic-cap" >}}
  {{< card link="/clases/clase-32/profundizacion" title="Profundización" subtitle="Math: derivación bayesiana de EWC, aproximación de Laplace, matriz de Fisher" icon="beaker" >}}
  {{< card link="/clases/clase-32/practica" title="Práctica de clase" subtitle="EWC desde cero y Experience Replay desde cero en triple framework" icon="code" >}}
  {{< card link="/laboratorios/lab-31" title="Lab 31 - Aprendizaje Reforzado (anterior)" subtitle="DQN sobre CartPole con experience replay" icon="arrow-left" >}}
{{< /cards >}}

---

> **Estado:** Lab completo. Recorrido celda a celda de las 64 celdas del notebook + las 4 actividades resueltas con experimentos propios (barrido de orden/escalabilidad, curva buffer↔accuracy↔memoria, barrido de λ en 8 órdenes de magnitud, comparación de memoria EWC vs Rehearsal). Notebook ejecutado en Colab (GPU T4) con todas las curvas embebidas. Cierra el Curso 3 y el diplomado completo.
