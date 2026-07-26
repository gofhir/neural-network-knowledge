---
title: "Lab 33 - Aprendizaje por Imitación y DAGGER"
weight: 330
sidebar:
  open: true
---

**Profesor:** Rodrigo Toro Icarte
**Curso 3 / Tópicos de profundización**
**Notebook origen:** `clase_33/material/Laboratorio/Lab_33_v18.ipynb`
**Notebook ejecutado:** [lab33.ipynb](/notebooks/lab33.ipynb) · [HTML](/notebooks-html/lab33.html)

## Encuadre

La contraparte práctica de la [clase 33](/clases/clase-33): entrenar un agente a jugar **Breakout** (Atari) por [aprendizaje por imitación](/fundamentos/aprendizaje-por-imitacion), copiando a un **experto DQN pre-entrenado**. El lab implementa las dos técnicas de imitación directa y las contrasta:

- **Behaviour Cloning (BC):** entrenar sobre estados que visita el *experto*. Simple, offline, pero sufre **covariate shift**.
- **DAGGER (Dataset Aggregation):** dejar que el *estudiante* conduzca y que el experto **etiquete** sus estados de error. Corrige el covariate shift a costa de necesitar el experto en el loop.

El hilo conductor es un solo número que hay que subir: el **score en Breakout**. Un experto DQN saca 10; el reto es cuánto se le acerca un estudiante que parte de pesos random.

## Resultados consolidados (medidos en el notebook)

| Régimen | Score (mediana) | Lectura |
|---|---|---|
| **Experto** (DQN pre-entrenado) | **10.00** | El techo — la línea roja del gráfico |
| **Behaviour Cloning puro** (fase 1) | **0.00** | Fracaso total: la loss baja pero no juega |
| **DAGGER** (18 fases) | **5.00** | Rescata el aprendizaje: llega al 50% del experto |

### Las lecciones del lab

1. **BC puro fracasó por completo (score 0.00).** La loss bajaba (1.32 → 1.17) mientras el score seguía en cero: el estudiante imitaba bien *en el dataset* pero no sabía jugar. Es la demostración empírica más nítida del [covariate shift](01-imitacion-y-covariate-shift) — la loss y el score **divergen dramáticamente**.
2. **DAGGER rescató el aprendizaje: 0 → 5.** La diferencia entre "no juega" y "juega a media máquina" es, literalmente, todo el valor del algoritmo. Y la única diferencia de código con BC es **quién conduce** durante la recolección.
3. **El estudiante se estancó en ~50% del experto.** La imitación tiene un techo estructural: *no puede superar al profesor*. 19 fases no bastaron para igualarlo, en parte por el buffer deslizante acotado (no es agregación pura).
4. **La curva de RL es ruidosa.** Oscila entre 2 y 6 pese a evaluar sobre 50 episodios y tomar la mediana. El ruido es intrínseco al campo.
5. **Misma red, dos semánticas de salida.** El experto lee sus 4 logits como **Q-values** (retorno esperado, números reales); el estudiante los entrena como **probabilidades** (softmax vía cross-entropy). La misma arquitectura DQN sirve para ambos.

## Bloques del lab

{{< cards >}}
  {{< card link="01-imitacion-y-covariate-shift" title="Imitación y el covariate shift" subtitle="Por qué Behaviour Cloning se rompe: cascada de errores, la cota O(εT²) → O(εT), y la reducción a no-regret online learning de Ross et al." icon="academic-cap" >}}
  {{< card link="02-pipeline-atari-modelo-experto" title="El pipeline: Atari, la CNN y el experto" subtitle="Wrappers de preprocesamiento (frame skip/stack, WarpFrame), la CNN de DQN compartida, y cargar el experto pre-entrenado (Q-values vs logits)" icon="cube-transparent" >}}
  {{< card link="03-dagger-el-algoritmo" title="DAGGER: el algoritmo y el loop" subtitle="El pseudocódigo, β, la diferencia de UNA línea entre BC y DAGGER, cross-entropy como imitación, y el buffer deslizante que se aparta de la teoría" icon="variable" >}}
  {{< card link="04-resultados-y-tarea" title="Resultados y las 5 preguntas" subtitle="BC=0, DAGGER=5, experto=10: la curva real, los tres matices que revela, las 5 respuestas de la tarea y el caso AlphaStar" icon="academic-cap" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/aprendizaje-por-imitacion" title="Aprendizaje por Imitación" subtitle="Behaviour Cloning, DAGGER, covariate shift — el fundamento transversal de este lab" icon="book-open" >}}
  {{< card link="/fundamentos/aprendizaje-reforzado-inverso" title="Aprendizaje Reforzado Inverso" subtitle="La alternativa: recuperar la recompensa del experto en vez de copiar su acción" icon="book-open" >}}
  {{< card link="/fundamentos/generalizacion-en-rl" title="Generalización en RL" subtitle="Por qué las políticas aprendidas a veces memorizan en vez de generalizar" icon="book-open" >}}
{{< /cards >}}

## Cross-links

{{< cards >}}
  {{< card link="/clases/clase-33" title="Clase 33 - Teoría" subtitle="Aprendizaje por imitación e IRL, BC + DAgger, refuerzo vs imitación (AlphaGo Zero)" icon="academic-cap" >}}
  {{< card link="/clases/clase-33/profundizacion" title="Profundización" subtitle="Math: cota O(T²ε) del BC, no-regret de DAgger, GAIL, IRL de Ng-Russell y MaxEnt" icon="beaker" >}}
  {{< card link="/clases/clase-33/practica" title="Práctica de clase" subtitle="Behavioral Cloning y DAgger desde cero en triple framework" icon="code" >}}
  {{< card link="/laboratorios/lab-31" title="Lab 31 - Aprendizaje Reforzado (relacionado)" subtitle="DQN sobre CartPole — el algoritmo que aquí actúa de experto" icon="arrow-left" >}}
  {{< card link="/laboratorios/lab-34" title="Lab 34 - Razonamiento (siguiente)" subtitle="Tool use, LoRA y optimización de prompt en LLMs — la siguiente clase del Curso 3" icon="arrow-right" >}}
{{< /cards >}}

---

> **Estado:** Lab completo. Recorrido celda a celda de las 34 celdas del notebook + las 5 preguntas de la tarea resueltas, con los resultados reales del notebook ejecutado en Colab (GPU T4): BC puro = 0, DAGGER = 5, experto = 10. Curva de aprendizaje embebida. Sin papers ni fundamentos nuevos (todos provienen de la clase 33).
