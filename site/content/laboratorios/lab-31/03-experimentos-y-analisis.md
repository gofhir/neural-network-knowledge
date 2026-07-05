---
title: "Experimentos propios y análisis"
weight: 3
---

El notebook original solo entrena el DQN una vez. Para *entender* por qué funciona, aquí lo sometemos a cuatro **ablations medidas**: quitamos piezas una por una y observamos qué se rompe. Cada experimento reentrena el mismo DQN cambiando una sola cosa, y reporta la **recompensa media en test** (100/30 episodios con ε=0) y los **episodios hasta resolver** (media móvil de 10 > 195).

> Reproducible: script `experiments.py` en el material del lab. Las ablations core usan 2 semillas; las grillas de hiperparámetros, 1 semilla (tendencia, no promedio robusto). Este harness es una **reproducción local** independiente de la corrida del notebook (que resuelve en 85 episodios): su baseline propio (replay, sin target) resuelve en **139.5** episodios con test **205.6**. Lo que importa aquí son las **comparaciones relativas** entre configuraciones, no el número absoluto.

## Panorama

![Comparación de ablations de DQN](/laboratorios/lab-31/ablations.png)

| Configuración | ¿Resuelve? | Episodios | Test (ε=0) | Lectura |
|---|---|---|---|---|
| **Baseline** (replay, sin target) | ✅ 2/2 | 139.5 | 205.6 | el DQN del notebook |
| **+ Target network** | ✅ 2/2 | **100.0** | 207.0 | más estable y **más rápido** |
| **Sin velocidades** (POMDP) | ❌ 0/2 | — | **45.6** | rompe Markov |
| **Sin replay buffer** | ❌ 0/2 | — | **9.4** | colapsa **bajo el azar** |

Los dos estabilizadores de DQN quedan a la vista: quitar el replay es catastrófico; quitar la información de estado es fatal; agregar la target network ausente ayuda.

## Experimento 1 — Enmascarar las velocidades (POMDP)

**Qué cambia:** multiplicamos el estado por una máscara `[1, 0, 1, 0]`, poniendo a cero la velocidad del carro ($\dot x$) y la angular ($\dot\theta$). El agente solo ve **posiciones**. Todo lo demás igual.

**Resultado:** el DQN **no resuelve** (0/2 semillas) y su test cae a **45.6** — muy por debajo del baseline (205.6), aunque todavía por encima del azar (21).

**Por qué.** Es la validación empírica de la [pregunta 2 de la tarea](04-actividades). Sin velocidades, el estado deja de ser markoviano: el bastón a $+5°$ *cayéndose* y a $+5°$ *volviendo* producen el mismo input pero exigen acciones opuestas. Ninguna $Q(x,\theta)$ puede acertar a ambos. Que aún saque 45.6 (> azar) muestra que la posición **sola** da una señal parcial —el agente reacciona a "el bastón está inclinado" aunque no sepa hacia dónde va— pero es insuficiente para equilibrar de forma sostenida. La solución del DQN de Atari a este mismo problema es **apilar frames** para que la red infiera las velocidades; aquí, simplemente, se las quitamos.

## Experimento 2 — Quitar el experience replay

**Qué cambia:** en vez de muestrear un mini-batch aleatorio del buffer, la red aprende **solo de la última transición** (un batch de 1, en orden temporal). Es DQN "online" puro, sin des-correlación.

**Resultado:** colapso total. **9.4** de test — **peor que actuar al azar** (21). No resuelve en ninguna semilla.

**Por qué.** Es la validación empírica de la [pregunta 1 de la tarea](04-actividades) y de la teoría del [replay buffer](02-dqn-implementacion). Entrenar con transiciones consecutivas viola el supuesto i.i.d. del SGD por partida doble: gradientes correlacionados (la red oscila) y distribución no estacionaria (se sobreajusta a la región reciente y olvida). El resultado no es "aprende más lento": es que **diverge activamente** por debajo del azar. El muestreo aleatorio del buffer no es un detalle de eficiencia — es lo que hace que DQN converja.

## Experimento 3 — Agregar la target network ausente

**Qué cambia:** introducimos la pieza que el notebook omite. El target $y = r + \gamma\max_{a'}Q_{\theta^-}(s',a')$ se calcula con una **copia congelada** $Q_{\theta^-}$, sincronizada con la red principal cada 20 pasos. Es la contribución central de [Mnih et al. 2015](/papers/dqn-nature-mnih-2015).

**Resultado:** resuelve en **100.0** episodios (vs 139.5 del baseline) con test **207.0** (vs 205.6). Más rápido y ligeramente mejor.

**Por qué.** Sin target network, el objetivo de regresión se mueve cuando actualizas la red (persigues un blanco móvil) → inestabilidad. Congelar el target rompe ese lazo de realimentación: la red apunta a un objetivo estable durante 20 pasos, converge más limpio y más rápido. En CartPole (problema fácil) el baseline igual resuelve, pero la ventaja de la target network ya es visible; en Atari es la diferencia entre converger y divergir.

## Experimento 4 — Sensibilidad a los hiperparámetros

### Factor de descuento γ

![Sensibilidad al factor de descuento gamma](/laboratorios/lab-31/gamma-sensibilidad.png)

| γ | Episodios hasta resolver | Test |
|---|---|---|
| 0.90 | 136 | 210 |
| 0.95 (default) | 154 | 210 |
| 0.99 | 209 | 210 |

**Lectura:** los tres resuelven, pero **γ más alto converge más lento**. Un γ mayor alarga el horizonte efectivo ($\frac{1}{1-\gamma}$: 10, 20, 100 pasos) → el problema de asignación de crédito se vuelve más difícil (la recompensa se propaga hacia atrás por más pasos) y el bootstrapping acumula más varianza. En CartPole, donde el horizonte real es corto, un γ moderado basta y aprende más rápido.

### Ritmo de decaimiento de ε

| ε-decay | Exploración | Episodios hasta resolver | Test |
|---|---|---|---|
| 0.990 | poca (decae rápido) | 125 | 198.5 |
| 0.995 (default) | media | 154 | 210 |
| 0.999 | mucha (decae lento) | **73** | 210 |

**Lectura:** con **más exploración** (decay 0.999, ε se mantiene alto más tiempo) el agente resuelve *más rápido* (73 episodios) y con política perfecta. Con **poca exploración** (0.990) resuelve pero converge a una política algo peor (test 198.5): al explotar prematuramente, se ancla a una solución subóptima antes de descubrir mejores estados. Es el dilema exploración/explotación en números: explorar más, aquí, sale a cuenta. *(Valores de 1 semilla — la tendencia es clara, pero hay ruido entre corridas.)*

## Síntesis

Las cuatro ablations cuentan una historia coherente con la teoría de [Mnih et al. 2015](/papers/dqn-nature-mnih-2015):

1. **El replay buffer es no-negociable** — sin él DQN diverge bajo el azar (9.4).
2. **El estado debe ser markoviano** — sin velocidades la tarea es irresoluble (45.6).
3. **La target network estabiliza y acelera** — aun en un problema fácil (100 vs 139.5 eps).
4. **Los hiperparámetros modulan la velocidad, no el techo** — en CartPole casi todo converge a 210; γ y ε-decay cambian *cuán rápido* y *cuán perfecto*.

---

**Ver también:** [Implementación de DQN](02-dqn-implementacion) · [Tarea: las dos preguntas](04-actividades) · [Clase 31 - Teoría](/clases/clase-31).
