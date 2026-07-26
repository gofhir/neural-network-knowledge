---
title: "Imitación y el covariate shift"
weight: 1
---

Antes de escribir una línea de código conviene entender **qué problema resuelve DAGGER** y por qué el enfoque ingenuo —copiar al experto— fracasa. Esta primera parte es el fundamento teórico: qué es el aprendizaje por imitación, por qué el *Behaviour Cloning* se rompe, y qué garantía formal aporta DAGGER.

## Imitación como aprendizaje supervisado

El [aprendizaje por imitación](/fundamentos/aprendizaje-por-imitacion) (IL) entrena una política π (mapeo estado → acción) **a partir de demostraciones de un experto**, en lugar de por ensayo-error maximizando una recompensa (como en [RL](/fundamentos/aprendizaje-reforzado)). La señal de aprendizaje no es un retorno futuro esperado, sino **la acción que el experto tomó en cada estado**.

Eso convierte la imitación en un problema de **clasificación supervisada**: el dataset es $\{(s_i, a_i^*)\}$ con $s_i$ un estado y $a_i^*$ la acción del experto. En este lab lo verás literal — la pérdida es una `cross_entropy` entre la acción del estudiante y la del experto, exactamente como en clasificación de imágenes.

El pariente cercano es el [aprendizaje reforzado inverso](/fundamentos/aprendizaje-reforzado-inverso) (IRL), que en vez de copiar la acción intenta **recuperar la recompensa** que el experto parece optimizar. Este lab **no** hace IRL: hace Behaviour Cloning y DAGGER, las dos técnicas de imitación directa.

## Behaviour Cloning: el enfoque ingenuo

BC es lo más simple imaginable: recolectas demostraciones del experto y entrenas un clasificador que, dado un estado, prediga su acción. Minimizas:

$$\hat\pi = \arg\min_\pi \; \mathbb{E}_{s \sim d_{\pi^*}}\big[\,\ell(\pi(s), \pi^*(s))\,\big]$$

Lo crucial está en el subíndice: **$s \sim d_{\pi^*}$** significa "estados muestreados de la distribución que visita el *experto*". Ahí está la trampa.

## El problema: covariate shift y cascada de errores

Todo aprendizaje supervisado asume que train y test vienen de la **misma distribución** (i.i.d.). En imitación, esa suposición **se viola por construcción**:

- Entrenaste sobre $d_{\pi^*}$ (estados del experto).
- Pero al desplegar al estudiante, es *él* quien conduce. Comete un error pequeño → llega a un estado **ligeramente distinto** de los que el experto visitaba → un estado sobre el que **nunca fue entrenado** → predice peor → error mayor → estado aún más raro → …

Esto es el **covariate shift**: la distribución de estados en test ($d_{\hat\pi}$, la del estudiante) difiere de la de train ($d_{\pi^*}$). El error no se acota: **se compone**.

{{< callout type="info" >}}
**La intuición del auto (Tesla).** Un conductor experto **siempre** va centrado en el carril, así que sus demostraciones **no contienen** ejemplos de "cómo volver al centro cuando estás por salirte". El estudiante que deriva al borde no tiene idea de cómo corregir —nunca vio ese estado— y se sale. BC no puede enseñar recuperación porque el experto nunca se equivoca.
{{< /callout >}}

## La cota cuadrática: por qué BC es catastrófico en horizontes largos

Ross, Gordon & Bagnell (2011) probaron que si el estudiante de BC tiene error de clasificación $\epsilon$ por paso, entonces sobre un episodio de horizonte $T$ el costo total crece como:

$$J(\hat\pi) \;\leq\; J(\pi^*) + O(\epsilon\,T^2)$$

El $T^2$ es la clave. El costo no crece lineal con la duración del episodio, sino **cuadrático**. Intuición: un error en el paso $t$ no cuesta solo ese paso — te empuja a una región mala donde pasas los $\sim(T-t)$ pasos restantes cometiendo más errores. Sumando sobre todos los $t$, sale $\propto T^2$. Para tareas largas (Breakout tiene cientos de pasos por episodio), ese $T^2$ es demoledor: un $\epsilon$ minúsculo se amplifica enormemente.

## Lo que logra DAGGER: de cuadrático a lineal

El teorema central de DAGGER reduce esa cota a **lineal**:

$$J(\hat\pi) \;\leq\; J(\pi^*) + O(\epsilon\,T)$$

Esa es, en una fórmula, **toda la ventaja de DAGGER sobre BC**. ¿Cómo lo logra? Cambiando el subíndice de la esperanza: en vez de entrenar sobre $d_{\pi^*}$ (estados del experto), entrena sobre $d_{\hat\pi}$ (**estados que el estudiante realmente visita**), pero pidiéndole al experto que los **etiquete**. Así el estudiante ve justamente los estados-de-error que BC nunca le mostraba, y aprende a recuperarse.

El apellido del método —**Dataset Aggregation**— es esto: *agregar* iterativamente al dataset los estados que visita el estudiante, etiquetados por el experto.

### La base teórica: No-Regret Online Learning

El aporte técnico de Ross et al. —y por qué el paper se titula *"A Reduction of Imitation Learning to No-Regret Online Learning"*— es demostrar que DAGGER equivale a un algoritmo de **online learning** (tipo Follow-the-Leader): en cada iteración el estudiante juega una política, el entorno le revela una nueva distribución de estados, y él se actualiza. Bajo esa lente, la garantía de "no-regret" del online learning se traduce en la cota lineal. Es una **reducción**: convierte un problema de control secuencial (con covariate shift) en un problema de online learning con teoría ya establecida.

## BC vs DAGGER, en una tabla

| | Behaviour Cloning | DAGGER |
|---|---|---|
| Entrena sobre | estados del **experto** $d_{\pi^*}$ | estados del **estudiante** $d_{\hat\pi}$ |
| Experto necesario | solo un dataset offline | **en el loop** (etiqueta online) |
| Cota de error | $O(\epsilon T^2)$ cuadrática | $O(\epsilon T)$ lineal |
| Aprende a recuperarse | ❌ nunca ve estados de error | ✅ ve y corrige sus propios errores |
| Problema que sufre | covariate shift / compounding | mitigado |

Esta tabla es la clave de las dos primeras preguntas de la tarea (ver [Resultados y tarea](04-resultados-y-tarea)): la **ventaja** de DAGGER sale de la fila de la cota; la **desventaja** de la fila "experto en el loop".

## Por qué Breakout y no un auto

El lab abre motivando con Tesla, pero implementa **Breakout** de Atari porque el experto puede ser un **DQN pre-entrenado** (no un humano): consultable gratis e infinitas veces, exactamente lo que DAGGER necesita para etiquetar cada estado nuevo. Es el "laboratorio de juguete" donde el bucle completo de DAGGER corre de punta a punta en minutos. La siguiente parte construye ese entorno.
