---
title: "Asignación Húngara"
weight: 134
math: true
---

El **algoritmo húngaro** (Kuhn, 1955) resuelve el **problema de asignación**: dadas $n$ tareas, $n$ agentes y una matriz de costos $C$ donde $c_{ij}$ es el costo de asignar el agente $i$ a la tarea $j$, encontrar la asignación biyectiva de costo total mínimo. En [seguimiento de objetos](/fundamentos/seguimiento-de-objetos) es el paso que convierte una matriz de similitudes en una decisión: qué detección pertenece a qué trayectoria.

Este fundamento acompaña a la [Clase 42](/clases/clase-42).

---

## 1. El problema

Formalmente, se busca una permutación $\sigma$ de $\{1,\dots,n\}$ que minimice

$$\sum_{i=1}^{n} c_{i,\sigma(i)}$$

o, en forma de programa lineal entero con variables binarias $x_{ij} \in \{0,1\}$:

$$\min \sum_{i,j} c_{ij}\,x_{ij} \quad \text{sujeto a} \quad \sum_j x_{ij} = 1 \;\;\forall i, \qquad \sum_i x_{ij} = 1 \;\;\forall j$$

Es el emparejamiento perfecto de costo mínimo en un grafo bipartito completo.

{{< concept-alert type="clave" >}}
**Por qué no basta con lo obvio.** La solución codiciosa —tomar el par más barato, eliminarlo, repetir— no es óptima. Un ejemplo mínimo con dos agentes:

$$C = \begin{bmatrix} 1 & 2 \\ 3 & 100 \end{bmatrix}$$

El codicioso toma $c_{11}=1$ y queda obligado a $c_{22}=100$: total **101**. El óptimo es $c_{12}=2$ y $c_{21}=3$: total **5**. Aceptar un costo local peor puede evitar un desastre global.

Y la fuerza bruta tampoco sirve: hay $n!$ permutaciones. Para $n=20$ son $2{,}4\times 10^{18}$.
{{< /concept-alert >}}

El algoritmo húngaro lo resuelve en tiempo polinomial: $O(n^4)$ en la formulación original de Kuhn, $O(n^3)$ con la implementación de Jonker-Volgenant que usa `scipy.optimize.linear_sum_assignment`.

## 2. La idea del algoritmo

Se apoya en una observación de invariancia:

> **Restar una constante a toda una fila (o a toda una columna) de $C$ no cambia cuál es la asignación óptima.**

La razón es que toda asignación válida usa **exactamente una** celda de cada fila y de cada columna. Restar $k$ a la fila $i$ baja el costo total de *todas* las asignaciones en el mismo $k$, así que el orden entre ellas se conserva.

El algoritmo explota esto para transformar $C$ en una matriz equivalente con suficientes ceros como para que exista una asignación completa de costo 0 — que en la matriz transformada es trivialmente óptima, y por la invariancia lo es también en la original. Los pasos clásicos:

1. Restar a cada fila su mínimo.
2. Restar a cada columna su mínimo.
3. Cubrir todos los ceros con el mínimo número de líneas (horizontales o verticales).
4. Si el número de líneas es $n$, existe asignación completa sobre ceros: terminar.
5. Si no, sea $\delta$ el mínimo de los elementos no cubiertos. Restar $\delta$ a todas las filas no cubiertas y sumarlo a todas las columnas cubiertas. Volver a 3.

El paso 3 se apoya en el **teorema de König**: en un grafo bipartito, el tamaño del emparejamiento máximo es igual al tamaño de la cobertura mínima por vértices. Es la razón por la que "no se puede cubrir con menos de $n$ líneas" equivale a "no existe emparejamiento perfecto sobre los ceros actuales".

## 3. En tracking: los cuatro ajustes prácticos

La formulación pura asume matriz cuadrada y asignación completa. En MOT ninguna de las dos cosas se cumple, y hay que adaptarla.

**Matriz rectangular.** Hay $N$ trayectorias activas y $M$ detecciones, con $N \neq M$ casi siempre. Se rellena con filas o columnas ficticias de costo constante (o se usa una implementación que acepte matrices rectangulares, como la de SciPy, que devuelve un emparejamiento de tamaño $\min(N,M)$).

**Costo contra similitud.** El algoritmo minimiza. Las medidas naturales de tracking son similitudes (IoU alto = bueno), así que se convierten:

$$c_{ij} = 1 - \mathrm{IoU}(b_i, d_j) \qquad\text{o}\qquad c_{ij} = -\mathrm{IoU}(b_i, d_j)$$

**Umbral de rechazo.** El húngaro asigna *todo* lo que puede, incluso pares absurdos, si eso baja el total. En SORT esto se corrige a posteriori: se ejecuta la asignación y luego se **descartan** los pares cuya IoU quede por debajo de $\mathrm{IoU}_{\min}$. Las detecciones que quedan sueltas inician trayectorias nuevas; las trayectorias sueltas envejecen y eventualmente mueren.

**Compuertas (*gating*).** Antes de asignar, se pone a infinito el costo de los pares que ninguna consideración física admite: distancia de [Mahalanobis](/fundamentos/filtro-de-kalman) sobre $\chi^2_{0{,}95;4}=9{,}4877$, o apariencia por encima del umbral coseno. Esto no solo mejora la calidad: reduce el tamaño efectivo del problema.

## 4. Variantes que aparecen en la literatura

**Cascada de matching (DeepSORT).** En vez de un problema global, se resuelve una **secuencia** de problemas húngaros ordenados por edad de la trayectoria: primero las vistas hace 1 frame contra todas las detecciones, luego las de edad 2 contra las sobrantes, y así hasta $A_{\max}$. La razón es la patología de Mahalanobis descrita en el [fundamento del filtro de Kalman](/fundamentos/filtro-de-kalman): sin la cascada, las trayectorias viejas y muy inciertas se roban las detecciones de las jóvenes y confiables.

**Asociación en dos etapas por confianza (ByteTrack).** Primera ronda húngara con las detecciones de score alto; segunda ronda, con las trayectorias que quedaron sin par, contra las detecciones de score **bajo** que normalmente se descartarían. La intuición: una caja de score 0,4 sobre un objeto ocluido es basura para un detector pero información valiosa para un tracker que ya sabe que ahí hay algo.

**Asociación codiciosa.** Algunos sistemas de producción usan asignación codiciosa por su costo $O(nm\log nm)$ y por ser trivial de implementar en streaming. Con compuertas agresivas la diferencia con el óptimo suele ser pequeña, pero es una degradación real y conviene medirla, no asumirla.

**Formulación de flujo (offline).** En el régimen batch, el problema deja de ser bipartito de dos frames y se convierte en *min-cost flow* sobre el grafo espacio-temporal completo, donde una trayectoria es un camino y el algoritmo húngaro es reemplazado por un solver de flujo. Es la generalización natural de la asignación al caso de $T$ frames.

## 5. El costo computacional

Para $n$ del orden de las decenas —el caso típico de MOT— el húngaro es despreciable frente al detector. SORT reporta **260 Hz** para todo su componente de seguimiento en un solo núcleo de CPU, con el detector fuera de la cuenta. El cuello de botella nunca es la asignación; es la extracción de features de apariencia (en DeepSORT, aproximadamente la mitad del tiempo de cómputo) y, sobre todo, el detector.

---

## Ver también

- [Seguimiento de Objetos](/fundamentos/seguimiento-de-objetos) — dónde encaja este paso.
- [Filtro de Kalman](/fundamentos/filtro-de-kalman) — de dónde salen las predicciones que se asocian.
- [SORT](/papers/sort-bewley-2016) — la aplicación canónica: IoU + húngaro.
- [DeepSORT](/papers/deepsort-wojke-2017) — la cascada de matching.
- [ByteTrack](/papers/bytetrack-zhang-2021) — la asociación en dos rondas por confianza.
- [Clase 42 — Práctica](/clases/clase-42/practica) — implementación desde cero.
