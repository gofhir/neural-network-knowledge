---
title: "Practica desde 0 - SORT y las métricas de MOT"
weight: 30
sidebar:
  open: true
---

Los dos objetos centrales de la Clase 42 se implementan completos en unas pocas decenas de líneas, y —lo más útil— **se pueden verificar contra una referencia calculada de otra forma**. El tracker se comprueba contra un *ground truth* sintético que se controla por completo; las métricas se comprueban reproduciendo los números publicados de MOT16 desde su definición.

El primer camino construye **SORT desde cero**: el filtro de Kalman con el estado de siete dimensiones del paper, el IoU por lotes, el algoritmo húngaro y la gestión del ciclo de vida. Sobre él se corren cuatro ablaciones que responden preguntas concretas: qué aporta el filtro, cuánto vale $T_{\text{lost}}$, cuándo el húngaro le gana al codicioso y a partir de qué ruido de detección el sistema colapsa.

El segundo implementa **MOTA, IDF1 y HOTA** desde su definición, reconstruye el ejemplo de tres trackers del paper de HOTA, despeja el $|\mathrm{gtDet}|$ que la tabla de DeepSORT no publica, y usa esa aritmética para medir cuánto vale realmente la contribución de DeepSORT. Cierra demostrando numéricamente la patología de Mahalanobis que obliga a la cascada de matching.

Cada uno en **triple framework**: PyTorch, TensorFlow y JAX. Los cuatro backends coinciden **hasta cero exacto** en el IoU por lotes y en el paso del filtro de Kalman.

## Caminos

{{< cards >}}
  {{< card link="01-sort-desde-cero" title="01 - SORT desde cero" subtitle="El estado de 7-D, predict/update batched, IoU vectorizado y el húngaro; y cuatro ablaciones: sin Kalman, T_lost, codicioso contra óptimo, y el colapso por ruido" icon="code" >}}
  {{< card link="02-metricas-mot" title="02 - Las tres métricas y lo que esconden" subtitle="MOTA, IDF1 y HOTA desde su definición; el ejemplo de los tres trackers reconstruido, los contrafactuales de MOT16, y por qué Mahalanobis premia la incertidumbre" icon="code" >}}
{{< /cards >}}

## Requisitos previos

- [Clase 42 - Teoría](/clases/clase-42/teoria) y la [Profundización](/clases/clase-42/profundizacion): el camino 01 implementa sus Partes I y IV, el camino 02 sus Partes II y III.
- [Detección de Objetos](/fundamentos/deteccion-de-objetos) para IoU y cajas; [Filtro de Kalman](/fundamentos/filtro-de-kalman) y [Asignación Húngara](/fundamentos/asignacion-hungara) para los dos componentes.
- Python intermedio, NumPy y `scipy.optimize.linear_sum_assignment`. Los ejemplos base no requieren ninguna librería de deep learning.
- **GPU no necesaria.** Todo corre en CPU en segundos. Es, precisamente, el argumento de SORT.

## Tecnologías usadas

| Camino | Stack principal | Frameworks secundarios |
|---|---|---|
| 01 - SORT desde cero | NumPy + SciPy | PyTorch (`torchvision.ops.box_iou`), TensorFlow, JAX (`vmap` + `jit`) |
| 02 - Las tres métricas | NumPy + SciPy | PyTorch, TensorFlow, JAX |

## Qué se verifica

| Afirmación | Dónde | Resultado |
|---|---|---|
| Los cuatro backends dan el mismo IoU por lotes | Camino 01 | diferencia máxima **0,000e+00** |
| Los cuatro backends dan el mismo paso de Kalman | Camino 01 | diferencia máxima **0,000e+00** en $x$ y en $P$ |
| Sin filtro de Kalman las identidades se intercambian al cruzarse | Camino 01 | IDF1 cae de 98,3 a **50,0** mientras MOTA *sube* |
| $T_{\text{lost}}$ debe superar la duración de la oclusión | Camino 01 | oclusión de 18 frames: IDF1 78,4 → **91,9** al pasar de $T_{\text{lost}}=1$ a 30 |
| El húngaro solo importa en escenas densas | Camino 01 | difiere en 1/20 semillas con 12 objetos; en **20/20** con 25 y ruido |
| El colapso por ruido de detección es abrupto | Camino 01 | MOTA 100 → 43,3 → **−78,3** entre $\sigma$ = 5, 10 y 20 px |
| La fórmula de MOTA reproduce la tabla de MOT16 | Camino 02 | **59,76** y **61,44** contra 59,8 y 61,4 publicados |
| Reducir los ID switches un 45 % vale, en MOTA | Camino 02 | **+0,35 puntos** |
| Los falsos positivos de $A_{\max}=30$ cuestan | Camino 02 | **−2,28 puntos**, 6,5 veces más |
| El ejemplo de HOTA da tres órdenes distintos | Camino 02 | MOTA C>B>A, IDF1 A>B>C, HOTA empate |
| Mahalanobis hace ganar a la trayectoria más incierta | Camino 02 | se invierte entre los **5 y 10 frames** de edad |
| La compuerta $\chi^2$ se autodesactiva | Camino 02 | radio admisible de 3,08 px a **300 px** en 30 frames |
