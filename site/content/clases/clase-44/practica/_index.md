---
title: "Practica desde 0 - Movimiento de primer orden y el informed guess"
weight: 30
sidebar:
  open: true
---

La clase termina en un laboratorio que **usa** un modelo preentrenado. Estos dos caminos hacen lo complementario: aíslan los dos mecanismos que hacen funcionar lo que la clase muestra, y los miden en un montaje controlado. No hay que entrenar nada — ambos corren en segundos y responden preguntas concretas.

El primero implementa el **modelo de movimiento de primer orden** de [FOMM](/papers/fomm-siarohin-2019): puntos clave, jacobianos, campo denso y *warping* diferenciable. Y responde la pregunta de diseño del paper: **¿cuánto vale realmente el jacobiano?** La respuesta tiene un giro — a igual presupuesto de parámetros, no siempre gana.

El segundo toma las dos palabras con que la clase define la super-resolución —***informed guess***— y las vuelve cuantitativas: cuánta información destruye la bajada de resolución, por qué la solución óptima en error cuadrático se ve borrosa, y por qué de ahí se sigue que estas herramientas **no sirven como evidencia forense**.

Cada uno en **triple framework**: PyTorch, TensorFlow y JAX. Los cuatro backends coinciden hasta $2{,}8\times10^{-16}$, y la implementación manual del *warping* coincide con `grid_sample` de PyTorch.

## Caminos

{{< cards >}}
  {{< card link="01-movimiento-primer-orden" title="01 - Movimiento de primer orden" subtitle="Puntos clave y jacobianos, el campo denso, back-warping bilineal, y cuánto vale el jacobiano — con el matiz de que a igual presupuesto no siempre gana" icon="code" >}}
  {{< card link="02-informed-guess" title="02 - El informed guess, medido" subtitle="La preimagen 3855 a 1, por qué el óptimo en MSE tiene nitidez cero, el intercambio distorsión-percepción, y dos priors que reconstruyen cosas distintas de lo mismo" icon="code" >}}
{{< /cards >}}

## Requisitos previos

- [Clase 44 - Teoría](/clases/clase-44/teoria) y la [Profundización](/clases/clase-44/profundizacion): el camino 01 implementa su Parte I y el camino 02 su Parte II.
- [Síntesis de Medios](/fundamentos/sintesis-de-medios) y [Super-resolución](/fundamentos/super-resolucion).
- Python intermedio y NumPy. Los ejemplos base no requieren ninguna librería de deep learning.
- **GPU no necesaria.**

## Tecnologías usadas

| Camino | Stack principal | Frameworks secundarios |
|---|---|---|
| 01 - Movimiento de primer orden | NumPy | PyTorch (`grid_sample`), TensorFlow, JAX (`vmap` anidado) |
| 02 - El informed guess | NumPy | — |

## Qué se verifica

| Afirmación | Dónde | Resultado |
|---|---|---|
| Los cuatro backends dan el mismo campo de movimiento | Camino 01 | máx. **2,8e−16** |
| El warping manual coincide con `grid_sample` | Camino 01 | **2,2e−16** |
| El jacobiano reduce el error a igual número de puntos | Camino 01 | **2,95×** con $K=10$ |
| Sobre rotación pura, la representación de orden 1 es exacta | Camino 01 | error ~10⁻¹⁶ contra 0,0086 a 40° |
| A igual presupuesto de parámetros, orden 0 gana con presupuestos chicos | Camino 01 | 48 y 72 parámetros |
| La bajada de resolución 4× es no invertible | Camino 02 | **3855 a 1** |
| El óptimo en MSE tiene nitidez cero | Camino 02 | 0,0000 contra 0,2500 |
| Distorsión y percepción se mueven en direcciones opuestas | Camino 02 | PSNR 6,02 → 3,04 |
| Dos priors dan reconstrucciones incompatibles del mismo LR | Camino 02 | ambas con suma 8 |
