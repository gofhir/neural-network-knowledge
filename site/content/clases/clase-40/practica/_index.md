---
title: "Practica desde 0 - Desplazamiento temporal y muestreo por segmentos"
weight: 30
sidebar:
  open: true
---

La Clase 40 hace dos afirmaciones que suenan a truco y se pueden verificar en un notebook sin GPU ni datos: que **desplazar canales equivale a una convolución temporal** y que **el consenso por promedio es ciego al orden**. Esta práctica implementa ambos mecanismos desde cero y los somete a prueba numérica.

El primer camino construye el módulo de desplazamiento y verifica su equivalencia con una convolución $3\times1\times1$ —encontrando que la equivalencia tiene letra chica: los soportes son disjuntos y los grados de libertad son un tercio—, además de medir el alcance temporal efectivo de una pila de módulos. El segundo implementa las dos estrategias de muestreo que la clase contrapone, mide su cobertura sobre videos de distinta duración y demuestra que el promedio de TSN produce salidas idénticas ante cualquier permutación de los segmentos.

Cada uno en **triple framework**: PyTorch, TensorFlow y JAX.

## Caminos

{{< cards >}}
  {{< card link="01-el-modulo-de-desplazamiento" title="01 - El módulo de desplazamiento" subtitle="Las tres asignaciones, la equivalencia exacta con una convolución temporal de soportes disjuntos, los casos degenerados que habilitan la ablación y el alcance efectivo √L/2" icon="code" >}}
  {{< card link="02-muestreo-por-segmentos" title="02 - Muestreo por segmentos contra denso" subtitle="Cobertura constante contra ventana fija, la aumentación temporal implícita, y la demostración empírica de que el consenso por promedio es invariante al orden" icon="code" >}}
{{< /cards >}}

## Requisitos previos

- [Clase 40 - Teoría](/clases/clase-40/teoria) y sobre todo la [Profundización](/clases/clase-40/profundizacion): el camino 01 implementa y verifica sus Partes I-III, el camino 02 su Parte IV.
- [Clase 36](/clases/clase-36) para el panorama del análisis de video, y [Clase 38](/clases/clase-38) para la estrategia alternativa del inflado.
- Python intermedio y NumPy. PyTorch a nivel de `nn.Module` para las versiones de framework.
- **GPU no necesaria.** Todas las verificaciones corren sobre tensores pequeños en CPU en segundos.

## Tecnologias usadas

| Camino | Stack principal | Frameworks secundarios |
|--------|------------------|------------------------|
| 01 - El módulo de desplazamiento | NumPy + PyTorch | TensorFlow 2.x / JAX |
| 02 - Muestreo por segmentos | NumPy | PyTorch, TensorFlow 2.x, JAX |

## Qué se verifica

| Afirmación | Dónde | Resultado esperado |
|---|---|---|
| shift + conv $1\times1$ = convolución temporal | Camino 01 | igualdad exacta hasta error de máquina (~$10^{-14}$) |
| Los soportes de columna son disjuntos | Camino 01 | 3× menos parámetros que una $3\times1\times1$ genuina |
| `fold_div` grande anula el módulo | Camino 01 | identidad exacta |
| `fold_div = 2` deja el presente sin canales | Camino 01 | 0 canales sin desplazar |
| El alcance efectivo crece como $\sqrt{L}$ | Camino 01 | $\sigma_{16} = 2{,}00$ frames |
| El muestreo por segmentos cubre todo el video | Camino 02 | cobertura 100 % independiente de la duración |
| El consenso por promedio es invariante al orden | Camino 02 | salida idéntica ante las $8! = 40\,320$ permutaciones |
