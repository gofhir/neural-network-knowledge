---
title: "Practica desde 0 - CTC y agregación VLAD"
weight: 30
sidebar:
  open: true
---

Los dos mecanismos centrales de la Clase 41 se pueden implementar en unas pocas decenas de líneas y —lo más útil— **verificar contra una referencia calculada por fuerza bruta**. Eso los convierte en el tipo de código que se entiende de verdad: no hay que creerle a nadie que el algoritmo sume sobre todas las alineaciones, se comprueba enumerándolas.

El primer camino implementa **CTC**: la función de colapso, la enumeración exhaustiva de alineaciones y la programación dinámica que las suma en tiempo polinomial. El segundo implementa **VLAD y NetVLAD**, muestra que el promedio es su caso degenerado, construye un ejemplo donde el promedio no puede distinguir lo que VLAD separa perfectamente, y cierra calculando una curva ROC con su EER.

Cada uno en **triple framework**: PyTorch, TensorFlow y JAX.

## Caminos

{{< cards >}}
  {{< card link="01-ctc-desde-cero" title="01 - CTC desde cero" subtitle="El colapso y su preimagen, la recursión de tres términos, la verificación contra fuerza bruta hasta el épsilon de máquina, y el blank obligatorio entre símbolos repetidos" icon="code" >}}
  {{< card link="02-agregacion-vlad" title="02 - Agregación VLAD" subtitle="Residuos contra promedio, el caso de idéntica media global que solo VLAD separa, la convergencia soft→hard, y el EER sobre una curva ROC" icon="code" >}}
{{< /cards >}}

## Requisitos previos

- [Clase 41 - Teoría](/clases/clase-41/teoria) y la [Profundización](/clases/clase-41/profundizacion): el camino 01 implementa sus Partes I-II y el camino 02 sus Partes IV-V.
- [Clase 13](/clases/clase-13) para seq2seq y atención, y [Clase 39](/clases/clase-39) para el contexto de audio.
- Python intermedio y NumPy. Los ejemplos base no requieren ninguna librería de deep learning.
- **GPU no necesaria.** Todo corre en CPU en segundos.

## Tecnologias usadas

| Camino | Stack principal | Frameworks secundarios |
|--------|------------------|------------------------|
| 01 - CTC desde cero | NumPy + `itertools` | PyTorch (`nn.CTCLoss`), TensorFlow, JAX |
| 02 - Agregación VLAD | NumPy | PyTorch, TensorFlow 2.x, JAX |

## Qué se verifica

| Afirmación | Dónde | Resultado |
|---|---|---|
| El forward de CTC suma exactamente sobre todas las alineaciones | Camino 01 | coincide con fuerza bruta, error ~$10^{-17}$ |
| El número de alineaciones es $\binom{T+U}{2U}$ | Camino 01 | 45, 495, 3 003, 12 870 para $T$ = 6, 8, 10, 12 |
| Un símbolo repetido reduce las alineaciones | Camino 01 | 70 en vez de las 126 que daría la fórmula |
| La implementación propia coincide con `nn.CTCLoss` | Camino 01 | igualdad hasta tolerancia numérica |
| El promedio es VLAD con $K=1$ y $c=0$ | Camino 02 | identidad exacta tras normalizar |
| NetVLAD converge a VLAD duro cuando $\tau \to 0$ | Camino 02 | coseno 1,000000 |
| Existe un caso que el promedio no puede distinguir | Camino 02 | margen 0,0000 contra 1,9998 |
