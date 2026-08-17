---
title: "Practica desde 0 - Destilación cross-modal y fusión audiovisual"
weight: 30
sidebar:
  open: true
---

Los dos papers de la Clase 43 se apoyan en mecanismos que caben en unas pocas decenas de líneas, y cuyas afirmaciones centrales **se pueden comprobar sin entrenar nada grande**. Ese es el punto de estos dos caminos: no reproducir SoundNet ni E2E-AVSR —haría falta un año de video y semanas de GPU— sino aislar el mecanismo que cada uno usa y medirlo en un montaje controlado.

El primer camino ataca la brecha más llamativa de la clase: los **25 puntos** entre la pérdida KL y la $\ell_2$ en la ablación de SoundNet, que aparentemente contradicen un teorema de Hinton. Se implementa la destilación con temperatura, se verifica el teorema numéricamente, y se descubre que la contradicción se disuelve al notar que **"$\ell_2$" nombra dos pérdidas distintas**.

El segundo reconstruye la figura que justifica el paper de Petridis et al.: la curva de exactitud contra relación señal-ruido donde **la línea del video es horizontal**. Se mide de dónde sale esa horizontalidad, por qué el canal visual tiene un techo que no depende del ruido, y en qué condiciones la fusión **empeora** el resultado.

Cada uno en **triple framework**: PyTorch, TensorFlow y JAX. Los cuatro backends coinciden **hasta cero exacto** en la pérdida de destilación y hasta $4\times10^{-17}$ en su gradiente.

## Caminos

{{< cards >}}
  {{< card link="01-destilacion-cross-modal" title="01 - Destilación cross-modal" subtitle="Dark knowledge y temperatura, el teorema del límite de T alta verificado, y por qué L2 sobre probabilidades colapsa mientras L2 sobre logits gana" icon="code" >}}
  {{< card link="02-fusion-audiovisual" title="02 - Fusión audiovisual bajo ruido" subtitle="La curva de SNR reconstruida, el techo de los visemas, la fusión que empeora con pesos fijos, y por qué la aumentación con ruido no es regularización" icon="code" >}}
{{< /cards >}}

## Requisitos previos

- [Clase 43 - Teoría](/clases/clase-43/teoria) y la [Profundización](/clases/clase-43/profundizacion): el camino 01 implementa su Parte II y el camino 02 su Parte III.
- [Destilación de Conocimiento](/fundamentos/destilacion-de-conocimiento) y [Aprendizaje Audiovisual](/fundamentos/aprendizaje-audiovisual).
- Python intermedio y NumPy. Los ejemplos base no requieren ninguna librería de deep learning.
- **GPU no necesaria.** Todo corre en CPU en segundos.

## Tecnologías usadas

| Camino | Stack principal | Frameworks secundarios |
|---|---|---|
| 01 - Destilación cross-modal | NumPy | PyTorch (`F.kl_div`), TensorFlow, JAX (`grad`) |
| 02 - Fusión audiovisual | NumPy + SciPy | PyTorch, TensorFlow, JAX |

## Qué se verifica

| Afirmación | Dónde | Resultado |
|---|---|---|
| Los cuatro backends dan la misma pérdida de destilación | Camino 01 | diferencia **0,00e+00**; gradiente **4,16e−17** |
| La temperatura revela estructura que la etiqueta destruye | Camino 01 | entropía de 0,173 a 1,562 nats entre $T$=1 y $T$=10 |
| A $T$ alta, KL converge a mínimos cuadrados sobre logits | Camino 01 | coseno 0,9557 → **1,000000** |
| $\ell_2$ sobre probabilidades colapsa | Camino 01 | **7,57 %** contra 92,57 % de $\ell_2$ sobre logits |
| El gradiente de $\ell_2$ sobre probabilidades es menor por un factor exacto | Camino 01 | **$K/2$ = 200×** |
| Temperatura alta baja el top-1 y sube la correlación de rango | Camino 01 | 67,9→57,1 % y 0,858→0,984 |
| La modalidad visual es invariante al ruido acústico | Camino 02 | 78,4 % a −5 dB, 78,1 % a 20 dB |
| La ganancia de la fusión crece al bajar el SNR | Camino 02 | **+0,00** a 20 dB, **+71,36** a −5 dB |
| El error visual es confusión dentro del par ambiguo | Camino 02 | **90,7 %** de los errores |
| Sumar log-verosimilitudes supera a promediar probabilidades | Camino 02 | 83,49 % contra 79,16 % a −5 dB |
| La fusión con pesos fijos puede perjudicar | Camino 02 | **−5,49 puntos** con el canal visual roto |
