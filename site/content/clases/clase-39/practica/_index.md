---
title: "Practica desde 0 - Modelos de deep learning para audio"
weight: 30
sidebar:
  open: true
---

La [Clase 39](/clases/clase-39) especifica dos arquitecturas concretas —el "Ejemplo 1" sobre log-mel y el "Ejemplo 2" sobre onda cruda— y hace dos afirmaciones cuantitativas sobre ellas: que las convoluciones dilatadas permiten cubrir **"miles de timesteps tras pocas capas"**, y que la capa de reducción de dimensión **"reduce parámetros sin pérdida de exactitud"**. Esta práctica construye ambas arquitecturas desde cero y **verifica las dos afirmaciones con números**, en lugar de aceptarlas.

El primer camino mide el campo receptivo de una pila dilatada de tres maneras —con la fórmula, propagando gradientes, y contando posiciones muertas— y encuentra que el Ejemplo 2 del slide, con la progresión de dilataciones canónica, cubre 6.6 milisegundos. También deriva y verifica la progresión que sí funciona. El segundo camino arma la CLDNN del Ejemplo 1 y desglosa dónde vive cada parámetro, mostrando que la capa de reducción decide el tamaño de toda la red.

Cada camino en **triple framework**: PyTorch, TensorFlow y JAX/Flax. Todas las salidas que aparecen son reales, producidas ejecutando el código con `torch 2.8.0`, `tensorflow 2.20.0`, `jax 0.4.30` y `flax 0.8.5` sobre CPU.

## Caminos

{{< cards >}}
  {{< card link="01-campo-receptivo-y-dilatacion" title="01 - Campo receptivo y dilatación" subtitle="Medir el campo receptivo por gradiente, la condición que evita el gridding, y la auditoría del Ejemplo 2 del slide" icon="code" >}}
  {{< card link="02-la-cldnn-del-ejemplo-1" title="02 - La CLDNN del Ejemplo 1" subtitle="Construirla en tres frameworks, desglosar sus parámetros y demostrar por qué el pooling va solo en frecuencia" icon="code" >}}
{{< /cards >}}

## Requisitos previos

- [Clase 39 - Teoría](/clases/clase-39/teoria) y sobre todo la [Profundización](/clases/clase-39/profundizacion): el camino 01 implementa su Parte I y el camino 02 su Parte II.
- [Clase 35](/clases/clase-35) para la representación de la señal (STFT, log-mel), y [Clase 37](/clases/clase-37) para el manejo del dato.
- Python intermedio y NumPy; PyTorch a nivel de `nn.Module`. Útil pero no obligatorio: TensorFlow/Keras y JAX/Flax.
- **GPU no necesaria.** Todo lo que se mide acá son formas de tensores, conteos de parámetros y trazas de gradiente sobre entradas sintéticas. Ninguna medición cambia con hardware distinto.

## Tecnologías usadas

| Camino | Stack principal | Frameworks secundarios |
|---|---|---|
| 01 - Campo receptivo | PyTorch 2.x + NumPy | TensorFlow 2.x, JAX |
| 02 - CLDNN | PyTorch 2.x | TensorFlow 2.x / Keras, JAX + Flax |

## El hilo conductor

1. **Medir antes de creer.** El campo receptivo tiene una fórmula cerrada, pero también se puede medir empíricamente: se pone un impulso de gradiente en una posición de la salida y se observa qué posiciones de la entrada lo reciben. Los dos métodos deben coincidir — y coinciden. La ventaja del segundo es que revela algo que la fórmula esconde: **cuántas de las posiciones dentro del campo receptivo son realmente consultadas**. Con dilataciones mal elegidas, el 56% de ellas no lo son.

2. **La progresión de dilataciones no es una convención.** Existe una condición, $d_{l+1} \le R_l$, que separa las progresiones sanas de las que dejan huecos. Tomar el máximo permitido en cada capa da el crecimiento más rápido posible sin gridding. Con kernel 2 esa regla reproduce exactamente la duplicación de WaveNet; con los kernels del Ejemplo 2 da algo muy distinto.

3. **Dónde vive el costo de una red híbrida.** En la CLDNN, ni las convoluciones ni las capas densas dominan el conteo de parámetros: lo domina la matriz entrada→estado del primer LSTM, cuyo tamaño depende de lo que se le entregue. Cambiar la capa que precede al LSTM mueve el total de la red entre 5 y 19 millones de parámetros sin tocar nada más.

---

**Ver también:** [Clase 39 - Teoría](/clases/clase-39/teoria) · [Clase 39 - Profundización](/clases/clase-39/profundizacion) · Fundamentos: [Convoluciones dilatadas](/fundamentos/convoluciones-dilatadas) · [CRNN](/fundamentos/crnn) · [Clasificación de audio](/fundamentos/clasificacion-de-audio).
