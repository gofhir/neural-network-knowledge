---
title: "Super-resolución"
weight: 141
math: true
---

La **super-resolución** aumenta la resolución de una señal —imagen, video o audio— más allá de la que tiene. La [Clase 44](/clases/clase-44) la describe con dos palabras que vale la pena tomar en serio: ***informed guess***, una conjetura informada. No es una figura retórica: es la descripción exacta de lo que ocurre, y este fundamento hace precisa esa frase.

---

## 1. Un problema mal planteado

Bajar la resolución es una operación que **destruye información**, y ninguna cantidad de procesamiento la recupera. Con un promediado por bloques de $f \times f$, cada píxel de salida conserva solo la suma del bloque; todas las configuraciones con la misma suma son indistinguibles.

Cuantificado sobre parches binarios (**medido**):

| factor | píxeles por bloque | parches HR posibles | valores LR posibles | preimagen media |
|---|---|---|---|---|
| 2× | 4 | 16 | 5 | 3,2 |
| 3× | 9 | 512 | 10 | 51,2 |
| 4× | 16 | **65 536** | **17** | **3855** |

A factor 4, la bajada es **3855 a 1**. La preimagen de una observación es enorme, y elegir un elemento de ella no es recuperación: es **selección según un prior**.

{{< concept-alert type="clave" >}}
Formalmente, si $y = Hx$ con $H$ el operador de bajada, el problema de hallar $x$ está mal planteado en el sentido de Hadamard: la solución **no es única**. Lo único que se puede hacer es imponer una distribución previa $p(x)$ y trabajar con la posterior $p(x \mid y)$.

Toda la super-resolución moderna es una elección de prior. Los métodos clásicos usaban priors explícitos y débiles (suavidad, variación total); los métodos aprendidos usan un prior implícito y fortísimo: **la distribución del conjunto de entrenamiento**.
{{< /concept-alert >}}

## 2. Por qué el óptimo en MSE se ve borroso

Si se entrena minimizando el error cuadrático, la solución óptima es conocida y es la **esperanza condicional**:

$$\hat{x}_{\mathrm{MSE}} = \mathbb{E}[x \mid y]$$

Es decir, el **promedio de todas las reconstrucciones compatibles**. Y el promedio de un conjunto de imágenes nítidas y distintas no es una imagen nítida: es un borrón.

Medido sobre un píxel LR concreto y sus reconstrucciones compatibles:

| | MSE esperado | nitidez (varianza espacial) |
|---|---|---|
| **promedio** (óptimo en MSE) | **0,2500** | **0,0000** |
| **muestra** de la posterior | 0,4969 | 0,2500 |

El promedio resultó ser gris uniforme en todo el bloque —varianza espacial exactamente cero— mientras que una muestra es un borde nítido. El promedio **gana** en MSE por definición y es la única de las dos que **no puede ser una fotografía**.

## 3. El intercambio distorsión-percepción

Interpolando entre ambos extremos y midiendo las dos cosas a la vez (**medido**):

| $\alpha$ | MSE | PSNR (dB) | nitidez | distancia a la estadística real |
|---|---|---|---|---|
| 0,00 (promedio) | 0,2500 | **6,02** | 0,0000 | **0,2500** |
| 0,25 | 0,2654 | 5,76 | 0,0162 | 0,2338 |
| 0,50 | 0,3117 | 5,06 | 0,0633 | 0,1867 |
| 0,75 | 0,3889 | 4,10 | 0,1412 | 0,1088 |
| 1,00 (muestra) | 0,4969 | **3,04** | 0,2500 | **0,0000** |

Las dos columnas relevantes se mueven **en direcciones opuestas y monótonamente**. El punto que maximiza el PSNR es el que más se aleja de la estadística de las imágenes reales, y viceversa. No hay un punto intermedio que optimice ambas.

{{< concept-alert type="advertencia" >}}
Esto no es un artefacto del montaje: Blau y Michaeli (2018) demostraron que existe un **límite teórico** al intercambio entre distorsión y calidad perceptual, y que mejorar una obliga a empeorar la otra por debajo de cierta frontera.

De ahí dos consecuencias prácticas que explican la literatura del área:

1. **Las métricas se dividieron en dos familias.** PSNR y SSIM miden distorsión; LPIPS, FID y NIQE miden realismo. Un método puede ganar claramente en una y perder en la otra, y ambas cosas ser ciertas.
2. **Los modelos generativos ganan perceptualmente mientras pierden en PSNR.** Las GAN y los modelos de difusión aplicados a super-resolución producen resultados que se ven mejor y puntúan peor. No es un defecto de la evaluación: es el intercambio, funcionando como se predijo.
{{< /concept-alert >}}

## 4. "Informed guess" tomado en serio

Dos priors distintos, aplicados a la **misma** observación, producen reconstrucciones incompatibles entre sí (**medido**, ambos parches bajan al mismo píxel LR de suma 8):

```
prior "bordes verticales"      prior "bordes horizontales"
    1  1  0  0                     0  0  0  0
    1  1  0  0                     0  0  0  0
    1  1  0  0                     1  1  1  1
    1  1  0  0                     1  1  1  1
```

Las dos son perfectamente consistentes con lo observado. **El modelo no recupera información: la aporta.**

{{< concept-alert type="clave" >}}
De aquí se sigue la advertencia práctica más importante del tema: **la super-resolución no es una operación forense**. Un rostro, una patente o un texto "recuperados" de un video de vigilancia de baja resolución son lo que el prior del modelo considera probable dado lo observado — no lo que había en la escena.

El caso conocido es el de los modelos de restauración facial que, aplicados a fotos pixeladas de personas de piel oscura, devolvían rostros de rasgos caucásicos: el prior estaba dominado por el conjunto de entrenamiento, y el resultado era plausible según ese prior y falso respecto de la realidad. En un contexto judicial, médico o de identificación, presentar una reconstrucción como evidencia es presentar la opinión del modelo como si fuera un dato.
{{< /concept-alert >}}

## 5. En audio

El mismo problema con otra geometría: el submuestreo elimina las frecuencias por encima de Nyquist, y reconstruirlas es inventar contenido espectral que no está. Los métodos de [super-resolución de audio](/papers/audio-superres-kuleshov-2017) predicen esas bandas con arquitecturas de codificador-decodificador convolucionales, análogas a las de imagen.

La particularidad es perceptual: el oído es más tolerante a la invención de armónicos plausibles que el ojo a la invención de rasgos faciales — un armónico inventado suena razonable, un rasgo inventado cambia quién es la persona. Eso hace que el intercambio distorsión-percepción sea, en audio, menos delicado en sus consecuencias.

---

## Ver también

- [Audio Super Resolution (2017)](/papers/audio-superres-kuleshov-2017) — el caso de audio.
- [Modelos Generativos](/fundamentos/modelos-generativos) y [Modelos de Difusión](/fundamentos/modelos-de-difusion) — los priors modernos.
- [Síntesis de Medios](/fundamentos/sintesis-de-medios) — la otra técnica donde el modelo aporta lo que no estaba.
- [Clase 44 — Práctica](/clases/clase-44/practica) — todos estos números, reproducibles.
