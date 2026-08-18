---
title: "02 - El informed guess, medido"
weight: 20
math: true
---

> La clase define la super-resolución con dos palabras sobre un diagrama: ***informed guess***. Son exactas, y se pueden volver cuantitativas. Este camino mide cuánta información destruye la bajada de resolución, por qué la solución óptima en error cuadrático se ve borrosa, y por qué de eso se sigue que estas herramientas no sirven como evidencia.

---

## 1. Cuánta información se destruye

Con promediado por bloques de $f \times f$ sobre parches binarios, cada píxel de salida conserva únicamente **la suma del bloque**. Todas las configuraciones con la misma suma son indistinguibles.

```python
for f in [2, 3, 4]:
    n = f * f
    print(f, 2**n, n + 1, 2**n / (n + 1))
```

| factor | píxeles por bloque | parches HR posibles | valores LR posibles | preimagen media |
|---|---|---|---|---|
| 2× | 4 | 16 | 5 | 3,2 |
| 3× | 9 | 512 | 10 | 51,2 |
| 4× | 16 | **65 536** | **17** | **3855** |

A factor 4 la bajada es **3855 a 1**. El problema está mal planteado en el sentido de Hadamard: la solución no es única, y nada en la observación permite elegir entre las candidatas.

## 2. La solución óptima en MSE es el promedio

Si se entrena minimizando error cuadrático, la solución óptima es conocida:

$$\hat{x}_{\mathrm{MSE}} = \mathbb{E}[x \mid y]$$

el **promedio de todas las reconstrucciones compatibles**. Tomando una distribución de parches con estructura (bordes en posiciones aleatorias) y observando un píxel LR con suma 8 de 16:

```
Parches compatibles: 6598

Promedio de todas las compatibles     Una muestra de la posterior
  0.50  0.50  0.50  0.50                0  0  1  1
  0.50  0.50  0.50  0.50                0  0  1  1
  0.50  0.50  0.50  0.50                0  0  1  1
  0.50  0.50  0.50  0.50                0  0  1  1
```

| | MSE esperado | nitidez (varianza espacial) |
|---|---|---|
| **promedio** (óptimo en MSE) | **0,2500** | **0,0000** |
| **muestra** de la posterior | 0,4969 (1,99× peor) | 0,2500 |

El promedio resultó **gris uniforme**, con varianza espacial exactamente cero. Gana en MSE por construcción y es la única de las dos que **no puede ser una fotografía**.

{{< concept-alert type="clave" >}}
Esta es la explicación completa de por qué los primeros modelos de super-resolución entrenados con MSE producían resultados borrosos, y por qué no era un defecto de la arquitectura: **estaban resolviendo correctamente el problema que se les planteó**. El borrón es el óptimo del objetivo elegido.
{{< /concept-alert >}}

## 3. El intercambio distorsión-percepción

Interpolando entre el promedio y una muestra, y midiendo ambas cosas:

| $\alpha$ | MSE | PSNR (dB) | nitidez | distancia a la estadística real |
|---|---|---|---|---|
| 0,00 (promedio) | 0,2500 | **6,02** | 0,0000 | **0,2500** |
| 0,25 | 0,2654 | 5,76 | 0,0162 | 0,2338 |
| 0,50 | 0,3117 | 5,06 | 0,0633 | 0,1867 |
| 0,75 | 0,3889 | 4,10 | 0,1412 | 0,1088 |
| 1,00 (muestra) | 0,4969 | **3,04** | 0,2500 | **0,0000** |

Las dos columnas relevantes se mueven **en direcciones opuestas y monótonamente**. El punto que maximiza el PSNR es el que más se aleja de la estadística de las imágenes reales.

{{< concept-alert type="advertencia" >}}
No hay un punto que optimice ambas. Blau y Michaeli (2018) demostraron que existe un **límite teórico** al intercambio: por debajo de cierta frontera, mejorar la fidelidad obliga a empeorar el realismo.

Dos consecuencias que explican la literatura del área:

1. **Las métricas se dividieron en dos familias** — PSNR y SSIM miden distorsión; LPIPS, FID y NIQE miden realismo. Un método puede ganar en una y perder en la otra, y ambas cosas ser ciertas.
2. **Los modelos generativos ganan perceptualmente mientras pierden en PSNR.** No es un defecto de la evaluación: es el intercambio funcionando como se predijo.
{{< /concept-alert >}}

## 4. Dos priors, dos reconstrucciones

Del mismo conjunto de candidatos compatibles con el píxel LR observado, filtrando por estructura:

```
compatibles con borde estrictamente vertical  : 3255
compatibles con borde estrictamente horizontal: 3343

prior "bordes verticales"      prior "bordes horizontales"
    1  1  0  0                     0  0  0  0
    1  1  0  0                     0  0  0  0
    1  1  0  0                     1  1  1  1
    1  1  0  0                     1  1  1  1

las dos bajan al MISMO pixel LR: suma 8 y 8
```

Dos imágenes completamente distintas, ambas perfectamente consistentes con lo observado. Lo que las separa no es la evidencia: es el prior.

{{< concept-alert type="advertencia" >}}
**La super-resolución no es una operación forense.** Un rostro, una patente o un texto "recuperados" de un video de vigilancia de baja resolución son lo que el prior del modelo considera probable dado lo observado — no lo que había en la escena.

El caso conocido son los modelos de restauración facial que, aplicados a fotos pixeladas de personas de piel oscura, devolvían rostros de rasgos caucásicos. El resultado era plausible **según ese prior** y falso respecto de la realidad. En un contexto judicial, médico o de identificación, presentar una reconstrucción como evidencia equivale a presentar la opinión del modelo como si fuera un dato.

**El modelo no recupera información: la aporta.** Y el argumento se extiende a las otras aplicaciones de la clase — [Speech2Face](/papers/speech2face-oh-2019) reconstruye un rostro compatible con los atributos que la voz sugiere, no la cara de nadie en particular; sus propios autores lo declaran en la sección de consideraciones éticas del paper.
{{< /concept-alert >}}

## 5. La pregunta que generaliza a las siete aplicaciones

Las siete aplicaciones de la clase comparten esta estructura: **la información necesaria no está en la entrada**, y lo que la completa es un prior aprendido.

| Aplicación | Qué falta en la entrada | Qué lo completa |
|---|---|---|
| Speech from silent video | el sonido | la correlación labios-fonema del corpus |
| Face from voice | el rostro | la correlación voz-apariencia del corpus |
| Source separation | qué componente es de quién | el rostro, como referencia |
| Speech enhancement | la voz limpia | el modelo de habla |
| Super-resolución | las altas frecuencias | la estadística de imágenes naturales |
| Deep fakes | los cuadros que no se filmaron | el modelo de movimiento y apariencia |

La pregunta útil frente a cualquiera de ellas, y el mejor resumen práctico del cierre del diplomado:

> **¿Qué parte de esta salida estaba en la entrada, y qué parte la puso el prior?**

---

## Qué se aprendió

1. **La bajada de resolución a factor 4 es 3855 a 1.** El problema está mal planteado, y no por limitaciones del método.
2. **El óptimo en MSE tiene nitidez cero.** Los resultados borrosos de los primeros modelos eran la solución correcta del objetivo equivocado.
3. **Distorsión y percepción se mueven en direcciones opuestas**, con un límite teórico al intercambio.
4. **Dos priors producen reconstrucciones incompatibles** de la misma observación.
5. **Estas herramientas no son forenses**, y confundirlas con eso tiene consecuencias reales.

---

**Volver a:** [Práctica](../) · [Profundización](/clases/clase-44/profundizacion) · [Teoría](/clases/clase-44/teoria)
