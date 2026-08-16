---
title: "02 - La varianza intra-clase y el micrófono"
weight: 20
math: true
---

> La actividad del práctico pide predecir un video de guitarra. El modelo acierta con 40,66 % de confianza, contra el 99,12 % del video de salto alto que trae el tutorial. Perseguir esa diferencia —con cuatro clips más y una mirada a las imágenes— lleva a una variable que no está en ninguna parte del enunciado.

---

## 1. El punto de partida: dos aciertos muy distintos

Mismo modelo, mismo pipeline, mismo protocolo de evaluación:

```
Top 5 — v_HighJump_g01_c02              Top 5 — v_PlayingGuitar_g01_c01
  high jump           : 99.12%            playing guitar      : 40.66%
  hurdling            :  0.86%            busking             : 24.04%
  pole vault          :  0.01%            playing clarinet    : 13.03%
  dunking basketball  :  0.00%            strumming guitar    :  6.74%
  long jump           :  0.00%            singing             :  3.17%
```

Los dos son correctos. La diferencia de 58 puntos en la confianza es lo que hay que explicar.

El top-5 del salto alto es semánticamente impecable: `hurdling`, `pole vault` y `long jump` son atletismo de pista con carrera y despegue, y `dunking basketball` comparte el salto vertical con elevación de brazos. El modelo está viendo lo que corresponde.

El de la guitarra es más raro. `busking` —tocar en la calle por propinas— y `singing` no describen el instrumento ni el gesto, sino la situación. Y `playing clarinet` es directamente otro instrumento, que no se parece a una guitarra.

---

## 2. Primera hipótesis: el vocabulario está fragmentado

Kinetics-400 contiene **cinco etiquetas de guitarra**:

| Índice | Clase |
|---|---|
| 221 | `playing bass guitar` |
| 232 | `playing guitar` |
| 249 | `playing ukulele` |
| 335 | `strumming guitar` |
| 350 | `tapping guitar` |

`high jump` (índice 151), en cambio, **no tiene sinónimos** en el vocabulario. La hipótesis natural es que la probabilidad de la guitarra se reparte entre etiquetas casi sinónimas y por eso el top-1 baja.

Es contrastable: basta sumar la masa sobre las cinco clases y ver cuánto se recupera. Con cinco clips de la misma categoría —`Σ guitarra` es esa suma:

| Video | Top-1 | p | Σ guitarra | brecha | busking |
|---|---|---|---|---|---|
| `g03_c01` | playing guitar | **99,49 %** | 100,00 % | 0,51 | **0,00 %** |
| `g04_c01` | playing guitar | 77,13 % | 86,30 % | 9,17 | 4,17 % |
| `g05_c01` | playing guitar | 59,63 % | 68,00 % | 8,37 | 6,04 % |
| `g01_c01` | playing guitar | 40,66 % | 48,62 % | 7,96 | 24,04 % |
| `g02_c01` | **playing harmonica** | 37,72 % | 24,34 % | — | 6,81 % |

La hipótesis se confirma **y se dimensiona**: la brecha entre el top-1 y Σ guitarra es consistente —9,2 / 8,4 / 8,0 puntos— pero **pequeña**. Las etiquetas hermanas se llevan unos 8 puntos, no los 58 que hay que explicar. Y en `g03` la fragmentación es literalmente cero: Σ guitarra = 100,00 % con top-1 = 99,49 %, o sea que las otras cuatro clases juntas se quedaron con medio punto.

{{< concept-alert type="nota" >}}
La fragmentación del vocabulario es real, medible y **de segundo orden**: ~8 puntos. Sirvió como hipótesis y el experimento la acotó. Lo que queda sin explicar es la varianza: **de 37,7 % a 99,5 % dentro de una sola clase de UCF-101**, un factor de 2,6×. Con esa dispersión, cualquier conclusión sacada de un solo video —como hace el tutorial— es esencialmente ruido.
{{< /concept-alert >}}

También cae una segunda hipótesis por el camino. `busking` y `playing clarinet`, que en `g01` parecían indicar que "el modelo integra el contexto de la escena", **no son sistemáticos**: busking va de 0,00 % a 24,04 % según el clip y clarinet no reaparece. Eran propiedades de ese video, no del modelo ni de la clase.

---

## 3. Lo que no explica la varianza

Antes de mirar las imágenes conviene descartar lo obvio.

**No es la duración.** Los cinco clips tienen exactamente **250 frames**. `ffprobe` lo confirma: `10.000000` segundos a 25 fps, y `nb_frames = 250` coincide con los JPEGs extraídos. No hay truncamiento en la cadena de preproceso, y la uniformidad es una propiedad de cómo se segmentó esta clase de UCF-101.

**No es el muestreo.** Con 250 frames y 8 segmentos, `_get_test_indices` produce siempre `[16, 47, 79, 110, 141, 172, 204, 235]`: los mismos índices relativos para los cinco.

**No es el preproceso.** Idéntico en los cinco: `scale=-1:331` → `GroupScale(256)` → `GroupCenterCrop(224)`.

Con esas tres descartadas, la variable tiene que estar en el contenido.

---

## 4. Mirar los clips

El frame central de cada uno, ordenados por la confianza obtenida:

![Los cinco clips de PlayingGuitar ordenados por confianza](/laboratorios/lab-40/cinco-clips-guitarra.jpg)

| clip | confianza | qué hay en cuadro |
|---|---|---|
| `g03` | 99,5 % | persona sentada en un sofá, plano frontal, guitarra despejada, **sin micrófono** |
| `g04` | 77,1 % | plano general, guitarra visible, **micrófono con pie**, lateral y alejado |
| `g05` | 59,6 % | **micrófono en primer plano** cruzando el cuadro, amplificador al fondo |
| `g01` | 40,7 % | **micrófono grande, cerca de la cara** |
| `g02` | 37,7 % | **micrófono justo frente a la boca**, fondo oscuro de escenario |

**La confianza decrece monótonamente con la prominencia del micrófono.** Los cinco clips tienen una sola persona, guitarra visible y encuadre comparable; lo que varía es un objeto de fondo.

Y no es solo una impresión visual: está en los números de la tabla anterior. **`busking` en `g03` es 0,00 %** —el único clip sin micrófono— y entre 4,2 % y 24,0 % en los otros cuatro.

---

## 5. Qué significa

`busking` no es una clase de "contexto genérico": describe **tocar música en la vía pública**, y su firma visual es el setup de actuación — micrófono, pie, amplificador, público. El modelo no está confundido sobre el instrumento; está respondiendo a una pregunta distinta de la que UCF-101 hace. UCF-101 pregunta *qué hace la persona*; Kinetics tiene etiquetas que responden *en qué situación está*.

El caso de `g02` es el más elocuente. Es el único error de los cinco, y la clase predicha es **`playing harmonica`**. La armónica es el instrumento que se sostiene con ambas manos **justo frente a la boca** — exactamente donde está el micrófono en ese clip. El modelo no confundió la guitarra con otra cosa: leyó un objeto delante de la boca y respondió en consecuencia.

{{< concept-alert type="clave" >}}
La variable que gobierna la varianza intra-clase no es la duración, ni el encuadre, ni el muestreo: es **un objeto de fondo que activa clases de contexto musical**. Con cinco clips y una variable ordinal juzgada visualmente esto es una observación fuerte, no una prueba —pero el 0,00 % de busking en el único clip sin micrófono es evidencia cuantitativa dentro de los datos, no interpretación de las imágenes.
{{< /concept-alert >}}

---

## 6. Por qué la guitarra es difícil y el salto alto no

Hay una segunda razón, independiente del micrófono, y tiene que ver con el muestreo.

Con 250 frames y 8 segmentos, el muestreo por segmentos toma **un frame cada 31, es decir cada 1,25 segundos**. A esa escala, tocar la guitarra es una acción **cuasi-estática**: el rasgueo ocurre varias veces por segundo y es completamente invisible entre dos muestras separadas por 1,25 s. Los 8 frames son casi la misma imagen, y el desplazamiento temporal —que mezcla instantes vecinos— encuentra poco que mezclar.

En `HighJump`, los 8 frames cubren **carrera → despegue → vuelo → caída**: una progresión ordenada e inequívoca. Ahí el desplazamiento tiene material.

Esa asimetría se puede medir directamente apagando el módulo, y es lo que hace el [siguiente bloque](03-la-ablacion-del-shift): en el video de salto alto la ablación cuesta **82,76 puntos**; en `g03`, **0,42**.

---

## 7. La lección metodológica

El tutorial concluye sobre un video. Con la varianza medida —37,7 % a 99,5 % en cinco clips de la misma clase— esa práctica no permite distinguir una propiedad del modelo de una propiedad del clip.

Concretamente, tres afirmaciones plausibles que el experimento con cinco clips **refutó o acotó**:

| Afirmación desde n=1 | Qué mostró n=5 |
|---|---|
| "El modelo integra el contexto de la escena" (busking 24 %) | busking va de 0,00 a 24,04 según el clip: propiedad del video |
| "Confunde guitarra con clarinete por la postura" | clarinet no reaparece con el shift activo — pero **sí** al apagarlo, ver el bloque 03 |
| "La fragmentación del vocabulario explica la baja confianza" | explica ~8 puntos de los 58 |

La tercera es el caso interesante: la hipótesis era correcta en el mecanismo y equivocada en la magnitud. Medirla no la descartó — la puso en su lugar.

---

## Ver también

- [03 - La ablación del shift](03-la-ablacion-del-shift) — el clarinete vuelve, y esta vez significa algo.
- [01 - El shift desarmado](01-el-shift-desarmado) — cómo se construye el modelo que produce estas predicciones.
- [Clase 40 - Práctica: muestreo por segmentos](/clases/clase-40/practica/02-muestreo-por-segmentos) — la cobertura constante y el paso de 31 frames, medidos.
- [Paper: Kinetics](/papers/kinetics-kay-2017) — las 400 clases y su granularidad.
- [Laboratorio 36](/laboratorios/lab-36) — el mismo fenómeno desde el otro lado: un modelo sin orden temporal que alcanza 85,9 %.
