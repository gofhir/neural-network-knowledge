---
title: "El frame rate como parámetro de dificultad"
weight: 4
---

La clase plantea que el seguimiento online *supone que el objeto se movió poco* entre frames consecutivos. De ahí se sigue una predicción natural: **más frames por segundo deberían facilitar el seguimiento**, porque el desplazamiento por frame se reduce y el solapamiento entre cajas aumenta.

El experimento la refutó. Y la explicación resultó ser un mecanismo que no aparece en la formulación clásica del problema.

## El montaje

`store-aisle-detection.mp4` está grabado a **59,94 fps** — el único video del laboratorio con frame rate alto. Se procesó el mismo tramo de 600 frames de dos maneras:

- **V2a**: los 600 frames, a 59,94 fps.
- **V2b**: uno de cada cuatro (`step=4`), lo que equivale a **~15 fps**.

Mismo contenido, misma escena, mismas personas. La única variable es cuántas veces por segundo se ejecuta el tracker.

## El resultado

| | V2a · 59,94 fps | V2b · ~15 fps |
|---|---|---|
| Frames procesados | 600 | 150 |
| **Identidades emitidas** | **8** | **3** ✅ |
| Nacimientos abortados | 16 | **5** |
| Cajas duplicadas (IoU > 0,8) | 12 | **0** |
| Tiempo de proceso | 22,8 s | **6,0 s** |

Menos identidades, menos basura y cuatro veces más rápido, con el frame rate más bajo.

Y no es que se hayan perdido objetos. Los tres tracks reales sobreviven con las proporciones exactas del submuestreo:

| V2a (60 fps) | V2b (15 fps) | Razón |
|---|---|---|
| `id 1` · 600 detecciones · 100 % | `id 1` · 150 · **100 %** | 4,00 |
| `id 4` · 490 detecciones · 95,1 % | `id 3` · 127 · **99,2 %** | 3,86 |
| `id 12` · 236 detecciones · 100 % | `id 8` · 54 · **100 %** | 4,37 |

Las mismas tres personas, seguidas igual de bien o mejor: la cobertura del segundo track sube de 95,1 % a 99,2 %.

Los cinco identificadores adicionales de V2a —`20`, `21`, `22`, `23`, `24`— son tracks de 1 a 30 frames, es decir **entre 0,02 y 0,5 segundos**. Al submuestrear, esos eventos brevísimos simplemente no caen en la grilla de muestreo y desaparecen.

## El mecanismo que faltaba

El razonamiento sobre el IoU es correcto pero incompleto. Sí: a más fps, menor desplazamiento por frame, mayor solapamiento, asociación más fácil. Lo que omite es que **el tracker ejecuta un `update()` por frame**.

A 60 fps el tracker hace cuatro veces más actualizaciones para cubrir la misma escena. Cada actualización es una oportunidad de que una detección espuria supere `new_track_thresh` y funde una identidad:

$$\text{calidad de cada asociación} \;\propto\; \text{fps}
\qquad\qquad
\text{ocasiones de fallar} \;\propto\; \text{fps}$$

{{< concept-alert type="clave" >}}
**El frame rate tiene dos efectos de signo opuesto.** Sube la calidad de cada asociación individual y sube la cantidad de ocasiones de crear una identidad falsa. Cuál domina no depende de la física del movimiento sino de la **tasa de falsos positivos del detector**.

Con `conf = 0,1` —el valor que `model.track()` fuerza automáticamente para alimentar la segunda etapa de ByteTrack— esa tasa es alta, y el segundo efecto gana.
{{< /concept-alert >}}

## Los números que sostienen las dos mitades

**La mitad que la teoría predice bien.** A 59,94 fps, una persona caminando a 1,4 m/s se desplaza

$$\frac{1{,}4}{59{,}94} = 0{,}023\ \text{m} \approx 2\ \text{px por frame}$$

sobre una caja de unos 50 px de ancho, lo que da un IoU entre frames consecutivos de **0,96**. La asociación es trivial, y de hecho los tres tracks reales de V2a tienen cobertura de 95 a 100 %. La geometría, a 60 fps, no falla nunca.

**La mitad que no aparece en la formulación.** Los 16 nacimientos abortados de V2a contra los 5 de V2b escalan casi exactamente con la razón de frames procesados (600/150 = 4; 16/5 = 3,2). Los 12 duplicados contra 0 apuntan en la misma dirección: más pasadas del detector, más ocasiones de emitir una caja de sobra.

## Consecuencias prácticas

**Submuestrear puede ser una decisión de calidad, no solo de costo.** En vigilancia, procesar todos los frames de una cámara de 60 fps puede dar peores identidades que procesar uno de cada cuatro — y cuesta cuatro veces más.

**`track_buffer` debe escalarse con el frame rate.** El valor por defecto de 30 frames significa cosas muy distintas según el material:

| Video | fps | `track_buffer = 30` equivale a |
|---|---|---|
| `people-detection` | 12 | **2,50 s** |
| `car-detection` | 12,5 | 2,40 s |
| `store-aisle` | **59,94** | **0,50 s** |

En `store-aisle`, el track `id:4` recuperó su identidad tras dos oclusiones que sumaron 25 frames. Con el buffer por defecto tenía medio segundo de margen: una oclusión un poco más larga habría producido un ID switch. El caso extremo aparece en la Actividad 2 sobre el mismo video, donde un track sobrevive un hueco de **92 frames (1,5 segundos)** por siete muestras de margen respecto del buffer.

**Y el corolario incómodo:** los valores por defecto de un tracker están calibrados implícitamente para un frame rate. Los de Ultralytics apuntan a material de ~30 fps. Nada en la configuración lo declara.

## Una advertencia sobre la comparación

Las dos corridas no son perfectamente comparables en un aspecto: V2a procesa 600 frames y V2b procesa 150 del mismo tramo temporal. Un track espurio que dure dos frames a 60 fps tiene una probabilidad baja de caer en la grilla de muestreo de V2b, así que parte de la reducción es **filtrado por muestreo** más que mejora del algoritmo.

Eso no invalida el resultado —el objetivo es contar identidades emitidas, y a 15 fps se emiten menos para las mismas tres personas—, pero conviene enunciarlo con precisión: el submuestreo no hace que el tracker se equivoque menos por actualización; hace que haya menos actualizaciones en las que equivocarse, y las que quedan son de mejor calidad geométrica de lo que la reducción de fps sugeriría.

---

**Siguiente:** [Lo que el detector nombra mal](../05-lo-que-el-detector-nombra-mal) — un video de autos cuya clase mayoritaria es `cell phone`.
