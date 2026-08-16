---
title: "Lab 40 - Reconocimiento de acciones con TSM"
weight: 400
sidebar:
  open: true
---

**Profesora:** Bianca Del Solar Medrano
**Módulo:** Video — reconocimiento de acciones eficiente
**Notebook origen:** `clase_40/material/Laboratorio/Lab_40_Reconocimiento_de_Acciones_FINAL.ipynb`
**Notebook ejecutado:** [lab40.ipynb](/notebooks/lab40.ipynb) · [HTML](/notebooks-html/lab40.html)

## Encuadre

La contraparte práctica de la [clase 40](/clases/clase-40): inferencia con el checkpoint oficial de **[TSM](/papers/tsm-lin-2019)** —ResNet-50 preentrenada en Kinetics-400 con desplazamiento temporal— sobre videos de [UCF-101](/papers/ucf101-soomro-2012). No se entrena nada. El práctico pide una actividad de código —predecir la acción de `v_PlayingGuitar_g01_c01.avi`— y cinco preguntas teóricas.

Nada de eso es lo que hace interesante al laboratorio.

Lo interesante es que **el módulo que se estudia no tiene parámetros**, y eso permite manipularlo sobre un modelo ya entrenado sin tocar un solo peso. Con `fold_div` se anula, se intensifica o se convierte en unidireccional; el modelo se degrada o no según el video, y esa diferencia **mide cuánta temporalidad contiene cada acción**. Cuatro experimentos construidos sobre esa palanca convierten un tutorial de inferencia en una medición del mecanismo.

El resultado que ordena todo: el mismo desplazamiento vale **+82,76 puntos** en un video de salto alto y **+0,42** en uno de guitarra. TSM no es una mejora uniforme — es un **mecanismo de rescate** que se activa cuando la apariencia estática no alcanza.

## Resultados consolidados (medidos en el notebook)

### La predicción del tutorial y la de la actividad

| Video | Top-1 | Probabilidad | Resto del top-5 |
|---|---|---|---|
| `v_HighJump_g01_c02` | `high jump` | **99,12 %** | hurdling 0,86 · pole vault 0,01 · dunking basketball 0,00 · long jump 0,00 |
| `v_PlayingGuitar_g01_c01` | `playing guitar` | **40,66 %** | busking 24,04 · playing clarinet 13,03 · strumming guitar 6,74 · singing 3,17 |

Mismo modelo, mismo protocolo, dos videos correctamente clasificados con confianzas que difieren en 58 puntos.

### Cinco clips de la misma clase

| Video | Top-1 | p | Σ guitarra | busking |
|---|---|---|---|---|
| `g03_c01` | playing guitar | **99,49 %** | 100,00 % | **0,00 %** |
| `g04_c01` | playing guitar | 77,13 % | 86,30 % | 4,17 % |
| `g05_c01` | playing guitar | 59,63 % | 68,00 % | 6,04 % |
| `g01_c01` | playing guitar | 40,66 % | 48,62 % | 24,04 % |
| `g02_c01` | **playing harmonica** | 37,72 % | 24,34 % | 6,81 % |

La varianza dentro de una sola clase de UCF-101 va de **37,7 % a 99,5 %**. Y la variable que la ordena resulta visible al mirar los clips: **el micrófono**.

![Los cinco clips de PlayingGuitar ordenados por confianza](/laboratorios/lab-40/cinco-clips-guitarra.jpg)

`g03`, el único sin micrófono en cuadro, obtiene 99,49 % y **0,00 % de busking**. A medida que el micrófono se acerca a la boca, la confianza cae y las clases de contexto musical suben. En `g02` —micrófono directamente frente a la boca— el modelo predice `playing harmonica`, el instrumento que se sostiene exactamente ahí.

### La ablación del desplazamiento

Anulando los 16 módulos (`fold_div` grande ⇒ `fold = 0` ⇒ identidad), sobre los mismos pesos:

| Video | con TSM | sin TSM | Δ | top-1 sin TSM |
|---|---|---|---|---|
| `v_HighJump_g01_c02` | 99,12 % | 16,37 % | **+82,76** | high jump (16,4 %) |
| `v_PlayingGuitar_g01_c01` | 40,66 % | 10,39 % | +30,27 | **playing clarinet** (38,2 %) |
| `v_PlayingGuitar_g02_c01` | 18,92 % | 24,05 % | −5,13 | **playing clarinet** (28,7 %) |
| `v_PlayingGuitar_g03_c01` | 99,49 % | 99,06 % | **+0,42** | playing guitar (99,1 %) |
| `v_PlayingGuitar_g04_c01` | 77,13 % | 66,30 % | +10,83 | playing guitar (66,3 %) |
| `v_PlayingGuitar_g05_c01` | 59,63 % | 42,90 % | +16,72 | playing guitar (42,9 %) |

`g03` funciona como **control negativo natural**: si anular el módulo produjera una degradación genérica del modelo, también habría caído. Se movió 0,42 puntos. El efecto es específico del contenido temporal, no un artefacto de sacar al modelo de su distribución.

Y `playing clarinet` emerge como top-1 en dos clips al quitar el tiempo: guitarra y clarinete son **casi indistinguibles por pose estática**, y lo que los separa es el movimiento.

### La curva de proporción y el modo online

| Configuración | HighJump | g01 | g02 | g03 | g04 | g05 |
|---|---|---|---|---|---|---|
| bidireccional, 100 % desplazado | 0,52 % | 11,05 % | 54,21 % | 99,19 % | 51,51 % | 55,16 % |
| bidireccional, 50 % | 3,53 % | 48,01 % | 31,35 % | 99,86 % | 91,19 % | 44,75 % |
| **bidireccional, 25 %** (entrenado) | **99,12 %** | 40,66 % | 18,92 % | 99,49 % | 77,13 % | 59,63 % |
| bidireccional, 12,5 % | 93,19 % | 33,57 % | 26,43 % | 99,66 % | 85,49 % | 53,90 % |
| bidireccional, 6,2 % | 76,11 % | 22,81 % | 28,32 % | 99,51 % | 81,04 % | 41,38 % |
| identidad (sin shift) | 16,37 % | 10,39 % | 24,05 % | 99,06 % | 66,30 % | 42,90 % |
| **unidireccional, 12,5 %** (online) | **92,73 %** | 33,38 % | 24,34 % | 99,71 % | 88,29 % | 61,34 % |

Tres lecturas. El máximo cae **exactamente en la proporción de entrenamiento**. La curva es **asimétrica**: quedarse corto degrada de forma gradual, pasarse colapsa —desplazar el 100 % de los canales deja el video de salto alto en 0,52 %, que es el *naive shift* que el paper descarta. Y a igual presupuesto de canales, **el modo online iguala al offline** (92,73 % contra 93,19 %): la ventaja del bidireccional no viene de ver el futuro sino de desplazar el doble de canales — un resultado que reproduce por otra vía la Tabla 6 del paper original.

En los videos de guitarra el barrido no muestra **ningún patrón**: los óptimos caen en configuraciones distintas en cada clip y `g03` varía 0,8 puntos entre todas. Esa ausencia de estructura es, en sí misma, la firma de una acción cuasi-estática.

## Bloques del lab

{{< cards >}}
  {{< card link="01-el-shift-desarmado" title="El shift desarmado" subtitle="Las tres líneas que son todo el modelo, los 16 módulos verificados, el nombre del checkpoint como archivo de configuración y el cuarto de canales que no es un octavo" icon="beaker" >}}
  {{< card link="02-la-varianza-intra-clase" title="La varianza intra-clase y el micrófono" subtitle="De 37,7 % a 99,5 % en cinco clips de la misma acción; las cinco etiquetas de guitarra que se reparten 8 puntos, y el objeto de fondo que explica el resto" icon="chart-bar" >}}
  {{< card link="03-la-ablacion-del-shift" title="La ablación y el control g03" subtitle="Anular 16 módulos sin tocar un peso, los 82,76 puntos del salto alto contra los 0,42 de la guitarra, y el clarinete que emerge cuando se apaga el tiempo" icon="adjustments" >}}
  {{< card link="04-la-curva-de-proporcion" title="La curva de proporción y el modo online" subtitle="El pico exacto en el valor entrenado, la asimetría entre perder tiempo y corromper espacio, y el futuro que no aporta nada medible" icon="trending-down" >}}
  {{< card link="05-los-defectos-del-notebook" title="Los defectos del notebook" subtitle="El consejo críptico del principio explicado, la función de descarga que es un no-op, el GIF que muestra otro video y la GPU que nunca se usa" icon="code" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/desplazamiento-temporal" title="Desplazamiento Temporal" subtitle="El mecanismo completo: partial shift, residual shift, y las manipulaciones que este lab aplica" icon="book-open" >}}
  {{< card link="/fundamentos/reconocimiento-de-acciones" title="Reconocimiento de Acciones" subtitle="Tareas, datasets y por qué Kinetics y Something-Something miden cosas distintas" icon="book-open" >}}
  {{< card link="/fundamentos/analisis-de-video" title="Análisis de Video" subtitle="Video, movimiento y el muestreo que decide qué ve el modelo" icon="book-open" >}}
  {{< card link="/fundamentos/inflado-de-convoluciones" title="Inflado de Convoluciones" subtitle="La estrategia del lab 38 sobre los mismos videos: inflar en vez de desplazar" icon="book-open" >}}
  {{< card link="/fundamentos/transfer-learning" title="Transfer Learning" subtitle="Por qué el checkpoint de Kinetics se aplica a UCF-101 sin fine-tuning" icon="book-open" >}}
  {{< card link="/fundamentos/redes-convolucionales" title="Redes Convolucionales" subtitle="El bloque bottleneck y la conv1 exacta que el módulo envuelve" icon="book-open" >}}
{{< /cards >}}

## Papers de este laboratorio

{{< cards >}}
  {{< card link="/papers/tsm-lin-2019" title="TSM (2019)" subtitle="Lin, Gan y Han — el modelo del lab: el desplazamiento, sus dos correcciones y la tabla de offline contra online que este lab reproduce" icon="document-text" >}}
  {{< card link="/papers/tsn-wang-2016" title="TSN (2016)" subtitle="Wang et al. — el muestreo por segmentos y el consenso; la clase que instancia el modelo en el código se llama literalmente TSN" icon="document-text" >}}
  {{< card link="/papers/kinetics-kay-2017" title="Kinetics (2017)" subtitle="Kay et al. — las 400 clases del checkpoint, y las cinco etiquetas de guitarra que fragmentan la predicción" icon="document-text" >}}
  {{< card link="/papers/ucf101-soomro-2012" title="UCF-101 (2012)" subtitle="Soomro et al. — los 13 320 videos del lab, con sus grupos g01-g25 y sus clips de 10 segundos exactos" icon="document-text" >}}
  {{< card link="/papers/i3d-carreira-2017" title="I3D (2017)" subtitle="Carreira y Zisserman — el modelo del lab 38 sobre los mismos videos: 306 GFLOPs contra 33" icon="document-text" >}}
  {{< card link="/papers/something-something-goyal-2017" title="Something-Something (2017)" subtitle="Goyal et al. — el dataset donde el mismo módulo vale 28 puntos en lugar de 3,5" icon="document-text" >}}
{{< /cards >}}

---

**Ver también:** [Clase 40 - Teoría](/clases/clase-40/teoria) · [Clase 40 - Profundización](/clases/clase-40/profundizacion) · [Clase 40 - Práctica](/clases/clase-40/practica) · [Lab 38 - Action Recognition con I3D](/laboratorios/lab-38) (la estrategia opuesta sobre los mismos videos) · [Lab 36 - Análisis de Video](/laboratorios/lab-36) (el *bag of frames* que alcanza 85,9 % ignorando el orden) · Dominio [Video](/dominios/video).
