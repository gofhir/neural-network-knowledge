---
title: "Lab 43 - End-to-End Audiovisual Speech Recognition sobre LRW"
weight: 430
sidebar:
  open: true
---

**Profesores:** Alain Raymond, Gabriel Sepúlveda y Álvaro Soto (IALab, DCC, Pontificia Universidad Católica de Chile)
**Módulo:** Audio y Video — donde confluyen los dos hilos
**Notebook origen:** `clase_43/material/Laboratorio/clase_43_lipreading.ipynb`
**Notebook ejecutado:** [lab43.ipynb](/notebooks/lab43.ipynb) · [HTML](/notebooks-html/lab43.html)

## Encuadre

La contraparte práctica de la [clase 43](/clases/clase-43), y el laboratorio donde se ejecuta el segundo de sus dos papers: [E2E-AVSR](/papers/e2e-avsr-petridis-2018) (Petridis, Stafylakis, Ma, Cai, Tzimiropoulos y Pantic, ICASSP 2018). Dos flujos end-to-end —uno que consume **píxeles crudos de la boca**, otro que consume **la forma de onda cruda**—, cada uno con su ResNet y su BiGRU, fusionados por un tercer BiGRU que modela la dinámica temporal conjunta.

No hay nada que entrenar: se descargan tres checkpoints, se evalúa sobre LRW y se responden tres preguntas de alternativa. Lo interesante aparece cuando se audita lo que se descargó.

{{< concept-alert type="clave" >}}
**El sistema acierta 2471 de 2500 clips (98,84 %) y los 29 errores no contienen ni uno arbitrario.** Todos son vecinos fonológicos: `SPEND→SPENT` con confianza 0,969 —difieren solo en la sonoridad de la consonante final—, `PHONE→THIRD` porque /f/ y /θ/ comparten visema, y `THERE→THEIR`, que en inglés británico son **/ðeə/** y **/ðeə/**: homófonos perfectos. Ese error es **irreducible**, y significa que LRW tiene un techo teórico bajo el 100 % por construcción del dataset.

Bajo azar, la probabilidad de que un error caiga justo sobre el pariente morfológico sería 0,2 %. Observado: **27,6 %** — un enriquecimiento de **~138×**.
{{< /concept-alert >}}

La tesis del laboratorio, en una línea: **el 1,16 % de error residual no es error de percepción sino de información ausente en la señal**. El sistema oye y ve bien; lo que no puede resolver es una `/s/` final de 50 ms, una diferencia de sonoridad invisible en los labios, o dos palabras que son físicamente el mismo evento acústico y articulatorio.

![Los 29 frames de un clip de LRW en escala de grises a 88x88: la entrada completa del stream de video](/laboratorios/lab-43/29-frames.jpg)

Eso es un clip entero para el stream visual: **1,16 segundos de una boca, sin color, sin el resto de la cara, a la resolución de un ícono**. De ahí sale una palabra entre 500 — y acierta el 82 % de las veces trabajando solo.

## Resultados consolidados (medidos)

### Evaluación sobre el test set completo de LRW

El mini test set distribuido resultó ser **500 palabras × 5 clips = 2500**, o sea el vocabulario íntegro a un décimo de densidad. La accuracy es directamente comparable al paper.

| | Medido | Paper |
|---|---|---|
| Accuracy audiovisual | **98,84 %** | 98,0 % |
| Errores | **29 / 2500** | ~50 / 2500 equivalentes |
| Error estándar (n = 2500) | 0,22 % | — |
| IC 95 % | **[98,41 %, 99,27 %]** | — |

El valor del paper cae **fuera** del intervalo. La explicación más plausible no es un modelo mejor que el publicado sino que el mini set son los 5 *primeros* clips de cada palabra, no 5 sorteados: si la numeración de LRW correlaciona con la calidad del alineamiento original de la BBC, son sistemáticamente más limpios.

### La anatomía de los 29 errores

| Tipo | n | Ejemplos |
|---|---|---|
| Homófonos perfectos | 1 | `THERE → THEIR` |
| Flexión morfológica | 8 | `SPEND→SPENT`, `QUESTIONS→QUESTION` |
| Contención léxica | 2 | `POSITION→OPPOSITION` |
| Rima o coda compartida | 7 | `ELECTION→ACTION` (×2), `WORDS→WOULD` |
| Esqueleto consonántico | 11 | `PHONE→THIRD`, `REASON→RECENT` |
| **Sin relación fonológica** | **0** | — |

Y el dato que cierra el argumento: **la confianza media en los errores morfológicos es 0,789 y en el resto 0,723.** El modelo no está menos seguro cuando confunde singular con plural — está *más* seguro. No es un modelo dudando, es un modelo convencido de una respuesta que la señal no permite descartar.

### El vocabulario tiene su propio techo

| Métrica sobre las 500 clases | Valor |
|---|---|
| Palabras en alguna familia morfológica | **103 (20,6 %)** |
| Familias distintas | 49 |
| Pares que difieren en ≤ 2 caracteres finales | **37** |

El stream de video del paper acierta 82,0 % — falla en el 18 %. El vocabulario tiene un 20,6 % de palabras con vecino confundible. Los sufijos que las separan son `/s/`, `/z/`, `/t/`, `/d/`: consonantes articuladas con la lengua, invisibles en los labios.

### Los tres checkpoints, auditados

| Checkpoint | Claves cargadas | Parámetros vivos | Parámetros **muertos** |
|---|---|---|---|
| Audio | 120/156 | 12.500.340 | **23,3 M (65 %)** |
| Video | 204/240 | 29.025.460 | 11,5 M (28 %) |
| Fusión | 18/18 | 13.107.700 | 0 |
| **Total** | | **54.633.500** | **34,9 M (39 % del ZIP)** |

Las 36 claves descartadas en cada stream resultaron ser **dos** estructuras muertas: el *backend temporal convolucional* de la fase 1 del entrenamiento —exactamente el que el paper dice haber "removido"— y un **BiLSTM completo** de una versión anterior del código, con topología de dos pilas independientes por dirección. Eso último explica un detalle que parecía un descuido: el `print("reload LSTM model")` sobre un objeto `GRU` **era literalmente correcto** para la versión que produjo estos pesos.

### El número 19456 descifrado

```
19456 muestras → conv1(k=80,s=4) → 4864 → /2/2/2 → 608 → avgpool(k=21) → 29
salto entre frames de audio :  672 muestras = 42,00 ms
frame de video a 25 fps     :  640 muestras = 40,00 ms
29 × 672 = 19488  ≈  19456
```

El recorte de audio no es arbitrario: es el largo exacto que hace que la cadena de convoluciones produzca **un frame cada 42 ms**, la cadencia de 25 fps del video. **La sincronización audiovisual está horneada en la aritmética de los strides** — no hay interpolación ni módulo de alineamiento.

## Bloques del lab

{{< cards >}}
  {{< card link="01-el-numero-19456" title="El número 19456" subtitle="Por qué el recorte de audio es 29 × 672 muestras, el campo receptivo de 71,5 ms con salto de 42, el crop fijo de la boca que cuesta 1 punto contra trackear landmarks, y la asimetría deliberada entre normalizar por clip y normalizar global" icon="variable" >}}
  {{< card link="02-los-dos-streams" title="Los dos streams" subtitle="Un frontend de 5 ms y 0,25 ms en lugar del banco mel, la Conv3D que no toca el tiempo, el reshape que vuelve espacial a la ResNet-34, el AvgPool2d que descarta 5 de 9 posiciones, y las dos discrepancias con el paper" icon="cube-transparent" >}}
  {{< card link="03-la-arqueologia-del-checkpoint" title="La arqueología del checkpoint" subtitle="Una función de carga que no puede fallar aunque no cargue nada, las 20 y 37 claves ausentes que cuadran al dígito con el conteo de BatchNorm, y las 36 muertas que son el registro material del entrenamiento por etapas" icon="beaker" >}}
  {{< card link="04-el-vocabulario-y-los-29-errores" title="El vocabulario y los 29 errores" subtitle="98,84 % contra el 98,0 % del paper y por qué la diferencia es de muestreo; los 37 pares mínimos de LRW, el enriquecimiento de 138× sobre el azar, y el error irreducible de dos homófonos perfectos" icon="exclamation" >}}
  {{< card link="05-los-defectos-del-notebook" title="Los defectos del notebook" subtitle="Un solo bug demostrable —una variable global filtrada que vuelve no determinista el índice impreso— más librosa sin backend para MP4, ffmpeg sin -y, un __len__ que puede mentir y un float64 que recorre todo el pipeline" icon="adjustments" >}}
  {{< card link="06-las-tres-actividades" title="Las tres actividades" subtitle="Robustez ante ruido, estabilidad del entrenamiento por etapas y dinámicas de corto plazo: las tres resueltas desde el código, con el descarte de cada alternativa falsa" icon="academic-cap" >}}
{{< /cards >}}

## Clase y fundamentos

{{< cards >}}
  {{< card link="/clases/clase-43" title="Clase 43 - Aplicaciones para Audio y Video" subtitle="El marco teórico: SoundNet usa la sincronía para entrenar, E2E-AVSR la usa para decidir" icon="academic-cap" >}}
  {{< card link="/clases/clase-43/profundizacion" title="Profundización de la clase" subtitle="La aritmética de SoundNet capa por capa, los 25 puntos entre KL y ℓ2, y la estructura de la complementariedad" icon="beaker" >}}
  {{< card link="/fundamentos/lectura-de-labios" title="Lectura de Labios" subtitle="Visemas y el techo de información, la arquitectura de tres bloques, LRW y el paso a oraciones" icon="book-open" >}}
  {{< card link="/fundamentos/aprendizaje-audiovisual" title="Aprendizaje Audiovisual" subtitle="La sincronía como etiqueta gratis, dónde fusionar y cuándo la fusión estorba" icon="book-open" >}}
  {{< card link="/fundamentos/representacion-de-audio" title="Representación de Audio" subtitle="Onda cruda contra MFCC: el empate de 97,7 % contra 97,7 % que este lab ejecuta" icon="book-open" >}}
  {{< card link="/fundamentos/lstm-gru" title="LSTM y GRU" subtitle="Las tres BiGRU del sistema, y el BiLSTM que el checkpoint conserva de una versión anterior" icon="book-open" >}}
  {{< card link="/fundamentos/redes-convolucionales" title="Redes Convolucionales" subtitle="ResNet, bloques residuales y la diferencia entre post-activación y pre-activación" icon="book-open" >}}
  {{< card link="/fundamentos/reconocimiento-de-voz" title="Reconocimiento de Voz" subtitle="La tarea que el stream de audio resuelve, y el contexto lingüístico que este modelo no tiene" icon="book-open" >}}
{{< /cards >}}

## Papers que aparecen en el laboratorio

{{< cards >}}
  {{< card link="/papers/e2e-avsr-petridis-2018" title="E2E Audiovisual Speech Recognition (2018)" subtitle="Petridis et al. — el paper que este lab ejecuta. +0,3 puntos en audio limpio, +14,1 a −5 dB: el video no mejora el caso fácil, sostiene el degradado" icon="document-text" >}}
  {{< card link="/papers/lipreading-resnet-stafylakis-2017" title="ResNet + LSTM para lipreading (2017)" subtitle="Stafylakis y Tzimiropoulos — la arquitectura que E2E-AVSR adopta, y la única línea de su tabla que no supera: 83,0 con ROI trackeada contra 82,0 con ROI fija" icon="document-text" >}}
  {{< card link="/papers/lipnet-assael-2016" title="LipNet (2016)" subtitle="Assael et al. — el primero a nivel de oración con CTC. La salida de largo variable que resolvería los errores de plural de este lab" icon="document-text" >}}
  {{< card link="/papers/av-hubert-shi-2022" title="AV-HuBERT (2022)" subtitle="Shi et al. — representaciones audiovisuales autosupervisadas sobre secuencias largas: el contexto lingüístico que separa 2018 de lo que vino después" icon="document-text" >}}
  {{< card link="/papers/soundnet-aytar-2016" title="SoundNet (2016)" subtitle="Aytar, Vondrick y Torralba — el otro paper de la clase, y la otra CNN 1D sobre onda cruda del módulo" icon="document-text" >}}
{{< /cards >}}

---

**Ver también:** [Lab 41 - Speaker Recognition](/laboratorios/lab-41) y [Lab 39 - Onda cruda y VGGish](/laboratorios/lab-39) (el hilo de audio) · [Lab 40 - TSM](/laboratorios/lab-40) y [Lab 38 - I3D](/laboratorios/lab-38) (el hilo de video, y la convolución 3D en toda la red en vez de solo en la primera capa) · [Lab 35 - Análisis de Audio](/laboratorios/lab-35) (el solapamiento de ventanas que reaparece en el campo receptivo) · Dominios [Audio](/dominios/audio) y [Video](/dominios/video).
