---
title: "musicnn: CNN con filtros musicalmente motivados para music tagging (2019)"
weight: 432
math: true
---

{{< paper-card
    title="musicnn: Pre-trained Convolutional Neural Networks for Music Audio Tagging"
    authors="Jordi Pons, Xavier Serra (Music Technology Group, Universitat Pompeu Fabra)"
    year="2019"
    venue="ISMIR 2019 (Late-Breaking/Demo) / arXiv:1909.06654"
    pdf="/papers/musicnn-pons-2019.pdf" >}}
Conviene decirlo antes que nada, porque casi todas las citas de este trabajo lo confunden: **musicnn no es un paper de arquitectura nueva.** Es una **librería de Python con cinco modelos preentrenados** (`MTT_musicnn`, `MSD_musicnn`, `MSD_musicnn_big`, `MTT_vgg`, `MSD_vgg`) y un API de tres funciones para etiquetar música, extraer *features* y transferir a otras tareas musicales; el documento que la anuncia es un **extended abstract de dos páginas** de la sesión Late-Breaking/Demo de ISMIR 2019, marcado `[Unrefereed]` en cada página del PDF. Toda la sustancia técnica —por qué un espectrograma no es una imagen, las formas de filtro, el experimento de escala que compara onda cruda contra espectrograma— está en **tres papers previos del mismo grupo**: CBMI 2016 (filtros con forma musical), EUSIPCO 2017 (invarianzas y timbre) e ISMIR 2018 (*End-to-end learning for music audio tagging at scale*, arXiv:1711.02520). La tesis que sostiene la línea completa cabe en una frase: en un espectrograma el eje vertical es frecuencia y el horizontal es tiempo, **no son intercambiables**, y por lo tanto la forma del filtro convolucional es una hipótesis sobre la estructura de la señal — el cuadrado $3\times3$ heredado de visión es una hipótesis, no la ausencia de una. El valor práctico de musicnn fue otro: entre 2019 y ~2022 sus *embeddings* fueron el extractor de *features* por defecto del MIR aplicado, el equivalente musical de lo que [VGGish](/papers/vggish-hershey-2017) fue para sonidos generales.
{{< /paper-card >}}

---

## Contexto: el music audio tagging como tarea

El *music audio tagging* consiste en **estimar los atributos de una canción a partir de su audio**, sin metadatos ni letra: *"moods, language of the lyrics, year of composition, genres, instruments, harmony, or rhythmic traits"*. El dataset privado de 1,2 M de canciones de ISMIR 2018 tiene 139 anotaciones por pista hechas por expertos, agrupadas en familias como métrica (triple-meter, cut-time), *feel* rítmico (swing, shuffle, syncopation), armonía, *mood* (angry, sad, joyful), voz, instrumentación, sonoridad (studio, live, acoustic) y género/subgénero.

Lo interesante es que **el vocabulario no es homogéneo**. `piano` es un evento acústico localizado; `swing` es una propiedad de la organización temporal a escala de compases; `00s` es una etiqueta de producción y contexto cultural sin firma acústica directa. Un mismo modelo debe resolver las tres cosas, y esa heterogeneidad es justamente lo que después hizo tan útil su representación intermedia para transferencia.

### Por qué es multi-etiqueta

Una canción puede llevar simultáneamente `rock`, `guitar`, `male vocal` y `90s`. Dado un vocabulario de $K$ etiquetas, la salida es $\hat{y} \in [0,1]^K$ con una **sigmoide independiente por etiqueta** y entropía cruzada binaria, no un *softmax* sobre $K$ clases. Choi et al. (ISMIR 2016) explicitan la consecuencia: con etiquetas exclusivas el clasificador elige entre $K$ vectores *one-hot*; con etiquetas múltiples el espacio de salida crece hasta $2^K$. El espacio observado es menor pero sigue siendo grande: **7.644 vectores de etiquetas únicos en MagnaTagATune y 12.348 en el Million Song Dataset.**

Corolario práctico: predecir todas las etiquetas **conjuntamente** con una sola red es mejor que un modelo por etiqueta. El *baseline* de *gradient boosted trees* de ISMIR 2018 entrena un GBT por etiqueta y *"predice con alta confianza etiquetas mutuamente excluyentes — por ejemplo, dio puntajes altos a East Coast y West Coast para una canción de rap de la costa este, o a baroque period y classic period para un aria de Bach"*. La red profunda, al compartir representación, codifica esas exclusiones implícitamente.

### ROC-AUC, y por qué PR-AUC es más honesta

Ambas se calculan por etiqueta y se promedian:

$$\text{TPR} = \frac{TP}{TP+FN}, \qquad \text{FPR} = \frac{FP}{FP+TN}, \qquad \text{Precision} = \frac{TP}{TP+FP}$$

La diferencia estructural está en **un solo término: la FPR tiene $TN$ en el denominador y la precisión no.** Esa asimetría es todo el argumento. Con prevalencia $\pi$ sobre $N$ ejemplos, si el modelo produce $F$ falsos positivos para alcanzar cierto *recall*:

- En ROC, $\text{FPR} = F/\big((1-\pi)N\big)$. Con $\pi = 0{,}006$ el denominador es prácticamente $N$: **hacen falta muchísimos falsos positivos para mover la aguja**.
- En PR, $\text{Precision} = TP/(TP+F)$ con $TP \le \pi N$. Los mismos $F$ se comparan contra una cantidad de verdaderos positivos $\pi$ veces menor, y **la precisión se desploma**.

El argumento definitivo es el de la línea base:

$$\text{ROC-AUC}_{\text{azar}} = 0{,}5 \;\text{ siempre}, \qquad \text{PR-AUC}_{\text{azar}} = \pi$$

ROC-AUC tiene un piso fijo que **no depende del desbalance**; PR-AUC tiene un piso que **es** la prevalencia. Por eso 0,88 de ROC-AUC no dice nada sobre cuán difícil era el problema, mientras que 0,289 de PR-AUC sobre etiquetas de prevalencia media cercana al 1% es una mejora de casi 30× sobre el azar. ISMIR 2018 lo justifica citando a **Davis y Goadrich (ICML 2006)**: *"ROC-AUC puede llevar a puntajes demasiado optimistas cuando los datos están desbalanceados; dado que las etiquetas están muy desbalanceadas, también consideramos PR-AUC porque es más indicativa en estos casos"*. Y el dato empírico que lo confirma: su mejor modelo mejora al *baseline* clásico en **+6,93 puntos de PR-AUC** (54,27 → 61,20) contra apenas **+0,89 de ROC-AUC** (91,61 → 92,50). Mirando solo ROC-AUC habrían concluido que el modelo profundo casi no aporta.

El desbalance real, medido por Choi et al.: en MTT la etiqueta más frecuente aparece 4.851 veces y la menos frecuente del top-50, 490 (~10×); en MSD, `rock` aparece 52.944 veces y `happy` 1.257 (~42×). Sobre las 214.284 pistas de MSD, `happy` tiene prevalencia de 0,59%.

{{< concept-alert type="clave" >}}
**Regla operativa para cualquier tarea multi-etiqueta desbalanceada** —incluido el *tagging* de audio, la clasificación de eventos sonoros y buena parte del etiquetado clínico—: reporta PR-AUC. ROC-AUC no está mal, está *saturada*. Sobre MTT, el rango completo de ROC-AUC entre el peor y el mejor modelo de esta línea es de 1,72 puntos; el de PR-AUC, 3,69. **Más del doble de resolución para distinguir modelos.**
{{< /concept-alert >}}

### Los datasets de referencia

**MagnaTagATune (MTT)** — Law et al., ISMIR 2009. Anotaciones obtenidas con **TagATune**, un *game with a purpose* donde dos jugadores escuchaban clips y describían lo que oían. Son **25.856 clips de ~29,1 s, mp3 a 16 kHz, 188 etiquetas** de las que se usan las 50 más frecuentes. Las cifras que circulan (~26k, ~21k, 19k) no se contradicen: ~26k es el dataset completo, ~21k la versión "limpia" que usó SampleCNN, y **19k es la partición de entrenamiento**, que es lo que reporta el abstract de musicnn.

**Million Song Dataset (MSD)** — Bertin-Mahieux et al., ISMIR 2011, con etiquetas de **Last.fm**. ISMIR 2018 aclara el malentendido del nombre: *"aunque el nombre indica que hay 1M de canciones disponibles, los archivos de audio con anotaciones de etiquetas apropiadas (top-50) solo están disponibles para ≈240k previews de 30 s"*. musicnn usa **~200k de entrenamiento**.

Los dos vocabularios de 50 etiquetas están replicados literalmente en el `configuration.py` de la librería, y basta leerlos para ver el problema. MTT contiene simultáneamente `vocal`, `vocals`, `voice`, `singing`, `male vocal`, `male voice`, `female vocal`, `female voice`, más las negaciones `no vocal`, `no vocals`, `no voice`. MSD incluye `beautiful`, `sexy` y `catchy`, que son juicios de valor, no propiedades del audio. El `FAQs.md` del repositorio responde a la pregunta obvia: *"¿Por qué el modelo MTT predice `no vocals` y `no vocal`? Porque el vocabulario está determinado por el dataset y lo usamos tal cual"*.

Para el mapa de los datasets del área —incluidos [GTZAN](/papers/gtzan-tzanetakis-2002) y [MusicNet](/papers/musicnet-thickstun-2017)— y del vocabulario de tareas, ver [Clasificación de audio](/fundamentos/clasificacion-de-audio) y el [dominio Audio](/dominios/audio).

## La tesis del dominio: filtros musicalmente motivados

Esta es la contribución conceptual de toda la línea, y el motivo por el cual musicnn merece un análisis largo pese a que su documento propio tenga dos páginas.

### El espectrograma no es una imagen

Un espectrograma log-mel es una matriz $X \in \mathbb{R}^{M \times N}$ con $M$ bandas de frecuencia y $N$ *frames* temporales. Visualmente es una imagen en escala de grises; estructuralmente no lo es. CBMI 2016 lo dice en la primera línea de su sección de diseño: los investigadores de MIR heredaron los filtros cuadrados de visión, *"sin embargo, nótese que **las dimensiones del filtro en imágenes tienen significado espacial, mientras que las dimensiones de los filtros de espectrogramas de audio corresponden a tiempo y frecuencia**"*.

Tres propiedades que una imagen tiene y un espectrograma no:

**(a) Isotropía.** En una imagen natural las estadísticas locales son aproximadamente invariantes a la rotación —un borde a 30° es tan probable como uno a 120°— y el filtro cuadrado es la forma neutral porque no privilegia ninguna dirección. En un espectrograma **no hay simetría rotacional**: rotar 90° convierte un armónico sostenido en un transiente de banda ancha, un objeto físico completamente distinto. Los ejes tienen unidades diferentes (Hz contra segundos) y no existe transformación que los intercambie de forma sensata.

**(b) Estacionariedad en ambos ejes.** Un gato es un gato esté arriba o abajo. En un espectrograma la traslación **en tiempo** sí es una invarianza deseable (un acorde en el segundo 3 o en el 17 es el mismo acorde), pero la traslación **en frecuencia** es semánticamente pesada: desplazar el patrón hacia arriba en el eje mel es **transponer**. A veces eso es lo que se quiere (invarianza al pitch para reconocer un instrumento) y a veces es exactamente lo que no se quiere (reconocimiento de acordes).

**(c) Localidad.** En visión un objeto ocupa una región compacta. Un sonido armónico **está deslocalizado en frecuencia por construcción**: la energía de una nota de violín aparece en $f_0, 2f_0, 3f_0, \dots$, en bandas separadas por decenas de *bins* mel. Un filtro $3\times3$ ve **como máximo tres bandas mel contiguas**, y por lo tanto es incapaz, en la primera capa, de representar la relación entre un fundamental y su tercer armónico. Sobre la geometría de esa representación, ver [Representación tiempo-frecuencia](/fundamentos/representacion-tiempo-frecuencia) y [MFCC y escala mel](/fundamentos/mfcc-y-escala-mel).

### Los dos ejes, las dos formas

Con espectrograma de $M$ *bins* por $N$ *frames* y filtro $m$ (frecuencia) $\times\, n$ (tiempo):

**Filtros temporales, $1 \times n$ — una banda, muchos frames.** *"Fijando la dimensión de frecuencia $m$ a 1, tales filtros no serán capaces de aprender características de frecuencia pero se especializarán en encontrar dependencias temporales (…). Desde la perspectiva musical, uno espera que aprendan patrones rítmicos/de tempo dentro del bin analizado."* Capturan **la envolvente temporal de energía en una banda**: onsets, periodicidad, tempo. La convolución es *bin-wise*, así que las capas superiores todavía pueden cruzar frecuencias sobre los mapas resultantes.

**Filtros frecuenciales, $m \times 1$ — muchas bandas, un frame.** *"Fijando la dimensión de tiempo $n$ a 1 (…) se especializarán en modelar características de frecuencia: pitch, timbre o configuraciones de ecualización."* Esto es literalmente **la envolvente espectral en un instante**, que es la definición operativa clásica de timbre. EUSIPCO 2017 lo ancla en la percepción —el timbre es el "color" del sonido (Wessel 1979), ligado a la forma de la envolvente espectral y su variación (Peeters et al. 2011)— y en la definición negativa de McAdams: *"un conjunto de atributos auditivos de eventos sonoros **además de** pitch, sonoridad, duración y posición espacial"*.

**Filtros rectangulares $m \times n$.** CBMI 2016 no los prohíbe, los sitúa: sirven para objetos que **sí** son compactos en ambos ejes, como un bombo (sub-banda grave, corto en tiempo). El punto no es que el rectángulo esté mal, es que **hay que elegir la forma sabiendo qué objeto musical se quiere capturar**, no por defecto.

### Por qué el $3\times3$ es una elección arbitraria aquí

El filtro cuadrado pequeño llegó al audio por importación desde visión, donde tiene una justificación buena y específica: *"una convolución $5\times5$ puede reemplazarse por dos $3\times3$ apiladas, resultando en menos parámetros"*. Es el argumento de VGG, sobre el que descansa buena parte de lo que se explica en [Redes convolucionales](/fundamentos/redes-convolucionales). **No se transfiere limpiamente al espectrograma, por dos razones.**

**Primera: ir profundo se paga en memoria sobre entradas largas.** ISMIR 2018 lo documenta como restricción de hardware concreta:

> *"Los frontends basados en una pila profunda de filtros $3\times3$ alcanzaban desempeños equivalentes al frontend [de muchos filtros verticales y horizontales] **cuando los segmentos de entrada eran más cortos que 10 s**. Pero al considerar entradas más largas (que daban mejor desempeño), el precio computacional aumenta (…). Nos abstuvimos de usar una pila profunda de $3\times3$ porque **nuestros 12 GB de VRAM no eran suficientes para ingresar 15 s de audio** cuando se usaba un backend."*

**Segunda: la pila profunda alcanza el campo receptivo pero pierde la interpretación y desperdicia capacidad.** EUSIPCO 2017 inspecciona filtros $12\times8$ entrenados y encuentra dos modos de falla. Uno, **ajustar ruido**: *"filter1 está repitiendo una copia ruidosa de un onset a lo largo del eje de frecuencia, y filter2 una copia ruidosa de tres parciales armónicos a lo largo del eje temporal"* — representaciones mucho más eficientes serían un $1\times3$ y un $12\times1$, de modo que el filtro cuadrado **gasta capacidad replicando el patrón a lo largo del eje que no le importa**. Dos, **contexto insuficiente**: el contexto frecuencial es demasiado pequeño para modelar la envolvente armónica completa, y de ahí que *"estos filtros pueden tener dificultades severas para aprender los patrones tiempo-frecuencia de platillos o cajas en la primera capa"*. Un platillo es de banda ancha con decaimiento largo: su firma es $m \approx M$, $n \ll N$, y un $3\times3$ no puede verla.

### Qué se gana: la evidencia de Ballroom

CBMI 2016 diseñó un experimento honesto para separar las hipótesis. Usa **Ballroom** (698 pistas, 8 géneros de baile) precisamente porque **se sabe que sus clases están determinadas casi por completo por el tempo**: Gouyon et al. reportaron **82,3% con un $k$-NN usando solo las anotaciones de BPM**. Entrada de 40 bandas mel, validación cruzada de 10 folds; notación $(m,n)$ = (frecuencia, tiempo).

| Arquitectura | Entrada $(M,N)$ | Filtro $(m,n)$ | #param | Exactitud (10-fold) | Referencia |
|---|---|---|---|---|---|
| Black-box | (40, 80) | (12, 8) | 3.275.312 | **87,25 ± 3,39 %** | 93,12 % (Marchand, clásico) |
| **Time** | (40, 80) | **(1, 60)** | **7.336** | **81,79 ± 4,72 %** | **82,3 % ($k$-NN sobre BPM)** |
| Time | (40, 250) | (1, 200) | 19.496 | 81,52 ± 3,87 % | 82,3 % |
| Frequency | (40, 80) | (40, 1) | 1.576 | 52,43 ± 5,63 % | 15,9 % (clase más probable) |
| **Frequency** | (40, 80) | **(32, 1)** | 3.368 | **59,59 ± 5,82 %** | 15,9 % |
| **Time-FrequencyInit** | (40, 80) | (1,60)-(32,1) | 196.816 | **87,68 ± 4,44 %** | 93,12 % |

1. **`Time`, con 7.336 parámetros, alcanza 81,79% — prácticamente el 82,3% del $k$-NN sobre BPM anotado a mano.** Un solo filtro $1\times60$ (una banda mel × 60 frames ≈ 1,4 s) aprende, sin ninguna supervisión de tempo, casi exactamente lo que un experto codificaría como BPM, con **446 veces menos parámetros** que la caja negra.
2. **`Frequency` llega a 59,59% contra un azar de 15,9%.** El timbre solo, sin información temporal en el filtro, ya discrimina géneros de baile muchísimo mejor que el azar — algo que el propio paper reconoce como sorpresa.
3. **El barrido de $m$ es la evidencia más limpia de invarianza al pitch.** Con $m = 40 = M$ el filtro cubre todo el espectro y **no puede convolucionar en frecuencia**: 52,43%. Al bajar a $m=32$ sí se desliza verticalmente y sube a 59,59% — **+7,2 puntos con 53% menos parámetros**. Por debajo de 32 vuelve a caer: hay un óptimo entre contexto espectral suficiente y margen para deslizarse.

La honestidad del paper merece mención: *"aunque está claro que los filtros temporales están aprendiendo dependencias temporales relevantes, **no podemos afirmar que sean tempo o ritmo**"*. Y ninguna variante alcanza el 93,12% del método clásico.

## La arquitectura

Hay que separar **tres cosas** que suelen confundirse: la arquitectura descrita en **ISMIR 2018**; la efectivamente **publicada** en `musicnn/models.py`, que difiere en detalles importantes; y la figura de alto nivel del abstract de 2019, que no tiene números. Lo que sigue documenta **el código publicado** —el que corre con `pip install musicnn`— y marca cada diferencia.

### La entrada

STFT sobre señal submuestreada a **16 kHz**, ventana de Hanning de 512 muestras con 50% de solape, **96 bandas mel**, compresión $\log_{10}(10000 \cdot x + 1)$.

| Magnitud | Valor | Derivación |
|---|---|---|
| Frecuencia de muestreo | 16.000 Hz | `SR` |
| Hop de la STFT | 256 muestras | `FFT_HOP` |
| Resolución temporal | **16 ms/frame** | $256/16000$ |
| Largo de ventana | 32 ms | $512/16000$ |
| Ancho de banda (Nyquist) | **8 kHz** | $16000/2$ |
| Bandas mel | **96** | `N_MELS` |
| Parche por defecto | **187 frames ≈ 3,0 s** | $187 \times 0{,}016$ |

La [Clase 39](/clases/clase-39) recomienda 44,1 kHz para música y musicnn descarta todo lo que esté sobre 8 kHz: decisión heredada del *tagging* —la información de género e instrumentación vive mayormente bajo esa frecuencia— pero limitación real donde el brillo importa. El tensor de entrada es `(batch, tiempo, mel)` = `(1, 187, 96)`, lo que fija la convención de los *kernels* como **`[tiempo, frecuencia]`** (los papers de 2016 y 2017 usan la inversa; ver la última sección). Lo primero que ocurre es una `batch_normalization`: **la normalización de entrada es una capa aprendida, no un preproceso fijo**.

### El frontend musicalmente motivado

Cinco bloques **en paralelo** sobre la misma entrada, en **una sola capa convolucional** — no una pila.

**Rama timbral (2 bloques).** Antes de convolucionar se rellena **solo el eje del tiempo**; el comentario del código explica por qué: *"padding only time domain for an efficient 'same' implementation (since we pool throughout all frequency afterwards)"*.

| Bloque | `kernel_size` | Forma (tiempo × mel) | Cobertura frecuencial | Cobertura temporal | #filtros |
|---|---|---|---|---|---|
| `f74` | `[7, int(0.4*96)]` | **7 × 38** | 38/96 ≈ 40% del espectro | 7 × 16 ms = **112 ms** | **204** |
| `f77` | `[7, int(0.7*96)]` | **7 × 67** | 67/96 ≈ 70% del espectro | **112 ms** | **204** |

Kernels **anchos en frecuencia y cortos en tiempo**: capturan la envolvente espectral en ~112 ms. `f74` cubre menos espectro y por lo tanto tiene más margen para deslizarse verticalmente (invarianza al pitch más fuerte); `f77` cubre casi todo y captura envolventes muy extendidas — el caso de platillos y cajas que un $3\times3$ no puede ver. Cada bloque es `conv2d(valid) → BatchNorm → max_pool sobre TODO el eje de frecuencia → squeeze`. Ese *max-pool* total implementa literalmente la invarianza al pitch de EUSIPCO 2017 (*"una representación puede ser invariante al pitch si se le aplica un max-pool que abarque todo el eje vertical del mapa de features"*), y colapsa la salida a `(batch, 187, 204)`: **una serie temporal de 204 descriptores timbrales, uno por frame**.

**Rama temporal (3 bloques).** Kernels `[k, 1]`: **una banda mel, muchos frames**.

| Bloque | `kernel_size` | Cobertura temporal | Escala musical | #filtros |
|---|---|---|---|---|
| `s1` | `[128, 1]` | **2,05 s** | nivel de compás | **51** |
| `s2` | `[64, 1]` | **1,02 s** | nivel de pulso | **51** |
| `s3` | `[32, 1]` | **0,51 s** | nivel de nota/subdivisión | **51** |

`padding="same"` → BatchNorm → max-pool sobre todo el eje de frecuencia → squeeze, con salida `(batch, 187, 51)` por bloque. **Concatenación:** $204 + 204 + 51 + 51 + 51 = \mathbf{561}$ descriptores por frame, de los cuales 408 (73%) son timbrales y 153 (27%) temporales. Las claves `timbral` y `temporal` del API de extracción son exactamente esas dos concatenaciones.

### El midend: convoluciones densas con residuales

Tres capas idénticas en estructura, todas con kernel `[7, ancho_completo_de_canales]` — **7 frames × todos los canales**, una convolución 1D temporal *full-channel* escrita con `conv2d` más transposiciones.

| Capa | Kernel | Filtros (`musicnn`) | Filtros (`musicnn_big`) | Residual |
|---|---|---|---|---|
| `conv1` | `[7, 561]` | 64 | 512 | — (adapta dimensiones) |
| `conv2` | `[7, 64]` / `[7, 512]` | 64 | 512 | `conv2 + bn_conv1` |
| `conv3` | `[7, 64]` / `[7, 512]` | 64 | 512 | `conv3 + res_conv2` |

Las dos residuales las justifica ISMIR 2018: *"aunque agregar más capas residuales no mejoró drásticamente nuestros resultados, observamos que **estabilizó el aprendizaje mientras mejoraba ligeramente el desempeño**"*. Y hay algo que el código hace y el paper no nombra así: la salida del midend no es solo `res_conv3` sino la **concatenación densa** de cuatro tensores (`[frontend, conv1, res_conv2, res_conv3]`), con enlace explícito a DenseNet en el `FAQs.md`:

$$561 + 64\cdot 3 = \mathbf{753} \text{ canales (musicnn)}; \qquad 561 + 512\cdot 3 = \mathbf{2097} \text{ (musicnn\_big)}$$

### El backend: pooling temporal y densas

```python
max_pool  = tf.reduce_max(feature_map, axis=1)          # máximo sobre el tiempo
mean_pool, _ = tf.nn.moments(feature_map, axes=[1])     # media sobre el tiempo
tmp_pool  = tf.concat([max_pool, mean_pool], 2)         # 2 × 753 = 1506
```

**El eje temporal desaparece con dos estadísticos globales.** Es lo que ISMIR 2018 llama *variable-length input back-end*: al agregar sobre el tiempo con un estadístico, la red acepta entradas de cualquier duración. Por eso el extractor permite `input_length` arbitrario para los modelos `musicnn` y **lanza un error explícito para los `vgg`** (`'the VGG models cannot handle different input lengths'`): la VGG termina en un `flatten` de tamaño fijo, musicnn en un *pooling* global. Ese `raise` es la diferencia arquitectónica hecha código.

La distinción media/máximo tampoco es cosmética: **la media captura "cuánto de esto hay en la canción" y el máximo "esto aparece al menos una vez"**. Para un vocabulario que mezcla propiedades globales (`ambient`, `slow`) con eventos puntuales (`harpsichord`, `sitar`), se necesitan ambas. Después: `flatten(1506) → BN → Dropout(0.5) → Dense(200 o 500, ReLU) → BN → Dropout(0.5)` —esta es la capa `penultimate`— y `Dense(50)` sin activación, con la sigmoide aplicada fuera del grafo.

### Las variantes publicadas

| Modelo | Frontend | Filtros midend | Unidades backend | Dataset | Parámetros* |
|---|---|---|---|---|---|
| `MTT_musicnn` | musicalmente motivado | 64 | 200 | MagnaTagATune | **≈0,78 M** |
| `MSD_musicnn` | idéntico | 64 | 200 | Million Song Dataset | **≈0,78 M** |
| `MSD_musicnn_big` | idéntico | **512** | **500** | Million Song Dataset | **≈7,97 M** |
| `MTT_vgg` / `MSD_vgg` | 5 capas de $3\times3$ | 128 por capa | — | MTT / MSD | — |

<small>\* Conteos calculados a partir de `models.py`; **ninguna de las fuentes publica el número de parámetros de estas variantes**.</small>

Dos lecturas. **El frontend musicalmente motivado es barato**: unos 162k parámetros, el 21% del modelo estándar y el 2% del grande — la inteligencia de dominio no cuesta capacidad, cuesta pensar. Y `MSD_musicnn_big` tiene **~10× los parámetros** de `MSD_musicnn` para comprar +0,40 puntos de ROC-AUC y +1,12 de PR-AUC: retornos decrecientes clarísimos, y en la dirección esperada (la ganancia se ve en PR-AUC). Los `vgg` son el **contraste deliberado**, y el `FAQs.md` lo dice sin rodeos: *"Porque son un buen baseline, y porque a la gente le gusta usar modelos de visión por computador para espectrogramas"*. El repositorio empaqueta el experimento controlado de la tesis del dominio.

### Advertencia: el código no es el paper

{{< concept-alert type="advertencia" >}}
**La arquitectura publicada en `musicnn/models.py` difiere de la descrita en ISMIR 2018.** No es un detalle menor: quien compare cifras entre el paper y la librería está comparando modelos distintos.
{{< /concept-alert >}}

| Aspecto | ISMIR 2018 (paper) | `models.py` (código) |
|---|---|---|
| Entrada | log-mel de **15 s** | log-mel de **3 s** por defecto (187 frames) |
| Formas timbrales | **seis**: 7×86, 3×86, 1×86, 7×38, 3×38, 1×38 | **dos**: 7×38 y 7×67 |
| Formas temporales | **cuatro**: 165×1, 128×1, 64×1, 32×1 | **tres**: 128×1, 64×1, 32×1 |
| Rama temporal | opera sobre una **envolvente de energía**, obtenida con **mean-pooling del eje de frecuencia** antes de convolucionar | convoluciona `[k,1]` **directamente sobre el espectrograma**, con **max-pool en frecuencia después** |
| Filtros del midend | 512 | 64 (`musicnn`) / 512 (`musicnn_big`) |
| Unidades del backend | 500 | 200 (`musicnn`) / 500 (`musicnn_big`) |
| Downsampling temporal ×2 | mencionado | **no aparece** en el código publicado |
| Parámetros | 5,9 M | ≈0,78 M / ≈7,97 M |

La diferencia más sustantiva es la de **la rama temporal**: promediar-y-después-convolucionar (paper) contra convolucionar-por-banda-y-después-quedarse-con-el-máximo (código). Son operaciones distintas — la primera busca periodicidad en la envolvente global, la segunda **en cada banda por separado**, quedándose luego con la banda de respuesta máxima; la segunda es estrictamente más expresiva y más cara. La segunda diferencia en importancia cambia cómo se leen los resultados: **`MSD_musicnn_big` es esencialmente el modelo del paper en midend/backend (512/500), y `MSD_musicnn` es una versión reducida ~10×.** Cuando el abstract dice que el grande existe "porque MSD tiene más datos", en realidad el grande es el que corresponde a la configuración básica de ISMIR 2018 y el chico es la novedad.

## El contrapunto de los frontends: onda cruda contra espectrograma informado

Este es **el hallazgo más importante de toda la línea**, y es esencial subrayar que **está en ISMIR 2018, no en el abstract de musicnn**, que ni siquiera menciona modelos de forma de onda.

ISMIR 2018 abre con una taxonomía que vale como vocabulario general: *"el front-end es la parte del modelo que interactúa con la señal de entrada para mapearla a un espacio latente, y el back-end predice la salida dada esa representación"*. Y clasifica los frontends en dos dimensiones ortogonales:

| | **Con conocimiento de dominio** | **Assumption-free** |
|---|---|---|
| **Forma de onda** | Filtros del largo de una ventana STFT (Dieleman & Schrauwen) | **Sample-level: pila profunda de $3\times1$** (SampleCNN) |
| **Espectrograma** | **Filtros verticales para timbre y horizontales para tiempo** (Pons) | Pila profunda de $3\times3$ (Choi et al. = la VGG) |

La propiedad que hace interesante el experimento: los modelos de filtros pequeños *"hacen **suposiciones mínimas sobre las estacionariedades locales de la señal**, de modo que cualquier estructura puede aprenderse combinando jerárquicamente representaciones de contexto pequeño (…) **dada suficiente profundidad y suficientes datos**"*. Ese "dada suficiente profundidad y suficientes datos" es la hipótesis que el paper pone a prueba.

Los contendientes: **SampleCNN** (siete capas de convolución 1D con filtros $3\times1$, nueve para el dataset mayor, sobre **15 s de audio crudo a 16 kHz sin ningún preproceso**) contra el **frontend musicalmente motivado**. **Ambos comparten exactamente el mismo backend**, y los tamaños son casi idénticos: **5,9 M de parámetros el de espectrograma, 5,5 M el de forma de onda**. No es una comparación entre un modelo grande y uno chico.

El experimento clave, sobre el dataset privado de 1,2 M de canciones anotadas por expertos, con tres corridas promediadas y particiones estratificadas filtradas por artista:

| Modelo | Tamaño de entrenamiento | ROC-AUC | PR-AUC | $\sqrt{\text{MSE}}$ ↓ |
|---|---|---|---|---|
| Baseline (features musicales + GBT) | 1,2 M | 91,61 % | 54,27 % | 0,1569 |
| **Waveform** | **1 M** | **92,50 %** | **61,20 %** | **0,1465** |
| Spectrogram | 1 M | 92,17 % | 59,92 % | 0,1473 |
| Waveform | 500k | 91,16 % | 56,42 % | 0,1504 |
| **Spectrogram** | **500k** | **91,61 %** | **58,18 %** | **0,1493** |
| Waveform | 100k | 90,27 % | 52,76 % | 0,1554 |
| Spectrogram | 100k | 90,14 % | 52,67 % | **0,1542** |

Cómo leer esto con precisión:

- **A 1 M el waveform gana en las tres métricas** (+0,33 de ROC-AUC, **+1,28 de PR-AUC**, menor error).
- **A 500k el espectrograma gana en las tres** (+0,45 de ROC-AUC, **+1,76 de PR-AUC**).
- **A 100k es un empate técnico y en direcciones mixtas.** La afirmación de que a poca escala gana el espectrograma **no sale de esa fila**: sale del ajuste de regresión lineal de la Figura 3, que promedia tendencias sobre los tres tamaños. Es una lectura de la **pendiente**, no de un punto.
- **Los modelos profundos entrenados con 100k son peores que el baseline clásico de *feature engineering*** (90,27/52,76 contra 91,61/54,27): *"los modelos de deep learning requieren datasets grandes para superar claramente a métodos fuertes basados en diseño de features — aunque nótese que **los datasets grandes generalmente no están disponibles para la mayoría de las tareas de audio**"*.

La explicación mecanística es la que uno esperaría del argumento de sesgo inductivo: *"los front-ends de forma de onda a nivel de muestra tienen un gran potencial para aprender de datos grandes, ya que su espacio de soluciones no está restringido por ninguna elección arquitectónica severa. Por otro lado, las elecciones que definen el front-end de espectrograma **podrían estar restringiendo el espacio de soluciones**. Si bien estas restricciones no son dañinas cuando los datos son escasos, **una regularización tan fuerte puede limitar la capacidad de aprendizaje cuando hay muchos datos disponibles**"*.

{{< concept-alert type="clave" >}}
**El punto de cruce está entre 500k y 1M de canciones etiquetadas.** Por debajo, el diseño informado por el dominio es una regularización que ayuda. Por encima, es una restricción que estorba. Con los datasets públicos disponibles —MTT tiene ~26k y MSD ~240k— **cualquiera que trabaje con música está siempre del lado donde el conocimiento de dominio gana**. Es una de las pocas mediciones limpias que existen sobre cuántos datos hace falta para que el *end-to-end* supere al prior de dominio en audio, y es reutilizable como criterio de diseño.
{{< /concept-alert >}}

Con datasets públicos la confirmación es consistente: sobre MTT, **el mejor modelo de espectrograma (90,40 / 38,11) supera al mejor de forma de onda (89,05 / 34,92)** por +1,35 y +3,19 puntos, y con menos de la mitad de los parámetros (5 M contra 11,8 M). Sobre MSD, el de espectrograma (88,75) **empata con el mejor estado del arte de la época** (88,78) y supera a todos los de forma de onda.

### Conexión con el debate "onda cruda contra features" de la clase

La [Clase 39](/clases/clase-39) plantea el debate en un slide titulado *"Can We Use Raw Audio Data"*: *"Spectrograms, log-mel, etc. are hand-crafted features. Can we use DL to directly learn features from raw data? Yes, but we need to consider some issues"* — y enumera el costo de muestrear a 15–20 kHz (44,1 kHz para música), la necesidad de filtros enormes o estructuras muy profundas, y la solución de las convoluciones dilatadas al estilo WaveNet. **La clase responde el "cómo". ISMIR 2018 responde el "cuándo", y esa es la pieza que falta.**

| | Espectrograma log-mel | Forma de onda cruda |
|---|---|---|
| Qué es la transformación | Un **prior fijo y no aprendido**: STFT + banco mel perceptual + compresión logarítmica | Nada; la red aprende todo |
| Qué asume | Que la fase es descartable, que la escala mel es la resolución correcta, que el log modela la sonoridad | Solo estacionariedad local a escala de ~3 muestras |
| Costo en datos | Bajo: el prior sustituye datos | Alto: hay que aprender el banco de filtros |
| Régimen donde gana | **< ~500k canciones** | **> ~1M canciones** |
| Costo en cómputo | 62,5 frames/s | 16.000 muestras/s → **256× más pasos temporales** |

Un dato refuerza la legitimidad del prior: cuando Dieleman y Schrauwen entrenaron un modelo *end-to-end* sobre forma de onda, *"los bancos de filtros aprendidos muestran similitudes con la escala mel"*. **El modelo end-to-end, si tiene datos suficientes, reinventa el mel-espectrograma** — la mejor defensa del prior (es correcto) y su mejor crítica (es prescindible con datos). La [profundización de la clase 39](/clases/clase-39/profundizacion) retoma este eje, y el *survey* de [Purwins et al.](/papers/dl-audio-purwins-2019) lo sitúa en el panorama general. Nota de coherencia: musicnn **es un modelo de espectrograma**, y la clase lo cita en su slide de referencias junto a **WaveNet**, el arquetipo del audio crudo: los dos lados del debate están en la misma lista, y musicnn es el que trae la evidencia cuantitativa del cruce.

## Resultados

{{< concept-alert type="advertencia" >}}
Esta es la sección donde más se mezclan las fuentes. **Las cifras de los modelos `musicnn` como tales salen del extended abstract de dos páginas y no tienen detrás una tabla arbitrada.** Todo lo demás —tablas comparativas, baselines históricos, experimento de escala— viene de ISMIR 2018, EUSIPCO 2017, CBMI 2016 o de Choi et al. (ISMIR 2016). Cada tabla lleva marcada su fuente.
{{< /concept-alert >}}

**Las cifras propias de musicnn (abstract 2019):**

| Modelo | Dataset | ROC-AUC | PR-AUC |
|---|---|---|---|
| `MTT_musicnn` | MagnaTagATune | **90,69** | **38,44** |
| `MTT_vgg` | MagnaTagATune | 90,26 | 38,19 |
| `MSD_musicnn` | Million Song Dataset | 88,01 | 28,90 |
| `MSD_musicnn_big` | Million Song Dataset | **88,41** | **30,02** |
| `MSD_vgg` | Million Song Dataset | 87,67 | 28,19 |
| musicnn + capa de atención | MagnaTagATune | **90,77** | **38,61** |
| musicnn + capa de atención | Million Song Dataset | **88,81** | **31,51** |

Lecturas honestas:

1. **En MTT, `musicnn` supera a `vgg` por +0,43 de ROC-AUC y +0,25 de PR-AUC.** Es una ventaja pequeña, y conviene decirlo: **no es una demostración contundente de la tesis del dominio**, ni el abstract la presenta como tal — es una tabla de modelos disponibles, no un experimento controlado. La evidencia fuerte está en ISMIR 2018, EUSIPCO 2017 y CBMI 2016.
2. **En MSD la ventaja es mayor y se ve donde debe verse**: `MSD_musicnn` supera a `MSD_vgg` por +0,71 de PR-AUC, y `MSD_musicnn_big` por +1,83.
3. **La variante con capa de atención, en lugar del pooling temporal, gana en los dos datasets**, y sobre MSD la ganancia es sustancial: **+2,61 puntos de PR-AUC sobre `MSD_musicnn`**, y +1,49 sobre el modelo grande con una fracción del cómputo. Señal clara de que el *pooling* media+máximo era el cuello de botella, y anticipo de lo que vendría (atención → Transformers). El abstract remite a documentación online sin publicar detalles: **la arquitectura de atención no está descrita en ninguna fuente verificable, y ninguno de los cinco modelos que se descargan la incluye.**
4. **La afirmación "state-of-the-art"** hay que leerla en su contexto de 2019 y de documento no arbitrado: frente al mejor espectrograma de ISMIR 2018 (90,40 / 38,11), `MTT_musicnn` mejora en +0,29 / +0,33, una mejora incremental sobre el trabajo previo del propio autor.

**Contexto histórico sobre MagnaTagATune** (ROC-AUC):

| Año | Método | ROC-AUC | #param | Fuente |
|---|---|---|---|---|
| 2014 | Convoluciones 1D (Dieleman & Schrauwen) | 88,2 | — | Choi et al. 2016 |
| 2016 | **FCN-4 (Choi et al., $3\times3$ sobre mel)** | **89,4** | 22 M | Choi et al. 2016 |
| 2017 | Small-rectangular ($3\times3$, ajustado) | 86,5 | **75k** | EUSIPCO 2017 |
| 2017 | Dieleman et al. (filtros altos) | 88,1 | **75k** | EUSIPCO 2017 |
| 2017 | **Timbre CNN de Pons (multi-forma)** | **88,9** | **75k** | EUSIPCO 2017 |
| 2018 | Mejor waveform de Pons | 89,05 (PR 34,92) | 11,8 M | ISMIR 2018 |
| 2018 | **Mejor espectrograma de Pons** | **90,40** (PR 38,11) | 5 M | ISMIR 2018 |
| 2019 | **`MTT_musicnn`** | **90,69** (PR 38,44) | ~0,78 M | abstract musicnn |

Las tres filas de EUSIPCO 2017 son la evidencia más limpia de toda la serie: **el mismo presupuesto de 75k parámetros**, tres formas de filtro, y el orden es exactamente el que predice la teoría del dominio — **rectangulares pequeños 86,5 < filtros altos 88,1 < múltiples formas musicalmente motivadas 88,9**. Y es un *ablation* puro: *"reproducen las mismas condiciones que Dieleman et al. (…) **únicamente modificamos la primera capa**; las otras capas se mantienen intactas"*. El dato de eficiencia asociado: **89,3 con 191k parámetros contra los 22 M de Choi et al. para 89,4**. En el estudio de capacidad de ISMIR 2018, la variante de espectrograma con solo **222k parámetros llega a 90,28 / 37,55**, mientras que **todo el rango de capacidades del modelo de forma de onda (88,3–89,05) queda por debajo de la peor variante de espectrograma**.

**Contexto histórico sobre el Million Song Dataset:**

| Año | Método | ROC-AUC | PR-AUC | Fuente |
|---|---|---|---|---|
| 2016 | FCN-4 / FCN-5 / FCN-6 (Choi et al.) | 80,8 / 84,8 / **85,1** | — | Choi et al. 2016 |
| 2017 | SampleCNN (Lee et al.) | 88,12 | — | ISMIR 2018 |
| 2017 | Multi-level & multi-scale (Lee & Nam) | **88,78** | — | ISMIR 2018 |
| 2018 | Waveform de Pons | 87,41 | 28,53 | ISMIR 2018 |
| 2018 | **Espectrograma de Pons** | **88,75** | **31,24** | ISMIR 2018 |
| 2019 | `MSD_vgg` / `MSD_musicnn` / `MSD_musicnn_big` | 87,67 / 88,01 / 88,41 | 28,19 / 28,90 / 30,02 | abstract musicnn |
| 2019 | musicnn + atención | **88,81** | **31,51** | abstract musicnn |

{{< concept-alert type="advertencia" >}}
**Una discrepancia abierta que conviene decir en voz alta.** `MSD_musicnn_big` (88,41 / 30,02, abstract 2019) es **peor** que el modelo de espectrograma de ISMIR 2018 (88,75 / 31,24) — del **mismo autor**, sobre el **mismo dataset** y con **midend y backend equivalentes** (512 filtros, 500 unidades). **Ninguna de las fuentes explica la diferencia.** La hipótesis más plausible es la duración de entrada (15 s en el paper contra ~3 s por defecto en la librería), pero es una hipótesis. La conclusión práctica es firme: **los modelos empaquetados en la librería no son idénticos a los del paper de escala, y sus números no son intercambiables.**
{{< /concept-alert >}}

La otra observación de la serie: en MSD **el salto grande no vino del frontend** sino de la profundidad (FCN-4 → FCN-5 es **+4,0 puntos**, el mayor de toda la serie) y, más tarde, del agregador temporal (pooling → atención, +2,6 de PR-AUC).

Y la lectura absoluta que importa más que cualquier ranking: **la PR-AUC de los mejores modelos está en torno a 0,30–0,38.** Aun el mejor modelo de la época acertaba menos de dos de cada cinco predicciones positivas, promediadas sobre el rango de *recall*. **El tagging musical en 2019 no era un problema resuelto**, aunque una ROC-AUC de 0,90 sugiriera lo contrario.

## musicnn como herramienta

Se instala con `pip install musicnn` y trae los pesos dentro del paquete. El abstract la organiza en tres usos.

**Uso 1 — etiquetador *out of the box*.**

```python
from musicnn.tagger import top_tags

# Trocea el audio en parches de `input_length` segundos, predice cada parche
# por separado y promedia las probabilidades a lo largo del tiempo.
top_tags('music_file.mp3', model='MTT_musicnn', topN=10)
```

**Uso 2 — extracción del taggram y de representaciones intermedias.** Esta es la funcionalidad que hizo útil a la librería:

```python
from musicnn.extractor import extractor

taggram, tags, features = extractor(file, model='MSD_musicnn',
                                    extract_features=True)

# taggram : matriz T × 50 de probabilidades. Cada FILA es un parche temporal
#           y cada COLUMNA una etiqueta. No es un vector por canción: es una
#           SERIE TEMPORAL de etiquetas — de ahí el nombre.
# tags    : los 50 strings que nombran las columnas.
# features: dict con las representaciones intermedias (ver tabla).

# Parches de 1 s con 0,5 s de solape → taggram de resolución fina.
# Solo con los modelos `musicnn`: los `vgg` lanzan ValueError si
# input_length != 3, porque terminan en un flatten de tamaño fijo.
taggram, tags = extractor('cancion.mp3', model='MSD_musicnn',
                          input_length=1, input_overlap=0.5,
                          extract_features=False)
```

| Clave de `features` | Qué es exactamente | Canales |
|---|---|---|
| `timbral` | `concat([f74, f77])` del frontend | 408 |
| `temporal` | `concat([s1, s2, s3])` del frontend | 153 |
| `cnn1` / `cnn2` / `cnn3` | las tres capas del midend | 64 (o 512) |
| `mean_pool` / `max_pool` | media y máximo temporales de la concatenación densa | 753 (o 2097) |
| `penultimate` | salida de la densa de 200 (o 500) unidades | 200 / 500 |

Que el `taggram` sea una serie temporal y no un vector por canción es lo que permite ver **en qué segundo entra la voz o dónde cambia la instrumentación**; y la flexibilidad de `input_length` es **consecuencia directa del pooling temporal global**.

**Uso 3 — transferencia.** *"Nuestros modelos preentrenados pueden ser afinados, junto con una red de salida que actúa como clasificador, para realizar cualquier otra tarea musical"*. Es el uso que más impacto tuvo y tiene su propia sección abajo.

### Estado de mantenimiento

Esto importa si alguien piensa usar la librería hoy. **El paquete nunca tuvo una segunda versión**: PyPI tiene solo la 0.1.0 de agosto de 2019, declarada `Development Status :: 3 - Alpha`, con dependencias `tensorflow>=1.14`, `librosa>=0.7.0` y **`numpy<1.17,>=1.14.5`**, sobre Python 3.7. Los problemas concretos: (a) ese *pin* de NumPy es de julio de 2019 y **cualquier entorno moderno lo viola**, de modo que `pip` fallará la resolución o degradará NumPy y romperá el resto del entorno; (b) el código está escrito íntegramente contra el API de sesiones de TF 1.x (`tf.compat.v1.layers`, `disable_eager_execution()`), deprecado desde hace años; (c) `BATCH_SIZE = 1` por defecto, y el `FAQs.md` lo reconoce —*"¿Mi modelo es lento, incluso con GPU? ¡Sí!"*— con la incómoda solución de editar el archivo de configuración **del paquete instalado**, porque no es un parámetro del API; (d) la lectura de MP3 depende de `audioread`, también deprecado en `librosa` reciente.

En la práctica, **hoy musicnn requiere un entorno aislado y congelado**, lo que es una barrera real para producción. El camino con menos fricción para usar estos pesos ya no es `pip install musicnn` sino los modelos reempaquetados como grafos congelados por el mismo grupo dentro del ecosistema Essentia. *(Esto último es contexto sobre el ecosistema del MTG, no está en los documentos.)*

## Transferencia y embeddings musicales

El abstract reporta un experimento de transferencia deliberadamente simple: **features + PCA a 128 dimensiones + SVM**, sobre **GTZAN (fault-filtered)** — la variante de [GTZAN](/papers/gtzan-tzanetakis-2002) con particiones corregidas para eliminar las repeticiones y fugas de artista que Sturm documentó en el original.

| Extractor de features | Exactitud GTZAN (fault-filtered) | Preentrenamiento |
|---|---|---|
| **VGGish** | **77,58 %** | AudioSet, ~2 M audios |
| **`MSD_musicnn`** | **77,24 %** | MSD, 200k canciones |
| OpenL3 | 74,65 % | AudioSet |
| `MTT_vgg` | 72,75 % | MTT, 19k canciones |
| `MTT_musicnn` | 71,37 % | MTT, 19k canciones |

1. **El tamaño del corpus de preentrenamiento domina sobre la arquitectura.** `MSD_musicnn` supera a `MTT_musicnn` por **5,87 puntos** con **la misma arquitectura exacta, los mismos parámetros, la misma tarea y el mismo vocabulario**: la única diferencia son 19k contra 200k canciones. Es el resultado más contundente del abstract, y es coherente con el hallazgo de escala de ISMIR 2018.
2. **Ser mejor en la tarea de origen no garantiza transferir mejor.** `MTT_vgg` (72,75) **supera** a `MTT_musicnn` (71,37) pese a que este último es el mejor tagger sobre MTT (90,69 contra 90,26). El abstract no comenta la inversión, pero es un recordatorio útil de que la calidad de un *embedding* y la calidad de la predicción no son la misma cosa.
3. **musicnn compite con VGGish usando 10× menos datos de preentrenamiento**: 77,24 contra 77,58.

{{< concept-alert type="advertencia" >}}
**Advertencia metodológica.** El pipeline comprime a **128 componentes principales** representaciones de dimensionalidades muy distintas (753 de `max_pool` en musicnn, 128 en VGGish, 512 en OpenL3) antes de la SVM. Es agresivo y el abstract no lo justifica. La comparación es honesta porque el pipeline es idéntico para todos, pero **las diferencias de 1–3 puntos no son interpretables** sin barras de error, que el abstract no reporta. La diferencia de 5,87 puntos sí es lo bastante grande para ser creíble.
{{< /concept-alert >}}

Sobre qué capa usar, el `FAQs.md` recomienda el **taggram** y **`max_pool`**: *"el taggram porque ya provee información musical de alto nivel, y `max_pool` porque provee una representación acústica relativamente dispersa"*. Que el taggram funcione bien como *embedding* es notable: 50 probabilidades interpretables por parche compitiendo con representaciones de cientos de dimensiones. Es la versión musical de usar los logits de ImageNet como features.

**El paralelo con VGGish:**

| | [**VGGish**](/papers/vggish-hershey-2017) | **musicnn** |
|---|---|---|
| Dominio | Sonidos generales | Música |
| Preentrenamiento | AudioSet, ~2 M audios | MTT / MSD, 19k / 200k canciones |
| Arquitectura | VGG sobre log-mel ($3\times3$) | Frontend musicalmente motivado + midend residual + pooling |
| Uso canónico | *embedding* congelado → clasificador ligero | idéntico |
| Feature típica | capa de *embedding* de 128 dims | `max_pool` (753) o `taggram` (50) |
| GTZAN + PCA128 + SVM | **77,58 %** | **77,24 %** |

La lógica de por qué esto funcionó tan bien entre 2019 y 2022 es la de ImageNet en visión: **una tarea de clasificación con vocabulario amplio, entrenada sobre un corpus grande, produce representaciones intermedias que transfieren dentro del dominio.** El *tagging* musical es un buen pretexto porque su vocabulario cubre simultáneamente timbre (`guitar`), estructura temporal (`fast`, `beat`), producción (`acoustic`) y afecto (`sad`), obligando a la red a representar todas esas dimensiones. Y el contraste que vale marcar: **AudioSet es 10× más grande que MSD, pero un modelo específico de dominio, entrenado con 10× menos datos, iguala su transferencia sobre una tarea musical.** Especializar el corpus vale tanto como escalarlo cuando la tarea final es del mismo dominio.

En la práctica musicnn se volvió componente estándar del MIR aplicado por tres razones nada glamorosas: instalación de una línea con pesos incluidos, inferencia trivial en CPU (~0,78 M de parámetros) y un taggram **auditable** —cada columna tiene nombre, a diferencia de un vector opaco—. Su uso típico era como extractor congelado más un clasificador ligero sobre tareas con pocas etiquetas: género en catálogos propios, instrumento predominante, *mood* para *playlisting*, similitud de contenido en recomendadores. *(Caracterización de usos: contexto del área, no está en los documentos.)*

## Limitaciones

**Sesgo de los datasets.** Ambos son **música occidental comercial**: MTT viene del catálogo de Magnatune, un sello independiente estadounidense, y MSD son *previews* comerciales etiquetados en Last.fm. **De las 50 etiquetas de MSD, siete son décadas** (`00s`, `80s`, `90s`, `70s`, `60s`) o marcadores de época (`oldies`, `classic rock`): **metadatos de catálogo, no propiedades del audio**. **La única etiqueta no occidental en MTT es `indian`** (más `sitar`, que es el mismo referente), y en MSD no hay ninguna: ni flamenco, ni cumbia, ni gamelan, ni maqam. Y hay etiquetas sin definición operacional: `beautiful`, `sexy`, `catchy`, `weird`. Para un contexto latinoamericano esto es directamente relevante: **`MSD_musicnn` aplicado a un catálogo chileno de cueca, nueva canción o cumbia va a producir la etiqueta del vocabulario que tenga más cerca**, no un "no sé". El `FAQs.md` lo admite: *"el modelo no puede predecir `bass` si esa etiqueta no es parte de su vocabulario"*.

**Vocabulario ruidoso y techo de cristal.** En MTT hay ocho etiquetas para presencia de voz y tres negaciones separadas; un modelo perfecto tendría que decidir entre `vocal` y `vocals`, distinción que no existe. Eso pone un **techo estructural sobre la métrica**, y Choi et al. lo detectaron antes que nadie: *"los desempeños similares de trabajos recientes parecen sugerir que **el desempeño está saturado, es decir, se ha alcanzado un techo de cristal debido al ruido en la anotación**"*, con *"muchos algoritmos mostrando solo diferencias pequeñas en el rango de AUC 0,88–0,89"*. Que musicnn llegue a 90,69 es una mejora real, pero **buena parte del margen restante es probablemente irrecuperable**.

**Etiquetas débiles.** **Todas las etiquetas son a nivel de clip completo** (30 s); no hay anotación temporal. Si un clip lleva `piano` y el piano suena solo en los últimos 5 s, la señal de entrenamiento le dice al modelo que los 30 s son `piano`. El *pooling* media+máximo es un paliativo de aprendizaje multi-instancia —el máximo permite que un evento local dispare la etiqueta— pero es un paliativo: **los taggrams son diagnósticos útiles, no detecciones calibradas temporalmente**, y ninguna fuente evalúa su precisión temporal.

**Resolución.**

| Limitación | Valor | Consecuencia |
|---|---|---|
| Frecuencia de muestreo | 16 kHz | **Todo sobre 8 kHz se descarta**: brillo, aire, detalle de platillos |
| Resolución temporal | 16 ms/frame | Un ataque de percusión de 5 ms se reparte en un frame; insuficiente para *onset detection* fino |
| Ventana STFT | 32 ms | ~31 Hz de resolución: **insuficiente para separar armónicos graves** (entre Do2 = 65,4 Hz y Do#2 = 69,3 Hz hay 3,9 Hz) |
| Bandas mel | 96 | Compresión perceptual; imposible recuperar $f_0$ exacto |
| Parche por defecto | 3 s | **Ningún mecanismo representa estructura de forma musical** (estrofa/estribillo) |

El filtro temporal más largo (`s1`, 2,05 s) es lo más lejos que llega la arquitectura en el eje del tiempo: aproximadamente **un compás a 120 BPM en 4/4**.

**El pooling temporal como cuello de botella.** Media y máximo son estadísticos de primer orden que **descartan todo el orden temporal**: una canción y la misma canción reproducida al revés producen exactamente el mismo `mean_pool` y casi el mismo `max_pool`. La evidencia de que eso cuesta rendimiento está en el propio abstract (+2,61 puntos de PR-AUC al sustituirlo por atención), y ISMIR 2018 ya lo sospechaba: *"emplea una estrategia de global pooling **que podría estar removiendo información potencialmente útil**"*.

## Por qué importa hoy

Cinco cosas desplazaron a musicnn. *(Sección de contexto histórico; las cifras específicas se omiten deliberadamente porque no están verificadas contra los documentos de origen.)*

**Transformers de audio (AST, PaSST y sucesores).** Aplican ViT al espectrograma: se trocea en parches, se linealizan y se procesan con auto-atención. **Qué reemplazó:** el pooling temporal media+máximo. **Por qué ganó:** la auto-atención agrega sobre el tiempo de forma **dependiente del contenido y preservando el orden**, en vez de con un estadístico ciego — es la generalización de la misma capa de atención que el abstract de musicnn ya reportaba como mejora. **Qué costó:** el prior de dominio desaparece (los parches vuelven a ser cuadrados) y hay que compensarlo con preentrenamiento masivo.

**Preentrenamiento autosupervisado.** El *tagging* supervisado necesita etiquetas; los objetivos contrastivos y de enmascaramiento no. **La restricción de escala que ISMIR 2018 identificó como cuello de botella desaparece por el otro lado**: ya no hay que esperar a que alguien publique un millón de canciones etiquetadas.

**CLAP y los modelos audio-texto contrastivos.** Dos torres —audio y texto— entrenadas con InfoNCE para alinear pares (audio, descripción). **Qué reemplazó:** el vocabulario cerrado de 50 etiquetas. **Por qué ganó:** habilita *zero-shot tagging* con cualquier frase en lenguaje natural ("guitarra acústica con voz femenina y ritmo lento") sin reentrenar, y elimina de raíz el problema de `vocal` contra `vocals`. **La analogía es exacta: CLAP es a musicnn lo que CLIP es a un clasificador de ImageNet.**

**MERT.** Preentrenamiento autosupervisado específico para música, con enmascaramiento sobre representaciones tokenizadas y objetivos acústicos y tonales. **Qué reemplazó:** el *tagging* supervisado sobre MSD como tarea de pretexto para producir *embeddings*. **Por qué ganó:** rompe el techo de datos, porque consume cualquier catálogo sin etiquetas.

**Modelos generativos de música (Jukebox, MusicGen, Stable Audio, Suno, Udio).** Estos **no reemplazan** a musicnn: cambian la pregunta, de "¿qué etiquetas tiene este audio?" a "genera audio que satisfaga esta descripción". Pero la conexión técnica es directa: **el tagging y la generación condicionada por texto son la misma alineación audio-texto recorrida en direcciones opuestas.** Un modelo texto-a-música necesita internamente un encoder que entienda qué significa "jazz melancólico con saxofón" en términos acústicos — exactamente el problema que musicnn resolvía en la dirección analítica.

Sobreviven al modelo tres cosas. **(a) El vocabulario "frontend / midend / backend"**, que se volvió lenguaje común en audio junto con la distinción entre backends de entrada fija y variable. **(b) El argumento de la forma del filtro**, que sobrevive intacto a los Transformers: hoy se traduce en cómo dividir el espectrograma en parches, si conviene atención factorizada tiempo/frecuencia y qué tamaño de parche usar — la pregunta cambió de forma, no de fondo. **(c) El mapa del compromiso datos-contra-priors, con números**: el umbral entre 500k y 1M de canciones etiquetadas.

## En la clase 39

### La familia "Music", que la clase trata brevemente

La [Clase 39](/clases/clase-39) divide las aplicaciones de audio en tres familias y declara su alcance sin ambigüedad: *"**General sounds** (Ex. environmental sound classification, audio tagging…), **Speech** (…), **Music** (Ex. song recognition, musical instrument identification…). This class mostly focuses on environmental sounds. Next class we will discuss speech and voice. **We will discuss music just briefly.**"* Y el slide *"Audio Applications: Music"* lista exactamente cuatro tareas. Esto es lo que hay dentro de cada una.

**(1) Song recognition — identificar *qué canción exacta* es esta grabación.** Es el problema de Shazam, y crucialmente **no es una tarea de deep learning y nunca lo fue**: es *audio fingerprinting*. Se extraen puntos característicos del espectrograma (máximos locales, o "constelaciones"), se codifican pares de picos en hashes robustos a ruido y compresión, y el *matching* es una búsqueda exacta en una tabla hash gigante. La razón es estructural: **es identificación exacta, no generalización**. Se quiere reconocer *esta* grabación, no la clase de grabaciones similares; una red que generaliza está haciendo lo contrario de lo que se pide. Relación con musicnn: **ninguna**, y confundirlas es el error conceptual más común de la familia. *(Descripción del fingerprinting: contexto del área.)*

**(2) Song / music-style similarity.** Aquí musicnn **era** la respuesta canónica de la época: `audio → musicnn → max_pool o taggram → vector → similitud coseno o vecinos más cercanos`. Es también la razón original por la que existe el *auto-tagging*; ISMIR 2018 abre con ella: *"un objetivo fundamental de la investigación en informática musical es estructurar automáticamente colecciones grandes de música (…); las estimaciones de etiquetas pueden definir un espacio semántico ventajoso para organizar bibliotecas musicales"*. Vale la advertencia paralela: igual que un *embedding* de nombres captura la ortografía pero no la identidad, el de musicnn captura textura sonora pero no autoría, letra ni estructura.

**(3) Music instrument detection.** Es una tarea de **timbre puro**, y por lo tanto donde la tesis de Pons aplica más directamente: la rama timbral del frontend está diseñada exactamente para esto. EUSIPCO 2017 lo evaluó sobre **IRMAS** (11 instrumentos tonales):

| Modelo | Micro-F1 | Macro-F1 | #param |
|---|---|---|---|
| Bosch et al. (bag-of-frames + SVM) | 0,503 | 0,432 | — |
| Han et al. (CNN profunda de $3\times3$, SOTA de la época) | **0,602** | 0,503 | 1.446k |
| Pons, una sola capa | 0,559 | 0,484 | **62k** |
| Pons, multicapa | 0,589 | **0,516** | 743k |

**Con la mitad de parámetros, la versión multicapa queda a 0,013 de micro-F1 del estado del arte y lo supera en macro-F1** —la métrica que no se deja dominar por los instrumentos frecuentes—; y la de una capa, con **23× menos parámetros que el SOTA**, ya supera al *bag-of-frames* clásico. El mismo paper reporta, en clasificación de fonemas de canto sobre ópera de Pekín y con **presupuesto de parámetros exactamente igualado (222k)**, la diferencia más marcada de toda la línea: **+11,0 y +7,3 puntos de exactitud sobre el modelo de $3\times3$**, con el comentario de que los modelos de filtros pequeños *"no funcionan tan bien como el propuesto **en estos datasets pequeños**"*. Es el patrón de ISMIR 2018 a escala aún menor: **cuanto menos datos, más paga el prior de dominio.**

**(4) Music transcription.** La más difícil de las cuatro y la más lejana de musicnn: requiere resolución de pitch, resolución temporal fina y separación de fuentes en música polifónica — el terreno de [MusicNet](/papers/musicnet-thickstun-2017). **musicnn no sirve para esto y su arquitectura lo hace imposible por construcción**: el mel de 96 bandas destruye la resolución de $f_0$; el max-pool sobre todo el eje de frecuencia **descarta deliberadamente en qué banda ocurrió la activación**, que es exactamente lo que la transcripción necesita; y el pooling temporal global elimina toda referencia temporal. Es un buen ejemplo pedagógico de que **las invarianzas que hacen bueno a un modelo para una tarea lo inutilizan para otra**: la invarianza al pitch es un activo para reconocer un violín y un pasivo absoluto para transcribir lo que el violín toca. Choi et al. hacen la observación complementaria —la CQT domina donde hay que identificar frecuencias fundamentales con precisión, el mel domina en *tagging*—, de modo que **la elección de representación de entrada ya decide qué tareas son posibles**.

| Tarea del slide | Enfoque canónico ~2019 | ¿Sirve musicnn? | Por qué |
|---|---|---|---|
| Song recognition | Audio fingerprinting (sin DL) | **No** | Identificación exacta, no generalización |
| Song / style similarity | *Embedding* + vecinos más cercanos | **Sí, era el estándar** | `max_pool` / `taggram` como vector |
| Instrument detection | CNN sobre espectrograma con filtros timbrales | **Sí, directamente** | Es la rama timbral del frontend |
| Music transcription | CQT + CNN/RNN con salida por nota | **No** | Mel + max-pool en frecuencia destruyen el pitch |

### El "Ejemplo 1" de la clase contra el frontend de musicnn

Este es el contraste que más vale la pena desarrollar, porque es donde la clase hace una elección de diseño y Pons ofrece el argumento de por qué esa elección **no es neutra**. El "Ejemplo 1", transcrito:

> **Input:** *"40D Log-mel feats for overlapped segments of 10-20ms. 5-10ms overlap."*
> **CNN:** *"2 convolutional layers. Each with: i) 256 filter, ii) **9x9 and 4x4 filter sizes**, respectively. Optional max-pooling **in frequency only**. Ex. Non-overlapped windows of size 3."*
> **RNN:** *"2 LSTM layers. Cells in LSTMs with 256D. Need to normalize sequence length in minibatch."*
> **MLP:** *"2 FC layers. Each FC layer has 1.024 hidden units."*

Y la receta general que la clase construye slide a slide: *"CNNs: good properties to learn local features. RNNs: good properties to learn temporal features. MLPs: good properties to classify input data."* Es esencialmente el **CLDNN** de Sainath et al. (2015), que la clase cita: una receta sensata y bien probada. **El punto de Pons no es que esté mal, sino que contiene una hipótesis oculta.**

| Dimensión | "Ejemplo 1" (clase 39) | Frontend de musicnn |
|---|---|---|
| Entrada | 40 bandas log-mel, segmentos de 10–20 ms | 96 bandas log-mel, hop de 16 ms, ventana de 32 ms |
| Primera capa | **1 forma de filtro: $9\times9$**, 256 filtros | **5 formas en paralelo**: 7×38, 7×67, 128×1, 64×1, 32×1 |
| Segunda capa | 1 forma: $4\times4$, 256 filtros | no hay: el frontend tiene **una sola capa** |
| Cobertura frecuencial de la capa 1 | 9/40 = **22,5%** del espectro | 40% y 70% (timbral); ~1% (temporal) |
| Cobertura temporal de la capa 1 | 9 frames ≈ **90–180 ms** | 112 ms (timbral); **0,51 / 1,02 / 2,05 s** (temporal) |
| Pooling en frecuencia | *"in frequency only"*, ventanas de 3 | **sobre TODO el eje de frecuencia** |
| Agregación temporal | **2 capas LSTM de 256D** | **media + máximo global** (sin parámetros, sin recurrencia) |
| Entrada de largo variable | **No** (la LSTM exige normalizar el largo) | **Sí**, por construcción del pooling global |

**El "Ejemplo 1" asume que la información relevante es local y aproximadamente isótropa en el plano tiempo-frecuencia**, y que el contexto largo se construye por composición: la CNN forma features locales y la LSTM las integra en el tiempo. Es una hipótesis limpia y modular, y funciona bien cuando la señal *es* localmente estructurada — para sonidos ambientales, que es donde la clase se enfoca, un ladrido o una sirena tienen firmas relativamente compactas y la hipótesis es razonable.

**musicnn asume que la señal musical tiene dos tipos de estructura, no uno, y que ambos deben capturarse en la primera capa**: el timbre está deslocalizado **en frecuencia** (armónicos separados por decenas de bandas mel) y el ritmo está deslocalizado **en tiempo** (segundos, no milisegundos). Ninguno cabe en un $9\times9$, y el problema no es solo de campo receptivo agregado sino de **cómo se gasta la capacidad**: un filtro cuadrado que quisiera cubrir 2 segundos y 70 bandas necesitaría ser $125 \times 70$, con 8.750 pesos por canal, la mayoría irrelevantes. musicnn cubre los mismos rangos con $7\times67 = 469$ y $128\times1 = 128$ pesos, porque **descarta explícitamente la parte del plano que no aporta a cada concepto**.

{{< concept-alert type="clave" >}}
En una sola frase: **el filtro cuadrado es la hipótesis de que las dos direcciones del plano tiempo-frecuencia son igualmente informativas a la misma escala; los filtros de musicnn son la hipótesis de que no lo son.** Ninguna de las dos es "la ausencia de una hipótesis".
{{< /concept-alert >}}

Los puntos concretos de fricción entre las dos recetas:

**(a) La CNN 2D no es una alternativa a la RNN: es su reemplazo si se la diseña bien.** La clase asigna a la CNN "features locales" y a la RNN "features temporales". CBMI 2016 muestra que **un solo filtro $1\times60$ (1,4 s) alcanza 81,79% sobre Ballroom contra el 82,3% de un $k$-NN sobre BPM anotado a mano, con 7.336 parámetros y sin ninguna recurrencia**. musicnn no tiene ninguna capa recurrente y aun así modela ritmo, con filtros de 32, 64 y 128 frames en la primera capa.

**(b) El pooling en frecuencia de la clase es tímido.** La clase propone *"optional max-pooling in frequency only, non-overlapped windows of size 3"*; musicnn colapsa **todo** el eje frecuencial de una vez. La diferencia no es de grado sino de intención: la clase reduce dimensionalidad, musicnn **impone invarianza al pitch como decisión arquitectónica** — y el barrido de $m$ de CBMI 2016 la respalda con +7,2 puntos por permitir que el filtro se deslice en frecuencia.

**(c) La LSTM no está gratis.** El "Ejemplo 1" exige *"normalize sequence length in minibatch"*; musicnn acepta cualquier duración y lo expone como parámetro. Para música, donde las canciones tienen duraciones arbitrarias, es una ventaja operativa concreta — **aunque el propio abstract muestre el límite de esa elección**: reemplazar el pooling por atención da +2,61 de PR-AUC. La agregación media/máximo es simple y robusta, pero deja rendimiento sobre la mesa.

**(d) La receta de la clase es correcta como *default*; Pons dice que hay uno mejor cuando se conoce el dominio.** Pons nunca afirma que el filtro cuadrado esté mal en general — ISMIR 2018 lo mide y encuentra que sobre entradas cortas (<10 s) *"alcanzaba desempeños equivalentes"*. Lo que dice es más preciso: **con entradas largas el $3\times3$ se vuelve caro en memoria, y con datasets pequeños desperdicia capacidad.** Ambas condiciones se cumplen en música. La conclusión general está enunciada de la forma más útil posible en el cierre de CBMI 2016:

> *"Es importante primero entender los datasets de entrenamiento que usan nuestros algoritmos de deep learning. Haciendo eso, los investigadores deberían poder usar ese conocimiento para diseñar arquitecturas que se ajusten mejor al problema. Esto es especialmente relevante para el MIR ya que **se ha señalado que los algoritmos de machine learning están aprendiendo a 'reproducir el ground truth' en lugar de aprender conceptos musicales** [Sturm 2014]. Abordar las arquitecturas de deep learning de una forma musical puede reducir ese riesgo."*

Esa referencia a los "caballos" de Sturm —sistemas que dan la respuesta correcta por la razón equivocada— es la mejor síntesis del aporte de la línea completa. **No es "usa estos filtros": es "sepa qué está aprendiendo su modelo, y diseñe la arquitectura para restringirlo a aprender lo que quiere".**

## Trazabilidad, erratas y matices

Esta sección existe porque el documento de musicnn tiene dos páginas y todo lo demás está en otro lado.

### Qué afirmación viene de qué documento

| Afirmación | Fuente real | ¿Está en el abstract de musicnn? |
|---|---|---|
| Nombres de los 5 modelos preentrenados | abstract 2019 | **Sí** |
| Los dos vocabularios de 50 etiquetas | abstract 2019 + `configuration.py` | **Sí** |
| Las cifras ROC-AUC / PR-AUC de los modelos y de la variante con atención | abstract 2019 | **Sí** |
| Los cinco porcentajes de GTZAN (77,58 / 77,24 / 74,65 / 72,75 / 71,37) | abstract 2019 | **Sí** |
| Nombres de las features extraíbles (`timbral`, `temporal`, `cnn1`…) | abstract 2019 + `models.py` | **Sí** |
| MTT = 19k y MSD = 200k canciones de entrenamiento | abstract 2019 | **Sí** |
| **Formas de filtro concretas (7×38, 7×67, 128×1, 64×1, 32×1)** | **código** (`models.py`) | **No** |
| **Formas de filtro del paper (7×86, 3×86, 1×86, 7×38, 3×38, 1×38; 165×1…)** | **ISMIR 2018** | **No** |
| **SR = 16 kHz, 96 mel, hop 256, FFT 512, parche de 3 s** | **código** (`configuration.py`, `FAQs.md`) | **No** |
| **Conteos de parámetros (~0,78 M / ~7,97 M)** | **cálculo propio desde `models.py`** | **No** |
| **Waveform contra spectrogram y el cruce entre 500k y 1M** | **ISMIR 2018, Tabla 1 y Figura 3** | **No** |
| **Todas las tablas de MTT y MSD con baselines** | **ISMIR 2018, Tablas 2–4** | **No** |
| **Tabla de Ballroom (`Time` / `Frequency` / `Time-Frequency`)** | **CBMI 2016** | **No** |
| **Teoría de invarianzas; IRMAS; ópera de Pekín** | **EUSIPCO 2017** | **No** |
| **Desbalance (`rock` 52.944 contra `happy` 1.257) y el "techo de cristal" de MTT** | **Choi et al., ISMIR 2016** | **No** |
| **Justificación de PR-AUC (Davis & Goadrich)** | **ISMIR 2018** | **No** |

### Erratas y trampas de lectura

**(a) El abstract de musicnn cita mal el año de su paper base.** Su referencia [5] dice *"End-to-end learning for music audio tagging at scale. In **ISMIR, 2017**"*. El paper es de **ISMIR 2018**, y el propio `FAQs.md` del repositorio da el BibTeX correcto (`booktitle={19th ISMIR Conference (ISMIR2018)}, year={2018}`). El origen probable del error es que el preprint v1 de arXiv es de noviembre de 2017.

**(b) La convención de ejes se invierte entre papers.** Esta es la trampa principal al leer la línea completa y explica varias confusiones que circulan.

| Documento | Convención | Ejemplo | Lectura correcta |
|---|---|---|---|
| **CBMI 2016** | **(frecuencia, tiempo)** | `Time` = $(1, 60)$; `Frequency` = $(32, 1)$ | 1 banda × 60 frames; 32 bandas × 1 frame |
| **EUSIPCO 2017** | **frecuencia × tiempo** | `50×1`, `70×10` | 50 bandas × 1 frame; 70 bandas × 10 frames |
| **ISMIR 2018** | **tiempo × frecuencia** (invertida) | `7×86`, `1×86`, `165×1` | 7 frames × 86 bandas; 1 frame × 86 bandas; 165 frames × 1 banda |
| **Código** | `kernel_size=[tiempo, frecuencia]` | `[7, 38]`, `[128, 1]` | 7 frames × 38 bandas; 128 frames × 1 banda |

**Cómo desambiguar:** identificar cuál número puede ser el total de bandas mel. Si el paper usa 96 bandas y aparece un 86, **ese 86 es el eje de frecuencia, esté escrito donde esté**. Consecuencia práctica: **un "filtro 1×86" de ISMIR 2018 y un "filtro 86×1" de EUSIPCO 2017 son exactamente el mismo objeto** (un frame de ancho, 86 bandas de alto), y ambos son "filtros verticales" en la intuición visual. Si alguien afirma que musicnn usa "filtros 165×1 para timbre", está mezclando convenciones: los 165×1 son **temporales**, 165 frames de largo.

**(c) MTT no tiene un solo tamaño.** ~26k (completo), ~21k (versión filtrada de SampleCNN), 19k (partición de entrenamiento). No es contradicción, pero **hace que los números de distintos papers no siempre sean comparables**: el 90,55 de SampleCNN se computó sobre la versión limpia de 21k, y al reproducirlo sobre la de 26k se obtiene 88,56 — **casi 2 puntos solo por la versión del dataset.**

**(d) MSD: 1M en el nombre, ~240k en la práctica.** Cualquier afirmación del tipo "musicnn se entrenó con un millón de canciones" es falsa: fueron ~200k de entrenamiento.

### Atribuciones incorrectas frecuentes

| Afirmación que circula | Estado | Corrección |
|---|---|---|
| "musicnn propone una arquitectura nueva" | **Falso** | Es una librería con pesos preentrenados; la arquitectura viene de ISMIR 2018, con raíces en CBMI 2016 y EUSIPCO 2017 |
| "El paper de musicnn muestra que la onda cruda gana a escala" | **Falso** | Está en ISMIR 2018; el abstract no menciona modelos de forma de onda |
| "musicnn está publicado en ISMIR 2019" | **Impreciso** | Es la sesión **Late-Breaking/Demo**, y el PDF dice `[Unrefereed]` en cada página |
| "musicnn usa filtros $3\times3$" | **Falso** para los modelos `musicnn` | Los `vgg` **sí** los usan: son los baselines de contraste |
| "musicnn tiene una capa de atención" | **Falso** para los 5 modelos publicados | Se menciona como resultado adicional; **no es ninguno de los modelos que se descargan** |
| "Se entrenó con el Million Song Dataset completo (1M)" | **Falso** | ~200k canciones de entrenamiento |
| "musicnn incluye RNN o LSTM" | **Falso** | No hay ninguna capa recurrente; la agregación temporal es media + máximo |
| "musicnn sirve para transcripción musical" | **Falso** | El max-pool sobre todo el eje frecuencial destruye el pitch **por diseño** |
| "El frontend tiene 6 formas timbrales" | **Falso para el código publicado** | ISMIR 2018 describe 6; `models.py` implementa **2** |
| "Procesa a 44,1 kHz porque es música" | **Falso** | `SR = 16000`; techo de 8 kHz |
| "El repositorio está mantenido" | **Falso** | Una sola versión en PyPI (0.1.0, agosto 2019), estado *Alpha*, `numpy<1.17` |

### Cómo citarlo correctamente

El `FAQs.md` pide citar **dos** trabajos, en un orden que dice bastante:

```bibtex
@inproceedings{pons2018atscale,
  title={End-to-end learning for music audio tagging at scale},
  author={Pons, Jordi and Nieto, Oriol and Prockup, Matthew and
          Schmidt, Erik M. and Ehmann, Andreas F. and Serra, Xavier},
  booktitle={19th International Society for Music Information
             Retrieval Conference (ISMIR2018)},
  year={2018},
}

@inproceedings{pons2019musicnn,
  title={musicnn: pre-trained convolutional neural networks for
         music audio tagging},
  author={Pons, Jordi and Serra, Xavier},
  booktitle={Late-breaking/demo session in 20th International Society
             for Music Information Retrieval Conference (LBD-ISMIR2019)},
  year={2019},
}
```

**ISMIR 2018 primero, el abstract de 2019 después.** Si se cita "musicnn, Pons y Serra 2019" como referencia de arquitectura, la referencia técnicamente correcta es **Pons et al., ISMIR 2018**; el documento de 2019 es el anuncio de la librería.

### Lo que no está verificado

Por transparencia: la arquitectura de la variante con atención; los conteos de parámetros de los modelos `vgg`; los resultados de trabajos posteriores sobre MTT (short-chunk CNN, AST, MERT, CLAP), mencionados solo cualitativamente; la descripción del *audio fingerprinting*; la disponibilidad de los pesos vía Essentia; y los hiperparámetros exactos con que se entrenaron los modelos publicados — ISMIR 2018 da los suyos (Adam, tasa inicial 0,001, dropout 0,5, parches de 15 s), pero no hay garantía de que el repositorio de entrenamiento haya usado los mismos.

## Notas y enlaces

- **Documento principal:** Jordi Pons, Xavier Serra, *"musicnn: Pre-trained convolutional neural networks for music audio tagging"*, Late-Breaking/Demo, ISMIR 2019 — arXiv:1909.06654. Dos páginas, no arbitrado, CC BY 4.0. El nombre se pronuncia "musician".
- **El paper que realmente sostiene la arquitectura y el hallazgo de escala:** Pons, Nieto, Prockup, Schmidt, Ehmann y Serra, *"End-to-end learning for music audio tagging at scale"*, ISMIR 2018 — arXiv:1711.02520.
- **El origen conceptual:** Pons, Lidy y Serra, *"Experimenting with musically motivated convolutional neural networks"*, CBMI 2016 (no está en arXiv; disponible en el sitio del autor).
- **La teoría de invarianzas y timbre:** Pons, Slizovskaia, Gong, Gómez y Serra, *"Timbre analysis of music audio signals with convolutional neural networks"*, EUSIPCO 2017 — arXiv:1703.06697.
- **El baseline `vgg`:** Choi, Fazekas y Sandler, *"Automatic tagging using deep convolutional neural networks"*, ISMIR 2016 — arXiv:1606.00298.
- **Código:** `github.com/jordipons/musicnn` (inferencia) y `musicnn-training` (entrenamiento), licencia ISC.

**En este sitio:** la [Clase 39](/clases/clase-39) y su [profundización](/clases/clase-39/profundizacion) sitúan este trabajo dentro del panorama de modelos de deep learning para audio. Para la representación de entrada, ver [Representación tiempo-frecuencia](/fundamentos/representacion-tiempo-frecuencia) y [MFCC y escala mel](/fundamentos/mfcc-y-escala-mel); para la maquinaria convolucional, [Redes convolucionales](/fundamentos/redes-convolucionales); para la tarea y sus métricas, [Clasificación de audio](/fundamentos/clasificacion-de-audio). El contrapunto de sonidos generales es [VGGish](/papers/vggish-hershey-2017); los datasets vecinos, [GTZAN](/papers/gtzan-tzanetakis-2002) y [MusicNet](/papers/musicnet-thickstun-2017); el panorama general del área, el *survey* de [Purwins et al.](/papers/dl-audio-purwins-2019). La línea temporal completa está en el [dominio Audio](/dominios/audio).
