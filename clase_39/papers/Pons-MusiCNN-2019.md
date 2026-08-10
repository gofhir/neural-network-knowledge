# musicnn: Pre-trained Convolutional Neural Networks for Music Audio Tagging (Pons y Serra, 2019) — Análisis interno

## 1. Metadata y resumen ejecutivo

### 1.1. Qué es exactamente musicnn

**musicnn no es un paper de arquitectura nueva.** Es una **librería de Python con modelos preentrenados**, y el documento que la anuncia es un *extended abstract* de dos páginas de la sesión Late-Breaking/Demo de ISMIR 2019 — explícitamente marcado como **`[Unrefereed]`** en el encabezado de cada página del PDF. El nombre se pronuncia "musician" (así lo dice la primera línea del abstract).

- **Autores:** **Jordi Pons** y **Xavier Serra**, ambos del **Music Technology Group (MTG)**, Universitat Pompeu Fabra, Barcelona.
- **Venue:** Late-Breaking/Demo del 20th International Society for Music Information Retrieval Conference (ISMIR), Delft, Países Bajos, 2019. Preprint **arXiv:1909.06654v1** (14 de septiembre de 2019).
- **Licencia del texto:** CC BY 4.0. **Licencia del código:** ISC (`setup.py`).
- **Repositorios:** `github.com/jordipons/musicnn` (inferencia), `github.com/jordipons/musicnn-training` (entrenamiento), `github.com/jordipons/sklearn-audio-transfer-learning` (experimento de transferencia).

Lo que la librería entrega son **cinco modelos preentrenados** —`MTT_musicnn`, `MSD_musicnn`, `MSD_musicnn_big`, `MTT_vgg`, `MSD_vgg`— más un API de tres funciones para (a) etiquetar música *out of the box*, (b) extraer *features* intermedias, y (c) usar esas *features* como representación para transferencia.

### 1.2. De qué papers deriva

El abstract cita su propio linaje en la primera frase: musicnn contiene *"un conjunto de redes convolucionales musicalmente motivadas preentrenadas **[4, 5]**"*, donde:

- **[4] = Pons, Lidy y Serra, "Experimenting with musically motivated convolutional neural networks", CBMI 2016.** El origen conceptual: filtros con forma musical (temporales $1\times n$, frecuenciales $m\times 1$) contra el filtro cuadrado "caja negra" heredado de visión.
- **[5] = Pons, Nieto, Prockup, Schmidt, Ehmann y Serra, "End-to-end learning for music audio tagging at scale".** El abstract lo cita como *"In ISMIR, 2017"*; **es ISMIR 2018** (arXiv:1711.02520v4, 15 de junio de 2018). El propio archivo `FAQs.md` del repositorio corrige la fecha en su BibTeX: `booktitle={19th ISMIR Conference (ISMIR2018)}`. Este es el paper que sustenta la arquitectura y donde están los resultados serios de escala.
- **[2] = Choi, Fazekas y Sandler, "Automatic tagging using deep convolutional neural networks", ISMIR 2016** (arXiv:1606.00298). No es de Pons: es el origen de los *baselines* `vgg` que la librería incluye para comparación.
- Una cuarta pieza, no citada en el abstract pero indispensable para entender el diseño: **Pons, Slizovskaia, Gong, Gómez y Serra, "Timbre Analysis of Music Audio Signals with Convolutional Neural Networks", EUSIPCO 2017** (arXiv:1703.06697v2), que es donde la teoría de invarianzas (pitch, loudness, duración, posición espacial) queda formalizada.

### 1.3. Estado del input y trazabilidad de las cifras

Este análisis combina cinco documentos. Para que nadie atribuya a musicnn resultados que están en otro lugar, cada tabla y cada cifra llevan la sigla de su fuente:

| Sigla | Documento | Estado |
|---|---|---|
| **LBD19** | Pons y Serra, extended abstract musicnn, ISMIR LBD 2019 (arXiv:1909.06654v1) | PDF verificado. **2 páginas, no arbitrado.** |
| **ISMIR18** | Pons et al., "End-to-end learning for music audio tagging at scale", ISMIR 2018 (arXiv:1711.02520v4) | PDF descargado y verificado |
| **CBMI16** | Pons, Lidy y Serra, "Experimenting with musically motivated CNNs", CBMI 2016 (`jordipons.me/media/CBMI16.pdf`) | PDF descargado y verificado |
| **EUSIPCO17** | Pons et al., "Timbre Analysis of Music Audio Signals with CNNs", EUSIPCO 2017 (arXiv:1703.06697v2) | PDF descargado y verificado |
| **ISMIR16** | Choi, Fazekas y Sandler, "Automatic tagging using deep CNNs", ISMIR 2016 (arXiv:1606.00298v1) | PDF descargado y verificado |
| **CÓDIGO** | `github.com/jordipons/musicnn`, rama `master`: `musicnn/models.py`, `musicnn/configuration.py`, `musicnn/extractor.py`, `musicnn/tagger.py`, `setup.py`, `FAQs.md` | Archivos leídos directamente |
| **CLASE39** | Slides "DL Models for Audio Processing", Gabriel Sepúlveda, PUC | Texto extraído del PDF de la clase |

Cuando una afirmación no la pude verificar contra ninguna de estas fuentes, lo digo explícitamente en el texto. **La regla operativa es simple: el LBD19 de dos páginas aporta los nombres de los modelos, los vocabularios de 50 etiquetas, cinco pares ROC-AUC/PR-AUC y los porcentajes de GTZAN. Todo lo demás —formas de filtro, dimensiones, tablas comparativas, el hallazgo de escala— viene de ISMIR18, CBMI16, EUSIPCO17 o del código.**

### 1.4. Resumen ejecutivo en cuatro frases

1. **La tesis:** en un espectrograma el eje vertical es frecuencia y el horizontal es tiempo, y **no son intercambiables**; por lo tanto la forma del filtro convolucional es una decisión de dominio, no un default heredado de visión.
2. **La arquitectura:** un *frontend* de una sola capa con filtros anchos-en-frecuencia (timbre) y largos-en-tiempo (ritmo), un *midend* de tres capas convolucionales con conexiones residuales y densas, y un *backend* de *pooling* temporal global (media + máximo) que acepta entradas de largo variable.
3. **El hallazgo de escala (ISMIR18, no LBD19):** con datasets públicos el modelo informado por dominio gana; recién con **1M de canciones** el modelo *assumption-free* sobre forma de onda cruda lo supera (92.50 vs 92.17 ROC-AUC, 61.20 vs 59.92 PR-AUC).
4. **El legado:** los *embeddings* de musicnn se volvieron *feature extractor* estándar del MIR entre 2019 y ~2022, hasta que los Transformers de audio, los modelos contrastivos audio-texto (CLAP) y MERT los desplazaron.

---

## 2. Contexto: el music audio tagging como tarea

### 2.1. Qué se predice

El *music audio tagging* (o *auto-tagging*) consiste en **estimar automáticamente los atributos musicales de una canción a partir de su audio**. ISMIR18 enumera el alcance en su introducción: *"moods, language of the lyrics, year of composition, genres, instruments, harmony, or rhythmic traits"*. El dataset privado de 1.2M canciones de Pandora que usan en ese trabajo tiene **139 anotaciones a nivel de pista hechas por expertos humanos**, agrupadas en (cito la Sección 3 de ISMIR18):

| Familia de etiquetas | Ejemplos textuales del paper |
|---|---|
| **Meter** | triple-meter, cut-time, compound-duple, odd |
| **Rhythmic feel** | swing, shuffle, back-beat strength, syncopation, danceability |
| **Harmonic** | major, minor, chromatic |
| **Mood** | angry, sad, joyful |
| **Vocal** | presencia de voz, male, female, vocal grittiness |
| **Instrumentation** | piano, guitar distortion |
| **Sonority** | studio, live, acoustic, synthesized |
| **Basic genre** | jazz, rock, rap, latin, disco |
| **Subgenre** | cool/fusion/hard bop (jazz), light/hard/punk (rock), east coast/old school (rap), cajun/indian (world), baroque/classical period |

El punto que interesa para la clase 39: **el vocabulario no es homogéneo**. `piano` es un evento acústico localizado; `swing` es una propiedad de la organización temporal a escala de compases; `00s` es una etiqueta de producción y contexto cultural que no tiene ninguna firma acústica directa. Un mismo modelo tiene que resolver las tres.

Los datasets públicos usan vocabularios más chicos y más ruidosos. Los dos que releva LBD19 (notas al pie 1 y 2 del abstract, replicadas literalmente en `configuration.py` como `MTT_LABELS` y `MSD_LABELS`):

- **MTT (50 etiquetas):** guitar, classical, slow, techno, strings, drums, electronic, rock, fast, piano, ambient, beat, violin, vocal, synth, female, indian, opera, male, singing, vocals, no vocals, harpsichord, loud, quiet, flute, woman, male vocal, no vocal, pop, soft, sitar, solo, man, classic, choir, voice, new age, dance, male voice, female vocal, beats, harp, cello, no voice, weird, country, metal, female voice, choral.
- **MSD (50 etiquetas):** rock, pop, alternative, indie, electronic, female vocalists, dance, 00s, alternative rock, jazz, beautiful, metal, chillout, male vocalists, classic rock, soul, indie rock, Mellow, electronica, 80s, folk, 90s, chill, instrumental, punk, oldies, blues, hard rock, ambient, acoustic, experimental, female vocalist, guitar, Hip-Hop, 70s, party, country, easy listening, sexy, catchy, funk, electro, heavy metal, Progressive rock, 60s, rnb, indie pop, sad, House, happy.

Basta leer las listas para ver el problema. MTT contiene simultáneamente `vocal`, `vocals`, `voice`, `male vocal`, `male voice`, `singing`, y sus negaciones `no vocal`, `no vocals`, `no voice`. El propio `FAQs.md` responde a la pregunta obvia: *"¿Por qué el modelo MTT predice `no vocals` y `no vocal`? Porque el vocabulario del modelo está determinado por el dataset MagnaTagATune y lo usamos tal cual"*. MSD, por su parte, incluye `beautiful`, `sexy` y `catchy`, que son juicios de valor de los usuarios de Last.fm, no propiedades del audio.

### 2.2. Por qué es multi-etiqueta y qué implica

Una canción puede llevar simultáneamente `rock`, `guitar`, `male vocal` y `90s`. Formalmente, dado un vocabulario de $K$ etiquetas, la salida es $\hat{y} \in [0,1]^K$ con una sigmoide **independiente por etiqueta** y pérdida de entropía cruzada binaria, no un softmax sobre $K$ clases.

ISMIR16 (Sección 3) explicita la consecuencia combinatoria: con $K$ etiquetas exclusivas el clasificador solo debe elegir entre $K$ vectores one-hot; con etiquetas múltiples el espacio de salida crece hasta $2^K$. Con $K=50$ eso es $2^{50}$ configuraciones posibles. En la práctica el espacio observado es mucho menor pero sigue siendo grande: ISMIR16 cuenta **7.644 vectores de etiquetas únicos** en MTT y **12.348** en MSD.

Hay un corolario que ISMIR18 observa cualitativamente y que vale la pena retener: predecir todas las etiquetas **conjuntamente** con una sola red es mejor que predecir cada una con un modelo independiente. Su *baseline* de gradient boosted trees entrena un GBT por etiqueta, y el resultado es que *"predice con alta confianza etiquetas mutuamente excluyentes — por ejemplo, dio puntajes altos a East Coast y West Coast para una canción de rap de la costa este, o a baroque period y classic period para un aria de Bach"*. El modelo profundo, al compartir representación, codifica esas exclusiones implícitamente.

### 2.3. Las métricas: ROC-AUC y por qué PR-AUC es más honesta

**ROC-AUC** integra la curva de *true positive rate* contra *false positive rate*:
$$\text{TPR} = \frac{TP}{TP+FN}, \qquad \text{FPR} = \frac{FP}{FP+TN}$$

**PR-AUC** integra precisión contra *recall*:
$$\text{Precision} = \frac{TP}{TP+FP}, \qquad \text{Recall} = \text{TPR} = \frac{TP}{TP+FN}$$

La diferencia estructural está en un solo término: **la FPR tiene $TN$ en el denominador; la precisión no.**

Considera una etiqueta con prevalencia $\pi$ sobre $N$ ejemplos. Los positivos son $\pi N$ y los negativos $(1-\pi)N$. Supón que el modelo, para alcanzar cierto *recall*, produce $F$ falsos positivos.

- En ROC: $\text{FPR} = \dfrac{F}{(1-\pi)N}$. Si $\pi = 0.006$ (el caso de `happy` en MSD, ver más abajo) el denominador es prácticamente $N$: **hacen falta muchísimos falsos positivos para mover la aguja**.
- En PR: $\text{Precision} = \dfrac{TP}{TP+F}$, y $TP \le \pi N$. Los mismos $F$ falsos positivos se comparan contra una cantidad de verdaderos positivos que es $\pi$ veces menor. **La precisión se desploma.**

El otro punto, más contundente, es el de la línea base. Un clasificador aleatorio tiene:
$$\text{ROC-AUC}_{\text{azar}} = 0.5 \quad \text{siempre}, \qquad \text{PR-AUC}_{\text{azar}} = \pi$$

ROC-AUC tiene un piso fijo en 0.5 que no depende del desbalance; PR-AUC tiene un piso que **es** la prevalencia. Eso significa que 0.88 de ROC-AUC no dice nada sobre cuán difícil era el problema, mientras que 0.289 de PR-AUC sobre etiquetas con prevalencia media de ~1% es una mejora de casi 30× sobre el azar. La segunda cifra es informativa; la primera está comprimida contra su techo.

ISMIR18 lo dice sin adornos en la Sección 5.1: *"ROC-AUC puede llevar a puntajes demasiado optimistas en casos donde los datos están desbalanceados [7]; dado que las etiquetas de clasificación están muy desbalanceadas, también consideramos la métrica PR-AUC porque es más indicativa que ROC-AUC en estos casos [7]"*. La referencia [7] es **Davis y Goadrich, "The relationship between precision-recall and ROC curves", ICML 2006** — el paper que estableció que una curva domina en espacio ROC si y solo si domina en espacio PR, pero que las *áreas* no se comportan igual bajo desbalance.

Y hay un detalle empírico que confirma la utilidad de la métrica: ISMIR18 observa que *"la mayor mejora respecto al baseline se ve en PR-AUC"* (54.27 → 61.20 sobre 1.2M, es decir **+6.93 puntos**, contra +0.89 puntos en ROC-AUC de 91.61 → 92.50). Si solo hubieran mirado ROC-AUC, habrían concluido que el modelo profundo apenas mejora al *feature engineering* clásico.

**Cuán desbalanceados están estos datasets, con números.** ISMIR16 (Secciones 5.2 y 5.3) da el conteo exacto:

| Dataset | Etiqueta más frecuente | Etiqueta menos frecuente (de las top-50) | Razón |
|---|---|---|---|
| MTT | 4.851 ocurrencias | 490 ocurrencias | ~10× |
| MSD | `rock`: 52.944 | `happy`: 1.257 | ~42× |

Sobre las 214.284 pistas de MSD que usa ISMIR16, `happy` tiene prevalencia $1257/214284 \approx 0.59\%$. Una PR-AUC de 0.289 sobre ese régimen es un resultado real; una ROC-AUC de 0.880 es un número casi decorativo.

### 2.4. Los datasets de referencia

**MagnaTagATune (MTT)** — Law, West, Mandel, Bay y Downie, *"Evaluation of algorithms using games: the case of music tagging"*, ISMIR 2009. Las anotaciones provienen de **TagATune**, un juego con propósito (*game with a purpose*) donde dos jugadores escuchaban clips y describían lo que oían. Tamaños según fuente:

| Fuente | Cifra reportada |
|---|---|
| ISMIR16, Sección 5.2 | 25.856 clips de 29,1 s, mp3 a 16 kHz, 188 etiquetas (se usan las top-50) |
| ISMIR18, Sección 3 | *"≈26k music audio clips of 30s"* |
| ISMIR18, Sección 5.2 | La versión "limpia" de SampleCNN, que solo incluye canciones con más de 1 etiqueta y de más de 29,1 s, tiene **≈21k** |
| LBD19 | *"the MTT of **19k** training songs"* |
| EUSIPCO17, Sección III.C | *"25.856 clips de ≈30 segundos muestreados a 16 kHz"* |

**Las tres cifras son compatibles**: ~26k es el dataset completo, ~21k la versión filtrada de SampleCNN, y 19k la **partición de entrenamiento** una vez descontados validación y test. LBD19 dice "training songs" con precisión.

**Million Song Dataset (MSD)** — Bertin-Mahieux, Ellis, Whitman y Lamere, ISMIR 2011. ISMIR18 aclara el malentendido de nombre: *"aunque el nombre indica que hay 1M de canciones disponibles, los archivos de audio con anotaciones de etiquetas apropiadas (top-50) solo están disponibles para **≈240k previews de 30 s**"*. Las etiquetas vienen de **Last.fm**, es decir, de folksonomía de usuarios. ISMIR16 usa 201.680 de entrenamiento + 12.605 de validación + 25.940 de test. LBD19 lo resume como *"the MSD of 200k training songs"*.

**Otros mencionados y descartados por ISMIR18:** Free Music Archive (FMA, ≈106k canciones) y AudioSet (≈2.1M audios). AudioSet queda fuera porque *"la mayor parte de su contenido no es música"* — lo cual es exactamente por qué VGGish, entrenado sobre AudioSet, es el punto de comparación *general-purpose* de la Sección 8.

---

## 3. La tesis del dominio: filtros musicalmente motivados

Este es el aporte conceptual de toda la línea de Pons, y es el motivo por el cual musicnn merece un análisis largo pese a que su documento propio tenga dos páginas.

### 3.1. El espectrograma no es una imagen

Un espectrograma log-mel es una matriz $X \in \mathbb{R}^{M \times N}$ donde $M$ son bandas de frecuencia (en escala mel) y $N$ son *frames* temporales. Visualmente es una imagen en escala de grises. Estructuralmente **no lo es**, y CBMI16 lo dice en la primera línea de su Sección II.B.1:

> *"En la literatura de procesamiento de imágenes, los filtros CNN pequeños y cuadrados (por ejemplo 5x5 o 12x12) son comunes. Como resultado de eso, los investigadores de MIR tienden a usar configuraciones de forma de filtro similares. Sin embargo, nótese que **las dimensiones del filtro en procesamiento de imágenes tienen significado espacial, mientras que las dimensiones de los filtros de espectrogramas de audio corresponden a tiempo y frecuencia**."*

La diferencia se puede enunciar como tres propiedades que una imagen tiene y un espectrograma no:

**(a) Isotropía.** En una imagen natural, las estadísticas locales son aproximadamente invariantes a la rotación: un borde a 30° es tan probable como uno a 120°. Un filtro cuadrado es la forma neutral porque no privilegia ninguna dirección. En un espectrograma **no hay simetría rotacional**: rotar 90° convierte un armónico sostenido en un transiente de banda ancha, que es un objeto físico completamente distinto. Los ejes tienen unidades diferentes (Hz vs segundos) y no hay ninguna transformación que los intercambie de forma sensata.

**(b) Estacionariedad local en ambos ejes.** En una imagen, un gato es un gato esté arriba o abajo. En un espectrograma, la traslación **en tiempo** es efectivamente una invarianza deseable (un acorde de guitarra en el segundo 3 o en el 17 es el mismo acorde), pero la traslación **en frecuencia** es semánticamente pesada: desplazar el patrón hacia arriba en el eje mel es transponer la nota. A veces eso es lo que quieres (invarianza al pitch para reconocer un instrumento), a veces es exactamente lo que no quieres (reconocimiento de acordes). EUSIPCO17 desarrolla justamente esta tensión.

**(c) Localidad de la información relevante.** En visión, un objeto ocupa una región compacta. En un espectrograma, un sonido armónico **está deslocalizado en frecuencia por construcción**: la energía de una nota de violín aparece en $f_0, 2f_0, 3f_0, \dots$, es decir, en bandas separadas por decenas de bins mel. Un filtro $3\times3$ ve **como máximo tres bandas mel contiguas** y por lo tanto es incapaz, en la primera capa, de representar la relación entre un fundamental y su tercer armónico.

### 3.2. Los dos ejes, las dos formas

De ahí sale el argumento central, que CBMI16 formula en tres viñetas (Sección II.B.1). Uso la notación de ese paper: espectrograma de $M$ bins de frecuencia por $N$ frames; filtro de $m \times n$ con $m$ = frecuencia y $n$ = tiempo.

**Filtros temporales, $1 \times n$ (una banda, muchos frames).**

> *"Fijando la dimensión de frecuencia $m$ a 1, tales filtros no serán capaces de aprender características de frecuencia pero se especializarán en encontrar dependencias temporales relevantes para la tarea. (…) Desde la perspectiva musical, uno espera que esos filtros temporales aprendan patrones rítmicos/de tempo relevantes dentro del bin analizado."*

El filtro se aplica banda por banda (la convolución es *bin-wise* porque $m=1$), así que las capas superiores todavía pueden explotar relaciones entre frecuencias sobre los mapas resultantes. Lo que el filtro captura es **la envolvente temporal de energía en una banda**: onsets, periodicidad, tempo.

**Filtros frecuenciales, $m \times 1$ (muchas bandas, un frame).**

> *"Fijando la dimensión de tiempo $n$ a 1, tales filtros no serán capaces de aprender características temporales pero se especializarán en modelar características de frecuencia. (…) Desde la perspectiva musical, uno espera que estos filtros de frecuencia aprendan pitch, timbre o configuraciones de ecualización."*

Esto es literalmente **la envolvente espectral en un instante**, que es la definición operativa clásica de timbre. EUSIPCO17 lo ancla en la literatura de percepción: *"El timbre se considera el 'color' o la 'calidad' de un sonido [Wessel 1979]. Se ha encontrado que está relacionado con la forma de la envolvente espectral y con la variación temporal del contenido espectral [Peeters et al. 2011]"*. Y con la definición negativa de McAdams: el timbre es *"un conjunto de atributos auditivos de eventos sonoros **además de** pitch, sonoridad, duración y posición espacial"*.

**Filtros rectangulares $m \times n$.** CBMI16 no los prohíbe; los sitúa. Sirven para objetos que **son** compactos en ambos ejes: un bombo (`kick`) es sub-banda grave y corto en tiempo, así que $m \ll M$, $n \ll N$ lo modela bien. El punto es que hay que elegir la forma sabiendo qué objeto musical se quiere capturar, no por defecto.

### 3.3. Por qué el $3\times3$ es una elección arbitraria aquí

El $3\times3$ (o el $9\times9$ del "Ejemplo 1" de CLASE39) llegó a la audio a través de una cadena de importaciones desde visión. En visión tiene una justificación específica y buena, que ISMIR16 recita correctamente (Sección 2.2.2): *"una convolución $5\times5$ puede reemplazarse por dos convoluciones $3\times3$ apiladas, resultando en menos parámetros"*. Es el argumento de VGG: apilar filtros chicos aproxima un campo receptivo grande con menos pesos y más no-linealidades.

**El argumento no se transfiere limpiamente al espectrograma por dos razones.**

Primera, **el costo de ir profundo se paga en memoria sobre entradas largas**. ISMIR18 lo documenta como una restricción de hardware concreta (Sección 4):

> *"Observamos que los frontends basados en una pila profunda de filtros $3\times3$ alcanzaban desempeños equivalentes al frontend (i) [muchos filtros verticales y horizontales] **cuando los segmentos de entrada eran más cortos que 10 s**. Pero al considerar entradas más largas (que daban mejor desempeño), el precio computacional de este modelo más profundo aumenta: entradas más largas implica tener mapas de features más grandes en cada capa y por lo tanto más consumo de memoria GPU. Por esa razón nos abstuvimos de usar una pila profunda de filtros $3\times3$ como frontend — porque **nuestros 12 GB de VRAM no eran suficientes para ingresar 15 s de audio** cuando se usaba un backend."*

Y la conclusión operativa: *"usar conocimiento de dominio también da orientación para minimizar el costo computacional del modelo — ya que usando una sola capa con muchos filtros verticales y horizontales uno puede capturar eficientemente el mismo campo receptivo **sin pagar el costo de ir profundo**"*.

Segunda, y más importante, **la pila profunda de filtros chicos alcanza el campo receptivo pero pierde la interpretación**. EUSIPCO17 lo argumenta con dos razones nombradas (Sección II):

> *"(i) el principio hebbiano de la neurociencia: 'las células que se disparan juntas se conectan juntas', y (ii) aprender patrones espectro-temporales completos dentro de un solo filtro permite inspeccionar e interpretar los filtros aprendidos de forma compacta."*

Hay además evidencia empírica de que el $3\times3$ efectivamente **falla en aprender lo que no cabe en su ventana**. EUSIPCO17 analiza filtros $12\times8$ entrenados (Figura 1, derecha) y encuentra dos modos de falla:

- **Riesgo (a), ajustar ruido:** *"filter1 está repitiendo una copia ruidosa de un onset a lo largo del eje de frecuencia, y filter2 está repitiendo una copia ruidosa de tres parciales armónicos a lo largo del eje temporal"*. Representaciones mucho más eficientes de esos mismos conceptos serían filtros $1\times3$ y $12\times1$ respectivamente — el filtro cuadrado gasta capacidad replicando el patrón a lo largo del eje que no le importa.
- **Riesgo (b), contexto insuficiente:** *"el contexto de frecuencia de filter2 es demasiado pequeño para modelar la envolvente espectral armónica completa, y solo puede aprender tres parciales armónicos"*.

Y la observación que remata: *"estos filtros pueden tener dificultades severas para aprender los patrones tiempo-frecuencia de platillos o cajas en la primera capa, ya que un contexto tan extendido no cabe dentro de un filtro pequeño-rectangular"*. Los platillos son de banda ancha con decaimiento fijo; su firma es $m \approx M$, $n \ll N$. Un $3\times3$ no puede verla, punto.

CBMI16 cierra el argumento con un caso concreto que es casi una predicción falsable. Comentando el trabajo de Choi et al. con filtros $12\times12$, que reportaron encontrar *onsets*, notas de bajo y bombos:

> *"Como se discutió previamente, los filtros pequeños y cuadrados pueden ser capaces de modelar instrumentos que aparecen en una sub-banda (bajo y bombo) y también de modelar características temporales (onsets) debido a su largo. Sin embargo, **lo que sería una sorpresa es observar que tal red esté modelando platillos o cajas**, lo cual puede ser definitivamente desafiante para una CNN con esas formas de filtro."*

### 3.4. Qué se gana: la evidencia de CBMI16 sobre Ballroom

CBMI16 diseñó un experimento honesto para separar las dos hipótesis. Usa el **dataset Ballroom** (698 pistas de ~30 s, 8 géneros de baile: cha-cha-cha, jive, quickstep, rumba, samba, tango, viennese-waltz, slow-waltz) precisamente porque **se sabe que sus clases están determinadas casi por completo por el tempo**: Gouyon et al. reportaron **82,3% de exactitud con un $k$-NN ($k=1$) usando solo las anotaciones de BPM**. Si un filtro temporal aprende ritmo, debería acercarse a ese número; si un filtro frecuencial aprende timbre, debería quedar muy por debajo pero muy por encima del azar.

Entrada: mel-espectrogramas de **40 bandas**, ventana Blackman-Harris de 2048 muestras con 50% de solapamiento a 44,1 kHz, fases descartadas, compresión logarítmica $\log(1 + C\cdot x)$ con $C = 10000$. Validación cruzada de 10 folds con particiones 80/10/10, voto de mayoría entre segmentos. Notación de la tabla: $(M,N)$ = (bandas, frames) de la entrada; $(m,n)$ = (frecuencia, tiempo) del filtro.

**Tabla I de CBMI16, íntegra:**

| Arquitectura | Entrada $(M,N)$ | Filtro $(m,n)$ | #param | Max-pool | Exactitud (10-fold) | Baseline |
|---|---|---|---|---|---|---|
| Black-box | (40,80) | (12,8) | 3.275.312 | (4,1) | **87,25 ± 3,39 %** | 93,12 % (Marchand et al.) |
| Black-box | (40,250) | (12,200) | 2.363.440 | (4,1) | 82,80 ± 5,12 % | 93,12 % |
| Time | (40,80) | (1,60) | **7.336** | (40,1) | **81,79 ± 4,72 %** | 82,3 % (Gouyon et al., k-NN sobre BPM) |
| Time | (40,250) | (1,200) | 19.496 | (40,1) | 81,52 ± 3,87 % | 82,3 % |
| Frequency | (40,80) | (30,1) | 3.816 | (1,80) | 59,45 ± 5,02 % | 15,9 % (clase más probable) |
| Frequency | (40,80) | (32,1) | 3.368 | (1,80) | **59,59 ± 5,82 %** | 15,9 % |
| Frequency | (40,80) | (34,1) | 2.920 | (1,80) | 58,17 ± 3,58 % | 15,9 % |
| Frequency | (40,80) | (36,1) | 2.472 | (1,80) | 57,88 ± 5,38 % | 15,9 % |
| Frequency | (40,80) | (38,1) | 2.024 | (1,80) | 57,45 ± 5,93 % | 15,9 % |
| Frequency | (40,80) | (40,1) | 1.576 | (1,80) | 52,43 ± 5,63 % | 15,9 % |
| Time-Frequency | (40,80) | (1,60)-(32,1) | 196.816 | (40,1)-(1,80) | 86,54 ± 4,29 % | 93,12 % |
| Time-FrequencyInit | (40,80) | (1,60)-(32,1) | 196.816 | (40,1)-(1,80) | **87,68 ± 4,44 %** | 93,12 % |

Lo que esta tabla demuestra, en orden de importancia:

1. **La arquitectura `Time`, con 7.336 parámetros, alcanza 81,79% — prácticamente el 82,3% del $k$-NN sobre BPM anotado a mano.** Un solo filtro $1\times60$ (una banda mel × 60 frames ≈ 1,4 s) aprende, sin supervisión de tempo, casi exactamente la información que un experto codificaría como BPM. **446 veces menos parámetros que la Black-box.**
2. **La arquitectura `Frequency` llega a 59,59% contra un azar de 15,9%.** Es decir: el timbre solo —sin ninguna información temporal en el filtro— ya discrimina géneros de baile mucho mejor que el azar. CBMI16 comenta que esto fue una sorpresa: *"las características de frecuencia son más relevantes para predecir las clases de Ballroom de lo esperado"*.
3. **El barrido de $m$ en `Frequency` es la evidencia más limpia de invarianza al pitch.** Con $m = 40 = M$, el filtro cubre todo el espectro y **no puede convolucionar en frecuencia**: 52,43%. Al bajar a $m=32$ el filtro sí se desliza verticalmente y sube a 59,59%, **+7,2 puntos con 53% menos parámetros**. CBMI16: *"Diseñar los filtros de modo que puedan convolucionar en frecuencia ($m < M$) ayuda a predecir las clases de Ballroom. Esto probablemente evita que los filtros aprendan pitches individuales, centrando su capacidad en modelar timbre"*. Pero por debajo de $m=32$ el desempeño vuelve a caer, así que hay un óptimo: suficiente contexto espectral para la envolvente, con margen para deslizarse.
4. **La combinación gana y es interpretable.** `Time-FrequencyInit` (87,68%) supera marginalmente a la `Black-box` (87,25%) con **17 veces menos parámetros**, e inicializar cada rama con su mejor modelo previo da +1,14 puntos sobre la inicialización aleatoria.
5. **Filtros más largos no son mejores.** `Time` con $n=200$ (≈4,6 s, hasta 4 tiempos del tempo más lento) da 81,52%, ligeramente peor que $n=60$. CBMI16 propone dos explicaciones: predecir la clase de Ballroom no es literalmente predecir el tempo, y con espectrogramas más largos se muestrean menos segmentos por pista, o sea **menos ejemplos de entrenamiento**.

La honestidad del paper merece una mención: *"aunque está claro que los filtros temporales están aprendiendo dependencias temporales relevantes, **no podemos afirmar que sean tempo o ritmo**. Se necesita más investigación"*. Y también reconoce que ninguna variante alcanza el 93,12% del método clásico de Marchand et al.

---

## 4. La arquitectura de musicnn en detalle

Aquí hay que separar con cuidado **tres cosas distintas** que suelen confundirse:

- **(A)** la arquitectura del *spectrogram front-end* descrita en el paper **ISMIR18**;
- **(B)** la arquitectura efectivamente **publicada** en `musicnn/models.py`, que difiere de (A) en detalles importantes;
- **(C)** la figura del LBD19, que es un diagrama de alto nivel sin números.

Lo que sigue documenta (B) —la que corre cuando escribes `pip install musicnn`— y marca cada diferencia con (A).

### 4.1. La entrada: log-mel

De `musicnn/configuration.py` (CÓDIGO), verbatim:

```python
SR       = 16000   # frecuencia de muestreo
FFT_HOP  = 256     # hop de la STFT, en muestras
FFT_SIZE = 512     # tamaño de la FFT, en muestras
N_MELS   = 96      # bandas mel
BATCH_SIZE = 1     # batch de predicción (por defecto)
```

Y `FAQs.md` completa la descripción: *"Calculamos la STFT de una señal submuestreada a 16 kHz, con una ventana de Hanning de largo 512 (50% de solapamiento). Usamos 96 bandas mel, y le aplicamos una compresión logarítmica (`np.log10(10000·x + 1)`)"*.

De ahí salen las magnitudes derivadas:

| Magnitud | Valor | Derivación |
|---|---|---|
| Resolución temporal | **16 ms/frame** | $256/16000 = 0{,}016$ s |
| Frames por segundo | **62,5** | $16000/256$ |
| Largo de ventana | **32 ms** | $512/16000$ |
| Ancho de banda (Nyquist) | **8 kHz** | $16000/2$ |
| Frames del parche por defecto | **187** | `librosa.time_to_frames(3, sr=16000, n_fft=512, hop_length=256) + 1` |
| Duración del parche por defecto | **≈3,0 s** | $187 \times 0{,}016$ |

Nota sobre el muestreo a **16 kHz**: la clase 39 dice que para música conviene 44,1 kHz. musicnn descarta todo por encima de 8 kHz. Es una decisión heredada del *tagging*: la información discriminativa de género e instrumentación vive mayormente por debajo de 8 kHz, y ISMIR16 aporta evidencia colateral en la misma dirección — un experimento piloto encontró desempeño similar a 12 y 16 kHz, y el mel-espectrograma superó a la STFT precisamente porque *"la alta resolución en el rango de baja frecuencia ayuda al tagging automático"*. Para tareas donde el brillo importa (calidad de producción, distinción de platillos), 8 kHz de techo es una limitación real.

**La entrada del tensor en `models.py` es `x` con forma `(batch, time, mel)`, es decir `(1, 187, 96)`, y `frontend()` hace `tf.expand_dims(x, 3)` para obtener `(1, 187, 96, 1)`.** Esto fija la convención de `kernel_size`: **`[tiempo, frecuencia]`**. Es crítico tenerlo claro porque CBMI16 y EUSIPCO17 usan la convención inversa (ver Sección 12).

Lo primero que ocurre es `tf.compat.v1.layers.batch_normalization` sobre la entrada expandida: **la normalización de entrada es una capa aprendida, no un preproceso fijo**.

### 4.2. El frontend musicalmente motivado

`build_musicnn()` llama a `frontend(x, is_training, config.N_MELS, num_filt=1.6, type='7774timbraltemporal')`. El string `'7774timbraltemporal'` selecciona **cinco bloques paralelos**: dos "timbrales" (los sufijos `77` y `74`) y tres "temporales".

**Rama timbral (2 bloques).** Antes de convolucionar, el tensor se rellena **solo en tiempo** con `tf.pad(..., [[0,0],[3,3],[0,0],[0,0]])`, y el comentario del código explica por qué: *"padding only time domain for an efficient 'same' implementation (since we pool throughout all frequency afterwards)"*. Los kernels:

| Bloque | `kernel_size` en código | Forma efectiva (tiempo × mel) | Cobertura en frecuencia | Cobertura en tiempo | #filtros |
|---|---|---|---|---|---|
| `f74` | `[7, int(0.4*96)]` | **7 × 38** | 38/96 ≈ 40% del espectro | 7 × 16 ms = **112 ms** | `int(1.6*128)` = **204** |
| `f77` | `[7, int(0.7*96)]` | **7 × 67** | 67/96 ≈ 70% del espectro | 112 ms | **204** |

Los kernels son **anchos en frecuencia y cortos en tiempo**: capturan la envolvente espectral en una ventana de ~112 ms. `f74` cubre menos del espectro y por lo tanto tiene más margen para deslizarse verticalmente (invarianza al pitch más fuerte); `f77` cubre casi todo y captura envolventes espectrales muy extendidas —el caso de platillos y cajas que un $3\times3$ no puede ver.

Cada bloque es `timbral_block`: `conv2d(padding="valid", activation=relu)` → `batch_normalization` → **`max_pooling2d` con `pool_size=[1, ancho_completo_del_eje_de_frecuencia]`** → `squeeze`. Ese *max-pool* sobre **todo** el eje de frecuencia restante es la implementación literal de la invarianza al pitch de EUSIPCO17: *"una representación más profunda puede ser invariante al pitch si se le aplica una capa de max-pool que abarque todo el eje vertical del mapa de features: MP(M', ·)"*. El resultado colapsa a `(batch, 187, 204)`: **una serie temporal de 204 descriptores timbrales, uno por frame**.

**Rama temporal (3 bloques).** Aquí los kernels son `[k, 1]`: **una sola banda mel, muchos frames**.

| Bloque | `kernel_size` | Forma (tiempo × mel) | Cobertura temporal | #filtros |
|---|---|---|---|---|
| `s1` | `[128, 1]` | **128 × 1** | 128 × 16 ms = **2,05 s** | `int(1.6*32)` = **51** |
| `s2` | `[64, 1]` | **64 × 1** | **1,02 s** | **51** |
| `s3` | `[32, 1]` | **32 × 1** | **0,51 s** | **51** |

Estas son las tres escalas de tiempo musicales: medio segundo (nivel de nota/subdivisión), un segundo (nivel de pulso), dos segundos (nivel de compás). Con un parche de 3 s, `s1` cubre dos tercios del parche completo — el equivalente a "varios tiempos" a tempos moderados.

`tempo_block` usa `padding="same"` (para no perder frames) → BN → `max_pooling2d` sobre todo el eje de frecuencia → `squeeze`. Salida: `(batch, 187, 51)` por bloque.

**Concatenación.** `tf.concat(frontend_features_list, 2)` apila los cinco bloques a lo largo del eje de canales:

$$204 + 204 + 51 + 51 + 51 = \mathbf{561}$$

El frontend entrega `(batch, 187, 561)`: **561 descriptores por frame, de los cuales 408 (73%) son timbrales y 153 (27%) temporales**. Es una única capa convolucional, no una pila.

Los nombres que expone el API para extracción de features salen exactamente de aquí:
- `timbral` = `concat([f74, f77])` → 408 canales;
- `temporal` = `concat([s1, s2, s3])` → 153 canales.

### 4.3. El midend: convoluciones densas con residuales

`midend()` toma `(batch, 187, 561)`, hace `expand_dims` para volver a 4D y aplica **tres capas convolucionales idénticas en estructura**, todas con kernel `[7, ancho_completo_de_canales]`, es decir **7 frames × todos los canales**. Cada capa mira 112 ms de contexto sobre la representación completa.

El detalle de implementación: después de cada conv se hace `tf.transpose(..., [0,1,3,2])` para intercambiar el eje de canales con el eje "espacial" restante, de modo que la siguiente conv vuelva a ver todos los canales. Es una forma de escribir una convolución 1D temporal full-channel usando `conv2d`.

| Capa | Kernel | Filtros (`musicnn`) | Filtros (`musicnn_big`) | Residual |
|---|---|---|---|---|
| `conv1` | `[7, 561]` | 64 | 512 | — (adapta dimensiones) |
| `conv2` | `[7, 64]` / `[7, 512]` | 64 | 512 | `res_conv2 = conv2 + bn_conv1_t` |
| `conv3` | `[7, 64]` / `[7, 512]` | 64 | 512 | `res_conv3 = conv3 + res_conv2` |

Las **dos conexiones residuales** son las que ISMIR18 justifica en la Sección 4: *"introdujimos conexiones residuales en nuestro modelo para explorar arquitecturas muy profundas (…). Aunque agregar más capas residuales no mejoró drásticamente nuestros resultados, observamos que **agregar estas conexiones residuales estabilizó el aprendizaje mientras mejoraba ligeramente el desempeño**"*.

Y hay algo que el código hace y el paper no describe con este nombre: la salida del midend no es solo `res_conv3`, sino la **concatenación densa** de cuatro tensores:

```python
return [front_end_output, bn_conv1_t, res_conv2, res_conv3]
```

`FAQs.md` lo confirma llamándolo *"some dense layers in the mid-end"* con enlace a **arXiv:1608.06993 (DenseNet)**. Es decir, el backend recibe simultáneamente las features crudas del frontend y las de las tres capas del midend:

$$\text{midend}_\text{out} = 561 + 64 + 64 + 64 = \mathbf{753} \text{ canales (musicnn)}$$
$$\text{midend}_\text{out} = 561 + 512 + 512 + 512 = \mathbf{2097} \text{ canales (musicnn\_big)}$$

Los nombres `cnn1`, `cnn2`, `cnn3` del API de extracción son exactamente `bn_conv1_t`, `res_conv2` y `res_conv3`.

### 4.4. El backend: pooling temporal y capas densas

```python
max_pool  = tf.reduce_max(feature_map, axis=1)          # máximo sobre el tiempo
mean_pool, var_pool = tf.nn.moments(feature_map, axes=[1])  # media sobre el tiempo
tmp_pool  = tf.concat([max_pool, mean_pool], 2)          # 2 × 753 = 1506
```

**El eje temporal desaparece con dos estadísticos globales, media y máximo.** Esta es la decisión que ISMIR18 llama *"variable-length input back-end"* (Sección 2): al agregar sobre el tiempo con un estadístico, la red acepta entradas de cualquier duración. Por eso `extractor.py` permite `input_length` arbitrario para los modelos `musicnn`, y **lanza un error explícito para los `vgg`**:

```python
if 'vgg' in model and input_length != 3:
    raise ValueError('Set input_length=3, the VGG models cannot handle different input lengths.')
```

Ese `raise` es la diferencia arquitectónica entre las dos familias hecha código: la VGG termina en un `flatten` de tamaño fijo; musicnn termina en un *pooling* global.

La distinción media/máximo tampoco es cosmética: **la media captura "cuánto de esto hay en la canción" y el máximo "esto aparece al menos una vez"**. Para un vocabulario que mezcla propiedades globales (`ambient`, `slow`) con eventos puntuales (`harpsichord`, `sitar`), necesitas ambas. ISMIR16 hace la observación complementaria al justificar por qué solo usa max-pool: *"algunas features resultan ser locales, por ejemplo la voz puede estar activa solo durante los últimos segundos de un clip"*.

Después:

```
flatten (1506)  →  BatchNorm  →  Dropout 0.5
   →  Dense(200 o 500, ReLU)  →  BatchNorm  →  Dropout 0.5   ← esto es `penultimate`
   →  Dense(50, sin activación)                              ← logits, sigmoide fuera
```

`models.py` devuelve `bn_dense` como `penultimate`; en `extractor.py` los logits pasan por sigmoide para producir el **taggram**.

### 4.5. Las variantes publicadas

De `define_model()` en `models.py`, textual:

| Modelo | Función | `num_filt_frontend` | `num_filt_midend` | `num_units_backend` | Dataset |
|---|---|---|---|---|---|
| `MTT_musicnn` | `build_musicnn` | 1.6 | **64** | **200** | MagnaTagATune |
| `MSD_musicnn` | `build_musicnn` | 1.6 | **64** | **200** | Million Song Dataset |
| `MSD_musicnn_big` | `build_musicnn` | 1.6 | **512** | **500** | Million Song Dataset |
| `MTT_vgg` | `vgg` | — | 128 filtros por capa | — | MagnaTagATune |
| `MSD_vgg` | `vgg` | — | 128 filtros por capa | — | Million Song Dataset |

**Punto clave que no está en el abstract:** `MTT_musicnn` y `MSD_musicnn` son **la misma arquitectura**, entrenada con datasets distintos. Y `MSD_musicnn_big` **comparte exactamente el mismo frontend** (`num_filt=1.6` está hardcodeado dentro de `build_musicnn`, el parámetro `num_filt_frontend` no se usa); lo único que crece es el midend (64→512) y el backend (200→500). LBD19 justifica el modelo grande en una línea: *"dado que el Million Song Dataset contiene más datos de entrenamiento, también proveemos un modelo musicnn más grande"*.

**Los `vgg` son el contraste deliberado.** `vgg()` es 5 capas de `conv2d` con kernel `[3,3]`, 128 filtros cada una, `padding='same'`, BN, max-pool y dropout 0.25 tras cada bloque, terminando en `flatten` + dropout 0.5 + `dense(50)`. `FAQs.md` explica por qué están ahí: *"Porque son un buen baseline, y porque a la gente le gusta usar modelos de visión por computador para espectrogramas. Así, en este repositorio puedes encontrar modelos basados en musicnn (redes convolucionales musicalmente motivadas) y vggs (una arquitectura de visión por computador aplicada a audio)."* Es decir: **el repositorio empaqueta el experimento controlado de la tesis del dominio.**

### 4.6. Tabla capa por capa y conteo de parámetros

**Advertencia de trazabilidad: los conteos de parámetros de esta subsección los calculé yo a partir de `models.py`. Ni LBD19 ni ISMIR18 publican el número de parámetros de las variantes `MTT_musicnn` / `MSD_musicnn` / `MSD_musicnn_big`.** No incluyen los parámetros de las capas de BatchNorm (que son del orden de $2C$ a $4C$ por capa y no cambian el orden de magnitud). Fórmula usada: $k_t \cdot k_f \cdot C_{in} \cdot C_{out} + C_{out}$.

**`MTT_musicnn` / `MSD_musicnn`**, entrada `(1, 187, 96)`, 50 clases:

| Bloque | Operación | Kernel | Salida | Parámetros |
|---|---|---|---|---|
| Entrada | BatchNorm | — | (187, 96, 1) | ~4 |
| FE timbral `f74` | pad t±3, conv valid + BN + MP(1, todo-freq) | 7 × 38 × 1 → 204 | (187, 204) | 54.468 |
| FE timbral `f77` | pad t±3, conv valid + BN + MP(1, todo-freq) | 7 × 67 × 1 → 204 | (187, 204) | 95.880 |
| FE temporal `s1` | conv same + BN + MP(1, todo-freq) | 128 × 1 × 1 → 51 | (187, 51) | 6.579 |
| FE temporal `s2` | conv same + BN + MP(1, todo-freq) | 64 × 1 × 1 → 51 | (187, 51) | 3.315 |
| FE temporal `s3` | conv same + BN + MP(1, todo-freq) | 32 × 1 × 1 → 51 | (187, 51) | 1.683 |
| **concat FE** | — | — | **(187, 561)** | **≈161.925** |
| ME `conv1` | pad t±3, conv valid + BN | 7 × 561 × 1 → 64 | (187, 64) | 251.392 |
| ME `conv2` | pad t±3, conv valid + BN, **+residual** | 7 × 64 × 1 → 64 | (187, 64) | 28.736 |
| ME `conv3` | pad t±3, conv valid + BN, **+residual** | 7 × 64 × 1 → 64 | (187, 64) | 28.736 |
| **concat densa ME** | `[FE, conv1, conv2, conv3]` | — | **(187, 753)** | **≈308.864** |
| BE pooling | `max` ⊕ `mean` sobre el eje temporal | — | (1506,) | 0 |
| BE denso 1 | BN + Dropout(0.5) + Dense(200) + BN + Dropout(0.5) | — | (200,) | 301.400 |
| BE salida | Dense(50), logits | — | (50,) | 10.050 |
| **Total** | | | | **≈ 0,78 M** |

**`MSD_musicnn_big`**, mismo frontend:

| Bloque | Kernel | Salida | Parámetros |
|---|---|---|---|
| Frontend (idéntico) | — | (187, 561) | ≈161.925 |
| ME `conv1` | 7 × 561 × 1 → 512 | (187, 512) | 2.011.136 |
| ME `conv2` (+res) | 7 × 512 × 1 → 512 | (187, 512) | 1.835.520 |
| ME `conv3` (+res) | 7 × 512 × 1 → 512 | (187, 512) | 1.835.520 |
| concat densa | — | **(187, 2097)** | — |
| BE pooling | — | (4194,) | 0 |
| BE denso 1 | Dense(500) | (500,) | 2.097.500 |
| BE salida | Dense(50) | (50,) | 25.050 |
| **Total** | | | **≈ 7,97 M** |

Dos observaciones que salen de estas tablas:

1. **El frontend musicalmente motivado es barato: ~162k parámetros, el 21% del modelo estándar y el 2% del grande.** La inteligencia de dominio no cuesta capacidad; cuesta pensar.
2. **`MSD_musicnn_big` tiene ~10× los parámetros de `MSD_musicnn` y compra +0,40 puntos de ROC-AUC y +1,12 de PR-AUC** (88.01/28.90 → 88.41/30.02, LBD19). Retornos decrecientes clarísimos, y en la dirección esperada: la ganancia se ve en PR-AUC, no en ROC-AUC.

### 4.7. Diferencias entre el paper de 2018 y el código publicado

Esto no está documentado en ningún lado y es material relevante para cualquiera que compare.

| Aspecto | ISMIR18 (Sección 4, "Spectrogram front-end") | `musicnn/models.py` (CÓDIGO) |
|---|---|---|
| Entrada | log-mel de **15 s**, 96 bandas | log-mel de **3 s** por defecto (187 frames), 96 bandas |
| Formas timbrales | **seis**: 7×86, 3×86, 1×86, 7×38, 3×38, 1×38 | **dos**: 7×38 y 7×67 |
| Filtros timbrales | 16, 32, 64, 16, 32, 64 respectivamente (**224 total**) | 204 + 204 = **408 total** |
| Formas temporales | **cuatro**: 165×1, 128×1, 64×1, 32×1 | **tres**: 128×1, 64×1, 32×1 |
| Filtros temporales | 16, 32, 64, 128 (**240 total**) | 51 × 3 = **153 total** |
| Entrada de la rama temporal | *"operan sobre una **envolvente de energía** (no directamente sobre el espectrograma) obtenida mediante **mean-pooling** del eje de frecuencia"* | conv `[k,1]` **directamente sobre el espectrograma normalizado**, seguido de **max-pool** sobre todo el eje de frecuencia |
| Filtros del midend | **512** por capa | **64** (`musicnn`) / **512** (`musicnn_big`) |
| Unidades del backend | **500** | **200** (`musicnn`) / **500** (`musicnn_big`) |
| Pooling temporal | *"global pooling with mean and max statistics"* | idéntico |
| Downsampling temporal ×2 | mencionado en el paper | **no aparece** en `midend()` del código publicado |
| Parámetros | 5,9 M (configuración básica) | ≈0,78 M / ≈7,97 M (mi cálculo) |

La diferencia más sustantiva es la de la **rama temporal**: mean-pool-antes-de-convolucionar (paper) versus convolucionar-por-banda-y-max-pool-después (código). Son operaciones distintas — la primera promedia y luego busca periodicidad en la envolvente global; la segunda busca periodicidad en cada banda por separado y luego se queda con la banda de respuesta máxima. La segunda es estrictamente más expresiva y más cara.

La segunda diferencia en importancia: **`MSD_musicnn_big` es esencialmente el modelo del paper en midend/backend (512/500), y `MSD_musicnn` es una versión reducida ~10×.** Cuando LBD19 dice que el grande existe "porque MSD tiene más datos", en realidad el grande es el que corresponde a la configuración básica de ISMIR18.

---

## 5. El contrapunto: los frontends que compara la línea de Pons

Este es, a mi juicio, **el hallazgo más importante de toda la línea**, y es esencial subrayar que **está en ISMIR18, no en el abstract de musicnn**.

### 5.1. El marco: frontend vs backend

ISMIR18 abre (Sección 2) con una taxonomía que vale como vocabulario general:

> *"Dividimos los modelos de deep learning en dos partes: **front-end** y **back-end**. El front-end es la parte del modelo que interactúa con la señal de entrada para mapearla a un espacio latente, y el back-end predice la salida dada la representación obtenida por el front-end."*

Los frontends se clasifican en dos dimensiones **ortogonales**:

| | **Conocimiento de dominio** | **Assumption-free** |
|---|---|---|
| **Forma de onda** | Filtros del largo de una ventana STFT (Dieleman & Schrauwen 2014); banco multi-resolución (Zhu et al. 2016) | **Sample-level: pila profunda de filtros $3\times1$** (Lee et al., SampleCNN) |
| **Espectrograma** | **Filtros verticales para timbre y horizontales para tiempo** (la línea de Pons) | Pila profunda de $3\times3$ (Choi et al. 2016 = la VGG) |

Y define la propiedad que hace interesante el experimento:

> *"Cuando no se usa conocimiento de dominio, es común emplear una pila profunda de filtros pequeños. (…) Estos modelos basados en filtros pequeños hacen **suposiciones mínimas sobre las estacionariedades locales de la señal**, de modo que cualquier estructura puede aprenderse combinando jerárquicamente representaciones de contexto pequeño. Estas arquitecturas con filtros pequeños son modelos flexibles capaces de aprender potencialmente cualquier estructura **dada suficiente profundidad y suficientes datos**."*

Ese "dada suficiente profundidad y suficientes datos" es la hipótesis que el paper va a poner a prueba.

### 5.2. Los dos contendientes

**Waveform front-end (sin conocimiento de dominio).** ISMIR18 lo eligió tras observar que el *sample-level* de Lee et al. era *"notablemente superior a los otros frontends basados en forma de onda"*. Estructura:

- **siete capas** de `1D-CNN` con filtros `3×1` + batch norm + max-pool;
- **64, 64, 64, 128, 128, 128, 256** filtros respectivamente;
- para el dataset de 1.2M usan una versión de **nueve capas**: 64, 64, 64, 128, 128, 128, 128, 128, 256;
- entrada: **15 s de audio crudo submuestreado a 16 kHz**, sin ningún preproceso.

Cita textual sobre por qué es el contendiente correcto: *"el sample-level puede verse como un front-end agnóstico al problema que tiene el potencial de aprender cualquier tarea de audio siempre que haya suficiente profundidad y datos. (…) su espacio de soluciones no está restringido por elecciones arquitectónicas severas que dependan de conocimiento de dominio"*.

**Spectrogram front-end (con conocimiento de dominio).** Es el que describí en la Sección 4.2: una sola capa con dos ramas, timbral (filtros anchos en frecuencia, cortos en tiempo, con max-pool sobre todo el eje frecuencial) y temporal (filtros largos en tiempo, un bin de ancho, sobre la envolvente de energía).

**Ambos comparten exactamente el mismo backend** — tres capas convolucionales de 512 filtros con dos residuales, pooling temporal global media+máximo, densa de 500 unidades, salida sigmoidal. ISMIR18: *"los experimentos comparten el mismo back-end, lo que permite una comparación justa entre los front-ends seleccionados"*.

Y los tamaños son casi idénticos: **5,9 M de parámetros el de espectrograma, 5,5 M el de forma de onda** (Sección 5.1). No es una comparación entre un modelo grande y uno chico.

### 5.3. El resultado: Tabla 1 de ISMIR18 (dataset privado de 1.2M)

Este es **el experimento clave** y es el único con suficiente escala para responder la pregunta. Tres tamaños de entrenamiento (100k, 500k, 1M canciones), tres corridas promediadas, mismos conjuntos de validación y test siempre, particiones estratificadas y **filtradas por artista**. Baseline: extractor de features musicales (timbre, ritmo, armonía) + gradient boosted trees, un GBT por etiqueta, entrenado con las 1.2M.

| Modelo | Tamaño de entrenamiento | ROC-AUC | PR-AUC | $\sqrt{\text{MSE}}$ ↓ |
|---|---|---|---|---|
| Baseline (GBT + features) | 1,2M | 91,61 % | 54,27 % | 0,1569 |
| **Waveform** | **1M** | **92,50 %** | **61,20 %** | **0,1465** |
| Spectrogram | 1M | 92,17 % | 59,92 % | 0,1473 |
| Waveform | 500k | 91,16 % | 56,42 % | 0,1504 |
| **Spectrogram** | **500k** | **91,61 %** | **58,18 %** | **0,1493** |
| Waveform | 100k | 90,27 % | 52,76 % | 0,1554 |
| Spectrogram | 100k | 90,14 % | 52,67 % | **0,1542** |

**Cómo leer esto con precisión.**

- **A 1M canciones el waveform gana en las tres métricas.** +0,33 puntos de ROC-AUC, **+1,28 de PR-AUC**, y menor error de regresión. ISMIR18: *"el modelo con mejor desempeño está basado en el front-end de forma de onda, siendo capaz de superar al modelo de espectrograma en **todas** las métricas cuando se entrena con 1M de canciones"*.
- **A 500k el espectrograma gana en las tres.** +0,45 de ROC-AUC, **+1,76 de PR-AUC**, menor error.
- **A 100k el resultado es un empate técnico y en direcciones mixtas**: el waveform va marginalmente arriba en ROC-AUC (90,27 vs 90,14) y PR-AUC (52,76 vs 52,67), pero abajo en $\sqrt{\text{MSE}}$ (0,1554 vs 0,1542). **Ninguna de esas diferencias es interpretable a partir de la tabla sola.** La afirmación de que a poca escala gana el espectrograma no sale de la fila de 100k: sale del **ajuste de regresión lineal de la Figura 3**, que promedia las tendencias sobre los tres tamaños. ISMIR18 es explícito: *"uno puede observar esto en la Figura 3, donde ajustamos modelos lineales a los resultados obtenidos. Cuando hay 100k canciones de entrenamiento disponibles: las líneas de tendencia muestran que los modelos de espectrograma tienden a desempeñarse mejor. Sin embargo, cuando hay 1M de canciones: las líneas muestran que los modelos de forma de onda superan a los de espectrograma"*. Es una lectura de la **pendiente**, no de un punto.
- **Los modelos profundos entrenados con 100k son PEORES que el baseline clásico** (90,27/52,76 vs 91,61/54,27). Este es un resultado que se cita poco y merece citarse: *"este resultado confirma que los modelos de deep learning requieren datasets grandes para superar claramente a métodos fuertes basados en diseño de features — aunque nótese que **los datasets grandes generalmente no están disponibles para la mayoría de las tareas de audio**"*.

La explicación mecanística que da el paper es exactamente la que uno esperaría del argumento de sesgo inductivo:

> *"Este resultado confirma que los front-ends de forma de onda a nivel de muestra tienen un gran potencial para aprender de datos grandes, ya que su espacio de soluciones no está restringido por ninguna elección arquitectónica severa. Por otro lado, las elecciones arquitectónicas que definen el front-end de espectrograma **podrían estar restringiendo el espacio de soluciones**. Si bien estas restricciones no son dañinas cuando los datos de entrenamiento son escasos, **una regularización tan fuerte del espacio de soluciones puede limitar la capacidad de aprendizaje del modelo en escenarios donde hay muchos datos de entrenamiento disponibles**."*

Y el costo computacional, para dimensionar: 100k canciones tomaron *"unos pocos días"*, 500k *"una semana"*, 1M *"menos de dos semanas"*.

### 5.4. La confirmación en datasets públicos

Con datasets públicos —o sea, con menos de 250k canciones— **el espectrograma gana consistentemente**.

**MTT (≈26k, entrada de 3 s).** ISMIR18 nota que en MTT *"entradas más largas dan peores resultados"*, por eso 3 s. Combino las Tablas 2 y 3:

| Frontend | Modelo | ROC-AUC | PR-AUC | #param | Fuente |
|---|---|---|---|---|---|
| Waveform | SampleCNN [Lee et al.] | 90,55 | — | 2,4M | ISMIR18 T2 (versión distinta de MTT) |
| Waveform | SampleCNN (reproducido por Pons) | 88,56 | 34,38 | 2,4M | ISMIR18 T2 |
| Waveform | Dieleman et al. | 84,87 | — | — | ISMIR18 T2 |
| Waveform | Dieleman et al. (reproducido) | 85,58 | 29,59 | 194k | ISMIR18 T2 |
| Waveform | **Mejor de Pons (#filtros ×4)** | **89,05** | 34,92 | 11,8M | ISMIR18 T2 |
| Waveform | Pons, #filtros ×2 (seleccionado) | 88,96 | 34,74 | 7M | ISMIR18 T2 |
| Spectrogram | VGG - Choi et al. | 89,40 | — | 22M | ISMIR18 T3 |
| Spectrogram | VGG (reproducido) | 89,99 | 37,56 | 450k | ISMIR18 T3 |
| Spectrogram | Timbre CNN [EUSIPCO17] | 89,30 | — | 191k | ISMIR18 T3 |
| Spectrogram | Timbre CNN (reproducido, 96 mel) | 89,07 | 34,92 | 220k | ISMIR18 T3 |
| Spectrogram | **Mejor de Pons (#filtros ×1/2)** | **90,40** | **38,11** | 5M | ISMIR18 T3 |

**El mejor modelo de espectrograma (90,40 / 38,11) supera al mejor de forma de onda (89,05 / 34,92) por +1,35 de ROC-AUC y +3,19 de PR-AUC.** Y observa el detalle de eficiencia: el mejor espectrograma usa 5M de parámetros contra 11,8M del mejor waveform.

Dato adicional relevante para el argumento de eficiencia: en el estudio de capacidad del backend sobre MTT, la variante de espectrograma con **64 filtros de CNN y 200 unidades densas — solo 222k parámetros — alcanza 90,28 / 37,55**, apenas 0,12 y 0,56 puntos por debajo del modelo de 5M. ISMIR18 lo resume: *"el desempeño es bastante robusto al número de parámetros del modelo (…) modelos relativamente pequeños (que son más fáciles de desplegar) pueden hacer un trabajo razonable"*. Para la versión waveform, en cambio, el rango completo de capacidades cubre 88,3–89,05 de ROC-AUC: **todas por debajo de la peor variante de espectrograma**.

**MSD (≈240k). Tabla 4 de ISMIR18:**

| Frontend | Modelo | ROC-AUC | PR-AUC | #param |
|---|---|---|---|---|
| Waveform | Pons et al. (este trabajo) | 87,41 | 28,53 | 5,3M |
| Waveform | SampleCNN [Lee et al. 2017] | 88,12 | — | 2,4M |
| Waveform | SampleCNN multi-level & multi-scale [Lee et al. 2018] | 88,42 | — | — |
| Spectrogram | **Pons et al. (este trabajo)** | **88,75** | **31,24** | 5,9M |
| Spectrogram | VGG + RNN [Choi et al. 2017] | 86,2 | — | 3M |
| Spectrogram | Multi-level & multi-scale [Lee & Nam 2017] | 88,78 | — | — |

El modelo de espectrograma (88,75) **empata con el mejor del estado del arte de la época** (88,78) y supera a todos los de forma de onda. ISMIR18 es autocrítico sobre por qué su waveform (87,41) queda por debajo del SampleCNN (88,12): propone dos causas —SampleCNN promedia diez estimaciones por canción y ellos solo dos; y el *global pooling* del backend compartido *"podría estar removiendo información potencialmente útil"*.

### 5.5. La conclusión, enunciada con precisión

Cito la Sección 6 de ISMIR18 completa, porque es la frase que hay que citar:

> *"Si bien nuestros resultados sugieren que los modelos que dependen de conocimiento de dominio juegan un rol relevante en escenarios donde no hay datasets de tamaño considerable disponibles, hemos mostrado que, **dados suficientes datos, los modelos assumption-free que procesan formas de onda superan a aquellos que dependen de conocimiento de dominio musical**."*

Y el corolario, en el cierre de la Sección 5.3:

> *"Considerando los resultados sobresalientes que reportamos cuando el modelo de forma de onda se entrena con 1M de canciones, uno podría argumentar que **la falta de datasets públicos más grandes está limitando los resultados de la investigación en deep learning para auto-tagging musical**."*

Formulado como umbral operativo: **el punto de cruce está entre 500k y 1M de canciones etiquetadas.** Por debajo de eso, el diseño informado por dominio es una regularización que ayuda. Por encima, es una restricción que estorba. Con los datasets públicos disponibles en 2018 —y en 2026 la situación no cambió tanto para *tagging* supervisado— **estás siempre del lado donde el conocimiento de dominio gana.**

### 5.6. Conexión con el debate "raw waveform vs features" de la clase 39

CLASE39 plantea el debate en un slide titulado **"Can We Use Raw Audio Data"**, y su desarrollo es puramente computacional. Lo transcribo tal cual:

> *"Spectrograms, log-mel, etc. are hand-crafted features. Can we use DL to directly learn features from raw data? Yes, but we need to consider some issues.*
> - *To avoid loss of info, we need to sample audio data using a high sample rate 15-20KHz (44.1 KHz for music). This implies lot of samples per second. Any problem?*
> - *Using a convolutional architecture, we need huge filters or a very deep structure, why?*
> - *We can increase the receptive field of neurons in intermediate layers using dilated convolution filters."*

La clase responde el "cómo" (convoluciones dilatadas, WaveNet, el "Ejemplo 2" con 4 capas dilatadas de filtros 20×1, 10×1, 10×1, 5×1). **ISMIR18 responde el "cuándo", y esa es la pieza que falta.**

La reformulación precisa del debate, después de leer ISMIR18:

| | Espectrograma log-mel | Forma de onda cruda |
|---|---|---|
| Qué es la transformación | Un **prior fijo y no aprendido**: STFT + banco mel perceptual + compresión logarítmica | Nada; la red aprende todo |
| Qué asume | Que fase es descartable, que la escala mel es la resolución frecuencial correcta, que la compresión log modela la sonoridad | Solo estacionariedad local a escala de ~3 muestras |
| Costo en datos | Bajo: el prior sustituye datos | Alto: hay que aprender el banco de filtros desde ejemplos |
| Régimen donde gana | **< ~500k canciones** (ISMIR18, Tabla 1 + Figura 3) | **> ~1M canciones** (ISMIR18, Tabla 1) |
| Costo en cómputo | 62,5 frames/s | 16.000 muestras/s → 256× más pasos temporales |

Y hay un dato que ISMIR16 aporta y que refuerza la legitimidad del prior mel: cuando Dieleman y Schrauwen entrenaron un modelo end-to-end sobre forma de onda, *"los bancos de filtros aprendidos (…) muestran similitudes con la escala mel, apoyando el uso de la no-linealidad conocida del sistema auditivo humano"*. **El modelo end-to-end, si tiene datos suficientes, reinventa el mel-espectrograma.** Lo cual es a la vez la mejor defensa del prior (es correcto) y la mejor crítica (es prescindible con datos).

Nota final sobre coherencia interna: musicnn **es un modelo de espectrograma**. La clase 39 lo cita en el slide de referencias junto a **WaveNet**, que es el arquetipo del *raw audio*. Los dos lados del debate están en la misma lista de referencias, y musicnn es el que trae la evidencia cuantitativa del cruce.

---

## 6. Resultados

Repito la advertencia porque en esta sección es donde más se confunden las fuentes: **las cinco cifras de musicnn como tal salen del extended abstract de dos páginas y no tienen detrás una tabla arbitrada.**

### 6.1. Las cifras propias de musicnn (LBD19)

De la penúltima sección del abstract, verbatim: *"These models achieve state-of-the-art performance on the MagnaTagATune dataset: (…) But also for the Million Song Dataset: (…)"*.

| Modelo | Dataset | ROC-AUC | PR-AUC | Fuente |
|---|---|---|---|---|
| `MTT_musicnn` | MagnaTagATune | **90,69** | **38,44** | LBD19 |
| `MTT_vgg` | MagnaTagATune | 90,26 | 38,19 | LBD19 |
| `MSD_musicnn` | Million Song Dataset | 88,01 | 28,90 | LBD19 |
| `MSD_musicnn_big` | Million Song Dataset | **88,41** | **30,02** | LBD19 |
| `MSD_vgg` | Million Song Dataset | 87,67 | 28,19 | LBD19 |
| musicnn + capa de atención | MagnaTagATune | **90,77** | **38,61** | LBD19 |
| musicnn + capa de atención | Million Song Dataset | **88,81** | **31,51** | LBD19 |

Lecturas:

1. **En MTT, `musicnn` supera a `vgg` por +0,43 de ROC-AUC y +0,25 de PR-AUC.** Es una ventaja pequeña. Honestamente: **no es una demostración contundente de la tesis del dominio**, y el propio abstract no la presenta como tal — es una tabla de modelos disponibles, no un experimento controlado. La evidencia fuerte para la tesis está en ISMIR18 y CBMI16, no aquí.
2. **En MSD la ventaja es mayor: `MSD_musicnn` supera a `MSD_vgg` por +0,34 de ROC-AUC y +0,71 de PR-AUC**, y `MSD_musicnn_big` por +0,74 y +1,83. La brecha en PR-AUC es lo relevante.
3. **La variante con capa de atención en lugar de pooling temporal gana en los dos datasets**, y la ganancia sobre MSD es sustancial: **+0,80 de ROC-AUC y +2,61 de PR-AUC** sobre `MSD_musicnn`, y **+1,49 de PR-AUC sobre `MSD_musicnn_big` con la fracción del cómputo**. Esto es una señal clara de que el pooling media+máximo era el cuello de botella y anticipa lo que vendría después (atención → Transformers). El abstract llama a esto *"esta nueva arquitectura"* y remite a documentación online sin publicar detalles; **la arquitectura de atención no está descrita en ninguna de las fuentes verificadas de este análisis.**
4. **La afirmación "state-of-the-art"** hay que tomarla en su contexto de 2019 y de documento no arbitrado. Comparada con ISMIR18 (Tabla 3, mejor espectrograma: 90,40 / 38,11), `MTT_musicnn` mejora en +0,29 / +0,33. Es una mejora incremental sobre el propio trabajo previo del autor.

### 6.2. Contexto histórico: qué había antes

Reúno aquí los *baselines* de la época, **con la fuente de cada número marcada**. Advertencia sobre comparabilidad: ISMIR16 y EUSIPCO17 reportan solo ROC-AUC (llamada "AUC"), y ISMIR18 advierte que el resultado de SampleCNN (90,55) *"fue calculado con una versión ligeramente distinta del dataset MTT"* de ~21k canciones en lugar de ~26k, más limpia — por eso al reproducirlo con la versión original obtienen 88,56.

**MagnaTagATune, ROC-AUC (histórico):**

| Año | Método | ROC-AUC | #param | Fuente |
|---|---|---|---|---|
| 2011 | Pooling MFCC (Hamel et al.) | 86,1 | — | ISMIR16 T4 |
| 2012 | Multi-scale (Dieleman & Schrauwen) | 89,8 | — | ISMIR16 T4 |
| 2014 | Convoluciones 1D (Dieleman & Schrauwen) | 88,2 | — | ISMIR16 T4 |
| 2014 | Transferred learning (Van den Oord et al.) | 88,0 | — | ISMIR16 T4 |
| 2015 | Bag of features + RBM (Nam et al.) | 88,8 | — | ISMIR16 T4 |
| 2016 | **FCN-4 (Choi et al., 3×3, mel)** | **89,4** | 22M | ISMIR16 T3/T4 |
| 2016 | FCN-3 / FCN-5 (mel) | 85,2 / 89,0 | — | ISMIR16 T3 |
| 2016 | FCN-4 con STFT / con MFCC | 84,6 / 86,2 | — | ISMIR16 T3 |
| 2017 | Small-rectangular (3×3, ajustado) | 86,5 | 75k | EUSIPCO17 T3 |
| 2017 | Dieleman et al. (filtros altos) | 88,1 | 75k | EUSIPCO17 T3 |
| 2017 | **Timbre CNN de Pons (propuesto)** | **88,9** | **75k** | EUSIPCO17 T3 |
| 2017 | Timbre CNN de Pons ×2 | 89,3 | 191k | EUSIPCO17 T3 |
| 2018 | **Mejor espectrograma de Pons** | **90,40** | 5M | ISMIR18 T3 |
| 2019 | **`MTT_musicnn`** | **90,69** | ~0,78M* | LBD19 (*param: mi cálculo) |
| 2019 | musicnn + atención | 90,77 | — | LBD19 |

La fila más elocuente de esta tabla es la de EUSIPCO17: **el mismo presupuesto de 75k parámetros**, tres formas de filtro distintas, y el resultado ordena exactamente como predice la teoría del dominio: **rectangulares pequeños 86,5 < filtros altos 88,1 < múltiples formas musicalmente motivadas 88,9**. EUSIPCO17 aclara además que su comparación es un ablation puro: *"nuestros experimentos reproducen las mismas condiciones que Dieleman et al., ya que el modelo propuesto adapta su arquitectura a la estrategia de diseño propuesta — **únicamente modificamos la primera capa** para tener muchas formas de filtro musicalmente motivadas. Las otras capas se mantienen intactas"*.

Y el dato de eficiencia: **la propuesta de Pons ×2 alcanza 89,3 con 191k parámetros contra los 22M de Choi et al. para 89,4 — 115× menos parámetros para el mismo resultado.** (EUSIPCO17 anota al pie que *"resultados equivalentes se pueden lograr con 750k parámetros"* en la versión de Choi, lo que reduce el factor a ~4×; sigue siendo a favor.)

**Million Song Dataset, ROC-AUC (histórico):**

| Año | Método | ROC-AUC | PR-AUC | Fuente |
|---|---|---|---|---|
| 2016 | FCN-3 / FCN-4 (Choi et al.) | 78,6 / 80,8 | — | ISMIR16 T5 |
| 2016 | FCN-5 / FCN-6 / FCN-7 | 84,8 / **85,1** / 84,5 | — | ISMIR16 T5 |
| 2017 | VGG + RNN (Choi et al.) | 86,2 | — | ISMIR18 T4 |
| 2017 | SampleCNN (Lee et al.) | 88,12 | — | ISMIR18 T4 |
| 2017 | Multi-level & multi-scale (Lee & Nam) | **88,78** | — | ISMIR18 T4 |
| 2018 | SampleCNN multi-level & multi-scale | 88,42 | — | ISMIR18 T4 |
| 2018 | Waveform de Pons | 87,41 | 28,53 | ISMIR18 T4 |
| 2018 | **Espectrograma de Pons** | **88,75** | **31,24** | ISMIR18 T4 |
| 2019 | `MSD_vgg` | 87,67 | 28,19 | LBD19 |
| 2019 | `MSD_musicnn` | 88,01 | 28,90 | LBD19 |
| 2019 | `MSD_musicnn_big` | 88,41 | 30,02 | LBD19 |
| 2019 | musicnn + atención | **88,81** | **31,51** | LBD19 |

**Un detalle que hay que decir en voz alta:** `MSD_musicnn_big` (88,41 / 30,02) es **peor** que el modelo de espectrograma de ISMIR18 (88,75 / 31,24), del mismo autor y con configuración de midend/backend equivalente. La diferencia más plausible es la entrada de 15 s de ISMIR18 contra los ~3 s por defecto del código publicado, más posibles diferencias de entrenamiento; **ninguna de las fuentes verificadas explica esta discrepancia**. La conclusión práctica: los modelos empaquetados en la librería no son idénticos a los del paper de escala, y sus números no son intercambiables.

La otra observación de la tabla: en MSD **el salto grande no vino de la arquitectura del frontend sino de dos cosas distintas** — la profundidad (FCN-4 → FCN-5: **+4,0 puntos**, el mayor salto de toda la serie, y ISMIR16 lo atribuye correctamente a que *"modelos más complejos se benefician de más datos de entrenamiento"*) y, más tarde, del agregador temporal (pooling → atención: +2,6 de PR-AUC).

### 6.3. La progresión de PR-AUC como termómetro real

Si se ordenan solo los pares que reportan ambas métricas, se ve algo que ROC-AUC oculta:

| Modelo | MTT ROC / PR | MSD ROC / PR | Fuente |
|---|---|---|---|
| Espectrograma ISMIR18 | 90,40 / 38,11 | 88,75 / 31,24 | ISMIR18 T3, T4 |
| Waveform ISMIR18 | 89,05 / 34,92 | 87,41 / 28,53 | ISMIR18 T2, T4 |
| `_vgg` (LBD19) | 90,26 / 38,19 | 87,67 / 28,19 | LBD19 |
| `_musicnn` (LBD19) | 90,69 / 38,44 | 88,01 / 28,90 | LBD19 |
| `_musicnn_big` (LBD19) | — | 88,41 / 30,02 | LBD19 |
| + atención (LBD19) | 90,77 / 38,61 | 88,81 / 31,51 | LBD19 |

**El rango completo de ROC-AUC en MTT es 89,05–90,77: 1,72 puntos. El de PR-AUC es 34,92–38,61: 3,69 puntos, más del doble de resolución.** Y en MSD la relación es aún más marcada: 1,40 vs 3,32. Si quieres distinguir modelos de *tagging*, mira PR-AUC.

Segunda lectura: **la PR-AUC absoluta está en torno a 0,30–0,38.** Traducido: aun el mejor modelo de la época acertaba menos de dos de cada cinco predicciones positivas en promedio ponderado sobre el rango de *recall*. **El *tagging* musical en 2019 no era un problema resuelto**, ni de lejos, aunque la ROC-AUC de 0,90 sugiriera lo contrario. Ese es el mejor argumento a favor de reportar PR-AUC que puedo dar.

---

## 7. musicnn como herramienta

### 7.1. Instalación y qué se instala

```bash
pip install musicnn
```

De `setup.py` (CÓDIGO):

```python
version='0.1.0',
license='ISC',
classifiers=[..., "Development Status :: 3 - Alpha", "Programming Language :: Python :: 3.7"],
install_requires=['librosa>=0.7.0',
                  'tensorflow>=1.14',
                  'numpy<1.17,>=1.14.5']
```

Los pesos vienen dentro del paquete (`include_package_data=True`), así que no hay descarga en el primer uso.

### 7.2. Los tres usos, con código

El abstract organiza la librería en tres preguntas ("What's the musicnn library for?"). Los ejemplos que siguen son los del abstract y de las *docstrings* del código, con mis comentarios.

**Uso 1 — Etiquetador *out of the box*.**

```python
from musicnn.tagger import top_tags

# Devuelve las 10 etiquetas de mayor probabilidad agregadas sobre toda la pista.
# El modelo trocea el audio en parches de `input_length` segundos, predice cada
# parche por separado, y promedia las probabilidades a lo largo del tiempo.
top_tags('music_file.mp3', model='MTT_musicnn', topN=10)
```

Firma real (de `tagger.py`):

```python
def top_tags(file_name, model='MTT_musicnn', topN=3,
             input_length=3, input_overlap=False,
             print_tags=True, save_tags=False):
```

Desde la línea de comandos:

```bash
python -m musicnn.tagger music.au  --model 'MTT_musicnn' --topN 10 --print
python -m musicnn.tagger audio.wav -m     'MTT_vgg'      --topN 5  --save out.tags
```

**Uso 2 — Extracción de features (el taggram y las representaciones intermedias).**

```python
from musicnn.extractor import extractor

output = extractor(file, model='MTT_musicnn', extract_features=True)
taggram, tags, features = output
```

- **`taggram`**: matriz $T \times 50$ de probabilidades. **Cada fila es un parche temporal y cada columna una etiqueta.** No es un vector por canción: es una **serie temporal de etiquetas**, y esa es la razón por la que el nombre existe. Permite ver, por ejemplo, en qué segundo entra la voz o dónde cambia la instrumentación.
- **`tags`**: la lista de 50 strings que nombra las columnas (`MTT_LABELS` o `MSD_LABELS` de `configuration.py`).
- **`features`**: un `dict` con las representaciones intermedias. Para los modelos `musicnn`, las claves son exactamente las que enumera LBD19, y ahora sabemos qué tensor es cada una (Sección 4):

| Clave | Qué es | Forma | Dimensión de canales |
|---|---|---|---|
| `timbral` | `concat([f74, f77])` del frontend | $T \times 408$ | 408 |
| `temporal` | `concat([s1, s2, s3])` del frontend | $T \times 153$ | 153 |
| `cnn1` | `bn_conv1_t` del midend | $T \times 64$ (o 512) | 64 / 512 |
| `cnn2` | `res_conv2` del midend | $T \times 64$ (o 512) | 64 / 512 |
| `cnn3` | `res_conv3` del midend | $T \times 64$ (o 512) | 64 / 512 |
| `mean_pool` | media temporal de la concat densa | $753$ (o 2097) | vector por parche |
| `max_pool` | máximo temporal de la concat densa | $753$ (o 2097) | vector por parche |
| `penultimate` | salida de la densa de 200 (o 500) unidades | $200$ / $500$ | vector por parche |

Para los modelos `vgg`, las claves son `pool1` … `pool5`.

**Uso 3 — Transferencia.**

El abstract: *"nuestros modelos preentrenados pueden ser afinados, junto con una red de salida que actúa como clasificador, para realizar cualquier otra tarea musical"*. El experimento concreto lo detalla la Sección 8.

**Un cuarto uso implícito, útil en la práctica: el control del troceo.**

```python
# Parches de 1 s con 0.5 s de solape → taggram con resolución temporal fina.
# Solo funciona con los modelos `musicnn`; los `vgg` lanzan ValueError si
# input_length != 3, porque terminan en flatten de tamaño fijo.
taggram, tags = extractor('cancion.mp3', model='MSD_musicnn',
                          input_length=1, input_overlap=0.5,
                          extract_features=False)
```

Esa flexibilidad es **consecuencia directa del pooling temporal global** del backend, y es una de las ventajas concretas de la arquitectura sobre la VGG.

### 7.3. Estado de mantenimiento y trampa práctica

Esto importa si alguien piensa usar la librería hoy.

| Hecho | Valor | Fuente |
|---|---|---|
| Versión publicada en PyPI | **0.1.0**, única versión | PyPI (verificado vía API JSON) |
| Fecha de subida | **18 de agosto de 2019** | PyPI |
| Estado declarado | `Development Status :: 3 - Alpha` | `setup.py` |
| Dependencia de TensorFlow | `tensorflow>=1.14` | `setup.py` |
| Restricción de NumPy | `numpy<1.17,>=1.14.5` | `setup.py` |
| Python declarado | 3.7 | clasificador de `setup.py` |
| Estrellas / forks / commits del repo | 713 / 102 / 371 | GitHub, agosto 2026 |

**El paquete nunca tuvo una segunda versión en PyPI en siete años.** Los problemas concretos:

1. **`numpy<1.17`.** Ese *pin* es de julio de 2019. NumPy 1.17 salió ese mismo mes; hoy la línea estable está muy por encima. **Cualquier entorno moderno viola esta restricción**, y `pip` o bien fallará la resolución o bien degradará NumPy y romperá todo lo demás del entorno.
2. **TensorFlow 1.x en un mundo de TF 2.x.** El código está escrito íntegramente contra el API de sesiones de TF1: `models.py` usa `tf.compat.v1.layers.conv2d`, `tf.compat.v1.layers.batch_normalization`, etc., y `extractor.py` abre con `tf.compat.v1.disable_eager_execution()`. Funciona bajo TF 2.x mientras el *shim* `tf.compat.v1` exista y siga soportando `tf.compat.v1.layers`, que está deprecado desde hace años y es candidato natural a desaparecer. El propio archivo declara la intención: `# disabling deprecation warnings (caused by change from tensorflow 1.x to 2.x)`.
3. **`BATCH_SIZE = 1` por defecto.** El `FAQs.md` lo reconoce: *"¿Mi modelo es lento, incluso con GPU? ¡Sí! En `./musicnn/configuration.py` puedes poner un batch size mayor. El default es `BATCH_SIZE = 1`, lo que puede ser lento — pero es seguro computacionalmente."* Para procesar un catálogo, hay que editar el archivo de configuración del paquete instalado. No es un parámetro del API.
4. **Lectura de MP3 vía `audioread`.** El FAQ explica que desde `librosa` 0.7 se usa `soundfile` por defecto con *fallback* a `audioread` para MP3. `audioread` está deprecado en versiones recientes de `librosa`. Es otro punto de fricción para audio comprimido.

**Recomendación práctica.** Si el objetivo es usar estos modelos hoy, el camino con menos fricción no es `pip install musicnn` sino los **modelos de `essentia-tensorflow`** publicados por el mismo MTG, que reempaquetaron los pesos de musicnn (y de VGGish, OpenL3 y otros) como grafos congelados servibles desde Essentia, sin arrastrar el *pin* de NumPy. *(Esta recomendación no está verificada contra ninguno de los PDFs de este análisis; es conocimiento de contexto sobre el ecosistema del MTG y debe tratarse como tal.)*

---

## 8. Transferencia y embeddings musicales

### 8.1. El experimento del abstract

LBD19 reporta un experimento de transferencia deliberadamente simple. Pipeline, textual: **"feature extraction" + 128 PCA + SVM**. Dataset: **GTZAN (fault-filtered)** — la variante de GTZAN con las particiones corregidas para eliminar las repeticiones y fugas de artista que Sturm documentó en el dataset original. Métrica: exactitud sobre el test.

| Extractor de features | Exactitud GTZAN (fault-filtered) | Dataset de preentrenamiento |
|---|---|---|
| **VGGish (AudioSet)** | **77,58 %** | AudioSet, ~2M audios |
| **`MSD_musicnn`** | **77,24 %** | MSD, 200k canciones |
| OpenL3 (AudioSet) | 74,65 % | AudioSet |
| `MTT_vgg` | 72,75 % | MTT, 19k canciones |
| `MTT_musicnn` | 71,37 % | MTT, 19k canciones |

Las dos conclusiones que el abstract extrae, textuales:

> *"Nótese que nuestros modelos preentrenados en MSD superan a los de MTT. Además, `MSD_musicnn` logra resultados similares a las features de VGGish audioset (que están entrenadas con un dataset mucho más grande: 2M de audios)."*

**Lo que esto realmente dice, ordenado por importancia:**

1. **El tamaño del dataset de preentrenamiento domina sobre la arquitectura.** `MSD_musicnn` (77,24) supera a `MTT_musicnn` (71,37) por **5,87 puntos**. Misma arquitectura exacta, mismo número de parámetros, misma tarea de *tagging*, mismo vocabulario de 50 etiquetas: **la única diferencia es 19k vs 200k canciones de entrenamiento**. Ese es el resultado más contundente del abstract, y es coherente con el hallazgo de escala de ISMIR18.
2. **La arquitectura importa poco *para la transferencia*, y en MTT hasta va en contra.** `MTT_vgg` (72,75) **supera** a `MTT_musicnn` (71,37) por 1,38 puntos, aunque `MTT_musicnn` es mejor tagger sobre MTT (90,69 vs 90,26 ROC-AUC). **Ser mejor en la tarea de origen no garantiza transferir mejor.** El abstract no comenta esta inversión.
3. **musicnn compite con VGGish usando 10× menos datos de preentrenamiento.** 77,24 vs 77,58 con 200k canciones contra ~2M audios. Es el argumento de eficiencia de datos, aplicado ahora al preentrenamiento en vez de a los parámetros.

**Advertencia metodológica importante.** El pipeline es `features → PCA a 128 dimensiones → SVM`. La compresión a 128 componentes principales **es agresiva y no está justificada en el abstract**, y aplasta representaciones de dimensionalidades muy distintas (753 de `max_pool` de musicnn, 128 de VGGish, 512 de OpenL3) a un espacio común. La comparación es honesta en el sentido de que el pipeline es idéntico para todos, pero **las diferencias de 1–3 puntos no son interpretables** sin barras de error, que el abstract no da. La diferencia de 5,87 puntos entre MTT y MSD sí es lo bastante grande para ser creíble.

Qué capa usar para transferencia, según `FAQs.md`:

> *"Aunque no hemos corrido tests exhaustivos, a través de nuestras visualizaciones y experimentos preliminares encontramos que el **taggram** y la capa **`max_pool`** son las mejores para este propósito. El taggram porque ya provee información musical de alto nivel, y la capa `max_pool` porque provee una representación acústica relativamente **dispersa** de la música."*

Que el `taggram` funcione bien como *embedding* es notable: son 50 probabilidades interpretables por parche, y aun así compiten con representaciones de cientos de dimensiones. Es la versión musical de usar las logits de ImageNet como features.

### 8.2. El paralelo con VGGish

La analogía es directa y vale la pena explicitarla porque la clase 39 vive del lado de los sonidos generales:

| | **VGGish** | **musicnn** |
|---|---|---|
| Dominio | Sonidos generales | Música |
| Tarea de preentrenamiento | Clasificación multi-etiqueta sobre AudioSet | Music audio tagging sobre MTT / MSD |
| Escala del preentrenamiento | ~2M audios (LBD19) | 19k (MTT) / 200k (MSD) canciones (LBD19) |
| Arquitectura | VGG sobre log-mel (filtros 3×3) | Frontend musicalmente motivado + midend residual + pooling temporal |
| Uso canónico | *Embedding* congelado → clasificador ligero | *Embedding* congelado → clasificador ligero |
| Salida usada como feature | Capa de *embedding* de 128 dims | `max_pool` (753) o `taggram` (50) |
| GTZAN fault-filtered + PCA128 + SVM | **77,58 %** | **77,24 %** (`MSD_musicnn`) |

La lógica de por qué esto funcionó tan bien en 2019–2022 es la misma que sostuvo a ImageNet como preentrenamiento en visión: **una tarea de clasificación con vocabulario amplio, entrenada sobre un corpus grande, produce representaciones intermedias que transfieren a tareas del mismo dominio.** El *tagging* musical resulta ser un buen pretexto porque su vocabulario cubre simultáneamente timbre (`guitar`, `piano`), estructura temporal (`fast`, `beat`), producción (`acoustic`, `synth`) y afecto (`sad`, `happy`), obligando a la red a representar todas esas dimensiones.

Y el punto de contraste que la clase 39 puede aprovechar: **AudioSet es 10× más grande que MSD, pero un modelo específico de dominio, entrenado con 10× menos datos, iguala su transferencia sobre una tarea musical.** Especializar el corpus vale tanto como escalarlo, cuando la tarea final es del mismo dominio.

### 8.3. Dónde se usó en la práctica

Los *embeddings* de musicnn se volvieron un componente estándar en el MIR aplicado por tres razones concretas y no glamorosas:

1. **Instalación de una línea y pesos incluidos.** No había que entrenar nada ni descargar checkpoints externos.
2. **CPU basta.** `FAQs.md`: *"¿Puedo correr musicnn en CPU? Sí, los modelos ya están entrenados."* Con ~0,78M de parámetros, la inferencia es trivial.
3. **El taggram es interpretable.** A diferencia de un vector opaco de 128 dimensiones, un taggram se puede graficar y auditar: cada columna tiene nombre.

Su uso típico era como *feature extractor* congelado seguido de un clasificador ligero (SVM, regresión logística, MLP pequeño) sobre tareas con pocos datos etiquetados: clasificación de género en catálogos propios, detección de instrumento predominante, estimación de *mood* para *playlisting*, filtros de similitud en sistemas de recomendación basados en contenido. **Esta caracterización de usos es conocimiento de contexto y no está verificada contra los PDFs de este análisis.**

---

## 9. Limitaciones

### 9.1. Sesgo de los datasets

**Ambos datasets son música occidental comercial.** MTT viene del catálogo de Magnatune, un sello independiente estadounidense; MSD son *previews* de 30 s de catálogo comercial con etiquetas de Last.fm. Basta mirar los vocabularios de la Sección 2.1 para ver el sesgo estructural:

- **De las 50 etiquetas de MSD, siete son décadas** (`00s`, `80s`, `90s`, `70s`, `60s`) o marcadores de época (`oldies`, `classic rock`). **Son metadatos de catálogo, no propiedades del audio.** Un modelo que los predice bien está aprendiendo convenciones de producción de una industria específica, no estructura musical.
- **La única etiqueta no-occidental en MTT es `indian`** (más `sitar`, que es el mismo referente). En MSD no hay ninguna. Ni `flamenco`, ni `cumbia`, ni `gamelan`, ni `maqam`, ni ninguna música de tradición no europea con vocabulario propio. El MTG, siendo el grupo que impulsó el proyecto CompMusic sobre tradiciones musicales no occidentales, no podía ignorar esto: los datasets simplemente no existían.
- **Etiquetas subjetivas sin definición operacional:** `beautiful`, `sexy`, `catchy`, `weird`, `soft`. No hay criterio de anotación reproducible detrás.

Para el contexto latinoamericano de la clase 39 esto es directamente relevante: **un `MSD_musicnn` aplicado a un catálogo chileno de cueca, nueva canción o cumbia va a producir etiquetas del vocabulario que tenga más cerca**, no un "no sé". El `FAQs.md` lo admite con precisión:

> *"¿Cuáles son los casos típicos donde el modelo falla? Cuando el audio de entrada tiene contenido que está fuera del vocabulario de 50 etiquetas. Aunque en estos casos las predicciones son consistentes y razonables, **el modelo no puede predecir `bass` si esa etiqueta no es parte de su vocabulario**."*

### 9.2. Vocabulario ruidoso y redundante

Ya lo mencioné pero merece su propio párrafo porque tiene consecuencias medibles. En MTT hay al menos siete etiquetas que refieren a presencia de voz (`vocal`, `vocals`, `voice`, `singing`, `male vocal`, `male voice`, `female vocal`, `female voice`) y tres negaciones separadas (`no vocal`, `no vocals`, `no voice`). Un modelo perfecto tendría que decidir, para un mismo clip, entre `vocal` y `vocals` — una distinción que no existe.

Esto pone un **techo estructural sobre la métrica**, y ISMIR16 lo detectó antes que nadie (Sección 5.1):

> *"El dataset MagnaTagATune ha sido relativamente popular para tagging basado en contenido, pero los desempeños similares de trabajos recientes parecen sugerir que **el desempeño está saturado, es decir, se ha alcanzado un techo de cristal debido al ruido en la anotación**."*

Y confirma en la Sección 5.2: *"muchos algoritmos diferentes solo muestran diferencias pequeñas en el rango de un puntaje AUC de 0,88 – 0,89, lo que hace difícil comparar sus desempeños"*. Ese es el rango donde vivieron todos los modelos de MTT entre 2012 y 2016. Que musicnn llegue a 90,69 en 2019 es una mejora real, pero **la mitad del margen que queda hasta 100 es probablemente irrecuperable**, porque es ruido de anotación.

### 9.3. Etiquetas débiles

**Todas las etiquetas son a nivel de clip completo** (30 s en MTT y MSD). No hay anotación temporal. La consecuencia arquitectónica está a la vista: musicnn produce un taggram con resolución de parche, pero **fue entrenado sin ninguna supervisión sobre dónde ocurre cada cosa**. Si un clip lleva `piano` y el piano suena solo en los últimos 5 s, la señal de entrenamiento le dice al modelo que los 30 s son `piano`.

El *pooling* media+máximo del backend es un paliativo de aprendizaje multi-instancia (*multiple instance learning*): el máximo permite que un evento local dispare la etiqueta sin obligar a que esté presente en todo el clip. Pero es un paliativo, no una solución. **Los taggrams son diagnósticos útiles, no detecciones calibradas temporalmente**, y no hay en ninguna de las fuentes verificadas una evaluación de su precisión temporal.

### 9.4. Resolución temporal y espectral

| Limitación | Valor | Consecuencia |
|---|---|---|
| Frecuencia de muestreo | 16 kHz | **Todo por encima de 8 kHz se descarta.** Brillo, aire, detalle de platillos: fuera. La clase 39 recomienda 44,1 kHz para música. |
| Resolución temporal | 16 ms/frame | Un ataque de percusión de 5 ms se reparte en un frame. Insuficiente para transcripción u *onset detection* fino. |
| Ventana STFT | 32 ms | Resolución frecuencial de ~31 Hz. **Insuficiente para separar armónicos graves**: entre un Do2 (65,4 Hz) y un Do#2 (69,3 Hz) hay 3,9 Hz. |
| Bandas mel | 96 | Compresión perceptual; imposible recuperar $f_0$ exacto. |
| Parche por defecto | 3 s | Cubre unos pocos compases. **No hay ningún mecanismo que represente estructura de forma musical** (estrofa/estribillo, desarrollo). |

Los filtros temporales más largos (`s1`, de 128 frames ≈ 2,05 s) son lo más lejos que llega la arquitectura en el eje del tiempo, y eso es aproximadamente **un compás a 120 BPM en 4/4**. Todo lo que sea estructura de canción está fuera de alcance por construcción.

### 9.5. El pooling temporal como cuello de botella

La evidencia está en el propio LBD19: sustituir el pooling temporal por una capa de atención da **+2,61 puntos de PR-AUC en MSD** (28,90 → 31,51). Y ISMIR18 ya lo sospechaba, ofreciéndolo como explicación de por qué su waveform quedaba por debajo de SampleCNN: *"el modelo de forma de onda emplea una estrategia de global pooling **que podría estar removiendo información potencialmente útil** para el modelo"*. Media y máximo son estadísticos de primer orden: descartan **todo el orden temporal**. Una canción y la misma canción reproducida al revés producen exactamente el mismo `mean_pool` y casi el mismo `max_pool`.

### 9.6. Deuda técnica

Cubierta en la Sección 7.3: `numpy<1.17`, TF 1.x vía `tf.compat.v1`, `BATCH_SIZE=1`, versión única desde 2019, estado *Alpha*. En términos prácticos, **hoy musicnn requiere un entorno aislado y congelado para funcionar**, lo cual es una barrera real para uso en producción.

### 9.7. Qué lo dejó obsoleto

Cinco cosas, en orden aproximado de impacto:

1. **Los Transformers de audio.** Modelos tipo AST/PaSST tratan el espectrograma como secuencia de parches con auto-atención global, resolviendo de raíz tanto el problema del campo receptivo como el del *pooling* temporal. La ganancia de la variante de atención del propio LBD19 es el primer indicio de esto en la línea de Pons.
2. **El preentrenamiento autosupervisado.** El *tagging* supervisado necesita etiquetas; los objetivos contrastivos y de enmascaramiento no. Con corpus de audio sin etiquetar mucho mayores que MSD, la restricción de escala que ISMIR18 identificó como cuello de botella desaparece por el otro lado.
3. **Los modelos audio-texto contrastivos (CLAP).** Eliminan el vocabulario cerrado de 50 etiquetas: la consulta puede ser cualquier frase en lenguaje natural. Esto ataca directamente la limitación 9.1 (sesgo de vocabulario) y 9.2 (redundancia de etiquetas).
4. **Mejores CNNs.** Aun dentro del paradigma convolucional, los estudios comparativos posteriores encontraron que arquitecturas de *chunk* corto con conexiones residuales superan a musicnn en MTT. *(No verificado contra PDF en este análisis.)*
5. **El ecosistema de software.** TF 1.x murió; PyTorch + `torchaudio` + Hugging Face se volvieron el camino por defecto.

---

## 10. Impacto y legado

### 10.1. Lo que musicnn dejó instalado

Tres cosas sobreviven al modelo.

**(a) El vocabulario "frontend / midend / backend".** La taxonomía de ISMIR18 —el frontend interactúa con la señal y la mapea a un espacio latente; el backend predice la salida— se volvió lenguaje común en audio deep learning. La distinción *fixed-length input back-end* vs *variable-length input back-end* sigue siendo la pregunta correcta al diseñar cualquier modelo de audio.

**(b) El argumento de la forma del filtro.** Es la contribución conceptual duradera. La conclusión no es "usa filtros 7×38"; es **"la forma del filtro es una hipótesis sobre la estructura de la señal, y el cuadrado es una hipótesis, no la ausencia de una"**. Ese argumento sobrevive intacto a los Transformers: hoy se traduce en cómo dividir el espectrograma en parches, si conviene atención factorizada tiempo/frecuencia, y qué tamaño de parche usar. La pregunta cambió de forma, no de fondo.

**(c) El mapa del compromiso datos-vs-priors, con números.** La Tabla 1 de ISMIR18 es una de las pocas mediciones limpias que existen sobre **cuántos datos hace falta para que el end-to-end supere al diseño informado por dominio** en audio. El umbral —entre 500k y 1M de canciones etiquetadas— es un dato reutilizable en cualquier decisión de arquitectura para audio.

### 10.2. Qué reemplazó a qué, y por qué

Lo que sigue es reconstrucción de contexto histórico. **Nada de esta subsección está verificado contra los PDFs de este análisis**; las cifras específicas se omiten deliberadamente por esa razón.

**Transformers de audio (AST, PaSST, y sucesores).** El *Audio Spectrogram Transformer* aplica ViT al espectrograma: se trocea en parches, se linealizan y se procesan con auto-atención. **Qué reemplazó:** el *pooling* temporal media+máximo de musicnn. **Por qué ganó:** la auto-atención agrega sobre el tiempo de forma **dependiente del contenido y preservando el orden**, en vez de con un estadístico ciego. Es la generalización de la capa de atención que el propio LBD19 reportó como mejora. **Qué costó:** el prior de dominio desaparece (los parches son cuadrados otra vez) y hay que compensarlo con preentrenamiento masivo, típicamente sobre AudioSet o con transferencia desde ImageNet.

Vale la pena notar que **CLASE39 es escéptica de los Transformers para audio** y da tres razones en su slide *"Audio and Transformers"*: falta de datasets de audio masivos, la dificultad de segmentar audio en entidades discretas (trivial en texto, no en audio), y que *"Transformers are not good to model long dependencies in sequences"*. Concluye: *"As a consequence, Transformers are not currently very popular for audio applications"*. **Esa evaluación quedó desactualizada.** Los tres problemas se resolvieron: AudioSet y sus sucesores dieron escala, el troceo en parches de espectrograma resolvió la segmentación, y las variantes eficientes de atención más el preentrenamiento resolvieron el largo de secuencia. Es un punto que conviene marcar al usar el material de la clase.

**CLAP y los modelos audio-texto contrastivos.** Dos torres —un codificador de audio y uno de texto— entrenadas con InfoNCE para alinear pares (audio, descripción). **Qué reemplazó:** el vocabulario cerrado de 50 etiquetas. **Por qué ganó:** habilita *zero-shot tagging* con cualquier frase ("guitarra acústica con voz femenina y ritmo lento") sin reentrenar, y elimina de raíz el problema de `vocal` vs `vocals`. **La analogía es exacta:** CLAP es a musicnn lo que CLIP es a un clasificador de ImageNet. La supervisión pasa de un conjunto fijo de clases a lenguaje natural.

**MERT.** Preentrenamiento autosupervisado específico para música, con enmascaramiento sobre representaciones tokenizadas y objetivos que incluyen información acústica y tonal. **Qué reemplazó:** el *tagging* supervisado sobre MSD como tarea de pretexto para producir *embeddings*. **Por qué ganó:** rompe el techo de datos. musicnn estaba limitado a las 200k canciones **etiquetadas** de MSD; un objetivo autosupervisado puede consumir cualquier catálogo de música sin etiquetas. Es exactamente la puerta de salida al problema que ISMIR18 diagnosticó ("la falta de datasets públicos más grandes está limitando los resultados").

**Modelos generativos de música (Jukebox, MusicGen, Stable Audio, Suno, Udio).** Estos **no reemplazan** a musicnn: cambian la pregunta. Pasan de "¿qué etiquetas tiene este audio?" a "genera audio que satisfaga esta descripción". Pero hay una conexión técnica directa que vale la pena hacer explícita: **el *tagging* y la generación condicionada por texto son la misma alineación audio-texto recorrida en direcciones opuestas.** Un modelo texto-a-música necesita, internamente, un encoder que entienda qué significa "jazz melancólico con saxofón" en términos acústicos, y ese es precisamente el problema que musicnn resolvía en la dirección analítica. La generación absorbió y superó la comprensión.

### 10.3. Cronología condensada

| Año | Hito | Contribución |
|---|---|---|
| 2009 | MagnaTagATune (Law et al.) | El dataset que definió el benchmark |
| 2011 | Million Song Dataset (Bertin-Mahieux et al.) | Escala pública, etiquetas de Last.fm |
| 2014 | Dieleman & Schrauwen, ICASSP | End-to-end sobre forma de onda; los filtros aprendidos se parecen a mel |
| **2016** | **CBMI16 (Pons, Lidy, Serra)** | **Filtros con forma musical: `Time`, `Frequency`, `Time-Frequency`** |
| 2016 | ISMIR16 (Choi et al.) | La FCN de 3×3 sobre mel; el *baseline* `vgg` de musicnn |
| **2017** | **EUSIPCO17 (Pons et al.)** | **Teoría de invarianzas y filtros multi-forma para timbre** |
| 2017 | SampleCNN (Lee et al.) | Pila profunda de filtros $3\times1$ sobre muestras crudas |
| **2018** | **ISMIR18 (Pons et al.)** | **El experimento de escala: el cruce entre 500k y 1M** |
| **2019** | **LBD19: musicnn** | **La librería y los pesos preentrenados** |
| 2019+ | Atención sobre pooling | Ya anticipado en el propio LBD19 (+2,61 PR-AUC en MSD) |
| ~2021 | Transformers de audio | Auto-atención global reemplaza al pooling |
| ~2022 | CLAP | Vocabulario abierto, *zero-shot* |
| ~2023 | MERT, MusicGen | Autosupervisión musical; generación texto-a-música |

---

## 11. Conexión con la clase 39

### 11.1. Qué contiene la familia "Music" que la clase trata brevemente

CLASE39 divide las aplicaciones de audio en tres familias y declara su alcance explícitamente:

> *"According to the nature of audio signals, we can divide their applications into 3 main categories: **General sounds** (Ex. environmental sound classification, audio tagging, ...), **Speech** (Ex. Speech recognition, speech translation, speaker identification, ...), **Music** (Ex. song recognition, musical instrument identification, ...). This class mostly focuses on environmental sounds. Next class we will discuss speech and voice. **We will discuss music just briefly.**"*

Y el slide *"Audio Applications: Music"* lista exactamente cuatro tareas:

> *"Song recognition. Song/music-style similarity. Music instrument detection. Music transcription."*

Lo que sigue llena ese hueco: qué es cada una y con qué arquitectura se abordaba alrededor de 2019.

**(1) Song recognition — identificar *qué canción exacta* es esta grabación.**

Es el problema de Shazam. Cruciamente, **no es una tarea de deep learning y nunca lo fue**: es *audio fingerprinting*. Se extraen puntos característicos del espectrograma (típicamente máximos locales, o "constelaciones") y se codifican pares de picos en hashes robustos a ruido, compresión y ecualización; el matching es una búsqueda exacta en una tabla hash gigante. La razón de que no sea deep learning es estructural: **es un problema de identificación exacta, no de generalización**. Se quiere reconocer *esta* grabación específica, no la clase de grabaciones similares. Una red que generaliza está haciendo exactamente lo contrario de lo que se pide.

Relación con musicnn: **ninguna directa, y es importante decirlo.** El taggram de musicnn describe *qué tipo* de música es; el fingerprint identifica *cuál* pieza es. Confundirlas es el error conceptual más común en esta familia. *(Descripción del fingerprinting: conocimiento de contexto, no verificado contra los PDFs de este análisis.)*

**(2) Song/music-style similarity — encontrar canciones parecidas.**

Aquí musicnn **es** la respuesta canónica de la época. El pipeline es exactamente el de la Sección 8: `audio → musicnn → mean_pool/max_pool/taggram → vector → similitud coseno o vecinos más cercanos`. El *embedding* de 753 dimensiones (o el taggram de 50) es la representación; la métrica hace el resto.

Y este es el punto donde la clase 39 se conecta con el trabajo cotidiano de Roberto: **es literalmente record linkage sobre vectores.** Un bi-encoder que mapea entidades a un espacio métrico donde la cercanía significa "lo mismo o casi lo mismo", y una etapa de recuperación aproximada de vecinos que actúa como *blocker* antes de un scorer más caro. La arquitectura del problema es idéntica; solo cambia el codificador. Vale también la advertencia paralela: igual que un *embedding* de nombres captura la ortografía pero no la identidad, el *embedding* de musicnn captura textura sonora pero no autoría, letra ni estructura.

Contexto histórico: esta tarea es la razón original por la que existe el *auto-tagging*. ISMIR18 abre con ella: *"Un objetivo fundamental de la investigación en informática musical es estructurar automáticamente colecciones grandes de música. (…) las estimaciones de etiquetas pueden ser útiles para definir un espacio semántico ventajoso para organizar automáticamente bibliotecas musicales."*

**(3) Music instrument detection — qué instrumentos suenan.**

Es una tarea de **timbre puro**, y por lo tanto es donde la tesis de Pons es más directamente aplicable. La rama timbral del frontend (filtros 7×38 y 7×67, anchos en frecuencia, cortos en tiempo, con max-pool sobre todo el eje frecuencial para invarianza al pitch) está diseñada exactamente para esto.

EUSIPCO17 lo evaluó explícitamente sobre **IRMAS** (6.705 fragmentos de 3 s con un instrumento predominante para entrenamiento, 2.874 de 5–20 s con múltiples instrumentos para test, 11 instrumentos tonales anotados):

| Modelo | Micro-F1 | Macro-F1 | #param | Fuente |
|---|---|---|---|---|
| Bosch et al. (bag-of-frames + SVM) | 0,503 | 0,432 | — | EUSIPCO17 T2 |
| Han et al. (CNN profunda de 3×3, SOTA) | **0,602** | **0,503** | 1.446k | EUSIPCO17 T2 |
| Pons, single-layer | 0,559 | 0,484 | **62k** | EUSIPCO17 T2 |
| Pons, multi-layer | 0,589 | **0,516** | 743k | EUSIPCO17 T2 |

**Con la mitad de parámetros, la versión multicapa de Pons queda a 0,013 de micro-F1 del estado del arte y lo supera en macro-F1** (0,516 vs 0,503; macro es la métrica que no se deja dominar por los instrumentos frecuentes). Y **la versión de una sola capa con 62k parámetros —23× menos que el SOTA— supera claramente al bag-of-frames clásico.**

También sobre etiquetas de instrumento en MTT (`guitar`, `piano`, `violin`, `drums`, `flute`, `harpsichord`, `sitar`, `harp`, `cello` son 9 de las 50) y en clasificación de fonemas de canto sobre ópera de Pekín, donde EUSIPCO17 reporta la diferencia más marcada de todo el paper con presupuesto de parámetros igualado:

| Modelo | Exactitud *dan* | Exactitud *laosheng* | #param |
|---|---|---|---|
| **Propuesto (multi-forma)** | **0,484** | **0,432** | 222k |
| Small-rectangular (3×3, Choi et al.) | 0,374 | 0,359 | 222k |
| GMM (13 MFCC + Δ + ΔΔ) | 0,290 | 0,322 | — |
| MLP | 0,284 | 0,282 | 481k / 430k |

**+11,0 y +7,3 puntos sobre el 3×3 con exactamente los mismos 222k parámetros.** EUSIPCO17 comenta: *"modelos profundos basados en filtros pequeños-rectangulares —que son estado del arte en otros datasets— no funcionan tan bien como el modelo propuesto **en estos datasets pequeños**"*. Es el mismo patrón de ISMIR18, a escala de dataset aún menor: **cuanto menos datos, más paga el prior de dominio.**

**(4) Music transcription — audio a partitura/MIDI.**

Es la tarea más difícil de las cuatro y la más lejana de musicnn. Requiere resolución en pitch (identificar $f_0$ exacto), resolución temporal (onsets y offsets precisos) y separación de fuentes en música polifónica. **musicnn no sirve para esto y su arquitectura lo hace imposible por construcción**, por tres razones concretas que ya vimos: (a) el mel de 96 bandas destruye la resolución de $f_0$; (b) el max-pool sobre todo el eje de frecuencia **descarta deliberadamente** en qué banda ocurrió la activación, que es exactamente la información que la transcripción necesita; (c) el *pooling* temporal global elimina toda referencia temporal.

Es un buen ejemplo pedagógico de que **las invarianzas que hacen bueno a un modelo para una tarea lo inutilizan para otra**. La invarianza al pitch es un activo para reconocer un violín y un pasivo absoluto para transcribir lo que el violín toca. ISMIR16 hace la observación complementaria sobre representaciones: *"la CQT se ha usado predominantemente donde las frecuencias fundamentales de las notas deben identificarse con precisión, por ejemplo reconocimiento de acordes y transcripción"*, mientras que mel domina en *tagging*. **La elección de representación de entrada ya decide qué tareas son posibles.**

Y sobre pooling, ISMIR16 lo dice de forma que aplica directo: *"el pooling en el eje del tiempo puede ser útil para reconocimiento de acordes, pero perjudicaría la resolución temporal en métodos de detección de fronteras"*.

**Resumen de la familia:**

| Tarea del slide | Enfoque canónico ~2019 | ¿Sirve musicnn? | Por qué |
|---|---|---|---|
| Song recognition | Audio fingerprinting (no DL) | **No** | Identificación exacta, no generalización |
| Song/style similarity | *Embedding* + vecinos más cercanos | **Sí, era el estándar** | `max_pool` / `taggram` como vector |
| Instrument detection | CNN sobre espectrograma con filtros timbrales | **Sí, directamente** | Es la rama timbral del frontend |
| Music transcription | CQT + CNN/RNN con salida por nota | **No** | Mel + max-pool en frecuencia destruyen el pitch |

### 11.2. El "Ejemplo 1" de la clase contra el frontend de musicnn

Este es el contraste que más vale la pena desarrollar, porque es donde la clase hace una elección de diseño y Pons ofrece el argumento de por qué esa elección no es neutra.

**El "Ejemplo 1" de CLASE39, transcrito literalmente:**

> **Input:** *"40D Log-mel feats for overlapped segments of 10-20ms. 5-10ms overlap."*
> **CNN:** *"2 convolutional layers. Each with: i) 256 filter, ii) **9x9 and 4x4 filter sizes**, respectively. Optional max-pooling **in frequency only**. Ex. Non-overlapped windows of size 3. Batch normalization is optional. Add 1x1 convolution to reduce dimension."*
> **RNN:** *"2 LSTM layers. Cells in LSTMs with 256D. Need to normalize sequence length in minibatch."*
> **MLP:** *"2 FC layers. Each FC layer has 1.024 hidden units. Output: softmax (or sigmoids) for class label(s)."*

Y la receta general que la clase construye slide a slide:

> *"CNNs: good properties to learn local features. RNNs: good properties to learn temporal features. MLPs: good properties to classify input data."*
> *"CNNs: Learn meaningful local features. RNNs: Learn distance and global temporal features. MLPs: Learn good classifiers."*

Esta arquitectura es esencialmente el **CLDNN** de Sainath et al. 2015, que la clase cita en sus referencias. Es una receta sensata y bien probada. **El punto de Pons no es que esté mal; es que contiene una hipótesis oculta.**

**Comparación estructural:**

| Dimensión | "Ejemplo 1" (CLASE39) | Frontend de musicnn |
|---|---|---|
| Entrada | 40 bandas log-mel, segmentos de 10–20 ms con 5–10 ms de solape | 96 bandas log-mel, hop de 16 ms, ventana de 32 ms |
| Primera capa | **1 forma de filtro: 9×9**, 256 filtros | **5 formas en paralelo:** 7×38, 7×67, 128×1, 64×1, 32×1 |
| Segunda capa | 1 forma: 4×4, 256 filtros | (no hay; el frontend tiene **una sola capa**) |
| Cobertura frecuencial de la capa 1 | 9/40 = **22,5%** del espectro | 40% y 70% (timbral); 1/96 ≈ **1%** (temporal) |
| Cobertura temporal de la capa 1 | 9 frames ≈ **90–180 ms** | 112 ms (timbral); **0,51 / 1,02 / 2,05 s** (temporal) |
| Pooling | *"in frequency only"*, ventanas de 3 | **Sobre TODO el eje de frecuencia** (invarianza al pitch total) |
| Agregación temporal | **2 capas LSTM de 256D** | **Media + máximo global** (sin parámetros, sin recurrencia) |
| Clasificación | 2 FC de 1.024 unidades | 1 densa de 200/500 + salida |
| Entrada de largo variable | **No** (LSTM requiere normalizar el largo en el minibatch) | **Sí**, por construcción del pooling global |

**Qué asume cada uno sobre la estructura de la señal.**

**El "Ejemplo 1" asume que la información relevante es local y aproximadamente isótropa en el plano tiempo-frecuencia**, y que el contexto largo se construye por composición: primero la CNN forma features locales, después la LSTM los integra en el tiempo. Es una hipótesis limpia y modular —"CNN para lo local, RNN para lo temporal, MLP para clasificar"— y funciona bien cuando la señal *es* localmente estructurada. Para sonidos ambientales (que es donde la clase se enfoca), un ladrido o una sirena tienen firmas relativamente compactas y esa hipótesis es razonable.

**musicnn asume que la señal musical tiene dos tipos de estructura, no uno, y que ambos deben capturarse en la primera capa**: el timbre está deslocalizado **en frecuencia** (armónicos separados por decenas de bandas mel) y el ritmo está deslocalizado **en tiempo** (segundos, no milisegundos). Ninguno de los dos cabe en un 9×9, y no por falta de campo receptivo agregado sino por **cómo se gasta la capacidad**: un filtro cuadrado que quiera cubrir 2 segundos y 70 bandas necesitaría ser $125 \times 70$, con 8.750 pesos por canal, la mayoría irrelevantes. Los filtros de musicnn cubren los mismos rangos con $7\times67 = 469$ y $128\times1 = 128$ pesos, porque **descartan explícitamente la parte del plano que no aporta a cada concepto**.

Puesto en una sola frase: **el filtro cuadrado es la hipótesis de que las dos direcciones del plano tiempo-frecuencia son igualmente informativas a la misma escala; los filtros de musicnn son la hipótesis de que no lo son.**

**Los cuatro puntos concretos de fricción entre las dos recetas:**

**(a) La CNN 2D no es una alternativa a la RNN; es su reemplazo si la diseñas bien.** La clase asigna a la CNN "features locales" y a la RNN "features temporales". CBMI16 muestra que **un solo filtro $1\times60$ (1,4 s) alcanza 81,79% sobre Ballroom, contra 82,3% de un $k$-NN sobre BPM anotado a mano, con 7.336 parámetros y sin ninguna recurrencia**. La estructura temporal de escala de segundos es capturable con convolución si el filtro tiene el largo adecuado. musicnn no tiene RNN en ninguna parte y aun así modela ritmo — lo hace con filtros de 32, 64 y 128 frames en la primera capa.

**(b) El pooling en frecuencia de la clase es tímido.** CLASE39 propone *"optional max-pooling in frequency only, non-overlapped windows of size 3"*. musicnn hace `max_pooling2d(pool_size=[1, ancho_completo])`: colapsa **todo** el eje frecuencial de una vez. La diferencia no es de grado sino de intención: la clase reduce dimensionalidad; musicnn **impone invarianza al pitch como decisión arquitectónica**, y CBMI16 lo respalda con el barrido de $m$ (52,43% con $m=M=40$ contra 59,59% con $m=32$; **+7,2 puntos por permitir que el filtro se deslice en frecuencia**).

**(c) La LSTM no está gratis.** El "Ejemplo 1" exige *"normalize sequence length in minibatch"*. musicnn, al agregar con media y máximo, acepta cualquier duración y `extractor()` lo expone como parámetro `input_length`. Para música, donde las canciones tienen duraciones arbitrarias, eso es una ventaja operativa concreta. **Aunque el propio LBD19 muestra el límite de esta elección**: reemplazar el pooling por atención da +2,61 de PR-AUC en MSD. La agregación media/máximo es simple y robusta, pero deja rendimiento sobre la mesa.

**(d) La receta de la clase es correcta como *default*; el punto de Pons es que hay un *default* mejor cuando conoces el dominio.** Vale enfatizar que Pons nunca dice que el 3×3 esté mal en general — ISMIR18 lo mide y encuentra que sobre entradas cortas (<10 s) *"alcanzaba desempeños equivalentes"*. Lo que dice es más preciso: **con entradas largas el 3×3 se vuelve caro en memoria, y con datasets pequeños desperdicia capacidad.** Ambas condiciones se cumplen en música. Y la conclusión general está enunciada de la forma más útil posible en la conclusión de CBMI16:

> *"Es importante primero entender los datasets de entrenamiento que usan nuestros algoritmos de deep learning. Haciendo eso, los investigadores deberían poder usar ese conocimiento para diseñar arquitecturas que se ajusten mejor al problema. Esto es especialmente relevante para el campo del MIR ya que **se ha señalado que los algoritmos de machine learning están aprendiendo a 'reproducir el ground truth' en lugar de aprender conceptos musicales** [Sturm 2014]. Abordar las arquitecturas de deep learning de una forma musical puede reducir ese riesgo."*

Esa cita —la referencia a los "caballos" de Sturm, sistemas que dan la respuesta correcta por la razón equivocada— es la mejor síntesis del aporte de la línea completa. **No es "usa estos filtros"; es "sepa qué está aprendiendo tu modelo, y diseña la arquitectura para restringirlo a aprender lo que quieres".**

---

## 12. Erratas, matices y cosas que se citan mal

### 12.1. Mapa definitivo: qué afirmación viene de qué documento

Esta es la tabla que hay que consultar antes de citar cualquier cosa de este análisis.

| Afirmación | Fuente real | ¿Está en LBD19 (musicnn)? |
|---|---|---|
| Nombres de los 5 modelos preentrenados | LBD19 | **Sí** |
| Los dos vocabularios de 50 etiquetas | LBD19 (notas al pie) + `configuration.py` | **Sí** |
| Las 5 cifras ROC-AUC/PR-AUC de los modelos | LBD19 | **Sí** |
| Las 2 cifras de la variante con atención | LBD19 | **Sí** |
| Los 5 porcentajes de GTZAN (77,58 / 77,24 / 74,65 / 72,75 / 71,37) | LBD19 | **Sí** |
| Nombres de las features extraíbles (`timbral`, `temporal`, `cnn1`…) | LBD19 + `models.py` | **Sí** |
| MTT = 19k y MSD = 200k canciones de entrenamiento | LBD19 | **Sí** |
| Ejemplos de código de `top_tags` y `extractor` | LBD19 | **Sí** |
| **Formas de filtro concretas (7×38, 7×67, 128×1, 64×1, 32×1)** | **CÓDIGO** (`models.py`) | **No** |
| **Formas de filtro del paper (7×86, 3×86, 1×86, 7×38, 3×38, 1×38; 165×1, 128×1, 64×1, 32×1)** | **ISMIR18, Sección 4** | **No** |
| **SR=16k, 96 mel, hop 256, FFT 512, parche de 3 s** | **CÓDIGO** (`configuration.py`, `FAQs.md`) | **No** |
| **Conteos de parámetros (~0,78M / ~7,97M)** | **Mi cálculo desde `models.py`** | **No** |
| **Waveform vs spectrogram y el cruce entre 500k y 1M** | **ISMIR18, Tabla 1 y Figura 3** | **No** |
| **Todas las tablas de MTT y MSD con baselines** | **ISMIR18, Tablas 2–4** | **No** |
| **Tabla de Ballroom, `Time` / `Frequency` / `Time-Frequency`** | **CBMI16, Tabla I** | **No** |
| **Invarianzas (pitch, loudness, duración, posición espacial)** | **EUSIPCO17, Sección II** | **No** |
| **Resultados de IRMAS y de fonemas de ópera de Pekín** | **EUSIPCO17, Tablas I y II** | **No** |
| **La FCN de 3×3 y sus AUC en MTT/MSD** | **ISMIR16, Tablas 1–5** | **No** |
| **Desbalance: rock 52.944 vs happy 1.257** | **ISMIR16, Sección 5.3** | **No** |
| **El "techo de cristal" por ruido de anotación en MTT** | **ISMIR16, Sección 5.1** | **No** |
| **Justificación de PR-AUC (Davis & Goadrich)** | **ISMIR18, Sección 5.1** | **No** |

### 12.2. La convención de ejes cambia entre papers — esta es la trampa principal

**Este es el error más fácil de cometer al leer la línea de Pons, y explica varias confusiones que circulan.**

| Documento | Convención | Ejemplo del documento | Lectura correcta |
|---|---|---|---|
| **CBMI16** | $(m, n)$ = **(frecuencia, tiempo)** | `Time` usa $(1, 60)$ | 1 banda mel × 60 frames |
| **CBMI16** | ídem | `Frequency` usa $(32, 1)$ | 32 bandas mel × 1 frame |
| **EUSIPCO17** | $m \times n$ = **frecuencia × tiempo** | `50×1`, `70×10`, `100×3` | 50 bandas × 1 frame; 70 bandas × 10 frames |
| **ISMIR18** | **tiempo × frecuencia** (¡invertida!) | `7×86`, `1×86`, `165×1` | 7 frames × 86 bandas; 1 frame × 86 bandas; 165 frames × 1 banda |
| **CÓDIGO** | `kernel_size=[tiempo, frecuencia]` | `[7, 38]`, `[128, 1]` | 7 frames × 38 bandas; 128 frames × 1 banda |

**Cómo desambiguar sin equivocarse:** identifica cuál número puede ser el total de bandas mel. Si el paper usa 96 bandas y ves un 86, ese 86 **es** el eje de frecuencia, esté donde esté escrito. En ISMIR18 los shapes timbrales son `7×86` con 86 ≈ 0,9 × 96 bandas → el 7 es tiempo. En EUSIPCO17 los shapes son `70×10` con entrada de 80 bandas → el 70 es frecuencia.

Consecuencia práctica: **un "filtro 1×86" de ISMIR18 y un "filtro 86×1" de EUSIPCO17 son el mismo objeto** (un frame de ancho, 86 bandas de alto), y ambos son "filtros altos" o "verticales" en la terminología visual. Si alguien te dice que musicnn usa "filtros 165×1 para timbre", está confundiendo las convenciones: los 165×1 son **temporales**, 165 frames de largo.

### 12.3. Erratas dentro de los propios documentos

**(a) LBD19 cita mal el año de su paper base.** La referencia [5] dice *"Jordi Pons, Oriol Nieto, Matthew Prockup, Erik Schmidt, Andreas Ehmann, and Xavier Serra. End-to-end learning for music audio tagging at scale. In **ISMIR, 2017**"*. El paper es de **ISMIR 2018** (arXiv:1711.02520v4 lo confirma, y el propio `FAQs.md` del repositorio da el BibTeX correcto: `booktitle={19th International Society for Music Information Retrieval Conference (ISMIR2018)}, year={2018}`). El origen probable del error: el preprint v1 de arXiv es de noviembre de 2017.

**(b) Diferencias de tamaño de MTT entre fuentes.** ~26k (dataset completo, ISMIR18/EUSIPCO17/ISMIR16 con 25.856) vs ~21k (versión filtrada de SampleCNN) vs 19k (partición de entrenamiento, LBD19). No es una contradicción: son cosas distintas. Pero **hace que los números de MTT de distintos papers no siempre sean comparables**, y ISMIR18 lo advierte explícitamente: el 90,55 de SampleCNN se computó sobre la versión limpia de 21k, y al reproducirlo sobre la de 26k obtienen 88,56 — **casi 2 puntos de diferencia solo por la versión del dataset.**

**(c) MSD: 1M en el nombre, ~240k en la práctica.** ISMIR18: *"aunque el nombre indica que hay 1M de canciones disponibles, los archivos de audio con anotaciones de etiquetas apropiadas (top-50) solo están disponibles para ≈240k previews de 30 s"*. Cualquier afirmación del tipo "musicnn se entrenó con un millón de canciones" es falsa: fueron ~200k de entrenamiento.

**(d) `Timbre CNN` reproducido con parámetros distintos.** ISMIR18 anota al pie de la Tabla 3: *"Reproducido usando 96 bandas mel en lugar de 128 como en [21]"*. Esto explica que el reproducido (89,07) quede por debajo del reportado original (89,30). Diferencia menor pero es exactamente el tipo de detalle que se pierde al copiar tablas.

**(e) El modelo de la librería no es el modelo del paper.** Documentado en detalle en la Sección 4.7. Lo más importante: `MSD_musicnn_big` (88,41 / 30,02, LBD19) es **peor** que el modelo de espectrograma de ISMIR18 (88,75 / 31,24), pese a compartir la configuración de midend/backend. **Ninguna fuente verificada explica la discrepancia.** La hipótesis más plausible es la duración de entrada (15 s en el paper contra ~3 s por defecto en la librería), pero es una hipótesis mía.

### 12.4. Atribuciones incorrectas frecuentes

| Afirmación que se ve por ahí | Estado | Corrección |
|---|---|---|
| "musicnn propone una arquitectura nueva" | **Falso** | Es una librería con pesos preentrenados; la arquitectura es de ISMIR18 y sus raíces en CBMI16/EUSIPCO17 |
| "El paper de musicnn muestra que waveform gana a escala" | **Falso** | Está en ISMIR18. LBD19 no menciona modelos de forma de onda en absoluto |
| "musicnn está publicado en ISMIR 2019" | **Impreciso** | Es la sesión **Late-Breaking/Demo**, y el PDF dice `[Unrefereed]` en cada página |
| "musicnn usa filtros 3×3" | **Falso** para los modelos `musicnn` | Los `MTT_vgg` / `MSD_vgg` **sí** usan 3×3; son los baselines de contraste |
| "musicnn tiene una capa de atención" | **Falso** para los 5 modelos publicados | La variante con atención se menciona en LBD19 como resultado adicional del framework `musicnn-training`; **no es ninguno de los modelos que se descargan** |
| "musicnn se entrenó con el Million Song Dataset completo (1M)" | **Falso** | ~200k canciones de entrenamiento |
| "musicnn incluye RNN/LSTM" | **Falso** | No hay ninguna capa recurrente; la agregación temporal es media + máximo |
| "musicnn sirve para transcripción musical" | **Falso** | El max-pool sobre todo el eje frecuencial destruye la información de pitch por diseño |
| "El frontend de musicnn tiene 6 formas timbrales" | **Falso para el código publicado** | ISMIR18 describe 6; `models.py` implementa **2** (7×38 y 7×67) |
| "musicnn procesa a 44,1 kHz porque es música" | **Falso** | `SR = 16000`; techo de 8 kHz |
| "El repositorio está mantenido" | **Falso** | Una sola versión en PyPI (0.1.0, agosto 2019), `Development Status :: 3 - Alpha`, `numpy<1.17` |

### 12.5. Cifras que NO pude verificar y por lo tanto no afirmo

Por transparencia, lo que quedó sin verificación contra PDF en este análisis:

- **La arquitectura de la variante con atención de LBD19.** El abstract da dos pares de cifras y remite a documentación online. No hay descripción de la capa en ninguna fuente verificada. (Contextualmente corresponde a la línea de trabajo de Minz Won et al. sobre *tagging* con auto-atención, pero **no lo verifiqué**.)
- **Los conteos de parámetros de los modelos `vgg` de la librería.** Los calculé aproximadamente (~0,6 M para 128 filtros por capa) pero las formas exactas de los mapas dependen de detalles de padding de `max_pooling2d` que no verifiqué ejecutando el grafo. **No los reporto como cifra.**
- **Los resultados de trabajos posteriores sobre MTT** (short-chunk CNN, ResNets, AST, MERT, CLAP). No descargué esos PDFs. Todas las menciones de las Secciones 9.7 y 10.2 son cualitativas y están marcadas como no verificadas.
- **La disponibilidad de los pesos de musicnn vía `essentia-tensorflow`.** Marcado como conocimiento de contexto en la Sección 7.3.
- **La descripción del audio fingerprinting** de la Sección 11.1. Marcado como conocimiento de contexto.
- **Los detalles del entrenamiento de los modelos publicados** (épocas, learning rate, aumentación, particiones exactas). ISMIR18 da los suyos (Adam, lr inicial 0,001, dropout 0,5 antes de cada densa, ReLU, entropía cruzada, parches de 15 s) pero **no hay garantía de que `musicnn-training` haya usado los mismos**, y no revisé ese repositorio.

### 12.6. La cita correcta

Del `FAQs.md` del repositorio, que pide citar **dos** trabajos, no uno:

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

El propio autor sitúa el orden de importancia: **ISMIR18 primero, LBD19 después.** Si en la clase 39 se cita "MusiCNN, Pons y Serra 2019" como referencia de arquitectura, la referencia técnicamente correcta es **Pons et al., ISMIR 2018**; el documento de 2019 es el anuncio de la librería.

---

## Apéndice: fuentes y ubicación de archivos

| Documento | Ubicación / obtención | Verificado |
|---|---|---|
| **LBD19** — Pons y Serra, musicnn, ISMIR LBD 2019 | `clase_39/papers/Pons-MusiCNN-2019.pdf` y `.txt` (arXiv:1909.06654v1) | Sí, íntegro (2 pp.) |
| **ISMIR18** — Pons et al., End-to-end learning at scale | Descargado de `arxiv.org/pdf/1711.02520` (v4, 15-jun-2018) | Sí, íntegro |
| **CBMI16** — Pons, Lidy y Serra, Musically motivated CNNs | Descargado de `jordipons.me/media/CBMI16.pdf`. **No está en arXiv** | Sí, íntegro |
| **EUSIPCO17** — Pons et al., Timbre Analysis with CNNs | Descargado de `arxiv.org/pdf/1703.06697` (v2, 2-jun-2017) | Sí, íntegro |
| **ISMIR16** — Choi, Fazekas y Sandler, Automatic tagging with deep CNNs | Descargado de `arxiv.org/pdf/1606.00298` (v1, 1-jun-2016) | Sí, íntegro |
| **CÓDIGO** — `musicnn` | `raw.githubusercontent.com/jordipons/musicnn/master/`: `musicnn/models.py`, `musicnn/configuration.py`, `musicnn/extractor.py`, `musicnn/tagger.py`, `setup.py`, `FAQs.md` | Sí |
| **PyPI** | `pypi.org/pypi/musicnn/json` | Sí |
| **CLASE39** | PDF de la clase, texto extraído | Sí |

**Nota sobre las descargas fallidas.** Dos de los identificadores de arXiv indicados en el encargo original **no corresponden** a los papers pedidos:

- **`1606.00298`** no es "Experimenting with musically motivated CNNs" de Pons, Lidy y Serra. Es **Choi, Fazekas y Sandler, "Automatic tagging using deep convolutional neural networks", ISMIR 2016** — que resultó igualmente valioso, porque es precisamente el paper `[2]` que sustenta los *baselines* `vgg` de la librería. El paper de Pons-Lidy-Serra **no está en arXiv**; lo obtuve del sitio personal del autor (`jordipons.me/media/CBMI16.pdf`).
- **`1608.08056`** es "Bayesian nonparametric forecasting of monotonic functional time series" de Canale y Ruggiero (stat.AP), sin relación alguna con audio. El paper de Pons y Serra, "Designing efficient architectures for modeling temporal features with CNNs" (ICASSP 2017), **no lo pude localizar**: no está en arXiv bajo ese identificador ni encontré una URL pública funcional en `jordipons.me`. **Toda referencia a ese trabajo en este análisis proviene de cómo lo citan y resumen ISMIR18 (referencia [20]) y EUSIPCO17 (referencia [1]), no del documento original.** Concretamente, las formas de filtro temporal `165×1, 128×1, 64×1, 32×1` están tomadas de la descripción que hace ISMIR18 en su Sección 4.

Como sustituto, descargué **EUSIPCO17** (arXiv:1703.06697), que resultó ser la pieza teórica más importante de las tres para el argumento de la Sección 3.
