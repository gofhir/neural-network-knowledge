# FSD50K: An Open Dataset of Human-Labeled Sound Events — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *FSD50K: An Open Dataset of Human-Labeled Sound Events*.
- **Autores:** Eduardo Fonseca, Xavier Favory, Jordi Pons, Frederic Font y Xavier Serra. Todos en el **Music Technology Group (MTG), Universitat Pompeu Fabra**, Barcelona.
- **Venue:** *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, volumen 30, 2022.
- **Preprint / año:** arXiv:2010.00475 (primera versión en 2020; versión revisada en 2022). Suele citarse como "Fonseca et al., 2020".
- **Recursos:** el dataset se descarga desde **Zenodo** (DOI 10.5281/zenodo.4060432); el código de los *baselines* está en GitHub; existe además un *companion site* en el Freesound Annotator para explorar el audio y reportar errores de etiquetado.

FSD50K (Freesound Dataset 50k) es un dataset abierto de **eventos sonoros etiquetados por humanos** que contiene **51.197 clips de audio**, con un total de **más de 100 horas** (108,3 h exactas), anotados de forma manual con **200 clases** extraídas de la **AudioSet Ontology**. Es un dataset **multi-etiqueta** (cada clip puede llevar una o más clases) y con **etiquetas débiles** (asignadas a nivel de clip, sin marcas de inicio/fin). Los clips provienen de **Freesound** —el repositorio colaborativo de audio del propio MTG— y están cubiertos por **licencias Creative Commons (CC)**, lo que permite redistribuir libremente el dataset **incluyendo las formas de onda**. Los autores lo describen como el **mayor dataset totalmente abierto de eventos sonoros etiquetados por humanos**, y el segundo más grande después de AudioSet.

La tesis del paper no es solo "aquí hay un dataset grande". Es una tesis sobre **disponibilidad y reproducibilidad**: AudioSet, el gran benchmark de la disciplina, **no es un dataset abierto**, porque se distribuye como *features* precalculados (no como audio) y porque descargar el audio original de YouTube tropieza con los Términos de Servicio de la plataforma y con la desaparición gradual de los videos (*link rot*). FSD50K nace explícitamente para llenar ese vacío: un benchmark **abierto, estable y redistribuible** para *sound event recognition* (SER).

Para la **Clase 37 (Datasets y Herramientas para Audio)** este es exactamente el contraste que la clase pone sobre la mesa: cuando eliges un dataset de audio, la primera pregunta pragmática es **"¿te dan el audio, o solo un enlace de YouTube que se cae?"**. FSD50K es la respuesta del lado "te dan el audio"; AudioSet es el arquetipo del lado "solo un enlace". Este documento desarrolla ese eje sin descuidar el resto: cómo se construyó el dataset, qué tan ruidosas son sus etiquetas y qué rinden sus *baselines*.

## 2. Contexto: por qué AudioSet no basta como benchmark abierto

El reconocimiento de eventos sonoros (SER) consiste en identificar automáticamente qué sonidos ocurren en un audio, asignándoles etiquetas dentro de un vocabulario objetivo. Es un campo con aplicaciones en **salud**, planificación de sonido urbano, monitoreo bioacústico, vigilancia y control de ruido industrial, impulsado en la última década por el *DCASE Challenge*. Al igual que en visión por computador, donde ImageNet catalizó los avances, en audio **los datasets son un insumo crítico** para los métodos de *deep learning*, que son hambrientos de datos.

En 2017, **AudioSet** transformó el campo: ≈2,1 millones de clips de audio etiquetados manualmente con **527 clases** de la AudioSet Ontology. Su tamaño, cobertura y diversidad no tenían precedente. Pero, según los autores de FSD50K, AudioSet arrastra un problema mayor: **no es un dataset abierto**. Los clips son fragmentos de videos de YouTube, que **no son libremente redistribuibles** por los Términos de Servicio de la plataforma. Por eso la *release* oficial de AudioSet no consiste en formas de onda, sino en **características de audio precalculadas** con un modelo preentrenado a una resolución temporal de 960 ms. Esto limita fuertemente la flexibilidad y la adopción de muchos métodos de SER: quien quiera trabajar con la señal cruda (por ejemplo, aprender representaciones directamente desde la forma de onda) simplemente no puede a partir de la *release* oficial.

La alternativa —bajar el audio directamente de los videos originales de YouTube— tiene dos costos que el paper documenta con números concretos:

1. **El *link rot* (podredumbre de enlaces).** Los videos van desapareciendo: se borran videos o cuentas, hay problemas de privacidad, reclamos de copyright, o disponibilidad dependiente del país. En un intento de descarga (datos del 11 de mayo de 2020) los autores lograron bajar **18.205 de 20.371 segmentos de evaluación** y **19.862 de 22.160 segmentos del *balanced train*** — pérdidas de **10,6 %** y **10,4 %** respectivamente. Peor aún, esa cantidad **decrece con el tiempo y de forma desigual** entre particiones, lo que erosiona la comparabilidad de AudioSet como *benchmark*: dos equipos que "usan AudioSet" en fechas distintas literalmente no evalúan sobre el mismo conjunto.
2. **La carga logística** de descargar masivamente datos desde una fuente no oficial.

Los datasets posteriores a AudioSet (FSDnoisy18k, FSDKaggle2019, SONYC-UST, etc.) resolvieron parcialmente la apertura pero son **específicos de tarea o de dominio**, con vocabularios de pocas decenas de clases. VGGSound, grande y de vocabulario amplio (≈200k clips, 300 clases), **hereda los problemas intrínsecos de estar basado en YouTube**. Así, el campo de SER quedó rezagado respecto de visión por computador en cuanto a **disponibilidad de datasets abiertos de propósito general**. FSD50K se propone cerrar esa brecha.

## 3. Contribución central

Las contribuciones del paper son cuatro:

1. Un **dataset abierto etiquetado por humanos**, diseñado principalmente para el desarrollo y evaluación de sistemas de **clasificación multi-etiqueta de eventos sonoros**, pero que habilita también otras tareas de investigación en sonido.
2. Una **descripción detallada del proceso de creación**, adaptado a las particularidades de los datos de Freesound, incluyendo los desafíos encontrados y las soluciones adoptadas — una práctica de documentación que en visión es habitual y que en audio recién se estaba proponiendo.
3. Una **caracterización exhaustiva del dataset** junto con una discusión de sus limitaciones y de los factores clave para su uso informado por los datos (incluida una **estimación cuantitativa del ruido de etiquetas**).
4. Un conjunto de **experimentos *baseline*** de clasificación de eventos sonoros, más un análisis de los factores a considerar al **particionar audio de Freesound** para SER (el "efecto uploader").

Por encima de todo, el aporte que la Clase 37 subraya es que FSD50K se distribuye **como audio real bajo licencias Creative Commons**, resolviendo de raíz el problema de disponibilidad de AudioSet.

## 4. Método: cómo se construyó FSD50K

El proceso parte de tres piezas —Freesound, la AudioSet Ontology y el Freesound Annotator— y encadena varias etapas que progresivamente filtran clips y clases.

### 4.1. Los tres cimientos

- **Freesound.** Sitio colaborativo de intercambio de clips de audio (más de 10 millones de usuarios registrados y más de 500.000 clips), creado y mantenido por el MTG. Cubre desde muestras musicales hasta sonidos ambientales y efectos. Los usuarios aportan metadatos (título, al menos tres *tags*, descripciones textuales) que resultan clave para el proceso. Es muy **heterogéneo** en origen, equipo de grabación y condiciones acústicas. Y, crucialmente, **todo su contenido está bajo licencias CC**, lo que permite distribución y reutilización.
- **AudioSet Ontology.** Una jerarquía de **632 clases** de eventos sonoros con una profundidad máxima de 6 niveles (AudioSet usa un subconjunto de 527). Cubre sonidos humanos, animales, naturales, musicales y misceláneos. Se eligió por ser el vocabulario más completo de sonidos cotidianos disponible.
- **Freesound Annotator (FSA).** Sitio de código abierto para la creación colaborativa de datasets de audio abiertos, con herramientas de anotación y de control de calidad.

### 4.2. Nominación de etiquetas candidatas

Se pobló automáticamente cada clase de la ontología con clips candidatos de Freesound **haciendo *matching* entre los *tags* provistos por los usuarios y un conjunto de *keywords* asociado a cada clase** (por ejemplo, para *Meow*: "meow", "meowing", "mew", "miaow", "miaou"). Se usó el **algoritmo de *stemming* de Porter** para normalizar términos y la API de Freesound. Este mapeo asoció más de 300.000 clips; tras filtrar los de más de 90 s quedaron **268.261 clips** con un promedio de **2,62 etiquetas candidatas**. El método es rápido y no requiere entrenar clasificadores, pero **induce errores** porque depende de la ambigüedad de las clases y, sobre todo, de cómo los usuarios eligen sus *tags*.

### 4.3. Tarea de validación (control de calidad, primera pasada)

Las etiquetas candidatas se validaron **manualmente**. A cada anotador se le presentaban clips y se le preguntaba: *¿está `<clase>` presente en los siguientes sonidos?*. Tras una **evaluación interna de calidad (IQA)** con 11 voluntarios, el prototipo final incorporó varios mecanismos de calidad que vale la pena enumerar, porque son el corazón de la "discusión del ruido de etiquetas":

- **FAQs por clase** para homogeneizar el criterio de los anotadores ante descripciones ambiguas de la ontología.
- **Distinción PP/PNP:** la respuesta "Present" se dividió en **"Present and predominant" (PP)** —el sonido está presente y es predominante, sin otros sonidos salvo ruido de fondo leve— y **"Present but not predominant" (PNP)** —presente pero acompañado de otros sonidos salientes o ruido fuerte. Esto permite separar un subconjunto de sonidos limpios y aislados de otro en condiciones más adversas.
- **Clips de verificación:** insertados periódicamente; si un anotador falla uno, se descartan sus respuestas de ese lapso (práctica habitual en *crowdsourcing*).
- **Acuerdo inter-anotador:** cada etiqueta candidata se presenta a varios anotadores **hasta que dos coinciden** en el tipo de respuesta; recién entonces se considera *ground truth*.
- **Espectrogramas** en la interfaz (más informativos que la mera forma de onda) y **normalización de sonoridad** según la recomendación **EBU R-128**.

Se descartaron las clases con más de 75 % de respuestas "Not Present", con muy pocos candidatos o demasiado ambiguas, dejando **395 clases** (una reducción de ≈35 %), y se descartaron los clips de más de 30 s. La **campaña de anotación** combinó *crowdsourcing* (clases fáciles y medias) con **anotadores contratados** (las 218 clases difíciles): en total participaron **más de 350 anotadores**, incluidos seis contratados y los tres primeros autores. El resultado fue **51.684 clips válidos** (con al menos una etiqueta "Present") y **59.981 etiquetas "Present"**.

### 4.4. División de los datos (*data split*)

Aquí aparece una decisión metodológica fina y muy citada. Al partir audio de Freesound hay que evitar la **contaminación**: como un mismo *uploader* puede subir muchos clips que comparten fuente, ubicación o equipo de grabación, si unos van a entrenamiento y otros a evaluación la similitud infla artificialmente el desempeño (el paper lo llama **"efecto uploader"**, en analogía con el "album effect" o "artist effect" de la recuperación musical). Los criterios de división fueron:

- **No divisibilidad de *uploaders*:** todo el contenido de un *uploader* va entero a *development* o entero a *evaluation*.
- ***Uploaders* pequeños para evaluación:** garantizan mayor diversidad de fuentes y condiciones, y tienden a subir clips más largos (más representativos del mundo real).
- Distribución de clases gruesa (no fina) en esta etapa.
- Foco en los **113 nodos hoja con más de 100 clips**.

Se ordenaron los *uploaders* con un puntaje que penaliza la concentración en una sola clase y premia la diversidad:

$$\text{score}_u = n\_labels^{\max}_u + \frac{1}{K_u}\sum_{k=1}^{K_u} n\_labels_{u c_k}$$

donde $n\_labels^{\max}_u$ es el máximo de etiquetas que $u$ aporta en cualquier clase, y $K_u$ el número de clases que toca. Procesando los **7229 uploaders**, se asignaron **2794 al conjunto de evaluación** (11.466 clips).

### 4.5. Tarea de refinamiento (etiquetado exhaustivo del *eval*)

El *split* del conjunto de evaluación se **re-anotó exhaustivamente**: se revisaron las etiquetas existentes y se **agregaron las etiquetas "Present" faltantes**, usando una herramienta que permite explorar interactivamente la jerarquía de la ontología. Esta pasada la hicieron 4 de los 6 anotadores contratados, ya expertos. Como resultado, **cada etiqueta del *eval* fue verificada por entre dos y cinco anotadores, incluido al menos un experto**. La consecuencia clave: en el conjunto de evaluación, **la ausencia de etiqueta significa ausencia de evento sonoro** (salvo error humano), lo que lo vuelve confiable para *benchmarking*. El conjunto de *development*, en cambio, queda como **"correcto pero potencialmente incompleto" (CpI)**.

## 5. Estadísticas y *baselines*

### 5.1. Estadísticas principales

FSD50K contiene **51.197 clips** repartidos de forma desigual en **200 clases** (144 nodos hoja + 56 nodos intermedios). Del total, **40.966 clips (80 %) van a *development*** y **10.231 (20 %) a *evaluation***. Otras cifras del dataset:

| Estadística | Total | Dev | Eval |
|---|---|---|---|
| Clips | 51.197 | 40.966 (80 %) | 10.231 (20 %) |
| Etiquetas (sin propagar) | 62.657 | 45.607 | 17.050 |
| Prom. etiquetas/clip | 1,22 | 1,11 | 1,67 |
| Etiquetas (propagadas) | 152.867 | 114.271 | 38.596 |
| Duración | 108,3 h | 80,4 h | 27,9 h |
| Prom. duración/clip | 7,6 s | 7,1 s | 9,8 s |
| Uploaders | 7225 | 4936 | 2289 |

Los clips duran entre **0,3 y 30 s** (longitud variable, un rasgo distintivo). De las 108,3 h, **31,5 h están etiquetadas exhaustivamente** (sobre todo *eval* y *val*). El contenido se subió a Freesound desde su lanzamiento en 2005 hasta comienzos de 2019, aportado por 7225 usuarios. Todos los clips se entregan como **audio PCM sin comprimir de 16 bits, 44,1 kHz, mono**.

### 5.2. Licencias

El dataset como conjunto se libera bajo **CC-BY**, pero **cada clip conserva su propia licencia específica**: CC0, CC-BY, CC-BY-NC o CC Sampling+. Las dos más permisivas, **CC0 y CC-BY, suman el 84,7 %** del dataset. Este detalle importa para uso clínico o comercial: el 15,3 % restante incluye la restricción *NonCommercial*, de modo que un usuario que planee un producto debe filtrar por licencia.

### 5.3. Baselines

El *pipeline* de referencia usa **log-mel espectrogramas de 96 bandas**, parches tiempo-frecuencia de 1 s (forma $101\times96$), optimizador Adam y pérdida de **entropía cruzada binaria** (adecuada para multi-etiqueta), implementado en TensorFlow. Se evalúa con tres métricas independientes del umbral de decisión: **mAP** (mean Average Precision), **$d'$** (d-prime, transformación monótona del ROC-AUC) y **lωlrap** (*label-weighted label-ranking average precision*, la métrica de DCASE 2019 Task 2). Resultados sobre *eval* (promedio de 3 corridas):

| Modelo | Pesos | mAP | $d'$ | lωlrap |
|---|---|---|---|---|
| CRNN | 0,96 M | 0,417 | 2,068 | **0,519** |
| **VGG-like** | 0,27 M | **0,434** | **2,167** | 0,514 |
| ResNet-18 | 11,3 M | 0,373 | 1,883 | 0,465 |
| DenseNet-121 | 12,5 M | 0,425 | 2,112 | 0,505 |

El hallazgo interesante es que **el mejor modelo es el VGG-like**, el más liviano (0,27 M de pesos) y menos moderno, por encima de ResNet-18 y DenseNet-121, arquitecturas mucho más pesadas tomadas "tal cual" de visión por computador. ResNet-18 es el peor. La lectura de los autores: **a esta escala de datos, modelos pequeños con diseño informado por audio superan a arquitecturas grandes de visión sin ajuste** — un contraste con AudioSet, donde las ResNets sí rinden en el estado del arte, presumiblemente por su mucho mayor volumen de datos de entrenamiento.

## 6. El eje de disponibilidad y licencia: CC vs. enlaces de YouTube

Este es el corazón del paper y el punto que la Clase 37 quiere que el estudiante internalice. La Tabla V del artículo compara ambos datasets punto por punto:

| Propiedad | FSD50K | AudioSet |
|---|---|---|
| Clases | 200 | 527 |
| **Contenido distribuido** | **Forma de onda** | **Features precalculados** |
| Clips dev | 40.966 | ≈2 M |
| Clips eval | 10.231 | 20.383 |
| Longitud de clip | 0,3–30 s | ≈10 s |
| Etiquetado dev | CpI | CpI |
| **Etiquetado eval** | **Exhaustivo** | **CpI** |
| **Fuente** | **Audio de Freesound (CC)** | **Video de YouTube** |
| SNR P.563 (media, mediana) | (26, 25) dB | (14, 10) dB |

La diferencia decisiva está en las dos filas resaltadas: **qué se distribuye y de dónde viene**. AudioSet entrega *features* a 960 ms bajo CC-BY-4.0 (los *features*, no el audio); el audio original vive en YouTube, es **inestable** (*link rot*) y su uso puede chocar con políticas de copyright. FSD50K entrega **las formas de onda completas** bajo licencias CC decididas por los usuarios de Freesound, descargables como archivos ZIP desde una página de Zenodo **estable y versionada**.

En términos de la pregunta de la clase —"¿te dan el audio o solo un enlace que se cae?"— FSD50K **te da el audio**, y con permiso explícito para redistribuirlo. Esto tiene tres consecuencias prácticas:

1. **Reproducibilidad exacta:** dos equipos que descargan FSD50K en fechas distintas obtienen el mismo conjunto de clips; en AudioSet, no.
2. **Flexibilidad metodológica:** al tener la forma de onda, se puede aprender directamente desde el audio crudo, recalcular cualquier representación (espectrogramas, MFCC, embeddings de otro modelo) o hacer *data augmentation* sobre la señal — todo imposible desde *features* congelados.
3. **Redistribución legal:** se pueden armar y compartir subconjuntos, algo que las licencias CC permiten y los Términos de Servicio de YouTube prohíben.

Como bonus, la Tabla V sugiere que el audio de Freesound tiene **mejor SNR** (media 26 dB vs. 14 dB): al grabarse con la intención de capturar sonido —a veces con equipo semiprofesional— tiende a ser más limpio que el audio incidental de videos de YouTube. Los autores son cuidadosos: la métrica P.563 está diseñada para voz humana, así que la usan solo como indicación gruesa, y concluyen que ambos datasets son **recursos complementarios** (AudioSet aporta más volumen y más diversidad de dispositivos y condiciones del mundo real).

## 7. El ruido de etiquetas, cuantificado

Un mérito del paper es **medir** su propio ruido de etiquetas en lugar de solo mencionarlo. Aprovechando la tarea de refinamiento (que re-anotó exhaustivamente un lote destinado mayormente a *eval*), cuantificaron sobre **11.847 clips** (13.681 etiquetas de entrada):

- **Etiquetas faltantes (falsos negativos):** **6030 clips (50,9 %)** recibieron al menos una etiqueta adicional, es decir, tenían material sin etiquetar. Esto afecta sobre todo al *development set*, cuyas etiquetas provienen de validar candidatos nominados por *tags*: si un evento no fue nominado, queda sin etiqueta.
- **Etiquetas incorrectas (falsos positivos):** de las 13.681 etiquetas de entrada, **773 (5,7 %) fueron rechazadas**, lo que implica que **el 94,3 % fue verificado como correcto**.

En síntesis: en el *dev set* las etiquetas son **mayormente correctas (≈94,3 %) pero potencialmente incompletas** (CpI); en el *eval set*, gracias al refinamiento exhaustivo, son **correctas y completas** salvo error humano. Esta honestidad —comparable a las estimaciones de error de AudioSet (>50 % de error en ≈18 % de las clases) o de ImageNet (≥100.000 imágenes mal etiquetadas)— es justamente lo que permite un uso "informado por los datos".

## 8. Impacto

FSD50K se consolidó como **el benchmark abierto de referencia para clasificación de eventos sonoros de gran vocabulario**. Habilita aprendizaje de representaciones desde la forma de onda, métodos de mitigación de ruido de etiquetas (aprovechando el etiquetado no exhaustivo del *dev*), enfoques multimodales audio+texto (usando los *tags* y descripciones de Freesound), clasificación jerárquica *ontology-aware*, y —por compartir vocabulario con AudioSet— **adaptación de dominio y evaluación cruzada** entre ambos. Subconjuntos de estos datos alimentaron varios *DCASE Challenges* (2018–2020). Gracias a su apertura, es un recurso estándar para evaluar arquitecturas y métodos de aprendizaje autosupervisado en audio.

## 9. Limitaciones

Los autores son explícitos sobre los límites del dataset:

- **Ruido de etiquetas residual:** persisten etiquetas "Present" faltantes (sobre todo en *dev*) e incorrectas ocasionales; ambas son *class-conditional* (más frecuentes en clases ambiguas).
- **Desbalance de datos:** por distribución no uniforme de clases, longitud variable de clips y por la propia jerarquía de la ontología (los nodos intermedios acumulan mucho más).
- **Sesgo en el *development set*:** al reservar los *uploaders* pequeños para *eval*, algunas clases del *dev* quedan dominadas por pocos *uploaders* grandes (p. ej. *Trumpet*), lo que podría inducir un sesgo aprendible.
- **Falta de especificidad en el vocabulario:** varios nodos hoja con pocos datos (p. ej. *Blender*, *Toothbrush*) se fusionaron con su nodo padre (*Domestic sounds, home sounds*), reduciendo el detalle.
- **Etiquetas débiles y longitud variable:** la debilidad de la etiqueta varía con la duración —clips cortos con anotación PP son casi *strong labels*, clips largos son mucho más débiles (*label density noise*)—, lo que impone decisiones de diseño (parches de longitud fija, *multiple-instance learning*).
- **Grabaciones no siempre "en el mundo real":** parte de Freesound son grabaciones tipo *foley* o generadas a propósito, con posible *acoustic mismatch* respecto de condiciones adversas.

## 10. Conexión con la Clase 37 (Datasets y Herramientas para Audio)

La Clase 37 presenta FSD50K junto a AudioSet como los dos grandes *benchmarks* de eventos sonoros, pero usa el par para enseñar un **criterio de selección de dataset que va más allá del tamaño: la disponibilidad**. La pregunta guía de la clase —"¿te dan el audio (FSD50K) o solo un enlace de YouTube que se cae (AudioSet)?"— resume el aporte del paper. El estudiante debería salir de la clase con tres ideas ancladas en este documento:

1. **Tamaño no es todo.** AudioSet es ≈40× más grande, pero su *release* como *features* congelados y su *link rot* (10,6 % de pérdida en *eval*, creciente en el tiempo) lo hacen **frágil como benchmark reproducible**. FSD50K, más chico, es **estable, redistribuible y flexible** porque entrega la forma de onda bajo licencia CC.
2. **La licencia es un criterio técnico, no solo legal.** Que el 84,7 % de FSD50K sea CC0/CC-BY define qué puedes hacer (redistribuir, derivar, usar comercialmente). En audio, la licencia condiciona directamente la reproducibilidad de un experimento y la posibilidad de desplegar un producto.
3. **La calidad del *eval* importa más que la del *train*.** El diseño de FSD50K prioriza un conjunto de evaluación **exhaustivamente etiquetado** (ausencia de etiqueta = ausencia de evento) y libre de contaminación entre *uploaders* — el "efecto uploader" es una lección transversal sobre cómo particionar datos para no inflar métricas.

**Enlaces internos sugeridos:**

- Clase: [/clases/clase-37](/clases/clase-37) — Datasets y Herramientas para Audio.
- Fundamento transversal: [/fundamentos/reconocimiento-de-eventos-sonoros](/fundamentos/reconocimiento-de-eventos-sonoros) — SER, etiquetas débiles vs. fuertes, multi-etiqueta.
- Paper de contraste: el propio AudioSet (Gemmeke et al., 2017), el *benchmark* cuyo problema de disponibilidad motiva FSD50K.

---

**Nota sobre relevancia para salud.** En investigación clínica el eje "disponibilidad + licencia clara" no es un detalle burocrático: es un requisito de reproducibilidad. Un modelo de detección de eventos sonoros clínicos —tos, sibilancias, ronquidos, sonidos respiratorios o de alarma en la UCI— solo es auditable y comparable si el dataset de referencia puede **redistribuirse íntegramente y re-descargarse idéntico** por revisores, reguladores u otros equipos; un *benchmark* atado a enlaces de YouTube que se caen (o a *features* congelados que impiden recalcular representaciones) rompe esa cadena de reproducibilidad justo donde la evidencia clínica más la exige. FSD50K muestra el estándar deseable: audio real, licencias explícitas por clip (con el 84,7 % en CC0/CC-BY, que habilita incluso uso comercial), versionado estable en Zenodo y una estimación honesta del ruido de etiquetas (94,3 % de corrección, 50,9 % de clips con etiquetas faltantes). Para quien construye datasets de audio en salud, esa combinación —redistribuibilidad, trazabilidad de licencia y transparencia sobre la calidad de las anotaciones— es precisamente lo que separa un recurso de investigación citable de una demo irreproducible.
