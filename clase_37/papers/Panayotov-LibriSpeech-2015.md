# LibriSpeech: An ASR Corpus Based on Public Domain Audio Books — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Librispeech: An ASR Corpus Based on Public Domain Audio Books*.
- **Autores:** Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev Khudanpur.
- **Afiliación:** Center for Language and Speech Processing & Human Language Technology Center of Excellence, The Johns Hopkins University, Baltimore. (Daniel Povey es el creador de **Kaldi**, el toolkit de ASR con el que se construyó y se distribuye el corpus.)
- **Venue:** *IEEE International Conference on Acoustics, Speech and Signal Processing* (ICASSP 2015).
- **Licencia y distribución:** el corpus es de libre descarga bajo la muy permisiva licencia **CC BY 4.0**, alojado en OpenSLR (`openslr.org/12`).
- **Términos índice del paper:** Speech Recognition, Corpus, LibriVox.

LibriSpeech es un corpus de **habla en inglés leída** pensado para entrenar y evaluar sistemas de reconocimiento automático del habla (ASR, *Automatic Speech Recognition*). Contiene **1000 horas de habla muestreadas a 16 kHz**, derivadas de audiolibros del proyecto **LibriVox** cuyos textos provienen mayormente de **Project Gutenberg**, ambos de dominio público. El paper no solo libera el audio segmentado y transcrito: también publica datos de entrenamiento de modelos de lenguaje, modelos de lenguaje pre-construidos y los *scripts* de Kaldi que permiten reproducir de punta a punta los sistemas reportados.

El resultado experimental que ancló su credibilidad es contraintuitivo y potente: modelos acústicos entrenados sobre LibriSpeech logran **menor tasa de error en los conjuntos de prueba del Wall Street Journal (WSJ)** que modelos entrenados sobre el propio WSJ. Es decir, la escala del corpus (1000 horas frente a las 82 horas del subconjunto `si-284` de WSJ) **compensa con creces** la diferencia de dominio y de calidad de grabación. Este hallazgo, sumado a la licencia libre y a los *splits* reproducibles, convirtió a LibriSpeech en **el** benchmark de ASR en inglés durante la década siguiente y en la base de datos sobre la que se preentrenaron y evaluaron los grandes modelos autosupervisados (wav2vec 2.0) y multitarea (Whisper) que dominan hoy el campo.

Para la **Clase 37 (Datasets y Herramientas para Audio)**, LibriSpeech es el ejemplo canónico de "LibriSpeech: habla transcrita" y de la distribución en **FLAC** (compresión sin pérdida): ilustra qué significa un corpus de referencia, por qué la reproducibilidad de los *splits* importa y cómo un dataset bien documentado y libre puede reorientar todo un subcampo.

## 2. Contexto: la necesidad de un corpus de ASR grande, libre y reproducible

A mediados de la década de 2010, el reconocimiento del habla ya se apoyaba en modelos estadísticos entrenados con datos, pero los corpora de referencia arrastraban un problema estructural: **las licencias**. Los dos benchmarks históricos del ASR en inglés eran de pago y de circulación restringida:

- **Wall Street Journal (WSJ)** [Paul y Baker, 1992]: habla leída de artículos periodísticos, distribuido por el *Linguistic Data Consortium* (LDC) bajo licencia comercial. Su subconjunto de entrenamiento estándar `si-284` contiene solo 82 horas.
- **Switchboard** (habla telefónica conversacional): también bajo licencia LDC de pago.

Estos corpora eran valiosos pero **caros e imposibles de redistribuir libremente**, lo que dificultaba la comparación abierta entre grupos de investigación y excluía a quien no pudiera pagar la licencia. Al mismo tiempo, la explosión de contenido multimedia en internet hacía factible **recolectar datos automáticamente**, sobre todo cuando la fuente ya está organizada en colecciones curadas y legibles por máquina. Dos proyectos de voluntariado ofrecían exactamente eso:

- **LibriVox**: un esfuerzo colaborativo que, al momento del paper, había producido cerca de **8000 audiolibros de dominio público**, la mayoría en inglés.
- **Project Gutenberg**: el repositorio de textos de dominio público sobre los que se basan esas grabaciones.

Existía un antecedente parcial: **VoxForge**, otro esfuerzo de voluntariado, incluía algo de audio de LibriVox, pero era mucho más pequeño (alrededor de 100 horas de inglés) y sufría **fuertes desbalances de género y de duración por hablante**. No había, entonces, ningún corpus de habla leída en inglés, libremente disponible y de la escala necesaria para entrenar y probar sistemas de ASR modernos. Ese es exactamente el vacío que LibriSpeech vino a llenar.

## 3. Contribución central

La contribución de LibriSpeech es doble y conviene separarla con claridad:

1. **Un corpus.** 1000 horas de habla leída en inglés, a 16 kHz, segmentada en enunciados cortos con sus transcripciones, libre bajo CC BY 4.0, acompañada de datos y modelos de lenguaje pre-construidos.
2. **Un protocolo reproducible.** El paper no libera solo los datos, sino la **receta completa** para producirlos y para usarlos: el procedimiento de alineación de audio largo, el criterio de división por calidad (*clean* vs. *other*), los *splits* estándar de entrenamiento/desarrollo/prueba, y los *scripts* de Kaldi que permiten a cualquiera replicar los sistemas base. Esto transforma un dataset en un **benchmark**: cifras comparables entre laboratorios porque todos entrenan y evalúan sobre exactamente las mismas particiones.

La combinación de ambos —dominio público + escala + reproducibilidad— es lo que le dio a LibriSpeech su longevidad. Un dataset grande pero con licencia restrictiva no habría podido convertirse en estándar universal; un dataset libre pero pequeño (como VoxForge) no habría bastado para los modelos de la época.

## 4. Método: de audiolibros crudos a enunciados alineados

El desafío técnico central es que los modelos acústicos esperan **enunciados relativamente cortos** (hasta unas pocas decenas de segundos) con su texto correspondiente, mientras que un audiolibro es una grabación continua de horas. Hay que **alinear** el audio con el texto del libro, **segmentarlo** y, sobre todo, **filtrar** los tramos donde el audio no corresponde exactamente al texto alineado.

### 4.1. Materia prima: LibriVox + Project Gutenberg

Para seleccionar las grabaciones se usa la **API de LibriVox** para recolectar información sobre lectores, proyectos de audiolibro y capítulos. Las URLs de los archivos de audio y de los textos de referencia se obtienen cruzando esa información con los metadatos del *Internet Archive* y los archivos RDF/XML de Project Gutenberg. Para una pequeña fracción de audiolibros sin coincidencia exacta de título en Gutenberg se permitió un **emparejamiento difuso** (*fuzzy matching*) de títulos para mejorar la cobertura.

### 4.2. Preprocesamiento de texto y modelo acústico de alineación

El texto de cada libro se **normaliza**: se convierte a mayúsculas, se elimina la puntuación y se expanden abreviaturas y palabras no estándar. Con el toolkit **SRILM** se entrena un modelo de lenguaje **bigrama** (suavizado Witten-Bell) sobre el texto de ese libro. El léxico se basa en **CMUdict**, quitando los marcadores numéricos de acento; para las palabras fuera de vocabulario (OOV) las pronunciaciones se generan con el toolkit de conversión grafema-a-fonema **Sequitur G2P**. Los capítulos se parten en segmentos de hasta 30 minutos y se reconocen con el decodificador `gmm-decode-faster` de Kaldi, usando un modelo de trifonos entrenado sobre VoxForge (con Boosted MMI, sobre MFCC procesados con *frame-splicing*, LDA y una transformación STC global).

### 4.3. Primera etapa de alineación

Se aplica el algoritmo de alineación **Smith-Waterman** para encontrar la mejor región única de correspondencia entre el audio reconocido y el texto del capítulo. Es análogo a una alineación de Levenshtein, pero **no exige consumir toda la referencia ni toda la hipótesis** de principio a fin, y usa pesos ajustables para los distintos tipos de error. Se toma la mayor región de similitud (que en la mayoría de los casos es el capítulo completo) y se descarta el resto. Dentro de esa región, una palabra del transcrito se marca como parte de una **"isla de confianza"** si coincide exactamente con la referencia en una secuencia de **12 fonemas o más**. Luego el audio se parte, mediante programación dinámica, en segmentos de **35 segundos o menos**, permitiendo cortes solo en silencios de al menos 0,5 segundos ubicados dentro de una isla de confianza. Así se obtiene un texto candidato para cada trozo.

### 4.4. Segunda etapa de alineación: filtrado de calidad

El objetivo de la segunda etapa es **descartar los segmentos cuyo texto candidato tenga alta probabilidad de ser inexacto**. Las fuentes de discrepancia texto-audio son varias: errores en los textos de Gutenberg, inserciones/eliminaciones/sustituciones/transposiciones introducidas por el lector, disfluencias involuntarias, normalización de texto imprecisa y errores de conversión grafema-a-fonema. El paper muestra ejemplos concretos: un lector que dice "arms" en vez de "arm" y repite "I rushed"; una entrada de diccionario auto-generada incorrecta para "Chamounix".

Para cada segmento se construye un **grafo de decodificación a medida** que combina la secuencia lineal de palabras del transcrito con un modelo de lenguaje de **bigramas a nivel de fonema**. El bigrama de fonemas permite modelar inserciones arbitrarias entre palabras o el reemplazo de palabras; **cualquier enunciado cuya decodificación se desvíe del transcrito se rechaza**. Para mantener el grafo manejable, se usa una sola copia de la parte de bigramas y se modifica el decodificador de modo que, tras entrar en ella desde la posición $x$ del transcrito, solo pueda retornar en la posición $x$ (inserción entre palabras) o $x+1$ (sustitución o eliminación) — como un transductor de pila que solo almacena un elemento. En esta segunda pasada se usa un modelo **adaptado al hablante** con transformaciones fMLLR estimadas por hablante.

El método detecta bien los desajustes, sobre todo para hablantes nativos, aunque produce algunos **falsos rechazos** (por ejemplo, la "asimilación" de una palabra corta de 1-2 fonemas dentro de un silencio vecino). Pero como el audio original de los audiolibros es enorme, **se puede permitir descartar un cierto porcentaje**. Todo el proceso de alineación tomó unas **65 horas sobre dos instancias `cc2.8xlarge` de Amazon EC2**, produciendo un conjunto inicial de audio alineado de aproximadamente **1200 horas**.

### 4.5. Segmentación final

Los segmentos de hasta 35 segundos con transcripción confiable se parten luego en piezas más pequeñas con **dos criterios distintos**: para los datos de **entrenamiento**, se corta en cualquier silencio de más de 0,3 segundos; para los datos de **prueba/desarrollo**, solo se corta si el silencio coincide con un fin de oración en el texto de referencia (los datos partidos en fronteras de oración son más fáciles de reconocer desde el punto de vista del modelado de lenguaje).

### 4.6. La división clean / other

Este es uno de los aportes más citados. Para garantizar que **no hubiera solapamiento de hablantes** entre entrenamiento, desarrollo y prueba, se excluyeron géneros como "Dramatic Reading" (multi-lector), se aplicó el toolkit de **diarización LIUM** para detectar capítulos multi-hablante y se usó una aplicación gráfica a medida para inspeccionar, filtrar grabaciones con problemas de calidad y etiquetar el género de cada hablante, cuidando el **balance de género** a nivel de hablantes y de cantidad de datos.

La separación por **calidad** se hizo con un procedimiento automático elegante: se entrenó un modelo acústico sobre el subconjunto `si-84` de WSJ y se lo usó para reconocer todo el audio del corpus (con un bigrama estimado sobre el texto de cada libro). Se computó la **tasa de error de palabra (WER, *Word Error Rate*)** de esa transcripción automática respecto de la referencia obtenida del texto del libro. Los hablantes se **ordenaron según su WER** y se dividieron aproximadamente por la mitad: los de **menor WER** se designaron **"clean"** (señal más limpia, acentos más cercanos al inglés estadounidense, pronunciación más clara) y los de **mayor WER**, **"other"** (más difíciles). La idea es que un WER bajo con un modelo estándar es un *proxy* de audio limpio y bien pronunciado, mientras que un WER alto señala ruido, acento marcado o mala grabación.

## 5. Estadísticas, splits estándar y baselines

Del *pool* "clean" se extrajeron al azar 20 hablantes masculinos y 20 femeninos para el conjunto de **desarrollo** y otros tantos para el de **prueba** (unos 8 minutos por hablante, ~5 horas 20 minutos cada conjunto). El resto del *pool* clean se partió al azar en dos conjuntos de entrenamiento de ~100 y ~360 horas, limitando cada hablante a 25 minutos para evitar desbalances. Del *pool* "other" se formaron dev/test y un único set de entrenamiento de ~500 horas; para dev/test-other se eligieron **deliberadamente hablantes más difíciles**, muestreados del tercer cuartil de la lista ordenada por WER.

Los **splits estándar** resultantes (Tabla 1 del paper) son el corazón del benchmark:

| Subconjunto | Horas | Min/hablante | Mujeres | Hombres | Total hablantes |
|---|---|---|---|---|---|
| dev-clean | 5,4 | 8 | 20 | 20 | 40 |
| test-clean | 5,4 | 8 | 20 | 20 | 40 |
| dev-other | 5,3 | 10 | 16 | 17 | 33 |
| test-other | 5,1 | 10 | 17 | 16 | 33 |
| train-clean-100 | 100,6 | 25 | 125 | 126 | 251 |
| train-clean-360 | 363,6 | 25 | 439 | 482 | 921 |
| train-other-500 | 496,7 | 30 | 564 | 602 | 1166 |

La nomenclatura `train-clean-100`, `train-clean-360` y `train-other-500` (más las cuatro particiones dev/test) se volvió vocabulario universal del ASR: un resultado "en test-clean" o "en test-other" es inmediatamente comparable entre cualquier par de trabajos.

**Modelos de lenguaje.** Para reproducibilidad, se liberaron datos y modelos de lenguaje construidos sobre libros de Gutenberg, **filtrando cuidadosamente cualquier solapamiento** con los textos de dev/test (se eliminaron los libros base de esos conjuntos y todo libro con similitud de título > 0,7 sobre 3-gramas de letras, y luego los que compartían más del 1% de sus 5-gramas mediante un índice invertido). Quedaron unos **14 500 libros de dominio público, ~803 millones de tokens y 900 000 palabras únicas**. Del léxico se tomaron las 200 000 palabras más frecuentes (que cubren ~97,5% de los tokens de evaluación). Los modelos 3-grama y 4-grama con suavizado Kneser-Ney modificado tienen perplejidad de **170 y ~150** respectivamente, con tasa OOV de ~0,4%.

**Baselines.** El experimento estrella (Tabla 2) muestra que modelos acústicos entrenados en LibriSpeech logran **menor WER en los test de WSJ** que los entrenados en el propio WSJ: por ejemplo, el modelo DNN entrenado sobre las 960 horas de LibriSpeech alcanza 3,63 en `eval'92` frente a 3,92 del DNN `si-284` de WSJ. Sobre los propios test de LibriSpeech (Tabla 3, con rescoring 4-grama), el mejor modelo DNN-960h logra **5,51 de WER en test-clean y 13,97 en test-other**. El contraste sistemático entre clean y other (típicamente un factor de 2 a 3 en WER) confirma que la división por calidad captura una dificultad real y da a la comunidad dos regímenes de evaluación: uno "fácil" y uno "difícil".

## 6. Impacto: por qué se volvió EL benchmark de ASR en inglés

Tres propiedades explican que LibriSpeech desplazara a WSJ y Switchboard como referencia:

1. **Dominio público y licencia libre (CC BY 4.0).** Cualquiera puede descargarlo, redistribuirlo y publicar sistemas entrenados sobre él sin barreras de licencia. Esto democratizó la investigación de ASR.
2. **Escala.** 1000 horas eran, en 2015, un orden de magnitud más que WSJ, suficiente para los modelos profundos que empezaban a dominar y, más tarde, para el preentrenamiento autosupervisado.
3. **Splits reproducibles.** Las particiones fijas por hablante y la separación clean/other hacen que las cifras sean directamente comparables entre laboratorios y entre años.

Sobre esta base se construyó buena parte del ASR moderno. **wav2vec 2.0** (Baevski et al., 2020) usó las 960 horas de LibriSpeech para su preentrenamiento autosupervisado y demostró que con muy pocas etiquetas (incluso 10 minutos) se podía alcanzar WER competitivo, estableciendo test-clean/test-other como la métrica estándar del paradigma *self-supervised*. **Whisper** (Radford et al., 2022), aun entrenado sobre datos web masivos y débilmente supervisados, reporta sus resultados en LibriSpeech test-clean como punto de comparación obligado. En la práctica, un número de LibriSpeech es hoy la primera cifra que se mira para situar un sistema de ASR en inglés.

## 7. Limitaciones

- **Habla leída, no conversacional.** LibriSpeech es habla **leída** de literatura: enunciados fluidos, bien articulados, gramaticalmente completos y sin las disfluencias, solapamientos, dudas ni cambios de turno del habla espontánea. Un sistema excelente en LibriSpeech puede rendir mucho peor en habla conversacional real (como la de Switchboard) o en habla de reunión.
- **Solo inglés.** El corpus es monolingüe. La escasez de recursos comparables en otros idiomas motivó proyectos posteriores como **Multilingual LibriSpeech (MLS)** y **Common Voice**, que la Clase 37 también presenta.
- **Sesgo de dominio y de acento.** El procedimiento clean privilegia acentos cercanos al inglés estadounidense; la literatura de dominio público (mayormente clásica) impone un registro léxico y temático particular, distinto del habla cotidiana.
- **Audio de origen MP3.** Las grabaciones de LibriVox provienen de audio comprimido con pérdida (MP3) y con prácticas de limpieza de ruido y normalización de volumen aplicadas de forma inconsistente. Por eso los autores usaron WSJ (audio no comprimido) como baseline complementario para evaluar el efecto de la compresión.
- **Ruido de alineación residual.** Pese al filtrado en dos etapas, subsisten falsos rechazos y algún desajuste texto-audio; el diseño asume que la escala compensa las pérdidas.

## 8. Conexión con la Clase 37 y con wav2vec 2.0 / Whisper

La Clase 37, "Datasets y Herramientas para Audio", cita LibriSpeech como el ejemplo de **"habla transcrita"** y menciona explícitamente su distribución **"en FLAC"**. FLAC (*Free Lossless Audio Codec*) es un formato de **compresión sin pérdida**: reduce el tamaño de archivo frente al WAV/PCM crudo sin degradar ni un bit la señal, algo esencial para un corpus de ~1000 horas donde la fidelidad acústica no debe sacrificarse pero el peso en disco sí importa. Que LibriSpeech se distribuya en FLAC ilustra un principio de la clase: elegir el formato correcto (sin pérdida para datasets de investigación, con pérdida para *streaming* de consumo) es parte del diseño de un recurso de audio.

LibriSpeech es, además, el hilo que conecta la clase con los modelos de audio de vanguardia. Es el **conjunto de preentrenamiento y evaluación de wav2vec 2.0**, el modelo autosupervisado que aprende representaciones de voz sin transcripciones y que la clase presenta como el salto hacia el aprendizaje con pocas etiquetas. Y es la **vara de medición de Whisper**, el modelo multitarea de OpenAI. En ambos casos, la existencia de un benchmark libre, grande y reproducible fue precondición para el progreso: sin LibriSpeech, no habría habido un terreno común sobre el cual comparar el ASR clásico basado en HMM-GMM/Kaldi (como el de este mismo paper) con los modelos neuronales end-to-end y autosupervisados que vinieron después.

**Enlaces internos:**

- Clase: [/clases/clase-37](/clases/clase-37) — Datasets y Herramientas para Audio.
- Fundamento transversal: [/fundamentos/reconocimiento-de-voz](/fundamentos/reconocimiento-de-voz) — ASR, WER, modelo acústico y de lenguaje.
- Papers relacionados: wav2vec 2.0 (representaciones autosupervisadas de voz) y Whisper (ASR multitarea débilmente supervisado), que usan LibriSpeech como benchmark.

## 9. Nota final: relevancia para salud

LibriSpeech es una lección sobre el valor de los **corpora libres y bien documentados**. Su impacto no vino de una arquitectura novedosa —el paper usa modelos HMM-GMM y DNN estándar de Kaldi— sino de haber liberado, bajo licencia permisiva, un recurso grande, con *splits* reproducibles y con la receta completa para reconstruirlo. Ese fue el detonante que permitió comparar, competir y progresar de forma abierta durante una década. El contraste con el mundo clínico es directo y aleccionador: en salud, los datos de audio (consultas médicas, dictados clínicos, llamadas de telemedicina, registros de voz como biomarcador de patologías respiratorias o neurológicas) están atados por privacidad, consentimiento y regulación, y **prácticamente no existen corpora abiertos, grandes y bien anotados** equivalentes a LibriSpeech. Esa escasez frena el desarrollo de ASR y de biomarcadores de voz específicos del dominio médico, obliga a depender de modelos preentrenados en dominios generales (con el sesgo de dominio que la sección 7 advierte) y hace que el progreso en salud digital dependa críticamente de esfuerzos —difíciles pero indispensables— por construir datasets clínicos anonimizados, consentidos y compartibles que reproduzcan, en su ámbito, lo que LibriSpeech hizo por el ASR en inglés.
