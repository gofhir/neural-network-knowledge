---
title: "LibriSpeech: corpus de ASR de dominio público (2015)"
weight: 414
math: true
---

{{< paper-card
    title="Librispeech: An ASR Corpus Based on Public Domain Audio Books"
    authors="Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev Khudanpur (JHU)"
    year="2015"
    venue="ICASSP 2015"
    pdf="/papers/librispeech-panayotov-2015.pdf" >}}
LibriSpeech es un corpus de **habla en inglés leída** para entrenar y evaluar reconocimiento automático del habla (ASR): **~1.000 horas** a 16 kHz, derivadas de audiolibros de **LibriVox** cuyos textos provienen de **Project Gutenberg**, ambos de dominio público. No libera solo el audio segmentado y transcrito, sino la **receta completa**: alineación, criterio de división *clean/other*, *splits* estándar y los *scripts* de Kaldi (Daniel Povey, su creador, es coautor). El resultado que ancló su credibilidad es contraintuitivo: modelos entrenados en LibriSpeech logran **menor WER en los test del Wall Street Journal** que modelos entrenados en el propio WSJ —la escala (1.000 h vs. 82 h de `si-284`) compensa la diferencia de dominio—. Eso, con licencia **CC BY 4.0** y *splits* reproducibles, lo convirtió en **el** benchmark de ASR en inglés y en la base de [wav2vec 2.0](/papers/wav2vec2-baevski-2020) y [Whisper](/papers/whisper-radford-2022). Es el ejemplo canónico de "habla transcrita" y de distribución en **FLAC** de la [Clase 37](/clases/clase-37).
{{< /paper-card >}}

---

## Contexto: la necesidad de un corpus grande, libre y reproducible

A mediados de la década de 2010, el ASR ya se apoyaba en modelos estadísticos entrenados con datos, pero los corpora de referencia arrastraban un problema estructural: **las licencias**. Los dos benchmarks históricos del ASR en inglés eran de pago y circulación restringida: **Wall Street Journal (WSJ)** —habla leída de artículos periodísticos, distribuido por el LDC, cuyo subconjunto de entrenamiento estándar `si-284` tiene solo 82 horas— y **Switchboard** —habla telefónica conversacional, también bajo licencia LDC—. Valiosos, pero caros e imposibles de redistribuir, lo que dificultaba la comparación abierta y excluía a quien no pudiera pagar.

Al mismo tiempo, la explosión de contenido multimedia hacía factible recolectar datos automáticamente, sobre todo desde colecciones curadas: **LibriVox** había producido cerca de 8.000 audiolibros de dominio público, y **Project Gutenberg** aportaba los textos base. Existía un antecedente parcial, **VoxForge**, pero era mucho más pequeño (~100 horas) y con fuertes desbalances de género y duración por hablante. No había ningún corpus de habla leída en inglés, libre y de la escala necesaria para el ASR moderno. Ese es el vacío que LibriSpeech vino a llenar.

## Composición: un corpus y un protocolo reproducible

La contribución es doble. Primero, **un corpus**: ~1.000 horas de habla leída, 16 kHz, segmentada en enunciados cortos con sus transcripciones, libre bajo CC BY 4.0, con datos y modelos de lenguaje pre-construidos. Segundo, **un protocolo reproducible**: la receta completa para producir los datos y usarlos —alineación de audio largo, criterio *clean/other*, *splits* estándar y *scripts* de Kaldi—. Esa combinación (dominio público + escala + reproducibilidad) es lo que le dio longevidad: un dataset grande pero de licencia restrictiva no habría sido estándar universal; uno libre pero pequeño (VoxForge) no habría bastado.

El desafío técnico central es que los modelos acústicos esperan enunciados cortos con su texto, mientras un audiolibro es una grabación continua de horas. El pipeline **alinea, segmenta y filtra** en dos etapas:

- **Materia prima.** Se cruzan la API de LibriVox, los metadatos del Internet Archive y los archivos RDF de Gutenberg (con *fuzzy matching* de títulos para casos sin coincidencia exacta).
- **Alineación (primera etapa).** El texto se normaliza; con SRILM se entrena un bigrama por libro y se decodifica con Kaldi (`gmm-decode-faster`, modelo de trifonos de VoxForge). El algoritmo **Smith-Waterman** encuentra la región de correspondencia entre audio reconocido y texto; se marcan "islas de confianza" (coincidencia exacta de ≥12 fonemas) y se corta en silencios de ≥0,5 s dentro de esas islas, en segmentos de ≤35 s.
- **Filtrado de calidad (segunda etapa).** Para cada segmento se construye un grafo de decodificación a medida con un bigrama a nivel de **fonema**; cualquier enunciado cuya decodificación se desvíe del transcrito se rechaza. Todo el proceso tomó ~65 horas en dos instancias EC2 `cc2.8xlarge`, produciendo ~1.200 horas alineadas.

## La división clean / other y los splits estándar

Uno de los aportes más citados. Para garantizar que **no hubiera solapamiento de hablantes** entre entrenamiento, desarrollo y prueba, se excluyeron lecturas multi-lector, se usó diarización LIUM y una app a medida para inspeccionar calidad y balancear género. La separación por **calidad** se hizo automáticamente: se entrenó un modelo acústico sobre `si-84` de WSJ, se reconoció todo el corpus y se computó la **tasa de error de palabra (WER)** contra la referencia. Los hablantes se ordenaron por WER y se partieron por la mitad: los de menor WER quedaron **clean** (señal limpia, acento cercano al inglés estadounidense) y los de mayor WER, **other** (más difíciles). Un WER bajo con un modelo estándar es un *proxy* de audio limpio y bien pronunciado.

Los *splits* resultantes (Tabla 1) son el corazón del benchmark:

| Subconjunto | Horas | Mujeres | Hombres | Total hablantes |
|---|---|---|---|---|
| dev-clean | 5,4 | 20 | 20 | 40 |
| test-clean | 5,4 | 20 | 20 | 40 |
| dev-other | 5,3 | 16 | 17 | 33 |
| test-other | 5,1 | 17 | 16 | 33 |
| train-clean-100 | 100,6 | 125 | 126 | 251 |
| train-clean-360 | 363,6 | 439 | 482 | 921 |
| train-other-500 | 496,7 | 564 | 602 | 1166 |

La nomenclatura `train-clean-100/360` y `train-other-500`, más las cuatro particiones dev/test, se volvió vocabulario universal del ASR: un resultado "en test-clean" o "en test-other" es inmediatamente comparable entre trabajos. El experimento estrella (Tabla 2) muestra que el DNN entrenado sobre las 960 horas de LibriSpeech alcanza **3,63 de WER en `eval'92` de WSJ** frente a 3,92 del DNN `si-284`; sobre los propios test (Tabla 3, rescoring 4-grama), el mejor DNN-960h logra **5,51 en test-clean y 13,97 en test-other**. El factor 2-3 entre clean y other confirma que la división captura una dificultad real y da dos regímenes de evaluación.

## Impacto

Tres propiedades explican que LibriSpeech desplazara a WSJ y Switchboard: **dominio público** (CC BY 4.0, sin barreras de licencia), **escala** (un orden de magnitud más que WSJ, suficiente para modelos profundos y luego para el preentrenamiento autosupervisado) y **splits reproducibles** (cifras directamente comparables entre laboratorios y años). Sobre esa base se construyó buena parte del ASR moderno: [wav2vec 2.0](/papers/wav2vec2-baevski-2020) usó sus 960 horas para preentrenamiento autosupervisado y demostró WER competitivo con muy pocas etiquetas; [Whisper](/papers/whisper-radford-2022), aun entrenado sobre datos web masivos, reporta LibriSpeech test-clean como punto de comparación obligado. Hoy, un número de LibriSpeech es la primera cifra que se mira para situar un sistema de ASR en inglés.

## Limitaciones

- **Habla leída, no conversacional.** Enunciados fluidos y bien articulados, sin las disfluencias, solapamientos ni cambios de turno del habla espontánea; un sistema excelente aquí puede rendir peor en habla real.
- **Solo inglés.** Corpus monolingüe; la escasez de recursos comparables motivó Multilingual LibriSpeech y Common Voice.
- **Sesgo de dominio y de acento.** El procedimiento clean privilegia acentos cercanos al inglés estadounidense; la literatura clásica impone un registro léxico particular.
- **Audio de origen MP3.** LibriVox proviene de audio comprimido con pérdida y limpieza inconsistente; por eso se usó WSJ (audio no comprimido) como baseline complementario.
- **Ruido de alineación residual.** Pese al filtrado en dos etapas subsisten falsos rechazos; el diseño asume que la escala compensa las pérdidas.

## Por qué importa para la Clase 37

La [Clase 37](/clases/clase-37), "Datasets y Herramientas para Audio", cita LibriSpeech como el ejemplo de **habla transcrita** y su distribución **en FLAC** (*Free Lossless Audio Codec*): compresión sin pérdida que reduce el tamaño frente al WAV/PCM crudo sin degradar la señal, esencial para un corpus de ~1.000 horas donde la fidelidad no debe sacrificarse pero el peso en disco sí importa. Elegir el formato correcto —sin pérdida para datasets de investigación, con pérdida para *streaming* de consumo— es parte del diseño de un recurso de audio y un criterio central entre los [datasets de audio](/fundamentos/datasets-de-audio).

LibriSpeech es, además, el hilo que conecta la clase con los modelos de vanguardia: es el conjunto de preentrenamiento y evaluación de [wav2vec 2.0](/papers/wav2vec2-baevski-2020) y la vara de medición de [Whisper](/papers/whisper-radford-2022). Su impacto no vino de una arquitectura novedosa —usa HMM-GMM y DNN estándar de Kaldi— sino de haber liberado, bajo licencia permisiva, un recurso grande con *splits* reproducibles y receta completa. El contraste con el mundo clínico es directo: en salud, los datos de audio (dictados clínicos, telemedicina, voz como biomarcador respiratorio o neurológico) están atados por privacidad y regulación, y prácticamente no existen corpora abiertos y grandes equivalentes a LibriSpeech. Esa escasez frena el ASR y los biomarcadores de voz médicos, y hace que el progreso en salud digital dependa de construir datasets clínicos anonimizados, consentidos y compartibles que reproduzcan, en su ámbito, lo que LibriSpeech hizo por el ASR en inglés.
