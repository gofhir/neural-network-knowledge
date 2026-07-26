# Robust Speech Recognition via Large-Scale Weak Supervision (Whisper) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Robust Speech Recognition via Large-Scale Weak Supervision*.
- **Autores:** Alec Radford\*, Jong Wook Kim\* (contribución equivalente), Tao Xu, Greg Brockman, Christine McLeavey, Ilya Sutskever. Todos en **OpenAI**, San Francisco.
- **Publicación:** informe técnico de OpenAI, diciembre de 2022. **Preprint:** arXiv:2212.04356v1 [eess.AS], 6 de diciembre de 2022.
- **Código y modelos:** liberados públicamente en [github.com/openai/whisper](https://github.com/openai/whisper), como base para investigación en reconocimiento robusto de voz.
- **Nombre:** *Whisper*, que los autores retroajustan al acrónimo **WSPSR** (*Web-scale Supervised Pretraining for Speech Recognition*).

El paper estudia qué capacidades emergen cuando se entrena un sistema de procesamiento de voz de la forma más simple posible: **predecir grandes cantidades de transcripciones de audio recolectadas de internet**. La apuesta central es de **escala de datos**: al llevar la supervisión débil hasta **680.000 horas** de audio multilingüe y multitarea, el modelo resultante generaliza bien a los benchmarks estándar y suele ser competitivo con sistemas totalmente supervisados, pero **en un régimen de transferencia zero-shot, sin ningún fine-tuning**. Comparado con transcriptores humanos profesionales, Whisper se aproxima a su exactitud y robustez.

La tesis es que el reconocimiento de voz venía subestimando el escalamiento simple de la **supervisión débil** (*weak supervision*). Mientras la vanguardia de 2020-2021 giraba en torno al pre-entrenamiento auto-supervisado (wav2vec 2.0 y sucesores) seguido de un fine-tuning obligatorio, Whisper muestra que basta con entrenar un solo modelo encoder-decoder Transformer sobre un dataset enorme, diverso y ruidoso, y enfocarse en la transferencia zero-shot, para obtener un sistema que funciona **"out of the box"** frente a ruido, acentos, idiomas y dominios que nunca vio explícitamente.

Para la **Clase 37 (Datasets y Herramientas para Audio)**, Whisper es *el* ejemplo de la frontera de escala del dato de audio. La clase señala que Whisper "no se entrenó con ninguno de los datasets clásicos: usó ~680.000 horas de audio web débilmente etiquetado". Este paper es exactamente la crónica de cómo se construye, filtra y aprovecha ese océano de datos, y por qué el ciclo de vida del dato de audio —cómo vive el audio en disco, cómo se representa, cómo se limpia y empareja con texto— es lo que en última instancia determina la robustez del modelo.

## 2. Contexto: el ASR supervisado clásico y el techo del auto-supervisado

Para entender la contribución hay que ver el mapa del campo hacia 2022, que el paper divide en dos líneas.

**Línea 1: supervisión de alta calidad, pero escasa.** El reconocimiento de voz supervisado clásico se entrena con transcripciones validadas por humanos (*gold-standard*). El problema es que estos datasets son pequeños: los datasets académicos típicos rondan las **1.000 horas**. Los intentos de combinar varios corpus de alta calidad —como SpeechStew, que mezcla 7 datasets— llegan solo a unas **5.140 horas**. Es útil, pero minúsculo. Y hay un hallazgo incómodo: los modelos entrenados en un único dataset como LibriSpeech alcanzan desempeño "sobrehumano" *dentro de la distribución* pero cometen muchísimos más errores que un humano al cambiar de dominio. El SOTA en LibriSpeech test-clean bajó de 5,3% a 1,4% de WER (muy por debajo del 5,8% humano), y sin embargo esos mismos modelos siguen siendo frágiles fuera de LibriSpeech. Aprenden las idiosincrasias del dataset, no a "escuchar".

**Línea 2: auto-supervisión escalable, pero sin decoder.** Wav2Vec 2.0 (Baevski et al., 2020) y sucesores aprenden representaciones directamente del **audio crudo sin etiquetas humanas**, lo que les permite escalar a cantidades masivas de habla no etiquetada —hasta **1.000.000 de horas** (Zhang et al., 2021)—. Estos encoders producen representaciones de alta calidad y mejoran el estado del arte, sobre todo en regímenes de pocos datos. Pero tienen una debilidad estructural: al ser puramente no supervisados, **carecen de un decoder igualmente competente** que mapee esas representaciones a una salida usable. Por eso **necesitan una etapa de fine-tuning** para efectivamente hacer reconocimiento de voz. Y ahí aparecen dos costos:

1. El fine-tuning es un proceso complejo que requiere un practicante experto, lo que limita el impacto real de estos modelos.
2. Fine-tunear sobre la distribución de despliegue arrastra el mismo riesgo de fragilidad: el aprendizaje de máquina es extraordinariamente hábil para encontrar patrones espurios que suben el desempeño *en el dataset de entrenamiento* pero no generalizan. El paper cita el caso análogo en visión (Radford et al., 2021): fine-tunear en ImageNet subió 9,2% la exactitud en ImageNet sin mejorar en absoluto la exactitud promedio sobre otros siete datasets de imágenes naturales.

La conclusión de los autores es que el objetivo de un sistema de reconocimiento de voz debería ser **funcionar confiablemente "out of the box"** en un amplio rango de entornos, sin necesitar fine-tuning supervisado de un decoder para cada distribución de despliegue. Y trabajos previos ya mostraban el camino: los sistemas pre-entrenados de forma supervisada sobre **muchos** datasets/dominios (Narayanan et al. 2018; Chan et al. 2021) exhiben más robustez que los entrenados en una sola fuente. El único ingrediente que faltaba era la escala.

## 3. Contribución central

Whisper cierra la brecha de escala con tres apuestas combinadas:

1. **Escala + supervisión débil.** En vez de exigir transcripciones humanas verificadas, se relaja el requisito y se usa audio de internet emparejado con transcripciones *tal como están*, aceptando ruido en las etiquetas a cambio de cantidad. Esto lleva la supervisión débil un orden de magnitud más allá que el trabajo previo (que había llegado a 10.000-30.000 horas): **680.000 horas de audio etiquetado**. La justificación viene por analogía con visión, donde moverse de datasets curados como ImageNet a datasets mucho más grandes pero débilmente supervisados mejora significativamente robustez y generalización.
2. **Multilingüe y multitarea.** El pre-entrenamiento no se limita al inglés. De las 680.000 horas, **117.000 cubren otros 96 idiomas** y hay **125.000 horas de datos de traducción X→inglés**. Es decir, cerca de **un tercio del dataset es no-inglés**. Los autores encuentran que, para modelos suficientemente grandes, entrenar conjuntamente en muchos idiomas y tareas **no perjudica y hasta beneficia** (transferencia positiva).
3. **Zero-shot en vez de fine-tuning.** Se demuestra que un modelo entrenado a esta escala transfiere bien a datasets existentes sin fine-tuning específico, eliminando por completo la etapa que hacía frágiles y engorrosos a los sistemas auto-supervisados.

Notablemente, todo esto se logra **sin** las técnicas de auto-supervisión ni self-training que dominaban el campo. Simplemente escala de supervisión débil.

## 4. Método

### 4.1. Datos y pipeline de filtrado

El corazón práctico del paper —y lo más relevante para la Clase 37— es cómo se construye el dataset. La filosofía es **minimalista en el pre-procesamiento**: se entrena a Whisper a predecir el **texto crudo** de las transcripciones, sin estandarización significativa, confiando en que el modelo seq2seq aprenda por sí mismo a mapear el habla a su forma escrita naturalista (con puntuación, mayúsculas, etc.). Esto elimina la etapa separada de normalización inversa de texto que arrastran los pipelines clásicos.

El dataset se construye a partir de audio emparejado con transcripciones en internet, lo que da una distribución muy diversa de entornos, micrófonos, hablantes e idiomas. La diversidad de *calidad de audio* ayuda a la robustez; pero la diversidad de *calidad de transcripción* no. La inspección inicial reveló muchas transcripciones deficientes, así que se desarrollaron varios **filtros automatizados**:

- **Detección y remoción de transcripciones generadas por máquina.** Muchas transcripciones en internet no son humanas, sino la salida de otros sistemas ASR. Entrenar sobre datos mezclados humano/máquina degrada el desempeño (Ghorbani et al., 2021), y aprender el "transcript-ese" de otros ASR es indeseable. La heurística explota que los sistemas ASR normalizan cosas difíciles de inferir solo del audio: una transcripción **todo en mayúsculas o todo en minúsculas** es casi con certeza generada por máquina; la ausencia sistemática de comas, signos de exclamación o interrogación, o de formato de párrafos, también delata origen automático. Este filtrado —detectar transcripciones automáticas para excluirlas— es un paso central del pipeline.
- **Coincidencia de idioma audio-texto.** Se usa un detector de idioma de audio (un prototipo fine-tuneado sobre VoxLingua107) para verificar que el idioma hablado coincide con el idioma de la transcripción (según CLD2). Si no coinciden, el par (audio, transcripción) **no se incluye** como ejemplo de reconocimiento. Excepción clave: si la transcripción está en inglés pero el audio no, el par se recicla como ejemplo de **traducción X→inglés**. Esta regla es la que genera automáticamente el corpus de traducción.
- **De-duplicación difusa** (*fuzzy de-duping*) de los textos para reducir duplicación y contenido autogenerado.

El audio se parte en **segmentos de 30 segundos** emparejados con el subconjunto de la transcripción que cae en esa ventana. Se entrena sobre *todo* el audio, incluidos segmentos sin habla (con probabilidad submuestreada), que sirven como datos para detección de actividad de voz (VAD).

Hay una segunda pasada de filtrado: tras entrenar un modelo inicial, se agrega su tasa de error por fuente de datos y se inspeccionan manualmente las fuentes ordenadas por *alta tasa de error × tamaño*, para identificar y remover eficientemente las de baja calidad (transcripciones parcialmente transcritas, mal alineadas o subtítulos automáticos que las heurísticas no detectaron). Finalmente, para evitar contaminación, se de-duplica a nivel de transcripción entre el dataset de entrenamiento y los datasets de evaluación con mayor riesgo de solapamiento (p. ej. TED-LIUM 3).

### 4.2. Arquitectura: encoder-decoder Transformer sobre log-Mel

Como el foco es estudiar las capacidades del pre-entrenamiento a gran escala —no proponer una arquitectura nueva—, se usa una arquitectura **estándar off-the-shelf** para no confundir los hallazgos con mejoras de modelo: un **Transformer encoder-decoder** (Vaswani et al., 2017), elegido por escalar de forma confiable.

El front-end de audio es el clásico del procesamiento de voz:

- Todo el audio se **re-muestrea a 16.000 Hz**.
- Se computa un **espectrograma log-Mel de 80 canales** (log-magnitud) sobre ventanas de **25 ms** con paso (*stride*) de **10 ms**.
- Normalización global de la entrada al rango $[-1, 1]$ con media aproximadamente cero sobre el dataset.

El encoder procesa esta representación con un *stem* de **dos capas convolucionales** (ancho de filtro 3, activación GELU), donde la segunda convolución tiene stride 2. Se añaden **embeddings posicionales sinusoidales** y luego los bloques Transformer del encoder (con bloques residuales pre-activación y una layer-norm final). El decoder usa **embeddings posicionales aprendidos** y representaciones de token de entrada/salida atadas (*tied*). Encoder y decoder tienen el mismo ancho y número de bloques. El tokenizador de texto es el mismo BPE a nivel de byte de GPT-2, con vocabulario reajustado (mismo tamaño) para los modelos multilingües, para evitar fragmentación excesiva en idiomas no ingleses.

La familia de modelos escala en profundidad y ancho:

| Modelo | Capas | Ancho | Cabezas | Parámetros |
|---|---|---|---|---|
| Tiny | 4 | 384 | 6 | 39 M |
| Base | 6 | 512 | 8 | 74 M |
| Small | 12 | 768 | 12 | 244 M |
| Medium | 24 | 1024 | 16 | 769 M |
| Large | 32 | 1280 | 20 | 1550 M |

**Entrenamiento:** paralelismo de datos en FP16 con *dynamic loss scaling* y *activation checkpointing*, optimizador AdamW con *gradient norm clipping*, decaimiento lineal del learning rate a cero tras un warmup de 2048 updates. Batch de **256 segmentos**, entrenados por $2^{20}$ updates, lo que equivale a **entre dos y tres pasadas** sobre el dataset. Como se entrenan solo pocas épocas, el sobreajuste no es preocupación mayor y **no se usa data augmentation ni regularización** en la versión original: se confía en la diversidad del dataset gigante para inducir generalización. (Tras el lanzamiento original, el modelo Large V2 se entrenó 2,5× más épocas y sí incorporó **SpecAugment**, Stochastic Depth y BPE Dropout como regularización; los resultados reportados usan esta versión mejorada salvo indicación contraria.)

### 4.3. Formato multitarea con tokens especiales

Aquí está la elegancia del diseño. Un sistema completo de procesamiento de voz no es solo "predecir qué palabras se dijeron": incluye detección de actividad de voz, diarización, normalización inversa de texto, identificación de idioma, alineación temporal, traducción. Tradicionalmente cada componente es un modelo aparte, y armar el sistema completo es complejo. Whisper quiere que **un solo modelo** haga todo el pipeline.

La solución es tratar todas las tareas como un problema de **predicción de secuencias**, especificando la tarea mediante **tokens especiales de entrada al decoder**. Como el decoder es un modelo de lenguaje condicionado al audio, la secuencia de tokens que lo condiciona es:

1. Opcionalmente, `<|PREV|>` seguido del **texto de la transcripción precedente** (con cierta probabilidad se añade el texto del segmento anterior como contexto, para que el modelo aprenda a usar contexto de largo alcance y resuelva audio ambiguo).
2. `<|startoftranscript|>` (SOT) marca el inicio de la predicción.
3. Un **token de idioma** (uno único por cada idioma, **99 en total**) — el modelo primero *identifica el idioma hablado*. Si no hay habla, predice `<|nospeech|>` (esto sirve de VAD).
4. Un token de **tarea**: `<|transcribe|>` (transcripción X→X) o `<|translate|>` (traducción X→inglés).
5. Un token que indica si se predicen **timestamps** o no (`<|notimestamps|>` en el caso sin timestamps).
6. La **salida**: los tokens de texto. En modo timestamp se predicen tiempos relativos al segmento actual, cuantizados a los **20 ms** más cercanos (la resolución nativa del modelo), intercalando token de tiempo-inicio antes de cada texto y tiempo-fin después.
7. `<|endoftranscript|>` (EOT) cierra la secuencia.

Así, una sola arquitectura y un solo objetivo de *next-token prediction* resuelven **transcripción multilingüe, traducción al inglés, identificación de idioma, detección de actividad de voz y transcripción alineada temporalmente**, seleccionando la tarea con tokens. La pérdida solo se enmascara sobre el texto de contexto previo; el resto de tokens se predicen normalmente.

## 5. Experimentos: la evidencia de robustez zero-shot

Todo se evalúa **zero-shot**: sin usar los splits de entrenamiento de ningún dataset de evaluación, para medir generalización amplia y no la memorización de formatos específicos. La métrica es WER (*word error rate*), con la salvedad de que el WER crudo penaliza diferencias inocuas de estilo de transcripción; por eso los autores desarrollaron un **normalizador de texto** (liberado públicamente) para descontar diferencias no semánticas antes de calcular el WER.

**Robustez efectiva vs. modelos supervisados.** Este es el resultado estrella (Figura 2, Tabla 2). El mejor Whisper zero-shot tiene un WER en LibriSpeech test-clean de **2,5**, nada espectacular (equivale a una baseline supervisada moderna). Pero cuando se lo compara con un modelo supervisado en LibriSpeech que tiene desempeño casi idéntico en test-clean (diferencia < 0,1%), Whisper **comete en promedio 55,2% menos errores** en los otros 12 datasets de evaluación. Es decir: dos modelos indistinguibles en la distribución de referencia se comportan drásticamente distinto fuera de ella. Incluso el **Whisper Tiny de 39 M de parámetros** (6,7 de WER en test-clean) es aproximadamente competitivo con el mejor modelo supervisado de LibriSpeech cuando se lo evalúa en *otros* datasets. Los autores explican la brecha con una idea conceptual fina: el desempeño humano en un test mide **generalización fuera de distribución** (el humano no entrenó en ese dataset), mientras que el desempeño de un modelo supervisado mide **generalización dentro de distribución**. Whisper, entrenado sobre una distribución amplia y evaluado zero-shot, mide lo mismo que el humano — y por eso **iguala la exactitud y robustez de un humano** en la frontera de robustez.

**Multilingüe.** En Multilingual LibriSpeech, Whisper zero-shot (7,3 WER) supera a XLS-R, mSLAM y Maestro. En VoxPopuli rinde peor, probablemente porque esos modelos incluyeron esa distribución en su pre-entrenamiento y hay mucho más dato supervisado para fine-tuning. Sobre el dataset Fleurs (más amplio), aparece una relación cuantitativa clave: la correlación cuadrada entre el logaritmo del WER y el logaritmo de la cantidad de datos de entrenamiento por idioma es **$r^2 = 0{,}83$**, y el ajuste implica que **el WER se reduce a la mitad por cada aumento de 16× en datos de entrenamiento**. Los mayores outliers negativos son idiomas de escritura única y lejanos al indoeuropeo (hebreo, telugu, chino, coreano), lo que sugiere límites de transferencia y del tokenizador BPE.

**Traducción.** En la tarea X→inglés de CoVoST2, Whisper alcanza un **nuevo estado del arte zero-shot de 29,1 BLEU**, sin usar nada del entrenamiento de CoVoST2. Lo atribuyen a las 68.000 horas de datos de traducción X→inglés (para esos idiomas) en el pre-entrenamiento, muy superiores a las 861 horas supervisadas de CoVoST2. Destaca especialmente en idiomas de bajos recursos.

**Robustez al ruido aditivo (SNR).** Se midió el WER agregando ruido blanco o *pub noise* (ruido ambiente de un bar, con murmullo indistinto) a distintas relaciones señal-ruido (*signal-to-noise ratio*, SNR). Muchos modelos entrenados en LibriSpeech superan a Whisper en ruido bajo (40 dB SNR), lo que no sorprende. Pero **todos se degradan más rápido**: bajo *pub noise* con SNR menor a 10 dB, Whisper supera a todos. Esta prueba —que la clase conecta con la data augmentation por SNR— muestra que la diversidad natural del dataset compra robustez acústica sin ingeniería explícita.

**Comparación con humanos.** Sobre 25 grabaciones de Kincaid46 transcritas por 5 servicios profesionales (uno asistido por computador, cuatro puramente humanos), el WER agregado de Whisper queda **a solo una fracción de punto** del de los transcriptores puramente humanos, y a 1,15 puntos del servicio asistido por computador. El ASR en inglés de Whisper no es perfecto, pero está muy cerca del nivel humano.

**Escalamiento.** Tanto el tamaño de modelo como el de dataset mejoran el desempeño de forma confiable (Figuras 8-9). La tabla de dataset scaling muestra que ir de 3.405 h a 681.070 h baja el WER de inglés de 30,5 a 9,9 y sube el BLEU de traducción de 0,2 a 24,8. Hay retornos decrecientes al pasar de 54.000 a 680.000 horas, lo que puede indicar o bien que los modelos actuales están sub-entrenados respecto al tamaño del dataset, o bien que el reconocimiento de voz se acerca al techo del escalamiento por datos. En modelos pequeños hay **transferencia negativa** entre tareas/idiomas, pero los modelos multilingües/multitarea escalan mejor y en los tamaños grandes **superan** a los solo-inglés (transferencia positiva).

## 6. Limitaciones

- **Alucinaciones y fallos de decodificación.** Al escalar, los errores puramente perceptuales (confundir palabras similares) se reducen de forma confiable. Pero persisten errores más tercos y "no humanos", sobre todo en audio largo: quedarse **atrapado en loops de repetición**, no transcribir las primeras o últimas palabras de un segmento, y —lo más grave— **alucinación completa**, donde el modelo emite una transcripción totalmente **no relacionada con el audio real**. Son fallos combinados de los modelos seq2seq, del modelo de lenguaje y de la alineación texto-audio. Las heurísticas de decodificación de la Sección 4.5 (beam search de 5 haces, *temperature fallback* escalando la temperatura cuando la log-prob promedio cae bajo −1 o la tasa de compresión gzip supera 2,4, condicionamiento con texto previo, restricción del timestamp inicial) ayudan, pero son parches; se necesita más investigación, posiblemente fine-tuning supervisado o RL para optimizar la decodificación.
- **Idiomas de bajos recursos.** El desempeño sigue siendo pobre en muchos idiomas. La regresión $r^2=0{,}83$ da la ruta clara de mejora (más datos), pero el pipeline de recolección es **fuertemente sesgado al inglés** (proviene de partes anglocéntricas de internet), y la mayoría de idiomas tiene menos de 1.000 horas. La identificación de idioma zero-shot (64,5% en Fleurs) tampoco es competitiva, en parte porque el dataset no tiene datos para 20 de los 102 idiomas de Fleurs.
- **Ventana de 30 segundos.** Whisper se entrena sobre trozos de 30 s y **no puede consumir audio más largo de una vez**. Para transcripción larga (minutos u horas, como en el mundo real) hace falta una estrategia de *buffered transcription*: transcribir segmentos de 30 s consecutivos y desplazar la ventana según los timestamps predichos. Como una transcripción errónea en una ventana contamina la siguiente, la fiabilidad depende de heurísticas frágiles.
- **Otras.** Solo se estudió zero-shot; no está claro cuánto de la robustez viene del encoder vs. del decoder-como-modelo-de-lenguaje; y no se exploraron objetivos auxiliares de auto-supervisión.

## 7. Conexión con la Clase 37 y con el dominio audio

La Clase 37 recorre el **ciclo de vida del dato de audio**: cómo vive el audio en disco (sample rate, bits, formatos, ffmpeg), cómo se representa (transforms, espectrogramas, tensores), cómo se aumenta (SNR, SpecAugment) y qué datasets existen. Whisper es la ilustración perfecta de que **la robustez de un modelo de audio se decide en el dato, no en la arquitectura**: los autores usan un Transformer deliberadamente estándar, y toda la diferencia la hacen las 680.000 horas y su pipeline de filtrado.

Cada eje de la clase tiene su correlato directo en el paper:

- **Audio en disco → representación.** El re-muestreo a 16 kHz, la ventana de 25 ms con paso de 10 ms y el **espectrograma log-Mel de 80 canales** son exactamente los *transforms* que la clase enseña como puente entre la forma de onda y el tensor de entrada. La cuantización de timestamps a 20 ms conecta la resolución temporal del modelo con la del audio.
- **Data augmentation.** La clase enseña augmentación por SNR y SpecAugment; el paper muestra la otra cara: la versión original **prescinde** de augmentation confiando en la diversidad natural del dataset gigante, y solo el Large V2 reincorpora SpecAugment. Y la evaluación de robustez al ruido (WER vs. SNR con ruido blanco y de bar) es literalmente el experimento que justifica por qué la augmentación por SNR importa.
- **Datasets y la frontera de escala.** Aquí Whisper es el caso paradigmático que la clase menciona: **no usó ninguno de los datasets clásicos**. Los usó solo para *evaluar* zero-shot. El dataset de entrenamiento se construyó recolectando audio web con transcripciones y filtrándolo — un cambio de mentalidad desde "curar un corpus limpio" hacia "recolectar a escala web y limpiar automáticamente", el mismo salto que ya había ocurrido en NLP (GPT) y visión.
- **El caso del call center.** La clase usa Whisper en el caso de transcripción de un call center. Es el escenario ideal para él: audio telefónico ruidoso, con acentos y jerga, exactamente el tipo de *distribution shift* donde los modelos supervisados en LibriSpeech colapsan y Whisper zero-shot brilla (la robustez efectiva de 55,2% de reducción de error). Pero también donde muerden sus limitaciones: la ventana de 30 s obliga a *buffered transcription* de llamadas largas, y el riesgo de alucinación exige verificación.

El contraste con **wav2vec 2.0** cierra la conexión con el dominio de audio del curso. Wav2vec 2.0 representa el paradigma auto-supervisado que **requiere fine-tuning por dominio**; Whisper representa el paradigma de supervisión débil a gran escala que **funciona zero-shot**. No es que uno reemplace al otro —wav2vec 2.0 aún gana en regímenes de muy pocos datos con fine-tuning— sino que Whisper redefine qué se le puede pedir a un sistema "de fábrica". Es el mismo giro que en la Clase 20 llevó del pre-entrenamiento + fine-tuning de BERT al zero-shot de GPT-3: la escala de datos y de modelo convierte una capacidad que antes exigía adaptación en una que emerge sin ella.

---

**Nota final — relevancia para salud.** Whisper es hoy la base de facto para transcripción de audio clínico y dictado médico: su robustez a ruido, acentos y jerga lo hace atractivo para transcribir consultas, rondas o dictados de especialistas sin fine-tuning por institución, y su formato multitarea permite además traducir al inglés o segmentar por actividad de voz con el mismo modelo. Pero la limitación de **alucinación** documentada en el paper —donde el modelo emite texto totalmente ajeno al audio— es especialmente peligrosa en contexto médico, y existe literatura que reporta que Whisper inserta frases inexistentes en transcripciones clínicas (incluyendo contenido inventado sobre medicamentos, tratamientos o incluso afirmaciones fabricadas), sobre todo en silencios o audio de baja calidad. En un dictado médico una alucinación no es un WER incómodo: puede introducir una indicación farmacológica o un hallazgo que nunca se dijo. Por eso, para el mismo tipo de casos que motivan mi trabajo en FHIR y linkage de registros clínicos, Whisper es una herramienta de transcripción de altísimo valor pero **nunca de escritura directa al registro sin revisión humana**: el ciclo de vida del dato de audio en salud debe cerrarse con verificación, no con confianza ciega en el modelo.
