---
title: "Whisper: ASR robusto por weak supervision (2022)"
weight: 407
math: true
---

{{< paper-card
    title="Robust Speech Recognition via Large-Scale Weak Supervision"
    authors="Alec Radford et al. (OpenAI)"
    year="2022"
    venue="arXiv:2212.04356"
    pdf="/papers/whisper-radford-2022.pdf" >}}
Whisper es la crónica de qué capacidades emergen cuando se entrena un sistema de voz de la forma más simple posible: **predecir grandes cantidades de transcripciones recolectadas de internet**. La apuesta es de **escala de datos**: llevar la supervisión débil hasta **680.000 horas** de audio multilingüe y multitarea, entrenar un solo **Transformer encoder-decoder** y enfocarse en **transferencia zero-shot**, sin ningún fine-tuning. El resultado es un sistema que funciona *out of the box* frente a ruido, acentos, idiomas y dominios que nunca vio explícitamente, y que se aproxima a la exactitud de transcriptores humanos profesionales. Frente a un modelo supervisado con desempeño casi idéntico en LibriSpeech test-clean, Whisper **comete en promedio 55,2% menos errores** en otros 12 datasets. Es *el* ejemplo de la frontera de escala del dato de audio en la [Clase 37](/clases/clase-37).
{{< /paper-card >}}

---

## Contexto: el techo del supervisado clásico y del auto-supervisado

Hacia 2022 el campo del reconocimiento de voz (ASR) se dividía en dos líneas, ambas con techo. **La supervisión de alta calidad era escasa**: los datasets académicos con transcripciones validadas por humanos rondan las **1.000 horas**, y combinaciones como SpeechStew (7 corpus) llegan solo a **5.140 horas**. Peor aún, un modelo entrenado en un único dataset como LibriSpeech alcanza desempeño *sobrehumano* dentro de su distribución (WER de 1,4% frente a 5,8% humano) pero se vuelve frágil al cambiar de dominio: aprende las idiosincrasias del dataset, no a "escuchar".

La segunda línea, la **auto-supervisión escalable** de [wav2vec 2.0](/papers/wav2vec2-baevski-2020), aprende representaciones del audio crudo sin etiquetas y escala a millones de horas, pero tiene una debilidad estructural: al ser puramente no supervisada **carece de un decoder competente** y **exige una etapa de fine-tuning** por cada distribución de despliegue. Ese fine-tuning es complejo (requiere un experto) y arrastra el mismo riesgo de fragilidad. La conclusión de los autores es que un sistema de voz debería **funcionar confiablemente sin fine-tuning por dominio**; el único ingrediente que faltaba era la escala.

## Contribución: escala + supervisión débil + zero-shot

Whisper cierra la brecha con tres apuestas combinadas. **Escala de supervisión débil**: en vez de exigir transcripciones humanas verificadas, usa audio web emparejado con transcripciones *tal como están*, aceptando ruido en las etiquetas a cambio de cantidad — **680.000 horas**, un orden de magnitud sobre el trabajo previo. **Multilingüe y multitarea**: de esas horas, **117.000 cubren otros 96 idiomas** y **125.000 son datos de traducción X→inglés** (cerca de un tercio del dataset es no-inglés). **Zero-shot en vez de fine-tuning**: el modelo transfiere a datasets existentes sin adaptación específica, eliminando la etapa que hacía frágiles a los sistemas auto-supervisados. Todo se logra **sin** técnicas de auto-supervisión ni self-training: solo escala de supervisión débil.

## Método: pipeline de datos y Transformer multitarea

El corazón práctico del paper es cómo se construye el dataset — lo más relevante para la [Clase 37](/clases/clase-37). La filosofía es **minimalista en el pre-procesamiento**: se predice el texto crudo, sin normalización, confiando en que el seq2seq aprenda a mapear el habla a su forma escrita naturalista. Sobre el audio web se aplican **filtros automatizados**: detección y remoción de **transcripciones generadas por máquina** (heurística: texto todo en mayúsculas/minúsculas o sin puntuación delata origen ASR), **coincidencia de idioma audio-texto** (si el audio no es inglés pero la transcripción sí, el par se recicla como ejemplo de **traducción X→inglés**), y **de-duplicación difusa**. Una segunda pasada agrega la tasa de error por fuente y remueve manualmente las de baja calidad. El audio se parte en **segmentos de 30 segundos**.

La arquitectura es deliberadamente **estándar** para no confundir los hallazgos con mejoras de modelo: un Transformer encoder-decoder. El front-end re-muestrea a **16 kHz** y computa un **espectrograma log-Mel de 80 canales** sobre ventanas de 25 ms con paso de 10 ms — los mismos *transforms* que la clase enseña como puente entre onda y tensor. El multitarea se resuelve con **tokens especiales** en el decoder: un token de idioma (99 en total), un token de tarea (`<|transcribe|>` o `<|translate|>`), y opcionalmente timestamps cuantizados a 20 ms. Así una sola arquitectura y un objetivo de *next-token prediction* resuelven transcripción multilingüe, traducción, identificación de idioma, VAD y alineación temporal. La familia escala de **39 M** (Tiny) a **1550 M** de parámetros (Large).

## Resultados

La **robustez efectiva** es el resultado estrella. El mejor Whisper zero-shot tiene un WER de **2,5** en LibriSpeech test-clean, nada espectacular; pero frente a un modelo supervisado con desempeño casi idéntico ahí, **comete 55,2% menos errores en promedio** sobre los otros 12 datasets. Incluso el **Tiny de 39 M** es competitivo con el mejor supervisado de LibriSpeech al evaluarse fuera de él. En **traducción** X→inglés (CoVoST2) alcanza un nuevo estado del arte zero-shot de **29,1 BLEU**. En **robustez al ruido**, bajo *pub noise* con SNR menor a 10 dB Whisper supera a todos los modelos LibriSpeech. Comparado con 5 servicios de transcripción profesional, su WER queda a **una fracción de punto** del de transcriptores humanos. Sobre Fleurs aparece una ley cuantitativa: la correlación entre $\log(\text{WER})$ y $\log(\text{datos por idioma})$ es $r^2 = 0{,}83$, y el WER **se reduce a la mitad por cada aumento de 16×** en datos.

## Limitaciones

- **Alucinaciones.** Al escalar, los errores perceptuales se reducen, pero persisten fallos "no humanos": loops de repetición y, lo más grave, **alucinación completa**, donde el modelo emite una transcripción totalmente no relacionada con el audio. Las heurísticas de decodificación (beam search, *temperature fallback*) son parches.
- **Idiomas de bajos recursos.** El pipeline es fuertemente **sesgado al inglés** y la mayoría de idiomas tiene menos de 1.000 horas; la identificación de idioma zero-shot (64,5% en Fleurs) no es competitiva.
- **Ventana de 30 segundos.** No puede consumir audio más largo de una vez; la transcripción larga exige *buffered transcription* con heurísticas frágiles, donde un error en una ventana contamina la siguiente.

## Por qué importa para la Clase 37

La [Clase 37](/clases/clase-37) recorre el ciclo de vida del dato de audio, y Whisper es la ilustración de que **la robustez de un modelo de audio se decide en el dato, no en la arquitectura**: los autores usan un Transformer deliberadamente estándar y toda la diferencia la hacen las 680.000 horas y su pipeline de filtrado. Cada eje de la clase tiene su correlato: el log-Mel de 80 canales es el *transform* de [representación](/fundamentos/datasets-de-audio); la evaluación de WER vs. SNR justifica la data augmentation por ruido; y el paper es el caso paradigmático de la **frontera de escala** — no usó ninguno de los [datasets clásicos](/fundamentos/datasets-de-audio), solo los usó para *evaluar* zero-shot.

El contraste con [wav2vec 2.0](/papers/wav2vec2-baevski-2020) cierra la conexión con el [dominio audio](/dominios/audio): wav2vec 2.0 encarna el paradigma auto-supervisado que **requiere fine-tuning por dominio**; Whisper, la supervisión débil a escala que **funciona zero-shot**. No es que uno reemplace al otro —wav2vec 2.0 aún gana en regímenes de muy pocos datos con fine-tuning— sino que Whisper redefine qué se le puede pedir a un sistema "de fábrica", el mismo giro que en NLP llevó del fine-tuning de BERT al zero-shot de GPT-3. En salud, Whisper es hoy base de facto para dictado médico, pero su alucinación documentada lo vuelve una herramienta de altísimo valor que **nunca debe escribir directo al registro sin revisión humana**.
