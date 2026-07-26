---
title: "Teoría - Datasets y Herramientas para Audio"
weight: 10
math: true
---

> **Recorrido de la Clase 37** del Diplomado IA UC (Alain Raymond y Gabriel Sepúlveda), Audio 2 de 5. Si la [Clase 35](/clases/clase-35) cubrió la **teoría de señales** (Fourier, muestreo, STFT, MFCC), esta clase baja **de la teoría al dato**: el ciclo de vida práctico del audio en un proyecto de machine learning. Cuatro bloques —el audio en el disco (formatos, tamaño, ffmpeg), qué cambia al trabajar con audio (representación, transforms, batching), data augmentation (SNR, SpecAugment), y datasets (cuáles existen, cómo elegir)— y el práctico donde todo se junta: un clasificador de géneros musicales, de los WAV al modelo.

---

## 1. El audio está en todas partes

Casi todo problema de audio cae, al menos, en una de **tres familias**:

- **Música** — géneros, transcripción, separación de fuentes, generación. Ej.: Shazam.
- **Habla** — reconocer, sintetizar, identificar al hablante. Ej.: Siri.
- **Ambiente** — alarmas, máquinas, fauna, ciudad. Ej.: un detector de fallas.

En la práctica **se mezclan**: música que tapa una voz, una máquina que falla bajo el ruido de la fábrica. Tres ejemplos donde el audio ya trabaja hoy:

- **Planta industrial.** Un rodamiento gastado cambia el sonido *antes* de romperse. Se pega un micrófono, se entrena un **autoencoder** solo con el sonido "sano" y se avisa cuando algo se sale del patrón —no es un clasificador: lo anómalo se reconstruye mal, y ese **error de reconstrucción es la alerta**.
- **Banco.** La voz funciona como huella biométrica: al llamar, el modelo la compara con un patrón guardado. Se volvió urgente porque hoy se clona una voz con pocos segundos de audio.
- **Call center.** Miles de llamadas al día → transcripción (Whisper) → **diarización** (¿quién habló?) → análisis (sentimiento, resumen). El cuello de botella suele ser **separar a los hablantes**, no medir el sentimiento.

---

## 2. El audio en el disco

El audio es **un arreglo de amplitudes**: la altura de la onda muchas veces por segundo. Estéreo = **2 filas**, como los canales de una imagen RGB —exactamente lo que entrega `torchaudio.load()`. Dos números deciden calidad y tamaño (ver [digitalización de audio](/fundamentos/digitalizacion-de-audio)):

- **Sample rate** (Hz): muestras por segundo. CD = 44.100; teléfono = 8.000.
- **Bit depth**: bits por muestra. 16 bits = 65.536 niveles.

$$
\text{tamaño}_{\text{sin comprimir}} = \text{sample rate} \times \text{bits} \times \text{canales} \times \text{duración}.
$$

Ejemplo: 3 min estéreo 44,1 kHz / 16 bits ≈ 31 MB; el mismo en MP3 ≈ 3 MB (Actividad 2 del lab).

### 2.1 Formatos y ffmpeg

- **Sin comprimir** (WAV, AIFF, PCM): pesado y exacto.
- **Lossless** (FLAC, ALAC): comprime ~40% sin perder nada.
- **Lossy** (MP3, AAC, OGG): liviano, descarta lo inaudible.

{{< concept-alert type="clave" >}}
La pregunta que ordena todo: **¿la señal que me importa vive donde el oído humano escucha?** Si sí, lossy sirve. Si no (fallas sobre ~16 kHz, bioacústica, sonidos cardíacos/pulmonares, forense), lossy puede botar **justo lo que buscabas**. Se desarrolla en [Representación de audio](/fundamentos/representacion-de-audio).
{{< /concept-alert >}}

El audio real casi nunca llega en el formato que se necesita. **ffmpeg** convierte prácticamente cualquier formato —el "paso 0" de casi todo proyecto: `ffmpeg -i nota.opus -ar 16000 -ac 1 nota.wav` extrae, resamplea y deja mono. Para el dataset, la decisión de comprimir depende de si el cuello es disco o CPU; como decodificar es barato (~1.000× tiempo real), los datasets grandes se distribuyen comprimidos (LibriSpeech en FLAC, Common Voice en MP3).

---

## 3. Qué cambia al trabajar con audio

El audio es la **tercera modalidad**: señal 1D en el tiempo (como el texto, secuencial; como la imagen, se puede ver como espectrograma 2D), pero **enorme** (~miles de muestras/seg). No hay una representación única: crudo, espectrograma, MFCC o **embeddings preentrenados** (Whisper/wav2vec). **Elegir la representación es la primera decisión del pipeline.** Cambian tres frentes:

- **El muestreo.** La regla de Nyquist: para capturar una frecuencia hay que muestrear al doble. Muestrear bajo Nyquist produce **aliasing** (la frecuencia alta se disfraza de una baja, *para siempre*). A 8 kHz, Nyquist queda en 4.000 Hz: la "s" pierde sus agudos y suena como "f" —por eso deletreamos por teléfono.
- **La representación.** Las **transforms** convierten el tensor en features. Parámetros clave: `win_length` (ventana), `hop_length` (avance), `n_mels` (bandas). *La ventana es como el kernel de una convolución; el hop, como el stride.* El tensor de salida:

$$
\text{frames} = \left\lfloor \frac{N - \text{win\_length}}{\text{hop\_length}} \right\rfloor + 1 \;\Longrightarrow\; (\text{canales}, \text{n\_mels}, \text{frames}).
$$

  Una llamada de 5 min: `(1, 2.400.000)` crudas → tras un Mel de 64 bandas, `(1, 64, ~4.700)`. De archivo gigante a imagen razonable.
- **La preparación.** Resamplear no crea información (solo interpola); el largo variable se resuelve con **`collate_fn`**; hay que decidir mono vs estéreo, normalizar y recortar silencios (VAD).

{{< concept-alert type="advertencia" >}}
**La parte humana no es opcional.** Grabar voz exige **consentimiento** (la voz es dato biométrico; las llamadas tienen PII); la música casi siempre tiene **copyright**; y hay que vigilar el **sesgo** —un ASR entrenado solo en inglés "limpio" falla con los clientes reales. Buenas prácticas: anonimizar, minimizar, documentar el origen.
{{< /concept-alert >}}

---

## 4. Data augmentation

Cuando faltan datos, se **fabrican**: variantes que suenan distinto pero conservan la etiqueta. Se desarrolla en [Data augmentation de audio](/fundamentos/data-augmentation-de-audio).

{{< concept-alert type="clave" >}}
El criterio, antes de escribir código: **¿la transformación es una invariancia real del problema?** ±4 semitonos vale para clasificar género, pero **destruye** la tarea si es detectar tonalidad o identificar al hablante. Y dos reglas: augmentation **solo en train**, y **distinta en cada época**.
{{< /concept-alert >}}

- **Sumar ruido: la perilla es el SNR** (relación señal-ruido en dB): $\text{SNR} = 10\log_{10}(P_{\text{señal}}/P_{\text{ruido}})$. Se sortea en un rango (p. ej. 10-20 dB). Mejor usar ruido **real** del entorno de despliegue.
- **[SpecAugment](/papers/specaugment-park-2019)**: hacer **cero** bandas de frecuencia y tramos de tiempo del espectrograma. Es **Cutout/dropout con estructura**, casi gratis (opera sobre el espectrograma ya en la GPU).
- **`pitch_shift` / `time_stretch`**: cambian tono o duración por separado (phase vocoder, caro → offline).

El **costo decide el lugar**: ruido barato → en la `collate_fn`; SpecAugment gratis → en el loop de entrenamiento; pitch/time caros → precalculados offline.

---

## 5. Datasets

El salto de los últimos años **no fue de algoritmos: fue de escala de datos**. Hay datasets de **tres tamaños**:

- **Didácticos** — [GTZAN](/papers/gtzan-tzanetakis-2002), [UrbanSound8K](/papers/urbansound8k-salamon-2014), [ESC-50](/papers/esc50-piczak-2015), [SpeechCommands](/papers/speech-commands-warden-2018). Para aprender (es lo del lab).
- **Benchmarks** — [LibriSpeech](/papers/librispeech-panayotov-2015), [MusicNet](/papers/musicnet-thickstun-2017), [AudioSet](/papers/audioset-gemmeke-2017), [FSD50K](/papers/fsd50k-fonseca-2020). Estándares de investigación.
- **Escala web** — [Common Voice](/papers/common-voice-ardila-2020), GigaSpeech, Emilia. La frontera: **[Whisper](/papers/whisper-radford-2022)** usó ~680.000 horas de audio web débilmente etiquetado.

**Cómo elegir** (cuatro ejes): escala vs. limpieza, disponibilidad (¿te dan el audio, o solo un link de YouTube que se cae?), licencia y privacidad, y representación/sesgo. Se desarrolla en [Datasets de audio](/fundamentos/datasets-de-audio). Un dato de terreno: no existe un dataset público de llamadas chilenas —el flujo real es **pre-entrenar con datos públicos y hacer fine-tune con los datos propios**.

---

## 6. Cierre

La clase recorrió el **ciclo de vida del dato de audio**: del archivo en disco (formatos, tamaño, ffmpeg) al tensor (transforms, batching), pasando por la augmentation (SNR, SpecAugment) y la elección del dataset (escala, licencia, sesgo). Es el puente entre la teoría de señales de la Clase 35 y los modelos de audio de las clases siguientes. El [laboratorio](/laboratorios/lab-37) lo junta todo en un clasificador de géneros sobre GTZAN, de los WAV al modelo, con un anexo de embeddings preentrenados (wav2vec 2.0).

---

**Ver también:** [Clase 37 - Profundización](/clases/clase-37/profundizacion) · [Clase 37 - Práctica](/clases/clase-37/practica) · Fundamentos: [Representación de audio](/fundamentos/representacion-de-audio) · [Datasets de audio](/fundamentos/datasets-de-audio) · [Data augmentation de audio](/fundamentos/data-augmentation-de-audio) · [Dominio: Audio](/dominios/audio).
