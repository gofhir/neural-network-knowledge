---
title: "Representación de audio para ML"
weight: 123
math: true
---

El audio es la **tercera modalidad** del deep learning, después del texto y la imagen —y toma prestado de ambas. La [Clase 35](/clases/clase-35) cubrió la teoría de señales (Fourier, muestreo, STFT, MFCC); este fundamento, que acompaña a la [Clase 37](/clases/clase-37), adopta la perspectiva **práctica del pipeline de machine learning**: cómo pasa el audio del archivo en disco al **tensor** que entra al modelo, qué representación elegir, y los detalles que hacen que un dataset de audio funcione (o falle) en un entrenamiento real.

---

## 1. El audio en el disco: un arreglo de amplitudes

Lo que queda guardado es simplemente **la altura de la onda muchas veces por segundo**: una secuencia de números (amplitudes). Dos parámetros deciden calidad y tamaño (ver [digitalización de audio](/fundamentos/digitalizacion-de-audio)):

- **Sample rate** (Hz): muestras por segundo. CD = 44.100 Hz; teléfono = 8.000 Hz.
- **Bit depth**: bits por muestra. 16 bits = 65.536 niveles de amplitud.

El tamaño sin comprimir es directo:

$$
\text{tamaño} = \text{sample rate} \times \text{bits} \times \text{canales} \times \text{duración}.
$$

Un audio estéreo de 3 min a 44,1 kHz / 16 bits pesa ≈ 31 MB; el mismo en MP3, ≈ 3 MB. Una llamada telefónica (8 kHz, 16 bits, mono) pesa ≈ 1 MB por minuto.

Los **canales** son como los de una imagen RGB: estéreo = **2 filas**. `torchaudio.load()` entrega exactamente un tensor `(canales, muestras)`.

### 1.1 Formatos: ¿comprimir o no?

- **Sin comprimir** (WAV, AIFF, PCM): pesado y exacto —el "BMP" del audio.
- **Lossless** (FLAC, ALAC): comprime ~40% **sin perder nada**.
- **Lossy** (MP3, AAC, OGG): liviano, **descarta lo inaudible** para el oído humano.

{{< concept-alert type="clave" >}}
La pregunta que ordena todo: **¿la señal que me importa vive donde el oído humano escucha?** Si sí (clasificar habla o música, transcribir), el formato lossy sirve —lo que MP3 botó tampoco lo necesitaba el modelo. Si no (fallas de máquinas sobre ~16 kHz, bioacústica, ultrasonido, sonidos cardíacos/pulmonares, audio forense), lo lossy puede botar **justo lo que buscabas**.
{{< /concept-alert >}}

El audio real casi nunca llega en el formato que se necesita (reuniones en `.mp4`, WhatsApp en `.opus`, notas en `.m4a`). La herramienta universal es **ffmpeg** —lee casi cualquier formato y lo convierte; es el "paso 0" de casi todo proyecto de audio:

```bash
ffmpeg -i reunion.mp4 audio.wav              # extrae el audio de un video
ffmpeg -i nota.opus -ar 16000 -ac 1 nota.wav  # y de paso resamplea y deja mono
```

---

## 2. El audio como tercera modalidad

El audio combina rasgos del texto y la imagen:

| | Texto | Imagen | **Audio** |
|---|---|---|---|
| Forma del dato | tokens en secuencia | grilla 2D de píxeles | **señal 1D en el tiempo** |
| Tamaño típico | corto | medio | **enorme (~miles/seg)** |
| ¿Secuencial? | sí | no | **sí** |
| Pre-entrenado | un LLM | ResNet/ViT | **Whisper / wav2vec** |

{{< concept-alert type="clave" >}}
**No hay una representación única.** El audio se trabaja a veces **crudo** (la forma de onda), a veces como **espectrograma** (y ahí es una imagen, apta para CNN), como **MFCC**, o como **embeddings preentrenados** ([wav2vec 2.0](/papers/wav2vec2-baevski-2020), [Whisper](/papers/whisper-radford-2022)). **Elegir la representación es la primera decisión del pipeline.**
{{< /concept-alert >}}

---

## 3. De la onda al tensor: las transforms

`torchaudio.load()` da el tensor de amplitudes; las **transforms** lo convierten en *features*. Los parámetros clave —los de la STFT y el banco Mel— son:

- **`win_length`**: tamaño de la ventana.
- **`hop_length`**: cada cuánto avanza la ventana.
- **`n_mels`**: cuántas bandas de frecuencia (Mel).

{{< concept-alert type="recordar" >}}
La analogía que conecta con las CNN: **la ventana es como el kernel de una convolución; el hop, como el stride.** Y el trade-off es el de Gabor ([representación tiempo-frecuencia](/fundamentos/representacion-tiempo-frecuencia)): ventana **grande** = mejor resolución en frecuencia, peor en tiempo. Ventana **corta** si importa el *cuándo* (golpes, consonantes); ventana **larga** si hay que separar frecuencias parecidas (notas, armónicos). En habla, lo típico es ~25 ms.
{{< /concept-alert >}}

### 3.1 El tensor que entra al modelo

Calcular sus dimensiones es como calcular el output de una convolución. Con $N$ muestras:

$$
\text{frames} = \left\lfloor \frac{N - \text{win\_length}}{\text{hop\_length}} \right\rfloor + 1 \quad\Longrightarrow\quad (\text{canales},\; \text{n\_mels},\; \text{frames}).
$$

El resultado es una **matriz 2D por canal**: un eje es frecuencia (`n_mels`), el otro es tiempo (`frames`). *Si es una imagen... se le puede tirar una CNN.* Ejemplo de la clase: una llamada de 5 min son `(1, 2.400.000)` muestras crudas; tras un Mel de 64 bandas queda `(1, 64, ~4.700)` —de archivo gigante a imagen razonable.

---

## 4. Detalles que conviene cuidar antes de entrenar

- **Resamplear no crea información.** Subir el sample rate solo **interpola** —como hacer zoom a una foto: no aparece detalle nuevo.
- **Largo variable → batching.** El audio no viene de tamaño fijo, y PyTorch no arma el batch solo: se resuelve con una **`collate_fn`** (recortar/rellenar al mismo largo).
- **Mono vs estéreo.** ¿Promediar los canales o no? A veces se pierde información espacial que importaba.
- **Normalización y silencios.** Conviene normalizar amplitudes y recortar silencios (VAD, *voice activity detection*); las etiquetas pueden no cubrir todo el audio.

---

## 5. La parte humana

Un pipeline de audio real no es solo técnico. La clase insiste en cuatro frentes: **consentimiento y privacidad** (grabar voz exige consentimiento; la voz es dato biométrico y las llamadas tienen PII), **copyright** (casi toda la música tiene derechos; lo abierto suele ser Creative Commons o instrumental), **sesgo** (¿el dataset cubre acentos, géneros e idiomas? —un ASR entrenado solo en inglés "limpio" falla con clientes reales), y las **buenas prácticas** (anonimizar, minimizar, documentar el origen de los datos). Se desarrollan en [Datasets de audio](/fundamentos/datasets-de-audio).

---

## 6. Relevancia para salud

Para software clínico que procesa voz o sonido, cada decisión de representación tiene consecuencias. El **formato** importa de forma crítica: un soplo cardíaco o una firma de falla respiratoria pueden vivir por encima de lo que el oído humano prioriza, así que un códec lossy optimizado para percepción puede **botar justo la señal diagnóstica** —regla de oro: guardar el original sin pérdida. El **sample rate** debe respetar Nyquist para la banda de interés clínico. La **elección de representación** (MFCC interpretables vs. embeddings preentrenados vs. espectrograma para CNN) condiciona tanto el desempeño como la auditabilidad. Y la **parte humana** —consentimiento, PII, sesgo demográfico— es, en salud, una obligación legal y ética, no una recomendación.

---

## Referencias

- Fundamentos relacionados: [Digitalización de audio](/fundamentos/digitalizacion-de-audio) · [Representación tiempo-frecuencia](/fundamentos/representacion-tiempo-frecuencia) · [MFCC y escala Mel](/fundamentos/mfcc-y-escala-mel) · [Datasets de audio](/fundamentos/datasets-de-audio) · [Data augmentation de audio](/fundamentos/data-augmentation-de-audio).
- Dominio: [Audio / Voz](/dominios/audio).
