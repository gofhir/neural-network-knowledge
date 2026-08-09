---
title: "Lab 37 - Datasets y Herramientas para Audio"
weight: 370
sidebar:
  open: true
---

**Profesores:** Alain Raymond · Gabriel Sepúlveda
**Módulo:** Audio y Video (parte de audio)
**Notebook origen:** `clase_37/material/Laboratorio/Practico_37_DINTA.ipynb`
**Notebook ejecutado:** [lab37.ipynb](/notebooks/lab37.ipynb) · [HTML](/notebooks-html/lab37.html)

## Encuadre

La contraparte práctica de la [clase 37](/clases/clase-37): **el ciclo de vida del dato de audio**, de un arreglo de amplitudes hasta un clasificador entrenado. La primera mitad recorre las herramientas —cargar, resamplear, convertir formatos, transformar a features, aumentar— y la segunda las aplica a un problema real: **clasificar géneros musicales en GTZAN** con una GRU, comparando features de fórmula (**MFCC**) contra embeddings preentrenados (**wav2vec 2.0**).

El lab plantea esa comparación como su pregunta central, y la respuesta que dan los números no es la que sugiere el enunciado. Wav2vec gana, pero **no principalmente por lo que aprendió del habla**: gana porque sus features vienen normalizadas y las del MFCC no, y esa diferencia de escala satura la red desde la inicialización.

## Resultados consolidados (medidos en el notebook)

### GTZAN — la matriz que separa dos efectos

|  | MFCC limpio | MFCC + ruido | wav2vec limpio | wav2vec + ruido |
|---|---|---|---|---|
| **Train** | 20,09% | 39,95% | **35,21%** | 39,28% |
| **Test** | **19,31%** | 31,03% | **28,62%** | 31,38% |
| Brecha train−test | +0,78 pp | +8,92 pp | +6,59 pp | +7,90 pp |

Con 10 clases balanceadas, el azar es 10%. Loss final: 2,152 (MFCC) contra 1,995 (wav2vec), sobre una loss de azar de $\ln(10) = 2{,}303$.

### La causa: saturación de la GRU

|  | MFCC | wav2vec |
|---|---|---|
| Norma del vector de entrada | 204,7 (limpio) / 267,3 (ruido) | **9,3 / 9,1** |
| Coeficiente dominante | $c_0 \approx 133$–$258$ (la energía) | ninguno |
| Preactivación de la GRU | **11,8 – 15,4** | **0,5** |
| Derivada de $\tanh$ (gradiente que fluye) | $10^{-10}$ – $10^{-13}$ | **0,79** |

### El gradiente de transferencia

| Situación | Coincidencia con el preentrenamiento de wav2vec | Resultado |
|---|---|---|
| GTZAN — música, 22 kHz | ninguna | **+9,31 pp** sobre MFCC |
| Speech Commands — habla en inglés, 16 kHz | idioma y modalidad | **+33,66 pp** sobre MFCC (79,83% vs 46,17%) |
| VoxPopuli ES — habla en español | modalidad, otro idioma y acento | ~3 errores en ~80 palabras |
| LibriSpeech `dev-clean` | total: corpus, idioma, tasa, estilo | **0,0% WER** |

### Las lecciones del lab

1. **El sample rate no vive en los datos.** Un audio son números; la tasa es metadata que viaja aparte. Cambiar el `rate` de reproducción altera tono y duración sin tocar una muestra; **resamplear** conserva el sonido y destruye información de forma irreversible.
2. **La STFT es una convolución 1D.** `n_fft` es el receptive field, `hop_length` el stride, `center` el padding — y los frames se calculan con la misma fórmula. Los defaults muerden: `hop = n_fft//2` y `center=True` convirtieron 108 frames en 56 sobre la misma señal.
3. **Cuatro maneras de "acelerar" un audio, y solo dos sirven como augmentation.** Reproducir a otro rate y resamplear acoplan tono y duración; `time_stretch` y `pitch_shift` los desacoplan, y por eso son las útiles.
4. **La augmentación aplicada siempre deja de ser augmentación.** El `collate_fn` suma ruido al 100% de las muestras de train, así que el modelo aprendió "música con SNR de 10-20 dB" y evaluarlo en audio limpio lo saca de su distribución: **+11,7 puntos de diferencia** entre test con ruido y test limpio.
5. **La escala de las features decide si la red puede aprender.** Los MFCC sin normalizar llegan a la GRU con norma ~205 y la saturan (gradiente de orden $10^{-13}$). Los embeddings de wav2vec llegan con norma ~9 y caen en la zona lineal. Es la diferencia entre 20,09% y 35,21% en train.
6. **El valor de un modelo preentrenado es función de la distancia al dominio.** El mismo wav2vec vale +9 puntos en música, +34 en habla en inglés, y transcribe con 0% de error el corpus con el que fue entrenado.

## Bloques del lab

{{< cards >}}
  {{< card link="01-el-dato-de-audio" title="El dato de audio" subtitle="De un arreglo de amplitudes al tensor de features: canales, sample rate, el caso del teléfono, formatos y peso, MFCC y Mel. Las actividades 1, 2 y 3" icon="variable" >}}
  {{< card link="02-data-augmentation" title="Data augmentation" subtitle="Las cuatro maneras de acelerar un audio, el phase vocoder y por qué las fases se regeneran, SpecAugment y el SNR verificado con su definición" icon="adjustments" >}}
  {{< card link="03-gtzan-mfcc-vs-wav2vec" title="GTZAN: MFCC vs wav2vec" subtitle="El pipeline completo, el batch que falla, la GRU, las dos corridas y el diagnóstico de saturación que explica la diferencia. La actividad 4" icon="chart-bar" >}}
  {{< card link="04-transferencia-y-dominio" title="Transferencia y dominio" subtitle="Embeddings de wav2vec, transcripción CTC en español y en inglés, y el gradiente que va de +9 puntos a 0% de WER según la distancia al preentrenamiento" icon="sparkles" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/representacion-de-audio" title="Representación de audio" subtitle="Del archivo en disco al tensor: sample rate, bit depth, canales, formatos" icon="book-open" >}}
  {{< card link="/fundamentos/datasets-de-audio" title="Datasets de audio" subtitle="Las tres escalas, cómo elegir, y la parte humana del dato" icon="book-open" >}}
  {{< card link="/fundamentos/data-augmentation-de-audio" title="Data augmentation de audio" subtitle="SNR, SpecAugment, pitch y time stretch, y dónde ubicarlas en el pipeline" icon="book-open" >}}
  {{< card link="/fundamentos/mfcc-y-escala-mel" title="MFCC y escala Mel" subtitle="Por qué las bandas Mel, la DCT y qué información conserva cada coeficiente" icon="book-open" >}}
  {{< card link="/fundamentos/representacion-tiempo-frecuencia" title="Representación tiempo-frecuencia" subtitle="STFT, ventanas, y el compromiso de Gabor entre resolución temporal y frecuencial" icon="book-open" >}}
  {{< card link="/fundamentos/ctc-loss" title="CTC" subtitle="El blank, el colapso de repeticiones y el decoding greedy — el mismo algoritmo que en scene text" icon="book-open" >}}
{{< /cards >}}

## Papers de este laboratorio

{{< cards >}}
  {{< card link="/papers/gtzan-tzanetakis-2002" title="GTZAN (2002)" subtitle="Tzanetakis y Cook — el dataset de la Parte 2 y el origen de la clasificación automática de géneros" icon="document-text" >}}
  {{< card link="/papers/wav2vec2-baevski-2020" title="wav2vec 2.0 (2020)" subtitle="Baevski et al. — los embeddings preentrenados que compiten contra los MFCC" icon="document-text" >}}
  {{< card link="/papers/specaugment-park-2019" title="SpecAugment (2019)" subtitle="Park et al. — enmascarar filas y columnas del espectrograma" icon="document-text" >}}
  {{< card link="/papers/urbansound8k-salamon-2014" title="UrbanSound8K (2014)" subtitle="Salamon et al. — sonidos urbanos, los 10 folds y el desbalance de las clases impulsivas" icon="document-text" >}}
  {{< card link="/papers/librispeech-panayotov-2015" title="LibriSpeech (2015)" subtitle="Panayotov et al. — el corpus con el que se preentrenó wav2vec 2.0" icon="document-text" >}}
  {{< card link="/papers/speech-commands-warden-2018" title="Speech Commands (2018)" subtitle="Warden — el dataset donde el preentrenamiento sí rinde: +33,7 puntos" icon="document-text" >}}
{{< /cards >}}

---

**Ver tambien:** [Clase 37 - Teoría](/clases/clase-37/teoria) · [Clase 37 - Profundización](/clases/clase-37/profundizacion) · [Clase 35 - Análisis de Audio](/clases/clase-35) (Fourier, STFT, MFCC) · Dominio [Audio](/dominios/audio).
