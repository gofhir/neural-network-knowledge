---
title: "Profundización - Datasets y Herramientas para Audio"
weight: 20
math: true
---

> **Desarrollo formal de la Clase 37.** La [teoría](/clases/clase-37/teoria) recorre el ciclo de vida del dato de audio de forma narrativa; aquí se formalizan los cálculos y algoritmos. Cinco partes: (1) tamaño y tensor; (2) la relación señal-ruido; (3) SpecAugment; (4) los embeddings autosupervisados (wav2vec 2.0); (5) la ley de escala de la weak supervision (Whisper).

---

## 1. Del archivo al tensor: los cálculos

### 1.1 Tamaño en disco

El audio sin comprimir ocupa, en bits:

$$
\text{tamaño} = f_s \times b \times c \times t,
$$

con $f_s$ el sample rate (Hz), $b$ el bit depth (bits/muestra), $c$ los canales y $t$ la duración (s). Para 3 min estéreo a 44,1 kHz / 16 bits: $44100 \times 16 \times 2 \times 180 \approx 2{,}5 \times 10^9$ bits $\approx 31$ MB. La compresión **lossless** (FLAC) reduce ~40% sin pérdida; la **lossy** (MP3) llega a ~1/10 descartando lo inaudible.

### 1.2 Dimensiones del tensor de features

Una señal de $N$ muestras, transformada con ventana `win_length` $= W$ y salto `hop_length` $= H$, produce un número de **frames** análogo al output de una convolución 1D:

$$
\text{frames} = \left\lfloor \frac{N - W}{H} \right\rfloor + 1.
$$

Con `n_mels` $= M$ bandas Mel, el tensor de salida es $(c, M, \text{frames})$: una imagen 2D por canal (frecuencia × tiempo). La analogía es exacta —$W$ es el **kernel**, $H$ es el **stride**— y el trade-off es el de Gabor ([representación tiempo-frecuencia](/fundamentos/representacion-tiempo-frecuencia)): $W$ grande da mejor resolución en frecuencia y peor en tiempo.

---

## 2. La relación señal-ruido (SNR)

La cantidad de ruido en una augmentation se calibra por **potencias**, no por amplitud. La potencia de una señal es el promedio de su amplitud al cuadrado, $P = \overline{x^2}$, y la SNR es su cociente en decibeles:

$$
\text{SNR} = 10\log_{10}\!\left(\frac{P_{\text{señal}}}{P_{\text{ruido}}}\right)\;[\text{dB}].
$$

La escala es **logarítmica**: cada 10 dB es un factor 10 en potencia. Para agregar ruido a un SNR objetivo, se **escala** la señal de ruido $n$ por un factor $\alpha$ tal que la mezcla $x + \alpha n$ tenga la SNR pedida:

$$
\alpha = \sqrt{\frac{P_{\text{señal}}}{P_{\text{ruido}} \cdot 10^{\text{SNR}/10}}}.
$$

En entrenamiento, la SNR se **sortea** en un rango (p. ej. $\mathcal{U}(10, 20)$ dB) para que cada época suene distinta. Detalle en [Data augmentation de audio](/fundamentos/data-augmentation-de-audio).

---

## 3. SpecAugment: augmentation sobre el espectrograma

[SpecAugment](/papers/specaugment-park-2019) (Park et al., 2019) opera directamente sobre el espectrograma log-Mel $S \in \mathbb{R}^{M \times T}$ (frecuencia × tiempo), con tres deformaciones:

1. **Frequency masking.** Se sortea un ancho $f \sim \mathcal{U}(0, F)$ y un inicio $f_0 \sim \mathcal{U}(0, M - f)$, y se hacen **cero** las filas $[f_0, f_0 + f)$:
$$
S[f_0 : f_0 + f, \; :] = 0.
$$
2. **Time masking.** Análogo sobre columnas: $t \sim \mathcal{U}(0, T_{\max})$, $t_0 \sim \mathcal{U}(0, T - t)$, y $S[:, \; t_0 : t_0 + t] = 0$.
3. **Time warping.** Una deformación temporal suave (menos influyente, y la más cara).

Como el espectrograma se normaliza a media cero, **poner a cero equivale a insertar la media** —una oclusión neutra. Es **Cutout/dropout con estructura**: apaga regiones contiguas, no unidades sueltas, forzando al modelo a no depender de una sola banda o instante. Y es casi gratis: opera sobre el tensor que **ya está en la GPU**. En LibriSpeech convirtió el sobreajuste en subajuste, logrando estado del arte sin modelo de lenguaje.

---

## 4. Embeddings autosupervisados: wav2vec 2.0

En vez de features hechas a mano (MFCC), se pueden usar **embeddings preentrenados**. [wav2vec 2.0](/papers/wav2vec2-baevski-2020) (Baevski et al., 2020) los aprende sin etiquetas:

1. Un **encoder convolucional** mapea el audio crudo a representaciones latentes $z_t$.
2. Estas se **cuantizan** a un codebook discreto $q_t$ (vía Gumbel-softmax).
3. Se **enmascaran** segmentos de los latentes (estilo BERT) y un **Transformer** produce contextos $c_t$.
4. Una **pérdida contrastiva** (InfoNCE) empuja a $c_t$ a identificar la unidad cuantizada correcta $q_t$ entre distractores $\tilde q$:

$$
\mathcal{L}_m = -\log \frac{\exp\!\big(\text{sim}(c_t, q_t)/\kappa\big)}{\sum_{\tilde q \sim Q} \exp\!\big(\text{sim}(c_t, \tilde q)/\kappa\big)},
$$

con $\text{sim}$ la similitud coseno y $\kappa$ una temperatura, más una **pérdida de diversidad** que fuerza a usar todo el codebook. El resultado: con ~53.000 h sin etiquetar de pretraining + tan poco como **10 minutos** de audio etiquetado se alcanza un WER competitivo —el paradigma "pretraining masivo autosupervisado + fine-tuning ligero". Conecta con el [aprendizaje contrastivo](/fundamentos/aprendizaje-contrastivo) y [autosupervisado](/fundamentos/aprendizaje-autosupervisado).

---

## 5. Weak supervision a escala: Whisper

[Whisper](/papers/whisper-radford-2022) (Radford et al., 2022) toma el camino opuesto a wav2vec 2.0: en vez de audio *sin* etiquetar + fine-tuning, usa **680.000 horas de audio web con transcripciones débiles** (encontradas, no verificadas) y un encoder-decoder Transformer estándar sobre log-Mel. La escala convierte lo "sucio" en robustez: el modelo generaliza **zero-shot** a ~99 idiomas, ruido y acentos, sin fine-tuning. El formato **multitarea** (transcripción, traducción, detección de idioma, timestamps) se resuelve con **tokens especiales** en la secuencia de salida de un único modelo.

{{< concept-alert type="clave" >}}
Whisper y wav2vec 2.0 son los dos polos del **eje escala vs. limpieza** ([datasets de audio](/fundamentos/datasets-de-audio)): wav2vec explota audio *sin etiquetas* con una pérdida autosupervisada y luego afina; Whisper explota *muchísimas* etiquetas *débiles* y funciona directo. Ambos superan al ASR supervisado clásico —limitado por la escasez de datos etiquetados limpios— por caminos opuestos.
{{< /concept-alert >}}

---

## 6. Síntesis

El ciclo de vida del dato de audio, en fórmulas: una señal se **dimensiona** ($f_s \times b \times c \times t$), se **transforma** a un tensor 2D ($\lfloor (N-W)/H\rfloor + 1$ frames × $M$ bandas), se **aumenta** calibrando el ruido por SNR y tapando el espectrograma (SpecAugment), y se representa —crudo, MFCC o **embeddings** de un modelo autosupervisado (wav2vec) o de weak supervision a escala (Whisper). Cada decisión —formato, sample rate, representación, augmentation, dataset— es un compromiso entre fidelidad, costo y sesgo, y todas juntas determinan si el modelo funcionará en el mundo real.

---

**Ver también:** [Clase 37 - Teoría](/clases/clase-37/teoria) · [Clase 37 - Práctica](/clases/clase-37/practica) · Fundamentos: [Representación de audio](/fundamentos/representacion-de-audio) · [Datasets de audio](/fundamentos/datasets-de-audio) · [Data augmentation de audio](/fundamentos/data-augmentation-de-audio).
