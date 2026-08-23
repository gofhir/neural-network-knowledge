---
title: "01 - El dataloader y el eje de la normalización"
weight: 10
math: true
---

> Tres funciones en cascada, cuatro parámetros muertos, un espejado que duplica el cómputo del lab entero para cambiar el 0,02 % del resultado, y un eje de normalización que el grupo de Oxford invirtió entre 2017 y 2019 después de haberlo llamado *crucial*.

---

## 1. El código, y de dónde viene

```python
def load_wav( vid_path, sr ):
  wav, sr_ret = librosa.load( vid_path, sr=sr )
  extended_wav = np.append(wav, wav[::-1])
  return extended_wav

def lin_spectogram_from_wav( wav, hop_length, win_length, n_fft = 1024 ):
  linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
  return linear.T

def load_data( path, win_length = 400, sr = 16000, hop_length = 160, n_fft = 512, spec_len = 250 ):
  wav = load_wav(path, sr=sr)
  linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
  mag, _ = librosa.magphase(linear_spect)  # magnitude
  mag_T = mag.T
  freq, time = mag_T.shape
  spec_mag = mag_T
  # preprocessing, subtract mean, divided by time-wise var
  mu = np.mean(spec_mag, axis = 0, keepdims=True)
  std = np.std(spec_mag, axis = 0, keepdims=True)
  return (spec_mag - mu) / (std + 1e-5)
```

Es un port literal de `utils.py` del repo [VGG-Speaker-Recognition](https://github.com/WeidiXie/VGG-Speaker-Recognition) de Weidi Xie, **al que le amputaron la rama de entrenamiento**. Casi todo lo raro que hay aquí es cicatriz de esa amputación:

| Residuo | Qué hacía en el original |
|---|---|
| `spec_len = 250` | el largo del crop aleatorio de 2,5 s |
| `freq, time = mag_T.shape` | alimentaba `np.random.randint(0, time - spec_len)` |
| `spec_mag = mag_T` | era la rama `else` de un `if mode == 'train'` |
| `np.append(wav, wav[::-1])` | garantizaba largo mínimo **para ese crop** |
| el `.T` de una función y el `.T` de la otra | se cancelan; venían de la convención de ejes de Keras |
| `n_fft = 1024` por defecto | letra muerta: `load_data` siempre pasa 512 |
| `import sys` (celda 3) | nunca se usa |

El original bifurcaba así:

```python
if mode == 'train':
    randtime = np.random.randint(0, time - spec_len)
    spec_mag = mag_T[:, randtime : randtime + spec_len]   # crop de 250 frames
else:
    spec_mag = mag_T                                       # <-- lo unico que quedo
```

**Consecuencia arquitectónica de que no haya crop:** cada audio produce un espectrograma de largo distinto, y eso fuerza `batch_size = 1` en la extracción de features. No es una elección de estilo: no se pueden apilar tensores de anchos diferentes sin *padding*, y rellenar exigiría enmascarar hasta dentro del softmax de VLAD, porque los frames de relleno recibirían masa de asignación y contaminarían la suma.

---

## 2. El espejado: 2× de cómputo por 0,02 % de resultado

```python
extended_wav = np.append(wav, wav[::-1])
```

Se concatena la señal con su **reverso temporal**. Un audio de 8 s pasa a tener 16.

**Para qué era.** El paper entrena con crops de 2,5 s (`spec_len = 250` frames). Si un clip de VoxCeleb duraba menos, `np.random.randint(0, time - spec_len)` recibiría un rango negativo y reventaría. Duplicar la señal blinda esa operación.

**Por qué espejo y no repetición.** Está en el empalme. Con `np.append(wav, wav)` la señal salta de `x[N-1]` a `x[0]`: una discontinuidad que en el espectrograma aparece como un **click de banda ancha**, un frame con energía en todas las frecuencias. Con `wav[::-1]` la secuencia es `…, x[N-2], x[N-1], x[N-1], x[N-2], …` — continua. Es el mismo truco que `np.pad(mode='reflect')`.

**Y acá no protege nada**, porque no hay crop. Solo duplica el trabajo:

| | frames de STFT | descriptores a VLAD |
|---|---|---|
| sin espejo | 821 | 51 |
| **con espejo** | **1.641** | **102** |

### La predicción y su verificación

Que el espejado fuera inocuo era predecible **antes** de medirlo, por dos propiedades del agregador:

1. VLAD acumula con `sum` sobre los descriptores → **invariante a permutaciones**. El orden temporal no existe para él, así que un tramo reversado aporta casi el mismo conjunto de descriptores.
2. La intra-normalización de `VladPooling` **cancela el factor de escala**: duplicar el conjunto de descriptores multiplica cada `v_k` por exactamente 2, y normalizar lo elimina (verificado en [NetVLAD desarmado](03-netvlad-desarmado): `‖v_dup‖/‖v_orig‖ = 2,000000`, y tras normalizar la diferencia es 2,2×10⁻⁸).

Medido con los **pesos entrenados** sobre dos voces sintéticas:

| | coseno con el embedding sin espejar |
|---|---|
| **voz A, con espejado** (lo que hace el lab) | **0,999813** |
| voz A, repetida sin espejar | 0,999923 |
| **voz B, con espejado** | **0,999646** |
| voz B, repetida sin espejar | 0,999827 |
| *referencia:* coseno(voz A, voz B) | **0,684575** |

El espejado altera el embedding en el **cuarto decimal**, mientras dos voces distintas se separan en 0,68: la perturbación es ~1.500× menor que la señal que el modelo debe medir.

{{< concept-alert type="clave" >}}
**El espejado duplica el costo de la celda más lenta del lab —4.715 forward passes— para cambiar el resultado en el 0,02 %.** Eliminarlo debería recortar el tiempo de extracción casi a la mitad sin mover el EER.

Y hay una lección de método: el primer intento de medir esto **con pesos aleatorios** dio coseno 0,9998 para el espejado… y también 0,9998 entre dos señales completamente distintas. El modelo sin entrenar colapsa todos los embeddings al mismo cono (la ReLU previa a la L2-norm los confina al ortante positivo, y el sesgo domina la dirección). El experimento no medía nada. **Un test de invarianza necesita un control negativo**: sin el 0,685 de referencia, el 0,9998 no significa nada.
{{< /concept-alert >}}

---

## 3. La STFT, y una ventana que contradice al paper

```python
librosa.stft(wav, n_fft=512, win_length=400, hop_length=160)
```

| Parámetro | Valor | En tiempo a 16 kHz | Rol |
|---|---|---|---|
| `win_length` | 400 | **25 ms** | ventana de análisis |
| `hop_length` | 160 | **10 ms** | paso (75 % de solape) |
| `n_fft` | 512 | — | **257 bins** = 512/2 + 1 |

Viene textual del paper:

> *"Spectrograms are generated in a sliding window fashion using a **hamming window** of width 25ms and step 10ms. We use a 512 point FFT, giving us 256 frequency components, which together with the DC component of each frame gives a short-time Fourier transform (STFT) of size 257 × 250 (frequency × temporal) out of every 2.5 second crop."*

Tres cosas que el paper no dice y el código sí:

**`win_length` = 400 < `n_fft` = 512.** No es un error: librosa toma la ventana de 400 muestras y le hace **zero-padding centrado** hasta 512, porque la FFT es mucho más rápida en potencias de 2 y 400 no lo es. El efecto secundario es interpolar el espectro: la resolución espectral real la fija la ventana de 25 ms (≈ 40 Hz), no el espaciado de 31,25 Hz que sugieren los 512 puntos.

**La ventana es Hann, no Hamming.** El paper dice *hamming*; `librosa.stft` usa `window='hann'` por defecto y el código no lo sobrescribe. **El código contradice al paper.** Para magnitud espectral de voz la diferencia es marginal —Hamming no llega a cero en los extremos (0,08), Hann sí— pero es el tipo de detalle que impide reproducir exactamente un número publicado.

**`center=True`** (por defecto): librosa rellena la señal con `n_fft//2 = 256` muestras a cada lado, así que el número de frames es `1 + len(wav)//hop_length`. Verificado: 64.000 muestras (4 s) → **401 frames**; con espejo, 128.000 → **801**.

Y `librosa.magphase` descompone el complejo en magnitud y fase, **descartando la fase**. Es una decisión de fondo, no un detalle: con solo la magnitud no se puede reconstruir el audio (de ahí Griffin-Lim y los vocoders neuronales). La justificación clásica es que la identidad del hablante vive en la envolvente espectral —formantes, estructura armónica, características del tracto vocal— y no en la fase absoluta. Es la misma decisión que en [MFCC](/fundamentos/mfcc-y-escala-mel).

---

## 4. El eje de la normalización, y el cambio silencioso entre dos papers

```python
# preprocessing, subtract mean, divided by time-wise var
mu = np.mean(spec_mag, axis = 0, keepdims=True)
std = np.std(spec_mag, axis = 0, keepdims=True)
return (spec_mag - mu) / (std + 1e-5)
```

`spec_mag` tiene forma `(freq=257, time=T)`. **`axis=0` recorre la frecuencia**, así que `mu` y `std` tienen forma `(1, T)`: **un valor por frame temporal**, calculado sobre las 257 bandas de ese frame.

Cada instante de 10 ms se normaliza contra sí mismo. **No es normalización a lo largo del tiempo**, y el comentario `divided by time-wise var` sugiere lo contrario. La descripción correcta sería *per-frame normalisation over frequency*.

**No es un bug: es fiel al paper**, que lo dice con precisión quirúrgica:

> *"The spectrogram is normalised by subtracting the mean and dividing by the standard deviation of **all frequency components in a single time step**."*
> — [Xie et al. 2019](/papers/utterance-level-xie-2019)

**Pero contradice al paper de VoxCeleb, del mismo grupo, que hace exactamente lo opuesto:**

> *"Mean and variance normalisation is performed on **every frequency bin** of the spectrum. **This normalisation is crucial**, leading to an almost 10% increase in classification accuracy."*
> — [Nagrani et al. 2017](/papers/voxceleb-nagrani-2017)

Normalizar «cada bin de frecuencia» es `axis=1`: el eje contrario. **El grupo de Oxford invirtió el eje de normalización entre 2017 y 2019, sobre el mismo dataset, después de declarar en 2017 que esa normalización era crucial y valía ~10 puntos de accuracy.** Ninguno de los dos papers menciona el cambio.

### Qué destruye cada eje

Medido sobre una señal sintética de 4 s con envolvente de energía variable y un tramo de silencio real:

| Medición | Espectro crudo | **por frame** (el lab / Xie 2019) | por bin (VoxCeleb 2017) |
|---|---|---|---|
| Energía por frame | varía **384×** | aplanada a **~0** (±5,7×10⁻⁵) | se conserva (−0,33 a 7,82) |
| RMS del tramo de silencio | 0,027 | **0,999** | — |
| RMS del tramo de voz | — | 1,000 | — |

Y la verificación reproducida en el notebook, con un tono al que se le atenúa medio segundo por un factor de 1.000:

```
RMS crudo   voz/silencio: 8.580156 / 0.008580156      <- factor 1000
RMS normal. voz/silencio: 0.99999887 / 0.998836       <- factor 1.001
```

**Mil veces de diferencia de volumen se convierten en un 0,1 %.**

### Las dos consecuencias que atraviesan el lab

**1. El modelo no puede usar el volumen.** Ni instantáneo ni global. Y eso es *deseable*: la distancia al micrófono, la ganancia del canal y el volumen al que alguien habla no son propiedades de su identidad. Lo que queda es la **forma** del espectro de cada instante, que sí lo es. La normalización es un filtro de invarianza deliberado.

**2. El silencio entra a la red con la misma intensidad que la voz.** Un frame casi vacío tiene `std` minúscula; dividir por ella amplifica el ruido de fondo **37×** hasta ocupar el mismo rango dinámico que un frame de habla plena. Y el paper es explícito:

> *"No voice activity detection (VAD), or automatic silence removal is applied."*

{{< concept-alert type="conexion" >}}
**Esta es la justificación entera de los ghost clusters.** El modelo necesita un mecanismo interno para descartar basura porque el preprocesamiento se la entrega **amplificada y sin marcar**. No hay VAD en el pipeline porque el agregador aprende a tirar el silencio: los dos clusters fantasma de [GhostVLAD](/papers/ghostvlad-zhong-2018) son el detector de actividad de voz implícito del modelo, y valen 0,35 puntos de EER.

El detalle del `+ 1e-5`: además de evitar la división por cero en un frame de silencio digital absoluto, actúa como freno de la amplificación — `1e-5` es *grande* comparado con la `std` típica de un frame vacío.
{{< /concept-alert >}}

---

## 5. El pipeline completo

```
archivo .wav                                    16 kHz, mono, float32
   ↓ librosa.load
señal x[n]                                      ~8,2 s → 131.200 muestras
   ↓ np.append(x, x[::-1])                        ← duplica el costo, no recorta nada
señal espejada                                  ~16,4 s → 262.400 muestras
   ↓ librosa.stft (Hann 25 ms, hop 10 ms, 512 pt)
espectrograma complejo                          257 × 1641
   ↓ magphase → se descarta la fase
magnitud                                        257 × 1641
   ↓ normalización POR FRAME (sobre las 257 bandas)
entrada a la red                                257 × 1641, largo variable → batch = 1
```

El `dtype` sobrevive como `float32` en toda la cadena (`stft` → `complex64` → `magphase` → `float32`), que es lo que el modelo espera. Si en algún punto pasara a `float64`, el forward lanzaría `RuntimeError: expected scalar type Float but found Double`.

---

**Siguiente:** [El Thin ResNet, la errata y el campo receptivo](02-el-thin-resnet-y-la-errata) — donde los 1.641 frames se convierten en 102 descriptores, y no en los 51 que declara el paper.
