---
title: "El dato de audio"
weight: 1
math: true
---

La primera parte del laboratorio recorre el camino que va de un archivo en disco al tensor que entra a un modelo. No hay modelos todavía: hay arreglos, sample rates, formatos y transformadas. Y en cada paso aparece la misma idea de fondo — **un audio digital no es un objeto especial, es una lista de números más la metadata que dice cómo interpretarla**.

## 1. Fabricar un audio desde cero

El lab no empieza descargando: empieza construyendo un tono de 440 Hz, la nota La del estándar de afinación.

```python
sr = 22050          # muestras por segundo
dur = 1.0           # segundos
t = np.arange(int(sr*dur)) / sr
la440 = 0.01 * np.sin(2*np.pi*440*t)
```

Es una decisión pedagógica acertada: al fabricarlo, sabes exactamente qué hay dentro, así que cualquier cosa rara después viene del pipeline y no del archivo. El `2\pi` convierte ciclos a radianes, y `np.arange(22050)/22050` produce los instantes de muestreo, separados **45,35 µs**.

La elección de 22.050 Hz tampoco es casual: es **el sample rate de GTZAN**, el dataset de la Parte 2. Los números que aprendes con el juguete sirven después con los datos reales.

![Dos paneles: a la izquierda un segundo completo de la sinusoide que aparece como una franja azul sólida, a la derecha un zoom a 10 milisegundos donde se ven cuatro ciclos con las muestras individuales marcadas como puntos](/laboratorios/lab-37/onda-y-muestras.png)

Los dos paneles dicen lo mismo a escalas distintas, y el contraste es el contenido. A la izquierda hay 22.050 puntos en ~600 píxeles —**37 muestras por píxel**— así que lo que se ve no es la onda sino su envolvente. **La forma de onda de un tono audible nunca es visible a escala de segundos.** Por eso todo el análisis de audio trabaja con espectrogramas.

A la derecha, 220 muestras (9,98 ms, ~4,4 ciclos) sí muestran el seno, con los puntos visibles. Y ahí está el detalle conceptual: **las rectas entre puntos no existen en los datos**. Son matplotlib uniendo muestras. El DAC de la placa de sonido tampoco las une con rectas: reconstruye con un filtro sinc, la interpolación de Whittaker-Shannon que garantiza reconstrucción exacta si se respetó Nyquist.

### El peso en disco

```python
sf.write("la440.wav", la440, sr, subtype="PCM_16")
# 44144 bytes
```

La cuenta: $22.050 \times 2\ \text{bytes} = 44.100$ bytes de datos. Sobran exactamente **44 bytes**, que son el header WAV canónico mínimo: 12 del chunk RIFF, 24 del `fmt` (formato, canales, sample rate, bits) y 8 del encabezado de `data`. Ese header es todo lo que convierte un montón de bytes en "un archivo de audio".

{{< callout type="info" >}}
**La fórmula que resuelve la Actividad 2** aparece acá por primera vez: $sr \times \text{bits} \times \text{canales} \times \text{duración}$. Con 22.050 × 16 × 1 × 1 = 352.800 bits = 44.100 bytes, que es exactamente el tamaño del archivo menos su header.
{{< /callout >}}

## 2. Canales, y el gotcha de los ejes

```python
estereo = np.stack([la440, mi659])                        # (2, 22050)
sf.write("estereo.wav", estereo.T, sr, subtype="PCM_16")  # ojo con el .T
ipd.Audio(estereo, rate=sr)                               # acá sin .T
```

Esas tres líneas usan **dos convenciones opuestas**:

| Librería | Forma esperada |
|---|---|
| **torchaudio** e IPython | `(canales, muestras)` |
| **soundfile** | `(muestras, canales)` |

De ahí el `.T` en una línea y no en la otra. Sin él, soundfile interpretaría el arreglo como 22.050 canales de 2 muestras. Es el error más común al mezclar ambas librerías, y el anexo del notebook lo lista como el primero de sus "errores frecuentes".

El detalle musical: 659,25 Hz es Mi5, y $659{,}25/440 = 1{,}4983 = 2^{7/12}$ — una **quinta justa**. Pero las amplitudes son 0,01 y 0,8: **38 dB de diferencia**. Como `ipd.Audio` normaliza por el máximo *global* del arreglo, el La queda al 1,25% del fondo de escala y prácticamente no se oye. El comentario del notebook ("La a la izquierda, Mi a la derecha") describe la intención, no lo que se percibe.

## 3. El sample rate: dos operaciones que suenan parecido y no lo son

Con un archivo real (una risa, 57.173 muestras estéreo a 44.100 Hz = 1,296 s), el lab contrasta dos cosas que se confunden a menudo:

|  | Cambiar el `rate` de reproducción | `Fa.resample` |
|---|---|---|
| Muestras | 57.173 (sin tocar) | 57.173 → **10.371** |
| Duración | **cambia** | se conserva |
| Tono | **cambia** | se conserva |
| Información | intacta | **se destruye** |

La primera solo cambia la etiqueta: los mismos números leídos a otro ritmo. Es el vinilo a 45 RPM. La segunda modifica los datos para conservar el sonido a otra tasa.

### El caso del teléfono

Resamplear a 8 kHz deja el Nyquist en 4 kHz, y **elimina el 81,9% del espectro representable**. Internamente torchaudio hace una conversión racional —$\gcd(44100, 8000) = 100$, así que la razón es 80/441— con un filtro sinc con ventana antes de decimar. Ese filtro no es opcional: sin él, el contenido sobre 4 kHz no desaparecería sino que se **doblaría** hacia abajo como frecuencias falsas.

Por qué el teléfono suena a teléfono: los fundamentales de la voz (85-255 Hz) sobreviven, pero las **consonantes fricativas** —"s", "f", "sh"— viven entre 4 y 10 kHz, justo en la banda amputada. De ahí que cueste distinguir "s" de "f" por teléfono, y que exista el alfabeto fonético de radio.

![Tres curvas de espectro superpuestas en escala logarítmica: la de 44100 Hz llega hasta 22 kHz, la de 8000 Hz se corta en seco a 4 kHz, la de 2000 Hz a 1 kHz](/laboratorios/lab-37/espectros-sample-rates.png)

Cada curva **termina en seco en su propio Nyquist**. Esa imagen es el teorema de muestreo hecho gráfico: el resampleo conserva la banda baja intacta y amputa el resto. Y explica por qué la degradación se acelera al bajar — no se pierde "calidad" difusa, se pierde una banda concreta cada vez más grande.

{{< callout type="warning" >}}
**Gotcha del navegador:** los reproductores de audio no aparecen para sample rates menores a 8 kHz. La especificación de Web Audio solo exige soportar de **8.000 a 96.000 Hz**; por debajo, el navegador no está obligado a decodificar nada. Los datos están bien —`Fa.resample` produjo el tensor correcto— pero no se pueden reproducir directo.

La solución es la misma operación de la celda siguiente: degradar a la tasa baja y **volver a subir** a 44.100 Hz. Como subir no recupera nada, se oye el daño a una tasa que el navegador sí acepta. De paso, todos los ejemplos quedan a la misma velocidad y la comparación aísla el contenido espectral.
{{< /callout >}}

### La irreversibilidad, medida

![Dos espectros superpuestos: el original a 44.1 kHz y el que pasó por 8 kHz y volvió; coinciden bajo los 4 kHz y el segundo se desploma justo en esa línea](/laboratorios/lab-37/resampleo-irreversible.png)

Al volver de 8 kHz a 44,1 kHz se recuperan **57.170 muestras** (contra 57.173 originales), pero solo **10.371 grados de libertad reales**. Los otros 46.799 valores son interpolación calculada a partir de esos.

Es matemáticamente imposible que sea de otra manera: bajar la tasa es una **proyección** a un subespacio de menor dimensión, e infinitas señales distintas se proyectan al mismo resultado. El upsampling elige una de esas preimágenes — la que no tiene nada sobre 4 kHz.

{{< callout type="info" >}}
**Por qué esto importa fuera del ejercicio.** Los call centers y la telefonía graban a **8 kHz**; wav2vec 2.0 y Whisper esperan **16 kHz**. Resamplear hacia arriba entrega el formato correcto con la información faltante: la banda de 4 a 8 kHz queda vacía, y el modelo —preentrenado con audio de banda ancha donde esa banda *tiene* contenido— opera fuera de su distribución. Es una de las razones concretas por las que la transcripción de llamadas rinde por debajo de las cifras publicadas.
{{< /callout >}}

### Actividad 1

> Un archivo de **1.653.750 muestras** a **44.100 Hz**: ¿cuántos segundos representa? ¿De qué tamaño debe ser la ventana para 0,01 s?

**Respuesta: `37.5s - 441`.** Todo sale de una relación despejada en las dos direcciones:

$$\text{duración} = \frac{1.653.750}{44.100} = 37{,}5 \text{ s} \qquad\qquad \text{ventana} = 44.100 \times 0{,}01 = 441 \text{ muestras}$$

Los distractores son errores identificables: **75 s** sale de usar 22.050 Hz (el sample rate del juguete en vez del que da el enunciado), **150 s** de usar el Nyquist como si fuera la tasa, **882** de un factor 2, y **4.410** de confundir 0,01 con 0,1.

Y el enunciado no eligió 0,01 s al azar: **10 ms es el hop estándar del procesamiento de voz**. Aparece literalmente en la celda opcional del propio notebook, que usa `hop_length=160` a 16 kHz — los mismos 10 ms.

## 4. Formatos: comprimir o no

```bash
ffmpeg -i Woman-Laughing.wav risa.flac        # el formato sale de la EXTENSIÓN
ffmpeg -i Woman-Laughing.wav -c:a aac risa.m4a  # salvo .m4a: es un CONTENEDOR
ffmpeg -i Woman-Laughing.wav -ar 16000 -ac 1 risa_16k_mono.wav
```

| Archivo | Tamaño | Familia | Cómo reduce |
|---|---|---|---|
| WAV original | 224K | sin comprimir | — |
| FLAC | 112K | **lossless** | predicción lineal; reconstruye bit a bit |
| MP3 / OGG / M4A | 24K | **lossy** | modelo psicoacústico: descarta lo enmascarado |
| WAV 16k mono | 44K | sin comprimir | no comprime: tiene menos datos |

Esa última fila es conceptualmente distinta y conviene no confundirla. Y tiene una simetría elegante: queda en **18,1%** del original, exactamente la misma fracción que conservaba el resampleo a 8 kHz, porque $\frac{16.000}{44.100} \times \frac{1}{2} = \frac{8.000}{44.100}$.

El último comando es el que se usa en proyectos reales: **convierte, resamplea y deja mono en una sola línea**. Es el preprocesamiento que exige cualquier modelo tipo wav2vec.

{{< callout type="warning" >}}
**`du -h` miente un poco.** Reporta bloques de disco, no bytes. Por eso los tres formatos lossy aparecen como "24K" idénticos: no pesan lo mismo, caen en el mismo número de bloques de 4 KiB. Para comparar MP3 contra Vorbis contra AAC de verdad hay que usar `ls -la`.
{{< /callout >}}

![Dos espectros superpuestos, WAV y MP3, casi idénticos en las frecuencias bajas y medias, con el del MP3 desplomándose en la banda alta](/laboratorios/lab-37/wav-vs-mp3.png)

Los dos suenan igual —el modelo psicoacústico está diseñado contra el sistema auditivo humano— y aun así el espectro delata el códec: un corte abrupto en la banda alta más huecos en las regiones enmascaradas.

{{< callout type="info" >}}
**Inaudible no es invisible para un modelo.** Una CNN sobre espectrograma puede aprender esos artefactos. De ahí un modo de falla real: si las clases tienen **procedencias distintas** (unas de archivos comprimidos, otras de WAV), el modelo puede aprender a detectar el códec en vez de la clase — accuracy altísima en validación, colapso en producción. El criterio no es "¿se oye igual?" sino "¿mi modelo usa la banda que el códec descarta?".
{{< /callout >}}

### Actividad 2

> De estos archivos **sin comprimir**, ¿cuál ocupa más? **A:** 60 s a 60 Hz, 16 bits · **B:** 30 s a 120 Hz, 32 bits · **C:** 15 s a 120 Hz, 128 bits · **D:** 60 s a 60 Hz, 8 bits

**Respuesta: `C`**, con 230.400 bits.

| | Cuenta | Bits |
|---|---|---|
| A | $60 \times 60 \times 16$ | 57.600 |
| B | $120 \times 30 \times 32$ | 115.200 |
| **C** | $120 \times 15 \times 128$ | **230.400** |
| D | $60 \times 60 \times 8$ | 28.800 |

El ejercicio está construido en potencias de 2: **cada opción duplica a la anterior**. Y C es la de **duración más corta** y aun así la más pesada — que es todo el punto: el peso lo determina el producto de los tres factores, no la duración sola.

El "sin comprimir" del enunciado es lo que hace válida la fórmula. Con FLAC o MP3 el tamaño dependería del **contenido**: el tono puro comprime muchísimo más que la risa, siendo idénticos en PCM.

## 5. De la onda al tensor

```python
args = dict(n_fft=2048, hop_length=512, win_length=2048, center=False)
MelSpectrogram(sample_rate=sr2, n_mels=64, **args)     # -> (64, 108)
MFCC(sample_rate=sr2, n_mfcc=20, melkwargs=dict(n_mels=64, **args))  # -> (20, 108)
```

La cadena de reducción sobre los 57.173 números del audio:

| Paso | Qué hace | Dimensión |
|---|---|---|
| STFT | ventanas de 2.048 → FFT | 1.025 bins por frame |
| Filtros Mel | agrupa en bandas perceptuales | 64 bandas (~16 bins cada una) |
| log + DCT | comprime y **decorrela** | 20 coeficientes |

**8,3× de reducción** con el mel, **26,5×** con los MFCC. La DCT existe porque las bandas Mel están fuertemente correlacionadas: decorrelarlas concentra la información en los primeros coeficientes, así que 20 de 64 no pierden casi nada. Es PCA con base fija.

**Los dos tienen 108 frames**, y eso no es casualidad: el MFCC cambia qué se calcula *dentro* de cada ventana, no cómo se corta el tiempo. Los dos ejes se determinan por separado.

### Los defaults que muerden

La misma señal con los parámetros por defecto da `(1, 64, 56)` en vez de 108 frames. La descomposición exacta:

| Cambio | Frames |
|---|---|
| `hop=512`, `center=False` | 108 |
| duplicar el hop al default (`n_fft//2 = 1024`) | 54 |
| agregar el padding de `center=True` | **56** |

Y hay algo elegante ahí. Las dos fórmulas que da el notebook —

```
center=False:  frames = ⌊(N − win_length) / hop_length⌋ + 1
center=True:   frames = ⌊N / hop_length⌋ + 1
```

— **no son reglas distintas**: la segunda es la primera con $N$ reemplazado por $N + n_{fft}$ (el largo con padding). El padding cancela exactamente el término $-\text{win\_length}$. Hay una sola fórmula.

{{< callout type="info" >}}
**La analogía que hace todo esto memorizable:** una STFT es una convolución 1D. `n_fft` es el receptive field, `win_length` el kernel size, **`hop_length` el stride**, `n_mels` los canales de salida y **`center` el padding**. Si sabes calcular el output de una conv, sabes calcular los frames de un espectrograma — es literalmente la misma fórmula.
{{< /callout >}}

Y la asimetría entre los dos ejes es la causa raíz del problema que va a explotar en la Parte 2: **las filas son un hiperparámetro** (no dependen del audio) y **las columnas dependen de los datos**. Dos canciones de largo distinto producen tensores que no se pueden apilar.

### Actividad 3

> Archivo **monoaural** a **32.768 Hz**, **30 segundos**, ventana **4.096**, hop **2.048**, **48** bins Mel, **sin padding**. ¿Tamaño tras el MFCC?

**Respuesta: `(48, 479)`.**

$$N = 32.768 \times 30 = 983.040 \qquad \text{frames} = \left\lfloor \frac{983.040 - 4.096}{2.048} \right\rfloor + 1 = 479$$

Los números están elegidos para que la división sea exacta (32.768 = 2¹⁵). Verificado construyendo la transform real con `center=False`.

{{< callout type="warning" >}}
**Una imprecisión del enunciado.** Da "48 bins Mel" y pregunta por la salida del **MFCC**, que tiene `n_mfcc` filas, no `n_mels` — son parámetros distintos, con la restricción `n_mfcc ≤ n_mels`. La respuesta (48, 479) asume `n_mfcc = n_mels = 48`; con el default de torchaudio (`n_mfcc=40`) sería (40, 479), que no figura entre las opciones. En la práctica casi nunca se usan iguales: el punto de la DCT es quedarse con menos coeficientes que bandas.
{{< /callout >}}

## Qué nos llevamos

- **Un audio son números más metadata.** El sample rate no vive en los datos; es un argumento o un campo de 4 bytes en el header.
- **`(canales, muestras)` en torchaudio, siempre** — incluso en mono. soundfile va al revés.
- **Cambiar el `rate` ≠ resamplear.** Lo primero altera tono y duración sin tocar datos; lo segundo conserva el sonido y destruye información de forma irreversible.
- **La STFT es una convolución 1D**, y los defaults (`hop = n_fft//2`, `center=True`) son la causa del 90% de los desajustes de dimensión.
- **El peso sin comprimir es determinista**; con compresión depende del contenido.

---

**Ver tambien:** [Lab 37 — hub](/laboratorios/lab-37) · Siguiente: [Data augmentation](02-data-augmentation) · Fundamentos: [Representación de audio](/fundamentos/representacion-de-audio) · [MFCC y escala Mel](/fundamentos/mfcc-y-escala-mel) · [Representación tiempo-frecuencia](/fundamentos/representacion-tiempo-frecuencia).
