---
title: "Escala Mel y MFCC"
weight: 3
---

La STFT usa frecuencias en escala lineal, pero el oído humano no percibe así. La última representación del lab —los **MFCC**— transforma el espectro a cómo *oímos* y lo comprime a pocos coeficientes: la representación clásica del reconocimiento de voz.

## La escala Mel: frecuencias "como las oye el humano"

El oído percibe muy bien las diferencias en frecuencias **bajas** (100 vs 200 Hz) pero mal en **altas** (8000 vs 8100 Hz). Nuestra percepción del tono es aproximadamente **logarítmica**: una octava (duplicar la frecuencia) suena como "el mismo salto" tanto de 100→200 como de 1000→2000 Hz.

La **escala Mel** transforma Hz → Mel para que distancias iguales en Mel suenen como saltos de tono iguales:

$$m = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$$

En la práctica se aplica un **banco de filtros Mel** al espectrograma (bandas angostas en graves, anchas en agudos), imitando la cóclea. El resultado —el **mel-espectrograma**— concentra la resolución donde el oído la necesita.

## De Mel a MFCC: el paso "cepstral"

Los MFCC dan un paso más. El pipeline completo:

```
señal → STFT → |·|² → filtros Mel → log → DCT → MFCC
```

Los dos últimos pasos son la clave:

- **`log`** — logaritmo de la energía Mel (percepción logarítmica del volumen).
- **`DCT`** (Transformada Discreta del Coseno) — el paso "cepstral". Aplica una transformada al espectro logarítmico con dos efectos: **descorrelaciona** las bandas Mel (muy correlacionadas) y **concentra** la información en pocos coeficientes; y **separa el timbre de la afinación** — los coeficientes bajos capturan la envolvente espectral (qué *tipo* de sonido, el timbre), los altos el detalle fino.

```python
mfccs = librosa.feature.mfcc( y=audio_data, sr=sampling_rate, n_mfcc=40 )
print( mfccs.shape )   # (40, 173): 40 coeficientes × 173 ventanas
```

![MFCC de un audio real: 40 coeficientes a lo largo del tiempo, con estructura que sigue los eventos del sonido](/laboratorios/lab-35/mfcc-audio-real.png)

La compresión es brutal: el espectrograma STFT del mismo audio tenía **1025 bins**; los MFCC lo reducen a **40 coeficientes descorrelacionados** — ~25× más compacto.

{{< callout type="info" >}}
**Por qué los MFCC dominaron el reconocimiento de voz.** Davis & Mermelstein los formalizaron en 1980. Antes del deep learning eran oro: features compactas, descorrelacionadas (ideales para GMM-HMM) y perceptualmente motivadas. Hoy los modelos grandes suelen usar mel-espectrogramas crudos (dejan que la red aprenda la compresión), pero los MFCC siguen siendo el baseline clásico. Ver [MFCC y escala Mel](/fundamentos/mfcc-y-escala-mel).
{{< /callout >}}

{{< callout type="warning" >}}
**MFCC ≠ mel-espectrograma (confusión clásica).** El **mel-espectrograma** es el espectro en escala Mel (eje Y = frecuencias Mel, interpretable). Los **MFCC** son la DCT del log-mel-espectrograma (eje Y = coeficientes abstractos, descorrelacionados). En el lab, `specshow(mfccs, y_axis='mel')` etiqueta el eje como 'mel', lo cual es técnicamente impreciso: los MFCC ya pasaron por la DCT y perdieron el eje de frecuencia directo. No afecta el cálculo, solo la etiqueta.
{{< /callout >}}

## La escalera completa de representaciones

El lab recorre cuatro representaciones del mismo sonido, cada una más condensada hacia "lo que importa":

| Representación | Ejes | Qué captura | Tamaño |
|---|---|---|---|
| **Forma de onda** | tiempo | la señal cruda | N muestras |
| **FFT** | frecuencia | qué frecuencias (sin *cuándo*) | 1 espectro |
| **STFT** | tiempo × frecuencia (lineal) | evolución frecuencial | ~1025 × N |
| **Mel-spec** | tiempo × frecuencia (perceptual) | como lo oye el humano | ~128 × N |
| **MFCC** | tiempo × coeficientes | timbre, comprimido | ~40 × N |

Es un embudo de la representación cruda (onda) a la semántica (MFCC), descartando lo redundante en cada paso. **Este embudo es el preprocesamiento de audio de casi cualquier sistema** — lo que precede a cada forward pass de un modelo de audio. Lo que hiciste a mano aquí, los labs 36–37 y los modelos modernos lo hacen automáticamente antes de que la red toque el sonido.
