---
title: "Data augmentation"
weight: 2
math: true
---

Los datasets de audio etiquetados son chicos —el de la Parte 2 tiene 100 canciones por género— y un modelo con suficientes parámetros se los aprende de memoria. Como conseguir más datos casi nunca es posible, se fabrican: transformar los que hay de manera que suenen distinto pero **conserven la etiqueta**.

El criterio que ordena toda la sección, y el concepto más transferible del laboratorio:

> ¿Esta transformación es una **invariancia real** del problema, o le estoy enseñando al modelo algo falso?

Cambiar el tono es válido para clasificar género y **destructivo** si la tarea fuera detectar tonalidad o identificar al hablante — porque ahí la etiqueta *es* lo que se está modificando. No hay una lista de augmentations buenas: depende de qué invarianza tiene el problema.

## Las cuatro maneras de "acelerar" un audio

| Operación | Duración | Tono | Nº de muestras |
|---|---|---|---|
| Reproducir a otro `rate` | cambia | cambia | igual |
| `resample` | igual | igual | **cambia** |
| `time_stretch` | cambia | **igual** | cambia |
| `pitch_shift` | **igual** | cambia | igual |

Las dos primeras acoplan tono y duración; las dos últimas los **desacoplan**, y por eso son las útiles como augmentation. Si quisieras enseñarle al modelo que un rock 10% más rápido sigue siendo rock, reproducir a otro rate no serviría: le estarías cambiando también la tonalidad.

## Estirar el tiempo sin mover el tono

El desacople lo hace un **phase vocoder**, y su idea cabe en una asimetría de la STFT: **la amplitud de cada bin es información sobre el *contenido*, y la fase es información sobre el *reloj*.**

```python
def time_stretch(y_np, rate, n_fft=2048, hop=512):
    vent = torch.hann_window(n_fft)
    Sx = torch.stft(torch.from_numpy(y_np).float(), n_fft=n_fft, hop_length=hop,
                    window=vent, return_complex=True)          # 1. analisis
    Sy = TimeStretch(hop_length=hop, n_freq=n_fft//2+1, fixed_rate=rate)(Sx)   # 2. vocoder
    return torch.istft(Sy, n_fft=n_fft, hop_length=hop, window=vent).numpy()   # 3. sintesis
```

Resultados medidos: 57.173 muestras (1,30 s) pasan a **81.408** con `rate=0.7` (1,85 s) y a **37.888** con `rate=1.5` (0,86 s).

Esos largos son raros a propósito. Con `rate=0.7` uno esperaría $57.173/0{,}7 = 81.676$, y salen 81.408. La razón es que **el estiramiento opera sobre frames, no sobre muestras**:

- La STFT produce 112 frames ($57.173/512 + 1$, con el `center=True` que es el default de `torch.stft`).
- `TimeStretch` los lleva a $\lceil 112/0{,}7 \rceil = 160$.
- La iSTFT reconstruye $(160-1) \times 512 = 81.408$ muestras.

El largo final siempre es **múltiplo del hop**, porque la unidad de trabajo es el frame.

### Por qué las fases se regeneran

Cada bin $k$ tiene una frecuencia central $f_k = k \cdot sr / n_{fft}$, y al avanzar un hop su fase *debería* avanzar $\Delta\phi = 2\pi k \cdot hop / n_{fft}$. La diferencia entre la fase observada y la esperada revela la **frecuencia instantánea real** de lo que suena en ese bin. El vocoder mide esa desviación y **reacumula la fase al nuevo ritmo de frames**, de modo que cada oscilador siga girando a su velocidad propia aunque los frames se hayan separado o juntado.

{{< callout type="warning" >}}
**El notebook promete una demostración que no entrega.** Dice que *"la celda siguiente muestra qué pasa si uno copia las fases en vez de regenerarlas"* — y ninguna celda lo hace. La construimos aparte: tomar las magnitudes re-temporizadas y multiplicarlas por las fases **copiadas** del frame origen, en vez de reacumularlas.

El resultado tiene nombre propio: **"phasiness"**. Un timbre metálico, acuoso y difuso, como con reverberación artificial. Es la demostración auditiva de que **la fase no es un detalle**: lleva la información temporal que mantiene coherente la señal al reconstruir por overlap-add.
{{< /callout >}}

### La ventana de Hann hace dos cosas

El comentario del código menciona una, y hay dos:

1. **Evitar leakage espectral** — apagar los bordes para que el corte no invente frecuencias.
2. **Cumplir la condición COLA** (constant overlap-add). Con `hop = n_fft/4` —que es exactamente el caso: 512 = 2048/4— las ventanas de Hann solapadas **suman una constante**, y la iSTFT reconstruye sin ondulaciones de amplitud. Con `hop = n_fft/3` la reconstrucción tendría un temblor periódico en el volumen.

## Mover el tono sin cambiar la duración

`pitch_shift` es una **composición** de las dos operaciones anteriores: estirar por $2^{n/12}$ y resamplear por el inverso. Los dos pasos **se cancelan en duración y se suman en frecuencia**.

```python
y_agudo = Fa.pitch_shift(torch.from_numpy(y2), sr2, n_steps=4)    # +4 semitonos = +26%
y_grave = Fa.pitch_shift(torch.from_numpy(y2), sr2, n_steps=-4)   # -4 semitonos = -20,6%
# Todos conservan el largo: (57173,) (57173,) (57173,)
```

![Tres espectrogramas Mel lado a lado: menos cuatro semitonos, original y más cuatro semitonos; los armónicos se desplazan verticalmente en bloque mientras el eje temporal permanece idéntico](/laboratorios/lab-37/pitch-shift-mel.jpg)

Los armónicos **suben o bajan en bloque**, manteniendo sus proporciones, y el eje del tiempo queda idéntico. Esa rigidez del patrón es lo que distingue un pitch shift de un cambio de velocidad.

{{< callout type="warning" >}}
**El warning de esta celda es un error de configuración real, y vale entenderlo.** torchaudio avisa que *"al menos un banco de filtros mel tiene todos sus valores en cero"*. La función auxiliar construye el `MelSpectrogram` **sin especificar `n_fft`**, así que toma el default de **400**. A 44.100 Hz eso da 201 bins de **110,25 Hz** cada uno — y las bandas Mel bajas son más angostas que un bin, así que quedan literalmente vacías.

**7 de 64 bandas mueren, y cubren de 0 a 374 Hz**: exactamente donde vive la fundamental de la voz humana. El gráfico no muestra la componente cuyo desplazamiento se quiere ver.

La causa raíz es un default pensado para otro sample rate: `n_fft=400` a 16 kHz son los **25 ms** de la ventana estándar de ASR; a 44.100 Hz son 9,1 ms, demasiado corta.
{{< /callout >}}

Subir a `n_fft=2048` elimina las bandas vacías, pero **no es una mejora gratuita**: la resolución frecuencial gana 5,1× (110,25 → 21,53 Hz por bin) y la temporal pierde exactamente 5,1× (286 → 56 columnas). Es la **incertidumbre de Gabor** en estado puro — el producto tiempo × frecuencia es constante y solo se puede redistribuir. Para *esta* celda conviene, porque la afirmación que se quiere verificar vive en el eje de frecuencia; para ver los ataques de cada "ja" de la risa, convendría lo contrario.

## SpecAugment

Si conoces **Cutout** de visión por computador, es lo mismo sobre el espectrograma: tapar con ceros filas (frequency masking) y columnas (time masking), de ancho y posición aleatorios.

```python
S_aug = TimeMasking(time_mask_param=25)(FrequencyMasking(freq_mask_param=12)(S_t))
# La forma no cambia: (1, 64, 56) -> (1, 64, 56)
```

![Dos espectrogramas Mel lado a lado: el original y el enmascarado, este último con una banda horizontal y una vertical en azul oscuro donde se aplicaron las máscaras](/laboratorios/lab-37/specaugment.png)

**Funciona porque es dropout con estructura.** El dropout clásico apaga unidades sueltas; esto apaga **regiones contiguas**. La diferencia importa: si el modelo aprendió "esto es metal porque hay energía en esta banda", tapar la banda **entera** lo obliga a buscar evidencia redundante en otro lado. Apagar píxeles sueltos no lo lograría, porque los vecinos reconstruyen la señal.

Dos detalles prácticos:

- **Los parámetros son el máximo, no el valor.** torchaudio sortea el ancho en $[0, \text{param}]$ y la posición al azar, así que cada llamada da una máscara distinta — la variabilidad por época sale gratis.
- **La forma no cambia.** El masking escribe ceros, no elimina. Por eso se puede aplicar dentro del `Dataset` sin romper el batching, a diferencia de `time_stretch`.

{{< callout type="info" >}}
**Los valores del notebook son agresivos para este espectrograma.** Con 56 frames, `time_mask_param=25` puede tapar hasta el **45% del tiempo** (580 ms de 1.296). En el paper original de SpecAugment, orientado a reconocimiento de voz, $T=40$ sobre espectrogramas de ~1.000 frames tapa ~4%. Acá es **proporcionalmente 11× más agresivo**.

Para la demostración visual está bien —se quiere que la máscara se vea— pero para entrenar sobre clips cortos habría que escalarlos.
{{< /callout >}}

## Sumar ruido: la perilla es el SNR

$$\text{SNR} = 10 \log_{10}\frac{P_{\text{señal}}}{P_{\text{ruido}}} \text{ [dB]}, \qquad P = \overline{x^2}$$

El factor es **10 y no 20** porque compara potencias, no amplitudes:

| SNR | Potencia | Amplitud |
|---|---|---|
| 20 dB | 100× | 10× |
| 10 dB | 10× | 3,16× |
| 3 dB | 2× | 1,41× |

`Fa.add_noise` resuelve el factor $k$ que multiplica al ruido para lograr el SNR pedido:

$$k = \sqrt{\frac{P_s}{P_n \cdot 10^{\text{SNR}/10}}}$$

**Lo mejor de esta celda es que no le cree a la librería.** Recupera el ruido escalado restando la señal de la mezcla, y aplica la definición para verificar:

```python
ruido_resultante = con_ruido - senal
snr_medido = 10 * torch.log10((senal**2).mean() / (ruido_resultante**2).mean())
# SNR pedido: 20 dB | medido con la formula: 20.0 dB
```

Da 20,0 / 10,0 / 3,0 dB, clavado. Ese patrón vale como hábito: hay implementaciones que definen el SNR sobre amplitudes o que normalizan antes, y terminas con 6 dB de diferencia sin notarlo.

Y en el pipeline real de la Parte 2, la verificación se repite dentro del `collate_fn`, que sortea el SNR **por muestra** en $[10, 20]$ dB. Medido sobre un batch de cuatro: **18,5 / 11,0 / 19,5 / 14,3 dB** — cuatro valores distintos, todos en rango.

## Las reglas, y la que falta

El notebook cierra con dos:

1. **Solo a train.** Nunca a validación ni test — si no, no sabes contra qué mides.
2. **Distinta en cada época.** Como `add_noise` sortea ruido y SNR nuevos en cada batch, sale gratis.

{{< callout type="error" >}}
**Falta una tercera, y muerde más fuerte: augmentar *antes* de dividir train/test es fuga de información.** Si generas cinco versiones de cada clip y después haces el split al azar, terminas con variantes del *mismo* audio en ambos lados. El modelo no generaliza, reconoce. Regla segura: **dividir primero, augmentar después, y solo el lado de train.**
{{< /callout >}}

{{< callout type="warning" >}}
**Y una cuarta que el laboratorio demuestra sin querer.** La regla 1 se cumple correctamente en la Parte 2 —el ruido va solo al `train_loader`— y aun así produce un problema: el `collate_fn` aplica ruido al **100%** de las muestras de entrenamiento, así que el modelo nunca ve audio limpio y aprende a clasificar "música con SNR de 10-20 dB". Evaluarlo en test limpio es evaluarlo **fuera de su distribución**, y cuesta **11,7 puntos**.

La augmentación debería aplicarse con **probabilidad menor a 1** (típicamente 50%), para que el modelo vea ambas distribuciones. Aplicada siempre, deja de ser augmentación y pasa a ser un cambio de dominio. Los números están en la [página siguiente](03-gtzan-mfcc-vs-wav2vec).
{{< /callout >}}

## Sobre el ruido real

El ruido blanco es **espectralmente plano y estacionario**; el ruido del mundo no es ninguna de las dos cosas. Una cafetería tiene voces (estructura armónica), platos (transitorios) y energía concentrada en ciertas bandas. Un modelo entrenado contra siseo plano no aprende a ignorar *eso*.

La fuente está en el propio notebook: **UrbanSound8K**, del anexo, es un catálogo de ruido urbano real (bocinas, taladros, sirenas, aire acondicionado). Cargar un clip de ahí y pasarlo como segundo argumento de `add_noise` es exactamente el mismo llamado.

## Qué nos llevamos

- **El criterio es la invarianza, no la lista.** Una augmentation es válida si el problema es invariante a ella; si modifica la etiqueta, es veneno.
- **Duración y tono vienen acoplados** salvo que se los separe explícitamente con el phase vocoder.
- **La fase lleva el reloj.** Copiarla en vez de regenerarla produce "phasiness" audible.
- **SpecAugment es dropout con estructura**, y no cambia la forma del tensor — por eso encaja en cualquier pipeline.
- **El SNR se verifica con su definición**, no se asume.
- **Aplicar la augmentación siempre la convierte en un cambio de dominio.**

---

**Ver tambien:** [Lab 37 — hub](/laboratorios/lab-37) · Anterior: [El dato de audio](01-el-dato-de-audio) · Siguiente: [GTZAN: MFCC vs wav2vec](03-gtzan-mfcc-vs-wav2vec) · Fundamentos: [Data augmentation de audio](/fundamentos/data-augmentation-de-audio) · [Data augmentation](/fundamentos/data-augmentation) · Paper: [SpecAugment](/papers/specaugment-park-2019).
