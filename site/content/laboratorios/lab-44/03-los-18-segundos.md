---
title: "Los 18 segundos de los 90"
weight: 3
math: true
---

El notebook descarga 90 segundos de voz de referencia y corta tres clips de 10 segundos. Parece que el modelo trabaja con 30 segundos de audio. Trabaja con 18 — y con 12,8 en la otra rama.

## El recorte que no se ve

`api.py`, en la ruta que alimenta al autoregresivo:

```python
def format_conditioning(clip, cond_length=132300):
    gap = clip.shape[-1] - cond_length
    if gap < 0:
        clip = F.pad(clip, pad=(0, abs(gap)))
    elif gap > 0:
        rand_start = random.randint(0, gap)
        clip = clip[:, rand_start:rand_start + cond_length]   # recorte ALEATORIO
    mel_clip = TorchMelSpectrogram()(clip.unsqueeze(0)).squeeze(0)
```

$$\frac{132300}{22050} = \textbf{6,0 segundos exactos}$$

Y en la ruta del difusor:

```python
sample = torchaudio.functional.resample(sample, 22050, 24000)
sample = pad_or_truncate(sample, 102400)      # t[..., :length]: los PRIMEROS
```

$$\frac{102400}{24000} = \textbf{4,27 segundos}$$

El balance completo:

| Etapa | Audio | Lo que llega al modelo |
|---|---:|---|
| MP3 descargado | 90 s | — |
| Tres clips cortados con ffmpeg | 30 s | — |
| Condicionamiento **autoregresivo** | | **18,0 s** (3 × 6,0 s, offset aleatorio) |
| Condicionamiento de **difusión** | | **12,8 s** (3 × 4,27 s, desde el inicio) |

**El 20 % de lo descargado.** Dos consecuencias prácticas:

1. **Dar más audio de referencia no mejora nada.** El techo está en el código, no en el material. Lo que sí importa es la *calidad* y la *diversidad* de esos 30 segundos.
2. **`rand_start` es aleatorio**, así que el conditioning latent cambia en cada corrida aunque los archivos sean idénticos. Cachear los latentes en un `.pth` elimina esa fuente de variabilidad — al costo de perder el re-ranking CVVP, como se ve en [la arquitectura](01-la-arquitectura-de-tortoise).

## Por qué los cortes están en 0, 40 y 80

El notebook corta en esos offsets sin explicar por qué. La razón es estadística, y se apoya en que el paper (§2.2.1) dice que los vectores de todos los clips **se promedian**.

La varianza del promedio de $n$ variables **correlacionadas** con correlación $\rho$ no es $\sigma^2/n$:

$$\operatorname{Var}(\bar{x}) = \sigma^2\left(\frac{1}{n} + \frac{n-1}{n}\rho\right)$$

Con $n=3$: si $\rho = 0$, la varianza cae a $0{,}33\sigma^2$. Si $\rho = 0{,}9$ —lo que se obtendría con tres cortes consecutivos de la misma frase— cae solo a $0{,}93\sigma^2$.

**Promediar tres clips casi idénticos no promedia nada.** Separarlos 40 segundos los vuelve razonablemente independientes: distinta frase, distinta entonación, distinto contenido fonético. Ahí el promedio sí estima el "centro" de la voz en vez de quedar sesgado hacia un fragmento.

Para material propio de otra duración, lo que importa no son los números 0/40/80 sino que los cortes estén **repartidos a lo largo del audio**. Y conviene verificar que los tres WAV pesen lo esperado: ffmpeg genera silenciosamente un archivo de 44 bytes —solo cabecera— si el `-ss` se pasa del final.

## El código contradice al paper

El paper describe así el encoder de condicionamiento:

> *"Estos clips se convierten a espectrogramas MEL y se pasan por un encoder que consiste en una pila de capas de auto-atención. **La salida de estas capas se promedia** para producir un solo vector."*

El código, en `models/autoregressive.py`:

```python
class ConditioningEncoder(nn.Module):
    def __init__(self, spec_dim, embedding_dim, attn_blocks=6, num_attn_heads=4,
                 do_checkpointing=False, mean=False):
        ...
    def forward(self, x):
        h = self.init(x)        # Conv1d(80 bandas MEL -> model_dim)
        h = self.attn(h)        # 6 bloques de auto-atencion
        if self.mean:
            return h.mean(dim=2)
        else:
            return h[:, :, 0]   # <- el default
```

Y la instanciación, en la línea 318, **no pasa `mean=True`**:

```python
self.conditioning_encoder = ConditioningEncoder(80, model_dim, num_attn_heads=heads)
```

{{< concept-alert type="clave" >}}
**Dentro de cada clip no hay promedio: hay un token tipo `[CLS]`.** Se lee la primera posición temporal después de que seis bloques de auto-atención dejaran que esa posición atendiera a todo el resto.

Funcionalmente puede agregar información global —es el mismo argumento que justifica el `[CLS]` de BERT— pero *no es un promedio*, y las dos operaciones no son equivalentes: un promedio pondera todas las posiciones por igual, un token aprendido decide qué pesar.

El promedio del paper sí existe, pero **un nivel más arriba, entre clips**:

```python
conds = torch.stack(conds, dim=1)
conds = conds.mean(dim=1)     # aqui si: promedio de los 3 clips
```

Así que la descripción del artículo es correcta a nivel de clips e incorrecta a nivel intra-clip. Es el mismo patrón que documenta el [Lab 41](/laboratorios/lab-41/04-el-checkpoint-abierto): **el paper describe el diseño pretendido, el checkpoint ejecuta el diseño que quedó**.
{{< /concept-alert >}}

## Un parámetro que nadie usa como dice el paper

```python
self.autoregressive = UnifiedVoice(max_mel_tokens=604, max_text_tokens=402,
                                   max_conditioning_inputs=2, layers=30, ...)
```

**`max_conditioning_inputs=2`**: el modelo fue entrenado viendo como máximo dos clips de condicionamiento a la vez. El notebook le pasa **tres**. No rompe nada —`get_conditioning` promedia los que haya— pero es un uso ligeramente fuera de distribución.

## El conditioning captura prosodia, no solo timbre

El paper describe la intuición del condicionamiento así: *"provee una forma de que los modelos infieran características vocales como **tono y prosodia**"*. Eso se puede medir.

La voz de referencia es un discurso de graduación: leído, pausado, con silencios enfáticos. Generando una frase de 11 palabras:

$$\frac{11\ \text{palabras}}{6{,}68\ \text{s}} = \mathbf{1{,}65\ \text{palabras/s}}$$

contra ~2,5 palabras/s del habla conversacional: **un 34 % más lento**. El modelo no copió solo el timbre — copió el ritmo de orador.

Tiene una consecuencia de diseño para quien use el laboratorio: **conviene que el texto generado pertenezca al mismo registro que la voz de referencia**. Pedirle a un descriptor de discurso solemne que diga una frase coloquial rápida es forzarlo fuera de distribución, y el resultado suena mal sin que sea obvio por qué.

## Las tasas de muestreo, y la asimetría

```
ffmpeg -y -loglevel error -stats -i obama.mp3 -ac 1 -ab 64000 -ar 22050 -t 90 reference.wav
```

- **`-ac 1`** — mono. El encoder trabaja sobre un MEL de un canal; una voz es una fuente puntual y el estéreo duplicaría sin aportar.
- **`-ar 22050`** — debe coincidir con `load_audio(cond_path, 22050)`, que está clavado en el código. El valor viene de la convención Tacotron/LJSpeech. Por Nyquist, da una banda útil de **11.025 Hz**: sobra para voz —fundamental entre 85 y 255 Hz, formantes bajo 4 kHz, fricativas hasta 8-10 kHz— e insuficiente para música. Es la discusión de la [clase 35](/clases/clase-35).
- **`-ab 64000`** — **no hace nada.** Fija el bitrate del códec, pero la salida es WAV, o sea PCM sin comprimir: su bitrate lo determina $22050 \times 16 \times 1 \approx 353$ kbps y no hay nada que negociar. Es un fragmento copiado de una receta que producía MP3.

Y la asimetría que importa para el notebook 3: **la entrada es a 22.050 Hz y la salida a 24.000**.

```python
torchaudio.save(f'/content/generated.wav', gen.squeeze(0).cpu(), 24000)
```

No es un error. El condicionamiento y el MEL operan a 22.050 (herencia Tacotron); el vocoder **UnivNet** sintetiza a 24.000. Guardar la salida a 22.050 la reproduce un ~9 % más lenta y más grave — sobre 10 segundos, casi un segundo de desfase al sincronizar con video.

*(El `squeeze(0)` tampoco es decorativo: el return real tiene forma `(1, 1, S)` y `torchaudio.save` espera 2D. La docstring dice "Shape 1,S if k=1", pero el código devuelve 3D.)*

## El idioma

El tokenizador es un `VoiceBpeTokenizer` con **255 tokens**, entrenado sobre texto en inglés, y el *cleaner* pasa por `unidecode`, que aplana los diacríticos: *"corazón"* → *"corazon"*.

- **En español el resultado es malo**: acento anglosajón marcado, fonemas mal mapeados —la /ɲ/ de "ñ" no existe en el inventario—, entonación rara. No está roto: está fuera de distribución.
- **La voz de referencia sí puede ser en español.** El descriptor codifica timbre y prosodia, no idioma.
- Límites duros: `assert text_tokens.shape[-1] < 400` y `max_mel_tokens=500` a 1/20 de segundo por unidad, o sea **25 segundos** máximo por invocación.

## El truco de los corchetes

```python
def __init__(self, autoregressive_batch_size=16, models_dir='.models', enable_redaction=True):
    """
    :param enable_redaction: When true, text enclosed in brackets are automatically redacted
                             from the spoken output (but are still rendered by the model).
                             This can be used for prompt engineering.
    """
```

Está activo por defecto y casi nadie lo usa. El texto entre corchetes **se genera** —condicionando la prosodia— y después **se elimina del audio** con alineamiento forzado wav2vec2:

```python
text = "[I am absolutely thrilled,] This is the last class of the diploma."
```

Sale solo la segunda parte, dicha con esa carga emocional. Es el equivalente TTS del *prompt engineering*, y explica una descarga de ~360 MB adicionales que no aparece en la lista de pesos: el modelo wav2vec2 que hace el alineamiento.

---

**Siguiente:** [Wan-Animate por dentro](04-wan-animate-por-dentro) — el otro modelo, seis años después.
