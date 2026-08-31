---
title: "Better speech synthesis through scaling — TorToise (2023)"
weight: 470
math: true
---

{{< paper-card
    title="Better speech synthesis through scaling"
    authors="James Betker"
    year="2023"
    venue="arXiv:2305.07243"
    arxiv="2305.07243"
    pdf="/papers/tortoise-tts-betker-2023.pdf" >}}
El sistema de clonación de voz que ejecuta el [Laboratorio 44](/laboratorios/lab-44), en reemplazo del [SV2TTS](/papers/sv2tts-jia-2018) que enseña la clase. Su tesis está en el título y se enuncia en la conclusión sin rodeos: *"casi ninguna parte de TorToise fue diseñada específicamente para procesamiento de audio, y sin embargo supera en realismo a todos los modelos de TTS previos"*. La receta es trasplantar a la voz lo que funcionó en imágenes —un decodificador **autoregresivo** al estilo DALL·E, un **DDPM** para decodificar, y **re-ranking contrastivo** al estilo CLIP— y entrenarlo sobre 49.000 horas. Entrenado por una sola persona sobre 8 RTX-3090 durante un año.
{{< /paper-card >}}

---

## El problema

La investigación en TTS venía optimizando **latencia**: modelos eficientes, entrenados sobre datasets pequeños, desplegables a escala. Betker lista las tres razones (§1.1): el deseo de sistemas rápidos, la falta de corpus grandes transcritos, y la dificultad de escalar las arquitecturas encoder-decoder tradicionales.

La generación de imágenes había tomado el camino opuesto: entrenar modelos que producen **resultados de alta calidad sin importar el tiempo de muestreo**. El paper toma esa apuesta y la aplica a voz. El nombre del repositorio es el chiste sobre la consecuencia.

## Las dos familias que combina

**Autoregresivos (DALL·E).** Buenos convirtiendo entre dominios **no alineados** — texto y audio no tienen correspondencia uno a uno: no se sabe de antemano cuántos frames dura cada fonema. Su debilidad es operar en el dominio discreto, y Betker atribuye al decodificador VQVAE aprendido *"la incoherencia borrosa que exhiben la mayoría de las muestras"* de DALL·E.

**DDPM.** Operan en el dominio **continuo**, sin el comportamiento *mean-seeking* que produce borrosidad ni el colapso de modos que produce falta de diversidad. Su limitación es que **no pueden resolver el alineamiento implícito** entre texto y audio, porque necesitan conocer la forma de salida antes de muestrear.

La combinación asigna a cada uno lo que sabe hacer: **el autoregresivo resuelve el alineamiento** (texto → tokens de voz) y **el DDPM resuelve la calidad** (tokens → espectrograma).

## Arquitectura

Cinco redes, que el laboratorio descarga como ~4,3 GB de pesos:

| Componente | Archivo | Rol |
|---|---|---|
| `UnifiedVoice` | `autoregressive.pth` (1,64 GB) | GPT-2 de 30 capas, dim 1024: texto → tokens de voz |
| `CLVP` | `clvp2.pth` (0,93 GB) | Contrastive Language-Voice: puntúa candidato contra texto |
| `CVVP` | `cvvp.pth` (0,14 GB) | Contrastive Voice-Voice: puntúa candidato contra la voz de referencia |
| `DiffusionTts` | `diffusion_decoder.pth` (1,12 GB) | DDPM: tokens → espectrograma MEL |
| `UnivNetGenerator` | `vocoder.pth` (0,37 GB) | Vocoder: MEL → onda a 24 kHz |

El proceso de inferencia (§4): generar muchos candidatos con el autoregresivo, puntuarlos con CLVP y CVVP, quedarse con los mejores $k$, decodificar cada uno con el DDPM y convertirlo a onda. Muestreo con **nucleus sampling** $P = 0{,}8$, penalización de repetición 2 y temperatura 0,8; difusión con **DDIM**, schedule lineal, 64 pasos y guía sin condicionamiento con constante 2.

## Las tres ideas propias

**1. La entrada de condicionamiento (§2.2.1).** Uno o más clips del hablante objetivo se convierten a espectrogramas MEL y pasan por un encoder de auto-atención. Los vectores resultantes se promedian. El autoregresivo y el DDPM tienen **cada uno su propio encoder de condicionamiento**, entrenados por separado.

La intuición que da el paper: el condicionamiento *"provee una forma de que los modelos infieran características vocales como tono y prosodia, de modo que el espacio de búsqueda de salidas posibles para un texto dado se reduce enormemente"*.

**2. El "TorToise Trick" (§2.2.2).** Durante casi todo el entrenamiento el DDPM convierte códigos discretos en espectrogramas. Al converger, se **fine-tunea sobre el espacio latente del autoregresivo** en vez de sobre los tokens. La razón: *"el espacio latente del AR es mucho más rico semánticamente que los tokens discretos"*. Betker lo describe como *"uno de los mayores contribuyentes a la calidad de salida"* entre todos sus ajustes.

**3. CLVP (§2.3).** El equivalente de CLIP para voz, entrenado sobre pares (audio, texto) de forma contrastiva. Y una decisión de diseño que resulta decisiva en la práctica: se entrenó para emparejar **tokens de voz discretos** con tokens de texto, *"de modo que CLVP puede rerankear múltiples salidas del AR sin invocar el costoso modelo de difusión"*.

{{< concept-alert type="clave" >}}
**Esa decisión se puede medir.** En el [Laboratorio 44](/laboratorios/lab-44/02-donde-esta-el-costo), puntuar 16 candidatos costó **1,47 s — el 2,9 % del tiempo total**, o sea 92 ms por candidato. Difundir esos mismos 16 para puntuarlos sobre audio habría costado 26,6 s: **18 veces más**. Sin ese detalle, el re-ranking —que es lo que da la calidad— no sería viable.
{{< /concept-alert >}}

## El dataset

LibriTTS y HiFiTTS suman 896 horas transcritas. Betker construyó además un dataset **extendido de 49.000 horas** raspando audiolibros y podcasts (apéndice A): corte en silencios de 500 ms, se conservan clips de 5 a 20 segundos, y un pipeline de clasificadores propios elimina *"audio con ruido de fondo, música, mala calidad (como llamadas telefónicas), múltiples voces hablando a la vez y reverberación"*.

La transcripción se hizo con un wav2vec2-large que él mismo fine-tuneó **para predecir puntuación**, porque comillas, comas y signos de exclamación importan para generar habla y no suelen incluirse en el entrenamiento de reconocimiento.

Ese filtro define implícitamente qué audio de referencia funciona: el que se parece a la dieta de entrenamiento.

## Donde el código no coincide con el paper

El paper dice que la salida del encoder de condicionamiento *"se promedia para producir un solo vector"*. El código de `models/autoregressive.py` instancia `ConditioningEncoder` con `mean=False`, y esa rama devuelve:

```python
def forward(self, x):
    h = self.init(x)        # Conv1d(80 bandas MEL -> model_dim)
    h = self.attn(h)        # 6 bloques de auto-atencion
    if self.mean:
        return h.mean(dim=2)
    else:
        return h[:, :, 0]   # <- la rama activa: la PRIMERA posicion temporal
```

No es un promedio: es un **token tipo `[CLS]`**, leído después de que seis bloques de auto-atención dejaran que esa posición atendiera a todo el resto. El promedio del paper sí existe, pero un nivel más arriba —**entre clips**— en `get_conditioning`.

Es la misma clase de desajuste que documenta el [Lab 41](/laboratorios/lab-41/04-el-checkpoint-abierto): el artículo describe el diseño pretendido, el checkpoint ejecuta el diseño que quedó.

## Lo que el modelo realmente ve

El laboratorio descarga 90 segundos de voz de referencia y corta tres clips de 10 s. Pero `api.py` los recorta otra vez:

```python
def format_conditioning(clip, cond_length=132300):
    gap = clip.shape[-1] - cond_length
    if gap > 0:
        rand_start = random.randint(0, gap)
        clip = clip[:, rand_start:rand_start + cond_length]
```

$132300 / 22050 = \textbf{6,0 segundos exactos}$, tomados desde un punto **aleatorio**. La rama de difusión recorta a `pad_or_truncate(102400)` a 24 kHz, o sea 4,27 s desde el inicio.

$$\text{rama autoregresiva: } 3 \times 6{,}0 = 18\ \text{s} \qquad \text{rama de difusión: } 3 \times 4{,}27 = 12{,}8\ \text{s}$$

De 90 segundos descargados llega al modelo el **20 %**. Dar más audio de referencia no mejora nada; el techo está en el código. Y como `rand_start` es aleatorio, **el descriptor de voz cambia entre ejecuciones** aunque los archivos sean idénticos.

## Un detalle que la clase no menciona

Junto con los pesos del generador se descarga `classifier.pth` (58 MB): un `AudioMiniEncoderWithClassifierHead` entrenado para **detectar audio producido por TorToise**. Betker publicó el detector con el generador.

Es el caso raro donde el mismo autor entrega las dos mitades, y conviene leerlo junto al [fundamento de síntesis de medios](/fundamentos/sintesis-de-medios): detecta *este* modelo, no "voz sintética" en general, y la generalización entre generadores es justo donde la detección falla — como documenta [FaceForensics++](/papers/faceforensics-rossler-2019).

## Limitaciones reconocibles

- **Latencia.** El propio nombre lo admite. Medido en el laboratorio: **RTF de 13 a 48** en una T4, o sea de 13 a 48 segundos de cómputo por segundo de audio.
- **Un solo idioma.** Los 49.000 h son audiolibros y podcasts en inglés; el tokenizador BPE tiene 255 tokens y pasa por `unidecode`, que aplana los diacríticos.
- **Longitud acotada.** `max_text_tokens=402` y `max_mel_tokens=500`, y cada unidad mel es 1/20 de segundo: el techo son **25 segundos** por invocación.
- **Evaluación propia.** El paper reconoce que comparar TTS es difícil porque los sistemas del estado del arte son cerrados, y construye su propia suite con CLVP como métrica de distancia (análoga al FID) más un wav2vec para medir inteligibilidad.

## Conexión con el laboratorio

El [Lab 44](/laboratorios/lab-44) ejecuta este modelo y mide lo que el paper no reporta:

- **El costo lo domina el decodificador autoregresivo (88 %), no la difusión (3-9 %)** — contraintuitivo, y explicable porque el difusor opera sobre MEL, ~256× comprimido, y paraleliza sobre la longitud.
- El preset `ultra_fast` es el único que fija `cond_free=False`, es decir **apaga la guía sin condicionamiento**. Medido: activarla cuesta **2,14× por paso**. Es classifier-free guidance con escala 3, el mismo mecanismo que el `guidance_scale` de [Stable Diffusion](/papers/latent-diffusion-rombach-2022).
- El condicionamiento codifica **prosodia**, no solo timbre: con la voz de un discurso de graduación, la salida sale a **1,65 palabras/s** contra ~2,5 conversacionales.

---

**Ver también:** [SV2TTS](/papers/sv2tts-jia-2018) (la clonación de voz que enseña la clase) · [Wan-Animate](/papers/wan-animate-2025) (la otra mitad del laboratorio) · [Clase 44](/clases/clase-44) · [Fundamento: síntesis de medios](/fundamentos/sintesis-de-medios) · [Fundamento: modelos de difusión](/fundamentos/modelos-de-difusion) · [Lab 44](/laboratorios/lab-44).
