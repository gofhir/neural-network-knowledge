---
title: "La arquitectura de TorToise"
weight: 1
math: true
---

El primer notebook clona un repositorio, instala once paquetes con versiones clavadas y llama a dos funciones. Debajo hay **cinco redes neuronales y 4,3 GB de pesos**, y ninguna de ellas fue diseñada para audio.

## Lo que se descarga

`TextToSpeech()` dispara `download_models()`, que trae ocho archivos del repositorio `jbetker/tortoise-tts-v2` en HuggingFace:

| Archivo | Tamaño | Qué es |
|---|---:|---|
| `autoregressive.pth` | 1.637 MB | `UnifiedVoice`: GPT-2 de 30 capas, dim 1024, 16 cabezas |
| `diffusion_decoder.pth` | 1.115 MB | `DiffusionTts`: el DDPM que produce el espectrograma |
| `clvp2.pth` | 930 MB | Contrastive Language-Voice: puntúa candidato contra texto |
| `vocoder.pth` | 373 MB | UnivNet: MEL → onda a 24 kHz |
| `cvvp.pth` | 144 MB | Contrastive Voice-Voice: puntúa contra la voz de referencia |
| `rlg_diffuser.pth` | 96 MB | Latentes aleatorios para `load_voice('random')` |
| `classifier.pth` | 58 MB | **Detector de audio generado por TorToise** |
| `rlg_auto.pth` | 24 MB | Ídem, rama autoregresiva |
| | **≈ 4,28 GB** | |

## El diseño, en una frase

El [paper](/papers/tortoise-tts-betker-2023) justifica la combinación por lo que cada familia sabe hacer:

```
   texto ──┐
           ├──► UnifiedVoice (GPT-2) ──► N candidatos de tokens de voz
 voz ref ──┘         ▲                              │
   (3 clips)         │                              ▼
        │            │                    CLVP + CVVP: puntúan y rankean
        └──► conditioning latent                    │
                     │                              ▼
                     │                        el mejor (top-k)
                     │                              │
                     └──────────────────►  DiffusionTts (DDPM, 30-400 pasos)
                                                    │
                                                    ▼
                                            espectrograma MEL
                                                    │
                                                    ▼
                                          UnivNet ──► onda 24 kHz
```

Los **autoregresivos** son buenos convirtiendo entre dominios **no alineados** —texto y audio no tienen correspondencia uno a uno: no se sabe cuántos frames dura cada fonema—, pero operan en el dominio discreto y Betker atribuye a eso *"la incoherencia borrosa"* de las muestras de DALL·E. Los **DDPM** operan en continuo sin borrosidad ni colapso de modos, pero **no pueden resolver el alineamiento** porque necesitan la forma de salida antes de muestrear.

El autoregresivo resuelve el alineamiento; el difusor resuelve la calidad.

## El TorToise Trick, visible en el código

Después del re-ranking, uno esperaría que el difusor recibiera los tokens ganadores. No es lo que pasa:

```python
best_latents = self.autoregressive(auto_conditioning.repeat(k, 1), text_tokens.repeat(k, 1),
                                   ..., best_results, ..., return_latent=True, clip_inputs=False)
```

El candidato ganador **se vuelve a pasar por el autoregresivo** para extraer su última capa oculta, y *eso* recibe el difusor. Cuesta un forward extra, y el paper (§2.2.2) lo llama *"uno de los mayores contribuyentes a la calidad de salida"* de todos sus ajustes, con la razón: *"el espacio latente del AR es mucho más rico semánticamente que los tokens discretos"*.

Es la misma lógica que condicionar un generador de imágenes en un text encoder congelado en vez de en tokens.

## El re-ranking, y la decisión que lo hace viable

```python
clvp = self.clvp(text_tokens.repeat(batch.shape[0], 1), batch, return_loss=False)
if auto_conds is not None:
    cvvp_accumulator = 0
    for cl in range(auto_conds.shape[1]):
        cvvp_accumulator = cvvp_accumulator + self.cvvp(auto_conds[:, cl].repeat(...), batch, ...)
    cvvp = cvvp_accumulator / auto_conds.shape[1]
    clip_results.append(clvp * clvp_cvvp_slider + cvvp * (1 - clvp_cvvp_slider))
best_results = samples[torch.topk(clip_results, k=k).indices]
```

Dos modelos puntúan cada candidato y se combinan según `clvp_cvvp_slider` (0,5 por defecto): **CLVP** mide *"¿este audio dice este texto?"* y **CVVP** *"¿suena como la voz de referencia?"*, promediado sobre los tres clips.

La decisión que hace viable todo el esquema está en §2.3 del paper: CLVP se entrenó para emparejar **tokens de voz discretos** con tokens de texto, *"de modo que puede rerankear múltiples salidas del AR sin invocar el costoso modelo de difusión"*. En [la medición](02-donde-esta-el-costo) eso cuesta 92 ms por candidato contra los 1,66 s que costaría difundir uno.

{{< concept-alert type="atencion" >}}
**El CVVP solo actúa si pasas `voice_samples`.** La rama es explícita:

```python
if auto_conds is not None:      # auto_conds existe solo si voice_samples != None
    ... cvvp ...
else:
    clip_results.append(clvp)   # solo CLVP
```

`load_voice` acepta también un `.pth` con latentes precomputados, que ahorra el encoder de condicionamiento. Pero al pasar latentes en vez de audio **se pierde silenciosamente el re-ranking por parecido de voz**, sin que nada lo advierta. El notebook está en la rama buena porque pasa los WAV.
{{< /concept-alert >}}

## Los presets, y lo que esconden

```python
kwargs.update({'temperature': .8, 'length_penalty': 1.0, 'repetition_penalty': 2.0,
               'top_p': .8, 'cond_free_k': 2.0, 'diffusion_temperature': 1.0})
presets = {
    'ultra_fast':   {'num_autoregressive_samples': 16,  'diffusion_iterations': 30, 'cond_free': False},
    'fast':         {'num_autoregressive_samples': 96,  'diffusion_iterations': 80},
    'standard':     {'num_autoregressive_samples': 256, 'diffusion_iterations': 200},
    'high_quality': {'num_autoregressive_samples': 256, 'diffusion_iterations': 400},
}
```

Los knobs comunes —`top_p=0.8`, `temperature=0.8`, `repetition_penalty=2`— coinciden exactamente con los que reporta §4 del paper. Código y artículo concuerdan aquí.

Lo que no salta a la vista: **`ultra_fast` es el único preset que fija `cond_free=False`**. No es "el mismo modelo más rápido" — corre un algoritmo distinto. La docstring del propio autor sobre ese parámetro:

> *"La difusión sin condicionamiento hace dos pasadas hacia adelante por cada paso: una con las salidas del modelo autoregresivo y otra sin priors de condicionamiento. La salida de ambas se mezcla según `cond_free_k`. **La difusión sin condicionamiento es lo de verdad, y mejora dramáticamente el realismo**."*

La fórmula está documentada en la línea siguiente:

$$\text{salida} = \text{cond} \cdot (k+1) - \text{uncond} \cdot k, \qquad k = 2$$

Es decir $3 \times \text{condicionado} - 2 \times \text{incondicional}$: **classifier-free guidance con escala 3**, el mismo mecanismo que el `guidance_scale` de [Stable Diffusion](/papers/latent-diffusion-rombach-2022) que se ajustó en el [Lab 29](/laboratorios/lab-29). Apagarla es el equivalente a `guidance_scale=1`, que en imágenes produce resultados lavados y genéricos. En voz produce lo que el enunciado llama "sonido robótico".

## Dos palancas que el notebook no muestra

`tts_with_preset` reenvía `**kwargs` a `tts()`, así que cualquier parámetro extra sobreescribe el preset:

- **`clvp_cvvp_slider`** (default 0,5). Cerca de 1 el candidato elegido sigue mejor el **texto**; cerca de 0 se parece más a la **voz**. Si el resultado "no suena a la persona", esta es la perilla — no el audio de referencia.
- **`k=3`** devuelve los tres mejores candidatos en vez de uno. Como los 96 se generan igual, solo agrega dos pasadas de difusión: **+6,7 % medido**, contra +200 % de re-ejecutar la celda tres veces.

## El detector que viene con el generador

`classifier.pth` es un `AudioMiniEncoderWithClassifierHead` entrenado para decidir si un clip fue producido por TorToise. Betker lo publicó junto al generador.

Es el caso raro donde el mismo autor entrega las dos mitades, y conviene leerlo junto al [fundamento de síntesis de medios](/fundamentos/sintesis-de-medios): detecta **este** modelo, no "voz sintética" en general, y la generalización entre generadores es justo donde la detección falla — como documenta [FaceForensics++](/papers/faceforensics-rossler-2019).

## Un detalle de ingeniería que se repetirá

Todos los modelos se cargan con `.cpu().eval()`. Ninguno vive en GPU. Luego:

```python
self.autoregressive = self.autoregressive.cuda()
auto_latent = self.autoregressive.get_conditioning(auto_conds)
self.autoregressive = self.autoregressive.cpu()      # lo baja apenas termina
```

Ese patrón aparece **cinco veces** en `tts()`: cada modelo sube a VRAM, hace su parte y baja. Es lo que permite que 4,3 GB de pesos quepan cómodamente en una T4 de 15 GB, al costo de mover gigabytes por PCIe en cada transición.

[Wan-Animate](04-wan-animate-por-dentro) lleva el mismo patrón al extremo: cada etapa corre en un **proceso separado**, porque con 14B parámetros vaciar la caché de PyTorch ya no alcanza.

---

**Siguiente:** [Dónde está el costo](02-donde-esta-el-costo) — el desglose medido de los 50 y los 203 segundos.
