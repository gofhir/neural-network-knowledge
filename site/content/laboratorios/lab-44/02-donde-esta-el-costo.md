---
title: "Dónde está el costo"
weight: 2
math: true
---

TorToise combina un decodificador autoregresivo con un modelo de difusión. La intuición dice que lo caro es el difusor — es lo que pasa en generación de imágenes, donde los pasos de denoising dominan todo. **Acá no.**

## El desglose

Instrumentando la misma frase con los dos presets sobre una Tesla T4:

| Etapa | `ultra_fast` | `fast` | Ratio |
|---|---:|---:|---:|
| Generación autoregresiva | 36,35 s | 178,2 s (6 × 29,70) | 4,90× |
| Re-ranking CLVP + CVVP | 1,47 s | 9,60 s (6 × 1,60) | 6,53× |
| Difusión + vocoder | 1,66 s (30 @ 18,05 it/s) | 9,48 s (80 @ 8,44 it/s) | 5,71× |
| **Subtotal** | 39,5 s | 197,3 s | |
| **Total medido** | **50,1 s** | **203,2 s** | 4,06× |
| *Residuo* | *10,6 s* | *5,9 s* | |

$$\frac{36{,}35}{50{,}1} = 72{,}6\ \% \qquad\qquad \frac{178{,}2}{203{,}2} = 87{,}7\ \%$$

**El difusor consume el 3,3 % y el 4,7 % del tiempo.** El residuo de la primera corrida —10,6 s contra 5,9 en la segunda— es el cálculo de los latentes de condicionamiento más las cinco transferencias de modelos entre RAM y VRAM, que en la segunda ya no paga warm-up.

## Por qué

Dos razones estructurales:

**El difusor opera sobre espectrogramas MEL**, ~256× comprimidos respecto de la onda cruda. 80 pasos sobre un tensor pequeño es barato.

**El autoregresivo decodifica token a token**, secuencialmente, sin paralelizar sobre la longitud. Cada token exige un forward completo del GPT-2 de 30 capas.

Es la asimetría inversa a la de imágenes, donde el "autoregresivo" (si lo hay) produce pocos tokens y el difusor trabaja sobre un latente grande.

## El warm-up del primer batch

36,35 s el batch único de `ultra_fast` contra 29,70 s promedio en `fast`: el primero es **22 % más lento**. Resolviendo $W + M = 36{,}35$ y $W + 6M = 178{,}2$:

$$M = 28{,}4\ \text{s/batch} \qquad W = 8{,}0\ \text{s de warm-up}$$

Compilación de kernels CUDA, subida de 1,6 GB a VRAM y asignación del *caching allocator*. Se paga una vez.

## `cond_free` medido

$$\frac{18{,}05\ \text{it/s}}{8{,}44\ \text{it/s}} = \mathbf{2{,}14\times}$$

La predicción teórica era exactamente 2×: `cond_free=True` hace dos forwards por paso, uno condicionado y otro sin condicionar. El 7 % extra es la mezcla $3\cdot\text{cond} - 2\cdot\text{uncond}$ y el overhead de kernel.

Esto confirma que `ultra_fast` **no es el mismo algoritmo acelerado**: corre difusión sin guía. La diferencia audible entre los dos presets viene de ahí, no de los 30 contra 80 pasos.

## El re-ranking es prácticamente gratis

1,47 s para puntuar 16 candidatos = **92 ms por candidato**, el **2,9 %** del total. Compáralo con el 1,66 s que cuesta difundir **uno solo**.

Eso valida en números el argumento de diseño del paper (§2.3):

> *"Para que esto funcione eficientemente en inferencia, entrené CLVP para emparejar tokens de voz **discretizados** con tokens de texto. De este modo, CLVP puede rerankear múltiples salidas del AR **sin invocar el costoso modelo de difusión**."*

Si CLVP puntuara audio, habría que difundir los 16 candidatos: $16 \times 1{,}66 = 26{,}6$ s en vez de 1,47 s, **18× más caro**. El re-ranking —que es de donde sale la calidad— no sería viable.

## La corrección a la tabla de presets

Es tentador estimar el costo de `standard` y `high_quality` desde los pasos de difusión. Con el desglose real, esa cuenta da mal. Usando $M = 28{,}4$ s/batch y $W = 8$ s:

| Preset | Batches AR | Cómputo AR | CLVP | Difusión | **Total estimado** | vs `ultra_fast` |
|---|---:|---:|---:|---:|---:|---:|
| `ultra_fast` | 1 | 36 s | 1,5 s | 1,7 s | **50 s** (medido) | 1,0× |
| `fast` | 6 | 178 s | 9,6 s | 9,5 s | **203 s** (medido) | 4,1× |
| `standard` | 16 | 462 s | 26 s | 24 s | **~8,6 min** | ~10,3× |
| `high_quality` | 16 | 462 s | 26 s | 47 s | **~9,0 min** | ~10,8× |

{{< concept-alert type="clave" >}}
**`high_quality` cuesta un 4,6 % más que `standard`, no el doble.** Comparten `num_autoregressive_samples=256` y difieren solo en pasos de difusión — o sea en la parte barata, que pesa el 5 % del total.

Eso explica en términos concretos por qué Betker escribe de `high_quality` que *"no vale realmente la pena el cómputo"*: no es que sea carísimo, es que **la calidad ya estaba saturada en `standard`** y lo único que agrega es la etapa que no limita el resultado.
{{< /concept-alert >}}

## El autoregresivo escala peor que lineal

Generando una frase más larga con `fast`:

| | Frase corta | Frase larga | Ratio |
|---|---:|---:|---:|
| Audio producido (promedio) | 4,02 s | 6,68 s | 1,66× |
| AR por batch | 29,70 s | 85,33 s | **2,87×** |
| Difusión (candidato 0) | 8,44 it/s | 5,78 it/s | 1,46× |

**La difusión escala lineal**, y con precisión notable: $8{,}44/5{,}78 = 1{,}460$ contra $6{,}12/4{,}22 = 1{,}450$ de duración — 0,7 % de error. Tiene sentido, porque procesa el espectrograma completo en paralelo en cada paso.

**El autoregresivo no.** 1,66× de audio produjo 2,87× de cómputo, y la causa no es la complejidad cuadrática de la atención (con KV cache, dimensión 1024 y ~130 tokens, el término del MLP domina). Está en el código:

```python
codes = self.autoregressive.inference_speech(..., num_return_sequences=16,
                                             max_generate_length=max_mel_tokens)  # 500
```

**Los 16 candidatos de un batch se generan en paralelo, así que el batch termina cuando termina el más largo** — no el promedio. El tope es `max_mel_tokens=500`, o sea 25 segundos de audio. Basta con que **un** candidato divague —repita, alargue, no emita el token de parada— para que los otros quince esperen.

Los tres ganadores duraron 6-7 s (≈130 tokens). Entre los 96 generados, alguno pudo irse a 300 o 400. **El costo lo fija el peor candidato del batch, no el que se elige.**

## La GPU se degrada

| Candidato | Duración | it/s medido | it/s según el modelo lineal | Desvío |
|---|---:|---:|---:|---:|
| 0 | 6,12 s | 5,78 | 5,82 | −0,7 % |
| 1 | 7,05 s | 4,34 | 5,05 | **−14 %** |
| 2 | 6,87 s | 4,45 | 5,18 | **−14 %** |

El primero calza con el modelo; los dos siguientes están consistentemente 14 % por debajo. Tras ~9 minutos de carga sostenida, la T4 de Colab **throttlea** — es una tarjeta pasiva de 70 W. Conviene tenerlo en cuenta al cronometrar corridas largas.

## La duración de la salida no es determinista

Tres candidatos de la **misma frase**: 6,12 s, 7,05 s y 6,87 s. Un 15 % de dispersión, por el muestreo estocástico del autoregresivo (dónde decide poner pausas y cuánto alarga las vocales).

Tiene una consecuencia práctica directa: **la sincronización con video debe planificarse sobre la duración medida, no estimada**. Es lo que se hace en [el playbook de alineación](06-el-playbook-de-alineacion).

## Tres fuentes de aleatoriedad

El notebook advierte que *"cada vez que se ejecuta el código se genera un audio diferente"* sin explicar por qué. Son tres, y conviene distinguirlas:

1. **`rand_start` en `format_conditioning`** — el recorte de 6 s dentro de cada clip de 10 s es aleatorio, así que **el descriptor de voz cambia entre corridas**. Es la que nadie ve venir; se detalla en [los 18 segundos](03-los-18-segundos).
2. **Nucleus sampling** del autoregresivo (`top_p=0.8`, `temperature=0.8`).
3. **El ruido inicial del difusor** (`diffusion_temperature=1.0`).

Con `ultra_fast` y solo 16 candidatos la varianza entre corridas es alta: el mejor de 16 muestras es bastante peor que el mejor de 96. El enunciado sugiere *"correr la celda varias veces hasta conseguir un resultado de su agrado"* — pero pasar a `fast` logra lo mismo por menos, porque muestrea 96 y devuelve uno.

---

**Siguiente:** [Los 18 segundos de los 90](03-los-18-segundos) — cuánto audio de referencia llega realmente al modelo.
