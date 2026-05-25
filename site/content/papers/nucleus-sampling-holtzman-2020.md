---
title: "Nucleus Sampling (Top-p Decoding)"
weight: 112
math: true
---

{{< paper-card
    title="The Curious Case of Neural Text Degeneration"
    authors="Holtzman, Buys, Du, Forbes, Choi"
    year="2020"
    venue="ICLR 2020"
    pdf="/papers/nucleus-sampling-holtzman-2020.pdf"
    arxiv="1904.09751" >}}
Identifica el fenómeno de **neural text degeneration**: los language models entrenados por maximum-likelihood, cuando se decodifican con beam search o greedy, producen texto repetitivo y genérico — peor según humanos que el propio texto humano, aunque tenga mayor probabilidad bajo el modelo. Simultáneamente, el muestreo puro de la distribución produce gibberish off-topic. El paper propone **Nucleus Sampling (top-$p$)**: en cada paso, samplear del subconjunto más pequeño de tokens cuya probabilidad acumulada supere un umbral $p$ (típicamente $0.95$). El tamaño del nucleus es **dinámico** — se contrae cuando el modelo está seguro y se expande cuando duda. Es el default de facto en GPT-2/3/4, Claude, Llama, HuggingFace `generate()` y prácticamente toda la generación de texto abierta moderna.
{{< /paper-card >}}

---

## El fenómeno de degeneration

La Figura 1 del paper — reproducida en el slide 45 del curso — es la pieza pedagógica más memorable. Dado el contexto canónico:

> "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."

**Beam Search ($b = 32$)** produce:

> "The study, published in the Proceedings of the National Academy of Sciences of the United States of America (PNAS), was conducted by researchers from the **Universidad Nacional Autónoma de México (UNAM)/Universidad Nacional Autónoma de México/Universidad Nacional Autónoma de México/…**"

El modelo entra en un bucle estable. Cada repetición es individualmente probable porque el contexto inmediato la refuerza, y la probabilidad acumulada del bucle supera la de cualquier continuación creativa.

**Pure Sampling** produce, en cambio:

> "They were cattle called Bolivian Cavalleros; they live in a remote desert uninterrupted by town, and they speak huge, beautiful, paradisiacal Bolivian linguistic thing…"

Gramaticalmente plausible, sintácticamente coherente, pero **off-topic** y plagado de neologismos sin sentido. El modelo, al muestrear sin filtro, recoge tokens de la cola poco fiable de la distribución y crea contextos absurdos que luego perpetúa.

El contraste es brutal: dos métodos canónicos fallan de maneras opuestas pero igualmente irrecuperables. Y la Figura 4 demuestra que el problema es estructural: si se fuerza a GPT-2 a generar la frase "I don't know." repetidamente, la probabilidad que asigna a esa misma frase aumenta a cada iteración hasta llegar prácticamente a 1.0. Existe un **positive feedback loop** intrínseco a la arquitectura cuando se decodifica por maximización.

---

## El likelihood paradox

La observación más subversiva del paper aparece en §4.3 — *Natural Language Does Not Maximize Probability*:

| Método | Perplexity ↓ |
|--------|--------------|
| Greedy | 1.50 |
| Beam $b=16$ | 1.48 |
| **Human** | **12.38** |
| Pure Sampling | 22.73 |
| Top-$k = 40$, $t = 0.7$ | 3.48 |
| Nucleus $p = 0.95$ | 13.13 |

La perplexity del texto humano (12.38) es **8× peor** que la del texto producido por beam search (1.48). Sin embargo, el HUSE — métrica que combina evaluación humana y estadística — muestra que beam search queda en niveles bajos mientras nucleus alcanza 0.97.

Significa que la hipótesis ingenua "el mejor decoding es el que produce texto más probable" es **falsa empíricamente**. El texto humano vive en una zona de probabilidad intermedia — ni picos de máxima certidumbre ni gibberish de cola — porque los humanos, según las máximas de Grice (1975), optimizan para informatividad, no para predictibilidad. Decir lo obvio sería violar el *Maxim of Quantity*. La cita exacta del paper:

> "Language models that assign probabilities one word at a time without a global model of the text will have trouble capturing this effect. Grice's Maxims of Communication show that people optimize against stating the obvious."

Cualquier estrategia que maximice likelihood término a término está condenada a producir texto sub-humano en open-ended generation, sin importar cuán bueno sea el modelo subyacente. La solución no es modelos más grandes — es decoding diferente.

---

## Crítica formal de beam search

Beam search mantiene $b$ hipótesis parciales y, en cada paso, las extiende manteniendo las $b$ secuencias con mayor probabilidad acumulada:

$$
\hat{x}_{m+1:m+n} = \arg\max_{x_{m+1:m+n}} \prod_{i=m+1}^{m+n} P(x_i \mid x_{1:i-1})
$$

Tres problemas conceptuales aparecen en open-ended generation:

- **Repetition collapse.** Como la probabilidad de una frase ya emitida tiende a aumentar al re-aparecer, beam search descubre rápidamente que un bucle ofrece la trayectoria de mayor producto acumulado. Una vez dentro del bucle, escapar requeriría una transición de baja probabilidad — y beam search es exactamente el algoritmo que evita esas transiciones.
- **Genericidad.** Las hipótesis elegidas tienden a usar palabras frecuentes ("the", "of", "is") que son seguras en todos los contextos. La diversidad léxica se desploma. La generación se vuelve "lowest common denominator".
- **Search error vs. model error.** Uno podría sospechar que el problema es que beam search no encuentra el máximo global. Stahlke & Byrne (2019) mostraron que **no**: incluso con búsqueda exacta el problema persiste. El issue es **el objetivo**, no la búsqueda.

El paper enfatiza que beam search sigue siendo apropiado para *directed generation* (traducción, summarization extractiva, data-to-text) donde el output está fuertemente restringido por el input. Solo en *open-ended generation* falla.

---

## Top-$k$ sampling: el predecesor

Fan et al. (2018) propusieron: en cada paso, restringir el sampling al conjunto de los $k$ tokens más probables, renormalizar, y muestrear. Formalmente:

$$
V^{(k)} = \{\text{los } k \text{ tokens con mayor } P(x \mid y_{<t})\}
$$

$$
P'(x) =
\begin{cases}
\dfrac{P(x \mid y_{<t})}{\sum_{x' \in V^{(k)}} P(x' \mid y_{<t})} & \text{si } x \in V^{(k)} \\
0 & \text{en otro caso}
\end{cases}
$$

Top-$k$ resolvió el problema de la cola y fue el default empírico (con $k = 40$) durante un par de años. Pero el nucleus paper diagnostica una limitación fundamental: **$k$ es constante, mientras que la distribución no lo es**.

La Figura 5 lo ilustra con dos contextos:

- **Distribución plana** ("She said, 'I never ___"): "thought", "knew", "saw", "did", "said", "wanted", "told", … decenas de continuaciones plausibles, todas con probabilidad similar. Un $k = 10$ corta opciones legítimas.
- **Distribución picuda** ("I ate the pizza while it was ___"): "still" y "hot" capturan ~99% de la masa. Un $k = 10$ incluye 8 tokens basura cuya probabilidad será inflada por la renormalización — exactamente el problema que el truncamiento pretendía evitar.

No existe $k$ óptimo universal. La solución natural: **truncar por masa de probabilidad, no por ranking**.

---

## Nucleus Sampling (Top-p)

Dado el contexto $x_{1:i-1}$ y la distribución $P(x \mid x_{1:i-1})$ sobre el vocabulario $V$, se define el **nucleus** o **top-$p$ vocabulary** $V^{(p)} \subset V$ como el conjunto **más pequeño** tal que:

$$
\sum_{x \in V^{(p)}} P(x \mid x_{1:i-1}) \geq p
$$

Operacionalmente:

1. Ordenar los tokens de $V$ por probabilidad descendente.
2. Acumular probabilidad de arriba hacia abajo.
3. Cortar en el primer token cuya suma acumulada supere $p$.

Después renormalizar dentro del nucleus:

$$
P'(x \mid x_{1:i-1}) =
\begin{cases}
\dfrac{P(x \mid x_{1:i-1})}{p'} & \text{si } x \in V^{(p)} \\
0 & \text{en otro caso}
\end{cases}
$$

donde $p' = \sum_{x \in V^{(p)}} P(x \mid x_{1:i-1})$. Se muestrea desde $P'$.

El valor recomendado en el paper es $p = 0.95$, con $p \in [0.9, 1)$ como rango razonable.

### Tamaño dinámico

A diferencia de top-$k$, el cardinal $|V^{(p)}|$ **varía paso a paso** según la entropía local. En distribuciones picudas (después de "Once upon a"), el nucleus puede tener 1–5 tokens. En distribuciones planas (después de "She said, 'I never"), puede tener 100–500 tokens. Es la primera estrategia de decoding que reacciona a la *confianza* del modelo en cada paso.

### ¿Por qué $p = 0.95$ y no $p = 1$?

Con $p = 1$ el nucleus es todo el vocabulario y caemos en pure sampling. Con $p$ muy bajo (e.g., 0.5) el nucleus se acerca a greedy. El sweet spot empírico ($p \in [0.9, 0.95]$) corta **el último 5–10% de masa de probabilidad**, que es exactamente donde habita la cola poco fiable. El nucleus es entonces la "región de confianza" del modelo — la región donde sus estimaciones son calibradas.

### Implementación canónica

```python
import torch
import torch.nn.functional as F

def nucleus_sampling(logits, p=0.95, temperature=1.0):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Marcar tokens a remover (manteniendo el primero que cruza p)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = False

    sorted_probs[sorted_indices_to_remove] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum()

    next_token_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_indices[next_token_sorted_idx]
```

La implementación de HuggingFace en `transformers.generation.LogitsProcessor.TopPLogitsWarper` sigue este patrón.

---

## Temperature scaling

Mecanismo ortogonal, no rival. Dado el vector de logits $u_{1:|V|}$ previo al softmax:

$$
P(y_t = w \mid x_{1:i-1}) = \frac{\exp(u_w / T)}{\sum_{j} \exp(u_j / T)}
$$

- $T < 1$: la distribución se **agudiza**, los tokens probables ganan más masa. Más determinista.
- $T = 1$: distribución original del modelo.
- $T > 1$: la distribución se **aplana**, todos los tokens se acercan a uniformidad. Más aleatorio.

Temperature se combina con top-$p$: primero ajustar $T$ sobre los logits, luego truncar por nucleus, luego renormalizar y muestrear. En la práctica industrial moderna (ChatGPT, Claude, Llama), el par $(T, p)$ se expone simultáneamente.

El paper advierte que **bajar temperatura** ($T = 0.7$, práctica común en el blog de GPT-2 de OpenAI) reintroduce el problema de la repetición: la Figura 9 muestra que top-$k = 40$ combinado con $T = 0.7$ produce más loops que el mismo top-$k$ a $T = 1$. Intuición: bajar $T$ acerca el sampling a greedy, y greedy es donde habitan los bucles.

---

## Resultados experimentales

**Setup**: GPT-2 Large (762M) entrenado sobre WebText (40 GB). 5.000 generaciones de hasta 200 tokens, condicionadas en el párrafo inicial de documentos held-out.

| Método | Perplexity | Self-BLEU ↓ | Zipf $\hat{s}$ | Repetition % ↓ | HUSE ↑ |
|--------|-----------:|------------:|---------------:|---------------:|-------:|
| **Human** | **12.38** | **0.31** | **0.934** | **0.28** | **1.00** |
| Greedy | 1.50 | — | — | 73.66 | — |
| Beam $b=16$ | 1.48 | 0.44 | 0.967 | 28.94 | — |
| Pure Sampling | 22.73 | 0.28 | 0.926 | 0.22 | 0.67 |
| Top-$k = 40$, $T = 0.7$ | 3.48 | 0.50 | 1.000 | 8.86 | 0.08 |
| Top-$k = 640$ | 13.82 | 0.32 | 0.960 | 0.39 | 0.94 |
| **Nucleus $p = 0.95$** | **13.13** | **0.32** | **0.949** | **0.36** | **0.97** |

Observaciones clave:

- **Greedy y Beam** tienen perplexity catastróficamente baja (1.5) y repetition altísima (28–73%) → degeneration extrema.
- **Pure Sampling** tiene perplexity *más alta* que humano y Zipf excelente, pero HUSE solo 0.67 — la cola produce incoherencia detectable.
- **Top-$k=40, T=0.7$** (la receta popular del blog de GPT-2) tiene HUSE de 0.08 (catastrófico). El paper la destaca como "trampa heurística".
- **Nucleus $p=0.95$** alcanza simultáneamente la perplexity más cercana a humano (13.13), la mejor Self-BLEU (0.32), Zipf 0.95, repetición 0.36% (humano-comparable), HUSE 0.97 (el más alto).

Nucleus es el **único método que aparece cerca del óptimo en todas las columnas simultáneamente**.

---

## Limitaciones

1. **$p$ todavía es un hyperparameter.** El paper recomienda $0.95$, pero distintos dominios y modelos requieren tuning. Lo que nucleus elimina es la rigidez de $k$, no la necesidad de un threshold.
2. **No resuelve hallucination.** Si el modelo asigna probabilidad alta a una afirmación falsa, el nucleus la mantendrá. Mejora la *forma* (coherencia, diversidad, no-repetición), no la *veracidad*.
3. **Métricas automáticas imperfectas.** Self-BLEU, Distinct-$n$, repetition rate son proxies. La evidencia más fuerte viene del HUSE — que sí incluye juicio humano — pero ninguna métrica por sí sola es definitiva.
4. **No considera estructura global.** Nucleus opera token a token; un texto puede ser localmente coherente pero globalmente incoherente. Esto requiere métodos más allá del decoding puntual — planning, lookahead, RLHF.
5. **Stochasticity vs reproducibility.** Nucleus es no-determinista por construcción. Para aplicaciones que requieren determinismo (tests, benchmarks) debe combinarse con seeds fijas.

---

## Impacto y adopción

El impacto del nucleus paper es difícil de exagerar:

- **HuggingFace `transformers`** — `model.generate()` expone `top_p` como parámetro nativo.
- **GPT-2, GPT-3, GPT-4 API** — todas exponen `top_p` con default 1.0 + `temperature` 1.0, ajustables.
- **Claude API** — expone `top_p` y `top_k` simultáneamente.
- **Llama, Mistral, Gemma** — todos los samplers open-source incluyen nucleus.
- **vLLM** — uno de los servers de inferencia más usados, expone `top_p` en `SamplingParams` por defecto.
- **Whisper** (OpenAI, 2022) — usa top-$p$ para la decodificación de audio-to-text.

El paper desencadenó una línea de investigación sobre **decoding strategies más sofisticadas**:

- **Mirostat** (Basu et al., 2021) — extensión adaptiva que mantiene una surprise objetivo constante a lo largo de la generación.
- **Typical Decoding** (Meister et al., 2022) — muestrea de tokens cuyo information content esté cerca de la entropía esperada, motivación information-theoretic más fuerte.
- **Min-$p$ Sampling** (Nguyen, 2024) — corta tokens cuya probabilidad sea menor que una fracción $\alpha$ del top-1. Popular en el ecosistema llama.cpp.
- **Contrastive Decoding** (Li et al., 2023) — usa la diferencia entre un modelo grande y uno chico para guiar el sampling.
- **Speculative Decoding** (Leviathan et al., 2023) — método de aceleración compatible con nucleus.

Pero el paradigma básico — truncar por umbral de masa y muestrear del subconjunto — sigue siendo la formulación dominante. Casi todo lo que se hace en 2025 es **una refinación de nucleus**, no un reemplazo. El paper acumula >5.000 citas en Google Scholar para 2024, entre los 20 papers más citados de ICLR 2020.

A nivel cultural y de producto, nucleus marcó el desplazamiento de la práctica industrial desde "tunear $k$ y rezar" hacia "tunear $p$ con default razonable". El parámetro `top_p` es hoy parte del vocabulario de cualquier API call de LLM.

---

## Conexión con la clase 22

El paper aparece directamente en tres slides del módulo *Text Generation*:

- **Slide 45 — *Decoding: Issues***: reproduce el ejemplo "unicorns / UNAM repetition" del paper. Es el motor visual para introducir el problema.
- **Slide 46 — *Top-p Sampling***: presenta la definición del nucleus, la condición $\sum P \geq p$, y la visualización del subconjunto truncado del vocabulario. Esencialmente la Figura 5 del paper trasladada al deck.
- **Slide 47 — *Temperature Scaling***: complementa con la fórmula softmax-con-$T$, mostrando que $p$ y $T$ son ortogonales y combinables.

La clase 22 trata generación de texto como herramienta integral dentro de **abstractive summarization**. La conexión es directa: cualquier sistema de resumen abstractivo basado en seq2seq o LLM moderno (T5, BART, Pegasus, GPT-4) genera el resumen token a token usando algún decoder. La elección afecta:

- **Si se busca un resumen único óptimo** → beam search (summarization es típicamente *directed*).
- **Si se busca diversidad de resúmenes candidatos** → nucleus + temperature.
- **Si se busca diversidad para re-ranking posterior** → nucleus con $p$ alto + múltiples samples.

---

## Notas y enlaces

- Paper original: [arXiv:1904.09751](https://arxiv.org/abs/1904.09751). ICLR 2020.
- **Figura 1** del paper: tabla del likelihood paradox (PPL beam < PPL humano).
- **Figura 4**: positive feedback loop — probabilidad creciente de la frase "I don't know." al repetirla.
- **Figura 5**: visualización de distribuciones planas vs picudas y el nucleus correspondiente.
- Implementación canónica: [`TopPLogitsWarper`](https://huggingface.co/docs/transformers/internal/generation_utils) en HuggingFace `transformers`.

Ver fundamentos: [Decoding strategies](/fundamentos/decoding-strategies) - [Text Summarization](/fundamentos/text-summarization) - [GPT family](/fundamentos/gpt-family) - [Transformer](/fundamentos/transformer).

Papers relacionados: [GPT-2 (Radford 2019)](/papers/gpt-2-radford-2019) - [GPT-3 (Brown 2020)](/papers/gpt-3-brown-2020).

Clase: [Clase 22 — Abstractive Summarization & Text Generation](/clases/clase-22).
