---
title: "Decoding Strategies"
weight: 90
math: true
---

Las **decoding strategies** son los algoritmos que convierten la distribución de probabilidad sobre tokens producida por un modelo autoregresivo en una **secuencia concreta** de texto. Son la pieza menos glamorosa del stack de generación —no se entrenan, no se publican papers de "decoding scaling laws"— y, sin embargo, son la palanca operacional más rentable que existe entre el modelo y el usuario. **El mismo modelo congelado, con dos estrategias de decoding distintas, produce outputs cualitativamente diferentes**: una traducción competente o un bucle de repetición, una respuesta factual o una alucinación creativa, código compilable o sopa de tokens.

Este fundamento es transversal: aparece como cierre del [mecanismo de atención](/fundamentos/mecanismo-atencion), como parámetro central de la API de cualquier LLM moderno, como criterio de diseño en [text summarization](/fundamentos/text-summarization) y como gating final del comportamiento de un [foundation model](/fundamentos/foundation-models) en producción.

---

## 1. El setup formal

Un modelo autoregresivo de lenguaje (Transformer decoder, GPT, BART, T5, etc.) computa en cada paso la distribución condicional:

$$P_\theta(y_t \mid x, y_{<t})$$

donde $x$ es el input (prompt, fuente a traducir, documento a resumir, imagen-features en captioning) y $y_{<t} = (y_1, y_2, \ldots, y_{t-1})$ es el prefijo ya generado. El objetivo es producir una secuencia completa:

$$\hat{\mathbf{y}} = (\hat{y}_1, \hat{y}_2, \ldots, \hat{y}_T)$$

Un **decoding algorithm** es una función $g$ que, dado el vector de logits o probabilidades en el paso $t$, elige el token a emitir:

$$\hat{y}_t = g\big(P_\theta(y_t \mid x, y_{<t})\big)$$

La función $g$ puede ser determinista ($\arg\max$), estocástica (sampling), o un algoritmo de búsqueda con estado (beam search). El bucle continúa hasta que $g$ emite un token especial `[EOS]` o se alcanza una longitud máxima.

{{< concept-alert type="clave" >}}
**El modelo solo entrega una distribución. Quién decide el token es el decoder, no el modelo.** Cambiar decoding nunca toca pesos: cambia comportamiento. Esto la hace la palanca más rápida (segundos), más barata (cero compute extra) y más subestimada del stack de generación.
{{< /concept-alert >}}

El espacio de búsqueda es astronómico: con vocabulario $|V| \approx 50\,000$ y longitud $T = 100$, hay $|V|^T = 50000^{100}$ secuencias posibles. **Encontrar el $\arg\max$ exacto sobre secuencias completas es intratable**. Todas las estrategias prácticas son aproximaciones a esa búsqueda imposible.

---

## 2. Greedy decoding

La estrategia más simple: en cada paso, elegir el token con máxima probabilidad.

$$\hat{y}_t = \arg\max_{w \in V} P_\theta(y_t = w \mid x, y_{<t})$$

```python
def greedy_decode(model, x, max_len=100, eos_id=2):
    y = []
    for t in range(max_len):
        logits = model(x, y)            # (|V|,)
        next_token = logits.argmax().item()
        if next_token == eos_id:
            break
        y.append(next_token)
    return y
```

**Costo**: $O(T)$ pasadas forward, sin ramificación. Es lo más rápido posible.

**Propiedades**:
- Determinista: misma entrada produce mismo output.
- Reproducible sin semilla.
- No requiere búsqueda ni ranking.

**Problema fundamental**: greedy es **localmente óptimo, globalmente subóptimo**. La secuencia greedy no es, en general, la secuencia de máxima probabilidad conjunta. Considera el caso donde el token más probable en el paso $t$ lleva a un callejón sin salida donde todos los tokens siguientes son improbables. Un decoder que aceptara una pérdida local pequeña podría llegar a un destino global mejor —pero greedy no mira hacia adelante.

En **open-ended generation** (storytelling, chatbots, completion sin referencia), greedy típicamente colapsa a bucles de repetición. El primer token "atractor" se vuelve cada vez más probable cada vez que se emite, y el modelo entra en un loop tipo `"...the same the same the same..."`. Holtzman et al. (2020) documentaron esto como el modo de falla canónico.

Para **traducción** o **summarization** corta y constrained, greedy es sorprendentemente competente: BLEU dentro de ~1-2 puntos de beam search, a un cuarto del costo. Para tareas factuales y código, es la elección por defecto razonable.

---

## 3. Beam search

Beam search mantiene los **$k$ mejores candidatos parciales** ("beams" o "hipótesis") en cada paso, expandiendo cada uno con todos los tokens posibles y reteniendo los $k$ con mayor log-probabilidad acumulada.

### 3.1 Algoritmo

En el paso $t$, dado el conjunto de beams $B_{t-1} = \{(\mathbf{y}^{(1)}, s^{(1)}), \ldots, (\mathbf{y}^{(k)}, s^{(k)})\}$ donde $s^{(i)}$ es la log-probabilidad acumulada:

$$s^{(i)} = \sum_{\tau=1}^{t-1} \log P_\theta(y_\tau^{(i)} \mid x, y_{<\tau}^{(i)})$$

se calculan los candidatos expandidos $\{(\mathbf{y}^{(i)} \oplus w, s^{(i)} + \log P_\theta(w \mid x, \mathbf{y}^{(i)}))\}$ para todo $i \in \{1, \ldots, k\}$ y todo $w \in V$. De los $k \cdot |V|$ candidatos, se retienen los **$k$ con mayor score**, formando $B_t$.

```python
def beam_search(model, x, k=4, max_len=100, eos_id=2):
    beams = [([], 0.0)]                          # (tokens, log_prob)
    finished = []
    for t in range(max_len):
        candidates = []
        for tokens, score in beams:
            if tokens and tokens[-1] == eos_id:
                finished.append((tokens, score))
                continue
            log_probs = model(x, tokens).log_softmax(-1)   # (|V|,)
            topk = log_probs.topk(k)
            for log_p, w in zip(topk.values, topk.indices):
                candidates.append((tokens + [w.item()], score + log_p.item()))
        beams = sorted(candidates, key=lambda b: b[1], reverse=True)[:k]
        if all(b[0] and b[0][-1] == eos_id for b in beams):
            break
    finished.extend(beams)
    return max(finished, key=lambda b: b[1])[0]
```

**Costo**: $O(T \cdot k \cdot |V|)$ —aunque en la práctica solo se calcula $k$ veces el forward (uno por beam vivo) y se mira el top-$k$ del vocabulario, así que el factor $|V|$ es solo de una operación de sort.

### 3.2 Length normalization

Beam search vainilla **prefiere secuencias cortas**: la log-probabilidad acumulada es suma de términos negativos, así que cada token adicional reduce el score. Esto es un sesgo crítico: el decoder corta antes de terminar la oración, o emite `[EOS]` prematuramente.

La mitigación estándar es dividir por una función de la longitud (Wu et al. 2016, "Google's NMT"):

$$\text{score}(\mathbf{y}) = \frac{1}{\text{lp}(|\mathbf{y}|)} \sum_t \log P_\theta(y_t \mid \cdot), \quad \text{lp}(L) = \frac{(5 + L)^\alpha}{(5 + 1)^\alpha}$$

con $\alpha \in [0.6, 1.0]$ típicamente. Sin length normalization, beam search en NMT cae 5-10 puntos BLEU; con ella, recupera y a veces supera a $\alpha = 1$ (división por longitud lineal).

### 3.3 Por qué beam search domina en NMT, summarization y captioning

Tareas con una **referencia objetiva** (traducción, summarization, image captioning) tienen una propiedad clave: existe un output "correcto" o una familia pequeña de outputs aceptables. Beam search aproxima el MAP (maximum a posteriori) y se aproxima a esa familia. La métrica de evaluación (BLEU, ROUGE, CIDEr) premia coincidencia exacta de n-gramas con la referencia, y beam search está optimizado para producir secuencias de alta probabilidad —que en regimen training-test concordante son cercanas a la referencia.

**Defaults industriales**:
- NMT: $k = 4$ a $8$ con length penalty.
- Summarization (BART, Pegasus, T5): $k = 4$ a $6$ con trigram blocking.
- Image captioning (Show, Attend and Tell; Bottom-Up): $k = 3$ a $5$.

### 3.4 Variantes

- **Diverse beam search** (Vijayakumar et al. 2018): divide los $k$ beams en grupos y penaliza similitud entre grupos. Produce $k$ outputs distintos en lugar de $k$ variaciones triviales de la misma frase.
- **Constrained beam search**: fuerza que ciertos tokens aparezcan en el output (útil para incluir entidades obligatorias en summarization o términos médicos en traducción clínica).
- **Best-first beam search**: ordena globalmente todas las hipótesis pendientes en lugar de podar por paso. Más caro, ocasionalmente mejor.

---

## 4. El problema de beam search en open-ended generation

En NMT y summarization, beam search es excelente. En **open-ended generation** —escribir un cuento, continuar un párrafo, dialogar— es desastroso. Holtzman et al. (2020), "The Curious Case of Neural Text Degeneration", documentaron tres patologías sistemáticas. Ver [el paper completo](/papers/nucleus-sampling-holtzman-2020).

### 4.1 Neural degeneration: bucles de repetición

Beam search en GPT-2 produce outputs como:

> "La Universidad Nacional Autónoma de México (UNAM). La Universidad Nacional Autónoma de México (UNAM). La Universidad Nacional Autónoma de México (UNAM)..."

El mecanismo es vicioso: la primera vez que aparece la frase, su probabilidad es alta porque es coherente. La segunda vez, el modelo —entrenado sobre texto que **sí repite títulos**— le asigna probabilidad **aún más alta** porque acaba de verla. Tercera vez: más alta todavía. Beam search, optimizando log-probabilidad acumulada, sigue ese gradiente local hasta el infinito.

Holtzman mostró que en GPT-2 generando 200 tokens con beam $k = 10$, **el 50% de las secuencias contienen al menos un trigrama repetido**, y muchas degeneran a bucles infinitos.

### 4.2 Surface-level fluency sin contenido

Las secuencias de beam search son gramaticalmente impecables pero semánticamente vacías o tautológicas. El modelo se queda en la "meseta segura" del lenguaje: clichés, conectores genéricos, frases de relleno. Suena fluido en superficie, no dice nada.

### 4.3 La paradoja de la verosimilitud

El hallazgo más contraintuitivo de Holtzman: **el texto humano tiene perplexity más alta que el output de beam search**. Si el modelo asigna probabilidad $P$ a cada token de un párrafo escrito por un humano y compara con $P$ de su propio output beam, el humano gana en log-probabilidad acumulada con frecuencia. Pero el humano se lee mucho mejor.

Implicación profunda: **maximizar log-probabilidad no es maximizar calidad** en generación abierta. La distribución del modelo tiene una "meseta de outputs aburridos de alta probabilidad" que beam search encuentra invariablemente. El texto humano vive en una región de probabilidad media-alta pero **diversa**, no en el pico.

Esto justifica el salto a sampling.

---

## 5. Pure (ancestral) sampling

La estrategia inversa a beam search: en cada paso, **muestrear** de la distribución completa.

$$\hat{y}_t \sim P_\theta(y_t \mid x, y_{<t})$$

```python
def pure_sampling(model, x, max_len=100, eos_id=2):
    y = []
    for t in range(max_len):
        probs = model(x, y).softmax(-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        if next_token == eos_id:
            break
        y.append(next_token)
    return y
```

**Ventajas**:
- Output diverso por construcción.
- No queda atrapado en bucles deterministas.
- Cero costo extra sobre greedy.

**Problema**: la **unreliable tail** de la distribución. Un modelo decente asigna ~99% de probabilidad a unos cientos de tokens razonables y reparte ~1% sobre **decenas de miles** de tokens absurdos. Aunque cada token absurdo individual tenga probabilidad $10^{-5}$, la masa total de la cola es no despreciable, y a lo largo de 100 pasos la probabilidad de muestrear al menos un token tóxico es alta. Una vez emitido, el modelo trata de continuar la secuencia coherentemente con basura previa —y produce más basura.

El output típico de pure sampling con GPT-2 es coherente por 20-30 tokens y luego entra en gibberish: cambios bruscos de tema, palabras inventadas, sintaxis rota.

La solución no es eliminar el sampling —es **truncar la cola**.

---

## 6. Top-k sampling

Propuesto por Fan et al. (2018) para generación de stories. La idea: en cada paso, **conservar solo los $k$ tokens más probables**, renormalizar, y samplear de esa distribución truncada.

$$V^{(k)} = \arg\text{top}_k\, P_\theta(\cdot \mid x, y_{<t})$$

$$P'(w) = \begin{cases} \dfrac{P_\theta(w)}{\sum_{u \in V^{(k)}} P_\theta(u)} & w \in V^{(k)} \\ 0 & \text{otro caso} \end{cases}$$

```python
def top_k_sampling(model, x, k=50, max_len=100, eos_id=2):
    y = []
    for t in range(max_len):
        logits = model(x, y)
        top_logits, top_indices = logits.topk(k)
        probs = top_logits.softmax(-1)
        choice = torch.multinomial(probs, num_samples=1).item()
        next_token = top_indices[choice].item()
        if next_token == eos_id:
            break
        y.append(next_token)
    return y
```

Defaults populares: $k = 40$ en GPT-2 original, $k = 50$ en HuggingFace.

**Problema**: $k$ es **fijo**, pero la entropía de la distribución **varía** por paso. En un contexto restringido ("El capital de Francia es...") la distribución es pico-aguda: el top-1 tiene 95% y el top-50 incluye decenas de tokens irrelevantes. En un contexto abierto ("Una vez en un bosque lejano...") la distribución es plana: tokens 51 a 200 son tan plausibles como tokens 1 a 50, y truncar a 50 corta opciones legítimas.

Top-k es una mejora clara sobre pure sampling, pero introduce un sesgo: subestima la diversidad cuando la entropía es alta, sobre-confía cuando la entropía es baja. La fix es dejar que el corte sea **adaptativo**.

---

## 7. Top-p (nucleus) sampling

Holtzman et al. (2020) propusieron **nucleus sampling**, hoy el estado del arte para open-ended generation. En lugar de fijar un $k$ constante, fijar una **masa de probabilidad** $p$ y dejar que el tamaño del conjunto resulte de los datos.

### 7.1 Definición

El **nucleus** $V^{(p)}$ es el conjunto más pequeño de tokens cuya probabilidad acumulada supera $p$:

$$V^{(p)} = \text{smallest } V' \subseteq V : \sum_{w \in V'} P_\theta(w) \geq p$$

Operativamente: ordenar los tokens por probabilidad descendente, ir acumulando, cortar cuando se cruza el umbral $p$. Luego renormalizar dentro del nucleus y muestrear.

$$P'(w) = \begin{cases} \dfrac{P_\theta(w)}{\sum_{u \in V^{(p)}} P_\theta(u)} & w \in V^{(p)} \\ 0 & \text{otro caso} \end{cases}$$

```python
def nucleus_sampling(model, x, p=0.95, max_len=100, eos_id=2):
    y = []
    for t in range(max_len):
        logits = model(x, y)
        probs = logits.softmax(-1)
        sorted_probs, sorted_idx = probs.sort(descending=True)
        cumulative = sorted_probs.cumsum(-1)
        # cortar al primer índice donde se cruza p
        cutoff = (cumulative > p).nonzero()[0].item() + 1
        nucleus_probs = sorted_probs[:cutoff]
        nucleus_probs = nucleus_probs / nucleus_probs.sum()
        choice = torch.multinomial(nucleus_probs, num_samples=1).item()
        next_token = sorted_idx[choice].item()
        if next_token == eos_id:
            break
        y.append(next_token)
    return y
```

### 7.2 Por qué funciona

El tamaño del nucleus $|V^{(p)}|$ es **dinámico**:
- En un paso de baja entropía (token muy predecible), $|V^{(p)}|$ puede ser 1 o 2 —el modelo se vuelve casi determinista.
- En un paso de alta entropía (continuación abierta), $|V^{(p)}|$ puede ser 200 o 500 —el modelo explora.

Esto es exactamente la propiedad que top-k no tiene: **el corte se adapta a la confianza del modelo**.

Empíricamente, Holtzman mostró que con $p = 0.95$, GPT-2 produce texto cuya distribución de longitudes de frase, diversidad de vocabulario (Zipf coefficient) y self-BLEU **coinciden con el texto humano** mucho mejor que cualquier variante de beam search o top-k.

Default industrial: $p = 0.9$ a $0.95$. Es la elección por defecto en ChatGPT, Claude, Llama-chat y prácticamente todos los LLMs comerciales para tareas conversacionales.

### 7.3 Top-p combinado con top-k

En la práctica se suelen aplicar ambos cortes: primero top-k (por seguridad: nunca samplear de más de $k$ tokens), luego top-p (para que el nucleus sea adaptativo dentro de esos $k$). Equivalente al pseudocódigo anterior con una intersección de conjuntos.

---

## 8. Temperature scaling

Ortogonal a las estrategias anteriores: **antes** de aplicar softmax sobre los logits $u_w$, dividir por una temperatura $T$.

$$P(y_t = w) = \frac{\exp(u_w / T)}{\sum_{j} \exp(u_j / T)}$$

```python
def apply_temperature(logits, T):
    return (logits / T).softmax(-1)
```

### 8.1 Efecto

- **$T < 1$** (típicamente 0.3 a 0.8): la distribución se vuelve **más aguda**, los tokens de alta probabilidad ganan masa, los de baja la pierden. Más determinista, output más conservador.
- **$T = 1$**: distribución original del modelo.
- **$T > 1$** (típicamente 1.2 a 2.0): la distribución se **aplana**, todos los tokens se vuelven más equiprobables. Más diversidad, output más creativo y más arriesgado.
- **$T \to 0$**: límite greedy ($\arg\max$).
- **$T \to \infty$**: distribución uniforme sobre $V$.

### 8.2 Combinabilidad

Temperature es ortogonal a top-k, top-p y beam: se aplica primero, los demás operan sobre la distribución ya temperada. Combinaciones típicas:

| Configuración | Caso de uso |
|---|---|
| `T=0.0` (greedy) | Q&A factual, código, extracción estructurada |
| `T=0.7, top_p=0.95` | Chatbot general, escritura asistida |
| `T=1.0, top_p=0.9` | Brainstorming, creative writing |
| `T=1.2, top_p=0.95` | Storytelling, poesía |
| `T=0.3, top_p=1.0` | Resumen factual, traducción consistente |

La regla operacional que uso en producción: **fijo `top_p=0.95` por defecto y modulo `T` según cuán determinista quiero el output**. Si necesito reproducibilidad bit-a-bit, voy a `T=0`. Si quiero variedad, subo `T` antes que tocar `top_p`.

---

## 9. Comparación de estrategias

| Strategy | Determinismo | Diversidad | Fluidez | Repetición | Costo |
|---|---|---|---|---|---|
| **Greedy** | Alto | Bajo | Alta | Alta | $O(T)$ |
| **Beam ($k=4$)** | Alto | Bajo | Alta | Alta (open) / Baja (NMT) | $O(T \cdot k)$ |
| **Pure sampling** | Bajo | Alto | Baja | Baja | $O(T)$ |
| **Top-k ($k=50$)** | Medio | Medio | Media | Media | $O(T)$ |
| **Top-p ($p=0.95$)** | Medio | Medio-alto | Alta | Baja | $O(T)$ |
| **Beam + n-gram blocking** | Alto | Bajo | Alta | Baja | $O(T \cdot k)$ |
| **Top-p + temperature 1.2** | Bajo | Alto | Alta | Baja | $O(T)$ |

Notas:
- "Fluidez" = ratio de oraciones gramaticalmente correctas y semánticamente plausibles.
- "Repetición" = frecuencia de n-gramas duplicados dentro del output.
- "Determinismo" = reproducibilidad bajo la misma semilla / mismo input.

El sweet spot para open-ended generation es top-p (eventualmente combinado con temperature). El sweet spot para constrained generation con referencia es beam search con length penalty y n-gram blocking.

---

## 10. Métricas para evaluar decoding

No hay una única métrica "calidad del decoding": cada una mide una propiedad distinta.

### 10.1 BLEU / ROUGE

**BLEU** (Papineni et al. 2002) para traducción: precisión de n-gramas (típicamente $n = 1, 2, 3, 4$) entre output y referencia, con brevity penalty.

$$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^N w_n \log p_n\right)$$

**ROUGE** (Lin 2004) para summarization: recall de n-gramas. ROUGE-1 (unigramas), ROUGE-2 (bigramas), ROUGE-L (longest common subsequence).

Ambas requieren **referencias** y miden coincidencia con outputs "correctos". Útiles para NMT y summarization, irrelevantes para open-ended.

### 10.2 Self-BLEU

Mide **diversidad**: BLEU calculado entre **distintos samples del mismo modelo** para la misma entrada. Self-BLEU bajo = diversidad alta. Greedy y beam tienen self-BLEU $\approx 1.0$ (idéntico siempre); top-p con $T > 0$ tiene self-BLEU $\sim 0.3-0.5$ comparable a humanos.

### 10.3 Distinct-n

$$\text{distinct-}n = \frac{\#\text{ n-gramas únicos en el output}}{\#\text{ n-gramas totales en el output}}$$

Mide la fracción de n-gramas que no se repiten. Distinct-1 cerca de 1.0 = sin repetición de tokens. Greedy/beam suelen tener distinct-2 $< 0.3$ en open-ended (mucha repetición); top-p alcanza 0.6-0.8.

### 10.4 Perplexity

$$\text{PPL}(\mathbf{y}) = \exp\left(-\frac{1}{T} \sum_t \log P_\theta(y_t \mid y_{<t})\right)$$

No es estrictamente una métrica de decoding sino del output. Greedy/beam minimizan perplexity por construcción; sampling produce perplexity mayor pero más cercana al texto humano —la paradoja de Holtzman.

### 10.5 Repetition rate

$$\text{rep-}n = \frac{\#\text{ n-gramas que aparecen} \geq 2 \text{ veces}}{\#\text{ n-gramas totales}}$$

Métrica simple y operacional. Si rep-4 > 0.1, hay degeneración seria.

### 10.6 Evaluación humana y MT-Bench

Para LLMs modernos, el ground truth son **rankings humanos** (Chatbot Arena) y **LLM-as-judge** (MT-Bench, AlpacaEval). Estas evaluaciones capturan calidad holística que ninguna métrica automática mide bien. Cuando se reportan resultados de decoding en papers de 2023+, se usa MT-Bench más que BLEU.

---

## 11. Por tarea, qué decoding usar

Esta es la tabla operacional. Defaults razonables que ahorran semanas de tuneo.

| Tarea | Estrategia | Parámetros |
|---|---|---|
| **NMT (traducción)** | Beam search + length penalty | $k = 4$ a $8$, $\alpha = 0.6$ a $1.0$ |
| **Summarization abstractiva** | Beam + trigram blocking | $k = 4$ a $6$, no-repeat-ngram-size = 3 |
| **Image captioning** | Beam search | $k = 3$ a $5$ |
| **Open-ended (storytelling, chat)** | Top-p + temperature | $p = 0.9$ a $0.95$, $T = 0.7$ a $1.0$ |
| **Code generation** | Greedy o low-T top-p | $T = 0$ a $0.2$, $p = 0.95$ |
| **Q&A factual** | Greedy | $T = 0$ |
| **Extracción estructurada (JSON)** | Greedy + constrained decoding | $T = 0$, schema validation |
| **Creative writing / brainstorm** | Top-p + alta temperature | $T = 1.0$ a $1.4$, $p = 0.95$ |
| **Self-consistency CoT** | Top-p sampling, $N$ runs, voto | $T = 0.7$, $N = 5$ a $40$ |
| **Razonamiento simbólico** | Greedy o muy baja $T$ | $T \leq 0.2$ |

Reglas mnemónicas que aplico:
- **Si hay referencia objetiva** (traducción, summary, caption): beam search.
- **Si hay diversidad como objetivo** (storytelling, brainstorm): top-p con temperature.
- **Si la corrección importa más que la variedad** (código, JSON, math): greedy o low-T.
- **Si necesitas $N$ outputs distintos** (re-ranking, self-consistency, A/B): top-p con $T > 0$ y semillas diferentes.

---

## 12. Repetition penalties

Mitigaciones explícitas contra la degeneración por repetición. Se aplican **sobre la distribución** antes del sampling/argmax.

### 12.1 N-gram blocking

Tras emitir un token, marcar como prohibidos todos los tokens que completarían un n-grama (típicamente trigrama) ya generado. Implementación:

```python
def no_repeat_ngram_filter(logits, y, n=3):
    # bloquea tokens que formarían un n-grama ya presente en y
    if len(y) < n - 1:
        return logits
    prefix = tuple(y[-(n-1):])
    banned = set()
    for i in range(len(y) - n + 1):
        if tuple(y[i:i+n-1]) == prefix:
            banned.add(y[i+n-1])
    logits[list(banned)] = float('-inf')
    return logits
```

Default `no_repeat_ngram_size=3` en HuggingFace `generate()` para summarization. Elimina bucles triviales sin sacrificar fluidez.

### 12.2 Frequency penalty (OpenAI API)

Penaliza tokens según **cuántas veces** ya han aparecido:

$$P'(w) \propto P(w) \cdot \exp(-\alpha \cdot \text{count}(w))$$

o equivalentemente sobre logits: $u'_w = u_w - \alpha \cdot \text{count}(w)$. Default sugerido por OpenAI: $\alpha \in [0.1, 0.5]$. Mayor a 1.0 produce outputs incoherentes.

### 12.3 Presence penalty (OpenAI API)

Versión binaria: penaliza por **presencia** (0 o 1), no por frecuencia:

$$u'_w = u_w - \beta \cdot \mathbb{1}[w \in y]$$

Más suave que frequency penalty, útil cuando se quiere variedad temática sin penalizar repetición funcional de stopwords.

### 12.4 Repetition penalty (Keskar et al. 2019, CTRL)

Divide (o multiplica si negativo) los logits de tokens ya emitidos:

$$u'_w = \begin{cases} u_w / \theta & w \in y \\ u_w & w \notin y \end{cases}$$

con $\theta \in [1.0, 1.3]$. Es el default en muchas implementaciones de Llama y derivados.

### 12.5 Cuándo aplicar

- **Summarization**: n-gram blocking obligatorio.
- **Chat / open-ended**: top-p ya reduce repetición; agregar repetition penalty solo si emerge.
- **Code**: no usar penalties —repetición es legítima (`for i in range(n):` muchas veces).
- **Razonamiento estructurado**: tampoco —pasos de CoT pueden compartir conectores legítimamente.

---

## 13. Estrategias avanzadas

Más allá del trío greedy / beam / top-p, el campo ha producido variantes que valen la pena conocer aunque rara vez sean default.

### 13.1 Diverse beam search (Vijayakumar et al. 2018)

Divide los $k$ beams en $G$ grupos. Dentro de cada grupo, beam search estándar. Entre grupos, agrega un término de **diversidad**:

$$\text{score}(\mathbf{y}^{(g)}) = \log P_\theta(\mathbf{y}^{(g)}) - \lambda \sum_{g' < g} \text{sim}(\mathbf{y}^{(g)}, \mathbf{y}^{(g')})$$

Produce $G$ outputs verdaderamente distintos, útiles para captioning multi-hipótesis y NMT con re-ranking.

### 13.2 Mirostat (Basu et al. 2021)

Adaptive sampling que **target una cross-entropy objetivo** $\tau$. En cada paso ajusta dinámicamente un parámetro de truncación para que la entropía instantánea se mantenga cerca de $\tau$. Resultado: outputs con calidad sostenida en generaciones largas, sin degenerar ni divergir. Implementado en llama.cpp y popular en LLMs locales para storytelling de muchos tokens.

### 13.3 Typical decoding (Meister et al. 2022)

Basado en la **information surprise** de cada token. La intuición: en texto humano, cada token suele tener una información cercana a la entropía condicional. Truncar la distribución a los tokens cuya información $-\log P(w)$ está cerca de $H(P)$:

$$V^{\text{typ}} = \{w : | -\log P(w) - H(P) | < \tau\}$$

En benchmarks de open-ended, typical decoding empata con top-p y a veces lo supera en métricas de coherencia.

### 13.4 Contrastive decoding (Li et al. 2022)

Usa **dos modelos**: un "expert" grande y un "amateur" pequeño. Penaliza tokens que el amateur también prefiere (porque suelen ser clichés y patrones spurious):

$$\hat{y}_t = \arg\max_w \left[\log P_{\text{expert}}(w) - \alpha \log P_{\text{amateur}}(w)\right]$$

Empíricamente mejora razonamiento y coherencia. Se usa en algunos LLMs open-source como mecanismo de "self-improvement" en inferencia.

### 13.5 Constrained decoding (grammar, regex, JSON schema)

En cada paso, **filtrar los logits** según restricciones simbólicas. Implementaciones:
- **regex / FSA**: solo permitir tokens consistentes con una expresión regular.
- **JSON schema**: solo tokens que mantengan el output dentro del schema (OpenAI Structured Outputs, Anthropic tool use).
- **CFG / grammar**: para SQL, Python, lenguajes formales (librerías: outlines, lm-format-enforcer, guidance).

Resultado: outputs sintácticamente válidos por construcción. Es la base de "Structured Outputs" en producción y obligatorio en pipelines con downstream parsers estrictos (FHIR, JSON-LD, SQL).

### 13.6 Speculative decoding (Leviathan et al. 2023)

Speedup, no calidad: usar un **draft model** pequeño para generar varios tokens, luego verificar con el modelo grande de una sola pasada. Si verifica, se aceptan; si no, se rechazan desde el punto de divergencia. Speedup 2-3× sin cambio de distribución (matemáticamente equivalente al sampling del modelo grande). Implementado en vLLM, llama.cpp, TGI.

### 13.7 Min-p sampling (2024)

Variante reciente que truncar por **probabilidad mínima relativa al token top-1**:

$$V^{\text{min-p}} = \{w : P(w) \geq p_{\min} \cdot P_{\max}\}$$

con $p_{\min} \in [0.05, 0.1]$. Más robusto que top-p cuando hay outliers en la distribución. Adoptado en algunos forks de Llama y en LM Studio.

---

## 14. APIs comerciales en 2026

Los parámetros estándar expuestos por los principales proveedores. Conocerlos es operacional, no académico.

### OpenAI API

```python
client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    temperature=0.7,          # T scaling
    top_p=0.95,               # nucleus sampling
    frequency_penalty=0.0,    # α en logit space
    presence_penalty=0.0,     # β binario
    max_tokens=1000,
    seed=42,                  # reproducibilidad (best-effort)
    response_format={"type": "json_schema", "json_schema": {...}},
)
```

No exponen `top_k` directamente. Soportan structured outputs vía JSON Schema (constrained decoding).

### Anthropic Claude API

```python
client.messages.create(
    model="claude-opus-4-7",
    messages=[...],
    temperature=1.0,         # default 1.0
    top_p=0.95,
    top_k=50,                # sí exponen top_k
    max_tokens=4000,
    stop_sequences=["</answer>"],
)
```

Anthropic recomienda usar **uno** entre `top_p` y `top_k`, no ambos simultáneamente. Para tool use, el constrained decoding está implícito en la API de tools.

### HuggingFace `generate()`

El más expresivo, expone prácticamente todo:

```python
model.generate(
    inputs,
    do_sample=True,           # False = greedy o beam
    num_beams=1,              # >1 activa beam search
    temperature=0.7,
    top_p=0.95,
    top_k=50,
    repetition_penalty=1.1,
    no_repeat_ngram_size=3,
    length_penalty=1.0,
    max_new_tokens=512,
    num_return_sequences=1,
    diversity_penalty=0.0,    # diverse beam search
    num_beam_groups=1,
    typical_p=1.0,            # typical decoding
)
```

Si querés entender exactamente qué hace cada estrategia, leer el código de `transformers.generation` es la mejor referencia.

### vLLM, llama.cpp, ollama

Implementaciones eficientes para inferencia self-hosted:
- **vLLM**: PagedAttention + speculative decoding + continuous batching. Throughput máximo en GPU.
- **llama.cpp**: quantización + CPU/Metal/CUDA + soporta mirostat y todos los samplers exóticos.
- **ollama**: wrapper amigable sobre llama.cpp, expone los samplers vía `options` en el JSON de request.

Todos exponen los mismos parámetros: temperature, top_p, top_k, repetition_penalty, mirostat, min_p.

---

## 15. Decisiones de diseño en producción

Algunas reglas operacionales para integrar decoding en sistemas reales.

### 15.1 Configuración por tarea, no global

Un mismo servicio que sirve traducción + chat + extracción **no puede tener una sola configuración de decoding**. Idealmente, expone presets nombrados (`"factual"`, `"creative"`, `"strict-json"`) y mapea cada endpoint a uno.

### 15.2 Versionado del decoding como parte del artefacto

El comportamiento del sistema depende del par (modelo, decoding config). Cambiar decoding **es** cambiar el sistema. En producción: versionar la config como parte del manifiesto del servicio, no como string mágico en código.

### 15.3 Reproducibilidad

Para QA y debugging, fijar `seed` (donde exista) y `temperature=0` produce outputs reproducibles. En APIs comerciales la reproducibilidad es **best-effort** —cambios de infraestructura del proveedor pueden romperla. Asumir reproducibilidad solo en self-hosted.

### 15.4 Costo

Beam search cuesta $\approx k\times$ greedy. Pure sampling y top-p cuestan lo mismo que greedy ($O(T)$ forwards). Self-consistency con $N = 40$ cuesta $40\times$ una sola query. Esto es real en la factura cloud.

### 15.5 Latencia

Beam con $k = 8$ tiene throughput ~1/8 de greedy (KV-cache reutilizable mitiga parcialmente). Speculative decoding compensa con 2-3× speedup. Para chat en producción, top-p + temperature en single pass es el mejor trade-off.

### 15.6 Cuándo NO tocar decoding

Si el output con defaults razonables es 80% correcto y necesitás llegar a 95%, **no** lo arregla decoding. Decoding ajusta varianza, no eleva ceiling de capacidad. Para subir ceiling: mejor prompt, RAG, [SFT](/fundamentos/sft) / fine-tuning, modelo más grande.

---

## 16. Conexiones con el resto del curso

Decoding cruza varias clases del programa.

- **[Clase 14 (Transformer)](/clases/clase-14)**: el decoder autoregresivo es el sustrato físico sobre el que opera todo decoding. Masked self-attention y causal masking son las condiciones para que `P(y_t | y_{<t})` esté bien definida.
- **[Clase 20 (ELMo, BERT, GPT, ChatGPT)](/clases/clase-20)**: los LLMs modernos son decoders autoregresivos. Toda la familia GPT depende crucialmente de decoding —ChatGPT con greedy es una experiencia muy distinta a ChatGPT con top-p 0.95.
- **[Clase 22 (Summarization)](/clases/clase-22)**: la clase donde este fundamento se introduce —slides 43-47. Beam search con trigram blocking es estado del arte allí.
- **Papers**:
  - [Nucleus sampling (Holtzman et al. 2020)](/papers/nucleus-sampling-holtzman-2020) — la referencia obligatoria.
- **Fundamentos**:
  - [Text Summarization](/fundamentos/text-summarization) — donde beam search domina.
  - [In-Context Learning](/fundamentos/in-context-learning) — decoding y prompting interactúan: temperature alta puede romper instruction-following.
  - [Transformer](/fundamentos/transformer) — la arquitectura sobre la que opera.
  - [GPT Family](/fundamentos/gpt-family) — los modelos donde top-p brilla.
  - [Mecanismo de atención](/fundamentos/mecanismo-atencion) — base para que el decoder sepa "dónde mirar" en cada paso.

---

## 17. Resumen

- **Decoding** convierte la distribución $P_\theta(y_t \mid x, y_{<t})$ del modelo en una secuencia concreta. No toca pesos: cambia comportamiento. Es la palanca más rápida y barata del stack de generación.
- **Greedy** (argmax por paso) es rápido y reproducible pero localmente óptimo: en open-ended cae en bucles.
- **Beam search** mantiene $k$ candidatos y aproxima el MAP. Domina en NMT, summarization, captioning —donde hay referencia objetiva. Requiere length normalization.
- **Holtzman et al. (2020)** mostraron que beam search **falla en open-ended**: degeneration, surface fluency, paradoja de la verosimilitud.
- **Pure sampling** evita el bucle pero sufre de la unreliable tail (gibberish por tokens improbables).
- **Top-k** (Fan 2018) trunca a los $k$ más probables. Mejora sobre pure sampling pero $k$ es fijo.
- **Top-p / nucleus** (Holtzman 2020) trunca al conjunto más pequeño cuya masa $\geq p$. El tamaño es **adaptativo**. Es el estado del arte para open-ended con $p \in [0.9, 0.95]$.
- **Temperature** es ortogonal: $T < 1$ determiniza, $T > 1$ diversifica. Combinable con todos los demás.
- **Por tarea**: NMT y summarization → beam con length penalty + n-gram blocking. Open-ended → top-p + temperature. Código y factual → greedy o low-T. JSON estricto → constrained decoding.
- **Repetition penalties**: n-gram blocking, frequency/presence penalty (OpenAI), repetition penalty (CTRL). Cada una con su nicho.
- **Estrategias avanzadas**: diverse beam, mirostat, typical, contrastive, constrained, speculative, min-p. Útiles en nichos específicos.
- **APIs 2026**: OpenAI (T, top_p, freq/presence penalty), Anthropic (T, top_p, top_k), HuggingFace (todo), vLLM/llama.cpp/ollama (self-hosted).
- **Decoding NO sube el ceiling**, solo ajusta varianza. Si el modelo no sabe la tarea, ningún decoder se la enseña.

### Enlaces externos

- [Holtzman et al. 2020 — The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
- [Fan et al. 2018 — Hierarchical Neural Story Generation](https://arxiv.org/abs/1805.04833)
- [Wu et al. 2016 — Google's Neural Machine Translation](https://arxiv.org/abs/1609.08144)
- [Vijayakumar et al. 2018 — Diverse Beam Search](https://arxiv.org/abs/1610.02424)
- [Basu et al. 2021 — Mirostat](https://arxiv.org/abs/2007.14966)
- [Meister et al. 2022 — Typical Decoding](https://arxiv.org/abs/2202.00666)
- [Li et al. 2022 — Contrastive Decoding](https://arxiv.org/abs/2210.15097)
- [Leviathan et al. 2023 — Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [HuggingFace — Generation strategies](https://huggingface.co/docs/transformers/generation_strategies)
- [OpenAI API — Sampling parameters](https://platform.openai.com/docs/api-reference/chat/create)
