---
title: "The Curious Case of Neural Text Degeneration (Holtzman et al., 2020)"
slug: nucleus-sampling-holtzman-2020
authors: ["Ari Holtzman", "Jan Buys", "Li Du", "Maxwell Forbes", "Yejin Choi"]
year: 2020
venue: "ICLR 2020"
arxiv: "1904.09751"
tags: ["decoding", "sampling", "language-generation", "top-p", "nucleus-sampling", "beam-search", "open-ended-generation"]
clase: 22
---

# The Curious Case of Neural Text *De*generation — Análisis exhaustivo (Holtzman et al., 2020)

**Referencia bibliográfica**
Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020). *The Curious Case of Neural Text Degeneration*. International Conference on Learning Representations (ICLR) 2020. arXiv:1904.09751.

Cinco autores de Paul G. Allen School (UW) y Allen Institute for AI, más Jan Buys de University of Cape Town. Subido a arXiv en abril 2019, versión final v2 en febrero 2020 tras la aceptación en ICLR. El paper introduce el término *neural text degeneration* y propone el método **Nucleus Sampling** (más conocido como **top-p sampling**), que se convirtió en el default de facto para muestreo en GPT-2, GPT-3, GPT-4, Claude, Llama y prácticamente toda la generación de texto abierta moderna. La cita es obligatoria en cualquier discusión seria de decoding estratégico.

---

## 1. Resumen ejecutivo

El paper hace tres cosas:

1. **Diagnostica un fenómeno** que los autores bautizan *neural text degeneration*: los language models entrenados con maximum-likelihood, cuando se decodifican con beam search o greedy, producen texto repetitivo, genérico e incoherente — peor según humanos que el propio texto humano, aunque tenga mayor likelihood bajo el modelo. Simultáneamente, el muestreo puro de la distribución produce gibberish off-topic. Ambos extremos son malos pero por razones distintas.

2. **Critica formalmente las alternativas existentes** (greedy, beam search, pure sampling, top-$k$, temperature) mostrando que cada una falla en algún régimen de entropía. Especialmente importante: ni $k$ fijo ni $t$ fijo se adaptan al hecho de que la distribución sobre el siguiente token cambia drásticamente de paso a paso — a veces es picuda (un solo token domina), a veces es plana (decenas o cientos de tokens son razonables).

3. **Propone Nucleus Sampling (top-$p$)**: en cada paso, samplear del subconjunto más pequeño de tokens cuya probabilidad acumulada supere un umbral $p$ (típicamente $p = 0.95$). El tamaño del *nucleus* es **dinámico** — se contrae cuando el modelo está seguro y se expande cuando duda.

Las contribuciones empíricas — sobre GPT-2 Large con WebText — muestran que el nucleus alcanza simultáneamente la perplexity más cercana a humano, la mejor Self-BLEU (diversidad), el coeficiente de Zipf más similar al humano, una tasa de repetición de bigramas comparable a la humana, y el mejor HUSE (Human Unified with Statistical Evaluation). Es el único método que satisface todos los criterios distribucionales simultáneamente.

El paper es notable también por su análisis del **likelihood paradox**: el texto humano tiene perplexity más alta (12.38) que el texto de beam search (1.48), pero los humanos prefieren claramente el primero. Esto refuta empíricamente la hipótesis de que "más probable bajo el modelo" implica "más humano".

---

## 2. Contexto histórico: el momento decoding-as-bottleneck (2018–2019)

Para entender la urgencia del paper conviene situarlo cronológicamente:

1. **Fan et al. (2018)** — *Hierarchical Neural Story Generation* introdujo **top-$k$ sampling** ($k = 10$) como alternativa a beam search para generación de historias. Fue el primer reconocimiento serio de que el muestreo era preferible a maximización en open-ended generation.

2. **Holtzman et al. (2018)** — *Learning to Write with Cooperative Discriminators* — primer paper del primer autor del nucleus paper, ya documentando que beam search "degenera" en bucles. El nucleus paper es la generalización formal de aquel hallazgo.

3. **Radford et al. (2019)** — **GPT-2** (febrero 2019). OpenAI publica el modelo de 1.5B parámetros y, en el blog post, muestra el ejemplo viral de los unicornios andinos. El generation del blog usa top-$k = 40$ + temperature $t = 0.7$, una combinación heurística sin justificación principlada.

4. **Welleck et al. (2020)** — *Neural Text Generation with Unlikelihood Training* — aborda el mismo problema desde el lado del entrenamiento (penalizar tokens repetidos en el loss). El nucleus paper se posiciona como complementario: solución de inferencia que no requiere reentrenar.

5. **Kulikov et al. (2019)**, **Vijayakumar et al. (2018)**, **Li et al. (2016)** — varios trabajos sobre *diverse beam search* tratan de modificar la búsqueda para forzar variedad. El nucleus argumenta que ninguna utility function rescata a beam search si el objetivo de fondo (maximizar likelihood) es el equivocado.

El paper aterriza en abril 2019, dos meses después del lanzamiento de GPT-2, justo cuando la comunidad NLP descubría empíricamente que tener un buen language model no garantizaba buena generación — el decoding era el cuello de botella visible.

---

## 3. El fenómeno: ejemplos llamativos de *degeneration*

La Figura 1 del paper, reproducida casi literalmente en el slide 45 del curso, es la pieza pedagógica más memorable. Dado el contexto:

> "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."

**Beam Search ($b = 32$)** produce:

> "The study, published in the Proceedings of the National Academy of Sciences of the United States of America (PNAS), was conducted by researchers from the **Universidad Nacional Autónoma de México (UNAM)/Universidad Nacional Autónoma de México/Universidad Nacional Autónoma de México/Universidad Nacional Autónoma de…**"

El modelo entra en un bucle estable. Cada repetición del nombre largo es individualmente probable (porque el contexto inmediato lo refuerza), y la probabilidad acumulada del bucle supera la de cualquier continuación creativa.

**Pure Sampling** (muestreo directo de $P$) produce:

> "They were cattle called Bolivian Cavalleros; they live in a remote desert uninterrupted by town, and they speak huge, beautiful, paradisiacal Bolivian linguistic thing…"

Gramaticalmente plausible, sintácticamente coherente, pero **off-topic** y plagado de neologismos sin sentido ("paradisiacal Bolivian linguistic thing", "umidauda" en otro ejemplo del paper). El modelo, al muestrear sin filtro, recoge tokens de la *cola* poco fiable de la distribución y crea contextos absurdos que luego perpetúa.

El contraste es brutal: dos métodos canónicos de decoding fallan de maneras opuestas pero igualmente irrecuperables. Y los autores demuestran (Figura 4 del paper) que el problema es estructural: si uno fuerza al modelo GPT-2 a generar la frase "I don't know." repetidamente, la probabilidad asignada a esa misma frase aumenta a cada iteración hasta llegar prácticamente a 1.0. Existe un **positive feedback loop** intrínseco a la arquitectura cuando se decodifica por maximización.

---

## 4. El likelihood paradox

Una de las observaciones más subversivas del paper aparece en §4.3 — *Natural Language Does Not Maximize Probability*. Los autores reportan:

| Método | Perplexity ↓ |
|--------|--------------|
| Greedy | 1.50 |
| Beam $b=16$ | 1.48 |
| **Human** | **12.38** |
| Pure Sampling | 22.73 |
| Top-$k = 40$, $t = 0.7$ | 3.48 |
| Nucleus $p = 0.95$ | 13.13 |

La perplexity del texto humano (12.38) es **8× peor** que la del texto producido por beam search (1.48). Sin embargo, el HUSE — métrica que combina evaluación humana y estadística — muestra que beam search se queda en niveles bajos mientras nucleus alcanza 0.97.

¿Qué significa? Significa que la hipótesis ingenua "el mejor decoding es el que produce texto más probable bajo el modelo" es **falsa empíricamente**. El texto humano vive en una zona de probabilidad intermedia — ni picos de máxima certidumbre ni gibberish de cola — porque los humanos, según las máximas de Grice (1975), optimizan para informatividad, no para predictibilidad. Decir lo obvio sería violar el *Maxim of Quantity*. La cita exacta del paper es:

> "Language models that assign probabilities one word at a time without a global model of the text will have trouble capturing this effect. Grice's Maxims of Communication show that people optimize against stating the obvious. Thus, making every word as predictable as possible will be disfavored."

Esta observación tiene consecuencias profundas: cualquier estrategia de decoding que maximice likelihood término a término (greedy, beam) está condenada a producir texto sub-humano en open-ended generation, sin importar cuán bueno sea el modelo subyacente. La solución no es modelos más grandes — es decoding diferente.

---

## 5. Crítica formal de beam search

Recordemos la decomposición autoregresiva left-to-right:

$$
P(x_{1:m+n}) = \prod_{i=1}^{m+n} P(x_i \mid x_1, \ldots, x_{i-1})
$$

donde $x_{1:m}$ es el contexto fijo y $x_{m+1:m+n}$ es la continuación a generar.

**Beam search** mantiene un conjunto de $b$ hipótesis parciales y, en cada paso, las extiende manteniendo las $b$ secuencias con mayor probabilidad acumulada. Asymptoticamente busca:

$$
\hat{x}_{m+1:m+n} = \arg\max_{x_{m+1:m+n}} \prod_{i=m+1}^{m+n} P(x_i \mid x_{1:i-1})
$$

Tres problemas conceptuales se hacen evidentes en open-ended generation:

**(a) Repetition collapse**. Como la probabilidad de una frase ya emitida tiende a aumentar al re-aparecer (Figura 4 del paper), beam search descubre rápidamente que un bucle ofrece la trayectoria de mayor producto acumulado. Una vez dentro del bucle, escapar requeriría una transición de baja probabilidad — y beam search es exactamente el algoritmo que evita esas transiciones.

**(b) Genericidad**. Las hipótesis que beam search elige tienden a usar palabras frecuentes ("the", "of", "is") que son seguras en todos los contextos. La diversidad léxica (Self-BLEU, Zipf) se desploma. La generación se vuelve "lowest common denominator", como dicen los autores.

**(c) Search error vs. model error**. Uno podría sospechar que el problema es que beam search no encuentra realmente el máximo global — que existen continuaciones de aún mayor probabilidad pero la búsqueda las pierde. Stahlke & Byrne (2019) mostraron que esto **no** es lo que ocurre: incluso con búsqueda exacta el problema persiste. El issue es **el objetivo**, no la búsqueda.

El paper enfatiza que beam search sigue siendo apropiado para *directed generation* (traducción, summarization extractiva, data-to-text) donde el output está fuertemente restringido por el input. Es solo en *open-ended generation* — story continuation, dialog libre, generación condicional con mucho grado de libertad — donde falla.

---

## 6. Crítica formal de pure sampling

El esquema opuesto es muestrear directamente de la distribución condicional:

$$
y_t \sim P(\cdot \mid y_{<t})
$$

El problema diagnosticado por el paper se llama **unreliable tail**. La distribución del modelo, después de softmax, asigna probabilidad estrictamente positiva a todos los $|V|$ tokens del vocabulario (∼50.000 en GPT-2). Los tokens más allá del top-100 pueden tener probabilidades individuales del orden $10^{-5}$, pero su **suma acumulada** representa típicamente entre 5% y 50% de la masa total — dependiendo de la entropía del paso.

Cuando se samplea sin filtrar, el evento "tomar un token de la cola" ocurre con probabilidad apreciable. Y un token de cola, casi por definición, no es contextualmente apropiado — el modelo le asignó probabilidad baja precisamente porque "no encaja". Una vez que ese token entra a la secuencia, el modelo lo condiciona en el siguiente paso, propagando el error y haciendo que la generación derive en territorio incoherente. El ejemplo de los "Bolivian Cavalleros" o el neologismo "umidauda" son síntomas directos: el sampling tomó un token inverosímil, y los pasos siguientes lo perpetuaron.

Formalmente: la cola de la distribución es **estimada poco confiablemente**. El modelo nunca vio suficientes ejemplos de cada token raro para asignarle una probabilidad calibrada — su mass es ruido residual del softmax. Pure sampling le da peso a ese ruido. Es la versión decoding del problema de calibración: el ranking de los top-$N$ tokens es informativo, pero las magnitudes absolutas más allá del top-$N$ son artefactos.

---

## 7. Top-$k$ sampling: el predecesor cercano

Fan et al. (2018) propusieron una solución pragmática: en cada paso, restringir el sampling al **conjunto de los $k$ tokens más probables**, renormalizar, y muestrear de ahí. Formalmente, dado el ranking descendente:

$$
V^{(k)} = \{\text{los } k \text{ tokens con mayor } P(x \mid y_{<t})\}
$$

$$
P'(x) =
\begin{cases}
\dfrac{P(x \mid y_{<t})}{\sum_{x' \in V^{(k)}} P(x' \mid y_{<t})} & \text{si } x \in V^{(k)} \\[6pt]
0 & \text{en otro caso}
\end{cases}
$$

Top-$k$ resolvió el problema de la cola en pure sampling y por unos años fue el default empírico (en particular con $k = 40$). Pero el nucleus paper diagnostica una limitación fundamental: **$k$ es un hyperparameter constante, mientras que la distribución no lo es**.

La Figura 5 del paper ilustra el punto con dos contextos:

- **Distribución plana** ("She said, ‘I never ___"): el siguiente token puede ser razonablemente "thought", "knew", "saw", "did", "said", "wanted", "told", "liked", "would", "heard", "want", "could", … decenas de continuaciones plausibles, todas con probabilidad similar. Aquí un $k = 10$ corta opciones legítimas.

- **Distribución picuda** ("I ate the pizza while it was ___"): "still" y "hot" capturan ~99% de la masa. Aquí un $k = 10$ incluye 8 tokens basura que tendrán su probabilidad inflada por la renormalización — exactamente el problema que el truncamiento pretendía evitar.

No existe un $k$ óptimo universal. Si es bajo, sub-truncamos las distribuciones planas (texto genérico). Si es alto, sobre-incluimos tokens basura en las picudas. La solución natural: **truncar por masa de probabilidad, no por ranking**.

---

## 8. Nucleus Sampling — la propuesta

Dado el contexto $x_{1:i-1}$ y la distribución $P(x \mid x_{1:i-1})$ sobre el vocabulario $V$, definimos el **nucleus** o **top-$p$ vocabulary** $V^{(p)} \subset V$ como el conjunto **más pequeño** tal que:

$$
\sum_{x \in V^{(p)}} P(x \mid x_{1:i-1}) \geq p
$$

Operacionalmente:

1. Ordenar los tokens de $V$ por probabilidad descendente.
2. Acumular probabilidad de arriba hacia abajo.
3. Cortar en el primer token cuya suma acumulada supere $p$.

Después, renormalizar dentro del nucleus:

$$
P'(x \mid x_{1:i-1}) =
\begin{cases}
\dfrac{P(x \mid x_{1:i-1})}{p'} & \text{si } x \in V^{(p)} \\[6pt]
0 & \text{en otro caso}
\end{cases}
$$

donde $p' = \sum_{x \in V^{(p)}} P(x \mid x_{1:i-1})$.

Y muestrear desde $P'$.

El value de referencia recomendado en el paper es $p = 0.95$, con $p \in [0.9, 1)$ como rango razonable.

### 8.1 La propiedad clave: tamaño dinámico del nucleus

A diferencia de top-$k$, el cardinal $|V^{(p)}|$ **varía paso a paso** según la entropía local de la distribución. En distribuciones picudas (e.g., después de "Once upon a"), el nucleus puede tener 1–5 tokens. En distribuciones planas (e.g., después de "She said, ‘I never"), puede tener 100–500 tokens. Es la primera estrategia de decoding que reacciona a la *confianza* del modelo en cada paso individual.

El slide 46 del curso reproduce literalmente la visualización de la Figura 5: el nucleus se dibuja como un sub-conjunto coloreado de las barras del histograma de probabilidades, dejando el "unreliable tail" cortado a cero.

### 8.2 ¿Por qué $p = 0.95$ y no $p = 1$?

Con $p = 1$ el nucleus es todo el vocabulario y caemos en pure sampling. Con $p$ muy bajo (e.g., 0.5) el nucleus se acerca a greedy. El sweet spot empírico ($p \in [0.9, 0.95]$) tiene una interpretación neta: cortamos **el último 5–10% de masa de probabilidad**, que es exactamente donde habita la cola poco fiable. El nucleus es entonces "la región de confianza" del modelo, en el sentido frecuentista — la región donde sus estimaciones son calibradas.

---

## 9. Temperature scaling — complementario, no rival

El paper dedica §3.3 a la temperatura, no como alternativa sino como mecanismo ortogonal. Recordemos: dado el vector de logits $u_{1:|V|}$ previo al softmax,

$$
P(y_t = V_\ell \mid x_{1:i-1}) = \frac{\exp(u_\ell / t)}{\sum_{\ell'} \exp(u_{\ell'} / t)}
$$

- $t < 1$: la distribución se **agudiza**, los tokens probables ganan más masa, la cola se aplana. Más determinista.
- $t = 1$: distribución original del modelo.
- $t > 1$: la distribución se **aplana**, todos los tokens se acercan a uniformidad. Más aleatorio.

Temperature se puede combinar con top-$p$ o top-$k$: primero ajustamos $t$ sobre los logits, luego truncamos por nucleus, luego renormalizamos y muestreamos. En la práctica industrial moderna (ChatGPT, Claude, Llama), el par $(t, p)$ se expone simultáneamente al usuario.

El paper advierte explícitamente que **bajar temperatura** ($t = 0.7$, una práctica común en el blog de GPT-2 de OpenAI) reintroduce el problema de la repetición: la Figura 9 muestra que top-$k = 40$ combinado con $t = 0.7$ produce más loops que el mismo top-$k$ a $t = 1$. La intuición: bajar $t$ acerca el sampling a greedy, y greedy es donde habitan los bucles.

---

## 10. Métricas usadas en el paper

El paper hace un esfuerzo importante por evaluar generación con **varios criterios independientes**, reconociendo que ninguno por sí solo basta. Los cinco principales son:

### 10.1 Perplexity de los samples

Calcular perplexity del texto generado bajo el propio modelo:

$$
\text{PPL} = \exp\!\left(-\frac{1}{n}\sum_{i} \log P(x_i \mid x_{<i})\right)
$$

El paper argumenta — esto es contra-intuitivo — que el decoding ideal debe producir texto cuya perplexity sea **cercana a la del texto humano** (12.38 sobre GPT-2 Large WebText held-out). Perplexity demasiado baja indica greedy/beam (texto sobre-predecible). Perplexity demasiado alta indica gibberish.

### 10.2 Self-BLEU (Zhu et al., 2018)

Para cada generación, computar BLEU contra todas las demás generaciones del mismo método como referencias. Lower Self-BLEU implica mayor diversidad inter-muestra. Es la métrica directa para detectar el síndrome de mode collapse — cuando un método produce siempre lo mismo.

Self-BLEU humano: 0.31. Beam $b=16$: 0.44. Nucleus $p=0.95$: 0.32 (casi indistinguible del humano).

### 10.3 Distinct-$n$

Razón entre $n$-gramas únicos y $n$-gramas totales en una generación. Captura novedad léxica intra-muestra. No reportado en la Tabla 1 principal pero usado en el análisis.

### 10.4 Coeficiente de Zipf

Los humanos producen texto cuyo ranking de frecuencias léxicas sigue la ley de Zipf: $f(r) \propto 1/r^s$ con $s \approx 0.93$. El paper computa $\hat{s}_{\text{zipf}}$ de cada método (Figura 7):

- Gold (humano): 0.934
- Pure sampling: 0.926 (muy cercano)
- Nucleus $p=0.95$: 0.949 (cercano)
- Beam $b=16$: 0.967 (sobre-rep. de palabras frecuentes)
- Top-$k=40, t=0.7$: 1.000 (muy desviado)

### 10.5 Repetition rate

Porcentaje de bigramas (o frases mínimas de longitud 2) que se repiten al menos tres veces dentro de la generación.

- Humano: 0.28%
- Beam $b=16$: 28.94% (100× más repetitivo)
- Greedy: 73.66% (catastrófico)
- Nucleus $p=0.95$: 0.36% (humano-comparable)

### 10.6 HUSE (Hashimoto et al., 2019)

Human Unified with Statistical Evaluation: combina anotaciones humanas y likelihood del modelo entrenando un discriminador KNN que distingue texto humano de texto generado a partir de dos features (probabilidad del modelo + juicio humano de tipicidad). Un HUSE = 1 implica indistinguibilidad perfecta.

- Pure sampling: 0.67
- Top-$k=40$: 0.19
- Top-$k=640$: 0.94
- Nucleus $p=0.95$: **0.97** (el más alto)

---

## 11. Resultados experimentales

**Setup**: GPT-2 Large (762M parámetros) entrenado sobre WebText (40 GB). 5.000 generaciones de hasta 200 tokens cada una, condicionadas en el párrafo inicial (1–40 tokens) de documentos held-out del WebText.

La Tabla 1 del paper (reproducida arriba en §4) es la prueba directa. Las observaciones clave:

1. **Greedy y Beam** tienen perplexity catastróficamente baja (1.5) y repetition altísima (28–73%) → degeneration extrema.

2. **Stochastic Beam** mejora la repetición pero todavía falla en HUSE.

3. **Pure Sampling** tiene perplexity *más alta* que humano (22.73 vs 12.38), Zipf coefficient excelente (0.93), pero el HUSE es solo 0.67 porque la cola produce incoherencia detectable por humanos.

4. **Sampling con $t = 0.9$**: trade-off, no logra el balance.

5. **Top-$k$ vs $k$**: con $k = 40$ se mejora vs pure, pero HUSE solo 0.19 (los humanos detectan la sub-truncación). Subir a $k = 640$ ayuda con HUSE (0.94) pero la perplexity ya cae a 13.82 y Zipf a 0.96.

6. **Top-$k = 40$, $t = 0.7$** (la receta popular del blog de GPT-2): repetition rate de 8.86%, perplexity de 3.48 (muy baja). HUSE de 0.08 (catastrófico). Esta es la métrica que el paper destaca como una "trampa heurística".

7. **Nucleus $p = 0.95$**: perplexity 13.13 (la más cercana a humano), Self-BLEU 0.32 (mejor diversidad), Zipf 0.95, repetición 0.36% (humano-comparable), HUSE 0.97 (el más alto).

Nucleus es el **único método que aparece en negrita o cerca del óptimo en todas las columnas simultáneamente**. La conclusión empírica es robusta.

---

## 12. Análisis del comportamiento por régimen

El paper aporta una taxonomía implícita de qué hace cada método en qué régimen:

| Régimen | Beam | Pure | Top-$k$ | Top-$p$ |
|---------|------|------|---------|---------|
| Distribución picuda (1–2 tokens dominan) | greedy-like, correcto | correcto | sobre-incluye basura si $k > 2$ | nucleus pequeño, correcto |
| Distribución plana (decenas de opciones) | colapsa a la moda → repetición | gibberish por tail | sub-trunca si $k$ pequeño | nucleus grande, correcto |
| Inicio de generación (alto contexto) | bien hasta el primer loop | OK | OK con $k$ moderado | OK |
| Tras 50 tokens (contexto largo, posibles loops) | bucle estable | drift off-topic | depende de $k$ | mantiene diversidad |

Una de las observaciones más relevantes pedagógicamente: **en directed generation (traducción, summarization estricta), beam search sigue ganando**. El input restringe tanto el output que la cola del modelo no es un problema y la maximización converge a una continuación válida. La conclusión del paper aplica **estrictamente a open-ended generation** — story continuation, dialog abierto, completion creativa. Esto es importante: nucleus no es "siempre mejor", es "mejor cuando el grado de libertad es alto".

---

## 13. Limitaciones reconocibles

El paper es notablemente sobrio en su sección de conclusión, pero las limitaciones — explícitas o implícitas — son:

1. **$p$ todavía es un hyperparameter**. El paper recomienda $0.95$, pero distintos dominios y modelos requieren tuning. Lo que nucleus elimina es la **rigidez** de $k$, no la necesidad de un threshold.

2. **No resuelve hallucination**. Si el modelo asigna probabilidad alta a una afirmación falsa, el nucleus la mantendrá dentro del sub-conjunto a muestrear. Nucleus mejora la *forma* del texto (coherencia, diversidad, no-repetición), no la *veracidad* del contenido.

3. **Métricas automáticas son imperfectas**. Self-BLEU, Distinct-$n$, repetition rate son proxies. La evidencia más fuerte del paper viene del HUSE, que sí incluye juicio humano. Pero el paper reconoce que ninguna métrica por sí sola es definitiva.

4. **No considera structure global**. Nucleus opera token a token; un texto puede ser localmente coherente pero globalmente incoherente (un argumento que se contradice a sí mismo, una historia que olvida personajes). Esto requiere métodos que vayan más allá del decoding puntual — planning, lookahead, RLHF.

5. **Sampling stochasticity vs reproducibility**. Por construcción, nucleus es no-determinista. Para aplicaciones que requieren determinismo (tests, benchmarks reproducibles), debe combinarse con seeds fijas — pero el método natural sigue siendo greedy/beam, lo cual reintroduce el problema.

---

## 14. Impacto y adopción posterior

El impacto del nucleus paper es difícil de exagerar. Solo a modo de evidencia:

- **HuggingFace `transformers`** — el método `model.generate()` expone `top_p` como parámetro nativo. La documentación oficial cita el paper.
- **GPT-2, GPT-3, GPT-4 API** — todas exponen `top_p` con default 1.0 + `temperature` 1.0, ajustables por el usuario.
- **Claude API** — expone `top_p` y `top_k` simultáneamente, con la recomendación general de modificar solo uno.
- **Llama, Mistral, Gemma** — todos los samplers open-source incluyen nucleus.
- **Whisper** (OpenAI, 2022) — usa top-$p$ para la decodificación de audio-to-text.

El paper también desencadenó una línea de investigación sobre **decoding strategies más sofisticadas**:

- **Mirostat** (Basu et al., 2021) — extensión adaptive del nucleus que mantiene una surprise objetivo constante a lo largo de la generación, ajustando dinámicamente el umbral. Útil para textos largos.

- **Typical Decoding** (Meister et al., 2022) — muestrea de los tokens cuya information content esté cerca de la entropía esperada, en lugar de cortar por masa de probabilidad. Argumenta motivación information-theoretic más fuerte.

- **Locally Typical Sampling** — variante de Meister que combina ideas de nucleus y typical.

- **Min-$p$ Sampling** (Nguyen, 2024) — corta tokens cuya probabilidad sea menor que un fracción $\alpha$ del top-1. Más simple que nucleus, populariza en el ecosistema de llama.cpp.

- **Contrastive Decoding** (Li et al., 2023) — usa la diferencia entre un modelo grande y uno chico para guiar el sampling.

- **DoLa** (Chuang et al., 2023) — decoding por contraste de capas en el mismo modelo, mejora factualidad.

- **Speculative Decoding** (Leviathan et al., 2023) — método de aceleración que usa un "draft model" pequeño para proponer múltiples tokens y un modelo grande para verificarlos. Compatible con nucleus, ya que el sampling del modelo grande puede ser top-$p$.

- **Eta Sampling** (Hewitt et al., 2022) — combina ideas de nucleus con un umbral de entropía relativa para decidir cuándo truncar más agresivamente.

Pero el paradigma básico — truncar por umbral de masa y muestrear del subconjunto — sigue siendo la formulación dominante. Casi todo lo que se hace en 2025 es **una refinación de nucleus**, no un reemplazo.

### 14.1 Adopción cuantitativa

A modo ilustrativo del impacto medible: el paper acumuló más de 5.000 citas en Google Scholar para 2024, ubicándose entre los 20 papers más citados de ICLR 2020. Es prácticamente imposible encontrar un sistema de generación de texto basado en LLMs publicado después de 2020 que no use top-$p$ o alguna de sus variantes directas. En la documentación del API de OpenAI, el parámetro `top_p` aparece como uno de los cinco controles principales junto con `temperature`, `max_tokens`, `frequency_penalty` y `presence_penalty`. Anthropic lista `top_p` como parámetro estándar en su Messages API. La biblioteca `vllm` (uno de los servers de inferencia LLM más usados en producción) lo expone como parte del `SamplingParams` por defecto.

### 14.2 Relación con RLHF y alineamiento

Un punto que el paper original no anticipa pero que se vuelve relevante con la llegada de ChatGPT (noviembre 2022): los modelos alineados por **RLHF** (Reinforcement Learning from Human Feedback) tienen distribuciones que son **menos planas** que sus contrapartes base. El proceso de alignment colapsa la diversidad en las direcciones que los anotadores prefieren. Esto significa que para un modelo RLHF-tuned, el nucleus $p = 0.95$ tiene un cardinal típico menor que para el modelo base — el método sigue funcionando, pero la "región de confianza" del modelo está objetivamente más estrecha. En la práctica, los modelos chat usan valores de $p$ ligeramente más altos ($p = 0.95$–$1.0$) para preservar diversidad.

A nivel cultural y de producto, nucleus marcó el desplazamiento de la práctica industrial desde "tunear $k$ y rezar" hacia "tunear $p$ con default razonable". El parámetro `top_p` es hoy parte del vocabulario de cualquier API call de LLM.

---

## 15. Conexión con la clase 22 del curso IA UC

El paper aparece directamente en tres slides del curso 22:

- **Slide 45 — *Decoding: Issues***: reproduce el ejemplo "unicorns / UNAM repetition" del paper. Es el motor visual para introducir el problema.

- **Slide 46 — *Top-p Sampling***: presenta la definición del nucleus, la condición $\sum P \geq p$, y la visualización del subconjunto truncado del vocabulario. Es esencialmente la Figura 5 del paper trasladada al deck.

- **Slide 47 — *Temperature Scaling***: complementa con la fórmula softmax-con-$t$, mostrando que $p$ y $t$ son ortogonales y combinables.

La clase 22 trata generación de texto como herramienta integral dentro de **abstractive summarization**. La conexión es directa: cualquier sistema de resumen abstractivo basado en seq2seq o LLM moderno (T5, BART, Pegasus, GPT-4) genera el resumen token a token usando algún decoder. La elección del decoder afecta:

- **Si se busca un resumen único óptimo** → beam search (porque summarization es típicamente *directed*, no open-ended).
- **Si se busca diversidad de resúmenes candidatos** → nucleus + temperature.
- **Si se busca diversity para re-ranking posterior** → nucleus con $p$ alto + múltiples samples.

El paper también engancha con los temas previos del curso:

- **Clase 14 (Transformers)**: el problema de degeneration es específico de arquitecturas autoregresivas left-to-right. No aparece (o aparece distinto) en encoder-decoder con cross-attention fuerte.
- **Clase 16 (NLP)**: la métrica Self-BLEU se relaciona con BLEU clásico de machine translation, pero invertida en signo (lower es mejor).
- **Clase 18 (Word Embeddings)**: la "unreliable tail" tiene una explicación parcial en el espacio de embeddings — las direcciones de baja densidad del espacio de embeddings de la salida producen logits poco calibrados.
- **Clase 20 (ELMo/BERT/GPT/ChatGPT)**: ChatGPT y la familia GPT exponen `top_p` directamente; saber qué hace este parámetro es necesario para entender el comportamiento del producto.

---

## 16. Pseudocódigo de referencia

Para fijar el método con una implementación canónica:

```python
import torch
import torch.nn.functional as F

def nucleus_sampling(logits, p=0.95, temperature=1.0):
    """
    logits: tensor (vocab_size,) — logits del paso actual.
    p: umbral de masa acumulada (default 0.95).
    temperature: escala los logits antes del softmax (default 1.0).
    Devuelve: índice del token muestreado.
    """
    # 1. Temperature scaling
    logits = logits / temperature

    # 2. Softmax
    probs = F.softmax(logits, dim=-1)

    # 3. Ordenar descendentemente
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # 4. Probabilidad acumulada
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 5. Marcar tokens a remover (los que exceden p — manteniendo el primero
    #    que cruza el umbral)
    sorted_indices_to_remove = cumulative_probs > p
    # Shift: el primer token que cruza p debe quedarse
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = False

    # 6. Aplicar mask
    sorted_probs[sorted_indices_to_remove] = 0.0

    # 7. Renormalizar
    sorted_probs = sorted_probs / sorted_probs.sum()

    # 8. Sample del nucleus
    next_token_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)
    next_token = sorted_indices[next_token_sorted_idx]

    return next_token
```

La implementación de HuggingFace en `transformers.generation.LogitsProcessor.TopPLogitsWarper` sigue esencialmente este patrón.

---

## 17. Conclusión

El nucleus paper es uno de esos trabajos que cambia el default de toda una comunidad. Antes de abril 2019 el debate decoding era "beam vs top-$k$ vs muestrear directo". Después del paper, el default se fija en top-$p$, y todas las alternativas posteriores se proponen como mejoras sobre nucleus, no como rivales.

Las ideas centrales que conviene retener:

1. **Decoding es un problema separado del modelado**. Un buen language model puede generar texto pésimo con el decoder equivocado.

2. **El likelihood paradox es real**. Texto humano no maximiza probabilidad bajo ningún modelo, por razones pragmáticas (Grice).

3. **La cola de la distribución es ruido residual**. Truncarla mejora muestreo sin perder expresividad.

4. **Truncar por masa de probabilidad** ($p$) es estrictamente más principlado que truncar por ranking ($k$), porque se adapta a la entropía local.

5. **Las métricas automáticas son útiles pero no suficientes**. HUSE — combinación humano + statistical — es la que diferencia nucleus del resto.

6. **Open-ended ≠ directed generation**. Beam search no está muerto — sigue siendo apropiado en traducción y summarization estricta.

Para alguien que esté construyendo sistemas de generación en 2025 — chatbots, summarizers, generadores de código, agentes — el nucleus es el punto de partida. Combinado con temperature y eventualmente con sampling adaptive (Mirostat, typical), constituye el toolkit estándar de inferencia para LLMs.
