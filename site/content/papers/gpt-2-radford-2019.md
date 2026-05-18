---
title: "GPT-2 (Language Models are Unsupervised Multitask Learners)"
weight: 295
math: true
---

{{< paper-card
    title="Language Models are Unsupervised Multitask Learners"
    authors="Radford, Wu, Child, Luan, Amodei, Sutskever"
    year="2019"
    venue="OpenAI Technical Report"
    pdf="/papers/gpt-2-radford-2019.pdf"
    arxiv="" >}}
GPT-2 valida que escalar un decoder-only Transformer entrenado con next-token prediction sobre texto diverso (WebText) produce **multitask learning emergente sin fine-tuning**. El modelo de 1.5B parámetros alcanza zero-shot SOTA en 7 de 8 datasets de language modeling y muestra señal positiva en CoQA, traducción WMT-14, resumen CNN/DailyMail, Winograd, CBT y Natural Questions sin un solo ejemplo etiquetado de esas tareas. Es la primera evidencia visual clara de **scaling laws** (Figura 1) y el paper que establece el prompting como técnica de evaluación. Su staged release inaugura el debate moderno openness vs safety.
{{< /paper-card >}}

---

## Contexto

GPT-2 se publica en **febrero de 2019**, en un momento muy específico del NLP. Para entender por qué fue disruptivo hay que reconstruir los 18 meses previos.

- **Junio 2017 — Vaswani et al., *Attention is All You Need*.** Aparece el Transformer; se valida en traducción, pero queda claro que self-attention es genérico, modular y escalable.
- **Junio 2018 — GPT-1** (Radford et al.). Primer Generative Pre-trained Transformer: decoder-only entrenado con language modeling sobre BookCorpus (~800M tokens), luego fine-tuned tarea por tarea con cabeza supervisada. Supera SOTA en 9 de 12 benchmarks.
- **Octubre 2018 — BERT** (Devlin et al.). Encoder bidireccional con Masked LM + NSP, 340M parámetros, pulveriza GLUE y SQuAD. "BERT" se vuelve sinónimo de NLP moderno.
- **Finales de 2018 — la pregunta abierta.** Si BERT bidireccional domina comprensión, ¿qué espacio le queda a un modelo autoregresivo unidireccional como GPT-1?

GPT-2 entra con una respuesta lateral: en lugar de competir con BERT en benchmarks supervisados, demuestra algo más radical: **un language model lo suficientemente grande, entrenado en un corpus lo suficientemente diverso, empieza a resolver tareas downstream sin fine-tuning alguno**. La métrica clave no es "ganar GLUE", sino "ganar zero-shot".

A esto se suma el **staged release**: OpenAI decide no liberar inmediatamente el modelo de 1.5B alegando "preocupaciones por mal uso" (desinformación, spam). Las versiones se liberan progresivamente entre febrero y noviembre 2019 (117M, 345M, 762M, 1.5B). Es el primer episodio público del debate "openness vs safety", patrón que se repetirá con GPT-3, ChatGPT y GPT-4.

---

## Ideas principales

### 1. Tesis: "Unsupervised Multitask Learners"

Un language model autoregresivo modela la distribución conjunta sobre secuencias vía la cadena clásica:

$$
p(x) = \prod_{i=1}^{n} p(s_i \mid s_1, \dots, s_{i-1})
$$

El salto conceptual del paper: en lugar de estimar $p(\text{output} \mid \text{input})$ con una cabeza supervisada por tarea, lo correcto es modelar

$$
p(\text{output} \mid \text{input}, \text{task}),
$$

donde **la tarea se codifica como tokens en el contexto** (un prompt), no como un vector aprendido ni una cabeza especializada. Ejemplos del paper:

- Traducción: `translate to french, english text, french text`
- Reading comprehension: `answer the question, document, question, answer`
- Resumen: artículo seguido del token `TL;DR:`

La pieza de razonamiento más profunda aparece en la sección 2:

> Since the supervised objective is the same as the unsupervised objective but only evaluated on a subset of the sequence, the **global minimum of the unsupervised objective is also the global minimum of the supervised objective**.

Si la web contiene suficientes ejemplos con forma "english sentence = french sentence", aprender a predecir el siguiente token sobre la web obliga al modelo a aprender, entre muchas cosas, a traducir. La tarea supervisada está *contenida* en el objetivo no supervisado. El paper reconoce explícitamente la deuda con decaNLP (McCann et al. 2018), pero con una diferencia clave: decaNLP entrena 10 tareas supervisadas; GPT-2 no entrena supervisado en absoluto.

El término **"unsupervised multitask learning"** se establece aquí y se vuelve la lente con la que se interpretan todos los LLMs posteriores.

### 2. WebText: data engineering como contribución

Los autores rechazan deliberadamente corpus single-domain (Wikipedia, BookCorpus, news). Common Crawl tiene la escala pero — citando experimentos preliminares — su calidad es "mostly unintelligible".

La solución es elegante: usar **señal humana de curaduría agregada** como filtro sin filtrar manualmente. **Reddit** funciona como proxy:

1. Scrapear todos los outbound links de Reddit con **karma ≥ 3** (al menos algunos usuarios encontraron el link interesante).
2. ~45 millones de links iniciales.
3. Extraer texto con Dragnet + Newspaper.
4. Excluir links posteriores a diciembre 2017 (contaminación temporal).
5. **Remover todos los documentos de Wikipedia** — decisión epistemológica fuerte, porque Wikipedia es la fuente de evaluación de WikiText-2/103 y entrenar en ella inflaría artificialmente los resultados zero-shot.

**Resultado: ~8 millones de documentos, ~40GB de texto, ~10B tokens.**

Tres observaciones que vale enfatizar:

1. **No-Wikipedia** permite que la evaluación en WikiText-103 sea honesta. Es una práctica que muchos papers posteriores no respetaron.
2. **Reddit como filtro humano** anticipa la idea de "instrucciones implícitas en datos curados" que llegará con instruction tuning y RLHF. WebText es, en cierto modo, el primer "RLHF light".
3. **40GB son pequeños** comparados con lo que vendría (GPT-3 entrenará sobre ~570GB), pero en 2019 son un orden de magnitud sobre BookCorpus.

**Análisis de overlap (Sección 4).** Los autores construyen Bloom filters de 8-gramos y miden contaminación con los test sets de los benchmarks usados. Promedio: 3.2%, comparable o menor al overlap entre los propios train/test de los benchmarks. Esto refuerza que los resultados zero-shot no son fruto de memorización masiva. Es una práctica metodológica que sería bueno ver más seguido.

### 3. Cambios arquitectónicos vs GPT-1

GPT-2 es esencialmente la misma arquitectura que GPT-1 (Transformer decoder-only, causal-masked self-attention) con cinco modificaciones:

**(a) Pre-LayerNorm.** GPT-1 (siguiendo el Transformer original) aplicaba LayerNorm **después** del residual: $x' = \mathrm{LN}(x + \mathrm{Sublayer}(x))$ (post-LN). GPT-2 lo aplica **antes**: $x' = x + \mathrm{Sublayer}(\mathrm{LN}(x))$ (pre-LN), análogo a las pre-activation ResNets. Además añade un LayerNorm final antes de la proyección al vocabulario. Empíricamente, pre-LN es **mucho más estable a entrenar** en redes profundas: post-LN sufre de gradientes explosivos en las primeras capas cuando $N$ crece. Para entrenar 48 capas (GPT-2 XL) deja de ser un detalle: es la diferencia entre converger y no converger. La decisión se vuelve canónica — GPT-3, LLaMA, Mistral, todos usan pre-LN.

**(b) Inicialización residual escalada por $1/\sqrt{N}$.** En una red residual de profundidad $N$, la varianza del residual stream crece linealmente con la profundidad si no se controla. El paper escala las proyecciones de salida de cada sub-bloque por $1/\sqrt{N}$ en inicialización para mantener la varianza acotada.

**(c) Byte-level BPE con vocabulario de 50,257.** BPE sobre Unicode requeriría >130K tokens base. BPE byte-level tiene base 256, pero produce merges subóptimos (`dog`, `dog!`, `dog?` aparecen como variantes redundantes). Solución: **prevenir merges que crucen categorías de caracteres** (excepto el espacio). Resultado: 50,257 tokens. Esto convierte a GPT-2 en un modelo que puede asignar probabilidad a **cualquier string Unicode sin `<UNK>`**, útil para evaluación cross-domain, código fuente, emojis y caracteres exóticos. Ver fundamento: [BPE](/fundamentos/bpe).

**(d) Context window extendido a 1024 tokens** (vs 512 en GPT-1). Importante para tareas como CoQA y resumen que requieren documentos completos.

**(e) Batch size de 512** (vs 64 en GPT-1). Combinado con secuencias de 1024 tokens, son ~524K tokens por step.

| Aspecto | GPT-1 (2018) | GPT-2 (2019) |
|---|---|---|
| LayerNorm | Post-LN | Pre-LN + LN final |
| Init residual | Estándar | Escalado por $1/\sqrt{N}$ |
| Vocab | ~40K BPE | 50,257 byte-level BPE |
| Context | 512 | 1024 |
| Batch size | 64 | 512 |
| Tamaño máximo | 117M | 1542M (~13x) |
| Dataset | BookCorpus 800M tokens | WebText 40GB / ~10B tokens |
| Adaptación downstream | Fine-tuning supervisado | Zero-shot prompting |

### 4. Cuatro escalas: scaling como mensaje

El paper entrena **cuatro modelos** con tamaños log-uniformemente espaciados. La decisión no es casual: permite trazar curvas de performance contra tamaño y observar **scaling**, anticipando los scaling laws de Kaplan et al. 2020 por casi un año.

| Parámetros | Layers ($N$) | $d_{\text{model}}$ | Nombre informal | Equivalencia |
|---:|---:|---:|---|---|
| 117M | 12 | 768 | GPT-2 Small | $\approx$ GPT-1 |
| 345M | 24 | 1024 | GPT-2 Medium | $\approx$ BERT-Large |
| 762M | 36 | 1280 | GPT-2 Large | — |
| 1542M | 48 | 1600 | GPT-2 XL (1.5B) | El "GPT-2" propiamente tal |

Detalle crucial: los autores remarcan que **todos los modelos siguen underfitting WebText** al momento de publicación. La held-out perplexity sigue mejorando con más entrenamiento. Esto implica que incluso a 1.5B parámetros los modelos no agotaron la señal del corpus. Es exactamente la grieta por la que entrará GPT-3.

La **Figura 1 del paper** muestra performance zero-shot en Reading Comprehension, Translation, Summarization y QA como función del número de parámetros. Las cuatro curvas son monótonas crecientes, log-lineales. Visto en retrospectiva, es una de las imágenes más influyentes del NLP moderno: el anuncio velado de scaling laws.

### 5. Zero-shot evaluation y prompting

GPT-2 popularizó el prompting como técnica de evaluación. Sin pesos ajustados, sin cabezas especializadas: el modelo simplemente recibe una descripción de la tarea como tokens en el contexto.

**Técnicas de decoding** que el paper usa o motiva:

- **Greedy decoding** para tareas extractivas (CoQA, QA); produce loops repetitivos en generación libre.
- **Top-k sampling**: truncar a los $k$ tokens más probables y renormalizar. El paper usa $k=2$ para resumen.
  $$
  p'(s_t) = \begin{cases} \dfrac{p(s_t)}{\sum_{s \in \text{Top-}k} p(s)} & \text{si } s_t \in \text{Top-}k \\ 0 & \text{en otro caso} \end{cases}
  $$
- **Nucleus / top-p sampling** (Holtzman et al. 2019) se formaliza meses después motivado por GPT-2: $V^{(p)} = \min\{ V' \subseteq V : \sum_{s \in V'} p(s) \geq p \}$.
- **Temperature**: $p_T(s_t) = \exp(z_t/T) / \sum_s \exp(z_s/T)$, con $T<1$ aguzando y $T>1$ aplanando.

La discusión de sampling en GPT-2 marca el inicio de un sub-campo: cómo extraer texto natural de un LM. Las decisiones de decoding empezaron a ser tan importantes como la arquitectura.

### 6. La polémica del staged release

OpenAI decidió no liberar el modelo de 1.5B en febrero 2019, citando preocupaciones de mal uso. Liberó 117M en febrero, 345M en mayo, 762M en agosto y 1.5B en noviembre.

Posiciones:

- **Críticos** (Hugging Face, varios académicos): no liberar es paternalista, frena la investigación, y la barrera de réplica es cómputo, no secreto. Brown y Vanwinkle replicaron el modelo (OpenGPT-2) demostrando que la barrera era de compute, no de información.
- **Defensores**: las amenazas de fake news automatizadas y spam personalizado son reales; vale ser cuidadoso.

OpenAI publicó un *follow-up report* en noviembre 2019 (*Release Strategies and the Social Impacts of Language Models*) documentando que no observaron uso malicioso significativo y procedió a liberar el 1.5B. Pero el precedente quedó: **modelos cerrados detrás de APIs** se vuelven la norma para actores comerciales (GPT-3, GPT-4, Claude, Gemini), mientras la línea open-weights (LLaMA, Mistral) queda como alternativa.

---

## Resultados experimentales

### Zero-shot language modeling (Tabla 3)

**SOTA zero-shot en 7 de 8 datasets**. Métricas: perplexity (PPL), bits per character (BPC) o accuracy según convención.

| Dataset | Métrica | SOTA previo | GPT-2 117M | 345M | 762M | 1542M |
|---|---|---:|---:|---:|---:|---:|
| LAMBADA | PPL | 99.8 | 35.13 | 15.60 | 10.87 | **8.63** |
| LAMBADA | ACC | 59.23 | 45.99 | 55.48 | 60.12 | **63.24** |
| CBT-CN | ACC | 85.7 | 87.65 | 92.35 | 93.45 | **93.30** |
| CBT-NE | ACC | 82.3 | 83.4 | 87.1 | 88.0 | **89.05** |
| WikiText-2 | PPL | 39.14 | 29.41 | 22.76 | 19.93 | **18.34** |
| PTB | PPL | 46.54 | 65.85 | 47.33 | 40.31 | **35.76** |
| enwik8 | BPC | 0.99 | 1.16 | 1.06 | 0.97 | **0.93** |
| text8 | BPC | 1.08 | 1.17 | 1.06 | 1.02 | **0.98** |
| WikiText-103 | PPL | 18.3 | 37.50 | 26.37 | 22.05 | **17.48** |
| 1BW | PPL | **21.8** | 75.20 | 55.72 | 44.575 | 42.16 |

El único dataset donde GPT-2 no logra SOTA es 1BW, por dos razones: es el dataset de entrenamiento más grande y **su preprocesamiento destruye toda estructura larga** — las oraciones están shuffled, eliminando dependencias cross-sentence donde un LM grande brilla.

LAMBADA — diseñado para medir dependencias largas — pasa de PPL 99.8 a 8.63, un orden de magnitud. Las mejoras más espectaculares se dan en datasets pequeños (PTB, WikiText-2) donde los LMs especializados tenían 1-2M tokens de training: el preentrenamiento masivo "rellena" la falta de datos en-distribución.

### Capacidades emergentes downstream

- **CoQA (Reading Comprehension).** Concatenar `documento + historial + A:` y generar con greedy. **55 F1**, igualando o superando 3 de 4 baselines supervisados. El SOTA supervisado (BERT) llega a ~89 F1 con 127K pares QA etiquetados.
- **Resumen CNN/DailyMail.** Añadir `TL;DR:` después del artículo y generar 100 tokens con top-$k=2$. GPT-2 con prompt: ROUGE-AVG 21.40 (vs 15.03 sin prompt, vs 32.75 del SOTA Bottom-Up Sum). Como sistema de resumen es flojo, pero la diferencia con/sin prompt (6.4 puntos) es evidencia clara de que **el prompting invoca task-specific behavior**.
- **Traducción WMT-14.** Condicionar con varios pares `english = french` y dar la frase inglesa. En$\to$Fr: 5 BLEU (peor que substitución palabra-por-palabra). Fr$\to$En: **11.5 BLEU**, superando varios baselines unsupervised. Lo sorprendente: WebText contiene solo **10MB de texto francés** — 500x menos que los corpus monolingües franceses usados en NMT no supervisada. Es decir, GPT-2 traduce un poco habiendo visto casi nada de francés.
- **Natural Questions (QA).** Prompting con pares ejemplo, luego pregunta sin contexto. GPT-2 XL: 4.1% exact match. Las respuestas con probabilidad >50% según el modelo aciertan ~63%: el modelo está **bien calibrado en su propia incertidumbre**. 5x mejor que la mejor publicación previa de QA con LMs.
- **Winograd Schema Challenge.** GPT-2 XL: **70.70%**, +7 sobre el SOTA previo (Trinh & Le 2018).
- **Children's Book Test.** CBT-CN 93.30%, CBT-NE 89.05%, cerrando la mayor parte del gap con humanos (~96% / ~92%).
- **El "unicorn passage".** La Tabla 13 del apéndice contiene el ejemplo viral: un prompt sobre el descubrimiento ficticio de unicornios en los Andes produce varios párrafos coherentes con citas inventadas de investigadores ficticios, conexiones con mitología incaica, etc. Para muchos profesionales que habían trabajado con seq2seq y LSTMs, fue **la primera vez que un LM "parecía haber leído el libro completo"**. Marca el momento en que el gran público empezó a notar que algo había cambiado en NLP.

---

## Limitaciones reconocibles

El paper es notablemente honesto. Las limitaciones que lista establecen la agenda de los años siguientes:

1. **Repetición en generación libre.** Greedy entra en loops; sampling resuelve parcialmente pero introduce drift temático.
2. **Coherencia larga sigue siendo frágil.** Los unicornios funcionan, pero pasajes de varios miles de tokens divergen, se contradicen y alucinan hechos.
3. **Heurísticas baratas en QA y resumen.** El modelo a veces resuelve tareas con shortcuts (responder con un nombre cualquiera del documento, copiar las primeras oraciones) en lugar de comprensión real.
4. **No alcanza fine-tuning supervisado.** En muchas tareas hay 30-50 puntos de gap con el SOTA supervisado.
5. **Generación de lyrics, código o dominios técnicos se degrada a gibberish.** Sub-dominios escasos en WebText producen representaciones débiles.
6. **Aún underfitting WebText.** La línea más importante del paper en retrospectiva: el modelo no agotó el corpus. Es una invitación abierta a escalar más.

---

## Por qué importa hoy

**Validación temprana de scaling laws.** Kaplan et al. 2020 formalizaría que la perplexity de un LM escala como una power-law en parámetros, datos y compute:

$$
L(N) \approx \left( \frac{N_c}{N} \right)^{\alpha_N}
$$

con $\alpha_N \approx 0.076$ para parámetros. Pero la **evidencia visual** ya está en la Figura 1 de GPT-2: cuatro tamaños, cuatro tareas, todas las curvas log-lineales. GPT-2 fue el primer paper en mostrar de manera tan limpia y deliberada las curvas de scaling.

**"Unsupervised multitask learning" como lente.** Después de GPT-2, hablar de un LLM como "modelo de language modeling" es casi un eufemismo: la comunidad acepta que un LM suficientemente grande es, *de facto*, un sistema multitarea. La lente se aplica a T5, GPT-3, PaLM, LLaMA.

**Debate openness vs safety.** El staged release establece el primer precedente del actual régimen de "modelos cerrados con APIs". Caso de estudio que vale citar cuando se discutan los releases de Claude, GPT-4 o LLaMA.

**Pavimenta GPT-3.** GPT-3 (Brown et al. 2020) es esencialmente "GPT-2 más grande": misma arquitectura básica, mismo objetivo, más datos, 175B parámetros (100x). El paper de GPT-3 introduce **in-context learning few-shot**, pero es extensión directa del prompting zero-shot que GPT-2 estableció. Sin GPT-2 no hay GPT-3.

**El arco GPT-1 → GPT-2 → GPT-3:**

| | GPT-1 (2018) | GPT-2 (2019) | GPT-3 (2020) |
|---|---|---|---|
| Tamaño máximo | 117M | 1.5B | 175B |
| Datos | BookCorpus 800M tokens | WebText 10B tokens | Common Crawl 400B tokens |
| Adaptación | Fine-tuning supervisado | **Zero-shot prompting** | **Few-shot in-context learning** |
| Tesis | Pre-training + FT es transferible | LMs grandes son multitask learners | LMs gigantes son meta-learners few-shot |
| Arquitectura | Decoder post-LN | Decoder pre-LN | Idem + sparse attention |
| Polémica | Ninguna | Staged release | API-only, RLHF |

El mensaje de GPT-2 es **radical en su simplicidad**: misma arquitectura, más grande, más datos diversos, sin cambios en el objetivo de entrenamiento — y la promesa de "el LM aprende todo lo que esté en la web". Es la primera vez que se enuncia con claridad la receta que define los LLMs modernos: **scale + diversity + autoregressive language modeling = emergent capabilities**.

GPT-2 es la **bisagra**: antes, pre-training + fine-tuning supervisado con encoder dominante (BERT) y cabezas especializadas; después, prompting, scaling, emergencia, modelos generales que se especializan en runtime y debate de safety/openness. Casi todo lo que define a los LLMs actuales — incluyendo el régimen comercial, el lenguaje del campo y las preocupaciones éticas — se reconoce ya, en germen, en este technical report de 24 páginas.

---

## Notas y enlaces

- Paper original: [OpenAI technical report](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), febrero 2019.
- Código y checkpoints: [github.com/openai/gpt-2](https://github.com/openai/gpt-2). Release escalonado: 117M (feb 2019), 345M (mayo), 762M (agosto), 1.5B (noviembre).
- Follow-up sobre release: [Release Strategies and the Social Impacts of Language Models](https://arxiv.org/abs/1908.09203), noviembre 2019.
- Implementaciones modernas: [HuggingFace Transformers](https://huggingface.co/docs/transformers) (`GPT2LMHeadModel`).
- **Figura 1** del paper: scaling zero-shot en 4 tareas — la imagen canónica del campo.
- **Tabla 3**: zero-shot SOTA en 7/8 datasets de LM.
- **Tabla 13** (apéndice): el famoso "unicorn passage".

Papers relacionados: [BERT (Devlin 2018)](/papers/bert-devlin-2018) - [Attention is All You Need (Vaswani 2017)](/papers/attention-is-all-you-need-vaswani-2017) - [Foundation Models (Bommasani 2021)](/papers/foundation-models-bommasani-2021).

Ver fundamentos: [Pre-training y BERT](/fundamentos/pretraining-bert) - [Transformer](/fundamentos/transformer) - [Self-attention](/fundamentos/self-attention) - [BPE](/fundamentos/bpe) - [Foundation Models](/fundamentos/foundation-models) - [Clase 14](/clases/clase-14) - [Clase 20](/clases/clase-20).
