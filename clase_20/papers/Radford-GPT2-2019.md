# Language Models are Unsupervised Multitask Learners (GPT-2, Radford et al., 2019)

Análisis exhaustivo del paper de GPT-2, escrito como material interno de estudio para el Diplomado IA UC, clase 20 (LLMs: del scaling a la alineación). El foco está en la tesis conceptual, las decisiones de ingeniería, las evidencias empíricas, las polémicas que despertó y su lugar en la genealogía que va de GPT-1 a GPT-3.

- **Título:** Language Models are Unsupervised Multitask Learners
- **Autores:** Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever
- **Afiliación:** OpenAI, San Francisco
- **Año:** 2019 (technical report, no enviado a conferencia revisada por pares)
- **Modelo insignia:** 1.5B parámetros (GPT-2 XL, también referido como 1542M)
- **Código:** https://github.com/openai/gpt-2 (release escalonado: 117M en feb 2019, 345M en mayo, 762M en agosto, 1.5B en noviembre 2019)

---

## 1. Contexto histórico: el momento en que apareció GPT-2

GPT-2 se publica en **febrero de 2019**, en un momento muy específico de la historia del NLP. Para entender por qué este paper fue tan disruptivo (y por qué su liberación fue tan polémica) hay que reconstruir el panorama de los 18 meses anteriores.

**Junio 2017 — Vaswani et al., *Attention is All You Need*.** Aparece el Transformer. La arquitectura se valida primero en traducción (WMT 2014) pero queda claro que el bloque de self-attention es genérico, modular y escalable.

**Junio 2018 — Radford et al., GPT-1 (*Improving Language Understanding by Generative Pre-Training*).** OpenAI publica el primer Generative Pre-trained Transformer. La receta es: (i) preentrenar un Transformer decoder-only con language modeling sobre BookCorpus (~7000 libros, 800M tokens) y (ii) hacer fine-tuning supervisado, tarea por tarea, agregando una cabeza de clasificación. Demuestra que un solo modelo, con ajustes mínimos, supera el SOTA en 9 de 12 benchmarks del momento (NLI, QA, classification, similarity).

**Octubre 2018 — Devlin et al., BERT (*Pre-training of Deep Bidirectional Transformers*).** Google publica BERT. Cambia dos cosas: (i) usa un Transformer encoder bidireccional con dos objetivos no autorregresivos — Masked LM y Next Sentence Prediction — y (ii) escala a 340M parámetros (BERT-Large). El impacto es enorme: BERT pulveriza GLUE, SQuAD y prácticamente cualquier benchmark de comprensión. Durante el último trimestre de 2018, "BERT" se convierte en sinónimo de NLP moderno.

**Noviembre–diciembre 2018 — la pregunta abierta.** Quedaba colgada una tensión teórica: si BERT (bidireccional, masked) era tan dominante en comprensión, ¿qué espacio le quedaba a un modelo autorregresivo unidireccional como GPT-1? La sospecha de muchos era que la línea decoder-only había quedado obsoleta para tareas downstream supervisadas.

GPT-2 entra a ese paisaje con una respuesta lateral: en lugar de competir con BERT en benchmarks supervisados con fine-tuning, demuestra algo aparentemente más radical: **un language model lo suficientemente grande, entrenado en corpus lo suficientemente diverso, empieza a resolver tareas downstream sin fine-tuning alguno**. La métrica clave del paper no es "ganar GLUE", sino "ganar zero-shot en 7 de 8 datasets de language modeling y mostrar señal positiva (no random) en CoQA, CNN/DailyMail, WMT-14, LAMBADA, Children's Book Test, Winograd Schema y Natural Questions sin haber visto un solo ejemplo etiquetado de esas tareas".

**El staged release.** Lo que convirtió a GPT-2 en un acontecimiento mediático no fue solo el paper sino la decisión política de OpenAI de **no liberar inmediatamente el modelo de 1.5B parámetros**, argumentando "preocupaciones por mal uso" (generación de desinformación, fake news, spam). En su lugar liberaron las versiones pequeñas progresivamente entre febrero y noviembre 2019. Esto desencadenó el primer gran debate público sobre "openness vs safety" en AI, un patrón que se repetiría con GPT-3, ChatGPT y GPT-4. Críticos como Nabla Bench, Hugging Face y la propia comunidad replicaron el modelo (OpenGPT-2 de Brown y Vanwinkle alcanzó capacidades comparables), demostrando que la barrera era principalmente de compute, no de secreto.

En suma: GPT-2 llega como una respuesta deliberadamente provocadora a BERT, y la polémica de su release establece el primer precedente del actual régimen de "modelos cerrados con APIs".

---

## 2. Tesis central: language models como multitask learners no supervisados

El título es la tesis y vale la pena desmenuzarlo.

### 2.1 La factorización clásica

Un language model autorregresivo modela una distribución conjunta sobre secuencias de símbolos $(s_1, s_2, \dots, s_n)$ vía la factorización en cadena:

$$
p(x) = \prod_{i=1}^{n} p(s_i \mid s_1, \dots, s_{i-1})
$$

Esta factorización es general: permite muestreo tractable y permite calcular cualquier condicional de la forma $p(s_{n-k}, \dots, s_n \mid s_1, \dots, s_{n-k-1})$.

### 2.2 El salto conceptual

El paper introduce una factorización más expresiva. Para realizar una tarea, lo usual en NLP es estimar $p(\text{output} \mid \text{input})$, lo que requiere una cabeza supervisada por tarea. Pero — argumentan los autores — un sistema general debería poder ejecutar muchas tareas distintas con el mismo input. Entonces lo correcto es modelar:

$$
p(\text{output} \mid \text{input}, \text{task})
$$

La pregunta operacional es: ¿cómo se codifica la tarea? La respuesta de GPT-2 es **el prompt**. El task descriptor no es un vector aprendido, no es una cabeza especializada, no es un meta-parámetro de MAML. Es **una secuencia de tokens más en el contexto**.

Ejemplos del paper:

- Traducción: `translate to french, english text, french text`
- Reading comprehension: `answer the question, document, question, answer`
- Resumen: artículo seguido del token `TL;DR:`

### 2.3 La intuición teórica

La pieza de razonamiento más profunda del paper aparece en la sección 2 (énfasis añadido):

> Since the supervised objective is the same as the unsupervised objective but only evaluated on a subset of the sequence, the **global minimum of the unsupervised objective is also the global minimum of the supervised objective**. In this slightly toy setting, the concerns with density estimation as a principled training objective discussed in Sutskever et al. (2015) are side stepped. The problem instead becomes whether we are able to, in practice, optimize the unsupervised objective to convergence.

En otras palabras: si la web contiene suficientes ejemplos de la forma "english sentence = french sentence", entonces aprender a predecir el siguiente token sobre la web obliga al modelo a aprender, entre muchas otras cosas, a traducir. La tarea supervisada está *contenida* en el objetivo no supervisado; solo hay que tener suficiente capacidad y suficientes datos para que el modelo no la trate como ruido sino como señal explotable. La Tabla 1 del paper muestra ejemplos reales de pares English-French que aparecen "naturalmente" en WebText (citas, traducciones inline, comentarios bilingües), evidencia de que la señal multitarea efectivamente existe en la distribución natural del texto web.

### 2.4 Conexión con decaNLP (McCann et al. 2018)

El paper reconoce explícitamente la deuda con McCann et al. 2018, *The Natural Language Decathlon* (decaNLP). En decaNLP los autores reformulan 10 tareas distintas (QA, traducción, resumen, NLI, dialogue, etc.) como question-answering, entrenan un solo modelo (MQAN) y muestran que es posible aprender múltiples tareas con un solo modelo. La diferencia clave de GPT-2 es que **decaNLP entrena supervisado con 10 pares (dataset, objective); GPT-2 no entrena supervisado en absoluto**: descubre las tareas en el texto web no etiquetado.

### 2.5 Por qué "multitask learners"

El argumento del título no es metafórico. Los autores sostienen que un LM suficientemente capaz, entrenado en un corpus suficientemente diverso, *de facto* está aprendiendo miles de tareas simultáneamente — cada una codificada como una "demostración natural" en la web (un artículo seguido de su TL;DR, una cita en otro idioma seguida de su traducción, una pregunta en un foro seguida de su respuesta, etc.). El término **"unsupervised multitask learning"** se establece aquí y se vuelve la lente con la que se interpretan todos los LLMs posteriores.

---

## 3. Diferencias arquitectónicas respecto de GPT-1

GPT-2 es esencialmente la misma arquitectura que GPT-1 (Transformer decoder-only, autorregresivo, causal-masked self-attention) con cinco modificaciones específicas. El paper las lista en la sección 2.3 de manera notablemente concisa:

### 3.1 Pre-LayerNorm (LN movido a la entrada de cada sub-bloque)

GPT-1 (siguiendo el Transformer original) aplicaba LayerNorm **después** del residual: `x' = LN(x + Sublayer(x))` (post-LN). GPT-2 lo aplica **antes**: `x' = x + Sublayer(LN(x))` (pre-LN), análogo a las pre-activation ResNets de He et al. 2016. Además añade un LayerNorm final tras el último bloque, antes de la cabeza de proyección al vocabulario.

¿Por qué importa? Empíricamente pre-LN es **mucho más estable a entrenar** en redes profundas. Post-LN sufre de gradientes explosivos en las primeras capas cuando $N$ crece, lo que obliga a usar warmup largos y learning rates muy bajos. Pre-LN permite warmup mucho más corto y tasas más agresivas. Para entrenar 48 capas (GPT-2 XL) esto deja de ser un detalle: es la diferencia entre converger y no converger. Esta decisión se vuelve canónica — GPT-3, LLaMA, Mistral, etc. todos usan pre-LN.

### 3.2 Inicialización residual escalada por $1/\sqrt{N}$

> A modified initialization which accounts for the accumulation on the residual path with model depth is used. We scale the weights of residual layers at initialization by a factor of $1/\sqrt{N}$ where $N$ is the number of residual layers.

La intuición: en una red residual de profundidad $N$, la varianza del residual stream crece linealmente con la profundidad si no se controla. Multiplicar los pesos de los proyecciones de salida de cada sub-bloque por $1/\sqrt{N}$ mantiene la varianza acotada en inicialización. Concretamente, las proyecciones de salida de la atención y de la MLP se inicializan con desviación estándar reducida en $1/\sqrt{N}$.

### 3.3 Vocabulario expandido a 50,257 con BPE byte-level

GPT-1 usaba BPE con vocabulario de ~40,000 sobre tokens estándar. GPT-2 introduce una variante muy importante: **byte-level BPE**. El razonamiento:

- BPE sobre Unicode requeriría un vocabulario base de >130,000 antes de cualquier merge (todos los code points).
- BPE byte-level tiene un vocabulario base de exactamente **256** (los bytes posibles).
- Pero aplicar BPE puro a bytes produce merges subóptimos (`dog`, `dog!`, `dog?`, `dog.` se mergean como variantes redundantes).
- Solución: **prevenir merges que crucen categorías de caracteres**, con la excepción del espacio. Esto preserva eficiencia de compresión y evita fragmentación absurda.

Resultado: vocabulario final de **50,257** tokens (256 bytes base + ~50,000 merges + token de fin de texto). Esto convierte a GPT-2 en un modelo que puede asignar probabilidad a *cualquier* string Unicode sin `<UNK>`, una propiedad muy útil para evaluación cross-domain y para manejar caracteres exóticos, emojis, código fuente, etc.

### 3.4 Context window extendido a 1024 tokens

GPT-1 tenía contexto de 512 tokens. GPT-2 lo duplica a 1024. Es una decisión importante porque tareas como CoQA y resumen requieren mantener documentos completos en contexto. (GPT-3 lo llevaría a 2048; GPT-4 a 8K/32K; los modelos actuales a 128K-1M.)

### 3.5 Batch size de 512

GPT-1 usaba batch size 64. GPT-2 lo eleva a 512 secuencias por batch. Combinado con secuencias de 1024 tokens, son ~524,288 tokens por step, lo que reduce ruido del gradiente y permite explorar la región de "large batch" que después Kaplan 2020 caracterizaría como crítica para scaling.

### 3.6 Cuadro resumen

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

---

## 4. Las cuatro escalas y la arquitectura

El paper entrena **cuatro modelos** con tamaños log-uniformemente espaciados. Esta decisión metodológica es clave: permite trazar curvas de performance contra tamaño y observar **scaling**, anticipando los scaling laws de Kaplan et al. 2020 por casi un año.

### 4.1 Tabla de hiperparámetros (Tabla 2 del paper)

| Parámetros | Layers ($N$) | $d_{\text{model}}$ | Nombre informal | Equivalencia |
|---:|---:|---:|---|---|
| 117M | 12 | 768 | GPT-2 Small | ≈ GPT-1 |
| 345M | 24 | 1024 | GPT-2 Medium | ≈ BERT-Large |
| 762M | 36 | 1280 | GPT-2 Large | — |
| 1542M | 48 | 1600 | GPT-2 XL (1.5B) | El "GPT-2" propiamente tal |

El número de heads no aparece explícitamente en la Tabla 2 pero por convención: 12 / 16 / 20 / 25 heads respectivamente (con $d_{\text{head}} = 64$).

### 4.2 Detalles de entrenamiento

- Learning rate ajustado manualmente para mejor perplexity en un held-out 5% de WebText.
- Schedule cosine con warmup.
- Optimizer Adam.
- **Todos los modelos siguen underfitting WebText** al momento de publicación — los autores remarcan que la held-out perplexity sigue mejorando con más entrenamiento. Este detalle es enorme: implica que incluso a 1.5B parámetros, los modelos no agotaron la señal del corpus. Es decir, había headroom de capacidad y de cómputo. Esta observación es exactamente la grieta por la que entró GPT-3.

### 4.3 La Figura 1 del paper: scaling como mensaje

La Figura 1 del paper, que aparece en la página 2, muestra la performance zero-shot en cuatro tareas (Reading Comprehension, Translation, Summarization, Question Answering) como función del número de parámetros del LM. Las cuatro curvas son **monótonas crecientes**, log-lineales aproximadamente. Es un anuncio velado del scaling laws: agregar capacidad, sin cambiar nada más, mejora la performance en tareas que el modelo nunca vio supervisadas. Visto en retrospectiva, esta figura es una de las imágenes más influyentes del NLP moderno.

---

## 5. WebText: el corpus

### 5.1 Motivación

Los autores rechazan deliberadamente los corpus single-domain (news de Jozefowicz et al. 2016, Wikipedia de Merity et al., ficción de BookCorpus). La hipótesis multitarea exige diversidad: tareas distintas aparecen en dominios distintos.

Common Crawl es el candidato obvio (escala masiva), pero — citando a Trinh & Le 2018 — su calidad es muy heterogénea: gran parte del contenido es "mostly unintelligible". Los experimentos preliminares de OpenAI con Common Crawl confirmaron el problema.

### 5.2 Construcción de WebText

La solución es elegante: usar **señal humana de curaduría** como filtro de calidad sin filtrar manualmente. Reddit funciona como proxy:

1. Scrapear **todos los outbound links de Reddit con karma ≥ 3** (el umbral de 3 indica que al menos algunos usuarios encontraron el link interesante, educativo o divertido).
2. Total: ~45 millones de links.
3. Extraer texto con Dragnet (Peters & Lecocq 2013) y Newspaper.
4. **Excluir links posteriores a diciembre 2017** (para evitar contaminación temporal).
5. Deduplicación basada en heurísticas.
6. **Remover todos los documentos de Wikipedia**. Esta exclusión es metodológicamente crucial: Wikipedia es la fuente de evaluación de muchos benchmarks (WikiText-2, WikiText-103, etc.) y entrenar en ella inflaría artificialmente los resultados zero-shot.

**Resultado final: ~8 millones de documentos, ~40GB de texto, ~10B tokens.**

### 5.3 Por qué importa

Tres observaciones sobre WebText que vale la pena enfatizar:

1. **No-Wikipedia es una decisión epistemológica fuerte.** Permite que las evaluaciones en WikiText-103 sean honestas. Es una práctica que muchos papers posteriores no respetaron.
2. **Reddit como filtro humano** anticipa la idea de "instrucciones implícitas en datos curados" que llegaría con RLHF e Instruction Tuning. WebText es, en cierto modo, el primer "RLHF light": filtrado por preferencia humana agregada.
3. **40GB son pequeños comparados con lo que vendría.** GPT-3 entrenaría sobre ~570GB (400B tokens, Common Crawl filtrado + WebText + Books + Wikipedia). Pero en 2019, 40GB era ya un orden de magnitud sobre BookCorpus.

### 5.4 Análisis de overlap (Sección 4 del paper)

Una sección frecuentemente subestimada: los autores construyen **Bloom filters de 8-gramos** sobre WebText y miden el overlap con los test sets de los benchmarks usados. Tabla 6 del paper:

| Dataset | Overlap test/dataset train | Overlap test/WebText train |
|---|---:|---:|
| PTB | 2.67% | 0.88% |
| WikiText-2 | 0.66% | **1.63%** |
| enwik8 | 7.50% | 6.31% |
| text8 | 2.34% | 3.94% |
| WikiText-103 | 9.09% | 2.42% |
| 1BW | **13.19%** | 3.75% |

Conclusión de los autores: el overlap WebText/test es modesto (promedio 3.2%) y comparable o menor al overlap entre los propios train/test de los benchmarks. Esto refuerza que los resultados zero-shot no son fruto de contaminación masiva. Es una práctica metodológica que sería bueno ver más seguido en papers de LLMs.

---

## 6. Evaluación zero-shot en language modeling

### 6.1 La Tabla 3 del paper

La Tabla 3 reporta resultados zero-shot en 8 datasets clásicos de LM. La unidad es **perplexity (PPL)** o **bits per character (BPC)** según convención, salvo CBT y LAMBADA donde se reporta accuracy.

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

**SOTA zero-shot en 7 de 8 datasets** (todos excepto 1BW). El único en el que GPT-2 no logra SOTA es 1BW (One Billion Word Benchmark), por dos razones identificadas por los autores: (i) es el dataset de entrenamiento más grande, y (ii) **su preprocesamiento destruye toda estructura larga** — las oraciones del 1BW están shuffled, eliminando dependencias cross-sentence que son justamente donde un LM grande brilla.

### 6.2 Observaciones críticas

- Las **mejoras más espectaculares se dan en datasets pequeños** (PTB, WikiText-2), donde los LMs especializados tenían 1-2M tokens de training. Aquí el preentrenamiento masivo "rellena" la falta de datos en-distribución.
- LAMBADA — diseñado específicamente para medir dependencias largas — pasa de 99.8 PPL a 8.63, un orden de magnitud. La accuracy salta de 59% a 63%, llegando incluso a 63.24% con un stop-word filter trivial.
- Para enwik8 y text8 (BPC) GPT-2 supera a Transformer-XL y arquitecturas especializadas en compresión de byte-level. Esto es notable porque GPT-2 ni siquiera trabaja a nivel de byte: opera sobre BPE.

### 6.3 De-tokenizers invertibles

Un detalle técnico que el paper menciona en pasada pero que es importante: GPT-2 reporta resultados aplicando **de-tokenizers invertibles** que remueven artefactos específicos de tokenización de cada dataset (estilo PTB con sus contracciones partidas, por ejemplo). Esto da 2.5–5 puntos de perplexity de ganancia sin contaminar el setting zero-shot porque los de-tokenizers son invertibles y no agregan información — solo "traducen" el formato del test set al formato natural de WebText.

---

## 7. Capacidades emergentes en tareas downstream

El corazón experimental del paper son las tareas downstream donde GPT-2 **no fue entrenado supervisadamente**.

### 7.1 Reading Comprehension (CoQA)

CoQA (Conversational Question Answering, Reddy et al. 2018) tiene documentos de 7 dominios con diálogos QA en lenguaje natural. El SOTA supervisado (BERT) usa 127,000 pares QA etiquetados.

GPT-2 procedure: concatenar `document + conversation_history + A:` y dejar que el modelo genere la respuesta con greedy decoding.

**Resultado: 55 F1**, igualando o superando a 3 de los 4 baselines supervisados. El SOTA supervisado (BERT-based) llega a ~89 F1. La inspección manual revela que GPT-2 a menudo usa heurísticas simples ("responder con un nombre del documento ante una pregunta `who`"), pero el hecho de que sin entrenamiento de QA logre 55 F1 es señal fuerte.

### 7.2 Resumen (CNN/DailyMail)

Procedimiento: añadir `TL;DR:` después del artículo y generar 100 tokens con top-k=2 sampling. Tomar las primeras 3 oraciones como resumen.

| Sistema | R-1 | R-2 | R-L | R-AVG |
|---|---:|---:|---:|---:|
| Bottom-Up Sum (SOTA) | **41.22** | **18.68** | **38.34** | **32.75** |
| Lede-3 | 40.38 | 17.66 | 36.62 | 31.55 |
| Seq2Seq + Attn | 31.33 | 11.81 | 28.83 | 23.99 |
| **GPT-2 TL;DR:** | 29.34 | 8.27 | 26.58 | 21.40 |
| Random-3 | 28.78 | 8.63 | 25.52 | 20.98 |
| **GPT-2 no hint** | 21.58 | 4.03 | 19.47 | 15.03 |

Observaciones:
- GPT-2 con `TL;DR:` apenas supera al baseline de "3 oraciones random". Como modelo de resumen, es flojo.
- Pero la diferencia **GPT-2 con prompt vs sin prompt** es de 6.4 puntos en ROUGE promedio — evidencia clara de que el prompt invoca task-specific behavior. Este es el dato conceptual importante: el prompting **funciona**, aun cuando el output no sea competitivo.

### 7.3 Traducción (WMT-14 English-French)

Procedimiento: condicionar al modelo con varios pares ejemplo `english sentence = french sentence`, luego dar la frase inglesa y generar con greedy decoding.

- WMT-14 En→Fr: **5 BLEU** (peor que substitución palabra-por-palabra con bilingual lexicon, ~9 BLEU).
- WMT-14 Fr→En: **11.5 BLEU** (supera varios baselines de traducción no supervisada de Artetxe et al. 2017 y Lample et al. 2017, aunque queda por debajo del SOTA unsupervised de 33.5 BLEU).

Lo sorprendente: cuando los autores corren un detector de idioma sobre WebText, encuentran solo **10MB de texto francés** — aproximadamente 500x menos que los corpus monolingües franceses usados en NMT no supervisada. Es decir, GPT-2 traduce *un poco* habiendo visto *casi nada* de francés. Esto refuerza la tesis: la tarea está latente en el corpus diverso.

### 7.4 Question Answering (Natural Questions)

Procedimiento: prompting con pares ejemplo question/answer, luego pregunta sin contexto.

GPT-2 XL responde correctamente al **4.1%** de las preguntas (exact match). GPT-2 Small responde 1.0% (no supera al baseline de "respuesta más frecuente para el tipo de pregunta"). La Tabla 5 del paper muestra las 30 respuestas de mayor confianza: 17 de 20 son correctas en las primeras posiciones (Charles Darwin como autor de *Origin of Species*, Albert Einstein con la teoría de la relatividad, Sigmund Freud como padre del psicoanálisis, etc.). Las respuestas con probabilidad >50% según el modelo aciertan ~63%, evidencia de que **el modelo está bien calibrado en su propia incertidumbre**.

Está muy por debajo de los sistemas open-domain QA (30-50%), pero es 5x mejor que la mejor publicación previa de QA con LMs. Y, otra vez, escala log-linealmente con el tamaño del modelo.

### 7.5 Winograd Schema Challenge

Test clásico de razonamiento de sentido común y resolución de ambigüedad pronominal. GPT-2 XL alcanza **70.70% accuracy**, superando el SOTA previo (Trinh & Le 2018) por 7 puntos. Solo 273 ejemplos en el dataset, así que el resultado es ruidoso, pero la tendencia con tamaño es clara y monótona.

### 7.6 Children's Book Test (CBT)

Test de cloze sobre libros de Project Gutenberg, evaluado en common nouns y named entities.

- CBT-CN: **93.30%** (Human: ~96%, previo SOTA: 85.7%)
- CBT-NE: **89.05%** (Human: ~92%, previo SOTA: 82.3%)

Los autores verificaron que *The Jungle Book*, uno de los libros del test set, está en WebText, así que reportan resultados sobre el validation set que no tiene overlap. Aun así, GPT-2 cierra la mayor parte del gap con humanos.

### 7.7 Generación de texto larga (el famoso "unicorn passage")

La sección 5 del paper incluye en el apéndice (Tabla 13) el ejemplo más famoso del paper, conocido coloquialmente como "the unicorn story". El prompt era una mini-noticia ficticia sobre el descubrimiento de unicornios en los Andes, y GPT-2 generó varios párrafos coherentes, con citas inventadas de investigadores ficticios, conexiones con mitología incaica, etc. — todo gramaticalmente impecable y temáticamente consistente durante cientos de tokens.

Este ejemplo se viralizó en febrero 2019 y fue **el momento en que el gran público empezó a notar que algo había cambiado en NLP**. La capacidad de generar texto coherente durante párrafos completos, sin perder el hilo, sin contradicciones internas obvias, era cualitativamente nueva. Para muchos profesionales que habían trabajado con seq2seq, LSTMs y modelos previos, fue la primera vez que un LM "parecía haber leído el libro completo".

---

## 8. Sampling, generación y técnicas de decoding

GPT-2 popularizó (aunque no inventó) varias técnicas de sampling que se volvieron estándar.

### 8.1 Greedy decoding y sus problemas

Para tareas extractivas (CoQA, QA) GPT-2 usa greedy decoding. Para generación libre, greedy produce loops repetitivos. El paper lo reconoce explícitamente en la sección de limitaciones.

### 8.2 Top-k sampling

En lugar de samplear de toda la distribución $p(s_t \mid s_{<t})$, se trunca a los $k$ tokens más probables, se renormaliza, y se samplea. Para summarization el paper usa $k=2$, lo que reduce repetición y "encourages more abstractive summaries than greedy decoding".

Formalmente:
$$
p'(s_t) =
\begin{cases}
\frac{p(s_t)}{\sum_{s \in \text{Top-}k} p(s)} & \text{si } s_t \in \text{Top-}k \\
0 & \text{en otro caso}
\end{cases}
$$

### 8.3 Nucleus / top-p sampling

GPT-2 no introduce top-p formalmente (Holtzman et al. 2019, *The Curious Case of Neural Text Degeneration*, lo formaliza meses después), pero el contexto de las generaciones de GPT-2 motivó la aparición de top-p. La idea: en vez de un $k$ fijo, definir un conjunto dinámico
$$
V^{(p)} = \min\left\{ V' \subseteq V : \sum_{s \in V'} p(s) \geq p \right\}
$$
y samplear sobre $V^{(p)}$ renormalizado. Con $p=0.9$ o $0.95$ típicamente.

### 8.4 Temperature

Aunque no es central en el paper, todo el ecosistema de uso de GPT-2 popularizó la temperatura $T$ aplicada a los logits:
$$
p_T(s_t) = \frac{\exp(z_t / T)}{\sum_s \exp(z_s / T)}
$$
con $T<1$ aguzando la distribución (más conservador) y $T>1$ aplanándola (más diverso).

### 8.5 El legado del sampling

La discusión de sampling en GPT-2 marca el inicio de un sub-campo: cómo extraer texto natural de un LM. Las decisiones de decoding empezaron a ser tan importantes como la arquitectura. Para 2020-2021 ya era estándar combinar nucleus + temperature + repetition penalty.

---

## 9. Limitaciones reconocidas

El paper es notablemente honesto sobre las limitaciones. Vale la pena listarlas porque establecen la agenda de los años siguientes.

1. **Repetición en generación libre.** Greedy decoding entra en loops. Sampling resuelve parcialmente pero introduce drift temático.
2. **Coherencia larga sigue siendo frágil.** Los unicornios funcionan, pero pasajes de varios miles de tokens divergen, contradicen contextos previos, "alucinan" hechos.
3. **Heurísticas baratas en QA y resumen.** El modelo a veces resuelve tareas con shortcuts (responder con un nombre cualquiera del documento, copiar las primeras oraciones) en lugar de comprensión real.
4. **No alcanza fine-tuning supervisado.** Zero-shot establece un *floor*; en muchas tareas hay todavía 30-50 puntos de gap con el SOTA supervisado.
5. **Generación de lyrics, código complejo o dominios técnicos se degrada a gibberish.** El modelo no tiene representaciones suficientemente fuertes en sub-dominios escasos en WebText.
6. **Aún underfitting WebText.** Los autores remarcan que GPT-2 XL "is still significantly worse than prior work on the One Billion Word Benchmark" y que "has as of yet improved given more training time". Es decir, dejan claro que el modelo no agotó el corpus.

Esta última observación es estratégicamente importante: **el paper de GPT-2 termina diciendo "no terminamos de entrenarlo y aun así está mejorando"**. Era una invitación abierta a escalar más.

---

## 10. Impacto y legado

### 10.1 Validación temprana de scaling laws

Kaplan et al. 2020 (*Scaling Laws for Neural Language Models*) formalizaría que la perplexity de un LM escala como una power-law en parámetros, datos y compute:
$$
L(N) \approx \left( \frac{N_c}{N} \right)^{\alpha_N}
$$
con $\alpha_N \approx 0.076$ para parámetros, similares para data y compute. Pero la **evidencia visual** de esta ley ya está en la Figura 1 de GPT-2: cuatro tamaños, cuatro tareas, todas las curvas log-lineales. GPT-2 fue el primer paper en mostrar de manera tan limpia y deliberada las curvas de scaling. La decisión de entrenar 4 tamaños log-uniformes no es casual: estaba pensada para mostrar la tendencia.

### 10.2 "Unsupervised multitask learning" como lente

El framing del paper se vuelve dominante. Después de GPT-2, hablar de un LLM como "modelo de language modeling" es casi un eufemismo: la comunidad acepta que un LM suficientemente grande es, *de facto*, un sistema multitarea. Esta lente se aplica a T5, GPT-3, PaLM, LLaMA, etc.

### 10.3 El debate de openness vs safety

El staged release de OpenAI generó posiciones encontradas:
- Críticos (Hugging Face, varios académicos): no liberar el modelo es paternalista, frena la investigación, y la barrera de réplica es solo cómputo (que está disponible para actores con recursos pero no para investigadores independientes).
- Defensores: las amenazas de fake news automatizadas, spam personalizado, etc., son reales y vale ser cuidadoso.

OpenAI publicó un *follow-up report* en noviembre 2019 (*Release Strategies and the Social Impacts of Language Models*) documentando que no observaron uso malicioso significativo en los releases parciales, y procedieron a liberar el 1.5B. Pero el precedente quedó establecido: **modelos cerrados detrás de APIs** se vuelven la norma para los actores comerciales (GPT-3, GPT-4, Claude, Gemini), mientras que la línea open-weights (LLaMA, Mistral, etc.) queda como alternativa.

### 10.4 Pavimentando GPT-3

GPT-3 (Brown et al. 2020) es esencialmente "GPT-2 más grande". Misma arquitectura básica (con sparse attention en algunas capas), mismo objetivo, más datos, más parámetros (175B vs 1.5B, 100x), más cómputo. El paper de GPT-3 introduce **in-context learning few-shot** como reemplazo del fine-tuning, pero esa idea es una extensión directa del prompting zero-shot que GPT-2 estableció. Sin GPT-2 no hay GPT-3.

### 10.5 Estética del paper

Algo subestimado: GPT-2 establece un cierto **estilo de paper técnico de OpenAI** — relativamente corto (24 páginas, gran parte apéndices), conversational, no enviado a conferencia con peer review formal, publicado como technical report con un blog post simultáneo. Este formato se vuelve dominante para los releases comerciales posteriores (GPT-3, GPT-4 system card, Claude papers, etc.). Tiene ventajas (rapidez, accesibilidad) y desventajas (menos rigor, menos reproducibilidad). Es parte del costo de la era LLM.

---

## 11. Conexión con la clase 20 y el arco GPT-1 → GPT-2 → GPT-3

La clase 20 del Diplomado IA UC se concentra en LLMs modernos. GPT-2 es la pieza intermedia y conceptualmente la más interesante del arco.

| | GPT-1 (2018) | GPT-2 (2019) | GPT-3 (2020) |
|---|---|---|---|
| Tamaño máximo | 117M | 1.5B | 175B |
| Datos | BookCorpus 800M tokens | WebText 10B tokens | Common Crawl + 400B tokens |
| Adaptación | Fine-tuning supervisado por tarea | **Zero-shot prompting** | **Few-shot in-context learning** |
| Tesis | Pre-training + FT es transferible | LMs grandes son multitask learners | LMs gigantes son meta-learners few-shot |
| Arquitectura | Transformer decoder post-LN | Transformer decoder pre-LN | Idem + sparse attention |
| Polémica | Ninguna | Staged release | API-only, RLHF |

El mensaje de GPT-2 al curso es **radical en su simplicidad**: misma arquitectura, más grande, más datos diversos, sin cambios en el objetivo de entrenamiento — y la promesa de "el LM aprende todo lo que esté en la web". Es la primera vez que se enuncia con claridad la receta que define los LLMs modernos: **scale + diversity + autoregressive language modeling = emergent capabilities**.

Más concretamente para clase 20:

- Si la clase introduce **prompt engineering**, GPT-2 es donde el prompting nace como técnica de evaluación. `TL;DR:`, `translate to french`, prompts con ejemplos — todo está aquí en embrión.
- Si la clase introduce **scaling laws**, la Figura 1 de GPT-2 es el primer plot canónico del campo.
- Si la clase introduce **emergent abilities**, los unicornios y CoQA zero-shot son el ejemplo motivador.
- Si la clase introduce **AI safety**, el staged release de GPT-2 es el primer episodio público del debate.
- Si la clase introduce **byte-level BPE / tokenización moderna**, GPT-2 es la referencia.
- Si la clase introduce **pre-LN**, GPT-2 es donde se vuelve canónico.

---

## 12. Notas para integrar al sitio del curso

Sugerencias concretas para `clase_20/papers/gpt-2-radford-2019.md` en el sitio Hugo:

### 12.1 Estructura propuesta del post

1. **Hero / TL;DR** (2-3 líneas): "Same architecture, just bigger, just more diverse — LMs start solving tasks they were never trained on. Scaling laws antes de tener nombre."
2. **Línea de tiempo visual** GPT-1 → BERT → GPT-2 → T5 → GPT-3, con fechas.
3. **Tabla de los 4 tamaños** (Tabla 2 del paper), enfatizando el espaciamiento log-uniforme.
4. **Las 5 modificaciones arquitectónicas vs GPT-1** como bullets.
5. **WebText**: tres bullets (Reddit karma ≥ 3 como filtro humano, no Wikipedia, 40GB).
6. **La Figura 1 del paper** reproducida o re-renderizada, con énfasis en la monotonía log-lineal.
7. **Tabla 3 de zero-shot LM** (7/8 SOTA).
8. **Capacidades emergentes**: CoQA, traducción, resumen, unicorn story, con un ejemplo concreto de prompt cada uno.
9. **El debate del staged release**: media página, con links al follow-up report de OpenAI.
10. **Conexión con GPT-3**: tabla comparativa.
11. **Math callouts**:
    - Factorización autoregresiva $p(x) = \prod_i p(s_i \mid s_{<i})$.
    - Factorización con tarea $p(\text{output} \mid \text{input}, \text{task})$.
    - Top-k sampling.
    - Init residual $1/\sqrt{N}$.
12. **Cross-links bidireccionales** con:
    - `papers/gpt-1-radford-2018.md`
    - `papers/bert-devlin-2018.md`
    - `papers/gpt-3-brown-2020.md`
    - Clase 14 (Transformer arquitectura)
    - Clase 19 (BERT y el paradigma pretrain + finetune)
    - Sección dedicada de scaling laws (cuando llegue)

### 12.2 Quotes citables

Algunas frases del paper que vale la pena destacar literalmente:

> "Our suspicion is that the prevalence of single task training on single domain datasets is a major contributor to the lack of generalization observed in current systems."

> "Since the supervised objective is the same as the unsupervised objective but only evaluated on a subset of the sequence, the global minimum of the unsupervised objective is also the global minimum of the supervised objective."

> "When a large language model is trained on a sufficiently large and diverse dataset it is able to perform well across many domains and datasets."

> "All models still underfit WebText and held-out perplexity has as of yet improved given more training time." — Esta es la línea más importante del paper en retrospectiva.

### 12.3 Pitfalls al integrar

- **No confundir prompting zero-shot de GPT-2 con few-shot in-context learning de GPT-3.** Son distintos: GPT-2 da una *descripción* o *hint* de la tarea (`TL;DR:`, `translate to french`); GPT-3 da varios *ejemplos completos* de la tarea como contexto. La transición conceptual es importante.
- **No olvidar el análisis de overlap (Sección 4).** Muchos resúmenes del paper omiten esto, pero es metodológicamente lo más importante para defender la legitimidad de los resultados zero-shot.
- **Mencionar el follow-up de release.** El staged release no es solo una anécdota: es un caso de estudio que vale citar cuando se discutan los releases de Claude, GPT-4, LLaMA, etc.

### 12.4 Material adicional sugerido

- Reproducir localmente con `transformers` un GPT-2 small y replicar el experimento de prompt zero-shot en CoQA o CBT (es factible en MPS/CUDA en menos de una hora).
- Mostrar la diferencia top-k vs top-p vs temperature con un ejemplo concreto del unicorn passage.
- Generar la Figura 1 manualmente con los 4 puntos de la Tabla 3 y mostrar la regresión log-lineal — es un ejercicio pedagógico potente.

---

## 13. Cierre

GPT-2 es, en la historia del NLP moderno, la **bisagra**. Antes: pre-training + fine-tuning supervisado, una arquitectura encoder dominante (BERT), tareas resueltas con cabezas especializadas. Después: prompting, scaling, emergencia, modelos generales que se especializan en runtime, debate de safety/openness, race for scale. Casi todo lo que define a los LLMs actuales — incluyendo el régimen comercial, el lenguaje del campo, las técnicas de evaluación y las preocupaciones éticas — se reconoce ya, en germen, en este paper de 24 páginas de OpenAI publicado un día de febrero de 2019.

El paper no resuelve nada definitivamente. Sus resultados son modestos en muchas tareas. Sus limitaciones son grandes. Pero su tesis — "un LM grande y diverso es, en el límite, un sistema multitarea no supervisado" — es la apuesta intelectual sobre la que se construye toda la línea de trabajo que va a GPT-3, ChatGPT, Claude, Gemini y todo lo que viene.

Para el curso, GPT-2 es el paper donde el estudiante debe **detenerse a entender la Figura 1**. Si esa figura se entiende en su radicalidad (cuatro puntos en log-log mostrando que escalar capacidad mejora monótonamente la zero-shot performance en tareas que el modelo no vio), el resto del arco LLM se vuelve, no obvio, pero al menos legible.
