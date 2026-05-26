# Análisis interno — Zellers et al. (2018) "SWAG: A Large-Scale Adversarial Dataset for Grounded Commonsense Inference"

> Documento complementario al material público del site sobre la clase 20 del Diplomado IA UC. Aquí se profundiza en aspectos que la lectura superficial del paper deja implícitos: la mecánica formal de Adversarial Filtering (AF), por qué un dataset construido contra LSTMs en 2018 colapsó frente a BERT meses después, la genealogía SWAG → HellaSwag → benchmarks LLM modernos, y la conexión técnica con `BertForMultipleChoice` y `XLNetForMultipleChoice` que aparecen en el lab 20.

- **Paper**: Zellers, Bisk, Schwartz, Choi. *SWAG: A Large-Scale Adversarial Dataset for Grounded Commonsense Inference*. arXiv:1808.05326v1 (16 Aug 2018). EMNLP 2018.
- **Autores y afiliación**: Rowan Zellers, Yonatan Bisk, Roy Schwartz, Yejin Choi — Paul G. Allen School of Computer Science & Engineering, University of Washington + Allen Institute for AI (AI2).
- **PDF local**: [`Zellers-SWAG-2018.pdf`](Zellers-SWAG-2018.pdf).
- **Sitio del dataset**: `https://rowanzellers.com/swag`.
- **Sigla**: Situations With Adversarial Generations.

---

## 1. Contexto histórico: 2018 y el "grial" del commonsense reasoning

Para entender por qué SWAG fue un paper bisagra hay que situarlo en la cronología del NLP de 2018. Ese año fue el más denso en cambios de paradigma desde la introducción de word2vec (2013). En los doce meses entre enero y diciembre de 2018, la comunidad asistió a:

| Mes | Evento |
|---|---|
| Ene 2018 | ULMFiT (Howard & Ruder) — fine-tuning gradual de LSTMs |
| Feb 2018 | ELMo (Peters et al.) — embeddings contextualizados con BiLSTM |
| Jun 2018 | GPT-1 (Radford et al.) — Transformer decoder + fine-tuning |
| **Ago 2018** | **SWAG** (Zellers et al.) — commonsense benchmark + Adversarial Filtering |
| Oct 2018 | BERT (Devlin et al.) — Transformer encoder bidireccional |
| Nov 2018 | BERT-large alcanza 86.3% en SWAG — el dataset queda casi "resuelto" 3 meses después de salir |

Esta proximidad temporal es la clave para leer SWAG. El paper se escribe asumiendo que el estado del arte en NLI son modelos como **ESIM+ELMo** (~59% en SWAG) y que la frontera humana (88%) está lejos. Tres meses después, BERT entra al ring y la frontera se cierra a 1.7 puntos. Esto define la suerte conceptual del paper: SWAG es simultáneamente un éxito (definió un formato de evaluación que persiste hoy en MMLU/HellaSwag) y un caso de estudio sobre la fragilidad de los datasets adversariales construidos contra modelos del momento.

### 1.1 ¿Por qué "grounded commonsense inference" en 2018?

A mediados de 2018 la comunidad de NLI estaba estancada por dos razones:

**Razón 1: NLI clásico se había vuelto un juego de atajos estadísticos.** SNLI (Bowman et al. 2015) y MultiNLI (Williams et al. 2018) eran los benchmarks de referencia para entailment. Pero dos papers casi simultáneos demostraron que estaban viciados:

- Gururangan et al. (NAACL 2018), *Annotation Artifacts in Natural Language Inference Data*, mostró que un clasificador que ve **solo la hipótesis** (sin la premisa) alcanza ~67% en SNLI — versus el baseline del 33% si las tres etiquetas estuvieran balanceadas. Es decir, los crowdworkers que escribieron las hipótesis introdujeron sesgos léxicos sistemáticos: hipótesis cortas y con negaciones tendían a ser contradicciones, hipótesis con generalizaciones léxicas ("animal" en lugar de "dog") tendían a ser entailments.
- Poliak et al. (StarSem 2018), *Hypothesis Only Baselines in Natural Language Inference*, replicó el resultado en SNLI, MultiNLI, SciTail y otros. La conclusión es contundente: muchos datasets de NLI **no requieren** la premisa para alcanzar performance sobre-humana respecto al random baseline.

Ambos papers son citados explícitamente por Zellers et al. y son la motivación directa de SWAG. La pregunta es: **¿se puede construir un dataset de NLI donde un modelo de bag-of-words sobre la hipótesis no funcione?**

**Razón 2: los datasets de commonsense existentes eran o muy pequeños o muy artificiales.** En el momento de escritura de SWAG el panorama era:

| Dataset | Año | Tamaño | Problema |
|---|---|---|---|
| **COPA** (Roemmele) | 2011 | 1K | Demasiado pequeño para entrenar |
| **ROCStories / Story Cloze** | 2016 | 50K stories, 3.7K cloze test | Cloze test tiene sesgos (Cai et al. 2017) |
| **JOCI** (Zhang et al.) | 2017 | 39K | Regresión a label ordinal, no multiple choice |
| **MCScript** | 2018 | 13K | Pequeño, dominio cerrado |
| **NarrativeQA** | 2018 | 47K | QA generativo, no multiple choice |
| **ARC** | 2018 | 7.7K | Preguntas de ciencia escolar, muy específico |

Ninguno combinaba (1) tamaño grande (>50K), (2) commonsense físico/situacional, (3) formato multiple choice, (4) bajo nivel de sesgos. SWAG cubrió el hueco con 113K ejemplos.

### 1.2 El nombre "grounded"

El adjetivo *grounded* (anclado, situado) en el título no es decorativo. Indica que las premisas se extraen de **descripciones de videos reales**, no de elicitación crowd ni de generación sintética. Los videos provienen de:

- **ActivityNet Captions** (Krishna et al. 2017): 20K clips de YouTube de 203 tipos de actividades (gimnasia, tocar guitarra, etc.), cada uno con párrafos describiendo el contenido temporalmente.
- **LSMDC** (Large Scale Movie Description Challenge, Rohrbach et al. 2017): 128K captions de descripciones de audio de películas, originalmente generadas para accesibilidad de personas con discapacidad visual.

Que las premisas vengan de captions de video es lo que ancla el dataset al mundo físico: cada premisa describe una situación que **efectivamente ocurrió** en algún video real, con todas las restricciones de plausibilidad física que eso implica. Es decir, SWAG no es un dataset de "razonamiento lógico abstracto" — es un dataset de "qué pasa físicamente en la siguiente toma del video". Esta elección deliberada por parte de Zellers et al. anticipa el énfasis del campo en *embodied AI* y *world models*.

### 1.3 La línea de investigación de Choi

Yejin Choi (UW + AI2) había estado empujando consistentemente la agenda de commonsense reasoning desde 2014. SWAG es parte de una secuencia de papers de su grupo:

- *Event2Mind* (Rashkin et al. 2018) — inferir intenciones y reacciones detrás de eventos.
- **SWAG** (Zellers et al. 2018) — inferir continuaciones plausibles.
- *HellaSwag* (Zellers et al. 2019) — versión post-BERT.
- *ATOMIC* (Sap et al. 2019) — knowledge graph de relaciones causales cotidianas.
- *Social IQA* (Sap et al. 2019) — commonsense social.
- *PIQA* (Bisk et al. 2020) — commonsense físico.
- *WinoGrande* (Sakaguchi et al. 2020) — versión escalada de Winograd Schema con AF.

Esta saga, leída en conjunto, es la respuesta de Choi a la hipótesis de que el commonsense reasoning requiere benchmarks específicos y técnicas dedicadas, no se resuelve "gratis" entrenando modelos más grandes en más texto. La predicción se demostraría parcialmente equivocada: GPT-3 (2020) y sus sucesores eventualmente saturaron casi todos estos benchmarks. Pero el formato evaluativo que la saga estableció es el que usan todos los benchmarks de LLMs hoy.

---

## 2. Definición formal de la tarea

SWAG es un problema de **multiple choice con 4 opciones**. Dada una premisa $c$, el modelo debe elegir la continuación verbal correcta entre $\{v_1, v_2, v_3, v_4\}$. Formalmente:

$$\hat{i} = \arg\max_{i \in \{1,2,3,4\}} f_\theta(s, n, v_i)$$

donde:
- $c = (s, n)$ es el contexto: una oración completa $s$ más una *noun phrase* $n$ que inicia la oración siguiente.
- $v_i$ es la *verb phrase* candidata que completa la oración iniciada por $n$.
- $f_\theta$ es el modelo, que asigna un score escalar a cada combinación premisa-candidato.

La etiqueta gold es $i_{\text{gold}} \in \{1,2,3,4\}$.

### 2.1 Estructura del input — por qué se separa $n$ y $v_i$

Una elección no obvia: las cuatro opciones comparten una *noun phrase* $n$ fija (el sujeto de la oración siguiente) y solo difieren en la *verb phrase* $v_i$. Por ejemplo, en el ejemplo canónico del paper:

> **Premisa $s$**: "On stage, a woman takes a seat at the piano."
> **Noun phrase $n$**: "She"
> **Verb phrases $v_i$**:
> - a) "sits on a bench as her sister plays with the doll."
> - b) "smiles with someone as the music plays."
> - c) "is in the crowd, watching the dancers."
> - **d) "nervously sets her fingers on the keys."** ← correcta

Esto no es decorativo. Las cuatro opciones **comparten sujeto sintáctico**, por lo que las pistas léxicas de "quién hace la acción" se neutralizan: no se puede distinguir la opción correcta por la persona/género del sujeto, porque siempre es "She". Toda la decisión está en la elección del verbo y su complemento.

La partición sintáctica se obtiene con el constituency parser de **Stern et al. (2017)** sobre la segunda caption. Si el parser no logra dividir limpiamente $s_2$ en (NP, VP), o si $s_2$ tiene menos de 5 tokens o contiene tokens raros (≤3 ocurrencias en el corpus), el par se descarta.

### 2.2 Tres ejemplos del paper (Table 1)

El paper abre con tres ejemplos canónicos que vale la pena leer juntos para captar el tipo de razonamiento:

**Ejemplo 1 — piano**:
> "On stage, a woman takes a seat at the piano. She **nervously sets her fingers on the keys.**"
> Distractores: "sits on a bench as her sister plays with the doll" / "smiles with someone as the music plays" / "is in the crowd, watching the dancers"

El razonamiento correcto requiere saber que (1) si alguien se sienta al piano *en un escenario*, está a punto de tocar; (2) un pianista pone los dedos en las teclas, no se sienta como espectador.

**Ejemplo 2 — monkey bars**:
> "A girl is going across a set of monkey bars. She **gets to the end and stands on a wooden plank.**"
> Distractores: "jumps up across the monkey bars" / "struggles onto the monkey bars to grab her head" / "jumps up and does a back flip"

Requiere razonar sobre la trayectoria física: si la niña *ya está cruzando*, lo que sigue es llegar al final, no comenzar a subirse.

**Ejemplo 3 — blow drying the dog**:
> "The woman is now blow drying the dog. The dog **walks into frame and walks towards the dog.**"
> [Nota: este ejemplo del paper está visiblemente mal — parece haber un error en la transcripción. La opción "correcta" según el formato del paper sería la primera: "is placed in the kennel next to a woman's feet". El paper original tiene una errata.]

Más allá de la errata específica, el patrón es claro: los distractores son **lingüísticamente fluidos** (no son gibberish) y **léxicamente relacionados con el dominio** (mencionan "monkey bars", "piano", "dog"), pero violan la causalidad o la plausibilidad física de la situación.

### 2.3 Por qué el formato multiple choice y no regresión

Una decisión de diseño explícita (Sección 6, comparando con JOCI): los autores eligen multiple choice sobre regresión a un score ordinal porque:

1. **Reduce ambigüedad de etiquetas.** Pedir a un annotador "qué tan plausible es esta continuación" (regresión) genera más ruido que pedir "elige la mejor de cuatro" (selección).
2. **Permite comparación directa con humanos.** Es trivial medir accuracy humana en multiple choice; no lo es en regresión.
3. **Encaja en pipelines de NLP estándar.** Una cabeza de clasificación de 4 vías sobre representaciones de input se implementa con cualquier framework.

Esta elección se vuelve canónica: MMLU, HellaSwag, ARC, OpenBookQA, PIQA, SIQA, todos los benchmarks de commonsense reasoning posteriores adoptan multiple choice. SWAG fue el que estableció la convención.

---

## 3. Construcción del dataset

El proceso completo se ilustra en la Figura 1 del paper y consta de cuatro fases:

**Fase 1 — Extracción de premisas reales.** De pares de captions consecutivas de ActivityNet o LSMDC, tomar la primera caption como premisa $s$ y la segunda dividida en (NP, VP) como (continuación inicial $n$, continuación correcta $v_{\text{gold}}$). Esto da pares (premisa, continuación gold) genuinos.

**Fase 2 — Oversampling masivo con un LM.** Generar 1023 continuaciones candidatas por premisa usando un LSTM language model. Las muestras se generan con greedy sampling (no beam search) — discutido más abajo.

**Fase 3 — Adversarial Filtering.** Iterativamente entrenar un ensemble de clasificadores estilísticos y mantener solo las continuaciones que el ensemble **no puede distinguir** de la gold. Esto reduce 1023 candidatos a 9 por premisa.

**Fase 4 — Validación humana.** De los 9 candidatos, 5 se pasan a Mechanical Turk para clasificar como likely/unlikely/gibberish. Los turkers también eligen el "best" y "second best". El resultado son los 3 distractores finales del dataset.

### 3.1 Estadísticas del dataset

Del Apéndice A.5 del paper:

| Métrica | Valor |
|---|---|
| Total questions | 113,557 |
| Train | 73,546 |
| Validation | 20,006 |
| Test | 20,005 |
| Questions with only generated endings (no found) | 25,618 |
| Questions with one original ending | 87,939 |
| Sentence pairs from ActivityNet | 51,439 |
| Sentence pairs from LSMDC | 62,118 |
| Unique contexts | 92,221 |
| Unique endings | 452,683 |
| Vocabulary size | 21,000 |
| Costo total Mechanical Turk | $23,000 (~$0.20 por ejemplo) |
| Pago promedio turker | $8.57/hora |

113K ejemplos era, en 2018, considerable: comparable con SNLI (570K, pero NLI estándar) y mucho más grande que cualquier dataset previo de commonsense (COPA: 1K, RocStories: 4K cloze, MCScript: 13K).

### 3.2 Por qué greedy sampling y no beam search

Detalle técnico crucial (Sección 3.3): para generar las 1023 continuaciones por premisa, el LM usa **greedy sampling** del distribución softmax sobre el vocabulario, **no beam search**.

La razón es contraintuitiva. Beam search produce secuencias con menor perplejidad bajo el LM, lo que las hace más "fluidas" desde el punto de vista lingüístico. Pero precisamente por eso son más fáciles de distinguir de las gold: un clasificador estilístico aprende rápido que las secuencias con perplejidad sospechosamente baja son distractores. Greedy sampling de la distribución completa genera continuaciones con más diversidad léxica, perplejidad más alta y comportamiento estilístico más cercano al texto humano.

Esto es una pre-visión del problema que más tarde se conocería como *neural text degeneration* (Holtzman et al. 2019, *The Curious Case of Neural Text Degeneration*). Holtzman et al. mostrarían que beam search produce texto con distribuciones léxicas anti-naturales (repetitivo, con palabras de alta probabilidad sobre-representadas) y propondrían nucleus sampling como alternativa. SWAG lo intuye un año antes y elige greedy sampling como pis aller.

### 3.3 Cinco LMs para evitar leakage

Otra decisión sutil (footnote 3 del paper): los autores **entrenan 5 LMs separados** sobre 5 splits cruzados del dataset. Cuando generan continuaciones para una premisa, usan el LM que **no vio esa premisa** durante entrenamiento.

La razón: si un único LM se entrena sobre todo el dataset, las continuaciones que genere para una premisa $i$ estarán contaminadas — el LM ha memorizado la continuación gold de esa premisa y tenderá a generar paráfrasis cercanas a ella. Esto contaminaría la separación entre "gold" y "distractor". La validación cruzada con 5 folds elimina ese leakage.

### 3.4 Por qué dos corpus de video y no uno

ActivityNet y LSMDC son complementarios:
- **ActivityNet**: clips de YouTube subidos por usuarios. Actividades estructuradas (deportes, manualidades, cocina, danza). Lenguaje informal pero descriptivo. ~20K videos.
- **LSMDC**: descripciones de audio de películas. Lenguaje narrativo cinematográfico. ~128K captions. Los nombres de personajes son reemplazados por "someone" (LSMDC procesado).

Combinarlos da cobertura amplia de situaciones físicas (ActivityNet) y narrativas (LSMDC). El Apéndice A.1 menciona que se descartó el dataset DiDeMo porque sus captions son fragmentos referenciales ("first time we see people"), no descripciones de eventos.

Limitación importante: ambos corpus tienen el sesgo de "lo que se sube a YouTube" y "lo que se hace película". Hay sobre-representación de actividades occidentales urbanas y sub-representación de contextos no anglo. Este sesgo se hereda por SWAG y por HellaSwag.

### 3.5 Calidad humana del dataset

Tabla 2 del paper, distribución de etiquetas sobre 1000 ejemplos de prueba:

| | Found ending | Generated ending |
|---|---|---|
| Best (etiquetado como mejor) | 53.5% | 9.3% |
| Second Best | 20.2% | 15.9% |
| Neither | 26.3% | 74.8% |
| Likely | 80.3% | 33.3% |
| Unlikely | 19.0% | 57.5% |
| Gibberish | 0.7% | 9.1% |

**Lecturas clave**:
- Los turkers identifican correctamente el found ending como "best" en 53.5% de los casos, "second best" en 20.2% — total 73.7%.
- Sólo 9.1% de las generaciones son etiquetadas como gibberish: el LM está produciendo texto fluido la gran mayoría del tiempo.
- Acuerdo entre annotators medido con **Krippendorff's α**: 0.43 para best/second/neither, 0.39 para likely/unlikely/gibberish. Pairwise agreement: 72% y 64% respectivamente.

Esto valida el resultado central del paper: **88% de accuracy humana** sobre 5 turkers majority-voted, con un único expert annotator (el primer autor) alcanzando 85%. El dataset es difícil incluso para humanos — el 12% restante corresponde a casos genuinamente ambiguos.

---

## 4. Adversarial Filtering — el algoritmo central

Esta es la contribución metodológica del paper y la que más impacto tuvo (se reusa textualmente en HellaSwag, WinoGrande, y muchos otros datasets posteriores).

### 4.1 Definición formal de "dataset adversarial"

Sección 3.1 del paper. Sea:
- $X$ el espacio de input, $Y = \{0, 1\}$ el espacio de etiquetas binarias (correcto/distractor).
- $f_\theta: X \to \mathbb{R}^{|Y|}$ un clasificador entrenable con parámetros $\theta$.
- $D = \{(x_i, y_i)\}_{i=1}^N$ el dataset.
- $L(f_\theta, D)$ la función de pérdida (cross-entropy en este caso).

Un dataset es **adversarial con respecto a la familia $f$** si, para todo split leave-one-out, el error empírico esperado es alto:

$$I(D, f) = \frac{1}{N} \sum_{i=1}^N L\left(f_{\theta_i^\star}, \{(x_i, y_i)\}\right)$$

donde $\theta_i^\star = \arg\min_\theta L(f_\theta, D \setminus \{(x_i, y_i)\})$ es el modelo óptimo entrenado en todo el dataset menos el ejemplo $i$.

Intuitivamente: un dataset es adversarial para $f$ si, no importa cómo se separe en train/test, $f$ no puede generalizar del train al test. Esto es exactamente lo contrario de "fácil para $f$".

### 4.2 El loop de filtrado

Como $A$ (la asignación de distractores) es global y discreta, optimizar directamente $I(D, f)$ es intratable. El Algoritmo 1 del paper propone una heurística iterativa:

```
while convergence not reached:
    1. Split D randomly into D_tr and D_te (80%/20%).
    2. Train ensemble f_theta on D_tr to minimize cross-entropy.
    3. For each context i in D_te:
       a. Identify "easy" distractors: A_i^easy = {j in A_i : f_theta(x_i^+) > f_theta(x_i_j^-)}
       b. Replace N^easy easy distractors with "adversarial" candidates k not in A_i
          satisfying f_theta(x_i_k^-) > f_theta(x_i_j^-).
```

En palabras:
- En cada iteración, partir el dataset en train/test ficticio.
- Entrenar el ensemble en train.
- Para cada ejemplo en test: identificar los distractores que el ensemble clasifica "demasiado bien" (los considera claramente distractor en comparación con el gold). Estos son los "easy" — el modelo está usando atajos estilísticos para distinguirlos.
- Reemplazarlos por candidatos del pool de 1023 generaciones que el modelo **misclassifica** (les da score más alto que al gold). Estos candidatos son "adversariales" en el sentido de que confunden al modelo actual.

Los hiperparámetros usados en SWAG: $N^- = 1023$ candidatos iniciales por premisa, $k = 9$ distractores finales, $N^{\text{easy}} = 2$ reemplazos por iteración, ~140 iteraciones totales (las primeras 100 solo entrenan el MLP, las últimas el ensemble completo).

### 4.3 El ensemble de filtros

Sección 3.4. El ensemble $f$ que se usa para filtrar es **deliberadamente estilístico y débil**:

1. **MLP sobre features de perplejidad del LM**: perplejidad del context, de la continuación, de la última palabra de la continuación, longitudes de context/continuación.
2. **Bag-of-words sobre word embeddings**: promedio de embeddings de la segunda oración.
3. **CNN 1D** con filtros 2-5 sobre la segunda oración.
4. **BiLSTM** sobre los 100 tokens más comunes (resto reemplazado por POS tags). El reemplazo por POS tags fuerza al modelo a aprender patrones sintácticos en lugar de léxicos.

Los cuatro modelos se concatenan en sus representaciones finales y pasan por un MLP de fusión. El ensemble se entrena jointly con cross-entropy.

**¿Por qué este ensemble y no algo más fuerte?** Decisión deliberada (Sección 3.4): "*we designed the model family f to pick up on low-level stylistic features that we posit should not be predictive of whether an event happens next in a video*". Es decir, AF debe eliminar **artefactos estilísticos** — sesgos que un modelo bag-of-words puede explotar. Si AF eliminara también las señales semánticas reales (las que un modelo de razonamiento físico explota), el dataset quedaría sin solución posible. La línea fina del AF está en filtrar lo suficiente para que los atajos no funcionen pero no tanto como para que la tarea deje de ser resoluble.

### 4.4 Convergencia: del 60% al 25%

Figura 2 del paper muestra la curva de test accuracy del ensemble durante 140 iteraciones de AF:

- **Iteración 0**: ensemble alcanza ~60% accuracy. Los distractores iniciales (generaciones random del LM) tienen patrones estilísticos detectables.
- **Iteración ~30**: el MLP solo (basado en perplejidad) ya ha sido derrotado, pero el ensemble completo aún logra ~50%.
- **Iteración 100**: ensemble cae a ~30%.
- **Iteración 140**: ensemble cerca de 25% — accuracy random sobre 4 opciones. AF ha convergido.

El criterio de parada es accuracy del ensemble cerca del random. En ese punto, **ningún modelo de la familia** puede distinguir gold de distractor — al menos no con las features estilísticas que el ensemble explora.

### 4.5 La limitación que el paper reconoce

Sección 5.4, *Where to go next?*:

> "the adversarial models used to filter the dataset are purely stylistic and focus on the second sentence; thus, subtle artifacts still likely remain in our dataset. These patterns are ostensibly picked up by the NLI models (particularly when using ELMo features), but the large gap between machine and human performance suggests that more is required to solve the dataset. As models are developed for commonsense inference, and more broadly as the field of NLP advances, we note that AF can be used again to create a more adversarial version of Swag using better language models and AF models."

Los autores anticipan exactamente lo que iba a ocurrir: un modelo mucho más fuerte que el ensemble de AF podría "romper" el dataset, explotando artefactos que el ensemble no capturó. Lo que no anticiparon es que ese modelo aparecería **2 meses después** (BERT). Esta predicción fue una de las más rápidamente confirmadas en la historia reciente de NLP.

---

## 5. Métricas y resultados de los baselines (era pre-BERT)

Tabla 3 del paper. Reproducimos los resultados clave (accuracy en %), con tres regímenes de input: solo ending, solo segunda oración, contexto completo + segunda oración. Se entrenó cada modelo sobre dos splits: solo found endings, o found + generated highly-ranked endings.

### 5.1 Baselines unarios (input = una sola span)

| Modelo | Ending only (test) | 2nd sentence (test) | Context + 2nd (test) |
|---|---|---|---|
| Random | 25.0 | 25.0 | 25.0 |
| Length (shortest) | 27.0 | — | — |
| ConceptNet (causal relations) | — | 26.0 | — |
| fastText | 26.9 | 27.8 | 28.0 |
| SkipThoughts | 32.1 | 32.4 | — |
| InferSent | 30.2 | 32.0 | — |
| LSTM + GloVe | 31.8 | 32.4 | 43.6 |
| LSTM + Numberbatch | 32.6 | 31.9 | 40.2 |
| **LSTM + ELMo** | **42.9** | **46.7** | **50.6** |

### 5.2 Baselines binarios (input = dos spans, modelo NLI)

| Modelo | 2nd sentence (test) | Context + 2nd (test) |
|---|---|---|
| DualBoW + GloVe | 31.3 | 34.7 |
| DualBoW + Numberbatch | 31.4 | 35.1 |
| InferSent-Bilinear | 31.3 | 40.3 |
| InferSent-MLP | 32.1 | 36.2 |
| SkipThoughts-Bilinear | 35.7 | 35.6 |
| SNLI-ESIM (pretrained on SNLI) | — | 36.1 |
| SNLI-DecompAttn (pretrained) | — | 35.8 |
| DecompAttn + GloVe (retrained) | 30.3 | 47.6 |
| DecompAttn + Numberbatch (retrained) | 31.7 | 48.0 |
| DecompAttn + ELMo (retrained) | 43.4 | 47.3 |
| ESIM + GloVe (retrained) | 35.1 | 52.7 |
| ESIM + Numberbatch (retrained) | 32.6 | 46.4 |
| **ESIM + ELMo** (retrained) | **45.7** | **59.2** |

### 5.3 Performance humana

| Tipo de annotator | Accuracy |
|---|---|
| 1 turker | 82.8 |
| 3 turkers (majority) | 85.1 |
| Expert (primer autor) | 85.0 |
| **5 turkers (majority)** | **88.0** |

### 5.4 Lecturas de la tabla

**1. fastText es un baseline crítico.** En SNLI, fastText (bag-of-n-grams sobre solo la hipótesis) alcanza 67.0% — un atajo léxico devastador. En SWAG, fastText alcanza 29.0% — apenas sobre random. Esto es el resultado más importante metodológicamente: SWAG **logró** lo que se propuso, eliminar los atajos léxicos que arruinaban a SNLI.

**2. El contexto sí ayuda.** En LSTM+ELMo, pasar de "ending only" (42.9%) a "context + 2nd sentence" (50.6%) da +7.7 puntos. En ESIM+ELMo (binario), el salto es de "2nd sentence" (45.7%) a "context + 2nd" (59.2%): +13.5 puntos. El dataset requiere genuinamente la premisa.

**3. ELMo ayuda mucho.** ESIM+GloVe alcanza 52.7%, ESIM+ELMo alcanza 59.2% — un salto de **+6.5 puntos** solo por cambiar embeddings estáticos por contextuales. Esto fue parte de la evidencia que aceleró la adopción de ELMo en 2018.

**4. SNLI-pretrained no transfiere bien.** Modelos pre-entrenados en SNLI y aplicados a SWAG sin retraining alcanzan ~36%. El paper interpreta esto como evidencia de que SWAG requiere un **tipo distinto de razonamiento** (temporal/causal/situacional) que el de SNLI (entailment lógico).

**5. La brecha humano-máquina es grande**: 88% - 59.2% = **28.8 puntos**. Esta brecha fue el "headroom" que justificaba publicar SWAG como benchmark. Tres meses después la brecha cayó a 1.7 puntos.

### 5.5 Análisis de errores cualitativo (Sección 5.2)

Los autores muestrearon 100 ejemplos donde ESIM+ELMo falló y pidieron a 5 turkers ranking las opciones y dar razones. Tabla 4 del paper:

| Razón | Frecuencia | Descripción |
|---|---|---|
| **Situational** | 53.7% | La opción correcta es mejor en el contexto específico |
| **Weirdness** | 18.1% | La opción incorrecta es semánticamente o gramaticalmente extraña |
| **Plausibility** | 14.4% | La opción incorrecta es implausible regardless del contexto |
| **Ambiguous** | 12.0% | Ambas opciones parecen igual de plausibles |
| **Novelty** | 1.8% | La opción incorrecta es redundante con el contexto |

Lectura: el cuello de botella real (53.7%) es **comprensión situacional**. ESIM+ELMo ya filtra bien lo absurdo (las categorías "weirdness" y "plausibility" suman 32.5%, donde el modelo a veces sí elige bien). Lo que no logra es entender qué acción sigue específicamente en este contexto físico particular. El 12% de "ambiguous" pone un techo blando al máximo alcanzable.

---

## 6. El "rompimiento" por BERT (Nov 2018)

Tres meses después de la publicación de SWAG, Devlin et al. (Oct 2018) liberan BERT. En la Tabla 4 del paper de BERT aparece SWAG:

| Sistema | Test accuracy |
|---|---|
| ESIM + GloVe (Zellers 2018) | 52.7 |
| ESIM + ELMo (Zellers 2018) | 59.2 |
| OpenAI GPT (fine-tuned) | 78.0 |
| **BERT-base** | **81.6** |
| **BERT-large** | **86.3** |
| Human (expert) | 85.0 |
| Human (5 turkers majority) | 88.0 |

**BERT-large alcanza 86.3% — 1.3 puntos sobre el expert humano y a 1.7 puntos del techo de 5-turker majority.** El dataset está esencialmente saturado.

### 6.1 ¿Por qué BERT rompe SWAG si AF estaba diseñado para resistir modelos fuertes?

La respuesta está en la limitación que Zellers et al. anticiparon (Sección 5.4): AF se entrenó contra un ensemble de modelos **estilísticos y léxicos** (MLP de perplejidad, BoW, CNN, BiLSTM sobre POS tags). Estos modelos capturan artefactos estilísticos pero **no capturan razonamiento bidireccional profundo ni representaciones contextuales pre-entrenadas en 3.3B tokens**.

BERT no es solo un modelo "más grande" — es un modelo de un tipo cualitativamente distinto:
1. **Pre-training masivo** sobre BooksCorpus + Wikipedia (3.3B palabras) le da conocimiento de mundo que el ensemble de AF no tiene.
2. **Bidireccionalidad profunda** vía Transformer encoder permite atención cruzada entre premisa y candidato a través de 24 capas.
3. **MLM** entrena al modelo a llenar huecos contextualmente — exactamente la habilidad que SWAG mide.
4. La cabeza `BertForMultipleChoice` (sec. 9) está diseñada para que el `[CLS]` token agregue la compatibilidad premisa-candidato en una sola representación de pooling.

En retrospectiva, lo notable no es que BERT rompa SWAG — es que SWAG sobrevivió 3 meses como benchmark difícil. Para muchos otros datasets de NLI/QA, BERT redujo la brecha humano-máquina a cero o la cruzó.

### 6.2 Lección general sobre adversarial filtering

SWAG dejó una lección metodológica que se repetiría: **AF construye un dataset que es adversarial contra la familia de modelos $f$ usada para filtrar, pero no garantiza dificultad contra modelos fuera de esa familia**. Si la familia $f$ está limitada (estilística + perplejidad + BoW + CNN + BiLSTM), nada impide que un modelo de paradigma distinto (Transformer pre-entrenado en masa) explote señales que $f$ no capturó.

Esto motivó a Zellers et al. a volver a aplicar AF, esta vez con BERT-large como discriminador.

---

## 7. HellaSwag (2019) — el sucesor con BERT como filtro

Un año después de SWAG, **Zellers, Holtzman, Bisk, Farhadi, Choi** publican *HellaSwag: Can a Machine Really Finish Your Sentence?* (ACL 2019). HellaSwag es la continuación directa, escrita por el mismo grupo y motivada explícitamente por el colapso de SWAG bajo BERT.

### 7.1 Cambios respecto a SWAG

| Aspecto | SWAG | HellaSwag |
|---|---|---|
| **Filtro AF** | Ensemble estilístico + LSTM | BERT-large |
| **Generador** | LSTM LM | GPT-1/GPT-2 fine-tuned |
| **Tamaño** | 113K | 70K |
| **Fuente** | ActivityNet + LSMDC | ActivityNet + WikiHow |
| **Acc humana** | 88% | ~95% |
| **Acc BERT-large** | 86.3% | **~47%** |
| **Acc GPT-3** | — | ~78% (zero-shot) / ~88% (few-shot) |
| **Acc GPT-4** | — | ~95% |

**Lectura clave**: HellaSwag bajó a BERT-large de 86.3% (en SWAG) a 47% (en HellaSwag). Lo hizo aplicando AF con BERT-large como discriminador, generando candidatos con LMs mucho más fuertes (GPT-2 ya estaba disponible), e incluyendo el dominio WikiHow para diversificar más allá de captions de video.

### 7.2 WikiHow como segundo dominio

WikiHow contiene ~230K artículos de "cómo hacer X" en lenguaje natural. Cada artículo tiene pasos enumerados que constituyen secuencias temporales explícitas. Esto es ideal para commonsense temporal: dado el paso $k$, predecir el paso $k+1$.

Incluir WikiHow rompe el sesgo de "lo que se filma en YouTube" que tenía SWAG y expone a los modelos a procedimientos cotidianos diversos (recetas de cocina, mantenimiento del hogar, trámites administrativos).

### 7.3 La "carrera adversarial" en commonsense

HellaSwag inicia un patrón que se repetiría:

1. Aparece dataset difícil construido con AF contra modelo del momento.
2. Modelo nuevo lo rompe en meses.
3. Mismo grupo (u otro) construye versión adversarial con el modelo nuevo como filtro.
4. Volver al paso 2.

Esta dinámica se llamaría más tarde *dataset arms race* o *moving target benchmarks*. WinoGrande (Sakaguchi et al. 2020) aplica AF a Winograd Schema Challenge usando RoBERTa. Adversarial NLI (Nie et al. 2020) hace AF en NLI estándar con un human-in-the-loop. BIG-Bench (BIG-Bench collaboration 2022) y BIG-Bench Hard intentan curar tareas que sigan siendo difíciles para modelos cada vez más grandes.

La saturación final de HellaSwag por GPT-4 (~95%) sugiere que la carrera adversarial tiene rendimientos decrecientes: en algún punto, los modelos absorben el commonsense necesario por puro escalado. Pero el formato evaluativo (multiple choice, premisa + 4 opciones) sobrevive como estándar.

---

## 8. Análisis del dataset: diversidad y sesgos (Sección 5.1)

### 8.1 Verbos: SWAG vs SNLI

Figura 4 del paper. Distribución de los 40 verbos más frecuentes en la unión de SNLI y SWAG. Hallazgos:

- SWAG tiene mayor proporción de **verbos dinámicos** ("move", "pull", "hit", "roll", "drop").
- SWAG tiene **verbos temporales** ("start", "continue", "come", "begin").
- SNLI tiene mayor proporción de **verbos estáticos** ("sit", "wear", "stand").
- La CDF de verbos muestra que SWAG tiene una **distribución más uniforme** (menos concentrada en los top-10 verbos).

Esto refleja la diferencia de fuente: SNLI proviene de captions de Flickr30K (imágenes estáticas), SWAG de captions de video (secuencias dinámicas). Para razonamiento temporal/causal, la base léxica de SWAG es más rica.

### 8.2 Ausencia de sesgos léxicos

Sección 5.1 reporta:
- En SNLI, fastText alcanza 67.0% solo con bag-of-n-grams sobre la hipótesis.
- En SWAG, fastText alcanza 29.0% — solo 4 puntos sobre random (25%).
- Las palabras más predictivas en SWAG son **infrecuentes**: "dotted" con $P(+|\text{dotted}) = 77\%$ con solo 10.3 ocurrencias, "similar" con $P(-|\text{similar}) = 81\%$ con 16.3 ocurrencias.

Es decir, ningún token frecuente carga señal predictiva. Esto es el éxito metodológico de AF: el dataset es **léxicamente neutro** respecto a la etiqueta.

### 8.3 Diversidad temática

Apéndice A.7, Tabla 7. Un topic model (LDA) sobre el dataset identifica clusters como:
- Deportes con pelota ("ball", "pull", "hit", "wall", "game")
- Cocina ("window", "red", "long", "drink", "bowl", "ingredient", "mix")
- Trepar/exterior ("arm", "speak", "appear", "climb", "tree", "roll")
- Agua/playa ("water", "bar", "board", "blue", "boat", "fly", "river")
- Movimientos faciales ("eye", "smile", "close", "lean", "cover", "remove", "lip")
- Caminar/calle ("walk", "outside", "street", "wave", "pass", "beach", "sidewalk")
- Conducir ("field", "drop", "slide", "drive", "right", "kick", "park", "road")
- Animales ("watch", "dog", "flip", "stick", "land", "demonstrate")
- Danza/deportes ("dance", "lift", "try", "line", "snow", "gun", "catch", "hill")
- Naturaleza/caer ("fall", "crowd", "pour", "shake", "finish", "raise", "grass", "wooden")
- Performance ("perform", "spin", "house", "stage", "routine", "fence", "bow")

Los temas son razonablemente diversos pero hay sesgo claro hacia actividades físicas grabables en video. Hay sub-representación de razonamiento abstracto, social, emocional, financiero, médico.

---

## 9. Conexión con BERT y MultipleChoice — la cabeza estándar

Esta sección es crítica para entender por qué SWAG aparece referenciado en el código de Hugging Face transformers y por qué el lab 20 del Diplomado usa `XLNetForMultipleChoice`.

### 9.1 `BertForMultipleChoice` — diseñada para SWAG

Devlin et al. (2018) en la Sección 4.4 del paper de BERT describen explícitamente la cabeza que diseñaron **para fine-tunear en SWAG**:

> "We construct four input sequences, each containing the concatenation of the given sentence (sentence A) and a possible continuation (sentence B). The only task-specific parameters introduced is a vector whose dot product with the [CLS] token representation C denotes a score for each choice which is normalized with a softmax layer."

Esta cabeza se implementó en el repo `google-research/bert` y luego se trasladó a `transformers` como `BertForMultipleChoice`. Es la cabeza canónica para tareas multiple choice con backbone BERT.

### 9.2 Formato del input

Para una premisa $s$ y $K = 4$ candidatos $\{v_1, v_2, v_3, v_4\}$:

1. Construir $K$ secuencias de input, cada una:
   ```
   [CLS] s [SEP] v_k [SEP]
   ```
   con segmentos $E_A$ para $s$ y $E_B$ para $v_k$.

2. Pasar cada secuencia por el backbone BERT, obteniendo $K$ representaciones de `[CLS]`:
   $$C_k = \text{BERT}([CLS], s, [SEP], v_k, [SEP])_{[CLS]} \in \mathbb{R}^H$$

3. Aplicar un vector lineal $w \in \mathbb{R}^H$ aprendido (parámetro nuevo, único):
   $$s_k = w^\top C_k \in \mathbb{R}$$

4. Softmax sobre los $K$ scores:
   $$p_k = \frac{\exp(s_k)}{\sum_{j=1}^K \exp(s_j)}$$

5. Loss: cross-entropy contra la etiqueta gold $y^\star$:
   $$\mathcal{L} = -\log p_{y^\star} = -\log \frac{\exp(s_{y^\star})}{\sum_{k=1}^K \exp(s_k)}$$

### 9.3 Shape de los tensores

En implementaciones reales (HuggingFace `BertForMultipleChoice.forward`), el input tiene shape:

```
input_ids: [batch_size, num_choices, seq_len]
attention_mask: [batch_size, num_choices, seq_len]
token_type_ids: [batch_size, num_choices, seq_len]
labels: [batch_size]  (índice de la opción correcta, 0..num_choices-1)
```

Internamente, el forward hace:
1. Aplanar a `[batch_size * num_choices, seq_len]`.
2. Pasar por el backbone → `[batch_size * num_choices, seq_len, hidden]`.
3. Extraer `[CLS]` → `[batch_size * num_choices, hidden]`.
4. Proyectar con la cabeza lineal → `[batch_size * num_choices, 1]`.
5. Reshape a `[batch_size, num_choices]`.
6. Softmax + cross-entropy contra `labels`.

Esto explica el truco del `.unsqueeze(0)` que aparece en muchos ejemplos: si tienes solo `[num_choices, seq_len]` para una premisa, necesitas añadir la dimensión de batch para que el modelo no se confunda. `tensor.unsqueeze(0)` convierte `[num_choices, seq_len]` en `[1, num_choices, seq_len]`.

### 9.4 La cabeza es muy barata

El único parámetro nuevo que añade `BertForMultipleChoice` sobre el backbone es el vector $w \in \mathbb{R}^H$. Para BERT-large, $H = 1024$ → **1024 parámetros adicionales**. Esto es comparable a la cabeza más pequeña del paper original de BERT.

Tan pocos parámetros hacen que el fine-tuning sea muy rápido y data-efficient: con los 73K ejemplos de SWAG y batch 16, dos o tres epochs convergen en menos de una hora en una sola TPU.

### 9.5 `XLNetForMultipleChoice` — el lab 20

XLNet (Yang et al. 2019) es un sucesor de BERT que usa **permutation language modeling** en lugar de MLM. La cabeza `XLNetForMultipleChoice` tiene exactamente la misma estructura que `BertForMultipleChoice`:

1. Pasar cada par (premisa, candidato) por el backbone XLNet.
2. Extraer la representación del último token (XLNet usa el último token, no `[CLS]`).
3. Proyectar a un score con una cabeza lineal.
4. Softmax + cross-entropy.

En el lab 20 del Diplomado, la celda 14 muestra un uso de `XLNetForMultipleChoice` con un ejemplo simplificado de 2 opciones (no 4 como SWAG real):

```python
# Pseudo-código del lab 20, celda 14
from transformers import XLNetTokenizer, XLNetForMultipleChoice
import torch

tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
model = XLNetForMultipleChoice.from_pretrained("xlnet-base-cased")

prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
choice0 = "It is eaten with a fork and a knife."
choice1 = "It is eaten while held in the hand."

encoding = tokenizer([[prompt, choice0], [prompt, choice1]],
                     return_tensors="pt", padding=True)
# encoding["input_ids"].shape == [num_choices=2, seq_len]

outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()})
# unsqueeze(0) añade dim de batch: [1, 2, seq_len]

logits = outputs.logits  # shape [1, 2]
```

Tres puntos pedagógicos del lab 20:

1. **El `unsqueeze(0)` es necesario** porque el tokenizer devuelve `[num_choices, seq_len]` y el modelo espera `[batch, num_choices, seq_len]`.
2. **El ejemplo simplifica a 2 opciones**. SWAG real tiene 4; el ejemplo del lab usa 2 por claridad. La cabeza generaliza a cualquier $K \ge 2$.
3. **El ejemplo no es de SWAG**, es un ejemplo casero de commonsense físico (cómo se come la pizza en Italia). Si quisieras fine-tunear en SWAG real, le pasarías el dataset al `Trainer` de HuggingFace.

### 9.6 Fine-tuning real en SWAG

El recipe canónico para fine-tunear `BertForMultipleChoice` en SWAG (siguiendo el `run_swag.py` del repo de HuggingFace):

```python
from datasets import load_dataset
from transformers import BertTokenizer, BertForMultipleChoice, Trainer, TrainingArguments

dataset = load_dataset("swag", "regular")  # 73K train, 20K val, 20K test
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def preprocess(example):
    # premisa = startphrase, 4 endings = ending0..ending3
    first = [[example["startphrase"]] * 4]
    second = [[example[f"ending{i}"]] for i in range(4)]
    encoding = tokenizer(first, second, truncation=True, padding="max_length", max_length=128)
    return {k: [v[i] for i in range(4)] for k, v in encoding.items()}

dataset = dataset.map(preprocess)

model = BertForMultipleChoice.from_pretrained("bert-base-uncased")
training_args = TrainingArguments(
    output_dir="./swag-bert",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)
trainer = Trainer(model=model, args=training_args, train_dataset=dataset["train"], eval_dataset=dataset["validation"])
trainer.train()
```

En una GPU H100 moderna, esto entrena en ~1 hora y alcanza ~82% accuracy con BERT-base — replicando aproximadamente el número reportado en el paper de BERT (81.6%).

---

## 10. Limitaciones del dataset

SWAG es un paper bien escrito que reconoce muchas de sus propias limitaciones (Sección 5.4). Listamos las que la literatura posterior ha confirmado:

### 10.1 AF contra LSTM es obsoleto

Como ya discutimos, AF con un ensemble estilístico LSTM-based no resiste BERT. Cualquier dataset construido con AF necesita ser revisado contra modelos más fuertes periódicamente. HellaSwag fue la corrección directa. Pero incluso HellaSwag fue saturado por GPT-4. La conclusión metodológica es que **AF tiene fecha de expiración** equivalente al ritmo de avance de los modelos.

### 10.2 Sesgo de dominio

El dataset hereda los sesgos de ActivityNet y LSMDC:
- Sobre-representación de actividades occidentales urbanas.
- Sub-representación de razonamiento social, emocional, abstracto.
- Sub-representación de idiomas no inglés (todo es inglés).
- Sesgo de género en LSMDC (películas hollywoodenses), aunque parcialmente mitigado por el reemplazo de nombres por "someone".

Sap et al. (2019) discutirían más adelante los sesgos demográficos de SWAG y otros datasets de commonsense de Choi-lab.

### 10.3 Solo cuatro opciones

La elección de $K = 4$ es arbitraria. Aumentar a $K = 8$ o $K = 16$ habría hecho el dataset más difícil mecánicamente (random baseline más bajo). HellaSwag mantiene $K = 4$. PIQA reduce a $K = 2$ (solo dos opciones, formato binario), lo que hace el dataset más fácil para humanos pero el random baseline sube a 50%.

### 10.4 Filtro estilístico, no semántico

AF en SWAG filtra atajos estilísticos (longitud, perplejidad, BoW). No filtra atajos semánticos profundos — por ejemplo, si las premisas tienden a mencionar "piano" y las continuaciones correctas siempre mencionan "keys" o "music", un modelo con conocimiento de mundo puede explotar esa co-ocurrencia. BERT capturó exactamente eso.

### 10.5 SWAG está hoy "resuelto"

En 2026, LLMs como GPT-4, Claude 4 y Gemini 2 alcanzan >95% en SWAG zero-shot. El dataset no es útil como benchmark contemporáneo. Su valor actual es:
- **Histórico**: marcar el momento en que commonsense reasoning entró al mainstream.
- **Pedagógico**: ejemplo canónico para enseñar Adversarial Filtering, multiple choice, y la cabeza `BertForMultipleChoice`.
- **Como dataset de fine-tuning ligero**: 73K ejemplos de razonamiento contextual sigue siendo útil para fine-tunear modelos pequeños.

### 10.6 Distinción ambigua entre situational y plausibility

El análisis de errores (Tabla 4) muestra que el 14.4% de los errores se debe a "plausibility" — el distractor es implausible regardless del contexto. Esto significa que parte del dataset puede resolverse sin la premisa, lo que contradice parcialmente la motivación original (eliminar baselines hipothesis-only). Esto es un subproducto inevitable de generar distractores con un LM débil.

---

## 11. Conexión con la clase 20 del Diplomado IA UC

La clase 20 trata la trilogía **ELMo → BERT → GPT/ChatGPT** y, en general, el paso de embeddings contextualizados a LLMs. SWAG conecta con la clase en tres niveles:

### 11.1 Como benchmark del momento (Camino 4)

La clase 20 incluye a BERT (Devlin 2018) y la cabeza `BertForMultipleChoice`. SWAG es el dataset **diseñado en colaboración** con el equipo de BERT — Devlin et al. citan a Zellers et al. y reportan SWAG en su Tabla 4. Comprender SWAG es comprender uno de los benchmarks canónicos sobre los que BERT se evaluó. Y, en sentido inverso, comprender BERT explica por qué SWAG colapsó a los 3 meses.

### 11.2 Como anticipación de benchmarks de LLMs modernos

El formato multiple choice + commonsense reasoning establecido por SWAG es exactamente el formato que adoptan MMLU, HellaSwag, ARC, TruthfulQA, GSM8K (con modificaciones), BIG-Bench, HumanEval (con modificaciones), MT-Bench. Casi todo el ecosistema de benchmarks de LLMs hoy hereda de SWAG en forma o en convención de evaluación. Mostrar SWAG en la clase 20 es mostrar el primer eslabón de esa genealogía.

### 11.3 Como ejemplo de "carrera adversarial" en NLP

La saturación rápida de SWAG por BERT, la respuesta con HellaSwag, la saturación de HellaSwag por GPT-3/4, y la siguiente generación de benchmarks (BIG-Bench Hard, GPQA, ARC-AGI) componen una historia que ilustra la dinámica de **moving target benchmarks** en NLP. Es una lección general sobre cómo evaluar capacidades emergentes en sistemas que mejoran exponencialmente. La clase 20, que cubre el período en que esta dinámica se hizo evidente, puede usar SWAG como caso de estudio.

---

## 12. Conexión con el lab 20

El lab 20 (Práctica de XLNet) tiene en su celda 14 un ejemplo de `XLNetForMultipleChoice` que es la implementación directa del paradigma SWAG en HuggingFace.

### 12.1 El ejemplo del lab

```python
from transformers import XLNetTokenizer, XLNetForMultipleChoice
import torch

tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
model = XLNetForMultipleChoice.from_pretrained("xlnet-base-cased")

prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
choice0 = "It is eaten with a fork and a knife."
choice1 = "It is eaten while held in the hand."

labels = torch.tensor(0).unsqueeze(0)  # choice0 es la correcta

encoding = tokenizer([[prompt, prompt], [choice0, choice1]],
                     return_tensors="pt", padding=True)
outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)
loss = outputs.loss
logits = outputs.logits  # [1, 2]
```

### 12.2 Análisis del shape

El ejemplo ilustra el patrón mecánico de `*ForMultipleChoice` que vimos en la sección 9:

- `encoding["input_ids"]` tiene shape `[num_choices=2, seq_len]`.
- `.unsqueeze(0)` añade dim de batch → `[batch=1, num_choices=2, seq_len]`.
- El modelo internamente aplana, pasa por el backbone, extrae el token de pooling, y devuelve logits de shape `[batch=1, num_choices=2]`.

Si el ejemplo fuera SWAG real, las shapes serían:
- `input_ids`: `[batch, num_choices=4, seq_len=128]`
- `logits`: `[batch, num_choices=4]`
- `labels`: `[batch]` con valores en `{0, 1, 2, 3}`.

### 12.3 Por qué el ejemplo del lab es simplificado

El ejemplo del lab tiene solo 2 opciones (no 4) por dos razones pedagógicas:

1. **Claridad**: con 2 opciones el ejemplo es más legible y el resultado es interpretable como binario.
2. **No necesita fine-tuning**: XLNet pre-entrenado ya tiene conocimiento de mundo suficiente para distinguir "pizza con tenedor" de "pizza con la mano" en un contexto formal italiano. El ejemplo es **zero-shot**.

Si el lab quisiera fine-tunear en SWAG real, el código sería esencialmente:

```python
from datasets import load_dataset
from transformers import XLNetTokenizer, XLNetForMultipleChoice, Trainer, TrainingArguments

dataset = load_dataset("swag", "regular")
tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
model = XLNetForMultipleChoice.from_pretrained("xlnet-base-cased")

def preprocess(example):
    first = [example["startphrase"]] * 4
    second = [example[f"ending{i}"]] for i in range(4)
    return tokenizer(first, second, truncation=True, padding="max_length", max_length=128)

dataset = dataset.map(preprocess, batched=False)
trainer = Trainer(model=model, args=TrainingArguments(...), train_dataset=dataset["train"], eval_dataset=dataset["validation"])
trainer.train()
```

Este es exactamente el recipe del `run_swag.py` de HuggingFace adaptado a XLNet. En la práctica, XLNet-base alcanza ~80% en SWAG, similar a BERT-base.

### 12.4 Por qué el lab eligió este ejemplo

Mostrar `XLNetForMultipleChoice` con el ejemplo italiano-pizza-tenedor cumple varios objetivos pedagógicos:

1. **Introducir la familia de cabezas `*ForMultipleChoice`** que aparecen en todos los backbones de HuggingFace (BERT, RoBERTa, ALBERT, XLNet, DeBERTa, etc.).
2. **Mostrar el patrón de `unsqueeze(0)`** que es necesario para casi todas las tareas multiple choice.
3. **Ilustrar zero-shot commonsense** sin entrar en la complejidad de fine-tuning.
4. **Conectar con SWAG** implícitamente — el alumno que se pregunta "¿en qué tarea real se usa esto?" tiene como respuesta directa SWAG y su sucesor HellaSwag.

El ejemplo italiano-pizza-tenedor además tiene una virtud no obvia: el conocimiento que requiere ("en Italia, la pizza en restaurantes se come con cubiertos") es **cultural específico**, no obvio para alguien que no haya estudiado etiqueta italiana. Esto demuestra que XLNet pre-entrenado tiene conocimiento de mundo no trivial, comparable al que SWAG mide.

---

## 13. Conclusiones para integrar al material del curso

SWAG es el paper más útil de la clase 20 para tres conversaciones distintas:

**Conversación 1 — Construcción de datasets en NLP**. SWAG es el ejemplo canónico de Adversarial Filtering. Es la cita estándar para explicar cómo se construye un benchmark "difícil" de manera escalable. Para alumnos que trabajen en construcción de datasets, leer SWAG es leer el manual de operaciones.

**Conversación 2 — Genealogía de benchmarks de LLMs**. SWAG es el primer benchmark de commonsense reasoning en formato multiple choice que escala. Toda la era de evaluación de LLMs (MMLU, HellaSwag, ARC, BIG-Bench, GPQA) hereda el formato de SWAG. Mostrar SWAG es mostrar la semilla del paradigma evaluativo moderno.

**Conversación 3 — La fragilidad de los benchmarks adversariales**. SWAG resistió 3 meses antes de ser saturado por BERT. Este caso de estudio es educativo sobre el ritmo de avance de NLP en 2018-2020 y sobre las limitaciones epistémicas de "construir un benchmark contra los modelos del momento". HellaSwag, WinoGrande y Adversarial NLI son las respuestas iterativas a este problema. La saturación de todos ellos por GPT-4 sugiere que el problema fundamental no es solvable con AF iterativo, solo postergable.

Sobre la implementación: cualquier alumno que use `BertForMultipleChoice`, `XLNetForMultipleChoice` o cualquier `*ForMultipleChoice` de HuggingFace está usando código que fue escrito **específicamente para SWAG** y luego generalizado. La cabeza no existiría sin el dataset. Esto cierra el círculo de por qué SWAG aparece en la clase 20 al lado de BERT.

---

## 14. Lectura recomendada complementaria

- **HellaSwag** (Zellers et al. 2019, ACL) — la versión post-BERT del mismo grupo. Lectura obligatoria para entender la evolución de AF.
- **Annotation Artifacts in NLI** (Gururangan et al. 2018, NAACL) — el paper que motivó AF. Muestra los sesgos en SNLI.
- **Hypothesis Only Baselines** (Poliak et al. 2018, StarSem) — complementario a Gururangan; documenta el problema en más datasets.
- **WinoGrande** (Sakaguchi et al. 2020, AAAI) — aplica AF con RoBERTa a Winograd Schema Challenge.
- **Adversarial NLI** (Nie et al. 2020, ACL) — variante de AF con human-in-the-loop.
- **The Curious Case of Neural Text Degeneration** (Holtzman et al. 2019, ICLR) — explica por qué SWAG usó greedy sampling y no beam search.
- **BERT** (Devlin et al. 2018, arXiv) — Sección 4.4 describe `BertForMultipleChoice` diseñada para SWAG.
- **XLNet** (Yang et al. 2019, NeurIPS) — el backbone que el lab 20 usa con `XLNetForMultipleChoice`.
- **PIQA** (Bisk et al. 2020, AAAI) — commonsense físico, versión binaria post-SWAG.
- **Social IQA** (Sap et al. 2019, EMNLP) — commonsense social, multiple choice tres opciones.
- **MMLU** (Hendrycks et al. 2021) — multiple choice general knowledge a escala LLM, hereda formato de SWAG.
- **TruthfulQA** (Lin et al. 2022) — multiple choice de honestidad, formato heredado.
