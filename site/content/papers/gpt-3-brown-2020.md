---
title: "GPT-3 (Language Models are Few-Shot Learners)"
weight: 295
math: true
---

{{< paper-card
    title="Language Models are Few-Shot Learners"
    authors="Brown, Mann, Ryder, Subbiah, Kaplan, Dhariwal, et al."
    year="2020"
    venue="NeurIPS 2020 (Best Paper)"
    pdf="/papers/gpt-3-brown-2020.pdf"
    arxiv="2005.14165" >}}
GPT-3 escala el decoder-only Transformer a **175B parametros** y demuestra que el **in-context learning** -- la capacidad de aprender una tarea nueva a partir de instrucciones o ejemplos provistos en el prompt, sin actualizar pesos -- emerge como propiedad del preentrenamiento masivo. Sobre 42 benchmarks zero/one/few-shot, GPT-3 se acerca o supera al SOTA fine-tuneado en varias familias de tareas (LAMBADA, TriviaQA, traduccion al ingles, PIQA, COPA, suma de dos digitos al 100%), genera articulos de noticias indistinguibles de la prosa humana (52% accuracy de detectores humanos, casi azar) y valida empiricamente las **scaling laws** de Kaplan et al. sobre dos ordenes de magnitud adicionales. Es el manifiesto que detono la era de los LLM frontier y pavimento el camino a ChatGPT.
{{< /paper-card >}}

---

## Contexto

A mediados de 2020 el campo del NLP convivia bajo el paradigma **pretrain-finetune** popularizado por BERT (Devlin 2018) y consolidado por RoBERTa, ALBERT, T5 y XLNet: preentrenar un Transformer sobre texto sin etiquetar y luego ajustarlo con miles a cientos de miles de ejemplos por tarea downstream. El paradigma funcionaba, pero arrastraba dos costos: (i) requerir un dataset etiquetado para cada nueva tarea; (ii) el riesgo de absorber correlaciones espurias de la distribucion de entrenamiento (McCoy 2019, Gururangan 2018).

Tres antecedentes inmediatos preparan el terreno para GPT-3:

- **GPT-2** (Radford 2019, 1.5B parametros): introduce la hipotesis "unsupervised multitask learning" y muestra zero-shot razonable, aunque todavia muy por debajo del SOTA fine-tuned (4% en Natural Questions, 55 F1 en CoQA).
- **T5** (Raffel 2019): confirma que escalar funciona en el formato text-to-text, pero sigue dependiendo de fine-tuning.
- **Scaling Laws** (Kaplan 2020): demuestra empiricamente que la perdida de validacion de un LM autoregresivo decae como una power law en funcion del computo $C$, del tamano del modelo $N$ y de la cantidad de datos $D$. Es la justificacion cuantitativa para invertir el orden de magnitud que separa GPT-2 (1.5B) de GPT-3 (175B).

OpenAI lee Kaplan et al. y se pregunta: que pasa si el zero-shot debil de GPT-2 era solo cuestion de escala? GPT-3 es la respuesta empirica masiva. El proyecto se ejecuta durante la primera mitad de 2020 sobre infraestructura V100 provista por Microsoft (la inversion de USD 1.000M de julio 2019 materializo el supercomputador Azure que hizo viable entrenar 175B parametros). El paper aparece en arXiv el 28 de mayo de 2020, semanas antes del lanzamiento del **GPT-3 API** (junio 2020), primera vez que OpenAI monetiza un LLM via endpoint. El paper es a la vez artefacto cientifico y "white paper" del primer producto comercial de LLM general-purpose.

El titulo -- *Language Models are Few-Shot Learners* -- es deliberadamente provocador: descarta el fine-tuning como mecanismo de adaptacion a tareas y reivindica el prompt como interfaz suficiente.

---

## Ideas principales

### 1. In-context learning: zero/one/few-shot

La afirmacion nuclear del paper: un modelo de lenguaje suficientemente grande, entrenado solo con next-token prediction sobre texto natural, adquiere durante el preentrenamiento la capacidad de **aprender nuevas tareas en el forward pass** -- sin actualizar pesos -- a partir de instrucciones o ejemplos provistos en el contexto.

Los autores formalizan esto como una estructura de **meta-learning** con dos loops anidados:

- **Outer loop**: la optimizacion por SGD durante el preentrenamiento. El modelo absorbe "un amplio conjunto de habilidades y patrones".
- **Inner loop**: el *in-context learning* propiamente tal. Ocurre en el forward pass sobre una sola secuencia. El prompt actua como especificacion de tarea; los ejemplos demostration actuan como "datos de entrenamiento" virtuales; la completacion es la prediccion.

El paper define un espectro de tres settings, todos sin gradient updates:

| Setting | Descripcion | K (ejemplos) |
|---|---|---|
| **Zero-Shot (0S)** | Solo descripcion de tarea en lenguaje natural. | 0 |
| **One-Shot (1S)** | Descripcion + 1 ejemplo. | 1 |
| **Few-Shot (FS)** | Descripcion + K ejemplos en el prompt. | 10-100 (limitado por $n_{ctx}=2048$) |

El **fine-tuning queda explicitamente fuera**. La motivacion es metodologica y filosofica: los humanos aprenden tareas de lenguaje desde una directiva breve o unos pocos ejemplos -- si queremos sistemas comparables a humanos, el few-shot es la metrica justa. Y operacionalmente, eliminar el fine-tuning elimina la dependencia de datasets etiquetados por tarea.

La distincion terminologica importa: lo que en GPT-2 se llamo "zero-shot transfer" aqui se rebautiza como **in-context learning sobre meta-learning**, precisamente para evitar confusion. "Zero-shot" significa cero ejemplos en el prompt; el modelo no aprende de cero, esta aprovechando la distribucion implicita de tareas internalizada durante el preentrenamiento.

### 2. La misma arquitectura, mucho mas grande (175B)

GPT-3 usa **exactamente la misma arquitectura que GPT-2**: decoder-only Transformer con pre-LayerNorm, BPE byte-level (vocab 50.257) e inicializacion modificada. La unica diferencia tecnica explicita es el uso de **patrones de atencion sparse alternados** estilo Sparse Transformer (Child 2019): capas con atencion densa se alternan con capas banded/local para reducir el costo cuadratico en secuencias largas. No hay novedad arquitectonica: es ingenieria para hacer factible el entrenamiento a esta escala.

Los autores entrenan **8 tamanos** que barren tres ordenes de magnitud para validar empiricamente la power law de Kaplan:

| Modelo | $n_{params}$ | $n_{layers}$ | $d_{model}$ | $n_{heads}$ | $d_{head}$ |
|---|---|---|---|---|---|
| GPT-3 Small | 125M | 12 | 768 | 12 | 64 |
| GPT-3 Medium | 350M | 24 | 1024 | 16 | 64 |
| GPT-3 Large | 760M | 24 | 1536 | 16 | 96 |
| GPT-3 XL | 1.3B | 24 | 2048 | 24 | 128 |
| GPT-3 2.7B | 2.7B | 32 | 2560 | 32 | 80 |
| GPT-3 6.7B | 6.7B | 32 | 4096 | 32 | 128 |
| GPT-3 13B | 13.0B | 40 | 5140 | 40 | 128 |
| **GPT-3 175B** | **175.0B** | **96** | **12288** | **96** | **128** |

Observaciones tecnicas:

- $d_{ff} = 4 \cdot d_{model}$ siempre (relacion estandar Transformer).
- Context window $n_{ctx} = 2048$ tokens para todos los modelos -- determina el techo del K en few-shot.
- El learning rate **decrece** con el tamano y el batch size **crece** -- exactamente la prescripcion de Kaplan et al. y McCandlish et al. (gradient noise scale).
- Paralelizacion mixta: tensor parallelism (dentro de cada matmul) + pipeline parallelism (entre capas), sobre cluster V100 de Microsoft.
- En el 175B, la matriz por cabeza $W_Q W_K^T$ tiene la misma dimensionalidad ($128 \times 128$) que en modelos chicos. La escala se gana en **ancho** y **profundidad**, no por cabeza.

### 3. Datos: Common Crawl + WebText2 + Books + Wikipedia

GPT-3 se entrena sobre una mezcla curada de cinco fuentes, con **sobremuestreo deliberado de las de mayor calidad**:

| Dataset | Tokens (B) | Peso en mix | Epocas a 300B tokens |
|---|---|---|---|
| Common Crawl (filtrado) | 410 | 60% | 0.44 |
| WebText2 | 19 | 22% | 2.9 |
| Books1 | 12 | 8% | 1.9 |
| Books2 | 55 | 8% | 0.43 |
| Wikipedia | 3 | 3% | 3.4 |

**Total entrenado**: ~300B tokens. Common Crawl, pese a ser ~13x mas grande que WebText2, recibe solo 60% del peso, y Wikipedia recibe 3% -- pero se pasa 3.4 veces durante el entrenamiento. La filosofia explicita: aceptar un poco de overfitting a cambio de mayor calidad de datos.

El **procesamiento de Common Crawl** ocupa el Apendice A: 41 shards mensuales de 2016 a 2019 (45 TB comprimidos, 570 GB tras filtrado). El pipeline tiene tres pasos: (i) filtrado por similitud a corpora de referencia (clasificador logistico binario WebText vs Common Crawl bruto, se conservan los que se parecen a WebText); (ii) deduplicacion fuzzy a nivel de documento (MinHash LSH); (iii) augmentacion con corpora de alta calidad (WebText2, Books1, Books2, Wikipedia).

Los autores **reconocen explicitamente un bug en el filtro de overlap**: una fraccion de overlap con test sets de benchmarks no se removio, y dado el costo de reentrenamiento (~USD 4.6M solo en computo) no se pudo rehacer. Toda la Seccion 4 esta dedicada a caracterizar y mitigar este sesgo a posteriori.

### 4. Validacion empirica de scaling laws

El resultado mas fundamental del paper es la Figura 3.1: la **perdida de validacion cross-entropy** sigue una power law lisa en funcion del computo a lo largo de **dos ordenes de magnitud adicionales** mas alla de lo que Kaplan habia medido:

$$ L(C) \approx 2.57 \cdot C^{-0.048} $$

donde $C$ se mide en PetaFLOP/s-days. Las desviaciones de la curva son pequenas y consistentes a lo largo de los 8 tamanos.

El computo total del 175B alcanza:

$$ \text{FLOPs(GPT-3 175B)} \approx 3.14 \times 10^{23} \text{ FLOPs} \approx 3640 \text{ PF/s-days} $$

Para contexto: BERT-Base ~0.96 PF/s-days, BERT-Large ~3.5, RoBERTa-Large ~49, T5-11B ~410, y GPT-3 175B ~3640. Es **~10.000x mas computo que GPT-2 1.5B**. El costo estimado por terceros: **~USD 4.6M solo en GPU**, sin contar salarios, infraestructura, R&D previa, ni iteracion. GPT-3 fija la barra para la "era de los frontier labs".

La curva agregada de accuracy (Figura 1.3) sobre los 42 benchmarks confirma el patron: few-shot sube de ~25% (125M) a ~58% (175B); one-shot va de ~25 a ~51%; zero-shot va de ~25 a ~42%. **El gap entre few-shot y zero-shot crece con la escala** -- interpretado por los autores como evidencia de que modelos mas grandes son **meta-learners mas proficientes**, no solo mejores ejecutores.

### 5. Capacidades emergentes con escala

Mas alla de la suavidad de la curva de loss, varias capacidades aparecen **discontinuamente** al escalar. El caso paradigmatico es la aritmetica: la accuracy en suma de 3 digitos few-shot permanece cercana a cero hasta 13B parametros, y se dispara entre 13B y 175B. ANLI Round 3 (NLI adversarial) tiene un patron similar: modelos por debajo de 175B se quedan en chance (~33%), el 175B salta a 40%. Esta nocion de **capacidad emergente** se volveria un area de investigacion completa post-GPT-3 (Wei et al. 2022).

### 6. Indistinguibilidad de prosa humana

El experimento de la Seccion 3.9.4: 80 evaluadores estadounidenses distinguen articulos de noticias de ~200 palabras generados por GPT-3 175B de articulos reales con accuracy media **52% -- apenas sobre el azar del 50%**. Con articulos largos (~500 palabras) la accuracy humana se mantiene en 52%. La Figura 3.13 muestra una power law clara: la accuracy de deteccion humana **decrece con el tamano del modelo**, desde 76% (Small) hasta 52% (175B). Un "control model" -- GPT-3 Small mal calibrado deliberadamente -- se detecta al 86%, confirmando que la dificultad creciente es atribuible al modelo y no a un sesgo en la prueba.

---

## Resultados experimentales

El paper evalua GPT-3 en 42 benchmarks accuracy-denominated organizados en 9 categorias. Resumen: zero-shot sube suave con tamano; few-shot sube mas rapido (la pendiente crece con el modelo). En varias familias de tareas GPT-3 few-shot iguala o supera al SOTA fine-tuneado.

### Language modeling, cloze, completion

| Benchmark | Setting | SOTA previo | GPT-3 |
|---|---|---|---|
| PTB (perplejidad zero-shot) | 0S | 35.8 | **20.5** |
| LAMBADA (accuracy) | FS | 68.0 | **86.4** |
| LAMBADA (perplejidad) | FS | 8.63 | **1.92** |
| HellaSwag | FS | 85.6 (ft) | 79.3 |
| StoryCloze | FS | 91.8 (ft) | 87.7 |

**LAMBADA** es revelador: requiere predecir la ultima palabra de oraciones que dependen de parrafos completos. Antes se postulaba que esta tarea estaba en "diminishing returns" y que escalar no era el camino. GPT-3 few-shot mejora **+18 puntos** sobre el SOTA. El truco que activa la mejora: enmarcar la tarea como cloze con few-shot examples del estilo *"Alice was friends with Bob. Alice went to visit her friend ____. -> Bob"*, lo que le indica al LM que la respuesta es una sola palabra (limitacion clasica de los LM en LAMBADA: asignan probabilidad a continuaciones validas mas largas).

### Closed-book Question Answering

| Dataset | SOTA fine-tuned | GPT-3 0S | GPT-3 1S | GPT-3 FS |
|---|---|---|---|---|
| TriviaQA | 68.0 (RAG open-domain) | 64.3 | 68.0 | **71.2** |
| WebQuestions | 45.5 (RAG) | 14.4 | 25.3 | 41.5 |
| Natural Questions | 44.5 (RAG) | 14.6 | 23.0 | 29.9 |

En **TriviaQA** GPT-3 few-shot supera incluso al modelo open-domain fine-tuneado RAG, que tiene acceso a un retriever sobre 15.3B documentos. El conocimiento factual esta **en los pesos**.

### Traduccion (WMT)

| Setting | En->Fr | Fr->En | En->De | De->En | En->Ro | Ro->En |
|---|---|---|---|---|---|---|
| SOTA supervisado | **45.6** | 35.0 | **41.2** | 40.2 | **38.5** | **39.9** |
| Unsup. NMT (mBART) | -- | -- | 29.8 | 34.0 | 35.0 | 30.5 |
| GPT-3 zero-shot | 25.2 | 21.2 | 24.6 | 27.2 | 14.1 | 19.9 |
| GPT-3 few-shot | 32.6 | **39.2** | 29.7 | **40.6** | 21.0 | **39.5** |

**Traducir al ingles** funciona mucho mejor que traducir desde ingles. Refleja que el corpus es 93% ingles por word count -- el LM tiene mejor modelo de la distribucion target inglesa que de cualquier otro idioma. En Fr->En y De->En, few-shot iguala o supera el SOTA supervisado y bate a unsupervised NMT por ~5 BLEU.

### Common sense y lectura

PIQA: **82.8% few-shot, supera SOTA fine-tuned** (79.4). Winograd: 88.6 few-shot (cerca del 90.1 fine-tuned). ARC-Challenge: 51.5%, lejos del 78 de UnifiedQA. CoQA: 85.0 (3 puntos por debajo del baseline humano). DROP, QuAC y RACE muestran las debilidades de GPT-3: requieren razonamiento simbolico/numerico, modelado de dialogo estructurado o comparacion entre fragmentos largos.

### SuperGLUE

Few-shot K=32 sobre el suite estandar:

| | BoolQ | CB | COPA | RTE | WiC | WSC | MultiRC | ReCoRD | Avg |
|---|---|---|---|---|---|---|---|---|---|
| Fine-tuned SOTA | 91.0 | 96.9 | 94.8 | 92.5 | **76.1** | 93.8 | 88.2 | 92.5 | **89.0** |
| BERT-Large fine-tuned | 77.4 | 83.6 | 70.6 | 71.7 | 69.6 | 64.6 | 24.1 | 71.3 | 69.0 |
| GPT-3 Few-Shot | 76.4 | 75.6 | **92.0** | 69.0 | 49.4 | 80.1 | 30.5 | 90.2 | 71.8 |

GPT-3 con 32 ejemplos en contexto **supera a un BERT-Large fine-tuneado en 4 de 8 tareas**, y necesita menos de 8 ejemplos por tarea para igualar a BERT-Large en promedio. Punto bajo: **WiC al 49.4% (azar)** -- GPT-3 falla sistematicamente en tareas que requieren *comparar dos oraciones*.

### Aritmetica y tareas sinteticas

Suma/resta de 2 a 5 digitos y multiplicacion de 2 digitos, todos few-shot, GPT-3 175B:

| Operacion | 0S | 1S | FS |
|---|---|---|---|
| 2D suma | 76.9 | 99.6 | **100.0** |
| 3D suma | 34.2 | 65.5 | 80.4 |
| 4D suma | 4.0 | 14.0 | 25.5 |
| 5D suma | 0.7 | 3.5 | 9.3 |
| 2D multiplicacion | 19.8 | 27.4 | 29.2 |
| 1D operaciones compuestas | 9.8 | 14.3 | 21.3 |

**100% en suma de 2 digitos few-shot**. La verificacion anti-memorizacion es clave: los autores buscaron los 2.000 problemas de suma de 3 digitos en el corpus de entrenamiento y encontraron solo 17 matches (0.8%). Errores tipicos como "no llevar el 1" sugieren que GPT-3 **computa, no recupera**. La capacidad aritmetica emerge solo a partir de 13B y se dispara en 175B.

**SAT analogies**: 65.2% few-shot -- mejor que el promedio de aplicantes universitarios estadounidenses pre-2005 (57%). **Word scrambling** (descifrar anagramas, reinsertar caracteres, palabras al reves): hasta 67% few-shot en algunas variantes, dificil porque BPE no representa caracteres individuales y el modelo debe "abrir" sus tokens. **Novel word use**: tras un solo demonstration, GPT-3 usa una palabra inventada ("Gigamuru") en una oracion semantica y gramaticalmente plausible.

---

## Limitaciones reconocidas

Los autores hacen una autocritica notablemente honesta en la Seccion 5:

- **Sintesis de texto largo**: perdida de coherencia, contradicciones, non-sequiturs, repeticion semantica al pasar de parrafos a paginas.
- **Common-sense physics**: "If I put cheese into the fridge, will it melt?" -- GPT-3 falla.
- **Tareas de comparacion entre fragmentos**: WiC (azar), ANLI, QuAC, RACE. Hipotesis estructural: la arquitectura **decoder-only autoregresiva** es suboptima para tareas que se benefician de bidireccionalidad estilo BERT. Los autores admiten: *"a large bidirectional model would be stronger at fine-tuning than GPT-3"*.
- **Objetivo de preentrenamiento agnostico**: cada token pesa igual; falta la nocion de "que importa predecir". Util -> sistemas dirigidos a metas (precursor explicito de RLHF).
- **Sample efficiency baja**: GPT-3 ve durante el preentrenamiento mucho mas texto del que un humano ve en toda su vida.
- **Ambiguedad del few-shot**: no esta claro si el modelo aprende "de cero" en el inner loop o si solo **reconoce** una tarea ya internalizada. La literatura posterior (Olsson et al. 2022 sobre induction heads; Akyurek et al. 2023 sobre ICL como gradient descent implicito) exploraria exactamente esto.
- **Costo de inferencia**: 175B es caro de servir; sugieren destilacion como direccion futura.
- **No interpretable**, **no actualizable** (el modelo "sabe" lo que sabia en octubre 2019), **no multimodal**, **sesgos heredados del corpus**.

La Seccion 4 sobre **contaminacion de benchmarks** es uno de los aportes metodologicos durables del paper: caracteriza a posteriori el impacto del bug de overlap, construye "clean subsets" excluyendo overlaps por dataset, y reporta shift clean vs full. La mayoria de los 42 benchmarks muestra shift cercano a cero; excepciones marcadas con asterisco; 4 benchmarks de language modeling (Wikipedia tasks y Children's Book Test) directamente removidos del reporte. Esto marca el inicio de la conciencia sistematica sobre data contamination -- tema que dominaria la evaluacion posterior de LLMs (HumanEval contamination, MMLU contamination, etc.).

La **Seccion 6 sobre Broader Impacts** es la primera vez que un paper de NLP dedica espacio extenso y tecnicamente concreto a impacto social. Cubre tres ejes:

- **Misuse**: misinformation, spam, phishing, ghostwriting academico, social engineering. Monitorearon foros de bajo/medio skill y conversaron con analistas de APTs.
- **Fairness, bias, representation**: analisis empirico de sesgos de genero (83% de 388 ocupaciones probadas sesgan masculinas; el prompt "competent" *amplifica* el sesgo), raza (sentiment con Senti WordNet: "Asian" rankea 1st en 3/7 modelos, "Black" ultimo en 5/7), religion (asociaciones estereotipadas reproducidas del corpus).
- **Energy usage**: argumentan amortizacion sobre la vida util del modelo y la posibilidad de destilacion.

---

## Por que importa hoy

GPT-3 cambio la economia y la cultura del NLP. Lo verificable ex post:

1. **Detono la era LLM**. Tras GPT-3 vinieron Jurassic-1 (AI21, 178B, 2021), Megatron-Turing NLG 530B (Microsoft-NVIDIA, 2022), Gopher 280B (DeepMind, 2021), Chinchilla 70B (DeepMind, 2022 -- la correccion compute-optima), PaLM 540B (Google, 2022), OPT 175B (Meta, open weights, 2022), BLOOM 176B (BigScience, open weights, 2022) y LLaMA-1 (Meta, 2023). Cada uno responde, directa o indirectamente, a GPT-3.

2. **GPT-3 API** (junio 2020): primera vez que un LLM frontier se vendio como producto, con modelo de negocio basado en tokens por request. Inspiro el ecosistema "GPT-3 powered" que prefigura el de hoy.

3. **Pavimento ChatGPT**. GPT-3 base es dificil de "controlar": para volverlo util como asistente conversacional, OpenAI desarrollo **InstructGPT** (Ouyang 2022) sumando SFT + RLHF sobre GPT-3. ChatGPT (noviembre 2022) es GPT-3.5 instruct-tuned. Sin GPT-3 base no hay ChatGPT.

4. **Cambio la economia del NLP**: de "fine-tunear un BERT por tarea" a "disenar un prompt". El termino *prompt engineering* como practica profesional nace aqui. Frameworks como LangChain, LlamaIndex y Semantic Kernel asumen la interfaz de prompts inaugurada por GPT-3.

5. **Validacion de scaling laws** que justifico la siguiente decada de inversion en computo (TPU v4, H100, supercomputadores dedicados).

6. **Conciencia sobre data contamination**: la Seccion 4 sento precedente metodologico.

7. **Inicio del debate alineamiento**: la Seccion 6 abrio la linea de trabajo de safety/alignment que llevaria a Constitutional AI, RLHF como practica estandar y sistema cards.

Aspectos donde GPT-3 quedo tecnicamente desplazado:

- **Compute-optimo**: Chinchilla (Hoffmann 2022) mostro que GPT-3 estaba *undertrained* (300B tokens para 175B parametros; lo optimo era ~3.5T tokens). LLaMA-1 (7B-65B sobre 1-1.4T tokens) demostro que modelos mas pequenos bien entrenados podian competir.
- **Bidireccionalidad**: la admision del paper se confirmo. Para tareas comparativas, encoder-decoders como T5 y modelos bidireccionales fine-tuned siguen siendo competitivos.
- **Long context**: $n_{ctx} = 2048$ se quedo muy corto. Las generaciones siguientes (GPT-4 Turbo, Claude 100K+, Gemini 1M+) operan en otras escalas.
- **Multimodalidad**: GPT-3 es solo texto. GPT-4V, Gemini y Claude 3+ inauguran la era multimodal.

GPT-3 no es ya state-of-the-art en nada. Pero **define el lenguaje y los reflejos** con los que pensamos LLMs media decada despues: terminologia (zero/one/few-shot, meta-learning, outer/inner loop), grilla de evaluacion (cualquier benchmark de LLM en 2026 reporta few-shot, no fine-tuning per-task), reflejos de ingenieria (prompt-first, fine-tuning como ultimo recurso), conciencia de contaminacion, y la asuncion de que un solo modelo puede hacer todo si es suficientemente capaz. Leerlo hoy sigue siendo obligatorio.

---

## Notas y enlaces

- Paper extenso (~75 paginas con anexos), pero la lectura util se concentra en Secciones 1-3 (tesis, setup, resultados), Seccion 4 (contaminacion), Seccion 5 (limitaciones) y Seccion 6 (broader impacts). Apendice A documenta el procesamiento de Common Crawl; Apendice B los hiperparametros de los 8 tamanos.
- **Figura clave**: 3.1 (power law $L(C) \approx 2.57 \cdot C^{-0.048}$) y 1.3 (curvas zero/one/few-shot agregadas).
- **Tabla clave**: 2.1 (los 8 tamanos) y 3.8 (SuperGLUE).
- 31 autores; primer firmante Tom Brown. NeurIPS 2020 Best Paper Award.
- Codigo no liberado (a diferencia de GPT-2). El acceso al modelo fue via API comercial desde junio 2020. Reimplementaciones open-weights llegaron despues con OPT-175B (Meta) y BLOOM-176B (BigScience).
- Timeline: GPT-1 (2018) -> BERT (2018) -> GPT-2 (2019) -> T5 (2019) -> Scaling Laws (enero 2020) -> **GPT-3 (mayo 2020)** -> GPT-3 API (junio 2020) -> InstructGPT (marzo 2022) -> ChatGPT (noviembre 2022).

Ver fundamentos: [In-context learning](/fundamentos/in-context-learning) - [GPT family](/fundamentos/gpt-family) - [Foundation models](/fundamentos/foundation-models) - [Clase 20](/clases/clase-20).
