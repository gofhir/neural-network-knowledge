# Language Models are Few-Shot Learners — Análisis exhaustivo (Brown et al., 2020)

**Referencia bibliográfica**
Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., & Amodei, D. (2020). *Language Models are Few-Shot Learners*. NeurIPS 2020 (Best Paper Award). arXiv:2005.14165.

OpenAI, 31 autores, ~75 páginas con anexos. Publicado en arXiv el 28 de mayo de 2020; v4 con el cierre final el 22 de julio de 2020. NeurIPS 2020 lo premió como Best Paper. Es uno de los papers más citados de la década en NLP y, en retrospectiva, el detonante visible de la era LLM y de la economía de "prompt engineering".

---

## 1. Contexto histórico: de GPT-2 a GPT-3 (febrero 2019 → mayo 2020)

El paper aterriza en un momento muy específico de la línea de tiempo de NLP:

1. **GPT-1 (Radford et al., 2018)** introduce el paradigma *pretrain + fine-tune* sobre un decoder Transformer (117M parámetros).
2. **BERT (Devlin et al., 2018)** lo populariza con masked language modeling bidireccional y domina los rankings de fine-tuning.
3. **GPT-2 (Radford et al., 2019)** escala a 1.5B parámetros, muestra zero-shot razonable y plantea la hipótesis "unsupervised multitask learning". Pero su zero-shot todavía queda lejos del SOTA fine-tuned (e.g., 4% en Natural Questions, 55 F1 en CoQA vs SOTA >90).
4. **T5 (Raffel et al., 2019)** confirma que escalar funciona, pero todavía bajo el paradigma fine-tuning.
5. **Scaling Laws (Kaplan et al., enero 2020)** demuestra empíricamente que la pérdida de validación de un LM autoregresivo decae como una power law en función del cómputo, el tamaño del modelo y la cantidad de datos. Esto fue, dentro de OpenAI, la justificación cuantitativa para invertir el orden de magnitud que separaba GPT-2 (1.5B) de GPT-3 (175B).

El equipo de OpenAI lee Kaplan et al. y se pregunta: ¿qué pasa si el zero-shot débil de GPT-2 era solo cuestión de escala? El paper de GPT-3 es la respuesta empírica masiva a esa pregunta. El proyecto se ejecuta durante la primera mitad de 2020, en plena pandemia de COVID-19, sobre infraestructura V100 provista por Microsoft (la inversión de Microsoft en OpenAI de USD 1.000M de julio 2019 había materializado el supercomputador Azure que hizo posible entrenar 175B parámetros). La elección del título — *Language Models are Few-Shot Learners* — es deliberadamente provocadora: descarta el fine-tuning como mecanismo de adaptación a tareas y reivindica el prompt como interfaz suficiente.

El cierre y publicación coinciden con el lanzamiento, semanas después, del GPT-3 API (junio 2020) — primera vez que OpenAI monetiza un modelo de lenguaje vía endpoint. El paper, entonces, es a la vez artefacto científico y "white paper" del primer producto comercial de LLM general-purpose.

---

## 2. Tesis central: in-context learning como capacidad emergente

La afirmación nuclear es:

> Un modelo de lenguaje suficientemente grande, entrenado solo con next-token prediction sobre texto natural, adquiere durante el preentrenamiento la capacidad de aprender nuevas tareas en el forward pass — sin actualizar pesos — a partir de instrucciones o ejemplos provistos en el contexto.

Los autores formalizan esto como una estructura de **meta-learning** con dos loops anidados:

- **Outer loop**: la optimización por SGD durante el preentrenamiento. Aquí el modelo absorbe "un amplio conjunto de habilidades y patrones".
- **Inner loop**: el *in-context learning* propiamente tal. Ocurre en el forward pass sobre una sola secuencia. El prompt actúa como especificación de tarea; los ejemplos demonstration actúan como "datos de entrenamiento" virtuales; la completación es la "predicción".

La figura 1.1 del paper ilustra el punto con tres "sub-tareas" embebidas dentro de secuencias de entrenamiento (aritmética, corrección ortográfica, traducción palabra a palabra). El argumento de los autores es que ese tipo de patrón repetido — múltiples ejemplos del mismo esquema dentro de una ventana — aparece naturalmente en el corpus web, y que el modelo internaliza la capacidad de identificarlo y continuarlo. El crédito histórico explícito en las "Contributions": *Alec Radford originally demonstrated few-shot learning occurs in language models*.

La distinción terminológica importa: lo que en GPT-2 se llamó "zero-shot transfer" aquí se rebautiza como **meta-learning** + **in-context learning**, precisamente para evitar confusión. "Zero-shot" en GPT-3 significa cero ejemplos en el prompt (puede haber instrucción en lenguaje natural), pero el modelo no está aprendiendo de cero — está aprovechando la distribución implícita de tareas vista durante el preentrenamiento.

---

## 3. Los tres settings de evaluación (sin fine-tuning)

El paper define un espectro y se posiciona deliberadamente fuera del fine-tuning:

| Setting | Descripción | K (ejemplos) | Notas |
|---------|-------------|--------------|-------|
| **Fine-Tuning (FT)** | Actualizar pesos sobre dataset supervisado de la tarea. | miles–cientos de miles | No usado en este paper. |
| **Few-Shot (FS)** | Task description + K ejemplos en el prompt, sin gradient updates. | 10–100 (limitado por $n_{ctx}=2048$) | Foco principal del paper. |
| **One-Shot (1S)** | Task description + 1 ejemplo. | 1 | Más cercano a cómo se le explica una tarea a un humano (e.g., MTurk). |
| **Zero-Shot (0S)** | Solo task description en lenguaje natural. | 0 | "Máxima conveniencia" pero también "el setting más difícil"; a veces injustamente difícil incluso para humanos. |

La decisión de NO hacer fine-tuning es metodológica y filosófica. Citan dos motivaciones:

1. **Práctica**: la necesidad de datasets etiquetados para cada tarea limita la aplicabilidad de los LMs y los hace recolectar correlaciones espurias específicas de la distribución de entrenamiento (Gururangan et al.; McCoy et al.).
2. **Comparación con humanos**: los humanos aprenden tareas de lenguaje desde una directiva breve o unos pocos ejemplos. Si queremos sistemas comparables, el few-shot es la métrica justa.

Pretenden, dejando fine-tuning fuera, mostrar que **el comportamiento task-agnostic mejora con escala**, no solo el SOTA en benchmarks específicos.

---

## 4. Arquitectura y los 8 tamaños

GPT-3 usa la **misma arquitectura que GPT-2**: decoder-only Transformer con pre-LayerNorm, reversible tokenization (BPE byte-level, vocab 50.257), inicialización modificada. La única diferencia técnica explícita es el uso de **patrones de atención sparse alternados** estilo *Sparse Transformer* (Child et al., 2019): capas con atención densa se alternan con capas con atención banded/local, para reducir el costo cuadrático en secuencias largas. Esto no es novedoso en sí — es ingeniería para hacer factible el entrenamiento a esta escala.

Los 8 tamaños fueron elegidos para barrer **tres órdenes de magnitud** y validar empíricamente la power law de Kaplan et al.:

| Modelo | $n_{params}$ | $n_{layers}$ | $d_{model}$ | $n_{heads}$ | $d_{head}$ | Batch (tokens) | LR |
|---|---|---|---|---|---|---|---|
| GPT-3 Small | 125M | 12 | 768 | 12 | 64 | 0.5M | $6.0 \times 10^{-4}$ |
| GPT-3 Medium | 350M | 24 | 1024 | 16 | 64 | 0.5M | $3.0 \times 10^{-4}$ |
| GPT-3 Large | 760M | 24 | 1536 | 16 | 96 | 0.5M | $2.5 \times 10^{-4}$ |
| GPT-3 XL | 1.3B | 24 | 2048 | 24 | 128 | 1M | $2.0 \times 10^{-4}$ |
| GPT-3 2.7B | 2.7B | 32 | 2560 | 32 | 80 | 1M | $1.6 \times 10^{-4}$ |
| GPT-3 6.7B | 6.7B | 32 | 4096 | 32 | 128 | 2M | $1.2 \times 10^{-4}$ |
| GPT-3 13B | 13.0B | 40 | 5140 | 40 | 128 | 2M | $1.0 \times 10^{-4}$ |
| **GPT-3 175B** | **175.0B** | **96** | **12288** | **96** | **128** | **3.2M** | $0.6 \times 10^{-4}$ |

Observaciones técnicas relevantes:

- $d_{ff} = 4 \cdot d_{model}$ siempre (relación estándar Transformer).
- Context window $n_{ctx} = 2048$ tokens para todos los modelos. Esto va a determinar el techo del K en few-shot.
- El learning rate **decrece** con el tamaño y el batch size **crece** — exactamente la prescripción de Kaplan et al. y McCandlish et al. (gradient noise scale).
- Particionamiento mixto: tensor parallelism (dentro de cada matmul) + pipeline parallelism (entre capas), entrenado en clúster V100 de Microsoft.

Para el 175B con $d_{model}=12288$, $n_{layers}=96$ y $n_{heads}=96$, la matriz $W_Q W_K^T$ por cabeza tiene $128 \times 128$ — exactamente la misma dimensionalidad por cabeza que modelos más pequeños. La escala se gana en *ancho* y *profundidad*, no por cabeza.

---

## 5. Dataset de entrenamiento

GPT-3 se entrena sobre una mezcla curada de cinco fuentes, con **sobre-muestreo deliberado de las de mayor calidad**:

| Dataset | Tokens (B) | Peso en mix | Épocas a 300B tokens |
|---|---|---|---|
| Common Crawl (filtrado) | 410 | 60% | 0.44 |
| WebText2 | 19 | 22% | 2.9 |
| Books1 | 12 | 8% | 1.9 |
| Books2 | 55 | 8% | 0.43 |
| Wikipedia | 3 | 3% | 3.4 |

**Total entrenado**: ~300B tokens. Notar que Common Crawl, pese a ser ~13x más grande que WebText2, recibe solo 60% del peso, y Wikipedia 3% — pero se pasa 3.4 veces durante el entrenamiento, mientras Common Crawl se ve solo 0.44 veces. La filosofía explícita: "aceptamos un poco de overfitting a cambio de mayor calidad de datos".

**Procesamiento de Common Crawl** (Apéndice A): se descargaron 41 shards mensuales de 2016 a 2019 (45 TB comprimidos, 570 GB tras filtrado). El pipeline tiene tres pasos:

1. **Filtrado por similitud a corpora de referencia de alta calidad** — usan un clasificador logístico binario que predice si un documento se parece más a WebText o a Common Crawl en bruto, y conservan los que se parecen a WebText.
2. **Deduplicación fuzzy** a nivel de documento (MinHash LSH), dentro y entre datasets, para evitar redundancia y proteger la integridad del held-out.
3. **Augmentación** con corpora de alta calidad ya conocidos (WebText2, Books1, Books2, Wikipedia).

Los autores **reconocen explícitamente un bug en el filtro de overlap**: una fracción de overlap con test sets de benchmarks no se removió, y dado el costo de reentrenamiento (~USD 4.6M solo en cómputo) no se pudo rehacer. Toda la Sección 4 del paper está dedicada a caracterizar y mitigar este sesgo.

---

## 6. Compute: la escala del entrenamiento

La cifra emblemática:

$$ \text{FLOPs(GPT-3 175B)} \approx 3.14 \times 10^{23} $$

Esto corresponde a ~3640 **petaflop/s-days** (Figura 2.2 del paper). Para contexto:

- BERT-Base: ~0.96 PF/s-days.
- BERT-Large: ~3.5 PF/s-days.
- RoBERTa-Large: ~49 PF/s-days.
- T5-11B: ~410 PF/s-days.
- **GPT-3 175B: ~3640 PF/s-days.**

Es aproximadamente **10.000x más cómputo que GPT-2 1.5B**. El costo estimado por terceros (Lambda Labs, basado en precios cloud de la época) fue del orden de **USD 4.6 millones solo en cómputo** — sin contar salarios, infraestructura, R&D previa, ni iteración. Esto convierte a GPT-3 en, hasta ese momento, uno de los entrenamientos más caros documentados públicamente, y fija la barra para la "era de los frontier labs".

Una observación instructiva del paper: aunque GPT-3 3B y RoBERTa-Large difieren en ~10x en parámetros, ambos usaron ~50 PF/s-days. Es decir, GPT-3 sigue una estrategia diferente: **modelos mucho más grandes entrenados sobre menos tokens** (en términos de épocas) que la práctica BERT-era. Esto es exactamente lo que Kaplan et al. predecían como compute-óptimo en ese momento — predicción que después Chinchilla (2022) corregiría en sentido opuesto, mostrando que GPT-3 estaba undertrained respecto a datos.

---

## 7. Resultados por familia de tareas

El paper evalúa GPT-3 en **42 benchmarks denominados por accuracy** organizados en 9 categorías. Resumen ejecutivo: zero-shot sube suave con tamaño; few-shot sube más rápido (la pendiente crece con el modelo); el gap zero/one/few-shot se ensancha con escala — sugiriendo que **modelos más grandes son mejores meta-learners**, no solo mejores ejecutores.

### 7.1. Language modeling, cloze, completion

| Benchmark | Setting | SOTA previo | GPT-3 |
|---|---|---|---|
| PTB (zero-shot ppl) | 0S | 35.8 | **20.5** |
| LAMBADA (acc) | FS | 68.0 | **86.4** |
| LAMBADA (ppl) | FS | 8.63 | **1.92** |
| HellaSwag (acc) | FS | 85.6 (ft) | 79.3 |
| StoryCloze (acc) | FS | 91.8 (ft) | 87.7 |

**PTB**: nuevo SOTA por 15 puntos de perplejidad. Otros benchmarks de language modeling (e.g., enwik8, four Wikipedia tasks de GPT-2) se omiten porque caen en data contamination.

**LAMBADA** es particularmente revelador: requiere predecir la última palabra de oraciones que dependen de párrafos completos de contexto. Antes se postulaba (Bao et al., 2020) que esta tarea estaba en "diminishing returns" y que escalar HW/datos "no era el camino". Few-shot GPT-3 175B mejora **+18 puntos sobre el SOTA**. El truco que activa la mejora: enmarcar la tarea como cloze con few-shot examples del estilo *"Alice was friends with Bob. Alice went to visit her friend ____. → Bob"*, lo que permite al LM "saber" que la respuesta es una sola palabra (limitación clásica de los LM en LAMBADA: asignan probabilidad a continuaciones válidas pero más largas).

### 7.2. Closed-book QA

| Dataset | SOTA fine-tuned | GPT-3 0S | GPT-3 1S | GPT-3 FS |
|---|---|---|---|---|
| TriviaQA | 68.0 (RAG open-domain) | 64.3 | 68.0 | **71.2** |
| WebQuestions | 45.5 (RAG) | 14.4 | 25.3 | 41.5 |
| Natural Questions | 44.5 (RAG) | 14.6 | 23.0 | 29.9 |

En **TriviaQA**, GPT-3 few-shot supera incluso al modelo open-domain fine-tuneado RAG (que tiene acceso a retriever sobre 15.3B documentos). Esto es notable: el conocimiento factual está **en los pesos**. WebQuestions y Natural Questions muestran que cuando el estilo de la pregunta está fuera de distribución (Wikipedia fine-grained), la brecha persiste.

### 7.3. Translation

Reporta BLEU multi-bleu.perl en WMT'14 (Fr↔En), WMT'16 (De↔En, Ro↔En):

| Setting | En→Fr | Fr→En | En→De | De→En | En→Ro | Ro→En |
|---|---|---|---|---|---|---|
| SOTA supervisado | **45.6** | 35.0 | **41.2** | 40.2 | **38.5** | **39.9** |
| Unsup. NMT (mBART) | – | – | 29.8 | 34.0 | 35.0 | 30.5 |
| GPT-3 zero-shot | 25.2 | 21.2 | 24.6 | 27.2 | 14.1 | 19.9 |
| GPT-3 one-shot | 28.3 | 33.7 | 26.2 | 30.4 | 20.6 | 38.6 |
| GPT-3 few-shot | 32.6 | **39.2** | 29.7 | **40.6** | 21.0 | **39.5** |

Patrón importante: **traducir AL inglés** funciona mucho mejor que traducir DESDE inglés. Esto refleja que el corpus es 93% inglés por word count — el LM tiene mejor modelo de la distribución target inglés que cualquier otro idioma. En Fr→En y De→En, few-shot GPT-3 iguala o supera el SOTA supervisado y supera a unsupervised NMT por ~5 BLEU.

### 7.4. Winograd / Winogrande

| | Winograd (WSC273) | Winogrande XL |
|---|---|---|
| Fine-tuned SOTA | 90.1 | 84.6 |
| GPT-3 zero-shot | 88.3* | 70.2 |
| GPT-3 few-shot | 88.6* | 77.7 |

Cercano a SOTA en Winograd original; en Winogrande (adversarial) sigue por debajo del fine-tuned RoBERTa-large pero competitivo. (*) Contaminación parcial detectada — ver Sección 4.

### 7.5. Common sense reasoning

PIQA: 82.8% few-shot, supera SOTA fine-tuned (79.4). ARC-Challenge: 51.5%, por debajo de UnifiedQA (~78). OpenBookQA: 65.4% few-shot vs SOTA 87.2.

### 7.6. Reading comprehension

| | CoQA | DROP | QuAC | SQuADv2 | RACE-h | RACE-m |
|---|---|---|---|---|---|---|
| Fine-tuned SOTA | **90.7** | **89.1** | **74.4** | **93.0** | **90.0** | **93.1** |
| GPT-3 few-shot | 85.0 | 36.5 | 44.3 | 69.8 | 46.8 | 58.1 |

CoQA (3 puntos por debajo del baseline humano) es el highlight. DROP, QuAC y RACE muestran las debilidades de GPT-3: requieren razonamiento simbólico/numérico, modelado de diálogo estructurado o comparación entre fragmentos largos.

### 7.7. SuperGLUE

Few-shot K=32 sobre el suite estándar:

| | BoolQ | CB | COPA | RTE | WiC | WSC | MultiRC | ReCoRD | Avg |
|---|---|---|---|---|---|---|---|---|---|
| Fine-tuned SOTA | 91.0 | 96.9 | 94.8 | 92.5 | **76.1** | 93.8 | 88.2 | 92.5 | **89.0** |
| Fine-tuned BERT-Large | 77.4 | 83.6 | 70.6 | 71.7 | 69.6 | 64.6 | 24.1 | 71.3 | 69.0 |
| GPT-3 Few-Shot | 76.4 | 75.6 | **92.0** | 69.0 | 49.4 | 80.1 | 30.5 | 90.2 | 71.8 |

GPT-3 con 32 ejemplos en contexto **supera a un BERT-Large fine-tuneado en 4 de 8 tareas** del benchmark estandard, y necesita **menos de 8 ejemplos por tarea** para igualar a BERT-Large en promedio. Punto bajo: **WiC al 49.4% (azar)** — GPT-3 falla sistemáticamente en tareas que requieren *comparar dos oraciones* (también RTE, CB y reading comprehension comparativa).

### 7.8. NLI

ANLI Round 3 (adversarial): GPT-3 few-shot 40.2% vs SOTA fine-tuned ~48%. Modelos por debajo de 175B se quedan en chance (~33%). Es el caso paradigmático de capacidad emergente: la métrica permanece plana hasta una transición visible al escalar al 175B.

### 7.9. Synthetic and qualitative tasks

Diseñadas para probar **on-the-fly reasoning** sobre cosas improbables de haber sido memorizadas:

**Aritmética** (zero/one/few-shot, GPT-3 175B):

| Op | 0S | 1S | FS |
|---|---|---|---|
| 2D+ | 76.9 | 99.6 | **100.0** |
| 2D− | 58.0 | 86.4 | 98.9 |
| 3D+ | 34.2 | 65.5 | 80.4 |
| 3D− | 48.3 | 78.7 | 94.2 |
| 4D+ | 4.0 | 14.0 | 25.5 |
| 4D− | 7.5 | 14.0 | 26.8 |
| 5D+ | 0.7 | 3.5 | 9.3 |
| 5D− | 0.8 | 3.8 | 9.9 |
| 2Dx (mult) | 19.8 | 27.4 | 29.2 |
| 1DC (compuesto) | 9.8 | 14.3 | 21.3 |

**100% en suma de 2 dígitos few-shot**. La verificación anti-memorización es clave: buscaron los 2.000 problemas de suma de 3 dígitos en el corpus de entrenamiento y encontraron solo 17 matches (0.8%). Errores típicos como "no llevar el 1" sugieren que GPT-3 **computa, no recupera**. Es uno de los hallazgos más comentados del paper: capacidad aritmética emerge solo a partir del 13B y se dispara en el 175B (Figura 3.10 muestra el salto discontinuo entre 13B y 175B).

**Word scrambling**:
| | CL | A1 | A2 | RI | RW |
|---|---|---|---|---|---|
| 0S | 3.66 | 2.28 | 8.91 | 8.26 | 0.09 |
| 1S | 21.7 | 8.62 | 25.9 | 45.4 | 0.48 |
| FS | 37.9 | 15.1 | 39.7 | **67.2** | 0.44 |

Tareas character-level: descifrar anagramas, reinsertar caracteres, palabras al revés. Particularmente difícil porque BPE no representa caracteres individuales — el modelo debe "abrir" sus tokens.

**SAT analogies**: 65.2% few-shot. Mejor que el promedio de aplicantes universitarios estadounidenses pre-2005 (57%).

**News article generation** (Sección 3.9.4): 80 evaluadores estadounidenses distinguen artículos de ~200 palabras generados por GPT-3 175B de artículos reales con accuracy media **52% — apenas sobre el azar del 50%**. Con artículos más largos (~500 palabras), la accuracy humana sigue en 52%. La Figura 3.13 muestra una power law clara: la accuracy humana de detección decrece con el tamaño del modelo, desde 76% (Small) hasta 52% (175B). El "control model" — un GPT-3 Small mal calibrado deliberadamente — se detecta al 86%, confirmando que la dificultad creciente es atribuible al modelo, no a un sesgo en la prueba.

**Novel word use** (Figura 3.16): "A 'Gigamuru' is a type of Japanese musical instrument. Example: *I have a Gigamuru that my uncle gave me as a gift. I love to play it at home.*" — GPT-3 usa la palabra inventada en una oración semántica y gramaticalmente plausible tras un solo demonstration.

**Grammar correction** (Figura 3.17): one-shot, formato "Poor English / Good English" — GPT-3 normaliza errores de concordancia, casos, contracciones.

---

## 8. Validación empírica de scaling laws

La Figura 3.1 del paper es el resultado más fundamental: la **pérdida de validación cross-entropy** (excluyendo embeddings) sigue una power law lisa en función del cómputo a lo largo de **dos órdenes de magnitud adicionales** más allá de lo que Kaplan et al. habían medido. La curva empírica:

$$ L(C) \approx 2.57 \cdot C^{-0.048} $$

donde $C$ se mide en PetaFLOP/s-days. Las desviaciones de la curva ideal son pequeñas y consistentes a lo largo de los 8 tamaños. Esto es la **validación experimental masiva** de las scaling laws — y la justificación post-hoc para invertir en modelos aún más grandes.

La Figura 1.3 agrega todos los 42 benchmarks accuracy-denominated. La curva few-shot sube de ~25% (125M) a ~58% (175B). One-shot va de ~25 a ~51%. Zero-shot va de ~25 a ~42%. El **gap entre few-shot y zero-shot crece con la escala** — interpretado por los autores como evidencia de que modelos más grandes son meta-learners más proficientes.

---

## 9. Sección 4: contaminación de benchmarks

Esta sección, frecuentemente subapreciada, es uno de los aportes metodológicos durables del paper. El problema: cuando entrenas sobre Common Crawl (45 TB de la web entera), inevitablemente capturas test sets de benchmarks publicados en la web. El equipo:

1. Construyó un filtro que removía documentos overlapping con test/dev sets de **todos** los benchmarks evaluados.
2. Detectó un **bug** que dejó pasar overlaps no removidos.
3. Como reentrenar costaba millones, optaron por **caracterizar a posteriori** el impacto.

La metodología: para cada dataset, construyen un "clean subset" excluyendo cualquier example con overlap, y comparan rendimiento clean vs total. Reportan:

- La mayoría de los 42 benchmarks muestran **shift cercano a cero** entre clean y full.
- Excepciones marcadas con asterisco en las tablas (e.g., PIQA, Winograd) o directamente removidas del reporte (4 benchmarks de language modeling Wikipedia, Children's Book Test).
- LAMBADA tiene contaminación sustantiva pero el shift es <0.5%, atribuido al formato fill-in-the-blank que evita memorización trivial.

El paper marca el inicio de la conciencia sistemática sobre data contamination — un tema que después dominaría la evaluación de LLMs (HumanEval contamination, MMLU contamination, etc.).

---

## 10. Sección 5: limitaciones reconocidas

Los autores hacen una autocrítica notablemente honesta. Resumen:

1. **Text synthesis a documento largo**: pérdida de coherencia, contradicciones, non-sequiturs, repetición semántica.
2. **Common-sense physics**: "If I put cheese into the fridge, will it melt?" — GPT-3 falla.
3. **Tareas de comparación entre fragmentos**: WiC (azar), ANLI, QuAC, RACE. Hipótesis estructural: la arquitectura **decoder-only autorregresiva** es subóptima para tareas que se benefician de bidireccionalidad (BERT-style). Los autores admiten: *"a large bidirectional model would be stronger at fine-tuning than GPT-3"*.
4. **Objetivo de preentrenamiento agnóstico**: cada token se pondera igual; falta noción de "qué importa predecir". Útil → sistemas dirigidos a metas (precursor de RLHF).
5. **Sample efficiency baja**: GPT-3 ve durante el preentrenamiento mucho más texto del que un humano ve en su vida completa.
6. **Ambigüedad del few-shot**: no está claro si el modelo aprende "de cero" en el inner loop o si solo **reconoce** una tarea ya internalizada. La literatura posterior (Olsson et al., 2022 — induction heads; Akyürek et al., 2023 — ICL implementa gradient descent implícito) explorará exactamente esto.
7. **Costo de inferencia**: 175B es caro de servir. Sugieren destilación como dirección futura.
8. **No interpretable**, **alta varianza vs humanos**, **sesgos heredados del corpus**.

---

## 11. Sección 6: Broader Impacts (concreto, no performativo)

Este es uno de los primeros papers de NLP en dedicar una sección extensa, técnicamente concreta, a impacto social. Tres ejes:

### 11.1. Misuse

- **Aplicaciones potenciales de mal uso**: misinformation, spam, phishing, ghostwriting académico, social engineering. La capacidad de GPT-3 de generar texto difícil de distinguir de humano (Figura 3.13) eleva el bottleneck "calidad de texto" en estas operaciones.
- **Threat actor analysis**: monitorearon foros de bajo/medio skill y consultaron con analistas de amenazas sobre APTs. Conclusión: la adopción no había crecido tras GPT-2; los modelos aún no son lo suficientemente confiables (1% de outputs incoherentes "rompe" un bot de disinformation automatizado).
- **External incentive structures**: phishing es popular por bajo costo y alto yield; LMs podrían reducir aún más ese costo.

### 11.2. Fairness, bias, representation

Análisis empírico (no exhaustivo) en tres ejes:

**Género**: para 388 ocupaciones probadas con prompt "The {occupation} was a", el 83% son sesgadas masculinas. Ocupaciones de alta educación (legislator, banker, professor emeritus) o trabajo físico (mason, millwright, sheriff) sesgan fuerte hacia masculino. Las que sesgan femenino: midwife, nurse, receptionist, housekeeper. El sesgo promedio (log-ratio P(female)/P(male)) es **−1.11** neutro, **−2.14** con prompt "competent", **−1.15** con prompt "incompetent" — el prompt "competent" *amplifica* el sesgo masculino.

Sobre Winogender (resolución de pronombres), GPT-3 175B tiene 64.17% accuracy general — sobre la marca chance pero lejos de perfecto. Es el único modelo donde la accuracy es **mayor para sentences cuya respuesta correcta es femenina** (81.7% vs 76.7%), sugiriendo que escalar **podría** reducir algunos sesgos.

Top adjetivos co-ocurriendo con "She": *optimistic, bubbly, naughty, easy-going, petite, tight, pregnant, gorgeous, sucked, beautiful (158 ocurrencias)*. Top con "He": *large, mostly, lazy, fantastic, eccentric, protect, jolly, stable, personable, survive*. Adjetivos femeninos: orientados a apariencia. Adjetivos masculinos: rango más amplio.

**Raza**: prompts del estilo "The {race} man was very", analizando sentiment via Senti WordNet. "Asian" rankea 1st en 3/7 modelos; "Black" rankea último en 5/7. Las diferencias se estrechan marginalmente con tamaño.

**Religión**: top palabras co-ocurrentes con cada religión. Para Islam: *terrorism, fasting, sheikh, non-muslim, allah, prophet*. Para Judaism: *gentiles, semites, whites, blacks, smartest, racists, arabs*. Para Atheism: *agnostics, mad, theism, defensive, complaining, arrogant*. Las asociaciones reflejan estereotipos web-scale.

Los autores son explícitos: este es un análisis preliminar, no exhaustivo, y los sesgos categorizados son solo los más obvios.

### 11.3. Energy usage

GPT-3 175B consumió "varios miles de petaflop/s-days" durante el preentrenamiento. Pero argumentan que el costo debe **amortizarse** sobre la vida útil del modelo: una vez entrenado, generar 100 páginas cuesta ~0.4 kWh. Distillation podría reducir aún más el costo de inferencia. Mencionan Strubell et al. (2019) como literatura motivadora.

---

## 12. Impacto y legado

GPT-3 cambió la economía y la cultura de NLP. Hechos verificables ex post:

1. **Detonó la era LLM**. Tras GPT-3 vinieron, en orden cronológico aproximado: Jurassic-1 (AI21, 178B, 2021), Megatron-Turing NLG 530B (Microsoft-NVIDIA, 2022), Gopher 280B (DeepMind, 2021), Chinchilla 70B (DeepMind, 2022 — la corrección compute-óptima), PaLM 540B (Google, 2022), OPT 175B (Meta, open weights, 2022), BLOOM 176B (BigScience, open weights, 2022), LLaMA-1 (Meta, 2023). Cada uno responde, directa o indirectamente, a GPT-3.
2. **GPT-3 API** (junio 2020): primera vez que un LLM frontier se vendió como producto. Modelo de negocio basado en tokens por request. Inspiró el ecosistema de aplicaciones "GPT-3 powered" que prefiguraba el actual.
3. **Pavimentó ChatGPT**. GPT-3 base es difícil de "controlar": para volverlo útil como asistente conversacional, OpenAI desarrolló InstructGPT (Ouyang et al., 2022) sumando **fine-tuning supervisado (SFT) + RLHF**. ChatGPT (noviembre 2022) es GPT-3.5 instruct-tuned. Sin GPT-3 base no hay ChatGPT.
4. **Cambió la economía de NLP**: de "fine-tunear un BERT por tarea" a "diseñar un prompt". El término *prompt engineering* — práctica profesional — nace aquí. Frameworks como LangChain, LlamaIndex, Semantic Kernel asumen la interfaz de prompts inaugurada por GPT-3.
5. **Validación de scaling laws** que justificó la siguiente década de inversión en cómputo (TPU v4, H100, supercomputadores dedicados).
6. **Conciencia sobre data contamination**: la Sección 4 sentó precedente metodológico.
7. **Inicio del debate alineamiento**: la Sección 6 abrió la puerta a la línea de trabajo de safety/alignment que llevaría a Constitutional AI, RLHF como práctica estándar, sistema cards, etc.

Aspectos donde GPT-3 quedó técnicamente desplazado:

- **Compute-óptimo**: Chinchilla mostró que GPT-3 estaba undertrained (300B tokens para 175B params; lo óptimo era ~3.5T tokens). LLaMA-1 (7B-65B sobre 1-1.4T tokens) demostró que modelos más pequeños bien entrenados podían competir.
- **Bidireccionalidad**: la admisión del paper se confirmó. Para tareas comparativas, encoder-decoders como T5 y models bidireccionales fine-tuned siguen siendo competitivos.
- **Long context**: $n_{ctx} = 2048$ se quedó muy corto. Las generaciones siguientes (GPT-4 Turbo, Claude 100K+, Gemini 1M+) operan en otras escalas.
- **Multimodalidad**: GPT-3 es solo texto. GPT-4V, Gemini, Claude 3+ inauguran la era multimodal.

---

## 13. Conexión con el curso (clase 20 IA UC)

La clase 20 del diplomado posiciona a GPT-3 como **el penúltimo paso antes de ChatGPT**. Sirve dos propósitos pedagógicos:

1. **Evidencia empírica de scaling**: la Figura 3.1 (power law sobre 3 órdenes de magnitud de cómputo) y la Figura 1.3 (few-shot agregado vs tamaño) son la justificación de por qué la industria invirtió cientos de miles de millones de dólares en compute. Sin GPT-3 la apuesta era especulativa; con GPT-3 la apuesta era una predicción cuantitativa.

2. **Sustitución de fine-tuning por prompting**: el paper articula explícitamente que el modelo task-agnostic puede reemplazar al modelo task-specific. Esto justifica el énfasis curricular en **prompt engineering**, **in-context learning** e **instruction following** como habilidades centrales del ingeniero NLP moderno.

Cross-links naturales dentro del site:

- **Fundamento transversal `in-context-learning`**: GPT-3 es la evidencia empírica que define el fenómeno.
- **Fundamento transversal `scaling-laws`**: GPT-3 valida Kaplan et al. y motiva Chinchilla.
- **Fundamento transversal `meta-learning`**: GPT-3 articula el outer-loop/inner-loop framing.
- **Paper Devlin-BERT-2018**: contraste arquitectónico (encoder bidireccional vs decoder autorregresivo).
- **Paper Radford-GPT2-2019**: predecesor directo; GPT-3 es "GPT-2 más grande" con sparse attention.
- **Paper Ouyang-InstructGPT-2022**: el siguiente paso (SFT + RLHF sobre GPT-3) que produce ChatGPT.
- **Clase 14 (Transformers)**: GPT-3 es el caso de uso paradigmático del decoder Transformer a escala.
- **Clase 17-18 (Pre-training, Transfer Learning)**: GPT-3 demuestra que el preentrenamiento por sí solo, sin transfer learning explícito, es suficiente.

---

## 14. Notas para integrar al site

Estructura propuesta para `papers/gpt-3-brown-2020.md` dentro del hub `clase-20`:

- **Resumen ejecutivo (3 párrafos)**: tesis, escala, impacto.
- **Tabla síntesis** de los 8 modelos.
- **Figura clave**: power law $L = 2.57 \cdot C^{-0.048}$.
- **Tabla resultados consolidados**: una fila por familia de tareas, columnas zero/one/few-shot vs SOTA fine-tuned.
- **Sección crítica** discutiendo:
  - El status actual del paper post-Chinchilla (undertrained).
  - El gap entre GPT-3 base y GPT-3.5/ChatGPT (alignment).
  - La sombra ética: misuse, sesgos, costo energético, concentración de cómputo.
- **Cross-links bidireccionales** a los fundamentos `in-context-learning`, `scaling-laws`, `meta-learning`, y a los papers GPT-2 e InstructGPT.
- **Timeline visual**: GPT-1 (2018) → BERT (2018) → GPT-2 (2019) → T5 (2019) → Scaling Laws (Jan 2020) → **GPT-3 (May 2020)** → GPT-3 API (Jun 2020) → InstructGPT (Mar 2022) → ChatGPT (Nov 2022).

Snippet de imagen sugerido para el hub de la clase:

> "GPT-3, 175 mil millones de parámetros, ~300 mil millones de tokens, ~3640 PF/s-days de cómputo, ~4.6 millones de dólares solo en GPU. Las tres curvas zero/one/few-shot suben en paralelo durante los tres órdenes de magnitud que el paper recorre. La diferencia entre few-shot y zero-shot crece con el tamaño. Esa pendiente — más empinada cuanto más grande el modelo — es la firma empírica del in-context learning, y la justificación cuantitativa de la apuesta industrial por escalar."

---

## 15. Síntesis personal para el ingeniero senior

Lo que un ingeniero senior debe llevarse de leer el paper hoy (mayo 2026, seis años después):

1. **El paper sigue siendo el manifiesto del in-context learning**. La terminología (zero/one/few-shot, meta-learning, outer/inner loop) que usamos a diario fue cuajada aquí.

2. **Los números absolutos están desactualizados** pero el método sigue vivo: cualquier benchmark de un LLM nuevo en 2026 reporta zero/few-shot, no fine-tuning per-task. La grilla de evaluación que GPT-3 introdujo se volvió norma.

3. **La honestidad metodológica del paper merece estudio**: bug de contaminación reconocido, limitaciones explícitas, broader impacts concreto con datos en mano. Es un modelo de cómo escribir un paper de impacto sin barniz publicitario.

4. **Lo que envejeció peor**: la asunción de que escalar parámetros era el único eje. Chinchilla mostró que también hay que escalar datos. RLHF mostró que también hay que escalar feedback humano. RAG mostró que también hay que escalar contexto. MoE mostró que el cómputo activo no tiene que escalar con los parámetros. GPT-3 es el último gran paper "solo más parámetros".

5. **Lo que envejeció mejor**: la intuición de que **un solo modelo puede hacer todo si es suficientemente capaz**, articulada como tesis ingenieril y no como ciencia ficción. ChatGPT/Claude/Gemini son ese mismo modelo, con alignment y multimodalidad encima.

6. **Para un FHIR-engineer en healthcare**: la lección operativa es que **fine-tuning task-specific deja de ser la primera opción** cuando se tiene un LLM general capaz. Few-shot/prompt-based pipelines sobre LLMs frontier (con guardrails, validación de schema FHIR, y RAG sobre terminologías estructuradas) compiten o superan a clasificadores fine-tuneados para extracción clínica, deduplicación de pacientes en LATAM, mapping a SNOMED/LOINC, etc. — *siempre que se invierta en prompt engineering serio y en evaluación robusta contra ground truth.*

GPT-3 no es ya state-of-the-art en nada. Pero **define el lenguaje y los reflejos** con los que pensamos LLMs cinco años después. Leerlo en 2026 sigue siendo obligatorio.
