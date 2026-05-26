# Análisis interno — Yang et al. (2019) "XLNet: Generalized Autoregressive Pretraining for Language Understanding"

> Documento complementario al material público del site (`papers/xlnet-yang-2019.md`, `fundamentos/embeddings-contextualizados.md`). Aquí se profundiza en lo que el site solo menciona: derivación formal del objetivo de Permutation Language Modeling, justificación matemática del two-stream attention, integración no trivial con Transformer-XL, anatomía de `XLNetForQuestionAnswering` (relevante para el lab 20), ablations de la Sección 3.4 y crítica posterior a la luz de RoBERTa y la era LLM.

- **Paper**: Yang, Dai, Yang, Carbonell, Salakhutdinov, Le. *XLNet: Generalized Autoregressive Pretraining for Language Understanding*. arXiv:1906.08237v2 (2 Jan 2020). **NeurIPS 2019** (Vancouver). Reconocido como uno de los papers más citados de la conferencia y referente obligado en la transición entre la era BERT y la era de los LLMs.
- **Versiones**: v1 (Jun 2019, arXiv) — release inicial con XLNet-Large entrenado en BooksCorpus + Wikipedia + Giga5 + ClueWeb. v2 (Jan 2020, NeurIPS) — agrega comparación frente a RoBERTa (publicado en julio 2019, tres semanas después que XLNet) y reescribe la Sección 3.2 para separar el efecto "más datos" del efecto "mejor objetivo".
- **PDF local**: [`Yang-XLNet-2019.pdf`](Yang-XLNet-2019.pdf)
- **Código y checkpoints**: [`github.com/zihangdai/xlnet`](https://github.com/zihangdai/xlnet). XLNet-Base (110M) y XLNet-Large (340M) liberados bajo Apache 2.0. Implementación oficial en TensorFlow; el port HuggingFace `transformers` (clases `XLNetModel`, `XLNetForQuestionAnswering`, `XLNetForMultipleChoice`, `XLNetTokenizer`) es el que el alumno usará en el lab 20.

---

## 1. Contexto histórico: el sprint post-BERT de 2019

Cuando BERT se publicó en octubre de 2018 dejó dos problemas técnicos abiertos que rondaron toda la comunidad NLP durante el primer semestre de 2019:

1. **El símbolo `[MASK]` no existe en datos reales.** Durante el pre-training BERT reemplaza el 15% de los tokens por `[MASK]`, pero al hacer fine-tuning ese símbolo nunca aparece en el input. Esto produce un *pretrain-finetune discrepancy*: el modelo aprende a depender de un símbolo que jamás verá en producción. La regla 80/10/10 (sustituir 80% por `[MASK]`, 10% por un token aleatorio, 10% mantener intacto) lo mitiga parcialmente pero no lo elimina.

2. **Las posiciones enmascaradas son tratadas como independientes.** El loss de BERT factoriza $p(\bar{x} \mid \hat{x}) \approx \prod_{t : m_t=1} p(x_t \mid \hat{x})$ asumiendo independencia condicional entre los tokens enmascarados. Si en una oración como "[MASK] [MASK] is a city" se enmascaran simultáneamente "New" y "York", BERT predice cada uno por separado sin modelar la dependencia entre ellos. La cadena de productos del autoregressive language modeling no sufre este problema, porque cada predicción $p(x_t \mid x_{<t})$ ve todos los tokens previos en orden.

Antes de XLNet, la comunidad ya había detectado ambos problemas pero los abordaba con parches incrementales:

| Mes | Modelo | Idea | Limitación que dejaba abierta |
|---|---|---|---|
| Oct 2018 | **BERT** (Devlin et al.) | MLM + NSP bidireccional, fine-tuning end-to-end | `[MASK]` discrepancy, independencia entre targets |
| Ene 2019 | **Transformer-XL** (Dai, Yang et al.) | Memoria recurrente entre segmentos + relative positional encoding | No bidireccional, no pre-training masivo |
| Feb 2019 | **GPT-2** (Radford et al.) | Decoder unidireccional escalado a 1.5B params, zero-shot | Unidireccional, sin fine-tuning estándar |
| May 2019 | **MASS / UniLM** (Microsoft) | Unifica AE y AR para seq2seq con atención mascarada por bloques | Limitado a setups encoder-decoder |
| **Jun 2019** | **XLNet** (Yang et al.) | Permutation LM + two-stream attention + Transformer-XL | Costo computacional, complejidad de implementación |
| Jul 2019 | **RoBERTa** (Liu et al.) | BERT sin NSP, más datos (160GB), batches grandes (8K), dynamic masking | Solo entrena más; misma arquitectura BERT |

La aparición de XLNet en junio de 2019 fue un evento muy comentado: por primera vez desde BERT, un paper afirmaba haber **superado a BERT-Large** en las 20 tareas evaluadas, no por un truco de ingeniería sino por una idea nueva de fondo. La narrativa "permutation language modeling supera al masked language modeling" fue contagiosa. Tres semanas después, sin embargo, RoBERTa publicaba que **simplemente entrenando BERT mejor** (más datos, batches grandes, sin NSP, dynamic masking, más pasos) se podía alcanzar resultados comparables a XLNet. Eso obligó a Yang et al. a publicar la v2 del paper con la Sección 3.2 "Fair Comparison with BERT" donde aislan el efecto del objetivo PLM controlando por datos.

Los autores comparten genealogía interesante: **Zhilin Yang** y **Zihang Dai** (primeros autores, contribución equivalente) son ambos doctorandos de CMU bajo Ruslan Salakhutdinov, y son también primeros autores de **Transformer-XL** (Dai et al. 2019). XLNet es, en gran parte, "Transformer-XL aplicado a pre-training con un objetivo nuevo que evita la trampa del masking". **Quoc V. Le**, el sexto autor, es senior researcher en Google Brain y co-autor de los papers fundacionales de seq2seq y AutoML.

La pregunta de fondo que el paper se hace es: *¿se puede tener bidireccionalidad sin pagar el costo de la asunción de independencia entre targets y sin el mismatch `[MASK]`?* La respuesta es elegante: **sí, si en vez de mascar enseñas al modelo a predecir tokens en todos los órdenes posibles de la secuencia**. Si el modelo ve $T!$ ordenamientos distintos durante el entrenamiento (aunque sea estocásticamente), cada posición termina condicionándose sobre todas las demás en expectativa, lo que recupera bidireccionalidad sin necesidad de un símbolo artificial.

Esta idea no es totalmente original. Germain et al. (2015) la propusieron en **MADE** (Masked Autoencoder for Distribution Estimation) y Uria et al. (2016) en **NADE** (Neural Autoregressive Distribution Estimation) — ambas son arquitecturas para estimación de densidad en datos discretos. La contribución de XLNet es **escalar esta idea a Transformers grandes con fines de pre-training**, lo que requiere resolver un problema técnico no trivial: el target-aware reparameterization (Sección 3 de este documento).

---

## 2. Contribuciones centrales

XLNet propone tres ideas técnicas que se sustentan mutuamente. Quitar cualquiera de las tres rompe el sistema:

### 2.1 Permutation Language Modeling (PLM)

Sea $x = [x_1, \ldots, x_T]$ una secuencia de tokens y $\mathcal{Z}_T$ el conjunto de todas las $T!$ permutaciones de los índices $[1, 2, \ldots, T]$. Para una permutación $z \in \mathcal{Z}_T$, denotamos $z_t$ al $t$-ésimo elemento y $z_{<t} = (z_1, \ldots, z_{t-1})$ a los primeros $t-1$ elementos.

El objetivo de Permutation Language Modeling (paper, ecuación 3) es:

$$\max_{\theta} \quad \mathbb{E}_{z \sim \mathcal{Z}_T}\left[\sum_{t=1}^{T} \log p_\theta(x_{z_t} \mid x_{z_{<t}})\right]$$

Es decir: muestrear una permutación $z$, factorizar la likelihood según ese orden, y entrenar como si fuera un AR LM. Como los parámetros $\theta$ se comparten entre todas las permutaciones, en expectativa cada token $x_t$ ha visto a todos los demás $x_i$ ($i \ne t$) como contexto.

**Punto crítico que el paper enfatiza** (Sección 2.2, "Remark on Permutation"): la permutación es sobre el **orden de factorización**, no sobre el orden de la secuencia. La secuencia se mantiene en su orden natural; lo que cambia es qué tokens se "ocultan" en cada paso de predicción y en qué orden se revelan. Esta diferencia es esencial porque durante el fine-tuning el modelo verá texto en orden natural. Implementacionalmente, la permutación se logra **manipulando la máscara de atención**, no permutando los token embeddings.

Comparación cuantitativa con el objetivo de BERT (Sección 2.6), tomando como ejemplo la oración "[New, York, is, a, city]" donde tanto BERT como XLNet eligen [New, York] como targets:

$$\mathcal{L}_{\text{BERT}} = \log p(\text{New} \mid \text{is a city}) + \log p(\text{York} \mid \text{is a city})$$

$$\mathcal{L}_{\text{XLNet}} = \log p(\text{New} \mid \text{is a city}) + \log p(\text{York} \mid \textbf{New}, \text{is a city})$$

Para el target "York", XLNet condiciona en "New", capturando la dependencia bigramica. BERT no puede hacerlo porque ambos están enmascarados simultáneamente y se predicen independientemente. El paper argumenta (Apéndice A.5.1) que para todo par target-contexto $(x, \mathcal{U})$, BERT cubre la dependencia solo si $\mathcal{U} \subseteq \mathcal{N}$ (no-targets), mientras XLNet la cubre si $\mathcal{U} \subseteq \mathcal{N} \cup \mathcal{T}_{<x}$ (no-targets más targets previos en el orden de factorización). Esto da estrictamente más cobertura de dependencias.

**Partial prediction**: predecir todos los $T$ tokens en cada orden es muy lento (convergencia muy lenta en experimentos preliminares). La solución del paper (Sección 2.3, ecuación 5) es predecir solo los **últimos $1/K$ tokens** del orden de factorización:

$$\max_{\theta} \quad \mathbb{E}_{z \sim \mathcal{Z}_T}\left[\sum_{t=c+1}^{|z|} \log p_\theta(x_{z_t} \mid x_{z_{<t}})\right] \quad \text{con} \quad |z| / (|z| - c) \approx K$$

Para XLNet-Large se usa $K=6$, es decir se predice aproximadamente el último 16% de cada permutación (notar la relación con el 15% de masking de BERT — no es coincidencia, ambos buscan un equilibrio entre señal de entrenamiento y costo computacional). Los tokens "tempranos" en el orden (que tendrían poco contexto) no se usan como targets; solo sirven como contexto.

### 2.2 Two-Stream Self-Attention: el problema técnico que hizo difícil PLM

La idea de PLM es simple. Implementarla con un Transformer estándar no funciona. El problema (Apéndice A.1, "A Concrete Example of How Standard LM Parameterization Fails"):

Consideremos dos permutaciones $z^{(1)}$ y $z^{(2)}$ tales que $z^{(1)}_{<t} = z^{(2)}_{<t}$ pero $z^{(1)}_t = i \ne j = z^{(2)}_t$. Si parametrizamos la próxima distribución como $p(X_{z_t} = x \mid x_{z_{<t}}) \propto \exp(e(x)^\top h_\theta(x_{z_{<t}}))$, entonces:

$$p(X_i = x \mid x_{z_{<t}}) = p(X_j = x \mid x_{z_{<t}}) = \frac{\exp(e(x)^\top h(x_{z_{<t}}))}{\sum_{x'} \exp(e(x')^\top h(x_{z_{<t}}))}$$

Dos posiciones distintas $i \ne j$ comparten **exactamente la misma predicción**, lo cual es absurdo: la ground-truth en posición $i$ es generalmente distinta a la ground-truth en posición $j$. El representation $h_\theta(x_{z_{<t}})$ no sabe **dónde** está prediciendo, solo qué tokens están en el contexto.

La solución es **reparametrizar** la distribución para que dependa explícitamente de la posición target $z_t$ (paper, ecuación 4):

$$p_\theta(X_{z_t} = x \mid x_{z_{<t}}) = \frac{\exp(e(x)^\top g_\theta(x_{z_{<t}}, z_t))}{\sum_{x'} \exp(e(x')^\top g_\theta(x_{z_{<t}}, z_t))}$$

donde $g_\theta(x_{z_{<t}}, z_t)$ es un nuevo tipo de representation que **toma la posición target como input adicional**.

Ahora aparece el problema operativo: ¿cómo computamos $g_\theta(x_{z_{<t}}, z_t)$ con un Transformer? Hay dos requerimientos contradictorios:

1. Para predecir $x_{z_t}$, la representación $g$ debe usar la posición $z_t$ pero **no** el contenido $x_{z_t}$ (de lo contrario, el modelo aprende identidad).
2. Para predecir tokens posteriores $x_{z_j}$ con $j > t$, la representación necesita el contenido $x_{z_t}$ (para que la cadena AR funcione).

Un Transformer estándar no puede satisfacer ambos: cada token tiene un único hidden state por capa que mezcla contenido y posición.

**La solución de XLNet: mantener dos streams de representations en paralelo**:

- **Content stream** $h_{z_t}^{(m)}$: como en un Transformer estándar. Codifica contexto $x_{z_{<t}}$ **más** el contenido $x_{z_t}$ del token actual. Se inicializa con la word embedding: $h_{z_t}^{(0)} = e(x_{z_t})$.

- **Query stream** $g_{z_t}^{(m)}$: codifica solo el contexto $x_{z_{<t}}$ y la posición $z_t$, **sin** el contenido $x_{z_t}$. Se inicializa con un vector aprendible único: $g_{z_t}^{(0)} = w$ (un mismo $w$ para todas las posiciones — la información posicional viene de la atención relativa de Transformer-XL, no de esta inicialización).

Las dos actualizaciones por capa $m$ (Sección 2.3, omitiendo detalles de multi-head, LayerNorm y FFN — ver Apéndice A.2):

$$g_{z_t}^{(m)} \leftarrow \text{Attention}(Q = g_{z_t}^{(m-1)}, KV = h_{z_{<t}}^{(m-1)}; \theta) \quad \text{(query: usa } z_t \text{ pero no ve } x_{z_t}\text{)}$$

$$h_{z_t}^{(m)} \leftarrow \text{Attention}(Q = h_{z_t}^{(m-1)}, KV = h_{z_{\le t}}^{(m-1)}; \theta) \quad \text{(content: usa } z_t \text{ y } x_{z_t}\text{)}$$

La diferencia operativa es solo el **rango de claves/valores**: el query stream usa $z_{<t}$ (estrictamente anteriores en el orden), el content stream usa $z_{\le t}$ (incluye el actual). Los parámetros $\theta$ son **compartidos** entre los dos streams. No es como tener dos modelos; es como tener dos vistas del mismo modelo con máscaras de atención distintas.

En la última capa $M$, la predicción usa **el query stream**: $p_\theta(X_{z_t} = x \mid x_{z_{<t}}) \propto \exp(e(x)^\top g_{z_t}^{(M)})$.

**Detalle de fine-tuning** (Sección 2.3, fin): durante el fine-tuning **solo se usa el content stream**. El query stream se descarta. Esto es importante porque significa que el modelo en producción se comporta como un Transformer-XL estándar — solo el pre-training requiere la maquinaria de dos streams. Sin esto, XLNet sería mucho más costoso de servir.

### 2.3 Integración con Transformer-XL

Dos componentes de Transformer-XL se importan a XLNet:

**(a) Segment recurrence (caching de memoria)**. Para procesar secuencias largas, Transformer-XL parte el corpus en segmentos de longitud $T$ (e.g., 512) y, al procesar el segmento $\tau+1$, **mantiene cacheados** los hidden states de cada capa del segmento $\tau$ y los inyecta como keys/values adicionales:

$$h_{z_t}^{(m)} \leftarrow \text{Attention}\left(Q = h_{z_t}^{(m-1)}, KV = \left[\tilde{h}^{(m-1)}, h_{z_{\le t}}^{(m-1)}\right]; \theta\right)$$

donde $\tilde{h}^{(m-1)}$ son los hidden states cacheados del segmento previo y $[\cdot, \cdot]$ es concatenación a lo largo de la dimensión secuencial. Esto da al modelo un contexto efectivo de $2T$ (o más, si se mantienen múltiples segmentos) sin pagar el costo cuadrático.

**Punto sutil** (Sección 2.4): la caché del segmento previo es **agnóstica a la permutación**. Es decir, $\tilde{h}^{(m)}$ se computó bajo alguna permutación $\tilde{z}$ del segmento previo, pero como las posicionales son relativas y no absolutas, el segmento actual puede reusarlas sin saber cuál fue $\tilde{z}$. Esto permite mezclar segmentos en el data pipeline sin tracking de permutaciones — un detalle de ingeniería que hace el sistema entrenable a escala.

**(b) Relative positional encoding**. Vaswani et al. (2017) usan position embeddings absolutos sumados a las token embeddings. Esto es problemático para PLM por dos razones:

1. Con permutaciones, el "orden" cambia constantemente. Si el modelo aprende que "posición 3" significa algo específico, eso se rompe en cada permutación.
2. Con segment recurrence, las posiciones absolutas conflictúan: el token en posición 1 del segmento $\tau+1$ y el token en posición 1 del segmento $\tau$ tendrían el mismo embedding absoluto, pero son tokens completamente distintos.

La solución (heredada de Shaw et al. 2018 y refinada por Dai et al. 2019 en Transformer-XL) es **codificar la distancia relativa $i - j$ entre query y key en el cálculo de atención**, no como un embedding sumado al input. La fórmula completa de Transformer-XL para el score de atención entre query en posición $i$ y key en posición $j$:

$$A_{ij}^{\text{rel}} = \underbrace{q_i^\top W^q W^k_{\text{e}} k_j}_{\text{content-content}} + \underbrace{q_i^\top W^q W^k_{\text{r}} R_{i-j}}_{\text{content-position}} + \underbrace{u^\top W^k_{\text{e}} k_j}_{\text{global content bias}} + \underbrace{v^\top W^k_{\text{r}} R_{i-j}}_{\text{global position bias}}$$

donde $R_{i-j}$ es un embedding sinusoidal de la distancia relativa $i-j$, y $u, v$ son vectores aprendibles globales. Los términos 3 y 4 son innovaciones de Transformer-XL respecto a Shaw 2018.

**Relative segment encoding** (Sección 2.5, extensión propia de XLNet sobre Transformer-XL). Para tareas multi-segmento (QA, NLI, etc.), BERT usa embeddings $E_A$ y $E_B$ absolutos sumados al input. XLNet generaliza esto al esquema relativo: dada una atención de posición $i$ hacia posición $j$, se calcula un score adicional $a_{ij} = (q_i + b)^\top s_{ij}$ donde $s_{ij} = s_+$ si ambas posiciones están en el mismo segmento o $s_-$ si están en segmentos distintos. **No** se codifica "el segmento A vs el segmento B" — solo "mismo segmento o no". Esto tiene dos beneficios: (1) el inductive bias de relative encoding mejora generalización; (2) habilita fine-tuning en tareas con más de dos segmentos (e.g., evidencia múltiple en QA multi-hop), algo que BERT no puede hacer porque solo tiene $E_A$ y $E_B$.

---

## 3. Arquitectura y entrenamiento

### 3.1 Tamaños y dimensiones

| | $L$ | $H$ | $A$ | $d_k$ | $d_{ff}$ | Params |
|---|---|---|---|---|---|---|
| XLNet-Base | 12 | 768 | 12 | 64 | 3072 | ~110M |
| XLNet-Large | 24 | 1024 | 16 | 64 | 4096 | ~340M |

Estas dimensiones son **idénticas** a BERT-Base y BERT-Large por diseño (Sección 3.1: "*Our largest model XLNet-Large has the same architecture hyperparameters as BERT-Large, which results in a similar model size*"). La decisión es estratégica: cualquier diferencia en resultados se puede atribuir al objetivo PLM + Transformer-XL, no al tamaño. Los ~340M de XLNet-Large son ligeramente menos que BERT-Large porque XLNet **no** tiene segment embedding absoluto (lo reemplaza por relative segment encoding por head).

Tabla 7 del Apéndice A.4.1 (hiperparámetros de pre-training):

| Hparam | Valor |
|---|---|
| Layers | 24 |
| Hidden size | 1024 |
| Attention heads | 16 |
| Attention head size | 64 |
| FFN inner hidden size | 4096 |
| Hidden dropout | 0.1 |
| GeLU dropout | 0.0 |
| Attention dropout | 0.1 |
| Partial prediction $K$ | 6 |
| Max sequence length | 512 |
| Batch size | 8192 |
| Learning rate | 4e-4 |
| Number of steps | 500K |
| Warmup steps | 40,000 |
| LR decay | linear |
| Adam epsilon | 1e-6 |
| Weight decay | 0.01 |

Comparado con BERT-Large (batch 256, LR 1e-4, 1M pasos), XLNet-Large usa **batches 32× más grandes** (8192 vs 256), **LR 4× más alta** (4e-4 vs 1e-4) y **la mitad de pasos** (500K vs 1M). Esta receta es muy parecida a la de RoBERTa publicada un mes después, y sugiere que parte de la ventaja empírica de XLNet sobre BERT viene de esta receta de entrenamiento (algo que los ablations del paper intentan controlar).

### 3.2 Corpus de pre-training

XLNet usa cinco corpus distintos (Sección 3.1), sumando 32.89B subword pieces tras tokenización con SentencePiece:

| Corpus | GB texto | Subwords (B) | Notas |
|---|---|---|---|
| BooksCorpus (Zhu et al. 2015) | ~4 | 1.09 | Mismo que BERT |
| English Wikipedia | ~9 (junto con BooksCorpus = 13GB) | 2.78 | Mismo que BERT |
| Giga5 (Parker et al. 2011) | 16 | 4.75 | Newswire en inglés |
| ClueWeb 2012-B | 19 (filtrado de 100+) | 4.30 | Web crawl, filtrado agresivo de artículos cortos/low-quality |
| Common Crawl | 110 (filtrado) | 19.97 | Web crawl, filtrado agresivo |
| **Total** | ~158 GB | **32.89B** | ~10× el corpus de BERT (3.3B palabras = ~13GB) |

Comparación crítica: BERT usa solo BooksCorpus + Wikipedia (13GB, ~3.3B tokens). XLNet-Large usa ~10× más datos. Una crítica común al paper original (v1) fue: *¿la mejora viene del objetivo PLM o de tener 10× más datos?* La v2 responde con la Sección 3.2 "Fair Comparison with BERT" donde entrena **XLNet-Large-wikibooks**, idéntico a XLNet-Large pero con solo BooksCorpus + Wikipedia (igual que BERT). Los resultados de esa comparación están en la Tabla 1 del paper y los discutimos en la Sección 5.1 de este documento.

### 3.3 Tokenización con SentencePiece

XLNet abandona WordPiece (que usa BERT) y adopta **SentencePiece** (Kudo & Richardson 2018) con BPE. La razón explícita no aparece en el paper pero es práctica: SentencePiece es language-agnostic, no requiere pre-tokenización basada en espacios, y es la convención que Google adoptaría para todos sus modelos posteriores (T5, mT5, etc.). Detalle relevante para el lab 20: HuggingFace `XLNetTokenizer` requiere instalar el paquete `sentencepiece` como dependencia adicional — si solo está instalado `transformers` sin `sentencepiece`, la carga del tokenizer falla con error críptico.

El vocabulario es de 32,000 tokens (más pequeño que los 30,522 de BERT). Convenciones:

- `<cls>` (singular, en minúscula, en vez de `[CLS]`) — se coloca al **final** de la secuencia, no al inicio. Esto es contraintuitivo y es consecuencia directa de la arquitectura AR: como la atención es causal-en-permutación, el último token es el que "ha visto" más contexto en expectativa, así que es el mejor candidato para clasificación.
- `<sep>` (singular) — separador entre segmentos.
- `<pad>`, `<unk>`, `<mask>`, `<eod>` — utilitarios. Notar que `<mask>` existe pero **no se usa en el objetivo PLM**; es un vestigio para usos auxiliares.
- `▁` (underscore U+2581) — marca **inicio** de palabra (al estilo BPE de SentencePiece), opuesto a `##` de WordPiece que marca continuación.

Ejemplo: `playing` → `▁play ing`. Nótese que ningún token lleva prefijo `##`; en cambio, los tokens "inicio-de-palabra" llevan `▁`.

### 3.4 Compute

- **Hardware**: 512 chips TPU v3.
- **Tiempo**: ~5.5 días.
- **Compute total estimado**: 512 chips × 5.5 días × 24 h/día × 420 TFLOPS/chip ≈ **2.4 × 10²² FLOPs**. Aproximadamente **8× más compute que BERT-Large** (que usó 64 TPUv2 chips × 4 días).

La nota al pie del paper (Sección 3.1) observa: "*It was observed that the model still underfits the data at the end of training.*" — sugiriendo que con más compute, XLNet podría dar más. Esto es consistente con las observaciones de scaling laws que Kaplan et al. publicarían en enero 2020.

Costo monetario en USD de 2019: a $8/chip-hora TPU v3 on-demand, ~$540K USD. Con preemptible y descuentos de Google, probablemente ~$100K-200K USD efectivos. Era un experimento accesible solo para Google, Facebook y unos pocos labs académicos con conexiones a TPU clouds.

### 3.5 Span-based prediction

Detalle de la Sección 3.1 que el paper menciona en una sola oración pero que el ablation (Tabla 6, fila 6) muestra que es importante: en vez de samplear posiciones individuales para predecir, XLNet samplea **spans contiguos**. Específicamente: samplear una longitud $L \in [1, 5]$ uniformemente, samplear una posición de inicio, y seleccionar el span de $L$ tokens contiguos como targets, dentro de un contexto de $K \cdot L$ tokens.

Esta es una versión light de la **Span-BERT** de Joshi et al. (2019), publicada en paralelo. La intuición: predecir spans completos (e.g., entidades nombradas, frases) entrena al modelo en dependencias más largas que predecir tokens aleatorios sueltos.

### 3.6 Bidirectional data pipeline

Para aprovechar la memoria recurrente de Transformer-XL en ambas direcciones, XLNet usa un pipeline de datos bidireccional (Sección 3.1): **la mitad del batch procesa la secuencia en orden natural, la otra mitad en orden inverso**. Esto significa que tanto las dependencias forward como backward se ven igualmente durante el entrenamiento. El ablation (Tabla 6, fila 7) muestra que quitar esto cuesta ~0.3 puntos en SQuAD2.0 F1 y ~0.3 puntos en MNLI.

---

## 4. Resultados

XLNet-Large reporta resultados en cinco familias de benchmarks. Reproduzco los números literales del paper.

### 4.1 GLUE (Tabla 5 del paper)

| Modelo | MNLI-m/mm | QNLI | QQP | RTE | SST-2 | MRPC | CoLA | STS-B | WNLI |
|---|---|---|---|---|---|---|---|---|---|
| BERT | 86.6/- | 92.3 | 91.3 | 70.4 | 93.2 | 88.0 | 60.6 | 90.0 | - |
| RoBERTa | 90.2/90.2 | 94.7 | 92.2 | 86.6 | 96.4 | 90.9 | 68.0 | 92.4 | - |
| **XLNet** | **90.8/90.8** | **94.9** | **92.3** | 85.9 | **97.0** | 90.8 | **69.0** | **92.5** | - |

XLNet supera a BERT-Large por **+4.2** en MNLI-m, **+2.6** en QNLI, **+15.5** en RTE, **+3.8** en SST-2, **+2.8** en MRPC, **+8.4** en CoLA, **+2.5** en STS-B. Frente a RoBERTa, las diferencias son mucho menores: +0.6 en MNLI, +0.2 en QNLI, -0.7 en RTE, +0.6 en SST-2, -0.1 en MRPC, +1.0 en CoLA, +0.1 en STS-B. XLNet gana en 6 de 8 tareas frente a RoBERTa pero por márgenes pequeños.

En la **multi-task ensemble del leaderboard de octubre 2019** XLNet logra:
- MNLI: 90.9/90.9
- QNLI: 99.0
- QQP: 90.4
- RTE: 88.5
- SST-2: 97.1
- MRPC: 92.9
- CoLA: 70.2
- STS-B: 93.0
- WNLI: 92.5

QNLI=99.0 es notable — efectivamente saturando el dataset, lo que llevó a la comunidad a buscar benchmarks más difíciles (SuperGLUE).

### 4.2 SQuAD 1.1 y SQuAD 2.0 (Tabla 3)

| | SQuAD2.0 EM | SQuAD2.0 F1 | SQuAD1.1 EM | SQuAD1.1 F1 |
|---|---|---|---|---|
| BERT (dev) | 78.98 | 81.77 | 84.1 | 90.9 |
| RoBERTa (dev) | 86.5 | 89.4 | 88.9 | 94.6 |
| **XLNet (dev)** | **87.9** | **90.6** | **89.7** | **95.1** |
| BERT (test) | 80.005 | 83.061 | 85.083 | 91.835 |
| RoBERTa (test) | 86.820 | 89.795 | - | - |
| **XLNet (test)** | **87.926** | **90.689** | **89.898** | **95.080** |

XLNet logra **+7.6 F1** sobre BERT en SQuAD2.0 dev y **+1.2 F1** sobre RoBERTa. En SQuAD1.1 la mejora sobre BERT es +4.2 F1. XLNet supera a humanos (91.2 F1 para human single, ~91.2 para human aggregate) en ambos benchmarks.

**El salto en SQuAD2.0 es donde XLNet brilla**, y no es casualidad: SQuAD2.0 incluye preguntas sin respuesta, lo que requiere una cabeza adicional de clasificación "answerable vs unanswerable" (la `answer_class` que veremos en la Sección 6). La arquitectura PLM + factorizable + dependencias entre targets ayuda a esta tarea porque la decisión de "no hay respuesta" requiere modelar la coherencia global del contexto.

### 4.3 RACE (Tabla 2)

| Modelo | Accuracy total | Middle (12-15 años) | High (15-18 años) |
|---|---|---|---|
| GPT | 59.0 | 62.9 | 57.4 |
| BERT | 72.0 | 76.6 | 70.1 |
| BERT+DCMN (ensemble) | 74.1 | 79.5 | 71.8 |
| RoBERTa | 83.2 | 86.5 | 81.8 |
| **XLNet** | **85.4** | **88.6** | **84.0** |

RACE (Lai et al. 2017) son preguntas de comprensión lectora de exámenes de inglés para estudiantes chinos de secundaria, con pasajes de **>300 palabras en promedio** (vs ~120 de SQuAD). XLNet mejora +13.4 sobre BERT y +2.2 sobre RoBERTa. La autoría atribuye esta superioridad al backbone Transformer-XL: "*This superiority at dealing with longer context could come from the Transformer-XL backbone in XLNet*" (Sección 3.3, primer bullet observation).

### 4.4 Clasificación de texto (Tabla 4)

Error rate (más bajo = mejor) en 7 datasets de clasificación:

| Modelo | IMDB | Yelp-2 | Yelp-5 | DBpedia | AG | Amazon-2 | Amazon-5 |
|---|---|---|---|---|---|---|---|
| CNN | - | 2.90 | 32.39 | 0.84 | 6.57 | 3.79 | 36.24 |
| DPCNN | - | 2.64 | 30.58 | 0.88 | 6.87 | 3.32 | 34.81 |
| Mixed VAT | 4.32 | - | - | 0.70 | 4.95 | - | - |
| ULMFiT | 4.6 | 2.16 | 29.98 | 0.80 | 5.01 | - | - |
| BERT | 4.51 | 1.89 | 29.32 | 0.64 | - | 2.63 | 34.17 |
| **XLNet** | **3.20** | **1.37** | **27.05** | **0.60** | **4.45** | **2.11** | **31.67** |

XLNet logra el mejor error rate en las 7 tareas. Notar que las mejoras son grandes en tareas grandes (Yelp-5 con 560K ejemplos, Amazon-5 con 3M) que ya estaban prácticamente saturadas — sugiriendo que el objetivo PLM extrae más señal del corpus de pre-training, lo que se traduce en mejores representaciones aún para tareas con abundante data downstream.

### 4.5 Document ranking (Tabla 2, columna derecha)

ClueWeb09-B (50M documentos, queries de TREC Web Tracks 2009-2012). Setup: usar XLNet pre-entrenado para extraer **word embeddings** (sin fine-tuning), luego rankear con kernel pooling network.

| Modelo | NDCG@20 | ERR@20 |
|---|---|---|
| DRMM | 24.3 | 13.8 |
| KNRM | 26.9 | 14.9 |
| Conv-KNRM | 28.7 | 18.1 |
| BERT (autores' impl.) | 30.53 | 18.67 |
| **XLNet** | **31.10** | **20.28** |

La ganancia sobre BERT en ERR@20 (+1.6 puntos) es relevante porque, en ranking, métricas que dan más peso a los top-1/top-3 (como ERR) son las que más correlacionan con satisfacción del usuario. Que XLNet gane más en ERR que en NDCG sugiere que captura mejor el matching fino top-doc/query.

### 4.6 Observaciones del paper

El paper reporta (Sección 3.3) dos observaciones interesantes:

1. **"For explicit reasoning tasks like SQuAD and RACE that involve longer context, the performance gain of XLNet is usually larger."** Atribuido al Transformer-XL backbone.
2. **"For classification tasks that already have abundant supervised examples such as MNLI (>390K), Yelp (>560K) and Amazon (>3M), XLNet still leads to substantial gains."** — En la era pre-XLNet se creía que tareas con mucho data downstream se beneficiaban menos del pre-training (porque ya tienen señal suficiente del data labeled). XLNet desafía esa creencia: incluso en tareas con millones de ejemplos labeled, mejor pre-training todavía ayuda.

---

## 5. Análisis del XLNetForQuestionAnswering: por qué tiene 3 cabezas raras

Esta sección es crítica para el lab 20 del curso. Al hacer `XLNetForQuestionAnswering.from_pretrained("xlnet-base-cased")`, el alumno verá un warning de HuggingFace listando que se inicializan aleatoriamente los siguientes pesos:

```
- start_logits.dense.weight, start_logits.dense.bias
- end_logits.dense_0.weight, end_logits.dense_0.bias
- end_logits.LayerNorm.weight, end_logits.LayerNorm.bias
- end_logits.dense_1.weight, end_logits.dense_1.bias
- answer_class.dense_0.weight, answer_class.dense_0.bias
- answer_class.dense_1.weight, answer_class.dense_1.bias
```

Tres sub-módulos: `start_logits`, `end_logits`, `answer_class`. Esto es **muy distinto a BERT**, que solo tiene dos vectores aprendidos $S, E \in \mathbb{R}^H$ para producir start/end logits. ¿Por qué XLNet usa una arquitectura mucho más rica para QA?

### 5.1 start_logits: cabeza de start (simple)

```
start_logits.dense: Linear(hidden_size=768, 1)
```

Para cada token $i$ del párrafo, computa un logit escalar $s_i = W_s \cdot h_i + b_s$ con $W_s \in \mathbb{R}^{1 \times H}$. Luego softmax sobre todos los tokens del párrafo para obtener la distribución de probabilidad de start position:

$$p(\text{start} = i \mid x) = \frac{\exp(s_i)}{\sum_j \exp(s_j)}$$

Esta es **funcionalmente idéntica a BERT**. La cabeza de start es trivial.

### 5.2 end_logits: cabeza de end **condicionada en start**

Aquí está la innovación. En vez de predecir end independientemente de start (como BERT), XLNet condiciona end en la representación del token de start:

```
end_logits.dense_0:  Linear(2 * hidden_size, hidden_size)  # toma [h_end, h_start] concatenados
end_logits.LayerNorm: LayerNorm(hidden_size)
end_logits.dense_1:  Linear(hidden_size, 1)
```

Matemáticamente, para cada candidato de end position $j$ y dado un start position $i$:

$$\tilde{h}_j = \text{LayerNorm}(W_{e,0} \cdot [h_j; h_i] + b_{e,0})$$

$$e_{j|i} = W_{e,1} \cdot \text{GELU}(\tilde{h}_j) + b_{e,1}$$

$$p(\text{end} = j \mid \text{start} = i, x) = \frac{\exp(e_{j|i})}{\sum_{k \ge i} \exp(e_{k|i})}$$

donde $[h_j; h_i]$ es la concatenación de los hidden states final layer del candidato end y del start fijo. La restricción $k \ge i$ garantiza que end no preceda start.

**¿Por qué condicionar en start?** Empíricamente, en QA real, dado un start específico la distribución de ends razonables se reduce drásticamente. Por ejemplo, si start="Thom" entonces los ends razonables son "Yorke", "Yorke,", "Yorke is", etc., pero no algo en otra oración. BERT trata start y end como independientes y luego enumera el span score $s_i + e_j$, esperando que la combinación max-likelihood sea coherente. XLNet captura la coherencia explícitamente: la cabeza de end ve qué es el start cuando computa sus logits.

Durante training, se usa el ground-truth start position. Durante inference, se computa la top-$k$ de start positions (típicamente $k=5$) y luego para cada una se computa la cabeza de end, eligiendo el span con mayor probabilidad conjunta $p(\text{start}=i) \cdot p(\text{end}=j \mid \text{start}=i)$.

Esta es esencialmente la arquitectura de **R-Net** (Wang et al. 2017) y **QANet** (Yu et al. 2018) — modelos pre-BERT de QA que ya usaban end condicionado en start. XLNet la reintroduce sobre el backbone Transformer.

### 5.3 answer_class: cabeza de "answerable" para SQuAD 2.0

SQuAD 2.0 incluye preguntas que no tienen respuesta en el contexto. La predicción correcta para esas es "no answer". XLNet tiene una cabeza dedicada para clasificación binaria answerable/unanswerable:

```
answer_class.dense_0: Linear(2 * hidden_size, hidden_size)  # toma [h_cls, h_start_top] o similar
answer_class.dense_1: Linear(hidden_size, 1)
```

Implementación típica (basada en código HuggingFace):

$$h_{\text{ans}} = \text{tanh}(W_{a,0} \cdot [\bar{h}_{\text{start}}; h_{\text{cls}}] + b_{a,0})$$

$$\text{logit}_{\text{answerable}} = W_{a,1} \cdot h_{\text{ans}} + b_{a,1}$$

donde $\bar{h}_{\text{start}} = \sum_i p(\text{start}=i) \cdot h_i$ es una representación ponderada del start position esperado, y $h_{\text{cls}}$ es la representación del token `<cls>` (que en XLNet va al **final** de la secuencia, no al inicio como BERT).

El loss total es:

$$\mathcal{L}_{\text{QA}} = \mathcal{L}_{\text{start}} + \mathcal{L}_{\text{end}|\text{start}} + \lambda \cdot \mathcal{L}_{\text{answerable}}$$

donde $\lambda$ es un hiperparámetro (típicamente 0.5) y los dos primeros loss son cross-entropy estándar.

**¿Por qué BERT no necesita esta cabeza?** BERT-Large logra 81.9 F1 en SQuAD 2.0 con una hack más simple: comparar el span score del mejor $(i, j)$ con $s_{\text{null}} = S \cdot C + E \cdot C$ (usando el token `[CLS]` como "no answer position"), y predecir no-answer si $s_{\text{null}} > s_{i,j} + \tau$ con $\tau$ tuneado en dev. Funciona, pero es subóptimo. XLNet con la cabeza dedicada `answer_class` alcanza 87.9 EM / 90.6 F1, una mejora de +5.1 F1.

### 5.4 Implicaciones prácticas

Para el alumno en el lab 20:

1. **El warning de pesos no inicializados es esperado y correcto**. Los 6 sub-módulos listados (`start_logits.dense`, `end_logits.dense_0`, `end_logits.LayerNorm`, `end_logits.dense_1`, `answer_class.dense_0`, `answer_class.dense_1`) **no existen en el checkpoint pre-entrenado** porque el pre-training de XLNet es PLM puro, no QA. Estos pesos se aprenden durante el fine-tuning en SQuAD.

2. **El backbone (`transformer.*`) sí carga del checkpoint**. Esos son los pesos de las 12 (base) o 24 (large) capas de XLNet pre-entrenado.

3. **Para usar el modelo en inferencia, hay que fine-tunear primero**. Cargar `XLNetForQuestionAnswering.from_pretrained("xlnet-base-cased")` y hacer inferencia directa da resultados aleatorios. Hay que hacer fine-tuning en SQuAD (o un dataset propio) primero.

4. **El warning **no** indica un bug**. HuggingFace lo emite porque su política es transparente sobre qué pesos vienen del checkpoint y cuáles se inicializan. En modelos donde la cabeza tarea-específica viene del checkpoint (como `bert-large-uncased-whole-word-masking-finetuned-squad`), este warning no aparece.

---

## 6. Ablations del paper (Tabla 6, Sección 3.4)

El paper hace ablation de 8 configuraciones de XLNet-Base entrenadas en BooksCorpus + Wikipedia (igual setup que BERT). Resultados son la **mediana de 5 runs**:

| # | Modelo | RACE | SQuAD2.0 F1 | SQuAD2.0 EM | MNLI m/mm | SST-2 |
|---|---|---|---|---|---|---|
| 1 | BERT-Base | 64.3 | 76.30 | 73.66 | 84.34/84.65 | 92.78 |
| 2 | DAE + Transformer-XL | 65.03 | 79.56 | 76.80 | 84.88/84.45 | 92.60 |
| 3 | XLNet-Base ($K=7$) | 66.05 | 81.33 | 78.46 | 85.84/85.43 | 92.66 |
| 4 | XLNet-Base ($K=6$) | **66.66** | 80.98 | 78.18 | 85.63/85.12 | **93.35** |
| 5 | — memoria | 65.55 | 80.15 | 77.27 | 85.32/85.05 | 92.78 |
| 6 | — span-based pred | 65.95 | 80.61 | 77.91 | 85.49/85.02 | 93.12 |
| 7 | — bidirectional data | 66.34 | 80.65 | 77.87 | 85.31/84.99 | 92.66 |
| 8 | + next-sent pred | 66.76 | 79.83 | 76.94 | 85.32/85.09 | 92.89 |

Lecturas críticas:

**Fila 2 vs Fila 1**: DAE (masked language modeling) sobre Transformer-XL (en vez de Transformer estándar) mejora todo: +0.7 en RACE, +3.3 en SQuAD2.0 F1, +0.5 en MNLI. Esto sugiere que **una parte significativa del beneficio de XLNet sobre BERT viene simplemente de Transformer-XL**, no del objetivo PLM. La memoria recurrente y relative pos enc son load-bearing.

**Fila 4 vs Fila 2**: agregar PLM (sobre la base ya de Transformer-XL) mejora otros ~1.4 puntos en SQuAD2.0 F1 y ~0.8 en MNLI. El PLM contribuye, pero menos que Transformer-XL.

**Fila 5 vs Fila 4** (quitar memoria): cae ~0.8 puntos en SQuAD2.0 F1 y ~1.1 en RACE. Confirma que la memoria recurrente es importante, especialmente en RACE que tiene contextos largos (>300 palabras promedio).

**Fila 6 vs Fila 4** (quitar span-based prediction): cae ~0.4 puntos en SQuAD2.0 F1 y ~0.7 en RACE. Pequeño pero medible.

**Fila 7 vs Fila 4** (quitar bidirectional data pipeline): cae ~0.3 puntos en SQuAD2.0 F1. Marginal pero consistente.

**Fila 8 vs Fila 4** (agregar NSP): **NSP no mejora nada y de hecho cae** en SQuAD2.0 (-1.15 F1) y MNLI (-0.31). Esto valida la decisión de no incluir NSP en XLNet-Large, y se alinea con las conclusiones de RoBERTa y ALBERT.

**Conclusión del ablation**: la jerarquía de importancia es Transformer-XL > PLM > memoria recurrente > span-based prediction ≈ bidirectional pipeline. NSP es neutro o levemente perjudicial.

Una crítica que se le puede hacer al ablation: **no hay una ablation que aísle la regla 80/10/10 vs PLM**. Es decir, no sabemos si BERT-Base entrenado con la misma receta de XLNet (mismo LR, mismo batch, mismo número de pasos, sobre el mismo corpus) habría dado resultados similares. Esa pregunta la respondería RoBERTa tres semanas después.

### 6.1 Fair comparison vs BERT (Tabla 1)

La Sección 3.2 reporta XLNet-Large-wikibooks (entrenado solo en BooksCorpus + Wikipedia, igual que BERT) comparado contra "best of 3 BERT variants" (BERT original, BERT con whole-word-masking, BERT sin NSP):

| Modelo | SQuAD1.1 EM/F1 | SQuAD2.0 EM/F1 | RACE | MNLI | QNLI | QQP | RTE | SST-2 | MRPC | CoLA | STS-B |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BERT-Large (best of 3) | 86.7/92.8 | 82.8/85.5 | 75.1 | 87.3 | 93.0 | 91.4 | 74.0 | 94.0 | 88.7 | 63.7 | 90.2 |
| XLNet-Large-wikibooks | **88.2/94.0** | **85.1/87.8** | **77.4** | **88.4** | **93.9** | **91.8** | **81.2** | **94.4** | **90.0** | **65.2** | **91.1** |

A igual data y receta, XLNet supera a BERT en **todos** los datasets. Las ganancias son notables: +2.3 EM / +2.3 F1 en SQuAD 2.0, +2.3 en RACE, +1.1 en MNLI, +7.2 en RTE, +1.5 en CoLA. Esto valida que **el objetivo PLM + two-stream + Transformer-XL aporta mejora real sobre el setup BERT, no es solo escala de datos**.

---

## 7. Limitaciones

### 7.1 Costo computacional ~2× BERT

XLNet-Large mantiene dos hidden states por token (content + query), lo que duplica memoria y compute durante el pre-training. El paper no lo declara explícitamente pero los hiperparámetros (batch 8192 × 500K pasos × 512 TPU v3) implican aproximadamente **8× más compute total que BERT-Large**. Parte de eso es por usar 10× más datos, pero la duplicación de streams en attention es una sobrecarga constante que no se va.

Durante el fine-tuning solo se usa el content stream, así que en producción XLNet es aproximadamente igual de costoso que BERT-Large. Pero el pre-training es caro y solo Google podía permitirse entrenarlo from scratch.

### 7.2 Complejidad de implementación

Two-stream attention con segment recurrence con relative positional encoding con permutation masking es una pila técnica densa. El código de referencia (`github.com/zihangdai/xlnet`) tiene ~3K líneas de TensorFlow, varias veces más complejo que el de BERT. Esto tuvo dos consecuencias:

1. La comunidad tardó en adoptarlo. Frameworks como HuggingFace `transformers` agregaron soporte para XLNet meses después que BERT.
2. Hubo **muy pocos derivados** de XLNet (a diferencia de BERT, que tiene RoBERTa, DistilBERT, ALBERT, DeBERTa, SpanBERT, etc.). XLNet-Spanish o XLNet-Chinese son raros. La complejidad de implementación desalentó la experimentación derivada.

### 7.3 RoBERTa lo superó con simplicidad + datos

Cuando RoBERTa salió en julio 2019 (un mes después de XLNet v1), la lección fue clara: **BERT con la receta correcta (más datos, batches grandes, sin NSP, dynamic masking, más pasos, sequence length 512 desde el comienzo) iguala o supera a XLNet en la mayoría de benchmarks**.

El Apéndice de la v2 del paper de XLNet intenta defender la posición — Sección 3.3, Tabla 5: XLNet gana en 6 de 8 tareas GLUE frente a RoBERTa, pero por márgenes pequeños (0.1-1.0 puntos). En la práctica, la elección entre RoBERTa y XLNet se decidía por consideraciones prácticas (RoBERTa más simple, mejor soporte, latencia equivalente), no por accuracy. **RoBERTa se convirtió en el modelo de elección post-2019 para encoders bidireccionales**, no XLNet.

### 7.4 Dificultad de paralelizar permutaciones

En el entrenamiento, cada batch usa una permutación distinta por secuencia, lo que implica que las máscaras de atención son distintas por secuencia. Esto rompe la asunción de "una sola máscara causal global" que los kernels eficientes de attention (FlashAttention, xFormers, scaled dot product attention de PyTorch) asumen. XLNet no se beneficia de FlashAttention out-of-the-box; requiere kernels custom. Esto es otra razón por la que la comunidad lo abandonó: por cada año que pasaba sin XLNet 2.0, BERT/RoBERTa acumulaban mejoras de infraestructura que XLNet no podía absorber fácilmente.

### 7.5 El target-aware reparameterization es elegante pero frágil

El truco del two-stream attention es matemáticamente elegante pero introduce una capa de complejidad conceptual que dificulta extensions. Por ejemplo, agregar un nuevo objetivo de pre-training (e.g., contrastive learning) requiere repensar cómo interactúa con los dos streams. Investigadores han reportado dificultades para fine-tunear XLNet en setups inusuales (multi-segment beyond 2, structured prediction, etc.) precisamente por la complejidad del backbone.

### 7.6 Sin generación

Como BERT, XLNet es exclusivamente representacional. Aunque el objetivo PLM es **formalmente** un AR LM (factoriza joint probability con product rule), no puede generar texto coherente porque durante training siempre ve permutaciones aleatorias del orden de factorización, no el orden natural izquierda-a-derecha que la generación requiere. Para generar habría que samplear con factorization order = identidad y aplicar beam search/sampling, pero los autores nunca demostraron que esto funcione. **XLNet no es un LLM en el sentido de GPT-2/3** — es un encoder con objetivo distinto.

---

## 8. Conexión con la clase 20 del Diplomado IA UC

La clase 20 del Diplomado IA UC traza el arco completo de **embeddings contextualizados a la era de los LLMs**: ELMo (2018, BiLSTM), BERT (2018, encoder Transformer), GPT (2018-2020, decoder Transformer escalado), ChatGPT/RLHF (2022). XLNet ocupa un lugar **transicional** muy específico en este arco:

1. **Es post-BERT y pre-LLM**. Demuestra que hay vida más allá del MLM puro, pero también demuestra (involuntariamente) que la simplicidad arquitectural + escala (RoBERTa, luego GPT-3) gana sobre la sofisticación arquitectural pura.

2. **Conserva ideas que sobreviven en LLMs modernos**:
   - **Relative positional encoding**: usado en T5, RoPE de LLaMA, ALiBi de MPT.
   - **Segment recurrence (memoria entre segmentos)**: precursor de Transformer-XL, que a su vez precede a context windows largos vía rotary embeddings y FlashAttention. Modelos modernos como Mamba y RWKV revisitan la idea de estado recurrente entre segmentos.
   - **Partial prediction**: similar al causal masking de decoders, donde solo se predicen los tokens "futuros" desde cada posición.

3. **Conexión con el Camino 4 del curso (entender texto)**:
   - Comparte tabla con BERT y RoBERTa como uno de los encoders contextualizados más fuertes del 2019-2020.
   - Es uno de los pocos modelos que mezcla AR y AE en un solo objetivo.
   - Ilustra el principio "objetivo de pre-training importa": no solo la arquitectura.

4. **Lo que XLNet no anticipó**:
   - **Scaling laws** (Kaplan et al. 2020): demostraron que más parámetros + más datos + más compute es predictivamente mejor que objetivos más sofisticados.
   - **In-context learning** (GPT-3, 2020): cambió la pregunta de "¿qué pre-training es mejor?" a "¿qué emerge con escala?". XLNet no escala bien (complejidad, costo, dificultad de implementación), así que no entró en la conversación de los LLMs.

XLNet es, retrospectivamente, **una bifurcación elegante pero efímera** en el árbol evolutivo del pre-training. La lección que dejó es valiosa: las ideas de modelado matter, pero la simplicidad + escala suele ganar al largo plazo. RoBERTa absorbió la mayoría de las mejoras prácticas de XLNet (no NSP, batches grandes, más datos, más pasos) sin la complejidad del two-stream attention.

---

## 9. Conexión con el lab 20 del Diplomado IA UC

El lab 20 toca XLNet en al menos tres celdas (12-14, según el material del curso). Los puntos relevantes:

### 9.1 Celda de carga: `XLNetModel`

```python
from transformers import XLNetModel, XLNetTokenizer
tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
model = XLNetModel.from_pretrained("xlnet-base-cased")
```

Esto carga el backbone puro (~110M params para `base-cased`). El alumno verá que `XLNetTokenizer` requiere `sentencepiece` como dependencia — si no está instalado, el comando falla con un error críptico ("Couldn't instantiate the backend tokenizer"). Solución: `pip install sentencepiece`.

Otro detalle a notar: el output de `XLNetModel(input_ids).last_hidden_state` tiene shape `(batch, seq_len, hidden_size)` igual que BERT, **pero la convención de tokens especiales es distinta**:

```
Input:  "Hello world"
BERT:   [CLS] Hello world [SEP]                # CLS al inicio
XLNet:  ▁Hello ▁world <sep> <cls>             # CLS al final
```

Si el alumno intenta usar `output.last_hidden_state[:, 0, :]` esperando obtener el token de clasificación (como en BERT), obtendrá el primer token del texto real, no el `<cls>`. Debe usar `output.last_hidden_state[:, -1, :]`.

### 9.2 Celda de QA: `XLNetForQuestionAnswering`

```python
from transformers import XLNetForQuestionAnswering
model = XLNetForQuestionAnswering.from_pretrained("xlnet-base-cased")
```

Aquí aparece el warning de los 6 sub-módulos no inicializados (`start_logits.dense`, `end_logits.dense_0`, `end_logits.LayerNorm`, `end_logits.dense_1`, `answer_class.dense_0`, `answer_class.dense_1`). El alumno debe entender que **esto es esperado** porque el checkpoint pre-entrenado no incluye cabezas de QA — esas se aprenden durante fine-tuning. La Sección 5 de este documento explica en detalle por qué XLNet tiene 6 sub-módulos en vez de 2 (como BERT): cabeza de end condicionada en start (R-Net-style) más cabeza dedicada de answerability para SQuAD 2.0.

Para hacer inferencia directa sin fine-tunear, el alumno debe usar un checkpoint ya fine-tuneado en SQuAD, e.g. `pierreguillou/xlnet-base-cased-squad-v2-french` o similar (estos están en HuggingFace Hub pero son escasos para XLNet).

### 9.3 Celda de multiple choice: `XLNetForMultipleChoice`

```python
from transformers import XLNetForMultipleChoice
model = XLNetForMultipleChoice.from_pretrained("xlnet-base-cased")
```

Para benchmarks como SWAG, RACE, ARC. La cabeza es una capa lineal `Linear(hidden_size, 1)` que toma el `<cls>` final (último token) de cada opción y produce un score; luego softmax sobre las opciones. Equivalente conceptualmente al setup de BERT para SWAG, pero con `<cls>` al final.

El warning de pesos no inicializados también aparece aquí (`sequence_summary.summary.weight`, `logits_proj.weight`). Misma explicación: cabeza tarea-específica no viene del pre-training.

### 9.4 Por qué `sentencepiece` es obligatorio

A diferencia de BERT (WordPiece, implementado nativamente en `transformers`), XLNet usa SentencePiece, que requiere un binding de C++ externo. La razón histórica: SentencePiece soporta tokenización language-agnostic sin pre-tokenización por espacios, lo que es importante para Asian languages (chino, japonés, coreano) donde no hay separación clara de palabras. Google adoptó SentencePiece como estándar para todos sus modelos post-2018, y XLNet hereda esa decisión.

---

## 10. Notas para integrar al site

Cosas que conviene agregar al `papers/xlnet-yang-2019.md` del site público (que es más superficial):

1. **Derivación formal del objetivo PLM** con ejemplo de [New, York, is, a, city] mostrando concretamente la diferencia $\mathcal{L}_{\text{BERT}}$ vs $\mathcal{L}_{\text{XLNet}}$.
2. **El problema técnico del two-stream attention**: por qué un Transformer estándar falla bajo PLM (Apéndice A.1, dos permutaciones distintas dan misma predicción).
3. **Tabla del ablation completo** (Tabla 6 del paper) con las 8 filas, no solo el resumen.
4. **Diagrama de las 3 cabezas de XLNetForQuestionAnswering** y por qué cada una existe (start, end condicionado en start, answerable).
5. **Comparación numérica fina con RoBERTa**: XLNet gana en 6 de 8 tareas GLUE pero por márgenes pequeños; ambos superan ampliamente a BERT.
6. **Costo computacional**: 512 TPU v3 chips × 5.5 días ≈ 2.4 × 10²² FLOPs, ~8× BERT-Large.
7. **Limitaciones que la era LLM hizo evidentes**: dificultad de paralelizar permutaciones, complejidad de implementación, sin generación.
8. **Conexión con Transformer-XL como precursor**: los mismos primeros autores (Yang & Dai), genealogía explícita.

El `fundamentos/embeddings-contextualizados.md` del site podría sumar una sección sobre XLNet como punto medio entre encoder puro (BERT) y decoder puro (GPT), enfatizando que la permutación es una forma de "tener bidireccionalidad sin máscara".

---

## 11. Lectura recomendada complementaria

- **Transformer-XL** (Dai, Yang et al. 2019, arXiv:1901.02860) — precursor directo de XLNet. Mismos primeros autores. Lectura obligatoria para entender la memoria recurrente y relative pos encoding.
- **RoBERTa** (Liu et al. 2019, arXiv:1907.11692) — el competidor inmediato que mostró que BERT bien entrenado iguala XLNet. Excelente contraste metodológico.
- **MADE** (Germain et al. 2015) y **NADE** (Uria et al. 2016) — los precursores conceptuales de PLM aplicados a estimación de densidad en datos discretos.
- **SpanBERT** (Joshi et al. 2019, arXiv:1907.10529) — propone span-based masking, comparable al span-based prediction de XLNet.
- **ELECTRA** (Clark et al. 2020, arXiv:2003.10555) — otro ataque al problema del mask discrepancy, distinto y más simple que XLNet.
- **DeBERTa** (He et al. 2020, arXiv:2006.03654) — adopta relative positional encoding (como XLNet) y disentangled attention. Considerado el sucesor moral de XLNet en el árbol de encoders bidireccionales.
- **A Primer in BERTology** (Rogers et al. 2020) — survey que contextualiza XLNet entre los modelos BERT-like.
- **Self-Attention with Relative Position Representations** (Shaw et al. 2018, arXiv:1803.02155) — el paper original de relative pos encoding que Transformer-XL y XLNet refinan.
- **Implementación HuggingFace de XLNet**: [`transformers/src/transformers/models/xlnet/modeling_xlnet.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/xlnet/modeling_xlnet.py) — el código de las 6 sub-cabezas de QA en `XLNetForQuestionAnswering`.
