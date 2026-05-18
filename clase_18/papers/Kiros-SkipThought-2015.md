# Kiros et al. 2015 — Skip-Thought Vectors

| Campo | Valor |
|---|---|
| **Autores** | Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel, Antonio Torralba, Raquel Urtasun, Sanja Fidler |
| **Afiliación** | University of Toronto, CIFAR, MIT |
| **Venue** | NeurIPS 2015 |
| **Pdf** | `Kiros-SkipThought-2015.pdf` (9 páginas) |
| **Citaciones** | >3.500 |
| **URL** | https://papers.nips.cc/paper/2015/hash/f442d33fa06832082290ad8544a8da27-Abstract.html |
| **Código** | https://github.com/ryankiros/skip-thoughts |

> *"We describe an approach for unsupervised learning of a generic, distributed sentence encoder."*

Skip-Thought es el **primer modelo no-supervisado que produce sentence embeddings transferibles**. Generaliza la idea de Word2Vec Skip-gram del nivel de palabra al nivel de oración: dada una oración, predecir las **oraciones adyacentes** en un corpus de texto continuo. La oración se codifica con una RNN y el estado oculto final se interpreta como su embedding.

---

## 1. Contexto y motivación

### 1.1 El problema de los sentence embeddings en 2014-2015

Word embeddings (Word2Vec, GloVe) estaban establecidos. Pero **representar oraciones** seguía siendo un problema abierto. Las opciones disponibles:

#### Opción A — Composición de word embeddings (sumar/promediar)

Simple pero pobre. `vec("the cat sat") = vec("the") + vec("cat") + vec("sat")` ignora el orden y la estructura sintáctica.

#### Opción B — Modelos supervisados con backpropagation

- **Recursive NN** (Socher 2011, 2013): construir representación bottom-up sobre el árbol sintáctico.
- **CNN para oraciones** (Kim 2014): convolucionar word embeddings y poolear.
- **Recurrent NN / LSTM** entrenado en clasificación.
- **Recursive autoencoders** (Socher 2011).

**Problema común**: requieren **etiquetas supervisadas para una tarea específica**. La representación queda tunada a esa tarea y no transfiere bien.

#### Opción C — Paragraph Vectors / Doc2Vec (Le & Mikolov 2014)

Aprenden un vector por documento como parte de un LM. **Limitación crítica**: en test time, hay que **re-entrenar** un vector para cada nuevo documento (gradient descent por inferencia). No es una verdadera función encoder.

### 1.2 La pregunta del paper

> *"Is there a task and a corresponding loss that will allow us to learn highly generic sentence representations?"*

Kiros et al. responden: **sí — generalizar Skip-gram a oraciones**.

---

## 2. La idea central

### 2.1 Analogía con Skip-gram

| Skip-gram (Word2Vec) | Skip-Thought |
|---|---|
| Unidad: palabra $w_t$ | Unidad: oración $s_i$ |
| Predecir: palabras del contexto $w_{t \pm j}$ | Predecir: oraciones adyacentes $s_{i \pm 1}$ |
| Encoder: embedding lookup | Encoder: RNN-GRU |
| Decoder: softmax sobre $|V|$ | Decoder: RNN-GRU palabra-por-palabra |
| Resultado: word embeddings | Resultado: sentence embeddings |

### 2.2 Setup

Dado un corpus de **texto continuo** (e.g., libros donde las oraciones se siguen una a otra), para cada triplet $(s_{i-1}, s_i, s_{i+1})$:

1. **Encodear** $s_i$ con una RNN-GRU → vector $\mathbf{h}_i \in \mathbb{R}^d$.
2. **Decodear** $s_{i+1}$ con una RNN-GRU condicionada en $\mathbf{h}_i$.
3. **Decodear** $s_{i-1}$ con una segunda RNN-GRU condicionada en $\mathbf{h}_i$.

**Loss**: cross-entropy negativa sumada sobre las dos predicciones:

$$
\mathcal{L} = \sum_t \log P(w_{i+1}^t \mid w_{i+1}^{<t}, \mathbf{h}_i) + \sum_t \log P(w_{i-1}^t \mid w_{i-1}^{<t}, \mathbf{h}_i). \quad (10)
$$

---

## 3. Arquitectura en detalle

### 3.1 Encoder — GRU estándar

GRU es la elección porque (cita del paper) *"GRU units have only 2 gates and do not require the use of a cell. While we use RNNs for our model, any encoder and decoder can be used so long as we can backpropagate through it."*

Para oración $s_i$ con palabras $w_i^1, \dots, w_i^N$, en cada paso temporal:

$$
\mathbf{r}^t = \sigma(\mathbf{W}_r \mathbf{x}^t + \mathbf{U}_r \mathbf{h}^{t-1}) \quad (1)
$$
$$
\mathbf{z}^t = \sigma(\mathbf{W}_z \mathbf{x}^t + \mathbf{U}_z \mathbf{h}^{t-1}) \quad (2)
$$
$$
\bar{\mathbf{h}}^t = \tanh(\mathbf{W} \mathbf{x}^t + \mathbf{U}(\mathbf{r}^t \odot \mathbf{h}^{t-1})) \quad (3)
$$
$$
\mathbf{h}^t = (1 - \mathbf{z}^t) \odot \mathbf{h}^{t-1} + \mathbf{z}^t \odot \bar{\mathbf{h}}^t \quad (4)
$$

- $\mathbf{r}^t$: reset gate.
- $\mathbf{z}^t$: update gate.
- $\bar{\mathbf{h}}^t$: candidate state.
- $\mathbf{h}^t$: state.

El **sentence embedding** $\mathbf{h}_i = \mathbf{h}^N$, el último estado tras procesar la oración completa.

### 3.2 Decoder — Conditional GRU

La novedad técnica: el decoder es un GRU **condicionado** en el sentence embedding del encoder. Las matrices $\mathbf{C}_r, \mathbf{C}_z, \mathbf{C}$ inyectan $\mathbf{h}_i$ en cada gate:

$$
\mathbf{r}^t = \sigma(\mathbf{W}_r^d \mathbf{x}^{t-1} + \mathbf{U}_r^d \mathbf{h}^{t-1} + \mathbf{C}_r \mathbf{h}_i) \quad (5)
$$
$$
\mathbf{z}^t = \sigma(\mathbf{W}_z^d \mathbf{x}^{t-1} + \mathbf{U}_z^d \mathbf{h}^{t-1} + \mathbf{C}_z \mathbf{h}_i) \quad (6)
$$
$$
\bar{\mathbf{h}}^t = \tanh(\mathbf{W}^d \mathbf{x}^{t-1} + \mathbf{U}^d (\mathbf{r}^t \odot \mathbf{h}^{t-1}) + \mathbf{C} \mathbf{h}_i) \quad (7)
$$
$$
\mathbf{h}_{i+1}^t = (1 - \mathbf{z}^t) \odot \mathbf{h}^{t-1} + \mathbf{z}^t \odot \bar{\mathbf{h}}^t \quad (8)
$$

La probabilidad de generar la palabra $w_{i+1}^t$ dado el contexto:

$$
P(w_{i+1}^t \mid w_{i+1}^{<t}, \mathbf{h}_i) \propto \exp(\mathbf{v}_{w_{i+1}^t} \mathbf{h}_{i+1}^t) \quad (9)
$$

donde $\mathbf{v}_w$ es la fila correspondiente a $w$ en la matriz de salida $\mathbf{V}$.

### 3.3 Dos decoders independientes

Un decoder para $s_{i+1}$ (siguiente) y otro para $s_{i-1}$ (previa), con **parámetros separados** ($\mathbf{W}^d, \mathbf{U}^d, \mathbf{C}$). Solo la matriz de vocabulario $\mathbf{V}$ se comparte.

### 3.4 Variantes del modelo

| Modelo | Encoder | Dimensión |
|---|---|---|
| **uni-skip** | Unidireccional, 2400 unidades | 2400 |
| **bi-skip** | Bidireccional (forward + backward), 1200 c/u | 1200 + 1200 = 2400 (concatenación) |
| **combine-skip** | concat(uni-skip, bi-skip) | 4800 |

En la práctica, **combine-skip es el ganador** en la mayoría de evaluaciones.

---

## 4. Vocabulary expansion — el truco crucial

### 4.1 Problema

El corpus de entrenamiento (BookCorpus) tiene vocab $V_{\text{rnn}} \approx 20$k palabras frecuentes. Pero **en test time queremos encodear oraciones con palabras que no estaban en BookCorpus** (e.g., nombres propios, jerga técnica). Sin alternativa, esas palabras serían `<UNK>` y degradarían el embedding.

### 4.2 Solución: regresión lineal entre espacios

Sea $V_{\text{w2v}}$ el vocabulario de Word2Vec preentrenado (~3M palabras, cobertura amplia, descargable de Google).

**Idea**: aprender una matriz lineal $\mathbf{W}$ tal que $\mathbf{W} \cdot \text{w2v}(w) \approx \text{rnn}(w)$ para palabras $w$ en la intersección $V_{\text{rnn}} \cap V_{\text{w2v}}$.

Proceso:
1. Tomar las palabras compartidas entre ambos vocabularios.
2. Resolver L2 regression: $\min_{\mathbf{W}} \sum_{w \in V_{\text{rnn}} \cap V_{\text{w2v}}} \| \mathbf{W} \cdot \text{w2v}(w) - \text{rnn}(w) \|^2$.
3. Para una palabra **nueva** $w' \in V_{\text{w2v}}$ pero $\notin V_{\text{rnn}}$, generar embedding $\text{rnn}'(w') = \mathbf{W} \cdot \text{w2v}(w')$.

**Resultado**: el vocabulario efectivo del encoder pasa de 20k a **930.911 palabras** (~46× expansión).

### 4.3 Origen de la idea

Mikolov 2013 (NAACL) había usado un truco similar para alinear embeddings de **diferentes idiomas** (translation). Skip-Thought aplica la misma idea para alinear vocabs **dentro del mismo idioma** entre Word2Vec y la RNN.

---

## 5. Corpus de entrenamiento — BookCorpus

| Estadística | Valor |
|---|---|
| # libros | 11.038 |
| # oraciones | 74.004.228 |
| # palabras (tokens) | 984.846.357 |
| # palabras únicas | 1.316.420 |
| Media palabras/oración | 13 |
| Géneros | 16 (Romance, Fantasy, Sci-Fi, Teen, etc.) |

**BookCorpus** fue construido por Zhu et al. (ICCV 2015) y contiene novelas no publicadas escritas por autores no profesionales (gratis). Razones para usarlo:

- Texto continuo y narrativo (las oraciones se siguen) — ideal para predicción contextual.
- Diálogo + emoción + interacción entre personajes — más rico que Wikipedia.
- ~1B palabras, suficiente escala.

**Nota histórica**: BookCorpus se eliminó posteriormente por problemas de copyright. Es el mismo corpus usado para preentrenar **BERT-base** y **GPT-1**.

---

## 6. Setup de entrenamiento

- **Inicialización**: matrices recurrentes con **inicialización ortogonal** (Saxe 2014); no-recurrentes uniformes en $[-0.1, 0.1]$.
- **Optimizer**: Adam.
- **Batch size**: 128.
- **Gradient clipping**: por norma, threshold 10.
- **Tiempo**: ~2 semanas en GPU.

---

## 7. Evaluación

Skip-Thought se evalúa en 8 tareas **transfer learning** estilo "linear probe": se extraen sentence embeddings y se entrena un clasificador lineal encima — sin fine-tuning.

### 7.1 Semantic relatedness (SICK)

Dataset SemEval 2014 Task 1: 4.500 train + 500 dev + 4.927 test pairs de oraciones con score humano de similitud en escala 1-5.

**Setup**: para pair $(u, v)$, computar features $u \cdot v$ (component-wise product) y $|u - v|$, concatenarlas, entrenar regresión logística.

**Resultados (Tabla 3 izquierda)**:

| Método | Pearson $r$ | Spearman $\rho$ | MSE |
|---|---|---|---|
| Mean vectors | 0.7577 | 0.6738 | 0.4557 |
| Dependency Tree-LSTM | **0.8676** | **0.8083** | **0.2532** |
| bow (baseline) | 0.7823 | 0.7235 | 0.3975 |
| **uni-skip** | 0.8477 | 0.7780 | 0.2872 |
| **bi-skip** | 0.8405 | 0.7696 | 0.2995 |
| **combine-skip** | **0.8584** | **0.7916** | **0.2687** |
| combine-skip+COCO | 0.8655 | 0.7995 | 0.2561 |

**Lectura**: Skip-Thought (combine-skip) supera al promedio de word vectors y todos los baselines de SemEval. Solo es vencido por Tree-LSTM, **que requiere un parser dependencial** (no aplicable a idiomas sin parser).

### 7.2 Paraphrase detection (MSR Paraphrase Corpus)

| Método | Acc | F1 |
|---|---|---|
| feats (Madnani 2012) | 73.2 | — |
| RAE+DP+feats | 76.8 | 83.6 |
| TF-KLD (Ji & Eisenstein) | **80.4** | **86.0** |
| bow | 67.8 | 80.3 |
| uni-skip | 73.0 | 81.9 |
| bi-skip | 71.2 | 81.2 |
| combine-skip | 73.0 | 82.0 |
| combine-skip + feats | **75.8** | **83.0** |

Skip-Thought (sin features adicionales) compite con métodos heavily-engineered.

### 7.3 Image-sentence ranking (Flickr30k)

Skip-Thought se usa como text encoder en un modelo de embedding conjunto imagen-texto, en complemento de un image encoder VGG. Image-to-text retrieval R@10 = 75.8% (vs 66.9% del baseline RNN entrenado from-scratch).

### 7.4 Clasificación de oraciones

5 datasets: MR (movie reviews), CR (customer reviews), SUBJ (subjectivity), MPQA (opinion polarity), TREC (question types).

| Method | MR | CR | SUBJ | MPQA | TREC |
|---|---|---|---|---|---|
| NB-SVM (baseline fuerte) | 79.4 | 81.8 | 93.2 | 86.3 | — |
| Mean word vectors | 77.7 | 79.8 | 90.9 | 88.3 | 81.4 |
| CNN (Kim 2014) | 81.5 | 85.0 | 93.4 | 89.6 | 93.6 |
| **combine-skip** | 76.5 | 80.1 | 93.6 | 87.1 | 92.2 |
| **combine-skip + NB** | 80.4 | 83.1 | 93.6 | 87.5 | 92.4 |

Skip-Thought es competitivo con métodos supervisados de la era, sin haber visto las labels.

---

## 8. Análisis cualitativo — nearest sentences

Tabla 2 del paper muestra nearest neighbors por cosine similarity sobre 500k oraciones. Ejemplos:

| Query | Vecino más cercano |
|---|---|
| "he ran his hand inside his coat, double-checking that the unopened letter was still there." | "he slipped his hand between his coat and his shirt, where the folded copies lay in a brown envelope." |
| "im sure youll have a glamorous evening, she said, giving an exaggerated wink." | "im really glad you came to the party tonight, he said, turning to her." |
| "an annoying buzz started to ring in my ears, becoming louder and louder as my vision began to swim." | "a weighty pressure landed on my lungs and my vision blurred at the edges, threatening my consciousness altogether." |

**Observación**: Skip-Thought captura **semántica de eventos y emociones**, no solo palabras compartidas. Las oraciones similares pueden no compartir vocabulario sustantivo pero describen situaciones análogas.

---

## 9. Limitaciones reconocidas

1. **Costo computacional**: 2 semanas en GPU para entrenar. Inviable para investigadores sin recursos.
2. **Vocabulary trick es awkward**: la expansión del vocab vía regresión lineal es ingeniosa pero un hack. BPE/WordPiece (BERT, 2018) resolverá esto de raíz.
3. **Captura limitada de polisemia y composicionalidad fina**: el modelo falla en distinguir "tricks on a motorcycle" vs "tricking a person on a motorcycle" (cita del paper).
4. **Solo evaluado en inglés**: requiere corpus narrativo (BookCorpus). Difícil de adaptar a low-resource languages.
5. **Encoder es secuencial**: lento en inferencia y limitado en dependencias largas.

---

## 10. Impacto y legado

### 10.1 Skip-Thought como punto de inflexión

- **Antes (2014)**: sentence embeddings solo en contexto supervisado o vía Doc2Vec (problemático).
- **Después (2015)**: Skip-Thought demuestra que **se puede aprender un encoder universal de oraciones sin supervisión**. Esta idea se vuelve canónica.

### 10.2 Sucesores conceptuales directos

| Año | Modelo | Innovación |
|---|---|---|
| 2017 | **InferSent** (Conneau, FAIR) | Reemplaza autosupervisión por supervisión en SNLI (NLI dataset). Mejor desempeño downstream. |
| 2018 | **Universal Sentence Encoder** (Cer, Google) | Transformer-based + multi-task training. |
| 2019 | **Sentence-BERT (SBERT)** (Reimers & Gurevych) | BERT siamés con contrastive loss → SOTA para sentence similarity. |
| 2020 | **SimCSE** (Gao 2021) | Contrastive learning con dropout como augmentation. |
| 2021 | **Sentence-T5** | T5 como encoder, scaling de SBERT. |
| 2022 | **gtr-t5**, **E5** | Embedding models specifically para retrieval. |

Todos heredan la idea fundamental de Skip-Thought: **un encoder universal aprendido sin supervisión / con autosupervisión que produce embeddings transferibles**.

### 10.3 Doble impacto: encoder universal + autosupervisión a nivel de oración

Skip-Thought contribuye dos ideas profundamente influyentes:

1. **Encoder universal de oraciones**: precede a InferSent, USE, SBERT.
2. **Objetivo autosupervisado a nivel de oración**: precede a **Next Sentence Prediction (NSP)** de BERT, **Sentence Order Prediction (SOP)** de ALBERT, y los objetivos contrastivos modernos.

---

## 11. Conexión con la clase 18

Slides 37-40 cubren Skip-Thought:

- **Slide 37**: portada del paper.
- **Slide 38**: idea central "aplicar misma idea del skip-gram pero a nivel de oraciones, para obtener sentence embeddings; dada oración del medio, predecir la anterior y la siguiente; una RNN encoder + dos RNN decoders".
- **Slide 39**: aplicaciones downstream — semantic relatedness, paraphrase, classification.
- **Slide 40**: Tabla 2 del paper con nearest sentence examples.

Lo que la slide NO menciona (y se cubre acá):
- La **arquitectura GRU específica** del encoder/decoder.
- La inyección via **conditional GRU** ($\mathbf{C}_r, \mathbf{C}_z, \mathbf{C}$).
- El **vocabulary expansion** trick.
- Los detalles cuantitativos (SICK $r$ = 0.86, MSRP F1 = 83).

---

## 12. Conexión con la clase moderna (BERT y más allá)

Aunque Skip-Thought fue eclipsado por BERT (2018), su lógica persiste:

| Skip-Thought | BERT NSP | SimCSE |
|---|---|---|
| Predecir oraciones adyacentes | Predecir si dos oraciones son adyacentes | Contrastive sobre oraciones (positive: dropout-aug, negative: random) |
| RNN encoder | Transformer encoder | Transformer encoder |
| Autosupervisado | Autosupervisado | Autosupervisado |
| Embedding = $\mathbf{h}_N$ | Embedding = `[CLS]` | Embedding = `[CLS]` |

La transición de RNN a Transformer es lo único que cambia conceptualmente. La autosupervisión a nivel de oración es la **misma idea** que Skip-Thought planteó primero.

---

## 13. Cita BibTeX

```bibtex
@inproceedings{kiros2015skip,
  title={Skip-thought vectors},
  author={Kiros, Ryan and Zhu, Yukun and Salakhutdinov, Russ R and Zemel, Richard and Urtasun, Raquel and Torralba, Antonio and Fidler, Sanja},
  booktitle={Advances in Neural Information Processing Systems},
  pages={3294--3302},
  year={2015}
}
```

---

## 14. Frase para recordar

> *"From words to thoughts — apply the Skip-gram trick one level up."* — Skip-Thought es la traducción más directa imaginable de Word2Vec al nivel de oraciones, y abrió la puerta a toda la era moderna de sentence embeddings.

---

## 15. Notas técnicas

- **Hiperparámetro clave**: dimensión 2400 (uni) o 1200×2 (bi) — son **muy grandes para 2015**. Es lo que permitió capacidad suficiente.
- **Optimizer Adam** (Kingma 2014) era nuevo en 2015 — Skip-Thought es uno de los primeros papers grandes en usarlo.
- **Pretrained embeddings**: disponibles en `https://github.com/ryankiros/skip-thoughts`. Hoy se usan rara vez — SBERT los superó por ~5-15 puntos en STS benchmarks.
