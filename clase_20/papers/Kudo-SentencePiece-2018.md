# Análisis interno — Kudo & Richardson (2018) "SentencePiece" + Kudo (2018) "Subword Regularization"

> Documento complementario al material público del site (`fundamentos/tokenizacion-subword.md`, `fundamentos/bpe.md` si existe, `clase-20/practica/_index.md`). Aquí se profundiza tanto en la **herramienta** SentencePiece (Kudo & Richardson 2018 EMNLP Demo) como en el **algoritmo Unigram Language Model** introducido en el paper hermano de Kudo (2018 ACL "Subword Regularization"). Ambos papers son piezas inseparables: SentencePiece es la implementación open-source que hizo masivo al algoritmo Unigram, y entender Unigram es prerrequisito para entender por qué la mayoría de modelos post-BERT (T5, XLNet, ALBERT, mBART, mT5, LLaMA, Mistral, Gemma) dependen del paquete `sentencepiece` en HuggingFace.

- **Paper 1 (herramienta)**: Kudo, T. & Richardson, J. *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing*. arXiv:1808.06226v1, 19 Aug 2018. EMNLP 2018 (System Demonstrations).
- **Paper 2 (algoritmo)**: Kudo, T. *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates*. arXiv:1804.10959, 29 Apr 2018. ACL 2018.
- **PDF local del paper de la herramienta**: `papers/Kudo-SentencePiece-2018.pdf`
- **Código y release**: `https://github.com/google/sentencepiece` — C++ con bindings Python, Apache 2.0, ~30K stars (2026).

---

## 1. El problema: tokenización pre-2018 era un campo minado

### 1.1 El paisaje fragmentado de tokenizadores

Hasta 2018 la pipeline típica de un sistema NMT (Neural Machine Translation) o de cualquier modelo de lenguaje involucraba **al menos tres pasos distintos de preprocesamiento de texto**, cada uno con sus propias reglas, dependencias y bugs:

1. **Normalización Unicode**: NFC, NFKC, lowercasing, mapeo de caracteres fullwidth a ASCII, eliminación de acentos. Distintas librerías (ICU, Python `unicodedata`, custom) producían resultados ligeramente distintos.
2. **Tokenización a nivel palabra**: separar por espacios y signos de puntuación. Para europeas, `moses-tokenizer.perl`; para japonés, MeCab o KyTea; para chino, Jieba; para tailandés, regex específicos. Cada herramienta hacía suposiciones distintas sobre clíticos, hyphens, números.
3. **Tokenización subword**: BPE de `subword-nmt` (Sennrich et al. 2016) que operaba **sobre el output ya tokenizado por palabra**, no sobre texto crudo.

Esta cadena tenía cuatro problemas graves que el paper de Kudo & Richardson enumera explícitamente:

#### 1.1.1 No-reversibilidad (lossy tokenization)

El ejemplo canónico del paper:

```
Raw text:    Hello world.
Tokenized:   [Hello] [world] [.]
```

La información de que **no hay espacio** entre `world` y `.` se pierde en la tokenización. La detokenización debe asumir reglas heurísticas: "punto pega al token anterior", "coma pega al token anterior", "apóstrofe depende del idioma". Estas reglas son **language-dependent** y rompen con casos exóticos. Para chino y japonés directamente no funcionan:

```
Raw text:    こんにちは世界。  (Hola mundo.)
Tokenized:   [こんにちは] [世界] [。]
```

No hay espacios. La detokenización en europeas que pone espacios entre tokens produce texto **incorrecto** en CJK (Chinese/Japanese/Korean).

#### 1.1.2 Dependencia de tokenizadores externos para idiomas sin espacios

Para entrenar un modelo NMT japonés-inglés con la pipeline pre-2018, el pipeline era:

```
texto japonés → KyTea (segmentación de palabras) → subword-nmt BPE → IDs
texto inglés  → moses-tokenizer → subword-nmt BPE → IDs
```

KyTea es un binario externo, mantenido por la comunidad académica japonesa, que requiere modelos pre-entrenados de segmentación. Era una dependencia frágil, difícil de empaquetar en Docker, e imposible de ejecutar en TPUs o entornos restringidos. Lo mismo para Jieba (chino) o ICU word break (tailandés).

#### 1.1.3 Bloqueo para sistemas multilingües

Cuando Google Translate quiso construir un solo modelo NMT multilingüe (Johnson et al. 2016, "Google's Multilingual NMT System"), tuvieron que mantener **104 pipelines de tokenización paralelas**, una por idioma. Cualquier inconsistencia entre ellas (diferentes versiones de Moses, diferentes modelos de KyTea) degradaba el modelo. Y al agregar un idioma nuevo, había que descubrir el tokenizador apropiado para él.

#### 1.1.4 Reproducibilidad rota

(Post 2018, "A Call for Clarity in Reporting BLEU Scores") documentó que dos papers que reportaban BLEU sobre WMT14 podían diferir en hasta 1.8 puntos solo por **diferencias de preprocesamiento** invisibles (versión de Moses, configuración de tokenizer, normalización Unicode). Como el preprocesamiento se hacía con scripts externos y sus configuraciones rara vez se reportaban, era imposible reproducir resultados.

### 1.2 BPE clásico: una solución parcial

BPE (Byte-Pair Encoding) entró a NLP con Sennrich et al. 2016 ("Neural Machine Translation of Rare Words with Subword Units"). La idea: en vez de mantener un vocabulario fijo de palabras (que sufre OOV crónico), aprender un vocabulario de **subword units** mediante merges iterativos.

Algoritmo BPE original:

1. Empezar con vocabulario de caracteres individuales más un símbolo end-of-word `</w>`.
2. Contar la frecuencia de **pares adyacentes** de símbolos en el corpus.
3. Fusionar el par más frecuente en un nuevo símbolo.
4. Repetir hasta alcanzar el número deseado de merges (típicamente 32K-50K).

El problema: este algoritmo asume que el corpus ya viene **pre-tokenizado a nivel palabra**. La razón es eficiencia — contar pares dentro de cada palabra es $O(N)$ con $N$ palabras, pero contar pares en todo el corpus crudo es $O(\text{chars})$, mucho más caro. `subword-nmt` requería ejecutar `moses-tokenizer` antes.

Y para japonés, sin pre-tokenización razonable, BPE producía vocabularios degenerados — fusionaba caracteres a través de límites semánticos que un humano nunca cruzaría.

### 1.3 La necesidad

Para 2018, con NMT moviéndose hacia arquitecturas language-agnostic (Transformer, multilingual training), el campo necesitaba un tokenizador que:

- Operara directamente sobre **texto Unicode crudo**, sin pre-tokenización.
- Fuera **reversible perfectamente**: `detokenize(tokenize(x)) == normalize(x)`.
- Fuera **rápido** suficiente para procesar millones de oraciones.
- Soportara **muestreo** de segmentaciones (para data augmentation, no solo una segmentación greedy).
- Fuera **self-contained**: el modelo embedded toda la normalización y vocabulario en un solo archivo.
- Tuviera **APIs** en C++ y Python para integración on-the-fly en frameworks NMT.

SentencePiece fue la respuesta a esa lista de requisitos.

---

## 2. Contribuciones de SentencePiece

El paper de Kudo & Richardson 2018 es un **system demonstration paper** (EMNLP Demo Track) — describe principalmente la herramienta y sus decisiones de diseño, no un algoritmo nuevo. El algoritmo (Unigram LM) fue introducido en Kudo 2018 ACL. Pero SentencePiece consolida varias contribuciones de ingeniería que cambiaron cómo se hace tokenización en NLP.

### 2.1 Stream de Unicode codepoints, espacio como caracter

La innovación conceptual central: tratar el input como **una secuencia de codepoints Unicode**, incluyendo el espacio como un símbolo más. Para distinguir el espacio del resto, SentencePiece lo escapa con un meta-símbolo: `▁` (U+2581, lower one eighth block).

```
Raw text:        Hello world.
Internal:        ▁Hello▁world.
Tokenized:       [▁Hello] [▁world] [.]
```

Esto resuelve el problema de la no-reversibilidad. Para detokenizar, basta con:

```python
detok = ''.join(tokens).replace('▁', ' ')
```

Esta operación es **idempotente** y **language-independent**. No requiere reglas heurísticas distintas por idioma.

Comparación con la convención de `subword-nmt` (que usa `@@` como marker intra-palabra):

```
subword-nmt:     [Hello] [wor@@] [ld] [.]
SentencePiece:   [▁Hello] [▁wor] [ld] [.]
```

Diferencia técnica importante: `subword-nmt` **no puede representar espacios múltiples consecutivos** (porque su convención asume un espacio implícito entre tokens "completos"). SentencePiece sí, porque cada espacio es un `▁` explícito. Esto importa para textos como código fuente o poesía con espaciado significativo.

### 2.2 Sin preprocesamiento de tokenización por palabra

Como SentencePiece opera sobre Unicode crudo, **no necesita** un tokenizador de palabras previo. Esto es revolucionario para CJK y otros idiomas sin espacios:

```python
# Antes (con subword-nmt):
texto_jp = "こんにちは世界。"
tokens_palabra = kytea.tokenize(texto_jp)   # ["こんにちは", "世界", "。"]
subwords = subword_nmt.encode(tokens_palabra)

# Con SentencePiece:
texto_jp = "こんにちは世界。"
subwords = sp.encode(texto_jp)              # directo
```

El experimento del paper (Tabla 1) muestra que para inglés-japonés en KFTT, SentencePiece **sin pre-tokenización** alcanza BLEU comparable o mejor que con pre-tokenización:

| Dirección | Setting | BLEU |
|---|---|---|
| ja→en | Word baseline (80K vocab) | 28.24 |
| ja→en | SentencePiece (8K shared) | **29.55** |
| ja→en | SentencePiece + pre-tok | 29.85 |
| en→ja | Word baseline | 20.06 |
| en→ja | SentencePiece (8K shared) | **21.62** |
| en→ja | SentencePiece + pre-tok | 20.86 |

Notar dos lecturas:

1. SentencePiece **mejora** sobre word-level baseline a pesar de usar **10× menos vocabulario** (8K vs 80K).
2. La pre-tokenización **no ayuda** en en→ja (de hecho la perjudica: 21.62 → 20.86). El modelo aprende mejores segmentaciones desde el texto crudo que las impuestas por KyTea.

### 2.3 Reversibilidad perfecta (lossless tokenization)

Por diseño:

$$\text{Decode}(\text{Encode}(\text{Normalize}(\text{text}))) = \text{Normalize}(\text{text})$$

Notar que **no** se garantiza que el resultado sea idéntico al input original — solo que es idéntico a la versión normalizada. La normalización (NFKC por default) es lossy de forma controlada: convierte caracteres semánticamente equivalentes pero visualmente distintos (e.g., fullwidth `Ａ` U+FF21 → ASCII `A` U+0041). Esta es una pérdida deseable y predecible.

### 2.4 Implementación rápida en C++ con bindings

La implementación es C++ con dos algoritmos optimizados:

- **BPE**: usa un binary heap (priority queue) para mantener los pares candidatos. Complejidad $O(N \log N)$ en vez del $O(N^2)$ naive.
- **Unigram LM**: complejidad linear en tamaño del input para entrenamiento y segmentación.

La Tabla 2 del paper muestra el speedup vs `subword-nmt` (que es puro Python):

| Tarea | Tool | Pre-tok | Japonés (s) | Inglés (s) |
|---|---|---|---|---|
| Train | subword-nmt | yes | 56.9 | 54.1 |
| Train | SentencePiece | yes | 10.1 | 16.8 |
| Train | subword-nmt | no | 528.0 | 94.7 |
| Train | SentencePiece | no | 217.3 | 21.8 |
| Seg. | subword-nmt | no | 216.2 | 36.1 |
| Seg. | SentencePiece | no | **5.9** | 20.3 |

En segmentación de japonés sin pre-tokenización, **SentencePiece es ~37× más rápido** (216.2 / 5.9). Para entrenamiento, **~50× más rápido en español/inglés sin pre-tok**. Estas diferencias de orden de magnitud son las que permitieron tokenizar corpora de varios TB para modelos como mT5, mBART y LLaMA.

Bindings disponibles:

- **Python**: `pip install sentencepiece` — wrapper directo del C++ con pybind11.
- **C++ API**: `#include <sentencepiece_processor.h>`.
- **TensorFlow ops**: el modelo se embebe como atributo del grafo, así no hay dependencia externa en serving.

### 2.5 Soporte de DOS algoritmos: BPE y Unigram LM

SentencePiece soporta cuatro modos de segmentación:

| Modo | Algoritmo | Cuando usar |
|---|---|---|
| `unigram` (default) | Unigram Language Model (Kudo 2018) | Default moderno. Permite sampling. |
| `bpe` | Byte-Pair Encoding (Sennrich 2016) | Compatibilidad con `subword-nmt`. |
| `char` | Character-level | Baselines, debugging. |
| `word` | Word-level (split por espacios) | Baselines. |

El default desde 2018 es `unigram`, y es lo que usan T5, XLNet, ALBERT, mBART, mT5, LLaMA, Mistral, Gemma. La sección 4 cubre BPE y la sección 5 Unigram en detalle.

### 2.6 Self-contained models (Protocol Buffer)

El modelo entrenado se serializa como un **Protocol Buffer binario** que contiene:

- El vocabulario completo con scores log-probabilísticos (Unigram) o reglas de merge (BPE).
- La tabla pre-compilada del **finite state transducer (FST)** para normalización Unicode.
- Todos los símbolos especiales (`<unk>`, `<s>`, `</s>`, `<pad>`, custom).
- Hyperparámetros usados en el entrenamiento.

Una sola llamada `sp.Load("spm.model")` carga todo. **No hay dependencias externas**. Esto resuelve el problema de reproducibilidad de Post 2018: si distribuyes el `.model`, cualquiera reproduce tu preprocesamiento exactamente.

Este detalle de diseño es lo que permitió que HuggingFace pudiera distribuir tokenizadores de T5/LLaMA/Mistral como un solo archivo (`tokenizer.model`) cargable con una línea.

### 2.7 Normalización customizable

Por default, SentencePiece aplica **NFKC** (Normalization Form Compatibility Composition). El paper aclara (footnote 3) que solo implementa un subset — específicamente no soporta la reordenación CCC (Canonical Combining Class) completa, porque es difícil de modelar en un FST. Para la mayoría de aplicaciones esto es invisible.

Usuarios pueden definir reglas custom via TSV:

```
U+41 U+302 U+300 <tab> U+1EA6
U+41 U+302 U+301 <tab> U+1EA4
```

Las reglas se compilan en un Aho-Corasick automaton, lo que permite leftmost-longest matching en $O(N)$.

---

## 3. BPE vs Unigram LM: comparación fundamental

Antes de entrar a la matemática del Unigram, conviene contrastar conceptualmente con BPE para entender qué resuelve cada uno.

### 3.1 BPE — greedy y determinista

**Algoritmo de entrenamiento**:

1. Inicializar vocab = conjunto de caracteres únicos en el corpus.
2. Mientras `|vocab| < target_size`:
   a. Contar todos los pares adyacentes $(a, b)$ en el corpus tokenizado actual.
   b. Encontrar el par $(a^*, b^*)$ con mayor frecuencia.
   c. Agregar $a^* b^*$ (concatenación) al vocab como nuevo símbolo.
   d. Reemplazar todas las ocurrencias de $a^* b^*$ en el corpus.
3. Output: lista ordenada de reglas de merge.

**Segmentación**:

1. Empezar con caracteres individuales.
2. Aplicar las reglas de merge **en el orden en que fueron aprendidas**.
3. Cada palabra produce **una única** segmentación.

**Propiedades**:

- **Determinista**: para un input dado, siempre el mismo output.
- **Greedy**: la decisión de merge en cada paso es local, no globalmente óptima.
- **Sin probabilidades**: las reglas son discretas, no hay noción de "probabilidad de un segmento".
- **No permite sampling**: por construcción, una sola segmentación.

### 3.2 Unigram LM — probabilístico y muestreable

**Modelo**: el vocabulario $V$ tiene asociada una distribución de probabilidad $P(v)$ sobre los tokens. La probabilidad de una segmentación $\mathbf{s} = (s_1, s_2, ..., s_n)$ de una secuencia es:

$$P(\mathbf{s}) = \prod_{i=1}^{n} P(s_i)$$

La probabilidad de la secuencia $x$ es la suma sobre todas las segmentaciones posibles:

$$P(x) = \sum_{\mathbf{s} \in \mathcal{S}(x)} P(\mathbf{s}) = \sum_{\mathbf{s} \in \mathcal{S}(x)} \prod_{i=1}^{|\mathbf{s}|} P(s_i)$$

donde $\mathcal{S}(x)$ es el conjunto de todas las segmentaciones legales de $x$ usando solo tokens en $V$.

**Propiedades**:

- **Probabilístico**: cada token tiene un score $\log P(v)$.
- **Globalmente óptimo (para el modelo dado)**: la segmentación elegida es la de máxima probabilidad bajo el modelo.
- **Permite sampling**: se puede muestrear de las top-$k$ segmentaciones más probables.
- **Permite la suma marginal**: se puede calcular la probabilidad total de $x$ marginalizando sobre segmentaciones (útil para regularización).

### 3.3 Comparación lado a lado

| Característica | BPE | Unigram LM |
|---|---|---|
| Tipo de modelo | Reglas de merge discretas | Distribución probabilística |
| Algoritmo de aprendizaje | Greedy bottom-up | EM (Expectation-Maximization) |
| Segmentación | Determinista | Viterbi (greedy en log-prob) o sampling |
| Multiplicidad de segmentaciones | 1 (única) | Top-$k$ disponibles |
| Permite data augmentation | No | Sí (subword regularization) |
| Inicialización del vocab | Caracteres | Vocabulario grande (e.g., 1M tokens) |
| Construcción del vocab | Aditiva (crece) | Sustractiva (se podan tokens) |
| Velocidad de segmentación | $O(N \log N)$ con heap | $O(N \cdot V_{max})$ con Viterbi |

### 3.4 Vocabularios resultantes

En la práctica, BPE y Unigram producen vocabularios **similares pero no idénticos** para el mismo corpus. Diferencias típicas:

- **BPE** tiende a producir vocabulario sesgado hacia subwords muy frecuentes en el corpus exacto, con muchos morfemas comunes (`ing`, `ed`, `ly` en inglés).
- **Unigram** tiende a producir vocabulario más balanceado, con presencia de subwords menos frecuentes pero más informativos. Maximiza likelihood global, no frecuencia local.

Ejemplo: para `tokenization` en un corpus técnico, BPE podría segmentar como `[token][ization]` (porque `token` e `ization` son muy frecuentes). Unigram podría preferir `[tokenize][ation]` si maximiza la likelihood global del corpus mejor.

Para corpora de varios idiomas, Unigram típicamente produce vocabularios más equilibrados entre idiomas, lo que es importante para modelos multilingües como mT5 y XLM-R.

---

## 4. Matemática del Unigram Language Model

Esta sección desarrolla en detalle el algoritmo de Kudo 2018 ACL, que es el corazón del modo `unigram` de SentencePiece.

### 4.1 Modelo generativo

Asumiendo el modelo de unigrams (cada token es independiente):

$$P(\mathbf{s}) = \prod_{i=1}^{n} P(s_i), \quad \sum_{v \in V} P(v) = 1$$

La probabilidad de la secuencia $x$ marginalizando sobre segmentaciones es:

$$P(x) = \sum_{\mathbf{s} \in \mathcal{S}(x)} P(\mathbf{s})$$

La log-likelihood del corpus $\mathcal{D} = \{x^{(1)}, x^{(2)}, ..., x^{(|\mathcal{D}|)}\}$ es:

$$\mathcal{L}(\theta) = \sum_{x \in \mathcal{D}} \log P(x) = \sum_{x \in \mathcal{D}} \log \left( \sum_{\mathbf{s} \in \mathcal{S}(x)} \prod_{i=1}^{|\mathbf{s}|} P(s_i) \right)$$

donde $\theta = \{P(v) : v \in V\}$ es el conjunto de probabilidades de tokens.

El problema de aprendizaje es **doble**:

1. **Estimación**: dado un vocabulario $V$, encontrar $P(v)$ que maximiza $\mathcal{L}$.
2. **Selección de vocabulario**: encontrar el $V$ óptimo (típicamente de tamaño fijo, e.g., 32K).

### 4.2 EM para estimar $P(v)$ dado $V$ fijo

Si fijamos el vocabulario $V$, encontrar $P(v)$ óptimo es un problema clásico de Expectation-Maximization, porque la segmentación $\mathbf{s}$ es una **variable latente**.

**E-step**: para cada palabra $x$ del corpus y cada token $v \in V$, calcular la probabilidad esperada de que $v$ aparezca en la segmentación de $x$:

$$\mathbb{E}[\text{count}(v, x)] = \sum_{\mathbf{s} \in \mathcal{S}(x)} P(\mathbf{s} \mid x) \cdot \text{count}(v, \mathbf{s})$$

donde $P(\mathbf{s} \mid x) = \frac{P(\mathbf{s})}{P(x)} = \frac{\prod P(s_i)}{\sum_{\mathbf{s}'} \prod P(s'_j)}$.

Este cálculo se hace eficientemente con el algoritmo **forward-backward** (igual que en HMMs). La complejidad es $O(|x| \cdot V_{\text{max}})$ donde $V_{\text{max}}$ es la longitud máxima de un token.

**M-step**: actualizar $P(v)$ como la frecuencia esperada normalizada:

$$P(v) = \frac{\sum_{x \in \mathcal{D}} \mathbb{E}[\text{count}(v, x)]}{\sum_{v' \in V} \sum_{x \in \mathcal{D}} \mathbb{E}[\text{count}(v', x)]}$$

Iterar E y M hasta convergencia (típicamente 2-3 iteraciones bastan en la práctica).

### 4.3 Selección de vocabulario por pruning iterativo

El segundo problema es **cuál** vocabulario $V$ elegir. Kudo 2018 propone un procedimiento de "vocabulary contraction":

**Algoritmo completo**:

1. **Inicialización del vocab grande**. Generar un vocabulario inicial $V_0$ muy grande (e.g., 1M tokens). Esto se hace típicamente con un algoritmo como:
   - Suffix array sobre el corpus para encontrar todas las substrings frecuentes.
   - Tomar las top-1M por frecuencia.
   - Alternativa: usar BPE para inicializar.

2. **Bucle hasta alcanzar tamaño objetivo $V^*$** (e.g., 32K):

   a. **Ejecutar EM** sobre $V_t$ para encontrar $P^*(v)$ que maximiza la likelihood actual.

   b. **Calcular la "loss" de eliminar cada token**: para cada $v \in V_t$, calcular $L_v = \mathcal{L}(V_t) - \mathcal{L}(V_t \setminus \{v\})$. Esto es la pérdida de log-likelihood si se elimina $v$. Tokens que se pueden recomponer fácilmente desde subwords más pequeños tienen $L_v$ bajo; tokens críticos (caracteres atómicos, símbolos únicos) tienen $L_v$ alto.

   c. **Ordenar tokens por $L_v$** y eliminar el bottom $\eta$% (e.g., 20%). Los tokens "salvables" son los que tienen `L_v` muy bajo.

   d. **Restricción de seguridad**: nunca eliminar caracteres atómicos del corpus. Esto garantiza que cualquier texto siga siendo segmentable (sin OOV).

3. **Output**: $V$ del tamaño objetivo, con probabilidades $P(v)$ asociadas.

### 4.4 Decodificación con Viterbi

Una vez entrenado el modelo, segmentar un nuevo input $x$ requiere encontrar la segmentación de máxima probabilidad:

$$\mathbf{s}^* = \arg\max_{\mathbf{s} \in \mathcal{S}(x)} \prod_{i=1}^{|\mathbf{s}|} P(s_i) = \arg\max_{\mathbf{s} \in \mathcal{S}(x)} \sum_{i=1}^{|\mathbf{s}|} \log P(s_i)$$

Esto es el algoritmo clásico de **Viterbi** sobre el "lattice" de segmentaciones posibles:

- Construir un grafo dirigido acíclico (DAG) donde los nodos son posiciones $[0, |x|]$ y las aristas son tokens $v \in V$ que coinciden con substrings de $x$ partiendo de una posición y terminando en otra.
- Peso de cada arista: $\log P(v)$.
- Encontrar el camino de máxima suma desde el nodo 0 hasta el nodo $|x|$ con programación dinámica.

Complejidad: $O(|x| \cdot V_{\text{max}})$. Para $|x| = 100$ y $V_{\text{max}} = 16$, esto son ~1,600 operaciones — extremadamente rápido.

### 4.5 N-best decoding y sampling

Para subword regularization (sección 5), necesitamos no solo la mejor segmentación sino las top-$k$ más probables. Esto se logra extendiendo Viterbi a **lazy n-best**:

- En vez de mantener un solo backpointer por nodo, mantener los top-$k$ caminos parciales.
- Al final, recuperar los $k$ caminos completos por backtracking.

Complejidad: $O(|x| \cdot V_{\text{max}} \cdot k)$.

Para sampling propiamente, hay dos opciones:

1. **Top-$k$ sampling con softmax temperado**: dadas las $k$ segmentaciones con scores $\log P(\mathbf{s}_1), ..., \log P(\mathbf{s}_k)$, samplear con probabilidad $\propto \exp(\alpha \cdot \log P(\mathbf{s}_i))$ donde $\alpha > 0$ es un hiperparámetro de smoothing.

2. **Forward-filtering / backward-sampling**: ejecutar forward, luego samplear backward usando las probabilidades posteriores. Esto da una muestra exacta de la posterior sin restringir a top-$k$.

SentencePiece implementa ambas opciones.

### 4.6 Estabilidad numérica

Las probabilidades $P(v)$ se almacenan como **log-probabilidades** (`float32`) para evitar underflow. La suma marginal $P(x)$ se calcula con la identidad `log-sum-exp`:

$$\log P(x) = \log \sum_{\mathbf{s}} P(\mathbf{s}) = \text{logsumexp}(\{\log P(\mathbf{s}_i)\}_i)$$

donde $\text{logsumexp}(a_1, ..., a_n) = \max_i a_i + \log \sum_i \exp(a_i - \max_i a_i)$.

### 4.7 Costo computacional comparado con BPE

Aspecto curioso: aunque Unigram es matemáticamente más sofisticado que BPE, su costo de entrenamiento es **lineal** en el tamaño del input (gracias al EM eficiente con forward-backward), mientras que BPE naive es $O(N^2)$. SentencePiece implementa BPE con un binary heap para reducirlo a $O(N \log N)$, pero Unigram sigue siendo más rápido en práctica para vocabularios grandes.

---

## 5. Subword Regularization

La motivación original de Kudo 2018 ACL para introducir Unigram LM no era reemplazar BPE per se, sino **habilitar data augmentation a nivel de tokenización**. Esto es lo que se conoce como **subword regularization**.

### 5.1 El problema: tokenización determinista limita generalización

Cuando un modelo NMT se entrena siempre viendo la misma segmentación de `interesting` como `[interest][ing]`, no aprende que también puede aparecer como `[inter][est][ing]` o `[i][nteresting]`. Esto crea un problema de **brittleness**: pequeños errores de input (typos, variantes morfológicas no vistas) producen segmentaciones drásticamente distintas y degradan la performance.

La hipótesis: si durante el entrenamiento exponemos al modelo a **múltiples segmentaciones válidas** del mismo input, aprenderá representaciones más robustas a la segmentación.

### 5.2 El procedimiento

Durante el entrenamiento, en cada step:

1. Para cada input $x$ del batch, muestrear una segmentación $\mathbf{s} \sim P_\alpha(\mathbf{s} \mid x)$.
2. Procesar $\mathbf{s}$ como input al modelo.
3. En el siguiente epoch, el mismo $x$ se segmentará distinto.

La distribución de muestreo es controlada por un hiperparámetro $\alpha$:

$$P_\alpha(\mathbf{s} \mid x) = \frac{P(\mathbf{s})^\alpha}{\sum_{\mathbf{s}'} P(\mathbf{s}')^\alpha}$$

- $\alpha = 0$: distribución uniforme sobre todas las segmentaciones. Máxima diversidad, pero también máximo ruido.
- $\alpha \to \infty$: distribución deltática en la segmentación más probable (equivalente a Viterbi).
- $\alpha \approx 0.1 - 0.5$: balance típico recomendado por el paper.

Y un segundo hiperparámetro $l$:

- $l$ = número de top candidatos considerados. $l = \infty$ considera toda la distribución; $l = 64$ trunca a las top-64.

### 5.3 Implementación en SentencePiece

La API expone esto directamente:

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load("spm.model")

# Segmentación determinista (Viterbi)
sp.EncodeAsPieces("New York")
# ['▁New', '▁York']

# Sampling con alpha=0.1
for _ in range(5):
    print(sp.SampleEncodeAsPieces("New York", -1, 0.1))
# ['▁', 'N', 'e', 'w', '▁York']
# ['▁', 'New', '▁York']
# ['▁', 'New', '▁Y', 'o', 'r', 'k']
# ['▁', 'New', '▁York']
# ['▁', 'New', '▁York']
```

(Reproducido textualmente de Figure 6 del paper.)

### 5.4 Resultados experimentales (Kudo 2018 ACL)

El paper de subword regularization reporta mejoras de **+1.2 a +2.4 BLEU** en NMT sobre múltiples pares de idiomas, especialmente en escenarios low-resource (donde la regularización ayuda más).

Tabla típica (reconstruida del paper):

| Lang pair | Baseline (Viterbi) | Subword Reg. ($\alpha=0.1$, $l=64$) |
|---|---|---|
| IWSLT15 en-vi | 25.3 | **27.7** (+2.4) |
| IWSLT15 en-zh | 12.9 | **14.7** (+1.8) |
| WMT14 en-de | 28.1 | **29.0** (+0.9) |

Notar que las ganancias son más grandes en low-resource (IWSLT) que en high-resource (WMT). Esto es consistente con la teoría de regularización: cuando hay poco data, prevenir overfitting ayuda más.

### 5.5 BPE-dropout: la respuesta de BPE

Provilkov et al. 2020 ("BPE-Dropout: Simple and Effective Subword Regularization") demostró que se puede obtener un efecto similar con BPE: simplemente, durante la segmentación, "dropear" cada merge con probabilidad $p$ (típicamente 0.1). Esto introduce estocasticidad en BPE sin requerir el framework completo de Unigram.

Sin embargo, BPE-dropout llegó **dos años después**. Para 2018-2019, Unigram LM era la única opción para subword regularization, lo que explica su adopción por T5, XLNet, ALBERT.

### 5.6 ¿Se usa en pre-training de LLMs modernos?

Sorprendentemente, **la mayoría de LLMs modernos no usan subword regularization durante pre-training**. La razón: con datasets de escala TB-scale, la regularización por sampling de segmentaciones es marginal frente a la cantidad de data ya vista. La motivación original (low-resource NMT) no aplica.

Pero el algoritmo Unigram en sí (sin regularización, usando Viterbi) **sí** es estándar en estos modelos. La distinción importante:

- LLaMA, T5, Gemma usan **Unigram LM como tokenizador** (Viterbi en inferencia).
- No usan **subword regularization** (no samplean durante training).

---

## 6. Comparación BPE vs Unigram vs WordPiece

Para cerrar la teoría, una tabla comparativa de los tres algoritmos dominantes de subword tokenization:

| Aspecto | BPE | Unigram LM | WordPiece |
|---|---|---|---|
| Año | 1994 (compresión) / 2016 (NLP) | 2018 | 2012 (japonés/coreano) / 2016 (Google NMT) |
| Paper canónico | Sennrich et al. 2016 | Kudo 2018 ACL | Schuster & Nakajima 2012 / Wu et al. 2016 |
| Criterio de merge | Frecuencia del par | Likelihood global | Likelihood (PMI del par) |
| Construcción del vocab | Bottom-up (crece) | Top-down (se poda) | Bottom-up (crece) |
| Determinismo | Determinista | Determinista (con Viterbi) o sampleable | Determinista |
| Sampling | No nativo (BPE-dropout 2020) | Nativo | No |
| Marca de boundary | `</w>` end-of-word o `Ġ` start (byte-BPE) | `▁` start | `##` continuación |
| Operación sobre raw text | Requiere pre-tokenización | No requiere (con SentencePiece) | Requiere pre-tokenización |
| Modelos populares | GPT-2, RoBERTa, BART (byte-BPE), original Sennrich | T5, XLNet, ALBERT, mBART, mT5, LLaMA, Mistral, Gemma | BERT, ELECTRA, DistilBERT, MobileBERT |
| Librería estándar | `subword-nmt`, `tokenizers` (HF), SentencePiece | SentencePiece, `tokenizers` (HF) | `tokenizers` (HF), TF Text |

### 6.1 BERT WordPiece en detalle

WordPiece (usado en BERT, ELECTRA) es muy parecido a BPE pero con un criterio de merge basado en likelihood. En cada iteración, en vez de fusionar el par más frecuente, fusiona el par $(a, b)$ que maximiza:

$$\Delta \mathcal{L} = \log \frac{P(ab)}{P(a) \cdot P(b)}$$

Esto es equivalente a maximizar la pointwise mutual information (PMI). Tokens que aparecen juntos mucho más de lo que predeciría su frecuencia individual se fusionan primero.

En la práctica, BPE y WordPiece producen vocabularios muy similares. La diferencia más visible es la convención de marcado:

- BPE: `playing` → `[play] [Ġing]` (en byte-BPE)
- WordPiece: `playing` → `[play] [##ing]`

### 6.2 GPT-2 byte-level BPE

Una variante importante: GPT-2 (Radford et al. 2019) introduce **byte-level BPE**. En vez de operar sobre caracteres Unicode, opera sobre **bytes UTF-8**. Esto significa:

- Vocabulario inicial: 256 bytes (en vez de los miles de caracteres Unicode).
- Cualquier secuencia UTF-8 puede ser tokenizada sin OOV (porque cada byte está en el vocab).
- Para caracteres no-ASCII (chino, emojis), múltiples bytes se fusionan en tokens.

Ventaja: cobertura universal. Desventaja: vocabularios menos eficientes para idiomas no-latinos (un caracter chino puede requerir 3 bytes → 3 tokens iniciales que luego se fusionan).

GPT-2, GPT-3, RoBERTa, BART usan byte-level BPE. **No usan SentencePiece**.

### 6.3 Resumen práctico: ¿cuándo usar cuál?

- **Si estás entrenando un nuevo modelo desde cero**: Unigram LM con SentencePiece. Es el default razonable para 2024+.
- **Si compatibilizas con GPT-2/GPT-3 ecosystem**: byte-level BPE (HF `tokenizers` o `tiktoken` de OpenAI).
- **Si compatibilizas con BERT ecosystem**: WordPiece. HuggingFace lo implementa nativamente.
- **Si necesitas data augmentation a nivel tokenización**: Unigram LM con sampling.
- **Si tu corpus es CJK o multilingual**: Unigram LM con SentencePiece, sin pre-tokenización.

---

## 7. Adopción industrial: el camino a omnipresencia

La adopción de SentencePiece + Unigram fue progresiva pero terminó dominando el ecosistema post-BERT.

### 7.1 Timeline de adopción

| Año | Modelo | Tokenizador | Notas |
|---|---|---|---|
| 2018 | BERT | WordPiece | 30,522 tokens, basado en `bert-vocab.txt` |
| 2018 | GPT-1 | BPE (fairseq) | 40K merges |
| 2019 | RoBERTa | byte-BPE | 50,265 tokens |
| 2019 | XLNet | **SentencePiece (Unigram)** | 32K tokens. Primer gran modelo en adoptarlo. |
| 2019 | ALBERT | **SentencePiece (Unigram)** | 30K tokens |
| 2019 | T5 | **SentencePiece (Unigram)** | 32K tokens. Default de Google Research. |
| 2019 | XLM-R | **SentencePiece (Unigram)** | 250K tokens, multilingual |
| 2020 | mBART | **SentencePiece (Unigram)** | 250K tokens |
| 2020 | mT5 | **SentencePiece (Unigram)** | 250K tokens |
| 2020 | GPT-3 | byte-BPE (tiktoken predecessor) | 50,257 tokens, hereda de GPT-2 |
| 2021 | DeBERTa-v3 | **SentencePiece (Unigram)** | 128K tokens |
| 2022 | PaLM | **SentencePiece (Unigram)** | 256K tokens |
| 2023 | LLaMA | **SentencePiece (Unigram)** | 32K tokens, basado en BPE-mode aunque! |
| 2023 | LLaMA 2 | **SentencePiece (Unigram)** | 32K tokens |
| 2023 | Mistral | **SentencePiece (Unigram)** | 32K tokens |
| 2023 | Gemma | **SentencePiece (Unigram)** | 256K tokens |
| 2024 | LLaMA 3 | **tiktoken** | 128K tokens. **Cambio**: LLaMA 3 abandona SentencePiece. |
| 2024 | GPT-4o | tiktoken (cl100k_base evolved) | ~200K tokens |

(Nota técnica: LLaMA original usa SentencePiece en **modo BPE**, no Unigram. Pero el archivo `tokenizer.model` y el código de carga es de SentencePiece. El paquete `sentencepiece` es dependencia obligatoria para `transformers` con LLaMA hasta LLaMA 2.)

### 7.2 División de mercado por familia

Por línea de descendencia:

- **Familia BERT (encoder-only para entendimiento)**: WordPiece. BERT, ELECTRA, DistilBERT, mBERT.
- **Familia T5/XLNet (encoder-decoder o decoder-only de Google Research)**: SentencePiece (Unigram). T5, XLNet, ALBERT, mT5, PaLM, Gemma.
- **Familia GPT (decoder-only de OpenAI)**: byte-level BPE. GPT-2, GPT-3, GPT-4 (con tiktoken).
- **Familia LLaMA (decoder-only open de Meta)**: SentencePiece (BPE-mode) hasta v2, tiktoken-style desde v3.
- **Familia multilingual de Facebook**: SentencePiece (Unigram). XLM-R, mBART.

La razón de la fragmentación es histórica: cada equipo escogió en su momento lo más estable disponible, y por compatibilidad de checkpoints ya no pueden migrar fácilmente.

### 7.3 Por qué `pip install sentencepiece` aparece tanto

Cualquier alumno o practicante de HuggingFace `transformers` que cargue un tokenizador de T5, XLNet, ALBERT, mBART, mT5, LLaMA, Mistral o Gemma necesita el paquete `sentencepiece`. HuggingFace ha **integrado SentencePiece como dependencia opcional** porque no es trivial empacar el binario C++ en wheels universales.

Si el paquete no está instalado, instanciar un tokenizer asociado da:

```
ImportError: 
T5Tokenizer requires the SentencePiece library but it was not found in your environment.
Checkout the instructions on the installation page of its repo:
https://github.com/google/sentencepiece#installation and follow the ones
that match your environment. Please note that you may need to restart your runtime
after installation.
```

Esto es exactamente lo que motiva la celda 8 del lab 20 del Diplomado IA UC.

---

## 8. Por qué se instala separado en HuggingFace

Razones técnicas específicas:

### 8.1 Es C++, no Python puro

SentencePiece es ~30K líneas de C++ con bindings Python vía pybind11. Compilar el código C++ es **plataforma-específico**:

- Linux x86_64: relativamente sencillo, hay wheels precompilados.
- macOS arm64 (M1/M2/M3): wheels disponibles desde 2022.
- Windows: requiere MSVC, históricamente problemático.
- ARM64 Linux (Raspberry Pi, AWS Graviton): wheels limitados.

Cuando HuggingFace empaqueta `transformers`, no puede asumir que el SentencePiece esté disponible en todas las plataformas target. Por eso lo marca como **dependencia opcional**.

### 8.2 Instalación

Las dos formas canónicas:

```bash
# Solo SentencePiece
pip install sentencepiece

# transformers con SentencePiece (extras)
pip install transformers[sentencepiece]
```

Para conda:

```bash
conda install -c conda-forge sentencepiece
```

### 8.3 Verificación rápida

```python
import sentencepiece as spm
print(spm.__version__)  # e.g., 0.2.0
```

### 8.4 Alternativas (parcial reemplazo)

HuggingFace ofrece `tokenizers` (Rust, parte del ecosistema HF) que implementa Unigram desde cero. Para muchos modelos hay un **fast tokenizer** equivalente (`T5TokenizerFast`, `XLNetTokenizerFast`) que no requiere SentencePiece. Pero **requiere haber convertido el `.model` a formato HF** primero, lo cual a su vez requería SentencePiece. Para muchos checkpoints solo está disponible el slow tokenizer, y `sentencepiece` sigue siendo necesario.

---

## 9. Conexión con la clase 20

La clase 20 del Diplomado IA UC trata el **Camino 4: Comprensión y generación de lenguaje** y específicamente cubre la transición de embeddings estáticos (Word2Vec, GloVe) a embeddings contextuales (ELMo, BERT) y modelos generativos (GPT, ChatGPT). En este recorrido, **el tokenizador es la tubería invisible** que conecta el texto humano con todos los modelos modernos.

Las conexiones específicas:

### 9.1 Pre-BERT: tokenizadores por palabra

Word2Vec y GloVe (clases 18-19 del Diplomado) operan a nivel palabra. Tienen vocabulario fijo (e.g., 1M palabras) y sufren OOV crónico — cualquier palabra fuera del vocabulario se reemplaza por `<unk>` y pierde toda su información. SentencePiece (junto con WordPiece y BPE) resuelve esto a nivel arquitectónico.

### 9.2 BERT WordPiece

BERT (paper canónico de la clase) usa WordPiece, no SentencePiece. Pero los descendientes inmediatos de BERT que la clase también cubre — ALBERT, RoBERTa (byte-BPE), DistilBERT — adoptan tokenizadores distintos. La heterogeneidad de tokenizadores en la familia BERT muestra que **el tokenizador es una decisión de ingeniería separable de la arquitectura**.

### 9.3 GPT-2/GPT-3 byte-level BPE

GPT-2 (clase) y GPT-3 (clase, mencionado como hito) usan byte-level BPE, **no** SentencePiece. Pero el principio es el mismo: vocabulario subword aprendido del corpus.

### 9.4 ChatGPT y RLHF

ChatGPT, GPT-4 y la familia OpenAI usan `tiktoken`, una librería custom de OpenAI que implementa byte-level BPE optimizado en Rust. Para tareas downstream con la API de OpenAI, el tokenizador es **invisible** al usuario — solo se ven los tokens en métricas de billing. Pero conceptualmente sigue siendo el mismo principio.

### 9.5 Modelos open-source post-LLaMA

Para la práctica de la clase (que cubre LLaMA, Mistral, Gemma), todos usan SentencePiece. Cualquier ejercicio de fine-tuning o inferencia con estos modelos en HuggingFace requiere `pip install sentencepiece`.

### 9.6 El rol de SentencePiece en la "stack moderna"

La stack típica para producción de un modelo open-source post-2023 es:

```
texto crudo
    ↓ SentencePiece (sp.encode)
IDs de tokens
    ↓ embedding lookup
embeddings
    ↓ Transformer decoder (LLaMA, Mistral, Gemma)
logits
    ↓ argmax o sampling
ID de token de output
    ↓ SentencePiece (sp.decode)
texto generado
```

SentencePiece está en **ambos extremos** de la inferencia. Es la pieza más estable de toda la stack — los modelos cambian, las arquitecturas evolucionan, pero el tokenizador (`tokenizer.model`) típicamente se congela en el momento del pre-training y nunca cambia.

---

## 10. Conexión con el lab 20

El laboratorio 20 del Diplomado IA UC ejecuta una sesión hands-on de tokenización y modelos pre-entrenados de HuggingFace. La estructura típica es:

1. Celdas iniciales: instalación de dependencias.
2. Celda 8: `pip install sentencepiece`.
3. Celdas 9-11: carga de tokenizadores BERT (WordPiece) y comparación de outputs.
4. Celda 12: carga de XLNetTokenizer (que requiere SentencePiece).
5. Celdas siguientes: comparación de segmentaciones entre BERT y XLNet sobre el mismo input.

### 10.1 Por qué la celda 8 es necesaria

Sin la celda 8, la celda 12 (`from transformers import XLNetTokenizer; tok = XLNetTokenizer.from_pretrained("xlnet-base-cased")`) produce el `ImportError` documentado en la sección 8. El alumno ve un error críptico que no entiende, pierde tiempo googleando, y solo después de instalar `sentencepiece` la celda funciona. La celda 8 previene esto.

### 10.2 Ejemplo concreto de comparación

Para el input `"Hello, my dog is cute"`:

```python
from transformers import BertTokenizer, XLNetTokenizer

bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")
xlnet_tok = XLNetTokenizer.from_pretrained("xlnet-base-cased")

text = "Hello, my dog is cute"

print(bert_tok.tokenize(text))
# ['hello', ',', 'my', 'dog', 'is', 'cute']

print(xlnet_tok.tokenize(text))
# ['▁Hello', ',', '▁my', '▁dog', '▁is', '▁cute']
```

Observaciones para discusión en clase:

- **BERT (WordPiece)**: no marca el inicio de palabra. La detokenización es heurística (asume espacio entre tokens salvo si empiezan con `##`).
- **XLNet (SentencePiece Unigram)**: marca explícitamente el inicio de palabra con `▁`. La detokenización es trivial: `''.join(tokens).replace('▁', ' ')`.
- **BERT lowercased**: `Hello` → `hello`. XLNet cased: preserva caso.
- **Reversibilidad**: XLNet es perfectamente reversible al texto normalizado. BERT no preserva información de espacios consecutivos o algunas variantes de puntuación.

### 10.3 Demostración de subword segmentation

Para una palabra rara como `tokenization`:

```python
print(bert_tok.tokenize("tokenization"))
# ['token', '##ization']

print(xlnet_tok.tokenize("tokenization"))
# ['▁token', 'ization']
```

Ambos producen 2 tokens. La diferencia es solo la convención de marcado (`##` continuación vs `▁` inicio).

Para una palabra muy rara o inventada:

```python
print(xlnet_tok.tokenize("supercalifragilisticexpialidocious"))
# ['▁super', 'cal', 'if', 'rag', 'ilist', 'ic', 'ex', 'pi', 'ali', 'do', 'cious']
```

Muestra cómo Unigram descompone una palabra desconocida en subwords reconocibles. La segmentación es **determinista** (Viterbi) — siempre produce este output para este input.

### 10.4 Demostración de sampling (extensión opcional)

Si el lab quiere extender, se puede ilustrar subword regularization:

```python
import sentencepiece as spm

# Asumiendo que se tiene el .model de XLNet
sp = spm.SentencePieceProcessor()
sp.Load("xlnet-base-cased.model")  # típicamente disponible internamente

for _ in range(5):
    print(sp.SampleEncodeAsPieces("interesting", -1, 0.3))
```

Mostraría diferentes segmentaciones del mismo input — la base teórica del Unigram LM en acción.

### 10.5 Implicaciones para el alumno

La conexión que conviene resaltar:

1. SentencePiece no es solo "una librería más a instalar" — es **el tokenizador estándar de facto** para modelos post-BERT.
2. La diferencia entre WordPiece y SentencePiece es marginal en performance, pero relevante en convenciones (`##` vs `▁`).
3. Para usar cualquier modelo de T5/XLNet/LLaMA/Mistral, `sentencepiece` es prerrequisito. Para usar BERT no.
4. La elección del tokenizador es **load-bearing** en el comportamiento del modelo. Tokenizadores distintos producen vocabularios distintos, y por tanto modelos distintos.

---

## 11. Lecturas complementarias

- **Sennrich et al. 2016** ("Neural Machine Translation of Rare Words with Subword Units") — paper canónico de BPE en NLP.
- **Kudo 2018 ACL** ("Subword Regularization") — paper hermano, introduce Unigram LM.
- **Provilkov et al. 2020** ("BPE-Dropout") — respuesta de BPE a subword regularization.
- **Bostrom & Durrett 2020** ("Byte Pair Encoding is Suboptimal for Language Model Pretraining") — argumento empírico de que Unigram es mejor que BPE para LM pre-training.
- **Wu et al. 2016** ("Google's Neural Machine Translation System") — paper original donde se introduce WordPiece para NMT a escala.
- **Schuster & Nakajima 2012** ("Japanese and Korean Voice Search") — paper original de WordPiece, varios años antes de BERT.
- **Mielke et al. 2021** ("Between words and characters: A Brief History of Open-Vocabulary Modeling and Tokenization in NLP") — survey histórico completo de tokenización subword.
- **Repo de SentencePiece**: `https://github.com/google/sentencepiece` — README con ejemplos completos.

---

## 12. Notas para integrar al site

Cosas que conviene exponer en el material público del site:

1. **Diagrama del flujo SentencePiece**: raw text → Normalizer (FST) → Trainer (EM + pruning) → Encoder (Viterbi) → IDs → Decoder → text. Útil como diagrama mermaid en el fundamento de tokenización subword.
2. **Tabla BPE vs WordPiece vs Unigram**: la de sección 6 directamente.
3. **Tabla de adopción industrial**: la de sección 7.1 ayuda al alumno a entender por qué `pip install sentencepiece` aparece tanto.
4. **Demostración de comparación BERT vs XLNet**: el ejemplo de sección 10.2 es perfecto para el `fundamentos/tokenizacion-subword.md` o como celda extra del notebook del lab.
5. **Explicación del símbolo `▁`**: muchos alumnos lo ven y no saben qué es. Vale dedicarle un párrafo en el material del lab.
6. **Distinción algoritmo vs herramienta**: SentencePiece la herramienta soporta múltiples algoritmos. "Usar SentencePiece" no significa "usar Unigram" — significa "usar la librería" (que puede correr en modo BPE o Unigram).
7. **Conexión con subword regularization**: explicar por qué existe esta capacidad aunque la mayoría de modelos no la usen — es un ejemplo de cómo decisiones de diseño pueden tener vidas más largas que su propósito original.

---

## 13. Resumen ejecutivo

SentencePiece (Kudo & Richardson 2018) es una librería open-source para tokenización subword que opera sobre texto Unicode crudo, sin pre-tokenización a nivel palabra. Implementa dos algoritmos: BPE (Sennrich 2016) y Unigram Language Model (Kudo 2018). Su contribución técnica central es la **lossless tokenization** vía el meta-símbolo `▁` para espacios, que permite reversibilidad perfecta y operación en cualquier idioma (incluidos CJK sin espacios).

El algoritmo Unigram LM modela cada token como una variable independiente con probabilidad $P(v)$, y aprende el vocabulario via EM más pruning iterativo: inicializar con un vocabulario muy grande, ejecutar EM para optimizar probabilidades, y eliminar iterativamente los tokens menos informativos hasta alcanzar el tamaño objetivo. Decoder Viterbi para inferencia, sampling top-$k$ para subword regularization.

Adopción: T5, XLNet, ALBERT, mBART, mT5, LLaMA, LLaMA 2, Mistral, Gemma y la mayoría de modelos open-source post-BERT usan SentencePiece. Por eso `pip install sentencepiece` es dependencia ubicua en cualquier notebook de HuggingFace que cargue estos modelos. BERT usa WordPiece (relacionado pero distinto); GPT-2/3/4 usan byte-level BPE.

Para la clase 20 del Diplomado, SentencePiece es la pieza invisible que conecta el texto humano con todos los modelos modernos. Para el lab 20, es la dependencia que permite que la celda 12 (XLNetTokenizer) no explote.
