---
title: "Bahdanau Attention (NMT)"
weight: 200
math: true
---

{{< paper-card
    title="Neural Machine Translation by Jointly Learning to Align and Translate"
    authors="Bahdanau, Cho, Bengio"
    year="2015"
    venue="ICLR 2015"
    pdf="/papers/bahdanau-attention-2015.pdf"
    arxiv="1409.0473" >}}
Introduce el **mecanismo de atencion** -- la idea que rompio el cuello de botella del vector de contexto fijo en Seq2Seq, habilito la traduccion neural practica de oraciones largas, y sembro las semillas del Transformer. El paper propone una extension al encoder-decoder donde el decoder "busca adaptativamente" (soft-search) en la oracion fuente para cada palabra a traducir, en lugar de depender de un solo vector comprimido.
{{< /paper-card >}}

---

## Contexto

A fines de 2014, **Sutskever 2014** (Seq2Seq) y **Cho 2014** (RNN encoder-decoder con GRU) habian mostrado que redes neuronales podian traducir end-to-end, igualando (casi) a sistemas estadisticos. Pero tenian un problema comun: comprimir toda la oracion fuente en un **vector fijo $c$** limitaba severamente el rendimiento en oraciones largas (>30-40 palabras).

Bahdanau, Cho y Bengio (Montreal/Jacobs U.) presentaron en ICLR 2015 una solucion elegante: eliminar el cuello de botella dejando que el decoder mire **todos** los estados del encoder con pesos adaptativos. Este paper introdujo el termino y la matematica de **attention** como hoy lo entendemos en deep learning.

---

## Ideas principales

### 1. Encoder bidireccional

En lugar de un LSTM unidireccional, usan una **BiRNN** (GRU) que produce anotaciones por posicion:

$$h_j = [\overrightarrow{h_j}; \overleftarrow{h_j}]$$

Cada $h_j$ resume la oracion entera **centrada en la palabra $j$** (tanto el pasado como el futuro estan representados).

### 2. Context vector adaptativo

En lugar de un unico $c$, se calcula un $c_i$ **distinto para cada paso del decoder**:

$$c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j$$

donde $\alpha_{ij}$ es la probabilidad de que la palabra target $i$ se **alinee** con la palabra source $j$.

### 3. Alignment model (additive attention)

Los pesos $\alpha_{ij}$ se calculan via softmax sobre scores producidos por una red feedforward pequeña:

$$e_{ij} = a(s_{i-1}, h_j) = V^T \tanh(W s_{i-1} + U h_j)$$

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})}$$

El alignment model $a$ es **entrenado jointly** con el resto del modelo, solo con la senal de cross-entropy sobre las palabras target. **No requiere supervision explicita de alineamiento** -- el modelo lo aprende como subproducto de traducir bien.

### 4. Decoder actualizado

$$p(y_i \mid y_{<i}, x) = g(y_{i-1}, s_i, c_i)$$

$$s_i = f(s_{i-1}, y_{i-1}, c_i)$$

El estado del decoder ahora depende del context vector $c_i$ adaptado, no solo del vector $c$ global.

### 5. Soft alignment, diferenciable

Crucialmente, $\alpha_{ij}$ es una distribucion continua (softmax), no una asignacion hard. Esto permite backpropagation estandar -- no se necesita REINFORCE ni variational methods. Training completamente end-to-end con SGD.

---

## Resultados experimentales

Task: WMT'14 English-French (348M words, ~30K vocabulario).

| Modelo | BLEU (todas las oraciones) | BLEU (solo palabras conocidas) |
|---|---|---|
| Moses (SMT, baseline industrial) | 33.30 | 35.63 |
| RNNencdec-30 (Cho 2014, train ≤30 words) | 13.93 | 19.97 |
| RNNencdec-50 (train ≤50 words) | 17.82 | 26.71 |
| **RNNsearch-30 (este paper)** | 21.50 | 31.44 |
| **RNNsearch-50** | **26.75** | **34.16** |

Notable:

- RNNsearch **duplica** BLEU sobre RNNencdec.
- RNNsearch-50 **compite con Moses** (SMT industrial entrenado con 418M palabras monolinguales adicionales).
- RNNsearch-30 (entrenado en oraciones cortas) **supera a RNNencdec-50** (entrenado en mas datos).

### Robustez a oraciones largas

Grafica clave (Figura 2): BLEU vs longitud de oracion.

- RNNencdec-50 cae de ~25 → ~10 para oraciones de 50+ palabras.
- RNNsearch-50 **se mantiene constante** ~26 a lo largo de toda la distribucion.

Exactamente lo que el mecanismo de atencion buscaba resolver.

### Alignments visualizados

La Figura 3 muestra matrices de atencion $\alpha_{ij}$ para pares ingles-frances. Los patrones son revealadores:

- Diagonal clara cuando el orden es similar: "The economic growth" ↔ "La croissance economique" (con inversion local adjective-noun reflejada como cross en la matriz).
- Soft alignments que capturan reordenings que un alineamiento hard no podria.
- Visualizacion interpretable: se puede "leer" que parte del ingles tradujo cada palabra francesa.

---

## Por que importa hoy

### Impacto inmediato (2015-2016)
- Cambio el paradigma de NMT de **fixed-length bottleneck** a **attention-based**.
- Luong et al. 2015 exploro variantes (local, global, dot-product).
- Google GNMT (2016) escalo attention a produccion con 16 layers + deep attention.

### Impacto a largo plazo (2017+)
- El Transformer (Vaswani et al. 2017) tomo la idea al extremo: **reemplazar completamente** las RNNs por attention, y aplicar **self-attention** dentro del encoder y decoder.
- BERT, GPT, Claude, Gemini, LLaMA -- todos son arquitecturas de attention sin recurrencia. Todos descienden conceptualmente de este paper.

### Lecciones transferibles
- **Soft alignment** como tecnica general: si tienes que combinar elementos de un conjunto segun relevancia, softmax + dot-product es frecuentemente la respuesta.
- **Joint training sin supervision explicita**: el modelo aprende estructura interpretable solo del error final.
- **Diferenciabilidad de extremo a extremo** es una virtud extrema -- evita REINFORCE y otros metodos menos estables.

---

## Limitaciones

- **Complejidad cuadratica** $O(T_x \cdot T_y)$ en tiempo y memoria -- para secuencias muy largas se vuelve costoso.
- **Secuencial** en el decoder: cada $s_i$ depende de $s_{i-1}$, no paralelizable. El Transformer resolveria esto.
- **Attention weights como explicacion**: estudios posteriores (Jain & Wallace 2019) muestran que las visualizaciones de atencion **no necesariamente reflejan la contribucion causal** de cada posicion.
- **Encoder bidireccional**: limita el uso a tareas offline -- no se puede hacer streaming con BiRNN.

---

## Notas y enlaces

- El paper tiene una estructura muy clara: Seccion 2 (background NMT), Seccion 3 (propuesta), Seccion 4-5 (experimentos). Lectura recomendada de 10 paginas.
- La **Figura 1** muestra la arquitectura global; la **Figura 3** es la visualizacion de alignments que se volvio icónica.
- Codigo original: [github.com/lisa-groundhog/GroundHog](https://github.com/lisa-groundhog/GroundHog).
- Follow-ups directos:
  - **Luong et al. 2015** "Effective Approaches to Attention-based Neural Machine Translation" -- variantes dot-product.
  - **Vaswani et al. 2017** "Attention Is All You Need" -- eliminar RNNs.
  - **Xu et al. 2015** [Show, Attend and Tell](/papers/show-attend-tell-xu-2015) -- aplicar atencion a captioning.

Ver fundamentos: [Mecanismo de Atencion](/fundamentos/mecanismo-atencion) · [Sequence to Sequence](/fundamentos/seq2seq) · [Redes Recurrentes](/fundamentos/redes-recurrentes).
