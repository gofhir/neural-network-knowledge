---
title: "Word2Vec - Distributed Representations & Negative Sampling"
weight: 249
math: true
---

{{< paper-card
    title="Distributed Representations of Words and Phrases and their Compositionality"
    authors="Mikolov, Sutskever, Chen, Corrado, Dean"
    year="2013"
    venue="NeurIPS 2013" 
    pdf="/papers/word2vec-distributed-mikolov-2013.pdf" >}}
Extiende el primer paper de Word2Vec con cuatro tecnicas que **hicieron a Word2Vec practico a escala**: (1) **negative sampling** -- alternativa simple al softmax exacto; (2) **subsampling** de palabras frecuentes; (3) **hierarchical softmax** con arbol Huffman; (4) **phrase embeddings** para "New York", "Air Canada", etc. Permite entrenar Skip-gram en 30B palabras con vocabulario 700k+ en un dia.
{{< /paper-card >}}

---

## Contexto

El primer paper de Word2Vec (Mikolov 2013a) habia introducido CBoW y Skip-gram. Pero el **softmax exacto** sobre $|V| = 10^6$ seguia siendo el bottleneck: $O(|V| \cdot N)$ por ejemplo. Este paper resuelve ese problema y agrega manejo de frases idiomaticas.

---

## Ideas principales

### 1. Negative Sampling (SGNS) -- el aporte mas influyente

Reemplaza el softmax completo por **clasificacion binaria**: distinguir el par real $(w, c)$ de pares "falsos" $(w, w_{\text{neg}})$ con $w_{\text{neg}}$ muestreado de una distribucion de ruido $P_n$.

$$\mathcal{L}_{\text{SGNS}} = \log \sigma(\mathbf{v}'_{w_O} \cdot \mathbf{v}_{w_I}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n} \left[ \log \sigma(-\mathbf{v}'_{w_i} \cdot \mathbf{v}_{w_I}) \right]$$

- Primer termino: el par real debe tener producto punto **alto**.
- Segundo termino: cada uno de los $k$ negativos debe tener producto punto **bajo**.
- $k$: 5-20 para datasets pequenos, 2-5 para grandes.

**Costo**: $O((k+1) \cdot N)$ por ejemplo -- **independiente de $|V|$**. Speedup ~$10^5$ vs softmax exacto.

### 2. El famoso exponente $3/4$

El paper investiga distintas $P_n(w)$ y descubre empiricamente:

$$P_n(w) \propto U(w)^{3/4}$$

donde $U(w)$ es la frecuencia unigrama. Sin justificacion teorica conocida -- pero **funciona consistentemente mejor** que uniforme o $U(w)$ puro.

### 3. Subsampling de palabras frecuentes

Cada ocurrencia de $w_i$ se descarta con probabilidad:

$$P_{\text{discard}}(w_i) = 1 - \sqrt{t / f(w_i)}, \quad t \approx 10^{-5}$$

Palabras como "the" ($f = 0.07$) se descartan **98.8%** de las veces. Resultado: **2-10x speedup + mejora en embeddings de palabras raras**.

### 4. Hierarchical Softmax con arbol Huffman

Organizar las $|V|$ palabras en un **arbol binario** donde palabras frecuentes tienen caminos cortos (Huffman optimo). Cada hoja es una palabra, cada nodo interno tiene un vector $\mathbf{v}'_n$.

$$P(w \mid w_I) = \prod_{j=1}^{L(w)-1} \sigma\left( [\![n(w, j+1) = \text{ch}(n(w,j))]\!] \cdot \mathbf{v}'_{n(w,j)} \cdot \mathbf{v}_{w_I} \right)$$

**Costo**: $O(\log V \cdot N)$ -- exacto (no aproximacion), suma a 1 sobre $V$.

### 5. Phrase embeddings

Detectar frases idiomaticas via score bigrama:

$$\text{score}(w_i, w_j) = \frac{\text{count}(w_i w_j) - \delta}{\text{count}(w_i) \cdot \text{count}(w_j)}$$

Reemplazar bigramas frecuentes por tokens unicos (`New_York`, `Air_Canada`) en 2-4 pasadas con threshold decreciente.

### 6. Composicionalidad aditiva

Observacion sorprendente: `vec(Russia) + vec(river)` esta cerca de `vec(Volga River)`. Explicacion: si $\mathbf{v}_w \cdot \mathbf{v}'_c \approx \log P(c \mid w)$, entonces:

$$(\mathbf{v}_{w_1} + \mathbf{v}_{w_2}) \cdot \mathbf{v}'_c \approx \log [P(c \mid w_1) \cdot P(c \mid w_2)]$$

Las palabras que aparecen en contextos **comunes a ambos $w_1, w_2$** son las que rankean alto. Es el **"AND" semantico**.

---

## Resultados experimentales

### Word analogies (Tabla 1)

| Metodo | Tiempo [min] | Semantico [%] | Sintactico [%] | Total [%] |
|---|---|---|---|---|
| NEG-5 (sin subsampling) | 38 | 54 | 63 | 59 |
| HS-Huffman | 41 | 40 | 53 | 47 |
| NEG-5 + subsampling | **14** | **58** | 61 | **60** |
| NEG-15 + subsampling | 36 | 61 | 61 | **61** |

### Phrase analogies (Tabla 3)

Mejor modelo (HS + 1000d + 33B palabras): **72% accuracy** en analogias de frases. Ejemplos del benchmark:

- `Steve Ballmer : Microsoft :: Larry Page : ?` -> Google
- `Boston : Boston Bruins :: Phoenix : ?` -> Phoenix Coyotes
- `Austria : Austrian Airlines :: Spain : ?` -> Spainair

### Composicionalidad aditiva

| Sum | Vecinos mas cercanos |
|---|---|
| Czech + currency | koruna, Check crown, Polish zolty |
| Vietnam + capital | Hanoi, Ho Chi Minh City |
| German + airlines | airline Lufthansa, carrier Lufthansa |
| Russian + river | Moscow, Volga River, upriver |

---

## Limitaciones

1. **Deteccion de frases es greedy**: heuristica bigrama, pierde frases con sintaxis flexible.
2. **Polisemia ignorada**: "apple" sigue siendo un unico vector.
3. **Sin subwords**: morfologia no se captura.
4. **No analisis teorico**: presentado empiricamente. [Levy & Goldberg 2014](/papers/sgns-implicit-mf-levy-goldberg-2014) lo suplio.
5. **Sesgos sociales**: estudios posteriores (Bolukbasi 2016) mostraron sesgos de genero profundos.

---

## Por que importa hoy

Las **tres ideas** de este paper sobreviven en arquitecturas modernas:

1. **Negative sampling** -> evoluciono a **InfoNCE** (van den Oord 2018), pilar de SimCLR, CLIP, Sentence-BERT, DPR.
2. **Subsampling** -> patron general de **importance sampling** y **curriculum learning**.
3. **Phrase detection** -> motivo a la era de **subword tokenization** (BPE, WordPiece, Unigram) estandar en todo Transformer.

El embedding preentrenado `GoogleNews-vectors-negative300.bin` (1.6 GB, 3M palabras) fue **el segundo "ImageNet de NLP"** despues del primer paper.

---

## Notas y enlaces

- **Codigo**: https://code.google.com/p/word2vec (C optimizado).
- **Predecesor**: [Word2Vec Efficient](/papers/word2vec-efficient-mikolov-2013).
- **Sucesor teorico**: [Levy & Goldberg - SGNS as Implicit MF](/papers/sgns-implicit-mf-levy-goldberg-2014).
- **Sucesor empirico**: [GloVe](/papers/glove-pennington-2014).
- **Clase asociada**: [Clase 18 - Modelos de lenguaje, Word2Vec, GloVe y SkipThought](/clases/clase-18).
- **Laboratorio asociado**: [Lab 18 - Word Embeddings en accion](/laboratorios/lab-18) (Negative Sampling sobre Google News 3M palabras).
- **Fundamentos relacionados**: [Word2Vec](/fundamentos/word2vec), [Embeddings distribuidos](/fundamentos/embeddings-distribuidos).
- **Cita BibTeX**:

```bibtex
@inproceedings{mikolov2013distributed,
  title={Distributed representations of words and phrases and their compositionality},
  author={Mikolov, Tomas and Sutskever, Ilya and Chen, Kai and Corrado, Greg S and Dean, Jeff},
  booktitle={Advances in Neural Information Processing Systems},
  volume={26},
  year={2013}
}
```
