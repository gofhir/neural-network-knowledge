---
title: "GloVe - Global Vectors for Word Representation"
weight: 250
math: true
---

{{< paper-card
    title="GloVe: Global Vectors for Word Representation"
    authors="Pennington, Socher, Manning"
    year="2014"
    venue="EMNLP 2014, Doha"
    pdf="/papers/glove-pennington-2014.pdf" >}}
La respuesta de Stanford NLP a Word2Vec. **Unifica las dos tradiciones** de word embeddings: metodos basados en conteos globales (LSA, HAL, PPMI) y metodos basados en ventanas locales (Word2Vec). GloVe entrena embeddings cuyo producto punto aproxima el **log de co-ocurrencia global**, via una loss cuadratica ponderada elegantemente derivada desde el ratio $P_{ik}/P_{jk}$.
{{< /paper-card >}}

---

## Contexto

En 2014 habia dos comunidades:

- **Count-based** (LSA Deerwester 1990, HAL Lund 1996, PPMI Bullinaria 2007): usan la matriz de co-ocurrencia completa pero pesan mal las celdas (palabras frecuentes dominan, palabras raras se diluyen).
- **Predict-based** (Word2Vec): usan ventanas locales y nunca ven la estadistica agregada.

Pennington diagnostica que ambas atacan el mismo problema desde lados opuestos. GloVe propone entrenar **directamente sobre la matriz de co-ocurrencia** con una loss bien disenada.

---

## Ideas principales

### 1. La observacion clave: el ratio $P_{ik}/P_{jk}$

Tabla 1 del paper con corpus de 6B tokens:

| | $k=$ solid | $k=$ gas | $k=$ water | $k=$ fashion |
|---|---|---|---|---|
| $P(k \mid \text{ice})$ | $1.9 \times 10^{-4}$ | $6.6 \times 10^{-5}$ | $3.0 \times 10^{-3}$ | $1.7 \times 10^{-5}$ |
| $P(k \mid \text{steam})$ | $2.2 \times 10^{-5}$ | $7.8 \times 10^{-4}$ | $2.2 \times 10^{-3}$ | $1.8 \times 10^{-5}$ |
| **Ratio** | **8.9** | **0.085** | **1.36** | **0.96** |

El **ratio** distingue palabras relevantes (solid -> ice, gas -> steam) de palabras irrelevantes (water y fashion, ratio ~1). El ruido se cancela en el ratio.

### 2. Derivacion en 5 pasos

1. **Postulado**: $F(\mathbf{w}_i, \mathbf{w}_j, \tilde{\mathbf{w}}_k) = P_{ik}/P_{jk}$.
2. **Diferencia vectorial**: $F((\mathbf{w}_i - \mathbf{w}_j), \tilde{\mathbf{w}}_k) = P_{ik}/P_{jk}$.
3. **Producto punto** para que ambos lados sean escalares: $F((\mathbf{w}_i - \mathbf{w}_j)^T \tilde{\mathbf{w}}_k) = P_{ik}/P_{jk}$.
4. **Simetria** (homomorfismo $(\mathbb{R}, +) \to (\mathbb{R}_{>0}, \times)$): la unica solucion es $F = \exp$. Sustituyendo: $\mathbf{w}_i^T \tilde{\mathbf{w}}_k = \log X_{ik} - \log X_i$.
5. **Absorber $\log X_i$** en bias $b_i$ y agregar bias $\tilde{b}_k$ por simetria.

Resultado:

$$\mathbf{w}_i^T \tilde{\mathbf{w}}_k + b_i + \tilde{b}_k = \log X_{ik}$$

### 3. Funcion objetivo

$$\boxed{\mathcal{J} = \sum_{i,j=1}^{V} f(X_{ij}) \left( \mathbf{w}_i^T \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2}$$

Least squares ponderada por una funcion $f$ de la co-ocurrencia.

### 4. La funcion de peso $f$

$$f(x) = \begin{cases} (x/x_{\max})^\alpha & \text{si } x < x_{\max} \\ 1 & \text{si } x \ge x_{\max} \end{cases}$$

Con $x_{\max} = 100$ y $\alpha = 3/4$. Tres desiderata:
1. $f(0) = 0$ (los ceros no contribuyen).
2. $f$ no-decreciente (pares raros no dominan).
3. $f$ acotada (pares frecuentes tampoco dominan).

**Nota**: el mismo exponente $3/4$ aparece en negative sampling de Word2Vec -- una constante "magica" robusta de las estadisticas Zipf.

### 5. Embedding final

$$\mathbf{w}_{\text{final}} = \mathbf{w} + \tilde{\mathbf{w}}$$

Se promedian ambas matrices por simetria.

---

## Resultados experimentales

### Word analogies (Tabla 2)

| Modelo | Dim | Size | Sem. | Syn. | Total |
|---|---|---|---|---|---|
| ivLBL | 100 | 1.5B | 55.9 | 50.1 | 53.2 |
| **GloVe** | 100 | 1.6B | **67.5** | **54.3** | **60.3** |
| Skip-gram | 300 | 1B | 61 | 61 | 61 |
| **GloVe** | 300 | 1.6B | **80.8** | 61.5 | **70.3** |
| Skip-gram | 300 | 6B | 73.0 | 66.0 | 69.1 |
| **GloVe** | 300 | 6B | 77.4 | 67.0 | **71.7** |
| **GloVe** | 300 | 42B | **81.9** | **69.3** | **75.0** |

GloVe domina en analogias semanticas; competitivo con SGNS en sintacticas.

### Complejidad

GloVe escala como $O(|C|^{0.8})$ vs Skip-gram $O(|C|)$ -- ventaja para corpora grandes.

### Embeddings preentrenados publicados

| Nombre | Corpus | Vocab | Dim |
|---|---|---|---|
| `glove.6B` | Wikipedia + Gigaword 5 (6B) | 400k | 50/100/200/300 |
| `glove.42B.300d` | Common Crawl uncased (42B) | 1.9M | 300 |
| `glove.840B.300d` | Common Crawl cased (840B) | 2.2M | 300 |
| `glove.twitter.27B` | Twitter (27B) | 1.2M | 25/50/100/200 |

Disponibles en https://nlp.stanford.edu/projects/glove/.

---

## Limitaciones

1. **Memoria intensiva**: la matriz $X$ ocupa TB para corpora Common Crawl. Stanford soluciona con streaming en C.
2. **OOV no manejado**: igual que W2V. FastText resolveria.
3. **Embedding no contextual**: un unico vector por palabra.
4. **Sin subwords**: morfologia no se captura.
5. **Ventana fija**: no dependencias largas.
6. **Asume estacionariedad**: $P(j \mid i)$ estable en el corpus.

---

## Relacion con Skip-gram

GloVe se puede derivar como **Skip-gram con cross-entropy reemplazada por least squares de logaritmos**:

$$\hat{\mathcal{J}} = \sum_{i,j} X_i \cdot (\log P_{ij} - \log Q_{ij})^2 \approx \sum_{i,j} X_i \cdot (\mathbf{w}_i^T \tilde{\mathbf{w}}_j - \log X_{ij})^2$$

Generalizando el peso $X_i$ a $f(X_{ij})$ se obtiene la formula de GloVe.

[Levy & Goldberg 2014](/papers/sgns-implicit-mf-levy-goldberg-2014) demuestran ademas que **SGNS factoriza implicitamente PMI shifted**, conectando ambos paradigmas.

---

## Por que importa hoy

GloVe se convirtio en el **segundo estandar** de embeddings preentrenados (junto con Word2Vec) durante 2014-2018:
- Inicializacion default en RNN/CNN para NLP.
- Baseline obligatorio en cualquier paper de la era.
- Aplicaciones cross-lingual via alineacion de espacios GloVe.

Sus dos contribuciones conceptuales sobreviven:

1. **Factorizacion log-bilineal**: $\mathbf{w}^T \mathbf{u} \approx \log f(\text{count})$ aparece en knowledge graph embeddings (TransE), recsys, etc.
2. **Funcion de peso saturada**: idea de pesar diferencialmente raros vs frecuentes precede focal loss (Lin 2017).

Hoy es eclipsado por embeddings contextuales (BERT, GPT) pero sobrevive en proyectos low-resource y como baseline.

---

## Notas y enlaces

- **Codigo**: https://github.com/stanfordnlp/GloVe (C optimizado).
- **Embeddings preentrenados**: https://nlp.stanford.edu/projects/glove/.
- **Predecesor**: [Word2Vec Distributed](/papers/word2vec-distributed-mikolov-2013).
- **Conexion teorica**: [Levy & Goldberg - SGNS as Implicit MF](/papers/sgns-implicit-mf-levy-goldberg-2014).
- **Clase asociada**: [Clase 18 - Modelos de lenguaje, Word2Vec, GloVe y SkipThought](/clases/clase-18).
- **Fundamentos relacionados**: [GloVe](/fundamentos/glove), [Embeddings distribuidos](/fundamentos/embeddings-distribuidos).
- **Cita BibTeX**:

```bibtex
@inproceedings{pennington2014glove,
  title={GloVe: Global Vectors for Word Representation},
  author={Pennington, Jeffrey and Socher, Richard and Manning, Christopher D},
  booktitle={EMNLP},
  pages={1532--1543},
  year={2014}
}
```
