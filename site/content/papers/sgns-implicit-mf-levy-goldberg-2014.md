---
title: "SGNS as Implicit Matrix Factorization"
weight: 251
math: true
---

{{< paper-card
    title="Neural Word Embedding as Implicit Matrix Factorization"
    authors="Levy, Goldberg"
    year="2014"
    venue="NeurIPS 2014"
    pdf="/papers/sgns-implicit-mf-levy-goldberg-2014.pdf" >}}
Demuestra formalmente que **Skip-gram with Negative Sampling (SGNS) es una factorizacion implicita de la matriz PMI shifted**. Unifica las dos tradiciones de word embeddings -- count-based (LSA, PPMI) y predict-based (Word2Vec) -- cerrando una brecha conceptual de dos decadas. Propone tambien **SPPMI** (Shifted Positive PMI), un metodo de embeddings basado solo en conteos que rivaliza con SGNS en tareas downstream.
{{< /paper-card >}}

---

## Contexto

Antes de este paper, habia dos comunidades separadas:

- **Count-based** (Harris 1954, Church-Hanks 1990, Deerwester 1990): construir matriz de asociacion (PMI, PPMI, log-count), aplicar SVD.
- **Predict-based** (Bengio 2003, Mikolov 2013): entrenar red neuronal a predecir palabras, obtener embeddings como subproducto.

Baroni 2014 ("Don't count, predict!") habia mostrado empiricamente que predict-based gana sistematicamente sobre count-based. Pero **nadie sabia por que** -- Word2Vec se presentaba con motivacion operacional, no teorica.

---

## Ideas principales

### 1. Demostracion: SGNS factoriza PMI shifted

El objetivo SGNS global, sumando sobre todos los pares observados $(w, c)$:

$$\ell = \sum_{w \in V_W} \sum_{c \in V_C} \#(w, c) \left( \log \sigma(\mathbf{w} \cdot \mathbf{c}) + k \cdot \mathbb{E}_{c_N \sim P_D} [\log \sigma(-\mathbf{w} \cdot \mathbf{c}_N)] \right)$$

Maximizando como funcion de $x = \mathbf{w} \cdot \mathbf{c}$ y resolviendo $\partial \ell / \partial x = 0$:

$$e^x = \frac{\#(w, c) \cdot |D|}{\#(w) \cdot \#(c) \cdot k}$$

Tomando log:

$$\boxed{\mathbf{w} \cdot \mathbf{c} = \log \frac{\#(w, c) \cdot |D|}{\#(w) \cdot \#(c)} - \log k = \text{PMI}(w, c) - \log k}$$

**Conclusion**: SGNS factoriza la matriz $M^{\text{SGNS}}_{ij} = \text{PMI}(w_i, c_j) - \log k$.

### 2. NCE factoriza una matriz distinta

Por la misma derivacion, NCE (Noise Contrastive Estimation) factoriza:

$$M^{\text{NCE}}_{ij} = \log P(w_i \mid c_j) - \log k$$

Diferente de PMI -- explica por que SGNS funciona mejor que NCE en tareas downstream.

### 3. Factorizacion ponderada vs SVD uniforme

SGNS no es equivalente a SVD truncado de PMI. SGNS factoriza con **peso $\#(w,c)$** -- pares observados muchas veces tienen mas influencia. Esto se ve en la loss local:

$$\ell(w, c) = \#(w, c) \log \sigma(\mathbf{w} \cdot \mathbf{c}) + k \cdot \#(w) \cdot \frac{\#(c)}{|D|} \log \sigma(-\mathbf{w} \cdot \mathbf{c})$$

Es **factorizacion matricial ponderada** estilo Koren 2009 (recsys).

### 4. SPPMI (Shifted Positive PMI)

Inspirados por el resultado: si SGNS factoriza $\text{PMI} - \log k$, ¿por que no usar **directamente** una matriz sparse con esos valores?

$$\text{SPPMI}_k(w, c) = \max(\text{PMI}(w, c) - \log k, 0)$$

Sin redes neuronales, sin entrenamiento iterativo -- solo conteos + PMI + shift + truncamiento.

### 5. SVD sobre SPPMI

Alternativo: aplicar SVD truncado a SPPMI para obtener vectores densos:

$$W^{\text{SVD}_{1/2}} = U_d \cdot \sqrt{\Sigma_d}, \quad C^{\text{SVD}_{1/2}} = V_d \cdot \sqrt{\Sigma_d}$$

Embedding **simetricos** (no $W = U_d \cdot \Sigma_d$) por analogia con la simetria de W2V.

---

## Resultados experimentales

### Optimizacion del objetivo (Tabla 1)

Porcentaje de desviacion del optimo:

| $k$ | SPPMI | SVD $d=100$ | SVD $d=500$ | SGNS $d=100$ | SGNS $d=500$ |
|---|---|---|---|---|---|
| 1 | 0.00009% | 26.1% | 25.2% | 31.4% | 29.4% |
| 5 | 0.00004% | 95.8% | 95.1% | 39.3% | 36.0% |
| 15 | 0.00002% | 266% | 266% | 7.80% | 6.37% |

**Lectura**: SPPMI es **casi optimo**. SVD truncado es muy malo con $k$ grande. SGNS se aproxima con $d$ grande.

### Word similarity (Tabla 2)

| Rep | $k$ | WS353 | MEN | Mixed Analogies | Syntactic Analogies |
|---|---|---|---|---|---|
| SPPMI | 5 | **0.691** | **0.735** | 0.655 | 0.466 |
| **SGNS** ($d=1000$) | 5 | 0.666 | 0.716 | **0.616** | **0.619** |
| SVD ($d=1000$) | 1 | 0.652 | 0.708 | 0.644 | 0.567 |

- **Word similarity**: SPPMI gana o empata SGNS.
- **Analogias sintacticas**: SGNS domina (factorizacion ponderada favorece palabras funcion frecuentes).

### Conjetura: por que SGNS gana en sintaxis

Las analogias sintacticas (`good:better :: smart:smarter`) dependen de palabras funcion ("the", "many") que son **muy frecuentes**. La weighted factorization de SGNS las favorece; SVD las trata uniformemente.

---

## Implicaciones

### 1. Word2Vec no es "magico"

Word2Vec no aprende algo cualitativamente distinto de SVD sobre PMI -- aprende algo **cuantitativamente** mejor por la factorizacion ponderada. La "magia" es el peso, no la red neuronal.

### 2. PMI es la moneda central

Cualquier word embedding razonable es alguna proyeccion de baja-rango de una matriz tipo PMI. Lo que importa son **los pesos**.

### 3. Lo simple gana cuando se entiende

SPPMI (4 lineas de codigo) rivaliza con SGNS (cientos de lineas optimizadas) en muchas tareas, una vez que sabemos **que matriz factorizar**.

---

## Limitaciones

1. **Aproximacion**: asume $d$ suficiente para reconstruccion perfecta. En la practica $d = 300 \ll |V|$.
2. **Solo SGNS**: hierarchical softmax tiene estructura distinta no cubierta.
3. **Ignora exponente 3/4**: el analisis usa $P_n = U$ unigrama pura. Con $U^{3/4}$ la PMI se generaliza pero los resultados cualitativos no cambian.

---

## Por que importa hoy

Este paper inicio una linea de **analisis empirico-teorico de embeddings** que continua en 2026:

- **Arora 2016** (RAND-WALK): justificacion teorica de PMI shifted desde primeros principios.
- **Hashimoto 2016**: extension a sentence/document embeddings.
- **Allen & Hospedales 2019** (Analogies Explained): usa Levy-Goldberg como punto de partida para explicar analogias.
- **Mu & Viswanath 2018**: propiedades geometricas (anisotropia) interpretadas via PMI.

Levy y Goldberg se convirtieron en figuras centrales del analisis de embeddings; Goldberg escribio el libro estandar "Neural Network Methods for NLP" (2017).

---

## Notas y enlaces

- **Codigo**: https://bitbucket.org/yoavgo/word2vecf -- una variacion de word2vec que recibe pares pre-extraidos.
- **Predecesores**: [Word2Vec Distributed](/papers/word2vec-distributed-mikolov-2013), [GloVe](/papers/glove-pennington-2014).
- **Sucesor teorico**: [Allen & Hospedales - Analogies Explained](/papers/analogies-explained-allen-hospedales-2019).
- **Clase asociada**: [Clase 18 - Modelos de lenguaje, Word2Vec, GloVe y SkipThought](/clases/clase-18).
- **Fundamentos relacionados**: [Word2Vec](/fundamentos/word2vec), [Embeddings distribuidos](/fundamentos/embeddings-distribuidos).
- **Cita BibTeX**:

```bibtex
@inproceedings{levy2014neural,
  title={Neural word embedding as implicit matrix factorization},
  author={Levy, Omer and Goldberg, Yoav},
  booktitle={Advances in Neural Information Processing Systems},
  volume={27},
  year={2014}
}
```
