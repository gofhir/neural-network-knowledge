---
title: "Linguistic Regularities in Sparse and Explicit Word Representations (3CosMul)"
weight: 252
math: true
---

{{< paper-card
    title="Linguistic Regularities in Sparse and Explicit Word Representations"
    authors="Levy, Goldberg"
    year="2014"
    venue="CoNLL 2014, pp. 171-180"
    pdf="/papers/linguistic-regularities-levy-goldberg-2014.pdf" >}}
Demuestra que la fórmula de analogías `arg max cos(x, b − a + a*)` de Mikolov es equivalente a una combinación lineal de tres similitudes pairwise (**3CosAdd**), y propone una alternativa multiplicativa más robusta (**3CosMul**) que evita el problema "soft-or" donde un atractor dominante aplasta a los demás. Es la operación que implementa `most_similar_cosmul` de gensim.
{{< /paper-card >}}

---

## Contexto histórico

Mikolov et al. (2013) demostraron empíricamente que los embeddings cumplen la propiedad de analogía `king − man + woman ≈ queen`. Pero **no explicaron por qué funciona** ni propusieron una variante robusta. Levy & Goldberg llenan ese hueco un año después con dos contribuciones:

1. **Demystificación**: la aritmética vectorial es equivalente a sumar similitudes pairwise.
2. **Refinamiento**: la versión multiplicativa **3CosMul** corrige fallos sutiles de la versión aditiva.

---

## Ideas principales

### 1. Equivalencia 3CosAdd

Bajo la asunción habitual de vectores normalizados ($\|v\|=1$), Mikolov resolvía analogías "a:a* :: b:?" con:

$$
b^* = \arg\max_{x} \cos(x, b - a + a^*)
$$

Levy & Goldberg muestran que esto es **matemáticamente equivalente** (Sección 3.3, ecuación 3) a:

$$
b^* = \arg\max_{x \in V} \big[\cos(x, b) - \cos(x, a) + \cos(x, a^*)\big]
$$

Llamado **3CosAdd**. La interpretación humana es: *"encuentra la palabra que es similar a $b$, similar a $a^*$ y diferente de $a$"*. Esto **demystifica** la aritmética vectorial — no es álgebra mágica, es una combinación lineal de tres similitudes pairwise.

### 2. El problema "soft-or" de 3CosAdd

3CosAdd suma tres términos. Una suma lineal exhibe **"soft-or" behavior**: si un término es lo suficientemente grande, **domina la expresión** y aplasta a los otros.

> *"A known property of such linear objectives is that they exhibit a 'soft-or' behavior and allow one sufficiently large term to dominate the expression."* — Sección 6, página 175.

#### Ejemplo canónico de fallo

Para *"London is to England as Baghdad is to ?"*:

| Palabra | ↑ England (atractor) | ↓ London (repulsor) | ↑ Baghdad (atractor) | Suma 3CosAdd |
|---|---|---|---|---|
| **Mosul** | 0.031 | 0.031 | **0.244** | 0.244 |
| **Iraq** | 0.049 | 0.038 | 0.206 | 0.217 |

3CosAdd elige `Mosul` (incorrecto) porque la similitud con `Baghdad` domina, opacando la dimensión "país" que conecta `Iraq` con `England`.

### 3. **3CosMul (fórmula 4 del paper) — la corrección multiplicativa**

Levy & Goldberg proponen reemplazar la suma por multiplicación (Sección 6, ecuación 4):

$$
\boxed{\;\;b^* = \arg\max_{x \in V} \frac{\cos(x, b) \cdot \cos(x, a^*)}{\cos(x, a) + \varepsilon}\;\;}
$$

con $\varepsilon = 0.001$ para evitar división por cero. Tomando logaritmo, esto equivale a **sumar log-similitudes** — el logaritmo amplifica diferencias entre valores pequeños y reduce diferencias entre valores grandes, **balanceando los tres aspectos**.

Aplicado al ejemplo Baghdad: 3CosMul devuelve `Iraq` (correcto) porque la dimensión "país" deja de ser eclipsada por la "geografía".

### 4. Implementación en gensim

3CosMul requiere similitudes no-negativas. Para embeddings densos (donde el coseno puede ser negativo), el paper transforma cosenos a $[0, 1]$:

$$
\cos_+(u, v) = \frac{\cos(u, v) + 1}{2}
$$

(Footnote 7 del paper). Esta es **exactamente la implementación** de `gensim.models.KeyedVectors.most_similar_cosmul`.

---

## Resultados clave

Comparación 3CosAdd vs 3CosMul sobre tres datasets de analogías (Tabla 3):

| Objetivo | Representación | MSR | GOOGLE |
|---|---|---|---|
| 3CosAdd | Embedding (SGNS) | 53.98% | 62.70% |
| 3CosAdd | Explicit (PPMI) | 29.04% | 45.05% |
| **3CosMul** | **Embedding (SGNS)** | **59.09%** | **66.72%** |
| **3CosMul** | **Explicit (PPMI)** | **56.83%** | **68.24%** |

**Dos observaciones**:

1. **3CosMul mejora 3CosAdd consistentemente**: +5% en embeddings densos, +20-27% en representaciones explícitas (PPMI sparse).
2. **Las representaciones explícitas PPMI alcanzan al embedding** con 3CosMul. Esto demuestra que **las analogías no son producto de la red neuronal**, sino que **están latentes en las co-ocurrencias del corpus** y son preservadas (no creadas) por el embedding.

---

## Limitaciones reconocibles

- **Solo inglés**: experimentos sobre Wikipedia EN únicamente.
- **Embedding único**: solo Skip-gram NEG-15 600d. No exploran CBOW ni GloVe.
- **Hiperparámetro $\varepsilon = 0.001$** sin justificación formal.
- **Transformación $(x+1)/2$ ad-hoc** para manejar cosenos negativos.

---

## Conexión con el laboratorio

Esta fórmula es **el motor del bloque de analogías del Práctico 18**:

- El notebook cita explícitamente este paper: *"para una formalización del procedimiento, ver la fórmula (4) en la Sección 6 de este artículo"*.
- Toda llamada a `google_wordvecs.most_similar_cosmul(positive=[...], negative=[...])` ejecuta exactamente fórmula (4).
- Cuando una analogía falla con resultados ruidosos (e.g. `Santiago + Venezuela − Chile` devuelve apellidos hispanos en vez de `Caracas`), la causa raíz suele ser **polisemia de una palabra** que crea el escenario "soft-or" descrito por Levy & Goldberg.

---

## Cross-links

{{< cards >}}
  {{< card link="/laboratorios/lab-18" title="Lab 18 - Word Embeddings" subtitle="3CosMul en acción sobre Google News" icon="academic-cap" >}}
  {{< card link="/clases/clase-18" title="Clase 18 - Word2Vec, GloVe, Skip-Thought" subtitle="Teoría de los embeddings densos" icon="academic-cap" >}}
  {{< card link="/papers/sgns-implicit-mf-levy-goldberg-2014" title="Levy-Goldberg NeurIPS 2014" subtitle="SGNS = factorización implícita de PMI" icon="document-text" >}}
  {{< card link="/papers/contrastive-analogies-ri-lee-verma-2023" title="Ri-Lee-Verma 2023" subtitle="Líneas paralelas con factor ζ" icon="document-text" >}}
{{< /cards >}}
