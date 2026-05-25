---
title: "Analogies Explained - Understanding Word Embeddings"
weight: 298
math: true
---

{{< paper-card
    title="Analogies Explained: Towards Understanding Word Embeddings"
    authors="Allen, Hospedales"
    year="2019"
    venue="ICML 2019"
    pdf="/papers/analogies-explained-allen-hospedales-2019.pdf"
    arxiv="1901.09813" >}}
La **primera prueba matematica rigurosa** de por que `vec(king) - vec(man) + vec(woman) ~ vec(queen)` funciona en embeddings Word2Vec/GloVe. Introduce una definicion probabilistica de **parafrasis** ("dos palabras son intercambiables si inducen distribuciones similares sobre contextos cercanos") y muestra que las analogias son **word transformations con parametros compartidos**. Identifica los terminos de error explicitos que rompen la analogia.
{{< /paper-card >}}

---

## Contexto

Desde Mikolov 2013, la propiedad de **composicionalidad aditiva** de los embeddings habia sido empiricamente celebrada pero **nadie habia probado rigurosamente por que funciona**. Explicaciones previas tenian todas problemas:

| Autores | Propuesta | Limitacion |
|---|---|---|
| Pennington 2014 (GloVe) | Argumento intuitivo sobre ratios | Solo motivacional |
| Arora 2016 (RAND-WALK) | Modelo latente Gaussiano uniforme | Asunciones no cumplidas |
| Gittens 2017 | Analisis via PMI matrix | Asume distribucion uniforme |
| Ethayarajh 2018 | Co-occurrence shifted PMI | No se sostiene en $\mathbb{R}^d$ |

Allen & Hospedales **construyen sobre** [Levy & Goldberg 2014](/papers/sgns-implicit-mf-levy-goldberg-2014): aceptando que SGNS factoriza PMI shifted, razonan desde ahi sin asunciones falaces.

---

## Ideas principales

### 1. Definicion probabilistica de parafrasis

> *Decimos que $w_*$ parafrasea $\mathcal{W} \subseteq \mathcal{E}$ si $w_*$ y $\mathcal{W}$ son semanticamente intercambiables en el texto -- en circunstancias donde **todo** $w_i \in \mathcal{W}$ apareceria, $w_*$ podria aparecer en su lugar.*

Formalmente, las distribuciones sobre palabras de contexto inducidas son similares: $p(c_j \mid w_*) \approx p(c_j \mid \mathcal{W})$.

### 2. Paraphrase error

$$\rho_j^{\mathcal{W}, w_*} = \log \frac{p(c_j \mid w_*)}{p(c_j \mid \mathcal{W})}, \quad c_j \in \mathcal{E}$$

Es el **log-ratio** entre las distribuciones inducidas. $\rho_j = 0$ si las distribuciones son iguales.

### 3. Lemma 1 -- Descomposicion clave de PMI

$$\mathbf{PMI}_* = \sum_{w_i \in \mathcal{W}} \mathbf{PMI}_i + \boldsymbol{\rho}^{\mathcal{W}, w_*} + \boldsymbol{\sigma}^{\mathcal{W}} - \tau^{\mathcal{W}} \mathbf{1}$$

La PMI de $w_*$ **no es** exactamente la suma de PMI de las palabras en $\mathcal{W}$, pero esta cerca **excepto por dos terminos de error**:

- $\boldsymbol{\rho}^{\mathcal{W}, w_*}$: error de **parafrasis** (entre $w_*$ y $\mathcal{W}$).
- $\boldsymbol{\sigma}^{\mathcal{W}} - \tau^{\mathcal{W}}\mathbf{1}$: error de **dependencia** (dentro de $\mathcal{W}$).

### 4. Theorem 1 -- Traduccion a embeddings

$$\mathbf{w}_* = \mathbf{w}_{\mathcal{W}} + \mathbf{C}^\dagger (\boldsymbol{\rho}^{\mathcal{W}, w_*} + \boldsymbol{\sigma}^{\mathcal{W}} - \tau^{\mathcal{W}} \mathbf{1})$$

Donde $\mathbf{w}_{\mathcal{W}} = \sum_{w_i \in \mathcal{W}} \mathbf{w}_i$ y $\mathbf{C}^\dagger = (\mathbf{C}\mathbf{C}^\top)^{-1} \mathbf{C}$ es la **pseudo-inversa de Moore-Penrose**.

**Corolario**: $\mathbf{w}_* \approx \mathbf{w}_{\mathcal{W}}$ si $w_*$ parafrasea $\mathcal{W}$ y las palabras de $\mathcal{W}$ son materialmente independientes.

### 5. Word transformations y analogias

Para transformar $w_x \to w_{x^*}$ hay que **agregar palabras $\mathcal{W}^+$ y quitar palabras $\mathcal{W}^-$**:

- $\{w_x\} \cup \mathcal{W}^+$ parafrasea $\{w_{x^*}\} \cup \mathcal{W}^-$.

Una **analogia $a:a^* :: b:b^*$ se cumple** si los **mismos** parametros $\mathcal{W}^+, \mathcal{W}^-$ transforman simultaneamente $w_a \to w_{a^*}$ y $w_b \to w_{b^*}$.

**Ejemplo**: `man:king :: woman:queen`:
- $\mathcal{W}^+ = \{\text{royal}\}$, $\mathcal{W}^- = \emptyset$.
- Agregar `royal` a `man` produce `king`. Agregar `royal` a `woman` produce `queen`. Los mismos parametros funcionan para ambos pares.

### 6. Resultado central

$$\boxed{\mathbf{w}_{b^*} \approx \mathbf{w}_{a^*} - \mathbf{w}_a + \mathbf{w}_b}$$

con terminos de error explicitos que dependen de paraphrase + dependence errors de cada par.

---

## Cuando falla la analogia

El paper identifica **3 fuentes de error**:

1. **Paraphrase error $\boldsymbol{\rho}$**: cuando la transformacion no es perfecta.
2. **Dependence error $\boldsymbol{\sigma}$**: cuando $\mathcal{W}$ tiene dependencias condicionales fuertes (ej. $\{\text{royal}, \text{monarch}\}$ son redundantes).
3. **Reconstruction error**: cuando $d \ll n$, la factorizacion es solo aproximada.

**Falsos positivos**: $\mathbf{w}_* \approx \mathbf{w}_{\mathcal{W}}$ **no implica** que $w_*$ parafrasee $\mathcal{W}$ -- puede haber **cancelacion de errores**. Esto explica resultados "correctos por casualidad".

---

## Implicaciones practicas

### 1. Cuando confiar en analogias

Si la diferencia $\mathbf{w}_{b^*} - (\mathbf{w}_{a^*} - \mathbf{w}_a + \mathbf{w}_b)$ es:
- **Pequena**: la analogia es valida.
- **Grande**: el error domina, no confiable.

### 2. Criticas a la metodologia estandar

La practica de Mikolov 2013 de excluir $a, b, c$ del `arg max` **infla artificialmente** el accuracy. Linzen 2016 ya habia mostrado esto empiricamente; Allen & Hospedales lo formalizan.

### 3. Aplicacion a dominios

Para embeddings entrenados en corpus medicos (BioWordVec, ClinicalBERT):

- **Las analogias se mantienen** si los terminos siguen la hipotesis distribucional.
- **Fallan en polisemia**: "depression" (medica vs economica) tendra grandes errores.

---

## Limitaciones del paper

1. **Analisis perturbativo**: solo valido cuando los errores son pequenos.
2. **Solo SGNS-like**: conjetura extension a GloVe pero no la prueba.
3. **No experimentos extensivos**: el paper es fundamentalmente teorico.
4. **Asunciones tecnicas**: rango pleno de $\mathbf{C}$, homomorfismo aproximado, probabilidades positivas. Todas razonables pero no triviales.

---

## Por que importa hoy

Es **el cierre teorico** del capitulo Word2Vec/GloVe. Su impacto:

- Cita obligatoria en cualquier paper sobre interpretabilidad de embeddings.
- Inspiracion para analisis similares en embeddings contextuales (BERTology).
- Validacion de la **hipotesis distribucional de Firth** desde primeros principios matematicos.

Sucesores que extienden estas ideas:
- Ethayarajh 2019: "How contextual are contextualized word representations?"
- Mu et al. 2018, Gao et al. 2021: Representation Degeneration, geometria y anisotropia.

---

## Notas y enlaces

- **Blog del autor**: ["Analogies Explained" Explained](https://carl-allen.github.io/nlp/2019/07/01/explaining-analogies-explained.html) -- explicacion informal del paper.
- **Extension 2023**: ["Contrastive Loss is All You Need to Recover Analogies as Parallel Lines"](https://arxiv.org/abs/2306.08221).
- **Predecesores**: [Levy & Goldberg - SGNS as MF](/papers/sgns-implicit-mf-levy-goldberg-2014), [GloVe](/papers/glove-pennington-2014).
- **Sucesor**: [Ri-Lee-Verma - Contrastive Analogies (2023)](/papers/contrastive-analogies-ri-lee-verma-2023) extiende a lineas paralelas con factor ζ.
- **Clase asociada**: [Clase 18 - Modelos de lenguaje, Word2Vec, GloVe y SkipThought](/clases/clase-18).
- **Laboratorio asociado**: [Lab 18 - Word Embeddings en accion](/laboratorios/lab-18) (verifica empiricamente ζ=1.16 en plot king/queen/man/woman).
- **Fundamentos relacionados**: [Word2Vec](/fundamentos/word2vec), [Embeddings distribuidos](/fundamentos/embeddings-distribuidos).
- **Cita BibTeX**:

```bibtex
@inproceedings{allen2019analogies,
  title={Analogies Explained: Towards Understanding Word Embeddings},
  author={Allen, Carl and Hospedales, Timothy},
  booktitle={Proceedings of the 36th International Conference on Machine Learning},
  series={PMLR},
  volume={97},
  pages={223--231},
  year={2019}
}
```
