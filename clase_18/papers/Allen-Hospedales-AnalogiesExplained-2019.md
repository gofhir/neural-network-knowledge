# Allen & Hospedales 2019 — Analogies Explained: Towards Understanding Word Embeddings

| Campo | Valor |
|---|---|
| **Autores** | Carl Allen, Timothy Hospedales |
| **Afiliación** | School of Informatics, University of Edinburgh |
| **Venue** | ICML 2019 (PMLR 97) |
| **Fecha** | 11 mayo 2019 (v2 arXiv) |
| **Pdf** | `Allen-Hospedales-AnalogiesExplained-2019.pdf` (11 páginas + apéndice) |
| **Citaciones** | >150 |
| **URL** | https://arxiv.org/abs/1901.09813 |

> *"We provide the first rigorous proof of the linear relationship between word embeddings of analogies, including explicit, interpretable error terms."*

Este paper responde finalmente a una pregunta abierta de 6 años: **¿por qué `king - man + woman ≈ queen` funciona** en embeddings de Word2Vec y GloVe? Es uno de los rated *recommended readings* en la slide 26 de la clase 18.

---

## 1. El misterio que persistía

Desde Mikolov 2013, la propiedad de **composicionalidad aditiva** de los word embeddings había sido empíricamente observada y celebrada:

$$
\mathbf{w}_{\text{king}} - \mathbf{w}_{\text{man}} + \mathbf{w}_{\text{woman}} \approx \mathbf{w}_{\text{queen}}.
$$

Pero **nadie había probado rigurosamente por qué**. Existían varias propuestas:

### 1.1 Explicaciones previas y sus fallas

| Autores | Año | Propuesta | Limitación |
|---|---|---|---|
| Pennington 2014 (GloVe) | 2014 | Argumento intuitivo basado en ratios $P_{ik}/P_{jk}$ | Solo motivacional |
| Arora et al. | 2016 | Modelo latente RAND-WALK | Asunciones fuertes (Gaussianas uniformes) **no cumplidas en práctica** |
| Gittens et al. | 2017 | Análisis vía PMI matrix | Asumen palabras siguen distribución **uniforme**, no Zipf |
| Ethayarajh et al. | 2018 | Co-occurrence shifted PMI | Requieren paralelogramas perfectos, no se sostiene en $\mathbb{R}^d$ |
| Mikolov 2013 | 2013 | Heurística "linear regularities" | Sin demostración |

**Problema común**: cada explicación previa asume condiciones que **fallan en la práctica** — distribución uniforme, embeddings idénticos input/output, etc.

### 1.2 Aporte de Allen & Hospedales

Proveer una explicación matemáticamente rigurosa que:

1. **No asume condiciones que no se cumplen** (asume Zipf no uniforme, distingue embeddings input/output, etc).
2. **Identifica explícitamente los términos de error**: cuando la analogía no funciona perfectamente, sabemos exactamente qué la rompe.
3. **Generaliza a analogías sobre conjuntos de palabras** (no solo single-word analogies).

---

## 2. Marco teórico

### 2.1 Hipótesis distribucional como punto de partida

Su intuición central retoma a Firth 1957:

> *"You shall know a word by the company it keeps."*

Pero la **formaliza** vía PMI: dos palabras son semánticamente equivalentes si **inducen la misma distribución sobre palabras de contexto cercano**.

### 2.2 Recuerdo: Word2Vec factoriza PMI shifted (Levy & Goldberg 2014)

Bajo asunciones adecuadas:
$$
\mathbf{w}_i^\top \mathbf{c}_j = \text{PMI}(w_i, c_j) - \log k. \quad (1)
$$

En forma matricial: $\mathbf{W}^\top \mathbf{C} = \mathbf{SPMI} \in \mathbb{R}^{n \times n}$ donde $\mathbf{SPMI}_{i,j} = \text{PMI}(w_i, c_j) - \log k$.

GloVe similarmente factoriza:
$$
\mathbf{w}_i^\top \mathbf{c}_j = \log p(w_i, c_j) - b_i - b_j + \log Z. \quad (3)
$$

### 2.3 El shift como artefacto

Allen & Hospedales argumentan que el **shift $-\log k$ es un artefacto del algoritmo W2V**, no una propiedad inherente. Si se ajusta W2V para evitar el shift directo (Le 2017), la calidad mejora. Por lo tanto, asumen embeddings que **factorizan PMI sin shift**:

$$
\mathbf{w}_i^\top \mathbf{c}_j = \text{PMI}(w_i, c_j), \quad \text{o equivalentemente} \quad \mathbf{W}^\top \mathbf{C} = \mathbf{PMI}. \quad (4)
$$

### 2.4 Asunciones del paper

- **A1**: $\mathbf{C}$ tiene rango pleno por filas.
- **A2**: $f: \mathbb{R}^n \to \mathbb{R}^d$, $f(\mathbf{M}_i) = \mathbf{w}_i$ es aproximadamente homomórfico respecto a la suma: $f(\mathbf{M}_i + \mathbf{M}_j) \approx f(\mathbf{M}_i) + f(\mathbf{M}_j)$.
- **A3**: $p(\mathcal{W}) > 0$ para todo conjunto $\mathcal{W} \subseteq \mathcal{E}$ con $|\mathcal{W}| < l$, donde $l$ es el tamaño de ventana. (Asegura que los PMI están bien definidos para todas las combinaciones consideradas.)

La asunción A2 es la **lupa nueva** del paper: si la proyección de baja-rango preserva linealidad **a la primera aproximación**, entonces las relaciones de los PMI columnas se trasladan a las relaciones de los embeddings.

---

## 3. La idea central — Paráfrasis

### 3.1 Definición probabilística de paráfrasis

> *"Decimos que palabra $w_*$ parafrasea al conjunto $\mathcal{W}$ si $w_*$ y $\mathcal{W}$ son semánticamente intercambiables en el texto, i.e., en circunstancias donde **todo** $w_i \in \mathcal{W}$ aparecería, $w_*$ podría aparecer en su lugar."*

Formalmente: el conjunto de palabras de contexto $\mathcal{C}_{\mathcal{W}}$ observado cuando $\mathcal{W}$ aparece junto, induce una distribución $p(c_j \mid \mathcal{W})$ sobre contextos. Y para $w_*$ también: $p(c_j \mid w_*)$.

$w_*$ **parafrasea** $\mathcal{W}$ ssi las dos distribuciones inducidas son cercanas:
$$
p(c_j \mid w_*) \approx p(c_j \mid \mathcal{W}).
$$

### 3.2 Paraphrase error

**Definición D1**: $w_* \in \mathcal{E}$ parafrasea $\mathcal{W} \subseteq \mathcal{E}$ si el **paraphrase error** $\boldsymbol{\rho}^{\mathcal{W}, w_*} \in \mathbb{R}^n$ es pequeño (element-wise):
$$
\rho_j^{\mathcal{W}, w_*} = \log \frac{p(c_j \mid w_*)}{p(c_j \mid \mathcal{W})}, \quad c_j \in \mathcal{E}. \quad (5)
$$

Es la **log-ratio** entre las distribuciones inducidas. Si ambas distribuciones son iguales, $\rho_j^{\mathcal{W}, w_*} = 0$ para todo $j$.

### 3.3 Lemma 1 — La descomposición clave

**Lemma 1**: Para cualquier $w_* \in \mathcal{E}$ y $\mathcal{W} \subseteq \mathcal{E}$ con $|\mathcal{W}| < l$:

$$
\mathbf{PMI}_* = \sum_{w_i \in \mathcal{W}} \mathbf{PMI}_i + \boldsymbol{\rho}^{\mathcal{W}, w_*} + \boldsymbol{\sigma}^{\mathcal{W}} - \tau^{\mathcal{W}} \mathbf{1}. \quad (5)
$$

Donde:
- $\mathbf{PMI}_*$, $\mathbf{PMI}_i$ son las **columnas** de la matriz PMI correspondientes a $w_*$, $w_i$.
- $\boldsymbol{\rho}^{\mathcal{W}, w_*}$ = paraphrase error (entre $w_*$ y $\mathcal{W}$).
- $\boldsymbol{\sigma}^{\mathcal{W}}_j = \log \frac{p(c_j \mid \mathcal{W})}{\prod_i p(c_j \mid w_i)}$ = **dependence error** (refleja dependencias condicionales **dentro** de $\mathcal{W}$).
- $\tau^{\mathcal{W}} = \log \frac{p(\mathcal{W})}{\prod_i p(w_i)}$ = medida escalar de **independencia mutua** entre las palabras de $\mathcal{W}$.

**Interpretación**: la suma de PMI vectores **no es** exactamente la PMI del parafraseador, pero está cerca **excepto por dos términos de error**:
- $\boldsymbol{\rho}^{\mathcal{W}, w_*}$: error de **paráfrasis** (entre $w_*$ y $\mathcal{W}$).
- $\boldsymbol{\sigma}^{\mathcal{W}} - \tau^{\mathcal{W}}\mathbf{1}$: error de **dependencia** (dentro de $\mathcal{W}$).

### 3.4 Theorem 1 — Traducción a embeddings

**Theorem 1 (Paráfrasis)**: Bajo A1, A2, A3, para cualquier $w_*, \mathcal{W}$ con $|\mathcal{W}| < l$:

$$
\mathbf{w}_* = \mathbf{w}_{\mathcal{W}} + \mathbf{C}^\dagger (\boldsymbol{\rho}^{\mathcal{W}, w_*} + \boldsymbol{\sigma}^{\mathcal{W}} - \tau^{\mathcal{W}} \mathbf{1}). \quad (6)
$$

Donde $\mathbf{w}_{\mathcal{W}} = \sum_{w_i \in \mathcal{W}} \mathbf{w}_i$ y $\mathbf{C}^\dagger = (\mathbf{C}\mathbf{C}^\top)^{-1} \mathbf{C}$ es la **pseudo-inversa de Moore-Penrose** de $\mathbf{C}^\top$.

**Lectura**: el embedding del parafraseador es la **suma de embeddings de las palabras parafraseadas más un término de error proyectado** que depende de los paraphrase y dependence errors.

**Corolario 1.2**: $\mathbf{w}_* \approx \mathbf{w}_{\mathcal{W}}$ si:
1. $w_*$ parafrasea $\mathcal{W}$ ($\boldsymbol{\rho}$ pequeño).
2. Las palabras de $\mathcal{W}$ son materialmente independientes ($\boldsymbol{\sigma}$, $\tau$ pequeños).

---

## 4. De paráfrasis a analogías

### 4.1 Word transformation

Hasta aquí se relaciona un word con un **conjunto** ($\mathcal{W}$ parafrasea $w_*$). Para llegar a analogías necesitan introducir **word transformations**.

**Idea**: para transformar $w_x$ en $w_{x^*}$, hay que **agregar y quitar palabras de contexto**:
- $\mathcal{W}^+$: palabras a "agregar" (e.g., para `man → king`, agregar `royal`).
- $\mathcal{W}^-$: palabras a "quitar" (e.g., para `queen → woman`, quitar `royal`).

Formalmente (Definición D3):
> *"Existe una word transformation desde $w_x$ a $w_{x^*}$ con parámetros $\mathcal{W}^+, \mathcal{W}^-$ si $\{w_x\} \cup \mathcal{W}^+$ parafrasea $\{w_{x^*}\} \cup \mathcal{W}^-$."*

### 4.2 Interpretación de "a is to a* as b is to b*"

**Definición D4**: la analogía $a:a^* :: b:b^*$ se cumple si **existen los mismos parámetros de transformación** $\mathcal{W}^+, \mathcal{W}^-$ que simultáneamente transforman $w_a \to w_{a^*}$ y $w_b \to w_{b^*}$.

**Ejemplo**: `man:king :: woman:queen`.
- $\mathcal{W}^+ = \{\text{royal}\}$ — agregar.
- $\mathcal{W}^- = \emptyset$ — nada que quitar.
- $w_a = \text{man}$, $w_{a^*} = \text{king}$. Agregar `royal` a `man` → `king`. ✓
- $w_b = \text{woman}$, $w_{b^*} = \text{queen}$. Agregar `royal` a `woman` → `queen`. ✓

### 4.3 Theorem 2 (Generalized Paraphrase) — Resultado central

**Lemma 3**: si la analogía $a:a^* :: b:b^*$ se cumple por D4 con parámetros $\mathcal{W}^+, \mathcal{W}^-$:

$$
\mathbf{PMI}_{b^*} = \mathbf{PMI}_{a^*} - \mathbf{PMI}_a + \mathbf{PMI}_b + (\text{términos de error pequeños}). \quad (13)
$$

Trasladando a embeddings (multiplicando por $\mathbf{C}^\dagger$):

$$
\boxed{\mathbf{w}_{b^*} \approx \mathbf{w}_{a^*} - \mathbf{w}_a + \mathbf{w}_b.}
$$

**¡Es exactamente la relación de analogía empírica!** Y ahora sabemos:
- **Cuándo se cumple**: cuando hay parámetros $\mathcal{W}^+, \mathcal{W}^-$ que transforman simultáneamente ambos pares.
- **Cuándo falla**: cuando los términos de error son grandes — i.e., $w_*$ no es un buen parafraseador, o las palabras de $\mathcal{W}$ no son independientes.

---

## 5. Términos de error explícitos

El paper identifica **3 fuentes de error** que rompen la analogía:

### 5.1 Paraphrase error $\boldsymbol{\rho}$

Cuando la transformación no es perfecta. Ejemplo: agregar `royal` a `man` no produce exactamente `king` — produce algo cercano. La discrepancia es $\boldsymbol{\rho}$.

### 5.2 Dependence error $\boldsymbol{\sigma}$

Cuando las palabras agregadas tienen dependencias condicionales fuertes. Ejemplo: si $\mathcal{W}^+ = \{\text{royal}, \text{monarch}\}$ son altamente redundantes, su contribución conjunta no es suma de contribuciones individuales — hay solapamiento.

### 5.3 Reconstruction error

Cuando $d \ll n$ (la dimensión del embedding es mucho menor que el vocabulario), la factorización $\mathbf{W}^\top \mathbf{C} = \mathbf{PMI}$ es solo aproximada — pierde información. Esto se controla con A2.

---

## 6. Verificación empírica (Figure 1 del paper)

El paper muestra una visualización 3D donde:
- $\mathbf{w}_K - \mathbf{w}_M + \mathbf{w}_W$ (resultado de la analogía) aterriza **muy cerca de $\mathbf{w}_{\text{queen}}$**.
- Pero **no exactamente** sobre `queen` — y la diferencia es exactamente el término de error predicho.

Esto valida cuantitativamente la teoría: la analogía es aproximada con error explicable.

---

## 7. Implicaciones prácticas

### 7.1 Cuándo confiar en analogías

Si la diferencia $\mathbf{w}_{b^*} - (\mathbf{w}_{a^*} - \mathbf{w}_a + \mathbf{w}_b)$ es:

- **Pequeña**: la analogía es válida, se puede confiar en `arg max`.
- **Grande**: el término de error domina, la analogía no es interpretable. Posibles causas: paráfrasis imperfecta, dependencias entre $\mathcal{W}$, embedding de baja rank.

### 7.2 Falsos positivos

Sección 5.3 del paper: si $\mathbf{w}_* \approx \mathbf{w}_{\mathcal{W}}$, NO necesariamente $w_*$ parafrasea $\mathcal{W}$ — puede haber **cancelación de errores**. Esto explica por qué algunos resultados de analogías parecen correctos pero son falsamente positivos.

### 7.3 Recomendaciones para uso de analogías

1. **Verificar el error**: si la analogía top-1 no es claramente mejor que top-5, sospechar.
2. **Evitar excluir las palabras de entrada**: la práctica estándar (Mikolov 2013) de excluir $a, b, c$ del `arg max` infla el accuracy artificialmente.
3. **Métricas alternativas**: 3CosMul (Levy & Goldberg 2014b) tiene mejor estabilidad.

---

## 8. Conexión con otros papers

### 8.1 Con Levy & Goldberg 2014

Allen & Hospedales **construyen sobre** Levy & Goldberg: aceptan que SGNS factoriza PMI shifted, y razonan desde ahí. El paper es **el siguiente paso teórico**: una vez sabemos qué factoriza SGNS, **¿por qué las analogías funcionan**?

### 8.2 Con GloVe

El paper conjetura (sección 3) que el mismo razonamiento aplica a GloVe, aunque con más flexibilidad debido a los biases adicionales. Por eso GloVe a veces se comporta diferente que SGNS en analogías.

### 8.3 Con la práctica moderna

Aunque BERT y GPT no usan word embeddings estáticos, **embeddings contextuales** se calculan internamente en la primera capa. Allen & Hospedales sugieren que sus resultados se extienden a embeddings estáticos derivados de modelos contextuales (e.g., averaging BERT embeddings) — pero no lo prueban formalmente. Es una dirección de trabajo futuro.

---

## 9. Limitaciones del paper

1. **Análisis perturbativo**: solo válido cuando los errores son pequeños. Para analogías muy "creativas" (e.g., `Trump:USA :: Putin:Russia`), la teoría no aplica directamente.
2. **Solo SGNS-like**: aunque conjeturan extensión a GloVe, no prueban casos más generales (FastText, ELMo).
3. **No experimentos extensivos**: el paper es teórico. Los gráficos son ilustrativos, no benchmarks sistemáticos.
4. **Asunciones técnicas A1-A3**: rango pleno, homomorfismo aproximado, probabilidades positivas. Todas razonables pero no triviales.
5. **Sin guía para mejorar embeddings**: identifica problemas pero no resuelve cómo entrenar embeddings con menos error de analogía.

---

## 10. Impacto

Allen & Hospedales es **el cierre teórico** del capítulo Word2Vec/GloVe iniciado por Mikolov en 2013. Aunque su impacto en producción es limitado (no propone un algoritmo nuevo), su impacto teórico es importante:

- Cita obligatoria en cualquier paper sobre interpretabilidad de embeddings.
- Inspiración para análisis similares en embeddings contextuales (BERTology).
- Validación de la hipótesis distribucional de Firth desde primeros principios matemáticos.

### 10.1 Sucesores

- **Ethayarajh 2019** (*"How contextual are contextualized word representations?"*) — extiende ideas a BERT/ELMo/GPT.
- **Mu et al. 2018** y **Gao et al. 2021** (*Representation Degeneration*) — geometría de embeddings, anisotropía.
- **Carlsson et al. 2021** (*"Semantic Re-tuning with Contrastive Tension"*) — corregir isotropía vía contrastive.

---

## 11. Conexión con la clase 18

La clase 18 lista este paper como **lectura recomendada** en la slide 26, junto con:
- "Analogies Explained" blog (Carl Allen): https://carl-allen.github.io/nlp/2019/07/01/explaining-analogies-explained.html
- "Contrastive Loss is All You Need to Recover Analogies as Parallel Lines": https://arxiv.org/abs/2306.08221 — un paper de 2023 que extiende a aprendizaje contrastivo moderno.

Estos tres recursos forman el **bloque de comprensión profunda de analogías**. Para un estudiante serio del curso IA UC, este paper responde la pregunta: *"y al final, ¿por qué `king - man + woman = queen` funciona?"*

---

## 12. Cita BibTeX

```bibtex
@inproceedings{allen2019analogies,
  title={Analogies Explained: Towards Understanding Word Embeddings},
  author={Allen, Carl and Hospedales, Timothy},
  booktitle={Proceedings of the 36th International Conference on Machine Learning},
  series={Proceedings of Machine Learning Research},
  volume={97},
  pages={223--231},
  year={2019},
  publisher={PMLR},
  url={https://arxiv.org/abs/1901.09813}
}
```

---

## 13. Frase para recordar

> *"Analogies are paraphrases in disguise — when the parameters of word transformation are shared."* — la frase compacta del paper. La idea es que la analogía `a:a* :: b:b*` es esencialmente decir "la misma transformación lingüística produce $a → a^*$ y $b → b^*$", y los embeddings preservan esto vía aritmética lineal porque factorizan PMI bajo la hipótesis distribucional.

---

## 14. Mini-glossary

| Término | Definición |
|---|---|
| **Paráfrasis (D1)** | $w_*$ parafrasea $\mathcal{W}$ si inducen distribuciones similares sobre contextos |
| **Paraphrase error $\boldsymbol{\rho}$** | $\log p(c_j \| w_*) - \log p(c_j \| \mathcal{W})$, mide qué tan buena es la paráfrasis |
| **Dependence error $\boldsymbol{\sigma}$** | Refleja dependencias condicionales dentro de $\mathcal{W}$ |
| **Word transformation** | Agregar $\mathcal{W}^+$ y quitar $\mathcal{W}^-$ para ir de $w_x$ a $w_{x^*}$ |
| **Analogía (D4)** | Misma transformación funciona para $a → a^*$ y $b → b^*$ |
| **Linear regularity** | El hecho empírico $\mathbf{w}_{a^*} - \mathbf{w}_a \approx \mathbf{w}_{b^*} - \mathbf{w}_b$ |

---

## 15. Aplicación práctica al lab de la clase

Si el lab usa Word2Vec / GloVe preentrenado para tareas clínicas (e.g., embeddings de términos médicos), Allen & Hospedales sugiere:

1. **Las analogías médicas se mantendrán** si los términos siguen la hipótesis distribucional: `aspirin:anti-inflammatory :: paracetamol:?` debe dar `analgesic`.
2. **Fallarán si los términos médicos son polisémicos**: e.g., "depression" (medical condition vs económica). El paraphrase error será grande.
3. **Modelo recomendado**: usar embeddings entrenados en **corpus médico específico** (BioWordVec, ClinicalBERT) para que la distribución de contextos esté alineada con el dominio.

Esta conexión con dominio clínico es central si el `Practico18.ipynb` aborda NLP médico.
