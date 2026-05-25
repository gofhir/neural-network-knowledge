# Levy & Goldberg 2014 — Linguistic Regularities in Sparse and Explicit Word Representations

| Campo | Valor |
|---|---|
| **Autores** | Omer Levy, Yoav Goldberg |
| **Afiliación** | Computer Science Department, Bar-Ilan University, Ramat-Gan, Israel |
| **Venue** | CoNLL 2014 (Eighteenth Conference on Computational Natural Language Learning), Baltimore, MD, USA |
| **Fecha** | 26-27 junio 2014 |
| **Pdf** | `Levy-Goldberg-LinguisticRegularities-2014.pdf` (10 páginas) |
| **ACL Anthology** | W14-1618 |
| **Citaciones** | >2.500 |
| **URL** | https://aclanthology.org/W14-1618/ |
| **Cita** | Levy, O. & Goldberg, Y. (2014). Linguistic Regularities in Sparse and Explicit Word Representations. *Proceedings of the Eighteenth Conference on Computational Natural Language Learning*, pp. 171–180. |

> *"We show that Mikolov et al.'s method of first adding and subtracting word vectors, and then searching for a word similar to the result, is equivalent to searching for a word that maximizes a linear combination of three pairwise word similarities. Based on this observation, we suggest an improved method of recovering relational similarities, improving the state-of-the-art results on two recent word-analogy datasets."*

Este es **el paper que da origen a `most_similar_cosmul`**, la función exacta que usas en las Actividades 1-2 del Práctico 18. El notebook lo cita explícitamente: *"para una formalización del procedimiento, ver la fórmula (4) en la Sección 6 de este artículo"*. Las tres contribuciones centrales son: (1) reinterpretar la aritmética vectorial `king − man + woman` como una combinación lineal de tres similitudes coseno (**3CosAdd**); (2) proponer una alternativa multiplicativa robusta a desbalances de escala (**3CosMul**); y (3) demostrar que representaciones **explícitas** dispersas (PPMI sobre co-ocurrencias) capturan tanta regularidad lingüística como los embeddings neuronales densos — un resultado que desmitifica el aura "mágica" de Word2Vec.

---

## 1. Contexto histórico

El paper sale a mediados de 2014, **un año y medio después** del primer Word2Vec (Mikolov et al. 2013, ICLR). Para entender su importancia hay que ubicarse en el clima de la época:

| Año | Hito | Estado del campo |
|---|---|---|
| 2003 | Bengio NPLM | Primer modelo neuronal con embeddings densos. Lento. |
| 2010 | Mikolov RNN-LM | Embeddings densos competitivos en perplejidad. |
| 2013 ene | Word2Vec — Efficient (Mikolov, ICLR) | CBOW y Skip-gram entrenan en miles de millones de palabras en horas. Aparecen los **vectores que cumplen analogías**. |
| 2013 mayo | Mikolov, Yih, Zweig (NAACL) | Primer paper que documenta y mide la propiedad de regularidad lingüística (`king − man + woman ≈ queen`). |
| 2013 oct | Word2Vec — Distributed Repr. (Mikolov, NeurIPS) | Negative sampling, subsampling, phrase embeddings. Word2Vec se convierte en estándar. |
| 2014 jun | **Levy & Goldberg (CoNLL) ← ESTE PAPER** | Demuestra que la aritmética vectorial = 3 similitudes pairwise. Propone 3CosMul. Compara contra PPMI sparse. |
| 2014 oct | Pennington et al., GloVe (EMNLP) | Embeddings basados en factorización de matriz log de co-ocurrencia. Compite con Word2Vec. |
| 2014 dec | Levy & Goldberg (NeurIPS) | Demuestra que SGNS es factorización implícita de PMI shifted. |

**Por qué este paper era necesario**: en 2013-2014 se había instalado la idea de que los embeddings neuronales **"capturaban algo mágico"** que la estadística clásica no podía. Levy & Goldberg vienen del mundo de la **distributional semantics tradicional** (PMI, LSA, distributional similarity) y se preguntaron: *¿es realmente el embedding neuronal el que produce esta magia, o las analogías están latentes en las propias estadísticas de co-ocurrencia?*

La respuesta tendrá consecuencias profundas:
- Si los embeddings descubren patrones nuevos → la receta es la red neuronal.
- Si los embeddings solo preservan patrones que ya estaban en los conteos → la receta es la estadística de co-ocurrencia, y la red es opcional.

Su conclusión empírica fuerte: **opción 2**. Esto motiva su paper de NeurIPS 6 meses después (SGNS = factorización implícita de PMI) y el paper de GloVe (que es explícitamente basado en conteos globales). Es **el preámbulo conceptual** de toda la oleada de "embeddings interpretables" 2014-2017.

---

## 2. La pregunta central — ¿por qué `king − man + woman ≈ queen`?

Mikolov et al. (2013) habían mostrado empíricamente que en sus embeddings:

```
queen ≈ king − man + woman
```

y propusieron resolver analogías "a:a* :: b:?" buscando:

$$
b^* = \arg\max_{x \in V} \, \text{sim}(x, \; b - a + a^*)
$$

con `sim` = similitud coseno. Llamemos a esto **3CosAdd**.

Levy & Goldberg observan que bajo la asunción habitual de **vectores normalizados a norma unitaria** (todos los `||v|| = 1`), la expresión se puede reescribir mediante álgebra básica como:

$$
b^* = \arg\max_{x \in V} \big[\cos(x, b) - \cos(x, a) + \cos(x, a^*)\big]
$$

Esta es la **ecuación (3) del paper**, llamada **3CosAdd**. La derivación es directa:

```
cos(x, b − a + a*) = x · (b − a + a*) / (||x|| · ||b − a + a*||)
                   = (x·b − x·a + x·a*) / (1 · ||b − a + a*||)
```

El denominador `||b − a + a*||` no depende de `x` (es constante para una pregunta dada), por lo que en el `argmax` se cancela. Como `||x|| = 1`, `x·b = cos(x,b)`, etc. Queda:

$$
\arg\max_x (\cos(x, b) - \cos(x, a) + \cos(x, a^*))
$$

### El insight central — "soft-or"

Levy & Goldberg explican que esta forma reescribe el problema:

> *"Encontrar b\* es buscar una palabra similar a b (atractor) **y** similar a a* (atractor) pero **distinta** de a (repulsor)."*

Por ejemplo, para "man:woman :: king:?":
- Queremos `x` similar a `woman` (atractor 1 — el sexo femenino).
- Queremos `x` similar a `king` (atractor 2 — la realeza).
- Queremos `x` distinto a `man` (repulsor — el sexo masculino).

`queen` es justamente quien cumple esas tres condiciones. Esta intuición es **mucho más clara** que pensar en sumas y restas de vectores en ℝ³⁰⁰, y es la que internalizan los textos modernos al explicar word embeddings.

---

## 3. El método alternativo — PairDirection

Antes de proponer 3CosMul, Levy & Goldberg discuten una segunda objetivo que Mikolov usaba (sin documentarlo explícitamente):

$$
b^* = \arg\max_{x} \cos(x - b, \; a^* - a)
$$

Llamado **PairDirection**: buscar `x` tal que la dirección `x − b` sea paralela a `a* − a` (mismo offset relacional).

**Resultados (Tabla 2 del paper)**:

| Representation | MSR | GOOGLE | SEMEVAL |
|---|---|---|---|
| Embedding | 9.26% | 14.51% | **44.77%** |
| Explicit (PPMI) | 0.66% | 0.75% | **45.19%** |

PairDirection **funciona muy bien en SEMEVAL** (donde las opciones están restringidas a un conjunto pequeño) pero **fracasa estrepitosamente** en MSR y GOOGLE (vocabulario abierto). ¿Por qué?

Porque PairDirection solo se fija en la **dirección** del offset (vector relacional), pero ignora la **distancia espacial**. En vocabulario abierto, hay muchas palabras con la dirección correcta pero que no se parecen a `b` ni a `a*`. Ejemplo: en "man:woman :: king:?" puedes encontrar muchas palabras femeninas que no son `queen`.

→ **PairDirection es útil solo cuando las candidatas ya están pre-seleccionadas** (como SEMEVAL). En vocabulario abierto, **3CosAdd domina**.

---

## 4. **El núcleo: 3CosMul (Sección 6, fórmula 4)** ★

Esta es la fórmula que el notebook te pide consultar. Toda la sección 6 está dedicada a explicar **por qué 3CosAdd falla** en algunos casos y **cómo 3CosMul lo corrige**.

### 4.1 El problema con 3CosAdd — el "soft-or"

3CosAdd es una **suma lineal** de tres similitudes:

$$
\cos(x, b) - \cos(x, a) + \cos(x, a^*)
$$

Una suma lineal tiene la propiedad conocida como **"soft-or"**: si **un solo término es lo suficientemente grande**, domina la expresión completa, opacando a los otros. Esto es problemático en analogías porque **cada término refleja un aspecto distinto** de similitud, **con escalas distintas**.

### 4.2 El ejemplo canónico: London-England-Baghdad-Iraq

Levy & Goldberg muestran un fallo concreto. La analogía:

> *"London is to England as Baghdad is to — ?"*

Respuesta correcta: **Iraq**. Pero con 3CosAdd, la respuesta del modelo es **Mosul** (otra ciudad iraquí). El cálculo en representación explícita (PPMI):

| Palabra | ↑ England (atractor) | ↓ London (repulsor) | ↑ Baghdad (atractor) | Suma |
|---|---|---|---|---|
| Mosul | 0.031 | 0.031 | **0.244** | 0.244 |
| Iraq | 0.049 | 0.038 | 0.206 | 0.217 |

**Lo que pasa**: el término `cos(x, Baghdad)` domina porque Mosul y Baghdad son **geográficamente** muy cercanos (similar contexto: ambas son ciudades iraquíes, mencionadas juntas). El aspecto "país" (similitud con England) es **mucho más débil** porque las palabras `Iraq` y `England` aparecen en contextos relativamente disjuntos.

La similitud con Baghdad (~0.2) **domina** la similitud con England (~0.05), y la suma se inclina hacia Mosul, que tiene la similitud geográfica más fuerte.

→ **Diagnóstico**: en una suma lineal, un término grande puede ocultar señales más débiles pero relevantes. Las analogías necesitan **balance**, no dominancia.

### 4.3 La solución — combinación multiplicativa

Levy & Goldberg proponen **multiplicar en vez de sumar**:

$$
\boxed{\;\;b^* = \arg\max_{x \in V} \; \frac{\cos(x, b) \cdot \cos(x, a^*)}{\cos(x, a) + \varepsilon}\;\;}
$$

con $\varepsilon = 0.001$ para evitar división por cero. Esta es la **ecuación (4) del paper**, **3CosMul**.

### 4.4 Por qué funciona — intuición logarítmica

Tomando logaritmo de la expresión multiplicativa:

$$
\log \frac{\cos(x,b)\cos(x,a^*)}{\cos(x,a) + \varepsilon} = \log\cos(x,b) + \log\cos(x,a^*) - \log(\cos(x,a) + \varepsilon)
$$

Es decir, en escala logarítmica 3CosMul es **una suma de log-similitudes**. ¿Qué hace el logaritmo?

- **Amplifica diferencias entre valores pequeños**: la diferencia entre 0.05 y 0.03 (log → -3.0 vs -3.5, brecha 0.5) es **más grande** que entre 0.5 y 0.7 (log → -0.69 vs -0.36, brecha 0.33).
- **Reduce diferencias entre valores grandes**: ya no domina el término con valor más alto.

En el ejemplo Baghdad-Iraq-Mosul:
- 3CosAdd: Mosul (0.244) > Iraq (0.217) → ❌
- 3CosMul: Iraq (0.259) > Mosul (0.236) → ✅

**El cambio cualitativo**: al amplificar las diferencias pequeñas (similitud con England), 3CosMul **vuelve a darle peso** al aspecto "país" que estaba siendo aplastado.

### 4.5 Detalle de implementación

3CosMul requiere que **todas las similitudes sean no-negativas** (para que la multiplicación tenga sentido). Esto trivialmente se cumple para PPMI (siempre ≥ 0). Para embeddings densos como Word2Vec, donde el coseno puede ser negativo:

> *"With embeddings, we transform cosine similarities to [0, 1] using (x + 1)/2 before calculating (4)."*

Esta es **exactamente la implementación de `gensim.models.KeyedVectors.most_similar_cosmul`** que usas en el lab. Si lees el source de gensim verás que aplica la transformación `(x+1)/2`.

---

## 5. Resultados experimentales

### Setup
- **Corpus**: Wikipedia inglesa, ~1.5B tokens, 77.5M oraciones.
- **Vocabulario**: 189.533 términos (umbral de mínimo 100 apariciones).
- **Ventana de contexto**: 2 palabras a cada lado (5-gram).
- **Embedding**: 600 dimensiones, Skip-gram + Negative Sampling 15.
- **Explicit (sparse)**: PPMI sobre la matriz |V|×|C|.

### Datasets
| Dataset | Tipo | Vocab |
|---|---|---|
| MSR (Mikolov 2013c) | 8.000 analogías morfo-sintácticas | Abierto |
| GOOGLE (Mikolov 2013a) | 19.544 analogías (14 categorías, mitad semánticas mitad sintácticas) | Abierto |
| SEMEVAL (Jurgens 2012) | Ranking de pares por similitud relacional | Cerrado |

### Tabla 3 del paper — la comparación clave

| Objetivo | Representación | MSR | GOOGLE |
|---|---|---|---|
| 3CosAdd | Embedding | 53.98% | 62.70% |
| 3CosAdd | Explicit | 29.04% | 45.05% |
| **3CosMul** | Embedding | **59.09%** (+5.11) | **66.72%** (+4.02) |
| **3CosMul** | Explicit | **56.83%** (+27.79) | **68.24%** (+23.19) |

**Dos observaciones devastadoras**:

1. **3CosMul mejora 3CosAdd consistentemente**, en ambas representaciones, en ambos datasets — entre +4% y +28% en accuracy absoluto.

2. **En GOOGLE, la representación EXPLÍCITA (PPMI sparse) supera al embedding** (68.24% vs 66.72%). **Esto es enorme**: significa que los embeddings neuronales **no son mejores que la matriz de co-ocurrencia clásica para capturar analogías**, una vez que ambas se usan con la métrica correcta (3CosMul).

> *"This suggests that the linguistic regularities apparent in neural embeddings are not a consequence of the embedding process, but rather are well preserved by it."*

### Breakdown por tipo de relación (Tabla 5)

| Relación | Embedding | Explicit |
|---|---|---|
| capital-common-countries | 90.51% | **99.41%** |
| capital-world | 77.61% | **92.73%** |
| city-in-state | 56.95% | **64.69%** |
| currency | 14.55% | 10.53% |
| family (gender) | **76.48%** | 60.08% |
| gram3-comparative | **86.11%** | 77.85% |
| gram4-superlative | 56.72% | **63.45%** |
| gram6-nationality-adjective | 89.37% | **90.56%** |

**Patrón**: las representaciones explícitas son **mejores en relaciones semánticas concretas** (geografía, plurales) mientras que los embeddings son mejores en **relaciones inflexionales abstractas** (familia, comparativos). Pero las diferencias son moderadas, no abismales.

---

## 6. La conexión directa con tu lab

### 6.1 `most_similar_cosmul` en gensim

El notebook del Práctico 18, celda 12, usa:

```python
google_wordvecs.most_similar_cosmul(positive=['woman','king'], negative=['queen'])
```

El nombre **`cosmul`** viene exactamente de **3CosMul**. La implementación en `gensim/models/keyedvectors.py` calcula:

$$
\text{score}(x) = \frac{\prod_{p \in \text{positive}} \cos_+(x, p)}{\prod_{n \in \text{negative}} \cos_+(x, n) + \varepsilon}
$$

donde $\cos_+(x, y) = (\cos(x,y) + 1) / 2$ (la transformación a [0,1]). Esta generalización a múltiples atractores y repulsores es trivial — extiende fórmula (4) del paper.

### 6.2 Sintaxis del notebook

```python
positive=['woman','king'], negative=['queen']
# significa: woman - queen + king ≈ ?
# o:        ? es atraído por woman y king, y repelido por queen
```

→ La interpretación es **"soft-or" multiplicativo balanceado**, exactamente como define el paper.

### 6.3 Por qué algunas analogías fallan en tu lab

Si una analogía no produce la respuesta esperada (Actividad 1: "use su creatividad y genere 3 ejemplos"), las razones más probables son:

| Causa | Cómo identificarla |
|---|---|
| Un atractor domina la dirección | Cambia a 3CosAdd (`most_similar`) y compara — si también falla, el problema es la dirección semántica. |
| Palabra fuera de vocabulario (OOV) | Excepción `KeyError: "word 'X' not in vocabulary"`. |
| Caja del corpus | Google News usa `Capitalized_Phrases`. Probar `New_York` no `new_york`. |
| Polisemia | El embedding mezcla varios sentidos (un solo vector por palabra). |

El paper anticipa la mayoría de estas limitaciones en su Sección 8 (Analysis).

---

## 7. Contribuciones formales

### 7.1 Equivalencia matemática

Bajo `||v|| = 1` para todo v:

$$
\arg\max_{x} \cos(x, b - a + a^*) \;\;\equiv\;\; \arg\max_x \big[\cos(x,b) - \cos(x,a) + \cos(x,a^*)\big]
$$

Esto no es trivial, requiere prueba (Sección 3.3 del paper). La consecuencia es que **la aritmética vectorial de Mikolov no es más que una combinación lineal de similitudes pairwise**, lo que demystifica el "vector arithmetic".

### 7.2 Refinamiento

$$
\arg\max_x \frac{\cos(x,b)\cdot\cos(x,a^*)}{\cos(x,a) + \varepsilon} \quad \text{(3CosMul)}
$$

Mejora del estado del arte en ambos benchmarks abiertos (MSR, GOOGLE).

### 7.3 Reproducción con representaciones explícitas

Resultado empírico: PPMI sparse vectors recuperan analogías **comparable a Word2Vec** cuando se usa 3CosMul. Esto desmitifica los embeddings neuronales.

---

## 8. Limitaciones reconocibles

- **Análisis solo en inglés** (Wikipedia EN).
- **Embedding único** (Skip-gram NEG-15, 600d). No exploran arquitecturas alternativas (CBOW, GloVe).
- **3CosMul requiere similitudes no-negativas** — la transformación `(x+1)/2` es ad-hoc. Otros estudios posteriores proponen alternativas (PairDistance, etc.).
- **Hiperparámetro `ε = 0.001`** elegido sin justificación formal.
- **Análisis de error superficial**: muestran que 11-15% de las analogías que cada representación resuelve son distintas, pero no caracterizan **por qué** una representación resuelve un patrón y la otra no en cada caso concreto.
- **No hay análisis de polisemia**: ¿qué pasa cuando `bank` significa banco-financiero vs banco-de-río?

---

## 9. Impacto y derivados

| Hito posterior | Conexión con este paper |
|---|---|
| Levy & Goldberg NeurIPS 2014 (SGNS = MF) | Sigue la misma línea: explicar la "magia" de Word2Vec con teoría clásica. **Está en tu carpeta `clase_18/papers/`.** |
| Pennington et al., GloVe (EMNLP 2014) | Inspirado en parte por las observaciones sobre estadística global. |
| Allen & Hospedales 2019 | Prueba teórica rigurosa de por qué `b−a+a*` funciona. **Está en tu carpeta `clase_18/papers/`.** |
| Ri, Lee & Verma 2023 | Demuestra que contrastive loss reproduce las propiedades de paralelogramo y refina la teoría. **Está en tu carpeta `clase_18/papers/`.** |
| gensim, fastText, todas las libs de embeddings | Implementan `most_similar_cosmul` siguiendo fórmula (4) de este paper. |

El paper tiene >2.500 citaciones (Google Scholar) y es **el referente canónico** para entender por qué funciona la aritmética de embeddings.

---

## 10. Conexión con el laboratorio

### Donde aparece en el notebook

- **Celda 11** (markdown introductorio del bloque de analogías) — cita explícita: *"ver la fórmula (4) en la Sección 6 de este artículo"*.
- **Celdas 12-18** — todos los `most_similar_cosmul(...)` ejecutan exactamente la fórmula (4) del paper.
- **Celdas 26-30** — `doesnt_match(...)` que no es directamente 3CosMul, pero usa el mismo mecanismo de similitud coseno como métrica.

### Por qué Mikolov estaba en `positive` y queen en `negative` en celda 12

```python
google_wordvecs.most_similar_cosmul(positive=['woman','king'], negative=['queen'])
```

Convención gensim: el resultado es el `b*` que es:
- **Atraído** por las palabras en `positive` (= `b` y `a*` en notación del paper).
- **Repelido** por las palabras en `negative` (= `a` en notación del paper).

Aplicado: woman (atractor) + king (atractor) − queen (repulsor) → **man** (espera).

### Para la Actividad 1 del lab

Cuando armes tus analogías, ten en cuenta:
1. **Equilibra la magnitud de las similitudes** — si un atractor está mucho más relacionado que los demás, dominará incluso con 3CosMul.
2. **Probar caja**: Google News normaliza nombres propios (`New_York`).
3. **Cuando falle**, calcula `cos(x, atractor1)`, `cos(x, atractor2)`, `cos(x, repulsor)` por separado para diagnosticar qué término está siendo aplastado.

### Para la Actividad 4 (teórica)

La pregunta del lab es *"¿cuál es la explicación de la propiedad `king − man + woman ≈ queen`?"*. Tienes dos lecturas obligadas para una respuesta completa:

1. **Este paper** — la explicación operativa: la aritmética vectorial es una combinación lineal de similitudes pairwise, balanceada por 3CosMul.
2. **Allen & Hospedales 2019** (`Allen-Hospedales-AnalogiesExplained-2019.md`) — la explicación rigurosa: prueba que la relación lineal proviene de la **forma de la matriz PMI** y de cómo Word2Vec/GloVe la factorizan.

Ambas son complementarias: Levy & Goldberg explican **qué** (operación matemática); Allen & Hospedales explican **por qué** (qué propiedad del corpus la hace emerger).

---

## 11. Lecturas relacionadas en tu carpeta

- [[Mikolov-Word2Vec-Efficient-2013]] — origen del problema (analogías observadas).
- [[Mikolov-Word2Vec-DistributedRepresentations-2013]] — análisis empírico de las analogías.
- [[Levy-Goldberg-SGNS-MF-2014]] — paper hermano de los mismos autores 6 meses después (SGNS = factorización PMI).
- [[Pennington-GloVe-2014]] — GloVe explota la misma idea (analogías ∼ ratios de probabilidades).
- [[Allen-Hospedales-AnalogiesExplained-2019]] — prueba teórica rigurosa de por qué las analogías funcionan.
- [[Ri-Lee-Verma-ContrastiveAnalogies-2023]] — extiende a parallel lines y contrastive loss.

---

## 12. Citas que valen recordar

> *"While (1) and (3) are equal, we find the intuition as to why (3) ought to find analogies clearer."*
> — Sección 3.3, sobre por qué reescribir como suma de similitudes ayuda a entender la magia.

> *"The linguistic regularities apparent in neural embeddings are not a consequence of the embedding process, but rather are well preserved by it."*
> — Conclusión de Sección 7, **el mensaje político del paper**.

> *"A known property of such linear objectives is that they exhibit a 'soft-or' behavior and allow one sufficiently large term to dominate the expression."*
> — Sección 6, motivación para pasar a multiplicativo.
