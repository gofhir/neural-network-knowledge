# Ri, Lee & Verma 2023 — Contrastive Loss is All You Need to Recover Analogies as Parallel Lines

| Campo | Valor |
|---|---|
| **Autores** | Narutatsu Ri, Fei-Tzin Lee, Nakul Verma |
| **Afiliación** | Columbia University, New York, USA |
| **Venue** | arXiv (también aceptado en ACL SRW 2023, *Student Research Workshop*) |
| **Fecha** | 14 junio 2023 (v1) |
| **Pdf** | `Ri-Lee-Verma-ContrastiveAnalogies-2023.pdf` (~10 páginas + apéndice) |
| **arXiv** | https://arxiv.org/abs/2306.08221 |
| **Código** | https://github.com/narutatsuri/cwm |
| **Cita** | Ri, N., Lee, F.-T., & Verma, N. (2023). Contrastive Loss is All You Need to Recover Analogies as Parallel Lines. *arXiv:2306.08221*. |

> *"We find that an elementary contrastive-style optimization employed over distributional information performs competitively with popular word embedding models on analogy recovery tasks, while achieving dramatic speedups in training time. Further, we demonstrate that a contrastive loss is sufficient to create these parallel structures in word embeddings."*

Este es el paper **más reciente** que el profesor Pablo Messina referencia en las slides de la clase 18 (link explícito: `https://arxiv.org/abs/2306.08221`). Junto a Allen & Hospedales 2019, conforma la lectura "estado del arte" sobre **por qué los word embeddings codifican analogías como estructuras geométricas paralelas**. Su contribución técnica: **Contrastive Word Model (CWM)**, un modelo entrenado con pérdida tipo hinge contrastiva que (1) alcanza la misma calidad de analogías que SGNS y GloVe, (2) entrena **49× más rápido**, y (3) viene con prueba teórica (Teorema 1) que vincula directamente las **estadísticas de co-ocurrencia** del corpus con la **estructura geométrica paralela** de los embeddings resultantes.

---

## 1. El cambio de marco — paralelogramos vs líneas paralelas

### 1.1 La visión clásica: analogías como paralelogramos

Desde Mikolov 2013 se asumió que las analogías están codificadas como **paralelogramos**:

```
   man ──────────── woman
    │                │
    │ (gender)        │
    │                │
   king ──────────── queen
```

Formalmente, esto implica:

$$
v_{\text{woman}} - v_{\text{man}} = v_{\text{queen}} - v_{\text{king}}
$$

Esto exige que los cuatro vectores formen un **paralelogramo exacto** en ℝ³⁰⁰. Es una condición geométricamente fuerte y, como mostraron varios trabajos críticos (Schluter 2018, Linzen 2016, Fournier 2020), **rara vez se cumple en los embeddings reales**.

### 1.2 La visión moderna: analogías como líneas paralelas

Arora et al. (2016, 2019) y Fournier et al. (2020) proponen un modelo **más débil pero más realista**: las analogías son **líneas paralelas** (no paralelogramos cerrados):

```
   man ────────► woman          run ────────► running
        ↑     gender                  ↑     +ing
        │                              │
   king ────────► queen          scream ────► screaming
        ↑     gender                  ↑     +ing
```

Formalmente:

$$
v_{\text{woman}} - v_{\text{man}} = \zeta \cdot (v_{\text{queen}} - v_{\text{king}})
$$

para algún escalar $\zeta \in \mathbb{R}$. **Si $\zeta = 1$, las líneas forman paralelogramo. Si $\zeta \neq 1$, forman trapezoide**, pero las direcciones siguen siendo paralelas.

→ Este es el cambio de marco central del paper. **Mismo paralelismo, distintas longitudes**.

### 1.3 Por qué importa para el lab

En el Práctico 18, cuando ejecutas `most_similar_cosmul(positive=['woman','king'], negative=['queen'])` y el resultado tiene similitud `0.7` en vez de `1.0`, lo que estás viendo NO es un paralelogramo exacto. Estás viendo **una línea paralela aproximada** con `ζ ≠ 1`. **Es la regla, no la excepción**. La visión paralelogramo era demasiado idealista.

---

## 2. Por qué este paper es importante

| Trabajo previo | Pregunta abierta que dejaba |
|---|---|
| Mikolov 2013 | Observa empíricamente que las analogías funcionan. **No explica**. |
| Levy & Goldberg 2014 (CoNLL) | Reescribe 3CosAdd como suma de similitudes y propone 3CosMul. **No explica geometría**. |
| Levy & Goldberg 2014 (NeurIPS) | SGNS = factorización implícita de PMI. **Explica el espacio pero no la analogía**. |
| Arora et al. 2016 | Modelo de variable latente para PMI-embeddings. Asume estructura geométrica. **Asume, no prueba**. |
| Gittens et al. 2017 | Parafraseo + uniformidad de palabras → vector additivity. **Restrictivo**. |
| Allen & Hospedales 2019 | Prueba que analogías son paralelogramos bajo condiciones específicas de PMI. **Asume modelo PMI**. |
| Ethayarajh et al. 2019 | csPMI (PMI shifted con log-probabilidades) y condiciones. **Refinamiento técnico**. |
| **Ri, Lee & Verma 2023** | **Prueba que contrastive loss SOLO basta para inducir líneas paralelas, dada una condición simple sobre conteos.** |

Lo nuevo del paper:

1. **Modelo más simple** que SGNS o GloVe — solo contrastive loss tipo hinge, sin softmax, sin reweighting.
2. **Análisis teórico (Teorema 1)** que conecta **directamente** una propiedad de las co-ocurrencias del corpus con la geometría resultante.
3. **Verificación empírica** de que las condiciones del teorema **realmente se cumplen** en datos reales (Sección 5.3).
4. **Performance competitivo** + **speedup de 49×** sobre SGNS/GloVe.

→ Es el paper **más completo end-to-end** sobre la teoría de analogías al 2023.

---

## 3. El Contrastive Word Model (CWM)

### 3.1 Notación

| Símbolo | Significado |
|---|---|
| $W$ | Vocabulario |
| $\#(i)$ | Cuenta de ocurrencias de palabra $i$ |
| $\#(i, j)$ | Co-ocurrencias de $i$ y $j$ en ventana $\Delta$ |
| $c$ | **Center word** (palabra de referencia) |
| $w$ | **Window word** (palabra que co-ocurre con $c$) |
| $w'$ | **Negative window word** (palabra que NO co-ocurre con $c$, muestreada uniformemente) |
| $D_{c,w}$ | Conjunto de negativas para el par $(c, w)$ |
| $v_w$ | Vector de la palabra $w$ |
| $\hat{v}_w$ | Versión normalizada: $v_w / \|v_w\|$ |

### 3.2 La pérdida CWM

$$
\mathcal{L}_{CWM} = \sum_{c \in W} \sum_{w \in W} \#(c, w) \sum_{w' \in D_{c,w}} \left[ m - \underbrace{\hat{v}_c \cdot \hat{v}_w}_{\text{pull}} + \underbrace{\hat{v}_c \cdot \hat{v}_{w'}}_{\text{push}} \right]_+
$$

donde $[\cdot]_+$ es la función hinge ($\max(0, \cdot)$) y $m$ es un margen.

**Interpretación**:
- **Pull**: queremos que $\hat{v}_c \cdot \hat{v}_w$ sea **alto** (palabras que co-ocurren → vectores alineados).
- **Push**: queremos que $\hat{v}_c \cdot \hat{v}_{w'}$ sea **bajo** (palabras que no co-ocurren → vectores no alineados).
- **Margen**: la pérdida se anula cuando `pull − push ≥ m`. Es decir, el ángulo entre $v_c$ y $v_w$ debe ser **al menos $m$ radianes más pequeño** que el ángulo entre $v_c$ y $v_{w'}$.

Esta es **exactamente la pérdida triplet/contrastive** clásica de la literatura de metric learning (Chopra-Hadsell-LeCun 2005, Weinberger 2005), adaptada al setting de palabras.

### 3.3 Por qué es contrastivo y nada más

A diferencia de SGNS o GloVe:
- **No hay softmax** sobre el vocabulario (Word2Vec) — solo dot products.
- **No hay reweighting** complejo (GloVe usa $f(\#(i,j))$) — solo conteos lineales.
- **No hay subsampling** de palabras frecuentes.
- **No hay phrase preprocessing**.

Esta minimalidad es deliberada — es lo que permite el speedup de 49× y simplifica el análisis teórico.

### 3.4 Relación con SGNS y GloVe

El paper muestra (sección 3.2) que **Skip-gram y GloVe también ejecutan implícitamente un push-pull**:

**Skip-gram (gradiente del log-likelihood):**
$$
v_c^{\text{new}} = v_c^{\text{old}} + \underbrace{\left(1 - \frac{e^{v_w^\top u_{c'}}}{\sum_{w'} e^{v_w^\top u_{w'}}}\right) v_w}_{\text{pull hacia } w} - \underbrace{\mathbb{E}_{w' \sim W}[v_{w'}]}_{\text{push contra promedio}} + \text{terms}
$$

**GloVe**: tres updates simultáneos sobre $c$, $w$, $w'$:
- $v_c^{\text{new}} = v_c^{\text{old}} + g(c, c')u_{c'}$ (pull entre $c$ y $c'$)
- $v_w^{\text{new}} = v_w^{\text{old}} + g(w, c')u_{c'}$ (pull entre $w$ y $c'$)
- $v_{w'}^{\text{new}} = v_{w'}^{\text{old}} - g(w', c')u_{c'}$ (push de $w'$ lejos de $c'$)

> *"We believe that part of the success of these word embedding models is due to their implicit push-pull dynamics. Hence, a natural question to consider is what happens when one purely optimizes for the push-pull action alone."*

→ **Si SGNS y GloVe funcionan porque hacen push-pull con extras**, ¿qué pasa si **solo** hacemos push-pull? Respuesta: CWM, y funciona igual de bien.

---

## 4. **El teorema central** (Teorema 1)

Este es el resultado teórico más importante del paper.

### 4.1 Vector óptimo bajo CWM

Bajo asunciones razonables (Apéndice A.1 del paper), el vector $v_c$ que minimiza la pérdida CWM tiene la forma:

$$
v_c = \rho_c \left[ \sum_{w \in W} \frac{\#(c, w)}{\#(c)} \hat{v}_w - \mathbb{E}_{w' \sim U(W)}[\hat{v}_{w'}] \right]
$$

Es decir: $v_c$ es proporcional a (**promedio ponderado de vecinos por frecuencia condicional**) menos (**promedio uniforme del vocabulario**). Esta forma es la receta canónica de los modelos de "Continuous Bag of Words" pero derivada **desde primeros principios**.

### 4.2 Teorema 1 (parafraseado)

**Enunciado**: Sea $(a, b, c, d)$ un cuádruple de palabras. Si existe una constante $\zeta \in \mathbb{R}$ tal que para **toda palabra $w \in W$**:

$$
\frac{\#(a, w)}{\#(a)} - \frac{\#(b, w)}{\#(b)} = \zeta \cdot \left[\frac{\#(c, w)}{\#(c)} - \frac{\#(d, w)}{\#(d)}\right]
$$

entonces los embeddings aprendidos por CWM cumplen:

$$
\hat{v}_a - \hat{v}_b = \zeta \cdot (\hat{v}_c - \hat{v}_d)
$$

### 4.3 Qué dice el teorema, en humano

Define el **"vector de co-ocurrencia diferencial"** entre dos palabras $a$ y $b$:

$$
\vec{C}_{a,b} = \left(\frac{\#(a, w_1)}{\#(a)} - \frac{\#(b, w_1)}{\#(b)}, \;\;...,\;\; \frac{\#(a, w_{|W|})}{\#(a)} - \frac{\#(b, w_{|W|})}{\#(b)}\right) \in \mathbb{R}^{|W|}
$$

Cada entrada del vector mide *"cuánto más probable es que $a$ co-ocurra con $w_i$ que $b$"*. Entonces:

- **Si $\vec{C}_{a,b}$ y $\vec{C}_{c,d}$ son colineales** (uno es múltiplo escalar del otro con factor $\zeta$), entonces $\hat{v}_a - \hat{v}_b$ y $\hat{v}_c - \hat{v}_d$ también son colineales con el mismo factor $\zeta$.

- **Si $\zeta = 1$** → paralelogramo (Mikolov-style).
- **Si $\zeta \neq 1$** → trapezoide / líneas paralelas.

→ **El teorema reduce la pregunta geométrica a una pregunta estadística sobre el corpus**.

### 4.4 Por qué este teorema es el "santo grial"

Hasta este paper, los análisis teóricos requerían asunciones complicadas sobre el modelo probabilístico subyacente (Arora et al. asumen un proceso latente; Allen & Hospedales asumen estructura PMI). Aquí la asunción es **directamente verificable**: solo necesitas contar co-ocurrencias en tu corpus y comprobar si existe $\zeta$.

Es **el puente más limpio** entre **estadística observada del corpus** y **geometría aprendida del embedding**.

---

## 5. Resultados experimentales

### 5.1 Setup

- **Corpus**: Wikimedia dump de marzo 2023.
- **Ventana**: $\Delta = 5$.
- **Margen**: $m = 0.2$ (validado por cross-validation en [0.1, 1]).
- **Dimensión**: $D = 300$.
- **Vectores normalizados** a $\|v\| = 1$.
- **Hardware**: 256 instancias AMD EPYC 7763.
- **Comparación**: SGNS y GloVe con parámetros default.
- **Dataset de analogías**: BATS (Gladkova et al. 2016) — más diverso que el clásico GOOGLE de Mikolov.

### 5.2 Métricas de alineamiento

Usan dos métricas propuestas por Fournier et al. (2020), que **no asumen paralelogramo**:

| Métrica | Qué mide |
|---|---|
| **PCS** (Pairing Consistency Score) | Precisión: ¿se alinean correctamente solo los pares de analogía verdadera y no los falsos? |
| **MSM** (Mean Similarity Measure) | Magnitud: ¿qué tan alineados están los pares de analogía verdadera (en promedio)? |

### 5.3 Tabla 1 del paper — resultados principales

| Modelo | PCS | MSM | Training Time (hrs) | Speedup |
|---|---|---|---|---|
| **CWM** | **0.677** | **0.469** | **0.59** | **49×** |
| SGNS | 0.675 | 0.433 | 29.27 | 1× |
| GloVe | 0.667 | 0.423 | 30.71 | 0.91× |

**Conclusiones**:
- En **PCS** (precisión): CWM ≈ SGNS > GloVe — empate técnico.
- En **MSM** (magnitud): CWM **supera 7-8%** a SGNS y GloVe — los pares de analogía están **más fuertemente alineados** en CWM.
- En **tiempo**: CWM tarda **35 minutos** contra **29-31 horas** de SGNS/GloVe. **49× más rápido**.

Esto es **un golpe técnico fuerte**: si una pérdida 5 líneas es 49× más rápida y produce embeddings con analogías mejor alineadas, ¿para qué seguir con SGNS y GloVe? La respuesta: SGNS y GloVe siguen dominando en otras tareas downstream (analogía no es todo), pero **para la propiedad de analogía específicamente, CWM domina**.

### 5.4 Verificación empírica de $\zeta$ (Sección 5.3)

**Pregunta**: ¿realmente existe ese $\zeta$ del Teorema 1 en datos reales?

**Setup**: Calcular $\vec{C}_{a,b}$ y $\vec{C}_{c,d}$ para tres tipos de cuádruples:
1. **Random**: $(a,b,c,d)$ aleatorios.
2. **Shuffled**: pares de analogía mezclados (no forman analogía).
3. **Analogy**: pares de analogía verdadera (BATS).

**Resultado (Figura 2 del paper)**:
- Random y Shuffled tienen $|\cos(\vec{C}_{a,b}, \vec{C}_{c,d})|$ cercano a **0** (no colineales).
- Analogy tiene $|\cos(\vec{C}_{a,b}, \vec{C}_{c,d})|$ cercano a **1** (colineales).

→ **Las analogías SÍ satisfacen la condición del Teorema 1 en datos reales**. El no-analogías **no**. Esto valida que el mecanismo predicho por la teoría es lo que está pasando en la práctica.

### 5.5 Estructura geométrica vs $\zeta$ (Sección 5.4)

**Pregunta**: si $\zeta \approx 1$ → paralelogramo. Si $\zeta \neq 1$ → trapezoide. ¿Se cumple esto en el embedding aprendido?

**Tabla 3 del paper**:

| Cuádruples con... | Top-1 accuracy | Top-5 accuracy |
|---|---|---|
| $\hat{\zeta} \approx 1$ (paralelogramo) | 0.652 | **0.871** |
| $\hat{\zeta} \not\approx 1$ (trapezoide) | **0.800** | 0.862 |

(Para todas las analogías sin filtrar: top-1 = 0.27)

**Lectura**: cuando uno **filtra solo cuádruples con $\hat{\zeta} \approx 1$**, la recuperación de paralelogramos sube de 27% a 87% (top-5). Cuando $\hat{\zeta} \neq 1$, las analogías se recuperan como trapezoides con accuracy similar. **Las dos predicciones del teorema se confirman empíricamente**.

### 5.6 Ejemplos cualitativos (Tabla 2)

**Cuádruples con alta similitud entre $\vec{C}_{a,b}$ y $\vec{C}_{c,d}$**:
- `fall:rise = under:over` (sim = 1.000) — antónimos espaciales, relación clara.
- `prevent:preventing = follow:following` (sim = 0.99) — present participle, regular.
- `lancaster:lancashire = salford:manchester` (sim = 0.98) — ciudad:condado.

**Cuádruples con baja similitud** (analogías "ambiguas" del BATS):
- `organized:arranged = dollars:bucks` (sim = 0.001) — sinónimos, pero de dominios distintos.
- `monkey:infant = fox:cub` (sim = 0.0001) — animal:cría, pero contextos disjuntos.

→ **Cuando la teoría falla**, falla en analogías donde **los contextos de las dos parejas son muy disjuntos**. Esto da una **prueba de diagnóstico**: si tu analogía no funciona en el lab, mira si los contextos de las dos parejas son similares.

---

## 6. Conexión con tu lab — diagnóstico de analogías fallidas

Esto es **directamente aplicable** a la Actividad 1 del Práctico 18.

### Caso 1: La analogía funciona ($\cos > 0.9$)
Felicitaciones: tus dos parejas tienen contextos similares en el corpus de Google News. El embedding las pone en líneas paralelas con $\zeta \approx 1$.

### Caso 2: La analogía falla ($\cos < 0.5$)
Probables causas, derivadas del paper:
1. **Contextos disjuntos**: las parejas viven en regiones del corpus que rara vez se mezclan. Ejemplo: `Chile:Santiago = Antártica:?`. En Google News, Chile y Santiago aparecen mucho juntos; Antártica aparece en otros contextos (ciencia, exploración) sin relación clara con su capital.
2. **Polisemia**: una palabra tiene múltiples sentidos que diluyen la dirección de la analogía. Ejemplo: `bank` (financial) vs `bank` (river). El vector queda en el centro de gravedad de ambos.
3. **Frecuencia desigual**: si una palabra es muy frecuente (alto $\#(c)$) y la otra muy rara, las estadísticas no son comparables.

### Caso 3: La analogía da una respuesta inesperada pero plausible
Esto es **trapezoide** ($\zeta \neq 1$). La dirección es correcta pero la magnitud no es 1:1. La respuesta del modelo es la palabra más cercana en esa dirección, no la palabra ideal. Ejemplo: `man:woman :: king:?` → puede dar `princess` o `monarch` en vez de `queen`.

---

## 7. Limitaciones reconocibles

El paper es honesto y discute (Sección 6):

- **Push-pull no es necesariamente la única forma** de inducir líneas paralelas. Otros mecanismos podrían producir el mismo resultado.
- **El Teorema 1 da condición suficiente**, no necesaria. Pueden existir cuádruples sin $\zeta$ exacto que igual aparecen como paralelos en el embedding.
- **Solo testean en inglés** (Wikipedia EN).
- **Embeddings estáticos** únicamente. No analizan BERT/GPT (los embeddings contextualizados de la clase 20).
- **Asume normalización**: todas las pruebas asumen $\|v\| = 1$. En la práctica, los embeddings sin normalizar pueden tener comportamientos distintos.
- **No conectan con la calidad downstream** (sentiment, NER). Pueden ser excelentes para analogías y mediocres para otras tareas.

---

## 8. Impacto y posición en el panorama

### Por qué el profesor lo incluyó en la slide

Pablo Messina explícitamente recomienda este paper junto a Allen-Hospedales 2019 en la slide 26 ("Embeddings: ..."). Las dos referencias se complementan:

| Paper | Contribución |
|---|---|
| Allen & Hospedales 2019 | Prueba que las analogías son paralelogramos bajo el modelo de Arora (PMI). |
| **Ri, Lee & Verma 2023** | Generaliza a líneas paralelas, prueba que basta con contrastive loss, valida empíricamente. |

Es **el cierre teórico moderno** sobre el problema que Mikolov abrió en 2013.

### Influencia hacia adelante

- Es **muy reciente** (2023), todavía está acumulando citaciones. Al momento del lab, su rol es más pedagógico que técnico: ilustra que **el campo sigue refinando la teoría de embeddings estáticos** incluso en la era de los LLMs.
- La pérdida CWM tiene posible aplicación en **embeddings multilingües** o **dominio-específicos** donde el speedup importa.

---

## 9. Conexión directa con el Práctico 18

### 9.1 Para la Actividad 4 (teórica)

**Pregunta del lab**: *"¿Cuál cree usted que sea la explicación de la propiedad mágica `king − man + woman ≈ queen`?"*

Respuesta canónica completa, en tres capas:

1. **Capa operativa** ([[Levy-Goldberg-LinguisticRegularities-2014]]): la aritmética vectorial es equivalente a buscar una palabra que maximiza una combinación de tres similitudes pairwise. La versión multiplicativa (3CosMul) es más robusta a desbalances.

2. **Capa estadística** (**ESTE PAPER**, Teorema 1): la propiedad emerge porque las **estadísticas de co-ocurrencia diferenciales** de las dos parejas $(a,b)$ y $(c,d)$ son colineales. Es decir, las dos parejas se relacionan con el resto del vocabulario de la misma manera.

3. **Capa de mecanismo** (**ESTE PAPER**, Sección 3.2): cualquier modelo que ejecute **push-pull contrastivo** sobre co-ocurrencias preservará esta colinealidad estadística como colinealidad geométrica. Skip-gram, GloVe y CWM son tres instancias distintas del mismo mecanismo.

### 9.2 Para la Actividad 1 (creatividad)

Cuando armes tus 3 analogías, intenta una de cada categoría:

| Tipo | Predicción | Ejemplo |
|---|---|---|
| Analogía estándar (contextos similares) | $\zeta \approx 1$, funciona | `Paris:France :: Tokyo:?` |
| Analogía con contextos disjuntos | $\zeta$ no existe o $\neq 1$, falla | `chemistry:beaker :: music:?` |
| Analogía con polisemia | Falla por mezcla de sentidos | `bat:mammal :: bass:?` |

Analiza por qué cada una funciona o falla **en términos del Teorema 1** — esto eleva tu respuesta del nivel "yo creo que..." al nivel "el corpus contiene/no contiene esta condición estadística".

---

## 10. Lecturas relacionadas en tu carpeta

- [[Mikolov-Word2Vec-Efficient-2013]] — origen del problema empírico.
- [[Mikolov-Word2Vec-DistributedRepresentations-2013]] — observaciones canónicas.
- [[Levy-Goldberg-LinguisticRegularities-2014]] — origen de 3CosMul, capa operativa.
- [[Levy-Goldberg-SGNS-MF-2014]] — SGNS como factorización implícita.
- [[Pennington-GloVe-2014]] — modelo competidor con push-pull explícito.
- [[Allen-Hospedales-AnalogiesExplained-2019]] — paralelogramos rigurosos.

---

## 11. Citas que valen recordar

> *"While static word embedding models are known to represent linguistic analogies as parallel lines in high-dimensional space, the underlying mechanism as to why they result in such geometric structures remains obscure."*
> — Abstract, **la motivación**.

> *"A contrastive loss is sufficient to create these parallel structures in word embeddings, and we establish a precise relationship between the co-occurrence statistics and the geometric structure of the resulting word embeddings."*
> — Abstract, **la contribución**.

> *"We believe that part of the success of these word embedding models is due to their implicit push-pull dynamics."*
> — Sección 3.2, **el insight unificador**.

> *"It remains unclear whether push-pull is a necessary condition for this phenomenon. Investigating alternative mechanisms and their ability to achieve similar results would provide further insight."*
> — Conclusión, **una pregunta abierta** para futura investigación.

---

## 12. Resumen ejecutivo (TL;DR)

- **Problema**: ¿por qué los word embeddings codifican analogías como líneas paralelas?
- **Respuesta clásica**: nadie sabía exactamente, varias teorías parciales.
- **Respuesta de este paper**: porque las **estadísticas de co-ocurrencia diferenciales** $\vec{C}_{a,b}$ y $\vec{C}_{c,d}$ son **colineales** para analogías verdaderas, y los modelos contrastivos preservan esa colinealidad como colinealidad de vectores.
- **Bonus técnico**: una pérdida contrastiva minimal (CWM) reproduce SGNS/GloVe en analogías con **49× speedup**.
- **Relevancia para el lab**: explica al nivel más profundo el "porqué" de `king − man + woman ≈ queen`, complementando [[Levy-Goldberg-LinguisticRegularities-2014]] (el "qué" operativo) y [[Allen-Hospedales-AnalogiesExplained-2019]] (el "qué" geométrico clásico).
