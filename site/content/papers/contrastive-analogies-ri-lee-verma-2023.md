---
title: "Contrastive Loss is All You Need to Recover Analogies as Parallel Lines"
weight: 258
math: true
---

{{< paper-card
    title="Contrastive Loss is All You Need to Recover Analogies as Parallel Lines"
    authors="Ri, Lee, Verma"
    year="2023"
    venue="arXiv:2306.08221 (ACL SRW)"
    pdf="/papers/contrastive-analogies-ri-lee-verma-2023.pdf" >}}
Demuestra que una pérdida contrastiva simple aplicada sobre co-ocurrencias **basta** para inducir analogías como líneas paralelas en el espacio de embeddings (Teorema 1), entrena 49× más rápido que SGNS/GloVe, y establece una relación precisa entre estadísticas del corpus y geometría resultante. Cambia el marco "analogías = paralelogramos" (Mikolov 2013) al marco más realista "analogías = líneas paralelas con factor ζ ≠ 1".
{{< /paper-card >}}

---

## Cambio de marco: paralelogramos → líneas paralelas

### La visión clásica (Mikolov 2013)

Desde Mikolov 2013 se asumió que las analogías son paralelogramos exactos:

$$
v_{\text{queen}} - v_{\text{king}} = v_{\text{woman}} - v_{\text{man}}
$$

Pero los embeddings reales **rara vez cumplen esto exactamente**. Schluter (2018), Linzen (2016), Fournier (2020) documentan que la igualdad falla en muchos casos.

### La visión refinada (Ri-Lee-Verma 2023)

Las analogías son **líneas paralelas con factor de escala**:

$$
v_{\text{woman}} - v_{\text{man}} = \zeta \cdot (v_{\text{queen}} - v_{\text{king}})
$$

para algún $\zeta \in \mathbb{R}$.

- Si $\zeta = 1$ → paralelogramo (caso ideal).
- Si $\zeta \neq 1$ → **trapezoide** con líneas paralelas pero distintas longitudes (caso típico).

---

## Ideas principales

### 1. Contrastive Word Model (CWM)

Pérdida tipo hinge contrastiva sobre triples (palabra central, ventana, negativa):

$$
\mathcal{L}_{CWM} = \sum_{c \in W} \sum_{w \in W} \#(c, w) \sum_{w' \in D_{c,w}} \left[ m - \underbrace{\hat{v}_c \cdot \hat{v}_w}_{\text{pull}} + \underbrace{\hat{v}_c \cdot \hat{v}_{w'}}_{\text{push}} \right]_+
$$

Donde $[\cdot]_+$ es la hinge function y $m$ es un margen (típicamente $m = 0.2$).

**Interpretación**:
- **Pull**: palabras que coocurren → vectores alineados.
- **Push**: palabras que no coocurren → vectores no alineados.
- **Margen $m$**: el ángulo entre vc y vw debe ser al menos $m$ radianes más pequeño que entre vc y vw'.

### 2. **Teorema 1 — la conexión estadísticas ↔ geometría**

**Enunciado** (Sección 4, página 4):

> Para todo cuádruple $(a, b, c, d) \in W$, si existe $\zeta \in \mathbb{R}$ tal que las estadísticas de coocurrencia satisfacen $\forall w \in W$:
>
> $$\frac{\#(a,w)}{\#(a)} - \frac{\#(b,w)}{\#(b)} = \zeta \cdot \left[\frac{\#(c,w)}{\#(c)} - \frac{\#(d,w)}{\#(d)}\right]$$
>
> entonces los vectores aprendidos cumplen: $\hat{v}_a - \hat{v}_b = \zeta (\hat{v}_c - \hat{v}_d)$.

**Lo que dice el teorema, en humano**: si las **co-ocurrencias diferenciales** de los dos pares son colineales, los **vectores diferencia** también lo son. La geometría es **consecuencia directa** de la estadística del corpus.

### 3. Skip-gram y GloVe también hacen push-pull

El paper muestra (Sección 3.2) que SGNS y GloVe **implícitamente ejecutan push-pull** con extras. La pregunta natural: ¿qué pasa si solo hacemos push-pull, sin extras?

**Respuesta**: el CWM funciona igual de bien que SGNS/GloVe en analogías y **49× más rápido** en entrenamiento (Tabla 1):

| Modelo | PCS | MSM | Training Time | Speedup |
|---|---|---|---|---|
| **CWM** | **0.677** | **0.469** | 0.59 hrs | **49×** |
| SGNS | 0.675 | 0.433 | 29.27 hrs | 1× |
| GloVe | 0.667 | 0.423 | 30.71 hrs | 0.91× |

---

## Verificación empírica del Teorema 1

### Sección 5.3 — ¿Existe ζ en datos reales?

Calculan la similitud coseno entre vectores de coocurrencia $\vec{C}_{a,b}$ y $\vec{C}_{c,d}$ para tres tipos de cuádruples:

- **Random**: $(a,b,c,d)$ aleatorios → coseno ≈ 0 (no colineales).
- **Shuffled**: pares de analogía mezclados → coseno ≈ 0.
- **Analogy**: pares de analogía verdadera → coseno **cercano a 1** (colineales).

→ **Las analogías sí satisfacen la condición del Teorema 1 en datos reales** mientras que cuádruples aleatorios o mezclados no.

### Sección 5.4 — ζ y geometría

| Cuádruples con... | Top-1 | Top-5 (paralelogramo recovery) |
|---|---|---|
| $\hat{\zeta} \approx 1$ | 0.652 | **0.871** |
| $\hat{\zeta} \not\approx 1$ (trapezoide) | **0.800** | 0.862 |

→ Cuando $\hat{\zeta} \approx 1$ los embeddings recuperan paralelogramos con 87% accuracy top-5. Cuando $\hat{\zeta} \neq 1$ recuperan trapezoides en lugar de paralelogramos.

---

## Limitaciones reconocibles

- **Solo inglés** (Wikipedia EN dump).
- **Embeddings estáticos únicamente** — no analizan BERT/GPT.
- **Push-pull es suficiente pero no necesario**: pueden existir otros mecanismos que produzcan la misma estructura.
- **Asume normalización**: pruebas asumen $\|v\| = 1$.

---

## Conexión con el laboratorio

Esta es **la teoría que explica empíricamente** lo observado en el Práctico 18:

- **Lab Celda 46** (king/queen/man/woman): calculé $\hat{\zeta} \approx 1.16$ entre `woman-man` y `queen-king` (cosenos ≈ 1.0, magnitudes distintas) → **trapezoide, no paralelogramo**. Confirmación directa del Teorema 1.
- **Lab Plot 3** (empresas-productos): las 4 líneas empresa→producto son visualmente paralelas en 2D PCA con cosenos ≈ 0.89 pero solo ≈ 0.28 en 300D — **PCA infla artificialmente el paralelismo**, pero la propiedad existe (correlación débil pero significativa, 3-7 desviaciones estándar sobre ruido aleatorio).
- **Limitaciones observadas en lab**: la propiedad $\zeta$ se rompe ante polisemia (Santiago, Brussels, Madrid) porque distribuye co-ocurrencias entre múltiples sentidos.

---

## Cross-links

{{< cards >}}
  {{< card link="/laboratorios/lab-18" title="Lab 18 - Word Embeddings" subtitle="Verificación empírica del Teorema 1 (ζ=1.16)" icon="academic-cap" >}}
  {{< card link="/papers/analogies-explained-allen-hospedales-2019" title="Allen-Hospedales 2019" subtitle="Teoría previa sobre paralelogramos" icon="document-text" >}}
  {{< card link="/papers/linguistic-regularities-levy-goldberg-2014" title="Levy-Goldberg 2014 CoNLL" subtitle="3CosMul - capa operacional" icon="document-text" >}}
  {{< card link="/papers/sgns-implicit-mf-levy-goldberg-2014" title="Levy-Goldberg 2014 NeurIPS" subtitle="SGNS = factorización PMI" icon="document-text" >}}
{{< /cards >}}
