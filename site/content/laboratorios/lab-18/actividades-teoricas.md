---
title: "Actividades teóricas (4, 5, 6, 7)"
weight: 50
math: true
---

Las 4 actividades teóricas del Práctico 18 con respuestas consolidadas, basadas en los datos cuantitativos verificados durante el lab y citas verbatim de papers descargados.

## Actividad 4 — ¿Por qué emerge la propiedad de álgebra semántica?

**Pregunta**: ¿Cuál es la explicación de la propiedad de álgebra semántica de Word2Vec? ¿A qué se debe que operando algebraicamente con vectores de palabras podamos hacer analogías, detectar términos excluidos, entre otras cosas?

### Respuesta

La propiedad se debe a una combinación de cuatro factores:

**1. Principio distribucional implementado vía Skip-gram + Negative Sampling**

Word2Vec entrena vectores tales que palabras en contextos similares quedan cercanas. [Mikolov et al. (2013, NeurIPS)](/papers/word2vec-distributed-mikolov-2013) introducen Negative Sampling: para cada par observado se maximiza la probabilidad de cercanía; para pares aleatorios se minimiza.

**2. Equivalencia con factorización implícita de PMI**

[Levy & Goldberg (2014, NeurIPS)](/papers/sgns-implicit-mf-levy-goldberg-2014) demuestran formalmente:

> *"We analyze skip-gram with negative-sampling (SGNS), a word embedding method introduced by Mikolov et al., and show that it is implicitly factorizing a word-context matrix, whose cells are the pointwise mutual information (PMI) of the respective word and context pairs, shifted by a global constant."*

→ Word2Vec NO es un modelo neuronal "mágico" — es factorización implícita de estadísticas conocidas en distributional semantics desde los 80s.

**3. Las regularidades estadísticas del corpus producen geometría paralela**

[Ri, Lee & Verma (2023)](/papers/contrastive-analogies-ri-lee-verma-2023), **Teorema 1**:

> *"For any quadruple of words a, b, c, d ∈ W, if there exists a constant ζ ∈ R where the co-occurrence statistics satisfy the condition: ∀w ∈ W,*
> $$\frac{\#(a,w)}{\#(a)} - \frac{\#(b,w)}{\#(b)} = \zeta \cdot \left[\frac{\#(c,w)}{\#(c)} - \frac{\#(d,w)}{\#(d)}\right]$$
> *then the corresponding word vectors satisfy the property:*
> $$\hat{v}_a - \hat{v}_b = \zeta (\hat{v}_c - \hat{v}_d)$$

Si $\zeta = 1$ → paralelogramo exacto. Si $\zeta \neq 1$ → líneas paralelas con factor de escala.

**4. La operación 3CosMul**

[Levy & Goldberg (2014, CoNLL)](/papers/linguistic-regularities-levy-goldberg-2014), fórmula (4):

$$
b^* = \arg\max_{x} \frac{\cos(x,b) \cdot \cos(x,a^*)}{\cos(x,a) + \varepsilon}
$$

es combinación **multiplicativa** robusta a desbalances ("soft-or"), mejor que la versión aditiva original de Mikolov.

### Verificación empírica propia

| Experimento | Hallazgo |
|---|---|
| Coseno de direcciones `pasado_verbal − infinitivo` para 4 verbos | Rango 0.15-0.60, media 0.35 → **no hay eje "pasado" universal** |
| Coseno de direcciones `capital − país` para 5 capitales monosémicas | Rango 0.40-0.72, media 0.56 → eje débil pero **Brussels outlier por polisemia funcional** |
| Plot canónico king/queen/man/woman | **ζ ≈ 1.16** (no exactamente 1) → confirmación directa del Teorema 1 |
| Plot empresas-productos: cosenos 2D vs 300D | **0.89 (2D) vs 0.28 (300D)** → PCA infla artificialmente el paralelismo |

### Limitaciones empíricas (4 tipos de polisemia identificados)

1. **Polisemia ortográfica**: `Santiago` (capital + apellido).
2. **Polisemia léxica**: `play` (deportes + teatro + audio).
3. **Polisemia funcional**: `Brussels` (capital + sede UE).
4. **Sesgo temporal**: `Windows_Mobile` (descontinuado 2010) supera a `Windows_Phone` (vigente 2013).

### Síntesis

**No hay magia en Word2Vec**. Lo que parece álgebra semántica es la manifestación geométrica de patrones estadísticos del corpus, capturados por factorización implícita de matrices y consultados con operaciones de coseno multiplicativas. La propiedad es frágil ante polisemia y sesgo temporal, ambas heredadas del corpus.

---

## Actividad 5 — Suma vs Promedio en sentiment

**Pregunta**: Compare las dos estrategias de combinación. ¿Cuál se desempeñó mejor y por qué?

### Resultados cuantitativos verificados

| Métrica | SUMA | PROMEDIO |
|---|---|---|
| Accuracy train | 0.985 | 0.938 |
| MAE train | 0.0374 | 0.1356 |
| **MAE test** | 0.3147 | **0.2884** |
| Gap train→test | 0.277 | **0.153** |
| Loss final | 0.053 | 0.187 |

### Conclusión

**PROMEDIO supera a SUMA** en test por 8.4% y tiene gap train-test 45% menor. La explicación es que SUMA usa la magnitud del vector (proporcional a longitud del tweet) como atajo para memorizar el training, pero ese atajo no transfiere al test.

### Contextualización del MAE 0.288

| Estrategia | MAE | Comentario |
|---|---|---|
| Siempre 0.5 | 0.36 | Baseline trivial |
| **PROMEDIO** | **0.288** | **20% mejor que baseline** |
| State-of-art BERT | ~0.15-0.20 | Lejos del techo moderno |

→ Aceptable como baseline introductorio, pero margen significativo respecto al state-of-art.

---

## Actividad 6 — Estrategias alternativas de combinación

**Pregunta**: ¿Se le ocurren otras estrategias para combinar N vectores de palabras en 1 vector de tweet?

### Tres propuestas que respetan la restricción (N vectores → 1 vector)

**1. Promedio ponderado por TF-IDF**

Ponderar cada vector por su importancia TF-IDF en el corpus:

$$v_{\text{tweet}} = \frac{\sum_w \text{tfidf}(w) \cdot v_w}{\sum_w \text{tfidf}(w)}$$

Palabras raras y distintivas (awesome, terrible) contribuyen más. Sigue conmutativa.

**2. Max-pooling (o concatenación mean + max + min)**

Tomar el máximo componente a componente, o concatenar varios estadísticos:

$$v_{\text{tweet}} = [\text{mean}(v_w); \text{max}(v_w); \text{min}(v_w)]$$

Resultado: 900 dimensiones. Preserva los picos más fuertes en cada dimensión semántica.

**3. Encoder secuencial (RNN/LSTM)**

Procesar como secuencia ordenada con estado interno. **Rompe la conmutatividad**:

```
h_0 = vector cero
for w en tweet (en orden):
    h_t = LSTM(v_w, h_{t-1})
v_tweet = h_final
```

→ "not good" y "good not" producen vectores distintos. Captura dependencias largas.

### Comparación

| Estrategia | Captura orden | Conmutativa | Parámetros extra |
|---|---|---|---|
| Promedio (baseline) | No | Sí | No |
| TF-IDF weighted | No | Sí | No |
| Max-pooling / concat | No | Sí | No |
| LSTM encoder | **Sí** | **No** | Muchos |

---

## Actividad 7 — Mejoras al preprocesamiento

**Pregunta**: Proponga al menos 2 mejoras y justifique si son compatibles con Google News Word2Vec o requieren reemplazar el embedder.

### Tres mejoras

**Mejora 1 — Preservar negaciones**

**Problema**: NLTK stopwords incluye `not`, `no`, `don`, `t`. En tweet #343 observé "Not Fun & Furious" → "fun furious" (inversión semántica).

```python
stop_words = set(stopwords.words('english')) - {
    'not', 'no', 'never', 'nor', 'none', 'nobody', 'nothing'
}
```

**¿Reemplaza el embedder?** ❌ NO. Las palabras `not`, `no`, `never` están en Google News con vectores propios. Solo cambia el texto, no el embedder.

**Mejora 2 — Normalizar repeticiones e informalidad**

**Problema**: tokens informales como `awww`, `noooo`, `shoulda` son OOV. Tasa OOV ~22-25%.

```python
# Colapsar repeticiones de 3+ letras a 2
text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # awww → aww, noooo → noo

# Diccionario de contracciones informales
informal_map = {'shoulda': 'should have', 'gonna': 'going to',
                'wanna': 'want to', 'gotta': 'got to'}
```

**¿Reemplaza el embedder?** ❌ NO. Convertimos formas informales a sus equivalentes formales (`hello`, `should`, `going`) que sí existen en Google News.

**Mejora 3 — Usar embeddings entrenados sobre Twitter**

**Problema**: mismatch dominio (prensa formal vs Twitter informal). Causa la alta tasa de OOV.

**Propuesta**: reemplazar Google News Word2Vec por **GloVe-Twitter** (entrenado sobre 2 mil millones de tweets, vocabulario de 1.2M incluyendo slang, hashtags).

**¿Reemplaza el embedder?** ✅ SÍ. Requiere reemplazar el embedder completo:
- Vectores en espacios distintos (direcciones semánticas no se mezclan entre entrenamientos).
- Vocabulario distinto (`lol`, `omg`, `wtf` solo en GloVe-Twitter).
- Habría que recalcular `X_train`/`X_test`; el MLP se puede mantener con reentrenamiento.

**Aclaración**: NO requiere entrenar desde cero. Existen modelos preentrenados sobre Twitter disponibles públicamente (GloVe-Twitter, fastText-Twitter, BERTweet).

### Síntesis

| Mejora | Cambio | ¿Reemplaza embedder? |
|---|---|---|
| 1. Preservar negaciones | Lista de stopwords | No (sigue Google News) |
| 2. Normalizar repeticiones | Función de limpieza | No (sigue Google News) |
| 3. Embeddings de Twitter | Cambiar embedder | Sí (preentrenado, no desde cero) |

---

## Cross-links

{{< cards >}}
  {{< card link="../" title="← Lab 18 - Hub" subtitle="Volver al índice del lab" icon="academic-cap" >}}
  {{< card link="../sentiment-analysis" title="Bloque 4 - Sentiment" subtitle="Los datos que respaldan Act 5" icon="academic-cap" >}}
  {{< card link="../visualizacion-pca" title="Bloque 3 - PCA" subtitle="Los datos que respaldan Act 4" icon="academic-cap" >}}
  {{< card link="/papers/contrastive-analogies-ri-lee-verma-2023" title="Teorema 1 (Ri-Lee-Verma)" subtitle="Base teórica para Act 4" icon="document-text" >}}
{{< /cards >}}
