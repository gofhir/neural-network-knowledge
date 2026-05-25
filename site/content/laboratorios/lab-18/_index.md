---
title: "Lab 18 - Word Embeddings: analogías, doesnt_match, PCA y sentiment analysis"
weight: 180
sidebar:
  open: true
---

**Profesor:** Pablo Messina
**Fecha:** Mayo 2026
**Notebook origen:** `clase_18/material/Laboratorio/Practico18.ipynb` (105 celdas)
**Notebook ejecutado:** [lab-18.ipynb](/notebooks/lab-18.ipynb) · [HTML](/notebooks-html/lab-18.html)

## Encuadre

Laboratorio que recorre **el ciclo completo del uso de word embeddings preentrenados**, organizado en dos partes:

- **Parte 1 — Propiedades intrínsecas** (Celdas 10-54): analogías con `most_similar_cosmul`, detección de outliers con `doesnt_match`, y visualización 2D con PCA. Comprueba empíricamente las propiedades geométricas de Word2Vec sobre Google News (3M palabras, 300 dim).
- **Parte 2 — Aplicación downstream** (Celdas 55-104): análisis de sentimiento de tweets (Sentiment140, 20k tweets sampleados de 1.6M) usando los embeddings como features para un MLP, comparando dos estrategias de combinación: **suma vs promedio**.

El lab demuestra de forma operativa los conceptos teóricos de la [clase 18](/clases/clase-18) y produce evidencia empírica directa para los papers asociados (Mikolov 2013, Levy-Goldberg 2014, Ri-Lee-Verma 2023).

## El modelo: Google News Word2Vec

- **3 millones de palabras y phrases** entrenadas con Skip-gram + Negative Sampling.
- **300 dimensiones** por vector.
- **Corpus**: 100 mil millones de palabras de Google News (artículos de prensa 2003-2013).
- **Cargado limitado a 100.000 palabras** (primeras 100k más frecuentes) para caber cómodamente en RAM.

```python
google_wordvecs = KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin.gz',
    binary=True, limit=100000
)
```

## Parte 1 — Operaciones algebraicas y visualización

### Analogías con 3CosMul

Implementación de la fórmula (4) de [Levy & Goldberg 2014 CoNLL](/papers/linguistic-regularities-levy-goldberg-2014):

$$
b^* = \arg\max_{x \in V} \frac{\cos_+(x, b) \cdot \cos_+(x, a^*)}{\cos_+(x, a) + \varepsilon}
$$

con $\varepsilon = 0.001$ y $\cos_+(u,v) = (\cos(u,v)+1)/2$. Es la operación que ejecuta `gensim.models.KeyedVectors.most_similar_cosmul`.

Resultados destacados de las 7 analogías guía + 3 propias:

| Analogía | Top-1 | Score | Calidad |
|---|---|---|---|
| `woman + king − queen` | man | 0.93 | Buena, con ruido de prensa policial |
| `actor + woman − man` | **actress** | 1.06 | Excelente, top-10 100% cine |
| `son + woman − man` | **daughter** | 1.05 | Gap top1-top2 = 0.04, husband/father intrusos |
| `Santiago + Venezuela − Chile` | **Caracas** | 0.95 | Gap 0.013, 8/10 apellidos hispanos |
| `Buenos_Aires + Chile − Santiago` | **Argentina** | 1.01 | Top-10 100% sudamericano |
| `saxophone + classical − jazz` | **cello** | 0.86 | Polisemia cultural (sitar/tabla) |
| `Yankees + basketball − baseball` | **Knicks** | 0.88 | Geografía NY > éxito legendario |
| `Microsoft + iPhone − Apple` | **Windows_Mobile** | 0.86 | Sesgo temporal (legacy > vigente) |

### doesnt_match (detección de outliers)

Operación independiente: calcula el **centroide** del grupo y devuelve la palabra **más lejana** del centroide.

| Grupo | Outlier | Test pedagógico |
|---|---|---|
| white, blue, red, Chile | Chile | Dominio disjunto trivial |
| April, May, September, Tuesday, July | Tuesday | Mes vs día (fino) |
| Lima, Paris, London, Madrid | Lima | Continente sudamericano |
| Microsoft, Apple, Google, **Amazon** | **Amazon** | Retail vs tech platform (sutil) |

→ El mecanismo de centroide es **más robusto a polisemia** que las analogías porque promedia múltiples vectores y suaviza el ruido individual.

### Visualización 2D con PCA

PCA reduce el espacio de 300 dimensiones a 2 — perdiendo **~93% de la varianza**. La proyección **es lossy** y a veces engañosa.

![Plot canónico: queen, king, woman, man](/laboratorios/lab-18/pca-king-queen-man-woman.png)

Calculando los vectores diferencia en este plot:

- `queen − king = (−0.097, −0.071)`, magnitud 0.120
- `woman − man = (−0.111, −0.083)`, magnitud 0.139
- $\cos(\vec{u}, \vec{v}) \approx 1.00$ (perfectamente paralelos)
- $\zeta = 0.139 / 0.120 \approx 1.16$ (no exactamente 1)

→ Confirmación empírica directa del **Teorema 1 de [Ri-Lee-Verma 2023](/papers/contrastive-analogies-ri-lee-verma-2023)**: las analogías son líneas paralelas con $\zeta \neq 1$, no paralelogramos exactos.

### Hallazgo crítico: PCA 2D infla artificialmente el paralelismo

Comparación de cosenos entre vectores diferencia empresa→producto en mi Plot 3 (Microsoft, Apple, Google, Amazon × Windows, iPhone, Android, Kindle):

| Par 1 | Par 2 | Coseno 2D | Coseno 300D |
|---|---|---|---|
| Microsoft→Windows | Apple→iPhone | 0.978 | **0.251** |
| Microsoft→Windows | Google→Android | 0.935 | **0.392** |
| Apple→iPhone | Google→Android | 0.989 | **0.321** |
| **Promedio** | | **0.893** | **0.278** |

El paralelismo en 2D (0.89) es **3.2× más alto** que en 300D (0.28). PCA preserva varianza global pero **no preserva ángulos** — las visualizaciones canónicas muestran una versión más limpia que la geometría real. Esto refuerza la advertencia del paper de Ri-Lee-Verma: la propiedad de paralelogramo opera en 300D, no en proyecciones 2D.

## Parte 2 — Sentiment analysis con MLP

### Pipeline

```
Sentiment140 (1.6M tweets) → sample 20k
  ↓
Limpieza (regex + BeautifulSoup + stopwords) → 6.9 palabras/tweet (mediana)
  ↓
Tweet vectors:
  - SUMA:    Σ v_w (magnitudes [0, 40])
  - PROMEDIO: Σ v_w / N (magnitudes [0, 6])
  ↓
MLPClassifier sklearn (50 neuronas, Adam, 200 iter max)
  ↓
Evaluación: MAE sobre test (498 tweets, 3 clases: 0/2/4)
```

### Resultados comparativos

| Métrica | SUMA | PROMEDIO |
|---|---|---|
| Accuracy train | 0.985 | 0.938 |
| MAE train | 0.0374 | 0.1356 |
| **MAE test** | 0.3147 | **0.2884** |
| Gap train→test | 0.277 | **0.153** |
| Loss final | 0.053 | 0.187 |

**PROMEDIO supera a SUMA en test por 8.4%** y tiene **gap train-test 45% menor**. La explicación: SUMA usa la magnitud del vector (proporcional a longitud del tweet) como atajo para memorizar el training, pero ese atajo no transfiere al test.

### Diagnósticos del preprocesamiento (Actividad 7)

| Hallazgo | Implicancia |
|---|---|
| 338 tweets con vector cero (1.7%) | Vacíos post-limpieza o todas palabras OOV |
| Tasa OOV ~22-25% | Mismatch dominio: prensa formal vs Twitter informal |
| `not`/`no` están en NLTK stopwords | "Not Fun & Furious" → "fun furious" (inversión semántica) |
| Encoding ISO-8859-1 deja `ï¿½` residuales | Caracteres Unicode mal decodificados |

Tweet de ejemplo del fallo:

```
Original: "@TheLeagueSF Not Fun & Furious? The new mantra..."
Limpio:   "fun furious new mantra bay breakers getting rambunctious..."
                ↑
                negación perdida → inversión de sentido
```

## Bloques pedagógicos del lab

{{< cards >}}
  {{< card link="analogias" title="Bloque 1 - Analogías con 3CosMul" subtitle="7 analogías guía + 3 propias, polisemia y modos de falla" icon="academic-cap" >}}
  {{< card link="doesnt-match" title="Bloque 2 - doesnt_match" subtitle="5 grupos guía + 3 propios, robustez a polisemia" icon="academic-cap" >}}
  {{< card link="visualizacion-pca" title="Bloque 3 - Visualización PCA 2D" subtitle="3 plots propios + análisis 2D vs 300D" icon="academic-cap" >}}
  {{< card link="sentiment-analysis" title="Bloque 4 - Sentiment Analysis" subtitle="Sentiment140 + MLP, suma vs promedio" icon="academic-cap" >}}
  {{< card link="actividades-teoricas" title="Actividades teóricas (4, 5, 6, 7)" subtitle="Respuestas con respaldo cuantitativo verificado" icon="academic-cap" >}}
{{< /cards >}}

## Papers verificados textualmente

{{< cards >}}
  {{< card link="/papers/word2vec-efficient-mikolov-2013" title="Mikolov 2013 ICLR" subtitle="Skip-gram y CBOW originales" icon="document-text" >}}
  {{< card link="/papers/word2vec-distributed-mikolov-2013" title="Mikolov 2013 NeurIPS" subtitle="Negative Sampling + phrases" icon="document-text" >}}
  {{< card link="/papers/linguistic-regularities-levy-goldberg-2014" title="Levy-Goldberg 2014 CoNLL" subtitle="3CosMul (fórmula 4) - el motor del lab" icon="document-text" >}}
  {{< card link="/papers/sgns-implicit-mf-levy-goldberg-2014" title="Levy-Goldberg 2014 NeurIPS" subtitle="SGNS = factorización implícita de PMI" icon="document-text" >}}
  {{< card link="/papers/contrastive-analogies-ri-lee-verma-2023" title="Ri-Lee-Verma 2023" subtitle="Teorema 1: líneas paralelas con factor ζ" icon="document-text" >}}
{{< /cards >}}

## Cross-links

{{< cards >}}
  {{< card link="/clases/clase-18" title="Clase 18 - Teoría" subtitle="Word2Vec, GloVe, Skip-Thought" icon="academic-cap" >}}
  {{< card link="/clases/clase-18/profundizacion" title="Profundización matemática" subtitle="Derivaciones de Skip-gram, PMI, etc." icon="academic-cap" >}}
  {{< card link="/laboratorios/lab-17" title="Lab 17 - Pose Recognition" subtitle="Lab anterior" icon="academic-cap" >}}
  {{< card link="/laboratorios/lab-19" title="Lab 19 - Deployment y MLOps con BentoML" subtitle="Lab siguiente: serving + benchmark de latencia/concurrencia + compresión JPEG" icon="academic-cap" >}}
{{< /cards >}}

---

> **Estado:** Lab completo. Cubre las 105 celdas del notebook original con 5 páginas temáticas, evidencia cuantitativa verificada en outputs reales (cosenos 2D vs 300D, MAE comparativo, distribuciones), análisis crítico de modos de falla (polisemia ortográfica/léxica/funcional, negaciones perdidas, sesgo temporal). Reproducible en Colab versión 2025.10 con CPU en ~5 minutos (sin GPU).
