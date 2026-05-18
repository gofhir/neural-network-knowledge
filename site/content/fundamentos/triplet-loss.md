---
title: "Triplet Loss y Metric Learning"
weight: 131
math: true
---

El **triplet loss** es una función de pérdida que entrena un modelo a producir **embeddings** tales que ejemplos de la misma clase queden **cercanos** y de clases distintas queden **lejanos**, en una métrica aprendida. Formalizado por Schultz & Joachims (2004) en el contexto de SVMs metric learning, popularizado masivamente por [FaceNet](/papers/facenet-schroff-2015) (Schroff et al., 2015) en face recognition. Es el ancestro conceptual de toda la familia de **contrastive learning** moderna (SimCLR, MoCo, BYOL, SimSiam) y de los **angular margin losses** que hoy dominan face recognition (SphereFace, ArcFace, CosFace).

---

## 1. La pregunta que responde

Hay dos paradigmas clásicos de aprendizaje supervisado en deep learning:

1. **Classification**: entrenar a predecir una clase $y \in \{1, ..., C\}$ via softmax cross-entropy. Funciona cuando $C$ es fijo y todas las clases aparecen en training.

2. **Metric learning**: entrenar a producir embeddings $f(x) \in \mathbb{R}^d$ tales que la distancia $\|f(x_1) - f(x_2)\|$ refleje la similaridad semántica. Funciona cuando $C$ es **abierto** (faces, products, persons en re-id) o cuando se quiere agrupar / hacer retrieval sin reentrenar.

El triplet loss es la herramienta canónica para el segundo paradigma.

{{< concept-alert type="clave" >}}
**Classification** entrena el modelo a separar clases conocidas. **Metric learning** entrena el modelo a representar la **relación de similaridad** — generaliza a clases nunca vistas.
{{< /concept-alert >}}

## 2. Formulación

Dado:
- $x^a$ — **anchor**: un ejemplo de la clase $c$.
- $x^p$ — **positive**: otro ejemplo de la **misma** clase $c$.
- $x^n$ — **negative**: un ejemplo de una clase **distinta** $c' \neq c$.

Y un embedding $f : \mathbb{R}^D \to \mathbb{R}^d$ (típicamente con $\|f(x)\|_2 = 1$ — proyectado a la esfera unitaria), el **triplet loss** con margen $\alpha$:

$$
\mathcal{L}(x^a, x^p, x^n) = \max\!\left(0,\ \|f(x^a) - f(x^p)\|_2^2 - \|f(x^a) - f(x^n)\|_2^2 + \alpha\right)
$$

**Interpretación**:
- Si $\|f(x^a) - f(x^n)\|^2 > \|f(x^a) - f(x^p)\|^2 + \alpha$ — el negative está suficientemente lejos. Loss = 0.
- En caso contrario, el modelo paga proporcional a *cuánto viola el margen*.

El operador hinge $\max(0, \cdot)$ es **clave**: el modelo no malgasta capacidad en triplets fáciles (donde el margen ya se cumple).

### Variante: triplet ranking loss

La forma usada en el slide 55 de la Clase 17 (Tomás Vergara):

$$
L(f(I_1), f(I_2), f(I_3)) := \max\!\left\{ 0,\ m - \|f(I_1) - f(I_3)\| + \|f(I_1) - f(I_2)\| \right\}
$$

donde $I_1$ es anchor, $I_2$ positive (cercana), $I_3$ negative (lejana). Es **conceptualmente idéntico** al de arriba — solo cambia la convención de signo (margen $m$ en vez de $\alpha$ y orden positive/negative invertido).

## 3. Geometría: por qué funciona

Si los embeddings viven en $\mathbb{R}^d$ sin restricción, el triplet loss puede explotar — el modelo puede *escalar* los embeddings arbitrariamente grandes y todos los gradientes terminan cero. Por eso casi siempre se **normaliza L2** a la esfera unitaria:

$$
f(x) = \frac{g(x)}{\|g(x)\|_2}, \qquad \|f(x)\|_2 = 1
$$

donde $g$ es la red sin normalizar. En la esfera unitaria, $\|f(x_1) - f(x_2)\|_2^2 = 2 - 2 f(x_1)^\top f(x_2)$ — directamente proporcional a $1 - \cos$ similarity.

Visualmente: el modelo aprende a **agrupar embeddings de la misma identidad en regiones cercanas de la esfera** y a separar identidades distintas en regiones lejanas.

## 4. Triplet selection — el problema dominante

Para $N$ identidades con $M$ ejemplos cada una, el número total de triplets es $\binom{N \cdot M}{1} \cdot \binom{M-1}{1} \cdot \binom{(N-1) M}{1}$ — astronómico. **No se puede entrenar con todos**.

Y peor: la mayoría son **triviales** (loss ya cero) o **dañinos** (hard negatives extremos llevan a colapso). Hay que **samplear inteligentemente**.

### 4.1 Estrategias de mining

| Estrategia | Descripción | Riesgo |
|---|---|---|
| **Random** | Samplear triplets uniformemente. | La mayoría son fáciles → gradiente cero, convergencia lentísima. |
| **Hardest** | Negative más cercano al anchor; positive más lejano. | Inestable, colapso del modelo si hay noisy labels. |
| **Semi-hard** | Negative que viola margen pero está más lejos del anchor que el positive. | **El mejor**, propuesto por FaceNet. |
| **Distance-weighted** | Samplear por una distribución sobre distancias. | Buen compromiso (Wu et al. 2017). |

### 4.2 Semi-hard mining (FaceNet)

Para un anchor $x^a$ con positive $x^p$ y un candidato $x^n$, **semi-hard** si:

$$
\|f(x^a) - f(x^p)\|^2 < \|f(x^a) - f(x^n)\|^2 < \|f(x^a) - f(x^p)\|^2 + \alpha
$$

Es decir: el negative está **más lejos** que el positive (por eso no causa gradiente extremo), pero **dentro del margen** (sí contribuye a la pérdida). El "Goldilocks zone" del mining.

### 4.3 Batch-hard (Hermans et al. 2017, *In Defense of the Triplet Loss*)

Estrategia más simple: dentro de cada mini-batch con **P identidades × K muestras** (PK sampler), para cada anchor:
- Positive = el **más lejano** dentro del batch (hardest positive).
- Negative = el **más cercano** dentro del batch (hardest negative).

Computacionalmente simple (no requiere búsqueda costosa), estable empíricamente y SOTA en Person Re-ID en 2017.

## 5. Sucesores y alternativas modernas

El triplet loss es elegante pero tiene problemas:

1. **Costo de mining**: requiere samplers especiales (PK sampler), no es training estándar.
2. **Sensibilidad al margen**: $\alpha$ depende del dataset.
3. **Convergencia lenta**: depende fuertemente de la calidad del mining.

### 5.1 Contrastive losses (par a par)

**Contrastive loss** (Hadsell, Chopra, LeCun 2006) usa pares en vez de triplets:

$$
\mathcal{L} = y \cdot d^2 + (1 - y) \cdot \max(0, m - d)^2
$$

con $y = 1$ si pareja positiva, $0$ si negativa, $d = \|f(x_1) - f(x_2)\|$.

### 5.2 InfoNCE — base de SimCLR, MoCo, CLIP

$$
\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(z, z^+) / \tau)}{\sum_{j=0}^N \exp(\text{sim}(z, z_j) / \tau)}
$$

donde el denominador suma sobre 1 positivo + $N$ negativos del batch. **Generaliza triplet a $N$-tuplets simultáneos** — más estable y eficiente. Es la base de los métodos modernos de [aprendizaje contrastivo](/fundamentos/aprendizaje-contrastivo).

### 5.3 Angular margin losses (la era moderna de face recognition)

Reemplazan distancia Euclidiana por **margen angular** en el espacio de la esfera:

| Loss | Margen | Año | Acc LFW |
|---|---|---|---|
| **FaceNet** (triplet) | Aditivo Euclidiano $\alpha$ | 2015 | 99.63 |
| **SphereFace** (A-Softmax) | Multiplicativo angular $m$ | 2017 | 99.42 |
| **CosFace** | Aditivo coseno $\cos\theta - m$ | 2018 | 99.73 |
| **ArcFace** | Aditivo angular $\cos(\theta + m)$ | 2019 | **99.83** |

ArcFace es hoy **el estándar de facto** en face recognition. Pero el triplet sigue siendo conceptualmente importante y aparece en re-id, image retrieval, recommender systems.

## 6. Aplicaciones canónicas

### 6.1 Face recognition

[FaceNet](/papers/facenet-schroff-2015), DeepFace, OpenFace, InsightFace. Embedding 128-D + threshold sobre distancia para verification, k-NN para identification, clustering para agrupar fotos de una misma persona.

### 6.2 Person Re-Identification

Dada una persona en cámara A, encontrarla en cámara B. Usa triplet/quadruplet sobre crops de cuerpo. Datasets: Market-1501, DukeMTMC-reID, MSMT17.

### 6.3 Image retrieval

Google Image Search, Pinterest Lens, Amazon visual search. Embedding por imagen + búsqueda nearest-neighbor en índices ANN (FAISS, ScaNN).

### 6.4 Speaker verification

x-vectors (Snyder 2018), ECAPA-TDNN — embeddings de voz para "¿es la misma persona hablando?".

### 6.5 Sentence-BERT (NLP)

Reimers y Gurevych 2019 — triplet/contrastive sobre pares de oraciones (anchor, paráfrasis, oración no relacionada) para semantic textual similarity. Base de RAG modernos.

### 6.6 Self-supervised representation learning

SimCLR (Chen 2020), MoCo (He 2020), BYOL (Grill 2020). Anchor y positive son **dos augmentaciones de la misma imagen**; los negatives son otras imágenes del batch. InfoNCE-style — descendiente del triplet.

## 7. Triplet Network — la arquitectura

El "triplet network" del slide 55 de la Clase 17 es simplemente **tres copias del mismo encoder** con **tied weights** (compartidos):

```
        Input 1 → Network → Features 1 ─┐
                  Tied weights          │
        Input 2 → Network → Features 2 ─┼─→ Loss(F1, F2, F3)
                  Tied weights          │
        Input 3 → Network → Features 3 ─┘
```

En práctica no se instancian 3 copias — se hace forward del batch entero y se computan las distancias relevantes después. Pero conceptualmente es útil pensarlo como "tres torres siamesas que comparten parámetros".

## 8. Conexiones con la Clase 17

- **Slide 54**: introducción de triplet networks como solución al face recognition.
- **Slide 55**: arquitectura triplet network + ecuación del triplet ranking loss.
- **Slide 56**: FaceNet de Google como aplicación SOTA — `~60% → >95%` accuracy.

El triplet loss conecta pose recognition con **cualquier técnica que requiera medir similaridad** — face, person re-id, sentence embeddings, recommender systems. Es una pieza más del toolkit común.

## 9. Recursos relacionados

- [FaceNet (Schroff 2015)](/papers/facenet-schroff-2015) — el paper canónico.
- [Aprendizaje contrastivo](/fundamentos/aprendizaje-contrastivo) — la generalización moderna.
- [Pose estimation](/fundamentos/pose-estimation) — la otra técnica de la Clase 17.
- *In Defense of the Triplet Loss for Person Re-Identification* (Hermans 2017) — batch-hard mining.
- ArcFace, CosFace, SphereFace papers — sucesores en face recognition.
