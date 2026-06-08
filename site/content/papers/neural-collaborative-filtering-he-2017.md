---
title: "Neural Collaborative Filtering"
weight: 254
math: true
---

{{< paper-card
    title="Neural Collaborative Filtering"
    authors="He, Liao, Zhang, Nie, Hu, Chua"
    year="2017"
    venue="WWW 2017"
    pdf="/papers/neural-collaborative-filtering-he-2017.pdf"
    arxiv="1708.05031" >}}
**NCF** propone reemplazar el **producto interno** de matrix factorization por una **arquitectura neuronal** que aprende la función de interacción usuario–ítem desde los datos. Define un framework general que **generaliza MF** (GMF), le suma no-linealidad con un **MLP**, y fusiona ambos en **NeuMF**. Entrena con **binary cross-entropy y negative sampling** sobre feedback implícito, y supera a los baselines MF (eALS, BPR) en MovieLens y Pinterest. Es la bisagra histórica de la recomendación: el paso de la factorización lineal al deep learning.
{{< /paper-card >}}

---

## Contexto

Hacia 2017 el deep learning dominaba voz, visión y NLP, pero apenas había tocado la
recomendación. El poco trabajo neuronal existente usaba redes solo para
**información auxiliar** (texto de ítems, audio de música, contenido visual),
mientras el corazón del filtrado colaborativo —la interacción usuario–ítem— seguía
resolviéndose con **matrix factorization** y un producto interno sobre los factores
latentes (ver [/papers/matrix-factorization-koren-2009](/papers/matrix-factorization-koren-2009)).

MF proyecta usuarios e ítems a un espacio latente compartido y modela la interacción
como $\hat{y}_{ui} = \mathbf{p}_u^\top \mathbf{q}_i$. El paper aísla su debilidad: el
producto interno **combina las dimensiones latentes de forma lineal y con el mismo
peso**, asumiéndolas independientes. Es un modelo lineal. Con un contraejemplo
geométrico (la Figura 1 del paper) muestran que, en baja dimensión, el producto
interno puede ser **incapaz** de respetar simultáneamente todas las relaciones de
similitud entre usuarios, incurriendo en pérdida de ranking. Subir la dimensión $K$
ayuda, pero sobreajusta en datos esparsos. La alternativa que proponen: **aprender la
función de interacción desde los datos** con redes neuronales, apoyándose en el
teorema de aproximación universal. Más contexto en
[/fundamentos/recommender-systems](/fundamentos/recommender-systems).

## Ideas principales

El framework **NCF** (Neural network-based Collaborative Filtering) representa
usuarios e ítems como **embeddings densos** (proyección de un one-hot vía una capa de
embedding) y mapea el par a un score con capas neuronales. Se enfoca en **feedback
implícito**: $y_{ui}=1$ si la interacción se observa, $0$ si no —donde el 0 mezcla
negativos reales y datos faltantes.

### GMF, MLP, NeuMF y la log loss

**Generalized Matrix Factorization (GMF).** Demuestra que MF es un caso especial de
NCF. La primera capa CF es el **producto elemento a elemento** de los embeddings,
proyectado a la salida:

$$\hat{y}_{ui} = a_{out}\!\left(\mathbf{h}^\top (\mathbf{p}_u \odot \mathbf{q}_i)\right).$$

Con $a_{out}$ identidad y $\mathbf{h}$ un vector de unos se recupera **MF exacto**. La
GUI implementada aprende $\mathbf{h}$ (peso distinto por dimensión) y usa sigmoide.

**Multi-Layer Perceptron (MLP).** Concatena los embeddings y aprende su interacción
con capas ocultas (la concatenación sola no modela interacción alguna):

$$\mathbf{z}_1 = \begin{bmatrix}\mathbf{p}_u \\ \mathbf{q}_i\end{bmatrix}, \quad \phi_l(\mathbf{z}_{l-1}) = a_l(\mathbf{W}_l^\top \mathbf{z}_{l-1} + \mathbf{b}_l), \quad \hat{y}_{ui} = \sigma(\mathbf{h}^\top \phi_L(\mathbf{z}_{L-1})).$$

Eligen **ReLU** (no saturada, activaciones esparsas) sobre tanh y sigmoide, en
estructura de **torre** (cada capa superior con la mitad de neuronas).

**NeuMF (Neural Matrix Factorization).** Fusiona GMF y MLP. En vez de compartir
embeddings (lo que forzaría el mismo tamaño), deja que cada rama **aprenda embeddings
separados** y concatena sus últimas capas ocultas:

$$\hat{y}_{ui} = \sigma\!\left(\mathbf{h}^\top \begin{bmatrix}\mathbf{p}_u^G \odot \mathbf{q}_i^G \\ \phi^{MLP}(\mathbf{p}_u^M, \mathbf{q}_i^M)\end{bmatrix}\right).$$

Combina la **linealidad de MF** con la **no-linealidad del MLP**. Se inicializa con
GMF y MLP **pre-entrenados** (luego SGD vanilla, no Adam).

**Log loss con negative sampling.** Como $y_{ui}$ es binario, tratan la recomendación
como **clasificación binaria**: restringen $\hat{y}_{ui}\in[0,1]$ con sigmoide y
optimizan la **binary cross-entropy**:

$$L = -\!\!\sum_{(u,i)\in\mathcal{Y}\cup\mathcal{Y}^-}\!\! y_{ui}\log\hat{y}_{ui} + (1-y_{ui})\log(1-\hat{y}_{ui}),$$

donde los negativos $\mathcal{Y}^-$ se **muestrean uniformemente** de las
interacciones no observadas en cada iteración.

## Resultados experimentales

Dos datasets, evaluación **leave-one-out** rankeando el ítem de test entre 100
negativos muestreados; métricas **HR@10** y **NDCG@10** (ver
[/fundamentos/ranking-metrics](/fundamentos/ranking-metrics)).

| Dataset | Interacciones | Ítems | Usuarios | Sparsity |
|---|---|---|---|---|
| MovieLens 1M | 1.000.209 | 3.706 | 6.040 | 95,53% |
| Pinterest | 1.500.809 | 9.916 | 55.187 | 99,73% |

(MovieLens es feedback explícito binarizado a implícito; **Pinterest** proviene de un
trabajo de recomendación de imágenes —el guiño multimodal del paper.)

**NeuMF gana en ambos datasets**, con mejora relativa promedio de **+4,5% sobre eALS**
y **+4,9% sobre BPR** ($p<0.01$). En Pinterest, NeuMF con 8 factores supera a
eALS/BPR con 64. GMF mejora consistentemente sobre BPR, validando la log loss.

**NeuMF con y sin pre-entrenamiento (HR@10 / NDCG@10):**

| Factors | Con pre-entr. HR@10 | NDCG@10 | Sin pre-entr. HR@10 | NDCG@10 |
|---|---|---|---|---|
| **MovieLens** 8 | 0,684 | 0,403 | 0,688 | 0,410 |
| 16 | 0,707 | 0,426 | 0,696 | 0,420 |
| 32 | 0,726 | 0,445 | 0,701 | 0,425 |
| 64 | 0,730 | 0,447 | 0,705 | 0,426 |
| **Pinterest** 8 | 0,878 | 0,555 | 0,869 | 0,546 |
| 16 | 0,880 | 0,558 | 0,871 | 0,547 |
| 32 | 0,879 | 0,555 | 0,870 | 0,549 |
| 64 | 0,877 | 0,552 | 0,872 | 0,551 |

**La profundidad ayuda.** HR@10 de MLP por número de capas (MovieLens, 64 factores):
MLP-0 = 0,453, MLP-1 = 0,687, MLP-2 = 0,696, MLP-3 = 0,702, **MLP-4 = 0,707**. MLP-0
(sin capas ocultas) ni siquiera supera a ItemPop, y apilar capas **lineales** rinde
mucho peor que ReLU: la ganancia viene de la no-linealidad. La **razón óptima de
negativos** está entre **3 y 6** por positivo; un solo negativo es insuficiente.

## Limitaciones reconocibles

El propio paper admite que se queda en aprendizaje **pointwise** (deja pairwise como
trabajo futuro), usa **muestreo uniforme** de negativos (un sesgo por popularidad
podría mejorar), es CF puro sin contenido, y NeuMF es no convexo (de ahí el
pre-entrenamiento). La crítica externa más fuerte llegó después: **Rendle et al.
(RecSys 2020), "Neural Collaborative Filtering vs. Matrix Factorization Revisited"**,
mostró que un **producto interno bien tuneado iguala o supera al MLP de NCF** en estos
mismos datasets, que el MLP no aprende fácilmente ni siquiera el dot product, y que
pierde la estructura que permite retrieval rápido por inner-product search. No
invalida el valor histórico de NCF, pero recuerda que la fuerza de los baselines
determina las conclusiones.

## Por qué importa hoy

NCF instaló tres ideas que son estándar: (1) representar usuarios e ítems como
**embeddings aprendidos** y la interacción con **capas neuronales**; (2) el patrón
**two-tower / dos ramas** (lineal tipo MF + profunda tipo MLP) que reaparece en
arquitecturas industriales de recomendación y retrieval; y (3) encuadrar la
recomendación implícita como **clasificación binaria con BCE + negative sampling**. Su
conclusión anticipa explícitamente los **recsys multimedia y multimodales**, abriendo
la avenida del deep learning para recomendación.

## Conexión con la Clase 25

La [/clases/clase-25](/clases/clase-25) es un case study de **recomendación multimodal
con redes neuronales**, y NCF es su bisagra histórica: el paso de matrix factorization
(lineal, shallow) al deep learning. Conecta por tres hilos. La **arquitectura neuronal
de embeddings** es la base sobre la que un recsys multimodal sustituye los embeddings
de ítem por representaciones de imagen (CNN), texto (transformers) o audio y los
fusiona con capas profundas —y el dataset **Pinterest** del paper ya viene de
recomendación de imágenes. Las **métricas de ranking** HR@10 y NDCG@10 con leave-one-out
son el vocabulario de evaluación que reutiliza el case study. Y el encuadre de
**feedback implícito + BCE + negative sampling** es el esqueleto de entrenamiento de los
recsys neuronales modernos.

## Notas y enlaces

- **Paper:** He, Liao, Zhang, Nie, Hu, Chua. *Neural Collaborative Filtering*. WWW 2017, Perth. DOI 10.1145/3038912.3052569. Licencia CC BY 4.0.
- **arXiv:** [1708.05031](https://arxiv.org/abs/1708.05031) (v2, 26 ago 2017).
- **Código original (Keras):** github.com/hexiangnan/neural_collaborative_filtering.
- **Crítica posterior:** Rendle, Krichene, Zhang, Anderson. *Neural Collaborative Filtering vs. Matrix Factorization Revisited*. RecSys 2020.
- Relacionados en el sitio: [/papers/matrix-factorization-koren-2009](/papers/matrix-factorization-koren-2009), [/fundamentos/recommender-systems](/fundamentos/recommender-systems), [/fundamentos/ranking-metrics](/fundamentos/ranking-metrics), [/clases/clase-25](/clases/clase-25).
