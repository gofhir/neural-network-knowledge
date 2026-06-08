# Neural Collaborative Filtering (He, Liao, Zhang, Nie, Hu, Chua — WWW 2017)

> Análisis interno exhaustivo. arXiv:1708.05031v2 (26 Aug 2017). DOI 10.1145/3038912.3052569.
> Grounded en el PDF fuente. Idioma: es-419 neutro.

---

## 1. Contexto: el límite del producto interno en Matrix Factorization

A mediados de la década de 2010, el deep learning ya había arrasado en
reconocimiento de voz, visión por computador y procesamiento de lenguaje natural,
pero su penetración en sistemas de recomendación seguía siendo marginal. Los
autores abren el paper con esa observación precisa: las redes neuronales habían
recibido "relativamente poco escrutinio" en recomendación. Y, lo que es más
sutil, el poco trabajo que sí existía usaba DNNs solo para modelar **información
auxiliar** —descripciones textuales de ítems, features acústicos de música,
contenido visual de imágenes— mientras que el corazón del filtrado colaborativo,
**la interacción usuario–ítem**, seguía resolviéndose con matrix factorization
(MF) y un producto interno sobre los factores latentes.

El filtrado colaborativo (CF) es la técnica central de la recomendación
personalizada: modela la preferencia de un usuario sobre ítems a partir de sus
interacciones pasadas (ratings, clicks). Entre las variantes de CF, MF era el
enfoque *de facto* desde que el Netflix Prize lo popularizó. MF proyecta usuarios
e ítems a un espacio latente compartido de dimensión $K$ y modela la interacción
$\hat{y}_{ui}$ como el producto interno de los vectores latentes:

$$\hat{y}_{ui} = \mathbf{p}_u^\top \mathbf{q}_i = \sum_{k=1}^{K} p_{uk} q_{ik}.$$

El paper aísla con claridad quirúrgica la debilidad estructural de esta elección.
El producto interno **combina las dimensiones latentes de manera lineal y con el
mismo peso**, asumiendo que cada dimensión es independiente de las demás. Es,
literalmente, un modelo lineal de factores latentes. He et al. argumentan que esa
linealidad puede ser insuficiente para capturar la estructura compleja de los
datos de interacción. Como evidencia indirecta citan un hecho conocido: en
predicción de ratings sobre feedback explícito, agregar términos de sesgo de
usuario e ítem a la función de interacción mejora MF. Ese "tweak trivial" del
operador producto interno apunta a que diseñar una función de interacción
dedicada y mejor sí tiene efecto positivo.

### El contraejemplo geométrico de la Figura 1

La parte más memorable del contexto es un contraejemplo geométrico (Figura 1 del
PDF) que demuestra que MF puede fallar de forma irreparable en baja dimensión.
Premisas: como MF mapea usuarios e ítems al mismo espacio, la similitud entre dos
usuarios se mide con el producto interno (equivalente al coseno del ángulo si los
vectores son unitarios); y se usa el **coeficiente de Jaccard** sobre las filas de
la matriz de interacción como similitud ground-truth a recuperar.

Con las tres primeras filas, las similitudes ordenan $s_{23}(0.66) > s_{12}(0.5)
> s_{13}(0.4)$, lo que fija las posiciones geométricas de $\mathbf{p}_1,
\mathbf{p}_2, \mathbf{p}_3$ en el plano latente. Llega un cuarto usuario $u_4$ con
$s_{41}(0.6) > s_{43}(0.4) > s_{42}(0.2)$: es más parecido a $u_1$, luego a $u_3$,
y por último a $u_2$. Pero si el modelo coloca $\mathbf{p}_4$ lo más cerca posible
de $\mathbf{p}_1$ (su vecino más fuerte), la geometría del producto interno lo
obliga a quedar **más cerca de $\mathbf{p}_2$ que de $\mathbf{p}_3$**, lo que
contradice el orden correcto y produce una pérdida grande de ranking. El espacio
latente de baja dimensión simplemente no tiene grados de libertad para satisfacer
todas las restricciones de similitud simultáneamente bajo un producto interno fijo.

Los autores anticipan la objeción obvia: ¿no basta con subir $K$? Responden que sí,
pero a costa de la generalización —en datos esparsos, un $K$ grande sobreajusta.
Su tesis es que la solución correcta no es más dimensiones, sino **aprender la
función de interacción desde los datos** con DNNs, apoyándose en el teorema de
aproximación universal (Hornik et al., 1989): una red feedforward puede aproximar
cualquier función continua.

---

## 2. Contribución: el framework NCF

El paper formaliza un enfoque de red neuronal para CF y enuncia tres
contribuciones explícitas:

1. Una **arquitectura de red neuronal** que modela los factores latentes de
   usuarios e ítems, y un framework general **NCF** (Neural network-based
   Collaborative Filtering) para CF basado en redes neuronales.
2. La demostración de que **MF es una especialización de NCF**, y el uso de un
   perceptrón multicapa (MLP) para dotar a NCF de un alto nivel de no-linealidad.
3. **Experimentos extensos** en dos datasets reales que demuestran la efectividad
   de los enfoques NCF y la promesa del deep learning para CF.

El foco es feedback **implícito** (ver videos, comprar, hacer click), no explícito
(ratings, reseñas). El feedback implícito es más fácil de recolectar
automáticamente pero más difícil de explotar: la satisfacción del usuario no se
observa directamente y hay escasez natural de feedback negativo. La idea central
del paper es cómo usar DNNs para modelar las señales ruidosas del feedback
implícito.

La propuesta de fondo: **reemplazar el producto interno por una arquitectura
neuronal que aprende una función arbitraria desde los datos**. NCF es genérico
—puede expresar y generalizar MF dentro de su marco— y, al sumar un MLP, le
inyecta no-linealidades.

---

## 3. Método

### 3.1 Aprendizaje desde datos implícitos

Con $M$ usuarios y $N$ ítems, se define la matriz de interacción $\mathbf{Y} \in
\mathbb{R}^{M \times N}$:

$$y_{ui} = \begin{cases} 1, & \text{si la interacción } (u, i) \text{ se observa};\\ 0, & \text{en caso contrario}. \end{cases}$$

Crucialmente, $y_{ui}=1$ **no** significa que a $u$ le guste $i$, y $y_{ui}=0$ no
significa que le disguste —puede ser que ni siquiera conozca el ítem. Las entradas
no observadas son una mezcla de negativos reales y datos faltantes. La
recomendación se formula como estimar los scores de las entradas no observadas
para rankear los ítems. Formalmente se aprende $\hat{y}_{ui} = f(u, i \mid
\Theta)$, donde $f$ es la **función de interacción**.

La literatura usaba dos tipos de objetivos: **pointwise loss** (regresión, minimiza
el error cuadrático entre $\hat{y}_{ui}$ e $y_{ui}$) y **pairwise loss** (maximiza
el margen entre una entrada observada y una no observada, p. ej. BPR). NCF
parametriza $f$ con redes neuronales y soporta naturalmente ambos; este paper se
queda en pointwise y deja pairwise como trabajo futuro.

### 3.2 El framework general

La arquitectura (Figura 2) es una pila de capas:

- **Input layer (esparsa):** vectores de features $\mathbf{v}_u^U$ y
  $\mathbf{v}_i^I$. En CF puro se usa solo la identidad de usuario e ítem como
  **one-hot encoding**. Los autores notan que con features de contenido en lugar
  de one-hot, el mismo esquema aborda el problema de cold-start.
- **Embedding layer:** capa fully-connected que proyecta el vector esparso a uno
  denso. Ese embedding es el vector latente del usuario (o ítem) en el sentido del
  modelo de factores latentes.
- **Neural CF layers:** arquitectura multicapa que mapea los vectores latentes a
  scores. Cada capa puede descubrir cierta estructura latente; la dimensión de la
  última capa oculta $X$ determina la capacidad del modelo.
- **Output layer:** el score predicho $\hat{y}_{ui}$.

El modelo predictivo es:

$$\hat{y}_{ui} = f(\mathbf{P}^\top \mathbf{v}_u^U, \mathbf{Q}^\top \mathbf{v}_i^I \mid \mathbf{P}, \mathbf{Q}, \Theta_f),$$

con $\mathbf{P} \in \mathbb{R}^{M \times K}$ y $\mathbf{Q} \in \mathbb{R}^{N \times
K}$ las matrices de factores latentes, y $f$ una red multicapa:

$$f = \phi_{out}(\phi_X(\dots \phi_2(\phi_1(\mathbf{P}^\top \mathbf{v}_u^U, \mathbf{Q}^\top \mathbf{v}_i^I))\dots)).$$

### 3.3 Aprendizaje de NCF: la log loss con negative sampling

Aquí está una de las aportaciones conceptuales más finas. La regresión con error
cuadrático asume que las observaciones provienen de una Gaussiana, lo que no encaja
con datos implícitos donde $y_{ui}$ es binario. Los autores adoptan un tratamiento
**probabilístico**: ven $y_{ui}$ como una etiqueta (1 = relevante, 0 = no), y
restringen $\hat{y}_{ui} \in [0,1]$ usando una función probabilística (logística)
como activación de salida $\phi_{out}$. La verosimilitud es:

$$p(\mathcal{Y}, \mathcal{Y}^- \mid \mathbf{P}, \mathbf{Q}, \Theta_f) = \prod_{(u,i)\in\mathcal{Y}} \hat{y}_{ui} \prod_{(u,j)\in\mathcal{Y}^-} (1 - \hat{y}_{uj}).$$

Tomando el negativo del logaritmo se llega exactamente a la **binary cross-entropy**
(log loss):

$$L = -\sum_{(u,i)\in\mathcal{Y}\cup\mathcal{Y}^-} y_{ui}\log\hat{y}_{ui} + (1-y_{ui})\log(1-\hat{y}_{ui}).$$

Así, la recomendación con feedback implícito se reformula como un **problema de
clasificación binaria**. Los autores subrayan que la log loss "consciente de la
clasificación" había sido rara vez investigada en la literatura de recomendación, y
muestran empíricamente su efectividad. Los negativos $\mathcal{Y}^-$ se **muestrean
uniformemente** de las interacciones no observadas en cada iteración, controlando la
razón de muestreo respecto a los positivos. Optimización con SGD (mini-batch Adam
al entrenar desde cero).

### 3.4 Generalized Matrix Factorization (GMF)

Para demostrar que MF es un caso especial de NCF, definen la primera capa CF como
el **producto elemento a elemento** de los embeddings:

$$\phi_1(\mathbf{p}_u, \mathbf{q}_i) = \mathbf{p}_u \odot \mathbf{q}_i,$$

y la proyectan a la salida con activación $a_{out}$ y pesos $\mathbf{h}$:

$$\hat{y}_{ui} = a_{out}(\mathbf{h}^\top (\mathbf{p}_u \odot \mathbf{q}_i)).$$

Si $a_{out}$ es la identidad y $\mathbf{h}$ es un vector de unos, se recupera **MF
exacto**. Pero si se permite **aprender $\mathbf{h}$** sin la restricción de
uniformidad, surge una MF que pondera distinto cada dimensión latente; y con una
$a_{out}$ no lineal se generaliza a un setting no lineal más expresivo. La
implementación concreta —llamada **GMF**— usa sigmoide como $a_{out}$ y aprende
$\mathbf{h}$ con la log loss.

### 3.5 Multi-Layer Perceptron (MLP)

La rama MLP **concatena** los embeddings de usuario e ítem y aprende su interacción
con capas ocultas. Los autores son explícitos: la concatenación sola no modela
ninguna interacción entre features (es insuficiente para el efecto colaborativo),
por eso agregan capas ocultas encima:

$$\mathbf{z}_1 = \begin{bmatrix}\mathbf{p}_u \\ \mathbf{q}_i\end{bmatrix}, \quad \phi_l(\mathbf{z}_{l-1}) = a_l(\mathbf{W}_l^\top \mathbf{z}_{l-1} + \mathbf{b}_l), \quad \hat{y}_{ui} = \sigma(\mathbf{h}^\top \phi_L(\mathbf{z}_{L-1})).$$

Sobre activaciones, analizan tres: la sigmoide satura (neuronas dejan de aprender
cerca de 0/1); tanh solo alivia parcialmente (es una sigmoide reescalada,
$\tanh(x/2) = 2\sigma(x)-1$); y eligen **ReLU**, no saturada, que fomenta
activaciones esparsas, apta para datos esparsos y menos propensa a overfitting.
Empíricamente ReLU > tanh ≫ sigmoide. La estructura sigue un **patrón de torre**:
cada capa superior tiene la mitad de neuronas que la anterior, para aprender
features más abstractos.

### 3.6 NeuMF: fusión de GMF y MLP

La pregunta clave: ¿cómo fusionar GMF (kernel lineal) y MLP (kernel no lineal) para
que se refuercen mutuamente? Una solución directa —compartir el mismo embedding—
limitaría el modelo, porque obligaría a GMF y MLP a usar el mismo tamaño de
embedding. La solución elegida da más flexibilidad: **GMF y MLP aprenden embeddings
separados** y se combinan **concatenando sus últimas capas ocultas**:

$$\phi^{GMF} = \mathbf{p}_u^G \odot \mathbf{q}_i^G, \qquad \phi^{MLP} = a_L(\mathbf{W}_L^\top(\dots) + \mathbf{b}_L),$$
$$\hat{y}_{ui} = \sigma\left(\mathbf{h}^\top \begin{bmatrix}\phi^{GMF} \\ \phi^{MLP}\end{bmatrix}\right).$$

Este modelo, **NeuMF** (Neural Matrix Factorization), combina la linealidad de MF y
la no-linealidad de las DNNs. La idea de combinar MF con MLP está parcialmente
inspirada en la Neural Tensor Network (NTN) de Socher et al., pero NeuMF es más
flexible al permitir conjuntos de embeddings distintos.

### 3.7 Pre-entrenamiento

Como el objetivo de NeuMF es no convexo, la optimización por gradiente solo halla
óptimos locales y la inicialización importa. Los autores **pre-entrenan GMF y MLP
por separado** (con Adam, inicialización Gaussiana $\mu=0, \sigma=0.01$) hasta
converger, y usan esos parámetros para inicializar las partes correspondientes de
NeuMF. En la capa de salida concatenan los pesos ponderados por $\alpha$:

$$\mathbf{h} \leftarrow \begin{bmatrix}\alpha\,\mathbf{h}^{GMF} \\ (1-\alpha)\,\mathbf{h}^{MLP}\end{bmatrix}.$$

Tras cargar los pre-entrenados, NeuMF se optimiza con **SGD vanilla** (no Adam),
porque Adam necesita la información de momentum que no se guarda al transferir solo
parámetros.

---

## 4. Experimentos

### 4.1 Datasets

| Dataset | Interacciones | Ítems | Usuarios | Sparsity |
|---|---|---|---|---|
| MovieLens (1M) | 1.000.209 | 3.706 | 6.040 | 95,53% |
| Pinterest | 1.500.809 | 9.916 | 55.187 | 99,73% |

**MovieLens 1M** es feedback explícito (ratings) que los autores **binarizan**
intencionalmente a implícito (1 si el usuario calificó el ítem). Cada usuario tiene
al menos 20 ratings. **Pinterest** —y este es el detalle que conecta con el case
study multimodal de la clase— es feedback implícito construido por Geng et al.
(ICCV 2015) para recomendación de imágenes basada en contenido; cada interacción es
"el usuario pinneó la imagen a su tablero". El dataset original era enorme pero muy
esparso (más del 20% de usuarios con un solo pin), así que lo filtraron igual que
MovieLens (mínimo 20 interacciones), quedando 55.187 usuarios y 1.500.809
interacciones.

### 4.2 Protocolo de evaluación

**Leave-one-out**: para cada usuario se retiene su última interacción como test.
Como rankear todos los ítems es caro, siguen la estrategia común de muestrear **100
ítems no interactuados** y rankear el ítem de test entre esos 100. Métricas:
**HR@10** (Hit Ratio: si el ítem de test está en el top-10) y **NDCG@10**
(Normalized Discounted Cumulative Gain: premia hits en posiciones altas). Lista
truncada en 10, promedio sobre usuarios.

**Baselines**: ItemPop (popularidad, no personalizado), ItemKNN (CF item-based),
**BPR** (MF con pérdida de ranking pairwise, fuerte para item recommendation) y
**eALS** (MF estado del arte que optimiza error cuadrático tratando todos los no
observados como negativos ponderados por popularidad). Implementación en **Keras**,
**4 negativos por positivo**, factores predictivos en $\{8,16,32,64\}$, tres capas
ocultas en MLP por defecto, $\alpha=0.5$ para NeuMF pre-entrenado.

### 4.3 Resultados (RQ1, RQ2, RQ3)

**RQ1 — ¿NCF supera el estado del arte?** Sí. NeuMF logra el mejor desempeño en
ambos datasets, superando a eALS y BPR por margen significativo: mejora relativa
promedio de **4,5% sobre eALS y 4,9% sobre BPR**. En Pinterest, NeuMF con apenas 8
factores supera a eALS/BPR con 64 factores. GMF y MLP también son fuertes; MLP queda
ligeramente por debajo de GMF con 3 capas (pero mejora con más capas). GMF mejora
consistentemente sobre BPR, lo que valida la log loss consciente de clasificación
(GMF y BPR aprenden el mismo modelo MF pero con objetivos distintos). Los t-tests
pareados confirman significancia $p<0.01$.

La **utilidad del pre-entrenamiento** (Tabla 2) — mejora relativa de 2,2%
(MovieLens) y 1,1% (Pinterest):

| | Con pre-entrenamiento | | Sin pre-entrenamiento | |
|---|---|---|---|---|
| Factors | HR@10 | NDCG@10 | HR@10 | NDCG@10 |
| **MovieLens** | | | | |
| 8 | 0,684 | 0,403 | 0,688 | 0,410 |
| 16 | 0,707 | 0,426 | 0,696 | 0,420 |
| 32 | 0,726 | 0,445 | 0,701 | 0,425 |
| 64 | 0,730 | 0,447 | 0,705 | 0,426 |
| **Pinterest** | | | | |
| 8 | 0,878 | 0,555 | 0,869 | 0,546 |
| 16 | 0,880 | 0,558 | 0,871 | 0,547 |
| 32 | 0,879 | 0,555 | 0,870 | 0,549 |
| 64 | 0,877 | 0,552 | 0,872 | 0,551 |

(Único caso donde pre-entrenar empeora: MovieLens, 8 factores.)

**RQ2 — ¿Funciona la log loss con negative sampling?** Sí. La pérdida de
entrenamiento baja y el desempeño sube con las iteraciones; las actualizaciones más
efectivas ocurren en las primeras 10 iteraciones (más allá, NeuMF puede
sobreajustar). El orden de pérdida y desempeño es consistente: **NeuMF > MLP > GMF**.
Sobre la razón de negativos: un solo negativo por positivo es insuficiente; el
**óptimo está entre 3 y 6 negativos** por positivo. En Pinterest, pasar de 7
negativos empieza a degradar. La ventaja del pointwise log loss sobre el pairwise
BPR es justamente esta flexibilidad en la razón de muestreo (BPR solo puede emparejar
un negativo).

**RQ3 — ¿Ayuda la profundidad?** Sí, de forma contundente. Apilar más capas mejora
el desempeño aun con la misma capacidad (Tablas 3 y 4). HR@10 de MLP por capas en
MovieLens, 64 factores: MLP-0 = 0,453, MLP-1 = 0,687, MLP-2 = 0,696, MLP-3 = 0,702,
MLP-4 = 0,707. **MLP-0** (sin capas ocultas, embedding proyectado directo) es tan
débil que **ni siquiera supera a ItemPop**, lo que confirma que concatenar embeddings
sin capas ocultas no modela la interacción. Apilar capas **lineales** (activación
identidad) da resultados muy peores que ReLU: la ganancia viene de la no-linealidad.

NDCG@10 de MLP por capas (extracto, MovieLens 64 factores): MLP-1 = 0,409 →
MLP-4 = 0,432; Pinterest 64 factores: MLP-1 = 0,538 → MLP-4 = 0,550.

---

## 5. Limitaciones reconocibles (y la crítica posterior de Rendle 2020)

**Limitaciones que el propio paper admite:**
- Se queda en aprendizaje **pointwise**; deja pairwise (BPR, margin-based) como
  trabajo futuro.
- Usa **muestreo uniforme** de negativos; reconoce que un muestreo sesgado por
  popularidad podría mejorar, pero no lo explora.
- No incorpora información auxiliar ni contenido (es CF puro), aunque la arquitectura
  lo permitiría vía features en la input layer.
- NeuMF es no convexo y depende de la inicialización (de ahí el pre-entrenamiento).

**La crítica externa más importante — Rendle et al. (RecSys 2020), "Neural
Collaborative Filtering vs. Matrix Factorization Revisited":** años después, un
equipo (varios de Google, incluyendo a Steffen Rendle, autor de BPR y de las
factorization machines) cuestionó la tesis central. Reentrenaron los baselines de MF
con buen tuning de hiperparámetros e inicialización y mostraron que un **dot product
bien ajustado iguala o supera al MLP aprendido de NCF** en los mismos datasets, y que
aprender la función de similitud con un MLP es difícil y costoso de entrenar para
producir algo que el producto interno ya hace de forma barata. El argumento de
Rendle es que el MLP no aprende fácilmente ni siquiera el producto interno, y que el
costo de inferencia (sin la estructura de producto interno no se puede usar maximum
inner product search para retrieval rápido) raras veces compensa. Esta réplica no
invalida el valor histórico de NCF —abrió la puerta al deep CF— pero matiza
fuertemente la afirmación de que reemplazar el dot product por un MLP es
intrínsecamente mejor. Es una lección metodológica clave: la fuerza de los baselines
y el tuning determinan las conclusiones.

---

## 6. Impacto

NCF es uno de los papers más citados de recomendación con redes neuronales. Su
impacto concreto:
- **Estableció el patrón de dos torres / two-pathway** (una rama lineal tipo MF + una
  rama profunda tipo MLP que comparten o no embeddings), que reaparece en innumerables
  arquitecturas industriales de recomendación y retrieval.
- **Popularizó tratar la recomendación implícita como clasificación binaria con BCE +
  negative sampling**, hoy estándar.
- Mostró empíricamente que **la profundidad ayuda en CF**, abriendo la "nueva avenida"
  del deep learning para recomendación que los propios autores anuncian en la
  conclusión.
- Su sección de trabajo futuro anticipa explícitamente los **sistemas de recomendación
  multimedia y multimodales** ("imágenes y videos contienen semántica visual rica…
  necesitamos aprender de datos multi-view y multi-modales"), que son justamente el
  tema del case study de la Clase 25.
- El código en Keras (github.com/hexiangnan/neural_collaborative_filtering) se volvió
  una referencia didáctica casi universal.

---

## 7. Conexión con la Clase 25 (recsys multimodal con redes neuronales)

La Clase 25 es un **case study de recomendación multimodal con redes neuronales**, y
NCF marca exactamente la **bisagra histórica** que el case study necesita: la
transición de matrix factorization (producto interno, lineal, shallow) al deep
learning en recomendación. Tres hilos conectan directamente:

1. **Arquitectura neuronal de embeddings.** NCF instala la idea —hoy obvia— de que
   usuarios e ítems se representan como embeddings densos aprendidos, y que la
   interacción se aprende con capas neuronales en vez de un producto interno fijo. Esa
   es la base sobre la que se construye cualquier recsys multimodal: los embeddings de
   ítem pueden venir ahora de imágenes (CNN), texto (transformers) o audio, y NCF ya
   muestra cómo fusionarlos vía concatenación + capas profundas. El propio dataset
   **Pinterest** del paper proviene de un trabajo de recomendación de imágenes, un
   guiño directo a lo multimodal.

2. **Métricas de ranking.** HR@10 y NDCG@10 con leave-one-out y muestreo de 100
   negativos son el vocabulario de evaluación que el case study reutiliza. Entender por
   qué NDCG premia las posiciones altas y por qué HR mide presencia en el top-K es
   prerequisito para discutir cualquier recsys.

3. **Feedback implícito + BCE + negative sampling.** El encuadre de la recomendación
   como clasificación binaria con muestreo de negativos es el esqueleto de
   entrenamiento que reaparece (con InfoNCE, sampled softmax, etc.) en los recsys
   neuronales modernos del case study.

El contrapunto pedagógico ideal: presentar NCF como el paper que abrió el deep CF, y
luego la crítica de Rendle 2020 como recordatorio de que un baseline bien tuneado
puede desinflar afirmaciones grandilocuentes —una lección transversal de todo el
curso.

---

## 8. Datos y referencias clave del paper

- **Autores:** Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, Tat-Seng Chua.
- **Venue:** WWW 2017 (Perth, Australia, 3–7 abril 2017). ACM 978-1-4503-4913-0/17/04.
- **arXiv:** 1708.05031v2 (26 Aug 2017). DOI: 10.1145/3038912.3052569. Licencia CC BY 4.0.
- **Modelos:** GMF, MLP, NeuMF (= GMF + MLP con embeddings separados, fusión por concatenación).
- **Resultados ancla:** NeuMF supera a eALS/BPR en +4,5%/+4,9% relativo promedio; HR@10 hasta 0,730 (MovieLens) y 0,880 (Pinterest); razón óptima de negativos 3–6.
- **Referencias internas relevantes:** Koren KDD 2008 [21] (MF + vecindario), BPR (Rendle UAI 2009) [27], eALS (He SIGIR 2016) [14], Hu-Koren-Volinsky ICDM 2008 [19] (CF implícito), Wide & Deep (Cheng 2016) [4], NTN (Socher 2013) [33], Hornik 1989 [17] (aproximación universal), Adam (Kingma-Ba) [20].
