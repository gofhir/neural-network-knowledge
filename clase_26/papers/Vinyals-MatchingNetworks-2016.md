# Matching Networks for One Shot Learning — Análisis interno exhaustivo

## 1. Metadata y resumen ejecutivo

**Título:** *Matching Networks for One Shot Learning*
**Autores:** Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Koray Kavukcuoglu, Daan Wierstra (Google DeepMind).
**Venue:** NeurIPS (NIPS) 2016. **Preprint:** arXiv:1606.04080 (v2, 29 dic 2017).
**Área:** meta-learning, métrica aprendida (metric learning), modelos no-paramétricos, one-shot learning.

Este paper es uno de los trabajos seminales del meta-learning moderno. Su tesis es deceptivamente simple y profundamente influyente: para que un modelo aprenda a clasificar a partir de **un solo ejemplo por clase** (one-shot), hay que entrenarlo de modo que las **condiciones de entrenamiento imiten las condiciones de test**. En lugar de entrenar un clasificador paramétrico estándar sobre un conjunto fijo de clases y luego intentar adaptarlo (con fine-tuning costoso) a clases nuevas, Matching Networks (MN) define directamente un clasificador **no-paramétrico** que, dado un pequeño *support set* etiquetado $S=\{(x_i,y_i)\}_{i=1}^k$ y un ejemplo de prueba $\hat{x}$, produce su etiqueta como una **suma ponderada por atención** de las etiquetas del support set:

$$\hat{y} = \sum_{i=1}^{k} a(\hat{x}, x_i)\, y_i.$$

Las contribuciones son dobles —y ambas resultaron históricas:

1. **A nivel de modelo:** un clasificador diferenciable end-to-end que combina (a) un *embedding* neuronal profundo (la parte "métrica", herencia de NCA y las redes Siamesas) con (b) un mecanismo de atención sobre una "memoria" externa que es el propio support set (herencia de seq2seq con atención, Memory Networks, Pointer Networks). El modelo clasifica clases **nunca vistas durante entrenamiento sin ningún cambio en la red** — no hay fine-tuning obligatorio.

2. **A nivel de procedimiento de entrenamiento:** el **muestreo episódico**. Cada minibatch es un "episodio" que reproduce una tarea $N$-way $k$-shot completa (elegir $N$ clases, dar $k$ ejemplos por clase como support, predecir un batch disjunto). Este protocolo se volvió el **estándar de facto** del campo.

**Resultados clave (números reales del paper):**

- **Omniglot 5-way:** 98.1% (1-shot), 98.9% (5-shot); **20-way:** 93.8% (1-shot), 98.5% (5-shot). Supera a Siamese Nets convolucionales (96.7% / 88.0%).
- **miniImageNet 5-way** (dataset *introducido en este paper*): 41.2% (1-shot) sin FCE, **46.6%** (1-shot) con FCE + fine-tuning; 60.0% (5-shot). El baseline classifier con cosine queda en 36.6% (1-shot).
- **Full ImageNet, rand split, 5-way 1-shot:** Matching Nets 93.2% vs Inception classifier 87.6% (mejora de ~5.6 puntos, *halving the errors*).
- **Penn Treebank one-shot language modeling** (tarea también introducida aquí): 32.4% / 36.1% / 38.2% con $k=1,2,3$, contra un oráculo LSTM-LM (que ve todas las palabras) al 72.8% y chance al 20%.

El abstract reporta además mejoras de 87.6%→93.2% en ImageNet y 88.0%→93.8% en Omniglot frente a los mejores competidores.

> Nota de lectura para Roberto: este es el paper que define el vocabulario que hoy damos por sentado — "$N$-way $k$-shot", "support set", "query/batch", "episodio". Si alguna vez evaluaste un sistema de matching de pacientes con pocos ejemplos por categoría, la disciplina de "evaluar exactamente en el régimen en que vas a desplegar" es la misma idea rectora.

---

## 2. Contexto: one-shot learning y métodos métricos previos

La motivación arranca con una observación cognitiva: un niño generaliza el concepto de "jirafa" desde una sola foto en un libro, mientras que los mejores sistemas de deep learning de 2016 necesitaban cientos o miles de ejemplos. El deep learning supervisado clásico (speech, visión con AlexNet/VGG, lenguaje con RNN-LM) es **notorio por su voracidad de datos**. Data augmentation y regularización alivian el sobreajuste en regímenes de bajos datos, pero no lo resuelven, y el aprendizaje sigue siendo lento: requiere muchas actualizaciones de pesos vía SGD.

El diagnóstico de los autores es preciso: el cuello de botella está en el **aspecto paramétrico** del modelo. Los ejemplos de entrenamiento deben "destilarse lentamente" dentro de los parámetros $\theta$ vía descenso de gradiente. Cada concepto nuevo exige reescribir pesos, con dos costos: lentitud y **olvido catastrófico** (catastrophic forgetting).

En contraste, los **modelos no-paramétricos** asimilan ejemplos nuevos de forma instantánea y no sufren olvido catastrófico. El ejemplo canónico es **$k$-nearest neighbors (kNN)**: no requiere entrenamiento, solo guardar los ejemplos, pero su desempeño depende críticamente de la **métrica** elegida (cita a Atkeson et al., *Locally weighted learning*). Aquí entra el **metric learning**: aprender una métrica/representación tal que vecinos cercanos compartan etiqueta.

Los antecedentes directos que el paper reconoce:

- **Neighbourhood Component Analysis (NCA)** (Roweis, Hinton, Salakhutdinov, 2004) y su versión no-lineal (Salakhutdinov & Hinton, 2007). NCA aprende una transformación lineal que maximiza la precisión esperada de un clasificador kNN suave (soft-kNN) usando una vecindad estocástica softmax. La pérdida de MN es "muy similar a la de NCA", con una diferencia clave: MN usa **el support set completo** en vez de comparaciones por pares.
- **Redes Siamesas convolucionales para one-shot** (Koch, Zemel, Salakhutdinov, ICML DL workshop 2015). Las Siamese Nets entrenan en una tarea de **"igual o distinto"** (same-or-different): dos imágenes entran por dos torres con pesos compartidos, y la red predice si pertenecen a la misma clase. En test, la última capa se usa como feature para nearest-neighbour matching. Es el baseline más fuerte que MN supera.
- **Large margin nearest neighbor (LMNN)** (Weinberger & Saul, 2009) y **triplet loss** (Hoffer & Ailon, 2015): familia de pérdidas que ya incluían la noción de conjunto pero con métricas menos potentes.

La idea de "aprender a aprender" (*learning to learn* / meta-learning) llega por dos vías citadas: Santoro et al. (2016, MANN — Memory-Augmented Neural Networks, donde una red con memoria externa tipo Neural Turing Machine aprende a clasificar datos presentados secuencialmente) y el trabajo de Hochreiter sobre LSTM que aprende a aprender. MN toma ese paradigma pero **trata los datos como un conjunto** (set), no como una secuencia — apoyándose en el framework "Order Matters: sequence to sequence for sets" del propio Vinyals (2015).

**El linaje de memoria y atención.** El paper sitúa explícitamente su arquitectura dentro de una ola de modelos que van "más allá de la clasificación estática de vectores fijos sobre sus clases" y que reformaron tanto la investigación como las aplicaciones industriales. Tres referencias estructuran ese linaje: (i) **seq2seq con atención por contenido** (Bahdanau, Cho, Bengio, 2014), que introdujo la atención diferenciable como mecanismo para leer una memoria; (ii) la **Neural Turing Machine** (Graves, Wayne, Danihelka, 2014) y arquitecturas "computer-like" con memoria direccionable; y (iii) las **Memory Networks** (Weston, Chopra, Bordes, 2014) y **Pointer Networks** (Vinyals, Fortunato, Jaitly, 2015), donde la atención "apunta" a posiciones de la entrada. El insight común es que todos estos modelos parametrizan $P(B\mid A)$ donde $A$ y/o $B$ pueden ser secuencias o, lo más relevante para MN, **conjuntos**. La contribución de MN es **encuadrar el one-shot learning dentro del framework set-to-set**: el support set $S$ es la "memoria" sobre la que se atiende, y la novedad es que, una vez entrenada, la red produce etiquetas sensatas para clases no observadas **sin ningún cambio en la red**.

**Zero-shot y la ausencia de literatura one-shot en ImageNet.** El paper menciona que existían trabajos de **zero-shot learning** sobre ImageNet (Norouzi et al., 2013, combinación convexa de embeddings semánticos), pero notablemente **poca literatura one-shot sobre ImageNet**. Esto motiva una de sus contribuciones de benchmark: definir tareas one-shot ejecutables sobre ImageNet (los splits rand/dogs y miniImageNet) para que otros grupos pudieran comparar. La elección de Omniglot (Lake et al., 2011) como banco de pruebas no fue casual: sus autores lo describían como el "transpuesto de MNIST" —muchas clases, pocos ejemplos por clase— exactamente la geometría que el one-shot necesita estresar.

---

## 3. El principio rector: "test and train conditions must match"

El corazón conceptual del paper cabe en una frase: **las condiciones de test y de entrenamiento deben coincidir**. Es un principio de machine learning elemental que casi nadie aplicaba al one-shot.

El razonamiento es el siguiente. En test, el modelo recibirá un support set $S'$ con clases **nunca vistas**, unos pocos ejemplos por clase, y deberá clasificar queries. Si en entrenamiento exponemos al modelo al régimen habitual (minibatches grandes de un conjunto fijo de clases con muchos ejemplos cada una), creamos un **desajuste de distribución entre train y test**: el modelo nunca practicó el acto de "mirar pocos ejemplos de clases nuevas y decidir". Por eso falla.

La solución es entrenar mostrando **solo unos pocos ejemplos por clase, cambiando la tarea de minibatch a minibatch**, exactamente como será evaluado. De ahí el muestreo episódico (Sección 7). Citando la conclusión del paper: *"one-shot learning is much easier if you train the network to do one-shot learning"*. Suena tautológico, pero antes de este trabajo el campo no lo hacía sistemáticamente.

Es importante separar este principio (procedimiento de entrenamiento) de la arquitectura (modelo no-paramétrico). Son contribuciones ortogonales que se potencian: la arquitectura no-paramétrica permite incorporar el support set sin reescribir pesos, y el entrenamiento episódico le enseña a la arquitectura a explotar bien ese support set.

---

## 4. La idea central: el clasificador no-paramétrico

Formalmente, MN aprende un mapeo $S \to c_S(\cdot)$: dado un support set $S=\{(x_i,y_i)\}_{i=1}^k$, produce un clasificador $c_S(\hat{x})$ que define una distribución de probabilidad sobre las etiquetas $\hat{y}$ de un ejemplo de prueba $\hat{x}$. Este mapeo se parametriza como una red neuronal $P_\theta(\hat{y}\mid \hat{x}, S)$. La predicción es $\arg\max_y P(y\mid\hat{x}, S)$.

La forma más simple del modelo (Ecuación 1) es:

$$\hat{y} = \sum_{i=1}^{k} a(\hat{x}, x_i)\, y_i,$$

donde $y_i$ son las etiquetas del support (codificadas one-hot) y $a(\hat{x}, x_i)$ es un **mecanismo de atención** — un peso escalar que mide cuánto se parece $\hat{x}$ a $x_i$. El resultado $\hat{y}$ es entonces una **combinación lineal de las etiquetas del support set**, ponderada por similitud. Si $a$ suma 1 (es un softmax), $\hat{y}$ es directamente la distribución de probabilidad sobre clases.

El paper expone tres lecturas de la Ecuación 1, todas iluminadoras:

1. **Estimador de densidad por kernel (KDE):** si $a$ es un kernel sobre $\mathcal{X}\times\mathcal{X}$, entonces (1) es un KDE — cada $x_i$ aporta densidad alrededor de su etiqueta.
2. **$k$-$b$ nearest neighbours:** si la atención es cero para los $b$ ejemplos más lejanos a $\hat{x}$ y constante para el resto, (1) se reduce a una variante de kNN. Por eso el paper afirma que la Ecuación 1 **subsume tanto KDE como kNN**: es una generalización suave y aprendible de ambos.
3. **Memoria asociativa:** $a$ actúa como atención y las $y_i$ son "memorias" ligadas a sus $x_i$. Dado un input, el modelo "apunta" (pointer) al ejemplo correspondiente en el support y recupera su etiqueta. A diferencia de mecanismos de memoria atencional paramétricos, (1) es **no-paramétrico**: cuando el support crece, crece la memoria. La forma funcional de $c_S$ es por tanto muy flexible y se adapta a cualquier support nuevo sin tocar pesos.

El paper subraya que, aunque emparentado con metric learning, el clasificador de (1) es **discriminativo**: para clasificar bien $\hat{x}$ basta con que esté suficientemente alineado con los pares $(x',y')\in S$ tales que $y'=y$ y desalineado con el resto. La pérdida es simple, diferenciable y optimizable end-to-end, y —argumentan— está "precisamente alineada" con la clasificación multi-clase one-shot, por lo que se espera supere a NCA, triplet loss o LMNN, que optimizan objetivos surrogate por pares o triples.

---

## 5. El attention kernel

La elección de $a(\cdot,\cdot)$ especifica completamente el clasificador. La forma propuesta (la que tiene "relaciones muy estrechas" con modelos de atención y kernels) es un **softmax sobre la distancia coseno** entre embeddings:

$$a(\hat{x}, x_i) = \frac{\exp\big(c(f(\hat{x}), g(x_i))\big)}{\sum_{j=1}^{k} \exp\big(c(f(\hat{x}), g(x_j))\big)},$$

donde:

- $f$ y $g$ son **funciones de embedding** (redes neuronales, potencialmente $f=g$) que mapean $\hat{x}$ y $x_i$ a un espacio de features. Para visión son CNN profundas (estilo VGG o Inception); para lenguaje, un word embedding simple.
- $c(\cdot,\cdot)$ es la **similitud coseno**: $c(u,v) = \frac{u^\top v}{\lVert u\rVert\,\lVert v\rVert}$.

Mecánicamente: se embeben el query con $f$ y cada elemento del support con $g$, se computa la similitud coseno de $\hat{x}$ contra cada $x_i$, y el softmax normaliza esas similitudes a pesos que suman 1. Es exactamente la estructura de **atención por contenido** de Bahdanau et al. (2014), pero aquí los "valores" sobre los que se atiende son las **etiquetas** del support, no estados ocultos de un decoder.

**Relación con kNN suave y KDE.** Esta forma es un **soft-kNN**: en vez de un voto duro de los $k$ vecinos más cercanos, cada vecino vota con peso proporcional a $\exp(c)$. La temperatura implícita del softmax controla cuán "duro" es el voto — con similitudes muy grandes el softmax se concentra en el vecino más cercano (kNN clásico), con similitudes pequeñas reparte uniformemente (promedio global, KDE de banda ancha). Frente a NCA, la diferencia es que MN normaliza sobre **todo el support set $S$** en un solo softmax, en lugar de acumular contribuciones por pares — lo que lo hace directamente entrenable para la decisión multi-way que se evaluará.

El uso de **coseno** (no distancia euclídea) es deliberado: normaliza la magnitud de los embeddings, de modo que la similitud depende solo de la dirección. Esto estabiliza el softmax y evita que features de norma grande dominen. (Más adelante, Prototypical Networks revisaría esta elección y mostraría que la distancia euclídea funciona mejor con prototipos promediados — pero esa es otra historia.)

**Sobre la temperatura implícita.** Conviene notar que la magnitud de los embeddings y el rango de $c$ actúan como una temperatura efectiva del softmax. El coseno está acotado en $[-1, 1]$, por lo que las diferencias de logits antes del softmax son pequeñas: la atención resultante es relativamente "blanda". En la práctica esto se compensa con la dimensionalidad del espacio de embedding y el entrenamiento, que aprende a separar las clases del support lo suficiente. Si el coseno se escalara por un factor $\tau$ (temperatura inversa), $a(\hat{x},x_i) \propto \exp(\tau\, c(f(\hat{x}), g(x_i)))$, valores grandes de $\tau$ aproximarían un argmax duro (kNN con $k=1$) y valores pequeños un promedio uniforme. El paper no introduce $\tau$ explícito, pero trabajos posteriores de few-shot lo añadieron como hiperparámetro o parámetro aprendido, precisamente porque controla el balance entre kNN duro y KDE suave que discutimos en la Sección 4.

**Por qué un único softmax sobre todo $S$ y no comparaciones por pares.** En NCA, la probabilidad de que el punto $i$ elija al punto $j$ como vecino es un softmax sobre distancias, y la pérdida acumula contribuciones de todos los pares de la misma clase. MN reescribe esto para que el softmax se compute sobre **el support completo del episodio** y la salida sea directamente $P(y\mid\hat{x},S)$ — la cantidad que la métrica de evaluación $N$-way mide. Esta alineación entre objetivo de entrenamiento y métrica de test es, otra vez, el principio rector aplicado a la pérdida misma: no optimizamos un surrogate por pares con la esperanza de que correlacione con la precisión multi-way; optimizamos directamente la log-verosimilitud de la decisión multi-way.

---

## 6. Full Context Embeddings (FCE)

Aquí está, según los propios autores, "la principal novedad del modelo": reinterpretar un framework bien estudiado (redes con memoria externa) para hacer one-shot, y hacerlo con embeddings **condicionados al contexto completo**.

**El problema con embeddings independientes.** En la forma simple, $g(x_i)$ embebe cada elemento del support **de manera miópica**: independientemente de los demás elementos de $S$. Pero la decisión final está condicionada a todo el support a través de $P(\cdot\mid\hat{x},S)$. Hay una inconsistencia: ¿por qué embeber cada punto ignorando a sus vecinos en la tarea? Si dos elementos $x_i, x_j$ del support son muy parecidos, podría convenir embeberlos de forma que se **separen** para discriminar mejor. Y simétricamente, el modo en que embebemos el query $\hat{x}$ debería poder depender del support $S$.

MN resuelve ambas cosas haciendo que los embeddings tomen $S$ como entrada.

**(a) Embedding $g(x_i, S)$ — bidirectional LSTM.** Se considera el support $S$ como una **secuencia** y se codifica cada $x_i$ en su contexto con un biLSTM sobre $g'(x_i)$ (features crudas de una CNN). Apéndice A.2:

$$\overrightarrow{h}_i, \overrightarrow{c}_i = \mathrm{LSTM}(g'(x_i), \overrightarrow{h}_{i-1}, \overrightarrow{c}_{i-1}),$$
$$\overleftarrow{h}_i, \overleftarrow{c}_i = \mathrm{LSTM}(g'(x_i), \overleftarrow{h}_{i+1}, \overleftarrow{c}_{i+1}),$$
$$g(x_i, S) = \overrightarrow{h}_i + \overleftarrow{h}_i + g'(x_i),$$

con una **skip connection** ($+\,g'(x_i)$) que preserva las features originales. La recursión hacia atrás arranca en $i=|S|$. (Hay una pequeña errata tipográfica en la transcripción del paper donde aparece $\overrightarrow{h}_i+\overrightarrow{h}_i$; la intención, por construcción biLSTM, es la suma de la dirección hacia adelante y hacia atrás.)

**(b) Embedding $f(\hat{x}, S)$ — LSTM con atención (read/process/write).** El query se embebe con un LSTM que **lee con atención sobre todo el support** $K$ veces:

$$f(\hat{x}, S) = \mathrm{attLSTM}(f'(\hat{x}), g(S), K),$$

donde $f'(\hat{x})$ son las features de la CNN (constantes en cada paso), $g(S)$ es el conjunto embebido sobre el que se atiende, y $K$ es el número fijo de pasos de "procesamiento". Esto está tomado directamente del bloque "Process" de *Order Matters* (Vinyals et al., 2015). Las ecuaciones (Apéndice A.1, eqs. 3-6) implementan el patrón **read → process → write**:

$$\hat{h}_k, c_k = \mathrm{LSTM}\big(f'(\hat{x}),\, [h_{k-1}, r_{k-1}],\, c_{k-1}\big), \tag{3}$$
$$h_k = \hat{h}_k + f'(\hat{x}), \tag{4}$$
$$r_{k-1} = \sum_{i=1}^{|S|} a(h_{k-1}, g(x_i))\, g(x_i), \tag{5}$$
$$a(h_{k-1}, g(x_i)) = \mathrm{softmax}\big(h_{k-1}^\top\, g(x_i)\big). \tag{6}$$

Interpretación paso a paso:

- **Read (5-6):** el estado actual $h_{k-1}$ atiende por contenido sobre todos los $g(x_i)$ del support; el read-out $r_{k-1}$ es la suma ponderada de los embeddings del support. Esto permite al modelo **ignorar selectivamente** ciertos elementos del support.
- **Process (3):** el LSTM procesa $f'(\hat{x})$ (input constante) usando como estado el par $[h_{k-1}, r_{k-1}]$ (el output anterior concatenado con el read-out).
- **Write (4):** skip connection que suma $f'(\hat{x})$ al output. Tras $K$ pasos, $\mathrm{attLSTM}(\cdot) = h_K$.

Esto añade "profundidad" al cómputo de la atención: el embedding del query se refina iterativamente en función del support completo.

**Hallazgo empírico clave sobre FCE.** FCE **no ayudó en Omniglot** (tarea fácil; se omitió de la Tabla 1 por espacio), pero **sí ayudó en miniImageNet**, que es mucho más difícil — mejorando típicamente ~2 puntos porcentuales (de 41.2% a 44.2% sin fine-tune; de 42.4% a 46.6% con fine-tune en 1-shot). La lección: condicionar los embeddings al contexto importa cuando la tarea es lo bastante compleja como para que la geometría del espacio de features se beneficie de "ver" todo el support.

---

## 7. Estrategia episódica de entrenamiento

Esta es la operacionalización del principio de la Sección 3. Definiciones:

- Una **tarea** $T$ es una **distribución sobre conjuntos de etiquetas posibles** $L$. Típicamente $T$ pondera uniformemente todos los conjuntos de hasta unas pocas clases únicas (p.ej. 5), con pocos ejemplos por clase (p.ej. hasta 5). Un $L\sim T$ tendrá entonces ~5 a 25 ejemplos.
- Un **episodio** se construye así: (1) muestrear $L\sim T$ — por ejemplo $L=\{\text{cats}, \text{dogs}\}$; (2) usar $L$ para muestrear un support set $S$ y un batch $B$ (ambos son ejemplos etiquetados de cats y dogs); (3) entrenar la Matching Net a **minimizar el error de predecir las etiquetas de $B$ condicionado a $S$**.

El objetivo de entrenamiento (Ecuación 2) es:

$$\theta = \arg\max_\theta\; \mathbb{E}_{L\sim T}\Bigg[\, \mathbb{E}_{S\sim L,\, B\sim L}\bigg[ \sum_{(x,y)\in B} \log P_\theta(y\mid x, S) \bigg]\Bigg].$$

Léase de adentro hacia afuera: para cada query $(x,y)$ en el batch, maximizar la log-probabilidad de su etiqueta verdadera dado el support $S$; promediar sobre los muestreos de $S$ y $B$ dentro de un conjunto de etiquetas; y promediar sobre los conjuntos de etiquetas $L$ extraídos de la tarea $T$. Es **meta-learning** porque el procedimiento de entrenamiento aprende explícitamente a **aprender de un support dado** para minimizar la pérdida sobre un batch.

El protocolo experimental se llama **$N$-way $k$-shot**: a cada método se le dan $k$ ejemplos etiquetados de cada una de $N$ clases **no vistas en entrenamiento**, y debe clasificar un batch disjunto de ejemplos no etiquetados en una de esas $N$ clases. El **chance** (rendimiento aleatorio) es $1/N$. Se denota $L'$ el subconjunto held-out de etiquetas usadas solo para one-shot; el entrenamiento es siempre sobre $\neq L'$ y el test en modo one-shot sobre $L'$.

La promesa teórica: entrenar $\theta$ con la Ecuación 2 produce un modelo que funciona bien al muestrear $S'\sim T'$ de una distribución **distinta** de etiquetas nuevas, **sin ningún fine-tuning** sobre las clases nunca vistas, gracias a su naturaleza no-paramétrica. La advertencia honesta: obviamente, si $T'$ diverge mucho de la $T$ con la que se aprendió $\theta$, el modelo no funcionará (se elabora en la Sección 10 con el caso de los perros de ImageNet).

---

## 8. Experimentos

Tres datasets de cualidades diversas en complejidad, tamaño y modalidad: Omniglot, ImageNet (incluido miniImageNet) y Penn Treebank.

### 8.1 Omniglot

Omniglot (Lake et al., 2011) — el "transpuesto de MNIST": **1623 caracteres** de **50 alfabetos**, cada uno dibujado a mano por **20 personas** distintas. Muchas clases, pocos ejemplos por clase (20): ideal para one-shot a pequeña escala. Setup: elegir $N$ clases de caracteres no vistas como $L$, dar un dibujo de cada uno como $S\sim L$ y un batch $B\sim L$. Siguiendo a Santoro et al., se aumentó con rotaciones aleatorias en múltiplos de 90°, se usaron **1200 caracteres para train** y el resto para evaluación.

Embedding: una CNN simple pero potente — pila de 4 módulos, cada uno = conv $3\times3$ con 64 filtros + batch norm + ReLU + max-pool $2\times2$. Con imágenes redimensionadas a $28\times28$, tras 4 módulos el feature map es $1\times1\times64$, que es $f(x)$. (Esta arquitectura "Conv-4" se volvió el backbone estándar de los benchmarks de few-shot.)

**Resultados (Tabla 1):**

| Modelo | Fn | FT | 5-way 1-shot | 5-way 5-shot | 20-way 1-shot | 20-way 5-shot |
|---|---|---|---|---|---|---|
| Pixels | Cosine | N | 41.7% | 63.2% | 26.7% | 42.6% |
| Baseline Classifier | Cosine | N | 80.0% | 95.0% | 69.5% | 89.1% |
| Baseline Classifier | Softmax | Y | 86.0% | 97.6% | 72.9% | 92.3% |
| MANN (no conv) | Cosine | N | 82.8% | 94.9% | – | – |
| Conv Siamese Net | Cosine | N | 96.7% | 98.4% | 88.0% | 96.5% |
| Conv Siamese Net | Cosine | Y | 97.3% | 98.4% | 88.1% | 97.0% |
| **Matching Nets** | Cosine | N | **98.1%** | **98.9%** | **93.8%** | **98.5%** |
| **Matching Nets** | Cosine | Y | 97.9% | 98.7% | 93.5% | 98.7% |

MN gana en todos los regímenes. Observaciones: más ejemplos ($k$-shot) ayuda a todos; 5-way es más fácil que 20-way; las Siamese Nets compiten bien con 5 ejemplos pero **se degradan rápido en 1-shot** (88.0% vs 93.8% de MN en 20-way 1-shot). El fine-tuning casi no cambia a MN (no lo necesita). Transferencia disjunta: entrenado en Omniglot, en **MNIST 10-way 1-shot** MN logra **72%** vs 70% de Siamese y 63% del baseline.

### 8.2 miniImageNet (introducido en este paper)

ImageNet completo es un "feat de ingeniería" para experimentación rápida. Vinyals et al. **crearon miniImageNet**: **100 clases** de ImageNet elegidas al azar, **600 imágenes** color de $84\times84$ cada una (**60,000** en total). Split: **80 clases para train, 20 para test** (las 20 nunca vistas en entrenamiento). El split exacto (los IDs WordNet de las clases y las 600 imágenes por clase) se documenta en el Apéndice B vía un archivo público. Más complejo que CIFAR-10 pero cabe en memoria.

> Esto es históricamente enorme: **miniImageNet se convirtió en EL benchmark estándar del few-shot learning** durante los siguientes ~8 años. Casi todo paper de few-shot reporta números en este split. Nota técnica: el split de 64/16/20 que hoy se cita usualmente como "miniImageNet estándar" es el de Ravi & Larochelle (2017); el split original de **este** paper es 80/20.

**Resultados (Tabla 2, 5-way):**

| Modelo | Fn | FT | 1-shot | 5-shot |
|---|---|---|---|---|
| Pixels | Cosine | N | 23.0% | 26.6% |
| Baseline Classifier | Cosine | N | 36.6% | 46.0% |
| Baseline Classifier | Softmax | Y | 38.4% | 51.2% |
| Matching Nets | Cosine | N | 41.2% | 56.2% |
| Matching Nets | Cosine | Y | 42.4% | 58.0% |
| Matching Nets | Cosine (FCE) | N | 44.2% | 57.0% |
| **Matching Nets** | Cosine (FCE) | Y | **46.6%** | **60.0%** |

Aquí FCE sí aporta (~2 pp), y miniImageNet es lo bastante difícil para evaluarlo "sensatamente". El mejor número 1-shot del paper es **46.6%** (el spec menciona ~43.6%, valor entre las variantes con/sin FCE; el pico reportado en la Tabla 2 es 46.6%).

### 8.3 Full ImageNet: rand y dogs

Dos splits para test one-shot 5-way: **$L_{rand}$** (118 clases removidas al azar del train) y **$L_{dogs}$** (todas las 118 clases descendientes de "perro" removidas — fine-grained). Backbone: **Inception** entrenado en todas las clases excepto las de test. Se inicializaron $f$ y $g$ con los pesos de Inception y luego se entrenó con tareas 5-way 1-shot + FCE. El baseline classifier es uno de los modelos ImageNet más fuertes publicados (79% top-1).

**Resultados (Tabla 3, 5-way 1-shot):**

| Modelo | $L_{rand}$ | $\neq L_{rand}$ | $L_{dogs}$ | $\neq L_{dogs}$ |
|---|---|---|---|---|
| Pixels (Cosine) | 42.0% | 42.8% | 41.4% | 43.0% |
| Inception Classifier | 87.6% | 92.6% | 59.8% | 90.0% |
| **Matching Nets (FCE)** | **93.2%** | 97.0% | 58.8% | 96.4% |
| Inception Oracle (full) | ≈99% | ≈99% | ≈99% | ≈99% |

En $L_{rand}$, MN mejora a Inception por casi 6 pp (87.6%→93.2%), **reduciendo el error a la mitad**. La Figura 2 muestra casos donde Inception falla prefiriendo imágenes "cluttered" o de color constante, mientras MN se recupera de esos outliers en el support. En $L_{dogs}$, en cambio, MN **empeora 1 punto** (58.8% vs 59.8%) — analizado en la Sección 10.

### 8.4 Penn Treebank one-shot language modeling

Tarea nueva introducida aquí: dada una oración query con una palabra faltante (`<blank_token>`) y un support de oraciones, cada una con una palabra faltante y su etiqueta 1-hot, elegir del support la etiqueta que mejor matchea el query. Ejemplo del paper: el query *"in late new york trading yesterday the `<blank>` was quoted at N marks..."* matchea la etiqueta **dollar**.

Setup: oraciones del PTB (Marcus et al., 1993); set y batch con oraciones no solapadas; elección 5-way; batch size 20; $k\in\{1,2,3\}$; **9000 palabras para train, 1000 para test** (ni palabras ni oraciones de test vistas en train). Chance = 20%.

**Resultados:** MN con un encoder simple logra **32.4% / 36.1% / 38.2%** para $k=1,2,3$. El oráculo LSTM-LM (que ve todas las palabras — ventaja injusta, cota superior) llega a **72.8%**. La conclusión: combinar modelos paramétricos (LSTM-LM) con componentes no-paramétricos (MN) es trabajo futuro prometedor. Tareas relacionadas: CNN QA (Hermann et al.) y Children's Book Test (Hill et al.), que dan contexto secuencial específico, mientras la tarea de MN da contexto genérico.

---

## 9. Por qué importa

Dos legados perdurables, ambos hoy infraestructura básica del campo:

1. **El protocolo episódico $N$-way $k$-shot.** Antes de MN, la evaluación de one-shot era ad hoc. Este paper formalizó el episodio (muestrear tarea → support + query disjuntos → predecir) y el objetivo de la Ecuación 2. Hoy "entrenar episódicamente" es el default en few-shot, y el vocabulario (support set, query set, $N$-way, $k$-shot, chance $=1/N$) es universal.

2. **miniImageNet.** Crear un benchmark del tamaño correcto —difícil pero ejecutable en una sola máquina— fue catalítico. Permitió que cientos de grupos compararan métodos de forma reproducible durante casi una década.

A nivel conceptual, MN unificó tres tradiciones (metric learning à la NCA/Siamese, atención/memoria à la seq2seq/Memory Networks, y meta-learning à la MANN) en un solo modelo diferenciable end-to-end, y articuló el principio "entrena como evalúas" de forma que cambió cómo se piensa el problema.

---

## 10. Limitaciones

Los autores son explícitos sobre los flancos débiles:

1. **Costo computacional cuadrático/lineal sobre el support.** A medida que $S$ crece, el cómputo de cada actualización de gradiente se encarece. La atención requiere comparar el query contra **cada** elemento del support; con FCE, el attLSTM hace $K$ reads sobre todo $S$ y el biLSTM lo procesa secuencialmente. Para support sets grandes esto no escala. Los autores apuntan a métodos sparse y de muestreo como mitigación, y lo señalan como foco de trabajo futuro.

2. **FCE complica la arquitectura.** Introduce LSTMs (uno bidireccional para $g$, uno con atención para $f$), pasos de procesamiento $K$, ordenamiento del support como secuencia (cuando conceptualmente es un conjunto sin orden), y solo paga cuando la tarea es difícil (no ayudó en Omniglot). Es complejidad adicional con beneficio condicional.

3. **Degradación bajo shift de distribución de etiquetas.** El caso de $L_{dogs}$ es la lección más honesta: MN empeora 1 punto porque entrena muestreando $S$ de una distribución **uniforme** sobre las hojas del árbol de clases de ImageNet (clases dispares), pero en test el support $L_{dogs}$ contiene clases **similares entre sí** (clasificación fine-grained de razas de perro). El desajuste entre la distribución de tareas de train ($T$) y la de test ($T'$) rompe la promesa. Los autores hipotetizan que muestrear $S$ de conjuntos fine-grained de etiquetas en entrenamiento cerraría la brecha — lo dejan como trabajo futuro. Esto es coherente con su propio principio: si las condiciones de test (tareas fine-grained) no coinciden con las de train (tareas dispares), el modelo sufre.

---

## 11. Legado

Matching Networks abrió el linaje de **meta-learning basado en métricas** (metric-based meta-learning). Descendientes directos:

- **Prototypical Networks** (Snell, Swersky, Zemel, 2017): simplifican MN promediando los embeddings de cada clase en un **prototipo** $c_n = \frac{1}{|S_n|}\sum g(x_i)$ y clasificando por **distancia euclídea** al prototipo (no coseno, no FCE). Más simple y a menudo más preciso — una crítica implícita a la complejidad de FCE.
- **Relation Networks** (Sung et al., 2018): reemplazan la métrica fija (coseno) por una **métrica aprendida** — un módulo CNN que toma el par concatenado (query, support) y aprende un score de relación.
- **Ravi & Larochelle (2017)**: un meta-learner LSTM que aprende la regla de actualización; redefinieron el split estándar de miniImageNet (64/16/20).
- Eventualmente **MAML** (Finn et al., 2017) abrió la rama complementaria (meta-learning basado en optimización), pero comparte el protocolo episódico que MN estableció.

**Conexión con la atención de Transformers.** El attention kernel de MN —softmax de similitudes, suma ponderada de "valores" (las etiquetas)— es estructuralmente la **misma operación** que la atención de *Attention Is All You Need* (2017): $\mathrm{softmax}(QK^\top/\sqrt{d})V$. En MN, el query $f(\hat{x})$ es $Q$, los $g(x_i)$ son las keys $K$, y las etiquetas $y_i$ son los values $V$. MN es, en retrospectiva, un mecanismo de **cross-attention sobre una memoria etiquetada no-paramétrica**. La conexión va más allá de lo superficial: la idea de que "clasificar es atender sobre un conjunto de ejemplos y agregar sus valores" reaparece en los Transformers y, más recientemente, en el **in-context learning** de los LLMs —donde el modelo "aprende" de los ejemplos del prompt sin actualizar pesos, exactamente el espíritu no-paramétrico de MN.

---

## 12. Conexión con la Clase 26 (métodos no-paramétricos) y relevancia para salud

**Para la Clase 26.** MN es el ejemplo arquetípico de un **clasificador no-paramétrico aprendido**. La taxonomía que vale la pena fijar:

- **kNN / KDE clásicos:** no-paramétricos, métrica fija (euclídea/coseno cruda), sin entrenamiento. Desempeño limitado por la métrica.
- **Metric learning (NCA, Siamese):** aprenden una representación tal que la métrica funcione, pero típicamente con pérdidas surrogate por pares/triples.
- **Matching Networks:** unifican ambos — embedding aprendido end-to-end + clasificación no-paramétrica por atención sobre el support, con un objetivo **directamente alineado** con la decisión $N$-way. La Ecuación 1 literalmente **subsume KDE y kNN** como casos particulares del attention kernel.

La frase de la Clase 26 a retener: la flexibilidad de un modelo no-paramétrico (la memoria crece con los datos, asimilación instantánea, sin olvido catastrófico) combinada con el poder de representación de un embedding profundo. El precio es el costo de inferencia que escala con el support.

**Relevancia para salud y oncología (FALP).** El régimen one-shot/few-shot es **endémico en medicina**, donde los datos escasos no son un accidente sino la norma:

- **Enfermedades raras y subtipos tumorales poco frecuentes:** por definición hay pocos casos etiquetados. Un clasificador paramétrico estándar sobreajusta; un esquema tipo MN —embedding profundo entrenado sobre subtipos comunes, clasificación no-paramétrica sobre un support de los subtipos raros— encaja con el problema. El support set sería literalmente "estos 3 casos confirmados de este subtipo".
- **Adaptación a clases nuevas sin reentrenar:** la propiedad de MN de clasificar clases nunca vistas **sin fine-tuning** es valiosa en producción clínica, donde reentrenar y revalidar un modelo es costoso y regulatoriamente pesado. Agregar una categoría diagnóstica nueva = agregar ejemplos al support, no reescribir pesos.
- **Matching de pacientes (tu dominio FHIR/MDM):** la analogía es directa. El "support set" son los registros candidatos; el "query" es el registro a vincular; la atención por similitud aprendida sobre embeddings es exactamente la mecánica de un bi-encoder blocker que recupera candidatos por similitud coseno. La lección de MN —"test and train conditions must match"— se traduce en: entrena tu scorer de matching con la **misma distribución de pares difíciles** que verás en producción (no pares aleatorios fáciles), o sufrirás el equivalente del caso $L_{dogs}$. Y la advertencia sobre el costo cuadrático de la atención sobre support grande es precisamente por qué en MDM se usa un blocker para reducir candidatos antes de scorear — no se puede atender sobre toda la población.
- **Calibración y shift de distribución:** la degradación de MN en fine-grained ($L_{dogs}$) es una advertencia clínica directa. Un modelo few-shot entrenado sobre casos heterogéneos puede fallar al enfrentarse a diferenciar entidades muy similares (p.ej. distinguir dos lesiones de aspecto casi idéntico). La distribución de tareas de entrenamiento debe reflejar la dificultad real del despliegue.

En suma: Matching Networks no es solo un hito histórico del meta-learning; es un marco mental — "aprende un embedding, clasifica por atención no-paramétrica sobre ejemplos de referencia, y entrena imitando exactamente el régimen de evaluación" — que sigue siendo directamente aplicable a problemas de bajos datos en salud.
