---
title: "Matching Networks (One-Shot Learning)"
weight: 263
math: true
---
{{< paper-card
    title="Matching Networks for One Shot Learning"
    authors="Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Koray Kavukcuoglu, Daan Wierstra"
    year="2016"
    venue="NeurIPS 2016 (Google DeepMind)"
    pdf="/papers/matching-networks-vinyals-2016.pdf"
    arxiv="1606.04080" />}}

## El problema

Un niño generaliza el concepto de "jirafa" desde una sola foto en un libro. Los mejores sistemas de deep learning de 2016 necesitaban cientos o miles de ejemplos por clase. El deep learning supervisado clásico —AlexNet, VGG, los RNN para lenguaje— es notorio por su voracidad de datos: data augmentation y regularización alivian el sobreajuste en regímenes de bajos datos, pero el aprendizaje sigue siendo lento porque exige muchas actualizaciones de pesos vía descenso de gradiente.

El diagnóstico de los autores es preciso: el cuello de botella está en el **aspecto paramétrico** del modelo. Los ejemplos de entrenamiento deben "destilarse lentamente" dentro de los parámetros $\theta$ a fuerza de SGD. Cada concepto nuevo exige reescribir pesos, con dos costos: lentitud y **olvido catastrófico**.

Los modelos **no-paramétricos**, en cambio, asimilan ejemplos nuevos de forma instantánea y no sufren olvido catastrófico. El ejemplo canónico es $k$-nearest neighbors: no requiere entrenamiento, solo guardar los ejemplos. Pero su desempeño depende críticamente de la **métrica** elegida. Aquí entra el *metric learning*: aprender una representación tal que vecinos cercanos compartan etiqueta. *Matching Networks* (MN) lleva esta idea al extremo y la combina con el paradigma de "aprender a aprender" (meta-learning), produciendo un clasificador que reconoce clases **nunca vistas durante el entrenamiento sin ningún cambio en la red** — sin fine-tuning obligatorio.

## El principio: "test and train conditions must match"

El corazón conceptual del paper cabe en una frase: **las condiciones de test y de entrenamiento deben coincidir**. Es un principio elemental de machine learning que casi nadie aplicaba al one-shot.

El razonamiento. En test, el modelo recibirá un *support set* $S'$ con clases nunca vistas, unos pocos ejemplos por clase, y deberá clasificar *queries*. Si en entrenamiento lo exponemos al régimen habitual —minibatches grandes de un conjunto fijo de clases, con muchos ejemplos cada una— creamos un **desajuste de distribución entre train y test**: el modelo nunca practicó el acto de "mirar pocos ejemplos de clases nuevas y decidir". Por eso falla.

La solución es entrenar mostrando solo unos pocos ejemplos por clase, **cambiando la tarea de minibatch a minibatch**, exactamente como será evaluado. De ahí el muestreo episódico. La conclusión del paper lo resume: *"one-shot learning is much easier if you train the network to do one-shot learning"*. Suena tautológico, pero antes de este trabajo el campo no lo hacía sistemáticamente.

Conviene separar este principio (un procedimiento de entrenamiento) de la arquitectura (un modelo no-paramétrico). Son contribuciones ortogonales que se potencian: la arquitectura no-paramétrica permite incorporar el support sin reescribir pesos, y el entrenamiento episódico le enseña a explotar bien ese support.

## El clasificador no-paramétrico

Formalmente, MN aprende un mapeo $S \to c_S(\cdot)$: dado un support set $S=\{(x_i,y_i)\}_{i=1}^k$, produce un clasificador $c_S(\hat{x})$ que define una distribución de probabilidad sobre las etiquetas de un ejemplo de prueba $\hat{x}$. Este mapeo se parametriza como una red neuronal $P_\theta(\hat{y}\mid \hat{x}, S)$ y la predicción es $\arg\max_y P(y\mid\hat{x}, S)$.

La forma más simple del modelo es:

$$\hat{y} = \sum_{i=1}^{k} a(\hat{x}, x_i)\, y_i,$$

donde las $y_i$ son las etiquetas del support (codificadas one-hot) y $a(\hat{x}, x_i)$ es un **mecanismo de atención**: un peso escalar que mide cuánto se parece $\hat{x}$ a $x_i$. El resultado $\hat{y}$ es entonces una **combinación lineal de las etiquetas del support**, ponderada por similitud. Si $a$ suma 1 (es un softmax), $\hat{y}$ es directamente la distribución de probabilidad sobre clases.

La elección del kernel de atención especifica todo el clasificador. La forma propuesta es un **softmax sobre la distancia coseno** entre embeddings:

$$a(\hat{x}, x_i) = \frac{\exp\big(c(f(\hat{x}), g(x_i))\big)}{\sum_{j=1}^{k} \exp\big(c(f(\hat{x}), g(x_j))\big)},$$

donde $f$ y $g$ son **funciones de embedding** —redes neuronales, potencialmente $f=g$— que mapean el query y cada elemento del support a un espacio de features (CNN profundas para visión; un word embedding simple para lenguaje), y $c(u,v) = \frac{u^\top v}{\lVert u\rVert\,\lVert v\rVert}$ es la **similitud coseno**.

Mecánicamente: se embebe el query con $f$ y cada elemento del support con $g$, se computa la similitud coseno de $\hat{x}$ contra cada $x_i$, y el softmax normaliza esas similitudes a pesos que suman 1. Es exactamente la estructura de **atención por contenido** de Bahdanau et al. (2014), pero los "valores" sobre los que se atiende son las **etiquetas** del support, no estados ocultos de un decoder.

El paper ofrece tres lecturas iluminadoras de esta ecuación:

1. **Estimador de densidad por kernel (KDE):** si $a$ es un kernel, cada $x_i$ aporta densidad alrededor de su etiqueta.
2. **$k$-$b$ nearest neighbours:** si la atención es cero para los ejemplos más lejanos y constante para el resto, se reduce a una variante de kNN.
3. **Memoria asociativa:** $a$ actúa como atención y las $y_i$ son "memorias" ligadas a sus $x_i$; el modelo "apunta" al ejemplo correspondiente y recupera su etiqueta.

Por eso la ecuación **subsume tanto KDE como kNN**: es una generalización suave y aprendible de ambos. El uso de coseno (no distancia euclídea) es deliberado: normaliza la magnitud de los embeddings, estabiliza el softmax y evita que features de norma grande dominen. La temperatura implícita del softmax controla cuán "duro" es el voto — con similitudes muy grandes se concentra en el vecino más cercano (kNN clásico), con similitudes pequeñas reparte uniformemente (KDE de banda ancha).

A diferencia de NCA, que acumula contribuciones por pares, MN normaliza sobre **todo el support set** en un solo softmax y la salida es directamente $P(y\mid\hat{x},S)$ — la cantidad exacta que la métrica $N$-way mide en test. Esa alineación entre objetivo de entrenamiento y métrica de evaluación es el principio rector aplicado a la pérdida misma: no se optimiza un surrogate por pares esperando que correlacione con la precisión multi-way; se optimiza directamente la log-verosimilitud de la decisión multi-way.

## Full Context Embeddings

Aquí está, según los propios autores, "la principal novedad del modelo". En la forma simple, $g(x_i)$ embebe cada elemento del support de manera **miópica**: independientemente de los demás elementos de $S$. Pero la decisión final está condicionada a todo el support a través de $P(\cdot\mid\hat{x},S)$. Hay una inconsistencia: si dos elementos del support son muy parecidos, podría convenir embeberlos de forma que se **separen** para discriminar mejor; y el modo de embeber el query $\hat{x}$ debería poder depender del support $S$. Los *Full Context Embeddings* (FCE) resuelven ambas cosas haciendo que los embeddings tomen $S$ como entrada.

**Embedding $g(x_i, S)$ — biLSTM.** Se trata el support como una secuencia y se codifica cada $x_i$ en su contexto con un LSTM bidireccional sobre las features crudas $g'(x_i)$ de una CNN:

$$g(x_i, S) = \overrightarrow{h}_i + \overleftarrow{h}_i + g'(x_i),$$

con una **skip connection** ($+\,g'(x_i)$) que preserva las features originales.

**Embedding $f(\hat{x}, S)$ — LSTM con atención.** El query se embebe con un LSTM que **lee con atención sobre todo el support** $K$ veces, $f(\hat{x}, S) = \mathrm{attLSTM}(f'(\hat{x}), g(S), K)$, implementando el patrón **read → process → write** tomado del bloque "Process" de *Order Matters* (Vinyals et al., 2015):

$$\hat{h}_k, c_k = \mathrm{LSTM}\big(f'(\hat{x}),\, [h_{k-1}, r_{k-1}],\, c_{k-1}\big),$$
$$h_k = \hat{h}_k + f'(\hat{x}),$$
$$r_{k-1} = \sum_{i=1}^{|S|} a(h_{k-1}, g(x_i))\, g(x_i),\qquad a(h_{k-1}, g(x_i)) = \mathrm{softmax}\big(h_{k-1}^\top\, g(x_i)\big).$$

En cada paso, el estado actual atiende por contenido sobre todos los $g(x_i)$ (read), produce un read-out $r_{k-1}$ que es la suma ponderada de los embeddings del support, el LSTM procesa el input constante $f'(\hat{x})$ con ese contexto (process), y una skip connection lo vuelve a sumar (write). Tras $K$ pasos, $f(\hat{x},S) = h_K$. Esto añade "profundidad" al cómputo de la atención: el embedding del query se refina iterativamente en función del support completo.

**Hallazgo empírico clave:** FCE **no ayudó en Omniglot** (tarea fácil, se omitió de la tabla por espacio) pero **sí en miniImageNet** (mucho más difícil), mejorando ~2 puntos porcentuales (de 41.2% a 44.2% sin fine-tune; de 42.4% a 46.6% con fine-tune en 1-shot). La lección: condicionar los embeddings al contexto importa cuando la tarea es lo bastante compleja como para que la geometría del espacio de features se beneficie de "ver" todo el support.

## El protocolo episódico N-way K-shot

Esta es la operacionalización del principio "test and train conditions must match". Una **tarea** $T$ es una distribución sobre conjuntos de etiquetas posibles $L$ (típicamente uniforme sobre conjuntos de hasta unas pocas clases únicas, con pocos ejemplos por clase). Un **episodio** se construye así: (1) muestrear $L\sim T$ —por ejemplo $L=\{\text{cats}, \text{dogs}\}$—; (2) usar $L$ para muestrear un support set $S$ y un batch $B$ disjunto; (3) entrenar la red para minimizar el error de predecir las etiquetas de $B$ condicionado a $S$.

El objetivo de entrenamiento es:

$$\theta = \arg\max_\theta\; \mathbb{E}_{L\sim T}\Bigg[\, \mathbb{E}_{S\sim L,\, B\sim L}\bigg[ \sum_{(x,y)\in B} \log P_\theta(y\mid x, S) \bigg]\Bigg].$$

Léase de adentro hacia afuera: para cada query del batch, maximizar la log-probabilidad de su etiqueta verdadera dado el support; promediar sobre los muestreos de $S$ y $B$; y promediar sobre los conjuntos de etiquetas $L$. Es **meta-learning** porque el procedimiento aprende explícitamente a aprender de un support dado.

El protocolo experimental se llama **$N$-way $k$-shot**: a cada método se le dan $k$ ejemplos etiquetados de cada una de $N$ clases **no vistas en entrenamiento**, y debe clasificar un batch disjunto de ejemplos no etiquetados en una de esas $N$ clases. El rendimiento aleatorio (*chance*) es $1/N$. La promesa: entrenar $\theta$ con esa ecuación produce un modelo que funciona bien al muestrear de una distribución distinta de etiquetas nuevas **sin ningún fine-tuning**, gracias a su naturaleza no-paramétrica. La advertencia honesta: si la distribución de tareas de test diverge mucho de la de entrenamiento, el modelo no funcionará.

Para evaluar este protocolo a escala manejable, los autores **crearon miniImageNet**: 100 clases de ImageNet elegidas al azar, 600 imágenes a color de $84\times84$ cada una (60.000 en total), con un split de 80 clases para entrenamiento y 20 para test (nunca vistas). Más complejo que CIFAR-10 pero cabe en memoria de una sola máquina.

## Resultados

Tres datasets de complejidad y modalidad diversas.

**Omniglot** —el "transpuesto de MNIST": 1623 caracteres de 50 alfabetos, cada uno dibujado a mano por 20 personas (muchas clases, pocos ejemplos). Embedding: una CNN simple (4 módulos de conv $3\times3$ con 64 filtros + batch norm + ReLU + max-pool), que se volvió el backbone "Conv-4" estándar del few-shot.

| Modelo | 5-way 1-shot | 5-way 5-shot | 20-way 1-shot | 20-way 5-shot |
|---|---|---|---|---|
| Conv Siamese Net | 96.7% | 98.4% | 88.0% | 96.5% |
| **Matching Nets** | **98.1%** | **98.9%** | **93.8%** | **98.5%** |

MN gana en todos los regímenes. Las Siamese Nets compiten bien con 5 ejemplos pero se degradan rápido en 1-shot (88.0% vs 93.8% de MN en 20-way). El fine-tuning casi no cambia a MN: no lo necesita. En transferencia disjunta —entrenado en Omniglot, evaluado en MNIST 10-way 1-shot— MN logra 72% vs 70% de Siamese y 63% del baseline.

**miniImageNet** (5-way):

| Modelo | 1-shot | 5-shot |
|---|---|---|
| Baseline Classifier (Cosine) | 36.6% | 46.0% |
| Matching Nets (Cosine) | 41.2% | 56.2% |
| **Matching Nets (Cosine, FCE, fine-tune)** | **46.6%** | **60.0%** |

Aquí FCE sí aporta ~2 puntos, y la dificultad de miniImageNet permite evaluarlo con sentido.

**Full ImageNet** (5-way 1-shot), con backbone Inception:

| Modelo | $L_{rand}$ | $L_{dogs}$ |
|---|---|---|
| Inception Classifier | 87.6% | 59.8% |
| **Matching Nets (FCE)** | **93.2%** | 58.8% |

En el split aleatorio, MN mejora a Inception por casi 6 puntos (87.6% → 93.2%), **reduciendo el error a la mitad**. En el split de perros (fine-grained), en cambio, MN empeora 1 punto — una lección sobre el shift de distribución que se discute más abajo.

**Penn Treebank one-shot language modeling** (tarea nueva introducida aquí): dada una oración query con una palabra faltante (`<blank>`) y un support de oraciones con su palabra faltante etiquetada, elegir la etiqueta que mejor matchea. Con elección 5-way (chance = 20%), MN logra **32.4% / 36.1% / 38.2%** para $k=1,2,3$, contra un oráculo LSTM-LM —que ve todas las palabras, ventaja injusta— al 72.8%.

## Por qué importa hoy

El kernel de atención de MN —softmax de similitudes, suma ponderada de "valores" (las etiquetas)— es estructuralmente la **misma operación** que la atención de *Attention Is All You Need* (2017): $\mathrm{softmax}(QK^\top/\sqrt{d})V$. En MN, el query $f(\hat{x})$ es $Q$, los embeddings $g(x_i)$ son las keys $K$, y las etiquetas $y_i$ son los values $V$. MN es, en retrospectiva, un mecanismo de **cross-attention sobre una memoria etiquetada no-paramétrica**.

La conexión va más allá de lo superficial. La idea de que "clasificar es atender sobre un conjunto de ejemplos y agregar sus valores" reaparece en los Transformers y, más recientemente, en el **in-context learning** de los LLMs: el modelo "aprende" de los ejemplos del prompt sin actualizar pesos, exactamente el espíritu no-paramétrico de MN. Un prompt few-shot es, en este sentido, un support set; la respuesta del modelo, una clasificación por atención sobre él.

Dos legados de infraestructura perduran. El **protocolo episódico $N$-way $k$-shot** es hoy el default del few-shot, y su vocabulario (support set, query set, $N$-way, $k$-shot, chance $=1/N$) es universal. Y **miniImageNet** se convirtió en EL benchmark estándar del campo durante casi una década. (Nota: el split de 64/16/20 que hoy se cita como "estándar" es el de Ravi & Larochelle (2017); el split original de este paper es 80/20.)

La principal limitación —reconocida por los autores— es el **costo computacional que escala con el support**: la atención compara el query contra cada elemento de $S$, y con FCE el attLSTM hace $K$ reads sobre todo $S$. Para support sets grandes esto no escala. Y la degradación en el split de perros enseña que un modelo few-shot entrenado sobre tareas dispares puede fallar al diferenciar entidades muy similares: la distribución de tareas de entrenamiento debe reflejar la dificultad real del despliegue — el principio rector, otra vez.

## Conexión con la Clase 26

MN es el ejemplo arquetípico de un **clasificador no-paramétrico aprendido**, el tema central de la clase. La taxonomía que vale la pena fijar:

- **kNN / KDE clásicos:** no-paramétricos, métrica fija (euclídea/coseno cruda), sin entrenamiento. Su desempeño está limitado por la métrica.
- **Metric learning (NCA, Siamese):** aprenden una representación tal que la métrica funcione, pero con pérdidas surrogate por pares o triples.
- **Matching Networks:** unifican ambos — embedding aprendido end-to-end + clasificación no-paramétrica por atención sobre el support, con un objetivo directamente alineado con la decisión $N$-way. La ecuación de MN literalmente subsume KDE y kNN como casos particulares.

La frase a retener: la flexibilidad de un modelo no-paramétrico (la memoria crece con los datos, asimilación instantánea, sin olvido catastrófico) combinada con el poder de representación de un embedding profundo. El precio es el costo de inferencia que escala con el support.

El régimen one-shot/few-shot es endémico en salud, donde los datos escasos son la norma y no un accidente: enfermedades raras y subtipos tumorales poco frecuentes (pocos casos etiquetados por definición), adaptación a categorías diagnósticas nuevas sin reentrenar ni revalidar un modelo, y *matching* de registros por similitud aprendida sobre embeddings. La lección "test and train conditions must match" se traduce directamente: entrena tu clasificador con la misma distribución de casos difíciles que verás en producción, o sufrirás el equivalente del caso de los perros.

## Notas y enlaces

**Fundamentos:** [Meta-aprendizaje](/fundamentos/meta-aprendizaje) · [Metric learning](/fundamentos/metric-learning) · [Few-shot learning](/fundamentos/few-shot-learning) · [Self-attention](/fundamentos/self-attention)

**Papers relacionados:** [Prototypical Networks (Snell, 2017)](/papers/prototypical-networks-snell-2017) · [Siamese Networks (Koch, 2015)](/papers/siamese-networks-koch-2015) · [MAML (Finn, 2017)](/papers/maml-finn-2017)

**Clase:** Ver [Clase 26 -- Meta-aprendizaje](/clases/clase-26).
