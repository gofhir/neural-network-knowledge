# Optimization as a Model for Few-Shot Learning (Meta-Learner LSTM) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Optimization as a Model for Few-Shot Learning*.
- **Autores:** Sachin Ravi y Hugo Larochelle (Twitter, Cambridge, USA; trabajo realizado durante una pasantía de Ravi en Twitter — Ravi era estudiante de doctorado en Princeton).
- **Venue:** *5th International Conference on Learning Representations* (ICLR 2017), Toulon, Francia. Publicado como conference paper.
- **Año:** 2017. **OpenReview:** [openreview.net/forum?id=rJY0-Kcll](https://openreview.net/forum?id=rJY0-Kcll).
- **Código:** `github.com/twitter/meta-learning-lstm`.

Este paper hace **dos aportes que el Laboratorio 26 usa sin nombrarlos**. El primero es conceptual: propone un **meta-learner LSTM**, un optimizador *aprendido* que descubre simultáneamente (a) la **regla de actualización** con la que se entrena un clasificador en el régimen few-shot y (b) la **inicialización** de ese clasificador. El segundo es de infraestructura: introduce la versión canónica del **split de Mini-ImageNet** —64 clases de entrenamiento, 16 de validación, 20 de test— que se volvió el benchmark estándar de clasificación few-shot y que librerías como `learn2learn` o `torchmeta` cargan por defecto.

La tesis del paper parte de una observación incómoda: el descenso de gradiente y sus variantes (momentum, Adagrad, Adadelta, ADAM) "weren't designed specifically to perform well under the constraint of a set number of updates". Cuando solo tienes 1 o 5 ejemplos por clase y un puñado de pasos de actualización, esos optimizadores —pensados para millones de iteraciones— fracasan, y peor aún, cada dataset nuevo arranca desde una inicialización aleatoria que "considerably hurts its ability to converge to a good solution after a few updates". La propuesta es no diseñar el optimizador a mano sino **aprenderlo**, aprovechando una analogía estructural precisa entre la actualización de SGD y la actualización de la celda de memoria de una LSTM.

Resultado de cabecera (Tabla 1 del paper, Mini-ImageNet 5-way, intervalos de confianza al 95%): el Meta-Learner LSTM logra **43.44 ± 0.77%** en 1-shot y **60.60 ± 0.71%** en 5-shot, igualando a Matching Networks en 1-shot (43.56%) y superándolas con margen claro en 5-shot (55.31%). Es competitivo con el estado del arte métrico de la época usando una idea radicalmente distinta: optimización aprendida en lugar de comparación de embeddings.

## 2. Contexto histórico: aprender a optimizar antes de MAML

El meta-aprendizaje —"learning at two levels", adquisición rápida de conocimiento dentro de cada tarea guiada por una extracción lenta de información común a todas las tareas— tiene raíces profundas, que el paper rastrea hasta Schmidhuber (1987, 1992, 1993), Bengio et al. (1990, 1995) y Thrun (1998). Lo nuevo en 2016-2017 era combinar esa idea con redes profundas y benchmarks estandarizados. En ese momento las dos familias dominantes eran muy distintas de lo que este paper propone:

**1. Métodos métricos / no paramétricos.** Las redes siamesas (Koch, 2015) aprenden un embedding donde ejemplos de la misma clase quedan cerca y de distintas clases lejos. Matching Networks (Vinyals et al., 2016) refinan la idea con una pérdida diferenciable de vecino más cercano basada en similitud coseno de embeddings, haciendo coincidir las condiciones de entrenamiento y test. Estos eran los **mejores métodos de la época** ("the best performing methods for few-shot learning have been mainly metric learning methods"), y son el rival directo que el paper enfrenta en su tabla de resultados.

**2. Optimizadores aprendidos / meta-learners de gradiente.** Aquí está la genealogía directa de este paper. Schmidhuber (1992, 1993) exploró redes que modifican sus propios pesos de forma diferenciable. Bengio et al. (1990, 1995) buscaron reglas de actualización biológicamente plausibles. Hochreiter et al. (2001) y Bosc usaron LSTMs para entrenar perceptrones multicapa. Y el trabajo más cercano y citado constantemente es **Andrychowicz et al. (2016)**, *"Learning to learn by gradient descent by gradient descent"*, que usa una LSTM para producir las actualizaciones de pesos de otra red. La diferencia crítica que Ravi & Larochelle marcan: Andrychowicz et al. buscan un optimizador *general* para clasificación a gran escala, mientras que ellos atacan específicamente el **régimen few-shot**, donde el número de actualizaciones es fijo y pequeño, y donde además importa la inicialización.

La pregunta que motiva el paper: ¿se puede aprender un procedimiento de optimización que, en pocos pasos y con pocos ejemplos, lleve a un clasificador a buena generalización, y que además provea automáticamente "a beneficial common initialization" sin las desventajas de transfer learning (cuyo beneficio "greatly decreases as the task the network was trained on diverges from the target task", Yosinski et al., 2014)?

Vale anclar la posición histórica: este paper de ICLR 2017 es **anterior** a MAML (Finn et al., ICML 2017) por unos meses. De hecho, MAML se define explícitamente en contraste con este trabajo y lo cita como el meta-learner "que aprende tanto la inicialización como el optimizador", criticando que expande el número de parámetros y ata el método a la arquitectura LSTM. Es decir: **MAML nace como una simplificación de la idea de este paper** — conservar la parte de "aprender una buena inicialización para que pocos pasos de gradiente basten", pero descartar la LSTM y usar SGD plano en el inner loop.

## 3. La contribución doble

### 3a. El meta-learner LSTM (la idea profunda)

El núcleo del paper es una analogía que, una vez vista, es difícil de no ver. La actualización estándar de descenso de gradiente para los parámetros $\theta$ del clasificador (el *learner*) es la Ecuación 1:

$$
\theta_t = \theta_{t-1} - \alpha_t \nabla_{\theta_{t-1}} \mathcal{L}_t
$$

donde $\theta_{t-1}$ son los parámetros tras $t-1$ actualizaciones, $\alpha_t$ es el learning rate, $\mathcal{L}_t$ la pérdida en el paso $t$, y $\nabla_{\theta_{t-1}}\mathcal{L}_t$ el gradiente.

La observación clave: **esta actualización tiene exactamente la misma forma que la actualización de la celda de memoria de una LSTM** (Hochreiter & Schmidhuber, 1997), la Ecuación 2:

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

La correspondencia es término a término. Si igualamos:

- $c_{t-1} = \theta_{t-1}$ — la **celda de memoria es el vector de parámetros del clasificador**.
- $\tilde{c}_t = -\nabla_{\theta_{t-1}}\mathcal{L}_t$ — la **celda candidata es el gradiente negativo** (la dirección de descenso).
- $i_t = \alpha_t$ — la **input gate juega el papel del learning rate**.
- $f_t = 1$ — la **forget gate fija en 1** recupera SGD puro.

Con estas sustituciones, $c_t = 1 \cdot \theta_{t-1} + \alpha_t \cdot (-\nabla_{\theta_{t-1}}\mathcal{L}_t) = \theta_{t-1} - \alpha_t \nabla_{\theta_{t-1}}\mathcal{L}_t$, que es precisamente SGD. La LSTM, con su forget gate y su input gate, **generaliza** SGD: SGD es el caso particular en que la forget gate vale constantemente 1 y la input gate es un escalar fijo.

El salto del paper es entonces inevitable: **¿por qué dejar $f_t$ e $i_t$ fijos si podemos aprenderlos?** La propuesta es entrenar una meta-learner LSTM cuya celda de estado *es* $\theta_t$, donde las gates se vuelven funciones aprendibles de información de optimización:

**Input gate como learning rate adaptativo.** En lugar de un $\alpha$ fijo:

$$
i_t = \sigma\big(W_I \cdot [\nabla_{\theta_{t-1}}\mathcal{L}_t,\ \mathcal{L}_t,\ \theta_{t-1},\ i_{t-1}] + b_I\big)
$$

El learning rate ahora es función del **gradiente actual, la pérdida actual, los parámetros actuales y el learning rate previo**. El meta-learner aprende a modular finamente cuánto avanzar en cada coordenada y en cada paso "so as to train the learner quickly while avoiding divergence".

**Forget gate como shrinkage / weight decay adaptativo.** El paper argumenta que el valor óptimo de $f_t$ no tiene por qué ser 1:

$$
f_t = \sigma\big(W_F \cdot [\nabla_{\theta_{t-1}}\mathcal{L}_t,\ \mathcal{L}_t,\ \theta_{t-1},\ f_{t-1}] + b_F\big)
$$

La intuición es elegante: encoger los parámetros (olvidar parte de su valor previo) tiene sentido "if the learner is currently in a bad local optima and needs a large change to escape" — una situación que se detecta cuando "the loss is high but the gradient is close to zero". Un gradiente casi nulo con pérdida alta es la firma de un mal mínimo local plano; en ese caso la forget gate puede contraer $\theta$ para reiniciar la búsqueda. Cuando $f_t < 1$, la actualización se vuelve $\theta_t = f_t \theta_{t-1} - i_t \nabla\mathcal{L}_t$, que es exactamente un paso de gradiente **con weight decay** (regularización L2) cuyo coeficiente el meta-learner controla dinámicamente.

**La inicialización aprendida $c_0 \leftrightarrow \theta_0$.** Esta es la pieza que conecta directamente con MAML. Como $c_0$ (el estado inicial de la celda) es un parámetro de la LSTM, **se puede aprender por descenso de gradiente como cualquier otro peso del meta-learner**. Pero $c_0 = \theta_0$ son los pesos iniciales del clasificador. Por tanto, el meta-learner aprende "the optimal initial weights of the learner so that training begins from a beneficial starting point that allows optimization to proceed rapidly". Esta es la misma intuición que MAML llevaría al extremo: una inicialización tal que pocos pasos basten. Aquí viene "gratis" como subproducto de tratar $\theta_0$ como el estado inicial de la celda LSTM.

El paper también nota que el modelo se parece a la actualización del estado oculto de una **GRU** (Cho et al., 2014), "with the exception that the forget and input gates aren't tied to sum to one" — a diferencia de la GRU, aquí $f_t$ e $i_t$ son independientes, lo que da más libertad (shrinkage y learning rate se controlan por separado).

### 3b. El split de Mini-ImageNet (la contribución de dataset)

Mini-ImageNet fue *propuesto* por Vinyals et al. (2016) como un benchmark que ofrece "the challenges of the complexity of ImageNet images, without requiring the resources and infrastructure necessary to run on the full ImageNet dataset". Pero —y esto es central— **Vinyals et al. nunca publicaron los splits exactos** que usaron. Ravi & Larochelle, al no poder reproducirlos, crearon su propia versión y, al liberar el código, fijaron de facto el estándar que la comunidad adoptó:

> "Because the exact splits used in Vinyals et al. (2016) were not released, we create our own version of the Mini-Imagenet dataset by selecting a random **100 classes from ImageNet** and picking **600 examples of each class**. We use **64, 16, and 20 classes for training, validation and testing**, respectively."

Estos números —100 clases, 600 imágenes por clase, split 64/16/20— son **exactamente lo que cargan hoy `learn2learn`, `torchmeta` y la mayoría de los pipelines de few-shot**. Cuando el notebook del Lab 26 invoca el dataset Mini-ImageNet "tal cual" y obtiene 64 clases de entrenamiento sin explicar de dónde salen, está usando esta convención de Ravi & Larochelle. (Nota menor: el paper de MAML reporta el split como 64/12/24; la versión 64/16/20 de este paper es la que prevaleció y la que las librerías estandarizaron.)

La importancia metodológica del split es la **disjunción de clases**: las clases de meta-train, meta-validation y meta-test no se solapan. El meta-test "cover[s] classes not present in any of the datasets in $\mathcal{D}_{meta-train}$". Esto garantiza que el few-shot learning evalúa generalización a **categorías nunca vistas**, no a ejemplos nuevos de categorías conocidas — la diferencia esencial entre meta-aprendizaje y clasificación ordinaria.

## 4. El setup few-shot N-way K-shot y los episodios

El paper formaliza con cuidado la jerarquía de datos del meta-aprendizaje, una terminología que el Lab 26 hereda. En ML estándar tenemos un dataset $D$ con su split train/test. En meta-aprendizaje manejamos **meta-sets** $\mathcal{D}$ que *contienen múltiples datasets regulares*, donde cada $D \in \mathcal{D}$ tiene a su vez su propio $D_{train}$ y $D_{test}$.

La tarea es **$k$-shot, $N$-class**: para cada dataset $D$, el conjunto de entrenamiento tiene $k$ ejemplos etiquetados por cada una de $N$ clases, es decir $D_{train}$ tiene $k \cdot N$ ejemplos, y $D_{test}$ un número fijo de ejemplos para evaluar. Siguiendo a Vinyals et al. (2016), cada uno de estos datasets se llama un **episodio**.

Hay tres meta-sets disjuntos:

- $\mathcal{D}_{meta-train}$: sobre el cual se entrena el meta-learner. El objetivo es que, dado un $D_{train}$, el meta-learner produzca un clasificador con alto desempeño promedio en el $D_{test}$ correspondiente.
- $\mathcal{D}_{meta-validation}$: para selección de hiperparámetros del meta-learner.
- $\mathcal{D}_{meta-test}$: para evaluar la generalización final.

El procedimiento para generar un episodio $D = (D_{train}, D_{test})$: se muestrean $N$ clases del meta-set correspondiente, luego $k$ ejemplos de cada clase para formar $D_{train}$, y un conjunto adicional (15 por clase en este paper) para $D_{test}$. Durante el meta-entrenamiento se muestrean episodios repetidamente; para validación y test se fija un conjunto de episodios para que el intervalo de confianza de la accuracy media sea pequeño. El paper considera $k=1$ y $k=5$ con $N=5$ clases.

## 5. Cómo se entrena el meta-learner

La pregunta operativa es: ¿cómo entrenar la LSTM para que sea efectiva? El principio guía, tomado de Vinyals et al. (2016), es que **"it is key to have training conditions match those of test time"**. Por eso el objetivo de entrenamiento del meta-learner es la **pérdida del clasificador producido sobre el conjunto de test del episodio**, $\mathcal{L}_{test}$.

El flujo (Algoritmo 1 del paper) por cada episodio $D \in \mathcal{D}_{meta-train}$:

1. Inicializar $\theta_0 \leftarrow c_0$ (la inicialización aprendida del meta-learner).
2. Para $t = 1, \dots, T$: tomar un batch de $D_{train}$, computar la pérdida $\mathcal{L}_t$ y el gradiente $\nabla_{\theta_{t-1}}\mathcal{L}_t$ del learner; pasar $(\nabla_{\theta_{t-1}}\mathcal{L}_t, \mathcal{L}_t)$ a la meta-learner LSTM, que produce $c_t$ vía la Ecuación 2; fijar $\theta_t \leftarrow c_t$.
3. Tras $T$ pasos, evaluar el clasificador con parámetros finales $\theta_T$ sobre $D_{test}$ para obtener $\mathcal{L}_{test}$.
4. Actualizar los parámetros $\Theta$ del meta-learner con $\nabla_\Theta \mathcal{L}_{test}$.

Hay tres detalles de implementación que importan:

**Parameter sharing entre coordenadas.** Un clasificador profundo tiene decenas de miles de parámetros; si la LSTM tuviera que procesarlos todos conjuntamente, su tamaño explotaría. Siguiendo a Andrychowicz et al. (2016), **se comparten los parámetros de la LSTM a través de todas las coordenadas del gradiente**: cada coordenada tiene su propio estado oculto y de celda, pero los pesos $W_I, W_F$ son los mismos para todas. Esto produce una LSTM compacta y, además, "the same update rule is used for each coordinate, but one that is dependent on the respective history of each coordinate" — una regla universal pero con memoria por-coordenada. Se implementa metiendo un batch de coordenadas $(\nabla_{\theta_t,i}\mathcal{L}_t, \mathcal{L}_t)$ como entrada.

**Preprocesamiento de gradientes y pérdidas.** Como las coordenadas pueden tener magnitudes muy distintas, se aplica el preprocesamiento de Andrychowicz et al. (2016), que separa magnitud y signo: para cada $x$, si $|x| \geq e^{-p}$ se mapea a $(\log(|x|)/p,\ \text{sgn}(x))$, y en otro caso a $(-1, e^p x)$. El paper usa $p=10$. Esto evita que el meta-learner se vea desbordado por escalas dispares.

**Independencia de gradientes (la simplificación clave).** En rigor, las pérdidas y gradientes del learner *dependen* de los parámetros del meta-learner, así que el gradiente de $\mathcal{L}_{test}$ respecto a $\Theta$ debería propagarse a través de esa dependencia — lo que implicaría **segundas derivadas**. Siguiendo a Andrychowicz et al., el paper **asume que esas contribuciones son despreciables y las ignora** ("we make the simplifying assumption that these contributions to the gradients aren't important and can be ignored, which allows us to avoid taking second derivatives, a considerably expensive operation"). Empíricamente el meta-learner entrena bien igual. Es interesante notar que este mismo dilema de segundo orden reaparecerá en MAML, que sí tiene la opción de computarlo exacto y luego lo aproxima con FOMAML.

**Inicialización y batch norm.** Se inicializa el bias de la forget gate alto (cerca de 1, para flujo de gradiente) y el de la input gate bajo (learning rate inicial pequeño), de modo que "the meta-learner starts close to normal gradient descent with a small learning rate", lo que estabiliza el inicio. Para batch normalization, el paper toma cuidado de **no filtrar estadísticas entre episodios**: durante meta-test se acumulan estadísticas por episodio y se borran al pasar al siguiente, manteniendo simétricas las condiciones meta-train/meta-test.

## 6. Experimentos y resultados

**Arquitecturas.** El learner es una CNN simple de 4 capas convolucionales (3×3, 32 filtros, batch norm, ReLU, max-pooling 2×2) con una capa lineal final y softmax — la arquitectura canónica del few-shot que MAML también adoptaría (MAML usa 64 filtros en Omniglot, 32 en Mini-ImageNet). El meta-learner es una LSTM de 2 capas: la primera es una LSTM normal que preprocesa, la segunda es la LSTM modificada que implementa la regla de actualización. Se entrena con ADAM (lr 0.001) y gradient clipping (0.25). Para 1-shot el meta-learner hace **12 actualizaciones**, para 5-shot hace **5** — y crucialmente "better performance for each task was attained if the meta-learner is explicitly trained to do the set number of updates during meta-testing that will be used".

**Resultados en Mini-ImageNet (Tabla 1, 5-class):**

| Modelo | 1-shot | 5-shot |
|---|---|---|
| Baseline-finetune | 28.86 ± 0.54% | 49.79 ± 0.79% |
| Baseline-nearest-neighbor | 41.08 ± 0.70% | 51.04 ± 0.65% |
| Matching Network | 43.40 ± 0.78% | 51.09 ± 0.71% |
| Matching Network FCE | 43.56 ± 0.84% | 55.31 ± 0.73% |
| **Meta-Learner LSTM (este paper)** | **43.44 ± 0.77%** | **60.60 ± 0.71%** |

Los dos baselines son instructivos. **Baseline-nearest-neighbor** entrena una red para clasificar conjuntamente todas las clases de meta-train, luego en meta-test embebe los ejemplos de soporte y clasifica por vecino más cercano. **Baseline-finetune** es "a coarser version of our meta-learner": entrena conjuntamente y luego hace fine-tuning con SGD (buscando learning rate y decay en meta-validation) sobre cada $D_{train}$ antes de evaluar. El hallazgo más revelador: **el baseline de fine-tuning es *peor* que el de nearest-neighbor** (28.86% vs 41.08% en 1-shot). La razón: "Because we are not regularizing the classifier, with very few updates the fine-tuning model overfits, especially in the 1-shot case." Este sobreajuste catastrófico es precisamente el argumento a favor de **meta-entrenar la inicialización end-to-end**: el meta-learner LSTM aprende a posicionar $\theta_0$ y a modular las gates para que pocos pasos no sobreajusten.

Contra Matching Networks (incluida la variante FCE con bidirectional-LSTM y attention-LSTM, el estado del arte de la época): en 1-shot el intervalo de confianza del meta-learner se solapa con el de Matching Networks (empate estadístico), y en **5-shot lo supera con claridad** (60.60% vs 55.31%). El paper aclara que sus números no coinciden con los de Vinyals et al. porque crearon su propia versión del dataset y reimplementaron los modelos.

**Visualización de la estrategia aprendida (Figura 3).** Inspeccionando los valores de $i_t$ y $f_t$ por paso, el meta-learner revela su estrategia. Las **forget gate values** muestran "a simple weight decay strategy that seems consistent across different layers" — confirma la interpretación de la forget gate como shrinkage/weight decay. Las **input gate values** son más difíciles de interpretar pero exhiben "a lot of variability between different datasets, indicating that the meta-learner isn't simply learning a fixed optimization strategy" — es decir, el learning rate aprendido se *adapta al episodio*, no es un schedule fijo. Además difieren entre 1-shot y 5-shot, mostrando que el meta-learner adopta métodos distintos según las condiciones de cada régimen.

## 7. Limitaciones

- **Costo y complejidad de parámetros.** Aunque el parameter sharing por coordenadas mantiene la LSTM compacta, el meta-learner *es* un modelo recurrente entrenable adicional — un costo que MAML eliminaría por completo. La crítica explícita de MAML es que este enfoque "increases the number of learned parameters" y ata el método a una arquitectura LSTM concreta.
- **La aproximación de independencia de gradientes.** Ignorar la dependencia de $\mathcal{L}_t$ y $\nabla\mathcal{L}_t$ respecto a $\Theta$ (para evitar segundas derivadas) es una simplificación admitida; funciona empíricamente pero no es el gradiente exacto del meta-objetivo.
- **Número de actualizaciones fijo y específico.** El meta-learner debe entrenarse para el número exacto de pasos que usará en test (12 para 1-shot, 5 para 5-shot). No generaliza libremente a otros presupuestos de pasos.
- **Alcance restringido.** El propio paper lo reconoce en la conclusión: "we focused our study to the few-shot and few-classes setting", y plantea como trabajo futuro extenderlo "across a full spectrum of settings" (muchos o pocos ejemplos, muchas o pocas clases). El enfoque está diseñado para el régimen pequeño.
- **Solo clasificación.** A diferencia de MAML, que es model-agnostic y se aplica a regresión y RL, este meta-learner se evalúa solo en clasificación de imágenes.

## 8. Impacto: el puente hacia MAML

La trascendencia de este paper es doble y ambas caras viven en el Lab 26.

**Como dataset:** el split 64/16/20 de Mini-ImageNet se convirtió en el benchmark de facto de la clasificación few-shot durante años. Prácticamente todo paper posterior (MAML, Prototypical Networks, Relation Networks, LEO, ...) reporta sobre esta partición. Las librerías de meta-aprendizaje la incorporaron como configuración por defecto, razón por la cual el Lab 26 la carga sin ceremonia.

**Como idea:** este es el ancestro conceptual directo de MAML. La cadena lógica es nítida. Ravi & Larochelle muestran que se puede aprender *tanto la regla de actualización como la inicialización* tratando $\theta_t$ como la celda de una LSTM. MAML (Finn et al., 2017), apenas meses después, observa que **la pieza que más rinde es la inicialización aprendida** y que la regla de actualización aprendida (la LSTM) se puede *descartar* sin perder casi nada: basta usar SGD plano en el inner loop y meta-aprender solo $\theta_0$. En la analogía de este paper, MAML equivale a **fijar la forget gate en 1 e instalar un learning rate fijo (SGD puro), pero conservar y optimizar $c_0 = \theta_0$**. MAML lo dice casi con esas palabras al posicionarse: critica que el meta-learner LSTM añade parámetros y arquitectura, y propone quedarse solo con la inicialización. La Tabla 1 de MAML compara *contra* este paper (cita "meta-learner LSTM" con 43.44%/60.60%) y reporta 48.70%/63.11%, superándolo — pero la deuda intelectual es total. La idea de "aprender a optimizar / aprender una inicialización para que pocos pasos basten" nace operativamente aquí.

La familia que estos dos papers fundan —el **meta-aprendizaje basado en optimización**— es una de las tres canónicas (junto a la métrica de Matching/Prototypical Networks y la basada en memoria de MANN). Este paper es donde "optimization as a model for few-shot learning" deja de ser una metáfora y se vuelve un algoritmo entrenable.

## 9. Conexión con el Laboratorio 26 (Meta-aprendizaje)

El Lab 26 usa **MAML** y carga **Mini-ImageNet**. Este paper es la fuente silenciosa de ambas cosas.

**El dataset que el lab carga sin explicar.** Cuando el notebook obtiene Mini-ImageNet vía `learn2learn` y aparecen 64 clases de meta-train, 16 de meta-validation y 20 de meta-test, esos números *son* la convención fijada en la Sección 5.1 de este paper. Entender el porqué del split —disjunción de clases para evaluar generalización a categorías nuevas, no a ejemplos nuevos— es entender qué mide realmente la accuracy few-shot que el lab reporta. Conviene también saber el origen práctico del split: nació de la *imposibilidad de reproducir* los splits no publicados de Vinyals et al., un recordatorio concreto de por qué liberar splits y semillas importa en ML reproducible (relevante para cualquier pipeline clínico auditable).

**De dónde viene el "aprender a optimizar" de MAML.** El lab corre MAML como caja relativamente cerrada: un inner loop de SGD y un outer loop que actualiza la inicialización. Este paper expone la pregunta anterior: *¿por qué SGD y no una regla aprendida?* Al mostrar que SGD es literalmente el caso particular de una celda LSTM con forget gate = 1 e input gate = learning rate fijo, el paper hace evidente que el inner loop de MAML es una *elección de diseño*, no una necesidad. MAML simplificó la LSTM a SGD plano y se quedó solo con la inicialización $c_0 = \theta_0$. Quien entiende la analogía gradient descent ↔ celda LSTM de este paper entiende *qué descartó MAML y por qué*: la regla de actualización aprendida (gates adaptativas) aportaba poco frente a la inicialización aprendida, que aportaba casi todo.

**El argumento del overfitting que justifica el meta-aprendizaje.** El hallazgo de que el baseline de fine-tuning (28.86%) es peor que el de vecino más cercano (41.08%) por sobreajuste en el régimen de pocos pasos es el mismo fenómeno que MAML ataca y que el lab observa: fine-tunear ingenuamente desde una inicialización cualquiera, con 1-5 ejemplos, sobreajusta. La solución de ambos papers —meta-entrenar la inicialización end-to-end para que la adaptación rápida *generalice* en vez de memorizar— es la lección central del Lab 26.

En síntesis para la clase: este paper es el eslabón que vuelve operativa la idea de "aprender a aprender mediante optimización", aporta el benchmark que toda la línea usaría, y deja preparado el terreno exacto sobre el que MAML construiría su simplificación elegante. Leerlo antes (o junto a) MAML convierte a MAML de receta en consecuencia lógica.
