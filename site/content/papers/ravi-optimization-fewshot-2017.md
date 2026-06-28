---
title: "Optimization as a Model for Few-Shot Learning"
weight: 269
math: true
---

{{< paper-card
    title="Optimization as a Model for Few-Shot Learning"
    authors="Sachin Ravi, Hugo Larochelle"
    year="2017"
    venue="ICLR 2017"
    pdf="/papers/ravi-optimization-fewshot-2017.pdf"
    arxiv="" >}}
Este paper hace dos aportes que el Laboratorio 26 usa sin nombrarlos. El primero es conceptual: un **meta-learner LSTM**, un optimizador *aprendido* que descubre a la vez la regla de actualización con que se entrena un clasificador few-shot y la inicialización de sus pesos. La idea nace de una analogía exacta: la actualización de descenso de gradiente $\theta_t = \theta_{t-1} - \alpha_t \nabla \mathcal{L}_t$ tiene la misma forma que la actualización de la celda de una LSTM, así que basta con *aprender* las gates en vez de fijarlas. El segundo aporte es de infraestructura: introduce el **split estándar de Mini-ImageNet (64/16/20 clases)** que se volvió el benchmark de facto del few-shot. Resultado de cabecera: **43.44%** (1-shot) y **60.60%** (5-shot) en Mini-ImageNet 5-way, igualando a Matching Networks en 1-shot y superándolas en 5-shot. Es el ancestro conceptual directo de MAML.
{{< /paper-card >}}

---

## El problema

El meta-aprendizaje persigue "aprender a dos niveles": una adquisición rápida de conocimiento dentro de cada tarea, guiada por una extracción lenta de lo común a todas las tareas. El régimen difícil es el **few-shot**: solo 1 o 5 ejemplos etiquetados por clase y un puñado de pasos de actualización.

La observación incómoda que motiva el paper es que el descenso de gradiente y sus variantes (momentum, Adagrad, Adadelta, ADAM) "no fueron diseñados específicamente para rendir bien bajo la restricción de un número fijo de actualizaciones". Esos optimizadores están pensados para millones de iteraciones; cuando solo tienes un puñado de pasos, fracasan. Y peor aún: cada dataset nuevo arranca desde una inicialización aleatoria que, según el paper, "perjudica considerablemente su capacidad de converger a una buena solución tras pocas actualizaciones".

¿La salida obvia, transfer learning? También tiene un costo: su beneficio "decrece notablemente a medida que la tarea sobre la que se entrenó la red diverge de la tarea objetivo" (Yosinski et al., 2014). La propuesta de Ravi y Larochelle es radical: no diseñar el optimizador a mano sino **aprenderlo**, y que ese mismo procedimiento provea automáticamente "una inicialización común beneficiosa" sin las desventajas del transfer learning.

---

## La contribución doble

### 1. El meta-learner LSTM (la idea profunda)

El núcleo es la propuesta de un optimizador entrenable: una **LSTM cuya celda de memoria *es* el vector de parámetros del clasificador**. En lugar de aplicar SGD a mano, una red recurrente aprende la regla de actualización completa —incluyendo el learning rate, una forma de weight decay y la inicialización— observando cómo se comporta la optimización paso a paso. Es "aprender a optimizar" tomado al pie de la letra.

### 2. El split estándar de Mini-ImageNet (la contribución de dataset)

Mini-ImageNet fue *propuesto* por Vinyals et al. (2016), pero **nunca publicaron los splits exactos** que usaron. Ravi y Larochelle, al no poder reproducirlos, crearon su propia versión y, al liberar el código, fijaron de facto el estándar que la comunidad adoptó:

> Seleccionan 100 clases aleatorias de ImageNet con 600 ejemplos cada una, y usan **64, 16 y 20 clases para entrenamiento, validación y test** respectivamente.

Estos números —100 clases, 600 imágenes por clase, split 64/16/20— son exactamente los que cargan hoy `learn2learn`, `torchmeta` y la mayoría de los pipelines de few-shot. Lo desarrollamos abajo, porque es la pieza que el Lab 26 toca directamente.

---

## El método: la analogía gradient descent ↔ celda LSTM

Aquí está la idea que, una vez vista, cuesta no ver. La actualización estándar de descenso de gradiente para los parámetros $\theta$ del clasificador (el *learner*) es:

$$
\theta_t = \theta_{t-1} - \alpha_t \nabla_{\theta_{t-1}} \mathcal{L}_t
$$

donde $\theta_{t-1}$ son los parámetros tras $t-1$ actualizaciones, $\alpha_t$ el learning rate, $\mathcal{L}_t$ la pérdida en el paso $t$ y $\nabla_{\theta_{t-1}}\mathcal{L}_t$ el gradiente.

La observación clave: **esta actualización tiene exactamente la misma forma que la de la celda de memoria de una LSTM** (Hochreiter y Schmidhuber, 1997):

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

La correspondencia es término a término. Si igualamos:

- $c_{t-1} = \theta_{t-1}$ — la **celda de memoria es el vector de parámetros** del clasificador.
- $\tilde{c}_t = -\nabla_{\theta_{t-1}}\mathcal{L}_t$ — la **celda candidata es el gradiente negativo** (la dirección de descenso).
- $i_t = \alpha_t$ — la **input gate hace de learning rate**.
- $f_t = 1$ — la **forget gate fija en 1** recupera SGD puro.

Con estas sustituciones, $c_t = 1 \cdot \theta_{t-1} + \alpha_t \cdot (-\nabla_{\theta_{t-1}}\mathcal{L}_t) = \theta_{t-1} - \alpha_t \nabla_{\theta_{t-1}}\mathcal{L}_t$, que es precisamente SGD. La LSTM **generaliza** el descenso de gradiente: SGD es el caso particular en que la forget gate vale constantemente 1 y la input gate es un escalar fijo. El salto es entonces inevitable: **¿por qué dejar $f_t$ e $i_t$ fijos si podemos aprenderlos?**

**Input gate $i_t$ ↔ learning rate adaptativo.** En lugar de un $\alpha$ fijo, la input gate se vuelve una función aprendida:

$$
i_t = \sigma\big(W_I \cdot [\nabla_{\theta_{t-1}}\mathcal{L}_t,\ \mathcal{L}_t,\ \theta_{t-1},\ i_{t-1}] + b_I\big)
$$

El learning rate pasa a ser función del gradiente actual, la pérdida actual, los parámetros actuales y el learning rate previo. El meta-learner aprende a modular finamente cuánto avanzar en cada coordenada y en cada paso, "para entrenar al learner rápidamente evitando la divergencia".

**Forget gate $f_t$ ↔ shrinkage / weight decay adaptativo.** El paper argumenta que el valor óptimo de $f_t$ no tiene por qué ser 1:

$$
f_t = \sigma\big(W_F \cdot [\nabla_{\theta_{t-1}}\mathcal{L}_t,\ \mathcal{L}_t,\ \theta_{t-1},\ f_{t-1}] + b_F\big)
$$

La intuición es elegante: encoger los parámetros (olvidar parte de su valor previo) tiene sentido "si el learner está atrapado en un mal mínimo local y necesita un cambio grande para escapar", situación que se detecta cuando "la pérdida es alta pero el gradiente es cercano a cero" — la firma de un mínimo plano malo. Cuando $f_t < 1$, la actualización se vuelve $\theta_t = f_t \theta_{t-1} - i_t \nabla\mathcal{L}_t$, que es exactamente un paso de gradiente **con weight decay** (regularización L2) cuyo coeficiente controla dinámicamente el meta-learner.

**Estado inicial $c_0$ ↔ inicialización $\theta_0$.** Esta es la pieza que conecta directo con MAML. Como $c_0$ (el estado inicial de la celda) es un parámetro de la LSTM, **se puede aprender por descenso de gradiente como cualquier otro peso del meta-learner**. Pero $c_0 = \theta_0$ son los pesos iniciales del clasificador. Por tanto el meta-learner aprende "los pesos iniciales óptimos del learner, de modo que el entrenamiento empiece desde un punto de partida beneficioso que permita optimizar rápidamente". Esta inicialización aprendida viene "gratis", como subproducto de tratar $\theta_0$ como el estado inicial de la celda.

El paper nota además un parecido con la **GRU** (Cho et al., 2014), "con la excepción de que la forget gate y la input gate no están atadas a sumar uno": aquí $f_t$ e $i_t$ son independientes, lo que permite controlar shrinkage y learning rate por separado.

Un detalle de entrenamiento que importa: siguiendo a Andrychowicz et al. (2016), los parámetros de la LSTM se **comparten entre todas las coordenadas** del gradiente (cada coordenada tiene su propio estado de celda, pero los pesos $W_I, W_F$ son comunes), lo que mantiene la LSTM compacta. Y para evitar segundas derivadas, se asume que la dependencia de $\mathcal{L}_t$ y $\nabla\mathcal{L}_t$ respecto a los parámetros del meta-learner es despreciable — la misma decisión de primer orden que reaparecerá en FOMAML.

---

## El split de Mini-ImageNet

La tarea se formaliza como **$N$-way $K$-shot**: para cada episodio se muestrean $N$ clases, con $K$ ejemplos etiquetados por clase para el conjunto de soporte ($D_{train}$ tiene $K \cdot N$ ejemplos) y un conjunto adicional —15 por clase en este paper— para evaluar ($D_{test}$). Se manejan tres **meta-sets disjuntos**: $\mathcal{D}_{meta\text{-}train}$ para entrenar el meta-learner, $\mathcal{D}_{meta\text{-}validation}$ para hiperparámetros y $\mathcal{D}_{meta\text{-}test}$ para la evaluación final.

La importancia metodológica del split es la **disjunción de clases**: las clases de meta-train, meta-validation y meta-test no se solapan. El meta-test "cubre clases no presentes en ningún dataset de $\mathcal{D}_{meta\text{-}train}$". Esto garantiza que el few-shot learning evalúa generalización a **categorías nunca vistas**, no a ejemplos nuevos de categorías conocidas — la diferencia esencial entre meta-aprendizaje y clasificación ordinaria.

El split 64/16/20 nació de una necesidad práctica: la *imposibilidad de reproducir* los splits no publicados de Vinyals et al. Es un recordatorio concreto de por qué liberar splits y semillas importa en ML reproducible (relevante para cualquier pipeline clínico auditable). Nota menor: el paper de MAML reporta el split como 64/12/24; la versión 64/16/20 de este paper es la que prevaleció y la que las librerías estandarizaron.

---

## Resultados

**Arquitecturas.** El learner es una CNN simple de 4 bloques convolucionales (3×3, 32 filtros, batch norm, ReLU, max-pooling 2×2) con capa lineal final y softmax — la arquitectura canónica del few-shot que MAML también adoptaría. El meta-learner es una LSTM de 2 capas. Para 1-shot hace **12 actualizaciones**, para 5-shot hace **5**, y rinde mejor "si se lo entrena explícitamente para hacer el número exacto de actualizaciones que usará en meta-test".

Resultados en Mini-ImageNet (Tabla 1, 5-way, IC 95%):

| Modelo | 1-shot | 5-shot |
|---|---|---|
| Baseline-finetune | 28.86 ± 0.54% | 49.79 ± 0.79% |
| Baseline-nearest-neighbor | 41.08 ± 0.70% | 51.04 ± 0.65% |
| Matching Network | 43.40 ± 0.78% | 51.09 ± 0.71% |
| Matching Network FCE | 43.56 ± 0.84% | 55.31 ± 0.73% |
| **Meta-Learner LSTM (este paper)** | **43.44 ± 0.77%** | **60.60 ± 0.71%** |

El hallazgo más revelador está en los baselines: **el de fine-tuning es *peor* que el de vecino más cercano** (28.86% vs 41.08% en 1-shot). La razón: "como no regularizamos el clasificador, con muy pocas actualizaciones el modelo de fine-tuning sobreajusta, sobre todo en el caso 1-shot". Este sobreajuste catastrófico es precisamente el argumento a favor de **meta-entrenar la inicialización end-to-end**: el meta-learner aprende a posicionar $\theta_0$ y a modular las gates para que pocos pasos no sobreajusten.

Contra Matching Networks (incluida la variante FCE, estado del arte de la época): en 1-shot hay empate estadístico, y en **5-shot el meta-learner supera con claridad** (60.60% vs 55.31%). Es competitivo con el mejor método métrico usando una idea radicalmente distinta: optimización aprendida en lugar de comparación de embeddings.

Inspeccionando los valores aprendidos de las gates, el meta-learner revela su estrategia: las **forget gates** muestran "una estrategia simple de weight decay consistente entre capas" (confirma la lectura de shrinkage), mientras que las **input gates** exhiben "mucha variabilidad entre datasets, indicando que el meta-learner no aprende una estrategia de optimización fija" — el learning rate aprendido se *adapta al episodio*, no es un schedule prefijado.

---

## Conexión con el Laboratorio 26

El Lab 26 usa **MAML** y carga **Mini-ImageNet**. Este paper es la fuente silenciosa de ambas cosas.

**(a) El split de Mini-ImageNet que el lab carga sin explicar.** Cuando el notebook obtiene Mini-ImageNet vía `learn2learn` y aparecen 64 clases de meta-train, 16 de meta-validation y 20 de meta-test, esos números *son* la convención fijada en este paper. Entender el porqué del split —disjunción de clases para evaluar generalización a categorías nuevas, no a ejemplos nuevos— es entender qué mide realmente la accuracy few-shot que el lab reporta.

**(b) MAML es la simplificación directa de esta idea.** El lab corre MAML (Parte 1) como un inner loop de SGD plano y un outer loop que actualiza la inicialización. Este paper expone la pregunta anterior: *¿por qué SGD y no una regla aprendida?* Al mostrar que SGD es literalmente el caso particular de una celda LSTM con forget gate $= 1$ e input gate $=$ learning rate fijo, el paper deja claro que el inner loop de MAML es una *elección de diseño*, no una necesidad. MAML (Finn et al., 2017), apenas meses después, observó que **la pieza que más rinde es la inicialización aprendida** y que la regla de actualización aprendida (la LSTM) se puede *descartar* sin perder casi nada: conservar solo $\theta_0$ y usar SGD normal en el inner loop. En la analogía de este paper, MAML equivale a **fijar la forget gate en 1, instalar un learning rate fijo (SGD puro), y conservar y optimizar solo $c_0 = \theta_0$**. Quien entiende la analogía gradient descent ↔ celda LSTM entiende exactamente *qué descartó MAML y por qué*: las gates adaptativas aportaban poco frente a la inicialización aprendida, que aportaba casi todo.

El mismo argumento del overfitting cierra el círculo: el fine-tuning ingenuo desde una inicialización cualquiera, con 1-5 ejemplos, sobreajusta (el 28.86% del baseline). Meta-entrenar la inicialización end-to-end para que la adaptación rápida *generalice* en vez de memorizar es la lección central que el Lab 26 observa con MAML — y que este paper articuló primero.

---

## Notas y enlaces

Fundamentos: [Meta-aprendizaje](/fundamentos/meta-aprendizaje) - [Optimización binivel](/fundamentos/optimizacion-binivel) - [Few-shot learning](/fundamentos/few-shot-learning)

Papers relacionados: [MAML (Finn 2017)](/papers/maml-finn-2017) - [Matching Networks (Vinyals 2016)](/papers/matching-networks-vinyals-2016) - [learn2learn (Arnold 2020)](/papers/learn2learn-arnold-2020)

Laboratorio: [Lab 26 - Meta-aprendizaje](/laboratorios/lab-26)

Clase: [Clase 26 - Meta-aprendizaje](/clases/clase-26)
