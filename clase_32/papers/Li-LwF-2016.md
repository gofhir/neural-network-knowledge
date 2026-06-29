# Learning without Forgetting (LwF) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Learning without Forgetting*.
- **Autores:** Zhizhong Li y Derek Hoiem (Member, IEEE), ambos del Department of Computer Science, University of Illinois at Urbana-Champaign.
- **Venue:** publicado originalmente en **ECCV 2016** (Li & Hoiem, "Learning without Forgetting", *European Conference on Computer Vision*, Springer, pp. 614-629); esta versión extendida corresponde a la **revista TPAMI (IEEE Transactions on Pattern Analysis and Machine Intelligence), 2017**.
- **Preprint:** arXiv:1606.09282v3, 14 de febrero de 2017, [arxiv.org/abs/1606.09282](https://arxiv.org/abs/1606.09282).
- **Términos índice:** Convolutional Neural Networks, Transfer Learning, Multi-task Learning, Deep Learning, Visual Recognition.

El paper ataca un problema concreto y práctico: **agregar nuevas capacidades a una CNN ya entrenada sin tener acceso a los datos de entrenamiento de las tareas viejas, y sin degradar el desempeño en ellas**. El ejemplo motivador es nítido: un robot llega a una casa con un set de reconocedores de objetos por defecto, pero hace falta agregar modelos específicos del sitio; o un sistema de seguridad en construcción que detecta casco y chaleco, al que un supervisor quiere agregarle la detección de calzado inadecuado. En todos estos casos los datos legacy pueden estar sin registrar, ser propietarios o ser demasiado engorrosos de reutilizar. El reto es aprender la tarea nueva compartiendo parámetros con las viejas, **sin sufrir olvido catastrófico** (*catastrophic forgetting*, McCloskey & Cohen 1989; Goodfellow et al. 2013).

La tesis central es elegante: usar **Knowledge Distillation** (Hinton, Vinyals & Dean 2014) como mecanismo de preservación. Antes de entrenar la tarea nueva, se registran las salidas del modelo viejo sobre los **datos nuevos**; durante el entrenamiento, una pérdida de distillation mantiene esas salidas estables mientras una pérdida de cross-entropy aprende la tarea nueva. El resultado: LwF supera a feature extraction y, sorprendentemente, a fine-tuning en la tarea nueva, mientras preserva la vieja mucho mejor que fine-tuning — todo **sin un solo dato de las tareas antiguas**.

Para la Clase 32 (Olvido Catastrófico) esto importa porque LwF se presenta como uno de los **métodos de regularización basados en funciones de distillation** dentro del aprendizaje continuo, y porque fue el puente conceptual que llevó la distillation al *continual learning*, sirviendo de base directa a iCaRL (Rebuffi et al. 2017).

## 2. Contexto: el problema de agregar tareas sin datos viejos

El escenario formal: una CNN tiene **parámetros compartidos** $\theta_s$ (p.ej. las cinco capas convolucionales y dos fully-connected de AlexNet), **parámetros específicos de tareas viejas** $\theta_o$ (la capa de salida para clasificación de ImageNet y sus pesos), y se quieren agregar **parámetros específicos de la tarea nueva** $\theta_n$ (p.ej. un clasificador de escenas), inicializados aleatoriamente. Conviene pensar en $\theta_o$ y $\theta_n$ como clasificadores que operan sobre features parametrizadas por $\theta_s$. La pregunta es cómo aprender $\theta_n$ aprovechando $\theta_s$ ya aprendidos, sin romper $\theta_o$.

El paper enumera y disecciona las tres estrategias usadas hasta entonces, cada una con un defecto fatal en este escenario:

- **Feature Extraction** (Donahue et al. 2014): $\theta_s$ y $\theta_o$ quedan congelados; las activaciones de una o más capas se usan como features para entrenar $\theta_n$. **No adapta** la representación compartida: típicamente rinde por debajo en la tarea nueva porque los parámetros compartidos no capturan información discriminativa que la tarea nueva necesita. La ventaja es que preserva exactamente la tarea vieja.

- **Fine-tuning** (Girshick et al. 2014): $\theta_s$ y $\theta_n$ se optimizan para la tarea nueva (con learning rate bajo para evitar deriva), mientras $\theta_o$ queda fijo. **Olvida**: al cambiar $\theta_s$ sin guía para las predicciones específicas viejas, degrada significativamente el desempeño en la tarea original. Una variante, **Fine-tuning FC**, congela las convolucionales y solo afina las fully-connected (compromiso entre fine-tuning y feature extraction), pero, como muestran los experimentos, todavía degrada la tarea nueva. Duplicar y afinar una red por tarea evita el olvido pero hace que el tiempo de test crezca linealmente con el número de tareas.

- **Joint Training / Multitask Learning** (Caruana 1997): todos los parámetros $\theta_s, \theta_o, \theta_n$ se optimizan conjuntamente, intercalando muestras de cada tarea. Da el mejor desempeño combinado y se usa como **cota superior** de lo que LwF puede aspirar a lograr. Pero **requiere los datos de todas las tareas**: es cada vez más engorroso a medida que se acumulan tareas, e **imposible si los datos de las tareas viejas no están disponibles** — que es precisamente el supuesto del paper.

El paper también ubica métodos contemporáneos (sección de related work): **A-LTM** (Furlanello et al. 2016, casi idéntico en método pero con conclusiones opuestas sobre la necesidad de los datos viejos), **Less Forgetting Learning** (Jung et al. 2016, que mantiene fija la capa final vieja en vez de preservar sus salidas), y las **Progressive Neural Networks** (Rusu et al. 2016) que agregan nodos nuevos congelando los viejos a costa de expandir cuadráticamente los parámetros.

## 3. Contribución central

La contribución es el método **Learning without Forgetting (LwF)**, que puede verse como un **híbrido de Knowledge Distillation y fine-tuning**. La idea clave:

> Usar solamente los datos de la tarea nueva para optimizar simultáneamente (a) alta exactitud en la tarea nueva y (b) **preservación de las respuestas del modelo original en las tareas viejas**.

El procedimiento (Fig. 3 del paper) tiene dos momentos:

1. **Antes de entrenar:** se registran las salidas $Y_o = \text{CNN}(X_n, \theta_s, \theta_o)$ — es decir, se pasan las imágenes de la tarea **nueva** $X_n$ por la red **vieja** y se guardan las probabilidades que produce sobre las clases viejas. Estas salidas registradas se convierten en *soft targets* (pseudo-etiquetas) para las cabezas viejas.

2. **Durante el entrenamiento:** se optimiza una pérdida combinada que aprende la tarea nueva (cross-entropy contra las etiquetas reales $Y_n$) **mientras** una pérdida de distillation obliga a las salidas actuales sobre las cabezas viejas $\hat{Y}_o$ a mantenerse cerca de las $Y_o$ registradas.

La diferencia con joint training es exactamente esa: **joint training usa las imágenes y etiquetas de la tarea vieja; LwF las sustituye por las imágenes de la tarea nueva $X_n$ y las respuestas registradas $Y_o$**. Esto elimina la necesidad de almacenar el dataset viejo, conserva el beneficio de la optimización conjunta de $\theta_s$, y además ahorra cómputo porque $X_n$ pasa por las capas compartidas una sola vez para ambas tareas.

Las tres ventajas que el paper reivindica: **(1) desempeño de clasificación** (LwF supera feature extraction y fine-tuning en la tarea nueva, y supera ampliamente a fine-tuning en la vieja); **(2) eficiencia computacional** (entrenamiento más rápido que joint training y solo un poco más lento que fine-tuning; test más rápido que usar múltiples redes afinadas); **(3) simplicidad de despliegue** (una vez aprendida la tarea, no hace falta retener ni reaplicar sus datos).

## 4. Método: la pérdida combinada

LwF minimiza, vía SGD, una suma de tres términos sobre los datos nuevos:

$$\theta_s^*, \theta_o^*, \theta_n^* \leftarrow \underset{\hat\theta_s, \hat\theta_o, \hat\theta_n}{\arg\min}\Big[\, \lambda_o\, \mathcal{L}_{old}(Y_o, \hat{Y}_o) + \mathcal{L}_{new}(Y_n, \hat{Y}_n) + R(\hat\theta_s, \hat\theta_o, \hat\theta_n) \,\Big]$$

### 4.1. Pérdida de la tarea nueva

Es la **logística multinomial** (cross-entropy) estándar de clasificación multiclase:

$$\mathcal{L}_{new}(y_n, \hat{y}_n) = -\,y_n \cdot \log \hat{y}_n$$

donde $\hat{y}_n$ es la salida softmax y $y_n$ el vector one-hot de la etiqueta verdadera. Si hay varias tareas nuevas o es multi-label, se suman las pérdidas.

### 4.2. Pérdida de distillation para las cabezas viejas

Para cada tarea vieja se quiere que las probabilidades de salida se mantengan cerca de las registradas por la red original. Se usa la **Knowledge Distillation loss** de Hinton et al. — una cross-entropy modificada que aumenta el peso de las probabilidades pequeñas:

$$\mathcal{L}_{old}(y_o, \hat{y}_o) = -H(y_o', \hat{y}_o') = -\sum_{i=1}^{l} y_o'^{(i)} \log \hat{y}_o'^{(i)}$$

donde las versiones modificadas (con **temperatura** $T$) de las probabilidades registradas $y_o^{(i)}$ y actuales $\hat{y}_o^{(i)}$ son:

$$y_o'^{(i)} = \frac{(y_o^{(i)})^{1/T}}{\sum_j (y_o^{(j)})^{1/T}}, \qquad \hat{y}_o'^{(i)} = \frac{(\hat{y}_o^{(i)})^{1/T}}{\sum_j (\hat{y}_o^{(j)})^{1/T}}$$

Fijar $T > 1$ **suaviza** la distribución: aumenta el peso de los logits pequeños y empuja a la red a codificar mejor las **similitudes entre clases** (la "dark knowledge" de Hinton). El paper usa **$T = 2$** según un grid search en un held-out set, alineado con la recomendación de los autores de distillation. Un hallazgo importante de las ablaciones: el uso de la distillation loss rinde solo *ligeramente* mejor que otras pérdidas razonables (L1, L2, cross-entropy simple) — lo crucial es **restringir las salidas viejas a parecerse a las del original; la medida de similitud exacta no es determinante** (Fig. 7c-d).

### 4.3. Balance y regularización

- $\lambda_o$ es un **peso de balance** entre tarea vieja y nueva; se fija en **1** para la mayoría de experimentos. Subir $\lambda_o$ favorece la tarea vieja; barriendo $\lambda_o$ se traza una curva de compromiso viejo-nuevo (Fig. 7).
- $R$ es un simple **weight decay de 0.0005**.

### 4.4. Procedimiento de entrenamiento

Primero un **warm-up step** (se congelan $\theta_s$ y $\theta_o$, se entrena solo $\theta_n$ hasta converger), luego un **joint-optimize step** (se entrenan todos los pesos). Una ablación reveladora (Tabla 2b): el warm-up **no es crucial para LwF ni para LFL**, pero es **esencial para el desempeño en la tarea vieja de fine-tuning** (sin warm-up, fine-tuning cae de 50.9 a 42.5 en ImageNet→CUB). Detalles de implementación: MatConvNet, momentum 0.9, dropout en las FC, inicialización Xavier para $\theta_n$, learning rate de 0.1–0.02 veces el original.

## 5. Experimentos

El diseño evalúa si LwF aprende efectivamente una tarea nueva preservando las viejas, comparando contra feature extraction, fine-tuning, fine-tuning FC, LFL y — como cota superior — joint training. Redes: principalmente **AlexNet**, verificado también con **VGG-16**.

- **Tareas viejas (grandes):** ILSVRC 2012 / **ImageNet** (1000 clases, >1.000.000 imágenes) y **Places365-standard** (365 clases de escenas, ~1.600.000 imágenes). Se parte de redes bien entrenadas.
- **Tareas nuevas (moderadas):** PASCAL **VOC** 2012 (5.717 img, muy similar a ImageNet), Caltech-UCSD Birds **CUB** (5.994 img, fine-grained, disímil a ambas), MIT indoor **Scenes** (5.360 img, similar a Places365), y **MNIST** (caso deliberadamente adversario: dígitos manuscritos sin relación con ImageNet).

**Hallazgos principales (Tabla 1):**

- *En la tarea nueva:* LwF supera consistentemente a fine-tuning, LFL, fine-tuning FC y feature extraction, salvo ImageNet→MNIST y Places365→CUB con fine-tuning. La ganancia sobre fine-tuning fue **inesperada** e indica que **preservar las salidas viejas actúa como regularizador** que mejora la tarea nueva. Esto motiva, según los autores, reemplazar fine-tuning por LwF como práctica estándar de adaptación.
- *En la tarea vieja:* LwF supera por mucho a fine-tuning (que degrada fuerte al mover $\theta_s$), aunque suele quedar por debajo de feature extraction, fine-tuning FC y a veces LFL.
- *Considerando ambas tareas (Fig. 7):* ajustando $\lambda_o$, LwF domina a LFL y fine-tuning FC en la curva de compromiso del primer par de tareas.
- *Versus joint training:* LwF rinde **similar** con AlexNet (tiende a superar levemente en la tarea nueva y a quedar algo por debajo en la vieja), un resultado positivo porque no usa los datos viejos. Con VGG, joint training gana más consistentemente, sugiriendo que redes con más capacidad representacional se benefician más de tener los datos viejos.

**Ablaciones y estudios de robustez:**
- *Domain shift fuerte:* tareas disímiles degradan más la tarea vieja. CUB (muy distinta de Places365) provoca pérdidas de 8.4% (fine-tuning), 3.8% (LwF) y 1.5% (joint). En ImageNet→MNIST, LwF no preserva bien la tarea vieja: los dígitos manuscritos dan una **supervisión indirecta pobre** de la tarea original.
- *Tamaño del dataset nuevo* (3%–100% de CUB): las observaciones se mantienen; LwF supera a fine-tuning en ambas tareas, con diferencias que tienden a crecer con más datos.
- *Múltiples tareas incrementales:* agregando VOC/Scenes por partes, LwF se degrada más lento que fine-tuning, pero queda por debajo de joint training en las tareas viejas a medida que se acumulan.
- *Alternativas descartadas:* más capas task-specific (sin ventaja clara), expansión de red (similar a LwF pero con costo cuadrático), **bajar el learning rate de $\theta_s$** (insuficiente para preservar la tarea vieja), y **restricción L2 sobre los pesos** $\frac{1}{2}\lambda_o\lVert w - w_0\rVert^2$ (LwF la supera: regularizar la **salida** preserva mejor que regularizar parámetros individuales, porque muchos cambios pequeños de pesos pueden alterar mucho la salida).
- *Más allá de clasificación (Apéndice A):* se aplica LwF al tracking de objetos reemplazando el fine-tuning online de MD-Net; mejora levemente (0.383 vs 0.373 EAO) pero la diferencia **no es estadísticamente significativa** ($p = 0.70$).

## 6. Limitaciones reconocidas

- **Depende de que los datos nuevos activen representaciones útiles de las viejas.** La preservación funciona porque las imágenes nuevas, al pasar por la red vieja, producen salidas informativas sobre las clases viejas. Si esa supervisión indirecta es pobre, falla — el caso paradigmático es **ImageNet→MNIST**, donde los dígitos manuscritos no estimulan de manera significativa las representaciones de objetos naturales.
- **Domain shift fuerte degrada la preservación.** Cuando la distribución de las imágenes nuevas difiere mucho de la vieja (CUB vs Places365), el cambio en $\theta_s$ no puede ser compensado por LwF, porque las salidas registradas no cubren bien el dominio viejo. Incluso joint training sufre aquí, pero LwF más.
- **No garantiza monotonía en escenarios incrementales largos.** En la adición secuencial de tareas, la tarea vieja se erosiona progresivamente respecto a joint training (que sí tiene los datos).
- **Cota superior insuperable sin datos viejos.** Por construcción, joint training es el techo; LwF lo aproxima pero, con redes de alta capacidad (VGG), la brecha se ensancha.
- **Validado principalmente en clasificación de imágenes.** Los autores señalan como trabajo futuro extender a segmentación semántica, detección y dominios fuera de visión, así como mantener un set de imágenes no etiquetadas representativas de las tareas viejas — una idea que anticipa exactamente el *exemplar set* de iCaRL.

## 7. Impacto y conexión con el aprendizaje continuo

LwF fue **el trabajo que introdujo Knowledge Distillation en el continual learning** y, con ello, fundó la familia de los **métodos de regularización basados en distillation**. Su contribución conceptual perdurable es la idea de que **las salidas del modelo en su estado anterior son un sustituto barato y eficaz de los datos viejos**: no hay que guardar el dataset, basta con guardar (o recomputar) cómo respondía la red.

Esta idea es la base directa de **iCaRL** (Rebuffi et al. 2017, *Incremental Classifier and Representation Learning*), que combina la distillation loss de LwF con un **exemplar set** (un pequeño conjunto de ejemplos viejos seleccionados por *herding*) y clasificación por *nearest-mean-of-exemplars*. Donde LwF prescinde por completo de datos viejos, iCaRL admite un presupuesto de memoria pequeño, atacando justamente la limitación de domain shift de LwF (que ese exemplar set ancla la distribución vieja). La línea continúa en LwM, EWC (regularización sobre parámetros en vez de salidas) y toda la generación posterior de *class-incremental learning*.

En la Clase 32, LwF aparece explícitamente en el slide **"Learning Without Forgetting (LwF)"** como un **método de regularización que usa funciones de distillation** para mitigar el olvido catastrófico — contrapuesto a los métodos basados en *rehearsal/replay* (guardar o generar datos viejos) y a los basados en *expansión arquitectural* (Progressive Networks). Es el ejemplo canónico de "regularizar la salida del modelo viejo" dentro de la taxonomía del aprendizaje continuo.

**Enlaces internos del curso:**
- Fundamento transversal: [/fundamentos/aprendizaje-continuo](/fundamentos/aprendizaje-continuo)
- Clase: [/clases/clase-32](/clases/clase-32)
- Paper sucesor (rehearsal + distillation): [/papers/icarl-rebuffi-2017](/papers/icarl-rebuffi-2017)
