---
title: "Learning without Forgetting (2016)"
weight: 357
math: true
---

{{< paper-card
    title="Learning without Forgetting"
    authors="Zhizhong Li, Derek Hoiem"
    year="2016"
    venue="ECCV 2016 / TPAMI 2017"
    pdf="/papers/lwf-li-2016.pdf"
    arxiv="1606.09282" >}}
Paper de la Universidad de Illinois (Urbana-Champaign) que ataca un problema central del [aprendizaje continuo](/fundamentos/aprendizaje-continuo): **agregar nuevas tareas a una CNN ya entrenada sin acceso a los datos de las tareas viejas y sin degradar su desempeño en ellas**. La idea clave es usar **Knowledge Distillation**: antes de entrenar, se registran las salidas del modelo viejo sobre los datos **nuevos**, y durante el entrenamiento una pérdida de distillation mantiene esas salidas estables mientras una cross-entropy aprende la tarea nueva. Sin guardar un solo dato antiguo, LwF supera a fine-tuning y feature extraction, y se acerca a joint training (la cota superior). Es el puente que llevó la distillation al continual learning y la base directa de [iCaRL](/papers/icarl-rebuffi-2017).
{{< /paper-card >}}

---

## Contexto: agregar tareas sin datos viejos

El escenario motivador es nítido y práctico: un robot doméstico que llega con reconocedores genéricos y necesita aprender objetos específicos del sitio; un sistema de seguridad que detecta casco y chaleco al que hay que agregar la detección de calzado inadecuado. En todos estos casos los **datos legacy** pueden estar sin registrar, ser propietarios o simplemente demasiado engorrosos de reutilizar. El reto es aprender la tarea nueva compartiendo parámetros con las viejas **sin sufrir olvido catastrófico** (*catastrophic forgetting*, McCloskey & Cohen 1989; Goodfellow et al. 2013).

Formalmente, una CNN tiene tres bloques de parámetros: los **compartidos** $\theta_s$ (las capas convolucionales y fully-connected de, por ejemplo, AlexNet), los **específicos de tareas viejas** $\theta_o$ (la cabeza de salida de ImageNet) y los **de la tarea nueva** $\theta_n$ (un clasificador inicializado al azar). Conviene pensar $\theta_o$ y $\theta_n$ como clasificadores que operan sobre features parametrizadas por $\theta_s$. La pregunta es cómo aprender $\theta_n$ aprovechando $\theta_s$ ya entrenados, sin romper $\theta_o$.

## Las tres estrategias previas y su defecto

El paper disecciona las tres alternativas de la época, cada una con una falla fatal en este escenario:

| Estrategia | Qué hace | Defecto |
|---|---|---|
| **Feature Extraction** (Donahue 2014) | Congela $\theta_s$ y $\theta_o$; usa activaciones como features para entrenar solo $\theta_n$ | **No adapta** la representación compartida: rinde bajo en la tarea nueva. Preserva exactamente la vieja. |
| **Fine-tuning** (Girshick 2014) | Optimiza $\theta_s$ y $\theta_n$ para la tarea nueva, con $\theta_o$ fijo | **Olvida**: al mover $\theta_s$ sin guía sobre las predicciones viejas, degrada fuerte la tarea original. |
| **Joint Training** (Caruana 1997) | Optimiza todo conjuntamente, intercalando muestras de cada tarea | Mejor desempeño combinado (**cota superior**), pero **requiere los datos de todas las tareas** — imposible aquí. |

La variante **Fine-tuning FC** (congelar las convolucionales y afinar solo las fully-connected) es un compromiso, pero los experimentos muestran que aun así degrada la tarea nueva. Duplicar una red por tarea evita el olvido, pero hace que el tiempo de test crezca linealmente con el número de tareas.

El paper también ubica métodos contemporáneos: **A-LTM** (Furlanello et al. 2016), casi idéntico en método pero con conclusiones opuestas sobre la necesidad de los datos viejos; **Less Forgetting Learning / LFL** (Jung et al. 2016), que mantiene fija la capa final vieja en lugar de preservar sus salidas; y las **Progressive Neural Networks** (Rusu et al. 2016), que agregan nodos nuevos congelando los viejos pero expanden los parámetros de forma cuadrática. LwF se distingue de todos por prescindir por completo de datos antiguos y por no crecer en parámetros.

## La idea central: distillation sobre datos nuevos

La contribución de **Learning without Forgetting (LwF)** puede verse como un híbrido de Knowledge Distillation (Hinton, Vinyals & Dean 2014) y fine-tuning. La tesis:

> Usar **solo los datos de la tarea nueva** para optimizar simultáneamente (a) alta exactitud en la tarea nueva y (b) **preservación de las respuestas del modelo original en las tareas viejas**.

El procedimiento tiene dos momentos:

1. **Antes de entrenar.** Se pasan las imágenes de la tarea **nueva** $X_n$ por la red **vieja** y se registran sus salidas $Y_o = \text{CNN}(X_n, \theta_s, \theta_o)$ —las probabilidades sobre las clases viejas. Estas salidas registradas se convierten en *soft targets* (pseudo-etiquetas) para las cabezas viejas.

2. **Durante el entrenamiento.** Se optimiza una pérdida combinada que aprende la tarea nueva (cross-entropy contra las etiquetas reales $Y_n$) **mientras** una pérdida de distillation obliga a las salidas actuales sobre las cabezas viejas $\hat{Y}_o$ a mantenerse cerca de las $Y_o$ registradas.

La diferencia con joint training es exactamente esa: **joint training usa las imágenes y etiquetas de la tarea vieja; LwF las sustituye por las imágenes de la tarea nueva y las respuestas registradas del modelo viejo**. Esto elimina la necesidad de almacenar el dataset antiguo, conserva el beneficio de optimizar $\theta_s$ conjuntamente y ahorra cómputo: $X_n$ pasa una sola vez por las capas compartidas para ambas tareas.

## La pérdida combinada

LwF minimiza por SGD, sobre los datos nuevos, la suma de tres términos:

$$\theta_s^*, \theta_o^*, \theta_n^* \leftarrow \arg\min \Big[\, \lambda_o\, \mathcal{L}_{old}(Y_o, \hat{Y}_o) + \mathcal{L}_{new}(Y_n, \hat{Y}_n) + R(\hat\theta_s, \hat\theta_o, \hat\theta_n) \,\Big]$$

**Tarea nueva.** Cross-entropy multiclase estándar: $\mathcal{L}_{new}(y_n, \hat{y}_n) = -\,y_n \cdot \log \hat{y}_n$, con $\hat{y}_n$ la salida softmax y $y_n$ el one-hot de la etiqueta verdadera.

**Distillation para las cabezas viejas.** Se usa la **Knowledge Distillation loss** de Hinton et al. —una cross-entropy modificada que aumenta el peso de las probabilidades pequeñas:

$$\mathcal{L}_{old}(y_o, \hat{y}_o) = -\sum_{i=1}^{l} y_o'^{(i)} \log \hat{y}_o'^{(i)}, \qquad y_o'^{(i)} = \frac{(y_o^{(i)})^{1/T}}{\sum_j (y_o^{(j)})^{1/T}}$$

Fijar la **temperatura** $T > 1$ **suaviza** la distribución: aumenta el peso de los logits pequeños y empuja a la red a codificar mejor las **similitudes entre clases** (la *dark knowledge* de Hinton). El paper usa $T = 2$. Un hallazgo notable de las ablaciones: la distillation rinde solo *ligeramente* mejor que L1, L2 o cross-entropy simple —lo crucial es **restringir las salidas viejas a parecerse a las del original; la medida exacta de similitud no es determinante**.

**Balance y regularización.** $\lambda_o$ pesa el compromiso viejo↔nuevo (se fija en 1; subirlo favorece la tarea vieja) y $R$ es un weight decay de 0.0005. El entrenamiento usa primero un *warm-up* (congelar $\theta_s, \theta_o$, entrenar solo $\theta_n$) y luego un *joint-optimize* de todos los pesos.

## Experimentos y hallazgos

Se evalúa con **AlexNet** (verificado con VGG-16), partiendo de redes bien entrenadas en tareas viejas grandes —**ImageNet** (1000 clases) y **Places365** (365 escenas)— y agregando tareas nuevas moderadas: PASCAL VOC, CUB (aves fine-grained), MIT Scenes y MNIST (caso adversario deliberado).

- **En la tarea nueva:** LwF supera consistentemente a fine-tuning, LFL, fine-tuning FC y feature extraction. La ganancia sobre fine-tuning fue **inesperada** e indica que preservar las salidas viejas actúa como un **regularizador** que mejora la tarea nueva.
- **En la tarea vieja:** LwF supera por mucho a fine-tuning (que degrada fuerte), aunque suele quedar por debajo de feature extraction y fine-tuning FC.
- **Versus joint training:** LwF rinde **similar** con AlexNet —resultado positivo, ya que no usa datos viejos. Con VGG, joint training gana de forma más consistente: las redes de mayor capacidad se benefician más de tener los datos antiguos.

**Robustez.** Un *domain shift* fuerte (CUB sobre Places365) degrada más la tarea vieja porque las salidas registradas no cubren bien el dominio antiguo: las pérdidas son de 8.4% con fine-tuning, 3.8% con LwF y 1.5% con joint training. El caso **ImageNet→MNIST** falla porque los dígitos manuscritos dan una supervisión indirecta pobre de los objetos naturales. Variando el **tamaño del dataset nuevo** (3% a 100% de CUB), las observaciones se mantienen y la ventaja de LwF sobre fine-tuning tiende a crecer con más datos. En la **adición secuencial de tareas**, LwF se degrada más lento que fine-tuning, aunque queda por debajo de joint training a medida que se acumulan. Finalmente, frente a una restricción L2 sobre los pesos $\frac{1}{2}\lambda_o\lVert w - w_0\rVert^2$, LwF gana: **regularizar la salida preserva mejor que regularizar parámetros individuales**, porque muchos cambios pequeños de pesos pueden alterar mucho la salida —un contraste directo con la filosofía de EWC.

## Limitaciones reconocidas

- **Depende de que los datos nuevos activen representaciones útiles de las viejas.** Si la supervisión indirecta es pobre (ImageNet→MNIST), la preservación falla.
- **Domain shift fuerte degrada la preservación**, porque las salidas registradas no representan bien el dominio antiguo.
- **No garantiza monotonía** en escenarios incrementales largos: la tarea vieja se erosiona progresivamente respecto a joint training.
- **Cota superior insuperable sin datos viejos**: joint training es el techo, y con redes de alta capacidad (VGG) la brecha se ensancha.
- **Validado principalmente en clasificación de imágenes.** Los autores anticipan mantener un set de imágenes no etiquetadas de las tareas viejas —idea que prefigura exactamente el *exemplar set* de iCaRL.

## Por qué importa para la Clase 32

En la [Clase 32](/clases/clase-32) (Olvido Catastrófico), LwF es el ejemplo canónico de los **métodos de regularización basados en funciones de distillation** dentro del [aprendizaje continuo](/fundamentos/aprendizaje-continuo), contrapuesto a los métodos de *rehearsal/replay* (guardar o generar datos viejos) y a los de *expansión arquitectural* (Progressive Networks).

Su aporte conceptual perdurable: **las salidas del modelo en su estado anterior son un sustituto barato y eficaz de los datos viejos** —no hace falta guardar el dataset, basta guardar (o recomputar) cómo respondía la red. Esta idea es la base directa de **[iCaRL](/papers/icarl-rebuffi-2017)** (Rebuffi et al. 2017), que combina la distillation loss de LwF con un *exemplar set* pequeño para atacar justamente la limitación de domain shift de LwF. La línea continúa en LwM y en **[EWC](/papers/ewc-kirkpatrick-2017)** (Kirkpatrick et al. 2017), que en vez de regularizar la salida regulariza los **parámetros** según su importancia para las tareas viejas —el otro gran enfoque de regularización en continual learning.

## Notas y enlaces

- Preprint: arXiv:1606.09282v3 (14 de febrero de 2017), [arxiv.org/abs/1606.09282](https://arxiv.org/abs/1606.09282).
- Versión original: ECCV 2016, Springer, pp. 614-629. Versión extendida: IEEE TPAMI 2017.
- Afiliación: Department of Computer Science, University of Illinois at Urbana-Champaign.
