---
title: "Comparación MAML vs Prototypical y teoría"
weight: 6
math: true
---

> **El cierre del laboratorio.** Construimos dos sistemas few-shot que resuelven *el mismo problema* con filosofías opuestas: MAML (optimization-based) aprende un buen **punto de partida** y se adapta con gradientes; Prototypical (metric-based) aprende un buen **espacio de embeddings** y clasifica por distancia. Aquí consolidamos los números reales del lab, contrastamos paradigmas y respondemos las tres preguntas teóricas de la Actividad 7.

## El gran arco del laboratorio

MAML y Prototypical Networks son las dos grandes familias del meta-aprendizaje moderno. Una optimiza la **inicialización de parámetros** para que pocos pasos de gradiente basten; la otra optimiza un **espacio métrico** donde clasificar es medir distancias. A lo largo del laboratorio entrenamos ambas sobre los mismos benchmarks, y los resultados cuentan una historia más interesante que "una le gana a la otra".

| Problema | MAML (mejor) | Prototypical (mejor) |
|---|---|---|
| Omniglot 4-way 1-shot | 0.877 (400 iters, 5 hiperparámetros) | **0.934** (CNN, 80 épocas) |
| Mini-ImageNet 4-way 1-shot | 0.324 | **0.377** |
| Mini-ImageNet 4-way 5-shot | 0.491 | **0.632** |

En los tres escenarios Prototypical termina arriba, pero el *motivo* por el que gana cambia radicalmente según el dataset. De estos números salen cuatro conclusiones transversales:

1. **En datos SIMPLES (Omniglot) el cuello de botella es la OPTIMIZACIÓN.** Las clases de Omniglot (caracteres manuscritos en blanco y negro) son fáciles de separar; lo difícil es *llegar* a una solución con pocos datos. En MAML eso se traduce en que mandan las **iteraciones** de meta-entrenamiento: más pasos de adaptación y más iteraciones suben la accuracy. ProtoNet, en cambio, llega más alto (0.934 vs 0.877) y de forma más simple, porque su clasificación por prototipos no necesita resolver un problema de optimización en test.

2. **En datos COMPLEJOS (Mini-ImageNet) el cuello de botella es la INFORMACIÓN.** Las imágenes naturales son ambiguas y de alta varianza intra-clase. Aquí lo único que da poder real son los **shots**: pasar de 1-shot a 5-shot sube ProtoNet de 0.377 a 0.632 (+25 puntos). Lo demás resultó ser un espejismo: más optimización no alcanza el techo, **reducir el número de clases** infla la accuracy cruda sin mejorar la calidad del modelo (es más fácil acertar entre 2 que entre 4), y backbones más grandes **no ayudaron** — cambiar a ResNet12 no movió la aguja porque el problema no era capacidad del extractor sino escasez de evidencia por clase.

3. **ProtoNet es ROBUSTO al número de clases (ways) en Omniglot.** Su accuracy cruda se mantiene casi constante al pasar de 2 a 8 ways, mientras que MAML se **desploma** al añadir clases. Pero esa robustez no es magia del método: depende de la **calidad de los embeddings**. Cuando los embeddings son excelentes (Omniglot), los prototipos quedan bien separados y agregar clases casi no estorba. Cuando los embeddings son pobres (Mini-ImageNet 1-shot), esa robustez **desaparece** — los centroides se solapan y sumar clases sí degrada el desempeño.

4. **El overfitting depende del DATASET, no del método.** Observamos sobreajuste en Mini-ImageNet (la brecha entre train y validación crece con las épocas) pero **no** en Omniglot, y eso ocurre tanto en MAML como en ProtoNet. La lección: la tendencia a sobreajustar es una propiedad de la **dificultad y tamaño del dataset**, no del paradigma de meta-aprendizaje que elijas.

## Tabla de contraste de paradigmas

Más allá de los números, conviene fijar las diferencias estructurales entre ambos enfoques. Esta tabla resume *en qué son distintos* MAML y Prototypical:

| Dimensión | MAML (optimization-based) | Prototypical (metric-based) |
|---|---|---|
| **Qué aprende** | Una inicialización $\theta$ de los parámetros | Un espacio de embeddings $f_\phi$ |
| **Adaptación en test** | Unos pocos pasos de gradiente sobre el support | Ninguna — solo calcula prototipos y distancias |
| **Estructura de entrenamiento** | Optimización **binivel** (inner + outer loop) | Supervisión episódica normal (un solo nivel) |
| **`clone()` / 2.º orden** | Sí — clona pesos y deriva a través de la adaptación | No — un solo `forward`, sin grafos anidados |
| **Costo en test** | `forward` + `backward` (gradientes en inferencia) | Solo `forward` + cálculo de distancias |
| **Salida del modelo** | `WAYS` logits (un clasificador adaptado) | Un vector embedding (clasificación por cercanía) |
| **Robustez al WAYS** | Frágil — se degrada al añadir clases | Robusta, **si** los embeddings son buenos |

La asimetría más reveladora está en la fila de adaptación: MAML **mueve los parámetros** en test, mientras ProtoNet los deja fijos y solo reubica la geometría de la decisión. Esa es la razón de fondo por la que MAML paga el costo de los gradientes de segundo orden y ProtoNet no.

## Actividad 7 — Pregunta 1: MAML vs fine-tuning tradicional

**¿En qué se diferencia MAML del fine-tuning tradicional?**

La diferencia clave no está en la arquitectura ni en el dato, sino en **qué se optimiza durante el meta-entrenamiento**.

El fine-tuning tradicional pre-entrena un modelo minimizando la pérdida $L(\theta)$ sobre una tarea grande, **sin saber que después será adaptado** a otra tarea. El pre-entrenamiento y la futura adaptación son dos etapas desconectadas: la primera busca una buena solución para *su* problema, y solo después se reutilizan esos pesos como inicialización casual.

MAML, en cambio, optimiza la pérdida **después de adaptar**. Su objetivo es

$$
\min_{\theta} \; \sum_{\mathcal{T}_i} L_{\mathcal{T}_i}\big(\theta_i'\big), \qquad \theta_i' = \theta - \alpha \, \nabla_\theta L_{\text{support}}^{\mathcal{T}_i}(\theta).
$$

Es decir, evalúa la pérdida en el conjunto de *query* usando los parámetros adaptados $\theta'$, y el **meta-gradiente** se propaga **a través** de los pasos de adaptación. Como $\theta'$ depende de $\theta$ vía un paso de gradiente, derivar $L(\theta')$ respecto de $\theta$ involucra la derivada de un gradiente — el término de **segundo orden**:

$$
\nabla_\theta L(\theta') = \nabla_{\theta'} L(\theta') \cdot \big(I - \alpha \, \nabla^2_\theta L_{\text{support}}(\theta)\big).
$$

En una frase: **el fine-tuning aprende una buena solución; MAML aprende un buen punto de partida para encontrar soluciones rápido.**

Los beneficios concretos de esta diferencia:

- **Adaptación con muy pocos datos y pasos (few-shot) sin sobreajustar.** Como $\theta$ ya está colocado en un punto desde el cual unos pocos pasos llevan a buenas soluciones, no hace falta mucho dato para adaptar.
- **Mejor generalización a tareas no vistas.** $\theta$ es un **punto de compromiso** de toda la distribución de tareas: no es óptimo para ninguna en particular, sino cercano al óptimo de *todas*.
- **Es model-agnostic.** No impone arquitectura ni función de pérdida; sirve para clasificación, regresión o RL — de ahí el nombre *Model-Agnostic Meta-Learning*.

## Actividad 7 — Pregunta 2: Siamese → Matching → Prototypical

**Describe la evolución de las redes métricas para few-shot learning.**

Las tres arquitecturas comparten una misma idea — *aprender un espacio de embeddings donde clasificar sea medir distancias* — pero cada una **simplifica y mejora** a la anterior.

- **Siamese Networks (Koch et al., 2015).** Dos redes **gemelas con pesos compartidos** procesan dos imágenes y producen embeddings; una capa final mide la **similitud** entre ambos vectores. Se entrenan por **pares** como un problema de verificación ("¿son de la misma clase?"). En test, se clasifica por el **vecino más cercano** aprendido: se compara la query contra cada ejemplo del support y se asigna la clase del más similar.

- **Matching Networks (Vinyals et al., 2016).** Tres mejoras sobre Siamese:
  1. **Entrenamiento episódico** que imita las condiciones de test — el principio *"test and train conditions must match"*. Se entrena con mini-tareas N-way K-shot, exactamente el formato en que se evaluará.
  2. **Clasificación por attention / weighted nearest neighbor**: en vez de quedarse solo con el vecino más cercano, aplica un **softmax sobre la similitud coseno** con *todos* los ejemplos del support, ponderando a cada uno.
  3. **Full Context Embeddings (FCE)**: un **LSTM bidireccional** hace que el embedding de cada ejemplo dependa del **support set completo**, no solo de la imagen aislada.

- **Prototypical Networks (Snell et al., 2017).** Tres mejoras sobre Matching:
  1. **Prototipos = promedio de embeddings por clase.** En lugar de comparar contra cada ejemplo, se calcula un centroide por clase: $c_k = \frac{1}{|S_k|}\sum_{(x_i,y_i)\in S_k} f_\phi(x_i)$. Esto **resuelve la ambigüedad de Matching con $K>1$** (¿contra cuál de los K ejemplos comparar?) y es más eficiente.
  2. **Distancia euclidiana en vez de coseno.** No es un detalle arbitrario: la distancia euclidiana al cuadrado es una **divergencia de Bregman**, y para esa familia el **promedio es el estimador óptimo** del prototipo. Es decir, usar centroides solo está teóricamente justificado con esta distancia.
  3. **Más simple y rinde igual o mejor.** Elimina el LSTM, las FCE y la attention compleja de Matching, quedándose con `embeddings → centroides → softmax sobre distancias`.

El hilo conductor: los tres aprenden un **espacio de embeddings** donde clasificar = medir distancias; cada uno **descarta complejidad** del anterior sin perder (y ganando) desempeño. Prototypical es el destilado final de esa línea evolutiva.

## Actividad 7 — Pregunta 3: ¿Cuándo elegir cada uno?

La síntesis práctica de todo el laboratorio. **MAML** brilla cuando las tareas exigen *adaptación real* del modelo — cuando la frontera de decisión cambia de forma entre tareas y un solo espacio métrico fijo no basta — y cuando puedes pagar el costo de gradientes en inferencia. **Prototypical** brilla cuando el problema es separable en un buen espacio de embeddings, cuando necesitas inferencia barata (un solo `forward`) y cuando quieres **escalar el número de clases** sin reentrenar nada. Para la mayoría de los problemas de clasificación few-shot "limpia", ProtoNet es el punto de partida sensato: más simple, más barato y, como vimos, frecuentemente más preciso.

## Conexión con FHIR / record linkage

Prototypical Networks **es**, conceptualmente, el paradigma del **bi-encoder** para *patient matching*: un encoder mapea cada registro a un vector, el **prototipo** es el centroide de los registros de una misma entidad-paciente, y la **clasificación por distancia** es exactamente lo que hace el *blocking* y el *matching*.

La propiedad que más importa aquí es la **robustez al número de clases** del enfoque metric-based. En record linkage el "número de clases" es el número de **entidades-paciente distintas**, que puede llegar a millones. Un clasificador de salida fija — con una neurona por entidad, como haría un modelo de clasificación clásico (o el clasificador adaptado de MAML) — sería **inviable**: tendrías que reentrenar cada vez que aparece un paciente nuevo, y la capa final crecería sin control. El metric-based esquiva ese problema por completo: agregar una entidad solo significa **calcular un centroide más** en el mismo espacio de embeddings, sin tocar los pesos. Esa es la razón de fondo por la que el bi-encoder escala a producción y el clasificador denso no.

## Enlaces

- **Páginas:** [MAML: fundamentos](maml-fundamentos) · [Experimentos MAML](experimentos-maml) · [Prototypical: fundamentos](prototypical-fundamentos) · [Experimentos Prototypical](experimentos-prototypical)
- **Papers:** [MAML (Finn 2017)](/papers/maml-finn-2017) · [Prototypical Networks (Snell 2017)](/papers/prototypical-networks-snell-2017) · [Matching Networks (Vinyals 2016)](/papers/matching-networks-vinyals-2016) · [Siamese Networks (Koch 2015)](/papers/siamese-networks-koch-2015)
- **Fundamentos:** [Meta-aprendizaje](/fundamentos/meta-aprendizaje) · [Metric Learning](/fundamentos/metric-learning)
- **Clase:** [Clase 26 - Meta-aprendizaje](/clases/clase-26) · [Profundización](/clases/clase-26/profundizacion)
