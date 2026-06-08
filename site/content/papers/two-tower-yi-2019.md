---
title: "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations"
weight: 256
math: true
---

{{< paper-card
    title="Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations"
    authors="Yi, Yang, Hong, Cheng, Heldt, Kumthekar, Zhao, Wei, Chi"
    year="2019"
    venue="RecSys 2019 (Google)"
    pdf="/papers/two-tower-yi-2019.pdf" >}}
Este paper de Google **formaliza el modelo two-tower** para retrieval a gran escala: una torre codifica `{usuario, contexto}` y otra el `{ítem}`, y el score es el **producto punto** de ambos embeddings. Su entrenamiento usa **in-batch negatives** (los demás ítems del minibatch como negativos), pero ese atajo introduce un **sesgo de muestreo** que sobre-penaliza a los ítems populares. La contribución central es **corregirlo con log-Q correction**, descontando de cada logit el logaritmo de la frecuencia del ítem, estimada **online en streaming** sin vocabulario fijo. Es la **referencia canónica del retrieval neuronal / candidate generation** moderno y la base teórica del case study de la [/clases/clase-25](/clases/clase-25).
{{< /paper-card >}}

---

## Contexto

Los sistemas de recomendación industriales conectan miles de millones de usuarios con catálogos de millones a miles de millones de ítems, bajo latencias estrictas. La receta dominante es tratar la recomendación como un problema de **retrieval-and-ranking en dos fases**: un modelo de *retrieval* (también llamado *candidate generation* o *nomination*) escalable recupera primero una pequeña fracción de ítems desde el corpus completo, y un modelo de *ranking* más pesado los reordena. Este paper se concentra en el **retrieval**.

Dado un triplet `{usuario, contexto, ítem}`, la solución estándar es aprender representaciones separadas para la consulta `{usuario, contexto}` y para el `{ítem}`, y comparar con una función simple — el **producto punto**. Separar las representaciones es lo que permite, en inferencia, **precalcular los embeddings de todos los ítems** y resolver el retrieval como una búsqueda de máximo producto interno (MIPS) en tiempo sublineal sobre el corpus.

El aprendizaje es difícil por dos razones que el paper enfatiza: el corpus puede ser enorme, y el feedback es **muy esparso** para la mayoría de los ítems, con predicciones de alta varianza en el *long-tail* y problemas de *cold-start*. La distribución de ítems sigue una **ley de potencias**: pocos ítems populares acumulan casi todas las interacciones. La factorización matricial clásica (incluso con features de contenido) solo captura interacciones de segundo orden; de ahí el giro hacia redes profundas, capaces de codificar estados de usuario y features de contenido heterogéneos (IDs esparsos, densos, texto, imagen) en embeddings de baja dimensión. Ver [/fundamentos/recommender-systems](/fundamentos/recommender-systems).

## Ideas principales

### Two-tower, batch softmax y corrección log-Q

El modelo aprende dos funciones de embedding parametrizadas — dos redes profundas que comparten parámetros `θ` —, una para la query y otra para el ítem, y el score es su producto interno:

$$ s(x, y) = \langle u(x, \theta),\, v(y, \theta) \rangle $$

A diferencia de Neural Collaborative Filtering, que **concatena** usuario e ítem y los pasa por una red conjunta (impidiendo precalcular ítems), el two-tower mantiene las torres **separadas hasta el producto punto final**, habilitando MIPS en inferencia. Ver [/fundamentos/two-tower-retrieval](/fundamentos/two-tower-retrieval).

El retrieval se plantea como **clasificación multiclase**: dada `x`, predecir el ítem `y` correcto entre `M` candidatos, con probabilidad softmax sobre los `M` ítems. Cuando `M` es de millones, calcular el denominador (la función de partición) es inviable. Como la torre del ítem es profunda, tampoco es eficiente muestrear muchos negativos desde un vocabulario fijo (cada uno exige un *forward pass*). La solución es usar **in-batch negatives**: para cada par `(x_i, y_i)` del minibatch de tamaño `B`, los demás ítems del batch sirven de negativos, reutilizando embeddings ya calculados. El **batch softmax** queda:

$$ P_B(y_i \mid x_i; \theta) = \frac{e^{s(x_i, y_i)}}{\sum_{j \in [B]} e^{s(x_i, y_j)}} $$

El problema es el **sesgo de muestreo**: como los ítems in-batch se muestrean según el tráfico (power-law), los **populares aparecen como negativos casi siempre y quedan sobre-penalizados**, distorsionando el aprendizaje respecto al softmax completo. La contribución central es la **corrección log-Q**, importada del sampled softmax adaptativo: a cada logit se le resta el log de la probabilidad de muestreo del ítem,

$$ s^{c}(x_i, y_j) = s(x_i, y_j) - \log(p_j) $$

donde `p_j` es la probabilidad de que el ítem `j` aparezca en un batch aleatorio. Esto **descuenta la ventaja artificial de los ítems frecuentes** y recupera un estimador insesgado del softmax full. La pérdida es la log-verosimilitud ponderada por una **recompensa** `r_i` (que captura grados de engagement, p. ej. watch time):

$$ L_B(\theta) = -\frac{1}{B} \sum_{i \in [B]} r_i \cdot \log\big( P_B^{c}(y_i \mid x_i; \theta) \big) $$

### Estimación de frecuencia en streaming

El reto práctico es que en datos **streaming** (el catálogo de YouTube cambia constantemente) no hay vocabulario fijo ni distribución conocida, así que `p_j` debe estimarse **online y de forma distribuida**. La idea: en vez de estimar `p` directamente, estimar `δ`, el **número promedio de pasos (global steps) entre dos apariciones consecutivas** del ítem, con `p = 1/δ`. El *global step* sincroniza implícitamente a múltiples workers y permite una **media móvil** adaptativa.

Como no hay vocabulario fijo, se usan **arreglos de hash** `A` y `B` de tamaño `H` con función `h`. `A[h(y)]` guarda el último paso en que apareció `y`; `B[h(y)]` la estimación de `δ`. Al aparecer `y` en el paso `t`:

$$ B[h(y)] \leftarrow (1-\alpha)\, B[h(y)] + \alpha \,(t - A[h(y)]), \qquad \hat{p} = 1/B[h(y)] $$

Esto es **SGD con learning rate fijo `α`** para aprender la media del gap entre apariciones. El paper prueba (Proposición 4.1) que el **sesgo → 0** cuando `t → ∞` y que una inicialización ideal lo anula en cada paso; la **varianza** queda acotada, con un trade-off claro: `α` alto = más adaptativo a cambios de distribución pero más varianza residual.

Para mitigar la **sobre-estimación de frecuencia por colisiones de hash**, el Algoritmo 3 usa, al estilo del **count-min sketch**, `m` funciones de hash independientes y toma el **máximo** de las `m` estimaciones de `δ` en inferencia.

### Normalización y temperatura

Dos detalles empíricos clave: **L2-normalization** de ambos embeddings (tras lo cual el producto punto es similitud coseno) mejora estabilidad y entrenabilidad; y una **temperatura** `τ` que afila las predicciones, `s(x,y) = ⟨u(x), v(y)⟩/τ`. El paper muestra que `τ` debe **tunearse con cuidado** — su efecto sobre el recall es notable.

## Resultados experimentales

**Simulación.** Con distribución power-law conmutada a mitad de camino, las curvas confirman la teoría: `α` alto adapta más rápido pero con más varianza; usar **más funciones de hash reduce el error** incluso a igualdad de parámetros.

**Wikipedia (5.3M páginas, 430M enlaces)**, retrieval de páginas destino, Recall@K contra el corpus completo. La corrección supera a la versión sin corregir **por amplio margen**:

| Método | R@10 | R@50 | R@100 | R@300 |
|---|---|---|---|---|
| mse-gramian | 0.0432 | 0.1326 | 0.2027 | 0.3530 |
| plain-sfx τ=0.07 | 0.0643 | 0.2423 | 0.3746 | 0.5991 |
| **correct-sfx τ=0.07** | **0.1065** | 0.3079 | 0.4664 | 0.7234 |
| **correct-sfx τ=0.05** | 0.0987 | **0.3202** | **0.4835** | **0.7413** |

R@10 casi se duplica (0.0643 → 0.1065) con la corrección; `τ=0.14` degrada, confirmando la sensibilidad a la temperatura.

**YouTube (offline + live).** Sistema real: torre de query (seed video + historial de watch como bag-of-words promediado) y torre de candidato, DNNs `[1024, 512, 128]`, índice de ~10M videos reconstruido cada pocas horas para MIPS, **sequential training** que consume los datos por días para adaptarse al stream.

Offline (Recall@K): `correct-sfx τ=0.05` logra **R@50 = 0.5322** frente a 0.4586 de `plain-sfx` y 0.1338 de mse-gramian. En el **A/B test live**, la corrección **casi duplica la mejora de engagement** sobre producción: **+0.37%** vs. +0.20% del softmax sin corregir — significativo a escala YouTube.

## Limitaciones reconocibles

- Los datos de YouTube son propietarios; las cifras live son mejoras relativas pequeñas (aunque significativas a escala).
- La corrección ataca la **frecuencia de muestreo**, no el sesgo de selección del sistema que generó el log (*feedback loop*) ni los *false negatives* (un ítem in-batch que sí le gustaría al usuario, tratado como negativo).
- Las **colisiones de hash** se mitigan pero no se eliminan; aportan error residual.
- `τ` y `α` exigen tuning cuidadoso, sin receta cerrada.
- La métrica de recompensa offline se simplifica a `r_i = 1` porque definir una métrica offline para recompensa continua "no es obvio".
- El **MIPS aproximado** introduce error de recuperación que el paper deliberadamente no analiza.

## Por qué importa hoy

Este paper es la **referencia canónica del two-tower / dual-encoder para candidate generation** a escala industrial. Estandarizó tres prácticas hoy ubicuas: **in-batch negatives + corrección log-Q**, **L2-normalization + temperatura** sobre el producto punto, y **serving vía MIPS** sobre embeddings de ítems precalculados. Conecta hacia atrás con [/papers/dssm-huang-2013](/papers/dssm-huang-2013) — el ancestro del dual-encoder — y el modelo de candidate generation de YouTube (Covington et al. 2016), y hacia adelante con el retrieval denso para QA, TensorFlow Recommenders (que implementa directamente este modelo y su pérdida) y prácticamente todo retrieval semántico moderno basado en embeddings. La estimación de la distribución de muestreo en streaming para corregir el sesgo del batch softmax sigue siendo el estado del arte para entrenar retrievers sobre catálogos dinámicos.

## Conexión con la Clase 25

La [/clases/clase-25](/clases/clase-25) es un *case study* de un recsys **multimodal** cuya arquitectura es, en esencia, exactamente este two-tower. Allí **una torre representa el pin** combinando una CNN (imagen) y un BERT (texto) fusionados en una capa FC para producir el embedding del ítem — es decir, la **torre de candidato `v(y)`** de este paper, con features de contenido multimodales en lugar de los IDs+densos de YouTube. La inferencia del case study compara el pin contra el usuario por **mínima distancia**, lo que es equivalente al **máximo producto punto** de aquí: con embeddings L2-normalizados, minimizar distancia euclidiana ⟺ maximizar similitud coseno ⟺ maximizar $\langle u(x), v(y) \rangle$.

Así, el scoring por **mínima distancia del case study es precisamente el dot product** $s(x,y) = \langle u(x), v(y)\rangle$ formalizado aquí; el retrieval del pin más cercano usa el mismo **MIPS aproximado**; y el entrenamiento contrastivo con in-batch negatives enfrenta exactamente el sesgo por popularidad que la corrección log-Q resuelve. Este paper aporta la **formalización rigurosa** de la arquitectura que la Clase 25 presenta de forma aplicada.

## Notas y enlaces

- **PDF:** [two-tower-yi-2019.pdf](/papers/two-tower-yi-2019.pdf) · RecSys 2019 · DOI 10.1145/3298689.3346996
- **Clase relacionada:** [/clases/clase-25](/clases/clase-25)
- **Fundamentos:** [/fundamentos/recommender-systems](/fundamentos/recommender-systems) · [/fundamentos/two-tower-retrieval](/fundamentos/two-tower-retrieval)
- **Paper relacionado:** [/papers/dssm-huang-2013](/papers/dssm-huang-2013) (ancestro del dual-encoder)
