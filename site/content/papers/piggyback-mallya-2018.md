---
title: "Piggyback: Learning to Mask Weights (2018)"
weight: 362
math: true
---

{{< paper-card
    title="Piggyback: Adapting a Single Network to Multiple Tasks by Learning to Mask Weights"
    authors="Arun Mallya, Dillon Davis, Svetlana Lazebnik"
    year="2018"
    venue="ECCV 2018"
    pdf="/papers/piggyback-mallya-2018.pdf"
    arxiv="1801.06519" >}}
Paper de la Universidad de Illinois Urbana-Champaign que ataca el **olvido catastrófico** con un método de **arquitectura** (*parameter isolation*): se parte de una red pre-entrenada y **congelada** (por ejemplo VGG-16 o ResNet-50 sobre ImageNet) y, para cada tarea nueva, se aprende una **máscara binaria por peso** —un valor en {0, 1} para cada parámetro— que decide qué pesos del backbone se usan y cuáles se apagan. Como los pesos base **nunca se modifican, no hay olvido posible** por construcción. Cada máscara añade solo 1 bit por parámetro, lo que cuesta entre **32× y 64× menos** que guardar una red completa fine-tuneada. El truco de entrenamiento: máscaras de valor real binarizadas por umbral, con gradiente *straight-through*. Es el ejemplo arquetípico de la familia de métodos basada en máscaras, predecesor directo de [SupSup](/papers/supsup-wortsman-2020).
{{< /paper-card >}}

---

## Contexto

El método estándar para adaptar una red a una tarea nueva es el **fine-tuning**: partir de un modelo pre-entrenado y reentrenar sus pesos. Pero en el escenario incremental el fine-tuning sufre **olvido catastrófico** (French, 1999): el rendimiento en la tarea vieja se degrada a medida que se aprende la nueva, lo que obliga a guardar un modelo especializado por tarea. El objetivo del [aprendizaje continuo](/fundamentos/aprendizaje-continuo) es justamente ampliar las capacidades de una red existente *evitando* el olvido y *minimizando* los parámetros adicionales.

Piggyback se posiciona frente a las tres familias clásicas:

- **Regularización / proxy loss.** *Learning without Forgetting* (LwF, 2016) y *Elastic Weight Consolidation* (EWC, 2017) penalizan el cambio de los pesos importantes para tareas previas. Problema: como **todos los pesos pueden cambiar** en algún grado, no se puede garantizar de antemano cuánto se degradará lo anterior; un *domain shift* fuerte rompe esa protección.
- **Pruning + fine-tuning.** *PackNet* (Mallya & Lazebnik, 2017), del mismo primer autor, es el antecedente directo: poda pesos poco importantes, los reentrena para la tarea nueva y fija los importantes. Limitaciones que Piggyback supera: es **sensible al orden** de las tareas y el número total queda **limitado por el tamaño fijo de la red**.
- **Parámetros task-specific.** *Progressive Neural Networks* (2016) duplican la arquitectura por tarea; *Residual Adapters* (2017) añaden capas. Overhead grande comparado con 1 bit por peso.

La base técnica viene de la **binarización de redes** (BinaryConnect, 2015; Binarized Neural Networks, 2016): mantener pesos reales que pasan por un binarizador en el forward y propagar el gradiente sobre los pesos binarizados. La diferencia clave: Piggyback **no entrena una red cuantizada desde cero**, sino que aprende **máscaras** binarias sobre pesos de valor real **fijos**.

## La idea central: enmascarar en vez de modificar

La tesis del paper cabe en una pregunta: *¿hace falta cambiar los pesos para aprender una tarea nueva, o basta con enmascarar selectivamente algunos?* La respuesta es que basta enmascarar, y de ahí salen tres propiedades que ninguna combinación de trabajo previo lograba a la vez:

1. **Cero olvido por construcción.** Los pesos del backbone son constantes fijas; nunca se tocan. El rendimiento en cualquier tarea previa —incluida ImageNet— es **exactamente** el de la red original. No hay competencia entre tareas ni degradación progresiva, sin necesidad de proxy losses ni regularizadores.

2. **Overhead mínimo y agnosticismo al orden.** Cada tarea añade una máscara de 1 bit por parámetro (~3.12% del backbone, es decir 1/32). Como las máscaras son independientes, el orden de adición no afecta a ninguna y se pueden añadir tareas sin límite teórico (acotado solo por el almacenamiento de máscaras, no por la capacidad de la red).

3. **Expresividad sorprendente de una máscara binaria.** Aunque los pesos están fijos, enmascarar individualmente materializa una enorme variedad de filtros. Un vector denso `[0.1, 0.9, −0.5, 1]` puede dar lugar, vía máscaras, a `[0.1, 0, 0, 1]`, `[0, 0.9, −0.5, 0]` o `[0, 0.9, −0.5, 1]`. El espacio combinatorio de subconjuntos es lo bastante rico como para igualar —y a veces superar levemente— a una red fine-tuneada completa.

## Método: de máscara real a máscara binaria

El núcleo técnico es hacer diferenciable el aprendizaje de una máscara binaria. Se mantiene un conjunto de **pesos de máscara de valor real** $m_r$, se los pasa por una **función de umbral** para obtener la **máscara binaria** $m$, y esta se aplica elemento a elemento a los pesos del backbone.

Sea $W$ la matriz de pesos del backbone (fija) y $m_r$ una matriz de máscara real del mismo tamaño. La máscara binaria se obtiene por umbral duro:

$$
m_{ji} = \begin{cases} 1, & \text{si } (m_r)_{ji} \ge \tau \\ 0, & \text{en otro caso} \end{cases}
$$

con $\tau$ un umbral fijo. La salida de la capa enmascarada es $y = (W \odot m)\,x$, donde $\odot$ es el producto elemento a elemento: el bit $m_{ji}$ enciende o apaga el peso $w_{ji}$.

**El paso clave (straight-through).** La función de umbral es **no diferenciable** (gradiente cero casi en todas partes). Piggyback usa el estimador *straight-through*: en el backward, los gradientes calculados respecto a la máscara *binarizada* $m$ se usan como **estimador ruidoso** de los gradientes de la máscara *real* $m_r$ —se ignora el umbral y se deja pasar el gradiente directamente. El gradiente de la máscara resulta:

$$
\delta m = (\delta y \cdot x^T) \odot W
$$

Solo se actualiza $m_r$; $W$ permanece constante. Tras entrenar una tarea, los pesos reales $m_r$ se **descartan** y se guarda solo la máscara binaria $m$ —de ahí el 1 bit por parámetro.

**Detalles de optimización que el paper resalta:**

- Como $|\delta m| \propto |W|$, la magnitud del gradiente escala con la de los pesos del backbone (que varía por capa). La mejor receta encontrada: **inicializar $m_r$ con una constante** (1e-2) y usar **Adam** (que ajusta la escala por sí solo), con $\tau = $ 5e-3 — unos 2% de accuracy sobre las alternativas.
- Conviene inicializar las máscaras de modo que **todas valgan 1** al principio (reproduce exactamente el backbone base); arrancar con mitad 0 y mitad 1 da mal resultado.
- Por defecto **no** se entrenan biases ni parámetros de batch-norm por tarea; sí ayuda entrenar BN por tarea cuando hay *domain shift* grande.
- Probaron máscaras **ternarias** {−1, 0, 1} sin mejora significativa, así que se quedaron con binarias.

## Experimentos y resultados

**Datasets.** Seis tareas de clasificación de dominio variado: dos de gran escala (ImageNet, Places365), tres *fine-grained* (CUBS aves, Stanford Cars, Oxford Flowers) y dos con **domain shift fuerte** (WikiArt, pinturas; Sketch, dibujos en blanco y negro).

**Baselines:** *Classifier Only* (solo el clasificador lineal sobre el backbone congelado), *Individual Networks* (una red fine-tuneada por tarea, el techo de rendimiento y el mayor costo) y *PackNet*.

| Resultado (VGG-16) | Piggyback | Red individual |
|---|---|---|
| CUBS (error) | **20.99%** | 21.30% |
| Stanford Cars (error) | **11.87%** | 12.49% |
| Sketch (error) | **22.70%** | 23.54% |
| ImageNet (olvido) | **0% (idéntico)** | — |
| Almacenamiento 6 tareas | **621 MB** | 3.222 MB |

Piggyback iguala o **supera levemente** a las redes individuales en varias tareas (atribuido al efecto regularizador de restringir el cambio al enmascarado), con una fracción del almacenamiento y **cero olvido** en ImageNet. PackNet mejora sobre Classifier Only pero **empieza a sufrir al pasar de 3 tareas** y es sensible al orden (sus errores suben 4-7% según cuándo se añada la tarea).

**Escala y otras arquitecturas.** Al añadir Places365 (1.8M imágenes), Piggyback queda dentro de 0.36% del top-1 individual con menos épocas. El método funciona sin cambios sobre VGG-16+BN, ResNet-50 y DenseNet-121, pero **cuanto más profunda la red, mayor la brecha** (~2% en ResNet/DenseNet, hasta 4-5% en WikiArt): enmascarar filtros cambia la magnitud de las activaciones y choca con los parámetros de BN heredados de ImageNet —mitigable entrenando BN por tarea (de 28.67% a 25.92% de error en WikiArt, ~1 MB extra).

**Sparsity aprendida.** El porcentaje de pesos apagados crece con la dificultad de la tarea: Flowers ~4.5%, WikiArt ~34% (VGG-16). Las **capas bajas (conv1-3) se reutilizan casi intactas** (features genéricas) y los cambios se concentran en capas medias/altas (task-specific). En el reto **Visual Decathlon** Piggyback logra score competitivo con el estado del arte usando el menor número de parámetros adicionales, y se extiende a segmentación semántica (mIOU 61.41 vs 61.08 del fine-tuning).

## Limitaciones reconocidas

- **Requiere el task ID en inferencia.** La limitación clave: para clasificar una imagen hay que saber **a qué tarea pertenece** y aplicar la máscara correspondiente. Piggyback es por tanto **task-incremental**: asume un oráculo de tarea en test. No resuelve el escenario *class-incremental*, donde el modelo debe además inferir la tarea.
- **No hay transferencia entre tareas.** El conocimiento aprendido para la tarea A no fluye hacia la B; cada tarea explota el mismo backbone congelado de forma aislada. Solo las features de la inicialización (ImageNet) se reutilizan.
- **Dependencia crítica del backbone.** El rendimiento depende fuertemente de la calidad y diversidad de la inicialización. Sobre un backbone pobre, el techo del enmascarado baja.
- **Brecha en redes profundas y domain shift.** Aparece una diferencia de 2-5% frente a redes individuales en ResNet/DenseNet y tareas muy distintas, solo parcialmente mitigable con BN por tarea (a costo de tocar parámetros más allá de la máscara binaria pura).

## Por qué importa para la Clase 32

En el mapa de la [Clase 32 — Olvido catastrófico](/clases/clase-32), Piggyback ocupa el casillero de los **métodos de arquitectura** (*parameter isolation*), en contraste con los de **regularización** (EWC, SI, LwF) y los de **repetición/replay**. Su aportación conceptual perdurable: el olvido catastrófico se puede **eliminar por completo —no solo mitigar— si se renuncia a modificar los pesos compartidos**. En lugar de negociar entre tareas sobre un mismo conjunto de pesos, se le da a cada tarea su propio "circuito" mediante una máscara. El precio de esa garantía es el task ID en inferencia y la ausencia de transferencia.

El hilo natural desde aquí es **[SupSup](/papers/supsup-wortsman-2020)** (Wortsman et al., 2020): mientras Piggyback *aprende* la máscara binaria por backprop sobre pesos reales, SupSup *encuentra* "supermáscaras" sobre una red **aleatoria fija** y, crucialmente, propone **inferir el task ID** en test mediante superposición de máscaras y minimización de entropía —atacando justamente la limitación de Piggyback de necesitar el ID dado.

## Notas y enlaces

- Preprint: arXiv:1801.06519 — [arxiv.org/abs/1801.06519](https://arxiv.org/abs/1801.06519).
- Código (PyTorch): [github.com/arunmallya/piggyback](https://github.com/arunmallya/piggyback).
- Fundamento transversal: [Aprendizaje continuo](/fundamentos/aprendizaje-continuo).
- Clase: [Clase 32 — Olvido catastrófico](/clases/clase-32).
- Paper sucesor en la familia de máscaras: [SupSup — Supermasks in Superposition (2020)](/papers/supsup-wortsman-2020).
