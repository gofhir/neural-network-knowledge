---
title: "VLAD: Vector of Locally Aggregated Descriptors (2010)"
weight: 440
math: true
---

{{< paper-card
    title="Aggregating local descriptors into a compact image representation"
    authors="Hervé Jégou, Matthijs Douze, Cordelia Schmid, Patrick Pérez (INRIA, Technicolor)"
    year="2010"
    venue="CVPR 2010"
    pdf="/papers/vlad-jegou-2010.pdf" >}}
El paper que introduce **VLAD**, el método de agregación que treinta años de visión por computador le prestaron al reconocimiento de hablante. El problema es de búsqueda a gran escala: representar una imagen —o cualquier objeto descrito por un conjunto de descriptores locales de longitud variable— con un **vector de dimensión fija y pequeña**, sin perder poder discriminativo. La idea es engañosamente simple: aprender $k$ centroides sobre el espacio de descriptores, asignar cada descriptor a su centroide más cercano, y en vez de contar cuántos cayeron en cada uno —que es lo que hace *bag of features*— **acumular sus residuos** $x - c_i$. El resultado es un vector de $D = k \times d$ que codifica no solo *qué* estructuras aparecen sino *cómo se desvían* del prototipo. Con pocos centroides se obtiene un descriptor compacto que rinde como representaciones órdenes de magnitud más grandes. Es el fundamento conceptual de la parte de speaker recognition de la [Clase 41](/clases/clase-41).
{{< /paper-card >}}

---

## Contexto: tres restricciones simultáneas

La búsqueda de imágenes a gran escala tiene que satisfacer tres cosas a la vez: **precisión** de la búsqueda, **eficiencia** temporal y **uso de memoria**. Las representaciones dominantes de la época —*bag of features* sobre descriptores SIFT— alcanzaban precisión con vectores de hasta un millón de dimensiones, cuya dispersión permitía usar listas invertidas. Pero la memoria se volvía prohibitiva: no se puede indexar cien millones de imágenes si cada una ocupa kilobytes.

La alternativa obvia —bajar la dimensión— destruía la precisión. El paper busca el punto donde un vector **corto y denso** conserve el poder discriminativo de uno largo y disperso.

## Método: acumular residuos en vez de contar

El punto de partida es *bag of features*: se aprende un vocabulario de $k$ "palabras visuales" con k-means, cada descriptor local se asigna al centroide más cercano, y la imagen se representa por el **histograma** de asignaciones. Un vector de $k$ dimensiones que solo dice *cuántos* descriptores cayeron en cada celda.

VLAD conserva el vocabulario y cambia lo que se acumula:

$$v_{i,j} = \sum_{x \;:\; \text{NN}(x) = c_i} \big(x_j - c_{i,j}\big)$$

Para cada centroide $c_i$ se suman las **diferencias** entre los descriptores asignados y el centroide, componente a componente. El vector resultante se aplana y se normaliza en L2.

La representación pasa de $k$ dimensiones a $D = k \times d$, con $d$ la dimensión del descriptor local. Con SIFT ($d = 128$) y $k = 16$ centroides, son 2 048 dimensiones — y esos 16 centroides bastan para resultados excelentes, frente a los cientos de miles de palabras visuales que necesitaba BOF.

{{< concept-alert type="clave" >}}
**El cambio conceptual es de conteo a desviación.** Un histograma dice "hay tres descriptores cerca del prototipo *esquina*". VLAD dice "hay tres descriptores cerca del prototipo *esquina*, y en promedio están desplazados así". La primera información es de ocupación; la segunda es de **estadística de primer orden** dentro de cada celda, y es mucho más discriminativa por dimensión gastada.

La misma distinción es la que separa el *temporal average pooling* del NetVLAD en reconocimiento de hablante: promediar frames dice dónde está el centro de masa; acumular residuos por cluster dice cómo se distribuye la voz respecto de prototipos aprendidos. En [Xie et al. (2019)](/papers/utterance-level-xie-2019) esa diferencia vale **7 puntos de EER**.
{{< /concept-alert >}}

Dos observaciones estructurales que el paper aprovecha:

- **Los vectores VLAD son dispersos y muy estructurados.** La mayor parte de la energía se concentra en pocos clusters, y dentro de cada bloque se conserva la geometría del descriptor original.
- **Esa estructura la captura bien un PCA.** El paper reduce la dimensión con PCA y luego cuantiza el vector con un cuantizador producto, optimizando conjuntamente reducción e indexación. Con eso, buscar en una base de **10 millones de imágenes toma unos 50 ms**.

## Resultados

VLAD con pocas decenas de centroides iguala o supera a BOF con vocabularios órdenes de magnitud mayores, usando una fracción de la memoria. Combinado con PCA y cuantización producto, permite representar cada imagen con **decenas de bytes** manteniendo precisión utilizable — el régimen en el que la búsqueda a escala de cientos de millones se vuelve posible.

## Limitaciones

- **El vocabulario no se aprende para la tarea.** Los centroides salen de k-means sobre los descriptores, con un criterio de reconstrucción, no de discriminación. Nada garantiza que las celdas separen lo que interesa separar — el problema que [NetVLAD](/papers/netvlad-arandjelovic-2016) resuelve haciendo los centroides entrenables.
- **La asignación es dura.** Cada descriptor va a un solo centroide, con `argmin`. Un descriptor equidistante de dos centroides se asigna arbitrariamente a uno, y pequeñas perturbaciones cambian el resultado de forma discontinua. Esto además **impide el gradiente**: `argmin` no es derivable.
- **Depende de descriptores locales diseñados a mano** (SIFT), que no se optimizan con el resto del sistema.
- **La dimensión crece con $k \times d$**, lo que obliga a un paso de reducción posterior.

## Por qué importa para la Clase 41

La clase dedica seis diapositivas a construir la intuición de VLAD antes de nombrarlo: un espacio de features con centroides que representan características básicas de la voz —*"voz grave"*, *"voz aguda"*, *"voz siseante"*—, la proyección de una señal en ese espacio y una representación nueva basada en el **nivel de pertenencia** a cada centroide. Después enumera los cinco pasos del cómputo exactamente como aparecen acá: proyectar, aprender $k$ centroides, asignar con `argmin`, calcular $x - c_i$, agregar residuos.

Y cierra con la pregunta que motiva todo lo que sigue: *"In this process we need to find $x_i$, $c_k$ y $a_k(x_i)$. **Is it possible to learn all of them end-to-end?**"*. La respuesta es [NetVLAD](/papers/netvlad-arandjelovic-2016), y su aplicación al habla es [Xie et al. (2019)](/papers/utterance-level-xie-2019).

El mecanismo desarrollado de forma autónoma, con el paso de asignación dura a blanda y su uso como capa de red, está en [Agregación VLAD](/fundamentos/agregacion-vlad).
