---
title: "NetVLAD (2016)"
weight: 441
math: true
---

{{< paper-card
    title="NetVLAD: CNN architecture for weakly supervised place recognition"
    authors="Relja Arandjelović, Petr Gronat, Akihiko Torii, Tomas Pajdla, Josef Sivic (INRIA, Tokyo Tech, CTU Prague)"
    year="2016"
    venue="CVPR 2016 / arXiv:1511.07247"
    pdf="/papers/netvlad-arandjelovic-2016.pdf" >}}
El paper que convierte a [VLAD](/papers/vlad-jegou-2010) en una **capa de red neuronal**. El obstáculo era concreto: VLAD asigna cada descriptor a su centroide más cercano con un `argmin`, que no es derivable — no hay gradiente que propagar, así que ni los centroides ni los descriptores pueden aprenderse para la tarea. La solución es reemplazar la asignación dura por una **asignación blanda** vía softmax, con pesos entrenables. Con eso la capa se vuelve enchufable en cualquier CNN y **todo** se aprende de punta a punta: el extractor de features, los centroides del diccionario y la función de pertenencia. Los autores lo demuestran en reconocimiento de lugares —identificar dónde fue tomada una foto— con una pérdida de ranking débilmente supervisada construida sobre Google Street View Time Machine. Es la pieza que la [Clase 41](/clases/clase-41) necesita para responder su propia pregunta: *"¿es posible aprender todo esto end-to-end?"*
{{< /paper-card >}}

---

## Contexto: dos mundos que no se hablaban

Hacia 2015 había dos familias de representaciones para búsqueda visual. Por un lado, los descriptores agregados clásicos —[VLAD](/papers/vlad-jegou-2010), Fisher Vectors— con excelente comportamiento en recuperación pero construidos sobre features diseñados a mano y vocabularios aprendidos con un criterio de reconstrucción, no de discriminación. Por el otro, las CNN entrenadas end-to-end, que aprendían features excelentes pero cuya agregación era un pooling ingenuo: promedio o máximo sobre el mapa de activaciones.

La combinación obvia —CNN para los descriptores locales, VLAD para agregarlos— se venía usando, pero **desconectada**: la CNN se entrenaba para clasificar, VLAD se calculaba encima, y el gradiente no cruzaba de uno a otro. El eslabón roto es el `argmin`.

## Método: de argmin a softmax

En VLAD clásico, el aporte de un descriptor $x_i$ al cluster $k$ es su residuo, ponderado por una pertenencia binaria:

$$V(j,k) = \sum_{i=1}^{N} a_k(x_i)\,\big(x_i(j) - c_k(j)\big), \qquad a_k(x_i) \in \{0, 1\}$$

con $a_k(x_i) = 1$ si y solo si $c_k$ es el centroide más cercano. Esa función indicadora es una escalera: constante a trozos, con derivada cero en casi todas partes y no definida en las fronteras.

NetVLAD la reemplaza por una **pertenencia blanda**:

$$\bar{a}_k(x_i) = \frac{e^{\,w_k^\top x_i + b_k}}{\sum_{k'} e^{\,w_{k'}^\top x_i + b_{k'}}}$$

Un softmax sobre puntajes lineales. Tres consecuencias:

1. **Es derivable en todas partes**, así que el gradiente fluye hacia los descriptores y hacia los parámetros.
2. Cada descriptor contribuye a **todos** los clusters, con peso decreciente según la distancia — no hay decisiones discontinuas en las fronteras.
3. Los parámetros $\{w_k\}$, $\{b_k\}$ y $\{c_k\}$ son **independientes entre sí y todos entrenables**. Esta es la diferencia más sutil: en VLAD original el peso de asignación y el centroide del residuo son el mismo objeto. Al desacoplarlos, la capa puede aprender un criterio de asignación distinto de la posición del prototipo, lo que le da flexibilidad que el VLAD clásico no tiene.

La capa se enchufa después de las convoluciones, tratando cada posición del mapa de activaciones como un descriptor local. La salida $K \times D$ se normaliza (intra-cluster y global en L2) y suele pasar por una capa lineal de reducción.

**La supervisión débil.** El segundo aporte del paper es cómo entrenar sin etiquetas precisas: usan Google Street View Time Machine, que provee fotos del mismo lugar en épocas distintas. No se sabe qué imágenes muestran exactamente la misma escena —solo que están cerca geográficamente— así que definen una **pérdida de ranking** que exige que el mejor candidato positivo esté más cerca que todos los negativos.

## Resultados

NetVLAD supera de forma significativa a las representaciones anteriores en los benchmarks de reconocimiento de lugares, y también compite en recuperación estándar de imágenes. Las ablaciones importan más que los números absolutos: **el mismo backbone con pooling promedio o máximo rinde claramente peor** que con la capa NetVLAD, lo que aísla la contribución de la agregación respecto del extractor de features.

## Limitaciones

- **La dimensión de salida es $K \times D$**, que crece rápido: 64 clusters sobre features de 512 dimensiones son 32 768 valores antes de reducir. La capa de reducción es obligatoria en la práctica.
- **Todos los descriptores contribuyen a todos los clusters.** La asignación blanda evita las discontinuidades pero también impide descartar información irrelevante: el ruido también se agrega, repartido. Es exactamente el hueco que llena **GhostVLAD**, agregando clusters "fantasma" cuyos residuos se descartan y que actúan como sumidero de lo irrelevante.
- **Sensible a la inicialización.** En la práctica se inicializa con k-means sobre los descriptores del backbone preentrenado; arrancar de cero es notablemente más difícil.
- **Costo de memoria en entrenamiento**, por tener que materializar los residuos de todos los descriptores contra todos los clusters.

## Por qué importa para la Clase 41

La clase llega a NetVLAD por necesidad argumental. Después de exponer los cinco pasos de VLAD, plantea: *"In this process we need to find $x_i$, $c_k$ and $a_k(x_i)$. Is it possible to learn all of them end-to-end?"*. La respuesta es esta capa, y la slide siguiente la aplica al habla con la fórmula del softmax escrita tal cual aparece acá.

Lo que conviene retener es **por qué** el problema era difícil: no es que a nadie se le hubiera ocurrido entrenar VLAD, es que el `argmin` bloquea el gradiente. Toda la contribución cabe en cambiar una función indicadora por un softmax — y en darse cuenta de que al hacerlo se puede desacoplar el peso de asignación del centroide.

El salto del dominio visual al del audio es directo porque la estructura del problema es la misma: **un conjunto de descriptores locales de cardinalidad variable que hay que resumir en un vector fijo**. En imágenes, las posiciones de un mapa de activaciones; en habla, los frames de un espectrograma. [Xie et al. (2019)](/papers/utterance-level-xie-2019) hacen exactamente esa traducción. Ver [Agregación VLAD](/fundamentos/agregacion-vlad) para el mecanismo completo.
