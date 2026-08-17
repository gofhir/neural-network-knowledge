---
title: "FairMOT: la equidad entre detección y re-ID (2020)"
weight: 455
math: true
---

{{< paper-card
    title="FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking"
    authors="Yifu Zhang, Chunyu Wang, Xinggang Wang, Wenjun Zeng, Wenyu Liu (HUST / Microsoft Research Asia)"
    year="2020"
    venue="International Journal of Computer Vision (IJCV) / arXiv:2004.01888"
    arxiv="2004.01888"
    pdf="/papers/fairmot-zhang-2020.pdf" >}}
Fusionar el detector y el extractor de features de identidad en una sola red es atractivo: se ahorra un *forward* completo y las dos tareas se optimizan juntas. Los primeros intentos, sin embargo, rendían peor que el pipeline en dos etapas de [DeepSORT](/papers/deepsort-wojke-2017), y este paper explica por qué: las dos tareas **compiten**, y la re-identificación pierde sistemáticamente porque el diseño de la red está pensado para detectar. FairMOT identifica tres fuentes concretas de esa injusticia —los anchors, la resolución del mapa de features y la dimensión del embedding— y las corrige sobre una arquitectura *anchor-free* (CenterNet), obteniendo un tracker de una sola etapa competitivo y en tiempo real.
{{< /paper-card >}}

---

## Las tres injusticias

**1. Los anchors no sirven para la re-identificación.** En un detector basado en anchors, el problema tiene dos caras:

- Varios anchors desplazados pueden ser responsables del **mismo objeto**, y cada uno extraería su feature de identidad desde una posición distinta. Para detectar da igual —la caja se regresa correctamente—; para identidad no, porque el descriptor debe ser estable.
- Un mismo anchor puede quedar asignado a **dos objetos distintos** en escenas densas, produciendo un descriptor contaminado.

La corrección es usar un detector *anchor-free* que extrae el feature en el **centro** del objeto: un punto, un descriptor. De ahí la elección de CenterNet como base. Ver [Detección Anchor-Free](/fundamentos/anchor-free-detection).

**2. La resolución de los features.** La detección tolera *strides* grandes porque una caja se puede regresar desde un mapa grueso. La re-identificación necesita **detalle fino** para distinguir personas parecidas. FairMOT extrae los descriptores de un mapa de alta resolución con fusión multi-escala, en lugar de la última capa del backbone.

**3. La dimensión del embedding.** Contra la práctica de la literatura de re-ID pura, que usa 512 o más dimensiones, FairMOT muestra que en el contexto conjunto los embeddings **de baja dimensión (128)** funcionan mejor. La razón es la competencia entre tareas: menos dimensiones significan menos capacidad disputada al objetivo de detección, y además menos sobreajuste dado que los datos de tracking tienen muchas menos identidades que los datasets de re-ID.

{{< concept-alert type="clave" >}}
La palabra "fairness" del título no tiene nada que ver con equidad algorítmica en el sentido social. Se refiere a **equidad entre tareas** en un modelo multi-tarea: cuando dos cabezas comparten un backbone, la que define la arquitectura gana, y la otra hereda decisiones de diseño que le son adversas. Es un caso de estudio general de aprendizaje multi-tarea, no específico de tracking.
{{< /concept-alert >}}

## Arquitectura

Un backbone con fusión multi-escala (DLA-34 modificado) y **dos cabezas homogéneas** —ninguna subordinada a la otra—:

- **Cabeza de detección**: mapa de calor de centros, offsets y tamaños de caja, al estilo CenterNet.
- **Cabeza de re-ID**: un descriptor de 128-D por posición, entrenado como clasificación sobre las identidades del conjunto de entrenamiento.

En inferencia, se detectan los centros, se leen los descriptores en esas posiciones, y la asociación se hace con [filtro de Kalman](/fundamentos/filtro-de-kalman) más distancia de apariencia — el esquema de DeepSORT, pero con un solo *forward*.

## Resultados y contexto

En MOT17 obtiene 73,7 MOTA, 72,3 IDF1 y 59,3 HOTA en tiempo real, superando claramente a los enfoques *one-shot* previos y quedando competitivo con los de dos etapas. La comparación honesta hay que hacerla con las tablas de [OC-SORT](/papers/oc-sort-cao-2022) y [ByteTrack](/papers/bytetrack-zhang-2021), que un año después superan esas cifras (63,1-63,2 HOTA) **sin ningún modelo de apariencia**, apoyándose en detectores más fuertes.

Esa comparación es la lección de época: entre 2020 y 2022 el área descubrió que, con un detector suficientemente bueno, la re-identificación aportaba menos de lo esperado en los benchmarks estándar. Sigue siendo indispensable en el escenario para el que se inventó —oclusiones muy largas y multi-cámara—, y ahí FairMOT y sus descendientes conservan su lugar.

## Por qué importa para la Clase 42

Es la respuesta directa a una pregunta que la [Clase 42](/clases/clase-42) deja implícita. La clase presenta un pipeline de dos etapas —detectar, después describir y asociar— y en la parte de DeepSORT muestra la CNN de apariencia como un módulo separado. La pregunta natural del estudiante es: *si el detector ya extrajo features de esa región, ¿por qué computarlos de nuevo?* FairMOT es la respuesta: se puede, ahorra cómputo, y es más delicado de lo que parece porque las dos tareas quieren cosas distintas de la misma representación.

---

**Ver también:** [DeepSORT (2017)](/papers/deepsort-wojke-2017) · [ByteTrack (2021)](/papers/bytetrack-zhang-2021) · [Tracktor (2019)](/papers/tracktor-bergmann-2019) · [Re-identificación](/fundamentos/re-identificacion) · [Detección Anchor-Free](/fundamentos/anchor-free-detection)
