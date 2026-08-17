---
title: "Re-identificación"
weight: 136
math: true
---

La **re-identificación** (re-ID) es el problema de decidir si dos imágenes recortadas —tomadas en instantes distintos, o por cámaras distintas— corresponden al **mismo objeto individual**. No es clasificación: las identidades de test no aparecen en el entrenamiento. Es un problema de **conjunto abierto**, y su solución canónica es aprender un espacio de *embeddings* donde la distancia signifique identidad.

Este fundamento acompaña a la [Clase 42](/clases/clase-42), donde la re-ID entra como el "paréntesis" que resuelve la asociación cuando la geometría ya no alcanza.

---

## 1. Por qué el tracking la necesita

Un tracker basado solo en movimiento supone que **el objeto se movió poco**. La clase formula la objeción con precisión: *¿es razonable suponer eso? Tal vez… pero ¿qué si la cámara se mueve mucho? ¿O si hay periodos grandes de oclusión?*

Cuando el supuesto se rompe, la predicción del [filtro de Kalman](/fundamentos/filtro-de-kalman) deja de ser informativa y la asociación por IoU o Mahalanobis se vuelve ruido. Lo único que sobrevive a una oclusión de dos segundos es la **apariencia**: si el objeto que reaparece se parece al que desapareció, es el mismo. Ese es exactamente el trabajo de un modelo de re-ID.

{{< concept-alert type="clave" >}}
Movimiento y apariencia son **complementarios en el tiempo**, no redundantes. El movimiento es preciso a corto plazo y se degrada con la duración de la oclusión; la apariencia es indiferente al tiempo transcurrido pero ambigua entre objetos parecidos. DeepSORT usa el primero como compuerta y el segundo como costo, en esa división de trabajo.
{{< /concept-alert >}}

## 2. Aprender una métrica: siamesas y tripletas

La clase presenta la maquinaria bajo el título "aprender una métrica de distancia", con la pregunta guía: *¿son estos pares de imágenes de la misma persona?*

**Red siamesa.** Dos ramas con **pesos compartidos** (*tied weights*) procesan las dos imágenes y producen sendos vectores; la pérdida opera sobre la distancia entre ellos. La versión de la clase, el *pairwise ranking loss* (o *contrastive loss*):

$$L(f(I_1), f(I_2)) = \begin{cases} \lVert f(I_1) - f(I_2) \rVert & \text{misma clase} \\[4pt] \max\{0,\; m - \lVert f(I_1) - f(I_2)\rVert\} & \text{clases distintas} \end{cases}$$

Los pares iguales se atraen sin límite; los distintos se repelen hasta el margen $m$ y después dejan de aportar gradiente.

**Por qué es inestable.** La clase plantea la pregunta y da la respuesta correcta: **hay atajos**. Cada término, aislado, tiene una solución degenerada trivial. Si el *batch* está lleno de pares positivos, la red colapsa todo a un punto ($L=0$ para todos). Si está lleno de negativos, basta con dispersar todo lo suficiente. El objetivo es correcto solo si ambos términos están **equilibrados dentro del mismo lote**, y eso depende del muestreo, no de la pérdida.

**Triplet network.** La corrección es hacer que los dos términos compartan un ancla y compitan en la misma expresión. Con $I_1$ ancla, $I_2$ positivo y $I_3$ negativo:

$$L = \max\{0,\; m - \lVert f(I_1)-f(I_3)\rVert + \lVert f(I_1)-f(I_2)\rVert\}$$

La pérdida es cero cuando el negativo está más lejos que el positivo **por al menos el margen** $m$. Ahora el colapso ya no es solución: si todo va al mismo punto, ambas distancias son 0 y la pérdida vale $m > 0$. La formulación es la de [FaceNet](/papers/facenet-schroff-2015), y el detalle que la hace funcionar en la práctica es la **minería de tripletas difíciles**: elegir negativos que estén cerca, porque los negativos fáciles dan gradiente nulo y desperdician el lote.

Ver [Triplet Loss](/fundamentos/triplet-loss) y [Metric Learning](/fundamentos/metric-learning) para el desarrollo completo.

**Generalidad.** La clase subraya, y es correcto, que esto **no es específico de caras**: el mismo esquema aprende un espacio donde sillas quedan cerca de sillas y payasos cerca de payasos. Lo que cambia es el criterio de "misma clase" con que se construyen los pares — y para tracking ese criterio es "misma identidad", no "misma categoría".

## 3. El descriptor de DeepSORT

El caso concreto de la clase. [DeepSORT](/papers/deepsort-wojke-2017) entrena **offline** una CNN sobre un dataset de re-identificación de personas y la usa **congelada** durante el seguimiento:

- Arquitectura: una *wide residual network* pequeña —dos convoluciones y seis bloques residuales—, **2 800 864 parámetros** en total.
- Salida: un vector de **128 dimensiones**, con normalización $\ell_2$ final que lo proyecta a la **hiperesfera unitaria**. Esa normalización es la que vuelve equivalentes producto punto y coseno, y permite usar $1 - r_j^\top r_k$ como distancia.
- Entrenamiento: sobre **MARS** (Zheng et al., 2016), 1 100 000 imágenes de 1 261 peatones.
- Costo: un *forward* de 32 cajas toma unos 30 ms en una GPU de portátil. En el tracker completo, la extracción de features consume aproximadamente **la mitad** del tiempo total.

La comparación no usa un solo vector por trayectoria sino una **galería**: se guardan los últimos $L_k = 100$ descriptores asociados a cada trayectoria y la distancia es el **mínimo** sobre la galería:

$$d^{(2)}(i,j) = \min\{\,1 - r_j^\top r_k^{(i)} \;\mid\; r_k^{(i)} \in \mathcal{R}_i \,\}$$

Usar el mínimo y no el promedio es deliberado: una persona cambia de apariencia al girar, y basta con que la detección nueva se parezca a **alguna** de sus vistas pasadas.

{{< concept-alert type="recordar" >}}
Esto es **vecino más cercano sin aprendizaje online**. Toda la complejidad se pagó en la etapa de pre-entrenamiento; durante el seguimiento no se ajusta nada. Es la misma filosofía de SORT —mantener el sistema en línea trivialmente barato— aplicada a la apariencia.
{{< /concept-alert >}}

## 4. Re-ID integrada: el problema de la equidad

La generación siguiente intentó **fusionar** el detector y el extractor de features en una sola red (*one-shot*, *joint detection and embedding*): un backbone, dos cabezas. Ahorra un *forward* completo.

[FairMOT](/papers/fairmot-zhang-2020) documenta por qué el resultado inicial fue peor de lo esperado: las dos tareas **compiten**, y la re-ID pierde sistemáticamente. Tres razones, todas concretas:

1. **Los anchors no sirven para re-ID.** En un detector basado en anchors, varios anchors desplazados responden por el mismo objeto y todos extraen el feature de identidad de posiciones distintas; y un mismo anchor puede ser responsable de dos objetos. La identidad exige un feature **centrado en el objeto**, lo que empuja hacia detectores *anchor-free* (CenterNet).
2. **La resolución del feature map.** La detección tolera *strides* grandes; la re-ID no, porque necesita detalle fino. Hay que extraer los descriptores de un mapa de alta resolución con fusión multi-escala.
3. **La dimensión del embedding.** Contra la intuición heredada de re-ID pura, dimensiones **bajas** (128) funcionan mejor en el contexto conjunto que las 512 habituales: reducen la competencia con la tarea de detección.

Ver [Detección Anchor-Free](/fundamentos/anchor-free-detection) para el primer punto.

## 5. Los límites

- **Cambio de ropa y de vista.** Los descriptores de re-ID de personas dependen fuertemente del color de la ropa. Ante un cambio de vestuario, o entre cámaras con iluminación muy distinta, el rendimiento cae.
- **Objetos visualmente idénticos.** El caso patológico es un equipo deportivo con uniformes iguales, o un enjambre de células. La apariencia no discrimina y hay que volver al movimiento — o a la estructura del grupo, como hace SocialTrack en UAV.
- **Sesgo del dataset.** Un descriptor entrenado en MARS (peatones de vigilancia) transferido a otro dominio hereda las correlaciones de MARS. Es la misma advertencia que en cualquier transferencia.
- **Costo.** Duplicar el pipeline con un extractor de apariencia es lo que separa a SORT (260 Hz) de DeepSORT (~20 Hz medidos). En *edge* y tiempo real, eso puede ser prohibitivo.

## 6. Más allá del tracking

La re-ID es una tarea con literatura propia (Market-1501, MARS, DukeMTMC, VeRi para vehículos) y aplicaciones fuera del video: **búsqueda de personas** en archivos de vigilancia, **conteo de visitantes únicos** en retail, y —el paralelo estructural más directo— el **record linkage** de registros clínicos, donde el problema es idéntico salvo que las "imágenes" son registros y la métrica se aprende sobre campos demográficos. En ambos casos se trata de conjunto abierto, se resuelve con un *embedding* más un umbral, y se evalúa con la misma maquinaria de pares.

---

## Ver también

- [Triplet Loss](/fundamentos/triplet-loss) y [Metric Learning](/fundamentos/metric-learning) — la maquinaria de aprendizaje.
- [Siamese Networks (Koch et al., 2015)](/papers/siamese-networks-koch-2015) y [FaceNet (Schroff et al., 2015)](/papers/facenet-schroff-2015) — los papers de referencia.
- [DeepSORT](/papers/deepsort-wojke-2017) — el descriptor de 128-D en acción.
- [FairMOT](/papers/fairmot-zhang-2020) — por qué integrar re-ID y detección es más difícil de lo que parece.
- [Seguimiento de Objetos](/fundamentos/seguimiento-de-objetos) — el contexto.
- [Reconocimiento de Hablante](/fundamentos/reconocimiento-de-hablante) — el mismo problema de conjunto abierto sobre audio.
