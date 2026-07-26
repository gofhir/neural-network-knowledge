---
title: "LRCN: CNN + LSTM para video (2015)"
weight: 403
math: true
---

{{< paper-card
    title="Long-term Recurrent Convolutional Networks for Visual Recognition and Description"
    authors="Jeff Donahue et al. (UC Berkeley)"
    year="2015"
    venue="CVPR 2015 / arXiv:1411.4389"
    pdf="/papers/lrcn-donahue-2015.pdf" >}}
LRCN es, literalmente, el **"2D CNN + RNN"** que la [Clase 36](/clases/clase-36) propone como solución para inyectar noción temporal. La idea es sencilla y potente: una **CNN extrae un vector de características por cada frame** y esos vectores se entregan como secuencia a una pila de **LSTM** que modela la dinámica temporal. Los autores llaman a estos modelos "doblemente profundos" (*doubly deep*) porque aprenden representaciones composicionales tanto en el **espacio** (capas de la CNN) como en el **tiempo** (desenrollado del LSTM). Su virtud clave es la **generalidad**: un mismo esquema resuelve reconocimiento de actividad (video → etiqueta), *image captioning* (imagen → oración) y descripción de video (video → oración), según qué extremo sea secuencial. La slide de la clase resume su balance en una línea —"funciona mejor que la 2D CNN sola, pero el RNN no se puede paralelizar"— y LRCN aporta la evidencia de ambos términos: en UCF-101 supera al baseline *single frame* en hasta **3.40%**, mientras el propio paper reconoce que la inferencia recurrente **debe correr secuencialmente**.
{{< /paper-card >}}

---

## Contexto: la 2D CNN por frame ignora el orden temporal

Hacia 2014–2015 las CNN dominaban el reconocimiento de imágenes estáticas, pero extenderlas a video planteaba un problema estructural: **una imagen es estática, un video es una secuencia** de longitud variable, y para descripción la salida también debe tener longitud variable (oraciones completas). La investigación previa había explorado dos extremos que el paper critica: los **filtros 3D espacio-temporales** (costosos y difíciles de preentrenar sobre imágenes) y las **representaciones frame-a-frame agregadas por ventanas fijas** con *pooling* temporal simple.

El problema de fondo del promediado temporal es que **destruye el orden**. Si un modelo clasifica cada frame de forma independiente con una 2D CNN y luego promedia las probabilidades, el resultado es idéntico sin importar en qué **secuencia** ocurrieron los frames. Levantarse de una silla y sentarse en ella producen frames casi idénticos; solo el **orden** los distingue, y una 2D CNN por frame seguida de promedio es ciega a esa diferencia. La alternativa es construir modelos **profundos también en el tiempo**: las [redes recurrentes](/fundamentos/redes-recurrentes) son "profundas en el tiempo" al desenrollarse, y el **LSTM** (Hochreiter & Schmidhuber, 1997) resuelve el desvanecimiento del gradiente con un estado de celda gobernado por compuertas, habilitando el aprendizaje de **dependencias de largo alcance**.

## Método: una CNN por frame, un LSTM sobre la secuencia

Cada entrada visual $x_t$ —una imagen o un frame— pasa por una CNN $\phi_V(\cdot)$ con parámetros $V$ que produce un vector de longitud fija $\phi_V(x_t)$. Esas salidas alimentan una pila de LSTM con parámetros $W$. Un LSTM básico se rige por:

$$
i_t = \sigma(W_{xi} x_t + W_{hi} h_{t-1} + b_i), \qquad f_t = \sigma(W_{xf} x_t + W_{hf} h_{t-1} + b_f)
$$
$$
o_t = \sigma(W_{xo} x_t + W_{ho} h_{t-1} + b_o), \qquad g_t = \tanh(W_{xc} x_t + W_{hc} h_{t-1} + b_c)
$$
$$
c_t = f_t \odot c_{t-1} + i_t \odot g_t, \qquad h_t = o_t \odot \tanh(c_t)
$$

donde $\odot$ es el producto elemento a elemento, $i_t$ la compuerta de entrada, $f_t$ la de olvido, $o_t$ la de salida y $c_t$ la **celda de memoria**. Como $i_t$ y $f_t$ son sigmoidales, actúan como "perillas" que el LSTM aprende para **olvidar selectivamente** su memoria o **considerar** su entrada, capturando dinámica de largo plazo. El entrenamiento minimiza la log-verosimilitud negativa de las salidas verdaderas:

$$
\mathcal{L}(V, W, \mathcal{D}) = -\frac{1}{|\mathcal{D}|} \sum_{(x_t, y_t)_{t=1}^{T} \in \mathcal{D}} \sum_{t=1}^{T} \log P(y_t \mid x_{1:t}, y_{1:t-1}, V, W)
$$

Un detalle de diseño **anticipa la tensión de la Clase 36**: la transformación visual $\phi_V(\cdot)$ es invariante al tiempo y sus pesos están **atados a través del tiempo**. Esto hace que la parte convolucional, que es cara, sea **paralelizable** sobre todos los pasos temporales; la parte LSTM, en cambio, no lo es. La elegancia del modelo está en que las tres tareas son instancias de tres clases de aprendizaje secuencial: **entrada secuencial → salida estática** (reconocimiento de actividad, con *late fusion* de las predicciones por paso), **entrada estática → salida secuencial** (captioning, duplicando la entrada en cada paso) y **entrada y salida secuenciales** (descripción de video, con esquema encoder-decoder).

## Resultados

En **reconocimiento de actividad sobre UCF101** (>12.000 videos, 101 clases, tres splits), la CNN base es un híbrido de CaffeNet preentrenado en ImageNet, y el baseline clave es *single frame* —los frames clasificados individualmente y promediados, **sin modelización de la secuencia**: exactamente la "2D CNN sola" de la clase. LRCN es esa misma CNN con una LSTM encima:

| Modelo | RGB | Flow | Ponderado (1/2, 1/2) | Ponderado (1/3, 2/3) |
|---|---|---|---|---|
| Single frame | 67.37 | 74.37 | 75.46 | 80.90 |
| LRCN-fc6 | 68.20 | 77.28 | 78.94 | 82.34 |

LRCN mejora al baseline en **0.83%** (RGB), **2.91%** (flujo) y **3.40%** con el promedio ponderado que favorece el flujo. El mensaje es inequívoco: **añadir la LSTM sobre la misma CNN mejora el reconocimiento**, porque la secuencia temporal aporta información que el promedio de frames destruía. En el análisis por clase, LRCN gana especialmente donde el **movimiento y su orden** son distintivos (*BoxingPunchingBag* +40.82, *HighJump* +29.73, *JumpRope*) y pierde algo donde basta reconocer objetos estáticos (*Knitting*, *Mixing*), pero las ganancias superan a las pérdidas. En perspectiva de la época, LRCN es comparable a otros modelos profundos (la two-stream de Simonyan & Zisserman reportaba 87.6%, la 3D CNN solo 65.4%), y una variante con cuatro LSTM apilados y preentrenamiento sobre Sports-1M alcanzó **88.6%**, confirmando que la receta CNN+LSTM escalaba. En *image captioning* (COCO 2014) la mejor configuración quedó 4.ª en CIDEr-D (0.934) y 3.ª en METEOR (0.335), competitiva con el estado del arte del 2015 COCO Caption Challenge.

## Limitaciones

- **El RNN es secuencial, no paralelizable (la desventaja de la slide).** El paper lo hace explícito: la inferencia recurrente **debe ejecutarse secuencialmente** —$h_1 = f_W(x_1, h_0)$, luego $h_2 = f_W(x_2, h_1)$, y así hasta $h_T$—. Mientras la **CNN por frame sí se paraleliza** (por eso se atan sus pesos), la recurrencia del LSTM introduce una dependencia estricta paso-a-paso: $h_t$ no puede calcularse hasta tener $h_{t-1}$. Esta serialización es precisamente la motivación que años después llevaría a reemplazar la recurrencia por **atención** (Transformers), que procesa toda la secuencia en paralelo.
- **Hambriento de datos.** El entrenamiento end-to-end requirió *dropout* muy agresivo (0.9) para no sobreajustar los datasets de video, pequeños; la capacidad temporal solo se explota plenamente con datasets masivos (Sports-1M).
- **Dinámica temporal limitada en los benchmarks.** Los propios autores reconocen que los datasets de actividad de la época no tienen dinámica temporal compleja, lo que matiza cuánto de la ganancia proviene del modelado de secuencia.

## Por qué importa para la Clase 36

**LRCN es, literalmente, el "2D CNN + RNN" de la [Clase 36](/clases/clase-36).** La clase recorre el problema de dotar de noción temporal a modelos de imagen y termina proponiendo esta arquitectura como solución. Resume su balance en una frase, y LRCN aporta la evidencia de cada término:

- **"Funciona mejor que la 2D CNN sola":** la tabla lo cuantifica —LRCN supera al *single frame* en hasta 3.40%—, porque el promedio de frames de una 2D CNN destruye el orden temporal y la LSTM lo recupera.
- **"El RNN es apto para procesar secuencias" (ventaja):** las compuertas del LSTM habilitan dependencias de **largo alcance**, justo lo que un *pooling* de ventana fija no puede capturar.
- **"El RNN no se puede paralelizar" (desventaja):** la inferencia recurrente debe correr secuencialmente, el defecto que motivaría el salto hacia arquitecturas basadas en atención.

Respecto del [laboratorio de la clase](/laboratorios/lab-36) (ResNet + componente temporal), LRCN es el molde conceptual: donde el paper usa CaffeNet/VGGNet como extractor $\phi_V$, el lab usa una **ResNet** preentrenada para obtener un vector por frame, y añade su propio bloque temporal sobre esa secuencia de *embeddings*. La lección transferible es **desacoplar** un extractor visual fuerte y preentrenable de un modelo temporal ligero encima, y **atar los pesos de la CNN en el tiempo** para paralelizar la parte cara y reservar la serialización solo para el módulo recurrente. Ver el fundamento transversal de [redes recurrentes](/fundamentos/redes-recurrentes) para RNN, LSTM y cómputo secuencial.
