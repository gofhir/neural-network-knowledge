---
title: "MANN (Memory-Augmented Neural Networks)"
weight: 262
math: true
---
{{< paper-card
    title="One-shot Learning with Memory-Augmented Neural Networks"
    authors="Adam Santoro, Sergey Bartunov, Matthew Botvinick, Daan Wierstra, Timothy Lillicrap"
    year="2016"
    venue="ICML 2016 (Google DeepMind)"
    pdf="/papers/mann-santoro-2016.pdf"
    arxiv="1605.06065" >}}
Aplica las Neural Turing Machines al one-shot learning planteado como meta-learning en dos escalas temporales: un aprendizaje lento en los pesos del controlador y uno rapido en una memoria externa direccionable. Introduce el modulo de acceso Least Recently Used Access (LRUA) y un esquema de entrenamiento episodico con etiquetas desfasadas que obliga a la red a ligar representaciones a etiquetas en memoria. Supera a humanos en la clasificacion one-shot sobre Omniglot.
{{< /paper-card >}}

## El problema

El deep learning clásico funciona aplicando descenso de gradiente a modelos de alta capacidad sobre grandes conjuntos de datos, con entrenamiento incremental y extenso. Pero muchas tareas reales exigen lo contrario: inferir rápido a partir de pocos ejemplos. En el límite del **one-shot learning**, una sola observación debería bastar para cambiar el comportamiento de forma correcta.

Para una red entrenada por gradiente esto es duro. Reentrenar los pesos con un puñado de ejemplos nuevos conduce a aprendizaje pobre y, peor aún, a **interferencia catastrófica**: los gradientes nuevos sobrescriben representaciones útiles ya aprendidas. Por eso los métodos no paramétricos como k-NN suelen adaptarse mejor a este régimen —no "olvidan" porque simplemente almacenan.

La alternativa que propone el paper es el **meta-learning**: aprender en dos escalas temporales. Un aprendizaje *lento*, gradual, que vive en los pesos $\theta$ entrenados por gradiente y captura conocimiento transversal a las tareas (cómo extraer buenas representaciones, cómo atar representaciones a etiquetas). Y un aprendizaje *rápido*, episódico, que almacena información específica del episodio actual tras una sola presentación. La pregunta central del paper es: ¿dónde vive ese aprendizaje rápido?

La respuesta de los autores: en una **memoria externa direccionable**, no en el estado oculto de una red recurrente. El argumento es que la memoria implícita de una LSTM no escala. Una solución que sí escala necesita dos propiedades: (1) la información debe almacenarse de forma estable y direccionable elemento a elemento, con acceso selectivo a las piezas relevantes; y (2) el número de parámetros no debe estar atado al tamaño de la memoria. En una LSTM, ampliar la memoria significa ampliar el estado oculto, lo que infla los parámetros y mezcla todo en un vector denso difícil de direccionar.

## De las Neural Turing Machines a MANN

La pieza que cumple ambos requisitos es la **Neural Turing Machine** (NTM, Graves et al., 2014), antecedente directo del MANN. Una NTM es una implementación totalmente diferenciable de una red aumentada con memoria: un **controlador** (feedforward o LSTM) interactúa con una **matriz de memoria externa** $M_t \in \mathbb{R}^{N\times m}$ de $N$ slots de tamaño $m$, mediante cabezales de lectura y escritura. Vectores entran y salen de memoria en cada paso temporal, y crucialmente el tamaño de $M_t$ es independiente del número de parámetros del controlador: agrandar la memoria no agranda la red.

Eso convierte a la NTM en candidata natural para one-shot: largo plazo vía actualizaciones lentas de los pesos, corto plazo vía la memoria externa. Los autores acuñan el término **MANN** (memory-augmented neural network) para esta clase de redes con memoria externa, en oposición a la memoria "interna" de las LSTM. El paper aporta dos cosas sobre esta base: un **setup episódico** que fuerza el uso de la memoria, y un **nuevo módulo de escritura llamado LRUA** que reemplaza el direccionamiento por ubicación de la NTM original.

## El setup episódico (labels con offset temporal, shuffle)

En aprendizaje supervisado clásico se eligen parámetros $\theta$ que minimizan un costo sobre *un* dataset. En meta-learning se minimiza el costo esperado sobre una **distribución de datasets** $p(D)$:

$$\theta^{*} = \arg\min_{\theta}\; \mathbb{E}_{D\sim p(D)}\big[L(D;\theta)\big].$$

El cambio es decisivo: $\theta$ ya no resuelve *un* problema, sino que aprende a resolver problemas de *una familia*. Lo que se acumula en $\theta$ es meta-conocimiento sobre la estructura compartida, no sobre el contenido de ningún dataset particular.

Un episodio es la presentación de un dataset $D = \{(x_t, y_t)\}_{t=1}^{T}$. El truco de diseño que define el paper es el **offset temporal de las etiquetas**: la etiqueta $y_t$ es a la vez el objetivo a predecir en el paso $t$ *y* una entrada que se presenta en el paso siguiente. La secuencia que ve la red es:

$$(x_1, \text{null}),\; (x_2, y_1),\; (x_3, y_2),\; \ldots,\; (x_T, y_{T-1}).$$

En el paso $t$ la red recibe la nueva consulta $x_t$ junto con la etiqueta correcta del ejemplo *anterior* $y_{t-1}$, y debe predecir la etiqueta de $x_t$. ¿Por qué este desfase es esencial? Si la red recibiera $(x_t, y_t)$ simultáneamente, la tarea sería trivial: bastaría copiar $y_t$ a la salida. El offset rompe el atajo. En el paso $t$ la red ve $x_t$ pero todavía no sabe su etiqueta: debe arriesgar una predicción. Solo en el paso $t+1$ recibe $y_t$, y ese es el momento en que puede *atar* (bind) la representación de $x_t$ con su etiqueta verdadera y guardar el binding en memoria. Cuando más tarde aparece otra muestra de la misma clase, debe *recuperar* el binding y acertar.

El segundo ingrediente es el **barajado** (shuffling): las clases, las etiquetas y las muestras se barajan entre episodios. La misma clase visual de Omniglot puede ser "etiqueta 3" en un episodio y "etiqueta 1" en otro. Esto impide que la red memorice en sus pesos la asociación "este carácter → clase 3"; si lo hiciera, la memoria externa sería innecesaria. Al barajar, la única asociación estable es *estructural* —"ata lo que veas a la etiqueta que venga después y recupéralo"— y no de contenido. Ese es exactamente el meta-conocimiento que se quiere forzar.

La consecuencia es que la conducta óptima en un episodio es: adivinar al azar en la primera presentación de cada clase (la etiqueta no puede inferirse por el barajado) y luego usar la memoria para alcanzar precisión perfecta.

## Acceso a memoria: content-based addressing y LRUA

El controlador es el cerebro que decide qué leer y qué escribir; la memoria es un sustrato pasivo direccionable. El mejor controlador resultó ser un **LSTM de 200 unidades ocultas**. Un detalle clave de su dinámica: la salida del controlador es la concatenación del estado oculto con lo leído de memoria, $o_t = (h_t, r_t)$. El LSTM no es un clasificador autosuficiente: su predicción depende explícitamente de lo recuperado. Esto crea el canal de gradiente que, durante el backprop a través del tiempo, le enseña *qué* clave generar para leer y escribir de forma útil.

**Lectura (content-based addressing).** Dada la clave $k_t$ producida por el controlador, se calcula la similitud coseno con cada fila $M_t(i)$:

$$K\big(k_t, M_t(i)\big) = \frac{k_t \cdot M_t(i)}{\lVert k_t \rVert\, \lVert M_t(i)\rVert}.$$

Las similitudes se normalizan con softmax para producir los pesos de lectura:

$$w_t^{r}(i) \leftarrow \frac{\exp\!\big(K(k_t, M_t(i))\big)}{\sum_j \exp\!\big(K(k_t, M_t(j))\big)},$$

y la memoria recuperada es la combinación convexa de las filas:

$$r_t \leftarrow \sum_i w_t^{r}(i)\, M_t(i).$$

Quien venga de Transformers reconocerá el patrón: $k_t$ es la query, las filas $M_t(i)$ son simultáneamente keys y values, similitud + softmax da la distribución de atención y $r_t$ es el contexto atendido. Se usaron **4 lecturas** simultáneas (concatenadas a la salida), análogas a las cabezas de la multi-head attention.

**Escritura (LRUA).** Aquí está la contribución original. La NTM original direcciona por contenido *y por ubicación*; el location-based addressing favorece avanzar por la "cinta" y saltar, lo cual ayuda en tareas secuenciales donde el orden importa. Pero en one-shot el orden no importa: lo que importa es atar muestra↔etiqueta. **LRUA (Least Recently Used Access)** es un escritor puramente basado en contenido que escribe en una de dos posiciones: la **menos usada** (preservando lo demás) o la **leída más recientemente** (actualizando información reciente).

Se mantienen pesos de uso $w_t^{u}$ que decaen y suman las lecturas y escrituras actuales:

$$w_t^{u} \leftarrow \gamma\, w_{t-1}^{u} + w_t^{r} + w_t^{w}, \qquad \gamma = 0.99.$$

Con ellos se define una máscara binaria $w_t^{lu}$ que marca con 1 los $n$ slots menos usados ($n$ igual al número de lecturas). Los pesos de escritura son una combinación convexa entre lecturas previas y least-used previos, modulada por una compuerta sigmoidea aprendible:

$$w_t^{w} \leftarrow \sigma(\alpha)\, w_{t-1}^{r} + \big(1 - \sigma(\alpha)\big)\, w_{t-1}^{lu}.$$

La interpretación es elegante: si $\sigma(\alpha)\to 1$ se escribe en la posición leída más recientemente (actualización); si $\sigma(\alpha)\to 0$ se escribe en la menos usada (depositar la novedad en un slot libre). El gate $\alpha$ se aprende por gradiente, así que la red *descubre* la política de escritura óptima para la familia de tareas. Finalmente, tras poner a cero el slot menos usado, la escritura es aditiva:

$$M_t(i) \leftarrow M_{t-1}(i) + w_t^{w}(i)\, k_t.$$

## Resultados en Omniglot (tabla por instancia 1st..10th, números reales)

Omniglot tiene más de **1600 clases** de caracteres con muy pocos ejemplos cada una —el "transpuesto de MNIST". Se entrenó con las 1200 clases originales (más augmentations) y se reservaron las 423 restantes para test, con imágenes reescaladas a 20×20. Tras **100 000 episodios** de 5 clases con etiquetas barajadas, se evaluó con los pesos congelados sobre clases nunca vistas.

**Precisión por instancia (one-hot, 5 clases/episodio):**

| Modelo | 1.ª | 2.ª | 3.ª | 4.ª | 5.ª | 10.ª |
|---|---|---|---|---|---|---|
| Human | 34.5 | 57.3 | 70.1 | 71.8 | 81.4 | 92.4 |
| Feedforward | 24.4 | 19.6 | 21.1 | 19.9 | 22.8 | 19.5 |
| LSTM | 24.4 | 49.5 | 55.3 | 61.0 | 63.6 | 62.5 |
| **MANN** | **36.4** | **82.8** | **91.0** | **92.6** | **94.9** | **98.1** |

El salto de 36.4% a 82.8% entre la primera y la segunda presentación de una clase es la firma del aprendizaje one-shot: la red ve un ejemplo, lo ata a su etiqueta, lo guarda, y al reencontrar la clase lo recupera. El feedforward no aprende (~20%, que es el azar para 5 clases): sin memoria no puede acumular información en el episodio. El LSTM aprende algo pero se satura cerca del 62%: su estado oculto no escala como almacén direccionable de bindings barajados. El MANN supera al humano en todas las instancias.

Un detalle fascinante: el MANN supera el azar incluso en la *primera* instancia (36.4% > 20%). Es una "adivinanza educada": si una muestra no hace buen match con ningún binding almacenado, la red infiere que es una clase nueva y evita las etiquetas ya asignadas. Los humanos reportaron una estrategia similar.

Para escalar a más clases se usaron etiquetas de strings de cinco caracteres ($5^5 = 3125$ combinaciones), reduciendo la chance de repetir etiqueta entre episodios. Con etiquetas string y **15 clases por episodio** se compara LRUA contra la NTM original:

| Modelo | Controlador | #Clases | 1.ª | 2.ª | 3.ª | 4.ª | 5.ª | 10.ª |
|---|---|---|---|---|---|---|---|---|
| kNN (deep features) | – | 15 | 0.4 | 32.7 | 41.2 | 47.1 | 50.6 | 60.0 |
| LSTM | – | 15 | 0.0 | 2.2 | 2.9 | 4.3 | 5.6 | 12.7 |
| **MANN (LRUA)** | **LSTM** | **15** | **0.1** | **62.6** | **79.3** | **86.6** | **88.7** | **95.3** |
| MANN (NTM) | LSTM | 15 | 0.0 | 35.4 | 61.2 | 71.7 | 77.7 | 88.4 |

Tres lecturas: (1) el controlador importa —MANN con LSTM aplasta a MANN feedforward; recurrencia y memoria son complementarias. (2) LRUA supera a la NTM con un margen enorme en la segunda instancia (62.6 vs 35.4): dedicar todo el direccionamiento al contenido y gestionar slots por recencia es mejor que modelar una estructura secuencial que la tarea no tiene. (3) MANN supera a un kNN con deep features de autoencoder, memoria ilimitada, más parámetros y triple de datos.

## Por qué importa hoy (puente a la atención key-value de Transformers)

La conexión más importante para entender la trayectoria del campo: la lectura de MANN —query $k_t$, keys/values $M_t(i)$, similitud + softmax, contexto $r_t = \sum_i w_t^r(i)\, M_t(i)$— es estructuralmente la **atención key-value** que un año después se vuelve el corazón de los Transformers (Vaswani et al., 2017). La self-attention de un Transformer puede leerse como una MANN *sin escritura persistente*: la "memoria" es el conjunto de tokens del contexto, recalculada en cada forward. A la inversa, los Transformers con memoria externa o recurrencia (Transformer-XL, Compressive Transformer, modelos con KV-cache y memoria de largo plazo) reintroducen la escritura persistente que MANN ya proponía. Lo único que separa conceptualmente ambos mundos es coseno vs producto punto escalado, y memoria escribible vs recalculada.

La tesis profunda del paper es la **separación explícita de dos mecanismos de aprendizaje**: aprendizaje gradual en los pesos (lento, integra sobre miles de episodios, conocimiento de fondo, análogo a la consolidación cortical) y almacenamiento rápido en memoria (instantáneo, específico del episodio, se borra entre tareas, análogo al hipocampo). Es la teoría de sistemas de memoria complementarios traducida a arquitectura. La moraleja de ingeniería sigue vigente: cuando el problema exige incorporar información nueva rápido sin destruir lo aprendido, no reentrenes los pesos —dale al modelo una memoria externa direccionable y entrena los pesos para usarla bien. Esa es exactamente la intuición que hoy sostiene los sistemas RAG y de memoria sobre LLMs, formalizada aquí por primera vez de extremo a extremo y diferenciable.

MANN también inspiró arquitecturas como Meta Networks (Munkhdalai & Yu, 2017) y SNAIL (Mishra et al., 2018), que reemplaza la memoria explícita por atención causal sobre la historia del episodio, y la línea de memoria episódica diferenciable de los Differentiable Neural Computers (Graves et al., 2016).

## Conexión con la Clase 26

En el arco de la Clase 26, MANN es el primer puente diferenciable entre redes profundas y memoria externa direccionable aplicado a aprendizaje con pocos ejemplos. Conecta hacia atrás con las Neural Turing Machines (memoria como cinta diferenciable) y hacia adelante con la atención key-value que define a los Transformers. Si la clase trata meta-learning, memoria o el linaje de la atención, MANN muestra *por qué* y *cómo* desacoplar "aprender a representar" (lento, en pesos) de "recordar lo recién visto" (rápido, en memoria), y por qué el offset temporal de etiquetas y el barajado son los que fuerzan esa separación.

La traducción a salud es directa. Las enfermedades raras y las clases de cola larga son el régimen one-shot/few-shot de la práctica clínica: miles de condiciones con pocos casos cada una. Un sistema que meta-aprenda la *estructura* del diagnóstico podría adaptarse a una condición nueva con un puñado de casos, sin reentrenar y sin interferencia catastrófica. Cada paciente nuevo es un episodio con estructura compartida (fisiología) pero contenido propio (su historia), y una memoria borrable por episodio separa bien el conocimiento poblacional (pesos) del estado individual (memoria). En record-linkage de pacientes, cada bloque de registros candidatos es un episodio donde la tarea estructural es estable pero el contenido es nuevo: la lógica de "atar representación↔decisión en memoria de contenido y recuperar por similitud" es análoga a un blocker/scorer basado en embeddings. La calibración de incertidumbre que MANN exhibe en regresión —varianza que crece lejos de lo observado— es además deseable en cualquier scorer clínico: saber *cuándo no sabe* importa tanto como acertar.

## Notas y enlaces

Limitaciones a tener presentes: la memoria (128 slots de tamaño 40) escala con dificultad —al subir a 50–100 clases el rendimiento decae gradualmente por interferencia entre bindings. MANN depende de un *reset* externo de memoria entre episodios; sin él aparece interferencia proactiva y desaparece el spike característico. El rendimiento depende fuertemente del controlador (la memoria sola no basta). Y en retrospectiva, los métodos métricos posteriores (Matching Networks, Prototypical Networks) resultaron a menudo más simples y fuertes en few-shot puro que las arquitecturas memory-augmented, más pesadas de entrenar por el BPTT largo y la dinámica de memoria sensible a hiperparámetros.

Fundamentos relacionados: [Meta-aprendizaje](/fundamentos/meta-aprendizaje), [Memory-Augmented Networks](/fundamentos/memory-augmented-networks), [Few-shot learning](/fundamentos/few-shot-learning), [Self-attention](/fundamentos/self-attention).

Papers relacionados: [Matching Networks (Vinyals, 2016)](/papers/matching-networks-vinyals-2016), [MAML (Finn, 2017)](/papers/maml-finn-2017), [Omniglot (Lake, 2015)](/papers/omniglot-lake-2015).

Ver [Clase 26 -- Meta-aprendizaje](/clases/clase-26).
