# HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Autores:** Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, Abdelrahman Mohamed. El grueso del equipo está en **Facebook AI Research**; Tsai y Salakhutdinov vienen de Carnegie Mellon.
- **Venue:** *IEEE/ACM Transactions on Audio, Speech, and Language Processing* (TASLP), vol. 29, 2021, pp. 3451–3460. El preprint es **arXiv:2106.07447v1, 14 de junio de 2021**, que es la versión sobre la que trabaja este análisis.
- **Código y pesos:** `fairseq/examples/hubert` (nota al pie 1 de la primera página).
- **Index terms declarados:** "Self-supervised learning, BERT". Literalmente dos palabras: el paper se posiciona como *BERT para habla*, y todo el aporte está en cómo se fabrica el vocabulario que BERT necesita.

El problema que ataca es concreto. BERT necesita un objetivo discreto para predecir en las posiciones enmascaradas; el habla es una señal continua sin tokens. wav2vec 2.0 había esquivado el problema con una pérdida contrastiva y cuantización interna. HuBERT lo ataca de frente: **un paso de clustering offline (k-means) fabrica las etiquetas discretas alineadas trama a trama, y sobre ellas se entrena una pérdida de entropía cruzada estilo BERT aplicada solo a las regiones enmascaradas.** El paso de clustering se itera: la primera vuelta agrupa MFCCs, la segunda agrupa las representaciones latentes del modelo entrenado en la primera vuelta.

La tesis conceptual, que es lo más valioso del paper y está en el abstract: *"HuBERT relies primarily on the **consistency** of the unsupervised clustering step rather than the intrinsic **quality** of the assigned cluster labels."* Las etiquetas del maestro pueden ser malas —y de hecho lo son— mientras sean consistentes.

**Cifras ancla (Tabla II y Tabla III del paper).** WER (%) en LibriSpeech, todos los modelos HuBERT con LM Transformer salvo BASE que usa 4-gram:

| Datos etiquetados | Modelo | Preentrenamiento | test-clean | test-other |
|---|---|---|---|---|
| **10 min** | wav2vec 2.0 LARGE | LL-60k | 4.8 | 8.2 |
| **10 min** | HuBERT LARGE | LL-60k | **4.7** | **7.6** |
| **10 min** | HuBERT X-LARGE | LL-60k | **4.6** | **6.8** |
| **1 h** | wav2vec 2.0 LARGE | LL-60k | 2.9 | 5.8 |
| **1 h** | HuBERT LARGE | LL-60k | **2.9** | **5.4** |
| **1 h** | HuBERT X-LARGE | LL-60k | **2.8** | **4.8** |
| **10 h** | wav2vec 2.0 LARGE | LL-60k | 2.6 | 4.9 |
| **10 h** | HuBERT LARGE | LL-60k | **2.4** | **4.6** |
| **10 h** | HuBERT X-LARGE | LL-60k | **2.3** | **4.0** |
| **100 h** | wav2vec 2.0 LARGE | LL-60k | **2.0** | 4.0 |
| **100 h** | HuBERT LARGE | LL-60k | 2.1 | **3.9** |
| **100 h** | HuBERT X-LARGE | LL-60k | **1.9** | **3.5** |
| **960 h** | wav2vec 2.0 LARGE | LL-60k | **1.8** | 3.3 |
| **960 h** | HuBERT LARGE | LL-60k | 1.9 | 3.3 |
| **960 h** | HuBERT X-LARGE | LL-60k | **1.8** | **2.9** |

Lectura honesta de esa tabla: **la ganancia de HuBERT sobre wav2vec 2.0 es grande donde hay muy pocas etiquetas y en las particiones difíciles (`-other`), y se desvanece —o se invierte por 0.1 puntos— en las particiones limpias con muchas etiquetas.** Con 960 h etiquetadas, HuBERT LARGE y wav2vec 2.0 LARGE están empatados. Lo que separa a HuBERT es el modelo X-LARGE de 1B de parámetros, que baja test-other de 3.3 a 2.9.

El titular del abstract —"up to 19% and 13% relative WER reduction on dev-other and test-other"— corresponde al salto de HuBERT LARGE a HuBERT X-LARGE, no al salto sobre wav2vec 2.0. Verificado aritméticamente contra la Tabla II: el 19% es la fila de **100 h** en dev-other ($3.7 \to 3.0$, $-18.9\%$) y el 13% es la fila de **10 h** en test-other ($4.6 \to 4.0$, $-13.0\%$). Son configuraciones distintas; ver Sección 13.

Y la cifra que realmente importa para la clase 39: **con 10 minutos de audio transcrito y 60.000 horas sin transcribir, HuBERT X-LARGE alcanza 4.6% de WER en test-clean.** En 2019, un sistema supervisado entrenado con esas 10 minutos habría sido inutilizable.

## 2. Contexto: el aprendizaje autosupervisado del habla en 2020-2021

### 2.1. Qué había resuelto wav2vec 2.0 y qué quedó abierto

wav2vec 2.0 (Baevski, Zhou, Mohamed y Auli, 2020; referencia [6] del paper) fijó el molde arquitectónico que HuBERT hereda casi sin cambios: un **encoder convolucional sobre la onda cruda** que produce una secuencia de tramas latentes, un **encoder Transformer** que la contextualiza, y **enmascaramiento** sobre las salidas del encoder convolucional. La diferencia está en el objetivo.

wav2vec 2.0 usa una **pérdida contrastiva**: en cada posición enmascarada, hay que identificar la cuantización correcta de esa trama entre distractores muestreados de otras posiciones del mismo enunciado. Para que exista un "objetivo correcto" discreto, el modelo cuantiza internamente la salida convolucional con **Gumbel-softmax** y codebooks aprendidos conjuntamente.

El paper de HuBERT hace una crítica quirúrgica a ese diseño (Sección III, *Related Work*), y vale la pena leerla como lista de deudas técnicas:

1. La pérdida contrastiva **exige un diseño cuidadoso de dónde muestrear los negativos**. En texto, los negativos son otras palabras del vocabulario; en habla, si se muestrean del mismo enunciado se corre el riesgo de que el negativo sea acústicamente idéntico al positivo (mismo fonema, mismo hablante, 40 ms más allá), y si se muestrean de otros enunciados el problema se vuelve trivial porque basta identificar al hablante.
2. Requiere una **pérdida auxiliar de diversidad** para que el codebook no colapse a unas pocas entradas.
3. Requiere un **calendario de recocido de la temperatura de Gumbel-softmax** bien afinado.
4. Solo explora **cuantizar la salida del encoder convolucional**, que puede no ser el mejor lugar para cuantizar dada la capacidad limitada de esa red. HuBERT muestra empíricamente que las capas intermedias del Transformer son mucho mejores (Figura 2).

Los puntos 1-3 son de ingeniería: el objetivo contrastivo funciona pero es frágil. El punto 4 es conceptual y es el que HuBERT explota: **la mejor representación para cuantizar no está en el frontend, sino a media altura del Transformer.**

El otro antecedente directo es **DiscreteBERT** (Baevski, Auli y Mohamed, 2019; referencia [51]), que ya hacía predicción enmascarada de unidades discretas. Su problema: cuantiza la entrada y **le entrega al Transformer los tokens discretos**, no la onda. HuBERT insiste en que la entrada al Transformer debe ser continua "para pasar tanta información como sea posible" a las capas de atención, y que la cuantización debe existir **solo en el lado del objetivo**. Los números respaldan la insistencia: con 10 h etiquetadas, DiscreteBERT da 5.9/14.1 en test-clean/test-other contra 4.3/9.4 de HuBERT BASE, con el mismo objetivo formal.

La otra familia con la que compite es el **pseudo-etiquetado** (*self-training*): IPL, slimIPL, Noisy Student. La introducción del paper argumenta dos desventajas estructurales de esa familia: (a) el estudiante solo puede imitar a un maestro que está limitado por su cantidad de datos supervisados y por la calidad de la anotación, mientras que un pretexto autosupervisado obliga a representar **toda** la señal de entrada; (b) el pseudo-etiquetado orienta todo el aprendizaje hacia **una** tarea downstream, mientras que las features autosupervisadas generalizan a muchas. Este segundo punto es el que se volvió profético (ver Sección 11).

### 2.2. Los tres problemas específicos del habla

Esta es la parte del paper que hay que leer con la clase 39 al lado. El abstract enumera **tres problemas que el habla tiene y el texto no**:

> *"Self-supervised approaches for speech representation learning are challenged by three unique problems: (1) there are multiple sound units in each input utterance, (2) there is no lexicon of input sound units during the pre-training phase, and (3) sound units have variable lengths with no explicit segmentation."*

Desglosados, con la traducción a lo que rompen:

**(a) Hay múltiples unidades sonoras por enunciado.** Esto rompe el supuesto de *instance classification* que sostiene a casi todo el SSL de visión de la época (SimCLR, MoCo, BYOL, SwAV). En visión, la unidad de análisis es la imagen entera: se la aumenta dos veces y se pide que ambas vistas colapsen al mismo punto del espacio latente. Un enunciado de 5 segundos contiene ~50 fonemas de identidades distintas; pedir que todo el enunciado colapse a un punto destruye exactamente la información que se quiere aprender. La consecuencia es que el SSL de habla tiene que operar **a nivel de trama**, no a nivel de ejemplo.

**(b) No hay un léxico de unidades sonoras de entrada durante el preentrenamiento.** En NLP el tokenizador viene dado: WordPiece, BPE, SentencePiece. Se puede escribir `p(w | contexto)` con un softmax sobre 30.000 entradas porque existen las 30.000 entradas. En habla, la pregunta previa es *cuáles son las clases*. Sin un léxico, cualquier pérdida predictiva es imposible de formular: no hay sobre qué poner el softmax. Esto no es una dificultad de ingeniería, es una imposibilidad de escritura de la función objetivo.

**(c) Las unidades sonoras tienen longitud variable y sin segmentación explícita.** Un fonema puede durar 30 ms o 300 ms según el hablante, el contexto y la prosodia, y los límites entre fonemas son físicamente difusos: la coarticulación hace que la transición de `/s/` a `/t/` sea un continuo. Nadie escribe espacios en la señal. En texto, el enunciado llega presegmentado por convención ortográfica.

**Esta tercera dificultad es, palabra por palabra, la segunda objeción del PDF de la clase 39** ("*Self-attention mechanism operates over a finite sequence of discrete entities. In the context of text, sentence segmentation is trivial, but for audio this is not the case*"). La diferencia es de rol: en la clase aparece como razón para no usar Transformers en audio; en el paper aparece como el problema que el método viene a resolver. Volveremos a esto en la Sección 12.

## 3. La contribución central: el maestro de clustering offline y el argumento de la consistencia

### 3.1. La construcción

La idea, formalizada en la Sección II-A del paper. Sea $X = [x_1, \dots, x_T]$ un enunciado de $T$ tramas. Se aplica un **modelo de clustering** $h$ —k-means en el caso por defecto— que produce

$$h(X) = Z = [z_1, \dots, z_T], \qquad z_t \in [C]$$

donde $z_t$ es una variable categórica de $C$ clases. Eso es todo el "maestro": un k-means. No es una red, no se entrena con gradiente, no ve etiquetas. Se ajusta una vez, offline, y produce una etiqueta por trama.

La analogía que el paper propone es con el **alineamiento forzado** de los sistemas semi-supervisados clásicos: un modelo acústico entrenado con pares texto-habla produce etiquetas pseudo-fonéticas por trama vía forced alignment. HuBERT hace lo mismo, pero sin el texto: el maestro es un modelo de descubrimiento no supervisado de unidades acústicas. El paper cita la tradición previa (Lee y Glass 2012 con enfoques bayesianos no paramétricos, Ondel et al. 2016, los HMM-VAE de Ebbers et al. 2017) y observa que incluso los modelos más ingenuos —k-means, GMMs— *"infer hidden units that exhibit non-trivial correlation with the underlying acoustic units"*. Correlación no trivial: no es que k-means descubra fonemas, es que descubre algo que correlaciona con fonemas lo suficiente para que sirva de andamio.

Con $Z$ en mano, el problema queda escrito en el formato de BERT: hay una secuencia, hay un vocabulario de $C$ símbolos, se enmascara y se predice. **El aporte de HuBERT no es el objetivo (es el de BERT) ni la arquitectura (es la de wav2vec 2.0): es la fabricación del vocabulario.**

### 3.2. Por qué funciona con etiquetas malas: el argumento de la consistencia

Aquí está la parte que hay que entender bien, porque es contraintuitiva y porque es la que sostiene todo lo demás.

Una etiqueta k-means sobre MFCCs es **mala** en un sentido preciso y medible. La Sección V-C del paper reporta que el clustering de MFCC con $C=100$ alcanza (cluster purity, phone purity, PNMI) = (0.099, 0.335, 0.255). *Phone purity* de 0.335 significa que si se transcribiera cada clúster con su fonema más probable, se acertaría el 33.5% de las tramas. Dos tercios de las etiquetas están mal. Una tasa de error de etiquetado del 66% destruiría cualquier sistema supervisado.

El argumento del paper —enunciado en la introducción y demostrado en la Tabla V— es que **la pérdida enmascarada no necesita que las etiquetas sean correctas; necesita que sean una función determinista y estable de la acústica local.** El razonamiento, desarrollado:

Supongamos que $z_t = h(x_t)$ es una función determinista de la trama, cualquiera sea. Para predecir $z_t$ en una posición enmascarada, el modelo no tiene acceso a $x_t$; solo tiene el contexto $X \setminus \{x_t\}$. Entonces la tarea que el modelo enfrenta no es "clasificar la trama $t$", sino:

$$\text{estimar } p\big(h(x_t) \mid \text{contexto acústico de } t\big)$$

Esa cantidad solo se puede estimar bien si el modelo aprende dos cosas: (i) cómo se ve la señal alrededor —modelado acústico— y (ii) **qué restricciones impone la estructura secuencial del habla sobre lo que puede ocurrir en $t$** —modelado de lenguaje. Si en el contexto se oye `/k/ /a/ /s/` y luego `/a/`, la trama faltante está fuertemente restringida por la fonotáctica y el léxico del español, independientemente de cómo se llame la clase. El nombre de la clase es arbitrario; la **estructura de coocurrencia entre clases** no lo es.

De ahí que lo que se necesita del maestro sea **consistencia**: que la misma configuración acústica reciba siempre la misma etiqueta. Si $h$ es consistente, entonces $Z$ hereda de $X$ toda su estructura secuencial —las transiciones, las duraciones, la fonotáctica, la sintaxis en la medida en que se refleje en la cadena sonora— aunque las clases sean particiones arbitrarias del espacio acústico. Si $h$ fuera ruidoso (la misma trama etiquetada distinto en distintas ocurrencias), esa estructura se destruiría y no quedaría nada que aprender.

Un maestro perfectamente consistente pero "incorrecto" —por ejemplo, uno que parte cada fonema en tres subclases según el hablante— sigue definiendo un lenguaje formal sobre $[C]^*$ con estructura rica. Un maestro "correcto" pero ruidoso —uno que acierta el fonema el 90% de las veces al azar— define una cadena parcialmente aleatoria. El primero es mejor material de entrenamiento que el segundo.

El paper valida el supuesto de consistencia empíricamente antes de usarlo (Sección V-B, Tabla IV): ajusta k-means diez veces por configuración, con distintas semillas y distintos tamaños de datos de ajuste (1 h, 10 h, 100 h), y reporta media y desviación estándar del PNMI. Las desviaciones son minúsculas:

| Feature | $C$ | 1 h | 10 h | 100 h |
|---|---|---|---|---|
| MFCC | 100 | $0.251 \pm 0.001$ | $0.253 \pm 0.001$ | $0.253 \pm 0.001$ |
| MFCC | 500 | $0.283 \pm 0.001$ | $0.285 \pm 0.000$ | $0.287 \pm 0.001$ |
| BASE-it1-L6 | 100 | $0.563 \pm 0.012$ | $0.561 \pm 0.012$ | $0.575 \pm 0.008$ |
| BASE-it1-L6 | 500 | $0.680 \pm 0.005$ | $0.684 \pm 0.003$ | $0.686 \pm 0.004$ |

(Tabla IV. PNMI medido sobre dev-clean + dev-other combinados.)

Dos conclusiones prácticas que el paper extrae de ahí: k-means es **estable** —la desviación estándar es de tercer decimal—, y ajustar el k-means con más datos apenas ayuda (la ganancia máxima es 0.012 de PNMI). Lo segundo importa por una razón de ingeniería que el paper declara sin rodeos: la implementación carga toda la matriz de features en RAM, y las features del Transformer son de 768 dimensiones. Poder ajustar con 1 h en vez de 960 h es la diferencia entre viable e inviable.

### 3.3. La deuda con DeepCluster

El paper reconoce la inspiración: **DeepCluster** (Caron et al., ECCV 2018; referencia [23]), que en visión alterna entre agrupar las features de la red con k-means y usar las asignaciones como pseudo-etiquetas para entrenar la red. La diferencia declarada: *"HuBERT benefits from the masked prediction loss over speech sequences to represent their sequential structure."*

Esa diferencia no es cosmética. DeepCluster clasifica **la imagen completa** en su clúster; el gradiente empuja a que la red reproduzca la partición. HuBERT clasifica **una trama que la red no puede ver**, obligada a inferirla del contexto. En DeepCluster la red puede en el límite volverse un imitador perfecto del clustering, y el techo del método es el clustering mismo. En HuBERT ese modo de falla existe y tiene nombre: es exactamente lo que ocurre con $\alpha = 0$, y la Tabla V muestra que produce un WER de 96.37%. La máscara es lo que impide que el estudiante degenere en copia del maestro.

## 4. El proceso de dos pasos, iterado

### 4.1. Iteración 1: k-means sobre MFCC

Sección IV-B. Para generar las etiquetas de la primera iteración sobre las 960 h de LibriSpeech se corre k-means con **$C = 100$ clústeres sobre features MFCC de 39 dimensiones**: 13 coeficientes cepstrales más sus derivadas de primer y segundo orden. Es decir, el frontend de ASR de 1985.

La elección de k-means es deliberadamente humilde. El paper: *"It is one of the most naive unit discovery models that can be treated as modeling an isotropic Gaussian with the same scalar variance for each acoustic unit."* La humildad es parte del argumento: si funcionara solo con un maestro sofisticado, el aporte sería el maestro. Que funcione con k-means sobre MFCCs es la evidencia de que lo que importa es la pérdida enmascarada.

Detalles de implementación (los reproduzco porque son directamente accionables): `MiniBatchKMeans` de `scikit-learn`, minibatch de **10.000 tramas**, inicialización **k-means++ con 20 reinicios aleatorios**.

### 4.2. Iteración 2: k-means sobre las latentes del propio modelo

Sección II-D y IV-B. Terminada la iteración 1, se tiene un modelo HuBERT BASE preentrenado (no ajustado finamente). La premisa: *"since we expect a pre-trained model to provide better representations than the raw acoustic feature such as MFCCs, we can create a new generation of clusters by training a discrete latent model over the learned latent representations."*

Concretamente: se extraen las activaciones de la **6ª capa Transformer** del modelo de la iteración 1 y se corre k-means con **$C = 500$**. Como la dimensión es 768 y no cabe la matriz completa de 960 h en memoria, se **muestrea aleatoriamente el 10% de los datos** para ajustar el k-means. Esas etiquetas alimentan la iteración 2, entrenada por 400k pasos.

Por qué mejora, con los números al lado. Comparando el PNMI del maestro (Tabla V, medido sobre las mismas condiciones):

| Maestro | $C$ | PNMI | dev-other WER con $\alpha=1$ |
|---|---|---|---|
| k-means sobre MFCC | 100 | 0.243 | 17.86 |
| k-means sobre BASE-it1, capa 6 | 500 | 0.637 | 11.91 |
| k-means sobre BASE-it2, capa 9 | 500 | 0.704 | 10.75 |
| Chenone (alineamiento forzado supervisado) | 8976 | 0.809 | 10.38 |

El PNMI del maestro salta de 0.243 a 0.637 en una sola iteración: **el modelo entrenado con etiquetas malas produce representaciones que, agrupadas, dan etiquetas mucho mejores que las que lo entrenaron.** Esa es la definición operativa de bootstrapping, y es lo mismo que ocurre en el pseudo-etiquetado iterativo (referencias [12] y [54] del paper), con la diferencia de que aquí no hay ninguna etiqueta humana en el ciclo.

Hay una asimetría que explica por qué el ciclo puede ser virtuoso y no viciosos. El maestro k-means es **memoryless**: mira una trama (o un vector de features local) y la asigna a un centroide. El modelo HuBERT es **contextual**: para producir la latente de la capa 6 en la posición $t$ ha integrado, vía self-attention, todo el enunciado. Al agrupar esas latentes contextuales, los clústeres resultantes ya no son "regiones del espacio espectral" sino "regiones del espacio de estados fonéticos contextualizados". El clustering hereda gratis el trabajo de desambiguación que hizo el Transformer.

### 4.3. La tercera iteración implícita de LARGE y X-LARGE

Sección IV-C, un detalle que suele citarse mal. Los modelos **LARGE y X-LARGE no reinician el proceso desde MFCCs**. En vez de eso:

> *"Instead of restarting the iterative process from clustering MFCC features, we extract features from the 9-th transformer layer of the second iteration BASE HuBERT for clustering and use those labels for training these two models. Hence, these two models can also be seen as the third iteration models."*

Es decir: la cadena completa es MFCC → BASE-it1 (250k pasos, LS-960, 32 GPUs) → k-means sobre capa 6 → BASE-it2 (400k pasos, LS-960) → k-means sobre capa 9 con $C=500$ → LARGE (400k pasos, LL-60k, 128 GPUs) y X-LARGE (400k pasos, LL-60k, 256 GPUs), ambos entrenados **en paralelo desde el mismo conjunto de etiquetas**.

Consecuencias que conviene tener presentes:

- Los modelos entrenados sobre 60.000 horas usan etiquetas producidas por un modelo entrenado sobre 960 horas. El maestro es más chico y ha visto 60× menos audio que el estudiante.
- La capa de extracción cambia de 6 (para it1) a 9 (para it2). No es arbitrario: la Figura 2 muestra que el perfil de calidad por capa se desplaza hacia arriba entre iteraciones (Sección 9).
- LARGE y X-LARGE hacen **una sola** pasada de preentrenamiento cada uno. Todo el costo iterativo está amortizado en BASE.

## 5. La función de pérdida

### 5.1. Formulación

Sección II-B. Sea $M \subset [T]$ el conjunto de índices enmascarados y $\tilde{X} = r(X, M)$ la versión corrompida de $X$ donde cada $x_t$ con $t \in M$ se reemplaza por un **embedding de máscara aprendido** $\tilde{x}$. El modelo $f$ consume $\tilde{X}$ y produce una distribución $p_f(\cdot \mid \tilde{X}, t)$ en cada paso.

La pérdida sobre posiciones enmascaradas, tal cual la escribe el paper en la ecuación (1):

$$L_m(f; X, M, Z) = \sum_{t \in M} \log p_f(z_t \mid \tilde{X}, t)$$

y $L_u$ es idéntica salvo que suma sobre $t \notin M$. (El paper llama a ambas "cross-entropy loss" pero las escribe sin el signo negativo; ver Sección 13.) La pérdida final:

$$L = \alpha L_m + (1 - \alpha) L_u$$

### 5.2. La distribución sobre códigos

Sección II-E, ecuación (3). Dada la salida del Transformer $o_t$, la distribución sobre los $C$ códigos se parametriza como

$$p_f^{(k)}(c \mid \tilde{X}, t) = \frac{\exp\big(\text{sim}(A^{(k)} o_t,\, e_c)/\tau\big)}{\sum_{c'=1}^{C} \exp\big(\text{sim}(A^{(k)} o_t,\, e_{c'})/\tau\big)}$$

donde $A^{(k)}$ es la matriz de proyección, $e_c$ es el embedding aprendido del código $c$, $\text{sim}(\cdot, \cdot)$ es la **similitud coseno** y $\tau = 0.1$ escala los logits. Cuando se usan ensembles de clustering, hay una matriz $A^{(k)}$ por cada modelo $k$.

Dos decisiones no obvias en esa ecuación, que la separan de un softmax lineal estándar:

- **Similitud coseno en vez de producto punto.** Al normalizar ambos vectores, la magnitud de $A o_t$ deja de influir en los logits; solo importa la dirección. Esto elimina un grado de libertad degenerado (inflar la norma de la salida para saturar el softmax) y da un objetivo de *metric learning*: los estados ocultos deben apuntar hacia el embedding de su código. Es el mismo mecanismo del clasificador coseno que se usa en aprendizaje con pocos ejemplos y en reconocimiento facial.
- **Embeddings de código aprendidos $e_c$ en vez de una capa de salida arbitraria.** La capa de salida es literalmente una tabla de embeddings de $C$ entradas, con la misma forma que la tabla de entrada de BERT. Consecuencia útil: **el espacio de códigos adquiere geometría**. Dos clústeres k-means que aparecen en contextos parecidos terminan con embeddings parecidos, lo que suaviza el objetivo: equivocarse entre dos códigos vecinos cuesta menos que equivocarse entre dos lejanos. Con un maestro cuyo *phone purity* es 0.335, esa tolerancia importa mucho.

Con $\tau = 0.1$, la similitud coseno vive en $[-1, 1]$ y los logits en $[-10, 10]$: rango suficiente para distribuciones agudas sin permitir saturación patológica.

### 5.3. El análisis de $\alpha$: los dos extremos

El paper interpreta $\alpha$ como una perilla entre dos regímenes clásicos del reconocimiento de habla, y la interpretación es exacta:

**$\alpha = 0$ (pérdida solo sobre lo no enmascarado).** El modelo ve $x_t$ y tiene que predecir $z_t = h(x_t)$. Como $h$ es determinista, la tarea es aprender la función $h$. El paper: *"this limits the learning process to mimicking the clustering model."* Es análogo al modelado acústico en un sistema híbrido HMM-DNN: mapear tramas a estados. El techo del método es el maestro.

**$\alpha = 1$ (pérdida solo sobre lo enmascarado).** El modelo no ve $x_t$ y tiene que inferir $z_t$ del contexto. El paper: *"analogous to language modeling. It forces the model to learn both the acoustic representation of unmasked segments and the long-range temporal structure of the speech data."* Nótese la doble exigencia: hay que representar bien lo que sí se ve (acústica) **y** modelar cómo se encadena (lenguaje). Un solo objetivo produce ambos modelos.

La hipótesis explícita: *"We hypothesize that the setup with $\alpha = 1$ is more resilient to the quality of cluster targets."*

**Los valores intermedios.** Con $\alpha = 0.5$ ambos términos pesan igual. El paper no lo dice, pero la asimetría de tamaño importa: con ~50% de tramas enmascaradas, $|M| \approx |M^c|$, así que $\alpha = 0.5$ pondera aproximadamente igual **por trama** dos tareas de dificultad radicalmente distinta. Predecir $h(x_t)$ viendo $x_t$ es fácil y su gradiente es grande y limpio; inferirlo del contexto es difícil. El término fácil domina el aprendizaje temprano y arrastra al modelo hacia la imitación del maestro.

Los resultados (Tabla V) confirman todo esto con una limpieza poco común:

| Maestro | $C$ | PNMI | $\alpha = 1.0$ | $\alpha = 0.5$ | $\alpha = 0.0$ |
|---|---|---|---|---|---|
| Chenone (top-line supervisado) | 8976 | 0.809 | 10.38 | **9.16** | 9.79 |
| k-means sobre MFCC | 50 | 0.227 | **18.68** | 31.07 | 94.60 |
| k-means sobre MFCC | 100 | 0.243 | **17.86** | 29.57 | 96.37 |
| k-means sobre MFCC | 500 | 0.276 | **18.40** | 33.42 | 97.66 |
| k-means sobre BASE-it1, capa 6 | 500 | 0.637 | **11.91** | 13.47 | 23.29 |
| k-means sobre BASE-it2, capa 9 | 500 | 0.704 | **10.75** | 11.59 | 13.79 |

(dev-other WER %. Modelos preentrenados 100k pasos y ajustados sobre el split de 10 h de Libri-Light.)

Lo que hay que leer en esa tabla:

1. **Con maestros malos, $\alpha = 0$ es catastrófico: 94-98% de WER.** No es "peor", es colapso total. El modelo aprendió a ser un k-means sobre MFCCs y no aprendió nada de habla.
2. **Con maestros malos, $\alpha = 1$ es el único régimen viable**, y la degradación al pasar a $\alpha = 0.5$ ya es brutal ($17.86 \to 29.57$).
3. **La penalización por incluir la pérdida no enmascarada decrece monótonamente con la calidad del maestro.** La brecha $\alpha=0$ menos $\alpha=1$ vale $+78.5$ para MFCC-100, $+11.4$ para it1-L6, $+3.0$ para it2-L9 y $-0.6$ para chenone. Con un maestro suficientemente bueno, imitarlo deja de ser un error.
4. **Con el maestro supervisado, $\alpha = 0.5$ gana.** Cuando las etiquetas son casi correctas, el término no enmascarado deja de ser un distractor y pasa a ser señal densa útil.
5. La distancia entre el mejor maestro no supervisado ($10.75$) y el top-line supervisado con su mejor $\alpha$ ($9.16$) es de **1.59 puntos de WER absolutos**. Ese es el precio total de no tener transcripciones en el ciclo.

El paper **no declara explícitamente qué $\alpha$ usan los modelos principales**, pero del abstract (*"applying the prediction loss over the masked regions only"*) y de esta tabla se sigue que es $\alpha = 1$.

### 5.4. Ensembles de clustering

Sección II-C. La extensión formal es directa: si $Z^{(k)}$ son las etiquetas del $k$-ésimo modelo de clustering,

$$L_m\big(f; X, \{Z^{(k)}\}_k, M\big) = \sum_{t \in M} \sum_{k} \log p_f^{(k)}\big(z_t^{(k)} \mid \tilde{X}, t\big)$$

Es *multi-task learning* con tareas creadas por clustering no supervisado. Cada modelo aporta una granularidad distinta: el paper menciona que un ensemble de k-means con distintos tamaños de codebook cubre desde clases de modo de articulación (vocal/consonante) hasta subestados fonéticos (senones).

El ensemble se combina bien con **cuantización por producto** (PQ): se parte el espacio de features en subespacios y se cuantiza cada uno por separado, con lo cual el tamaño teórico del espacio de objetivos es el producto de los tamaños de codebook. Esto resuelve un problema real de k-means en alta dimensión con features heterogéneas de escalas muy distintas —justo el caso de los MFCC empalmados, donde los coeficientes de orden cero, primero y segundo tienen magnitudes incomparables.

Resultados (Tabla VI, dev-other WER):

| Maestro | WER |
|---|---|
| k-means {50, 100} | 17.81 |
| k-means {50, 100, 500} | 17.56 |
| Product k-means-0-100 | 19.26 |
| Product k-means-1-100 | 17.64 |
| Product k-means-2-100 | 18.46 |
| Product k-means-{0,1,2}-100 | **16.73** |

Contra el mejor k-means simple sobre MFCC de la Tabla V ($C=100$, $17.86$), el ensemble de las tres cuantizaciones por producto baja a $16.73$: **1.13 puntos**. Detalle interesante: por separado, la cuantización del subespacio de derivadas primeras ($17.64$) es mejor que la de los coeficientes estáticos ($19.26$) — la información dinámica agrupa mejor que la estática. Y las tres juntas superan a cualquiera sola.

**Los modelos principales del paper no usan ensembles.** Toda la Tabla II y la Tabla III se obtienen con un único k-means. La técnica queda documentada como ablación, no desplegada.

## 6. La arquitectura

### 6.1. Encoder convolucional de forma de onda

Idéntico para las tres configuraciones (Tabla I): **7 capas convolucionales de 512 canales**, con

- strides: $[5, 2, 2, 2, 2, 2, 2]$
- anchos de kernel: $[10, 3, 3, 3, 3, 2, 2]$

El producto de los strides es $5 \cdot 2^6 = 320$. A 16 kHz, eso da **una trama cada 20 ms, o sea 50 tramas por segundo**, cifra que el paper declara explícitamente en la Sección II-E.

El **campo receptivo** no lo da el paper; se deriva de los strides y kernels con la recursión estándar $r_i = r_{i-1} + (k_i - 1)\prod_{j<i} s_j$:

| Capa | $k$ | $s$ | Campo receptivo (muestras) | Salto acumulado |
|---|---|---|---|---|
| 1 | 10 | 5 | 10 | 5 |
| 2 | 3 | 2 | 20 | 10 |
| 3 | 3 | 2 | 40 | 20 |
| 4 | 3 | 2 | 80 | 40 |
| 5 | 3 | 2 | 160 | 80 |
| 6 | 2 | 2 | 240 | 160 |
| 7 | 2 | 2 | **400** | 320 |

**400 muestras = 25 ms** a 16 kHz. Es exactamente el tamaño de ventana canónico del análisis de habla (25 ms de ventana, 10 ms de salto en un frontend MFCC clásico; aquí el salto es 20 ms). El frontend convolucional reproduce, aprendiéndolos, los parámetros de ventaneo que la comunidad de DSP fijó por razones psicoacústicas: 25 ms es aproximadamente el rango donde la señal puede considerarse cuasi-estacionaria y donde cabe al menos un par de períodos de pitch de una voz grave.

Este bloque es también, y esto es central para la Sección 12, **la respuesta al costo cuadrático de la self-attention**: 10 segundos de audio son 160.000 muestras, pero solo 500 tramas después del encoder. Sin ese factor de 320×, un Transformer sobre habla sería literalmente imposible.

Durante el fine-tuning de ASR el encoder convolucional **se congela** (Sección II-E y IV-D).

### 6.2. Encoder Transformer y cabezal

| | BASE | LARGE | X-LARGE |
|---|---|---|---|
| **CNN — strides** | 5, 2, 2, 2, 2, 2, 2 | ídem | ídem |
| **CNN — kernels** | 10, 3, 3, 3, 3, 2, 2 | ídem | ídem |
| **CNN — canales** | 512 | 512 | 512 |
| **Transformer — capas** | 12 | 24 | 48 |
| **Transformer — dim. embedding** | 768 | 1024 | 1280 |
| **Transformer — dim. FFN interna** | 3072 | 4096 | 5120 |
| **Transformer — prob. de layerdrop** | 0.05 | 0 | 0 |
| **Transformer — cabezas de atención** | 8 | 16 | 16 |
| **Proyección — dim.** | 256 | 768 | 1024 |
| **Parámetros** | **95M** | **317M** | **964M** |
| **Datos de preentrenamiento** | LS-960 | LL-60k | LL-60k |
| **GPUs / pasos** | 32 / 250k + 400k | 128 / 400k | 256 / 400k |
| **Batch por GPU** | ≤ 87.5 s de audio | ≤ 56.25 s | ≤ 22.5 s |
| **LR pico** | 5e-4 | 1.5e-3 | 3e-3 |

(Tabla I y Sección IV-C. La introducción cita 90M / 300M / 1B; la Tabla I es la fuente precisa.)

BASE y LARGE siguen de cerca a wav2vec 2.0 BASE y LARGE. X-LARGE es original de este paper y se dimensiona a la escala del **Conformer XXL** de Zhang et al. (referencia [40]), que era el competidor de escala en 2020-2021. Nótese que X-LARGE duplica la profundidad de LARGE (48 capas) pero solo aumenta la dimensión de 1024 a 1280 y mantiene 16 cabezas: **el escalado es casi puramente en profundidad**. Con 48 capas y una tasa de layerdrop de 0, el entrenamiento a LR pico 3e-3 es agresivo; el batch por GPU cae a 22.5 s por presión de memoria, así que el batch efectivo con 256 GPUs es de ~5760 s = 1.6 h de audio por paso.

Sobre el **layerdrop 0.05 solo en BASE**: es regularización estocástica de profundidad. Que se apague en LARGE y X-LARGE es coherente con el régimen de datos: BASE se entrena sobre 960 h y necesita regularización; LARGE y X-LARGE ven 60.000 h y el sobreajuste deja de ser el cuello de botella.

**Cabezal.** Tras el Transformer viene la capa de proyección $A$ (de dimensión 256/768/1024 según el tamaño) y el embedding de código $e_c$, con la parametrización coseno de la Sección 5.2. Para el fine-tuning de ASR se **elimina la proyección** y se la reemplaza por un softmax inicializado al azar, entrenado con **CTC**. El vocabulario de CTC es minimalista: **26 letras del alfabeto inglés, un token de espacio, un apóstrofo y el símbolo blank de CTC** — 29 símbolos. No hay tokenizador subpalabra, no hay léxico de pronunciación, no hay diccionario fonético.

Un detalle de la arquitectura que el paper **no describe** y conviene tener en cuenta: la codificación posicional. El paper dice solo que sigue la arquitectura de wav2vec 2.0, la cual usa una **capa convolucional de embedding posicional relativo** (kernel 128, 16 grupos) en vez de codificaciones sinusoidales o absolutas aprendidas. Es una elección relevante para el argumento sobre dependencias largas: no hay una longitud máxima codificada en el modelo.

### 6.3. Decodificación

Sección IV-D. Se usa el decodificador de búsqueda por haz de **wav2letter++** envuelto en fairseq, optimizando

$$\log p_{\text{CTC}}(Y \mid X) + w_1 \log P_{\text{LM}}(Y) + w_2 |Y|$$

con $w_1$ el peso del modelo de lenguaje y $w_2$ el *word score*. Los hiperparámetros de decodificación se buscan con **Ax** (la caja de optimización bayesiana de Facebook). Se consideran LM de n-gramas y LM Transformer, ambos entrenados sobre los datos oficiales de modelado de lenguaje de LibriSpeech.

Esto es importante para interpretar las cifras titulares y volveré sobre ello: **el 4.6% de WER con 10 minutos de audio etiquetado usa un LM Transformer entrenado sobre el corpus de texto de LibriSpeech (~800M de palabras).** La supervisión acústica es de 10 minutos; la supervisión lingüística es enorme.

## 7. La estrategia de enmascaramiento

Sección II-B y IV-C. HuBERT adopta el esquema de **SpanBERT** y wav2vec 2.0: se seleccionan aleatoriamente el $p\%$ de los pasos temporales como **índices de inicio**, y desde cada uno se enmascara un **span de $l$ pasos**. Los valores por defecto: $l = 10$ y $p = 8\%$.

Dos cosas que conviene precisar porque casi siempre se citan mal:

**$p$ no es la fracción enmascarada, es la probabilidad de inicio.** Con spans de largo 10 que pueden solaparse, la fracción esperada de tramas cubiertas es aproximadamente

$$1 - (1 - p)^{l} = 1 - 0.92^{10} \approx 0.57$$

Es decir, **alrededor del 50-57% de las tramas quedan enmascaradas**, no el 8%. (El paper no reporta esta cifra derivada; la fórmula reproduce el ~49% que wav2vec 2.0 declara con su $p = 6.5\%$, lo que valida la aproximación.) Comparado con el 15% de BERT, es una tasa de enmascaramiento enorme, y la razón es la redundancia de la señal: tramas contiguas de 20 ms dentro del mismo fonema son casi copias.

**El span de 10 tramas cubre 200 ms.** A 20 ms por trama, un span borra el equivalente a dos o tres fonemas completos. Esto es esencial: si se enmascarara una sola trama, la tarea sería trivial por interpolación local — el modelo copiaría el vecino. Los spans largos hacen que la interpolación acústica sea insuficiente y que haya que apoyarse en estructura de nivel superior. Es el mismo razonamiento por el que SpanBERT enmascara spans en vez de tokens sueltos, amplificado por la redundancia del audio.

La ablación (Figura 3, izquierda) barre $p \in \{2, 4.5, 6.5, 8, 9\}$ y encuentra el óptimo en **$p = 8\%$**, con el WER subiendo apreciablemente hacia ambos extremos del rango explorado. Poco enmascaramiento da una tarea demasiado fácil; demasiado destruye el contexto necesario para resolverla.

**Dónde se enmascara: sobre las salidas del encoder convolucional, no sobre la onda cruda.** El paper lo declara en la Sección II-E (*"The audio encoded features are then randomly masked"*) pero no argumenta la elección. Las razones son varias y vale la pena tenerlas explícitas:

1. **Alineación con el objetivo.** Las etiquetas $z_t$ están definidas por trama, a 50 Hz. Enmascarar en el dominio de la onda dejaría los límites de máscara desalineados respecto de las tramas objetivo, y el campo receptivo de 25 ms produciría tramas parcialmente contaminadas en los bordes.
2. **Fuga de información.** Poner ceros en la onda cruda no borra la información: **crea una discontinuidad brutal que el encoder convolucional detectaría trivialmente**, y peor, la envolvente de energía alrededor del silencio artificial revela mucho sobre el contexto. Un embedding de máscara aprendido, sustituido a nivel de trama, es una señal limpia de "aquí no hay dato".
3. **El embedding de máscara tiene que vivir en el mismo espacio que las tramas.** En BERT, `[MASK]` es una entrada del vocabulario. Aquí, $\tilde{x}$ es un vector aprendido del mismo tamaño que la salida convolucional. Ese objeto no tiene análogo en el dominio de la señal: no existe "la forma de onda de la máscara".
4. **El encoder convolucional está congelado en fine-tuning** y no recibe la máscara de la misma manera durante ambas fases; mantener el enmascaramiento por encima de él lo deja como un frontend puro y estable.

La consecuencia conceptual es que **el encoder convolucional es el que fabrica las "entidades discretas" sobre las que opera la self-attention.** Este punto es directamente relevante para la objeción 2 de la clase (Sección 12).

## 8. Experimentos y resultados

### 8.1. Datos

**Preentrenamiento (sin etiquetas):** las 960 h completas de LibriSpeech, o las **60.000 h de Libri-Light**. Ambos derivan de LibriVox: grabaciones en inglés de audiolibros de dominio público leídos por voluntarios de internet.

**Fine-tuning (con etiquetas):** cinco particiones — Libri-Light de **10 minutos, 1 hora y 10 horas**, más LibriSpeech **100 h** (`train-clean-100`) y **960 h** (`train-clean-100` + `train-clean-360` + `train-other-500`). Detalle metodológico que importa: los tres splits de Libri-Light son subconjuntos del split de entrenamiento de LibriSpeech, y **cada uno tiene la mitad del audio de `train-clean-*` y la otra mitad de `train-other-500`**. O sea, los regímenes de bajos recursos están balanceados entre condiciones limpia y difícil por construcción, no son "10 minutos de audio fácil".

**Selección de modelo:** se barren LR pico, calendario de LR, número de pasos, `freeze-step` (cuántos pasos se mantiene congelado el Transformer entrenando solo el softmax nuevo) y probabilidad de enmascaramiento, usando el **WER de dev-other** como criterio, para cada combinación de tamaño de modelo y split. Es un barrido considerable, y conviene registrarlo: los números de la Tabla II son el resultado de una búsqueda de hiperparámetros por celda.

### 8.2. Regímenes de bajos recursos (Tabla II)

| Modelo | Datos sin etiquetar | LM | dev-clean | dev-other | test-clean | test-other |
|---|---|---|---|---|---|---|
| **10 minutos etiquetados** | | | | | | |
| DiscreteBERT | LS-960 | 4-gram | 15.7 | 24.1 | 16.3 | 25.2 |
| wav2vec 2.0 BASE | LS-960 | 4-gram | 8.9 | 15.7 | 9.1 | 15.6 |
| wav2vec 2.0 LARGE | LL-60k | 4-gram | 6.3 | 9.8 | 6.6 | 10.3 |
| wav2vec 2.0 LARGE | LL-60k | Transformer | 4.6 | 7.9 | 4.8 | 8.2 |
| HuBERT BASE | LS-960 | 4-gram | 9.1 | **15.0** | 9.7 | **15.3** |
| HuBERT LARGE | LL-60k | 4-gram | **6.1** | **9.4** | 6.6 | **10.1** |
| HuBERT LARGE | LL-60k | Transformer | **4.3** | **7.0** | **4.7** | **7.6** |
| HuBERT X-LARGE | LL-60k | Transformer | 4.4 | **6.1** | **4.6** | **6.8** |
| **1 hora etiquetada** | | | | | | |
| DeCoAR 2.0 | LS-960 | 4-gram | – | – | 13.8 | 29.1 |
| DiscreteBERT | LS-960 | 4-gram | 8.5 | 16.4 | 9.0 | 17.6 |
| wav2vec 2.0 BASE | LS-960 | 4-gram | **5.0** | **10.8** | **5.5** | 11.3 |
| wav2vec 2.0 LARGE | LL-60k | Transformer | 2.9 | 5.4 | **2.9** | 5.8 |
| HuBERT BASE | LS-960 | 4-gram | 5.6 | 10.9 | 6.1 | 11.3 |
| HuBERT LARGE | LL-60k | Transformer | **2.6** | **4.9** | **2.9** | **5.4** |
| HuBERT X-LARGE | LL-60k | Transformer | **2.6** | **4.2** | **2.8** | **4.8** |
| **10 horas etiquetadas** | | | | | | |
| SlimIPL | LS-960 | 4-gram + Transformer | 5.3 | 7.9 | 5.5 | 9.0 |
| DeCoAR 2.0 | LS-960 | 4-gram | – | – | 5.4 | 13.3 |
| DiscreteBERT | LS-960 | 4-gram | 5.3 | 13.2 | 5.9 | 14.1 |
| wav2vec 2.0 BASE | LS-960 | 4-gram | **3.8** | 9.1 | 4.3 | 9.5 |
| wav2vec 2.0 LARGE | LL-60k | Transformer | 2.4 | 4.8 | 2.6 | 4.9 |
| HuBERT BASE | LS-960 | 4-gram | 3.9 | **9.0** | 4.3 | **9.4** |
| HuBERT LARGE | LL-60k | Transformer | **2.2** | **4.3** | **2.4** | **4.6** |
| HuBERT X-LARGE | LL-60k | Transformer | **2.1** | **3.6** | **2.3** | **4.0** |
| **100 horas etiquetadas** | | | | | | |
| IPL | LL-60k | 4-gram + Transformer | 3.19 | 6.14 | 3.72 | 7.11 |
| SlimIPL | LS-860 | 4-gram + Transformer | 2.2 | 4.6 | 2.7 | 5.2 |
| Noisy Student | LS-860 | LSTM | 3.9 | 8.8 | 4.2 | 8.6 |
| DeCoAR 2.0 | LS-960 | 4-gram | – | – | 5.0 | 12.1 |
| DiscreteBERT | LS-960 | 4-gram | 4.0 | 10.9 | 4.5 | 12.1 |
| wav2vec 2.0 BASE | LS-960 | 4-gram | 2.7 | 7.9 | 3.4 | **8.0** |
| wav2vec 2.0 LARGE | LL-60k | Transformer | 1.9 | 4.0 | **2.0** | 4.0 |
| HuBERT BASE | LS-960 | 4-gram | 2.7 | **7.8** | 3.4 | 8.1 |
| HuBERT LARGE | LL-60k | Transformer | **1.8** | **3.7** | 2.1 | **3.9** |
| HuBERT X-LARGE | LL-60k | Transformer | **1.7** | **3.0** | **1.9** | **3.5** |

**Dónde la ganancia es grande.** En el régimen de 10 minutos con LM Transformer, HuBERT LARGE baja test-other de 8.2 a 7.6 (−7.3% relativo) y X-LARGE lo lleva a 6.8 (−17.1% respecto de wav2vec 2.0 LARGE). En 1 hora, test-other pasa de 5.8 a 4.8 con X-LARGE (−17.2%). En 10 horas, de 4.9 a 4.0 (−18.4%). La mejora se concentra sistemáticamente en **`-other`**, que es la partición con hablantes y condiciones de grabación más difíciles.

**Dónde la ganancia es marginal o negativa.** HuBERT **BASE no supera a wav2vec 2.0 BASE** en los regímenes muy bajos: en 10 minutos es 0.6 puntos peor en test-clean (9.7 vs 9.1); en 1 hora es peor en tres de las cuatro columnas (5.6 vs 5.0, 10.9 vs 10.8, 6.1 vs 5.5) y empata en la cuarta. En 10 h y 100 h BASE empata. Es decir: **con el mismo presupuesto de datos y parámetros (LS-960, ~95M), el objetivo de HuBERT no es mejor que el contrastivo de wav2vec 2.0.** La ventaja aparece al escalar a LARGE/X-LARGE con 60k horas.

Esto es una lectura importante y el paper la deja mal contada (ver Sección 13). No invalida el método: lo que dice es que HuBERT **escala mejor**, no que sea uniformemente superior.

**La comparación con DiscreteBERT** sí es aplastante y es la comparación conceptualmente limpia, porque el objetivo formal es idéntico. Con 10 h: 5.9/14.1 contra 4.3/9.4. Con 10 minutos: 16.3/25.2 contra 9.7/15.3. El paper atribuye la brecha a dos causas: (i) **la entrada debe ser la onda, no unidades cuantizadas** —cuantizar la entrada pierde información irrecuperable—; y (ii) aunque vq-wav2vec (que provee las unidades de DiscreteBERT) descubra mejores unidades que k-means sobre MFCC, **el refinamiento iterativo termina superándolo** porque el maestro mejora con el estudiante mientras que el de DiscreteBERT es fijo.

### 8.3. Régimen de altos recursos (Tabla III)

| Categoría | Modelo | Datos sin etiquetar | LM | dev-clean | dev-other | test-clean | test-other |
|---|---|---|---|---|---|---|---|
| Supervisado | Conformer L | – | LSTM | – | – | 1.9 | 3.9 |
| Self-training | IPL | LL-60k | 4-gram + Transformer | 1.85 | 3.26 | 2.10 | 4.01 |
| Self-training | Noisy Student | LV-60k | LSTM | 1.6 | 3.4 | 1.7 | 3.4 |
| Pre-training | wav2vec 2.0 LARGE | LL-60k | Transformer | 1.6 | 3.0 | 1.8 | 3.3 |
| Pre-training | Conformer XXL | LL-60k | LSTM | 1.5 | 3.0 | 1.5 | 3.1 |
| Pre + self-training | wav2vec 2.0 + self-training | LL-60k | Transformer | 1.1 | 2.7 | 1.5 | 3.1 |
| Pre + self-training | Conformer XXL + Noisy Student | LL-60k | LSTM | 1.3 | 2.6 | 1.4 | **2.6** |
| **Este trabajo** | HuBERT LARGE | LL-60k | Transformer | 1.5 | 3.0 | 1.9 | 3.3 |
| **Este trabajo** | HuBERT X-LARGE | LL-60k | Transformer | 1.5 | **2.5** | 1.8 | 2.9 |

El paper resume con precisión: HuBERT supera a los métodos supervisados y de self-training, **está a la par de los dos mejores resultados de solo preentrenamiento** (ambos basados en el contrastivo de wav2vec 2.0), y **queda detrás de los métodos que combinan preentrenamiento con self-training**. La conjetura razonable que ofrecen —apoyada en las referencias [63] y [40]— es que HuBERT combinado con self-training debería alcanzar o superar a esos, ya que el modelo preentrenado de partida es igual o mejor.

Nótese que HuBERT X-LARGE logra el **mejor dev-other absoluto de la tabla (2.5)**, incluyendo a los combinados. En test-other queda 0.3 detrás del Conformer XXL + Noisy Student.

### 8.4. El efecto del número de pasos (Tabla VII)

| Maestro | $C$ | 100k | 250k | 400k | 800k |
|---|---|---|---|---|---|
| k-means MFCC | 50 | 18.68 | 13.65 | 12.40 | 11.82 |
| k-means MFCC | 100 | 17.86 | 12.97 | 12.32 | **11.68** |
| DiscreteBERT | 13.5k | – | 26.6 | – | – |

(dev-other WER; $p = 6.5\%$ en esta tabla.)

Dos hechos. Primero, **entrenar más ayuda de forma consistente y no saturado**: de 100k a 800k pasos el WER cae de 17.86 a 11.68, un 35% relativo, y la curva todavía baja al final. Segundo, la comparación directa con DiscreteBERT a 250k pasos (12.97 contra 26.6) es un factor de 2 con el mismo objetivo formal y las mismas features MFCC de partida.

El paper agrega una hipótesis sobre por qué DiscreteBERT falla tanto: usa **13.500 unidades** para cuantizar los mismos MFCC. Con tantos clústeres, las unidades codifican variación inter e intra-hablante en vez de conceptos fonéticos amplios. HuBERT con 100 o 500 clústeres captura *"broad phonetic concepts without delving into inter/intra-speaker variation"*. Esto conecta con el punto de la Sección 9 sobre por qué más $C$ no es mejor.

La ablación de tamaño de batch (Figura 3, derecha) barre 8, 16 y 32 GPUs y muestra una mejora muy fuerte al aumentar el batch efectivo — el eje vertical va de ~40% a menos de 20% de WER en el rango explorado. Es el hallazgo estándar de los modelos tipo BERT y el paper lo declara como tal.

## 9. Ablations y análisis

### 9.1. Las métricas de calidad del maestro

Sección IV-E. Para medir la correlación entre los clústeres y la fonética real, se derivan transcripciones fonéticas alineadas trama a trama con un sistema ASR híbrido, y se estima la distribución conjunta entre etiquetas fonéticas $y$ y etiquetas k-means $z$ por conteo:

$$p_{yz}(i, j) = \frac{\sum_{t=1}^{T} [\,y_t = i \wedge z_t = j\,]}{T}$$

Sobre esa conjunta se definen tres métricas.

**Pureza de fonema** (*phone purity*): para cada clúster $j$, se toma el fonema más probable $y^*(j) = \arg\max_i p_{yz}(i,j)$ y se promedia su probabilidad condicional:

$$\text{PhnPur} = \mathbb{E}_{p_z(j)}\big[\,p_{y|z}(y^*(j) \mid j)\,\big]$$

Es la **exactitud de fonema a nivel de trama** si se transcribiera cada clúster con su fonema más probable. Caveat que el propio paper señala: **no es comparable entre configuraciones con distinto número de unidades**, porque en el límite degenerado en que cada trama recibe una etiqueta única la pureza sería del 100%.

**Pureza de clúster** (*cluster purity*): la contraparte, para cada fonema $i$ se toma su clúster más probable $z^*(i)$:

$$\text{ClsPur} = \mathbb{E}_{p_y(i)}\big[\,p_{z|y}(z^*(i) \mid i)\,\big]$$

Mide cuán concentrado está cada fonema en un solo clúster. Típicamente **decrece** al aumentar el número de unidades, por la razón opuesta.

**Información mutua normalizada por fonema** (PNMI):

$$\text{PNMI} = \frac{I(y; z)}{H(y)} = \frac{H(y) - H(y \mid z)}{H(y)} = 1 - \frac{H(y \mid z)}{H(y)}$$

Es el **porcentaje de incertidumbre sobre el fonema que se elimina al observar la etiqueta k-means**. Es la métrica principal del paper porque es la única de las tres que penaliza simultáneamente la sobresegmentación y la subsegmentación, y por eso es la que aparece en las tablas comparativas.

Estas métricas son puramente diagnósticas: se usan un alineamiento forzado supervisado que el método nunca ve durante el entrenamiento.

### 9.2. Calidad por capa y por iteración (Figura 2)

Esta es la figura más informativa del paper. Se toman los dos modelos BASE (it1 e it2), se extraen features de las 12 capas Transformer más la entrada a la primera (denotada "Layer 0") —26 features en total entre ambos modelos—, se ajustan tres k-means ($C \in \{100, 500, 1000\}$) sobre un subconjunto de 100 h, y se grafican las tres métricas contra el índice de capa.

Los valores de referencia con MFCC, dados en el texto de la Sección V-C: (ClsPur, PhnPur, PNMI) = **(0.099, 0.335, 0.255)** para $C=100$ y **(0.031, 0.356, 0.287)** para $C=500$.

Lo que se lee en la figura:

1. **Cualquier capa de HuBERT es mucho mejor que MFCC.** La pureza de clúster con $C=100$ pasa de 0.099 (MFCC) a ~0.27 en el pico; PNMI pasa de 0.255 a ~0.69-0.72; la pureza de fonema de ~0.34 a ~0.72.
2. **BASE-it1 tiene un pico marcado en las capas centrales, alrededor de la 6**, y luego **se degrada dramáticamente**: PNMI cae de ~0.69 en la capa 6-7 a ~0.40-0.47 en la capa 12, y la pureza de fonema de ~0.72 a ~0.42-0.48. Esa es la justificación empírica de extraer de la capa 6 para el segundo clustering.
3. **BASE-it2 mejora monótonamente con la profundidad y no colapsa al final**: el PNMI se estabiliza en ~0.70-0.72 desde la capa 7 hasta la 12. De ahí la elección de la capa 9 para producir las etiquetas de LARGE y X-LARGE.
4. **La mejor feature de it2 supera a la mejor de it1 en pureza de fonema y PNMI, pero es levemente peor en pureza de clúster.** El paper lo consigna sin explicarlo.

La explicación que el paper ofrece para el colapso de las capas finales de it1 es exactamente la que uno esperaría del marco teórico y merece subrayarse:

> *"the quality of the last few layers degrades dramatically for BASE-it1, potentially because it is trained on target assignments of worse quality, and therefore the last few layers learn to mimic their bad label behavior."*

Traducido: **las capas superiores se especializan en resolver la tarea de salida, y si la tarea de salida es basura, las capas superiores aprenden basura.** Las capas intermedias, en cambio, están suficientemente lejos del objetivo como para que el gradiente las use para construir la representación genérica que la tarea *requiere* sin arrastrarlas a los idiosincrasias del maestro. Es el mismo fenómeno que se observa en BERT (donde las últimas capas se especializan en MLM y las intermedias transfieren mejor) y en visión con clasificadores de ImageNet, pero aquí es más extremo porque el objetivo es explícitamente ruidoso.

Y da la regla práctica que sigue vigente: **cuando el objetivo de preentrenamiento es de baja calidad, extraer del medio; cuando mejora, se puede extraer más arriba.** La ubicación óptima de extracción se desplaza hacia arriba con las iteraciones.

### 9.3. El efecto de $C$

Con maestro MFCC y $\alpha = 1$ (Tabla V), el WER en dev-other es 18.68 para $C=50$, **17.86 para $C=100$** y 18.40 para $C=500$. Óptimo en 100, y no monótono.

El detalle relevante: el PNMI del maestro **sí es monótono creciente** en $C$ (0.227 → 0.243 → 0.276), pero el WER no. Es decir, **el PNMI del maestro no predice el desempeño downstream cuando se compara entre distintos números de unidades**. Es coherente con la advertencia del propio paper sobre la no comparabilidad de las métricas de pureza entre configuraciones de distinto $C$, y es un recordatorio útil: la información mutua sube trivialmente al partir más fino, pero un vocabulario objetivo demasiado grande vuelve la tarea de predicción enmascarada demasiado difícil y ruidosa. Con 500 clústeres sobre MFCCs, los clústeres empiezan a codificar identidad de hablante y condiciones de canal, que son impredecibles desde el contexto fonético.

El caso extremo es DiscreteBERT con 13.500 unidades: PNMI presumiblemente altísimo y WER de 26.6.

En la segunda iteración, en cambio, se usa $C = 500$ y funciona bien — porque las features contextuales ya han descartado buena parte de la variación de hablante, así que 500 clústeres sobre ellas parten el espacio fonético, no el espacio acústico crudo.

### 9.4. El resultado central: calidad del maestro contra desempeño

Reorganizando la Tabla V por PNMI creciente, con $\alpha = 1$:

| Maestro | PNMI | dev-other WER | $\Delta$ PNMI | $\Delta$ WER |
|---|---|---|---|---|
| k-means MFCC, $C=100$ | 0.243 | 17.86 | — | — |
| k-means BASE-it1-L6, $C=500$ | 0.637 | 11.91 | +0.394 | −5.95 |
| k-means BASE-it2-L9, $C=500$ | 0.704 | 10.75 | +0.067 | −1.16 |
| Chenone supervisado, $C=8976$ | 0.809 | 10.38 | +0.105 | −0.37 |

**La relación es monótona pero con retornos fuertemente decrecientes.** El primer salto de calidad del maestro (+0.394 de PNMI) compra 5.95 puntos de WER. El último (+0.105, y que requiere transcripciones humanas y un sistema HMM completo) compra 0.37.

Ahí está la validación cuantitativa de la tesis del paper, y conviene enunciarla con precisión porque es fácil sobreinterpretarla:

**La consistencia del maestro es condición suficiente para que el método arranque; la corrección del maestro sí importa, pero su valor marginal se agota rápido.** Con un maestro cuya exactitud fonética es del 33%, HuBERT llega a 17.86 de WER — malo, pero **funcional**, y funcional es todo lo que se necesita para que el bootstrapping tome vuelo. Dos iteraciones después, el sistema está a 0.37 puntos de lo que se lograría con alineamientos forzados supervisados.

El complemento imprescindible de esa lectura es la columna $\alpha = 0$ de la Tabla V. El maestro malo con $\alpha = 0$ da **96.37%** de WER. Mismo maestro, misma arquitectura, mismos datos; lo único que cambia es dónde se aplica la pérdida. Eso demuestra que **el mecanismo que rescata al maestro malo no es el maestro sino la máscara**: es la imposibilidad de ver la trama objetivo lo que convierte "aprende esta partición arbitraria" en "aprende la estructura del habla que hace predecible esta partición arbitraria".

Y la contracara: con el maestro supervisado, $\alpha = 0.5$ (9.16) es mejor que $\alpha = 1$ (10.38). Cuando las etiquetas son buenas, imitarlas es útil. La regla completa es: **$\alpha$ óptimo es función decreciente de la calidad del maestro.**

### 9.5. Estabilidad del clustering

Ya cubierta en la Sección 3.2 con la Tabla IV. El resumen operativo: k-means sobre estas features es estable a través de semillas (desviaciones estándar de 0.000 a 0.012 en PNMI) y no requiere ajustarse sobre todo el corpus. Es lo que hace practicable un pipeline de dos fases: si el clustering fuera inestable, cada iteración introduciría deriva y el bootstrapping no convergería.

## 10. Limitaciones

**Costo de preentrenamiento.** El paper da una sola cifra de tiempo, en la Sección IV-C: *"Training for 100k steps takes about 9.5 hours"*, referida a BASE sobre 32 GPUs. De ahí se deriva, para BASE, $650\text{k}$ pasos $\approx 61.75$ horas de reloj $\times 32$ GPUs $\approx$ **2.000 GPU-horas**, sin contar el clustering ni el fine-tuning. **Para LARGE y X-LARGE el paper no reporta tiempo de reloj**, solo que se entrenan 400k pasos sobre 128 y 256 GPUs respectivamente. Un límite inferior grosero, suponiendo optimistamente la misma velocidad por paso que BASE (irreal: X-LARGE tiene 10× más parámetros y batch por GPU 4× menor), sería $38\text{ h} \times 256 \approx$ 9.700 GPU-horas para X-LARGE; el número real es sustancialmente mayor. En cualquier caso, **el paper no permite calcular el costo total del sistema**, y esa omisión es en sí una limitación de reporte.

**El pipeline de múltiples etapas.** Esta es la limitación que los propios autores declaran en la conclusión: *"For future work, we plan to improve the HuBERT training procedure to consist of a single phase."* wav2vec 2.0 es end-to-end: un solo entrenamiento, un solo objetivo, sin artefactos intermedios. HuBERT requiere, en orden: extraer MFCCs de 960 h → ajustar k-means → inferir etiquetas para todo el corpus → preentrenar 250k pasos → extraer activaciones de la capa 6 sobre el 10% del corpus → ajustar k-means → inferir etiquetas → preentrenar 400k pasos → extraer activaciones de la capa 9 → ajustar k-means → inferir etiquetas → preentrenar LARGE/X-LARGE. Son cuatro artefactos intermedios materializados en disco (las etiquetas de 60.000 horas a 50 Hz son $1.08 \times 10^{10}$ enteros) y tres modelos de clustering versionados. Desde la perspectiva de un sistema en producción, es un DAG frágil: cada etapa es un punto de falla y de deriva de versiones, y no hay forma de reentrenar parcialmente.

**La dependencia de una elección de capa que se descubre a posteriori.** Las capas 6 y 9 no salen de ningún principio: salen de mirar la Figura 2, que solo se puede construir después de haber entrenado el modelo. Para un idioma o dominio nuevo, no hay garantía de que el óptimo esté en la misma capa, y el diagnóstico requiere un alineamiento forzado supervisado —justo el recurso que el método pretende no necesitar. Se puede sortear con métricas no supervisadas, pero el paper no lo explora.

**El sesgo hacia el inglés leído.** Todo el paper vive dentro del universo LibriVox: audiolibros de dominio público en inglés, leídos por voluntarios, sin ruido de fondo significativo, sin solapamiento de hablantes, sin habla espontánea, sin disfluencias, sin acentos no nativos en proporción representativa, sin código-switching. La introducción argumenta explícitamente que el SSL es crucial para lenguas y dialectos con pocos recursos y ortografías no estandarizadas — y ese argumento no se prueba en ningún experimento del paper. El preentrenamiento y el fine-tuning y la evaluación salen del mismo dominio. La generalización a habla conversacional, telefónica o multilingüe queda como afirmación de intención.

**Las unidades descubiertas no son fonemas.** La mejor pureza de fonema que se alcanza es ~0.72 (Figura 2), es decir, cerca de un tercio de las tramas se etiquetan con un clúster cuyo fonema mayoritario no corresponde. Los clústeres correlacionan con la fonética pero no la recuperan: mezclan identidad de hablante, contexto coarticulatorio, posición en la sílaba y condiciones de grabación. Cualquier lectura de HuBERT como "descubridor de fonemas no supervisado" es incorrecta, y el paper es cuidadoso en no hacerla.

**El techo del maestro supervisado.** La Tabla V muestra que un maestro con alineamiento forzado supervisado y $\alpha=0.5$ da 9.16 contra los 10.75 del mejor maestro no supervisado. Existe una brecha residual de 1.59 puntos que dos iteraciones de refinamiento no cierran, y el paper no explora si más iteraciones la cerrarían.

**Ensembles no desplegados.** La Tabla VI muestra que los ensembles y la cuantización por producto dan más de un punto de WER, y ninguno de los modelos principales los usa. Queda sin responder si las cifras de la Tabla II mejorarían.

**Todo se evalúa en ASR.** La conclusión promete explorar tareas de reconocimiento y generación más allá del ASR, pero el paper no lo hace. En un trabajo cuyo argumento central es que las features autosupervisadas generalizan a muchas tareas downstream (frente al pseudo-etiquetado que se especializa en una), evaluar una sola tarea es una tensión interna notable. (La historia posterior le dio la razón; ver Sección 11.)

## 11. Impacto y legado

**HuBERT ganó la discusión sobre cómo se hace SSL de habla.** En 2021 competían tres familias: contrastiva (wav2vec 2.0), autorregresiva (APC, CPC) y predicción enmascarada de unidades discretas (DiscreteBERT, HuBERT). La tercera se impuso, y la razón es en buena medida la que el paper anticipa en la Sección III: el objetivo contrastivo tiene demasiadas piezas móviles (muestreo de negativos, pérdida de diversidad, recocido de Gumbel) mientras que una entropía cruzada sobre un vocabulario fijo es robusta y se afina sola.

**Textless NLP y la generación de habla.** Esta es la línea de impacto más grande y no es sobre ASR. Si un modelo produce una secuencia de unidades discretas a partir del audio, entonces se puede entrenar un modelo de lenguaje **sobre esas unidades** y generar habla sin pasar nunca por texto. **GSLM** (*Generative Spoken Language Modeling from Raw Audio*, Lakhotia et al., TACL 2021 — nótese que Kushal Lakhotia es coautor de HuBERT) construyó exactamente eso: HuBERT como tokenizador, un LM sobre unidades, y un vocoder que sintetiza audio desde unidades. La configuración que se volvió canónica en esa línea es **HuBERT BASE, capa 6, k-means con 50-200 clústeres**. De ahí salen la resíntesis de habla, la traducción directa habla-a-habla sin texto (útil precisamente para lenguas sin ortografía estandarizada, cerrando el argumento de la introducción del paper), y los modelos de diálogo hablado.

La distinción entre **tokens semánticos** (derivados de un modelo SSL como HuBERT o w2v-BERT, que capturan contenido) y **tokens acústicos** (derivados de un códec neuronal como SoundStream o EnCodec, que capturan timbre y detalle de la señal) es la arquitectura de referencia de **AudioLM** (Borsos et al., 2022) y de casi todo lo que vino después en generación de audio con LLMs. La idea de "unidades semánticas descubiertas por clustering de un modelo enmascarado" es literalmente esta contribución.

**Descendencia directa.** **WavLM** (Chen et al., IEEE JSTSP 2022) es HuBERT más dos cambios: simulación de habla solapada y ruido durante el preentrenamiento (para que el modelo sirva a tareas de separación de hablantes y diarización, no solo ASR) y sesgo posicional relativo con compuertas en la atención. Es hoy el punto de partida por defecto para tareas de habla que no son ASR. **data2vec** (Baevski et al., 2022) es el movimiento contrario: elimina las unidades discretas y predice representaciones **continuas y contextualizadas** de una red maestra actualizada por media móvil, unificando habla, visión y texto bajo el mismo objetivo — el sucesor conceptual que responde a la limitación del pipeline multietapa que HuBERT declara en su conclusión. **AV-HuBERT** (Shi et al., 2022) extiende la receta a audio y video de labios. En multilingüe, **mHuBERT** y la familia de modelos masivamente multilingües de Meta aplican la misma receta a cientos de idiomas.

**Infraestructura.** HuBERT vive en las dos bibliotecas de referencia. En `transformers`: `HubertModel`, `HubertForCTC`, con checkpoints `facebook/hubert-base-ls960`, `facebook/hubert-large-ll60k`, `facebook/hubert-xlarge-ll60k` y sus versiones ajustadas para ASR. En `torchaudio.pipelines`: `HUBERT_BASE`, `HUBERT_LARGE`, `HUBERT_XLARGE`, más los bundles `HUBERT_ASR_LARGE` y `HUBERT_ASR_XLARGE`. Que un método esté en `torchaudio.pipelines` es el indicador más duro de que dejó de ser investigación y pasó a ser infraestructura.

**SUPERB.** El benchmark SUPERB, que evalúa encoders de habla congelados en una decena de tareas (reconocimiento de fonemas, ASR, identificación de hablante, verificación, diarización, detección de intención, clasificación de emoción, *slot filling*, *query-by-example*), consolidó la práctica de usar un encoder SSL congelado con cabezales livianos. HuBERT y WavLM ocupan la parte alta de esa tabla desde su aparición. Ahí es donde se validó la promesa de la introducción del paper —que las features autosupervisadas generalizan a muchas tareas— que el paper mismo no había probado.

**Y una nota sobre el otro camino.** Whisper (Radford et al., 2022) tomó la vía opuesta: supervisión débil masiva (680.000 horas de audio con transcripciones de calidad variable raspadas de internet) sobre un Transformer encoder-decoder convencional, sin ningún SSL. Es un contrapunto honesto: HuBERT es la respuesta cuando no hay etiquetas; Whisper es la respuesta cuando se pueden conseguir muchas etiquetas ruidosas. Ambos son Transformers, ambos aparecieron antes del PDF de la clase 39, y ambos son estado del arte en su nicho.

## 12. HuBERT contra las tres objeciones de la clase 39

El PDF de la clase (abril de 2024) sostiene que los Transformers "no son actualmente muy populares para aplicaciones de audio" y da tres razones. Evalúo cada una con la evidencia del paper, que es de 2021.

### 12.1. Objeción 1: "faltan datasets de audio masivos"

**Lo que la objeción tiene de cierto.** Los corpus de habla **etiquetada** son y siguen siendo órdenes de magnitud más chicos que los corpus de texto. LibriSpeech, el estándar de la comunidad durante una década, tiene 960 horas transcritas. El costo de transcribir es alto y crece con la dificultad del audio: transcripción cuidadosa de habla espontánea toma varias veces el tiempo real, y para lenguas sin ortografía estandarizada el problema no es de costo sino de posibilidad. Comparado con los 750 GB de texto de C4 o el corpus de 3.300 millones de palabras de BERT, el habla etiquetada es un desierto. La introducción del paper dice exactamente eso: *"The time needed to collect large labeled datasets covering each of these scenarios is the real bottleneck in the current fast-moving AI industry."*

**Lo que la objeción pasa por alto.** El SSL cambia el recurso escaso. La pregunta deja de ser "cuántas horas transcritas hay" y pasa a ser "cuántas horas de audio hay". Y de audio hay muchísimo: **Libri-Light son 60.000 horas de audiolibros de dominio público leídos por voluntarios, sin una sola transcripción usada durante el preentrenamiento.**

Vale la pena dimensionarlo en las unidades en las que piensa un Transformer. A 50 tramas por segundo:

$$60.000 \text{ h} \times 3600 \frac{\text{s}}{\text{h}} \times 50 \frac{\text{tramas}}{\text{s}} = 1{,}08 \times 10^{10} \text{ tramas}$$

**10.800 millones de tramas por época.** El corpus de preentrenamiento de BERT (BooksCorpus + Wikipedia) tenía unos 3.300 millones de palabras. Es decir, **60.000 horas de audiolibros producen más de tres veces los tokens del corpus original de BERT**, y esas 60.000 horas se obtienen de un solo proyecto de voluntariado de dominio público. En bytes, ese mismo corpus a 16 kHz y 16 bits son unos 6,9 TB de PCM.

La objeción confunde "no hay datasets masivos" con "no hay datasets masivos **etiquetados**". La primera afirmación es falsa; la segunda es cierta y es precisamente el problema que el SSL resuelve. Como decía la introducción del paper, la anotación además **descarta** información: *"labels, annotations, and text-only material ignores rich information in the input signal"* — la transcripción tira a la basura el timbre, la prosodia, la emoción, el ruido estructurado.

**Con cuántas etiquetas alcanza para un sistema usable.** Este es el remate. Tabla II, 10 minutos de audio etiquetado:

- HuBERT X-LARGE: **4.6% de WER en test-clean, 6.8% en test-other.**

Diez minutos. Un sistema con 4.6% de WER es perfectamente usable para dictado, subtitulado asistido o búsqueda por voz. Y con 1 hora se llega a 2.8/4.8, que está en el rango de lo que un sistema supervisado con cientos de horas producía pocos años antes.

**El matiz honesto que hay que declarar.** Esas cifras usan un LM Transformer entrenado sobre los datos oficiales de modelado de lenguaje de LibriSpeech, que son unos 800 millones de palabras de texto. La supervisión **acústica** es de 10 minutos; la supervisión **lingüística** no lo es. Comparando con la misma columna sin LM Transformer: HuBERT LARGE con 10 minutos y 4-gram da 6.6/10.1 en vez de 4.7/7.6. El LM aporta ~2 puntos. Sigue siendo un resultado extraordinario para 10 minutos de transcripción, pero "10 minutos de supervisión total" sería una exageración. Este matiz **no está bien señalado en el paper** (ver Sección 13).

**Veredicto.** La objeción confunde el eje. En el régimen supervisado tenía razón; el SSL la desactiva por construcción, y HuBERT es la demostración. Además, para abril de 2024 la premisa fáctica ya no se sostenía ni siquiera para datos etiquetados: Whisper se entrenó con 680.000 horas con transcripción débil.

### 12.2. Objeción 2: "la self-attention opera sobre entidades discretas y el audio no se segmenta trivialmente"

Esta es la objeción que HuBERT responde con más precisión, porque **es literalmente el problema que el paper se plantea resolver**. Comparemos los enunciados:

| Clase 39 (abril 2024) | HuBERT (junio 2021), abstract |
|---|---|
| "Self-attention mechanism operates over a finite sequence of discrete entities. In the context of text, sentence segmentation is trivial, but for audio this is not the case." | "(3) sound units have variable lengths with **no explicit segmentation**." |

Es la misma observación. La diferencia es que la clase la presenta como razón para no aplicar Transformers al audio, y el paper la presenta como el problema de investigación que va a resolver. La respuesta de HuBERT tiene dos partes, y conviene separarlas porque la objeción mezcla dos cosas distintas.

**Parte A: la self-attention no necesita segmentación semántica, necesita una secuencia de vectores.** Aquí hay una confusión conceptual en la objeción. La self-attention opera sobre una secuencia de vectores de dimensión fija; no le importa qué representan esos vectores ni si sus fronteras coinciden con unidades lingüísticas. En NLP los tokens son subpalabras porque el texto viene así, pero el mecanismo no lo exige — de hecho, ViT parte imágenes en parches de $16\times16$ píxeles que **no** corresponden a ningún objeto ni a ninguna frontera semántica, y funciona perfectamente. La segmentación semántica nunca fue un requisito de la atención.

HuBERT hace exactamente lo mismo que ViT, en el eje temporal: **el encoder convolucional de 7 capas parte el audio en tramas regulares de 20 ms con campo receptivo de 25 ms.** No hay ninguna pretensión de que esas fronteras coincidan con fonemas. Son parches temporales de tamaño fijo, igual que los parches espaciales de ViT. La afirmación de que "el audio no se segmenta trivialmente" es cierta para la segmentación fonética y **completamente irrelevante** para alimentar una self-attention.

**Parte B: sí hay un lugar donde se necesitan entidades discretas —el objetivo— y HuBERT las fabrica.** El problema real, que la objeción roza sin nombrar, no está en la entrada de la atención sino en la **salida**: para escribir una pérdida de predicción enmascarada estilo BERT hace falta un vocabulario finito sobre el cual poner el softmax, y el habla no lo trae. Es el problema (2) del abstract: *"there is no lexicon of input sound units during the pre-training phase... hindering the use of predictive losses."*

La solución de HuBERT es **no segmentar y en cambio inventar las clases**. En vez de buscar dónde terminan los fonemas, se corre k-means sobre las features de cada trama y se declara que los $C$ centroides son el vocabulario. Cada trama recibe una etiqueta. No hay segmentación explícita en ningún momento: la segmentación emerge implícitamente como corridas de tramas con la misma etiqueta, y nunca hace falta materializarla.

Y el resultado es que funciona con un maestro que se equivoca dos tercios de las veces (pureza de fonema 0.335), porque —como establece la Tabla V— lo que importa es que la asignación sea **consistente**, no correcta. La dificultad genuina que la objeción identifica ("no se puede segmentar bien el audio") queda esquivada mostrando que **no hace falta segmentar bien**.

**Cronología.** El PDF de la clase es de abril de 2024. HuBERT es de junio de 2021, y su versión revisada de TASLP es de octubre de 2021. wav2vec 2.0, que ya había atacado el mismo problema por otra vía, es de junio de 2020. La objeción describe un problema abierto de 2019 que llevaba casi cuatro años resuelto y desplegado en `torchaudio` cuando se escribió la diapositiva.

**Veredicto.** La objeción identifica correctamente una dificultad real y específica del habla, y la formula casi con las mismas palabras que los autores del paper. Pero la trata como impedimento cuando ya era un problema resuelto, y además ubica mal el problema: la self-attention nunca necesitó entidades semánticas en la entrada; era la pérdida predictiva la que necesitaba clases en la salida, y esas se fabrican con k-means.

### 12.3. Objeción 3: "los Transformers no son buenos para modelar dependencias largas"

Esta es la más problemática de las tres, porque invierte la motivación original del mecanismo.

**Lo que dice el paper fundacional.** Vaswani et al. (2017), Sección 4, dedican una tabla completa a comparar tipos de capa por tres criterios: complejidad por capa, número de operaciones secuenciales y **longitud máxima de camino** entre dos posiciones cualesquiera.

| Tipo de capa | Complejidad por capa | Ops. secuenciales | Longitud máxima de camino |
|---|---|---|---|
| Self-attention | $O(n^2 \cdot d)$ | $O(1)$ | $\mathbf{O(1)}$ |
| Recurrente | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolucional | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k n)$ |
| Self-attention restringida (ventana $r$) | $O(r \cdot n \cdot d)$ | $O(1)$ | $O(n/r)$ |

Y el argumento explícito: *"Learning long-range dependencies is a key challenge in many sequence transduction tasks. One key factor affecting the ability to learn such dependencies is the length of the paths forward and backward signals have to traverse in the network. The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies."*

**La longitud máxima de camino de la self-attention es $O(1)$: constante, independiente de la distancia.** Cualquier par de posiciones se conecta en un solo paso de atención. En una RNN, la señal entre las posiciones 1 y 500 tiene que atravesar 500 multiplicaciones matriciales sucesivas, con el consiguiente desvanecimiento o explosión de gradiente — el problema que las LSTM mitigan sin eliminar. Modelar dependencias largas es **la razón por la que se inventó la self-attention**, no su debilidad.

**Cuál es la limitación real.** Es el **costo**, no la capacidad: $O(n^2)$ en tiempo y memoria. Con secuencias muy largas el mecanismo se vuelve inviable computacionalmente, y de ahí toda la literatura de atención eficiente (Longformer, Performer, atención por ventanas, FlashAttention). Pero "es caro para $n$ grande" y "no puede modelar dependencias largas" son afirmaciones distintas, y la segunda es falsa. Confundirlas es como decir que una lista enlazada "no puede indexar" cuando lo que ocurre es que indexar cuesta $O(n)$.

Un matiz que sí sería una crítica válida y que la objeción no hace: en secuencias muy largas, el softmax sobre $n$ posiciones dispersa la atención y hay evidencia de que los modelos degradan en el aprovechamiento del contexto largo. Pero eso es un fenómeno de entrenamiento y calibración, no una limitación arquitectónica del camino $O(1)$, y en 2024 ya coexistía con modelos de contexto de cientos de miles de tokens.

**Cómo lo resuelve HuBERT concretamente.** El costo cuadrático en audio es genuinamente amenazante: 10 segundos a 16 kHz son 160.000 muestras, y una matriz de atención de $160.000^2$ es absurda. La solución está en la arquitectura y ya la describimos: **el encoder convolucional reduce 320×**. Diez segundos pasan de 160.000 muestras a **500 tramas**, tamaño perfectamente cómodo. La reducción de resolución no es un parche: es el frontend que hace del habla una secuencia de longitud manejable, exactamente como el parcheado de $16\times16$ hace de una imagen de $224\times224$ una secuencia de 196 tokens en ViT.

Y hay un dato cuantitativo que cierra el argumento de costo. Según la tabla de Vaswani, la self-attention es **más barata** que una capa recurrente cuando $n < d$. Para HuBERT BASE, $d = 768$; para LARGE, $d = 1024$. Un enunciado de 10 segundos son 500 tramas. **$500 < 768$: en el régimen operativo real de HuBERT, la self-attention es computacionalmente más barata que una LSTM equivalente, además de tener camino $O(1)$.** La objeción no solo es falsa en capacidad; es falsa también en costo para las longitudes que el habla produce después del frontend convolucional.

**La evidencia del propio paper.** HuBERT no es neutral respecto de las dependencias largas: **su objetivo entero está diseñado para forzarlas.** Sección II-B:

> *"to reduce the prediction error, the model needs to capture the long-range temporal relations between learned representations"*

y sobre $\alpha = 1$:

> *"It forces the model to learn both the acoustic representation of unmasked segments and the **long-range temporal structure** of the speech data."*

Concretamente, con spans de 200 ms enmascarados y ~57% de tramas borradas, el modelo tiene que inferir dos o tres fonemas completos desde contexto que puede estar a cientos de milisegundos. Y funciona: es el mecanismo que produce los 4.6% de WER con 10 minutos de etiquetas. La Tabla V lo prueba por contraste — cuando se le permite al modelo resolver la tarea localmente ($\alpha = 0$, ver la trama que hay que predecir), el desempeño colapsa a 96%. **El modelo solo aprende algo útil cuando se le obliga a usar contexto largo.**

**Veredicto.** La objeción invierte el argumento de diseño de la self-attention. La limitación real es el costo cuadrático, HuBERT la resuelve con el downsampling de 320× del frontend convolucional, y en el régimen resultante ($n \approx 500 < d$) la atención es además más barata que una recurrente. El paper usa las dependencias largas como **el mecanismo central de su objetivo de aprendizaje**, no las padece.

### 12.4. Balance

| Objeción de la clase 39 | Estado en 2021 según el paper | Veredicto |
|---|---|---|
| 1. Faltan datasets de audio masivos | Cierto para audio **etiquetado**; el SSL cambia el recurso escaso. 60.000 h sin etiquetar → 10.800 M de tramas, más que el corpus de BERT | **Parcialmente cierta históricamente, desactivada por el método** |
| 2. La self-attention necesita entidades discretas y el audio no se segmenta | Es el problema (3) del abstract, enunciado por los autores. Se resuelve fabricando unidades con k-means, sin segmentar | **Correcta como diagnóstico, incorrecta como impedimento; resuelta desde 2020-2021** |
| 3. Los Transformers no modelan bien dependencias largas | Camino $O(1)$ contra $O(n)$ de una RNN; la limitación es el costo, resuelto con downsampling 320×; el objetivo de HuBERT exige dependencias largas | **Incorrecta; invierte la motivación del mecanismo** |

Lo justo con el material de la clase es reconocer que la objeción 1 tenía un núcleo verdadero y sigue teniéndolo en su forma correcta: **el habla etiquetada es cara y escasa, y para la mayoría de las 7.000 lenguas del mundo directamente no existe.** Ese hecho es el que motiva todo este paper. Lo que hay que corregir es la conclusión: la escasez de etiquetas no es un argumento contra los Transformers en audio, es exactamente el argumento a favor del preentrenamiento autosupervisado de Transformers en audio.

## 13. Erratas, matices y cosas que se citan mal

**Conteos de parámetros inconsistentes.** La introducción dice "BASE (90M parameters), LARGE (300M), and X-LARGE (1B)". La Tabla I dice **95M, 317M y 964M**. La Tabla I es la fuente precisa; las de la introducción son redondeos generosos, especialmente el "1B" para 964M.

**"19% y 13%" vienen de configuraciones distintas.** El abstract y la introducción dicen que X-LARGE muestra "up to 19% and 13% relative WER improvement from LARGE models on dev-other and test-other". Verificado contra la Tabla II, ninguna fila da ambas cifras a la vez:

| Split de fine-tuning | dev-other LARGE → X-LARGE | Δ relativo | test-other LARGE → X-LARGE | Δ relativo |
|---|---|---|---|---|
| 10 min | 7.0 → 6.1 | −12.9% | 7.6 → 6.8 | −10.5% |
| 1 h | 4.9 → 4.2 | −14.3% | 5.4 → 4.8 | −11.1% |
| 10 h | 4.3 → 3.6 | −16.3% | 4.6 → 4.0 | **−13.0%** |
| 100 h | 3.7 → 3.0 | **−18.9%** | 3.9 → 3.5 | −10.3% |
| 960 h | 3.0 → 2.5 | −16.7% | 3.3 → 2.9 | −12.1% |

El "19%" es el split de **100 h** en dev-other; el "13%" es el split de **10 h** en test-other. Cada uno es el máximo de su columna, tomado de una fila distinta. Es un uso legítimo de "up to", pero citarlo como si describiera una configuración es incorrecto.

**La afirmación sobre las "únicas excepciones" es falsa.** Sección V-A: *"The superiority of HuBERT persists across setups with different amounts of labeled data, with the only exceptions being fine-tuning on 100 hours of labeled data, where HuBERT LARGE is 0.1% WER higher than wav2vec 2.0 LARGE on test-clean, and HuBERT BASE is 0.1% WER higher than wav2vec 2.0 BASE on test-other."*

Contra la Tabla II, hay más excepciones y algunas mayores:

| Split | Columna | wav2vec 2.0 BASE | HuBERT BASE | Diferencia |
|---|---|---|---|---|
| 10 min | dev-clean | 8.9 | 9.1 | +0.2 |
| 10 min | test-clean | 9.1 | 9.7 | **+0.6** |
| 1 h | dev-clean | 5.0 | 5.6 | **+0.6** |
| 1 h | dev-other | 10.8 | 10.9 | +0.1 |
| 1 h | test-clean | 5.5 | 6.1 | **+0.6** |

Y en la Tabla III, con 960 h etiquetadas, HuBERT LARGE (1.5/3.0/1.9/3.3) es **peor en test-clean** que wav2vec 2.0 LARGE (1.6/3.0/1.8/3.3) y empata en las otras tres columnas.

La conclusión correcta, que el paper no enuncia: **con el mismo tamaño (BASE) y los mismos datos (LS-960), HuBERT no supera a wav2vec 2.0; es equivalente o levemente peor en las condiciones limpias de muy bajos recursos.** La ventaja de HuBERT es de escalamiento — aparece en LARGE con LL-60k y se hace clara en X-LARGE. Esto no le quita valor al paper, pero cambia qué conclusión se puede extraer.

**La ecuación (1) está escrita sin signo negativo.** El texto la introduce como *"the cross-entropy loss computed over masked and unmasked timesteps"* pero escribe $L_m = \sum_{t\in M} \log p_f(z_t \mid \tilde{X}, t)$, que es una log-verosimilitud a **maximizar**, no una pérdida a minimizar. Es un descuido de notación; la implementación obviamente minimiza $-\sum \log p$.

**Los PNMI del mismo maestro no coinciden entre tablas.** Para el k-means sobre MFCC con $C=100$: la Tabla IV reporta 0.251-0.253, la Tabla V reporta **0.243**, y el texto de la Sección V-C (referido a la Figura 2) reporta **0.255**. Para BASE-it1-capa6 con $C=500$: la Tabla IV reporta 0.680-0.686 y la Tabla V reporta **0.637**, una diferencia de casi 0.05. El paper no explica las discrepancias; presumiblemente difieren el conjunto de ajuste del k-means (1/10/100 h contra el 10% de 960 h) y el conjunto de evaluación (dev combinado contra otro). Conviene no mezclar números de PNMI entre tablas.

**Los modelos principales no usan ensembles ni cuantización por producto.** La Tabla VI muestra que el ensemble de tres cuantizaciones por producto baja el WER de 17.86 a 16.73 sobre maestros MFCC. Ninguna de las cifras de la Tabla II ni de la Tabla III usa esa técnica. Se cita a veces "HuBERT usa ensembles de clustering"; la formulación existe en la Sección II-C y se ablaciona, pero no está desplegada.

**$\alpha$ nunca se declara para los modelos principales.** Solo se infiere del abstract ("applying the prediction loss over the masked regions only") y de la Tabla V. El paper no dice "usamos $\alpha=1$" en ninguna parte de la Sección IV.

**Las ablations se corren con $p = 6.5\%$, no con el $p = 8\%$ óptimo.** La leyenda de la Tabla VII lo dice explícitamente, y el valor 17.86 aparece tanto en la Tabla V como en la columna de 100k pasos de la Tabla VII, lo que confirma que ambas comparten esa configuración. Es decir, **la Tabla V no está en el punto óptimo de enmascaramiento**; sus conclusiones cualitativas sobre $\alpha$ y calidad del maestro son sólidas, pero sus WER absolutos no son comparables con los de la Tabla II.

**"$p = 8\%$ de las tramas se enmascaran" es incorrecto.** $p$ es la probabilidad de que una trama sea **índice de inicio** de un span de 10. La fracción efectivamente enmascarada es de aproximadamente $1 - 0.92^{10} \approx 57\%$. El paper no reporta la fracción resultante, así que la confusión es fácil.

**"10 minutos de supervisión" es una media verdad.** El 4.6% de WER de X-LARGE con 10 minutos usa un LM Transformer entrenado sobre el corpus de texto de LibriSpeech. Comparando con la misma configuración sin LM neuronal (HuBERT LARGE, 10 min, 4-gram: 6.6/10.1 contra 4.7/7.6 con Transformer), el LM aporta cerca de 2 puntos. La supervisión acústica es de 10 minutos; la lingüística es enorme y viene de texto. El paper no lo señala como caveat en la discusión de resultados.

**"HuBERT se entrena en 60.000 horas" es incorrecto para BASE.** BASE se preentrena sobre **LS-960** en ambas iteraciones (Sección IV-C). Solo LARGE y X-LARGE usan Libri-Light 60k. El checkpoint más usado en la práctica, `hubert-base-ls960`, vio 960 horas.

**"HuBERT hace dos iteraciones" es incompleto.** BASE hace dos. LARGE y X-LARGE son efectivamente **terceras iteraciones**, porque parten de etiquetas extraídas de la capa 9 de BASE-it2 en vez de reiniciar desde MFCC. El propio paper lo dice: *"these two models can also be seen as the third iteration models."*

**"HuBERT descubre fonemas" es incorrecto.** La pureza de fonema máxima observada es ~0.72 (Figura 2, con $C=500$ o $1000$ sobre BASE-it2). Los clústeres correlacionan con fonética pero también con hablante, canal y contexto. Y el paper es explícito en que la métrica de pureza no es comparable entre distintos $C$, así que ni siquiera ese 0.72 se puede leer como "72% de exactitud fonética" en sentido absoluto.

**"HuBERT es un BERT" es una analogía, no una identidad.** Lo que se comparte es el objetivo de predicción enmascarada sobre un vocabulario finito y la arquitectura de encoder Transformer bidireccional. Lo que no: la entrada no es discreta (es la onda cruda pasada por convoluciones), no hay tokenizador, no hay `[CLS]`, no hay next-sentence prediction, la capa de salida usa similitud coseno con temperatura en vez de un softmax lineal atado a la tabla de embeddings de entrada, y el "vocabulario" se descubre en vez de definirse.

**Erratas tipográficas del preprint v1**, para quien lea el PDF: "BERT-like **per-training**" (introducción), "The **fisrt** two follow the architectures" (Sección II-E), "predicts a distribution over the target **indeces**" (Sección II-B), "**Superivsed**" (encabezado de la Tabla III), "using fixed **hyperaprameters**" (Sección V-D), "**We** use the connectionist temporal classification" con mayúscula a media oración (Sección II-E), y "LV-60k" en la fila de Noisy Student de la Tabla III donde el resto de la tabla usa "LL-60k".

## 14. Cómo se ve hoy

HuBERT se usa en la práctica de dos maneras: como **extractor de embeddings continuos** para cabezales downstream, y como **tokenizador discreto** vía k-means sobre una capa intermedia (el pipeline de textless NLP). El código siguiente cubre ambas.

```python
# pip install transformers torch soundfile scikit-learn
import torch
import numpy as np
from transformers import AutoFeatureExtractor, HubertModel
from sklearn.cluster import MiniBatchKMeans

CKPT = "facebook/hubert-base-ls960"   # 95M, 12 capas, preentrenado en LS-960
# alternativas: facebook/hubert-large-ll60k (317M, 24 capas, LL-60k)
#               facebook/hubert-xlarge-ll60k (964M, 48 capas, LL-60k)

fe    = AutoFeatureExtractor.from_pretrained(CKPT)
model = HubertModel.from_pretrained(CKPT).eval()

# GOTCHA 1: hubert-base-ls960 tiene do_normalize=False, mientras que las variantes
# large/xlarge usan do_normalize=True (normalización de media/varianza por enunciado).
# Usar siempre el feature extractor del checkpoint; hardcodear la normalización
# degrada silenciosamente los embeddings.
# GOTCHA 2: la entrada debe ser mono a 16 kHz. El modelo no remuestrea.

wav = np.random.randn(16000 * 5).astype(np.float32)   # 5 s simulados
inputs = fe(wav, sampling_rate=16000, return_tensors="pt")

with torch.inference_mode():
    out = model(**inputs, output_hidden_states=True)

# out.hidden_states es una tupla de 13 tensores para BASE:
#   índice 0  = entrada al primer bloque Transformer  ("Layer 0" de la Figura 2)
#   índices 1..12 = salidas de las 12 capas Transformer
# La indexación coincide con la del paper, lo que hace directamente reutilizables
# las conclusiones de la Figura 2.
print(len(out.hidden_states), out.hidden_states[6].shape)
# -> 13, torch.Size([1, 249, 768])
# 249 tramas para 5 s: 250 esperadas a 50 fps, menos el borde del campo
# receptivo de 400 muestras del encoder convolucional.

# --- Pipeline de unidades discretas (el "maestro" de la iteración 2) ---------
# Réplica de la Sección IV-B: MiniBatchKMeans, batch de 10.000 tramas,
# k-means++ con 20 reinicios. El paper ajusta sobre un 10% de las 960 h porque
# la matriz de features de 768 dimensiones no cabe completa en RAM.

LAYER, N_CLUSTERS = 6, 500       # config del paper para la segunda iteración

def frame_features(wavs, layer=LAYER):
    """Devuelve (N_tramas, 768) apilando las latentes de `layer` de varios audios."""
    feats = []
    for w in wavs:
        x = fe(w, sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            h = model(**x, output_hidden_states=True).hidden_states[layer]
        feats.append(h.squeeze(0).cpu().numpy())
    return np.concatenate(feats, axis=0)

corpus = [np.random.randn(16000 * 4).astype(np.float32) for _ in range(32)]
X = frame_features(corpus)

km = MiniBatchKMeans(
    n_clusters=N_CLUSTERS,
    batch_size=10_000,        # Sección IV-B
    init="k-means++",
    n_init=20,                # "20 random starts", Sección IV-B
    max_no_improvement=100,
    reassignment_ratio=0.0,   # evita reasignar centroides poco poblados
).fit(X)

units = km.predict(frame_features([corpus[0]]))   # una etiqueta cada 20 ms
print(units[:40])
# Colapsar corridas da la "segmentación" implícita, que nunca hubo que calcular:
runs = [u for i, u in enumerate(units) if i == 0 or u != units[i - 1]]
print(len(units), "tramas ->", len(runs), "unidades tras colapsar repeticiones")
```

**Qué capa usar para qué.** Lo que el paper establece (Figura 2, Sección V-C):

- Para **clustering en la segunda iteración de un modelo de primera iteración**: capa 6 de 12. Es donde el perfil de calidad tiene su pico en BASE-it1, y donde cae bruscamente después.
- Para **clustering a partir de un modelo de segunda iteración**: capa 9. En BASE-it2 la calidad ya no colapsa arriba y se mantiene alta de la capa 7 a la 12.
- **Nunca las últimas capas de un modelo entrenado con maestro malo**: se especializan en reproducir los defectos del maestro.

Lo que viene de la práctica posterior, fuera de este paper y por lo tanto sin respaldo en él:

- **Tokenización para textless NLP / GSLM / resíntesis**: HuBERT BASE, **capa 6**, k-means con 50-200 clústeres. Es la configuración canónica de esa línea, heredada directamente de la elección del paper.
- **Tareas de contenido** (ASR, reconocimiento de fonemas, keyword spotting): capas intermedias-altas (8-12 en BASE), o directamente fine-tuning completo con CTC.
- **Tareas de hablante** (identificación, verificación, diarización): **capas bajas** (1-4). La información paralingüística se descarta a medida que se sube, porque el objetivo la castiga. Para estas tareas WavLM es una mejor opción de partida que HuBERT, ya que su preentrenamiento incluye simulación de solapamiento y ruido.
- Si el objetivo es ASR y no interesa la interpretabilidad por capa, `facebook/hubert-large-ls960-ft` ya viene con cabezal CTC entrenado y se usa vía `HubertForCTC` sin nada más.

Una última nota de ingeniería directamente relevante para pipelines de producción: **el encoder convolucional se congela durante el fine-tuning** (Sección II-E). En una arquitectura de servicio, eso significa que las 7 capas convolucionales pueden extraerse una sola vez, cachearse como tensores de 512 canales a 50 Hz, y compartirse entre múltiples cabezales downstream. Para 60.000 horas de audio, la diferencia entre recomputar el frontend por tarea y cachearlo es sustancial: es el mismo argumento que justifica materializar features en cualquier pipeline de ML, y aquí está avalado por el diseño del modelo.
