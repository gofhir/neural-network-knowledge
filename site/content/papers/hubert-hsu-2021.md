---
title: "HuBERT: aprendizaje autosupervisado del habla por predicción de unidades ocultas (2021)"
weight: 435
math: true
---

{{< paper-card
    title="HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units"
    authors="Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, Abdelrahman Mohamed (Facebook AI Research)"
    year="2021"
    venue="IEEE/ACM TASLP 2021 / arXiv:2106.07447"
    pdf="/papers/hubert-hsu-2021.pdf" >}}
BERT necesita un vocabulario discreto para poder escribir su pérdida de predicción enmascarada; el habla es una señal continua que no trae tokens. [wav2vec 2.0](/papers/wav2vec2-baevski-2020) había esquivado el problema con una pérdida contrastiva y cuantización interna; HuBERT lo ataca de frente y **fabrica el vocabulario**: un k-means offline agrupa las tramas de audio y produce una etiqueta discreta cada 20 ms, y sobre esas etiquetas se entrena una entropía cruzada estilo BERT aplicada **solo a las regiones enmascaradas**. El paso de clustering se itera —la primera vuelta agrupa MFCC de 39 dimensiones, la segunda agrupa las latentes de una capa intermedia del propio modelo ya entrenado— y con eso el maestro mejora solo, sin ninguna etiqueta humana en el ciclo. La tesis conceptual, que es lo más valioso del paper, es que **lo que importa del maestro es la consistencia de sus asignaciones, no su corrección**: con un k-means que acierta el fonema apenas el 33.5% de las veces, el modelo aprende igual, porque la máscara convierte "reproduce esta partición arbitraria" en "aprende la estructura del habla que hace predecible esta partición arbitraria". El resultado material: **con 10 minutos de audio transcrito y 60.000 horas sin transcribir, HuBERT X-LARGE alcanza 4.6% de WER en test-clean de [LibriSpeech](/papers/librispeech-panayotov-2015)**. Además de ser el paper que ganó la discusión sobre cómo se hace SSL de habla, es la respuesta directa —publicada casi tres años antes— a las tres objeciones que la [Clase 39](/clases/clase-39) levanta contra el uso de Transformers en audio.
{{< /paper-card >}}

---

## Contexto: el SSL del habla en 2020-2021

[wav2vec 2.0](/papers/wav2vec2-baevski-2020) (Baevski, Zhou, Mohamed y Auli, 2020) había fijado el molde arquitectónico que HuBERT hereda casi sin cambios: un **encoder convolucional sobre la onda cruda** que produce una secuencia de tramas latentes, un **encoder Transformer** que la contextualiza, y **enmascaramiento** aplicado sobre las salidas del encoder convolucional. Lo que cambia es el objetivo.

wav2vec 2.0 usa una **pérdida contrastiva**: en cada posición enmascarada hay que identificar la cuantización correcta de esa trama entre distractores muestreados de otras posiciones del mismo enunciado. Para que exista un "objetivo correcto" discreto, el modelo cuantiza internamente la salida convolucional con Gumbel-softmax y codebooks aprendidos conjuntamente.

El paper hace una crítica quirúrgica a ese diseño, que conviene leer como lista de deudas técnicas. Tres puntos son de ingeniería —el objetivo contrastivo funciona, pero tiene demasiadas piezas móviles—: **exige diseñar con cuidado de dónde se muestrean los negativos** (si salen del mismo enunciado, el negativo puede ser acústicamente idéntico al positivo; si salen de otro enunciado, la tarea se vuelve trivial porque basta identificar al hablante), requiere una **pérdida auxiliar de diversidad** para que el codebook no colapse, y requiere un **calendario de recocido de la temperatura de Gumbel-softmax** bien afinado. El cuarto punto es conceptual y es el que HuBERT explota: wav2vec 2.0 solo cuantiza la salida del encoder convolucional, y **la mejor representación para cuantizar no está en el frontend, sino a media altura del Transformer**.

El otro antecedente directo es **DiscreteBERT** (Baevski, Auli y Mohamed, 2019), que ya hacía predicción enmascarada de unidades discretas pero cuantiza la entrada y **le entrega al Transformer los tokens discretos**, no la onda. HuBERT insiste en que la entrada debe ser continua, para pasar tanta información como sea posible a las capas de atención, y que la cuantización debe existir **solo del lado del objetivo**. Los números respaldan la insistencia: con 10 horas etiquetadas DiscreteBERT da 5.9/14.1 en test-clean/test-other contra 4.3/9.4 de HuBERT BASE, con el mismo objetivo formal.

La familia rival es el **pseudo-etiquetado** (*self-training*): IPL, slimIPL, Noisy Student. El paper le atribuye dos desventajas estructurales: el estudiante solo puede imitar a un maestro limitado por su cantidad de datos supervisados, mientras que un pretexto autosupervisado obliga a representar *toda* la señal; y el pseudo-etiquetado orienta el aprendizaje hacia **una** tarea downstream, mientras que las features autosupervisadas generalizan a muchas. Este segundo punto resultó profético.

### Los tres problemas específicos del habla

Esta es la parte del paper que hay que leer con la [Clase 39](/clases/clase-39) al lado. El abstract enumera **tres problemas que el habla tiene y el texto no**:

> *"Self-supervised approaches for speech representation learning are challenged by three unique problems: (1) there are multiple sound units in each input utterance, (2) there is no lexicon of input sound units during the pre-training phase, and (3) sound units have variable lengths with no explicit segmentation."*

**(1) Hay múltiples unidades sonoras por enunciado.** Esto rompe el supuesto de *instance classification* que sostenía a casi todo el [SSL de visión](/fundamentos/aprendizaje-autosupervisado) de la época (SimCLR, MoCo, BYOL, SwAV). Ahí la unidad de análisis es la imagen entera: se la aumenta dos veces y se pide que ambas vistas colapsen al mismo punto del espacio latente. Un enunciado de 5 segundos contiene unos 50 fonemas de identidades distintas; pedir que todo el enunciado colapse a un punto destruye exactamente la información que se quiere aprender. La consecuencia es que el SSL de habla tiene que operar **a nivel de trama**, no a nivel de ejemplo.

**(2) No hay un léxico de unidades sonoras durante el preentrenamiento.** En NLP el tokenizador viene dado: WordPiece, BPE, SentencePiece. Se puede escribir $p(w \mid \text{contexto})$ con un softmax sobre 30.000 entradas porque las 30.000 entradas existen. En habla la pregunta previa es *cuáles son las clases*. Sin léxico, cualquier pérdida predictiva es imposible de **escribir**: no hay sobre qué poner el softmax. No es una dificultad de ingeniería, es una imposibilidad de formulación.

**(3) Las unidades sonoras tienen longitud variable y sin segmentación explícita.** Un fonema puede durar 30 ms o 300 ms según el hablante, el contexto y la prosodia, y los límites son físicamente difusos: la coarticulación hace que la transición de `/s/` a `/t/` sea un continuo. Nadie escribe espacios en la señal. En texto el enunciado llega presegmentado por convención ortográfica.

{{< concept-alert type="clave" >}}
El tercer problema —**unidades de longitud variable sin segmentación explícita**— es, palabra por palabra, la segunda objeción que levanta el PDF de la [Clase 39](/clases/clase-39) contra los Transformers en audio. La diferencia está en el rol: en la clase aparece como razón para no usarlos; en el paper aparece como el problema que el método viene a resolver. Ver la sección final de esta página.
{{< /concept-alert >}}

## La contribución: el maestro de clustering offline

### La construcción

Sea $X = [x_1, \dots, x_T]$ un enunciado de $T$ tramas. Se aplica un **modelo de clustering** $h$ —k-means en el caso por defecto— que produce

$$h(X) = Z = [z_1, \dots, z_T], \qquad z_t \in [C]$$

donde $z_t$ es una variable categórica de $C$ clases. Eso es todo el "maestro": un k-means. No es una red, no se entrena con gradiente, no ve etiquetas. Se ajusta una vez, offline, y produce una etiqueta por trama.

La analogía que propone el paper es con el **alineamiento forzado** de los sistemas semi-supervisados clásicos, donde un modelo acústico entrenado con pares texto-habla produce etiquetas pseudo-fonéticas por trama. HuBERT hace lo mismo pero sin el texto. Citando la tradición previa de descubrimiento no supervisado de unidades acústicas (Lee y Glass 2012, Ondel et al. 2016, los HMM-VAE de Ebbers et al. 2017), observa que incluso los modelos más ingenuos —k-means, GMM— *"infer hidden units that exhibit non-trivial correlation with the underlying acoustic units"*. Correlación no trivial: no es que k-means descubra fonemas, es que descubre algo que correlaciona con fonemas lo suficiente para servir de andamio.

Con $Z$ en mano, el problema queda escrito en el formato de [BERT](/fundamentos/pretraining-bert): hay una secuencia, hay un vocabulario de $C$ símbolos, se enmascara y se predice. **El aporte de HuBERT no es el objetivo (es el de BERT) ni la arquitectura (es la de wav2vec 2.0): es la fabricación del vocabulario.**

### Por qué funciona con etiquetas malas: el argumento de la consistencia

Esta es la parte contraintuitiva, y es la que sostiene todo lo demás.

Una etiqueta k-means sobre MFCC es **mala** en un sentido preciso y medible. El paper reporta que el clustering de [MFCC](/fundamentos/mfcc-y-escala-mel) con $C = 100$ alcanza (pureza de clúster, pureza de fonema, PNMI) = (0.099, 0.335, 0.255). *Pureza de fonema* de 0.335 significa que si se transcribiera cada clúster con su fonema más probable, se acertaría el 33.5% de las tramas. **Dos tercios de las etiquetas están mal.** Una tasa de error de etiquetado del 66% destruiría cualquier sistema supervisado.

El argumento del paper es que **la pérdida enmascarada no necesita que las etiquetas sean correctas; necesita que sean una función determinista y estable de la acústica local.** Desarrollado:

Supongamos que $z_t = h(x_t)$ es una función determinista de la trama, cualquiera sea. Para predecir $z_t$ en una posición enmascarada, el modelo no tiene acceso a $x_t$: solo tiene el contexto $X \setminus \{x_t\}$. La tarea que enfrenta no es "clasificar la trama $t$", sino

$$\text{estimar } p\big(h(x_t) \mid \text{contexto acústico de } t\big)$$

Esa cantidad solo se estima bien si el modelo aprende dos cosas: cómo se ve la señal alrededor —**modelado acústico**— y qué restricciones impone la estructura secuencial del habla sobre lo que puede ocurrir en $t$ —**modelado de lenguaje**. Si en el contexto se oye `/k/ /a/ /s/` y luego `/a/`, la trama faltante está fuertemente restringida por la fonotáctica y el léxico, independientemente de cómo se llame la clase. **El nombre de la clase es arbitrario; la estructura de coocurrencia entre clases no lo es.**

De ahí que lo que se necesita del maestro sea **consistencia**: que la misma configuración acústica reciba siempre la misma etiqueta. Si $h$ es consistente, $Z$ hereda de $X$ toda su estructura secuencial —transiciones, duraciones, fonotáctica, sintaxis en la medida en que se refleje en la cadena sonora— aunque las clases sean particiones arbitrarias del espacio acústico. Si $h$ fuera ruidoso (la misma trama etiquetada distinto en distintas ocurrencias), esa estructura se destruiría y no quedaría nada que aprender.

{{< concept-alert type="clave" >}}
Un maestro **perfectamente consistente pero "incorrecto"** —por ejemplo, uno que parte cada fonema en tres subclases según el hablante— sigue definiendo un lenguaje formal sobre $[C]^*$ con estructura rica. Un maestro **"correcto" pero ruidoso** —uno que acierta el fonema el 90% de las veces al azar— define una cadena parcialmente aleatoria. **El primero es mejor material de entrenamiento que el segundo.**
{{< /concept-alert >}}

El paper valida el supuesto de consistencia empíricamente antes de usarlo: ajusta k-means diez veces por configuración, con distintas semillas y distintos tamaños de datos de ajuste, y reporta media y desviación estándar del PNMI.

| Feature | $C$ | 1 h | 10 h | 100 h |
|---|---|---|---|---|
| MFCC | 100 | $0.251 \pm 0.001$ | $0.253 \pm 0.001$ | $0.253 \pm 0.001$ |
| MFCC | 500 | $0.283 \pm 0.001$ | $0.285 \pm 0.000$ | $0.287 \pm 0.001$ |
| BASE-it1-L6 | 100 | $0.563 \pm 0.012$ | $0.561 \pm 0.012$ | $0.575 \pm 0.008$ |
| BASE-it1-L6 | 500 | $0.680 \pm 0.005$ | $0.684 \pm 0.003$ | $0.686 \pm 0.004$ |

Dos conclusiones prácticas: k-means es **estable** (la desviación estándar está en el tercer decimal) y ajustarlo con más datos apenas ayuda (la ganancia máxima es 0.012 de PNMI). Lo segundo importa por ingeniería: la implementación carga toda la matriz de features en RAM y las del Transformer son de 768 dimensiones, así que poder ajustar con 1 hora en vez de 960 es la diferencia entre viable e inviable.

**La deuda con DeepCluster.** El paper reconoce la inspiración de **DeepCluster** (Caron et al., ECCV 2018), que en visión alterna entre agrupar las features de la red con k-means y usar las asignaciones como pseudo-etiquetas. La diferencia declarada —*"HuBERT benefits from the masked prediction loss over speech sequences to represent their sequential structure"*— no es cosmética: DeepCluster clasifica **la imagen completa** en su clúster, así que el gradiente empuja a la red a reproducir la partición y el techo del método es el clustering mismo. HuBERT clasifica **una trama que la red no puede ver**. Ese modo de falla existe también en HuBERT y tiene nombre: es exactamente lo que ocurre con $\alpha = 0$, que produce un WER de 96.37%. **La máscara es lo que impide que el estudiante degenere en copia del maestro.**

## El proceso iterado de dos pasos

**Iteración 1: k-means sobre MFCC.** Para generar las etiquetas de la primera iteración sobre las 960 horas de LibriSpeech se corre k-means con **$C = 100$ clústeres sobre features MFCC de 39 dimensiones** —13 coeficientes cepstrales más sus derivadas de primer y segundo orden—, o sea el frontend de ASR de 1985. La elección es deliberadamente humilde: *"it is one of the most naive unit discovery models that can be treated as modeling an isotropic Gaussian with the same scalar variance for each acoustic unit."* La humildad es parte del argumento: si el método funcionara solo con un maestro sofisticado, el aporte sería el maestro; que funcione con k-means sobre MFCC es la evidencia de que lo que importa es la pérdida enmascarada. Los detalles son directamente accionables: `MiniBatchKMeans` de `scikit-learn`, minibatch de **10.000 tramas**, inicialización **k-means++ con 20 reinicios aleatorios**.

**Iteración 2: k-means sobre las latentes del propio modelo.** Terminada la iteración 1 se tiene un HuBERT BASE preentrenado (sin ajuste fino), y la premisa es que *"since we expect a pre-trained model to provide better representations than the raw acoustic feature such as MFCCs, we can create a new generation of clusters by training a discrete latent model over the learned latent representations."* Concretamente: se extraen las activaciones de la **6.ª capa Transformer** del modelo de la iteración 1 y se corre k-means con **$C = 500$**; como la dimensión es 768 y la matriz completa de 960 h no cabe en memoria, se muestrea al azar el **10% de los datos**. Esas etiquetas alimentan la iteración 2, entrenada por 400k pasos.

Por qué mejora, con los números al lado:

| Maestro | $C$ | PNMI | dev-other WER ($\alpha = 1$) |
|---|---|---|---|
| k-means sobre MFCC | 100 | 0.243 | 17.86 |
| k-means sobre BASE-it1, capa 6 | 500 | 0.637 | 11.91 |
| k-means sobre BASE-it2, capa 9 | 500 | 0.704 | 10.75 |
| Chenone (alineamiento forzado supervisado) | 8976 | 0.809 | 10.38 |

El PNMI del maestro salta de 0.243 a 0.637 en una sola iteración: **el modelo entrenado con etiquetas malas produce representaciones que, agrupadas, dan etiquetas mucho mejores que las que lo entrenaron.** Esa es la definición operativa de bootstrapping, y es lo mismo que ocurre en el pseudo-etiquetado iterativo, con la diferencia de que aquí no hay ninguna etiqueta humana en el ciclo.

Hay una asimetría que explica por qué el ciclo es virtuoso y no vicioso. El maestro k-means es **sin memoria**: mira un vector de features local y lo asigna a un centroide. El modelo HuBERT es **contextual**: para producir la latente de la capa 6 en la posición $t$ ha integrado, vía [self-attention](/fundamentos/self-attention), todo el enunciado. Al agrupar esas latentes contextuales, los clústeres ya no son "regiones del espacio espectral" sino "regiones del espacio de estados fonéticos contextualizados". **El clustering hereda gratis el trabajo de desambiguación que hizo el Transformer.**

**La tercera iteración implícita.** Un detalle que suele citarse mal: los modelos **LARGE y X-LARGE no reinician el proceso desde MFCC**. En cambio:

> *"Instead of restarting the iterative process from clustering MFCC features, we extract features from the 9-th transformer layer of the second iteration BASE HuBERT for clustering and use those labels for training these two models. Hence, these two models can also be seen as the third iteration models."*

La cadena completa es: MFCC → BASE-it1 (250k pasos, LS-960, 32 GPU) → k-means sobre la capa 6 → BASE-it2 (400k pasos, LS-960) → k-means sobre la capa 9 con $C = 500$ → LARGE (400k pasos, LL-60k, 128 GPU) y X-LARGE (400k pasos, LL-60k, 256 GPU), ambos entrenados **en paralelo desde el mismo conjunto de etiquetas**.

Consecuencias que conviene tener presentes:

- Los modelos entrenados sobre 60.000 horas usan etiquetas producidas por un modelo entrenado sobre 960. **El maestro es más chico y ha visto 60 veces menos audio que el estudiante.**
- La capa de extracción cambia de la 6 (para it1) a la 9 (para it2). No es arbitrario: el perfil de calidad por capa se desplaza hacia arriba entre iteraciones.
- LARGE y X-LARGE hacen **una sola** pasada de preentrenamiento cada uno. Todo el costo iterativo está amortizado en BASE.

## La función de pérdida

Sea $M \subset [T]$ el conjunto de índices enmascarados y $\tilde{X} = r(X, M)$ la versión corrompida de $X$, donde cada $x_t$ con $t \in M$ se reemplaza por un **embedding de máscara aprendido** $\tilde{x}$. El modelo $f$ consume $\tilde{X}$ y produce una distribución $p_f(\cdot \mid \tilde{X}, t)$ en cada paso. La pérdida sobre posiciones enmascaradas, tal cual la escribe el paper:

$$L_m(f; X, M, Z) = \sum_{t \in M} \log p_f(z_t \mid \tilde{X}, t)$$

y $L_u$ es idéntica salvo que suma sobre $t \notin M$. La pérdida final combina ambas con un solo hiperparámetro:

$$L = \alpha L_m + (1 - \alpha)\, L_u$$

(El paper llama a ambas "cross-entropy loss" pero las escribe sin el signo negativo, o sea como log-verosimilitud a maximizar; ver las erratas al final.)

### La distribución sobre códigos

Dada la salida del Transformer $o_t$, la distribución sobre los $C$ códigos se parametriza como

$$p_f^{(k)}(c \mid \tilde{X}, t) = \frac{\exp\big(\mathrm{sim}(A^{(k)} o_t,\, e_c)/\tau\big)}{\sum_{c'=1}^{C} \exp\big(\mathrm{sim}(A^{(k)} o_t,\, e_{c'})/\tau\big)}$$

donde $A^{(k)}$ es una matriz de proyección, $e_c$ es el embedding aprendido del código $c$, $\mathrm{sim}(\cdot,\cdot)$ es la **similitud coseno** y $\tau = 0.1$ escala los logits. Dos decisiones no obvias la separan de un softmax lineal estándar:

- **Similitud coseno en vez de producto punto.** Al normalizar ambos vectores, la magnitud de $A o_t$ deja de influir en los logits y solo importa la dirección. Esto elimina un grado de libertad degenerado —inflar la norma de la salida para saturar el softmax— y convierte el objetivo en uno de *metric learning*: los estados ocultos deben apuntar hacia el embedding de su código. Con $\tau = 0.1$, la similitud vive en $[-1, 1]$ y los logits en $[-10, 10]$: rango suficiente para distribuciones agudas sin saturación patológica.
- **Embeddings de código aprendidos $e_c$.** La capa de salida es literalmente una tabla de embeddings de $C$ entradas, con la misma forma que la tabla de entrada de BERT. La consecuencia útil es que **el espacio de códigos adquiere geometría**: dos clústeres que aparecen en contextos parecidos terminan con embeddings parecidos, lo que suaviza el objetivo —equivocarse entre dos códigos vecinos cuesta menos que entre dos lejanos—. Con un maestro cuya pureza de fonema es 0.335, esa tolerancia importa mucho.

### El análisis de $\alpha$: los dos extremos

El paper interpreta $\alpha$ como una perilla entre dos regímenes clásicos del reconocimiento de habla, y la interpretación es exacta.

**$\alpha = 0$ (pérdida solo sobre lo no enmascarado).** El modelo ve $x_t$ y tiene que predecir $z_t = h(x_t)$. Como $h$ es determinista, la tarea se reduce a aprender la función $h$. El paper: *"this limits the learning process to mimicking the clustering model."* Es análogo al **modelado acústico** de un sistema híbrido HMM-DNN: mapear tramas a estados. El techo del método es el maestro.

**$\alpha = 1$ (pérdida solo sobre lo enmascarado).** El modelo no ve $x_t$ y tiene que inferir $z_t$ del contexto. El paper: *"analogous to language modeling. It forces the model to learn both the acoustic representation of unmasked segments and the long-range temporal structure of the speech data."* Nótese la doble exigencia: hay que representar bien lo que sí se ve (acústica) **y** modelar cómo se encadena (lenguaje). Un solo objetivo produce ambos modelos. La hipótesis explícita es que *"the setup with $\alpha = 1$ is more resilient to the quality of cluster targets"*.

**Los valores intermedios.** Con $\alpha = 0.5$ ambos términos pesan igual. El paper no lo dice, pero la asimetría de dificultad importa: predecir $h(x_t)$ *viendo* $x_t$ es fácil y su gradiente es grande y limpio; inferirlo del contexto es difícil. El término fácil domina el aprendizaje temprano y arrastra al modelo hacia la imitación del maestro.

Los resultados confirman todo esto con una limpieza poco común:

| Maestro | $C$ | PNMI | $\alpha = 1.0$ | $\alpha = 0.5$ | $\alpha = 0.0$ |
|---|---|---|---|---|---|
| Chenone (top-line supervisado) | 8976 | 0.809 | 10.38 | **9.16** | 9.79 |
| k-means sobre MFCC | 50 | 0.227 | **18.68** | 31.07 | 94.60 |
| k-means sobre MFCC | 100 | 0.243 | **17.86** | 29.57 | 96.37 |
| k-means sobre MFCC | 500 | 0.276 | **18.40** | 33.42 | 97.66 |
| k-means sobre BASE-it1, capa 6 | 500 | 0.637 | **11.91** | 13.47 | 23.29 |
| k-means sobre BASE-it2, capa 9 | 500 | 0.704 | **10.75** | 11.59 | 13.79 |

(WER % en dev-other. Modelos preentrenados 100k pasos y ajustados sobre el split de 10 h de Libri-Light.)

Lo que hay que leer ahí:

1. **Con maestros malos, $\alpha = 0$ es catastrófico: 94-98% de WER.** No es "peor": es colapso total. El modelo aprendió a ser un k-means sobre MFCC y no aprendió nada de habla.
2. **Con maestros malos, $\alpha = 1$ es el único régimen viable**, y la degradación al pasar a $\alpha = 0.5$ ya es brutal ($17.86 \to 29.57$).
3. **La penalización por incluir la pérdida no enmascarada decrece monótonamente con la calidad del maestro.** La brecha $\alpha=0$ menos $\alpha=1$ vale $+78.5$ para MFCC-100, $+11.4$ para it1-L6, $+3.0$ para it2-L9 y $-0.6$ para chenone.
4. **Con el maestro supervisado, $\alpha = 0.5$ gana.** Cuando las etiquetas son casi correctas, el término no enmascarado deja de ser distractor y pasa a ser señal densa útil. La regla completa es: **el $\alpha$ óptimo es función decreciente de la calidad del maestro.**
5. La distancia entre el mejor maestro no supervisado (10.75) y el top-line supervisado con su mejor $\alpha$ (9.16) es de **1.59 puntos de WER absolutos**. Ese es el precio total de no tener transcripciones en el ciclo.

El paper **no declara explícitamente qué $\alpha$ usan los modelos principales**; se infiere del abstract (*"applying the prediction loss over the masked regions only"*) y de esta tabla que es $\alpha = 1$.

**Ensembles de clustering.** La extensión formal es directa: si $Z^{(k)}$ son las etiquetas del $k$-ésimo modelo de clustering,

$$L_m\big(f; X, \{Z^{(k)}\}_k, M\big) = \sum_{t \in M} \sum_{k} \log p_f^{(k)}\big(z_t^{(k)} \mid \tilde{X}, t\big)$$

Es *multi-task learning* con tareas creadas por clustering no supervisado, donde cada modelo aporta una granularidad distinta. Combinado con **cuantización por producto** (partir el espacio de features en subespacios y cuantizar cada uno por separado), el ensemble de las tres cuantizaciones baja el WER de dev-other de 17.86 a **16.73** sobre maestros MFCC.

{{< concept-alert type="advertencia" >}}
**Los modelos principales del paper no usan ensembles.** Toda la tabla de resultados se obtiene con un único k-means. La técnica queda documentada como ablación, no desplegada, y a veces se cita mal como si formara parte del sistema.
{{< /concept-alert >}}

## La arquitectura

### Encoder convolucional de forma de onda

Idéntico para las tres configuraciones: **7 capas convolucionales de 512 canales**, con

- strides: $[5, 2, 2, 2, 2, 2, 2]$
- anchos de kernel: $[10, 3, 3, 3, 3, 2, 2]$

El producto de los strides es $5 \cdot 2^6 = 320$. A 16 kHz eso da **una trama cada 20 ms, o sea 50 tramas por segundo**, cifra que el paper declara explícitamente.

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

{{< concept-alert type="recordar" >}}
**400 muestras = 25 ms** a 16 kHz. Esta cifra es **derivada de los strides y kernels de la tabla del paper, no está reportada en el paper**. Es exactamente el tamaño de ventana canónico del análisis de habla: el frontend convolucional reproduce, aprendiéndolos, los parámetros de ventaneo que la comunidad de DSP fijó por razones psicoacústicas —25 ms es aproximadamente el rango donde la señal puede considerarse cuasi-estacionaria y donde cabe al menos un par de períodos de pitch de una voz grave—. Ver [MFCC y escala Mel](/fundamentos/mfcc-y-escala-mel).
{{< /concept-alert >}}

Este bloque es también **la respuesta al costo cuadrático de la self-attention**: 10 segundos de audio son 160.000 muestras, pero solo 500 tramas después del encoder. Sin ese factor de 320×, un Transformer sobre habla sería literalmente imposible. Durante el fine-tuning de ASR el encoder convolucional **se congela**.

### Encoder Transformer y cabezal

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
| **GPU / pasos** | 32 / 250k + 400k | 128 / 400k | 256 / 400k |
| **Batch por GPU** | ≤ 87.5 s de audio | ≤ 56.25 s | ≤ 22.5 s |
| **LR pico** | 5e-4 | 1.5e-3 | 3e-3 |

BASE y LARGE siguen de cerca a wav2vec 2.0 BASE y LARGE. X-LARGE es original de este paper y se dimensiona a la escala del [Conformer](/papers/conformer-gulati-2020) XXL de Zhang et al., el competidor de escala en 2020-2021. Nótese que X-LARGE duplica la profundidad de LARGE (48 capas) pero solo sube la dimensión de 1024 a 1280 y mantiene 16 cabezas: **el escalado es casi puramente en profundidad**. Con batch por GPU de 22.5 s y 256 GPU, el batch efectivo es de unos 5.760 s = 1.6 h de audio por paso. El **layerdrop 0.05 solo en BASE** es coherente con el régimen de datos: BASE se entrena sobre 960 h y necesita regularización estocástica de profundidad; LARGE y X-LARGE ven 60.000 h y el sobreajuste deja de ser el cuello de botella.

**Cabezal.** Tras el Transformer viene la capa de proyección $A$ (dimensión 256/768/1024) y el embedding de código $e_c$. Para el fine-tuning de ASR se **elimina la proyección** y se la reemplaza por un softmax inicializado al azar, entrenado con **CTC**. El vocabulario de CTC es minimalista: 26 letras del alfabeto inglés, un token de espacio, un apóstrofo y el símbolo *blank* — 29 símbolos. No hay tokenizador subpalabra, no hay léxico de pronunciación, no hay diccionario fonético.

Un detalle que el paper **no describe**: la codificación posicional. Solo dice que sigue la arquitectura de wav2vec 2.0, que usa una **capa convolucional de embedding posicional relativo** (kernel 128, 16 grupos) en vez de codificaciones sinusoidales o absolutas aprendidas. Es relevante para el argumento sobre dependencias largas: no hay una longitud máxima codificada en el modelo.

**Decodificación.** Se usa la búsqueda por haz de wav2letter++ envuelta en fairseq, optimizando $\log p_{\text{CTC}}(Y \mid X) + w_1 \log P_{\text{LM}}(Y) + w_2 |Y|$ con los pesos buscados por optimización bayesiana. Se consideran LM de n-gramas y LM Transformer, ambos entrenados sobre los datos oficiales de modelado de lenguaje de LibriSpeech. **Esto es esencial para interpretar las cifras titulares** y se retoma en la sección de erratas.

## El enmascaramiento

HuBERT adopta el esquema de SpanBERT y wav2vec 2.0: se seleccionan al azar el $p\%$ de los pasos temporales como **índices de inicio**, y desde cada uno se enmascara un **span de $l$ pasos**. Los valores por defecto son $l = 10$ y $p = 8\%$.

{{< concept-alert type="advertencia" >}}
**$p$ no es la fracción enmascarada, es la probabilidad de inicio.** Con spans de largo 10 que pueden solaparse, la fracción esperada de tramas cubiertas es

$$1 - (1 - p)^{l} = 1 - 0.92^{10} \approx 0.57$$

Es decir, **alrededor del 57% de las tramas quedan enmascaradas, no el 8%** que suele citarse. El paper no reporta esta cifra derivada, así que la confusión es fácil; la fórmula reproduce el ~49% que wav2vec 2.0 declara con su $p = 6.5\%$, lo que valida la aproximación. Comparado con el 15% de BERT es una tasa de enmascaramiento enorme, y la razón es la redundancia de la señal: tramas contiguas de 20 ms dentro del mismo fonema son casi copias.
{{< /concept-alert >}}

**El span de 10 tramas cubre 200 ms.** A 20 ms por trama, un span borra el equivalente a dos o tres fonemas completos. Esto es esencial: si se enmascarara una sola trama, la tarea sería trivial por interpolación local —el modelo copiaría el vecino—. Los spans largos hacen que la interpolación acústica sea insuficiente y obligan a apoyarse en estructura de nivel superior. Es el mismo razonamiento por el que SpanBERT enmascara spans en vez de tokens sueltos, amplificado por la redundancia del audio. La ablación barre $p \in \{2, 4.5, 6.5, 8, 9\}$ y encuentra el óptimo en $p = 8\%$, con el WER subiendo hacia ambos extremos: poco enmascaramiento da una tarea demasiado fácil; demasiado destruye el contexto necesario para resolverla.

**Dónde se enmascara: sobre las salidas del encoder convolucional, no sobre la onda cruda.** El paper lo declara pero no lo argumenta. Las razones:

1. **Alineación con el objetivo.** Las etiquetas $z_t$ están definidas por trama, a 50 Hz. Enmascarar en el dominio de la onda dejaría los límites de máscara desalineados respecto de las tramas objetivo, y el campo receptivo de 25 ms produciría tramas parcialmente contaminadas en los bordes.
2. **Fuga de información.** Poner ceros en la onda cruda no borra información: **crea una discontinuidad brutal que el encoder convolucional detectaría trivialmente**, y la envolvente de energía alrededor del silencio artificial revela mucho sobre el contexto. Un embedding de máscara aprendido, sustituido a nivel de trama, es una señal limpia de "aquí no hay dato".
3. **El embedding de máscara tiene que vivir en el mismo espacio que las tramas.** En BERT, `[MASK]` es una entrada del vocabulario; aquí $\tilde{x}$ es un vector aprendido del mismo tamaño que la salida convolucional. Ese objeto no tiene análogo en el dominio de la señal: no existe "la forma de onda de la máscara".
4. **El encoder convolucional se congela en fine-tuning**, así que mantener el enmascaramiento por encima de él lo deja como un frontend puro y estable.

La consecuencia conceptual —central para la última sección de esta página— es que **el encoder convolucional es el que fabrica las "entidades discretas" sobre las que opera la self-attention.**

## Resultados

**Datos.** El preentrenamiento (sin etiquetas) usa las 960 h completas de LibriSpeech o las **60.000 h de Libri-Light**; ambos derivan de LibriVox, grabaciones en inglés de audiolibros de dominio público leídos por voluntarios. El fine-tuning (con etiquetas) usa cinco particiones: Libri-Light de **10 minutos, 1 hora y 10 horas**, más LibriSpeech **100 h** y **960 h**. Un detalle metodológico que importa: los tres splits de Libri-Light tienen **la mitad del audio de `train-clean-*` y la otra mitad de `train-other-500`**, o sea que los regímenes de bajos recursos están balanceados entre condiciones limpia y difícil por construcción, no son "10 minutos de audio fácil".

| Modelo | Datos sin etiquetar | LM | dev-clean | dev-other | test-clean | test-other |
|---|---|---|---|---|---|---|
| **10 minutos etiquetados** | | | | | | |
| DiscreteBERT | LS-960 | 4-gram | 15.7 | 24.1 | 16.3 | 25.2 |
| wav2vec 2.0 BASE | LS-960 | 4-gram | 8.9 | 15.7 | 9.1 | 15.6 |
| wav2vec 2.0 LARGE | LL-60k | Transformer | 4.6 | 7.9 | 4.8 | 8.2 |
| HuBERT BASE | LS-960 | 4-gram | 9.1 | **15.0** | 9.7 | **15.3** |
| HuBERT LARGE | LL-60k | Transformer | **4.3** | **7.0** | **4.7** | **7.6** |
| HuBERT X-LARGE | LL-60k | Transformer | 4.4 | **6.1** | **4.6** | **6.8** |
| **1 hora etiquetada** | | | | | | |
| DiscreteBERT | LS-960 | 4-gram | 8.5 | 16.4 | 9.0 | 17.6 |
| wav2vec 2.0 BASE | LS-960 | 4-gram | **5.0** | **10.8** | **5.5** | 11.3 |
| wav2vec 2.0 LARGE | LL-60k | Transformer | 2.9 | 5.4 | **2.9** | 5.8 |
| HuBERT BASE | LS-960 | 4-gram | 5.6 | 10.9 | 6.1 | 11.3 |
| HuBERT LARGE | LL-60k | Transformer | **2.6** | **4.9** | **2.9** | **5.4** |
| HuBERT X-LARGE | LL-60k | Transformer | **2.6** | **4.2** | **2.8** | **4.8** |
| **10 horas etiquetadas** | | | | | | |
| SlimIPL | LS-960 | 4-gram + Transformer | 5.3 | 7.9 | 5.5 | 9.0 |
| DiscreteBERT | LS-960 | 4-gram | 5.3 | 13.2 | 5.9 | 14.1 |
| wav2vec 2.0 BASE | LS-960 | 4-gram | **3.8** | 9.1 | 4.3 | 9.5 |
| wav2vec 2.0 LARGE | LL-60k | Transformer | 2.4 | 4.8 | 2.6 | 4.9 |
| HuBERT BASE | LS-960 | 4-gram | 3.9 | **9.0** | 4.3 | **9.4** |
| HuBERT LARGE | LL-60k | Transformer | **2.2** | **4.3** | **2.4** | **4.6** |
| HuBERT X-LARGE | LL-60k | Transformer | **2.1** | **3.6** | **2.3** | **4.0** |
| **100 horas etiquetadas** | | | | | | |
| Noisy Student | LS-860 | LSTM | 3.9 | 8.8 | 4.2 | 8.6 |
| DiscreteBERT | LS-960 | 4-gram | 4.0 | 10.9 | 4.5 | 12.1 |
| wav2vec 2.0 BASE | LS-960 | 4-gram | 2.7 | 7.9 | 3.4 | **8.0** |
| wav2vec 2.0 LARGE | LL-60k | Transformer | 1.9 | 4.0 | **2.0** | 4.0 |
| HuBERT BASE | LS-960 | 4-gram | 2.7 | **7.8** | 3.4 | 8.1 |
| HuBERT LARGE | LL-60k | Transformer | **1.8** | **3.7** | 2.1 | **3.9** |
| HuBERT X-LARGE | LL-60k | Transformer | **1.7** | **3.0** | **1.9** | **3.5** |

Y el régimen de 960 h etiquetadas, donde compiten los sistemas completos:

| Categoría | Modelo | Datos sin etiquetar | LM | dev-clean | dev-other | test-clean | test-other |
|---|---|---|---|---|---|---|---|
| Supervisado | Conformer L | – | LSTM | – | – | 1.9 | 3.9 |
| Self-training | Noisy Student | LV-60k | LSTM | 1.6 | 3.4 | 1.7 | 3.4 |
| Preentrenamiento | wav2vec 2.0 LARGE | LL-60k | Transformer | 1.6 | 3.0 | 1.8 | 3.3 |
| Preentrenamiento | Conformer XXL | LL-60k | LSTM | 1.5 | 3.0 | 1.5 | 3.1 |
| Pre + self-training | wav2vec 2.0 + self-training | LL-60k | Transformer | 1.1 | 2.7 | 1.5 | 3.1 |
| Pre + self-training | Conformer XXL + Noisy Student | LL-60k | LSTM | 1.3 | 2.6 | 1.4 | **2.6** |
| **Este trabajo** | HuBERT LARGE | LL-60k | Transformer | 1.5 | 3.0 | 1.9 | 3.3 |
| **Este trabajo** | HuBERT X-LARGE | LL-60k | Transformer | 1.5 | **2.5** | 1.8 | 2.9 |

**Dónde la ganancia es grande.** En 10 minutos con LM Transformer, HuBERT LARGE baja test-other de 8.2 a 7.6 (−7.3% relativo) y X-LARGE lo lleva a 6.8 (−17.1% respecto de wav2vec 2.0 LARGE). En 1 hora, test-other pasa de 5.8 a 4.8 con X-LARGE (−17.2%). En 10 horas, de 4.9 a 4.0 (−18.4%). **La mejora se concentra sistemáticamente en `-other`**, la partición con hablantes y condiciones de grabación más difíciles. Con 960 h, HuBERT X-LARGE logra el **mejor dev-other absoluto de la tabla (2.5)**, incluyendo a los métodos combinados.

**Dónde la ganancia es marginal o negativa.** HuBERT **BASE no supera a wav2vec 2.0 BASE** en los regímenes muy bajos: en 10 minutos es 0.6 puntos peor en test-clean, y en 1 hora es peor en tres de las cuatro columnas. En 10 h y 100 h empata. Y con 960 h, HuBERT LARGE está empatado con wav2vec 2.0 LARGE. La lectura correcta se detalla en la sección de erratas.

**La comparación con DiscreteBERT** sí es aplastante y es la conceptualmente limpia, porque el objetivo formal es idéntico: con 10 h, 5.9/14.1 contra 4.3/9.4; con 10 minutos, 16.3/25.2 contra 9.7/15.3. El paper atribuye la brecha a que **la entrada debe ser la onda, no unidades cuantizadas** —cuantizar la entrada pierde información irrecuperable— y a que el refinamiento iterativo termina superando a un maestro fijo.

**El efecto del número de pasos.** Entrenar más ayuda de forma consistente y sin saturarse: con maestro MFCC-100, el WER de dev-other cae de 17.86 (100k pasos) a 12.97 (250k), 12.32 (400k) y 11.68 (800k) — un 35% relativo, y la curva todavía baja al final. A 250k pasos, la comparación directa con DiscreteBERT (12.97 contra 26.6) es un factor de 2 con el mismo objetivo formal y las mismas features de partida. El paper conjetura que DiscreteBERT falla porque usa **13.500 unidades**: con tantos clústeres, las unidades codifican variación inter e intra-hablante en vez de conceptos fonéticos amplios.

## Ablations y análisis

### Las métricas de calidad del maestro

Para medir la correlación entre los clústeres y la fonética real se derivan transcripciones fonéticas alineadas trama a trama con un sistema ASR híbrido y se estima la distribución conjunta entre etiqueta fonética $y$ y etiqueta k-means $z$ por conteo, $p_{yz}(i, j) = \frac{1}{T}\sum_{t} [\,y_t = i \wedge z_t = j\,]$. Sobre ella se definen tres métricas, **puramente diagnósticas** (usan un alineamiento forzado supervisado que el método nunca ve durante el entrenamiento):

- **Pureza de fonema.** Para cada clúster $j$ se toma el fonema más probable $y^*(j) = \arg\max_i p_{yz}(i,j)$ y se promedia su probabilidad condicional: $\text{PhnPur} = \mathbb{E}_{p_z(j)}\big[\,p_{y|z}(y^*(j) \mid j)\,\big]$. Es la exactitud de fonema a nivel de trama si se transcribiera cada clúster con su fonema más probable. Caveat que el propio paper señala: **no es comparable entre configuraciones con distinto número de unidades**, porque en el límite degenerado en que cada trama recibe etiqueta única la pureza sería del 100%.
- **Pureza de clúster.** La contraparte: $\text{ClsPur} = \mathbb{E}_{p_y(i)}\big[\,p_{z|y}(z^*(i) \mid i)\,\big]$ con $z^*(i)$ el clúster más probable del fonema $i$. Mide cuán concentrado está cada fonema en un solo clúster, y típicamente **decrece** al aumentar el número de unidades.
- **Información mutua normalizada por fonema (PNMI).** $\text{PNMI} = I(y; z)/H(y) = 1 - H(y \mid z)/H(y)$: el **porcentaje de incertidumbre sobre el fonema que se elimina al observar la etiqueta k-means**. Es la métrica principal del paper porque es la única de las tres que penaliza simultáneamente la sobresegmentación y la subsegmentación.

### Calidad por capa y por iteración

El análisis más informativo del paper toma los dos modelos BASE (it1 e it2), extrae features de las 12 capas Transformer más la entrada a la primera, ajusta tres k-means ($C \in \{100, 500, 1000\}$) y grafica las tres métricas contra el índice de capa. La referencia con MFCC es (ClsPur, PhnPur, PNMI) = (0.099, 0.335, 0.255) para $C=100$.

1. **Cualquier capa de HuBERT es mucho mejor que MFCC.** Con $C=100$, la pureza de clúster pasa de 0.099 a ~0.27 en el pico; el PNMI de 0.255 a ~0.69-0.72; la pureza de fonema de ~0.34 a ~0.72.
2. **BASE-it1 tiene un pico marcado alrededor de la capa 6 y luego se degrada dramáticamente**: el PNMI cae de ~0.69 en las capas 6-7 a ~0.40-0.47 en la capa 12. Esa es la justificación empírica de extraer de la capa 6 para el segundo clustering.
3. **BASE-it2 mejora monótonamente con la profundidad y no colapsa al final**: el PNMI se estabiliza en ~0.70-0.72 desde la capa 7 hasta la 12. De ahí la elección de la capa 9 para las etiquetas de LARGE y X-LARGE.

La explicación que el paper da para el colapso de las capas finales de it1 merece subrayarse:

> *"the quality of the last few layers degrades dramatically for BASE-it1, potentially because it is trained on target assignments of worse quality, and therefore the last few layers learn to mimic their bad label behavior."*

Traducido: **las capas superiores se especializan en resolver la tarea de salida, y si la tarea de salida es basura, las capas superiores aprenden basura.** Las capas intermedias, en cambio, están suficientemente lejos del objetivo como para construir la representación genérica que la tarea *requiere* sin arrastrarse a las idiosincrasias del maestro. Es el mismo fenómeno que se observa en BERT, pero aquí más extremo porque el objetivo es explícitamente ruidoso. La regla práctica que sigue vigente: **cuando el objetivo de preentrenamiento es de baja calidad, extraer del medio; cuando mejora, se puede extraer más arriba.**

### El efecto de $C$

Con maestro MFCC y $\alpha = 1$, el WER en dev-other es 18.68 para $C=50$, **17.86 para $C=100$** y 18.40 para $C=500$: óptimo en 100, y no monótono. El detalle relevante es que el PNMI del maestro **sí es monótono creciente** en $C$ (0.227 → 0.243 → 0.276) pero el WER no.

{{< concept-alert type="recordar" >}}
**El PNMI del maestro no predice el desempeño downstream cuando se compara entre distintos números de unidades.** La información mutua sube trivialmente al partir más fino, pero un vocabulario objetivo demasiado grande vuelve la predicción enmascarada demasiado difícil y ruidosa: con 500 clústeres sobre MFCC, los clústeres empiezan a codificar identidad de hablante y condiciones de canal, que son impredecibles desde el contexto fonético. En la segunda iteración, en cambio, $C = 500$ funciona bien, porque las features contextuales ya descartaron buena parte de la variación de hablante y 500 clústeres sobre ellas parten el espacio fonético, no el acústico crudo.
{{< /concept-alert >}}

### El resultado que valida la tesis

Reorganizando por PNMI creciente, con $\alpha = 1$:

| Maestro | PNMI | dev-other WER | $\Delta$ PNMI | $\Delta$ WER |
|---|---|---|---|---|
| k-means MFCC, $C=100$ | 0.243 | 17.86 | — | — |
| k-means BASE-it1-L6, $C=500$ | 0.637 | 11.91 | +0.394 | −5.95 |
| k-means BASE-it2-L9, $C=500$ | 0.704 | 10.75 | +0.067 | −1.16 |
| Chenone supervisado, $C=8976$ | 0.809 | 10.38 | +0.105 | −0.37 |

**La relación es monótona pero con retornos fuertemente decrecientes.** El primer salto de calidad del maestro (+0.394 de PNMI) compra 5.95 puntos de WER. El último (+0.105, y que requiere transcripciones humanas y un sistema HMM completo) compra 0.37.

{{< concept-alert type="clave" >}}
**La consistencia del maestro es condición suficiente para que el método arranque; la corrección sí importa, pero su valor marginal se agota rápido.** Con un maestro cuya exactitud fonética es del 33%, HuBERT llega a 17.86 de WER: malo, pero **funcional**, y funcional es todo lo que se necesita para que el bootstrapping tome vuelo. Dos iteraciones después, el sistema está a 0.37 puntos de lo que se lograría con alineamientos forzados supervisados.

El complemento imprescindible es la columna $\alpha = 0$: el mismo maestro malo, la misma arquitectura, los mismos datos, cambiando solo dónde se aplica la pérdida, da **96.37%** de WER. Eso demuestra que **el mecanismo que rescata al maestro malo no es el maestro sino la máscara**: es la imposibilidad de ver la trama objetivo lo que convierte "aprende esta partición arbitraria" en "aprende la estructura del habla que hace predecible esta partición arbitraria".
{{< /concept-alert >}}

## Limitaciones

**Costo de preentrenamiento.** El paper da una sola cifra de tiempo —*"training for 100k steps takes about 9.5 hours"*, referida a BASE sobre 32 GPU—, de donde salen unas **2.000 GPU-horas** para BASE sin contar clustering ni fine-tuning. **Para LARGE y X-LARGE no se reporta tiempo de reloj**, solo 400k pasos sobre 128 y 256 GPU. Un límite inferior grosero para X-LARGE sería ~9.700 GPU-horas suponiendo la misma velocidad por paso que BASE (irreal: 10× más parámetros y batch por GPU 4× menor); el número real es sustancialmente mayor. **El paper no permite calcular el costo total del sistema**, y esa omisión es en sí una limitación de reporte.

**El pipeline de múltiples etapas.** Es la limitación que los propios autores declaran en la conclusión: *"we plan to improve the HuBERT training procedure to consist of a single phase."* wav2vec 2.0 es end-to-end: un solo entrenamiento, un solo objetivo, sin artefactos intermedios. HuBERT requiere, en orden: extraer MFCC de 960 h → ajustar k-means → inferir etiquetas para todo el corpus → preentrenar 250k pasos → extraer activaciones de la capa 6 → ajustar k-means → inferir etiquetas → preentrenar 400k pasos → extraer activaciones de la capa 9 → ajustar k-means → inferir etiquetas → preentrenar LARGE/X-LARGE. Son cuatro artefactos intermedios materializados en disco (las etiquetas de 60.000 horas a 50 Hz son $1.08 \times 10^{10}$ enteros) y tres modelos de clustering versionados. Desde la perspectiva de producción es un DAG frágil: cada etapa es un punto de falla y de deriva de versiones, y no hay forma de reentrenar parcialmente. A esto se suma que **la elección de capa se descubre a posteriori**: las capas 6 y 9 no salen de ningún principio sino de mirar el perfil de calidad, que solo se puede construir después de entrenar el modelo, y cuyo diagnóstico requiere un alineamiento forzado supervisado — justo el recurso que el método pretende no necesitar.

**El sesgo hacia el inglés leído.** Todo el paper vive dentro del universo LibriVox: audiolibros de dominio público en inglés, leídos por voluntarios, sin ruido de fondo significativo, sin solapamiento de hablantes, sin habla espontánea, sin disfluencias, sin acentos no nativos en proporción representativa, sin cambio de código. La introducción argumenta explícitamente que el SSL es crucial para lenguas y dialectos con pocos recursos y ortografías no estandarizadas, **y ese argumento no se prueba en ningún experimento del paper**: preentrenamiento, fine-tuning y evaluación salen del mismo dominio.

**Las unidades descubiertas no son fonemas.** La mejor pureza de fonema observada es ~0.72, o sea que cerca de un tercio de las tramas se etiquetan con un clúster cuyo fonema mayoritario no corresponde. Los clústeres correlacionan con la fonética pero no la recuperan: mezclan identidad de hablante, contexto coarticulatorio, posición en la sílaba y condiciones de grabación. **Cualquier lectura de HuBERT como "descubridor de fonemas no supervisado" es incorrecta**, y el paper es cuidadoso en no hacerla.

**Todo se evalúa en ASR.** La conclusión promete explorar tareas más allá del ASR, pero el paper no lo hace. En un trabajo cuyo argumento central es que las features autosupervisadas generalizan a muchas tareas downstream —frente al pseudo-etiquetado, que se especializa en una—, evaluar una sola tarea es una tensión interna notable. La historia posterior le dio la razón, pero el paper no la demuestra.

## Por qué importa hoy

**HuBERT ganó la discusión sobre cómo se hace SSL de habla.** En 2021 competían tres familias: contrastiva (wav2vec 2.0), autorregresiva (APC, CPC) y predicción enmascarada de unidades discretas (DiscreteBERT, HuBERT). Se impuso la tercera, en buena medida por lo que el paper anticipa: el objetivo contrastivo tiene demasiadas piezas móviles —muestreo de negativos, pérdida de diversidad, recocido de Gumbel— mientras que una entropía cruzada sobre un vocabulario fijo es robusta y se afina sola.

**Textless NLP y la generación de habla.** Esta es la línea de impacto más grande, y no es sobre ASR. Si un modelo produce una secuencia de unidades discretas a partir del audio, se puede entrenar un modelo de lenguaje **sobre esas unidades** y generar habla sin pasar nunca por texto. **GSLM** (*Generative Spoken Language Modeling from Raw Audio*, Lakhotia et al., TACL 2021 — Kushal Lakhotia es coautor de HuBERT) construyó exactamente eso: HuBERT como tokenizador, un LM sobre unidades y un vocoder que sintetiza audio desde unidades, con la configuración que se volvió canónica en esa línea (**HuBERT BASE, capa 6, k-means con 50-200 clústeres**). De ahí salen la resíntesis de habla, la traducción directa habla-a-habla sin texto —útil precisamente para lenguas sin ortografía estandarizada, cerrando el argumento de la introducción— y los modelos de diálogo hablado. La distinción entre **tokens semánticos** (de un modelo SSL como HuBERT, que capturan contenido) y **tokens acústicos** (de un códec neuronal como SoundStream o EnCodec, que capturan timbre y detalle) es la arquitectura de referencia de **AudioLM** (Borsos et al., 2022) y de casi todo lo que vino después en generación de audio con LLMs.

**Descendencia directa.** **WavLM** (Chen et al., IEEE JSTSP 2022) es HuBERT más dos cambios: simulación de habla solapada y ruido durante el preentrenamiento —para servir a separación de hablantes y diarización, no solo ASR— y sesgo posicional relativo con compuertas en la atención; es hoy el punto de partida por defecto para tareas de habla que no son ASR. **data2vec** (Baevski et al., 2022) es el movimiento contrario: elimina las unidades discretas y predice representaciones **continuas y contextualizadas** de una red maestra actualizada por media móvil, unificando habla, visión y texto bajo el mismo objetivo — el sucesor conceptual que responde a la limitación del pipeline multietapa. **AV-HuBERT** extiende la receta a audio y video de labios; **mHuBERT** y la familia masivamente multilingüe de Meta la aplican a cientos de idiomas.

**Infraestructura.** HuBERT vive en las dos bibliotecas de referencia. En `transformers`: `HubertModel` y `HubertForCTC`, con los checkpoints `facebook/hubert-base-ls960`, `-large-ll60k` y `-xlarge-ll60k`. En `torchaudio.pipelines`: `HUBERT_BASE`, `HUBERT_LARGE`, `HUBERT_XLARGE` más los bundles `HUBERT_ASR_LARGE` y `HUBERT_ASR_XLARGE` — y que un método esté en `torchaudio.pipelines` es el indicador más duro de que dejó de ser investigación y pasó a ser infraestructura. El benchmark **SUPERB**, que evalúa encoders de habla congelados en una decena de tareas, consolidó la práctica de usar un encoder SSL congelado con cabezales livianos; HuBERT y WavLM ocupan la parte alta de esa tabla desde su aparición, y ahí se validó la promesa que el paper mismo no había probado.

**Y el otro camino.** [Whisper](/papers/whisper-radford-2022) (Radford et al., 2022) tomó la vía opuesta: supervisión débil masiva —680.000 horas de audio con transcripciones de calidad variable— sobre un Transformer encoder-decoder convencional, sin ningún SSL. Es un contrapunto honesto: **HuBERT es la respuesta cuando no hay etiquetas; Whisper es la respuesta cuando se pueden conseguir muchas etiquetas ruidosas.** Ambos son Transformers, ambos son anteriores al material de la [Clase 39](/clases/clase-39), y ambos son estado del arte en su nicho.

**Qué capa usar para qué.** Lo que establece el paper: capa **6** de 12 para el clustering a partir de un modelo de primera iteración; capa **9** a partir de un modelo de segunda iteración; y **nunca las últimas capas de un modelo entrenado con maestro malo**. Lo que viene de la práctica posterior, sin respaldo en el paper: tokenización para textless NLP con capa 6 y 50-200 clústeres; tareas de contenido (ASR, reconocimiento de fonemas, *keyword spotting*) con capas intermedias-altas; tareas de hablante (identificación, verificación, diarización) con **capas bajas** (1-4), porque la información paralingüística se descarta a medida que se sube, ya que el objetivo la castiga.

## Erratas y matices

**El "19% y 13%" del abstract sale de dos configuraciones distintas.** El abstract dice que X-LARGE muestra *"up to 19% and 13% relative WER improvement from LARGE models on dev-other and test-other"*. Verificado contra la tabla de resultados, **ninguna fila da ambas cifras a la vez**:

| Split de fine-tuning | dev-other LARGE → X-LARGE | Δ relativo | test-other LARGE → X-LARGE | Δ relativo |
|---|---|---|---|---|
| 10 min | 7.0 → 6.1 | −12.9% | 7.6 → 6.8 | −10.5% |
| 1 h | 4.9 → 4.2 | −14.3% | 5.4 → 4.8 | −11.1% |
| 10 h | 4.3 → 3.6 | −16.3% | 4.6 → 4.0 | **−13.0%** |
| 100 h | 3.7 → 3.0 | **−18.9%** | 3.9 → 3.5 | −10.3% |
| 960 h | 3.0 → 2.5 | −16.7% | 3.3 → 2.9 | −12.1% |

El "19%" es el split de **100 h** en dev-other; el "13%" es el split de **10 h** en test-other. Cada uno es el máximo de su columna, tomado de una fila distinta. Es un uso legítimo de "up to", pero citarlo como si describiera una configuración es incorrecto. Nótese además que esas cifras comparan **HuBERT LARGE contra HuBERT X-LARGE**, no HuBERT contra wav2vec 2.0.

**La afirmación sobre las "únicas excepciones" es falsa.** El paper sostiene que *"the superiority of HuBERT persists across setups with different amounts of labeled data, with the only exceptions being fine-tuning on 100 hours..."* y menciona dos casos de 0.1 puntos. Contra la tabla hay más excepciones y algunas mayores:

| Split | Columna | wav2vec 2.0 BASE | HuBERT BASE | Diferencia |
|---|---|---|---|---|
| 10 min | dev-clean | 8.9 | 9.1 | +0.2 |
| 10 min | test-clean | 9.1 | 9.7 | **+0.6** |
| 1 h | dev-clean | 5.0 | 5.6 | **+0.6** |
| 1 h | dev-other | 10.8 | 10.9 | +0.1 |
| 1 h | test-clean | 5.5 | 6.1 | **+0.6** |

Es decir, **HuBERT BASE es peor que wav2vec 2.0 BASE en 3 de las 4 columnas con 1 hora etiquetada** (y empata en la cuarta). Y con 960 horas etiquetadas, HuBERT LARGE (1.5/3.0/1.9/3.3) empata con wav2vec 2.0 LARGE (1.6/3.0/1.8/3.3) — mejor en dev-clean, peor en test-clean, igual en las otras dos.

{{< concept-alert type="advertencia" >}}
**La conclusión correcta, que el paper no enuncia:** con el mismo tamaño (BASE) y los mismos datos (LS-960), el objetivo de HuBERT **no es mejor** que el contrastivo de wav2vec 2.0; es equivalente o levemente peor en las condiciones limpias de muy bajos recursos. **HuBERT no gana a igual tamaño: gana al escalar.** La ventaja aparece en LARGE con 60.000 horas y se hace clara en X-LARGE. Esto no le quita valor al paper —escalar mejor es exactamente lo que importa—, pero cambia qué conclusión se puede extraer de las tablas.
{{< /concept-alert >}}

**El "4.6% de WER con 10 minutos" usa un LM Transformer.** La cifra de X-LARGE con 10 minutos etiquetados se obtiene decodificando con un LM Transformer entrenado sobre el corpus oficial de texto de LibriSpeech (unos 800 millones de palabras). Comparando la misma configuración sin LM neuronal —HuBERT LARGE, 10 min, 4-gram: 6.6/10.1 contra 4.7/7.6 con Transformer— **el LM aporta cerca de 2 puntos**. La supervisión **acústica** es de 10 minutos; la **lingüística** es enorme y viene de texto. Sigue siendo un resultado extraordinario, pero "10 minutos de supervisión total" es una exageración, y el paper no lo señala como caveat en la discusión de resultados.

**Hay tres valores distintos de PNMI para el mismo maestro.** Para el k-means sobre MFCC con $C=100$: la Tabla IV reporta 0.251-0.253, la Tabla V reporta **0.243** y el texto de la Sección V-C reporta **0.255**. Para BASE-it1-capa6 con $C=500$: la Tabla IV reporta 0.680-0.686 y la Tabla V reporta **0.637**, una diferencia de casi 0.05. **El paper no explica las discrepancias**; presumiblemente difieren el conjunto de ajuste del k-means y el conjunto de evaluación. Conviene no mezclar números de PNMI entre tablas.

**Otros matices menores.**

- **Conteos de parámetros inconsistentes.** La introducción dice "90M / 300M / 1B"; la tabla de arquitectura dice **95M / 317M / 964M**. La tabla es la fuente precisa.
- **La pérdida está escrita sin signo negativo.** El texto la introduce como *cross-entropy loss* pero escribe $L_m = \sum \log p_f$, que es una log-verosimilitud a maximizar. Es un descuido de notación; la implementación obviamente minimiza $-\sum \log p$.
- **$\alpha$ nunca se declara para los modelos principales.** Solo se infiere del abstract y de la tabla de ablación.
- **Las ablations se corren con $p = 6.5\%$, no con el $p = 8\%$ óptimo.** Sus conclusiones cualitativas sobre $\alpha$ y la calidad del maestro son sólidas, pero sus WER absolutos no son comparables con los de la tabla principal.
- **"HuBERT se entrena en 60.000 horas" es incorrecto para BASE.** BASE se preentrena sobre LS-960 en ambas iteraciones. Solo LARGE y X-LARGE usan Libri-Light. El checkpoint más usado en la práctica, `hubert-base-ls960`, vio 960 horas.
- **"HuBERT hace dos iteraciones" es incompleto.** BASE hace dos; LARGE y X-LARGE son efectivamente terceras iteraciones, como el propio paper reconoce.
- **"HuBERT es un BERT" es una analogía, no una identidad.** Se comparte el objetivo de predicción enmascarada sobre un vocabulario finito y el encoder Transformer bidireccional. No se comparte: la entrada no es discreta, no hay tokenizador, no hay `[CLS]`, no hay *next-sentence prediction*, la capa de salida usa similitud coseno con temperatura, y el vocabulario se descubre en vez de definirse.

## En la clase 39: la objeción que ya estaba resuelta

El PDF de la [Clase 39](/clases/clase-39), fechado en abril de 2024, sostiene que los Transformers "no son actualmente muy populares para aplicaciones de audio" y da tres razones. HuBERT, de junio de 2021, responde a las tres. Vale la pena evaluarlas una por una, con cuidado de reconocer lo que cada una tiene de cierto.

### Objeción 1: "faltan datasets de audio masivos"

**Lo que tiene de cierto.** Los corpus de habla **etiquetada** son y siguen siendo órdenes de magnitud más chicos que los de texto. [LibriSpeech](/papers/librispeech-panayotov-2015), el estándar de la comunidad durante una década, tiene 960 horas transcritas. El costo de transcribir es alto y crece con la dificultad del audio: la transcripción cuidadosa de habla espontánea toma varias veces el tiempo real, y para lenguas sin ortografía estandarizada el problema no es de costo sino de posibilidad. Comparado con los 750 GB de texto de C4 o los 3.300 millones de palabras del corpus de BERT, el habla etiquetada es un desierto. La introducción del propio paper dice exactamente eso: *"The time needed to collect large labeled datasets covering each of these scenarios is the real bottleneck in the current fast-moving AI industry."*

**Lo que pasa por alto.** El [SSL](/fundamentos/aprendizaje-autosupervisado) cambia cuál es el recurso escaso. La pregunta deja de ser "cuántas horas transcritas hay" y pasa a ser "cuántas horas de audio hay". Y de audio hay muchísimo: **Libri-Light son 60.000 horas de audiolibros de dominio público, sin una sola transcripción usada durante el preentrenamiento.**

Conviene dimensionarlo en las unidades en las que piensa un Transformer. A 50 tramas por segundo:

$$60{.}000 \text{ h} \times 3600 \frac{\text{s}}{\text{h}} \times 50 \frac{\text{tramas}}{\text{s}} = 1{,}08 \times 10^{10} \text{ tramas}$$

**10.800 millones de tramas por época.** El corpus de preentrenamiento de BERT (BooksCorpus + Wikipedia) tenía unos 3.300 millones de palabras. O sea: **60.000 horas de audiolibros producen más de tres veces los tokens del corpus original de BERT**, y esas 60.000 horas salen de un solo proyecto de voluntariado de dominio público (en bytes, unos 6,9 TB de PCM a 16 kHz y 16 bits).

La objeción confunde "no hay datasets masivos" con "no hay datasets masivos **etiquetados**". La primera afirmación es falsa; la segunda es cierta y es precisamente el problema que el SSL resuelve. El paper agrega un argumento adicional: la anotación además **descarta** información — *"labels, annotations, and text-only material ignores rich information in the input signal"*. La transcripción tira a la basura el timbre, la prosodia, la emoción y el ruido estructurado.

**Con cuántas etiquetas alcanza para un sistema usable.** Este es el remate: **HuBERT X-LARGE con 10 minutos de audio etiquetado da 4.6% de WER en test-clean y 6.8% en test-other.** Diez minutos. Un sistema con 4.6% de WER es perfectamente usable para dictado, subtitulado asistido o búsqueda por voz. Con 1 hora se llega a 2.8/4.8, que está en el rango de lo que un sistema supervisado con cientos de horas producía pocos años antes. (Recordando el matiz de la sección anterior: la supervisión acústica es de 10 minutos, la lingüística viene de un LM Transformer sobre texto.)

**Balance.** En el régimen supervisado la objeción tenía razón, y su núcleo verdadero sigue en pie: **el habla etiquetada es cara y escasa, y para la mayoría de las ~7.000 lenguas del mundo directamente no existe.** Ese hecho es el que motiva todo este paper. Lo que hay que corregir es la conclusión: la escasez de etiquetas no es un argumento contra los Transformers en audio, es exactamente el argumento a favor del preentrenamiento autosupervisado de Transformers en audio. Y para abril de 2024, la premisa fáctica ya no se sostenía ni siquiera para datos etiquetados: Whisper se entrenó con 680.000 horas con transcripción débil.

### Objeción 2: "la self-attention opera sobre entidades discretas y el audio no se segmenta trivialmente"

Esta es la que HuBERT responde con más precisión, porque **es literalmente el problema que el paper se plantea resolver**. Puestos lado a lado:

| Clase 39 (abril de 2024) | HuBERT (junio de 2021), abstract |
|---|---|
| "Self-attention mechanism operates over a finite sequence of discrete entities. In the context of text, sentence segmentation is trivial, but for audio this is not the case." | "(3) sound units have variable lengths with **no explicit segmentation**." |

Es la misma observación, formulada casi con las mismas palabras. La diferencia está en el rol: la clase la presenta como razón para no aplicar Transformers al audio; los autores del paper la toman como el problema de investigación que van a resolver. La respuesta de HuBERT tiene dos partes, y conviene separarlas porque la objeción mezcla dos cosas distintas.

**Parte A: la self-attention no necesita segmentación semántica, necesita una secuencia de vectores.** La [self-attention](/fundamentos/self-attention) opera sobre una secuencia de vectores de dimensión fija; no le importa qué representan esos vectores ni si sus fronteras coinciden con unidades lingüísticas. En NLP los tokens son subpalabras porque el texto viene así, pero el mecanismo no lo exige — de hecho, ViT parte imágenes en parches de $16\times16$ píxeles que **no** corresponden a ningún objeto ni a ninguna frontera semántica, y funciona perfectamente. Lo mismo hace el [AST](/papers/ast-gong-2021) sobre el espectrograma. La segmentación semántica nunca fue un requisito de la atención.

HuBERT hace exactamente lo mismo en el eje temporal: **el encoder convolucional de 7 capas parte el audio en tramas regulares de 20 ms con campo receptivo de 25 ms.** No hay ninguna pretensión de que esas fronteras coincidan con fonemas. Son parches temporales de tamaño fijo, igual que los parches espaciales de ViT. La afirmación de que "el audio no se segmenta trivialmente" es cierta para la segmentación fonética e irrelevante para alimentar una self-attention.

**Parte B: sí hay un lugar donde se necesitan entidades discretas —el objetivo— y HuBERT las fabrica.** El problema real, que la objeción roza sin nombrar, no está en la entrada de la atención sino en la **salida**: para escribir una pérdida de predicción enmascarada estilo BERT hace falta un vocabulario finito sobre el cual poner el softmax, y el habla no lo trae. Es el problema (2) del abstract: *"there is no lexicon of input sound units during the pre-training phase... hindering the use of predictive losses."*

La solución de HuBERT es **no segmentar y en cambio inventar las clases**. En vez de buscar dónde terminan los fonemas, se corre k-means sobre las features de cada trama y se declara que los $C$ centroides son el vocabulario. Cada trama recibe una etiqueta. **No hay segmentación explícita en ningún momento**: la segmentación emerge implícitamente como corridas de tramas con la misma etiqueta, y nunca hace falta materializarla. Y funciona con un maestro que se equivoca dos tercios de las veces, porque lo que importa es que la asignación sea consistente, no correcta.

**Balance.** La objeción identifica correctamente una dificultad real y específica del habla. Pero la trata como impedimento cuando ya era un problema resuelto —HuBERT es de junio de 2021, wav2vec 2.0 de junio de 2020, y ambos estaban en `torchaudio` cuando se escribió la diapositiva— y además ubica mal el problema: la self-attention nunca necesitó entidades semánticas en la entrada; era la pérdida predictiva la que necesitaba clases en la salida, y esas se fabrican con k-means.

### Objeción 3: "los Transformers no son buenos para dependencias largas"

Esta es la más problemática de las tres, porque invierte la motivación original del mecanismo.

**Lo que dice el paper fundacional.** [Vaswani et al. (2017)](/papers/attention-is-all-you-need-vaswani-2017) dedican una tabla completa a comparar tipos de capa por tres criterios: complejidad por capa, número de operaciones secuenciales y **longitud máxima de camino** entre dos posiciones cualesquiera.

| Tipo de capa | Complejidad por capa | Ops. secuenciales | Longitud máxima de camino |
|---|---|---|---|
| Self-attention | $O(n^2 \cdot d)$ | $O(1)$ | $\mathbf{O(1)}$ |
| Recurrente | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolucional | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k n)$ |
| Self-attention restringida (ventana $r$) | $O(r \cdot n \cdot d)$ | $O(1)$ | $O(n/r)$ |

Y el argumento explícito: *"Learning long-range dependencies is a key challenge in many sequence transduction tasks. One key factor affecting the ability to learn such dependencies is the length of the paths forward and backward signals have to traverse in the network. The shorter these paths [...] the easier it is to learn long-range dependencies."*

**La longitud máxima de camino de la self-attention es $O(1)$: constante, independiente de la distancia.** Cualquier par de posiciones se conecta en un solo paso de atención. En una RNN, la señal entre las posiciones 1 y 500 tiene que atravesar 500 multiplicaciones matriciales sucesivas, con el consiguiente desvanecimiento o explosión de gradiente — el problema que las LSTM mitigan sin eliminar. **Modelar dependencias largas es la razón por la que se inventó la self-attention, no su debilidad.**

**Cuál es la limitación real.** Es el **costo**, no la capacidad: $O(n^2)$ en tiempo y memoria. Con secuencias muy largas el mecanismo se vuelve computacionalmente inviable, y de ahí toda la literatura de atención eficiente (Longformer, Performer, atención por ventanas, FlashAttention). Pero "es caro para $n$ grande" y "no puede modelar dependencias largas" son afirmaciones distintas, y la segunda es falsa. Confundirlas es como decir que una lista enlazada "no puede indexar" cuando lo que ocurre es que indexar cuesta $O(n)$.

**Cómo lo resuelve HuBERT.** El costo cuadrático en audio es genuinamente amenazante: 10 segundos a 16 kHz son 160.000 muestras, y una matriz de atención de $160.000^2$ es absurda. La solución está en la arquitectura: **el encoder convolucional reduce 320×**, y esos diez segundos pasan de 160.000 muestras a **500 tramas**, tamaño perfectamente cómodo. La reducción de resolución no es un parche: es el frontend que hace del habla una secuencia de longitud manejable, exactamente como el parcheado de $16\times16$ hace de una imagen de $224\times224$ una secuencia de 196 tokens en ViT.

Y hay un dato cuantitativo que cierra el argumento de costo. Según la tabla de Vaswani, la self-attention es **más barata** que una capa recurrente cuando $n < d$. Para HuBERT BASE, $d = 768$; para LARGE, $d = 1024$. Un enunciado de 10 segundos son 500 tramas. **$500 < 768$: en el régimen operativo real de HuBERT, la self-attention es computacionalmente más barata que una LSTM equivalente, además de tener camino $O(1)$.**

**La evidencia del propio paper.** HuBERT no es neutral respecto de las dependencias largas: **su objetivo entero está diseñado para forzarlas.**

> *"to reduce the prediction error, the model needs to capture the long-range temporal relations between learned representations"*

y sobre $\alpha = 1$: *"It forces the model to learn both the acoustic representation of unmasked segments and the **long-range temporal structure** of the speech data."*

Concretamente, con spans de 200 ms enmascarados y ~57% de tramas borradas, el modelo tiene que inferir dos o tres fonemas completos desde contexto que puede estar a cientos de milisegundos. Y funciona: es el mecanismo que produce el 4.6% de WER con 10 minutos de etiquetas. La ablación de $\alpha$ lo prueba por contraste — cuando se le permite al modelo resolver la tarea localmente ($\alpha = 0$, viendo la trama que hay que predecir), el desempeño colapsa a 96%. **El modelo solo aprende algo útil cuando se le obliga a usar contexto largo.**

### Balance de las tres

| Objeción de la clase 39 | Estado en 2021 según el paper | Veredicto |
|---|---|---|
| 1. Faltan datasets de audio masivos | Cierto para audio **etiquetado**; el SSL cambia cuál es el recurso escaso. 60.000 h sin etiquetar producen 10.800 M de tramas, más que el corpus de BERT | Parcialmente cierta en su momento histórico, desactivada por el método |
| 2. La self-attention necesita entidades discretas y el audio no se segmenta | Es el problema (3) del abstract, enunciado por los propios autores. Se resuelve fabricando unidades con k-means, sin segmentar nunca | Correcta como diagnóstico, incorrecta como impedimento; resuelta desde 2020-2021 |
| 3. Los Transformers no modelan bien dependencias largas | Camino $O(1)$ contra $O(n)$ de una RNN; la limitación es el costo, resuelto con el downsampling de 320×; el objetivo de HuBERT exige dependencias largas | Incorrecta; invierte la motivación del mecanismo |

Lo justo con el material de la clase es reconocer que la objeción 1 tenía un núcleo verdadero. Lo que hay que ajustar es la conclusión que se extrae de las tres: para 2024 el audio ya no era territorio hostil para los Transformers, sino uno de los dominios donde el preentrenamiento autosupervisado había producido sus resultados más espectaculares. Ver también la [profundización de la Clase 39](/clases/clase-39/profundizacion) y el recorrido general del [dominio audio](/dominios/audio).

## Notas y enlaces

- **PDF del paper:** [`hubert-hsu-2021.pdf`](/papers/hubert-hsu-2021.pdf) (arXiv:2106.07447v1, 14 de junio de 2021; versión de revista en IEEE/ACM TASLP vol. 29, 2021, pp. 3451-3460).
- **Código y pesos:** `fairseq/examples/hubert`. Checkpoints en `transformers` (`facebook/hubert-base-ls960`, `-large-ll60k`, `-xlarge-ll60k`) y en `torchaudio.pipelines`.
- **Gotcha de implementación.** `hubert-base-ls960` tiene `do_normalize=False`, mientras que las variantes *large* y *xlarge* usan `do_normalize=True` (normalización de media y varianza por enunciado). Hay que usar siempre el *feature extractor* del checkpoint: fijar la normalización a mano degrada los embeddings en silencio. La entrada debe ser mono a 16 kHz — el modelo no remuestrea.
- **Indexación de capas.** `output_hidden_states=True` devuelve 13 tensores para BASE: el índice 0 es la entrada al primer bloque Transformer y los índices 1 a 12 son las salidas de las 12 capas. La indexación coincide con la del paper, así que las conclusiones sobre qué capa usar son directamente reutilizables.
- **Nota de ingeniería para producción.** El encoder convolucional se congela durante el fine-tuning, así que las 7 capas convolucionales pueden extraerse una sola vez, cachearse como tensores de 512 canales a 50 Hz y compartirse entre múltiples cabezales downstream. Para corpus grandes, la diferencia entre recomputar el frontend por tarea y cachearlo es sustancial, y aquí está avalada por el diseño del modelo.
- **Papers relacionados en el site:** [wav2vec 2.0](/papers/wav2vec2-baevski-2020) (el antecedente arquitectónico directo), [Whisper](/papers/whisper-radford-2022) (el camino opuesto: supervisión débil masiva), [Conformer](/papers/conformer-gulati-2020) (el competidor de escala de la época), [AST](/papers/ast-gong-2021) (el Transformer puro sobre espectrogramas), [LibriSpeech](/papers/librispeech-panayotov-2015) (el corpus de evaluación), [Attention Is All You Need](/papers/attention-is-all-you-need-vaswani-2017) (el criterio de longitud de camino).
- **Fundamentos:** [aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado), [self-attention](/fundamentos/self-attention), [preentrenamiento tipo BERT](/fundamentos/pretraining-bert), [MFCC y escala Mel](/fundamentos/mfcc-y-escala-mel).
