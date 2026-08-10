---
title: "CLDNN: Convolutional, Long Short-Term Memory, Fully Connected DNNs (2015)"
weight: 425
math: true
---

{{< paper-card
    title="Convolutional, Long Short-Term Memory, Fully Connected Deep Neural Networks"
    authors="Tara N. Sainath, Oriol Vinyals, Andrew Senior, Haşim Sak (Google)"
    year="2015"
    venue="ICASSP 2015"
    pdf="/papers/cldnn-sainath-2015.pdf" >}}
Cinco páginas, ninguna operación nueva, ni un solo conteo de parámetros reportado — y aun así este paper fijó el **orden canónico** con el que la comunidad de audio armó modelos durante los cinco años siguientes. La tesis es que las tres familias de redes que competían en 2014 no son alternativas sino **complementarias**: *"CNNs are good at reducing frequency variations, LSTMs are good at temporal modeling, and DNNs are appropriate for mapping features to a more separable space"*. De ahí el apilamiento **C → L → D**: dos capas convolucionales de 256 mapas sobre el espectrograma log-mel (con max-pooling **solo en frecuencia**), una capa lineal que reduce a 256, dos capas LSTM con proyección, y dos capas densas de 1024 antes del softmax — todo **entrenado conjuntamente**, no ensamblado a posteriori. El argumento de fondo no es "juntemos tres cosas": es que un LSTM es profundo *solo en el tiempo* (Pascanu et al.), y CLDNN le rellena sus dos transiciones superficiales — la CNN es la transición entrada→estado profunda, el DNN la de estado→salida. El orden se justifica además por herencia histórica: en GMM/HMM se normalizaba el hablante (VTLN, fMLLR) **antes** de modelar el tiempo, y la CNN es aquí una **VTLN aprendida**. Sobre tráfico real de Google Voice Search, la ganancia va de **3.9% a 7.4% relativo en WER** sobre el LSTM (la mejor cifra, 18.8 → 17.4, en habla ruidosa con entrenamiento de secuencia). El patrón sobrevivió al modelo: se lo rebautizó **CRNN** y colonizó la detección de eventos sonoros y el etiquetado de música, y su retórica reaparece literalmente en [Conformer (2020)](/papers/conformer-gulati-2020) — con self-attention en el lugar del LSTM. De CLDNN sobrevivió la "C", no la "L".
{{< /paper-card >}}

---

## Contexto: dos mejoras que no se hablaban entre sí

Para 2012 el reemplazo grande ya había ocurrido. El modelo acústico basado en DNN había desplazado al GMM en reconocimiento de vocabulario extenso, y la arquitectura de producción era estable y aburrida: un MLP de 6 capas de 1024 unidades que recibe un stack de features log-mel con contexto aplanado en un vector, y emite una posterior sobre miles de estados dependientes de contexto (*CD states*) de un HMM. En este paper esa configuración es literalmente el baseline: seis capas de 1024 unidades, con 20 frames de contexto pasado y 5 de contexto futuro.

Conviene fijar desde el principio un punto que se distorsiona todo el tiempo al citar este trabajo: **nada de esto es end-to-end**. La red es solo el reemplazo de la función de verosimilitud del GMM dentro de un sistema HMM completo, con árbol de decisión fonético, léxico y modelo de lenguaje n-grama. Los 13 522 objetivos de salida del paper son hojas de ese árbol de decisión.

Sobre ese baseline aparecieron, a partir de 2013, dos líneas de mejora con argumentos independientes y sin contacto entre sí.

**La rama CNN.** Sainath et al., *Deep Convolutional Neural Networks for LVCSR* (ICASSP 2013), y el trabajo paralelo de Abdel-Hamid et al. El argumento: el espectrograma log-mel es una imagen con dos ejes de significado físico distinto, y aplanarlo en un vector para meterlo en un MLP tira esa estructura a la basura. Compartir pesos a lo largo del eje de frecuencia, y hacer pooling en ese eje, absorbe una fuente concreta de variabilidad: **el desplazamiento de las formantes entre hablantes**. En las tablas de este paper el baseline CNN da 18.0 de WER contra 18.4 del DNN.

**La rama LSTM.** Graves et al. en TIMIT y, decisivamente para este trabajo, Sak, Senior y Beaufays, *Long Short-Term Memory Recurrent Neural Network Architectures for Large Scale Acoustic Modeling* (Interspeech 2014) — dos de cuyos autores firman también aquí. El argumento es ortogonal al anterior: el contexto que un DNN puede ver está acotado por el tamaño de la ventana que se le aplana en la entrada, y agrandarla cuesta parámetros linealmente y deja de ayudar pasado cierto punto. Un [LSTM](/fundamentos/lstm-gru) tiene memoria de longitud no acotada por construcción y consume **un solo frame a la vez**. La contribución de ingeniería de Sak et al. fue la **LSTMP** — LSTM con capa de proyección recurrente — que permite muchas celdas sin que la matriz recurrente explote, y entrenarla con SGD asíncrono distribuido a escala de miles de horas.

### Por qué nadie las había combinado

La pregunta correcta, y la respuesta no es "no se les ocurrió". Hay razones concretas.

**Los contratos de entrada eran incompatibles.** La CNN de 2013 consumía una ventana grande y simétrica y emitía **una** decisión. El LSTM de 2014 consumía **un frame**, y su contexto venía de la recurrencia. Concatenarlos exige decidir qué hace la CNN con el eje temporal, y esa decisión no es obvia: si la CNN ya consume 25 frames y el LSTM se desenrolla 20 pasos, el contexto efectivo se multiplica y deja de estar claro qué modela quién. El paper lo mide y encuentra el límite: con 20 frames de contexto izquierdo el WER **empeora**, *"likely since the LSTM is then unrolled for 20 time steps, so the total context processed by the LSTM is 40"*.

**La forma canónica de combinar modelos era el ensamble, no la arquitectura.** El paper cita explícitamente a Deng y Platt (Interspeech 2014), donde *"the three models were first trained separately and then the three outputs were combined through a combination layer"*. Esa era la práctica: entrenar CNN, DNN y RNN por separado y combinar posteriores o lattices. Es más fácil de operar, se paraleliza trivialmente y no exige que ninguna pieza cambie. La contribución de CLDNN se define por contraste con eso: combinar los tres en un marco único **entrenado conjuntamente**.

**Entrenar LSTMs a escala era frágil.** El propio paper aporta la evidencia: con inicialización gaussiana de varianza $1/n_{\text{in}}$ el LSTM daba 18.0, y con inicialización uniforme en $[-0.02, 0.02]$ daba 17.7 — una mejora de 1.7% relativo por cambiar *solo* la inicialización, más grande que la ganancia entera de la CNN sobre el DNN. Apilar capas convolucionales debajo de algo tan sensible al condicionamiento era una apuesta razonable de evitar.

**Y no había un argumento que dijera en qué orden ponerlas.** Ahí está la contribución conceptual real.

## La tesis de la complementariedad

### El argumento no es "apilemos tres cosas"

La lectura superficial es que la CNN, la RNN y el MLP hacen cosas distintas y por lo tanto conviene tenerlas todas. El paper tiene un argumento bastante más específico, y viene de una sola referencia que cita cuatro veces: **Pascanu, Gulcehre, Cho y Bengio, *How to Construct Deep Recurrent Neural Networks* (ICLR 2014)**.

La observación de ese trabajo es que una RNN tiene **tres transiciones separables**, y "profundidad" significa algo distinto en cada una:

- entrada → estado oculto, $x_t \to h_t$;
- estado oculto → estado oculto, $h_{t-1} \to h_t$;
- estado oculto → salida, $h_t \to y_t$.

Un LSTM estándar es profundo **solo en el tiempo**: las otras dos transiciones son afines de una sola capa. Sainath et al. lo citan dos veces, casi textualmente: *"One issue with LSTMs is that the temporal modeling is done on the input feature $x_t$ (i.e., log-mel feature). However, higher-level modeling of $x_t$ can help to disentangle underlying factors of variation within the input"*, y *"in LSTMs the mapping between $h_t$ and output $y_t$ is also not deep, meaning there is no intermediate nonlinear hidden layer"*.

{{< concept-alert type="clave" >}}
Bajo esa lectura, la CLDNN no es "CNN + LSTM + DNN". Es **un LSTM al que se le rellenaron sus dos transiciones superficiales**: la CNN es la transición entrada→estado profunda, el DNN es la transición estado→salida profunda. **El LSTM es el modelo; la CNN y el DNN son adaptadores de sus interfaces.** Esa reformulación explica de una vez todas las decisiones del paper: por qué ese orden y no otro, por qué dos capas de cada cosa y no seis, y por qué la ganancia satura tan rápido — una vez que la transición dejó de ser afín, agregar más capas rinde poco.
{{< /concept-alert >}}

### El segundo argumento: la receta ya existía en GMM/HMM

El paper da una segunda justificación del orden, distinta y complementaria, y es históricamente la más elegante:

> *"state-of-the-art GMM/HMM systems perform speaker adaptation, using techniques such as vocal tract length normalization (VTLN) and feature-space maximum likelihood linear regression (fMLLR), **before** performing temporal modeling via HMMs"*.

O sea: **normalizar primero, modelar el tiempo después** ya era el orden canónico de veinte años de ASR estadístico. VTLN estima un factor de warping $\alpha$ por hablante y reescala el eje de frecuencia $f \to \alpha f$ para compensar diferencias de longitud del tracto vocal; fMLLR aplica una transformación afín en el espacio de features. Ambas son normalizaciones que se aplican **antes** del HMM. CLDNN sustituye esa normalización explícita, estimada por EM, por una **normalización aprendida y discriminativa**: las capas convolucionales. El paper refuerza la analogía citando a Mohamed, Hinton y Penn: *"it has been shown that CNNs learn speaker-adapted/discriminatively trained features"*.

Esta es la clave que hace que CLDNN se sienta inevitable en retrospectiva: **no inventó una jerarquía, recapituló la que ya funcionaba, con módulos entrenables.**

### Un matiz de vocabulario que no es cosmético: "varianza", no "invarianza"

Es habitual justificar el bloque convolucional diciendo que aporta *invarianza a traslación*. Esa es la formulación del canon de visión por computador, y no es la del paper. La palabra `invariance` —y cualquier variante— aparece **cero veces** en el texto completo. El vocabulario del paper es otro: `variance` (cuatro veces), `variation`/`variations` (ocho veces), en frases como *"reduce frequency variance"*, *"reduce spectral variation"*, *"remove variation in the input"*.

¿Es la misma afirmación con otras palabras? Casi, y la diferencia importa por tres razones.

- **Compartir pesos produce equivarianza, no invarianza.** El paper dice que los filtros *"are shared across the entire time-frequency space"*. Si la entrada se desplaza $k$ bins en frecuencia, la salida de la convolución se desplaza $k$ bins: eso es **equivarianza**. La invarianza local aparece únicamente donde hay pooling, y en este paper el pooling está en un solo lugar: max-pooling no solapado de tamaño 3, **solo en frecuencia**, y **solo tras la primera capa**.
- **"Reducir la variación" es un objetivo; "invarianza a traslación" es un mecanismo.** El paper apunta al objetivo y lo ancla en el argumento VTLN/fMLLR: quiere una representación estable frente a cambios de **hablante**, no frente a traslaciones arbitrarias. La versión precisa de la tesis es *la CNN es un VTLN aprendido*.
- **Y solo coinciden aproximadamente.** Es lo que desarrolla la subsección siguiente.

### En qué eje se aplica, y por qué

Este es el detalle que más se malinterpreta al pasar de visión a audio, y el paper lo resuelve en una frase fácil de leer sin notar lo que implica: ***"pooling in frequency only is performed"***.

**Por qué en frecuencia sí.** El tracto vocal de un adulto mide típicamente entre 13 y 18 cm. La teoría fuente-filtro dice que las frecuencias de resonancia —las formantes— escalan aproximadamente de forma inversamente proporcional a esa longitud: una persona con tracto más corto tiene todas sus formantes desplazadas hacia arriba por un factor multiplicativo $\alpha$ casi constante. Es exactamente lo que VTLN modela con $f \to \alpha f$.

Ahora bien, un desplazamiento **multiplicativo** en Hz solo se convierte en un desplazamiento **aditivo** —es decir, en una traslación, que es lo que una CNN puede absorber— si el eje es logarítmico. La [escala mel](/fundamentos/mfcc-y-escala-mel) es

$$m(f) = 2595 \, \log_{10}\!\left(1 + \frac{f}{700}\right),$$

aproximadamente **lineal** para $f \ll 700$ Hz y aproximadamente **logarítmica** para $f \gg 700$ Hz. Por lo tanto

$$m(\alpha f) - m(f) \;\approx\; 2595 \log_{10} \alpha \qquad \text{solo cuando } f \gg 700\ \text{Hz}.$$

Con 40 bins mel sobre un rango típico de 125–7500 Hz, el eje completo abarca unos 2588 mels, o sea unos **64.7 mels por bin**. Eso permite traducir un warping de tracto vocal a bins:

| Factor de warping $\alpha$ | $\Delta m$ (mels) | Desplazamiento en bins mel |
|---|---|---|
| 1.05 | 55 | 0.85 |
| 1.10 | 107 | **1.66** |
| 1.20 | 206 | **3.18** |

Una diferencia de tracto vocal del 10% —del orden de la que separa a un hombre adulto de una mujer adulta— produce un desplazamiento de **menos de dos bins mel** en la región logarítmica. Un pooling de tamaño 3 en frecuencia cubre exactamente ese rango. El "3" del paper no es un número redondo cualquiera: es del orden de magnitud correcto para la física del problema. (El paper no hace esta derivación; se limita a citar un trabajo previo del mismo grupo donde la estrategia de pooling se estudió empíricamente. Pero el número cierra.)

Dos matices que conviene no perder. La invarianza es **aproximada** y se degrada por debajo de ~700 Hz, justo donde vive F1: en la región lineal de la escala mel el warping VTLN no es una traslación sino un escalamiento, y ninguna cantidad de max-pooling lo absorbe. Y con un solo pooling de tamaño 3 en una red de dos capas convolucionales, la invarianza total es modesta: esto no es un pipeline profundo de invarianza al estilo de una CNN de ImageNet con cinco etapas de pooling, es una normalización superficial y localizada.

**Por qué en tiempo no.** Hay tres razones, y son acumulativas.

1. **La tarea es clasificación por frame.** El modelo acústico híbrido debe emitir $P(s \mid x)$ sobre los 13 522 estados CD **para cada frame de 10 ms**, porque esa posterior alimenta la búsqueda de Viterbi del HMM. Hacer pooling con stride en tiempo diezmaría la tasa de frames y rompería el contrato con el decodificador.
2. **La información discriminativa está en la transición, no en el estado.** La diferencia entre /b/ y /d/ está esencialmente en la *dirección de la transición de F2* en los primeros ~30 ms tras la explosión; la diferencia entre una oclusiva y una fricativa está en la abruptez del ataque. Un max-pooling temporal de tamaño 3 opera sobre tres frames de 10 ms, que es precisamente la escala de esos eventos. Se estaría destruyendo la señal que se quiere clasificar.
3. **El tiempo ya tiene dueño en esta arquitectura.** Toda la tesis del paper es que el LSTM se ocupa del eje temporal. Tirar resolución temporal antes de que el LSTM la vea es sabotear la etapa siguiente.

Un cuarto matiz que suele confundirse: **la CNN sí convoluciona sobre el tiempo**. Los filtros son "frequency-time" de $9\times9$ y $4\times3$; el segundo tiene extensión 3 en el eje temporal. Convolucionar sobre el tiempo da **equivarianza** temporal —deseable: comparte pesos y modela dinámica local—. Lo que no hay es **pooling** temporal, que es lo que daría invarianza.

| Eje | Compartición de pesos (equivarianza) | Pooling (invarianza local) | Consecuencia |
|---|---|---|---|
| Frecuencia | Sí | **Sí**, tamaño 3, no solapado, capa 1 | Absorbe el warping tipo VTLN |
| Tiempo | Sí | **No** | Preserva la resolución de 10 ms para el LSTM y el HMM |

{{< concept-alert type="advertencia" >}}
Convolución sobre un eje $\neq$ invarianza sobre ese eje. CLDNN es un caso limpio donde la distinción tiene consecuencias: la red convoluciona en ambos ejes, pero solo hace pooling en frecuencia. En audio, la invarianza a traslación temporal sería un **error de diseño**, no una virtud heredada de visión.
{{< /concept-alert >}}

## La arquitectura, capa por capa

### Entrada

Cada frame $x_t$ es un vector **log-mel de 40 dimensiones**, calculado cada **10 ms**. El paper **no especifica** la longitud de la ventana de análisis ni el rango de frecuencias del banco de filtros; solo el hop y la dimensión.

La entrada a la red es $[x_{t-l}, \dots, x_{t+r}]$, con $l$ frames de contexto a la izquierda y $r$ a la derecha. Para la CLDNN se fija **$r = 0$** por una razón explícita de producción: *"In order to ensure that the LSTM does not see more than 5 frames of future context, which would increase the decoding latency, we set $r = 0$ for CLDNNs."* Esos "5 frames de futuro" vienen de otra parte: la etiqueta de salida se **retrasa 5 frames**, el truco estándar de Sak et al. para dar algo de lookahead a un LSTM unidireccional. Sumado, el sistema tiene 50 ms de lookahead y ni un milisegundo más — requisito duro para Voice Search.

El mejor contexto izquierdo resulta ser $l = 10$ (17.6 de WER, contra 17.8 con $l=0$ y 17.9 con $l=20$). Así que la entrada a la CNN en la mejor configuración es una "imagen" de $40 \times 11$: 40 bins de frecuencia por 11 frames de tiempo.

### Bloque convolucional

Dos capas, cada una con **256 mapas de features**. Filtro **$9\times9$** frecuencia-tiempo en la primera; filtro **$4\times3$** en la segunda —$4\times3$, no $4\times4$; el texto es literal: *"followed by a 4x3 filter for the second convolutional layer"*—. Los filtros se comparten *"across the entire time-frequency space"*, o sea **compartición completa de pesos** en ambos ejes, no la *limited weight sharing* de la literatura de [CNN](/fundamentos/redes-convolucionales) para ASR de 2012. Y max-pooling **no solapado, solo en frecuencia, tamaño 3, solo tras la primera capa**: *"A pooling size of 3 was used for the first layer, and no pooling was done in the second layer"*.

El paper **no especifica el padding**, y eso tiene consecuencias: no se pueden reconstruir con certeza las formas de los tensores. Peor aún, reporta una configuración con $l = 0$ —o sea, una entrada de un solo frame en el eje temporal— a la que se aplica un filtro de extensión temporal 9. Eso exige padding masivo o alguna convención no documentada. Es una ambigüedad genuina del paper.

### La capa lineal de reducción

Es la pieza que más se malinterpreta, y la que el paper justifica con más claridad:

> *"The dimension of the last layer of the CNN is large, due to the number of **feature-maps × time × frequency context**. Thus, we add a linear layer to reduce feature dimension, before passing this to the LSTM layer... we found that reducing the dimensionality, such that we have **256 outputs** from the linear layer, was appropriate."*

Dos cosas críticas. Primero, **es una capa lineal sobre el tensor aplanado completo, no una convolución $1\times1$**: la dimensión grande viene del producto *mapas × tiempo × frecuencia*, y la capa lineal colapsa ese producto a 256. Una convolución $1\times1$ reduciría solo el eje de canales y dejaría intactos los ejes de frecuencia y tiempo. Segundo, **no hay no-linealidad**: el paper dice *"linear layer"* y la figura la etiqueta `linear layer / dim red`. Es una factorización de rango bajo pura — exactamente el mismo objeto que la capa de proyección de la LSTMP, aplicada en otro lugar de la red.

Cuánto ahorra, asumiendo convoluciones válidas (sin padding) sobre la entrada de $40 \times 11$:

| Etapa | Forma de salida | Cómputo |
|---|---|---|
| Entrada | $1 \times 40 \times 11$ | — |
| Conv1 $9\times9$, 256 mapas | $256 \times 32 \times 3$ | $40-9+1=32$; $11-9+1=3$ |
| MaxPool 3 en frecuencia | $256 \times 10 \times 3$ | $\lfloor 32/3 \rfloor = 10$ |
| Conv2 $4\times3$, 256 mapas | $256 \times 7 \times 1$ | $10-4+1=7$; $3-3+1=1$ |
| Aplanado | **1792** | $256 \times 7 \times 1$ |
| Capa lineal | **256** | matriz $1792 \times 256$ |

Sin la capa lineal, la matriz de entrada del primer LSTM (832 celdas, cuatro compuertas) sería $4 \times 832 \times 1792 = 5\,963\,776$ parámetros. Con ella, se paga la suma de la lineal y la nueva matriz de entrada:

$$\underbrace{1792 \times 256}_{458\,752} \;+\; \underbrace{4 \times 832 \times 256}_{851\,968} \;=\; 1\,310\,720.$$

**Ahorro: ~4.65M de parámetros**, cerca del 20% del modelo completo. La afirmación del paper de *"reduction in parameters with no loss in accuracy"* es, cuantitativamente, sustancial. (Estas formas y estos conteos son una reconstrucción: dependen del padding que el paper no documenta, y el paper no reporta parámetros en ningún lugar.)

### Bloque LSTM y bloque DNN

El bloque recurrente sigue la receta de Sak et al. 2014: **2 capas LSTM**, **832 celdas** por capa, **capa de proyección de 512 unidades** (arquitectura LSTMP), desenrollado de **20 pasos** para BPTT truncado. El paper señala, citando el mismo trabajo, que *"adding extra LSTM layers to this configuration was not found to help"*.

La proyección merece una nota. En una LSTMP la salida recurrente que se realimenta no es $h_t \in \mathbb{R}^{832}$ sino $r_t = W_{\text{proj}} h_t \in \mathbb{R}^{512}$. Eso rompe el escalamiento cuadrático de la matriz recurrente: en vez de $4 \times 832 \times 832$ se paga $4 \times 832 \times 512 + 832 \times 512$. **CLDNN usa dos factorizaciones de rango bajo, por la misma razón, en dos lugares distintos**: la capa lineal en el eje de entrada, la proyección en el eje recurrente.

El bloque denso son **capas totalmente conectadas de 1024 unidades**, **dos** en la configuración final porque el rendimiento satura ahí, y un softmax sobre **13 522 objetivos CD**.

### La tabla completa

Reconstrucción para la configuración final ($l=10$, $r=0$). Las columnas de forma y de parámetros son derivadas con las asunciones de padding recién descritas.

| # | Capa | Hiperparámetros (del paper) | Forma de salida | Parámetros |
|---|---|---|---|---|
| 0 | Entrada | log-mel 40D, hop 10 ms, $l=10$, $r=0$ | $1 \times 40 \times 11$ | — |
| 1 | Conv 2D | 256 mapas, filtro $9\times9$ (freq × tiempo) | $256 \times 32 \times 3$ | 20 992 |
| 2 | MaxPool | tamaño 3, **solo frecuencia**, no solapado | $256 \times 10 \times 3$ | 0 |
| 3 | Conv 2D | 256 mapas, filtro $4\times3$ | $256 \times 7 \times 1$ | 786 688 |
| 4 | Lineal (dim red) | 256 salidas, **sin no-linealidad** | $256$ | 458 752 |
| 5 | LSTMP #1 | 832 celdas, proyección 512 | $512$ | ~2.99M |
| 6 | LSTMP #2 | 832 celdas, proyección 512 | $512$ | ~3.84M |
| 7 | FC #1 | 1024 unidades | $1024$ | 525 312 |
| 8 | FC #2 | 1024 unidades | $1024$ | 1 049 600 |
| 9 | Softmax | 13 522 objetivos CD | $13522$ | ~13.86M |
| | **Total** | | | **~23.5M** |

Dos observaciones que salen solo de hacer la cuenta. **La capa de salida es el 59% del modelo**: 13 522 estados CD × 1024 unidades ≈ 13.85M parámetros, más que todo el resto junto. Cuando se dice que CLDNN "es una red grande", la mayor parte de ese tamaño no es CLDNN: es el árbol de decisión fonético del sistema HMM. Y **el bloque convolucional es barato en parámetros (~1.27M) pero no en cómputo**: la CNN se evalúa una vez por frame de salida sobre una ventana de 11 frames, así que en una implementación ingenua cada frame de entrada se reprocesa hasta once veces.

## Las conexiones de salto: una funciona, la otra no

El paper introduce lo que llama *"multi-scale additions"*: dos flujos adicionales, dibujados con línea punteada en su Figura 1. La motivación declarada viene de visión — *"Each CNN, LSTM and DNN block captures information about the input representation at different scales"*, citando a Sermanet y LeCun (IJCNN 2011).

**Flujo 1: el frame crudo $x_t$ concatenado a la salida de la CNN, antes del LSTM.** La CNN consume $[x_{t-10}, \dots, x_t]$ y emite un vector de 256 dimensiones que resume esa ventana larga; al LSTM se le pasa la concatenación de ese vector con el frame crudo de 40 dimensiones, o sea 296. El argumento es de escalas: hay información en el frame actual que la CNN puede haber promediado o normalizado, y el trabajo original de LSTM consumía exactamente eso, features de corto plazo sin contexto. **Aporta 17.0 → 16.8, un 1.2% relativo.** El costo es de +133 120 parámetros en la matriz de entrada del LSTM, un 0.57% del modelo: la afirmación del paper de que el aumento es *"negligible"* se sostiene.

**Flujo 2: la salida de la CNN concatenada a la del LSTM, antes del DNN.** El bloque denso recibiría 768 dimensiones (512 del LSTM + 256 de la CNN). La motivación: *"we explore if there is complementarity between modeling the output of the CNN temporally with an LSTM, as well as discriminatively with a DNN"*. **Aporta 17.0 → 17.0. Nada.** El paper es directo:

> *"Table 7 indicates that this does not yield gains over the CLDNN alone. This indicates that temporal processing of CNN features using the LSTM is sufficient, and more information is not gained by additionally passing CNN features into the DNN."*

Y en consecuencia lo **abandona** para todos los experimentos a gran escala: *"we just include results passing short and long-term features into the CNN, and omit passing the CNN into both the LSTM and DNN, as only the first technique showed gains"*.

{{< concept-alert type="advertencia" >}}
De las dos conexiones de salto del paper, **una funciona marginalmente y la otra no funciona en absoluto**, y el propio paper la descarta. Cualquier descripción de CLDNN que presente ambas como parte de la arquitectura final está describiendo la Figura 1 —que dibuja las dos, en punteado— y no los resultados.
{{< /concept-alert >}}

### ¿Es esto un antecedente de las conexiones residuales?

Cronológicamente, el preprint está compilado en octubre de 2014 y el paper es de ICASSP 2015; *Highway Networks* es de mayo de 2015 y *Deep Residual Learning* de diciembre de 2015. **CLDNN precede a ambos.** Pero el mecanismo es distinto, y la diferencia no es cosmética.

| | CLDNN (2015) | ResNet (2015) | DenseNet (2016) |
|---|---|---|---|
| Operación | **Concatenación** | **Suma** | **Concatenación** |
| Dimensiones | Pueden diferir libremente | Deben coincidir o proyectarse | Pueden diferir |
| Efecto en parámetros | Crece la capa receptora | Cero | Crece la capa receptora |
| Motivación declarada | **Multi-escala** (fusión de información) | **Optimización** (flujo de gradiente, degradación) | Reutilización de features |
| Profundidad involucrada | 2 saltos en una red de 9 capas | Cada bloque, en redes de 50-152 capas | Todos contra todos por bloque |

Por mecanismo, CLDNN es antecedente de **DenseNet**, no de ResNet: concatena, no suma. Y por motivación no es antecedente de ninguno de los dos — el paper habla de escalas de representación y nunca de gradientes, de degradación con la profundidad ni de facilitar la optimización. Lo correcto es decir que CLDNN pertenece al momento en que la comunidad estaba descubriendo, por caminos independientes, que **una capa no tiene por qué recibir su entrada solo de la capa inmediatamente anterior**. Hay una ironía útil: la conexión que CLDNN mide y **descarta** es la que topológicamente más se parece a un salto residual de largo alcance.

## Resultados

Los datos son tráfico real de Google, anonimizado y transcrito a mano. Tres conjuntos: uno mediano limpio (300k enunciados, ~200 h), uno grande limpio (3M enunciados, 2000 h) y uno grande ruidoso (los mismos 3M corrompidos artificialmente con un simulador de sala, con SNR entre 5 y 30 dB y ruidos tomados de YouTube y de grabaciones ambientales). El test son 30 000 enunciados (20 h). La pérdida es entropía cruzada por frame contra un alineamiento forzado; los experimentos a gran escala añaden una segunda fase de **entrenamiento de secuencia** sobre lattices. La optimización es SGD asíncrono distribuido sobre DistBelief.

El paper advierte explícitamente que *"the training and test sets used in this paper are different than those in [Sak et al. 2014], and therefore numbers cannot directly be compared"*.

### Conjunto de 200 horas (solo entropía cruzada)

**Baselines individuales.** Los tres modelos aislados, con inicialización gaussiana en el LSTM:

| Método | WER |
|---|---|
| DNN | 18.4 |
| CNN | 18.0 |
| LSTM | 18.0 |

**CNN+LSTM contra DNN+LSTM, variando el contexto izquierdo.** Este experimento hace dos cosas a la vez: encuentra el óptimo de contexto y verifica que la ventaja de la convolución no queda absorbida por la recurrencia.

| Contexto | Pasos de desenrollado | WER con CNN | WER con DNN |
|---|---|---|---|
| $l=0$, $r=0$ | 20 | 17.8 | 18.2 |
| $l=10$, $r=0$ | 20 | **17.6** | 18.2 |
| $l=20$, $r=0$ | 20 | 17.9 | 18.5 |

El óptimo está en 10 y crecerlo a 20 empeora. Y la CNN gana al DNN en las tres filas: *"the benefits of CNNs over DNNs continue to hold even when combined with LSTMs"*.

**El control que descarta la explicación aburrida.** Es el experimento metodológicamente más importante del paper y es fácil pasarlo por alto:

| Método | WER |
|---|---|
| LSTM, $l=0$, unroll = 20 | 18.0 |
| LSTM, $l=10$, unroll = 20 | 18.0 |
| LSTM, $l=0$, unroll = 30 | 18.2 |

Dar al LSTM exactamente el mismo contexto $l=10$ que recibe la CNN **no mueve el WER en absoluto**, y darle más recurrencia lo empeora. La CLDNN no gana porque le entreguen más frames: **la ganancia viene del procesamiento convolucional de esos frames, no de su disponibilidad.**

**Cuántas capas densas después del LSTM.** Satura en dos: 0 → 18.0, 1 → 17.8, 2 → **17.6**, 3 → 17.6.

**La composición completa.**

| Método | WER | Delta absoluto vs LSTM |
|---|---|---|
| LSTM | 18.0 | — |
| CNN+LSTM | 17.6 | −0.4 |
| LSTM+DNN | 17.6 | −0.4 |
| **CLDNN** | **17.3** | **−0.7** |

Si las dos contribuciones fueran perfectamente aditivas se esperaría $18.0 - 0.4 - 0.4 = 17.2$; se observa 17.3. Las ganancias son **~87% aditivas**: se solapan, pero muy poco. Ese es el contenido empírico de la palabra "complementary" en este paper, y es un resultado más fuerte de lo que suele reconocerse.

**El control anti-escepticismo: ¿y si el LSTM solo estuviera mal inicializado?**

| Método | WER, init gaussiana | WER, init uniforme |
|---|---|---|
| LSTM | 18.0 | 17.7 |
| CLDNN | 17.3 | 17.0 |

La brecha se preserva casi exactamente: 0.7 puntos absolutos y ~4% relativo en ambos regímenes. La ganancia no es un artefacto de un baseline mal condicionado.

**Adiciones multi-escala** (ya sobre inicialización uniforme):

| Método | WER |
|---|---|
| LSTM | 17.7 |
| CLDNN (feature de largo plazo al LSTM) | 17.0 |
| + feature de corto plazo al LSTM | **16.8** |
| + CNN al LSTM y al DNN | 17.0 |

### Conjunto de 2000 horas (con y sin entrenamiento de secuencia)

Aquí ya se descarta la conexión de salto que no funcionó, y aparece la distinción que más se pierde al citar el paper: **las cifras con entrenamiento de secuencia no son comparables con las de entropía cruzada**, porque el entrenamiento de secuencia mejora *todos* los modelos, incluido el baseline.

**2000 h limpio, test limpio:**

| Método | WER (entropía cruzada) | WER (tras entrenamiento de secuencia) |
|---|---|---|
| LSTM | 14.6 | 13.7 |
| CLDNN | 14.0 | 13.1 |
| multi-escala CLDNN | **13.8** | **13.1** |

**2000 h ruidoso, test ruidoso:**

| Método | WER (entropía cruzada) | WER (tras entrenamiento de secuencia) |
|---|---|---|
| LSTM | 20.3 | 18.8 |
| CLDNN | 19.4 | **17.4** |
| multi-escala CLDNN | **19.2** | **17.4** |

### Qué es grande y qué es marginal

Los relativos son cálculo sobre los WER del paper.

| Comparación | Absoluto | Relativo | Veredicto |
|---|---|---|---|
| LSTM → CLDNN, 2000 h ruidoso, **secuencia** | −1.4 | **7.4%** | **La mejor cifra del paper**, y en la condición más difícil |
| LSTM → CLDNN, 2000 h ruidoso, CE | −0.9 | 4.4% | Sólido |
| LSTM → CLDNN, 2000 h limpio, CE | −0.6 | 4.1% | Sólido |
| LSTM → CLDNN, 200 h, cualquier init | −0.7 | ~4% | Sólido y robusto a la inicialización |
| CLDNN → multi-escala, 200 h | −0.2 | 1.2% | Marginal |
| CLDNN → multi-escala, 2000 h limpio, CE | −0.2 | 1.4% | Marginal |
| CLDNN → multi-escala, 2000 h ruidoso, CE | −0.2 | 1.0% | Marginal |
| CLDNN → multi-escala, **tras secuencia**, ambos conjuntos | 0.0 | **0%** | **Desaparece** (13.1 vs 13.1; 17.4 vs 17.4) |

La última fila es la más interesante y el paper no la comenta: **el aporte multi-escala se evapora por completo tras el entrenamiento de secuencia**, en las dos condiciones. La lectura razonable es que el entrenamiento de secuencia recupera por su cuenta la información que la conexión de salto de corto plazo estaba aportando. Simétricamente, el aporte del **núcleo** CLDNN hace lo contrario: crece con el entrenamiento de secuencia en la condición ruidosa, de 4.4% a 7.4%. Esa asimetría es la mejor evidencia disponible de que la contribución central es real y la periférica es ruido de bajo nivel.

## Ablations: lo que el paper mide y lo que no

### Lo medido

| Pregunta | Resultado | Conclusión soportada |
|---|---|---|
| ¿La CNN antes del LSTM ayuda? | 17.6 vs 18.0 | **Sí**, 2.2% relativo |
| ¿El DNN después del LSTM ayuda? | 17.6 vs 18.0 | **Sí**, 2.2% relativo |
| ¿Se suman? | CLDNN 17.3 | **Sí**, ~87% aditivo |
| ¿CNN es mejor que DNN en esa posición? | 17.8/17.6/17.9 vs 18.2/18.2/18.5 | **Sí**, en las tres filas |
| ¿Cuánto contexto izquierdo? | óptimo en $l=10$ | Existe un óptimo interior |
| ¿Es solo contexto extra? | LSTM con $l=10$ da 18.0, igual que con $l=0$ | **No.** Es el procesamiento |
| ¿Es solo mala inicialización del LSTM? | brecha idéntica en ambos regímenes | **No** |
| ¿Cuántas capas DNN? | satura en 2 | 2 |
| ¿Skip de corto plazo al LSTM? | 17.0 → 16.8 | **Sí**, ~1%, y se anula tras secuencia |
| ¿Skip de CNN al DNN? | 17.0 → 17.0 | **No.** Descartada |
| ¿Escala a 2000 h? | 4-7% relativo | **Sí** |
| ¿Sobrevive al ruido? | mejora más que en limpio | **Sí**, y aumenta |

### Lo no medido, pese a que suele atribuírsele

Esta lista importa tanto como la anterior.

- **El orden de las capas nunca se ablaciona.** No hay ni un solo experimento con el LSTM antes de la CNN, ni con el DNN antes del LSTM, ni con capas intercaladas. El orden C→L→D se justifica **por argumento** (Pascanu et al., la analogía VTLN/fMLLR→HMM) y se valida solo indirectamente por el hecho de que funciona. Decir "el paper demostró que este es el mejor orden" es falso; lo que demostró es que **este orden funciona mejor que sus componentes aislados**, que es una afirmación mucho más débil.
- **El número de capas convolucionales nunca se ablaciona.** Se fijan 2 capas de 256 mapas heredándolas del trabajo de 2013 del mismo grupo. No hay tabla de 1 vs 2 vs 3 capas, ni de 128 vs 256 vs 512 mapas — en contraste con el barrido del número de capas densas, que sí está. **La asimetría es reveladora: el bloque DNN se sintonizó en este paper, el bloque CNN se importó.**
- **Los tamaños de filtro $9\times9$ y $4\times3$ nunca se ablacionan.** También heredados.
- **La utilidad del pooling nunca se ablaciona.** Es el hueco más llamativo. No hay ningún experimento *sin* pooling, ni con pooling de tamaño 2 o 5, ni —crucialmente— **con pooling en tiempo**. La decisión de hacer pooling solo en frecuencia se toma por cita a un trabajo previo (ASRU 2013) y por argumento físico, **no por evidencia presentada aquí**. Es una decisión correcta y bien fundada, pero su evidencia está en otro paper.
- **La capa lineal de reducción tampoco se ablaciona aquí**: *"In [12] we found that adding this linear layer... allows for a reduction in parameters with no loss in accuracy"*.
- **No hay comparación controlada con CNN+DNN sin LSTM.** El baseline CNN *es* una CNN+DNN, pero con 4 capas densas en vez de 2, sin la capa lineal de reducción y con contexto simétrico en vez de causal. Da 18.0. La comparación que se puede extraer —18.0 contra 17.3— tiene al menos tres variables confundidas.

De las once decisiones de diseño de la arquitectura, este paper ablaciona **dos**: el número de capas DNN y el contexto izquierdo. Todo lo demás está heredado de tres trabajos previos del mismo grupo. CLDNN es, en el sentido más literal, un paper de **integración**: su contribución es mostrar que tres recetas afinadas por separado se componen sin interferencia destructiva.

## Limitaciones

**No es end-to-end en ningún sentido moderno.** Es la limitación más importante y la que más se distorsiona al citar el paper. La CLDNN es **el modelo acústico de un sistema híbrido DNN/HMM**: la salida son 13 522 estados dependientes de contexto que son hojas de un árbol de decisión fonético construido con métodos GMM; la pérdida por entropía cruzada requiere una etiqueta por frame que solo puede venir de un **alineamiento forzado** producido por un sistema previo ya entrenado; el entrenamiento de secuencia opera sobre **lattices**, que exigen decodificador, léxico y modelo de lenguaje; y la inferencia requiere búsqueda de Viterbi. La frase *"unified architecture"* del abstract se refiere a unificar CNN+LSTM+DNN **entre sí**, no a unificar el reconocedor. La transición real a end-to-end en ASR llega después, con CTC, LAS (2016) y RNN-T en producción (2019).

**Latencia.** El paper la gestiona explícitamente —$r = 0$ se elige para no agregar lookahead más allá del retraso de 5 frames de la etiqueta— pero eso mismo es una restricción: la CLDNN es causal en la entrada por obligación operacional, no por diseño óptimo. El límite es ~50 ms.

**Costo de entrenamiento y de inferencia: no reportados.** Ni parámetros, ni FLOPs, ni tiempo de entrenamiento, ni número de máquinas, ni tiempo real de decodificación. Para un paper cuya decisión de diseño central está motivada por latencia, no reportar latencia es una omisión notable. Y la aritmética sugiere que el bloque CNN, en una implementación ingenua, se evalúa una vez por frame de salida sobre una ventana de 11 frames: **~11× de trabajo redundante** frente a una implementación que comparta cómputo entre ventanas solapadas.

**Sin dispersión, sin múltiples semillas, sin significancia estadística.** Los deltas que sostienen la mitad del paper son de 0.2 a 0.4 puntos de WER absoluto. No hay intervalos de confianza ni test de significancia, pese a que en ASR era práctica estándar. Dado que el propio paper demuestra que **cambiar solo la inicialización mueve el WER 0.3 puntos**, cuesta argumentar que un delta de 0.2 —el aporte multi-escala— esté por encima del ruido de entrenamiento. Que ese aporte desaparezca tras el entrenamiento de secuencia refuerza la sospecha.

**Learning rates sintonizados por red y no reportados.** *"the learning rate is chosen specific to each network, and is chosen to be the largest value such that training remains stable"*. Cada arquitectura recibió su propio ajuste, sin protocolo documentado ni valores. Es la fuente de confusión clásica en comparaciones arquitectónicas: no se puede saber cuánta de la diferencia es arquitectura y cuánta es presupuesto de sintonía.

**La separabilidad tiempo/frecuencia es una asunción, no un resultado.** La arquitectura asigna la frecuencia a la CNN y el tiempo al LSTM, con una zona gris —los filtros convolucionales también cubren tiempo—. Esa asignación descansa en tres supuestos no verificados: que la variabilidad relevante en frecuencia es *traslacional* (cierto solo aproximadamente, y solo en la región logarítmica de la escala mel); que un mismo kernel es apropiado en todas las bandas (la compartición completa de pesos lo asume, mientras que la literatura de *limited weight sharing* argumentaba lo contrario, porque la estructura espectral de un sonido nasal a 300 Hz y de una fricativa a 6 kHz no es la misma); y que las interacciones tiempo-frecuencia de largo alcance no importan, ya que el campo receptivo convolucional es acotado y todo lo demás queda para el LSTM, que lo ve solo a través del cuello de botella de 256 dimensiones.

**El log-mel ya es una decisión irreversible.** Descarta la fase, fija una resolución tiempo-frecuencia y aplica una compresión perceptual no aprendida. La CNN opera sobre lo que sobrevive a esa transformación. Sainath atacó exactamente eso el mismo año, con CLDNNs sobre forma de onda cruda.

**Robustez al ruido, evaluada en condiciones emparejadas.** Se entrena con ruido del simulador y se evalúa con ruido del mismo simulador. No hay evaluación *desemparejada* —entrenar limpio, evaluar ruidoso—, que es la que mide robustez de verdad. Lo que se demuestra es que CLDNN aprovecha mejor el aumento de datos, no que sea intrínsecamente más robusta.

**El conteo de objetivos domina el tamaño.** Con 13 522 estados CD, la capa de salida es ~13.9M de ~23.5M parámetros. Cualquier comparación de "tamaño" entre CLDNN, LSTM y DNN en este paper está dominada por una constante compartida que no tiene nada que ver con las arquitecturas comparadas.

## Por qué importa hoy

### El patrón sobrevivió al modelo: CRNN

Lo que se replicó fuera de Google no fue la CLDNN exacta —nadie más necesitaba 13 522 estados CD— sino su **esqueleto**: *convoluciones 2D sobre el espectrograma para extraer features locales, con pooling en frecuencia; recurrencia sobre el eje temporal para agregación; capas densas para clasificar*. El nombre que se le puso fue [**CRNN**](/fundamentos/crnn), y colonizó todo el audio que no era ASR:

- **Detección de eventos sonoros.** Çakır, Parascandolo, Heittola, Huttunen y Virtanen, *Convolutional Recurrent Neural Networks for Polyphonic Sound Event Detection* (IEEE/ACM TASLP 2017), baseline oficial de los desafíos DCASE durante años.
- **Etiquetado de música.** Choi, Fazekas, Sandler y Cho, *Convolutional Recurrent Neural Networks for Music Classification* (ICASSP 2017), sobre Million Song Dataset y MagnaTagATune.
- **Clasificación de escenas acústicas**, detección de actividad de voz y reconocimiento de emociones, con la misma plantilla.

El cambio para pasar de CLDNN a CRNN es literalmente de una línea: reemplazar el softmax de 13 522 estados CD por un pooling temporal y una cabeza del tamaño del número de clases. El survey de [Purwins et al. (2019)](/papers/dl-audio-purwins-2019) documenta esa migración a lo largo de todas las subáreas del [dominio de audio](/dominios/audio).

Vale anotar una confusión frecuente: el acrónimo CRNN designa también la arquitectura CNN+BiLSTM+CTC de reconocimiento de texto en escena (Shi, Bai y Yao, TPAMI 2017), que apareció en la clase 21. Son linajes **independientes** que convergieron a la misma topología porque el problema tiene la misma forma: una entrada 2D con un eje "espacial" que conviene comprimir y un eje secuencial que conviene modelar recurrentemente. En texto ese eje espacial es la altura de la imagen; en audio es la frecuencia. **Es exactamente el mismo argumento de asignación de ejes.** Que dos comunidades sin contacto llegaran a la misma solución es la mejor evidencia de que la intuición era correcta.

### Cuándo dejó de ser estado del arte

Dos golpes sucesivos, por razones distintas. Primero, **el modelo acústico híbrido murió**: entre 2016 y 2019 el campo migró a modelos verdaderamente end-to-end, y eso volvió obsoleta la *función* de CLDNN —emitir posteriores sobre estados CD para un decodificador HMM— con independencia de si su arquitectura era buena. CLDNN no fue superada; su puesto de trabajo desapareció. Segundo, **el LSTM fue reemplazado por atención**. Y ahí el paralelo se vuelve importante.

### Conformer: la misma tesis, otro operador

El argumento de apertura de [Conformer (2020)](/papers/conformer-gulati-2020) es, punto por punto, el mismo argumento de complementariedad de CLDNN, con self-attention en el lugar del LSTM:

> *"self-attention... [is] good at modeling content-based global interactions, while CNNs exploit local features"* — abstract de Conformer.

Compárese con el abstract de CLDNN: *"CNNs are good at reducing frequency variations, LSTMs are good at temporal modeling, and DNNs are appropriate for mapping features to a more separable space"*. **Es literalmente la misma estructura retórica**: dos operadores con sesgos inductivos distintos, mejor juntos que por separado. Cinco años después, con el operador de largo alcance cambiado.

Pero las diferencias son sustantivas y no se reducen a "LSTM → attention":

| | CLDNN (2015) | Conformer (2020) |
|---|---|---|
| Topología | **Pipeline**: bloque conv → bloque recurrente → bloque denso, una vez | **Bloque híbrido repetido**: FFN → MHSA → Conv → FFN, ×16 o ×17 |
| Cuándo se mezclan local y global | **Una sola vez**, en un orden fijo | **En cada nivel de profundidad**, alternando |
| Operador de largo alcance | LSTM unidireccional, $O(T)$ pasos secuenciales | Self-attention multi-cabeza, $O(T^2)$ pero paralelizable |
| Eje de la convolución | **2D sobre (frecuencia, tiempo)** | **1D depthwise sobre tiempo**, kernel 32; la frecuencia ya la colapsó el front-end |
| Pooling | Max-pool en frecuencia, tamaño 3 | Sin max-pooling; submuestreo por convolución con stride |
| Conexiones de salto | 2, concatenativas, una descartada | Residuales aditivas en cada submódulo |
| Normalización | Ninguna mencionada | LayerNorm + BatchNorm dentro del módulo conv |
| Salida | Posteriores sobre 13 522 estados CD, HMM aparte | RNN-T, end-to-end |
| Escala | ~23.5M (derivado) | 10.3M / 30.7M / 118.8M (S/M/L) |

Cuatro observaciones hacen que el paralelo valga la pena.

**Sobrevivió la "C", no la "L".** Conformer conserva un front-end convolucional 2D que submuestrea antes del stack de bloques, exactamente por la razón de CLDNN: la estructura tiempo-frecuencia local existe y una convolución la captura barato. Lo que murió fue el LSTM. Que la parte del paper que sobrevivió sea la convolucional es lo interesante: en 2015 la CNN era la pieza nueva y arriesgada, y el LSTM el caballo de batalla.

**Pipeline contra intercalado es la diferencia arquitectónica real.** En CLDNN, el LSTM ve solo lo que la CNN le dejó pasar por un cuello de botella de 256 dimensiones, y el DNN solo lo que el LSTM le dejó. Es una jerarquía estricta, y su rigidez es precisamente lo que las conexiones de salto estaban intentando aflojar, con éxito muy modesto. Conformer resuelve el mismo problema estructuralmente: al alternar mezcla local y global en cada bloque, la información local no tiene que sobrevivir intacta a través de todo el bloque global. **Las conexiones de salto de CLDNN son un parche a un problema que Conformer resuelve por diseño.**

**La caída del LSTM no fue solo de calidad, fue de paralelización.** La recurrencia impone $T$ pasos secuenciales en el forward y en el BPTT, mientras que la self-attention es una sola multiplicación de matrices grande. Cuando el cuello de botella pasó a ser la utilización de TPUs y GPUs, un operador $O(T^2)$ paralelizable superó a uno $O(T)$ secuencial. Es la misma historia que en NLP.

**Pero la asignación de ejes se mantiene.** Conformer tampoco hace pooling en tiempo, y su convolución es depthwise sobre el eje temporal: la resolución temporal se preserva o se reduce de forma controlada, nunca se colapsa por invarianza. El insight central de CLDNN sobrevivió intacto cinco años y un cambio completo de operador base.

Y conviene registrar el otro camino, el que sí eliminó la convolución: **AST** (Gong, Chung y Glass, 2021) trata el espectrograma como una secuencia de parches y aplica un ViT puro, sin ninguna convolución, inicializado desde ImageNet. Es la refutación más limpia de la tesis de CLDNN: con suficientes datos y suficiente preentrenamiento, el sesgo inductivo convolucional deja de ser necesario. La tesis de la complementariedad no es una ley — es una afirmación sobre el régimen de datos de 2015.

## En la clase 39

El **"Ejemplo 1"** de la [Clase 39](/clases/clase-39) es esta arquitectura, sin nombrarla. El mapeo es casi uno a uno.

| Elemento del ejemplo | Qué dice el paper | Veredicto |
|---|---|---|
| Entrada log-mel 40D | *"each frame $x_t$ is a 40-dimensional log-mel feature"* | **Coincide** |
| Ventanas de 10-20 ms con 5-10 ms de solape | *"computed every 10ms"*; la longitud de ventana **no se especifica** | Coincide el hop; la ventana es de la clase, no del paper |
| 2 capas convolucionales | *"we use 2 convolutional layers"* | **Coincide** |
| 256 filtros por capa | *"each with 256 feature maps"* | **Coincide** |
| Kernel $9\times9$ | *"a 9x9 frequency-time filter for the first convolutional layer"* | **Coincide** |
| Kernel $4\times4$ | *"a **4x3** filter for the second convolutional layer"* | **Difiere** |
| Max-pooling solo en frecuencia | *"pooling in frequency only is performed"* | **Coincide** |
| Ventanas no solapadas de tamaño 3 | *"non-overlapping max pooling... A pooling size of 3 was used for the first layer"* | **Coincide** (solo en la primera capa) |
| Convolución $1\times1$ para reducir dimensión | *"we add a **linear layer** to reduce feature dimension... 256 outputs"* | **Difiere la operación**; coinciden el propósito y el 256 |
| 2 capas LSTM | *"we use 2 LSTM layers"* | **Coincide** |
| 256 celdas por LSTM | *"each LSTM layer has **832 cells**, and a **512 unit projection layer**"* | **Difiere** |
| 2 capas FC de 1024 | *"Each fully connected layer has 1,024 hidden units"* | **Coincide** |

Son **tres números cambiados**, y no tienen la misma importancia.

**El kernel $4\times4$ en vez de $4\times3$.** La diferencia más menor. Vale la pena notar que el $4\times3$ del paper no es arbitrario: con las formas reconstruidas, entra sobre un mapa de $10\times3$ (frecuencia × tiempo) y consume **exactamente** los 3 frames temporales que quedan, dejando el eje de tiempo en 1. Un $4\times4$ ni siquiera cabría sin padding. La versión cuadrada es probablemente una simetrización involuntaria: el primer filtro es $9\times9$, y es fácil asumir que el segundo también es cuadrado.

**El LSTM de 256 celdas en vez de 832 con proyección de 512.** El paper hereda esa configuración de Sak et al. 2014, y la capa de proyección no es decorativa: es lo que permite tener 832 celdas sin que la matriz recurrente sea de $832 \times 832$ por compuerta. Un LSTM de 256 celdas sin proyección es una red sustancialmente más chica — el bloque recurrente pasaría de ~6.8M a menos de 1M de parámetros. Como configuración didáctica es perfectamente razonable: 256 celdas es lo que corre cómodo en un Colab.

**La convolución $1\times1$ en vez de la capa lineal — y esta es la que más importa.** No porque sea el número más grande, sino porque es la única de las tres que **cambia la función de la capa, no su tamaño**:

- Una **convolución $1\times1$** con 256 salidas mapea, en cada posición $(f, t)$ de la grilla, un vector de 256 canales a otro de 256 canales. **Preserva los ejes de frecuencia y tiempo.** Con las formas reconstruidas, su salida sería $256 \times 7 \times 1$, o sea 1792 valores — y no habría reducción de dimensión alguna: el LSTM seguiría recibiendo 1792.
- La **capa lineal del paper** aplana los tres ejes —*"feature-maps × time × frequency context"*— y mapea $1792 \to 256$. Es una reducción real, del orden de $7\times$, y es todo el punto de la capa: ~4.65M de parámetros menos en la matriz de entrada del primer LSTM.

O sea que la sustitución no cambia un hiperparámetro: **anula la función que motiva la existencia de la capa**. La confusión es del todo comprensible, porque en el vocabulario moderno "capa de reducción de dimensión con $1\times1$" es un idiom estándar —Inception, los bottlenecks de ResNet— que el paper de 2015 simplemente no usa. La formulación precisa para la clase es: *aplanar los tres ejes y proyectar linealmente a 256, sin no-linealidad*.

Dos apuntes más sobre la lectura de la clase. Primero, el ejemplo justifica el bloque convolucional con "invarianza a traslación"; como se explicó arriba, el mecanismo es genuinamente el de una CNN, pero el paper habla de *reducir la variación en frecuencia*, y ese matiz es justamente el punto pedagógico: **la invarianza que se busca es específicamente en el eje de frecuencia, y su justificación es la anatomía del tracto vocal, no un principio general de visión**. Sin ese matiz se pierde lo que distingue audio de imagen — en audio, la invarianza a traslación temporal sería un error. Segundo, el ejemplo presenta el orden C→L→D como consecuencia de las propiedades de cada red, cuando el paper lo ancla además en dos argumentos externos: las tres transiciones de Pascanu et al. y la herencia de VTLN/fMLLR antes del HMM. La [profundización de la clase 39](/clases/clase-39/profundizacion) desarrolla ambos.

Y vale rescatar el número que **sí** hay que defender, porque es el único con una justificación física derivable: el pooling de tamaño 3 **solo en frecuencia**.

## Erratas y matices

### Del propio paper

| Errata | Detalle |
|---|---|
| *"we combine CNNs, LSTMs and **CNNs** into one unified framework"* | Debe decir **DNNs**. Es el paper describiendo su contribución central y nombrando mal uno de sus tres componentes |
| *"it could be beneficial to **proceed** LSTMs with a few fully connected CNN layers"* | Debe decir *precede*; además *"fully connected CNN layers"* es contradictorio |
| Glorot y Bengio, AISTATS, **2014** | El paper de Glorot-Bengio es de AISTATS **2010** |
| *"F. Grezl and M. **Karafat**"* | Los apellidos son Grézl y **Karafiát** |
| *"an 4% relative improvement"*, *"an 4-6% relative reduction"* | Artículo incorrecto, dos veces |

Las dos primeras son sintomáticas de un preprint de cinco páginas escrito contra la fecha límite de ICASSP. No afectan la sustancia, pero conviene conocerlas antes de citar el texto literalmente.

### El "4-6%" del abstract no cubre el rango real

Calculando los relativos sobre los WER de las tablas:

| Afirmación del texto | Números de la tabla | Relativo real | Veredicto |
|---|---|---|---|
| *"a **6%** relative reduction... after CE training"* | 14.6 → 13.8 | **5.48%** | Redondeo hacia arriba |
| *"a **5%** relative improvement after sequence training"* | 13.7 → 13.1 | **4.38%** | Redondeo hacia arriba |
| *"a **4%** relative reduction"* (CE, ruidoso) | 20.3 → 19.4 | 4.43% | Correcto |
| *"a **7%** relative improvement"* (secuencia, ruidoso) | 18.8 → 17.4 | 7.45% | Correcto |
| *"**4-6%** relative improvement in WER over an LSTM"* | Rango real medido | **3.9% a 7.4%** | La banda declarada **no contiene el mejor resultado del paper** |

El piso real es 3.9% (200 h, 18.0 → 17.3), ligeramente por debajo de 4; el techo real es **7.4%** (ruidoso con entrenamiento de secuencia, 18.8 → 17.4), muy por encima de 6; y el "6%" del extremo superior solo aparece redondeando un 5.48% hacia arriba. La lectura caritativa es que la banda se armó con los resultados de entropía cruzada —donde el rango es 3.9-5.5%— y no se actualizó tras agregar el entrenamiento de secuencia. Es un caso poco frecuente de un abstract que **infravende** su propio resultado. Si se cita este paper, el número honesto es *"entre 4% y 7% relativo según la condición, con las mayores ganancias en habla ruidosa tras entrenamiento de secuencia"*.

### El baseline LSTM cambia de valor a mitad del paper

{{< concept-alert type="advertencia" >}}
Las primeras cinco tablas usan **LSTM = 18.0** (inicialización gaussiana); las dos siguientes usan **LSTM = 17.7** (inicialización uniforme). Comparar entre tablas de regímenes distintos produce cifras infladas: el **16.8** de la tabla multi-escala **no** debe compararse contra el **18.0** de la tabla de baselines para calcular una mejora del 6.7%. La comparación válida es contra 17.7, o sea **5.1%**.
{{< /concept-alert >}}

Relacionado: los baselines CNN y DNN nunca se re-evalúan tras el cambio de inicialización. La conclusión *"with proper weight initialization, the LSTM is better than the CNN or DNN"* compara el LSTM mejorado (17.7) contra números obtenidos en otra fase del estudio (18.0, 18.4). Es técnicamente defendible —el problema era específico de las capas recurrentes, y CNN/DNN ya usaban Glorot— pero mezcla dos regímenes.

Y la tabla de adiciones multi-escala lista cuatro filas cuando la arquitectura final tiene tres: la cuarta es el experimento fallido que el paper descarta. La Figura 1 dibuja ambos flujos punteados, lo que facilita creer que ambos forman parte del modelo.

### Lo que se le atribuye y no está ahí

- **"CLDNN demostró que el orden CNN→LSTM→DNN es el óptimo."** Falso: el orden nunca se ablaciona.
- **"CLDNN introdujo las conexiones residuales en audio."** Falso en dos sentidos: son **concatenativas**, no residuales, y la motivación es multi-escala, no de optimización.
- **"CLDNN es un modelo end-to-end."** Falso: es el modelo acústico de un sistema híbrido DNN/HMM.
- **"CLDNN demostró que hay que hacer pooling solo en frecuencia."** Falso: lo **hace**, citando trabajo previo, pero **no lo mide** aquí. No hay ninguna ablación de pooling.
- **"CLDNN mejora ~5% sobre el LSTM."** Impreciso: entre 3.9% y 7.4% según la condición, y la mejora **crece** con ruido y con entrenamiento de secuencia.
- **"Las adiciones multi-escala son parte de la CLDNN."** A medias: una de las dos se descarta, y la que sobrevive aporta ~1% con entropía cruzada y **exactamente 0%** tras el entrenamiento de secuencia.
- **"CLDNN usa una convolución $1\times1$ para reducir dimensión."** No: usa una capa lineal sobre el tensor aplanado completo.

## Notas y enlaces

- **Paper:** Tara N. Sainath, Oriol Vinyals, Andrew Senior y Haşim Sak, *Convolutional, Long Short-Term Memory, Fully Connected Deep Neural Networks*, ICASSP 2015. PDF local: [cldnn-sainath-2015.pdf](/papers/cldnn-sainath-2015.pdf). El archivo disponible es el preprint de autor, sin la paginación de las actas.
- **Los tres papers de los que hereda.** Sainath et al., *Deep Convolutional Neural Networks for LVCSR* (ICASSP 2013) aporta el bloque convolucional completo — número de capas, mapas y tamaños de filtro. Sak, Senior y Beaufays (Interspeech 2014) aporta la LSTMP con proyección recurrente y el régimen de entrenamiento distribuido. Sainath et al. (ASRU 2013) aporta la estrategia de pooling. Sin esos tres, CLDNN no tendría hiperparámetros.
- **La referencia conceptual.** Pascanu, Gulcehre, Cho y Bengio, *How to Construct Deep Recurrent Neural Networks* (ICLR 2014): el argumento de las tres transiciones separables de una RNN, que es lo que convierte a CLDNN en algo más que un apilamiento.
- **La continuación inmediata.** Sainath, Weiss, Senior, Wilson y Vinyals, *Learning the Speech Front-end With Raw Waveform CLDNNs* (Interspeech 2015): reemplaza el log-mel por una capa convolucional 1D sobre la forma de onda cruda y muestra que la red aprende un banco de filtros con respuesta parecida a mel. Ataca directamente la limitación del front-end fijo, usando la CLDNN como chasis.
- **Fundamentos relacionados:** [redes convolucionales](/fundamentos/redes-convolucionales), [LSTM y GRU](/fundamentos/lstm-gru), [CRNN](/fundamentos/crnn), [MFCC y escala mel](/fundamentos/mfcc-y-escala-mel).
- **En el sitio:** [Clase 39](/clases/clase-39) y su [profundización](/clases/clase-39/profundizacion); el [dominio de audio](/dominios/audio); [Conformer (2020)](/papers/conformer-gulati-2020) como continuación directa de la tesis; y el survey de [Purwins et al. (2019)](/papers/dl-audio-purwins-2019) para el mapa completo del período.
