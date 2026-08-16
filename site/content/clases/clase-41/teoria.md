---
title: "Teoría - Speech Recognition y Speaker Recognition"
weight: 10
math: true
---

> **Recorrido de la Clase 41** del Diplomado IA UC (Gabriel Sepúlveda, IALab PUC). Son dos presentaciones: *DL Models for Audio Processing: Speech Recognition* (52 diapositivas) y *Audio Analytics Laboratory: Speaker Recognition* (36). Se recorren en ese orden porque la segunda se apoya en una distinción que la primera establece: **el reconocimiento de voz quiere resolución temporal; el de hablante quiere colapsarla**.

---

# Parte 1 — Reconocimiento de voz

## 1. Dónde estamos (slides 2-5)

La clase 39 había dividido las aplicaciones de audio en tres:

| Familia | Ejemplos |
|---|---|
| Sonidos ambientales | clasificación de sonidos, audio tagging |
| Música | reconocimiento de canciones, identificación de instrumentos |
| **Habla** | reconocimiento de voz, traducción, identificación de hablante |

Esta clase se ocupa de la tercera. Y recuerda la receta con la que cerró la anterior: **CNN** para features locales significativos, **RNN** para features temporales globales y de larga distancia, **MLP** para clasificar. Con la nota al pie de que si se entra con onda cruda hacen falta [convoluciones dilatadas](/fundamentos/convoluciones-dilatadas) y submuestreo agresivo en la primera capa.

## 2. La pregunta que ordena todo (slides 6-21)

> *"Can we use a similar architecture to the case of speech recognition?"*
> *"Sure you do, but it does not achieve state-of-the-art performance."*

Y a continuación, por qué. La clase construye el argumento en dos tiempos, con una secuencia de diapositivas que van revelando viñetas.

**En sonidos ambientales, lo decisivo es aprender features:**
- hay casi un número infinito de sonidos distintos;
- features específicos pueden ser altamente relevantes para caracterizar cada sonido;
- aprender un conjunto representativo y discriminativo es la clave.

**En habla, lo decisivo es aprender la secuencia:**
- para un idioma dado, **el espacio de fonemas no es tan grande**;
- discriminar fonemas no es tan difícil, aun lidiando con voces distintas, estilos y ruido;
- **la segmentación es el problema clave**: ¿cómo detectar palabras significativas en una señal continua?
- decodificar habla desde una señal de audio no es un problema fácil.

{{< concept-alert type="clave" >}}
*"Feature learning is important but **sequence learning is the real key**."*

Esta es la tesis de la primera mitad, y explica por qué la arquitectura cambia. No es que las CNN dejen de servir —siguen extrayendo los features locales— sino que el problema difícil se mudó: de **qué representar** a **cómo alinear** una secuencia de cientos de frames con una de decenas de caracteres, sin saber cuál corresponde a cuál. Ver [Reconocimiento de voz](/fundamentos/reconocimiento-de-voz).
{{< /concept-alert >}}

## 3. El catálogo de modelos (slide 22)

La clase enumera diez familias: MLP, CNN, RNN, GAN, **Seq2Seq**, **Seq2Seq con atención**, Transformers y modelos relacionales, aprendizaje reforzado, aprendizaje por imitación y modelos neuro-simbólicos.

Y desarrolla la quinta y la sexta. Vale notar lo que queda fuera: los **Transformers** aparecen listados pero no se desarrollan, y en 2019-2020 esa omisión era razonable — [Whisper](/papers/whisper-radford-2022) es de 2022. Hoy la arquitectura dominante en ASR es precisamente la que la clase menciona al pasar.

## 4. Recordando seq2seq (slides 23-28)

**[Seq2Seq](/papers/seq2seq-sutskever-2014) (Sutskever et al., 2014).** El encoder lee la secuencia de entrada y produce un vector; el decoder genera la salida a partir de él. La clase subraya el problema en negrita: **el embedding contextual intermedio $C$ es fijo**. Toda la oración de entrada tiene que caber en un vector de tamaño constante, sin importar su longitud.

**[Seq2Seq con atención](/papers/bahdanau-attention-2015) (Bahdanau et al., 2015).** El decoder consulta todos los estados del encoder en cada paso y construye un contexto distinto: **$C_t$ es adaptativo**. El ejemplo de la clase es una traducción con reordenamiento —*"El auto rojo de Carlos está averiado"* → *"Carlos's red car is not working"*— donde la correspondencia entre posiciones no es monótona.

## 5. Aplicarlo al habla (slides 29-40)

> *"Can we apply a Seq2Seq+Att model to speech recognition? What are the inputs and outputs?"*

**La entrada.** Vectores de features **mel de 40 dimensiones**, uno por ventana. La clase muestra $x_1, x_2, x_3, \dots$ apilándose en el tiempo. El detalle de la configuración aparece en el Ejemplo 1: **segmentos solapados de 10-20 ms, con 5-10 ms de solape**.

**La salida.** Caracteres: `"a...z"`, `"."`, `","`, `" "`, `"EOS"`, `"0-9"`, con clasificación softmax. Trabajar a nivel de caracteres —y no de palabras— evita el problema del vocabulario cerrado: las palabras raras se deletrean.

**Ejemplo 1 de arquitectura:**

| Componente | Configuración |
|---|---|
| Entrada | 40D log-mel, ventanas de 10-20 ms, solape de 5-10 ms |
| Encoder | 4 capas BiLSTM apiladas, estado oculto 256D |
| Decoder | 4 capas BiLSTM apiladas, estado oculto 256D |
| Salida | caracteres, softmax |

Esa configuración es esencialmente la de [Graves et al. (2013)](/papers/deep-rnn-speech-graves-2013), el primer trabajo que mostró que apilar BiLSTM en profundidad —y no solo en el tiempo— mejoraba el reconocimiento, con **17,7 % de PER en TIMIT**.

## 6. El problema de la segmentación (slides 41-44)

Acá está el nudo de la primera mitad:

> *"One problem with RNNs is that, at each prediction time, they are trained to perform discrete label classifications. This works fine when inputs and outputs are clearly segmented."*
>
> *"In the case of speech recognition signal segmentation is an issue. Outputs correspond to segmented chars. **However, inputs producing each output are not pre-segmented.** Relevant information is hidden in the Mel features."*

Un audio de 3 segundos son ~300 vectores de features; su transcripción, quizá 40 caracteres. No hay ninguna marca que diga qué frames produjeron cuál. Y un clasificador frame a frame necesitaría exactamente esa información para entrenarse.

## 7. La idea del blank (slides 45-50)

> **IDEA: Add to the output a blank token.**
>
> *"The use of a blank output label facilitates the task of sequence alignment between the input and output, it can be considered as a waiting or pause output symbol."*

Con un símbolo que significa "acá no emito nada", cada frame puede tener una etiqueta sin que haga falta saber la alineación real: los frames que no corresponden a ningún carácter emiten *blank*. La secuencia de largo $T$ se colapsa después a la transcripción de largo $U$ eliminando blanks y repeticiones.

La clase menciona dos métodos que aprovechan la idea:

- **[CTC](/papers/ctc-graves-2006)** (Graves et al., 2006): *"CTC network has a softmax output layer with an extra label corresponding to observing a blank output"*.
- **Pooling Over Time** ([Bahdanau et al., 2016](/papers/e2e-lvsr-bahdanau-2016)).

{{< concept-alert type="cuidado" >}}
**Los dos métodos que la slide pone en paralelo no resuelven lo mismo.**

**CTC** es un objetivo de entrenamiento: introduce el blank y **suma sobre todas las alineaciones** compatibles con la transcripción, de modo que la alineación deja de hacer falta. Resuelve la desalineación.

**El pooling sobre el tiempo** es una modificación del encoder: agrega frames vecinos para **acortar la secuencia de entrada**. No resuelve la alineación — resuelve el **costo** de que la atención tenga que recorrer cientos de posiciones por cada carácter emitido. Es complementario a la atención, no alternativo.

Se combinan sin problema, y los sistemas modernos suelen hacerlo. La derivación está en la [profundización](profundizacion).
{{< /concept-alert >}}

**Ejemplo 2 de arquitectura**, que es el Ejemplo 1 más las dos ideas:

| Componente | Cambio respecto del Ejemplo 1 |
|---|---|
| Decoder | 4 capas bidireccionales **con pooling temporal de ventana 2** |
| Salida | softmax **usando CTC** |

Ese *pooling in time window size = 2* es la **pirámide de [LAS](/papers/las-chan-2016)**: cada capa reduce a la mitad los pasos temporales. Con tres capas, la reducción es de 8×. Y hay un matiz que la clase no menciona: en LAS la pirámide **no es una optimización**, es un requisito de convergencia — los autores reportan que sin ella, tras un mes de entrenamiento, los errores seguían muy por encima.

## 8. Cómo se mide (slide 52)

$$\text{WER} = \frac{\#\text{Insertions} + \#\text{Deletions} + \#\text{Substitutions}}{\#\text{tokens}}$$

La tasa de error de palabras, calculada alineando hipótesis y referencia por distancia de edición. Tres propiedades que conviene tener presentes y que la slide no menciona: **puede superar el 100 %** (las inserciones no están acotadas), **trata todos los errores por igual** (confundir "no" por "sí" cuesta lo mismo que equivocar un artículo), y **depende críticamente de la normalización** de mayúsculas, puntuación y números.

---

# Parte 2 — Reconocimiento de hablante

## 9. La tarea y su primera trampa (slides 2-7)

> *"Process of automatically recognizing who is speaking by using the speaker-specific information included in speech waves to verify identities being claimed by people accessing systems."*

La clase construye el argumento como una escalera de preguntas:

- *¿Cómo modelamos el reconocedor?* → **Como un clasificador** que calcula la probabilidad de cada hablante dada la señal.
- *¿Cómo lo entrenamos?*
- *¿Cómo incorporamos gente nueva?* → **"Our model must be trained entirely for each new speaker!"**

Ese signo de exclamación es el punto de quiebre. Un clasificador con $N$ salidas asume un **conjunto cerrado** de identidades: agregar una persona significa cambiar la última capa y reentrenar. Inviable en cualquier sistema real.

## 10. La solución: descriptores (slides 8-9)

> *"A better idea would be to calculate a descriptor… and then, to calculate the similarity between them."*

$$\text{score} = v_1 \cdot v_2$$

*"If score is high, then $v_1$ y $v_2$ come from the same person. If score is low, they come from different persons."*

El cambio es de **clasificar** a **medir**. La red produce un vector por enunciado, y la decisión se toma comparando vectores. Incorporar a alguien es calcular su vector y guardarlo: el modelo no se toca. Es el mismo movimiento que [FaceNet](/papers/facenet-schroff-2015) hizo en reconocimiento facial — ver [Metric learning](/fundamentos/metric-learning).

## 11. Qué hace difícil la señal (slides 10-12)

**Características del audio:** longitud variable, contenido irrelevante (ruido ambiental), frecuencia de muestreo común asumida.

**Objetivos de diseño:** representar cada señal con un vector de dimensión fija, filtrar las partes irrelevantes, capturar los componentes básicos de la voz humana.

**Estrategia:** dividir la señal en frames (descriptores locales) y **agregar los features a lo largo del tiempo, incluyendo solo los componentes relevantes**.

Esa última frase contiene todo el problema de la segunda mitad. Agregar es fácil —se promedia y listo—; agregar **incluyendo solo lo relevante** es lo difícil.

## 12. El modelo conceptual (slides 13-18)

Seis diapositivas construyen la intuición antes de nombrar el método. Un espacio de features con centroides que representan características básicas de la voz humana: **"voz grave"**, **"voz aguda"**, **"voz siseante"**. Sobre él:

1. se quiere **proyectar** una señal de voz;
2. para proyectarla, se calculan **features temporales locales**;
3. y finalmente se crea **una representación nueva basada en el nivel de pertenencia a cada centroide**.

Es una buena forma de introducirlo, con una salvedad: los centroides reales no son interpretables como "voz grave" o "voz siseante". Son puntos en un espacio de features aprendido, sin significado nombrable. La analogía ayuda a entender el mecanismo y no debe tomarse literalmente.

## 13. VLAD (slides 19-25)

> *"VLAD is a popular pooling method for both instance level retrieval and classification. It captures information about the statistics of local descriptors."*

Los cinco pasos, tal como los enumera la clase:

1. Proyectar el conjunto de entrenamiento en el espacio de features.
2. Aprender $k$ centroides $c_1, \dots, c_k$.
3. Asignar descriptores a cada centroide: $c(x) = \arg\min_{c_i} \lVert c_i - x \rVert^2$.
4. Calcular $x - c_i$.
5. Agregación de residuales.

$$v(j,k) = \sum_{i=1}^{N} a_k(x_i) \cdot \big(x_i(j) - c_k(j)\big)$$

El resultado es una matriz de $J \times K$ — dimensionalidad del espacio por número de centroides.

Lo que hace de VLAD algo distinto de un histograma es que **no cuenta ocupación sino que acumula desviaciones**: no dice "cayeron tres descriptores cerca de este prototipo" sino "los que cayeron cerca están desplazados así". La [profundización](profundizacion) muestra un caso donde dos distribuciones con **idéntica media global** son perfectamente separables por VLAD e indistinguibles por promedio.

Y la clase cierra la sección con la pregunta correcta:

> *"In this process we need to find $x_i$, $c_k$ and $a_k(x_i)$. **Is it possible to learn all of them end-to-end?**"*

## 14. NetVLAD (slides 26-29)

La respuesta es el paper de [Xie et al. (2019)](/papers/utterance-level-xie-2019), *Utterance-level Aggregation for Speaker Recognition in the Wild*, presentado como cinco etapas: **Encoding → Projection → Centroid Ownership → Feature Aggregation → Final representation**.

Lo que hay que aprender:

- **Encoder**: un *thin ResNet* — una ResNet-34 con los canales recortados, de 3 millones de parámetros contra los 22 de la estándar.
- **Centroides** $c_k$: una matriz de $J \times K$.
- **La función de pertenencia**, que es la pieza clave. La asignación dura $a_k(x_i)$ se reemplaza por una **asignación blanda a múltiples clusters**:

$$\bar{a}_k(x_i) = \frac{e^{\,w_k x_i + b_k}}{\sum_{k'} e^{\,w_{k'} x_i + b_{k'}}}$$

{{< concept-alert type="clave" >}}
**Por qué hacía falta este cambio.** El `argmin` del paso 3 es constante a trozos: su derivada es cero en casi todas partes y no existe en las fronteras entre celdas. **No hay gradiente que propagar**, y por lo tanto ni los centroides ni el extractor de features pueden aprenderse para la tarea.

Reemplazarlo por un softmax lo vuelve derivable en todas partes, y de paso permite **desacoplar** el peso de asignación ($w_k$, $b_k$) del centro del residuo ($c_k$), que en VLAD clásico eran el mismo objeto. Toda la contribución de [NetVLAD](/papers/netvlad-arandjelovic-2016) cabe en eso. Ver [Agregación VLAD](/fundamentos/agregacion-vlad).
{{< /concept-alert >}}

**El postprocesamiento** (slide 29), que la clase detalla paso a paso con sus rangos:

$$x = \text{DimReduction}(x) \;\Rightarrow\; x_i \in [-\infty, \infty]$$
$$x = \text{ReLU}(x) \;\Rightarrow\; x_i \in [0, \infty]$$
$$x = \text{L2\_norm}(x) \;\Rightarrow\; x_i \in [0, 1]$$

La cadena no es decorativa: garantiza que el producto punto entre dos descriptores caiga en $[0,1]$, que es lo que permite interpretarlo directamente como puntaje de similitud sin normalizaciones adicionales.

## 15. El protocolo (slide 30)

- Entrenado en **[VoxCeleb2](/papers/voxceleb2-chung-2018)** (5 994 hablantes).
- Evaluado en **[VoxCeleb1](/papers/voxceleb-nagrani-2017)**.
- *"VoxCeleb1 and VoxCeleb2 are completely disjoint!"*

Esa disyunción es lo que hace válida la evaluación: el modelo nunca vio a las personas del test, así que no puede haber memorizado sus voces. Solo puede haber aprendido a **representar voces en general** — que es exactamente lo que se afirmó en la slide 8 al pasar de clasificador a descriptor.

Lo que la clase no muestra, y es el resultado más contundente del paper: **con el mismo backbone y los mismos datos**, cambiar la agregación de promedio temporal a NetVLAD lleva el EER de **10,48 % a 3,57 %**.

## 16. Del score al umbral (slides 31-35)

Con la cadena ReLU + L2 el score cae en $[0,1]$, y queda la pregunta operativa:

> *"How to determine which value is low and which value is high? In other words, what is the best threshold?"*

La clase responde construyendo la curva ROC. Primero las dos tasas, **en función del umbral**:

$$\text{TPR} = \frac{TP}{TP+FN}, \qquad \text{FPR} = \frac{FP}{TN+FP}$$

Y después: *"we can plot TPR and FPR for different threshold values in order to obtain the Receiver Operating Characteristic (ROC) curve"*.

La clase termina ahí, en la curva. Falta el paso siguiente, que es el que produce el número que todos los papers del área reportan: el **EER** (*equal error rate*), el punto de la curva donde la tasa de falsos positivos iguala a la de falsos negativos. Es lo que permite resumir un sistema en un solo número sin comprometerse con un umbral operativo — con la advertencia de que en producción los dos errores casi nunca cuestan lo mismo. Ver [Reconocimiento de hablante](/fundamentos/reconocimiento-de-hablante).

---

## Las dos mitades, juntas

| | Reconocimiento de voz | Reconocimiento de hablante |
|---|---|---|
| Pregunta | ¿qué se dijo? | ¿quién lo dijo? |
| Representación deseada | **por frame** | **por enunciado** |
| Problema central | alineación entrada-salida | agregación temporal |
| Solución de la clase | blank + CTC, pooling temporal | descriptor + VLAD |
| Conjunto de salida | cerrado (caracteres) | **abierto** (identidades) |
| Métrica | WER | EER sobre curva ROC |
| Qué se descarta | la identidad del hablante | el contenido lingüístico |

La última fila es la más elocuente: **cada tarea trata como ruido lo que la otra trata como señal**.

---

## Ver también

- [Profundización](profundizacion) — la matemática: la suma sobre alineaciones de CTC, la independencia condicional y su costo, el gradiente bloqueado por `argmin` y la geometría de los residuos.
- [Práctica desde 0](practica) — implementar CTC y VLAD, y verificarlos numéricamente en triple framework.
- [Clase 39](/clases/clase-39) — la receta CNN+RNN+MLP que esta clase pone en duda.
- [Clase 13](/clases/clase-13) — seq2seq y atención en su contexto original.
