---
title: "CRNN: arquitecturas convolucional-recurrentes"
weight: 129
math: true
---

Una **CRNN** (*convolutional recurrent neural network*) apila una red convolucional y una recurrente en un solo modelo entrenado de punta a punta: la convolución extrae features locales sobre una entrada estructurada, y la recurrencia modela la evolución de esos features a lo largo de una secuencia. Es la respuesta arquitectónica a una observación simple — **los dos operadores tienen sesgos inductivos complementarios y ninguno reemplaza al otro** — y durante la segunda mitad de los 2010 fue la arquitectura por defecto en reconocimiento de voz, reconocimiento de texto en imágenes, detección de eventos sonoros y etiquetado musical.

La familia se formalizó en audio con la [CLDNN de Sainath et al. (2015)](/papers/cldnn-sainath-2015), que le agregó un tercer bloque de capas densas y le puso nombre a la receta: **C**onvolutional, **L**ong short-term memory, **D**eep **N**eural **N**etwork. En visión apareció casi simultáneamente con el CRNN de Shi et al. (2015) para reconocimiento de texto, combinado con [CTC](/fundamentos/ctc-loss).

Este fundamento cubre el patrón de forma autónoma: la tesis de la complementariedad y qué aporta cada bloque, la interfaz entre ellos (que es donde vive el costo y donde se cometen los errores), sus variantes por dominio, y el veredicto histórico — qué parte de la receta sobrevivió y cuál fue reemplazada.

---

## 1. La tesis: tres sesgos inductivos distintos

Cada bloque aporta una restricción sobre qué funciones puede representar la red, y esas restricciones no se solapan.

| Bloque | Qué explota | Qué descubre | Su límite |
|---|---|---|---|
| **CNN** | Localidad del filtro y equivariancia a traslación | Patrones **locales** repetidos en cualquier posición | Su campo receptivo crece lentamente con la profundidad |
| **RNN** | Un estado que se propaga paso a paso | Dependencias **temporales de largo alcance** | El camino entre dos posiciones es $O(T)$; no paraleliza |
| **MLP** | Nada estructural | Un mapeo a un espacio **linealmente separable** por clase | Sin estructura, necesita features ya informativos |

El orden importa y no es arbitrario. La convolución va primero porque opera sobre la entrada de alta dimensión y la reduce: es la única de las tres que puede procesar una grilla grande de forma barata. La recurrencia va en el medio porque necesita una secuencia de vectores, no una grilla. Las densas van al final porque son las más caras por unidad de entrada y conviene alimentarlas con lo mínimo indispensable.

{{< concept-alert type="clave" >}}
**El argumento no es "más bloques es mejor", sino que ninguno cubre lo del otro.** Una CNN pura puede modelar dependencias largas, pero necesita profundidad o dilatación para lograrlo, y su noción de "largo" está fijada en la arquitectura. Una RNN pura puede procesar la entrada cruda, pero desperdicia su capacidad aprendiendo detectores locales que una convolución obtiene con mucha menos supervisión gracias a la compartición de pesos.

En reconocimiento de voz el argumento tiene además una forma concreta: la convolución cumple el papel que en el pipeline clásico cumplía la **normalización de la longitud del tracto vocal (VTLN)** — compensar que un mismo fonema aparece a frecuencias distintas según la anatomía de quien habla. Por eso su pooling va en el eje de frecuencia y no en el de tiempo.
{{< /concept-alert >}}

---

## 2. La interfaz: donde vive el costo

El punto de contacto entre el bloque convolucional y el recurrente es la decisión de diseño más consecuente de una CRNN, y la que menos atención suele recibir.

El bloque convolucional produce un tensor de forma $(\text{mapas}, T', F')$. El bloque recurrente necesita una secuencia de vectores, $(T'', d)$. Hay que convertir uno en el otro, y hay tres maneras:

| Estrategia | Qué hace | Consecuencia |
|---|---|---|
| **Aplanar todo** | Colapsa mapas, tiempo y frecuencia en un vector | El LSTM recibe un solo paso; se pierde la secuencia |
| **Aplanar mapas × frecuencia, conservar el tiempo** | Cada trama es un vector de $\text{mapas} \times F'$ | Lo estándar: la secuencia se preserva |
| **Reducir antes de aplanar** | Una capa lineal o una convolución $1\times1$ baja la dimensión | Lo que hace la CLDNN, y es lo que hace la red asequible |

La razón por la que la tercera existe está en la fórmula del costo de un LSTM con $h$ celdas y entrada de $d_{\text{in}}$ dimensiones:

$$\text{params} = 4\,\big(d_{\text{in}}\, h + h^2 + h\big)$$

El término $d_{\text{in}} \cdot h$ es la matriz entrada→estado, y crece linealmente con lo que se le entregue. Un bloque convolucional de 256 mapas sobre una grilla de $9 \times 8$ aplanado da 18.432 valores; multiplicado por 256 celdas, cuatro compuertas y dos capas, son **19.7 millones de parámetros solo en el bloque recurrente**.

{{< concept-alert type="advertencia" >}}
**En una CRNN, el cuello de botella de parámetros casi nunca está donde uno mira.** Midiendo la CLDNN del "Ejemplo 1" de la [Clase 39](/clases/clase-39):

| Bloque | Parámetros | % del total |
|---|---|---|
| Primera convolución $9\times9$ | 20.992 | 0.3% |
| Segunda convolución | 786.688 | 10.0% |
| **Capa de reducción** | **4.718.848** | **59.7%** |
| Dos LSTM de 256 celdas | 1.052.672 | 13.3% |
| Dos densas de 1.024 | 1.312.768 | 16.6% |

La convolución que "aprende los features locales" —el bloque al que se dedica toda la justificación conceptual— es el 0.3% de la red. Casi el 60% está en la capa de interfaz. Es el primer lugar donde mirar al recortar un modelo, y la medición completa está en la [práctica de la Clase 39](/clases/clase-39/practica/02-la-cldnn-del-ejemplo-1).
{{< /concept-alert >}}

### El pooling asimétrico

La otra decisión de interfaz, específica del audio: **el pooling del bloque convolucional debe ir en frecuencia y no en tiempo**.

El argumento es directo. La capa siguiente es un modelo de secuencia cuyo trabajo entero es modelar la evolución temporal; hacer pooling en tiempo antes de él le quita su entrada. Un pooling temporal de 3 reduce las tramas disponibles en dos tercios. En una CNN pura de clasificación eso sería inofensivo —al final se colapsa todo igual— pero en una arquitectura híbrida es destructivo.

El pooling en frecuencia, en cambio, es deseable: da invarianza local a los pequeños desplazamientos espectrales que introduce la anatomía de cada hablante, sin borrar la identidad del sonido. Ver [MFCC y escala Mel](/fundamentos/mfcc-y-escala-mel).

---

## 3. Variantes por dominio

**Audio y habla (CLDNN).** Entrada log-mel, convoluciones 2D con pooling en frecuencia, capa de reducción, LSTM apiladas, capas densas. Salida por trama para reconocimiento de voz, o agregada para clasificación. Es el patrón que la [Clase 39](/clases/clase-39) presenta como "Ejemplo 1".

**Detección de eventos sonoros.** La misma estructura, pero con salida **por trama** y sigmoides independientes por clase, porque los eventos se solapan. La recurrencia es habitualmente **bidireccional**, ya que en detección offline no hay restricción de causalidad. Ver [clasificación de audio](/fundamentos/clasificacion-de-audio).

**Reconocimiento de texto en imágenes (CRNN + CTC).** La convolución recorre la imagen de izquierda a derecha produciendo una secuencia de columnas de features, el LSTM bidireccional las procesa, y [CTC](/fundamentos/ctc-loss) resuelve el alineamiento entre la secuencia de salida y el texto sin necesidad de anotación por carácter. Es la arquitectura base del reconocimiento de texto en escenas — ver [scene text recognition](/fundamentos/scene-text-recognition).

**Video.** El equivalente es CNN 2D por fotograma seguida de LSTM sobre la secuencia de features, la familia LRCN. Ver [análisis de video](/fundamentos/analisis-de-video).

**Música.** El frontend convolucional se diseña con **filtros de forma musical** en lugar de cuadrados: verticales y angostos para el timbre, horizontales y largos para el ritmo. Ver [musicnn](/papers/musicnn-pons-2019).

---

## 4. Las conexiones de salto

La CLDNN introdujo dos conexiones que saltan bloques, y su resultado es instructivo:

- **Entrada cruda concatenada a la salida de la CNN, antes del LSTM.** Funciona: recupera la resolución en frecuencia que el pooling descartó, que el LSTM puede aprovechar.
- **Salida de la CNN concatenada a la del LSTM, antes de las densas.** **No funciona**: el paper mide 17.0 contra 17.0 de WER y la descarta.

Son un antecedente conceptual de las conexiones residuales, y también un síntoma: **son un parche a un problema de la estructura en pipeline**. Cuando los bloques se apilan en serie, la información que uno descarta se pierde para siempre, y hay que reinyectarla a mano. Las arquitecturas posteriores resolvieron eso por diseño, repitiendo un bloque híbrido que contiene ambos operadores en vez de encadenar bloques homogéneos.

---

## 5. El veredicto histórico

La CRNN dejó de ser el estado del arte alrededor de 2020, y vale precisar exactamente qué fue reemplazado.

**Lo que sobrevivió: la convolución, y la tesis de la complementariedad.** Todos los modelos de audio con Transformer tienen un frontend convolucional. [Conformer](/papers/conformer-gulati-2020) lo hace explícito: mantiene la tesis de que hacen falta un operador local y uno global, y los fusiona dentro de un mismo bloque repetido. wav2vec 2.0, HuBERT y Whisper usan convoluciones para reducir la señal antes de atender.

**Lo que fue reemplazado: la recurrencia como operador global.** La self-attention conecta cualquier par de posiciones en una sola operación —camino de longitud $O(1)$ frente a $O(T)$ en una RNN— y además paraleliza el entrenamiento, que una RNN no puede por construcción.

{{< concept-alert type="clave" >}}
**Por qué la convolución no desapareció, cuantificado.** La self-attention cuesta $O(T^2 d)$ y una capa recurrente $O(T d^2)$, así que el cruce está en $T = d$. Con $d = 512$, atender sobre diez segundos de onda cruda a 16 kHz ($T = 160\,000$) cuesta **312 veces más** que una capa recurrente; sobre la misma señal reducida a 50 tramas por segundo ($T = 500$), cuesta **lo mismo**.

Esa reducción de 320× es lo que hace el encoder convolucional. Su función principal en un Transformer de audio no es "extraer features locales": es **bajar $T$ hasta el régimen donde la atención es asequible**. De la receta CRNN sobrevivió la C, no la R — y sobrevivió por una razón de presupuesto tan importante como la conceptual.
{{< /concept-alert >}}

**Dónde sigue siendo la elección correcta.** En modelos pequeños con restricciones de latencia o de memoria, y en dispositivos embebidos: una CRNN de pocos millones de parámetros con streaming causal sigue siendo competitiva para keyword spotting o detección de eventos on-device, donde un Transformer no cabe.

---

## 6. Errores frecuentes

**Hacer pooling en tiempo antes del bloque recurrente.** Ya cubierto: destruye la resolución que la recurrencia consume.

**Entregarle al LSTM el bloque convolucional aplanado sin reducir.** Produce una red donde el 90% de los parámetros están en una matriz que nadie inspeccionó.

**Usar softmax con eventos que se solapan.** En audio la simultaneidad es la norma, no la excepción — los sonidos se suman en lugar de ocluirse. Ver [clasificación de audio](/fundamentos/clasificacion-de-audio).

**No normalizar el largo de secuencia en el batch.** El audio no viene en clips de largo fijo, y un modelo recurrente sobre un tensor rectangular exige que todo el batch tenga la misma longitud: se rellena y se enmascara, o se agrupan ejemplos de largo parecido. Ver el manejo de `collate_fn` en la [Clase 37](/clases/clase-37).

**Comparar conteos de parámetros entre frameworks sin más.** PyTorch parametriza el LSTM con dos vectores de sesgo por capa y Keras con uno; la diferencia es $2 \times 4 \times h$ por capa. Son matemáticamente equivalentes y numéricamente distintos en el reporte.

---

## Ver también

**Papers:** [CLDNN (Sainath 2015)](/papers/cldnn-sainath-2015) · [Conformer (2020)](/papers/conformer-gulati-2020) · [musicnn (2019)](/papers/musicnn-pons-2019) · [Deep Learning for Audio Signal Processing (2019)](/papers/dl-audio-purwins-2019) · [LRCN (2015)](/papers/lrcn-donahue-2015).

**Fundamentos:** [Redes Convolucionales](/fundamentos/redes-convolucionales) · [LSTM y GRU](/fundamentos/lstm-gru) · [Redes Recurrentes](/fundamentos/redes-recurrentes) · [Self-attention](/fundamentos/self-attention) · [CTC Loss](/fundamentos/ctc-loss) · [Clasificación de audio](/fundamentos/clasificacion-de-audio) · [Convoluciones dilatadas](/fundamentos/convoluciones-dilatadas).

**Clases:** [Clase 39 - Modelos de DL para audio](/clases/clase-39) · [Práctica: la CLDNN del Ejemplo 1](/clases/clase-39/practica/02-la-cldnn-del-ejemplo-1) · [Dominio: Audio / Voz](/dominios/audio).
