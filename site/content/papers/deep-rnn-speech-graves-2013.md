---
title: "Deep RNNs para Speech Recognition (2013)"
weight: 435
math: true
---

{{< paper-card
    title="Speech Recognition with Deep Recurrent Neural Networks"
    authors="Alex Graves, Abdel-rahman Mohamed, Geoffrey Hinton (University of Toronto)"
    year="2013"
    venue="ICASSP 2013 / arXiv:1303.5778"
    pdf="/papers/deep-rnn-speech-graves-2013.pdf" >}}
La pregunta del paper cabe en una línea: **una RNN ya es profunda en el tiempo, ¿gana algo con ser profunda en el espacio?** Hasta 2013 las RNN rendían decepcionantemente en reconocimiento de voz —las redes feedforward profundas les ganaban— y los autores sospechan que el problema no era la recurrencia sino la falta de jerarquía. Apilan varias capas **LSTM bidireccionales** una sobre otra, las entrenan de punta a punta con [CTC](/papers/ctc-graves-2006) y obtienen **17.7 % de PER en TIMIT**, el mejor resultado publicado en ese momento. Es el paper que fija la arquitectura que la [Clase 41](/clases/clase-41) presenta como su "Ejemplo 1": varias capas BiLSTM apiladas sobre features log-mel, con salida a nivel de caracteres o fonemas.
{{< /paper-card >}}

---

## Contexto: la recurrencia no bastaba

En 2012-2013 las redes profundas ya habían desplazado a los GMM en el modelado acústico, pero las que funcionaban eran **feedforward**: ventanas de contexto fijas alimentando un MLP profundo dentro de un sistema DNN-HMM. Las RNN, que en teoría son la familia natural para secuencias, quedaban atrás.

El diagnóstico de los autores es que se estaba comparando mal. Una RNN **ya es profunda en el tiempo** —su estado oculto depende de todos los estados anteriores— pero eso no le da lo mismo que la profundidad espacial le da a una feedforward: una **jerarquía de representaciones**, de features acústicos de bajo nivel a estructuras fonéticas de alto nivel. La pregunta del paper es si ambas profundidades son complementarias.

## Método: apilar BiLSTM y entrenar end-to-end

Tres ingredientes que hoy suenan obvios y en 2013 no lo eran:

**Bidireccionalidad.** Cada capa procesa la secuencia hacia adelante y hacia atrás, y concatena. En reconocimiento de voz esto es casi obligatorio: la identidad de un fonema depende de lo que viene *después* tanto como de lo que vino antes — la coarticulación es simétrica.

**Profundidad espacial.** Varias capas BiLSTM apiladas, cada una tomando como entrada la salida de la anterior. El paper explora de 1 a 5 capas y encuentra que el rendimiento mejora con la profundidad hasta saturar.

**Entrenamiento end-to-end con CTC.** Sin alineación forzada previa, sin HMM: [CTC](/papers/ctc-graves-2006) marginaliza sobre todas las alineaciones posibles entre la secuencia de entrada y la de etiquetas. El paper también evalúa la variante **RNN Transducer**, que agrega un modelo de lenguaje neuronal acoplado al acústico y permite modelar dependencias entre las salidas — algo que CTC no puede por su supuesto de independencia condicional.

Dos técnicas de regularización resultan críticas: **early stopping** y **ruido en los pesos** (*weight noise*, agregado una vez por secuencia de entrenamiento).

## Resultados

Sobre **TIMIT**, la tarea de reconocimiento de fonemas que era el banco de pruebas estándar:

| Configuración | PER |
|---|---|
| Deep Bidirectional LSTM + CTC | ~18 % |
| **Deep Bidirectional LSTM + RNN Transducer** | **17.7 %** |

El 17.7 % era el mejor resultado publicado sobre TIMIT en ese momento. Los hallazgos secundarios importan tanto como el número: la profundidad ayuda **más** que el tamaño de las capas (una red profunda y angosta gana a una somera y ancha con los mismos parámetros), y la bidireccionalidad aporta consistentemente.

## Limitaciones

- **TIMIT es pequeño y de fonemas.** 630 hablantes, habla leída, y la métrica es PER sobre fonemas, no [WER](/fundamentos/reconocimiento-de-voz) sobre palabras. La extrapolación a vocabulario grande no era evidente en 2013, y de hecho tardó un par de años en materializarse.
- **CTC asume independencia condicional entre salidas.** Cada predicción de frame es independiente dadas las entradas, lo que impide modelar dependencias lingüísticas dentro del propio decodificador. El RNN Transducer lo mitiga; los modelos con atención lo resuelven de otra forma.
- **Bidireccional implica offline.** Ver el futuro requiere el enunciado completo: no hay reconocimiento en vivo con esta arquitectura tal cual.
- **Costo secuencial.** Las capas recurrentes no se paralelizan sobre el tiempo, lo que limita el entrenamiento a escala. Es exactamente la restricción que [Deep Speech 2](/papers/deep-speech-2-amodei-2015) atacará con capas convolucionales y GRU, y que los Transformers eliminarán después.

## Por qué importa para la Clase 41

Este paper **es** el Ejemplo 1 de la clase. Cuando el material especifica:

> *Encoder: 4 stacked bidirectional-LSTM layers. Hidden state 256D. Head: character based output, softmax*

está describiendo esta arquitectura. La clase la usa como punto de partida y a continuación señala su problema —la **segmentación entrada-salida**, que las etiquetas de caracteres no vienen alineadas con los frames de audio— para introducir el token *blank* y CTC. La secuencia histórica es exactamente esa: Graves apila BiLSTM y las hace entrenables end-to-end **porque** CTC resuelve la alineación.

Su lugar en el linaje del [reconocimiento de voz](/fundamentos/reconocimiento-de-voz): es el primer eslabón enteramente neuronal del pipeline. Antes había DNN-HMM (red profunda dentro de un sistema estadístico); después vendrán los modelos con [atención](/papers/attention-asr-chorowski-2015) y encoder-decoder ([LAS](/papers/las-chan-2016)), que reemplazan CTC por un decodificador autorregresivo. Hoy los dos caminos conviven: [Whisper](/papers/whisper-radford-2022) usa encoder-decoder con atención, mientras que [wav2vec 2.0](/papers/wav2vec2-baevski-2020) hace fine-tuning con CTC.
