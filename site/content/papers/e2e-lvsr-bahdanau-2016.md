---
title: "End-to-End Attention-based LVSR (2016)"
weight: 438
math: true
---

{{< paper-card
    title="End-to-End Attention-based Large Vocabulary Speech Recognition"
    authors="Dzmitry Bahdanau, Jan Chorowski, Dmitriy Serdyuk, Philémon Brakel, Yoshua Bengio"
    year="2016"
    venue="ICASSP 2016 / arXiv:1508.04395"
    pdf="/papers/e2e-lvsr-bahdanau-2016.pdf" >}}
La continuación directa de [Chorowski et al. (2015)](/papers/attention-asr-chorowski-2015): llevar el reconocimiento con atención de los fonemas de TIMIT al **vocabulario grande** del Wall Street Journal, reemplazando el HMM por un generador de secuencias recurrente que predice **caracteres** directamente. El obstáculo práctico es el costo: en cada carácter emitido, la atención debe recorrer todos los frames del enunciado. El paper propone dos formas de abaratarlo —**limitar el barrido** a un subconjunto de frames prometedores y hacer **pooling sobre el tiempo** para acortar la secuencia fuente— y esa segunda es la que la [Clase 41](/clases/clase-41) cita literalmente como alternativa a [CTC](/papers/ctc-graves-2006) para lidiar con la desalineación. Integrando un modelo de lenguaje n-grama en la decodificación, alcanza precisiones comparables a otros sistemas RNN sin HMM.
{{< /paper-card >}}

---

## Contexto: sacar el HMM del sistema, no solo del entrenamiento

Los sistemas LVCSR punteros de 2015 eran **híbridos**: una red neuronal para el modelado acústico, dentro de un armazón de HMM que se encargaba de la alineación y la decodificación, más un modelo de lenguaje y un léxico. La objeción de los autores es precisa: *"the acoustic model is not directly trained to minimize the final objective of interest"* — cada componente optimiza su propia función, ninguna la métrica que importa.

Ya había alternativas en marcha. [CTC](/papers/ctc-graves-2006) había mostrado buenos resultados sobre Wall Street Journal y estado del arte en Switchboard, prediciendo secuencias de caracteres que después se combinaban con un modelo de lenguaje de palabras. Este paper explora la **otra** vía: reemplazar el HMM por un generador de secuencias con atención (ARSG), que aprende la alineación como parte del modelo en lugar de marginalizarla.

## Método: la atención como cuello de botella, y cómo abrirlo

El modelo es un encoder-decoder con atención sobre caracteres. El problema que domina el diseño es el **costo del mecanismo de atención**: para cada carácter emitido hay que puntuar todos los frames de entrada. Un enunciado de 10 segundos a 100 frames/segundo son 1 000 posiciones, y una transcripción de 100 caracteres implica 100 000 evaluaciones del puntaje. Dos remedios:

**Barrido limitado.** En lugar de puntuar todos los frames, restringir el cálculo a un subconjunto de candidatos prometedores — típicamente una ventana alrededor de donde la atención estuvo en el paso anterior. Aprovecha la **monotonía** del habla: el siguiente carácter casi nunca corresponde a un frame muy anterior.

**Pooling sobre el tiempo.** Agregar información de frames vecinos en el encoder, reduciendo la longitud de la secuencia fuente antes de que la atención la vea. Es la misma idea que la pirámide de [LAS](/papers/las-chan-2016): si el encoder entrega la mitad de los vectores, la atención hace la mitad del trabajo — y, dado que los frames de audio contiguos son enormemente redundantes, se pierde poco.

**Integración del modelo de lenguaje.** Un n-grama a nivel de palabras participa en la decodificación por haces, lo que corrige errores ortográficos y de palabras raras que el modelo de caracteres comete por su cuenta.

## Resultados

Sobre Wall Street Journal, con la integración del modelo de lenguaje en la decodificación, el sistema alcanza precisiones **comparables a los demás enfoques RNN sin HMM** de la época — sin igualar todavía a los híbridos DNN-HMM más maduros.

El valor del paper no está en el número sino en la demostración de viabilidad: un modelo de atención puro, sin HMM ni alineación forzada, escala de fonemas a vocabulario grande. Y en el diagnóstico de qué lo hacía inviable: el costo del mecanismo de atención sobre secuencias largas.

## Limitaciones

- **No alcanza a los híbridos** en el WSJ de 2016.
- **Sigue necesitando un modelo de lenguaje externo** para resultados competitivos, lo que relativiza el "end-to-end" del título.
- **El pooling temporal pierde resolución.** Acortar la secuencia fuente ayuda a la atención pero descarta detalle acústico; el factor de reducción es un hiperparámetro con un óptimo que depende de la tarea.
- **El barrido limitado introduce un supuesto de monotonía** que casi siempre se cumple en habla, pero cuando falla el error es catastrófico: la atención no puede recuperar una posición que descartó.

## Por qué importa para la Clase 41

Es la referencia que la clase cita textualmente:

> *"Methods such as Connectionist Temporal Classification or CTC (Graves et al., 2006) or **Pooling Over Time (Bahdanau et al., 2016)** help to deal with the misalignment issue."*

Conviene notar que las dos soluciones que la slide pone en paralelo **no son del mismo tipo**, y esa distinción es el aporte de leer el paper:

| | CTC | Pooling over time |
|---|---|---|
| Qué resuelve | la **alineación**: marginaliza sobre todas las posibles | el **costo**: acorta la secuencia que la atención debe recorrer |
| Cómo | token *blank* + suma sobre alineaciones | agregación de frames vecinos en el encoder |
| Es alternativa a… | la atención, como forma de alinear | nada — es complementario, se usa *con* atención |

CTC es un objetivo de entrenamiento que hace innecesaria la alineación explícita. El pooling temporal es una modificación arquitectónica que hace tratable la atención. Se pueden combinar, y de hecho los sistemas modernos suelen hacerlo: subsampling convolucional en el encoder más una pérdida híbrida CTC-atención.

Para el panorama completo, ver el fundamento [Reconocimiento de voz](/fundamentos/reconocimiento-de-voz).
