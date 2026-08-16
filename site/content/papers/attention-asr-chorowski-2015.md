---
title: "Attention-Based Models for Speech Recognition (2015)"
weight: 436
math: true
---

{{< paper-card
    title="Attention-Based Models for Speech Recognition"
    authors="Jan Chorowski, Dzmitry Bahdanau, Dmitriy Serdyuk, Kyunghyun Cho, Yoshua Bengio"
    year="2015"
    venue="NeurIPS 2015 / arXiv:1506.07503"
    pdf="/papers/attention-asr-chorowski-2015.pdf" >}}
El primer trabajo que traslada seriamente la [atención de Bahdanau](/papers/bahdanau-attention-2015) —diseñada para traducción— al reconocimiento de voz, y el primero en documentar por qué **no funciona directamente**. Aplicada tal cual, la atención alcanza un competitivo **18.7 % de PER en TIMIT**, pero solo sobre enunciados de longitud parecida a los de entrenamiento: con secuencias más largas se rompe. El diagnóstico es que la atención de traducción es puramente **basada en contenido**, y el habla contiene fragmentos acústicamente casi idénticos repartidos por todo el enunciado —silencios, la misma vocal dicha tres veces— que confunden al mecanismo. La solución es agregarle **conciencia de ubicación**: alimentar la atención con dónde miró en el paso anterior. Con eso baja a **18 % de PER** y sobrevive a enunciados 10 veces más largos; con un ajuste extra que evita concentrarse demasiado en un solo frame, llega a **17.6 %**.
{{< /paper-card >}}

---

## Contexto: por qué el habla no es traducción

En 2015 el esquema [seq2seq con atención](/papers/bahdanau-attention-2015) dominaba la traducción automática, y el paralelo con el reconocimiento de voz parecía directo: una secuencia entra, otra sale, y el decodificador elige a qué parte de la entrada mirar en cada paso. La [Clase 41](/clases/clase-41) plantea exactamente esa analogía en su slide *"Can we apply a Seq2Seq+Att model to speech recognition?"*.

Este paper es la respuesta detallada, y empieza señalando tres diferencias que la analogía esconde:

| | Traducción | Reconocimiento de voz |
|---|---|---|
| Longitud de la entrada | decenas de tokens | **cientos o miles** de frames |
| Relación entrada-salida | reordenamiento libre | **monótona**: el tiempo no retrocede |
| Ambigüedad local | palabras distinguibles | fragmentos acústicamente **idénticos** repetidos |

La tercera es la que rompe el mecanismo. La atención de Bahdanau puntúa cada posición de la entrada por su **contenido**: $e_{t,i} = \text{score}(s_{t-1}, h_i)$. Si el enunciado contiene tres silencios acústicamente indistinguibles, los tres reciben la misma puntuación y el decodificador no tiene forma de saber cuál es el que corresponde *ahora*.

## Método: atención consciente de la ubicación

La contribución técnica es agregar al puntaje de atención un término que depende de **dónde miró el paso anterior**:

$$e_{t,i} = w^\top \tanh\big(W s_{t-1} + V h_i + U f_{t,i} + b\big)$$

donde $f_{t} = F * \alpha_{t-1}$ son *features de ubicación*: la distribución de atención del paso previo, convolucionada con un banco de filtros aprendidos $F$. Esos filtros pueden aprender patrones como "el siguiente frame relevante suele estar un poco a la derecha del anterior", que es precisamente la estructura monótona del habla.

El segundo aporte es un **ajuste de nitidez** de la distribución. La atención sobre secuencias muy largas tiende a dispersarse (el softmax sobre cientos de posiciones aplana), y el paper prueba dos correcciones: una temperatura sobre el softmax y una variante que restringe la atención a una ventana alrededor de la posición previa. La segunda además abarata el cómputo: no hace falta puntuar los mil frames en cada paso.

## Resultados

Sobre TIMIT, reconocimiento de fonemas:

| Modelo | PER (enunciado normal) | PER (enunciado 10× más largo) |
|---|---|---|
| Atención por contenido (estilo traducción) | 18.7 % | **se degrada severamente** |
| + conciencia de ubicación | 18.0 % | 20 % |
| + control de nitidez | **17.6 %** | — |

El resultado que importa no es el 17.6 % —comparable al 17.7 % de [Graves con CTC](/papers/deep-rnn-speech-graves-2013)— sino la **columna de la derecha**: sin location-awareness el modelo no generaliza a longitudes distintas de las vistas en entrenamiento, y eso es descalificatorio para un sistema real, donde los enunciados duran lo que duran.

## Limitaciones

- **TIMIT sigue siendo pequeño.** Reconocimiento de fonemas sobre habla leída, no vocabulario grande en condiciones reales. La extensión a LVCSR es el trabajo siguiente del mismo grupo, [Bahdanau et al. (2016)](/papers/e2e-lvsr-bahdanau-2016).
- **El costo de la atención es cuadrático** en la longitud de la entrada: cada paso de decodificación puntúa todos los frames. Con cientos de frames por segundo, eso es caro — el problema que motiva tanto el pooling temporal como el encoder piramidal de [LAS](/papers/las-chan-2016).
- **Sin garantía de monotonía.** El mecanismo *aprende* a avanzar, pero nada se lo impone: la atención puede saltar hacia atrás y producir repeticiones o saltos en la transcripción. Las familias posteriores de atención monótona atacan justamente eso.

## Por qué importa para la Clase 41

La clase construye su Ejemplo 1 sobre la analogía seq2seq+atención → habla, y este paper es el que documenta la letra chica de esa analogía. Tres puntos que la clase no menciona y que conviene tener presentes:

1. **La atención por contenido sola no basta** para el habla, por la repetición acústica.
2. **El costo crece con la longitud de la entrada**, lo que explica por qué el Ejemplo 2 de la clase agrega *pooling* temporal: no es solo eficiencia, es viabilidad.
3. **La estructura monótona del habla es información gratis** que la atención de traducción desaprovecha por diseño.

En el linaje del [reconocimiento de voz](/fundamentos/reconocimiento-de-voz), este paper abre la rama de atención que corre en paralelo a la de [CTC](/papers/ctc-graves-2006): dos respuestas distintas al mismo problema de alineación. CTC marginaliza sobre todas las alineaciones posibles y asume independencia entre salidas; la atención aprende la alineación explícitamente y modela dependencias entre caracteres. [LAS](/papers/las-chan-2016) lleva esta rama a vocabulario abierto y [Whisper](/papers/whisper-radford-2022) la lleva a escala industrial con Transformers.
