---
title: "LipNet: lectura de labios a nivel de oración (2016)"
weight: 462
math: true
---

{{< paper-card
    title="LipNet: End-to-End Sentence-level Lipreading"
    authors="Yannis M. Assael, Brendan Shillingford, Shimon Whiteson, Nando de Freitas (Universidad de Oxford / DeepMind)"
    year="2016"
    venue="arXiv:1611.01599"
    arxiv="1611.01599"
    pdf="/papers/lipnet-assael-2016.pdf" >}}
El primer modelo de lectura de labios que predice **oraciones completas** end-to-end, en vez de clasificar palabras de un vocabulario cerrado. Combina convoluciones espacio-temporales, redes recurrentes y la pérdida [CTC](/fundamentos/ctc-loss), y opera a nivel de **secuencia de caracteres**, con lo que la salida deja de estar acotada a un conjunto fijo. Sobre GRID alcanza **95,2 % de exactitud de palabra** en la partición con hablantes solapados y **88,6 %** con hablantes no vistos. La comparación más elocuente del paper: lectores humanos entrenados obtienen, en la misma tarea, un **52,3 %**.
{{< /paper-card >}}

---

## Qué cambia respecto de clasificar palabras

Los sistemas de lectura de labios anteriores —y el de la [Clase 43](/clases/clase-43)— resuelven **clasificación**: dada una ventana fija de fotogramas, elegir entre $N$ palabras. Es una tarea cerrada, con un softmax de tamaño fijo, y no puede producir nada fuera de su vocabulario.

LipNet plantea **predicción de secuencia**: la entrada es un video de largo variable y la salida una secuencia de caracteres de largo variable, sin alineación conocida entre ambas. Ese es exactamente el problema que resuelve CTC, sumando sobre todas las alineaciones posibles mediante un símbolo *blank* — la misma maquinaria que la [Clase 41](/clases/clase-41) desarrolla para el reconocimiento de voz, aplicada aquí a píxeles.

{{< concept-alert type="clave" >}}
El salto de clasificación a secuencia es el que separa un demostrador de un sistema utilizable. Un clasificador de 500 palabras necesita re-entrenarse para agregar una palabra; un modelo con CTC sobre caracteres puede producir cualquier cadena, incluidas palabras que nunca vio.

Es la respuesta directa a las tres limitaciones que [E2E-AVSR](/papers/e2e-avsr-petridis-2018) declara al cerrar: vocabulario fijo, palabras aisladas, y mala generalización a largos de secuencia distintos.
{{< /concept-alert >}}

## Resultados y su matiz

95,2 % en GRID con hablantes solapados, 88,6 % con hablantes no vistos, contra 52,3 % de lectores humanos entrenados sobre la misma tarea.

El matiz importante es **GRID**: es un corpus de estudio, con gramática fija —oraciones de la forma *comando + color + preposición + letra + dígito + adverbio*— y 34 hablantes en condiciones controladas. Comparado con LRW (más de 1000 hablantes de la BBC, pose e iluminación libres), es un entorno mucho más benigno, y parte del 95,2 % lo aporta la estructura sintáctica rígida, no la lectura de labios.

Que un método sea *sentence-level* y otro alcance mejor exactitud sobre palabras aisladas no los pone en competencia: miden cosas distintas sobre datos distintos.

## Por qué importa para la Clase 43

Marca la dirección hacia donde la clase apunta sin recorrerla. E2E-AVSR (2018) es posterior a LipNet (2016) y sin embargo resuelve una tarea más restringida — porque atacó otro eje: la **fusión audiovisual sobre datos no controlados**. LipNet resolvió la salida de largo variable, pero solo con video y sobre un corpus de estudio.

La síntesis de ambas líneas —secuencias abiertas, fusión audiovisual, datos en libertad— llega con [AV-HuBERT](/papers/av-hubert-shi-2022) en 2022.

---

**Ver también:** [E2E-AVSR (2018)](/papers/e2e-avsr-petridis-2018) · [Stafylakis y Tzimiropoulos (2017)](/papers/lipreading-resnet-stafylakis-2017) · [AV-HuBERT (2022)](/papers/av-hubert-shi-2022) · [CTC (2006)](/papers/ctc-graves-2006) · [Lectura de Labios](/fundamentos/lectura-de-labios) · [CTC Loss](/fundamentos/ctc-loss)
