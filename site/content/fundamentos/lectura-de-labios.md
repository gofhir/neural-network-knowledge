---
title: "Lectura de Labios"
weight: 139
math: true
---

La **lectura de labios** —o *reconocimiento visual de habla* (VSR)— consiste en determinar qué se dijo a partir únicamente del movimiento de la boca, sin audio. Es una tarea con un techo de exactitud que no depende del modelo ni de los datos, sino de la física: **hay sonidos que se producen de forma invisible desde fuera**, y ninguna cámara los distingue.

Este fundamento acompaña a la [Clase 43](/clases/clase-43), donde el flujo visual de [E2E-AVSR](/papers/e2e-avsr-petridis-2018) resuelve exactamente este problema.

---

## 1. Visemas: por qué hay un techo

Un **fonema** es la unidad mínima de sonido; un **visema**, la unidad mínima visualmente distinguible. El mapa entre ambos no es biyectivo sino **muchos a uno**: varios fonemas comparten la misma configuración visible.

El caso canónico es `/p/`, `/b/` y `/m/`. Los tres son bilabiales: los labios se cierran y se abren igual. Lo que los separa es la sonoridad (vibración de las cuerdas vocales) y la nasalidad (paso de aire por la nariz) — **ninguna de las dos es visible desde afuera**. Lo mismo ocurre con `/k/`, `/g/` y `/ŋ/`, articulados en el velo del paladar, fuera del campo de visión.

{{< concept-alert type="clave" >}}
La consecuencia es un **techo de información**, no de capacidad. Ampliar el dataset o la red no ayuda: si dos palabras producen la misma secuencia de visemas, la señal para distinguirlas **no está en el video**.

Medido sobre un montaje sintético donde 40 clases se agrupan en 20 pares visualmente casi idénticos: la exactitud visual se estanca en **78,0 %**, mientras que la exactitud de "acertar el par" llega al **97,96 %**. El **90,7 % de los errores cae dentro del par ambiguo**. El modelo sabe perfectamente en qué grupo está; simplemente no puede elegir dentro de él.
{{< /concept-alert >}}

## 2. Los otros tres problemas

**Coarticulación.** La forma de la boca durante un fonema depende de los que vienen antes y después. En [LRW](/papers/e2e-avsr-petridis-2018) las palabras aparecen **en medio de un enunciado**, así que los bordes están contaminados por las palabras vecinas — a diferencia de los datasets de palabras aisladas, donde cada muestra empieza y termina en silencio.

**Homófonos visuales del vocabulario.** LRW incluye deliberadamente pares como *America* / *American* o singulares y plurales de la misma palabra. La diferencia está en el final, que suele ser justamente donde la coarticulación es peor.

**Variabilidad del hablante y de la toma.** Más de mil hablantes, con pose de cabeza e iluminación muy variables. La misma palabra se ve distinta según la persona y el ángulo.

## 3. La arquitectura estándar

Prácticamente todos los sistemas modernos comparten el mismo esqueleto de tres bloques, que es el que la [Clase 43](/clases/clase-43) presenta:

**1. Frente espacio-temporal.** Una convolución 3D —en E2E-AVSR, 64 núcleos de $5\times 7\times 7$ (tiempo × alto × ancho)— que captura la dinámica de corto plazo de la boca. Su presencia es lo que distingue a esta familia de un CNN por fotograma: aun cuando hay un recurrente detrás, el frente 3D mejora el resultado, porque el movimiento de milisegundos ya es informativo.

**2. Backbone espacial.** Una ResNet aplicada a cada paso temporal, que reduce progresivamente la dimensión espacial hasta un vector por fotograma. E2E-AVSR usa **ResNet-34 entrenada desde cero** — sin pesos de ImageNet, porque, según sus autores, están *"optimizados para tareas completamente distintas"*: imágenes estáticas a color contra recortes de boca en escala de grises.

**3. Modelo temporal.** Recurrentes bidireccionales (2 capas de BiGRU con 1024 celdas), o Transformers en los sistemas posteriores a 2020. La bidireccionalidad es viable porque la tarea es de reconocimiento diferido, no en vivo.

## 4. Los datasets

| Dataset | Año | Contenido | Escala |
|---|---|---|---|
| **GRID** | 2006 | oraciones con gramática fija, en estudio | 34 hablantes |
| **OuluVS / OuluVS2** | 2009-2015 | frases cortas, condiciones controladas | decenas de hablantes |
| **LRW** | 2016 | 500 palabras en contexto, de la BBC | 488 766 clips de entrenamiento |
| **LRS2 / LRS3** | 2017-2018 | oraciones completas, sin restricción | cientos de horas |

**LRW** (*Lip Reading in the Wild*, Chung y Zisserman) es el que usa la clase. Cada segmento dura exactamente **1,16 s (29 fotogramas)**, hay 500 palabras objetivo, entre 800 y 1000 secuencias por palabra en entrenamiento y 50 en validación y test — **488 766 / 25 000 / 25 000** ejemplos en total. El salto de escala respecto de los datasets previos (que tenían de 10 a 50 palabras) es lo que hizo viable entrenar redes profundas desde cero.

## 5. La progresión de resultados en LRW

| Sistema | Año | Exactitud |
|---|---|---|
| Chung y Zisserman | 2016 | 76,2 % |
| [Stafylakis y Tzimiropoulos](/papers/lipreading-resnet-stafylakis-2017) | 2017 | **83,0 %** |
| [E2E-AVSR](/papers/e2e-avsr-petridis-2018), flujo visual | 2018 | 82,0 % |

Vale detenerse en la tercera fila: el flujo visual de E2E-AVSR queda **por debajo** de la referencia de 2017. Los propios autores explican por qué —usan una caja fija de 96×96 para la boca, mientras que Stafylakis y Tzimiropoulos la extraen siguiendo puntos faciales— y no lo ocultan. La contribución de E2E-AVSR no está en el canal visual sino en la **fusión** y en trabajar sobre onda cruda.

Como referencia del techo humano: en tareas de lectura de labios sin audio, los lectores entrenados rara vez superan el 30-50 % de palabras en vocabulario abierto. Los sistemas de 2017 en adelante ya superan cómodamente a las personas en el escenario acotado de LRW.

## 6. De palabras aisladas a oraciones

Las limitaciones que E2E-AVSR declara —*"está limitado a un conjunto fijo de palabras aisladas"* y *"no generaliza bien a variaciones en el largo de la secuencia"*— señalan la frontera siguiente. Clasificar entre 500 palabras con 29 fotogramas fijos es una tarea cerrada; transcribir habla continua exige una salida de largo variable.

La salida es la misma que en [reconocimiento de voz](/fundamentos/reconocimiento-de-voz): **[CTC](/fundamentos/ctc-loss)** o *seq2seq* con atención. [LipNet](/papers/lipnet-assael-2016) (2016) fue el primero en llevar la lectura de labios a nivel de oración end-to-end con CTC, y [AV-HuBERT](/papers/av-hubert-shi-2022) (2022) es el estado del arte actual — con un resultado que resume el arco de toda la Clase 43: alcanza **32,5 % de WER en LRS3 con solo 30 horas etiquetadas**, superando a un sistema previo entrenado con **mil veces más** datos transcritos. La supervisión que faltaba no eran etiquetas; era la correspondencia audiovisual, que estaba gratis en el video.

---

## Ver también

- [E2E-AVSR (2018)](/papers/e2e-avsr-petridis-2018) · [Stafylakis y Tzimiropoulos (2017)](/papers/lipreading-resnet-stafylakis-2017) · [LipNet (2016)](/papers/lipnet-assael-2016) · [AV-HuBERT (2022)](/papers/av-hubert-shi-2022)
- [Aprendizaje Audiovisual](/fundamentos/aprendizaje-audiovisual) — por qué combinar con audio resuelve justamente lo que el video no puede.
- [Reconocimiento de Voz](/fundamentos/reconocimiento-de-voz) y [CTC Loss](/fundamentos/ctc-loss) — la maquinaria del paso a oraciones.
- [Análisis de Video](/fundamentos/analisis-de-video) · [Clase 43](/clases/clase-43)
