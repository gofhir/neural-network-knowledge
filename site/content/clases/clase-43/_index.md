---
title: "Clase 43 - Aplicaciones para Audio y Video"
weight: 430
sidebar:
  open: true
---

**Profesores:** Alain Raymond, Gabriel Sepúlveda y Álvaro Soto (IALab, Departamento de Ciencia de la Computación, PUC)
**Módulo:** Audio y Video — donde confluyen los dos hilos

Esta es la clase donde se juntan los dos recorridos que venían alternándose en el diplomado: el de video ([36](/clases/clase-36), [38](/clases/clase-38), [40](/clases/clase-40), [42](/clases/clase-42)) y el de audio ([35](/clases/clase-35), [37](/clases/clase-37), [39](/clases/clase-39), [41](/clases/clase-41)). Son dos papers presentados uno tras otro, y lo que los une es una propiedad del video tan obvia que resulta fácil pasarla por alto: **la imagen y el sonido llegan sincronizados y describen la misma escena**. Nadie tuvo que anotar esa correspondencia — viene en el archivo.

Lo interesante es que cada paper la usa para algo distinto, y en momentos distintos.

{{< concept-alert type="clave" >}}
**[SoundNet](/papers/soundnet-aytar-2016) (2016) la usa para entrenar.** El problema es la falta de datasets de audio etiquetados —ESC-50 tiene 2000 clips; DCASE, diez ejemplos por categoría—. La solución no es conseguir etiquetas sino conseguir un **maestro**: una red visual ya entrenada clasifica los fotogramas de dos millones de videos de Flickr, y una CNN sobre la onda cruda aprende a reproducir esas distribuciones. En inferencia, la red no necesita ver nada.

**[E2E-AVSR](/papers/e2e-avsr-petridis-2018) (2018) las usa para decidir.** Dos flujos —boca y forma de onda— fusionados por una BiGRU para reconocer palabras. Su resultado en audio limpio es deliberadamente modesto: 98,0 % contra 97,7 % del audio solo. Bajo ruido, **+14,1 puntos**. La razón es que el ruido acústico no toca al canal visual, y la figura que lo muestra —una línea horizontal— es el argumento entero del paper.

**Lo que la clase no nombra** es que el primer mecanismo tiene nombre propio: [destilación de conocimiento](/fundamentos/destilacion-de-conocimiento). Ponérselo permite leer su ablación más llamativa —25 puntos entre la pérdida KL y la $\ell_2$— con las herramientas de [Hinton (2015)](/papers/distillation-hinton-2015), y descubrir que la brecha no mide lo que parece.
{{< /concept-alert >}}

## Apuntes de clase

{{< cards >}}
  {{< card link="teoria" title="Teoria" subtitle="Las 49 diapositivas: la falta de etiquetas de audio y el maestro visual, el objetivo KL, la ablación de profundidad que prueba la tesis, y después los dos flujos, LRW, el currículo de cinco etapas y la curva de ruido" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="La aritmética de SoundNet capa por capa —campo receptivo y dónde está el 80 % del modelo—, por qué 25 puntos entre KL y L2 contradicen un teorema y qué los explica, la estructura de la complementariedad, y aprender contra diseñar la representación" icon="beaker" >}}
  {{< card link="practica" title="Practica desde 0" subtitle="Destilación con temperatura y el teorema del límite alto verificado; la curva de SNR reconstruida con el techo de los visemas y la fusión que empeora — en triple framework" icon="code" >}}
  {{< card link="/laboratorios/lab-43" title="Laboratorio 43" subtitle="El práctico: E2E-AVSR ejecutado sobre las 500 palabras de LRW. Reproduce el paper con 98,84 % y los 29 errores resultan ser todos vecinos fonológicos —THERE→THEIR es irreducible por homofonía—; más el recorte de 19456 muestras descifrado como 29 × 672 y el 39 % del ZIP de pesos que es el backend descartado de la fase 1" icon="beaker" >}}
  {{< card link="/clases/clase-41" title="Clase anterior: Speech y Speaker Recognition" subtitle="El hilo de audio que llega hasta acá: CTC, alineación y descriptores de hablante" icon="adjustments" >}}
  {{< card link="/clases/clase-42" title="Clase anterior: Tracking de Objetos" subtitle="El hilo de video que llega hasta acá: asociación, identidad y oclusión" icon="eye" >}}
  {{< card link="/clases/clase-39" title="Relacionada: Modelos de DL para Audio" subtitle="CNN sobre onda cruda para sonidos ambientales — el terreno donde SoundNet compite" icon="adjustments" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/destilacion-de-conocimiento" title="Destilación de Conocimiento" subtitle="Dark knowledge, temperatura, el teorema del límite alto, y las dos cosas distintas que se llaman L2" icon="book-open" >}}
  {{< card link="/fundamentos/aprendizaje-audiovisual" title="Aprendizaje Audiovisual" subtitle="La sincronía como etiqueta gratis, las tres formas de explotarla, dónde fusionar y cuándo la fusión estorba" icon="book-open" >}}
  {{< card link="/fundamentos/lectura-de-labios" title="Lectura de Labios" subtitle="Visemas y el techo de información, la arquitectura de tres bloques, LRW y el paso a oraciones" icon="book-open" >}}
  {{< card link="/fundamentos/clasificacion-de-audio" title="Clasificación de Audio" subtitle="La tarea que SoundNet resuelve, y los datasets que la limitaban" icon="book-open" >}}
  {{< card link="/fundamentos/representacion-de-audio" title="Representación de Audio" subtitle="Onda cruda contra MFCC: el intercambio que la tabla del segundo paper resuelve de forma inesperada" icon="book-open" >}}
  {{< card link="/fundamentos/aprendizaje-autosupervisado" title="Aprendizaje Autosupervisado" subtitle="La familia mayor: extraer supervisión de la estructura de los datos" icon="book-open" >}}
  {{< card link="/fundamentos/transfer-learning" title="Transfer Learning" subtitle="La alternativa clásica — copiar pesos en vez de imitar salidas" icon="book-open" >}}
  {{< card link="/fundamentos/lstm-gru" title="LSTM y GRU" subtitle="Las BiGRU de dos capas que modelan la dinámica de cada flujo y la fusión" icon="book-open" >}}
{{< /cards >}}

## Papers de esta clase

### Los dos papers presentados

{{< cards >}}
  {{< card link="/papers/soundnet-aytar-2016" title="SoundNet (2016)" subtitle="Aytar, Vondrick y Torralba — dos millones de videos de Flickr, un maestro visual y una CNN 1D sobre onda cruda. 88 % en DCASE. El 80 % de sus parámetros se descarta al usarla" icon="document-text" >}}
  {{< card link="/papers/e2e-avsr-petridis-2018" title="E2E Audiovisual Speech Recognition (2018)" subtitle="Petridis et al. — boca y onda cruda fusionadas por BiGRU sobre LRW. +0,3 puntos en limpio, +14,1 a −5 dB" icon="document-text" >}}
{{< /cards >}}

### El marco que la clase no nombra

{{< cards >}}
  {{< card link="/papers/distillation-hinton-2015" title="Destilación (2015)" subtitle="Hinton, Vinyals y Dean — soft targets, temperatura, y el teorema de que a T alta destilar equivale a mínimos cuadrados sobre logits" icon="document-text" >}}
  {{< card link="/papers/look-listen-learn-arandjelovic-2017" title="Look, Listen and Learn (2017)" subtitle="Arandjelović y Zisserman — la alternativa simétrica: correspondencia audiovisual sin maestro preentrenado" icon="document-text" >}}
  {{< card link="/papers/objects-that-sound-arandjelovic-2018" title="Objects that Sound (2018)" subtitle="La continuación: localizar en el cuadro el objeto que produce el sonido" icon="document-text" >}}
{{< /cards >}}

### La línea de lectura de labios

{{< cards >}}
  {{< card link="/papers/lipreading-resnet-stafylakis-2017" title="ResNet + LSTM para lipreading (2017)" subtitle="Stafylakis y Tzimiropoulos — la arquitectura de tres bloques que E2E-AVSR adopta, y la única línea de su tabla que no logra superar (83,0 contra 82,0)" icon="document-text" >}}
  {{< card link="/papers/lipnet-assael-2016" title="LipNet (2016)" subtitle="Assael et al. — el primero a nivel de oración con CTC: la salida de largo variable que E2E-AVSR declara como limitación pendiente" icon="document-text" >}}
  {{< card link="/papers/av-hubert-shi-2022" title="AV-HuBERT (2022)" subtitle="Shi et al. — 32,5 % de WER en LRS3 con 30 horas etiquetadas, contra un sistema entrenado con 31 000. Cierra el arco que abre SoundNet" icon="document-text" >}}
{{< /cards >}}

### Datasets y contexto

{{< cards >}}
  {{< card link="/papers/esc50-piczak-2015" title="ESC-50 (2015)" subtitle="Piczak — 2000 clips, 50 categorías, y el 81,3 % de desempeño humano que SoundNet casi alcanza" icon="document-text" >}}
  {{< card link="/papers/vggish-hershey-2017" title="VGGish (2017)" subtitle="Hershey et al. — la ruta alternativa: espectrogramas y etiquetas masivas en vez de onda cruda y destilación" icon="document-text" >}}
{{< /cards >}}

---

**Ver también:** [Clase 39 - Modelos de DL para Audio](/clases/clase-39) · [Clase 41 - Speech y Speaker Recognition](/clases/clase-41) · [Clase 42 - Tracking](/clases/clase-42) · [Clase 28 - Aprendizaje Autosupervisado](/clases/clase-28) · Dominios [Audio](/dominios/audio), [Video](/dominios/video) y [Multimodal](/dominios/multimodal).
