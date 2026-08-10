---
title: "Clase 39 - Modelos de Deep Learning para Audio"
weight: 390
sidebar:
  open: true
---

**Profesor:** Gabriel Sepúlveda (DCC PUC)
**Módulo:** Audio y Video (Audio 3 de 5)

Tercera clase del hilo de audio, y la que responde la pregunta que las dos anteriores dejaron abierta. La [Clase 35](/clases/clase-35) cubrió la naturaleza de la señal y la [Clase 37](/clases/clase-37) el ciclo de vida del dato; acá se define **qué modelo se pone encima del espectrograma, y qué pasa si se prescinde del espectrograma**.

La clase tiene dos mitades y una coda. La primera dice: el espectrograma es una imagen, usa una CNN 2D, y combínala con una RNN porque cada operador aporta algo que el otro no. La segunda dice: también puedes trabajar directamente sobre la onda cruda, pero entonces el problema pasa a ser el **campo receptivo**, y la respuesta son las **convoluciones dilatadas**. Las dos mitades son la misma pregunta —cómo cubrir suficiente contexto temporal sin que la red explote— resuelta con presupuestos distintos.

La coda descarta los Transformers en audio con tres argumentos. Es la parte que más ha envejecido, y la que este material audita con la evidencia de 2020-2022.

## Apuntes de clase

{{< cards >}}
  {{< card link="teoria" title="Teoria" subtitle="Recorrido de las diapositivas: taxonomía de tareas, el espectrograma como imagen, la receta CNN+RNN+MLP, la onda cruda y la dilatación, y la auditoría de la sección sobre Transformers" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="Math: la aritmética del campo receptivo y la condición que evita el gridding, la contabilidad de la CLDNN, por qué un espectrograma no es una imagen, y el costo comparado de convolución, recurrencia y atención" icon="beaker" >}}
  {{< card link="practica" title="Practica desde 0" subtitle="Medir el campo receptivo por gradiente y construir la CLDNN del Ejemplo 1 — en PyTorch, TensorFlow y JAX, con salidas reales" icon="code" >}}
  {{< card link="/clases/clase-37" title="Clase anterior de audio: Datasets y Herramientas" subtitle="El ciclo de vida del dato: formatos, librerías, augmentation y datasets" icon="arrow-left" >}}
  {{< card link="/clases/clase-35" title="Relacionada: Análisis de Audio" subtitle="La teoría de señales — Fourier, muestreo, STFT y MFCC" icon="academic-cap" >}}
  {{< card link="/clases/clase-41" title="Clase siguiente de audio: Speech y Speaker" subtitle="Reconocimiento de voz e identificación de hablante" icon="arrow-right" >}}
{{< /cards >}}

## Los dos hallazgos de esta clase

{{< concept-alert type="clave" >}}
**1. El "Ejemplo 2" no cumple la promesa del slide anterior.** La clase afirma que las convoluciones dilatadas permiten cubrir "miles de timesteps tras pocas capas", y propone una arquitectura de 4 capas con kernels $20, 10, 10, 5$. Con la progresión de dilataciones canónica —$1, 2, 4, 8$, la de la figura del slide 55— esas capas cubren **106 muestras: 6.6 milisegundos**. Ni siquiera una ventana de análisis estándar.

El principio es correcto; el ejemplo no lo instancia. La progresión que sí funciona con esos kernels es $1, 20, 200, 2000$, que da **10.000 muestras (625 ms) sin un solo hueco** — y sale de una condición general, $d_{l+1} \le R_l$, que además explica por qué la duplicación clásica es el óptimo para kernel 2 y un desperdicio para kernels grandes. Medido y verificado en la [práctica](practica/01-campo-receptivo-y-dilatacion).

**2. La sección sobre Transformers está fechada, y una de sus tres objeciones es directamente falsa.** El slide sostiene que "los Transformers no son buenos para modelar dependencias largas", que es lo contrario de la motivación original de la self-attention: el camino entre dos posiciones es $O(1)$ frente a $O(T)$ en una RNN. El problema real es el costo cuadrático, no la capacidad. En abril de 2024, fecha del PDF, wav2vec 2.0, Conformer, AST, HuBERT y Whisper ya llevaban años siendo el estado del arte.

Lo que sí conviene rescatar: la tesis de que un modelo de audio necesita **un operador local y uno global** es correcta y sigue vigente. Lo que cambió es cuál es el mejor operador global.
{{< /concept-alert >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/convoluciones-dilatadas" title="Convoluciones dilatadas" subtitle="El operador, la aritmética del campo receptivo, la condición que evita el gridding y cuándo NO usarlo" icon="book-open" >}}
  {{< card link="/fundamentos/crnn" title="CRNN: arquitecturas convolucional-recurrentes" subtitle="La tesis de la complementariedad, la interfaz donde vive el costo, y qué parte de la receta sobrevivió" icon="book-open" >}}
  {{< card link="/fundamentos/clasificacion-de-audio" title="Clasificación de audio" subtitle="Tagging vs detección de eventos, etiquetas fuertes y débiles, softmax vs sigmoides, y las trampas de evaluación" icon="book-open" >}}
  {{< card link="/fundamentos/representacion-de-audio" title="Representación de audio" subtitle="Del archivo al tensor: formatos, transforms y batching" icon="book-open" >}}
  {{< card link="/fundamentos/mfcc-y-escala-mel" title="MFCC y escala Mel" subtitle="Por qué el eje de frecuencia viene deformado por diseño" icon="book-open" >}}
  {{< card link="/fundamentos/lstm-gru" title="LSTM y GRU" subtitle="El bloque recurrente de la receta" icon="book-open" >}}
  {{< card link="/fundamentos/self-attention" title="Self-attention" subtitle="El operador que reemplazó a la recurrencia, y el argumento de la longitud del camino" icon="book-open" >}}
  {{< card link="/fundamentos/data-augmentation-de-audio" title="Data augmentation de audio" subtitle="SNR, SpecAugment, pitch y time stretching — y cuáles destruyen la etiqueta" icon="book-open" >}}
{{< /cards >}}

## Papers de esta clase

### Las dos arquitecturas del profesor

{{< cards >}}
  {{< card link="/papers/cldnn-sainath-2015" title="CLDNN (2015)" subtitle="Sainath et al. — el 'Ejemplo 1' capa por capa, con tres números que el slide cambió" icon="document-text" >}}
  {{< card link="/papers/wavenet-oord-2016" title="WaveNet (2016)" subtitle="van den Oord et al. — el origen de la convolución dilatada en audio y la figura del slide 55" icon="document-text" >}}
{{< /cards >}}

### La onda cruda y los modelos pre-entrenados (los del laboratorio)

{{< cards >}}
  {{< card link="/papers/raw-waveforms-dai-2017" title="Familia M / Very Deep CNN (2017)" subtitle="Dai et al. — llegar a 1.5 s de campo receptivo con stride y pooling, sin dilatación" icon="document-text" >}}
  {{< card link="/papers/vggish-hershey-2017" title="VGGish (2017)" subtitle="Hershey et al. — la validación a gran escala de tratar el espectrograma como imagen" icon="document-text" >}}
  {{< card link="/papers/audioset-gemmeke-2017" title="AudioSet (2017)" subtitle="Gemmeke et al. — la ontología y el dataset sobre el que se pre-entrena VGGish" icon="document-text" >}}
  {{< card link="/papers/urbansound8k-salamon-2014" title="UrbanSound8K (2014)" subtitle="Salamon et al. — el dataset del práctico, y sus 10 folds que no hay que re-barajar" icon="document-text" >}}
{{< /cards >}}

### El contrapunto: lo que pasó con los Transformers en audio

{{< cards >}}
  {{< card link="/papers/conformer-gulati-2020" title="Conformer (2020)" subtitle="Gulati et al. — acepta la tesis de la complementariedad y solo cambia el operador global" icon="document-text" >}}
  {{< card link="/papers/ast-gong-2021" title="AST (2021)" subtitle="Gong et al. — parches de espectrograma: la tokenización no necesita ser semántica" icon="document-text" >}}
  {{< card link="/papers/hubert-hsu-2021" title="HuBERT (2021)" subtitle="Hsu et al. — fabricar las unidades discretas que supuestamente faltaban" icon="document-text" >}}
  {{< card link="/papers/wav2vec2-baevski-2020" title="wav2vec 2.0 (2020)" subtitle="Baevski et al. — el SSL que disolvió el problema de los datasets etiquetados" icon="document-text" >}}
  {{< card link="/papers/whisper-radford-2022" title="Whisper (2022)" subtitle="Radford et al. — 680k horas de supervisión débil sobre un Transformer puro" icon="document-text" >}}
{{< /cards >}}

### El resto de las referencias del slide final

{{< cards >}}
  {{< card link="/papers/dl-audio-purwins-2019" title="Deep Learning for Audio Signal Processing (2019)" subtitle="Purwins et al. — el survey del que se estructuró la clase entera" icon="document-text" >}}
  {{< card link="/papers/musicnn-pons-2019" title="musicnn (2019)" subtitle="Pons y Serra — por qué en música el filtro no debería ser cuadrado" icon="document-text" >}}
  {{< card link="/papers/sv2tts-jia-2018" title="SV2TTS (2018)" subtitle="Jia et al. — síntesis de voz como augmentation, y transferencia entre tareas de audio" icon="document-text" >}}
  {{< card link="/papers/scaper-salamon-2017" title="Scaper (2017)" subtitle="Salamon et al. — sintetizar paisajes sonoros para obtener anotaciones exactas" icon="document-text" >}}
{{< /cards >}}

## Dominio relacionado

{{< cards >}}
  {{< card link="/dominios/audio" title="Dominio: Audio / Voz" subtitle="Línea de tiempo: de MFCC y HMM-GMM a wav2vec, Whisper y los foundation models de audio" icon="globe-alt" >}}
{{< /cards >}}
