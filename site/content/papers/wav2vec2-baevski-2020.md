---
title: "wav2vec 2.0: SSL de representaciones de voz (2020)"
weight: 408
math: true
---

{{< paper-card
    title="wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"
    authors="Alexei Baevski et al. (Facebook AI)"
    year="2020"
    venue="NeurIPS 2020 / arXiv:2006.11477"
    pdf="/papers/wav2vec2-baevski-2020.pdf" >}}
wav2vec 2.0 demuestra, según los autores por primera vez, que aprender representaciones potentes a partir de **audio de voz solo** (sin transcripciones) y luego hacer fine-tuning ligero puede **superar a los mejores métodos semi-supervisados**. El modelo encadena un **encoder convolucional** sobre la onda cruda, una **cuantización** aprendida, un **Transformer** de contexto, un **enmascaramiento estilo BERT** en el espacio latente y una **pérdida contrastiva**. El hallazgo disruptivo es el régimen de bajos recursos: con **apenas 10 minutos** de audio etiquetado más pre-training sobre **53.000 horas** sin etiquetar, logra **4,8/8,2 WER** (test clean/other); con 1 hora ya supera el estado del arte previo que usaba 100 horas. Es el ejemplo canónico de los **embeddings preentrenados** como representación del audio en la [Clase 37](/clases/clase-37).
{{< /paper-card >}}

---

## Contexto: el ASR supervisado era caro y el SSL llegó al audio

Las redes neuronales se benefician de datos etiquetados, pero en voz los **datos etiquetados son mucho más escasos que los no etiquetados**. Los sistemas ASR de la época requerían **miles de horas de voz transcrita**, algo no disponible para la inmensa mayoría de las ~7.000 lenguas del mundo. El cuello de botella no es el audio (abundante) sino su **anotación** humana, lenta y costosa. Los autores lo enmarcan con una analogía cognitiva: aprender solo de ejemplos etiquetados no se parece a cómo los humanos adquieren el lenguaje — los bebés aprenden a hablar **escuchando**, lo que exige aprender buenas representaciones de la voz *antes* de asociarlas a transcripciones.

El [aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado) (SSL) emergió como el paradigma para aprender representaciones generales de datos no etiquetados y luego afinar sobre datos etiquetados. La receta había triunfado en NLP (línea ELMo/GPT/BERT); wav2vec 2.0 es el trabajo que **la traslada al audio crudo** de forma contundente. El antecedente inmediato del propio grupo, **vq-wav2vec**, cuantizaba en un primer paso y *después* entrenaba representaciones contextualizadas: un pipeline de dos etapas. La apuesta de wav2vec 2.0 es resolver ambos problemas de una sola vez, **end-to-end**.

## Contribución central

El marco unifica **SSL contrastivo con cuantización aprendida conjuntamente**, con cuatro ideas de diseño. **Enmascaramiento en el espacio latente**: en vez de enmascarar tokens de entrada (BERT de texto) o reconstruir *filter banks*, se enmascaran las representaciones latentes que produce el encoder convolucional. **Cuantización aprendida end-to-end**: las unidades discretas de voz se aprenden al mismo tiempo que las representaciones contextualizadas, vía Gumbel-softmax, y no en una etapa separada. **Objetivo contrastivo sobre latentes cuantizados**: el modelo debe identificar el latente cuantizado correcto para cada paso enmascarado entre distractores, y los autores encuentran que targets cuantizados funcionan mejor que continuos. **Demostración de ASR de ultra-bajos recursos**: que 10 minutos etiquetados basten para un WER competitivo es la evidencia que cambia el paradigma. Aprender conjuntamente unidades discretas y contexto rinde **sustancialmente mejor** que fijarlas en un paso previo, reduciendo el WER cerca de un tercio respecto de vq-wav2vec.

## Método: encoder + cuantización + Transformer + pérdida contrastiva

El modelo fluye desde la onda cruda hacia representaciones contextualizadas. El **feature encoder** convolucional $f: \mathcal{X} \mapsto \mathcal{Z}$ tiene siete bloques (convolución temporal + layer norm + GELU) que producen latentes $z_1, \dots, z_T$ a **49 Hz**, con paso de ~20 ms y campo receptivo de 25 ms: cada $z_t$ resume una ventana de voz análoga a un *frame* acústico, pero **aprendida**, no diseñada como las MFCC. El **Transformer de contexto** $g: \mathcal{Z} \mapsto \mathcal{C}$ construye representaciones $c_t$ que capturan toda la secuencia (BASE 95 M / LARGE 317 M), usando una capa convolucional como embedding posicional relativo.

El **módulo de cuantización** discretiza $z$ vía *product quantization*: con $G=2$ codebooks de $V=320$ entradas se eligen y concatenan entradas, dando un máximo teórico de $320^2 = 102{.}400$ palabras de código. Como el argmax no es diferenciable, se usa **Gumbel-softmax**:

$$p_{g,v} = \frac{\exp\big((l_{g,v} + n_v)/\tau\big)}{\sum_{k=1}^{V} \exp\big((l_{g,k} + n_k)/\tau\big)}$$

con temperatura $\tau$ que se recuece de 2 a 0,5/0,1, y ruido Gumbel $n = -\log(-\log(u))$: selección dura hacia adelante, gradiente continuo hacia atrás (*straight-through*). El **enmascaramiento** estilo BERT reemplaza tramos de latentes por un vector entrenado; con $p=0{,}065$ y $M=10$ se enmascara ~49% de los pasos. El objetivo combina **pérdida contrastiva** $L_m$ (identificar $q_t$ entre $K=100$ distractores de la misma locución) y **pérdida de diversidad** $L_d$ (maximizar el uso equitativo del codebook):

$$L_m = -\log \frac{\exp\big(\text{sim}(c_t, q_t)/\kappa\big)}{\sum_{\tilde{q}\sim Q_t} \exp\big(\text{sim}(c_t, \tilde{q})/\kappa\big)}, \qquad L = L_m + \alpha L_d$$

es la forma de una pérdida InfoNCE con similitud coseno. Tras el pre-training, el **fine-tuning** agrega una proyección lineal y minimiza una **pérdida CTC** (29 tokens para Librispeech), congelando el feature encoder.

## Resultados

En **bajos recursos** (LARGE preentrenado sobre LV-60k, 53.200 h): con **10 minutos** etiquetados logra **4,8/8,2** (test clean/other) frente al 16,3/25,2 del Discrete BERT previo; con **1 hora** llega a 2,9/5,8, superando el estado del arte que usaba 100 horas (100× menos etiqueta); con **100 horas**, 2,0/4,0, una reducción relativa de WER de 45%/42% sobre el *pseudo-labeling* previo. En **altos recursos** (960 h completas) alcanza **1,8/3,3 WER**, nuevo estado del arte en voz ruidosa — y lo consigue **a pesar de una arquitectura más débil**: el mismo modelo desde cero logra 2,1/4,6, así que **el pre-training aporta la diferencia**. Una ablación clave justifica el diseño: **entradas continuas al Transformer con targets cuantizados** da 7,97 de WER promedio, contra 8,58 con targets continuos — cuantizar los targets evita que capturen artefactos (identidad del hablante, ruido) que facilitarían la tarea de forma tramposa. En TIMIT, el análisis muestra que muchos latentes discretos **se especializan en fonemas** sin supervisión: descubre estructura lingüística real.

## Limitaciones

- **Arquitectura acústica subóptima.** Usan Transformer con CTC y vocabulario de caracteres, que no calza con el vocabulario de palabras del LM; esperan ganancias con seq2seq y *word pieces*.
- **Sin balanceo ni self-training.** El self-training es complementario al pre-training y su combinación podría dar más; no se explora aquí.
- **Costo de pre-training.** Aunque el fine-tuning es barato, preentrenar sobre 53k horas exige **128 GPUs V100 por varios días**: el ahorro está en la etiqueta humana, no en el cómputo.
- **Evaluación mayormente en inglés / voz leída.** Los experimentos se centran en Librispeech/LibriVox (audiolibros limpios); la promesa multilingüe se enuncia pero no se demuestra a fondo.

## Por qué importa para la Clase 37

La [Clase 37](/clases/clase-37) presenta distintas formas de [representar el audio](/fundamentos/representacion-de-audio): desde la onda cruda y el espectrograma, pasando por features diseñadas como las MFCC, hasta los **embeddings preentrenados**. wav2vec 2.0 es el ejemplo canónico de esta última categoría, y materializa el [aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado): en lugar de calcular MFCC —un pipeline fijo de FFT, banco de filtros mel y DCT diseñado en los años 80— se pasa el audio por un modelo autosupervisado y se **extraen las activaciones de la red de contexto** como vector de features. Esas activaciones, aprendidas de 53.000 horas, codifican estructura fonética y contextual que las MFCC no capturan — el mismo patrón *pre-training + fine-tuning ligero* que revolucionó NLP con BERT, ahora en audio.

El [Laboratorio 37](/laboratorios/lab-37/03-gtzan-mfcc-vs-wav2vec) pone esa promesa a prueba y mide **el gradiente de la transferencia**: los mismos embeddings valen **+9,3 puntos** sobre MFCC en clasificación de géneros musicales, **+33,7 puntos** en habla en inglés, y transcriben con **0% de WER** un clip de LibriSpeech — el corpus con el que fueron preentrenados. El valor del preentrenamiento resulta ser función de la distancia al dominio, y esa función es medible. El lab también documenta un hallazgo lateral que matiza cualquier comparación de este tipo: buena parte de la ventaja sobre los MFCC no viene de lo aprendido, sino de que **los embeddings salen normalizados por LayerNorm** mientras los MFCC crudos llegan a la red con norma ~205 y **saturan sus activaciones desde la inicialización** (gradiente de orden $10^{-13}$). Al comparar features aprendidas contra features de fórmula, la escala es una variable de confusión que hay que controlar. Y una nota sobre qué capa usar: el lab extrae la **séptima de doce**, porque las finales están especializadas en el objetivo de preentrenamiento y las intermedias transfieren mejor a tareas distintas del reconocimiento de voz.

El paper también desacopla el costo caro (anotación humana) del recurso abundante (audio crudo): hacer ASR en una lengua nueva ya no exige miles de horas transcritas, sino preentrenar una vez y afinar con minutos. La relación con [Whisper](/papers/whisper-radford-2022) cierra el arco del dominio audio: wav2vec 2.0 es **autosupervisado** y requiere fine-tuning por dominio; Whisper apuesta por **supervisión débil masiva** (680.000 horas web) y funciona zero-shot. Comparten el diagnóstico —escalar los datos es la palanca— pero difieren en la fuente de señal. En la práctica conviven: los embeddings de wav2vec 2.0 siguen siendo el *backbone* de features para tareas de voz de bajo recurso, mientras Whisper domina la transcripción end-to-end. En salud, donde la voz es un biomarcador (Parkinson, depresión, deterioro cognitivo) y los datos etiquetados son escasísimos, extraer embeddings preentrenados permite entrenar clasificadores clínicos robustos con muy pocos ejemplos — exactamente el régimen de 10 minutos donde este paper demostró que el pre-training recupera casi todo el desempeño.
