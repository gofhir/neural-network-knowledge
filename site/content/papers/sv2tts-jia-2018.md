---
title: "SV2TTS: transferencia de verificación de hablante a síntesis multi-hablante (2018)"
weight: 430
math: true
---

{{< paper-card
    title="Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis"
    authors="Ye Jia, Yu Zhang, Ron J. Weiss, Quan Wang, Jonathan Shen, Fei Ren, Zhifeng Chen, Patrick Nguyen, Ruoming Pang, Ignacio Lopez Moreno, Yonghui Wu (Google)"
    year="2018"
    venue="NeurIPS 2018 / arXiv:1806.04558"
    pdf="/papers/sv2tts-jia-2018.pdf" >}}
Hasta 2018 agregar una voz nueva a un sistema TTS neuronal costaba **decenas de minutos de audio de estudio con transcripción alineada** más un reentrenamiento completo: un proyecto de producción de audio por cada voz. Este paper propone en cambio **desacoplar el modelado del hablante de la síntesis del habla** y entrenar tres redes **de forma completamente independiente**: un *speaker encoder* —tres LSTM apiladas que producen un vector unitario de 256 dimensiones— entrenado en la tarea de **verificación de hablante** con la pérdida GE2E sobre **18 000 hablantes de audio ruidoso y sin transcribir**; un sintetizador **Tacotron 2** que genera un mel-espectrograma desde fonemas, condicionado concatenando ese vector a la salida del encoder de texto en cada paso; y un vocoder **WaveNet** que invierte el mel a forma de onda **sin recibir jamás la identidad del hablante**. El resultado es **clonación de voz zero-shot**: unos pocos segundos de audio de referencia sin transcribir bastan para sintetizar en la voz de un hablante nuevo **sin actualizar ningún parámetro**, con MOS de naturalidad de **4.12** y de similitud de **3.03** sobre hablantes no vistos de LibriSpeech. El hallazgo estructural —el que sobrevivió a todas las arquitecturas que lo reemplazaron— es que **la diversidad de hablantes del encoder, y no la calidad del corpus de TTS, es la variable que gobierna la generalización a voces nuevas**: pasar el encoder de 1.2K a 18K hablantes sube la similitud en 0.80 puntos de MOS y divide por tres la tasa de igual error, **sin tocar un solo parámetro del sintetizador**. Es el paper que hizo de la clonación de voz un problema resuelto en lo esencial, con todo lo que eso implica: la [Clase 39](/clases/clase-39) lo retoma tanto por su valor como generador de datos sintéticos como por lo que ilustra sobre transferencia entre tareas de audio.
{{< /paper-card >}}

---

## Contexto: el TTS multi-hablante antes de 2018

Para 2018 el TTS neuronal ya había resuelto la naturalidad **para un hablante**. Tacotron (Wang et al., 2017) y Tacotron 2 (Shen et al., 2018) mostraron que un seq2seq con atención podía ir de grafemas a mel-espectrograma sin ningún *feature* lingüístico intermedio, y que enchufándole [WaveNet](/papers/wavenet-oord-2016) como vocoder se alcanzaba naturalidad cercana a la humana. Pero, como dice la introducción del paper, Tacotron 2 combinaba "la prosodia de Tacotron con la calidad de audio de WaveNet" y **"solo soportaba un hablante"**.

El costo de extenderlo era brutal en datos. Sintetizar habla natural requería "entrenar sobre una gran cantidad de pares habla-transcripción de alta calidad", y soportar muchos hablantes usaba típicamente **decenas de minutos de datos de entrenamiento por hablante**. Grabar horas de audio limpio de estudio, transcrito y alineado, para cada voz que se quiera soportar, no escala: no es un problema de modelado, es un problema de producción.

El paper ordena los enfoques previos en una escalera de tres peldaños, y ninguno cierra el problema:

| Enfoque | Ejemplos | Qué necesita para una voz nueva | Limitación |
|---|---|---|---|
| **Tabla de embeddings aprendida** | Deep Voice 2 (2017), Deep Voice 3 (2018) | Reentrenar; el hablante debe estar en el set de entrenamiento | **Conjunto cerrado**: solo sintetiza voces vistas |
| **Adaptación / fine-tuning** | VoiceLoop (2018) | "Decenas de minutos de habla de *enrollment* **y transcripciones**" | Transcripciones más cientos de iteraciones de backprop por voz |
| ***Few-shot* con encoder de hablante** | Neural Voice Cloning (Arik et al., 2018), Nachmani et al. (2018) | Unos segundos de audio sin transcribir | El encoder se entrena **conjuntamente** con el sintetizador, sobre el mismo dataset de TTS; la diversidad de hablantes queda acotada por ese dataset |

La tercera familia es la más cercana, y el paper marca la diferencia con precisión: *"nuestro trabajo es el más similar a los modelos de speaker encoding en [Arik et al., Nachmani et al.], excepto que usamos una red **entrenada independientemente** para una tarea de verificación de hablante sobre un dataset grande de audio sin transcribir de decenas de miles de hablantes"*. Y añade el hallazgo cuantitativo que justifica la maniobra: *"encontramos que la transferencia zero-shot requiere entrenar sobre **miles** de hablantes"*, muchos más de los que cabían en cualquier corpus de TTS.

{{< concept-alert type="clave" >}}
**Clonación de voz zero-shot**, según la definición operativa del paper: *"unos pocos segundos de audio de referencia **sin transcribir** de un hablante objetivo se usan para sintetizar habla nueva en la voz de ese hablante, **sin actualizar ningún parámetro del modelo**"*.

Los tres calificadores eliminan un costo distinto:

- **Sin transcribir** → no hace falta pipeline de anotación; sirve cualquier grabación.
- **Sin actualizar parámetros** → la adaptación es un *forward pass* de milisegundos, no cientos de pasos de gradiente. Se puede hacer en línea, por *request*, en un servidor multi-tenant.
- **Pocos segundos** → la calidad satura alrededor de los **5 segundos**, y con **2 segundos** ya se está cerca del óptimo.
{{< /concept-alert >}}

Las motivaciones declaradas son de accesibilidad ("restaurar la capacidad de comunicarse naturalmente a usuarios que perdieron su voz y por lo tanto **no pueden proveer muchos ejemplos nuevos de entrenamiento**"), traducción habla-a-habla preservando la voz, y TTS en escenarios de bajos recursos. En la misma introducción aparece la advertencia sobre "el potencial de mal uso de esta tecnología", que se discute más abajo.

## La contribución: desacoplar identidad y contenido

El argumento cabe en una frase de la introducción:

> *"Nuestro enfoque es **desacoplar el modelado del hablante de la síntesis de habla** entrenando independientemente una red de embedding discriminativa de hablantes que capture el espacio de características de hablante, y entrenando un modelo TTS de alta calidad sobre un dataset más pequeño, condicionado por la representación aprendida por la primera red. Desacoplar las redes permite entrenarlas sobre **datos independientes**, lo que reduce la necesidad de obtener datos multi-hablante de alta calidad."*

Todo lo demás en el paper es la instanciación de esa frase y la evidencia de que funciona. El sistema es una cadena de tres contratos:

$$\text{audio de referencia} \xrightarrow{\;\text{encoder}\;} \mathbf{e} \in \mathbb{R}^{256},\ \|\mathbf{e}\|_2 = 1$$

$$\text{fonemas} + \mathbf{e} \xrightarrow{\;\text{Tacotron 2}\;} \text{mel-espectrograma (80 canales)} \xrightarrow{\;\text{WaveNet}\;} \text{forma de onda}$$

| Componente | Base | Dataset de entrenamiento | Qué le exige a ese dataset |
|---|---|---|---|
| **Speaker encoder** | Wan et al., ICASSP 2018 (GE2E) | Corpus interno de *voice search*: 36M utterances, **18K hablantes** | Muchísimos hablantes; **audio ruidoso, con reverberación, sin transcripción**; solo etiquetas de identidad anonimizadas |
| **Sintetizador** | Tacotron 2 | VCTK (44 h, 109 hablantes) o LibriSpeech clean (436 h, 1 172 hablantes) | Pares texto-audio de calidad. **Ninguna etiqueta de identidad** |
| **Vocoder** | WaveNet, 30 capas de convolución dilatada | El mismo corpus del sintetizador (uno por corpus) | Audio de calidad, de muchos hablantes |

### Por qué entrenar por separado es *la* decisión

Aquí está el argumento no obvio, y conviene enunciarlo sin adornos: **los datos ideales para cada componente son mutuamente incompatibles**.

- Para que el encoder generalice a voces nuevas necesita haber visto **muchísimas voces** —miles— y voces reales, con acústica de mundo real.
- Para que el sintetizador produzca audio natural necesita **transcripciones exactas y audio limpio**, porque cualquier ruido en el *target* se aprende y se reproduce. El paper lo verifica: el sintetizador de LibriSpeech "aprendió a reproducir" parte del ruido de fondo del corpus.

Un dataset con miles de hablantes, transcrito y grabado en estudio, sencillamente no existe a escala. Un corpus de 18K hablantes de *voice search* sí existe, pero es inservible para entrenar un TTS: no está transcrito y el audio es de teléfono con ruido de calle. **El desacople permite usar cada dataset para lo único en lo que sirve.**

Lo que gana cada pieza:

- **El encoder gana diversidad de hablantes a costo casi nulo.** El requisito de datos del speaker encoder *"es mucho más barato que el entrenamiento TTS completo, porque no son necesarias transcripciones, y la calidad de audio puede ser más baja"*.
- **El sintetizador gana simplicidad.** No usa etiquetas explícitas de identidad —el "label" del hablante *es* el propio audio pasado por el encoder congelado— y no necesita pérdidas auxiliares. Frente a Nachmani et al., el esquema *"no requiere pérdidas adicionales de triplete o contrastivas"* (véase [triplet loss](/fundamentos/triplet-loss)).
- **El sistema gana composicionalidad.** Se puede reemplazar el encoder por uno mejor sin tocar el sintetizador. La ablación central del paper es exactamente eso: cinco encoders distintos contra un mismo sintetizador.

Hay un matiz honesto que el propio paper documenta en su apéndice. Compararon el entrenamiento separado contra dos *baselines* entrenados conjuntamente, todos sobre LibriSpeech Clean y con embedding de 64 dimensiones. A igual dataset y a igual dimensión, **la separación no gana por mucho**: 3.73 contra 3.71 de naturalidad en hablantes no vistos, 2.23 contra 2.12 de similitud. La superioridad del modelo propuesto (4.12 / 3.03) **no viene de la separación por sí misma: viene de que la separación habilita entrenar el encoder sobre 18K hablantes**. Separar no es mejor por elegancia arquitectónica; es mejor porque desbloquea un dataset que de otro modo sería inaccesible.

## El encoder de hablante

La tarea de verificación de hablante impone naturalmente los tres invariantes que la síntesis necesita. El paper los enumera: hace falta *"una representación que capture las características de distintos hablantes, y la capacidad de identificar esas características usando solo una señal de adaptación corta, **independientemente de su contenido fonético y del ruido de fondo**"*. Es decir: invarianza al **contenido** (la verificación es *text-independent*), invarianza al **canal y al ruido** (porque el corpus de entrenamiento es ruidoso), y funcionamiento con **audio corto**.

**Arquitectura.** Entrada de log-mel espectrogramas de **40 canales**; cuerpo de **3 capas LSTM de 768 celdas, cada una con una proyección a 256 dimensiones** (la arquitectura LSTMP estándar de Google: la proyección recurrente reduce el estado que se realimenta, abarata la matriz recurrente y actúa como cuello de botella); salida tomada de la capa superior **en el último frame** y sometida a **normalización $L_2$**. El resultado es el **d-vector** de 256 dimensiones, viviendo sobre la hiperesfera unitaria $\mathbb{S}^{255}$.

Que el embedding esté normalizado no es cosmético. Es lo que hace que la **similitud coseno sea equivalente al producto punto**, lo que permite muestrear "hablantes ficticios" uniformemente sobre la esfera, y lo que le da al espacio una geometría acotada que el sintetizador puede recorrer sin encontrarse regiones degeneradas.

**Ventanas: entrenamiento e inferencia difieren.** Durante el entrenamiento el corpus se segmenta en trozos de **1.6 segundos** con su etiqueta de hablante, y *"no se usan transcripciones"*. En inferencia, una utterance de longitud arbitraria se parte en **ventanas de 800 ms con 50 % de solapamiento**, la red corre independientemente sobre cada ventana, y las salidas se **promedian y se re-normalizan**:

$$\mathbf{e} = \frac{\bar{\mathbf{e}}}{\|\bar{\mathbf{e}}\|_2}, \qquad \bar{\mathbf{e}} = \frac{1}{W}\sum_{w=1}^{W} f(\mathbf{x}_w)$$

Ese promedio sobre $W$ ventanas es lo que reduce la varianza del embedding, y es la razón mecánica por la que más audio de referencia mejora la similitud —hasta que satura.

### La pérdida GE2E

El paper no escribe la fórmula: describe el objetivo —*"los embeddings de utterances del mismo hablante tengan alta similitud coseno, mientras que los de hablantes distintos queden lejos en el espacio de embeddings"*— y cita a Wan, Wang, Papir y Lopez Moreno, *Generalized end-to-end loss for speaker verification* (ICASSP 2018). La formulación que sigue viene de **ese** paper, no de este.

El batch es una matriz de $N \times M$ utterances: $N$ hablantes, $M$ utterances por hablante. Con $\mathbf{e}_{ji}$ el embedding normalizado de la utterance $i$ del hablante $j$, el centroide del hablante $k$ es

$$\mathbf{c}_k = \frac{1}{M}\sum_{m=1}^{M} \mathbf{e}_{km}$$

y la matriz de similitud es la **similitud coseno escalada y desplazada** por dos escalares aprendidos $w > 0$ y $b$:

$$S_{ji,k} = w \cdot \cos(\mathbf{e}_{ji}, \mathbf{c}_k) + b$$

El truco central de GE2E: al comparar un embedding contra el centroide de **su propio** hablante, ese centroide se calcula **excluyendo la utterance en cuestión**, para evitar la solución trivial de compararse consigo mismo:

$$\mathbf{c}_j^{(-i)} = \frac{1}{M-1}\sum_{m \neq i} \mathbf{e}_{jm}, \qquad
S_{ji,k} = \begin{cases} w\cos(\mathbf{e}_{ji}, \mathbf{c}_j^{(-i)}) + b & k = j \\[2pt] w\cos(\mathbf{e}_{ji}, \mathbf{c}_k) + b & k \neq j \end{cases}$$

Sobre esa matriz se define la pérdida en variante **softmax**,

$$L(\mathbf{e}_{ji}) = -S_{ji,j} + \log \sum_{k=1}^{N} \exp(S_{ji,k})$$

o en variante **contrast**, que penaliza solo al impostor más difícil:

$$L(\mathbf{e}_{ji}) = 1 - \sigma(S_{ji,j}) + \max_{k \neq j} \sigma(S_{ji,k})$$

Tres razones por las que no cualquier pérdida discriminativa serviría igual:

1. **Es verificación, no clasificación.** Un softmax sobre 18K identidades produce una capa de salida atada al conjunto de entrenamiento, y el penúltimo layer no tiene ninguna garantía métrica. GE2E optimiza directamente la geometría del espacio —mismo hablante junto, distinto hablante lejos, medido en coseno—, y ese espacio **es** la interfaz con el sintetizador. Es el mismo argumento de FaceNet frente a un clasificador de identidades; véase [metric learning](/fundamentos/metric-learning).
2. **Compara contra centroides, no contra pares ni tripletes.** Con $N \times M$ embeddings se obtienen $N \times M \times N$ comparaciones por batch, sin minería de tripletes.
3. **La escala aprendida importa.** Los embeddings están en la esfera unitaria, así que $\cos \in [-1,1]$; sin el factor $w$, el softmax sería casi uniforme y los gradientes minúsculos. Es el mismo motivo por el que los métodos de *margin softmax* en reconocimiento facial introducen un factor de escala.

Y el comentario más interesante del paper sobre este componente:

> *"Aunque la red **no está optimizada directamente** para aprender una representación que capture características de hablante relevantes para la **síntesis**, encontramos que entrenar en una tarea de discriminación de hablante lleva a un embedding que es **directamente adecuado** para condicionar la red de síntesis."*

No había ninguna razón *a priori* para que un espacio optimizado para responder "¿son la misma persona?" fuera un espacio del que se pueda **generar**. Es una afirmación empírica, y las ablaciones son la evidencia.

## El sintetizador

La base es Tacotron 2: encoder de texto convolucional más BiLSTM, atención sensible a la localización, decoder autorregresivo LSTM que emite frames de mel, post-net convolucional residual y un *stop token*. Los cambios que SV2TTS introduce son cuatro:

- **Entrada fonémica.** *"Mapeamos el texto a una secuencia de fonemas, lo cual lleva a convergencia más rápida y a mejor pronunciación de palabras raras y nombres propios."*
- **Target espectral:** ventanas de 50 ms, salto de 12.5 ms, banco de filtros mel de **80 canales**, con compresión logarítmica de rango dinámico. (Nótese la asimetría con el encoder de hablante, que usa 40 canales: son dos front-ends distintos, no compartidos.)
- **Pérdida combinada:** al $L_2$ sobre el espectrograma se le suma un término $L_1$, $\mathcal{L} = \|\hat{Y} - Y\|_2^2 + \|\hat{Y} - Y\|_1$, porque *"esta pérdida combinada es más robusta sobre datos de entrenamiento ruidosos"*. El $L_1$ le baja el peso a los outliers, que en audio ruidoso son ruido y no señal.
- **Sin pérdidas auxiliares** basadas en el embedding del hablante.

### Dónde se inyecta el embedding, y por qué ahí

> *"Un vector de embedding para el hablante objetivo se **concatena con la salida del encoder del sintetizador en cada paso temporal**."*

Si el encoder de texto produce $H = [\mathbf{h}_1, \dots, \mathbf{h}_L]$ con un vector por fonema, la secuencia que ve la atención es

$$\tilde{\mathbf{h}}_l = [\mathbf{h}_l \,;\, \mathbf{e}] \in \mathbb{R}^{d_h + 256}, \qquad l = 1, \dots, L$$

es decir, el mismo $\mathbf{e}$ replicado a lo largo de toda la secuencia de texto. Hay al menos cuatro puntos de inyección plausibles, y vale la pena ver qué falla en cada uno:

| Punto de inyección | Problema |
|---|---|
| Estado inicial del decoder | El decoder es recurrente sobre cientos de frames: la información del hablante **se diluye** con el tiempo. El timbre derivaría a lo largo de la utterance |
| Concatenar al input del decoder en cada paso | Funciona, pero deja a la **atención ciega al hablante**: el alineamiento texto↔audio no puede adaptarse a la velocidad de habla ni a la duración fonémica del hablante |
| Solo al post-net o al vocoder | Es una corrección *post hoc* sobre un mel ya generado como si fuera de un hablante promedio. La estructura de $F_0$ y de formantes debe estar en el mel desde su generación, no maquillada después |
| **A la salida del encoder de texto** | **Cada vector de contexto lleva la identidad en todos los pasos, y las *keys* de la atención son función del hablante** |

Esa última propiedad tiene dos consecuencias que se componen. El vector de contexto es $\mathbf{c}_t = \sum_l \alpha_{t,l} \tilde{\mathbf{h}}_l$; como todos los $\tilde{\mathbf{h}}_l$ contienen $\mathbf{e}$ y $\sum_l \alpha_{t,l} = 1$, el embedding **pasa intacto al decoder en cada paso**, sin depender de dónde esté mirando la atención: es un canal de condicionamiento persistente y gratuito. Y al mismo tiempo, como las *keys* que la atención puntúa dependen de $\mathbf{e}$, el **alineamiento** —o sea la duración asignada a cada fonema, o sea la velocidad de habla y el ritmo— puede condicionarse al hablante. El paper confirma que esto ocurre: *"la velocidad de habla característica también es capturada hasta cierto punto por el embedding del hablante"*.

Frente a Deep Voice 2, que inyectaba embeddings en múltiples sitios de la red, el paper reporta una simplificación: *"encontramos que **simplemente pasar los embeddings a la capa de atención** converge a través de distintos hablantes"*. Un solo punto basta.

{{< concept-alert type="advertencia" >}}
**Un detalle de la configuración de entrenamiento con consecuencias en toda la línea.** El sintetizador se entrena con el speaker encoder **congelado**, y *"la señal de referencia del hablante es la misma que el habla objetivo durante el entrenamiento"*: el embedding se extrae del audio que hay que reproducir.

Eso es lo que permite prescindir de etiquetas de identidad, pero es también una **fuga**. Cualquier información prosódica residual que sobreviva en el embedding se vuelve predictiva durante el entrenamiento, y el decoder aprende a explotarla. En inferencia, con una referencia distinta al *target*, esa misma dependencia se convierte en **transferencia de prosodia no deseada**: el paper observa que *"la prosodia del habla sintetizada a veces imita la de la referencia"*, y propone como remedio entrenar sobre pares de referencia y target elegidos aleatoriamente del mismo hablante.
{{< /concept-alert >}}

## El vocoder, y por qué no necesita saber quién habla

La sección del vocoder es la más corta del paper y la más fácil de citar mal:

> *"Usamos el WaveNet autorregresivo muestra a muestra como vocoder para invertir los mel-espectrogramas sintetizados en formas de onda... compuesto por **30 capas de convolución dilatada**. **La red no está condicionada directamente por la salida del speaker encoder.** El mel-espectrograma predicho por la red de síntesis **captura todo el detalle relevante** necesario para síntesis de alta calidad de una variedad de voces, permitiendo construir un vocoder multi-hablante simplemente entrenándolo sobre datos de muchos hablantes."*

**Por qué es cierto.** El argumento es de **suficiencia de la representación intermedia**: prácticamente toda la información que un oyente usa para identificar una voz es espectral y está presente en el log-mel. El paper lo verifica visualmente sobre mel-espectrogramas de tres hablantes distintos diciendo la misma frase:

- **Frecuencia fundamental $F_0$** (la altura de la voz), que aparece como el espaciado de los armónicos: el hablante masculino tiene *"frecuencia fundamental notablemente más baja, visible en el espaciado armónico más denso en las frecuencias bajas"*.
- **Formantes**, o sea las resonancias del tracto vocal, que son esencialmente la geometría anatómica de la persona: *"el F2 del hablante masculino está en el canal mel 35, mientras que el F2 del hablante del medio aparece más cerca del canal 40"*.
- **Distribución de energía en sibilantes**: *"la 's' contiene más energía en frecuencias bajas en la voz masculina que en las femeninas"*.
- **Inclinación espectral, ancho de banda de formantes y velocidad de habla**, todo visible en el mel.

Lo que **no** está en el mel es la **fase** y la estructura fina de la forma de onda por debajo de la resolución de 12.5 ms y de los 80 canales. Reconstruir eso es la tarea de WaveNet, y es —en primera aproximación— un problema acústico **genérico**: dado un contorno espectral y una $F_0$ implícita, generar una excitación glotal plausible y filtrarla. No hace falta saber *quién* habla porque el *qué* ya viene especificado frame a frame. El mel-espectrograma funciona como una **frontera de abstracción**: el sintetizador es responsable de "quién habla y qué dice", el vocoder de "cómo suena una señal con este espectro".

Dicho eso, la afirmación es más débil de lo que la cita suelta sugiere, y el paper es más cuidadoso que sus lectores. Tres matices, todos con evidencia en el propio texto:

**(a) El vocoder sí debe ser *entrenado* multi-hablante.** La frase completa dice "permitiendo construir un vocoder multi-hablante **simplemente entrenándolo sobre datos de muchos hablantes**". No es que el vocoder sea agnóstico al hablante: es que no necesita **condicionamiento explícito**, porque le basta haber visto suficiente variedad de espectros para que el mapeo mel→onda esté bien definido en todo el rango de $F_0$ y de calidades vocales que se le van a pedir. Un WaveNet entrenado con una sola voz masculina y evaluado sobre un mel de voz femenina aguda estaría extrapolando.

**(b) Entrenaron un vocoder por corpus, no uno solo.** *"Entrenamos redes de síntesis y de vocoder separadas para cada uno de estos dos corpus"*: VCTK a 24 kHz, LibriSpeech a 16 kHz. En la práctica el vocoder tampoco es transferible entre dominios acústicos sin reentrenar.

**(c) La receta cambia según el ruido del corpus, y esto anticipa el argumento de la clase.** *"Para el dataset VCTK, cuyo audio es bastante limpio, encontramos que el vocoder entrenado sobre **mel-espectrogramas ground truth** funcionaba bien. Sin embargo para LibriSpeech, que es más ruidoso, encontramos **necesario entrenar el vocoder sobre espectrogramas predichos por la red de síntesis**."* Es *exposure bias* entre etapas: si el vocoder solo vio mels reales y en inferencia recibe mels predichos —más suaves, con artefactos propios de la pérdida $L_2$/$L_1$—, hay desajuste de distribución. Entrenarlo sobre las predicciones del sintetizador lo alinea con lo que verá en producción. Es la brecha de dominio en miniatura, dentro del propio sistema.

## La asimetría de los datasets

| Dataset | Rol | Hablantes | Volumen | Transcripción | Calidad de audio |
|---|---|---|---|---|---|
| **Corpus interno de *voice search*** | Speaker encoder (modelo principal) | **18 000** | 36M utterances, duración mediana 3.9 s | **No** (identidades anonimizadas) | Habla real, ruidosa |
| **VoxCeleb** (2017) | Speaker encoder (ablación) | **1 211** | 139K utterances | No | Entrevistas de YouTube, ruidosas |
| **VoxCeleb2** (2018) | Speaker encoder (ablación) | **5 994** | 1.09M utterances | No | Ídem |
| **LibriSpeech Other** | Speaker encoder (ablación) | **1 166** | 461 horas | Sí (no usada aquí) | Audiolibros, variable |
| **[LibriSpeech](/papers/librispeech-panayotov-2015) clean** | Sintetizador + vocoder | **1 172** | 436 horas, 16 kHz | **Sí** (sin puntuación) | Audiolibros, con ruido de fondo apreciable |
| **VCTK** (2017) | Sintetizador + vocoder | **109** (98 en entrenamiento) | 44 horas, 24 kHz | **Sí** | Limpio, mayoría acento británico |

El encoder principal ve **18 000 hablantes**; el sintetizador ve **1 172** en el mejor caso y **98** en el peor. Un factor de **15× a 180×**. Y esa asimetría es el punto entero del paper: el corpus de 18K hablantes **no tiene transcripciones** (inservible para entrenar un TTS), **es ruidoso** (inservible como target de síntesis) y tiene **36M utterances con 18K identidades** (perfecto para verificación de hablante). VCTK es el espejo exacto: limpio, transcrito, 24 kHz, ideal para TTS, pero con 109 hablantes no permite aprender el espacio de variación de la voz humana.

**La evidencia está en la ablación del encoder.** El sintetizador es siempre el mismo (LibriSpeech Clean), evaluado sobre hablantes *held-out*; lo único que cambia es el dataset del speaker encoder:

| Dataset del speaker encoder | Hablantes | Dim. emb. | Naturalidad | Similitud | SV-EER |
|---|---|---|---|---|---|
| LibriSpeech Clean | 1.2K | 64 | 3.73 ± 0.06 | 2.23 ± 0.08 | 16.60 % |
| LibriSpeech Other | 1.2K | 64 | 3.60 ± 0.06 | 2.27 ± 0.09 | 15.32 % |
| LS-Other + VoxCeleb | 2.4K | 256 | 3.83 ± 0.06 | 2.43 ± 0.09 | 11.95 % |
| LS-Other + VoxCeleb + VoxCeleb2 | 8.4K | 256 | 3.82 ± 0.06 | 2.54 ± 0.09 | 10.14 % |
| **Interno (*voice search*)** | **18K** | **256** | **4.12 ± 0.05** | **3.03 ± 0.09** | **5.08 %** |

De 1.2K a 18K hablantes: naturalidad **+0.39**, similitud **+0.80**, y la tasa de igual error dividida por más de tres (16.60 % → 5.08 %). **Sin tocar el sintetizador ni un solo parámetro.**

Dos lecturas finas que el paper hace y que suelen perderse. Primero, las dos filas iniciales son un control: LS-Clean es la condición *matched* (encoder y sintetizador entrenados sobre los mismos datos) y LS-Other es disjunta con el mismo número de hablantes. *"Esta condición matched da naturalidad ligeramente mejor y un puntaje de similitud similar"* — es decir, **entrenar el encoder sobre datos que el sintetizador nunca vio no cuesta prácticamente nada**; lo único que importa es cuántos hablantes distintos vio el encoder. Segundo, hay una confusión parcial en la tabla: las dos primeras filas usan arquitectura reducida y embedding de 64 dimensiones ("para evitar sobreajuste"), de modo que el salto entre las filas 2 y 3 mezcla *más hablantes* con *más capacidad*. El salto de 8.4K a 18K, en cambio, es limpio —ambas con 256 dimensiones— y explica por sí solo **+0.30 de naturalidad, +0.49 de similitud y −5 puntos de EER**. Ese salto aislado es el argumento más fuerte del paper.

{{< concept-alert type="clave" >}}
**El valor de un dataset no es su calidad promedio, es la diversidad que aporta sobre el eje que la tarea necesita.** 18K hablantes de audio de teléfono sin transcribir valen más, para generalizar a voces nuevas, que 1.2K hablantes de audio impecable y transcrito.
{{< /concept-alert >}}

## Resultados

**Protocolo.** MOS crowdsourced en escala ACR (ITU-T P.800), de 1 a 5 en incrementos de 0.5, sobre 100 frases que no aparecen en ningún set de entrenamiento. Once hablantes vistos y once no vistos en VCTK; diez y diez en LibriSpeech. Para cada hablante se elige **al azar una utterance de unos 5 segundos** para calcular el embedding. Alrededor de 1 000 utterances sintetizadas por evaluación, cada muestra calificada por **un solo rater**, y cada evaluación es independiente: las salidas de distintos modelos **no se compararon directamente entre sí**, de modo que comparar MOS entre tablas distintas es arriesgado. En la evaluación de similitud, cada utterance sintetizada se aparea con una utterance real del mismo hablante y la instrucción es explícita: *"no debe juzgar el contenido, la gramática ni la calidad de audio... enfóquese solo en la similitud de los hablantes entre sí"*.

### Naturalidad

| Sistema | VCTK vistos | VCTK **no vistos** | LibriSpeech vistos | LibriSpeech **no vistos** |
|---|---|---|---|---|
| Ground truth | 4.43 ± 0.05 | 4.49 ± 0.05 | 4.49 ± 0.05 | 4.42 ± 0.07 |
| Tabla de embeddings (baseline) | 4.12 ± 0.06 | N/A | 3.90 ± 0.06 | N/A |
| **Modelo propuesto** | 4.07 ± 0.06 | **4.20 ± 0.06** | 3.89 ± 0.06 | **4.12 ± 0.05** |

El modelo **empata con el baseline de tabla de embeddings** en hablantes vistos (indistinguibles dentro del intervalo de confianza): la generalización zero-shot **no se paga con naturalidad**. VCTK saca unos 0.2 puntos a LibriSpeech, por dos causas que el paper identifica: *"(i) la falta de puntuación en las transcripciones, que hace difícil que el modelo aprenda a pausar naturalmente, y (ii) el mayor nivel de ruido de fondo... parte del cual el sintetizador ha aprendido a reproducir"*. La primera es instructiva: **la puntuación es el único portador de estructura prosódica en la entrada textual**; sin comas ni puntos no hay señal de dónde pausar.

Vale mencionar el preproceso de denoising de LibriSpeech, porque está bien pensado: sustracción espectral con el espectro de ruido estimado como *"el percentil 10 de la energía en cada banda de frecuencia a lo largo de la señal completa"* — y, detalle crucial, *"este proceso solo se usó sobre el **target de síntesis**; el habla ruidosa original se pasó al **speaker encoder**"*. Denoisan lo que el sintetizador debe imitar, pero dejan el ruido en lo que el encoder debe caracterizar, porque el encoder fue entrenado sobre audio ruidoso y es robusto a eso.

### Similitud de hablante

| Sistema | Condición | VCTK | LibriSpeech |
|---|---|---|---|
| Ground truth | Mismo hablante | 4.67 ± 0.04 | 4.33 ± 0.08 |
| Ground truth | Mismo género, distinto hablante | 2.25 ± 0.07 | 1.83 ± 0.07 |
| Ground truth | Género distinto | 1.15 ± 0.04 | 1.04 ± 0.03 |
| Tabla de embeddings | Vistos | 4.17 ± 0.06 | 3.70 ± 0.08 |
| **Modelo propuesto** | **Vistos** | **4.22 ± 0.06** | **3.28 ± 0.08** |
| **Modelo propuesto** | **No vistos** | **3.28 ± 0.07** | **3.03 ± 0.09** |

Las tres filas de ground truth son la **calibración de la escala** y son lo más valioso de la tabla: dos utterances reales del mismo hablante puntúan 4.67, dos hablantes distintos del mismo género puntúan 2.25, y de género distinto 1.15. Eso ancla qué significa un 3.28 —que el paper describe como *"entre 'moderadamente similar' y 'muy similar'"*.

**La brecha entre hablantes vistos y no vistos es la cifra central del paper: 4.22 → 3.28 en VCTK (−0.94) y 3.28 → 3.03 en LibriSpeech (−0.25).** Y hay un resultado honesto que se omite al citar el paper: en LibriSpeech con hablantes vistos, **el modelo propuesto pierde contra la tabla de embeddings** (3.28 contra 3.70), atribuido al *"mayor grado de variación intra-hablante y al nivel de ruido de fondo del dataset"*. Tiene sentido: la tabla promedia sobre todas las utterances del hablante, mientras el encoder ve una sola de 5 segundos.

El diagnóstico cualitativo del paper es la mejor descripción de lo que este sistema realmente hace: *"el modelo propuesto es capaz de transferir **los trazos gruesos** de las características del hablante para hablantes no vistos, reflejando claramente el género, el pitch y los rangos de formantes correctos... pero los puntajes de similitud significativamente reducidos sobre hablantes no vistos sugieren que **algunos matices, por ejemplo los relacionados con la prosodia característica, se pierden**"*. Hay además un confusor de acento: el speaker encoder está entrenado *"solo sobre habla con acento norteamericano"*, y VCTK es mayoritariamente británico, lo que *"restringe nuestro desempeño en similitud de hablante sobre VCTK"*.

### Evaluación cruzada entre datasets

Sintetizador entrenado en un corpus, evaluado sobre hablantes no vistos del otro. **El speaker encoder es idéntico en ambas filas.**

| Sintetizador entrenado en | Evaluado en | Naturalidad | Similitud |
|---|---|---|---|
| VCTK (98 hablantes) | LibriSpeech | 4.28 ± 0.05 | **1.82 ± 0.08** |
| LibriSpeech (1.2K hablantes) | VCTK | 4.01 ± 0.06 | **2.77 ± 0.08** |

La naturalidad se mantiene, pero **la similitud se derrumba**: el 1.82 del sintetizador de VCTK está, en la escala calibrada de la tabla anterior, por debajo del "mismo género, distinto hablante". En términos prácticos, **fuera de su dominio el sintetizador de VCTK no clona nada**. La conclusión del paper es directa: *"entrenar el sintetizador sobre solo 100 hablantes es insuficiente para habilitar transferencia de hablante de alta calidad"*.

Esto matiza el mensaje principal de forma importante: **la diversidad de hablantes del encoder es necesaria pero no suficiente**. El sintetizador también necesita haber visto suficientes hablantes para aprender a *usar* el embedding como una dirección de control real, en lugar de memorizar un puñado de voces.

### Verificación de hablante sobre el audio sintetizado

Para no depender del propio espacio de embeddings, entrenaron un **encoder de evaluación independiente** —misma topología, distinto training set: 28M utterances de 113K hablantes— y midieron la **tasa de igual error de verificación (SV-EER)** sobre las formas de onda sintetizadas, con 21 000 a 23 100 *trials* por evaluación.

| Sintetizador | Hablantes de entrenamiento | SV-EER en VCTK | SV-EER en LibriSpeech |
|---|---|---|---|
| **Ground truth** (audio real) | — | **1.53 %** | **0.93 %** |
| VCTK | 98 | 10.46 % | **29.19 %** |
| LibriSpeech | 1.2K | 6.26 % | **5.08 %** |

El audio real da 1-1.5 %: el verificador funciona. El sintetizador de LibriSpeech da 5-6 % consistente en ambos corpus. El de VCTK **colapsa fuera de dominio a 29.19 %**, cerca del azar, en perfecta coincidencia con el 1.82 de similitud subjetiva. El habla sintética se parece al hablante objetivo, pero un verificador la distingue del habla real con facilidad.

### Duración del audio de referencia

Sobre VCTK, variando la longitud de la utterance de referencia:

| | 1 s | 2 s | 3 s | 5 s | 10 s |
|---|---|---|---|---|---|
| Naturalidad | **4.28 ± 0.05** | 4.26 ± 0.05 | 4.18 ± 0.06 | 4.20 ± 0.06 | 4.16 ± 0.06 |
| Similitud | 2.85 ± 0.07 | 3.17 ± 0.07 | **3.31 ± 0.07** | 3.28 ± 0.07 | 3.18 ± 0.07 |
| SV-EER | 17.28 % | 11.30 % | **10.80 %** | 10.46 % | 11.50 % |

Tres observaciones contraintuitivas hasta que se explican. La similitud **satura alrededor de los 3-5 segundos**, y con 2 segundos ya se está cerca del óptimo. Las referencias **más cortas dan naturalidad ligeramente mejor**, *"porque calzan mejor con las duraciones de las utterances de referencia usadas para entrenar el sintetizador, cuya duración mediana es 1.8 segundos"* — puro desajuste train/test. Y después de 5 segundos, más audio **no ayuda e incluso empeora un poco**, porque *"la saturación del desempeño usando solo 5 segundos de habla resalta una limitación del modelo propuesto, que está restringido por la **pequeña capacidad del embedding de hablante**"*. Un vector de 256 dimensiones es un cuello de botella duro.

{{< concept-alert type="advertencia" >}}
**La configuración estrella del paper no es reproducible fuera de Google.** Todas las cifras principales —naturalidad 4.12, similitud 3.03, SV-EER 5.08 %— provienen del encoder entrenado sobre el **corpus interno propietario de 18K hablantes de *voice search***, que no es público. VoxCeleb y VoxCeleb2 aparecen **solo en la ablación**.

La mejor fila reproducible con datos públicos es **LS-Other + VoxCeleb + VoxCeleb2 (8.4K hablantes)**:

| | Reproducible (8.4K, público) | Paper (18K, interno) |
|---|---|---|
| Naturalidad | **3.82 ± 0.06** | 4.12 ± 0.05 |
| Similitud | **2.54 ± 0.09** | 3.03 ± 0.09 |
| SV-EER | **10.14 %** | 5.08 % |

La diferencia no es cosmética: **medio punto de similitud y el doble de EER**. Al citar el paper o al intentar replicarlo conviene declarar cuál de las dos columnas se está usando.
{{< /concept-alert >}}

Un último recordatorio de higiene al citar: este paper tiene siete tablas de MOS con configuraciones distintas, y el 3.28 aparece **dos veces** (similitud de VCTK no vistos y de LibriSpeech vistos, coincidencia numérica pura). Si se cita "el MOS de SV2TTS" hay que decir **cuál métrica** (naturalidad o similitud), **sobre cuál corpus** (VCTK o LibriSpeech) y **en cuál condición** (vistos o no vistos). Sin esos tres calificadores el número no significa nada.

## El espacio de embeddings

El paper visualiza con PCA y t-SNE los embeddings de utterances de LibriSpeech, reales y sintéticas, y reporta tres hallazgos.

**Separación por hablante.** *"Distintos hablantes están bien separados unos de otros en el espacio de embeddings."*

**Separación por género.** *"Los hablantes aparecen bien separados por género tanto en PCA como en t-SNE, con **todas las hablantes femeninas apareciendo a la izquierda y todos los masculinos a la derecha**. Esto es un indicio de que el speaker encoder ha aprendido una representación razonable del espacio de hablantes."* Que el género emerja como un eje principal en PCA —o sea, como dirección de máxima varianza o cercana a ella— sin haber sido supervisado nunca indica que la dimensión acústicamente más discriminativa entre hablantes es la que separa rangos de $F_0$ y de formantes. Fonéticamente no sorprende; lo que confirma es que el espacio tiene estructura interpretable y no es un embedding arbitrario.

**Real y sintético forman clusters distintos pero adyacentes.** *"Las utterances sintetizadas tienden a caer muy cerca del habla real del mismo hablante. Sin embargo, siguen siendo **fácilmente distinguibles** del habla humana real... las utterances de cada hablante sintético forman un **cluster distinto adyacente** al cluster de utterances reales del hablante correspondiente."* Esa es la contraparte geométrica del EER: la síntesis introduce un **sesgo sistemático** en el espacio de embeddings, un desplazamiento consistente que un verificador puede detectar. En términos de detección de deepfakes: existe una "firma del vocoder" separable.

### Hablantes ficticios

El experimento más provocador del paper **se salta el speaker encoder por completo** y condiciona el sintetizador sobre **puntos aleatorios muestreados uniformemente sobre la superficie de la hiperesfera unitaria** de 256 dimensiones. Diez hablantes ficticios; se enrolan sus diez vecinos más cercanos en los sets de entrenamiento y se computa el EER.

| Vecinos más cercanos en | Similitud coseno | SV-EER | Naturalidad |
|---|---|---|---|
| Set de entrenamiento del **sintetizador** | 0.222 | **56.77 %** | 3.65 ± 0.06 |
| Set de entrenamiento del **speaker encoder** | 0.245 | **38.54 %** | (misma) |

La naturalidad de **3.65** es comparable a la de hablantes reales bajo el mismo encoder: *"aunque estos hablantes son totalmente ficticios, el sintetizador y el vocoder son capaces de generar audio **tan natural como para hablantes reales vistos o no vistos**"*. La similitud coseno de 0.22-0.25 al vecino más cercano es esencialmente ortogonalidad práctica —recuérdese que el habla sintética de un hablante real alcanzaba coseno superior a 0.6 con su objetivo—. Y el EER de **56.77 %** está **por encima del azar**: el verificador no logra asociar el hablante ficticio con ningún hablante de entrenamiento. Muestras generadas con distintos embeddings aleatorios contienen *"contenido fonético consistente, pero hay variación clara en la frecuencia fundamental y en la velocidad de habla"*.

Qué implica, en orden de importancia:

1. **El espacio de embeddings es una variedad densa y navegable, no un conjunto de puntos memorizados.** Si el sintetizador solo hubiera aprendido a mapear los embeddings vistos a sus voces, un punto aleatorio caería en una región no entrenada y produciría basura o colapsaría a la voz más cercana. No ocurre ninguna de las dos cosas: produce **una voz nueva, coherente y natural**. El paper lo concluye así: *"el modelo ha aprendido a utilizar una representación realista del espacio de variación de hablantes"*.
2. **Es la mejor prueba de que la generalización zero-shot es real y no recuperación aproximada.** Si el mecanismo fuera "encontrar al hablante de entrenamiento más parecido y usar su voz", este experimento sería imposible. La generalización a hablantes no vistos es interpolación o extrapolación genuina sobre una variedad continua, no un *nearest neighbor* encubierto. Es, en términos de [modelos generativos](/fundamentos/modelos-generativos), la diferencia entre haber aprendido una distribución y haber memorizado un catálogo.
3. **Se pueden fabricar identidades vocales que no pertenecen a nadie.** Éticamente esto corta en ambos sentidos: es el camino para generar voces sintéticas sin consentimiento de nadie —voces de marca, personajes, narradores— evitando el problema de la suplantación, y es el ancestro directo de los "diseñadores de voz" comerciales que hoy permiten especificar atributos en vez de subir una muestra.

Conviene una aclaración porque se le atribuyen resultados que no tiene: el paper **no reporta** experimentos de interpolación entre dos embeddings reales ni de aritmética vectorial en el espacio de hablantes. El muestreo uniforme sobre la esfera es lo único que hace.

## Limitaciones

**La brecha con hablantes no vistos.** De 4.22 a 3.28 en similitud sobre VCTK. Dicho con las palabras del paper: el modelo transfiere *"los trazos gruesos"* pero *"algunos matices se pierden"*. En 2018, el zero-shot cloning producía una voz **del mismo tipo** que el objetivo, no la voz del objetivo. A eso se suma que el sistema **no alcanza naturalidad humana** (4.12 contra 4.42-4.49 del ground truth) *"a pesar del uso de un vocoder WaveNet"*.

### La limitación central: transfiere timbre, no prosodia ni acento

Es la crítica más citada del paper y la que definió toda la agenda de investigación posterior, así que merece desarrollo.

**Lo que SV2TTS transfiere es el timbre**: la firma espectral estática de un hablante —posición de formantes (esencialmente la geometría del tracto vocal), rango de $F_0$, inclinación espectral, calidad de fonación—. Todo eso cabe en 256 dimensiones y todo eso el modelo lo reproduce con fidelidad razonable.

**Lo que no transfiere es el estilo de habla**, que es un objeto **temporal y contextual**: el contorno de entonación (cómo sube y baja el pitch a lo largo de una frase, dónde se pone el acento tonal, cómo se marcan las preguntas), el ritmo y la duración fonémica, las pausas y las dudas, el acento y el dialecto, el estado emocional y el registro.

Y no es un defecto de implementación: **el diseño no puede transferirlo, estructuralmente**, por tres razones que se componen.

1. **Un vector estático no puede representar una función del tiempo.** El embedding es un punto fijo replicado en todos los pasos temporales; el estilo prosódico es una **trayectoria**. Se puede codificar el *promedio* del pitch de alguien, pero no su patrón de contorno, en un vector que no varía. Es una limitación de tipo, no de capacidad.
2. **La pérdida GE2E está entrenada para ser invariante a la prosodia.** La verificación *text-independent* debe reconocer a la misma persona diciendo cosas distintas, con emociones distintas, a velocidades distintas. Para esa tarea la prosodia es **ruido de nuisance que hay que descartar**, y el encoder está literalmente optimizado para tirar a la basura exactamente la información que haría falta para transferir estilo. Esa es la ironía del diseño: **lo que lo hace robusto es lo que lo hace prosódicamente ciego**.
3. **La configuración de entrenamiento contamina lo poco que queda.** Como la referencia *es* el target durante el entrenamiento, el decoder aprende a exprimir cualquier residuo prosódico del embedding. En inferencia eso se manifiesta como transferencia **parcial y no controlada** de la prosodia de la referencia. El paper lo observa en pruebas de escucha informales y propone integrar un encoder de prosodia (Skerry-Ryan et al., Global Style Tokens) o entrenar sobre pares referencia/target distintos del mismo hablante.

Es lo peor de ambos mundos: no transfiere el estilo del objetivo de forma útil y controlable, pero sí filtra la prosodia de la muestra de referencia de forma impredecible. **La consecuencia práctica**: si se clona la voz de alguien desde 5 segundos y se le hace leer un párrafo, el resultado suena como esa persona **leyendo con la prosodia genérica del corpus TTS**, no como esa persona hablando. Para un oyente que conoce al hablante esa diferencia es enorme, y es lo que hacía que el clon de 2018 "casi funcionara" sin engañar a un familiar.

El **acento** es un caso particular del mismo problema, y el paper lo reconoce por separado: *"una limitación adicional está en la incapacidad del modelo de transferir acentos"*, con la causa a la vista en los datos (encoder entrenado solo sobre inglés norteamericano, sintetizador de VCTK sobre acentos británicos). Los comentarios de los raters lo confirman: *"nuestro modelo a veces produjo un acento distinto al del ground truth, lo que llevó a puntajes más bajos"*.

### Otras limitaciones

- **Capacidad del embedding.** Los 256 números saturan a los 5 segundos de referencia. Aprovechar más audio requiere **adaptación de parámetros**, no más contexto: el paper es explícito en que *"modelar la variación de hablante usando un vector de baja dimensión limita la capacidad de aprovechar grandes cantidades de habla de referencia"*.
- **Dependencia de la calidad del audio de referencia, sin medir.** El paper afirma robustez al ruido —y de hecho al encoder le pasan el audio sin denoisar—, pero **no hay ningún experimento controlado que varíe el SNR, el códec o la reverberación de la referencia**. La robustez es un argumento de diseño (el corpus de entrenamiento era ruidoso), no un resultado medido. El único eje de la referencia que sí se estudia es la duración.
- **Costo de inferencia del WaveNet.** WaveNet autorregresivo genera **muestra por muestra**: a 24 kHz son 24 000 pasos secuenciales por segundo de audio. Ninguna de las tres primeras etapas importa frente a eso. Es la razón por la que ninguna implementación práctica usó WaveNet tal cual: se reemplazó por WaveRNN primero, y por HiFi-GAN, Parallel WaveGAN o BigVGAN después, todos no autorregresivos.
- **Un solo idioma.** Encoder sobre inglés norteamericano, VCTK inglés británico, LibriSpeech inglés estadounidense. No hay evaluación cross-lingüe, y dado el hallazgo sobre acentos es razonable esperar degradación fuerte en otro idioma —tanto por el encoder como por el sintetizador—. Los propios autores retomaron este hilo un año después.
- **Sesgo demográfico no auditado.** 18K hablantes de *voice search* en EE. UU. no son una muestra uniforme de la variación vocal humana. No hay desglose por edad, variedad dialectal ni características vocales atípicas — y dado que el modelo se propone como tecnología de accesibilidad para personas que perdieron la voz, la ausencia de evaluación sobre **habla disártrica o atípica** es una brecha relevante para el caso de uso declarado.
- **Varianza por hablante grande.** El apéndice lo documenta: en VCTK el hablante "p240" queda casi indistinguible de su ground truth, mientras "p260" queda **medio punto completo** por debajo. El MOS agregado esconde que el sistema funciona muy bien para algunas voces y mediocre para otras, sin que se sepa qué predice cuál.

## El problema ético y de seguridad

El paper **sí** aborda el tema, en la introducción y en la sección de verificación, y conviene reconocerlo porque en 2018 no era la norma: *"es también importante notar el potencial de mal uso de esta tecnología, por ejemplo **suplantando la voz de alguien sin su consentimiento**. Para abordar preocupaciones de seguridad... **verificamos que las voces generadas por el modelo propuesto pueden distinguirse fácilmente de voces reales**"*.

Esa verificación es un experimento concreto: expandieron el conjunto de hablantes enrolados con **diez versiones sintéticas de los diez hablantes reales de LibriSpeech**, formando una tarea de discriminación entre veinte voces, y obtuvieron **EER de 2.86 %**. La lectura del paper: *"mientras el habla sintética tiende a estar cerca del hablante objetivo (similitud coseno > 0.6), es casi siempre **aún más cercana a otras utterances sintéticas del mismo hablante** (similitud > 0.7). De esto podemos concluir que el modelo propuesto puede generar habla que se parece al hablante objetivo, **pero no lo suficientemente bien como para ser confundible con un hablante real**."*

**El problema con esa mitigación** es que es un argumento sobre el estado del arte del momento, no sobre una propiedad estructural. La detectabilidad venía de artefactos del vocoder de 2018 y de la brecha de similitud, y ambas cosas se cerraron en los años siguientes. El método se transfirió íntegro; la garantía de seguridad, no.

**El vector de fraude concreto** es directo. Cinco segundos de audio es el umbral operativo, y dos segundos ya están cerca del óptimo. Cinco segundos de la voz de una persona se obtienen de un mensaje de voz, de un video en redes sociales o de una llamada de "¿aló?, ¿aló?". De ahí salen las dos aplicaciones documentadas desde 2019 en adelante: **fraude por suplantación en llamadas** —el esquema de "emergencia familiar" o la autorización de transferencias por voz— y **evasión de biometría de voz** en sistemas de autenticación telefónica del tipo "mi voz es mi contraseña", que la banca desplegó masivamente en la década de 2010.

{{< concept-alert type="recordar" >}}
**La ironía central.** El mismo componente que la industria usa para **verificar** identidad por voz —un d-vector entrenado con GE2E, exactamente lo que se despliega en verificación de hablante en producción— es el que aquí permite **suplantarla**.

La ecuación no tiene solución dentro de este paradigma: un espacio de embeddings suficientemente bueno para discriminar hablantes es, por construcción, un espacio suficientemente informativo para condicionar un generador. **La calidad del verificador acota por abajo la calidad del clonador**, y mejorar la biometría de voz mejora, gratis, la capacidad de vulnerarla.
{{< /concept-alert >}}

Las dos líneas de defensa que se desarrollaron después están fuera del paper y conviene enunciarlas sin exageración. La primera es la **detección de deepfakes de voz** (*anti-spoofing*), cuyo marco de referencia es la serie de desafíos **ASVspoof** (2015, 2017, 2019, 2021, 2024), con pistas de *logical access* —habla sintética y convertida— y *physical access* —replay—. Su dificultad conocida es la **generalización a ataques no vistos**: los detectores funcionan bien contra los sintetizadores con los que fueron entrenados y se degradan contra métodos nuevos. Es una carrera armamentista, no un problema resuelto. La segunda son las **marcas de agua de audio**, que insertan una señal imperceptible en el audio generado para que sobreviva a compresión y edición y permita atribución; **AudioSeal** (Meta, 2024) es el trabajo más citado en marcas localizadas y robustas para habla generada. Su limitación es estructural: una marca de agua solo cubre a los generadores que **eligen** ponerla, así que protege contra el uso descuidado de servicios comerciales, no contra alguien que corre un modelo abierto en su propia máquina.

En el plano regulatorio, la FCC estadounidense declaró en febrero de 2024 que las llamadas robotizadas con voces generadas por IA caen bajo la TCPA —ilegales sin consentimiento previo—, y el AI Act de la UE impone obligaciones de transparencia para contenido sintético. Ninguna de las dos es una defensa técnica; son cambios en el costo esperado para el atacante. Del lado de la industria, los servicios comerciales de clonación exigen hoy verificación de consentimiento del hablante.

La postura razonable sobre el paper: hizo lo correcto para 2018 —nombró el riesgo, corrió un experimento de verificación y publicó el resultado—. Lo que no podía hacer, y no hizo, es garantizar que la propiedad medida sobreviviera al progreso del campo. Ese es el patrón general de las mitigaciones basadas en "el output es detectable": son válidas hasta que dejan de serlo, y no hay forma de saber cuándo.

## Por qué importa hoy

**Continuación directa por los mismos autores.** El equipo siguió sobre la misma pila casi de inmediato: **Translatotron** (Jia et al., Interspeech 2019) hace traducción habla-a-habla directa, sin ASR ni texto intermedio, **usando el speaker encoder de este paper para preservar la voz original en el idioma destino** —la materialización de la aplicación que la introducción anunciaba—; y *Learning to Speak Fluently in a Foreign Language* (Zhang, Weiss, Jia et al., Interspeech 2019) ataca la limitación monolingüe con clonación de voz entre idiomas.

**La implementación abierta que lo popularizó.** `Real-Time-Voice-Cloning` (Corentin Jemine, 2019), desarrollada como tesis de maestría en la Universidad de Lieja, es una reimplementación en PyTorch de los tres componentes —encoder GE2E, Tacotron condicionado, y **WaveRNN en lugar de WaveNet** para lograr velocidad práctica—. Se volvió uno de los repositorios de audio más populares de GitHub, y es la razón por la que el acrónimo "SV2TTS" existe: **no aparece en el paper**, se lo puso la comunidad a partir de ese repo. Su rol histórico es doble y hay que decir las dos partes: **democratizó** la clonación de voz para investigación y accesibilidad, y **democratizó** la clonación de voz para todo lo demás. Es el momento en que el vector de ataque descrito arriba pasó de "posible para Google" a "posible para cualquiera con una GPU".

**La línea de sucesión.**

| Sistema | Año | Qué cambió respecto de SV2TTS |
|---|---|---|
| **YourTTS** (Casanova et al., ICML 2022) | 2022 | Sobre VITS (flujos normalizantes más entrenamiento adversarial, end-to-end texto→onda). Zero-shot **multilingüe** y con conversión de voz. **Elimina el vocoder separado** |
| **VALL-E** (Microsoft, 2023) | 2023 | Cambio de paradigma completo: **modelo de lenguaje sobre tokens de códec neuronal**. 60 000 horas de entrenamiento, prompt acústico de 3 segundos, preserva **emoción y entorno acústico** de la referencia |
| **XTTS** (Coqui) | 2023-24 | Zero-shot masivamente multilingüe (17 idiomas), abierto. Fue el caballo de batalla de la comunidad open-source |
| **ElevenLabs, PlayHT, Resemble.ai** | 2022– | Comercialización, con calidad indistinguible para un oyente casual y controles de estilo y emoción |
| **VALL-E 2, NaturalSpeech 3, Voicebox, F5-TTS** | 2023– | Paridad humana reclamada en benchmarks; difusión y *flow matching* **no autorregresivos**, que resuelven además el problema de velocidad |

### Qué cambió con los modelos de lenguaje sobre tokens de audio

Este contraste explica **por qué** la limitación de "timbre sí, prosodia no" desapareció. SV2TTS es una cadena de tres etapas con un **cuello de botella explícito**: toda la identidad debe caber en $\mathbf{e} \in \mathbb{R}^{256}$, y la representación intermedia (el mel) está diseñada a mano. VALL-E, en cambio, codifica el audio de referencia como **tokens acústicos discretos** con un códec neuronal, y los pasa como **prefijo del contexto** de un Transformer autorregresivo que genera los tokens de salida.

Tres diferencias importan:

1. **Desaparece el cuello de botella de dimensión fija.** En SV2TTS la identidad se comprime a 256 números, y ahí está el techo duro que hace saturar la referencia a los 5 segundos. En VALL-E el "embedding" **es la secuencia de tokens acústicos de la referencia misma**: no hay compresión, hay *in-context learning*. Más referencia significa más contexto, y la atención decide qué usar.
2. **La condición ya no es solo la identidad, es todo el contexto acústico.** Como el prompt son tokens acústicos crudos, arrastra **prosodia, emoción, acento y hasta las características del entorno de grabación**. La limitación central de SV2TTS se resuelve **por construcción**, no porque alguien haya modelado la prosodia explícitamente: se resuelve al eliminar el cuello de botella que la descartaba. El precio simétrico: **se pierde el desacople**. En SV2TTS se puede tomar el timbre de A con la prosodia genérica del corpus; en VALL-E el estilo del prompt viene con el paquete, y controlarlo por separado vuelve a ser un problema abierto.
3. **La escala reemplaza al diseño arquitectónico.** SV2TTS ganaba generalización mediante **una elección de arquitectura** —desacoplar componentes para poder usar 18K hablantes sin transcribir—; VALL-E la gana con **60 000 horas de datos débilmente supervisados**. Es la misma transición que ocurrió en NLP y en visión: los sesgos inductivos cuidadosamente diseñados ceden ante el pre-entrenamiento a escala.

**Lo que sobrevive.** El concepto de **speaker embedding como condicionamiento** sigue vivo en todas partes: conversión de voz, separación de hablantes condicionada, diarización, TTS controlable. Y sobrevive como **métrica**: la similitud coseno de speaker embedding es hoy el estándar objetivo para reportar calidad de clonación —precisamente el protocolo de este paper, convertido en benchmark de facto—. Sobre todo sobrevive la tesis: **entrenar componentes por separado sobre los datos que a cada uno le convienen, en lugar de exigir un dataset que satisfaga a todos**.

La forma correcta de leer SV2TTS hoy no es como una arquitectura a implementar —hay reemplazos mejores para cada una de sus tres etapas— sino como el paper que estableció que **el condicionamiento por speaker embedding funciona en zero-shot, y que la diversidad de hablantes del encoder, no la calidad del corpus TTS, es la variable que gobierna la generalización a voces nuevas**. Ese resultado sobrevivió a todas las arquitecturas que lo reemplazaron.

## En la clase 39

La [Clase 39](/clases/clase-39) toca este trabajo desde dos ángulos distintos, y conviene separarlos.

### Hilo 1: data augmentation con datos sintéticos

La clase advierte que *"el uso de técnicas de síntesis de sonido es una estrategia conveniente, pero el desempeño sobre datos reales puede ser pobre si el modelo se entrena solo con datos sintéticos: hace falta fine-tuning sobre datos reales"*. SV2TTS es el ejemplo canónico de la parte conveniente: un TTS multi-hablante zero-shot convierte **texto** en **pares (audio, transcripción) perfectamente alineados**, en cantidad arbitraria, con la identidad de hablante que uno elija. Para ASR de bajos recursos parece la solución perfecta —hay muchísimo texto y muy poco audio transcrito—, y para *keyword spotting* o comandos de voz la promesa es todavía más directa. Es la contraparte "generativa" del enfoque de composición de escenas de [Scaper](/papers/scaper-salamon-2017), que fabrica *soundscapes* etiquetados mezclando eventos reales sobre fondos reales; véase [data augmentation de audio](/fundamentos/data-augmentation-de-audio).

**Por qué falla entrenando solo con sintético: la brecha de dominio.** Conviene descomponerla en cuatro ejes, y lo notable es que **SV2TTS provee la evidencia de tres de ellos dentro de su propio paper**:

| Eje de la brecha | Qué falta en el habla sintética | Evidencia en SV2TTS |
|---|---|---|
| **Diversidad de hablantes** | El TTS genera desde un espacio aprendido de ~1.2K hablantes: más pobre y más *suave* que la variación humana real | La ablación del encoder: la calidad depende críticamente de cuántos hablantes vio. Con 98 hablantes, el sintetizador de VCTK no transfiere identidad fuera de dominio (similitud 1.82) |
| **Diversidad prosódica** | La prosodia es genérica y promediada; falta variación de ritmo, énfasis, duda, velocidad | El modelo transfiere timbre, no estilo. Toda la variación prosódica del hablante objetivo se pierde |
| **Realismo acústico / artefactos** | El vocoder deja una firma sistemática. El audio es "demasiado limpio": sin ruido de fondo real, sin reverberación de sala, sin efectos de canal ni códec | El habla sintética forma **clusters propios y separables** de los reales en el espacio de embeddings; EER de 2.86 % en la tarea real-vs-sintético |
| **Realismo lingüístico** | El texto de entrada es escrito: sin disfluencias, sin repeticiones, sin habla espontánea | No abordado en el paper |

Los tres primeros ejes son exactamente los que el paper mide como **ventajas** de su sistema, y por eso son tan buena evidencia en contra del uso ingenuo: **si un verificador de hablante distingue habla real de sintética con EER de 2.86 %, un modelo de ASR entrenado solo sobre habla sintética está aprendiendo la distribución equivocada** — va a sobreajustar los artefactos del generador, no el habla. Y hay un cierre elegante: **el propio paper tropieza con esta brecha internamente**, cuando tuvo que entrenar el vocoder de LibriSpeech sobre **espectrogramas predichos** en vez de reales, porque el vocoder entrenado con mels reales no funcionaba bien con mels sintéticos. Misma lección en miniatura: un modelo aguas abajo entrenado sobre datos reales se degrada cuando en inferencia recibe datos sintéticos, y la solución es alinear las distribuciones.

**Qué dice la literatura de ASR.** Dos referencias representativas. **Rosenberg, Zhang, Ramabhadran, Jia, Moreno, Wu y Wu**, *Speech Recognition with Augmented Synthesized Speech* (ASRU 2019) —nótese que **Ye Jia y Yu Zhang son coautores**: es el mismo equipo aplicando su tecnología a ASR— concluye que **sí se logran mejoras aumentando los datos con material sintetizado, pero permanece una brecha sustancial** entre reconocedores entrenados sobre habla humana y sobre habla sintetizada. Y **Rossenbach, Zeyer, Schlüter y Ney**, *Generating Synthetic Audio Data for Attention-Based Speech Recognition Systems* (ICASSP 2020), reporta hasta **33 % de mejora relativa en WER** en un escenario de bajos recursos, cerrando más de la mitad de la brecha respecto de un experimento oráculo — con dos hallazgos muy pertinentes: el sistema con Global Style Tokens superó claramente al que usaba i-vectors (**la calidad de la representación de hablante del TTS determina cuánto sirve la augmentation**), y las mejoras por datos sintéticos resultaron **mayormente aditivas** con las de SpecAugment.

Síntesis honesta de qué funciona y qué no:

| Práctica | Veredicto |
|---|---|
| Entrenar **solo** con habla sintética | **No funciona** para ASR de propósito general. La brecha es sustancial y consistente en toda la literatura |
| **Mezclar** sintético con real (el real domina el mix) más fine-tuning final sobre real | **Funciona.** Es la receta estándar, y es exactamente lo que afirma la clase |
| Sintético para **cubrir vocabulario nuevo o raro** (nombres propios, términos de dominio, comandos) | **El caso de uso más sólido**: cuando el problema es que ciertas palabras nunca aparecen en el audio de entrenamiento, generarlas ataca la carencia real |
| Sintético para **bajos recursos** en un idioma sin datos | **Funciona con matices.** Hay un problema del huevo y la gallina: el TTS también necesita datos para existir |
| **Maximizar la diversidad** del sintético (muchos hablantes, prosodia variada) | **Es el factor determinante.** Toda la literatura converge en que lo limitante es la diversidad, no el volumen |
| Aplicar **SpecAugment, ruido, reverberación o códec** *sobre* el audio sintético | **Ayuda.** Rompe parcialmente los artefactos de "demasiado limpio" del vocoder y acerca las distribuciones |

El puente conceptual: la afirmación de la clase es la formulación aplicada de un principio general de *domain adaptation*. Cuando $p_{\text{train}} \neq p_{\text{test}}$, el modelo minimiza el riesgo empírico sobre la distribución equivocada. El pre-entrenamiento sobre sintético sirve para aprender la estructura **compartida** —fonética, léxico, alineamiento acústico-fonético—, que es lo que abunda en el sintético; el fine-tuning sobre real corrige la estructura **específica del dominio** —canal, ruido, prosodia espontánea—, que es exactamente lo que el sintético no tiene. Es la misma estructura de argumento que justifica ImageNet → dataset objetivo en visión, o el pre-entrenamiento en simulación en robótica, donde se lo llama *sim-to-real gap*.

### Hilo 2: transferencia entre tareas de audio

El segundo ángulo es más conceptual, y es el que hace de SV2TTS un caso de estudio y no solo una herramienta. Es [transfer learning](/fundamentos/transfer-learning) entre tareas de **naturaleza opuesta**:

$$\underbrace{\text{Verificación de hablante}}_{\text{discriminativa},\ p(\text{misma persona} \mid x_1, x_2)} \;\longrightarrow\; \underbrace{\text{Síntesis de habla}}_{\text{generativa},\ p(\text{audio} \mid \text{texto},\, \mathbf{e})}$$

Lo transferido no son features de bajo nivel ni pesos de una red compartida: es **un espacio de representación completo, con su geometría, congelado**. El sintetizador nunca ajusta el encoder. Y el paper es transparente en que esto no estaba garantizado — la red *"no está optimizada directamente"* para la síntesis y sin embargo produce un embedding *"directamente adecuado"* para condicionarla.

El principio general que ilustra es el mismo que la clase invoca al hablar de **pre-entrenar en idiomas con más datos transcritos**: *la tarea de pre-entrenamiento no tiene que parecerse a la tarea objetivo; tiene que forzar al modelo a codificar la misma información latente*. Verificación y síntesis son opuestas en su forma, pero ambas dependen de la misma variable latente —**quién es el hablante**—: una la debe leer, la otra la debe escribir. Comparten el objeto, no la operación. La misma lógica opera en los otros ejemplos que la clase menciona: pre-entrenar ASR en un idioma con muchos datos y hacer fine-tuning en uno con pocos funciona porque la estructura acústico-fonética de bajo nivel es en buena medida universal (XLSR, Whisper multilingüe); wav2vec 2.0 y HuBERT transfieren desde un objetivo autosupervisado que no tiene ninguna relación de forma con ASR; Whisper transfiere desde supervisión débil a escala masiva.

Y la restricción práctica que SV2TTS ilustra mejor que ningún otro trabajo del [dominio audio](/dominios/audio): **el desacople de tareas es lo que permite desacoplar los datasets, y desacoplar los datasets es lo que permite escalar**.

## Erratas y matices

### La fecha: el slide dice 2019, el paper es de 2018

El material de la clase cita este trabajo como "Jia et al., 2019". El pie de la primera página del PDF dice textualmente *"32nd Conference on Neural Information Processing Systems (**NeurIPS 2018**), Montréal, Canada"*, y el identificador `arXiv:1806.04558` codifica **junio de 2018** (`YYMM = 1806`) como fecha del v1. El "2019" viene de la marca lateral del PDF, `arXiv:1806.04558v4 [cs.CL] 2 Jan 2019`, que corresponde a una **revisión posterior a la conferencia**. A eso se suman dos confusores razonables: NeurIPS 2018 se celebró en diciembre, casi en el límite del año, y Ye Jia tiene varios papers de 2019 sobre temas adyacentes (Translatotron y cross-language voice cloning, ambos en Interspeech 2019).

**La cita correcta es 2018:**

> Jia, Y., Zhang, Y., Weiss, R. J., Wang, Q., Shen, J., Ren, F., Chen, Z., Nguyen, P., Pang, R., Lopez Moreno, I., & Wu, Y. (2018). *Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis*. En **Advances in Neural Information Processing Systems 31 (NeurIPS 2018)**. arXiv:1806.04558.

Es una errata sin consecuencias sobre el contenido, pero vale la pena tenerla registrada.

### "El modelo funciona igual de bien con hablantes no vistos" — falso

Es la lectura errónea más común, y viene de mirar solo la tabla de naturalidad, donde efectivamente los no vistos superan a los vistos:

| | Naturalidad | Similitud |
|---|---|---|
| VCTK, vistos → no vistos | 4.07 → **4.20** (+0.13) | 4.22 → **3.28** (−0.94) |
| LibriSpeech, vistos → no vistos | 3.89 → **4.12** (+0.23) | 3.28 → **3.03** (−0.25) |

**El habla suena igual de bien; se parece bastante menos al objetivo.** Son cosas distintas, y la evaluación las separó deliberadamente —los raters de similitud reciben la instrucción explícita de no juzgar la calidad de audio—. Además, la explicación del "no vistos > vistos" en naturalidad **no es que el modelo generalice mejor**: el paper la atribuye a *"la utterance de referencia elegida aleatoriamente para cada hablante, que a veces contiene prosodia despareja y no neutra"*, interactuando con la fuga de prosodia del entrenamiento. Es un artefacto del sorteo de la referencia; con otros hablantes de evaluación y otras referencias el orden podría invertirse.

### Otros matices que se citan mal

- **"Bastan 5 segundos de audio"** es correcto pero incompleto: **2 segundos ya están cerca del óptimo**, y **más de 5 no ayuda** —a 10 segundos la similitud baja de 3.28 a 3.18 y la EER sube de 10.46 % a 11.50 %—. No es "5 segundos es el mínimo": es "5 segundos es el techo".
- **"El sistema es zero-shot"** es correcto en cuanto a que no actualiza parámetros, pero la generalización **fuera de dominio** puede ser muy mala (similitud 1.82 para el sintetizador de VCTK sobre hablantes de LibriSpeech). El zero-shot funciona dentro de un rango acústico y de acento razonablemente cercano al de entrenamiento.
- **"El vocoder es agnóstico al hablante"** requiere los tres matices ya discutidos: no está *condicionado* explícitamente, pero sí debe *entrenarse* sobre muchos hablantes, se entrena uno por corpus, y en LibriSpeech hubo que entrenarlo sobre espectrogramas predichos.
- **"El encoder es Tacotron / es parte del TTS"** — no. El speaker encoder es una red LSTM **completamente separada**, entrenada antes, con parámetros **congelados** durante el entrenamiento del sintetizador.
- **La fórmula de GE2E no está en este paper**; se cita de Wan et al. (ICASSP 2018). Atribuir la formulación de la pérdida a SV2TTS es un error de atribución.
- **VCTK "de 109 hablantes"** entrena en realidad con **98**: once quedan *held out*.
- **El nombre "SV2TTS" no aparece en el paper.** Se lo puso la comunidad a partir de la implementación abierta de Jemine.

## Notas y enlaces

- **Fundamentos relacionados**:
  - [Data augmentation de audio](/fundamentos/data-augmentation-de-audio) — el hilo de datos sintéticos de la clase 39
  - [Transfer learning](/fundamentos/transfer-learning) — el mecanismo que el paper explota entre tareas opuestas
  - [Metric learning](/fundamentos/metric-learning) — la familia a la que pertenece GE2E
  - [Triplet loss](/fundamentos/triplet-loss) — la alternativa que GE2E reemplaza al comparar contra centroides
  - [Modelos generativos](/fundamentos/modelos-generativos) — marco para leer el experimento de hablantes ficticios
- **Papers relacionados**:
  - [WaveNet — van den Oord et al. 2016](/papers/wavenet-oord-2016) — el vocoder de la tercera etapa, y el cuello de botella de velocidad del sistema
  - [LibriSpeech — Panayotov et al. 2015](/papers/librispeech-panayotov-2015) — corpus del sintetizador principal y del que salen las cifras más citadas
  - [Scaper — Salamon et al. 2017](/papers/scaper-salamon-2017) — la otra vía para fabricar datos de audio etiquetados, por composición en vez de por generación
- **Dominio**: [Audio](/dominios/audio)
- **Clase**: [Clase 39 — datos sintéticos y transferencia en audio](/clases/clase-39)
- **Enlaces externos**: [muestras de audio del paper](https://google.github.io/tacotron/publications/speaker_adaptation) · [arXiv:1806.04558](https://arxiv.org/abs/1806.04558)
