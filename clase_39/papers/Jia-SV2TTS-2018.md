# Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis (SV2TTS) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Autores:** Ye Jia\*, Yu Zhang\*, Ron J. Weiss\* (contribución igualitaria), Quan Wang, Jonathan Shen, Fei Ren, Zhifeng Chen, Patrick Nguyen, Ruoming Pang, Ignacio Lopez Moreno, Yonghui Wu. Todos en **Google Inc.** (encabezado del paper, p. 1).
- **Venue:** **NeurIPS 2018** — el pie de página de la primera página dice literalmente *"32nd Conference on Neural Information Processing Systems (NeurIPS 2018), Montréal, Canada"*.
- **arXiv:** `1806.04558` — el identificador codifica **junio de 2018** como fecha del v1. La marca lateral del PDF que se está analizando es `arXiv:1806.04558v4 [cs.CL] 2 Jan 2019`, es decir, una **revisión posterior a la conferencia**.
- **Nombre coloquial:** **SV2TTS** (*Speaker Verification to Text-To-Speech*), que no aparece en el paper; se lo puso la comunidad, sobre todo a partir de la implementación abierta `Real-Time-Voice-Cloning`.
- **Página de muestras:** `https://google.github.io/tacotron/publications/speaker_adaptation` (nota al pie 1, Sección 2).

El sistema tiene **tres redes entrenadas de forma independiente**: un *speaker encoder* entrenado en verificación de hablante sobre audio ruidoso y **sin transcribir** de miles de hablantes; un sintetizador seq2seq basado en **Tacotron 2** que produce un mel-espectrograma a partir de texto, condicionado por el embedding del hablante; y un vocoder **WaveNet** autorregresivo que convierte el mel en forma de onda. El objetivo es **zero-shot**: unos pocos segundos de audio de referencia de un hablante nuevo bastan para sintetizar en su voz **sin actualizar ningún parámetro**.

Cifras ancla (Tablas 1 y 2 del paper, MOS con intervalos de confianza al 95%):

| Métrica | VCTK Seen | VCTK Unseen | LibriSpeech Seen | LibriSpeech Unseen |
|---|---|---|---|---|
| **Naturalidad** — modelo propuesto | 4.07 ± 0.06 | **4.20 ± 0.06** | 3.89 ± 0.06 | **4.12 ± 0.05** |
| **Naturalidad** — ground truth | 4.43 ± 0.05 | 4.49 ± 0.05 | 4.49 ± 0.05 | 4.42 ± 0.07 |
| **Similitud** — modelo propuesto | 4.22 ± 0.06 | **3.28 ± 0.07** | 3.28 ± 0.08 | **3.03 ± 0.09** |
| **Similitud** — ground truth (mismo hablante) | 4.67 ± 0.04 | 4.67 ± 0.04 | 4.33 ± 0.08 | 4.33 ± 0.08 |

Dos lecturas que hay que separar con cuidado, porque se citan mal todo el tiempo:

1. **La naturalidad no se degrada en hablantes no vistos.** Es más: **sube** (4.07 → 4.20 en VCTK; 3.89 → 4.12 en LibriSpeech). El paper explica el porqué en la Sección 3.1 y no es magia — es un artefacto del muestreo de la utterance de referencia (ver §14).
2. **La similitud sí se degrada, y bastante.** En VCTK cae de **4.22 a 3.28** (casi un punto entero de MOS) al pasar de hablantes vistos a no vistos. Esa es la brecha real del zero-shot cloning en 2018.

La métrica objetiva complementaria es la **tasa de igual error de verificación de hablante (SV-EER)** medida sobre el audio sintetizado con un encoder de evaluación independiente: **5.08%** para el sintetizador de LibriSpeech contra **0.93%** del audio real (Tabla 4). El habla sintética se parece al hablante objetivo, pero un verificador la distingue del habla real con facilidad.

---

## 2. Contexto: el TTS multi-hablante antes de 2018

### 2.1. De dónde venía el campo

Para 2018 el TTS neuronal ya había resuelto el problema de la **naturalidad para un hablante**. Tacotron (Wang et al., Interspeech 2017) y Tacotron 2 (Shen et al., ICASSP 2018) habían mostrado que un seq2seq con atención podía ir de grafemas a mel-espectrograma sin ningún feature lingüístico intermedio, y que enchufándole WaveNet como vocoder se alcanzaba naturalidad cercana a la humana. El paper lo dice explícitamente en la introducción: Tacotron 2 combina "la prosodia de Tacotron con la calidad de audio de WaveNet", pero **"solo soportaba un hablante"**.

El costo de eso era brutal en datos. La introducción cuantifica el estado del arte: sintetizar habla natural "requiere entrenar sobre una gran cantidad de pares habla-transcripción de alta calidad, y soportar muchos hablantes normalmente usa **decenas de minutos de datos de entrenamiento por hablante**" (referencia a Deep Voice 2, Gibiansky et al., NeurIPS 2017). Grabar horas de audio limpio de estudio, con transcripción alineada, para cada voz que se quiera soportar, no escala: es un proyecto de producción de audio por voz.

### 2.2. Las tres familias previas y por qué ninguna cerraba el problema

El paper hace una revisión ordenada en la introducción, y vale la pena entenderla como una escalera:

| Enfoque | Ejemplos | Qué necesita para una voz nueva | Limitación |
|---|---|---|---|
| **Tabla de embeddings aprendida** | Deep Voice 2 (Gibiansky et al. 2017), Deep Voice 3 (Ping et al., ICLR 2018, hasta 2,400 hablantes de LibriSpeech) | Reentrenar: el hablante debe estar en el set de entrenamiento | **Conjunto cerrado**. Solo sintetiza voces vistas durante el entrenamiento |
| **Adaptación / fine-tuning** | VoiceLoop (Taigman et al., ICLR 2018) | "Decenas de minutos de habla de *enrollment* **y transcripciones**" para el hablante nuevo | Requiere transcripciones y cientos de iteraciones de backprop por voz |
| ***Few-shot* con encoder de hablante** | Neural Voice Cloning (Arik et al. 2018), Nachmani et al. 2018 | Unos segundos de audio sin transcribir | **Encoder entrenado conjuntamente** con el sintetizador, sobre el mismo dataset de TTS. La diversidad de hablantes queda acotada por el dataset de TTS |

La tercera familia es la más cercana a SV2TTS y el paper es explícito: *"Nuestro trabajo es el más similar a los modelos de speaker encoding en [Arik et al., Nachmani et al.], excepto que usamos una red **entrenada independientemente** para una tarea de verificación de hablante sobre un dataset grande de audio sin transcribir de decenas de miles de hablantes"*. Nachmani et al. entrenaban todo junto, con pérdida de triplete y de consistencia cíclica. Arik et al. comparaban adaptación por fine-tuning contra *speaker encoding* y encontraban que el encoder era más eficiente en datos y en cómputo.

También cita a Doddipatla et al. (Interspeech 2017), que ya usaba d-vectors de un clasificador de hablante pre-entrenado para condicionar un TTS. La diferencia que SV2TTS reclama: (i) sintetizador end-to-end sin features lingüísticos intermedios, (ii) un encoder que **no está limitado a un conjunto cerrado de hablantes** (clasificador vs. verificador), y (iii) el análisis cuantitativo de cuántos hablantes hacen falta — "encontramos que la transferencia zero-shot requiere entrenar sobre **miles** de hablantes, muchos más de los que se usaron en [Doddipatla]".

### 2.3. Qué es zero-shot voice cloning, con precisión

La definición operativa del paper (Sección 1): *"abordamos específicamente un escenario de aprendizaje **zero-shot**, donde unos pocos segundos de audio de referencia **sin transcribir** de un hablante objetivo se usan para sintetizar habla nueva en la voz de ese hablante, **sin actualizar ningún parámetro del modelo**"*.

Los tres calificadores importan y cada uno elimina un costo:

- **Sin transcribir** → no hace falta un pipeline de anotación; sirve cualquier grabación.
- **Sin actualizar parámetros** → la adaptación es un *forward pass*, del orden de milisegundos, no cientos de iteraciones de backprop. Se puede hacer en línea, por request, en un servidor multi-tenant.
- **Pocos segundos** → el Apéndice C mide esto: la calidad **satura alrededor de los 5 segundos** en VCTK, y con **2 segundos** ya se está cerca del óptimo.

Las motivaciones que el paper declara en la introducción son de accesibilidad ("restaurar la capacidad de comunicarse naturalmente a usuarios que perdieron su voz y por lo tanto **no pueden proveer muchos ejemplos nuevos de entrenamiento**"), traducción habla-a-habla preservando la voz, y TTS en escenarios de bajos recursos. Y, en la misma introducción, la advertencia: *"es importante notar el potencial de mal uso de esta tecnología, por ejemplo suplantar la voz de alguien sin su consentimiento"* (ver §11).

---

## 3. La contribución central: desacoplar la identidad del hablante del contenido

### 3.1. El argumento en una frase

De la introducción: *"Nuestro enfoque es **desacoplar el modelado del hablante de la síntesis de habla** entrenando independientemente una red de embedding discriminativa de hablantes que capture el espacio de características de hablante, y entrenando un modelo TTS de alta calidad sobre un dataset más pequeño, condicionado por la representación aprendida por la primera red. Desacoplar las redes permite entrenarlas sobre **datos independientes**, lo que reduce la necesidad de obtener datos multi-hablante de alta calidad."*

Ese es el paper entero. Todo lo demás es la instanciación y la evidencia.

### 3.2. Los tres componentes y sus contratos

```
audio de referencia ──> [Speaker Encoder] ──> e ∈ R^256, ||e||₂ = 1
                                                    │
texto (grafemas/fonemas) ──> [Encoder de texto] ──> concat ──> [Atención] ──> [Decoder] ──> mel (80 ch)
                                                                                              │
                                                                                    [WaveNet] ──> waveform
```

(Figura 1 del paper, "Model overview. Each of the three components are trained independently.")

| Componente | Base | Dataset de entrenamiento | Qué necesita ese dataset |
|---|---|---|---|
| **Speaker encoder** | Wan et al., ICASSP 2018 (GE2E) | Corpus propietario de *voice search*: 36M utterances, 18K hablantes | Muchos hablantes; **audio ruidoso, con reverberación, sin transcripción**; solo etiquetas de identidad anonimizadas |
| **Sintetizador** | Tacotron 2 (Shen et al., ICASSP 2018) | VCTK (44 h, 109 hablantes) o LibriSpeech clean (436 h, 1,172 hablantes) | Pares texto-audio de calidad. **Ninguna etiqueta de identidad** |
| **Vocoder** | WaveNet (van den Oord et al., 2016), 30 capas de convolución dilatada | El mismo corpus del sintetizador (uno por corpus) | Audio de calidad de muchos hablantes |

### 3.3. Por qué entrenar por separado es *la* decisión

Esta es la parte que conviene entender bien porque es donde está el argumento no obvio.

**Los datos ideales para cada componente son mutuamente incompatibles.**

- Para que el encoder generalice a voces nuevas necesita haber visto **muchísimas voces**. Miles, según la evidencia de la Sección 3.5. Y voces reales, con acústica de mundo real.
- Para que el sintetizador produzca audio natural necesita **transcripciones exactas y audio limpio**, porque cualquier ruido en el target se aprende y se reproduce (el paper lo verifica: el sintetizador de LibriSpeech "aprendió a reproducir" parte del ruido de fondo, Sección 3.1).

Un dataset con miles de hablantes, transcrito y grabado en estudio, sencillamente no existe a escala. Un dataset con 18K hablantes de *voice search* sí existe, pero es inservible para entrenar un TTS: no está transcrito y el audio es de teléfono con ruido de calle. El desacople permite **usar cada dataset para lo único en lo que sirve**.

Lo que gana cada componente:

- **El encoder gana diversidad de hablantes** a costo casi nulo. La conclusión del paper: *"el requisito de datos para el speaker encoder es **mucho más barato** que el entrenamiento TTS completo, porque no son necesarias transcripciones, y la calidad de audio puede ser más baja"* (Sección 3.5).
- **El sintetizador gana simplicidad**. No necesita etiquetas de identidad (Sección 2.2: *"No se usan etiquetas explícitas de identidad de hablante durante el entrenamiento"*), y no necesita pérdidas auxiliares. De la conclusión: entrenar los componentes por separado *"simplifica significativamente la configuración de entrenamiento del sintetizador comparado con [Nachmani et al.], ya que no requiere pérdidas adicionales de triplete o contrastivas"*.
- **El sistema gana composicionalidad**. Se puede reemplazar el encoder por uno mejor sin tocar el sintetizador. La Tabla 5 es exactamente ese experimento: cinco encoders distintos contra el mismo sintetizador de LibriSpeech.

**La evidencia de que separar es mejor está en el Apéndice A.** Entrenaron dos baselines conjuntos sobre LibriSpeech Clean, con embedding de 64 dimensiones: uno sin restricción sobre el encoder (análogo a Skerry-Ryan et al.) y otro con una pérdida de discriminación de hablante (proyección lineal → softmax → cross-entropy). Tabla 7:

| Sistema | Emb. dim | Nat. Seen | Nat. Unseen | Sim. Seen | Sim. Unseen |
|---|---|---|---|---|---|
| Entrenamiento conjunto | 64 | 3.72 ± 0.06 | 3.59 ± 0.07 | 2.47 ± 0.08 | 2.44 ± 0.09 |
| Entrenamiento conjunto + speaker loss | 64 | 3.71 ± 0.06 | 3.71 ± 0.06 | 2.82 ± 0.08 | 2.12 ± 0.08 |
| Entrenamiento separado (Tabla 5) | 64 | 3.88 ± 0.06 | 3.73 ± 0.06 | 2.64 ± 0.08 | 2.23 ± 0.08 |
| Tabla de embeddings | 64 | 3.90 ± 0.06 | N/A | 3.70 ± 0.08 | N/A |
| **Modelo propuesto** (encoder sobre 18K hablantes) | **256** | 3.89 ± 0.06 | **4.12 ± 0.05** | 3.28 ± 0.08 | **3.03 ± 0.09** |

El matiz honesto: **a igual dataset (LibriSpeech Clean) y a igual dimensión (64), el entrenamiento separado no gana por mucho** — 3.73 vs 3.71 de naturalidad en unseen, 2.23 vs 2.12 de similitud. La conclusión del apéndice es que el modelo propuesto *"supera significativamente a todos los baselines"*, pero la razón de esa superioridad **no es la separación por sí misma: es que la separación habilita entrenar el encoder sobre 18K hablantes**. Separar no es mejor por elegancia arquitectónica; es mejor porque desbloquea un dataset que de otro modo sería inaccesible.

---

## 4. El encoder de hablante en detalle

### 4.1. Requisitos y arquitectura

De la Sección 2.1: *"Crítico para una buena generalización es el uso de una representación que capture las características de distintos hablantes, y la capacidad de identificar esas características usando solo una señal de adaptación corta, **independientemente de su contenido fonético y del ruido de fondo**."*

Esos son los tres invariantes que la tarea de verificación de hablante impone naturalmente: invarianza al **contenido** (es verificación *text-independent*), invarianza al **canal/ruido** (porque el dataset de entrenamiento es ruidoso), y funcionamiento con **audio corto**.

Arquitectura exacta (Sección 2.1):

- **Entrada:** log-mel espectrogramas de **40 canales**.
- **Cuerpo:** pila de **3 capas LSTM de 768 celdas**, cada una seguida de una **proyección a 256 dimensiones**. (Es la arquitectura LSTMP estándar de Google para reconocimiento y verificación: la proyección recurrente reduce el estado que se realimenta, abarata la matriz recurrente y actúa como cuello de botella.)
- **Salida:** se toma la salida de la capa superior **en el último frame** y se le aplica **normalización $L_2$**. El resultado es el **d-vector** de 256 dimensiones, sobre la superficie de la hiperesfera unitaria.

Que el embedding viva en $\mathbb{S}^{255}$ no es cosmético: es lo que hace que la **similitud coseno sea equivalente al producto punto**, lo que permite muestrear "hablantes ficticios" uniformemente sobre la esfera (Sección 3.6) y lo que le da al espacio una geometría acotada que el sintetizador puede recorrer sin encontrarse regiones degeneradas.

### 4.2. Ventanas: entrenamiento vs inferencia

Un detalle que se pasa por alto y que importa al implementar:

- **Entrenamiento:** el dataset se segmenta en trozos de **1.6 segundos** con su etiqueta de hablante. *"No se usan transcripciones"* (Sección 2.1).
- **Inferencia:** una utterance de longitud arbitraria se parte en **ventanas de 800 ms con 50% de solapamiento**; la red se corre independientemente sobre cada ventana, y las salidas se **promedian y se re-normalizan** para formar el embedding final de la utterance.

Es decir, el embedding de inferencia es $\mathbf{e} = \frac{\bar{\mathbf{e}}}{\|\bar{\mathbf{e}}\|_2}$ con $\bar{\mathbf{e}} = \frac{1}{W}\sum_{w=1}^{W} f(\mathbf{x}_w)$ sobre las $W$ ventanas. El promedio en el espacio de embeddings antes de re-normalizar es un promedio euclidiano proyectado a la esfera; para vectores ya normalizados y razonablemente alineados es una buena aproximación al promedio esférico, y es lo que reduce la varianza que explica por qué más audio de referencia mejora la similitud (Tabla 9).

### 4.3. La pérdida GE2E

El paper **no escribe la fórmula**: dice *"la red se entrena para optimizar una pérdida generalized end-to-end de verificación de hablante, de modo que los embeddings de utterances del mismo hablante tengan **alta similitud coseno**, mientras que los de hablantes distintos queden **lejos** en el espacio de embeddings"*, y cita a Wan, Wang, Papir y Lopez Moreno, *"Generalized end-to-end loss for speaker verification"*, ICASSP 2018 (referencia [22]).

La formulación siguiente **viene del paper de GE2E, no de este PDF** — lo dejo explícito porque la regla es citar la ubicación de cada cifra.

El batch es una matriz de $N \times M$ utterances: $N$ hablantes, $M$ utterances por hablante. Sea $\mathbf{e}_{ji}$ el embedding normalizado de la utterance $i$ del hablante $j$. El centroide del hablante $k$ es

$$\mathbf{c}_k = \frac{1}{M}\sum_{m=1}^{M} \mathbf{e}_{km}$$

y la **matriz de similitud** es la similitud coseno **escalada y desplazada** por dos escalares aprendidos $w > 0$ y $b$:

$$S_{ji,k} = w \cdot \cos(\mathbf{e}_{ji}, \mathbf{c}_k) + b$$

El truco central de GE2E: cuando se compara un embedding contra el centroide de **su propio** hablante, ese centroide se calcula **excluyendo la utterance en cuestión**, para evitar la solución trivial de que el embedding se compare consigo mismo:

$$\mathbf{c}_j^{(-i)} = \frac{1}{M-1}\sum_{m \neq i} \mathbf{e}_{jm}, \qquad
S_{ji,k} = \begin{cases} w\cos(\mathbf{e}_{ji}, \mathbf{c}_j^{(-i)}) + b & k = j \\ w\cos(\mathbf{e}_{ji}, \mathbf{c}_k) + b & k \neq j \end{cases}$$

Sobre esa matriz se define la pérdida, en variante **softmax**:

$$L(\mathbf{e}_{ji}) = -S_{ji,j} + \log \sum_{k=1}^{N} \exp(S_{ji,k})$$

o en variante **contrast**, que penaliza solo al impostor más difícil:

$$L(\mathbf{e}_{ji}) = 1 - \sigma(S_{ji,j}) + \max_{1 \le k \le N,\; k \neq j} \sigma(S_{ji,k})$$

y la pérdida total es $L_G = \sum_{j,i} L(\mathbf{e}_{ji})$.

**Por qué esto importa para TTS y no cualquier pérdida discriminativa sirve igual:**

1. **Es *verificación*, no *clasificación*.** Un softmax sobre 18K identidades produce una capa de salida atada al conjunto de entrenamiento, y el penúltimo layer no tiene ninguna garantía métrica. GE2E optimiza directamente la geometría del espacio: mismo hablante junto, distinto hablante lejos, medido en coseno. Ese espacio **es** la interfaz con el sintetizador, así que optimizar la geometría directamente es optimizar lo que se usa. Es exactamente el mismo argumento de FaceNet frente a un clasificador de identidades.
2. **Compara contra centroides, no contra pares o tripletes.** Con $N \times M$ embeddings se obtienen $N \times M \times N$ comparaciones por batch, sin minería de tripletes. GE2E fue precisamente el trabajo que hizo esto escalable.
3. **La similitud coseno escalada** ($w, b$ aprendidos) desacopla la escala del logit de la geometría del embedding. Los embeddings están en la esfera unitaria, así que $\cos \in [-1,1]$; sin el escalamiento, el softmax sería casi uniforme y los gradientes minúsculos. Es el mismo motivo por el que los métodos de *margin softmax* en reconocimiento facial introducen un factor de escala $s$.

El comentario más interesante del paper sobre este componente (final de la Sección 2.1): *"Aunque la red **no está optimizada directamente** para aprender una representación que capture características de hablante relevantes para la **síntesis**, encontramos que entrenar en una tarea de discriminación de hablante lleva a un embedding que es **directamente adecuado** para condicionar la red de síntesis."* Eso es el resultado sorprendente del paper y merece ser reconocido como tal: no había ninguna razón *a priori* para que un espacio optimizado para decir "¿son la misma persona?" fuera un espacio del que se pueda **generar**. Es una afirmación empírica, y las Tablas 5 y 6 son la evidencia.

---

## 5. El sintetizador

### 5.1. Base y modificaciones

Tacotron 2 (Shen et al., ICASSP 2018): encoder de texto convolucional + BiLSTM, atención sensible a la localización, decoder autorregresivo LSTM que emite frames de mel, post-net convolucional residual, y un *stop token*. SV2TTS lo extiende a multi-hablante *"siguiendo un esquema similar a [Deep Voice 2]"* (Sección 2.2).

Cambios concretos que el paper declara:

- **Entrada fonémica.** *"Mapeamos el texto a una secuencia de fonemas, lo cual lleva a **convergencia más rápida** y a **mejor pronunciación de palabras raras y nombres propios**"*. En las evaluaciones subjetivas usan siempre fonemas, "para controlar la pronunciación".
- **Target espectral:** ventanas de **50 ms**, salto de **12.5 ms**, banco de filtros mel de **80 canales**, seguido de **compresión logarítmica de rango dinámico**. (Nótese la asimetría con el encoder de hablante, que usa 40 canales: son dos front-ends distintos, no compartidos.)
- **Pérdida:** *"extendemos [Tacotron 2] aumentando la pérdida $L_2$ sobre el espectrograma predicho con una pérdida $L_1$ adicional. En la práctica encontramos que esta pérdida combinada es **más robusta sobre datos de entrenamiento ruidosos**"*. Es decir, $\mathcal{L} = \|\hat{Y} - Y\|_2^2 + \|\hat{Y} - Y\|_1$: el término $L_1$ le baja el peso a los outliers, que en audio ruidoso son ruido y no señal.
- **Sin pérdidas auxiliares:** *"En contraste con [Nachmani et al.], no introducimos términos de pérdida adicionales basados en el embedding del hablante."*

### 5.2. Dónde se inyecta el embedding, exactamente

*"Un vector de embedding para el hablante objetivo se **concatena con la salida del encoder del sintetizador en cada paso temporal**"* (Sección 2.2).

Formalmente: si el encoder de texto produce $H = [\mathbf{h}_1, \dots, \mathbf{h}_L]$ con $\mathbf{h}_l \in \mathbb{R}^{d_h}$ (uno por fonema), la secuencia que ve la atención es

$$\tilde{\mathbf{h}}_l = [\mathbf{h}_l \,;\, \mathbf{e}] \in \mathbb{R}^{d_h + 256}, \quad l = 1, \dots, L$$

El mismo $\mathbf{e}$ replicado a lo largo de toda la secuencia de texto.

**Por qué ese punto y no otro.** Hay al menos cuatro alternativas obvias, y conviene ver qué falla en cada una:

| Punto de inyección | Problema |
|---|---|
| Estado inicial del decoder | El decoder es recurrente y autorregresivo sobre secuencias de cientos de frames. La información del hablante **se diluye** con el tiempo: es exactamente el problema de memoria que motivó la atención. El timbre derivaría a lo largo de la utterance |
| Concatenar al input del decoder en cada paso | Funciona, pero deja a la **atención ciega al hablante**: el alineamiento texto↔audio no puede adaptarse a la velocidad de habla ni a la duración fonémica características del hablante |
| Solo al post-net o al vocoder | Es una corrección *post hoc* sobre un mel ya generado como si fuera de un hablante promedio. La estructura de F0 y formantes debe estar en el mel desde su generación, no maquillada después |
| **Concatenar a la salida del encoder de texto** | **Cada vector de contexto que el decoder recibe, en todos los pasos, ya lleva la identidad; y las *keys* de la atención son función del hablante** |

Esa última propiedad es la clave y explica el resultado empírico. El vector de contexto es $\mathbf{c}_t = \sum_l \alpha_{t,l} \tilde{\mathbf{h}}_l$; como todos los $\tilde{\mathbf{h}}_l$ contienen $\mathbf{e}$, se cumple $\sum_l \alpha_{t,l} = 1$ y por lo tanto **$\mathbf{e}$ pasa intacto al decoder en cada paso**, sin depender de dónde esté mirando la atención. Es un canal de condicionamiento persistente y gratuito. Y al mismo tiempo, como las *keys* que la atención puntúa dependen de $\mathbf{e}$, el **alineamiento** —o sea la duración asignada a cada fonema, o sea la velocidad de habla y el ritmo— **puede condicionarse al hablante**. La Sección 2.4 confirma que esto ocurre: *"la velocidad de habla característica también es capturada hasta cierto punto por el embedding del hablante, como se ve en la mayor duración de la señal en la fila inferior comparada con las dos superiores"* (Figura 2).

El paper señala además una **simplificación frente a Deep Voice 2**, que inyectaba embeddings en múltiples sitios de la red: *"En contraste con [Gibiansky et al.], encontramos que **simplemente pasar los embeddings a la capa de atención**, como en la Figura 1, **converge** a través de distintos hablantes."* Un solo punto de inyección basta.

### 5.3. La configuración de entrenamiento y su consecuencia oculta

*"La red se entrena en una configuración de transfer learning, usando un speaker encoder pre-entrenado (**cuyos parámetros están congelados**) para extraer un embedding del **audio objetivo**, es decir, **la señal de referencia del hablante es la misma que el habla objetivo durante el entrenamiento**"* (Sección 2.2).

Esto merece subrayarse porque es la raíz de dos fenómenos que aparecen después:

1. Es lo que permite prescindir de etiquetas de identidad: el "label" de hablante es el propio audio pasado por el encoder congelado.
2. Es también una **fuga**: el modelo aprende que $\mathbf{e}$ fue calculado sobre la utterance exacta que debe reproducir. Cualquier información **prosódica residual** que sobreviva en el embedding se vuelve una señal útil durante el entrenamiento, y el decoder aprende a explotarla. En inferencia, con una referencia distinta al target, esa misma dependencia se convierte en **transferencia de prosodia no deseada** (ver §10 y §14). El propio paper propone el remedio en la Sección 3.1: *"entrenar sobre pares de referencia y target elegidos aleatoriamente del mismo hablante"*.

---

## 6. El vocoder, y por qué no necesita saber quién habla

Sección 2.3, completa y corta: *"Usamos el WaveNet autorregresivo muestra a muestra como vocoder para invertir los mel-espectrogramas sintetizados en formas de onda. La arquitectura es la misma descrita en [Tacotron 2], compuesta por **30 capas de convolución dilatada**. **La red no está condicionada directamente por la salida del speaker encoder.** El mel-espectrograma predicho por la red de síntesis **captura todo el detalle relevante** necesario para síntesis de alta calidad de una variedad de voces, permitiendo construir un vocoder multi-hablante simplemente entrenándolo sobre datos de muchos hablantes."*

### 6.1. Por qué es cierto

El argumento es de **suficiencia de la representación intermedia**. Prácticamente toda la información que un oyente usa para identificar una voz es **espectral y está presente en el log-mel**:

- **Frecuencia fundamental $F_0$** (altura de la voz): aparece como el espaciado de los armónicos. La Sección 2.4 lo verifica visualmente sobre la Figura 2: el hablante masculino tiene *"frecuencia fundamental notablemente más baja, visible en el espaciado armónico más denso (franjas horizontales) en las frecuencias bajas"*.
- **Formantes** (posición de las resonancias del tracto vocal, que es esencialmente la geometría anatómica de la persona): *"visibles en los picos de frecuencia media presentes durante los sonidos vocálicos, como la 'i' en 0.3 segundos — el F2 del hablante masculino superior está en el canal mel 35, mientras que el F2 del hablante del medio aparece más cerca del canal 40"*.
- **Distribución de energía en sibilantes**: *"la 's' en 0.4 segundos contiene más energía en frecuencias bajas en la voz masculina que en las femeninas"*.
- **Inclinación espectral, ancho de banda de formantes, y velocidad de habla** — todo visible en el mel.

Lo que **no** está en el mel es la **fase** y la estructura fina de la forma de onda por debajo de la resolución de 12.5 ms y de los 80 canales mel. La tarea de WaveNet es reconstruir eso. Y esa reconstrucción es, en primera aproximación, **un problema acústico genérico**: dado un contorno espectral y una $F_0$ implícita, generar una excitación glotal plausible y filtrarla. No requiere saber *quién* es el hablante, porque el *qué* ya viene especificado frame a frame.

Dicho de otro modo: el mel-espectrograma actúa como una **frontera de abstracción**. El sintetizador es responsable de "quién habla y qué dice"; el vocoder es responsable de "cómo suena una señal con este espectro". La identidad ya se resolvió aguas arriba.

### 6.2. Por qué la afirmación es más débil de lo que parece

Aquí conviene ser preciso, porque el paper es más cuidadoso de lo que la cita suelta sugiere. Tres matices, todos con evidencia en el propio texto:

**(a) El vocoder sí debe ser *entrenado* multi-hablante.** La frase completa es "permitiendo construir un vocoder multi-hablante **simplemente entrenándolo sobre datos de muchos hablantes**". No es que el vocoder sea agnóstico al hablante; es que **no necesita condicionamiento explícito** — le basta con haber visto suficiente variedad de espectros durante el entrenamiento para que el mapeo mel→onda esté bien definido en todo el rango de $F_0$ y de calidades vocales que se le van a pedir. Un WaveNet entrenado con una sola voz masculina y evaluado sobre un mel de voz femenina aguda estaría extrapolando.

**(b) Entrenaron un vocoder por corpus, no uno solo.** Sección 3: *"Entrenamos redes de síntesis y de vocoder **separadas para cada uno de estos dos corpus**"*. VCTK a 24 kHz, LibriSpeech a 16 kHz. Es decir, en la práctica el vocoder tampoco es transferible entre dominios acústicos sin reentrenar.

**(c) La receta cambia según el ruido del corpus.** Sección 3: *"Para el dataset VCTK, cuyo audio es bastante limpio, encontramos que el vocoder entrenado sobre **mel-espectrogramas ground truth** funcionaba bien. Sin embargo para LibriSpeech, que es más ruidoso, encontramos **necesario entrenar el vocoder sobre espectrogramas predichos por la red de síntesis**."* Ese es el clásico problema de *exposure bias* entre etapas: si el vocoder solo vio mels reales y en inferencia recibe mels predichos (más suaves, con artefactos propios del $L_2$/$L_1$), hay desajuste de distribución. Entrenarlo sobre las predicciones del sintetizador lo alinea con lo que verá en producción. **Esto es un anticipo de exactamente el argumento de brecha de dominio que aparece en §13.**

**(d) La calidad no alcanza el nivel humano pese a WaveNet.** De la conclusión: *"El modelo propuesto **no alcanza naturalidad a nivel humano**, a pesar del uso de un vocoder WaveNet (junto con su **muy alto costo de inferencia**), en contraste con los resultados de un solo hablante de [Tacotron 2]."* La causa que dan: la dificultad adicional de generar habla para muchos hablantes con mucho menos dato por hablante, más la menor calidad de los datasets.

---

## 7. Los datasets y su asimetría

### 7.1. La tabla completa

Todas las cifras vienen de la Sección 3 y de la Sección 3.5.

| Dataset | Rol | Hablantes | Volumen | Transcripción | Calidad de audio |
|---|---|---|---|---|---|
| **Corpus interno de voice search** | Speaker encoder (modelo principal) | **18K** (inglés de EE.UU.) | **36M utterances**, duración mediana **3.9 s** | **No** ("identidades anonimizadas") | Habla real, ruidosa |
| **VoxCeleb** (Nagrani et al. 2017) | Speaker encoder (ablación) | **1,211** | **139K utterances** | No (para este uso) | Entrevistas de YouTube, ruidosas |
| **VoxCeleb2** (Chung et al. 2018) | Speaker encoder (ablación) | **5,994** | **1.09M utterances** | No | Ídem |
| **LibriSpeech Other** | Speaker encoder (ablación) | **1,166** | **461 horas** | Sí (no usada aquí) | Audiolibros, variable |
| **LibriSpeech clean** (Panayotov et al. 2015) | Sintetizador + vocoder | **1,172** | **436 horas**, 16 kHz | **Sí** (sin puntuación) | Audiolibros, con ruido de fondo apreciable |
| **VCTK** (Veaux et al. 2017) | Sintetizador + vocoder | **109** (98 en train) | **44 horas**, 24 kHz | **Sí** | Limpio, mayoría acento británico |
| **Corpus de evaluación** (independiente) | Encoder de evaluación (SV-EER) | **113K** | **28M utterances** | No | — |

### 7.2. La asimetría, en números

El encoder principal se entrena sobre **18,000 hablantes**. El sintetizador, sobre **1,172** en el mejor caso, y sobre **98** en el peor. Un factor de **15× a 180×**.

Y esa asimetría es **el punto entero del paper**. El corpus de 18K hablantes:

- No tiene transcripciones → **inservible para entrenar un TTS**.
- Es ruidoso → **inservible como target de síntesis**.
- Tiene 36M utterances y 18K identidades → **perfecto para verificación de hablante**.

VCTK, al revés: limpio, transcrito, 24 kHz — ideal para TTS — pero **109 hablantes** no permiten aprender el espacio de variación de la voz humana.

### 7.3. La evidencia: Tabla 5

Este es el experimento más importante del paper después de las Tablas 1 y 2. **El sintetizador es siempre el mismo (LibriSpeech Clean), evaluado sobre hablantes held-out. Solo cambia el dataset del speaker encoder.**

| Dataset del Speaker Encoder | Hablantes | Dim. embedding | Naturalidad | Similitud | SV-EER |
|---|---|---|---|---|---|
| LibriSpeech Clean | 1.2K | 64 | 3.73 ± 0.06 | 2.23 ± 0.08 | 16.60% |
| LibriSpeech Other | 1.2K | 64 | 3.60 ± 0.06 | 2.27 ± 0.09 | 15.32% |
| LS-Other + VoxCeleb | 2.4K | 256 | 3.83 ± 0.06 | 2.43 ± 0.09 | 11.95% |
| LS-Other + VoxCeleb + VoxCeleb2 | 8.4K | 256 | 3.82 ± 0.06 | 2.54 ± 0.09 | 10.14% |
| **Interno (voice search)** | **18K** | **256** | **4.12 ± 0.05** | **3.03 ± 0.09** | **5.08%** |

De 1.2K a 18K hablantes: naturalidad **+0.39**, similitud **+0.80**, y la EER **se divide por más de 3** (16.60% → 5.08%). Todo eso **sin tocar el sintetizador ni un solo parámetro**.

El paper también aísla un control importante: las dos primeras filas comparan LS-Clean (donde encoder y sintetizador se entrenan **sobre los mismos datos**, condición "matched") contra LS-Other (disjunto, mismo número de hablantes). *"Esta condición matched da naturalidad ligeramente mejor y un puntaje de similitud similar."* O sea: **entrenar el encoder sobre datos que el sintetizador nunca vio no cuesta nada**. Lo único que importa es cuántos hablantes distintos vio el encoder.

Y la conclusión que el paper saca (Sección 3.5): *"Estos resultados tienen una implicación importante para el entrenamiento de TTS multi-hablante. El requisito de datos para el speaker encoder es mucho más barato que el entrenamiento TTS completo, ya que **no son necesarias transcripciones**, y la **calidad de audio puede ser más baja** que para entrenamiento TTS. Hemos mostrado que es posible sintetizar TTS muy natural combinando una red de speaker encoder entrenada sobre grandes cantidades de datos sin transcribir con una red TTS entrenada sobre un conjunto más pequeño de datos de alta calidad."*

**Nota de caveat, honesta:** hay una confusión parcial en esta tabla. Las dos primeras filas usan **arquitectura más pequeña** (celdas LSTM de 256, proyecciones de 64) y **embedding de 64 dimensiones**, mientras las tres últimas usan 256. El paper lo declara ("para evitar sobreajuste, los speaker encoders entrenados sobre datasets pequeños usan una arquitectura más pequeña"), y es una decisión razonable, pero implica que el salto entre las filas 2 y 3 mezcla *más hablantes* con *más capacidad de embedding*. El salto de la fila 4 a la 5 (8.4K → 18K, ambas con 256 dims) sí es limpio, y ahí solo la diversidad de hablantes explica **+0.30 de naturalidad, +0.49 de similitud y −5 puntos de EER**. Ese salto solo es el argumento más fuerte del paper.

---

## 8. Experimentos y resultados

### 8.1. Protocolo de evaluación

- **MOS** crowdsourced, escala ACR (ITU-T P.800), **1 a 5 en incrementos de 0.5**.
- Set de evaluación: **100 frases** que no aparecen en ningún set de entrenamiento.
- Hablantes: **11 seen + 11 unseen** en VCTK, **10 seen + 10 unseen** en LibriSpeech (Apéndice D lista los IDs exactos).
- Para cada hablante se elige **al azar una utterance de ~5 segundos** para calcular el embedding.
- **~1,000 utterances sintetizadas por evaluación**. Cada muestra la califica **un solo rater**, y **cada evaluación es independiente: las salidas de distintos modelos no se compararon directamente entre sí**. (Consecuencia: comparar MOS entre tablas distintas es arriesgado.)
- **Similitud:** cada utterance sintetizada se aparea con una utterance real elegida al azar del mismo hablante, y la instrucción al rater es explícita: *"No debe juzgar el contenido, la gramática, ni la calidad de audio de las oraciones; en cambio, enfóquese solo en la similitud de los hablantes entre sí."*

### 8.2. Naturalidad (Tabla 1)

| Sistema | VCTK Seen | VCTK Unseen | LibriSpeech Seen | LibriSpeech Unseen |
|---|---|---|---|---|
| Ground truth | 4.43 ± 0.05 | 4.49 ± 0.05 | 4.49 ± 0.05 | 4.42 ± 0.07 |
| Tabla de embeddings | 4.12 ± 0.06 | N/A | 3.90 ± 0.06 | N/A |
| **Modelo propuesto** | 4.07 ± 0.06 | 4.20 ± 0.06 | 3.89 ± 0.06 | 4.12 ± 0.05 |

Lecturas:

- **El modelo propuesto empata con el baseline de tabla de embeddings** en hablantes vistos (4.07 vs 4.12 en VCTK; 3.89 vs 3.90 en LibriSpeech, indistinguibles dentro del IC). Es decir: la generalización zero-shot **no se paga con naturalidad** en el caso conocido.
- **VCTK saca ~0.2 puntos a LibriSpeech en seen.** El paper da dos causas: *"(i) la falta de puntuación en las transcripciones, que hace difícil que el modelo aprenda a pausar naturalmente, y (ii) el mayor nivel de ruido de fondo comparado con VCTK, parte del cual el sintetizador ha aprendido a reproducir, a pesar de haber denoisado los targets de entrenamiento"*. La primera causa es especialmente instructiva: **la puntuación es el único portador de la estructura prosódica en la entrada textual**. Sin comas ni puntos, no hay señal de dónde pausar.
- **Unseen > Seen.** Ver §14: es un artefacto de muestreo, no una propiedad del modelo.

El preproceso de denoising de LibriSpeech vale la pena mencionarlo porque es un detalle práctico bien pensado: **sustracción espectral** con el espectro de ruido estimado como *"el percentil 10 de la energía en cada banda de frecuencia a lo largo de la señal completa"*. Y — detalle crucial — *"Este proceso solo se usó sobre el **target de síntesis**; el habla ruidosa original se pasó al **speaker encoder**"*. Denoisan lo que el sintetizador debe imitar, pero dejan el ruido en lo que el encoder debe caracterizar, porque el encoder fue entrenado sobre audio ruidoso y es robusto a eso.

### 8.3. Similitud de hablante (Tabla 2)

| Sistema | Conjunto | VCTK | LibriSpeech |
|---|---|---|---|
| Ground truth | Mismo hablante | 4.67 ± 0.04 | 4.33 ± 0.08 |
| Ground truth | Mismo género | 2.25 ± 0.07 | 1.83 ± 0.07 |
| Ground truth | Género distinto | 1.15 ± 0.04 | 1.04 ± 0.03 |
| Tabla de embeddings | Seen | 4.17 ± 0.06 | 3.70 ± 0.08 |
| **Modelo propuesto** | **Seen** | **4.22 ± 0.06** | **3.28 ± 0.08** |
| **Modelo propuesto** | **Unseen** | **3.28 ± 0.07** | **3.03 ± 0.09** |

Las tres filas de ground truth son la **calibración de la escala**, y son lo más valioso de la tabla: dos utterances reales del mismo hablante puntúan 4.67, dos hablantes reales distintos del mismo género puntúan 2.25, y de género distinto 1.15. Eso ancla qué significa un 3.28.

Lecturas:

- **La brecha seen → unseen es la cifra central de todo el paper: 4.22 → 3.28 en VCTK (−0.94) y 3.28 → 3.03 en LibriSpeech (−0.25).** El paper describe el 3.28 de VCTK como *"entre 'moderadamente similar' y 'muy similar' en la escala"*.
- **En seen, VCTK empata al baseline** (4.22 vs 4.17) pero **en LibriSpeech el propuesto pierde contra el baseline** (3.28 vs 3.70). El paper atribuye esto al *"mayor grado de variación intra-hablante (Apéndice B) y al nivel de ruido de fondo del dataset"*. Es un resultado honesto que se suele omitir al citar el paper: **cuando el hablante está en el set de entrenamiento y hay mucha variación intra-hablante, una tabla de embeddings aprendida sigue siendo mejor.** Tiene sentido: la tabla promedia sobre todas las utterances del hablante, mientras el encoder ve una sola de 5 segundos.
- **Diagnóstico cualitativo del paper:** *"el modelo propuesto es capaz de transferir **los trazos gruesos** de las características del hablante para hablantes no vistos, reflejando claramente el **género, el pitch y los rangos de formantes** correctos... pero los puntajes de similitud significativamente reducidos sobre hablantes no vistos sugieren que **algunos matices, por ejemplo los relacionados con la prosodia característica, se pierden**."*
- **Confusor de acento.** *"El speaker encoder está entrenado **solo sobre habla con acento norteamericano**. Como resultado, el desajuste de acento restringe nuestro desempeño en similitud de hablante sobre VCTK"* — que es mayoritariamente británico. Los comentarios de los raters muestran que *"nuestro modelo a veces produjo un acento distinto al del ground truth, lo que llevó a puntajes más bajos"*, aunque *"unos pocos raters comentaron que el tono e inflexión de las voces sonaban muy similares a pesar de las diferencias de acento"*.

### 8.4. Evaluación cruzada entre datasets (Tabla 3)

Sintetizador entrenado en un corpus, hablantes no vistos del otro. **El speaker encoder es idéntico en ambas filas.**

| Sintetizador entrenado en | Evaluado en | Naturalidad | Similitud |
|---|---|---|---|
| VCTK (98 hablantes) | LibriSpeech | 4.28 ± 0.05 | **1.82 ± 0.08** |
| LibriSpeech (1.2K hablantes) | VCTK | 4.01 ± 0.06 | **2.77 ± 0.08** |

Este experimento es más importante de lo que su tamaño sugiere. **La naturalidad se mantiene** (4.28 y 4.01, comparables a los unseen in-domain de la Tabla 1) pero **la similitud se derrumba**, sobre todo para el sintetizador de VCTK: **1.82**, que en la escala calibrada de la Tabla 2 está por debajo del "mismo género, distinto hablante" (1.83 en LibriSpeech). En términos prácticos, el sintetizador de VCTK **no clona nada** fuera de su dominio.

La conclusión del paper: *"la mejor generalización del modelo de LibriSpeech sugiere que **entrenar el sintetizador sobre solo 100 hablantes es insuficiente** para habilitar transferencia de hablante de alta calidad."*

Esto matiza el mensaje principal de forma importante: **la diversidad de hablantes del encoder es necesaria pero no suficiente**. El sintetizador también necesita haber visto suficientes hablantes para aprender a *usar* el embedding como una dirección de control real, en lugar de memorizar un puñado de voces. Es la misma condición que enuncia la conclusión: *"**dada suficiente diversidad de hablantes en el set de entrenamiento del sintetizador**, la calidad de transferencia de hablante puede mejorarse significativamente aumentando la cantidad de datos de entrenamiento del speaker encoder"*.

### 8.5. Verificación de hablante sobre audio sintético (Tabla 4)

Protocolo (Sección 3.3): entrenaron un **encoder de evaluación nuevo**, misma topología pero **distinto training set** (28M utterances de 113K hablantes), *"para asegurar que las métricas no fueran válidas solo sobre un espacio de embeddings específico"*. Enrolaron 21 hablantes reales (11 VCTK + 10 LibriSpeech), todos no vistos en el entrenamiento del sintetizador, y puntuaron las formas de onda sintetizadas contra el conjunto de enrolados. 100 utterances de test por hablante → **21,000 o 23,100 trials por evaluación**.

| Sintetizador entrenado en | Hablantes de entrenamiento | SV-EER en VCTK | SV-EER en LibriSpeech |
|---|---|---|---|
| **Ground truth** (audio real) | — | **1.53%** | **0.93%** |
| VCTK | 98 | 10.46% | **29.19%** |
| LibriSpeech | 1.2K | 6.26% | 5.08% |

- El audio real da EER de 1-1.5%: el verificador funciona.
- El sintetizador de LibriSpeech da **5-6% en ambos datasets** — consistente, generaliza.
- El sintetizador de VCTK **colapsa fuera de dominio: 29.19%**, cerca del azar. Coincide con el 1.82 de similitud subjetiva de la Tabla 3. *"Estos resultados son consistentes con la evaluación subjetiva de la Tabla 3."*

**El experimento de seguridad (Sección 3.3).** Expandieron el conjunto de enrolados con **10 versiones sintéticas de los 10 hablantes reales de LibriSpeech**, formando una tarea de discriminación entre 20 voces. Resultado: **EER de 2.86%**. La interpretación del paper: *"mientras el habla sintética tiende a estar cerca del hablante objetivo (**similitud coseno > 0.6**), es casi siempre **aún más cercana a otras utterances sintéticas del mismo hablante** (**similitud > 0.7**). De esto podemos concluir que el modelo propuesto puede generar habla que se parece al hablante objetivo, **pero no lo suficientemente bien como para ser confundible con un hablante real**."*

Ese es el "safety check" que la introducción anunciaba: *"verificamos que las voces generadas por el modelo propuesto pueden distinguirse fácilmente de voces reales"*. Se cumplió en 2018. Ya no se cumple (ver §11).

### 8.6. Duración del audio de referencia (Apéndice C, Tabla 9)

Evaluado sobre VCTK:

| | 1 s | 2 s | 3 s | 5 s | 10 s |
|---|---|---|---|---|---|
| Naturalidad (MOS) | **4.28 ± 0.05** | 4.26 ± 0.05 | 4.18 ± 0.06 | 4.20 ± 0.06 | 4.16 ± 0.06 |
| Similitud (MOS) | 2.85 ± 0.07 | 3.17 ± 0.07 | **3.31 ± 0.07** | 3.28 ± 0.07 | 3.18 ± 0.07 |
| SV-EER | 17.28% | 11.30% | **10.80%** | 10.46% | 11.50% |

Tres observaciones del paper, todas contraintuitivas hasta que se explican:

1. **La similitud satura a los ~5 segundos**, y con 2 segundos ya se está cerca del óptimo. *"Aumentar la longitud del habla de referencia mejoró significativamente la similitud, porque podemos calcular un embedding de hablante más preciso."*
2. **Las referencias más cortas dan naturalidad ligeramente mejor**, *"porque calzan mejor con las duraciones de las utterances de referencia usadas para entrenar el sintetizador, cuya duración mediana es 1.8 segundos"*. Es un desajuste train/test puro.
3. **Después de 5 segundos, más audio no ayuda — incluso empeora un poco.** El paper es directo sobre el porqué: *"La saturación del desempeño usando solo 5 segundos de habla resalta una **limitación** del modelo propuesto, que está restringido por la **pequeña capacidad del embedding de hablante**."* Un vector de 256 dimensiones es un cuello de botella duro: llegado cierto punto, ya no cabe más información. La conclusión reitera: *"modelar la variación de hablante usando un vector de baja dimensión **limita la capacidad de aprovechar grandes cantidades de habla de referencia**. Mejorar la similitud de hablante dado más de unos pocos segundos de habla de referencia requiere un enfoque de **adaptación del modelo**"* (citando a Arik et al. y a Chen et al. 2018, *Sample Efficient Adaptive Text-to-Speech*).

---

## 9. Análisis del espacio de embeddings

### 9.1. Estructura (Sección 3.4, Figura 3)

Visualizan embeddings extraídos de utterances de LibriSpeech, reales y sintéticas, con **PCA** y **t-SNE**. Tres hallazgos:

1. **Separación por hablante.** *"Distintos hablantes están bien separados unos de otros en el espacio de embeddings."*
2. **Separación por género.** *"Los hablantes aparecen bien separados por género tanto en PCA como en t-SNE, con **todas las hablantes femeninas apareciendo a la izquierda y todos los masculinos a la derecha**. Esto es un indicio de que el speaker encoder ha aprendido una representación razonable del espacio de hablantes."* Que el género emerja como un **eje principal** en PCA —o sea, como la dirección de máxima varianza o cercana a ella— sin haber sido supervisado nunca, indica que la dimensión acústicamente más discriminativa entre hablantes es la que separa rangos de $F_0$ y de formantes. No es sorprendente fonéticamente, pero es la confirmación de que el espacio tiene estructura interpretable y no es un embedding arbitrario.
3. **Real y sintético forman clusters distintos pero adyacentes.** *"La visualización PCA (izquierda) muestra que las utterances sintetizadas tienden a caer **muy cerca** del habla real del mismo hablante. Sin embargo, las utterances sintéticas siguen siendo **fácilmente distinguibles** del habla humana real, como demuestra la visualización t-SNE (derecha) donde las utterances de cada hablante sintético forman un **cluster distinto adyacente** al cluster de utterances reales del hablante correspondiente."*

Esa tercera observación es la contraparte geométrica del EER de 2.86%: la síntesis introduce un **sesgo sistemático** en el espacio de embeddings, un desplazamiento consistente que el verificador puede detectar. En términos de detección de deepfakes: existe una "firma del vocoder" separable linealmente.

### 9.2. Hablantes ficticios (Sección 3.6, Tabla 6, Figura 5)

El experimento más provocador del paper. **Se salta el speaker encoder por completo** y se condiciona el sintetizador sobre **puntos aleatorios muestreados uniformemente sobre la superficie de la hiperesfera unitaria** de 256 dimensiones.

Configuración: sintetizador entrenado en LS-Clean, speaker encoder entrenado en LS-Other + VoxCeleb + VoxCeleb2. 10 hablantes ficticios. Se enrolan los 10 vecinos más cercanos en los sets de entrenamiento y se computa el EER con el setup de la Sección 3.3.

| Vecinos más cercanos en | Similitud coseno | SV-EER | Naturalidad MOS |
|---|---|---|---|
| Set de entrenamiento del **sintetizador** | 0.222 | **56.77%** | 3.65 ± 0.06 |
| Set de entrenamiento del **speaker encoder** | 0.245 | **38.54%** | (mismo) |

Cómo leer esto:

- **Naturalidad 3.65.** Compárese con 3.73 (encoder de 1.2K hablantes) y 4.12 (encoder de 18K) de la Tabla 5 sobre el mismo sintetizador. *"Aunque estos hablantes son totalmente ficticios, el sintetizador y el vocoder son capaces de generar audio **tan natural como para hablantes reales vistos o no vistos**."*
- **Similitud coseno de 0.22-0.25 al vecino más cercano.** Recuérdese que en la Sección 3.3 el habla sintética de un hablante real alcanzaba **similitud > 0.6** con su target. 0.22 es esencialmente ortogonalidad práctica.
- **EER de 56.77%.** Por encima del 50%: el verificador está **peor que el azar** al intentar asociar el hablante ficticio con su vecino más cercano de entrenamiento. *"La baja similitud coseno a las utterances del vecino más cercano y la EER muy alta indican que son efectivamente **distintos** de los hablantes de entrenamiento."*

La Figura 5 (Apéndice E) muestra seis mel-espectrogramas de la misma frase ("this is a big red apple") con seis embeddings aleatorios: *"Todas las muestras contienen contenido fonético consistente, pero hay **variación clara en la frecuencia fundamental y en la velocidad de habla**."*

**Qué implica.** Tres cosas, en orden de importancia:

1. **El espacio de embeddings es una variedad densa y navegable, no un conjunto de puntos memorizados.** Si el sintetizador solo hubiera aprendido a mapear los ~1,200 embeddings vistos a sus voces, un punto aleatorio caería en una región no entrenada y produciría basura, o colapsaría a la voz más cercana. No ocurre ninguna de las dos: produce **una voz nueva, coherente y natural**. La conclusión del paper lo dice: *"implicando que el modelo ha aprendido a utilizar una **representación realista del espacio de variación de hablantes**"*.
2. **Es la mejor prueba de que la generalización zero-shot es real y no recuperación aproximada.** Si el mecanismo fuera "encontrar al hablante de entrenamiento más parecido y usar su voz", la Tabla 6 sería imposible. La generalización a hablantes no vistos es interpolación/extrapolación genuina sobre una variedad continua, no un *nearest neighbor* encubierto.
3. **Se pueden fabricar identidades vocales que no pertenecen a nadie.** Éticamente esto es interesante en ambos sentidos: es el camino para **generar voces sintéticas sin consentimiento de nadie** (voces de marca, personajes, narradores) evitando el problema de la suplantación; y es también, técnicamente, la demostración de que el modelo domina el espacio de identidades vocales de forma generativa. Es el ancestro directo de los "voice designers" comerciales que hoy permiten sintetizar una voz especificando atributos en lugar de subiendo una muestra.

El paper **no reporta** experimentos de interpolación entre dos embeddings reales ni de aritmética vectorial en el espacio de hablantes. Muestreo uniforme sobre la esfera es lo único que hace. Vale aclararlo porque a veces se le atribuyen resultados de interpolación que no están en el PDF.

---

## 10. Limitaciones

### 10.1. Las que el paper reconoce

**(a) La brecha de similitud en hablantes no vistos.** 4.22 → 3.28 en VCTK. El paper no la esconde, pero conviene decirla con las palabras que corresponden: *"transfiere los trazos gruesos... pero algunos matices se pierden"*. En 2018 el zero-shot cloning producía una voz **del mismo tipo** que el objetivo, no la voz del objetivo.

**(b) No alcanza naturalidad humana.** 4.12 contra 4.42-4.49 del ground truth, *"a pesar del uso de un vocoder WaveNet (junto con su muy alto costo de inferencia)"*.

**(c) No transfiere acento.** De la conclusión: *"Una limitación adicional está en la **incapacidad del modelo de transferir acentos**. Dados suficientes datos de entrenamiento, esto podría abordarse condicionando el sintetizador sobre **embeddings independientes de hablante y de acento**."* La causa es evidente en los datos: encoder entrenado solo sobre inglés norteamericano, sintetizador de VCTK sobre acentos británicos.

**(d) No aísla la voz de la prosodia de la referencia.** Última frase de la conclusión: *"notamos que el modelo tampoco es capaz de **aislar completamente la voz del hablante de la prosodia del audio de referencia**, una tendencia similar a la observada en [Skerry-Ryan et al.]."*

**(e) Capacidad limitada del embedding.** El embedding de 256 dimensiones satura a los 5 segundos de referencia (Apéndice C). Más audio no se puede aprovechar sin adaptación de parámetros.

**(f) Costo de inferencia del WaveNet.** Explícito en la conclusión. WaveNet autorregresivo genera **muestra por muestra**: a 24 kHz son 24,000 forward passes secuenciales por segundo de audio. Sin las técnicas de destilación paralela de la época (Parallel WaveNet, ClariNet) esto está muy lejos de tiempo real.

### 10.2. La limitación más importante: transfiere timbre, no estilo

Esta merece desarrollo propio porque es **la crítica más citada del paper** y la que define toda la agenda de investigación posterior.

**Qué transfiere SV2TTS:** el **timbre**. La firma espectral estática de un hablante — posición de formantes (que es esencialmente la geometría del tracto vocal), rango de $F_0$, inclinación espectral, calidad de fonación (breathiness, ronquera). Todo eso está codificado en 256 dimensiones y todo eso el modelo lo reproduce con fidelidad razonable.

**Qué no transfiere:** el **estilo de habla**, que es un objeto **temporal y contextual**:

- **Contorno de entonación** — cómo sube y baja el pitch a lo largo de una frase, dónde pone el acento tonal, cómo marca las preguntas.
- **Ritmo y duración fonémica** — patrones de acentuación, cuánto alarga las vocales tónicas, cuánto reduce las átonas.
- **Pausas** — dónde y cuánto respira, dónde duda.
- **Acento y dialecto** — que es, esencialmente, un mapeo fonema→realización acústica distinto, más un patrón prosódico distinto. El paper lo reconoce por separado en (c).
- **Estado emocional y registro**.

**Por qué el diseño no puede transferirlo, estructuralmente.** Tres razones que se componen:

1. **Un vector estático no puede representar una función del tiempo.** El embedding es un punto fijo replicado en todos los pasos temporales. El estilo prosódico es una **trayectoria**. Se puede codificar el *promedio* del pitch de alguien, pero no su patrón de contorno, en un vector que no varía. Esto es una limitación de tipo, no de capacidad.
2. **La pérdida GE2E está entrenada para ser invariante a la prosodia.** La verificación de hablante *text-independent* debe reconocer a la misma persona diciendo cosas distintas, con emociones distintas, a velocidades distintas. La prosodia es, para esa tarea, **ruido de nuisance que hay que descartar**. El encoder está literalmente optimizado para tirar a la basura exactamente la información que haría falta para transferir estilo. **Esa es la ironía del diseño:** lo que lo hace robusto es lo que lo hace prosódicamente ciego.
3. **La configuración de entrenamiento contamina lo poco que queda.** Como se vio en §5.3, durante el entrenamiento la referencia **es** el target. El decoder aprende a exprimir cualquier residuo prosódico del embedding, porque durante el entrenamiento ese residuo es predictivo. En inferencia eso se manifiesta como transferencia **parcial y no controlada** de la prosodia de la referencia. El paper lo observa: *"En pruebas de escucha informales encontramos que **la prosodia del habla sintetizada a veces imita la de la referencia**... Esto sugiere que debe tenerse cuidado adicional para **desenredar la identidad del hablante de la prosodia** dentro de la red de síntesis, quizás integrando un **encoder de prosodia** como en [Skerry-Ryan et al., Wang et al. Style Tokens], o **entrenando sobre pares de referencia y target elegidos aleatoriamente del mismo hablante**."*

Es lo peor de ambos mundos: no transfiere el estilo del hablante objetivo de forma útil y controlable, pero sí filtra la prosodia de la muestra de referencia de forma impredecible.

**Consecuencia práctica.** Si se clona la voz de alguien a partir de 5 segundos y se le hace leer un párrafo, el resultado suena como esa persona **leyendo con la prosodia genérica del corpus TTS**, no como esa persona hablando. Para un oyente que conoce al hablante, esa es una diferencia enorme; es lo que hace que el clon "casi funcione" pero no engañe a un familiar. Y es exactamente por eso que la línea posterior del campo se dedicó a modelar prosodia y estilo por separado: **Global Style Tokens** (Wang et al. 2018, citado en el paper), **prosody transfer** (Skerry-Ryan et al. 2018, citado), y eventualmente los modelos de lenguaje sobre tokens acústicos que resolvieron el problema de raíz al no usar un cuello de botella estático.

### 10.3. Limitaciones que el paper no reconoce del todo

**(g) Dependencia de la calidad del audio de referencia.** El paper afirma robustez al ruido ("independientemente de su contenido fonético y **del ruido de fondo**", y de hecho al speaker encoder le pasan el audio ruidoso sin denoisar). Pero **no hay ningún experimento controlado que varíe el SNR de la referencia**. No hay tabla de degradación por ruido, ni por códec, ni por reverberación. La robustez es un argumento de diseño (el corpus de entrenamiento era ruidoso), no un resultado medido. El único eje de la referencia que sí se estudia es la **duración** (Apéndice C).

**(h) Un solo idioma.** Todo el trabajo es en inglés: encoder sobre inglés norteamericano, VCTK inglés británico, LibriSpeech inglés estadounidense. No se evalúa nada cross-lingüe. Dado el hallazgo sobre acentos, es razonable esperar degradación fuerte al aplicarlo a otro idioma, tanto por el encoder como por el sintetizador. (Este hilo lo retomaron los propios autores en trabajo posterior — ver §12.)

**(i) Sesgo demográfico no auditado.** 18K hablantes de *voice search* en EE.UU. no es una muestra uniforme de la variación vocal humana. No hay desglose por edad, por variedad dialectal ni por características vocales atípicas. Y dado que el modelo se propone como tecnología de accesibilidad para personas que perdieron la voz, la ausencia de evaluación sobre **habla disártrica o atípica** es una brecha relevante para ese caso de uso declarado.

**(j) Varianza por hablante grande.** El Apéndice B lo documenta: en VCTK, el hablante "p240" alcanza MOS de 4.48 contra 4.57 del ground truth (casi indistinguible) mientras "p260" queda **medio punto completo** por debajo del suyo. El MOS agregado esconde que el sistema funciona muy bien para algunas voces y mediocre para otras, sin que se sepa qué predice cuál.

**(k) Un rater por muestra.** *"Cada muestra fue calificada por un solo rater"*. Es lo estándar en evaluaciones crowdsourced grandes, pero significa que la varianza de rater no está promediada por ítem; los intervalos de confianza reportados asumen independencia entre ~1,000 juicios de raters distintos.

---

## 11. El problema ético y de seguridad

El paper **sí** aborda el tema, en la introducción y en la Sección 3.3, y conviene reconocerlo porque en 2018 no era la norma.

**Lo que el paper dice.** Introducción: *"es también importante notar el potencial de mal uso de esta tecnología, por ejemplo **suplantando la voz de alguien sin su consentimiento**. Para abordar preocupaciones de seguridad consistentes con principios como [los Principios de IA de Google], **verificamos que las voces generadas por el modelo propuesto pueden distinguirse fácilmente de voces reales**."* La verificación es el experimento de la Sección 3.3: EER de 2.86% en la tarea de discriminación de 20 voces (10 reales + 10 sintéticas), más la Figura 3 mostrando clusters t-SNE separados.

**El problema con esa mitigación.** Es un argumento sobre **el estado del arte del momento**, no sobre una propiedad estructural. La detectabilidad venía de artefactos del vocoder WaveNet de 2018 y de la brecha de similitud. Ambas cosas se cerraron en los años siguientes. Un sistema de 2018 que "se distingue fácilmente" no dice nada sobre un sistema de 2024. Y el paper que se publicó — el método — sí se transfirió íntegro; la garantía de seguridad, no.

**El vector de fraude concreto.** Cinco segundos de audio de referencia es el umbral operativo (Apéndice C: 2 segundos ya está cerca del óptimo). Cinco segundos de la voz de una persona se obtienen de un mensaje de voz, de un video en redes sociales, de una llamada de "¿aló?, ¿aló?". Las dos aplicaciones directas:

1. **Fraude por suplantación en llamadas** — el esquema de "emergencia familiar" o de autorización de transferencias por voz. Documentado ampliamente desde 2019 en adelante.
2. **Evasión de biometría de voz** — sistemas de autenticación telefónica del tipo "mi voz es mi contraseña", que la banca desplegó masivamente en la década de 2010.

**La ironía central.** El mismo componente que la industria usa para **verificar** identidad por voz (un d-vector entrenado con GE2E, exactamente lo que describen Wan et al. 2018 y lo que se despliega en sistemas de verificación de hablante en producción) es el componente que aquí permite **suplantarla**. La ecuación es simple y no tiene solución dentro de este paradigma: un espacio de embeddings suficientemente bueno para discriminar hablantes es, por construcción, un espacio suficientemente informativo para condicionar un generador. **La calidad del verificador acota por abajo la calidad del clonador.** Mejorar la biometría de voz mejora, gratis, la capacidad de vulnerarla.

**Las dos líneas de defensa que se desarrollaron después.** Ambas están fuera del paper; las menciono como contexto, sin cifras:

- **Detección de deepfakes de voz (anti-spoofing).** La serie de desafíos **ASVspoof** (ediciones 2015, 2017, 2019, 2021, 2024) es el marco de referencia de la comunidad, con pistas de *logical access* (habla sintética y convertida) y *physical access* (replay). La dificultad conocida es la **generalización a ataques no vistos**: los detectores funcionan bien contra los sintetizadores con los que fueron entrenados y se degradan contra métodos nuevos. Es una carrera armamentista, no un problema resuelto.
- **Marcas de agua de audio.** Insertar una señal imperceptible en el audio generado que sobreviva a compresión, re-grabación y edición, y que permita atribuir la generación. **AudioSeal** (Meta, 2024) es el trabajo más citado en marcas de agua localizadas y robustas para habla generada. La limitación estructural es obvia: una marca de agua solo cubre a los generadores que **eligen** ponerla, así que protege contra el uso descuidado de servicios comerciales, no contra un actor que corre un modelo abierto en su propia máquina.

**Regulación.** En EE.UU. la FCC declaró en febrero de 2024 que las llamadas robotizadas con voces generadas por IA caen bajo la TCPA, lo que las hace ilegales sin consentimiento previo. El AI Act de la UE impone obligaciones de transparencia para contenido sintético. Ninguna de las dos cosas es una defensa técnica; son cambios en el costo esperado para el atacante.

**Postura razonable.** El paper hizo lo correcto para 2018: nombró el riesgo, hizo un experimento de verificación y publicó el resultado. Lo que no podía hacer, y no hizo, es garantizar que la propiedad medida sobreviviera al progreso del campo. Ese es el patrón general de las mitigaciones basadas en "el output es detectable": son válidas hasta que dejan de serlo, y no hay forma de saber cuándo.

---

## 12. Impacto y legado

### 12.1. Continuación directa por los mismos autores

El equipo siguió trabajando sobre la misma pila casi inmediatamente:

- **Translatotron** (Jia et al., Interspeech 2019, *Direct speech-to-speech translation with a sequence-to-sequence model*): traducción habla-a-habla directa, sin ASR ni texto intermedio, **usando el speaker encoder de este paper para preservar la voz del hablante original en el idioma destino**. Es la materialización de la aplicación que la introducción anunciaba.
- **Cross-language voice cloning** (Zhang, Weiss, Jia et al., Interspeech 2019, *Learning to Speak Fluently in a Foreign Language*): TTS multilingüe con clonación de voz entre idiomas, atacando la limitación (h) de §10.3.

### 12.2. La implementación abierta que lo popularizó

**`Real-Time-Voice-Cloning`** (Corentin Jemine, 2019), desarrollada como tesis de maestría en la Universidad de Lieja. Es una reimplementación en PyTorch de los tres componentes de SV2TTS — encoder GE2E, Tacotron con condicionamiento, y **WaveRNN en lugar de WaveNet** para lograr velocidad práctica. Se volvió uno de los repos de audio más populares de GitHub (decenas de miles de estrellas) y es la razón por la que el acrónimo "SV2TTS" existe: el repo lo usa como nombre del método.

Su rol histórico es doble y hay que decir las dos partes: **democratizó** la clonación de voz para investigación y accesibilidad, y **democratizó** la clonación de voz para todo lo demás. Es el momento en que el vector de ataque de §11 pasó de "posible para Google" a "posible para cualquiera con una GPU".

### 12.3. La línea de sucesión

| Sistema | Año | Qué cambió respecto a SV2TTS |
|---|---|---|
| **YourTTS** (Casanova et al., ICML 2022) | 2022 | Sobre VITS (flujos normalizantes + entrenamiento adversarial, end-to-end texto→onda). Zero-shot **multilingüe** y con **conversión de voz**. Elimina el vocoder separado |
| **VALL-E** (Wang et al., Microsoft, 2023) | 2023 | Cambio de paradigma completo: **modelo de lenguaje sobre tokens de códec neuronal** (EnCodec). Entrenado sobre **60,000 horas** (LibriLight), dos órdenes de magnitud más que SV2TTS. Prompt acústico de **3 segundos**. Preserva **emoción y entorno acústico** de la referencia |
| **XTTS** (Coqui, Interspeech 2024) | 2023-24 | Zero-shot masivamente multilingüe (17 idiomas), disponible abiertamente. Fue el caballo de batalla de la comunidad open-source |
| **ElevenLabs, PlayHT, Resemble.ai** | 2022– | Comercialización. Calidad que en muchos casos es indistinguible para un oyente casual, con controles de estilo y emoción |
| **VALL-E 2, NaturalSpeech 3, Voicebox, F5-TTS** | 2023– | Paridad humana reclamada en benchmarks; modelos de difusión / flow matching **no autorregresivos**, que resuelven además el problema de velocidad |

### 12.4. Qué cambió con los modelos de lenguaje sobre tokens de audio

Este es el contraste conceptual que vale la pena entender, porque explica **por qué** la limitación de §10.2 desapareció.

**SV2TTS (tres etapas, cuello de botella explícito):**

```
referencia ──[encoder]──> e ∈ R^256   ← toda la identidad debe caber acá
texto ──[Tacotron 2 | e]──> mel        ← representación intermedia diseñada a mano
mel ──[WaveNet]──> onda
```

**VALL-E (una etapa, prompt en el mismo espacio que la salida):**

```
audio de referencia ──[códec EnCodec]──> tokens acústicos discretos
[texto fonemizado ; tokens de referencia] ──[Transformer autorregresivo]──> tokens acústicos de salida
tokens ──[decoder del códec]──> onda
```

Las tres diferencias que importan:

1. **Desaparece el cuello de botella de dimensión fija.** En SV2TTS la identidad debe comprimirse a 256 números — y §8.6 mostró que ese es el techo duro: 5 segundos de referencia saturan porque no cabe más. En VALL-E el "embedding" es **la secuencia de tokens acústicos de la referencia misma**, que entra como prefijo del contexto. No hay compresión: hay *in-context learning*. Más referencia significa más contexto, y el mecanismo de atención decide qué usar.
2. **La condición ya no es solo la identidad, es todo el contexto acústico.** Como el prompt son tokens acústicos crudos, arrastra **prosodia, emoción, acento y hasta las características del entorno de grabación**. La limitación central de SV2TTS (§10.2) — transfiere timbre, no estilo — se resuelve **por construcción**, no porque alguien haya modelado la prosodia explícitamente: se resuelve al eliminar el cuello de botella que la descartaba. El precio simétrico: **se pierde el desacople**. En SV2TTS se puede tomar el embedding de A y la prosodia genérica del corpus; en VALL-E, el estilo del prompt viene con el paquete, y controlarlo por separado vuelve a ser un problema abierto.
3. **La escala reemplaza al diseño arquitectónico.** SV2TTS ganaba generalización mediante **una elección de arquitectura** (desacoplar componentes para poder usar 18K hablantes sin transcribir). VALL-E la gana mediante **60,000 horas de datos débilmente supervisados**. Es la misma transición que ocurrió en NLP y en visión: los sesgos inductivos cuidadosamente diseñados ceden ante el pre-entrenamiento a escala.

**Lo que sobrevive de SV2TTS.** El concepto de **speaker embedding como condicionamiento** sigue vivo en todas partes: en conversión de voz, en separación de hablantes con condicionamiento (VoiceFilter, del mismo grupo de Google), en diarización, en TTS controlable, y como componente auxiliar de evaluación (la **similitud coseno de speaker embedding** es hoy la métrica objetiva estándar para reportar calidad de clonación — precisamente el protocolo de la Sección 3.3 de este paper, ahora convertido en benchmark de facto). Y sobre todo sobrevive la **tesis**: entrenar componentes por separado sobre los datos que a cada uno le convienen, en lugar de exigir un dataset que satisfaga a todos.

---

## 13. Conexión con la Clase 39

La clase menciona este trabajo en la sección de **data augmentation** y afirma que *"el uso de técnicas de síntesis de sonido es una estrategia conveniente, pero el desempeño sobre datos reales puede ser pobre si el modelo se entrena solo con datos sintéticos: hace falta fine-tuning sobre datos reales"*. Hay dos hilos que tirar.

### 13.1. Hilo 1 — Data augmentation con habla sintética, y por qué falla sola

**El atractivo es obvio.** Un TTS multi-hablante zero-shot convierte **texto** en **pares (audio, transcripción)** perfectamente alineados, en cantidad arbitraria, con la identidad de hablante que uno elija. Para ASR de bajos recursos eso parece la solución perfecta: hay muchísimo texto y muy poco audio transcrito. Y para tareas que la clase 39 cubre —keyword spotting, comandos de voz, reconocimiento de vocabulario específico de dominio— la promesa es aún más directa: generar ejemplos de las palabras que faltan.

**Por qué no funciona entrenando solo con sintético: la brecha de dominio.** Es útil descomponerla en cuatro ejes, y **SV2TTS provee la evidencia de tres de ellos dentro de su propio paper**:

| Eje de la brecha | Qué falta en el habla sintética | Evidencia en SV2TTS |
|---|---|---|
| **Diversidad de hablantes** | Un TTS multi-hablante genera desde un espacio aprendido de ~1.2K hablantes. Es más pobre y más *suave* que la variación humana real | Tabla 5: la calidad depende críticamente de cuántos hablantes vio el encoder. Tabla 3: con 98 hablantes el sintetizador no transfiere identidad fuera de dominio |
| **Diversidad prosódica** | La prosodia es genérica y promediada; falta la variación de ritmo, énfasis, duda, velocidad | §10.2 completa. El modelo transfiere timbre, no estilo. Toda la variación prosódica del target se pierde |
| **Realismo acústico / artefactos** | El vocoder deja una firma sistemática. El audio es "demasiado limpio": sin ruido de fondo real, sin reverberación de sala, sin efectos de canal ni códec | Sección 3.3 y Figura 3: el habla sintética forma **clusters propios, distintos y separables** de los reales en el espacio de embeddings. EER de 2.86% |
| **Realismo lingüístico** | El texto de entrada es texto escrito, sin disfluencias, sin repeticiones, sin habla espontánea | No abordado en el paper |

Los tres primeros ejes son exactamente los que el paper mide como *ventajas* de su sistema, y por eso son tan buena evidencia: **si un verificador de hablante distingue habla real de sintética con EER de 2.86%, un modelo de ASR entrenado solo sobre habla sintética está aprendiendo la distribución equivocada.** Un clasificador entrenado sobre una distribución que un discriminador separa fácilmente de la distribución objetivo va a sobreajustar los artefactos del generador, no el habla.

Hay un cierre elegante: **el propio paper tropieza con esta misma brecha internamente.** En la Sección 3 tuvieron que entrenar el vocoder de LibriSpeech sobre **espectrogramas predichos por el sintetizador** en lugar de sobre espectrogramas reales, precisamente porque el vocoder entrenado con mels reales no funcionaba bien con mels sintéticos. Es la misma lección en miniatura: **un modelo aguas abajo entrenado sobre datos reales se degrada cuando en inferencia recibe datos sintéticos, y la solución es alinear las distribuciones de entrenamiento e inferencia.**

**Qué dice la literatura de ASR (con la honestidad que corresponde).** Dos referencias representativas, cuyos hallazgos verifiqué por búsqueda pero cuyas cifras detalladas **no verifiqué contra los PDFs originales**:

- **Rosenberg, Zhang, Ramabhadran, Jia, Moreno, Wu, Wu — "Speech Recognition with Augmented Synthesized Speech", ASRU 2019** (arXiv:1909.11699). Nótese que **Ye Jia y Yu Zhang son coautores**: es el equipo de SV2TTS aplicando su propia tecnología a ASR. La conclusión publicada: **sí se logran mejoras aumentando los datos de entrenamiento con material sintetizado, pero permanece una brecha sustancial de desempeño entre reconocedores entrenados sobre habla humana y los entrenados sobre habla sintetizada.** Usan un TTS multi-hablante que aprende espacios latentes de prosodia, hablante y estilo, precisamente para poder inyectar variación.
- **Rossenbach, Zeyer, Schlüter, Ney — "Generating Synthetic Audio Data for Attention-Based Speech Recognition Systems", ICASSP 2020** (arXiv:1912.09257). Extienden un ASR atencional con audio sintético generado por un TTS entrenado **sobre el propio corpus de ASR**. Usan **Global Style Tokens** para embeddings de hablante no supervisados. Reportan hasta **33% de mejora relativa en WER** en un escenario de bajos recursos combinando con modelo de lenguaje y SpecAugment, y cierran más del 50% de la brecha respecto de un experimento oráculo. Hallazgo relevante para nuestro tema: **el sistema con GST superó claramente al que usaba i-vectors** — es decir, la calidad de la representación de hablante del TTS determina cuánto sirve la augmentation. Y las mejoras de datos sintéticos resultaron **mayormente independientes** de las de SpecAugment, o sea aditivas.

**Síntesis honesta de qué funciona y qué no:**

| Práctica | Veredicto |
|---|---|
| Entrenar **solo** con habla sintética | **No funciona** para ASR de propósito general. La brecha es sustancial y consistente en toda la literatura |
| **Mezclar** sintético con real (típicamente el real domina el mix) + fine-tuning final sobre real | **Funciona.** Es la receta estándar y es exactamente lo que la clase afirma |
| Sintético para **cubrir vocabulario nuevo o raro** (nombres propios, términos de dominio, comandos) | **Es el caso de uso más sólido.** Cuando el problema es que ciertas palabras nunca aparecen en el audio de entrenamiento, generarlas sintéticamente ataca la carencia real |
| Sintético para **bajos recursos** con un idioma sin datos | **Funciona con matices.** Rossenbach et al. reportan hasta 33% relativo, pero el TTS necesita datos para existir: hay un problema del huevo y la gallina |
| **Maximizar la diversidad** del sintético (muchos hablantes, prosodia variada, augmentation acústica encima) | **Es el factor determinante.** Toda la literatura converge en que la diversidad —no el volumen— es lo que limita el beneficio |
| Aplicar **SpecAugment / ruido / reverberación / códec** *sobre* el audio sintético | **Ayuda.** Rompe parcialmente los artefactos de "demasiado limpio" del vocoder y acerca las distribuciones |

**El puente conceptual con la clase.** La afirmación de la clase —"hace falta fine-tuning sobre datos reales"— es la formulación aplicada de un principio general de *domain adaptation*: cuando $p_{\text{train}} \neq p_{\text{test}}$, el modelo minimiza el riesgo empírico sobre la distribución equivocada. El pre-entrenamiento sobre sintético sirve para aprender la estructura **compartida** (fonética, léxico, sintaxis, alineamiento acústico-fonético), que es lo que abunda en el sintético; el fine-tuning sobre real sirve para corregir la estructura **específica del dominio** (canal, ruido, prosodia espontánea, artefactos ausentes), que es exactamente lo que el sintético no tiene. Es la misma estructura de argumento que justifica ImageNet → dataset objetivo en visión, o el pre-entrenamiento sobre datos simulados en robótica (donde se lo llama *sim-to-real gap*).

### 13.2. Hilo 2 — Transferencia entre tareas de audio

SV2TTS es un caso de **transfer learning entre tareas de naturaleza opuesta**, y esa es la parte que lo hace interesante conceptualmente más allá del TTS:

$$\underbrace{\text{Verificación de hablante}}_{\text{discriminativa, } p(\text{misma persona} \mid x_1, x_2)} \;\longrightarrow\; \underbrace{\text{Síntesis de habla}}_{\text{generativa, } p(\text{audio} \mid \text{texto}, e)}$$

Lo transferido no son features de bajo nivel ni pesos de una red compartida: es **un espacio de representación completo**, con su geometría, congelado. El sintetizador nunca ajusta el encoder. Y el paper es transparente en que esto no estaba garantizado: *"aunque la red **no está optimizada directamente** para aprender una representación que capture características de hablante relevantes para la síntesis, encontramos que entrenar en una tarea de discriminación de hablante lleva a un embedding que es directamente adecuado para condicionar la red de síntesis"*.

**El principio general que ilustra, y que es el mismo que la clase invoca al hablar de pre-entrenar en idiomas con más datos transcritos:** *la tarea de pre-entrenamiento no tiene que parecerse a la tarea objetivo; tiene que **forzar al modelo a codificar la misma información latente**.* Verificación y síntesis son tareas opuestas en su forma, pero ambas dependen de la misma variable latente: **quién es el hablante**. Una la debe leer, la otra la debe escribir. Comparten el objeto, no la operación.

La misma lógica se aplica en los otros ejemplos que la clase 39 menciona:

- **Pre-entrenar ASR en un idioma con muchos datos transcritos y fine-tunear en uno con pocos** funciona porque la estructura acústico-fonética de bajo nivel (formantes, transiciones, categorías fonéticas amplias) es en buena medida universal. Es lo que hacen XLSR y Whisper multilingüe.
- **wav2vec 2.0 / HuBERT**: pre-entrenamiento **autosupervisado** (predicción contrastiva de segmentos enmascarados) transferido a ASR. Ninguna relación de forma con la tarea objetivo; la misma información latente.
- **Whisper**: entrenamiento **débilmente supervisado** a escala masiva sobre transcripciones de internet, transferido a decenas de tareas.

Y la restricción práctica que SV2TTS ilustra mejor que ninguno: **el desacople de tareas es lo que permite desacoplar los datasets**, y desacoplar los datasets es lo que permite escalar. La Tabla 5 lo dice numéricamente: 18K hablantes de audio basura sin transcribir valen más, para generalizar a voces nuevas, que 1.2K hablantes de audio impecable y transcrito. **El valor de un dataset no es su calidad promedio, es la diversidad que aporta sobre el eje que la tarea necesita.**

---

## 14. Erratas, matices y cosas que se citan mal

### 14.1. La fecha: el slide dice 2019, el paper es de 2018

**Verificado contra el PDF.** El pie de página de la primera página dice textualmente: *"32nd Conference on Neural Information Processing Systems (**NeurIPS 2018**), Montréal, Canada."*

La cita correcta es **2018**. La confusión tiene tres fuentes plausibles, y las tres son razonables:

1. **La marca lateral del PDF dice `2 Jan 2019`.** Pero esa es la fecha del **v4**, una revisión posterior. El identificador `arXiv:1806.04558` codifica **junio de 2018** como fecha de la primera versión (`YYMM` = `1806`). Citar por la fecha de la última revisión es un error frecuente.
2. **NeurIPS 2018 se celebró en diciembre de 2018**, casi en el límite del año, y los proceedings circulan en la frontera. Aun así, el volumen es *Advances in Neural Information Processing Systems 31* (2018).
3. **Ye Jia tiene varios papers de 2019 sobre temas adyacentes** — Translatotron (Interspeech 2019) y cross-language voice cloning (Interspeech 2019) — que se confunden fácilmente con este.

**Cita correcta:**
> Jia, Y., Zhang, Y., Weiss, R. J., Wang, Q., Shen, J., Ren, F., Chen, Z., Nguyen, P., Pang, R., Lopez Moreno, I., & Wu, Y. (2018). *Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis*. En **Advances in Neural Information Processing Systems 31 (NeurIPS 2018)**. arXiv:1806.04558.

Errata menor del material del curso, sin consecuencias sobre el contenido, pero vale la pena tenerla registrada.

### 14.2. "El modelo funciona igual de bien con hablantes no vistos" — falso, y a qué se debe la confusión

Es la lectura errónea más común y viene de mirar solo la Tabla 1 (naturalidad), donde efectivamente **unseen > seen**. La tabla completa exige mirar las dos métricas:

| | Naturalidad | Similitud |
|---|---|---|
| VCTK Seen → Unseen | 4.07 → **4.20** (+0.13) | 4.22 → **3.28** (−0.94) |
| LibriSpeech Seen → Unseen | 3.89 → **4.12** (+0.23) | 3.28 → **3.03** (−0.25) |

**El habla suena igual de bien. Se parece bastante menos al objetivo.** Son cosas distintas y la evaluación las separó deliberadamente (los raters de similitud reciben la instrucción explícita de no juzgar calidad de audio).

Y la explicación del "unseen > seen" en naturalidad **no es que el modelo generalice mejor**. El paper la da en la Sección 3.1: *"Esto es consecuencia de la **utterance de referencia elegida aleatoriamente** para cada hablante, que a veces contiene prosodia despareja y no neutra... la prosodia del habla sintetizada a veces imita la de la referencia."* O sea: es un artefacto del sorteo de la referencia interactuando con la fuga de prosodia de §5.3. Con distintos hablantes de evaluación y distintas referencias sorteadas, el orden podría invertirse. Los intervalos de confianza (±0.05-0.06) no cubren la diferencia, pero la varianza real entre sorteos de referencia no está caracterizada.

### 14.3. A qué configuración corresponde cada MOS — mapa de desambiguación

Este paper tiene **siete tablas de MOS** con configuraciones distintas, y mezclarlas es fácil. Mapa:

| Cifra | Tabla | Configuración exacta |
|---|---|---|
| **4.12 ± 0.05** naturalidad | Tablas 1 y 5 | LibriSpeech, **unseen**, encoder de 18K hablantes, embedding 256-d |
| **3.03 ± 0.09** similitud | Tablas 2 y 5 | LibriSpeech, **unseen**, encoder de 18K hablantes |
| **4.20 ± 0.06** naturalidad | Tabla 1 | **VCTK**, unseen, mismo encoder de 18K |
| **3.28 ± 0.07** similitud | Tabla 2 | **VCTK**, **unseen**. Ojo: 3.28 aparece **dos veces** en la Tabla 2 — también es la similitud de LibriSpeech **seen**. Coincidencia numérica pura |
| **4.22 ± 0.06** similitud | Tabla 2 | VCTK, **seen** |
| **3.73 / 2.23** | Tabla 5 | Encoder sobre **LS-Clean, 1.2K hablantes, embedding de 64-d y arquitectura reducida**. No es el modelo propuesto |
| **3.65 ± 0.06** naturalidad | Tabla 6 | **Hablantes ficticios**, con encoder de LS-Other+VC+VC2 (**8.4K**, no 18K) |
| **3.88 / 2.64** | Tabla 7 (Apéndice A) | Baseline separado de 64-d, para comparar contra entrenamiento conjunto |
| **5.08%** EER | Tablas 4 y 5 | Sintetizador LibriSpeech, hablantes LibriSpeech unseen, encoder de 18K |

**Regla práctica:** si se cita "el MOS de SV2TTS", hay que decir cuál de los dos (naturalidad o similitud), sobre cuál corpus (VCTK o LibriSpeech) y en cuál condición (seen o unseen). Sin esos tres calificadores el número no significa nada.

### 14.4. Otros matices

- **"5 segundos de audio bastan"** es correcto pero incompleto: la Tabla 9 muestra que **2 segundos ya están cerca del óptimo** y que **más de 5 no ayuda** — de hecho a 10 segundos la similitud **baja** (3.28 → 3.18) y la EER **sube** (10.46% → 11.50%). No es "5 segundos es el mínimo"; es "5 segundos es el techo".
- **"El sistema es zero-shot"** es correcto en el sentido de que no actualiza parámetros. Pero la Tabla 3 muestra que la generalización **out-of-domain** puede ser muy mala (similitud 1.82 para el sintetizador de VCTK sobre hablantes de LibriSpeech). El zero-shot funciona dentro de un rango acústico y de acento razonablemente cercano al de entrenamiento.
- **"El vocoder es agnóstico al hablante"** requiere el matiz de §6.2: no está *condicionado* explícitamente, pero sí debe ser *entrenado* sobre muchos hablantes, se entrena uno por corpus, y en LibriSpeech hubo que entrenarlo sobre espectrogramas predichos.
- **"El encoder es Tacotron/es parte del TTS"** — no. El speaker encoder es una red LSTM **completamente separada**, entrenada antes, con parámetros **congelados** durante el entrenamiento del sintetizador.
- **La fórmula de GE2E no está en este paper.** Se cita de Wan et al., ICASSP 2018. Si alguien atribuye la formulación de la pérdida a SV2TTS, está mal atribuida.
- **El VCTK "de 109 hablantes"** entrena con **98**: 11 quedan held out (Sección 3 y Tabla 4, columna "Training Speakers"). El texto de la Sección 3.2 lo redondea a "solo 100 hablantes".
- **VoxCeleb y VoxCeleb2 no se usaron en el modelo principal.** Aparecen solo en la ablación de la Tabla 5 (filas 3 y 4). El modelo que produce las cifras de las Tablas 1-4 usa el **corpus interno propietario de 18K hablantes**, que **no es público**. Esto es relevante para reproducibilidad: la mejor configuración del paper **no es reproducible fuera de Google**. La mejor configuración reproducible es la fila LS-Other + VC + VC2 (8.4K hablantes): naturalidad **3.82**, similitud **2.54**, EER **10.14%** — bastante por debajo de las cifras que se citan del paper.
- **No hay experimentos de interpolación entre embeddings de hablantes reales.** Solo muestreo uniforme sobre la esfera (Sección 3.6). Y no hay aritmética vectorial en el espacio de hablantes.

---

## 15. Cómo se ve hoy

### 15.1. El pipeline de inferencia, comentado

El punto que este pseudo-código busca dejar claro es lo barato que es la adaptación: **un solo forward pass de una LSTM chica**, y de ahí en adelante es TTS normal.

```python
import torch
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────
# 1. SPEAKER ENCODER  —  audio de referencia  →  embedding de 256-d
#    Sección 2.1: 3 LSTM de 768 celdas, cada una con proyección a 256.
#    Entrenado con pérdida GE2E (Wan et al., ICASSP 2018) sobre 18K
#    hablantes de audio SIN TRANSCRIBIR. Parámetros CONGELADOS.
# ─────────────────────────────────────────────────────────────────────
class SpeakerEncoder(torch.nn.Module):
    def __init__(self, n_mels=40, hidden=768, proj=256, layers=3):
        super().__init__()
        # LSTMP: LSTM con proyección recurrente. PyTorch lo soporta con proj_size.
        self.lstm = torch.nn.LSTM(n_mels, hidden, layers,
                                  batch_first=True, proj_size=proj)

    def forward(self, mel_40ch):            # (B, T, 40) log-mel
        out, _ = self.lstm(mel_40ch)
        # Sección 2.1: "L2-normalizando la salida de la capa superior
        # EN EL FRAME FINAL". No es un pooling sobre toda la secuencia.
        return F.normalize(out[:, -1, :], p=2, dim=-1)   # (B, 256), ||e||=1


@torch.no_grad()
def embed_utterance(encoder, wav, sr):
    """
    Sección 2.1, inferencia: una utterance de longitud arbitraria se parte
    en ventanas de 800 ms con 50% de solapamiento; la red corre sobre cada
    ventana de forma independiente y las salidas se PROMEDIAN y RE-NORMALIZAN.

    Detalle: durante el ENTRENAMIENTO los segmentos son de 1.6 s, no de 800 ms.
    """
    win, hop = int(0.800 * sr), int(0.400 * sr)          # 800 ms, 50% overlap
    windows = [wav[i:i + win] for i in range(0, len(wav) - win + 1, hop)]
    if not windows:                                       # audio < 800 ms
        windows = [torch.nn.functional.pad(wav, (0, win - len(wav)))]

    mels = torch.stack([log_mel(w, sr, n_mels=40) for w in windows])
    e = encoder(mels).mean(dim=0)                         # promedio euclidiano
    return F.normalize(e, p=2, dim=-1)                    # proyectado a la esfera


# ─────────────────────────────────────────────────────────────────────
# 2. SINTETIZADOR  —  Tacotron 2 condicionado
#    Sección 2.2: el embedding se CONCATENA A LA SALIDA DEL ENCODER DE
#    TEXTO EN CADA PASO TEMPORAL, antes de la atención. Un único punto
#    de inyección (a diferencia de Deep Voice 2, que inyectaba en varios).
# ─────────────────────────────────────────────────────────────────────
class MultiSpeakerTacotron2(torch.nn.Module):
    def forward(self, phoneme_ids, speaker_emb):
        # H: (B, L, d_h) — un vector por fonema
        H = self.text_encoder(phoneme_ids)

        # ── LA LÍNEA QUE DEFINE EL PAPER ────────────────────────────
        # e se replica a lo largo de TODA la secuencia de texto.
        # Consecuencia 1: como sum_l alpha[t,l] = 1, el vector de contexto
        #   c_t = sum_l alpha[t,l] * H_cond[l] arrastra e INTACTO en todos
        #   los pasos del decoder. Condicionamiento persistente y gratuito.
        # Consecuencia 2: las KEYS de la atención dependen de e, así que el
        #   ALINEAMIENTO (duración por fonema → velocidad de habla, ritmo)
        #   puede adaptarse al hablante. Esto es lo que un condicionamiento
        #   inyectado solo en el decoder NO permitiría.
        e = speaker_emb.unsqueeze(1).expand(-1, H.size(1), -1)   # (B, L, 256)
        H_cond = torch.cat([H, e], dim=-1)                       # (B, L, d_h+256)
        # ────────────────────────────────────────────────────────────

        # Decoder autorregresivo con atención sensible a la localización.
        # Target: 80 canales mel, ventana 50 ms, hop 12.5 ms, log-compresión.
        # Pérdida de entrenamiento: L2 + L1 (el L1 añadido da robustez
        # sobre datos ruidosos — Sección 2.2).
        mel, stop, align = self.decoder(H_cond)
        return mel + self.postnet(mel), stop, align


# ─────────────────────────────────────────────────────────────────────
# 3. VOCODER  —  WaveNet autorregresivo, 30 capas dilatadas
#    Sección 2.3: NO recibe el speaker embedding. Toda la identidad ya
#    está en el mel (F0 vía espaciado armónico, formantes, tilt espectral).
#    Pero SÍ debe entrenarse sobre muchos hablantes, y en la práctica el
#    paper entrenó uno por corpus (VCTK 24 kHz / LibriSpeech 16 kHz).
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def clone_voice(encoder, synthesizer, vocoder, ref_wav, sr, text):
    e   = embed_utterance(encoder, ref_wav, sr)          # ~5 s bastan; 2 s casi
    ph  = text_to_phonemes(text)                         # fonemas > grafemas
    mel = synthesizer(ph, e.unsqueeze(0))[0]
    return vocoder(mel)                                  # 24k pasos secuenciales
                                                         # por segundo de audio
```

**Coste de la adaptación a un hablante nuevo:** un forward de tres LSTM sobre unas pocas ventanas de 800 ms. Milisegundos en CPU. **Cero pasos de gradiente.** Ese es el resultado del paper reducido a una línea de código.

**Dónde está el cuello de botella real:** en la última línea. WaveNet autorregresivo genera muestra por muestra; a 24 kHz son 24,000 pasos secuenciales por segundo de audio. Ninguna de las tres primeras etapas importa frente a eso. Es el motivo por el que ninguna implementación práctica usó WaveNet: `Real-Time-Voice-Cloning` lo reemplazó por **WaveRNN**, y la industria migró a **HiFi-GAN**, **Parallel WaveGAN** o **BigVGAN**, todos no autorregresivos y varios órdenes de magnitud más rápidos.

### 15.2. Implementaciones actuales

| Opción | Qué es | Nota |
|---|---|---|
| **`Real-Time-Voice-Cloning`** (Corentin Jemine) | Reimplementación fiel de las tres etapas de SV2TTS, con WaveRNN | Valor **pedagógico**: es el mapa 1:1 con el paper. Modelos de 2019, calidad muy por debajo del estado del arte actual. Sin mantenimiento activo |
| **`resemblyzer`** | Solo el speaker encoder (GE2E) del repo anterior, empaquetado | Sigue siendo útil de forma independiente: diarización, verificación, deduplicación de hablantes, y **métrica de similitud para evaluar clonación** |
| **Coqui TTS / `XTTS-v2`** | Zero-shot multilingüe (17 idiomas), arquitectura moderna | Fue el estándar open-source. Coqui cerró como empresa en 2024; el código y los pesos siguen disponibles, con licencia que conviene revisar |
| **`YourTTS`** | VITS + condicionamiento por speaker embedding, zero-shot multilingüe | Disponible dentro de Coqui TTS |
| **`SpeechBrain` / `WeSpeaker` / `pyannote`** | Speaker embeddings modernos (ECAPA-TDNN, ResNet) | Reemplazo directo y superior del encoder LSTM+GE2E del paper para cualquier uso |
| **`F5-TTS`, `Fish-Speech`, `MaskGCT`** | Generación de última hornada (flow matching, modelos de códec no autorregresivos) | Calidad y velocidad muy superiores. El diseño de SV2TTS ya no es competitivo, solo instructivo |
| **ElevenLabs / PlayHT / Azure Custom Neural Voice** | Servicios comerciales | Todos exigen **verificación de consentimiento** del hablante para clonación. Es la respuesta de la industria al problema de §11 |

**La forma correcta de leer SV2TTS hoy:** no como una arquitectura a implementar, sino como **el paper que estableció que el condicionamiento por speaker embedding funciona en zero-shot, y que la diversidad de hablantes del encoder —no la calidad del corpus TTS— es la variable que gobierna la generalización a voces nuevas.** Ese resultado sobrevivió a todas las arquitecturas que lo reemplazaron.
