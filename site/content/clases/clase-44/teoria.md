---
title: "Teoría - Aplicaciones de Audio y Video"
weight: 10
math: true
---

> **Recorrido de la Clase 44** del Diplomado IA UC (Carlos Aspillaga, DCC PUC), la **última del programa**. Cuarenta y seis diapositivas en tres tiempos: una recapitulación de todo lo visto, un recorrido por siete aplicaciones audiovisuales, y el desarrollo de la última de ellas —los deep fakes— hasta el laboratorio final.

---

# Parte 1 — La recapitulación (diapositivas 2-5)

La clase abre con cuatro diapositivas que enumeran el temario completo del diplomado. No son un trámite: son el argumento de la clase. Aparecen en dos pasos, y cada paso agrega una imagen que le da sentido.

## 1.1. Primer bloque: los fundamentos y las dos modalidades clásicas

| **Deep Learning** | **Texto** | **Imágenes** |
|---|---|---|
| Aprendizaje de máquina y deep learning | NLP clásico | Redes *fully-convolutional* |
| Redes neuronales | Modelos de lenguaje | Reconocimiento de objetos |
| CNNs | Embeddings de palabras | Detección de poses y rostros |
| RNN | ELMo, BERT, GPT | Reconocimiento visual de texto |
| ResNet, Inception, etc. | Summarization | Recomendación con imágenes y texto |
| Transformer | Modelos pregunta-respuesta | VQA, captions |
| PyTorch | | |
| Grafos de cómputo | | |
| Inicialización de pesos | | |
| Funciones de activación | | |
| Funciones de pérdida | | |
| Regularización | | |
| Tareas auxiliares | | |
| Visualización de pesos | | |
| Mecanismos de optimización | | |
| Paradigma sequence-to-sequence | | |
| Fine-tuning | | |
| Data augmentation | | |
| Mecanismos de atención | | |
| Hardware para deep learning | | |

La diapositiva siguiente repite exactamente la misma lista y agrega, al costado, la fotografía de **una caja de herramientas abierta y llena**.

## 1.2. Segundo bloque: lo avanzado, audio y video, y las aplicaciones

| **IA Avanzado** | **Audio y Video** | **Aplicaciones** |
|---|---|---|
| Redes relacionales | Introducción a audio y video | Clasificación de imágenes |
| Olvido catastrófico | Datasets y herramientas públicas | Clasificación de texto |
| Aprendizaje incremental | CNNs para audio y video | Generación de texto |
| Redes neuronales de grafos | Modelos pre-entrenados | Generación de resúmenes |
| Meta-learning | Reconocimiento de acciones en video | Pregunta-respuesta |
| GANs | Tracking en video | Visual question answering |
| Self-supervised learning | Reconocimiento de voz | Recomendación |
| Aprendizaje reforzado | Transcripción de audio | etc… |
| Imitation learning | Aplicaciones de audio | |
| Modelos con memoria externa | | |
| Computadores neuronales | | |
| Razonamiento en deep learning | | |

Y de nuevo la repetición, esta vez con una flecha azul que dice **"USTEDES"** apuntando a la silueta de dos superhéroes de pie sobre un risco.

{{< concept-alert type="clave" >}}
La metáfora es explícita y vale desarmarla, porque ordena toda la clase: **el temario es la caja de herramientas; ustedes son quienes la usan**. Lo que viene después —siete aplicaciones que parecen magia— está construido enteramente con piezas de esas listas. La diapositiva 6 lo dice sin rodeos: *"aplicaciones sorprendentes de audio y video **usando las cosas que ya conocemos**"*.

Es un cierre pedagógico, no un anexo: la clase quiere demostrar que la distancia entre "lo que estudiamos" y "lo que sale en las noticias" es menor de lo que parece.
{{< /concept-alert >}}

## 1.3. El temario, mapeado al material del curso

Como esta es la clase de cierre, vale usar su propia lista como índice. Cada ítem del temario, con la clase donde se desarrolla:

**Deep Learning.** Redes neuronales, grafos de cómputo, inicialización, activaciones ([Clase 05](/clases/clase-05), [07](/clases/clase-07)) · funciones de pérdida y regularización ([Clase 08](/clases/clase-08)) · CNNs ([Clase 05](/clases/clase-05), [09](/clases/clase-09)) · optimización y *learning rate* ([Clase 10](/clases/clase-10)) · RNN ([Clase 11](/clases/clase-11)) · data augmentation, fine-tuning y transfer learning ([Clase 12](/clases/clase-12)) · seq2seq y atención ([Clase 13](/clases/clase-13)) · Transformer ([Clase 14](/clases/clase-14)) · hardware y deployment ([Clase 19](/clases/clase-19)).

**Texto.** NLP clásico ([Clase 16](/clases/clase-16)) · embeddings de palabras ([Clase 18](/clases/clase-18)) · ELMo, BERT, GPT ([Clase 20](/clases/clase-20)) · summarization ([Clase 22](/clases/clase-22)) · pregunta-respuesta ([Clase 24](/clases/clase-24)).

**Imágenes.** Reconocimiento de objetos ([Clase 15](/clases/clase-15)) · detección de poses y rostros ([Clase 17](/clases/clase-17)) · reconocimiento visual de texto ([Clase 21](/clases/clase-21)) · VQA y captions ([Clase 23](/clases/clase-23)) · recomendación con imágenes y texto ([Clase 25](/clases/clase-25)).

**IA Avanzado.** Meta-learning ([Clase 26](/clases/clase-26)) · redes neuronales de grafos ([Clase 27](/clases/clase-27)) · self-supervised learning ([Clase 28](/clases/clase-28)) · GANs y modelos generativos ([Clase 29](/clases/clase-29)) · modelos con memoria externa y computadores neuronales ([Clase 30](/clases/clase-30)) · aprendizaje reforzado ([Clase 31](/clases/clase-31)) · olvido catastrófico y aprendizaje incremental ([Clase 32](/clases/clase-32)) · imitation learning ([Clase 33](/clases/clase-33)) · razonamiento ([Clase 34](/clases/clase-34)).

**Audio y Video.** Introducción al audio ([Clase 35](/clases/clase-35)) · introducción al video ([Clase 36](/clases/clase-36)) · datasets y herramientas ([Clase 37](/clases/clase-37)) · CNNs y modelos pre-entrenados para video ([Clase 38](/clases/clase-38)) · modelos de DL para audio ([Clase 39](/clases/clase-39)) · reconocimiento de acciones ([Clase 40](/clases/clase-40)) · reconocimiento de voz y de hablante ([Clase 41](/clases/clase-41)) · tracking ([Clase 42](/clases/clase-42)) · aplicaciones audiovisuales ([Clase 43](/clases/clase-43)).

---

# Parte 2 — Siete aplicaciones (7-32)

El formato se repite para cada una: una diapositiva *"¿Qué es?"* con un diagrama de entrada → **IA** → salida y el objetivo en una línea, una de ejemplos con papers y arquitecturas, y una o dos de demos en video.

## 2.1. Speech Reconstruction from Silent Videos

> **Objetivo:** reconstruir los features de audio a partir del video.

- [Vid2Speech](/papers/vid2speech-ephrat-2017) (Ephrat y Peleg, 2017) — encoder-decoder CNN.
- *Vocoder-Based Speech Synthesis from Silent Videos* (Michelsanti et al., 2020) — convolución 3D + GRU.

Nótese que el objetivo dice **features de audio**, no texto: se salta el lenguaje y se predice directamente la representación acústica, con lo que se conserva prosodia y entonación que el texto no codifica.

## 2.2. Face Reconstruction from Voice

> **Objetivo:** reconstruir la cara a partir de la voz.

- [Speech2Face](/papers/speech2face-oh-2019) (Oh et al., 2019) — encoder-decoder CNN.

Es la dirección inversa de la anterior, y juntas forman un par simétrico. Conviene leer el paper hasta su sección de **consideraciones éticas**: los autores declaran que el método *no puede* recuperar la identidad de una persona y documentan el sesgo demográfico de su conjunto de entrenamiento.

## 2.3. Audio Source Separation in Video

> **Objetivo:** identificar las distintas fuentes de sonido y obtener su sonido aislado.

- [Looking to Listen at the Cocktail Party](/papers/looking-to-listen-ephrat-2018) (Ephrat et al., 2018) — CNNs + BiLSTM.
- [Learning to Separate Object Sounds by Watching Unlabeled Video](/papers/separating-object-sounds-gao-2018) (Gao et al., 2018) — CNNs + factorización de matrices no negativas.

El primero usa **rostros** para separar voces; el segundo, **objetos** para separar sonidos ambientales.

## 2.4. Audio-Visual Speech Enhancement

> **Objetivo:** aislar la voz en un video ruidoso.

- *CochleaNet* (Gogate et al., 2019) — CNNs + LSTMs + MLPs.
- *Face Landmark-based Speaker-Independent Audio-Visual Speech Enhancement* (Morrone et al., 2018) — BiLSTM apiladas.

La diferencia con la anterior es de objetivo: separación entrega **todas** las fuentes; *enhancement* entrega solo la que interesa, tratando el resto como ruido.

## 2.5. Audio-Video Synchronization

> **Objetivo:** alinear audio y video cuando hay desfase entre ellos.

- *Out of time: automated lip sync in the wild* (Chung y Zisserman, 2016) — CNNs. Es el trabajo conocido como **SyncNet**.

La ilustración de la clase es elocuente: la palabra HOTEL escrita dos veces, desalineada arriba y alineada abajo.

{{< concept-alert type="recordar" >}}
SyncNet merece una nota que la clase no hace. Se entrena con una tarea autosupervisada —*¿este fragmento de audio corresponde a este fragmento de video?*— usando como negativos los desplazamientos temporales del mismo video. Es exactamente el principio de correspondencia audiovisual de la [Clase 43](/clases/clase-43).

Y tiene un uso que va más allá de sincronizar: **detectar deep fakes**. Un video manipulado suele tener desincronización sutil entre labios y fonemas, y un detector de sincronía la encuentra. El mismo modelo sirve para alinear y para desconfiar.
{{< /concept-alert >}}

## 2.6. Upscaling / Super-resolution

> **Objetivo:** incrementar la resolución de instancias de audio y video.

Y una segunda diapositiva con una sola anotación sobre el diagrama, que es la definición más precisa de todo el bloque:

> ***informed guess***

- DLSS de NVIDIA y DirectML de AMD — autoencoder convolucional.
- [Audio Super Resolution with Neural Networks](/papers/audio-superres-kuleshov-2017) (Kuleshov et al., 2017) — bloques convolucionales de sobremuestreo y submuestreo.

{{< concept-alert type="clave" >}}
Esas dos palabras son literalmente correctas y merecen tomarse en serio. Bajar la resolución **destruye** información de forma irreversible: con factor 4 sobre parches binarios, **3855 imágenes distintas producen la misma salida** (medido en la [práctica](practica)). Ningún procesamiento las distingue.

Lo que hace un modelo de super-resolución es elegir un elemento de esa preimagen según un **prior aprendido**. No recupera información: la aporta. La [profundización](profundizacion) desarrolla las dos consecuencias —por qué el óptimo en error cuadrático se ve borroso, y por qué esto significa que la super-resolución no es una herramienta forense—.
{{< /concept-alert >}}

## 2.7. Deep Fakes

> **Objetivo:** reemplazar partes de audio y/o video por una versión modificada intencionalmente.

Seis diapositivas encadenadas, cada una agregando una viñeta:

> Aunque es altamente controversial, existen aplicaciones muy útiles:
> - Reemplazo de actores que murieron o envejecieron (cine).
> - Recrear personajes históricos con fines educacionales.
> - Visualización de cómo se vería la persona usando el producto.
> - **Inmortalidad digital.**
> - **Traducción simultánea:** sintetizar video para que gesticule la frase traducida por un intérprete.
> - **Sintetizador de voz para personas que perdieron la voz.**

---

# Parte 3 — El método (33-45)

## 3.1. El paper

La clase enlaza el PDF de NeurIPS 2019 sin nombrarlo. Es **[First Order Motion Model for Image Animation](/papers/fomm-siarohin-2019)** (Siarohin, Lathuilière, Tulyakov, Ricci y Sebe). El dataset que menciona es **VoxCeleb**, con la nota correcta al pie: *"en realidad también usan datasets de videos a cuerpo completo"*.

## 3.2. Los cuatro pasos

**Tracking de puntos.** Tres cuadros del mismo hablante con una docena de puntos de colores sobre el rostro, que se mueven con él.

**Estimación de la transformación afín entre cada frame (y para cada keypoint).** El diagrama muestra un punto ampliado y, debajo, una grilla que se inclina — la transformación local.

**Entrenar para regenerar el video.** Entrada: un cuadro de aspecto más los puntos de cada instante. Salida: los cuadros reconstruidos.

**Separación de aspecto y movimiento.** Dos bocadillos rotulan la entrada: *"información de aspecto"* sobre la imagen, *"información de movimiento"* sobre las secuencias de puntos.

**Generación del deep fake: ¡reemplazar frame inicial y listo!** La misma figura con la imagen de aspecto tachada y sustituida por otra persona; la salida es esa segunda persona moviéndose como la primera.

{{< concept-alert type="advertencia" >}}
Tres precisiones sobre esta sección, desarrolladas en la [profundización](profundizacion):

**Los puntos clave no son landmarks faciales.** Son aprendidos **sin supervisión**: nadie le dice a la red dónde está la comisura de los labios. Actúan como cuello de botella que fuerza una representación compacta del movimiento, y por eso el método funciona igual sobre caricaturas o cuerpos completos, donde no existen landmarks predefinidos.

**El nombre del paper es su contribución.** "Primer orden" es la expansión de Taylor: además de la posición, el detector emite el **jacobiano** en cada punto, que expresa rotación y escala locales. Medido: sobre una rotación pura, la representación con jacobiano tiene error **cero numérico** mientras que la de solo posiciones crece linealmente con el ángulo.

**Falta la máscara de oclusión, y sin ella no funcionaría.** Cuando una cabeza gira aparece una oreja que **no está** en la imagen fuente. Ningún campo de deformación puede producirla. El generador incluye una máscara que separa *lo que se puede deformar* de *lo que hay que inventar*.
{{< /concept-alert >}}

## 3.3. El audio

> Jia et al.: *Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis*

Es [SV2TTS](/papers/sv2tts-jia-2018), ya visto en el diplomado. La diapositiva final del método lo integra con la misma estructura que el video: entrada de **información de voz** (features del hablante) más **información de contenido** (el texto *"This is a red apple"*), y salida el espectrograma sintetizado.

La simetría con la parte visual es exacta y es el remate conceptual de la clase:

| | Se conserva | Se transfiere |
|---|---|---|
| **Video (FOMM)** | aspecto (quién) | movimiento (qué hace) |
| **Audio (SV2TTS)** | timbre (quién) | contenido (qué dice) |

Es la misma factorización que la [Clase 41](/clases/clase-41) usó para separar reconocimiento de voz de reconocimiento de hablante — ahí para **analizar**, aquí para **generar**.

## 3.4. Cierre

La clase termina con una diapositiva que dice *"Deep Fakes: Laboratorio"* y el mismo fotograma de Obama con que abrió.

---

## Lo que la clase deja fuera

Tres huecos, desarrollados en la [profundización](profundizacion):

1. **Detección.** Seis diapositivas de aplicaciones útiles y ninguna sobre cómo se detectan estos medios ni sobre los daños documentados. [FaceForensics++](/papers/faceforensics-rossler-2019) aporta la mitad que falta, con el resultado incómodo de que **la detección generaliza mal** entre métodos y se degrada con la compresión.

2. **La distinción entre *face swap* y *reenactment*.** FOMM anima una imagen fija; no reemplaza un rostro dentro de un video existente. La diferencia cambia qué hace falta —una foto en vez de miles— y qué se obtiene.

3. **Qué pasó después de 2019.** El método de la clase es de 2019; entre 2022 y 2024 la generación de video pasó a modelos de difusión a gran escala, con capacidades y limitaciones distintas.

---

**Siguiente:** [Profundización](profundizacion) — la aproximación de primer orden y cuánto vale el jacobiano, el "informed guess" hecho preciso, y la asimetría entre generar y detectar. Después, la [práctica](practica): ambos mecanismos implementados y medidos, en triple framework.
