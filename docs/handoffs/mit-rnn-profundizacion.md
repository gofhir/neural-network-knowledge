# Profundizacion: MIT 6.S191 (2020) Lecture 2 -- RNNs

> Material complementario al video de Ava Soleimany sobre Recurrent Neural Networks. Esta nota se concibe como **acompanamiento** del lecture original y de los apuntes ya disponibles en el curso UC: no re-deriva matematicas que ya aparecen en [Profundizacion de Clase 11](/clases/clase-11/profundizacion), sino que contextualiza, identifica los papers que la clase MIT cita o alude, y mapea coberturas entre ambas fuentes.

---

## 1. Contexto del curso MIT 6.S191

**MIT 6.S191 -- Introduction to Deep Learning** es el curso introductorio de aprendizaje profundo del MIT, dictado intensivamente cada enero (IAP, *Independent Activities Period*) desde 2017 por **Alexander Amini** y **Ava Soleimany**. Ambos eran estudiantes de doctorado en CSAIL cuando lanzaron el curso, y han continuado reeditandolo y actualizandolo cada ano. El **Lecture 2 -- Recurrent Neural Networks** corresponde a la edicion 2020 y es presentado por Ava Soleimany.

El formato del curso es muy particular: **una semana intensiva** con clases de aproximadamente una hora cada una, acompanadas de software labs en TensorFlow/Keras, charlas invitadas de la industria (Google, NVIDIA, IBM) y un competition final. Todo el material -- slides, videos, codigo -- es **abierto y gratuito**, lo que lo ha convertido en uno de los recursos de referencia mundiales para introducirse al deep learning aplicado.

Es relevante para nuestro curso UC por varias razones:

1. **Cobertura paralela**: el lecture 2 cubre RNNs, LSTM, encoder-decoder y attention -- exactamente la materia de las clases 11, 12 y 13 del curso.
2. **Estilo aplicado**: complementa el enfoque mas formal del curso UC con intuiciones visuales y demos en codigo.
3. **Conexion con papers**: aunque la clase MIT no abruma con citas, su backbone se apoya en una serie de papers fundamentales que ya documentamos en `/papers/`.

La edicion 2020 es un buen punto de corte historico: es **anterior a la explosion de los Transformers como arquitectura dominante de NLP** (que ocurre entre 2018 y 2021), por lo que aun trata RNNs y attention como herramientas centrales y no como "predecesores historicos". En 2024-2025 la perspectiva cambia, como veremos en la seccion 4.

---

## 2. Papers citados o aludidos en la clase MIT

A pesar de que la clase MIT no introduce explicitamente la mayoria de las citas, su material descansa sobre un conjunto bien identificado de papers fundamentales. Para cada uno, indicamos el aporte y el momento de la clase MIT donde aparece su huella.

### LSTM -- Hochreiter & Schmidhuber (1997)
Cross-link: [/papers/lstm-hochreiter-1997](/papers/lstm-hochreiter-1997)

El paper que introduce **Long Short-Term Memory**, la arquitectura sobre la que descansa toda la segunda mitad del lecture MIT (slides 56-72). Hochreiter y Schmidhuber proponen un mecanismo de **constant error carousel** mediante una *cell state* que se actualiza de forma aditiva, evitando el desvanecimiento del gradiente. La intuicion de Soleimany (forget gate / input gate / output gate como "compuertas que regulan el flujo de informacion") es una sintesis pedagogica de la formulacion original mas el refinamiento de Gers, Schmidhuber y Cummins (2000) que introdujo el forget gate moderno.

### GRU -- Cho et al. (2014)
Cross-link: [/papers/gru-cho-2014](/papers/gru-cho-2014)

La **Gated Recurrent Unit** simplifica LSTM combinando forget e input gate en una sola **update gate**, y eliminando la cell state separada. Soleimany la menciona brevemente al final de la seccion de gating como "una alternativa mas eficiente". El paper original de Cho et al. fue publicado en el contexto de seq2seq para traduccion automatica, y de hecho es el mismo trabajo donde se introduce la nocion de RNN encoder-decoder que la clase usa despues.

### Pascanu, Mikolov & Bengio (2013) -- "On the difficulty of training RNNs"
Cross-link: [/papers/difficulty-training-rnns-pascanu-2013](/papers/difficulty-training-rnns-pascanu-2013)

Este paper es el **fundamento teorico del problema de vanishing/exploding gradients** que Soleimany describe en los slides 41-48. Soleimany no lo cita explicitamente pero su exposicion -- analizar el producto de jacobianos a traves del tiempo y proponer **gradient clipping** como solucion para exploding -- proviene directamente de este trabajo. La derivacion formal completa esta en [Profundizacion de Clase 11, Parte II](/clases/clase-11/profundizacion#parte-ii-an%C3%A1lisis-formal-del-vanishing-gradient).

### Sutskever, Vinyals & Le (2014) -- "Sequence to Sequence Learning with Neural Networks"
Cross-link: [/papers/seq2seq-sutskever-2014](/papers/seq2seq-sutskever-2014)

Es el paper que **define el patron encoder-decoder** que Soleimany describe en los slides 76-78 para traduccion automatica. Sutskever et al. demuestran que dos LSTMs apiladas (una encoder y una decoder) con context vector unico pueden alcanzar estado del arte en WMT 2014 (ingles-frances). La clase MIT presenta exactamente esta arquitectura, incluyendo el truco de **invertir la secuencia de entrada** (que Sutskever reporta como crucial pero no es central en la pedagogia de Soleimany). La discusion se profundiza en la [Clase 13 del curso UC](/clases/clase-13/teoria).

### Bahdanau, Cho & Bengio (2015) -- "Neural Machine Translation by Jointly Learning to Align and Translate"
Cross-link: [/papers/bahdanau-attention-2015](/papers/bahdanau-attention-2015)

El paper donde **nace el mecanismo de atencion** en deep learning. Soleimany usa exactamente la formulacion de Bahdanau (slides 80-83): score aditivo $V^T \tanh(W_1 s_{t-1} + W_2 h_i)$, softmax sobre los scores, context vector ponderado $C_t = \sum \alpha_{t,i} h_i$. El ejemplo de alineamiento palabra-a-palabra entre lenguajes que la clase muestra es esencialmente la figura 3 del paper original.

### Vinyals et al. (2015) -- "Show and Tell: A Neural Image Caption Generator"
Cross-link: [/papers/show-and-tell-vinyals-2015](/papers/show-and-tell-vinyals-2015)

Soleimany usa **image captioning** como ejemplo de arquitectura *one-to-many* (slide 84-85). El paper fundacional es Show and Tell: una CNN preentrenada extrae features de la imagen, y una LSTM las usa como estado inicial $h_0$ para generar la descripcion palabra a palabra. El refinamiento con attention espacial que la clase menciona corresponde al paper "Show, Attend and Tell" de Xu et al. (2015), tambien disponible en `/papers/show-attend-tell-xu-2015`. La extension con bottom-up attention (Anderson et al., 2018, slide 88) tambien esta documentada en `/papers/bottom-up-attention-anderson-2018`.

---

## 3. Comparacion lado-a-lado con la Clase 11 del curso UC

Ambos cursos cubren material muy similar pero con enfasis distintos. La siguiente tabla resume coberturas:

| Tema | MIT 6.S191 L2 | UC Clase 11 + Profundizacion |
|---|---|---|
| Motivacion datos secuenciales | Si, con ejemplos visuales | Si, mas formal |
| Limitaciones de MLP/ventana fija | Detalle pedagogico (3 ideas malas) | Mencion breve |
| Ecuacion recurrente $h_t = f(h_{t-1}, x_t)$ | Si | Si |
| Configuraciones (1-to-N, N-to-1, N-to-N, seq2seq) | Si, central | Si, en clase 13 |
| BPTT setup | Conceptual con grafo desplegado | **Derivacion completa con cadena de gradientes** |
| Vanishing/exploding gradient -- intuicion | Si | Si |
| Vanishing gradient -- analisis espectral formal | No | **Si, con valores singulares y norma de jacobianos** |
| Gradient clipping | Si, ecuacion | Si, mas detalle |
| LSTM equations | Si, completas | Si, completas |
| Por que LSTM evita vanishing | Argumento intuitivo | **Demostracion formal** |
| GRU equations | Mencion | Si, completas |
| Comparacion LSTM vs GRU | Mencion breve | Si, con criterios de eleccion |
| Encoder-decoder seq2seq | Si | Si, en clase 13 |
| Attention de Bahdanau | Si, central en la clase | Si, en clase 13 |
| Image captioning + attention espacial | Si | No cubierto en clases 11-13 |
| Demos en codigo | Si (lab asociado) | Notebooks en clase 11 |

### Que profundiza MIT que la Clase 11 no cubre

- **Pedagogia visual**: las "tres ideas malas" (ventana pequena, bag of words, ventana grande sin sharing) son una motivacion pedagogica excelente que el curso UC no desarrolla con tanto detalle.
- **Aplicaciones visuales**: image captioning, attention espacial sobre imagenes, bottom-up attention -- la Clase 11 UC es mas pura en NLP.
- **Demos en vivo**: la generacion de Shakespeare y de musica con character-level RNN es un *hook* didactico potente.

### Que profundiza la Clase 11 UC que MIT no cubre

- **BPTT formal**: derivacion paso a paso con $\delta_t^h$, $\delta_t^z$, gradientes respecto a $W_{xh}$, $W_{hh}$, $W_{hy}$ -- la clase MIT solo lo presenta conceptualmente.
- **Analisis espectral del vanishing gradient**: cota con $\sigma_{\max}(W_{hh})^T$ y conexion explicita con Pascanu et al. (2013). Ver [profundizacion seccion 4](/clases/clase-11/profundizacion).
- **Demostracion formal de por que LSTM evita vanishing**: el argumento $\partial c_t / \partial c_{t-1} = f_t$ es intuitivo en MIT pero la Clase 11 muestra el calculo completo.
- **Comparacion criteriada LSTM vs GRU**: cuando elegir cada uno, complejidad de parametros, recomendaciones empiricas de la literatura.

El espiritu es **complementario**: ver primero el video MIT por su intuicion visual, luego trabajar la Clase 11 UC para el rigor formal.

---

## 4. Conceptos NO cubiertos por MIT que vale la pena conocer

La edicion 2020 del lecture es anterior a varios desarrollos importantes. Listamos los mas relevantes para un estudiante actual.

### 4.1 Transformers como sucesor natural

El lecture termina con attention como un mecanismo *complementario* a las RNNs. **El paper "Attention Is All You Need" (Vaswani et al., 2017)** demostro que se puede prescindir totalmente de la recurrencia y construir una arquitectura basada exclusivamente en self-attention. Hoy (2025), **los Transformers dominan virtualmente todo NLP, vision (ViT), audio (Whisper) y multimodal (CLIP, GPT-4)**. La conexion conceptual con la clase MIT es directa: el `score(s_{t-1}, h_i)` de Bahdanau es analogo al producto query-key del Transformer; el context vector ponderado es analogo al output de cada cabeza de atencion. Para profundizar, ver la [Clase 13 del curso UC](/clases/clase-13/teoria) que cubre esta transicion.

### 4.2 Bidirectional RNNs en mas detalle

MIT menciona **BiLSTM** brevemente (slide 83) en el contexto del encoder. Pero merece tratamiento independiente: una BiLSTM apila una RNN forward y una backward, concatenando sus estados. Es estandar para tareas donde el contexto futuro es accesible (etiquetado de secuencias, reconocimiento de voz offline, comprension lectora). No es aplicable a generacion autorregresiva ni a streaming.

### 4.3 Deep RNNs / Stacked RNNs

La idea de **apilar varias capas de RNN** (la salida de la capa $\ell$ alimenta a la capa $\ell+1$) tampoco se desarrolla en MIT pero es util en la practica: dos o tres capas suelen mejorar el rendimiento. La Clase 11 UC trata stacked RNNs en su seccion 7.

### 4.4 Alternativas modernas: S4, Mamba, RWKV

Desde 2022 han emergido familias de modelos que **combinan ventajas de RNNs y Transformers**: complejidad lineal en la longitud de la secuencia (como RNN), pero entrenamiento paralelo (como Transformer):

- **S4** (Gu et al., 2022): structured state space models.
- **Mamba** (Gu & Dao, 2023): selective state space models con paralelismo en GPU.
- **RWKV**: una recurrencia lineal con receptive field infinito.

Estos modelos son **investigacion activa** en 2024-2025 y compiten con Transformers en tareas de contexto largo.

### 4.5 Conexion con la Clase 13 sobre attention

La Clase 13 del curso UC retoma exactamente desde donde MIT termina: profundiza en seq2seq, en la atencion de Bahdanau (additive) y la de Luong (multiplicative), y conecta hacia self-attention y Transformers. Es el complemento natural si despues de ver el lecture MIT se quiere continuar.

---

## 5. Recursos adicionales

- **GitHub oficial del curso**: [github.com/aamini/introtodeeplearning](https://github.com/aamini/introtodeeplearning) -- contiene todos los labs en TensorFlow/Keras, slides en PDF y referencias por edicion.
- **Lab 1 (RNN Music Generation)**: implementa una RNN character-level que genera musica en formato ABC notation. Acompana el lecture 2 directamente. Util como punto de entrada practico antes de los notebooks del curso UC.
- **Sitio web del curso**: [introtodeeplearning.com](http://introtodeeplearning.com) -- mantiene siempre la edicion mas reciente. La edicion 2024 incluye lecciones nuevas sobre LLMs, foundation models y Transformers que no estan en la 2020.
- **Canal YouTube MIT 6.S191**: tiene playlists separadas por ano, lo que permite comparar la evolucion del material.
- **Para el estudiante UC**: tras ver el video, recomendamos seguir con [Profundizacion de Clase 11](/clases/clase-11/profundizacion) y [Teoria de Clase 13](/clases/clase-13/teoria) para cerrar el ciclo formal.
