---
title: "Children's Book Test (The Goldilocks Principle)"
weight: 120
math: true
---

{{< paper-card
    title="The Goldilocks Principle: Reading Children's Books with Explicit Memory Representations"
    authors="Felix Hill, Antoine Bordes, Sumit Chopra, Jason Weston"
    year="2016"
    venue="ICLR 2016 (arXiv 1511.02301)"
    pdf="/papers/childrens-book-test-hill-2016.pdf"
    arxiv="1511.02301" >}}
Introduce el **Children's Book Test (CBT)**, un benchmark cloze a gran escala construido sobre libros infantiles de Project Gutenberg, y lo usa como instrumento de diagnostico para una pregunta fina: que tipos de palabra requieren *memoria* del contexto amplio y cuales se predicen bien con un modelo de lenguaje local. La pieza de diseno clave es generar preguntas por **tipo de palabra omitida** -- Named Entities, Common Nouns, Verbs, Prepositions -- usando POS/NER de Stanford CoreNLP. El hallazgo, el "Goldilocks Principle": la memoria ayuda para entidades y sustantivos (contenido semantico) pero no para verbos y preposiciones (funcion sintactica), y existe un tamano *justo* de representacion de memoria, ni la palabra suelta ni la oracion entera, sino ventanas sub-oracionales. Las **Memory Networks** con window memory y self-supervision baten al estado del arte en CBT-NE y en CNN QA.
{{< /paper-card >}}

---

## El problema

A mediados de 2015, Hermann et al. ("Teaching Machines to Read and Comprehend", NIPS 2015) liberaron el dataset [CNN/Daily Mail](/papers/cnn-dailymail-hermann-2015), que popularizo el formato **cloze a gran escala** para *machine reading*: se anonimizan las entidades de un articulo de noticias y se pide al modelo recuperar la entidad faltante de un resumen en vinetas. Ese trabajo demostro que se podia construir, de forma automatica y barata, un corpus supervisado masivo de comprension lectora.

El CBT nace en ese mismo ecosistema pero con una motivacion mas quirurgica. Hermann et al. preguntaban "pueden las maquinas leer y comprender?"; Hill et al. preguntan **que mide realmente una tarea cloze, y de que depende que un modelo necesite memoria amplia o no**.

El insight de partida es linguistico-cognitivo. La evaluacion clasica de modelos de lenguaje se basa en *perplejidad promedio* sobre todas las palabras. Como la frecuencia de palabras sigue una distribucion de Zipf, la perplejidad pondera desproporcionadamente las palabras frecuentes (preposiciones, articulos, verbos auxiliares) que transmiten poca carga semantica, y subpondera las palabras de baja frecuencia (nombres propios, sustantivos de contenido) que cargan el grueso del significado. Un modelo puede tener excelente perplejidad y aun asi ser pobre prediciendo justo las palabras que importan para traduccion, dialogo o QA.

El CBT desacopla esto: en lugar de promediar sobre todo, evalua **accuracy separada por clase de palabra omitida**. Esto convierte el benchmark en un microscopio para estudiar el rol de la memoria.

---

## Idea central -- cloze por tipo de palabra

El CBT se construye sobre libros infantiles libres de Project Gutenberg. La eleccion no es casual: garantizan una **estructura narrativa clara**, que hace mas saliente el rol del contexto (los referentes -- quien es quien, que objeto se menciono -- se mantienen a lo largo del capitulo).

El mecanismo de construccion de cada pregunta $x$ es mecanico y reproducible:

1. Se enumeran **21 oraciones consecutivas** de un capitulo.
2. Las primeras **20 oraciones** forman el contexto $S$ (lista ordenada de oraciones).
3. De la **oracion 21** se elimina una palabra $a$; esa oracion con el hueco es la query $q = q_1, \dots, q_l$.
4. El modelo debe identificar $a$ entre **10 candidatos** $C$, donde $|C| = 10$, $a \in C$, y todo candidato $w \in C$ aparece en el contexto o la query ($w \in q \cup S$).

Formalmente, un par pregunta-respuesta es $(x, a)$ con $x = (q, S, C)$.

La pieza de diseno que hace al paper especial es **variar el tipo de la palabra omitida**. Usando el POS tagger y el NER de Stanford CoreNLP, se generan cuatro clases de preguntas segun el tipo de $a$:

- **Named Entities** (Elvis, France)
- **Common Nouns** (ball, table)
- **Verbs** (run, eat)
- **Prepositions** (on, at)

Los nueve candidatos incorrectos se eligen al azar entre palabras del contexto **del mismo tipo** que la respuesta. Esto es importante: un modelo no puede resolver la pregunta por POS trivial (todos los candidatos son del mismo POS); debe distinguir *cual* nombre, *cual* verbo, dentro de la clase.

A diferencia de CNN QA, **el CBT no anonimiza las entidades**. Es una decision deliberada: quieren incentivar modelos que combinen conocimiento de fondo con el contexto inmediato y amplio. CNN QA, al anonimizar, fuerza al modelo a depender solo del articulo. Son filosofias opuestas y por eso complementarias.

---

## El Goldilocks Principle

El nombre viene del cuento ingles: Ricitos de Oro prueba tres tazones de avena y elige el del medio, el que no esta "ni muy caliente ni muy frio". El principio que el paper identifica es doble:

**(a) Que palabras necesitan memoria.** Los humanos predicen *todos* los tipos de palabra con accuracy similar, pero **dependen del contexto amplio solo para entidades y sustantivos**; para verbos y preposiciones de alta frecuencia el contexto amplio es irrelevante. Los modelos de lenguaje neuronales (RNN-LSTM) hacen lo contrario de lo deseable: son excelentes con preposiciones y verbos (incluso superan a humanos en preposiciones), pero quedan muy atras en nombres y entidades, porque sus predicciones se basan casi exclusivamente en contexto local. La carga semantica vive justamente en las palabras donde los LSTM fallan.

**(b) El "sweet spot" del tamano de memoria.** Aqui esta el corazon empirico. La forma de representar el contexto amplio en memoria es critica, y existe un **tamano optimo de representacion entre la palabra individual y la oracion completa**. No demasiado grande (oracion entera, que diluye la senal), no demasiado pequeno (palabra suelta, que pierde contexto local), sino justo: **ventanas sub-oracionales** centradas en las palabras candidatas. Y ese tamano optimo *depende de la clase de palabra a predecir*.

Lo generalizan mas alla del CBT: la observacion de que las representaciones mas informativas para redes neuronales son *chunks* sub-oracionales es consistente con la atencion local en traduccion neuronal (Luong et al. 2015 restringen la atencion a ventanas pequenas de la oracion fuente) y explica por que las RNN bidireccionales de los reading models funcionan: el estado oculto combinado de una BiRNN en cada palabra se enfoca naturalmente en un chunk tipo ventana del texto circundante, igual que una window memory.

---

## Construccion del dataset

| Estadistica | Training | Validation | Test |
|-------------|---------:|-----------:|-----:|
| Numero de libros | 98 | 5 | 5 |
| Preguntas (contexto + query) | 669,343 | 8,000 | 10,000 |
| Promedio de palabras en contextos | 465 | 435 | 445 |
| Promedio de palabras en queries | 31 | 27 | 29 |
| Candidatos distintos | 37,242 | 5,485 | 7,108 |

El vocabulario total reportado es 53,628. El texto de entrenamiento equivale a aproximadamente **5.5M de palabras** (los LSTM se entrenan sobre eso). Cada pregunta tiene 10 candidatos, contra los 5 de su antecesor mas cercano.

Recursos relacionados que ayudan a situar el CBT:

- **MSR Sentence Completion Challenge** (Zweig & Burges, 2011): tambien sobre Gutenberg, pero cada ejemplo es *una sola oracion* sin contexto amplio, 1,040 preguntas de test, 5 candidatos. El CBT es mas grande (10,000 vs 1,040), tiene mas candidatos (10 vs 5), separa por tipo de POS y trae sets de train/val masivos que igualan la forma del test.
- **CNN/Daily Mail QA** (Hermann et al., 2015): foco en *parafrasis* de resumenes de noticias, entidades anonimizadas; el CBT pide *inferencias y predicciones* desde el contexto narrativo, sin anonimizar.
- **MCTest** (Richardson et al., 2013): historias infantiles con preguntas de opcion multiple, pero su training set tiene solo 300 ejemplos, insuficiente para entrenar modelos estadisticos.

---

## Modelos -- Memory Networks

El paper aplica un abanico amplio de arquitecturas: baselines sin aprendizaje (maxima frecuencia, sliding window, word distance), n-gram LM con Kneser-Ney y cache, modelos de embeddings supervisados, LSTM y Contextual LSTM. El centro tecnico, sin embargo, son las **Memory Networks** (Weston et al. 2015).

Tres formatos de **codificacion de memorias** desde el contexto $S$, usando una feature-map $\phi(s)$ que mapea secuencias de palabras a representaciones:

- **Lexical memory**: cada palabra ocupa un slot. Para codificar orden se agregan *time features* (embeddings del indice de cada memoria).
- **Window memory**: cada slot es una ventana de texto centrada en una mencion de un candidato $c$ en $S$, ventanas $\{w_{i-(b-1)/2} \dots w_i \dots w_{i+(b-1)/2}\}$ con $w_i \in C$. El ancho $b$ se tunea en validacion.
- **Sentential memory**: cada slot es una oracion completa de $S$, exactamente 20 memorias por pregunta, con Positional Encoding.

**End-to-End Memory Networks (MemN2N).** La arquitectura de Sukhbaatar et al. (2015) permite entrenamiento directo por backpropagation. Query y memorias se embeben; el match pasa por un softmax que da la distribucion de atencion $\{\alpha_i\}$ y se devuelve la supporting memory:

$$m_{o1} = \sum_{i=1}^{n} \alpha_i \mathbf{m}_i, \qquad \alpha_i = \frac{e^{\mathbf{c}_i^\top \mathbf{q}}}{\sum_j e^{\mathbf{c}_j^\top \mathbf{q}}}$$

El proceso puede repetirse $K$ veces (multiples *hops*) con $\mathbf{q}_k = H\mathbf{q}_{k-1} + m_{o,k-1}$. La distribucion de respuesta final $\hat{a} = \mathrm{softmax}(U\mathbf{q}_{K+1})$ se restringe a candidatos: $\hat{a} = \arg\max_{w \in C} \hat{a}(w)$.

**Self-supervision para window memories.** Observacion clave: los multiples hops solo ayudaban en lexical memory. Para window memory probaron una MemNN de **un solo hop** con una senal de aprendizaje mas fuerte. La supervision de memoria (a que memoria atender) no esta dada, pero se infiere: como en entrenamiento se conoce la respuesta correcta, se hipotetiza que la supporting memory esta entre las window memories cuyo candidato es la respuesta. Se entrena con seleccion *dura*:

$$m_{o1} = \arg\max_{i=1,\dots,n} \mathbf{c}_i^\top \mathbf{q}$$

Conceptualmente, la self-supervision es una forma de lograr **hard attention** sobre memorias. La hard attention dio grandes mejoras en image captioning (Xu et al. 2015), pero alli se uso REINFORCE; aqui la heuristica permite **backpropagation directa** sin gradiente de politica, sin ninguna etiqueta nueva mas alla de los datos de entrenamiento. En test, en vez de seleccion dura, se scorea cada candidato con la *suma* de los $\alpha_i$ de todas las ventanas en que aparece.

Notar el contraste de hiperparametros: lexical memory necesita $K=7$ hops, mientras window basta con $K=1$.

---

## Resultados

### CBT test set (accuracy por tipo de palabra)

| Metodo | Named Entities | Common Nouns | Verbs | Prepositions |
|--------|---:|---:|---:|---:|
| Humanos (query)* | 0.520 | 0.644 | 0.716 | 0.676 |
| Humanos (context+query)* | 0.816 | 0.816 | 0.828 | 0.708 |
| Max frequency (context) | 0.335 | 0.281 | 0.285 | 0.275 |
| Word distance model | 0.398 | 0.364 | 0.380 | 0.237 |
| Kneser-Ney LM | 0.390 | 0.544 | 0.778 | 0.768 |
| Kneser-Ney LM + cache | 0.439 | 0.577 | 0.772 | 0.679 |
| Embedding model (query) | 0.351 | 0.400 | 0.614 | 0.535 |
| Embedding model (window+position) | 0.402 | 0.506 | 0.736 | 0.670 |
| LSTMs (query) | 0.408 | 0.541 | 0.813 | 0.802 |
| LSTMs (context+query) | 0.418 | 0.560 | 0.818 | 0.791 |
| Contextual LSTMs (window context) | 0.436 | 0.582 | 0.805 | 0.806 |
| MemNN (lexical memory) | 0.431 | 0.562 | 0.798 | 0.764 |
| MemNN (window memory) | 0.493 | 0.554 | 0.692 | 0.674 |
| MemNN (sentential memory + PE) | 0.318 | 0.305 | 0.502 | 0.326 |
| **MemNN (window memory + self-sup.)** | **0.666** | **0.630** | 0.690 | 0.703 |

\* Humanos sobre 10% del test set.

Lectura del Goldilocks Principle en la tabla:

1. **Verbos y preposiciones: el LM basta.** Los LSTM dominan (0.813 / 0.802 con solo query). Notablemente, **superan a humanos en preposiciones** (0.802 vs 0.708 humano context+query). La memoria del contexto amplio *no aporta*: LSTM (query) es practicamente igual a LSTM (context+query) en verbos (0.813 vs 0.818) y preposiciones (0.802 vs 0.791). La explicacion de los autores: los modelos estan mejor afinados a la distribucion de verbos en libros infantiles, mientras los humanos se dejan influir por su conocimiento de todos los estilos de lenguaje.

2. **Entidades y sustantivos: la memoria ayuda.** Aqui los LSTM se estancan (~0.41 en entidades, ~0.55 en sustantivos) y **leer el contexto no los mejora** (0.408 query vs 0.418 context+query en entidades), sintoma del problema clasico de dependencias de largo plazo en RNN. La MemNN **window+self-sup** salta a **0.666 en entidades** y **0.630 en sustantivos**, muy por encima de cualquier LM. El Embedding model (query), que es la MemNN sin memoria contextual, queda en 0.351 / 0.400; la diferencia es justamente *el acceso a memoria*.

3. **Goldilocks del tamano de chunk.** Sentential memory + PE es desastroso (0.318 entidades, 0.305 sustantivos): oracion entera = muy grande. Lexical memory funciona para preposiciones / verbos (0.764 / 0.798) pero flojo en entidades / sustantivos: palabra suelta = muy pequena para semantica. Window memory centrada en candidatos es lo "justo" para nombres y entidades, y la self-supervision la lleva al tope.

4. **Brecha con humanos.** Los humanos con contexto llegan a 0.816 en entidades; el mejor modelo, 0.666. El benchmark deja margen amplio justo en las clases semanticas, que es su proposito de diseno.

### CNN QA (generalizacion)

Aplicada al [CNN/Daily Mail](/papers/cnn-dailymail-hermann-2015) de Hermann et al. (93k articulos), la mejor MemNN confirma que el principio generaliza:

| Metodo | Validation | Test |
|--------|---:|---:|
| Contextual LSTMs ("Attentive Reader")* | 0.616 | 0.630 |
| Contextual LSTMs ("Impatient Reader")* | 0.618 | 0.638 |
| MemNN (window memory) | 0.580 | 0.606 |
| MemNN (window memory + self-sup.) | 0.634 | 0.668 |
| MemNN (window + self-sup. + ensemble + excluding co-occurrences) | **0.662** | **0.694** |

\* Tomados de Hermann et al. (2015).

La window memory sin self-supervision (0.606 test) iguala a los reading models con ensemble; al **agregar self-supervision la MemNN supera el estado del arte** (0.668), y con ensemble y la heuristica de excluir co-ocurrencias sube a **0.694**. Lo que distingue a la mejor MemNN no es *como representa* el contexto sino **como accede/recupera**: la hard attention via self-supervision hace mas tratable el aprendizaje del acceso a memoria.

Un par de ablations relevantes: anonimizar los candidatos en el CBT (como hace CNN QA) tiene impacto *bajo* en entidades (0.666 → 0.581) pero *grande* en tareas sintacticas (verbos 0.690 → 0.474, preposiciones 0.703 → 0.522), porque la identidad lexica de verbos / preposiciones es justo lo que el modelo usa. Quitar el *soft memory weighting* en test (volver a max duro) baja el test de CNN QA de 0.668 a 0.620: es el componente mas importante.

---

## Limitaciones

- **Dominio restringido.** Libros infantiles de Gutenberg (siglos XIX-XX, dominio publico): vocabulario, sintaxis y estilo particulares. Brillar aqui no garantiza transferencia a texto tecnico, conversacional o contemporaneo.
- **Formato cloze artificial.** Predecir la palabra omitida de la oracion 21 dadas 20 oraciones es un proxy, no comprension natural. La generacion automatica via POS/NER introduce sesgos del tagger y casos ambiguos (varias preposiciones "correctas", donde los modelos baten a humanos no por comprender mejor sino porque los anotadores prefieren la opcion menos frecuente).
- **Candidatos limitados y del mismo POS.** 10 candidatos del mismo tipo acotan el espacio de respuesta; el modelo nunca enfrenta la tarea abierta de generar la palabra.
- **No anonimizacion = fuga de conocimiento de fondo.** Decision deliberada, pero parte del accuracy puede venir de regularidades lexicas o de frecuencia mas que de comprension genuina del contexto.
- **Solo evidencia, no explica.** El benchmark no resuelve *por que* las RNN no retienen contexto largo; solo lo evidencia.

---

## Por que importa hoy

- **Refino que mide un cloze task.** La contribucion conceptual mas duradera: separar prediccion de *function words* (sintaxis, frecuencia alta) de *content words* (semantica, frecuencia baja) y mostrar que el rol de la memoria es especifico de la clase. Un accuracy global esconde dos regimenes distintos.
- **Valido las representaciones sub-oracionales.** El "sweet spot" sub-oracional dio sustento empirico a por que las RNN bidireccionales con atencion y la atencion local en NMT funcionan: todas convergen, por vias distintas, a representar chunks pequenos de texto. Esto influyo en el diseno de *attention / memory readers* posteriores.
- **Self-supervision como hard attention sin REINFORCE.** Inferir la supervision de memoria desde la etiqueta y entrenar hard attention con backprop directa fue una idea practica reutilizada.
- **Benchmark estandar junto a CNN/DM.** El CBT se volvio uno de los benchmarks de referencia para machine reading 2016-2017. Trabajos posteriores (Attention Sum Reader, Gated-Attention Reader, EpiReader, AoA Reader) reportaron rutinariamente en CBT-NE y CBT-CN, las dos clases semanticas donde esta el interes.
- **Precedente de SQuAD.** El CBT pertenece a la generacion cloze que *precede* a [SQuAD](/papers/squad-rajpurkar-2016) (Rajpurkar et al. 2016), que cambio el paradigma a *span extraction* sobre preguntas reales redactadas por humanos, superando varias limitaciones del cloze automatico.

---

## Conexion con la clase 24

La [Clase 24](/clases/clase-24) lista, entre los datasets de Question Answering / Reading Comprehension previos a la era de los grandes modelos, **"bAbI, Children's Book Test"** dentro de los datasets pre-2015/2016. Esto ubica al CBT en su lugar historico:

- **Familia cloze que antecede a SQuAD.** La clase traza la evolucion de los datasets de comprension lectora: de los cloze automaticos (CNN/Daily Mail, CBT, [bAbI](/papers/babi-weston-2015)) hacia datasets con preguntas redactadas por humanos y respuestas extractivas (SQuAD) y luego generativas. El CBT es un eslabon clave de la primera etapa: cloze, gran escala, generado automaticamente, multiple-choice.
- **Conexion directa con CNN/Daily Mail.** Ambos comparten el formato cloze a escala y el linaje de FAIR / DeepMind. El propio paper de CBT valida sus conclusiones reproduciendo y superando los *Attentive / Impatient Readers* de Hermann sobre CNN QA.
- **Motivacion de los attention/memory readers.** El CBT es el banco de pruebas donde se disecciono *por que* y *cuando* la atencion / memoria ayuda: para el contenido semantico (entidades, sustantivos), no para la funcion sintactica. Esa distincion justifica arquitecturalmente los memory / attention readers que la clase presenta.
- **bAbI como hermano.** El CBT se distribuye junto a [bAbI](/papers/babi-weston-2015) (`fb.ai/babi`), de los mismos autores, y comparte la maquinaria de Memory Networks. La clase agrupa ambos como la contribucion de FAIR al estudio de memoria y razonamiento en NLP pre-transformers.

En sintesis: **el CBT es el experimento que mostro que "memoria" no es un ingrediente uniforme; su utilidad depende de que se quiere predecir**. Ese matiz sobrevive a la transicion hacia SQuAD y, mas tarde, hacia los transformers, donde la atencion se vuelve el mecanismo universal de acceso a contexto que aqui se estudiaba todavia de forma explicita y modular.

---

## Notas y enlaces

- **arXiv:** `https://arxiv.org/abs/1511.02301` (v4, 1 Apr 2016).
- **Dataset:** `http://fb.ai/babi/` (CBT distribuido con bAbI). Textos: Project Gutenberg.
- **Implementacion:** Torch; codigo de MemNN en `https://github.com/facebook/MemNN`.
- **Toolkits usados:** Stanford CoreNLP (POS + NER), KenLM (n-gram Kneser-Ney).
- **Nomenclatura:** el nombre alude al cuento "Goldilocks and the Three Bears" -- el principio de "ni muy grande, ni muy pequeno, sino justo" aplicado al tamano de la representacion de memoria.

Ver fundamentos: [Question Answering](/fundamentos/question-answering) - [Machine Reading Comprehension](/fundamentos/machine-reading-comprehension).

Ver papers: [CNN/Daily Mail (Hermann 2015)](/papers/cnn-dailymail-hermann-2015) - [bAbI (Weston 2015)](/papers/babi-weston-2015) - [SQuAD (Rajpurkar 2016)](/papers/squad-rajpurkar-2016).

Ver clase: [Clase 24 -- Question Answering y Reading Comprehension](/clases/clase-24).
