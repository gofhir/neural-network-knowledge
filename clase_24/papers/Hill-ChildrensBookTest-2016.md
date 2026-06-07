# The Goldilocks Principle: Reading Children's Books with Explicit Memory Representations — Hill et al. (2016)

## Metadata

| Campo | Valor |
|-------|-------|
| Título | The Goldilocks Principle: Reading Children's Books with Explicit Memory Representations |
| Autores | Felix Hill, Antoine Bordes, Sumit Chopra, Jason Weston |
| Afiliación | Facebook AI Research (FAIR), 770 Broadway, New York, USA. Felix Hill completó el trabajo en University of Cambridge, Computer Laboratory |
| Venue | ICLR 2016 (Published as a conference paper) |
| arXiv | 1511.02301v4 [cs.CL], 1 Apr 2016 (versión inicial nov. 2015) |
| Dataset | Children's Book Test (CBT), distribuido junto con bAbI en `http://fb.ai/babi/` |
| Tarea | Cloze multiple-choice; predicción de palabra omitida por tipo (Named Entities, Common Nouns, Verbs, Prepositions) |

El paper introduce un benchmark, el **Children's Book Test (CBT)**, y lo usa como instrumento de diagnóstico para responder una pregunta fina: ¿qué tipo de palabras requieren *memoria* del contexto amplio y cuáles se predicen bien con un modelo de lenguaje local? La respuesta es el "Goldilocks Principle".

---

## 1. Contexto: el momento cloze de 2015-2016

A mediados de 2015, Hermann et al. ("Teaching Machines to Read and Comprehend", NIPS 2015) habían liberado el dataset CNN/Daily Mail (CNN QA), que popularizó el formato **cloze a gran escala** para machine reading: se anonimizan las entidades de un artículo de noticias y se pide al modelo recuperar la entidad faltante de un resumen en viñetas. Ese trabajo demostró que se podía construir, de forma automática y barata, un corpus supervisado masivo para comprensión lectora, y propuso los *Attentive Reader* e *Impatient Reader* basados en RNN bidireccionales con atención.

El CBT nace en ese mismo ecosistema, pero con una motivación distinta y complementaria. Hermann et al. preguntaban "¿pueden las máquinas leer y comprender?"; Hill et al. preguntan algo más quirúrgico: **¿qué mide realmente una tarea cloze, y de qué depende que un modelo necesite memoria amplia o no?** El insight de partida es lingüístico-cognitivo. La evaluación clásica de modelos de lenguaje se basa en *perplejidad promedio* sobre todas las palabras del texto. Como la frecuencia de palabras sigue una distribución de Zipf, la perplejidad pondera desproporcionadamente las palabras frecuentes (preposiciones, artículos, verbos auxiliares) que transmiten poca carga semántica, y subpondera las palabras de baja frecuencia (nombres propios, sustantivos de contenido) que cargan el grueso del significado (Baayen & Lieber, 1996). Un modelo puede tener excelente perplejidad y aun así ser pobre prediciendo justo las palabras que importan para traducción, diálogo o QA.

El CBT desacopla esto: en lugar de promediar sobre todo, evalúa **accuracy separada por clase de palabra omitida**. Esto convierte el benchmark en un microscopio para estudiar el rol de la memoria.

---

## 2. Idea central: cloze sobre libros infantiles del Project Gutenberg

El CBT se construye a partir de libros infantiles libres de Project Gutenberg. La elección de libros infantiles no es casual: garantizan una **estructura narrativa clara**, lo que hace más saliente el rol del contexto (los referentes — quién es quién, qué objeto se mencionó — se mantienen a lo largo del capítulo).

El mecanismo de construcción de cada "pregunta" $x$ es mecánico y reproducible:

1. Se enumeran **21 oraciones consecutivas** de un capítulo.
2. Las primeras **20 oraciones** forman el contexto $S$ (una lista ordenada de oraciones).
3. De la **oración 21** se elimina una palabra $a$; esa oración con el hueco se convierte en la query $q = q_1, \dots, q_l$.
4. El modelo debe identificar $a$ entre **10 candidatos** $C$, donde $|C| = 10$, $a \in C$, y todo candidato $w \in C$ aparece en el contexto o la query ($w \in q \cup S$).

Formalmente, un par pregunta-respuesta es $(x, a)$ con $x = (q, S, C)$.

La pieza de diseño que hace al paper especial es **variar el tipo de la palabra omitida**. Usando el POS tagger y el NER del Stanford CoreNLP, se generan cuatro clases de preguntas según el tipo de $a$:

- **Named Entities** (Elvis, France)
- **Common Nouns** (ball, table)
- **Verbs** (run, eat)
- **Prepositions** (on, at)

Los nueve candidatos incorrectos se eligen al azar entre palabras del contexto **del mismo tipo** que la respuesta. Esto es importante: significa que un modelo no puede resolver la pregunta por POS trivial (todos los candidatos son del mismo POS); debe distinguir *cuál* nombre, *cuál* verbo, dentro de la clase.

A diferencia de CNN QA, **el CBT no anonimiza las entidades**. Los autores lo dejan deliberadamente: quieren incentivar modelos que combinen conocimiento de fondo (background knowledge) con información del contexto inmediato y amplio. CNN QA, al anonimizar, fuerza al modelo a depender solo del artículo. Son filosofías opuestas y por eso complementarias.

---

## 3. El Goldilocks Principle

El nombre viene del cuento inglés (Hassall, 1904): Ricitos de Oro prueba tres tazones de avena y elige el que no está "ni muy caliente ni muy frío", el del medio. El principio que el paper identifica es doble:

**(a) Qué palabras necesitan memoria.** Humanos predicen *todos* los tipos de palabra con accuracy similar, pero **dependen del contexto amplio solo para entidades y sustantivos**; para verbos y preposiciones de alta frecuencia el contexto amplio es irrelevante. Los modelos de lenguaje neuronales (RNN-LSTM) hacen lo contrario de lo deseable: son excelentes con preposiciones y verbos (incluso superan a humanos en preposiciones), pero quedan muy atrás en nombres y entidades, porque sus predicciones se basan casi exclusivamente en contexto local. La carga semántica vive justamente en las palabras donde los LSTM fallan.

**(b) El "sweet spot" del tamaño de memoria.** Aquí está el corazón empírico. La forma en que se representa el contexto amplio en memoria es crítica, y existe un **tamaño óptimo de la representación de memoria entre la palabra individual y la oración completa**. No demasiado grande (oración entera → diluye la señal), no demasiado pequeño (palabra suelta → pierde contexto local), sino justo: **ventanas sub-oracionales** centradas en las palabras candidatas. Y ese tamaño óptimo *depende de la clase de palabra a predecir*.

Este es el hallazgo transversal del paper, y lo generalizan más allá del CBT: la observación de que las representaciones más informativas para redes neuronales corresponden a *chunks* sub-oracionales es consistente con la traducción neuronal (Luong et al. 2015 restringen la atención a ventanas pequeñas de la oración fuente) y explica por qué las RNN bidireccionales de los reading models funcionan: el estado oculto combinado de una BiRNN en cada palabra se enfoca naturalmente en un chunk tipo ventana del texto circundante, igual que una window memory.

---

## 4. Construcción del dataset: estadísticas

| Estadística | Training | Validation | Test |
|-------------|---------:|-----------:|-----:|
| Número de libros | 98 | 5 | 5 |
| Número de preguntas (contexto + query) | 669,343 | 8,000 | 10,000 |
| Promedio de palabras en contextos | 465 | 435 | 445 |
| Promedio de palabras en queries | 31 | 27 | 29 |
| Candidatos distintos | 37,242 | 5,485 | 7,108 |
| Tamaño de vocabulario | — | 53,628 | — |

(El desglose por clase de pregunta se distribuye con los archivos del dataset; el vocabulario total reportado es 53,628.)

El texto de entrenamiento equivale a aproximadamente **5.5M de palabras** (los LSTM se entrenan sobre eso). Cada pregunta tiene 10 candidatos, contra los 5 de su antecesor más cercano.

**Recursos relacionados** (sección 2.1), útiles para situar el CBT:

- **MSR Sentence Completion Challenge** (Zweig & Burges, 2011): también sobre Gutenberg, pero cada ejemplo es *una sola oración* sin contexto amplio, 1,040 preguntas de test, 5 candidatos. El CBT es más grande (10,000 vs 1,040), tiene más candidatos (10 vs 5), separa por tipo de POS, y trae sets de train/val masivos que igualan la forma del test.
- **CNN/Daily Mail QA** (Hermann et al., 2015): foco en *paráfrasis* de resúmenes de noticias, entidades anonimizadas; el CBT en cambio pide *inferencias y predicciones* desde el contexto narrativo, sin anonimizar.
- **MCTest** (Richardson et al., 2013): historias infantiles escritas por anotadores con 4 preguntas de opción múltiple cada una; pero su training set tiene solo 300 ejemplos, insuficiente para entrenar modelos estadísticos.

---

## 5. Modelos

El paper aplica un abanico amplio de arquitecturas. Las divido en no-aprendizaje, LM clásicos/neuronales, y las Memory Networks (el centro técnico).

### 5.1 Baselines sin aprendizaje

- **Maximum frequency (corpus)**: elige el candidato más frecuente en todo el corpus de entrenamiento.
- **Maximum frequency (context)**: el candidato más frecuente dentro del contexto de la pregunta.
- **Sliding window** (de MCTest, Richardson et al. 2013): desliza ventanas de "query + candidato" sobre el contexto; el score en cada posición es el solape de palabras ponderado estilo TF-IDF (para enfatizar palabras poco frecuentes). Se elige el candidato cuya ventana logra el solape máximo en cualquier posición.
- **Word distance model** (de Hermann et al. 2015): para cada instancia de un candidato $w_i$ en el contexto, se "superpone" la query sobre el contexto alineando el hueco con $w_i$, definiendo una subsecuencia $s$. Por cada palabra $q_i$ de la query se incurre una penalización de alineamiento $P = \min(\min_{j=1\dots|s|}\{|i-j| : s_j = q_i\}, m)$. Se predice el candidato con menor penalización total. Se tunea $m = 5$ en validación.

### 5.2 Modelos de lenguaje n-gram

n-gram LM con **KenLM** (Heafield et al. 2013), suavizado **Kneser-Ney**, ventana de 5 (mejor en validación). Variante con **cache** (Kuhn & De Mori, 1990): se interpolan linealmente las probabilidades del n-gram con probabilidades unigrama calculadas sobre el contexto. El cache es justamente lo que da algo de memoria del contexto al n-gram, y se nota en entidades.

### 5.3 Modelos de embeddings supervisados

Inspirados en Weston et al. (2010). Se aprenden matrices de input y output $A, B \in \mathbb{R}^{p \times d}$ ($p$ = dimensión de embedding, $d$ = tamaño de vocabulario). Para una query $q$ y un candidato $w$, el score es:

$$S(q, w) = \phi(q)\, A^\top B\, \phi(w)$$

con $\phi$ la función de features one-hot/bag-of-words. Los autores los describen explícitamente como **"Memory Networks lobotomizadas con cero hops"**: se elimina por completo el componente de atención sobre la memoria. Sirven para medir cuánto del CBT se resuelve con buenas representaciones densas, sin memoria. Variantes según qué se codifica como input: contexto+query, solo query, una ventana de máximo $b$ palabras alrededor del hueco, y window+position (matriz distinta por posición de la ventana). Se tunea $d=5$ (window).

### 5.4 Modelos de lenguaje recurrentes (LSTM)

RNN-LSTM probabilísticos entrenados sobre las historias (5.5M palabras) con minibatch SGD maximizando la log-verosimilitud de la siguiente palabra. Mejor configuración: capa oculta y embeddings de dimensión 512. Dos variantes: "burn-in" leyendo todo el contexto+query, o leyendo solo la query (sin acceso al contexto). A diferencia del LM canónico, todos los modelos ven las palabras de la query *después* del hueco: si $k$ es la posición del hueco, se rankea el candidato $c$ por $p(q_1 \dots q_{k-1}, c, q_{k+1} \dots q_l)$ y no solo por $p(q_1 \dots q_{k-1}, c)$.

**Contextual LSTM (CLSTM)**: inspirado en Mikolov & Zweig (2012) y en la representación convolucional del contexto de Rush et al. (2015, summarization). Aprende una *atención convolucional sobre ventanas del contexto* con el objetivo de predecir todas las palabras de la query. Ventana $w=5$. Se entrena sobre el texto corrido (no sobre el formato estructurado query/contexto de las MemNN), lo que resultó más efectivo.

### 5.5 Memory Networks (el centro técnico)

Las Memory Networks (Weston et al. 2015b) son la clase de modelo que motiva el paper. Tres formatos de **codificación de memorias** desde el contexto $S$, usando una feature-map $\phi(s)$ que mapea secuencias de palabras a representaciones one-hot en $[0,1]^d$:

- **Lexical memory**: cada palabra ocupa un slot (cada $s$ es una palabra, $\phi(s)$ tiene una sola feature no nula). Para codificar orden se agregan *time features* como embeddings del índice de cada memoria (Sukhbaatar et al. 2015). En este formato las memorias se forman de las $n$ palabras *previas* al hueco, vengan del contexto o de la query, y el embedding de la query se fija al vector constante $0.1$.
- **Window memory**: cada $s$ es una ventana de texto centrada en una mención de un candidato $c$ en $S$. Los slots se llenan con ventanas $\{w_{i-(b-1)/2} \dots w_i \dots w_{i+(b-1)/2}\}$ donde $w_i \in C$. El número de ventanas suele exceder $|C|$ porque un candidato puede aparecer varias veces. $b$ se tunea en validación. Mejor codificación: un diccionario por posición de la ventana (mejor que bag-of-words).
- **Sentential memory**: cada $s$ es una oración completa de $S$ → exactamente 20 memorias por pregunta. Usa Positional Encoding (PE) de Sukhbaatar et al. (2015).

Para sentential y window, el orden importa menos, así que en vez de embedding completo del índice temporal se usa un *escalar* de posición (de 1 al número de memorias) con un parámetro que escala su importancia (tuneado en validación). Las time features solo dieron un boost marginal (Apéndice C).

**End-to-End Memory Networks (MemN2N).** La arquitectura de Sukhbaatar et al. (2015) permite entrenamiento directo por backpropagation. Primero se recuperan las "supporting memories". Query y memorias se embeben con $A \in \mathbb{R}^{p \times d}$, dando $\mathbf{q} = A\phi(q)$ y $\{\mathbf{c}_i = A\phi(s_i)\}$. El match entre query y cada memoria pasa por un softmax que da la distribución de atención $\{\alpha_i\}$, y se devuelve la primera supporting memory:

$$m_{o1} = \sum_{i=1\dots n} \alpha_i \mathbf{m}_i, \qquad \alpha_i = \frac{e^{\mathbf{c}_i^\top \mathbf{q}}}{\sum_j e^{\mathbf{c}_j^\top \mathbf{q}}}, \quad i = 1,\dots,n \tag{1}$$

donde $\{\mathbf{m}_i\}$ se obtienen igual que los $\mathbf{c}_i$ pero con otra matriz $B \in \mathbb{R}^{p \times d}$.

Las MemNN pueden hacer **varios hops**: el proceso se repite $K$ veces usando recursivamente $\mathbf{q}_k = H\mathbf{q}_{k-1} + m_{o,k-1}$, con $H \in \mathbb{R}^{p \times p}$ una proyección lineal entre hops y compartiendo $A, B$ entre capas. (En lexical memory se aplica ReLU a la mitad de las unidades de cada capa, siguiendo Sukhbaatar et al.)

Segunda etapa: distribución de respuesta $\hat{a} = \mathrm{softmax}(U\mathbf{q}_{K+1})$, con $U \in \mathbb{R}^{d \times p}$, sobre todo el vocabulario. La predicción se restringe a candidatos: $\hat{a} = \arg\max_{w \in C} \hat{a}(w)$. Se entrena con cross-entropy estándar contra la etiqueta verdadera $a$ frente a *todas* las palabras del diccionario (los candidatos no entran en la loss de entrenamiento), con SGD.

**Self-supervision para window memories (sección 3.3).** Observación clave: hacer múltiples hops solo ayudaba en lexical memory. Para window memory probaron una MemNN de **un solo hop** con una señal de aprendizaje más fuerte. La supervisión de memoria (saber a qué memoria atender) no está dada, pero se infiere: como en entrenamiento se conoce la respuesta correcta, se hipotetiza que la supporting memory correcta está entre las window memories cuyo candidato es la respuesta correcta. Si hay varias, se elige $\tilde{m}$, la que el propio modelo ya scorea más alto en el espacio de $A$. Se entrena con SGD para forzar que $\tilde{m}$ reciba mayor score que cualquier otra memoria de cualquier otro candidato, usando selección *dura*:

$$m_{o1} = \arg\max_{i=1,\dots,n} \mathbf{c}_i^\top \mathbf{q} \tag{2}$$

Si $m_{o1} \neq \tilde{m}$, se actualiza el modelo. En test, en vez de selección dura, se scorea cada candidato con la *suma* de los $\alpha_i$ (softmax) de todas las ventanas en que aparece, relajando el efecto del max (mejor que selección dura, Apéndice C).

Conceptualmente, la self-supervision es una forma de lograr **hard attention** sobre memorias (vs la soft attention de la sección 3.2). La hard attention dio grandes mejoras en image captioning (Xu et al. 2015), pero ellos usaron REINFORCE para entrenar a través del max; la heurística de self-supervision aquí permite **backpropagation directa**, sin gradiente de política. No usa ninguna etiqueta nueva más allá de los datos de entrenamiento.

**Hiperparámetros óptimos en CBT** (Apéndice A): MemNN lexical $n=200, \lambda=0.01, p=200, K=7$; window $n=\text{all}, b=5, \lambda=0.005, p=100, K=1$; sentential+PE $n=\text{all}, \lambda=0.001, p=100, K=1$; window+self-sup $n=\text{all}, b=5, \lambda=0.01, p=300$. Notar el contraste: lexical necesita $K=7$ hops, window basta con $K=1$. Todo implementado en Torch.

---

## 6. Resultados

### 6.1 CBT test set (accuracy por tipo de palabra)

| Método | Named Entities | Common Nouns | Verbs | Prepositions |
|--------|---:|---:|---:|---:|
| Humanos (query)* | 0.520 | 0.644 | 0.716 | 0.676 |
| Humanos (context+query)* | 0.816 | 0.816 | 0.828 | 0.708 |
| Max frequency (corpus) | 0.120 | 0.158 | 0.373 | 0.315 |
| Max frequency (context) | 0.335 | 0.281 | 0.285 | 0.275 |
| Sliding window | 0.168 | 0.196 | 0.182 | 0.101 |
| Word distance model | 0.398 | 0.364 | 0.380 | 0.237 |
| Kneser-Ney LM | 0.390 | 0.544 | 0.778 | 0.768 |
| Kneser-Ney LM + cache | 0.439 | 0.577 | 0.772 | 0.679 |
| Embedding model (context+query) | 0.253 | 0.259 | 0.421 | 0.315 |
| Embedding model (query) | 0.351 | 0.400 | 0.614 | 0.535 |
| Embedding model (window) | 0.362 | 0.415 | 0.637 | 0.589 |
| Embedding model (window+position) | 0.402 | 0.506 | 0.736 | 0.670 |
| LSTMs (query) | 0.408 | 0.541 | 0.813 | 0.802 |
| LSTMs (context+query) | 0.418 | 0.560 | 0.818 | 0.791 |
| Contextual LSTMs (window context) | 0.436 | 0.582 | 0.805 | 0.806 |
| MemNN (lexical memory) | 0.431 | 0.562 | **0.798** | **0.764** |
| MemNN (window memory) | 0.493 | 0.554 | 0.692 | 0.674 |
| MemNN (sentential memory + PE) | 0.318 | 0.305 | 0.502 | 0.326 |
| **MemNN (window memory + self-sup.)** | **0.666** | **0.630** | 0.690 | 0.703 |

\* Humanos sobre 10% del test set.

**Lectura del Goldilocks Principle en la tabla:**

1. **Verbos y preposiciones → el LM basta.** Los LSTM dominan: 0.813/0.802 (query). Notablemente, **superan a humanos en preposiciones** (0.802 vs 0.708 humano context+query, vs 0.676 humano query). Incluso con solo contexto local (query), LSTM y n-gram predicen *verbos* mejor que humanos en modo query (0.813/0.778 vs 0.716). La explicación de los autores: los modelos están mejor afinados a la distribución de verbos en libros infantiles, mientras los humanos se dejan influir por su conocimiento de todos los estilos de lenguaje. La memoria del contexto amplio aquí *no aporta*: LSTM (query) ≈ LSTM (context+query) en verbos (0.813 vs 0.818) y preposiciones (0.802 vs 0.791).

2. **Entidades y sustantivos → la memoria ayuda.** Aquí los LSTM se estancan: ~0.41 en entidades, ~0.55 en sustantivos, y **leer el contexto no los mejora** (0.408 query vs 0.418 context+query en entidades) — confirmando que no explotan efectivamente el contexto, síntoma del problema clásico de dependencias de largo plazo en RNN (Bengio et al. 1994). La MemNN **window+self-sup** salta a **0.666 en entidades** y **0.630 en sustantivos**, muy por encima de cualquier LM. El Embedding model (query), que es la MemNN sin memoria contextual, queda en 0.351/0.400 — la diferencia es justamente *el acceso a memoria*.

3. **Goldilocks del tamaño de chunk.** Sentential memory+PE es desastroso (0.318 entidades, 0.305 sustantivos): oración entera = muy grande. Lexical memory funciona para preposiciones/verbos (0.764/0.798) pero flojo en entidades/sustantivos (0.431/0.562): palabra suelta = muy pequeño para semántica. Window memory centrada en candidatos es lo "justo" para nombres y entidades. La self-supervision sobre window es lo que la lleva al tope.

4. **Brecha con humanos.** Los humanos con contexto llegan a 0.816 en entidades; el mejor modelo, 0.666. El benchmark deja margen amplio en justo las clases semánticas — su propósito de diseño.

### 6.2 CNN QA (generalización)

Para probar que el principio generaliza a otro estilo de lenguaje (noticias) y otra tarea (donde la respuesta es siempre una named entity anonimizada), aplican la mejor MemNN al CNN QA de Hermann et al. (93k artículos CNN):

| Método | Validation | Test |
|--------|---:|---:|
| Max frequency (article)* | 0.305 | 0.332 |
| Sliding window | 0.005 | 0.006 |
| Word distance model* | 0.505 | 0.509 |
| Deep LSTMs (article+query)* | 0.550 | 0.570 |
| Contextual LSTMs ("Attentive Reader")* | 0.616 | 0.630 |
| Contextual LSTMs ("Impatient Reader")* | 0.618 | 0.638 |
| MemNN (window memory) | 0.580 | 0.606 |
| MemNN (window memory + self-sup.) | 0.634 | 0.668 |
| MemNN (window memory + ensemble) | 0.612 | 0.638 |
| MemNN (window memory + self-sup. + ensemble) | 0.649 | 0.684 |
| MemNN (window + self-sup. + ensemble + excluding co-occurrences) | **0.662** | **0.694** |

\* Tomados de Hermann et al. (2015).

La window memory sin self-supervision (0.606 test) iguala más o menos a los reading models con ensemble; al **agregar self-supervision la MemNN supera claramente el estado del arte** (0.668), y con ensemble (0.684) y la heurística de excluir co-ocurrencias — quitar de la lista de candidatos las entidades que ya aparecen en el resumen — sube a **0.694**. El ensemble (11 modelos en self-sup) lo usan como sustituto del dropout de Hermann et al. (el promedio de ensemble tiene efecto similar al dropout, Wan et al. 2013).

La conclusión del análisis de CNN QA: los reading models, el CLSTM y la window MemNN comparten que representan el contexto en *chunks tipo ventana sub-oracional* (la BiRNN se enfoca naturalmente en una ventana alrededor de cada palabra). Lo que distingue a la mejor MemNN no es *cómo representa* sino **cómo accede/recupera**: la hard attention vía self-supervision hace el aprendizaje del acceso a memoria más tratable (aprender a acceder y usar información conjuntamente es una optimización difícil).

### 6.3 Ablations relevantes (apéndices)

- **Anonimización (Apéndice D):** anonimizar los candidatos en el CBT (como hace CNN QA) tiene impacto *bajo* en entidades (0.666 → 0.581) pero *grande* en tareas más sintácticas: verbos 0.690 → 0.474, preposiciones 0.703 → 0.522. Tiene sentido: la identidad léxica de verbos/preposiciones es justo lo que el modelo usa, y la anonimización la borra.
- **Componentes de la MemNN self-sup en CNN QA (Apéndice C):** quitar el *soft memory weighting* en test (volver a max duro) baja test de 0.668 a 0.620; quitar time features baja a 0.659. El soft weighting en test es el componente más importante.
- **Variantes de ventanas/targets (Apéndice E):** "all windows", "all targets" y "LM" rinden similar a candidate-windows (todas superan al baseline sin self-sup en entidades/sustantivos), pero impactan velocidad de train/test.

---

## 7. Limitaciones

1. **Dominio restringido.** Libros infantiles de Gutenberg: vocabulario, sintaxis y estilo narrativo particulares (siglos XIX-XX, dominio público). Que un modelo brille aquí no garantiza transferencia a texto técnico, conversacional o contemporáneo. Los propios autores reconocen que no hicieron que los anotadores humanos "calentaran" leyendo las 98 novelas de entrenamiento, lo que habría dado una comparación más justa con los modelos.
2. **Formato cloze artificial.** Predecir la palabra omitida de la oración 21 dadas 20 oraciones no es una tarea de comprensión natural; es un proxy. La generación automática vía POS/NER introduce sesgos del tagger y casos ambiguos (varias preposiciones "correctas", donde los modelos baten a humanos no por comprender mejor sino porque los anotadores prefieren la opción menos frecuente).
3. **Candidatos limitados y mismo POS.** 10 candidatos del mismo tipo acotan el espacio de respuesta; el modelo nunca enfrenta la tarea abierta de generar la palabra. La señal de self-supervision depende de tener candidatos conocidos (aunque el Apéndice E muestra que se puede prescindir de ellos).
4. **No anonimización = fuga de conocimiento de fondo.** Es una decisión de diseño deliberada, pero significa que parte del accuracy puede venir de regularidades léxicas/de frecuencia más que de comprensión genuina del contexto.
5. **Brecha con humanos solo medida en entidades/sustantivos al alza.** El benchmark no resuelve la pregunta de *por qué* las RNN no retienen contexto largo; solo lo evidencia.

---

## 8. Impacto

- **Refinó qué mide un cloze task.** La contribución conceptual más duradera: separar predicción de *function words* (sintaxis, frecuencia alta) de *content words* (semántica, frecuencia baja) y mostrar que el rol de la memoria es específico de la clase. Esto cambió cómo se interpreta el accuracy en lectura cloze: un número global esconde dos regímenes distintos.
- **Validó las window/sub-sentential representations.** El "sweet spot" sub-oracional dio sustento empírico a por qué las RNN bidireccionales con atención (Hermann et al.) y la atención local en NMT (Luong et al.) funcionan: todas convergen, por vías distintas, a representar chunks pequeños de texto. Esto influyó en el diseño de *attention/memory readers* posteriores.
- **Self-supervision como hard attention sin REINFORCE.** Mostrar que se puede inferir la supervisión de memoria desde la etiqueta y entrenar hard attention con backprop directa (sin gradiente de política) fue una idea práctica reutilizada.
- **Benchmark estándar junto a CNN/DM.** El CBT se volvió, junto con CNN/Daily Mail, uno de los benchmarks de referencia para machine reading 2016-2017. Trabajos posteriores como el *Attention Sum Reader*, *Gated-Attention Reader*, *EpiReader* y *AoA Reader* reportaron rutinariamente en CBT-NE y CBT-CN (las dos clases semánticas, donde está el interés), empujando el estado del arte sobre la línea que este paper estableció.
- **Precedente de SQuAD.** El CBT pertenece a la generación cloze que *precede* a SQuAD (Rajpurkar et al. 2016), el cual cambió el paradigma a *span extraction* sobre preguntas reales redactadas por humanos, superando varias de las limitaciones del cloze automático.

---

## 9. Conexión con la Clase 24

El PDF de la clase lista, entre los datasets de Question Answering / Reading Comprehension previos a la era de los grandes modelos, **"bAbI, Children's Book Test"** dentro de los datasets pre-2015/2016 (slide 21). Esto ubica al CBT exactamente en su lugar histórico:

- **Familia cloze que antecede a SQuAD.** La clase traza la evolución de los datasets de comprensión lectora: de los cloze automáticos (CNN/Daily Mail de Hermann, CBT, bAbI) hacia datasets con preguntas redactadas por humanos y respuestas extractivas (SQuAD) y luego abstractivas/generativas. El CBT es un eslabón clave de la primera etapa: cloze, gran escala, generado automáticamente, multiple-choice.
- **Conexión directa con CNN/Daily Mail (Hermann 2015).** Ambos comparten el formato cloze a escala y el linaje de FAIR/DeepMind. El propio paper de CBT valida sus conclusiones reproduciendo y superando los *Attentive/Impatient Readers* de Hermann sobre CNN QA. En la clase, ambos datasets aparecen como el sustrato sobre el que se desarrollaron los primeros *neural readers* con atención.
- **Motivación de los attention/memory readers.** La clase introduce los lectores neuronales basados en atención y memoria como el puente entre los LM clásicos y los modelos de comprensión modernos. El CBT es precisamente el banco de pruebas donde se diseccionó *por qué* y *cuándo* la atención/memoria ayuda: ayuda para el contenido semántico (entidades, sustantivos), no para la función sintáctica. Esa distinción es el insight que justifica arquitecturalmente los memory/attention readers que la clase presenta.
- **bAbI como hermano.** El CBT se distribuye junto a bAbI (`fb.ai/babi`), el suite de tareas de razonamiento de juguete de Weston et al., y comparte la maquinaria de Memory Networks. La clase agrupa ambos como la contribución de FAIR al estudio de memoria y razonamiento en NLP pre-transformers.

En síntesis para la clase: **el CBT es el experimento que mostró que "memoria" no es un ingrediente uniforme — su utilidad depende de qué se quiere predecir**. Ese matiz es lo que sobrevive a la transición hacia SQuAD y, más tarde, hacia los transformers, donde la atención se vuelve el mecanismo universal de acceso a contexto que aquí se estudiaba todavía de forma explícita y modular.

---

## 10. Notas y enlaces

- **arXiv:** `https://arxiv.org/abs/1511.02301` (v4, 1 Apr 2016).
- **Dataset:** `http://fb.ai/babi/` (CBT distribuido con bAbI). Fuente de textos: Project Gutenberg, `https://www.gutenberg.org/`.
- **Implementación:** Torch (`torch.ch`); código de MemNN lexical en `https://github.com/facebook/MemNN`.
- **Toolkits usados:** Stanford CoreNLP (POS + NER, Manning et al. 2014), KenLM (n-gram Kneser-Ney, Heafield et al. 2013).
- **Antecedentes técnicos clave:** End-to-End Memory Networks (Sukhbaatar et al. 2015), Memory Networks (Weston et al. 2015b), Teaching Machines to Read and Comprehend / CNN-DM (Hermann et al. 2015), Show Attend and Tell / hard attention (Xu et al. 2015), atención local en NMT (Luong et al. 2015).
- **Pieza de nomenclatura:** el nombre alude al cuento "Goldilocks and the Three Bears" (Hassall, 1904) — el principio de "ni muy grande, ni muy pequeño, sino justo" aplicado al tamaño de la representación de memoria.
- **Glosario rápido:** *cloze* = completar palabra omitida; *function words* = palabras de función sintáctica (preposiciones, artículos); *content words* = palabras de contenido semántico (nombres, sustantivos); *self-supervision* aquí = inferir la memoria de soporte desde la etiqueta para entrenar hard attention con backprop directa.
