---
title: "Memory Networks (2014)"
weight: 341
math: true
---

{{< paper-card
    title="Memory Networks"
    authors="Jason Weston, Sumit Chopra, Antoine Bordes"
    year="2014"
    venue="ICLR 2015"
    pdf="/papers/memory-networks-weston-2014.pdf"
    arxiv="1410.3916" >}}
Paper de Facebook AI Research que funda toda una familia de arquitecturas con **memoria externa explícita**. Su tesis es simple y, vista desde hoy, casi obvia: los modelos de machine learning carecen de una forma fácil de leer y escribir sobre una memoria de largo plazo (potencialmente enorme) y combinarla con la inferencia, desperdiciando "uno de los grandes activos de una computadora moderna" —la memoria direccionable. Propone una memoria de slots leíble y escribible más cuatro componentes entrenables (**I, G, O, R**), capaz de **razonamiento multi-paso** (multi-hop) encadenando varias frases de soporte para responder preguntas. Su única limitación seria —que requiere **supervisión fuerte** de cuáles frases son las de soporte— es justamente lo que motiva el siguiente modelo de la [Clase 30](/clases/clase-30), las [End-to-End Memory Networks](/papers/e2e-memnn-sukhbaatar-2015).
{{< /paper-card >}}

---

## Contexto: el límite de la memoria implícita

Hacia 2014 el paradigma dominante para secuencias era la **RNN** y su variante con compuertas, la **LSTM**. En principio una RNN entrenada como modelo de lenguaje podría resolver el *question answering* sobre historias: lee un flujo de palabras y predice las siguientes. El paper explica por qué esto falla en la práctica, y ese argumento es el corazón de su motivación.

La memoria de una RNN está **codificada en los estados ocultos y en los pesos**. Eso la hace *implícita* y *distribuida*: el conocimiento se comprime en vectores densos de tamaño fijo. El paper enumera tres problemas. Primero, es **demasiado pequeña**: el estado oculto tiene dimensión fija y no crece con la longitud de la historia, así que recordar muchos hechos obliga a sobrescribir información. Segundo, **no está suficientemente compartimentada**: los hechos no viven en ubicaciones separables y recuperables, sino mezclados en el mismo vector denso, lo que los hace difíciles de recordar con exactitud. Tercero, **las RNN tienen dificultad notoria con la memorización**: la simple tarea de copiar la secuencia que acaban de leer ya las pone en aprietos.

La alternativa es una **memoria explícita**: un arreglo de objetos (vectores o cadenas de texto) indexado por posición, separado de los pesos del modelo, que crece para acomodar todos los hechos vistos y del cual el modelo puede leer y al cual puede escribir selectivamente. Esto contrasta con dos tradiciones: las **memorias asociativas clásicas**, donde la memoria está distribuida en toda la red de pesos en vez de compartimentada; y el *memory-based learning* tipo vecino más cercano, que almacena ejemplos en compartimentos pero solo los usa para encontrar etiquetas cercanas. Memory Networks combina lo mejor de ambos: **compartimentos explícitos más módulos neuronales que aprenden a leer y escribir en ellos**, potencialmente de forma sucesiva para razonar.

Un dato de contexto que el propio paper subraya: el trabajo se subió a arXiv justo antes que las *Neural Turing Machines* de Graves et al. (2014), su pariente más cercano. La diferencia experimental es reveladora: las NTM limitan la memoria a 128 ubicaciones y atacan problemas algorítmicos (ordenar, copiar) con solución conocida; Memory Networks escala a **14 millones de frases** y ataca lenguaje y razonamiento, donde no hay solución algorítmica cerrada.

## El marco I/G/O/R

La contribución es un **marco general** —no un algoritmo único— que define una memoria de largo plazo más cuatro componentes entrenables que operan sobre ella. Una *memory network* consiste en una memoria $m$ (un arreglo de objetos indexado por $m_i$) y cuatro componentes potencialmente aprendidos:

- **I — *input feature map*:** convierte la entrada a la representación interna de características (parsing, *coreference*, codificación en vectores dispersos o densos).
- **G — *generalization*:** **actualiza** las memorias dado el nuevo input. Se llama "generalización" porque hay oportunidad de comprimir y generalizar las memorias para uso futuro. Es el componente de **escritura**.
- **O — *output feature map*:** produce una salida dado el nuevo input y el estado de la memoria. Es el componente de **lectura e inferencia**: calcula cuáles memorias son relevantes.
- **R — *response*:** convierte la salida de O al formato deseado —una respuesta textual o una acción.

El flujo, ante una entrada $x$, es: convertir $x$ a representación interna, $I(x)$; actualizar memorias, $m_i = G(m_i, I(x), m)$; computar características de salida, $o = O(I(x), m)$; y decodificar la respuesta final, $r = R(o)$. Este proceso se aplica **tanto en entrenamiento como en test**: las memorias se escriben también en test (la historia nueva entra a la memoria), pero los parámetros de I/G/O/R no se actualizan. Es una propiedad de diseño importante —el modelo *almacena información nueva en inferencia sin reentrenar*—, justo lo que una memoria explícita habilita y los pesos de una RNN no.

La generalidad del marco es su gracia: I/G/O/R pueden usar cualquier idea de la literatura (SVMs, árboles, redes neuronales). Cuando los cuatro son redes neuronales, los autores las llaman **MemNN** (*memory neural networks*).

## La instanciación MemNN para texto

En la arquitectura básica, **I** toma una oración (un hecho o una pregunta) y **G** simplemente la escribe en el siguiente slot libre, en su forma original: $m_N = x$. Es decir, G aquí solo *almacena*; no toca memorias previas. El núcleo de la inferencia vive en **O y R**. O encuentra $k$ **memorias de soporte** dado $x$. Para $k=1$ se recupera la de mayor puntaje:

$$o_1 = O_1(x, m) = \arg\max_{i=1,\dots,N} s_O(x, m_i)$$

Para $k=2$ se busca una **segunda** memoria de soporte condicionada a la primera:

$$o_2 = O_2(x, m) = \arg\max_{i=1,\dots,N} s_O([x, m_{o_1}], m_i)$$

Aquí está el **razonamiento multi-paso** (*multi-hop*): el segundo hop puntúa cada candidato respecto *tanto* a la pregunta original *como* a la primera memoria ya encontrada. El ejemplo canónico del paper lo ilustra: ante $x =$ "Where is the milk now?", O primero recupera $m_{o_1} =$ "Joe left the milk"; luego, dado $[x, m_{o_1}]$, recupera $m_{o_2} =$ "Joe travelled to the office"; y R produce $r =$ "office". Sin encadenar **dos** frases de soporte la respuesta es imposible: ese es el punto de demostración del paper.

R produce la respuesta textual. La versión evaluada la limita a **una sola palabra**, rankeando el vocabulario completo. Tanto $s_O$ como $s_R$ tienen la forma de un **modelo de embeddings**:

$$s(x, y) = \Phi_x(x)^\top U^\top U \, \Phi_y(y)$$

donde $U$ es una matriz $n \times D$ y $\Phi_x, \Phi_y$ mapean el texto al espacio de características. La representación más simple es *bag of words*; $s_O$ y $s_R$ usan matrices distintas $U_O$ y $U_R$.

## Supervisión fuerte: el detalle que define al modelo

Aquí está el rasgo más importante de esta versión, y su limitación. El entrenamiento es **completamente supervisado** en un sentido fuerte: además de los inputs y las respuestas deseadas, **las oraciones de soporte están etiquetadas como tales en los datos de entrenamiento**. Es decir, durante el entrenamiento el modelo *sabe* cuál es la elección correcta de ambos $\arg\max$. El paper lo nota de pasada pero es crucial: las RNN y LSTM no pueden usar fácilmente esta información, lo que da a la MemNN una ventaja de supervisión que sus baselines no tienen.

El entrenamiento usa una **pérdida de ranking con margen** y SGD: para cada hop y para la palabra-respuesta, empuja el puntaje de la elección correcta por encima de cualquier elección incorrecta por al menos un margen $\gamma$. La consecuencia conceptual es directa: **el modelo no aprende a buscar las memorias relevantes desde cero; aprende a puntuar memorias que ya le marcaron como relevantes**. Sin esas etiquetas de soporte, esta formulación no tiene gradiente que entrenar los hops.

Varias extensiones lo hacen robusto fuera del juguete: una **función de segmentación entrenable** para flujos de palabras no pre-segmentadas; **hashing** de la memoria (por palabra o por *clustering* K-means de embeddings) para no puntuar las 14M memorias por pregunta —el clustering con $K=1000$ da ~80× de aceleración manteniendo desempeño; **modelado del tiempo de escritura** con rasgos de orden relativo sobre tripletas, necesarios para las preguntas "before"; y manejo de **palabras nunca vistas** vía contexto izquierdo/derecho más un término de *matching* exacto.

## Experimentos

**QA a gran escala.** Sobre el dataset de Fader et al. (2013): **14M de afirmaciones** como tripletas (sujeto, relación, objeto) extraídas de ClueWeb09, más 35M de preguntas parafraseadas de WikiAnswers. Con $k=1$, la MemNN con rasgos *bag of words* alcanza **F1 = 0.82**, el mejor sistema (vs. 0.73 de Bordes et al. 2014b y 0.54 de Fader et al.). El *hashing* por clusters preserva casi todo (0.80) reduciendo los candidatos de 14M a 177k.

**QA del mundo simulado (estilo bAbI).** Esta es la parte que demuestra **razonamiento**. Los autores construyen una simulación con 4 personajes, 3 objetos y 5 habitaciones, transcrita a texto con sinónimos verbales. Esta tarea es el antecedente directo del **dataset bAbI** ([/papers/babi-weston-2015](/papers/babi-weston-2015)), que los mismos autores formalizarían poco después como 20 tareas de razonamiento.

| Método | Dif. 1 actor | Dif. 1 actor+obj | Dif. 5 actor+obj |
|---|---|---|---|
| RNN | 60.9% | 27.9% | 17.8% |
| LSTM | 64.8% | 49.1% | 29.0% |
| MemNN $k=1$ (+time) | 60.2% | 42.5% | 44.4% |
| **MemNN $k=2$ (+time)** | **100%** | **100%** | **99.9%** |

Las lecciones son nítidas. RNN y LSTM resuelven la tarea más simple pero **colapsan** cuando la frase relevante está lejos en el tiempo o cuando hay preguntas "before" —exactamente el límite de la memoria implícita. Los rasgos de tiempo son necesarios para esas preguntas. Y el **segundo hop ($k=2$) es necesario** para la tarea actor+objeto: con $k=1$ la MemNN falla igual que las RNN/LSTM, mientras que con $k=2$ alcanza casi el 100%. Es la prueba experimental de que la inferencia de dos etapas resuelve lo que un solo paso no puede. Experimentos adicionales confirman la generalidad: la MemNN responde una historia de *El Señor de los Anillos* con nombres nunca vistos descubriendo patrones verbales, y un *ensemble* mezcla conocimiento general con preguntas sobre la historia local.

## Limitaciones

La limitación que define el legado del paper es la **supervisión fuerte de los hops**: el modelo necesita que los datos de entrenamiento etiqueten *cuáles* frases son las de soporte. Los propios autores lo señalan: los settings débilmente supervisados son muy importantes, ya que muchos datasets solo tienen pares pregunta-respuesta. Esto es exactamente lo que las **End-to-End Memory Networks** (Sukhbaatar et al., 2015) resolverían poco después, reemplazando los $\arg\max$ duros y supervisados por una **atención *softmax* diferenciable** sobre toda la memoria, entrenable solo con la respuesta final.

Otras limitaciones: **G es trivial** en la versión básica (solo escribe en el siguiente slot libre; no actualiza, comprime ni olvida); **costo lineal de lectura** sin *hashing* (lento con 14M de hechos); **datos de juguete** sin *coreference* ni estructura rica, presentados como prueba de concepto; y un **número fijo y pequeño de hops** ($k \leq 2$).

## Impacto: la familia de las memory networks

Memory Networks **fundó una familia** de arquitecturas con memoria externa que dominó el QA y el razonamiento sobre texto en 2015–2017. La línea directa es: este paper → **dataset bAbI** (Weston et al., 2015), que estandarizó las 20 tareas de razonamiento → **End-to-End Memory Networks** (Sukhbaatar et al., 2015), que eliminó la supervisión fuerte vía atención *softmax* → **Key-Value Memory Networks**, **Dynamic Memory Networks**, y la convergencia conceptual con las *Neural Turing Machines* / *Differentiable Neural Computers* de Graves et al. Conceptualmente, la idea de **leer de una memoria mediante puntajes de coincidencia y razonar en múltiples pasos** es un antecedente directo del mecanismo de **atención** que pocos años después se volvería el núcleo del Transformer: un hop de O es, en esencia, una consulta que atiende sobre un conjunto de memorias.

## Por qué importa para la Clase 30

La [Clase 30](/clases/clase-30) construye su narrativa sobre el arco de este paper. Primero **motiva** la memoria explícita mostrando dónde fallan RNN/LSTM (memoria implícita, pequeña, no compartimentada, que decae con la distancia temporal). Luego presenta Memory Networks como el **modelo básico**: el primer modelo concreto que materializa una memoria de slots leíble y escribible más los cuatro componentes I/G/O/R. La clase usa el ejemplo de la leche y Joe para hacer visible el razonamiento de dos hops, y el dataset [bAbI](/papers/babi-weston-2015) como banco de pruebas.

El punto pedagógico clave es la **limitación de supervisión fuerte**: como esta versión necesita que le digan cuáles frases son de soporte, no escala a los datasets reales que solo tienen pares pregunta-respuesta. Esa fricción es la bisagra que motiva el siguiente modelo de la clase, las [End-to-End Memory Networks](/papers/e2e-memnn-sukhbaatar-2015), donde la lectura se vuelve una atención diferenciable entrenable de punta a punta. Entender Memory Networks es entender *por qué hizo falta* el modelo end-to-end. Para profundizar en los conceptos transversales —memoria direccionable, lectura/escritura diferenciable, la relación con atención y con las Neural Turing Machines— ver el fundamento [/fundamentos/redes-de-memoria](/fundamentos/redes-de-memoria).
