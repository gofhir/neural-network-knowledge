# Memory Networks — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Memory Networks*.
- **Autores:** Jason Weston, Sumit Chopra y Antoine Bordes (Facebook AI Research, 770 Broadway, Nueva York).
- **Venue:** Publicado como conference paper en **ICLR 2015**. Preprint en arXiv:1410.3916 (la versión v11 es del 29 nov 2015; la primera subida es de octubre de 2014, de ahí que el modelo se cite a veces como "2014" y a veces como "ICLR 2015").
- **Dominio:** Aprendizaje automático con **memoria externa explícita**, aplicado a *question answering* (QA) sobre texto.
- **Contribución de una línea:** introduce una **clase de modelos** —las *memory networks*— que combinan un componente de inferencia entrenable con un **componente de memoria de largo plazo direccionable que se puede leer y escribir**, y se entrenan para usar ambos conjuntamente.

Este paper funda toda una familia. La tesis central es deliberadamente simple y, vista desde hoy, casi obvia: la mayoría de los modelos de machine learning **carecen de una forma fácil de leer y escribir sobre una porción de una memoria de largo plazo (potencialmente enorme) y combinarla con la inferencia**. El paper observa que esto desaprovecha "uno de los grandes activos de una computadora moderna" —la memoria de acceso aleatorio direccionable— y propone reintroducirla en la arquitectura neuronal de forma compartimentada y explícita.

El caso de uso que el paper elige para concretar la idea es el de *contar una historia o un conjunto de hechos a un sistema y luego hacerle preguntas sobre ese tema*. En ese contexto la memoria de largo plazo "actúa efectivamente como una base de conocimiento (dinámica)", y la salida es una respuesta textual. Los autores evalúan el modelo en dos extremos del espectro: una tarea de **QA a gran escala** (14M de hechos extraídos de ClueWeb09) y una tarea **de juguete pero más compleja**, generada por un mundo simulado, donde se exhibe la capacidad de **razonamiento multi-paso** encadenando varias frases de soporte para responder preguntas que exigen entender la intensión de los verbos ("picked up", "left") y el orden temporal de los eventos.

Para la **Clase 30 (modelos con memoria externa)** este es *el modelo básico*: la clase motiva primero por qué hace falta una memoria explícita y luego presenta Memory Networks como la implementación canónica de esa idea. Su única limitación seria —que requiere **supervisión fuerte** de cuáles frases son las de soporte— es justamente lo que motiva el siguiente paso de la clase, las *End-to-End Memory Networks*. Conviene tener presente esa flecha narrativa al leer lo que sigue.

## 2. Contexto histórico: el límite de RNN/LSTM y memoria implícita vs. explícita

Hacia 2014 el paradigma dominante para modelar secuencias era la **RNN** (Mikolov et al., 2010) y su variante con compuertas, la **LSTM** (Hochreiter & Schmidhuber, 1997). En principio una RNN entrenada como modelo de lenguaje podría resolver el QA sobre historias: lee un flujo de palabras y predice las siguientes. El paper argumenta por qué esto falla en la práctica, y el argumento es el corazón de su motivación.

La memoria de una RNN está **codificada en los estados ocultos y en los pesos**. Eso la hace *implícita* y *distribuida*: el conocimiento se comprime en vectores densos de tamaño fijo. El paper enumera tres problemas de esa memoria implícita. Primero, es **demasiado pequeña**: el estado oculto tiene dimensión fija y no crece con la longitud de la historia, de modo que recordar muchos hechos pasados obliga a sobrescribir información. Segundo, **no está suficientemente compartimentada**: los hechos no viven en ubicaciones separables y recuperables, sino mezclados en el mismo vector denso, lo que los hace difíciles de recordar con exactitud. Tercero —y esto el paper lo cita como evidencia dura— **las RNN tienen dificultad notoria con la memorización**: la simple tarea de copiar la secuencia de entrada que acaban de leer ya las pone en aprietos (Zaremba & Sutskever, 2014).

El experimento del mundo simulado (§5.2) hace tangible este límite: cuando la frase relevante para responder está más de 1 paso atrás en el tiempo, la *accuracy* de la RNN se desploma y la de la LSTM cae también, aunque menos —exactamente lo que se espera de una memoria que decae con la distancia temporal. El paper lo dice sin rodeos: "el pobre desempeño de la RNN se debe a su incapacidad de codificar memoria de largo plazo".

La alternativa que propone es una **memoria explícita**: un arreglo de objetos (vectores o cadenas de texto) indexado por posición, separado de los pesos del modelo, que crece para acomodar todos los hechos vistos y del cual el modelo puede leer y al cual puede escribir selectivamente. Esto contrasta con dos tradiciones que el paper sitúa en su sección de trabajo relacionado: las **memorias asociativas clásicas** (content-addressable), donde la memoria está "distribuida a lo largo de toda la red de pesos en vez de compartimentada en ubicaciones"; y el *memory-based learning* tipo vecino más cercano, que sí almacena ejemplos en compartimentos pero solo los usa para encontrar etiquetas cercanas. Memory Networks **combina lo mejor de ambos**: compartimentos de memoria explícitos *más* módulos neuronales que aprenden a leer y escribir en ellos, potencialmente de forma sucesiva (iterativa) para razonar.

Un dato de contexto que el propio paper subraya: el trabajo se subió a arXiv **justo antes** que las *Neural Turing Machines* de Graves et al. (2014), su pariente más cercano. La diferencia experimental es reveladora: las NTM limitan la memoria a 128 ubicaciones y atacan problemas algorítmicos (ordenar, copiar, recordar) con solución conocida; Memory Networks escala a **14M de frases** y ataca lenguaje y razonamiento, donde no hay solución algorítmica cerrada.

## 3. Contribución central

La contribución es un **marco general** —no un algoritmo único— que define una memoria de largo plazo más cuatro componentes entrenables que operan sobre ella. Formalmente, una *memory network* consiste en una memoria $m$ (un arreglo de objetos indexado por $m_i$) y cuatro componentes potencialmente aprendidos, **I, G, O y R**:

- **I — *input feature map*:** convierte la entrada entrante a la representación interna de características. Puede hacer preprocesamiento estándar (parsing, *coreference*, resolución de entidades) o codificar el texto en un vector disperso o denso.
- **G — *generalization*:** **actualiza** las memorias viejas dado el nuevo input. Se llama "generalización" porque hay una oportunidad de que la red **comprima y generalice** sus memorias en esta etapa para un uso futuro. Es el componente de **escritura**.
- **O — *output feature map*:** produce una nueva salida (en el espacio de representación de características) dado el nuevo input y el estado actual de la memoria. Es el componente de **lectura e inferencia**: calcula cuáles memorias son relevantes.
- **R — *response*:** convierte la salida de O al formato de respuesta deseado —por ejemplo una respuesta textual o una acción.

El flujo, ante una entrada $x$ (un carácter, palabra, frase, imagen o señal de audio, según la granularidad), es:

1. Convertir $x$ a representación interna: $I(x)$.
2. Actualizar las memorias: $m_i = G(m_i, I(x), m)$, para todo $i$.
3. Computar características de salida: $o = O(I(x), m)$.
4. Decodificar la salida a la respuesta final: $r = R(o)$.

Este proceso se aplica **tanto en entrenamiento como en test**: las memorias se escriben también en test (la historia nueva entra a la memoria), pero los parámetros de I/G/O/R no se actualizan en test. Esta es una propiedad de diseño importante —el modelo *almacena información nueva en inferencia sin reentrenar*—, justamente lo que una memoria explícita habilita y los pesos de una RNN no.

La genialidad del marco es su **generalidad**: I/G/O/R pueden usar cualquier idea de la literatura (SVMs, árboles de decisión, redes neuronales). Cuando los cuatro son redes neuronales, los autores las llaman **MemNN** (*memory neural networks*). El resto del paper instancia una MemNN concreta para texto.

## 4. Método: la instanciación MemNN para texto

### 4.1. El modelo básico y los *hops* de O

En la arquitectura básica, **I** toma una oración de entrada (la afirmación de un hecho o una pregunta) y **G** simplemente la escribe en el siguiente slot de memoria libre, en su forma original: $m_N = x$, $N = N+1$. Es decir, G aquí solo *almacena*; no toca memorias previas. El paper deja claro que variantes más sofisticadas de G podrían volver atrás y actualizar memorias anteriores, o, si la memoria fuera enorme (todo Freebase o Wikipedia), organizar los slots por entidad o tópico mediante una función de selección de slot $H(\cdot)$, e incluso implementar un "olvido" sobrescribiendo el slot menos útil. Pero en la versión básica nada de eso se usa.

El núcleo de la inferencia vive en **O y R**. O produce características de salida encontrando $k$ **memorias de soporte** (*supporting memories*) dado $x$. Para $k=1$ se recupera la memoria de mayor puntaje:

$$o_1 = O_1(x, m) = \arg\max_{i=1,\dots,N} s_O(x, m_i)$$

donde $s_O$ es una función que puntúa la coincidencia entre el par de oraciones $x$ y $m_i$. Para $k=2$ se busca una **segunda** memoria de soporte condicionada a la primera:

$$o_2 = O_2(x, m) = \arg\max_{i=1,\dots,N} s_O([x, m_{o_1}], m_i)$$

Aquí está el **razonamiento multi-paso** (*multi-hop*): el segundo hop puntúa cada candidato respecto *tanto* a la pregunta original *como* a la primera memoria de soporte ya encontrada. La salida final es la lista $o = [x, m_{o_1}, m_{o_2}]$, que se pasa a R. El paper usa $k$ hasta 2, pero el procedimiento generaliza a $k$ mayores.

El ejemplo canónico del paper (Figura 1) lo ilustra perfectamente. Ante la pregunta $x =$ "Where is the milk now?", O primero recupera $m_{o_1} =$ "Joe left the milk"; luego, dado $[x, m_{o_1}]$, recupera $m_{o_2} =$ "Joe travelled to the office" (el último lugar donde Joe fue antes de soltar la leche); y finalmente R produce $r =$ "office". Sin encadenar **dos** frases de soporte la respuesta es imposible: ese es el punto de demostración del paper sobre razonamiento.

### 4.2. El componente R y la función de puntaje

R debe producir la respuesta textual $r$. La opción más simple es devolver $m_{o_k}$ tal cual (la frase recuperada). Para generación real de oraciones se puede usar una RNN condicionada en la salida de O. El compromiso que los autores evalúan limita la respuesta a **una sola palabra**, rankeando todas las del vocabulario:

$$r = \arg\max_{w \in W} s_R([x, m_{o_1}, m_{o_2}], w)$$

Tanto $s_O$ como $s_R$ tienen la forma de un **modelo de embeddings**:

$$s(x, y) = \Phi_x(x)^\top U^\top U \, \Phi_y(y)$$

donde $U$ es una matriz $n \times D$ ($D$ = número de características, $n$ = dimensión del embedding), y $\Phi_x, \Phi_y$ mapean el texto al espacio de características $D$-dimensional. La representación más simple es *bag of words*; el paper usa $D = 3|W|$ porque cada palabra tiene tres representaciones distintas (una para $\Phi_y$ y dos para $\Phi_x$, según si la palabra viene del input real $x$ o de las memorias de soporte, para modelarlas distinto). $s_O$ y $s_R$ usan matrices de peso **distintas** $U_O$ y $U_R$.

### 4.3. La supervisión fuerte: el detalle que define al modelo

Aquí está el rasgo más importante de esta versión, el que la Clase 30 destaca como su limitación. El entrenamiento es **completamente supervisado** en un sentido fuerte: además de los inputs y las respuestas deseadas, **las oraciones de soporte están etiquetadas como tales en los datos de entrenamiento** (no en test). Es decir, durante el entrenamiento el modelo *sabe* cuál es la elección correcta de ambos $\arg\max$ de las ecuaciones de O. El paper lo nota de pasada pero es crucial: "métodos como las RNN y LSTM no pueden usar fácilmente esta información", lo que da a la MemNN una ventaja de supervisión que sus baselines no tienen.

El entrenamiento usa una **pérdida de ranking con margen** y SGD. Para una pregunta $x$ con respuesta verdadera $r$ y soportes $m_{o_1}, m_{o_2}$, se minimiza una suma de tres familias de términos *hinge*:

$$\sum_{\bar{f} \neq m_{o_1}} \max(0,\ \gamma - s_O(x, m_{o_1}) + s_O(x, \bar{f}))$$
$$+ \sum_{\bar{f}' \neq m_{o_2}} \max(0,\ \gamma - s_O([x, m_{o_1}], m_{o_2}) + s_O([x, m_{o_1}], \bar{f}'))$$
$$+ \sum_{\bar{r} \neq r} \max(0,\ \gamma - s_R([x, m_{o_1}, m_{o_2}], r) + s_R([x, m_{o_1}, m_{o_2}], \bar{r}))$$

donde $\bar{f}, \bar{f}', \bar{r}$ son las elecciones incorrectas y $\gamma$ es el margen. Cada término empuja el puntaje de la elección correcta (el soporte correcto en el hop 1, en el hop 2, y la palabra-respuesta correcta) por encima de cualquier elección incorrecta por al menos $\gamma$. En cada paso de SGD se **muestrea** un negativo en lugar de computar la suma completa (siguiendo Weston et al., 2011). Si R es una RNN, el último término se reemplaza por la log-verosimilitud estándar de modelado de lenguaje sobre la secuencia $[x, o_1, o_2, r]$.

La consecuencia conceptual es directa: **el modelo no aprende a buscar las memorias relevantes desde cero; aprende a puntuar memorias que ya le marcaron como relevantes**. Sin esas etiquetas de soporte, esta formulación no tiene gradiente que entrenar los hops.

### 4.4. Extensiones (lo que hace el modelo robusto en la práctica)

El paper agrega varias extensiones que importan para que el modelo funcione fuera del juguete:

- **Secuencias de palabras como input (§3.2):** si las palabras llegan en flujo y no pre-segmentadas en frases, se añade una **función de segmentación entrenable** $\text{seg}(c) = W_{seg}^\top U_S \Phi_{seg}(c)$ que dispara cuando reconoce un segmento completo, escribiéndolo entonces en memoria. Esto da a la MemNN un componente de aprendizaje **en la operación de escritura**.
- **Memoria eficiente vía *hashing* (§3.3):** puntuar las 14M memorias por pregunta es prohibitivo. Se *hashea* el input en *buckets* y solo se puntúan las memorias del mismo bucket. Dos variantes: (i) *hashing* por palabra (un bucket por palabra del diccionario) y (ii) *clustering* de embeddings con K-means tras entrenar $U_O$. El *clustering* (con $K=1000$) da ~80× de aceleración manteniendo desempeño, mientras que el *hashing* por palabra es más rápido pero pierde calidad (no empareja respuestas que no comparten palabras).
- **Modelado del tiempo de escritura (§3.4):** para historias importa *cuándo* se escribió cada slot. En vez de codificar el índice absoluto, se aprende una función sobre **tripletas** $s_{Ot}(x, y, y')$ con tres rasgos binarios de orden relativo (si $x$ es más viejo que $y$, $x$ más viejo que $y'$, $y$ más viejo que $y'$), y el $\arg\max$ se reemplaza por un bucle que mantiene la memoria ganadora. Estos rasgos resultan **necesarios** para las preguntas "before" o de dificultad > 1.
- **Palabras nunca vistas (§3.5–3.6):** para manejar palabras nuevas (el ejemplo es "Boromir" la primera vez que aparece en *El Señor de los Anillos*), se almacena por cada palabra una *bag of words* de su contexto izquierdo y derecho, expandiendo $D$ a $5|W|$, y se entrena con una especie de *dropout* que finge no haber visto la palabra el $d\%$ del tiempo. Además, como los embeddings de baja dimensión $n$ no capturan bien las **coincidencias exactas** de palabras, se agrega un término de *matching* tipo *bag of words* (llegando a $D = 8|W|$).

## 5. Experimentos

### 5.1. QA a gran escala

Sobre el dataset de Fader et al. (2013): **14M de afirmaciones** almacenadas como tripletas (sujeto, relación, objeto) —extracciones REVERB de ClueWeb09, p.ej. (milne, authored, winnie-the-pooh)— más 35M de pares de preguntas parafraseadas de WikiAnswers. La tarea es re-rankear respuestas candidatas midiendo F1. Con $k=1$:

| Método | F1 |
|---|---|
| Fader et al. (2013) | 0.54 |
| Bordes et al. (2014b) | 0.73 |
| MemNN (solo embedding) | 0.72 |
| **MemNN (con rasgos BoW)** | **0.82** |

La MemNN con rasgos BoW es **el mejor sistema**. El *hashing* por *clusters* preserva casi todo el desempeño (0.80 vs 0.82) reduciendo los candidatos de 14M a 177k (~80× más rápido).

### 5.2. QA del mundo simulado (estilo bAbI)

Esta es la parte que demuestra **razonamiento**. Los autores construyen una simulación con 4 personajes, 3 objetos y 5 habitaciones; los personajes se mueven, recogen y sueltan objetos; las acciones se transcriben a texto con una gramática simple (con sinónimos: *get* → "picked up"/"got"/"grabbed"/"took"). Se generan 7k afirmaciones y 3k preguntas para entrenar y otras tantas para test. Esta tarea es el **antecedente directo del dataset bAbI** ([/papers/babi-weston-2015](/papers/babi-weston-2015)), que los mismos autores formalizarían poco después como un conjunto de 20 tareas de razonamiento. Los resultados (respuesta de una palabra) comparan MemNN contra RNN y LSTM:

| Método | Dif. 1 actor s/before | Dif. 1 actor | Dif. 1 actor+obj | Dif. 5 actor | Dif. 5 actor+obj |
|---|---|---|---|---|---|
| RNN | 100% | 60.9% | 27.9% | 23.8% | 17.8% |
| LSTM | 100% | 64.8% | 49.1% | 35.2% | 29.0% |
| MemNN $k=1$ | 97.8% | 31.0% | 24.0% | 21.9% | 18.5% |
| MemNN $k=1$ (+time) | 99.9% | 60.2% | 42.5% | 60.8% | 44.4% |
| **MemNN $k=2$ (+time)** | **100%** | **100%** | **100%** | **100%** | **99.9%** |

Las lecciones son nítidas. RNN y LSTM resuelven la tarea más simple (actor, dificultad 1, sin "before") pero **colapsan** cuando la frase relevante está lejos en el tiempo (dificultad 5) o cuando hay preguntas "before" —exactamente el límite de la memoria implícita. Los **rasgos de tiempo son necesarios** para las preguntas "before" y dificultad > 1 (sin ellos $s_O$ puede elegir una frase sobre el paradero de una persona que ya se movió). Y el **segundo hop ($k=2$) es necesario** para la tarea actor+objeto: la MemNN con $k=1$ falla igual que las RNN/LSTM, mientras que con $k=2$ alcanza prácticamente el 100%. Es la prueba experimental de que la inferencia de 2 etapas resuelve lo que un solo paso no puede.

Dos experimentos adicionales confirman la generalidad. Con la extensión de **palabras nuevas** (§5.2.1), la MemNN responde correctamente una historia de *El Señor de los Anillos* con nombres nunca vistos (Bilbo, Frodo, Sauron, Gollum) descubriendo patrones verbales como (X, dropped, Y); sin el modelado de palabras nuevas, falla por completo. Y un **ensemble** de los modelos de §5.1 y §5.2 (§5.3) permite un diálogo que mezcla conocimiento general ("Where does milk come from? → milk come from cow") con preguntas sobre la historia local ("Where is the milk? → office").

## 6. Limitaciones reconocidas

La limitación que define el legado del paper es la **supervisión fuerte de los hops**: el modelo necesita que los datos de entrenamiento etiqueten *cuáles* frases son las de soporte para cada pregunta. Los propios autores lo señalan en las conclusiones como dirección de trabajo futuro: "los settings débilmente supervisados son muy importantes y deberían explorarse, ya que muchos datasets solo tienen supervisión en la forma de pares pregunta-respuesta, y no de hechos de soporte como usamos aquí". Esto es exactamente lo que **End-to-End Memory Networks** (Sukhbaatar et al., 2015) resolvería poco después, reemplazando los $\arg\max$ duros y supervisados por una **atención *softmax* diferenciable** sobre toda la memoria, entrenable solo con la respuesta final.

Otras limitaciones que el paper admite o deja entrever:

- **G es trivial en la versión básica:** solo escribe en el siguiente slot libre; no actualiza, comprime ni olvida memorias previas, pese a que el marco lo permitiría. El "olvido" no se exploró experimentalmente.
- **Costo lineal de lectura:** sin *hashing*, puntuar todas las memorias es lineal en el tamaño de la memoria (lento con 14M de hechos). El *hashing* mitiga pero introduce un *trade-off* velocidad-exactitud.
- **Datos de juguete:** la simulación carece de *coreference* ("He picked up the milk"), frases nominales compuestas y estructura rica; los autores la presentan como un *prueba de concepto* y prerrequisito, no como sustituto de datos reales.
- **Número fijo y pequeño de hops:** se usa $k \leq 2$; tareas que requieran más saltos de inferencia exigirían más hops y, con la supervisión fuerte, más etiquetas.

## 7. Impacto: la familia de las memory networks

Memory Networks **fundó una familia** de arquitecturas con memoria externa que dominó el QA y el razonamiento sobre texto en 2015–2017. La línea directa es: este paper → **dataset bAbI** (Weston et al., 2015), que estandarizó las 20 tareas de razonamiento derivadas de la simulación de §5.2 → **End-to-End Memory Networks** (Sukhbaatar et al., 2015), que eliminó la supervisión fuerte vía atención *softmax* → **Key-Value Memory Networks**, **Dynamic Memory Networks**, y la convergencia conceptual con las *Neural Turing Machines* / *Differentiable Neural Computers* de Graves et al. Conceptualmente, la idea de **leer de una memoria mediante puntajes de coincidencia y razonar en múltiples pasos** es un antecedente directo del mecanismo de **atención** que pocos años después se volvería el núcleo del Transformer: un hop de O es, en esencia, una consulta que atiende sobre un conjunto de memorias.

## 8. Conexión con la Clase 30 (modelos con memoria externa)

La Clase 30 ([/clases/clase-30](/clases/clase-30)) construye su narrativa exactamente sobre el arco de este paper. Primero **motiva** la memoria explícita mostrando dónde fallan RNN/LSTM (memoria implícita, pequeña, no compartimentada, que decae con la distancia temporal). Luego presenta Memory Networks bajo el rótulo de **"Modelo básico — Memory networks"**: es el primer modelo concreto que materializa la idea de una memoria de slots leíble y escribible más los cuatro componentes I/G/O/R. La clase usa el ejemplo de la leche y Joe (Figura 1) para hacer visible el **razonamiento de 2 hops**, y el dataset **bAbI** ([/papers/babi-weston-2015](/papers/babi-weston-2015)) —descendiente directo de la simulación de §5.2— como banco de pruebas de las 20 tareas de razonamiento.

El punto pedagógico clave que la clase extrae es la **limitación de supervisión fuerte**: como esta versión necesita que le digan cuáles frases son de soporte, no escala a los datasets reales que solo tienen pares pregunta-respuesta. Esa fricción es la bisagra que motiva el siguiente modelo de la clase, **End-to-End Memory Networks**, donde la lectura se vuelve una atención diferenciable entrenable de punta a punta. Entender Memory Networks es, por tanto, entender *por qué hizo falta* el modelo end-to-end: no es un modelo distinto por capricho, sino la respuesta directa al cuello de botella que este paper dejó explícitamente abierto.

Para profundizar en los conceptos transversales —memoria direccionable, lectura/escritura diferenciable, la relación con atención y con las Neural Turing Machines— ver el fundamento [/fundamentos/memory-augmented-networks](/fundamentos/memory-augmented-networks).
