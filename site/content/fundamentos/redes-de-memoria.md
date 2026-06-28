---
title: "Redes con Memoria Externa"
weight: 107
math: true
---

Una **red con memoria externa** (memory-augmented network) es una arquitectura neuronal que separa explícitamente dos cosas que las redes clásicas mantienen fundidas: el **programa** —cómo procesar la información, codificado en los pesos— y el **conocimiento** —los hechos concretos sobre los que se razona, almacenados en una memoria direccionable aparte de los pesos. En lugar de comprimir todo lo que sabe en vectores densos de estado oculto, el modelo escribe los hechos en un arreglo de **slots** que puede leer, actualizar y consultar selectivamente. Esa separación, banal en un computador pero ausente en una RNN, es lo que vuelve la memoria **interpretable, editable y separable** del razonamiento. Este fundamento es el núcleo conceptual de la [Clase 30](/clases/clase-30): recorre por qué hizo falta la memoria explícita, las dos grandes estirpes de modelos que la implementan, el mecanismo común que las une —la lectura como atención suave— y su legado directo en los Transformers y en la generación aumentada por recuperación.

---

## 1. El problema: la memoria implícita de RNN y LSTM

El paradigma dominante para modelar secuencias hacia 2014 era la **RNN** y su variante con compuertas, la **LSTM**. En teoría, una RNN entrenada como modelo de lenguaje podría leer una historia y responder preguntas sobre ella; en la práctica falla, y el porqué define todo el campo. La memoria de una RNN está **codificada en los estados ocultos y en los pesos**: es *implícita*, *distribuida* y de tamaño fijo. El conocimiento se comprime en un vector denso que no crece con la longitud de la historia.

Esto acarrea tres problemas concretos. Primero, la memoria es **demasiado pequeña**: el estado oculto tiene dimensión fija, de modo que recordar muchos hechos obliga a sobrescribir información antigua. Segundo, **no está compartimentada**: los hechos no viven en ubicaciones separables y recuperables, sino mezclados en el mismo vector, lo que los hace difíciles de recordar con exactitud. Tercero, **decae con la distancia temporal**: cuando la frase relevante para responder está varios pasos atrás, la *accuracy* se desploma —las RNN tienen dificultad notoria incluso con la tarea de copiar la secuencia que acaban de leer.

El defecto de fondo es que la memoria está **fundida con el programa**. Para agregar un hecho nuevo —que cierta película la dirigió cierta persona— habría que reentrenar la red y modificar sus pesos, sin garantía de no dañar lo aprendido. No se puede inspeccionar "dónde" vive un hecho ni editarlo quirúrgicamente. La alternativa es una **memoria explícita**: un arreglo de objetos (vectores o cadenas de texto) indexado por posición, separado de los pesos, que crece para acomodar todos los hechos vistos y del cual el modelo lee y al cual escribe selectivamente.

{{< concept-alert type="clave" >}}
La distinción rectora de todo el tema es **memoria implícita vs. explícita**. La memoria *implícita* (pesos y estado oculto de una RNN) es densa, opaca y de tamaño fijo: agregar o editar un hecho exige reentrenar. La memoria *explícita* (slots direccionables) es interpretable —cada slot guarda algo identificable—, editable —se escribe un slot sin tocar el resto— y separable del "programa" que la opera. Toda la Clase 30 gira en torno a esta separación.
{{< /concept-alert >}}

---

## 2. Memoria como pesos vs. memoria como base de conocimiento: la analogía von Neumann

El marco que mejor ilumina la idea es la **arquitectura von Neumann** (1945), base de todo computador moderno. Un programa de computador descansa sobre tres mecanismos: operaciones elementales (aritmética), control de flujo lógico (ramificación) y una **memoria externa direccionable** que se lee y escribe durante el cómputo. El aprendizaje automático había tenido un éxito enorme con las operaciones, pero "había descuidado el control de flujo y la memoria externa". Las redes con memoria externa vienen a tapar ese hueco.

La analogía es precisa. En una red clásica, **cómputo y memoria están mezclados** en los pesos y en las activaciones: es un "pasivo importante", porque la red no puede asignar almacenamiento nuevo cuando crecen las demandas de una tarea, ni aprender con facilidad algoritmos que actúen con independencia de los valores concretos de las variables. En un computador, en cambio, el **procesador (CPU)** ejecuta un programa fijo que opera sobre **datos (RAM)** intercambiables: para aplicar el mismo procedimiento a un dato u otro basta cambiar la dirección de la que se lee. Esa capacidad de tratar los contenidos de la memoria como **variables** es justo lo que da *generalidad algorítmica*.

Las redes con memoria externa importan esa separación al mundo neuronal. El **controlador** (una red, recurrente o feedforward) hace de CPU con instrucciones *aprendidas* en vez de predefinidas; la **matriz de memoria** hace de RAM; y, si el controlador es recurrente, sus activaciones ocultas son los **registros**. Cuando el caso de uso es contarle hechos a un sistema y luego preguntarle sobre ellos, esa memoria de largo plazo "actúa efectivamente como una base de conocimiento dinámica". La diferencia decisiva con un computador real es que **toda la maquinaria es diferenciable**: el modelo no ejecuta un programa escrito a mano, sino que **aprende a operar la memoria** por descenso de gradiente, a partir de ejemplos de entrada-salida.

---

## 3. Las dos grandes estirpes

De esta idea fundacional —una memoria explícita, separada de los pesos, que el modelo lee y edita— surgieron **dos linajes** distintos, con orígenes, motivaciones y estilos de direccionamiento diferentes. Conviene fijarlos antes de entrar al detalle.

| Aspecto | Memory Networks (Weston, FAIR) | Memoria diferenciable tipo computador (Graves, DeepMind) |
|---|---|---|
| Origen | *Memory Networks* (Weston et al., 2014) | *Neural Turing Machine* (Graves et al., 2014) |
| Motivación | QA y razonamiento sobre **texto** | Aprender **algoritmos** (copiar, ordenar, recall) |
| Memoria | Slots que almacenan **hechos/frases** | Matriz $N \times M$ tipo cinta de Turing |
| Direccionamiento | Por **contenido** (similitud con la consulta) | Por contenido **y** por **ubicación** (índices) |
| Escritura | Trivial (escribe el siguiente slot libre) | Borrar + añadir, con asignación dinámica (DNC) |
| Escala probada | 14M de hechos | ~128 ubicaciones, tareas algorítmicas |
| Linaje | MemNN → E2E → KV → EntNet | NTM → DNC; y MANN para one-shot |

La diferencia esencial está en el **estilo de direccionamiento** —cómo decide el modelo a qué slot acceder—. La línea Weston trata la memoria como un **conjunto de slots de contenido**: cada slot es un hecho, y se accede a él por su *parecido semántico* con la pregunta. No importa *dónde* esté guardado el hecho, sino que su representación coincida con la consulta; el modelo nunca necesita la noción de "la posición siguiente". La línea Graves, en cambio, trata la memoria como la **RAM de un computador**: una cinta que se recorre por **índices**, donde además del parecido por contenido el modelo puede *desplazar el foco* a la posición contigua, asignar ubicaciones libres y liberarlas. Esa capacidad de iterar por direcciones es lo que permite aprender algoritmos —para copiar una secuencia hay que leer la posición $n$, escribirla, avanzar a $n+1$, repetir— pero es innecesaria, y hasta estorbosa, para emparejar una pregunta con un hecho de texto.

Ambas comparten el ADN —memoria explícita, separada de los pesos, y diferenciable de punta a punta— pero una mira al *razonamiento lingüístico* y la otra a la *computación algorítmica*. Curiosamente, los dos papers fundadores se subieron a arXiv casi simultáneamente en octubre de 2014, sin conocerse: Memory Networks escala a millones de frases de lenguaje real, la NTM a un centenar de celdas para problemas con solución algorítmica conocida. La Clase 30 se construye sobre la línea Weston —es la que aparece en las slides— pero ningún tratamiento del tema queda completo sin la línea Graves, porque es la que define con mayor pureza qué significa dotar a una red de una memoria "tipo computador".

---

## 4. La línea Memory Networks

Es la estirpe que la Clase 30 recorre en detalle, una evolución de cuatro modelos donde cada uno resuelve la limitación del anterior.

**Memory Networks** ([Weston et al., 2014](/papers/memory-networks-weston-2014)) es el marco fundacional: una memoria de slots más cuatro componentes entrenables. **I** (*input feature map*) convierte la entrada en representación interna; **G** (*generalization*) **escribe** —actualiza las memorias dado el nuevo input—; **O** (*output feature map*) **lee** —encuentra las memorias de soporte relevantes—; y **R** (*response*) decodifica la salida al formato final (una respuesta textual). En su versión básica, G simplemente almacena cada frase en el siguiente slot libre, y O recupera la memoria de mayor puntaje; para razonamiento multi-paso busca una **segunda** memoria condicionada a la primera —los llamados *hops*—. El ejemplo canónico: ante "Where is the milk now?", el primer hop recupera "Joe left the milk", el segundo "Joe travelled to the office", y R responde "office". Su limitación decisiva es la **supervisión fuerte**: el entrenamiento necesita que los datos etiqueten *cuáles* frases son las de soporte de cada pregunta, lo que no escala a corpus reales que solo traen pares pregunta-respuesta.

**End-to-End Memory Networks** ([Sukhbaatar et al., 2015](/papers/e2e-memnn-sukhbaatar-2015)), o MemN2N, resuelve esa limitación reemplazando la selección dura (`arg max`) por una **atención softmax** continua sobre toda la memoria. Como la cadena de entrada a salida es ahora suave, los gradientes se retropropagan a través de los accesos a memoria, y el modelo **aprende solo a atender el hecho correcto entre distractores**, sin etiquetas de soporte. Apila **múltiples hops** (típicamente 3), cada uno refinando el estado interno, y demuestra empíricamente que más hops mejoran el desempeño. Es el modelo entrenable extremo a extremo que vuelve aplicable la idea a escenarios realistas.

**Key-Value Memory Networks** ([Miller et al., 2016](/papers/key-value-memnn-miller-2016)) generaliza MemN2N partiendo cada slot en un par **(clave, valor)**: la **clave** se usa para *direccionar* (emparejar con la pregunta) y el **valor** para *responder* (devolver el contenido). La idea —que la representación con la que buscas no tiene por qué ser la que devuelves— habilita codificar conocimiento heterogéneo en la misma maquinaria: el caso canónico es la ventana de texto, donde la clave es la ventana entera de palabras (más probable de emparejar con la pregunta) y el valor es solo la entidad central (más probable de ser la respuesta). Fijando clave = valor se recupera exactamente MemN2N. Su aplicación es **leer documentos directamente**, midiendo cuánto cae el QA al pasar de una base de conocimiento perfecta a texto crudo.

**Recurrent Entity Networks** ([Henaff et al., 2017](/papers/entity-networks-henaff-2017)), o EntNet, da el salto de "memoria como archivo que se consulta" a "memoria como **estado del mundo** que se mantiene vivo". Dedica **un slot por entidad** (una persona, un objeto, un lugar), cada uno con una clave fija que lo identifica y un contenido dinámico con sus atributos. Al leer cada frase, un **gate** independiente por slot decide qué entidades actualizar —combinando un término de *contenido* y uno de *localización*— y los slots irrelevantes quedan intactos. El procesamiento es **online y en una sola pasada**: razona mientras lee, no solo cuando le preguntan. Fue el **primer modelo en resolver las 20 tareas bAbI** (error medio 0,5 %), y su interpretabilidad permite leer directamente, slot por slot, dónde está cada entidad al final de la historia.

---

## 5. La línea de memoria diferenciable tipo computador

La otra estirpe nace en DeepMind con una ambición distinta: que una red **aprenda algoritmos** acoplándola a una memoria que se comporte como la RAM de un computador.

**Neural Turing Machine** ([Graves et al., 2014](/papers/ntm-graves-2014)) es el modelo seminal de la memoria externa diferenciable. Acopla un **controlador** (típicamente LSTM) a un **banco de memoria** —una matriz $N \times M$— mediante **cabezas** de lectura y escritura, por analogía con las cabezas de la máquina de Turing. La clave del invento es que las operaciones son *borrosas*: cada cabeza interactúa en mayor o menor grado con *todas* las ubicaciones, ponderadas por un vector de pesos normalizado que define un foco atencional continuo y, por tanto, diferenciable. El direccionamiento combina dos mecanismos: **por contenido** (comparar una clave emitida por el controlador con cada fila de memoria) y **por ubicación** (desplazar el foco a posiciones contiguas, para iterar sobre la cinta). La escritura imita las compuertas LSTM: un paso de **borrado** seguido de uno de **adición**. Entrenada solo con ejemplos, la NTM **infiere algoritmos** —copiar, recall asociativo, ordenar— que generalizan a secuencias mucho más largas que las de entrenamiento, signo de que aprendió un procedimiento y no una tabla.

**Differentiable Neural Computer** ([Graves et al., 2016](/papers/dnc-graves-2016)), o DNC, es el sucesor directo de la NTM —publicado en *Nature*— que corrige tres defectos concretos del predecesor. Añade **asignación dinámica de memoria** (un vector de uso y *free gates* que entregan una ubicación libre a la vez y permiten liberar memoria, resolviendo el solapamiento de bloques), **enlaces temporales** (una *temporal link matrix* que registra el orden en que se hicieron las escrituras, recuperable aunque la cabeza salte por la memoria) y **modos de lectura** mejorados (contenido más recorrido temporal hacia adelante y hacia atrás). Con esto el DNC navega **estructuras de datos** —grafos, árboles genealógicos, mapas de metro— respondiendo preguntas que exigen seguir enlaces, algo que su memoria como variables direccionables habilita naturalmente.

Vale la pena detenerse en *cómo* la NTM produce el foco de atención, porque ilumina la diferencia con la línea Weston. El **direccionamiento por contenido** funciona como en Memory Networks: el controlador emite una clave $k_t$, se mide su similitud coseno con cada fila de memoria y un softmax con un parámetro de **fuerza de clave** $\beta_t$ —que afila o difumina el foco— produce los pesos. Pero a eso la NTM le suma un **direccionamiento por ubicación**: una compuerta de interpolación mezcla el foco actual con el del paso anterior, un **desplazamiento circular** (convolución con un kernel aprendido) corre el foco a posiciones vecinas, y un paso de *sharpening* lo reconcentra. Esa segunda vía es la que no existe en la línea Weston, y es justo la que habilita recorrer la memoria como una cinta para ejecutar bucles. La **escritura** imita las compuertas de la LSTM: primero un vector de **borrado** $e_t$ apaga selectivamente componentes de cada fila ($\tilde{M}_t(i) \leftarrow M_{t-1}(i)\,[1 - w_t(i)\,e_t]$) y luego un vector de **adición** $a_t$ inyecta contenido nuevo ($M_t(i) \leftarrow \tilde{M}_t(i) + w_t(i)\,a_t$), dando control fino sobre qué se modifica y qué se conserva.

El detalle mecánico completo de la NTM y de las **Memory-Augmented Neural Networks** (MANN) para aprendizaje *one-shot* —cómo el direccionamiento por contenido permite a una red memorizar un ejemplo visto una sola vez y recuperarlo después, encadenando memoria externa con meta-aprendizaje— está desarrollado en el fundamento transversal de [redes con memoria aumentada](/fundamentos/memory-augmented-networks), que conviene leer en paralelo a esta sección.

---

## 6. El mecanismo común: la lectura como atención suave

Bajo la diversidad de modelos late **un mismo mecanismo**, y reconocerlo es la idea más rentable de todo el tema. Leer de la memoria es **direccionamiento por contenido**: se toma una consulta, se mide su similitud con cada slot, se normaliza con un softmax para obtener una distribución de pesos, y se devuelve la **suma ponderada** de los contenidos. En la forma más limpia, la de End-to-End Memory Networks, la consulta embebida $u$ se compara con cada memoria de entrada $m_i$:

$$
p_i = \mathrm{Softmax}(u^\top m_i), \qquad
o = \sum_i p_i\, c_i,
$$

donde $p$ es el vector de atención sobre los slots y $o$ es la lectura: una combinación convexa de los vectores de salida $c_i$. Toda lectura de memoria —el hop de O en Memory Networks, el `arg max` suavizado de KV-MemNN, el módulo de salida del EntNet (que es, literalmente, una MemN2N de un solo hop con una no-linealidad extra), la lectura por contenido de la NTM y del DNC— es una variación de esta misma ecuación: **similitud, softmax, suma ponderada**. La diferencia entre la selección *dura* del Memory Network original (un `arg max` que toma un único slot, no diferenciable) y esta selección *suave* (un softmax que promedia todos los slots, diferenciable) es exactamente lo que permitió entrenar estos modelos extremo a extremo: al reemplazar el máximo por una distribución continua, el gradiente fluye hacia atrás a través del acceso a memoria.

Aquí está la conexión más profunda del curso. **La self-attention de los Transformers ES lectura de memoria.** El producto interno $u^\top m_i$ seguido de softmax es el ancestro directo del *scaled dot-product attention*: el estado interno $u$ juega el papel de la *query*, las memorias $m_i$ el de las *keys*, los vectores de salida $c_i$ el de los *values*, y la suma ponderada $o = \sum_i p_i c_i$ es exactamente la salida de una capa de atención. Apilar capas de self-attention es lo que MemN2N hacía como **múltiples hops**, donde la "memoria" pasa a ser las representaciones de todos los tokens de la secuencia. Más aún, la separación clave/valor que KV-MemNN introdujo prefigura la distinción *key*/*value* de los Transformers. End-to-End Memory Networks (2015) es, así, un **precursor directo** del mecanismo que dos años después dominaría el NLP: el detalle de cómo funciona esa lectura como atención está en el fundamento de [self-attention](/fundamentos/self-attention).

---

## 7. El benchmark bAbI

Para medir si un modelo realmente *razona* sobre texto —y no solo memoriza correlaciones superficiales— hizo falta un banco de pruebas. Ese es **bAbI** ([Weston et al., 2015](/papers/babi-weston-2015)), un conjunto de **20 tareas** de razonamiento sintético, cada una aislando una habilidad: uno, dos o tres hechos de soporte, relaciones de argumentos, preguntas sí/no, conteo, listas y conjuntos, negación, coreferencia, deducción, inducción, razonamiento posicional y temporal, *path finding*, motivación del agente, etc. Cada tarea consiste en una historia de afirmaciones seguida de una pregunta de respuesta breve, donde **solo un subconjunto** de las frases es relevante y el resto son distractores. La meta declarada es exigente: que **un solo modelo** resuelva las 20 tareas, demostrando razonamiento general en vez de trucos por tarea.

bAbI nació de la simulación de mundos del paper original de Memory Networks y se volvió el termómetro estándar del campo. Permite ordenar a los modelos con nitidez: las RNN/LSTM colapsan cuando la frase relevante está lejos en el tiempo; MemN2N falla 3 de 20; el DNC, 2; y el EntNet fue el primero en resolver las 20. Para el QA sobre **documentos reales** (no sintéticos), Key-Value Memory Networks introdujo **WikiMovies**: ~100 000 preguntas sobre películas respondibles desde tres fuentes que codifican el mismo conocimiento —una base de conocimiento anotada, una extraída automáticamente y los documentos de Wikipedia crudos—, diseñado para medir exactamente cuánto se pierde al pasar de una KB perfecta a texto libre.

---

## 8. Ventajas, limitaciones y legado

Las **ventajas** de la memoria explícita son las que motivaron todo el programa. Es **interpretable**: se puede inspeccionar qué guarda cada slot —el EntNet permite leer la ubicación de cada entidad al final de la historia—. Es **editable**: agregar un hecho es escribir un slot, sin reentrenar la red ni arriesgar el resto del conocimiento; de hecho, las Memory Networks escriben información nueva *en tiempo de inferencia*, algo imposible para los pesos de una RNN. Y es **separable**: el conocimiento (slots) vive aparte del razonamiento (pesos del controlador), de modo que el mismo programa opera sobre bases de conocimiento distintas.

Las **limitaciones** también son reales. La atención softmax sobre toda la memoria **no escala** trivialmente: ponderar millones de slots en cada paso es prohibitivo, lo que obligó a introducir *hashing* por solapamiento de palabras y recuperación previa para restringir los candidatos a un subconjunto manejable —exactamente el cuello de botella que en los Transformers reaparece como el costo cuadrático de la atención—. La supervisión fuerte original era impráctica (resuelta por la versión end-to-end). La **eficiencia de muestras** sigue siendo un punto débil: el EntNet resuelve las 20 tareas bAbI con 10 000 ejemplos por tarea, pero se degrada drásticamente con solo 1 000. Y los benchmarks sintéticos como bAbI, por su regularidad de plantillas, no capturan la variedad del lenguaje abierto: que un modelo resuelva las 20 tareas no garantiza que razone sobre texto real arbitrario.

El **legado** es doble y enorme. Por un lado, la mecánica de "atención por producto interno + softmax + lectura ponderada, repetida en varios pasos" es exactamente el núcleo de la atención de los **Transformers**: las redes de memoria son uno de los eslabones que conectan la atención de RNNsearch con la era que domina el NLP moderno. Por otro lado, la idea de una memoria externa de la que el modelo recupera lo que necesita revive hoy en la **generación aumentada por recuperación** (RAG): un LLM consulta una base de conocimiento externa —típicamente un almacén vectorial donde se indexa por un *embedding* (clave) y se devuelve el pasaje original (valor)— para fundamentar sus respuestas en hechos editables y actualizables sin reentrenar. RAG es, conceptualmente, la **memoria externa moderna**, y hereda directamente la lección de KV-MemNN: desacoplar la representación con la que buscas de la que devuelves.

---

## 9. Conexión con el curso y resumen

La [Clase 30](/clases/clase-30) construye su narrativa sobre el arco completo de este fundamento. Parte motivando por qué la memoria *implícita* de RNN/LSTM no basta (sección 1), presenta la línea Memory Networks como la implementación canónica de la memoria *explícita* —MemNN, End-to-End, Key-Value, Entity Networks— y cierra con el ejemplo de rastrear entidades de una historia que es la esencia del EntNet. La otra estirpe, la memoria diferenciable tipo computador de la NTM y el DNC, completa el panorama y conecta el tema con el aprendizaje de algoritmos.

El hilo conductor que conviene retener es uno solo: **separar el conocimiento del programa, hacer la memoria diferenciable, y leerla como atención suave**. Esa receta —nacida para QA sobre texto y para aprender algoritmos— resultó ser la misma que, llevada al extremo, define los Transformers, y la misma que hoy estructura los sistemas de recuperación que dan a los grandes modelos de lenguaje una memoria externa editable. Visto desde 2024, estos modelos de 2014–2017 fueron un experimento conceptual cuyas dos lecciones —la lectura como atención y la memoria desacoplada del cómputo— se separaron y triunfaron por caminos distintos: la primera se volvió la arquitectura dominante (Transformers), la segunda el patrón de ingeniería dominante (RAG, almacenes vectoriales, agentes con memoria persistente). Entender las redes con memoria externa es, por tanto, entender de dónde viene la atención y hacia dónde va la memoria de los sistemas de IA contemporáneos.

---

## Para profundizar

- [Memory Networks (Weston et al., 2014)](/papers/memory-networks-weston-2014) — el marco fundacional con los componentes I/G/O/R y la supervisión fuerte de los hops.
- [End-to-End Memory Networks (Sukhbaatar et al., 2015)](/papers/e2e-memnn-sukhbaatar-2015) — atención softmax diferenciable, múltiples hops, entrenable sin etiquetas de soporte.
- [Key-Value Memory Networks (Miller et al., 2016)](/papers/key-value-memnn-miller-2016) — clave para direccionar, valor para responder; leer documentos directamente; dataset WikiMovies.
- [Recurrent Entity Networks (Henaff et al., 2017)](/papers/entity-networks-henaff-2017) — un slot por entidad, gate de actualización, estado del mundo; primero en resolver las 20 tareas bAbI.
- [Neural Turing Machine (Graves et al., 2014)](/papers/ntm-graves-2014) — controlador + memoria, direccionamiento por contenido y ubicación, aprende algoritmos.
- [Differentiable Neural Computer (Graves et al., 2016)](/papers/dnc-graves-2016) — asignación dinámica, enlaces temporales, razonamiento sobre grafos.
- [bAbI (Weston et al., 2015)](/papers/babi-weston-2015) — las 20 tareas de razonamiento que estandarizaron el benchmark.

**Fundamentos relacionados:** [Redes con Memoria Aumentada](/fundamentos/memory-augmented-networks) · [Self-Attention](/fundamentos/self-attention) · [Clase 30](/clases/clase-30)
