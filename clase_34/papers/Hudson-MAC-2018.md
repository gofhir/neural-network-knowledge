# Compositional Attention Networks for Machine Reasoning (red MAC) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Compositional Attention Networks for Machine Reasoning*.
- **Autores:** Drew A. Hudson y Christopher D. Manning, ambos del Department of Computer Science, **Stanford University**.
- **Venue:** *International Conference on Learning Representations (ICLR 2018)*, publicado como conference paper.
- **Año:** 2018. **Preprint:** arXiv:1803.03067v2 (24 abr 2018). Implementación en TensorFlow disponible en `github.com/stanfordnlp/mac-network`.
- **Linaje:** se inscribe en la línea de trabajo sobre *visual question answering* (VQA) y razonamiento visual estructurado que arranca con el dataset CLEVR (Johnson et al., 2017a) y con las *neural module networks* de Andreas et al. (2016), y bebe explícitamente de la investigación en memoria y atención neuronal (Dynamic Memory Networks de Kumar y Xiong; Neural Turing Machines y el Differentiable Neural Computer de Graves et al.).

El paper presenta la **red MAC** (Memory, Attention and Composition), una arquitectura neuronal **totalmente diferenciable** diseñada para facilitar un razonamiento **explícito y expresivo**. Su tesis central es que se puede dotar a una red profunda de una **estructura que favorezca el razonamiento composicional** sin renunciar a la diferenciabilidad end-to-end y sin recurrir a supervisión de programa. MAC lo consigue descomponiendo cada problema en una **secuencia de $p$ pasos de razonamiento basados en atención**, cada uno ejecutado por una celda recurrente MAC que mantiene una separación estricta entre **control** (qué operación hacer, derivada de la pregunta) y **memoria** (el resultado intermedio acumulado a partir de la imagen).

Sobre el dataset **CLEVR** de razonamiento visual, MAC alcanza una precisión de **98.9%**, un nuevo estado del arte que **más que reduce a la mitad la tasa de error** del mejor modelo previo (FiLM, 97.7%). Más importante aún para la práctica, MAC es **eficiente en cómputo y en datos**: requiere **5 veces menos datos** que los modelos existentes para lograr resultados fuertes, y con solo el 10% de CLEVR (70k ejemplos) alcanza 85.5% mientras los demás modelos fracasan (49.0%–54.9%). También lidera en **CLEVR-Humans** (preguntas escritas por personas), demostrando robustez ante la variación lingüística.

Para la **Clase 34 (Razonamiento)** este paper es la encarnación paradigmática de la estrategia "dar a la red una estructura que favorezca el razonamiento composicional" que se discute en la sección "Intentando aumentar a DL". MAC se sitúa deliberadamente en un punto medio entre la **atención monolítica** (redes que fusionan pregunta e imagen en un solo vector y son cajas negras) y la **ejecución de programas** (module networks que ensamblan módulos especializados según un programa provisto como supervisión fuerte). MAC toma lo mejor de ambos: la versatilidad y entrenabilidad por retropropagación de las redes neuronales, y la estructura secuencial e interpretable del razonamiento simbólico.

## 2. Contexto: VQA, el desafío del razonamiento composicional y CLEVR

El *visual question answering* (VQA) es una tarea multimodal que exige responder preguntas en lenguaje natural sobre imágenes. El problema, señalado por Agrawal et al. (2016), es que la primera generación de modelos VQA exitosos adquiría solo una **comprensión superficial** tanto de la imagen como de la pregunta: **explotaban sesgos del dataset** en lugar de capturar un proceso de percepción y razonamiento genuino. Un modelo puede responder "¿de qué color es el plátano?" correctamente sin mirar la imagen, simplemente porque los plátanos suelen ser amarillos. Esto es lo que Sturm (2014) llama, metafóricamente, detectar "un caballo" (un modelo que parece resolver la tarea pero se apoya en pistas espurias).

**CLEVR** (Johnson et al., 2017a) se creó precisamente para atacar este problema. Es un dataset **diagnóstico**: imágenes renderizadas de objetos 3D (formas, materiales, colores y tamaños diversos) acompañadas de preguntas composicionales de múltiples pasos, generadas por máquina. Las preguntas están diseñadas para ser **insesgadas** y para exigir una batería de habilidades de razonamiento difíciles —relaciones transitivas y lógicas, conteo, comparaciones— sin dejar atajos que permitan esquivar el razonamiento. El ejemplo canónico ("¿el bloque frente al cilindro amarillo pequeño y la cosa pequeña a la derecha del objeto verde grande brillante tienen el mismo color?") obliga a resolver referencias indirectas encadenadas. Cada pregunta viene además con un **programa funcional de estructura de árbol** que especifica las operaciones de razonamiento necesarias para computar la respuesta — un recurso que algunos modelos usan como supervisión y que MAC deliberadamente ignora.

El diagnóstico teórico del paper es contundente: las arquitecturas de deep learning **suelen tener dificultades en tareas de naturaleza composicional y estructurada** (Garnelo et al., 2016; Lake et al., 2017). La mayoría de las redes neuronales son "grandes motores de correlación" que se aferran a cualquier patrón estadístico —potencialmente espurio— que les permita modelar los datos observados. La profundidad, el tamaño y la naturaleza estadística que las hace robustas al ruido limitan al mismo tiempo su interpretabilidad y su capacidad de realizar inferencias explícitas y sólidas, que son vitales para resolver problemas. El desafío, entonces, es diseñar una red que realice el razonamiento **estructurado e iterativo** que exige la resolución de problemas complejos.

## 3. Contribución central

Frente a este desafío, existían dos respuestas previas, y ambas tenían costos:

1. **Estructuras simbólicas / module networks** (Andreas et al., 2016; Johnson et al., 2017b): adoptan árboles de expresión al estilo de lenguajes de programación, componiendo módulos neuronales de una colección predefinida y fija. Su debilidad: **dependen de representaciones estructuradas provistas externamente** —programas funcionales, parsers frágiles hechos a mano o demostraciones de expertos— y requieren esquemas de entrenamiento multi-etapa complejos con aprendizaje por refuerzo. La rigidez de su estructura y el inventario de módulos especializados por operación **socavan su robustez y su capacidad de generalización**.

2. **CNNs aumentadas** (Relation Networks de Santoro et al., 2017; FiLM de Perez et al., 2017): complementan un stack estándar de CNNs con componentes que ayudan al razonamiento, sin necesitar los programas. Más entrenables, pero **monolíticas**: fusionan pregunta e imagen y pierden transparencia y la estructura explícita del razonamiento.

La contribución de MAC es **una tercera vía**. Los autores buscan un equilibrio entre la versatilidad y robustez de los enfoques neuronales end-to-end, por un lado, y la necesidad de un razonamiento más explícito y estructurado, por el otro. Proponen realizar el razonamiento estructurado **encadenando una nueva celda recurrente MAC**, diseñada para capturar el funcionamiento interno de un paso de razonamiento elemental pero de propósito general, inspirándose en los principios de diseño de las **arquitecturas de computadores**. La celda separa explícitamente **memoria de control**, ambas representadas recurrentemente, y consta de tres unidades operativas que trabajan en conjunto: **control**, **lectura** y **escritura**.

Este diseño universal de la celda MAC funciona como un **prior estructural** que induce a la red a resolver problemas descomponiéndolos en una secuencia de operaciones de razonamiento basadas en atención, inferidas directamente de los datos **sin recurrir a supervisión fuerte**. Con conexiones de auto-atención entre celdas, la red MAC puede representar de forma blanda **grafos de razonamiento acíclicos arbitrariamente complejos** (DAGs), manteniendo a la vez una estructura físicamente secuencial y diferenciabilidad end-to-end, entrenable por simple retropropagación. Aunque cada celda tiene un rango limitado de comportamientos continuos posibles, orientados a una operación simple, al encadenarlas el sistema completo se vuelve expresivo y potente — realizando la visión de un modelo **algebraico y composicional** de la inferencia propuesta por Bottou (2014).

## 4. Método: la arquitectura MAC

Una red MAC es una arquitectura diferenciable end-to-end para un proceso de razonamiento explícito de múltiples pasos, construida encadenando $p$ celdas recurrentes MAC, cada una responsable de un paso de razonamiento. Dada una **base de conocimiento** $K$ (para VQA, una imagen) y una **descripción de la tarea** $q$ (una pregunta), el modelo infiere una descomposición en $p$ operaciones que interactúan con la base de conocimiento, agregando y manipulando información iterativamente. Tiene tres componentes: (1) una unidad de entrada, (2) la red recurrente central de $p$ celdas MAC, y (3) una unidad de salida.

### 4.1. Unidad de entrada

La unidad de entrada transforma las entradas crudas en representaciones vectoriales distribuidas.

- **Pregunta.** La cadena de longitud $S$ se convierte en *embeddings* de palabra aprendidos y se procesa con un **biLSTM** de dimensión $d$, produciendo: (1) las **palabras contextuales** $cw_1, \dots, cw_S$, que representan cada palabra en el contexto de la pregunta, y (2) la **representación de la pregunta** $q = [\overleftarrow{cw_1}, \overrightarrow{cw_S}]$, la concatenación de los estados ocultos finales de las pasadas hacia atrás y hacia adelante. Además, para cada paso $i = 1, \dots, p$, la pregunta se transforma linealmente en un vector **consciente de la posición** $q_i = W_i^{d \times 2d} q + b_i^d$, que representa los aspectos de la pregunta relevantes para el $i$-ésimo paso de razonamiento.

- **Imagen.** Se procesa primero con un extractor de características fijo pre-entrenado en ImageNet (características `conv4` de ResNet101), siguiendo el trabajo previo en CLEVR. El tensor resultante pasa por dos capas CNN con $d$ canales de salida, obteniendo la representación final, la **base de conocimiento** $K^{H \times W \times d} = \{k_{h,w}^d\}$, donde $H = W = 14$ son alto y ancho de la imagen procesada, correspondiendo a cada una de sus regiones.

### 4.2. La celda MAC: dualidad control–memoria

La celda MAC es una celda recurrente que captura la noción de una operación de razonamiento **atómica y universal**. Para cada paso $i$, la celda mantiene dos estados ocultos duales de dimensión $d$: **control** $c_i$ y **memoria** $m_i$, inicializados a parámetros aprendidos $c_0$ y $m_0$.

- El **control** $c_i$ representa la operación de razonamiento que la celda debe realizar en el paso $i$, enfocándose selectivamente en algún aspecto de la pregunta. Se representa como un **promedio ponderado por atención blanda** de las palabras de la pregunta $cw_s$.
- La **memoria** $m_i$ contiene el resultado intermedio del proceso de razonamiento hasta el paso $i$, computado recurrentemente integrando el estado previo $m_{i-1}$ con nueva información $r_i$ recuperada de la imagen. Análogamente al control, $r_i$ es un promedio ponderado sobre las regiones $\{k_{h,w}\}$.

La idea clave —tomada de la organización de los computadores— es que las tres unidades **imponen una interfaz que regula la interacción entre control y memoria**. El estado de control, que es función de la pregunta, guía la integración del contenido de la imagen en la memoria **solo por medios indirectos**: mapas de atención blanda y compuertas sigmoidales. Es decir, la interacción entre las dos modalidades (visual y textual, base de conocimiento y consulta) está **mediada exclusivamente por distribuciones de probabilidad**. Esto contrasta frontalmente con los enfoques comunes que fusionan pregunta e imagen en un mismo espacio vectorial mediante combinaciones lineales, multiplicación o concatenación. Mantener una **separación estricta entre los espacios representacionales** de pregunta e imagen, que solo pueden interactuar a través de distribuciones discretas interpretables, mejora enormemente la generalización y la transparencia — como confirman las ablaciones.

### 4.3. La unidad de control (CU)

La unidad de control determina la operación de razonamiento del paso $i$, atendiendo a alguna parte de la pregunta y actualizando el estado de control $c_i$. Recibe las palabras contextuales $cw_s$, la representación posicional $q_i$ y el control previo $c_{i-1}$, y opera en dos etapas:

$$cq_i = W^{d \times 2d} [c_{i-1}, q_i] + b^d$$
$$ca_{i,s} = W^{1 \times d} (cq_i \odot cw_s) + b^1$$
$$cv_{i,s} = \mathrm{softmax}(ca_{i,s})$$
$$c_i = \sum_{s=1}^{S} cv_{i,s} \cdot cw_s$$

Primero se combina $q_i$ con $c_{i-1}$ en $cq_i$, tomando en cuenta tanto la pregunta sesgada hacia el paso $i$ como la operación previa (esto permite que la decisión del paso $i$ **se apoye en la operación anterior**). Luego se proyecta $cq_i$ sobre el espacio de las palabras: se mide la similitud entre $cq_i$ y cada palabra $cw_s$, se pasa por un softmax (obteniendo una distribución de atención sobre las palabras) y se suman las palabras según esa distribución para producir $c_i$. Este "anclaje" del control de vuelta en las palabras de la pregunta actúa como **regularización** —restringe el espacio de operaciones válidas— y mejora la transparencia, pues podemos interpretar el comportamiento de la celda según las palabras que atiende.

### 4.4. La unidad de lectura (RU)

La unidad de lectura inspecciona la base de conocimiento (la imagen) y recupera la información $r_i$ necesaria para la operación $c_i$, mediante un proceso de atención de **dos etapas** que considera tanto el control $c_i$ como la memoria previa $m_{i-1}$:

$$I_{i,h,w} = [W_m^{d \times d} m_{i-1} + b_m^d] \odot [W_k^{d \times d} k_{h,w} + b_k^d]$$
$$I'_{i,h,w} = W^{d \times 2d} [I_{i,h,w}, k_{h,w}] + b^d$$
$$ra_{i,h,w} = W^{d \times d} (c_i \odot I'_{i,h,w}) + b^d$$
$$rv_{i,h,w} = \mathrm{softmax}(ra_{i,h,w})$$
$$r_i = \sum_{h,w=1,1}^{H,W} rv_{i,h,w} \cdot k_{h,w}$$

Primero se computa la **interacción directa** entre cada elemento $k_{h,w}$ y la memoria $m_{i-1}$, resultando en $I_{i,h,w}$: esto mide la relevancia del elemento respecto del resultado intermedio previo, habilitando **razonamiento transitivo** (considerar contenido que ahora parece importante a la luz de lo computado antes). Luego se concatena $k_{h,w}$ a $I_{i,h,w}$, para considerar también información nueva no directamente relacionada con el resultado previo (necesario, por ejemplo, para un OR lógico o para unir conjuntos y contar). Finalmente se mide la similitud con el control $c_i$, se aplica softmax y se computa el promedio ponderado $r_i$.

El ejemplo del paper ilustra esta transitividad: ante una pregunta que refiere indirectamente a un "cilindro morado", el modelo procede por pasos — primero atiende al "bloque azul pequeño" (actualizando $m_1$), luego el control decide buscar "la esfera enfrente" (guardando eso en $c_2$), y la unidad de lectura, combinando $m_1$ y $c_2$, encuentra la esfera correcta, y así traversa la cadena de referencias hasta el cilindro morado.

### 4.5. La unidad de escritura (WU)

La unidad de escritura computa el resultado intermedio $m_i$ integrando la información recuperada $r_i$ con el resultado previo $m_{i-1}$, guiada por $c_i$. Procede en tres pasos, el primero obligatorio y los otros dos opcionales:

$$m_i^{\text{info}} = W^{d \times 2d} [r_i, m_{i-1}] + b^d \tag{w1}$$

**1. Integración básica** (obligatoria): una transformación lineal combina $r_i$ y $m_{i-1}$.

**2. Auto-atención (opcional):** para soportar razonamiento no secuencial (árboles o grafos), cada celda puede considerar **todos** los resultados intermedios previos, no solo el inmediato:
$$sa_{ij} = \mathrm{softmax}\big(W^{1 \times d}(c_i \odot c_j) + b^1\big), \quad m_i^{sa} = \sum_{j=1}^{i-1} sa_{ij} \cdot m_j$$
$$m_i' = W_s^{d \times d} m_i^{sa} + W_p^{d \times d} m_i^{\text{info}} + b^d$$
Nótese que la atención se computa sobre los **estados de control** pero se usa para promediar sobre los **estados de memoria** — un mecanismo que, como observan los autores, **recuerda a las Key-Value Memory Networks** (Miller et al., 2016): el control juega el papel de la clave y la memoria el del valor.

**3. Compuerta de memoria (opcional):** una compuerta sigmoidal interpola entre la memoria previa y el nuevo candidato, condicionada en $c_i$:
$$c_i' = W^{1 \times d} c_i + b^1, \quad m_i = \sigma(c_i')\, m_{i-1} + (1 - \sigma(c_i'))\, m_i'$$
Esto permite **ajustar dinámicamente la longitud efectiva** del razonamiento: la celda puede "saltarse" un paso si la pregunta es simple, pasando la memoria previa hacia adelante. Ambos mecanismos opcionales reducen dependencias de largo alcance; para CLEVR casi el mismo rendimiento se logra solo con el paso obligatorio.

### 4.6. Unidad de salida

Un clasificador softmax de 2 capas procesa la concatenación de la representación de la pregunta $q$ y la memoria final $m_p$, produciendo una distribución sobre las respuestas candidatas (para CLEVR, un conjunto fijo de 28 posibilidades). Incluir $q$ además de $m_p$ es importante: algunas preguntas refieren a aspectos que no tienen contraparte en la base de conocimiento.

## 5. Resultados

### 5.1. CLEVR: estado del arte

Entrenado sobre el dataset primario de 700k ejemplos, con $d = 512$ y $p = 12$ celdas, **sin usar los programas funcionales** como supervisión, MAC alcanza **98.94%** de precisión global, superando a todo el trabajo previo tanto en el total como en cada categoría de habilidad. Esto **más que reduce a la mitad la tasa de error** del mejor modelo anterior, FiLM (97.7%). La comparación con el rendimiento humano (92.6%) es notable: MAC lo supera holgadamente. Frente a las module networks con supervisión de programa, MAC (98.9%) supera incluso a PG+EE entrenado con 700k programas (96.9%), **sin usar ni un solo programa**.

Un punto especialmente relevante es el **conteo y la comparación numérica**, categorías notoriamente difíciles para los modelos VQA. MAC alcanza 97.1% en conteo (frente a ~94% de FiLM) y 99.5% en comparación de números, nuevamente casi reduciendo a la mitad el error. Los autores atribuyen esto a que la atención, a diferencia de las CNNs, favorece operaciones como el conteo que requieren **agregación global** de información a través de distintas regiones de la imagen.

### 5.2. CLEVR-Humans: robustez lingüística

CLEVR-Humans consiste en preguntas en lenguaje natural recolectadas por *crowdsourcing*, con vocabulario diverso, variaciones lingüísticas y habilidades de razonamiento más variadas. Como el set de entrenamiento es pequeño (18k), se hace *fine-tuning* de un modelo pre-entrenado en CLEVR. MAC logra estado del arte tanto antes como después del fine-tuning, alcanzando **81.5%**, superando al siguiente mejor modelo por **5.6 puntos**. La atención blanda sobre la pregunta permite al modelo enfocarse en las palabras críticas ignorando variaciones lingüísticas irrelevantes. El modelo responde correctamente incluso a habilidades no vistas en entrenamiento (obstrucciones, unicidad de objetos, distancias relativas, superlativos, conceptos nuevos).

### 5.3. Eficiencia computacional y de datos

Aquí está el segundo gran aporte. MAC **aprende mucho más rápido**: mientras Santoro et al. (Relation Networks) reportan 1.4 millones de iteraciones (~125 épocas) para alcanzar 95.5%, MAC logra precisión comparable **tras solo 3 épocas** — una reducción de 40× en la longitud del entrenamiento. Frente a FiLM (4 días, 80 épocas, 97.7%), MAC logra mayor precisión en 6 épocas, **9.5 horas en total** — una reducción de 10× en tiempo de entrenamiento.

La **eficiencia de datos** es aún más dramática. Sobre subconjuntos aleatorios de CLEVR:

- Con **50%** de los datos (350k): otros modelos rinden entre 70% y 93%, MAC alcanza 97.6%.
- Con **25%** (175k): otros entre 50% y 77%, MAC mantiene 94.3%.
- Con **10%** (70k): MAC es el **único que generaliza bien**, con 85.5%, mientras los demás fracasan (49.0%–54.9%, apenas por encima del baseline de 42.1% que predice la respuesta más frecuente por tipo de pregunta).

Los autores presentan la eficiencia de datos como la evidencia más fuerte del valor del **prior estructural**: la estructura de MAC guía a la red hacia las habilidades de razonamiento correctas incluso cuando los datos escasean, en lugar de dejarla memorizar correlaciones espurias.

### 5.4. Ablaciones

Los estudios de ablación confirman qué elementos importan:

- **Atención sobre las palabras de la pregunta:** usar la pregunta completa $q$ en vez del estado de control basado en atención causa una caída de 18.5% en precisión; no anclar el control en las palabras causa una ralentización de 6× en la convergencia. La descomposición de la pregunta en operaciones simples es clave.
- **Separación control–memoria:** un variante con un solo estado oculto que cumple ambos roles cae de 98.9% a 93.75% en el dataset completo y 20.2% en el subconjunto del 10%. La separación en dos vías duales —una que descompone la información lingüística, otra que reconstruye la visual— es una decisión de diseño fundamental.
- **Longitud de la red:** correlación positiva entre longitud y rendimiento, con mejoras significativas hasta $p = 8$. Esto distingue a MAC de otras arquitecturas multi-hop que se benefician de solo 2–3 iteraciones, y sugiere que MAC **usa efectivamente las celdas recurrentes** para razonamiento composicional.
- **Compartir pesos** entre las $p$ celdas resulta útil: la misma celda MAC adapta su comportamiento a la tarea según el contexto, en contraste con los módulos especializados de las module networks.

## 6. Limitaciones

- **Dominio sintético.** CLEVR es un banco de pruebas diagnóstico con objetos 3D renderizados y vocabulario controlado. Aunque los autores creen que la arquitectura será útil para otras tareas de razonamiento multi-paso (comprensión lectora, QA textual, VQA del mundo real), el paper no lo demuestra; queda como conjetura.
- **Errores residuales.** En CLEVR, la mayoría de los errores son de conteo *off-by-one* o por oclusiones fuertes de objetos. Curiosamente, el modelo tiende a **subestimar** cuando hay oclusión pesada, lo que sugiere que realiza un conteo "continuo" más que discreto — una señal de que no razona simbólicamente sino con aproximaciones blandas.
- **Habilidades no entrenadas.** En CLEVR-Humans, cerca de la mitad de los errores provienen de habilidades ausentes en CLEVR (propiedades físicas como sombras y reflejos, estabilidad, distancias relativas, cantidades relativas, comunalidades, preguntas negativas). El modelo también malinterpreta a veces palabras no vistas capturando semánticas plausibles pero incorrectas (interpretar "caramelo" como amarillo en vez de café). Los autores especulan que muchos de estos errores se deben al **tamaño pequeño** del dataset humano.
- **Número fijo de pasos.** La longitud $p$ es un hiperparámetro; aunque la compuerta de memoria permite acortar dinámicamente el razonamiento efectivo, el modelo no decide por sí mismo cuántos pasos totales necesita. No hay un mecanismo de "detención adaptativa" completo.

## 7. Conexión con la Clase 34 (Razonamiento) y con la línea de memoria externa

En la Clase 34, dentro de la sección "Intentando aumentar a DL", MAC ilustra la estrategia de **dar a la red una estructura que favorezca el razonamiento composicional**. Conviene presentarla contrastando las tres familias que compiten sobre CLEVR:

| Enfoque | Ejemplo | Estructura del razonamiento | Supervisión | Transparencia |
|---|---|---|---|---|
| Atención monolítica | Stacked Attention, FiLM, Relation Nets | Implícita, fusionada en un vector | Solo respuesta | Baja |
| Ejecución de programas | Neural Module Networks, PG+EE | Árbol de módulos especializados | **Programa funcional** | Media (pero rígida) |
| MAC | red MAC | Secuencia diferenciable de $p$ pasos de atención | **Solo respuesta** | Alta (mapas de atención) |

El aporte conceptual que el estudiante debe internalizar es que **la composicionalidad no exige supervisión simbólica**. Las module networks obtienen estructura a costa de necesitar programas, parsers frágiles y entrenamiento con RL en varias etapas; la atención monolítica obtiene entrenabilidad a costa de perder estructura e interpretabilidad. MAC demuestra que se puede tener **estructura composicional explícita, diferenciable y sin supervisión de programa**, imponiendo un **prior arquitectónico** (la celda con separación control–memoria, encadenada $p$ veces) en lugar de un prior de datos. La estructura no se ejecuta como un programa discreto; **emerge de forma blanda** de la secuencia de mapas de atención, y por eso sigue siendo entrenable por retropropagación. Este es exactamente el sentido de "aumentar deep learning con estructura" que vertebra la clase. La conexión con CLEVR-CoGenT es directa: los splits Condición A / Condición B miden si un modelo aprende **combinaciones forma–color composicionales** (entrenando con ciertas combinaciones y evaluando con otras no vistas), y la separación estricta entre el espacio de la pregunta y el de la imagen que impone MAC es precisamente el tipo de sesgo inductivo que se espera favorezca esa generalización composicional.

La conexión con la **línea de memoria externa** (ya presente en el site, clase-30) es explícita en el propio paper. MAC se inspira en la investigación sobre memoria y atención neuronal, y en particular en las **Neural Turing Machines** y el **Differentiable Neural Computer** de Graves et al. (2014; 2016). Al igual que en esos trabajos, MAC tiene un "controlador" que realiza operaciones de **lectura y escritura** sobre una memoria mediante atención blanda. Pero hay una diferencia arquitectónica clave que los autores subrayan: mientras las NTM/DNC leen y escriben iterativamente en **múltiples slots de un recurso de memoria compartido y global**, MAC emplea una **estructura de memoria recurrente** donde cada celda tiene su propio estado de memoria y **construye una nueva memoria sobre las anteriores**. Esto evita el problema de *content blurring* (difuminado de contenido) que puede surgir de múltiples escrituras globales, a la vez que soporta procesos de razonamiento complejos que interactúan progresivamente con las memorias e resultados intermedios previos. La auto-atención opcional de la unidad de escritura, que atiende sobre estados de memoria pasados usando claves de control, es además un eco directo de las **Key-Value Memory Networks** (Miller et al., 2016), también de la misma línea. Así, MAC puede leerse como el punto donde la tradición de la memoria externa diferenciable (Graves, Weston, Miller) se especializa y se domestica para la tarea concreta del razonamiento visual composicional: en vez de una máquina de Turing neuronal de propósito general, una celda de razonamiento compacta y repetible cuyo prior estructural la hace eficiente en datos y transparente.

## 8. Nota final: relevancia para salud

Para el razonamiento clínico sobre datos multimodales, el aporte más valioso de MAC no es su precisión sino su **arquitectura auditable**. Un sistema que integre imágenes (radiografías, histopatología, dermatoscopía) con texto (motivo de consulta, antecedentes) debe poder responder preguntas composicionales del tipo "¿la lesión adyacente al nódulo calcificado presenta el mismo patrón de realce que la lesión del lóbulo contralateral?", encadenando referencias transitivas exactamente como en CLEVR. La separación estricta control–memoria y los mapas de atención por paso permitirían, en principio, **exhibir la traza del razonamiento** —qué región de la imagen y qué términos del texto pesaron en cada paso intermedio—, un requisito prácticamente innegociable para la validación clínica, la trazabilidad regulatoria y la confianza del profesional, frente a la opacidad de los modelos monolíticos. Igualmente relevante es su **eficiencia de datos**: en medicina los datos etiquetados de alta calidad son escasos, y un prior estructural que generalice bien desde una fracción del dataset (como el 10% de CLEVR) es más deseable que un modelo que exige cientos de miles de ejemplos. La misma lógica composicional y auditable se traslada al *record linkage* y al *master patient index*: razonar de forma estructurada sobre atributos de identidad heterogéneos encadenando decisiones interpretables, en vez de fusionar todo en un score opaco, mejora tanto la generalización a combinaciones no vistas como la posibilidad de auditar por qué dos registros fueron considerados la misma persona.
