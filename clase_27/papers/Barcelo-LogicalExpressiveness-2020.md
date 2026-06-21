# The Logical Expressiveness of Graph Neural Networks — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *The Logical Expressiveness of Graph Neural Networks*.
- **Autores:** Pablo Barceló (IMC, PUC & IMFD Chile), Egor V. Kostylev (University of Oxford), Mikaël Monet (IMFD Chile), Jorge Pérez (DCC, UChile & IMFD Chile), Juan Reutter (DCC, PUC & IMFD Chile), Juan-Pablo Silva (DCC, UChile).
- **Venue:** *International Conference on Learning Representations* (ICLR 2020), conference paper.
- **Financiamiento:** Millennium Institute for Foundational Research on Data (IMFD Chile, imfd.cl).
- **Código y datos:** [github.com/juanpablos/GNN-logic](https://github.com/juanpablos/GNN-logic). Implementación en PyTorch Geometric (Fey & Lenssen, 2019).

**Conexión local.** Este es, literalmente, un paper "de la casa". Cinco de sus seis autores trabajan en Chile —entre la PUC, la Universidad de Chile y el Instituto Milenio Fundamentos de los Datos (IMFD)— y dos de ellos, **Pablo Barceló** y **Jorge Pérez**, son figuras centrales del grupo chileno de fundamentos teóricos de la IA. Jorge Pérez aparece citado en los créditos de la Clase 27 del curso ("The Logical Expressiveness of Graph Neural Networks", Tópicos Avanzados de IA 2019), de modo que el material de la clase no solo referencia el resultado sino que proviene del propio coautor. Para un estudiante en Chile esto tiene un valor doble: es teoría de primera línea publicada en el venue más competitivo del área (ICLR) y, a la vez, es producción local, demostrando que el aporte fundacional —no solo aplicado— a las redes neuronales de grafos salió de universidades chilenas.

**Tesis del paper.** La pregunta es de **expresividad**: ¿qué clasificadores de nodos puede o no puede expresar una GNN? Trabajos previos (Morris et al., 2019; Xu et al., 2019 — el paper de GIN) habían caracterizado el poder *discriminativo* de las GNN en términos del test de Weisfeiler-Lehman (WL) para isomorfismo de grafos. Pero esa caracterización combinatoria no responde qué *clasificadores booleanos* —funciones que asignan verdadero o falso a cada nodo— es capaz de capturar una GNN. Este paper aborda esa pregunta cambiando de lente: en vez de medir expresividad contra un algoritmo combinatorio (WL), la mide contra la **lógica de primer orden**. El resultado central es una caracterización lógica precisa: las GNN de message passing estándar (que el paper llama **AC-GNN**) capturan exactamente la **lógica modal graduada** (un fragmento estricto de FOC2), ni más ni menos; y para capturar todo FOC2 basta con añadir un componente global de lectura (**readout**), obteniendo las **ACR-GNN**. Es la contraparte declarativa y lógica del enfoque combinatorio de GIN.

## 2. Contexto: la pregunta de expresividad de las GNN y la conexión con WL

Las GNN (Merkwirth & Lengauer, 2005; Scarselli et al., 2009) se volvieron populares para datos estructurados —clasificación de moléculas, completación de grafos de conocimiento, ranking de páginas web— bajo una idea simple: las conexiones entre neuronas reflejan la estructura del dato de entrada, generalizando convolucionales y recurrentes. Pero, advierte el paper, sus propiedades teóricas no estaban bien comprendidas. El programa de investigación del paper es **conectar las GNN con formalismos lógicos conocidos**, porque eso permite entender el comportamiento *procedural* (paso a paso, capa a capa) de las GNN en términos del sabor *declarativo* de los lenguajes lógicos. La ganancia es transferible: si se demuestra que dos arquitecturas de GNN corresponden a dos lógicas, toda la teoría conocida sobre la relación entre esas lógicas (equivalencia, incomparabilidad de expresividad) se traslada gratis al mundo de las GNN.

**El punto de partida: WL y GIN.** Dos trabajos contemporáneos (Morris et al., 2019; Xu et al., 2019) habían establecido la conexión clave entre GNN y el **test de Weisfeiler-Lehman**. El test WL construye, de forma incremental, una coloración de los nodos: en cada ronda asigna a cada nodo un nuevo color que tiene correspondencia uno-a-uno con su color previo y con el *multiconjunto* de colores de sus vecinos. Decide isomorfismo comparando las coloraciones de dos grafos. La GNN básica —la que actualiza el vector de features de un nodo combinándolo con la agregación de los vectores de sus vecinos— es lo que el paper bautiza como **aggregate-combine GNN (AC-GNN)**. La observación central de aquellos trabajos:

- **Proposición 2.1 (Morris/Xu).** Si el test WL asigna el mismo color a dos nodos, entonces *toda* AC-GNN los clasifica igual (ambos verdaderos o ambos falsos). En otras palabras, la coloración de WL *refina* la de cualquier AC-GNN; WL es un techo superior para el poder discriminativo de las AC-GNN. Y existen AC-GNN que reproducen exactamente la coloración WL (rondas de WL = capas de una AC-GNN con funciones de agregación y combinación inyectivas), de modo que las AC-GNN pueden ser tan poderosas como WL para *distinguir* nodos.

**El hueco que el paper detecta.** Que WL refine la clasificación de una AC-GNN no implica que la AC-GNN pueda capturar *todo* clasificador refinado por WL. De hecho hay clasificadores muy simples refinados por WL que ninguna AC-GNN expresa. El ejemplo trivial: "asignar verdadero a todo nodo si y solo si el grafo tiene un nodo aislado". El poder *discriminativo* (distinguir dos nodos) y el poder *expresivo* (computar una función booleana concreta sobre los nodos) son cosas distintas, y la literatura previa solo había resuelto el primero. La pregunta abierta que el paper aborda: **¿qué clasificadores de nodos puede capturar una arquitectura como AC-GNN?**

## 3. Conexión entre GNN y lógica: clasificadores lógicos y FOC2

### 3.1. Clasificadores lógicos

El paper se restringe a **clasificadores lógicos**: fórmulas unarias (con exactamente una variable libre) expresables en lógica de primer orden (FO) sobre grafos no dirigidos donde cada vértice tiene un único color. Una fórmula ϕ(x) clasifica como verdadero a los nodos v tales que (G, v) ⊨ ϕ. El ejemplo del paper:

```
α(x) := Red(x) ∧ ∃y (E(x,y) ∧ Blue(y)) ∧ ∃z (E(x,z) ∧ Green(z))
```

que es verdadera exactamente en los nodos rojos que tienen al menos un vecino azul y al menos un vecino verde. **Definición de captura (3.1):** una GNN *captura* un clasificador lógico ϕ(x) si ambos coinciden sobre todo nodo de todo grafo posible.

### 3.2. La lógica FOC2

Comparar AC-GNN contra toda FO es demasiado: FO es demasiado poderosa. El paper elige **FOC2**, el fragmento de FO que (a) permite solo **dos variables** pero (b) añade **cuantificadores de conteo** ∃^{≥N}, que expresan "existen al menos N nodos que satisfacen ϕ" (Cai et al., 1992).

¿Por qué esta elección y no otra? Por una razón de fondo: reducir el número de variables reduce drásticamente la expresividad, pero los cuantificadores de conteo recuperan parte del poder. El paper lo ilustra con la fórmula:

```
β(x) := Red(x) ∧ ∃y (¬E(x,y) ∧ ∃z1 ∃z2 (E(y,z1) ∧ E(y,z2) ∧ z1≠z2 ∧ Blue(z1) ∧ Blue(z2)))
```

"x es rojo y hay otro nodo y no conectado a x que tiene al menos dos vecinos azules". β usa cuatro variables y no admite una equivalente con menos de tres. Pero con conteo se puede expresar con solo dos:

```
γ(x) := Red(x) ∧ ∃y (¬E(x,y) ∧ ∃^{≥2}x (E(y,x) ∧ Blue(x)))    (Ec. 4)
```

Así, **FOC2 es estrictamente menos expresiva que FO** (el conteo se puede emular en FO con más variables y desigualdades) pero **estrictamente más expresiva que FO2** (la FO de dos variables sin conteo, donde β no es expresable).

**La justificación profunda de FOC2** es un resultado clásico (Cai, Fürer & Immerman, 1992):

- **Proposición 3.2 (Cai et al.).** Para todo grafo G y nodos u, v: el test WL colorea u y v igual tras cualquier número de rondas *si y solo si* u y v son clasificados igual por todos los clasificadores FOC2.

Es decir, **WL y FOC2 son dos caras de la misma moneda discriminativa**. Combinado con la Proposición 2.1 (WL refina a AC-GNN), FOC2 es la vara natural para medir AC-GNN.

### 3.3. Por qué AC-GNN ≠ FOC2

Aquí está el giro fino del argumento, y la trampa que el paper evita. Uno podría encadenar las dos proposiciones —WL refina a AC-GNN, y WL ↔ FOC2— para concluir "toda AC-GNN captura todo FOC2". **Es falso.** Que dos nodos indistinguibles por WL sean indistinguibles por todo FOC2 y por toda AC-GNN no dice nada sobre si una fórmula FOC2 *concreta* puede *expresarse* como una AC-GNN. La indistinguibilidad es una relación de equivalencia; la captura es la realización de una función específica.

- **Proposición 3.3.** Existe un clasificador FOC2 que ninguna AC-GNN captura.

El testigo más simple es α(x) := Red(x) ∧ ∃y Green(x), "x es rojo y el grafo tiene algún nodo verde". La **intuición de la imposibilidad** (formalizada en el Apéndice A) es la **localidad**: una AC-GNN con L capas no puede mover información más allá de distancia L de cada nodo a lo largo de las aristas. En γ(x) (Ec. 4), el nodo rojo puede estar más lejos que L del nodo con vecinos azules; peor aún, ambos pueden estar en componentes conexas distintas, donde *ningún* número de capas alcanza. La demostración construye una cadena de L+2 nodos rojos: la AC-GNN debe etiquetar el primer nodo como falso, y como colorear de verde el último nodo (a distancia > L) no cambia nada visible para el primer nodo, sigue clasificándolo falso —contradiciendo que ahora la fórmula es verdadera. El argumento persiste incluso permitiendo un número arbitrario de capas, usando dos nodos desconectados.

Esto abre las dos preguntas que estructuran el resto del paper: (1) ¿qué fragmento de FOC2 *sí* capturan las AC-GNN? (2) ¿qué hay que añadirles para capturar todo FOC2?

## 4. El poder expresivo de las AC-GNN: lógica modal graduada

La respuesta a la primera pregunta es nítida y elegante: como el problema de las AC-GNN es la localidad, la lógica que capturan debe construirse con esa misma limitación de localidad. Esa lógica existe y es bien conocida: la **lógica modal graduada** (de Rijke, 2000), equivalente a la lógica de descripción **ALCQ** (Baader et al., 2003), fundamental en representación del conocimiento —el lenguaje de ontologías web OWL 2 se apoya en ALCQ.

**La idea: guardar todo con la arista.** La lógica modal graduada fuerza a que toda subfórmula esté *guardada* por el predicado de arista E. No se puede expresar ∃y ϕ(y) ("hay algún nodo que satisface ϕ"); solo se puede preguntar si *algún vecino* y de x satisface ϕ, es decir ∃y (E(x,y) ∧ ϕ(y)). Sintácticamente, una fórmula de lógica modal graduada es Col(x), o bien una de:

```
¬ϕ(x),    ϕ(x) ∧ ψ(x),    ∃^{≥N} y (E(x,y) ∧ ϕ(y))
```

Así δ(x) := Red(x) ∧ ∃y (E(x,y) ∧ Blue(y)) *sí* está en la lógica, pero γ(x) de la Ec. 4 *no*, porque usa ¬E(x,y) como guarda (un cuantificador sobre los *no* vecinos), lo cual está prohibido.

- **Proposición 4.1.** Todo clasificador de lógica modal graduada es capturado por una AC-GNN simple y homogénea.

La construcción (Apéndice B) es muy instructiva y conecta directamente con el contenido de la clase. La idea: cada **dimensión** del vector de features representa una **subfórmula** del clasificador. Se enumeran las subfórmulas (ϕ1,...,ϕL) en orden topológico, y la AC-GNN tiene L capas. El invariante que se demuestra por inducción: tras la capa i, la componente ℓ del vector del nodo v vale 1 si y solo si v ⊨ ϕℓ. La función de combinación es COM(x,y) = σ(xC + yA + b) con σ la **ReLU truncada** σ(x) = min(max(0,x),1) y la agregación es la **suma** de vecinos. Las matrices A, C y el sesgo b se construyen columna a columna según el tipo de subfórmula:

- **Color** (ϕℓ = Col): Cℓℓ = 1 (copia la feature inicial del color).
- **Conjunción** (ϕℓ = ϕj ∧ ϕk): Cjℓ = Ckℓ = 1, bℓ = −1 → la ReLU truncada da 1 solo si ambas valen 1.
- **Negación** (ϕℓ = ¬ϕk): Ckℓ = −1, bℓ = 1 → da 1 solo si ϕk vale 0.
- **Cuantificador graduado** (ϕℓ = ∃^{≥N}(E(x,y) ∧ ϕk)): Akℓ = 1 (suma sobre vecinos cuántos satisfacen ϕk), bℓ = −N+1 → la ReLU truncada da 1 si y solo si hay al menos N vecinos que satisfacen ϕk.

L es la "profundidad de cuantificación" del clasificador y *no* depende del grafo. La clasificación final extrae la componente correspondiente a ϕ = ϕL. Es notable que basta con AC-GNN **simples y homogéneas** (mismos parámetros en todas las capas), y que los resultados *negativos* valen para AC-GNN arbitrarias.

**El recíproco —y aquí está el resultado más fuerte:**

- **Teorema 4.2.** Un clasificador lógico es capturado por AC-GNN *si y solo si* es expresable en lógica modal graduada.

La dirección "←" es la Proposición 4.1. La dirección "→" (lo único que las AC-GNN pueden aprender con precisión son fórmulas de lógica modal graduada) se apoya en una extensión reciente de resultados profundos de teoría de modelos finitos (Otto, 2019). La maquinaria (Apéndice C) usa la noción de *unravelling* (el árbol de despliegue de un nodo a profundidad L) y una versión graduada del teorema de van Benthem–Rosen: el fragmento de FO unaria que depende solo del unravelling de un nodo es exactamente la lógica modal graduada. La dirección "→" vale **sin importar** qué operadores de agregación y combinación se elijan: es una limitación *arquitectónica* de las AC-GNN, no de las funciones específicas.

## 5. ACR-GNN: añadir readout global para capturar todo FOC2

### 5.1. La arquitectura

La segunda pregunta: ¿qué extensión mínima permite capturar todo FOC2? El defecto es la localidad, así que la cura natural es permitir un **cómputo global** en cada capa —un *global attribute* en el marco de Battaglia et al. (2018). El paper lo llama **readout**. Una **aggregate-combine-readout GNN (ACR-GNN)** añade funciones de lectura {READ^{(i)}} que agregan los vectores de *todos* los nodos del grafo, generalizando la recursión (Ec. 5):

```
x_v^{(i)} = COM^{(i)}( x_v^{(i-1)},  AGG^{(i)}({{x_u^{(i-1)} | u ∈ N(v)}}),  READ^{(i)}({{x_u^{(i-1)} | u ∈ G}}) )
```

Cada capa primero lee (suma) sobre todos los nodos de G, luego agrega sobre los vecinos de v, y combina la feature de v con ambos vectores. Una ACR-GNN **simple** usa la suma como READ y COM(x1,x2,x3) = f(x1·C + x2·A + x3·R + b). Es importante distinguir este *readout intermedio* (operación global por capa, usada para clasificación de **nodos**) del *readout final* de clasificación de **grafo** del que hablaban Morris/Xu.

### 5.2. ACR-GNN captura FOC2

Para ver cómo el readout rompe la localidad, retomemos γ(x) (Ec. 4: nodo rojo con otro nodo *no conectado* que tiene dos vecinos azules), que ninguna AC-GNN captura. Una ACR-GNN lo implementa así: (1) una agregación local guarda en cada nodo si satisface B = "tener al menos 2 vecinos azules"; (2) un **readout** cuenta cuántos nodos satisfacen B en *todo* el grafo; (3) otra agregación local cuenta cuántos *vecinos* de cada nodo satisfacen B. Entonces γ marca verdadero a todo nodo rojo cuyo número de *vecinos* con B sea *menor* que el total de nodos con B en el grafo (la resta da los no-vecinos con B).

- **Teorema 5.1.** Todo clasificador FOC2 es capturado por una ACR-GNN simple y homogénea.

La construcción (Apéndice D) extiende la de la Proposición 4.1: además de los vectores por nodo, se mantienen vectores globales x_G^{(i)} donde cada componente cuenta cuántos nodos del grafo satisfacen la subfórmula correspondiente. El readout y la agregación simplemente suman. La clave teórica es un paso intermedio: una caracterización de FOC2 mediante una **versión extendida de la lógica modal graduada con parámetros modales** (la lógica EMLC de Lutz et al., 2001). Un lema muestra que todo parámetro modal se reduce a 8 casos canónicos sobre tres conjuntos disjuntos —el nodo mismo (id), sus vecinos (e), y los no-vecinos distintos del nodo (¬e ∩ ¬id)— y para cada caso se da una columna explícita de las matrices A, C, R, b. El readout aporta el conteo global (las componentes con Rkℓ = 1) necesario para los cuantificadores sobre no-vecinos. Queda como problema abierto desafiante si FOC2 es *exactamente* lo que capturan las ACR-GNN.

### 5.3. ¿Cuántos readouts hacen falta? AC-FR-GNN

La construcción anterior usa readouts una cantidad de veces que depende de la fórmula. Como un cómputo global es costoso, surge la pregunta de cuántos readouts se necesitan realmente.

- **Teorema 5.2.** Todo clasificador FOC2 es capturado por una **AC-FR-GNN** (aggregate-combine con un *único* readout final).

Es decir, **un solo readout basta** para igualar el poder expresivo de múltiples readouts, pero a costa de complicar severamente la red (la AC-FR-GNN nunca es simple ni homogénea). La construcción (Apéndice E) *no* evalúa la fórmula incrementalmente capa a capa, sino que refina la arquitectura **GIN** de Xu et al. (2019): primero construye una AC-GNN (llamada A_primes) que mapea cada nodo a un número natural codificando inyectivamente su *unravelling* completo de profundidad L (usando una codificación por **números primos** para manejar grafos de grado no acotado, ya que la codificación original de GIN asume multiconjuntos de tamaño acotado). El readout agrega cuántas veces aparece cada tipo de vecindario en el grafo, y la combinación final decodifica los vecindarios para evaluar las construcciones no-locales de FOC2.

## 6. Experimentos

Los experimentos (PyTorch Geometric) son *sintéticos por diseño*: el objetivo no es batir benchmarks sino mostrar que las diferencias teóricas entre AC-GNN y ACR-GNN **se observan al aprender de ejemplos**. Grafos con 5 colores en codificación one-hot, tres conjuntos: train (5k grafos, 50–100 nodos), test mismo tamaño (500 grafos), y test de mayor tamaño (500 grafos, 100–200 nodos) para evaluar **generalización a tamaños no vistos**. Optimizador Adam. Se prueban además las GIN de Xu et al. adaptadas a clasificación de nodos. La precisión se mide sobre el total de nodos correctamente clasificados.

**Experimento 1 — separar AC-GNN de ACR-GNN.** Clasificador trivial α(x) := Red(x) ∧ ∃y Blue(y) ("nodo rojo en un grafo que contiene al menos un nodo azul"). Resultados (Tabla 1):

| Modelo | Line Train | Line Test (bigger) | E-R Train | E-R Test (bigger) |
|---|---|---|---|---|
| AC-5 | 0.887 | 0.892 | 0.951 | 0.929 |
| AC-7 | 0.892 | 0.897 | 0.967 | 0.958 |
| GIN-5 | 0.861 | 0.867 | 0.830 | 0.817 |
| GIN-7 | 0.863 | 0.870 | 0.818 | 0.813 |
| **ACR-1** | **1.000** | **1.000** | **1.000** | **1.000** |

Una ACR-GNN de **una sola capa** logra precisión perfecta —como predice la teoría, dada la trivialidad de la propiedad. En cambio AC-GNN y GIN **no logran ajustar ni el train** en grafos de línea, ni con 7 capas. En grafos aleatorios Erdős–Rényi rinden mejor con más capas, y mejoran cuanto **más densos** son los grafos: consistente con la teoría, porque las AC-GNN mueven información local hasta distancia = número de capas, y los grafos densos acortan las distancias máximas entre nodos.

**Experimento 2 — propiedades FOC2 complejas.** Clasificadores anidados α_{i+1}(x) := ∃^{[N,M]} y (α_i(y) ∧ ¬E(x,y)) sobre α_0 = Blue, donde ∃^{[N,M]} ("entre N y M nodos") combina ∃^{≥N} y ¬∃^{≥M+1}. Cada α_i está en FOC2 y usa cuantificadores sobre *no-vecinos* (no-locales). Resultados (Tabla 2, E-R densos): al aumentar la profundidad de anidamiento se necesitan más capas. Las **ACR-GNN suben con la profundidad** (ACR-1 → ACR-3 mejoran en α2 y α3) mientras AC-GNN y GIN **no logran despegar del baseline trivial del 50%** en el test grande, ni con 10 capas. Las AC-FR-GNN (un readout final) también capturan, confirmando el Teorema 5.2; pero en la práctica **más de un readout ayuda al aprendizaje** de propiedades complejas (la diferencia entre el resultado teórico y la facilidad de optimización).

**Hallazgos de implementación:** el **agregador y readout de suma** rinden consistentemente mejor que promedio o máximo, justo como predicen las construcciones de la Proposición 4.1 y el Teorema 5.1; la elección de la función de combinación no fue significativa.

**Benchmark real PPI.** En Protein-Protein Interaction (Zitnik & Leskovec, 2017) ambos modelos rinden alto (AC: 97.5 F1; ACR: 95.4 F1) pero **no se observa mejora de ACR sobre AC**. El paper lo atribuye, citando a Chen et al. (2019), a que los benchmarks habituales son inadecuados para distinguir variantes avanzadas de GNN: las propiedades no-locales que ACR-GNN sabe expresar simplemente no son discriminantes en PPI.

## 7. Limitaciones

El propio paper es honesto sobre los límites del resultado:

- **No se cierra la caracterización de ACR-GNN.** Se demuestra que ACR-GNN captura *todo* FOC2, pero queda abierto si FOC2 es *exactamente* lo que captura (a diferencia del Teorema 4.2, que sí cierra el "si y solo si" para AC-GNN y lógica modal graduada).
- **Solo clasificación de nodos.** No se aborda clasificación de grafos (donde el readout es la operación final), que es el foco de Morris/Xu.
- **Construcciones teóricas no necesariamente prácticas.** La AC-FR-GNN del Teorema 5.2 usa una codificación por números primos que el paper admite "no afirmamos que pueda realizarse en la práctica"; es una construcción de existencia.
- **Operadores fijos.** Las construcciones positivas usan suma + ReLU truncada. Queda como trabajo futuro si valen con max, sigmoid u otros operadores.
- **La ventaja no se traduce a benchmarks reales** (PPI): la expresividad extra es teóricamente real pero empíricamente latente cuando el dataset no exige propiedades no-locales.
- **El conteo de capas depende de la fórmula**, lo que limita la transferencia a grafos donde la profundidad de cuantificación requerida no se conoce de antemano.

## 8. Impacto

Este paper entrega el **fundamento lógico** de la expresividad de las GNN, complementario y de igual peso que el enfoque combinatorio de WL/GIN:

- **Dos varas para el mismo problema.** GIN (Xu et al., 2019) midió las GNN contra el algoritmo combinatorio WL; este paper las mide contra la lógica de primer orden FOC2. La equivalencia clásica WL ↔ FOC2 (Cai et al., 1992) es el puente que hace que ambos enfoques hablen entre sí. Juntos forman el par teórico canónico de "qué pueden y qué no pueden hacer las GNN".
- **Un mapa preciso.** AC-GNN = lógica modal graduada (= ALCQ); AC-GNN + readout global = FOC2. Esto da al practicante una regla mental clara: si la propiedad que quiero clasificar es *local y guardada por aristas*, una GNN estándar basta; si involucra cuantificadores *no-locales* (sobre no-vecinos, conteos globales), necesito un readout.
- **Puente con representación del conocimiento.** Identificar que las AC-GNN capturan exactamente ALCQ —el corazón de OWL 2— conecta el aprendizaje de grafos con décadas de teoría de lógicas de descripción y ontologías.
- **Línea de investigación local.** Confirma a Chile (PUC / UChile / IMFD) como un polo de fundamentos teóricos de la IA, con un resultado publicado en ICLR que es citado y enseñado internacionalmente.

## 9. Conexión con la Clase 27 (Redes Neuronales de Grafos)

- **Es un paper de los créditos de la clase.** La Clase 27 cita explícitamente "The Logical Expressiveness of Graph Neural Networks" (Tópicos Avanzados de IA 2019) atribuido a Jorge Pérez, coautor. El material teórico de la clase sobre expresividad proviene de la misma fuente que este paper.
- **El par teórico con GIN.** La clase presenta el message passing (AC-GNN) y el test WL como medida del poder discriminativo (vía GIN de Xu et al.). Este paper es la otra mitad: la medida del poder *expresivo* vía lógica. Entender ambos es entender el techo teórico de las GNN de message passing.
- **El readout global y el mean pooling de grafo.** La clase explicó la clasificación de grafo mediante *mean pooling*, g = (1/N) Σ_v h_v —una agregación global de todos los nodos. El componente READ de las ACR-GNN es exactamente esa misma idea de agregación global, pero usada de forma *intermedia* (en cada capa, para clasificar nodos) en lugar de solo al final. El paper muestra que ese mismo gesto arquitectónico —mirar el grafo entero, no solo el vecindario— es lo que convierte una GNN local en una capaz de expresar propiedades globales (FOC2 completo). Es la justificación teórica de por qué el pooling global importa.
- **Por qué apilar capas no basta.** La clase puede mostrar que más capas amplían el campo receptivo; este paper demuestra el límite duro: con L capas fijas la información local no viaja más allá de distancia L, y para propiedades entre componentes desconectadas *ningún* número de capas alcanza. El readout es la respuesta arquitectónica a esa barrera.
