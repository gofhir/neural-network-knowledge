---
title: "Neural Turing Machines (2014)"
weight: 345
math: true
---

{{< paper-card
    title="Neural Turing Machines"
    authors="Alex Graves, Greg Wayne, Ivo Danihelka"
    year="2014"
    venue="arXiv / DeepMind"
    pdf="/papers/ntm-graves-2014.pdf"
    arxiv="1410.5401" >}}
Paper seminal de Google DeepMind que **fundó la línea de la memoria externa diferenciable**: una red neuronal (el *controlador*) acoplada a una matriz de memoria direccionable, entrenable end-to-end por descenso de gradiente. La Neural Turing Machine (NTM) accede a la memoria mediante **cabezas de lectura/escritura diferenciables**, combinando direccionamiento por **contenido** (similitud) y por **ubicación** (desplazamiento), y aprende algoritmos simples —copiar, repetir, ordenar— que **generalizan a secuencias mucho más largas** que las vistas en entrenamiento. No está citado en las slides de la [Clase 30](/clases/clase-30), pero es el modelo canónico de la familia "memoria tipo computador" frente a las [Memory Networks](/fundamentos/redes-de-memoria) de Weston.
{{< /paper-card >}}

---

## Contexto: la memoria que le faltaba al aprendizaje automático

El paper arranca con la arquitectura de von Neumann (1945): un computador usa tres mecanismos —operaciones elementales, control de flujo lógico y **memoria externa** legible/escribible durante el cómputo. El aprendizaje automático moderno, observan los autores, había modelado datos complejos con éxito pero "descuidado ampliamente el control de flujo lógico y la memoria externa". La NTM se propone tapar ese hueco.

Las redes recurrentes (RNN) son, en teoría, *Turing-completas* (Siegelmann y Sontag, 1995): con el cableado correcto podrían simular cualquier procedimiento. Pero "lo que es posible en principio no siempre es simple en la práctica". Las RNN estándar no *aprenden algoritmos*: aprenden transformaciones estadísticas que generalizan mal fuera del rango de entrenamiento. La gran innovación previa fue la **LSTM** (Hochreiter y Schmidhuber, 1997), que evita el gradiente que se desvanece embebiendo *integradores perfectos* con compuertas programables. La NTM enriquece la RNN como Turing enriqueció a las máquinas de estados finitos —dándole una cinta de memoria grande y direccionable— pero con una diferencia decisiva: es **un computador diferenciable**, entrenable por gradiente, lo que la convierte en un mecanismo práctico para *aprender programas* a partir de ejemplos entrada-salida.

Hay también un marco cognitivo: la noción de **memoria de trabajo** (Baddeley et al., 2009), con un "ejecutivo central" que enfoca la atención sobre un buffer. La NTM se le parece porque usa un **proceso atencional** para leer y escribir, pero *aprende a usar* su memoria en vez de ejecutar procedimientos fijos. Es también una respuesta de ingeniería a la vieja crítica de Fodor y Pylyshyn (1988) sobre la incapacidad de las redes para *variable binding*.

## Arquitectura: controlador + memoria externa diferenciable

La contribución combina dos componentes y una propiedad transversal:

1. Un **controlador** (red neuronal recurrente —LSTM— o feedforward) que interactúa con el exterior vía vectores de entrada y salida.
2. Un **banco de memoria** externo: una matriz $M_t$ de tamaño $N \times M$, con $N$ ubicaciones y vectores de tamaño $M$ en cada una.
3. La propiedad que lo hace funcionar: **todo componente es diferenciable**. El controlador accede a la memoria mediante **cabezas** (heads) de lectura y escritura, por analogía con la máquina de Turing.

La clave del invento es que las operaciones son *borrosas* (blurry): en vez de acceder a una sola celda, cada cabeza interactúa en mayor o menor grado con *todas* las ubicaciones, ponderadas por un vector de pesos normalizado $w_t$ que define un foco atencional. Como ese foco es continuo, **es diferenciable**, y toda la arquitectura se entrena por descenso de gradiente. Una cabeza puede atender **agudamente** a una ubicación o **débilmente** a muchas. La analogía con el computador es explícita: el **controlador es la CPU** (con instrucciones aprendidas en vez de predefinidas), la **matriz de memoria es la RAM**, y —si el controlador es recurrente— sus **activaciones ocultas son los registros**.

### Lectura

Dado el vector de pesos $w_t$ (normalizado: suma 1, componentes en $[0,1]$), el vector leído $r_t$ es la **combinación convexa** de las filas de la memoria:

$$r_t \leftarrow \sum_i w_t(i)\, M_t(i)$$

Es claramente diferenciable respecto a la memoria y a los pesos. En esencia es el mismo mecanismo de **atención suave** que Bahdanau et al. (2014) introdujeron ese mismo año para traducción —el paper los cita como precursores— pero aplicado a una memoria persistente y editable.

### Escritura: borrar + añadir (estilo compuertas LSTM)

Inspirada en las compuertas *input* y *forget* de la LSTM, cada escritura se descompone en dos pasos. Primero un **borrado** con un vector $e_t$ de componentes en $(0,1)$:

$$\tilde{M}_t(i) \leftarrow M_{t-1}(i)\,[1 - w_t(i)\, e_t]$$

Una ubicación se pone a cero solo si *tanto* su peso *como* el elemento de borrado valen 1; si cualquiera es cero, la memoria queda intacta. Luego una **adición** con el vector $a_t$:

$$M_t(i) \leftarrow \tilde{M}_t(i) + w_t(i)\, a_t$$

Como $e_t$ y $a_t$ tienen $M$ componentes independientes, la red tiene **control fino** sobre qué elementos de cada ubicación modifica. Con múltiples cabezas el orden es irrelevante (los borrados conmutan, las adiciones se suman). El paralelo con LSTM es directo: borrar es la compuerta de olvido, añadir es la de entrada —pero sobre una memoria externa direccionable en lugar del estado de celda interno.

### Direccionamiento: contenido + ubicación

La parte conceptualmente más rica es *cómo* se construye $w_t$, combinando **dos mecanismos complementarios**.

**(a) Por contenido (focusing by content).** Cada cabeza emite un **vector clave** $k_t$ que se compara con cada fila $M_t(i)$ por **similitud coseno**. Una softmax con un parámetro de **fuerza de clave** $\beta_t > 0$ —que afila o atenúa el foco— produce el peso por contenido. La recuperación es simple: basta que el controlador genere una aproximación a una parte del dato y el sistema devuelve el valor exacto. El paper lo relaciona con el direccionamiento de las **redes de Hopfield** (1982).

**(b) Por ubicación (focusing by location).** No todo se presta al contenido: en aritmética el contenido de una variable es arbitrario pero la variable necesita un *nombre/dirección*. El direccionamiento por ubicación facilita la **iteración** sobre celdas contiguas y los **saltos** mediante desplazamiento rotacional, en tres pasos:

1. **Interpolación:** una compuerta escalar $g_t \in (0,1)$ mezcla el peso por contenido actual con el del paso anterior. Si $g=0$ se reusa el peso previo; si $g=1$ solo cuenta el contenido.
2. **Shift:** la cabeza emite una distribución $s_t$ sobre desplazamientos enteros permitidos (p. ej. $\{-1, 0, +1\}$); la rotación es una **convolución circular** con aritmética de índices módulo $N$.
3. **Afilado (sharpening):** la convolución puede emborronar los pesos; un escalar $\gamma_t \geq 1$ afila la distribución elevando cada peso a $\gamma_t$ y renormalizando.

El sistema combinado opera en tres modos: elegir por contenido sin más; elegir por contenido y desplazar (saltar a una ubicación *adyacente*); o rotar el peso anterior sin contenido (*iterar* por una secuencia avanzando lo mismo cada paso). El direccionamiento por contenido es estrictamente más general, pero proveer la ubicación como primitiva resultó **esencial para ciertas formas de generalización**, así que se usan ambos.

### El controlador

La decisión arquitectónica más significativa es el tipo de controlador. Uno **recurrente (LSTM)** tiene memoria interna que complementa la matriz (sus activaciones funcionan como registros). Uno **feedforward** ofrece **mayor transparencia** (el patrón de acceso es más fácil de interpretar), pero su número de cabezas impone un cuello de botella: con una sola cabeza de lectura solo hace una transformación unaria por paso.

## Experimentos: aprendiendo cinco algoritmos

El objetivo no era solo resolver, sino resolver **aprendiendo programas compactos** cuyo sello es **generalizar más allá del rango de entrenamiento**. Se compararon tres arquitecturas: NTM feedforward, NTM con LSTM, y LSTM estándar. Tareas supervisadas con objetivos binarios, salida sigmoide, entropía cruzada, RMSProp y gradientes recortados a $(-10, 10)$. Memoria de $128 \times 20$.

- **Copy.** Reproducir secuencias de vectores binarios de 8 bits (longitud 1–20 en entrenamiento). La NTM aprendió mucho más rápido y a costo menor que la LSTM, y siguió copiando secuencias mucho más largas (hasta el límite de 128 ubicaciones), mientras la LSTM se degradó pasada la longitud 20. Inspeccionando la interacción controlador-memoria, los autores reconstruyen el **pseudocódigo** aprendido —mover al inicio, escribir incrementando posición, volver al inicio, leer e incrementar—, "esencialmente cómo lo haría un programador humano en bajo nivel". La NTM aprendió a **crear e iterar sobre arrays**.

- **Repeat Copy.** Emitir la secuencia copiada un número dado de veces (un **bucle `for`**). Generalizó a más repeticiones, pero **no logró llevar la cuenta** fuera del rango —limitación atribuida a representar el número de repeticiones *numéricamente*.

- **Associative Recall.** Indirección: dado un ítem, devolver el *siguiente*. La NTM llegó cerca de costo cero en ~30.000 episodios; la LSTM no lo alcanzó tras un millón. La **NTM feedforward aprendió más rápido que la NTM-LSTM**, indicio de que la memoria externa mantiene mejor la estructura de datos que el estado interno. El algoritmo: escribir una representación comprimida de cada ítem, recomputarla en la consulta, buscarla por contenido y **desplazar en 1**.

- **Dynamic N-Grams.** Usar la memoria como **tabla re-escribible** para contar estadísticas de transición (6-gramas binarios). La NTM superó levemente a la LSTM pero sin alcanzar el **estimador bayesiano óptimo**.

- **Priority Sort.** Ordenar 20 vectores por prioridad escalar. La NTM **usa la prioridad para fijar la ubicación de cada escritura** (ajuste *lineal* prioridad→posición), y luego lee en orden. Requirió **ocho cabezas paralelas** con controlador feedforward, reflejo del cuello de botella unario.

Detalle revelador: el **número de parámetros de la NTM no crece con las ubicaciones de memoria**, mientras en la LSTM crece cuadráticamente con las unidades ocultas. La NTM resuelve copy con ~17 mil parámetros frente a >1.3 millones de la LSTM equivalente.

## Limitaciones reconocidas

- **Conteo numérico frágil:** representar las repeticiones como escalar impide generalizar el "cuándo terminar".
- **Capacidad de memoria fija:** superadas las 128 ubicaciones, los shifts cíclicos dan la vuelta y sobrescriben. La memoria no crece dinámicamente.
- **Subóptimo en tareas probabilísticas:** solo aproxima el estimador bayesiano.
- **Resultados "preliminares":** tareas sintéticas simples, sin lenguaje natural ni gran escala; prueba de concepto.
- **Cuello de botella del controlador feedforward:** el número de cabezas limita la aridad por paso.

## Impacto: dos líneas de memoria externa

La NTM separó limpiamente *cómputo* (controlador) de *almacenamiento* (matriz) e hizo el acceso aprendible, abriendo un programa con dos descendencias directas de los mismos autores y su entorno:

- **Differentiable Neural Computer (DNC)** (Graves et al., *Nature*, 2016): continuación directa que corrige las limitaciones de capacidad añadiendo **asignación dinámica de memoria** y **enlace temporal** del orden de escritura, resolviendo razonamiento sobre grafos. → ver [`/papers/dnc-graves-2016`](/papers/dnc-graves-2016).
- **Memory-Augmented Neural Networks (MANN)** (Santoro et al., 2016): aplican la maquinaria NTM al **meta-aprendizaje** de *one-shot learning*, con un direccionamiento variante (Least Recently Used Access). → ver [`/papers/mann-santoro-2016`](/papers/mann-santoro-2016).

**El contraste con la línea Weston (Memory Networks)** es la distinción conceptual central de la Clase 30:

- La **NTM** implementa memoria al estilo **computador**: matriz de tamaño fijo accedida por **direccionamiento** (contenido *y* ubicación), escritura por borrar+añadir, iteración/saltos por shifts. Inspiración von Neumann/Turing. Memoria **de bajo nivel, granular y editable celda a celda**; el modelo *aprende a programar* sobre ella.
- Las **Memory Networks** (Weston, Chopra, Bordes, 2014; y la versión *end-to-end* de Sukhbaatar et al., 2015) usan memoria por **slots de contenido**: cada slot guarda un hecho/frase, el acceso es por **atención sobre contenido** en varios *hops*, y la escritura es trivial. Su caso natural es la **respuesta a preguntas** (bAbI).

Ambas familias comparten la idea fundacional —una **memoria explícita, separada de los pesos, leída por atención diferenciable y entrenable end-to-end**— pero difieren en filosofía: la NTM quiere *aprender algoritmos* sobre una RAM granular; las Memory Networks quieren *razonar sobre hechos*. La convergencia de ambas líneas con la atención de Bahdanau desemboca años después en el **mecanismo de atención del Transformer**, legible como una memoria de contenido sin escritura persistente.

## Por qué importa para la Clase 30

La [Clase 30](/clases/clase-30) organiza su material en torno a dotar a las redes de una **memoria explícita y editable**, separada de los parámetros. La NTM es su complemento obligatorio por tres razones:

1. **Es el origen genealógico de la materia.** Sin ella falta la *otra mitad* del mapa: la familia "tipo computador, por direccionamiento" de Graves/DeepMind frente a la familia "de slots de contenido" de Weston. Presentar ambas da el eje conceptual completo —direccionamiento por ubicación *vs.* slots de contenido.
2. **Hace tangible qué significa "memoria diferenciable".** Los cinco mecanismos —combinación convexa para leer, borrar+añadir para escribir, similitud coseno para contenido, convolución circular para shift, afilado por $\gamma$— son el repertorio concreto que vuelve *derivable* el acceso a una memoria discreta.
3. **Conecta con el resto del curso.** La lectura de la NTM es exactamente la atención suave de traducción y Transformers; la diferencia es la *persistencia* y la *editabilidad*. La generalización a secuencias más largas motiva tanto el meta-aprendizaje (ver [MANN](/papers/mann-santoro-2016)) como el debate sobre razonamiento composicional.

Para quien trabaja con datos clínicos/tabulares, la lección transferible es la separación cómputo/almacenamiento: un sistema que debe *recordar y editar* registros estructurados (un historial que se actualiza, una tabla de hechos del paciente) tiene en la NTM el arquetipo de cómo una red lee, escribe y revisa una memoria externa de forma diferenciable, en vez de comprimirlo todo en pesos fijos.

## Notas y enlaces

- Preprint: arXiv:1410.5401v2 (10 dic 2014), [arxiv.org/abs/1410.5401](https://arxiv.org/abs/1410.5401).
- Afiliación: Google DeepMind, Londres. Uno de los primeros trabajos emblemáticos del laboratorio tras su adquisición por Google.
- Fundamentos: [`/fundamentos/memory-augmented-networks`](/fundamentos/memory-augmented-networks) · [`/fundamentos/redes-de-memoria`](/fundamentos/redes-de-memoria).
- Descendientes: [`/papers/dnc-graves-2016`](/papers/dnc-graves-2016) · [`/papers/mann-santoro-2016`](/papers/mann-santoro-2016).
