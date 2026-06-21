---
title: "The Logical Expressiveness of Graph Neural Networks (2020)"
weight: 309
math: true
---

{{< paper-card
    title="The Logical Expressiveness of Graph Neural Networks"
    authors="Pablo Barceló, Egor V. Kostylev, Mikaël Monet, Jorge Pérez, Juan Reutter, Juan Pablo Silva"
    year="2020"
    venue="ICLR 2020"
    pdf="/papers/logical-expressiveness-barcelo-2020.pdf" >}}
Paper teórico que caracteriza, mediante **lógica de primer orden**, exactamente qué clasificadores de nodos puede expresar una GNN. Las GNN de message passing estándar (que el paper llama **AC-GNN**) capturan exactamente la **lógica modal graduada** —un fragmento estricto de FOC2—, ni más ni menos; añadiéndoles un **readout global** (las **ACR-GNN**) se captura todo FOC2. Es la contraparte declarativa y lógica del enfoque combinatorio Weisfeiler-Lehman/[GIN](/papers/gin-xu-2019), complemento natural del material de la [Clase 27](/clases/clase-27) sobre [expresividad de GNN](/fundamentos/expresividad-gnn).
{{< /paper-card >}}

---

## Un paper "de la casa"

Este es, literalmente, un paper local. Cinco de sus seis autores trabajan en Chile —entre la Pontificia Universidad Católica, la Universidad de Chile y el **Instituto Milenio Fundamentos de los Datos (IMFD)**— y dos de ellos, **Pablo Barceló** y **Jorge Pérez**, son figuras centrales del grupo chileno de fundamentos teóricos de la IA. Jorge Pérez aparece citado en los créditos de la [Clase 27](/clases/clase-27) del curso, de modo que el material teórico de la clase sobre expresividad no solo referencia este resultado, sino que proviene del propio coautor.

Para un estudiante en Chile esto tiene un valor doble: es teoría de primera línea publicada en el venue más competitivo del área (ICLR 2020) y, a la vez, es producción local, demostrando que el aporte fundacional —no solo aplicado— a las [redes neuronales de grafos](/fundamentos/redes-neuronales-de-grafos) salió de universidades chilenas. El código está disponible en [github.com/juanpablos/GNN-logic](https://github.com/juanpablos/GNN-logic), implementado en PyTorch Geometric.

## Contexto: la pregunta de expresividad

Trabajos previos (Morris et al., 2019; Xu et al., 2019 —el paper de [GIN](/papers/gin-xu-2019)) habían caracterizado el poder **discriminativo** de las GNN en términos del test de **Weisfeiler-Lehman (WL)** para isomorfismo de grafos. El test WL construye, ronda a ronda, una coloración de los nodos: a cada nodo le asigna un color nuevo según su color previo y el *multiconjunto* de colores de sus vecinos. La GNN básica —la que actualiza el vector de *features* de un nodo combinándolo con la agregación de los de sus vecinos— es lo que el paper bautiza **aggregate-combine GNN (AC-GNN)**.

El resultado conocido era: si WL asigna el mismo color a dos nodos, *toda* AC-GNN los clasifica igual; WL es el techo del poder discriminativo de las AC-GNN, y existen AC-GNN que reproducen exactamente la coloración WL.

**El hueco que el paper detecta.** Que WL refine la clasificación de una AC-GNN no implica que la AC-GNN pueda capturar *todo* clasificador refinado por WL. El poder **discriminativo** (distinguir dos nodos) y el poder **expresivo** (computar una función booleana concreta sobre los nodos) son cosas distintas, y la literatura previa solo había resuelto el primero. La pregunta abierta: ¿qué clasificadores de nodos puede capturar una arquitectura como AC-GNN? El paper cambia de lente: en vez de medir la expresividad contra un algoritmo combinatorio (WL), la mide contra la **lógica de primer orden**.

## Clasificadores lógicos y FOC2

El paper se restringe a **clasificadores lógicos**: fórmulas unarias (con una variable libre) en lógica de primer orden (FO) sobre grafos con vértices coloreados. Una fórmula $\varphi(x)$ clasifica como verdadero a los nodos $v$ tales que $(G, v) \models \varphi$. Por ejemplo:

$$\alpha(x) := \text{Red}(x) \wedge \exists y\,(E(x,y) \wedge \text{Blue}(y)) \wedge \exists z\,(E(x,z) \wedge \text{Green}(z))$$

verdadera en los nodos rojos con al menos un vecino azul y al menos uno verde. Una GNN **captura** un clasificador $\varphi(x)$ si ambos coinciden sobre todo nodo de todo grafo posible.

Comparar AC-GNN contra toda FO es demasiado: FO es excesivamente poderosa. El paper elige **FOC2**, el fragmento de FO que (a) permite solo **dos variables** pero (b) añade **cuantificadores de conteo** $\exists^{\geq N}$ ("existen al menos $N$ nodos que satisfacen $\varphi$"). Reducir el número de variables baja la expresividad, pero el conteo recupera parte del poder. Resultado: **FOC2 es estrictamente menos expresiva que FO, pero estrictamente más que FO2** (la FO de dos variables sin conteo).

La justificación profunda de FOC2 es un resultado clásico de Cai, Fürer e Immerman (1992): para todo grafo y nodos $u, v$, **WL los colorea igual si y solo si todos los clasificadores FOC2 los clasifican igual**. WL y FOC2 son dos caras de la misma moneda discriminativa, lo que hace de FOC2 la vara natural para medir las AC-GNN.

### Por qué AC-GNN ≠ FOC2

Hay una trampa tentadora: encadenar "WL refina a AC-GNN" con "WL ↔ FOC2" para concluir "toda AC-GNN captura todo FOC2". **Es falso.** Que dos nodos indistinguibles por WL lo sean por todo FOC2 y por toda AC-GNN no dice nada sobre si una fórmula FOC2 *concreta* puede *expresarse* como una AC-GNN. La indistinguibilidad es una relación de equivalencia; la captura es la realización de una función específica.

> **Proposición 3.3.** Existe un clasificador FOC2 que ninguna AC-GNN captura.

El testigo más simple es $\alpha(x) := \text{Red}(x) \wedge \exists y\,\text{Green}(y)$ ("x es rojo y el grafo tiene algún nodo verde"). La intuición de la imposibilidad es la **localidad**: una AC-GNN con $L$ capas no puede mover información más allá de la distancia $L$ a lo largo de las aristas. Si el nodo rojo y el nodo verde están a distancia mayor que $L$ —o peor, en componentes conexas distintas, donde *ningún* número de capas alcanza— la AC-GNN no puede ver el verde. Esto abre las dos preguntas que estructuran el paper: ¿qué fragmento de FOC2 *sí* capturan las AC-GNN, y qué hay que añadirles para capturar todo FOC2?

## AC-GNN = lógica modal graduada

Como el problema de las AC-GNN es la localidad, la lógica que capturan debe construirse con esa misma limitación. Esa lógica existe y es bien conocida: la **lógica modal graduada** (de Rijke, 2000), equivalente a la lógica de descripción **ALCQ**, fundamental en representación del conocimiento —el lenguaje de ontologías web OWL 2 se apoya en ella.

**La idea: guardar todo con la arista.** La lógica modal graduada fuerza a que toda subfórmula esté *guardada* por el predicado de arista $E$. No se puede expresar $\exists y\, \varphi(y)$ ("hay algún nodo que satisface $\varphi$"); solo si *algún vecino* lo satisface, es decir $\exists y\,(E(x,y) \wedge \varphi(y))$. Sus fórmulas son $\text{Col}(x)$, o bien:

$$\neg\varphi(x), \qquad \varphi(x) \wedge \psi(x), \qquad \exists^{\geq N} y\,(E(x,y) \wedge \varphi(y))$$

> **Teorema 4.2.** Un clasificador lógico es capturado por AC-GNN **si y solo si** es expresable en lógica modal graduada.

La dirección "←" se construye explícitamente (Proposición 4.1): cada **dimensión** del vector de *features* representa una **subfórmula**, las subfórmulas se enumeran en orden topológico, y la AC-GNN tiene una capa por subfórmula. Con suma como agregación y una **ReLU truncada** $\sigma(x) = \min(\max(0,x),1)$ como combinación, las matrices se arman columna a columna: la conjunción usa sesgo $-1$, la negación invierte el signo, el cuantificador graduado $\exists^{\geq N}$ usa sesgo $-N+1$ para activarse solo si hay al menos $N$ vecinos. Basta con AC-GNN **simples y homogéneas** (mismos parámetros en todas las capas). La dirección "→" se apoya en una versión graduada del teorema de van Benthem–Rosen (Otto, 2019) y vale **sin importar** qué operadores de agregación y combinación se elijan: es una limitación *arquitectónica*, no de las funciones específicas.

## ACR-GNN = todo FOC2 (con readout global)

¿Qué extensión mínima permite capturar todo FOC2? El defecto es la localidad, así que la cura es permitir un **cómputo global** en cada capa. El paper lo llama **readout**. Una **aggregate-combine-readout GNN (ACR-GNN)** añade funciones de lectura que agregan los vectores de *todos* los nodos del grafo:

$$x_v^{(i)} = \text{COM}^{(i)}\!\left( x_v^{(i-1)},\; \text{AGG}^{(i)}\big(\{\!\{x_u^{(i-1)} : u \in \mathcal{N}(v)\}\!\}\big),\; \text{READ}^{(i)}\big(\{\!\{x_u^{(i-1)} : u \in G\}\!\}\big) \right)$$

Cada capa lee (suma) sobre todos los nodos, agrega sobre los vecinos y combina los tres. Conviene distinguir este *readout intermedio* (operación global por capa, para clasificar **nodos**) del *readout final* de clasificación de **grafos** del enfoque GIN.

> **Teorema 5.1.** Todo clasificador FOC2 es capturado por una ACR-GNN simple y homogénea.

Para ver cómo el readout rompe la localidad, tomemos un clasificador con cuantificador sobre **no-vecinos** ("nodo rojo con otro nodo no conectado que tiene dos vecinos azules"). Una ACR-GNN lo resuelve: una agregación local marca qué nodos tienen $\geq 2$ vecinos azules; un **readout** cuenta cuántos nodos cumplen eso en *todo* el grafo; otra agregación cuenta cuántos *vecinos* lo cumplen; la resta da los **no-vecinos** que cumplen. La construcción mantiene vectores globales $x_G^{(i)}$ que cuentan, por subfórmula, cuántos nodos la satisfacen, y se apoya en una caracterización de FOC2 vía la lógica modal graduada *extendida con parámetros modales* (EMLC).

Una variante, la **AC-FR-GNN** con un *único readout final* (Teorema 5.2), también captura FOC2, refinando la arquitectura GIN con una codificación por **números primos** para grafos de grado no acotado —aunque el paper admite que es una construcción de existencia, no necesariamente práctica.

## Experimentos

Los experimentos (PyTorch Geometric) son **sintéticos por diseño**: el objetivo no es batir benchmarks, sino mostrar que las diferencias teóricas entre AC-GNN y ACR-GNN **se observan al aprender de ejemplos**. Grafos con 5 colores en *one-hot*, con un conjunto de test de mayor tamaño (100–200 nodos) para evaluar generalización a tamaños no vistos.

**Experimento 1 — separar AC-GNN de ACR-GNN.** Sobre el clasificador trivial $\text{Red}(x) \wedge \exists y\,\text{Blue}(y)$:

| Modelo | Line Train | Line Test (bigger) | E-R Train | E-R Test (bigger) |
|---|---|---|---|---|
| AC-5 | 0.887 | 0.892 | 0.951 | 0.929 |
| AC-7 | 0.892 | 0.897 | 0.967 | 0.958 |
| GIN-5 | 0.861 | 0.867 | 0.830 | 0.817 |
| GIN-7 | 0.863 | 0.870 | 0.818 | 0.813 |
| **ACR-1** | **1.000** | **1.000** | **1.000** | **1.000** |

Una ACR-GNN de **una sola capa** logra precisión perfecta, como predice la teoría. En cambio AC-GNN y GIN **no logran ajustar ni el train** en grafos de línea, ni con 7 capas. En grafos aleatorios Erdős–Rényi mejoran cuanto **más densos** son los grafos: consistente con la teoría, porque las distancias máximas se acortan y la información local alcanza más lejos.

**Experimento 2 — propiedades FOC2 complejas.** Con clasificadores anidados que usan cuantificadores sobre *no-vecinos*, las **ACR-GNN suben con la profundidad** mientras AC-GNN y GIN **no despegan del baseline del 50%** ni con 10 capas. Las AC-FR-GNN (un readout final) también capturan, pero en la práctica más de un readout ayuda al aprendizaje. Hallazgo de implementación: el **agregador y readout de suma** rinden consistentemente mejor que promedio o máximo, justo como predicen las construcciones teóricas.

**Benchmark real PPI.** En Protein-Protein Interaction ambos modelos rinden alto (AC: 97.5 F1; ACR: 95.4 F1) pero **no se observa mejora de ACR sobre AC**: las propiedades no-locales que ACR-GNN sabe expresar simplemente no son discriminantes en ese dataset.

## Limitaciones

- **No se cierra la caracterización de ACR-GNN.** Se demuestra que captura *todo* FOC2, pero queda abierto si FOC2 es *exactamente* lo que captura (a diferencia del "si y solo si" cerrado para AC-GNN).
- **Solo clasificación de nodos**, no de grafos (donde el readout es la operación final).
- **Construcciones teóricas no necesariamente prácticas**: la codificación por números primos de la AC-FR-GNN es de existencia.
- **Operadores fijos** (suma + ReLU truncada); queda abierto si valen con max o sigmoid.
- **La ventaja no se traduce a benchmarks reales** (PPI): la expresividad extra es teóricamente real pero empíricamente latente cuando el dataset no exige propiedades no-locales.

## Por qué importa

Este paper entrega el **fundamento lógico** de la [expresividad de las GNN](/fundamentos/expresividad-gnn), complementario y de igual peso que el enfoque combinatorio WL/[GIN](/papers/gin-xu-2019):

- **Dos varas para el mismo problema.** [GIN](/papers/gin-xu-2019) midió las GNN contra el algoritmo combinatorio WL; este paper las mide contra la lógica FOC2. La equivalencia clásica WL ↔ FOC2 (Cai et al., 1992) es el puente que hace que ambos enfoques hablen entre sí.
- **Un mapa preciso.** AC-GNN = lógica modal graduada (= ALCQ); AC-GNN + readout global = FOC2. Regla mental para el practicante: si la propiedad es *local y guardada por aristas*, una GNN estándar basta; si involucra cuantificadores *no-locales* (sobre no-vecinos, conteos globales), hace falta un readout.
- **Puente con representación del conocimiento.** Que las AC-GNN capturen exactamente ALCQ —el corazón de OWL 2— conecta el aprendizaje de grafos con décadas de teoría de lógicas de descripción y ontologías.
- **Por qué apilar capas no basta.** Con $L$ capas fijas la información local no viaja más allá de la distancia $L$, y entre componentes desconectadas *ningún* número de capas alcanza. El readout —la misma idea del *mean pooling* global de grafo, pero usada de forma *intermedia*— es la respuesta arquitectónica a esa barrera. Es la justificación teórica de por qué el pooling global importa, conectada directamente con el contenido de la [Clase 27](/clases/clase-27).

## Notas y enlaces

- Código: [github.com/juanpablos/GNN-logic](https://github.com/juanpablos/GNN-logic) (PyTorch Geometric).
- Venue: *International Conference on Learning Representations* (ICLR 2020), conference paper.
- Financiamiento: Millennium Institute for Foundational Research on Data (IMFD Chile).
- Afiliaciones: PUC Chile, Universidad de Chile, University of Oxford, IMFD.
