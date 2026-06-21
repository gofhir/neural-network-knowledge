---
title: "GIN: How Powerful are Graph Neural Networks? (2019)"
weight: 305
math: true
---

{{< paper-card
    title="How Powerful are Graph Neural Networks?"
    authors="Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka"
    year="2019"
    venue="ICLR 2019"
    pdf="/papers/gin-xu-2019.pdf"
    arxiv="1810.00826" >}}
Paper canónico que da **teoría de expresividad** a las redes neuronales de grafos: demuestra que toda GNN de paso de mensajes es **a lo más tan poderosa como el test de isomorfismo de Weisfeiler-Lehman de 1 dimensión (1-WL)**, y propone **GIN (Graph Isomorphism Network)**, la arquitectura que provablemente alcanza ese límite usando un agregador inyectivo: suma de vecinos + MLP, $h_v = \text{MLP}\big((1+\epsilon)h_v + \sum_{u} h_u\big)$. La lección central: elegir el agregador (suma, promedio o máximo) **no es cosmético**, decide cuántos grafos distintos el modelo puede distinguir. No está citado en las slides de la [Clase 27](/clases/clase-27), pero es el par teórico de [Barceló (Logical Expressiveness)](/papers/logical-expressiveness-barcelo-2020): ambos delimitan qué pueden y qué no pueden distinguir las GNN.
{{< /paper-card >}}

---

## Contexto

Hacia 2017-2018, el aprendizaje sobre grafos —moléculas, redes sociales, biológicas, financieras— vivía un auge. La receta dominante quedó cristalizada bajo el nombre de *neighborhood aggregation* o *message passing*: cada nodo actualiza recursivamente su vector de características agregando los de sus vecinos. Tras $k$ iteraciones, la representación de un nodo captura información estructural de su vecindario a $k$ saltos; agrupando las representaciones de todos los nodos se obtiene la del grafo entero. Bajo esta plantilla cabían [GCN (Kipf & Welling, 2017)](/papers/gcn-kipf-2017), GraphSAGE, GAT y más de una docena de variantes.

El problema era epistemológico: estas [redes de grafos](/fundamentos/redes-neuronales-de-grafos) *funcionaban*, pero nadie sabía *por qué unas funcionaban mejor que otras*, ni cuáles eran sus límites fundamentales. El diseño se basaba "en intuición empírica, heurísticas y prueba y error experimental". No existía una caracterización formal de *qué estructuras de grafo* una GNN dada puede o no puede distinguir.

El aporte conceptual de Xu et al. fue notar una analogía profunda: el paso de mensajes de las GNN es **estructuralmente idéntico** al test de isomorfismo de **Weisfeiler-Lehman** (1968), un algoritmo clásico de teoría de grafos que también actualiza iterativamente la etiqueta de cada nodo agregando las de sus vecinos. El test WL es notablemente potente —distingue una amplia clase de grafos— y lo que lo hace potente es precisamente su **agregación inyectiva**: mapea vecindarios distintos a etiquetas distintas. La intuición clave: *una GNN puede ser tan poderosa como el test WL si y solo si su esquema de agregación puede modelar funciones inyectivas*.

## Conexión con el test Weisfeiler-Lehman

El primer movimiento técnico es representar el vecindario de un nodo como un **multiconjunto** (multiset): un conjunto que admite elementos repetidos, $X = (S, m)$ con $S$ el conjunto subyacente y $m: S \to \mathbb{N}_{\geq 1}$ las multiplicidades. La abstracción es exacta: distintos vecinos pueden tener vectores de características *idénticos*. Con esta lente, la agregación de vecinos es **una función sobre multiconjuntos**, y la pregunta del poder expresivo se reduce a: *¿puede el agregador mapear multiconjuntos distintos a representaciones distintas?* Una GNN máximamente poderosa nunca colapsaría dos vecindarios distintos; su agregación debe ser **inyectiva**.

De ahí salen los dos teoremas que definen el [marco de expresividad de GNN](/fundamentos/expresividad-gnn):

- **Cota superior (Lema 2).** Si una GNN mapea dos grafos a *embeddings* distintos, entonces el test WL también los declara no isomorfos. **Consecuencia: ninguna GNN basada en agregación puede ser más poderosa que 1-WL.** El test WL es un *techo* para toda la familia de modelos de paso de mensajes. La prueba es por inducción: misma etiqueta WL en cada iteración implica misma característica GNN, porque la misma agregación aplicada a la misma entrada produce la misma salida.

- **Alcanzar la cota (Teorema 3).** Una GNN *iguala* el poder de WL si, con suficientes capas, (a) la actualización de nodos usa funciones inyectivas sobre el multiconjunto de vecinos, y (b) el readout a nivel de grafo es inyectivo. La intuición visual: tras $k$ iteraciones, la etiqueta WL de un nodo representa un *subárbol enraizado de altura $k$*; si la agregación captura el multiconjunto completo, captura recursivamente esos subárboles y por tanto iguala a WL.

Una ventaja conceptual: las características WL son esencialmente *one-hot* y no capturan similitud entre subárboles, mientras que una GNN que satisface el Teorema 3 *generaliza* WL al embeber esos subárboles en un espacio de baja dimensión, mapeando estructuras *similares* a embeddings *similares* —lo que ayuda a generalizar cuando la co-ocurrencia de subárboles es escasa o hay ruido.

## GIN: el agregador inyectivo suma + MLP

¿Cómo construir agregadores inyectivos sobre multiconjuntos? El paper desarrolla una "teoría de multiconjuntos profundos", extendiendo a multiconjuntos el resultado de *Deep Sets* (Zaheer et al., 2017).

**Lema 5.** Si el universo de características $\mathcal{X}$ es contable, existe $f: \mathcal{X} \to \mathbb{R}^n$ tal que $h(X) = \sum_{x \in X} f(x)$ es **único para cada multiconjunto** de tamaño acotado; y *cualquier* función de multiconjunto $g$ se descompone como $g(X) = \varphi\big(\sum_{x \in X} f(x)\big)$. En otras palabras, **la suma puede representar funciones inyectivas —de hecho, universales— sobre multiconjuntos**. La distinción crucial frente a los conjuntos: el promedio, que sí es inyectivo sobre conjuntos, *no* lo es sobre multiconjuntos.

**Corolario 6.** Para incorporar también el nodo central $c$, existe $f$ tal que, para infinitos valores de $\epsilon$ (incluidos todos los irracionales), $h(c, X) = (1+\epsilon)\cdot f(c) + \sum_{x \in X} f(x)$ es único para cada par $(c, X)$. La prueba es ingeniosa: si $\epsilon$ es irracional, el lado izquierdo de cualquier ecuación de colisión sería irracional mientras el derecho es suma finita de racionales —contradicción.

De aquí sale la **regla de actualización de GIN**:

$$ h_v^{(k)} = \text{MLP}^{(k)}\!\left( \big(1 + \epsilon^{(k)}\big) \cdot h_v^{(k-1)} + \sum_{u \in N(v)} h_u^{(k-1)} \right) $$

Cada pieza justificada por la teoría:
- **La suma** $\sum_{u \in N(v)}$ es el agregador inyectivo sobre multiconjuntos (Lema 5).
- **El término $(1+\epsilon)\cdot h_v$** mezcla el nodo central preservando la inyectividad del par (nodo, vecindario) (Corolario 6). $\epsilon$ puede ser aprendido (**GIN-$\epsilon$**) o fijado a 0 (**GIN-0**).
- **El MLP** modela las funciones universales $f$ y $\varphi$, apoyándose en el teorema de aproximación universal. El Lema 7 demuestra que un perceptrón de 1 capa ($\sigma \circ W$, como el de GCN) *no* puede distinguir ciertos multiconjuntos finitos; por eso GIN insiste en MLP de al menos 2 capas.

Para clasificación de grafos, GIN no usa solo la última capa: adopta una arquitectura tipo *Jumping Knowledge* que concatena la suma de las características de los nodos de *cada* capa, lo que hace que GIN generalice provablemente el test WL *y* el kernel de subárboles WL.

## Por qué suma > promedio > máximo

Este es el resultado que da sentido directo a la elección del agregador. Los tres son funciones de multiconjunto bien definidas (todas permutación-invariantes), pero capturan aspectos *distintos*, ordenados por poder de distinción:

- **Suma — captura el multiconjunto completo.** Es inyectiva (Lema 5). Distingue tanto los elementos distintos como sus multiplicidades.
- **Promedio — captura la distribución/proporción.** El Corolario 8 muestra que dos multiconjuntos $(S, m)$ y $(S, k\cdot m)$ (mismo conjunto, multiplicidades escaladas) reciben el *mismo* embedding. Capta proporciones, no el multiconjunto exacto.
- **Máximo — captura solo el conjunto subyacente.** El Corolario 9 muestra que el max-pooling colapsa un multiconjunto a sus elementos distintos, ignorando del todo las multiplicidades.

Los ejemplos lo vuelven tangible. En un grafo no etiquetado donde todo nodo tiene la característica $a$, promedio y máximo de $f(a)$ siempre devuelven $f(a)$ —no capturan *ninguna* estructura—, mientras la suma distingue $2\cdot f(a)$ de $3\cdot f(a)$ (un nodo con dos vecinos frente a uno con tres). Para $\{$verde, rojo$\}$ vs $\{$verde, verde, rojo, rojo$\}$, tanto promedio como máximo fallan porque $\frac{1}{2}(h_g + h_r) = \frac{1}{4}(h_g + h_g + h_r + h_r)$.

Esto también explica *cuándo* bastan las opciones débiles: el **promedio** funciona bien cuando importa la información distribucional más que la estructura exacta (clasificación de nodos con características ricas, donde los vecindarios raramente repiten); el **máximo** sirve para identificar elementos representativos, robusto a ruido y outliers (PointNet sobre nubes de puntos 3D).

## Experimentos

**Datasets (9 benchmarks de clasificación de grafos).** Cuatro de bioinformática —MUTAG (188 compuestos nitroaromáticos mutagénicos), PROTEINS (estructura secundaria), PTC (carcinogenicidad), NCI1 (4110 compuestos anticáncer)— y cinco de redes sociales —IMDB-BINARY/MULTI (colaboración actoral), REDDIT-BINARY/MULTI5K (hilos de discusión) y COLLAB. Detalle clave: para forzar el aprendizaje *estructural*, en REDDIT *todos* los nodos reciben el mismo escalar (característica no informativa) y en las demás el one-hot del grado. Configuración: 10-fold CV, 5 capas, MLP de 2 capas, batch norm, Adam. Notablemente, **mean–1-capa ≈ GCN** y **max–1-capa ≈ GraphSAGE**.

**Accuracy de entrenamiento como medidor de expresividad.** Es la validación empírica más directa de la teoría: un modelo más expresivo debería *ajustar* mejor el entrenamiento. Exactamente lo observado: GIN-$\epsilon$ y GIN-0 fijan casi perfectamente *todos* los conjuntos de entrenamiento, mientras las variantes con promedio/máximo o perceptrón de 1 capa **subajustan severamente**. El patrón sigue el ranking teórico: MLP > 1-capa, suma > promedio/máximo. Y el accuracy de entrenamiento de las GNN **nunca supera al del kernel WL** —consistente con que WL es la cota superior.

**Accuracy de test.** GIN-0 iguala o supera a las variantes débiles en los 9 datasets, logrando estado del arte. En REDDIT, donde todos los nodos comparten el mismo escalar, las GNN de suma capturan la estructura mientras que **las de promedio no superan al azar** (mean–MLP cae a 71.2% en REDDIT-BINARY y 41.3% en REDDIT-MULTI5K). Valores representativos de GIN-0: IMDB-B 75.1%, RDT-B 92.4%, COLLAB 80.2%, MUTAG 89.4%, NCI1 82.7%. GIN-0 generaliza ligera y consistentemente mejor que GIN-$\epsilon$, probablemente por su mayor simplicidad —aprender $\epsilon$ no aportó ganancia empírica.

## Limitaciones

La limitación fundamental está incorporada en el resultado central: **1-WL es un techo, y 1-WL no es perfecto**. El test WL falla en distinguir ciertos grafos —notablemente algunos **grafos regulares**, donde todos los nodos tienen el mismo grado y vecindarios estructuralmente idénticos. Como GIN iguala pero no supera a 1-WL, *hereda* exactamente esos puntos ciegos: hay pares de grafos no isomorfos que GIN nunca podrá distinguir. Esto motivó toda una línea posterior de GNN *más* poderosas que 1-WL ($k$-WL, GNN con identificadores de nodo, subgraph GNN).

Otras limitaciones reconocidas: la teoría asume un **universo de características contable** (el caso continuo queda para trabajo futuro); el marco caracteriza solo GNN *de paso de mensajes*; agregadores no estándar como atención (GAT) o LSTM-pooling no se analizan; y los resultados son sobre *poder de distinción* (capacidad), no sobre **generalización** —la conexión expresividad → buen test accuracy es plausible y se observa, pero no se demuestra formalmente.

## Impacto

GIN se convirtió en la referencia canónica para razonar sobre el poder expresivo de las GNN. Tres contribuciones perduran:

1. **El lenguaje WL.** "Tan poderosa como 1-WL" pasó a ser la unidad de medida estándar; toda arquitectura nueva se evalúa, al menos conceptualmente, contra esta jerarquía.
2. **La arquitectura GIN.** Baseline omnipresente en *graph classification* y bloque de construcción frecuente, por simple, teóricamente fundamentado y empíricamente fuerte.
3. **El criterio "agregadores inyectivos".** El principio "suma + MLP para máximo poder" guía el diseño de GNN hasta hoy, y abrió el programa de investigación *más allá* de 1-WL.

## Conexión con la Clase 27

GIN aporta el **fundamento teórico** que la [Clase 27](/clases/clase-27) presenta de forma operativa pero sin demostrar:

- **Por qué la "función conmutativa de combinación" importa.** La clase introduce la agregación de vecinos como función permutación-invariante y ofrece suma, promedio y máximo como opciones. GIN demuestra que esta elección **es la decisión de diseño más consecuente para el poder del modelo**: al elegir el agregador se está eligiendo, literalmente, cuántos grafos distintos el modelo podrá distinguir.

- **Complemento con [Barceló (Logical Expressiveness)](/papers/logical-expressiveness-barcelo-2020),** citado en los créditos de la clase. Ambos responden a la misma pregunta desde ángulos complementarios: GIN da una caracterización **algorítmica** (las GNN de paso de mensajes equivalen a 1-WL); Barceló et al. una caracterización **lógica** (capturan las propiedades expresables en $\text{FOC}_2$, fragmento de lógica de primer orden con contadores, estrechamente ligado a 1-WL). La lección unificada: las GNN de paso de mensajes son potentes pero tienen un límite *preciso y demostrable*, propiedad estructural del paradigma de agregación de vecinos.

- **Del concepto a la arquitectura.** Si la clase muestra [GCN](/papers/gcn-kipf-2017) como ejemplo de GNN, GIN explica qué le falta a GCN (su promedio + 1-capa lo hacen estrictamente menos expresivo que 1-WL) y cómo arreglarlo (suma + MLP + readout por suma sobre capas). Es el puente entre "aquí hay una GNN que funciona" y "aquí está la GNN provablemente más poderosa de su clase, con la prueba".
