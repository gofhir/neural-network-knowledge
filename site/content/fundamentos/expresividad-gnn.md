---
title: "Expresividad de las GNN (WL y lógica)"
weight: 102
math: true
---

Las **redes neuronales de grafos** (GNN) de paso de mensajes funcionan: alcanzan el estado del arte en clasificación de moléculas, predicción de enlaces y ranking. Pero durante años faltó una respuesta a una pregunta básica: ¿qué pueden y qué **no** pueden distinguir? Dos arquitecturas que agregan vecindarios pueden parecer equivalentes y, sin embargo, una distinguir grafos que la otra confunde irremediablemente. Este fundamento traza la frontera exacta del **poder expresivo** de las GNN de paso de mensajes, desde dos perspectivas que terminan siendo la misma moneda: la **combinatoria** —el test de isomorfismo de Weisfeiler-Lehman, formalizada por [GIN (Xu et al. 2019)](/papers/gin-xu-2019)— y la **lógica** —la lógica de primer orden con conteo, formalizada por [Barceló et al. (2020)](/papers/logical-expressiveness-barcelo-2020), un trabajo de la PUC Chile—. El resultado es un mapa preciso de las capacidades del paradigma que se presenta en la [Clase 27](/clases/clase-27), y una guía concreta para diseñar agregadores y features. Conviene tener antes a mano los fundamentos de [redes neuronales de grafos](/fundamentos/redes-neuronales-de-grafos) y de [message passing](/fundamentos/message-passing).

---

## 1. La pregunta de expresividad: distinguir grafos no isomorfos

Una GNN para clasificar grafos aprende una función $\mathcal{A}: \mathcal{G} \to \mathbb{R}^d$ que mapea cada grafo a un vector (su *embedding*). Si dos grafos $G_1$ y $G_2$ son **isomorfos** —idénticos salvo el renombre de sus nodos—, deben recibir el mismo embedding: la red debe ser invariante a permutaciones. La pregunta inversa es la interesante: si $G_1$ y $G_2$ **no** son isomorfos (son genuinamente distintos), ¿les asigna la GNN embeddings distintos? Si los colapsa al mismo vector, ningún clasificador posterior podrá separarlos jamás, por más que se entrene. La expresividad es, entonces, la **capacidad de distinción**: cuántos pares de grafos no isomorfos la arquitectura puede separar.

Por qué importa, en términos prácticos: si una molécula tóxica y una inocua tienen estructuras distintas que la GNN colapsa al mismo embedding, el modelo está condenado a errar en ambas a la vez. El poder expresivo es un **techo duro** sobre el accuracy alcanzable, independiente de los datos, del optimizador o de cuántas épocas se entrene. Una GNN poco expresiva no es un problema de entrenamiento: es un problema de diseño que ninguna cantidad de datos resuelve.

{{< concept-alert type="clave" >}}
La expresividad de una GNN se mide por su **poder de distinción**: ¿puede asignar embeddings distintos a grafos no isomorfos distintos? Es una propiedad **arquitectónica**, no de entrenamiento. Si dos estructuras colapsan al mismo vector, el límite es infranqueable por más datos que haya. Toda la teoría de esta página establece exactamente *dónde* está ese techo.
{{< /concept-alert >}}

---

## 2. El test de Weisfeiler-Lehman (1-WL) y el refinamiento de colores

El **test de Weisfeiler-Lehman de una dimensión** (1-WL), de 1968, es un algoritmo clásico de teoría de grafos para decidir (heurísticamente) si dos grafos son isomorfos. Procede por **refinamiento de colores** (color refinement):

1. **Inicialización.** Cada nodo recibe un color inicial (su etiqueta, o un color uniforme si el grafo no está etiquetado).
2. **Refinamiento.** En cada ronda, cada nodo construye un nuevo color a partir de un par: su color actual y el **multiconjunto** de colores de sus vecinos. Se aplica una función de *hashing* inyectiva: vecindarios distintos producen colores nuevos distintos.
3. **Iteración.** Se repite hasta que la partición de colores se estabiliza.
4. **Decisión.** Dos grafos se declaran no isomorfos si, en alguna ronda, sus **histogramas de colores** difieren. (Si coinciden siempre, el test no puede distinguirlos: puede tratarse de grafos isomorfos o de un par que 1-WL no separa.)

La pieza clave es el **multiconjunto** (multiset): un conjunto que admite repeticiones, $X = (S, m)$ con $S$ los elementos distintos y $m: S \to \mathbb{N}_{\geq 1}$ sus multiplicidades. El vecindario de un nodo es genuinamente un multiconjunto, porque dos vecinos distintos pueden tener el mismo color. Tras $k$ rondas, el color de un nodo codifica su **subárbol enraizado de altura $k$**: toda la estructura del vecindario a $k$ saltos.

La conexión con las GNN es que **1-WL y el message passing son el mismo proceso**. La actualización de una GNN,

$$
h_v^{(k)} = \varphi\Big(h_v^{(k-1)},\; f\big(\{\!\!\{\, h_u^{(k-1)} : u \in N(v) \,\}\!\!\}\big)\Big),
$$

es estructuralmente idéntica a una ronda de refinamiento de colores: agregar el multiconjunto de vecinos y combinarlo con el estado propio. Esta analogía, observada por [Xu et al.](/papers/gin-xu-2019) y por Morris et al. en paralelo (2019), es el corazón de toda la teoría. De ella se deriva el rol de 1-WL como **cota superior**: como veremos, ninguna GNN de paso de mensajes puede distinguir más grafos que 1-WL.

---

## 3. GIN: igualar el techo con suma + MLP

[Xu et al. (2019)](/papers/gin-xu-2019) formalizaron la analogía en dos teoremas.

**Cota superior.** Para todo par de grafos $G_1, G_2$ no isomorfos, si una GNN de paso de mensajes los mapea a embeddings distintos, entonces 1-WL también los declara no isomorfos. En consecuencia, **ninguna GNN de agregación de vecindarios es más poderosa que 1-WL**. El test es un techo para toda la familia. La intuición de la demostración: si dos nodos comparten color WL en cada ronda, la misma agregación y combinación aplicadas a la misma entrada producen el mismo estado en la GNN; la igualdad se hereda hasta el nivel del grafo.

**Alcanzar el techo.** Una GNN *iguala* el poder de 1-WL —distingue todo lo que 1-WL distingue— si y solo si, con suficientes capas, tanto la función de agregación sobre el multiconjunto de vecinos como el *readout* final que agrega los nodos del grafo son **inyectivos**: mapean multiconjuntos distintos a salidas distintas. Una agregación inyectiva nunca confunde dos vecindarios diferentes.

¿Cómo se construye un agregador inyectivo sobre multiconjuntos? Extendiendo *Deep Sets* a multiconjuntos, Xu et al. demuestran que la **suma** lo logra: existe una codificación $f$ tal que $h(X) = \sum_{x \in X} f(x)$ es único para cada multiconjunto $X$ acotado, y cualquier función de multiconjunto se descompone como $\varphi\big(\sum_{x\in X} f(x)\big)$. La suma es, de hecho, un agregador **universal** sobre multiconjuntos. Incorporando el nodo central de forma que preserve la inyectividad del par (nodo, vecindario), nace la regla de actualización de **GIN** (Graph Isomorphism Network):

$$
h_v^{(k)} = \text{MLP}^{(k)}\!\left( \big(1 + \epsilon^{(k)}\big)\cdot h_v^{(k-1)} + \sum_{u \in N(v)} h_u^{(k-1)} \right).
$$

Cada pieza está justificada por la teoría: la **suma** es el agregador inyectivo; el término $(1+\epsilon)\,h_v$ mezcla el nodo central preservando la inyectividad del par ($\epsilon$ puede aprenderse —**GIN-$\epsilon$**— o fijarse a 0 —**GIN-0**—); el **MLP** modela las funciones universales (un perceptrón de una sola capa no basta: existen multiconjuntos finitos que ningún mapeo lineal seguido de no linealidad puede distinguir).

### Por qué suma > promedio > máximo

Los tres agregadores son permutación-invariantes y válidos, pero capturan aspectos distintos del multiconjunto, ordenados por poder:

- **Suma — captura el multiconjunto completo.** Inyectiva. Distingue elementos *y* multiplicidades.
- **Promedio (mean) — captura solo la distribución/proporción.** Dos multiconjuntos con el mismo conjunto de elementos pero multiplicidades escaladas ($X$ y $kX$) reciben el mismo embedding. Captura proporciones, no cantidades.
- **Máximo (max) — captura solo el conjunto subyacente.** Colapsa el multiconjunto a sus elementos distintos, ignorando del todo las multiplicidades.

Los ejemplos concretos lo vuelven tangible. En un grafo no etiquetado donde todo nodo tiene la misma característica $a$: promedio y máximo de $f(a)$ siempre devuelven $f(a)$ —no capturan estructura alguna—, mientras que la suma distingue un nodo con dos vecinos ($2\,f(a)$) de uno con tres ($3\,f(a)$). Para los vecindarios $\{\!\!\{\text{verde}, \text{rojo}\}\!\!\}$ y $\{\!\!\{\text{verde}, \text{rojo}, \text{rojo}\}\!\!\}$, el máximo da $\max(h_g, h_r)$ en ambos casos y los colapsa, pero la suma los separa. Y ni promedio ni máximo distinguen $\{\!\!\{\text{verde}, \text{rojo}\}\!\!\}$ de $\{\!\!\{\text{verde}, \text{verde}, \text{rojo}, \text{rojo}\}\!\!\}$, porque $\tfrac{1}{2}(h_g+h_r) = \tfrac{1}{4}(2h_g + 2h_r)$.

| Agregador | Captura | Inyectivo sobre multisets | GNN típica |
|---|---|---|---|
| **Suma** | multiconjunto completo (elementos + multiplicidades) | Sí | GIN |
| **Promedio** | distribución / proporciones | No | GCN |
| **Máximo** | conjunto subyacente (solo elementos) | No | GraphSAGE-pool |

Empíricamente, GIN ajusta casi perfectamente el conjunto de entrenamiento de 9 benchmarks de clasificación de grafos, mientras que las variantes con promedio o máximo subajustan. En REDDIT, donde todos los nodos comparten el mismo escalar y solo importa la estructura, las GNN de promedio **no superan al azar**, justo como predice la teoría. Esto explica por qué GCN (promedio) y GraphSAGE (máximo) son estrictamente menos expresivos que 1-WL.

{{< concept-alert type="importante" >}}
La elección del agregador **no es cosmética**: determina cuántos grafos distintos el modelo podrá separar. Suma > promedio > máximo en poder de distinción. Las opciones débiles bastan cuando importa la estadística del vecindario (promedio, útil en clasificación de nodos con features ricas) o solo el "esqueleto" robusto a outliers (máximo). Pero para capturar estructura exacta en grafos pobres en features, hay que usar **suma + MLP**.
{{< /concept-alert >}}

---

## 4. La caracterización lógica de Barceló (PUC Chile)

GIN responde la pregunta del **poder discriminativo** (distinguir dos nodos o grafos). Pero hay una pregunta distinta: ¿qué **clasificadores booleanos concretos** —funciones que asignan verdadero/falso a cada nodo— puede *expresar* una GNN? [Barceló et al. (2020)](/papers/logical-expressiveness-barcelo-2020) la abordan cambiando de lente: en vez de medir contra un algoritmo combinatorio, miden contra la **lógica de primer orden**. Es un trabajo "de la casa": cinco de sus seis autores trabajan en Chile —entre la **PUC**, la **Universidad de Chile** y el **IMFD** (Instituto Milenio Fundamentos de los Datos)—, y **Pablo Barceló** y **Jorge Pérez** son figuras centrales del grupo chileno de fundamentos teóricos de la IA. Jorge Pérez aparece citado en los créditos de la [Clase 27](/clases/clase-27): el material de la clase proviene del propio coautor.

El paper bautiza la GNN básica de message passing como **AC-GNN** (*aggregate-combine*) y la mide contra **FOC2**: el fragmento de la lógica de primer orden que permite solo **dos variables** pero añade **cuantificadores de conteo** $\exists^{\geq N}$ ("existen al menos $N$ nodos que satisfacen $\varphi$"). FOC2 es estrictamente menos expresiva que toda la FO, pero el conteo recupera mucho poder. La elección no es arbitraria: el resultado clásico de **Cai, Fürer e Immerman (1992)** establece que **WL y FOC2 son dos caras de la misma moneda discriminativa** —el test WL colorea dos nodos igual tras cualquier número de rondas *si y solo si* todos los clasificadores FOC2 los tratan igual—. Ese teorema es el puente que conecta el enfoque combinatorio de GIN con el lógico de Barceló.

El hallazgo fino del paper es que **distinguir** y **capturar** son cosas distintas. Que WL refine la clasificación de una AC-GNN no implica que la AC-GNN pueda expresar todo clasificador FOC2. De hecho:

- **Las AC-GNN no capturan todo FOC2.** El testigo más simple: $\alpha(x) := \text{Rojo}(x) \wedge \exists y\,\text{Verde}(y)$ ("$x$ es rojo y el grafo contiene algún nodo verde"). Ninguna AC-GNN lo captura. La razón es la **localidad**: una AC-GNN de $L$ capas no propaga información más allá de distancia $L$ a lo largo de las aristas. El nodo verde puede estar más lejos que $L$ —incluso en otra componente conexa, donde *ningún* número de capas alcanza—, de modo que el nodo rojo nunca "se entera" de su existencia.

- **Las AC-GNN capturan exactamente la lógica modal graduada.** El fragmento de FOC2 que las AC-GNN sí capturan, y *exactamente* ese (un "si y solo si"), es la **lógica modal graduada** (equivalente a la lógica de descripción **ALCQ**, corazón del lenguaje de ontologías OWL 2). Su característica es que toda subfórmula está **guardada por la arista**: no se puede preguntar "¿hay algún nodo verde?", solo "¿tengo al menos $N$ *vecinos* verdes?", es decir $\exists^{\geq N} y\,(E(x,y) \wedge \varphi(y))$. La construcción es instructiva y conecta con la clase: cada **dimensión** del vector de features representa una **subfórmula**, y con suma como agregación más una ReLU truncada $\sigma(x)=\min(\max(0,x),1)$ se evalúa cada conectivo (conjunción, negación, cuantificador graduado) capa a capa.

- **Las ACR-GNN (con readout global) sí capturan todo FOC2.** Como el defecto es la localidad, la cura es un **cómputo global**: en cada capa, un *readout* que agrega los vectores de **todos** los nodos del grafo. La **ACR-GNN** (*aggregate-combine-readout*) añade esa lectura global y captura todo FOC2. Volviendo al ejemplo: el readout *cuenta* cuántos nodos verdes hay en el grafo entero, rompiendo la barrera de la distancia. Basta incluso con un único readout final (AC-FR-GNN) para igualar el poder expresivo, aunque en la práctica varios readouts facilitan el aprendizaje.

{{< concept-alert type="ojo" >}}
Regla mental para el practicante: si la propiedad que quiero clasificar es **local y guardada por aristas** ("tengo $\geq 2$ vecinos azules"), una GNN estándar basta. Si involucra cuantificadores **no-locales** (sobre no-vecinos, conteos globales, propiedades entre componentes desconectadas), necesito un **readout global**. Apilar más capas no resuelve lo no-local: amplía el campo receptivo hasta distancia $L$, pero entre componentes desconectadas ningún $L$ alcanza.
{{< /concept-alert >}}

---

## 5. Dos perspectivas complementarias: combinatoria y lógica

GIN y Barceló responden la misma pregunta —*¿qué pueden y qué no pueden hacer las GNN?*— desde ángulos que se refuerzan. GIN da una caracterización **algorítmica** (las GNN de paso de mensajes equivalen a 1-WL); Barceló da una caracterización **lógica** (las AC-GNN capturan exactamente la lógica modal graduada; con readout, FOC2). El teorema de Cai-Fürer-Immerman (WL $\leftrightarrow$ FOC2) garantiza que ambas hablan de lo mismo.

| Eje | Perspectiva combinatoria (GIN) | Perspectiva lógica (Barceló) |
|---|---|---|
| Vara de medida | Test 1-WL (refinamiento de colores) | Lógica FOC2 (FO con 2 variables + conteo) |
| Pregunta | Poder **discriminativo**: ¿distingue dos grafos? | Poder **expresivo**: ¿captura este clasificador? |
| GNN básica | Message passing $\leq$ 1-WL; GIN $=$ 1-WL | AC-GNN $=$ lógica modal graduada ($\subsetneq$ FOC2) |
| Receta para el techo | Suma + MLP (agregador inyectivo) | + Readout global $\Rightarrow$ FOC2 completo |
| Foco | Clasificación de **grafos** | Clasificación de **nodos** |
| Puente | Cai-Fürer-Immerman | 1-WL $\leftrightarrow$ FOC2 |

La lección unificada: las GNN de paso de mensajes son potentes pero tienen un límite **preciso y demostrable**, que no es un detalle de implementación sino una propiedad estructural del paradigma de agregación de vecinos.

---

## 6. Implicaciones prácticas

La teoría se traduce en decisiones de diseño concretas:

- **Elegir el agregador con intención.** Para capturar estructura exacta en grafos con pocas features (moléculas no etiquetadas, redes sin atributos), usar **suma + MLP** (GIN). El promedio basta cuando importa la distribución del vecindario y este rara vez repite features (clasificación de nodos con atributos ricos: temas de artículos, comunidades); el máximo, para identificar elementos representativos robustos a ruido.

- **Añadir features o codificaciones posicionales.** El techo de 1-WL es un límite sobre **grafos**, no sobre grafos enriquecidos. Inyectar identificadores de nodo, **inicialización aleatoria de nodos** (random node init) o codificaciones posicionales/estructurales rompe simetrías que 1-WL no ve, permitiendo superar la cota. Es la vía estándar para ir "más allá de 1-WL" sin cambiar el paradigma de message passing.

- **Cuidado con los grafos regulares.** El punto ciego más conocido de 1-WL —que GIN *hereda*— son ciertos **grafos regulares**: todos los nodos tienen el mismo grado y vecindarios estructuralmente idénticos, de modo que el refinamiento de colores nunca los separa. Hay pares de grafos no isomorfos (por ejemplo, dos grafos 3-regulares con distinto número de triángulos o ciclos) que ninguna GNN de paso de mensajes puede distinguir. Si el dominio tiene esa regularidad, hay que enriquecer features o usar arquitecturas más expresivas (k-WL, subgraph GNN).

- **Usar readout global para propiedades no-locales.** Si la tarea exige conteos globales o relaciones entre nodos distantes/desconectados, un *readout* por capa (estilo ACR-GNN) es la respuesta arquitectónica, no más capas.

- **No olvidar que la teoría es de capacidad, no de generalización.** Mayor expresividad implica mayor capacidad de *ajustar* datos, no necesariamente mejor accuracy de test. En benchmarks reales como PPI, la ventaja teórica de ACR sobre AC puede quedar latente si las propiedades no-locales no son discriminantes en ese dataset.

---

## 7. Conexión con la Clase 27 y el mecanismo de combinación

La [Clase 27](/clases/clase-27) presenta el **message passing** de forma operativa: cada nodo agrega los vectores de sus vecinos con una **función conmutativa de combinación** —suma, promedio o máximo— y combina el resultado con su propio estado. Esta página es la fundamentación teórica de ese gesto:

- **Por qué la función de combinación importa.** La clase ofrece suma/promedio/máximo como opciones de diseño. GIN demuestra que es la decisión **más consecuente** para el poder del modelo: al elegir el agregador, se elige —literalmente— cuántos grafos distintos la red podrá distinguir. La suma es inyectiva (captura todo); el promedio solo la distribución; el máximo solo el conjunto.

- **Por qué el pooling global importa.** La clase clasifica grafos con *mean pooling*, $g = \tfrac{1}{N}\sum_v h_v$, una agregación global al final. El componente *readout* de las ACR-GNN es exactamente esa idea de mirar el grafo entero, pero usada de forma **intermedia** en cada capa. Barceló muestra que ese mismo gesto es lo que convierte una GNN local en una capaz de expresar propiedades globales (FOC2 completo). Es la justificación teórica de por qué el pooling global no es solo un truco de agregación final.

- **Por qué apilar capas no basta.** La clase muestra que más capas amplían el campo receptivo. Esta página delimita el límite duro: con $L$ capas la información local no viaja más allá de distancia $L$, y entre componentes desconectadas *ningún* número de capas alcanza. El readout es la respuesta a esa barrera.

- **Producción local.** Que el material teórico de la clase provenga de Jorge Pérez y del grupo PUC/UChile/IMFD demuestra que el aporte fundacional —no solo aplicado— a las GNN salió de universidades chilenas, con un resultado publicado en ICLR y enseñado internacionalmente.

---

## 8. Resumen

El poder expresivo de las GNN de paso de mensajes está **completamente caracterizado** desde dos ángulos equivalentes. Combinatoriamente ([GIN](/papers/gin-xu-2019)): toda GNN de agregación es a lo más tan poderosa como el test **1-WL**, y GIN alcanza ese techo con un agregador **inyectivo** —suma + MLP—, estrictamente más expresivo que promedio (solo distribución) o máximo (solo conjunto). Lógicamente ([Barceló, PUC Chile](/papers/logical-expressiveness-barcelo-2020)): las AC-GNN capturan **exactamente** la lógica modal graduada, un fragmento estricto de FOC2; no capturan propiedades no-locales hasta que se añade un **readout global**, que las eleva a todo FOC2. El teorema de Cai-Fürer-Immerman (1-WL $\leftrightarrow$ FOC2) une ambas perspectivas. En la práctica, esto guía la elección del agregador (suma para estructura exacta), el enriquecimiento con features/posicionales para superar el techo de 1-WL, la cautela frente a grafos regulares, y el uso de readout para propiedades globales —el corazón de los mecanismos de combinación que presenta la [Clase 27](/clases/clase-27).

---

## Para profundizar

- [How Powerful are Graph Neural Networks? (Xu et al. 2019)](/papers/gin-xu-2019) — la caracterización combinatoria: GNN $\leq$ 1-WL, y GIN alcanza el techo con suma + MLP.
- [The Logical Expressiveness of Graph Neural Networks (Barceló et al. 2020)](/papers/logical-expressiveness-barcelo-2020) — la caracterización lógica desde la PUC Chile: AC-GNN $=$ lógica modal graduada; ACR-GNN $\supseteq$ FOC2.

**Fundamentos relacionados:** [Redes Neuronales de Grafos](/fundamentos/redes-neuronales-de-grafos) · [Message Passing](/fundamentos/message-passing) · [Clase 27](/clases/clase-27)
