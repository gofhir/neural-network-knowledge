# How Powerful are Graph Neural Networks? (GIN) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *How Powerful are Graph Neural Networks?* (presenta la arquitectura **Graph Isomorphism Network, GIN**).
- **Autores:** Keyulu Xu (MIT), Weihua Hu (Stanford University), Jure Leskovec (Stanford University), Stefanie Jegelka (MIT). Xu y Hu con contribución igualitaria; parte del trabajo se realizó durante estadías en RIKEN AIP y la Universidad de Tokio (Prof. Ken-ichi Kawarabayashi).
- **Venue:** *International Conference on Learning Representations* (**ICLR 2019**), conference paper.
- **Preprint:** arXiv:1810.00826v3 (22 feb 2019), [arxiv.org/abs/1810.00826](https://arxiv.org/abs/1810.00826).
- **Código:** [github.com/weihua916/powerful-gnns](https://github.com/weihua916/powerful-gnns).

**Aclaración importante sobre el rol en la Clase 27.** Este es un paper *canónico* del campo de las Redes Neuronales de Grafos, pero **no está citado explícitamente en las slides de la clase**. Lo incluimos porque es la pieza teórica fundacional que responde a la pregunta que la clase deja implícita: *¿qué tan poderosas son realmente las GNN, y qué decide ese poder?* Junto con el trabajo de Barceló et al. sobre *Logical Expressiveness* (que la clase sí menciona en los créditos), GIN forma el par de referencias teóricas que delimitan qué pueden y qué no pueden distinguir las GNN de paso de mensajes. Mientras Barceló razona en términos de *lógica de primer orden con contadores*, GIN razona en términos del *test de isomorfismo de Weisfeiler-Lehman*; ambos llegan a fronteras de expresividad estrechamente relacionadas.

El paper hace algo que en 2018 faltaba en el campo: dar **teoría** a un éxito que hasta entonces era puramente empírico. Las GNN —GCN (Kipf & Welling, 2017), GraphSAGE (Hamilton et al., 2017a), GAT (Velickovic et al., 2018)— ya alcanzaban el estado del arte en clasificación de nodos, predicción de enlaces y clasificación de grafos, pero su diseño se basaba "en intuición empírica, heurísticas y prueba y error experimental". No había una caracterización formal de *qué estructuras de grafo* una GNN dada puede o no puede distinguir. El paper provee esa caracterización: (1) demuestra que las GNN de paso de mensajes son **a lo más tan poderosas como el test de Weisfeiler-Lehman de 1 dimensión (1-WL)** para distinguir grafos; (2) establece las condiciones bajo las cuales una GNN *alcanza* ese límite (agregadores y readout inyectivos); (3) identifica estructuras concretas que GCN y GraphSAGE no pueden distinguir; y (4) diseña **GIN**, una arquitectura simple que provablemente alcanza el poder del test WL y que empíricamente fija casi perfectamente los datos de entrenamiento y rinde estado del arte.

Para la Clase 27 esto importa de forma muy concreta: la clase introduce la "función conmutativa de combinación" (agregación de vecinos) como una elección de diseño entre suma, promedio o máximo. GIN demuestra que esa elección **no es cosmética**: determina el poder expresivo del modelo. La suma es estrictamente más poderosa que el promedio, y el promedio más que el máximo. Entender por qué es entender la lección central de este paper.

## 2. Contexto histórico: GNN que funcionaban sin teoría que las explicara

Hacia 2017–2018, el aprendizaje sobre datos estructurados en grafos —moléculas, redes sociales, biológicas y financieras— vivía un auge. La receta común quedó cristalizada bajo el nombre de *neighborhood aggregation* o *message passing* (Gilmer et al., 2017; Xu et al., 2018): cada nodo actualiza recursivamente su vector de características agregando los de sus vecinos. Tras $k$ iteraciones, la representación de un nodo captura información estructural de su vecindario a $k$ saltos; sumando (o agrupando) las representaciones de todos los nodos se obtiene la representación del grafo entero. Bajo esta plantilla cabían GCN, GraphSAGE, GAT, Gated Graph NN, redes de interacción, y más de una docena de variantes que el paper enumera.

El problema era epistemológico: estas redes *funcionaban*, pero nadie sabía *por qué unas funcionaban mejor que otras*, ni cuáles eran sus límites fundamentales. El paper lo formula sin rodeos: "hay poco entendimiento teórico de las propiedades y limitaciones de las GNN, y el análisis formal de su capacidad representacional es limitado". Las excepciones previas eran escasas y específicas: Scarselli et al. (2009a) mostraron que el modelo GNN más temprano podía aproximar funciones medibles en probabilidad; Lei et al. (2017) ubicaron su arquitectura en el RKHS de ciertos kernels de grafos, pero sin estudiar qué grafos podía distinguir. Ninguno daba un marco general aplicable a *múltiples* arquitecturas.

El aporte conceptual de Xu et al. fue notar una analogía profunda y aprovecharla: el paso de mensajes de las GNN es **estructuralmente idéntico** al test de isomorfismo de **Weisfeiler-Lehman** (Weisfeiler & Lehman, 1968), un algoritmo clásico de teoría de grafos que también actualiza iterativamente la etiqueta de cada nodo agregando las etiquetas de sus vecinos. El test WL es notablemente potente —distingue una amplia clase de grafos (Babai & Kucera, 1979)— y lo que lo hace potente es precisamente su **agregación inyectiva**: mapea vecindarios distintos a etiquetas distintas. La intuición clave del paper: *una GNN puede ser tan poderosa como el test WL si y solo si su esquema de agregación es altamente expresivo y puede modelar funciones inyectivas*. Esa intuición es el eje de todo lo que sigue.

## 3. Contribución central: conectar GNN con Weisfeiler-Lehman

### 3.1. El multiconjunto como abstracción

El primer movimiento técnico es representar el vecindario de un nodo no como un conjunto ni como una secuencia, sino como un **multiconjunto** (multiset): un conjunto que admite elementos repetidos. Formalmente (Definición 1 del paper), un multiconjunto es un par $X = (S, m)$ donde $S$ es el conjunto subyacente de elementos distintos y $m : S \to \mathbb{N}_{\geq 1}$ da la multiplicidad de cada elemento. Esta abstracción es exacta: distintos nodos vecinos pueden tener vectores de características *idénticos*, de modo que el vecindario es genuinamente un multiconjunto, no un conjunto.

Con esta lente, la agregación de vecinos de una GNN es **una función sobre multiconjuntos**. Y la pregunta sobre el poder expresivo se reduce a una pregunta sobre funciones de multiconjuntos: *¿puede el agregador mapear multiconjuntos distintos a representaciones distintas?* Una GNN máximamente poderosa nunca mapearía dos vecindarios distintos (dos multiconjuntos distintos) a la misma representación; es decir, su agregación debe ser **inyectiva**. El paper asume a lo largo del texto que las características de entrada provienen de un universo *contable*, lo que es razonable para grafos finitos y permite tratar la inyectividad rigurosamente (Lema 4 demuestra que el rango de cada capa permanece contable).

### 3.2. Los dos teoremas fundamentales

**Cota superior (Lema 2).** Sean $G_1$ y $G_2$ dos grafos no isomorfos. Si una GNN $\mathcal{A}: \mathcal{G} \to \mathbb{R}^d$ los mapea a *embeddings* distintos, entonces el test WL también decide que son no isomorfos. **Consecuencia: ninguna GNN basada en agregación puede ser más poderosa que el test WL.** El test WL es un *techo* para toda esta familia de modelos. La demostración (Apéndice A) es por inducción: si dos nodos reciben la misma etiqueta WL en cada iteración, reciben también la misma característica GNN, porque la misma AGGREGATE y COMBINE aplicadas a la misma entrada producen la misma salida; el readout permutación-invariante hereda esa igualdad al nivel del grafo.

**Alcanzar la cota (Teorema 3).** Una GNN $\mathcal{A}$ mapea a *embeddings distintos* cualquier par de grafos que WL declara no isomorfos —es decir, **iguala el poder de WL**— siempre que, con suficientes capas, se cumplan dos condiciones:

- (a) La actualización de nodos $h_v^{(k)} = \varphi\big(h_v^{(k-1)}, f(\{h_u^{(k-1)} : u \in N(v)\})\big)$ usa funciones $f$ (sobre el multiconjunto de vecinos) y $\varphi$ que son **inyectivas**.
- (b) El readout a nivel de grafo, que opera sobre el multiconjunto $\{h_v^{(k)}\}$, es **inyectivo**.

La demostración (Apéndice B) construye por inducción una función inyectiva $\varphi$ que relaciona las características GNN con las etiquetas WL en cada iteración. La intuición visual del paper (Figura 1) es elegante: tras $k$ iteraciones, la etiqueta WL de un nodo representa un *subárbol enraizado de altura $k$*; si la agregación de la GNN captura el multiconjunto completo de vecinos, captura recursivamente esos subárboles y por tanto iguala a WL.

Una ventaja conceptual que el paper destaca: las características WL son esencialmente *one-hot* y no capturan similitud entre subárboles, mientras que una GNN que satisface el Teorema 3 *generaliza* WL al embeber esos subárboles en un espacio de baja dimensión, mapeando estructuras *similares* a embeddings *similares*. Esto ayuda a la generalización cuando la co-ocurrencia de subárboles es escasa o hay ruido (Yanardag & Vishwanathan, 2015).

### 3.3. GIN: alcanzar la inyectividad con suma + MLP

¿Cómo construir agregadores inyectivos sobre multiconjuntos? El paper desarrolla una "teoría de multiconjuntos profundos" (*deep multisets*), extendiendo a multiconjuntos el resultado de *Deep Sets* (Zaheer et al., 2017).

**Lema 5.** Si $\mathcal{X}$ es contable, existe $f: \mathcal{X} \to \mathbb{R}^n$ tal que $h(X) = \sum_{x \in X} f(x)$ es **único para cada multiconjunto** $X$ de tamaño acotado; y *cualquier* función de multiconjunto $g$ se descompone como $g(X) = \varphi\big(\sum_{x \in X} f(x)\big)$. En otras palabras, **la suma puede representar funciones inyectivas —de hecho, universales— sobre multiconjuntos**. La distinción crucial frente a los conjuntos: ciertas funciones de conjunto inyectivas, como el promedio, *no* son inyectivas sobre multiconjuntos.

**Corolario 6.** Para incorporar también el nodo central $c$, existe $f$ tal que, para infinitos valores de $\epsilon$ (incluidos todos los irracionales), $h(c, X) = (1+\epsilon)\cdot f(c) + \sum_{x \in X} f(x)$ es único para cada par $(c, X)$. La demostración (Apéndice E) es ingeniosa: si $\epsilon$ es irracional, el lado izquierdo de la ecuación de colisión es irracional mientras el derecho es una suma finita de racionales, una contradicción.

De aquí sale la **regla de actualización de GIN**:

$$ h_v^{(k)} = \text{MLP}^{(k)}\!\left( \big(1 + \epsilon^{(k)}\big) \cdot h_v^{(k-1)} + \sum_{u \in N(v)} h_u^{(k-1)} \right) $$

Las piezas, justificadas una a una por la teoría:
- **La suma** $\sum_{u \in N(v)}$ es el agregador inyectivo sobre multiconjuntos (Lema 5).
- **El término $(1+\epsilon)\cdot h_v$** mezcla el nodo central preservando la inyectividad del par (nodo, vecindario) (Corolario 6). $\epsilon$ puede ser un parámetro aprendido (variante **GIN-$\epsilon$**) o fijado a 0 (variante **GIN-0**).
- **El MLP** modela las funciones universales $f$ y $\varphi$ del Corolario 6, apoyándose en el teorema de aproximación universal (Hornik et al., 1989; Hornik, 1991). En la práctica, un solo MLP modela la composición $f^{(k+1)} \circ \varphi^{(k)}$, ya que los MLP representan composiciones de funciones. En la primera iteración, si las entradas son one-hot, no hace falta MLP antes de la suma porque la suma sola ya es inyectiva.

El paper aclara que GIN es *un* ejemplo entre muchas GNN máximamente poderosas posibles; su mérito es ser simple y provablemente óptima.

### 3.4. Readout a nivel de grafo

Para clasificación de grafos, GIN no usa solo la última capa. Observando que las representaciones de iteraciones tempranas a veces generalizan mejor (capturan estructura más local), adopta una arquitectura tipo *Jumping Knowledge Networks* (Xu et al., 2018): concatena, a través de todas las iteraciones, la suma de las características de los nodos de cada capa:

$$ h_G = \text{CONCAT}\!\Big( \text{READOUT}\big(\{h_v^{(k)} : v \in G\}\big) \,\Big|\, k = 0, 1, \ldots, K \Big) $$

Usar **suma** como READOUT (sin MLP extra, por el mismo argumento del Lema 5) hace que GIN generalice provablemente el test WL *y* el kernel de subárboles WL (Shervashidze et al., 2011).

## 4. Por qué suma > promedio > máximo

Este es el resultado que da sentido directo a la sección de la clase sobre la elección del agregador. Los tres son funciones de multiconjunto bien definidas (todos son permutación-invariantes), pero capturan aspectos *distintos* del multiconjunto, ordenados por poder de distinción (Figura 2 del paper):

- **Suma — captura el multiconjunto completo.** Es inyectiva (Lema 5). Distingue tanto los elementos distintos como sus multiplicidades.
- **Promedio — captura la distribución/proporción.** El Corolario 8 demuestra que dos multiconjuntos $X_1 = (S, m)$ y $X_2 = (S, k\cdot m)$ (mismo conjunto, multiplicidades escaladas por $k$) reciben el *mismo* embedding bajo el promedio. El promedio captura las *proporciones* de cada tipo de elemento, pero no el multiconjunto exacto.
- **Máximo — captura solo el conjunto subyacente.** El Corolario 9 demuestra que el max-pooling colapsa un multiconjunto a su conjunto de elementos distintos, ignorando por completo las multiplicidades. Trata el multiconjunto como un simple conjunto.

Los ejemplos concretos del paper (Figura 3) lo vuelven tangible. **(a)** En un grafo no etiquetado donde todo nodo tiene la característica $a$: promedio y máximo de $f(a)$ siempre devuelven $f(a)$, por lo que no capturan *ninguna* estructura; en cambio, la suma distingue $2\cdot f(a)$ de $3\cdot f(a)$ (un nodo con dos vecinos frente a uno con tres). **(b)** Para vecindarios $\{$verde, rojo$\}$ y $\{$verde, rojo, rojo$\}$: el máximo da $\max(h_g, h_r)$ en ambos casos —colapsan—, pero la suma los distingue. **(c)** Tanto promedio como máximo fallan en distinguir $\{$verde, rojo$\}$ de $\{$verde, verde, rojo, rojo$\}$ porque $\frac{1}{2}(h_g + h_r) = \frac{1}{4}(h_g + h_g + h_r + h_r)$.

El paper también caracteriza *cuándo* las opciones débiles bastan, lo que explica su éxito empírico previo: el **promedio** funciona bien cuando la información estadística/distribucional importa más que la estructura exacta —típico en clasificación de nodos con características ricas (temas de artículos, detección de comunidades), donde los vecindarios raramente repiten características y el promedio iguala a la suma. El **máximo** sirve para identificar elementos representativos o el "esqueleto", robusto a ruido y *outliers* (Qi et al., 2017, PointNet sobre nubes de puntos 3D).

Un segundo eje de ablación: el **MLP frente al perceptrón de 1 capa** ($\sigma \circ W$, usado por GCN y otras). El Lema 7 demuestra que existen multiconjuntos finitos $X_1 \neq X_2$ que ningún perceptrón lineal puede distinguir (ejemplo del Apéndice F: $X_1 = \{1,1,1,1,1\}$ y $X_2 = \{2,3\}$ suman lo mismo, y por homogeneidad de ReLU colisionan). El perceptrón de 1 capa, incluso con sesgo, no es un aproximador universal de funciones de multiconjunto; por eso GIN insiste en MLP de al menos 2 capas.

## 5. Experimentos

**Datasets (9 benchmarks de clasificación de grafos).** Cuatro de **bioinformática**: MUTAG (188 compuestos nitroaromáticos mutagénicos, 7 etiquetas), PROTEINS (nodos = elementos de estructura secundaria; etiquetas hélice/lámina/giro), PTC (344 compuestos, carcinogenicidad en ratas), NCI1 (4110 compuestos anticáncer del National Cancer Institute). Cinco de **redes sociales**: IMDB-BINARY e IMDB-MULTI (redes de colaboración actoral, clasificar el género), REDDIT-BINARY y REDDIT-MULTI5K (hilos de discusión, clasificar el subreddit/comunidad) y COLLAB (redes de colaboración científica en tres campos de física).

Detalle metodológico clave: el objetivo es forzar a los modelos a **aprender de la estructura**, no de las características de entrada. En las redes sociales los nodos no tienen características; para REDDIT *todos* los nodos reciben el mismo escalar (característica no informativa), y para las demás se usa el one-hot del grado del nodo. Esto convierte a REDDIT en una prueba pura de captura de estructura.

**Configuración.** 10-fold cross-validation con LIB-SVM; 5 capas de GNN (incluida la de entrada); todos los MLP de 2 capas; batch normalization en cada capa oculta; optimizador Adam (lr inicial 0.01, decaído por 0.5 cada 50 épocas). Variantes débiles: reemplazar la suma por promedio o max-pooling, o el MLP por perceptrón de 1 capa. Notablemente, **mean–1-layer corresponde a GCN** y **max–1-layer a GraphSAGE** (salvo modificaciones menores). Se compara contra el kernel de subárboles WL y contra DCNN, PATCHY-SAN, DGCNN y AWL.

**Resultado 1 — el accuracy de entrenamiento como medidor de expresividad.** Esta es la validación empírica más directa de la teoría: un modelo con mayor poder representacional debería poder *ajustar* mejor los datos de entrenamiento. Y eso es exactamente lo que se observa (Figura 4): GIN-$\epsilon$ y GIN-0 fijan casi perfectamente *todos* los conjuntos de entrenamiento, mientras que las variantes con promedio/máximo o perceptrón de 1 capa **subajustan severamente**. El patrón de accuracy de entrenamiento sigue el ranking teórico: MLP > 1-capa, y suma > promedio/máximo. Además, **el accuracy de entrenamiento de las GNN nunca supera al del kernel WL** —consistente con que WL es la cota superior—; en IMDB-BINARY, ningún modelo fija perfectamente el entrenamiento, alcanzando a lo más el accuracy del kernel WL.

**Resultado 2 — accuracy de test (Tabla 1).** GIN (especialmente GIN-0) iguala o supera a las variantes débiles en los 9 datasets, logrando estado del arte. Brilla en redes sociales con muchos grafos de entrenamiento. En REDDIT, donde todos los nodos comparten el mismo escalar, las GNN de suma capturan la estructura y superan ampliamente al resto, mientras que **las GNN de promedio no superan al azar** (fallan en capturar cualquier estructura de grafos no etiquetados, justo como predice la Sección 5.2): mean–MLP cae a 71.2% en REDDIT-BINARY y 41.3% en REDDIT-MULTI5K incluso usando el grado como característica. Valores representativos de GIN-0: IMDB-B 75.1%, RDT-B 92.4%, COLLAB 80.2%, MUTAG 89.4%, NCI1 82.7%.

**GIN-0 vs GIN-$\epsilon$.** Ambos ajustan el entrenamiento igual de bien, pero GIN-0 generaliza ligera y consistentemente mejor en test, probablemente por su mayor simplicidad. Aprender $\epsilon$ no aportó ganancia empírica.

## 6. Limitaciones

La limitación fundamental está incorporada en el propio resultado central: **1-WL es un techo, y 1-WL no es perfecto**. El test WL falla en distinguir ciertos grafos —notablemente algunos **grafos regulares**— donde todos los nodos tienen el mismo grado y vecindarios estructuralmente idénticos (Cai et al., 1992; Douglas, 2011; Evdokimov & Ponomarenko, 1999). Como GIN iguala pero no supera a 1-WL, *hereda* exactamente esos puntos ciegos: hay pares de grafos no isomorfos que GIN nunca podrá distinguir. Esto motivó toda una línea posterior de investigación en GNN *más* poderosas que 1-WL (k-WL, GNN con identificadores de nodo, etc.).

Otras limitaciones reconocidas:
- **Universo contable.** La teoría asume características de entrada de un universo contable. El caso continuo/incontable (características reales) "necesita consideraciones adicionales" que el paper deja para trabajo futuro, junto con caracterizar *cuán cerca* quedan las características aprendidas en la imagen de la función.
- **Solo paso de mensajes.** El marco caracteriza GNN *basadas en agregación de vecinos*. Ir más allá del paso de mensajes para lograr arquitecturas aún más poderosas es señalado como dirección futura.
- **Agregadores no estándar sin analizar.** Atención ponderada (GAT) y LSTM-pooling no se cubren, aunque el paper afirma que el marco es lo bastante general para analizarlos.
- **La teoría no habla directamente de generalización.** Los resultados son sobre *poder de distinción* (capacidad), no sobre generalización; la conexión expresividad → buen test accuracy es plausible y se observa, pero no demostrada formalmente.

## 7. Impacto: el marco estándar para razonar sobre expresividad de GNN

GIN se convirtió en la referencia canónica para discutir el poder expresivo de las redes neuronales de grafos. Tres contribuciones perduran:

1. **El lenguaje WL para GNN.** "Tan poderosa como 1-WL" pasó a ser la unidad de medida estándar del poder de una GNN. Toda arquitectura nueva se evalúa, al menos conceptualmente, contra esta jerarquía.
2. **La arquitectura GIN.** Es un baseline omnipresente en *graph classification* y un bloque de construcción frecuente, por ser simple, teóricamente fundamentado y empíricamente fuerte.
3. **El criterio de diseño "agregadores inyectivos".** El principio "suma + MLP para máximo poder" guía el diseño de GNN hasta hoy, y abrió el programa de investigación de GNN *más allá* de 1-WL (modelos $k$-WL, subgraph GNN, GNN con codificaciones posicionales/estructurales), todas las cuales se posicionan explícitamente respecto a la cota que GIN estableció.

## 8. Conexión con la Clase 27

GIN aporta el **fundamento teórico** que la Clase 27 presenta de forma operativa pero sin demostrar. Tres puentes directos:

- **Por qué la "función conmutativa de combinación" importa.** La clase introduce la agregación de vecinos como una función permutación-invariante (conmutativa) y ofrece suma, promedio y máximo como opciones. GIN demuestra que esta elección **es la decisión de diseño más consecuente para el poder del modelo**: la suma es inyectiva sobre multiconjuntos (captura todo), el promedio captura solo la distribución, el máximo solo el conjunto de elementos distintos. Cuando en el lab/clase se elige el agregador, se está eligiendo —literalmente— cuántos grafos distintos el modelo podrá distinguir. Los ejemplos de la Figura 3 (dos vecinos vs tres vecinos con la misma característica) son la justificación de por qué GCN/GraphSAGE, que usan promedio/máximo, fallan en grafos no etiquetados donde GIN triunfa.

- **Complemento con Barceló (Logical Expressiveness), citado en los créditos de la clase.** Ambos trabajos responden a la misma pregunta —*¿qué pueden y qué no pueden hacer las GNN?*— desde ángulos complementarios. GIN da una caracterización **algorítmica**: las GNN de paso de mensajes equivalen al test 1-WL. Barceló et al. dan una caracterización **lógica**: las GNN de agregación capturan exactamente las propiedades expresables en un fragmento de lógica de primer orden con contadores ($\text{FOC}_2$), que resulta estrechamente ligado a 1-WL. Para el estudiante de la clase, la lección unificada es: las GNN de paso de mensajes son potentes pero tienen un límite *preciso y demostrable*, y ese límite no es un detalle de implementación sino una propiedad estructural del paradigma de agregación de vecinos.

- **Del concepto a la arquitectura.** Si la clase muestra GCN como ejemplo de GNN, GIN explica qué le falta a GCN (su promedio + 1-capa lo hacen estrictamente menos expresivo que 1-WL) y cómo arreglarlo (suma + MLP + readout por suma sobre capas). Es el puente natural entre "aquí está una GNN que funciona" y "aquí está la GNN provablemente más poderosa de su clase, y aquí está la prueba".
