---
title: "Omniglot y Bayesian Program Learning"
weight: 267
math: true
---

{{< paper-card
    title="Human-level concept learning through probabilistic program induction"
    authors="Brenden M. Lake, Ruslan Salakhutdinov, Joshua B. Tenenbaum"
    year="2015"
    venue="Science 2015" >}}

Este es uno de los papers más influyentes de la frontera entre ciencia cognitiva y machine learning de la década de 2010. Su contribución es doble: introdujo el dataset **Omniglot**, que se convirtió en *el* benchmark estándar de few-shot y meta-learning, y propuso **Bayesian Program Learning (BPL)**, un modelo generativo que representa los conceptos como programas probabilísticos composicionales. BPL alcanzó nivel humano en clasificación one-shot y, a la vez, hizo cosas que las redes de la época no podían: generar ejemplos nuevos, parsear objetos en partes y crear conceptos completamente nuevos.

{{< /paper-card >}}

## El dataset Omniglot (1623 caracteres, 50 alfabetos, el "transpose de MNIST")

Omniglot es el sustrato empírico del paper y, en retrospectiva, su legado más duradero. Su caracterización exacta:

- **1623 caracteres** (clases de concepto) provenientes de **50 sistemas de escritura / alfabetos** del mundo.
- **20 instancias por carácter**, cada una dibujada por una persona distinta vía Amazon Mechanical Turk (AMT).
- Se recolectaron **tanto las imágenes como los trazos del lápiz** (*pen strokes*): no solo el bitmap final, sino la **secuencia temporal de movimientos** — dónde empezó cada trazo, en qué orden, en qué dirección. Esta información de trayectoria es lo que permite a BPL modelar el proceso *causal* de generación, y es lo que la mayoría de los métodos puramente perceptuales (incluidas las convnets) ignoran.

**Por qué se diseñó así — el "transpose de MNIST".** MNIST tiene 10 clases (dígitos) con miles de ejemplos cada una; está pensado para el régimen de muchos datos por clase. Omniglot invierte deliberadamente esa proporción: **muchísimas clases (1623), pocos ejemplos por clase (20)**, con una sola instancia disponible en el régimen one-shot de evaluación. Los caracteres manuscritos se eligen porque son un terreno parejo (*even footing*) para comparar humanos y máquinas: son cognitivamente naturales — las personas los producen y reconocen rutinariamente — y a la vez constituyen un benchmark clásico de algoritmos de aprendizaje, el mismo dominio del que nació MNIST.

**Partición background / evaluación.** El paper usa una separación limpia, crucial para la honestidad del experimento one-shot. Un **background set de 30 alfabetos** (con imágenes *y* stroke data) se usa para que el modelo "aprenda a aprender": ajustar las distribuciones empíricas del modelo generativo (número de trazos, sub-trazos, primitivas, relaciones). Este mismo conjunto se usó para preentrenar los modelos de deep learning alternativos, para que la comparación fuera justa. Los **20 alfabetos restantes** (que incluyen los 10 usados en clasificación) son los de **evaluación**; durante la evaluación los modelos solo reciben **imágenes crudas** de caracteres novedosos. Esta estructura — entrenar conocimiento *transferible* sobre un conjunto de conceptos y evaluar sobre conceptos disjuntos — es exactamente la formulación que después se canonizaría como el setup de **meta-learning / episodic few-shot** (N-way K-shot).

## La pregunta: aprender conceptos desde un solo ejemplo

El paper abre planteando dos aspectos del conocimiento conceptual humano que, hacia 2015, habían eludido a las máquinas.

**Primero, la eficiencia en datos.** Para la mayoría de categorías, una persona aprende un concepto nuevo a partir de uno o un puñado de ejemplos. Basta ver *un* ejemplo de un vehículo novedoso para captar los límites del concepto. En contraste, los enfoques líderes de ML de la época — en especial el deep learning que dominaba reconocimiento de objetos y voz — son los más hambrientos de datos, requiriendo decenas, cientos o miles de ejemplos por clase.

Aquí hay una tensión teórica que el paper hace explícita. Bajo la teoría clásica del aprendizaje (el dilema sesgo-varianza, PAC learning), **ajustar un modelo más complejo requiere más datos, no menos**, para generalizar bien. Sin embargo, las personas navegan ese trade-off con notable agilidad: aprenden conceptos ricos que generalizan bien desde datos escasos. ¿Cómo? La respuesta del paper es que el cerebro no parte de cero; trae un **sesgo inductivo fuertísimo** — priors aprendidos — que reduce drásticamente el espacio de hipótesis plausibles.

**Segundo, la riqueza de las representaciones.** Incluso para conceptos simples, las personas aprenden representaciones que sirven para muchas más funciones que clasificar: crear nuevos ejemplares del concepto, parsear un objeto en partes y relaciones, y crear nuevas categorías abstractas a partir de las existentes. Los mejores clasificadores de la época no hacían nada de esto, o requerían algoritmos ad hoc para cada función. El desafío central: explicar cómo el aprendizaje humano puede tener éxito **desde datos tan escasos** *y* producir **representaciones tan ricas, abstractas y flexibles** al mismo tiempo.

El framing es importante: el paper no es anti-deep-learning, sino un argumento de que **la estructura del modelo importa**. La eficiencia de datos no surge de tener más parámetros, sino del sesgo inductivo correcto — un modelo generativo que captura el proceso causal real que produce los datos (la mano que dibuja el carácter).

## Bayesian Program Learning: conceptos como programas probabilísticos

La idea central de BPL es representar cada concepto como un **programa probabilístico**: un modelo generativo expresado como un procedimiento estructurado en un lenguaje de descripción abstracto. No es una plantilla estática ni un vector de features, sino un *programa estocástico* que, al ejecutarse, **genera** instancias del concepto. Es, en palabras de los autores, "a generative model for generative models": el nivel superior genera *tipos* de concepto (una "A", una "B"), y cada tipo es a su vez un modelo generativo que produce *tokens* (instancias).

El framework articula tres ideas que habían sido influyentes por separado durante décadas:

**(1) Composicionalidad.** Los conceptos ricos se construyen a partir de primitivas más simples. Un carácter no es un blob de píxeles, sino una combinación de **partes** (strokes), que a su vez son combinaciones de **sub-partes** (sub-strokes tomados de una biblioteca discreta), ensambladas mediante **relaciones espaciales**. La reutilización de piezas es lo que permite generar conceptos nuevos: se recombinan trozos de programas existentes.

**(2) Causalidad.** La semántica del programa refleja la **estructura causal del proceso real** que produce los ejemplos. En los caracteres, ese proceso es la **mano humana escribiendo**: los trazos se inician al presionar el lápiz y terminan al levantarlo; las sub-partes son movimientos primitivos separados por pausas breves. Modelar este proceso causal — y no solo la apariencia final — es lo que explica la ventaja de BPL sobre las convnets. Las redes profundas modelan la *apariencia*; BPL modela la *generación*.

**(3) Learning-to-learn.** El modelo desarrolla **priors jerárquicos** que permiten que la experiencia con conceptos relacionados facilite el aprendizaje de conceptos nuevos. Estos priors son un sesgo inductivo aprendido que abstrae las regularidades que se sostienen tanto *entre tipos* de concepto como *entre tokens* del mismo concepto: cuántas partes suelen tener los caracteres, qué hace que un trazo sea razonable, cómo se relacionan espacialmente. Aprender los hiperparámetros del proceso de aprendizaje mismo es, precisamente, meta-learning.

## El modelo generativo jerárquico (tipo → token → imagen)

La maquinaria formal define una **distribución conjunta** sobre tres niveles: el tipo $\psi$, un conjunto de $M$ tokens $\theta^{(1)}, \dots, \theta^{(M)}$ de ese tipo, y las imágenes binarias correspondientes $I^{(1)}, \dots, I^{(M)}$. La factorización es:

$$
P\!\left(\psi, \theta^{(1)}, \dots, \theta^{(M)}, I^{(1)}, \dots, I^{(M)}\right) = P(\psi) \, \prod_{m=1}^{M} P\!\left(I^{(m)} \mid \theta^{(m)}\right) \, P\!\left(\theta^{(m)} \mid \psi\right)
$$

Una jerarquía de tres niveles que se lee de arriba hacia abajo: **tipo → token → imagen**.

**Nivel de tipo $P(\psi)$.** El tipo es un esquema abstracto de partes, sub-partes y relaciones. Su proceso generativo: (1) se muestrea el **número de partes $\kappa$** y, por cada parte, su **número de sub-partes $n_i$**, desde distribuciones empíricas medidas en el background set; (2) se construye cada parte muestreando sub-partes de un **conjunto discreto de acciones primitivas**, donde la probabilidad de la siguiente acción **depende de la anterior** (un modelo de Markov sobre la secuencia de sub-trazos); (3) las partes se aterrizan como **curvas paramétricas (splines)** muestreando puntos de control y escala; (4) las partes se posicionan según la **relación $R_i$** — una parte puede empezar de forma independiente, al inicio, al final, o a lo largo de partes previas.

**Nivel de token $P(\theta^{(m)} \mid \psi)$.** Dado el tipo, cada token concreto se produce **ejecutando** las partes y relaciones: se añade **ruido motor** a los puntos de control y la escala (ninguna persona dibuja el mismo carácter dos veces idéntico), se muestrea la **ubicación de inicio precisa** de cada trayectoria desde la relación correspondiente, y se aplican **transformaciones globales** — un warp afín y parámetros de ruido adaptativo.

**Nivel de imagen $P(I^{(m)} \mid \theta^{(m)})$.** Se crea una imagen binaria mediante una **función de renderizado estocástica**: se "pintan" las trayectorias con tinta en escala de grises, y los valores de píxel se interpretan como **probabilidades Bernoulli independientes**. Cada píxel se prende o apaga según cuánta tinta lo cubra, cerrando el puente entre el programa latente y los datos crudos.

La elegancia de esta jerarquía es que separa limpiamente **lo que es el concepto** ($\psi$, invariante de clase), **cómo varía una instancia** ($\theta$, variabilidad intra-clase por ruido motor y warps) y **cómo se ve el píxel final** ($I$, el modelo de ruido de observación). Cada nivel tiene hiperparámetros aprendidos del background set: ahí vive el learning-to-learn.

La **inferencia** es el problema duro. Generar es solo ejecutar el programa hacia adelante; invertir el proceso — dada una imagen, inferir el programa latente — requiere buscar en el enorme espacio combinatorio de programas posibles. La estrategia es bottom-up + refinamiento: métodos rápidos clásicos de análisis de trazos proponen un rango de **parses candidatos**, que luego se refinan mediante optimización continua y búsqueda local, formando una aproximación discreta a la posterior $P(\psi, \theta^{(m)} \mid I^{(m)})$. La clasificación one-shot se hace por **probabilidad predictiva posterior**: dado un ejemplo $I^{(1)}$, se descubren sus programas; cada programa se re-ajusta a la imagen de test $I^{(2)}$ y se computa $\log P(I^{(2)} \mid I^{(1)})$. Dos imágenes son de la misma clase si existe *un mismo programa* que pudo generar ambas con alta probabilidad — razonamiento por analogía generativa, no por distancia en un espacio de features.

## Resultados (one-shot 20-way ~3.3% error vs humanos 4.5%)

El resultado central es la clasificación **one-shot 20-way** (chance = 95% de error):

| Modelo | Error one-shot 20-way |
|---|---|
| Modified Hausdorff distance (baseline) | 38.8 % |
| Hierarchical Deep (HD) model | 34.8 % |
| Deep Convnet | 13.5 % |
| BPL lesión sin composicionalidad | 14.0 % |
| BPL lesión sin learning-to-learn (token) | 11.0 % |
| Deep Siamese Convnet (optimizada one-shot) | 8.0 % |
| **Humanos (N = 40)** | **4.5 %** |
| **BPL (completo)** | **3.3 %** |

Lecturas clave: BPL (**3.3%**) **iguala o supera ligeramente al humano (4.5%)** y queda muy por delante de todas las redes profundas de la época. La red Siamesa optimizada para one-shot logra 8.0% — el mejor competidor neuronal, pero aún cerca del doble del error. Las **ablaciones confirman la tesis de los tres principios**: quitar learning-to-learn sube el error a 11.0%, quitar composicionalidad a 14.0%. Cada ingrediente contribuye.

**Visual Turing tests** (ID level, donde **50% = ideal**, indistinguible de un humano). En generación de **nuevos ejemplares** (estático), BPL logra **52% ID** (apenas sobre chance; solo 3 de 48 jueces fiablemente por encima del azar). Sus lesiones lo delatan: sin learning-to-learn → 80% ID, sin composicionalidad → 65% ID, prueba de que esos principios importan. En generación **dinámica** (películas del trazo), BPL logra **59% ID**; aleatorizar el prior de orden y dirección de trazos lo sube a 71%. En **conceptos nuevos a partir de un alfabeto**, BPL logra **49% ID** (no distinguible del azar). En **conceptos free-form**, **51-57% ID** según el prior usado. El resumen que destacan los autores: en cada visual Turing test, **menos del 25% de los jueces tuvo desempeño significativamente mejor que el azar** — BPL es, en la práctica, indistinguible de un humano en estas tareas creativas.

**Robustez con poca experiencia de fondo.** Reentrenando con solo **5 alfabetos** de background (en vez de 30), BPL mantiene un error de **4.3% y 4.0%** (vs 3.3%), mientras que la convnet profunda se desploma a **24.0% y 22.3%** (vs 13.5%). La estructura causal/composicional de BPL le permite aprovechar casi por completo una experiencia de fondo muy limitada.

## Por qué importa: Omniglot como benchmark de facto del few-shot/meta-learning

El impacto del paper tiene dos legados.

**Legado 1 — Omniglot como benchmark de few-shot / meta-learning.** Tras 2015, Omniglot se volvió el banco de pruebas estándar para validar cualquier método de aprendizaje de pocos ejemplos. La lista de trabajos que lo adoptaron es esencialmente el árbol genealógico del meta-learning moderno: **Memory-Augmented Neural Networks** (Santoro et al., 2016), **Matching Networks** (Vinyals et al., 2016, que popularizó el protocolo episódico N-way K-shot que Omniglot habilita), **Prototypical Networks** (Snell et al., 2017), **MAML** (Finn et al., 2017) y las **Siamese Networks** de Koch et al. (2015), ya citadas *dentro* de este propio paper como competidor. Estos métodos reportan resultados sobre splits estándar 5-way y 20-way, 1-shot y 5-shot. Muchos eventualmente **saturaron** Omniglot (errores < 1-2%), lo que motivó benchmarks más difíciles — pero todos pasaron por aquí primero.

**Legado 2 — El debate "estructura composicional vs. aprendizaje end-to-end".** Este paper es una de las declaraciones más fuertes y empíricamente respaldadas de la posición *structure-first*: el camino hacia la inteligencia humana no es solo escalar redes con más datos, sino dotar a los modelos de los sesgos inductivos correctos (composicionalidad, causalidad, learning-to-learn). Lake reforzó esto en su comentario "Building Machines That Learn and Think Like People" (2017). La tensión con la escuela puramente conexionista — que apuesta a que esos sesgos *emergen* del entrenamiento a escala — es una de las divisiones intelectuales centrales del ML de la última década, viva todavía en los debates sobre razonamiento composicional en LLMs. El paper también anticipa lo que hoy llamamos **program synthesis / neurosymbolic AI** (DreamCoder es descendiente directo de esta línea).

**Limitaciones honestas.** Los autores son explícitos: BPL ve menos estructura que las personas (le falta conocimiento de líneas paralelas, simetría, elementos opcionales). Su crítica más fuerte para un practitioner es la **especificidad de dominio**: BPL requiere cablear a mano el proceso generativo (trazos de lápiz, presionar/levantar, renderizar tinta), por lo que **no escala trivialmente a imágenes naturales**, donde no hay un proceso causal tan limpio y conocido. Y su inferencia — búsqueda combinatoria de parses más optimización continua — es un motor bespoke, caro, frente al simple forward pass de una red.

## Conexión con la Clase 26

Omniglot es exactamente el dataset que la Clase 26 usa para introducir el paradigma de few-shot / one-shot learning. El marco que este paper fija y que conviene tener claro:

- El **protocolo N-way K-shot** (aquí 20-way 1-shot) y la **separación background/evaluación** que se convirtió en el setup episódico de meta-learning.
- La distinción entre el enfoque de este paper (**generativo, estructurado, causal** — BPL) y los enfoques que la clase cubre después (**discriminativos / métricos / basados en gradiente** — Siamese, Matching, Prototypical, MAML). Que la Siamese de Koch aparezca *citada dentro* de este propio paper como competidor (8.0% error) ubica la cronología: 2015 es el año bisagra en que estas dos familias compiten sobre el mismo benchmark.
- La idea de que **few-shot learning no es magia: es un prior fuerte**. La eficiencia de datos viene de transferir estructura aprendida de tareas relacionadas, no de un algoritmo que aprende de la nada.

## Notas y enlaces

- Publicado en *Science*, vol. 350, nº 6266, pp. 1332-1338 (11 de diciembre de 2015). DOI: `10.1126/science.aab3050`.
- Recursos liberados por los autores: dataset Omniglot (`github.com/brendenlake/omniglot`), código de BPL (`github.com/brendenlake/BPL`) y los archivos de los visual Turing tests (`github.com/brendenlake/visual-turing-tests`).
- La continuación conceptual del argumento es Lake, Ullman, Tenenbaum & Gershman, "Building Machines That Learn and Think Like People" (2017).

Ver fundamentos: [Meta-aprendizaje](/fundamentos/meta-aprendizaje) - [Few-shot learning](/fundamentos/few-shot-learning).

Ver papers: [Memory-Augmented NN (Santoro 2016)](/papers/mann-santoro-2016) - [Matching Networks (Vinyals 2016)](/papers/matching-networks-vinyals-2016) - [Prototypical Networks (Snell 2017)](/papers/prototypical-networks-snell-2017) - [MAML (Finn 2017)](/papers/maml-finn-2017).

Ver clase: [Clase 26 -- Meta-aprendizaje](/clases/clase-26).
