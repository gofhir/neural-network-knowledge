---
title: "EWC: Elastic Weight Consolidation (2017)"
weight: 356
math: true
---

{{< paper-card
    title="Overcoming catastrophic forgetting in neural networks"
    authors="James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, et al. (DeepMind)"
    year="2017"
    venue="PNAS 2017"
    pdf="/papers/ewc-kirkpatrick-2017.pdf"
    arxiv="1612.00796" >}}
Paper de DeepMind que introduce **Elastic Weight Consolidation (EWC)**, el método de **regularización** fundacional del [aprendizaje continuo](/fundamentos/aprendizaje-continuo) moderno. La tesis: el **olvido catastrófico** —perder lo aprendido en una tarea A al entrenar después en B— no es inevitable, sino que se mitiga **ralentizando selectivamente el aprendizaje sobre los pesos que importan para las tareas viejas**. La inspiración es neurocientífica (consolidación sináptica en el neocórtex mamífero) y la implementación es una penalización cuadrática por parámetro, ponderada por su **importancia**, estimada con la **diagonal de la matriz de información de Fisher**. Lo validan en MNIST permutado (supervisado) y en juegos de Atari en secuencia (refuerzo, sobre DQN).
{{< /paper-card >}}

---

## El problema: olvido catastrófico

El **olvido catastrófico** (catastrophic forgetting o interference), documentado desde fines de los 80 (McCloskey y Cohen, 1989; French, 1999), es el fenómeno por el cual una red entrenada secuencialmente en varias tareas pierde de forma **abrupta** —no gradual— el desempeño en las tareas previas a medida que aprende las nuevas. La causa mecánica es directa: la red comparte un único conjunto de parámetros entre todas las tareas, y el descenso de gradiente que optimiza la tarea B **sobreescribe** los pesos que eran importantes para A. El gradiente de B no "sabe" nada de A, y nada impide que destruya su solución.

El paper sitúa esto como una barrera hacia la inteligencia general: un agente debe aprender tareas consecutivas sin olvidar las anteriores, en un mundo donde las tareas no vienen etiquetadas, cambian de forma impredecible y pueden no recurrir por largos intervalos. El estado del arte hacia 2016 **evitaba** el problema en vez de resolverlo. El paradigma de *multitask learning* entrelaza (interleaves) los datos de todas las tareas durante el entrenamiento, de modo que los pesos se optimizan en conjunto y el olvido nunca ocurre, pero exige tener todos los datos simultáneamente. La alternativa secuencial era almacenar todas las experiencias y *reproducirlas* (replay), impráctica porque la memoria crece de forma proporcional al número de tareas.

## Inspiración neurocientífica

El corazón conceptual del trabajo es la **consolidación sináptica**. A diferencia de las redes artificiales, los mamíferos aprenden de forma continua. Evidencia reciente (Cichon y Gan, 2015; Yang et al., 2009) muestra que el neocórtex protege el conocimiento previo en sus circuitos: cuando un ratón adquiere una habilidad, una proporción de sinapsis excitatorias se **fortalece** (las espinas dendríticas aumentan de volumen) y **persiste** pese al aprendizaje posterior de otras tareas, lo que explica la retención meses después. Y cuando se borran selectivamente esas espinas (optogenética), la habilidad correspondiente se olvida —evidencia *causal* de que proteger las sinapsis fortalecidas es crítico para la retención. EWC es la traducción algorítmica de ese principio: vuelve menos plásticos los pesos importantes para tareas ya aprendidas.

## La idea: el resorte elástico

La contribución de EWC es **una pérdida de regularización** que, al entrenar la tarea B, penaliza el desplazamiento de cada peso respecto a su valor óptimo en A, con una penalización individualizada por parámetro proporcional a su importancia. La analogía que da nombre al método es el **resorte (spring) elástico**: cada peso queda anclado a su valor previo $\theta^*_A$ por un resorte cuya rigidez no es uniforme, sino mayor para los pesos que más importan al desempeño de A. De ahí *elastic weight consolidation*: consolidación (volver estables los pesos importantes) implementada de forma elástica (un anclaje cuadrático suave, no una congelación dura).

El insight que lo hace viable es la **sobreparametrización**: una red profunda tiene tantos parámetros que muchas configuraciones distintas de $\theta$ logran el mismo desempeño. Es probable, entonces, que exista una solución para B, $\theta^*_B$, *cercana* a la solución previa $\theta^*_A$. La Figura 1 lo ilustra: el gradiente de B solo minimiza B pero destruye A; una restricción uniforme con el mismo coeficiente para todos los pesos (equivalente a regularización L2) es demasiado severa y solo recuerda A a costa de no aprender B; EWC encuentra una solución para B sin pérdida significativa sobre A, porque calcula explícitamente cuán importante es cada peso para A.

## El método: de Bayes a la penalización de Fisher

### Justificación bayesiana

Optimizar los parámetros equivale a hallar los más probables dados los datos $D$, vía Bayes. Suponiendo que los datos se dividen en dos partes independientes —tarea A ($D_A$) y tarea B ($D_B$)— la regla se reordena así:

$$\log p(\theta \mid D) = \log p(D_B \mid \theta) + \log p(\theta \mid D_A) - \log p(D_B)$$

Esta reformulación es el núcleo del argumento. El lado derecho solo depende de la pérdida de la tarea B, $\log p(D_B\mid\theta)$, más el **posterior de la tarea A**, $\log p(\theta\mid D_A)$. Toda la información sobre A —incluyendo qué parámetros le fueron importantes— quedó absorbida en ese posterior. La idea elegante: **el posterior de la tarea A se convierte en el prior para aprender la tarea B**. Aprender de forma continua es, formalmente, encadenar inferencias bayesianas.

### Aproximación de Laplace y matriz de Fisher

El posterior verdadero $p(\theta\mid D_A)$ es intratable. Siguiendo la **aproximación de Laplace** (MacKay, 1992), EWC lo aproxima por una **Gaussiana** centrada en los parámetros óptimos $\theta^*_A$, con precisión diagonal dada por la **diagonal de la matriz de información de Fisher** $F$. La elección de Fisher se justifica por tres propiedades: (a) cerca de un mínimo equivale a la segunda derivada de la pérdida (aproxima la curvatura/Hessiano), de modo que un peso con Fisher alto es uno donde la pérdida sube rápido si se lo mueve; (b) se calcula **solo con derivadas de primer orden**, barato incluso para modelos grandes; y (c) es semidefinida positiva, lo que vuelve la penalización convexa y bien comportada.

### La función objetivo

Con esta aproximación, EWC minimiza al entrenar la tarea B:

$$L(\theta) = L_B(\theta) + \sum_i \frac{\lambda}{2}\, F_i\, (\theta_i - \theta^*_{A,i})^2$$

donde $L_B(\theta)$ es la pérdida de la tarea B *sola*; el sumatorio recorre cada parámetro $i$; $F_i$ es el i-ésimo término diagonal de la matriz de Fisher de A (la importancia del peso $i$ para A); $(\theta_i - \theta^*_{A,i})^2$ es el desplazamiento cuadrático respecto a su valor óptimo en A; y $\lambda$ es un escalar que fija cuán importante es la tarea vieja respecto a la nueva (la rigidez global de los resortes).

La interpretación mecánica es transparente: cada parámetro es jalado de vuelta hacia $\theta^*_{A,i}$ con fuerza proporcional a $F_i$. Pesos con $F_i$ alto (cruciales para A) quedan casi congelados; pesos con $F_i$ bajo (irrelevantes) quedan libres para reaprender en B. Es exactamente la "ralentización selectiva del aprendizaje". La aproximación diagonal ignora las correlaciones entre pesos —lo que la abarata, pero introduce su principal debilidad teórica. Al pasar a una tercera tarea C, dado que la suma de dos penalizaciones cuadráticas es a su vez cuadrática, los términos se pueden consolidar (lo que más tarde motivó la *online EWC*, con una sola Gaussiana acumulada).

## Experimentos

### MNIST permutado (supervisado)

El primer banco de pruebas es **permuted MNIST**: por cada tarea se genera una permutación aleatoria fija de los píxeles, que se aplica a todas las imágenes. Cada tarea tiene la misma dificultad que el MNIST original pero requiere una solución distinta. Se entrena una red densa (ReLU) en secuencia, sin volver a entrenar sobre cada dataset tras su turno. Los resultados (Figura 2) son la demostración fundacional:

- **SGD plano** sufre olvido catastrófico: el desempeño en B sube empinado mientras el de A cae en picada, y empeora con cada tarea.
- **Regularización L2** (restricción cuadrática uniforme) protege A pero **no logra aprender B**: protege todos los pesos por igual y deja poca capacidad libre.
- **EWC** es el único que mantiene alto desempeño en las tareas viejas *y* conserva la capacidad de aprender las nuevas, escalando a muchas tareas con solo un crecimiento modesto del error (frente a SGD con *dropout*, la mejor línea base previa, que solo escalaba a dos permutaciones).

Un experimento secundario mide el **solapamiento entre matrices de Fisher** por profundidad: cuando dos tareas son muy similares dependen de pesos similares en toda la red; cuando son disímiles, la red asigna capacidad separada. EWC comparte representación donde las tareas comparten estructura y subdivide donde no.

### Atari 2600 secuencial (refuerzo)

El segundo banco es mucho más exigente: **deep RL** con **DQN** (Mnih et al., 2015) sobre diez juegos Atari elegidos al azar, presentados en orden aleatorio con posibilidad de volver a juegos previos. A diferencia de enfoques previos que añadían capacidad (Progressive Networks) o destilaban redes separadas (Policy Distillation), EWC usa **una sola red de recursos fijos**. Añade dos componentes a un agente Double DQN:

1. Un **módulo de reconocimiento de tarea**, necesario porque en RL no hay etiquetas explícitas. El contexto se trata como variable latente de un *Hidden Markov Model* bayesiano no paramétrico, que puede crear modelos generativos nuevos si explican mejor los datos recientes.
2. La **penalización EWC**, calculando la diagonal de Fisher en cada cambio de tarea ($\lambda = 400$).

Resultado (Figura 3): con descenso de gradiente plano el agente nunca aprende más de un juego; con EWC sí aprende a jugar múltiples. Proveer la etiqueta de tarea verdadera (en vez de inferirla con el HMM) solo aporta una mejora modesta, validando el módulo de reconocimiento. Un control adicional (Figura 3C) confirma que el agente es **más robusto** a perturbaciones moldeadas por el inverso de Fisher que a perturbaciones uniformes —validando a Fisher como estimador de importancia—, aunque revela que el método es **sobreconfiado** sobre qué parámetros son irrelevantes.

## Limitaciones reconocidas

- **Fisher diagonal:** ignora las correlaciones entre pesos (posterior gaussiano factorizado). Eso vuelve el método lineal en el número de parámetros, pero sacrifica las interacciones entre pesos.
- **Estimación puntual de la varianza (Laplace):** el paper la reconoce como "debilidad significativa". El método **subestima la incertidumbre** de los parámetros, siendo sobreconfiado sobre cuáles son prescindibles; sugieren mejorarlo con redes bayesianas.
- **Crecimiento de términos:** cada tarea nueva añade un anclaje (mitigado al sumar las penalizaciones cuadráticas; en secuencias largas motivó la *online EWC*).
- **Brecha con redes separadas:** en Atari, EWC evita el olvido pero **no alcanza** el puntaje de entrenar diez DQNs separados. Hay un costo por compartir capacidad fija.

## Por qué importa: el método de regularización fundacional

EWC es **el método de regularización fundacional del aprendizaje continuo moderno**. Definió toda una familia de enfoques —los basados en *penalización de pesos*— que estiman una importancia por parámetro y penalizan su cambio. Su descendencia directa incluye [Synaptic Intelligence](/papers/synaptic-intelligence-zenke-2017) (Zenke et al., 2017), que calcula la importancia de forma *online* a lo largo de la trayectoria de entrenamiento en vez de en puntos discretos de cambio de tarea; Memory Aware Synapses (MAS); y la *online EWC* (Schwarz et al., 2018). Junto con las otras dos grandes familias —*replay/rehearsal* (iCaRL, GEM) y *arquitecturas dinámicas* (Progressive Networks, PackNet)— EWC define la taxonomía estándar del campo. Y aportó una contribución conceptual de doble sentido: no solo construyó un algoritmo a partir de neurobiología, sino que mostró que las teorías de consolidación sináptica **escalan** a sistemas de aprendizaje a gran escala.

## Conexión con la Clase 32

La [Clase 32](/clases/clase-32) (Olvido Catastrófico / Aprendizaje Continuo) presenta EWC como el **primer método de la familia de regularización**, contrapunto fundacional a las familias de *replay* y de *arquitecturas dinámicas*. Lo que el estudiante debe internalizar: (1) el olvido ocurre porque el gradiente de la tarea nueva sobreescribe los pesos importantes de la vieja; (2) la solución no es congelar todo (eso es L2 uniforme, que mata el aprendizaje nuevo) ni añadir capacidad (eso es la familia de arquitecturas), sino **penalizar selectivamente el cambio de los pesos importantes**; (3) la importancia se mide con la diagonal de Fisher, barata y justificada como aproximación de la curvatura; (4) todo tiene una lectura bayesiana limpia —el posterior de A es el prior de B—; y (5) sus límites motivaron la siguiente generación de métodos.

## Notas y enlaces

- Fundamento transversal: [/fundamentos/aprendizaje-continuo](/fundamentos/aprendizaje-continuo) — regularización, replay y arquitecturas dinámicas; los tres escenarios de continual learning.
- Clase: [/clases/clase-32](/clases/clase-32) — Olvido Catastrófico / Aprendizaje Continuo.
- Paper hermano (importancia online): [/papers/synaptic-intelligence-zenke-2017](/papers/synaptic-intelligence-zenke-2017) — Synaptic Intelligence, evolución directa de EWC.
- Familia *learning without forgetting* (destilación): [/papers/lwf-li-2016](/papers/lwf-li-2016).
- Preprint: arXiv:1612.00796v2 (25 ene 2017). Venue: *PNAS*, 2017.
