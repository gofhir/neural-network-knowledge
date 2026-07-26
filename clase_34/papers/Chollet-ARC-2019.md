# On the Measure of Intelligence (ARC) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *On the Measure of Intelligence*.
- **Autor:** **François Chollet** (Google, Inc.), creador de Keras. Agradece comentarios a José Hernández-Orallo, Julian Togelius, Christian Szegedy y Martin Wicke.
- **Publicación:** preprint arXiv:1911.01547v2 [cs.AI], 25 de noviembre de 2019 (versión original del 5 de noviembre de 2019). Documento extenso (64 páginas), estructurado en tres partes.
- **Aporte doble:** (1) una **crítica y redefinición formal** de qué significa "inteligencia" en IA, basada en Teoría de la Información Algorítmica (AIT); y (2) un **benchmark concreto**, el **Abstraction and Reasoning Corpus (ARC)**, diseñado para instanciar esa definición.

El argumento central de Chollet es que la comunidad de IA ha estado midiendo lo incorrecto. Durante décadas se ha equiparado inteligencia con **habilidad (skill)** en tareas específicas: ajedrez, Go, Atari, DotA2. Chollet muestra que esta equivalencia es un error, porque la habilidad se puede **"comprar"** con recursos: con suficientes *priors* codificados a mano o con suficientes datos de entrenamiento, un desarrollador puede alcanzar niveles arbitrarios de desempeño en cualquier tarea dada, **sin que el sistema exhiba ninguna capacidad de generalización propia**. Batir a un humano en un juego, por tanto, no mide inteligencia; mide cuánto cómputo y datos se invirtieron.

Frente a esto, Chollet propone que la inteligencia es la **eficiencia en la adquisición de habilidades** (*skill-acquisition efficiency*): la tasa a la que un sistema convierte sus *priors* y su experiencia en habilidad nueva sobre tareas previamente desconocidas que involucran novedad e incertidumbre. La habilidad no es inteligencia; es apenas el **artefacto cristalizado** que produce el proceso de la inteligencia. El paper formaliza esta intuición con AIT (complejidad de Kolmogorov) y aterriza en un test —ARC— construido sobre *priors* de **Core Knowledge** (conocimiento nuclear innato humano), con tareas *few-shot* de transformación de grillas que resisten la memorización.

Para la **Clase 34 (Razonamiento)** el paper importa porque ofrece un criterio riguroso para distinguir **razonamiento genuino** —abstracción, composición, sistematicidad— de la **interpolación estadística** sobre datos densamente muestreados. ARC es la prueba concreta de que composición y abstracción son difíciles: fácilmente resoluble por humanos, resistente a las técnicas de aprendizaje profundo que dominaban en 2019.

## 2. Contexto: el debate sobre qué mide un benchmark

Chollet abre situando dos concepciones históricas de la inteligencia que han guiado —implícitamente— tanto a la psicología como a la IA:

1. **Inteligencia como colección de habilidades específicas.** Es la visión de la psicología evolutiva (Darwin, Minsky, *The Society of Mind*): la mente como un ensamblaje de programas verticales, especializados, cada uno resolviendo un problema concreto. De aquí sale la definición de Minsky de 1968: "IA es la ciencia de hacer máquinas capaces de realizar tareas que requerirían inteligencia si las hiciera un humano". Bajo esta óptica, la evaluación se centra en el desempeño en tareas específicas.

2. **Inteligencia como capacidad general de aprendizaje.** Es la visión de la *Tabula Rasa* (Locke, Turing): la mente como proceso flexible que convierte experiencia arbitraria en conocimiento. Turing (1950) la resume con la imagen del cerebro del niño como un cuaderno en blanco. Esta visión, latente hasta el resurgimiento del machine learning en los años 80, se volvió dominante con el Deep Learning —hasta el punto de que muchos investigadores conceptualizan hoy la mente como una "red neuronal inicializada al azar" que deriva sus habilidades de los "datos de entrenamiento".

Chollet sostiene que **ambas visiones son incorrectas** (lo desarrolla vía psicología del desarrollo): la mente no es ni una colección de programas fijos ni una pizarra en blanco. Es un sistema con *priors* innatos que, lejos de limitar la generalización, **son su fuente**.

El problema práctico que denuncia es la deriva de la evaluación. Los benchmarks —reproducibles, justos, escalables— fueron motores genuinos de progreso (ImageNet, DARPA Grand Challenge). Pero optimizar una sola métrica induce **atajos**: los modelos ganadores en Kaggle suelen sobreespecializarse y no transferir al problema real. En IA, la fijación en el desempeño de tareas específicas —"sin poner condición alguna sobre cómo el sistema llega a ese desempeño"— produjo sistemas que hacen la tarea bien pero no exhiben inteligencia.

Chollet distingue un **espectro de generalización** que estructura todo el documento:

- **Ausencia de generalización:** sistemas sin incertidumbre (tic-tac-toe por enumeración exhaustiva).
- **Generalización local ("robustez"):** manejar puntos nuevos de una distribución conocida, dada una muestra densa (un clasificador de gatos vs. perros). Es lo que el ML ha hecho desde los años 50.
- **Generalización amplia ("flexibilidad"):** manejar una categoría amplia de tareas y situaciones no previstas por los creadores (un auto autónomo L5, un robot que pasa el test de la taza de café de Wozniak). Chollet señala que ni los sistemas más avanzados de 2019 pertenecen aquí.
- **Generalización extrema:** manejar tareas enteramente nuevas que solo comparten similitudes abstractas con lo visto antes. Solo las formas biológicas de inteligencia lo logran. Cuando el alcance se restringe a las tareas del ámbito humano, Chollet lo llama **"generalidad"**.

Este espectro **refleja la jerarquía de habilidades cognitivas** de la psicometría (teoría CHC): el factor *g* en la cima (generalización extrema), habilidades amplias en el medio (generalización amplia), y habilidades o tareas específicas en la base (generalización local). La crítica clave: los benchmarks multi-tarea (GLUE, SuperGLUE, Arcade Learning Environment) parecen medir algo más general, pero tienen un defecto fatal: **el conjunto de tareas es conocido de antemano por los desarrolladores**, que pueden entrenar específicamente para ellas. Ampliar la evaluación de habilidad a más tareas no produce un tipo cualitativamente distinto de evaluación: sigue midiendo habilidad, no capacidad.

## 3. Contribución: definición formal de inteligencia + ARC

La tesis que Chollet argumenta y luego formaliza es:

> **La inteligencia de un sistema es una medida de su eficiencia en la adquisición de habilidades sobre un alcance de tareas, con respecto a *priors*, experiencia y dificultad de generalización.**

Intuitivamente: dados dos sistemas que parten de *priors* similares y atraviesan una cantidad similar de experiencia sobre tareas no conocidas de antemano, el **más inteligente es el que termina con mayor habilidad** —el que convirtió sus *priors* y su experiencia en habilidad de forma más eficiente. La habilidad, insiste, es solo la salida del proceso; no es poseída por el sistema inteligente sino que es una propiedad del **artefacto** (el "programa de habilidad") que ese sistema genera.

De esta redefinición se desprenden dos conjuntos de consecuencias:

- **Para la investigación:** enfocarse en program synthesis (separar el "sistema inteligente" —motor de síntesis— del "programa de habilidad" —artefacto no inteligente—), en optimización de currículo, y en construir sistemas fundados en *priors* tipo Core Knowledge.
- **Para la evaluación:** un buen benchmark debe controlar *priors* y experiencia (no debe poder "comprarse" desempeño con datos ilimitados), medir generalización *developer-aware*, describir explícitamente sus *priors*, y funcionar de forma justa para humanos y máquinas.

ARC es la instanciación concreta de todas estas guías.

## 4. La formalización: *priors*, experiencia, generalización e inteligencia

Chollet monta el andamiaje formal sobre **Teoría de la Información Algorítmica (AIT)**. La noción central es la **Complejidad Algorítmica** (o complejidad de Kolmogorov) $H(s)$ de una cadena binaria $s$: la longitud del programa más corto que produce $s$ al ejecutarse en una máquina de Turing universal fija. La **complejidad algorítmica relativa** $H(s_1 \mid s_2)$ es la longitud del programa más corto que, tomando $s_2$ como entrada, produce $s_1$. Como todo programa es una cadena binaria, esto permite medir cuán "cerca" está un programa de otro.

El setup del problema distingue un **sistema inteligente** (*Intelligent System*, IS) —que genera programas de habilidad— de los **programas de habilidad** (*skill programs*) —que ejecutan la conducta—. La interacción se divide en una **fase de entrenamiento** (el IS genera un programa de habilidad que debe generalizar) y una **fase de evaluación** (se mide ese programa fijo sobre situaciones nuevas, sin el IS presente). Un juego como el ajedrez, o una tarea de ARC, constituyen una "tarea"; un tablero concreto o una grilla de entrada de ARC constituyen una "situación".

Sobre esta base, Chollet define cuantitativamente los tres factores a controlar:

**Dificultad de generalización.** Dada una tarea $T$, sea $Sol^{\theta}_T$ la solución más corta que alcanza al menos habilidad $\theta$, y $TrainSol^{opt}_{T,C}$ la solución de entrenamiento óptima más corta bajo un currículo $C$. La dificultad de generalización es:

$$GD^{\theta}_{T,C} = \frac{H\left(Sol^{\theta}_T \mid TrainSol^{opt}_{T,C}\right)}{H\left(Sol^{\theta}_T\right)}$$

Es la fracción de la complejidad de la solución de evaluación que **no** queda explicada por la mejor solución de entrenamiento: cuánto hay que "editar" el programa óptimo de entrenamiento para convertirlo en un programa que funcione en evaluación. Está acotada entre 0 y 1. Si el programa más simple que era óptimo en entrenamiento también basta en evaluación, la dificultad es cero: no hay incertidumbre, no hay generalización. Chollet advierte que esto es contraintuitivo respecto de la navaja de Occam: prepararse para la incertidumbre futura **tiene un costo** antagónico a la compresión de la política pasada. El programa de entrenamiento más corto descarta todo lo estrictamente innecesario para las situaciones vistas, y al hacerlo bota información que podría haber servido para las nuevas.

Para capturar la generalización *developer-aware* (que contabiliza el conocimiento que el desarrollador inyectó en el sistema), se condiciona además al estado inicial del sistema $IS_{t=0}$:

$$GD^{\theta}_{IS,T,C} = \frac{H\left(Sol^{\theta}_T \mid TrainSol^{opt}_{T,C},\, IS_{t=0}\right)}{H\left(Sol^{\theta}_T\right)}$$

**Priors.** Miden cuánta información relevante para la tarea viene ya embebida en el sistema inicial:

$$P^{\theta}_{IS,T} = \frac{H\left(Sol^{\theta}_T\right) - H\left(Sol^{\theta}_T \mid IS_{t=0}\right)}{H\left(Sol^{\theta}_T\right)}$$

Es cuán cerca de una solución arranca el sistema. Crucialmente, esto **no** es la cantidad total de información en el sistema, sino solo la fracción *relevante*: un sistema grande con mucho conocimiento irrelevante casi no es penalizado.

**Experiencia.** La información relevante y novedosa que el sistema recibe en cada paso del currículo. En el paso $t$:

$$E^{\theta}_{IS,T,t} = H\left(Sol^{\theta}_T \mid IS_t\right) - H\left(Sol^{\theta}_T \mid IS_t,\, data_t\right)$$

es decir, cuánto reduce ese paso de datos la incertidumbre del sistema sobre la solución. Al sumar sobre pasos usando información *novedosa* (no el contenido total del currículo), el formalismo **penaliza a los aprendices lentos** —que necesitan repetición— y **no penaliza** a los que atraviesan currículos ruidosos o repetitivos.

**Inteligencia.** Reuniendo las piezas, la inteligencia de un sistema sobre un alcance de tareas (caso suficiente) es:

$$I_{IS,scope} = \operatorname*{Avg}_{T \in scope}\left[\; \omega_T \cdot \theta_T \sum_{C \in Cur^{\theta_T}_T} P_C \cdot \frac{GD^{\theta_T}_{IS,T,C}}{P^{\theta_T}_{IS,T} + E^{\theta_T}_{IS,T,C}}\;\right]$$

Esquemáticamente, la contribución de cada tarea es:

$$\mathbb{E}\left[\frac{\text{habilidad} \cdot \text{generalización}}{\text{priors} + \text{experiencia}}\right]$$

ponderada por un valor $\omega_T$ que hace comparable la habilidad entre tareas de distinta escala. En prosa: **la inteligencia es la tasa a la que un aprendiz convierte su experiencia y sus *priors* en habilidad nueva sobre tareas valiosas que involucran incertidumbre y adaptación.** Un sistema de alta inteligencia genera soluciones de alta habilidad para tareas de alta dificultad de generalización usando **poca** experiencia y pocos *priors*: es una "tasa de conversión" entre información sobre una parte del espacio de situaciones y la capacidad de desempeñarse bien sobre un área máxima del espacio futuro, que incluirá novedad.

Observaciones clave del propio Chollet: la alta habilidad **no** es alta inteligencia (son conceptos distintos); la inteligencia **no** es curve-fitting (un sistema que solo produce el programa más simple consistente con los datos solo rinde en tareas de dificultad de generalización cero); y la medida está atada a un **alcance** (scope) y a una **función de valor** sobre tareas. Chollet contrasta explícitamente su definición con otras basadas en AIT —el C-Test de Hernández-Orallo, AIXI de Hutter, y la "Inteligencia Universal" de Legg y Hutter— de las que se distancia por rechazar el anti-antropocentrismo: para Chollet, **un marco de referencia antropocéntrico no solo es legítimo, es necesario**, porque el espacio de tareas relevantes para humanos es el único que hoy podemos aproximar y evaluar de forma significativa.

## 5. Core Knowledge: los *priors* innatos

Si la inteligencia debe medirse controlando *priors*, y si la generalidad debe compararse contra la inteligencia humana, entonces necesitamos **saber cuáles son los *priors* innatos humanos**. Chollet recurre a la teoría del **Core Knowledge** de la psicología del desarrollo (Spelke y Kinzler, 2007), que identifica cuatro sistemas de conocimiento innato, ancestrales y en gran parte compartidos con otras especies (por eso no requieren un canal evolutivo de alto ancho de banda para transmitirse por el ADN):

1. **Objetualidad y física elemental (*objectness*):** el entorno se parsea en "objetos" gobernados por **cohesión** (los objetos se mueven como todos continuos, conectados y acotados), **persistencia** (no dejan de existir ni se materializan de golpe) y **contacto** (no actúan a distancia ni se interpenetran).
2. **Agentividad y direccionalidad a metas (*agentness, goal-directedness*):** algunos objetos son "agentes" con intenciones propias que actúan para lograr metas y muestran eficiencia en ello (si A persigue a B, inferimos que B huye de A).
3. **Números naturales y aritmética elemental:** representaciones numéricas abstractas innatas para números pequeños, aplicables a través de cualquier modalidad sensorial, que pueden sumarse, restarse, compararse y ordenarse.
4. **Geometría y topología elementales:** nociones de distancia, orientación, relaciones dentro/fuera, que sustentan la facilidad innata para orientarse y navegar en 2D y 3D.

Chollet distingue estos *priors de conocimiento* de dos categorías que deja fuera: los *priors sensorimotores de bajo nivel* (reflejos), demasiado específicos salvo que se busque construir un cuerpo humano artificial; y los *priors de meta-aprendizaje* (las estrategias con que el cerebro convierte experiencia en conocimiento), que **no** deben controlarse porque **son** la inteligencia misma, no un factor modulador externo. Son los *priors de conocimiento* los que un test justo debe fijar: un sistema que no los posee está en desventaja frente a los humanos; un sistema con *priors* codificados **más** extensos sobre la tarea no puede compararse de forma justa, porque —como demostró la crítica— esos *priors* extra permiten "comprar" desempeño. De ahí la exigencia de que un test justo describa sus *priors* de forma **explícita y exhaustiva**, algo que los tests psicométricos humanos jamás hacen (asumen implícitamente conocimiento adquirido del examinado).

## 6. ARC: diseño y propiedades

El **Abstraction and Reasoning Corpus** es el intento de Chollet por implementar todas las guías anteriores. Puede leerse como benchmark de IA general, como benchmark de *program synthesis*, o como un test psicométrico de inteligencia fluida, cercano en formato a las **Matrices Progresivas de Raven** (test de CI de los años 30).

**Estructura.** ARC tiene **1.000 tareas únicas** en total: un conjunto de entrenamiento de **400 tareas** y uno de evaluación de **600 tareas**, este último dividido en un conjunto público (**400 tareas**) y uno privado (**200 tareas**). Los conjuntos de entrenamiento y de test son **disjuntos**; todas las tareas son únicas.

**Formato de una tarea.** Cada tarea consta de un puñado de **ejemplos de demostración** (3,3 en promedio) y, en general, **un** ejemplo de test (raramente 2 o 3). Cada ejemplo es un par de **grillas** de entrada y salida. Una grilla es una matriz literal de símbolos, con **10 símbolos únicos** (visualizados como colores), de dimensión entre 1×1 y 30×30 (mediana 9 de alto, 10 de ancho). El examinado ve los pares demostración completos y la grilla de entrada del test, y debe **construir desde cero** la grilla de salida: decidir sus dimensiones, qué símbolos poner y dónde. Se resuelve la tarea solo si produce la respuesta **exacta** (medida binaria de éxito); se permiten **3 intentos** por ejemplo de test, con retroalimentación puramente binaria (correcto/incorrecto). El puntaje de un sistema es la **fracción de tareas del conjunto de evaluación** que resuelve.

**Los *priors* de ARC son exactamente los de Core Knowledge.** ARC codifica los cuatro sistemas y **evita** deliberadamente cualquier conocimiento adquirido:

- *Objetualidad:* cohesión (parsear grillas en objetos por contigüidad espacial o continuidad de color), persistencia (objetos que sobreviven al ruido o la oclusión; tareas de *denoising*), e influencia por contacto (un objeto que se traslada hasta tocar otro; una línea que "rebota" contra un obstáculo).
- *Direccionalidad a metas:* muchos pares entrada/salida se modelan como estados inicial y final de un proceso intencional (útil aunque no estrictamente necesario).
- *Números y conteo:* contar, ordenar por tamaño, comparar cantidades (qué símbolo aparece más/menos veces), repetir patrones un número fijo de veces, sumar y restar. **Todas las cantidades en ARC son menores a aproximadamente 10.**
- *Geometría y topología:* líneas, rectángulos, simetrías, rotaciones, traslaciones, escalado, contención dentro/fuera de un perímetro, conexión de puntos, proyecciones ortogonales, copiado.

**Propiedades de diseño que resisten la memorización.** ARC se distingue de los tests psicométricos clásicos y del C-Test en formas fundamentales, todas orientadas a impedir atajos:

- Mide **solo inteligencia fluida** (razonamiento y abstracción), no habilidades cristalizadas; no usa lenguaje, imágenes de objetos reales ni sentido común del mundo real.
- Las tareas de evaluación son **únicas y desconocidas para los desarrolladores** de los sistemas examinados. Esto impide que un desarrollador resuelva las tareas él mismo y **codifique la solución en forma de programa** —el fraude que arruinó los intentos de aplicar tests de CI a máquinas—. El conjunto privado permite hacer cumplir esto estrictamente en competencias.
- Tiene **alta diversidad de tareas** (cientos de tareas con escaso solapamiento), lo que reduce la rentabilidad de codificar soluciones tarea por tarea.
- Las tareas están **generadas manualmente**, no programáticamente. Chollet ve la generación programática desde un "programa maestro" estático como una debilidad: bastaría con *ingeniería inversa* de ese programa (presumiblemente simple) para resolver todas las tareas. La generación manual aumenta la diversidad y reduce el riesgo de atajos imprevistos.

**Cómo se vería una solución.** Chollet postula que ARC es, para quien intente resolverlo, un problema de **síntesis de programas**: dado un puñado de pares entrada/salida (exactamente el formato de la especificación en *program synthesis*), generar el programa que transforma unas en otras. Un solucionador hipotético definiría un **lenguaje específico de dominio (DSL)** que codifique los *priors* de Core Knowledge como funciones base abstractas y combinables, generaría programas candidatos reutilizando subprogramas útiles en otras tareas, y seleccionaría los mejores por simplicidad o verosimilitud. Chollet advierte —coherente con su definición de dificultad de generalización— que **elegir simplemente el programa más simple que funciona en los pares de entrenamiento no generaliza bien** a los de test.

**Verificación humana y resistencia al ML.** Cada tarea de ARC ha sido resuelta con éxito por al menos uno de un grupo de tres humanos de CI alto que no se comunicaron entre sí, lo que demuestra factibilidad; un humano típico resuelve la mayoría del conjunto de evaluación **sin entrenamiento previo**. En contraste, y hasta donde Chollet sabía en 2019, ARC **no parecía abordable por ninguna técnica de machine learning existente, incluido el Deep Learning**, precisamente por su foco en generalización amplia y aprendizaje *few-shot*, y porque el conjunto de evaluación solo contiene tareas ausentes del entrenamiento.

**Debilidades reconocidas.** Chollet es explícito: ARC es *work in progress*. La dificultad de generalización **no está cuantificada** (planea estimarla vía desempeño humano); la validez estadística no está establecida; el tamaño (1.000 tareas) y la diversidad pueden ser limitados y vulnerables a atajos; y el formato binario 0/1 con 3 intentos es demasiado cerrado —una mejor versión dejaría al examinado interactuar con un generador de ejemplos, midiendo cuánta retroalimentación necesita, lo que se acercaría más a la definición formal de inteligencia. Como alternativas de largo plazo propone reconvertir benchmarks de habilidad en tests de generalización (entrenar en un juego X y evaluar en variantes novedosas $X_1,\dots,X_n$) y sistemas abiertos "maestro-alumno" (al estilo POET) que generen tareas cada vez más difíciles.

## 7. Implicancias: por qué los LLMs baten pero no dominan ARC

*(Esta sección conecta el paper de 2019 con desarrollos posteriores; las cifras concretas de desempeño de LLMs quedan fuera del texto de Chollet y no se citan aquí.)*

El marco de Chollet explica con precisión por qué los grandes modelos de lenguaje —entrenados años después de este paper— tensionan pero no resuelven ARC. Un LLM se entrena sobre una fracción gigantesca del texto humano: en los términos del paper, dispone de **experiencia y *priors* casi ilimitados**. La definición de inteligencia como cociente $\frac{\text{habilidad} \cdot \text{generalización}}{\text{priors} + \text{experiencia}}$ predice que, cuando el denominador crece sin límite, el desempeño alto en una tarea deja de ser evidencia de inteligencia. Los LLMs pueden interpolar sobre un espacio de situaciones densamente muestreado —el mecanismo del *hashtable* con hash sensible a la localidad que Chollet describe como generalización local—, pero ARC está construido precisamente para que ese muestreo denso **no exista**: cada tarea de evaluación es única, novedosa y desconocida para el desarrollador, y las cantidades de datos por tarea son minúsculas (3,3 demostraciones). No hay forma de "comprar" desempeño porque no se puede generar más datos de la tarea a voluntad.

Que los LLMs *avancen* en ARC (sobre todo con andamiajes de *program synthesis*, muestreo de programas candidatos y búsqueda guiada, y con *test-time adaptation*) confirma la otra mitad del argumento: la síntesis de programas sobre *priors* nucleares es efectivamente el camino que Chollet señaló. Pero que **no lo dominen** —quedando por debajo del humano típico que resuelve la mayoría sin práctica— confirma que la interpolación estadística no equivale a **abstracción composicional**. ARC exige recombinar un vocabulario pequeño de conceptos nucleares (objetos, contacto, conteo, simetría) de maneras nunca vistas, con dificultad de generalización estrictamente positiva. Ese es el hueco entre memorizar y razonar.

## 8. Conexión con la Clase 34 (Razonamiento)

La Clase 34 dedica sus slides 44-45 a ARC como prueba de razonamiento genuino, y el paper conecta directamente con los tres ejes de la clase:

- **Abstracción.** Para Chollet, la generalización extrema se logra "a través de una fuerte abstracción". Resolver una tarea de ARC requiere inferir la *regla abstracta* que gobierna los pares demostración (p. ej. "completar la simetría", "extrapolar la línea que rebota", "seleccionar el objeto más frecuente") a partir de poquísimos ejemplos, y aplicarla a una entrada nueva. No hay superficie estadística que explotar: la señal está en la estructura, no en la textura.
- **Sistematicidad y generalización composicional.** El diseño de ARC —un DSL de funciones base (los *priors* de Core Knowledge) que deben **recombinarse** de formas novedosas para cada tarea— es una prueba operacional de sistematicidad: si un sistema entiende "simetría" y "conteo" por separado, debería manejar tareas que los compongan de maneras no vistas. La resistencia de ARC a los sistemas de generalización local es exactamente el síntoma de que carecen de esta composicionalidad sistemática.
- **Generalización sobre memorización.** La lección transversal de la clase —que un sistema puede rendir alto sin razonar— queda formalizada aquí: la habilidad es el artefacto cristalizado; la inteligencia es el proceso que la produce, medible solo controlando *priors* y experiencia. ARC es el instrumento que hace ese control operativo.

**Enlaces internos:**

- Clase: [/clases/clase-34](/clases/clase-34) — Razonamiento (abstracción, sistematicidad, generalización composicional).
- Fundamento transversal sugerido: razonamiento / generalización composicional.
- Contraste value/skill vs. proceso: [/papers/a3c-mnih-2016](/papers/a3c-mnih-2016) — RL como ejemplo de sistemas de alta habilidad y baja generalización (OpenAI Five, citado por Chollet, entrenó 45.000 años de juego y resultó frágil).

## 9. Nota final: relevancia para salud

Para un sistema clínico, la distinción de Chollet entre **habilidad memorizada** y **eficiencia de adquisición de habilidad** es directamente accionable. Un modelo puede saturar un benchmark médico —alcanzando o superando a especialistas en un conjunto de test conocido— sin que ello prediga nada sobre su comportamiento ante una presentación atípica, una comorbilidad rara, una población no representada en los datos, o un protocolo de un centro distinto: en los términos del paper, ese desempeño se "compró" con datos y *priors*, y su dificultad de generalización *developer-aware* es baja. Lo que realmente importa en la práctica clínica es la capacidad de **generalizar a casos nuevos con pocos ejemplos** —el análogo médico del *few-shot* de ARC—: cuánta evidencia nueva necesita el sistema para alcanzar un umbral de competencia ante una entidad que ni él ni sus desarrolladores anticiparon. Evaluar sistemas de salud, entonces, debería medir eficiencia de adquisición y generalización controlada —currículos de casos verdaderamente novedosos, con *priors* explícitos— antes que celebrar puntajes altos en benchmarks saturados que pueden estar midiendo memorización disfrazada de razonamiento.
