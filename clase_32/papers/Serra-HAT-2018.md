# Overcoming Catastrophic Forgetting with Hard Attention to the Task — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Overcoming Catastrophic Forgetting with Hard Attention to the Task*.
- **Autores:** Joan Serrà (Telefónica Research, Barcelona), Dídac Surís (Telefónica Research / Universitat Politècnica de Catalunya), Marius Miron (Telefónica Research / Universitat Pompeu Fabra), Alexandros Karatzoglou (Telefónica Research). Correspondencia: joan.serra@telefonica.com.
- **Venue:** *Proceedings of the 35th International Conference on Machine Learning* (ICML 2018), Estocolmo, Suecia, PMLR 80.
- **Año:** 2018. **Preprint:** arXiv:1801.01423v3 (29 may 2018), [arxiv.org/abs/1801.01423](https://arxiv.org/abs/1801.01423).
- **Código:** [github.com/joansj/hat](https://github.com/joansj/hat) (PyTorch 0.2.0).

Este es un paper de **método de arquitectura** para combatir el *olvido catastrófico* (catastrophic forgetting): la tendencia de una red neuronal a borrar lo aprendido en una tarea cuando se entrena en tareas posteriores. El olvido catastrófico es uno de los obstáculos centrales para el *aprendizaje continuo* o *lifelong learning*, donde un modelo debe absorber una secuencia de tareas sin re-procesar los datos antiguos. La contribución es **HAT (Hard Attention to the Task)**: un mecanismo de atención dura, condicionado por la identidad de la tarea, que aprende —tarea a tarea— **máscaras casi binarias sobre las unidades (activaciones) de cada capa**, y usa esas máscaras para **proteger del gradiente los pesos importantes** de tareas previas mientras deja libre el resto de la capacidad de la red.

La tesis central es elegante: si pudiéramos saber *qué unidades* de cada capa son cruciales para cada tarea, podríamos congelar los pesos asociados a esas unidades (estabilidad) y dejar las unidades restantes libres para aprender tareas futuras (plasticidad). HAT no impone esa partición a mano ni con heurísticas previas: la **aprende junto con la red**, vía backpropagation y SGD, mediante un *gating* sigmoide de un *embedding de tarea* cuyo factor de escala se va recociendo (*annealing*) hasta volver las máscaras casi binarias. El resultado experimental es contundente: HAT **reduce el olvido entre un 45% y un 80%** respecto a los mejores métodos contemporáneos, con solo dos hiperparámetros de interpretación intuitiva.

Para la **Clase 32 (Olvido Catastrófico)** este paper importa porque la clase presenta HAT como el representante canónico de los **métodos de arquitectura** (en contraste con los métodos de regularización tipo EWC/SI y los de *rehearsal*/memoria): el método que usa atención por tarea para determinar la importancia de cada peso y proteger los relevantes. Entender HAT es entender cómo se puede particionar dinámicamente la capacidad de una red entre tareas sin pre-asignarla a ciegas.

## 2. Contexto: el olvido catastrófico y las tres familias de soluciones

El olvido catastrófico (también *interferencia catastrófica*) fue descrito por McCloskey & Cohen (1989) y Ratcliff (1990): cuando una red se entrena a convergencia en una tarea y luego en una segunda, olvida cómo resolver la primera. El paper sitúa el problema en el marco del *lifelong learning* (Thrun & Mitchell, 1995) y lo motiva con ejemplos prácticos —un robot que no puede reentrenar su modelo desde cero cada vez que encuentra un objeto nuevo, o el costo prohibitivo del entrenamiento multitarea concurrente a escala—. El paper organiza las soluciones previas en familias bien diferenciadas, lo que es útil para ubicar a HAT:

- **Rehearsal y pseudo-rehearsal (memoria).** La estrategia más antigua (Robins, 1995) almacena instancias previas y las re-procesa al aprender la nueva tarea; trabajos modernos usan módulos de memoria (Rebuffi et al., 2017; Lopez-Paz & Ranzato, 2017) o redes generativas que sintetizan datos antiguos (Shin et al., 2017; Nguyen et al., 2017). El problema: ambas implican alguna forma de *aprendizaje concurrente* —re-procesar lo viejo— y entrenar un generador para una secuencia de tareas es difícil.

- **Regularización estructural (reducir el solapamiento representacional, de forma *suave*).** Se añade un término al *loss* que penaliza cambiar los pesos importantes de tareas anteriores. Aquí caen **EWC** (Elastic Weight Consolidation; Kirkpatrick et al., 2017), que mide la importancia de cada peso *después* de entrenar vía la información de Fisher, y **SI** (Synaptic Intelligence; Zenke et al., 2017), que la calcula *durante* el entrenamiento. **IMM** (Lee et al., 2017) es una evolución de EWC con un paso separado de fusión de modelos. Estos métodos previenen cambios pero de manera *blanda*: el peso puede moverse, solo paga un costo.

- **Métodos de arquitectura (dedicar sub-partes de la red a cada tarea).** Aquí vive HAT. **Progressive Neural Networks** (PNN; Rusu et al., 2016) asignan una columna de pesos por tarea con *adapters* para reutilizar conocimiento, pero el número de parámetros crece sin tope. **PathNet** (Fernando et al., 2017) pre-asigna capacidad y usa un algoritmo genético para encontrar *paths* entre módulos. **PackNet** (Mallya & Lazebnik, 2017), aparecido durante el desarrollo de HAT, encuentra una máscara binaria *sobre los pesos* mediante poda heurística con ratios de compresión pre-asignados. **DEN** (Dynamically Expandable Networks; Yoon et al., 2018) expande y reentrena selectivamente con una mezcla compleja de heurísticas.

El *trade-off* transversal a todo el campo —y que el paper nombra explícitamente— es **estabilidad vs. plasticidad**: distribuir efectivamente la capacidad de la red entre tareas, manteniendo los pesos importantes (estabilidad) y reutilizando conocimiento previo sin bloquear el aprendizaje nuevo (plasticidad). HAT se posiciona como el método de arquitectura que controla ese balance de forma *aprendida* y con grano fino (a nivel de unidad, no de columna ni de módulo).

## 3. Contribución central: Hard Attention to the Task

La contribución de HAT puede resumirse en una frase: **aprender, por tarea y mediante backpropagation, máscaras de atención dura (casi binarias) sobre las unidades de cada capa, y usar las máscaras acumuladas de tareas previas para condicionar el gradiente de las tareas futuras, protegiendo los pesos importantes.**

Las piezas conceptuales:

1. **Atención por unidad, no por peso.** A diferencia de PackNet (máscara sobre pesos) o PNN/PathNet (sobre columnas/módulos), HAT pone una máscara sobre las **unidades** (neuronas *fully-connected* o filtros convolucionales) de cada capa. La máscara sobre los pesos se *deriva automáticamente* de las máscaras de unidades. Esto da una estructura **liviana** (se almacena un vector por capa, no una matriz) y permite que la propia red dimensione cuántas unidades dedica a cada tarea.

2. **La máscara está condicionada por un *embedding* de tarea.** Cada tarea $t$ tiene su propio *embedding* $e^t_l$ por capa, que al pasar por un *gate* sigmoide produce el vector de atención $a^t_l$. Es la *identidad de la tarea* la que activa o desactiva unidades.

3. **Las máscaras son *casi binarias* y actúan como sinapsis inhibitorias.** Inspirándose en McCulloch & Pitts (1943), la atención $a^t_l \to \{0,1\}$ activa o desactiva la salida de cada unidad. Al ser casi binarias, dinámicamente *crean y destruyen paths* a través de las capas (como PathNet), pero a nivel de unidad individual y sin necesidad de pre-asignar tamaño de módulo ni número máximo de módulos.

4. **Protección del gradiente.** Las máscaras acumuladas de todas las tareas previas se usan para **modular el gradiente** de la tarea actual: los pesos que fueron importantes antes quedan congelados; los demás se adaptan. Esta es una regularización estructural *dura* (a diferencia de la *suave* de EWC/SI), aplicada tanto en el *loss* como directamente sobre la magnitud del gradiente.

5. **Penalización de capacidad para dejar espacio libre.** Un término de regularización promueve esparsidad en las atenciones, de modo que cada tarea use la menor cantidad de unidades posible y reserve capacidad para tareas futuras.

Las diferencias con PackNet, que el paper detalla, son tres y resumen la filosofía: (a) la máscara de HAT es **basada en unidades** (las de peso se derivan), lo que da una estructura más liviana; (b) la máscara es **aprendida**, no heurística ni dirigida por reglas —no hay que pre-asignar ratios de compresión—; y (c) la máscara **no es necesariamente binaria**, ya que el parámetro de estabilidad controla el grado de binarización, lo que habilita reutilización parcial de pesos o un modo más *online*.

## 4. Método: cómo se entrena HAT

### 4.1. La máscara de atención (forward pass)

Dada la salida de las unidades de la capa $l$, $h_l$, HAT la multiplica elemento a elemento por la máscara de atención: $h'_l = a^t_l \odot h_l$. A diferencia de la atención clásica, $a^t_l$ **no forma una distribución de probabilidad**: es la versión *gateada* de un *embedding* de tarea de una sola capa,

$$a^t_l = \sigma(s\, e^t_l),$$

donde $\sigma$ es un *gate* sigmoide y $s>0$ es un **parámetro de escala**. Todas las capas $l = 1,\dots,L-1$ operan igual; la última capa $L$ tiene una máscara binaria fija (*hard-coded*), equivalente a una salida *multi-head* —un cabezal de salida por tarea—, práctica habitual en olvido catastrófico.

### 4.2. *Annealing* del *gating* sigmoide (hard attention training)

Para obtener una máscara totalmente binaria bastaría una función escalón, pero el escalón no es diferenciable y los *embeddings* $e^t_l$ se entrenan con backpropagation. La solución es usar la sigmoide con escala $s$ como **pseudo-escalón diferenciable**: cuando $s \to \infty$, $a^t_{l,i} \to \{0,1\}$ (escalón duro); cuando $s \to 0$, $a^t_{l,i} \to 1/2$ (todas las unidades igualmente activas).

La estrategia es **recocer (annealing) $s$ durante el entrenamiento**: cada época *empieza* con $s$ bajo (todas las unidades activas, máxima plasticidad para explorar) y se va incrementando linealmente a lo largo de los *batches* de la época para polarizar progresivamente las unidades:

$$s = \frac{1}{s_{\max}} + \left(s_{\max} - \frac{1}{s_{\max}}\right)\frac{b-1}{B-1},$$

con $b$ el índice de *batch* y $B$ el total de *batches*. En *test* se fija $s = s_{\max} \gg 1$, de modo que la sigmoide aproxima un escalón unitario y las máscaras son efectivamente binarias. El hiperparámetro $s_{\max}$ controla el balance estabilidad/plasticidad: un $s_{\max}$ cercano a 1 hace que el *gate* opere como sigmoide normal (mucha plasticidad, la red puede olvidar); un $s_{\max}$ grande lo vuelve un escalón (mucha estabilidad, los pesos importantes quedan congelados).

### 4.3. Modificación del gradiente con máscaras acumuladas (backward pass)

Para preservar la información de tareas previas, HAT **acumula** las atenciones. Tras aprender la tarea $t$ y obtener $a^t_l$, se calcula recursivamente la atención acumulada con el máximo elemento a elemento:

$$a^{\le t}_l = \max\!\left(a^t_l,\, a^{\le t-1}_l\right),$$

partiendo del vector cero. Así se preservan las atenciones de cualquier unidad que haya sido importante en *alguna* tarea previa. Para condicionar el entrenamiento de la tarea $t+1$, el gradiente $g_{l,ij}$ del peso que conecta la unidad $j$ de la capa $l-1$ con la unidad $i$ de la capa $l$ se modula con el **reverso del mínimo de la atención acumulada en la capa actual y la anterior**:

$$g'_{l,ij} = \left[1 - \min\!\left(a^{\le t}_{l,i},\, a^{\le t}_{l-1,j}\right)\right] g_{l,ij}.$$

Intuición: un peso solo se considera "protegido" si **tanto** la unidad de entrada **como** la de salida fueron importantes en alguna tarea previa (de ahí el mínimo). Si $a^{\le t} \to 1$ para ambas unidades, el factor $1 - \min(\cdot) \to 0$ y el gradiente se anula: el peso queda congelado. Si alguna unidad estaba libre, el peso puede actualizarse. Esto deriva automáticamente una máscara *sobre pesos* a partir de las máscaras *sobre unidades*. (Sobre datos de entrada complejos como imágenes o audio no se aplica atención; si los datos fueran características separadas, podrían tratarse como salida de una capa y aplicarles la misma metodología.)

### 4.4. Compensación del gradiente del *embedding*

En análisis preliminares los autores observaron que los *embeddings* $e^t_l$ casi no cambiaban: el gradiente sobre ellos era débil, efecto colateral del *annealing* de $s$. Para corregirlo introducen una **compensación del gradiente del embedding**: dividen el gradiente $q_{l,i}$ por la derivada de la sigmoide recocida y lo multiplican por la compensación deseada, lo que tras simplificar da

$$q'_{l,i} = \frac{s_{\max}\,[\cosh(s\,e^t_{l,i}) + 1]}{s\,[\cosh(e^t_{l,i}) + 1]}\, q_{l,i}.$$

Esto restaura un rango y una magnitud de gradiente deseables (rango amplio como con $s=1$, magnitud grande como con $s=s_{\max}$). Por estabilidad numérica se hace *clamp* $|s\,e^t_{l,i}| \le 50$ y se restringe $e^t_{l,i} \in [-6,6]$.

### 4.5. Regularización de capacidad (promoting low capacity usage)

Como las unidades con $a^t_{l,i} \to 1$ quedan "dedicadas" a la tarea $t$, hay que reservar capacidad para el futuro promoviendo **esparsidad** en las atenciones. HAT añade al *loss* un término

$$L' = L(y,\hat y) + c\, R(A^t, A^{<t}),$$

donde $R$ es una **regularización L1 ponderada y normalizada** sobre las atenciones de la tarea actual. El peso de cada unidad lo definen las atenciones acumuladas de tareas previas: si $a^{<t}_{l,i} \to 1$ (unidad ya usada antes), su atención actual recibe peso $\approx 0$ y queda *excluida* de la regularización —se incentiva su reutilización sin penalizar—. El hiperparámetro $c \ge 0$ actúa como **constante de compresibilidad**: a mayor $c$, menos unidades activas y red más esparsa. Se fija un $c$ global y HAT adapta la compresión a cada tarea individual. (A diferencia del L1 plano sobre pesos de DEN, HAT regulariza las *atenciones de unidad*, en una única fase de entrenamiento sin heurísticas adicionales.)

## 5. Experimentos

**Protocolo de evaluación.** Los autores critican los *setups* habituales (permutaciones de MNIST, *splits* de MNIST, transferencia entre dos tareas) por sesgar las conclusiones —las permutaciones de MNIST favorecen artificialmente a ciertos métodos por la abundancia de valores cercanos a 0—. Su protocolo principal usa **secuencias aleatorias de 8 datasets de clasificación de imágenes** distintos (CIFAR10, CIFAR100, FaceScrub, FashionMNIST, NotMNIST, MNIST, SVHN, TrafficSigns), todos adaptados a entradas $32\times32\times3$, con 10 a 100 clases. El orden de las tareas se aleatoriza y todo se repite **10 veces con 10 semillas**. La arquitectura base es tipo AlexNet (3 capas conv. de 64/128/256 filtros + 2 capas *fully-connected* de 2048), con todos los *baselines* igualados a 7.1 M de parámetros.

**Métrica.** Introducen el **forgetting ratio** $\rho^{\tau\le t}$, que normaliza la precisión de la tarea $\tau$ tras aprender $t$ tareas entre el clasificador aleatorio ($\rho \approx -1$) y el clasificador multitarea conjunto ($\rho \approx 0$). Se reporta el promedio $\rho^{\le t}$.

**Resultados en la secuencia de 8 tareas.** HAT supera consistentemente a los 11 *baselines* (SGD, SGD-F, EWC, IMM-Mean/Mode, LWF, LFL, PathNet, PNN) para todo $t \ge 2$:

- En $t=2$: HAT obtiene $\rho^{\le 2} = -0.02$ frente al mejor *baseline* EWC con $-0.08$ → **75% de reducción del olvido**.
- En $t=8$: HAT obtiene $\rho^{\le 8} = -0.06$ frente al mejor *baseline* PNN con $-0.11$ → **45% de reducción**.

Además, la desviación estándar de HAT es menor que la de casi todos los *baselines* (0.01 en $t=8$), lo que indica robustez frente a distintas secuencias, *splits*, datasets e inicializaciones. PathNet y PNN, por construcción, nunca olvidan, pero pierden capacidad de aprendizaje al pre-asignar pesos: PathNet se degrada para $t \ge 2$ y PNN sufre en las primeras tareas; ninguno supera al resto consistentemente como sí lo hace HAT.

**Setups adicionales.**
- *Incremental class learning* (CIFAR10/100): HAT $\rho^{\le 10} = -0.09$ vs. mejor *baseline* EWC $-0.18$ → **55% de reducción**.
- *Permuted MNIST*: HAT $A^{\le 10} = 98.6\%$ vs. SI $97.1\%$ → **52% de reducción de la tasa de error**.
- *Split MNIST*: HAT $A^{\le 2} = 99.0\%$ vs. *conceptor-aided backpropagation* $94.9\%$ → **80% de reducción del error**.

**Hiperparámetros y monitoreo.** HAT tiene solo dos hiperparámetros con interpretación directa: $s_{\max}$ (estabilidad/plasticidad) y $c$ (compacidad). Operan bien en rangos amplios ($s_{\max} \in [25,800]$, $c \in [0.1,2.5]$); por defecto se usa $s_{\max}=400$, $c=0.75$. El mecanismo además permite **monitorear** el uso de capacidad por época y capa, y la **reutilización de pesos entre tareas**. Como subproducto, sirve para **poda y compresión de red**: con $c=1.5$ e inicialización $U(0,2)$, HAT comprime la red a entre el 1% y el 21% de su tamaño original (más compacto que el 25–50% de PackNet o el 18–52% de DEN), aprendiendo la poda vía backpropagation simultáneamente a los pesos.

## 6. Limitaciones

- **Requiere el identificador de tarea (*task ID*) en inferencia.** Es la limitación más relevante. HAT necesita saber a qué tarea pertenece cada entrada para seleccionar la máscara y el cabezal de salida correcto. Esto lo ubica en el escenario *task-incremental* (con *task ID* conocido), no en el más difícil *class-incremental* (sin *task ID*), donde el modelo debe inferir también a qué tarea pertenece la entrada. Para muchas aplicaciones reales el *task ID* no está disponible en *test*.
- **Capacidad finita.** Aunque la regularización de capacidad reserva espacio, la red base es de tamaño fijo; con suficientes tareas el espacio libre se agota y la plasticidad cae. HAT lo mitiga con compresión adaptativa, pero no lo elimina.
- **Salida *multi-head* fija.** La última capa es binaria *hard-coded* por tarea, lo que es consistente con el supuesto de *task ID* conocido pero limita los escenarios aplicables.
- **Evaluación acotada a clasificación de imágenes.** El protocolo, aunque riguroso, se centra en visión; la extensión a otros dominios y a entradas complejas (sobre las que no se aplica atención) queda esbozada pero no demostrada.

## 7. Impacto

HAT se consolidó como uno de los **métodos de arquitectura de referencia** en aprendizaje continuo, junto a PackNet, PNN y PathNet, y es citado sistemáticamente como el ejemplo canónico de *parameter isolation* mediante máscaras aprendidas a nivel de unidad. Su combinación de bajo olvido, robustez a hiperparámetros, estructura liviana y capacidad de monitoreo lo volvió un *baseline* obligado en benchmarks posteriores de *continual learning*. Su limitación de necesitar *task ID* impulsó líneas de trabajo sucesoras que intentan relajar ese supuesto hacia el escenario *class-incremental*. El mecanismo de *gating* con *annealing* del factor de escala —para obtener máscaras casi binarias diferenciables— ha sido reutilizado y adaptado más allá del olvido catastrófico, en contextos de compresión de red y *online learning*, tal como anticipan los propios autores en la conclusión.

## 8. Conexión con la Clase 32 (Olvido Catastrófico)

La Clase 32 organiza las soluciones al olvido catastrófico en tres grandes familias —**regularización** (EWC, SI), **rehearsal/memoria** (replay, generativo) y **arquitectura**— y presenta **HAT como el representante de los métodos de arquitectura** (slide HAT): el método que usa **atención por tarea para determinar la importancia de los pesos** y proteger los relevantes de cambios futuros.

El valor pedagógico de HAT en este punto del curso es triple:

1. **Hace tangible el *trade-off* estabilidad–plasticidad.** El parámetro $s_{\max}$ es literalmente una perilla que mueve el modelo entre "recordar todo" (estabilidad, máscaras duras, pesos congelados) y "aprender rápido lo nuevo" (plasticidad, sigmoide blanda). La clase puede usarlo para ilustrar concretamente el dilema central del aprendizaje continuo.

2. **Contrasta la regularización *dura* vs. la *suave*.** Frente a EWC/SI —que penalizan *suavemente* mover pesos importantes vía un término en el *loss*—, HAT impone una protección *dura*: anula el gradiente de los pesos protegidos. Comparar ambos enfoques sobre la misma métrica (forgetting ratio) es el tipo de análisis que la clase busca.

3. **Muestra el grano fino de la atención por unidad.** A diferencia de los métodos que pre-asignan columnas (PNN) o módulos (PathNet) a ciegas, HAT *aprende* qué unidades importan, de forma adaptativa y sin heurísticas previas. Esto conecta el concepto de "atención" —ya familiar del módulo de NLP— con un uso nuevo: no para ponderar tokens, sino para particionar la capacidad de la red entre tareas.

Para profundizar en el marco general del que HAT es un caso, ver el fundamento [/fundamentos/aprendizaje-continuo](/fundamentos/aprendizaje-continuo) y la página de la clase [/clases/clase-32](/clases/clase-32).
