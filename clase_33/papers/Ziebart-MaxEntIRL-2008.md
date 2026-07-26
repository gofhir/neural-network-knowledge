# Maximum Entropy Inverse Reinforcement Learning — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Maximum Entropy Inverse Reinforcement Learning*.
- **Autores:** Brian D. Ziebart, Andrew Maas, J. Andrew Bagnell, Anind K. Dey (todos de la **School of Computer Science, Carnegie Mellon University**).
- **Venue:** *Proceedings of the Twenty-Third AAAI Conference on Artificial Intelligence* (AAAI 2008), pp. 1433–1438.
- **Año:** 2008.
- **Una línea:** aplica el **principio de máxima entropía** de Jaynes para resolver la ambigüedad estructural del aprendizaje por refuerzo inverso (IRL) —el hecho de que muchas funciones de recompensa, y muchas distribuciones sobre trayectorias, explican igual de bien el mismo comportamiento demostrado— eligiendo la distribución **exponencial sobre trayectorias** $P(\zeta) \propto e^{\theta^\top f_\zeta}$, que iguala las esperanzas de features observadas sin introducir ningún sesgo adicional.

El trabajo de Ziebart y colaboradores es uno de los pilares canónicos del IRL moderno. Aunque no aparezca citado textualmente en las diapositivas de la Clase 33, resuelve de forma limpia el problema conceptual que dejaban abierto los dos antecedentes fundacionales del área: **Ng & Russell (2000)**, que planteó el IRL como problema formal, y **Abbeel & Ng (2004)**, que propuso el *apprenticeship learning* vía matching de feature expectations. Ambos dejaban una ambigüedad irresuelta: la función de recompensa que hace óptimo el comportamiento observado **no es única**. MaxEnt IRL cierra ese hueco con una respuesta probabilística principiada.

El paper está motivado por un problema muy concreto y de gran escala: **modelar las preferencias de ruta de conductores reales** a partir de más de **100.000 millas** de datos GPS recolectados de taxis en Pittsburgh. El resultado no es solo un algoritmo de imitación, sino un modelo probabilístico completo que —a diferencia de los métodos de margen— se integra de forma natural con otras técnicas probabilísticas, permitiendo por ejemplo inferir **destinos y rutas futuras** a partir de trayectorias parciales mediante el teorema de Bayes. Los autores destacan además que su enfoque maneja de manera principiada el hecho de que el comportamiento humano demostrado es **ruidoso e imperfecto**, algo que los métodos anteriores no capturaban bien.

Para la **Clase 33 (Aprendizaje por Imitación y Aprendizaje Reforzado Inverso)**, este paper es la pieza que transforma el IRL de una idea elegante pero mal definida en una herramienta estadística robusta. Introduce un modelo que "normaliza globalmente sobre comportamientos" y que puede entenderse como una extensión de los *chain conditional random fields* que incorpora la dinámica del sistema de planificación y se extiende al horizonte infinito.

## 2. Contexto: la ambigüedad del IRL previo

### 2.1. El problema del aprendizaje por imitación estructurado

En el aprendizaje por imitación, la meta es predecir el comportamiento y las decisiones que tomaría un agente: los movimientos de una persona para agarrar un objeto, o la ruta que un conductor elegiría de su casa al trabajo. Capturar este comportamiento **secuencial y con propósito** es difícil para el aprendizaje estadístico de propósito general, porque los algoritmos deben razonar sobre las consecuencias de acciones **muy adelante en el futuro**.

La idea potente que había emergido —y que la Clase 33 recoge— es estructurar el espacio de políticas aprendidas como **soluciones de un Problema de Decisión de Markov (MDP)**. La intuición: los agentes actúan para optimizar una función de recompensa desconocida (que se asume **lineal en las features**), y hay que encontrar los pesos de recompensa que hacen que el comportamiento demostrado parezca (casi) óptimo. El problema de imitación se reduce entonces a recuperar una función de recompensa que induzca el comportamiento demostrado, con el algoritmo de búsqueda sirviendo para "coser" secuencias largas y coherentes de decisiones.

Formalmente, se observa la trayectoria $\zeta$ de un agente, compuesta de estados $s_i$ y acciones $a_i$. Cada estado tiene un vector de features $f_{s_j} \in \mathbb{R}^k$ que se mapea a un valor de recompensa mediante los pesos $\theta$. La recompensa de una trayectoria es la suma de recompensas de estados, que equivale a aplicar los pesos al **conteo de features del camino**:

$$\text{reward}(f_\zeta) = \theta^\top f_\zeta = \sum_{s_j \in \zeta} \theta^\top f_{s_j}, \qquad f_\zeta = \sum_{s_j \in \zeta} f_{s_j}.$$

El agente demuestra trayectorias individuales $\tilde\zeta_i$, de las cuales se obtiene el **conteo de features empírico esperado** promediando sobre las $m$ demostraciones:

$$\tilde f = \frac{1}{m}\sum_i f_{\tilde\zeta_i}.$$

### 2.2. La ambigüedad: un problema mal planteado

Recuperar los pesos de recompensa exactos del agente es un **problema mal planteado** (*ill-posed*). Muchos pesos de recompensa —incluyendo degeneraciones triviales como el vector de todos ceros— hacen que las trayectorias demostradas sean óptimas. Esta es la ambigüedad fundamental que arrastraba el IRL desde Ng & Russell (2000): **cada política puede ser óptima para muchas funciones de recompensa, y muchas políticas conducen a los mismos conteos de features**.

Los antecedentes atacaban el problema por caminos distintos, pero ninguno resolvía la ambigüedad:

- **Maximum Margin Planning (MMP)** de Ratliff, Bagnell & Zinkevich (2006) plantea el IRL como predicción estructurada de máximo margen: usa funciones de pérdida que miden el desacuerdo entre el agente y la política aprendida, y aprende la recompensa vía una relajación convexa. El problema: **falla cuando ninguna función de recompensa hace el comportamiento demostrado simultáneamente óptimo y significativamente mejor que las alternativas**. Esto ocurre con frecuencia cuando el comportamiento del agente es imperfecto, o cuando el algoritmo de planificación solo captura parte del espacio de estados relevante.

- **Abbeel & Ng (2004)** proponen el *matching de feature expectations*: igualar las esperanzas de features entre la política observada y la del aprendiz. Demuestran que este matching es **necesario y suficiente** para alcanzar el mismo desempeño que el agente, si este realmente resolvía un MDP con recompensa lineal en esas features. La condición central es:

$$\sum_{\zeta_i} P(\zeta_i)\, f_{\zeta_i} = \tilde f. \tag{1}$$

El punto crítico que el paper subraya: **tanto el concepto de IRL como el matching de conteos de features son ambiguos**. Cuando se demuestra comportamiento subóptimo, se requieren mezclas de políticas para igualar los conteos de features, y muchas mezclas distintas satisfacen esa condición. **Ninguno de los métodos anteriores propone una forma de resolver esa ambigüedad**: cualquier distribución dentro del conjunto consistente puede exhibir una preferencia por ciertos caminos sobre otros que no está implicada por las features. Ahí es donde entra la máxima entropía.

## 3. Contribución central: un marco probabilístico

La contribución de Ziebart et al. es adoptar un enfoque **completamente probabilístico** para razonar sobre la incertidumbre en la imitación. En lugar de razonar sobre políticas, consideran una distribución sobre **toda la clase de comportamientos posibles** —es decir, sobre caminos de longitud (potencialmente) variable.

El paso clave es reconocer que, entre todas las distribuciones sobre caminos que satisfacen el matching de feature expectations (Ecuación 1), hay que elegir una. El **principio de máxima entropía** (Jaynes, 1957) resuelve esta ambigüedad de forma canónica: elegir la distribución que **no exhibe ninguna preferencia adicional más allá de la que imponen las restricciones**. Es la distribución "menos comprometida" —la de máxima incertidumbre— consistente con lo observado. Cualquier otra distribución consistente estaría inyectando información que los datos no justifican.

El resultado formal es que la distribución de máxima entropía sujeta a la restricción de matching de features, para MDPs deterministas, es una **distribución exponencial (de Boltzmann) parametrizada por los pesos de recompensa** $\theta$:

$$P(\zeta_i \mid \theta) = \frac{1}{Z(\theta)}\, e^{\theta^\top f_{\zeta_i}} = \frac{1}{Z(\theta)}\, e^{\sum_{s_j \in \zeta_i} \theta^\top f_{s_j}}. \tag{2}$$

Bajo este modelo, la interpretación es directa y elegante:

- **Planes con recompensa equivalente tienen probabilidad equivalente.** El modelo no favorece arbitrariamente un camino sobre otro de igual retorno de features —justamente lo que exige la máxima entropía.
- **Planes con mayor recompensa son exponencialmente preferidos.** La probabilidad crece exponencialmente con el retorno $\theta^\top f_\zeta$.

Esto es lo que distingue radicalmente a MaxEnt IRL de los modelos que **normalizan localmente** en cada estado (como los usados en Bayesian IRL de Ramachandran & Amir 2007 o el IRL híbrido de Neu & Szepesvári 2007): al normalizar **globalmente** sobre trayectorias completas mediante la función de partición $Z(\theta)$, el modelo evita el problema de **sesgo de etiqueta** (*label bias*) heredado de la literatura de CRFs, que se discute en la sección de experimentos.

### 3.1. Convergencia de la función de partición

Un detalle técnico importante: dados los pesos, la función de partición $Z(\theta)$ **siempre converge** para problemas de horizonte finito y para horizonte infinito con recompensas descontadas. Para problemas de horizonte infinito con estados absorbentes de recompensa cero, $Z(\theta)$ puede **no converger** incluso cuando todas las recompensas de estado son negativas. Sin embargo, dado que las trayectorias demostradas se absorben en un número finito de pasos, los pesos de recompensa que maximizan la entropía deben ser convergentes. Esta observación es la que hace viable aplicar el modelo a los datos reales de taxis.

## 4. Método: formulación, MLE, gradiente y algoritmo

### 4.1. Distribuciones de caminos no deterministas

En MDPs generales, las acciones producen transiciones no deterministas según la distribución de transición $T$. Los caminos quedan determinados tanto por las decisiones del agente como por los resultados aleatorios del entorno. La distribución exacta sobre caminos condicionada a $T$ requiere sumar sobre el espacio de resultados de acción (*outcome samples* $o$), y es en general **intratable**:

$$P(\zeta \mid \theta, T) = \sum_{o \in T} P_T(o)\, \frac{e^{\theta^\top f_\zeta}}{Z(\theta, o)}\, I_{\zeta \in o}, \tag{3}$$

donde $I_{\zeta \in o}$ es la función indicadora que vale 1 cuando el camino es compatible con el resultado $o$. Para hacerlo tratable, los autores introducen una **suposición simplificadora**: asumen que la aleatoriedad de transición tiene un efecto limitado sobre el comportamiento y que la función de partición es aproximadamente constante para todos los $o \in T$. Esto produce una distribución aproximada tratable:

$$P(\zeta \mid \theta, T) \approx \frac{e^{\theta^\top f_\zeta}}{Z(\theta, T)} \prod_{s_{t+1}, a_t, s_t \in \zeta} P_T(s_{t+1} \mid a_t, s_t). \tag{4}$$

Esta distribución sobre caminos induce una **política estocástica** (una distribución sobre las acciones disponibles en cada estado) cuando la función de partición converge. La probabilidad de una acción queda ponderada por las recompensas exponenciadas esperadas de todos los caminos que comienzan con esa acción:

$$P(\text{acción } a \mid \theta, T) \propto \sum_{\zeta: a \in \zeta_{t=0}} P(\zeta \mid \theta, T). \tag{5}$$

### 4.2. Aprendizaje por máxima verosimilitud

Maximizar la entropía de la distribución sujeta a las restricciones de features es **equivalente a maximizar la verosimilitud** de los datos observados bajo la distribución exponencial (familia exponencial) derivada arriba —una dualidad clásica de Jaynes. El objetivo es:

$$\theta^* = \arg\max_\theta L(\theta) = \arg\max_\theta \sum_{\text{ejemplos}} \log P(\tilde\zeta \mid \theta, T).$$

Esta función es **convexa** para MDPs deterministas, de modo que los óptimos se obtienen con métodos de optimización basados en gradiente. Esta convexidad es una ventaja notable frente a los modelos que normalizan localmente, que llevan a problemas de optimización no convexos con múltiples mínimos.

### 4.3. El gradiente: features empíricas menos esperadas

El corazón del método es la forma del gradiente. Resulta ser la **diferencia entre los conteos de features empíricos esperados y los conteos de features esperados del aprendiz**, que a su vez se expresan en términos de las **frecuencias esperadas de visita a estados** $D_{s_i}$:

$$\nabla L(\theta) = \tilde f - \sum_\zeta P(\zeta \mid \theta, T)\, f_\zeta = \tilde f - \sum_{s_i} D_{s_i}\, f_{s_i}. \tag{6}$$

Esta expresión es profundamente intuitiva y estructuralmente idéntica al gradiente de un modelo de máxima entropía / familia exponencial estándar (como un CRF): en el óptimo, cuando $\nabla L = 0$, **las esperanzas de features del modelo igualan las empíricas**. Eso garantiza —invocando el resultado de Abbeel & Ng (2004)— que el aprendiz se desempeña de forma equivalente al comportamiento demostrado del agente, **sin importar cuáles fueran los pesos de recompensa reales** que el agente intentaba optimizar. El aprendizaje se reduce a "empujar" $\theta$ hasta que las visitas esperadas de estados generen el mismo perfil de features que se observó.

Los autores señalan que en la práctica se miden esperanzas empíricas basadas en muestras, no las verdaderas. Un argumento estándar de cotas de unión y Hoeffding provee cotas de alta probabilidad sobre el error, con una dependencia de solo $O(\log K)$ en el número de features $K$ —en contraste con los métodos de margen y los localmente normalizados, que escalan **linealmente** en el número de features. Además, conectando con Dudík & Schapire (2006), la máxima entropía con incertidumbre acotada en las esperanzas de features equivale a un problema de máximo a posteriori con **regularización $l_1$**, cuya fuerza depende de la incertidumbre de cada feature; en los experimentos usan descenso de gradiente exponenciado en línea, eficiente y con efecto regularizador tipo $l_1$.

### 4.4. El algoritmo forward-backward para las frecuencias de visita

El gradiente (Ecuación 6) es fácil de computar **una vez que se conocen las frecuencias esperadas de visita a estados** $D_{s_i}$. El enfoque ingenuo —enumerar todos los caminos posibles— es inviable por el crecimiento exponencial de caminos con el horizonte temporal del MDP. La solución de los autores es un algoritmo eficiente análogo al **algoritmo forward-backward de los Conditional Random Fields** o a la **iteración de valor** del aprendizaje por refuerzo. Aproxima las frecuencias del horizonte infinito usando un horizonte fijo grande $N$.

El **Algoritmo 1 (Expected Edge Frequency Calculation)** tiene tres fases:

**Paso hacia atrás (backward pass):** calcula la masa de probabilidad "hacia atrás" desde cada estado terminal posible, computando la función de partición local en cada acción y estado.

1. Inicializar $Z_{s_i, 0} = 1$.
2. Recursivamente, para $N$ iteraciones:
$$Z_{a_{i,j}} = \sum_k P(s_k \mid s_i, a_{i,j})\, e^{\text{reward}(s_i \mid \theta)}\, Z_{s_k}, \qquad Z_{s_i} = \sum_{a_{i,j}} Z_{a_{i,j}}.$$

**Cálculo de probabilidades de acción locales:**

3. $P(a_{i,j} \mid s_i) = \dfrac{Z_{a_{i,j}}}{Z_{s_i}}$.

**Paso hacia adelante (forward pass):** propaga la masa de probabilidad desde el estado inicial.

4. Inicializar $D_{s_i, t} = P(s_i = s_{\text{inicial}})$.
5. Recursivamente, para $t = 1$ hasta $N$:
$$D_{s_i, t+1} = \sum_{a_{i,j}} \sum_k D_{s_k, t}\, P(a_{i,j} \mid s_i)\, P(s_k \mid a_{i,j}, s_i).$$

**Suma de frecuencias:**

6. $D_{s_i} = \sum_t D_{s_i, t}$.

La estructura es transparente: el *backward pass* "retrocede" desde cada estado terminal calculando las particiones parciales, de las que salen las probabilidades de acción locales (Paso 3); el *forward pass* propaga la masa desde el inicio para obtener las frecuencias por timestep, que se suman para el total. Este algoritmo de complejidad polinomial es lo que hace escalable a MaxEnt IRL sobre un MDP con cientos de miles de estados.

## 5. Experimentos: predicción de rutas de taxistas en Pittsburgh

### 5.1. El problema como MDP de gran escala

El trabajo estuvo motivado por el modelado de las decisiones de ruta de conductores. La red vial de Pittsburgh se modela como un **MDP determinista con más de 300.000 estados** (segmentos de carretera) y **900.000 acciones** (transiciones en intersecciones). Se asume que los conductores intentan alcanzar una meta mientras optimizan un compromiso entre tiempo, seguridad, estrés, combustible, mantenimiento y otros factores —un valor que los autores llaman **costo** (recompensa negativa). El destino se representa como un **estado absorbente** sin costos adicionales. Como los pesos de recompensa se asumen independientes del estado meta, un único vector de pesos puede aprenderse de muchos MDPs que difieren solo en su destino. Los autores señalan que, hasta donde saben, este era el **problema de IRL de mayor escala investigado hasta la fecha** en términos de tamaño de datos demostrados.

### 5.2. Datos GPS

Se recolectaron trazas GPS de **25 taxis Yellow Cab durante 12 semanas** a toda hora, produciendo más de **100.000 millas** de viaje en más de **3.000 horas** de conducción alrededor de Pittsburgh. Un filtro de partículas ajustó los datos GPS dispersos a la red vial, y las trazas se segmentaron en unos **13.000 viajes** distintos usando un umbral temporal para detectar paradas. Se descartó cerca del 30% de los viajes (demasiado cortos —menos de 10 segmentos—, demasiado cíclicos o ruidosos). Del resto, se dividió 20% en entrenamiento y 80% en un conjunto de prueba de **7.403 ejemplos**.

Las **features de camino** cubren cuatro dimensiones de características del segmento: tipo de carretera, velocidad, número de carriles y tipo de transición (recto, izquierda, derecha, giro cerrado a izquierda o derecha), dando 22 conteos distintos. Un camino se describe por cuántas millas de cada categorización contiene y el número de cada tipo de transición.

### 5.3. El problema del sesgo de etiqueta (label bias)

Los autores ilustran con la Figura 2 la diferencia clave frente a los modelos basados en acción (*Action*), que asignan probabilidad a cada acción según la recompensa futura esperada de la mejor política tras tomarla, $P(a \mid s) \propto e^{Q^*(s,a)}$. En un ejemplo con tres caminos de A a B de igual recompensa, el modelo MaxEnt les da **probabilidad igual (1/3 cada uno)**, mientras que el modelo basado en acción da 50% a uno y 25% a los otros dos, simplemente por la estructura de ramificación. Esto es el **sesgo de etiqueta** de la literatura de CRFs (Lafferty, McCallum & Pereira, 2001): los caminos solo compiten por masa de probabilidad localmente a nivel de acción, no contra caminos que se ramificaron antes. La consecuencia indeseable: **la política de mayor recompensa puede no ser la más probable**, y políticas de igual recompensa esperada pueden tener probabilidades distintas. MaxEnt evita esto porque normaliza globalmente.

### 5.4. Evaluación comparativa

Se comparó la capacidad de cada modelo para predecir rutas del conjunto de prueba, dado origen y destino, con tres métricas: (1) porcentaje promedio de distancia de ruta compartida entre la predicción y la ruta real; (2) porcentaje de rutas de prueba que coinciden en al menos 90% de distancia; (3) log-probabilidad promedio de los caminos (solo posible en modelos de estimación de densidad). Los resultados (Tabla 1):

| Modelo | Matching | 90% Match | Log Prob |
|---|---|---|---|
| Time-based | 72.38% | 43.12% | N/A |
| Max Margin | 75.29% | 46.56% | N/A |
| Action | 77.30% | 50.37% | −7.91 |
| Action (costs) | 77.74% | 50.75% | N/A |
| **MaxEnt paths** | **78.79%** | **52.98%** | **−6.85** |

El modelo de máxima entropía muestra mejoras **estadísticamente significativas** ($\alpha < 0.01$) sobre todos los demás en cada métrica. Nótese que la mejor log-probabilidad (−6.85 vs −7.91 del modelo basado en acción) confirma que MaxEnt asigna densidad de manera más fiel, no solo predice mejor la ruta modal.

### 5.5. Inferencia de destinos con Bayes

Una ventaja distintiva del marco probabilístico: se puede inferir el **destino** a partir de un camino parcial aplicando el teorema de Bayes sobre el modelo de preferencia de ruta:

$$P(\text{dest} \mid \tilde\zeta_{A\to B}) \propto P(\tilde\zeta_{A\to B} \mid \text{dest})\, P(\text{dest}) \propto \frac{\sum_{\zeta_{B\to \text{dest}}} e^{\theta^\top f_\zeta}}{\sum_{\zeta_{A\to \text{dest}}} e^{\theta^\top f_\zeta}}\, P(\text{dest}).$$

Estas cantidades se computan directamente con el Algoritmo 1. En un experimento con cinco destinos alrededor de la ciudad (Figuras 4 y 5), la precisión posterior de predicción de destino crece a medida que se observa una fracción mayor del camino: un tramo que se dirige hacia el oeste es una ruta muy improbable hacia destinos orientales, lo que descarta esos destinos. Esto habilita aplicaciones como avisar de tráfico inesperado en la ruta sin preguntar explícitamente al usuario, optimizar el consumo de combustible en vehículos híbridos, o activar el clima del hogar antes de la llegada del conductor.

## 6. Limitaciones

- **Requiere un MDP conocido y (efectivamente) finito.** El método asume que la estructura del mundo (la red vial, las transiciones) es conocida. En el caso de los taxis esto se cumple porque el mapa está dado; pero en dominios donde la dinámica es desconocida el método necesita un modelo del entorno.
- **Cómputo de la función de partición.** El algoritmo forward-backward es polinomial, pero requiere iterar sobre todo el espacio de estados durante $N$ iteraciones. Para MDPs muy grandes esto sigue siendo costoso, y la aproximación del horizonte infinito por un horizonte fijo $N$ introduce error. En el caso de los taxis, los autores restringen el cálculo a "una clase más pequeña de caminos razonablemente buenos" en vez de todos los caminos posibles, para acelerar sin introducir no convexidad.
- **MDPs no deterministas solo aproximados.** La tratabilidad para MDPs estocásticos depende de la suposición simplificadora de que la aleatoriedad tiene efecto limitado y la partición es constante sobre los resultados (Ecuación 4). El tratamiento exacto (Ecuación 3) es intratable. Los autores esbozan un algoritmo tipo EM para el caso estocástico, pero la log-verosimilitud completa es no convexa y no cabía en el paper.
- **Recompensa lineal en features.** Como todo el linaje de Abbeel & Ng, se asume que la recompensa es lineal en features diseñadas a mano. La calidad del modelo depende de que esas features capturen lo relevante; features pobres limitan el poder expresivo. Esta es precisamente la limitación que Deep MaxEnt IRL levantaría años después.

## 7. Legado

MaxEnt IRL se convirtió en la formulación de referencia del IRL probabilístico y en la base directa de una genealogía de métodos posteriores muy influyentes:

- **Deep MaxEnt IRL (Wulfmeier et al., 2015)** reemplaza la recompensa lineal en features por una **red neuronal profunda** que aprende la función de recompensa directamente, manteniendo el marco de máxima entropía y el gradiente de matching de features. Levanta la limitación de las features hechas a mano.
- **Guided Cost Learning / GCL (Finn et al., 2016)** extiende MaxEnt IRL a entornos con dinámica desconocida y espacios continuos de alta dimensión, usando *importance sampling* para estimar la función de partición intratable, permitiendo IRL sobre tareas robóticas reales.
- **Conexión formal con GANs y GAIL.** Finn et al. (2016) y Ho & Ermon (**GAIL**, 2016) establecieron una equivalencia matemática profunda entre el IRL de máxima entropía y las **Generative Adversarial Networks**: la recompensa aprendida juega el rol del discriminador y la política el del generador. GAIL (*Generative Adversarial Imitation Learning*) explota esta conexión para hacer imitación sin recuperar explícitamente la recompensa, escalando a problemas donde MaxEnt IRL clásico no llegaba. Esta relación es una de las contribuciones conceptuales más citadas que descienden del paper de 2008.

La distribución de Boltzmann sobre trayectorias, $P(\zeta) \propto e^{\theta^\top f_\zeta}$, reaparece además como pieza central del **RL de máxima entropía** moderno (Soft Actor-Critic y afines), donde el objetivo agrega un término de entropía de la política para fomentar exploración y robustez —una línea que hereda directamente el espíritu de Jaynes que Ziebart et al. trajeron al control.

## 8. Conexión con la Clase 33

La Clase 33 (*Aprendizaje por Imitación y Aprendizaje Reforzado Inverso*, prof. Rodrigo Toro Icarte) contrasta tres enfoques para aprender comportamiento a partir de demostraciones: la **imitación directa** (behavioral cloning), el **IRL** (recuperar la recompensa y luego planificar), y la relación entre RL e imitación. MaxEnt IRL es el eslabón que profesionaliza el IRL:

- **Cierra la ambigüedad de Ng & Russell (2000) y Abbeel & Ng (2004).** La clase presenta el IRL como el problema de recuperar una recompensa que explique las demostraciones; MaxEnt es la respuesta canónica a la pregunta "¿cuál de las infinitas recompensas consistentes elegir?". La respuesta —la de máxima entropía— es la única que no inyecta sesgos que los datos no justifican.
- **Maneja demostradores subóptimos.** El aprendizaje por imitación asume implícitamente que el experto es casi óptimo; MaxEnt IRL modela explícitamente el ruido y la imperfección del comportamiento humano, dando probabilidad exponencialmente decreciente (pero no nula) a caminos peores. Esto lo hace mucho más robusto que el behavioral cloning ante demostraciones ruidosas.
- **Es el puente hacia el IRL profundo y GAIL.** Entender la formulación de 2008 es requisito para entender por qué GAIL y GCL —métodos modernos de imitación que la clase puede mencionar— tienen la estructura que tienen, y por qué el discriminador de una GAN equivale a una recompensa de máxima entropía.

Lecturas relacionadas dentro del dominio: [/fundamentos/aprendizaje-reforzado](/fundamentos/aprendizaje-reforzado) para los fundamentos de MDP, política, valor y recompensa; y [/papers/schulman-ppo-2017](/papers/schulman-ppo-2017) para el algoritmo de policy optimization que, vía RLHF, también aprende de preferencias humanas —un primo aplicado del problema de inferir intención humana que MaxEnt IRL formaliza.

---

**Nota para el lector experto en FHIR / MDM / record linkage.** La formulación de MaxEnt IRL ofrece una lente valiosa para modelar las **decisiones de match de un data steward humano**. En un flujo de *master data management*, un steward que resuelve manualmente pares candidatos (¿estos dos recursos `Patient` son la misma persona?) es exactamente un demostrador de comportamiento con propósito, sujeto a ruido e imperfección: aplica implícitamente una función de "recompensa" sobre features del par (concordancia de nombre, fecha de nacimiento, identificadores, dirección) que ningún manual documenta por completo. Modelar sus decisiones como una **distribución de máxima entropía** $P(\text{decisión}) \propto e^{w^\top f}$ sobre los features del par tiene tres virtudes concretas: primero, recupera los **pesos $w$ implícitos** del steward sin exigir que su comportamiento sea perfectamente consistente, tolerando que a veces se equivoque o dude —justo lo que MaxEnt hace con demostradores subóptimos, evitando el sobreajuste a decisiones aberrantes que sufriría un clasificador entrenado por imitación pura; segundo, produce **probabilidades calibradas globalmente** sobre la decisión de match en vez de reglas locales frágiles, lo que se traduce en umbrales de confianza más honestos para separar auto-match, auto-no-match y revisión manual; y tercero, la robustez ante stewards imperfectos es exactamente la propiedad que se necesita cuando el "gold standard" de entrenamiento proviene de varios revisores humanos con criterios ligeramente distintos, un escenario habitual en la curación de identidades de pacientes a escala poblacional.
