# Quantifying Generalization in Reinforcement Learning (CoinRun) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Quantifying Generalization in Reinforcement Learning*.
- **Autores:** Karl Cobbe, Oleg Klimov, Chris Hesse, Taehoon Kim, John Schulman. Todos en **OpenAI** (San Francisco, CA).
- **Venue:** *Proceedings of the 36th International Conference on Machine Learning (ICML 2019)*, Long Beach, California. PMLR volumen 97.
- **Año:** 2019. **Preprint:** arXiv:1812.02341v3 (14 jul 2019), [arxiv.org/abs/1812.02341](https://arxiv.org/abs/1812.02341).
- **Linaje:** sale del grupo que también desarrolló PPO (Schulman et al., 2017) e IMPALA está en su árbol de referencias (Espeholt et al., 2018). Es la respuesta de OpenAI al problema de "entrenar y evaluar en el mismo conjunto de ambientes" que aquejaba a los benchmarks de RL de la época, y prolonga la línea del Sonic Benchmark (Nichol et al., 2018).

El paper aborda de frente un problema que la comunidad de RL profundo reconocía pero no medía con rigor: el **sobreajuste (overfitting)**. En los benchmarks más populares —Atari, MuJoCo— es costumbre usar **exactamente los mismos ambientes para entrenar y para evaluar**, práctica que ofrece muy poca información sobre la capacidad real de un agente para *generalizar*. Los autores lo ilustran con una imagen memorable: un agente que domina diez niveles de un videojuego "a menudo fracasa catastróficamente al encontrar por primera vez el undécimo". Los humanos generalizan sin esfuerzo entre tareas similares; los agentes de RL, no.

La contribución estrella es **CoinRun**, un ambiente de plataformas **generado proceduralmente** diseñado específicamente como *benchmark de generalización*. Al generar cada nivel de forma determinista a partir de una semilla, CoinRun permite construir **conjuntos de entrenamiento y de prueba disjuntos** extraídos de la misma distribución —exactamente el protocolo train/test del aprendizaje supervisado, trasladado a RL. Con esta herramienta, el paper cuantifica cuánto sobreajustan los agentes y arroja el hallazgo central: **se requieren MILES de niveles de entrenamiento para cerrar la brecha de generalización**, muchos más que los usados en trabajos previos de transferencia en RL. Además, muestra que arquitecturas convolucionales más profundas (IMPALA-CNN) y **técnicas de regularización clásicas del aprendizaje supervisado** —regularización L2, dropout, data augmentation, batch normalization e inyección de estocasticidad— reducen el sobreajuste también en RL.

Para la **Clase 33 (Aprendizaje por Imitación y Aprendizaje Reforzado Inverso)** este paper importa porque el profesor lo cita explícitamente (slides 17–18) como evidencia de que **el RL sí generaliza si se entrena variando las dimensiones del problema** —en CoinRun, la dimensión es "el nivel", y variarla significa entrenar sobre miles de niveles distintos. El experimento se replicó luego en un dominio más complejo, y esa idea —que la generalización de una política escala con la *diversidad* de la experiencia de entrenamiento, no con el tiempo de entrenamiento sobre pocos escenarios— es el puente conceptual con el resto de la clase.

## 2. Contexto: de "entrenar en el test set" a medir generalización

Durante años, el RL profundo midió el progreso por el score alcanzado en un ambiente fijo. Un agente de Atari se entrenaba en *Breakout* y se reportaba su score en *Breakout*: el mismo juego, la misma dinámica, la misma disposición de bloques. Esta práctica, que los autores llaman "training on the test set", confunde dos capacidades muy distintas: **resolver** un problema concreto y **generalizar** a instancias nuevas del mismo tipo de problema. Un agente puede memorizar la secuencia de acciones que resuelve un nivel sin haber aprendido nada transferible.

El trabajo se inscribe en una corriente creciente de esfuerzos por separar explícitamente entrenamiento y evaluación:

- **Sonic Benchmark** (Nichol et al., 2018), la inspiración más directa: mide generalización entrenando y evaluando en conjuntos disjuntos de niveles del videojuego *Sonic the Hedgehog*, permitiendo hasta 1 millón de timesteps de fine-tuning en test.
- **Farebrother et al. (2018):** reconocen que confundir train y test contribuyó a la *ausencia de regularización* en el RL profundo, y proponen usar distintos *game modes* de Atari 2600 para medir generalización. Ya encuentran que L2 y dropout ayudan a aprender features más generalizables.
- **Packer et al. (2018):** un benchmark sobre seis ambientes clásicos con parámetros internos expuestos, para cuantificar interpolación y extrapolación.
- **Zhang et al. (2018a):** miden sobreajuste en dominios continuos y observan que la generalización mejora al aumentar el número de *seeds* de entrenamiento; usan recompensas aleatorizadas para detectar memorización indeseada.
- **Justesen et al. (2018):** usan el framework GVG-AI para generar niveles proceduralmente y muestran que la capacidad de generalizar a niveles diseñados por humanos depende fuertemente de los *generadores* de niveles usados en entrenamiento.
- **Zhang et al. (2018b):** experimentan sobre laberintos gridworld generados proceduralmente y reportan que los agentes tienen alta capacidad de *memorizar* niveles específicos, y que técnicas pensadas para mitigar el sobreajuste (sticky actions, random starts) a menudo fallan.

CoinRun continúa esta tradición pero eleva la vara: lleva la medición de generalización a un **dominio procedural más rico** (un plataformas visual con obstáculos móviles y estáticos) y, sobre todo, propone un **protocolo cuantitativo limpio** en el que la única variable manipulada es el número de niveles de entrenamiento. El trabajo refleja a Zhang et al. (2018b) al cuantificar la relación entre sobreajuste y número de ambientes de entrenamiento, pero añade la demostración de que varios métodos —muchos importados del aprendizaje supervisado— reducen el sobreajuste en este benchmark.

## 3. Contribución central

El paper enuncia tres contribuciones explícitas:

1. **El número de ambientes de entrenamiento necesarios para una buena generalización es mucho mayor que el usado por el trabajo previo en transferencia en RL.** Este es el hallazgo cuantitativo de mayor impacto: no bastan decenas ni centenas de niveles; hacen falta miles.
2. **Una métrica de generalización basada en CoinRun**, que provee una señal útil sobre la cual iterar. Como cada nivel se genera deterministamente desde una semilla, existe un suministro *arbitrariamente grande y fácilmente cuantificable* de datos de entrenamiento, y la brecha train–test es medible con precisión.
3. **Una evaluación del impacto de distintas arquitecturas convolucionales y formas de regularización**, mostrando que estas decisiones pueden mejorar significativamente el desempeño de generalización.

La contribución metodológica de fondo es conceptual: **tratar los niveles generados proceduralmente como el análogo directo de los conjuntos train/test del aprendizaje supervisado**. En clasificación de imágenes, se entrena sobre un conjunto de imágenes y se evalúa sobre imágenes *distintas* de la misma distribución; la brecha entre error de entrenamiento y error de prueba mide el sobreajuste. CoinRun replica exactamente esa estructura: train y test se muestrean de la misma distribución de niveles, y **la brecha entre desempeño en train y en test determina la magnitud del sobreajuste**.

## 4. Método

### 4.1. El ambiente CoinRun

El objetivo de cada nivel de CoinRun es simple: **recoger la única moneda** situada al final. El agente controla un personaje que aparece en el extremo izquierdo; la moneda está en el extremo derecho. Entre ambos hay obstáculos estacionarios y no estacionarios; **una colisión con cualquiera provoca la muerte inmediata** del agente. La única recompensa del ambiente se obtiene al recoger la moneda, y es una **constante positiva fija**. El episodio termina cuando el agente muere, cuando recoge la moneda, o tras 1000 timesteps.

CoinRun fue diseñado para ser **tratable** para los algoritmos existentes: con suficientes niveles de entrenamiento y suficiente tiempo, los algoritmos aprenden una política casi óptima para *todos* los niveles. Cada nivel se genera **deterministamente desde una semilla**, lo que da acceso a un suministro esencialmente ilimitado de datos. El ambiente imita el estilo de plataformas como *Sonic*, pero es mucho más simple; para evaluar generalización, esa simplicidad es una ventaja.

Un detalle de diseño relevante: cada nivel tiene una **dificultad de 1 a 3**, muestreada uniformemente al generar el nivel. La dificultad condiciona el número de secciones, su largo y alto, y la frecuencia de obstáculos. Como los niveles varían ampliamente en dificultad, **la distribución de niveles forma naturalmente un *currículo*** para el agente. La observación es una imagen RGB de $64 \times 64 \times 3$ centrada en el agente. Como el agente necesita conocer su velocidad para actuar óptimamente, esta se codifica pintando dos pequeños cuadrados en la esquina superior izquierda (alternativamente, frame stacking o un modelo recurrente sirven al mismo fin, aunque con generalización algo menor).

### 4.2. Curvas de generalización: el protocolo

El corazón experimental es medir cuán bien un agente generaliza de un conjunto de niveles de entrenamiento a un conjunto de test no visto. Como ambos se extraen de la misma distribución, **la brecha train–test cuantifica el sobreajuste**. La expectativa es que, al crecer el número de niveles de entrenamiento disponibles, el desempeño en test mejore, *incluso cuando el agente se entrena por un número fijo de timesteps*. En test se mide el desempeño **zero-shot**: no se aplica ningún fine-tuning a los parámetros del agente.

Se entrenan **9 agentes**, cada uno sobre un conjunto de entrenamiento con distinto número de niveles. Durante el entrenamiento, cada nuevo episodio muestrea uniformemente un nivel del conjunto correspondiente. Los primeros 8 agentes se entrenan sobre conjuntos que van de **100 a 16.000 niveles**. El noveno se entrena sobre un conjunto **ilimitado**, donde cada nivel se siembra aleatoriamente: con $2^{32}$ semillas posibles las colisiones son improbables, y aunque este agente encuentra aproximadamente **2 millones de niveles únicos** durante el entrenamiento, no ve ningún nivel de test hasta el momento de evaluación. Todo el experimento se repite 5 veces, regenerando los conjuntos de entrenamiento cada vez.

Todos los agentes se entrenan con **Proximal Policy Optimization (PPO)** por un total de **256M timesteps** a través de 8 workers, con el mismo número de timesteps *independientemente* del tamaño del conjunto de entrenamiento. Se usa $\gamma = 0.999$, ya que un agente óptimo tarda entre 50 y 500 timesteps en resolver un nivel según su dificultad. La arquitectura base es la CNN de 3 capas de Mnih et al. (2015), a la que llaman **Nature-CNN**. Cada punto de las curvas se obtiene promediando el desempeño del agente final sobre **10.000 episodios**.

### 4.3. Arquitecturas comparadas

El paper contrasta la Nature-CNN con la arquitectura convolucional de **IMPALA** (Espeholt et al., 2018), la **IMPALA-CNN**, que usa bloques residuales. Además experimentan con una variante más profunda y ancha, **IMPALA-Large**, que usa **5 bloques residuales en lugar de 3, con el doble de canales en cada capa**. La comparación se hace tanto sobre el conjunto ilimitado (donde es imposible sobreajustar y el desempeño mide la capacidad de generalizar continuamente) como sobre un conjunto fijo de 500 niveles.

## 5. Resultados

### 5.1. Las curvas de generalización y el hallazgo de los "miles de niveles"

Con la Nature-CNN, los resultados (Figura 2a y Tabla 1) muestran que **ocurre sobreajuste sustancial cuando hay menos de 4.000 niveles de entrenamiento**, y que **incluso con 16.000 niveles el sobreajuste sigue siendo apreciable**. Los agentes rinden mejor cuando se entrenan sobre el conjunto ilimitado. Las cifras exactas de la Tabla 1 (Nature-CNN, promedio sobre 5 seeds) hacen visceral el efecto:

| # Niveles | Train | Test | Brecha |
|---|---|---|---|
| 100 | $99.45\%$ | $66.79\%$ | $\approx 33$ pts |
| 500 | $97.85\%$ | $70.54\%$ | $\approx 27$ pts |
| 1000 | $95.7\%$ | $72.51\%$ | $\approx 23$ pts |
| 4000 | $90.18\%$ | $78.35\%$ | $\approx 12$ pts |
| 16000 | $89.24\%$ | $87.58\%$ | $\approx 1.7$ pts |
| $\infty$ | $90.87\%$ | $90.04\%$ | $\approx 0.8$ pts |

La lectura es contundente: con 100 niveles el agente resuelve casi el 100% de *sus* niveles pero apenas dos tercios de los niveles nuevos. La brecha solo se cierra realmente al llegar a los **miles** de niveles. Nótese además un fenómeno propio del RL, ausente en el supervisado clásico: **el desempeño de entrenamiento *baja* al añadir niveles** (de $99.45\%$ a $\approx 90\%$), porque con pocos niveles el agente los memoriza casi perfectamente, mientras que con muchos debe aprender habilidades genuinamente generalizables.

### 5.2. Efecto de la arquitectura

La IMPALA-CNN (Figura 2b, Tabla 1) **supera a la Nature-CNN en test en todos los conjuntos de entrenamiento**. Por ejemplo, con 500 niveles IMPALA-CNN alcanza $80.25\%$ de test frente a $70.54\%$ de Nature-CNN (casi 10 puntos); con 2000 niveles, $90.92\%$ frente a $75.6\%$. Sobre el conjunto ilimitado, IMPALA-CNN es además **sustancialmente más eficiente en muestras** (Figura 3a). La IMPALA-Large rinde aún mejor, aunque los autores observan **retornos decrecientes** al aumentar el tamaño de red más allá de IMPALA-Large, sobre todo porque el tiempo de reloj de entrenamiento crece dramáticamente. La conclusión conecta con el aprendizaje supervisado: como allí, se espera que redes mayores tengan mayor capacidad de generalización.

Un matiz metodológico importante: los autores advierten que **aprender más rápido sobre el conjunto ilimitado no siempre correlaciona positivamente con mejor generalización**. Hiperparámetros bien elegidos pueden acelerar el entrenamiento sin mejorar la generalización. Por eso sostienen que **la métrica más útil es entrenar sobre un conjunto fijo** (usan 500 niveles) y medir directamente train vs. test.

### 5.3. Regularización clásica aplicada a RL

Aquí está la segunda gran tesis del paper. La regularización ha jugado siempre un rol central en el aprendizaje supervisado, donde la generalización es una preocupación inmediata; en cambio se emplea poco en RL profundo, presumiblemente porque **no ofrece beneficios perceptibles cuando train y test son el mismo conjunto**. Ahora que CoinRun mide la generalización directamente, hay razón para esperar que la regularización vuelva a ser efectiva. Todos estos experimentos se hacen sobre el **conjunto fijo de 500 niveles** con la IMPALA-CNN de 3 bloques.

- **Regularización L2 y dropout** (Figuras 4a y 4b). Se entrenan agentes con probabilidad de dropout $p \in [0, 0.25]$ o con penalización L2 $w \in [0, 2.5 \times 10^{-4}]$. Los agentes con L2 se entrenan 256M timesteps; los de dropout, 512M (convergen más lento). **Ambos reducen la brecha de generalización de forma apreciable, aunque dropout tiene un impacto menor.** Los valores empíricamente más efectivos son $p = 0.1$ y $w = 10^{-4}$.

- **Data augmentation** (Figura 4c). Usan una versión modificada de **Cutout** (Devries y Taylor, 2017): para cada observación, se enmascaran múltiples regiones rectangulares de tamaño variable, asignándoles un color aleatorio. Los autores notan que este método se asemeja mucho al *domain randomization* (Tobin et al., 2017) usado en robótica para transferir de simulación al mundo real. La augmentation aporta un impulso claro en CoinRun, y esperan que otras formas de augmentation sean similarmente efectivas, con eficacia variable según el ambiente.

- **Batch normalization** (Figura 4c). Aumentan la IMPALA-CNN con batch norm tras cada capa convolucional. Los workers de entrenamiento normalizan con las estadísticas del batch actual; los de test, con un promedio móvil de esas estadísticas. Batch norm ofrece **un impulso de desempeño significativo**.

- **Estocasticidad** (Figura 5). Evalúan dos vías de inyectar aleatoriedad. Primero, **estocasticidad del ambiente** vía selección de acción $\varepsilon$-greedy: con probabilidad $\varepsilon$ en cada paso, se reemplaza la acción preferida del agente por una aleatoria. Segundo, **estocasticidad de la política** aumentando el **bonus de entropía** de PPO (el baseline ya usa $k_H = 0.01$). Como el entrenamiento se hace más lento, se extiende a 512M timesteps. El resultado es notable: **un aumento de la estocasticidad del ambiente o de la política mejora la generalización, y cada método por separado ofrece un impulso similar**. Aún más llamativo: **entrenar con mayor estocasticidad mejora la generalización más que cualquiera de los métodos de regularización anteriores.** Los autores esperan que este efecto varíe mucho entre ambientes —menor en ambientes cuya dinámica ya es muy estocástica.

- **Combinación de métodos** (Figura 4c). Combinar data augmentation, batch norm y L2 produce un desempeño de test *ligeramente* mejor que usar cualquiera por separado. La pequeña magnitud del efecto sugiere que estos métodos **atacan causas subyacentes similares** de la mala generalización. Por razones no aclaradas, tuvieron poco éxito combinando $\varepsilon$-greedy y bonus de entropía alto con las otras formas de regularización.

### 5.4. Ambientes adicionales: CoinRun-Platforms y RandomMazes

Para corroborar que el sobreajuste no es un artefacto de un solo ambiente, aplican el mismo protocolo a dos ambientes más, ambos con IMPALA-CNN seguida de un **LSTM** (la memoria es necesaria para explorar óptimamente):

- **CoinRun-Platforms:** varias monedas dispersas en plataformas que el agente debe explorar activamente dentro de 1000 pasos, a veces retrocediendo. Es mucho más difícil, por lo que se entrena **2B timesteps**. La Figura 7 muestra sobreajuste hasta cerca de **4000 niveles** de entrenamiento. Aquí aparece un fenómeno inverso al supervisado: **el desempeño de *entrenamiento* aumenta con el número de niveles** más allá de cierto umbral, atribuido al currículo implícito de la distribución de niveles generados —con más datos, el agente aprende habilidades que generalizan incluso entre niveles de entrenamiento.

- **RandomMazes:** laberintos cuadrados de dimensión muestreada uniformemente de 3 a 25, generados con el algoritmo de Kruskal, parcialmente observados (el agente ve un parche de $9 \times 9$ celdas a su alrededor). La Figura 8 revela **sobreajuste particularmente fuerte, con una brecha sizeable incluso entrenando sobre 20.000 niveles** (la Tabla 3 muestra la brecha cerrándose recién hacia los 64.000–128.000 niveles).

Estos resultados refuerzan cuán susceptibles al sobreajuste son los algoritmos, y muestran que la magnitud del problema *escala con la complejidad del ambiente*: cuanto más rico el dominio, más niveles se necesitan para generalizar.

## 6. Limitaciones

- **Simplicidad deliberada de CoinRun.** El ambiente fue diseñado para ser tratable, lo que es una fortaleza metodológica pero también una limitación de alcance: las conclusiones podrían no transferirse tal cual a dominios mucho más complejos. Los autores lo reconocen al esperar que "las lecciones aprendidas apliquen en contextos más complejos".
- **Memoria y recurrencia poco exploradas.** En CoinRun-Platforms y RandomMazes el agente debe usar recurrencia y memoria, y no está claro cuán bien adaptados están los LSTM a esta tarea. Los autores dejan explícitamente para trabajo futuro investigar si otras arquitecturas recurrentes generalizarían mejor.
- **Combinación de regularizadores poco entendida.** El efecto pequeño de combinar métodos y el fracaso al mezclar estocasticidad con otras regularizaciones ("por razones desconocidas") quedan sin explicación mecanicista.
- **Costo computacional.** Los experimentos requieren cientos de millones a miles de millones de timesteps; el propio paper nota que CoinRun-Platforms no convergió del todo ni con 2B timesteps (convergió hacia 6B). Esto limita la reproducibilidad para grupos con menos cómputo.

## 7. Conexión con la Clase 33 (Aprendizaje por Imitación y Aprendizaje Reforzado Inverso)

El profesor Rodrigo Toro Icarte cita este paper (slides 17–18) para respaldar una tesis específica: **el RL sí generaliza, pero solo si se entrena variando las dimensiones relevantes del problema**. CoinRun es la demostración limpia de esa afirmación. La "dimensión" que se varía es *el nivel*; variarla equivale a entrenar sobre miles de niveles distintos en lugar de repetir unos pocos. La curva de generalización (Sección 5.1) es la evidencia: con 100 niveles el agente memoriza y falla en test; con miles, aprende una política que transfiere zero-shot. El profesor menciona además que el experimento se replicó en un dominio más complejo —dentro del propio paper, CoinRun-Platforms y RandomMazes cumplen ese rol, mostrando que la conclusión se sostiene y que el número de niveles requerido crece con la complejidad.

¿Por qué importa esto en una clase de imitación y RL inverso? Porque el aprendizaje por imitación y el IRL son técnicas para *obtener buenas políticas cuando la recompensa es difícil de especificar o los datos de experto son escasos*, y su valor último se mide por cuánto **generalizan** a situaciones no vistas —no por cuán bien reproducen las trayectorias de entrenamiento. CoinRun aporta el marco conceptual y la advertencia cuantitativa: una política que parece excelente sobre su conjunto de entrenamiento puede estar simplemente memorizando. La lección "necesitas diversidad, no solo más entrenamiento sobre lo mismo" aplica directamente a la imitación (necesitas demostraciones de expertos en *muchas* situaciones distintas) y al IRL (la recompensa inferida debe validarse en escenarios nuevos).

**Relación con Zhang y con Witty.** El paper dialoga explícitamente con **Zhang et al. (2018b)**, cuyos experimentos en laberintos gridworld ya habían cuantificado la relación entre sobreajuste y número de ambientes de entrenamiento, y habían mostrado que técnicas como sticky actions y random starts a menudo *no* mitigan el sobreajuste. CoinRun replica y extiende ese hallazgo a un dominio visual más rico, y añade la demostración positiva de que la regularización supervisada *sí* ayuda. La línea de **Witty et al.** (trabajo posterior sobre medir y entender la generalización de agentes de RL, típicamente citado junto a CoinRun en la clase) continúa esta agenda: caracterizar *dónde* y *por qué* fallan los agentes fuera de su distribución de entrenamiento. En conjunto, estos trabajos convergen en el mensaje que la Clase 33 quiere dejar: la generalización en RL no es gratis, se compra con diversidad de experiencia, y debe medirse con protocolos train/test honestos, tal como en el aprendizaje supervisado.

## 8. Nota final: conexión con salud, MDM y record linkage

Para un lector experto en FHIR y *master data management*, CoinRun ofrece una analogía sorprendentemente precisa. Un matcher de pacientes —sea un modelo de *record linkage* probabilístico, un GBM sobre features de similitud, o un bi-encoder de embeddings— es, en el fondo, una política que debe generalizar a registros que no vio en entrenamiento. El hallazgo de Cobbe et al. tiene una traducción directa: **la capacidad de generalización de un matcher no escala con la cantidad de ejemplos de una sola fuente, sino con la *diversidad* de fuentes, formatos y patrones de error representados en el entrenamiento**, igual que CoinRun necesita miles de niveles —y no muchas repeticiones de unos pocos— para cerrar la brecha train/test. Un modelo entrenado solo con registros de un hospital (una "distribución de niveles" estrecha) memorizará las convenciones de ese sitio: su formato de RUT, sus abreviaturas de nombres, sus patrones de digitación, sus valores nulos característicos; y "morirá catastróficamente" al enfrentar el "nivel 11" —una nueva institución con otra codificación de direcciones o transliteraciones distintas. La receta de CoinRun sugiere el remedio: entrenar sobre pares de muchas fuentes heterogéneas (el análogo de los miles de niveles), aplicar *data augmentation* que simule ruido realista (typos, transposiciones de campos, variantes de fecha —el equivalente al Cutout y al domain randomization del paper), y sobre todo **evaluar siempre sobre un conjunto de fuentes disjunto del de entrenamiento**, nunca sobre los mismos registros con que se ajustó el umbral. La brecha entre la precisión reportada en validación interna y la observada al conectar una institución nueva es, literalmente, la brecha de generalización de CoinRun manifestándose en un pipeline de interoperabilidad clínica.
