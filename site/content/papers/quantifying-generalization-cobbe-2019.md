---
title: "Quantifying Generalization in RL: CoinRun (2019)"
weight: 377
math: true
---

{{< paper-card
    title="Quantifying Generalization in Reinforcement Learning"
    authors="Karl Cobbe, Oleg Klimov, Chris Hesse, Taehoon Kim, John Schulman (OpenAI)"
    year="2019"
    venue="ICML 2019"
    pdf="/papers/quantifying-generalization-cobbe-2019.pdf" >}}
El paper aborda de frente un problema que la comunidad de RL profundo reconocía pero no medía con rigor: el **sobreajuste**. En los benchmarks más populares —Atari, MuJoCo— es costumbre usar **exactamente los mismos ambientes para entrenar y evaluar**, práctica que dice muy poco sobre la capacidad real de *generalizar*. La imagen memorable de los autores: un agente que domina diez niveles de un videojuego "a menudo fracasa catastróficamente al encontrar por primera vez el undécimo". La contribución estrella es **CoinRun**, un ambiente de plataformas **generado proceduralmente**: como cada nivel se genera deterministamente desde una semilla, permite construir conjuntos de entrenamiento y prueba **disjuntos** de la misma distribución —el protocolo train/test del aprendizaje supervisado, trasladado a RL—. El hallazgo central: se requieren **miles** de niveles de entrenamiento para cerrar la brecha de generalización, muchos más que los usados por el trabajo previo. Es el experimento de las diapositivas 17-18 de la [Clase 33](/clases/clase-33) y la confirmación cuantitativa de la [generalización en RL](/fundamentos/generalizacion-en-rl) que [Zhang et al. (2018)](/papers/overfitting-rl-zhang-2018) había abierto.
{{< /paper-card >}}

---

## Contexto: de "entrenar en el test set" a medir generalización

Durante años, el RL profundo midió el progreso por el score alcanzado en un ambiente fijo: un agente se entrenaba en *Breakout* y se reportaba su score en *Breakout* —el mismo juego, la misma dinámica, la misma disposición de bloques—. Esta práctica, que los autores llaman "training on the test set", confunde dos capacidades distintas: **resolver** un problema concreto y **generalizar** a instancias nuevas del mismo tipo. Un agente puede memorizar la secuencia de acciones que resuelve un nivel sin haber aprendido nada transferible.

El trabajo se inscribe en una corriente que buscaba separar entrenamiento y evaluación: el **Sonic Benchmark** (Nichol et al., 2018), la inspiración más directa; Farebrother et al. (2018), que ya notaron que confundir train y test contribuyó a la ausencia de regularización en RL; Packer et al. (2018) sobre interpolación y extrapolación; y sobre todo [Zhang et al. (2018b)](/papers/overfitting-rl-zhang-2018), cuyos experimentos en laberintos gridworld ya habían cuantificado la relación entre sobreajuste y número de ambientes, mostrando que técnicas como *sticky actions* y *random starts* a menudo **no** mitigan el sobreajuste. CoinRun continúa esa tradición pero eleva la vara: lleva la medición a un **dominio procedural más rico** —un plataformas visual con obstáculos móviles y estáticos— y propone un **protocolo cuantitativo limpio** en el que la única variable manipulada es el número de niveles de entrenamiento.

## Método

**El ambiente CoinRun.** El objetivo de cada nivel es simple: recoger la única **moneda** al final. El agente aparece en el extremo izquierdo; la moneda está a la derecha; en medio hay obstáculos estacionarios y no estacionarios, y **una colisión provoca la muerte inmediata**. La única recompensa es una constante positiva fija al recoger la moneda. El episodio termina al morir, al recoger la moneda o tras 1000 timesteps. Cada nivel se genera deterministamente desde una semilla, con una **dificultad de 1 a 3** muestreada uniformemente, de modo que la distribución de niveles forma naturalmente un **currículo**. La observación es una imagen RGB de $64\times64\times3$.

**El protocolo de curvas de generalización.** Como train y test se extraen de la misma distribución, la brecha train–test cuantifica el sobreajuste. En test se mide el desempeño **zero-shot**: cero fine-tuning. Se entrenan **9 agentes**, cada uno sobre un conjunto de distinto tamaño —los primeros 8 sobre conjuntos de **100 a 16.000 niveles**, y el noveno sobre un conjunto **ilimitado** (semillas aleatorias de un espacio de $2^{32}$, ~2 millones de niveles únicos vistos)—. Todos se entrenan con **PPO** por **256M timesteps**, con $\gamma=0.999$, independientemente del tamaño del conjunto. La arquitectura base es la CNN de 3 capas de Mnih et al. (2015), la **Nature-CNN**, contrastada contra la **IMPALA-CNN** de bloques residuales (Espeholt et al., 2018) y una variante mayor, **IMPALA-Large** (5 bloques residuales, doble de canales).

## Resultados

Con la Nature-CNN, ocurre **sobreajuste sustancial con menos de 4.000 niveles**, y **con 16.000 niveles el sobreajuste sigue siendo apreciable**. Las cifras de la Tabla 1 (promedio sobre 5 seeds) hacen visceral el efecto:

| # Niveles | Train | Test | Brecha |
|---|---|---|---|
| 100 | $99.45\%$ | $66.79\%$ | $\approx 33$ pts |
| 500 | $97.85\%$ | $70.54\%$ | $\approx 27$ pts |
| 1000 | $95.7\%$ | $72.51\%$ | $\approx 23$ pts |
| 4000 | $90.18\%$ | $78.35\%$ | $\approx 12$ pts |
| 16000 | $89.24\%$ | $87.58\%$ | $\approx 1.7$ pts |
| $\infty$ | $90.87\%$ | $90.04\%$ | $\approx 0.8$ pts |

Con 100 niveles el agente resuelve casi el 100% de *sus* niveles pero apenas dos tercios de los nuevos; la brecha solo se cierra al llegar a los **miles**. Nótese un fenómeno propio del RL: el desempeño de **entrenamiento *baja*** al añadir niveles (de $99.45\%$ a $\approx 90\%$), porque con pocos niveles el agente memoriza, mientras que con muchos debe aprender habilidades genuinamente generalizables.

- **Arquitectura.** La IMPALA-CNN supera a la Nature-CNN en test en todos los conjuntos (con 500 niveles, $80.25\%$ vs $70.54\%$; con 2000, $90.92\%$ vs $75.6\%$) y es más eficiente en muestras. La IMPALA-Large rinde aún mejor, con **retornos decrecientes** al agrandar más la red.
- **Regularización clásica aplicada a RL.** Sobre un conjunto fijo de **500 niveles**: la **L2** ($w=10^{-4}$) y el **dropout** ($p=0.1$) reducen la brecha (dropout con impacto menor); la **data augmentation** —una versión de *Cutout*, emparentada con el *domain randomization* de robótica— da un impulso claro; la **batch normalization** da un impulso significativo. Combinarlas mejora *ligeramente*, lo que sugiere que atacan causas subyacentes similares.
- **Estocasticidad.** Inyectar aleatoriedad vía selección de acción $\varepsilon$-greedy o aumentando el **bonus de entropía** de PPO mejora la generalización —y **más que cualquiera de los métodos de regularización anteriores**—, aunque los autores esperan que el efecto varíe mucho entre ambientes.
- **Otros ambientes.** El mismo protocolo, con IMPALA-CNN + **LSTM**, se aplica a **CoinRun-Platforms** (varias monedas, exploración activa, sobreajuste hasta ~4000 niveles) y **RandomMazes** (laberintos de Kruskal parcialmente observados, con brecha apreciable **incluso con 20.000 niveles**). La magnitud del problema **escala con la complejidad del ambiente**.

## Limitaciones

La **simplicidad deliberada de CoinRun** es fortaleza metodológica pero limitación de alcance: las conclusiones podrían no transferirse tal cual a dominios más complejos. La **memoria y recurrencia** están poco exploradas (no está claro cuán adaptados están los LSTM a la tarea). La **combinación de regularizadores** —efecto pequeño al sumarlos, y el fracaso al mezclar estocasticidad con las demás "por razones desconocidas"— queda sin explicación mecanicista. Y el **costo computacional** es enorme: cientos de millones a miles de millones de timesteps (CoinRun-Platforms no convergió del todo ni con 2B, sino hacia 6B), lo que limita la reproducibilidad para grupos con menos cómputo.

## Por qué importa para la Clase 33

En la [Clase 33](/clases/clase-33), este paper respalda una tesis específica sobre [generalización en RL](/fundamentos/generalizacion-en-rl): **el RL sí generaliza, pero solo si se entrena variando las dimensiones relevantes del problema.** CoinRun es la demostración limpia: la "dimensión" que se varía es *el nivel*, y variarla equivale a entrenar sobre miles de niveles distintos en lugar de repetir unos pocos. La curva de generalización es la evidencia —con 100 niveles el agente memoriza y falla en test; con miles, aprende una política que transfiere zero-shot— y CoinRun-Platforms y RandomMazes confirman que la conclusión se sostiene y que el número de niveles requerido crece con la complejidad.

CoinRun es la continuación natural del programa que [Zhang et al. (2018)](/papers/overfitting-rl-zhang-2018) abre: *procedural generation* + train/test split como estándar de evaluación. Zhang mostró que los agentes memorizan (incluso ruido) y que la estocasticidad no siempre detecta el sobreajuste; Cobbe replica y extiende ese hallazgo a un dominio visual más rico y **añade la demostración positiva** de que la regularización supervisada sí ayuda. En conjunto convergen en el mensaje que la clase quiere dejar: **la generalización en RL no es gratis, se compra con diversidad de experiencia, y debe medirse con protocolos train/test honestos**, tal como en el aprendizaje supervisado.

Para un lector experto en FHIR y *master data management*, la analogía es precisa. Un matcher de pacientes —*record linkage* probabilístico, un GBM sobre features de similitud, o un bi-encoder de embeddings— es una política que debe generalizar a registros no vistos. La lectura de Cobbe et al.: la capacidad de generalización de un matcher **no escala con la cantidad de ejemplos de una sola fuente, sino con la diversidad de fuentes, formatos y patrones de error** representados en el entrenamiento —igual que CoinRun necesita miles de niveles distintos, no repeticiones de unos pocos—. Un modelo entrenado solo con registros de un hospital memorizará sus convenciones (RUT, abreviaturas, digitación) y "morirá catastróficamente" en el "nivel 11": una institución nueva. El remedio se traduce directo: entrenar sobre pares de muchas fuentes heterogéneas, aplicar *data augmentation* que simule ruido realista (typos, transposiciones, variantes de fecha) y, sobre todo, **evaluar siempre sobre un conjunto de fuentes disjunto del de entrenamiento**.
