---
title: "Maximum Entropy Inverse RL (2008)"
weight: 371
math: true
---

{{< paper-card
    title="Maximum Entropy Inverse Reinforcement Learning"
    authors="Brian D. Ziebart, Andrew Maas, J. Andrew Bagnell, Anind K. Dey (CMU)"
    year="2008"
    venue="AAAI 2008"
    pdf="/papers/maxent-irl-ziebart-2008.pdf" >}}
El [aprendizaje reforzado inverso](/fundamentos/aprendizaje-reforzado-inverso) arrastra desde [Ng & Russell (2000)](/papers/irl-ng-russell-2000) una **ambigüedad estructural**: muchas funciones de recompensa —incluso el vector de todos ceros— hacen óptimo el mismo comportamiento demostrado, y muchas distribuciones sobre trayectorias satisfacen las mismas restricciones. Ziebart y colaboradores cierran ese hueco aplicando el **principio de máxima entropía** de Jaynes: entre todas las distribuciones consistentes con lo observado, elegir la que **no introduce ningún sesgo adicional**. El resultado es una **distribución de Boltzmann sobre trayectorias**, $P(\zeta) \propto e^{\theta^\top f_\zeta}$, que iguala las esperanzas de features observadas sin comprometerse a nada más. Motivado por un problema de gran escala —modelar las preferencias de ruta de taxistas a partir de más de **100.000 millas** de datos GPS en Pittsburgh— el paper entrega no solo un algoritmo de imitación, sino un modelo probabilístico completo que permite, vía Bayes, inferir destinos y rutas futuras. Es uno de los pilares del IRL moderno y la base directa de **GAIL** y **Deep MaxEnt IRL**.
{{< /paper-card >}}

---

## Contexto: la ambigüedad del IRL previo

La idea que estructura el área —y que la [Clase 33](/clases/clase-33) recoge— es plantear el espacio de políticas aprendidas como **soluciones de un MDP**: los agentes actúan para optimizar una recompensa desconocida (asumida **lineal en features**), y hay que encontrar los pesos que hacen que el comportamiento demostrado parezca (casi) óptimo. Formalmente, se observa la trayectoria $\zeta$ de un agente y la recompensa de una trayectoria es la suma de recompensas de estado, equivalente a aplicar los pesos al **conteo de features del camino**:

$$\text{reward}(f_\zeta) = \theta^\top f_\zeta = \sum_{s_j \in \zeta} \theta^\top f_{s_j}, \qquad f_\zeta = \sum_{s_j \in \zeta} f_{s_j}.$$

Recuperar los pesos exactos es un **problema mal planteado** (*ill-posed*). Los antecedentes lo atacaban por caminos distintos pero ninguno resolvía la ambigüedad:

- **Maximum Margin Planning** (Ratliff et al., 2006) plantea el IRL como predicción estructurada de máximo margen, pero **falla cuando ninguna recompensa hace el comportamiento demostrado simultáneamente óptimo y significativamente mejor** que las alternativas —algo frecuente cuando el experto es imperfecto.
- [Abbeel & Ng (2004)](/papers/apprenticeship-abbeel-ng-2004) proponen igualar las esperanzas de features entre la política observada y la del aprendiz, $\sum_{\zeta_i} P(\zeta_i)\, f_{\zeta_i} = \tilde f$, condición **necesaria y suficiente** para igualar el desempeño del experto. Pero cuando el comportamiento es subóptimo se requieren mezclas de políticas, y **muchas mezclas distintas satisfacen esa condición**.

El punto crítico: tanto el concepto de IRL como el matching de conteos de features son **ambiguos**, y ninguno de los métodos anteriores propone cómo resolver esa ambigüedad. Ahí entra la máxima entropía.

## Contribución: un marco probabilístico principiado

En lugar de razonar sobre políticas, Ziebart et al. consideran una distribución sobre **toda la clase de comportamientos posibles** (caminos de longitud variable). Entre todas las distribuciones que satisfacen el matching de features hay que elegir una, y el **principio de máxima entropía** (Jaynes, 1957) lo hace de forma canónica: elegir la distribución "menos comprometida" —la de máxima incertidumbre— consistente con lo observado. Cualquier otra estaría inyectando información que los datos no justifican.

El resultado formal, para MDPs deterministas, es una **distribución exponencial (de Boltzmann)** parametrizada por los pesos de recompensa:

$$P(\zeta_i \mid \theta) = \frac{1}{Z(\theta)}\, e^{\theta^\top f_{\zeta_i}} = \frac{1}{Z(\theta)}\, e^{\sum_{s_j \in \zeta_i} \theta^\top f_{s_j}}.$$

La interpretación es directa: **planes con recompensa equivalente tienen probabilidad equivalente** (justo lo que exige la máxima entropía) y **planes con mayor recompensa son exponencialmente preferidos**. Esto es lo que distingue a MaxEnt IRL de los modelos que **normalizan localmente** en cada estado: al normalizar **globalmente** sobre trayectorias completas mediante la función de partición $Z(\theta)$, el modelo evita el **sesgo de etiqueta** (*label bias*) heredado de los CRFs.

## Método: MLE, gradiente y forward-backward

### Máxima verosimilitud y gradiente

Maximizar la entropía sujeta a las restricciones de features es **equivalente a maximizar la verosimilitud** de los datos bajo la distribución exponencial —una dualidad clásica de Jaynes. El objetivo $\theta^* = \arg\max_\theta \sum \log P(\tilde\zeta \mid \theta, T)$ es **convexo** para MDPs deterministas, así que se resuelve con métodos de gradiente. El corazón del método es la forma del gradiente: la **diferencia entre los conteos de features empíricos y los esperados del aprendiz**, expresados mediante las **frecuencias esperadas de visita a estados** $D_{s_i}$:

$$\nabla L(\theta) = \tilde f - \sum_\zeta P(\zeta \mid \theta, T)\, f_\zeta = \tilde f - \sum_{s_i} D_{s_i}\, f_{s_i}.$$

Esta expresión es estructuralmente idéntica al gradiente de un modelo de familia exponencial estándar (un CRF): en el óptimo, cuando $\nabla L = 0$, **las esperanzas de features del modelo igualan las empíricas**. Eso garantiza —invocando el resultado de Abbeel & Ng (2004)— que el aprendiz iguala el desempeño del agente **sin importar cuáles fueran los pesos reales** que el experto optimizaba. El aprendizaje se reduce a "empujar" $\theta$ hasta que las visitas esperadas de estados generen el mismo perfil de features observado. Una ventaja adicional: la dependencia en el número de features $K$ es de solo $O(\log K)$, frente a la dependencia **lineal** de los métodos de margen y los localmente normalizados.

### El algoritmo forward-backward

El gradiente es fácil de computar una vez que se conocen las frecuencias de visita $D_{s_i}$. Enumerar todos los caminos es inviable (crecen exponencialmente con el horizonte), así que los autores usan un algoritmo eficiente análogo al **forward-backward de los CRFs** o a la **iteración de valor** del RL, aproximando el horizonte infinito con un horizonte fijo $N$. Tiene tres fases: un **paso hacia atrás** (*backward*) que calcula la masa de probabilidad desde cada estado terminal computando las particiones locales; el **cálculo de las probabilidades de acción locales** $P(a_{i,j}\mid s_i) = Z_{a_{i,j}}/Z_{s_i}$; y un **paso hacia adelante** (*forward*) que propaga la masa desde el estado inicial para obtener las frecuencias por timestep, que se suman en $D_{s_i} = \sum_t D_{s_i, t}$. La complejidad polinomial es lo que hace escalable a MaxEnt IRL sobre un MDP con cientos de miles de estados.

## Experimentos: rutas de taxistas en Pittsburgh

La red vial de Pittsburgh se modela como un **MDP determinista con más de 300.000 estados** (segmentos de carretera) y **900.000 acciones** (transiciones en intersecciones). Se asume que los conductores optimizan un compromiso entre tiempo, seguridad, estrés, combustible y mantenimiento —un **costo** (recompensa negativa)— y el destino es un estado absorbente. Se recolectaron trazas GPS de **25 taxis durante 12 semanas**: más de **100.000 millas** en más de **3.000 horas**, segmentadas en unos 13.000 viajes; tras descartar viajes cortos, cíclicos o ruidosos, quedó un conjunto de prueba de **7.403 ejemplos**. Las features de camino cubren tipo de carretera, velocidad, número de carriles y tipo de transición (22 conteos).

El **sesgo de etiqueta** se ilustra con un ejemplo de tres caminos de igual recompensa entre A y B: MaxEnt les da probabilidad igual (1/3 cada uno), mientras que un modelo basado en acción da 50% a uno y 25% a los otros dos por la estructura de ramificación —de modo que la política de mayor recompensa puede no ser la más probable. MaxEnt lo evita al normalizar globalmente.

En la comparación de predicción de rutas dado origen y destino, MaxEnt supera a todos los baselines con significancia estadística ($\alpha < 0.01$):

| Modelo | Matching | 90% Match | Log Prob |
|---|---|---|---|
| Time-based | 72.38% | 43.12% | N/A |
| Max Margin | 75.29% | 46.56% | N/A |
| Action | 77.30% | 50.37% | −7.91 |
| Action (costs) | 77.74% | 50.75% | N/A |
| **MaxEnt paths** | **78.79%** | **52.98%** | **−6.85** |

La mejor log-probabilidad (−6.85 vs −7.91) confirma que MaxEnt asigna densidad más fielmente, no solo predice mejor la ruta modal. Como es un modelo de densidad completo, además permite **inferir el destino** a partir de un camino parcial vía Bayes: $P(\text{dest} \mid \tilde\zeta_{A\to B}) \propto P(\tilde\zeta_{A\to B} \mid \text{dest})\, P(\text{dest})$, computable directamente con el forward-backward. La precisión posterior crece al observar una fracción mayor del camino (un tramo hacia el oeste descarta destinos orientales), habilitando avisos de tráfico, optimización de combustible en híbridos o climatización anticipada del hogar.

## Limitaciones

- **Requiere un MDP conocido y (efectivamente) finito.** El método asume la estructura del mundo (red vial, transiciones) conocida; en dominios con dinámica desconocida necesita un modelo del entorno.
- **Cómputo de la función de partición.** El forward-backward es polinomial pero itera sobre todo el espacio de estados durante $N$ iteraciones, y aproximar el horizonte infinito con $N$ fijo introduce error. En el caso de los taxis se restringe el cálculo a "una clase más pequeña de caminos razonablemente buenos".
- **MDPs no deterministas solo aproximados.** La tratabilidad para MDPs estocásticos depende de una suposición simplificadora (la aleatoriedad tiene efecto limitado y la partición es constante sobre resultados); el tratamiento exacto es intratable.
- **Recompensa lineal en features.** Como todo el linaje de Abbeel & Ng, la calidad depende de que las features diseñadas a mano capturen lo relevante —la limitación que Deep MaxEnt IRL levantaría después.

## Por qué importa para la Clase 33

MaxEnt IRL es el eslabón que profesionaliza el IRL, transformándolo de una idea elegante pero mal definida en una herramienta estadística robusta:

- **Cierra la ambigüedad de [Ng & Russell (2000)](/papers/irl-ng-russell-2000) y [Abbeel & Ng (2004)](/papers/apprenticeship-abbeel-ng-2004).** A la pregunta "¿cuál de las infinitas recompensas consistentes elegir?", la respuesta canónica es la de máxima entropía: la única que no inyecta sesgos que los datos no justifican.
- **Maneja demostradores subóptimos.** Mientras la clonación de comportamiento asume implícitamente un experto casi óptimo, MaxEnt modela explícitamente el ruido humano, dando probabilidad exponencialmente decreciente (pero no nula) a caminos peores —mucho más robusto ante demostraciones ruidosas.
- **Es el puente hacia el IRL profundo y GAIL.** Deep MaxEnt IRL (Wulfmeier et al., 2015) reemplaza la recompensa lineal por una red neuronal; Guided Cost Learning (Finn et al., 2016) lo extiende a dinámica desconocida; y Finn et al. junto con Ho & Ermon (**GAIL**, 2016) establecen la equivalencia formal entre el IRL de máxima entropía y las **GANs** —la recompensa aprendida juega el rol del discriminador y la política el del generador. La distribución de Boltzmann sobre trayectorias reaparece además como pieza central del **RL de máxima entropía** moderno (Soft Actor-Critic).

Para el fundamento de MDP, política, valor y recompensa, ver [/fundamentos/aprendizaje-reforzado](/fundamentos/aprendizaje-reforzado).
