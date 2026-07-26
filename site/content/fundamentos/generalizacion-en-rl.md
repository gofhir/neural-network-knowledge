---
title: "Generalización en Aprendizaje Reforzado"
weight: 112
math: true
---

La **generalización** es la capacidad de un modelo de funcionar bien en situaciones que **no vio durante el entrenamiento**. En [aprendizaje supervisado](/fundamentos/aprendizaje-reforzado) es un concepto central y celosamente vigilado: separamos train y test, medimos la brecha entre ambos y tenemos una regla de oro —*nunca uses tus datos de test para entrenar*. Sorprendentemente, durante años el [aprendizaje reforzado](/fundamentos/aprendizaje-reforzado) profundo ignoró esta disciplina: era práctica común **entrenar y evaluar al agente en exactamente el mismo ambiente**. Este fundamento acompaña a la [Clase 33](/clases/clase-33), que abre preguntando algo incómodo: *¿el aprendizaje reforzado realmente generaliza?*

---

## 1. El pecado original: entrenar y testear en el mismo dominio

Recordemos la disciplina del aprendizaje supervisado. Entrenamos un modelo (fase lenta), lo evaluamos en datos separados (fase rápida) y cuidamos que no haya **overfitting**: que el modelo no memorice el conjunto de entrenamiento en vez de aprender un patrón que transfiera. La regla es inviolable: los datos de test jamás tocan el entrenamiento.

En RL profundo, en cambio, la práctica habitual durante años fue reportar el desempeño del agente **en el mismo ambiente donde fue entrenado**. Un agente de Atari se entrenaba en Breakout y se reportaba su puntaje en Breakout —el mismo juego, la misma semilla, el mismo layout. Como señala la clase, *"estamos entrenando y testeando en el mismo dominio"*. Esto oculta por completo la pregunta de la generalización: no sabemos si el agente **aprendió a jugar** o simplemente **memorizó una secuencia de acciones** que funciona en esa instancia particular.

{{< concept-alert type="advertencia" >}}
El problema metodológico es sutil pero grave: un puntaje altísimo en el ambiente de entrenamiento **no dice nada** sobre si el agente generaliza. Podría estar memorizando. Para medir generalización en RL hay que hacer lo mismo que en aprendizaje supervisado: **separar las instancias de entrenamiento de las de evaluación**.
{{< /concept-alert >}}

---

## 2. Medir el overfitting: el estudio de Zhang et al.

Zhang et al. (2018), en *A Study on Overfitting in Deep Reinforcement Learning*, hicieron el experimento que la clase describe en detalle. Diseñaron un **gridworld** formalizado como un MDP $M = \langle S, A, r, p, \mu\rangle$ con una función de recompensa concreta:

- **diamante:** $+1$
- **rayo:** $-1$
- **bomba:** $-1$
- **llave:** $+0.1$ (y termina el episodio)

La pieza clave es $\mu$: un **generador de mapas iniciales**. Con $\mu$ produjeron un gran conjunto $\bar S_0$ de configuraciones distintas del dominio —variando las posiciones del agente, las murallas y los objetos— y lo **dividieron en un conjunto de train y uno de test**, exactamente como en aprendizaje supervisado. El agente se entrenaba sobre las configuraciones de train y se evaluaba sobre las de test, que nunca había visto.

El hallazgo fue contundente: los agentes **memorizaban**. Alcanzaban desempeño casi perfecto en las configuraciones de entrenamiento pero fallaban en las de test, revelando una **brecha de generalización** grande. Incluso demostraron memorización de puro ruido —el agente podía "aprender" configuraciones cuya recompensa se había invertido al azar, prueba de que estaba memorizando, no razonando. Y mostraron que técnicas que se creían útiles (acciones pegajosas, inicios aleatorios, políticas estocásticas) resultaban **insuficientes** tanto para regularizar como para detectar el overfitting.

---

## 3. Cuantificar la brecha: CoinRun y Cobbe et al.

Cobbe et al. (2019), en *Quantifying Generalization in Reinforcement Learning*, replicaron y extendieron el experimento en un dominio más rico: **CoinRun**, un juego de plataformas cuyos niveles se **generan proceduralmente**. Cada nivel generado es análogo a un ejemplo de un dataset supervisado: se puede construir un conjunto de train con $N$ niveles y evaluar en niveles de test nunca vistos, midiendo la brecha de generalización **en función de $N$**.

El resultado central es una curva reveladora: **se necesitan miles de niveles de entrenamiento para cerrar la brecha**. Con pocos niveles el agente memoriza y falla en test; a medida que crece la diversidad de entrenamiento, la brecha se reduce. Además, Cobbe et al. probaron que las técnicas de **regularización clásicas del deep learning** —hasta entonces ignoradas en RL— **sí ayudan** a la generalización en RL:

- **Dropout** y **regularización $L_2$**
- **Data augmentation** (perturbaciones de la observación)
- **Batch normalization**
- **Estocasticidad** (mayor entropía de la política, $\varepsilon$-greedy)
- **Arquitecturas más grandes** (la red IMPALA-CNN generaliza mejor que una CNN pequeña)

---

## 4. Caracterizar el fallo: Witty et al.

Witty et al. (2018), en *Measuring and Characterizing Generalization in Deep RL*, aportaron la dimensión cualitativa. No basta con **medir** cuánto generaliza un agente (una métrica agregada); hay que **caracterizar dónde y cómo falla**. Evaluaron agentes aparentemente competentes en estados **fuera de su distribución de entrenamiento** pero perfectamente legítimos, distinguiendo generalización por **interpolación** (estados alcanzables bajo la política) de la más exigente **extrapolación** (estados novedosos que el agente nunca alcanzaría por sí mismo). El hallazgo: agentes con excelente desempeño de entrenamiento **fallan de formas específicas e interpretables** en estados nuevos, lo que exige un análisis de modos de fallo, no solo un número global.

---

## 5. La respuesta de la Clase 33

Con esta evidencia, la Clase 33 responde su propia pregunta:

> **¿Podemos aprender políticas que generalicen usando aprendizaje reforzado?**
> Sí, **pero debemos entrenar en ambientes que varíen en las dimensiones en que queremos generalizar.**

Es la misma lógica del aprendizaje supervisado, llevada al RL: si quieres que el agente generalice a nuevos layouts, entrena con muchos layouts distintos; si quieres que generalice a nuevas velocidades de enemigos, varía las velocidades en entrenamiento. La generalización no aparece gratis —emerge de la **diversidad del entrenamiento**. Un agente entrenado en una sola instancia memorizará esa instancia; uno entrenado sobre miles de variaciones aprenderá el patrón subyacente. Esta conclusión conecta directamente con la sección de la clase sobre [aprendizaje por imitación](/fundamentos/aprendizaje-por-imitacion): en ambos casos, la clave para que la política se comporte bien fuera de su experiencia directa es exponerla, durante el entrenamiento, a la variabilidad que enfrentará después.

---

## 6. Relevancia para MDM y record linkage

La trampa que denuncia esta línea de trabajo tiene un paralelo exacto en el **matching de pacientes**. Un modelo de deduplicación entrenado y evaluado sobre registros del **mismo hospital, el mismo sistema origen, el mismo período** exhibirá métricas excelentes que **no dicen nada** sobre su desempeño real cuando llegue a producción y enfrente registros de otra institución, con otras convenciones de nombres, otros formatos de fecha, otra prevalencia de datos faltantes. Es el equivalente clínico de entrenar y testear en el mismo dominio. La lección de Zhang, Cobbe y Witty se traduce sin cambios: hay que **evaluar con splits por sitio/fuente/período**, hay que **entrenar sobre la diversidad de fuentes** que se espera encontrar (como CoinRun necesita miles de niveles), y hay que **caracterizar los modos de fallo** —en qué tipos de registro el matcher se equivoca— y no conformarse con un F1 global, porque en salud un modo de fallo silencioso puede tener consecuencias graves.

---

## Referencias

- Zhang, C., Vinyals, O., Munos, R. & Bengio, S. (2018). *A Study on Overfitting in Deep Reinforcement Learning*. arXiv:1804.06893.
- Cobbe, K., Klimov, O., Hesse, C., Kim, T. & Schulman, J. (2019). *Quantifying Generalization in Reinforcement Learning* (CoinRun). ICML.
- Witty, S., Lee, J.K., Tosch, E., Atrey, A., Littman, M. & Jensen, D. (2018). *Measuring and Characterizing Generalization in Deep Reinforcement Learning*. arXiv:1812.02868.
