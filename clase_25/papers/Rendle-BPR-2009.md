# BPR: Bayesian Personalized Ranking from Implicit Feedback

**Autores:** Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, Lars Schmidt-Thieme (Machine Learning Lab, University of Hildesheim)
**Venue:** UAI 2009 (Conference on Uncertainty in Artificial Intelligence), pp. 452-461.
**arXiv:** 1205.2618 (versión posterior, 2012).

---

## Contexto

### El escenario de recomendación de ítems

La tarea central que aborda el paper es la **recomendación de ítems** (*item recommendation*): producir, para cada usuario, un *ranking* personalizado sobre un conjunto de ítems (sitios web, películas, productos). No se trata de predecir una calificación numérica de un ítem aislado, sino de ordenar todos los ítems de modo que los más relevantes para ese usuario queden arriba. Esta es la formulación que importa en la práctica: una tienda online no muestra un número, muestra una lista ordenada.

### Feedback explícito vs. feedback implícito

Hasta ese momento, la mayor parte de la investigación en sistemas de recomendación se concentraba en **feedback explícito**: ratings que el usuario declara voluntariamente (1 a 5 estrellas). El paper hace notar que en escenarios reales la inmensa mayoría del feedback es **implícito**: clics, tiempos de visualización, compras, accesos registrados en logs de servidores web. El feedback implícito es mucho más fácil de recolectar porque el usuario no tiene que expresar su gusto explícitamente; de hecho ya está disponible en casi cualquier sistema de información.

La característica que define al feedback implícito, y que estructura todo el paper, es esta: **solo se observan ejemplos positivos**. Si un usuario compró un ítem, eso es una señal positiva. Pero los pares usuario-ítem *no observados* son una mezcla de:

1. **Feedback negativo real** (al usuario no le interesa el ítem), y
2. **Valores faltantes** (el usuario podría querer el ítem en el futuro, simplemente no lo ha visto todavía).

No hay forma de distinguir, a priori, cuál es cuál.

### El problema del enfoque ingenuo: optimizar puntajes absolutos

El enfoque usual en los recomendadores de ítems (incluyendo MF y kNN) es predecir un puntaje personalizado $\hat{x}_{ui}$ que refleje la preferencia del usuario $u$ por el ítem $i$, y luego ordenar los ítems por ese puntaje. Para entrenar estos modelos, la práctica estándar (Hu et al. 2008; Pan et al. 2008) consiste en:

- Asignar etiqueta **1** (positivo) a los pares $(u,i) \in S$ observados, y
- Asignar etiqueta **0** (negativo) a todo el resto $(U \times I) \setminus S$.

El paper identifica el defecto fundamental de esto: **todos los elementos que el modelo debería rankear en el futuro** $((U \times I) \setminus S)$ **se le presentan durante el entrenamiento como feedback negativo**. Un modelo con suficiente expresividad (capaz de ajustar exactamente los datos de entrenamiento) **no puede rankear en absoluto**, porque aprendería a predecir solo ceros para todo lo no observado. La única razón por la que estos métodos logran rankear algo es porque las estrategias contra el sobreajuste (regularización) les impiden alcanzar ese ajuste perfecto. En otras palabras: el ranking funciona *a pesar* del criterio de optimización, no *gracias* a él.

### Por qué optimizar pares y no puntajes absolutos

La propuesta del paper es cambiar la unidad de optimización: en vez de "scorear ítems individuales", usar **pares de ítems** como datos de entrenamiento y optimizar para **ordenar correctamente esos pares**. La intuición es directa: si un usuario $u$ vio (compró/clickeó) el ítem $i$ pero no el ítem $j$, asumimos que prefiere $i$ sobre $j$, es decir $i >_u j$. Esto reconstruye, a partir de $S$, partes del orden total latente $>_u$.

Esta reformulación tiene dos ventajas que el paper subraya:

1. Los datos de entrenamiento incluyen pares positivos, pares negativos y valores faltantes. Crucialmente, **los pares faltantes (entre dos ítems no observados) son exactamente los que deben rankearse en el futuro**. Desde un punto de vista de pares, el conjunto de entrenamiento $D_S$ y el de test son disjuntos.
2. Los datos se crean para el objetivo real (ranking), usando el subconjunto observado $D_S$ de $>_u$ como datos de entrenamiento.

---

## Contribución

El paper hace cuatro contribuciones explícitas:

1. **BPR-Opt**: un criterio de optimización genérico para ranking personalizado, derivado como el estimador *maximum a posteriori* (MAP) de un análisis bayesiano del problema. Se muestra su analogía con la maximización del área bajo la curva ROC (AUC).
2. **LearnBPR**: un algoritmo de aprendizaje genérico para maximizar BPR-Opt, basado en descenso de gradiente estocástico (SGD) con *bootstrap sampling* de tripletas de entrenamiento. Se muestra que es superior al SGD estándar para este problema.
3. La aplicación de LearnBPR a dos clases de modelos del estado del arte: factorización de matrices (MF) y kNN adaptativo.
4. Evidencia empírica de que, para ranking personalizado, entrenar con BPR supera a otros métodos de aprendizaje.

El mensaje de fondo, repetido en la conclusión, es que **la calidad de predicción no depende solo del modelo sino, en gran medida, del criterio de optimización**. Dos modelos idénticos (la misma MF) producen rankings muy distintos según con qué criterio se entrenen.

---

## Método

### Formalización del ranking personalizado

Sean $U$ el conjunto de usuarios e $I$ el de ítems. El feedback implícito es $S \subseteq U \times I$. El sistema debe entregar a cada usuario un **orden total** $>_u \subset I^2$ que cumpla totalidad, antisimetría y transitividad. Se definen:

$$I_u^+ := \{i \in I : (u,i) \in S\}, \qquad U_i^+ := \{u \in U : (u,i) \in S\}$$

Los datos de entrenamiento por pares se construyen como:

$$D_S := \{(u,i,j) \mid i \in I_u^+ \wedge j \in I \setminus I_u^+\}$$

La semántica de $(u,i,j) \in D_S$ es: el usuario $u$ prefiere $i$ sobre $j$. Como $>_u$ es antisimétrico, los casos negativos quedan considerados implícitamente.

### Derivación bayesiana del criterio (MAP)

La formulación bayesiana busca maximizar la probabilidad posterior del parámetro $\Theta$ del modelo (que puede ser cualquier clase de modelo, p. ej. MF):

$$p(\Theta \mid >_u) \propto p(>_u \mid \Theta)\, p(\Theta)$$

Se asume que los usuarios actúan independientemente entre sí, y que el orden de cada par $(i,j)$ para un usuario es independiente del orden de cualquier otro par. Bajo estos supuestos, la verosimilitud de todos los usuarios se factoriza. Combinando con la indicatriz $\delta$ y aprovechando totalidad y antisimetría, la verosimilitud se simplifica a:

$$\prod_{u \in U} p(>_u \mid \Theta) = \prod_{(u,i,j) \in D_S} p(i >_u j \mid \Theta)$$

La probabilidad individual de que un usuario prefiera $i$ sobre $j$ se modela con la **sigmoide logística** aplicada a un puntaje de diferencia:

$$p(i >_u j \mid \Theta) := \sigma(\hat{x}_{uij}(\Theta)), \qquad \sigma(x) := \frac{1}{1 + e^{-x}}$$

donde $\hat{x}_{uij}(\Theta)$ es una función real arbitraria de los parámetros que captura la relación entre el usuario $u$, el ítem $i$ y el ítem $j$. **Esta es la pieza genérica clave**: el marco BPR delega en el modelo subyacente (MF, kNN, lo que sea) la tarea de estimar $\hat{x}_{uij}$.

### El prior y BPR-Opt

Para completar el modelo bayesiano se introduce un prior gaussiano de media cero sobre los parámetros:

$$p(\Theta) \sim N(0, \Sigma_\Theta), \qquad \Sigma_\Theta = \lambda_\Theta I$$

Tomando el logaritmo del posterior (estimador MAP), se obtiene el criterio **BPR-Opt**:

$$\text{BPR-Opt} := \ln p(\Theta \mid >_u) = \ln p(>_u \mid \Theta)\, p(\Theta)$$
$$= \ln \prod_{(u,i,j) \in D_S} \sigma(\hat{x}_{uij})\, p(\Theta) = \sum_{(u,i,j) \in D_S} \ln \sigma(\hat{x}_{uij}) + \ln p(\Theta)$$
$$= \sum_{(u,i,j) \in D_S} \ln \sigma(\hat{x}_{uij}) - \lambda_\Theta \lVert \Theta \rVert^2$$

donde $\lambda_\Theta$ son parámetros de regularización específicos del modelo. El prior gaussiano se convierte, vía logaritmo, en el término de regularización L2. **Este es el objetivo canónico de ranking pairwise con feedback implícito.**

### Analogía con AUC

El AUC por usuario se define como:

$$\text{AUC}(u) := \frac{1}{|I_u^+|\, |I \setminus I_u^+|} \sum_{i \in I_u^+} \sum_{j \in I \setminus I_u^+} \delta(\hat{x}_{uij} > 0)$$

Reescrito con la notación $D_S$:

$$\text{AUC}(u) = \sum_{(u,i,j) \in D_S} z_u\, \delta(\hat{x}_{uij} > 0)$$

con $z_u$ una constante normalizadora. La analogía con BPR-Opt es directa: salvo la constante $z_u$, **solo difieren en la función de pérdida**. AUC usa la pérdida no diferenciable de Heaviside $\delta(x > 0) = H(x)$; BPR usa la pérdida diferenciable $\ln \sigma(x)$. Es práctica común reemplazar Heaviside por una función de forma similar (a menudo la propia $\sigma$) de manera heurística; la contribución teórica de BPR es que **la sustitución $\ln \sigma(x)$ no es heurística sino que está motivada por la estimación de máxima verosimilitud (MLE)**. La Figura 3 del paper compara Heaviside, hinge, sigmoide y $\ln$-sigmoide.

### LearnBPR: SGD con bootstrap sampling

El criterio es diferenciable, así que el descenso de gradiente es la opción natural. El gradiente de BPR-Opt es:

$$\frac{\partial\, \text{BPR-Opt}}{\partial \Theta} \propto \sum_{(u,i,j) \in D_S} \frac{-e^{-\hat{x}_{uij}}}{1 + e^{-\hat{x}_{uij}}} \cdot \frac{\partial}{\partial \Theta}\hat{x}_{uij} - \lambda_\Theta \Theta$$

Pero el SGD estándar **no funciona bien aquí**, por dos razones:

- **Gradiente completo (batch)**: hay $O(|S|\,|I|)$ tripletas en $D_S$, así que calcular el gradiente completo en cada paso es inviable. Además, la **asimetría (skewness)** de los pares causa mala convergencia: un ítem $i$ muy popular aparece en muchísimos términos $\hat{x}_{uij}$ (se compara contra todos los $j$ negativos para muchos usuarios), de modo que el gradiente de los parámetros que dependen de $i$ domina y obliga a tasas de aprendizaje muy pequeñas.
- **SGD recorriendo los datos por usuario o por ítem**: produce mala convergencia porque hay muchas actualizaciones consecutivas sobre el mismo par $(u,i)$ (un solo $(u,i)$ tiene muchos $j$ asociados).

La solución es **elegir las tripletas al azar (uniformemente) con reemplazo** — *bootstrap sampling*. Así la probabilidad de tomar la misma combinación usuario-ítem en pasos consecutivos es pequeña. El muestreo con reemplazo permite además detener el entrenamiento en cualquier paso, lo que es útil porque el número de ejemplos es enorme y a menudo basta una fracción de un ciclo completo para converger. El número de pasos se elige linealmente en función del feedback positivo observado $|S|$.

El algoritmo (Figura 4):

```
procedure LearnBPR(D_S, Θ)
  inicializar Θ
  repeat
    draw (u, i, j) from D_S        # bootstrap, uniforme con reemplazo
    Θ ← Θ + α ( (e^{-x̂_uij} / (1 + e^{-x̂_uij})) · ∂x̂_uij/∂Θ + λ_Θ · Θ )
  until convergencia
  return Θ̂
```

La Figura 5 muestra empíricamente que LearnBPR (sobre BPR-MF de 16 dimensiones, dataset Rossmann) converge mucho más rápido que el SGD por usuario.

### Aplicación a Matrix Factorization (BPR-MF)

Para usar BPR solo se necesita descomponer la estimación de la tripleta y conocer su gradiente. Se define:

$$\hat{x}_{uij} := \hat{x}_{ui} - \hat{x}_{uj}$$

es decir, **la diferencia de los dos puntajes individuales**. En MF, la matriz objetivo $X$ se aproxima por el producto de dos matrices de bajo rango $W: |U| \times k$ y $H: |I| \times k$, con $\hat{X} := W H^t$, de modo que:

$$\hat{x}_{ui} = \langle w_u, h_i \rangle = \sum_{f=1}^{k} w_{uf} \cdot h_{if}$$

Los parámetros son $\Theta = (W, H)$. Las derivadas necesarias para LearnBPR son:

$$\frac{\partial}{\partial \theta}\hat{x}_{uij} = \begin{cases} (h_{if} - h_{jf}) & \text{si } \theta = w_{uf} \\ w_{uf} & \text{si } \theta = h_{if} \\ -w_{uf} & \text{si } \theta = h_{jf} \\ 0 & \text{en otro caso} \end{cases}$$

Se usan tres constantes de regularización: $\lambda_W$ para las features de usuario, $\lambda_{H^+}$ para actualizaciones positivas sobre $h_{if}$ y $\lambda_{H^-}$ para actualizaciones negativas sobre $h_{jf}$.

### Aplicación a kNN adaptativo (BPR-kNN)

En kNN basado en ítems, la predicción depende de la similitud del ítem $i$ con los ítems que el usuario ya vio ($I_u^+$):

$$\hat{x}_{ui} = \sum_{l \in I_u^+ \wedge l \neq i} c_{il}$$

donde $C: I \times I$ es la matriz simétrica de similitud entre ítems, y $\Theta = C$. En lugar de fijar $C$ con una heurística (p. ej. similitud coseno $c_{i,j}^{\text{cosine}} = |U_i^+ \cap U_j^+| / \sqrt{|U_i^+| \cdot |U_j^+|}$), BPR-kNN **aprende $C$ directamente** optimizando BPR-Opt. Las derivadas son $+1$, $-1$ o $0$ según el parámetro, con constantes $\lambda_+$ y $\lambda_-$.

---

## Experimentos

### Datasets y metodología

Dos datasets de aplicaciones distintas:

- **Rossmann** (tienda online): historial de compras de 10.000 usuarios sobre 4.000 ítems, 426.612 compras.
- **Netflix** (alquiler de DVD): submuestra de 10.000 usuarios, 5.000 ítems, 565.738 acciones. Como Netflix es originalmente explícito (ratings 1-5), **se eliminaron los puntajes** para tratarlo como feedback implícito; la tarea pasa a ser predecir si un usuario calificará una película. La submuestra exige al menos 10 ítems por usuario y 10 usuarios por ítem.

**Esquema de evaluación:** *leave-one-out*. Para cada usuario se remueve aleatoriamente una acción (un par usuario-ítem) para el test, dejando train y test disjuntos. La métrica es el **AUC promedio**:

$$\text{AUC} = \frac{1}{|U|} \sum_u \frac{1}{|E(u)|} \sum_{(i,j) \in E(u)} \delta(\hat{x}_{ui} > \hat{x}_{uj})$$

con $E(u) := \{(i,j) \mid (u,i) \in S_{\text{test}} \wedge (u,j) \notin (S_{\text{test}} \cup S_{\text{train}})\}$. AUC de 0.5 = azar; 1 = perfecto. Todos los experimentos se repiten 10 veces con splits nuevos; hiperparámetros por grid search en la primera ronda, fijos en las 9 restantes.

### Resultados

Los métodos comparados: **BPR-MF**, **BPR-kNN** (las dos propuestas), contra **WR-MF** (Hu et al. / Pan et al.), **SVD-MF**, **Cosine-kNN**, el baseline **most-popular** ($\hat{x}_{ui}^{\text{most-pop}} := |U_i^+|$), y la cota teórica superior $np_{\max}$ para cualquier método **no personalizado**.

Hallazgos (Figura 6, dimensiones de 8 a 128):

1. **Los dos métodos BPR superan a todos los demás** en ambos datasets.
2. Comparando modelos idénticos: las tres MF (SVD-MF, WR-MF, BPR-MF) comparten exactamente el mismo modelo, pero su calidad difiere mucho — evidencia directa de que **el criterio importa**.
3. **SVD-MF** sobreajusta: su calidad *decrece* al aumentar dimensiones (mejor ajuste least-square en train, peor en test).
4. **WR-MF** es más robusto: gracias a la regularización su calidad sube de forma sostenida con más dimensiones.
5. **BPR-MF supera claramente a WR-MF**. Dato concreto reportado: en Netflix, una MF de **8 dimensiones** entrenada con BPR-MF logra calidad comparable a una MF de **128 dimensiones** entrenada con WR-MF.
6. Incluso métodos personalizados simples como Cosine-kNN superan ampliamente la cota $np_{\max}$ de cualquier método no personalizado. (Nota técnica del paper: el most-popular sobre test en Netflix da AUC 0.8794 vs. la cota superior estimada de 0.8801.)

---

## Limitaciones

- **Métrica única (AUC):** toda la evaluación se basa en AUC, que pesa por igual todas las posiciones del ranking. No se reportan métricas sensibles al tope de la lista (precision@k, NDCG, MAP), que en recomendación práctica suelen importar más, ya que el usuario solo ve las primeras posiciones. Esto fue señalado por trabajos posteriores.
- **Muestreo uniforme de negativos:** LearnBPR muestrea $j$ uniformemente entre los no observados. Esto trata todos los negativos por igual y converge lento cuando casi todos los negativos ya están bien rankeados; trabajos posteriores (p. ej. WARP, *dynamic/adaptive negative sampling*) propusieron muestreos que priorizan negativos "difíciles".
- **Supuesto de independencia entre pares:** la derivación bayesiana asume que el orden de cada par es independiente del de los demás, lo que no es estrictamente cierto dado que un orden total impone transitividad. Es una aproximación que funciona en la práctica pero no es exacta.
- **Solo dos clases de modelo y dos datasets:** la validación empírica es de 2009 (MF y kNN; Rossmann y Netflix submuestreado), sin modelos neuronales ni datasets de gran escala modernos.
- **Sin features de contenido:** el método es puramente colaborativo (IDs de usuario e ítem). No incorpora atributos de ítem/usuario ni multimodalidad — algo central en recsys moderno.
- **Sesgo de exposición no modelado:** asumir $i >_u j$ porque el usuario vio $i$ y no $j$ ignora que la no observación de $j$ puede deberse a que nunca se le mostró, no a desinterés.

---

## Impacto

BPR se convirtió en el **criterio de pérdida estándar de facto para recomendación con feedback implícito**. Sus contribuciones perduran:

- La **pérdida pairwise BPR** ($\ln \sigma(\hat{x}_{ui} - \hat{x}_{uj})$) es la función objetivo por defecto en innumerables sistemas de recomendación y la línea base obligatoria de comparación. Implementaciones de BPR-MF existen en prácticamente toda librería de recsys (LightFM, implicit, RecBole, Cornac, etc.).
- Es un **ancestro directo del aprendizaje de ranking pairwise y del triplet/contrastive learning** en deep learning. La estructura "ancla (usuario) + positivo + negativo" y la optimización de la diferencia de puntajes reaparece en *triplet loss*, en *negative sampling* de word2vec, en recomendadores neuronales (NCF, two-tower / dual-encoder) y en *contrastive learning* moderno.
- Estableció la lección metodológica de **"optimizar el modelo para el criterio correcto"**: separar la clase de modelo (lo que estima $\hat{x}$) del criterio de entrenamiento (cómo se compara). Esta separación genérica permite reutilizar BPR-Opt sobre cualquier modelo que produzca un puntaje real, incluidos los modelos profundos posteriores.
- El énfasis en **feedback implícito y datos one-class** anticipó la dirección dominante de la industria, donde los logs de interacción (no los ratings) son la materia prima.

---

## Conexión con la Clase 25

La Clase 25 es un *case study* de recomendación multimodal con feedback implícito y métricas de ranking, donde aparece el **metric learning** con **triplet loss**. BPR es el puente conceptual directo:

- **Misma estructura tripleta ancla-positivo-negativo.** En triplet loss se aprende un *embedding* tal que el ancla esté más cerca del positivo que del negativo por un margen. En BPR, el "ancla" es el usuario $u$, el positivo es el ítem visto $i$, el negativo es el ítem no visto $j$, y el objetivo es que el puntaje $\hat{x}_{ui}$ supere a $\hat{x}_{uj}$. Ambos optimizan una **relación de orden relativo entre un par (positivo, negativo) respecto de un ancla**, no un valor absoluto.

- **Ítems relevantes vs. no relevantes.** El dato fundamental de BPR — "$i$ es relevante para $u$, $j$ no lo es (o no lo sabemos)" — es exactamente el setup del feedback implícito de la clase: positivos observados, negativos muestreados. La construcción de $D_S$ por muestreo de negativos es el mismo procedimiento que alimenta una *triplet network*.

- **Diferencia clave: pérdida suave vs. margen duro.** BPR usa $\ln \sigma(\hat{x}_{ui} - \hat{x}_{uj})$ (pérdida logística suave, derivada del MLE), mientras que la triplet loss clásica usa $\max(0, m - (s_{\text{pos}} - s_{\text{neg}}))$ (hinge con margen $m$). El propio paper hace explícita esta cercanía al comparar BPR con MMMF, mostrando que la versión hinge de BPR-MF es $\sum \max(0, 1 - \langle w_u, h_i - h_j \rangle) + \text{reg}$ — literalmente una triplet/hinge loss sobre la diferencia de puntajes. BPR es entonces la **contraparte probabilística suave de la triplet loss**.

- **Métricas de ranking.** La clase usa métricas de ranking; BPR establece la conexión teórica entre la pérdida de entrenamiento ($\ln \sigma$) y la métrica de evaluación (AUC, vía la analogía con Heaviside), explicando *por qué* optimizar pares mejora el ranking medido.

- **De $\hat{x}_{ui} = \langle w_u, h_i \rangle$ al two-tower multimodal.** El puntaje de BPR-MF es un producto punto entre un embedding de usuario y uno de ítem. En la clase multimodal, esos embeddings se obtienen de torres neuronales (que pueden ingerir imagen, texto, etc.) en vez de matrices de lookup, pero la pérdida de ranking pairwise sobre el producto punto sigue siendo BPR. Es el mismo esqueleto, con encoders más ricos.

En síntesis: BPR-OPT define el **objetivo de optimización canónico** para ranking con feedback implícito, y la triplet loss de la Clase 25 es su realización en el lenguaje del metric learning profundo.
