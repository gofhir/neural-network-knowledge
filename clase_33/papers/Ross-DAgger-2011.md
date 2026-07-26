# A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*.
- **Autores:** Stéphane Ross, Geoffrey J. Gordon, J. Andrew Bagnell (los tres de **Carnegie Mellon University** — Robotics Institute y Machine Learning Department).
- **Venue:** *Proceedings of the 14th International Conference on Artificial Intelligence and Statistics* (**AISTATS 2011**), Fort Lauderdale, FL. Volumen 15 de JMLR: W&CP.
- **Año:** 2011. **Preprint:** arXiv:1011.0686v3 (16 mar 2011).
- **Una línea:** propone **DAgger** (*Dataset Aggregation*), un meta-algoritmo iterativo de aprendizaje por imitación que entrena una política **estacionaria y determinista** con garantías de desempeño **lineales** en el horizonte $T$, tratando la imitación como una reducción a **aprendizaje online sin arrepentimiento** (*no-regret*).

Este es, junto con el trabajo de Ross y Bagnell (2010), el paper que diagnostica y **cura** el problema estructural del aprendizaje por imitación ingenuo (el *behavioral cloning* o clonación de comportamiento): cuando se entrena un clasificador para imitar a un experto usando solo demostraciones del experto, las predicciones del aprendiz alteran las observaciones futuras, violando la suposición i.i.d. que sostiene el aprendizaje supervisado clásico. El resultado es que **los errores se componen**: una política con tasa de error $\epsilon$ bajo la distribución de estados del experto puede cometer hasta $O(T^2\epsilon)$ errores bajo la distribución de estados que ella misma induce. DAgger reduce esa cota a $O(T\epsilon)$ —lineal en el horizonte— y lo hace con un algoritmo notablemente simple: no tiene parámetros libres más allá del aprendiz supervisado que se use como subrutina.

Para la **Clase 33 (Aprendizaje por Imitación y Aprendizaje Reforzado Inverso**, prof. Rodrigo Toro Icarte), DAgger es exactamente el eje teórico del **problema** que la clase señala en la clonación de comportamiento (slide 36: *"DAgger: dataset aggregation"*) y es el algoritmo que el **laboratorio** implementa sobre Atari Breakout con un experto DQN. Entender por qué el *behavioral cloning* falla —y cómo DAgger lo arregla consultando al experto sobre los estados que el aprendiz **efectivamente visita**— es la columna vertebral de toda la clase.

## 2. Contexto: por qué falla el behavioral cloning

### 2.1. La suposición i.i.d. rota

El enfoque tradicional de la imitación es reducirla a un problema de aprendizaje supervisado: se recolecta un conjunto de pares (observación, acción experta) mientras el experto conduce el sistema, y se entrena un clasificador o regresor que prediga la acción del experto dada la observación. Formalmente, sea $\Pi$ la clase de políticas y $T$ el horizonte de la tarea. Para una política $\pi$, denotamos $d_\pi^t$ la distribución de estados en el paso $t$ si el aprendiz ejecuta $\pi$ desde el paso 1 hasta $t-1$, y

$$d_\pi = \frac{1}{T}\sum_{t=1}^{T} d_\pi^t$$

la **distribución promedio de estados** al seguir $\pi$ durante $T$ pasos. Con $C(s,a)$ el costo inmediato esperado de la acción $a$ en el estado $s$ (acotado en $[0,1]$) y $C_\pi(s) = \mathbb{E}_{a\sim\pi(s)}[C(s,a)]$, el costo total de ejecutar $\pi$ es

$$J(\pi) = \sum_{t=1}^{T}\mathbb{E}_{s\sim d_\pi^t}[C_\pi(s)] = T\,\mathbb{E}_{s\sim d_\pi}[C_\pi(s)].$$

En imitación no siempre observamos $C$; observamos demostraciones del experto $\pi^*$ y minimizamos una **pérdida sustituta** (*surrogate*) $\ell(s,\pi)$ —por ejemplo la pérdida 0-1 del aprendiz respecto de $\pi^*$ en el estado $s$, o una pérdida cuadrática o *hinge*. El objetivo *deseable* es encontrar

$$\hat\pi = \arg\min_{\pi\in\Pi}\mathbb{E}_{s\sim d_\pi}[\ell(s,\pi)],$$

es decir, minimizar la pérdida **bajo la propia distribución inducida** por la política. El problema es que $d_\pi$ depende de $\pi$ y no se puede calcular; solo se puede muestrear ejecutando $\pi$. Esta dependencia entre la política y la distribución de entradas es lo que rompe la suposición i.i.d. y vuelve el objetivo **no convexo** incluso cuando $\ell(s,\cdot)$ es convexa para cada estado.

El *behavioral cloning* ignora este acoplamiento y minimiza la pérdida bajo la distribución del **experto**, $d_{\pi^*}$:

$$\hat\pi_{sup} = \arg\min_{\pi\in\Pi}\mathbb{E}_{s\sim d_{\pi^*}}[\ell(s,\pi)].$$

Esto sí es un problema supervisado estándar (los estados provienen de una distribución fija, la del experto), pero optimiza la distribución equivocada.

### 2.2. Los dos motivos del fallo: compounding error y estados fuera de distribución

El paper identifica —y la Clase 33 subraya— **dos causas entrelazadas** del fracaso:

1. **Compounding error (error que se compone).** En cuanto el aprendiz comete un error, se desvía de la trayectoria que habría seguido el experto y **encuentra observaciones completamente distintas** a las de la demostración. Cada nuevo error empuja el sistema a regiones aún más ajenas, y los errores se acumulan multiplicativamente a lo largo del horizonte.

2. **Estados fuera de distribución (*distribution shift*).** El aprendiz jamás vio en entrenamiento los estados que visita tras equivocarse —justamente porque el experto, siendo bueno, nunca los visita— así que no aprendió a **recuperarse** de sus propios errores. El clasificador puede tener baja pérdida sobre $d_{\pi^*}$ y aun así ser desastroso sobre $d_{\hat\pi}$.

La cuantificación formal es el **Teorema 2.1** (Ross y Bagnell, 2010): si $\mathbb{E}_{s\sim d_{\pi^*}}[\ell(s,\pi)] = \epsilon$, con $\ell$ una cota superior de la pérdida 0-1, entonces

$$J(\pi) \le J(\pi^*) + T^2\epsilon.$$

La cota es **cuadrática en el horizonte**. Y no es un artefacto de una demostración floja: **es ajustada** (*tight*). El paper cita el ejemplo de Kääriäinen (2006) en predicción de secuencias, donde una tasa de error $\epsilon>0$ al predecir el siguiente símbolo (usando el símbolo correcto previo como entrada) produce en test un número esperado de errores de

$$\frac{T}{2} + \frac{T+1}{2} - \frac{1-(1-2\epsilon)^{T+1}}{4\epsilon},$$

que está acotado por $T^2\epsilon$ y se comporta como $\Theta(T^2)$ para $\epsilon$ pequeño. Ross y Bagnell (2010) construyeron además un ejemplo de imitación donde exactamente $J(\hat\pi_{sup}) = (1-T)J(\pi^*) + T^2\epsilon$.

**La intuición del crecimiento cuadrático.** Hay $T$ pasos en los que el aprendiz puede cometer un error nuevo; cada error, además del costo inmediato, cambia la distribución de estados de los pasos siguientes hacia regiones no cubiertas por el entrenamiento, donde la probabilidad de error ya no está controlada por $\epsilon$. Con costo por paso acotado por 1 y hasta $T$ pasos de "cola" afectados por cada uno de los $T$ posibles errores iniciales, el peor caso escala como $T\cdot T\cdot\epsilon = T^2\epsilon$. Duplicar el horizonte cuadruplica el costo del error: catastrófico para tareas largas.

### 2.3. Antecedentes que ya lograban linealidad

Antes de DAgger, Ross y Bagnell (2010) ya habían propuesto dos remedios con crecimiento (casi) lineal en $T$, ambos con inconvenientes:

- **Forward Training.** Entrena una política **no estacionaria**: una política $\pi_t$ distinta para cada paso $t$, iterando $T$ veces. En la iteración $t$, $\pi_t$ se entrena para imitar a $\pi^*$ sobre la distribución de estados en el paso $t$ inducida por las políticas ya entrenadas $\pi_1,\dots,\pi_{t-1}$. Así cada $\pi_t$ ve la distribución real que encontrará en ejecución. El **Teorema 2.2** (versión generalizada en el paper) da la clave que reutilizará DAgger: si $\pi$ tiene pérdida $\epsilon$ bajo su propia distribución y $Q^{\pi^*}_{T-t+1}(s,a)-Q^{\pi^*}_{T-t+1}(s,\pi^*)\le u$ para toda acción (es decir, el sobrecosto de un único error, medido como diferencia de costo-por-ir hasta el horizonte, está acotado por $u$), entonces

$$J(\pi) \le J(\pi^*) + u\,T\epsilon.$$

  Cuando $u$ es $O(1)$ —por ejemplo si el experto sabe **recuperarse** rápido de los errores del aprendiz, o si la cadena de Markov mezcla rápido— la cota es **lineal** en $T$. El defecto: entrenar $T$ políticas secuenciales es impracticable cuando $T$ es grande o indefinido, y no se puede detener antes de completar las $T$ iteraciones.

- **SMILe** (*Stochastic Mixing Iterative Learning*, emparentado con SEARN de Daumé III et al. 2009 y CPI de Kakade-Langford 2002). Entrena una política **estacionaria pero estocástica**: una mezcla finita de políticas, añadiendo una nueva a la mezcla en cada iteración con la actualización $\pi_n = \pi_{n-1} + \alpha(1-\alpha)^{n-1}(\hat\pi_n - \pi_0)$. Con $\alpha$ del orden $O(1/T^2)$ y $N$ del orden $O(T^2\log T)$ garantiza arrepentimiento casi-lineal. El defecto: la mezcla contiene políticas peores que otras y el controlador resultante puede ser **inestable** (a veces toma acciones malas por la estocasticidad de la mezcla).

DAgger busca lo mejor de ambos: una política **estacionaria y determinista**, con garantía lineal, sin entrenar $T$ redes ni cargar con una mezcla inestable.

## 3. Contribución central

La aportación es doble:

1. **El algoritmo DAgger**, un meta-algoritmo de imitación que en cada iteración (a) ejecuta la política actual para recolectar los estados que realmente visita, (b) consulta al experto qué acción tomaría en **esos** estados, (c) **agrega** esos pares al conjunto de datos acumulado y (d) reentrena la política sobre todo el conjunto agregado. Es simple de implementar, no tiene parámetros libres (más allá del aprendiz supervisado) y maneja de forma natural predicciones tanto discretas como continuas.

2. **La reducción a aprendizaje online no-regret.** DAgger se interpreta como un algoritmo *Follow-The-Leader* (FTL): en la iteración $n$ elige la política óptima **en retrospectiva** sobre todos los datos vistos hasta entonces. Los autores demuestran que **cualquier** algoritmo online sin arrepentimiento, aplicado de esta manera, encuentra una política con buen desempeño bajo su propia distribución inducida. Esto convierte décadas de teoría de aprendizaje online (con sus garantías de arrepentimiento) en garantías de imitación.

El contraste conceptual clave con el *behavioral cloning* es el de **la distribución de datos** sobre la que se entrena:

- **BC (supervisado):** entrena sobre $d_{\pi^*}$, la distribución del **experto**. Nunca ve sus propios errores.
- **DAgger:** entrena sobre una mezcla creciente de $d_{\pi_i}$, las distribuciones que **el propio aprendiz** genera. Ve exactamente los estados problemáticos donde necesita mejorar, y le pide al experto la respuesta correcta ahí.

## 4. Método: DAgger y sus garantías no-regret

### 4.1. El algoritmo

```
Inicializar D ← ∅.
Inicializar π̂₁ a cualquier política en Π.
para i = 1 hasta N:
    Sea πᵢ = βᵢ·π* + (1 − βᵢ)·π̂ᵢ          # política de mezcla experto/aprendiz
    Muestrear trayectorias de T pasos usando πᵢ.
    Formar Dᵢ = {(s, π*(s))} : estados VISITADOS por πᵢ, acciones dadas por el experto.
    Agregar conjuntos: D ← D ∪ Dᵢ.
    Entrenar el clasificador π̂ᵢ₊₁ sobre D.
fin para
Retornar la mejor π̂ᵢ según validación.
```

En su forma más simple: en la primera iteración se usa al experto para recolectar trayectorias y se entrena $\hat\pi_2$ para imitarlo; luego, en la iteración $n$, se usa $\hat\pi_n$ para recolectar **más** trayectorias, se agregan al conjunto $D$, y la siguiente política $\hat\pi_{n+1}$ es la que mejor imita al experto sobre **todo** $D$. La intuición: iteración tras iteración se va construyendo el conjunto de entradas que la política aprendida es probable de encontrar en ejecución, y así el entrenamiento converge hacia la distribución correcta.

**El punto crucial —y lo que la Clase 33 y el lab enfatizan— es la línea de las etiquetas:** el experto etiqueta los estados que **visita el aprendiz** ($\pi_i$), no los que visitaría el experto. Ahí está el arreglo del *distribution shift*: se le pregunta al experto "¿qué harías tú en el lío en el que yo me metí?".

### 4.2. La mezcla $\beta_i$

DAgger permite opcionalmente rodar una política de mezcla $\pi_i = \beta_i\pi^* + (1-\beta_i)\hat\pi_i$ que cede el control al experto una fracción $\beta_i$ del tiempo mientras recolecta datos. Esto ayuda en las primeras iteraciones, cuando $\hat\pi_i$ —entrenada con pocos datos— comete tantos errores que visita estados irrelevantes que dejarán de importar a medida que mejore. Convenciones:

- Se usa típicamente $\beta_1 = 1$, para no tener que especificar una política inicial $\hat\pi_1$: la primera iteración es pura demostración experta (equivale a recolectar el dataset de BC).
- Se puede tomar $\beta_i = p^{i-1}$ para que el uso del experto **decaiga exponencialmente**, como en SMILe/SEARN.
- El **único requisito teórico** es que el promedio $\bar\beta_N = \frac{1}{N}\sum_{i=1}^N \beta_i \to 0$ cuando $N\to\infty$.
- La versión **sin parámetros** que suele rendir mejor en la práctica es $\beta_i = \mathbb{I}(i=1)$ (la indicadora): el experto conduce solo en la primera iteración y de ahí en adelante manda siempre el aprendiz.

### 4.3. La reducción a no-regret y la garantía lineal

En aprendizaje online, un algoritmo entrega en la iteración $n$ una política $\pi_n$ que incurre una pérdida $\ell_n(\pi_n)$; tras observarla, entrega $\pi_{n+1}$ para la siguiente. Un algoritmo es **no-regret** si el arrepentimiento promedio respecto de la mejor política fija en retrospectiva tiende a cero:

$$\frac{1}{N}\sum_{i=1}^{N}\ell_i(\pi_i) - \min_{\pi\in\Pi}\frac{1}{N}\sum_{i=1}^{N}\ell_i(\pi) \le \gamma_N,\qquad \lim_{N\to\infty}\gamma_N = 0.$$

Para pérdidas fuertemente convexas, muchos algoritmos (entre ellos FTL) garantizan $\gamma_N = \tilde{O}(1/N)$. La reducción consiste en **elegir las funciones de pérdida online como la pérdida bajo la distribución de la política actual**:

$$\ell_i(\pi) = \mathbb{E}_{s\sim d_{\pi_i}}[\ell(s,\pi)].$$

Tratando cada mini-lote de trayectorias bajo una política como un único ejemplo de aprendizaje online, DAgger **es** FTL sobre estas pérdidas. El análisis necesita acotar cuánto difieren la distribución de $\pi_i$ (que aún llama al experto) y la de $\hat\pi_i$. El **Lema 4.1** lo hace vía distancia de variación total:

$$\|d_{\pi_i} - d_{\hat\pi_i}\|_1 \le 2T\beta_i,$$

que se obtiene notando que $\pi_i$ ejecuta $\hat\pi_i$ durante los $T$ pasos con probabilidad $(1-\beta_i)^T \ge 1-\beta_i T$. Esta cota es útil (mejor que la trivial de 2) precisamente cuando $\beta_i \le 1/T$ — de ahí que el promedio $\bar\beta_N$ deba tender a cero.

Con $\epsilon_N = \min_{\pi\in\Pi}\frac{1}{N}\sum_{i=1}^N \mathbb{E}_{s\sim d_{\pi_i}}[\ell(s,\pi)]$ la pérdida de la mejor política en retrospectiva, el **Teorema 4.1** establece que existe una política $\hat\pi$ en la secuencia $\hat\pi_{1:N}$ tal que

$$\mathbb{E}_{s\sim d_{\hat\pi}}[\ell(s,\hat\pi)] \le \epsilon_N + \gamma_N + \frac{2\ell_{max}}{N}\Big[n_\beta + T\sum_{i=n_\beta+1}^N \beta_i\Big],$$

donde $n_\beta$ es el mayor índice con $\beta_n > 1/T$. Bajo la suposición de reducción de error (para cualquier distribución de entrada existe alguna $\pi\in\Pi$ con pérdida $\epsilon$) y con $\bar\beta_N\to 0$, en el límite se garantiza hallar una política con pérdida $\epsilon$ bajo **su propia** distribución de estados. Si además $\beta_i = (1-\alpha)^{i-1}$, el término de penalización extra es $\le \frac{1}{N\alpha}[\log T + 1]$ y se vuelve despreciable con $N$ del orden $\tilde{O}(T)$ —el mismo número de iteraciones que ya se necesita para que $\gamma_N$ sea despreciable.

Reuniendo las piezas, los teoremas principales del paper (caso de muestras infinitas) son:

- **Teorema 3.1.** Si $N$ es $\tilde{O}(T)$, existe $\hat\pi\in\hat\pi_{1:N}$ con $\mathbb{E}_{s\sim d_{\hat\pi}}[\ell(s,\hat\pi)] \le \epsilon_N + O(1/T)$. Si el costo de la tarea $C$ coincide con (o está acotado por) la pérdida sustituta $\ell$, esto da directamente $J(\hat\pi)\le T\epsilon_N + O(1)$.
- **Teorema 3.2.** Para un costo de tarea arbitrario $C$, combinando con el Teorema 2.2: si $N$ es $\tilde{O}(uT)$, existe $\hat\pi$ con

$$J(\hat\pi) \le J(\pi^*) + uT\epsilon_N + O(1).$$

Esta es **la victoria central**: crecimiento **lineal** en $T$ (frente al $T^2\epsilon$ del *behavioral cloning*), siempre que exista una buena política en la clase bajo la distribución agregada. El resultado se apoya únicamente en la propiedad no-regret del FTL subyacente sobre pérdidas fuertemente convexas, así que vale para cualquier otro aprendiz online sin arrepentimiento.

**Nota sobre muestras finitas.** En la práctica se muestrean $m$ trayectorias por iteración y se observa la pérdida empírica. Vía la desigualdad de **Azuma-Hoeffding** (los residuos forman una martingala acotada), el **Teorema 3.3** garantiza, con probabilidad $\ge 1-\delta$, que existe $\hat\pi$ con $\mathbb{E}_{s\sim d_{\hat\pi}}[\ell(s,\hat\pi)] \le \hat\epsilon_N + O(1/T)$ tomando $N$ del orden $O(T^2\log(1/\delta))$ y $m$ del orden $O(1)$; aprovechando la convexidad fuerte se puede reducir a $\tilde{O}(T\log(1/\delta))$ trayectorias. El término de generalización que aparece es $\ell_{max}\sqrt{2\log(1/\delta)/(mN)}$.

## 5. Experimentos

El paper valida DAgger en dos problemas exigentes de imitación y una tarea de etiquetado de secuencias, comparando siempre contra el supervisado (BC), SMILe y SEARN.

- **Super Tux Kart (conducción 3D).** Se entrena la computadora para dirigir un kart a velocidad fija en la pista "Star Track" (que flota en el espacio, así que el kart puede caerse por el borde). La entrada son características de imagen (valores LAB de cada pixel de una imagen reescalada a 25×19); el controlador base es una regresión lineal *ridge* que actualiza el volante a 5 Hz. La métrica es el **número promedio de caídas por vuelta**. Resultados nítidos: el **supervisado no mejora** aunque se recolecten más datos —las vueltas de entrenamiento son todas similares y no enseñan a recuperarse de errores. SMILe mejora algo pero sigue cayéndose unas dos veces por vuelta tras 20 iteraciones (en parte por la estocasticidad de su mezcla). **DAgger** ($\beta_i = \mathbb{I}(i=1)$) logra una política que **nunca se cae** de la pista tras 15 iteraciones, y ya tras 5 iteraciones casi no se cae, superando claramente a ambos; además su conducción es cualitativamente más suave.

- **Super Mario Bros.** (videojuego de plataformas). El objetivo es jugar a partir de características de imagen; el **experto es un planificador casi óptimo** con acceso al estado interno del juego que simula exactamente las consecuencias de las acciones futuras. La acción son 4 variables binarias (izquierda/derecha/salto/velocidad) y el aprendiz base son 4 SVM lineales a 5 Hz sobre 27152 características binarias (muy dispersas). La métrica es la distancia promedio recorrida por etapa (rango $\approx[0,4300]$). El supervisado se **estanca**: bajo el controlador aprendido, Mario suele quedarse **atascado contra un obstáculo** en vez de saltarlo, porque el experto siempre salta con antelación y nunca demuestra cómo desatascarse estando pegado al obstáculo. Los métodos iterativos aprenden a desatascarse al encontrar esas situaciones en iteraciones posteriores. DAgger supera a SMILe y a SEARN para todo $\alpha$ probado. Detalle interesante sobre $\beta_i$: $\beta_i = 0.9^{i-1}$ (D0.9) converge notablemente más lento (usar mucho al experto tarde retrasa el aprendizaje); $\beta_i=0.5^{i-1}$ (D0.5, distancia 3030) rinde ligeramente mejor que la indicadora (D0, distancia 2980), porque usar al experto una fracción pequeña del tiempo **desatasca** a Mario y permite recolectar datos más variados y útiles, en lugar de acumular datos donde queda atascado en el mismo sitio.

- **Reconocimiento de escritura a mano (handwriting / OCR).** Siguiendo a Daumé III et al. (2009), se trata la predicción estructurada como una forma degenerada de imitación (dinámica determinista y trivial: las predicciones previas se pasan como entradas de las futuras). Sobre el dataset de Taskar et al. (2003) (~6600 palabras, >52000 caracteres, 10 *folds*), prediciendo cada carácter de izquierda a derecha con un SVM lineal y usando el carácter previo predicho como característica. Resultados: la base sin estructura (predecir cada carácter independientemente) logra 82 %; el supervisado con la característica del carácter previo **correctamente etiquetado** sube a 83.6 %; **DAgger** sube a **85.5 %**. SEARN con $\alpha=1$ (iteración de política pura) rinde sorprendentemente bien aquí —similar a DAgger— porque solo una pequeña parte de la entrada (el carácter previo) está influida por la política, lo que vuelve el problema mucho menos inestable que en imitación general. Con decodificación *greedy* de una sola pasada, DAgger ya es competitivo con el estado del arte.

## 6. Limitaciones

- **Requiere un experto interactivo consultable durante el entrenamiento.** Esta es la limitación práctica más importante y la que la Clase 33 destaca: DAgger necesita poder preguntarle al experto $\pi^*(s)$ para **cualquier** estado $s$ que el aprendiz visite, no solo sobre trayectorias grabadas de antemano. Eso exige un experto disponible *en el bucle* —costoso o inviable cuando el experto es un humano (habría que etiquetar a mano cada estado raro que produce el aprendiz) o cuando visitar ciertos estados en el mundo real es peligroso (un auto que "practica" caerse por un barranco). En el lab de la clase esto no es problema porque el experto es un **DQN** consultable a voluntad, pero fuera de la simulación es una barrera real.
- **Supuesto de convexidad fuerte / no-regret.** Las garantías requieren un método no-regret o una pérdida sustituta fuertemente convexa —una hipótesis más fuerte (aunque común) que las reducciones de error puras (Beygelzimer et al. 2005), que solo requieren clasificación.
- **Supuesto de realizabilidad implícito.** La cota es útil solo si $\epsilon_N$ (la pérdida de la mejor política en retrospectiva bajo la distribución agregada) es pequeña; si la clase $\Pi$ no contiene ninguna política capaz de imitar bien al experto sobre los estados agregados, la garantía se vacía.
- **Depende de $u$ para el caso de costo arbitrario.** Cuando el costo de la tarea no coincide con la pérdida sustituta, el factor $u$ (sobrecosto de un error, del Teorema 2.2) puede en el peor caso ser $O(T)$, devolviendo el crecimiento cuadrático. DAgger brilla cuando el experto se recupera bien de los errores ($u = O(1)$).
- **Muestras finitas exigen bastantes iteraciones/datos.** $N$ del orden $O(T^2\log(1/\delta))$ en el análisis básico puede ser mucho para horizontes largos, aunque la convexidad fuerte lo mejora.

## 7. Conexión con la Clase 33 y con el laboratorio

DAgger es el **corazón teórico** de la Clase 33. La clase presenta la imitación en dos niveles:

- **Behavioral cloning (imitación pura)** como el punto de partida natural y su **fallo diagnosticado**: los dos motivos —*compounding error* y estados fuera de distribución— son literalmente los que este paper formaliza con la cota $O(T^2\epsilon)$. Cuando la clase muestra que un clonador entrenado con demostraciones se descarrila apenas se desvía un poco, está mostrando el ejemplo ajustado de Ross-Bagnell/Kääriäinen en acción.
- **DAgger (slide 36)** como la corrección: iterar rodando la política actual, consultar al experto sobre los **estados visitados**, agregarlos al dataset y reentrenar, con la garantía lineal $O(T\epsilon)$ que se sigue de la reducción a no-regret.

El **laboratorio de la clase implementa DAgger sobre Atari Breakout con un experto DQN**. El mapeo es directo: la política aprendida es una red que predice la acción a partir de la imagen del juego; el experto $\pi^*$ es un agente DQN preentrenado (el mismo linaje de la Clase 31, [/papers/mnih-dqn-nature-2015](/papers/mnih-dqn-nature-2015)) que se puede consultar en cualquier estado —justo lo que DAgger necesita y un humano no podría dar. El lab hace visible, sobre un juego concreto, la diferencia entre entrenar sobre $d_{\pi^*}$ (BC, que se descarrila cuando el aprendiz se mete en configuraciones de ladrillos que el DQN nunca dejó ocurrir) y entrenar sobre la distribución **agregada** de estados que el propio aprendiz visita (DAgger, que le pregunta al DQN cómo reaccionar ahí). El experimento de Mario del paper —donde el supervisado se queda atascado contra obstáculos que el experto nunca demuestra— es el análogo directo de lo que se observa en Breakout.

## 8. Relación con IRL / GAIL

DAgger pertenece a la familia de imitación que aprende **directamente la política** (mapeo estado→acción). La Clase 33 la contrasta con el **aprendizaje reforzado inverso** (IRL), que en vez de copiar acciones intenta recuperar la **función de recompensa** que el experto parece estar optimizando (Ng-Russell 2000; Abbeel-Ng 2004, *apprenticeship learning*; Ziebart et al. 2008, *MaxEnt IRL*), para luego optimizar una política con RL sobre esa recompensa. La ventaja de IRL es que la recompensa es un objeto más **transferible y compacto** que puede generalizar mejor a dinámicas nuevas; su desventaja es el costo (resolver un problema de RL en el bucle interno) y la ambigüedad (muchas recompensas explican el mismo comportamiento).

El mismo trío de CMU anticipó la conexión en la sección de trabajo futuro: mencionan usar clasificadores base basados en **control óptimo inverso** (IRL) para aprender una función de costo que ayude a la predicción, y sugieren que técnicas de agregación de datos similares a DAgger, apoyadas en una estimación de *cost-to-go*, podrían dar garantías en RL. Esa intuición desemboca años después en **GAIL** (*Generative Adversarial Imitation Learning*, Ho-Ermon 2016, [/clase-33 papers Ho-GAIL-2016]): GAIL evita el bucle de IRL emparejando distribuciones de ocupación estado-acción del aprendiz y del experto mediante un discriminador adversarial, sin necesidad de un experto interactivo consultable como DAgger —solo demostraciones fijas— pero pagando con la inestabilidad del entrenamiento adversarial y la necesidad de interactuar con el entorno. Se puede leer la trayectoria de la clase como: BC (falla por *distribution shift*) → **DAgger** (arregla el shift consultando al experto interactivo) → IRL (recupera la recompensa, más transferible) → GAIL (empareja distribuciones adversarialmente, sin experto interactivo). DAgger es el eslabón que hace explícito *por qué* la distribución de datos —y no solo la calidad del clasificador— es el problema.

## 9. Nota para el lector experto en salud / MDM (FHIR)

El *distribution shift* que hunde al *behavioral cloning* tiene un análogo exacto en el **matching de identidades (MDM)**: un clasificador de emparejamiento entrenado con pares etiquetados por *stewards* aprende la distribución de casos que los *stewards* revisan —típicamente los "interesantes" o dudosos que llegan a su cola— pero en producción el sistema enfrenta la distribución **real** de pares candidatos que su propio *blocking* genera, que puede ser muy distinta (más registros con transliteraciones, RUT mal digitados, nombres compuestos truncados por sistemas origen). El clasificador tiene baja pérdida sobre $d_{\text{steward}}$ y aun así falla sistemáticamente sobre $d_{\text{producción}}$: es el mismo desajuste entre la distribución de entrenamiento y la distribución inducida por el sistema desplegado. DAgger sugiere la cura correcta: en vez de re-etiquetar aleatoriamente, hacer **active learning dirigido a los estados que el sistema efectivamente visita** —rodar el matcher en producción (o en *shadow mode*), capturar precisamente los pares candidatos donde decide, y llevarle **esos** al *steward* (el "experto interactivo") para que los etiquete, agregándolos al conjunto y reentrenando. Cada ciclo cierra la brecha entre la distribución de entrenamiento y la de operación, exactamente como DAgger construye iterativamente el conjunto de estados que su política encontrará en ejecución.
