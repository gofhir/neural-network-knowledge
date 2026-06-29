---
title: "Profundizacion - Olvido Catastrófico"
weight: 20
math: true
---

> Esta pagina complementa la [teoria de la clase 32](/clases/clase-32/teoria) con las derivaciones formales del aprendizaje continuo. Seis bloques: **Parte I** formaliza el olvido y el dilema estabilidad-plasticidad, con las metricas ACC/BWT/FWT. **Parte II** deriva EWC desde Bayes y la aproximacion de Laplace, mas Synaptic Intelligence por integral de camino. **Parte III** desarrolla la destilacion de conocimiento (LwF) e iCaRL (herding + nearest-mean). **Parte IV** plantea GEM como programa cuadratico de proyeccion de gradientes y su dual. **Parte V** trata los metodos de mascara y arquitectura (PiggyBack, SupSup, HAT). **Parte VI** formaliza los tres escenarios de [van de Ven](/papers/three-scenarios-van-de-ven-2019) y explica por que la regularizacion colapsa en class-incremental learning. Marco transversal en [Aprendizaje Continuo](/fundamentos/aprendizaje-continuo).

---

## Parte I — Formalizacion del olvido y el dilema estabilidad-plasticidad

### I.1 El setup de tareas secuenciales

El aprendizaje continuo (o *continual / lifelong learning*) abandona la hipotesis i.i.d. del aprendizaje supervisado clasico. En vez de un unico dataset $\mathcal{D}$, recibimos una **secuencia** de tareas $\mathcal{T}_1, \mathcal{T}_2, \dots, \mathcal{T}_T$, cada una con su propia distribucion de datos $\mathcal{D}_t = \{(x_n^t, y_n^t)\}$. La restriccion definitoria es que, al entrenar la tarea $t$, **el acceso a los datos de las tareas $1,\dots,t-1$ esta vedado** (o severamente limitado). El objetivo es minimizar el riesgo esperado sobre **todas** las tareas vistas:

$$
\theta^* = \arg\min_\theta \; \sum_{t=1}^{T} \mathbb{E}_{(x,y)\sim\mathcal{D}_t}\big[\ell(f_\theta(x), y)\big]
$$

pero solo se observa, en cada momento $t$, el sumando $\mathcal{L}_t(\theta) = \mathbb{E}_{(x,y)\sim\mathcal{D}_t}[\ell(f_\theta(x),y)]$. La cota superior de referencia es el *joint training* (u *offline*), donde se entrena con la mezcla de todos los datos a la vez; la cota inferior es el *fine-tuning* secuencial ingenuo, que sufre el olvido en su forma mas cruda.

### I.2 Por que el SGD mueve los pesos fuera del optimo viejo

Sea $\theta^*_{t-1}$ una solucion de bajo error para las tareas $1,\dots,t-1$. Al entrenar la tarea $t$ con descenso de gradiente, el paso es

$$
\theta \leftarrow \theta - \eta\, \nabla_\theta \mathcal{L}_t(\theta)
$$

y este gradiente **no contiene ninguna informacion** sobre las pérdidas viejas $\mathcal{L}_{<t}$. Expandiendo la pérdida de una tarea vieja $\nu < t$ por Taylor alrededor de su óptimo $\theta^*_\nu$ (donde $\nabla\mathcal{L}_\nu(\theta^*_\nu)=0$):

$$
\mathcal{L}_\nu(\theta) \approx \mathcal{L}_\nu(\theta^*_\nu) + \tfrac{1}{2}(\theta - \theta^*_\nu)^\top H_\nu (\theta - \theta^*_\nu), \qquad H_\nu = \nabla^2_\theta \mathcal{L}_\nu(\theta^*_\nu)
$$

El término lineal se anula porque estamos en un mínimo. El incremento de la pérdida vieja al desplazarse $\Delta\theta = \theta - \theta^*_\nu$ es entonces **cuadrático** y modulado por el Hessiano $H_\nu$: moverse en direcciones de alta curvatura (autovalores grandes de $H_\nu$) destruye la tarea $\nu$ rapidamente, mientras que moverse en el espacio nulo de $H_\nu$ no la afecta. El olvido catastrófico es exactamente esto: el gradiente de $t$, ciego a $H_\nu$, suele tener componente grande en direcciones de alta curvatura de las tareas viejas. Esta misma expansión cuadrática es la base teórica de **toda la familia de regularización** (Parte II): si pudiéramos estimar $H_\nu$ (o un sustituto barato), podríamos penalizar selectivamente los movimientos dañinos.

{{< concept-alert type="clave" >}}
El **dilema estabilidad-plasticidad** es el trade-off central del campo. **Estabilidad** = no perturbar los pesos importantes para lo viejo (anti-olvido). **Plasticidad** = mover los pesos libremente para aprender lo nuevo. Congelar todo (estabilidad maxima) mata el aprendizaje nuevo; dejar todo libre (plasticidad maxima) produce olvido catastrófico. Cada metodo de la clase es una receta distinta para resolver este balance: regularizacion lo resuelve con un resorte por peso; replay con datos viejos; arquitectura con capacidad dedicada.
{{< /concept-alert >}}

### I.3 Las metricas: ACC, BWT, FWT

[Lopez-Paz y Ranzato](/papers/gem-lopez-paz-2017) introdujeron el protocolo de evaluacion estandar. Tras entrenar la secuencia, se construye una **matriz $R \in \mathbb{R}^{T\times T}$** donde $R_{i,j}$ es la accuracy de test sobre la tarea $t_j$ **despues** de haber terminado de aprender la tarea $t_i$. Sea $\bar{b}_j$ la accuracy de test sobre $t_j$ con el modelo recién inicializado al azar (la línea base aleatoria). Se definen tres metricas:

$$
\text{ACC} = \frac{1}{T}\sum_{i=1}^{T} R_{T,i}
\qquad\text{(accuracy promedio al final del entrenamiento)}
$$

$$
\text{BWT} = \frac{1}{T-1}\sum_{i=1}^{T-1}\big(R_{T,i} - R_{i,i}\big)
\qquad\text{(backward transfer)}
$$

$$
\text{FWT} = \frac{1}{T-1}\sum_{i=2}^{T}\big(R_{i-1,i} - \bar{b}_i\big)
\qquad\text{(forward transfer)}
$$

Interpretacion:

| Metrica | Mide | Signo deseable |
| --- | --- | --- |
| **ACC** | Rendimiento medio sobre todas las tareas al final | Alto |
| **BWT** | Cómo aprender $t_i$ afectó a las tareas previas $t_{j<i}$ | Positivo (transferencia útil); **muy negativo = olvido catastrófico** |
| **FWT** | Cómo aprender afecta a tareas futuras aún no vistas | Positivo (capacidad *zero-shot* vía descriptores de tarea) |

**BWT** es la definicion operativa del olvido: $R_{T,i} - R_{i,i}$ es cuánto cayó la tarea $i$ entre el momento en que se aprendió ($R_{i,i}$, en la diagonal) y el final ($R_{T,i}$). Entre dos métodos con ACC similar, se prefiere el de mayor BWT. Estas tres metricas se volvieron el *estándar de facto* del campo.

---

## Parte II — EWC y la matriz de Fisher

### II.1 La interpretacion bayesiana: posterior de A como prior de B

[Elastic Weight Consolidation](/papers/ewc-kirkpatrick-2017) parte de leer el aprendizaje como inferencia bayesiana. Por la regla de Bayes, el log-posterior de los parametros dado todo el dataset $\mathcal{D}$ es

$$
\log p(\theta\mid\mathcal{D}) = \log p(\mathcal{D}\mid\theta) + \log p(\theta) - \log p(\mathcal{D})
$$

donde $\log p(\mathcal{D}\mid\theta) = -\mathcal{L}(\theta)$ (la verosimilitud es el negativo de la pérdida). Si los datos se parten en dos subconjuntos **independientes** —tarea A ($\mathcal{D}_A$) y tarea B ($\mathcal{D}_B$)— se puede reordenar:

$$
\boxed{\;\log p(\theta\mid\mathcal{D}) = \underbrace{\log p(\mathcal{D}_B\mid\theta)}_{\text{pérdida de B}} + \underbrace{\log p(\theta\mid\mathcal{D}_A)}_{\text{posterior de A = prior de B}} - \log p(\mathcal{D}_B)\;}
$$

Este es el corazón del argumento. Toda la información sobre A —incluyendo qué parámetros fueron importantes— quedó absorbida en el posterior $p(\theta\mid\mathcal{D}_A)$. Aprender continualmente es **encadenar inferencias bayesianas**: el posterior de cada tarea se convierte en el prior de la siguiente.

### II.2 La aproximacion de Laplace y por que la diagonal de Fisher mide la importancia

El posterior verdadero $p(\theta\mid\mathcal{D}_A)$ es intratable. La **aproximacion de Laplace** (MacKay, 1992) lo reemplaza por una Gaussiana centrada en el óptimo $\theta^*_A$ con precisión igual a la curvatura local:

$$
p(\theta\mid\mathcal{D}_A) \approx \mathcal{N}\big(\theta^*_A,\; (F)^{-1}\big), \qquad \log p(\theta\mid\mathcal{D}_A) \approx -\tfrac{1}{2}(\theta - \theta^*_A)^\top F\,(\theta - \theta^*_A) + \text{const}
$$

donde $F$ es la **matriz de informacion de Fisher** evaluada en $\theta^*_A$:

$$
F = \mathbb{E}_{x\sim\mathcal{D}_A}\;\mathbb{E}_{y\sim p_\theta(y\mid x)}\Big[\nabla_\theta \log p_\theta(y\mid x)\, \nabla_\theta \log p_\theta(y\mid x)^\top\Big]
$$

Fisher se elige por tres propiedades (Pascanu y Bengio, 2013): **(a)** cerca de un mínimo es equivalente a la segunda derivada de la pérdida, es decir aproxima el Hessiano $H_\nu$ de la Parte I —un peso con Fisher alto es uno donde la pérdida sube rápido si se lo mueve; **(b)** se calcula **solo con derivadas de primer orden** (productos exteriores de gradientes), barato incluso en modelos grandes; **(c)** es semidefinida positiva, lo que vuelve la penalización convexa. Para hacerlo tratable, EWC retiene **solo la diagonal** $F_i$: un posterior gaussiano factorizado que ignora correlaciones entre pesos.

### II.3 La funcion de pérdida de EWC

Sustituyendo la Gaussiana de Laplace como prior y tomando la diagonal, minimizar el negativo del log-posterior al entrenar B da:

$$
\boxed{\;\mathcal{L}(\theta) = \mathcal{L}_B(\theta) + \sum_i \frac{\lambda}{2}\, F_i\, (\theta_i - \theta^*_{A,i})^2\;}
$$

donde $\mathcal{L}_B$ es la pérdida de la tarea nueva, $F_i$ la diagonal de Fisher de A, $(\theta_i - \theta^*_{A,i})^2$ el desplazamiento cuadrático del peso $i$, y $\lambda$ la rigidez global del resorte. La lectura mecánica: cada parámetro queda anclado a $\theta^*_{A,i}$ por un **resorte elástico** cuya constante $\lambda F_i$ es grande para pesos cruciales (casi congelados) y pequeña para pesos irrelevantes (libres de reaprender). Esto es exactamente la "ralentización selectiva del aprendizaje" que da nombre al método. Nótese que la regularización L2 uniforme es el caso degenerado $F_i = \text{const}$: protege todos los pesos por igual y por eso no logra aprender B.

### II.4 Online EWC

En la formulación original, cada tarea nueva añade un término de anclaje, por lo que el número de penalizaciones crece con $T$. Como **la suma de dos penalizaciones cuadráticas es a su vez cuadrática**, *Online EWC* (Schwarz et al., 2018) mantiene un **único** término con una Fisher acumulada:

$$
\mathcal{L}(\theta) = \mathcal{L}_t(\theta) + \frac{\lambda}{2}\sum_i \tilde{F}_i^{(t-1)}\,(\theta_i - \theta^{*(t-1)}_i)^2, \qquad \tilde{F}^{(t)} = \gamma\,\tilde{F}^{(t-1)} + F^{(t)}
$$

con $\gamma\in(0,1]$ un factor de olvido que decae la importancia de tareas muy antiguas. El costo es así **constante** en el número de tareas.

### II.5 Synaptic Intelligence: importancia por integral de camino

[Synaptic Intelligence](/papers/synaptic-intelligence-zenke-2017) (SI) es el hermano gemelo de EWC: misma penalización cuadrática, **distinta forma de medir la importancia**. En vez de calcular Fisher *offline en un punto* (el mínimo de cada tarea), SI estima la importancia *online a lo largo de toda la trayectoria de entrenamiento* $\theta(t)$.

El cambio de la pérdida en un paso infinitesimal se aproxima por $\mathcal{L}(\theta + \delta) - \mathcal{L}(\theta) \approx \sum_k g_k\,\delta_k$ con $g_k = \partial\mathcal{L}/\partial\theta_k$. Como el gradiente es un **campo conservativo**, integrar a lo largo del camino recupera la diferencia de pérdida entre extremos, y la integral se descompone por parámetro. La contribución del parámetro $k$ durante la tarea $\mu$ es

$$
\omega_k^\mu = -\int_{t_{\mu-1}}^{t_\mu} g_k(t)\, \dot{\theta}_k(t)\, dt \;\approx\; -\sum_{\text{pasos}} g_k(t)\,\big(\theta_k(t+1) - \theta_k(t)\big)
$$

(el signo menos porque interesa la *reducción* de pérdida). La fuerza de regularización por parámetro acumula esto, normalizado por la distancia recorrida $\Delta_k^\nu = \theta_k(t_\nu) - \theta_k(t_{\nu-1})$:

$$
\Omega_k^\mu = \sum_{\nu<\mu} \frac{\omega_k^\nu}{(\Delta_k^\nu)^2 + \xi}, \qquad
\tilde{\mathcal{L}}^\mu = \mathcal{L}^\mu + c\sum_k \Omega_k^\mu\,(\tilde{\theta}_k - \theta_k)^2
$$

El denominador $(\Delta_k^\nu)^2$ asegura que el término tenga **las mismas unidades** que la pérdida y normaliza por cuánto se movió el peso; $\xi$ es un amortiguamiento que acota la expresión cuando $\Delta_k\to 0$; $c$ es la fuerza global ($c<1$ en la práctica, porque el ruido de SGD hace que $\omega$ **sobreestime** la importancia verdadera).

{{< concept-alert type="recordar" >}}
**Fisher vs. integral de camino.** Para una pérdida cuadrática, la Fisher empírica **en el mínimo es 0** (el gradiente se anula), mientras la integral de camino de SI es proporcional a la diagonal del Hessiano. SI obtiene así un estimador útil de la curvatura **sin gradientes adicionales**, justo donde la Fisher empírica colapsaría. Ambos métodos convergen en producir un resorte cuadrático $\sum_i (\text{importancia}_i)(\theta_i - \theta^*_i)^2$; difieren solo en cómo estiman la importancia.
{{< /concept-alert >}}

---

## Parte III — Distillation (LwF) e iCaRL

### III.1 Knowledge distillation con soft targets y temperatura

[Learning without Forgetting](/papers/lwf-li-2016) (LwF) no guarda datos viejos: usa los **datos de la tarea nueva** y, para preservar las salidas de las cabezas viejas, aplica destilación de conocimiento. Antes de entrenar, registra las salidas del modelo original sobre los nuevos inputs, $y_o$, y luego las usa como *soft targets*. La pérdida total combina tres términos:

$$
\arg\min_{\hat\theta_s,\hat\theta_o,\hat\theta_n}\Big[\, \lambda_o\,\mathcal{L}_{old}(y_o, \hat{y}_o) + \mathcal{L}_{new}(y_n, \hat{y}_n) + R(\hat\theta) \,\Big]
$$

donde $\theta_s$ son parámetros compartidos, $\theta_o$ las cabezas viejas, $\theta_n$ la cabeza nueva, y $R$ un weight decay. La pérdida nueva $\mathcal{L}_{new}$ es la cross-entropy multiclase estándar. La pérdida vieja es la **Knowledge Distillation loss** de Hinton, una cross-entropy con probabilidades suavizadas por **temperatura** $T$:

$$
\mathcal{L}_{old}(y_o, \hat{y}_o) = -\sum_{i=1}^{l} y_o'^{(i)} \log \hat{y}_o'^{(i)}, \qquad
y_o'^{(i)} = \frac{(y_o^{(i)})^{1/T}}{\sum_j (y_o^{(j)})^{1/T}}, \quad
\hat{y}_o'^{(i)} = \frac{(\hat{y}_o^{(i)})^{1/T}}{\sum_j (\hat{y}_o^{(j)})^{1/T}}
$$

Con $T>1$ se **suaviza** la distribución, aumentando el peso de los logits pequeños y forzando a la red a codificar mejor las *similitudes entre clases* (la "dark knowledge"). LwF usa $T=2$. El parámetro $\lambda_o$ balancea viejo contra nuevo (típicamente $\lambda_o=1$). Un hallazgo notable de las ablaciones: lo crucial es **restringir las salidas viejas a parecerse a las del original**; la medida de similitud exacta (KD, L1, L2) importa poco.

### III.2 iCaRL: representacion + exemplars

[iCaRL](/papers/icarl-rebuffi-2017) ataca el caso *class-incremental* verdadero (un único clasificador sobre todas las clases vistas). Combina destilación con un presupuesto fijo de exemplars reales y desacopla el **extractor de features** $\varphi$ del clasificador.

**Clasificador: nearest-mean-of-exemplars.** Para cada clase $y$ se calcula un prototipo como la media de los features de sus exemplars $P_y$, y se clasifica por el prototipo más cercano:

$$
\mu_y = \frac{1}{|P_y|}\sum_{p\in P_y}\varphi(p), \qquad y^* = \arg\min_{y=1,\dots,t}\|\varphi(x) - \mu_y\|
$$

Con features L2-normalizados, esto equivale a $y^* = \arg\max_y \mu_y^\top \varphi(x)$: un clasificador lineal cuyo vector de peso **no está desacoplado** de la representación, sino que cambia consistentemente con $\varphi$ —clave para que no quede obsoleto cuando $\varphi$ se actualiza.

**Aprendizaje de representación.** Al llegar clases nuevas, iCaRL minimiza una pérdida que une clasificación (clases nuevas) y destilación binaria (clases viejas, reproduciendo los scores sigmoidales $q_i^y$ pre-actualización):

$$
\ell(\Theta) = -\!\!\sum_{(x_i,y_i)\in D}\Bigg[\underbrace{\sum_{y=s}^{t}\!\big(\delta_{y=y_i}\log g_y(x_i) + \delta_{y\neq y_i}\log(1-g_y(x_i))\big)}_{\text{clasificación}} + \underbrace{\sum_{y=1}^{s-1}\!\big(q_i^y\log g_y(x_i) + (1-q_i^y)\log(1-g_y(x_i))\big)}_{\text{destilación}}\Bigg]
$$

**Herding para seleccionar exemplars.** Con presupuesto fijo $K$ y $t$ clases, se asignan $m=K/t$ exemplars por clase. El *herding* los elige **iterativamente** para que la media de features de los exemplars escogidos aproxime mejor la media de la clase completa $\mu$. Tras elegir $p_1,\dots,p_{k-1}$, el siguiente es

$$
p_k = \arg\min_{x\in X}\;\Big\|\,\mu - \tfrac{1}{k}\big(\varphi(x) + \textstyle\sum_{j=1}^{k-1}\varphi(p_j)\big)\Big\|
$$

El resultado es una **lista priorizada**: los primeros exemplars son los más representativos. Esto hace trivial la **reducción** cuando baja $m$ —se descartan los últimos y cualquier prefijo sigue siendo buena aproximación—, sin recalcular medias (los datos viejos ya no están disponibles).

---

## Parte IV — GEM: proyeccion de gradientes

### IV.1 La formulacion con restricciones

[Gradient Episodic Memory](/papers/gem-lopez-paz-2017) (GEM) pertenece a la familia de memoria/replay, pero con una idea distinta: no reentrena sobre la memoria, la usa como **conjunto de restricciones**. Mantiene una memoria episódica $M_k$ por tarea pasada $k$. Al observar el ejemplo $(x,t,y)$, resuelve

$$
\min_\theta\; \ell(f_\theta(x,t), y)
\quad\text{s.a.}\quad \ell(f_\theta, M_k) \le \ell(f_\theta^{t-1}, M_k)\;\;\forall k < t
$$

es decir: minimizar la pérdida del ejemplo actual **sin que la pérdida en la memoria de ninguna tarea pasada supere** el valor que tenía al terminar esa tarea ($f_\theta^{t-1}$ es el estado al final de $t-1$).

### IV.2 Linealizacion: del valor de pérdida al angulo de gradientes

Dos observaciones lo vuelven práctico: (1) no hace falta guardar predictores viejos, basta garantizar que la pérdida vieja no suba; (2) asumiendo **linealidad local** (pasos pequeños), un aumento de la pérdida de la tarea $k$ se diagnostica por el ángulo entre su gradiente $g_k = \nabla_\theta\ell(f_\theta, M_k)$ y la actualización propuesta $g = \nabla_\theta\ell(f_\theta(x,t),y)$. Las restricciones se reescriben como **productos internos no negativos**:

$$
\langle g, g_k\rangle \ge 0, \qquad \forall k < t
$$

Si todas se cumplen, $g$ no aumenta la pérdida vieja y se aplica directo. Si alguna se viola, hay olvido inminente y se proyecta.

### IV.3 El programa cuadratico y su dual

Cuando hay violaciones, se busca el gradiente $\tilde{g}$ **más cercano** a $g$ (en norma $\ell_2$) que satisfaga todas las restricciones:

$$
\min_{\tilde{g}}\; \tfrac{1}{2}\|g - \tilde{g}\|_2^2 \quad\text{s.a.}\quad \langle\tilde{g}, g_k\rangle \ge 0\;\;\forall k<t
$$

El primal tiene $p$ variables (el número de parámetros de la red, millones). La maniobra clave es pasar al **dual**: con $G = (g_1,\dots,g_{t-1})$ la matriz de gradientes de memoria,

$$
\min_v\; \tfrac{1}{2}v^\top G G^\top v + g^\top G^\top v \quad\text{s.a.}\quad v \ge 0
$$

un QP en **solo $t-1$ variables** (el número de tareas vistas, $t-1 \ll p$). Resuelto para $v^\star$, el gradiente proyectado se recupera como

$$
\tilde{g} = G^\top v^\star + g, \qquad \theta \leftarrow \theta - \alpha\tilde{g}
$$

### IV.4 A-GEM: la relajacion promedio

El cuello de botella de GEM es que **cada iteración requiere un backward pass por tarea previa** (más resolver el QP). **A-GEM** (Chaudhry et al., 2019) reemplaza las $t-1$ restricciones por **una sola**, contra el gradiente promedio $g_{\text{ref}}$ de un lote muestreado de la memoria total. La proyección colapsa a una fórmula cerrada (sin QP iterativo): si $\langle g, g_{\text{ref}}\rangle < 0$,

$$
\tilde{g} = g - \frac{g^\top g_{\text{ref}}}{g_{\text{ref}}^\top g_{\text{ref}}}\, g_{\text{ref}}
$$

que es simplemente la **proyección ortogonal** de $g$ fuera de la dirección de $g_{\text{ref}}$. A-GEM conserva la idea esencial a una fracción del costo, y se volvió la variante de referencia.

---

## Parte V — Metodos de mascara y arquitectura

Esta familia ataca el olvido aislando **parámetros dedicados** por tarea sobre un backbone, en vez de regularizar o reentrenar. Comparten la dificultad técnica de aprender **máscaras binarias**, que no son diferenciables.

### V.1 PiggyBack: mascara binaria con straight-through

[PiggyBack](/papers/piggyback-mallya-2018) mantiene el backbone $W$ **congelado** y aprende una máscara binaria por tarea. Se conserva una máscara real $m_r$, se la umbrala para obtener la binaria $m$, y se la aplica elementwise:

$$
m_{ji} = \begin{cases} 1, & (m_r)_{ji} \ge \tau \\ 0, & \text{en otro caso}\end{cases}, \qquad y = (W \odot m)\,x
$$

El umbral es no diferenciable (gradiente cero casi en todas partes). El **straight-through estimator** (Courbariaux et al.) lo resuelve ignorando el umbral en el backward: el gradiente respecto de $m$ se usa directamente como estimador del gradiente respecto de $m_r$,

$$
\delta m = (\delta y \cdot x^\top) \odot W
$$

Solo se actualiza $m_r$; tras entrenar, se descarta $m_r$ y se guarda solo la máscara binaria $m$ (**1 bit por parámetro**). Como $|\delta m| \propto |W|$, conviene inicializar $m_r$ a una constante pequeña, usar Adam, e inicializar las máscaras a todo-1 (que reproduce el backbone base).

### V.2 SupSup: supermascara + inferencia de tarea por entropia

[SupSup](/papers/supsup-wortsman-2020) lleva la idea al extremo: el backbone $W$ es **aleatorio fijo** (basta guardar la semilla). Para la tarea $i$ se aprende una supermáscara $M^i$ (vía Edge-Popup con straight-through), y se computa $p = f(x, W \odot M^i)$.

Lo distintivo es **inferir la tarea sin task-ID**. A cada una de las $k$ máscaras se le asocia un coeficiente $\alpha_i\in[0,1]$ (creencia de que $M^i$ es la correcta), inicializado en $1/k$, y se computa la salida con una **superposición ponderada**:

$$
p(\alpha) = f\Big(x,\; W \odot \textstyle\sum_i \alpha_i M^i\Big)
$$

La intuición: la máscara correcta produce una salida **confiada, de baja entropía**; las equivocadas, salidas inciertas. Por tanto se buscan los $\alpha$ que **minimizan la entropía** $H(p(\alpha))$ y empujan $\alpha$ a una esquina del símplex (un único 1). El algoritmo *One-Shot* infiere la tarea con **un solo gradiente**:

$$
\hat{t} = \arg\max_i\Big(-\frac{\partial H(p(\alpha))}{\partial \alpha_i}\Big)
$$

la coordenada en que la entropía decrece más rápido (un paso de Frank-Wolfe). En el escenario sin fronteras de tarea, si $\nu = \text{softmax}(-\nabla_\alpha H)$ es aproximadamente uniforme ($k\cdot\max_i\nu_i < 1+\epsilon$), se asigna una **máscara nueva**; si no, se usa $\arg\max_i\nu_i$ —así SupSup infiere por sí solo los límites entre tareas.

### V.3 HAT: gating sigmoide con annealing

[Hard Attention to the Task](/papers/hat-serra-2018) (HAT) aprende máscaras **sobre unidades** (no sobre pesos) de forma diferenciable. Para la capa $l$ y tarea $t$, gatea la activación con un embedding de tarea pasado por sigmoide escalada:

$$
h'_l = a^t_l \odot h_l, \qquad a^t_l = \sigma(s\, e^t_l)
$$

El escalar $s>0$ controla la "dureza": $s\to\infty$ da máscara binaria $\{0,1\}$; $s\to 0$ da $1/2$ (todas activas). La gracia es el **annealing**: cada época empieza con $s$ bajo (máxima plasticidad) y lo incrementa linealmente sobre los $B$ batches,

$$
s = \frac{1}{s_{\max}} + \Big(s_{\max} - \frac{1}{s_{\max}}\Big)\frac{b-1}{B-1}
$$

En test se fija $s = s_{\max}\gg 1$ (máscara casi binaria). $s_{\max}$ es el control estabilidad/plasticidad.

**Modulacion del gradiente con mascaras acumuladas.** HAT acumula atenciones con el máximo elementwise, $a^{\le t}_l = \max(a^t_l, a^{\le t-1}_l)$, y modula el gradiente de cada peso por el **reverso del mínimo** de la atención acumulada en las unidades que conecta:

$$
g'_{l,ij} = \Big[\,1 - \min\big(a^{\le t}_{l,i},\; a^{\le t}_{l-1,j}\big)\Big]\, g_{l,ij}
$$

Un peso queda protegido (gradiente $\to 0$) solo si **tanto** la unidad de entrada **como** la de salida fueron importantes en alguna tarea previa —de ahí el mínimo. Esto deriva automáticamente una máscara sobre *pesos* a partir de las máscaras sobre *unidades*. Un término L1 ponderado sobre las atenciones, $L' = L + c\,R(A^t, A^{<t})$, promueve esparsidad para reservar capacidad futura (con $c$ la constante de compresibilidad, y peso $\approx 0$ a las unidades ya usadas, incentivando su reutilización).

---

## Parte VI — Los tres escenarios formalmente

### VI.1 La taxonomia de van de Ven

[Van de Ven y Tolias](/papers/three-scenarios-van-de-ven-2019) organizaron el campo según **qué información de tarea está disponible en test**, no según la arquitectura. Tres escenarios, en dificultad creciente:

| Escenario | Task-ID en test | El modelo debe... | Salida (split MNIST) |
| --- | --- | --- | --- |
| **Task-IL** | Sí (siempre) | Resolver la tarea dada | "Dada la tarea X, ¿clase 1 o 2?" (multi-head) |
| **Domain-IL** | No | Resolver la tarea actual, **sin** identificarla | "¿Es primera o segunda clase?" (par/impar) |
| **Class-IL** | No | Resolver **e inferir** entre todas las clases | "¿Qué dígito (0-9)?" (single-head, todas activas) |

Formalmente, el clasificador debe modelar una distribución sobre etiquetas. En **Task-IL** dispone del task-ID $t$, así que basta modelar $p(y \mid x, t)$ —puede usar una cabeza dedicada por tarea (multi-head), y el problema se restringe a las pocas clases de esa tarea. En **Domain-IL** la estructura de salida es fija (mismas clases lógicas en todas las tareas), modela $p(y\mid x)$ con $y$ en un espacio compartido y constante. En **Class-IL** debe modelar la distribución conjunta sobre **todas las clases vistas sin task-ID**:

$$
p(y \mid x) = \sum_t p(y \mid x, t)\, p(t \mid x)
$$

El término crítico es $p(t\mid x)$: inferir a qué tarea pertenece el input. Class-IL exige aprender la frontera entre clases que **nunca se vieron juntas en un mismo lote** (las clases de la tarea 1 y la tarea 2 jamás co-ocurrieron durante el entrenamiento).

### VI.2 Por que la regularizacion falla en Class-IL

Aquí está el resultado más citado del paper. Los métodos de regularización (EWC, Online EWC, SI) **colapsan al nivel del azar** en Class-IL:

| Método (split MNIST) | Task-IL | Domain-IL | Class-IL |
| --- | --- | --- | --- |
| None (cota inferior) | 87.2% | 59.2% | 19.9% |
| EWC | 98.6% | 64.0% | **20.0%** |
| Online EWC | 99.1% | 64.3% | **20.0%** |
| SI | 99.1% | 65.4% | **20.0%** |
| LwF | 99.6% | 71.5% | 23.9% |
| DGR (replay) | 99.5% | 95.7% | **90.8%** |
| iCaRL (exemplars) | — | — | **94.6%** |

EWC, Online EWC y SI rinden **igual que la cota inferior** (~20% = $1/5$, adivinar entre las 2 clases de la última tarea). La razón es estructural: la regularización solo protege los **pesos** que producen $p(y\mid x, t)$ correctamente *dado* el task-ID. Pero no aporta ninguna señal para estimar $p(t\mid x)$ —nunca optimizó la frontera entre clases de tareas distintas, porque esos pares nunca co-ocurrieron. Un resorte cuadrático sobre los pesos no puede crear una capacidad discriminativa que el gradiente jamás recibió.

{{< concept-alert type="clave" >}}
**Por qué el replay es necesario en Class-IL.** Solo los métodos que *reintroducen datos* de clases viejas —DGR (replay generativo), iCaRL (exemplars), incluso LwF (pseudo-replay vía soft targets)— superan el 90%. Reproducir ejemplos viejos junto a los nuevos hace que, dentro de un mismo lote, co-ocurran clases de tareas distintas; el gradiente *sí* recibe entonces la señal para aprender la frontera $p(t\mid x)$ entre ellas. La regularización no puede sustituir esa señal: la lección operativa de van de Ven es que **en el escenario class-incremental, alguna forma de replay es indispensable**.
{{< /concept-alert >}}

Un matiz honesto: en Task-IL **todos** los métodos funcionan bien (>98%) —el escenario fácil no discrimina—. Los autores incluso rescataron EWC en Task-IL (contra reportes previos que lo daban por fallido) explorando un rango de $\lambda$ varios órdenes de magnitud mayor, porque en split MNIST las tareas son tan fáciles que la Fisher resultante es minúscula.

---

## Sintesis matematica

| Familia | Mecanismo | Ecuacion central |
| --- | --- | --- |
| Olvido (Taylor) | Incremento cuadrático de pérdida vieja | $\Delta\mathcal{L}_\nu \approx \tfrac{1}{2}\Delta\theta^\top H_\nu \Delta\theta$ |
| EWC (regularización) | Resorte por peso vía Fisher | $\mathcal{L}_B + \sum_i \tfrac{\lambda}{2}F_i(\theta_i-\theta^*_{A,i})^2$ |
| SI (regularización) | Importancia por integral de camino | $\omega_k = -\int g_k\,\dot\theta_k\,dt$ |
| LwF (destilación) | Soft targets con temperatura | $y_o'^{(i)} = (y_o^{(i)})^{1/T}/\sum_j(y_o^{(j)})^{1/T}$ |
| iCaRL (exemplars) | Nearest-mean + herding | $y^*=\arg\min_y\|\varphi(x)-\mu_y\|$ |
| GEM (memoria) | Proyección de gradiente (dual) | $\min_v \tfrac12 v^\top GG^\top v + g^\top G^\top v,\; v\ge0$ |
| PiggyBack/SupSup/HAT (arquitectura) | Máscara binaria + straight-through | $y=(W\odot m)x,\; m=\mathbb{1}[m_r\ge\tau]$ |
| Tres escenarios | Descomposición de la salida | $p(y\mid x)=\sum_t p(y\mid x,t)\,p(t\mid x)$ |

El hilo conductor: **el olvido es un incremento cuadrático de la pérdida vieja gobernado por la curvatura $H_\nu$**. Cada familia lo combate de un modo distinto —regularización aproxima $H_\nu$ y penaliza el desplazamiento; replay/exemplars reintroducen los datos que generan el gradiente correcto; arquitectura aísla parámetros dedicados con máscaras—. Y el escenario class-incremental, donde hay que estimar $p(t\mid x)$ sin task-ID, es donde la regularización por sí sola no basta.

---

**Ver tambien:** [Teoria de la clase 32](/clases/clase-32/teoria) · Fundamento: [Aprendizaje Continuo](/fundamentos/aprendizaje-continuo) · Papers: [EWC](/papers/ewc-kirkpatrick-2017) · [Synaptic Intelligence](/papers/synaptic-intelligence-zenke-2017) · [LwF](/papers/lwf-li-2016) · [iCaRL](/papers/icarl-rebuffi-2017) · [GEM](/papers/gem-lopez-paz-2017) · [PiggyBack](/papers/piggyback-mallya-2018) · [SupSup](/papers/supsup-wortsman-2020) · [HAT](/papers/hat-serra-2018) · [Tres escenarios](/papers/three-scenarios-van-de-ven-2019).
