# Proximal Policy Optimization Algorithms — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Proximal Policy Optimization Algorithms*.
- **Autores:** John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (todos de **OpenAI**).
- **Venue:** Preprint de arXiv (nunca pasó por revisión formal en conferencia, pero es uno de los papers de RL más citados e influyentes de la década).
- **Año:** 2017. **Preprint:** arXiv:1707.06347v2 (28 ago 2017), [arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347).
- **Una línea:** propone PPO, una familia de métodos *policy gradient* que alcanza la estabilidad de TRPO usando **solo optimización de primer orden** y un objetivo *surrogate* recortado (*clipped*), simple de implementar y que permite varias épocas de SGD sobre el mismo lote de datos.

PPO es, en esencia, una respuesta de ingeniería a un dilema que arrastraba el RL basado en políticas hacia 2017: los métodos *vanilla policy gradient* son simples pero frágiles (un paso de actualización demasiado grande puede destruir la política aprendida), mientras que TRPO (Trust Region Policy Optimization, Schulman et al. 2015) es estable pero **complejo** —requiere optimización de segundo orden con gradiente conjugado y una restricción dura sobre la divergencia KL— y además es incompatible con arquitecturas que comparten parámetros entre política y función de valor, o que usan ruido como *dropout*. PPO busca "lo mejor de ambos mundos": la fiabilidad y eficiencia de muestras de TRPO, pero "mucho más simple de implementar, más general, y con mejor complejidad de muestras (empíricamente)", según las palabras del abstract.

El truco central es un **objetivo recortado** que, en vez de imponer una restricción explícita de confianza, *desincentiva* que la nueva política se aleje demasiado de la vieja directamente dentro de la función de pérdida. Eso convierte el problema de optimización restringida de segundo orden de TRPO en un problema no restringido de primer orden que se resuelve con Adam o SGD ordinario —"solo unas pocas líneas de cambio respecto a una implementación de policy gradient vanilla". El resultado fue adoptado masivamente: hoy PPO es, posiblemente, el algoritmo de RL profundo más usado en la práctica, y es la pieza de optimización detrás del **RLHF** que afinó ChatGPT e InstructGPT.

Para la **Clase 31 (Aprendizaje Reforzado)**, PPO representa el estado del arte práctico de los métodos *policy-based*, complementando la línea *value-based* de la clase (Q-learning → DQN y sus variantes). Es además el eslabón que conecta el RL "clásico" de control con el RL aplicado a modelos de lenguaje que se vio en la Clase 20.

## 2. Contexto: el dilema de los métodos policy gradient

### 2.1. Vanilla policy gradient: simple pero inestable

Los métodos *policy gradient* optimizan directamente una política estocástica $\pi_\theta(a \mid s)$ ascendiendo por el gradiente de la recompensa esperada. El estimador de gradiente más común tiene la forma

$$\hat{g} = \hat{\mathbb{E}}_t\!\left[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\,\hat{A}_t\right],$$

donde $\hat{A}_t$ es un estimador de la **función de ventaja** (advantage) en el paso $t$ —cuánto mejor que el promedio fue tomar la acción $a_t$ en el estado $s_t$. En las implementaciones con autodiferenciación esto se obtiene construyendo el objetivo $L^{PG}(\theta) = \hat{\mathbb{E}}_t[\log \pi_\theta(a_t \mid s_t)\,\hat{A}_t]$ y derivándolo.

El problema que el paper diagnostica explícitamente: es tentador hacer **varios pasos de optimización sobre la misma trayectoria** $L^{PG}$ para exprimir los datos, pero "hacerlo no está bien justificado, y empíricamente suele llevar a actualizaciones de política destructivamente grandes". Es decir, el estimador es válido solo *localmente*, alrededor de la política que generó los datos; insistir con muchos pasos de gradiente sobre el mismo lote empuja la política lejos de esa región de validez y la colapsa. Por eso el vanilla policy gradient gasta un gradiente por muestra y luego recolecta datos nuevos: tiene **mala eficiencia de muestras** y poca robustez.

### 2.2. TRPO: estable pero pesado de segundo orden

TRPO ataca la inestabilidad limitando *cuánto* puede cambiar la política en cada actualización. Maximiza el objetivo *surrogate*

$$\underset{\theta}{\text{maximizar}}\;\; \hat{\mathbb{E}}_t\!\left[\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}\,\hat{A}_t\right] \quad\text{sujeto a}\quad \hat{\mathbb{E}}_t\!\left[\mathrm{KL}[\pi_{\theta_{old}}(\cdot \mid s_t),\, \pi_\theta(\cdot \mid s_t)]\right] \le \delta.$$

La **restricción de KL** define una "región de confianza" (*trust region*): la nueva política no puede alejarse de la vieja más de $\delta$ en divergencia KL. Eso garantiza que el *surrogate* siga siendo una aproximación fiable y produce mejoras casi monótonas. El precio es la complejidad: el problema se resuelve aproximadamente con el **algoritmo de gradiente conjugado**, tras una aproximación lineal del objetivo y **cuadrática de la restricción** —es decir, necesita información de segundo orden (productos con la matriz de Fisher). El paper observa además que TRPO "es relativamente complicado y no es compatible con arquitecturas que incluyen ruido (como dropout) o compartición de parámetros".

La teoría de TRPO sugería en realidad usar una **penalización** en vez de una restricción dura:

$$\underset{\theta}{\text{maximizar}}\;\; \hat{\mathbb{E}}_t\!\left[\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}\,\hat{A}_t - \beta\,\mathrm{KL}[\pi_{\theta_{old}}, \pi_\theta]\right],$$

pero TRPO prefiere la restricción porque es difícil elegir un único $\beta$ que funcione bien en todos los problemas —o incluso a lo largo de un mismo problema, cuyas características cambian durante el aprendizaje. Este detalle es importante: PPO retomará ambas ideas (penalización y restricción) y propondrá alternativas más manejables.

## 3. Contribución central: el clipped surrogate objective

La aportación clave de PPO es reemplazar la restricción de segundo orden de TRPO por un **recorte (clipping)** dentro del objetivo, que se optimiza con gradiente de primer orden. Sea

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$$

el **ratio de probabilidades** entre la política nueva y la vieja para la acción tomada. Nótese que $r_t(\theta_{old}) = 1$ por construcción. El objetivo *surrogate* de TRPO/CPI (de *conservative policy iteration*, Kakade & Langford 2002) es simplemente $L^{CPI}(\theta) = \hat{\mathbb{E}}_t[r_t(\theta)\,\hat{A}_t]$. Sin restricción, maximizar esto lleva a actualizaciones excesivamente grandes. PPO propone en cambio:

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\!\left[\min\!\Big(r_t(\theta)\,\hat{A}_t,\;\; \mathrm{clip}(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon)\,\hat{A}_t\Big)\right],$$

donde $\epsilon$ es un hiperparámetro (típicamente $\epsilon = 0.2$). La mecánica del objetivo:

- **El primer término** dentro del `min` es $L^{CPI} = r_t\,\hat{A}_t$, el objetivo sin recortar.
- **El segundo término** recorta el ratio al intervalo $[1-\epsilon,\, 1+\epsilon]$, lo que **elimina el incentivo a mover $r_t$ fuera de esa banda**.
- **Tomar el mínimo** de ambos hace que el objetivo final sea una **cota inferior pesimista** del objetivo sin recortar. La consecuencia fina: solo se *ignora* el cambio del ratio cuando este haría *mejorar* el objetivo; cuando lo empeora, se incluye. Dicho de otro modo, el recorte impide ganancias "baratas" por alejarse demasiado, pero no protege de empeoramientos.

La intuición geométrica que da el paper (su Figura 1): para una **ventaja positiva** ($\hat{A}_t > 0$, la acción fue buena, queremos subir su probabilidad) el objetivo crece con $r_t$ pero se **aplana** una vez que $r_t > 1+\epsilon$ —ya no hay recompensa por seguir subiendo la probabilidad más allá del recorte. Para una **ventaja negativa** ($\hat{A}_t < 0$, la acción fue mala) el objetivo se aplana cuando $r_t < 1-\epsilon$. En ambos casos, el aplanamiento le quita al optimizador el incentivo de dar pasos enormes, y por eso varios pasos de SGD sobre el mismo lote ya no destruyen la política. Es importante notar que $L^{CLIP}(\theta) = L^{CPI}(\theta)$ a primer orden alrededor de $\theta_{old}$ (donde $r=1$); los objetivos solo divergen cuando $\theta$ se aleja.

Esta es la razón del nombre **proximal**: las nuevas políticas se mantienen "próximas" a las viejas, no por una restricción explícita, sino porque el objetivo deja de premiar el alejamiento. El beneficio práctico decisivo es que esto **habilita múltiples épocas de SGD por minibatch sobre el mismo lote de datos**, justo lo que el vanilla policy gradient no podía hacer sin colapsar —y de ahí viene buena parte de la mejor eficiencia de muestras.

## 4. Método: las variantes y el algoritmo completo

### 4.1. Variante con penalización KL adaptativa

Como alternativa (o complemento) al recorte, el paper presenta una segunda variante que sí usa la penalización KL, pero **adaptando el coeficiente $\beta$** para alcanzar un valor objetivo $d_{targ}$ de divergencia KL en cada actualización. En cada paso de política:

1. Optimizar con varias épocas de SGD el objetivo penalizado $L^{KLPEN}(\theta) = \hat{\mathbb{E}}_t[r_t(\theta)\,\hat{A}_t - \beta\,\mathrm{KL}[\pi_{\theta_{old}}, \pi_\theta]]$.
2. Medir $d = \hat{\mathbb{E}}_t[\mathrm{KL}[\pi_{\theta_{old}}, \pi_\theta]]$ y ajustar:
   - si $d < d_{targ}/1.5$, entonces $\beta \leftarrow \beta/2$ (la KL quedó chica, relajar la penalización);
   - si $d > d_{targ}\times 1.5$, entonces $\beta \leftarrow \beta\times 2$ (la KL se pasó, endurecer).

El $\beta$ ajustado se usa en la siguiente actualización. Esto resuelve precisamente el problema que llevó a TRPO a rechazar la penalización fija: $\beta$ ya no es un hiperparámetro delicado, porque el algoritmo lo corrige solo. Las constantes 1.5 y 2 son heurísticas y el método es poco sensible a ellas. El paper la incluye como **línea base importante**, pero en sus experimentos esta variante **rinde peor que el recorte**, por lo que el recorte es la versión recomendada.

### 4.2. Estimación de la ventaja con GAE

El estimador $\hat{A}_t$ no es trivial: hay que estimar la ventaja con poca varianza pero poco sesgo. PPO usa la **Generalized Advantage Estimation (GAE)** (Schulman et al. 2015a), una versión truncada que combina los errores temporales de Bellman:

$$\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \cdots + (\gamma\lambda)^{T-t+1}\delta_{T-1}, \qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t),$$

donde $V$ es una función de valor de estado aprendida, $\gamma$ es el factor de descuento y $\lambda$ controla el balance sesgo–varianza (con $\lambda=1$ se recupera la estimación tipo Monte Carlo de Mnih et al. 2016). Esto requiere correr la política $T$ pasos (con $T$ mucho menor que la longitud del episodio) y usar el segmento de trayectoria para la actualización —un estilo bien adaptado a recolección paralela.

### 4.3. El objetivo conjunto actor-critic

Cuando la red **comparte parámetros entre política (actor) y función de valor (critic)** —algo que TRPO no soportaba bien— hay que combinar tres términos en una sola pérdida que se maximiza cada iteración:

$$L_t^{CLIP+VF+S}(\theta) = \hat{\mathbb{E}}_t\!\left[L_t^{CLIP}(\theta) - c_1\,L_t^{VF}(\theta) + c_2\,S[\pi_\theta](s_t)\right],$$

donde $L_t^{VF} = (V_\theta(s_t) - V_t^{targ})^2$ es el **error cuadrático de la función de valor**, $S$ es un **bono de entropía** que fomenta la exploración (evita que la política colapse prematuramente a determinismo), y $c_1, c_2$ son coeficientes de ponderación.

### 4.4. Algoritmo 1 — PPO estilo actor-critic

El bucle completo, con recolección paralela:

- En cada iteración, cada uno de $N$ **actores paralelos** corre la política $\pi_{\theta_{old}}$ durante $T$ pasos en el entorno y computa estimaciones de ventaja $\hat{A}_1, \dots, \hat{A}_T$.
- Se construye la pérdida *surrogate* sobre los $N \times T$ pasos recolectados y se optimiza con **minibatch SGD (o Adam, normalmente mejor)** durante **$K$ épocas**, con minibatch de tamaño $M \le NT$.
- Se actualiza $\theta_{old} \leftarrow \theta$ y se repite.

Hiperparámetros típicos de MuJoCo (Tabla 3 del paper): horizonte $T=2048$, paso de Adam $3\times10^{-4}$, $K=10$ épocas, minibatch 64, $\gamma=0.99$, $\lambda_{GAE}=0.95$. En Atari (Tabla 5): $T=128$, $K=3$ épocas, 8 actores, $\epsilon=0.1\times\alpha$ con $\alpha$ recocido linealmente de 1 a 0, más $c_1=1$ (valor) y $c_2=0.01$ (entropía).

## 5. Experimentos

### 5.1. Comparación de objetivos surrogate (MuJoCo)

El paper primero contrasta las variantes del *surrogate* en **7 tareas de control continuo de robótica simulada** (HalfCheetah, Hopper, doble péndulo invertido, péndulo invertido, Reacher, Swimmer, Walker2d) sobre el motor de física **MuJoCo** en OpenAI Gym, con 1 millón de pasos por tarea, normalizando los puntajes (política aleatoria = 0, mejor resultado = 1). Resultados (Tabla 1, puntaje normalizado promedio sobre 21 corridas):

| Variante | Puntaje |
|---|---|
| Sin recorte ni penalización | −0.39 |
| Recorte, $\epsilon=0.1$ | 0.76 |
| **Recorte, $\epsilon=0.2$** | **0.82** |
| Recorte, $\epsilon=0.3$ | 0.70 |
| KL adaptativa, $d_{targ}=0.01$ | 0.74 |
| KL fija, $\beta=1$ | 0.71 |

El veredicto es claro: **el recorte con $\epsilon=0.2$ gana**, la KL adaptativa queda en segundo lugar, y la versión sin ninguna protección colapsa (puntaje negativo, peor que la política aleatoria, porque en HalfCheetah diverge catastróficamente). Esto valida empíricamente que el mecanismo de recorte es el que importa.

### 5.2. Comparación contra otros algoritmos (control continuo)

PPO (con recorte, $\epsilon=0.2$) se compara contra TRPO, el método de entropía cruzada (CEM), policy gradient vanilla con paso adaptativo, **A2C** (Advantage Actor Critic, la versión síncrona de A3C) y A2C con región de confianza. Sobre las mismas tareas MuJoCo de 1 millón de pasos (Figura 3), "PPO supera a los métodos previos en casi todos los entornos de control continuo". Es decir, no solo iguala a TRPO siendo mucho más simple, sino que típicamente lo supera.

### 5.3. Showcase: humanoide 3D

Para exhibir PPO en problemas de alta dimensión, lo entrenan en tareas de un **humanoide 3D** con Roboschool: correr, dirigirse a un objetivo que cambia de posición (Flagrun) y una versión "harder" donde al robot lo bombardean con cubos y debe levantarse del suelo. Las curvas de aprendizaje (Figura 4) muestran que PPO aprende políticas de locomoción complejas y robustas.

### 5.4. Atari

En el **Arcade Learning Environment** (49 juegos de Atari) PPO se compara contra A2C y **ACER** (un método actor-critic con *experience replay*, más sofisticado). Con dos métricas de puntuación —recompensa promedio durante todo el entrenamiento (favorece aprendizaje rápido) y sobre los últimos 100 episodios (favorece desempeño final)— el conteo de juegos "ganados" (Tabla 2):

| Métrica | A2C | ACER | PPO | Empate |
|---|---|---|---|---|
| (1) recompensa durante todo el entrenamiento | 1 | 18 | **30** | 0 |
| (2) recompensa últimos 100 episodios | 1 | 28 | 19 | 1 |

PPO domina claramente en **velocidad de aprendizaje / eficiencia de muestras** (métrica 1), superando incluso a ACER que es mucho más complejo; en desempeño final asintótico (métrica 2) ACER lo supera, pero PPO sigue muy por encima de A2C. El resumen del paper: PPO logra un balance favorable entre complejidad de muestras, **simplicidad** y *wall-time* (tiempo real de cómputo).

## 6. Limitaciones

- **El recorte no garantiza monotonía teórica.** A diferencia de TRPO, que tiene garantías de mejora casi monótona derivadas de su cota inferior, PPO sustituye esa teoría por una heurística (el `min` recortado) que *en la práctica* funciona muy bien, pero sin la misma fundamentación formal. El propio paper la presenta como una aproximación pragmática.
- **Sensibilidad a hiperparámetros y detalles de implementación.** Aunque $\epsilon$ es robusto en el rango $0.1$–$0.3$, el rendimiento depende de muchos detalles (normalización de ventajas, recorte del gradiente, recocido del paso de aprendizaje, número de épocas). Trabajos posteriores mostrarían que buena parte del éxito de PPO viene de estos "trucos de código" además del objetivo en sí.
- **La variante KL adaptativa rinde peor** que el recorte, según los propios experimentos; se incluye como base, no como recomendación.
- **Sigue siendo on-policy.** PPO descarta los datos tras unas pocas épocas (no usa un *replay buffer* persistente como DQN o ACER), lo que limita la reutilización de experiencia frente a métodos *off-policy* en regímenes de muestras muy escasas.
- **El recorte puede saturar.** Una vez que el ratio queda fuera de la banda, el gradiente de ese término se anula; esto estabiliza, pero también puede frenar el aprendizaje si grandes fracciones del lote quedan recortadas.

## 7. Impacto

PPO se convirtió, con los años, **posiblemente en el algoritmo de RL profundo más usado en la práctica**, precisamente por la propiedad que el paper vendía: combina robustez con simplicidad de implementación. Esa combinación lo hizo el caballo de batalla por defecto en bibliotecas (OpenAI Baselines, Stable-Baselines, RLlib), en robótica simulada, en videojuegos (fue la base de OpenAI Five para Dota 2) y en investigación general.

El impacto que más conecta con este curso es otro: PPO es el **algoritmo de optimización detrás del RLHF** (Reinforcement Learning from Human Feedback) que afinó **InstructGPT** y **ChatGPT**. En ese contexto, la "política" es el modelo de lenguaje, la "acción" es generar el siguiente token, y la "recompensa" la entrega un *reward model* entrenado con preferencias humanas; PPO ajusta el modelo para maximizar esa recompensa **manteniéndose próximo** al modelo original (un término KL contra la política de referencia cumple el rol de evitar que el modelo "se rompa" optimizando en exceso). La misma idea de "no alejarse demasiado de la política previa" que motivó el recorte de 2017 reaparece, una década después, como el mecanismo central que hizo viable alinear modelos de lenguaje gigantes. Ver [/fundamentos/rlhf](/fundamentos/rlhf).

## 8. Conexión con la Clase 31 (Aprendizaje Reforzado)

La Clase 31 recorre el RL en dos grandes familias. La línea **value-based** —Q-learning tabular, y su salto a redes profundas con DQN ([/papers/mnih-dqn-nature-2015](/papers/mnih-dqn-nature-2015)) y variantes (Double DQN, Dueling, PER)— aprende una función de valor y deriva la política de ella. PPO pertenece a la línea complementaria, **policy-based / actor-critic**, que optimiza la política directamente y es la opción natural cuando el espacio de acciones es **continuo** (control robótico), donde tomar el `argmax` de una Q-función es inviable. Por eso PPO y DQN no compiten tanto como se complementan: cubren regímenes distintos del problema de control.

Mapeo concreto con la clase:

- **Es el estado del arte práctico de los métodos policy-based.** Donde DQN es la referencia *value-based*, PPO es la referencia *policy gradient*. Entender ambos da el panorama completo del RL profundo moderno.
- **Cierra el linaje de policy gradient** que la clase introduce: REINFORCE (gradiente con alta varianza) → actor-critic con ventaja (A2C/A3C, [/papers/a3c-mnih-2016](/papers/a3c-mnih-2016)) → región de confianza (TRPO) → recorte de primer orden (PPO). PPO es el destino pragmático de esa evolución: la estabilidad de TRPO sin su maquinaria de segundo orden.
- **Conecta con la Clase 20 (RLHF y alineamiento de LLMs).** PPO es el puente literal entre el RL de control de esta clase y el ajuste de modelos de lenguaje que se vio antes: el mismo algoritmo, aplicado a un "entorno" donde las acciones son tokens y la recompensa viene de preferencias humanas.

Lecturas relacionadas dentro del dominio: [/fundamentos/aprendizaje-reforzado](/fundamentos/aprendizaje-reforzado) para los fundamentos de MDP, política, valor y ventaja; [/clases/clase-31](/clases/clase-31) para el hub de la clase; [/papers/a3c-mnih-2016](/papers/a3c-mnih-2016) para el antecedente actor-critic directo de PPO; y [/fundamentos/rlhf](/fundamentos/rlhf) para la aplicación que volvió a PPO célebre fuera del control.
