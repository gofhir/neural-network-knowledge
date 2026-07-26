---
title: "Profundización - Razonamiento"
weight: 20
math: true
---

> **Desarrollo formal de la Clase 34.** La [teoría](/clases/clase-34/teoria) recorre el panorama de forma narrativa; aquí se formaliza. Cinco partes: (1) la jerarquía causal de Pearl; (2) el razonamiento por prompting (scratchpad, CoT, self-consistency, ToT); (3) el cómputo en tiempo de inferencia (Pass@k, leyes de cobertura); (4) el RL para razonamiento (GRPO) y la crítica de Yue; (5) la formalización de la inteligencia de Chollet (ARC).

---

## 1. La jerarquía causal

Pearl ordena el razonamiento en tres niveles, distinguidos por el operador que involucran:

1. **Asociación:** $P(Y \mid X)$ — probabilidad de $Y$ **dado que observo** $X$. Todo el aprendizaje supervisado.
2. **Intervención:** $P(Y \mid \text{do}(X{=}x))$ — probabilidad de $Y$ **si fijo** $X=x$ interviniendo el sistema. En general $P(Y\mid \text{do}(X)) \neq P(Y\mid X)$: observar que sube el barómetro no es lo mismo que **forzarlo** a subir.
3. **Contrafactual:** $P(Y_{x'} \mid X{=}x, Y{=}y)$ — dado que ocurrió $(x,y)$, ¿qué habría pasado con $x'$? Requiere un modelo causal completo (structural causal model).

Un modelo puramente asociativo —un LLM entrenado por máxima verosimilitud— no puede, en general, responder preguntas de los niveles 2 y 3 sin un modelo causal subyacente. Esta es la limitación de fondo que motiva toda la clase.

---

## 2. Razonamiento por prompting

### 2.1 Chain-of-Thought

Sea un LLM $p_\theta$ y una tarea con entrada $x$ y respuesta $y$. El **prompting estándar** estima $p_\theta(y \mid x)$ directamente. El **Chain-of-Thought** introduce una **cadena de razonamiento** $c$ (una secuencia de tokens intermedios) como variable explícita:

$$
p_\theta(y \mid x) = \sum_{c} p_\theta(y \mid c, x)\, p_\theta(c \mid x).
$$

En la práctica no se marginaliza: se **muestrea greedy** una cadena $\hat c \sim p_\theta(c\mid x)$ y luego $\hat y = \arg\max_y p_\theta(y \mid \hat c, x)$. El few-shot CoT condiciona con $k$ ejemplos $\langle x_i, c_i, y_i\rangle$ para inducir el formato. La ventaja computacional: generar $c$ da al modelo tokens intermedios donde "desplegar" cómputo que un solo forward pass no permite.

### 2.2 Self-consistency: marginalizar por muestreo

Wang et al. (2022) **sí** aproximan la marginalización, muestreando $m$ cadenas y agregando por **voto mayoritario** sobre la respuesta:

$$
\hat y = \arg\max_{y} \sum_{i=1}^{m} \mathbb{1}\!\left[\,y_i = y\,\right], \qquad (c_i, y_i) \sim p_\theta(\cdot \mid x).
$$

Esto es un estimador de moda de la distribución marginal $p_\theta(y\mid x)$ inducida sobre los caminos de razonamiento. Reduce la varianza del decoding greedy (que es frágil: un error temprano arruina la cadena).

### 2.3 Tree-of-Thoughts: búsqueda

Yao et al. (2023) generalizan el estado a un prefijo de pensamientos $s = [x, c_{1..i}]$ y definen tres operadores:

- **Generador** $G(p_\theta, s, k)$: propone $k$ pensamientos siguientes candidatos.
- **Evaluador** $V(p_\theta, S)$: puntúa estados (mediante *value* con etiquetas tipo sure/maybe/impossible, o por *vote*).
- **Búsqueda**: BFS o DFS sobre el árbol, con **poda** y **backtracking**.

CoT es el caso degenerado de un árbol de anchura 1 sin backtracking. El costo de inferencia crece con el factor de ramificación y la profundidad —el precio del Sistema 2.

---

## 3. Cómputo en tiempo de inferencia

### 3.1 Pass@k y cobertura

La métrica **Pass@k** es la probabilidad de que **al menos una** de $k$ muestras sea correcta. Con $n \ge k$ muestras de las que $c$ son correctas, el estimador insesgado (Chen et al., 2021) es:

$$
\text{pass@}k = \mathbb{E}\left[\,1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\,\right].
$$

Brown et al. (2024) miden la **cobertura** $c(k)$ —fracción de problemas resueltos por alguna de $k$ muestras— y encuentran que crece de forma casi log-lineal sobre cuatro órdenes de magnitud, ajustable por una ley de potencia exponenciada:

$$
c(k) \approx \exp\!\big(a\, k^{b}\big).
$$

{{< concept-alert type="advertencia" >}}
La cobertura es un **límite superior** de lo alcanzable: presupone un **selector perfecto** que identifica la muestra correcta. En dominios **verificables** (código con tests, matemáticas con checker) ese selector existe; en **no verificables**, la selección por voto/reward model se **satura** muy por debajo de la cobertura. La brecha cobertura–selección es la medida de cuánto falta un buen verificador.
{{< /concept-alert >}}

### 3.2 GRPO: RL para razonamiento

DeepSeek-R1 optimiza la política con **Group Relative Policy Optimization (GRPO)**, una variante de [PPO](/papers/ppo-schulman-2017) que **elimina la red de valor**. Para cada pregunta se muestrea un grupo de $G$ respuestas $\{o_1,\dots,o_G\}$ con recompensas $\{r_1,\dots,r_G\}$; la **ventaja** de cada una se estima por normalización dentro del grupo:

$$
A_i = \frac{r_i - \operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G)}.
$$

El objetivo es el surrogate recortado de PPO con una penalización KL contra un modelo de referencia:

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^{G} \min\!\Big(\rho_i A_i,\; \operatorname{clip}(\rho_i, 1{-}\varepsilon, 1{+}\varepsilon)A_i\Big) - \beta\, D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})\right],
$$

donde $\rho_i = \pi_\theta(o_i)/\pi_{\theta_{\text{old}}}(o_i)$. La recompensa es **basada en reglas**: $r = r_{\text{accuracy}} + r_{\text{format}}$, con $r_{\text{accuracy}}$ verificando el resultado (matemáticas/código) y $r_{\text{format}}$ premiando la estructura (p.ej. encerrar el razonamiento en etiquetas). Al no usar un modelo de recompensa aprendido, evita el *reward hacking* de la recompensa neuronal —a cambio de aplicar solo a dominios verificables. Contrastar con [RLHF](/fundamentos/rlhf), que sí usa un modelo de recompensa de preferencias humanas.

### 3.3 La crítica de Yue: ¿expandir o reordenar?

Yue et al. (2025) usan Pass@k para preguntar si el RL **expande** la capacidad de razonamiento. Observan un **cruce de curvas**: llamando $\pi_{\text{RL}}$ al modelo entrenado con RL verificable y $\pi_{\text{base}}$ al modelo base,

$$
\text{pass@}1(\pi_{\text{RL}}) > \text{pass@}1(\pi_{\text{base}}), \qquad \text{pero} \qquad \lim_{k \to \text{grande}} \text{pass@}k(\pi_{\text{base}}) \gtrsim \text{pass@}k(\pi_{\text{RL}}).
$$

Interpretación: el RL **estrecha** la distribución de salida hacia los caminos correctos que el modelo base **ya podía muestrear** (mejora la eficiencia por muestra), pero **no genera** caminos de razonamiento nuevos fuera del soporte del base —incluso puede **reducir** la cobertura a $k$ grande. La capacidad de razonamiento reside, entonces, en el **pre-entrenamiento**; la **destilación** sí puede expandir la frontera al inyectar caminos de un modelo más capaz.

---

## 4. La formalización de la inteligencia (Chollet)

Chollet (2019) critica medir inteligencia por **habilidad en una tarea** (comprable con datos y cómputo) y propone medir la **eficiencia en la adquisición de habilidades** relativa a los priors y la experiencia. Esquemáticamente, la inteligencia de un sistema es una esperanza, sobre un espacio de tareas, de la habilidad alcanzada **ponderada por la generalización requerida** y **dividida por** los priors y la experiencia consumidos:

$$
\text{Inteligencia} \;\propto\; \mathbb{E}_{\text{tareas}}\!\left[\frac{\text{generalización} \cdot \text{habilidad}}{\text{priors} + \text{experiencia}}\right].
$$

Un sistema que alcanza alta habilidad **consumiendo enormes datos/priors** (un LLM entrenado en todo internet) puntúa **bajo** en esta medida: no razona, memoriza. El **ARC** (Abstraction and Reasoning Corpus) materializa el test: tareas few-shot de transformación de grillas, generadas a mano para resistir la memorización, que solo asumen **Core Knowledge priors** (objetividad, numerosidad, geometría/topología, agentividad). ARC mide justamente **abstracción y sistematicidad** —los cimientos del razonamiento que abre la clase. → [paper](/papers/arc-chollet-2019)

---

## 5. Síntesis

El arco de la clase, en una línea: el razonamiento exige **abstracción** (olvidar diferencias) y **sistematicidad** (componer y generalizar), capacidades que el deep learning asociativo aproxima de forma incompleta. Las dos estrategias para cerrar la brecha son **estructura** (memoria externa, MAC) y **cómputo de inferencia** (CoT → self-consistency → ToT → RL verificable). Todas mejoran el razonamiento aparente, pero ninguna garantiza razonamiento correcto: la advertencia de Ye & Durrett y la crítica de Yue delimitan qué es real y qué es reordenamiento de lo que el pre-entrenamiento ya contenía.

---

**Ver también:** [Clase 34 - Teoría](/clases/clase-34/teoria) · [Clase 34 - Práctica](/clases/clase-34/practica) · Fundamentos: [Razonamiento en IA](/fundamentos/razonamiento) · [Chain-of-Thought](/fundamentos/chain-of-thought) · [Test-time compute](/fundamentos/test-time-compute).
