---
title: "Profundización - Imitación e IRL"
weight: 20
math: true
---

> **Desarrollo matemático de la Clase 33.** La [teoría](/clases/clase-33/teoria) recorre las cuatro ideas de forma narrativa; aquí se formalizan. Cinco partes: (1) el marco formal común (MDP, retorno, ocupación); (2) el aprendizaje reforzado inverso, de Ng-Russell a MaxEnt; (3) el aprendizaje por imitación, del behavioral cloning a DAgger con sus cotas; (4) GAIL, el puente adversario entre IRL e imitación; (5) la formalización de la generalización en RL.

---

## 1. Marco formal común

Trabajamos sobre un **proceso de decisión de Markov** (MDP) $M = \langle S, A, P, r, \gamma, \mu_0\rangle$: estados $S$, acciones $A$, dinámica $P(s'\mid s,a)$, recompensa $r(s)$ (o $r(s,a)$), descuento $\gamma \in [0,1)$ y distribución inicial $\mu_0$. Una **política** $\pi(a\mid s)$ induce trayectorias $\zeta = (s_0, a_0, s_1, a_1, \dots)$. El **retorno esperado descontado** de $\pi$ bajo la recompensa $r$ es

$$
\eta_r(\pi) = \mathbb{E}_{\pi}\!\left[\sum_{t=0}^{\infty} \gamma^t r(s_t)\right].
$$

Dos objetos serán protagonistas. Primero, si la recompensa es **lineal en features** $\phi : S \to \mathbb{R}^k$, es decir $r(s) = w^\top \phi(s)$, el retorno se factoriza:

$$
\eta_r(\pi) = w^\top \underbrace{\mathbb{E}_\pi\!\left[\sum_{t=0}^\infty \gamma^t \phi(s_t)\right]}_{\mu(\pi)\ \text{(feature expectations)}}.
$$

Segundo, la **medida de ocupación** estado-acción $\rho_\pi(s,a) = \sum_{t=0}^\infty \gamma^t P(s_t = s, a_t = a \mid \pi)$, que cumple $\eta_r(\pi) = \sum_{s,a}\rho_\pi(s,a)\, r(s,a)$. Existe una correspondencia **biunívoca** entre políticas y medidas de ocupación válidas: esta es la clave técnica de GAIL.

---

## 2. Aprendizaje reforzado inverso

### 2.1 Ng & Russell (2000): el problema y su degeneración

Dado un experto óptimo $\pi^*$, buscamos $r$ tal que $\pi^*$ sea óptima. Para un MDP finito con política $\pi^*(s) \equiv a_1$, la condición de optimalidad de Bellman se puede escribir en forma matricial: $\pi^*$ es óptima bajo $R$ si y solo si

$$
(P_{a_1} - P_a)\,(I - \gamma P_{a_1})^{-1} R \;\succeq\; 0 \qquad \forall\, a \in A,
$$

donde $P_a$ es la matriz de transición de la acción $a$ y $R$ el vector de recompensas por estado. El problema es que este sistema tiene la **solución trivial** $R = 0$ (y muchas otras): está **mal planteado**. Ng y Russell lo resuelven maximizando el **margen** con el que las acciones del experto superan a las alternativas, más una **regularización** $L_1$ que favorece recompensas simples:

$$
\max_R \; \sum_{s} \Big( \min_{a \neq a_1} \big\{ (P_{a_1}(s) - P_a(s))(I-\gamma P_{a_1})^{-1} R \big\} \Big) - \lambda \lVert R \rVert_1,
$$

un **programa lineal** resoluble de forma estándar. El paper extiende esto a espacios grandes vía aproximación lineal $R = \alpha^\top\phi$ y, cuando solo se observan trayectorias, estima los valores por **Monte Carlo** y genera restricciones de forma incremental.

### 2.2 Abbeel & Ng (2004): igualar feature expectations

En vez de recuperar $w^*$, basta **igualar las feature expectations**. La observación fundamental es una desigualdad de Cauchy-Schwarz: si $\lVert \mu(\tilde\pi) - \mu_E \rVert_2 \le \epsilon$, entonces para *cualquier* peso verdadero con $\lVert w^*\rVert_1 \le 1$,

$$
\big| \eta_{r^*}(\tilde\pi) - \eta_{r^*}(\pi_E) \big| = \big| w^{*\top}(\mu(\tilde\pi) - \mu_E) \big| \le \lVert w^*\rVert_2 \, \lVert \mu(\tilde\pi) - \mu_E\rVert_2 \le \epsilon.
$$

Es decir, **igualar las features garantiza igualar el desempeño**, sin conocer la recompensa verdadera. El algoritmo alterna: (i) un paso **max-margin** tipo SVM que busca el $w$ que más separa $\mu_E$ de las feature expectations de las políticas ya generadas, y (ii) un paso de **RL forward** que produce la política óptima bajo ese $w$; su $\mu$ se añade al conjunto y se repite.

### 2.3 Ziebart et al. (2008): máxima entropía

La ambigüedad persiste: muchas políticas igualan las features. MaxEnt IRL la elimina eligiendo, entre todas las distribuciones sobre trayectorias que igualan las features, **la de máxima entropía** —la menos comprometida. Esto produce una distribución exponencial (Boltzmann):

$$
P(\zeta \mid w) = \frac{1}{Z(w)}\exp\!\big(w^\top f_\zeta\big), \qquad f_\zeta = \sum_{s_t \in \zeta} \phi(s_t),
$$

con $Z(w)$ la función de partición. Los pesos se ajustan por **máxima verosimilitud**; su gradiente iguala las features empíricas con las esperadas:

$$
\nabla_w \mathcal{L} = \tilde f - \sum_\zeta P(\zeta\mid w)\, f_\zeta = \tilde f - \sum_s D_s\, \phi(s),
$$

donde las **frecuencias esperadas de visita** $D_s$ se calculan con un algoritmo *forward-backward* de programación dinámica. MaxEnt tolera expertos sub-óptimos (no exige optimalidad, solo que el comportamiento sea probable) y es la base del IRL profundo moderno.

---

## 3. Aprendizaje por imitación

### 3.1 Behavioral cloning y su cota cuadrática

El behavioral cloning entrena $\hat\pi$ minimizando la pérdida de clasificación sobre la distribución de estados del **experto** $d_{\pi^*}$:

$$
\hat\pi = \arg\min_\pi \; \mathbb{E}_{s \sim d_{\pi^*}}\big[\ell(s, \pi)\big], \qquad \mathbb{E}_{s\sim d_{\pi^*}}[\ell(s,\hat\pi)] = \epsilon.
$$

El problema es que en ejecución la política visita $d_{\hat\pi}$, **no** $d_{\pi^*}$. Ross et al. (2011) prueban que, en el peor caso, el costo total $J(\hat\pi)$ sobre un horizonte $T$ se degrada **cuadráticamente**:

$$
J(\hat\pi) \le J(\pi^*) + \mathcal{O}(T^2 \epsilon).
$$

La intuición: un error de probabilidad $\epsilon$ lleva a un estado nuevo; a lo largo de $T$ pasos, la probabilidad de haber "descarrilado" se acumula, y cada estado descarrilado puede costar hasta $\mathcal{O}(T)$. El factor $T^2$ hace que el behavioral cloning sea frágil en tareas largas.

### 3.2 DAgger y la reducción a no-regret

DAgger convierte la imitación en **aprendizaje online**. En la iteración $i$ se rueda una política mezcla $\pi_i = \beta_i \pi^* + (1-\beta_i)\hat\pi_i$ (con $\beta_i \to 0$), se recolectan los estados visitados, se **etiquetan con el experto** y se agregan al dataset $\mathcal{D}$; luego $\hat\pi_{i+1} = \arg\min_\pi \mathbb{E}_{s\sim\mathcal{D}}[\ell(s,\pi)]$ (Follow-The-Leader). Un lema clave acota la deriva de distribución por iteración:

$$
\lVert d_{\pi_i} - d_{\hat\pi_i}\rVert_1 \le 2\,T\,\beta_i.
$$

Como el procedimiento es un algoritmo online sin arrepentimiento, si el regret promedio $\gamma_N \to 0$ existe una política en la secuencia con garantía **lineal** en el horizonte:

$$
J(\hat\pi) \le J(\pi^*) + \mathcal{O}(T\,\epsilon_N) + \mathcal{O}(T\,\gamma_N),
$$

pasando de $\mathcal{O}(T^2\epsilon)$ a $\mathcal{O}(T\epsilon)$. El precio es requerir un **experto consultable** durante el entrenamiento —justamente lo que provee un experto DQN en el [laboratorio](/laboratorios/lab-33).

---

## 4. GAIL: el puente adversario

Ho y Ermon (2016) formalizan el IRL como un problema con regularizador $\psi$ sobre la recompensa:

$$
\text{IRL}_\psi(\pi_E) = \arg\max_{r}\; \Big(\min_\pi -H(\pi) - \eta_r(\pi)\Big) + \eta_r(\pi_E) - \psi(r).
$$

Su resultado central (Proposición 3.2) es que **hacer RL sobre la recompensa que devuelve el IRL equivale a igualar medidas de ocupación**:

$$
\text{RL}\circ\text{IRL}_\psi(\pi_E) = \arg\min_\pi \; -H(\pi) + \psi^*\!\big(\rho_\pi - \rho_{\pi_E}\big),
$$

donde $\psi^*$ es la conjugada convexa de $\psi$. Eligiendo un regularizador $\psi_{GA}$ particular, $\psi^*$ resulta ser exactamente la **divergencia de Jensen-Shannon** entre $\rho_\pi$ y $\rho_{\pi_E}$, y el problema toma la forma de una **GAN**:

$$
\min_\pi \max_{D} \; \mathbb{E}_{\pi}\big[\log D(s,a)\big] + \mathbb{E}_{\pi_E}\big[\log(1 - D(s,a))\big] - \lambda H(\pi).
$$

El **discriminador** $D$ intenta distinguir las transiciones del experto de las del aprendiz (y su log actúa como recompensa aprendida), mientras la **política** —el generador— intenta engañarlo. La gran ventaja: en vez de resolver un RL completo por cada actualización de la recompensa (como el IRL clásico), GAIL **entrelaza** un paso de gradiente del discriminador con un paso de política (TRPO), evitando el bucle interno costoso.

{{< concept-alert type="clave" >}}
GAIL unifica los tres hilos de la clase: parte del **IRL** (aprender de demostraciones), realiza **imitación** (iguala el comportamiento del experto vía occupancy matching) y lo hace con maquinaria de **modelos generativos** (el objetivo min-max de una GAN). Es el eslabón conceptual entre esta clase y la [Clase 29 (modelos generativos)](/clases/clase-29).
{{< /concept-alert >}}

---

## 5. Formalización de la generalización en RL

El marco de Zhang et al. (2018) y Cobbe et al. (2019) trata cada instancia del ambiente como un **ejemplo muestreado**. Un generador $\mu$ (o un semillero de niveles procedurales) define una distribución sobre MDPs $M \sim \mu$; se muestrean conjuntos disjuntos de train y test. El desempeño de generalización es

$$
\text{Gen}(\pi) = \mathbb{E}_{M\sim\mu}\big[\eta_M(\pi)\big] \;\approx\; \frac{1}{|{\text{test}}|}\sum_{M\in\text{test}} \eta_M(\pi),
$$

y la **brecha de generalización** es la diferencia entre el retorno promedio en train y en test. El hallazgo empírico: con pocas instancias la brecha es enorme (el agente **memoriza**), y solo se cierra con **miles** de instancias distintas —o aplicando regularización clásica (dropout, $L_2$, data augmentation, batch norm, mayor estocasticidad de la política). La conclusión formaliza la respuesta de la clase: la generalización en RL requiere **variar, durante el entrenamiento, las dimensiones donde se quiere generalizar**. Detalle en el fundamento [Generalización en RL](/fundamentos/generalizacion-en-rl).

---

**Ver también:** [Clase 33 - Teoría](/clases/clase-33/teoria) · [Clase 33 - Práctica](/clases/clase-33/practica) · [Laboratorio: DAgger sobre Breakout](/laboratorios/lab-33) · Fundamentos: [IRL](/fundamentos/aprendizaje-reforzado-inverso) · [Imitación](/fundamentos/aprendizaje-por-imitacion) · [Generalización en RL](/fundamentos/generalizacion-en-rl) · [Aprendizaje Reforzado](/fundamentos/aprendizaje-reforzado).
