---
title: "Aprendizaje Reforzado Inverso (IRL)"
weight: 110
math: true
---

El **aprendizaje reforzado inverso** (inverse reinforcement learning, IRL) da vuelta la pregunta central del [aprendizaje reforzado](/fundamentos/aprendizaje-reforzado). En el RL clásico conocemos la **función de recompensa** y buscamos la **política óptima** que la maximiza. En el IRL tenemos lo contrario: observamos el comportamiento de un **agente experto** —una colección de demostraciones— y queremos **inferir la función de recompensa** que ese experto parece estar optimizando. La motivación es profundamente práctica: en muchísimos problemas del mundo real es **muy difícil escribir a mano una buena recompensa**, pero es fácil mostrar ejemplos de buen comportamiento. Este fundamento acompaña a la [Clase 33](/clases/clase-33), donde el IRL aparece como la respuesta a la pregunta "¿qué hago si no sé cómo definir la recompensa de mi problema?".

---

## 1. El problema de especificar recompensas

El objetivo de un agente de RL es obtener la mayor recompensa posible; para ello ajusta su política $\pi(a\mid s)$. El problema es que un agente **está dispuesto a hacer cualquier cosa con tal de conseguir recompensa**, incluso cosas que no queríamos. Esto convierte el diseño de la recompensa —la *reward engineering*— en un arte peligroso y lleno de efectos colaterales.

El ejemplo canónico de la clase es **conducir un auto**. ¿Cuál es una buena recompensa para conducir bien?

- ¿Llegar al destino? (un agente que solo optimiza esto atropella peatones para ahorrar segundos)
- ¿No cruzar en rojo? (necesario, pero insuficiente)
- ¿No chocar? ¿No matar a nadie? ¿Ir cómodo? ¿Gastar poca gasolina? ...

Ninguna combinación manual de estos términos captura del todo lo que un humano entiende por "conducir bien". Cada término que agregamos introduce nuevos *trade-offs* que hay que ponderar, y cada peso mal calibrado abre la puerta a un comportamiento degenerado. La pregunta que abre el IRL es directa: **¿será posible aprender la función de recompensa en lugar de escribirla?**

{{< concept-alert type="clave" >}}
El IRL parte de una observación simple pero poderosa: **si observamos a un experto actuar, podemos inferir qué está optimizando**. Un conductor humano frena ante los peatones, respeta los carriles y mantiene distancias: de ese comportamiento se puede *deducir* que la recompensa penaliza atropellar, salirse del carril y chocar. El IRL formaliza matemáticamente esa deducción.
{{< /concept-alert >}}

---

## 2. La formalización: recuperar la recompensa desde el comportamiento

Sea $\pi^*$ la política que sigue el experto. La intuición del IRL es que, bajo la recompensa correcta $R$, **la política del experto debe ser al menos tan buena como cualquier otra**. Formalmente, buscamos una función $R : S \to \mathbb{R}$ tal que

$$
\mathbb{E}_{\pi^*}\!\left[\sum_{t=0}^{\infty} \gamma^t R(s_t)\right] \;\ge\; \mathbb{E}_{\pi}\!\left[\sum_{t=0}^{\infty} \gamma^t R(s_t)\right] \quad \text{para toda } \pi \neq \pi^*.
$$

Es decir, el valor esperado (descontado por $\gamma$) de las trayectorias del experto debe superar al de cualquier otra política. Esta es exactamente la caracterización que Ng y Russell (2000) propusieron en el paper fundacional del campo.

### 2.1. El problema mal planteado (*ill-posed*)

Esta formulación tiene un defecto grave: es **degenerada**. La recompensa trivial $R(s) = 0$ para todo estado satisface la desigualdad (todas las políticas empatan en 0), y lo mismo ocurre con infinitas otras recompensas. El problema está **mal planteado**: muchas funciones de recompensa distintas explican las mismas demostraciones. Este es el desafío central que ha guiado la evolución del IRL durante dos décadas.

Ng y Russell lo atacan de dos maneras. Primero, convierten las desigualdades en un problema de **optimización con margen**: en vez de pedir solo que el experto gane, piden que gane **por el mayor margen $m$ posible**,

$$
\max_{R}\; m \quad \text{s.a.}\quad \mathbb{E}_{\pi^*}\!\left[\sum_t \gamma^t R(s_t)\right] \ge \mathbb{E}_{\pi}\!\left[\sum_t \gamma^t R(s_t)\right] + m \quad \forall\, \pi \neq \pi^*.
$$

Segundo, agregan un término de **regularización** (típicamente una penalización $L_1$ sobre $R$) que prefiere recompensas *simples* —con pocos estados relevantes— para desempatar entre las muchas soluciones posibles.

### 2.2. La estrategia incremental

Enumerar *todas* las políticas $\pi \neq \pi^*$ es imposible (hay una cantidad exponencial). El truco práctico es un **bucle iterativo** que va agregando restricciones a medida que las necesita:

1. Elige una política inicial cualquiera $\pi_0$.
2. Resuelve el problema con margen usando solo $\{\pi_0\}$ como conjunto de rivales, obteniendo $R_1$.
3. Calcula la política **óptima** $\pi_1$ respecto a $R_1$ (esto es un problema de RL *forward* estándar).
4. Si $\pi_1$ resulta **mejor que el experto** bajo $R_1$, es que $R_1$ es mala: agrega $\pi_1$ al conjunto de rivales y resuelve de nuevo para $R_2$.
5. Repite hasta que ninguna política nueva supere al experto: $\pi^* \ge \pi_n$ bajo $R_n$.

Cada iteración requiere resolver un RL completo por dentro, lo que hace al IRL clásico **computacionalmente costoso**. Esta limitación será clave para entender por qué aparecieron métodos posteriores como GAIL.

---

## 3. Apprenticeship learning: igualar al experto sin recuperar la recompensa "verdadera"

Abbeel y Ng (2004) hicieron un giro conceptual decisivo. En vez de obsesionarse con recuperar la recompensa *verdadera* del experto (que es inobservable y ambigua), se conforman con algo más modesto y alcanzable: **encontrar una política que se desempeñe tan bien como el experto**. A esto lo llamaron *apprenticeship learning* (aprendizaje por aprendizaje/aprendiz).

La herramienta clave son las **expectativas de features** (*feature expectations*). Se supone que la recompensa es **lineal** en un vector de características $\phi(s) \in \mathbb{R}^k$:

$$
R(s) = w^\top \phi(s).
$$

Entonces el valor esperado de una política se factoriza limpiamente:

$$
\mathbb{E}_\pi\!\left[\sum_t \gamma^t R(s_t)\right] = w^\top \underbrace{\mathbb{E}_\pi\!\left[\sum_t \gamma^t \phi(s_t)\right]}_{\mu(\pi)},
$$

donde $\mu(\pi)$ es el **vector de expectativas de features** de la política. La observación crucial es: **si logramos que $\mu(\tilde\pi) \approx \mu_E$** (las features del aprendiz igualan las del experto), entonces

$$
\lvert w^\top \mu(\tilde\pi) - w^\top \mu_E \rvert \le \lVert w \rVert \, \lVert \mu(\tilde\pi) - \mu_E \rVert,
$$

y el desempeño del aprendiz se acerca al del experto **sin importar cuáles sean los verdaderos pesos $w$**. Se esquiva por completo la ambigüedad: no necesitamos la recompensa correcta, solo igualar el comportamiento agregado. El algoritmo alterna un paso *max-margin* (tipo SVM) para hallar el $w$ que más separa al experto de las políticas candidatas, y un paso de RL *forward* que produce la mejor política bajo ese $w$.

Este es exactamente el método que la Clase 33 muestra en dos ejemplos famosos de Abbeel y Ng: **aprender a conducir** con distintos estilos (agresivo, prudente, respetando o no las líneas) y **aprender a estacionar** un auto robótico en un estacionamiento, donde se aprende la *función de costo* del planificador de movimiento a partir de demostraciones.

---

## 4. Maximum Entropy IRL: resolver la ambigüedad con probabilidad

Ziebart et al. (2008) dieron el siguiente salto conceptual con el **IRL de máxima entropía** (MaxEnt IRL). En lugar de elegir *una* recompensa, modelan una **distribución de probabilidad sobre trayectorias** completas, donde las trayectorias con mayor retorno de features son exponencialmente más probables:

$$
P(\zeta) \;\propto\; \exp\!\left(w^\top f_\zeta\right),
$$

donde $f_\zeta = \sum_{s \in \zeta} \phi(s)$ es el vector de features acumulado a lo largo de la trayectoria $\zeta$. Esta es una **distribución de Boltzmann** sobre el comportamiento.

¿Por qué máxima entropía? Porque entre todas las distribuciones consistentes con las expectativas de features observadas, la de **máxima entropía** es la que **no introduce ningún supuesto adicional** más allá de los datos. Elige la explicación menos comprometida posible, lo que resuelve elegantemente la ambigüedad del IRL. Además, tolera **expertos sub-óptimos**: no exige que el experto sea perfecto, solo que su comportamiento sea *probable*. Los pesos $w$ se ajustan por **máxima verosimilitud**, con un gradiente que tiene una forma muy intuitiva —la diferencia entre las features empíricas del experto y las esperadas bajo el modelo:

$$
\nabla_w \mathcal{L} = \tilde f - \sum_\zeta P(\zeta)\, f_\zeta = \tilde f - \sum_s D_s\, \phi(s),
$$

donde $D_s$ son las **frecuencias esperadas de visita** de estados, calculables con un algoritmo *forward-backward* de programación dinámica. MaxEnt IRL es la base de casi todo el IRL moderno (Deep MaxEnt IRL, Guided Cost Learning).

---

## 5. GAIL: el puente hacia las GANs

El IRL clásico es caro: cada actualización de la recompensa exige resolver un problema de RL completo por dentro. Ho y Ermon (2016) demostraron un resultado elegante: **"hacer RL sobre la recompensa que recupera el IRL" equivale a igualar la distribución de ocupación** estado-acción $\rho_\pi(s,a)$ del aprendiz con la del experto. Con un regularizador particular, este *matching* de distribuciones toma exactamente la forma de una **red generativa adversaria (GAN)**:

$$
\min_\pi \max_D \; \mathbb{E}_\pi[\log D(s,a)] + \mathbb{E}_{\pi_E}[\log(1 - D(s,a))] - \lambda H(\pi).
$$

Aquí el **discriminador $D$** aprende a distinguir las transiciones del experto de las del aprendiz —jugando el rol de "recompensa aprendida"— y la **política actúa como generador** que intenta engañarlo produciendo comportamiento indistinguible del experto. **Generative Adversarial Imitation Learning (GAIL)** evita el costoso bucle interno de RL del IRL clásico y se convirtió en uno de los métodos de imitación más influyentes, conectando de forma profunda tres campos: IRL, imitación y modelos generativos.

---

## 6. IRL frente al aprendizaje por imitación

El IRL y el [aprendizaje por imitación](/fundamentos/aprendizaje-por-imitacion) parten del mismo insumo —demostraciones expertas— pero difieren en la estrategia:

- El **aprendizaje por imitación** directo (behavioral cloning) aprende la política $\pi^*(a\mid s)$ **directamente**, como un problema de clasificación supervisada: estado → acción del experto.
- El **IRL** aprende primero la **recompensa** $R$ y *después* deriva la política óptima resolviendo el RL con esa recompensa.

El IRL es más indirecto y costoso, pero tiene una ventaja fundamental: la **recompensa es un objeto más transferible y compacto que la política**. Una recompensa aprendida puede reutilizarse cuando cambia la dinámica del ambiente, generaliza mejor a situaciones nuevas y captura la *intención* del experto en vez de solo su superficie de comportamiento. La clase resume el trade-off: la imitación directa es más simple, pero el IRL entiende *por qué* el experto actúa como actúa.

{{< concept-alert type="recordar" >}}
El **RLHF** que alinea a los grandes modelos de lenguaje (Clase 20) es, en esencia, una forma de IRL: a partir de **preferencias humanas** entre respuestas se aprende un **modelo de recompensa**, y luego se optimiza la política del LLM con PPO respecto a esa recompensa aprendida. La misma lógica de "no sé escribir la recompensa, pero puedo mostrar/juzgar buen comportamiento" que motivó a Ng y Russell en el año 2000 es la que hoy sostiene a ChatGPT.
{{< /concept-alert >}}

---

## 7. Relevancia para MDM y record linkage

Para quien construye un sistema de **matching de pacientes** (MDM), el IRL ofrece una analogía directa y útil. Definir a mano la función de *scoring* que decide si dos registros clínicos son la misma persona —cuánto pesa un nombre igual, una fecha de nacimiento distinta por un dígito, un RUT ausente— es exactamente el tipo de *reward engineering* frágil que el IRL busca evitar. En cambio, los *data stewards* humanos toman miles de decisiones de match/no-match que constituyen **demostraciones expertas**. Un enfoque estilo apprenticeship learning permitiría **calibrar el scorer para que reproduzca el comportamiento agregado de los stewards** (igualar sus "expectativas de features") sin exigir replicar cada decisión individual, y un enfoque MaxEnt sería robusto ante stewards imperfectos que a veces se equivocan. La lección de Ng y Russell —es más fácil reconocer el buen comportamiento que especificarlo— es tan válida para conducir un auto como para decidir si dos historias clínicas pertenecen al mismo paciente.

---

## Referencias

- Ng, A. & Russell, S. (2000). *Algorithms for Inverse Reinforcement Learning*. ICML. — [análisis interno](/clases/clase-33)
- Abbeel, P. & Ng, A. (2004). *Apprenticeship Learning via Inverse Reinforcement Learning*. ICML.
- Abbeel, P., Dolgov, D., Ng, A. & Thrun, S. (2008). *Apprenticeship Learning for Motion Planning with Application to Parking Lot Navigation*. IROS.
- Ziebart, B., Maas, A., Bagnell, J.A. & Dey, A. (2008). *Maximum Entropy Inverse Reinforcement Learning*. AAAI.
- Ho, J. & Ermon, S. (2016). *Generative Adversarial Imitation Learning*. NeurIPS.
