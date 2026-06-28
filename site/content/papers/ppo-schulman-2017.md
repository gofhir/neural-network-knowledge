---
title: "PPO: Proximal Policy Optimization (2017)"
weight: 354
math: true
---

{{< paper-card
    title="Proximal Policy Optimization Algorithms"
    authors="John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov"
    year="2017"
    venue="arXiv / OpenAI"
    pdf="/papers/ppo-schulman-2017.pdf"
    arxiv="1707.06347" >}}
Paper de **OpenAI** que propone **PPO**, una familia de métodos *policy gradient* que alcanza la estabilidad de TRPO usando **solo optimización de primer orden**. Su núcleo es un objetivo *surrogate* recortado (*clipped*) que desincentiva que la política nueva se aleje demasiado de la vieja directamente dentro de la función de pérdida —en vez de imponer una restricción dura de KL como TRPO. El resultado es simple de implementar ("unas pocas líneas sobre un policy gradient vanilla"), permite varias épocas de SGD por lote, y se volvió **posiblemente el algoritmo de RL profundo más usado hoy**. Su impacto mayor para este curso: es la **base del RLHF** que afinó ChatGPT e InstructGPT.
{{< /paper-card >}}

---

## Contexto: el dilema de los policy gradient

Los métodos *policy gradient* optimizan directamente una política estocástica $\pi_\theta(a \mid s)$ ascendiendo por el gradiente de la recompensa esperada. El estimador típico tiene la forma

$$\hat{g} = \hat{\mathbb{E}}_t\!\left[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\,\hat{A}_t\right],$$

donde $\hat{A}_t$ es un estimador de la **función de ventaja** (*advantage*): cuánto mejor que el promedio fue tomar la acción $a_t$ en el estado $s_t$.

El problema que el paper diagnostica: es tentador hacer **varios pasos de optimización sobre el mismo lote** de datos para exprimirlos, pero "hacerlo no está bien justificado, y empíricamente suele llevar a actualizaciones de política destructivamente grandes". El estimador solo es válido *localmente*, alrededor de la política que generó los datos; insistir con muchos gradientes sobre el mismo lote empuja la política fuera de esa región de validez y la colapsa. Por eso el *vanilla policy gradient* gasta un gradiente por muestra y luego recolecta datos nuevos: **mala eficiencia de muestras** y poca robustez.

**TRPO** (Trust Region Policy Optimization, 2015) ataca esa inestabilidad limitando *cuánto* puede cambiar la política. Maximiza un objetivo *surrogate* sujeto a una **restricción de divergencia KL** entre la política nueva y la vieja:

$$\underset{\theta}{\text{maximizar}}\;\; \hat{\mathbb{E}}_t\!\left[\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}\,\hat{A}_t\right] \quad\text{sujeto a}\quad \hat{\mathbb{E}}_t\!\left[\mathrm{KL}[\pi_{\theta_{old}},\, \pi_\theta]\right] \le \delta.$$

La restricción define una "región de confianza" (*trust region*) que garantiza mejoras casi monótonas. El precio es la **complejidad**: se resuelve con gradiente conjugado tras aproximar el objetivo linealmente y la restricción cuadráticamente —es decir, requiere **información de segundo orden** (productos con la matriz de Fisher). Además, TRPO "no es compatible con arquitecturas que incluyen ruido (como dropout) o compartición de parámetros" entre política y función de valor. PPO busca lo mejor de ambos mundos: la fiabilidad de TRPO con la simplicidad del primer orden.

## Contribución central: el clipped surrogate objective

La aportación clave de PPO es reemplazar la restricción de segundo orden de TRPO por un **recorte (*clipping*)** dentro del objetivo, optimizable con gradiente de primer orden. Sea

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$$

el **ratio de probabilidades** entre la política nueva y la vieja para la acción tomada. Nótese que $r_t(\theta_{old}) = 1$ por construcción. El *surrogate* sin restricción es $L^{CPI}(\theta) = \hat{\mathbb{E}}_t[r_t(\theta)\,\hat{A}_t]$; maximizarlo sin freno lleva a pasos excesivos. PPO propone en cambio:

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\!\left[\min\!\Big(r_t(\theta)\,\hat{A}_t,\;\; \mathrm{clip}(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon)\,\hat{A}_t\Big)\right],$$

con $\epsilon$ un hiperparámetro (típicamente $\epsilon = 0.2$). La mecánica:

- **El primer término** es $L^{CPI} = r_t\,\hat{A}_t$, el objetivo sin recortar.
- **El segundo término** recorta el ratio al intervalo $[1-\epsilon,\, 1+\epsilon]$, **eliminando el incentivo a mover $r_t$ fuera de esa banda**.
- **Tomar el mínimo** convierte el objetivo en una **cota inferior pesimista**: solo se *ignora* el cambio del ratio cuando este haría *mejorar* el objetivo; cuando lo empeora, se incluye. El recorte impide ganancias "baratas" por alejarse, pero no protege de empeoramientos.

La intuición geométrica del paper: para **ventaja positiva** ($\hat{A}_t > 0$, la acción fue buena, queremos subir su probabilidad) el objetivo crece con $r_t$ pero se **aplana** una vez que $r_t > 1+\epsilon$. Para **ventaja negativa** ($\hat{A}_t < 0$) se aplana cuando $r_t < 1-\epsilon$. En ambos casos el aplanamiento le quita al optimizador el incentivo de dar pasos enormes, de modo que **varios pasos de SGD sobre el mismo lote ya no destruyen la política**. A primer orden alrededor de $\theta_{old}$ (donde $r=1$), $L^{CLIP} = L^{CPI}$; solo divergen cuando $\theta$ se aleja.

De ahí viene el nombre **proximal**: las nuevas políticas se mantienen "próximas" a las viejas, no por una restricción explícita, sino porque el objetivo deja de premiar el alejamiento. El beneficio práctico decisivo es que esto **habilita múltiples épocas de SGD por minibatch sobre el mismo lote** —justo lo que el *vanilla policy gradient* no podía hacer sin colapsar— y de ahí buena parte de la mejor eficiencia de muestras.

## Las variantes y el algoritmo completo

**Penalización KL adaptativa.** Como alternativa al recorte, el paper presenta una variante que usa una penalización KL pero **adapta el coeficiente $\beta$** para alcanzar un objetivo $d_{targ}$ de divergencia: si la KL medida quedó muy baja se relaja $\beta$ (a la mitad), si se pasó se endurece (al doble). Esto resuelve el problema que llevó a TRPO a rechazar la penalización fija: $\beta$ deja de ser un hiperparámetro delicado. Sin embargo, en los experimentos **rinde peor que el recorte**, así que el recorte es la versión recomendada.

**Estimación de ventaja con GAE.** PPO usa **Generalized Advantage Estimation** (Schulman et al. 2015), que combina los errores temporales de Bellman:

$$\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \cdots + (\gamma\lambda)^{T-t+1}\delta_{T-1}, \qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t),$$

donde $V$ es una función de valor aprendida, $\gamma$ el factor de descuento y $\lambda$ controla el balance sesgo–varianza. Esto requiere correr la política $T$ pasos (mucho menor que la longitud del episodio), un estilo bien adaptado a recolección paralela.

**Objetivo conjunto actor-critic.** Cuando la red **comparte parámetros entre política (actor) y valor (critic)** —algo que TRPO no soportaba—, se combinan tres términos en una sola pérdida:

$$L_t^{CLIP+VF+S}(\theta) = \hat{\mathbb{E}}_t\!\left[L_t^{CLIP}(\theta) - c_1\,L_t^{VF}(\theta) + c_2\,S[\pi_\theta](s_t)\right],$$

con $L_t^{VF}$ el error cuadrático de la función de valor, $S$ un **bono de entropía** que fomenta la exploración, y $c_1, c_2$ coeficientes de ponderación.

**Algoritmo (estilo actor-critic).** En cada iteración, $N$ **actores paralelos** corren $\pi_{\theta_{old}}$ durante $T$ pasos y computan ventajas; se construye la pérdida *surrogate* sobre los $N \times T$ pasos y se optimiza con **minibatch SGD/Adam durante $K$ épocas**; luego se actualiza $\theta_{old} \leftarrow \theta$ y se repite. Hiperparámetros típicos de MuJoCo: $T=2048$, Adam $3\times10^{-4}$, $K=10$ épocas, minibatch 64, $\gamma=0.99$, $\lambda_{GAE}=0.95$.

## Experimentos

**Comparación de objetivos surrogate (MuJoCo).** Sobre 7 tareas de control continuo, puntaje normalizado promedio:

| Variante | Puntaje |
|---|---|
| Sin recorte ni penalización | −0.39 |
| Recorte, $\epsilon=0.1$ | 0.76 |
| **Recorte, $\epsilon=0.2$** | **0.82** |
| Recorte, $\epsilon=0.3$ | 0.70 |
| KL adaptativa, $d_{targ}=0.01$ | 0.74 |
| KL fija, $\beta=1$ | 0.71 |

El veredicto es claro: **el recorte con $\epsilon=0.2$ gana**, la KL adaptativa queda segunda, y la versión sin protección colapsa (peor que la política aleatoria). El mecanismo de recorte es lo que importa.

**Contra otros algoritmos.** En las mismas tareas MuJoCo, PPO (recorte, $\epsilon=0.2$) "supera a los métodos previos en casi todos los entornos de control continuo" —incluyendo TRPO, CEM, *policy gradient* vanilla y A2C. No solo iguala a TRPO siendo mucho más simple: típicamente lo supera. En un *showcase* de **humanoide 3D** (Roboschool), PPO aprende políticas de locomoción complejas y robustas (correr, dirigirse a objetivos, levantarse tras ser bombardeado con cubos).

**Atari.** En 49 juegos del Arcade Learning Environment, contra A2C y ACER (más sofisticado, con *experience replay*), conteo de juegos ganados:

| Métrica | A2C | ACER | PPO |
|---|---|---|---|
| (1) recompensa durante todo el entrenamiento | 1 | 18 | **30** |
| (2) recompensa últimos 100 episodios | 1 | **28** | 19 |

PPO domina en **velocidad de aprendizaje / eficiencia de muestras** (métrica 1), superando a ACER pese a ser mucho más simple; en desempeño final asintótico (métrica 2) ACER lo supera, pero PPO sigue muy por encima de A2C.

## Limitaciones

- **El recorte no garantiza monotonía teórica.** A diferencia de TRPO, PPO sustituye las garantías de mejora casi monótona por una heurística (el `min` recortado) que funciona muy bien en la práctica pero sin la misma fundamentación formal.
- **Sensibilidad a detalles de implementación.** Aunque $\epsilon \in [0.1, 0.3]$ es robusto, el rendimiento depende de normalización de ventajas, recorte de gradiente, recocido del paso, número de épocas. Trabajos posteriores mostraron que parte del éxito viene de estos "trucos de código" además del objetivo.
- **La variante KL adaptativa rinde peor** que el recorte (incluida como base, no recomendación).
- **Sigue siendo *on-policy*.** PPO descarta los datos tras unas pocas épocas (sin *replay buffer* persistente como DQN o ACER), lo que limita la reutilización de experiencia en regímenes de muestras escasas.

## Impacto y conexión con la Clase 31

PPO se convirtió en **posiblemente el algoritmo de RL profundo más usado en la práctica**, precisamente por la propiedad que vendía: robustez con simplicidad. Es el caballo de batalla por defecto en bibliotecas (OpenAI Baselines, Stable-Baselines, RLlib), en robótica simulada y en videojuegos (base de OpenAI Five para Dota 2).

El impacto que más conecta con este curso: PPO es el **algoritmo de optimización detrás del RLHF** (*Reinforcement Learning from Human Feedback*) que afinó **InstructGPT** y **ChatGPT**. Ahí la "política" es el modelo de lenguaje, la "acción" es generar el siguiente token, y la "recompensa" la entrega un *reward model* entrenado con preferencias humanas; PPO ajusta el modelo **manteniéndose próximo** al original (un término KL contra la política de referencia evita que el modelo "se rompa" optimizando en exceso). La misma idea de "no alejarse demasiado de la política previa" que motivó el recorte de 2017 reaparece, una década después, como el mecanismo que hizo viable alinear modelos de lenguaje gigantes. Ver [/fundamentos/rlhf](/fundamentos/rlhf).

La [Clase 31](/clases/clase-31) recorre el RL en dos familias. La línea **value-based** (Q-learning → DQN y variantes) aprende una función de valor y deriva la política de ella. PPO pertenece a la línea complementaria, **policy-based / actor-critic**, que optimiza la política directamente y es la opción natural cuando el espacio de acciones es **continuo** (control robótico), donde tomar el `argmax` de una Q-función es inviable. PPO cierra el linaje de *policy gradient*: REINFORCE → actor-critic con ventaja ([A3C/A2C](/papers/a3c-mnih-2016)) → región de confianza (TRPO) → recorte de primer orden (PPO), el destino pragmático de esa evolución.

## Notas y enlaces

- Preprint: arXiv:1707.06347 (2017), [arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347).
- Fundamentos: [Aprendizaje reforzado](/fundamentos/aprendizaje-reforzado) (MDP, política, valor, ventaja) y [RLHF](/fundamentos/rlhf).
- Antecedente actor-critic directo: [A3C (Mnih et al. 2016)](/papers/a3c-mnih-2016).
- Hub de la clase: [Clase 31](/clases/clase-31).
