---
title: "KL Implicito (en DPO)"
weight: 288
math: true
---

El **KL implicito** es la forma de regularizacion que aparece en [DPO](/fundamentos/dpo) sin que escribamos explicitamente un termino de KL divergencia en la loss. La magia: los **log-ratios** $\log \pi_\theta(y \mid x) / \pi_{\text{ref}}(y \mid x)$ que aparecen dentro de la sigmoide ya estan haciendo el trabajo de penalizar a la policy cuando se aleja de la referencia. El hiperparametro $\beta$ controla la fuerza de esa penalizacion.

Esta entrada compara el **KL explicito** del RLHF clasico (PPO + KL penalty) con el **KL implicito** de DPO, muestra por que matematicamente son equivalentes en su efecto, y discute cuando preferir uno u otro en la practica.

---

## 1. KL divergencia: rol regularizador en RL

La **divergencia de Kullback-Leibler** entre dos distribuciones $p$ y $q$ sobre el mismo soporte es:

$$
D_{\text{KL}}(p \| q) = \mathbb{E}_{y \sim p} \log \frac{p(y)}{q(y)}.
$$

Es no-negativa, vale 0 si y solo si $p = q$, y mide "cuanta informacion extra" se gasta usando $q$ cuando la distribucion verdadera es $p$. Para policies $\pi(\cdot \mid x)$, el KL en cada $x$ es la divergencia entre las distribuciones de respuestas.

En **RL con KL penalty**, agregamos un termino $-\lambda D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ al objetivo. La intuicion: queremos maximizar reward, **pero** sin alejarnos demasiado del comportamiento de la policy de referencia (porque ese reward fue medido sobre datos cercanos a $\pi_{\text{ref}}$, y extrapolar lejos es peligroso -- *reward hacking*).

Esta penalizacion aparece desde TRPO (Schulman 2015) y se vuelve estandar en RLHF.

---

## 2. KL explicito en RLHF clasico (PPO con KL penalty)

En la receta InstructGPT/PPO-RLHF, el objetivo es:

$$
\max_\theta \; \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot \mid x)} \big[ r_\phi(x, y) \big] - \lambda \, \mathbb{E}_{x} \big[ D_{\text{KL}}(\pi_\theta(\cdot \mid x) \| \pi_{\text{ref}}(\cdot \mid x)) \big].
$$

Tres componentes:

- **Reward** $r_\phi$: aprendido previamente con [Bradley-Terry](/fundamentos/bradley-terry) sobre preferencias humanas.
- **KL explicito** $D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$: estimado en cada paso de PPO via sampling y log-prob ratios.
- **Coeficiente $\lambda$**: hiperparametro que pondera reward vs proximidad. Tipicamente $\lambda \in [0.01, 0.5]$.

En la practica el KL se aproxima por **muestreo Monte Carlo**:

$$
\hat{D}_{\text{KL}} = \frac{1}{N} \sum_{i=1}^N \log \frac{\pi_\theta(y_i \mid x)}{\pi_{\text{ref}}(y_i \mid x)}, \quad y_i \sim \pi_\theta.
$$

Lo que se incorpora es el reward modificado: $\tilde{r}(x, y) = r_\phi(x, y) - \lambda \log \pi_\theta(y \mid x) / \pi_{\text{ref}}(y \mid x)$. PPO optimiza este $\tilde{r}$.

Ventajas: control fino, $\lambda$ ajustable on-the-fly. Desventajas: necesitas sampling on-policy en cada step (caro), variancia alta del estimador, ingenieria PPO completa.

---

## 3. KL implicito en DPO: la magia de los log-ratios

DPO empieza desde el mismo objetivo regularizado:

$$
\max_\pi \; \mathbb{E}_{x, y \sim \pi} [r(x, y)] - \beta \, D_{\text{KL}}(\pi(\cdot \mid x) \| \pi_{\text{ref}}(\cdot \mid x)).
$$

Pero **resuelve la optimizacion en forma cerrada**. La policy optima $\pi^*$ satisface:

$$
\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right),
$$

con $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp(r(x, y)/\beta)$ la constante de normalizacion. Despejando $r$:

$$
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x).
$$

Sustituyendo en la loss Bradley-Terry sobre preferencias (donde solo importa $r_w - r_l$, asi que $\log Z(x)$ se cancela):

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[ \log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right) \right].
$$

**Aqui esta el KL implicito**: dentro de la sigmoide, el log-ratio $\log \pi_\theta / \pi_{\text{ref}}$ es el integrando de $D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ evaluado en muestras especificas $y_w, y_l$ del dataset. No tenemos que samplear de $\pi_\theta$ ni estimar el KL via Monte Carlo: la **estructura algebraica** de la loss ya lo penaliza.

Si $\pi_\theta(y_w \mid x)$ aumenta mucho mas que $\pi_{\text{ref}}(y_w \mid x)$, el log-ratio crece y la sigmoide satura -- pero el coeficiente $\beta$ frena cuanto puede crecer antes de que el gradiente se desvanezca. Equivalentemente, $\beta$ controla la magnitud de log-ratios que la loss tolera.

---

## 4. Por que policy != ref se penaliza automaticamente

Considera el extremo: si $\pi_\theta = \pi_{\text{ref}}$, los log-ratios son 0, la diferencia dentro de la sigmoide es 0, y la loss vale $-\log\sigma(0) = \log 2 \approx 0.69$. Ese es el "punto de partida" -- DPO arranca con loss $\log 2$ en cualquier $(\theta, \text{ref})$ tal que $\theta = \text{ref}$.

Conforme entrena, $\pi_\theta(y_w \mid x)$ crece (porque queremos producir chosen) y/o $\pi_\theta(y_l \mid x)$ baja (porque queremos evitar rejected). El log-ratio chosen sube, el rejected baja, la diferencia crece, $\sigma$ aumenta, la loss baja. Pero **el ratio entero** $\log \pi_\theta / \pi_{\text{ref}}$ es lo que crece, no $\log \pi_\theta$ solo. Eso significa que si el modelo "se aleja mucho" de $\pi_{\text{ref}}$ en un sentido neutral (no alineado con preferencias), el log-ratio diverge sin razon y eso dispara el costo de la loss en el siguiente par de preferencias.

Resultado: la policy aprende a moverse **solo en la direccion** que distingue chosen de rejected, **manteniendose cerca de $\pi_{\text{ref}}$** en todo lo demas. Eso es exactamente lo que el KL explicito quiere lograr.

---

## 5. Beta como peso del KL implicito

El hiperparametro $\beta$ aparece multiplicando los log-ratios. Su rol es exactamente analogo al $\lambda$ del KL explicito, pero con interpretacion **invertida**:

| Caso | $\beta$ chico ($\sim 0.01$) | $\beta$ grande ($\sim 1.0$) |
|---|---|---|
| Sigmoide $\sigma(\beta(\Delta_w - \Delta_l))$ | Casi lineal en torno a 0 | Satura rapido |
| Tolerancia a divergencia $\pi_\theta$ vs $\pi_{\text{ref}}$ | Permite mucha divergencia | Penaliza fuerte cualquier divergencia |
| KL implicito efectivo | **Debil** | **Fuerte** |
| Gradiente lejos del optimo | Pequeno (sigmoide casi lineal) | Pequeno (saturada) |
| Riesgo | Modelo deriva, alucinaciones nuevas | Apenas se mueve del SFT |

Valores tipicos en papers: $\beta \in [0.01, 0.5]$. Llama-Chat reporta $\beta = 0.1$ como default. La regla heuristica: si DPO no esta moviendo el modelo lo suficiente, baja $\beta$; si esta destruyendo capacidades, sube $\beta$.

---

## 6. KL explicito vs implicito: cuando preferir cada uno

| Aspecto | KL explicito (PPO-RLHF) | KL implicito (DPO) |
|---|---|---|
| Necesita sampling on-policy | Si (caro) | No |
| Necesita reward model | Si | No |
| Estabilidad de training | Sensible a hyperparams PPO | Mas estable |
| Permite reward shaping ad-hoc | Si | Limitado |
| Permite explorar fuera del dataset | Si | No (offline puro) |
| Compute total | 5-10x DPO | Baseline |
| Resultado calidad-vs-costo | Levemente mejor en algunas evals | Comparable, mucho mas barato |

DPO domina cuando el dataset de preferencias esta dado y el objetivo es offline. PPO-RLHF sigue siendo preferible cuando: (a) puedes recoger preferencias online iterativamente, (b) el dataset es muy chico y necesitas explorar, o (c) tu reward model es altamente custom (ej. un verificador de codigo, no un MLP de preferencias).

En la practica de 2024-2026, **DPO y sus variantes** (IPO, KTO, ORPO, SimPO) se han vuelto el default para alineacion offline, justamente por la simplicidad del KL implicito.

---

## 7. Conexion con regularizacion de policy en general

El KL implicito de DPO es un caso particular de un patron mas general: **insertar la regularizacion en la parametrizacion** en vez de agregarla como termino aditivo al loss.

- En estadistica clasica, agregar L2 penalty al loss es equivalente a colocar un prior gaussiano sobre los parametros (regularizacion via parametrizacion bayesiana).
- En aprendizaje contrastivo, los `target encoder` y `momentum encoder` (BYOL, MoCo) cumplen el rol de "ref" implicito.
- En RL clasico, **KL trust regions** (TRPO) ya jugaban este truco con linear constraints.

DPO es la version de este patron para alineamiento offline con preferencias.

---

## 8. Resumen

- **KL divergencia** mide distancia entre distribuciones; en RL regulariza la policy hacia una referencia.
- **KL explicito** (PPO-RLHF): se agrega como termino aditivo al objetivo, se estima por sampling on-policy. Caro, pero flexible.
- **KL implicito** (DPO): aparece dentro de la sigmoide via log-ratios $\log \pi_\theta / \pi_{\text{ref}}$. Barato, sin sampling, sin reward model.
- $\beta$ es el peso del KL implicito: chico tolera divergencia, grande la penaliza.
- Si $\pi_\theta = \pi_{\text{ref}}$ los log-ratios son 0 y la loss vale $\log 2$; el entrenamiento solo desplaza la policy en la direccion de las preferencias.
- DPO es el default en 2024-2026 para alineacion offline; PPO sigue util para online iterativo o rewards custom.
- **Patron general**: regularizar via parametrizacion en vez de via termino aditivo. DPO es ese patron aplicado a alineamiento.

## Ver tambien

- [DPO](/fundamentos/dpo) -- la loss completa donde aparece el KL implicito.
- [SFT](/fundamentos/sft) -- produce el $\pi_{\text{ref}}$ que el KL implicito penaliza salir.
- [Bradley-Terry](/fundamentos/bradley-terry) -- la base de la loss DPO en la que se inserta el KL implicito.
- [Regularizacion](/fundamentos/regularizacion) -- contexto general de regularizacion en redes neuronales.
- [Clase 14 cap 27 - DPO loss](/clases/clase-14/practica/27-dpo-loss) -- derivacion donde aparece el KL implicito.
