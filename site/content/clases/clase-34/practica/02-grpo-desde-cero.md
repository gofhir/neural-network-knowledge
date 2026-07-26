---
title: "GRPO desde cero"
weight: 2
math: true
---

El [camino 01](/clases/clase-34/practica/01-self-consistency-y-pass-at-k-desde-cero) usó un razonador **fijo** y estudió cómo agregar sus muestras. Aquí damos el salto de la Clase 34: **entrenar** la política con **Aprendizaje Reforzado y recompensa verificable** —la receta que llevó a [DeepSeek-R1](/papers/deepseek-r1-2025). Implementamos **GRPO** (Group Relative Policy Optimization) desde cero sobre una tarea de juguete verificable, en **triple framework**. GRPO es la variante de [PPO](/papers/ppo-schulman-2017) que **elimina la red de valor** y estima la ventaja **normalizando dentro de un grupo** de respuestas muestreadas —simple, elegante y sorprendentemente efectiva.

> **Lecturas de apoyo:** el fundamento [Test-time compute](/fundamentos/test-time-compute) y el [análisis de DeepSeek-R1](/papers/deepseek-r1-2025); la [Clase 31](/clases/clase-31) para policy gradient.

---

## 1. La tarea verificable

Necesitamos una tarea con **recompensa objetiva** (correcta / incorrecta), sin etiquetas supervisadas —la política debe descubrir la respuesta **solo del reward**, como R1-Zero. Usamos un mapeo que la política no conoce y debe aprender: para un problema $t \in \{0,\dots,T-1\}$, la respuesta correcta es $f(t) = (3t + 1) \bmod K$.

```python
import numpy as np
T, K = 8, 8                       # 8 problemas, 8 respuestas posibles
def correct_answer(t): return (3 * t + 1) % K     # el verificador (regla oculta)
def reward(t, a):     return 1.0 if a == correct_answer(t) else 0.0
```

La política nunca ve `correct_answer` directamente: solo recibe **+1 cuando acierta**. Es RL con **recompensa verificable basada en reglas**, exactamente el tipo de señal de DeepSeek-R1 (donde el "verificador" comprueba matemáticas o código).

---

## 2. GRPO: la idea en una fórmula

Para cada problema $t$, GRPO muestrea un **grupo** de $G$ respuestas $\{a_1,\dots,a_G\}$ de la política actual, obtiene sus recompensas $\{r_1,\dots,r_G\}$, y estima la **ventaja** de cada respuesta **normalizándola dentro del grupo**:

$$
A_i = \frac{r_i - \operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G) + \epsilon}.
$$

Luego actualiza la política por **policy gradient**, subiendo la probabilidad de las respuestas con ventaja positiva (mejores que el promedio del grupo) y bajando las de ventaja negativa:

$$
\mathcal{L}(\theta) = -\frac{1}{G}\sum_{i=1}^{G} A_i \, \log \pi_\theta(a_i \mid t).
$$

{{< concept-alert type="clave" >}}
La genialidad de GRPO: la **media del grupo** hace de *baseline* (el rol que en PPO cumple una red de valor). No hace falta entrenar un crítico —el grupo de muestras se evalúa a sí mismo. Si en un problema 2 de 8 muestras aciertan, esas 2 obtienen ventaja positiva y el resto negativa; la política se corre hacia las que funcionaron.
{{< /concept-alert >}}

---

## 3. Implementación

La política es una tabla de logits $\theta \in \mathbb{R}^{T \times K}$ (un vector de logits por problema) —el mínimo modelo entrenable. En un LLM real serían los logits del transformer; aquí, para ver el algoritmo puro, basta una tabla.

### PyTorch

```python
import torch, torch.nn.functional as F

def train_grpo_torch(steps=300, G=8, lr=0.1):
    logits = torch.zeros(T, K, requires_grad=True)     # política: logits por problema
    opt = torch.optim.Adam([logits], lr=lr)
    for step in range(steps):
        opt.zero_grad()
        loss = 0.0
        for t in range(T):
            probs = F.softmax(logits[t], dim=-1)
            dist = torch.distributions.Categorical(probs)
            a = dist.sample((G,))                       # grupo de G respuestas
            r = torch.tensor([reward(t, int(ai)) for ai in a])
            adv = (r - r.mean()) / (r.std() + 1e-8)     # ventaja relativa al grupo
            loss = loss - (adv * dist.log_prob(a)).mean()
        loss.backward(); opt.step()
    return logits.detach()

logits = train_grpo_torch()
acc = np.mean([logits[t].argmax().item() == correct_answer(t) for t in range(T)])
print("accuracy tras GRPO:", acc)        # -> 1.0: aprendió la regla solo del reward
```

### TensorFlow

```python
import tensorflow as tf

def train_grpo_tf(steps=300, G=8, lr=0.1):
    logits = tf.Variable(tf.zeros((T, K)))
    opt = tf.keras.optimizers.Adam(lr)
    for step in range(steps):
        with tf.GradientTape() as tape:
            loss = 0.0
            for t in range(T):
                probs = tf.nn.softmax(logits[t])
                a = tf.random.categorical(tf.math.log(probs)[None], G)[0]   # G muestras
                r = tf.constant([reward(t, int(ai)) for ai in a.numpy()])
                adv = (r - tf.reduce_mean(r)) / (tf.math.reduce_std(r) + 1e-8)
                logp = tf.math.log(tf.gather(probs, a) + 1e-12)
                loss -= tf.reduce_mean(adv * logp)
        grads = tape.gradient(loss, [logits])
        opt.apply_gradients(zip(grads, [logits]))
    return logits.numpy()
```

### JAX (con optax)

```python
import jax, jax.numpy as jnp, optax

def train_grpo_jax(steps=300, G=8, lr=0.1, seed=0):
    logits = jnp.zeros((T, K))
    opt = optax.adam(lr); opt_state = opt.init(logits)
    key = jax.random.PRNGKey(seed)

    def loss_fn(logits, samples, rewards):
        # samples: [T, G] acciones ; rewards: [T, G]
        logp = jax.nn.log_softmax(logits)                       # [T, K]
        logp_a = jnp.take_along_axis(logp, samples, axis=1)     # [T, G]
        adv = (rewards - rewards.mean(1, keepdims=True)) / (rewards.std(1, keepdims=True) + 1e-8)
        return -(adv * logp_a).mean()

    for step in range(steps):
        key, sk = jax.random.split(key)
        probs = jax.nn.softmax(logits)
        samples = jax.random.categorical(sk, logits[:, None, :].repeat(G, 1), axis=-1)  # [T, G]
        rewards = jnp.array([[reward(t, int(samples[t, i])) for i in range(G)] for t in range(T)])
        loss, grads = jax.value_and_grad(loss_fn)(logits, samples, rewards)
        updates, opt_state = opt.update(grads, opt_state)
        logits = optax.apply_updates(logits, updates)
    return logits

logits = train_grpo_jax()
print("accuracy:", np.mean([int(logits[t].argmax()) == correct_answer(t) for t in range(T)]))
```

Las tres comparten la **misma receta GRPO**: muestrear un grupo → recompensa verificable → ventaja normalizada al grupo → policy gradient. No hay red de valor, no hay etiquetas supervisadas: la política **descubre la regla oculta solo del reward**.

---

## 4. Por qué funciona (y qué observa la clase)

Al inicio la política es uniforme; por puro azar, algunas de las $G$ muestras de cada problema aciertan. La normalización de ventaja **premia** esas muestras y **castiga** las demás, corriendo la distribución hacia la respuesta correcta. Tras unos cientos de pasos, la política concentra su masa en $f(t)$ para cada $t$ —**aprendió la regla sin que nadie se la dijera**, solo verificando.

{{< concept-alert type="advertencia" >}}
Esto reproduce, en miniatura, el hallazgo de la Clase 34 y la [crítica de Yue et al.](/papers/rl-reasoning-yue-2025): GRPO **concentra la probabilidad** en respuestas que la política ya podía muestrear al inicio (por azar). Aquí la tarea es tan simple que "concentrar" basta para resolverla; pero en LLMs reales, el debate es si el RL **descubre** razonamientos nuevos o solo **reordena** los que el modelo base ya tenía en su soporte. Nuestro experimento ilustra el mecanismo: GRPO **reponderó** el muestreo, no inventó respuestas fuera del vocabulario.
{{< /concept-alert >}}

### 4.1 Experimento: ¿qué pasa si la respuesta correcta es inalcanzable?

Si extendiéramos la tarea para que la respuesta correcta **estuviera fuera** del vocabulario de la política (probabilidad inicial cero), GRPO **nunca** la encontraría: sin una muestra correcta en el grupo, todas las ventajas son cero y no hay señal. Es exactamente el límite que señala Yue: **el RL no puede premiar lo que la política nunca muestrea**. La capacidad tiene que estar en el soporte del modelo base.

---

## 5. Qué nos llevamos

- **GRPO** entrena una política por policy gradient usando la **media del grupo como baseline** —sin red de valor, sin etiquetas, solo **recompensa verificable**.
- La **ventaja normalizada al grupo** $A_i = (r_i - \bar r)/\sigma_r$ es todo el secreto: convierte "acerté / fallé" en una señal de gradiente centrada.
- El algoritmo es idéntico en PyTorch, TensorFlow y JAX: muestrear grupo → verificar → normalizar → policy gradient.
- Su límite conceptual —GRPO reordena el muestreo, no expande el soporte— es la miniatura de uno de los debates abiertos de la clase.

---

**Ver también:** [Clase 34 - Teoría](/clases/clase-34/teoria) · [Clase 34 - Profundización](/clases/clase-34/profundizacion) · [Camino 01: Self-Consistency y Pass@k](/clases/clase-34/practica/01-self-consistency-y-pass-at-k-desde-cero) · [Laboratorio](/laboratorios/lab-34).
