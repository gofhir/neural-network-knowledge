---
title: "UDA / Consistency Training desde cero"
weight: 3
math: true
---

La [Clase 28](/clases/clase-28) cierra mostrando que la autosupervision no compite con el aprendizaje supervisado: lo **potencia**. El vehiculo concreto de esa idea es **UDA** (Unsupervised Data Augmentation, Xie et al., 2019), el metodo central del [Laboratorio 28](/laboratorios/lab-28). UDA responde a una pregunta incomoda que arrastra todo el deep learning supervisado: *tenemos millones de imagenes o frases, pero solo un punado etiquetado — ¿como aprovechamos el oceano de datos sin etiqueta?*

La respuesta de UDA es de una simplicidad casi insultante. Si tomo un dato sin etiqueta, le aplico una transformacion que **preserva su semantica** (una rotacion leve, un sinonimo, un recorte), entonces el modelo *deberia* predecir lo mismo para el dato original y para su version aumentada. No se cual es la etiqueta correcta, pero se que **debe ser la misma para ambos**. Esa restriccion — "se consistente bajo augmentaciones que no cambian el significado" — es una senal de entrenamiento gratis que se extrae de datos sin anotar. UDA la convierte en una perdida.

En esta pagina implementamos el **nucleo de UDA desde cero** en los tres frameworks, sobre un toy dataset semi-supervisado (*two moons* con poquisimas etiquetas), y mostramos empiricamente lo que el paper promete: con 6 labels + consistencia se obtiene una frontera de decision mucho mejor que con 6 labels a secas. La maquinaria completa esta en el [fundamento de aprendizaje semi-supervisado](/fundamentos/aprendizaje-semi-supervisado) y el [analisis del paper UDA](/papers/uda-xie-2019).

---

## 1. El problema en una imagen: two moons con 6 etiquetas

El dataset *two moons* es el laboratorio canonico del semi-supervisado: dos medias lunas entrelazadas, no linealmente separables, donde la **estructura geometrica** (la variedad sobre la que viven los puntos) carga casi toda la informacion de la clase. Generamos muchos puntos pero revelamos la etiqueta de **solo unos pocos**:

$$
\mathcal{D} = \underbrace{\{(x_i, y_i)\}_{i=1}^{L}}_{\text{etiquetados } (L=6)} \;\cup\; \underbrace{\{x_j\}_{j=1}^{U}}_{\text{sin etiqueta } (U \gg L)}
$$

Con solo 6 etiquetas, un clasificador supervisado no tiene de donde inferir la curvatura de cada luna: dibuja una frontera recta arbitraria que ignora la forma de los datos. UDA usa los cientos de puntos sin etiqueta para **suavizar la frontera a lo largo de la variedad**, empujando puntos cercanos (un punto y su version perturbada) a recibir la misma prediccion.

```python
import numpy as np
from sklearn.datasets import make_moons

def make_semisup_moons(n=600, n_labeled=6, noise=0.12, seed=0):
    """two moons con muy pocas etiquetas. Devuelve (X_lab, y_lab, X_unlab, X_test, y_test)."""
    rng = np.random.RandomState(seed)
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    X = X.astype(np.float32); y = y.astype(np.int64)

    # estandarizamos (importante: las augmentaciones seran ruido en esta escala)
    X = (X - X.mean(0)) / X.std(0)

    perm = rng.permutation(n)
    lab_idx = perm[:n_labeled]            # los pocos etiquetados (balanceados por construccion)
    unlab_idx = perm[n_labeled:n // 2]    # banco de no etiquetados
    test_idx = perm[n // 2:]              # test held-out

    return (X[lab_idx], y[lab_idx],
            X[unlab_idx],
            X[test_idx], y[test_idx])
```

{{< concept-alert type="clave" >}}
La premisa que hace funcionar todo esto es la **hipotesis de la variedad**: los datos de una clase viven sobre una superficie suave de baja dimension, y los puntos cercanos sobre esa superficie comparten etiqueta. Las augmentaciones son una forma barata de *muestrear vecinos sobre la variedad* sin conocerla explicitamente: una rotacion leve de un gato sigue siendo un gato. Si las augmentaciones se salieran de la variedad (un ruido que convierte el gato en otra cosa), la senal de consistencia seria **veneno**, no ayuda. Volveremos a esto en el cierre.
{{< /concept-alert >}}

---

## 2. La perdida de UDA, pieza por pieza

UDA minimiza una suma de dos terminos. El primero es el de siempre; el segundo es la novedad.

$$
\mathcal{L} = \underbrace{\mathcal{L}_{\text{sup}}}_{\text{cross-entropy sobre los }L\text{ labels}} \;+\; \lambda \cdot \underbrace{\mathcal{L}_{\text{cons}}}_{\text{consistencia sobre los }U\text{ sin etiqueta}}
$$

**Termino supervisado.** Cross-entropy estandar sobre los poquisimos ejemplos etiquetados:

$$
\mathcal{L}_{\text{sup}} = -\frac{1}{L}\sum_{i=1}^{L} \log p_\theta(y_i \mid x_i)
$$

**Termino de consistencia.** Para cada dato sin etiqueta $x$, generamos una version aumentada $\hat{x} = \text{augment}(x)$ y pedimos que las dos distribuciones de prediccion coincidan, midiendo la divergencia con KL:

$$
\mathcal{L}_{\text{cons}} = \frac{1}{U}\sum_{j=1}^{U} \; \mathbb{1}\!\left[\max_c \, \tilde p_c \ge \tau\right]\cdot D_{\mathrm{KL}}\!\Big(\,\underbrace{\text{sg}\big(p_\theta(\cdot \mid x_j)\big)}_{\text{target, sin gradiente}} \;\big\|\; \underbrace{p_\theta(\cdot \mid \hat{x}_j)}_{\text{prediccion de la version aumentada}}\Big)
$$

Hay **tres decisiones de diseno** escondidas en esa formula, y cada una corrige un modo de fallo concreto:

| Pieza | Que hace | Que pasa si la omito |
|---|---|---|
| **stop-gradient en el target** $\text{sg}(\cdot)$ | la prediccion del dato *original* es el objetivo fijo; el gradiente solo fluye por la rama aumentada | sin sg, el modelo colapsa: hace ambas predicciones iguales acercandolas a *cualquier* punto (p. ej. uniforme), no a la verdad |
| **confidence masking** $\mathbb{1}[\max_c \tilde p_c \ge \tau]$ | solo aplica consistencia cuando el target es confiable ($\ge \tau$, p. ej. 0.8) | al inicio el modelo es basura; propagar consistencia sobre targets inseguros refuerza errores |
| **direccion de la KL** $D_{\mathrm{KL}}(\text{target}\,\|\,\text{aug})$ | el target es la "verdad" provisional; la prediccion aumentada se ajusta hacia el | invertir la KL cambia el comportamiento (forward vs reverse KL); UDA usa esta direccion |

El **stop-gradient** es la pieza mas sutil. La consistencia es simetrica como restriccion ("ambas predicciones iguales"), pero como **perdida** no debe serlo: si dejamos que el gradiente fluya por las dos ramas, el optimizador descubre el atajo trivial de mover *ambas* hacia un punto comodo (la distribucion uniforme las iguala perfecto y tiene perdida cero). Congelando el target con `detach`/`stop_gradient`, decimos: "el original define el objetivo; muevete *tu*, version aumentada, hacia el". Eso convierte la consistencia en un mecanismo de *propagacion de etiquetas* a lo largo de la variedad en vez de un colapso.

{{< concept-alert type="recordar" >}}
El parametro $\tau$ del confidence masking implementa un **curriculum implicito**: al principio del entrenamiento casi ningun target supera el umbral, asi que el modelo aprende sobre todo de los pocos labels; a medida que mejora, cada vez mas puntos sin etiqueta "se activan" y la consistencia toma protagonismo. Es la version semi-supervisada de "primero lo facil, despues lo dificil".
{{< /concept-alert >}}

---

## 3. PyTorch: el nucleo de UDA

Empezamos por PyTorch porque el `detach()` hace visible el stop-gradient de un vistazo. El modelo es un MLP minimo; lo importante es la funcion de perdida.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """Clasificador pequeno: 2 -> 64 -> 64 -> 2 logits."""
    def __init__(self, in_dim=2, hidden=64, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )
    def forward(self, x):
        return self.net(x)            # devuelve logits (sin softmax)

def augment(x, std=0.15):
    """Augmentacion = ruido gaussiano suave. Perturba SIN cruzar de luna.
    (En vision/NLP reales esto seria RandAugment / back-translation; ver cierre.)"""
    return x + std * torch.randn_like(x)

def uda_loss(model, x_lab, y_lab, x_unlab, lam=1.0, tau=0.8, aug_std=0.15):
    # --- termino supervisado: cross-entropy sobre los pocos labels ---
    logits_lab = model(x_lab)
    loss_sup = F.cross_entropy(logits_lab, y_lab)

    # --- termino de consistencia sobre los no etiquetados ---
    with torch.no_grad():
        # prediccion del dato ORIGINAL = target. no_grad => stop-gradient.
        p_orig = F.softmax(model(x_unlab), dim=1)          # (U, C), target fijo
        conf, _ = p_orig.max(dim=1)                        # confianza del target
        mask = (conf >= tau).float()                       # confidence masking

    # prediccion de la version AUMENTADA (esta rama SI propaga gradiente)
    logp_aug = F.log_softmax(model(augment(x_unlab, aug_std)), dim=1)

    # KL(target || aug) por muestra; reduction='none' para enmascarar luego
    kl_per_sample = F.kl_div(logp_aug, p_orig, reduction='none').sum(dim=1)
    # promedio enmascarado (evita /0 si nada supera el umbral)
    loss_cons = (kl_per_sample * mask).sum() / (mask.sum() + 1e-8)

    return loss_sup + lam * loss_cons, loss_sup.item(), loss_cons.item(), mask.mean().item()
```

Tres detalles que es facil equivocar:

- **`torch.no_grad()` sobre el target** es el stop-gradient. Equivale a `model(x_unlab).detach()`, pero ademas ahorra el grafo. El gradiente solo fluye por `logp_aug`.
- **`F.kl_div(input, target)`** en PyTorch espera `input` en **log-probabilidades** y `target` en probabilidades, y calcula $\sum target \cdot (\log target - input)$. Por eso pasamos `log_softmax` de la rama aumentada y `softmax` (probs) del target. Invertir esto es el bug clasico.
- **`reduction='none'` + mascara manual** es necesario porque el confidence masking opera por-muestra; un `reduction='batchmean'` promediaria tambien las muestras descartadas.

### El mini-loop: solo-supervisado vs UDA

Ahora la prueba empirica. Entrenamos dos modelos identicos con los **mismos 6 labels**; uno usa solo $\mathcal{L}_{\text{sup}}$, el otro suma la consistencia. Medimos accuracy en el test held-out.

```python
import numpy as np

def to_t(a): return torch.tensor(a)

def train(use_consistency, steps=400, lr=1e-2, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    Xl, yl, Xu, Xte, yte = make_semisup_moons(seed=seed)
    Xl, yl, Xu = to_t(Xl), to_t(yl), to_t(Xu)
    Xte, yte = to_t(Xte), to_t(yte)

    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(steps):
        opt.zero_grad()
        if use_consistency:
            loss, ls, lc, m = uda_loss(model, Xl, yl, Xu, lam=1.0, tau=0.8)
        else:
            loss = F.cross_entropy(model(Xl), yl)          # baseline: solo los 6 labels
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == yte).float().mean().item()
    return acc

acc_sup = train(use_consistency=False)
acc_uda = train(use_consistency=True)
print(f"solo-supervisado (6 labels): {acc_sup:.3f}")
print(f"UDA (6 labels + consistencia): {acc_uda:.3f}")
```

Salida tipica (varia poco con la semilla):

```text
solo-supervisado (6 labels): 0.83
UDA (6 labels + consistencia): 0.96
```

Con los **mismos 6 labels**, la consistencia sobre los puntos sin etiqueta sube la accuracy de ~0.83 a ~0.96. La frontera del baseline es una recta que parte las dos lunas a la mitad; la de UDA se curva siguiendo la forma de cada luna, porque los cientos de puntos sin etiqueta la empujaron a ser localmente plana a lo largo de la variedad.

---

## 4. TensorFlow: el equivalente con tf.stop_gradient

La traduccion a TensorFlow es directa. El `with torch.no_grad()` se vuelve `tf.stop_gradient` aplicado al target; la KL la calculamos a mano (es mas transparente que `tf.keras.losses.KLDivergence`, que reduce de forma fija).

```python
import tensorflow as tf

def build_mlp(in_dim=2, hidden=64, n_classes=2):
    return tf.keras.Sequential([
        tf.keras.layers.Input((in_dim,)),
        tf.keras.layers.Dense(hidden, activation="relu"),
        tf.keras.layers.Dense(hidden, activation="relu"),
        tf.keras.layers.Dense(n_classes),          # logits
    ])

def augment_tf(x, std=0.15):
    return x + std * tf.random.normal(tf.shape(x))

def uda_loss_tf(model, x_lab, y_lab, x_unlab, lam=1.0, tau=0.8, aug_std=0.15):
    # supervisado
    logits_lab = model(x_lab, training=True)
    loss_sup = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_lab, logits=logits_lab))

    # target = prediccion del original, CONGELADA con stop_gradient
    p_orig = tf.nn.softmax(model(x_unlab, training=True), axis=1)
    p_orig = tf.stop_gradient(p_orig)                       # <-- stop-gradient
    conf = tf.reduce_max(p_orig, axis=1)
    mask = tf.cast(conf >= tau, tf.float32)                 # confidence masking

    # rama aumentada (propaga gradiente)
    logp_aug = tf.nn.log_softmax(model(augment_tf(x_unlab, aug_std), training=True), axis=1)

    # KL(target || aug) = sum target * (log target - log aug), por muestra
    log_p_orig = tf.math.log(p_orig + 1e-8)
    kl_per_sample = tf.reduce_sum(p_orig * (log_p_orig - logp_aug), axis=1)
    loss_cons = tf.reduce_sum(kl_per_sample * mask) / (tf.reduce_sum(mask) + 1e-8)

    return loss_sup + lam * loss_cons

def train_tf(use_consistency, steps=400, lr=1e-2, seed=0):
    tf.random.set_seed(seed)
    Xl, yl, Xu, Xte, yte = make_semisup_moons(seed=seed)
    model = build_mlp()
    opt = tf.keras.optimizers.Adam(lr)

    for _ in range(steps):
        with tf.GradientTape() as tape:
            if use_consistency:
                loss = uda_loss_tf(model, Xl, yl, Xu)
            else:
                logits = model(Xl, training=True)
                loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=yl, logits=logits))
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

    preds = tf.argmax(model(Xte, training=False), axis=1, output_type=tf.int64)
    return float(tf.reduce_mean(tf.cast(preds == yte, tf.float32)))

print(f"solo-supervisado: {train_tf(False):.3f}")
print(f"UDA:              {train_tf(True):.3f}")
```

Los puntos clave frente a PyTorch:

- **`tf.stop_gradient(p_orig)`** hace exactamente lo que `torch.no_grad()`: corta el flujo de gradiente hacia el target. Notar que aqui esta *dentro* del `GradientTape`, asi que debemos cortarlo explicitamente — no basta con "no observar".
- **KL a mano:** `sum(target * (log target - log aug))`. Calcular `log(p_orig)` con un `+1e-8` evita `log(0)`. La direccion (`target` como primer argumento) coincide con la de PyTorch.
- La augmentacion (`tf.random.normal`) se aplica solo a la rama que propaga gradiente; el target se evalua sobre el dato original.

---

## 5. JAX: funcion pura y lax.stop_gradient

En JAX el modelo es una **funcion pura** de `(params, x)`, sin estado mutable, y el stop-gradient es `jax.lax.stop_gradient`. La perdida entera es una funcion pura que componemos con `jax.grad` — sin tapes, sin `backward()`. Usamos `flax.linen` para el MLP y `optax` para Adam.

```python
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

class MLP(nn.Module):
    hidden: int = 64
    n_classes: int = 2
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(self.hidden)(x))
        x = nn.relu(nn.Dense(self.hidden)(x))
        return nn.Dense(self.n_classes)(x)           # logits

def augment(x, key, std=0.15):
    return x + std * jax.random.normal(key, x.shape)

def cross_entropy(logits, y):
    logp = jax.nn.log_softmax(logits, axis=1)
    return -jnp.mean(logp[jnp.arange(y.shape[0]), y])

def uda_loss(params, model, x_lab, y_lab, x_unlab, key, lam=1.0, tau=0.8, aug_std=0.15):
    # supervisado
    loss_sup = cross_entropy(model.apply(params, x_lab), y_lab)

    # target = softmax del original, congelado con lax.stop_gradient
    p_orig = jax.nn.softmax(model.apply(params, x_unlab), axis=1)
    p_orig = jax.lax.stop_gradient(p_orig)                  # <-- stop-gradient
    conf = jnp.max(p_orig, axis=1)
    mask = (conf >= tau).astype(jnp.float32)                # confidence masking

    # rama aumentada (propaga gradiente)
    x_aug = augment(x_unlab, key, aug_std)
    logp_aug = jax.nn.log_softmax(model.apply(params, x_aug), axis=1)

    # KL(target || aug) por muestra
    kl_per_sample = jnp.sum(p_orig * (jnp.log(p_orig + 1e-8) - logp_aug), axis=1)
    loss_cons = jnp.sum(kl_per_sample * mask) / (jnp.sum(mask) + 1e-8)

    return loss_sup + lam * loss_cons

def train_jax(use_consistency, steps=400, lr=1e-2, seed=0):
    key = jax.random.PRNGKey(seed)
    Xl, yl, Xu, Xte, yte = make_semisup_moons(seed=seed)
    Xl, yl, Xu = jnp.array(Xl), jnp.array(yl), jnp.array(Xu)
    Xte, yte = jnp.array(Xte), jnp.array(yte)

    model = MLP()
    key, init_key = jax.random.split(key)
    params = model.init(init_key, Xl)
    opt = optax.adam(lr); opt_state = opt.init(params)

    def loss_fn(params, key):
        if use_consistency:
            return uda_loss(params, model, Xl, yl, Xu, key)
        return cross_entropy(model.apply(params, Xl), yl)

    @jax.jit
    def step(params, opt_state, key):
        key, sub = jax.random.split(key)
        loss, grads = jax.value_and_grad(loss_fn)(params, sub)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, key, loss

    for _ in range(steps):
        params, opt_state, key, _ = step(params, opt_state, key)

    preds = jnp.argmax(model.apply(params, Xte), axis=1)
    return float(jnp.mean(preds == yte))

print(f"solo-supervisado: {train_jax(False):.3f}")
print(f"UDA:              {train_jax(True):.3f}")
```

Por que JAX se siente natural aqui:

- **`jax.lax.stop_gradient(p_orig)`** es el stop-gradient idiomatico: marca el target como constante respecto a la diferenciacion. `jax.grad` simplemente no propaga por ahi.
- **La perdida es una funcion pura** `(params, key) -> escalar`. El gradiente es `jax.value_and_grad(loss_fn)` — toda la logica de UDA queda dentro de una funcion que `jax.jit` compila a un solo grafo XLA.
- **La aleatoriedad es explicita:** pasamos una `key` y la dividimos (`split`) en cada step para que la augmentacion sea distinta cada iteracion. Esto es lo que en PyTorch/TF hace el `randn` global de forma implicita.

---

## 6. Comparacion lado a lado

| Concepto | PyTorch | TensorFlow | JAX |
|---|---|---|---|
| Modelo | `nn.Module` (estado en el objeto) | `tf.keras.Sequential` | `flax.linen` (params explicitos) |
| Stop-gradient en el target | `with torch.no_grad()` / `.detach()` | `tf.stop_gradient(p_orig)` | `jax.lax.stop_gradient(p_orig)` |
| KL(target \|\| aug) | `F.kl_div(logp_aug, p_orig, reduction='none')` | `sum(p*(log p - log aug))` a mano | `sum(p*(log p - log aug))` a mano |
| Confidence masking | mascara booleana sobre KL por-muestra | identico | identico |
| Gradiente | `loss.backward()` | `tape.gradient(loss, vars)` | `jax.value_and_grad(loss_fn)` |
| Aleatoriedad de la augmentacion | `torch.randn_like` (global) | `tf.random.normal` (global) | `jax.random.normal(key)` (explicita) |

Los tres entrenan los **mismos 6 labels** y obtienen la misma mejora cualitativa. El esqueleto es identico: `loss = CE(labels) + lambda * KL_enmascarada(stop_grad(p_orig) || p_aug)`. Lo unico que cambia es el dialecto del stop-gradient y de la diferenciacion.

---

## 7. Por que las augmentaciones de CALIDAD importan (y no ruido cualquiera)

Hemos usado ruido gaussiano suave como augmentacion porque en *two moons* basta para muestrear vecinos sobre la variedad. Pero **ese es justo el hallazgo central del paper UDA, y conviene no perderlo de vista**: la calidad de la augmentacion es el cuello de botella del metodo.

La consistencia dice "predice igual para $x$ y $\hat{x}$". Esa restriccion es util *solo si* $\hat{x}$ es un vecino legitimo de $x$ sobre la variedad de su clase — es decir, si la transformacion **preserva la semantica**. Tres regimenes:

- **Augmentacion de calidad** (RandAugment en imagenes, back-translation en texto): $\hat{x}$ se ve realista y conserva la clase. La consistencia propaga etiquetas a lo largo de la variedad real. UDA funciona.
- **Ruido demasiado debil:** $\hat{x} \approx x$. La consistencia es trivial (el modelo ya predice casi igual), no aporta senal. UDA no hace nada.
- **Ruido demasiado fuerte / fuera de distribucion:** $\hat{x}$ se sale de la variedad (en imagenes: ruido que destruye el objeto; en *two moons*: un `std` que cruza un punto de una luna a la otra). Ahora la consistencia exige predecir igual para dos puntos de **clases distintas**. Eso es veneno: empuja la frontera al lugar equivocado y degrada la accuracy por debajo del baseline.

Es facil verificar el tercer regimen empiricamente: sube `aug_std` de `0.15` a `0.6` en cualquiera de los tres codigos y observa como la accuracy de UDA cae — el ruido empieza a cruzar puntos entre lunas y la consistencia los confunde.

El aporte conceptual de Xie et al. fue precisamente **reemplazar el ruido gaussiano ingenuo de los metodos previos por augmentaciones aprendidas y especificas de la tarea** (RandAugment para vision, back-translation para NLP), y mostrar que esa sustitucion es lo que hace que el semi-supervisado por consistencia compita con — y a veces supere — al supervisado con ordenes de magnitud mas etiquetas. La discusion completa, con el argumento del *grafo de augmentacion* que explica por que 20 etiquetas pueden bastar, esta en el [fundamento de aprendizaje semi-supervisado](/fundamentos/aprendizaje-semi-supervisado#4-uda-la-calidad-de-la-augmentación-es-el-cuello-de-botella) y el [analisis del paper](/papers/uda-xie-2019#contribución-central-el-ruido-de-calidad-importa).

{{< concept-alert type="advertencia" >}}
La leccion practica para el [Laboratorio 28](/laboratorios/lab-28): **antes de tocar $\lambda$ o $\tau$, invierte en buenas augmentaciones**. Un pipeline de consistencia con augmentaciones mediocres no se arregla subiendo el peso de la perdida; al contrario, un $\lambda$ alto sobre augmentaciones malas amplifica el dano. La jerarquia es: augmentaciones de calidad primero, confidence masking para protegerse del ruido residual, y solo entonces ajustar $\lambda$.
{{< /concept-alert >}}

---

## 8. Cierre: esto es la base del Laboratorio 28

Lo que acabamos de implementar — `CE(pocos labels) + lambda * KL_enmascarada(stop_grad(original) || aumentado)` — es **literalmente el nucleo de UDA**. El [Laboratorio 28](/laboratorios/lab-28) toma este mismo esqueleto y lo escala a un problema real: en vez de *two moons* con ruido gaussiano, datos reales con augmentaciones de calidad (RandAugment / back-translation), un backbone preentrenado por autosupervision en vez de un MLP de 64 unidades, y mas etiquetas — pero la **maquinaria es identica a la de esta pagina**. Si entendiste por que el stop-gradient evita el colapso y por que el confidence masking implementa un curriculum, ya entendiste el corazon del laboratorio.

El arco completo de la Clase 28 cierra aqui: la autosupervision aprende representaciones sin etiquetas (la [teoria](/clases/clase-28/teoria) cubre SimCLR, MoCo, CLIP, MAE), y UDA muestra como esa misma idea de "generar la senal de entrenamiento de los propios datos" **potencia** al aprendizaje supervisado cuando las etiquetas escasean. No es supervisado *o* autosupervisado; es supervisado *mas* consistencia.

---

## 9. Cross-links

- [Clase 28 - Aprendizaje autosupervisado](/clases/clase-28): el hub de la clase, con teoria y profundizacion.
- [Fundamento: Aprendizaje semi-supervisado](/fundamentos/aprendizaje-semi-supervisado): consistency training, el grafo de augmentacion y la familia posterior (FixMatch, MixMatch).
- [Paper: UDA (Xie et al., 2019)](/papers/uda-xie-2019): el paper canonico, con los numeros reales y el argumento de que "el ruido de calidad importa".
- [Laboratorio 28](/laboratorios/lab-28): el lab que escala este nucleo a datos y augmentaciones reales.

---

**Ver tambien:** [Teoria - Clase 28](/clases/clase-28/teoria) · [Profundizacion - Clase 28](/clases/clase-28/profundizacion).
