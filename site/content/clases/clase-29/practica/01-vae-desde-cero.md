---
title: "VAE desde cero"
weight: 1
math: true
---

La clase 29 abre la puerta a los **modelos generativos**: en vez de preguntar *¿a que clase pertenece esta imagen?*, preguntamos *¿como genero una imagen nueva que parezca real?* El primer integrante de esa familia, y el mas didactico para programar de cero, es el **Variational Autoencoder** (VAE) de Kingma y Welling (2013). Es la pieza que conecta dos mundos: el autoencoder clasico —que comprime y reconstruye— y la inferencia probabilistica —que aprende una *distribucion* sobre el espacio latente, no un punto fijo.

En este camino construimos un VAE completo sobre **MNIST** en los **tres frameworks** —PyTorch, TensorFlow/Keras y JAX—, verificando dimensiones en cada paso. El objetivo no es solo que el codigo corra, sino que entiendas *por que* cada linea esta ahi: por que el encoder produce dos vectores ($\mu$ y $\log\sigma^2$) en vez de uno, por que necesitamos el *reparameterization trick* para poder retropropagar a traves del muestreo, y por que la perdida tiene dos terminos que tiran en direcciones opuestas. Cuando termines, vas a poder muestrear digitos nuevos desde ruido gaussiano e interpolar suavemente entre un 3 y un 8 en el espacio latente.

La base teorica esta en el [fundamento de modelos generativos](/fundamentos/modelos-generativos) y el desarrollo matematico completo en el [paper de Kingma y Welling 2013](/papers/vae-kingma-2013). Aqui nos concentramos en bajar esas ecuaciones a codigo correcto y verificable.

---

## 1. La idea en cuatro piezas

Un VAE tiene cuatro engranajes. Fijemos los cuatro antes de tocar codigo, porque el resto del capitulo es solo implementarlos.

**1) Encoder probabilistico $q_\phi(z \mid x)$.** En un autoencoder normal, el encoder mapea cada imagen $x$ a *un* vector latente $z$. En un VAE, mapea $x$ a los **parametros de una distribucion gaussiana**: una media $\mu(x)$ y una varianza diagonal. Por estabilidad numerica no predecimos $\sigma^2$ directamente (debe ser positivo), sino su logaritmo $\log\sigma^2$, que vive en todo $\mathbb{R}$:

$$
q_\phi(z \mid x) = \mathcal{N}\big(z;\ \mu_\phi(x),\ \operatorname{diag}(\sigma_\phi^2(x))\big), \qquad \sigma = \exp\!\big(\tfrac{1}{2}\log\sigma^2\big). \tag{1}
$$

**2) Reparameterization trick.** Para entrenar por gradiente necesitamos muestrear $z \sim q_\phi(z\mid x)$, pero muestrear es una operacion estocastica y no se puede retropropagar a traves de ella. El truco consiste en sacar el azar *afuera* de los parametros: muestreamos un ruido fijo $\varepsilon \sim \mathcal{N}(0, I)$ y construimos $z$ de forma deterministica a partir de $\mu$, $\sigma$ y $\varepsilon$:

$$
z = \mu + \sigma \odot \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I). \tag{2}
$$

Ahora $z$ es una funcion **diferenciable** de $\mu$ y $\sigma$ (el ruido $\varepsilon$ es una constante en cada paso), y el gradiente fluye limpiamente hacia el encoder. El simbolo $\odot$ es producto elemento a elemento.

**3) Decoder $p_\theta(x \mid z)$.** Toma el latente $z$ y reconstruye la imagen. Para MNIST con pixeles normalizados en $[0,1]$, la salida pasa por una sigmoide y se interpreta como la probabilidad de que cada pixel este "encendido" (Bernoulli por pixel). De ahi que la perdida de reconstruccion natural sea la **entropia cruzada binaria** (BCE).

**4) La perdida: ELBO.** Entrenamos maximizando el *Evidence Lower BOund*, equivalentemente minimizando su negativo, que tiene dos terminos:

$$
\mathcal{L}(x) = \underbrace{-\mathbb{E}_{q_\phi(z\mid x)}\big[\log p_\theta(x\mid z)\big]}_{\text{reconstruccion}} \;+\; \underbrace{D_{\mathrm{KL}}\big(q_\phi(z\mid x)\,\|\,p(z)\big)}_{\text{regularizacion}}. \tag{3}
$$

El primer termino empuja a reconstruir bien la imagen. El segundo, la **divergencia KL** entre la posterior aproximada $q_\phi(z\mid x)$ y la prior $p(z) = \mathcal{N}(0, I)$, empuja a que el espacio latente se parezca a una gaussiana estandar. Esos dos terminos tiran en direcciones opuestas —reconstruir perfecto querria un latente sin restricciones; la KL querria colapsar todo a $\mathcal{N}(0,I)$— y el equilibrio entre ambos es lo que hace que el latente sea **continuo y muestreable** (seccion 7).

### 1.1 La KL en forma cerrada

La magia practica del VAE es que, cuando tanto $q_\phi(z\mid x)$ como la prior $p(z)$ son gaussianas, la KL **no necesita muestreo**: tiene una formula analitica. Para una posterior gaussiana diagonal $\mathcal{N}(\mu, \sigma^2)$ contra la prior $\mathcal{N}(0, I)$ en $J$ dimensiones latentes:

$$
D_{\mathrm{KL}}\big(\mathcal{N}(\mu, \sigma^2)\,\|\,\mathcal{N}(0, I)\big) = -\frac{1}{2}\sum_{j=1}^{J}\Big(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\Big). \tag{4}
$$

Esta es la Ecuacion (10) del apendice B de Kingma y Welling. En codigo, con `logvar` $=\log\sigma^2$, queda literalmente:

```text
KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
```

Verifica el caso degenerado: si $\mu = 0$ y $\sigma^2 = 1$ (es decir $\log\sigma^2 = 0$), cada termino es $1 + 0 - 0 - 1 = 0$, asi que $\mathrm{KL} = 0$. Tiene sentido: la posterior ya *es* la prior, no hay divergencia. Esa verificacion mental es la mejor forma de detectar un signo equivocado en el codigo.

### 1.2 La arquitectura que vamos a construir

Para que las tres implementaciones sean comparables, fijamos una configuracion minima de MLP (sin convoluciones, para que el foco este en el VAE y no en la CNN):

| Hiperparametro | Valor | Comentario |
|---|---:|---|
| Dim de entrada | 784 | MNIST $28\times28$ aplanado |
| Capa oculta encoder | 400 | una capa ReLU |
| Dim latente $J$ | 20 | suficiente para MNIST, comodo para inspeccionar |
| Capa oculta decoder | 400 | simetrica al encoder |
| Salida decoder | 784 + sigmoide | probabilidad por pixel |
| Perdida reconstruccion | BCE sumada | Bernoulli por pixel |
| Optimizador | Adam, lr $10^{-3}$ | |

El flujo de shapes para un batch de tamano $B$:

| Etapa | Shape | Operacion |
|---|---|---|
| Entrada $x$ | $(B, 784)$ | imagen aplanada |
| Hidden encoder | $(B, 400)$ | Linear + ReLU |
| $\mu$, $\log\sigma^2$ | $(B, 20)$ cada uno | dos cabezas lineales |
| $z$ | $(B, 20)$ | reparameterize, Ec. (2) |
| Hidden decoder | $(B, 400)$ | Linear + ReLU |
| Reconstruccion $\hat{x}$ | $(B, 784)$ | Linear + sigmoide |

---

## 2. Seccion 1: PyTorch

PyTorch es el framework de referencia para prototipar modelos generativos: el `forward` define-by-run hace que el reparameterize y el doble retorno $(\mu, \log\sigma^2)$ se lean de forma natural.

### 2.1 Imports y modelo

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

INPUT_DIM = 784    # 28*28
HIDDEN_DIM = 400
LATENT_DIM = 20


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        # --- Encoder: x -> hidden -> (mu, logvar) ---
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)      # cabeza para mu
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # cabeza para log(sigma^2)
        # --- Decoder: z -> hidden -> reconstruccion ---
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        # x: (B, 784) -> dos vectores (B, 20)
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        # Ec. (2): z = mu + sigma * eps,  eps ~ N(0, I)
        std = torch.exp(0.5 * logvar)          # sigma = exp(0.5 * logvar)
        eps = torch.randn_like(std)            # ruido (B, 20), mismo device/dtype
        return mu + std * eps                  # (B, 20), diferenciable en mu y std

    def decode(self, z):
        # z: (B, 20) -> reconstruccion (B, 784) en [0, 1]
        h = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))

    def forward(self, x):
        mu, logvar = self.encode(x)            # (B, 20), (B, 20)
        z = self.reparameterize(mu, logvar)    # (B, 20)
        x_hat = self.decode(z)                 # (B, 784)
        return x_hat, mu, logvar
```

Las dos cabezas `fc_mu` y `fc_logvar` comparten la representacion oculta `h` y se separan al final: es la traduccion directa de la Ecuacion (1). Predecimos `logvar` (no `var`) por dos motivos: vive en todo $\mathbb{R}$ (no hay que forzar positividad) y la formula de la KL lo usa directamente.

### 2.2 La perdida: reconstruccion + KL

```python
def vae_loss(x_hat, x, mu, logvar):
    """
    x_hat: (B, 784) reconstruccion en [0, 1]
    x:     (B, 784) original en [0, 1]
    mu, logvar: (B, 20)
    Devuelve la perdida total SUMADA sobre el batch (no promediada),
    para que ambos terminos esten en la misma escala.
    """
    # Reconstruccion: BCE por pixel, sumada sobre pixeles y batch
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum")

    # KL en forma cerrada, Ec. (4):
    #   KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon + kl, recon, kl
```

Un detalle que rompe muchas implementaciones: la **escala**. Si la reconstruccion va sumada (`reduction="sum"`) y la KL tambien va sumada, los dos terminos viven en la misma escala y el balance es el del paper original. Si mezclas `mean` en un termino y `sum` en otro, uno domina por un factor de cientos y el modelo o ignora la KL (latente sin estructura) o colapsa (reconstrucciones borrosas e iguales). Aqui sumamos ambos y dividimos por el tamano del batch al reportar.

### 2.3 El loop de entrenamiento sobre MNIST

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# MNIST: tensores en [0, 1], aplanados a 784 en el loop
transform = transforms.ToTensor()
train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

model = VAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for epoch in range(10):
    total = 0.0
    for x, _ in train_loader:
        x = x.view(x.size(0), -1)          # (B, 1, 28, 28) -> (B, 784)
        x_hat, mu, logvar = model(x)
        loss, recon, kl = vae_loss(x_hat, x, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()

    n = len(train_loader.dataset)
    print(f"epoch {epoch:2d}  loss/img={total/n:.2f}")
```

El `x.view(x.size(0), -1)` aplana cada imagen $28\times28$ a un vector de 784. MNIST de `torchvision` ya entrega los pixeles en $[0,1]$ gracias a `ToTensor()`, que es justo lo que la BCE necesita (probabilidades, no logits).

### 2.4 Muestreo desde la prior $\mathcal{N}(0, I)$

Aqui esta el pago: una vez entrenado, generamos digitos nuevos **sin imagen de entrada**. Muestreamos $z$ directamente de la prior y lo pasamos por el decoder:

```python
@torch.no_grad()
def sample(model, n=16, latent_dim=20):
    model.eval()
    z = torch.randn(n, latent_dim)         # z ~ N(0, I), shape (n, 20)
    x_hat = model.decode(z)                # (n, 784) en [0, 1]
    return x_hat.view(n, 1, 28, 28)        # de vuelta a imagenes

imgs = sample(model, n=16)                 # 16 digitos generados de la nada
```

Esto solo funciona *porque* la KL forzo a que el latente se parezca a $\mathcal{N}(0,I)$ durante el entrenamiento: muestreamos de la misma distribucion que el modelo aprendio a decodificar. En un autoencoder normal (sin KL), muestrear de $\mathcal{N}(0,I)$ daria basura, porque el encoder pudo haber colocado los codigos en cualquier region arbitraria del espacio.

---

## 3. Seccion 2: TensorFlow / Keras

La version en TF 2.x usa subclassing de `keras.Model` y `tf.GradientTape` para los gradientes. El reparameterize muestrea con `tf.random.normal`.

### 3.1 Modelo

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)

INPUT_DIM, HIDDEN_DIM, LATENT_DIM = 784, 400, 20


class VAE(keras.Model):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        # Encoder
        self.enc_h = layers.Dense(hidden_dim, activation="relu")
        self.enc_mu = layers.Dense(latent_dim)        # mu
        self.enc_logvar = layers.Dense(latent_dim)    # log(sigma^2)
        # Decoder
        self.dec_h = layers.Dense(hidden_dim, activation="relu")
        self.dec_out = layers.Dense(input_dim, activation="sigmoid")

    def encode(self, x):
        h = self.enc_h(x)                              # (B, 400)
        return self.enc_mu(h), self.enc_logvar(h)      # (B, 20), (B, 20)

    def reparameterize(self, mu, logvar):
        std = tf.exp(0.5 * logvar)                     # sigma
        eps = tf.random.normal(tf.shape(std))          # eps ~ N(0, I), (B, 20)
        return mu + std * eps                          # Ec. (2)

    def decode(self, z):
        return self.dec_out(self.dec_h(z))             # (B, 784) en [0, 1]

    def call(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
```

### 3.2 Perdida y paso de entrenamiento

```python
def vae_loss(x_hat, x, mu, logvar):
    # BCE por pixel, sumada sobre pixeles, promediada... no:
    # sumamos pixeles, sumamos batch -> mismo criterio que PyTorch
    bce = keras.losses.binary_crossentropy(x, x_hat)   # (B,) ya suma pixeles? NO
    # keras.losses.binary_crossentropy promedia sobre el ultimo eje (pixeles).
    # Para SUMAR pixeles multiplicamos por la dim, o usamos la forma explicita:
    recon = tf.reduce_sum(
        keras.losses.binary_crossentropy(x, x_hat) * tf.cast(tf.shape(x)[-1], tf.float32)
    )
    # KL en forma cerrada, Ec. (4)
    kl = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar))
    return recon + kl, recon, kl
```

Cuidado con `keras.losses.binary_crossentropy`: **promedia** sobre el ultimo eje (los 784 pixeles), no los suma. Para igualar el criterio "BCE sumada" de PyTorch hay que multiplicar por el numero de pixeles (o reducir con `reduce_sum`). La alternativa mas limpia y sin ambiguedad es calcular la BCE binaria a mano:

```python
def recon_bce_sum(x_hat, x, eps=1e-8):
    # -[x*log(x_hat) + (1-x)*log(1-x_hat)] sumado sobre todo
    x_hat = tf.clip_by_value(x_hat, eps, 1.0 - eps)    # evita log(0)
    bce = -(x * tf.math.log(x_hat) + (1 - x) * tf.math.log(1 - x_hat))
    return tf.reduce_sum(bce)                          # suma pixeles y batch
```

El `clip_by_value` evita el `log(0)` cuando la sigmoide satura: es un gotcha clasico que da `NaN` en la perdida. PyTorch lo maneja internamente en `binary_cross_entropy`; en la version manual de TF/JAX hay que hacerlo explicito. El paso de entrenamiento con `GradientTape`:

```python
model = VAE()
optimizer = keras.optimizers.Adam(1e-3)

(x_train, _), _ = keras.datasets.mnist.load_data()
x_train = (x_train.reshape(-1, 784) / 255.0).astype("float32")  # [0,1], (60000, 784)
ds = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(128)

@tf.function
def train_step(x):
    with tf.GradientTape() as tape:
        x_hat, mu, logvar = model(x)
        recon = recon_bce_sum(x_hat, x)
        kl = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar))
        loss = recon + kl
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

for epoch in range(10):
    total = 0.0
    for x in ds:
        total += float(train_step(x))
    print(f"epoch {epoch:2d}  loss/img={total/len(x_train):.2f}")
```

El muestreo desde la prior es identico en espiritu al de PyTorch: `z = tf.random.normal((n, LATENT_DIM))` y luego `model.decode(z)`.

---

## 4. Seccion 3: JAX

JAX trabaja con **funciones puras** y maneja la aleatoriedad de forma explicita: no hay un generador global de numeros aleatorios; cada muestreo recibe una `PRNGKey`. Esto es perfecto para el reparameterize, donde el azar es justamente el ingrediente que queremos controlar. Implementamos el VAE con `jax.numpy` puro (sin Flax) para que cada operacion sea visible, parametros en un diccionario de pytrees.

### 4.1 Inicializacion de parametros

```python
import jax
import jax.numpy as jnp
from jax import random

INPUT_DIM, HIDDEN_DIM, LATENT_DIM = 784, 400, 20


def init_params(key):
    """Inicializa todas las matrices con He/Glorot escalado simple."""
    keys = random.split(key, 6)
    def dense(k, n_in, n_out):
        # W: (n_in, n_out), b: (n_out,)
        W = random.normal(k, (n_in, n_out)) * jnp.sqrt(2.0 / n_in)
        b = jnp.zeros((n_out,))
        return {"W": W, "b": b}
    return {
        "enc_h":      dense(keys[0], INPUT_DIM, HIDDEN_DIM),
        "enc_mu":     dense(keys[1], HIDDEN_DIM, LATENT_DIM),
        "enc_logvar": dense(keys[2], HIDDEN_DIM, LATENT_DIM),
        "dec_h":      dense(keys[3], LATENT_DIM, HIDDEN_DIM),
        "dec_out":    dense(keys[4], HIDDEN_DIM, INPUT_DIM),
    }


def dense_fwd(layer, x):
    return x @ layer["W"] + layer["b"]
```

### 4.2 Forward puro con `PRNGKey` para el reparameterize

```python
def encode(params, x):
    h = jax.nn.relu(dense_fwd(params["enc_h"], x))     # (B, 400)
    mu = dense_fwd(params["enc_mu"], h)                # (B, 20)
    logvar = dense_fwd(params["enc_logvar"], h)        # (B, 20)
    return mu, logvar


def reparameterize(key, mu, logvar):
    # Ec. (2): el azar entra por 'key', no por estado global
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(key, mu.shape)                 # eps ~ N(0, I), (B, 20)
    return mu + std * eps


def decode(params, z):
    h = jax.nn.relu(dense_fwd(params["dec_h"], z))     # (B, 400)
    logits = dense_fwd(params["dec_out"], h)           # (B, 784)
    return jax.nn.sigmoid(logits)                      # [0, 1]


def vae_forward(params, key, x):
    mu, logvar = encode(params, x)
    z = reparameterize(key, mu, logvar)                # key explicita
    x_hat = decode(params, z)
    return x_hat, mu, logvar
```

La diferencia conceptual con PyTorch/TF: el muestreo de $\varepsilon$ es una **funcion pura** de la `key`. La misma `key` da el mismo $\varepsilon$ siempre; para obtener ruido nuevo en cada paso, *dividimos* la key con `random.split`. Esto hace el reparameterize completamente reproducible y es lo que permite que `jax.grad` lo trate como deterministico (el azar es un argumento, no un efecto secundario).

### 4.3 Perdida y paso de entrenamiento

```python
import optax

def vae_loss(params, key, x, eps=1e-8):
    x_hat, mu, logvar = vae_forward(params, key, x)
    # Reconstruccion: BCE binaria a mano, sumada
    x_hat = jnp.clip(x_hat, eps, 1.0 - eps)            # evita log(0)
    bce = -(x * jnp.log(x_hat) + (1 - x) * jnp.log(1 - x_hat))
    recon = jnp.sum(bce)
    # KL en forma cerrada, Ec. (4)
    kl = -0.5 * jnp.sum(1 + logvar - jnp.square(mu) - jnp.exp(logvar))
    return recon + kl


key = random.PRNGKey(42)
params = init_params(key)
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)


@jax.jit
def train_step(params, opt_state, key, x):
    # value_and_grad diferencia SOLO respecto a params; key y x son constantes
    loss, grads = jax.value_and_grad(vae_loss)(params, key, x)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


# x_train: (60000, 784) en [0,1], cargado igual que en la version TF
for epoch in range(10):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, x_train.shape[0])
    total = 0.0
    for i in range(0, x_train.shape[0], 128):
        batch = x_train[perm[i:i + 128]]
        key, step_key = random.split(key)              # key fresca por paso
        params, opt_state, loss = train_step(params, opt_state, step_key, batch)
        total += float(loss)
    print(f"epoch {epoch:2d}  loss/img={total/x_train.shape[0]:.2f}")
```

El patron clave es la **cadena de keys**: en cada iteracion hacemos `key, step_key = random.split(key)` y pasamos `step_key` al paso. Si pasaramos siempre la misma key, el $\varepsilon$ del reparameterize seria identico en cada batch y perderiamos la estocasticidad que el VAE necesita. El muestreo desde la prior tambien usa una key:

```python
def sample(params, key, n=16):
    z = random.normal(key, (n, LATENT_DIM))            # z ~ N(0, I)
    return decode(params, z).reshape(n, 28, 28)        # n digitos nuevos
```

---

## 5. Verificacion de dimensiones de punta a punta

Las tres implementaciones son **isomorfas**: mismo MLP, mismo reparameterize, misma KL cerrada, misma BCE sumada. Esta tabla es el diccionario de traduccion del nucleo del VAE:

| Concepto | PyTorch | TensorFlow/Keras | JAX |
|---|---|---|---|
| Modelo | `class VAE(nn.Module)` | `class VAE(keras.Model)` | dict de params + funciones puras |
| Dos cabezas $\mu,\log\sigma^2$ | `fc_mu`, `fc_logvar` | `enc_mu`, `enc_logvar` | `params["enc_mu"]`, `["enc_logvar"]` |
| Ruido $\varepsilon$ | `torch.randn_like(std)` | `tf.random.normal(shape)` | `random.normal(key, shape)` |
| Reparameterize | `mu + std*eps` | `mu + std*eps` | `mu + std*eps` (key explicita) |
| Sigmoide salida | `torch.sigmoid` | `activation="sigmoid"` | `jax.nn.sigmoid` |
| BCE sumada | `F.binary_cross_entropy(..., reduction="sum")` | a mano con `clip` + `reduce_sum` | a mano con `clip` + `jnp.sum` |
| KL cerrada | `-0.5*torch.sum(1+logvar-mu**2-logvar.exp())` | `-0.5*tf.reduce_sum(...)` | `-0.5*jnp.sum(...)` |
| Gradiente | `loss.backward()` | `tf.GradientTape` | `jax.value_and_grad` |
| Azar | estado global (seed) | estado global (seed) | **`PRNGKey` explicita** |

Conviene hacer una verificacion de shapes con datos sinteticos antes de entrenar, en cualquiera de los tres frameworks. En PyTorch:

```python
x = torch.rand(8, 784)                  # batch falso
x_hat, mu, logvar = VAE()(x)
assert x_hat.shape == (8, 784)          # reconstruccion del tamano de la entrada
assert mu.shape == (8, 20)              # latente
assert logvar.shape == (8, 20)
loss, recon, kl = vae_loss(x_hat, x, mu, logvar)
assert loss.ndim == 0                   # escalar
print("shapes OK, loss =", loss.item())
```

Si esto pasa, la mecanica esta bien cableada; lo que falte sera ajuste de hiperparametros, no un bug estructural.

---

## 6. Que esperar al entrenar

Con 10 epocas de MLP sobre MNIST y `LATENT_DIM=20`, la perdida por imagen baja a la zona de **~100-110 nats** (la BCE sumada de 784 pixeles domina; la KL aporta del orden de 15-25 nats). Numeros de referencia para sanity-check:

| Epoca | loss/img aprox. | Lo que ves |
|---:|---:|---|
| 0 | ~190 | reconstrucciones grises, muestras = ruido informe |
| 3 | ~120 | digitos reconocibles pero borrosos |
| 10 | ~105 | reconstrucciones nitidas; muestras de $\mathcal{N}(0,I)$ ya parecen digitos |

Si la loss no baja de ~180, el sospechoso numero uno es la **escala** de los dos terminos (mezclar `mean` y `sum`). Si baja muy rapido pero las muestras desde la prior son basura, probablemente la KL esta con el signo cambiado o multiplicada por cero (verifica el caso degenerado de la seccion 1.1).

---

## 7. Por que funciona: las tres preguntas que cierran el VAE

### 7.1 Por que la KL regulariza el latente

Sin el termino KL, el VAE seria un autoencoder estocastico cualquiera: el encoder podria colocar los codigos de cada digito en regiones arbitrarias y lejanas del espacio, dejando enormes "huecos" entre clusters. Muestrear de $\mathcal{N}(0,I)$ caeria en esos huecos —zonas que el decoder nunca vio— y produciria basura.

La KL hace dos cosas a la vez. Primero, **tira cada posterior $q_\phi(z\mid x)$ hacia $\mathcal{N}(0,I)$**: penaliza medias $\mu$ lejos del origen ($\mu^2$ en la Ec. 4) y varianzas muy chicas o muy grandes (los terminos $\log\sigma^2$ y $\sigma^2$ se balancean en $\sigma^2=1$). Segundo, al forzar varianzas no triviales, hace que las distribuciones de digitos vecinos **se solapen**: no hay huecos, el espacio queda densamente cubierto. El resultado es un latente **continuo y completo**, donde cualquier punto razonable decodifica a algo plausible. Esa es exactamente la propiedad que un autoencoder normal no garantiza, y es lo que separa un VAE (modelo *generativo*) de un autoencoder (modelo de *compresion*).

El equilibrio es delicado: si subes el peso de la KL (es el famoso $\beta$-VAE con $\beta>1$), el latente se vuelve mas estructurado y "desenredado" pero las reconstrucciones empeoran; si lo bajas, reconstruye mejor pero pierde la capacidad de muestrear. El VAE estandar usa $\beta=1$, que es justo el ELBO de la Ecuacion (3).

### 7.2 Por que las muestras salen algo borrosas

Las imagenes de un VAE son notoriamente mas borrosas que las de una GAN, y la culpa es de la **forma de la perdida de reconstruccion**. Tanto la BCE como el MSE corresponden a verosimilitudes con ruido independiente por pixel (Bernoulli o gaussiana, respectivamente). Bajo una perdida tipo MSE/gaussiana, el optimo ante incertidumbre es **predecir el promedio** de todas las salidas plausibles. Si dado un latente hay varios digitos igualmente validos (por ejemplo, un 4 que podria cerrarse en 9), el decoder minimiza el error prediciendo algo intermedio: un promedio difuso de ambos. Ese promediado pixel-a-pixel es literalmente el borroneo.

Dicho de otro modo: la verosimilitud gaussiana/Bernoulli **no penaliza la falta de detalle de alta frecuencia** tanto como castiga errores de baja frecuencia. Bordes nitidos y texturas finas se sacrifican porque equivocarse en un borde cuesta poco en MSE. Las GANs evitan esto reemplazando la perdida por pixel por un discriminador que castiga *cualquier* imagen que no parezca real, incluido lo borroso —de ahi su nitidez, a costa de un entrenamiento mucho mas inestable. Es uno de los vertices del *trilema generativo* que la clase 29 desarrolla: el VAE ofrece muestreo rapido y latente interpretable, a costa de calidad de imagen.

### 7.3 Como interpolar en el espacio latente

Aqui se ve la magia del latente continuo. Tomamos dos imagenes reales, las codificamos a sus medias latentes $\mu_A$ y $\mu_B$ (usamos $\mu$, no $z$ muestreado, para una interpolacion deterministica), recorremos la recta entre ambas y decodificamos cada punto intermedio:

```python
@torch.no_grad()
def interpolate(model, x_a, x_b, steps=10):
    model.eval()
    mu_a, _ = model.encode(x_a.view(1, -1))      # (1, 20)
    mu_b, _ = model.encode(x_b.view(1, -1))      # (1, 20)
    # alphas de 0 a 1: recta z = (1-a)*mu_a + a*mu_b
    alphas = torch.linspace(0, 1, steps).view(-1, 1)   # (steps, 1)
    zs = (1 - alphas) * mu_a + alphas * mu_b           # (steps, 20)
    recons = model.decode(zs)                          # (steps, 784)
    return recons.view(steps, 1, 28, 28)
```

Si interpolas entre un 3 y un 8, veras una transicion **suave** —el digito se va deformando de uno a otro pasando por formas intermedias plausibles, no por un salto brusco ni por ruido. Esa continuidad es la prueba visual de que la KL hizo su trabajo: el espacio no tiene huecos entre los dos puntos. En un autoencoder sin regularizacion, la misma interpolacion atravesaria regiones vacias y produciria imagenes degeneradas a mitad de camino.

Un ejercicio revelador: en vez de interpolar entre dos imagenes, recorre **una sola dimension latente** dejando las otras 19 fijas (`z[:, j] = linspace(-3, 3)`). Muchas dimensiones codificaran atributos interpretables —grosor del trazo, inclinacion, ancho— precisamente porque la KL empujo el latente hacia ejes independientes alineados con la prior $\mathcal{N}(0,I)$ diagonal.

---

## 8. Limitaciones y hacia donde sigue

El VAE es el cimiento de los modelos generativos modernos, pero tiene techo:

1. **Borrosidad** (seccion 7.2): la verosimilitud por pixel limita la nitidez. Las GANs y los modelos de difusion la superan con criterios de calidad distintos.
2. **Posterior collapse:** con decoders muy potentes, el modelo a veces ignora el latente (la KL colapsa a 0 y $z$ no aporta informacion). Tecnicas como *KL annealing* o $\beta < 1$ inicial lo mitigan.
3. **Latente continuo, no discreto:** para datos con estructura simbolica (tokens, codigos), el [VQ-VAE](/papers/vq-vae-oord-2017) reemplaza la gaussiana por un *codebook* discreto, base de generadores de imagen y audio de ultima generacion.

El siguiente eslabon en la clase son los **modelos de difusion**, que pueden verse como una pila de muchos pasos de "denoising" —conceptualmente, una jerarquia de VAEs encadenados— y que hoy dominan la generacion de imagenes (Stable Diffusion) precisamente porque resuelven el problema de la borrosidad sin la inestabilidad de las GANs.

---

**Ver tambien:** [Clase 29 - Modelos Generativos en Vision](/clases/clase-29) · [Fundamento de modelos generativos](/fundamentos/modelos-generativos) · [Paper VAE (Kingma y Welling 2013)](/papers/vae-kingma-2013) · [Paper VQ-VAE (van den Oord et al. 2017)](/papers/vq-vae-oord-2017) · [Teoria de la clase](../teoria) · [Profundizacion matematica](../profundizacion).
