---
title: "Inflar una CNN 2D a 3D desde cero"
weight: 1
math: true
---

La [teoría de la Clase 38](/clases/clase-38/teoria) resume la desventaja que define a C3D en una frase: **"no puede aprovechar el pre-entrenamiento de ImageNet"**. Y aclara que no es una limitación de implementación sino un problema de **forma de los tensores**: un kernel de ImageNet es una matriz $k \times k$, un kernel 3D necesita $t \times k \times k$, y no hay manera de cargar el primero en el segundo. Este capítulo construye la solución de I3D —el **inflado**— desde cero, y hace lo que ningún slide hace: **verifica numéricamente** que el punto fijo del video aburrido se cumple. Vamos a inflar una capa, después una ResNet-18 pre-entrenada completa, y en el camino vamos a encontrar el borde exacto donde la igualdad deja de ser exacta.

> **Lecturas de apoyo:** el fundamento [Inflado de Convoluciones](/fundamentos/inflado-de-convoluciones); la [profundización](/clases/clase-38/profundizacion) deriva en su Parte I la condición de punto fijo que este código implementa, y el paper [I3D](/papers/i3d-carreira-2017) es la fuente original.

---

## 1. La condición de punto fijo, en una función

La profundización demuestra que la equivalencia entre la red 2D y su versión inflada sobre un video aburrido se cumple **si y solo si** los pesos inflados suman el peso 2D a lo largo del eje temporal:

$$\sum_{\tau=1}^{N} \widetilde{W}[c_o, c_i, \tau, u, v] \;=\; W[c_o, c_i, u, v]$$

Esto es una condición **sobre la suma**, no sobre cada peso. Tiene infinitas soluciones; dos aparecen en implementaciones reales. Empecemos en NumPy puro, sin frameworks, para que la matemática quede a la vista.

```python
import numpy as np

def inflate_kernel(W2d, N, mode="uniform"):
    """Infla un kernel 2D al eje temporal cumpliendo sum_tau W3d[:, :, tau] == W2d.

    W2d : array (C_out, C_in, kh, kw)  -- convención PyTorch
    N    : extensión temporal del kernel inflado
    mode : "uniform" (reparto W/N, el de I3D) o "center" (delta de Dirac temporal)
    →     array (C_out, C_in, N, kh, kw)
    """
    if N < 1:
        raise ValueError("N debe ser >= 1")
    W2d = np.asarray(W2d)
    if mode == "uniform":
        # repite el kernel N veces en el nuevo eje 2 y reparte la masa entre los taps
        return np.repeat(W2d[:, :, None, :, :], N, axis=2) / N
    if mode == "center":
        W3d = np.zeros(W2d.shape[:2] + (N,) + W2d.shape[2:], dtype=W2d.dtype)
        tau = int(np.ceil(N / 2)) - 1        # índice 0-based del tap central
        W3d[:, :, tau, :, :] = W2d           # todo el peso en un solo tap
        return W3d
    raise ValueError(f"modo desconocido: {mode!r}")
```

La verificación es una línea: sumar sobre el eje 2 y comparar con el original.

```python
rng = np.random.default_rng(38)
W2d = rng.normal(size=(4, 3, 3, 3)).astype(np.float32)   # 4 filtros, 3 canales, 3x3

for N in (1, 3, 5, 7):
    for mode in ("uniform", "center"):
        W3d = inflate_kernel(W2d, N, mode)
        ok = np.allclose(W3d.sum(axis=2), W2d, atol=1e-6)
        print(f"N={N} mode={mode:8s} forma={W3d.shape} suma_temporal==W2d -> {ok}")
```

```text
N=1 mode=uniform  forma=(4, 3, 1, 3, 3) suma_temporal==W2d -> True
N=1 mode=center   forma=(4, 3, 1, 3, 3) suma_temporal==W2d -> True
N=3 mode=uniform  forma=(4, 3, 3, 3, 3) suma_temporal==W2d -> True
N=3 mode=center   forma=(4, 3, 3, 3, 3) suma_temporal==W2d -> True
N=5 mode=uniform  forma=(4, 3, 5, 3, 3) suma_temporal==W2d -> True
N=5 mode=center   forma=(4, 3, 5, 3, 3) suma_temporal==W2d -> True
N=7 mode=uniform  forma=(4, 3, 7, 3, 3) suma_temporal==W2d -> True
N=7 mode=center   forma=(4, 3, 7, 3, 3) suma_temporal==W2d -> True
```

Los dos modos cumplen la condición, pero **no son el mismo tensor**. La norma lo delata: el reparto uniforme distribuye la masa y la achica, la delta la concentra intacta.

```python
W3u, W3c = inflate_kernel(W2d, 3, "uniform"), inflate_kernel(W2d, 3, "center")
print(round(float(np.linalg.norm(W3u)), 4),   # → 6.4256
      round(float(np.linalg.norm(W3c)), 4),   # → 11.1294
      round(float(np.linalg.norm(W2d)), 4))   # → 11.1294
```

{{< concept-alert type="clave" >}}
**La delta central conserva la norma del kernel 2D; el reparto uniforme la divide por $\sqrt{N}$.** Ambos producen la misma salida sobre un video aburrido, pero arrancan el entrenamiento en puntos distintos del espacio de pesos: el uniforme actúa como un pasa-bajos temporal (promedia antes de convolucionar), la delta es literalmente la red 2D aplicada frame a frame. La división por $N$ del slide es **la elección de I3D, no una necesidad matemática**.
{{< /concept-alert >}}

---

## 2. PyTorch: inflar una capa y verificar el punto fijo

Ahora la misma función aplicada a una capa real. Tres cosas hay que trasladar: el kernel inflado, el **bias sin tocar**, y la geometría (stride y padding espaciales se copian; el temporal se elige).

```python
import torch, torch.nn as nn

def inflate_conv2d(conv2d, N=3, mode="uniform"):
    kh, kw = conv2d.kernel_size
    sh, sw = conv2d.stride
    ph, pw = conv2d.padding
    conv3d = nn.Conv3d(conv2d.in_channels, conv2d.out_channels,
                       kernel_size=(N, kh, kw),
                       stride=(1, sh, sw),        # stride temporal 1: I3D no decima el tiempo temprano
                       padding=(N // 2, ph, pw),  # "same" temporal, con ceros
                       bias=conv2d.bias is not None)
    W2d = conv2d.weight.detach().numpy()
    with torch.no_grad():
        conv3d.weight.copy_(torch.from_numpy(inflate_kernel(W2d, N, mode)))
        if conv2d.bias is not None:
            conv3d.bias.copy_(conv2d.bias.detach())   # el bias NO se escala
    return conv3d
```

El **video aburrido** es una sola línea: agregar un eje temporal y repetir la imagen.

```python
torch.manual_seed(38)
conv2d = nn.Conv2d(3, 8, kernel_size=3, padding=1)
conv3d = inflate_conv2d(conv2d, N=3, mode="uniform")
print(tuple(conv2d.weight.shape), "->", tuple(conv3d.weight.shape))
# → (8, 3, 3, 3) -> (8, 3, 3, 3, 3)

T = 5
x = torch.randn(2, 3, 16, 16)                    # [B, C, H, W]
video = x.unsqueeze(2).repeat(1, 1, T, 1, 1)     # [B, C, T, H, W]  <- video aburrido

with torch.no_grad():
    y2d, y3d = conv2d(x), conv3d(video)

for t in range(T):
    err = float((y3d[:, :, t] - y2d).abs().max())
    print(f"  t={t}  err_max={err:.6f}")
```

```text
  t=0  err_max=0.749047
  t=1  err_max=0.000001
  t=2  err_max=0.000001
  t=3  err_max=0.000001
  t=4  err_max=0.749047
```

Ahí está el resultado central del capítulo, y no es el que uno esperaría de leer solamente el slide.

{{< concept-alert type="advertencia" >}}
**El punto fijo se cumple exactamente en el interior temporal, no en los bordes.** La derivación supone que la entrada es constante **en toda la ventana del kernel**. En $t=0$ y $t=T-1$ la ventana incluye padding de ceros, que no es la imagen: la suma efectiva se hace sobre $N-1$ taps en lugar de $N$, así que la activación queda escalada por $\frac{N-1}{N}$ más el sesgo. En los frames interiores el error es $10^{-6}$ —ruido de `float32`—; en los bordes es $0{,}75$. Con clips de 64 frames y kernel 3 el efecto es marginal; con clips de 8 frames y kernel temporal 7, no lo es.
{{< /concept-alert >}}

Los dos modos coinciden sobre el video aburrido y **difieren sobre video real** —exactamente lo que predice la profundización:

```python
conv3dc = inflate_conv2d(conv2d, N=3, mode="center")
real = torch.randn(2, 3, T, 16, 16)
with torch.no_grad():
    print(torch.allclose(y3d[:, :, 2], conv3dc(video)[:, :, 2], atol=1e-6))  # → True
    print(float((conv3d(real) - conv3dc(real)).abs().mean()))                # → 0.347
```

Sobre el video aburrido son indistinguibles; sobre movimiento real difieren en promedio $0{,}35$ por activación. Es la misma función en el punto fijo y dos funciones distintas fuera de él.

---

## 3. Inflar una red completa pre-entrenada

Una capa no prueba nada: lo que interesa es que la propiedad **se propague por toda la red**. Recorremos recursivamente un modelo de torchvision y sustituimos cada módulo por su versión 3D.

```python
import copy
from torchvision.models import resnet18, ResNet18_Weights

def _pair(v): return v if isinstance(v, tuple) else (v, v)

def inflate_bn(b):
    """BatchNorm2d -> BatchNorm3d: se copian parámetros Y estadísticas acumuladas."""
    b3 = nn.BatchNorm3d(b.num_features, eps=b.eps, momentum=b.momentum)
    with torch.no_grad():
        b3.weight.copy_(b.weight); b3.bias.copy_(b.bias)
        b3.running_mean.copy_(b.running_mean); b3.running_var.copy_(b.running_var)
        b3.num_batches_tracked.copy_(b.num_batches_tracked)
    return b3

def inflate_conv(c, N, mode):
    kh, kw = _pair(c.kernel_size); sh, sw = _pair(c.stride); ph, pw = _pair(c.padding)
    Nt = 1 if (kh == 1 and kw == 1) else N     # las convs 1x1 NO se inflan
    c3 = nn.Conv3d(c.in_channels, c.out_channels, (Nt, kh, kw),
                   stride=(1, sh, sw), padding=(Nt // 2, ph, pw),
                   bias=c.bias is not None)
    with torch.no_grad():
        c3.weight.copy_(torch.from_numpy(
            inflate_kernel(c.weight.detach().numpy(), Nt, mode)))
        if c.bias is not None: c3.bias.copy_(c.bias.detach())
    return c3

def inflate_module_(m, N, mode):
    for name, child in m.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(m, name, inflate_conv(child, N, mode))
        elif isinstance(child, nn.BatchNorm2d):
            setattr(m, name, inflate_bn(child))
        elif isinstance(child, nn.MaxPool2d):
            kh, kw = _pair(child.kernel_size); sh, sw = _pair(child.stride)
            ph, pw = _pair(child.padding)
            setattr(m, name, nn.MaxPool3d((N, kh, kw), stride=(1, sh, sw),
                                          padding=(N // 2, ph, pw)))   # sin reescalar
        elif isinstance(child, nn.AdaptiveAvgPool2d):
            setattr(m, name, nn.AdaptiveAvgPool3d((1, 1, 1)))
        else:
            inflate_module_(child, N, mode)     # recursión: Sequential, BasicBlock, downsample
    return m

def inflate_model(model2d, N=3, mode="uniform"):
    return inflate_module_(copy.deepcopy(model2d), N, mode)
```

Un detalle no cosmético: las convoluciones $1\times1$ se inflan con $N_t = 1$, así que su condición de punto fijo se satisface **sin división alguna** —se copian los pesos tal cual. Es el mismo argumento que la profundización usa para el kernel espacial de las arquitecturas $(2+1)$D.

La verificación end-to-end: el logit de la red inflada sobre el video aburrido contra el logit de la red 2D sobre la imagen.

```python
torch.manual_seed(38)
net2d = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()   # eval() es crítico
x = torch.randn(1, 3, 112, 112); T = 8
video = x.unsqueeze(2).repeat(1, 1, T, 1, 1)

with torch.no_grad(): l2 = net2d(x)
for mode in ("uniform", "center"):
    net3d = inflate_model(net2d, N=3, mode=mode).eval()
    with torch.no_grad(): l3 = net3d(video)
    print(f"mode={mode:8s} err_max={float((l3-l2).abs().max()):.3e} "
          f"top1_2d={int(l2.argmax())} top1_3d={int(l3.argmax())}")
```

```text
mode=uniform  err_max=2.764e+00 top1_2d=5 top1_3d=5
mode=center   err_max=9.537e-07 top1_2d=5 top1_3d=5
```

El modo `center` reproduce el logit **exactamente** ($10^{-7}$, precisión de máquina). El modo `uniform` se desvía en $2{,}76$ logits. La causa es la de la sección anterior: el *global average pooling* final promedia sobre los $T$ frames e **incluye los bordes contaminados**. Y si esa es la causa, el error debe diluirse como $1/T$. Lo es:

| $T$ | `uniform` err. máx. | `center` err. máx. |
|---|---|---|
| 4 | 5,2975 | 0,00e+00 |
| 8 | 2,7640 | 9,54e-07 |
| 16 | 1,3952 | 9,54e-07 |
| 32 | 0,6976 | 9,54e-07 |
| 64 | 0,3488 | 9,54e-07 |

Cada duplicación de $T$ reduce el error a la mitad, exactamente. Es la confirmación empírica de que el único culpable es el padding temporal: hay un número **fijo** de frames corrompidos y el promedio global los diluye.

{{< concept-alert type="clave" >}}
**La delta central es inmune al padding temporal, y por eso el punto fijo le sale exacto.** Con `mode="center"` los taps vecinos valen $0$, así que los ceros del padding se multiplican por $0$ y nunca entran en la suma: la igualdad se cumple en **todos** los frames, incluidos los bordes. El max-pooling tampoco la rompe, porque después de un ReLU todas las activaciones son $\geq 0$ y un cero de padding no puede ganar un máximo. Si necesitas equivalencia bit a bit al cargar un checkpoint inflado —por ejemplo para un test de regresión— usa la delta central, no el reparto uniforme.
{{< /concept-alert >}}

El costo del inflado, medido:

```python
p2 = sum(p.numel() for p in net2d.parameters())
p3 = sum(p.numel() for p in inflate_model(net2d, N=3).parameters())
print(f"2D={p2/1e6:.2f}M  3D={p3/1e6:.2f}M  factor={p3/p2:.2f}")
# → 2D=11.69M  3D=33.68M  factor=2.88
```

El factor es $2{,}88$ y no $3$ porque las convoluciones $1\times1$, los BatchNorm y la capa densa final no se multiplican. Coincide con la Parte II de la profundización: **el inflado multiplica los pesos convolucionales por $N$**, no los ahorra.

---

## 4. Los tres gotchas de implementación

### 4.1 El max-pooling no se divide por $N$

```python
relu = nn.ReLU()
h2d, h3d = relu(y2d), relu(y3d)
mp2 = nn.MaxPool2d(2)
mp3 = nn.MaxPool3d((3, 2, 2), stride=(1, 2, 2), padding=(1, 0, 0))
with torch.no_grad(): p2_, p3_ = mp2(h2d), mp3(h3d)

print(torch.allclose(p3_[:, :, 2], p2_, atol=1e-6))       # → True   (sin reescalar)
print(torch.allclose(p3_[:, :, 2] / 3, p2_, atol=1e-6))   # → False  (err máx. 1.5047)
```

{{< concept-alert type="advertencia" >}}
**Dividir por $N$ aplica solo a operaciones que suman sobre el eje temporal.** El máximo de $N$ copias idénticas ya *es* la copia: $\max_\tau \tilde{x}[t+\tau] = x$. Inflar un max-pool es extender la ventana al tiempo y nada más. El mismo razonamiento aplica al *average pooling*: su normalización por el tamaño de la ventana ya está incorporada, así que tampoco se reescala. La división por $N$ pertenece **exclusivamente a los pesos convolucionales**, porque son los únicos que aparecen dentro de una suma no normalizada. Es el error más frecuente al implementar el inflado a mano.
{{< /concept-alert >}}

### 4.2 BatchNorm: `eval()` vs `train()`

```python
net3d = inflate_model(net2d, N=3, mode="center").train()      # <-- train(), a propósito
with torch.no_grad(): lt = net3d(video)
print(round(float((lt - l2).abs().max()), 3))                 # → 9.379

net2d_t = copy.deepcopy(net2d).train()
with torch.no_grad(): l2t = net2d_t(x)
print(round(float((l2t - l2).abs().max()), 3))                # → 9.379
print(round(float((lt - l2t).abs().max()), 6))                # → 1e-06
```

Las tres cifras cuentan una historia precisa. En `train()` el error contra la referencia salta a $9{,}4$ logits, pero la red **2D** en `train()` se desvía **exactamente lo mismo** de sí misma en `eval()`, y la 3D en `train()` coincide con la 2D en `train()` hasta $10^{-6}$. Es decir: en `train()` no se rompió el inflado, se rompió la *referencia*. Con `train()` y un lote de una imagen, BatchNorm descarta las estadísticas de ImageNet y normaliza con las del lote; la red deja de calcular la función pre-entrenada, y además **muta `running_mean` y `running_var` en cada forward**, degradando el checkpoint en silencio.

{{< concept-alert type="clave" >}}
**Por qué `eval()` es crítico.** Solo en `eval()` BatchNorm usa las estadísticas heredadas de ImageNet, que es lo único que hace bien definida la comparación 2D contra 3D. Hay un detalle fino que vale conocer: sobre un video **exactamente** constante en el tiempo, la media y varianza sobre el volumen $(B, T, H, W)$ **coinciden** con las de la imagen $(B, H, W)$ —son $T$ copias del mismo tensor—, así que el punto fijo sobrevive incluso en `train()`. Con el modo `uniform` no sobrevive: los bordes contaminados desbalancean las estadísticas del volumen y el error contra la 2D en `train()` es $0{,}308$. Y sobre **video real** las estadísticas del volumen son genuinamente distintas de las de imagen, así que **hay que re-estimarlas durante el fine-tuning**. Ver [Regularización](/fundamentos/regularizacion).
{{< /concept-alert >}}

### 4.3 El bias no se toca

$b[c_o]$ aparece **una sola vez** en la ecuación de la convolución 3D, fuera de la suma sobre $\tau$: no depende del tiempo, así que nada hay que repartir. Dividirlo por $N$ es un error tentador —"si escalé los pesos, escalo todo"— que introduce un desplazamiento de $b(1 - 1/N)$ en cada canal y rompe el punto fijo sin dar ningún error de forma. Lo mismo vale para $\gamma$ y $\beta$ de BatchNorm.

---

## 5. TensorFlow / Keras: cuidado con el orden de los ejes

Todo lo anterior es idéntico en Keras salvo un punto que hay que mirar dos veces: la convención de ejes. Keras es `channels_last`, y su kernel guarda las dimensiones espaciales **primero** y los canales al final. El eje temporal se inserta **al principio**, no en el medio.

| Framework | Kernel 2D | Kernel 3D inflado | Eje temporal |
|---|---|---|---|
| PyTorch | `(C_out, C_in, kh, kw)` | `(C_out, C_in, N, kh, kw)` | `axis=2` |
| Keras / Flax | `(kh, kw, C_in, C_out)` | `(N, kh, kw, C_in, C_out)` | `axis=0` |

```python
import numpy as np, tensorflow as tf

def inflate_kernel_last(W2d, N, mode="uniform"):
    """channels_last: (kh, kw, C_in, C_out) -> (N, kh, kw, C_in, C_out)."""
    W2d = np.asarray(W2d)
    if mode == "uniform":
        return np.repeat(W2d[None, ...], N, axis=0) / N       # <-- axis=0, no axis=2
    W3d = np.zeros((N,) + W2d.shape, dtype=W2d.dtype)
    W3d[int(np.ceil(N / 2)) - 1] = W2d
    return W3d

tf.keras.utils.set_random_seed(38)   # en Keras 3, tf.random.set_seed NO siembra los inicializadores
N, T = 3, 5
conv2d = tf.keras.layers.Conv2D(8, 3, padding="same")
conv3d = tf.keras.layers.Conv3D(8, (N, 3, 3), padding="same")

img   = tf.random.normal((2, 16, 16, 3))            # (B, H, W, C)
video = tf.repeat(img[:, None, ...], T, axis=1)     # (B, T, H, W, C) <- video aburrido
_ = conv2d(img); _ = conv3d(video)                  # construye los pesos

W2, b2 = conv2d.get_weights()                       # Keras 3 devuelve arrays de NumPy
print(W2.shape, conv3d.get_weights()[0].shape)      # → (3, 3, 3, 8) (3, 3, 3, 3, 8)

W3 = inflate_kernel_last(W2, N, "uniform")
print(np.allclose(W3.sum(axis=0), W2, atol=1e-6))   # → True  (condición de punto fijo)
conv3d.set_weights([W3, b2])                        # el bias se copia sin tocar

y2, y3 = conv2d(img).numpy(), conv3d(video).numpy()
for t in range(T):
    print(f"  t={t} err_max={float(np.abs(y3[:, t] - y2).max()):.6f}")
```

```text
  t=0 err_max=0.853675
  t=1 err_max=0.000001
  t=2 err_max=0.000001
  t=3 err_max=0.000001
  t=4 err_max=0.853675
```

El mismo patrón: interior exacto, bordes desviados. Y el mismo remedio con la delta central, que en Keras también sale exacta en todos los frames:

```python
conv3d.set_weights([inflate_kernel_last(W2, N, "center"), b2])
y3c = conv3d(video).numpy()
print(np.allclose(y3c[:, T // 2], y2, atol=1e-5),  # → True  (interior)
      np.allclose(y3c[:, 0],      y2, atol=1e-5))  # → True  (borde: inmune al padding)
```

{{< concept-alert type="advertencia" >}}
**El gotcha de portar código de inflado entre frameworks.** `np.repeat(W[:, :, None], N, axis=2)` en PyTorch y `np.repeat(W[None, ...], N, axis=0)` en Keras/Flax hacen conceptualmente lo mismo, pero intercambiarlos no da un error de forma sino un tensor **de rango correcto y semántica equivocada** —en Keras, `axis=2` replicaría el eje de canales de entrada. El código corre, la red entrena, y el punto fijo simplemente no se cumple. Verificar la condición con `np.allclose(W3.sum(axis=eje_temporal), W2)` antes de entrenar cuesta una línea y detecta este error de inmediato.
{{< /concept-alert >}}

---

## 6. JAX / Flax

Flax comparte la convención de Keras: el kernel de `flax.linen.Conv` tiene forma `(*kernel_size, C_in, C_out)`, así que el inflado es el de la sección anterior. La diferencia está en que los parámetros son un **diccionario anidado inmutable**, y la sustitución del kernel se hace aplanando ese árbol con `flax.traverse_util`.

```python
import jax, jax.numpy as jnp, numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict

class Conv2DNet(nn.Module):
    @nn.compact
    def __call__(self, x):                      # (B, H, W, C)
        return nn.Conv(features=8, kernel_size=(3, 3), padding="SAME", name="conv")(x)

class Conv3DNet(nn.Module):
    N: int = 3
    @nn.compact
    def __call__(self, x):                      # (B, T, H, W, C)
        return nn.Conv(features=8, kernel_size=(self.N, 3, 3), padding="SAME", name="conv")(x)

k_init, k_data = jax.random.split(jax.random.key(38))
img   = jax.random.normal(k_data, (2, 16, 16, 3))
T = 5
video = jnp.repeat(img[:, None, ...], T, axis=1)      # video aburrido

net2d, net3d = Conv2DNet(), Conv3DNet(N=3)
p2 = net2d.init(k_init, img)                          # inicialización explícita
p3 = net3d.init(k_init, video)
print(p2["params"]["conv"]["kernel"].shape,           # → (3, 3, 3, 8)
      p3["params"]["conv"]["kernel"].shape)           # → (3, 3, 3, 3, 8)

W2 = np.asarray(p2["params"]["conv"]["kernel"])
b2 = np.asarray(p2["params"]["conv"]["bias"])

flat = flatten_dict(p3)                               # árbol -> dict de tuplas
flat[("params", "conv", "kernel")] = jnp.asarray(inflate_kernel_last(W2, 3, "uniform"))
flat[("params", "conv", "bias")]   = jnp.asarray(b2)  # el bias, intacto
p3 = unflatten_dict(flat)

print(np.allclose(np.asarray(p3["params"]["conv"]["kernel"]).sum(axis=0), W2, atol=1e-6))
# → True

y2, y3 = net2d.apply(p2, img), net3d.apply(p3, video)
for t in range(T):
    print(f"  t={t} err_max={float(jnp.abs(y3[:, t] - y2).max()):.6f}")
```

```text
  t=0 err_max=1.170400
  t=1 err_max=0.000001
  t=2 err_max=0.000001
  t=3 err_max=0.000001
  t=4 err_max=1.170400
```

Idéntico a los otros dos: interior exacto, bordes desviados. Tres frameworks, tres convenciones de ejes, una sola matemática.

{{< concept-alert type="recordar" >}}
En versiones de Flax anteriores a la 0.10 los parámetros venían envueltos en un `FrozenDict` y había que llamar a `flax.core.unfreeze` antes de modificarlos. `flatten_dict` / `unflatten_dict` funcionan con las dos convenciones, así que es la forma portable de hacer la sustitución. Verificado con `jax 0.11.0` y `flax 0.12.8`.
{{< /concept-alert >}}

---

## 7. Qué nos llevamos

- El inflado resuelve un problema de **forma de tensores**, no de arquitectura: la condición es $\sum_\tau \widetilde{W}[\cdot,\tau,\cdot] = W$, y tiene infinitas soluciones. `uniform` (I3D) y `center` (delta) son dos, y **se comportan distinto fuera del punto fijo**.
- El punto fijo se cumple **exactamente en el interior temporal**. Con reparto uniforme y padding de ceros, el error end-to-end de una ResNet-18 inflada es de $2{,}76$ logits con $T=8$ y decae como $1/T$ —hasta $0{,}35$ con $T=64$. El slide dice "exactamente igual"; la implementación dice "exactamente igual salvo $2\lfloor N/2 \rfloor$ frames".
- La **delta central es inmune al padding temporal** y reproduce el logit 2D a precisión de máquina ($10^{-7}$) en todos los frames. Es la inicialización a elegir si se quiere equivalencia verificable.
- **La división por $N$ es solo para los pesos convolucionales.** Max-pooling, average pooling, bias, $\gamma$ y $\beta$ de BatchNorm se copian tal cual. Las convoluciones $1\times1$ se inflan con $N_t=1$ y tampoco se dividen.
- `model.eval()` no es higiene, es parte de la definición del experimento: en `train()` BatchNorm reemplaza las estadísticas de ImageNet por las del lote y **muta el checkpoint en cada forward**. Sobre video real esas estadísticas sí cambian y hay que re-estimarlas al hacer fine-tuning.
- El inflado **triplica** los pesos convolucionales ($11{,}69$M $\to$ $33{,}68$M en ResNet-18, factor $2{,}88$). El ahorro de parámetros que el slide le atribuye a I3D viene de la topología de Inception, no del inflado.

La consecuencia de diseño está en la Parte III de la profundización: si la delta central es más limpia que el reparto uniforme, y si un kernel $1\times k \times k$ se infla sin dividir nada, entonces la factorización $(2+1)$D es la forma **natural** de inflar —y es la razón por la que S3D y R(2+1)D desplazaron a I3D como punto de partida estándar.

---

**Ver tambien:** [Clase 38 - Teoria](/clases/clase-38/teoria) · [Clase 38 - Profundizacion](/clases/clase-38/profundizacion) · [Inflado de Convoluciones](/fundamentos/inflado-de-convoluciones) · [I3D](/papers/i3d-carreira-2017)
