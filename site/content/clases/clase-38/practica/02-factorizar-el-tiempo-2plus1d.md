---
title: "Factorizar el tiempo: bloques (2+1)D desde cero"
weight: 2
math: true
---

La tabla de I3D en la [teoría](/clases/clase-38/teoria) cierra con tres desventajas: **"tiene una gran cantidad de parámetros"**, **"es computacionalmente costoso"** y **"la inferencia no es más rápida que los modelos anteriores"**. La respuesta que dieron [S3D](/papers/s3d-xie-2018) y [R(2+1)D](/papers/r2plus1d-tran-2018) en 2018 es la misma idea vista desde dos backbones distintos: **descomponer el kernel cúbico** $t \times k \times k$ en una convolución espacial $1 \times k \times k$ seguida de una temporal $t \times 1 \times 1$, con una no linealidad en medio. Este capítulo implementa ese bloque en los tres frameworks, verifica con código la fórmula de canales intermedios de la [profundización](/clases/clase-38/profundizacion) —la que hace justa la comparación con 3D— y mide el costo real en wall-clock. El resultado de esa medición no es el esperado, y es lo más útil del capítulo.

> **Lecturas de apoyo:** las Partes II y III de la [profundización](/clases/clase-38/profundizacion); el fundamento [Inflado de Convoluciones](/fundamentos/inflado-de-convoluciones); el [camino 01](/clases/clase-38/practica/01-inflar-una-cnn-2d-a-3d) sobre el inflado de I3D.

---

## 1. La cuenta antes del código

La Parte III de la profundización deriva el hiperparámetro central de R(2+1)D. En lugar de presumir que factorizar es mejor porque ahorra parámetros, los autores eligen los canales intermedios $M$ **para igualar exactamente el conteo de parámetros** del bloque 3D. Igualando

$$C_{\text{in}} M k^2 + M C_{\text{out}} t \;=\; C_{\text{in}} C_{\text{out}} k^2 t
\qquad\Longrightarrow\qquad
M \;=\; \frac{t \, k^2 \, C_{\text{in}} \, C_{\text{out}}}{k^2 \, C_{\text{in}} + t \, C_{\text{out}}}$$

Antes de tocar un framework, la fórmula en Python puro:

```python
def channels_for_equal_params(C_in, C_out, k, t):
    """Canales intermedios M que igualan los parámetros de un bloque 3D t*k*k.
    Devuelve el valor REAL (no redondeado): la fórmula rara vez da un entero."""
    return (t * k**2 * C_in * C_out) / (k**2 * C_in + t * C_out)

def params_3d(C_in, C_out, k, t):
    return C_in * C_out * k * k * t          # kernel t x k x k

def params_2plus1d(C_in, C_out, k, t, M):
    return C_in * M * k * k + M * C_out * t  # espacial (1,k,k) + temporal (t,1,1)
```

Primero el caso del texto: $k=3$, $t=3$, $C_{\text{in}} = C_{\text{out}} = C$.

```python
for C in (16, 64, 128, 256, 512):
    M = channels_for_equal_params(C, C, 3, 3)
    print(f"C={C:4d}  M={M:8.2f}  M/C={M/C:.4f}")

# C=  16  M=   36.00  M/C=2.2500
# C=  64  M=  144.00  M/C=2.2500
# C= 128  M=  288.00  M/C=2.2500
# C= 256  M=  576.00  M/C=2.2500
# C= 512  M= 1152.00  M/C=2.2500
```

El $2{,}25\,C$ de la profundización, verificado y estable en todo el rango: la razón $M/C$ no depende de $C$ cuando $C_{\text{in}} = C_{\text{out}}$, porque numerador y denominador son homogéneos en $C$ (grado 2 y grado 1), así que $M = 27C^2 / 12C$.

Ahora el caso general, incluyendo bloques que cambian la cantidad de canales, con la comprobación de que los parámetros empatan:

```python
filas = [(64, 64), (64, 128), (128, 128), (128, 256), (256, 256)]
print(f"{'C_in':>5} {'C_out':>6} {'M exacto':>9} {'M':>5} {'P_3D':>10} {'P_(2+1)D':>10} {'delta %':>8}")
for C_in, C_out in filas:
    Me = channels_for_equal_params(C_in, C_out, 3, 3)
    M  = int(round(Me))                       # los canales tienen que ser enteros
    p3 = params_3d(C_in, C_out, 3, 3)
    pf = params_2plus1d(C_in, C_out, 3, 3, M)
    print(f"{C_in:5d} {C_out:6d} {Me:9.2f} {M:5d} {p3:10d} {pf:10d} {100*(pf-p3)/p3:+8.3f}")
```

| $C_{\text{in}}$ | $C_{\text{out}}$ | $M$ exacto | $M$ entero | $P_{\text{3D}}$ | $P_{(2+1)\text{D}}$ | delta |
|---|---|---|---|---|---|---|
| 64 | 64 | 144,00 | 144 | 110.592 | 110.592 | +0,000 % |
| 64 | 128 | 230,40 | 230 | 221.184 | 220.800 | −0,174 % |
| 128 | 128 | 288,00 | 288 | 442.368 | 442.368 | +0,000 % |
| 128 | 256 | 460,80 | 461 | 884.736 | 885.120 | +0,043 % |
| 256 | 256 | 576,00 | 576 | 1.769.472 | 1.769.472 | +0,000 % |

{{< concept-alert type="advertencia" >}}
**El empate es exacto solo cuando $M$ cae en un entero.** Con $C_{\text{in}} = C_{\text{out}}$ y $C$ múltiplo de 4, $M = 2{,}25\,C$ es entero y la igualdad se cumple al parámetro. En los bloques de transición, donde $C_{\text{out}} = 2\,C_{\text{in}}$, la fórmula da $230{,}4$ o $460{,}8$ y hay que redondear: el empate queda con un error de **0,04 % a 0,2 %**. Es irrelevante para la comparación —nadie atribuye una diferencia de precisión a 384 parámetros de 221.184— pero conviene decirlo en lugar de afirmar una igualdad que el código no produce.
{{< /concept-alert >}}

---

## 2. PyTorch: el bloque (2+1)D

En PyTorch los tensores de video son `[N, C, T, H, W]`. El bloque son dos `nn.Conv3d` con kernels degenerados, y el detalle que hay que cuidar es el **padding asimétrico**: la convolución espacial no debe tocar el eje temporal y la temporal no debe tocar los espaciales.

```python
import torch, torch.nn as nn

class Conv2Plus1D(nn.Module):
    """Bloque factorizado de R(2+1)D: (1,k,k) -> BN -> ReLU -> (t,1,1) -> BN -> ReLU."""
    def __init__(self, c_in, c_out, k=3, t=3, mid=None):
        super().__init__()
        M = mid if mid is not None else int(round(channels_for_equal_params(c_in, c_out, k, t)))
        self.mid_channels = M
        # espacial: kernel (1,k,k), padding (0, k//2, k//2)  -> NO toca el tiempo
        self.spatial  = nn.Conv3d(c_in, M, (1, k, k), padding=(0, k // 2, k // 2), bias=False)
        self.bn       = nn.BatchNorm3d(M)
        self.relu     = nn.ReLU(inplace=True)
        # temporal: kernel (t,1,1), padding (t//2, 0, 0)     -> NO toca el espacio
        self.temporal = nn.Conv3d(M, c_out, (t, 1, 1), padding=(t // 2, 0, 0), bias=False)
        self.bn_out   = nn.BatchNorm3d(c_out)
        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x):                     # [N, C_in, T, H, W]
        x = self.relu(self.bn(self.spatial(x)))
        return self.relu_out(self.bn_out(self.temporal(x)))

class Conv3DBlock(nn.Module):
    """Bloque 3D de referencia: un solo kernel cúbico (t,k,k) -> BN -> ReLU."""
    def __init__(self, c_in, c_out, k=3, t=3):
        super().__init__()
        self.conv = nn.Conv3d(c_in, c_out, (t, k, k),
                              padding=(t // 2, k // 2, k // 2), bias=False)
        self.bn   = nn.BatchNorm3d(c_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
```

Los dos `bias=False` no son cosmética: la `BatchNorm` que sigue tiene su propio $\beta$, así que un sesgo sería redundante y además rompería el empate de la fórmula, que solo cuenta pesos de kernel. El conteo:

```python
def n_params(m):
    return sum(p.numel() for p in m.parameters())

def n_conv_params(m):
    return sum(p.numel() for mod in m.modules()
               if isinstance(mod, nn.Conv3d) for p in mod.parameters())

a, b = Conv3DBlock(64, 64), Conv2Plus1D(64, 64)
print("M =", b.mid_channels)                              # M = 144
print("3D    total", n_params(a), "| conv", n_conv_params(a))   # total 110720 | conv 110592
print("2+1D  total", n_params(b), "| conv", n_conv_params(b))   # total 111008 | conv 110592

x = torch.randn(2, 64, 16, 56, 56)
with torch.no_grad():
    print(tuple(a(x).shape), tuple(b(x).shape))
# (2, 64, 16, 56, 56) (2, 64, 16, 56, 56)   <- misma forma, T/H/W preservados
```

Los pesos convolucionales empatan al parámetro: **110.592 en ambos**. Los totales difieren en 288, y la razón es localizable: el factorizado tiene dos `BatchNorm3d` en lugar de una, y la primera vive sobre $M = 144$ canales. Es un 0,26 % del bloque.

{{< concept-alert type="recordar" >}}
**Las tres formas del kernel.** El espacial es `(1, k, k)`: un $3\times3$ ordinario en un solo plano temporal. El temporal es `(t, 1, 1)`: un filtro FIR de $t$ taps aplicado píxel a píxel, sin extensión espacial —una convolución 1D en el tiempo, con mezcla de canales. Y el 3D es `(t, k, k)`, el cubo que los dos anteriores reemplazan. Si el padding no acompaña la forma del kernel, las salidas dejan de coincidir y la comparación pierde sentido.
{{< /concept-alert >}}

---

## 3. Medir el costo real

Los parámetros empatan por construcción, y los FLOPs también: como ninguno de los dos bloques cambia la resolución, cada peso se evalúa en las mismas $T \cdot H \cdot W$ posiciones, así que igualar parámetros iguala multiplicaciones.

```python
T, H, W = 16, 56, 56
pos = T * H * W                                        # 50 176 posiciones de salida
print("3D     GMAC/muestra:", params_3d(64, 64, 3, 3) * pos / 1e9)         # 5.549
print("(2+1)D GMAC/muestra:", params_2plus1d(64, 64, 3, 3, 144) * pos / 1e9)  # 5.549
#   desglose: espacial 4.162 GMAC + temporal 1.387 GMAC
```

Mismos parámetros, mismos FLOPs. ¿Por qué habría entonces diferencia de tiempo? Midámoslo.

```python
import time

def benchmark(m, x, reps=20, warmup=5):
    """Mediana de `reps` forwards, descartando `warmup` iteraciones de calentamiento."""
    m.eval()
    dev = x.device.type
    with torch.no_grad():
        for _ in range(warmup):            # cuDNN elige algoritmo, se alocan buffers,
            m(x)                           # el reloj de la GPU sube de frecuencia
        if dev == "cuda": torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            m(x)
            if dev == "cuda": torch.cuda.synchronize()   # IMPRESCINDIBLE
            ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]        # mediana, no media: robusta a outliers del SO
```

{{< concept-alert type="advertencia" >}}
**Sin `torch.cuda.synchronize()` la medición es ficción.** Las llamadas a CUDA son **asíncronas**: `m(x)` encola kernels en el stream y retorna de inmediato, antes de que la GPU haya calculado nada. Si se toma `perf_counter()` justo después, se mide el tiempo de *encolar* —microsegundos, idéntico para los dos bloques. `synchronize()` bloquea el hilo de Python hasta que el stream se vació, y solo entonces el delta corresponde al cómputo. Hay que llamarlo **después del warm-up** (para no contar la selección de algoritmo de cuDNN) **y después de cada repetición**. En CPU no hace falta: la ejecución ya es síncrona.
{{< /concept-alert >}}

Resultado sobre `(2, 64, 16, 56, 56)` en un Apple M-series (PyTorch 2.11), en CPU con distinta cantidad de hilos y en MPS:

```python
x = torch.randn(2, 64, 16, 56, 56)
a, b = Conv3DBlock(64, 64), Conv2Plus1D(64, 64)
print(benchmark(a, x) * 1e3, benchmark(b, x) * 1e3)
```

| Backend | 3D `(3,3,3)` | (2+1)D | ratio (2+1)D / 3D |
|---|---|---|---|
| CPU, 1 hilo | 78,6 ms | 48,8 ms | **0,62×** |
| CPU, 4 hilos | 32,2 ms | 23,1 ms | 0,72× |
| CPU, 8 hilos | 24,1 ms | 19,3 ms | 0,80× |
| MPS (GPU integrada) | 17,5 ms | 18,0 ms | **1,03×** |

En CPU el bloque factorizado gana, y bastante: dos kernels chicos y con buena localidad de memoria se vectorizan mejor que un cubo de 27 taps. Pero **la ventaja se erosiona monótonamente a medida que se agregan hilos** (0,62 → 0,72 → 0,80) y **desaparece por completo en la GPU**, donde el factorizado resulta un 3 % más lento.

{{< concept-alert type="advertencia" >}}
**La ganancia de (2+1)D no es de latencia — es de precisión y de optimización.** Con parámetros y FLOPs idénticos, el bloque factorizado paga tres costos que el 3D no paga: (1) son **dos lanzamientos de kernel en secuencia** en lugar de uno; (2) hay un **tensor intermedio de $M = 2{,}25\,C$ canales** que se materializa en memoria —57,8 MB contra los 25,7 MB de la salida, o sea **2,25× más tráfico de memoria**; (3) hay una `BatchNorm` extra sobre ese tensor ancho. En un backend con suficiente paralelismo, esos tres costos se comen exactamente lo que ahorra la forma más simple del kernel. Lo que reportan los papers es **mejor precisión a igual presupuesto** y **menor error de entrenamiento** —el argumento de optimización de R(2+1)D—, no una aceleración automática. Y el número de esta tabla es de un M-series: en una A100 con cuDNN, o con otros canales y resoluciones, el ratio cambia. **Lo transferible no es el número, es el método de medirlo.**
{{< /concept-alert >}}

---

## 4. La ganancia que sí es estructural: el doble de no linealidades

Lo que la factorización agrega gratis, y esto no depende del hardware, es **una no linealidad más por bloque**. Donde el 3D tiene un $\mathrm{ReLU}$, el factorizado tiene dos:

```python
def count_relu(m):
    return sum(1 for mod in m.modules() if isinstance(mod, nn.ReLU))

print(f"{'N bloques':>10} {'ReLU 3D':>8} {'ReLU (2+1)D':>12} {'params 3D':>11} {'params (2+1)D':>14}")
for N in (1, 4, 8, 16):
    s3 = nn.Sequential(*[Conv3DBlock(64, 64)  for _ in range(N)])
    sf = nn.Sequential(*[Conv2Plus1D(64, 64) for _ in range(N)])
    print(f"{N:10d} {count_relu(s3):8d} {count_relu(sf):12d} "
          f"{n_conv_params(s3):11d} {n_conv_params(sf):14d}")
```

| N bloques | ReLU 3D | ReLU (2+1)D | params conv 3D | params conv (2+1)D |
|---|---|---|---|---|
| 1 | 1 | 2 | 110.592 | 110.592 |
| 4 | 4 | 8 | 442.368 | 442.368 |
| 8 | 8 | 16 | 884.736 | 884.736 |
| 16 | 16 | 32 | 1.769.472 | 1.769.472 |

La columna de parámetros es idéntica en todas las filas; la de $\mathrm{ReLU}$ es el doble. Por qué importa: una red con $\mathrm{ReLU}$ es una función **lineal por trozos**, y la cantidad de regiones lineales que puede representar crece con la cantidad de $\mathrm{ReLU}$ en el camino, no con la de pesos. Duplicarlas a parámetros constantes es capacidad expresiva sin costo en memoria de modelo. Es el mismo argumento por el que [VGG](/papers/vggnet-simonyan-2014) reemplazó un $5\times5$ por dos $3\times3$.

---

## 5. El inflado de un bloque separable

La Parte III de la profundización cierra con un punto que hace de la factorización algo más que un ahorro: **el bloque separable se infla de forma más limpia que el cúbico**.

- El kernel **espacial** $1 \times k \times k$ tiene extensión temporal $N = 1$. La condición de punto fijo $\sum_{\tau} \widetilde{W}[\cdot,\tau,\cdot] = W$ se satisface con $\widetilde{W} = W$ **sin dividir por nada**: los pesos de ImageNet se copian tal cual.
- El kernel **temporal** $t \times 1 \times 1$ no tiene análogo 2D. Se inicializa con la **delta central**: todo el peso en $\tau = \lceil t/2 \rceil$ y cero en el resto.

Con esa combinación el bloque arranca siendo exactamente la red 2D aplicada frame a frame. Verifiquémoslo contra un pipeline 2D explícito.

```python
k, t, C_in, C_out, M = 3, 3, 16, 24, 32

# --- el modelo 2D "pre-entrenado" del que heredamos ---
conv2d  = nn.Conv2d(C_in, M, k, padding=k // 2, bias=False)   # el (k,k) de ImageNet
bn2d    = nn.BatchNorm2d(M)
point2d = nn.Conv2d(M, C_out, 1, bias=False)                  # el 1x1 que sigue
with torch.no_grad():                                          # estadísticas no triviales
    bn2d.weight.normal_(1.0, 0.1);  bn2d.bias.normal_(0, 0.1)
    bn2d.running_mean.normal_(0, 0.5); bn2d.running_var.uniform_(0.5, 2.0)

blk = Conv2Plus1D(C_in, C_out, k=k, t=t, mid=M)
with torch.no_grad():
    # 1) espacial: unsqueeze en el eje temporal, SIN división
    blk.spatial.weight.copy_(conv2d.weight.unsqueeze(2))       # (M,C_in,k,k) -> (M,C_in,1,k,k)
    blk.bn.load_state_dict(bn2d.state_dict())                  # BN se copia sin tocar
    # 2) temporal: delta central. Todo el peso en tau = t//2
    blk.temporal.weight.zero_()
    blk.temporal.weight[:, :, t // 2, 0, 0] = point2d.weight[:, :, 0, 0]
    # bn_out en identidad, para aislar el test
    blk.bn_out.weight.fill_(1.0); blk.bn_out.bias.zero_()
    blk.bn_out.running_mean.zero_(); blk.bn_out.running_var.fill_(1.0)
blk.eval(); bn2d.eval()

clip = torch.randn(2, C_in, 5, 12, 12)          # [N, C, T, H, W]
with torch.no_grad():
    y_3d = blk(clip)
    # el mismo cálculo, frame a frame, con capas 2D
    y_2d = torch.stack([
        torch.relu(point2d(torch.relu(bn2d(conv2d(clip[:, :, i])))))
        for i in range(clip.shape[2])
    ], dim=2)

print("max |y_3d - y_2d| =", (y_3d - y_2d).abs().max().item())
# max |y_3d - y_2d| = 6.1e-06     <- ruido de float32, no error de lógica
```

La diferencia es del orden de $10^{-6}$: ruido de acumulación en `float32`. El bloque inflado **es** la red 2D aplicada por frame.

{{< concept-alert type="clave" >}}
**La delta central es exacta también en los bordes temporales.** Con `padding=(t//2,0,0)`, la salida en el frame $i$ es $\sum_\tau W[\tau] \cdot x[i + \tau - \lfloor t/2 \rfloor]$, y como solo el tap central es no nulo, **el padding con ceros nunca entra en la cuenta**. Por eso el test da $10^{-6}$ en todo el clip, incluidos el primer y el último frame. El reparto uniforme de I3D no tiene esa propiedad: si se repite el test cambiando la delta por $W/t$, la diferencia sube a **0,55**, porque sobre video real el filtro uniforme promedia temporalmente y ya no equivale a la red 2D. Es la tabla de la [Parte I](/clases/clase-38/profundizacion) hecha número.
{{< /concept-alert >}}

---

## 6. TensorFlow / Keras

El mismo bloque en Keras, con la diferencia de convención que hay que tener presente todo el tiempo.

{{< concept-alert type="recordar" >}}
**`data_format`: el eje de canales está en otro lugar.** PyTorch usa `channels_first`: el tensor es `[N, C, T, H, W]`, el canal en la posición 1. Keras usa `channels_last` por defecto: `[N, T, H, W, C]`, el canal al final. Los `kernel_size=(1,k,k)` y `(t,1,1)` se escriben igual en ambos —siempre se refieren a los ejes espacio-temporales, nunca al de canales— pero **el tensor de entrada hay que transponerlo** al portar código. Y el kernel de `Conv3D` en Keras tiene forma `(kd, kh, kw, C_in, C_out)` contra `(C_out, C_in, kd, kh, kw)` de PyTorch: cargar pesos de un framework en el otro exige un `transpose`, no solo un `reshape`.
{{< /concept-alert >}}

```python
import tensorflow as tf
L = tf.keras.layers

def conv_2plus1d_tf(T, H, W, c_in, c_out, k=3, t=3, mid=None):
    M = mid if mid is not None else int(round(channels_for_equal_params(c_in, c_out, k, t)))
    return tf.keras.Sequential([
        tf.keras.Input(shape=(T, H, W, c_in)),          # channels_last: C al final
        L.Conv3D(M, (1, k, k), padding="same", use_bias=False),   # espacial
        L.BatchNormalization(), L.ReLU(),
        L.Conv3D(c_out, (t, 1, 1), padding="same", use_bias=False),  # temporal
        L.BatchNormalization(), L.ReLU(),
    ], name=f"conv2plus1d_M{M}")

def conv_3d_tf(T, H, W, c_in, c_out, k=3, t=3):
    return tf.keras.Sequential([
        tf.keras.Input(shape=(T, H, W, c_in)),
        L.Conv3D(c_out, (t, k, k), padding="same", use_bias=False),
        L.BatchNormalization(), L.ReLU(),
    ], name="conv3d")

def conv_params_tf(m):
    return sum(int(tf.size(w)) for l in m.layers
               if isinstance(l, L.Conv3D) for w in l.weights)

m3, mf = conv_3d_tf(16, 56, 56, 64, 64), conv_2plus1d_tf(16, 56, 56, 64, 64)
print("3D    count_params", m3.count_params(), "| conv", conv_params_tf(m3))
# 3D    count_params 110848 | conv 110592
print("2+1D  count_params", mf.count_params(), "| conv", conv_params_tf(mf))
# 2+1D  count_params 111424 | conv 110592

x = tf.random.normal((2, 16, 56, 56, 64))
print(m3(x, training=False).shape, mf(x, training=False).shape)
# (2, 16, 56, 56, 64) (2, 16, 56, 56, 64)
```

Los pesos convolucionales dan **110.592 en ambos**, el mismo número que PyTorch. El `padding="same"` de Keras resuelve solo el padding asimétrico: con `(1,k,k)` no agrega nada en el tiempo y con `(t,1,1)` nada en el espacio, así que no hay que escribir la tupla a mano. Con kernels de tamaño **par** la equivalencia con PyTorch se rompe; con $k = t = 3$ coinciden.

{{< concept-alert type="advertencia" >}}
**`count_params()` de Keras y `sum(p.numel())` de PyTorch no cuentan lo mismo.** Los totales de arriba (110.848 y 111.424) son más altos que los de PyTorch (110.720 y 111.008) porque `count_params()` incluye los **pesos no entrenables**: `BatchNormalization` aporta cuatro tensores por canal ($\gamma$, $\beta$, `moving_mean`, `moving_variance`), mientras que en PyTorch `running_mean` y `running_var` son *buffers* y no aparecen en `.parameters()`. Cuatro por canal contra dos. Al comparar conteos entre frameworks conviene mirar solo los pesos convolucionales, o usar `sum(tf.size(w) for w in m.trainable_weights)`.
{{< /concept-alert >}}

---

## 7. JAX / Flax

Flax también es `channels_last`: el tensor es `(N, T, H, W, C)`, igual que en Keras. La diferencia estructural es que los parámetros **no viven dentro del módulo**: son un `dict` anidado (un *pytree*) que `init` devuelve y que hay que pasar explícitamente en cada `apply`.

```python
import jax, jax.numpy as jnp
from flax import linen as nn_f

class Conv2Plus1DFlax(nn_f.Module):
    c_out: int
    mid: int
    k: int = 3
    t: int = 3

    @nn_f.compact                              # define las capas en línea, dentro del __call__
    def __call__(self, x, train: bool = False):
        x = nn_f.Conv(self.mid, (1, self.k, self.k),
                      padding="SAME", use_bias=False)(x)        # espacial
        x = nn_f.BatchNorm(use_running_average=not train)(x)
        x = nn_f.relu(x)
        x = nn_f.Conv(self.c_out, (self.t, 1, 1),
                      padding="SAME", use_bias=False)(x)        # temporal
        x = nn_f.BatchNorm(use_running_average=not train)(x)
        return nn_f.relu(x)

class Conv3DFlax(nn_f.Module):
    c_out: int
    k: int = 3
    t: int = 3

    @nn_f.compact
    def __call__(self, x, train: bool = False):
        x = nn_f.Conv(self.c_out, (self.t, self.k, self.k),
                      padding="SAME", use_bias=False)(x)
        x = nn_f.BatchNorm(use_running_average=not train)(x)
        return nn_f.relu(x)

def tree_size(tree):
    """Suma el tamaño de todas las hojas del pytree de parámetros."""
    return sum(leaf.size for leaf in jax.tree_util.tree_leaves(tree))

C = 64
M = int(round(channels_for_equal_params(C, C, 3, 3)))           # 144
x  = jnp.zeros((2, 16, 56, 56, C))
key = jax.random.PRNGKey(0)

v3 = Conv3DFlax(c_out=C).init(key, x)
vf = Conv2Plus1DFlax(c_out=C, mid=M).init(key, x)

print("3D    params", tree_size(v3["params"]), "| batch_stats", tree_size(v3["batch_stats"]))
# 3D    params 110720 | batch_stats 128
print("2+1D  params", tree_size(vf["params"]), "| batch_stats", tree_size(vf["batch_stats"]))
# 2+1D  params 111008 | batch_stats 416

# solo los kernels convolucionales: los submódulos cuyo nombre empieza con "Conv"
def conv_params_flax(v):
    return sum(v["params"][name]["kernel"].size
               for name in v["params"] if name.startswith("Conv"))
print("conv 3D", conv_params_flax(v3), "| conv (2+1)D", conv_params_flax(vf))
# conv 3D 110592 | conv (2+1)D 110592

# la forma de cada kernel, para ver la convención channels_last de Flax
print(jax.tree_util.tree_map(jnp.shape, vf["params"]))
# {'BatchNorm_0': {'bias': (144,), 'scale': (144,)},
#  'BatchNorm_1': {'bias': (64,),  'scale': (64,)},
#  'Conv_0': {'kernel': (1, 3, 3, 64, 144)},      <- (kd, kh, kw, C_in, C_out)
#  'Conv_1': {'kernel': (3, 1, 1, 144, 64)}}

y = Conv2Plus1DFlax(c_out=C, mid=M).apply(vf, x, train=False)
print(y.shape)                                                  # (2, 16, 56, 56, 64)
```

Los **110.592** pesos convolucionales aparecen por tercera vez, y los totales con `BatchNorm` (110.720 y 111.008) coinciden exactamente con PyTorch: Flax también separa los parámetros aprendidos (`params`) de las estadísticas acumuladas (`batch_stats`), igual que PyTorch separa parámetros de buffers. Mismo bloque, misma cuenta, tres convenciones de dónde guardar cada cosa.

---

## 8. Qué nos llevamos

- La fórmula $M = \frac{t k^2 C_{\text{in}} C_{\text{out}}}{k^2 C_{\text{in}} + t C_{\text{out}}}$ funciona: con $k=t=3$ y $C_{\text{in}}=C_{\text{out}}=C$ da $2{,}25\,C$ exacto, y los pesos convolucionales empatan **al parámetro** (110.592 en los tres frameworks). Donde $M$ no cae en un entero, el redondeo deja un error de 0,04 % a 0,2 %.
- Igualar parámetros iguala también los **FLOPs** (5,549 GMAC por muestra en ambos), porque ninguno cambia la resolución.
- **El bloque factorizado no es automáticamente más rápido.** En CPU midió 0,62× a 0,80× según los hilos, pero en MPS resultó 1,03× —marginalmente **más lento**—, por los dos lanzamientos de kernel en secuencia y el tensor intermedio 2,25× más ancho. La ganancia de R(2+1)D y S3D es de **precisión a igual presupuesto** y de **optimización**, no de latencia.
- Lo estructural, independiente del hardware: **el doble de no linealidades** a parámetros constantes ($2N$ contra $N$ $\mathrm{ReLU}$ en una pila de $N$ bloques).
- El **inflado del bloque separable es más limpio** que el del cúbico: el kernel espacial copia ImageNet sin dividir por nada y el temporal arranca con la delta central. El bloque recién inicializado reproduce la red 2D frame a frame con error de $10^{-6}$, y a diferencia del reparto uniforme de I3D, **también en los bordes temporales**.
- Al medir en GPU, `synchronize()` no es opcional: sin él se mide el encolado de kernels, no el cómputo.

---

**Ver tambien:** [Clase 38 - Teoria](/clases/clase-38/teoria) · [Clase 38 - Profundizacion](/clases/clase-38/profundizacion) · [R(2+1)D](/papers/r2plus1d-tran-2018) · [S3D](/papers/s3d-xie-2018) · [Inflado de Convoluciones](/fundamentos/inflado-de-convoluciones)
