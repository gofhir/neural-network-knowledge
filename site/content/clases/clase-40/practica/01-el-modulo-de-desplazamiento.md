---
title: "01 - El módulo de desplazamiento"
weight: 10
math: true
---

> Construir el módulo de [TSM](/papers/tsm-lin-2019) desde cero y verificar sus tres afirmaciones: que equivale a una convolución temporal, que no cuesta parámetros, y que su campo receptivo crece con la profundidad. Las tres resultan ciertas — con matices que solo aparecen al medirlas.

---

## 1. Las tres asignaciones

El módulo completo, en NumPy, sin ninguna dependencia:

```python
import numpy as np

def temporal_shift(x, fold_div=8):
    """x: (T, C, H, W) -> (T, C, H, W). Desplazamiento bidireccional."""
    T, C, H, W = x.shape
    f = C // fold_div
    out = np.zeros_like(x)
    out[:-1, :f]     = x[1:, :f]      # futuro (t+1) -> presente
    out[1:, f:2*f]   = x[:-1, f:2*f]  # pasado  (t-1) -> presente
    out[:,  2*f:]    = x[:,  2*f:]    # el resto, sin tocar
    return out
```

Tres líneas, cero parámetros. La variante **unidireccional** (modo online) es una línea menos:

```python
def temporal_shift_uni(x, fold_div=8):
    """Solo el pasado entra al presente: causal, apto para streaming."""
    T, C, H, W = x.shape
    f = C // fold_div
    out = np.zeros_like(x)
    out[1:, :f] = x[:-1, :f]
    out[:,  f:] = x[:,  f:]
    return out
```

Conviene inspeccionar qué queda en cada frame antes de seguir:

```python
T, C = 5, 8
x = np.arange(T * C, dtype=float).reshape(T, C, 1, 1)   # canal c del frame t = t*C + c
print(temporal_shift(x, fold_div=8)[:, :, 0, 0])
```

Con `C = 8` y `fold_div = 8` sale `f = 1`, así que el canal 0 trae el futuro, el canal 1 el pasado y los seis restantes se quedan quietos:

```
[[ 8.  0.  2.  3.  4.  5.  6.  7.]   <- t=0: canal 1 (pasado) = 0, no hay t=-1
 [16.  1. 10. 11. 12. 13. 14. 15.]
 [24.  9. 18. 19. 20. 21. 22. 23.]
 [32. 17. 26. 27. 28. 29. 30. 31.]
 [ 0. 25. 34. 35. 36. 37. 38. 39.]]  <- t=4: canal 0 (futuro) = 0, no hay t=5
```

Los ceros de las esquinas son el **relleno de bordes**: el primer frame no tiene pasado y el último no tiene futuro. En una pila de 16 módulos ese efecto se acumula sobre el 25 % de los segmentos cuando $T = 8$.

---

## 2. La verificación central: ¿es realmente una convolución temporal?

La afirmación de la clase es que el desplazamiento, seguido de la convolución que ya estaba, produce el efecto de una convolución temporal. Se puede comprobar exactamente.

Una convolución temporal $3\times1\times1$ genuina es

$$y_t = W^{(-1)} x_{t-1} + W^{(0)} x_t + W^{(+1)} x_{t+1}$$

con tres matrices independientes. La hipótesis es que `shift` + `conv1x1(W)` equivale a esa expresión con $W^{(\tau)} = W P_\tau$, donde $P_\tau$ proyecta sobre el bloque de canales correspondiente.

```python
rng = np.random.default_rng(0)
T, C, H, W_, Cout, d = 8, 64, 5, 5, 32, 8
x  = rng.standard_normal((T, C, H, W_))
Wm = rng.standard_normal((Cout, C))
F  = C // d

conv1x1 = lambda z, M: np.einsum('oc,tchw->tohw', M, z)

# camino A: desplazar y aplicar la 1x1
y_tsm = conv1x1(temporal_shift(x, d), Wm)

# camino B: convolución temporal explícita con soportes disjuntos
P_fut = np.r_[np.ones(F),   np.zeros(C - F)]
P_pas = np.r_[np.zeros(F),  np.ones(F), np.zeros(C - 2*F)]
P_pre = np.r_[np.zeros(2*F), np.ones(C - 2*F)]

xp1 = np.concatenate([x[1:], np.zeros((1, C, H, W_))])    # x_{t+1}
xm1 = np.concatenate([np.zeros((1, C, H, W_)), x[:-1]])   # x_{t-1}

y_conv = (conv1x1(xp1, Wm * P_fut)
        + conv1x1(xm1, Wm * P_pas)
        + conv1x1(x,   Wm * P_pre))

print(np.allclose(y_tsm, y_conv), np.abs(y_tsm - y_conv).max())
```

```
True 2.4868995751603507e-14
```

Igualdad exacta hasta el error de punto flotante. **El desplazamiento sí implementa una convolución temporal.**

### La letra chica

Ahora la parte que la clase no menciona. Cuenta los parámetros de ambos caminos:

```python
print(f"conv 3x1x1 general : {3*C*Cout:>6}")
print(f"TSM + la 1x1 que ya existía : {C*Cout:>6}")
```

```
conv 3x1x1 general :   6144
TSM + la 1x1 que ya existía :   2048
```

**Un tercio.** La razón está en las proyecciones: `P_fut`, `P_pas` y `P_pre` seleccionan bloques de canales **disjuntos**, de modo que cada canal de entrada contribuye a un único instante temporal. Una convolución $3\times1\times1$ real permite que el canal $c$ aporte desde $t-1$, $t$ y $t+1$ con tres pesos independientes; TSM le asigna un instante y listo.

{{< concept-alert type="clave" >}}
El desplazamiento no vuelve gratis a la convolución temporal: la **reemplaza por una versión restringida** cuyo costo coincide exactamente con lo que la red ya gastaba. Que esa restricción alcance es un resultado empírico, no una consecuencia matemática. El paper lo pone a prueba con un control —reemplazar cada módulo por una $3\times1\times1$ genuina, tres veces más parámetros— y el resultado es **más lento y menos preciso**.
{{< /concept-alert >}}

---

## 3. Los dos casos degenerados

Ambos extremos del hiperparámetro tienen consecuencias que el laboratorio explota.

```python
# fold_div mayor que C  ->  f = 0  ->  el módulo es la identidad
print(np.allclose(temporal_shift(x, 10**9), x))          # True

# fold_div = 2  ->  se desplaza el 100 %: ningún canal ve el presente
print(x.shape[1] - 2*(C // 2))                           # 0
```

El primero es la palanca de ablación: con `fold_div` grande, las dos primeras asignaciones quedan vacías y la tercera copia el tensor completo. **El módulo se anula sin tocar un solo peso**, lo que permite medir su contribución sobre un checkpoint entrenado. Es exactamente lo que hace el [Laboratorio 40](/laboratorios/lab-40/03-la-ablacion-del-shift), donde anular los 16 módulos hace caer la confianza de un video de salto alto del 99,12 % al 16,37 %.

El segundo es el *naive shift* que el paper descarta: con `fold_div = 2` se desplaza todo el tensor y no queda ningún canal representando el instante actual. La [curva de proporción](/laboratorios/lab-40/04-la-curva-de-proporcion) medida sobre el checkpoint real muestra el colapso: 0,52 %.

---

## 4. El alcance temporal efectivo

La clase afirma que cada módulo amplía el campo receptivo temporal en 2. Es cierto como **cota**: con 16 módulos la información puede recorrer hasta 16 frames en cada dirección. La pregunta interesante es cuánta información recorre efectivamente esa distancia.

Modelando el transporte como una caminata aleatoria —cada canal se corre $+1$ con probabilidad $1/8$, $-1$ con $1/8$, y $0$ con $3/4$:

```python
rng = np.random.default_rng(0)
for L in (4, 8, 16, 32):
    pasos = rng.choice([-1, 0, 1], size=(200_000, L), p=[1/8, 3/4, 1/8]).sum(1)
    print(f"L={L:3d}  sigma medida = {pasos.std():.3f}   sqrt(L)/2 = {np.sqrt(L)/2:.3f}")
```

```
L=  4  sigma medida = 1.001   sqrt(L)/2 = 1.000
L=  8  sigma medida = 1.415   sqrt(L)/2 = 1.414
L= 16  sigma medida = 1.997   sqrt(L)/2 = 2.000
L= 32  sigma medida = 2.824   sqrt(L)/2 = 2.828
```

La predicción analítica $\sigma_L = \sqrt{L}/2$ se cumple. Para la ResNet-50 del laboratorio, con $L = 16$: **el campo receptivo teórico es de ±16 frames, pero la masa de información se concentra en ±2**.

{{< concept-alert type="nota" >}}
El alcance efectivo crece como $\sqrt{L}$, no como $L$. Duplicar la profundidad lo multiplica por 1,41. Es un modelo idealizado —entre módulo y módulo las convoluciones mezclan canales, así que la información no viaja por un canal fijo— y sirve para el orden de magnitud y la dependencia funcional, no para predecir valores exactos. La conclusión cualitativa se sostiene: TSM modela **movimiento local** entre segmentos vecinos; la cobertura del video completo la aporta el otro mecanismo, el muestreo por segmentos.
{{< /concept-alert >}}

---

## 5. El módulo en los tres frameworks

Lo esencial es idéntico; cambia el orden de ejes y la mutabilidad de los tensores.

### PyTorch

La implementación de referencia opera sobre un tensor `(N·T, C, H, W)` —el tiempo viene aplanado en el lote— y lo reinterpreta internamente:

```python
import torch
import torch.nn as nn

class TemporalShift(nn.Module):
    """Envuelve una capa e inserta el desplazamiento antes de ella."""
    def __init__(self, net, n_segment=8, n_div=8):
        super().__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div

    @staticmethod
    def shift(x, n_segment, fold_div=8):
        nt, c, h, w = x.size()
        x = x.view(nt // n_segment, n_segment, c, h, w)
        fold = c // fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold]      = x[:, 1:,  :fold]
        out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]
        out[:, :,  2*fold:]     = x[:, :,   2*fold:]
        return out.view(nt, c, h, w)

    def forward(self, x):
        return self.net(self.shift(x, self.n_segment, self.fold_div))
```

Insertarlo en una ResNet es reemplazar la `conv1` de cada bloque bottleneck — el *residual shift* que discute la [profundización](/clases/clase-40/profundizacion):

```python
import torchvision

net = torchvision.models.resnet50(weights=None)
for stage in (net.layer1, net.layer2, net.layer3, net.layer4):
    for b in stage:
        b.conv1 = TemporalShift(b.conv1, n_segment=8, n_div=8)

n = sum(isinstance(m, TemporalShift) for m in net.modules())
p = sum(q.numel() for q in net.parameters())
print(f"módulos insertados: {n}   parámetros: {p:,}")
```

```
módulos insertados: 16   parámetros: 25,557,032
```

Los 16 módulos son 3+4+6+3, y el conteo de parámetros es **idéntico** al de la ResNet-50 sin modificar: el módulo no aporta ninguno.

{{< concept-alert type="cuidado" >}}
`b.conv1 = TemporalShift(b.conv1, ...)` cambia la ruta de los parámetros en el `state_dict`: `layer1.0.conv1.weight` pasa a ser `layer1.0.conv1.**net**.weight`. Un checkpoint entrenado con desplazamiento solo carga sobre un modelo que tenga los mismos envoltorios, y viceversa. En el [Laboratorio 40](/laboratorios/lab-40/05-los-defectos-del-notebook) esa correspondencia es lo que hace que el nombre del archivo `.pth` funcione como archivo de configuración.
{{< /concept-alert >}}

### TensorFlow / Keras

Sin asignación por rebanadas sobre tensores, así que se construye por concatenación:

```python
import tensorflow as tf

class TemporalShift(tf.keras.layers.Layer):
    def __init__(self, n_segment=8, fold_div=8, **kw):
        super().__init__(**kw)
        self.n_segment, self.fold_div = n_segment, fold_div

    def call(self, x):                      # x: (N*T, H, W, C)
        shape = tf.shape(x)
        nt, h, w = shape[0], shape[1], shape[2]
        c = x.shape[-1]
        f = c // self.fold_div
        x = tf.reshape(x, (nt // self.n_segment, self.n_segment, h, w, c))

        fut = tf.concat([x[:, 1:,  ..., :f],      tf.zeros_like(x[:, :1, ..., :f])],      axis=1)
        pas = tf.concat([tf.zeros_like(x[:, :1, ..., f:2*f]), x[:, :-1, ..., f:2*f]],     axis=1)
        pre = x[..., 2*f:]

        out = tf.concat([fut, pas, pre], axis=-1)
        return tf.reshape(out, (nt, h, w, c))
```

Dos diferencias que importan: el formato es `NHWC` en vez de `NCHW`, así que el eje de canales es el último; y los desplazamientos se arman concatenando un bloque de ceros en el extremo correspondiente, que es la forma funcional del mismo relleno.

### JAX

Sin mutación, con `jnp.pad` y recortes:

```python
import jax.numpy as jnp

def temporal_shift(x, n_segment=8, fold_div=8):
    """x: (N*T, C, H, W)"""
    nt, c, h, w = x.shape
    f = c // fold_div
    x = x.reshape(nt // n_segment, n_segment, c, h, w)

    pad_t = lambda z, izq: jnp.pad(z, ((0, 0), (izq, 1 - izq), (0, 0), (0, 0), (0, 0)))

    fut = pad_t(x[:, 1:,  :f],       0)[:, :n_segment]      # desplaza hacia atrás
    pas = pad_t(x[:, :-1, f:2*f],    1)[:, :n_segment]      # desplaza hacia adelante
    pre = x[:, :, 2*f:]

    return jnp.concatenate([fut, pas, pre], axis=2).reshape(nt, c, h, w)
```

La versión JAX es la que mejor expone el mecanismo: **desplazar es rellenar por un lado y recortar por el otro**. Bajo `jit`, XLA suele fusionar el pad y el slice con la convolución siguiente, lo que en principio elimina el costo de materializar el tensor intermedio — el problema de movimiento de datos que la [profundización](/clases/clase-40/profundizacion) discute en su Parte V.

---

## 6. Verificación cruzada

Vale confirmar que las tres implementaciones producen lo mismo. Cuidado con el orden de ejes al comparar TensorFlow con las otras dos:

```python
x_np = rng.standard_normal((8, 64, 5, 5)).astype(np.float32)

r_np = temporal_shift(x_np, 8)
r_pt = TemporalShift.shift(torch.from_numpy(x_np), n_segment=8, fold_div=8).numpy()
r_jx = np.asarray(temporal_shift_jax(jnp.asarray(x_np), 8, 8))
r_tf = TemporalShiftTF(8, 8)(tf.constant(x_np.transpose(0, 2, 3, 1))).numpy().transpose(0, 3, 1, 2)

for nombre, r in [("torch", r_pt), ("jax", r_jx), ("tf", r_tf)]:
    print(f"{nombre:6} igual a numpy: {np.allclose(r_np, r, atol=1e-6)}")
```

Las cuatro implementaciones deben coincidir exactamente: la operación es determinista y sin aritmética de punto flotante — solo mueve bytes.

---

## Qué queda establecido

| Afirmación | Verificación | Resultado |
|---|---|---|
| shift + $1\times1$ = convolución temporal | comparación numérica directa | exacta, $2{,}5 \times 10^{-14}$ |
| ...pero con soportes disjuntos | conteo de parámetros | 2 048 contra 6 144: **3× menos** |
| El módulo no agrega parámetros | `sum(p.numel())` tras insertarlo | 25 557 032, idéntico a la ResNet-50 base |
| `fold_div` grande lo anula | comparación con la entrada | identidad exacta |
| `fold_div = 2` deja el presente vacío | conteo de canales sin desplazar | 0 |
| El alcance efectivo es $\sqrt{L}/2$ | simulación, 200 000 trayectorias | $\sigma_{16} = 1{,}997$ contra 2,000 predicho |

---

## Ver también

- [02 - Muestreo por segmentos contra denso](02-muestreo-por-segmentos) — el otro mecanismo de la clase, y la demostración de la invarianza al orden.
- [Profundización](/clases/clase-40/profundizacion) — la derivación formal de todo lo que acá se verifica numéricamente.
- [Laboratorio 40](/laboratorios/lab-40) — las mismas manipulaciones sobre un checkpoint entrenado en Kinetics-400.
- [Clase 38 - Práctica: inflar una CNN 2D a 3D](/clases/clase-38/practica/01-inflar-una-cnn-2d-a-3d) — la estrategia opuesta, verificada con el mismo método.
