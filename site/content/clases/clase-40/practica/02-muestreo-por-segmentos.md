---
title: "02 - Muestreo por segmentos contra denso"
weight: 20
math: true
---

> El otro mecanismo de la clase. [TSN](/papers/tsn-wang-2016) reemplaza el muestreo a tasa fija por uno basado en segmentos, y la clase lo ilustra con tres diapositivas que repiten "Epoch 1, Epoch 1, Epoch 1" frente a "Epoch 1, Epoch 2". Esta práctica implementa ambas estrategias, mide su cobertura sobre videos de distinta duración y termina demostrando el defecto que TSM viene a corregir: **el consenso por promedio es matemáticamente ciego al orden**.

---

## 1. Las dos estrategias

Ambas producen $K$ índices de frame. La diferencia está en de dónde salen.

```python
import numpy as np

def indices_por_segmentos(n_frames, K=8, new_length=1):
    """TSN en modo test: el frame del centro de cada uno de los K segmentos."""
    tick = (n_frames - new_length + 1) / float(K)
    return np.array([int(tick / 2.0 + tick * x) for x in range(K)]) + 1

def indices_densos(n_frames, K=8, start=0):
    """Estilo I3D: K frames de una ventana contigua de 64."""
    t_stride = 64 // K
    return np.array([(i * t_stride + start) % n_frames for i in range(K)]) + 1
```

Sobre un clip de 250 frames —la duración exacta de los videos de `PlayingGuitar` en UCF-101, 10 segundos a 25 fps:

```python
n = 250
print("segmentos:", indices_por_segmentos(n).tolist())
print("denso    :", indices_densos(n).tolist())
```

```
segmentos: [16, 47, 79, 110, 141, 172, 204, 235]
denso    : [1, 9, 17, 25, 33, 41, 49, 57]
```

La diferencia salta a la vista. El muestreo por segmentos reparte los 8 frames a lo largo de los 250 —un frame cada 31, es decir cada 1,25 s—; el denso se queda en los primeros 57, un **22,8 %** del video.

---

## 2. Cobertura contra duración

Esa diferencia no es un detalle de este clip: es estructural. Vale medirla en función de la duración.

```python
print(f"{'frames':>7} {'paso':>6} {'cobertura segmentos':>20} {'cobertura densa':>17}")
for n in (25, 75, 175, 250, 500, 1775):
    s, d = indices_por_segmentos(n), indices_densos(n)
    print(f"{n:7d} {s[1]-s[0]:6d} {(s[-1]-s[0])/n*100:19.1f}% {(d[-1]-d[0])/n*100:16.1f}%")
```

```
 frames   paso  cobertura segmentos   cobertura densa
     25      3                88.0%             24.0%
     75     10                88.0%             74.7%
    175     22                88.0%             32.0%
    250     31                87.6%             22.4%
    500     62                87.4%             11.2%
   1775    222                87.5%              3.2%
```

**La cobertura por segmentos es constante en ~87,5 %** para cualquier duración: lo que se estira es el paso entre frames, no el rango. La cobertura densa se **desmorona** con la duración, porque la ventana de 64 frames es fija: en un video de 71 segundos —el más largo de UCF-101— ve el 3,2 %.

{{< concept-alert type="clave" >}}
Esa es toda la diferencia. El muestreo por segmentos hace que **el costo sea independiente de la duración del video** y que la cobertura sea completa; el denso mantiene la resolución temporal fina y sacrifica la cobertura. Ninguna es mejor en abstracto: dependen de si la acción se define por una **progresión larga** o por un **gesto breve**.

El precio del muestreo por segmentos es exactamente ese paso de 31 frames: entre dos frames separados por 1,25 s no hay continuidad de movimiento que modelar. El [Laboratorio 40](/laboratorios/lab-40/02-la-varianza-intra-clase) mide la consecuencia sobre videos de guitarra, donde el rasgueo es invisible a esa escala.
{{< /concept-alert >}}

---

## 3. La aumentación temporal implícita

Las diapositivas "Epoch 1 / Epoch 2" de la clase señalan algo que en el código es una sola diferencia: en entrenamiento, el frame se sortea **dentro** de su segmento.

```python
def indices_entrenamiento(n_frames, K=8, new_length=1, rng=None):
    """TSN en modo train: un frame aleatorio dentro de cada segmento."""
    dur = (n_frames - new_length + 1) // K
    return np.multiply(range(K), dur) + rng.integers(0, dur, size=K) + 1
```

Cuánto aporta eso, en números:

```python
rng = np.random.default_rng(0)
n, K = 250, 8
dur = (n - 1 + 1) // K
print(f"duración de cada segmento: {dur} frames")
print(f"combinaciones posibles   : {dur}^{K} = {dur**K:.3e}")

vistos = {tuple(indices_entrenamiento(n, rng=rng)) for _ in range(10_000)}
print(f"10 000 muestreos          -> {len(vistos)} combinaciones distintas")
```

```
duración de cada segmento: 31 frames
combinaciones posibles   : 31^8 = 8.529e+11
10 000 muestreos          -> 10000 combinaciones distintas
```

Diez mil sorteos sin una sola colisión. Con $8{,}5 \times 10^{11}$ combinaciones disponibles, **el modelo nunca ve dos veces el mismo conjunto de frames** de un mismo video. Es aumentación de datos que no cuesta nada: no hay transformación que aplicar, solo un índice distinto.

Contrasta con el muestreo determinista de las tres diapositivas iguales de la clase, donde cada época repite exactamente los mismos frames y el modelo puede memorizarlos.

En test la aleatoriedad desaparece —se toma el centro de cada segmento— porque la evaluación tiene que ser reproducible.

---

## 4. El defecto: el consenso es ciego al orden

Acá está la razón de existir de TSM. La arquitectura TSN completa, en cinco líneas:

```python
def tsn(frames, f_theta):
    """f_theta: CNN 2D compartida. Devuelve logits por video."""
    logits = np.stack([f_theta(x) for x in frames])   # (K, n_clases)
    return logits.mean(axis=0)                        # consenso por promedio
```

El promedio es conmutativo, así que la predicción no puede depender del orden. Verificarlo sobre **todas** las permutaciones de 8 segmentos:

```python
from itertools import permutations

rng = np.random.default_rng(0)
logits = rng.standard_normal((8, 5))       # 8 segmentos, 5 clases
base = logits.mean(0)

peor = max(np.abs(logits[list(p)].mean(0) - base).max()
           for p in permutations(range(8)))
print(f"máxima diferencia sobre las {np.math.factorial(8)} permutaciones: {peor:.2e}")
```

```
máxima diferencia sobre las 40320 permutaciones: 2.22e-16
```

Cero, hasta el épsilon de máquina. **Ninguna de las 40 320 reordenaciones cambia la predicción**, incluida la reversión temporal completa. Un TSN no puede distinguir "abrir una puerta" de "cerrarla", y no por falta de capacidad del backbone: la información de orden se destruye en el promedio, después de que cada frame fue procesado por separado.

### Cómo lo rompe TSM

Con desplazamiento, la entrada efectiva de cada bloque deja de ser su propio frame:

```python
def tsm(frames, g_theta, fold_div=8):
    """g_theta ve el frame k con canales contaminados por k-1 y k+1."""
    x = np.stack(frames)                       # (K, C, H, W)
    x_shift = temporal_shift(x, fold_div)      # del camino 01
    logits = np.stack([g_theta(z) for z in x_shift])
    return logits.mean(axis=0)                 # el promedio sigue siendo conmutativo...
```

El promedio exterior sigue siendo conmutativo, pero **los sumandos ya no son los mismos**: permutar la secuencia cambia qué frame es vecino de cuál, y por lo tanto qué contiene cada canal desplazado. La simetría se rompe **antes** de la agregación, no en ella.

```python
x = rng.standard_normal((8, 16, 4, 4))
perm = rng.permutation(8)

sin_shift = x.mean(0)
con_shift_orig = temporal_shift(x, 8).mean(0)
con_shift_perm = temporal_shift(x[perm], 8).mean(0)

print("sin shift, permutado == original:",
      np.allclose(x[perm].mean(0), sin_shift))          # True
print("con shift, permutado == original:",
      np.allclose(con_shift_perm, con_shift_orig))      # False
```

{{< concept-alert type="nota" >}}
Que la simetría esté rota en la arquitectura no garantiza que el modelo **use** el orden. El [Laboratorio 38](/laboratorios/lab-38/05-invertir-el-tiempo) midió que I3D —que también la tiene rota— predice prácticamente lo mismo con el video invertido, porque Kinetics no lo obliga a distinguirlo. Poder y necesitar son cosas distintas, y la [tabla de TSM](/clases/clase-40/teoria#6-comparación-de-modelos) mide exactamente esa brecha: +3,5 puntos donde el orden no hace falta, +28,0 donde sí.
{{< /concept-alert >}}

---

## 5. En los tres frameworks

El muestreo es cálculo de índices sobre enteros, así que es idéntico en los tres. Lo que cambia es cómo se cargan y agregan los frames.

### PyTorch

```python
import torch
from torch.utils.data import Dataset

class VideoPorSegmentos(Dataset):
    def __init__(self, rutas, K=8, entrenamiento=True, transform=None):
        self.rutas, self.K = rutas, K
        self.entrenamiento, self.transform = entrenamiento, transform

    def __getitem__(self, i):
        frames_disponibles = contar_frames(self.rutas[i])
        idx = (indices_entrenamiento(frames_disponibles, self.K, rng=np.random)
               if self.entrenamiento else
               indices_por_segmentos(frames_disponibles, self.K))
        imgs = [cargar_frame(self.rutas[i], j) for j in idx]
        x = self.transform(imgs)               # (K*3, H, W): apilado por canal
        return x

    def __len__(self):
        return len(self.rutas)
```

El detalle que define la implementación de referencia: los $K$ frames se apilan **sobre el eje de canales** ($K \times 3 = 24$), no en una dimensión temporal separada. Es lo que permite que la ResNet-50 los procese como un lote de imágenes y que el módulo de desplazamiento sea la única pieza que conoce el tiempo.

El consenso, en `nn.Module`:

```python
class ConsensoPromedio(torch.nn.Module):
    def __init__(self, n_segment=8):
        super().__init__()
        self.n_segment = n_segment

    def forward(self, base_out):               # (N*K, n_clases)
        return base_out.view(-1, self.n_segment, base_out.size(1)).mean(dim=1)
```

### TensorFlow

```python
import tensorflow as tf

def dataset_por_segmentos(rutas, K=8, entrenamiento=True):
    def _cargar(ruta):
        n = contar_frames(ruta.numpy().decode())
        idx = (indices_entrenamiento(n, K, rng=np.random) if entrenamiento
               else indices_por_segmentos(n, K))
        return np.stack([cargar_frame(ruta, j) for j in idx])

    ds = tf.data.Dataset.from_tensor_slices(rutas)
    ds = ds.map(lambda r: tf.py_function(_cargar, [r], tf.float32),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

consenso = lambda logits, K=8: tf.reduce_mean(
    tf.reshape(logits, (-1, K, tf.shape(logits)[-1])), axis=1)
```

El `tf.py_function` es necesario porque el número de frames depende del archivo y no se conoce en tiempo de grafo.

### JAX

```python
import jax, jax.numpy as jnp

def indices_entrenamiento_jax(key, n_frames, K=8):
    dur = (n_frames) // K
    offsets = jax.random.randint(key, (K,), 0, dur)
    return jnp.arange(K) * dur + offsets + 1

@jax.jit
def consenso(logits, K=8):                    # (N*K, n_clases)
    return logits.reshape(-1, K, logits.shape[-1]).mean(axis=1)
```

En JAX la aleatoriedad es explícita: la clave `key` se divide por muestra y por época, lo que vuelve el muestreo **exactamente reproducible** — una propiedad útil cuando hay que aislar el efecto de la aumentación temporal de otras fuentes de varianza.

---

## Qué queda establecido

| Afirmación | Verificación | Resultado |
|---|---|---|
| La cobertura por segmentos no depende de la duración | 6 duraciones de 25 a 1775 frames | constante en ~87,5 % |
| La cobertura densa se desmorona | mismas duraciones | de 88 % a 3,2 % |
| El muestreo aleatorio es aumentación real | 10 000 sorteos sobre un clip de 250 | $8{,}5\times10^{11}$ combinaciones, 0 colisiones |
| El consenso por promedio es invariante al orden | las $8! = 40\,320$ permutaciones | diferencia máxima $2{,}2\times10^{-16}$ |
| El desplazamiento rompe esa invarianza | permutación con y sin shift | sin shift: idéntico. Con shift: distinto |

---

## Ver también

- [01 - El módulo de desplazamiento](01-el-modulo-de-desplazamiento) — el otro mecanismo, y de dónde sale la función `temporal_shift` usada acá.
- [Profundización, Parte IV](/clases/clase-40/profundizacion) — la demostración formal de la invarianza al orden.
- [Laboratorio 40](/laboratorios/lab-40/02-la-varianza-intra-clase) — el costo del paso de 31 frames, medido sobre videos reales.
- [Laboratorio 36](/laboratorios/lab-36) — el experimento complementario: un *bag of frames* sin orden alguno que alcanza 85,9 % en UCF-11.
