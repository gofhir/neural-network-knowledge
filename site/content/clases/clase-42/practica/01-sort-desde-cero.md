---
title: "01 - SORT desde cero"
weight: 10
math: true
---

> SORT completo —filtro de Kalman con el estado de siete dimensiones del paper, IoU, algoritmo húngaro y gestión del ciclo de vida— en unas cien líneas de NumPy, más las dos piezas vectorizables en PyTorch, TensorFlow y JAX. Después, cuatro ablaciones sobre una escena sintética que se controla por completo, que es lo que permite atribuir cada error a su causa.

---

## 1. Las dos piezas vectorizables

Dos operaciones concentran todo el cómputo y las dos son puramente algebraicas, así que se escriben igual en cualquier framework: el **IoU por lotes** entre $N$ predicciones y $M$ detecciones, y el **paso predict-update** del filtro de Kalman para las $N$ trayectorias a la vez.

### 1.1. IoU por lotes

{{< tabs >}}
{{< tab name="NumPy" >}}
```python
import numpy as np

def iou_np(a, b):
    """a: (N,4), b: (M,4) en formato (x1,y1,x2,y2). Devuelve (N,M)."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter)
```
{{< /tab >}}
{{< tab name="PyTorch" >}}
```python
import torch

def iou_torch(a, b):
    a, b = torch.as_tensor(a), torch.as_tensor(b)
    x1 = torch.maximum(a[:, None, 0], b[None, :, 0])
    y1 = torch.maximum(a[:, None, 1], b[None, :, 1])
    x2 = torch.minimum(a[:, None, 2], b[None, :, 2])
    y2 = torch.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter)

# Referencia oficial, para contrastar:
from torchvision.ops import box_iou
```
{{< /tab >}}
{{< tab name="TensorFlow" >}}
```python
import tensorflow as tf

def iou_tf(a, b):
    a, b = tf.constant(a), tf.constant(b)
    x1 = tf.maximum(a[:, None, 0], b[None, :, 0])
    y1 = tf.maximum(a[:, None, 1], b[None, :, 1])
    x2 = tf.minimum(a[:, None, 2], b[None, :, 2])
    y2 = tf.minimum(a[:, None, 3], b[None, :, 3])
    inter = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter)
```
{{< /tab >}}
{{< tab name="JAX" >}}
```python
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

@jax.jit
def iou_jax(a, b):
    x1 = jnp.maximum(a[:, None, 0], b[None, :, 0])
    y1 = jnp.maximum(a[:, None, 1], b[None, :, 1])
    x2 = jnp.minimum(a[:, None, 2], b[None, :, 2])
    y2 = jnp.minimum(a[:, None, 3], b[None, :, 3])
    inter = jnp.clip(x2 - x1, 0) * jnp.clip(y2 - y1, 0)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter)
```
{{< /tab >}}
{{< /tabs >}}

Verificación sobre 6 trayectorias contra 8 detecciones aleatorias:

```
=== IoU por lotes: los cuatro backends ===
  PyTorch                    max|dif| vs NumPy = 0.000e+00
  torchvision.ops.box_iou    max|dif| vs NumPy = 0.000e+00
  TensorFlow                 max|dif| vs NumPy = 0.000e+00
  JAX (jit)                  max|dif| vs NumPy = 0.000e+00
```

Cero exacto, no "dentro de la tolerancia": las cuatro implementaciones ejecutan la misma secuencia de operaciones en doble precisión.

### 1.2. Kalman batched

Con el estado de SORT, $x = [u, v, s, r, \dot u, \dot v, \dot s]^\top$, las matrices son

$$F = \begin{bmatrix}
1 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}, \qquad H = \begin{bmatrix} I_4 & 0_{4\times 3}\end{bmatrix}$$

La cuarta fila es la que importa: **no tiene un 1 fuera de la diagonal**, porque $r$ no tiene velocidad asociada. Es la razón de aspecto constante del paper.

{{< tabs >}}
{{< tab name="NumPy" >}}
```python
F = np.eye(7)
for i in range(3):
    F[i, 4 + i] = 1.0          # u += u̇, v += v̇, s += ṡ   (r no tiene velocidad)
H = np.zeros((4, 7)); H[:, :4] = np.eye(4)
Q = np.eye(7) * 0.01
R = np.eye(4) * 1.0

def kalman_np(X, P, Z):
    """X: (N,7)  P: (N,7,7)  Z: (N,4). Un paso predict + update para N trayectorias."""
    Xp = X @ F.T
    Pp = F @ P @ F.T + Q
    y  = Z - Xp @ H.T                       # innovación
    S  = H @ Pp @ H.T + R                   # covarianza de la innovación
    K  = Pp @ H.T @ np.linalg.inv(S)        # ganancia
    Xn = Xp + np.einsum('nij,nj->ni', K, y)
    Pn = (np.eye(7) - K @ H) @ Pp
    return Xn, Pn, S
```
{{< /tab >}}
{{< tab name="PyTorch" >}}
```python
def kalman_torch(X, P, Z):
    Ft, Ht = torch.as_tensor(F), torch.as_tensor(H)
    Qt, Rt = torch.as_tensor(Q), torch.as_tensor(R)
    X, P, Z = map(torch.as_tensor, (X, P, Z))
    Xp = X @ Ft.T
    Pp = Ft @ P @ Ft.T + Qt
    y  = Z - Xp @ Ht.T
    S  = Ht @ Pp @ Ht.T + Rt
    K  = Pp @ Ht.T @ torch.linalg.inv(S)
    Xn = Xp + torch.einsum('nij,nj->ni', K, y)
    Pn = (torch.eye(7, dtype=P.dtype) - K @ Ht) @ Pp
    return Xn, Pn, S
```
{{< /tab >}}
{{< tab name="TensorFlow" >}}
```python
def kalman_tf(X, P, Z):
    Ft, Ht = tf.constant(F), tf.constant(H)
    Qt, Rt = tf.constant(Q), tf.constant(R)
    X, P, Z = tf.constant(X), tf.constant(P), tf.constant(Z)
    Xp = X @ tf.transpose(Ft)
    Pp = Ft @ P @ tf.transpose(Ft) + Qt
    y  = Z - Xp @ tf.transpose(Ht)
    S  = Ht @ Pp @ tf.transpose(Ht) + Rt
    K  = Pp @ tf.transpose(Ht) @ tf.linalg.inv(S)
    Xn = Xp + tf.einsum('nij,nj->ni', K, y)
    Pn = (tf.eye(7, dtype=P.dtype) - K @ Ht) @ Pp
    return Xn, Pn, S
```
{{< /tab >}}
{{< tab name="JAX" >}}
```python
# En JAX se escribe para UNA trayectoria y vmap la replica sobre todas.
def kalman_one(x, P, z):
    Fj, Hj = jnp.array(F), jnp.array(H)
    xp = Fj @ x
    Pp = Fj @ P @ Fj.T + jnp.array(Q)
    y  = z - Hj @ xp
    S  = Hj @ Pp @ Hj.T + jnp.array(R)
    K  = Pp @ Hj.T @ jnp.linalg.inv(S)
    return xp + K @ y, (jnp.eye(7) - K @ Hj) @ Pp, S

kalman_jax = jax.jit(jax.vmap(kalman_one))
```
{{< /tab >}}
{{< /tabs >}}

```
=== Un paso predict+update del filtro de Kalman, para N trayectorias a la vez ===
  PyTorch          max|dif x| = 0.000e+00   max|dif P| = 0.000e+00
  TensorFlow       max|dif x| = 0.000e+00   max|dif P| = 0.000e+00
  JAX (vmap+jit)   max|dif x| = 0.000e+00   max|dif P| = 0.000e+00
```

{{< concept-alert type="clave" >}}
La diferencia interesante entre los backends no es numérica sino **de estilo**. En NumPy, PyTorch y TensorFlow hay que escribir la versión por lotes desde el principio, y aparece el `einsum('nij,nj->ni')` para aplicar $N$ ganancias distintas a $N$ innovaciones — el punto donde es fácil equivocarse de eje.

En JAX se escribe el filtro **para una sola trayectoria**, con la notación matemática literal (`K @ y`, sin índices de lote), y `vmap` produce la versión por lotes. Es el caso de uso canónico de `vmap`: un algoritmo cuya formulación natural es por elemento y cuya ejecución eficiente es por lote.
{{< /concept-alert >}}

## 2. El tracker completo

Con las piezas anteriores, el resto es contabilidad. Las conversiones entre formatos de caja:

```python
def box_to_xysr(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w/2, b[1] + h/2, w*h, w/max(h, 1e-6)])

def xysr_to_box(z):
    u, v, s, r = z
    w = np.sqrt(max(s, 1.0) * max(r, 1e-3)); h = max(s, 1.0) / max(w, 1e-6)
    return np.array([u - w/2, v - h/2, u + w/2, v + h/2])
```

La inicialización sigue al paper — velocidades en cero pero con covarianza grande:

```python
class Track:
    def __init__(self, box):
        self.kf = KF(box_to_xysr(box))
        self.kf.P[4:, 4:] *= 1000.0     # velocidad no observada: incertidumbre alta
        self.kf.P *= 10.0
        self.time_since_update = 0
```

Y el bucle por frame, que es literalmente el algoritmo de la clase:

```python
def run_sort(frames, iou_min=0.3, t_lost=1, use_kf=True, use_hungarian=True):
    tracks, out = [], []
    for dets in frames:
        for t in tracks:
            t.predict()                                    # 2.1 modelo de movimiento
        C = -iou_np(np.array([t.box for t in tracks]),
                    np.array(dets))                        # 2.2 medida de similaridad
        r, c = linear_sum_assignment(C)                    # asignación óptima
        for i, j in zip(r, c):
            if -C[i, j] >= iou_min:                        # umbral de rechazo
                tracks[i].update(dets[j])
        for j in sin_asignar:
            tracks.append(Track(dets[j]))                  # nacen identidades
        tracks = [t for t in tracks if t.time_since_update <= t_lost]   # y mueren
        out.append({t.id: t.box for t in tracks if t.time_since_update == 0})
    return out
```

## 3. La escena de prueba

Una escena sintética donde el *ground truth* se conoce exactamente: dos objetos que se mueven en direcciones opuestas y **se cruzan en el frame 30**, con ruido gaussiano controlable sobre las cajas y oclusión programable.

```python
def scene(n_frames=60, occlude=None, noise=1.0, seed=0):
    """Objeto 0: izquierda→derecha. Objeto 1: derecha→izquierda. Se cruzan en t=30."""
    for t in range(n_frames):
        x0, x1 = 10 + 4.0*t, 250 - 4.0*t
        ...
```

El cruce es el punto crítico: es donde las dos cajas se solapan y donde un tracker sin modelo de movimiento no tiene forma de saber cuál es cuál.

## 4. Ablación 1 — ¿qué aporta el filtro de Kalman?

Se compara SORT completo contra dos variantes: una que usa la **caja del frame anterior** en vez de la predicción (equivale a un modelo de velocidad cero), y otra que reemplaza el húngaro por asignación codiciosa.

| Variante | MOTA | IDF1 | HOTA | ID switches | trayectorias |
|---|---|---|---|---|---|
| SORT completo | 96,67 | **98,33** | **97,53** | 4 | 2 |
| sin Kalman (caja anterior) | **98,33** | **50,00** | 57,74 | 2 | 2 |
| asociación codiciosa | 96,67 | 98,33 | 97,53 | 4 | 2 |

{{< concept-alert type="advertencia" >}}
**La fila del medio es el resultado más instructivo de todo el camino.**

Sin filtro de Kalman, MOTA **sube** (98,33 contra 96,67) y los ID switches **bajan** (2 contra 4). Por las dos métricas que la clase menciona implícitamente, la variante sin filtro es mejor.

IDF1 dice que es un desastre: **50,00 contra 98,33**.

Lo que ocurre es que al cruzarse, las dos cajas se solapan y sin predicción de movimiento el húngaro las **intercambia de forma permanente**. Eso son dos ID switches —uno por objeto— y desde ahí en adelante cada trayectoria sigue al objeto equivocado. MOTA cuenta dos errores en 120 detecciones: 1,7 %. IDF1, que empareja trayectorias completas, ve que la mitad de cada trayectoria está del lado equivocado y castiga proporcionalmente.

SORT completo, en cambio, tiene **más** ID switches (4) pero **breves**: la predicción de velocidad resuelve la ambigüedad del cruce y la identidad se recupera. Cuatro errores momentáneos son mucho mejores que dos permanentes, y el conteo de ID switches no distingue entre esos dos casos.

**Contar cambios de identidad no dice cuánto duran.** Es la razón de existir de IDF1 y de AssA.
{{< /concept-alert >}}

## 5. Ablación 2 — el parámetro $T_{\text{lost}}$

El objeto 0 se ocluye durante un número creciente de frames, y se varía el número de frames que una trayectoria sobrevive sin detecciones.

| Duración de la oclusión | $T_{\text{lost}}$ | MOTA | IDF1 | HOTA | IDs | trayectorias |
|---|---|---|---|---|---|---|
| **3 frames** | 1 | 95,83 | 74,26 | 83,74 | 2 | 3 |
| | 3 | 97,50 | **98,73** | **97,50** | 0 | 2 |
| | 10 | 97,50 | 98,73 | 97,50 | 0 | 2 |
| **8 frames** | 1 | 91,67 | 75,00 | 81,52 | 2 | 3 |
| | 3 | 92,50 | 75,86 | 82,48 | 1 | 3 |
| | 10 | 93,33 | **96,55** | **93,47** | 0 | 2 |
| **18 frames** | 1 | 83,33 | 78,38 | 78,21 | 2 | 3 |
| | 10 | 84,17 | 79,28 | 79,21 | 1 | 3 |
| | 30 | 85,00 | **91,89** | **86,04** | 0 | 2 |

El patrón es nítido: **$T_{\text{lost}}$ tiene que superar la duración de la oclusión** o la identidad se pierde. Y de nuevo, la magnitud del efecto depende de la métrica con que se mire. En la oclusión de 18 frames, pasar de $T_{\text{lost}}=1$ a 30 mueve MOTA **1,7 puntos** y IDF1 **13,5 puntos**.

Esto explica de dónde sale la diferencia entre SORT ($T_{\text{lost}}=1$) y DeepSORT ($A_{\max}=30$) mucho mejor que el descriptor de apariencia. En la escena limpia, subir ese único parámetro ya recupera la identidad; el descriptor solo hace falta cuando el objeto reaparece lejos de donde el modelo lo predice.

**El costo, en la escena real, son falsos positivos.** Aquí no aparecen porque las detecciones sintéticas no tienen ruido espurio; en MOT16 son los 4154 falsos positivos adicionales que [la profundización](/clases/clase-42/profundizacion) cuantifica en −2,28 puntos de MOTA.

## 6. Ablación 3 — húngaro contra codicioso

El caso de libro donde el codicioso falla es fácil de construir:

```python
C = [[1.0, 2.0], [3.0, 100.0]]
# Húngaro:  filas [0,1] -> cols [1,0], costo   5
# Codicioso: toma el 1 y queda obligado al 100, costo 101
```

Pero eso no dice con qué frecuencia importa. Midiéndolo sobre escenas densas, con 20 semillas por configuración:

| Escena | Método | MOTA | IDF1 | HOTA | ID switches | difieren |
|---|---|---|---|---|---|---|
| 12 objetos, $\sigma=1$ px | húngaro | 99,93 | 99,95 | 99,93 | 0,40 | **1 / 20** |
| | codicioso | 99,89 | 99,91 | 99,89 | 0,65 | |
| 25 objetos, $\sigma=6$ px | húngaro | **67,20** | **67,05** | **67,33** | **120,35** | **20 / 20** |
| | codicioso | 62,80 | 62,40 | 63,16 | 145,50 | |

Con objetos bien separados y detecciones limpias, la compuerta de IoU deja tan pocos candidatos por trayectoria que la asignación es casi forzada: las dos estrategias coinciden en 19 de 20 semillas. Con 25 objetos y ruido de 6 px difieren **siempre**, y el húngaro gana **4,4 puntos de MOTA y 25 ID switches** en promedio.

La lección de ingeniería es sobre el diseño del experimento: una comparación hecha en una escena fácil concluye "da lo mismo", y esa conclusión es correcta *y* engañosa. El régimen de densidad al que se va a desplegar el sistema tiene que estar en el banco de pruebas.

## 7. Ablación 4 — el colapso por ruido

Se aumenta el ruido gaussiano sobre las cajas detectadas, sin oclusión:

| $\sigma$ [px] | MOTA | IDF1 | HOTA | ID switches | FN | trayectorias creadas |
|---|---|---|---|---|---|---|
| 0,5 | 100,00 | 100,00 | 100,00 | 0 | 0 | 2 |
| 2,0 | 100,00 | 100,00 | 100,00 | 0 | 0 | 2 |
| 5,0 | 100,00 | 100,00 | 100,00 | 0 | 0 | 2 |
| **10,0** | **43,33** | 58,33 | 53,01 | 10 | 29 | **30** |
| 20,0 | **−78,33** | 3,33 | 4,57 | 14 | 100 | **84** |

{{< concept-alert type="clave" >}}
El colapso **no es gradual**. Entre $\sigma = 5$ y $\sigma = 10$ px, el sistema pasa de perfecto a inutilizable. La causa es el umbral $\mathrm{IoU}_{\min} = 0{,}3$: mientras el ruido mantiene el IoU entre predicción y detección por encima de él, la asociación funciona; cuando lo cruza, **ninguna** asociación se acepta, cada detección crea una trayectoria nueva y el sistema genera 30 identidades para 2 objetos.

Dos observaciones prácticas:

- **MOTA negativa.** Con $\sigma=20$ px, MOTA vale −78,33 porque la fórmula $1 - (\mathrm{FN}+\mathrm{FP}+\mathrm{IDSW})/|\mathrm{gtDet}|$ no está acotada por abajo: hay más errores que objetos verdaderos.
- **El umbral es un acantilado, no una pendiente.** Ajustar $\mathrm{IoU}_{\min}$ mirando una secuencia limpia produce un valor que funciona hasta que deja de funcionar por completo. Conviene medir el margen —cuánto ruido tolera antes del salto— y no solo el rendimiento en el punto de operación.
{{< /concept-alert >}}

---

## Qué se aprendió

1. **El IoU y el filtro de Kalman son álgebra pura**, idéntica en los cuatro backends hasta cero exacto. Lo que cambia es el estilo: `einsum` con eje de lote explícito contra `vmap` sobre la formulación de un elemento.
2. **Sin modelo de movimiento las identidades se intercambian al cruzarse**, y MOTA no lo nota — de hecho mejora. IDF1 cae a la mitad.
3. **$T_{\text{lost}}$ explica más de la diferencia SORT/DeepSORT que el descriptor de apariencia**, al menos cuando el objeto reaparece cerca de donde se predijo.
4. **El algoritmo húngaro es irrelevante en escenas fáciles y vale 4,4 puntos de MOTA en escenas densas.**
5. **El umbral de IoU produce un colapso abrupto**, no una degradación suave.

---

**Siguiente:** [02 - Las tres métricas y lo que esconden](02-metricas-mot) — MOTA, IDF1 y HOTA desde su definición, y por qué la comparación SORT/DeepSORT que la clase presenta se ve completamente distinta según con qué se mida.
