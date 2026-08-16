---
title: "02 - Agregación VLAD"
weight: 20
math: true
---

> Implementar [VLAD](/papers/vlad-jegou-2010) y [NetVLAD](/papers/netvlad-arandjelovic-2016) desde cero, comprobar que el promedio es su caso degenerado, construir un ejemplo donde el promedio **no puede distinguir** lo que VLAD separa perfectamente, y cerrar el circuito calculando el EER sobre una curva ROC — que es la métrica con la que se reporta todo el reconocimiento de hablante.

---

## 1. Los dos agregadores

Un agregador toma $N$ descriptores de dimensión $d$ —con $N$ variable— y devuelve un vector fijo.

```python
import numpy as np

def mean_pool(X):
    """X: (N, d) -> (d,)"""
    return X.mean(0)

def vlad(X, C, tau=None):
    """X: (N, d) descriptores.  C: (K, d) centroides.  -> (K*d,)
    tau=None -> asignación dura (VLAD).   tau>0 -> asignación blanda (NetVLAD)."""
    if tau is None:
        a = np.zeros((len(X), len(C)))
        a[np.arange(len(X)), np.argmin(((X[:, None] - C[None])**2).sum(-1), 1)] = 1
    else:
        s = -((X[:, None] - C[None])**2).sum(-1) / tau
        e = np.exp(s - s.max(1, keepdims=True))
        a = e / e.sum(1, keepdims=True)

    V = np.einsum('nk,nkd->kd', a, X[:, None, :] - C[None]).ravel()
    n = np.linalg.norm(V)
    return V / n if n > 0 else V
```

El `einsum` es la fórmula de la clase, $v(j,k) = \sum_i a_k(x_i)\,(x_i(j) - c_k(j))$, escrita de un tirón: pondera cada residuo por su pertenencia y suma sobre los descriptores.

---

## 2. El promedio es VLAD con un centroide en el origen

Antes de comparar conviene notar que no son métodos rivales: uno es caso particular del otro. Con $K=1$ y $c_1 = 0$:

$$v = \sum_{i} (x_i - 0) = N\,\bar{x}$$

que tras normalizar en L2 **es el promedio normalizado**:

```python
X = np.random.default_rng(0).normal(size=(200, 4))
V = X.sum(0);  V /= np.linalg.norm(V)
m = X.mean(0); m /= np.linalg.norm(m)
print(np.allclose(V, m))     # True
```

*Average pooling* no es una alternativa a VLAD: es **su versión más pobre**, con un diccionario de un solo elemento colocado en un punto arbitrario. Toda la capacidad extra viene de tener varios prototipos y de que estén donde están los datos.

---

## 3. El caso que el promedio no puede resolver

Un promedio descarta todo salvo el primer momento, así que dos conjuntos con la misma media son indistinguibles **por definición**. Construyamos ese caso.

Diccionario de dos prototipos en $c_1 = (-2, 0)$ y $c_2 = (2, 0)$, y dos "hablantes" que ocupan **los mismos clusters** pero se desvían en direcciones opuestas:

```python
rng = np.random.default_rng(0)
C     = np.array([[-2.0, 0.0], [2.0, 0.0]])
delta = np.array([0.0, 0.6])

def emite(tipo, n=400, s=0.15):
    """A: c1+delta y c2-delta.   B: al revés.  => misma media global."""
    d1, d2 = (delta, -delta) if tipo == 'A' else (-delta, delta)
    base = np.vstack([np.repeat((C[0] + d1)[None], n//2, 0),
                      np.repeat((C[1] + d2)[None], n//2, 0)])
    r = rng.normal(0, s, (n, 2)); r -= r.mean(0)     # ruido de media exacta cero
    return base + r

A1, A2, B1, B2 = emite('A'), emite('A'), emite('B'), emite('B')
for nom, X in [('A1',A1), ('A2',A2), ('B1',B1), ('B2',B2)]:
    print(f"   {nom}: {mean_pool(X).round(6)}")
```

```
   A1: [0. 0.]      A2: [-0. -0.]
   B1: [-0. -0.]    B2: [-0. -0.]
```

Las cuatro muestras tienen **exactamente** la misma media. Ahora la comparación, midiendo similitud del coseno entre muestras del mismo "hablante" y de distintos:

```python
cos = lambda u, v: float(u @ v / (np.linalg.norm(u)*np.linalg.norm(v) + 1e-12))

print(f"{'método':16} {'mismo':>10} {'distinto':>10} {'margen':>9} {'dim':>5}")
for nom, f in [('mean pooling',   mean_pool),
               ('VLAD (hard)',    lambda X: vlad(X, C)),
               ('NetVLAD (soft)', lambda X: vlad(X, C, tau=0.5))]:
    sm = (cos(f(A1), f(A2)) + cos(f(B1), f(B2))) / 2
    sd = (cos(f(A1), f(B1)) + cos(f(A2), f(B2))) / 2
    print(f"{nom:16} {sm:10.4f} {sd:10.4f} {sm-sd:9.4f} {f(A1).size:5d}")
```

```
método                mismo   distinto    margen   dim
mean pooling         0.0000    -0.0000    0.0000     2
VLAD (hard)          0.9999    -0.9999    1.9998     4
NetVLAD (soft)       0.9999    -0.9999    1.9998     4
```

El promedio colapsa todo al vector nulo: **margen cero, no distingue nada**. VLAD separa con el margen máximo posible. Y los vectores muestran dónde quedó la información:

```python
print("A:", vlad(A1, C).round(3))    # [-0.005  0.707  0.005 -0.707]
print("B:", vlad(B1, C).round(3))    # [-0.001 -0.707  0.001  0.707]
```

Las componentes del eje $x$ (1ª y 3ª) son ~0: en esa dirección los descriptores **sí** están centrados en sus prototipos, y no hay nada que codificar. Toda la información discriminativa está en las del eje $y$, con signo invertido entre A y B.

{{< concept-alert type="clave" >}}
El caso está construido para que el promedio falle del todo, y en datos reales las medias nunca coinciden con esa exactitud. Pero el **mecanismo** es el que opera en el problema real: promediar sobre un enunciado con ruido y silencios produce un centro de masa que se mueve mucho entre grabaciones de la misma persona. Es lo que [Xie et al. (2019)](/papers/utterance-level-xie-2019) miden como 10,48 % contra 3,57 % de EER con el mismo backbone.
{{< /concept-alert >}}

---

## 4. NetVLAD generaliza VLAD

La asignación blanda converge a la dura cuando baja la temperatura:

```python
h = vlad(A1, C)
for tau in (5.0, 1.0, 0.3, 0.1, 0.01):
    print(f"   tau={tau:<5}  cos(NetVLAD_tau, VLAD_hard) = {cos(vlad(A1, C, tau=tau), h):.6f}")
```

```
   tau=5.0    cos(NetVLAD_tau, VLAD_hard) = 0.960035
   tau=1.0    cos(NetVLAD_tau, VLAD_hard) = 1.000000
   tau=0.3    cos(NetVLAD_tau, VLAD_hard) = 1.000000
```

La convergencia es rápida acá porque los dos prototipos están muy separados (distancia 4); con un diccionario denso hace falta $\tau$ bastante más chico.

La versión de NetVLAD que se usa en la práctica parametriza el softmax con pesos propios en vez de con las distancias:

$$\bar{a}_k(x) = \frac{e^{\,w_k^\top x + b_k}}{\sum_{k'} e^{\,w_{k'}^\top x + b_{k'}}}$$

lo que **desacopla** el criterio de asignación de la posición del centroide. Con $w_k = 2c_k/\tau$ y $b_k = -\lVert c_k\rVert^2/\tau$ se recupera el caso de arriba; dejándolos libres, la capa aprende un criterio distinto.

---

## 5. La capa, en los tres frameworks

### PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetVLAD(nn.Module):
    def __init__(self, K=8, D=512, G=0):
        """K clusters reales, G clusters fantasma (GhostVLAD)."""
        super().__init__()
        self.K, self.G = K, G
        self.conv = nn.Conv1d(D, K + G, kernel_size=1)      # w_k y b_k
        self.centroids = nn.Parameter(torch.randn(K + G, D) * 0.01)

    def forward(self, x):                                   # x: (N, D, T)
        a = F.softmax(self.conv(x), dim=1)                  # (N, K+G, T)
        # residuos: (N, K+G, D)
        V = torch.einsum('nkt,ntd->nkd', a, x.transpose(1, 2)) \
            - a.sum(-1).unsqueeze(-1) * self.centroids.unsqueeze(0)
        V = V[:, :self.K]                                   # descartar los fantasma
        V = F.normalize(V, dim=-1)                          # L2 intra-cluster
        return F.normalize(V.flatten(1), dim=-1)            # L2 global
```

Dos detalles que importan. El **einsum reescrito**: en vez de materializar el tensor de residuos $(N, T, K, D)$ —que para $T=250$, $K=8$, $D=512$ son millones de valores por muestra— se usa la identidad

$$\sum_t a_{k,t}(x_t - c_k) = \sum_t a_{k,t} x_t - c_k \sum_t a_{k,t}$$

que evita construirlo. Y la **doble normalización**: primero dentro de cada cluster, después sobre el vector completo, para que un cluster muy poblado no domine.

Los `G` clusters fantasma participan del softmax pero se descartan antes de concatenar: los frames irrelevantes pueden depositar ahí su peso.

### TensorFlow

```python
import tensorflow as tf

class NetVLAD(tf.keras.layers.Layer):
    def __init__(self, K=8, G=0, **kw):
        super().__init__(**kw)
        self.K, self.G = K, G

    def build(self, shape):                                 # (N, T, D)
        D = shape[-1]
        self.conv = tf.keras.layers.Conv1D(self.K + self.G, 1)
        self.centroids = self.add_weight('centroids', (self.K + self.G, D),
                                         initializer='random_normal')

    def call(self, x):                                      # (N, T, D)
        a = tf.nn.softmax(self.conv(x), axis=-1)            # (N, T, K+G)
        V = tf.einsum('ntk,ntd->nkd', a, x) \
            - tf.reduce_sum(a, 1)[..., None] * self.centroids
        V = V[:, :self.K]
        V = tf.nn.l2_normalize(V, axis=-1)
        return tf.nn.l2_normalize(tf.reshape(V, (tf.shape(x)[0], -1)), axis=-1)
```

### JAX / Flax

```python
import jax.numpy as jnp
import flax.linen as fnn

class NetVLAD(fnn.Module):
    K: int = 8
    G: int = 0

    @fnn.compact
    def __call__(self, x):                                  # (N, T, D)
        D = x.shape[-1]
        a = fnn.softmax(fnn.Dense(self.K + self.G)(x), axis=-1)
        c = self.param('centroids', fnn.initializers.normal(0.01), (self.K + self.G, D))
        V = jnp.einsum('ntk,ntd->nkd', a, x) - a.sum(1)[..., None] * c
        V = V[:, :self.K]
        V = V / (jnp.linalg.norm(V, axis=-1, keepdims=True) + 1e-12)
        V = V.reshape(V.shape[0], -1)
        return V / (jnp.linalg.norm(V, axis=-1, keepdims=True) + 1e-12)
```

---

## 6. Del descriptor al veredicto: ROC y EER

Con descriptores normalizados, el puntaje es el producto punto. Falta el paso que la clase plantea y no cierra: **elegir el umbral**.

```python
def curva_roc(scores, etiquetas, n=1000):
    """etiquetas: 1 = mismo hablante, 0 = distinto."""
    umbrales = np.linspace(0, 1, n)
    P, N = (etiquetas == 1).sum(), (etiquetas == 0).sum()
    TPR = np.array([((scores >= u) & (etiquetas == 1)).sum() / P for u in umbrales])
    FPR = np.array([((scores >= u) & (etiquetas == 0)).sum() / N for u in umbrales])
    return umbrales, TPR, FPR

def eer(umbrales, TPR, FPR):
    FNR = 1 - TPR
    i = np.argmin(np.abs(FNR - FPR))
    return (FNR[i] + FPR[i]) / 2, umbrales[i]
```

Sobre puntajes sintéticos (2 000 pares del mismo hablante, 2 000 de distintos):

```python
rng = np.random.default_rng(1)
s_mismo    = np.clip(rng.normal(0.75, 0.12, 2000), 0, 1)
s_distinto = np.clip(rng.normal(0.35, 0.15, 2000), 0, 1)
scores = np.r_[s_mismo, s_distinto]
etiq   = np.r_[np.ones(2000), np.zeros(2000)]

u, TPR, FPR = curva_roc(scores, etiq)
e, umbral = eer(u, TPR, FPR)
print(f"EER = {e*100:.2f}%  en umbral {umbral:.3f}")
print(f"AUC = {np.trapezoid(TPR[::-1], FPR[::-1]):.4f}")
```

```
EER = 6.57%  en umbral 0.572   (FPR=6.55%, FNR=6.60%)
AUC = 0.9810
```

### El EER no es un punto de operación

Este es el punto que conviene no perder de vista. El EER resume el sistema en un número **eligiendo el umbral donde ambos errores se igualan**, y eso casi nunca es lo que se quiere en producción:

```python
for objetivo in (0.01, 0.05, 0.10):
    j = np.argmin(np.abs(FPR - objetivo))
    print(f"   FPR={FPR[j]*100:5.2f}%  ->  TPR={TPR[j]*100:5.2f}%  umbral={u[j]:.3f}")
```

```
   FPR= 1.00%  ->  TPR=65.75%  umbral=0.700
   FPR= 5.10%  ->  TPR=90.50%  umbral=0.592
   FPR=10.00%  ->  TPR=96.10%  umbral=0.540
```

El **mismo sistema**, con EER de 6,57 %, opera al 65,75 % de aciertos si se exige 1 % de falsos positivos, y al 96,10 % si se toleran 10 %. En control de acceso a un sistema clínico el falso positivo —dejar entrar a quien no es— es mucho más caro que el falso negativo —pedir que se reintente—, así que el punto de operación va arriba a la izquierda de la curva, no en el EER.

{{< concept-alert type="cuidado" >}}
El EER sirve para **comparar sistemas**, no para desplegarlos. Reportar "3,22 % de EER" dice cuán separables son las distribuciones; no dice a qué umbral operar, y elegirlo requiere conocer el costo relativo de cada error. Las evaluaciones NIST usan por eso el **DCF**, que pondera explícitamente ambos errores según la aplicación.
{{< /concept-alert >}}

---

## Qué queda establecido

| Afirmación | Verificación | Resultado |
|---|---|---|
| El promedio es VLAD con $K=1$, $c=0$ | comparación tras normalizar | identidad exacta |
| Existe un caso que el promedio no distingue | medias idénticas por construcción | margen 0,0000 contra 1,9998 |
| NetVLAD converge a VLAD duro | barrido de temperatura | coseno 1,000000 con $\tau \leq 1$ |
| La dimensión de VLAD es $K \times d$ | conteo directo | 4 contra 2 del promedio |
| El EER no determina el punto de operación | tres objetivos de FPR | TPR de 65,75 % a 96,10 % |

---

## Ver también

- [01 - CTC desde cero](01-ctc-desde-cero) — el mecanismo de la primera mitad de la clase.
- [Profundización, Partes IV-V](/clases/clase-41/profundizacion) — por qué el `argmin` bloquea el gradiente y la geometría de los residuos.
- [Fundamento: Agregación VLAD](/fundamentos/agregacion-vlad) · [Reconocimiento de hablante](/fundamentos/reconocimiento-de-hablante).
- [Paper: VLAD (2010)](/papers/vlad-jegou-2010) · [NetVLAD (2016)](/papers/netvlad-arandjelovic-2016) · [Utterance-level Aggregation (2019)](/papers/utterance-level-xie-2019).
