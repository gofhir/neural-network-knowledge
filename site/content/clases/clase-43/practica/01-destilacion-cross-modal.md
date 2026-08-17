---
title: "01 - Destilación cross-modal"
weight: 10
math: true
---

> La ablación de SoundNet reporta 72,9 % con pérdida KL y 47,8 % con $\ell_2$ sobre ESC-50. Veinticinco puntos. Pero [Hinton, Vinyals y Dean (2015)](/papers/distillation-hinton-2015) demostraron que a temperatura alta las dos pérdidas son **equivalentes**. Este camino implementa ambas, verifica el teorema, y encuentra que la contradicción se disuelve al notar que "$\ell_2$" nombra dos cosas distintas.

---

## 1. Dark knowledge: qué hay fuera del argmax

El punto de partida de toda la destilación es que una distribución de salida contiene más que su máximo. Con logits del maestro $[6{,}0,\; 2{,}0,\; 1{,}8,\; -1{,}0,\; -3{,}0]$ sobre las clases *perro, lobo, zorro, auto, silla*:

```python
def softmax(z, T=1.0):
    z = z / T
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)

v = np.array([6.0, 2.0, 1.8, -1.0, -3.0])
for T in [1.0, 2.0, 5.0, 10.0]:
    p = softmax(v, T)
    print(T, p, -(p * np.log(p)).sum())
```

| $T$ | perro | lobo | zorro | auto | silla | entropía |
|---|---|---|---|---|---|---|
| 1 | 0,9668 | 0,0177 | 0,0145 | 0,0009 | 0,0001 | 0,173 nats |
| 2 | 0,7698 | 0,1042 | 0,0943 | 0,0232 | 0,0086 | 0,788 nats |
| 5 | 0,4361 | 0,1960 | 0,1883 | 0,1075 | 0,0721 | 1,425 nats |
| 10 | 0,3095 | 0,2075 | 0,2034 | 0,1537 | 0,1259 | 1,562 nats |

A $T=1$ el maestro dice *perro* y poco más. A $T=5$ se hace visible que **lobo y zorro están casi empatados entre sí y muy por encima de auto** — la estructura relativa que la etiqueta `perro` destruye.

## 2. La pérdida, en cuatro backends

La implementación de referencia y su gradiente analítico:

```python
def kd_loss(z, v, T):
    p, q = softmax(v, T), softmax(z, T)
    return (p * (np.log(p) - np.log(q))).sum(-1).mean() * T**2

def kd_grad(z, v, T):
    return (softmax(z, T) - softmax(v, T)) / T * T**2 / z.shape[0]
```

El factor $T^2$ es la corrección de Hinton: los gradientes de los objetivos blandos escalan como $1/T^2$, así que sin multiplicar por $T^2$ subir la temperatura equivaldría a bajar la tasa de aprendizaje.

{{< tabs >}}
{{< tab name="PyTorch" >}}
```python
import torch, torch.nn.functional as F

zt = torch.tensor(z, requires_grad=True)
vt = torch.tensor(v)
L = F.kl_div(F.log_softmax(zt / T, -1),
             F.log_softmax(vt / T, -1),
             reduction="batchmean", log_target=True) * T**2
L.backward()          # zt.grad
```
Ojo con `F.kl_div`: espera **log-probabilidades** en el primer argumento, y `log_target=True` indica que el segundo también lo es. Es la fuente de error más común al implementar destilación en PyTorch.
{{< /tab >}}
{{< tab name="TensorFlow" >}}
```python
import tensorflow as tf

zf = tf.Variable(z); vf = tf.constant(v)
with tf.GradientTape() as tape:
    p    = tf.nn.softmax(vf / T)
    logq = tf.nn.log_softmax(zf / T)
    L = tf.reduce_mean(tf.reduce_sum(p * (tf.math.log(p) - logq), -1)) * T**2
g = tape.gradient(L, zf)
```
{{< /tab >}}
{{< tab name="JAX" >}}
```python
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

def kd_loss_jax(z, v, T):
    logp = jax.nn.log_softmax(v / T)
    logq = jax.nn.log_softmax(z / T)
    return jnp.mean(jnp.sum(jnp.exp(logp) * (logp - logq), -1)) * T**2

L = kd_loss_jax(z, v, T)
g = jax.grad(kd_loss_jax)(z, v, T)      # gradiente por diferenciación automática
```
{{< /tab >}}
{{< /tabs >}}

```
NumPy      loss = 7.2431250628
PyTorch    loss = 7.2431250628   dif = 0.00e+00   grad max|dif| = 4.16e-17
TensorFlow loss = 7.2431250628   dif = 0.00e+00   grad max|dif| = 4.16e-17
JAX        loss = 7.2431250628   dif = 0.00e+00   grad max|dif| = 4.16e-17
```

La pérdida coincide **exactamente** y el gradiente hasta el épsilon de máquina — incluido el de JAX, que se obtiene por diferenciación automática y no de la fórmula cerrada.

## 3. El teorema de Hinton, verificado

El resultado teórico: en el límite de $T$ grande, con logits centrados en cero,

$$\frac{\partial C}{\partial z_i} \;\approx\; \frac{1}{NT^2}(z_i - v_i)$$

que es el gradiente de $\tfrac{1}{2}(z_i - v_i)^2$. Comparando el gradiente exacto de KL con esa aproximación:

```python
def grad_kl(z, v, T):        return (softmax(z, T) - softmax(v, T)) / T
def grad_l2_logits(z, v, T): return (z - v) / (N * T**2)
```

| $T$ | 1 | 2 | 5 | 10 | 25 | 100 | 1000 |
|---|---|---|---|---|---|---|---|
| coseno | 0,9557 | 0,9739 | 0,9924 | 0,9977 | 0,9996 | 0,99997 | **1,000000** |
| razón de normas | 1,1841 | 1,1622 | 1,0770 | 1,0393 | 1,0158 | 1,0039 | 1,0004 |

El teorema se cumple, y la convergencia es rápida: a $T=10$ el coseno ya es 0,998. **A $T=1$ no**, que es la temperatura a la que destila SoundNet.

## 4. La pregunta: entonces, ¿de dónde salen 25 puntos?

Si a temperatura alta ambas pérdidas coinciden y a $T=1$ el coseno es 0,956 —bastante alto—, la brecha de SoundNet parece demasiado grande para explicarse solo por la temperatura.

La clave está en una línea del paper de SoundNet: probaron *"pérdida $\ell_2$ sobre las **salidas objetivo**"*. Sobre las **probabilidades**, no sobre los logits. Y esas dos pérdidas se comportan de forma completamente distinta.

```python
# L2 sobre logits: no atraviesa el softmax
g_logits = 2 * (z - v) / K

# L2 sobre probabilidades: hay que pasar por el jacobiano del softmax
d = 2 * (q - p)
g_probs = q * (d - (d * q).sum(-1, keepdims=True))
```

El factor $q$ del frente en la segunda es el problema: **donde el estudiante asigna probabilidad casi nula, el gradiente es casi nulo**, por equivocada que esté esa clase.

## 5. El experimento

Montaje que replica la situación de SoundNet: **muchas clases** y un maestro **concentrado** —lo que produce un clasificador de ImageNet aplicado a fotogramas de Flickr, donde una imagen activa unas pocas categorías y las demás son ruido.

```python
K, N, D = 400, 3000, 32
logits_true = np.clip(X @ W_true * 6.0, -60, 60)   # maestro concentrado
p_teacher   = softmax(logits_true, 1.0)
```

```
Clases: 400. Masa media en el top-5 del maestro: 0.9253
Probabilidad mediana fuera del top-5: 1.60e-08
```

Entrenando un estudiante lineal con cada pérdida:

| Pérdida | top-1 | solape top-5 | correlación de rango |
|---|---|---|---|
| KL, $T=1$ | 67,87 % | 56,15 % | 0,8577 |
| KL, $T=2$ | 62,43 % | 59,85 % | 0,9385 |
| KL, $T=4$ | 57,10 % | 65,09 % | 0,9837 |
| **$\ell_2$ sobre probabilidades** | **7,57 %** | 8,45 % | 0,1674 |
| **$\ell_2$ sobre logits** | **92,57 %** | 94,36 % | 0,9987 |

{{< concept-alert type="clave" >}}
**Los 25 puntos de SoundNet no miden "KL contra mínimos cuadrados".** Miden "objetivo que atraviesa un softmax saturado contra objetivo que no". Con $\ell_2$ sobre **logits**, la brecha no solo desaparece: se invierte, y esa pérdida gana a todas las variantes de KL.

Al leer «usamos pérdida L2» en un paper de destilación hay que preguntar **sobre qué**. Las dos variantes comparten nombre y no comparten comportamiento.
{{< /concept-alert >}}

## 6. Por qué se satura, medido

Magnitud del gradiente por clase, con el estudiante sin entrenar (distribución uniforme) y el maestro concentrado:

| rango de la clase | $p_{\text{maestro}}$ | $\lvert\nabla_{\mathrm{KL}}\rvert$ | $\lvert\nabla_{\ell_2\text{ probs}}\rvert$ | razón |
|---|---|---|---|---|
| 0 | 2,594e−01 | 2,569e−01 | 1,285e−03 | 200,0× |
| 1 | 1,829e−01 | 1,804e−01 | 9,018e−04 | 200,0× |
| 4 | 5,323e−02 | 5,073e−02 | 2,536e−04 | 200,0× |
| 20 | 2,747e−03 | 2,469e−04 | 1,235e−06 | 200,0× |
| 100 | 2,947e−05 | 2,471e−03 | 1,235e−05 | 200,0× |
| 399 | 5,783e−14 | 2,500e−03 | 1,250e−05 | 200,0× |

La razón es **exactamente $K/2 = 200$** en todas las filas: el jacobiano del softmax introduce un factor $q \approx 1/K$ que aplasta el gradiente de forma uniforme. Con 1401 salidas, como en SoundNet, el factor sería 700.

Con una tasa de aprendizaje ajustada para KL, la rama de $\ell_2$ sobre probabilidades avanza dos órdenes de magnitud más lento — y en un presupuesto de entrenamiento fijo, eso solo se ve como "esta pérdida funciona peor".

## 7. El intercambio de la temperatura

De vuelta a la tabla de la sección 5, en la familia KL:

| $T$ | top-1 | correlación de rango |
|---|---|---|
| 1 | **67,87 %** | 0,8577 |
| 2 | 62,43 % | 0,9385 |
| 4 | 57,10 % | **0,9837** |

Subir la temperatura **baja el top-1** y **sube la correlación de rango**. No es una contradicción: son dos objetivos distintos. Una temperatura alta enseña la estructura relativa completa del espacio de clases a costa de la decisión puntual.

{{< concept-alert type="recordar" >}}
Cuál conviene depende de para qué se quiere al estudiante. Si va a ser un **clasificador**, importa el top-1. Si va a ser un **extractor de features** —que es exactamente el caso de SoundNet, donde se descarta la capa de salida y se usa pool5 con un SVM encima— importa la estructura.

Visto así, SoundNet destila a $T=1$ optimizando un objetivo que no es el que finalmente evalúa. Es una hipótesis que la ablación del paper no explora y que este montaje sugiere que valdría la pena.
{{< /concept-alert >}}

---

## Qué se aprendió

1. **La pérdida de destilación es idéntica en los cuatro backends** hasta cero exacto, y su gradiente hasta $4\times10^{-17}$ — incluida la versión de JAX por diferenciación automática.
2. **La temperatura revela estructura**: la entropía del objetivo pasa de 0,173 a 1,562 nats entre $T=1$ y $T=10$.
3. **El teorema de Hinton se verifica**: el coseno entre el gradiente de KL y el de mínimos cuadrados sobre logits va de 0,956 a 1,000000.
4. **"$\ell_2$" nombra dos pérdidas distintas.** Sobre probabilidades colapsa (7,57 %); sobre logits gana (92,57 %). La diferencia es el jacobiano del softmax, y el factor es exactamente $K/2$.
5. **Temperatura alta cambia qué se aprende**, no solo cuánto: menos top-1, más estructura relativa.

---

**Siguiente:** [02 - Fusión audiovisual bajo ruido](02-fusion-audiovisual) — la curva que justifica el segundo paper de la clase, reconstruida desde cero.
