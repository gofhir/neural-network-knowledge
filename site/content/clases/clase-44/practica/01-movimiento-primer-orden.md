---
title: "01 - Movimiento de primer orden"
weight: 10
math: true
---

> La clase describe el método en cuatro pasos: seguir puntos, estimar la transformación afín de cada uno, entrenar para regenerar el video, y reemplazar el cuadro inicial. Este camino implementa el corazón de eso —la representación del movimiento y el *warping*— y responde la pregunta de diseño que el nombre del paper plantea: **¿cuánto vale el jacobiano?**

---

## 1. La representación

FOMM guarda, por cada punto clave $k$: su posición en la imagen fuente, su posición en el cuadro conductor, y un **jacobiano** $J_k \in \mathbb{R}^{2\times 2}$. El campo denso se obtiene combinando las aproximaciones locales de primer orden con pesos que dependen de la distancia:

$$\hat{\mathcal{T}}(p) = \sum_{k=1}^{K} w_k(p)\Big[\,p_k^{S} + J_k\,(p - p_k^{D})\,\Big], \qquad w_k(p) \propto \exp\!\left(-\frac{\lVert p - p_k^{D}\rVert^2}{2\sigma^2}\right)$$

El término $J_k(p - p_k^D)$ es lo que agrega el orden 1. Sin él, cada punto aporta una traslación constante en su vecindad.

{{< tabs >}}
{{< tab name="NumPy" >}}
```python
def campo(grid, kp, kp_d, jac, sigma):
    """grid: (H,W,2)  kp, kp_d: (K,2)  jac: (K,2,2)  ->  (H,W,2)"""
    d = grid[:, :, None, :] - kp_d[None, None, :, :]        # (H,W,K,2)
    w = np.exp(-(d**2).sum(-1) / (2*sigma**2))              # (H,W,K)
    w = w / w.sum(-1, keepdims=True)
    loc = kp[None, None, :, :] + np.einsum('kij,hwkj->hwki', jac, d)
    return (w[..., None] * loc).sum(2)
```
{{< /tab >}}
{{< tab name="PyTorch" >}}
```python
def campo_torch(grid, kp, kp_d, jac, sigma):
    g, k, kd, J = map(torch.as_tensor, (grid, kp, kp_d, jac))
    d = g[:, :, None, :] - kd[None, None, :, :]
    w = torch.exp(-(d**2).sum(-1) / (2*sigma**2))
    w = w / w.sum(-1, keepdim=True)
    loc = k[None, None, :, :] + torch.einsum('kij,hwkj->hwki', J, d)
    return (w[..., None] * loc).sum(2)
```
{{< /tab >}}
{{< tab name="TensorFlow" >}}
```python
def campo_tf(grid, kp, kp_d, jac, sigma):
    g, k, kd, J = map(tf.constant, (grid, kp, kp_d, jac))
    d = g[:, :, None, :] - kd[None, None, :, :]
    w = tf.exp(-tf.reduce_sum(d**2, -1) / (2*sigma**2))
    w = w / tf.reduce_sum(w, -1, keepdims=True)
    loc = k[None, None, :, :] + tf.einsum('kij,hwkj->hwki', J, d)
    return tf.reduce_sum(w[..., None] * loc, 2)
```
{{< /tab >}}
{{< tab name="JAX" >}}
```python
# Se escribe para UN pixel; dos vmap anidados lo aplican a toda la imagen.
def campo_local(p, kp, kp_d, jac, sigma):
    d = p - kp_d                                   # (K,2)
    w = jnp.exp(-(d**2).sum(-1) / (2*sigma**2))
    w = w / w.sum()
    loc = kp + jnp.einsum('kij,kj->ki', jac, d)
    return (w[:, None] * loc).sum(0)

campo_jax = jax.jit(jax.vmap(jax.vmap(campo_local,
                    in_axes=(0, None, None, None, None)),
                    in_axes=(0, None, None, None, None)))
```
{{< /tab >}}
{{< /tabs >}}

```
=== Campo de movimiento de primer orden ===
  PyTorch                max|dif| vs NumPy = 2.776e-16
  TensorFlow             max|dif| vs NumPy = 2.220e-16
  JAX (vmap anidado)     max|dif| vs NumPy = 2.220e-16
```

{{< concept-alert type="clave" >}}
La versión de JAX vuelve a mostrar el patrón de las clases anteriores: se escribe la fórmula **para un píxel**, tal como aparece en el paper, y `vmap` anidado —uno por cada eje espacial— produce la versión por lotes. En los otros tres backends hay que escribir directamente la forma tensorial, con el `einsum('kij,hwkj->hwki')` que aplica $K$ jacobianos distintos a $K$ vectores por píxel; es el punto donde es fácil confundir un eje.
{{< /concept-alert >}}

## 2. El warping

Con el campo, se deforma la imagen fuente por muestreo bilineal:

```python
def warp(img, campo):
    u = np.clip((campo[..., 0] + .5) * (W - 1), 0, W - 1)
    v = np.clip((campo[..., 1] + .5) * (H - 1), 0, H - 1)
    x0, y0 = np.floor(u).astype(int), np.floor(v).astype(int)
    x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)
    a, b = u - x0, v - y0
    return (img[y0, x0]*(1-a)*(1-b) + img[y0, x1]*a*(1-b) +
            img[y1, x0]*(1-a)*b     + img[y1, x1]*a*b)
```

```
PyTorch grid_sample      max|dif| vs NumPy = 2.220e-16
```

La implementación manual coincide con `torch.nn.functional.grid_sample`, que es la que usa FOMM.

{{< concept-alert type="recordar" >}}
Esto explica una decisión del paper que suele pasar sin comentario: FOMM estima **flujo óptico hacia atrás** ($\mathcal{T}_{S\leftarrow D}$, del conductor a la fuente) y no hacia adelante.

Con flujo hacia atrás, cada píxel de salida **lee** de la fuente por interpolación bilineal: es una operación de *gather*, diferenciable y con kernel optimizado. Con flujo hacia adelante, cada píxel de entrada **escribe** en la salida, lo que produce colisiones y huecos, y exige una dispersión (*scatter*) que no es diferenciable de forma natural.

La dirección del flujo no se eligió por conveniencia conceptual sino por **diferenciabilidad**.
{{< /concept-alert >}}

## 3. ¿Cuánto vale el jacobiano?

El experimento: aproximar campos de movimiento suaves **arbitrarios** —generados como ruido filtrado, deliberadamente fuera de la familia que la representación puede expresar— y ajustar los parámetros por mínimos cuadrados en ambos casos.

```python
def base(centros, sigma, orden):
    """orden 0: 1 columna por keypoint (2 params)
       orden 1: 3 columnas por keypoint (6 params)"""
    ...
def ajusta(campo, centros, sigma, orden):
    B = base(centros, sigma, orden)
    coef, *_ = np.linalg.lstsq(B, campo, rcond=None)   # el MEJOR ajuste posible
    return np.abs(B @ coef - campo).mean()
```

| $K$ | params o.0 | error o.0 | params o.1 | error o.1 | mejora |
|---|---|---|---|---|---|
| 4 | 8 | 0,02879 | 24 | 0,01689 | 1,71× |
| 6 | 12 | 0,02436 | 36 | 0,01192 | 2,04× |
| 8 | 16 | 0,02102 | 48 | 0,00899 | 2,34× |
| **10** | 20 | 0,01873 | 60 | 0,00635 | **2,95×** |
| 16 | 32 | 0,01241 | 96 | 0,00247 | 5,02× |
| 24 | 48 | 0,00786 | 144 | 0,00074 | 10,64× |

A los $K = 10$ que usa FOMM, casi **tres veces** menos error.

## 4. Dónde importa: rotación

El jacobiano no agrega expresividad genérica. Agrega exactamente rotación, escala y cizalla locales. Sobre una rotación pura con 8 puntos:

| rotación | error orden 0 | error orden 1 |
|---|---|---|
| 2° | 0,000493 | 2,6 × 10⁻¹⁶ |
| 5° | 0,000918 | 5,7 × 10⁻¹⁶ |
| 10° | 0,002181 | 1,3 × 10⁻¹⁵ |
| 20° | 0,004524 | 2,4 × 10⁻¹⁵ |
| 40° | 0,008623 | 4,4 × 10⁻¹⁵ |

Con jacobiano, **cero numérico**: una rotación es afín y la representación de primer orden la expresa exactamente. Sin él, el error crece linealmente con el ángulo — y por eso el método se degrada cuando la pose del conductor se aleja mucho de la de la imagen fuente, que es la limitación declarada del paper.

## 5. El matiz que complica la lectura fácil

Comparando **a igual presupuesto de parámetros** en vez de a igual número de puntos:

| parámetros | $K$ orden 0 | error | $K$ orden 1 | error | gana |
|---|---|---|---|---|---|
| 48 | 24 | **0,00789** | 8 | 0,00914 | orden 0 |
| 72 | 36 | **0,00436** | 12 | 0,00478 | orden 0 |
| 96 | 48 | 0,00284 | 16 | **0,00250** | orden 1 |
| 144 | 72 | 0,00182 | 24 | **0,00074** | orden 1 |
| 192 | 96 | 0,00136 | 32 | **0,00031** | orden 1 |

Con presupuestos chicos, **muchos puntos simples ganan**. Si el criterio fuera error por parámetro, el jacobiano no sería obviamente la elección correcta.

{{< concept-alert type="clave" >}}
La resolución está en identificar cuál es el recurso escaso, y **no son los parámetros**.

Cada punto clave es una parte del objeto que la red debe **descubrir sin supervisión** y seguir de forma consistente entre cuadros. Nada externo dice dónde deberían estar: solo los sostienen la pérdida de reconstrucción y la de equivarianza. Diez puntos coherentes son alcanzables; noventa y seis no — se vuelven inestables, se solapan o colapsan.

**El jacobiano compra precisión sin aumentar el número de partes que hay que descubrir.** Es una decisión sobre la dificultad del aprendizaje, no sobre la capacidad de representación — y es el tipo de razonamiento que no aparece en la tabla de resultados de un paper.
{{< /concept-alert >}}

## 6. Lo que falta para que esto sea FOMM

Este código implementa la representación del movimiento y el *warping*. El sistema completo agrega tres cosas, y vale saber cuáles porque explican por qué no basta con lo anterior:

1. **Un detector de puntos clave** entrenado sin supervisión —codificador-decodificador que emite posiciones y jacobianos— con una **pérdida de equivarianza** que evita que los puntos degeneren.
2. **Una máscara de oclusión**, que decide qué se deforma desde la fuente y qué hay que **inpaintar**. Cuando una cabeza gira aparece una oreja que no está en la imagen fuente: ningún campo de deformación puede producirla.
3. **Un generador adversarial** que hace el *inpainting* y produce la textura final.

El *warping* transporta lo que existe. Todo lo demás del sistema está para el problema complementario: **generar lo que no existe**.

---

## Qué se aprendió

1. **El campo de movimiento de primer orden es idéntico en los cuatro backends** (2,8e−16) y el *warping* manual coincide con `grid_sample`.
2. **El flujo hacia atrás se eligió por diferenciabilidad**, no por conveniencia conceptual.
3. **El jacobiano vale 2,95× a $K=10$**, y es exacto sobre rotaciones.
4. **A igual presupuesto de parámetros, no siempre gana** — el recurso escaso es el número de partes que hay que descubrir sin supervisión.
5. **La deformación es solo la mitad del método**: la otra mitad es decidir qué inventar.

---

**Siguiente:** [02 - El informed guess, medido](02-informed-guess).
