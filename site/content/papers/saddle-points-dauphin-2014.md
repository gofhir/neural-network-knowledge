---
title: "Saddle Points"
weight: 85
math: true
---

{{< paper-card
    title="Identifying and attacking the saddle point problem in high-dimensional non-convex optimization"
    authors="Dauphin, Pascanu, Gulcehre, Cho, Ganguli, Bengio"
    year="2014"
    venue="NeurIPS 2014"
    pdf="/papers/2_SaddlePoints_Dauphin2014.pdf"
    arxiv="1406.2572" >}}
Demuestra que el problema principal en optimizacion de redes neuronales no son los minimos locales sino los saddle points, y propone el Saddle-Free Newton method para escaparlos eficientemente.
{{< /paper-card >}}

---

## El problema real: saddle points, no minimos locales

Durante decadas se creia que la dificultad de entrenar redes profundas era quedar atrapado en **minimos locales malos**. Este paper demuestra que en alta dimension el problema son los **saddle points** (puntos silla).

{{< concept-alert type="clave" >}}
En un punto critico con $N$ parametros, cada eigenvalor de la Hessiana puede ser positivo o negativo. La probabilidad de que **todos** sean del mismo signo es $\sim (1/2)^N$. Para redes con millones de parametros, casi todos los puntos criticos son saddle points.
{{< /concept-alert >}}

En un **minimo local** la curvatura es positiva en todas las direcciones. En un **saddle point** hay una mezcla de curvaturas positivas y negativas:

- Minimo local: $f(x) = x^2 + y^2$ -- eigenvalores $+2, +2$
- Saddle point: $f(x) = x^2 - y^2$ -- eigenvalores $+2, -2$

## Fundamento teorico

Basandose en resultados de Bray y Dean (2007) sobre campos Gaussianos aleatorios, los autores muestran que:

1. Los puntos criticos con **error alto** son casi siempre **saddle points** (indice $\alpha$ alto)
2. Los minimos locales (indice $\alpha \approx 0$) tienen error **cercano al minimo global**
3. Los saddle points estan rodeados de **plateaus** que frenan el aprendizaje

## Comportamiento de los algoritmos

**Gradient descent** cerca de un saddle point se mueve en la direccion correcta pero con paso proporcional a $|\lambda_i|$. Si $|\lambda_i|$ es pequeno, se genera un **plateau**.

**Newton method** invierte la direccion en eigenvalores negativos -- el saddle point se convierte en un **atractor**:

$$\Delta\theta = -H^{-1} \nabla f \quad \text{(converge AL saddle point)}$$

## Saddle-Free Newton (SFN)

{{< math-formula title="Saddle-Free Newton method" >}}
\Delta\theta = -|H|^{-1} \nabla f
{{< /math-formula >}}

donde $|H|$ toma el **valor absoluto** de cada eigenvalor de la Hessiana (manteniendo los eigenvectores). Esto preserva la direccion de descenso (como SGD) pero re-escala por curvatura (como Newton).

| Propiedad | SGD | Newton | SFN |
|---|---|---|---|
| Direccion cerca de saddle | Correcta (escapa) | Invertida (converge) | Correcta (escapa) |
| Velocidad en plateaus | Lenta ($|\lambda_i|$) | Rapida ($1/|\lambda_i|$) | Rapida ($1/|\lambda_i|$) |
| Saddle points | Repulsor lento | Atractor | Repulsor rapido |
| Costo por iteracion | $O(N)$ | $O(N^2\text{-}N^3)$ | $O(kN)$ con Krylov |

## Resultados experimentales

- **MLPs en MNIST**: SFN alcanza error ~0.1% vs ~1.5% de SGD con 50 hidden units
- **Deep autoencoder** (7 capas): MSE 0.57 vs 0.69 de Hessian-Free, escapando el plateau donde SGD se estanca
- **RNN en Penn Treebank**: SFN escapa el plateau donde SGD falla, confirmando que los saddle points explican parte de la dificultad de entrenar RNNs

{{< concept-alert type="clave" >}}
Este paper cambio la narrativa de la optimizacion en Deep Learning. El algoritmo SFN no se adopto ampliamente (costoso computacionalmente), pero la **comprension teorica** -- que saddle points dominan sobre minimos locales -- es hoy conocimiento estandar en la comunidad.
{{< /concept-alert >}}
