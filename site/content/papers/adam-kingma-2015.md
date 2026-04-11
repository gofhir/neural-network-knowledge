---
title: "Adam"
weight: 60
math: true
---

{{< paper-card
    title="Adam: A Method for Stochastic Optimization"
    authors="Kingma, Ba"
    year="2015"
    venue="ICLR 2015"
    arxiv="1412.6980" >}}
Optimizador adaptativo que combina las ventajas de AdaGrad y RMSProp. Se convirtio en el optimizador por defecto en la mayoria de aplicaciones de deep learning.
{{< /paper-card >}}

---

## Ideas principales

- **Adam** (Adaptive Moment Estimation) mantiene promedios moviles exponenciales del primer momento (media) y segundo momento (varianza no centrada) de los gradientes.
- Las reglas de actualizacion son:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

- **Correccion de sesgo** ($\hat{m}_t$, $\hat{v}_t$): compensa el hecho de que los promedios moviles estan inicializados en cero, especialmente importante en las primeras iteraciones.
- **Hiperparametros por defecto**: $\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.
- Computacionalmente eficiente, requiere poca memoria, e invariante al reescalamiento diagonal de los gradientes.
- Se convirtio en el optimizador mas utilizado en deep learning por su robustez y facilidad de uso.
