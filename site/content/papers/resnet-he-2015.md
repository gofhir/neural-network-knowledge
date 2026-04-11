---
title: "ResNet"
weight: 50
math: true
---

{{< paper-card
    title="Deep Residual Learning for Image Recognition"
    authors="He, Zhang, Ren, Sun"
    year="2015"
    venue="CVPR 2016"
    pdf="/papers/3_ResNet_He2015.pdf"
    arxiv="1512.03385" >}}
Introduce las conexiones residuales (skip connections) que permiten entrenar redes de cientos de capas. Ganador de ILSVRC-2015 con 3.57% top-5 error.
{{< /paper-card >}}

---

## Ideas principales

- **Problema de degradacion**: redes mas profundas (sin residuales) no solo sufren de vanishing gradient, sino que tienen mayor error de entrenamiento que redes mas superficiales.
- **Conexiones residuales**: en lugar de aprender una funcion $\mathcal{H}(x)$ directamente, cada bloque aprende el residuo $\mathcal{F}(x) = \mathcal{H}(x) - x$, y la salida es $\mathcal{F}(x) + x$.
- **Intuicion**: es mas facil aprender perturbaciones pequenas sobre la identidad que aprender la transformacion completa desde cero.
- Redes de **152 capas** (y hasta 1202 en experimentos) que se entrenan exitosamente gracias a las skip connections.
- **Ganador de ILSVRC-2015**: 3.57% top-5 error, superando el rendimiento humano estimado.
- Los bloques residuales se han convertido en un componente fundamental de practicamente toda arquitectura moderna de deep learning.
