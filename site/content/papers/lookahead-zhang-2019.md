---
title: "Lookahead Optimizer"
weight: 70
math: true
---

{{< paper-card
    title="Lookahead Optimizer: k steps forward, 1 step back"
    authors="Zhang, Lucas, Hinton, Ba"
    year="2019"
    venue="NeurIPS 2019"
    arxiv="1907.08610" >}}
Meta-optimizador que mejora la estabilidad y convergencia de cualquier optimizador base, manteniendo dos conjuntos de pesos: uno "rapido" que explora y uno "lento" que se actualiza periodicamente.
{{< /paper-card >}}

---

## Ideas principales

- **Dos conjuntos de pesos**: los pesos "rapidos" ($\theta$) se actualizan $k$ veces con un optimizador interno (e.g., Adam o SGD), y luego los pesos "lentos" ($\phi$) se interpolan hacia los rapidos.
- La actualizacion del peso lento es:

$$\phi_{t+1} = \phi_t + \alpha (\theta_{t+k} - \phi_t)$$

- **Reduce la varianza** de la actualizacion del optimizador interno al promediar sobre $k$ pasos.
- **Es un wrapper**: se puede aplicar sobre cualquier optimizador base sin modificar su implementacion.
- Hiperparametros tipicos: $k = 5$ pasos internos, $\alpha = 0.5$ para la interpolacion.
- Mejora la convergencia en multiples tareas (clasificacion de imagenes, modelado de lenguaje, traduccion) con costo computacional minimo.
