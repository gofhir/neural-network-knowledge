---
title: "GoogLeNet"
weight: 40
math: true
---

{{< paper-card
    title="Going Deeper with Convolutions"
    authors="Szegedy et al."
    year="2014"
    venue="CVPR 2015"
    pdf="/papers/2_GoogLeNet_Szegedy2014.pdf"
    arxiv="1409.4842" >}}
Introduce la arquitectura Inception con modulos que aplican multiples tamaños de filtro en paralelo, logrando mayor eficiencia computacional. Ganador de ILSVRC-2014.
{{< /paper-card >}}

---

## Ideas principales

- **Modulo Inception**: aplica convoluciones de $1 \times 1$, $3 \times 3$ y $5 \times 5$ en paralelo, concatenando los resultados. Esto permite capturar patrones a multiples escalas.
- **Convoluciones $1 \times 1$**: se usan como "bottleneck" para reducir la dimensionalidad antes de las convoluciones mas costosas, disminuyendo dramaticamente el costo computacional.
- **22 capas de profundidad** con solo ~5 millones de parametros (12x menos que AlexNet), gracias al diseno eficiente del modulo Inception.
- **Clasificadores auxiliares** en capas intermedias para combatir el vanishing gradient durante el entrenamiento.
- Ganador de ILSVRC-2014 con 6.67% top-5 error.
