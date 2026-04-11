---
title: "VGGNet"
weight: 30
math: true
---

{{< paper-card
    title="Very Deep Convolutional Networks for Large-Scale Image Recognition"
    authors="Simonyan, Zisserman"
    year="2014"
    venue="ICLR 2015"
    pdf="/papers/1_VGGNet_Simonyan2014.pdf"
    arxiv="1409.1556" >}}
Demuestra que la profundidad de la red es un factor critico para el rendimiento en reconocimiento de imagenes. Introduce arquitecturas de 16 y 19 capas usando exclusivamente filtros convolucionales de $3 \times 3$.
{{< /paper-card >}}

---

## Ideas principales

- **Profundidad importa**: incrementar la profundidad de la red (hasta 19 capas) mejora consistentemente el rendimiento en clasificacion de imagenes.
- **Filtros pequenos**: usar pilas de filtros $3 \times 3$ en lugar de filtros mas grandes ($5 \times 5$ o $7 \times 7$) incrementa la capacidad representacional con menos parametros.
- Dos stacks de $3 \times 3$ tienen el mismo campo receptivo que un $5 \times 5$, pero con mas no-linealidades y menos parametros.
- Las configuraciones **VGG-16** y **VGG-19** se convirtieron en backbones ampliamente utilizados para transfer learning y extraccion de features.
- Segundo lugar en ILSVRC-2014 (clasificacion) y primer lugar en localizacion.
