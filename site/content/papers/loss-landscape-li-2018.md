---
title: "Loss Landscape"
weight: 90
math: true
---

{{< paper-card
    title="Visualizing the Loss Landscape of Neural Nets"
    authors="Li, Xu, Taylor, Studer, Goldstein"
    year="2018"
    venue="NeurIPS 2018"
    pdf="/papers/3_LossLandscape_Li2018.pdf"
    arxiv="1712.09913" >}}
Introduce filter normalization para visualizar loss landscapes de forma significativa y comparable entre arquitecturas. Explica visualmente por que skip connections y redes anchas son mas entrenables.
{{< /paper-card >}}

---

## El problema de la visualizacion

Los loss landscapes viven en espacios de millones de dimensiones. Para visualizarlos se proyectan a 1D o 2D usando direcciones aleatorias, pero estas visualizaciones ingenuas son **enganosas** porque no consideran la invariancia de escala de las redes.

```text
PROBLEMA: Con ReLU y/o batch normalization, multiplicar los pesos
de una capa por 10 y dividir los de la siguiente por 10 produce
una red EQUIVALENTE. Pero los landscapes lucen completamente
diferentes -- la version con pesos grandes parece "plana" y
la de pesos pequenos parece "afilada".
```

## Filter Normalization

{{< math-formula title="Normalizacion por filtro" >}}
d_{i,j} \leftarrow \frac{d_{i,j}}{\|d_{i,j}\|} \cdot \|\theta_{i,j}\|
{{< /math-formula >}}

Para cada filtro $j$ en cada capa $i$, la direccion aleatoria $d$ se reescala para tener la misma norma de Frobenius que los pesos del filtro correspondiente $\theta_{i,j}$. Esto elimina el efecto artificial de la escala.

La visualizacion completa evalua:

$$f(\alpha, \beta) = L(\theta^* + \alpha \cdot \delta + \beta \cdot \eta)$$

donde $\delta$ y $\eta$ son direcciones filter-normalized independientes.

## El debate sharp vs flat

{{< concept-alert type="clave" >}}
**Sin normalizacion**, las comparaciones de sharpness entre minimizadores son enganosas (weight decay cambia la escala de los pesos, distorsionando la geometria aparente). **Con filter normalization**, la sharpness SI correlaciona consistentemente con la generalizacion.
{{< /concept-alert >}}

## Efecto de skip connections y profundidad

Este es el hallazgo mas impactante. En CIFAR-10:

| Arquitectura | Con skip connections | Sin skip connections |
|---|---|---|
| 20 capas | Convexo, suave (7.37%) | Convexo, suave (8.18%) |
| 56 capas | Convexo, suave (5.89%) | **Caotico** (13.31%) |
| 110 capas | Convexo, suave (5.79%) | **Muy caotico** (16.44%) |

Las skip connections **previenen la transicion a comportamiento caotico**. Sin ellas, redes de 56+ capas tienen landscapes caoticos y son dificiles o imposibles de entrenar.

## Efecto del ancho de la red

Wide-ResNet-56 en CIFAR-10 con skip connections:

| Factor k | Error (%) | Landscape |
|---|---|---|
| k=1 | 5.89 | Algunas irregularidades |
| k=4 | 4.34 | Muy suave |
| k=8 | 3.93 | Extremadamente suave |

Redes mas anchas producen landscapes mas suaves y mejor generalizacion.

## Trayectorias de optimizacion

Las trayectorias de SGD viven en un subespacio de muy baja dimension. Dos vectores aleatorios en $\mathbb{R}^n$ tienen similitud coseno esperada $\approx \sqrt{2/(\pi n)}$ -- para $n = 10^6$, esto es $\sim 0.0008$. La solucion es usar **PCA sobre la trayectoria** en lugar de direcciones aleatorias.

| Factor | Efecto en landscape | Efecto en generalizacion |
|---|---|---|
| Skip connections | Previene transicion a caos | Mejora dramatica en redes profundas |
| Mas profundidad (sin skip) | Convexo a caotico | Degradacion severa |
| Mas profundidad (con skip) | Se mantiene convexo | Mejora consistente |
| Mas ancho | Mas suave | Mejora consistente |
| Batch size grande | Ligeramente mas sharp | Peor generalizacion |

{{< concept-alert type="clave" >}}
Filter normalization es hoy **estandar** para visualizar loss landscapes. El paper inspiro directamente a SAM (Sharpness-Aware Minimization) y establecio que la relacion "landscape suave = buena generalizacion" es real cuando se mide correctamente.
{{< /concept-alert >}}
