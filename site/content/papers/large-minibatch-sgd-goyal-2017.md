---
title: "Large Minibatch SGD"
weight: 110
math: true
---

{{< paper-card
    title="Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
    authors="Goyal, Dollar, Girshick, Noordhuis, Wesolowski, Kyrola, Tulloch, Jia, He"
    year="2017"
    venue="arXiv preprint"
    pdf="/papers/7_LargeMinibatchSGD_Goyal2017.pdf"
    arxiv="1706.02677" >}}
Demuestra como escalar SGD a minibatches de 8192 sin perder accuracy usando linear scaling rule y gradual warmup. Entreno ResNet-50 en ImageNet en 1 hora con 256 GPUs.
{{< /paper-card >}}

---

## El problema del minibatch grande

Al usar mas GPUs en paralelo, el minibatch total crece. Pero minibatches grandes causan **degradacion de accuracy** por dificultades de optimizacion al inicio del entrenamiento.

## Linear Scaling Rule

{{< math-formula title="Linear Scaling Rule" >}}
\text{Si el minibatch se multiplica por } k, \text{ multiplicar el learning rate por } k.
{{< /math-formula >}}

**Justificacion**: $k$ iteraciones con minibatch $n$ y LR $\eta$ son aproximadamente equivalentes a 1 iteracion con minibatch $kn$ y LR $k\eta$, bajo la hipotesis de que los gradientes no cambian mucho entre iteraciones consecutivas:

$$\nabla l(x, w_t) \approx \nabla l(x, w_{t+j}) \quad \text{para } j < k$$

Ejemplo practico:
- Baseline: minibatch 256, $\eta = 0.1$
- 256 GPUs: minibatch 8192, $\eta = 0.1 \times (8192/256) = 3.2$

## Gradual Warmup

La linear scaling rule falla al inicio (los pesos cambian rapidamente). La solucion es un **warmup lineal durante 5 epochs**: empezar con $\eta = 0.1$ y subir linealmente hasta $\eta = 3.2$.

{{< concept-alert type="clave" >}}
El gradual warmup es hoy **tecnica estandar** en practicamente todo entrenamiento de Deep Learning: BERT, GPT, ViT, y todo modelo moderno lo usa. Este paper lo establecio como practica fundamental.
{{< /concept-alert >}}

## Detalles criticos de implementacion

El paper detalla varias trampas sutiles:

- **Batch Normalization**: estadisticas locales por GPU ($n=32$ fijo), no a traves de todos los workers
- **Inicializacion gamma=0**: en la ultima BN de cada bloque residual, haciendo que al inicio cada bloque sea la identidad
- **Weight decay**: escalar solo el LR, no la loss completa (el weight decay no debe escalarse)
- **Momentum correction**: al cambiar el LR, corregir el termino historico del momentum
- **Data shuffling**: un solo shuffle global por epoch, distribuido entre workers

## Resultados principales

| Configuracion | GPUs | Minibatch | $\eta$ | Top-1 error (%) | Tiempo |
|---|---|---|---|---|---|
| Baseline | 8 | 256 | 0.1 | 23.60 $\pm$ 0.12 | 29 horas |
| **Gradual warmup** | **256** | **8192** | **3.2** | **23.74 $\pm$ 0.09** | **1 hora** |
| Sin warmup | 256 | 8192 | 3.2 | 24.84 $\pm$ 0.37 | - |
| Constant warmup | 256 | 8192 | 3.2 | 25.88 $\pm$ 0.56 | - |

Con gradual warmup, la diferencia es solo **0.14%** -- dentro del ruido estadistico.

### Limites del escalado

| Minibatch (kn) | Top-1 error (%) |
|---|---|
| 64 - 8192 | ~23.5-23.7% (estable) |
| 16,384 | 24.79% (empieza a degradar) |
| 32,768 | 27.55% |
| 65,536 | 33.96% |

{{< concept-alert type="clave" >}}
**El problema con minibatches grandes es de optimizacion, no de generalizacion.** Modelos pre-entrenados con kn=256 hasta kn=8192 dan la misma accuracy en transfer learning (Mask R-CNN en COCO), confirmando que no hay degradacion de las representaciones aprendidas.
{{< /concept-alert >}}

## Transfer learning a COCO

| kn pre-training | ImageNet error | COCO box AP | COCO mask AP |
|---|---|---|---|
| 256 | 23.60% | 35.9 | 33.9 |
| 8192 | 23.74% | 35.8 | 33.9 |
| 16384 | 24.79% | 35.1 | 33.2 |

Mientras la accuracy de ImageNet se mantenga baja (kn $\leq$ 8k), la generalizacion a deteccion y segmentacion no se degrada.

## Legado

Este paper no fue solo un resultado empirico -- fue una **guia practica** con recetas claras que democratizo el entrenamiento distribuido. La linear scaling rule y el gradual warmup son la base del entrenamiento distribuido de GPT-3, LLaMA, y todo modelo moderno a gran escala.
