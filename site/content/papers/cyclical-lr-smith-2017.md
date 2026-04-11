---
title: "Cyclical Learning Rates"
weight: 100
math: true
---

{{< paper-card
    title="Cyclical Learning Rates for Training Neural Networks"
    authors="Leslie Smith"
    year="2017"
    venue="WACV 2017"
    pdf="/papers/5_CyclicalLR_Smith2017.pdf"
    arxiv="1506.01186" >}}
Introduce learning rates ciclicos y el LR Range Test. Variar ciclicamente el learning rate entre un minimo y maximo alcanza misma o mejor accuracy en significativamente menos iteraciones.
{{< /paper-card >}}

---

## Intuicion

Un learning rate fijo o que solo decrece puede quedar atrapado en saddle points del landscape de la loss. Aumentar periodicamente el LR ayuda a atravesarlos, con efecto negativo a corto plazo pero beneficio a largo plazo.

## Politica Triangular

El LR sube linealmente de `base_lr` a `max_lr` y luego baja linealmente, formando un triangulo:

$$\text{cycle} = \left\lfloor 1 + \frac{\text{iter}}{2 \cdot \text{stepsize}} \right\rfloor$$

$$x = \left| \frac{\text{iter}}{\text{stepsize}} - 2 \cdot \text{cycle} + 1 \right|$$

$$\text{lr} = \text{base\_lr} + (\text{max\_lr} - \text{base\_lr}) \cdot \max(0, 1 - x)$$

**Tres politicas propuestas**:
- **triangular**: amplitud constante en cada ciclo
- **triangular2**: amplitud se reduce a la mitad cada ciclo
- **exp_range**: amplitud decae exponencialmente con $\gamma^{\text{iter}}$

## LR Range Test

{{< concept-alert type="clave" >}}
El **LR Range Test** es uno de los aportes mas practicos del paper: entrenar pocas epochs con LR creciendo linealmente y observar donde la accuracy sube (`base_lr`) y donde cae (`max_lr`). Alternativa: `max_lr` = mayor LR que converge, `base_lr` = `max_lr / 3`.
{{< /concept-alert >}}

**Stepsize recomendado**: 2 a 8 veces las iteraciones por epoch.

## Resultados experimentales

### CIFAR-10 con arquitectura Caffe

| Metodo | Iteraciones | Accuracy (%) |
|---|---|---|
| Baseline (fixed LR) | 70,000 | 81.4 |
| **CLR triangular2** | **25,000** | **81.4** |
| **CLR exp_range** | 42,000 | **82.2** |
| Decay (solo decrece) | 25,000 | 78.5 |

CLR alcanza misma accuracy en **2.8x menos iteraciones**. La politica decay (solo decrece) logra solo 78.5%, demostrando que el comportamiento **ciclico** es esencial.

### Arquitecturas modernas en CIFAR-10/100

| Arquitectura | Mejor LR fijo | CLR | Mejora |
|---|---|---|---|
| ResNet | 93.3% | **93.6%** | +0.3% |
| DenseNet | 94.5% | **94.9%** | +0.4% |
| GoogLeNet (ImageNet) | 63.0% | **64.4%** | +1.4% |

### Compatibilidad con optimizadores adaptativos

CLR funciona con Nesterov, Adam, RMSProp y AdaGrad. Para Nesterov y Adam, CLR permite alcanzar en 25K iteraciones la accuracy que toma 70K con LR fijo. Los beneficios son mayores con SGD/Nesterov que con Adam (que ya adapta el LR internamente).

## Por que solo decrementar no es suficiente

{{< math-formula title="Evidencia del comportamiento ciclico" >}}
\text{Decay (solo decrece): } 78.5\% \quad \text{vs} \quad \text{Triangular (sube y baja): } 81.4\%
{{< /math-formula >}}

El beneficio no viene de reducir el LR sino de la **combinacion de subir y bajar**, que permite escapar saddle points y explorar mejor el landscape.

{{< concept-alert type="clave" >}}
El LR Range Test se convirtio en herramienta **estandar** para cualquier proyecto de Deep Learning. CLR inspiro directamente SGDR (warm restarts), la 1cycle policy (super-convergence), y los warmup schedules de Transformers modernos. Implementado en PyTorch como `CyclicLR`.
{{< /concept-alert >}}
