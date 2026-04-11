---
title: "SGDR (Cosine Annealing)"
weight: 95
math: true
---

{{< paper-card
    title="SGDR: Stochastic Gradient Descent with Warm Restarts"
    authors="Loshchilov, Hutter"
    year="2017"
    venue="ICLR 2017"
    pdf="/papers/4_SGDR_Loshchilov2017.pdf"
    arxiv="1608.03983" >}}
Introduce cosine annealing con warm restarts periodicos del learning rate. Se convirtio en el scheduler estandar para Deep Learning, implementado como CosineAnnealingWarmRestarts en PyTorch.
{{< /paper-card >}}

---

## Problema con schedules clasicos

Los schedules de step decay requieren decidir de antemano los epochs donde reducir el learning rate, tienen mal **anytime performance** (si paras antes de tiempo, el modelo es mediocre), y una vez que el learning rate baja, el optimizador queda atrapado en el minimo local mas cercano.

## Cosine Annealing con Warm Restarts

{{< math-formula title="Formula de cosine annealing (Ecuacion 5 del paper)" >}}
\eta_t = \eta_{\min}^i + \frac{1}{2}(\eta_{\max}^i - \eta_{\min}^i)\left(1 + \cos\left(\frac{T_{\text{cur}}}{T_i}\pi\right)\right)
{{< /math-formula >}}

donde $T_{\text{cur}}$ es el numero de epochs desde el ultimo restart y $T_i = T_0 \cdot T_{\text{mult}}^{i-1}$ es la duracion del $i$-esimo ciclo.

- Cuando $T_{\text{cur}} = 0$: $\cos(0) = 1 \Rightarrow \eta_t = \eta_{\max}$ (learning rate maximo)
- Cuando $T_{\text{cur}} = T_i$: $\cos(\pi) = -1 \Rightarrow \eta_t = \eta_{\min}$ (learning rate minimo)

### Parametros clave

- **$T_0$**: duracion del primer ciclo (en epochs)
- **$T_{\text{mult}}$**: factor multiplicativo ($T_{\text{mult}} = 1$ para ciclos fijos, $T_{\text{mult}} = 2$ para duplicar cada ciclo)
- **$\eta_{\max}$**: learning rate maximo (tipicamente 0.05)
- **$\eta_{\min}$**: learning rate minimo (tipicamente 0)

## Por que funciona

{{< concept-alert type="clave" >}}
Los warm restarts permiten **escapar de minimos locales angostos** (sharp minima). Un minimo ancho (flat) retiene al modelo incluso con learning rate alto, mientras que un minimo angosto lo pierde facilmente. Los minimos anchos generalizan mejor.
{{< /concept-alert >}}

**Cosine annealing vs step decay**:
- Step decay tiene transiciones abruptas y epochs "desperdiciados" con learning rate constante
- Cosine annealing tiene transicion suave: fase inicial (lr alto) para explorar, fase final (lr bajo) para converger

## Resultados experimentales

Con Wide Residual Networks en 200 epochs totales:

| Metodo | Red | CIFAR-10 (%) | CIFAR-100 (%) |
|---|---|---|---|
| Step decay (baseline) | WRN-28-10 | 4.13 | 20.21 |
| **SGDR** $T_0$=200 | WRN-28-10 | **3.86** | 19.98 |
| **SGDR** $T_0$=10, $T_{\text{mult}}$=2 | WRN-28-10 | 4.03 | **19.58** |
| **SGDR** $T_0$=10, $T_{\text{mult}}$=2 | WRN-28-20 | **3.74** | **18.70** |

### Ensembles "gratis"

Los snapshots al final de cada ciclo (cuando $\eta_t = \eta_{\min}$) son modelos diversos que se combinan en un ensemble sin costo adicional:

| Configuracion | CIFAR-10 (%) | CIFAR-100 (%) |
|---|---|---|
| 1 run, 1 snapshot | 4.03 | 19.57 |
| 1 run, 3 snapshots | 3.51 | 17.75 |
| **16 runs, 3 snapshots c/u** | **3.14** | **16.21** |

Estos fueron **state-of-the-art** al momento de publicacion.

## Legado

{{< concept-alert type="clave" >}}
Cosine annealing es hoy **el scheduler mas usado en Deep Learning**. GPT-3, LLaMA, BERT y practicamente todo entrenamiento de modelos grandes lo usa. Complementa a AdamW (del mismo autor) como combinacion dominante.
{{< /concept-alert >}}

```python
# PyTorch - Cosine Annealing con warm restarts (SGDR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=0
)
```
