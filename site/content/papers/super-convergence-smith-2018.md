---
title: "Super-Convergence"
weight: 105
math: true
---

{{< paper-card
    title="Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates"
    authors="Leslie Smith, Nicholay Topin"
    year="2018"
    venue="Preprint"
    pdf="/papers/6_SuperConvergence_Smith2018.pdf"
    arxiv="1708.07120" >}}
Demuestra que se puede entrenar redes un orden de magnitud mas rapido usando learning rates muy grandes con la politica 1cycle. Adoptado por fastai como metodo default e implementado en PyTorch como OneCycleLR.
{{< /paper-card >}}

---

## El fenomeno de super-convergence

El entrenamiento estandar usa LR ~0.1 con step decay durante ~80K iteraciones. Super-convergence usa LR de **1.0 a 3.0** con la politica 1cycle, logrando **mejor accuracy en 8-13x menos iteraciones**.

El **LR Range Test** es el diagnostico clave: si la accuracy se mantiene alta para LR mucho mayores que 0.1, super-convergence es posible. Para ResNet-56 en CIFAR-10, la accuracy se mantiene alta hasta LR = 3.0.

## La politica 1cycle

{{< math-formula title="Politica 1cycle" >}}
\text{Fase 1 (warmup, 45\%):} \quad \eta: \eta_{\min} \to \eta_{\max} \\
\text{Fase 2 (decay, 45\%):} \quad \eta: \eta_{\max} \to \eta_{\min} \\
\text{Fase 3 (aniquilacion, 10\%):} \quad \eta: \eta_{\min}/10 \to \eta_{\min}/100
{{< /math-formula >}}

## El insight fundamental: LR grande es regularizacion

{{< concept-alert type="clave" >}}
**El learning rate grande actua como regularizador**: produce gradientes mas ruidosos, lo que ayuda a encontrar minimos anchos y planos que generalizan mejor. Por tanto, al aumentar el LR se debe **reducir** weight decay y dropout para mantener el balance total de regularizacion.
{{< /concept-alert >}}

Evidencia: entre LR = 0.2 y LR = 2.0, el training loss **sube** (la red memoriza menos) mientras que el test loss **baja** (generaliza mejor) -- definicion exacta de regularizacion.

## Resultados principales

### CIFAR-10 con ResNet-56

| Metodo | Iteraciones | Accuracy (%) | Velocidad |
|---|---|---|---|
| Piecewise constant (LR=0.35) | 80,000 | 91.2 | 1x |
| **1cycle (LR 0.1-3)** | **10,000** | **92.4** | **8x** |
| **1cycle (LR 0.1-3)** | 6,000 | 92.1 | 13x |

### Beneficio mayor con datos limitados

| Muestras | Estandar (80K iter) | 1cycle (10K iter) | Ventaja |
|---|---|---|---|
| 50,000 | 91.2% | 92.4% | +1.2% |
| 20,000 | 82.7% | 87.9% | +5.2% |
| 10,000 | 71.4% | 80.6% | **+9.2%** |

### Otras arquitecturas

| Arquitectura/Dataset | Estandar | 1cycle | Mejora |
|---|---|---|---|
| CIFAR-100 / ResNet-56 | 59.8% | **68.6%** | +8.8 pts |
| MNIST / LeNet | 99.03% (85 ep) | **99.25%** (12 ep) | 7x mas rapido |
| ImageNet / Inception-ResNet-v2 | ~67.6% (100 ep) | **74.0%** (20 ep) | 5x mas rapido |

### Adam NO logra super-convergence

Los metodos adaptativos por si solos no descubren la utilidad de LR grandes. Adam es **incompatible** con super-convergence incluso con CLR.

## Receta practica

| Hiperparametro | Estandar | Super-Convergence |
|---|---|---|
| max_lr | 0.1-0.35 | 1.0-3.0 |
| Weight decay | $10^{-4}$ | $10^{-4}$ a $10^{-6}$ |
| BN moving avg | 0.999 | 0.95 |
| Iteraciones | 80K-200K | 6K-20K |
| Batch size | 128-256 | 512-1536 |

{{< concept-alert type="clave" >}}
La politica 1cycle fue adoptada por **fastai** como `fit_one_cycle()` y por **PyTorch** como `OneCycleLR`. El concepto de warmup + cosine decay que domina el entrenamiento de Transformers modernos tiene sus raices directas en este trabajo.
{{< /concept-alert >}}
