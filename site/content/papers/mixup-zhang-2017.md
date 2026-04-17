---
title: "Mixup"
weight: 170
math: true
---

{{< paper-card
    title="mixup: Beyond Empirical Risk Minimization"
    authors="Zhang, Cisse, Dauphin, Lopez-Paz"
    year="2017"
    venue="ICLR 2018"
    pdf="/papers/mixup-zhang-2017.pdf"
    arxiv="1710.09412" >}}
Introduce **mixup**, una tecnica de data augmentation sorprendentemente simple: entrenar con **combinaciones convexas de pares de ejemplos y sus etiquetas**. Mejora la generalizacion, reduce memorizacion de labels corruptos, aumenta la robustez a adversarial examples y estabiliza el entrenamiento de GANs. Implementable en 10 lineas de codigo.
{{< /paper-card >}}

---

## Contexto

La mayoria de los modelos deep learning se entrenan con **Empirical Risk Minimization (ERM)**: minimizar el error promedio sobre el dataset de entrenamiento. Vapnik & Chervonenkis (1971) mostraron que ERM converge cuando la capacidad del modelo (VC dimension) no crece con los datos -- pero **las redes modernas tienen VC dimension que crece linealmente con el numero de parametros** (Harvey et al. 2017), lo que contradice la suposicion.

Las consecuencias: memorizacion de labels aleatorios (Zhang et al. 2017 -- el famoso "Understanding deep learning requires rethinking generalization"), sensibilidad a adversarial examples (Szegedy 2014), falta de robustez a perturbaciones.

**Mixup** es una propuesta minimalista para abordar todos estos problemas a la vez.

---

## Ideas principales

### 1. Formulacion

Dados dos ejemplos aleatorios $(x_i, y_i)$ y $(x_j, y_j)$ del training set:

$$
\begin{aligned}
\tilde{x} &= \lambda \, x_i + (1 - \lambda) \, x_j \\
\tilde{y} &= \lambda \, y_i + (1 - \lambda) \, y_j \\
\lambda &\sim \text{Beta}(\alpha, \alpha)
\end{aligned}
$$

donde:

- $x$ son inputs raw (tipicamente imagenes)
- $y$ son **one-hot labels**
- $\lambda \in [0, 1]$ es el coeficiente de mezcla
- $\alpha$ es el unico hiperparametro (tipicamente 0.2)

Como $\alpha \to 0$: $\lambda \to 0$ o $1$ (sin mezcla, recuperamos ERM). Como $\alpha \to \infty$: $\lambda \to 0.5$ (mezcla total).

### 2. Framework teorico: Vicinal Risk Minimization

Mixup se justifica desde el **Vicinal Risk Minimization** de Chapelle et al. 2000. La idea: aproximar la distribucion desconocida $P(x, y)$ con una **distribucion de vecindad**:

$$P_\nu(\tilde{x}, \tilde{y}) = \frac{1}{n} \sum_{i=1}^{n} \nu(\tilde{x}, \tilde{y} \mid x_i, y_i)$$

Mixup define:

$$\nu(\tilde{x}, \tilde{y} \mid x_i, y_i) = \mathbb{E}_\lambda \left[ \delta(\tilde{x} = \lambda x_i + (1-\lambda) x_j, \tilde{y} = \lambda y_i + (1-\lambda) y_j) \right]$$

con $\lambda \sim \text{Beta}(\alpha, \alpha)$. Este es un prior elegante: **entre dos ejemplos, la distribucion es un segmento de recta**.

### 3. Intuicion geometrica

Sin mixup, el modelo aprende decision boundaries arbitrarias entre clases. Con mixup, el modelo es **forzado a comportarse linealmente** entre ejemplos de distintas clases. El paper muestra un toy problem (Figura 1b) donde las decision boundaries con mixup son **suaves y lineales**, mientras que ERM produce boundaries abruptas.

Consecuencias:

- **Menor confianza fuera del training set** → menos overfitting.
- **Norms de gradientes menores** → entrenamiento mas estable.
- **Resistencia a adversarial examples** → perturbaciones pequenas no saltan decision boundaries.

### 4. Implementacion: 10 lineas de PyTorch

```python
for (x1, y1), (x2, y2) in zip(loader1, loader2):
    lam = numpy.random.beta(alpha, alpha)
    x = lam * x1 + (1 - lam) * x2
    y = lam * y1 + (1 - lam) * y2  # y debe ser one-hot
    optimizer.zero_grad()
    loss = criterion(net(x), y)
    loss.backward()
    optimizer.step()
```

En la practica se usa un **trick** con una sola minibatch: mezclarla consigo misma permutada. Produce los mismos resultados con la mitad de I/O.

---

## Resultados experimentales

### ImageNet-2012

| Modelo | ERM | Mixup ($\alpha=0.2$) |
|---|---|---|
| ResNet-50 (90 ep) | 23.5 | **23.3** |
| ResNet-101 (90 ep) | 22.1 | **21.5** |
| ResNeXt-101 32×4d (100 ep) | 21.2 | **20.7** |
| ResNeXt-101 64×4d (100 ep) | 20.4 | **19.8** |
| ResNet-50 (200 ep) | 23.6 | **22.1** |

Mejora consistente de 0.2 a 1.5 puntos en top-1 error. Modelos mas grandes y entrenamientos mas largos **se benefician mas**.

### CIFAR-10 / CIFAR-100

| Dataset | Modelo | ERM | Mixup |
|---|---|---|---|
| CIFAR-10 | PreAct ResNet-18 | 5.6 | **4.2** |
| CIFAR-10 | WideResNet-28-10 | 3.8 | **2.7** |
| CIFAR-10 | DenseNet-BC-190 | 3.7 | **2.7** |
| CIFAR-100 | PreAct ResNet-18 | 25.6 | **21.1** |
| CIFAR-100 | WideResNet-28-10 | 19.4 | **17.5** |
| CIFAR-100 | DenseNet-BC-190 | 19.0 | **16.8** |

Mejoras de 1-4 puntos. Mas dramaticas que en ImageNet.

### Otros efectos

- **Memorizacion de labels corruptos**: mixup (especialmente con $\alpha \in \{4, 8, 32\}$) **supera a dropout** como defensa contra training con labels aleatorios.
- **Adversarial robustness**: mixup aumenta robustez a FGSM y iterative attacks (Seccion 3.5).
- **Speech recognition** (Google Commands): VGG-11 baja de 4.6 → **3.4** error con mixup.
- **Tabular data** (UCI datasets): mixup ayuda en 2 de 4 datasets probados.
- **GAN training**: mixup estabiliza el training de GANs (Seccion 3.7).

---

## Por que importa hoy

- **Mixup se ha vuelto estandar** en pipelines modernos de vision: ImageNet, medical imaging, autonomous driving.
- **Variantes importantes** que lo extienden:
  - **CutMix** (Yun et al. 2019): pegar un parche de una imagen sobre otra.
  - **Manifold Mixup** (Verma et al. 2019): mezclar en espacios de features, no en pixels.
  - **PuzzleMix** (Kim et al. 2020): mixup guiado por saliency.
- **Simplicidad**: 10 lineas de codigo, un solo hiperparametro, sin overhead de computo.
- **Combinable con otras tecnicas**: mixup + dropout es mejor que cualquiera solo.
- **Inspiracion teorica**: llevo a redescubrir y formalizar VRM, que influyo en Manifold Mixup, supervised contrastive learning y otros.

---

## Limitaciones

- **No siempre ayuda en tareas con salida estructurada** (seq2seq, detection). Requiere formulaciones especializadas.
- **Interpolaciones entre 3+ examples** (via Dirichlet) no aportan mejora adicional segun el paper.
- **$\alpha$ grande ($> 1$) puede empeorar** resultados si el modelo no tiene capacidad suficiente o entrenamiento largo.
- En datasets muy grandes (> 1M ejemplos), el beneficio se reduce.

---

## Notas y enlaces

- La Seccion 2 ("From ERM to mixup") contiene la derivacion teorica completa.
- La Seccion 3.8 ("Ablation Studies") es un modelo de rigor empirico -- evalua cada decision de diseno.
- Codigo oficial: [facebookresearch/mixup-cifar10](https://github.com/facebookresearch/mixup-cifar10)
- Referencia canonica para VRM: Chapelle, Weston, Bottou, Vapnik (2000) "Vicinal Risk Minimization".

Ver fundamentos: [Data Augmentation](/fundamentos/data-augmentation) · [Regularizacion](/fundamentos/regularizacion).
