---
title: "AlexNet"
weight: 25
math: true
---

{{< paper-card
    title="ImageNet Classification with Deep Convolutional Neural Networks"
    authors="Krizhevsky, Sutskever, Hinton"
    year="2012"
    venue="NeurIPS 2012"
    pdf="/papers/alexnet-krizhevsky-2012.pdf" >}}
El paper que **detono la era de deep learning**. AlexNet gano ILSVRC-2012 con 15.3% top-5 error (vs 26.2% del segundo lugar), demostrando por primera vez que las CNNs profundas entrenadas en GPUs podian dominar vision por computadora. Introduce ReLU, dual-GPU training, dropout y las primeras tecnicas sistematicas de **data augmentation** en deep learning.
{{< /paper-card >}}

---

## Contexto

Hasta 2012, ImageNet y tareas de reconocimiento visual estaban dominadas por pipelines con **feature engineering** manual (SIFT, HOG, Fisher Vectors) seguidos de clasificadores poco profundos (SVM). El top-5 error en ILSVRC-2011 era ~26% -- muy lejos del rendimiento humano.

Krizhevsky, Sutskever y Hinton (U. Toronto) trajeron tres ingredientes juntos por primera vez:

1. **CNN profunda** (8 capas) con ReLU.
2. **GPU dual** con kernel CUDA custom para convolucion.
3. **Dropout + data augmentation** para controlar overfitting.

El resultado rompio el benchmark por **~11 puntos absolutos** y marco el inicio de la era deep learning.

---

## Ideas principales

### 1. Arquitectura

8 capas con pesos:

- **5 convolucionales**: filtros de 11×11, 5×5, 3×3.
- **3 fully-connected**: 4096, 4096, 1000 (softmax).
- **60 millones de parametros**, 650K neuronas.

Input: imagen 224×224×3, output: softmax sobre 1000 clases.

### 2. ReLU Nonlinearity

Reemplazar $\tanh(x)$ y sigmoide por $f(x) = \max(0, x)$:

- **6x mas rapido** para alcanzar 25% training error en CIFAR-10.
- **No saturante**: gradiente no decae en region activa.
- **Sparse**: muchas unidades producen cero, regularizacion natural.

Esta decision hizo factible entrenar redes profundas en tiempo razonable. Es ahora el default universal.

### 3. Dual-GPU Training

GTX 580 de 3GB era insuficiente para el modelo completo. Solucion: **partir la red en dos GPUs**, comunicando solo en ciertas capas:

- Capas 1, 2, 4, 5: comparten informacion solo entre features de la misma GPU.
- Capas 3 y FC: comparten entre GPUs.

Esta particion redujo top-1/top-5 errors en 1.7%/1.2% vs una red de tamano equivalente en una sola GPU.

### 4. Local Response Normalization (LRN)

Normalizacion inspirada en "lateral inhibition" biologica:

$$b_{x,y}^i = \frac{a_{x,y}^i}{\left(k + \alpha \sum_{j=\max(0, i-n/2)}^{\min(N-1, i+n/2)} (a_{x,y}^j)^2\right)^\beta}$$

Con $k=2, n=5, \alpha=10^{-4}, \beta=0.75$. Mejora top-1/top-5 en 1.4%/1.2%.

**Hoy es obsoleto**: reemplazado por **Batch Normalization** (Ioffe & Szegedy 2015), que es significativamente mejor.

### 5. Overlapping Pooling

Usar stride $s=2$ y kernel $z=3$ (en vez del $s=z=2$ estandar), creando **traslapa entre pooling windows**. Mejora top-1/top-5 en 0.4%/0.3% y reduce overfitting.

### 6. Dropout en FC Layers

Recien publicado por Hinton et al. 2012, dropout en las dos primeras capas FC:

- En training: cada neurona se desactiva con probabilidad 0.5.
- En inference: multiplicar salidas por 0.5.

Efectivamente entrena un ensemble de redes diferentes, mejora generalizacion.

### 7. Data Augmentation

Dos formas:

#### 7.1 Random crops + horizontal flips

- Extraer crops aleatorios 224×224 de imagenes 256×256.
- Cada flip horizontal es un nuevo ejemplo.
- Factor de aumento: **2048x** el dataset original.

En inference: promediar predicciones sobre **10 crops** (4 corners + center + sus flips).

#### 7.2 PCA color augmentation

Alterar las intensidades de canales RGB segun las **componentes principales** del training set:

$$\begin{bmatrix} \Delta R \\ \Delta G \\ \Delta B \end{bmatrix} = [\mathbf{p}_1, \mathbf{p}_2, \mathbf{p}_3] \begin{bmatrix} \alpha_1 \lambda_1 \\ \alpha_2 \lambda_2 \\ \alpha_3 \lambda_3 \end{bmatrix}$$

donde $\mathbf{p}_i$ y $\lambda_i$ son los eigenvectors y eigenvalues de la matriz de covarianza 3×3 de pixeles RGB, y $\alpha_i \sim \mathcal{N}(0, 0.1)$.

Reduce top-1 error en ~1%. Captura la invariance de identidad respecto a cambios en la iluminacion.

---

## Resultados experimentales

### ILSVRC-2010 (con labels de test disponibles)

| Sistema | Top-1 Error | Top-5 Error |
|---|---|---|
| Sparse coding | 47.1% | 28.2% |
| SIFT + Fisher Vectors | 45.7% | 25.7% |
| **AlexNet** | **37.5%** | **17.0%** |

### ILSVRC-2012 competition

| Sistema | Top-5 Error |
|---|---|
| Segundo lugar | 26.2% |
| **AlexNet** | **15.3%** |

Una mejora absoluta de **~11 puntos** sobre sistemas clasicos. El split entre deep learning y feature engineering se volvio evidente.

### Training

- **2 GPUs GTX 580 3GB** cada una.
- **5-6 dias** de entrenamiento total.
- **90 epochs** sobre 1.2M imagenes.
- Momentum 0.9, weight decay 0.0005, initial lr 0.01.

---

## Por que importa hoy

- **Marca el inicio de la era deep learning moderno**. Todos los papers de vision post-2012 descienden conceptualmente de AlexNet.
- **ReLU** es el activador estandar en todas las arquitecturas.
- **Dropout** sigue siendo relevante, aunque en vision moderna ha sido parcialmente reemplazado por BatchNorm.
- **Data augmentation** como componente estandar del pipeline -- todos los modelos subsiguientes lo usan.
- **GPU training** se volvio el default; sin esto no existirian ResNets, Transformers, ni foundation models.
- **Inspiracion institucional**: el exito de AlexNet llevo a la adquisicion de DNNresearch por Google, la fundacion de OpenAI (Sutskever), y una década de inversión masiva en deep learning.

---

## Limitaciones y que vino despues

- **LRN se demostro sub-optimo** (reemplazado por BatchNorm, Ioffe & Szegedy 2015).
- **Dual-GPU splitting** obsoleto con GPUs de mas memoria.
- **Arquitectura shallow** por estandares modernos (8 capas vs 152 de ResNet).
- **60M parametros** hoy es pequeno (GPT-4 tiene ~1.7T).

Evolucion directa de AlexNet:

- **VGGNet** (2014): mas profundo (16-19 capas), filtros 3×3 puros.
- **GoogLeNet** (2014): Inception modules, parametros eficientes.
- **ResNet** (2015): skip connections, cientos de capas.
- **EfficientNet** (2019): scaling compound.

---

## Notas y enlaces

- La Seccion 4 ("Reducing Overfitting") es la referencia canonica para **data augmentation basica** en deep learning.
- Codigo original en CUDA: `http://code.google.com/p/cuda-convnet/` (deprecado, pero el paper contiene todos los detalles).
- El modelo **no usa padding explicito** en algunas capas -- un detalle que causo confusion en reimplementaciones posteriores.
- La primera capa aprende filtros tipo **Gabor y blobs de color** -- un hallazgo que llevo a la investigacion de transferibilidad (ver [Yosinski 2014](/papers/transferable-features-yosinski-2014)).

Ver fundamentos: [Redes Convolucionales](/fundamentos/redes-convolucionales) · [Data Augmentation](/fundamentos/data-augmentation) · [Transfer Learning](/fundamentos/transfer-learning) · [Regularizacion](/fundamentos/regularizacion).
