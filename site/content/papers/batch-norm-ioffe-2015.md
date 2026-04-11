---
title: "Batch Normalization"
weight: 20
math: true
---

{{< paper-card
    title="Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
    authors="Ioffe, Szegedy"
    year="2015"
    venue="ICML"
    pdf="/papers/ioffe2015_batch_normalization.pdf" >}}
Normaliza las activaciones de cada capa usando estadisticas del mini-batch para estabilizar el entrenamiento. Uno de los componentes mas ubicuos en arquitecturas modernas.
{{< /paper-card >}}

---

## Datos Clave

| Aspecto | Detalle |
|---|---|
| **Ano** | 2015 (enviado Feb 2015, publicado en ICML 2015) |
| **Citas** | >50,000 citas |
| **Autores notables** | Christian Szegedy (creador de GoogLeNet/Inception), Sergey Ioffe (Google Research) |
| **Idea central** | Normalizar las entradas de cada capa usando estadisticas del mini-batch |
| **Impacto** | Componente estandar de practicamente toda arquitectura moderna de CNNs |

## Problema: Internal Covariate Shift

La distribucion de las entradas de cada capa cambia durante el entrenamiento, a medida que los parametros de las capas anteriores se actualizan.

> **Internal Covariate Shift**: El cambio en la distribucion de las activaciones internas de la red durante el entrenamiento, causado por la actualizacion de los parametros de las capas precedentes.

### Consecuencias

1. **Learning rates bajos obligatorios**: con learning rates altos, los cambios en las distribuciones se amplifican capa a capa.
2. **Inicializacion cuidadosa requerida**: pesos mal calibrados pueden saturar las activaciones desde el inicio.
3. **Saturacion de sigmoides**: cuando $|x|$ es grande, $g'(x) \approx 0$ y los gradientes se desvanecen.
4. **Entrenamiento lento**: las capas gastan esfuerzo adaptandose a distribuciones cambiantes en vez de aprender features utiles.

### Por que no simplemente whitening?

Si la normalizacion se computa fuera del grafo computacional, el optimizador no la tiene en cuenta y los parametros pueden crecer sin limite. La solucion es hacer que la normalizacion sea **parte del modelo** y sea **diferenciable**.

## El Metodo: Batch Normalization

### Simplificaciones clave

1. **Normalizar cada dimension independientemente** (sin decorrelacionar).
2. **Usar estadisticas del mini-batch** (permite backpropagation).

### Parametros aprendibles $\gamma$ y $\beta$

Solo normalizar podria limitar la capacidad representacional. Se agregan parametros aprendibles:

$$y = \gamma \cdot \hat{x} + \beta$$

Si la red necesita recuperar la distribucion original, puede aprender $\gamma = \sqrt{\text{Var}[x]}$ y $\beta = E[x]$, deshaciendo la normalizacion.

### Algoritmo: Batch Normalizing Transform

Dados valores de $x$ sobre un mini-batch $\mathcal{B} = \{x_1 \ldots x_m\}$:

$$\mu_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^{m} x_i$$

$$\sigma^2_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_{\mathcal{B}})^2$$

$$\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma^2_{\mathcal{B}} + \epsilon}}$$

$$y_i = \gamma \cdot \hat{x}_i + \beta$$

### Backpropagation a traves de BN

Los gradientes respecto a $x_i$ dependen de **todos** los elementos del mini-batch (a traves de $\mu_{\mathcal{B}}$ y $\sigma^2_{\mathcal{B}}$), introduciendo una dependencia entre ejemplos dentro del batch.

$$\frac{\partial \ell}{\partial \gamma} = \sum_i \frac{\partial \ell}{\partial y_i} \cdot \hat{x}_i \qquad \frac{\partial \ell}{\partial \beta} = \sum_i \frac{\partial \ell}{\partial y_i}$$

### Donde se inserta BN

$$z = g(\text{BN}(Wu))$$

- El bias $b$ se **elimina** porque BN ya incluye $\beta$.
- BN se aplica **antes** de la no-linealidad.
- Para capas convolucionales, se usa un par $(\gamma, \beta)$ por feature map.

## Entrenamiento e Inferencia

| | Entrenamiento | Inferencia |
|---|---|---|
| **Estadisticas** | $\mu_{\mathcal{B}}, \sigma^2_{\mathcal{B}}$ del mini-batch | Running mean/var (promedio movil) |
| **Dependencia** | Salida depende del batch completo | Determinista e independiente |
| **BN es** | No-lineal | Lineal (transformacion afin fija) |
| **PyTorch** | `model.train()` | `model.eval()` |

**Error comun**: olvidar llamar `model.eval()` antes de inferencia, lo que produce resultados erraticos.

## Propiedades Teoricas

### Permite learning rates mas altos

$$\text{BN}(Wu) = \text{BN}((aW)u) \quad \forall a$$

La escala de los pesos no afecta la activacion normalizada. Ademas, pesos grandes producen gradientes mas pequenos, creando una **auto-regulacion** que estabiliza el entrenamiento.

### Jacobianos con valores singulares cercanos a 1

BN hace que $J \cdot J^T \approx I$, lo que significa que los gradientes no se amplifican ni se atenuan durante backpropagation.

### BN como regularizador

Cada ejemplo se normaliza usando estadisticas que dependen de los otros ejemplos del batch, introduciendo ruido similar a Dropout. En algunos casos, BN puede **reemplazar** Dropout.

## Resultados en ImageNet

| Modelo | Steps para 72.2% | Max accuracy |
|---|---|---|
| **Inception** (baseline) | 31.0 $\times 10^6$ | 72.2% |
| **BN-Baseline** | 13.3 $\times 10^6$ | 72.7% |
| **BN-x5** (lr $\times$ 5) | 2.1 $\times 10^6$ | 73.0% |
| **BN-x30** (lr $\times$ 30) | 2.7 $\times 10^6$ | 74.8% |

- Solo agregar BN alcanza 72.2% en **menos de la mitad** de los pasos.
- BN + learning rate $\times 5$ requiere **14 veces menos pasos**.
- BN permite entrenar con **sigmoide** (imposible sin BN en redes profundas).
- Ensemble de BN-Inception: **4.82% top-5 error**, superando la precision humana estimada.

## Limitaciones

### Dependencia del batch size

| Batch size | Comportamiento |
|---|---|
| Grande ($\geq 32$) | Buena estimacion de $\mu$ y $\sigma^2$, entrenamiento estable |
| Pequeno ($\leq 4$) | Estimaciones ruidosas, entrenamiento inestable |
| 1 | $\sigma^2 = 0$, division por cero. BN no funciona |

Esto motivo alternativas:
- **Layer Normalization** (Ba et al., 2016): normaliza sobre features. Usado en Transformers.
- **Instance Normalization** (Ulyanov et al., 2016): para style transfer.
- **Group Normalization** (Wu & He, 2018): compromiso entre BN y LN.

## Impacto Historico

- **2015-2016**: Adopcion masiva. ResNet, VGG con BN, Inception v2/v3 lo incorporan como estandar.
- **2017+**: Debate teorico. Santurkar et al. (2018) argumentan que el beneficio no es por reducir ICS, sino por **suavizar el landscape de optimizacion**.
- **Hoy**: BN sigue siendo estandar en CNNs. Transformers usan Layer Normalization.

## Sinergia con otras tecnicas

En la practica moderna:

$$\text{ReLU} + \text{BN} + \text{(Dropout opcional)} = \text{receta estandar}$$

- **ReLU**: resuelve vanishing gradient por saturacion.
- **Dropout**: resuelve overfitting por co-adaptacion.
- **BN**: resuelve Internal Covariate Shift, permite learning rates altos, y actua como regularizador.
