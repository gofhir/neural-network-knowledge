---
title: "Dropout"
weight: 10
math: true
---

{{< paper-card
    title="Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
    authors="Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov"
    year="2014"
    venue="JMLR"
    pdf="/papers/srivastava2014_dropout.pdf" >}}
Tecnica de regularizacion que apaga neuronas al azar durante el entrenamiento para prevenir overfitting. Uno de los papers mas citados en Deep Learning.
{{< /paper-card >}}

---

## Datos Clave

| Aspecto | Detalle |
|---|---|
| **Ano** | 2014 (enviado Nov 2013, publicado Jun 2014) |
| **Citas** | Uno de los papers mas citados en Deep Learning (>50,000 citas) |
| **Autores notables** | Geoffrey Hinton (padre del Deep Learning), Ilya Sutskever (co-fundador OpenAI), Alex Krizhevsky (creador de AlexNet) |
| **Idea central** | Apagar neuronas al azar durante entrenamiento previene overfitting |
| **Impacto** | Se convirtio en tecnica estandar en practicamente toda red neuronal profunda |

## Problema que Resuelve

Las redes neuronales profundas con muchos parametros son modelos muy expresivos, pero con datos de entrenamiento limitados tienden a **memorizar** los datos (overfitting). Muchas de las relaciones complejas que aprenden son producto del **ruido en los datos**, no de patrones reales.

Con computacion ilimitada, lo ideal seria promediar las predicciones de todos los posibles modelos ponderados por su probabilidad posterior (inferencia Bayesiana), pero esto es computacionalmente imposible para redes grandes.

**Solucion propuesta**: Dropout es una aproximacion eficiente que combina exponencialmente muchas redes "delgadas" (thinned networks) con parametros compartidos.

## Motivacion Biologica

Los autores presentan una motivacion inspirada en la **reproduccion sexual en la evolucion**:

- **Reproduccion asexual**: los genes se co-adaptan (dependen unos de otros).
- **Reproduccion sexual**: mezcla genes de dos padres al azar, rompe las co-adaptaciones, y cada gen debe ser util por si mismo.

**Analogia con Dropout**: sin Dropout, las neuronas desarrollan dependencias complejas entre ellas que funcionan en entrenamiento pero no generalizan. Con Dropout, cada neurona debe aprender a ser util independientemente.

> "Diez conspiraciones de cinco personas cada una probablemente causan mas estragos que una gran conspiracion que requiere que cincuenta personas desempeñen su papel correctamente."

## Modelo Formal

### Red con Dropout

Se agrega una **mascara Bernoulli** a cada capa:

$$r_j^{(l)} \sim \text{Bernoulli}(p)$$

$$\tilde{y}^{(l)} = r^{(l)} \odot y^{(l)}$$

$$z_i^{(l+1)} = w_i^{(l+1)} \cdot \tilde{y}^{(l)} + b_i^{(l+1)}$$

$$y_i^{(l+1)} = f(z_i^{(l+1)})$$

Donde $p$ es la probabilidad de retener cada neurona y $\odot$ denota el producto elemento a elemento.

### En test time (inferencia)

Los pesos se escalan multiplicandolos por $p$:

$$W_{\text{test}}^{(l)} = p \cdot W^{(l)}$$

Si durante entrenamiento cada neurona estaba presente solo el 60% del tiempo ($p=0.6$), en test se mantiene siempre pero se reduce su peso al 60%.

### Interpretacion como ensamble

Una red con $n$ neuronas y Dropout puede verse como $2^n$ posibles sub-redes, todas compartiendo pesos. En cada iteracion de entrenamiento se samplea una sub-red. En test, el escalado por $p$ aproxima el promedio de todas.

## Entrenamiento

### Backpropagation con Dropout

Para cada minibatch:
1. Samplear una sub-red (generar mascaras Bernoulli)
2. Forward pass solo por la sub-red
3. Backpropagation solo por la sub-red
4. Actualizar pesos

### Max-norm Regularization

Dropout funciona mejor combinado con **max-norm regularization**:

$$\|w\|_2 \leq c \quad \text{para cada neurona, con } c \in [3, 4]$$

Esto evita que los pesos exploten por el ruido de Dropout y permite usar learning rates mas altos.

## Resultados Experimentales

### MNIST

| Metodo | Error % |
|---|---|
| Red estandar (sin dropout) | 1.60 |
| Dropout + max-norm (ReLU, 1024 units) | 1.06 |
| Dropout + max-norm (ReLU, 8192 units) | **0.95** |
| DBM + dropout finetuning | **0.79** |

La red con 8192 unidades y Dropout (65M parametros en 60K datos) **no overfittea**, algo impensable sin Dropout.

### SVHN (Street View House Numbers)

| Metodo | Error % |
|---|---|
| Conv Net + max-pooling (sin dropout) | 3.95 |
| + dropout en fully connected | 3.02 |
| + dropout en TODAS las capas | 2.55 |

Agregar Dropout a las capas convolucionales (no solo las FC) da una mejora adicional significativa.

### CIFAR-10 y CIFAR-100

| Metodo | CIFAR-10 | CIFAR-100 |
|---|---|---|
| Conv Net (sin dropout) | 15.60 | 43.48 |
| + dropout en todas las capas | **12.61** | **37.20** |

### ImageNet (ILSVRC-2012)

AlexNet con Dropout gano ILSVRC-2012, un momento decisivo en Deep Learning. La diferencia con metodos clasicos fue abrumadora (~26% a 16.4% en top-5 error).

### Comparacion con otros regularizadores (MNIST)

| Metodo | Error % |
|---|---|
| L2 | 1.62 |
| Max-norm | 1.35 |
| Dropout + L2 | 1.25 |
| **Dropout + Max-norm** | **1.05** |

## Analisis de Propiedades

### Efecto en la calidad de features

- **Sin Dropout**: features ruidosas y poco interpretables; las neuronas se co-adaptan.
- **Con Dropout**: features nitidas y significativas; cada neurona detecta bordes, trazos o puntos especificos.

### Sparsity

Dropout automaticamente induce **sparsity**: la activacion media baja de ~2.0 a ~0.7 (reduccion del 65%), sin necesidad de un regularizador explicito.

### Tasa optima de Dropout

- $p$ muy bajo (0.1--0.2): demasiadas neuronas apagadas, underfitting.
- $p = 0.4$ a $0.8$: zona optima.
- $p = 1.0$: sin Dropout, error mayor.
- **Sweet spot**: $p \approx 0.5$ para capas ocultas.

## Dropout como Regularizacion

Para regresion lineal, Dropout es equivalente a una forma especial de **regularizacion L2 adaptativa**:

$$\min \|y - X\tilde{w}\|^2 + \frac{1-p}{p} \|\Gamma \tilde{w}\|^2$$

donde $\Gamma = \text{diag}(\sqrt{\text{diag}(X^T X)})$. Las features con alta varianza se regularizan mas.

## Variante: Dropout Gaussiano

En lugar de mascaras Bernoulli (0 o 1), se multiplica por $r \sim \mathcal{N}(1, \sigma^2)$ con $\sigma^2 = (1-p)/p$. Funciona ligeramente mejor en algunos casos.

## Guia Practica

| Aspecto | Recomendacion |
|---|---|
| Capas de entrada | $p = 0.8$ |
| Capas ocultas FC | $p = 0.5$ |
| Capas convolucionales | $p = 0.75$ |
| Tamano de red | Al menos $n/p$ neuronas por capa |
| Learning rate | 10--100x mayor que sin Dropout |
| Momentum | 0.95--0.99 |
| Max-norm | $c = 3$ a $4$ |

## Impacto Historico

- **2012**: AlexNet gana ImageNet con Dropout, catalizando la era del Deep Learning.
- **2014**: Este paper formaliza y analiza Dropout.
- **2016+**: Variantes surgen: DropConnect, Spatial Dropout, Variational Dropout, DropBlock, MC Dropout.
- **Hoy**: Dropout sigue usandose en practicamente toda red neuronal, incluyendo Transformers (tipicamente $p=0.1$).

**Limitacion principal**: el entrenamiento toma 2--3x mas tiempo porque los gradientes son ruidosos.
