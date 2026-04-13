---
title: "Funciones de Perdida"
weight: 10
math: true
---

## Contexto: el problema de Machine Learning

El flujo basico de cualquier modelo de Machine Learning sigue un ciclo: los datos pasan por el modelo, el modelo genera una prediccion, y una **funcion de perdida** evalua que tan lejos esta esa prediccion del valor real. Con esa informacion, el optimizador corrige los parametros del modelo. Este ciclo se repite miles de veces hasta que el modelo converge.

La funcion de perdida es el componente que le dice al modelo **que tan mal le fue**. Sin ella, el modelo no tiene manera de saber en que direccion ajustar sus pesos.

---

## Definicion formal

Una funcion de perdida es una metrica del error de prediccion. El objetivo del entrenamiento es encontrar el modelo que minimiza esta metrica en promedio para todos los datos posibles:

$$f^* \approx f^*_{Tr} = \underset{f \in \mathcal{H}}{\arg\min} \frac{1}{N} \sum_{x_i \in Tr} \mathcal{L}(f(x_i), y_i)$$

Donde:

- $f^*$ es el mejor modelo posible
- $\mathcal{H}$ es el espacio de hipotesis (todas las configuraciones posibles de la red)
- $\mathcal{L}$ es la funcion de perdida
- $f(x_i)$ es la prediccion del modelo para la entrada $x_i$
- $y_i$ es la etiqueta real
- $N$ es el numero de datos de entrenamiento

Como no tenemos acceso a todos los datos posibles del universo, estimamos el mejor modelo usando el set de entrenamiento.

---

## MSE (Mean Squared Error)

### Formula

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$$

MSE toma la diferencia entre la prediccion $\hat{y}_i$ y el valor real $y_i$, la eleva al cuadrado y promedia sobre todos los datos.

### Ejemplo numerico

Consideremos dos modelos que "en promedio" se equivocan por 5.0 puntos. Cual tiene mejor MSE?

| | Real ($y_i$) | Modelo ($\hat{y}_i$) | Diferencia | Diferencia$^2$ |
|---|---|---|---|---|
| **Modelo A** | 5.0 | 0.0 | 5.0 | 25 |
| | 10.0 | 10.0 | 0.0 | 0 |
| | 20.0 | 10.0 | 10.0 | 100 |
| | **Promedio:** | | 5.0 | **MSE = 41.7** |

| | Real ($y_i$) | Modelo ($\hat{y}_i$) | Diferencia | Diferencia$^2$ |
|---|---|---|---|---|
| **Modelo B** | 5.0 | 10.0 | 5.0 | 25 |
| | 10.0 | 5.0 | 5.0 | 25 |
| | 20.0 | 15.0 | 5.0 | 25 |
| | **Promedio:** | | 5.0 | **MSE = 25.0** |

{{< concept-alert type="clave" >}}
Ambos modelos tienen el mismo error promedio (5.0), pero MSE penaliza mas al Modelo A porque tiene un error de 10.0 que al cuadrado es 100. MSE **castiga desproporcionadamente los errores grandes**. Esto es una propiedad del cuadrado: duplicar el error cuadruplica la penalizacion.
{{< /concept-alert >}}

### Cuando usar MSE

- Problemas de **regresion**: predecir un valor continuo (precio, temperatura, coordenadas)
- Cuando importa **que tan lejos** esta la prediccion del valor real
- Cuando se quiere penalizar mas los errores grandes

### En PyTorch

```python
import torch.nn as nn
import torch.nn.functional as F

# Como modulo
loss_fn = nn.MSELoss()
loss = loss_fn(predictions, target)
loss.backward()

# Como funcion
loss = F.mse_loss(predictions, target)
loss.backward()
```

---

## Cross-Entropy

### Fundamento: Teoria de la Informacion

Cross-Entropy viene de la Teoria de la Informacion. Mide la **distancia entre dos distribuciones de probabilidad**:

- $p(x)$: la distribucion **real** (la respuesta correcta, codificada como one-hot)
- $q(x)$: la distribucion **estimada** (lo que la red predice)

### Formula

$$\text{Cross-Entropy} = - \sum_{i} y_i \log(\hat{y}_i)$$

Donde $y_i$ es la etiqueta real (one-hot) y $\hat{y}_i$ es la probabilidad estimada por el modelo para la clase $i$.

### Ejemplo numerico

Clasificacion de 10 clases, la clase correcta es la 3:

```text
Distribucion real (one-hot):     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
                                        ^
                                    clase 3 = 1

Distribucion estimada (red):     [0.01, 0.02, 0.89, 0.0, 0.03, 0.01, 0.01, 0.02, 0.0, 0.01]
                                               ^^^^
                                           clase 3 = 0.89
```

Como $y_i$ es one-hot (solo un 1), la formula se simplifica a:

$$\text{Cross-Entropy} = -\log(\hat{y}_{\text{clase correcta}})$$

Evaluando:
- Si la red da **0.89** a la clase correcta: $-\log(0.89) = 0.117$ (loss bajo, buena prediccion)
- Si la red da **0.01** a la clase correcta: $-\log(0.01) = 4.605$ (loss alto, mala prediccion)

{{< concept-alert type="clave" >}}
Cross-Entropy solo mira la probabilidad asignada a la clase correcta. Si el modelo asigna alta probabilidad a la respuesta correcta, el loss es bajo. Si asigna baja probabilidad, el loss crece logaritmicamente. El caso limite $\log(0) = -\infty$ se evita en PyTorch sumando un epsilon interno ($10^{-10}$).
{{< /concept-alert >}}

---

## Softmax: de logits a probabilidades

La red neuronal produce numeros "crudos" (logits) que pueden ser cualquier valor real. Cross-Entropy necesita **probabilidades** (valores entre 0 y 1 que sumen 1). Softmax hace esa conversion:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

### Ejemplo

```text
Logits (salida cruda de la red):      Softmax (probabilidades):
  [-58.0]                               [0.000000]
  [ 18.3]                               [0.999684]  <- la mas alta
  [  0.008]                             [0.000000]
  [  0.935]                             [0.000000]
  [ -0.156]                             [0.000000]
  [-88.72]                              [0.000000]
  [  0.01]                              [0.000000]
  [ 10.24]                              [0.000316]
  [  3.333]                             [0.000000]
  [  2.5]                               [0.000000]
```

El valor mas alto (18.3) concentra casi toda la probabilidad. Los valores negativos se comprimen hacia 0. La suma de todas las probabilidades es exactamente 1.0.

### Estabilidad numerica: el truco log-sum-exp

Cuando los logits son muy grandes, $e^{z_i}$ puede desbordar (overflow). Para evitarlo, se resta el maximo antes de aplicar la exponencial:

$$\text{softmax}(z_i) = \frac{e^{z_i - \max(z)}}{{\sum_j e^{z_j - \max(z)}}}$$

Esto es matematicamente equivalente pero numericamente estable. PyTorch lo hace internamente en `nn.CrossEntropyLoss`.

### Softmax + Cross-Entropy en PyTorch

`nn.CrossEntropyLoss` aplica **Softmax + Cross-Entropy** internamente. No se debe aplicar softmax antes:

```python
loss_fn = nn.CrossEntropyLoss()

# La red produce logits (numeros crudos), NO probabilidades
logits = model(x)              # [-58.0, 18.3, 0.008, ...]
loss = loss_fn(logits, label)  # PyTorch aplica softmax internamente

# Si necesitas las probabilidades para otra cosa:
probs = torch.softmax(logits, dim=1)
```

---

## Binary Cross-Entropy

Para problemas de clasificacion binaria (si/no, spam/no-spam, enfermo/sano) existe una version especializada:

$$\text{BCE} = -[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]$$

En PyTorch se usa `nn.BCEWithLogitsLoss`, que internamente aplica la funcion sigmoide antes de calcular la perdida:

```python
loss_fn = nn.BCEWithLogitsLoss()
logit = model(x)              # un solo numero crudo
loss = loss_fn(logit, label)  # label es 0.0 o 1.0
```

---

## MSE para clasificacion: por que NO funciona

La clase demuestra un problema fundamental de usar MSE para clasificacion. Si el modelo predice un solo numero (la clase como valor numerico):

```text
Clases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

Si el modelo SIEMPRE predice "0":
  MSE = (0^2 + 1^2 + 2^2 + ... + 9^2) / 10 = 28.5

Si el modelo SIEMPRE predice "5":
  MSE = (5^2 + 4^2 + 3^2 + ... + 4^2) / 10 = 11.0  <- MENOR
```

MSE prefiere predecir valores intermedios porque estan "mas cerca de todo" en terminos numericos. Pero las clases no tienen un orden numerico significativo: la clase 5 no esta "entre" la 3 y la 7 de forma semantica. Un gato (clase 3) no esta "entre" un ciervo (clase 4) y una rana (clase 6).

{{< concept-alert type="clave" >}}
MSE trata las clases como un continuo numerico con distancia significativa. Cross-Entropy trata las clases como categorias discretas sin orden. Para clasificacion, Cross-Entropy es la eleccion correcta.
{{< /concept-alert >}}

---

## Resumen: cuando usar cual

| Tipo de problema | Funcion de perdida | PyTorch | Ejemplo |
|---|---|---|---|
| **Clasificacion** (N clases) | Cross-Entropy | `nn.CrossEntropyLoss()` | digitos 0-9, tipo de animal |
| **Clasificacion binaria** | Binary Cross-Entropy | `nn.BCEWithLogitsLoss()` | spam?, enfermo? |
| **Regresion** (valor continuo) | MSE | `nn.MSELoss()` | precio, temperatura |
| **Regresion robusta** (con outliers) | L1 / MAE | `nn.L1Loss()` | tiempos de respuesta |

### Preguntas para pensar

| Problema | MSE o Cross-Entropy? | Razon |
|---|---|---|
| Estimar probabilidad de compra (0% a 100%) | MSE | Es un valor continuo |
| Sentimiento positivo/negativo/neutro | Cross-Entropy | Son categorias discretas |
| Estimar precio del dolar | MSE | Es un valor continuo |
| Cantidad de alumnos en un curso | MSE | Aunque es discreta, hay orden numerico |
| Predecir color de cada pixel (RGB) | MSE | Cada canal es un valor continuo 0-255 |
| Aproximar funcion continua no lineal | MSE | Regresion sobre funcion continua |

---

## Codigo: comparacion en CIFAR-10

El laboratorio entrena una CNN simple en CIFAR-10 con ambas funciones de perdida. La diferencia clave esta en la dimension de salida y la funcion de perdida:

```python
import torch.nn as nn
import torchvision

# Dataset CIFAR-10 (imagenes 32x32 a color, 10 clases)
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

class Net(nn.Module):
    def __init__(self, output_dim=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Con Cross-Entropy: 10 salidas (una por clase)
model_ce = Net(output_dim=10)
criterion_ce = nn.CrossEntropyLoss()

# Con MSE: 1 salida (la "clase" como numero)
model_mse = Net(output_dim=1)
criterion_mse = nn.MSELoss()
```

Cross-Entropy produce 10 probabilidades (una por clase) y elige la mayor. MSE produce un solo numero y lo redondea a la clase mas cercana. La diferencia en accuracy es significativa a favor de Cross-Entropy.
