---
title: "Regularizacion"
weight: 20
math: true
---

## El problema: overfitting

Cuando un modelo tiene muchos parametros y los datos de entrenamiento son pocos, el modelo **memoriza** los datos en vez de aprender patrones generales. A esto se le llama **overfitting**: el modelo se desempeña excelente en entrenamiento pero falla con datos nuevos.

```text
Sin regularizacion (red con 1000 neuronas, pocos datos):
  Train accuracy: 100%    <- memorizó todo
  Test accuracy:  60%     <- no generaliza

Con regularizacion:
  Train accuracy: 95%     <- sacrifica un poco en train
  Test accuracy:  85%     <- pero generaliza mucho mejor
```

La manifestacion visual del overfitting se observa en los limites de clasificacion: sin regularizacion, los limites son extremadamente complejos y se ajustan a cada punto individual. Con regularizacion, los limites son mas suaves y capturan la tendencia general de los datos.

---

## Tradeoff bias-varianza

El overfitting se entiende mejor a traves del tradeoff bias-varianza:

- **Bias alto** (underfitting): el modelo es demasiado simple para capturar los patrones. Falla tanto en train como en test.
- **Varianza alta** (overfitting): el modelo es tan complejo que se adapta al ruido de los datos de entrenamiento. Funciona perfecto en train pero falla en test.

La regularizacion introduce un **sesgo controlado** (bias) al modelo para reducir su varianza. Es preferible un modelo ligeramente sesgado que generalice bien, a un modelo sin sesgo que memorice.

---

## Regularizacion L2 (Weight Decay)

### Idea central

Penalizar los pesos grandes. Si un peso es muy grande, significa que la red depende excesivamente de un feature especifico, lo que puede ser indicio de overfitting.

### Formula

$$L_{\text{total}} = L_{\text{original}} + \lambda \sum_{i} w_i^2$$

Donde:

- $L_{\text{original}}$ es la funcion de perdida base (Cross-Entropy, MSE, etc.)
- $\lambda$ es el peso de la regularizacion (que tanta importancia le damos)
- $w_i$ son todos los pesos de la red

### Ejemplo numerico

```text
Sin regularizacion:
  Loss = CrossEntropy(prediccion, real) = 0.5
  -> El optimizador solo se preocupa de predecir bien

Con regularizacion L2 (lambda = 0.2):
  Loss = CrossEntropy + 0.2 * (w1^2 + w2^2 + w3^2 + ...)
  Loss = 0.5 + 0.2 * (10^2 + 5^2 + 3^2 + ...)
  Loss = 0.5 + 0.2 * 134
  Loss = 0.5 + 26.8 = 27.3
  -> El optimizador tiene que predecir bien Y mantener los pesos chicos
```

### Efecto sobre los pesos

```text
Sin L2:  pesos = [50.0, -30.0, 0.01, 0.01, 0.0, 0.0]
  -> Pocas neuronas hacen todo el trabajo (overfitting)

Con L2:  pesos = [3.2, -2.1, 1.5, -1.8, 0.9, -0.7]
  -> Pesos mas distribuidos, limites de clasificacion mas suaves
```

Los pesos grandes son "caros" porque aumentan el loss. La red prefiere distribuir la importancia entre muchos pesos pequenos en vez de depender de pocos pesos grandes.

### En PyTorch

L2 se implementa como `weight_decay` en el optimizador. Es un solo parametro:

```python
import torch.optim as optim

# Sin regularizacion
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)

# Con regularizacion L2 (lambda = 0.2)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.2)
```

PyTorch hace internamente: `loss_total = loss + weight_decay * sum(param ** 2)`

### Valores tipicos de weight_decay

| Valor | Efecto |
|---|---|
| 0.0 | Sin regularizacion |
| 0.0001 | Regularizacion sutil (comun en practica) |
| 0.001 | Regularizacion moderada |
| 0.01 | Regularizacion fuerte |
| 0.2 | Regularizacion muy fuerte (valor del lab) |
| 100000 | Regularizacion extrema: la red no aprende nada (pesos en ~0) |

{{< concept-alert type="clave" >}}
Un $\lambda$ demasiado alto destruye la capacidad del modelo. Con $\lambda = 100000$, la penalizacion de los pesos domina completamente sobre la tarea de prediccion, y los pesos convergen a cero. El modelo resultante no clasifica nada porque todos sus parametros son practicamente nulos.
{{< /concept-alert >}}

---

## Regularizacion L1

### Formula

$$L_{\text{total}} = L_{\text{original}} + \lambda \sum_{i} |w_i|$$

Similar a L2, pero usa el **valor absoluto** en vez del cuadrado.

### Diferencia clave con L2

```text
L2 (cuadrado):
  Penaliza MUCHO los pesos grandes, poco los chicos
  Efecto: pesos se hacen chicos pero NO llegan a cero
  pesos: [0.01, 0.02, 0.03, 0.01, 0.02, 0.01]

L1 (absoluto):
  Penaliza IGUAL pesos grandes y chicos
  Efecto: muchos pesos se hacen EXACTAMENTE cero
  pesos: [0.0, 0.5, 0.0, 0.0, 0.3, 0.0]   <- sparse
```

### Por que L1 produce sparsity

El gradiente de $|w|$ es constante ($+1$ o $-1$), independiente del valor de $w$. Esto significa que L1 empuja a todos los pesos hacia cero con la misma fuerza, sin importar si son grandes o chicos. Los pesos pequenos llegan a cero y se quedan ahi.

En cambio, el gradiente de $w^2$ es $2w$, que se hace mas debil cuando $w$ es pequeno. L2 frena antes de llegar a cero.

L1 produce **sparsity**: es como si la red seleccionara automaticamente que features son importantes (pesos distintos de cero) y descartara el resto (pesos en cero).

### En PyTorch

L1 no viene como parametro del optimizador. Se implementa manualmente:

```python
l1_lambda = 0.001
l1_norm = sum(p.abs().sum() for p in model.parameters())
loss = loss_fn(predictions, labels) + l1_lambda * l1_norm
loss.backward()
```

---

## Comparacion: L1 vs L2

| Propiedad | L2 (Weight Decay) | L1 |
|---|---|---|
| **Formula** | $\lambda \sum w_i^2$ | $\lambda \sum \|w_i\|$ |
| **Gradiente** | $2\lambda w_i$ (proporcional al peso) | $\lambda \cdot \text{sign}(w_i)$ (constante) |
| **Efecto** | Pesos chicos, distribuidos | Pesos en cero (sparse) |
| **Util cuando** | Se quiere suavizar el modelo | Se sospecha que pocos features importan |
| **En PyTorch** | `weight_decay=0.2` en optimizer | Manual: `sum(p.abs().sum())` |

---

## Dropout

En la Clase 7 se introdujo Dropout como tecnica de regularizacion. Funciona **apagando neuronas al azar** durante el entrenamiento con probabilidad $p$.

### Por que funciona: interpretacion como ensemble

Cada forward pass con Dropout activo entrena una "sub-red" diferente (porque distintas neuronas estan activas). El modelo final es un promedio implicito de todas esas sub-redes. Esta interpretacion como **ensemble** explica por que Dropout reduce el overfitting: combinar multiples modelos reduce la varianza.

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(p=0.5),       # 50% de las neuronas se apagan
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(p=0.3),       # 30% de las neuronas se apagan
    nn.Linear(128, 10),
)
```

{{< concept-alert type="clave" >}}
Dropout solo se aplica durante el entrenamiento. En modo evaluacion (`model.eval()`), todas las neuronas estan activas pero sus salidas se escalan por $(1-p)$ para compensar. PyTorch maneja esto automaticamente.
{{< /concept-alert >}}

---

## Early stopping

Otra tecnica de regularizacion implicita: detener el entrenamiento cuando el **loss de validacion** deja de mejorar, aunque el loss de entrenamiento siga bajando. El punto donde el loss de validacion empieza a subir marca el inicio del overfitting.

```text
Epoca   Train Loss   Val Loss
  1       2.50         2.48
  5       1.20         1.25
 10       0.50         0.65
 15       0.20         0.80    <- val loss sube, overfitting
 20       0.05         1.10    <- cada vez peor en validacion

Early stopping detendrÍa en la epoca 5-10.
```

---

## Data augmentation como regularizacion implicita

Aplicar transformaciones aleatorias a las imagenes de entrenamiento (rotaciones, flips, recortes, cambios de brillo) aumenta artificialmente el tamano del dataset. El modelo nunca ve la misma imagen dos veces exactamente igual, lo que dificulta la memorizacion y fuerza la generalizacion.

```python
from torchvision import transforms

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
])
```

---

## Combinando tecnicas

En la practica es comun combinar varias tecnicas de regularizacion:

```python
# Red con Dropout + L2 (weight_decay)
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.3),           # Dropout
    nn.Linear(256, 10),
)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01,         # L2
)
```

Cada tecnica actua sobre un aspecto diferente: L2 suaviza los pesos, Dropout crea redundancia, Early stopping limita la exposicion a los datos, y Data augmentation aumenta la diversidad de los datos. Juntas proporcionan una defensa robusta contra el overfitting.

---

## Codigo del laboratorio: efecto visual de L2

El laboratorio usa un dataset 2D simple con una red grande (1000 neuronas por capa) para visualizar el efecto de L2:

```python
class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 1000)     # 2 entradas, 1000 neuronas
        self.layer2 = nn.Linear(1000, 1000)  # 1000 -> 1000
        self.layer3 = nn.Linear(1000, 1)     # 1000 -> 1 salida

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# Sin regularizacion: limites complejos (overfitting)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)

# Con regularizacion L2: limites suaves (mejor generalizacion)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.2)
```

La unica diferencia entre ambos entrenamientos es el parametro `weight_decay`. Sin regularizacion, la red con 1000 neuronas por capa genera limites de clasificacion extremadamente retorcidos que pasan exactamente por cada punto de entrenamiento. Con `weight_decay=0.2`, los mismos limites se suavizan y capturan la tendencia general de los datos.
