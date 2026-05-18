---
title: "Clasificador MLP con representaciones PifPaf"
weight: 40
math: true
---

El primer lado del A/B test: entrenar un MLP de 3 capas sobre los **keypoints flatten de PifPaf** para clasificar las 4 acciones del subset (`playing_guitar`, `climbing`, `riding_a_horse`, `cutting_vegetables`).

## La arquitectura del MLP

```python
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden3 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.output(x)
        return x
```

**MLP "muy normal"** según comentario del profesor — 3 capas ocultas con ReLU + 1 capa de salida sin activación. La arquitectura más simple que califica como deep learning.

### Tres detalles fundamentales

1. **`hidden_dim=128`** — sweet spot canónico: suficientemente expresivo para combinar 51 keypoints, no tan grande que cause overfit con ~400 muestras.

2. **3 capas ocultas, no 1 o 7**: una capa ya sería universal aproximador (teorema), pero 3 da expresividad gradual sin requerir batch norm o skip connections. Más capas serían overkill.

3. **Sin activación en la capa de salida**: porque `BCEWithLogitsLoss` aplica sigmoid **internamente** combinando con cross-entropy de manera numéricamente estable. Aplicar sigmoid acá también sería **doble sigmoid → bug**.

Total de parámetros: **~45K**.

## El pivote de los datos

El experimento principal usa `data` filtrado del caché:

```python
def prepare_pifpaf_data_for_a_subset_of_labels(labels):
    data = []
    for filename, predictions in dataset.pifpaf_predictions.items():
        label = re.sub(r"_[0-9]+\.jpg", '', filename)
        if label not in labels:
            continue
        integer_label = LABEL_TO_INDEX[label]

        for prediction in predictions:
            flattened_tensor = prediction.data.reshape(-1)  # (17, 3) → (51,)
            one_hot_label = F.one_hot(torch.tensor([integer_label]),
                                      num_classes=len(DATA_LABELS))
            data.append((flattened_tensor, one_hot_label))
    return data

pifpaf_data = prepare_pifpaf_data_for_a_subset_of_labels(SUBSET_LABELS)
```

### Patrón conceptual

```
dataset.pifpaf_predictions (dict[filename → list[Annotation]])
   │
   │ filtro por SUBSET_LABELS
   │
   ▼ por cada predicción: flatten + one-hot
   │
   ▼
pifpaf_data: list[(tensor 51-D, tensor 1×40)]
```

Es **el patrón estándar de feature engineering en ML aplicado**: pivotar del formato "fuente" (organizado por necesidad de inferencia) al formato "ML" (organizado por necesidad de entrenamiento).

### Riesgo silencioso — multi-persona contribuye múltiples muestras

`predictions` es **una lista** de `Annotation` (una por persona detectada). Si una imagen tiene 3 personas, **las 3 contribuyen muestras al training set con el mismo label de la imagen**. Esto puede ser problemático:

- La persona principal está tocando guitarra → label correcto.
- La persona secundaria (audiencia, otro músico) → mismo label `playing_guitar`, pero **puede no estar tocando guitarra**.

El label se asigna **por imagen, no por persona**. Es **ruido en el training set** que el profesor acepta por simplicidad. En producción anotarías "qué persona específica hace la acción" (que de hecho hace el dataset original con sus bboxes — pero el lab los ignora).

## Conversión a tensores y split

```python
features, labels = zip(*pifpaf_data)  # truco de unzipping
features_tensor = torch.tensor(features, dtype=torch.float32)
labels_tensor = torch.stack([label.float() for label in labels])
labels_tensor = labels_tensor.squeeze(1)

X_train, X_test, y_train, y_test = train_test_split(
    features_tensor, labels_tensor, test_size=0.2)

train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
```

### Detalle no obvio del split

`train_test_split` **NO tiene `random_state` fijo**. Cada ejecución produce un split diferente, lo que añade ±2-5% de varianza al accuracy final. **No es reproducible** entre runs.

Para hacer el A/B test verdaderamente justo, deberías:

```python
X_train, X_test, y_train, y_test = train_test_split(
    features_tensor, labels_tensor,
    test_size=0.2,
    random_state=42,
    stratify=labels_tensor.argmax(dim=1)
)
```

El lab no lo hace. **Es una imperfección operacional** que el experimento principal hereda.

## La santa trinidad PyTorch

```python
model = MLP(input_size=X_train.shape[1], hidden_dim=128, output_size=y_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.BCEWithLogitsLoss()
```

Tres líneas que crean los **tres objetos centrales** de cualquier training loop PyTorch:

| Objeto | Función |
|---|---|
| `model` | La red a entrenar (~45K params) |
| `optimizer` | Adam con LR=1e-3 (default canónico) |
| `loss_function` | BCEWithLogitsLoss (multi-label binarized) |

### Por qué `BCEWithLogitsLoss` y no `CrossEntropyLoss`

El experimento es **multi-class** (cada imagen es exactamente una clase), por lo que técnicamente `CrossEntropyLoss` sería más apropiado. Pero el profesor construyó los labels como **one-hot vectors de 40 dimensiones**, lo que fuerza usar BCE (que trata cada output como un clasificador binario independiente).

**Costo**: ~5% peor accuracy que un CE bien implementado con int labels. **Aceptable** para un lab pedagógico.

## El training loop — los 5 pasos canónicos

```python
epochs = 100
training_losses = []
testing_accuracies = []

for epoch in range(epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()              # 1. limpia gradientes acumulados
        outputs = model(inputs)            # 2. forward pass
        loss = loss_function(outputs, targets)  # 3. calcula loss
        loss.backward()                    # 4. calcula gradientes
        optimizer.step()                   # 5. actualiza pesos

    training_loss = loss.item()
    training_accuracy = calculate_accuracy(model, train_loader)
    testing_accuracy = calculate_accuracy(model, test_loader)

    training_losses.append(training_loss)
    testing_accuracies.append(testing_accuracy)
```

**Los 5 statements `zero_grad → model(x) → loss → backward → step`** son el "Hola Mundo" de PyTorch. **Memorízalos** — vas a escribirlos miles de veces si haces ML aplicado.

### El bug clásico: olvidar `zero_grad`

PyTorch **acumula gradientes por default**. Si haces `loss.backward()` dos veces sin limpiar, los gradientes se suman. Sin `optimizer.zero_grad()`, los gradientes acumulan entre batches y **el modelo entrena incorrectamente** (loss puede oscilar o divergir misteriosamente).

## La función de evaluación

```python
def calculate_accuracy(model, data_loader):
    model.eval()                           # modo eval
    correct = 0
    total = 0

    with torch.no_grad():                  # sin gradientes — más rápido, menos memoria
        for inputs, targets in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # argmax sobre 40 logits
            _, labels = torch.max(targets, 1)     # argmax sobre 40 one-hots
            total += targets.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total
```

**Dos patrones críticos**:

1. **`model.eval()`**: desactiva Dropout y pone BatchNorm en modo eval. Para este MLP simple sin dropout/BN no cambia nada, pero es **buena higiene defensiva**.

2. **`with torch.no_grad()`**: desactiva el cómputo de gradientes. Memoria 2-3× menor, velocidad ~50% más rápida. **Patrón profesional crítico** en cualquier función de evaluación.

## Resultados — gráficos del experimento

Después de 100 epochs (~30-60 segundos en Colab):

### Curva de pérdida

![Pérdida de training PifPaf](/laboratorios/lab-17/mlp-pifpaf-loss.png)

Caída agresiva en las primeras 5 epochs (de ~0.5 a ~0.05), luego plateau cerca de cero. Forma esperada — el MLP converge rápidamente sobre 4 clases bien separadas.

### Curva de precisión

![Precisión de test PifPaf](/laboratorios/lab-17/mlp-pifpaf-accuracy.png)

Subida de ~25% (chance level para 4 clases) hasta plateau en **~75-85%** alrededor de la epoch 30-50. Posible decaimiento gradual después de epoch 50 (overfitting con dataset pequeño).

### Interpretación de las 3 fases

| Fase | Epochs | Qué pasa |
|---|---|---|
| Aprendizaje rápido | 1-20 | Loss cae, accuracy sube fuertemente |
| Plateau | 20-50 | Modelo se acerca al límite de lo que puede aprender con 400 muestras |
| Overfit (opcional) | 50-100 | Train accuracy sigue subiendo, test accuracy estanca o cae |

Si tu gráfica muestra **las 3 fases**, has visto el ciclo completo del ML aplicado en miniatura.

## Cross-links

{{< cards >}}
  {{< card link="../dataset-stanford40" title="Stanford 40 Dataset" subtitle="Paso previo: preparación del dataset" icon="academic-cap" >}}
  {{< card link="../clasificador-openpose" title="Clasificador MLP con OpenPose" subtitle="Siguiente: lado B del A/B test" icon="academic-cap" >}}
  {{< card link="/papers/pifpaf-kreiss-2019" title="Paper PifPaf" subtitle="El modelo que produce las features" icon="document-text" >}}
{{< /cards >}}
