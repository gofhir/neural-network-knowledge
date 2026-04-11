"""
Lab Experimento 1b — Entrenar CIFAR-10 con MSE (funciona MAL)

Reproduce el segundo experimento del lab:
La misma CNN pero con MSELoss y output_dim=1.
Demuestra por que MSE no sirve para clasificacion.

Ejecutar:
  docker run --rm clase8 python -u ejemplos_c8/pytorch/07_lab_exp1_mse_cifar10.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np

SEPARATOR = "=" * 60

# =============================================
# 1. Cargar CIFAR-10
# =============================================
print(SEPARATOR)
print("EXPERIMENTO 1b: CIFAR-10 con MSE (MAL)")
print(SEPARATOR)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='/tmp/data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='/tmp/data', train=False, download=True, transform=transform)

classes = ('avion', 'auto', 'pajaro', 'gato', 'ciervo',
           'perro', 'rana', 'caballo', 'barco', 'camion')

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


# =============================================
# 2. Red con output_dim=1
# =============================================
class Net(nn.Module):
    def __init__(self, output_dim=1):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_dim)  # ← 1 sola salida!

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


print("""
  El problema con MSE para clasificacion:

  Cross-Entropy (correcto):
    La red produce 10 numeros → uno por clase
    Salida: [0.1, 0.2, 0.1, 8.5, ...]  → "es clase 3"
    Las clases no tienen ORDEN ni DISTANCIA

  MSE (incorrecto):
    La red produce 1 numero → la "clase" como valor numerico
    Salida: 3.7  → redondeamos → "es clase 4"
    Problema: MSE asume que clase 4 esta "entre" 3 y 5
    y que equivocarse por 1 es mejor que equivocarse por 5
    Pero 'gato' no esta "entre" 'pajaro' y 'perro'!
""")

# =============================================
# 3. Por que MSE prefiere la clase del medio
# =============================================
print(SEPARATOR)
print("POR QUE MSE PREFIERE LA CLASE DEL MEDIO")
print(SEPARATOR)

print(f"\n  Si el modelo SIEMPRE predice la misma clase:\n")
for pred_class in [0, 3, 5, 7, 9]:
    errors_sq = [(pred_class - c) ** 2 for c in range(10)]
    mse = sum(errors_sq) / 10
    print(f"    Siempre predice {pred_class} ({classes[pred_class]:>8s}): MSE = {mse:.1f}")

print(f"""
  → Predecir siempre clase 4-5 tiene el MSE MAS BAJO!
     No porque sea la mejor prediccion,
     sino porque esta "en el medio" numericamente.

  Esto no tiene sentido: 'ciervo' (4) no es
  el promedio de 'avion' (0) y 'camion' (9).
""")

# =============================================
# 4. Entrenar con MSE
# =============================================
print(SEPARATOR)
print("ENTRENAR CON MSE")
print(SEPARATOR)

torch.manual_seed(0)
model = Net(output_dim=1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

n_epochs = 5
for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in trainloader:
        outputs = model(images).squeeze()
        # MSE necesita el label como float (no como indice)
        targets = labels.float()

        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # Predecir: redondear al entero mas cercano y clampear a [0,9]
        predicted = outputs.round().clamp(0, 9).long()
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100.0 * correct / total

    # Evaluar
    model.eval()
    correct_test = 0
    total_test = 0
    predictions_all = []
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images).squeeze()
            predicted = outputs.round().clamp(0, 9).long()
            total_test += labels.size(0)
            correct_test += predicted.eq(labels).sum().item()
            predictions_all.extend(outputs.tolist())

    test_acc = 100.0 * correct_test / total_test
    print(f"  Epoca {epoch+1}/{n_epochs}: "
          f"Loss={running_loss/len(trainloader):.4f}, "
          f"Train={train_acc:.1f}%, Test={test_acc:.1f}%")

# =============================================
# 5. Ver que predice
# =============================================
print(f"\n{SEPARATOR}")
print("QUE PREDICE EL MODELO CON MSE")
print(SEPARATOR)

predictions_tensor = torch.tensor(predictions_all)
print(f"\n  Estadisticas de las predicciones (deberian ser 0-9):")
print(f"    Media:  {predictions_tensor.mean():.2f}")
print(f"    Std:    {predictions_tensor.std():.2f}")
print(f"    Min:    {predictions_tensor.min():.2f}")
print(f"    Max:    {predictions_tensor.max():.2f}")

# Distribucion de predicciones
pred_rounded = predictions_tensor.round().clamp(0, 9).long()
print(f"\n  Distribucion de predicciones (redondeadas):")
for c in range(10):
    count = (pred_rounded == c).sum().item()
    bar = "#" * (count // 20)
    print(f"    Clase {c} ({classes[c]:>8s}): {count:5d}  {bar}")

print(f"""
  → El modelo tiende a predecir valores CENTRALES (4-5)
     porque MSE penaliza menos los errores cuando estas en el medio.

  → Accuracy final con MSE: ~{test_acc:.0f}%
     Accuracy con CrossEntropy (ejemplo anterior): ~55-60%

  MSE es MUCHO peor para clasificacion porque:
  1. Trata las clases como numeros con distancia
  2. Prefiere predecir la clase del medio
  3. No aprovecha la estructura de probabilidades
""")
