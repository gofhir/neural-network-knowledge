"""
Lab Experimento 1a — Entrenar CIFAR-10 con Cross-Entropy

Reproduce el primer experimento del laboratorio:
CNN en CIFAR-10 usando CrossEntropyLoss (lo correcto para clasificacion).

Ejecutar:
  docker run --rm clase8 python -u ejemplos_c8/pytorch/06_lab_exp1_crossentropy_cifar10.py
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
print("EXPERIMENTO 1a: CIFAR-10 con Cross-Entropy")
print(SEPARATOR)

print("""
  CIFAR-10: 60,000 imagenes a color (32x32) de 10 clases.
  Mucho mas dificil que MNIST (digitos blanco/negro).
""")

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

print(f"\n  Train: {len(trainset)} imagenes")
print(f"  Test:  {len(testset)} imagenes")
print(f"  Clases: {classes}")

# Mostrar una imagen como numeros
img, label = trainset[0]
print(f"\n  Una imagen:")
print(f"    Shape: {img.shape} → (3 canales RGB, 32x32 pixeles)")
print(f"    Clase: {classes[label]} ({label})")

# =============================================
# 2. Red CNN (igual que en el lab)
# =============================================
print(f"\n{SEPARATOR}")
print("RED CNN (del laboratorio)")
print(SEPARATOR)


class Net(nn.Module):
    def __init__(self, output_dim=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)       # 3 canales → 6 filtros
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)      # 6 → 16 filtros
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_dim)   # 10 clases para CrossEntropy

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 32→28→14
        x = self.pool(F.relu(self.conv2(x)))   # 14→10→5
        x = x.view(-1, 16 * 5 * 5)            # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net(output_dim=10)  # 10 salidas (una por clase)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n  Arquitectura: Conv(3→6) → Conv(6→16) → FC(400→120→84→10)")
print(f"  Parametros: {total_params:,}")
print(f"  output_dim=10 → una salida por clase")

# =============================================
# 3. Entrenar con Cross-Entropy
# =============================================
print(f"\n{SEPARATOR}")
print("ENTRENAR CON CROSS-ENTROPY")
print(SEPARATOR)

print(f"""
  Cross-Entropy:
    - La red produce 10 numeros (logits)
    - Softmax los convierte en probabilidades
    - Loss = -log(probabilidad de la clase correcta)
""")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

n_epochs = 5
for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(trainloader):
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100.0 * correct / total

    # Evaluar en test
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = outputs.max(1)
            total_test += labels.size(0)
            correct_test += predicted.eq(labels).sum().item()

    test_acc = 100.0 * correct_test / total_test
    print(f"  Epoca {epoch+1}/{n_epochs}: "
          f"Loss={running_loss/len(trainloader):.4f}, "
          f"Train={train_acc:.1f}%, Test={test_acc:.1f}%")

# Accuracy por clase
print(f"\n  Accuracy por clase:")
class_correct = [0] * 10
class_total = [0] * 10
model.eval()
with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = outputs.max(1)
        for i in range(len(labels)):
            label = labels[i].item()
            class_total[label] += 1
            if predicted[i] == label:
                class_correct[label] += 1

for i in range(10):
    acc = 100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
    bar = "#" * int(acc / 2)
    print(f"    {classes[i]:>8s}: {acc:5.1f}%  {bar}")

print(f"\n  → Cross-Entropy funciona correctamente para clasificacion")
print(f"     Trata las 10 clases como CATEGORIAS independientes")
