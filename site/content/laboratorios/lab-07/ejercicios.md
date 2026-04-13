---
title: "Ejercicios del Laboratorio"
weight: 40
math: true
---

## Actividades

Estos ejercicios aplican los conceptos vistos en el laboratorio sobre el modelo MiAlexNet y el dataset Flowers (102 clases). Cada actividad se construye sobre la anterior.

---

### Actividad 1: Entrenar MiAlexNet

**Contexto:** Entrenar el modelo `MiAlexNet` (sin modificaciones) por 10 epocas con el dataset Flowers y evaluar el rendimiento.

```python
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

model = MiAlexNet(num_classes=102).to(device)
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters())

# Entrenar por 10 epocas
for epoch in range(1, 11):
    model.train()
    running_loss = 0.0
    for x, target in train_loader:
        x, target = x.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoca {epoch}/10 - Loss: {running_loss/len(train_loader):.4f}")

# Evaluar
train_acc = evaluar(model, train_loader, device)
test_acc = evaluar(model, test_loader, device)
print(f"Train accuracy: {train_acc:.2%}")
print(f"Test accuracy: {test_acc:.2%}")
```

**Que observar:**
- Cual es el accuracy en train y en test?
- El modelo tiene overfitting (gran diferencia entre train y test)?
- Como evoluciona la perdida epoca a epoca?

---

### Actividad 2: Agregar Dropout

**Contexto:** Modificar `MiAlexNet` para incluir una capa de `Dropout` antes de FC6 y FC7. Entrenar por 10 epocas y comparar con la Actividad 1.

```python
class MiAlexNetDropout(nn.Module):
    def __init__(self, num_classes=102):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),               # Dropout antes de FC6
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),               # Dropout antes de FC7
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

**Que observar:**
- Mejoro el accuracy en test respecto a la Actividad 1?
- Se redujo el gap entre train y test?
- Dropout actua como regularizador: el train accuracy puede bajar, pero si el test accuracy sube, el modelo generaliza mejor.

{{< concept-alert type="clave" >}}
Dropout desactiva neuronas aleatoriamente **solo durante `model.train()`**. Durante `model.eval()`, todas las neuronas estan activas y las salidas se escalan automaticamente. Por eso es critico llamar `model.eval()` antes de evaluar.
{{< /concept-alert >}}

---

### Actividad 3: Agregar BatchNorm

**Contexto:** Agregar capas de `BatchNorm2d` antes de Conv3, Conv4 y Conv5 en el modelo. Entrenar por 10 epocas y observar el efecto en la convergencia.

```python
class MiAlexNetBN(nn.Module):
    def __init__(self, num_classes=102):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(192),              # BN antes de Conv3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),              # BN antes de Conv4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),              # BN antes de Conv5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

**Que observar:**
- La perdida converge mas rapido que sin BatchNorm?
- El entrenamiento es mas estable (menos oscilaciones en la perdida)?
- BatchNorm normaliza las activaciones intermedias, lo que combate el problema de *internal covariate shift*.

---

### Actividad 4: Fine-tuning desde ImageNet

**Contexto:** Usar el modelo AlexNet preentrenado en ImageNet (1.2M imagenes, 1000 clases) y adaptarlo para Flowers (102 clases). Entrenar por 10 epocas.

```python
import torchvision.models as models

# Cargar AlexNet preentrenado
model = models.alexnet(weights="IMAGENET1K_V1")

# Reemplazar la ultima capa del clasificador
# AlexNet original: classifier[6] = Linear(4096, 1000)
# Para Flowers:     classifier[6] = Linear(4096, 102)
model.classifier[6] = nn.Linear(4096, 102)
model = model.to(device)

# Entrenar normalmente
optimizer = Adam(model.parameters(), lr=0.001)
criterion = CrossEntropyLoss()

for epoch in range(1, 11):
    model.train()
    for x, target in train_loader:
        x, target = x.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

**Que observar:**
- El modelo preentrenado converge mas rapido que el entrenado desde cero?
- El accuracy en test es significativamente mejor?
- Cuanto gap de generalizacion hay?

{{< concept-alert type="clave" >}}
Fine-tuning aprovecha que las capas convolucionales de un modelo preentrenado en ImageNet ya aprendieron detectores de bordes, texturas y formas generales. Solo es necesario adaptar la ultima capa (o las ultimas pocas capas) al nuevo problema. Esto reduce drasticamente la cantidad de datos y tiempo necesarios para obtener buenos resultados.
{{< /concept-alert >}}

---

## Tabla Comparativa

Al completar las 4 actividades, llenar esta tabla:

| Modelo | Train Acc | Test Acc | Gap | Observaciones |
|---|---|---|---|---|
| MiAlexNet base | | | | Sin regularizacion |
| MiAlexNet + Dropout | | | | Dropout p=0.5 en FC6, FC7 |
| MiAlexNet + BatchNorm | | | | BN antes de Conv3, Conv4, Conv5 |
| AlexNet fine-tuned | | | | Preentrenado en ImageNet |
