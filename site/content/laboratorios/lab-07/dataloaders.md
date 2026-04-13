---
title: "DataLoaders y Manejo de Datos"
weight: 35
math: true
---

## 1. El Problema

Los datasets de deep learning son demasiado grandes para caber en memoria. Un dataset de imagenes puede ocupar decenas de gigabytes. Ademas, el entrenamiento con SGD requiere procesar los datos en **mini-batches** --- subconjuntos aleatorios del dataset.

PyTorch resuelve este problema con dos abstracciones:

- **`Dataset`** --- abstrae el acceso a los datos individuales
- **`DataLoader`** --- maneja la creacion de batches, shuffling y carga en paralelo

```
Dataset (accede a muestras individuales)
    ↓
DataLoader (agrupa en batches, mezcla, paraleliza)
    ↓
Training loop (consume batches)
```

---

## 2. La Interfaz Dataset

Todo dataset custom debe heredar de `torch.utils.data.Dataset` e implementar dos metodos:

- **`__len__(self)`** --- retorna el numero total de muestras
- **`__getitem__(self, idx)`** --- retorna la muestra en la posicion `idx` (tipicamente una tupla `(dato, etiqueta)`)

```python
from torch.utils.data import Dataset
from PIL import Image
import os

class FlowersDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Recorrer subdirectorios (cada uno es una clase)
        for class_idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
```

{{< concept-alert type="clave" >}}
`__getitem__` se llama una vez por cada muestra. La carga de la imagen se hace **bajo demanda** (lazy loading), no al crear el dataset. Esto permite trabajar con datasets que no caben en memoria.
{{< /concept-alert >}}

---

## 3. DataLoader

`DataLoader` toma un `Dataset` y produce iteradores de batches:

```python
from torch.utils.data import DataLoader

dataset = FlowersDataset(root_dir='flowers/train', transform=transform)

train_loader = DataLoader(
    dataset,
    batch_size=32,       # Muestras por batch
    shuffle=True,        # Mezclar al inicio de cada epoca
    num_workers=4,       # Procesos paralelos para cargar datos
    drop_last=True       # Descartar ultimo batch si es incompleto
)
```

### Parametros importantes

| Parametro | Descripcion | Valor tipico |
|---|---|---|
| `batch_size` | Muestras por batch | 32, 64, 128 |
| `shuffle` | Mezclar datos cada epoca | `True` para train, `False` para test |
| `num_workers` | Procesos paralelos de carga | 2-8 (depende del hardware) |
| `drop_last` | Descartar ultimo batch incompleto | `True` para train (evita batches irregulares) |
| `pin_memory` | Fijar memoria para transferencia rapida a GPU | `True` si se usa GPU |

### Iteracion

```python
# Una epoca = una pasada completa por todo el dataset
for batch_idx, (images, labels) in enumerate(train_loader):
    images = images.to(device)  # [batch_size, 3, 224, 224]
    labels = labels.to(device)  # [batch_size]

    output = model(images)
    loss = criterion(output, labels)
    # ...
```

Si el dataset tiene 1000 muestras y `batch_size=32`, cada epoca tendra $\lceil 1000/32 \rceil = 32$ iteraciones (31 batches de 32 y 1 batch de 8, o 31 batches si `drop_last=True`).

---

## 4. Transformaciones con torchvision.transforms

Las transformaciones se aplican a cada muestra antes de que llegue al modelo. Se componen con `Compose`:

### Transformaciones basicas

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),    # Redimensionar a 224x224
    transforms.ToTensor(),            # PIL Image -> Tensor [0, 1]
    transforms.Normalize(             # Normalizar por canal
        mean=[0.485, 0.456, 0.406],   # Media de ImageNet
        std=[0.229, 0.224, 0.225]     # Desviacion estandar de ImageNet
    )
])
```

{{< concept-alert type="recordar" >}}
`ToTensor()` convierte una imagen PIL (H x W x C, valores 0-255) a un tensor PyTorch (C x H x W, valores 0.0-1.0). Esta conversion de formato y rango es obligatoria antes de alimentar el modelo.
{{< /concept-alert >}}

### Transformaciones de aumento (data augmentation)

El aumento de datos genera variaciones de las imagenes de entrenamiento para mejorar la generalizacion. **Solo se aplican durante entrenamiento**, no durante evaluacion.

```python
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),        # Recorte aleatorio
    transforms.RandomHorizontalFlip(),        # Espejado horizontal (50%)
    transforms.RandomRotation(15),            # Rotacion aleatoria +-15 grados
    transforms.ColorJitter(                   # Variacion de color
        brightness=0.2, contrast=0.2,
        saturation=0.2, hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),            # Sin aumento para test
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

---

## 5. Datasets Predefinidos de torchvision

PyTorch provee datasets estandar listos para usar:

```python
from torchvision import datasets

# CIFAR-10: 60,000 imagenes 32x32, 10 clases
train_cifar = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# MNIST: 70,000 imagenes 28x28 de digitos, 10 clases
train_mnist = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
)

# ImageNet: 1.2M imagenes, 1000 clases (requiere descarga manual)
# train_imagenet = datasets.ImageNet(root='./imagenet', split='train')
```

Estos datasets ya implementan `__len__` y `__getitem__`, por lo que se pueden usar directamente con `DataLoader`.

---

## 6. Ejemplo Completo: Flowers Dataset

Este es el flujo completo del laboratorio, desde la definicion del dataset hasta la creacion de los DataLoaders:

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# 1. Definir transformaciones
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 2. Crear datasets
train_dataset = FlowersDataset(
    root_dir='flowers_dataset/train',
    transform=transform_train
)
test_dataset = FlowersDataset(
    root_dir='flowers_dataset/test',
    transform=transform_test
)

print(f"Train: {len(train_dataset)} muestras")
print(f"Test: {len(test_dataset)} muestras")

# 3. Crear DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 4. Verificar un batch
images, labels = next(iter(train_loader))
print(f"Batch de imagenes: {images.shape}")   # [64, 3, 224, 224]
print(f"Batch de etiquetas: {labels.shape}")  # [64]
```

---

## 7. Concepto de Epoca

Una **epoca** es una pasada completa por todo el dataset de entrenamiento. Cuando iteramos sobre un `DataLoader`, cada iteracion completa del `for` constituye una epoca.

```python
n_epochs = 20

for epoch in range(1, n_epochs + 1):
    for batch in train_loader:   # Una iteracion completa = 1 epoca
        # ... entrenar ...
        pass
    print(f"Epoca {epoch} completada")
```

La cantidad de epocas es un hiperparametro. Muy pocas epocas producen underfitting (el modelo no aprendio lo suficiente). Demasiadas epocas producen overfitting (el modelo memorizo los datos de entrenamiento). Monitorear la perdida en el conjunto de validacion ayuda a determinar cuando detener el entrenamiento.
