---
title: "Flores, Overfitting y Fine-tuning"
weight: 35
math: true
---

## 1. Contexto

Hasta ahora hemos visualizado modelos preentrenados de ImageNet. Ahora aplicaremos las mismas tecnicas a nuestros propios modelos entrenados.

El dataset que utilizaremos es **Oxford 102 Flowers**, que contiene 102 clases de flores. Algunas clases de ejemplo: 53 = sunflower, 72 = water lily, 73 = rose.

---

## 2. MiAlexNet: Modelo Entrenado desde Cero

La arquitectura completa consta de 5 capas convolucionales y 3 capas fully connected, con 102 clases de salida:

```python
import torch.nn as nn

class MiAlexNet(nn.Module):
    def __init__(self):
        super(MiAlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 384, 3, padding=1),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2))
        self.fc1 = nn.Sequential(nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(4096, 4096), nn.ReLU(inplace=True))
        self.fc3 = nn.Linear(4096, 102)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
```

Para cargar los pesos y poner el modelo en modo evaluacion:

```python
base_alexnet = MiAlexNet()
weights = torch.load("base_alexnet.pth")
base_alexnet.load_state_dict(weights)
base_alexnet = base_alexnet.to(device).eval()
```

Resultados: **98.17% train, 80.2% test** (gap de 18 puntos porcentuales).

{{< concept-alert type="clave" >}}
Esta diferencia de rendimiento se llama **overfitting**: el modelo memorizo los datos de entrenamiento en vez de aprender patrones generalizables. Rinde excelente en datos que ya vio, pero mal en datos nuevos.
{{< /concept-alert >}}

---

## 3. Flowers Dataset

Clase personalizada para cargar el dataset:

```python
from os import listdir
from os.path import join
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, Normalize

class Flowers(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.imagenes, self.imgs_to_class = self.armar_indices(root)

    def armar_indices(self, root):
        lista_imagenes, imgs_to_class = [], []
        for clase in sorted(listdir(root)):
            directorio = join(root, clase)
            for img in sorted(listdir(directorio)):
                lista_imagenes.append(join(directorio, img))
                imgs_to_class.append(int(clase))
        return lista_imagenes, imgs_to_class

    def __getitem__(self, index):
        imagen = Image.open(self.imagenes[index]).convert('RGB')
        if self.transform:
            imagen = self.transform(imagen)
        return imagen, self.imgs_to_class[index]

    def __len__(self):
        return len(self.imagenes)
```

Las transforms se definen **sin normalizacion**, porque el modelo fue entrenado sin ella:

```python
transforms = Compose([Resize((224, 224)), ToTensor()])
ds_train = Flowers("flowers_dataset/train", transform=transforms)
ds_test = Flowers("flowers_dataset/test", transform=transforms)
```

---

## 4. Feature Visualization del Modelo Base

Las capas disponibles son: `conv1`, `conv3`, `conv5`, `labels`.

```python
get_images(base_alexnet, ['conv1', 'conv3', 'conv5', 'labels'], 3, 3, preprocess=False)
```

Es critico usar `preprocess=False` porque el modelo fue entrenado sin normalizacion de ImageNet.

### Observaciones

- **Imagenes grises**: muchos filtros producen imagenes completamente grises, lo que indica que no hubo activacion significativa. Esto sugiere filtros sobrantes; la red tiene mas capacidad de la necesaria para la tarea.
- **Imagenes de clases (labels)**: las visualizaciones de las neuronas de salida no tienen forma de flor definida. No se reconocen patrones florales claros.
- **Hipotesis**: el modelo sufre overfitting. En lugar de aprender la forma real de las flores, aprendio correlaciones espurias del dataset de entrenamiento que no generalizan a datos nuevos.

---

## 5. Fine-tuning desde ImageNet

El concepto de **fine-tuning** consiste en partir de pesos ya entrenados en otro dataset (ImageNet) en vez de pesos aleatorios.

### Estrategias de transferencia

| Estrategia | Que se entrena | Cuando usarla |
|---|---|---|
| **Transfer Learning** | Termino general para reutilizar conocimiento | — |
| **Fine-tuning** | Todas las capas (o algunas) con learning rate bajo | Datos similares al dataset original |
| **Feature Extraction** | Solo ultima capa, resto congelado | Muy pocos datos |

Para adaptar AlexNet al dataset de flores, se reemplaza el clasificador:

```python
from torchvision import models

finetuned_alexnet = models.alexnet()
finetuned_alexnet.classifier = nn.Sequential(
    nn.Dropout(),
    nn.Linear(256 * 6 * 6, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    nn.Linear(4096, 102),
)
```

La normalizacion de ImageNet es obligatoria para fine-tuning:

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transforms = Compose([Resize((224, 224)), ToTensor(), Normalize(mean, std)])
```

Resultados: **~100% train, 96% test** (gap de 4pp vs 18pp antes).

{{< concept-alert type="clave" >}}
Fine-tuning redujo el gap de overfitting de 18pp a 4pp. Las representaciones pre-aprendidas en ImageNet (bordes, texturas, formas) son transferibles y generalizan mucho mejor que aprender desde cero con pocos datos.
{{< /concept-alert >}}

---

## 6. Comparacion Visual: Base vs Fine-tuned

Comparamos las visualizaciones de 3 clases de flores: 53 (sunflower), 72 (water lily), 73 (rose).

### Modelo base

Se usa `preprocess=False` porque no uso normalizacion:

```python
_ = render.render_vis(base_alexnet, 'labels:53', show_image=True, preprocess=False)
_ = render.render_vis(base_alexnet, 'labels:72', show_image=True, preprocess=False)
_ = render.render_vis(base_alexnet, 'labels:73', show_image=True, preprocess=False)
```

### Modelo fine-tuned

Se usa `preprocess=True` (por defecto) porque usa normalizacion de ImageNet:

```python
_ = render.render_vis(finetuned_alexnet, 'labels:53', show_image=True)
_ = render.render_vis(finetuned_alexnet, 'labels:72', show_image=True)
_ = render.render_vis(finetuned_alexnet, 'labels:73', show_image=True)
```

### Resultados

- **Modelo base**: imagenes difusas, sin forma de flor reconocible. Las visualizaciones no revelan patrones semanticos claros.
- **Modelo fine-tuned**: colores y texturas claramente asociados a cada tipo de flor (amarillo para sunflower, azul/verde para water lily, rojo para rose).

**Si no se tienen suficientes datos, es recomendable entrenar a partir de un modelo preentrenado. La calidad de las representaciones internas explica directamente el rendimiento del modelo.**
