# Laboratorio Clase 9 — Visualización e Interpretabilidad en CNNs
**Diplomado Inteligencia Artificial UC | Profesor: Miguel Fadic**

---

## Tabla de Contenidos

1. [Objetivos del Laboratorio](#1-objetivos-del-laboratorio)
2. [Herramientas y Librerías](#2-herramientas-y-librer%C3%ADas)
3. [Parte 1: Feature Visualization con torch-lucent](#3-parte-1-feature-visualization-con-torch-lucent)
4. [Parte 2: Arquitecturas Disponibles](#4-parte-2-arquitecturas-disponibles)
5. [Parte 3: Visualización por Capas](#5-parte-3-visualizaci%C3%B3n-por-capas)
6. [Parte 4: Dataset de Flores y Modelos Propios](#6-parte-4-dataset-de-flores-y-modelos-propios)
7. [Parte 5: Atribución con Perturbación Extremal](#7-parte-5-atribuci%C3%B3n-con-perturbaci%C3%B3n-extremal)
8. [Parte 6: Comparación Base vs Fine-tuned](#8-parte-6-comparaci%C3%B3n-base-vs-fine-tuned)
9. [Ejercicios](#9-ejercicios)
10. [Conceptos Técnicos Clave](#10-conceptos-t%C3%A9cnicos-clave)

---

## 1. Objetivos del Laboratorio

Al terminar este laboratorio deberás ser capaz de:

1. **Aplicar Feature Visualization** sobre distintas arquitecturas de CNN para entender qué aprende cada capa.
2. **Navegar la estructura de capas** de AlexNet, VGG19, GoogLeNet y ResNet50 usando herramientas de PyTorch.
3. **Comparar representaciones** entre modelos entrenados desde cero vs. modelos fine-tuned con Transfer Learning.
4. **Usar Extremal Perturbation** para identificar qué región de una imagen impulsa la predicción de un modelo.
5. **Analizar el impacto del fine-tuning** en la calidad de las representaciones aprendidas.

---

## 2. Herramientas y Librerías

### Setup inicial

```python
import torch
import torchvision
from torchvision import models, transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
```

### Librerías especializadas

#### torch-lucent

Port en PyTorch de la librería **Lucid** (originalmente en TensorFlow, de Distill/Google Brain). Implementa algoritmos de Feature Visualization con regularización.

```python
# Instalación
!pip install torch-lucent

# Imports principales
from lucent.modelzoo import *
from lucent.optvis import render, param, transform, objectives
from lucent.modelzoo.util import get_model_layers
```

#### torchray

Librería de Facebook Research para métodos de atribución.

```python
# Instalación
!pip install torchray

from torchray.attribution.extremal_perturbation import extremal_perturbation, contrastive_reward
from torchray.utils import get_example_data
from torchray.visualization.plots import plot_attribution
```

### Dispositivo de cómputo

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando: {device}")
```

> **Nota:** El laboratorio es intensivo computacionalmente. Se recomienda fuertemente usar GPU (Google Colab con T4 o A100).

---

## 3. Parte 1: Feature Visualization con torch-lucent

### Concepto

La idea central es realizar **gradient ascent sobre la imagen de entrada** para maximizar la activación de un objetivo dentro de la red. Los pesos de la red se congelan; lo que se optimiza es la imagen.

```
imagen_inicial (ruido)
       ↓
    red(imagen)  →  activación_objetivo
       ↓ backward
  grad respecto a imagen
       ↓
  imagen += learning_rate × grad  (gradient ascent)
```

### Función principal: `render.render_vis()`

```python
imgs = render.render_vis(
    model,          # modelo PyTorch
    "nombre_capa:canal",  # objetivo (ej: "features_10:45")
    show_inline=True
)
```

La función devuelve una lista de imágenes numpy que representan el input optimizado.

### Función auxiliar del laboratorio: `get_images()`

```python
def get_images(model, layer_list, rows, cols):
    """
    Genera una grilla de visualizaciones de features.
    
    Args:
        model: modelo CNN
        layer_list: lista de nombres de capas a visualizar
        rows, cols: dimensiones de la grilla
    """
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    
    for i, layer_name in enumerate(layer_list):
        for j in range(cols):
            imgs = render.render_vis(model, f"{layer_name}:{j}")
            ax = axes[i][j] if rows > 1 else axes[j]
            ax.imshow(imgs[0])
            ax.axis('off')
            ax.set_title(f"{layer_name}\nchannel {j}", fontsize=8)
    
    plt.tight_layout()
    plt.show()
```

---

## 4. Parte 2: Arquitecturas Disponibles

### Carga de modelos pre-entrenados

```python
# Todos pre-entrenados en ImageNet
alexnet   = models.alexnet(pretrained=True).to(device).eval()
googlenet = models.googlenet(pretrained=True).to(device).eval()
vgg19     = models.vgg19(pretrained=True).to(device).eval()
resnet50  = models.resnet50(pretrained=True).to(device).eval()
```

### Explorar estructura de capas

torch-lucent incluye `get_model_layers()` para listar todos los nombres de capas disponibles:

```python
print(get_model_layers(alexnet))
# ['features_0', 'features_1', 'features_2', ...]
```

### AlexNet — Estructura de capas

| Nombre lucent | Operación | Salida |
|--------------|-----------|--------|
| `features_0` | Conv2d(3, 64, 11×11, stride=4) | 64 canales |
| `features_1` | ReLU | — |
| `features_2` | MaxPool2d | — |
| `features_3` | Conv2d(64, 192, 5×5) | 192 canales |
| `features_4` | ReLU | — |
| `features_5` | MaxPool2d | — |
| `features_6` | Conv2d(192, 384, 3×3) | 384 canales |
| `features_8` | Conv2d(384, 256, 3×3) | 256 canales |
| `features_10` | Conv2d(256, 256, 3×3) | 256 canales |
| `features_12` | MaxPool2d | — |
| `classifier_1` | Dropout | — |
| `classifier_2` | Linear(9216, 4096) | — |
| `classifier_4` | Linear(4096, 4096) | — |
| `classifier_6` | Linear(4096, 1000) | — |

> **Convención PyTorch:** Las capas dentro de `nn.Sequential` se nombran con guion bajo: `features_0`, `features_1`, etc.

### VGG19 — Estructura de capas

VGG19 tiene **16 capas convolucionales** más 3 FC. El módulo `features` contiene todas las convoluciones:

```
features_0:  Conv2d(3, 64, 3×3)      → conv1_1
features_2:  Conv2d(64, 64, 3×3)     → conv1_2
features_5:  Conv2d(64, 128, 3×3)    → conv2_1
features_7:  Conv2d(128, 128, 3×3)   → conv2_2
features_10: Conv2d(128, 256, 3×3)   → conv3_1
...
features_36: Conv2d(512, 512, 3×3)   → conv5_4  (última conv)
```

### GoogLeNet (Inception v1) — Estructura de capas

```python
# Las primeras 20 capas
get_model_layers(googlenet)[:20]
# ['conv1', 'conv1_relu_conv1', 'pool1', 'lrn1',
#  'conv2', 'conv2_relu_conv2', 'conv3', ...,
#  'mixed3a', 'mixed3a_1x1', ...]
```

Estructura:
- 2 capas conv iniciales
- 9 módulos Inception: `mixed3a`, `mixed3b`, `mixed4a`–`mixed4e`, `mixed5a`, `mixed5b`
- Las sub-capas de cada módulo: `_1x1`, `_3x3`, `_5x5`, `_pool_proj`

### ResNet50 — Estructura de capas

ResNet50 es la más compleja. Sus grupos principales:

```
conv1     → 7×7 conv, stride 2
bn1
relu
maxpool
layer1    → 3 bloques bottleneck
layer2    → 4 bloques bottleneck
layer3    → 6 bloques bottleneck
layer4    → 3 bloques bottleneck
avgpool
fc
```

> **Recomendación del lab:** Para ResNet50 enfocarse en los grupos principales (`layer1`, `layer2`, `layer3`, `layer4`) en lugar de navegar cada sub-bloque individualmente.

---

## 5. Parte 3: Visualización por Capas

### ¿Qué esperar en cada etapa de la red?

La visualización revela la **jerarquía de representaciones** que aprenden las CNNs:

```
Capas tempranas → Capas medias → Capas profundas
    Bordes           Texturas       Objetos
  Gradientes         Patrones       Partes
   Colores            Formas       Conceptos
```

### Ejemplo con AlexNet

```python
# Capas de interés para visualizar
layers_to_visualize = [
    "features_1",   # Primera capa: bordes y colores
    "features_6",   # Capa media: texturas y patrones
    "features_10",  # Última conv: partes de objetos
]

get_images(alexnet, layers_to_visualize, rows=3, cols=3)
```

### Ejemplo con GoogLeNet

```python
layers_googlenet = [
    "conv2d0",   # Bordes
    "mixed3a",   # Texturas
    "mixed4a",   # Patrones
    "mixed4e",   # Objetos
]
```

### Visualización de clases completas

También es posible visualizar qué imagen maximiza la probabilidad de una clase específica:

```python
# label:162 → Beagle (perro)
# label:8   → Gallo
imgs = render.render_vis(model, "labels:162")
```

> **Observación importante:** Las imágenes generadas para "labels" son visualizaciones de la clase entera, no de imágenes reales. Son alucinaciones del modelo que revelan qué patrones asocia con cada clase.

---

## 6. Parte 4: Dataset de Flores y Modelos Propios

### El Dataset: Oxford 102 Flowers

```python
# Descargar y extraer
!wget -O lab1_CNN_IA.zip "URL_DROPBOX"
!unzip lab1_CNN_IA.zip
!tar -xzf flowers.tar.gz
```

- **102 categorías** de flores
- Estructura: `flores/train/clase_id/imagen.jpg`
- Ejemplo de clases: 53=girasol, 72=lirio de agua, 73=rosa

### Modelo MiAlexNet (entrenado desde cero)

```python
class MiAlexNet(nn.Module):
    def __init__(self, num_classes=102):
        super(MiAlexNet, self).__init__()
        
        self.features = nn.Sequential(
            # conv1: 3×224×224 → 96×55×55 → (MaxPool) → 96×27×27
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # conv2: 96×27×27 → 256×27×27 → (MaxPool) → 256×13×13
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # conv3: 256×13×13 → 384×13×13
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # conv4: 384×13×13 → 384×13×13
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # conv5: 384×13×13 → 256×13×13 → (MaxPool) → 256×6×6
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),          # 256×6×6 = 9216
            nn.Linear(9216, 4096), # fc6
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096), # fc7
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),  # fc8 → 102 clases
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

#### Resultado de MiAlexNet (entrenado en flores desde cero)

| Métrica | Valor |
|---------|-------|
| Accuracy entrenamiento | **98.21%** |
| Accuracy test | **80.23%** |
| Gap (overfitting) | ~18 puntos |

El modelo memorizó el dataset de entrenamiento pero no generalizó bien.

### Modelo Fine-tuned (Transfer Learning desde ImageNet)

```python
# Cargar AlexNet pre-entrenado
model_ft = models.alexnet(pretrained=True)

# Reemplazar la última capa para 102 clases
model_ft.classifier[6] = nn.Linear(4096, 102)

# Cargar pesos fine-tuned
model_ft.load_state_dict(torch.load('finetuned_alexnet.pth'))
```

#### Resultado del modelo Fine-tuned

| Métrica | Valor |
|---------|-------|
| Accuracy entrenamiento | **99.98%** |
| Accuracy test | **95.91%** |
| Gap (overfitting) | ~4 puntos |

### Comparación directa

| Modelo | Train | Test | Gap |
|--------|-------|------|-----|
| MiAlexNet (scratch) | 98.21% | 80.23% | **18 pp** |
| AlexNet Fine-tuned | 99.98% | 95.91% | **4 pp** |

> **Conclusión clave:** El fine-tuning desde ImageNet reduce el gap de overfitting en ~14 puntos porcentuales. Las representaciones pre-aprendidas (bordes, texturas, formas) son transferibles y generalizan mucho mejor que aprender desde cero con un dataset pequeño.

### Dataset personalizado: clase Flowers

```python
class Flowers(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.transform = transform
        self.images, self.labels = self.armar_indices(split)
    
    def armar_indices(self, split):
        images, labels = [], []
        split_dir = os.path.join(self.root, split)
        for class_id in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_id)
            for img_name in os.listdir(class_dir):
                images.append(os.path.join(class_dir, img_name))
                labels.append(int(class_id))
        return images, labels
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]
    
    def __len__(self):
        return len(self.images)

# Transformaciones estándar ImageNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

---

## 7. Parte 5: Atribución con Perturbación Extremal

### Setup

```python
from torchray.attribution.extremal_perturbation import (
    extremal_perturbation, 
    contrastive_reward
)
from torchray.utils import get_example_data
from torchray.visualization.plots import plot_attribution
```

### Uso básico

```python
# Cargar imagen de prueba
img = Image.open('test_image.jpg')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
input_tensor = transform(img).unsqueeze(0).to(device)

# Categorías de interés
cat_dog = 245   # "bulldog" en ImageNet
cat_cat = 285   # "Egyptian cat"

# Aplicar Extremal Perturbation
mask_dog, _ = extremal_perturbation(
    alexnet,                # modelo
    input_tensor,           # imagen
    cat_dog,                # clase objetivo
    reward_func=contrastive_reward,
    debug=True,
    areas=[0.1]             # mostrar 10% del área
)
```

### Parámetros importantes

| Parámetro | Descripción | Valor típico |
|-----------|-------------|-------------|
| `areas` | Porcentaje del área visible | `[0.05, 0.1, 0.2]` |
| `reward_func` | Función de recompensa | `contrastive_reward` |
| `debug` | Mostrar progreso de optimización | `True` |
| `perturbation` | Tipo de perturbación | `'blur'` (por defecto) |

### Visualización de resultados

```python
# Visualizar con overlay
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Imagen original
axes[0].imshow(img)
axes[0].set_title('Original')

# Máscara para clase "dog"
axes[1].imshow(mask_dog.squeeze().cpu(), cmap='jet')
axes[1].set_title(f'Attribution: dog (id={cat_dog})')

# Superposición
axes[2].imshow(img)
axes[2].imshow(mask_dog.squeeze().cpu(), alpha=0.6, cmap='jet')
axes[2].set_title('Overlay')

plt.tight_layout()
plt.show()
```

### Interpretación de la máscara

- **Zonas rojas/cálidas:** Alta importancia para la predicción
- **Zonas azules/frías:** Baja importancia
- Una máscara bien centrada sobre el objeto indica que el modelo aprendió la feature correcta
- Una máscara dispersa o en el fondo indica bias o shortcuts

---

## 8. Parte 6: Comparación Base vs Fine-tuned

### Funciones auxiliares

```python
def get_wrong_indices(model, dataset, correct_class, limit=10):
    """
    Encuentra imágenes que el modelo clasifica incorrectamente
    pero que pertenecen a correct_class.
    """
    wrong = []
    model.eval()
    with torch.no_grad():
        for idx in range(len(dataset)):
            img, label = dataset[idx]
            if label != correct_class:
                continue
            output = model(img.unsqueeze(0).to(device))
            pred = output.argmax(1).item()
            if pred != label:
                wrong.append(idx)
            if len(wrong) >= limit:
                break
    return wrong


def view_example(dataset, idx):
    """Muestra imagen del dataset dado su índice."""
    img, label = dataset[idx]
    # Desnormalizar para visualización
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    img_show = img * std + mean
    img_show = img_show.permute(1,2,0).numpy().clip(0,1)
    plt.imshow(img_show)
    plt.title(f"Clase: {label}")
    plt.show()
```

### Análisis comparativo

```python
# Misma imagen → dos modelos → atribuciones diferentes

img_tensor, label = dataset[idx]
img_input = img_tensor.unsqueeze(0).to(device)

# Modelo base (entrenado en flores desde cero)
mask_base, _ = extremal_perturbation(
    model_base, img_input, label, areas=[0.1]
)

# Modelo fine-tuned (pre-entrenado en ImageNet)
mask_ft, _ = extremal_perturbation(
    model_finetuned, img_input, label, areas=[0.1]
)

# Comparar side-by-side
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(desnormalizar(img_tensor))
axes[0].set_title('Imagen original')
axes[1].imshow(mask_base.squeeze().cpu(), cmap='hot')
axes[1].set_title('Base (desde cero)\nFoco en fondo/contexto')
axes[2].imshow(mask_ft.squeeze().cpu(), cmap='hot')
axes[2].set_title('Fine-tuned (ImageNet)\nFoco en la flor')
plt.tight_layout()
plt.show()
```

### Resultados esperados y por qué

| Modelo | Dónde mira la red | Por qué |
|--------|-------------------|---------|
| Entrenado desde cero | Fondo, suelo, contexto | Dataset pequeño → correlaciones espurias |
| Fine-tuned | La flor en sí | Representaciones de ImageNet son generales y robustas |

> Este experimento ilustra por qué **el fine-tuning no solo mejora el accuracy**, sino también la **calidad y relevancia de las features aprendidas**.

---

## 9. Ejercicios

### Ejercicio 1: Visualización de flores con Feature Visualization

**Objetivo:** Comparar imágenes reales de flores vs. imágenes generadas por optimización.

```python
# 3 flores a elegir (ej: girasol=53, lirio=72, rosa=73)
flores = [53, 72, 73]

for clase in flores:
    # Imagen real del dataset
    idx = dataset_train.label_to_idx[clase][0]
    view_example(dataset_train, idx)
    
    # Imagen generada por la red (fine-tuned)
    imgs = render.render_vis(model_finetuned, f"labels:{clase}")
    plt.imshow(imgs[0])
    plt.title(f"Clase {clase} — Feature Visualization")
    plt.show()
```

**Preguntas guía:**
- ¿Las imágenes generadas se parecen a flores reales?
- ¿Qué colores y patrones predominan?
- ¿El modelo fine-tuned genera imágenes más reconocibles que el base?

---

### Ejercicio 2: Jerarquía de features

**Objetivo:** Visualizar 4 canales en 3 etapas distintas de la red (primera conv, capa media, última conv) para cualquiera de los modelos.

```python
# Ejemplo con GoogLeNet fine-tuned
capas = [
    "conv2d0",    # Primera capa → bordes
    "mixed3b",    # Capa media → texturas/patrones
    "mixed5b",    # Última capa → objetos/partes
]

get_images(model_finetuned, capas, rows=3, cols=4)
```

**Análisis esperado:**
- Capa temprana: bordes, gradientes de color
- Capa media: texturas, repeticiones, patrones locales
- Capa profunda: partes reconocibles de objetos

---

### Ejercicio 3: ¿Qué "ve" la red en las flores?

**Objetivo:** Visualizar las imágenes generadas para las 3 flores elegidas y evaluar si tienen sentido biológico.

```python
# Para el modelo fine-tuned
for clase_id in [53, 72, 73]:
    imgs_base = render.render_vis(model_base, f"labels:{clase_id}")
    imgs_ft   = render.render_vis(model_finetuned, f"labels:{clase_id}")
    
    # Comparar lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(imgs_base[0])
    axes[0].set_title(f"Base — clase {clase_id}")
    axes[1].imshow(imgs_ft[0])
    axes[1].set_title(f"Fine-tuned — clase {clase_id}")
    plt.show()
```

---

### Ejercicio 4: Fine-tuning de otra arquitectura

**Objetivo:** Tomar VGG19, GoogLeNet o ResNet50, fine-tunearlo en el dataset de flores, y comparar con AlexNet.

```python
# Opción A: VGG19
model_vgg = models.vgg19(pretrained=True)
model_vgg.classifier[6] = nn.Linear(4096, 102)

# Opción B: GoogLeNet
model_google = models.googlenet(pretrained=True)
model_google.fc = nn.Linear(1024, 102)

# Opción C: ResNet50
model_resnet = models.resnet50(pretrained=True)
model_resnet.fc = nn.Linear(2048, 102)

# Training loop (mínimo 20 épocas)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(20):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Evaluar en test
    model.eval()
    # ... calcular accuracy
```

**Comparativa a reportar:**

| Arquitectura | Parámetros | Tiempo/época | Accuracy Test |
|-------------|-----------|-------------|--------------|
| AlexNet | ~61M | — | 95.91% |
| VGG19 | 144M | (más lento) | ? |
| GoogLeNet | ~6.8M | (más rápido) | ? |
| ResNet50 | ~25M | — | ? |

---

### Ejercicio 5: Visualización del nuevo modelo

**Objetivo:** Repetir la visualización de las 3 flores con la nueva arquitectura entrenada y comparar con AlexNet fine-tuned.

```python
# Visualizar mismas 3 flores con nueva arquitectura
for clase_id in [53, 72, 73]:
    imgs_alexnet = render.render_vis(model_alexnet_ft, f"labels:{clase_id}")
    imgs_nuevo   = render.render_vis(model_nuevo_ft,   f"labels:{clase_id}")
    
    # Side-by-side
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(imgs_alexnet[0])
    axes[0].set_title("AlexNet fine-tuned")
    axes[1].imshow(imgs_nuevo[0])
    axes[1].set_title("Nueva arquitectura")
    plt.show()
```

**Análisis:**
- ¿Las visualizaciones son más claras con GoogLeNet/ResNet?
- ¿Las features de flores son más específicas o más genéricas?
- ¿Hay diferencias notables entre las representaciones?

---

## 10. Conceptos Técnicos Clave

### Gradient Ascent en Feature Visualization

Mientras que el entrenamiento usa **gradient descent** sobre los pesos para minimizar la loss, Feature Visualization usa **gradient ascent** sobre la imagen para maximizar una activación:

```python
# Gradient descent (entrenamiento)
loss = criterion(output, target)
loss.backward()
optimizer.step()  # actualiza PESOS

# Gradient ascent (feature visualization, simplificado)
activation = model(img)[target_layer][target_channel]
activation.backward()
img = img + learning_rate * img.grad  # actualiza IMAGEN
```

La red se congela; la imagen es el parámetro a optimizar.

### Transfer Learning vs Fine-tuning

| Concepto | Descripción |
|---------|-------------|
| **Transfer Learning** | Usar pesos pre-entrenados como punto de inicio |
| **Fine-tuning** | Continuar entrenando todos (o parte) de los pesos en el nuevo dataset |
| **Feature extraction** | Congelar la backbone; solo entrenar la capa final |

### Normalización ImageNet

Casi todos los modelos pre-entrenados esperan esta normalización:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

# Aplicar
transform = transforms.Normalize(mean=mean, std=std)

# Desnormalizar para visualizar
def desnormalizar(img_tensor):
    mean_t = torch.tensor(mean).view(3,1,1)
    std_t  = torch.tensor(std).view(3,1,1)
    return (img_tensor * std_t + mean_t).clamp(0, 1)
```

### Batch Normalization y su rol en visualización

Batch Normalization puede causar problemas en Feature Visualization porque las estadísticas del batch cambian cuando optimizamos una sola imagen. Para esto se pone el modelo en modo `.eval()` antes de visualizar.

### Extremal Perturbation: detalles matemáticos

La optimización busca la máscara `m` que maximiza:

```
Φ(m ⊗ x) = Φ(m · x + (1-m) · blur(x))
```

Donde `blur(x)` es la imagen difuminada (el "fondo neutro"). La máscara `m` es suave (valores entre 0 y 1), con la restricción de que su área promedio sea `a`.

El proceso de optimización tiene ~800 iteraciones y usa una **pirámide de perturbaciones** (múltiples escalas de blur) para hacer la máscara más robusta.

---

## Resumen de Resultados Esperados

### Feature Visualization

| Capa | Tipo de feature visible |
|------|------------------------|
| Primera conv | Bordes diagonales, gradientes de color |
| Conv media | Texturas repetitivas, patrones geométricos |
| Conv profunda | Partes de objetos, patrones complejos |
| Clase (labels) | "Alucinación" del objeto completo |

### Attribution con Extremal Perturbation

| Modelo | Comportamiento de la máscara |
|--------|------------------------------|
| Entrenado desde cero | Difuso, en fondo o contexto |
| Fine-tuned ImageNet | Concentrado sobre el objeto de interés |

### Transfer Learning

| Estrategia | Accuracy Test (aprox.) | Overfitting |
|------------|----------------------|-------------|
| Desde cero (pequeño dataset) | ~80% | Alto |
| Fine-tuning desde ImageNet | ~96% | Bajo |

---

## Glosario del Laboratorio

| Término | Definición |
|---------|-----------|
| `render.render_vis()` | Función principal de torch-lucent para feature visualization |
| `get_model_layers()` | Lista los nombres de capas de un modelo para usar con lucent |
| `extremal_perturbation()` | Función de torchray que aprende la máscara de atribución |
| `contrastive_reward` | Función de reward que penaliza otras clases al maximizar la objetivo |
| `areas` | Fracción del área de imagen que puede revelar la máscara (0.05 = 5%) |
| Decorrelated space | Espacio de optimización en frecuencias para generar imágenes más naturales |
| Transformation robustness | Aplicar jitter/rotate/scale durante optimización para features más robustas |
| `.eval()` | Modo evaluación: desactiva Dropout y usa estadísticas globales en BatchNorm |
| `.train()` | Modo entrenamiento: activa Dropout y calcula estadísticas de BatchNorm por batch |
