---
title: "Resolución del Laboratorio"
weight: 60
math: true
---

Resolución completa de las 5 actividades del laboratorio. Arquitectura elegida: **ResNet50**.

---

## Actividad 1 — ResNet50 desde cero

### Configuración del modelo

```python
import torch
import torchvision.models as models
import torch.nn as nn

# Crear ResNet50 con 102 clases (flores)
model = models.resnet50()
model.fc = nn.Linear(2048, 102)

# Descargar pesos entrenados desde cero en flores (40 épocas)
!wget -q https://www.dropbox.com/s/q308auknnt8szzw/base_resnet50.pth

# Cargar pesos
model.load_state_dict(torch.load('base_resnet50.pth', map_location='cpu'))
model = model.to(device)
model.eval()
```

**¿Por qué `model.fc = nn.Linear(2048, 102)`?**

ResNet50 termina con un `avgpool` que produce un vector de 2048 dimensiones, seguido de `fc` que por defecto mapea a 1000 clases (ImageNet). Se reemplaza esa última capa para producir 102 salidas, una por clase de flor.

### Evaluación (sin normalización)

```python
transform_base = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # sin Normalize — así fue entrenado
])

dataset_train = Flowers(root='flowers_dataset/train', transform=transform_base)
dataset_test  = Flowers(root='flowers_dataset/test',  transform=transform_base)

loader_train = DataLoader(dataset_train, batch_size=64, shuffle=False)
loader_test  = DataLoader(dataset_test,  batch_size=64, shuffle=False)

test_model(model, loader_train)  # Train accuracy: 91.82%
test_model(model, loader_test)   # Test accuracy:  82.30%
```

### Resultados

| Modelo                 | Train  | Test   | Gap    |
|------------------------|--------|--------|--------|
| MiAlexNet (desde cero) | 98.21% | 80.23% | 18 pp  |
| ResNet50 (desde cero)  | 91.82% | 82.30% | 9.5 pp |

### Análisis

ResNet50 generaliza mejor que AlexNet (80.23% test) a pesar de entrenar desde cero. El gap de overfitting es menor (9.5 pp vs 18 pp) gracias a dos mecanismos:

- **Batch Normalization** — presente en cada bloque residual, actúa como regularizador implícito y estabiliza el entrenamiento.
- **Conexiones residuales (skip connections)** — permiten que los gradientes fluyan sin degradarse, lo que hace que el modelo aprenda representaciones más robustas en lugar de memorizar.

El accuracy de train es menor que el de AlexNet (91% vs 98%), lo que indica que ResNet50 resiste más la memorización. A cambio, generaliza ligeramente mejor en test (82% vs 80%).

---

## Actividad 2 — Feature Visualization por capas

### Capas seleccionadas (Actividad 2)

```python
capas_resnet = [
    'conv1',           # Primera capa: 7×7, 64 canales
    'layer2_1_conv2',  # Capa media: 128 canales
    'layer4_2_conv3',  # Última conv: 2048 canales
]

get_images(model, capas_resnet, rows=3, cols=4, preprocess=False)
```

### Resultados (Actividad 2)

**`conv1` — Colores puros:** Bloques grandes de color uniforme (negro, blanco, magenta, amarillo, cyan, rojo). Sin estructura espacial — el filtro 7×7 solo alcanza a detectar presencia de color en zonas amplias.

**`layer2_1_conv2` — Texturas y patrones:** Grillas de puntos, líneas diagonales, ondas, dameros. Hay repetición estructurada y direccionalidad — ya no es solo color sino organización espacial.

**`layer4_2_conv3` — Partes de objetos:** Formas orgánicas y curvas. Manchas con estructura que recuerdan pétalos y centros de flores. La red combina textura, forma y contexto.

### Preguntas (Actividad 2)

**¿Cómo son los features?**

Los features varían según la profundidad de la capa. En `conv1` son detectores de color puro — cada canal responde a un tono dominante (rojo, cyan, magenta) sin distinguir formas. En `layer2_1_conv2` aparecen texturas con dirección y frecuencia: grillas, ondas, rayas diagonales. En `layer4_2_conv3` los features son orgánicos y curvos, con estructura interna que recuerda pétalos o centros de flores.

**¿Se aprecia un aumento en la complejidad de los features a medida que las capas son más profundas?**

Sí, claramente. Cada capa construye sobre la anterior en forma jerárquica:

- `conv1` detecta presencia de color en zonas amplias
- `layer2` combina colores para detectar texturas y patrones repetitivos
- `layer4` combina texturas para representar partes de objetos concretos

Este aumento progresivo de complejidad es la razón fundamental por la que las CNNs profundas funcionan: no aprenden todo de una vez, sino abstrayendo el input en etapas sucesivas.

---

## Actividad 3 — Visualización de clases de flores

### Código (Actividad 3)

```python
from lucent.optvis import render

# 3 clases: 53=sunflower, 72=water lily, 73=rose
_ = render.render_vis(model, 'labels:53', show_image=True, preprocess=False)
_ = render.render_vis(model, 'labels:72', show_image=True, preprocess=False)
_ = render.render_vis(model, 'labels:73', show_image=True, preprocess=False)
```

### Resultados (Actividad 3)

**53 — Sunflower:** Color amarillo/verde con puntos. Patrón granulado que evoca semillas de girasol.

**72 — Water lily:** Rosa/magenta intenso. Formas difusas sin estructura clara.

**73 — Rose:** Naranja/rojo con forma espiral. Hay una forma circular que recuerda una rosa.

### Preguntas (Actividad 3)

**¿Tienen sentido las imágenes generadas?**

Parcialmente. Los colores dominantes son coherentes con las flores reales — amarillo para el girasol, rosa para el lirio, naranja/rojo para la rosa. Sin embargo, las formas son caóticas y psicodélicas. No se puede distinguir claramente la morfología de cada flor.

**¿Representan las flores esperadas?**

Solo en términos de color, no de forma. El modelo aprendió correlaciones de color asociadas a cada clase, pero no capturó la estructura visual de las flores. Esto es consecuencia de haber entrenado desde cero con un dataset pequeño — la red memorizó señales superficiales en lugar de aprender representaciones visuales ricas. En la Actividad 5 se compararán estas imágenes con las del modelo fine-tuned.

---

## Actividad 4 — Fine-tuning desde ImageNet

### Código (Actividad 4)

```python
import torch.optim as optim

# ResNet50 con pesos ImageNet, backbone congelada
model_ft = models.resnet50(weights="IMAGENET1K_V1")
for param in model_ft.parameters():
    param.requires_grad = False
model_ft.fc = nn.Linear(2048, 102)
model_ft = model_ft.to(device)

# DataLoaders con normalización ImageNet
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
transform_ft = Compose([Resize((224, 224)), ToTensor(), Normalize(mean, std)])

dataset_train_ft = Flowers(root='flowers_dataset/train', transform=transform_ft)
dataset_test_ft  = Flowers(root='flowers_dataset/test',  transform=transform_ft)

loader_train_ft = DataLoader(dataset_train_ft, batch_size=64, shuffle=True)
loader_test_ft  = DataLoader(dataset_test_ft,  batch_size=64, shuffle=False)

# Entrenamiento 20 épocas — solo se actualiza model_ft.fc
optimizer = optim.Adam(model_ft.fc.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(20):
    model_ft.train()
    running_loss = 0.0
    for imgs, labels in loader_train_ft:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model_ft(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Época {epoch+1}/20 — Loss: {running_loss/len(loader_train_ft):.4f}")

test_model(model_ft, loader_train_ft)
test_model(model_ft, loader_test_ft)
```

### Evolución del loss

| Época | Loss   | Época | Loss   |
|-------|--------|-------|--------|
| 1     | 2.6770 | 11    | 0.0794 |
| 2     | 0.8403 | 12    | 0.0710 |
| 3     | 0.4883 | 13    | 0.0603 |
| 4     | 0.3342 | 14    | 0.0533 |
| 5     | 0.2482 | 15    | 0.0483 |
| 6     | 0.1941 | 16    | 0.0434 |
| 7     | 0.1530 | 17    | 0.0380 |
| 8     | 0.1301 | 18    | 0.0362 |
| 9     | 0.1111 | 19    | 0.0322 |
| 10    | 0.0929 | 20    | 0.0307 |

### Resultados (Actividad 4)

| Modelo                  | Train   | Test   | Gap    |
|-------------------------|---------|--------|--------|
| MiAlexNet (desde cero)  | 98.21%  | 80.23% | 18 pp  |
| ResNet50 (desde cero)   | 91.82%  | 82.30% | 9.5 pp |
| AlexNet (fine-tuned)    | ~100%   | ~96%   | ~4 pp  |
| ResNet50 (fine-tuned)   | 100.00% | 97.97% | ~2 pp  |

### Análisis (Actividad 4)

El fine-tuning mejora el accuracy en test de 82.30% a 97.97% (+16 pp) y reduce el gap de overfitting de 9.5 pp a 2 pp. Las features de ImageNet son tan generales que la red casi no sobreajusta, a diferencia del modelo entrenado desde cero.

La caída del loss de 2.67 a 0.03 en 20 épocas muestra una convergencia muy limpia. ResNet50 fine-tuned supera también a AlexNet fine-tuned (~98% vs ~96%) gracias a:

- **Mayor capacidad de extracción de features** — los bloques residuales de ResNet50 aprenden representaciones más ricas que las capas secuenciales de AlexNet.
- **Batch Normalization** — estabiliza la distribución de activaciones en cada bloque, lo que hace que las features extraídas sean más consistentes y útiles para la capa de clasificación.

---

## Actividad 5 — Comparación visual post fine-tuning

### Código (Actividad 5)

```python
# Mismas 3 flores, ahora con el modelo fine-tuned
# Sin preprocess=False — este modelo sí usa normalización ImageNet
_ = render.render_vis(model_ft, 'labels:53', show_image=True)
_ = render.render_vis(model_ft, 'labels:72', show_image=True)
_ = render.render_vis(model_ft, 'labels:73', show_image=True)
```

### Comparación con Actividad 3

| Clase | Modelo base (Act. 3) | Modelo fine-tuned (Act. 5) |
|-------|---------------------|---------------------------|
| 53 — Sunflower | Amarillo/verde, granulado simple | Estructuras fractales complejas, verdes y dorados |
| 72 — Water lily | Rosa/magenta difuso, sin forma | Patrones geométricos densos en verdes oscuros |
| 73 — Rose | Rojo/naranja con espiral simple | Espirales morado/rosa intrincadas, más detalle |

### Análisis (Actividad 5)

El modelo fine-tuned genera imágenes notablemente más complejas que el modelo base. Mientras el modelo desde cero producía colores planos y formas simples (solo captó correlaciones de color), el fine-tuned muestra estructuras fractales con múltiples capas de detalle. Ambos producen imágenes abstractas — feature visualization no genera fotos reales, sino que optimiza para maximizar la activación de una clase.

Esta mayor riqueza visual refleja que las representaciones internas heredadas de ImageNet son más sofisticadas. El modelo base aprendió features simples (colores planos, patrones básicos) con un dataset pequeño. El fine-tuned heredó texturas, formas y jerarquías profundas, lo que explica directamente la diferencia en accuracy: 82.30% vs 97.97%.
