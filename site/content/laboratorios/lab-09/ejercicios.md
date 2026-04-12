---
title: "Ejercicios y Glosario"
weight: 50
math: true
---

## 1. Ejercicios del Laboratorio

Estos ejercicios aplican las tecnicas de feature visualization y attribution vistas en el laboratorio sobre nuevas arquitecturas y datos.

---

### Ejercicio 1: Adaptar Arquitectura para Flowers

**Contexto:** Modificar VGG19, GoogLeNet o ResNet50 para que sea compatible con las 102 clases del Flower Dataset.

```python
import torch
import torchvision.models as models
import torch.nn as nn

# VGG19
model = models.vgg19()
model.classifier = nn.Sequential(
    nn.Linear(25088, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(),
    nn.Linear(4096, 102),
)

# GoogLeNet
model = models.googlenet()
model.fc = nn.Linear(1024, 102)

# ResNet50
model = models.resnet50()
model.fc = nn.Linear(2048, 102)
```

**Que observar:** cada arquitectura tiene un componente de clasificacion diferente. En VGG19 es `model.classifier` (un Sequential completo). En GoogLeNet y ResNet50 es `model.fc` (una sola capa Linear).

---

### Ejercicio 2: Feature Hierarchy

**Contexto:** Visualizar 4 canales de la primera capa convolucional, una capa intermedia y la ultima capa convolucional del modelo entrenado.

```python
# Ejemplo para VGG19
get_images(model, ['features_1', 'features_17', 'features_35'], 2, 2, preprocess=False)
```

**Que observar:**

- Primera capa: patrones simples (bordes, colores)
- Capa intermedia: texturas y patrones mas complejos
- Ultima capa: partes de objetos o conceptos
- Se aprecia un aumento en complejidad a medida que las capas son mas profundas?

{{< concept-alert type="clave" >}}
Si el modelo no fue entrenado con datos normalizados, usar `preprocess=False`. Si usaste fine-tuning desde ImageNet (con normalizacion), dejar `preprocess=True` (default).
{{< /concept-alert >}}

---

### Ejercicio 3: Label Visualization de 3 Flores

**Contexto:** Escoger 3 clases de flores y visualizar la imagen generada para cada una.

Clases sugeridas: 53 (sunflower), 72 (water lily), 73 (rose). Lista completa de clases disponible en la [documentacion del dataset](https://gist.github.com/JosephKJ/94c7728ed1a8e0cd87fe6a029769cde1) (restar 1 al valor de la linea, ya que las clases comienzan en 0).

```python
_ = render.render_vis(model, 'labels:53', show_image=True, preprocess=False)
_ = render.render_vis(model, 'labels:72', show_image=True, preprocess=False)
_ = render.render_vis(model, 'labels:73', show_image=True, preprocess=False)
```

**Que observar:** tienen sentido las imagenes generadas? Se pueden distinguir colores o texturas asociadas a cada flor? Comparar con fotos reales del dataset.

---

### Ejercicio 4: Fine-tuning desde ImageNet

**Contexto:** Usar la version preentrenada en ImageNet del modelo escogido y entrenar por al menos 20 epocas en el Flower Dataset.

```python
# VGG19 con pesos preentrenados
model = models.vgg19(weights="IMAGENET1K_V1")
for param in model.parameters():
    param.requires_grad = False  # congelar todas las capas
model.classifier = nn.Sequential(
    nn.Linear(25088, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(),
    nn.Linear(4096, 102),
)

# GoogLeNet con pesos preentrenados
model = models.googlenet(weights="IMAGENET1K_V1")
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(1024, 102)

# ResNet50 con pesos preentrenados
model = models.resnet50(weights="IMAGENET1K_V1")
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(2048, 102)
```

**Que observar:** llenar esta tabla comparativa:

| Modelo | Parametros totales | Accuracy train | Accuracy test | Gap |
|---|---|---|---|---|
| MiAlexNet (desde cero) | ~60M | 98.17% | 80.2% | 18pp |
| AlexNet (fine-tuned) | ~60M | ~100% | ~96% | ~4pp |
| Tu modelo (desde cero) | | | | |
| Tu modelo (fine-tuned) | | | | |

---

### Ejercicio 5: Comparacion Visual post Fine-tuning

**Contexto:** Repetir las visualizaciones del ejercicio 3 con el modelo fine-tuned y comparar.

```python
# Ahora SIN preprocess=False (el modelo fine-tuned usa normalizacion)
_ = render.render_vis(model, 'labels:53', show_image=True)
_ = render.render_vis(model, 'labels:72', show_image=True)
_ = render.render_vis(model, 'labels:73', show_image=True)
```

**Que observar:**

- Las imagenes son mas claras y especificas que las del ejercicio 3?
- Se distinguen mejor los colores y texturas de cada flor?
- Comparar con las visualizaciones de AlexNet fine-tuned — son mejores o peores?

---

## 2. Glosario del Laboratorio

| Termino | Descripcion |
|---|---|
| `render.render_vis()` | Genera la imagen que maximiza la activacion de una capa, canal o clase |
| `get_model_layers()` | Lista los nombres de capas accesibles de un modelo PyTorch |
| `extremal_perturbation()` | Encuentra la mascara optima que maximiza la respuesta de una clase dada un area fija |
| `contrastive_reward` | Funcion de recompensa que maximiza la clase objetivo y minimiza el resto |
| `areas` | Fraccion de la imagen visible a traves de la mascara (ej: 0.12 = 12%) |
| Decorrelated space | Optimizar la imagen en espacio de frecuencias en vez de pixeles para resultados mas naturales |
| Transformation robustness | Aplicar jitter, rotacion y escala durante la optimizacion para mayor robustez |
| `.eval()` | Modo evaluacion de PyTorch: batch norm usa estadisticas globales, dropout desactivado |
| `preprocess` | Flag de lucent para aplicar (True) o no (False) la normalizacion de ImageNet sobre la imagen |
