---
title: "Librerias del Laboratorio"
weight: 10
math: true
---

## 1. Vision General

Para entender que aprende una CNN en cada capa, PyTorch base no es suficiente. Necesitamos herramientas especializadas que permitan visualizar las **features** internas del modelo y atribuir decisiones a regiones especificas de la imagen de entrada. Este laboratorio combina cinco librerias que cubren desde la carga de modelos pre-entrenados hasta la generacion de visualizaciones interpretables.

| Libreria | Proposito | Origen |
|---|---|---|
| PyTorch + torchvision | Framework DL + modelos pre-entrenados | Meta AI |
| torch-lucent | Feature Visualization (optimizacion de input) | Port de Lucid (Distill/Google Brain) |
| torchray | Attribution (Extremal Perturbation) | Facebook Research |
| PIL (Pillow) | Manipulacion de imagenes | Python Imaging Library |
| matplotlib | Visualizacion de grillas de imagenes | Python |

---

## 2. PyTorch y torchvision

**torch** es el framework base que provee tensores, autograd y soporte GPU. **torchvision.models** ofrece una coleccion de arquitecturas pre-entrenadas listas para usar.

El parametro `weights="IMAGENET1K_V1"` descarga automaticamente los pesos entrenados en ImageNet (1000 clases, ~1.2 millones de imagenes). El metodo `.to(device)` mueve el modelo a GPU si esta disponible, y `.eval()` lo pone en modo evaluacion.

```python
import torch
import torchvision.models as models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

alexnet = models.alexnet(weights="IMAGENET1K_V1").to(device).eval()
vgg19 = models.vgg19(weights="IMAGENET1K_V1").to(device).eval()
googlenet = models.googlenet(weights="IMAGENET1K_V1").to(device).eval()
resnet50 = models.resnet50(weights="IMAGENET1K_V1").to(device).eval()
```

{{< concept-alert type="clave" >}}
Batch Normalization usa estadisticas del batch en modo entrenamiento. Con una sola imagen de optimizacion, esas estadisticas no tienen sentido. `.eval()` fuerza a usar estadisticas globales pre-calculadas. Sin `.eval()`, las visualizaciones seran incorrectas.
{{< /concept-alert >}}

---

## 3. torch-lucent

**torch-lucent** es el port a PyTorch de **Lucid**, la libreria de Feature Visualization desarrollada por Distill/Google Brain (originalmente en TensorFlow). Implementa el algoritmo de **optimizacion de input** con regularizacion: dado un modelo congelado, genera una imagen sintetica que maximiza la activacion de una neurona o canal objetivo.

```python
from lucent.optvis import render, param, transform, objectives
from lucent.modelzoo.util import get_model_layers
```

### render

La funcion principal es `render_vis()`. Recibe un modelo y un objetivo, y genera la imagen que maximiza la activacion especificada. Internamente ejecuta gradient ascent sobre los pixeles de entrada.

### param

Controla la **parametrizacion del input**. En lugar de optimizar directamente en el espacio de pixeles, lucent trabaja en un **decorrelated space** (espacio de frecuencias). Esto produce imagenes mas naturales y con menos ruido de alta frecuencia.

### transform

Implementa **transformation robustness**: durante la optimizacion, aplica jitter, rotacion y escala aleatorios a la imagen. Esto fuerza a que la feature visualizada sea robusta a transformaciones geometricas, evitando patrones que solo funcionan en una posicion exacta.

### objectives

Define los objetivos de optimizacion. Puede apuntar a un canal especifico de una capa, a una capa completa, o a un label de clasificacion. Permite componer objetivos con operadores aritmeticos.

### get_model_layers

Funcion utilitaria que lista todos los nombres de capas accesibles de un modelo PyTorch. Util para explorar la arquitectura y elegir que capas visualizar.

---

## 4. torchray

**torchray** es una libreria de Facebook Research que implementa multiples metodos de **attribution**. En este laboratorio usamos **Extremal Perturbation**, que encuentra la mascara minima sobre la imagen que preserva la prediccion del modelo.

> Paper: Fong et al. (2019). *Understanding Deep Networks via Extremal Perturbations and Smooth Masks.*

```python
from torchray.attribution.extremal_perturbation import extremal_perturbation
from torchray.benchmark import get_example_data, plot_example
```

- `extremal_perturbation`: funcion principal que calcula la mascara optima. Resuelve un problema de optimizacion para encontrar la region mas pequena de la imagen que mantiene la clasificacion original.
- `get_example_data`: funcion de conveniencia que retorna un modelo, una imagen de ejemplo y 2 categorias para experimentar rapidamente.
- `plot_example`: despliega la imagen original con la mascara superpuesta, mostrando que region fue relevante para la decision.

---

## 5. PIL y matplotlib

**PIL (Pillow)** se usa para abrir y manipular imagenes. En Google Colab, a veces las imagenes PIL no se muestran correctamente en las celdas. La siguiente linea fuerza el uso del viewer de IPython:

```python
from PIL import ImageShow
ImageShow._viewers = [ImageShow.IPythonViewer()]
```

Esta configuracion asegura que al llamar `.show()` o al evaluar un objeto PIL en una celda, la imagen se renderice dentro del notebook en lugar de intentar abrir un visor externo.

**matplotlib** se utiliza exclusivamente para construir grillas de visualizacion. Combinamos `plt.figure`, `gridspec` e `imshow` para organizar multiples feature visualizations en una sola figura comparativa.

---

## 6. Normalizacion de ImageNet

Todos los modelos pre-entrenados en ImageNet esperan que las imagenes de entrada esten **normalizadas**: se resta la media y se divide por la desviacion estandar del dataset de entrenamiento. Los valores estandar son:

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

**lucent** aplica esta normalizacion por defecto cuando el parametro `preprocess=True` (valor default). Esto significa que internamente transforma la imagen antes de pasarla por el modelo. Si el modelo NO fue entrenado con datos normalizados de ImageNet, debes usar `preprocess=False`.

Para visualizar imagenes que ya fueron normalizadas, necesitamos revertir el proceso:

```python
def desnormalizar(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor
```

{{< concept-alert type="clave" >}}
lucent asume que el modelo fue entrenado con ImageNet. Si entrenaste tu modelo sin normalizar los datos, debes usar `preprocess=False` en todas las funciones de lucent. Olvidar esto produce visualizaciones sin sentido.
{{< /concept-alert >}}
