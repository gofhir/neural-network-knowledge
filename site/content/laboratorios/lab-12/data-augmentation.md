---
title: "Data Augmentation"
weight: 10
math: true
---

## Setup del problema — Oxford 102 Flowers

La Actividad I trabaja sobre el dataset **Oxford 102 Flowers**, una colección de 8,189 imágenes de flores etiquetadas en 102 especies. Es un dataset clásico de fine-grained classification: hay variaciones sutiles entre especies (color, forma de pétalo, número de pétalos, simetría) que un clasificador debe capturar.

```text
Imágenes totales:  8,189
Imágenes train:    5,687  (~70%)
Imágenes val:      1,224  (~15%)
Imágenes test:     1,278  (~15%)
```

Con 102 clases y ~80 imágenes promedio por clase, un baseline aleatorio acierta ~0.98%. Cualquier modelo que supere ese piso ya está aprendiendo algo. La dificultad alta del dataset hace que el contraste entre baseline (sin augmentation, sin transfer learning) y finetuning sea muy didáctico.

![Sample del dataset flowers](/laboratorios/lab-12/sample-flower.png)

## Modelo base — ResNet18 from scratch

Antes de aplicar augmentation, se entrena un baseline para comparar. La arquitectura es **ResNet18 inicializada desde cero** (sin transfer learning), capa final reemplazada por `nn.Linear(512, 102)` y entrenada 5 épocas con Adam (`lr=5e-4`, `batch_size=128`) y `CrossEntropyLoss`.

### Resultados modelo base

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 0 (val inicial) | — | — | 5.281 | 1.72% |
| 1 | 3.541 | 16.94% | 3.703 | 14.14% |
| 2 | 2.694 | 30.92% | 3.120 | 22.58% |
| 3 | 2.345 | 37.41% | 2.590 | 32.19% |
| 4 | 2.074 | 43.91% | 2.464 | 34.84% |
| 5 | **1.819** | **49.91%** | **2.166** | **40.55%** |

**Test final:** Loss = 2.171, Acc = **42.58%**.

![Loss base](/laboratorios/lab-12/loss-base.png)
![Accuracy base](/laboratorios/lab-12/acc-base.png)

El val acc inicial (1.72%) es básicamente random sobre 102 clases — confirma que el modelo no tenía conocimiento previo. Después de 5 épocas alcanza ~40% en val, una mejora dramática sobre random pero todavía lejos de un buen clasificador (humanos expertos ~95%).

### Predicción cualitativa (sample 166)

Para sample 166 (`bougainvillea`), el modelo base **no acierta**:

```text
Ground Truth: bougainvillea

Top-5 Predictions (modelo base)
  1. lotus              (p=0.1670)
  2. columbine          (p=0.1366)
  3. purple coneflower  (p=0.0795)
  4. snapdragon         (p=0.0515)
  5. sword lily         (p=0.0464)
```

Las 5 clases predichas comparten paleta cromática (morados/rosados) con bougainvillea — el modelo aprendió **color** pero no estructura morfológica. La suma top-5 (≈ 48%) indica una distribución dispersa: el modelo no está confiado en ninguna predicción.

![Predicción sample 166](/laboratorios/lab-12/pred-sample166.png)

## Catálogo de transformaciones disponibles

`torchvision.transforms` ofrece transformaciones que pueden aplicarse de forma **aleatoria por sample** durante el entrenamiento. Cada época ve versiones ligeramente diferentes de cada imagen, lo que actúa como regularizador y reduce overfitting al obligar al modelo a aprender invarianzas.

### Reflexiones (`RandomHorizontalFlip(p=0.5)`)

Cada imagen tiene 50% de probabilidad de ser reflejada horizontalmente.

![Reflexiones horizontales](/laboratorios/lab-12/aug-flips.png)

Es la augmentación más segura y efectiva en datasets de objetos naturales (animales, flores, autos), porque la reflexión horizontal preserva la semántica: una flor reflejada sigue siendo la misma flor. **No** es válida en datos donde la orientación importa (texto, dígitos, asimetrías biológicas).

### Recortes (`RandomCrop(size=(240, 320), pad_if_needed=True)`)

Cada imagen es recortada en una ventana aleatoria del tamaño indicado.

![Recortes aleatorios](/laboratorios/lab-12/aug-crops.png)

Simula que la flor puede aparecer en cualquier región del frame y a distintas escalas. Es especialmente útil cuando combinada con `Resize` previo: primero se redimensiona la imagen a `256×256` y luego se recorta `224×224` aleatoriamente.

### Rotaciones (`RandomRotation((-30, 30))`)

Cada imagen rota un ángulo aleatorio en el rango especificado.

![Rotaciones](/laboratorios/lab-12/aug-rotations.png)

Genera robustez ante variaciones en la pose del fotógrafo. Las áreas vacías generadas por la rotación se rellenan con negro (o el `fill` que especifiquemos).

### Transformaciones afín (`RandomAffine(30)`)

Combinan rotación, traslación, escalado y cizalla en una sola matriz afín.

![Transformaciones afín](/laboratorios/lab-12/aug-affine.png)

Más expresivas que rotación pura, pero hay que cuidar que los ángulos extremos no destruyan el contenido.

### Composición (`transforms.Compose([...])`)

Las transformaciones se encadenan secuencialmente:

```python
randomized_transforms = transforms.Compose([
    transforms.RandomRotation((-5, 5)),
    transforms.Resize([256, 256]),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
])
```

![Composición randomizada](/laboratorios/lab-12/aug-randomized-compose.png)

La separación entre el pipeline **randomizado** (train) y el **determinista** (val/test) es crítica:

```python
deterministic_transforms = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.CenterCrop(size=(224, 224)),
])
```

![Composición determinista](/laboratorios/lab-12/aug-deterministic-compose.png)

Las transformaciones aleatorias **solo** se aplican en train; val y test usan crops centrales determinísticos para que la métrica sea reproducible.

## Transformaciones custom

PyTorch permite definir transforms propias como cualquier objeto callable (clase con `__call__` o función). Esto es útil para introducir invarianzas específicas del dominio.

### Ejemplo: ruido gaussiano aditivo

```python
class AddGaussianNoise(Module):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, img):
        img = np.array(img)
        noise = np.random.normal(loc=self.mu, scale=self.sigma, size=img.shape)
        noisy = np.clip(img + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)
```

![Imagen original sin transformar](/laboratorios/lab-12/aug-noise-original.png)

Aplicando `AddGaussianNoise(mu=50, sigma=10)`:

![Imagen con ruido gaussiano](/laboratorios/lab-12/aug-noise-applied.png)

Hace al modelo más robusto frente a fotos tomadas con sensores de baja calidad o iluminación deficiente.

## Modelo con augmentation

Se entrena un segundo modelo `aug_model` con la misma arquitectura, hiperparámetros y semilla del baseline, pero usando `randomized_transforms` (rotación ±5°, resize 256, random crop 224, flip horizontal p=0.5) durante el entrenamiento.

### Resultados modelo aug

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 0 (val inicial) | — | — | 5.238 | 0.86% |
| 1 | 3.515 | 16.91% | 3.783 | 15.86% |
| 2 | 2.656 | 31.65% | 2.850 | 26.95% |
| 3 | 2.306 | 37.62% | 2.864 | 25.47% (dip) |
| 4 | 2.044 | 44.46% | 2.195 | 39.14% |
| 5 | **1.833** | **49.46%** | **2.175** | **39.92%** |

**Test final:** Loss = 2.139, Acc = **44.45%**.

![Loss aug](/laboratorios/lab-12/loss-aug.png)
![Accuracy aug](/laboratorios/lab-12/acc-aug.png)

### Comparación base vs aug

| Métrica | Base | Aug | Δ |
|---|---|---|---|
| Train Loss | 1.819 | 1.833 | +0.014 |
| Train Acc | 49.91% | 49.46% | −0.45pp |
| Val Acc | 40.55% | 39.92% | −0.63pp |
| **Test Acc** | **42.58%** | **44.45%** | **+1.87pp** |
| Test Loss | 2.171 | 2.139 | −0.032 |

**Interpretación:**

- En train la augmentation **dificulta** levemente el ajuste (49.91% → 49.46%) porque las imágenes vistas son más variadas. Esto es **deseable** — significa que el modelo no se está aprendiendo de memoria los samples del set de entrenamiento.
- El val tiene un *dip* en epoch 3 (25.47% tras 26.95%) — comportamiento típico con augmentation porque cada época ve imágenes distintas, generando más varianza en métricas intermedias.
- En **test** la augmentation gana **+1.87pp** sobre el baseline — pequeña pero consistente. La ventaja de la augmentation se nota más en los conjuntos de evaluación (val, test) porque ahí es donde la generalización se mide.

### Predicción cualitativa con aug (sample 166)

```text
Ground Truth: bougainvillea

Top-5 Predictions (modelo aug)
  1. lotus              (p=0.2476)  ← más confiado en respuesta INCORRECTA
  2. sweet pea          (p=0.0673)
  3. pink primrose      (p=0.0489)
  4. mexican aster      (p=0.0447)
  5. columbine          (p=0.0442)
```

Hallazgo curioso: el modelo aug **refuerza** el sesgo "lotus" (probabilidad 0.247 vs 0.167 del base). Las transformaciones usadas (`flip`, `crop`, `rotation ±5°`) **preservan el color** de la imagen, así que el modelo siguió aprendiendo features cromáticas — y de hecho las consolidó.

Para romper el sesgo de color habría que añadir transformaciones que alteren la paleta: `ColorJitter`, `Grayscale`, `RandomEqualize`, etc. La lección importante: **la augmentación debe atacar la invarianza que se quiere enseñar**. Si el modelo confunde "morado lotus" con "morado bougainvillea", aplicar más rotaciones no soluciona nada.

## Conclusiones de la sección

1. **La augmentation funciona, pero su magnitud depende del dataset y de las transformaciones elegidas.** En este lab, +1.87pp en test no parece dramático, pero es robusto y consistente.
2. **No toda transformación es buena.** Vertical flip (Ejercicio I) degrada el modelo porque las flores boca abajo no aparecen en val/test.
3. **La augmentation regulariza, no aumenta capacidad.** Si el bottleneck es la arquitectura o el preentrenamiento (como veremos con finetuning), la augmentation por sí sola no sustituye una mejor inicialización.
4. **Train/val/test transforms deben ser distintos.** Train usa pipeline randomizado, val/test usan determinístico (resize + center crop). De lo contrario las métricas no son reproducibles.
