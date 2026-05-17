# Análisis: ResNet (He et al., 2015)

> **Cita completa**
> He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Deep Residual Learning for Image Recognition*. CVPR 2016 (publicado en arXiv en diciembre 2015).
>
> arXiv: [1512.03385](https://arxiv.org/abs/1512.03385)
> Citas (2026): >200.000 — el paper más citado de la historia del deep learning.
> Premios: **CVPR 2016 Best Paper Award**, ganador de **ImageNet 2015** (ILSVRC) en clasificación, detección, localización y segmentación, y de **COCO 2015** en detección y segmentación.

PDF local: [resnet_he_2015.pdf](resnet_he_2015.pdf)

---

## 1. Contexto y problema

Antes de ResNet (2015), la sabiduría popular en deep learning era:

> "Redes más profundas = más capacidad = mejor accuracy."

Pero al apilar más de ~20 capas convolucionales aparecía un problema **contraintuitivo**:

- Una red de 56 capas tenía **mayor error de entrenamiento** que una de 20 capas (Figura 1 del paper).
- Esto **no es sobreajuste** (el error de validación también es peor).
- Tampoco es gradiente que se desvanece, gracias a Batch Normalization y a inicializaciones tipo He/Xavier.

El paper lo llama el **degradation problem**: las redes profundas, simplemente, **son más difíciles de optimizar**.

### El argumento por construcción del paper

Si una red de 20 capas obtiene cierto error, **debería ser trivial construir una de 56 capas con el mismo error**: las primeras 20 capas hacen lo mismo, y las 36 capas restantes implementan la identidad ($f(x) = x$). Eso prueba que el espacio de funciones de la red profunda **incluye** al de la superficial.

Sin embargo, **los solvers (SGD + backprop) no logran encontrar esa solución**. La degradación indica que el problema es de **optimización**, no de expresividad.

---

## 2. La contribución central — Residual Learning

### Reformular qué aprende cada bloque

En vez de pedirle a un bloque de capas que aprenda una función arbitraria $\mathcal{H}(x)$, **pedirle que aprenda el residuo** $\mathcal{F}(x) = \mathcal{H}(x) - x$, y luego sumar la entrada por una **conexión de atajo** (shortcut):

$$ y = \mathcal{F}(x, \{W_i\}) + x $$

```
        x
        │
        ├─────────── shortcut (identity) ──┐
        ▼                                  │
   [conv → BN → ReLU]                      │
        │                                  │
        ▼                                  │
   [conv → BN]                             │
        │                                  │
        ▼                                  │
        + ◄────────────────────────────────┘
        │
        ▼
       ReLU
        │
        ▼
        y
```

### Por qué funciona — intuición

1. **Inicialización cercana a la identidad**: si los pesos de las capas internas son pequeños (como en inicializaciones estándar), $\mathcal{F}(x) \approx 0$ y la salida del bloque es $\approx x$. La red profunda parte siendo equivalente a una superficial con capas extra "neutrales".

2. **La identidad es un punto fijo fácil**: si la función óptima es cercana a la identidad, aprender un residuo pequeño es más fácil que aprender una función desde cero. Si la función óptima es lejana, el residuo se aprende normalmente.

3. **Las conexiones de atajo dan caminos directos al gradiente**: durante backprop, el gradiente puede fluir por la rama identidad sin atenuarse, llegando a las capas iniciales sin desvanecerse.

### Conexiones de atajo sin parámetros extra

> *"Identity shortcut connections add neither extra parameter nor computational complexity."*

Esto es clave: el atajo identidad es **gratis**. No añade pesos, no añade FLOPs (solo una suma elemento a elemento). Permite comparar honestamente "red plana" vs "red residual" del mismo tamaño.

Cuando las dimensiones de entrada y salida no coinciden (al cambiar canales o resolución), se usa una **projection shortcut**: una conv 1×1 que proyecta a las dimensiones correctas.

---

## 3. Arquitecturas concretas (Tabla 1 del paper)

| Capa | Output | ResNet-18 | ResNet-34 | ResNet-50 | ResNet-101 | ResNet-152 |
|------|--------|-----------|-----------|-----------|-----------|-----------|
| conv1 | 112×112 | 7×7, 64, /2 | 7×7, 64, /2 | 7×7, 64, /2 | 7×7, 64, /2 | 7×7, 64, /2 |
| | 56×56 | maxpool /2 | maxpool /2 | maxpool /2 | maxpool /2 | maxpool /2 |
| conv2_x | 56×56 | [3×3, 64]×2 ×2 | [3×3, 64]×2 ×3 | [bottleneck]×3 | [bottleneck]×3 | [bottleneck]×3 |
| conv3_x | 28×28 | [3×3, 128]×2 ×2 | [3×3, 128]×2 ×4 | [bottleneck]×4 | [bottleneck]×4 | [bottleneck]×8 |
| conv4_x | 14×14 | [3×3, 256]×2 ×2 | [3×3, 256]×2 ×6 | [bottleneck]×6 | [bottleneck]×23 | [bottleneck]×36 |
| conv5_x | 7×7 | [3×3, 512]×2 ×2 | [3×3, 512]×2 ×3 | [bottleneck]×3 | [bottleneck]×3 | [bottleneck]×3 |
| | 1×1 | avgpool, fc 1000 | avgpool, fc 1000 | avgpool, fc 1000 | avgpool, fc 1000 | avgpool, fc 1000 |
| **FLOPs** | | 1.8×10⁹ | 3.6×10⁹ | 3.8×10⁹ | 7.6×10⁹ | 11.3×10⁹ |

### Bloque "básico" (ResNet-18 y 34)

```
input → [3×3 conv, BN, ReLU] → [3×3 conv, BN] → + input → ReLU → output
```

### Bloque "bottleneck" (ResNet-50, 101, 152)

```
input (256 ch)
   │
   ├─── identity ───────────────────────────┐
   ▼                                         │
[1×1 conv, 256→64]  ← reduce dimensión       │
[BN, ReLU]                                   │
   ▼                                         │
[3×3 conv, 64→64]   ← trabajo costoso        │
[BN, ReLU]              en espacio reducido  │
   ▼                                         │
[1×1 conv, 64→256]  ← expande dimensión      │
[BN]                                         │
   ▼                                         │
   + ◄───────────────────────────────────────┘
   │
   ▼
 ReLU
   │
   ▼
output (256 ch)
```

El bottleneck es **~4× más eficiente** que dos convs 3×3 directas con 256 canales. Permite redes mucho más profundas con costo similar.

ResNet-50 = stem + (3 + 4 + 6 + 3) × 3 convs por bottleneck = **50 capas con peso aprendido**.

---

## 4. Resultados experimentales

### ImageNet 2015 (clasificación)

| Modelo | Capas | Top-1 error (%) | Top-5 error (%) |
|--------|-------|-----------------|-----------------|
| VGG-16 | 16 | 28.07 | 9.33 |
| GoogLeNet (Inception v1) | 22 | — | 9.15 |
| **ResNet-34** | 34 | 21.53 | 5.60 |
| **ResNet-50** | 50 | 20.74 | 5.25 |
| **ResNet-101** | 101 | 19.87 | 4.60 |
| **ResNet-152** | 152 | 19.38 | 4.49 |
| **Ensemble ResNet** | — | — | **3.57** ← ganador ILSVRC 2015 |

### Más allá de ImageNet (Tabla 2)

| Tarea | ResNet | Mejora vs estado del arte previo |
|-------|--------|----------------------------------|
| **ImageNet detection** | 1er lugar 2015 | +8.5% mAP vs 2do lugar |
| **ImageNet localization** | 1er lugar 2015 | +13.4% vs 2do |
| **COCO detection** | 1er lugar 2015 | +6.3% mAP (28% mejora relativa) |
| **COCO segmentation** | 1er lugar 2015 | — |

### Profundidad extrema (CIFAR-10)

ResNet-110 (1.7M parámetros): 6.43% error.
ResNet-1202 (19.4M parámetros): 7.93% error → empieza a sobreajustar por exceso de capacidad, pero **sí entrena**, demostrando que 1200 capas son factibles con conexiones residuales. Esto era impensable antes de 2015.

---

## 5. Impacto e influencia

### En arquitecturas

- **Pre-norm Transformers** (BERT, GPT, ViT, LLaMA): todas las arquitecturas modernas de NLP y visión usan conexiones residuales. Sin ResNet, los transformers profundos no entrenarían.
- **U-Net, DenseNet, Highway Networks**: variantes con distintas formas de combinar features de distintas capas.
- **Wide ResNet, ResNeXt, SE-ResNet**: extensiones que mejoran ResNet manteniendo la idea residual.
- **ConvNeXt** (2022): rediseño moderno de ResNet con ideas de Transformers; sigue usando residuals.

### Como backbone universal

ResNet-50 / ResNet-101 son **los backbones más usados de la historia** para tareas down-stream:

- Detección: Faster R-CNN, Mask R-CNN, RetinaNet, FCOS, Detectron2.
- Segmentación: DeepLab, PSPNet, U-Net++.
- Pose estimation: HRNet, AlphaPose.
- Self-supervised: MoCo, SimCLR, BYOL (todos usan ResNet-50 como backbone canónico).

---

## 6. Conexión con el laboratorio

La pieza `backbone.body` que vimos en el `print(model)`:

```python
(body): IntermediateLayerGetter(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))   ← stem
  (bn1): FrozenBatchNorm2d(64)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1)
  (layer1): Sequential( 3 × Bottleneck )    ← conv2_x del paper
  (layer2): Sequential( 4 × Bottleneck )    ← conv3_x
  (layer3): Sequential( 6 × Bottleneck )    ← conv4_x
  (layer4): Sequential( 3 × Bottleneck )    ← conv5_x
)
```

**Es exactamente la ResNet-50 del paper**, con dos diferencias:

1. **`FrozenBatchNorm2d` en vez de `BatchNorm2d`**: los detectores usan batches pequeños (1-4 imágenes por las grandes resoluciones), y BatchNorm con batches chicos es inestable. Se congelan las estadísticas y los parámetros afines desde el pre-entrenamiento.

2. **No tiene `avgpool` ni `fc`**: el `IntermediateLayerGetter` corta la red antes del clasificador final, exponiendo las 4 salidas C2-C5 para que las consuma la FPN.

Cuando el lab usa `trainable_backbone_layers=3`, congela `stem + layer1` y deja entrenable `layer2 + layer3 + layer4`. Esto refleja la intuición clásica: las capas iniciales aprenden features universales (bordes, texturas) y rara vez necesitan re-entrenarse en un dataset pequeño.

---

## 7. Una nota histórica

He et al. son el equipo que en 2015 dominó **literalmente todas** las competencias de visión:

- Mismos autores: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
- Mismo año (2015): publican **Faster R-CNN** (detector) y **ResNet** (backbone).
- Faster R-CNN + ResNet → ganan ImageNet detection 2015.
- ResNet sin más → gana ImageNet classification 2015.

Es el momento en que Microsoft Research Asia se consolidó como un polo del estado del arte en visión. Kaiming He después se mudó a Facebook AI Research y siguió liderando avances (Mask R-CNN 2017, MoCo 2019, MAE 2022).
