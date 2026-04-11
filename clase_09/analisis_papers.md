# Análisis de Papers — Clase 9: Deep Network Architectures & Interpretability
**Diplomado IA UC | Profesor: Miguel Fadic**

> Análisis detallado de los 6 papers referenciados en la clase, con contexto histórico, contribuciones técnicas, metodología experimental y lecciones aplicables.

---

## Índice de Papers

| # | Paper | Año | Venue | Impacto |
|---|-------|-----|-------|---------|
| [1](#paper-1-vggnet) | Very Deep Convolutional Networks for Large-Scale Image Recognition | 2014 | ICLR 2015 | ~100,000 citas |
| [2](#paper-2-googlenet--inception) | Going Deeper with Convolutions | 2014 | CVPR 2014 | ~40,000 citas |
| [3](#paper-3-resnet) | Deep Residual Learning for Image Recognition | 2015 | CVPR 2016 | ~230,000 citas |
| [4](#paper-4-análisis-comparativo-de-dnns) | An Analysis of Deep Neural Network Models for Practical Applications | 2016 | arXiv | ~3,000 citas |
| [5](#paper-5-feature-visualization) | Feature Visualization | 2017 | Distill | Influyente en XAI |
| [6](#paper-6-extremal-perturbations) | Understanding Deep Networks via Extremal Perturbations and Smooth Masks | 2019 | ICCV 2019 | Oral presentation |

---

## Paper 1: VGGNet

### Ficha técnica

| Campo | Detalle |
|-------|---------|
| **Título** | Very Deep Convolutional Networks for Large-Scale Image Recognition |
| **Autores** | Karen Simonyan, Andrew Zisserman (University of Oxford, VGG) |
| **Publicado** | arXiv:1409.1556 — Septiembre 2014; ICLR 2015 |
| **Dataset** | ImageNet ILSVRC 2012-2014 |
| **Resultado principal** | 2° lugar ILSVRC 2014 clasificación (7.3% top-5), 1° lugar localización |

### Contexto y Motivación

En 2014, el estado del arte en ImageNet era AlexNet (2012) y sus variaciones. Simonyan y Zisserman del Visual Geometry Group de Oxford se preguntaron: **¿cuál es el efecto de la profundidad de la red en la accuracy?**

Su hipótesis era simple pero elegante: si mantienes **todo lo demás constante** (solo 3×3 filters) y solo varía la profundidad, puedes medir el efecto puro de la profundidad. Esto era ciencia controlada aplicada al diseño de redes.

> "Our main contribution is a thorough evaluation of networks of increasing depth using an architecture with very small (3×3) convolution filters"

### Contribuciones principales

#### 1. Diseño sistemático con filtros 3×3

La decisión de usar **exclusivamente filtros 3×3** con stride 1 no es arbitraria. El paper argumenta formalmente:

- **Campo receptivo equivalente con menos parámetros:**
  - 1 capa 7×7 = 49C² parámetros
  - 3 capas 3×3 = 27C² parámetros → **reducción del 44%**
  - 1 capa 5×5 = 25C² parámetros
  - 2 capas 3×3 = 18C² parámetros → **reducción del 28%**

- **Regularización implícita:** Al descomponer una capa grande en varias pequeñas, se añaden no-linealidades intermedias que actúan como regularizadores.

#### 2. Cinco configuraciones evaluadas sistemáticamente

| Config | Peso layers | Conv layers | Parámetros | Top-1 Error | Top-5 Error |
|--------|------------|-------------|-----------|------------|------------|
| A | 11 | 8 | 133M | 29.6% | 10.4% |
| B | 13 | 10 | 134M | 28.7% | 9.9% |
| C | 16 | 13 | 138M | 28.1% | 9.4% |
| D (VGG-16) | 16 | 13 | 138M | **27.0%** | **8.8%** |
| E (VGG-19) | 19 | 16 | 144M | 27.3% | 9.0% |

**Observación notable:** VGG-19 (config E) no es mejor que VGG-16 (config D) en top-5. Esto sugiere que más profundidad tiene rendimientos decrecientes con este diseño.

#### 3. Multi-scale training y testing

El paper introduce el **scale jittering**: durante el entrenamiento, las imágenes se redimensionan a una escala aleatoria `S ∈ [256, 512]` antes de recortar. Esto hace el modelo más robusto a objetos de distintos tamaños.

- **Entrenamiento a escala fija (S=256):** Baseline
- **Entrenamiento con jittering (S∈[256,512]):** Mejora significativa la accuracy

En test, se evalúa con múltiples escalas `Q = {256, 384, 512}` y se promedian las predicciones.

### Metodología de entrenamiento

```
Optimizador: SGD con momentum 0.9
Batch size: 256
Weight decay (L2): 5×10⁻⁴
Dropout en FC: 0.5
Learning rate inicial: 10⁻²
  → Divide por 10 tres veces durante el entrenamiento
Épocas totales: 74 (370K iteraciones)
Inicialización: Config A entrenada con init aleatoria;
                las más profundas se inicializan con los primeros 4 
                conv layers y últimas 3 FC layers de A
```

**¿Por qué esta inicialización en cascada?** En 2014, Xavier/He init no estaban completamente establecidas. Para redes muy profundas, la inicialización aleatoria pura causaba que el gradiente no fluyera bien desde el inicio. Usar pesos pre-entrenados de redes más pequeñas como punto de partida fue la solución pragmática del paper.

### Evaluación y resultados

#### Comparación con el estado del arte (ILSVRC 2014)

| Método | Redes | Top-5 Test Error |
|--------|-------|-----------------|
| **VGG (ensemble 2)** | 2 | **6.8%** |
| GoogLeNet (ensemble 7) | 7 | 6.7% |
| Zeiler & Fergus | 6 | 14.8% |
| OverFeat | 7 | 13.6% |
| AlexNet (Krizhevsky) | 5 | 16.4% |

VGG con solo 2 modelos iguala a GoogLeNet con 7. Esto demuestra la calidad de sus representaciones individuales.

#### Transferibilidad de features

| Dataset | VGG-D | VGG-E | Ensemble |
|---------|-------|-------|----------|
| VOC-2007 (mAP) | 89.3% | 89.3% | 89.7% |
| Caltech-101 | 91.8% | 92.3% | 92.7% |
| Caltech-256 | 85.0% | 85.1% | 86.2% |

Las features aprendidas por VGG en ImageNet transfieren excelentemente a otros datasets de visión. Esto estableció VGG como el backbone estándar de Transfer Learning para años.

### Limitaciones y legado

**Limitaciones identificadas en el paper:**
- 138M parámetros es enorme → lento de entrenar e inferir
- Las capas FC son el 89% de los parámetros (ineficiente)
- Sin mecanismo para controlar el vanishing gradient en entrenamiento profundo

**Legado:**
- Las features VGG son el estándar de facto para perceptual loss en style transfer
- La filosofía 3×3 influyó en todas las arquitecturas posteriores
- El paper introdujo la práctica de reportar ablations sistemáticos (configs A-E)

### ¿Qué aprendemos para el diseño de arquitecturas?

1. **Profundidad ayuda, pero no indefinidamente** — hay un punto de rendimientos decrecientes sin mecanismos especiales.
2. **Filtros pequeños son mejores** — el campo receptivo se construye compositivamente.
3. **La transferibilidad es una propiedad inherente de las capas convolucionales** — si aprendes features genéricas (bordes, texturas), transfieren.
4. **El entrenamiento sistemático y reproducible es tan importante como la arquitectura** — VGG popularizó los ablation studies.

---

## Paper 2: GoogLeNet / Inception

### Ficha técnica

| Campo | Detalle |
|-------|---------|
| **Título** | Going Deeper with Convolutions |
| **Autores** | Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich (Google) |
| **Publicado** | arXiv:1409.4842 — Septiembre 2014; CVPR 2014 |
| **Dataset** | ImageNet ILSVRC 2014 |
| **Resultado principal** | 1° lugar ILSVRC 2014 (6.67% top-5, clasificación y detección) |

### Contexto y Motivación

Mientras Oxford (VGG) apostaba por la profundidad con filtros uniformes, Google apostó por la **eficiencia**. La restricción de diseño fue explícita: el modelo debía funcionar dentro de **1.5 billion multiply-adds** para ser desplegable en aplicaciones reales.

El nombre "Inception" viene de la película (y el meme "we need to go deeper"), pero la motivación técnica es seria: **¿cómo aumentar la profundidad y anchura de una red sin explotar el costo computacional?**

> "The main hallmark of this architecture is the improved utilization of the computing resources inside the network"

### Fundamentos teóricos: Principio Hebbiano y redes esparsas

El paper se fundamenta en el trabajo de Arora et al. sobre redes neuronales esparsas:

- **Teorema de Arora:** Si la distribución de probabilidad de un dataset puede ser representada por una red neuronal esparsa, una arquitectura óptima se puede construir capa a capa correlacionando las activaciones de la última capa.
- **Principio de Hebb:** "Neurons that fire together, wire together" — las neuronas que se activan juntas deberían estar conectadas.

La implicación práctica: en lugar de conectar densamente todas las neuronas, **conectar principalmente aquellas con alta correlación**. El módulo Inception es una aproximación a esta estructura esparsa usando bloques densos (convoluciones) que son computacionalmente eficientes en hardware moderno.

### El Módulo Inception: diseño detallado

#### Versión naïve

```
                      Previous layer (H×W×C)
           ┌────────────┬───────────┬──────────────┬───────────┐
           │            │           │              │           │
         1×1 conv    3×3 conv    5×5 conv    3×3 MaxPool
         (64 filters) (128 filters)(32 filters)  (→ C filters)
           │            │           │              │
           └────────────┴───────────┴──────────────┘
                    Filter Concatenation
                   (H×W×(64+128+32+C))
```

**Problema:** Si C=256 (capa anterior), solo el MaxPool produce 256 canales de salida adicionales. El costo de la 5×5 es 256×5×5×32 = 204,800 pesos **por posición espacial**. Y en cada módulo Inception, el número de canales crece, haciendo la red cada vez más costosa.

#### Versión con reducción de dimensionalidad (Inception module v1)

```
Previous layer (H×W×192)
    │
    ├─── 1×1 conv (64) ──────────────────────────────────────────┐
    │                                                             │
    ├─── 1×1 conv (96) ──→ 3×3 conv (128) ──────────────────────┤
    │                                                             │ concat
    ├─── 1×1 conv (16) ──→ 5×5 conv (32) ───────────────────────┤
    │                                                             │
    └─── 3×3 MaxPool ──→ 1×1 conv (32) ─────────────────────────┘
                                                                  │
                                                        Output: H×W×256
```

**Parámetros comparados (rama 5×5, módulo inception_3a):**

| Configuración | Cálculo | Parámetros |
|--------------|---------|-----------|
| Sin 1×1 (directo) | 192 × 5 × 5 × 32 | **153,600** |
| Con 1×1 (16 canales) | 192×1×1×16 + 16×5×5×32 | **15,872** |
| Reducción | — | **89.7%** |

### La arquitectura GoogLeNet completa

**22 capas con parámetros, ~6.8M parámetros totales:**

```
Capa             Tipo              Output           Params
─────────────────────────────────────────────────────────
conv1            7×7/2 conv        112×112×64       2.7K
pool1            3×3/2 maxpool     56×56×64         —
conv2_reduce     1×1 conv          56×56×64         4K
conv2            3×3 conv          56×56×192         112K
pool2            3×3/2 maxpool     28×28×192         —
inception_3a     módulo inception  28×28×256         159K
inception_3b     módulo inception  28×28×480         380K
pool3            3×3/2 maxpool     14×14×480         —
inception_4a     módulo inception  14×14×512         364K
  ↑ Aux Classifier 1
inception_4b     módulo inception  14×14×512         437K
inception_4c     módulo inception  14×14×512         463K
inception_4d     módulo inception  14×14×528         580K
  ↑ Aux Classifier 2
inception_4e     módulo inception  14×14×832         840K
pool4            3×3/2 maxpool     7×7×832           —
inception_5a     módulo inception  7×7×832           1072K
inception_5b     módulo inception  7×7×1024          1388K
avg_pool         7×7/1 avgpool     1×1×1024          —
dropout(40%)     —                 1×1×1024          —
linear           —                 1×1×1000          1024K
softmax          —                 1×1×1000          —
```

**Total: ~6.8M parámetros** vs 138M de VGG-16 → **20× más eficiente**

### Los clasificadores auxiliares: análisis técnico

**Posición:** Attached a las salidas de `inception_4a` y `inception_4d`.

**Estructura de cada clasificador auxiliar:**
```
5×5 Average Pool (stride 3) → 4×4×512 (o 4×4×528)
1×1 conv (128 filters) → 4×4×128
Flatten → 2048
Fully Connected → 1024
ReLU
Dropout (70%)
Fully Connected → 1000
Softmax
```

**Función durante el entrenamiento:**
```
Loss total = Loss_main + 0.3 × Loss_aux1 + 0.3 × Loss_aux2
```

**Por qué funcionan:** En redes de 22 capas, el gradiente de la loss principal se atenúa exponencialmente hacia las capas tempranas. Los clasificadores auxiliares "inyectan" gradiente directamente en la capa 9 y 12 aproximadamente, asegurando que incluso las capas medias reciban señal útil.

**Nota sobre versiones posteriores:** Inception-v3 (Szegedy et al., 2016) mostró que los clasificadores auxiliares tienen poco efecto durante el entrenamiento excepto cerca de convergencia, y actúan principalmente como regularizadores. En Inception-v4, se eliminaron.

### Entrenamiento y evaluación

**Sistema:** DistBelief (precursor de TensorFlow), múltiples réplicas asíncronas con SGD y momentum 0.9.

**Augmentación:** Recortes aleatorios del 8-100% del área de la imagen, aspect ratios 3/4 a 4/3, distorsiones fotométricas, y distintos métodos de interpolación (una práctica que luego se llamaría "AutoAugment").

**Ensemble en test:** 7 modelos con 144 recortes por imagen → top-5 error de 6.67%.

### Detección: R-CNN con Inception

Para la tarea de detección en ILSVRC 2014:
- Base: R-CNN framework (Region-based CNN de Girshick et al.)
- Backbone: Inception como clasificador de regiones
- Propuestas: Selective search + multi-box proposals (60% de R-CNN proposals, 93% coverage)
- Resultado: **43.9% mAP**, 1° lugar

### ¿Por qué Inception venció a VGG en eficiencia?

| Métrica | VGG-16 | GoogLeNet |
|---------|--------|-----------|
| Parámetros | 138M | 6.8M |
| Operaciones | ~30 G-Ops | ~1.5 G-Ops |
| Top-5 error (1 modelo) | 7.3% | 6.67% |
| Top-5 error (ensemble) | 6.8% (2 modelos) | 6.67% (7 modelos) |

GoogLeNet con 7 modelos necesita **menos cómputo total** que VGG con 2 modelos.

### Legado e influencia

El paper de Inception desencadenó toda una familia de arquitecturas:
- **Inception-v2/v3 (2016):** Factorización de convoluciones (5×5 → dos 3×3; 3×3 → 1×3 + 3×1)
- **Inception-v4 (2017):** Inception + ResNet connections
- **Xception (2017):** Depthwise separable convolutions (llevó la idea 1×1 al extremo)
- **MobileNet:** Depthwise separable = máxima eficiencia

### Lecciones aplicables

1. **La restricción computacional es un principio de diseño válido**, no un inconveniente. GoogLeNet fue diseñado con un presupuesto de operaciones explícito.
2. **La modularidad facilita el scaling.** El módulo Inception se puede apilar y ajustar sin rediseñar la red completa.
3. **1×1 convolutions son fundamentales** para controlar la anchura de la red en cualquier punto.
4. **El training instability en redes profundas** requiere intervenciones especiales (entonces: aux classifiers; ahora: BN + residuales).

---

## Paper 3: ResNet

### Ficha técnica

| Campo | Detalle |
|-------|---------|
| **Título** | Deep Residual Learning for Image Recognition |
| **Autores** | Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft Research) |
| **Publicado** | arXiv:1512.03385 — Diciembre 2015; CVPR 2016 |
| **Dataset** | ImageNet ILSVRC 2015, CIFAR-10, COCO |
| **Resultado principal** | 1° lugar ILSVRC 2015 (3.57% top-5), superando el error humano (~5%) |
| **Premio** | Best Paper Award CVPR 2016 |

### El Problema Identificado: Degradación

He et al. observaron experimentalmente lo siguiente en CIFAR-10:

```
Red "plain" (sin residuales):
  20 capas: training error ≈ 7.0%, test error ≈ 8.0%
  56 capas: training error ≈ 9.0%, test error ≈ 10.0%
```

La red de 56 capas es **peor que la de 20 capas en entrenamiento**. Esto es crucial: no es overfitting (el test error sería peor pero el train error sería mejor). Es un **problema de optimización**.

**¿Por qué pasa esto?**

La hipótesis del paper: no todos los mappings son igualmente fáciles de optimizar. Una red de N+1 capas podría, en principio, resolver el mismo problema que una de N capas si la capa extra aprendiera la identidad `f(x) = x`. Pero en la práctica, los optimizadores modernos no pueden encontrar fácilmente la función identidad a través de composiciones de capas con activaciones no-lineales.

### La Solución: Residual Learning

#### Formulación matemática

Sea `H(x)` el mapeo deseado de un bloque. En lugar de aprender `H(x)` directamente, las capas aprenden:

```
F(x) := H(x) - x
```

Por lo tanto, el bloque computa:

```
H(x) = F(x) + x
```

La shortcut connection suma la entrada `x` directamente a la salida del bloque.

#### ¿Por qué funciona teóricamente?

Si el mapeo óptimo es la identidad (la capa extra no debería hacer nada), entonces:
- **Sin residual:** Aprender `H(x) = x` requiere que los pesos configuren una transformación identidad no-trivial.
- **Con residual:** Aprender `F(x) = 0` es trivial. Si los pesos se inicializan cerca de cero (lo cual es estándar), `F(x) ≈ 0` desde el principio, y la red puede mantener ese comportamiento fácilmente.

> "We hypothesize that it is easier to optimize the residual mapping than to optimize the original, unreferenced mapping."

**Evidencia experimental:** El paper muestra que en ResNets entrenadas, las respuestas de las capas residuales son "generally smaller than their plain counterparts", confirmando que las capas aprenden perturbaciones pequeñas alrededor de la identidad.

### Diseño del bloque residual

#### Bloque básico (ResNet-18, 34)

```python
# Forward pass del bloque básico
def forward(x):
    residual = x                          # shortcut
    
    out = conv3x3(x, planes)             # Conv 3×3
    out = batch_norm(out)                 # BN
    out = relu(out)                       # ReLU
    
    out = conv3x3(out, planes)           # Conv 3×3
    out = batch_norm(out)                 # BN
    
    if downsample:
        residual = conv1x1(x, planes, stride=2)  # proyección
        residual = batch_norm(residual)
    
    out = out + residual                 # suma residual
    out = relu(out)                      # ReLU final
    return out
```

**Detalles importantes:**
- El ReLU **después** de la suma (no dentro de cada rama)
- BN después de cada convolución, antes de ReLU
- La shortcut usa una convolución 1×1 solo cuando cambian las dimensiones (stride 2 o canales distintos)

#### Bloque bottleneck (ResNet-50, 101, 152)

Para redes muy profundas, el costo de 2 capas 3×3 es demasiado alto. Se usa un diseño de 3 capas:

```
Entrada (256 canales)
    ↓
  1×1 conv, 64 canales  ← compresión (ratio 4:1)
  BN + ReLU
    ↓
  3×3 conv, 64 canales  ← operación espacial (costo reducido)
  BN + ReLU
    ↓
  1×1 conv, 256 canales ← expansión (restaura dimensionalidad)
  BN
    ↓
  + shortcut (256 canales)
  ReLU
```

**Comparación de costos:**

| Bloque | Params (64→64) | FLOPs (por posición) |
|--------|---------------|---------------------|
| Básico (3×3, 3×3) | 2 × 3×3×64×64 = 73,728 | 2 × 9 × 64 × 64 = 73,728 |
| Bottleneck (1×1-3×3-1×1) | 256×64 + 9×64×64 + 64×256 = 69,632 | 256×1 + 64×9 + 64×256 = 17,152 |

El bottleneck es **~4× más eficiente en operaciones** con similar número de parámetros.

### Opciones de shortcut connections

El paper evalúa 3 variantes:

| Opción | Descripción | Parámetros extra | Top-1 Error |
|--------|-------------|-----------------|------------|
| A | Zero-padding para más canales | 0 | 27.94% |
| B | Proyección 1×1 solo cuando cambian dims | Pocos | 27.88% |
| C | Proyección 1×1 siempre | Muchos | 27.58% |

> "Option B is slightly better than A. We conjecture that this is because the zero-padded dimensions in A indeed have no residual learning."

La diferencia es marginal — lo importante es que **todas superan ampliamente las redes plain**, validando que las shortcut connections son el factor clave.

### Configuraciones de arquitectura

| Capa | Output | ResNet-18 | ResNet-34 | ResNet-50 | ResNet-101 | ResNet-152 |
|------|--------|-----------|-----------|-----------|-----------|-----------|
| conv1 | 112×112 | 7×7, 64, s2 | → | → | → | → |
| pool | 56×56 | 3×3 max, s2 | → | → | → | → |
| conv2_x | 56×56 | [3×3,64]×2 | [3×3,64]×3 | [1×1,64; 3×3,64; 1×1,256]×3 | →×3 | →×3 |
| conv3_x | 28×28 | [3×3,128]×2 | ×4 | [1×1,128; 3×3,128; 1×1,512]×4 | →×4 | →×8 |
| conv4_x | 14×14 | [3×3,256]×2 | ×6 | [1×1,256; 3×3,256; 1×1,1024]×6 | →×23 | →×36 |
| conv5_x | 7×7 | [3×3,512]×2 | ×3 | [1×1,512; 3×3,512; 1×1,2048]×3 | →×3 | →×3 |
| — | 1×1 | avg pool, 1000-d fc, softmax | → | → | → | → |
| FLOPs | — | 1.8B | 3.6B | 3.8B | 7.6B | 11.3B |

### Experimentos en CIFAR-10

El paper reporta resultados desde redes de 20 hasta **1202 capas** en CIFAR-10:

| Arquitectura | Capas | Error CIFAR-10 |
|-------------|-------|--------------|
| Plain | 20 | 8.75% |
| Plain | 56 | 11.35% (¡peor!) |
| ResNet | 20 | 8.75% |
| ResNet | 56 | 7.52% |
| ResNet | 110 | **6.43%** |
| ResNet | 1202 | 7.93% (overfitting leve) |

La red de 1202 capas **converge sin problemas de optimización**, lo que habría sido imposible con redes plain. El ligero aumento en error (vs 110 capas) se debe a overfitting, no a problemas de gradiente.

> "We argue that there is no optimization difficulty in training a 1000-layer network."

### Resultados en ImageNet

**Single-model performance:**

| Modelo | Top-1 Val Error | Top-5 Val Error |
|--------|----------------|----------------|
| VGG-16 | 28.5% | 9.9% |
| GoogLeNet (1 modelo) | — | 10.07% |
| ResNet-34 | 25.03% | 7.76% |
| ResNet-50 | 22.85% | 6.71% |
| ResNet-101 | 21.75% | 6.05% |
| ResNet-152 | **21.43%** | **5.71%** |

**Ensemble (6 modelos):**
- Top-5 test error: **3.57%**
- Error humano estimado: ~5%
- **ResNet superó al humano en ILSVRC 2015**

### Batch Normalization: su rol en ResNet

BN se aplica después de **cada** convolución, antes de la activación. Su papel en ResNet es triple:

1. **Estabiliza las distribuciones de activaciones** → aprendizaje más rápido
2. **Permite learning rates mayores** → convergencia más rápida
3. **Regularización implícita** → el paper no usa Dropout en las capas convolucionales

**Sin BN, entrenar ResNet-50 sería extremadamente difícil.** La combinación BN + residuales + inicialización He es lo que hace posible redes de 100+ capas.

### Detección y segmentación

En COCO:
- ResNet-101 → **28% mejora relativa en mAP@[0.5:0.95]** vs VGG-16
- 1° lugar en ILSVRC 2015 detection, localization, COCO detection, COCO segmentation

Las skip connections son especialmente valiosas para detección porque **preservan información de alta resolución** que se necesita para localizar objetos.

### Legado e influencia (el paper más citado de la historia del deep learning)

ResNet influenció prácticamente toda arquitectura posterior:
- **ResNeXt:** Residuales con "cardinality" (grupos de convoluciones)
- **DenseNet:** Conecta cada capa con todas las anteriores
- **U-Net:** Skip connections entre encoder y decoder
- **Transformer:** El "Add & Norm" en self-attention es residual
- **EfficientNet:** Residuales en un compound scaling framework

---

## Paper 4: Análisis Comparativo de DNNs

### Ficha técnica

| Campo | Detalle |
|-------|---------|
| **Título** | An Analysis of Deep Neural Network Models for Practical Applications |
| **Autores** | Alfredo Canziani, Adam Paszke, Eugenio Culurciello |
| **Publicado** | arXiv:1605.07678 — Mayo 2016 |
| **Hardware** | NVIDIA Jetson TX1 (GPU embebida) |
| **Contribución** | Análisis empírico de eficiencia en hardware real |

### Motivación

Los papers de arquitecturas compiten en accuracy en ImageNet, pero **¿qué pasa cuando se despliegan en producción?** El accuracy no es el único factor relevante. En sistemas embebidos, robots, vehículos autónomos y móviles, los recursos son limitados.

> "The resource utilisation of winning models has not been properly taken into account"

### Métricas medidas

Los autores midieron **6 dimensiones** en **15 arquitecturas**:

1. **Accuracy** — Top-1 en ImageNet validation (single crop)
2. **Memory footprint** — RAM del sistema durante la inferencia
3. **Parameters** — Número de pesos entrenables
4. **Operations count** — Multiply-adds para un forward pass
5. **Inference time** — Milisegundos por imagen
6. **Power consumption** — Watts medidos en tiempo real

**Modelos analizados:** AlexNet, BN-AlexNet, Network-in-Network, GoogLeNet, VGG-16, VGG-19, ResNet-18/34/50/101/152, Inception-v3/v4, ENet.

### Hallazgos principales

#### 1. El consumo de potencia es constante

```
Potencia promedio: 11.8W ± 0.7W
Independiente de: batch size, arquitectura
```

Esto tiene una implicación profunda: **la potencia no es un factor diferenciador entre arquitecturas** (en hardware dedicado). Lo que diferencia es la energía por predicción (potencia × tiempo).

#### 2. Relación hiperbólica accuracy-tiempo

Existe un **trade-off hiperbólico** entre accuracy y velocidad de inferencia:

```
accuracy ≈ 1 - k / inference_time
```

Para pasar del 60% al 70% de accuracy se necesita X tiempo extra. Para pasar del 70% al 80% se necesita ~10X tiempo más. Los últimos puntos de accuracy son exponencialmente más caros.

#### 3. Operaciones = estimador confiable del tiempo

Existe correlación lineal entre `operations_count` y `inference_time`, lo que permite **estimar velocidad en fase de diseño** sin necesidad de hardware.

#### 4. EfficientNet vs. modelos legacy

| Modelo | Parámetros | Accuracy | Params/Accuracy |
|--------|-----------|----------|----------------|
| AlexNet | 60M | 56.9% | 1.05M/% |
| VGG-19 | 144M | 74.5% | 1.93M/% |
| GoogLeNet | 6.8M | 69.8% | 0.097M/% |
| ResNet-50 | 25.6M | 75.1% | 0.34M/% |
| ENet | 0.4M | 65% | 0.006M/% |

ENet logra "13× más información por parámetro que AlexNet" y "24× respecto a VGG-19".

#### 5. El footprint de memoria crece linealmente con parámetros

```
Memory ≈ 200MB (base fija) + 1.30 × (parámetros en MB)
```

Los primeros 200MB son overhead del sistema/framework. Luego, cada MB de parámetros cuesta ~1.3MB de RAM (los 0.3MB extra son activaciones intermedias).

### Recomendaciones prácticas

| Contexto | Recomendación |
|----------|---------------|
| **Máxima accuracy sin restricciones** | Inception-v4, ResNet-152 |
| **Balance general** | ResNet-50, Inception-v3 |
| **Aplicación en tiempo real (servidor)** | ResNet-18, GoogLeNet |
| **Dispositivo embebido** | ENet, MobileNet, SqueezeNet |
| **Restricción de energía** | Elegir la arquitectura más lenta que cumple accuracy → minimiza energía por predicción |
| **No usar** | VGG (nunca es Pareto-optimal) |

### Limitaciones del paper

1. Solo evalúa en NVIDIA Jetson TX1 (GPU ARM). Los resultados pueden variar en CPUs modernas o GPUs de servidor.
2. No considera modelos post-2016 (EfficientNet, MobileNet-v2, etc.)
3. Mide top-1 single-crop (no el mejor resultado posible de cada modelo)

### Contribución metodológica

El paper popularizó la idea de **análisis de Pareto para arquitecturas CNN**: la frontera de Pareto en el espacio (accuracy, costo) define qué modelos son genuinamente mejores. VGG está **dominado**: hay modelos con más accuracy y menos costo. Esto estableció un estándar de cómo comparar arquitecturas.

---

## Paper 5: Feature Visualization

### Ficha técnica

| Campo | Detalle |
|-------|---------|
| **Título** | Feature Visualization |
| **Autores** | Chris Olah, Alexander Mordvintsev, Ludwig Schubert |
| **Publicado** | Distill.pub — Noviembre 2017 |
| **Código** | TensorFlow Lucid (ahora también PyTorch Lucent) |
| **Tipo** | Artículo interactivo (no PDF tradicional) |

### Contexto

Distill es una publicación científica dedicada a la **comunicación clara** en machine learning. Los artículos son interactivos (código ejecutable, visualizaciones dinámicas) y pasan por peer review. El paper de Feature Visualization es uno de los más influyentes en el campo de XAI (Explainable AI).

### Pregunta central

> "Feature visualization answers questions about what a network — or parts of a network — are looking for by generating examples."

A diferencia de la atribución (que analiza una imagen específica), la Feature Visualization genera **ejemplos sintéticos** que maximizan una activación. Es una herramienta para entender el modelo, no las predicciones.

### Marco conceptual: objectives y optimization

#### Fórmula general

```
x* = argmax_{x} obj(Φ(x)) - R(x)

donde:
  x: imagen (parámetro de optimización)
  Φ: red neuronal (parámetros fijos)
  obj: función objetivo (qué se quiere maximizar)
  R: regularización (para imágenes más naturales)
```

#### Taxonomía de objetivos

| Objetivo | Notación | ¿Qué visualiza? | Caso de uso |
|---------|----------|----------------|-------------|
| Neurona | `layer_n[x,y,z]` | Patrón que activa una posición específica | Análisis fino de neuronas |
| Canal | `layer_n[:,:,z]` | Patrón general del filtro z | Entender un filtro |
| Layer (DeepDream) | `||layer_n||²` | Amplifica patrones presentes | Arte, exploración |
| Class Logits | `pre_softmax[k]` | "Imagen ideal" de la clase k | Entender clasificadores |
| Class Probability | `softmax[k]` | Similar pero con competencia entre clases | Menos recomendado |

**¿Por qué logits > probabilidades?**

```
softmax[k] = exp(z_k) / Σ_j exp(z_j)

Para maximizar softmax[k], el optimizador puede:
  1. Aumentar z_k (✓ deseable)
  2. Disminuir todos los z_j (✗ produce imágenes que suprimen otras clases)

Los logits evitan este problema al optimizar z_k directamente.
```

### El problema de las imágenes adversariales

Sin regularización, la optimización produce **imágenes adversariales**: inputs que activan fuertemente la red pero que los humanos perciben como ruido.

**¿Por qué?** El espacio de entradas que maximizan una neurona es enorme y no restringido. La mayor parte de ese espacio es "ruido estructurado" que explota las no-linealidades de la red sin parecerse a imágenes naturales.

El paper identifica dos fuentes de artefactos:
1. **Checkerboard patterns:** Causados por capas con stride. La backpropagación a través de estas capas crea patrones periódicos en el gradiente.
2. **High-frequency noise:** El espacio natural de imágenes tiene muy poco poder en altas frecuencias; el optimizador ignora esta restricción sin regularización.

### Las tres familias de regularización

#### Familia 1: Penalización de frecuencia

Penaliza explícitamente las altas frecuencias:

| Técnica | Fórmula | Efecto |
|---------|---------|--------|
| L₁ | `λ × Σ|x_i|` | Reduce la magnitud de los píxeles |
| Total Variation | `λ × Σ|x_{i+1}-x_i|` | Suaviza las transiciones entre píxeles |
| Blur | Aplicar Gaussiano cada N pasos | Elimina directamente las frecuencias altas |

#### Familia 2: Transformation Robustness

Antes de evaluar la activación en cada paso, se aplica una transformación estocástica a la imagen:

```python
# Pipeline de transformaciones (Lucid/Lucent)
x_transformed = (
    x                           # imagen actual
    |> pad(12)                  # padding para evitar artefactos en bordes
    |> jitter(8)                # desplazamiento aleatorio ±4px
    |> random_scale([1,0.975,1.025,0.95,1.05])
    |> random_rotate([-5,5,-4,4,-3,3,-2,2,-1,1,0])
    |> jitter(4)                # segundo jitter
    |> crop_or_pad(224)         # recortar a tamaño estándar
)
```

**Intuición:** La imagen resultante debe activar la red **independientemente de la posición exacta de los píxeles**. Esto fuerza a que los patrones sean estructuralmente robustos, no simples ataques adversariales posición-dependientes.

#### Familia 3: Priors aprendidos (más sofisticado)

En lugar de optimizar directamente en el espacio de píxeles, se puede optimizar en el espacio latente de un VAE o GAN. Esto garantiza que las imágenes generadas sean "fotorrealistas".

**Limitación:** Es difícil separar qué aprendió la red de lo que "impone" el prior.

### La innovación técnica principal: preconditioning en espacio decorrelado

Esta es la contribución más técnica del paper. El gradiente en el espacio RGB es ineficiente porque los canales R, G, B están fuertemente correlacionados en imágenes naturales.

**Solución: optimizar en el espacio de Fourier**

```python
# Parametrización en espacio de Fourier
# 1. Inicializar en espacio de Fourier
spectrum = random_complex_tensor(H, W)

# 2. Escalar por 1/f (decorrelación de frecuencias)
spectrum_scaled = spectrum / scale_factor(frequencies)

# 3. Transformar a espacio RGB
image = real(ifft(spectrum_scaled))

# 4. Aplicar decorrelación de color (Cholesky de la cov de colores naturales)
image_decorrelated = image @ color_decorrelation_matrix

# 5. Gradiente se calcula en el espacio de Fourier
```

**¿Por qué funciona?** En el espacio de Fourier, los coeficientes de distintas frecuencias son menos correlacionados. El gradient descent en este espacio distribuye las actualizaciones más uniformemente, produciendo imágenes más naturales.

El paper reporta que esta técnica por sí sola produce imágenes radicalmente mejores que el espacio RGB directo.

### Diversidad: revelando la polisemia neuronal

Un resultado fundamental del paper: las neuronas individuales pueden responder a **múltiples conceptos**, no a uno solo.

**El experimento de diversidad:**

En lugar de buscar un solo input que maximice la activación, se buscan N inputs que:
1. Cada uno maximice la activación
2. Sean mutuamente diferentes (pairwise cosine similarity penalizada)

Término de diversidad:
```
diversity_term = - Σ_{i≠j} cosine_similarity(gram(x_i), gram(x_j))
```

Los resultados muestran que el mismo canal de `mixed4a` en GoogLeNet responde a múltiples patrones visualmente distintos.

**Hipótesis de superposición (*Superposition Hypothesis*, posterior a este paper):**
> Las redes neuronales aprenden muchos más "features" que el número de neuronas disponibles. Para lograrlo, almacenan múltiples features en combinaciones lineales de neuronas. Una neurona puede participar en múltiples features.

### La jerarquía emergente en GoogLeNet

| Capa | Descripción de features encontradas |
|------|-------------------------------------|
| `conv2d0` | Detectores de borde orientados (como filtros Gabor), gradientes de color |
| `conv2d1-2` | Bordes más complejos, esquinas, texturas simples |
| `mixed3a-3b` | Texturas repetitivas (puntos, rayas, cuadrícula) |
| `mixed4a-4b` | Patrones complejos (flores abstractas, redes, escamas) |
| `mixed4c-4d` | Partes de objetos (ojos, ruedas, patas, picos) |
| `mixed4e-5a` | Partes más grandes (caras, partes de vehículos) |
| `mixed5b` | Objetos completos, combinaciones complejas |

Esta jerarquía es **emergente** — nadie la programó. Surge del entrenamiento con ejemplos de ImageNet.

### Implicaciones para entender las CNN

1. **Las CNN aprenden conceptos compositivos** — bordes → texturas → partes → objetos.
2. **Una neurona ≠ un concepto** — la representación es distribuida.
3. **La visualización sin regularización no sirve** — es obligatorio usar preconditioning.
4. **Los dataset examples y la optimización son complementarios** — dataset examples muestran qué activa la neurona en la práctica; la optimización muestra qué está buscando realmente.

### Legado

El paper estableció el campo de **mechanistic interpretability** y fue el precursor de:
- **Circuits (Olah et al., 2020):** Análisis de subgrafos específicos en CNN
- **Polysemanticity (Elhage et al., 2022):** Evidencia formal de que las neuronas son polisémicas
- **Toy Models of Superposition (Elhage et al., 2022):** Modelo matemático de por qué ocurre la polisemia
- **Monosemanticity (Templeton et al., 2023):** Intentos de factorizar redes en neuronas monosémicas

---

## Paper 6: Extremal Perturbations

### Ficha técnica

| Campo | Detalle |
|-------|---------|
| **Título** | Understanding Deep Networks via Extremal Perturbations and Smooth Masks |
| **Autores** | Ruth Fong, Mandela Patrick, Andrea Vedaldi (Oxford + Facebook AI Research) |
| **Publicado** | arXiv:1910.08485 — Octubre 2019; **ICCV 2019 (oral)** |
| **Código** | TorchRay library (PyTorch) |
| **Distinción** | Oral presentation en ICCV → top ~3% de los papers aceptados |

### Motivación y contexto

Existía una proliferación de métodos de atribución (Grad-CAM, RISE, SmoothGrad, Integrated Gradients, etc.), pero ninguno tenía un fundamento teórico sólido ni eliminaba los hiperparámetros de optimización.

> "An important family of attribution methods is based on measuring the effect of perturbations applied to the input."

El paper propone un método de atribución **basado en perturbaciones** con fundamentos matemáticos sólidos y sin hiperparámetros de regularización a tunear.

### El Problema de Atribución: formulación formal

Dado un modelo `Φ: X → ℝ` y una imagen `x₀`, se busca identificar qué región de `x₀` es responsable de la predicción `Φ(x₀)[c]` para la clase `c`.

**Definición de perturbación extremal:**

Una perturbación extremal de área `a` es la máscara `m_a ∈ [0,1]^{H×W}` que:

```
m_a* = argmax_{m: area(m)=a}  Φ(m ⊗ x₀)[c]
```

Donde `m ⊗ x₀` es la imagen perturbada: la máscara revela parte de `x₀` y oculta el resto con una versión difuminada.

### La restricción de área: softrank constraint

La restricción `area(m) = a` es no-trivial de implementar en optimización continua. La solución es usar un **ranking-based loss**:

1. Vectorizar `m` en `vecsort(m)` (ordenar de menor a mayor)
2. Comparar con el vector objetivo `r_a` (cero hasta `(1-a)×HW`, uno después)
3. Minimizar: `R_a(m) = ||vecsort(m) - r_a||²`

Esta función actúa como un **logarithmic barrier**: penaliza fuertemente desviarse del área objetivo sin hacer la optimización inestable. Permite usar `λ` muy grande (restricción estricta) sin explotar el gradiente.

### La parametrización de máscara suave

En lugar de optimizar directamente sobre `m ∈ [0,1]^{H×W}`, se usa una parametrización de **dos pasos**:

**Paso 1:** Parámetros a baja resolución `m̄ ∈ [0,1]^{h×w}` (más pequeño que `H×W`)

**Paso 2:** Aplicar max-convolution para obtener `m`:

```
m(u) = max_{v ∈ Ω} k(u-v) · m̄(v)
```

Donde `k` es un kernel que determina la suavidad. En la práctica se usa **smooth-max (smax)** en lugar de max para permitir backpropagation:

```
smax_T(a, b) = (a·exp(a/T) + b·exp(b/T)) / (exp(a/T) + exp(b/T))
```

**Propiedades garantizadas:**
- La máscara `m` es **Lipschitz-continua** con constante 1 → suavidad garantizada
- Los valores de `m` son casi binarios (cerca de 0 o 1) → máscaras interpretables
- La suavidad es controlable con el kernel `k`

### El operador de perturbación: pirámide de blur

La perturbación que "oculta" las regiones con `m(u) ≈ 0` usa Gaussian blur:

```
(m ⊗ x)(u) = π_g(x; u, σ_max · (1 - m(u)))
```

Donde `σ_max · (1 - m(u))` es el nivel de blur en posición `u`. Cuando `m(u) = 1` (revelado), no hay blur. Cuando `m(u) = 0` (ocultado), máximo blur.

**¿Por qué blur y no negro?**

Si se usa negro (zero-padding), se le pasa al modelo un input que **nunca apareció durante el entrenamiento** (imágenes con parches negros artificiales). El modelo puede responder de forma arbitraria a inputs fuera de la distribución de entrenamiento.

Con blur, la región oculta contiene información plausible pero no informativa. Se mantiene dentro de la distribución de entrenamiento.

**Pirámide de blur:** Para eficiencia, se precomputan versiones de `x₀` con distintos niveles de blur `σ_1 < σ_2 < ... < σ_K = σ_max` y se interpola entre ellos según el valor de `m`.

### La función de recompensa contrastiva

El paper introduce `contrastive_reward` para hacer las máscaras más **discriminativas**:

```python
def contrastive_reward(output, class_idx):
    # En lugar de solo maximizar P(class_idx),
    # maximizar la diferencia con la siguiente clase más probable
    return output[class_idx] - max(output[j] for j != class_idx)
```

Esto produce máscaras que no solo activan la clase objetivo sino que **la distinguen de otras clases**, lo que es más útil para entender por qué el modelo predice `c` y no `c'`.

### Evaluación experimental

#### Propiedad de monotocidad

El paper verifica experimentalmente que la función de área es monotónica:

```
Si a₁ < a₂, entonces Φ(m_{a₁} ⊗ x₀) ≤ Φ(m_{a₂} ⊗ x₀)
```

**Resultado:** 98.45% de imágenes en ImageNet validation exhiben monotocidad estricta. Esto valida que el concepto de "perturbación extremal" tiene sentido (revelar más región siempre es mejor o igual que revelar menos).

#### Pointing Game Benchmark

El "pointing game" evalúa si el máximo de la máscara de atribución cae dentro del bounding box del objeto anotado.

| Método | VOC07 (All) | VOC07 (Diff) | COCO14 (All) | COCO14 (Diff) |
|--------|------------|-------------|-------------|--------------|
| Gradient | 76.3% | 56.9% | 37.7% | 31.4% |
| GBP (Guided BP) | 80.7% | 65.1% | 54.6% | 50.7% |
| Grad-CAM | 86.6% | 74.0% | 54.2% | 49.0% |
| RISE | 74.1% | 49.2% | 41.5% | 36.3% |
| **Extremal Pert.** | **88.0%** | **76.1%** | **51.5%** | **45.9%** |

Extremal Perturbation obtiene **el mejor resultado en VOC07** y es competitivo en COCO.

#### Sanity Check: sensibilidad a parámetros

Los autores realizan un "sanity check" importante: si se **aleatoriza los pesos** de la red, las visualizaciones deberían cambiar radicalmente. Los métodos de backpropagation (Guided BP, GBP) fallan este test — sus visualizaciones apenas cambian con pesos aleatorios, lo que sugiere que detectan estructura del input más que del modelo. Extremal Perturbation pasa el test correctamente.

### Atribución de canales intermedios

Además de atribución en el input, el paper extiende el método a **capas intermedias**:

```
m*_channel = argmax_{m: area(m)=a}  Φ(m ⊗ Φ_l(x₀))[c]
```

Donde `Φ_l(x₀)` son las activaciones de la capa `l`. La máscara `m` ahora selecciona **canales** en lugar de posiciones espaciales.

**Resultado:** Solo ~25 canales de los disponibles son suficientes para mantener la clasificación original. Esto revela que la representación de la red es esparcida — pocos canales por capa son realmente discriminativos para cada clase.

Combinando la atribución de canales con **feature inversion** (reconstruir la imagen a partir de activaciones), se pueden visualizar qué patrones específicos de cada canal son relevantes para la predicción.

### Comparación cualitativa de métodos

Para una imagen con "chocolate sauce":

| Método | Descripción del mapa | Calidad |
|--------|---------------------|---------|
| Gradient | Muy disperso, ruidoso, difícil de interpretar | Baja |
| Guided BP | Más nítido pero aún ruidoso | Media-baja |
| Contrast Excitation | Región central difusa | Media |
| Grad-CAM | Mapa grueso (7×7), semánticamente correcto | Media |
| Occlusion | Mapa correcto pero discontinuo | Media |
| **Extremal Pert.** | **Región compacta, precisa, sin ruido** | **Alta** |

### Legado e influencia

El paper estableció el **estándar para métodos de atribución basados en perturbaciones** con justificación teórica. Influyó en:
- RISE v2 y variantes
- Investigación posterior en atribución causal
- La biblioteca TorchRay se usa extensamente en investigación de XAI
- El concepto de "monotonic attribution" como propiedad deseable

---

## Síntesis Comparativa: Los 6 Papers

### Evolución del pensamiento en CNN research (2014-2019)

```
2014: ¿Cuántas capas y de qué tamaño?
       VGG: Profundidad + filtros 3×3
       Inception: Parallelismo + eficiencia

2015: ¿Cómo entrenar redes MUY profundas?
       ResNet: Residual learning → redefinir el problema

2016: ¿Qué tan costosas son en producción?
       Canziani et al.: Análisis empírico de eficiencia real

2017: ¿Qué está aprendiendo la red?
       Olah et al.: Feature Visualization → gradient ascent con regularización

2019: ¿Qué parte de esta imagen causó esta predicción?
       Fong et al.: Extremal Perturbation → atribución como optimización
```

### Comparativa técnica de contribuciones

| Paper | Problema resuelto | Técnica clave | Impacto práctico |
|-------|-----------------|---------------|-----------------|
| VGG | Profundidad sin eficiencia | Filtros 3×3 uniformes | Transfer learning estándar durante años |
| GoogLeNet | Profundidad con eficiencia | Módulo Inception + 1×1 conv | Arquitecturas móviles, detección eficiente |
| ResNet | Degradación con profundidad | Skip connections + BN | Backbone universal para todo tipo de tareas |
| Canziani et al. | Selección práctica de arquitectura | Análisis multi-métrica en hardware real | Diseño de sistemas AI en producción |
| Feature Viz. | Opacidad de la red | Gradient ascent con preconditioning | Debugging de modelos, mechanistic interpretability |
| Extremal Pert. | Atribución confiable | Optimización con restricción de área | Auditoría de predicciones, bias detection |

### Las tres "eras" de las CNN que estos papers cubren

**Era 1: Scaling (VGG, Inception)**
La pregunta era "¿cómo hacer redes más grandes y profundas?" Las respuestas fueron dos filosofías distintas que hoy coexisten.

**Era 2: Trainability (ResNet)**
La pregunta cambió a "¿cómo entrenar redes tan profundas que sería imposible sin trucos especiales?" La respuesta reformuló el problema de aprendizaje mismo.

**Era 3: Understanding (Feature Viz., Attribution)**
Con redes que funcionan mejor que los humanos en algunos benchmarks, la pregunta pasó a "¿podemos confiar en ellas? ¿Qué aprendieron?" Estas técnicas son la respuesta.

---

## Guía de Lectura Recomendada

Para profundizar más allá de estos 6 papers:

### Sobre arquitecturas
1. **He et al. (2016)** — Identity Mappings in Deep Residual Networks (ResNet v2)
2. **Huang et al. (2017)** — Densely Connected Convolutional Networks (DenseNet)
3. **Tan & Le (2019)** — EfficientNet: Rethinking Model Scaling

### Sobre interpretabilidad
1. **Lundberg & Lee (2017)** — A Unified Approach to Interpreting Model Predictions (SHAP)
2. **Sundararajan et al. (2017)** — Axiomatic Attribution for Deep Networks (Integrated Gradients)
3. **Olah et al. (2020)** — Zoom In: An Introduction to Circuits (mechanistic interp.)

### Sobre transferibilidad de features
1. **Yosinski et al. (2014)** — How transferable are features in deep neural networks?
2. **Kornblith et al. (2019)** — Do Better ImageNet Models Transfer Better?
