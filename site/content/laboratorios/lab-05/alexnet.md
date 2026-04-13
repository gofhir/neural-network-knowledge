---
title: "AlexNet"
weight: 20
math: true
---

## 1. Contexto historico

En 2012, Alex Krizhevsky, Ilya Sutskever y Geoffrey Hinton presentaron **AlexNet** en el NIPS (ahora NeurIPS), ganando el ImageNet Large Scale Visual Recognition Challenge (ILSVRC) con un margen enorme sobre los metodos tradicionales. Este resultado marco el inicio de la era del deep learning en vision por computador.

### ImageNet y el ILSVRC

**ImageNet** es un dataset masivo de imagenes:

- 15 millones de imagenes
- 22,000 categorias
- Imagenes recolectadas de la web y etiquetadas por humanos (Amazon Mechanical Turk)

El **ILSVRC-2010** (la competencia que motivo AlexNet) usaba un subconjunto:

- 1,000 categorias
- 1.2 millones de imagenes de entrenamiento (~1,000 por categoria)
- 50,000 imagenes de validacion
- 150,000 imagenes de test
- Imagenes RGB escaladas a $256 \times 256$ pixeles

Los objetivos de clasificacion eran:
- **Top-1 error**: acertar la clase correcta en un solo intento
- **Top-5 error**: que la clase correcta este entre las 5 predicciones mas probables

AlexNet redujo el top-5 error del 26% (metodos tradicionales) al 15.3%, una mejora sin precedentes.

---

## 2. Arquitectura capa por capa

AlexNet tiene 5 capas convolucionales seguidas de 3 capas fully connected. A continuacion se detalla cada capa con sus dimensiones.

| Capa | Operacion | Entrada | Salida |
|------|-----------|---------|--------|
| conv1 | Conv2d(3, 96, 11, stride=4, pad=2) + MaxPool(3, stride=2) + ReLU | $3 \times 224 \times 224$ | $96 \times 27 \times 27$ |
| conv2 | Conv2d(96, 256, 5, stride=1, pad=2) + MaxPool(3, stride=2) + ReLU | $96 \times 27 \times 27$ | $256 \times 13 \times 13$ |
| conv3 | Conv2d(256, 384, 3, stride=1, pad=1) + ReLU | $256 \times 13 \times 13$ | $384 \times 13 \times 13$ |
| conv4 | Conv2d(384, 384, 3, stride=1, pad=1) + ReLU | $384 \times 13 \times 13$ | $384 \times 13 \times 13$ |
| conv5 | Conv2d(384, 256, 3, stride=1, pad=1) + MaxPool(3, stride=2) + ReLU | $384 \times 13 \times 13$ | $256 \times 6 \times 6$ |
| flat | Flatten | $256 \times 6 \times 6$ | $9216$ |
| fc6 | Linear(9216, 4096) + ReLU | $9216$ | $4096$ |
| fc7 | Linear(4096, 4096) + ReLU | $4096$ | $4096$ |
| fc8 | Linear(4096, 1000) | $4096$ | $1000$ |

### Verificacion de dimensiones (conv1)

Aplicando la formula de salida a la primera capa:

Convolucion: $O = \left\lfloor\frac{224 - 11 + 2 \cdot 2}{4}\right\rfloor + 1 = 55$

MaxPool: $O = \left\lfloor\frac{55 - 3}{2}\right\rfloor + 1 = 27$

La salida de conv1 es $96 \times 27 \times 27$.

---

## 3. Innovaciones de AlexNet

### ReLU (Rectified Linear Unit)

Antes de AlexNet, las funciones de activacion dominantes eran **sigmoid** y **tanh**. AlexNet popularizo el uso de **ReLU**:

$$\text{ReLU}(x) = \max(0, x)$$

Ventajas de ReLU sobre sigmoid/tanh:

- **No hay saturacion** para valores positivos — los gradientes no se desvanecen
- **Calculo eficiente** — solo una comparacion, sin exponenciales
- **Convergencia mas rapida** — el entrenamiento con ReLU es hasta 6 veces mas rapido segun el paper original

### Dropout

AlexNet introdujo **Dropout** como tecnica de regularizacion en las capas fully connected. Durante el entrenamiento, cada neurona se desactiva con probabilidad $p = 0.5$. Esto obliga a la red a no depender de neuronas especificas y reduce el sobreajuste.

```python
nn.Dropout(p=0.5)  # desactiva el 50% de las neuronas durante entrenamiento
```

### Diseno multi-GPU

El paper original de AlexNet dividio la red en dos GPUs (GTX 580 con 3GB de VRAM cada una), distribuyendo los filtros entre ambas. Hoy en dia las GPUs tienen suficiente memoria para ejecutar la red completa en una sola.

---

## 4. Implementacion en PyTorch

```python
import torch
import torch.nn as nn

class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()

        # conv1: Input 3x224x224 -> 96x55x55 -> MaxPool -> 96x27x27
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96,
                      kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU()
        )

        # conv2: 96x27x27 -> 256x27x27 -> MaxPool -> 256x13x13
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256,
                      kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU()
        )

        # conv3: 256x13x13 -> 384x13x13
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384,
                      kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU()
        )

        # conv4: 384x13x13 -> 384x13x13
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384,
                      kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU()
        )

        # conv5: 384x13x13 -> 256x13x13 -> MaxPool -> 256x6x6
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256,
                      kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU()
        )

        self.flat = nn.Flatten()

        # fc6: 9216 -> 4096
        self.fc6 = nn.Sequential(nn.Linear(9216, 4096), nn.ReLU())

        # fc7: 4096 -> 4096
        self.fc7 = nn.Sequential(nn.Linear(4096, 4096), nn.ReLU())

        # fc8: 4096 -> 1000 (clases ImageNet)
        self.fc8 = nn.Sequential(nn.Linear(4096, 1000))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flat(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        return x
```

### Verificacion

```python
modelo = AlexNet()
x = torch.randn(1, 3, 224, 224)
y = modelo(x)
print(y.shape)  # torch.Size([1, 1000])

# Conteo de parametros
total = sum(p.numel() for p in modelo.parameters())
print(f"Total parametros: {total:,}")  # ~62 millones
```

---

## 5. Conteo de parametros por capa

| Capa | Calculo | Parametros |
|------|---------|-----------|
| conv1 | $3 \times 96 \times 11 \times 11 + 96$ | 34,944 |
| conv2 | $96 \times 256 \times 5 \times 5 + 256$ | 614,656 |
| conv3 | $256 \times 384 \times 3 \times 3 + 384$ | 885,120 |
| conv4 | $384 \times 384 \times 3 \times 3 + 384$ | 1,327,488 |
| conv5 | $384 \times 256 \times 3 \times 3 + 256$ | 884,992 |
| fc6 | $9216 \times 4096 + 4096$ | 37,752,832 |
| fc7 | $4096 \times 4096 + 4096$ | 16,781,312 |
| fc8 | $4096 \times 1000 + 1000$ | 4,097,000 |
| **Total** | | **~62.4M** |

{{< concept-alert type="clave" >}}
La gran mayoria de los parametros de AlexNet estan en las capas fully connected (fc6, fc7, fc8), que suman ~58.6 millones de los ~62.4 millones totales. Las 5 capas convolucionales juntas solo tienen ~3.7 millones de parametros gracias al weight sharing.
{{< /concept-alert >}}

---

## 6. Transfer learning

El laboratorio incluye una aplicacion practica de AlexNet: clasificar 102 tipos de flores usando el dataset Flowers de Oxford. En vez de entrenar desde cero en ImageNet, se reutilizan los pesos ya aprendidos y se adapta la ultima capa para las nuevas clases.

El concepto basico de **transfer learning** es:

1. Tomar un modelo preentrenado en un dataset grande (ImageNet)
2. Reemplazar la ultima capa de clasificacion por una nueva que se ajuste al numero de clases del problema ($1000 \to 102$ en este caso)
3. Entrenar solo las capas nuevas (o hacer fine-tuning de toda la red)

```python
import torchvision.models as models

# Cargar AlexNet preentrenado en ImageNet
modelo = models.alexnet(weights="IMAGENET1K_V1")

# Reemplazar la ultima capa para 102 clases
modelo.classifier[6] = nn.Linear(4096, 102)
```

El modelo MiAlexNet que se construye en las actividades del laboratorio es una version modificada de AlexNet adaptada para trabajar con imagenes mas pequenas y menos clases.
