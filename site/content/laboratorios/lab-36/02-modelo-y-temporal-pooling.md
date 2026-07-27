---
title: "El modelo y el temporal pooling"
weight: 2
---

El corazón del lab: cómo clasificar un video. La estrategia es la más simple posible —promediar los frames— y es exactamente lo que expone su límite fundamental: la pérdida del orden temporal.

## VideoNet: ResNet-34 + average pooling

```python
class VideoNet(nn.Module):
    def __init__(self, num_classes, num_segments=8):
        self.net = models.resnet34(pretrained=True)          # backbone 2D pre-entrenado
        self.fc = nn.Linear(self.net.fc.in_features, num_classes)
        self.net.fc = nn.Identity()                          # ResNet como extractor de features

    def forward(self, inputs):
        inputs = inputs.view((-1, 3) + inputs.size()[-2:])   # (B,8*3,H,W) -> (B*8,3,H,W)
        out = self.net(inputs)                               # ResNet a cada frame -> (B*8, 512)
        out = out.view((-1, self.num_segments) + out.size()[1:])  # (B*8,512) -> (B,8,512)
        out = torch.mean(out, dim=1)                         # TEMPORAL POOLING: (B,8,512)->(B,512)
        out = self.fc(out)                                   # (B,512) -> (B,11)
        return out
```

La estrategia en una frase: **pasa cada frame por un ResNet-34, promedia los features de los 8 frames, y clasifica ese promedio.** No hay convoluciones 3D, ni LSTM, ni flujo óptico.

## Transfer learning: reciclar ResNet-34

Entrenar una CNN de video desde cero necesitaría millones de videos. En su lugar:
- **`models.resnet34(pretrained=True)`** carga un ResNet ya entrenado en ImageNet — sabe extraer features visuales genéricas (bordes, texturas, objetos).
- **`self.net.fc = nn.Identity()`** reemplaza la cabeza de ImageNet por una identidad → ResNet se vuelve un **extractor de features** puro (devuelve el vector de 512-D antes de clasificar).
- Una **cabeza nueva** (`self.fc`) mapea esos 512-D a las 11 clases de UCF11.

Un frame de video *es* una imagen, así que el conocimiento de ImageNet transfiere directo. Esto es lo que hace viable entrenar con ~1200 videos en 3 épocas. También exige normalizar los frames con las estadísticas de ImageNet (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`) — el "contrato" con el modelo pre-entrenado.

## El viaje de las dimensiones

El truco del `forward` es cómo maneja las formas para aplicar una CNN 2D a una secuencia:

```
(B, 24, 224, 224)   batch de videos, 8 frames RGB apilados en canales
   │ view → desapilar
(B*8, 3, 224, 224)   frames como imágenes independientes
   │ ResNet-34 (2D, SIN noción de tiempo)
(B*8, 512)           un vector de features por frame
   │ view → reagrupar por video
(B, 8, 512)          features agrupados
   │ torch.mean(dim=1) ← AQUÍ se pierde el orden
(B, 512)             UN vector promedio por video
   │ fc
(B, 11)              logits de las 11 clases
```

Este patrón "aplanar temporal → CNN 2D → reagrupar → promediar" es esencialmente el enfoque de **TSN (Temporal Segment Networks)** con consenso por promedio. Nota que **`num_segments` debe coincidir con `num_frames`** — si cambias los frames pero no el modelo, el `view` de reagrupación falla.

## El límite fundamental: invarianza al orden

```python
out = torch.mean(out, dim=1)   # el average temporal pooling
```

{{< callout type="warning" >}}
**El average pooling es invariante al orden temporal.** Como $\text{mean}(f_1,...,f_8) = \text{mean}$ de cualquier permutación, **el modelo da la misma predicción sin importar el orden de los frames**. Barajarlos no cambiaría nada. Consecuencias:
- **No distingue acciones que son inversas temporales**: "sentarse" vs "pararse", "abrir" vs "cerrar", "entrar" vs "salir".
- **No captura dinámica ni dirección de movimiento**; solo la "apariencia promedio".
- **Trata el video como un "bag of frames"** (bolsa desordenada), no como una secuencia.

Funciona bien para acciones reconocibles por apariencia/contexto estático (diving → piscina + pose), pero falla en acciones definidas por su dinámica. Métodos como C3D/I3D (convolución 3D), LRCN (LSTM) o Two-Stream con [flujo óptico](/fundamentos/flujo-optico) resuelven esto modelando el tiempo explícitamente. Esta es la limitación central que la [clase 36](/clases/clase-36) motiva.
{{< /callout >}}

## El entrenamiento (resultado real)

Fine-tuning con `SGD(lr=0.01)`, `CrossEntropyLoss`, batch de 40 videos (= 40×8 = 320 imágenes por batch), 3 épocas:

| Época | Train Acc | Val Acc |
|-------|-----------|---------|
| 0 | 40.8% | 64.4% |
| 1 | 78.3% | 77.6% |
| 2 | 89.5% | **84.6%** |

![Curva de accuracy train vs val a lo largo de 3 épocas, ambas ascendentes, con 8 frames](/laboratorios/lab-36/curva-8-frames.png)

**Best val Acc: 84.6%** en 4m30s. La curva es sana (train y val suben juntas, sin overfitting grave), confirmando que **el "bag of frames" funciona bien en UCF11** — muchas de sus acciones se reconocen por apariencia/contexto, no por dinámica. El transfer learning de ResNet + 3 épocas basta.

El bucle de entrenamiento usa el patrón canónico de dos fases (`for epoch → for phase in ['train','val']`), con `model.train()`/`model.eval()` (importante aquí porque ResNet tiene BatchNorm), gradientes solo en train (`torch.set_grad_enabled(phase=='train')`), y guardado de los mejores pesos por val accuracy (early-stopping implícito).

{{< callout type="info" >}}
**Un detalle didáctico:** al buscar ejemplos mal clasificados, el `bad_examples` salió **vacío** — el modelo acertó los 40 videos del primer batch evaluado. Es señal de lo bien que entrenó (pero rompe la celda que quiere mostrar un error; el fix es evaluar todo el conjunto de validación quitando el `break` de `getInferenceModel`).
{{< /callout >}}
