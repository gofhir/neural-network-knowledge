# Análisis: Mask R-CNN (He et al., 2017)

> **Cita completa**
> He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). *Mask R-CNN*. ICCV 2017.
>
> arXiv: [1703.06870](https://arxiv.org/abs/1703.06870)
> Citas (2026): ~50.000.
> Premios: **ICCV 2017 Best Paper Award (Marr Prize)**.

PDF local: [mask_rcnn_he_2017.pdf](mask_rcnn_he_2017.pdf)

---

## 1. Contexto

Hasta 2017 había dos tareas relacionadas pero distintas en visión:

- **Detección de objetos** (Faster R-CNN): bounding boxes + clases.
- **Segmentación semántica** (FCN, DeepLab): clase por píxel, sin distinguir instancias.

Faltaba una tercera tarea: **segmentación de instancias**, que combina ambas:

> "Para cada objeto, dame su máscara binaria a nivel de píxel **y** distinguirlo de otros objetos de la misma clase."

```
Detección              Segmentación semántica          Segmentación de instancias
                       
[caja]🐕               🐕🐕🐕🐕🐕🐕🐕                   🐕(rojo)🐕(rojo)
[caja]🐕               🐕🐕🐕🐕🐕🐕🐕                   🐕(verde)🐕(verde)
[caja]🐕               🐕🐕🐕🐕🐕🐕🐕                   🐕(azul)🐕(azul)
                       Solo dice "es perro"            Distingue cada instancia
```

Mask R-CNN resolvió esto extendiendo Faster R-CNN. Pero para nuestro propósito en el lab (que usa Faster R-CNN sin máscaras), **lo importante del paper es RoIAlign** — el reemplazo de RoIPool que torchvision usa.

---

## 2. La contribución que afecta al laboratorio: RoIAlign

### El problema de RoIPool

`RoIPool` (paper Fast R-CNN, 2015) cuantiza dos veces:

1. **Cuantización de coordenadas**: una propuesta de tamaño $w \times h$ se mapea al feature map dividiendo entre el stride (digamos 16). El resultado típicamente NO es entero, pero RoIPool lo redondea:
   $$ x_{feature} = \lfloor x_{imagen} / 16 \rfloor $$

2. **Cuantización de bins**: la región resultante se divide en una grilla 7×7. Si la región mide 6.25×6.25 celdas del feature map y queremos 7×7 bins → cada bin debería tener 0.893×0.893 celdas, pero RoIPool cuantiza otra vez para dar bins de tamaño entero.

**Consecuencia**: errores de hasta 10-20 píxeles en la imagen original (con stride 32). Para clasificación es irrelevante (la red es robusta), para segmentación de máscaras es catastrófico (los bordes quedan dentados o desplazados).

### La solución de RoIAlign (Figura 3 del paper)

> *"We avoid any quantization of the RoI boundaries or bins."*

```
   feature map (dashed grid, valores enteros)
   ┌───┬───┬───┬───┐
   │   │   │   │   │
   ├───┼───┼───┼───┤
   │   │ ╋ │ ╋ │   │    ← 4 puntos de muestreo
   ├───┼───┼───┼───┤      (sampling_ratio=2 → 2×2=4 por bin)
   │   │ ╋ │ ╋ │   │
   ├───┼───┼───┼───┤
   │   │   │   │   │
   └───┴───┴───┴───┘
       └─ RoI (solid line, coords flotantes) ─┘
       └─ 1 bin de los 7×7 que queremos ─┘
       
   Cada punto ╋ se calcula con interpolación bilineal
   de los 4 píxeles vecinos del feature map.
   Luego se promedia (o se hace max-pool) los 4 puntos del bin.
```

**Sin cuantizaciones**: coordenadas de la RoI en floats, bins de tamaño float, interpolación bilineal precisa.

### Resultados (Tabla 2c del paper)

| Método | mask AP | mask AP@0.5 | mask AP@0.75 |
|--------|---------|-------------|--------------|
| RoIPool | 26.9 | 48.8 | 26.4 |
| RoIWarp (intento previo) | 27.2 | 49.2 | 27.1 |
| **RoIAlign** | **30.2** | 51.0 | **31.8** |

**+3.3 puntos de AP** solo cambiando la operación de extracción. **+5.4 puntos en AP@0.75** (métrica de localización estricta).

Y para detección (no segmentación) tambien gana **+1.3 puntos box AP**.

---

## 3. La arquitectura completa de Mask R-CNN (referencia)

Es Faster R-CNN con una **tercera rama** en el RoI head, en paralelo a clasificación y regresión:

```
                                ┌──→ cls head (softmax sobre K clases)
RoI features                    │
(via RoIAlign)  ──→ FC layers ──┼──→ box head (4×K deltas)
                                │
                                └──→ mask head (K × m × m máscaras binarias)
                                     (un FCN, no FCs)
```

### Detalles de la rama de máscara

- **m × m**: 28×28 por máscara (resolución pequeña pero suficiente, se redimensiona luego).
- **K máscaras por RoI** (una por clase). Pero solo se usa la máscara correspondiente a la clase predicha por el cls head.
- **FCN (no FCs)**: una pequeña red totalmente convolucional → conserva la estructura espacial.
- **Sigmoid + binary cross-entropy** por máscara, **no softmax** entre clases. Esto desacopla clasificación y segmentación.

### Loss multi-task

$$ L = L_{cls} + L_{box} + L_{mask} $$

Donde $L_{mask}$ es BCE solo sobre la clase predicha (las otras máscaras no contribuyen al loss).

---

## 4. Resultados (Tabla 1 del paper)

Sobre COCO test-dev, segmentación de instancias:

| Método | Backbone | mask AP | AP@0.5 | AP@0.75 | AP_S | AP_M | AP_L |
|--------|----------|---------|--------|---------|------|------|------|
| MNC (ganador 2015) | ResNet-101-C4 | 24.6 | 44.3 | 24.8 | 4.7 | 25.9 | 43.6 |
| FCIS+++ (ganador 2016) | ResNet-101-C5-dilated | 33.6 | 54.5 | — | — | — | — |
| **Mask R-CNN** | ResNet-101-C4 | 33.1 | 54.9 | 34.8 | 12.1 | 35.6 | 51.1 |
| **Mask R-CNN** | ResNet-101-FPN | **35.7** | 58.0 | 37.8 | 15.5 | 38.1 | 52.4 |
| **Mask R-CNN** | ResNeXt-101-FPN | **37.1** | 60.0 | 39.4 | 16.9 | 39.9 | 53.5 |

Supera al ganador del COCO 2016 (FCIS+++) **sin bells & whistles** (sin multi-scale train/test, sin OHEM, sin ensembles).

Para **detección** (no segmentación) también gana frente a Faster R-CNN base por **~+1.3 box AP**, ganancia "gratis" gracias a RoIAlign.

---

## 5. Otras contribuciones secundarias

- **Decoupling mask and class**: predecir K máscaras independientes (sigmoid + BCE binario) en vez de una máscara multi-clase con softmax. Esto permite que las clases no compitan entre sí en la rama de máscara.

- **Demostración de generalidad**: solo cambiando la rama final, Mask R-CNN hace **keypoint detection** (estimación de pose humana 2D), tratando cada keypoint como una máscara one-hot. Ganó COCO Keypoints 2017.

- **Inferencia rápida**: 5 fps con ResNet-101-FPN (similar a Faster R-CNN). El head de máscara solo se aplica a las top-100 detecciones, no a las 1000 propuestas.

---

## 6. Conexión con el laboratorio

En el `print(frcnn_model)` ves:

```python
(box_roi_pool): MultiScaleRoIAlign(
    featmap_names=['0', '1', '2', '3'],
    output_size=(7, 7),
    sampling_ratio=2
)
```

- **MultiScaleRoIAlign** = combinación de RoIAlign de Mask R-CNN + asignación a niveles de FPN.
- **`output_size=(7, 7)`**: cada propuesta se extrae como un tensor de 7×7 con interpolación precisa.
- **`sampling_ratio=2`**: 2×2 = 4 puntos de muestreo por bin (compromiso accuracy/velocidad). El default del paper.

El lab **no usa la rama de máscara** (porque usa `fasterrcnn_resnet50_fpn`, no `maskrcnn_resnet50_fpn`), pero **sí hereda RoIAlign**. Esto es parte del por qué los resultados del lab son mejores que los del paper Faster R-CNN original de 2015: la implementación moderna se beneficia de:

1. Backbone más profunda (ResNet-50 en vez de VGG-16/ZF).
2. FPN multi-escala.
3. **RoIAlign en vez de RoIPool** (esta es la herencia directa de Mask R-CNN).
4. Joint training en vez de 4-step alternating training.

---

## 7. Una observación de diseño general

Mask R-CNN es un caso de estudio sobre **cómo el progreso en deep learning suele venir de "decoupling"**: identificar acoplamientos perjudiciales y separarlos.

- **Mask R-CNN** desacopla:
  - Clasificación de segmentación (sigmoid binario por clase, no softmax multi-clase en máscaras).
  - Localización de extracción (RoIAlign elimina la cuantización que acoplaba ambas).

- **Faster R-CNN** ya había desacoplado:
  - Generación de propuestas (RPN) de clasificación (RoI head).
  - Backbone (compartida) de tareas específicas (cabezas distintas).

Esta filosofía de "modular cleanly" es lo que permite reusar Faster R-CNN como base para Mask R-CNN cambiando solo un componente, y reusarlo de nuevo para Keypoint R-CNN cambiando solo la cabeza. En el lab vemos que **lo mismo aplica al fine-tuning**: cambiamos solo `FastRCNNPredictor` y mantenemos el resto intacto.
