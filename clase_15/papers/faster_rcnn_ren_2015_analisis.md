# Análisis: Faster R-CNN (Ren et al., 2015)

> **Cita completa**
> Ren, S., He, K., Girshick, R., & Sun, J. (2015). *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*. Advances in Neural Information Processing Systems (NeurIPS) 28. Versión extendida publicada en IEEE TPAMI 2017.
>
> arXiv: [1506.01497](https://arxiv.org/abs/1506.01497) (v3, 6 enero 2016)
> Código original: <https://github.com/ShaoqingRen/faster_rcnn> (MATLAB), <https://github.com/rbgirshick/py-faster-rcnn> (Python/Caffe)
> Citas (a 2026): >70.000 según Google Scholar — uno de los papers de visión computacional más citados de la historia.

PDF local: [faster_rcnn_ren_2015.pdf](faster_rcnn_ren_2015.pdf)

---

## 1. Contexto histórico y problema

### El cuello de botella antes de Faster R-CNN

Hacia 2015 el estado del arte en detección de objetos era **Fast R-CNN** (Girshick, ICCV 2015), que ya compartía cómputo convolucional entre todas las propuestas de regiones de una imagen. Sin embargo, el paso de **proponer regiones candidatas** seguía siendo externo al modelo neuronal:

| Método de propuestas | Tiempo CPU/imagen | Calidad |
|----------------------|-------------------|---------|
| **Selective Search** (Uijlings 2013) | ~2 s | Buena, basada en superpíxeles |
| **EdgeBoxes** (Zitnick & Dollár, 2014) | ~0.2 s | Mejor tradeoff existente |
| Fast R-CNN (detector, sin propuestas) | ~0.2 s en GPU | — |

El detector tardaba 200 ms en GPU pero las propuestas tardaban 1–2 segundos en CPU. **El paso de "proponer" se había vuelto el cuello de botella.**

La pregunta del paper: ¿podemos hacer que la red neuronal *misma* proponga regiones, reutilizando los feature maps que ya calcula para detectar?

### Las tres formas previas de manejar múltiples escalas (Figura 1 del paper)

1. **Pirámide de imágenes**: redimensionar la imagen a varias escalas y correr el detector en cada una. Preciso pero ~k× más lento.
2. **Pirámide de filtros**: usar filtros de varios tamaños/aspect ratios sobre un único feature map (modelo DPM clásico). Más rápido pero requiere muchos modelos paralelos.
3. **Pirámide de anchors (propuesta del paper)**: un único feature map, un único filtro deslizante, y como referencias geométricas se usan **anchor boxes** de varias escalas y aspect ratios. Mucho más eficiente.

---

## 2. Contribución central

> *"We introduce a Region Proposal Network (RPN) that shares full-image convolutional features with the detection network, thus enabling nearly cost-free region proposals."*

Tres contribuciones de fondo:

1. **RPN (Region Proposal Network)** — una pequeña red totalmente convolucional que, sobre el feature map de la backbone, predice simultáneamente:
   - Una **probabilidad de objectness** (objeto / no-objeto) por cada anchor.
   - **4 desplazamientos** (Δx, Δy, Δw, Δh) que refinan cada anchor hacia la caja real.

2. **Anchor boxes con multi-escala y multi-aspect-ratio** como *referencias de regresión*. Cada posición del feature map tiene k anchors fijos (en el paper, k=9 = 3 escalas × 3 aspect ratios). La red aprende a clasificar y refinar cada uno.

3. **Sharing de features convolucionales entre propuesta y detección** mediante un esquema de entrenamiento alternado de 4 pasos. Esto convierte propuesta + detección en una sola red unificada que cuesta solo ~10 ms adicionales por imagen para las propuestas.

El abstract resume el impacto cuantitativo: **5 fps end-to-end con VGG-16 (17 fps con ZF), 300 propuestas en lugar de 2000, y mAP igual o mejor que Selective Search**.

---

## 3. Arquitectura en detalle

### 3.1 Estructura general (Figura 2 del paper)

```
                 image
                   │
            ┌──────▼──────┐
            │ conv layers │   (backbone compartida: ZF o VGG-16)
            │ (ResNet en  │
            │  versiones  │
            │  posteriores)│
            └──────┬──────┘
                   │ feature map (W×H×256 o 512)
        ┌──────────┴──────────┐
        │                     │
   ┌────▼────┐         ┌──────▼──────┐
   │   RPN   │ ──────► │ proposals   │
   │ (3×3 +  │         │ (~300 tras  │
   │  2×1×1) │         │  NMS)       │
   └─────────┘         └──────┬──────┘
                              │
                       ┌──────▼──────┐
                       │ RoI Pooling │  (toma propuestas + feature map)
                       └──────┬──────┘
                              │
                       ┌──────▼──────┐
                       │  Detector   │  (Fast R-CNN head: clasificación
                       │ (FC layers) │   por clase + regresión refinada)
                       └─────────────┘
```

La RPN actúa como **mecanismo de atención**: le dice al detector dónde mirar.

### 3.2 Region Proposal Network (RPN)

Es una red totalmente convolucional (FCN) construida así:

1. Una ventana deslizante de **3×3** sobre el feature map. Cada posición se mapea a un vector de **256-d** (ZF) o **512-d** (VGG) con ReLU.
2. Dos cabezas hermanas, ambas implementadas como convoluciones **1×1**:
   - **cls layer**: 2k salidas (objeto/no-objeto por cada anchor).
   - **reg layer**: 4k salidas (Δx, Δy, Δw, Δh por cada anchor).

Con k=9 anchors y un feature map típico de ~60×40 → **~20.000 anchors por imagen**.

### 3.3 Anchors — el truco geométrico clave

Los **anchors** son cajas de referencia *fijas a priori* que se "pegan" a cada posición del feature map. En el paper:

- **3 escalas**: 128², 256², 512² píxeles (referidas a la imagen original).
- **3 aspect ratios**: 1:1, 1:2, 2:1.
- → **k = 9 anchors por posición**.

La Tabla 1 del paper muestra los tamaños medios *aprendidos* de las propuestas para cada anchor (interesante: la red aprende a regresar a cajas mayores que el receptive field cuando es necesario, infiriendo objetos parcialmente visibles).

**Propiedad clave: invariancia a la traslación.** Si trasladas un objeto en la imagen, el anchor responsable también se traslada, y la función que predice la propuesta es la misma. MultiBox (Erhan et al., 2014) usaba k-means para generar 800 anchors no-invariantes; Faster R-CNN tiene órdenes de magnitud menos parámetros en la cabeza de propuestas:

- MultiBox output layer: ~6.1 M parámetros.
- Faster R-CNN output layer: ~2.8 × 10⁴ parámetros.

→ Menos sobreajuste en datasets pequeños.

### 3.4 Loss function (Ecuación 1 del paper)

$$ L(\{p_i\}, \{t_i\}) = \frac{1}{N_{cls}} \sum_i L_{cls}(p_i, p_i^*) + \lambda \frac{1}{N_{reg}} \sum_i p_i^* L_{reg}(t_i, t_i^*) $$

Donde:
- $i$ = índice del anchor en el mini-batch.
- $p_i$ = probabilidad predicha de que el anchor i sea objeto.
- $p_i^*$ = etiqueta ground-truth (1 si positivo, 0 si negativo).
- $t_i$ = coordenadas parametrizadas de la caja predicha.
- $t_i^*$ = coordenadas ground-truth asociadas a un anchor positivo.
- $L_{cls}$ = log loss binaria (objeto/no-objeto).
- $L_{reg}$ = **smooth L1 loss** (Huber loss), aplicada solo a anchors positivos (de ahí el $p_i^* L_{reg}$).
- $N_{cls} \approx 256$ (tamaño del mini-batch), $N_{reg} \approx 2400$ (número de anchors), $\lambda=10$ para balancear.

#### Asignación de etiquetas a anchors

Un anchor es **positivo** si:
1. Tiene el IoU más alto con alguna caja ground-truth (asegura que cada GT tenga al menos un anchor positivo), **o**
2. Tiene IoU > 0.7 con cualquier ground-truth.

Es **negativo** si su IoU < 0.3 con todas las GT.

Anchors con IoU entre 0.3 y 0.7 se **ignoran** (no contribuyen al loss).

#### Parametrización de cajas (Ecuación 2)

Sigue R-CNN:

$$ t_x = (x - x_a)/w_a, \quad t_y = (y - y_a)/h_a, \quad t_w = \log(w/w_a), \quad t_h = \log(h/h_a) $$

Es regresión relativa al anchor: posiciones normalizadas por la anchura/altura del anchor, dimensiones en log-espacio (estabilidad numérica + sesgo de aspect ratio).

### 3.5 Esquema de entrenamiento 4-Step Alternating Training

Compartir convoluciones entre RPN y Fast R-CNN es delicado: si entrenas independiente, ambas redes querrán cambiar las features compartidas en direcciones diferentes. La solución del paper:

1. **Paso 1**: entrenar RPN, inicializada con pesos ImageNet, fine-tuned end-to-end para propuestas.
2. **Paso 2**: entrenar Fast R-CNN (también desde ImageNet) usando las propuestas del paso 1. *Aún no comparten convoluciones.*
3. **Paso 3**: re-inicializar RPN con las convoluciones del detector del paso 2, **congelar las capas convolucionales compartidas** y ajustar solo las capas únicas de la RPN.
4. **Paso 4**: con las convoluciones congeladas, fine-tunear solo las capas únicas de Fast R-CNN. Ahora ambas redes comparten las convoluciones.

El paper también discute *approximate joint training* (más simple, ~25–50% más rápido) que ignora el gradiente respecto a las coordenadas de las propuestas. En implementaciones modernas (incluida torchvision usada en el lab) se usa joint training.

### 3.6 Detalles de implementación

- **Escala única**: imagen redimensionada a lado corto = 600 px. No usa pirámide de imágenes.
- **NMS sobre propuestas**: threshold IoU = 0.7 sobre los scores de objectness → reduce 20k anchors → ~2000 propuestas → se queda con top-N=300 para entrenamiento del detector.
- **Cross-boundary anchors**: durante entrenamiento se ignoran los anchors que cruzan los límites de la imagen (~6000 sobreviven de los ~20000). En test, se recortan al borde.
- **Mini-batch**: imagen-céntrico, 256 anchors muestreados por imagen con ratio positivos:negativos hasta 1:1.

---

## 4. Resultados experimentales

### 4.1 PASCAL VOC 2007 (Tabla 2 del paper)

Con Fast R-CNN + ZF como detector base:

| Propuestas | Test propuestas | # cajas test | mAP |
|------------|----------------|--------------|-----|
| Selective Search | SS | 2000 | 58.7% |
| EdgeBoxes | EB | 2000 | 58.6% |
| **RPN+ZF, shared** | RPN+ZF, shared | **300** | **59.9%** |

→ La RPN iguala o supera SS/EB con **~6× menos propuestas**.

### 4.2 PASCAL VOC 2007 con VGG-16 (Tabla 3)

| Método | Datos | mAP |
|--------|-------|-----|
| SS 2000 | VOC07+12 | 70.0% |
| RPN+VGG, shared 300 | VOC07 | 69.9% |
| RPN+VGG, shared 300 | VOC07+12 | **73.2%** |
| RPN+VGG, shared 300 | COCO+07+12 | **78.8%** |

### 4.3 Velocidad (Tabla 5 — la cifra famosa)

| Sistema | Conv | Propuesta | Region-wise | Total | FPS |
|---------|------|-----------|-------------|-------|-----|
| SS + Fast R-CNN (VGG) | 146 ms | 1510 ms | 174 ms | 1830 ms | 0.5 |
| **RPN + Fast R-CNN (VGG)** | 141 ms | **10 ms** | 47 ms | **198 ms** | **5** |
| **RPN + Fast R-CNN (ZF)** | 31 ms | 3 ms | 25 ms | **59 ms** | **17** |

Reducción del paso de propuestas de **1510 → 10 ms** (150×).

### 4.4 MS COCO (Tabla 11)

| Método | Propuestas | mAP@0.5 | mAP@[.5,.95] |
|--------|-----------|---------|--------------|
| Fast R-CNN [paper] | SS, 2000 | 39.3% | 19.3% |
| **Faster R-CNN** | RPN, 300 | **42.7%** | **21.9%** |

Faster R-CNN no solo es más rápido, **es más preciso** — especialmente en mAP@[.5,.95], la métrica COCO más exigente que premia localización precisa (+2.2 puntos absolutos).

### 4.5 Ablation studies importantes

**Tabla 8 — efecto de los anchors:**

| Escalas × Ratios | mAP |
|------------------|-----|
| 1 × 1 (1 anchor) | 65.8% |
| 1 × 3 | 68.8% |
| 3 × 1 | **69.8%** |
| **3 × 3 (default)** | **69.9%** |

→ La diversidad de escalas es clave (+4 puntos); añadir aspect ratios da algo más.

**Tabla 9 — sensibilidad a λ:**

| λ | 0.1 | 1 | 10 | 100 |
|---|-----|---|----|----|
| mAP | 67.2% | 68.9% | 69.9% | 69.1% |

→ El método es robusto a λ en ~2 órdenes de magnitud.

**Tabla 10 — Two-stage vs One-stage (vs OverFeat estilo):**

| Sistema | Propuestas | mAP |
|---------|------------|-----|
| Two-Stage (RPN+ZF) | 300 | **58.7%** |
| One-Stage emulando OverFeat | 20000 | 53.8% |

→ El cascading de propuestas + clasificación class-specific da ~5 puntos sobre detectar directamente en sliding windows. Justifica la elección two-stage.

### 4.6 Resultados en competencias 2015

- **ImageNet Detection 2015 (1er lugar)** con ResNet-101 backbone.
- **ImageNet Localization 2015 (1er lugar)**.
- **COCO Detection 2015 (1er lugar)**.
- **COCO Segmentation 2015 (1er lugar)** vía Mask R-CNN preliminar.

---

## 5. Limitaciones reconocibles (algunas ya conocidas, otras emergentes)

1. **Velocidad aún lejos del tiempo real estricto**: 5 fps con VGG-16. YOLOv1 (2016) llegaría a 45 fps con la idea one-stage.
2. **Cross-boundary anchors**: hay que tratarlos cuidadosamente o el entrenamiento no converge.
3. **Single-scale feature map**: el paper usa solo el último feature map del backbone. Esto limita la detección de objetos pequeños — problema resuelto por la **Feature Pyramid Network** (FPN, Lin et al. CVPR 2017), que es exactamente lo que usa la implementación del laboratorio (`fasterrcnn_resnet50_fpn`).
4. **RoI Pooling tiene cuantización**: el RoI se discretiza en celdas de manera grosera, perdiendo precisión sub-píxel. Resuelto por **RoIAlign** en Mask R-CNN (He et al., ICCV 2017).
5. **Anchors hard-coded**: requiere elegir escalas/ratios manualmente. Detectores posteriores (CenterNet, DETR) eliminan los anchors.

---

## 6. Impacto e influencia

Faster R-CNN es la **arquitectura semilla** de toda la familia de detectores two-stage moderna:

- **Mask R-CNN** (2017): añade rama de segmentación de instancias, reemplaza RoI Pooling por RoIAlign.
- **Cascade R-CNN** (2018): cascada de detectores con thresholds IoU crecientes.
- **HTC, DetectoRS** (2019-2020): detectores híbridos de cascada.

La idea **anchor + RPN + RoI head compartiendo backbone** persiste hasta hoy, incluso cuando los detectores modernos (DETR, Deformable DETR, DINO) reemplazan los anchors por queries de transformer.

En aplicaciones, Faster R-CNN sigue siendo **una baseline producción-grade** en torchvision, Detectron2 y mmdetection: es el detector "que funciona bien por defecto" para problemas de fine-tuning sobre datasets pequeños — exactamente el caso del laboratorio con el dataset Raccoon.

---

## 7. Conexión directa con el Laboratorio

El laboratorio usa `torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)`, que difiere del paper original en tres aspectos:

| Aspecto | Paper 2015 | Laboratorio (torchvision) |
|---------|-----------|---------------------------|
| Backbone | VGG-16 / ZF | **ResNet-50 + FPN** |
| Feature map | Único (último conv) | **Pirámide multi-escala (FPN)** |
| RoI extraction | RoI Pooling | **MultiScaleRoIAlign** |
| Pre-entrenamiento | ImageNet → fine-tune VOC/COCO | Pre-entrenado en **COCO 2017** completo |
| Entrenamiento | 4-step alternating | Joint training |

Todo lo demás del paper aplica directamente:

- La sección **Análisis del modelo** del notebook recorre exactamente las componentes descritas aquí: `transform`, `backbone`, `rpn`, `roi_heads`.
- La función `get_prediction` filtra por **score threshold** — los scores son el output del cls head visto en la Sección 3.2 de este análisis.
- El **fine-tuning** del lab consiste en reemplazar solo el `FastRCNNPredictor` (las dos cabezas finales: clasificación N+1 clases + regresión 4·(N+1) coordenadas), congelando o no la backbone. Esto es exactamente el patrón estándar derivado del paper.
- La función `iou()` del notebook implementa la Ecuación implícita de IoU usada para asignar anchors positivos/negativos y para evaluación.
- El **NMS** del lab (`torchvision.ops.nms` y `filter_by_class_nms`) implementa la post-procesamiento descrito en la Sección 3.6 del paper.

---

## 8. Lectura complementaria recomendada

| Paper | Año | Relevancia para el lab |
|-------|-----|----------------------|
| **Fast R-CNN** (Girshick) | 2015 | Predecesor directo; explica RoI Pooling y multi-task loss |
| **FPN** (Lin et al.) | 2017 | Es lo que añade torchvision a la backbone — explica el bloque `BackboneWithFPN` y la elección multi-escala |
| **ResNet** (He et al.) | 2015 | Backbone del lab; entender residual blocks |
| **Mask R-CNN** (He et al.) | 2017 | Sucesor que reemplaza RoI Pooling por RoIAlign |
| **COCO dataset** (Lin et al.) | 2014 | Las 91 clases del lab |

Cuando lleguemos a las celdas correspondientes del notebook (FPN, COCO, RoIAlign) descargaré y analizaré también sus papers.
