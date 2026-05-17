# Análisis: Feature Pyramid Networks (Lin et al., 2017)

> **Cita completa**
> Lin, T.-Y., Dollár, P., Girshick, R., He, K., Hariharan, B., & Belongie, S. (2017). *Feature Pyramid Networks for Object Detection*. CVPR 2017.
>
> arXiv: [1612.03144](https://arxiv.org/abs/1612.03144)
> Citas (2026): ~30.000.

PDF local: [fpn_lin_2017.pdf](fpn_lin_2017.pdf)

---

## 1. Contexto y problema

### El problema de las múltiples escalas en detección

Detectar objetos a escalas muy diferentes (desde 30 px hasta 500 px en la misma imagen) es difícil. Las soluciones previas tenían trade-offs:

**Figura 1 del paper — cuatro arquitecturas comparadas:**

```
(a) Pirámide de imágenes (featurized image pyramid)
    imagen → escala 1 → CNN → predict
    imagen → escala 2 → CNN → predict
    imagen → escala 3 → CNN → predict
    PROBLEMA: 4× más lento. Inviable entrenando.

(b) Single feature map (Fast R-CNN, Faster R-CNN original)
    imagen → CNN → último feature map → predict
    PROBLEMA: pobre con objetos pequeños.

(c) Pyramidal feature hierarchy (SSD)
    imagen → CNN ──┬── feature map C3 → predict
                   ├── feature map C4 → predict
                   └── feature map C5 → predict
    PROBLEMA: capas tempranas (C2, C3) tienen poca semántica
              → SSD las descarta y empieza desde conv4_3 de VGG.
              → pierde resolución útil.

(d) Feature Pyramid Network (FPN) ←─ propuesta del paper
    imagen → CNN ──┬── C3 ─┐    P3 → predict
                   ├── C4 ─┼─→  P4 → predict
                   └── C5 ─┘    P5 → predict
                              ↑
                       top-down + lateral merge
    Aporta: P3, P4, P5 todos con semántica alta y resoluciones distintas.
```

La clave: ConvNets **ya tienen** una jerarquía piramidal natural (los stages C2, C3, C4, C5 son resoluciones cada vez menores). Pero **C2 tiene semántica débil** porque está cerca de la entrada. SSD evita usarla; FPN propone enriquecerla con información de C5 a través del top-down pathway.

---

## 2. Contribución central

> *"We propose a top-down architecture with lateral connections [...] for building high-level semantic feature maps at all scales."*

Tres elementos:

1. **Bottom-up pathway**: la backbone CNN estándar (ResNet), produciendo $\{C_2, C_3, C_4, C_5\}$ con strides $\{4, 8, 16, 32\}$.

2. **Top-down pathway**: empezando desde $C_5$, ir hacia arriba doblando la resolución espacial en cada paso (upsampling × 2 por vecino más cercano). Esto **alucina** mapas de alta resolución desde features semánticamente fuertes pero gruesas.

3. **Lateral connections**: en cada nivel, mezclar el output del top-down con el del bottom-up correspondiente (después de pasar por una conv 1×1 que iguala canales). Esto **inyecta resolución precisa** (del bottom-up) a las features ricas semánticamente (del top-down).

Resultado: una pirámide $\{P_2, P_3, P_4, P_5\}$ donde **todos los niveles tienen 256 canales** y **todos tienen semántica fuerte**.

---

## 3. Arquitectura en detalle

### Bloque de fusión (Figura 3 del paper)

```
                    feature map nivel superior (más profundo, más semántica)
                              │
                              │  upsample 2× (nearest neighbor)
                              ▼
       C_i ──[1×1 conv]──→  +  ──[3×3 conv]──→  P_i
       │       (lateral)      ▲
       │                      │
       │     reduce a 256 ch  │
       └──────────────────────┘
```

**Tres operaciones por nivel:**

1. **Lateral 1×1 conv** sobre $C_i$:
   - $C_2$ entra con 256 ch → sale con 256 ch.
   - $C_3$ entra con 512 ch → sale con 256 ch.
   - $C_4$ entra con 1024 ch → sale con 256 ch.
   - $C_5$ entra con 2048 ch → sale con 256 ch.
   - Solo cambia número de canales, no resolución.

2. **Upsample × 2** del nivel superior por vecino más cercano (sin parámetros).

3. **Suma elemento a elemento** + **conv 3×3** para suavizar artefactos del upsampling.

### Para $P_5$ (el más alto)

No tiene nivel "más alto" del cual venir el top-down. Solo se aplica la lateral 1×1 sobre $C_5$:
$$ P_5 = \text{Conv}_{3 \times 3}(\text{Conv}_{1 \times 1}(C_5)) $$

### Niveles totales en torchvision

El paper menciona $\{P_2, P_3, P_4, P_5\}$. Pero torchvision añade un nivel **$P_6$** = MaxPool 1×1 stride 2 sobre $P_5$, para detectar objetos muy grandes (stride /64). Por eso en el `print(model)` del lab ves 5 niveles ('0', '1', '2', '3', 'pool').

### Detalles importantes

- **Todos los niveles de la FPN comparten cabezas** (RPN, RoI head). Los pesos son los mismos. Esto refleja la creencia de que **la semántica es similar entre niveles** — solo cambia la escala.
- **No hay no-linealidades extra** en las laterales/top-down (no ReLU dentro del bloque). Los autores reportan que tienen impacto mínimo.

---

## 4. Asignación de anchors a niveles

### Para RPN

- Cada nivel $P_k$ tiene anchors de **una sola escala**:
  - $P_2$: anchors de área $32^2$.
  - $P_3$: anchors de área $64^2$.
  - $P_4$: anchors de área $128^2$.
  - $P_5$: anchors de área $256^2$.
  - $P_6$: anchors de área $512^2$.
- En cada nivel, **3 aspect ratios** (1:1, 1:2, 2:1) → 3 anchors por celda.
- **Total**: 15 "tipos" de anchors a través de la pirámide (vs 9 apilados en una sola escala del paper Faster R-CNN original).

### Para RoI Pooling / RoIAlign (Fast R-CNN head)

¿Cómo asignar una propuesta (de tamaño $w \times h$) al nivel correcto de la pirámide?

$$ k = \lfloor k_0 + \log_2(\sqrt{wh} / 224) \rfloor $$

Con $k_0 = 4$ (es decir, una RoI de $224 \times 224$ se asigna a $P_4$). Propuestas pequeñas van a niveles de alta resolución; propuestas grandes a niveles profundos.

Es exactamente el comportamiento de **`MultiScaleRoIAlign`** que verás en el `roi_heads.box_roi_pool` del lab.

---

## 5. Resultados experimentales (Tabla 1 del paper — ablation)

Sobre COCO, evaluando recall de propuestas RPN:

| Variante | AR@1k (recall 1000 props) | AR_s (objetos pequeños) |
|----------|---------------------------|--------------------------|
| (a) RPN single-scale C4 | 47.4 | 14.4 |
| (b) RPN single-scale C5 | 47.5 | 14.5 |
| **(c) FPN completo** | **56.3** (+8.0 vs baseline) | **27.4** (+12.9 vs baseline) |
| (d) FPN sin top-down | 50.0 | 23.4 (peor) |
| (e) FPN sin laterales | 46.5 | — (peor que single-scale) |
| (f) Solo head en $P_2$ | 51.4 | — |

**Conclusiones clave:**

- **+8 puntos absolutos en AR@1k** vs RPN single-scale.
- **+12.9 puntos en objetos pequeños** — la mejora más grande, justo donde fallaba Faster R-CNN.
- **Quitar el top-down (d) degrada mucho**: indica que la semántica que baja desde $C_5$ es crítica.
- **Quitar las laterales (e) degrada todavía más**: la resolución precisa del bottom-up es indispensable.

### Detección completa (COCO, Faster R-CNN + ResNet-50)

| Backbone | mAP@[.5,.95] |
|----------|--------------|
| ResNet-50 single-scale (C4) | 31.9 |
| **ResNet-50 + FPN** | **36.2** (+4.3) |
| ResNet-101 + FPN | 39.1 |

---

## 6. Impacto

FPN se convirtió **inmediatamente** en componente estándar de toda detección moderna:

- **Mask R-CNN** (2017): usa FPN por defecto.
- **RetinaNet** (2017): one-stage detector built on FPN.
- **EfficientDet** (2019): generaliza FPN a BiFPN (bidirectional).
- **YOLO v4, v5, v8**: todos usan variantes de FPN (PANet, etc.).
- **DETR** (2020) y derivados: aunque eliminan anchors, mantienen multi-escala con queries que atienden a múltiples niveles.

En `torchvision`, todos los detectores listados en su API (`fasterrcnn_resnet50_fpn`, `maskrcnn_resnet50_fpn`, `retinanet_resnet50_fpn`, `fcos_resnet50_fpn`, `keypointrcnn_resnet50_fpn`) llevan **"fpn"** explícito en el nombre. Es el componente que los hace competitivos.

---

## 7. Conexión con el laboratorio

En el `print(frcnn_model)` del notebook ves:

```python
(fpn): FeaturePyramidNetwork(
    (inner_blocks): ModuleList(   ← las 4 laterales 1×1
        (0): Conv2d(256,  256, kernel_size=(1, 1))   # para C2
        (1): Conv2d(512,  256, kernel_size=(1, 1))   # para C3
        (2): Conv2d(1024, 256, kernel_size=(1, 1))   # para C4
        (3): Conv2d(2048, 256, kernel_size=(1, 1))   # para C5
    )
    (layer_blocks): ModuleList(   ← las 4 convs 3×3 suavizadoras
        (0-3): Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1))
    )
    (extra_blocks): LastLevelMaxPool()   ← genera P6
)
```

- **`inner_blocks`** son las laterales 1×1 (una por nivel).
- **`layer_blocks`** son las 3×3 suavizadoras (una por nivel).
- **`extra_blocks`** añade $P_6$ que el paper original no tiene (es una extensión propia de torchvision/Detectron2).

Y en `roi_heads.box_roi_pool` ves:

```python
(box_roi_pool): MultiScaleRoIAlign(
    featmap_names=['0', '1', '2', '3'],   # P2, P3, P4, P5 (P6 solo para RPN)
    output_size=(7, 7),
    sampling_ratio=2
)
```

Que implementa exactamente la fórmula $k = \lfloor 4 + \log_2(\sqrt{wh}/224) \rfloor$ del paper.

La FPN es lo que permite que el laboratorio detecte bien tanto las **zebras** (objetos grandes en la primera imagen de prueba) como los objetos pequeños en la imagen de **oficina** con threshold 0.9 — sin FPN, los objetos pequeños desaparecerían a stride /32.
