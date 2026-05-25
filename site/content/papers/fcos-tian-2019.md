---
title: "FCOS (Fully Convolutional One-Stage Detection)"
weight: 103
math: true
---

{{< paper-card
    title="FCOS: Fully Convolutional One-Stage Object Detection"
    authors="Tian, Shen, Chen, He"
    year="2019"
    venue="ICCV 2019"
    pdf="/papers/fcos-tian-2019.pdf"
    arxiv="1904.01355" >}}
Primer detector **one-stage anchor-free** competitivo con los anchor-based de su época. Reformula la detección como **predicción densa por píxel** análoga a una FCN de segmentación: cada location $(x, y)$ del feature map predice clase, cuatro offsets $(l, t, r, b)$ a los lados del bbox, y un escalar de **center-ness** que penaliza predicciones lejos del centro. Alcanza **44.7 AP** en COCO test-dev con ResNeXt-64x4d-101-FPN, superando a RetinaNet y CornerNet con menos hiperparámetros y ~9× menos outputs por location. Es el **backbone arquitectónico de ABCNet** (clase 21).
{{< /paper-card >}}

---

## El problema: la era anchor-based y sus cuatro dolores

Entre 2015 y 2018 la detección estuvo dominada por **anchor boxes** pre-definidas: Faster R-CNN (9 anchors/location), SSD, YOLOv2/v3 (k-means sobre el dataset), RetinaNet (FPN + focal loss + 9 anchors en $P_3$–$P_7$). Cada anchor actúa como *training sample* y como *regression prior*. El paradigma ganaba performance pero acumulaba problemas:

1. **Hiperparámetros sensibles.** RetinaNet muestra que variar escalas/aspect ratios puede cambiar **hasta 4 AP** en COCO. La grilla es ortogonal al dataset y debe re-tunearse para cada dominio.
2. **Generalización limitada.** Anchors optimizados para COCO fallan en escenas con aspect ratios fuera de distribución (texto largo, peatones verticales, anatomía médica). Hay que rediseñar para cada task.
3. **Memory blow-up y desbalance FG/BG.** Una imagen con lado corto 800 + FPN puede tener **>180k anchors**, casi todos negativos. Calcular IoU contra GT para todos durante training es costoso.
4. **Complejidad de assignment.** Etiquetar cada anchor (positivo / negativo / ignored) requiere umbrales de IoU — otra capa de hiperparámetros y otra fuente de varianza.

**CornerNet** (Law & Deng, ECCV 2018) había sido el primer intento anchor-free moderno detectando pares de esquinas (top-left, bottom-right) con heatmaps + embeddings, pero introdujo problemas propios: *corner pooling* custom, post-processing complejo para agrupar esquinas, backbone Hourglass-104 muy pesado. Logró 40.5 AP pero con un detector arquitectónicamente "raro".

El paper plantea explícitamente la pregunta: *"Can we solve object detection in the neat per-pixel prediction fashion, analogue to FCN for semantic segmentation?"* — y la responde afirmativamente, devolviendo la detección al paradigma FCN.

## Ideas principales

### Per-pixel prediction directo

Para una location $(x, y)$ en el feature map mapeada a la imagen original, si cae dentro de un GT box $B_i = (x_0, y_0, x_1, y_1, c)$ se considera **positiva**. Los regression targets son las **cuatro distancias al borde**:

$$
l^* = x - x_0, \quad t^* = y - y_0, \quad r^* = x_1 - x, \quad b^* = y_1 - y.
$$

Todas son no negativas por construcción. En inferencia se reconstruye el bbox invirtiendo: $\hat{x}_0 = x - \hat{l}$, $\hat{x}_1 = x + \hat{r}$, análogo para $y$.

Por location la red predice $C$ canales de clasificación + 4 canales de regresión + 1 canal de centerness = **85 outputs**, vs $9 \times (80 + 4) = 756$ de RetinaNet (**~9× menos**). La regresión usa $\exp(s_i \cdot x)$ por nivel para mapear a positivos y calibrar la magnitud.

**Dense positive sampling:** a diferencia de RetinaNet (un anchor es positivo solo si IoU ≥ 0.5), en FCOS **toda location dentro de un GT box es positiva**. Esto multiplica los samples de regresión y explica una parte de la ganancia en AP estricto (IoU=0.90).

### Multi-level FPN assignment

Igual que RetinaNet, FCOS usa **FPN sobre ResNet/ResNeXt** con niveles $P_3$–$P_7$ (strides 8, 16, 32, 64, 128). Cada nivel solo regresiona cajas en un rango de tamaño:

| Nivel | Stride | Rango $\max(l, t, r, b)$ |
|-------|--------|--------------------------|
| $P_3$ | 8      | $[0, 64]$                |
| $P_4$ | 16     | $[64, 128]$              |
| $P_5$ | 32     | $[128, 256]$             |
| $P_6$ | 64     | $[256, 512]$             |
| $P_7$ | 128    | $[512, \infty)$          |

**Esto resuelve la ambigüedad de overlap** — el problema histórico que había enterrado a DenseBox. La mayoría de overlaps reales suceden entre objetos de escalas muy diferentes (persona grande + bate pequeño), así que FPN los asigna a niveles distintos. La Tabla 2 del paper: sin FPN, **23.16%** de las muestras positivas son ambiguas; con FPN baja a **7.14%**, y filtrando ambigüedades intra-clase (que son benignas) a **3.75%**. El residuo se rompe asignando al GT box con **menor área**.

El **Best Possible Recall** tampoco resulta ser problema: FCOS con FPN logra **98.40 BPR**, comparable a RetinaNet con todos los low-quality matches.

### Center-ness branch

Después de FPN y dense sampling, FCOS aún quedaba 3-4 AP detrás de RetinaNet por un patrón consistente: **muchos bboxes con score alto pero IoU bajo con GT**, provenientes de locations lejos del centro del objeto. La intuición: una predicción desde la esquina tiene $(l, r)$ muy asimétrico (e.g., $l=5$, $r=195$); regresar cuatro distancias muy desbalanceadas es estadísticamente más difícil, así que la regresión es ruidosa, pero el feature local sí ve el objeto y entrega score de clasificación alto. Estos falsos positivos pasan NMS y degradan precision.

**Solución:** una rama de un solo layer paralela a clasificación que predice un escalar por location con target:

$$
\text{centerness}^* = \sqrt{\frac{\min(l^*, r^*)}{\max(l^*, r^*)} \cdot \frac{\min(t^*, b^*)}{\max(t^*, b^*)}}.
$$

Propiedades: rango $[0, 1]$, vale **1 en el centro exacto** ($l=r$, $t=b$) y **decae a 0** en las esquinas; el $\sqrt{\cdot}$ suaviza el decaimiento para hacerlo entrenable; es independiente de la clase. Se entrena con **BCE** contra ese target soft.

En **inferencia** se multiplica el score de clasificación por el centerness antes de NMS:

$$
\text{score}_{\text{final}} = \hat{p}_{\text{cls}} \times \hat{c}.
$$

Locations lejos del centro se aplastan y NMS las descarta.

**Ablation (Tabla 4 del paper):**

| Configuración                     | AP   | AP$_{50}$ | AP$_{75}$ |
|-----------------------------------|------|-----------|-----------|
| Sin center-ness                   | 33.5 | 52.6      | 35.2      |
| Center-ness desde regression vec  | 33.5 | 52.4      | 35.1      |
| **Center-ness branch separado**   | **37.1** | **55.9** | **39.8**  |

**+3.6 AP** solo por agregar la rama. Crucialmente, calcular el centerness desde el vector de regresión predicho no funciona — la rama dedicada es necesaria, probablemente porque aprende un proxy de "calidad de la regresión" que el vector solo no expone.

## Arquitectura

- **Backbone**: ResNet-50/101 o ResNeXt-32x8d-101 / 64x4d-101.
- **FPN**: niveles $P_3$ a $P_7$. $P_6$ y $P_7$ se obtienen con stride-2 conv sobre $P_5$ y $P_6$ (no sobre $C_5$ como en RetinaNet — pequeña diferencia que mejora ligeramente la performance).
- **Tres heads convolucionales compartidos entre niveles** (con escalar $s_i$ entrenable por nivel para calibrar la exponencial de regresión):
  - **Classification head**: 4 convs $3 \times 3$ con 256 canales + conv final de $C$ canales con **sigmoid** (no softmax — $C$ clasificadores binarios independientes, como RetinaNet).
  - **Regression head**: 4 convs + conv final de 4 canales con $\exp(s_i \cdot x)$.
  - **Center-ness branch**: un solo layer adicional sobre el regression head, con sigmoid.
- **GroupNorm** (Wu & He 2018) reemplaza BatchNorm en los heads — importante porque el batch size efectivo (16 imágenes en 8 GPUs) es pequeño y BatchNorm se vuelve ruidoso en ese régimen.
- **Loss**: focal loss para clasificación ($\alpha=0.25$, $\gamma=2$), **IoU loss** para regresión (UnitBox 2016, después reemplazada por **GIoU** en improvements), BCE para centerness. Suma sobre todas las locations de todos los niveles, normalizada por $N_{\text{pos}}$.

$$
L = \frac{1}{N_{\text{pos}}} \sum_{x,y} L_{\text{cls}}(p_{x,y}, c^*_{x,y}) + \frac{\lambda}{N_{\text{pos}}} \sum_{x,y} \mathbb{1}_{\{c^* > 0\}} \, L_{\text{reg}}(t_{x,y}, t^*_{x,y}) + L_{\text{ctr}}.
$$

con $\lambda = 1$.

**Training:** SGD, lr 0.01, momentum 0.9, weight decay 1e-4, 90k iters, decay ×0.1 en 60k y 80k, lado corto 800. Crucialmente, **mismos hiperparámetros que RetinaNet**, sin retuning específico para FCOS.

## Resultados

### Ablation FCOS vs RetinaNet (Tabla 3, minival ResNet-50-FPN)

| Setting                          | AP   | AP$_{50}$ | AP$_{75}$ |
|----------------------------------|------|-----------|-----------|
| RetinaNet ($C_5$)                | 35.9 | 56.0      | 38.2      |
| FCOS ($C_5$)                     | 36.3 | 54.8      | 38.7      |
| FCOS ($P_5$, NMS=0.60, GN)       | **37.1** | 55.9 | 39.8    |
| FCOS + improvements              | **38.6** | 57.4 | 41.4    |

FCOS base ya supera RetinaNet. Los improvements (centerness en regression branch, central sampling, GIoU, normalization de targets por stride) suman +1.5 AP "casi gratis".

### State of the art COCO test-dev (Tabla 5, single-model / single-scale)

| Método                  | Backbone                  | AP    | AP$_{50}$ | AP$_{75}$ | AP$_S$ | AP$_M$ | AP$_L$ |
|-------------------------|---------------------------|-------|-----------|-----------|--------|--------|--------|
| Faster R-CNN + FPN      | ResNet-101-FPN            | 36.2  | 59.1      | 39.0      | 18.2   | 39.0   | 48.2   |
| RetinaNet               | ResNet-101-FPN            | 39.1  | 59.1      | 42.3      | 21.8   | 42.7   | 50.2   |
| CornerNet               | Hourglass-104             | 40.5  | 56.5      | 43.1      | 19.4   | 42.7   | 53.9   |
| **FCOS**                | ResNet-101-FPN            | 41.5  | 60.7      | 45.0      | 24.4   | 44.8   | 51.6   |
| **FCOS**                | ResNeXt-64x4d-101-FPN     | 43.2  | 62.8      | 46.6      | 26.5   | 46.2   | 53.3   |
| **FCOS w/ improvements**| ResNeXt-64x4d-101-FPN     | **44.7** | 64.1   | 48.4      | 27.6   | 47.5   | 55.6   |

**+2.4 AP** sobre RetinaNet con el mismo backbone, **+4.2 AP** sobre CornerNet con menor complejidad de post-processing.

### FCOS como RPN (Tabla 6)

| Método                         | # samples | AR$^{100}$ | AR$^{1k}$ |
|--------------------------------|-----------|------------|-----------|
| RPN w/ FPN + GN                | ~200k     | 44.7       | 56.9      |
| **FCOS w/ FPN + GN como RPN**  | **~66k**  | **52.8**   | **60.3**  |

Mejora absoluta de **+8.1% AR$^{100}$** usando ~3× menos samples. FCOS sirve también como **proposal generator** para arquitecturas two-stage.

## Por qué FCOS es backbone de ABCNet

Esta es la conexión clave para la clase 21. **ABCNet (Liu et al., CVPR 2020)** — el detector/spotter de texto curvo de la clase — está construido sobre FCOS literal. El diff es **un solo layer**:

| Componente              | FCOS                              | ABCNet                                                    |
|-------------------------|-----------------------------------|-----------------------------------------------------------|
| Backbone + FPN          | ResNet-50/101 + $P_3$–$P_7$       | ResNet-50 + $P_3$–$P_5$                                   |
| Classification head     | 80 clases (COCO)                  | 1 clase (texto/no-texto)                                  |
| **Regression head**     | **4 canales** $(l, t, r, b)$      | **16 canales** (8 control points $xy$ de 2 curvas Bezier) |
| Center-ness branch      | Sí                                | Sí (idéntica)                                             |
| Asignación multi-level  | Por $\max(l, t, r, b)$            | Por bbox que envuelve la curva                            |
| Loss                    | Focal + IoU + BCE                 | Focal + Smooth L1 sobre control points + BCE              |

Tres razones específicas por las que FCOS encaja perfecto en text spotting:

- **Anchor-free escala a annotations no-rectangulares.** Si tu output no son 4 valores sino N (16 para Bezier, 14 para polígonos, $7 \times 2$ para keypoints), anchor-based se rompe — habría que diseñar "anchor polígonos" o "anchor curvas". Per-pixel regression desde el centro es trivialmente generalizable: solo cambias el número de canales del regression head.
- **Centerness es ideal para texto.** Aspect ratios extremos (texto largo y delgado) amplifican las predicciones malas lejos del centro. Centerness las filtra exactamente donde más hace falta.
- **Hiperparámetros transferibles.** ABCNet hereda los hiperparámetros de COCO casi sin retuning para ICDAR/TotalText.

Por eso post-2019 prácticamente todo text spotter (ABCNet, ABCNetv2, TextFuseNet, Mask TextSpotter v3) hereda el patrón FCOS + centerness.

## Sucesores

FCOS abrió la puerta a toda una generación de detectores:

- **CenterNet (Zhou et al., 2019)** — "Objects as Points". Predice un heatmap de centros + offsets sub-pixel. Más extremo que FCOS (un punto por objeto, no todos los interiores). Útil para tiempo real (~52 FPS).
- **ATSS (Zhang et al., CVPR 2020)** — "Adaptive Training Sample Selection". Mostró que la diferencia esencial entre FCOS y RetinaNet **no es anchor-free vs anchor-based** sino *cómo se asignan samples positivas*. Propone assignment adaptativo basado en estadísticas de IoU. El paradigma se reinterpreta: FCOS es RetinaNet con 1 anchor + dense positive sampling + centerness.
- **PAA (Kim & Lee, ECCV 2020)** — "Probabilistic Anchor Assignment". Modela el assignment como mezcla de gaussianas + EM.
- **DETR (Carion et al., ECCV 2020)** — Transformer + Hungarian matching. Rompe con anchors **y** con NMS **y** con dense prediction. Trade-off: convergencia mucho más lenta (500 epochs vs 90k iters), pero abre la era end-to-end set prediction.
- **FCOS3D, CondInst, SOLOv2, DirectPose, BorderDet** — extensiones del patrón FCOS a 3D detection, instance segmentation, pose estimation y representaciones alternativas de bbox.

## Limitaciones

- **NMS sigue siendo necesaria.** FCOS no es end-to-end en el sentido de DETR.
- **Hiperparámetros reducidos pero no eliminados.** Score threshold (0.05), NMS threshold (0.6), y los rangos $(m_i)$ del multi-level assignment siguen siendo hiperparámetros heredados de FPN. ATSS y PAA después demostraron que el assignment adaptativo es estrictamente mejor.
- **Convergencia inicial lenta** comparada con anchor-based con dense pre-defined priors — los autores compensan con weight init cuidadoso y schedule estándar, pero los primeros iters el detector "no sabe dónde mirar".
- **Overlap ambiguity parcial.** El tie-breaker por mínima área es heurístico. Funciona para el 3.75% residual de COCO pero puede fallar en dominios con muchos overlaps inter-clase del mismo tamaño.
- **Sin teoría del centerness.** La fórmula exacta es ingeniería empírica — variantes (centerness², centerness en regression branch) muestran que la forma específica no es única. Funciona porque captura "qué tan central estoy", pero no hay derivación que diga "esta es **la** función correcta".

## Por qué importa hoy

FCOS no es solo "RetinaNet sin anchors". Es un **cambio de paradigma** hacia detección como dense prediction, alineado con FCN, segmentación, depth, keypoints y counting. Las consecuencias prácticas hoy:

- **Paradigma dominante post-2019.** Casi todos los detectores modernos (YOLOX, YOLOv6, YOLOv8, RTMDet, RT-DETR para inferencia rápida, NanoDet) usan heads anchor-free FCOS-style con variantes de centerness o quality scores.
- **Base de text spotters.** Toda la familia ABCNet + Mask TextSpotter v3 hereda directamente.
- **Generalización a outputs complejos.** El patrón "FCOS + N canales custom" es la receta estándar para extender detección a curvas Bezier, polígonos, keypoints, voxels 3D, máscaras.
- **Pedagógicamente claro.** Es el detector más fácil de explicar: clasifica el píxel, regresiona cuatro distancias, suprime los píxeles no-centrales. Sin anchors, sin matching, sin RoIs.

Es uno de esos papers que en retrospectiva parecen obvios — pero requirió coraje argumentar contra el "de facto standard" anchor-based y demostrar empíricamente que la simplicidad gana.

## Notas y enlaces

- **Fundamentos:**
  - [Anchor-free detection]({{< relref "/fundamentos/anchor-free-detection" >}}) — el paradigma del que FCOS es referente.
  - [Detección de objetos]({{< relref "/fundamentos/deteccion-de-objetos" >}}) — contexto y métricas (AP, AR, IoU).
  - [Scene text recognition]({{< relref "/fundamentos/scene-text-recognition" >}}) — dominio donde FCOS habilitó ABCNet.
- **Papers relacionados:**
  - [ABCNet (Liu 2020)]({{< relref "/papers/abcnet-liu-2020" >}}) — extiende FCOS a text spotting curvo con head Bezier de 16 canales.
  - [FPN (Lin 2017)]({{< relref "/papers/fpn-lin-2017" >}}) — la columna vertebral multi-escala que FCOS hereda.
  - [GIoU (Rezatofighi 2019)]({{< relref "/papers/giou-rezatofighi-2019" >}}) — mejor regression loss, adoptada en los improvements.
  - [Faster R-CNN (Ren 2015)]({{< relref "/papers/faster-rcnn-ren-2015" >}}) — el contrapunto anchor-based contra el que FCOS se posiciona.
- **Clase:** [Clase 21 — Scene Text Recognition con ABCNet]({{< relref "/clases/clase-21" >}}).
