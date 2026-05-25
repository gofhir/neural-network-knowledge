---
title: "FCOS: Fully Convolutional One-Stage Object Detection"
authors: ["Zhi Tian", "Chunhua Shen", "Hao Chen", "Tong He"]
year: 2019
venue: "ICCV 2019"
slug: fcos-tian-2019
arxiv: "1904.01355"
affiliation: "The University of Adelaide, Australia"
tags: ["object-detection", "anchor-free", "fpn", "one-stage", "center-ness", "fcos", "abcnet-prereq"]
---

# FCOS: Fully Convolutional One-Stage Object Detection

> **Cita:** Tian, Z., Shen, C., Chen, H., & He, T. (2019). *FCOS: Fully Convolutional One-Stage Object Detection*. ICCV 2019. arXiv:1904.01355.

## 1. Resumen ejecutivo

FCOS (Tian et al., ICCV 2019) propone el **primer detector one-stage anchor-free competitivo** con los detectores anchor-based de su época. La idea central es reformular la detección de objetos como **predicción densa por píxel** — análoga a la segmentación semántica — donde cada *location* $(x, y)$ del feature map predice directamente:

1. La clase del objeto al que pertenece (o background),
2. Cuatro offsets continuos $(l^*, t^*, r^*, b^*)$ desde la location hacia los cuatro lados del bounding box, y
3. Un escalar de **center-ness** que mide qué tan centrada está la location respecto al objeto al cual regresiona.

Esta formulación elimina por completo los anchor boxes y todos los hiperparámetros que conllevan (escalas, aspect ratios, umbrales de IoU). En MS-COCO `test-dev`, FCOS con ResNeXt-64x4d-101-FPN alcanza **44.7 AP** en single-model/single-scale, superando a RetinaNet (39.1 AP) con el mismo backbone y a CornerNet (40.5 AP) con menor complejidad de post-procesamiento. Las tres palancas que lo hacen funcionar son: **(a)** asignación a niveles de FPN por rango de tamaño, que resuelve la ambigüedad de overlaps; **(b)** **center-ness branch**, que suprime predicciones de baja calidad lejos del centro (ganancia de ~3-4 AP); y **(c)** uso de **todos los píxeles dentro del bbox** como muestras positivas, no solo los que matchean un anchor con IoU alto.

Para el curso, FCOS es **prerrequisito directo de ABCNet (Liu 2020, clase 21)**: ABCNet conserva backbone, FPN, center-ness y clasificación, y simplemente reemplaza la regresión de 4 offsets por **16 coordenadas de control points Bezier** para texto curvo. Sin entender FCOS, ABCNet aparenta ser magia; con FCOS claro, ABCNet se vuelve un *extension trivial*.

## 2. Contexto histórico: la era anchor-based y sus dolores

### 2.1. Detectores anchor-based

Entre 2015 y 2018 la detección estuvo dominada por una familia con un denominador común: **anchor boxes pre-definidas**. Los hitos:

- **Faster R-CNN (Ren et al., NeurIPS 2015)** — RPN con 9 anchors por location (3 escalas × 3 aspect ratios) + segunda etapa para clasificación fina.
- **SSD (Liu et al., ECCV 2016)** — Single-shot multibox con anchors en múltiples feature maps.
- **YOLOv2/v3 (Redmon & Farhadi, CVPR 2017 / 2018)** — k-means clustering de anchors sobre el dataset.
- **RetinaNet (Lin et al., ICCV 2017)** — FPN + focal loss + 9 anchors por location en $P_3$–$P_7$.

Un anchor box es simplemente un *reference rectangle* pre-definido en cada location del feature map. El detector clasifica cada anchor (FG/BG, y eventualmente clase) y regresiona offsets refinados sobre las coordenadas del anchor. Los anchors actúan como **training samples** y como **regresión priors**.

### 2.2. Los cuatro dolores del paradigma anchor-based

El paper enumera cuatro problemas concretos que hacen costoso este enfoque:

1. **Hiperparámetros sensibles**. RetinaNet muestra que variar escalas/aspect ratios puede cambiar hasta **4 AP** en COCO. La grilla de anchors es ortogonal al dataset y debe re-tunearse para cada nuevo dominio.

2. **Generalización limitada**. Anchors optimizados en COCO no funcionan en escenas con aspect ratios fuera de distribución (texto largo, peatones verticales, objetos médicos). Hay que rediseñar para cada task.

3. **Memory blow-up y desbalance FG/BG**. Una imagen de lado corto 800 con FPN puede tener **>180k anchors**. La inmensa mayoría son negativos, lo que exacerba el desbalance que motivó focal loss. Calcular IoU contra ground truth para todos esos anchors durante training tiene costo no trivial.

4. **Complejidad de assignment**. Etiquetar cada anchor como positivo/ignored/negativo requiere computar IoU con todos los GT bboxes y aplicar umbrales (típicamente $[0.5, 1]$ positivo, $[0, 0.4]$ negativo, intermedio ignored). Es otra capa de hiperparámetros y otra fuente de varianza.

### 2.3. CornerNet (Law & Deng, ECCV 2018) — el primer anchor-free moderno

CornerNet propuso detectar pares de esquinas (top-left, bottom-right) usando heatmaps + embeddings de agrupación. Eliminó anchors pero introdujo problemas propios:

- **Corner pooling** custom — operador caro y específico.
- **Post-processing complejo** — agrupar esquinas que pertenecen al mismo objeto vía embedding distance.
- Backbone Hourglass-104 muy pesado.

CornerNet logró 40.5 AP en COCO test-dev pero con un detector "raro" arquitectónicamente. FCOS llegó un año después con una propuesta mucho más limpia: **detectar el objeto desde sus puntos interiores**, no desde sus esquinas. Eso permitió reutilizar arquitecturas familiares (FPN + heads convolucionales) sin operadores custom.

### 2.4. Antecedentes anchor-free olvidados

El paper rescata dos predecesores que habían quedado en el margen:

- **DenseBox (Huang et al., arXiv 2015)** — Predicción per-pixel de 4 distancias y clase. Usado con éxito en face detection y scene text, pero descartado para detección genérica por dos razones que FCOS demolerá: (a) presunta dificultad con bboxes overlapped, y (b) bajo recall. DenseBox además requería crop+resize de imágenes a escala fija (image pyramids), violando la filosofía "compute all convolutions once" de FCN.
- **YOLOv1 (Redmon et al., CVPR 2016)** — Predecía bboxes desde el centro de celdas de una grilla burda. Eliminado en YOLOv2 a favor de anchors, ya que YOLOv1 sufría bajo recall (solo los puntos cerca del centro generaban predicciones).
- **UnitBox (Yu et al., ACM MM 2016)** — Introdujo IoU loss para regresión de bboxes en face detection, en un contexto per-pixel similar a DenseBox. FCOS hereda este loss directamente.

FCOS demuestra que **multi-level FPN prediction** resuelve la ambigüedad de overlap (problema (a)) y que usar **todos los puntos interiores** del bbox resuelve el bajo recall (problema (b)). Esto desbloquea retroactivamente la línea anchor-free per-pixel: lo que DenseBox y YOLOv1 intentaron en 2015-2016 funciona, pero les faltaban FPN (que no existía hasta 2017) y centerness (que es la contribución original de FCOS). Es un caso interesante de "el método correcto fue propuesto demasiado temprano".

### 2.5. La pregunta detrás del paper

El abstract plantea explícitamente: *"Can we solve object detection in the neat per-pixel prediction fashion, analogue to FCN for semantic segmentation, for example?"* Esa pregunta es el corazón del paper. La respuesta afirmativa unifica detección con segmentación, depth estimation, keypoint detection y counting — todas tareas resueltas por FCNs con dense prediction. Detección había quedado como la excepción aberrante por su dependencia de anchors. FCOS la reintegra al paradigma FCN.

## 3. Idea central de FCOS: detección como predicción per-pixel

### 3.1. Formulación matemática del target

Sea $F_i \in \mathbb{R}^{H \times W \times C}$ el feature map en el nivel $i$ de la backbone con stride total $s$ respecto a la imagen de entrada. Los ground truth bboxes de una imagen son $\{B_i\}$ con $B_i = (x_0^{(i)}, y_0^{(i)}, x_1^{(i)}, y_1^{(i)}, c^{(i)}) \in \mathbb{R}^4 \times \{1, \ldots, C\}$, donde $(x_0, y_0)$ y $(x_1, y_1)$ son las esquinas top-left y bottom-right y $c^{(i)}$ es la clase.

Para una location $(x, y)$ en el feature map, se mapea al punto $\left(\lfloor s/2 \rfloor + xs,\ \lfloor s/2 \rfloor + ys\right)$ en la imagen original, que está cerca del centro del receptive field. Si ese punto cae dentro de algún ground truth box $B_i$, la location se considera **positiva** con etiqueta $c^* = c^{(i)}$. De lo contrario es negativa con $c^* = 0$.

Los **regression targets** de la location positiva son las cuatro distancias desde el punto a los lados del bbox:

$$
l^* = x - x_0^{(i)}, \quad t^* = y - y_0^{(i)}, \quad r^* = x_1^{(i)} - x, \quad b^* = y_1^{(i)} - y. \tag{1}
$$

Notación: $l$ = left, $t$ = top, $r$ = right, $b$ = bottom. Todos son no-negativos por construcción (la location está dentro del bbox). En inferencia, dado el punto $(x, y)$ y los $(\hat{l}, \hat{t}, \hat{r}, \hat{b})$ predichos, se reconstruye el bbox invirtiendo (1):

$$
\hat{x}_0 = x - \hat{l}, \quad \hat{y}_0 = y - \hat{t}, \quad \hat{x}_1 = x + \hat{r}, \quad \hat{y}_1 = y + \hat{b}.
$$

### 3.2. Outputs de red

La red predice por location:

- **Classification:** $C$ canales con sigmoid (no softmax) — $C$ clasificadores binarios independientes, siguiendo RetinaNet.
- **Regression:** 4 canales con $\exp(s_i \cdot x)$ aplicado al output, donde $s_i$ es un escalar entrenable por nivel del FPN. La exponencial mapea $\mathbb{R} \to (0, \infty)$ (los targets son positivos); el escalar $s_i$ "calibra" automáticamente el rango de magnitudes que cada nivel del FPN debe regresar.
- **Center-ness:** 1 canal con sigmoid (sección 5).

**Conteo de outputs.** FCOS produce un vector de 80D para clasificación + 4D para regression + 1D para center-ness = **85 outputs por location**. RetinaNet con 9 anchors produce $9 \times (80 + 4) = 756$ outputs por location — aproximadamente **9× más**. Eso reduce memoria del último layer convolucional ~9× y simplifica el computo de IoU/matching.

### 3.3. Muestreo positivo agresivo

Una diferencia conceptual con anchor-based: en RetinaNet, un anchor es positivo solo si su IoU con un GT box es $\geq 0.5$, lo cual filtra muchos píxeles "interiores" pero mal posicionados. En FCOS, **toda location dentro de un GT box es muestra positiva**. Esto multiplica el número de samples por las que pasa el gradiente de regresión, lo que el paper sugiere como una de las razones por las que FCOS regresiona bboxes más precisos (sección 7 del paper, precision-recall a IoU=0.90 mostrando ganancia de 2.7 AP sobre RetinaNet).

## 4. Multi-level FPN prediction: el truco que destrabó anchor-free

### 4.1. Arquitectura FPN

Igual que RetinaNet, FCOS usa **Feature Pyramid Network** sobre la backbone (ResNet/ResNeXt/HRNet):

| Nivel | Origen | Stride | Rango de tamaño $[m_{i-1}, m_i]$ |
|-------|--------|--------|----------------------------------|
| $P_3$ | de $C_3$ vía $1\times 1$ conv + top-down | 8 | $[0, 64]$ |
| $P_4$ | de $C_4$ | 16 | $[64, 128]$ |
| $P_5$ | de $C_5$ | 32 | $[128, 256]$ |
| $P_6$ | conv stride-2 sobre $P_5$ | 64 | $[256, 512]$ |
| $P_7$ | conv stride-2 sobre $P_6$ | 128 | $[512, \infty)$ |

Detalle implementación: FCOS aplica el stride-2 sobre $P_5$ (no sobre $C_5$ como en RetinaNet) — pequeña diferencia que mejora un poco la performance. Los heads convolucionales son **compartidos entre niveles**, pero con un escalar $s_i$ entrenable por nivel para calibrar la exponencial de regresión.

### 4.2. Asignación de location a nivel

La regla es simple: una location positiva en el nivel $i$ debe satisfacer

$$
m_{i-1} \leq \max(l^*, t^*, r^*, b^*) \leq m_i.
$$

Si $\max(l^*, t^*, r^*, b^*) > m_i$ o $< m_{i-1}$, la location se marca como negativa en ese nivel, sin importar que esté dentro del bbox. Los thresholds son $m_2 = 0$, $m_3 = 64$, $m_4 = 128$, $m_5 = 256$, $m_6 = 512$, $m_7 = \infty$.

### 4.3. Resolución de la ambigüedad de overlap

El problema clásico de DenseBox: si dos bboxes se solapan, un píxel interior cae dentro de ambos — ¿a cuál regresionar? Sin solución, esto envenena el gradiente.

**Insight clave**: la mayoría de overlaps reales suceden entre objetos de **escalas muy diferentes** (e.g., una persona grande con un bate pequeño en su mano). Multi-level FPN asigna esos objetos a niveles distintos del FPN, así que en cada nivel la ambigüedad casi desaparece. La Tabla 2 del paper cuantifica:

- Sin FPN (solo $P_4$): 23.16% de muestras positivas son ambiguas.
- Con FPN: solo **7.14%**, y filtrando ambigüedades dentro de la misma clase (que son benignas: predecir cualquier instancia es correcto si la clase coincide), baja a **3.75%**.

Para el residuo, FCOS aplica un tie-breaker simple: **asignar la location al GT box con menor área**. Heurística defendible — el objeto pequeño es el que tiene menos "área de evidencia" y se beneficia más de ese píxel.

### 4.4. Best Possible Recall (BPR)

Otra preocupación histórica con FCN-based detectors era el BPR — el techo teórico de recall. Tabla 1 del paper:

- RetinaNet con FPN + low-quality matches (IoU $\geq 0.4$): 90.92 BPR; con **todos** los low-quality matches: 99.23 BPR.
- FCOS sin FPN: 95.55 BPR.
- **FCOS con FPN: 98.40 BPR** — comparable a RetinaNet con todos los low-quality matches.

Conclusión empírica: el BPR no es un problema de FCOS. La intuición "los strides grandes hacen que objetos pequeños se pierdan" no se materializa en COCO.

## 5. Center-ness branch: el detalle que cierra la brecha con anchor-based

### 5.1. Motivación

Después de aplicar FPN y entrenar la red base, FCOS aún quedaba unos 3-4 AP detrás de RetinaNet. El análisis mostró que el detector producía **muchos bboxes con confidence alta pero IoU bajo con el GT** — falsos positivos visualmente plausibles pero con regresión imprecisa. Estos típicamente provenían de locations **lejos del centro** del objeto, donde la regresión a los 4 lados es asimétrica y poco confiable.

La intuición geométrica: un punto cerca de la esquina de un bbox tiene un par $(\min(l, r), \max(l, r))$ muy desbalanceado — por ejemplo $l = 5$, $r = 195$ para un bbox de 200 píxeles de ancho. Predecir cuatro distancias asimétricas con esa magnitud es estadísticamente más difícil que predecir cuatro distancias balanceadas (centro: $l \approx r \approx 100$). La red aprende a regresionar bien desde el centro pero peor desde los bordes, así que las predicciones de bordes tienen ruido alto, lo que se traduce en IoU bajo. Estas predicciones, si reciben score de clasificación alto (porque el feature local sí ve el objeto), pasan el filtro de NMS y degradan precision.

### 5.2. Definición

FCOS añade una **rama de un solo layer**, paralela a clasificación, que predice un escalar de center-ness por location. El target de center-ness para una location positiva es:

$$
\text{centerness}^* = \sqrt{\frac{\min(l^*, r^*)}{\max(l^*, r^*)} \times \frac{\min(t^*, b^*)}{\max(t^*, b^*)}}. \tag{3}
$$

Propiedades:

- **Rango $[0, 1]$**. En el centro exacto, $l^* = r^*$ y $t^* = b^*$, así que centerness $= \sqrt{1 \cdot 1} = 1$. En la esquina del bbox, $\min(l^*, r^*) \to 0$, así que centerness $\to 0$.
- **Decae suave**: el $\sqrt{\cdot}$ mitiga la caída para hacer el target más entrenable (un decaimiento abrupto sería casi binario y poco informativo).
- **Independiente de la clase**: es una propiedad geométrica pura de la location dentro del bbox.

### 5.3. Loss y uso en inferencia

Center-ness se entrena con **binary cross-entropy** con el target soft definido por (3):

$$
\mathcal{L}_{\text{ctr}} = -\text{centerness}^* \log(\hat{c}) - (1 - \text{centerness}^*) \log(1 - \hat{c}),
$$

donde $\hat{c}$ es la predicción tras sigmoid. La loss se suma a la loss total (sección 6).

En **inference**, FCOS multiplica el score de clasificación por el centerness predicho **antes de NMS**:

$$
\text{score}_{\text{final}} = \hat{p}_{\text{cls}} \times \hat{c}.
$$

Locations lejos del centro tienen $\hat{c} \to 0$, así que su score final se aplasta y NMS las descarta. Locations cerca del centro mantienen el score alto. El efecto neto se ve en la Figura 7 del paper: un scatter plot de classification score vs IoU con GT antes y después de multiplicar por centerness muestra que las predicciones de bajo IoU pero alto score se desplazan a la izquierda (score reducido).

### 5.4. Ablation (Tabla 4 del paper)

| Configuración | AP | AP$_{50}$ | AP$_{75}$ |
|---------------|------|-----------|-----------|
| Sin center-ness | 33.5 | 52.6 | 35.2 |
| Center-ness desde regresión vector | 33.5 | 52.4 | 35.1 |
| **Center-ness branch separado** | **37.1** | **55.9** | **39.8** |

Conclusión: **+3.6 AP** solo por agregar esa rama. Y crucialmente, calcular center-ness desde el vector de regresión predicho (sin rama dedicada) **no funciona** — la rama explícita es necesaria. La hipótesis: la rama dedicada aprende un proxy de "calidad de la regresión" que el vector regresionado por sí solo no expone.

### 5.5. Center-ness vs IoUNet (Jiang et al. 2018)

IoUNet entrena una red separada que predice IoU entre bboxes predichos y GT. Center-ness comparte el objetivo (suprimir baja calidad) pero es:

- **Mucho más simple**: un layer adicional vs. una red separada.
- **Entrenado conjuntamente** con el detector.
- **No usa el bbox predicho como input**, accede directamente al feature de la location.

## 6. Loss completa

$$
L(\{p_{x,y}\}, \{t_{x,y}\}) = \frac{1}{N_{\text{pos}}} \sum_{x,y} L_{\text{cls}}(p_{x,y}, c^*_{x,y}) + \frac{\lambda}{N_{\text{pos}}} \sum_{x,y} \mathbb{1}_{\{c^*_{x,y} > 0\}} L_{\text{reg}}(t_{x,y}, t^*_{x,y}). \tag{2}
$$

Más el término de center-ness sumado afuera. Componentes:

- **$L_{\text{cls}}$**: **focal loss** (Lin et al. 2017), igual que RetinaNet. Maneja el desbalance FG/BG masivo. $\alpha = 0.25$, $\gamma = 2$.
- **$L_{\text{reg}}$**: **IoU loss** (Yu et al. UnitBox 2016): $1 - \text{IoU}(\hat{\mathbf{t}}, \mathbf{t}^*)$. Más tarde se mejora a **GIoU** (Rezatofighi et al. CVPR 2019), que penaliza la diferencia entre el área de unión y el rectángulo envolvente.
- **$L_{\text{ctr}}$**: BCE como descrito.
- **$\lambda = 1$**: balance.
- **$N_{\text{pos}}$**: número de samples positivas para normalizar.
- **$\mathbb{1}_{\{c^*_{x,y} > 0\}}$**: indicador de que la location es positiva (regresión solo sobre positivas).

La suma corre sobre **todas las locations de todos los niveles FPN**.

## 7. Resultados en COCO

### 7.1. FCOS vs RetinaNet (ablation, miniva ResNet-50-FPN, Tabla 3)

| Setting | C5/P5 | GN | NMS thr | AP | AP$_{50}$ | AP$_{75}$ |
|---------|-------|----|---------|------|-----------|-----------|
| RetinaNet | $C_5$ | | 0.50 | 35.9 | 56.0 | 38.2 |
| FCOS | $C_5$ | | 0.50 | 36.3 | 54.8 | 38.7 |
| FCOS | $P_5$ | | 0.50 | 36.4 | 54.9 | 38.8 |
| FCOS | $P_5$ | | 0.60 | 36.5 | 54.5 | 39.2 |
| FCOS | $P_5$ | ✓ | 0.60 | **37.1** | 55.9 | 39.8 |
| **+ improvements** | $P_5$ | ✓ | 0.60 | **38.6** | 57.4 | 41.4 |

FCOS base con GroupNorm + NMS=0.60 ya supera RetinaNet (37.1 vs 35.9). Con improvements (center-ness en regression branch, central sampling, GIoU loss, target normalization) llega a 38.6.

### 7.2. State of the art en COCO test-dev (Tabla 5)

| Método | Backbone | AP | AP$_{50}$ | AP$_{75}$ | AP$_S$ | AP$_M$ | AP$_L$ |
|--------|----------|------|-----------|-----------|--------|--------|--------|
| Faster R-CNN + FPN | ResNet-101-FPN | 36.2 | 59.1 | 39.0 | 18.2 | 39.0 | 48.2 |
| YOLOv2 | DarkNet-19 | 21.6 | 44.0 | 19.2 | 5.0 | 22.4 | 35.5 |
| SSD513 | ResNet-101 | 31.2 | 50.4 | 33.3 | 10.2 | 34.5 | 49.8 |
| RetinaNet | ResNet-101-FPN | 39.1 | 59.1 | 42.3 | 21.8 | 42.7 | 50.2 |
| CornerNet | Hourglass-104 | 40.5 | 56.5 | 43.1 | 19.4 | 42.7 | 53.9 |
| FSAF | ResNeXt-64x4d-101-FPN | 42.9 | 63.8 | 46.3 | 26.6 | 46.2 | 52.7 |
| **FCOS** | ResNet-101-FPN | 41.5 | 60.7 | 45.0 | 24.4 | 44.8 | 51.6 |
| **FCOS** | HRNet-W32-5l | 42.0 | 60.4 | 45.3 | 25.4 | 45.0 | 51.0 |
| **FCOS** | ResNeXt-32x8d-101-FPN | 42.7 | 62.2 | 46.1 | 26.0 | 45.6 | 52.6 |
| **FCOS** | ResNeXt-64x4d-101-FPN | 43.2 | 62.8 | 46.6 | 26.5 | 46.2 | 53.3 |
| **FCOS w/ improvements** | ResNeXt-64x4d-101-FPN | **44.7** | 64.1 | 48.4 | 27.6 | 47.5 | 55.6 |

FCOS supera RetinaNet por 2.4 AP en mismo backbone (ResNet-101-FPN: 41.5 vs 39.1) y a CornerNet por 4.2 AP con menor complejidad. Con todos los improvements y ResNeXt-64x4d-101, alcanza **44.7 AP** — competitivo con detectores two-stage modernos y muy por encima de YOLOv3/SSD.

### 7.3. FCOS como RPN (Tabla 6)

FCOS también puede usarse como Region Proposal Network reemplazando los anchors del RPN tradicional:

| Método | # samples | AR$^{100}$ | AR$^{1k}$ |
|--------|-----------|------------|-----------|
| RPN w/ FPN + GN | ~200k | 44.7 | 56.9 |
| FCOS w/ FPN w/o center-ness | ~66k | 48.0 | 59.3 |
| **FCOS w/ FPN + GN** | **~66k** | **52.8** | **60.3** |

Mejora absoluta de **8.1% AR$^{100}$** y **3.4% AR$^{1k}$** sobre RPN+FPN, usando ~3× menos samples. Esto valida que FCOS no solo sirve como detector standalone, sino también como **proposal generator** para arquitecturas two-stage.

## 8. Detalles de implementación

### 8.1. Heads compartidas + GroupNorm

Los heads (4 convs $H \times W \times 256$ antes de la última capa) se comparten entre los cinco niveles del FPN, lo cual:

- **Reduce parámetros** vs heads por nivel.
- **Mejora performance** según la ablation — actúa como una forma de regularización implícita.

Como los rangos de regresión por nivel son distintos ($[0, 64]$ para $P_3$, $[64, 128]$ para $P_4$, etc.), un mismo exp() no es adecuado. La solución: $\exp(s_i \cdot x)$ con $s_i$ entrenable por nivel.

**GroupNorm** (Wu & He, ECCV 2018) reemplaza BatchNorm en los heads. Esto es importante porque el batch size efectivo de FCOS es pequeño (16 imágenes en 8 GPUs), un régimen donde BatchNorm se vuelve ruidoso. GroupNorm es independiente del batch size.

### 8.2. Training

- **Optimizer**: SGD, lr inicial 0.01, momentum 0.9, weight decay 0.0001.
- **Schedule**: 90k iters, decay ×0.1 en 60k y 80k.
- **Batch size**: 16 imágenes.
- **Input**: lado corto 800, lado largo $\leq 1333$.
- **Backbone**: inicializado con pesos ImageNet pre-trained.
- **Heads nuevos**: inicialización aleatoria estándar.

### 8.3. Inference

1. Forward pass → scores $p_{x,y}$, regresión $\mathbf{t}_{x,y}$, centerness $\hat{c}_{x,y}$ por location.
2. Filter: locations con $p_{x,y} > 0.05$ (igual que RetinaNet).
3. Multiplicar $p_{x,y} \times \hat{c}_{x,y}$.
4. Invertir (1) para obtener bboxes.
5. Per-level top-k (typically 1000) → multi-class NMS con threshold 0.6.

Nota importante: el mismo set de hiperparámetros de RetinaNet (lr, NMS thr) funciona out-of-the-box. El paper enfatiza que **no tunearon nada para FCOS** y que probablemente hay AP extra disponible con tuning específico.

### 8.4. Improvements post-submisión

Cuatro tweaks "casi gratis" que llevaron de 37.1 a 38.6 AP en miniva:

1. **Center-ness en regression branch** (no classification): +0.3 AP. La regresión expone mejor la geometría de la bbox.
2. **Central sampling**: solo locations en el centro 1.5× stride del bbox cuentan como positivas. +0.7 AP.
3. **GIoU loss** (Rezatofighi 2019) reemplazando IoU loss: +0.2 AP.
4. **Normalization de regression targets**: dividir $(l^*, t^*, r^*, b^*)$ por el stride del nivel del FPN antes del loss. +0.3 AP. Estabiliza la escala de la regresión a través de niveles.

## 9. Por qué FCOS es prerrequisito para STR (Clase 21)

Esta es la conexión clave para el curso. **ABCNet (Liu et al., CVPR 2020)** — el detector/spotter de texto curvo que se estudia en clase 21 — está construido sobre FCOS literal. Específicamente:

### 9.1. ABCNet = FCOS + 16-channel Bezier head

ABCNet reemplaza el regression head de FCOS (4 canales $(l, t, r, b)$) por un **head de 16 canales** que predice 8 puntos de control $(x, y)$ de dos curvas Bezier cúbicas (cuatro puntos por curva, una para cada lado del texto curvo). Todo lo demás se conserva:

- **Backbone + FPN**: idéntico.
- **Classification branch**: idéntico (binario texto/no-texto en lugar de 80 clases).
- **Center-ness branch**: **idéntico** — clave porque el texto tiene asimetría centro-periferia pronunciada y centerness ayuda a suprimir falsas detecciones en los bordes.
- **Loss**: focal loss + regression (ahora L1 sobre control points) + centerness BCE.
- **Asignación multi-level FPN**: idéntica.

ABCNet añade después una rama de recognition (BezierAlign + atención sobre la curva muestreada). Pero la **detección** es FCOS modificado.

### 9.2. Por qué centerness se volvió estándar en text spotting

Los detectores de texto post-2019 (ABCNet, TextFuseNet, Mask TextSpotter v3, ABCNetv2) heredaron centerness por dos razones:

1. **Texto largo y delgado**: el aspect ratio extremo amplifica las predicciones de baja calidad lejos del centro. Centerness las filtra.
2. **No anchors**: el texto curvo o de aspect ratio variable es exactamente donde el diseño de anchors falla. Anchor-free + centerness es la combinación natural.

### 9.3. Anchor-free escala a annotations complejas

El gran insight estructural: si tu output no son 4 valores sino N (16 para Bezier de ABCNet, 14 para polígonos arbitrarios, 7×2 para keypoints, etc.), el paradigma anchor-based se rompe porque tendrías que diseñar "anchor polígonos" o "anchor curvas". Per-pixel regression desde el centro es trivialmente generalizable: solo cambias el número de canales del regression head.

Esto explica por qué FCOS desbloqueó toda una generación de detectores en dominios atípicos: text spotting, instance segmentation (CondInst, BlendMask), pose estimation (DirectPose), 3D detection (FCOS3D), keypoint detection. Todos siguen la receta "FCOS + N canales custom para el target".

### 9.4. Ejemplo concreto: cómo ABCNet extiende FCOS

Para ilustrar la magnitud del cambio (que es pequeña), enumeramos los diffs entre FCOS y la rama de detección de ABCNet:

| Componente | FCOS | ABCNet |
|------------|------|--------|
| Backbone | ResNet-50/101 | ResNet-50 |
| FPN | $P_3$-$P_7$ | $P_3$-$P_5$ (recortado, texto raro alcanza 512+ px) |
| Classification head | 80 clases (COCO) | 1 clase (texto/no-texto) |
| Regression head | 4 canales $(l, t, r, b)$ | **16 canales** (8 control points x,y de 2 curvas Bezier) |
| Centerness branch | Sí | Sí (idéntica) |
| Asignación multi-level | Por $\max(l,t,r,b)$ | Por bounding box que envuelve la curva |
| Loss | Focal + IoU + BCE | Focal + Smooth L1 sobre control points + BCE |
| Post-NMS | Bboxes axis-aligned | Curvas Bezier |

La rama de **recognition** de ABCNet (BezierAlign + GRU/atención) es separada y se entrena conjuntamente, pero la detección — el 70% del trabajo — es FCOS modificado. Esa es la propuesta de valor del paper: en lugar de inventar un detector nuevo para texto curvo, partir de FCOS y cambiar solo el último layer del regression head. La parameter efficiency y la transferencia de hiperparámetros desde COCO a ICDAR/TotalText son consecuencia directa de esta decisión.

## 10. Sucesores y línea de tiempo anchor-free

FCOS abrió la compuerta. Lo que vino después:

### 10.1. CenterNet (Zhou, Wang & Krähenbühl, arXiv 2019)

"Objects as Points". Predice un heatmap de centros (uno por clase) y a partir del centro regresiona size $(w, h)$ y offset sub-pixel. Más extremo que FCOS — un solo punto por objeto, no todos los puntos interiores. Útil para detección en tiempo real (~52 FPS en COCO).

### 10.2. ATSS (Zhang et al., CVPR 2020)

"Adaptive Training Sample Selection". Mostró que la **diferencia esencial** entre FCOS y RetinaNet no es anchor-free vs anchor-based, sino **cómo se asignan samples positivas**. ATSS propone una estrategia adaptativa basada en estadísticas de IoU que iguala el performance de FCOS con un detector "anchor-based" (anchor único). Conclusión importante: el verdadero secret sauce de FCOS es centerness + dense sampling de positives, no la ausencia de anchors per se.

### 10.3. PAA (Kim & Lee, ECCV 2020)

"Probabilistic Anchor Assignment". Modela la asignación de samples como un problema probabilístico, asumiendo una mezcla de gaussianas sobre scores y asignando samples al modo "positivo" con EM. Combinable con FCOS-style heads.

### 10.4. DETR (Carion et al., ECCV 2020)

El salto definitivo: **Transformer + Hungarian matching**. DETR predice un set fijo de $N$ predictions y las matchea bipartitamente con los GT boxes vía Hungarian algorithm. Rompe con anchors **y** con NMS **y** con dense prediction. Trade-off: convergencia mucho más lenta (500 epochs vs 12-90k iters de FCOS), pero abre la era end-to-end set prediction.

### 10.5. FCOSv2 / FCOS++ (mismos autores, ~2020-2021)

Mejoras incrementales: mejor sampling, mejor manejo de bordes, integración con dynamic head. La rama FCOS sigue viva en MMDetection y Detectron2.

### 10.6. La tabla genealógica abreviada

Para fijar la cronología en la cabeza:

| Año | Paper | Aporte |
|-----|-------|--------|
| 2015 | DenseBox (Huang) | Per-pixel 4-distance prediction (face/text) |
| 2015 | Faster R-CNN (Ren) | RPN + anchors estándar |
| 2016 | YOLOv1 (Redmon) | Grid-based, anchor-free pero baja recall |
| 2016 | SSD (Liu) | Multi-scale anchors single-shot |
| 2016 | UnitBox (Yu) | IoU loss para regresión |
| 2017 | FPN (Lin) | Feature pyramid multi-escala |
| 2017 | RetinaNet (Lin) | FPN + focal loss + 9 anchors/loc |
| 2018 | CornerNet (Law) | Anchor-free vía corner pairs + embeddings |
| 2018 | GroupNorm (Wu & He) | Norm independiente de batch size |
| 2019 | **FCOS (Tian)** | **Per-pixel + FPN + centerness, anchor-free competitivo** |
| 2019 | FSAF (Zhu) | Feature selective anchor-free module |
| 2019 | CenterNet (Zhou) | Objects as Points, centro + size |
| 2019 | GIoU (Rezatofighi) | Mejor regression loss |
| 2020 | **ABCNet (Liu)** | **FCOS + Bezier para text spotting curvo** |
| 2020 | ATSS (Zhang) | Adaptive Training Sample Selection |
| 2020 | PAA (Kim) | Probabilistic Anchor Assignment |
| 2020 | DETR (Carion) | Transformer + Hungarian matching, end-to-end |
| 2020 | FCOS3D (Wang) | FCOS extendido a detección 3D |

## 11. Conexiones con otros papers del curso

- **Clase 17 (Pose recognition)**: el patrón de "predict per-pixel heatmaps + offsets" de OpenPose, AlphaPose y derivados es estructuralmente análogo. La diferencia: pose predice keypoint locations; FCOS predice bbox sides. Ambos heredan del legado de FCN (Long et al. 2015).
- **FPN (Lin et al., CVPR 2017)**: la columna vertebral multi-escala de FCOS.
- **Focal Loss (Lin et al. ICCV 2017, paper de RetinaNet)**: el classification loss de FCOS. Sin focal loss, FCOS no convergería con dense sampling.
- **IoU loss / UnitBox (Yu et al., ACM MM 2016)**: el regression loss original. Después reemplazado por GIoU.
- **GIoU (Rezatofighi et al., CVPR 2019)**: mejor regression loss para bboxes; FCOS la adopta en improvements.
- **GroupNorm (Wu & He, ECCV 2018)**: enabler en regímenes de batch chico.
- **Faster R-CNN (Ren et al., NeurIPS 2015)**: el contrapunto anchor-based contra el que FCOS se posiciona.
- **CornerNet (Law & Deng, ECCV 2018)**: el predecesor anchor-free al que FCOS supera en simplicidad y AP.
- **ABCNet (Liu et al., CVPR 2020, clase 21)**: hereda FCOS literal como detector base.
- **FCN (Long et al., CVPR 2015)**: la inspiración filosófica — "todo es predicción densa".

## 12. Lectura crítica

### 12.1. Lo que FCOS hace bien

- **Simplicidad arquitectónica**: el diff vs RetinaNet es pequeño (regression head cambia, se agrega centerness branch). Fácil de implementar, fácil de extender.
- **Reducción de hiperparámetros**: ~10 menos que RetinaNet (escalas, aspect ratios, IoU thresholds desaparecen).
- **Memory footprint**: 9× menos outputs en el último layer. No trivial cuando se hace inference en edge.
- **Versatilidad**: el patrón "per-pixel regression de N targets" se generaliza a muchos dominios sin redesign.

### 12.2. Lo que NO resuelve

- **NMS sigue siendo necesaria**. FCOS no es end-to-end en el sentido de DETR.
- **El threshold de score (0.05) y el NMS threshold (0.6)** siguen siendo hiperparámetros. FCOS reduce, no elimina.
- **Multi-level assignment con thresholds fijos $(m_i)$** es otro set de hiperparámetros que el paper hereda de FPN. ATSS y PAA después demostraron que assignment adaptativo es estrictamente mejor.
- **Tie-breaker por mínima área** es heurístico. ATSS lo reemplaza con assignment estadístico.
- **No hay teoría** de por qué centerness es la fórmula correcta — es ingeniería empírica. Sus variantes (centerness²,  centerness en regression branch, etc.) muestran que la formulación exacta no es única.

### 12.3. ¿Es realmente anchor-free?

Punto provocativo de ATSS: si interpretas "una location del feature map" como un "anchor único de tamaño cero", FCOS es equivalente a RetinaNet con 1 anchor por location + dense positive sampling. Es un sano recordatorio de que las categorías ("anchor-based" vs "anchor-free") son menos crujientes de lo que parecen. Lo que **sí** es genuinamente novel en FCOS es:

1. **Center-ness branch** (esto es nuevo).
2. **Dense positive sampling** (todos los pixels interiores).
3. **Multi-level FPN assignment por max(l, t, r, b)** en lugar de por área del bbox.

### 12.4. Notas sobre la rigurosidad experimental

El paper hace un esfuerzo serio por aislar variables. Tres prácticas destacables:

- **Mismo hiperparámetro set que RetinaNet** out-of-the-box, sin re-tuning específico para FCOS. Esto remueve la sospecha de cherry-picking de hyperparams. Los autores explícitamente especulan que tuning específico podría mejorar AP — pero prefieren no hacerlo para mantener la comparación limpia.
- **Tablas de ablation que reportan también AR$_1$, AR$_{10}$, AR$_{100}$** (Average Recall), no solo AP. AR es importante para validar que FCOS no está sacrificando recall (la queja histórica de DenseBox).
- **Ablation sobre número de anchors de RetinaNet** (Tabla 8): RetinaNet con #A=1 (un solo anchor) obtiene 32.5 AP; con #A=9 obtiene 35.7. FCOS "puro" con $C_5$ obtiene 35.7 — igual a RetinaNet con 9 anchors. Esto es importante porque sugiere que FCOS está "compensando" la falta de anchors múltiples con dense positive sampling, no haciendo trampa.

### 12.5. Impacto a largo plazo

FCOS es citado >10000 veces (2026). Es uno de los detectores más implementados en frameworks (MMDetection, Detectron2, AdelaiDet) y su patrón "per-pixel dense head + centerness" es la base de:

- ABCNet, ABCNetv2 (text spotting).
- CondInst, SOLOv2 (instance segmentation).
- FCOS3D (3D detection en autonomous driving).
- DirectPose (pose estimation).
- BorderDet, PolyYOLO (detección con representaciones alternativas de bbox).

Sin FCOS, ABCNet no existiría tal como existe. Sin FCOS, la era anchor-free no se hubiera consolidado. Es uno de esos papers que, en retrospectiva, parecen obvios — pero requirió coraje argumentar contra el "de facto standard" de la época y demostrar empíricamente que la simplicidad gana.

## 13. Resumen para la sesión de clase

- **Problema**: detectores anchor-based tienen demasiados hiperparámetros, demasiada memoria, desbalance FG/BG severo, no generalizan fuera de COCO.
- **Solución**: predicción per-pixel densa de (clase, 4 distancias a los lados del bbox, centerness).
- **Tres palancas** para que funcione:
  1. Multi-level FPN assignment por rango de $\max(l, t, r, b)$ → resuelve overlap ambiguity.
  2. Dense positive sampling: todos los pixels interiores del bbox son positivos → más samples de regresión.
  3. Center-ness branch → suprime predicciones lejos del centro, +3.6 AP.
- **Resultado**: 44.7 AP en COCO test-dev con ResNeXt-64x4d-101 — supera RetinaNet, CornerNet, Faster R-CNN+FPN, con menos hiperparámetros y 9× menos outputs por location.
- **Para clase 21**: ABCNet es literal FCOS con regression head de 16 canales (Bezier control points) en lugar de 4. Entender FCOS es entender la mitad de ABCNet.
- **Sucesores**: ATSS (assignment adaptativo), PAA (assignment probabilístico), CenterNet (centros + size), DETR (transformers + Hungarian, end-to-end sin NMS).
- **Bottom line**: FCOS no es solo "RetinaNet sin anchors" — es un cambio de paradigma hacia detección como dense prediction, que generaliza a tareas con outputs complejos (curvas, polígonos, keypoints).
