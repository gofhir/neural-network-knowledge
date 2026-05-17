---
title: "Lab 15 - Faster R-CNN: Inferencia COCO y Fine-tuning para Mapaches"
weight: 150
sidebar:
  open: true
---

**Profesor:** Juan Pablo de Vicente
**Fecha:** Mayo 2026
**Notebook origen:** `clase_15/material/Laboratorio/FasterRCNN_Practico_v8.ipynb`

## Encuadre

Laboratorio dividido en **dos partes** que cubren el ciclo completo de un detector moderno:

- **Parte 1 — Inferencia con Faster R-CNN pre-entrenado en COCO**: instanciar `torchvision.models.detection.fasterrcnn_resnet50_fpn` con pesos pre-entrenados, recorrer su arquitectura (transform, backbone ResNet-50 + FPN, RPN con anchors, RoI heads con MultiScaleRoIAlign), y correr inferencia sobre imagenes con clases COCO (zebras, oficina, gente comiendo).
- **Parte 2 — Fine-tuning para detectar mapaches**: motivacion didactica (el modelo COCO predice `bear:0.5` sobre un mapache), dataset Raccoon de 200 imagenes, reemplazo del `FastRCNNPredictor` para pasar de 91 clases COCO a 2 clases (background + raccoon), loop de entrenamiento con utilities (IoU, NMS por clase, warmup LR, metricas), y verificacion sobre imagenes nuevas.

Para la teoria detras de la arquitectura ver la [clase 15](/clases/clase-15/).

## Resultados consolidados

Entrenamiento real sobre dataset Raccoon (160 train + 40 val), 4 epocas, GPU T4 en Colab. **~5 minutos** de entrenamiento total.

| Epoca | Train loss | TP | FN | FP | Recall | Precision |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.357 | 30 | 16 | 153 | 65.2% | 16.4% |
| 1 | 0.134 | 35 | 11 | 44 | 76.1% | 44.3% |
| 2 | 0.095 | 36 | 10 | 21 | 78.3% | 63.2% |
| **3** | **0.084** | **36** | **10** | **19** | **78.3%** | **65.5%** ⭐ |

**Modelo guardado**: epoca 3, F1 ≈ 0.713. Inferencia sobre 3 imagenes de validacion (raccoon-42, raccoon-31, raccoon-191): **deteccion correcta con score >0.9** en las tres.

## Recursos del lab — Parte 1 (Inferencia COCO)

{{< cards >}}
  {{< card link="arquitectura" title="Arquitectura de Faster R-CNN" subtitle="Las 4 piezas: transform, backbone+FPN, RPN, roi_heads" icon="academic-cap" >}}
  {{< card link="inferencia-coco" title="Inferencia con modelo COCO" subtitle="get_prediction, object_detection_api, 3 imagenes de prueba" icon="academic-cap" >}}
{{< /cards >}}

## Recursos del lab — Parte 2 (Fine-tuning)

{{< cards >}}
  {{< card link="experimento-mapache" title="Experimento didactico" subtitle="El modelo COCO predice 'bear' sobre un mapache" icon="academic-cap" >}}
  {{< card link="dataset-y-dataloader" title="Dataset Raccoon y DataLoader custom" subtitle="200 imagenes, formato .txt, RaccoonDataLoader, collate" icon="academic-cap" >}}
  {{< card link="fine-tuning-setup" title="Reemplazo del clasificador" subtitle="FastRCNNPredictor de 91 -> 2 clases" icon="academic-cap" >}}
  {{< card link="utilities" title="Utilities del entrenamiento" subtitle="IoU, print_stats, stats_2_metrics, NMS por clase, warmup LR" icon="academic-cap" >}}
  {{< card link="entrenamiento" title="Loop de entrenamiento" subtitle="train_one_epoch, eval_epoch, train_model con multi-task loss" icon="academic-cap" >}}
  {{< card link="inferencia-finetuneada" title="Lanzamiento + inferencia final" subtitle="SGD + StepLR, resultados reales 4 epocas, inferencia post-fine-tuning" icon="academic-cap" >}}
{{< /cards >}}

## Cierre

{{< cards >}}
  {{< card link="tarea-final" title="Tarea final" subtitle="Respuestas razonadas a las 2 preguntas + lecciones consolidadas" icon="check-circle" >}}
{{< /cards >}}

## Notebook (Colab + descarga)

{{< cards >}}
  {{< card link="/notebooks/lab15.ipynb" title="Notebook ejecutado" subtitle="FasterRCNN_Practico_v8 con outputs completos (.ipynb descargable, 15 MB)" icon="document" >}}
  {{< card link="/notebooks-html/lab15.html" title="Render HTML" subtitle="Notebook ejecutado renderizado en HTML (16 MB, todas las imagenes embebidas)" icon="document-text" >}}
{{< /cards >}}

## Papers relacionados

{{< cards >}}
  {{< card link="/papers/faster-rcnn-ren-2015" title="Faster R-CNN (Ren 2015)" subtitle="El detector base con RPN end-to-end" icon="document-text" >}}
  {{< card link="/papers/resnet-he-2015" title="ResNet (He 2015)" subtitle="La backbone con conexiones residuales" icon="document-text" >}}
  {{< card link="/papers/fpn-lin-2017" title="FPN (Lin 2017)" subtitle="Piramide multi-escala con top-down + lateral" icon="document-text" >}}
  {{< card link="/papers/mask-rcnn-he-2017" title="Mask R-CNN (He 2017)" subtitle="Aporta RoIAlign (heredado por torchvision)" icon="document-text" >}}
  {{< card link="/papers/coco-lin-2014" title="Microsoft COCO (Lin 2014)" subtitle="El dataset estandar de deteccion" icon="document-text" >}}
{{< /cards >}}

## Cross-links

{{< cards >}}
  {{< card link="/clases/clase-15" title="Clase 15 - Teoria" subtitle="R-CNN, Fast/Faster R-CNN, YOLO, FPN: recorrido de las diapositivas" icon="academic-cap" >}}
  {{< card link="/fundamentos/deteccion-de-objetos" title="Fundamento: Deteccion de Objetos" subtitle="IoU, NMS, anchors, RPN, RoIAlign, FPN, family tree" icon="book-open" >}}
  {{< card link="/fundamentos/redes-convolucionales" title="Fundamento: CNNs" subtitle="Backbone arquitectonico: AlexNet, VGG, ResNet, Inception" icon="book-open" >}}
  {{< card link="/fundamentos/transfer-learning" title="Fundamento: Transfer Learning" subtitle="Pretraining COCO y fine-tuning a nuevas clases" icon="book-open" >}}
{{< /cards >}}

---

> **Estado:** Lab completo. Cubre las celdas 0-95 del notebook con 9 paginas tematicas, resultados reales del entrenamiento integrados y respuestas a la tarea final. El recorrido es reproducible end-to-end en Colab con GPU T4 en ~5 minutos de entrenamiento + ~10 minutos de revision interactiva.
