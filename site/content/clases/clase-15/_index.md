---
title: "Clase 15 - Reconocimiento de Objetos"
weight: 110
sidebar:
  open: true
---

**Profesor:** Juan Pablo de Vicente
**Fecha:** 2026-04-30

Reconocimiento de objetos como problema region-based: del clasificador holistico de imagen completa a la deteccion de multiples objetos con cajas y clases. Recorrido por R-CNN (Girshick 2014), Fast R-CNN, Faster R-CNN (RPN, RoI Pool, anchors), YOLO single-shot y FPN para deteccion multi-escala. Conceptos: IoU, mAP, NMS, smooth L1, multi-task loss.

{{< cards >}}
  {{< card link="teoria" title="Teoria" subtitle="Recorrido de las 47 diapositivas: R-CNN, FCN, YOLO, Faster R-CNN, FPN" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="Math detallado: smooth L1, parametrizacion offsets, IoU/AP/mAP, RoI Align, DETR" icon="beaker" >}}
  {{< card link="/laboratorios/lab-15" title="Laboratorio 15" subtitle="Faster R-CNN: inferencia COCO + fine-tuning para mapaches" icon="cube-transparent" >}}
  {{< card link="/fundamentos/deteccion-de-objetos" title="Fundamento: Deteccion de Objetos" subtitle="IoU, NMS, anchors, RPN, RoIAlign, FPN, family tree" icon="book-open" >}}
  {{< card link="/fundamentos/redes-convolucionales" title="Fundamento: CNNs" subtitle="Backbone arquitectonico: AlexNet, VGG, ResNet, Inception" icon="book-open" >}}
  {{< card link="/fundamentos/transfer-learning" title="Fundamento: Transfer Learning" subtitle="Pretraining ImageNet y fine-tuning para deteccion" icon="book-open" >}}
{{< /cards >}}

## Papers de esta clase

- Alexe, Deselaers, Ferrari (2012) "Measuring the Objectness of Image Windows" (PAMI)
- Uijlings, van de Sande, Gevers, Smeulders (2013) "Selective Search for Object Recognition"
- Girshick, Donahue, Darrell, Malik (2014) "Rich feature hierarchies for accurate object detection and semantic segmentation" (R-CNN)
- He, Zhang, Ren, Sun (2014) "Spatial Pyramid Pooling in Deep Convolutional Networks" (SPP-Net)
- Girshick (2015) "Fast R-CNN"
- Ren, He, Girshick, Sun (2015) "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
- Redmon, Farhadi (2017) "YOLO9000: Better, Faster, Stronger"
- Lin, Dollar, Girshick, He, Hariharan, Belongie (2017) "Feature Pyramid Networks for Object Detection"
