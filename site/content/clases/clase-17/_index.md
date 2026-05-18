---
title: "Clase 17 - Pose Recognition"
weight: 130
sidebar:
  open: true
---

**Profesor:** Tomás Vergara Browne
**Fecha:** 2026-05-07

Reconocimiento de poses humanas en imágenes y video. La clase extiende la [Clase 15 (Detección de Objetos)](/clases/clase-15) agregando una **tercera cabeza** al pipeline Faster R-CNN — una cabeza de **keypoints** — y desarrolla las representaciones modernas del cuerpo humano: keypoints discretos, **DensePose** (correspondencia densa imagen-superficie SMPL), aproximaciones **bottom-up** vía PifPaf, y la era **Vision Transformer** con ViTPose. Cierra con técnicas auxiliares (face recognition vía triplet networks, FaceNet) y una sección de **ética** sobre vigilancia y aplicaciones militares.

{{< cards >}}
  {{< card link="teoria" title="Teoria" subtitle="Recorrido de las 59 diapositivas de la clase" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="Math de heatmaps Gaussianos, Laplace loss, UV mapping y triplet ranking" icon="beaker" >}}
  {{< card link="/clases/clase-15" title="Clase anterior" subtitle="Clase 15 — Detección de Objetos (Faster R-CNN)" icon="arrow-left" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/pose-estimation" title="Pose Estimation 2D" subtitle="Keypoints, top-down vs bottom-up, heatmaps, OKS/GPS" icon="book-open" >}}
  {{< card link="/fundamentos/dense-correspondence" title="Dense Correspondence" subtitle="UV mapping, MDS, geodésicas sobre SMPL" icon="book-open" >}}
  {{< card link="/fundamentos/triplet-loss" title="Triplet Loss" subtitle="Metric learning, semi-hard mining, FaceNet" icon="book-open" >}}
  {{< card link="/fundamentos/deteccion-de-objetos" title="Detección de Objetos" subtitle="IoU, NMS, anchors, RoIAlign — base de la clase" icon="book-open" >}}
  {{< card link="/fundamentos/redes-convolucionales" title="Redes Convolucionales" subtitle="Backbones CNN para visión" icon="book-open" >}}
{{< /cards >}}

## Papers de esta clase

{{< cards >}}
  {{< card link="/papers/densepose-guler-2018" title="DensePose (2018)" subtitle="Güler et al. — correspondencia densa imagen-SMPL, COCO-DensePose" icon="document-text" >}}
  {{< card link="/papers/pifpaf-kreiss-2019" title="PifPaf (2019)" subtitle="Kreiss et al. — bottom-up multi-persona, composite fields" icon="document-text" >}}
  {{< card link="/papers/vitpose-xu-2022" title="ViTPose (2022)" subtitle="Xu et al. — Vision Transformer plain como backbone, SOTA COCO 80.9 AP" icon="document-text" >}}
  {{< card link="/papers/facenet-schroff-2015" title="FaceNet (2015)" subtitle="Schroff et al. — embedding 128-D + triplet loss para face recognition" icon="document-text" >}}
  {{< card link="/papers/smpl-loper-2015" title="SMPL (2015)" subtitle="Loper et al. — modelo paramétrico de cuerpo humano, sustrato 3D" icon="document-text" >}}
{{< /cards >}}
