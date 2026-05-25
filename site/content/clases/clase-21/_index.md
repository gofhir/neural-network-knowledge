---
title: "Clase 21 - Scene Text Recognition"
weight: 210
sidebar:
  open: true
---

**Profesor:** Miguel Fadic
**Fecha:** 2026-05-24

Quinta clase del bloque de visión avanzada. Recorre el campo de **Scene Text Recognition (STR)**: cómo leer texto incrustado en imágenes naturales (señales, vitrinas, productos), donde el problema es radicalmente distinto del OCR clásico sobre documentos escaneados. La clase organiza el campo en seis estaciones — definición, aplicaciones, pipeline en stages, datasets, métricas y ABCNet como caso de estudio profundo del estado del arte 2020.

ABCNet (Liu et al. CVPR 2020) sintetiza el material: combina un detector anchor-free (FCOS) sobre FPN, una representación de texto curvado con **curvas Bézier cúbicas** parametrizadas por 8 puntos de control, una alineación geométrica (BezierAlign) que generaliza RoIAlign, y un recognizer attention-based — todo en un único pipeline end-to-end real-time. La clase complementa la [Clase 09 (CNN)](/clases/clase-09) con la backbone visual, la [Clase 14 (Transformers)](/clases/clase-14) con el mecanismo de atención del decoder, y la [Clase 17 (Pose Recognition)](/clases/clase-17) con el patrón "dense prediction sobre feature maps".

## Apuntes de clase

{{< cards >}}
  {{< card link="teoria" title="Teoria" subtitle="Recorrido de las 40 diapositivas: STR vs OCR, stages, datasets, evaluation, ABCNet" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="Math detallada: curvas Bezier, BezierAlign, CTC vs attention, IoU/GIoU, FCOS centerness, Levenshtein" icon="beaker" >}}
  {{< card link="/clases/clase-20" title="Clase anterior: ELMo, BERT, GPT, ChatGPT" subtitle="Embeddings contextualizados y LLMs" icon="arrow-left" >}}
  {{< card link="/clases/clase-17" title="Base: Pose Recognition" subtitle="Dense prediction y heads regression sobre features" icon="academic-cap" >}}
  {{< card link="/clases/clase-14" title="Base: Transformers" subtitle="Self-attention y attention decoder" icon="academic-cap" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/scene-text-recognition" title="Scene Text Recognition" subtitle="Pipeline 4-stages, datasets, metricas, evolucion historica" icon="book-open" >}}
  {{< card link="/fundamentos/bezier-curves" title="Curvas de Bezier" subtitle="Bernstein polynomials, De Casteljau, control points y representacion de texto curvado" icon="book-open" >}}
  {{< card link="/fundamentos/ctc-loss" title="CTC Loss" subtitle="Connectionist Temporal Classification: blank symbol, forward-backward, decoding" icon="book-open" >}}
  {{< card link="/fundamentos/anchor-free-detection" title="Anchor-Free Detection" subtitle="FCOS, CenterNet, CornerNet, DETR: per-pixel prediction sin anchors" icon="book-open" >}}
  {{< card link="/fundamentos/deteccion-de-objetos" title="Deteccion de Objetos" subtitle="IoU, NMS, mAP, RPN, FPN, RoI extraction" icon="book-open" >}}
  {{< card link="/fundamentos/mecanismo-atencion" title="Mecanismo de Atencion" subtitle="Bahdanau attention en sequence decoders" icon="book-open" >}}
  {{< card link="/fundamentos/redes-convolucionales" title="Redes Convolucionales" subtitle="Backbones VGG, ResNet, DenseNet para feature extraction" icon="book-open" >}}
{{< /cards >}}

## Papers de esta clase

{{< cards >}}
  {{< card link="/papers/abcnet-liu-2020" title="ABCNet (2020)" subtitle="Liu et al. -- Real-time scene text spotting con Adaptive Bezier-Curve Network" icon="document-text" >}}
  {{< card link="/papers/text-recognition-wild-chen-2020" title="STR Survey (2020)" subtitle="Chen et al. -- Survey de referencia que estructura el campo en 4 stages" icon="document-text" >}}
  {{< card link="/papers/fcos-tian-2019" title="FCOS (2019)" subtitle="Tian et al. -- Anchor-free one-stage detection: backbone de ABCNet" icon="document-text" >}}
  {{< card link="/papers/giou-rezatofighi-2019" title="GIoU (2019)" subtitle="Rezatofighi et al. -- IoU diferenciable acotada para bbox regression" icon="document-text" >}}
  {{< card link="/papers/total-text-chng-2017" title="Total-Text (2017)" subtitle="Ch'ng & Chan -- Primer dataset focado en texto curvado" icon="document-text" >}}
  {{< card link="/papers/fast-rcnn-girshick-2015" title="Fast R-CNN (2015)" subtitle="Girshick -- Bisagra entre R-CNN multi-stage y Faster R-CNN end-to-end" icon="document-text" >}}
{{< /cards >}}

## Papers canonicos (base teorica)

{{< cards >}}
  {{< card link="/papers/crnn-shi-2017" title="CRNN (2017)" subtitle="Shi et al. -- CNN+BLSTM+CTC: el baseline universal de text recognition" icon="document-text" >}}
  {{< card link="/papers/ctc-graves-2006" title="CTC (2006)" subtitle="Graves et al. -- Connectionist Temporal Classification: entrenar RNN sin alineamiento" icon="document-text" >}}
  {{< card link="/papers/stn-jaderberg-2015" title="STN (2015)" subtitle="Jaderberg et al. -- Modulo diferenciable de transformacion espacial aprendida" icon="document-text" >}}
{{< /cards >}}

## Recursos del laboratorio

{{< cards >}}
  {{< card link="/laboratorios/lab-21" title="Laboratorio 21" subtitle="Practico de Scene Text Recognition con notebook ejecutado" icon="academic-cap" >}}
{{< /cards >}}

## Dominio relacionado

{{< cards >}}
  {{< card link="/dominios/vision" title="Dominio: Vision" subtitle="Linea de tiempo completa: de OCR Tesseract a foundation models multimodales" icon="globe-alt" >}}
{{< /cards >}}
