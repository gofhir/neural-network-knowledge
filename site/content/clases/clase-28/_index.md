---
title: "Clase 28 - Aprendizaje Autosupervisado"
weight: 280
sidebar:
  open: true
---

**Profesor:** Sebastián Amenábar
**Curso 3 / Tópicos de profundización:** Relacional, GANs, RL, Meta-Learning, Razonamiento y Memoria

Clase de profundización sobre **aprendizaje autosupervisado (SSL)**: aprender representaciones útiles **sin etiquetas humanas**, generando automáticamente el objetivo a predecir desde el propio dato. La motivación es doble: etiquetar es caro (más aún si requiere profesionales, como en medicina) y los humanos aprendemos sin tantas etiquetas. La clase recorre las tres grandes familias de **pretext tasks** —**predicción/generativos** (autoencoders, inpainting, colorización, masked modeling), **transformaciones** (rotación, orden temporal, contrastive learning) y **multimodalidad** (audio-visión, imagen-texto)— y cierra con el uso del SSL para **potenciar el aprendizaje supervisado** con datos sin etiquetar (**UDA**, semi-supervisado, el método del laboratorio).

La clase integra todo el curso: [autoencoders](/fundamentos/aprendizaje-autosupervisado) y CNN para imágenes, [BERT/GPT (Clase 20)](/clases/clase-20) como SSL en lenguaje (MLM/NTP), [ViT (Clase 23)](/clases/clase-23) que habilita el MAE, [CLIP](/papers/clip-radford-2021) como contrastivo imagen-texto, y el [aprendizaje contrastivo](/fundamentos/aprendizaje-contrastivo) como columna vertebral de SimCLR/MoCo.

## Apuntes de clase

{{< cards >}}
  {{< card link="teoria" title="Teoria" subtitle="Recorrido de las 42 diapositivas: motivación, pretext tasks (predicción/transformaciones/multimodal), contrastive learning, MAE, UDA" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="Math: autoencoders y PCA, InfoNCE/NT-Xent, SimCLR vs MoCo (EMA), MAE, KL de consistencia (UDA), rotación" icon="beaker" >}}
  {{< card link="practica" title="Practica desde 0" subtitle="SimCLR, MAE y UDA desde cero en triple framework (PyTorch, TensorFlow, JAX)" icon="code" >}}
  {{< card link="/clases/clase-27" title="Clase anterior: Redes Neuronales de Grafos" subtitle="GNN, message passing, GCN/GAT" icon="arrow-left" >}}
  {{< card link="/clases/clase-20" title="Base: ELMo, BERT, GPT, ChatGPT" subtitle="SSL en lenguaje: MLM y next-token prediction" icon="academic-cap" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/aprendizaje-autosupervisado" title="Aprendizaje Autosupervisado" subtitle="Pretext tasks, taxonomía, evaluación, generativo vs contrastivo, aplicaciones" icon="book-open" >}}
  {{< card link="/fundamentos/aprendizaje-contrastivo" title="Aprendizaje Contrastivo" subtitle="Acercar positivos, alejar negativos: SimCLR, MoCo, CLIP" icon="book-open" >}}
  {{< card link="/fundamentos/aprendizaje-semi-supervisado" title="Aprendizaje Semi-Supervisado" subtitle="Consistency training, UDA, pseudo-labeling — la base del lab" icon="book-open" >}}
  {{< card link="/fundamentos/transfer-learning" title="Transfer Learning" subtitle="Usar representaciones pre-entrenadas como inicialización" icon="book-open" >}}
  {{< card link="/fundamentos/representacion-datos" title="Representación de Datos" subtitle="Qué hace que una representación sea buena y transferible" icon="book-open" >}}
{{< /cards >}}

## Papers de esta clase

{{< cards >}}
  {{< card link="/papers/context-encoders-pathak-2016" title="Context Encoders (2016)" subtitle="Pathak et al. — inpainting como pretext task" icon="document-text" >}}
  {{< card link="/papers/colorization-zhang-2016" title="Colorful Colorization (2016)" subtitle="Zhang et al. — colorización como pretext, color por clasificación" icon="document-text" >}}
  {{< card link="/papers/context-prediction-doersch-2015" title="Context Prediction (2015)" subtitle="Doersch et al. — posicionamiento relativo de parches" icon="document-text" >}}
  {{< card link="/papers/shuffle-and-learn-misra-2016" title="Shuffle and Learn (2016)" subtitle="Misra et al. — verificación de orden temporal en video" icon="document-text" >}}
  {{< card link="/papers/rotnet-gidaris-2018" title="RotNet (2018)" subtitle="Gidaris et al. — predecir rotaciones, la simplicidad como virtud" icon="document-text" >}}
  {{< card link="/papers/invariant-spreading-ye-2019" title="Invariant & Spreading (2019)" subtitle="Ye et al. — el puente conceptual hacia el contrastivo" icon="document-text" >}}
  {{< card link="/papers/simclr-chen-2020" title="SimCLR (2020)" subtitle="Chen et al. — framework contrastivo simple, NT-Xent" icon="document-text" >}}
  {{< card link="/papers/moco-he-2019" title="MoCo (2019)" subtitle="He et al. — cola de negativos + momentum encoder" icon="document-text" >}}
  {{< card link="/papers/mae-he-2022" title="MAE (2022)" subtitle="He et al. — masked autoencoders escalables con ViT" icon="document-text" >}}
  {{< card link="/papers/uda-xie-2019" title="UDA (2019)" subtitle="Xie et al. — consistency training semi-supervisado (el paper del lab)" icon="document-text" >}}
{{< /cards >}}

## Papers complementarios

{{< cards >}}
  {{< card link="/papers/moco-v2-chen-2020" title="MoCo v2 (2020)" subtitle="Chen et al. — supera a SimCLR sin TPUs" icon="document-text" >}}
  {{< card link="/papers/convirt-zhang-2020" title="ConVIRT (2020)" subtitle="Zhang et al. — contrastivo médico imagen-texto, precursor de CLIP" icon="document-text" >}}
  {{< card link="/papers/look-listen-learn-arandjelovic-2017" title="Look, Listen and Learn (2017)" subtitle="Arandjelović & Zisserman — correspondencia audio-visual" icon="document-text" >}}
  {{< card link="/papers/objects-that-sound-arandjelovic-2018" title="Objects that Sound (2018)" subtitle="Arandjelović & Zisserman — localizar el objeto que suena" icon="document-text" >}}
  {{< card link="/papers/visualbert-li-2019" title="VisualBERT (2019)" subtitle="Li et al. — MLM auto-supervisado multimodal" icon="document-text" >}}
  {{< card link="/papers/urban-ssl-stalder-2023" title="SSL urbano (2023)" subtitle="Stalder et al. — el tiempo como segunda vista gratis" icon="document-text" >}}
  {{< card link="/papers/mae-video-feichtenhofer-2022" title="MAE en video (2022)" subtitle="Feichtenhofer et al. — masked autoencoders espaciotemporales" icon="document-text" >}}
  {{< card link="/papers/clip-radford-2021" title="CLIP (2021)" subtitle="Radford et al. — contrastivo imagen-texto a escala, zero-shot" icon="document-text" >}}
{{< /cards >}}

## Dominio relacionado

{{< cards >}}
  {{< card link="/dominios/vision" title="Dominio: Visión" subtitle="Línea de tiempo: el SSL como paradigma de pre-entrenamiento sin etiquetas" icon="globe-alt" >}}
{{< /cards >}}
