---
title: "Visión"
weight: 2
sidebar:
  open: true
---

# Visión

## El problema central

Una imagen es una grilla de píxeles. Esa grilla tiene **estructura espacial fuerte**: un píxel se parece mucho a sus vecinos, y los objetos relevantes son combinaciones jerárquicas de patrones locales — bordes que forman texturas, texturas que forman partes, partes que forman objetos. Una arquitectura para visión gana o pierde según cuán bien aproveche esa estructura.

Tres tensiones recorren toda la historia: (1) cómo construir **invariancia a traslación, escala y deformaciones** sin perder discriminabilidad, (2) cómo entrenar redes **profundas** sin que el gradiente colapse, y (3) cómo combinar el **sesgo inductivo de localidad** (CNNs) con el **alcance global** (atención) — la pregunta que el Transformer Visual terminó respondiendo en 2020.

## Línea de tiempo

{{< timeline >}}
  {{< era name="Era pre-neural" years="1959-2010" >}}
    {{< hito year="1959" name="Hubel y Wiesel" status="minimal" >}}
      Descubrimiento de células simples y complejas en V1 del gato. **Por qué importó:** inspiró el campo receptivo local, base de toda CNN posterior.
    {{< /hito >}}
    {{< hito year="1980" name="Neocognitron (Fukushima)" status="minimal" >}}
      Arquitectura jerárquica con capas alternadas de detección y agrupamiento. **Por qué importó:** prototipo conceptual de la convolución y el pooling.
    {{< /hito >}}
    {{< hito year="1998" name="LeNet-5" status="minimal" >}}
      LeCun: CNN entrenable con backprop para dígitos manuscritos (MNIST). **Por qué importó:** demostró que las CNNs eran prácticas, pero fueron ignoradas por el campo durante una década por falta de cómputo.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era CNN" years="2012-2014" >}}
    {{< hito year="2012" name="AlexNet" status="covered" link="/papers/alexnet-krizhevsky-2012" >}}
      Ganó ImageNet 2012 por margen abrumador con GPUs, ReLU y dropout. El paper que reinició el deep learning moderno.
    {{< /hito >}}
    {{< hito year="2014" name="VGGNet" status="covered" link="/papers/vggnet-simonyan-2014" >}}
      Profundidad uniforme (3x3, stride 1) hasta 19 capas. Demostró que más profundidad = mejor representación.
    {{< /hito >}}
    {{< hito year="2014" name="GoogLeNet / Inception" status="covered" link="/papers/googlenet-szegedy-2014" >}}
      Módulos Inception con convoluciones a múltiples escalas. Más profundo y más eficiente que VGG.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era residual" years="2015-2017" >}}
    {{< hito year="2015" name="ResNet" status="covered" link="/papers/resnet-he-2015" >}}
      Conexiones residuales (skip connections) que permitieron entrenar redes de 152+ capas sin que el gradiente colapsara. Cambió permanentemente cómo se diseñan redes profundas.
    {{< /hito >}}
    {{< hito year="2016" name="DenseNet" status="minimal" >}}
      Conexiones densas: cada capa recibe la concatenación de todas las anteriores. **Por qué importó:** reutilización máxima de features con menos parámetros que ResNet.
    {{< /hito >}}
    {{< hito year="2017" name="MobileNet" status="minimal" >}}
      Convoluciones separables en profundidad para móviles. **Por qué importó:** primera familia diseñada para inferencia eficiente en dispositivos.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era de detección y segmentación" years="2014-2018" >}}
    {{< hito year="2014" name="COCO" status="covered" link="/papers/coco-lin-2014" >}}
      Microsoft COCO: 328k imágenes, 80 categorías, 7.7 objetos/imagen, segmentación a nivel de instancia. **Por qué importó:** dataset estándar de detección moderno y la métrica mAP@[.5:.95] que penaliza imprecisión en la caja.
    {{< /hito >}}
    {{< hito year="2014" name="R-CNN" status="minimal" >}}
      Region proposals + CNN para clasificar cada región. **Por qué importó:** primera arquitectura de detección extremo a extremo basada en CNNs.
    {{< /hito >}}
    {{< hito year="2015" name="Faster R-CNN" status="covered" link="/papers/faster-rcnn-ren-2015" >}}
      Region Proposal Network integrado dentro de la CNN, anchors $k=9$, NMS. **Por qué importó:** detección viable en tiempo casi real (5 fps con VGG-16), elimina el cuello de botella de Selective Search externo.
    {{< /hito >}}
    {{< hito year="2015" name="U-Net" status="minimal" >}}
      Encoder-decoder con skip connections para segmentación médica. **Por qué importó:** sigue siendo el caballo de batalla de segmentación biomédica.
    {{< /hito >}}
    {{< hito year="2016" name="YOLO" status="minimal" >}}
      Detección como única regresión sobre toda la imagen. **Por qué importó:** detección a 60+ FPS, abrió la puerta a robótica y video.
    {{< /hito >}}
    {{< hito year="2017" name="FPN" status="covered" link="/papers/fpn-lin-2017" >}}
      Feature Pyramid Network: combina bottom-up con top-down + lateral connections para construir una pirámide multi-escala con semántica fuerte en todos los niveles. **Por qué importó:** se volvió componente estándar de Faster R-CNN, Mask R-CNN, RetinaNet y todos los detectores modernos. +12.9 puntos de AP en objetos pequeños.
    {{< /hito >}}
    {{< hito year="2017" name="Mask R-CNN" status="covered" link="/papers/mask-rcnn-he-2017" >}}
      Extiende Faster R-CNN con rama de segmentación paralela y reemplaza RoI Pool por **RoIAlign** sin cuantización. **Por qué importó:** ganó ICCV 2017 Best Paper Award, unificó detección + segmentación de instancias + keypoints en un framework, y RoIAlign aportó +1.3 box AP "gratis" a Faster R-CNN.
    {{< /hito >}}
    {{< hito year="2017" name="RetinaNet" status="minimal" >}}
      Detector single-stage con **focal loss** $(1-p_t)^\gamma \log p_t$ para el desbalance fondo:objeto. **Por qué importó:** demostró que single-shot puede competir con two-stage en precisión.
    {{< /hito >}}
    {{< hito year="2020" name="DETR" status="minimal" >}}
      End-to-End Object Detection with Transformers: trata detección como set prediction con bipartite matching. **Por qué importó:** elimina anchors y NMS, primer detector verdaderamente end-to-end.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era de pose humana" years="2014-2022" >}}
    {{< hito year="2014" name="DeepPose" status="minimal" >}}
      Toshev & Szegedy: primera regresión directa $(x, y)$ de keypoints con CNN. **Por qué importó:** estableció pose estimation como tarea de regresión sobre CNN; superado rápidamente por heatmaps Gaussianos.
    {{< /hito >}}
    {{< hito year="2015" name="SMPL" status="covered" link="/papers/smpl-loper-2015" >}}
      Loper et al.: modelo paramétrico realista del cuerpo humano (10 shape + 72 pose params → malla de 6890 vértices) con Linear Blend Skinning compatible con engines de animación estándar. **Por qué importó:** se volvió el modelo paramétrico de cuerpo humano más usado del decenio — sustrato de DensePose, HMR, VIBE, AMASS y casi todo body recovery moderno.
    {{< /hito >}}
    {{< hito year="2015" name="FaceNet" status="covered" link="/papers/facenet-schroff-2015" >}}
      Schroff et al. (Google): embedding 128-D + triplet loss con online semi-hard mining. **Por qué importó:** 99.63% en LFW (30% menos error que SOTA previo), estableció *metric learning* como paradigma y es ancestro de SimCLR, MoCo, ArcFace.
    {{< /hito >}}
    {{< hito year="2017" name="OpenPose / Part Affinity Fields" status="minimal" >}}
      Cao et al. (CMU): primer método bottom-up multi-person de gran escala vía PAFs discretos. **Por qué importó:** democratizó pose real-time, base de muchas aplicaciones de fitness/dance/AR.
    {{< /hito >}}
    {{< hito year="2018" name="DensePose" status="covered" link="/papers/densepose-guler-2018" >}}
      Güler et al. (Facebook AI): mapea cada píxel humano a la superficie 3D del cuerpo (SMPL) vía $(c, U, V)$. Introduce COCO-DensePose con ~5M correspondencias manuales. **Por qué importó:** rompió la limitación de 17 keypoints discretos, abrió virtual try-on y dense human reasoning.
    {{< /hito >}}
    {{< hito year="2019" name="HRNet" status="minimal" >}}
      Sun et al.: arquitectura multi-resolución manteniendo features de alta resolución a lo largo de toda la red. **Por qué importó:** dominó pose estimation 2D durante 2019-2021 con AP ~76 en COCO, baseline canónico de la era pre-transformer.
    {{< /hito >}}
    {{< hito year="2019" name="PifPaf" status="covered" link="/papers/pifpaf-kreiss-2019" >}}
      Kreiss, Bertoni, Alahi (EPFL): bottom-up con Part Intensity Field + Part Association Field + Laplace loss para incertidumbre aprendida. **Por qué importó:** SOTA en baja resolución (self-driving), +18% AP sobre OpenPose a 321 px, base de openpifpaf en producción.
    {{< /hito >}}
    {{< hito year="2022" name="ViTPose" status="covered" link="/papers/vitpose-xu-2022" >}}
      Xu et al.: ViT plain como backbone + decoder lightweight, 80.9 AP en COCO test-dev (ViTPose-G de 1B params). **Por qué importó:** demostró que pose no requiere arquitecturas multi-resolución (HRNet); el ViT pretrained con MAE carga toda la representación. Nuevo SOTA con simplicidad.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era Transformer" years="2020-presente" >}}
    {{< hito year="2020" name="Vision Transformer (ViT)" status="deep" link="/fundamentos/vision-transformer" >}}
      Aplica un Transformer puro sobre parches de la imagen. Con suficiente data y escala, supera a CNNs sin sesgos inductivos visuales explícitos.
    {{< /hito >}}
    {{< hito year="2021" name="Swin Transformer" status="minimal" >}}
      Transformer jerárquico con ventanas locales que recuperan parte del sesgo inductivo de las CNNs. **Por qué importó:** ViT eficiente para tareas densas (detección, segmentación).
    {{< /hito >}}
    {{< hito year="2021" name="CLIP" status="minimal" >}}
      ViT entrenado por contraste con texto en pares imagen-caption. **Por qué importó:** puente con el dominio multimodal; visión cero-shot por texto.
    {{< /hito >}}
    {{< hito year="2023" name="SAM (Segment Anything)" status="minimal" >}}
      Foundation model para segmentación con prompts. **Por qué importó:** segmentación zero-shot sobre cualquier imagen.
    {{< /hito >}}
    {{< hito year="2024-2025" name="Modelos generativos (Diffusion / Sora)" status="minimal" >}}
      Stable Diffusion 3, Imagen, DALL·E 3, Sora. **Por qué importó:** generación fotorrealista por texto y video como aplicaciones masivas.
    {{< /hito >}}
  {{< /era >}}
{{< /timeline >}}

## Era 1 — Pre-neural (1959-2010)

### Problema heredado

Antes de las redes profundas, la visión por computador era principalmente **ingeniería de features**: SIFT, HOG, SURF y descriptores hechos a mano que codificaban robustez a iluminación, rotación y escala. Sobre esos features se entrenaban SVMs o random forests. Los resultados eran respetables pero saturaban: cada nueva tarea requería diseñar features específicas, y los benchmarks como ImageNet se resistían a superar el ~25% de error en top-5.

### Idea clave

**Aprender la jerarquía de features en lugar de diseñarla.** Las CNNs llevaban décadas de existencia conceptual — Hubel y Wiesel habían descrito el campo receptivo local en V1, Fukushima había construido el Neocognitron, y LeCun había mostrado con LeNet-5 que una CNN entrenada con backpropagation funcionaba sobre dígitos. Pero el cómputo y los datos disponibles no alcanzaban para imágenes naturales.

### Qué la destronó

Tres factores convergieron: GPUs entrenables (CUDA, 2007), ImageNet (2009) con 1.2M imágenes etiquetadas, y un equipo de Toronto dispuesto a entrenar una CNN profunda sobre ese dataset. El resultado fue AlexNet.

## Era 2 — CNNs (2012-2014)

### Problema heredado

LeNet-5 funcionaba para dígitos pero no para imágenes naturales: 6 capas no alcanzaban para representar la complejidad visual del mundo real. La pregunta era si las CNNs podían escalar.

### Idea clave

**Escalar CNNs en datos, profundidad y cómputo.** AlexNet (Krizhevsky, 2012) puso 8 capas, 60M parámetros, dos GPUs, ReLU en lugar de tanh, dropout para regularizar y data augmentation. Ganó ImageNet 2012 con 15.3% top-5 error contra 26.2% del segundo lugar. Ese delta convenció al campo de que el deep learning no era ruido.

VGG (2014) llevó la idea al extremo de la simplicidad: solo 3x3 conv stride 1 + 2x2 max-pool, hasta 19 capas. Demostró que la profundidad uniforme era una receta robusta. GoogLeNet (2014) tomó la dirección opuesta — módulos Inception con convoluciones a múltiples escalas en paralelo — y mostró que se podía ser más profundo y más eficiente simultáneamente.

### Qué la destronó

Al intentar pasar de 20 a 30, 50, 100 capas, los gradientes colapsaban. La red simplemente no entrenaba mejor con más profundidad — empeoraba.

## Era 3 — Conexiones residuales (2015-2017)

### Problema heredado

El problema de degradación en redes profundas: agregar capas a una red ya buena la empeoraba, incluso al ignorar overfitting. Era un problema de **optimización**, no de capacidad.

### Idea clave

**Skip connections.** ResNet (He et al., 2015) propuso que cada bloque aprendiera una *función residual* $F(x) = H(x) - x$ en lugar de la transformación completa $H(x)$. La identidad $x$ se sumaba directamente a la salida del bloque. Esto convertía el camino del gradiente en una autopista: si el bloque no aprendía nada útil, no estorbaba. Permitió entrenar redes de 152 capas, ganar ImageNet 2015 y volverse el bloque básico de prácticamente toda arquitectura profunda posterior — incluyendo el Transformer.

DenseNet generalizó la idea con conexiones densas, MobileNet la adaptó a inferencia móvil con convoluciones separables. La era residual no fue un nuevo paradigma sino la consolidación del paradigma CNN.

### Qué la destronó

Las CNNs eran imbatibles en clasificación de imágenes naturales, pero tenían un sesgo fuerte: la convolución asume que las features útiles son locales. Eso es cierto en muchas escalas, pero no en todas. Para tareas que requieren razonamiento global o relaciones de largo alcance, la atención prometía algo mejor.

## Era 4 — Detección y segmentación (2014-2018)

### Problema heredado

Mientras las eras 2 y 3 perfeccionaban la **clasificación**, una rama paralela atacaba problemas más exigentes: **detección** (¿dónde está cada objeto?) y **segmentación** (¿qué píxel pertenece a qué clase?). La clasificación de imágenes responde *qué hay en esta imagen*. Pero las aplicaciones reales — conducción autónoma, robótica, imagenología médica — requieren saber dónde está cada cosa y a veces delinearla a nivel de píxel.

### Idea clave

Esta era es paralela a las eras CNN y residual; se desarrolló sobre las mismas backbones (AlexNet, VGG, ResNet). Tres familias de arquitecturas emergieron:

- **R-CNN family** (R-CNN 2014, Fast 2015, Faster R-CNN 2015, Mask R-CNN 2017): **two-stage** — generar propuestas de regiones y clasificar cada una. La pieza maestra fue [Faster R-CNN](/papers/faster-rcnn-ren-2015) con su **Region Proposal Network** (RPN) entrenable end-to-end, que eliminó el cuello de botella de Selective Search externo. Mask R-CNN agregó una rama de segmentación de instancias y reemplazó RoI Pool por **RoIAlign** (sin cuantización via interpolación bilineal).
- **Single-stage** (YOLO 2016, SSD 2016, RetinaNet 2017): tratar la detección como una única regresión densa sobre la imagen. Mucho más rápidos (real-time) pero históricamente menos precisos. RetinaNet cerró la brecha con **focal loss**, que pondera el cross-entropy para no ahogarse en negativos fáciles.
- **Encoder-decoder con skip connections** (U-Net 2015, [FPN 2017](/papers/fpn-lin-2017)): para segmentación, donde cada píxel necesita una predicción. FPN se volvió la pieza estándar para detección multi-escala: combina top-down (semántica) con bottom-up (resolución) via lateral connections, y se usa hoy en casi cualquier detector competitivo.

Conceptos transversales que cristalizaron en esta era — [IoU](/fundamentos/deteccion-de-objetos), NMS (por clase), mAP@[.5:.95] (penaliza imprecisión en la caja), anchors con parametrización log, smooth L1 para regresión robusta, transfer learning desde ImageNet — son la base del campo aplicado moderno. El dataset estándar es [COCO](/papers/coco-lin-2014) (80 categorías, 7.7 objetos/imagen, segmentación a nivel de instancia).

### Qué la destronó

Estas arquitecturas siguen vigentes en producción (torchvision, Detectron2, mmdetection ofrecen Faster R-CNN y Mask R-CNN como baselines), pero el ecosistema migró progresivamente a backbones Transformer y a frameworks que eliminan heurísticas no diferenciables. **DETR** (2020) trató detección como **set prediction** con bipartite matching — sin anchors, sin NMS. La llegada de **SAM** (2023) cambió además la conversación en segmentación: ya no se entrena un modelo por dataset, sino que se prompt-tunea un foundation model.

## Era de pose humana (2014-2022)

### Problema heredado

Las eras 2-4 perfeccionaron clasificación, detección y segmentación. Pero el cuerpo humano es un objeto **estructurado** — no basta con localizarlo en un bbox; muchas aplicaciones (deportes, salud, robótica, VR/AR, vigilancia) necesitan saber **dónde están sus articulaciones** o **cómo se deforma su superficie**.

### Idea clave

Dos paradigmas dominantes coexisten:

- **Keypoints** + heatmaps Gaussianos: representar el cuerpo como ~17 puntos discretos (COCO), entrenar una CNN a predecir un heatmap por keypoint. Top-down (Mask R-CNN keypoints, HRNet, [ViTPose](/papers/vitpose-xu-2022)) detecta personas primero y estima pose dentro de cada bbox. Bottom-up (OpenPose, [PifPaf](/papers/pifpaf-kreiss-2019)) detecta partes en toda la imagen y luego asocia.

- **Dense correspondence**: mapear cada píxel humano a la superficie 3D del cuerpo, parametrizada por el modelo [SMPL](/papers/smpl-loper-2015) (Loper 2015). [DensePose](/papers/densepose-guler-2018) (Güler 2018) introdujo COCO-DensePose con ~5M correspondencias manuales y abrió virtual try-on, body-aware rendering, y la transferencia de texturas entre personas.

En paralelo, **face recognition** se reinventó con [FaceNet](/papers/facenet-schroff-2015) (Schroff 2015) — embeddings 128-D entrenados con **triplet loss** y online semi-hard mining. Es ancestro conceptual de SimCLR, MoCo y ArcFace.

[ViTPose](/papers/vitpose-xu-2022) (2022) cerró el arco demostrando que un ViT plain como backbone + decoder lightweight alcanza SOTA (80.9 AP en COCO test-dev) — los conceptos de la era CNN siguen válidos, solo cambia el backbone.

### Qué viene

La frontera es **3D body recovery** (HMR, VIBE, 4DHumans) que fittea SMPL completo desde una imagen 2D, **animal pose** (Continuous Surface Embeddings), y **vision-language pose** (modelos que aceptan instrucciones textuales para inferir poses category-agnostic). Toda esta era está plagada de implicaciones éticas — vigilancia masiva, aplicaciones militares, sesgos demográficos — que el ingeniero responsable debe contemplar y a menudo rechazar.

## Era 5 — Vision Transformer y foundation models (2020-presente)

### Problema heredado

Las CNNs habían dominado por un sesgo inductivo: localidad y equivariancia a traslación. Pero ese sesgo es también una limitación: la información de un lado de la imagen tarda muchas capas en interactuar con el otro lado. La pregunta abierta de finales de 2010s era si una arquitectura sin sesgo inductivo visual explícito — atención pura sobre parches — podía funcionar dada suficiente data.

### Idea clave

**Tratar una imagen como una secuencia de parches.** ViT (Dosovitskiy et al., 2020) divide la imagen en parches 16x16, los proyecta a tokens y los pasa por un Transformer encoder estándar. Sin convoluciones, sin pooling, sin sesgo de localidad. Con datasets enormes (JFT-300M) y entrenamiento suficiente, ViT supera a CNNs comparables. La interpretación: el sesgo inductivo de las CNNs es una ayuda con poca data y un techo con mucha data.

Swin Transformer (2021) combinó lo mejor de ambos mundos: jerarquía y atención local en ventanas. CLIP (2021) entrenó ViT por contraste con texto, abriendo la era multimodal. SAM (2023) volvió la segmentación un problema zero-shot. Y los modelos generativos (Stable Diffusion 3, Sora) llevaron la generación de imágenes y video a producción masiva.

### Qué viene

La frontera actual son los **modelos visuales fundacionales** entrenados a la escala de los LLMs, capaces de razonar sobre imágenes complejas (GPT-4V, Gemini, Claude con visión), y los **modelos generativos de video** que requieren coherencia temporal — la próxima gran prueba.

## Estado del arte hoy

{{< callout type="info" >}}

**Visión como sub-modalidad de los foundation models (2024-2025).** La visión moderna no compite por sí sola; aparece como capacidad nativa de modelos generales.

- **GPT-4V / GPT-5 Vision** — análisis de imágenes complejas, gráficos, diagramas, documentos.
- **Claude Vision** — Anthropic. Foco en comprensión profunda de imágenes técnicas y diagramas.
- **Gemini 2.5 Pro Vision** — multimodal nativo desde el pretraining.
- **SAM 2** — Meta. Segmentación zero-shot extendida a video.
- **Stable Diffusion 3 / Imagen 3** — generación texto-imagen producción.
- **Sora** — OpenAI. Generación de video con coherencia temporal extendida.
- **DINOv2** — Meta. Foundation model self-supervised para visión, base de muchas pipelines downstream.

{{< /callout >}}

## Casos de uso reales

- **Conducción autónoma**: Tesla FSD, Waymo, Cruise — detección, segmentación, predicción de trayectorias en tiempo real.
- **Diagnóstico médico**: dermatología, radiología, oftalmología — desde Inception en retinopatía diabética (2016) hasta foundation models médicos actuales.
- **Búsqueda visual**: Google Lens, Pinterest Lens, Amazon — search por imagen + texto combinado (CLIP-style).
- **Moderación de contenido**: detección de NSFW, copyright, deepfakes a escala en redes sociales.
- **Robótica industrial**: pick-and-place, control de calidad, inspección visual.
- **Generación creativa**: Midjourney, Stable Diffusion, Adobe Firefly — del concept art al diseño de producto.
- **OCR y comprensión de documentos**: facturas, formularios, contratos — donde visión + texto se cruzan.

## Qué viene

Las apuestas activas en visión hoy: **modelos generativos de video con coherencia temporal extendida** (más allá de Sora), **agentes visuales** (modelos que actúan sobre interfaces gráficas como humanos), **3D nativo** (Gaussian splatting, NeRFs como paso previo a foundation models 3D), **visión embebida** (modelos eficientes para edge), y **vision-language-action** para robótica (RT-2, π0). La integración cada vez más profunda con LLMs hace probable que el campo de "visión por computador" como disciplina aislada se diluya en favor de modelos multimodales generales.

## Recursos relacionados

**Fundamentos:**
- [Redes convolucionales](/fundamentos/redes-convolucionales).
- [Detección de objetos](/fundamentos/deteccion-de-objetos) — IoU, NMS, anchors, RPN, RoIAlign, FPN, family tree completa.
- [Vision Transformer](/fundamentos/vision-transformer).
- [Transfer learning](/fundamentos/transfer-learning).
- [Data augmentation](/fundamentos/data-augmentation).
- [Regularización](/fundamentos/regularizacion).
- [Aprendizaje contrastivo](/fundamentos/aprendizaje-contrastivo).

**Papers:**
- [AlexNet (Krizhevsky 2012)](/papers/alexnet-krizhevsky-2012).
- [VGGNet (Simonyan 2014)](/papers/vggnet-simonyan-2014).
- [GoogLeNet (Szegedy 2014)](/papers/googlenet-szegedy-2014).
- [Microsoft COCO (Lin 2014)](/papers/coco-lin-2014) — dataset estándar de detección.
- [ResNet (He 2015)](/papers/resnet-he-2015).
- [Faster R-CNN (Ren 2015)](/papers/faster-rcnn-ren-2015) — RPN end-to-end.
- [FPN (Lin 2017)](/papers/fpn-lin-2017) — pirámide multi-escala.
- [Mask R-CNN (He 2017)](/papers/mask-rcnn-he-2017) — RoIAlign + segmentación de instancias.
- [SMPL (Loper 2015)](/papers/smpl-loper-2015) — modelo paramétrico del cuerpo humano.
- [FaceNet (Schroff 2015)](/papers/facenet-schroff-2015) — embeddings 128-D + triplet loss.
- [DensePose (Güler 2018)](/papers/densepose-guler-2018) — correspondencia densa imagen-SMPL.
- [PifPaf (Kreiss 2019)](/papers/pifpaf-kreiss-2019) — pose bottom-up con composite fields.
- [ViTPose (Xu 2022)](/papers/vitpose-xu-2022) — SOTA pose con ViT plain.
- [ViT (Dosovitskiy 2021)](/papers/vit-dosovitskiy-2021).
- [Batch Normalization (Ioffe 2015)](/papers/batch-norm-ioffe-2015).
- [Dropout (Srivastava 2014)](/papers/dropout-srivastava-2014).
- [Mixup (Zhang 2017)](/papers/mixup-zhang-2017).
- [Transferable features (Yosinski 2014)](/papers/transferable-features-yosinski-2014).

**Clases del diplomado:**
- [Clase 15 — Reconocimiento de Objetos](/clases/clase-15) — R-CNN, Fast/Faster R-CNN, YOLO, FPN.
- [Laboratorio 15 — Faster R-CNN práctico](/laboratorios/lab-15) — inferencia COCO + fine-tuning para mapaches con torchvision.
- [Clase 17 — Pose Recognition](/clases/clase-17) — keypoints, DensePose, PifPaf, ViTPose, FaceNet, ética.
- Clases sobre CNNs, ResNets y backbones modernas.
- Clases sobre ViT y atención visual.

---

*Última actualización: 2026-05-17.*
