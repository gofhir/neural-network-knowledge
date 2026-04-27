---
title: "MIT 6.S191 (2026): Deep Computer Vision"
weight: 10
sidebar:
  open: true
---

**Curso:** MIT 6.S191 - Introduction to Deep Learning (2026)
**Instructor:** Alexander Amini
**Lecture:** 3 - Deep Computer Vision
**Fecha:** 6 de enero de 2026
**Slides oficiales:** [PDF local (4.4 MB)](/videos/mit-6s191-l3-2026/slides.pdf) - [Original MIT](https://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L3.pdf)

{{< youtube pqIcoskUuWs >}}

## Atribución

Material original de **MIT 6.S191** ([introtodeeplearning.com](https://introtodeeplearning.com)), Alexander Amini, 6 de enero de 2026, distribuido bajo licencia del curso. Las notas en español son una elaboración independiente, sin afiliación oficial con MIT.

## Resumen

El tercer lecture del curso MIT 6.S191 edición 2026 (titulado **Deep Computer Vision**) construye desde primeros principios la maquinaria de las **redes convolucionales (CNNs)** aplicadas a visión por computador. Las primeras 22 slides motivan por qué las redes fully-connected fracasan sobre imágenes y por qué la **convolución** —localidad + weight sharing + múltiples filtros— es la inducción de bias correcta. Las slides 26-50 desarrollan la operación paso a paso (con la animación canónica de filtro $3 \times 3$ deslizándose sobre input $5 \times 5$), construyen la arquitectura `Conv → ReLU → Pool` apilada, y muestran cómo la **jerarquía de features** emerge automáticamente al entrenar end-to-end.

La segunda mitad del lecture (slides 51-69) es lo que distingue a la versión 2026: en vez de quedarse en clasificación, recorre **una arquitectura, muchas aplicaciones** — clasificación médica (cribado de cáncer de mama, McKinney *Nature* 2020), detección de objetos (sliding window → R-CNN → Faster R-CNN), segmentación semántica (FCN, U-Net), y un caso end-to-end de conducción autónoma con control probabilístico (Amini ICRA 2019). El cierre apunta al lab 2 (Facial Detection Systems), donde se evalúan sesgos demográficos del modelo entrenado.

Para nuestro curso UC, este lecture **complementa** la clase [09](/clases/clase-09/) (CNNs en Profundidad, Miguel Fadic), que profundiza en arquitecturas específicas (VGG, Inception, ResNet) que el lecture MIT solo nombra al pasar. La aritmética del receptive field, los filtros $1 \times 1$, y las decisiones de diseño detrás de cada arquitectura están en clase-09; las aplicaciones más allá de clasificación están aquí.

## Estructura del lecture (70 slides en 22 secciones)

| Bloque | Slides | Tema |
|---|---|---|
| **I. Motivación y representación** | | |
| 1 | 1-8 | Visión computacional: definición, impacto (medicina, robótica, conducción autónoma) |
| 2 | 9-13 | Imágenes como números: tensores, regresión vs clasificación |
| 3 | 14-17 | Por qué la extracción manual de features falla (6 fuentes de variación) |
| 4 | 18 | Aprender una jerarquía de features: low / mid / high level |
| 5 | 19-22 | Por qué las redes fully-connected fallan en imágenes |
| 6 | 23-25 | Estructura espacial: del patch a la convolución |
| **II. La convolución y la arquitectura CNN** | | |
| 7 | 26-29 | Caso de estudio: detectar una "X" deforme con tres filtros locales |
| 8 | 30-42 | La operación de convolución, paso a paso (animación 5×5 vs 3×3) |
| 9 | 43-44 | Pipeline CNN para clasificación: `Conv → ReLU → Pool` |
| 10 | 45-46 | Capas convolucionales: matemática y conectividad local |
| 11 | 47 | Volúmenes 3D: profundidad, stride, padding, receptive field |
| 12 | 48 | No-linealidad: ReLU |
| 13 | 49 | Pooling: max pool, average pool, invarianza local |
| 14 | 50 | Representación aprendida: jerarquía emergente en capas profundas |
| **III. Aplicaciones** | | |
| 15 | 51-52 | Pipeline completo: feature learning + clasificación con softmax |
| 16 | 53-54 | Implementación práctica (TensorFlow y PyTorch) |
| 17 | 55-56 | Una arquitectura, muchas aplicaciones (backbone reutilizable) |
| 18 | 57 | Clasificación médica: cribado de cáncer de mama (McKinney 2020) |
| 19 | 58-62 | Detección de objetos: sliding window → R-CNN → Faster R-CNN |
| 20 | 63 | Segmentación semántica: FCN y U-Net |
| 21 | 64-66 | Control continuo: conducción autónoma end-to-end |
| 22 | 67-69 | Cierre: alcance e impacto + pointer al Lab 2 |

(La slide 70 contiene anuncios del curso y se omite en las notas.)

## Diferencia con el material UC

Este lecture y la clase-09 del curso UC se complementan sin duplicarse:

- **Empezar con el video MIT** si vienes sin base de visión computacional. La motivación de las primeras 22 slides es excepcional, y las aplicaciones (detección, segmentación, control end-to-end) cubren territorio que clase-09 no toca.
- **Pasar a la clase-09 UC** una vez tengas claro qué es una CNN. Ahí están las arquitecturas específicas (VGG, Inception, ResNet) con derivaciones, código y tradeoffs — lo que MIT solo nombra de paso.
- **Las herramientas transversales** (transfer learning, data augmentation, interpretabilidad) viven en `fundamentos/` del sitio UC, con cobertura más profunda que cualquiera de los dos lectures.

## Recursos

{{< cards >}}
  {{< card link="notas" title="Notas" subtitle="Recorrido temático de las 70 diapositivas en 22 secciones" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundización" subtitle="Papers seminales (LeNet a Grad-CAM), derivaciones, frontera moderna" icon="beaker" >}}
  {{< card link="glosario" title="Glosario" subtitle="55 términos bilingües CNN + detección + segmentación" icon="book-open" >}}
  {{< card link="slides.pdf" title="Slides PDF" subtitle="70 páginas, copia oficial MIT" icon="document" >}}
{{< /cards >}}

## Cross-links del curso

{{< cards >}}
  {{< card link="/clases/clase-09" title="Clase 09 — CNNs en Profundidad" subtitle="VGG, Inception, ResNet, interpretabilidad — Miguel Fadic" icon="academic-cap" >}}
  {{< card link="/fundamentos/redes-convolucionales" title="Fundamento: Redes Convolucionales" subtitle="Operación, dimensión de salida, AlexNet" icon="book-open" >}}
  {{< card link="/fundamentos/transfer-learning" title="Fundamento: Transfer Learning" subtitle="Feature extraction, fine-tuning, lr diferenciado" icon="book-open" >}}
  {{< card link="/fundamentos/data-augmentation" title="Fundamento: Data Augmentation" subtitle="Geometric, foto-metric, Mixup, VRM" icon="book-open" >}}
  {{< card link="/fundamentos/interpretabilidad" title="Fundamento: Interpretabilidad" subtitle="Feature Visualization, Saliency, Grad-CAM" icon="book-open" >}}
  {{< card link="/videos/mit-6s191-l2-2026" title="Video MIT 6.S191 (2026) L2" subtitle="Lecture previa del mismo curso — Deep Sequence Modeling" icon="film" >}}
{{< /cards >}}

---

> Material adaptado de **MIT 6.S191 (2026) Lecture 3: Deep Computer Vision**, Alexander Amini, 6 de enero de 2026.
> [Video](https://www.youtube.com/watch?v=pqIcoskUuWs) - [Slides oficiales](https://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L3.pdf) - [Sitio del curso](https://introtodeeplearning.com/).
> Notas en español como elaboración independiente. Sin afiliación oficial con MIT.
