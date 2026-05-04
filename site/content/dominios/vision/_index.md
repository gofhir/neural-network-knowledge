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
    {{< hito year="2014" name="R-CNN" status="minimal" >}}
      Region proposals + CNN para clasificar cada región. **Por qué importó:** primera arquitectura de detección extremo a extremo basada en CNNs.
    {{< /hito >}}
    {{< hito year="2015" name="Faster R-CNN" status="minimal" >}}
      Region Proposal Network integrado dentro de la CNN. **Por qué importó:** detección viable en tiempo casi real.
    {{< /hito >}}
    {{< hito year="2015" name="U-Net" status="minimal" >}}
      Encoder-decoder con skip connections para segmentación médica. **Por qué importó:** sigue siendo el caballo de batalla de segmentación biomédica.
    {{< /hito >}}
    {{< hito year="2016" name="YOLO" status="minimal" >}}
      Detección como única regresión sobre toda la imagen. **Por qué importó:** detección a 60+ FPS, abrió la puerta a robótica y video.
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

Esta era es paralela a las eras CNN y residual; se desarrolló sobre las mismas backbones (AlexNet, VGG, ResNet). Las ideas centrales son:

- **R-CNN family** (R-CNN, Fast, Faster R-CNN, Mask R-CNN): generar propuestas de regiones y clasificar cada una.
- **Single-stage** (YOLO, SSD, RetinaNet): tratar la detección como una única regresión densa sobre la imagen.
- **Encoder-decoder con skip connections** (U-Net, FPN): para segmentación, donde cada píxel necesita una predicción.

### Qué la destronó

Estas arquitecturas siguen vigentes, pero el ecosistema migró progresivamente a backbones Transformer. Y la llegada de SAM (2023) cambió la conversación: ya no se entrena un modelo de segmentación por dataset, sino que se prompt-tunea un foundation model.

## Era 5 — Vision Transformer y foundation models (2020-presente)

### Problema heredado

Las CNNs habían dominado por un sesgo inductivo: localidad y equivariancia a traslación. Pero ese sesgo es también una limitación: la información de un lado de la imagen tarda muchas capas en interactuar con el otro lado. La pregunta abierta de finales de 2010s era si una arquitectura sin sesgo inductivo visual explícito — atención pura sobre parches — podía funcionar dada suficiente data.

### Idea clave

**Tratar una imagen como una secuencia de parches.** ViT (Dosovitskiy et al., 2020) divide la imagen en parches 16x16, los proyecta a tokens y los pasa por un Transformer encoder estándar. Sin convoluciones, sin pooling, sin sesgo de localidad. Con datasets enormes (JFT-300M) y entrenamiento suficiente, ViT supera a CNNs comparables. La interpretación: el sesgo inductivo de las CNNs es una ayuda con poca data y un techo con mucha data.

Swin Transformer (2021) combinó lo mejor de ambos mundos: jerarquía y atención local en ventanas. CLIP (2021) entrenó ViT por contraste con texto, abriendo la era multimodal. SAM (2023) volvió la segmentación un problema zero-shot. Y los modelos generativos (Stable Diffusion 3, Sora) llevaron la generación de imágenes y video a producción masiva.

### Qué viene

La frontera actual son los **modelos visuales fundacionales** entrenados a la escala de los LLMs, capaces de razonar sobre imágenes complejas (GPT-4V, Gemini, Claude con visión), y los **modelos generativos de video** que requieren coherencia temporal — la próxima gran prueba.
