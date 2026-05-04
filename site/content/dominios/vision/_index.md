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
