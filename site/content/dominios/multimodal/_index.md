---
title: "Multimodal"
weight: 5
sidebar:
  open: true
---

# Multimodal

## El problema central

Las modalidades (texto, imagen, audio, video) viven en **espacios incompatibles**. El texto es discreto y composicional — un vocabulario finito de tokens combinables por reglas sintácticas. Una imagen es una grilla continua de píxeles con estructura espacial. Un audio es una señal temporal continua. Hacerlas dialogar requiere proyectarlas a un espacio común o conectarlas con atención cruzada — y la elección entre ambas opciones vertebra una década de arquitecturas.

Tres tensiones definen el campo: (1) cómo **alinear modalidades sin pares anotados** — los datasets curados (COCO, Flickr30k) son pequeños y caros, mientras que pares ruidosos imagen-caption raspados de la web son virtualmente infinitos; (2) cómo dejar que un LLM **"vea" sin destruir su capacidad lingüística** — entrenamiento conjunto vs adaptadores ligeros que conectan un encoder visual congelado al LLM; (3) cómo **generar coherencia cross-modal** — un modelo que produce video debe respetar física, identidad y composición a través del tiempo, no solo sintaxis local.

## Línea de tiempo

{{< timeline >}}
  {{< era name="Era de captioning temprano" years="2014-2016" >}}
    {{< hito year="2015" name="Show and Tell" status="covered" link="/papers/show-and-tell-vinyals-2015" >}}
      Vinyals et al.: CNN encoder + LSTM decoder para generar descripciones de imágenes extremo a extremo. Primer captioner neural competente.
    {{< /hito >}}
    {{< hito year="2015" name="Show, Attend and Tell" status="covered" link="/papers/show-attend-tell-xu-2015" >}}
      Xu et al.: añade atención sobre regiones de la imagen en cada paso del decoder. El modelo "mira" donde necesita para generar la siguiente palabra.
    {{< /hito >}}
    {{< hito year="2015" name="VQA dataset" status="covered" link="/papers/vqa-antol-2015" >}}
      Antol et al.: Visual Question Answering como benchmark — *"¿De qué color es el casco?"* sobre una imagen. **Por qué importó:** forzó al campo más allá del captioning hacia razonamiento sobre imagen + texto. Cubierto en la [Clase 23](/clases/clase-23).
    {{< /hito >}}
    {{< hito year="2016" name="Stacked Attention Networks" status="covered" link="/papers/stacked-attention-yang-2016" >}}
      Yang et al.: atención espacial **multi-hop** guiada por la pregunta — razonar en varios pasos refinando dónde mirar. Antecedente directo de la top-down attention de VQA.
    {{< /hito >}}
    {{< hito year="2016" name="MCB (Compact Bilinear)" status="covered" link="/papers/mcb-fukui-2016" >}}
      Fukui et al.: fusión multimodal por **bilinear pooling** aproximado con Count Sketch + FFT. Ganó el VQA Challenge 2016. Abrió la línea de fusión bilineal.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era de atención visual estructurada" years="2017-2018" >}}
    {{< hito year="2017" name="Relation Networks" status="covered" link="/papers/relation-networks-santoro-2017" >}}
      Santoro et al.: módulo que razona sobre **pares de objetos** detectados, no sobre features uniformes de la imagen. Resolvió tareas de razonamiento visual donde CNNs puras fallaban.
    {{< /hito >}}
    {{< hito year="2017" name="VQAv2 (dataset balanceado)" status="covered" link="/papers/vqav2-goyal-2017" >}}
      Goyal et al.: rebalancea VQA con **pares de imágenes complementarias** (misma pregunta, respuesta distinta) para neutralizar los language priors. Se vuelve el benchmark estándar. Las slides 7-8 de la [Clase 23](/clases/clase-23).
    {{< /hito >}}
    {{< hito year="2017" name="MUTAN (Tucker Fusion)" status="covered" link="/papers/mutan-ben-younes-2017" >}}
      Ben-younes et al.: fusión bilineal parametrizada por **descomposición de Tucker**, controlando el trade-off expresividad/parámetros. Generaliza MCB y MLB.
    {{< /hito >}}
    {{< hito year="2018" name="Bottom-Up and Top-Down Attention" status="covered" link="/papers/bottom-up-attention-anderson-2018" >}}
      Anderson et al.: la atención visual opera sobre regiones detectadas con Faster R-CNN, no sobre la grilla densa. Estado del arte en captioning y VQA durante 2018-2019.
    {{< /hito >}}
    {{< hito year="2018" name="Pythia v0.1" status="covered" link="/papers/pythia-jiang-2018" >}}
      Jiang et al.: extiende Bottom-Up/Top-Down con mejoras incrementales (weight norm, ReLU, fusión Hadamard, ensemble) y **gana el VQA Challenge 2018**. El modelo central de la [Clase 23](/clases/clase-23).
    {{< /hito >}}
    {{< hito year="2019" name="GQA / VCR" status="minimal" >}}
      Hudson & Manning (GQA), Zellers et al. (VCR): benchmarks de razonamiento visual con preguntas composicionales y de sentido común. **Por qué importó:** revelaron lo lejos que estaban los modelos de razonar realmente.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era de pretraining multimodal" years="2019-2020" >}}
    {{< hito year="2019" name="ViLBERT" status="minimal" >}}
      Lu et al.: dos torres BERT (texto + imagen) con cross-attention entre ellas, preentrenadas con MLM extendido. **Por qué importó:** primer pretraining multimodal transferible.
    {{< /hito >}}
    {{< hito year="2019" name="LXMERT" status="minimal" >}}
      Tan & Bansal: variante con tres encoders (texto, visión, cross-modal) y cinco tareas de pretraining. **Por qué importó:** fijó el patrón "encoder por modalidad + cross-attention".
    {{< /hito >}}
    {{< hito year="2020" name="UNITER / OSCAR" status="minimal" >}}
      Chen et al. (UNITER), Li et al. (OSCAR): unifican BERT + visión en un solo Transformer y agregan etiquetas de objetos como "ancla" entre texto e imagen. **Por qué importó:** simplificaron la arquitectura sin perder rendimiento.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era contrastiva y zero-shot" years="2021-2022" >}}
    {{< hito year="2021" name="CLIP" status="deep" link="/fundamentos/aprendizaje-contrastivo" >}}
      Radford et al. (OpenAI): un encoder de texto y uno de imagen entrenados contrastivamente sobre 400M pares imagen-caption raspados de la web. Clasificación zero-shot competitiva con modelos supervisados.
    {{< /hito >}}
    {{< hito year="2021" name="ALIGN" status="minimal" >}}
      Jia et al. (Google): replica CLIP a 1.8B pares ruidosos, demostrando que la calidad del scraping importa menos que el volumen. **Por qué importó:** validó el contraste a escala como receta robusta.
    {{< /hito >}}
    {{< hito year="2022" name="BLIP" status="minimal" >}}
      Li et al. (Salesforce): unifica entendimiento (CLIP-style) y generación (captioning) en un mismo modelo, con pretraining sobre datos auto-filtrados. **Por qué importó:** primer paso hacia VLMs unificados.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era de VLMs y generación" years="2022-presente" >}}
    {{< hito year="2022" name="DALL·E 2 / Stable Diffusion / Imagen" status="minimal" >}}
      Ramesh et al. (DALL·E 2), Rombach et al. (Stable Diffusion), Saharia et al. (Imagen): generación texto-imagen de alta calidad con diffusion condicionado en CLIP/T5 text encoders. **Por qué importó:** primer caso en que generación multimodal se vuelve producto masivo.
    {{< /hito >}}
    {{< hito year="2022" name="Flamingo" status="minimal" >}}
      Alayrac et al. (DeepMind): few-shot VLM que intercala visión y texto en una secuencia, con cross-attention adaptado sobre un LLM congelado. **Por qué importó:** fijó el patrón "LLM congelado + adaptador visual".
    {{< /hito >}}
    {{< hito year="2023" name="LLaVA" status="minimal" >}}
      Liu et al.: VLM open-source con un proyector lineal entre encoder visual (CLIP) y LLM (LLaMA). Entrenamiento simple sobre pares instrucción-imagen sintetizados con GPT-4. **Por qué importó:** democratizó VLMs competentes.
    {{< /hito >}}
    {{< hito year="2023-2025" name="GPT-4V / Gemini / Claude Vision" status="covered" link="/fundamentos/foundation-models" >}}
      OpenAI, Google DeepMind, Anthropic: visión integrada nativamente al razonamiento de los frontier LLMs. Análisis de gráficos, documentos, código en pantalla, escenas complejas.
    {{< /hito >}}
    {{< hito year="2024-2025" name="Sora / Veo" status="minimal" >}}
      OpenAI (Sora), Google DeepMind (Veo): generación de video con coherencia temporal extendida (10-60 segundos), física aproximada y control fino por prompt. **Por qué importó:** mueve la frontera de imagen estática a video como producto masivo.
    {{< /hito >}}
  {{< /era >}}
{{< /timeline >}}

## Era 1 — Captioning temprano (2014-2016)

### Problema heredado

A inicios de los 2010s la visión por computador y el procesamiento de lenguaje vivían en silos: las CNNs clasificaban imágenes en categorías cerradas, los RNNs traducían texto. Conectar ambas modalidades era un problema abierto. La **descripción automática de imágenes** (image captioning) se volvió el primer terreno de prueba: dada una foto, generar una oración que la describa.

### Idea clave

**Combinar el encoder de visión con el decoder de lenguaje.** Show and Tell (Vinyals et al., 2015) tomó una CNN preentrenada en ImageNet, extrajo el vector de la última capa antes de la clasificación y lo pasó como estado inicial a una LSTM que generaba la descripción palabra por palabra. La arquitectura era una traducción directa de Seq2Seq: en lugar de "encoder de oración fuente", "encoder de imagen".

Show, Attend and Tell (Xu et al., 2015) agregó **atención sobre regiones** de la imagen: en cada paso del decoder el modelo aprende a mirar dónde necesita. Las visualizaciones del paper — el modelo mirando una pelota mientras genera la palabra "ball" — se volvieron icónicas.

### Qué la destronó

La atención uniforme sobre la grilla de features convolucionales no respetaba la estructura del mundo: los objetos no son píxeles aislados sino regiones cohesivas. La idea natural era operar sobre **objetos detectados** en lugar de la grilla densa.

## Era 2 — Atención visual estructurada (2017-2018)

### Problema heredado

Show, Attend and Tell atendía a una grilla 14x14 de features convolucionales — geometría arbitraria que no se alinea con los objetos reales en la escena. Las preguntas de VQA como *"¿de qué color es la pelota a la izquierda del hombre?"* exigen razonar sobre objetos, sus atributos y sus relaciones. Una grilla uniforme no provee esa estructura.

### Idea clave

**Atender a regiones detectadas, no a píxeles uniformes.** Anderson et al. (2018) propusieron usar Faster R-CNN para detectar 36 regiones por imagen, y atender sobre esas regiones desde el decoder de lenguaje. Las regiones traen consigo features semánticamente coherentes (objetos, partes), y la atención sobre ellas es interpretable.

En paralelo, Santoro et al. (2017) introdujeron **Relation Networks**: un módulo que razona explícitamente sobre **pares** de objetos. Para cada par $(o_i, o_j)$ se calcula una función de relación, y las relaciones se agregan para responder la pregunta. Resolvió el dataset CLEVR — un benchmark de razonamiento visual sintético — donde CNNs puras fallaban.

### Qué la destronó

Estas arquitecturas eran ad hoc para captioning/VQA. Cada nueva tarea requería diseñar un módulo nuevo. Faltaba un **paradigma de pretraining transferible** — el equivalente a BERT para multimodal.

## Era 3 — Pretraining multimodal (2019-2020)

### Problema heredado

BERT (2018) había mostrado que pretraining masivo + fine-tuning destronaba a soluciones especializadas en NLP. La pregunta natural: ¿se puede aplicar la misma receta a texto + imagen?

### Idea clave

**BERT-style para texto e imagen.** ViLBERT (Lu et al., 2019) propuso dos torres BERT — una para texto, otra para imagen como secuencia de regiones detectadas — con **cross-attention** entre ellas en cada capa. El pretraining usa Masked Language Modeling extendido: enmascarar tokens de texto, predecirlos con contexto visual; enmascarar regiones, predecirlas con contexto textual.

LXMERT (Tan & Bansal, 2019) fijó el patrón con tres encoders (texto, visión, cross-modal) y cinco tareas de pretraining sobre Conceptual Captions. UNITER y OSCAR (2020) lo simplificaron a un solo Transformer unificado, agregando **etiquetas de objetos** como anclas semánticas que aproximan textos e imágenes en el mismo vocabulario.

### Qué la destronó

Estos modelos dependían de **detectores Faster R-CNN preentrenados** — caros, lentos y un cuello de botella conceptual. Y los datasets curados (Conceptual Captions, COCO) eran pequeños comparados con el texto disponible para pretraining de NLP. CLIP demostró que se podía evitar ambos problemas con datos web ruidosos a escala.

## Era 4 — Contrastiva y zero-shot (2021-2022)

### Problema heredado

Los modelos de pretraining multimodal eran lentos (cada imagen requería detección + atención cruzada profunda) y limitados a las tareas vistas en pretraining. Cambiar el dominio de aplicación requería fine-tuning. La pregunta abierta: ¿hay una arquitectura más simple que aprenda representaciones genéricamente útiles?

### Idea clave

**Aprendizaje contrastivo a escala web.** CLIP (Radford et al., 2021) entrenó un encoder de texto (Transformer) y uno de imagen (ViT o ResNet) sobre 400 millones de pares imagen-caption raspados de internet. La pérdida es contrastiva: dado un batch de N pares, el modelo debe identificar cuál caption corresponde a cuál imagen entre las $N \times N$ combinaciones posibles.

El resultado son dos encoders cuyos embeddings viven en un **espacio compartido**. Para clasificar zero-shot una imagen entre clases arbitrarias, basta describir cada clase como texto ("una foto de un akita"), embeber esos textos, embeber la imagen, y elegir la clase con mayor similitud coseno. CLIP iguala o supera a modelos supervisados especializados en docenas de benchmarks sin haber visto un solo ejemplo de las tareas.

ALIGN (2021, Google) replicó la idea a 1.8B pares ruidosos. BLIP (2022) unificó entendimiento y generación. El paradigma se volvió la base de toda la generación texto-imagen posterior — Stable Diffusion, DALL·E 2 e Imagen condicionan sobre el text encoder de CLIP.

### Qué la destronó

CLIP entendía pero no generaba, y no razonaba. Su embedding compartido es útil pero los detalles finos (composición, conteo, posiciones relativas) escapan a la similitud coseno. La frontera se movió a integrar visión **dentro** de los LLMs y a generación cross-modal de alta fidelidad.

## Era 5 — VLMs y generación (2022-presente)

### Problema heredado

CLIP había abierto dos preguntas: (1) cómo darle visión a un LLM general sin reentrenarlo desde cero, y (2) cómo invertir el proceso — generar imagen y video desde texto, no solo entenderlos.

### Idea clave

Dos líneas convergentes:

1. **Modelos visión-lenguaje (VLMs).** Flamingo (DeepMind, 2022) introdujo el patrón "**LLM congelado + adaptador visual**": un encoder de imagen (ViT preentrenado) produce tokens visuales que se inyectan en el LLM vía cross-attention adaptado, sin tocar los pesos del LLM. LLaVA (2023) lo simplificó a un proyector lineal entre CLIP-ViT y LLaMA, entrenado con instrucciones sintéticas generadas por GPT-4. GPT-4V, Gemini y Claude llevaron la idea a producción con visión nativa desde el pretraining.

2. **Generación texto-a-imagen/video.** Stable Diffusion, DALL·E 2 e Imagen (2022) usan **modelos de difusión** condicionados en el text encoder de CLIP o T5. La difusión aprende a denoise iterativamente desde ruido gaussiano hasta la imagen final, guiada por el embedding de texto en cada paso. Sora (2024) y Veo (2024-2025) extendieron la receta a video con coherencia temporal de 10-60 segundos.

### Qué viene

El campo está convergiendo hacia **modelos any-to-any** que aceptan y producen cualquier combinación de modalidades (texto, imagen, audio, video, acción). Las direcciones activas: agentes visuales que actúan sobre interfaces gráficas, generación de video con física correcta, integración con robótica vía Vision-Language-Action (RT-2, π0, OpenVLA — pendientes de la Ola 4 de Dominios), y modelos pequeños multimodales competentes (eficiencia por destilación). La pregunta abierta: si los frontier LLMs se vuelven multimodales nativos por defecto, ¿queda "multimodal" como disciplina aislada o se diluye en el modelado general de secuencias?

## Estado del arte hoy

{{< callout type="info" >}}

**Frontier multimodal (2024-2025).** La visión y el lenguaje ya no compiten por arquitectura; conviven dentro de los frontier LLMs como capacidad nativa. La generación cross-modal alcanza calidad de producto.

- **GPT-5 (Vision)** — OpenAI. Multimodal nativo desde el pretraining; razona sobre imagen, audio y texto en una sola sesión.
- **Gemini 2.5 Pro Vision** — Google DeepMind. Razonamiento sobre imágenes y video largo, integración con búsqueda y herramientas.
- **Claude Opus 4.7** — Anthropic. Visión integrada al razonamiento extendido; foco en imágenes técnicas y diagramas.
- **LLaMA 4 Multimodal** — Meta. Open weights con visión nativa; competitivo con frontera cerrada en muchos benchmarks.
- **Sora 2 / Veo 2** — OpenAI / Google DeepMind. Generación de video con coherencia temporal extendida (60s+) y control fino por prompt.
- **Stable Diffusion 3 / Imagen 3** — Stability AI / Google. Generación texto-imagen producción.
- **DINOv2** — Meta. Foundation model self-supervised para visión, base de muchas pipelines downstream multimodales.

{{< /callout >}}

## Casos de uso reales

- **Búsqueda visual con texto**: Google Lens, Pinterest Lens, Amazon — search por imagen + descripción combinada (CLIP-style retrieval).
- **Asistentes conversacionales con visión**: ChatGPT, Claude, Gemini — leer documentos, analizar gráficos, depurar capturas de pantalla, describir escenas.
- **Generación creativa**: Midjourney, DALL·E, Adobe Firefly — concept art, marketing, diseño de producto, ilustración editorial.
- **Generación de video**: Sora, Veo — previsualización, marketing, contenido corto en redes.
- **Accesibilidad**: descripción automática de imágenes para personas con discapacidad visual; transcripción enriquecida con contexto visual.
- **Análisis de documentos visión + texto**: facturas, historias clínicas, papers científicos con figuras, contratos con sellos y firmas — donde el OCR puro se queda corto.
- **Robótica vision-language-action**: RT-2, π0, OpenVLA — el robot recibe instrucciones en lenguaje natural y ve su entorno (cubierto en profundidad en la Ola 4 de Dominios, dominio Robótica/RL).

## Qué viene

Las apuestas activas en multimodal hoy: **modelos any-to-any** (entrada y salida arbitraria entre texto, imagen, audio, video, acción), **generación de video con física correcta** (más allá de Sora — coherencia de identidad, conservación de masa, causalidad), **agentes visuales** que actúan sobre interfaces gráficas como un humano (clicks, scrolls, formularios), **modelos pequeños multimodales** competentes vía destilación, e **integración profunda con herramientas** (visión + búsqueda web + ejecución de código). La pregunta abierta de 2025: si los frontier LLMs se vuelven multimodales por defecto, ¿queda "multimodal" como disciplina aislada o se diluye en el modelado general de secuencias?

## Recursos relacionados

**Fundamentos:**
- [Aprendizaje contrastivo (CLIP)](/fundamentos/aprendizaje-contrastivo) — el puente entre visión y lenguaje.
- [Foundation models](/fundamentos/foundation-models) — el contexto general de los VLMs frontier.
- [Mecanismo de atención](/fundamentos/mecanismo-atencion) — la base de show-attend-tell y todas las arquitecturas posteriores.
- [Self-attention](/fundamentos/self-attention) y [Vision Transformer](/fundamentos/vision-transformer) — las piezas que CLIP combinó.

**Papers:**
- [Show and Tell (Vinyals 2015)](/papers/show-and-tell-vinyals-2015).
- [Show, Attend and Tell (Xu 2015)](/papers/show-attend-tell-xu-2015).
- [VQA (Antol 2015)](/papers/vqa-antol-2015) — el paper fundacional de Visual Question Answering.
- [Stacked Attention (Yang 2016)](/papers/stacked-attention-yang-2016) — atención visual multi-hop.
- [MCB (Fukui 2016)](/papers/mcb-fukui-2016) — fusión bilineal compacta.
- [Relation Networks (Santoro 2017)](/papers/relation-networks-santoro-2017).
- [VQAv2 (Goyal 2017)](/papers/vqav2-goyal-2017) — dataset balanceado contra language priors.
- [MUTAN (Ben-younes 2017)](/papers/mutan-ben-younes-2017) — fusión por descomposición de Tucker.
- [Bottom-Up Attention (Anderson 2018)](/papers/bottom-up-attention-anderson-2018).
- [Pythia (Jiang 2018)](/papers/pythia-jiang-2018) — ganador del VQA Challenge 2018.
- [BLEU (Papineni 2002)](/papers/bleu-papineni-2002) — métrica de evaluación de captions.
- [CLIP (Radford 2021)](/papers/clip-radford-2021).
- [Bahdanau attention (2015)](/papers/bahdanau-attention-2015) — antecedente de show-attend-tell.

**Fundamentos:**
- [Visual Question Answering](/fundamentos/visual-question-answering) y [Image Captioning](/fundamentos/image-captioning) — las dos tareas de la [Clase 23](/clases/clase-23).
- [BLEU Metric](/fundamentos/bleu-metric) — la métrica de evaluación de generación.

**Dominios relacionados:**
- [Texto / NLP](/dominios/texto) — la mitad lingüística de los VLMs.
- [Visión](/dominios/vision) — la mitad visual.

---

*Última actualización: 2026-06-01.*
