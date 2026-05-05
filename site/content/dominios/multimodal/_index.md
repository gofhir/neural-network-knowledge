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
    {{< hito year="2015" name="VQA dataset" status="minimal" >}}
      Antol et al.: Visual Question Answering como benchmark — *"¿De qué color es el casco?"* sobre una imagen. **Por qué importó:** forzó al campo más allá del captioning hacia razonamiento sobre imagen + texto.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era de atención visual estructurada" years="2017-2018" >}}
    {{< hito year="2017" name="Relation Networks" status="covered" link="/papers/relation-networks-santoro-2017" >}}
      Santoro et al.: módulo que razona sobre **pares de objetos** detectados, no sobre features uniformes de la imagen. Resolvió tareas de razonamiento visual donde CNNs puras fallaban.
    {{< /hito >}}
    {{< hito year="2018" name="Bottom-Up and Top-Down Attention" status="covered" link="/papers/bottom-up-attention-anderson-2018" >}}
      Anderson et al.: la atención visual opera sobre regiones detectadas con Faster R-CNN, no sobre la grilla densa. Estado del arte en captioning y VQA durante 2018-2019.
    {{< /hito >}}
    {{< hito year="2018-2019" name="GQA / VCR" status="minimal" >}}
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
