---
title: "BLIP (2022)"
weight: 247
math: true
---

{{< paper-card
    title="BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation"
    authors="Li, Li, Xiong, Hoi"
    year="2022"
    venue="ICML 2022 / arXiv"
    pdf="/papers/blip-li-2022.pdf"
    arxiv="2201.12086" >}}
BLIP (Salesforce Research) es un *framework* de pre-entrenamiento visión-lenguaje que persigue un objetivo único: que **un solo modelo** transfiera bien tanto a tareas de **comprensión** (image-text retrieval, [VQA](/fundamentos/visual-question-answering)) como de **generación** ([image captioning](/fundamentos/image-captioning)). Lo logra con dos contribuciones complementarias: **MED** (*Multimodal mixture of Encoder-Decoder*), una arquitectura que se viste de encoder o de decoder según convenga, y **CapFilt**, un método de *bootstrapping* que limpia y enriquece los datos web ruidosos. Resultados estado del arte: **+2.7% recall@1** en retrieval, **+2.8% CIDEr** en captioning, **78.17 en VQA test-std**, y un *zero-shot* a video sorprendentemente fuerte. Es el modelo "vivo" del [Laboratorio 23](/laboratorios/lab-23), en contraste con [Pythia](/papers/pythia-jiang-2018), el VQA clásico de la teoría.
{{< /paper-card >}}

---

## Contexto: encoder vs encoder-decoder, y datos web ruidosos

A inicios de 2022 el pre-entrenamiento visión-lenguaje (VLP) arrastraba dos limitaciones que el paper enuncia con precisión.

**La limitación de modelo.** Los métodos caían en dos campos arquitectónicos y ninguno servía para todo:

- **Modelos basados en encoder.** [CLIP](/papers/clip-radford-2021) y ALBEF alinean imagen y texto con objetivos contrastivos o de *matching*. Son excelentes para **comprensión** (retrieval, clasificación), pero "no se transfieren de forma directa a generación de texto": un encoder no produce secuencias autoregresivas, así que hacer captioning con CLIP exige acoplarle un decoder ajeno.
- **Modelos encoder-decoder.** SimVLM y VL-T5 sí generan texto, pero "no han sido adoptados con éxito para image-text retrieval": un encoder-decoder con cross-attention densa no produce de forma natural los *embeddings* comparables que el retrieval necesita.

El diagnóstico de BLIP: ni los encoder puros ni los encoder-decoder puros sobresalen en ambos tipos de tarea a la vez. La solución no es elegir uno, sino **una mezcla flexible** que pueda vestirse de cualquiera de los tres modos según la necesidad — de ahí *mixture of encoder-decoder*. BLIP es el sucesor directo de ALBEF (del mismo grupo), del que hereda el [ViT](/papers/vit-dosovitskiy-2021) como encoder de imagen, la pérdida contrastiva con encoder de momento y el *hard negative mining*.

**La limitación de datos.** El estado del arte (CLIP, ALBEF, SimVLM, ALIGN) se entrenaba con pares imagen–*alt-text* rastreados de la web, que "a menudo no describen con precisión el contenido visual". Es una señal ruidosa que la comunidad había ignorado porque escalar el dataset compensaba en agregado. BLIP demuestra que ese ruido es subóptimo y que la web se aprovecha mejor **"bootstrapeando" los captions**: generar texto sintético limpio y filtrar el ruido.

---

## Contribución (a): MED, la arquitectura camaleón

**El image encoder es un ViT, no un detector.** BLIP abandona el detector de objetos de la era up-down/Pythia. La imagen se divide en parches, se codifica como una secuencia de *embeddings* y se antepone un token **[CLS]** que resume la característica global. Es "más amigable computacionalmente" que extraer características con detectores (lentos, requieren cajas anotadas). Se exploran dos variantes, **ViT-B/16** y **ViT-L/16**, inicializadas desde pesos de ImageNet.

**Los tres modos del MED.** El texto se procesa con un transformer inicializado desde BERT$_{\text{base}}$. El mismo bloque puede operar en tres modos, cada uno con su objetivo:

1. **Unimodal encoder — objetivo ITC.** Codifica imagen y texto **por separado**, sin interacción cruzada; el texto recibe un token **[CLS]**. Se entrena con la **pérdida contrastiva imagen-texto (ITC)**, que alinea el espacio del ViT con el del encoder de texto (positivos cerca, negativos lejos). Sigue la formulación de ALBEF, con encoder de momento y *soft labels*.
2. **Image-grounded text encoder — objetivo ITM.** Inyecta información visual con **una capa de cross-attention adicional** entre la self-attention (aquí **bidireccional**) y la FFN de cada bloque. Un token **[Encode]** produce la **representación multimodal** del par. Se entrena con la **pérdida de emparejamiento imagen-texto (ITM)**, una clasificación binaria *matched/unmatched* con *hard negative mining*.
3. **Image-grounded text decoder — objetivo LM.** Reemplaza la self-attention bidireccional por **self-attention causal** (cada token solo ve los anteriores, como GPT), con un token **[Decode]**. Se entrena con la **pérdida de modelado de lenguaje (LM)**, la cross-entropy autoregresiva que dota al modelo de la **capacidad generativa** que los modelos tipo BERT no tienen.

**El truco del compartir parámetros: todo excepto la SA.** Aquí está la elegancia del MED. El encoder y el decoder de texto **comparten todos los parámetros salvo las capas de self-attention (SA)**. La razón: "las diferencias entre codificar y decodificar se capturan mejor en las capas SA". El encoder necesita SA **bidireccional** (ver el input completo); el decoder necesita SA **causal** (no ver el futuro) — funciones incompatibles en una misma capa. En cambio, embeddings, cross-attention y FFN funcionan igual en ambas tareas, así que compartirlas mejora la eficiencia y aprovecha el *multi-task learning*. La ablación lo confirma: compartir todo-excepto-SA da NoCaps CIDEr 102.2 con 252M parámetros, mejor que no compartir nada (101.8, 361M); compartir **también** las SA degrada el desempeño "por el conflicto entre codificación y decodificación".

Cada par imagen-texto requiere **un solo forward pass por el ViT** (el componente pesado) y tres por el transformer de texto (uno por objetivo), de modo que el cuello de botella visual se computa una sola vez.

---

## Contribución (b): CapFilt, el bootstrapping de datos

CapFilt es la idea más memorable del paper. Resuelve el ruido web sin renunciar a la escala de la web. El punto de partida: hay **pocos** pares imagen-texto de alta calidad anotados por humanos $\{(I_h, T_h)\}$ (principalmente COCO) y **muchísimos** pares web ruidosos $\{(I_w, T_w)\}$.

A partir del **mismo MED pre-entrenado** se derivan dos módulos, ambos afinados sobre COCO:

- **El Captioner** es un *image-grounded text decoder* afinado con LM. Aplicado a las imágenes web $I_w$, genera **captions sintéticos $T_s$**.
- **El Filter** es un *image-grounded text encoder* afinado con ITC+ITM. Recorre **tanto los textos web $T_w$ como los sintéticos $T_s$** y marca como ruidoso todo el que la cabeza ITM prediga como *unmatched*.

El dataset bootstrapeado combina los pares que sobreviven al filtro más los pares humanos:

$$D = \{(I_w, T_w)\}_{\text{filtrados}} + \{(I_w, T_s)\}_{\text{filtrados}} + \{(I_h, T_h)\}.$$

Con $D$ se **pre-entrena un modelo nuevo desde cero**.

**Nucleus sampling, no beam search.** Un hallazgo no obvio: *cómo* se generan los captions sintéticos importa. El [nucleus sampling](/fundamentos/decoding-strategies) (muestreo estocástico del núcleo, $p=0.9$) **gana** al beam search determinista (TR@1 80.6 vs 79.6; NoCaps CIDEr 105.1 vs 103.5), a pesar de producir captions más ruidosos (25% de rechazo del filtro vs 19%). La hipótesis: nucleus sampling genera "captions más diversos y sorprendentes, con más información nueva de la que el modelo puede beneficiarse", mientras beam search produce "captions seguros y comunes". La lección destilada: **más diversidad rinde más ganancia**.

**Captioner y Filter deben estar desacoplados.** Si comparten parámetros, el desempeño cae por **sesgo de confirmación**: un Filter con los pesos del Captioner es menos propenso a rechazar el ruido que el propio Captioner produjo (la tasa de ruido baja artificialmente a 8% vs 25%). Por eso se afinan **por separado**. El paper enmarca todo CapFilt como una forma de *knowledge distillation*, lo que explica por qué conviene entrenar un modelo **nuevo** (el estudiante no se inicializa desde el profesor) en vez de continuar el entrenamiento.

---

## Experimentos clave

**El efecto de CapFilt (la tabla pivote).** Sobre 14M imágenes con ViT-B, partir del web crudo da TR@1 78.4 / NoCaps CIDEr 102.2; solo Filter sube a 79.1 / 102.7; solo Captioner a 79.7 / 103.4; y **C+F juntos** a **80.6 / 105.1**. Además escala: con 129M imágenes y módulos basados en ViT-L, el modelo base llega a TR@1 81.2 / CIDEr 109.6.

**Image-text retrieval (COCO, Flickr30K).** Con fine-tuning vía ITC+ITM y un reranking eficiente (k candidatos por similitud, reordenados por score ITM), BLIP supera a ALBEF en **+2.7% average recall@1** con las mismas 14M imágenes. En *zero-shot* a Flickr30K supera a CLIP (400M pares) y ALIGN (1.8B), pese a usar solo 14M–129M.

**Image captioning (COCO, NoCaps).** Con un prompt "a picture of", BLIP-129M con ViT-L alcanza **COCO BLEU@4 = 40.4 / CIDEr = 136.7** y NoCaps CIDEr 113.2, competitivo con LEMON y SimVLM (que usan detectores pesados o 1.8B imágenes) pero **sin detector** y a menor resolución, mucho más rápido en inferencia.

**VQA (VQAv2) — crucial para la Clase 23.** En lugar de tratar VQA como **clasificación multi-respuesta** (el enfoque de Pythia), BLIP la formula como **generación de respuestas** (open-ended): imagen+pregunta se codifican en embeddings multimodales que se pasan a un **answer decoder** entrenado con LM. Con 14M imágenes supera a ALBEF en +1.64%; con 129M alcanza **test-dev 78.24 / test-std 78.17**, superando a SimVLM que usó **13× más datos**.

**Zero-shot a video.** Transfiriendo directamente los modelos de COCO-retrieval y VQA, muestreando frames y concatenando sus features, **ignorando toda información temporal**, BLIP *zero-shot* en MSRVTT (R@1 = 43.3) **supera incluso a modelos afinados sobre el dataset de video** en +12.4% recall@1. Es la demostración más fuerte de su generalización.

---

## Limitaciones

El paper es explícito sobre frentes abiertos: CapFilt se aplica en **una sola ronda** (sugiere múltiples); se genera **un solo caption sintético** por imagen; no hay **ensemble** de Captioners/Filters; la transferencia a video **ignora el tiempo** (un TimeSformer afinado mejoraría); y Captioner/Filter dependen de COCO, heredando su sesgo de dominio (visible en el *domain gap* de NLVR2). A esto se suma una limitación implícita clave para el laboratorio: como modelo **generativo**, BLIP puede **alucinar** respuestas plausibles pero incorrectas ante conceptos fuera de su distribución de entrenamiento.

---

## Por qué importa hoy

- **BLIP-2 (2023)** introduce el **Q-Former**, un transformer ligero que puentea un encoder de imagen **congelado** con un **LLM congelado** (OPT, FlanT5), abaratando drásticamente el entrenamiento. **InstructBLIP** añade *instruction tuning* sobre BLIP-2. El patrón "encoder de imagen congelado + puente entrenable + LLM" es ancestral de LLaVA, MiniGPT-4 y los VLMs instruccionales modernos.
- **CapFilt anticipa el re-captioning.** Usar el propio modelo para limpiar y enriquecer sus datos de entrenamiento prefigura las pipelines de *re-captioning* sintético de hoy (los captions densos de DALL·E 3 y de muchos datasets de VLMs actuales).
- **El MED como plantilla.** El transformer que conmuta entre encoder bidireccional y decoder causal compartiendo todo menos la SA reaparece en numerosos diseños posteriores de modelos unificados comprensión-generación.

> **Aclaración importante (confusión común).** BLIP-1 — el modelo de este paper y del Lab 23 — usa el **MED**, no el Q-Former. El **Q-Former es de BLIP-2 (2023)**, un trabajo posterior y distinto. El material de la Clase 23 mezcla ambos; conviene tener presente que cuando el Lab carga `blip-vqa-base` está usando la arquitectura MED original, no el puente a LLMs de BLIP-2.

---

## Conexión con el Laboratorio 23

El [Laboratorio 23](/laboratorios/lab-23) **usa BLIP en código**, lo que lo convierte en la contraparte práctica de la teoría de Pythia. Dos checkpoints de HuggingFace: `Salesforce/blip-vqa-base` (vía `BlipForQuestionAnswering`) para VQA y `Salesforce/blip-image-captioning-base` (vía `BlipForConditionalGeneration`) para captioning, ambos invocados con `model.generate(**inputs)`.

**El contraste pedagógico: generación vs clasificación.** La teoría explica VQA con [Pythia](/papers/pythia-jiang-2018), que es **clasificación multi-etiqueta sobre un vocabulario cerrado** (~3129 respuestas, clasificador sigmoide). BLIP hace VQA como **generación**: `model.generate` produce la respuesta token a token con el decoder. En Pythia la respuesta sale de un `argmax` sobre clases; en BLIP se **genera** como texto libre. La virtud es la apertura; el riesgo es la alucinación.

**El error del ornitorrinco.** El lab pregunta a propósito `"What kind of animal is this?"` sobre una imagen de ornitorrinco. Como es un animal raro y subrepresentado en los datos de pre-entrenamiento (COCO, Visual Genome, web), BLIP no tiene el concepto bien aprendido y genera con seguridad una etiqueta equivocada (pato, castor). Esto ilustra dos puntos a la vez: (1) **la generación abierta alucina** — un decoder generativo afirma una respuesta incorrecta, la cara oscura de la flexibilidad; (2) **los huecos de cobertura de los datos web importan** — CapFilt mejora la *calidad* de los captions, pero no inventa *conceptos nuevos* que la web no contiene. Comparar las respuestas de BLIP con las fallas de Pythia cierra el arco de la clase: del VQA-clasificación (detector + suma ponderada + softmax) al VQA-generación moderno (ViT + cross-attention + decoder autoregresivo), con sus nuevas fortalezas y sus nuevos modos de fallo.

---

## Notas y enlaces

- **Paper:** arXiv:2201.12086 — `https://arxiv.org/abs/2201.12086`
- **Código, modelos y dataset bootstrapeado:** `https://github.com/salesforce/BLIP`
- **Checkpoints del lab:** `Salesforce/blip-vqa-base`, `Salesforce/blip-image-captioning-base` (HuggingFace).
- **Predecesor imprescindible:** Li et al., *Align before Fuse (ALBEF)*, NeurIPS 2021 — BLIP hereda su ITC con encoder de momento, el *hard negative mining* de ITM y el VQA como generación.
- **Referencias clave:** [CLIP (Radford 2021)](/papers/clip-radford-2021) (contraste imagen-texto, encoder puro), SimVLM (Wang 2021, encoder-decoder con PrefixLM), [ViT (Dosovitskiy 2021)](/papers/vit-dosovitskiy-2021) (el encoder de imagen), BERT (Devlin 2019, init del transformer de texto), Holtzman et al. (nucleus sampling), Hinton et al. (KD).

Ver fundamentos: [Vision-Language Models](/fundamentos/vision-language-models) · [Visual Question Answering](/fundamentos/visual-question-answering) · [Image Captioning](/fundamentos/image-captioning) · [Estrategias de decodificación](/fundamentos/decoding-strategies) · [BLEU](/fundamentos/bleu-metric). Dominio: [Multimodal](/dominios/multimodal). Clase: [Clase 23](/clases/clase-23). Laboratorio: [Lab 23](/laboratorios/lab-23).
