---
title: "VisualBERT (2019)"
weight: 317
math: true
---

{{< paper-card
    title="VisualBERT: A Simple and Performant Baseline for Vision and Language"
    authors="Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, Kai-Wei Chang"
    year="2019"
    venue="arXiv"
    pdf="/papers/visualbert-li-2019.pdf"
    arxiv="1908.03557" >}}
Paper germinal de la familia de **Vision-Language Models (VLM) pre-entrenados**, citado en la [Clase 28](/clases/clase-28) como puente entre la autosupervisión en lenguaje y la autosupervisión multimodal. Su tesis es deliberadamente minimalista: **no hace falta una arquitectura de fusión elaborada para tareas de visión-y-lenguaje; basta con tomar BERT, meterle dentro las regiones de la imagen como si fueran más tokens, y dejar que la self-attention descubra sola las alineaciones palabra-región**. Un único Transformer ("single-stream"), pre-entrenado de forma auto-supervisada sobre captions de COCO con *masked language modeling* visualmente fundamentado, iguala o supera modelos task-specific mucho más complejos en VQA, VCR, NLVR² y grounding.
{{< /paper-card >}}

---

## Contexto

Las tareas que combinan **visión y lenguaje** —captioning, *visual question answering* (VQA), razonamiento visual, grounding— exigen mucho más que reconocer objetos: requieren entender atributos, partes, relaciones espaciales, acciones e intenciones, y cómo todos esos conceptos se *anclan* (ground) en lenguaje natural. Hacia 2018-2019 el paradigma dominante eran pipelines task-specific de cuatro piezas: un codificador de texto, un extractor de features de imagen (casi siempre un detector tipo Faster R-CNN con atención *bottom-up*), un módulo de fusión multimodal hecho a mano y un clasificador de respuestas, todo diseñado a medida para cada tarea.

El otro hilo que confluye es el de los **codificadores de lenguaje pre-entrenados** con objetivos auto-supervisados: ELMo, GPT y sobre todo [BERT](/papers/bert-devlin-2018). BERT demostró que pre-entrenar un Transformer con *masked language modeling* sobre texto crudo —una señal sin etiquetas humanas— produce representaciones que transfieren a casi cualquier tarea de NLP con solo fine-tuning. VisualBERT responde afirmativamente a la pregunta natural: **¿se puede hacer lo mismo con imagen + texto?**

Dos trabajos concurrentes comparten la idea: **VideoBERT** (Sun et al., 2019), que empareja palabras habladas con frames de video, y **ViLBERT** (Lu et al., 2019), que también propone pre-entrenamiento tipo BERT pero usa **dos Transformers separados** (uno de visión, uno de lenguaje) que se atienden vía co-atención, duplicando los parámetros. La marca de identidad de VisualBERT frente a ViLBERT es **un solo stack, no dos**: texto e imagen viven en la misma secuencia desde la primera capa y la self-attention cruza modalidades libremente (*early fusion*).

## Arquitectura: single-stream

VisualBERT usa la configuración exacta de **BERT-base** (12 capas, hidden 768, 12 cabezas) e inicializa desde sus pesos públicos. La única adición es un conjunto de **embeddings visuales** `F`. Cada `f ∈ F` corresponde a una **región acotada (bounding region)** de la imagen, derivada de un detector de objetos, y se computa sumando tres componentes en paralelo perfecto al diseño de BERT:

- `f_o`: la **representación visual** de la región, computada por una CNN (las features del detector).
- `f_s`: un **segment embedding** que indica que es un embedding de imagen y no de texto.
- `f_p`: un **position embedding**, usado cuando las alineaciones palabra-región vienen dadas en la entrada (caso VCR); se fija a la suma de los position embeddings de las palabras alineadas.

Estos embeddings visuales se concatenan con los de texto en **una sola secuencia** y se pasan al Transformer multicapa. El modelo descubre implícitamente alineaciones útiles entre ambos conjuntos y construye una representación conjunta, sin módulo de fusión especializado ni parámetros extra significativos. Las regiones se tratan como **tokens no ordenados** —no hay un orden natural entre cajas— a diferencia de las palabras. Reutilizar la self-attention para hacer el trabajo que antes hacían módulos de atención a medida es el corazón de la propuesta.

## Pre-entrenamiento auto-supervisado

VisualBERT recurre a datos pareados —**COCO**, donde cada imagen viene con 5 captions— y se entrena en tres fases:

**Task-Agnostic Pre-Training (sobre COCO).** Dos objetivos [auto-supervisados](/fundamentos/aprendizaje-autosupervisado) visualmente fundamentados:

1. *Masked language modeling con la imagen.* Se enmascaran tokens de texto y el modelo debe predecirlos; las regiones de imagen **no** se enmascaran. La predicción debe apoyarse, por tanto, en el texto restante *y* en el contexto visual. Es el [MLM de BERT](/papers/bert-devlin-2018) llevado a lo multimodal.
2. *Sentence-image prediction.* Se da un par de captions: una describe la imagen, la otra tiene 50% de probabilidad de ser otra caption de la misma imagen y 50% de ser aleatoria. El modelo aprende a distinguir ambos casos —análogo multimodal del *next sentence prediction*.

Ambos son auto-supervisados: la señal proviene de la **estructura de los datos** (texto enmascarado, emparejamientos imagen-texto correctos vs. corruptos), no de anotaciones humanas.

**Task-Specific Pre-Training.** Antes del fine-tuning se entrena con los datos de la tarea destino usando el MLM con imagen, para adaptar el modelo al dominio nuevo (p.ej. escenas de películas en VCR, muy distintas de COCO).

**Fine-Tuning.** Igual que en BERT: entrada, salida y objetivo específicos de la tarea. El pre-entrenamiento sobre COCO toma menos de un día en 4 Tesla V100; todos los experimentos caben en a lo más 4 V100 de 16 GB.

## Resultados

Se evalúan cuatro tareas. Por cada una se reportan tres variantes de diagnóstico: el modelo completo, **w/o Early Fusion** (la imagen se combina solo en una capa final, no desde la primera) y **w/o COCO Pre-training** (sin la fase task-agnostic).

| Tarea | Métrica | SOTA previo | VisualBERT |
|---|---|---|---|
| **VQA 2.0** (test-dev) | accuracy | Pythia v0.3: 68.71 | **70.80** |
| **VCR** (Q→AR test) | accuracy | R2C: 44.0 | **52.4** |
| **NLVR²** (test-P) | accuracy | MaxEnt: 54.8 | **67.0** |
| **Flickr30K** (R@1 test) | recall | BAN: 69.69 | **71.33** |

En condiciones comparables (mismas features, mismo número de regiones) VisualBERT supera a métodos significativamente más complejos. En **VQA** bate a Pythia siendo más simple. En **VCR** —razonamiento de sentido común sobre escenas de películas— incluso la variante sin pre-entrenamiento COCO supera a R2C por amplio margen, y el modelo completo mejora más: el pre-entrenamiento sobre COCO ayuda **pese a la enorme brecha de dominio**. En **NLVR²** (verificar una caption sobre un par de imágenes) las dos ablaciones ya superan a MaxEnt y el modelo completo ensancha la brecha. En **Flickr30K** (region-to-phrase grounding) supera al SOTA BAN.

Las **ablaciones** sobre NLVR² aíslan las decisiones clave: las dos más importantes son el **pre-entrenamiento task-agnostic** (quitarlo cae de 66.7 a 62.9 dev) y el **early fusion** (sin él, 61.4, el peor resultado). La inicialización desde BERT y el objetivo sentence-image prediction ayudan pero menos: el modelo recupera mucho durante el pre-entrenamiento multimodal.

## Grounding emergente

El hallazgo más conceptual: tras el pre-entrenamiento, **muchas cabezas de atención anclan entidades del texto a las regiones correctas de la imagen con alta precisión, sin ninguna supervisión de grounding**. Sobre Flickr30K como diagnóstico, antes de cualquier fine-tuning, VisualBERT supera a un baseline que siempre elige la región de mayor confianza de detección, y la precisión **mejora en las capas altas**: el modelo está menos seguro al sintetizar ambas entradas abajo y se vuelve más certero arriba.

Hay también **grounding sintáctico**: para cada tipo de relación de dependencia existe al menos una cabeza que predice el anclaje correcto muy por encima del azar, asociando argumentos con verbos (*pobj*, *nsubj*, *dobj*). Cualitativamente, VisualBERT refina alineaciones a través de las capas —en un ejemplo, "husband" y "woman" atienden inicialmente ambos a la región de la mujer y al final el modelo los ha **desenredado**, resolviendo incluso correferencia ("her" → la mujer). El grounding palabra-región no se programa: *emerge* del pretexto auto-supervisado.

## Limitaciones reconocidas

- **Dependencia de un detector externo.** Las regiones vienen de un detector pre-entrenado (Faster R-CNN / Detectron / ResNet50 según la tarea); la calidad del grounding queda acotada por qué objetos detecta. El modelo razona sobre regiones ya propuestas, no sobre píxeles crudos.
- **Brecha de dominio.** Se pre-entrena sobre captions de COCO (escenas cotidianas); el paso de *task-specific pre-training* existe precisamente para cerrar esa brecha.
- **Escala modesta.** ~100k imágenes de COCO; el propio paper apunta a pre-entrenar sobre datasets mayores (Visual Genome, Conceptual Captions) como trabajo futuro.
- **Carácter de "Work in Progress".** Es un preprint preliminar; varias decisiones (número de propuestas, features por tarea) se exploran de forma limitada.

## Por qué importa para la Clase 28

La [Clase 28](/clases/clase-28) cita a Li et al. 2019 en la sección de multimodalidad, y la conexión es directa:

- **El MLM auto-supervisado, de una modalidad a dos.** La clase explica la [autosupervisión](/fundamentos/aprendizaje-autosupervisado) en lenguaje con BERT: enmascarar y predecir, sin etiquetas. VisualBERT toma ese *mismo pretexto* y lo vuelve multimodal —el ejemplo canónico de cómo un objetivo de SSL probado en lenguaje se generaliza a visión-y-lenguaje.
- **Continuidad con BERT.** VisualBERT *es* [BERT](/papers/bert-devlin-2018) con embeddings visuales añadidos e inicializado desde sus pesos; la ablación muestra cuánto (y cuán poco) aporta esa inicialización una vez que entra el pre-entrenamiento multimodal.
- **Grounding emergente como señal de SSL exitoso.** Que las cabezas aprendan a anclar entidades *sin supervisión de grounding* es el tipo de "estructura útil que emerge del pretexto" que la clase celebra en RotNet, SimCLR o MAE; aquí el pretexto es multimodal y lo que emerge es alineación palabra-región.
- **Tarea sustrato.** La evaluación principal incluye [Visual Question Answering](/fundamentos/visual-question-answering), la tarea multimodal por antonomasia, que VisualBERT aborda con un Transformer único y pre-entrenamiento auto-supervisado.
- **Puente hacia los VLM.** VisualBERT, junto con ViLBERT y VideoBERT, marca el momento en que el paradigma *pretraining-then-finetuning* cruza a lo multimodal, antecediendo a LXMERT, UNITER, OSCAR y, en el [dominio multimodal](/dominios/multimodal), a CLIP y los VLM modernos que sustituyen el detector por features de patches tipo ViT.

## Notas y enlaces

- Preprint: arXiv:1908.03557v1 (9 ago 2019), [arxiv.org/abs/1908.03557](https://arxiv.org/abs/1908.03557).
- Stack: Transformer encoder con configuración de BERT-base, inicializado desde los pesos públicos de Devlin et al. (2019). Features de imagen de detectores tipo Faster R-CNN.
- Afiliaciones: UCLA, Allen Institute for AI, Peking University.

**Cross-links:** [Clase 28](/clases/clase-28) · [BERT (Devlin et al., 2018)](/papers/bert-devlin-2018) · [Aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado) · [Visual Question Answering](/fundamentos/visual-question-answering) · [Dominio multimodal](/dominios/multimodal).
