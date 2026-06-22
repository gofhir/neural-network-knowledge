# VisualBERT: A Simple and Performant Baseline for Vision and Language — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *VisualBERT: A Simple and Performant Baseline for Vision and Language*.
- **Autores:** Liunian Harold Li (UCLA), Mark Yatskar (Allen Institute for Artificial Intelligence), Da Yin (Peking University), Cho-Jui Hsieh (UCLA), Kai-Wei Chang (UCLA).
- **Venue:** Preprint marcado como *Work in Progress*. Es la versión germinal del trabajo que luego se presentaría en la comunidad de visión-y-lenguaje (el modelo se volvió referencia citada como "VisualBERT, Li et al. 2019").
- **Año:** 2019. **Preprint:** arXiv:1908.03557v1 (9 ago 2019), [arxiv.org/abs/1908.03557](https://arxiv.org/abs/1908.03557).
- **Stack:** Transformer encoder con la configuración de BERT-base (12 capas, hidden 768, 12 cabezas de self-attention), inicializado desde los pesos públicos de BERT-base de Devlin et al. (2019). Features de imagen de detectores tipo Faster R-CNN.

La tesis del paper es minimalista y deliberadamente provocadora: **no hace falta una arquitectura de fusión multimodal elaborada para tareas de visión-y-lenguaje; basta con tomar BERT, meterle dentro las regiones de la imagen como si fueran más tokens, y dejar que la self-attention descubra sola las alineaciones entre palabras y regiones**. Los modelos previos de visión-y-lenguaje (VQA, razonamiento visual, grounding) se construían como pipelines de cuatro piezas: un codificador de texto, un extractor de features de imagen, un módulo de fusión multimodal (casi siempre con atención) y un clasificador de respuestas, y cada modelo se diseñaba a medida para una tarea concreta. VisualBERT colapsa esas piezas en **un único stack de Transformer** que procesa texto e imagen conjuntamente, y que se adapta a tareas nuevas con cambios mínimos.

La segunda mitad de la tesis es la que conecta directamente con la Clase 28: ese Transformer único se **pre-entrena de forma auto-supervisada** sobre datos de captions (COCO), usando objetivos de modelado de lenguaje *visualmente fundamentados* (visually-grounded language model objectives). Es decir, el masked language modeling de BERT —el objetivo auto-supervisado por excelencia en lenguaje— se lleva al territorio multimodal: el modelo predice palabras enmascaradas no solo a partir del texto restante sino también del contexto visual. Eso convierte a VisualBERT en uno de los primeros ejemplos limpios de **SSL multimodal de tipo BERT**, y en antecedente directo de toda la familia de VLM (Vision-Language Models) pre-entrenados que vendría después.

Para la Clase 28 (Aprendizaje Autosupervisado) importa porque el material cita explícitamente a Li et al. 2019 / VisualBERT en la sección de multimodalidad como el puente entre la autosupervisión en lenguaje (que la clase explica con BERT y el masked language modeling) y la autosupervisión multimodal. Entender este paper es entender cómo el *mismo* pretexto auto-supervisado —enmascarar y predecir— se generaliza de una sola modalidad a dos.

## 2. Contexto histórico: de pipelines a medida al pre-entrenamiento auto-supervisado multimodal

Las tareas que combinan visión y lenguaje —captioning, *visual question answering* (VQA), razonamiento visual— son, como dice el paper, un *test-bed* rico para evaluar la capacidad de razonamiento de sistemas visualmente informados. Van mucho más allá de reconocer qué objetos hay en una imagen (ImageNet, COCO detection): exigen entender atributos, partes, relaciones espaciales, acciones, intenciones, y cómo todos esos conceptos se refieren y se *anclan* (ground) en lenguaje natural.

Hacia 2018–2019 el paradigma dominante eran modelos task-specific con módulos de atención hechos a mano para fusionar las dos modalidades. Algunos trabajos modelaban explícitamente relaciones entre objetos con grafos (Li et al., 2019, relation-aware GAT); otros usaban atención para modelar implícitamente esas relaciones (Santoro et al., 2017; Norcliffe-Brown et al., 2018; Cadene et al., 2019). El extractor de features visuales casi siempre era un detector tipo Faster R-CNN con atención *bottom-up* (Anderson et al., 2018, el famoso *bottom-up top-down*), a veces enriquecido con anotaciones de atributos de Visual Genome.

El otro hilo que confluye es el de los **codificadores universales de lenguaje pre-entrenados con objetivos de modelado de lenguaje**: ELMo (Peters et al., 2018), GPT (Radford et al., 2018; 2019) y sobre todo **BERT** (Devlin et al., 2019). BERT demostró que pre-entrenar un Transformer con masked language modeling sobre texto crudo —una señal auto-supervisada, sin etiquetas humanas— produce representaciones que transfieren a casi cualquier tarea de NLP con solo fine-tuning. La pregunta natural, que VisualBERT responde afirmativamente, es: *¿se puede hacer lo mismo con imagen + texto?*

El paper reconoce dos trabajos concurrentes que comparten la idea:

- **VideoBERT** (Sun et al., 2019): convierte un video en palabras habladas emparejadas con una serie de imágenes y aplica un Transformer para aprender representaciones conjuntas. Arquitectura similar, pero evaluado sobre captioning de videos de cocina, no sobre el abanico de tareas que cubre VisualBERT.
- **ViLBERT** (Lu et al., 2019): también propone un pre-entrenamiento tipo BERT para visión-y-lenguaje, pero usa **dos Transformers separados** (uno de visión, uno de lenguaje) que solo se atienden mutuamente vía co-atención, lo que **duplica los parámetros**. Se pre-entrena sobre Conceptual Captions. Los resultados son consistentes con los de VisualBERT (que supera a ViLBERT en una de las dos tareas que comparten), aunque no son del todo comparables porque usan representaciones visuales y recursos de pre-entrenamiento distintos.

La diferencia arquitectónica con ViLBERT es la marca de identidad de VisualBERT: **un solo stack ("single-stream"), no dos**. Texto e imagen viven en la misma secuencia desde la primera capa, y la self-attention es libre de cruzar modalidades en cualquier capa. Esto es lo que el paper llamará *early fusion* y que las ablaciones demostrarán esencial.

## 3. Contribución central

VisualBERT aporta tres cosas que se sostienen mutuamente:

1. **Una arquitectura "single-stream" minimalista.** Un único Transformer (idéntico en configuración a BERT-base) que recibe la concatenación de embeddings de texto y embeddings visuales de regiones, y deja que la self-attention alinee implícitamente palabras y regiones, sin módulo de fusión especializado ni parámetros extra significativos. Reutiliza el mecanismo de self-attention para hacer el trabajo que antes hacían módulos de atención multimodal a medida.

2. **Dos objetivos de pre-entrenamiento auto-supervisados visualmente fundamentados**, aplicados sobre captions de COCO: (a) *masked language modeling con la imagen* y (b) *sentence-image prediction* (un análogo multimodal del *next sentence prediction* de BERT). Ambos son auto-supervisados: la señal proviene de la estructura de los datos (texto enmascarado, emparejamientos imagen-texto correctos vs. corruptos), no de anotaciones humanas adicionales.

3. **Evidencia de que la self-attention aprende grounding sin supervisión explícita.** El análisis muestra que, tras el pre-entrenamiento, muchas cabezas de atención anclan entidades del texto a las regiones correctas de la imagen con alta precisión, e incluso son sensibles a relaciones sintácticas (verbos atendiendo a las regiones de sus argumentos). Esto se logra **sin ninguna supervisión directa de grounding** durante el pre-entrenamiento.

La idea de diseño que une todo: el grounding palabra-región no se programa, *emerge* del pre-entrenamiento auto-supervisado. La self-attention, forzada a predecir palabras enmascaradas usando contexto visual, aprende por sí sola a mirar las regiones relevantes.

## 4. Método

### 4.1. Trasfondo: BERT

BERT es un Transformer con subpalabras como entrada. Cada subpalabra se mapea a un embedding `e` que es la suma de tres componentes: un *token embedding* `e_t` (específico de la subpalabra), un *segment embedding* `e_s` (indica de qué parte del texto viene el token, p.ej. premisa vs. hipótesis) y un *position embedding* `e_p` (posición en la secuencia). Esos embeddings pasan por un Transformer multicapa que construye representaciones contextualizadas. Se entrena en dos fases: pre-entrenamiento (masked language modeling + next sentence prediction) y fine-tuning task-specific.

### 4.2. La adaptación: embeddings visuales `F`

El núcleo de VisualBERT es introducir, además de todos los componentes de BERT, un conjunto de **embeddings visuales** `F`. Cada `f ∈ F` corresponde a una **región acotada (bounding region) de la imagen**, derivada de un detector de objetos. Cada `f` se computa sumando tres embeddings, en paralelo perfecto al diseño de BERT:

- `f_o`: la **representación visual** de la región, computada por una CNN (las features del detector).
- `f_s`: un **segment embedding** que indica que es un embedding de imagen y no de texto.
- `f_p`: un **position embedding**, usado cuando las alineaciones entre palabras y regiones vienen dadas como parte de la entrada (caso VCR); se fija a la suma de los position embeddings de las palabras alineadas.

Estos embeddings visuales se pasan al Transformer multicapa **junto con** los embeddings de texto, en una sola secuencia. El modelo descubre implícitamente alineaciones útiles entre ambos conjuntos y construye una representación conjunta. (Si texto e imagen tienen dimensiones distintas, las features visuales se proyectan al espacio del texto.) Las regiones se tratan como **tokens no ordenados** —no hay un orden natural entre cajas— a diferencia de las palabras.

### 4.3. Entrenamiento en tres fases

VisualBERT necesita aprender a acomodar ambas modalidades, así que recurre a datos pareados: **COCO**, donde cada imagen viene con 5 captions independientes. El procedimiento tiene tres fases:

- **Task-Agnostic Pre-Training (sobre COCO).** Dos objetivos auto-supervisados visualmente fundamentados:
  1. *Masked language modeling con la imagen.* Se enmascaran algunos tokens de texto y el modelo debe predecirlos; los vectores de las regiones de imagen **no** se enmascaran. La predicción de la palabra debe apoyarse, por tanto, en el texto restante *y* en el contexto visual.
  2. *Sentence-image prediction.* Como COCO tiene varias captions por imagen, se da un segmento de texto con dos captions: una describe la imagen, y la otra tiene 50% de probabilidad de ser otra caption de la misma imagen y 50% de ser una caption aleatoria. El modelo aprende a distinguir ambos casos. Es el análogo multimodal del next sentence prediction.
- **Task-Specific Pre-Training.** Antes del fine-tuning, se entrena con los datos de la tarea destino usando el objetivo de masked language modeling con imagen. Esto adapta el modelo al dominio nuevo (p.ej. escenas de películas en VCR, muy distintas de COCO).
- **Fine-Tuning.** Igual que en BERT: se introducen entrada, salida y objetivo específicos de la tarea y se entrena el Transformer para maximizar el rendimiento.

Optimización con Adam (SGD + Adam), warm-up del 10% de los pasos, secuencias de texto recortadas a 128 tokens, batch sizes ajustados al hardware. El pre-entrenamiento sobre COCO toma menos de un día en 4 Tesla V100; todos los experimentos se replican con a lo más 4 V100 de 16 GB.

## 5. Experimentos: cuatro tareas y números reales

Se evalúa sobre cuatro tipos de tareas de visión-y-lenguaje. Para todas, el pre-entrenamiento task-agnostic usa el split Karpathy de COCO (~100k imágenes × 5 captions). Por cada dataset se reportan tres variantes para diagnóstico: el modelo completo (**VisualBERT**), **w/o Early Fusion** (las representaciones de imagen no se combinan con el texto en la primera capa sino al final, en una capa extra) y **w/o COCO Pre-training** (se salta el pre-entrenamiento task-agnostic sobre COCO).

### 5.1. VQA 2.0 (Goyal et al., 2017)

Dada una imagen y una pregunta, responder correctamente. Más de 1 millón de preguntas sobre imágenes de COCO. Se entrena para predecir las 3.129 respuestas más frecuentes, con features de un Faster R-CNN basado en ResNeXt pre-entrenado sobre Visual Genome.

| Modelo | Test-Dev | Test-Std |
|---|---|---|
| Pythia v0.1 (Jiang et al., 2018) | 68.49 | — |
| Pythia v0.3 (Singh et al., 2019) | 68.71 | — |
| VisualBERT w/o Early Fusion | 68.18 | — |
| VisualBERT w/o COCO Pre-training | 70.18 | — |
| **VisualBERT** | **70.80** | **71.00** |

En condiciones comparables (mismas features, mismo número de regiones) VisualBERT supera a Pythia v0.1 y v0.3 siendo significativamente más simple. Métodos con datos extra (VG, múltiples detectores, ensembles) llegan más alto (MCAN + VG + Multiple Detectors + BERT + Ensemble: 75.0), pero no son comparables.

### 5.2. VCR — Visual Commonsense Reasoning (Zellers et al., 2019)

290k preguntas derivadas de 110k escenas de películas, centradas en sentido común visual. Se descompone en dos subtareas multi-opción: question answering (Q→A) y justificación de la respuesta (QA→R). Features de un ResNet50 y cajas/segmentaciones "gold" del dataset; se aprovechan las alineaciones palabra-región (mismos position embeddings para palabras y regiones emparejadas).

| Modelo | Q→A Test | QA→R Test | Q→AR Test |
|---|---|---|---|
| R2C (Zellers et al., 2019) | 65.1 | 67.3 | 44.0 |
| B2T2 (leaderboard, sin publicar) | 72.6 | 75.7 | 55.0 |
| VisualBERT w/o COCO Pre-training (dev) | 67.9 | 69.5 | 47.9 |
| **VisualBERT** | **71.6** | **73.2** | **52.4** |

La variante sin pre-entrenamiento COCO ya supera a R2C (que dispone del mismo recurso) por amplio margen pese a ser mucho más simple. El modelo completo mejora más, demostrando que el pre-entrenamiento sobre COCO ayuda significativamente **incluso con la enorme diferencia de dominio** entre captions de COCO y escenas de películas.

### 5.3. NLVR² — Natural Language for Visual Reasoning (Suhr et al., 2019)

Determinar si una caption en lenguaje natural es verdadera respecto a un **par de imágenes**. Más de 100k ejemplos. Se modifica el mecanismo de segment embeddings para asignar features de imágenes distintas a segments distintos; detector off-the-shelf de Detectron, 144 propuestas por imagen.

| Modelo | Dev | Test-P | Test-U |
|---|---|---|---|
| MaxEnt (Suhr et al., 2019) | 54.1 | 54.8 | 53.5 |
| VisualBERT w/o Early Fusion | 64.6 | — | — |
| VisualBERT w/o COCO Pre-training | 63.5 | — | — |
| **VisualBERT** | **67.4** | **67.0** | **67.3** |

Las dos ablaciones ya superan a MaxEnt por amplio margen y el modelo completo ensancha la brecha. Un experimento preliminar sobre el número de propuestas por imagen muestra mejora monótona: 9→64.8, 18→65.5, 36→66.7, 72→67.1, 144→67.4 (dev).

### 5.4. Flickr30K Entities — Region-to-Phrase Grounding (Plummer et al., 2015)

Dado un span de la oración, seleccionar las regiones que le corresponden. 30k imágenes, ~250k anotaciones. Se sigue el setting de BAN (features de Faster R-CNN sobre Visual Genome); para fine-tuning se añade un bloque de self-attention extra y se usan los pesos promedio de atención de cada cabeza para predecir la alineación caja-frase.

| Modelo | R@1 Test | R@5 Test | R@10 Test |
|---|---|---|---|
| BAN (Kim et al., 2018) | 69.69 | 84.22 | 86.35 |
| **VisualBERT** | **71.33** | **84.98** | **86.51** |

VisualBERT supera al estado del arte BAN. Aquí la diferencia entre el modelo completo y la variante sin early fusion es pequeña, lo que sugiere que para esta tarea una arquitectura más superficial podría bastar.

### 5.5. Ablaciones (sobre NLVR², Tabla 5)

Para acelerar cómputo, todas con 36 features por imagen. Cuatro componentes bajo lupa:

- **C1 — Task-agnostic pre-training:** quitarlo del todo (62.9 dev) o pre-entrenar solo con texto sin imágenes (63.9) degrada respecto al completo (66.7). Pre-entrenar sobre datos pareados imagen-texto es importante.
- **C2 — Early fusion:** sin early fusion cae a 61.4, el peor resultado. La interacción imagen-texto a lo largo de todo el stack es esencial.
- **C3 — Inicialización desde BERT:** sin ella, 64.7. Baja, pero menos de lo esperado: el modelo aprende durante el pre-entrenamiento COCO mucho de lo que BERT aportaría.
- **C4 — Objetivo sentence-image prediction:** sin él, 64.9. Efecto positivo pero menos significativo que C1 y C2.

Conclusión: las decisiones más importantes son **el pre-entrenamiento task-agnostic (C1)** y **el early fusion (C2)**.

### 5.6. Análisis de atención: grounding emergente

Sobre Flickr30K como dataset de diagnóstico, *antes* de fine-tuning sobre ninguna tarea:

- **Entity grounding.** Para cada entidad de la oración y cada cabeza, se mira la región que recibe más atención (enmascarando la atención a palabras). VisualBERT logra una precisión de grounding **notablemente alta sin supervisión directa**, superando a un baseline que siempre elige la región de mayor confianza de detección. La precisión **mejora en las capas altas**: el modelo está menos seguro al sintetizar ambas entradas en capas bajas y se vuelve más certero arriba.
- **Syntactic grounding.** Dadas dos palabras conectadas por una relación de dependencia (parseadas con AllenNLP), existe al menos una cabeza por cada tipo de relación que predice el grounding correcto muy por encima del azar. Muchas cabezas asocian argumentos con verbos (relaciones *pobj*, *nsubj*, *dobj*), lo que sugiere que VisualBERT resuelve esos argumentos a elementos visuales implícita y sin supervisión.
- **Análisis cualitativo.** VisualBERT refina alineaciones a través de las capas. En un ejemplo, "husband" y "woman" atienden inicialmente ambos a la región de la mujer; al final el modelo ha **desenredado** mujer y hombre, alineando ambos correctamente. Hasta resuelve cierta correferencia ("her" → la mujer).

## 6. Limitaciones reconocidas

- **Dependencia de un detector externo.** Las regiones provienen de un detector de objetos pre-entrenado (Faster R-CNN / Detectron / ResNet50 según la tarea). La calidad del grounding queda acotada por qué objetos detecta ese detector; el modelo no aprende a "ver" píxeles crudos, solo a razonar sobre regiones ya propuestas. Además, **cada tarea usa un detector distinto** para poder comparar de igual a igual con el estado del arte previo, lo que impide aislar limpiamente la contribución del modelo de la del detector.
- **Brecha de dominio del pre-entrenamiento.** Se pre-entrena sobre captions de COCO, un dominio de escenas cotidianas. Aunque transfiere sorprendentemente bien incluso a VCR (escenas de películas), el paso de *task-specific pre-training* es necesario precisamente porque esa brecha existe.
- **Escala modesta.** El pre-entrenamiento es sobre ~100k imágenes de COCO. El propio paper apunta como trabajo futuro pre-entrenar sobre datasets mucho mayores (Visual Genome, Conceptual Captions).
- **No reporta cómputo ni ablaciones exhaustivas de escala.** El foco está en simplicidad y rendimiento comparativo, no en barridos de eficiencia.
- **Carácter de "Work in Progress".** Es un preprint preliminar; varias decisiones (number of proposals, features por tarea) se exploran de forma limitada.

## 7. Impacto: VisualBERT como antecedente de los VLM pre-entrenados

VisualBERT, junto con ViLBERT y VideoBERT, marca el momento en que **el paradigma pretraining-then-finetuning de BERT cruza a lo multimodal**. La idea de tratar regiones de imagen como tokens y dejar que un Transformer único las alinee con palabras vía self-attention —y de pre-entrenar ese Transformer con objetivos auto-supervisados visualmente fundamentados— se volvió la plantilla de una generación entera de modelos de visión-y-lenguaje (LXMERT, UNITER, OSCAR, VinVL, y más adelante los VLM modernos que sustituyen el detector por features de patches tipo ViT).

Su aporte conceptual más duradero es doble: (1) la demostración de que **la simplicidad arquitectónica gana** cuando se la combina con buen pre-entrenamiento —no hacía falta un módulo de fusión barroco—; y (2) la evidencia de que **el grounding palabra-región emerge gratis** del masked language modeling multimodal, sin supervisión de alineación. Esto último es lo que lo ata firmemente al relato del aprendizaje auto-supervisado: la señal de supervisión está *en la estructura de los datos pareados imagen-texto*, no en etiquetas humanas, y de esa señal nace una capacidad (anclar lenguaje en visión) que nadie programó explícitamente.

## 8. Conexión con la Clase 28 (Aprendizaje Autosupervisado)

El material de la Clase 28 cita a Li et al. 2019 / VisualBERT en la sección de **multimodalidad**, y la conexión es directa y precisa:

- **El MLM auto-supervisado, de una modalidad a dos.** La clase explica la autosupervisión en lenguaje con BERT: enmascarar tokens y predecirlos a partir del contexto restante, una señal que no requiere etiquetas. VisualBERT toma ese *mismo pretexto* y lo vuelve multimodal: el *masked language modeling con la imagen* predice las palabras enmascaradas usando texto restante **más** regiones de imagen. Es el ejemplo canónico de cómo un objetivo de SSL probado en lenguaje se generaliza a visión-y-lenguaje. Ver [Aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado).

- **Continuidad con BERT.** VisualBERT *es* BERT con embeddings visuales añadidos e inicializado desde los pesos de BERT-base. Para seguir la línea del masked language modeling original, ver [BERT (Devlin et al., 2018)](/papers/bert-devlin-2018); la ablación C3 del paper muestra cuánto (y cuán poco) aporta esa inicialización una vez que entra el pre-entrenamiento multimodal.

- **Grounding emergente como señal de SSL exitoso.** Que las cabezas de atención aprendan a anclar entidades y resolver dependencias sintácticas *sin supervisión de grounding* es exactamente el tipo de "estructura útil que emerge del pretexto" que la clase celebra en los métodos auto-supervisados (igual que las representaciones de RotNet, SimCLR o MAE emergen de pretextos sin etiquetas). Aquí el pretexto es multimodal y lo que emerge es alineación palabra-región.

- **Puente hacia los VLM y el dominio multimodal.** VisualBERT es uno de los hitos que abren la era del pre-entrenamiento multimodal en el [dominio multimodal](/dominios/multimodal), antecediendo a CLIP y a los VLM contrastivos y generativos posteriores.

- **Tarea sustrato.** La evaluación principal incluye VQA, la tarea multimodal por antonomasia; ver [Visual Question Answering](/fundamentos/visual-question-answering) para el contexto de la tarea que VisualBERT aborda con un Transformer único y pre-entrenamiento auto-supervisado.

Enlaces de la clase: [Clase 28 — Aprendizaje Autosupervisado](/clases/clase-28) · [BERT (Devlin et al., 2018)](/papers/bert-devlin-2018) · [Aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado) · [Dominio multimodal](/dominios/multimodal) · [Visual Question Answering](/fundamentos/visual-question-answering).
