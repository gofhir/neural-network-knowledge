---
título: "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation"
autores: "Junnan Li, Dongxu Li, Caiming Xiong, Steven Hoi"
afiliación: "Salesforce Research"
venue: "ICML 2022 (Proceedings of the 39th International Conference on Machine Learning)"
año: 2022
arxiv: "2201.12086"
arxiv_version: "v2 (15 Feb 2022)"
link: "https://arxiv.org/abs/2201.12086"
código: "https://github.com/salesforce/BLIP"
clase: "Clase 23 — Visual Question Answering e Image Captioning"
rol_en_clase: "Modelo del Laboratorio 23 — VQA y captioning con blip-vqa-base y blip-image-captioning-base de HuggingFace"
---

# BLIP: Bootstrapping Language-Image Pre-training

> "Most existing pre-trained models only excel in either understanding-based tasks or generation-based tasks. Furthermore, performance improvement has been largely achieved by scaling up the dataset with noisy image-text pairs collected from the web, which is a suboptimal source of supervision."
> — Resumen del paper. Las dos frases condensan exactamente los dos problemas que BLIP ataca: la dicotomía *encoder vs decoder* (modelo) y el ruido de los datos web (datos).

## 1. Ficha bibliográfica y resumen ejecutivo

- **Título:** *BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation*.
- **Autores:** Junnan Li, Dongxu Li, Caiming Xiong, Steven Hoi.
- **Afiliación:** Salesforce Research.
- **Venue:** ICML 2022. arXiv:2201.12086, versión v2 (15 de febrero de 2022).
- **Código, modelos y datasets:** `https://github.com/salesforce/BLIP` (liberados, incluido el dataset bootstrapeado).

BLIP es un **framework de pre-entrenamiento visión-lenguaje (VLP)** que persigue un objetivo ambicioso: que un solo modelo transfiera bien tanto a tareas de **comprensión** (image-text retrieval, VQA, razonamiento visual) como a tareas de **generación** (image captioning). El paper hace dos contribuciones complementarias, una desde la perspectiva del **modelo** y otra desde la perspectiva de los **datos**:

1. **MED (Multimodal mixture of Encoder-Decoder):** una arquitectura única que, según se la configure, opera como *unimodal encoder*, como *image-grounded text encoder*, o como *image-grounded text decoder*. Se pre-entrena conjuntamente con tres objetivos —contraste imagen-texto (ITC), emparejamiento imagen-texto (ITM) y modelado de lenguaje (LM)— que comparten parámetros de forma inteligente.
2. **CapFilt (Captioning and Filtering):** un método de *bootstrapping* de datos. A partir del MED pre-entrenado se derivan dos módulos —un **Captioner** que genera descripciones sintéticas para imágenes web, y un **Filter** que descarta las descripciones ruidosas (tanto las web originales como las sintéticas)—. El dataset depurado y enriquecido se usa para re-entrenar el modelo desde cero.

Los resultados son estado del arte en un abanico amplio de tareas: **+2.7% en average recall@1** en image-text retrieval, **+2.8% en CIDEr** en image captioning, y **+1.6% en VQA score**, además de un desempeño **zero-shot** sorprendentemente fuerte al transferir directamente a tareas de video-lenguaje. La importancia pedagógica para la Clase 23 es directa: el Laboratorio 23 usa los checkpoints `blip-vqa-base` y `blip-image-captioning-base` de HuggingFace, de modo que BLIP es el modelo "vivo" con el que se interactúa en código, en contraste con Pythia, que es el modelo "clásico" de la parte teórica.

## 2. Contexto histórico — qué había antes y qué fallaba

A inicios de 2022, el pre-entrenamiento visión-lenguaje vivía un boom, pero arrastraba dos limitaciones que el paper enuncia con precisión quirúrgica.

### 2.1. La limitación de modelo: encoder-only vs encoder-decoder

Los métodos existentes caían en uno de dos campos arquitectónicos, y ninguno servía para todo:

- **Modelos basados en encoder.** **CLIP** (Radford et al., 2021) y **ALBEF** (Li et al., 2021a) aprenden representaciones alineando imagen y texto con objetivos contrastivos o de matching. Son excelentes para **comprensión** (retrieval, clasificación), pero **no se transfieren de forma directa a generación de texto**: un encoder no produce secuencias autoregresivas, así que hacer image captioning con CLIP exige acoplarle un decoder ajeno. El paper lo dice literalmente: "encoder-based models are less straightforward to directly transfer to text generation tasks".
- **Modelos encoder-decoder.** **SimVLM** (Wang et al., 2021) y el framework unificado de **Cho et al. (2021, VL-T5)** sí generan texto, pero "have not been successfully adopted for image-text retrieval tasks". Un encoder-decoder con cross-attention densa no produce de forma natural los *embeddings* comparables que el retrieval necesita.

Intentos previos de unificar todo en un solo encoder-decoder (Zhou et al., 2020, VLP) también "limitan la capacidad del modelo". El diagnóstico de BLIP: ni los encoder puros ni los encoder-decoder puros sobresalen en ambos tipos de tarea simultáneamente. La solución no es elegir uno, sino **una mezcla flexible** que pueda vestirse de cualquiera de los tres según la necesidad — de ahí "mixture of encoder-decoder".

Vale recordar el linaje. **VirTex** (Desai & Johnson, 2021), aunque no se cita explícitamente, había mostrado que generar captions densos es una señal de pre-entrenamiento eficiente en datos. **CLIP** popularizó el contraste imagen-texto a escala de 400M pares. **ALBEF** —el predecesor directo de BLIP, del mismo grupo— introdujo "align before fuse": primero alinear con ITC, luego fusionar con cross-attention, más la *momentum distillation* para lidiar con ruido. BLIP hereda mucho de ALBEF (el ViT como encoder de imagen, la ITC con encoder de momento, el *hard negative mining* para ITM) y añade el componente generativo y CapFilt.

### 2.2. La limitación de datos: la web es ruidosa

La segunda observación es sobre los **datos**. El estado del arte (CLIP, ALBEF, SimVLM, ALIGN) se entrenaba con pares imagen–*alt-text* rastreados de la web (Conceptual Captions, Conceptual 12M, SBU, LAION). El problema: los *alt-texts* "a menudo no describen con precisión el contenido visual de las imágenes". Son una señal ruidosa. La comunidad había **ignorado largamente el impacto negativo del ruido**, porque escalar el dataset compensaba en agregado. BLIP demuestra que ese ruido es subóptimo y que se puede aprovechar mejor la web "bootstrapeando" los captions: generar texto sintético limpio y filtrar el ruido. La Figura 1 del paper lo ilustra con un *alt-text* genérico ("blue sky bakery in sunset park") que el Filter rechaza frente a un caption sintético descriptivo que se acepta.

## 3. Contribución (a): el MED en detalle

### 3.1. El image encoder: un ViT, no un detector

BLIP abandona el detector de objetos de la era up-down/Pythia. El encoder de imagen es un **Vision Transformer (ViT)** (Dosovitskiy et al., 2021) que divide la imagen en parches, los codifica como una secuencia de *embeddings* y antepone un token **[CLS]** que resume la característica global de la imagen. El paper justifica la elección: usar un ViT "es más amigable computacionalmente y ha sido adoptado por los métodos más recientes", a diferencia de extraer características con detectores pre-entrenados (como OSCAR/UNITER), que son lentos y requieren cajas anotadas. Se exploran dos variantes: **ViT-B/16** y **ViT-L/16**; salvo que se diga lo contrario, "BLIP" se refiere a ViT-B. El ViT se inicializa desde pesos pre-entrenados en ImageNet (Touvron et al., 2020).

### 3.2. Los tres modos del MED

El texto se procesa con un transformer inicializado desde **BERTbase** (Devlin et al., 2019). La clave es que el mismo bloque de texto puede operar en tres modos (Figura 2 del paper), y cada modo activa un objetivo distinto:

**(1) Unimodal encoder — objetivo ITC.**
Codifica imagen y texto **por separado**, sin interacción cruzada. El texto recibe un token **[CLS]** al inicio que resume la oración (igual que BERT). Este modo se entrena con la **pérdida contrastiva imagen-texto (ITC)**, que alinea el espacio de características del ViT con el del encoder de texto: empuja a los pares positivos a tener representaciones similares y a los negativos a estar lejos. BLIP sigue la formulación de ITC de ALBEF, con un **encoder de momento** que produce características estables y **etiquetas blandas** (*soft labels*) que reconocen que algunos "negativos" en el batch podrían ser positivos potenciales.

**(2) Image-grounded text encoder — objetivo ITM.**
Inyecta información visual insertando **una capa de cross-attention (CA) adicional** entre la self-attention (SA) y la feed-forward network (FFN) de cada bloque del transformer de texto. Aquí la **self-attention es bidireccional** (cada token ve a todos los demás, como BERT). Se antepone un token específico **[Encode]**, y su *embedding* de salida sirve como la **representación multimodal** del par imagen-texto. Este modo se entrena con la **pérdida de emparejamiento imagen-texto (ITM)**, una clasificación binaria (una cabeza lineal predice *matched/unmatched*) que captura la alineación fina entre visión y lenguaje. Para hacerla informativa se usa **hard negative mining** (de ALBEF): los negativos con mayor similitud contrastiva dentro del batch tienen más probabilidad de elegirse, forzando al modelo a distinguir casos difíciles.

**(3) Image-grounded text decoder — objetivo LM.**
Reemplaza las capas de **self-attention bidireccional** del modo (2) por capas de **self-attention causal** (cada token solo ve los anteriores, como GPT). Usa un token **[Decode]** para señalar el inicio de la secuencia y un token de fin de secuencia para el final. Se entrena con la **pérdida de modelado de lenguaje (LM)**: cross-entropy autoregresiva que maximiza la verosimilitud del texto token a token, con **label smoothing de 0.1**. A diferencia de la pérdida MLM (masked language modeling) usada por la mayoría de VLP, la LM "dota al modelo de la capacidad de generalización para convertir información visual en captions coherentes" — es decir, le da la capacidad generativa que los modelos tipo BERT no tienen.

### 3.3. El truco del compartir parámetros: todo excepto la SA

Aquí está la elegancia del MED. El encoder de texto y el decoder de texto **comparten todos los parámetros excepto las capas de self-attention (SA)**. El razonamiento del paper: "las diferencias entre las tareas de codificación y decodificación se capturan mejor en las capas SA". El encoder necesita SA **bidireccional** (construir representaciones del input completo); el decoder necesita SA **causal** (predecir el siguiente token sin ver el futuro). Esas dos funciones son incompatibles en una misma capa SA. En cambio, las **capas de embedding, las capas de cross-attention (CA) y las FFN funcionan de forma similar** en ambas tareas, así que compartirlas mejora la eficiencia de entrenamiento mientras se beneficia del *multi-task learning*.

La Tabla 3 del paper valida esto empíricamente comparando estrategias de compartición durante el pre-entrenamiento:

| Capas compartidas | #parámetros | CIDEr (NoCaps ZS) |
|---|---|---|
| Todas | 224M | 100.9 |
| Todas excepto CA | 252M | 101.2 |
| **Todas excepto SA** | **252M** | **102.2** |
| Ninguna | 361M | 101.8 |

Compartir todo excepto SA gana, y además ahorra parámetros (252M vs 361M de no compartir nada). Si se comparten también las SA, el desempeño **se degrada** "por el conflicto entre la tarea de codificación y la de decodificación". Es decir: la SA es exactamente la capa donde encoder y decoder deben diferir, y nada más.

### 3.4. El coste computacional del pre-entrenamiento

Cada par imagen-texto requiere **un solo forward pass por el ViT** (el componente pesado) y **tres forward passes por el transformer de texto** (uno por cada objetivo: ITC, ITM, LM). Esto hace el pre-entrenamiento multitarea relativamente barato: el cuello de botella visual se computa una vez. El optimizador es **AdamW** con weight decay 0.05; el learning rate hace *warmup* hasta 3e-4 (ViT-B) / 2e-4 (ViT-L) y decae linealmente. Se pre-entrena por **20 épocas** con batch size 2880 (ViT-B) / 2400 (ViT-L) en dos nodos de 16 GPUs. La resolución es 224×224 en pre-entrenamiento y sube a 384×384 en fine-tuning.

## 4. Contribución (b): CapFilt, el bootstrapping de datos

CapFilt es la otra mitad —y, en muchos sentidos, la idea más memorable del paper. Resuelve el problema del ruido web sin renunciar a la escala de la web.

### 4.1. El punto de partida: datos limpios escasos, datos web abundantes

Existe un número **limitado** de pares imagen-texto de alta calidad anotados por humanos, $\{(I_h, T_h)\}$ — principalmente **COCO**. Y existe una cantidad **mucho mayor** de pares imagen–alt-text web $\{(I_w, T_w)\}$, abundantes pero ruidosos. La idea de CapFilt: usar los datos limpios para entrenar dos herramientas que limpien y enriquezcan los datos web.

### 4.2. Inicialización: dos módulos desde el mismo MED

Tanto el **Captioner** como el **Filter** se inicializan desde el **mismo MED pre-entrenado** y se hacen *fine-tune individualmente* sobre COCO (un procedimiento ligero):

- **El Captioner es un image-grounded text decoder.** Se hace fine-tune con el objetivo **LM** para decodificar texto dado una imagen. Aplicado a las imágenes web $I_w$, genera **captions sintéticos $T_s$** (uno por imagen).
- **El Filter es un image-grounded text encoder.** Se hace fine-tune con los objetivos **ITC e ITM** para aprender si un texto coincide con una imagen. El Filter recorre **tanto los textos web originales $T_w$ como los sintéticos $T_s$**, y marca un texto como ruidoso si la cabeza ITM lo predice como *unmatched*.

### 4.3. El dataset bootstrapeado

El resultado se combina así: se toman los pares web que sobreviven al filtro, más los captions sintéticos que sobreviven al filtro, más los pares humanos COCO. Formalmente, el nuevo dataset $D$ es:

$$D = \{(I_w, T_w)\}_{\text{filtrados}} + \{(I_w, T_s)\}_{\text{filtrados}} + \{(I_h, T_h)\}.$$

Con este dataset **se pre-entrena un modelo nuevo desde cero**. La Figura 4 del paper muestra ejemplos cualitativos: para una foto, el alt-text web "from bridge near my house" es rechazado (rojo) mientras el caption sintético "a flock of birds flying over a lake at sunset" es aceptado (verde) — el sintético es más descriptivo que el original.

### 4.4. Nucleus sampling vs beam search: la diversidad es la clave

Un hallazgo no obvio: **cómo** se generan los captions sintéticos importa mucho. El paper compara dos estrategias de decodificación (Tabla 2):

| Generación | Noise ratio | TR@1 (COCO) | CIDEr (NoCaps ZS) |
|---|---|---|---|
| Ninguna (solo web) | N.A. | 78.4 | 102.2 |
| Beam search | 19% | 79.6 | 103.5 |
| **Nucleus sampling** | **25%** | **80.6** | **105.1** |

El **nucleus sampling** (muestreo del núcleo, Holtzman et al., 2020) es estocástico: cada token se muestrea del conjunto mínimo de tokens cuya masa de probabilidad acumulada supera un umbral $p = 0.9$. El **beam search** es determinista y busca la secuencia de máxima probabilidad. Contraintuitivamente, **nucleus sampling gana**, a pesar de producir captions más ruidosos (25% de tasa de rechazo del filtro vs 19% del beam search). La hipótesis del paper: nucleus sampling genera "captions más diversos y sorprendentes, que contienen más información nueva de la que el modelo puede beneficiarse", mientras que beam search tiende a generar "captions seguros y comunes en el dataset", aportando menos conocimiento extra. Esta es la observación destilada del paper: **"more diverse captions yield larger gains"**.

### 4.5. Captioner y Filter deben estar desacoplados

La Tabla 4 muestra otro detalle fino: si el Captioner y el Filter **comparten parámetros** (como en pre-entrenamiento), el desempeño cae. La razón es **sesgo de confirmación** (*confirmation bias*): un Filter que comparte pesos con el Captioner es menos propenso a rechazar los captions ruidosos que el propio Captioner produjo (la tasa de ruido baja a 8% vs 25% cuando están desacoplados). Por eso ambos módulos se hacen fine-tune **por separado**, end-to-end, sobre COCO.

### 4.6. CapFilt como destilación de conocimiento

El paper enmarca CapFilt como una forma de **knowledge distillation (KD)** específica para VLP. El Captioner "destila su conocimiento a través de captions sintéticos semánticamente ricos" y el Filter "destila su conocimiento removiendo captions ruidosos". Esto explica un detalle de ingeniería (Tabla 13): **continuar entrenando** el modelo pre-entrenado sobre el dataset bootstrapeado ayuda **menos** que entrenar un modelo nuevo desde cero. Concuerda con la práctica común en KD: "el modelo estudiante no puede inicializarse desde el profesor". Además, la Tabla 12 verifica que la ganancia de CapFilt **no se debe a entrenar más tiempo** — replicar los textos web ruidosos para igualar el número de muestras no mejora nada.

## 5. Experimentos clave

### 5.1. Datasets de pre-entrenamiento

Dos configuraciones principales:
- **14M imágenes:** COCO + Visual Genome (anotadas por humanos) + Conceptual Captions + Conceptual 12M + SBU captions (web). Misma fuente que ALBEF.
- **129M imágenes:** las anteriores + **LAION** (115M, mucho más ruidoso; por su tamaño se usa 1/5 por época).

### 5.2. El efecto de CapFilt (Tabla 1)

La tabla pivote del paper. Sobre 14M imágenes con ViT-B/16, aplicar Captioner (C) y Filter (F) por separado ya mejora; aplicados **juntos** se complementan. Algunos números (COCO retrieval-FT TR@1 / NoCaps CIDEr ZS):

| Bootstrap | Backbone | TR@1 | CIDEr (NoCaps) |
|---|---|---|---|
| Ninguno (web crudo) | ViT-B | 78.4 | 102.2 |
| Solo Filter | ViT-B | 79.1 | 102.7 |
| Solo Captioner | ViT-B | 79.7 | 103.4 |
| **C + F** | ViT-B | **80.6** | **105.1** |

Y CapFilt **escala**: con 129M imágenes y un Captioner/Filter basados en ViT-L, el modelo base llega a TR@1 = 81.2 y NoCaps CIDEr = 109.6, confirmando que escala en tamaño de datos y de modelo.

### 5.3. Image-text retrieval (COCO, Flickr30K)

Con fine-tuning vía ITC + ITM (y un reranking eficiente: seleccionar k candidatos por similitud de features y reordenar por score ITM, k=256 COCO / k=128 Flickr), BLIP es estado del arte (Tabla 5). Usando las mismas 14M imágenes, BLIP **supera a ALBEF en +2.7% en average recall@1 en COCO**. En COCO 5K test, BLIP-14M alcanza TR@1 = 80.6 / IR@1 = 63.1; con 129M sube a 81.9 / 64.3; con ViT-L llega a 82.4 / 65.1. En **zero-shot** (Tabla 6), transfiriendo el modelo afinado en COCO directo a Flickr30K, BLIP también supera por amplio margen a CLIP y ALIGN — notable porque CLIP usó 400M pares y ALIGN 1.8B, frente a los 14M-129M de BLIP.

### 5.4. Image captioning (COCO, NoCaps)

Se añade un prompt "a picture of" al inicio de cada caption (mejora ligera, idea de SimVLM). En la Tabla 7, BLIP-14M ya supera sustancialmente a métodos con datos comparables; BLIP-129M con ViT-L alcanza **COCO Karpathy test BLEU@4 = 40.4 / CIDEr = 136.7** y **NoCaps overall CIDEr = 113.2 / SPICE = 14.8**, competitivo con LEMON (200M imágenes, detector pesado) y SimVLM (1.8B imágenes), pero con un modelo **detector-free** y de menor resolución de entrada (384×384 vs 800×1333), lo que lo hace mucho más rápido en inferencia.

### 5.5. VQA (VQAv2)

Crucial para la Clase 23. En lugar de formular VQA como **clasificación multi-respuesta** (el enfoque de Pythia/UNITER), BLIP la trata como **generación de respuestas** (open-ended VQA), siguiendo a ALBEF. Durante fine-tuning (Figura 5a) se reorganiza el modelo: imagen+pregunta se codifican en *embeddings* multimodales que se pasan a un **answer decoder**, entrenado con pérdida LM usando las respuestas ground-truth como objetivo. Resultados (Tabla 8): con 14M imágenes BLIP supera a ALBEF en **+1.64%** en el test set; con 129M imágenes alcanza **test-dev = 78.24 / test-std = 78.17**, superando a SimVLM que usó **13× más datos** y un backbone visual mayor.

### 5.6. NLVR2, VisDial y zero-shot a video

- **NLVR2** (Suhr et al., 2019): predecir si una oración describe un par de imágenes. BLIP modifica cada bloque del encoder para tener **dos capas de cross-attention** (una por imagen), cuyas salidas se fusionan (average pooling en las primeras 6 capas, concatenación + proyección lineal en las capas 6-12). Alcanza test-P = 83.08 (129M). Curiosamente, NLVR2 **no se beneficia mucho** de más imágenes web, probablemente por el *domain gap* entre datos web y datos downstream.
- **VisDial** (Tabla 9): estado del arte en VisDial v1.0 validation (MRR 69.41).
- **Zero-shot a video** (Tablas 10-11): transfiriendo directamente los modelos entrenados en COCO-retrieval y VQA, muestreando n frames por video (n=8 retrieval, n=16 QA) y concatenando sus features, **ignorando toda información temporal**. Aun así, en text-to-video retrieval sobre MSRVTT, **BLIP zero-shot (R@1 = 43.3) supera incluso a modelos afinados sobre el dataset de video objetivo en +12.4% en recall@1**. Es la demostración más fuerte de la generalización del modelo.

## 6. Limitaciones reconocidas

El paper es explícito sobre los frentes abiertos en su Conclusión, proponiendo direcciones que son a la vez limitaciones del trabajo presentado:

- **Una sola ronda de bootstrapping.** CapFilt se aplica una vez; el paper sugiere **múltiples rondas** como mejora potencial.
- **Un solo caption sintético por imagen.** Se podría **generar múltiples captions sintéticos** por imagen para agrandar el corpus.
- **Sin ensemble de Captioners/Filters.** Entrenar varios Captioners y Filters distintos y combinarlos en CapFilt.
- **Sin modelado temporal en video.** La transferencia zero-shot a video ignora por completo el tiempo; el paper anota que reemplazar el ViT por un TimeSformer y afinar sobre video daría mejoras.
- **Dependencia de COCO para el fine-tuning de CapFilt.** Captioner y Filter se calibran sobre datos anotados por humanos; la calidad del filtrado hereda el sesgo de dominio de COCO (visible en el *domain gap* de NLVR2).

A esto se suma una limitación implícita relevante para el laboratorio: como modelo **generativo**, BLIP puede **alucinar** respuestas plausibles pero incorrectas cuando enfrenta conceptos fuera de su distribución de entrenamiento (ver Sección 8).

## 7. Impacto y legado

BLIP fue extraordinariamente influyente y dio origen a una familia:

- **BLIP-2** (Li et al., 2023): introduce el **Q-Former**, un transformer ligero que actúa de puente entre un encoder de imagen **congelado** y un **LLM congelado** (OPT, FlanT5). Reduce drásticamente el coste de entrenamiento al no afinar los modelos grandes, y conecta la visión con LLMs potentes.
- **InstructBLIP** (Dai et al., 2023): añade *instruction tuning* sobre BLIP-2, condicionando el Q-Former a la instrucción para mejorar el seguimiento de instrucciones en tareas visión-lenguaje zero-shot.
- **Influencia en la era multimodal.** El patrón "encoder de imagen congelado + puente entrenable + LLM" que BLIP-2 popularizó es ancestral de **LLaVA**, **MiniGPT-4** y los VLMs instruccionales modernos. La idea de CapFilt —usar el propio modelo para limpiar y enriquecer sus datos de entrenamiento— anticipa las modernas pipelines de **re-captioning** sintético (p. ej. los captions densos de DALL·E 3 y de los datasets de entrenamiento de muchos VLMs actuales).
- **MED como plantilla.** La idea de un transformer multimodal que conmuta entre encoder bidireccional y decoder causal compartiendo todo menos la SA reaparece en numerosos diseños posteriores de modelos unificados de comprensión-generación.

## 8. Conexión con el Laboratorio 23

El Laboratorio 23 (`Lab23_VQA_ImageCaptioning_v3.ipynb`) **usa BLIP en código**, lo que lo convierte en la contraparte práctica de la teoría de Pythia. Dos checkpoints de HuggingFace:

- **`Salesforce/blip-vqa-base`** vía `BlipForQuestionAnswering`: para VQA. El lab carga el modelo y el `BlipProcessor`, arma `inputs = processor(image, question, return_tensors="pt")` y llama **`model.generate(**inputs)`**, decodificando la salida con `processor.decode(...)`.
- **`Salesforce/blip-image-captioning-base`** vía `BlipForConditionalGeneration`: para captioning, también con `model.generate(**inputs, max_length=20)`.

**El contraste pedagógico clave: generación vs clasificación.** La parte teórica de la Clase 23 explica VQA con **Pythia**, que es **clasificación multi-etiqueta sobre un vocabulario cerrado** (~3129 respuestas, clasificador sigmoide). BLIP, en cambio, hace VQA como **generación** — `model.generate` produce la respuesta token a token con el image-grounded text decoder (Sección 5.5). Esta es exactamente la diferencia que el laboratorio pone frente a los ojos del estudiante: en Pythia la respuesta sale de un `argmax` sobre clases; en BLIP la respuesta se **genera** como texto libre. La virtud es la apertura (puede responder cosas fuera de un vocabulario fijo); el riesgo es la **alucinación**.

**El error del ornitorrinco.** El lab incluye deliberadamente una imagen de un ornitorrino y pregunta `"What kind of animal is this?"`. Como el ornitorrinco es un animal raro y casi seguramente subrepresentado en los datos de pre-entrenamiento de BLIP (COCO, Visual Genome y captions web), el modelo **no tiene el concepto bien aprendido** y genera una respuesta plausible pero incorrecta — clasifica el animal como algo más común (pato, castor, etc.). Esto ilustra dos puntos del paper a la vez:

1. **La generación abierta alucina.** A diferencia de un clasificador de vocabulario cerrado que simplemente erraría dentro de su lista, el decoder generativo de BLIP produce con seguridad una etiqueta equivocada. Es la cara oscura de la flexibilidad generativa.
2. **El ruido y los huecos de cobertura de los datos web importan.** Justamente el problema que CapFilt intenta mitigar — pero CapFilt mejora la *calidad* de los captions, no agrega *conceptos nuevos* que la web no contiene. Un animal de cola larga (long-tail) como el ornitorrinco cae en el hueco de cobertura del dataset, y ni el Captioner ni el Filter pueden inventar conocimiento que el modelo nunca vio. NoCaps, el benchmark de la Sección 5.4, existe precisamente para medir *novel object captioning* (describir objetos no vistos en COCO), y los números de out-of-domain de la Tabla 7 muestran que incluso BLIP sufre fuera de dominio.

El lab también prueba preguntas espaciales ("Is the dog in front of the chair?") y de atributos ("What color is the cat?"), las mismas categorías donde la Clase 23 mostraba fallar a Pythia. Comparar las respuestas de BLIP con las limitaciones de Pythia cierra el arco de la clase: del VQA-clasificación clásico (atención bottom-up + suma ponderada + softmax) al VQA-generación moderno (ViT + cross-attention + decoder autoregresivo), con sus nuevas fortalezas y sus nuevos modos de fallo.

## 9. Notas y enlaces

- **Paper:** arXiv:2201.12086 — `https://arxiv.org/abs/2201.12086`
- **Código, modelos y dataset bootstrapeado:** `https://github.com/salesforce/BLIP`
- **Checkpoints del lab:** `Salesforce/blip-vqa-base`, `Salesforce/blip-image-captioning-base` (HuggingFace).
- **Predecesor imprescindible:** Li et al., *Align before Fuse (ALBEF)*, NeurIPS 2021 — BLIP hereda su ITC con encoder de momento, el hard negative mining de ITM, y el VQA como generación.
- **Referencias clave citadas:**
  - Radford et al., *CLIP*, 2021 — contraste imagen-texto a escala, encoder puro.
  - Wang et al., *SimVLM*, 2021 — encoder-decoder con supervisión débil (PrefixLM).
  - Dosovitskiy et al., *ViT*, ICLR 2021 — el encoder de imagen.
  - Devlin et al., *BERT*, NAACL 2019 — inicialización del transformer de texto.
  - Holtzman et al., *The Curious Case of Neural Text Degeneration (Nucleus Sampling)*, ICLR 2020 — la decodificación de los captions sintéticos.
  - Hinton et al., *Distilling the Knowledge in a Neural Network*, 2015 — el marco de KD que explica por qué CapFilt entrena un modelo nuevo.
  - Loshchilov & Hutter, *AdamW*, 2017 — el optimizador.
  - Agrawal et al., *NoCaps*, ICCV 2019 — el benchmark de captioning de objetos novedosos.
