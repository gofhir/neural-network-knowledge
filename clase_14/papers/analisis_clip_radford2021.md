# Analisis del Paper: Learning Transferable Visual Models From Natural Language Supervision (CLIP)

**Autores**: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever
**Institucion**: OpenAI, San Francisco
**Publicado en**: ICML 2021 (preprint arXiv:2103.00020, 26 Feb 2021)
**Codigo**: [github.com/OpenAI/CLIP](https://github.com/OpenAI/CLIP)

> PDF descargado en: [clip-radford-2021.pdf](clip-radford-2021.pdf) (48 paginas)

---

## Datos Clave del Paper

| Aspecto | Detalle |
|---|---|
| **Ano** | 2021 |
| **Citas** | >35.000 (uno de los papers mas citados de la decada) |
| **Idea central** | Pre-entrenar conjuntamente un encoder de imagen y uno de texto con un objetivo contrastivo simetrico sobre 400M pares (imagen, texto) extraidos de la web, obteniendo representaciones que transfieren *zero-shot* a docenas de tareas de vision via prompts en lenguaje natural |
| **Innovacion clave** | Escalar contrastive learning imagen-texto de ~100K pares (ConVIRT, VirTex) a 400M, demostrando que el lenguaje natural es una senal de supervision suficiente y mas general que las etiquetas curadas |
| **Resultado emblema** | ZS ImageNet 76.2% (ViT-L/14@336px) -- iguala ResNet-50 supervisado sin ver una sola etiqueta de ImageNet |
| **Impacto** | Base de Stable Diffusion (text encoder), DALL-E 2, BLIP, LLaVA. Inicio de la era de modelos visuales abiertos guiados por lenguaje |

---

## 1. Resumen ejecutivo

CLIP demuestra que la **supervision con lenguaje natural a escala web** es competitiva con la supervision curada para aprender representaciones visuales generales. En vez de entrenar un clasificador para 1000 clases predefinidas (ImageNet), el modelo aprende a *predecir cual texto acompana cual imagen* dentro de un batch. Esa simple tarea de matching, escalada a 400M pares y con encoders modernos (ViT, Transformer), produce features que:

1. **Transfieren zero-shot** a docenas de datasets via prompts ("A photo of a {class}").
2. **Resisten distribution shifts** mucho mejor que modelos entrenados solo en ImageNet.
3. **Compiten o superan** linear probes de los mejores modelos supervisados/self-supervisados (BiT-M, SimCLRv2, EfficientNet-NS) cuando se evalua sobre 27 datasets diversos.

El paper combina (a) una receta tecnica concreta, (b) un benchmark exhaustivo sobre 30+ datasets, y (c) una discusion honesta de limitaciones y sesgos sociales.

---

## 2. Aporte central: unificar vision y lenguaje via contrastive a escala

### 2.1. Por que contrastive y no captioning

Los autores comenzaron probando un enfoque *generativo* tipo VirTex: un CNN + Transformer prediciendo el caption exacto. Encontraron que era **3x mas lento** que un baseline de bag-of-words (que solo predice si las palabras estan o no en el texto), y este a su vez **4x mas lento** que un objetivo *contrastivo* (que solo predice cual texto va con cual imagen). En total, **CLIP es 12x mas eficiente** que el baseline generativo en alcanzar la misma zero-shot ImageNet accuracy (Figura 2 del paper).

```text
Eficiencia de zero-shot ImageNet vs imagenes procesadas:

  Bag-of-words contrastivo (CLIP)     ━━━━━━━━━ 4x sobre BoW Pred
  Bag-of-words prediction              ━━━━━━ 3x sobre Transformer LM
  Transformer language model           ━━ baseline lento

INSIGHT: el caption *exacto* es muy informativo pero muy dificil
de predecir (alta entropia condicional). Predecir solo la
*correspondencia* relaja el problema y acelera el aprendizaje.
```

### 2.2. Posicionamiento frente a literatura previa

| Trabajo | Pares | Aporte distintivo |
|---|---|---|
| Mori 1999, Quattoni 2007 | ~10K | Primeros usos de NL para retrieval/representacion |
| Joulin 2016, Li 2017 | YFCC100M (~15M filtrado) | Bag-of-words sobre titulos/tags |
| Visual N-Grams (Li 2017) | ~30M | Primer zero-shot a ImageNet (11.5%) |
| VirTex (Desai 2020) | COCO ~100K | Caption generativo para representacion |
| ICMLM (Sariyildiz 2020) | COCO | Masked LM con contexto visual |
| ConVIRT (Zhang 2020) | medical pairs | Contrastive imagen-texto en medicina |
| **CLIP (este paper)** | **400M** | **Escala 400x ConVIRT, mejor receta, ViT, evaluacion masiva** |

Mahajan 2018 (Instagram tags, ~3.5B imagenes) y Kolesnikov 2019 (JFT-300M) usaron mas data pero con etiquetas softmax fijas (1K-18K clases), no lenguaje libre, lo que limita zero-shot.

---

## 3. Datos: el dataset WIT (WebImageText)

### 3.1. Construccion

```text
PIPELINE DE CURACION DE WIT:
─────────────────────────────────────────────────────

1. QUERY LIST (500.000 queries):
   - Todas las palabras con frecuencia >=100 en
     Wikipedia ingles
   - Bigramas con alto pointwise mutual information
   - Nombres de articulos de Wikipedia con volumen
     de busqueda alto
   - WordNet synsets faltantes

2. SCRAPING:
   - Buscar pares (imagen, texto) en fuentes
     publicamente disponibles donde el texto contenga
     una de las queries
   - El "texto" es el alt-text, caption o descripcion
     que acompana la imagen

3. BALANCEO:
   - Limitar a 20.000 pares por query
   - Total: ~400 millones de pares (image, text)
   - Aprox mismo word count que WebText (GPT-2)
```

### 3.2. Comparacion con datasets previos

| Dataset | Tamano | Etiquetas | Apertura |
|---|---|---|---|
| MS-COCO | ~100K | captions humanos | abierto |
| Visual Genome | ~100K | captions humanos | abierto |
| YFCC100M filtrado (English titles) | ~15M | metadata variable | abierto |
| ImageNet-1K | 1.28M | 1000 clases | abierto |
| JFT-300M | 300M | 18K clases (softmax) | cerrado (Google) |
| Instagram (Mahajan) | 3.5B | hashtags | cerrado (FB) |
| **WIT** | **400M** | **texto libre** | **cerrado (OpenAI)** |

Punto clave: WIT no esta publico. Esto motivo trabajos posteriores como **LAION-400M/2B** y **MetaCLIP** que documentan y abren recetas de curacion analogas.

---

## 4. Arquitectura: dual-encoder con proyecciones lineales

### 4.1. Image encoder (5 ResNets + 3 ViTs evaluados)

**Variantes ResNet** (modificaciones sobre He 2016):
- ResNet-D improvements (He 2019) -- stem, downsampling
- Antialiased rect-2 blur pooling (Zhang 2019)
- Global average pool reemplazado por **attention pooling**: una capa de QKV multi-head donde el query es el GAP de la imagen.

Variantes: **RN50**, **RN101**, y EfficientNet-style scaled: **RN50x4**, **RN50x16**, **RN50x64** (4x, 16x, 64x el compute de RN50).

**Variantes ViT**: ViT-B/32, ViT-B/16, ViT-L/14, todas con minor mod (LayerNorm extra antes del Transformer en patch+pos embeddings). Adicionalmente entrenan ViT-L/14 a 336px por 1 epoca extra (analogo a FixRes), denominado **ViT-L/14@336px** -- el modelo *flagship*.

### 4.2. Text encoder

```text
ARQUITECTURA TEXTO:
─────────────────────────────────────────────────────
- Transformer (Vaswani 2017) con mods de Radford 2019 (GPT-2)
- 12 capas, 8 heads, ancho 512
- ~63M parametros
- Tokenizacion: lower-cased BPE, vocabulario 49.152
- Secuencia maxima: 76 tokens
- Tokens especiales: [SOS] ... [EOS]
- Self-attention enmascarada (causal, decoder-style)
  → preserva opcion de language modeling auxiliar
- Feature representation: activacion del [EOS] en la
  ultima capa, layer-normalizada, proyectada
  linealmente al espacio multimodal
```

Los autores escalaron *width* de las ResNets en proporcion al compute, pero *no* escalaron el text encoder con los modelos mas grandes ("CLIP performance is less sensitive to capacity of text encoder"). Es una observacion interesante: el modelo es asimetrico en favor de la vision.

### 4.3. Proyecciones y normalizacion

Cada encoder produce un vector que pasa por una **proyeccion lineal** $W_I \in \mathbb{R}^{d_I \times d_e}$ o $W_T \in \mathbb{R}^{d_T \times d_e}$ al espacio compartido $d_e=512$.

Notablemente, **no usan proyeccion no lineal** (a diferencia de SimCLR/BYOL): "no notamos diferencia en eficiencia de entrenamiento entre las dos versiones, y especulamos que las proyecciones no-lineales pueden estar co-adaptadas a detalles del image-only self-supervised". Una decision pragmatica que simplifica.

Despues de la proyeccion: **L2 normalize** $\to$ los vectores viven en la esfera unitaria, y la similitud coseno se reduce a producto punto.

---

## 5. Loss simetrico (InfoNCE)

### 5.1. Formulacion matematica

Dado un batch de $N$ pares $\{(I_i, T_i)\}_{i=1}^N$:

$$z^{I}_i = \frac{W_I \cdot \text{enc}_I(I_i)}{\|W_I \cdot \text{enc}_I(I_i)\|_2}, \quad z^{T}_i = \frac{W_T \cdot \text{enc}_T(T_i)}{\|W_T \cdot \text{enc}_T(T_i)\|_2}$$

$$\text{logits}_{ij} = \frac{z^I_i \cdot z^T_j}{\tau}$$

donde $\tau$ es la **temperatura aprendible**, parametrizada como $\log \tau$ para estabilidad y clipped si los logits exceden 100.

Las dos cross-entropies (filas y columnas):

$$\mathcal{L}_{i \to t} = -\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(\text{logits}_{ii})}{\sum_{j=1}^N \exp(\text{logits}_{ij})}$$

$$\mathcal{L}_{t \to i} = -\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(\text{logits}_{ii})}{\sum_{j=1}^N \exp(\text{logits}_{ji})}$$

$$\mathcal{L}_{\text{CLIP}} = \frac{1}{2}(\mathcal{L}_{i \to t} + \mathcal{L}_{t \to i})$$

### 5.2. Pseudocodigo (Figura 3 del paper)

```python
# image_encoder  - ResNet o Vision Transformer
# text_encoder   - CBOW o Text Transformer
# I[n, h, w, c]  - minibatch de imagenes alineadas
# T[n, l]        - minibatch de textos alineados
# W_i[d_i, d_e]  - proyeccion learned imagen -> embedding
# W_t[d_t, d_e]  - proyeccion learned texto  -> embedding
# t              - parametro de temperatura learned

# 1) extraer features de cada modalidad
I_f = image_encoder(I)            # [n, d_i]
T_f = text_encoder(T)             # [n, d_t]

# 2) embedding multimodal conjunto [n, d_e]
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)

# 3) similitudes coseno escaladas [n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t)

# 4) loss simetrica
labels = np.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss   = (loss_i + loss_t) / 2
```

### 5.3. Por que simetrico

Cada imagen tiene un texto correcto y $N-1$ negativos; cada texto tiene una imagen correcta y $N-1$ negativos. Sumar ambas direcciones balancea la senal y evita degeneraciones donde el modelo solo aprende una direccion del matching.

### 5.4. Por que un solo crop como augmentacion

CLIP usa **solo random square crop de imagenes redimensionadas** como data augmentation. No color jitter, no Gaussian blur, no horizontal flip elaborado. La razon: el dataset es tan grande (400M) que el overfitting *no es preocupacion*, y la augmentacion agresiva podria conflictuar con la semantica del texto.

---

## 6. Training

| Hiperparametro | Valor |
|---|---|
| Modelos entrenados | 5 ResNets (RN50, RN101, RN50x4/16/64) + 3 ViTs (B/32, B/16, L/14) |
| Epocas | 32 |
| Optimizador | Adam con decoupled weight decay (Loshchilov & Hutter 2017) |
| LR schedule | cosine decay (Loshchilov & Hutter 2016) |
| Batch size | **32.768** |
| Hyperparam tuning | grid + random + manual sobre RN50 a 1 epoca |
| Temperatura inicial | $\tau$ = 0.07 (Wu 2018) |
| Mixed precision | si (Micikevicius 2017) |
| Gradient checkpointing | si (Griewank & Walther 2000) |
| Half-precision Adam stats | si (Dhariwal 2020) |
| Sharded similarity computation | si -- cada GPU computa su submatrix |

**Costo de los modelos mas grandes**:
- RN50x64: **18 dias en 592 V100 GPUs** (~10.700 GPU-dias)
- ViT-L/14: **12 dias en 256 V100 GPUs** (~3.072 GPU-dias)
- ViT-L/14@336px: 1 epoca extra a mayor resolucion (FixRes-style)

Esto es el modelo *flagship* y es la referencia "CLIP" en todos los resultados a menos que se especifique otra cosa.

```text
INSIGHT DE EFICIENCIA: ViT vs ResNet
─────────────────────────────────────────────
ViT-L/14:       12 dias x 256 GPUs = 3.072 GPU-dias
RN50x64:        18 dias x 592 GPUs = 10.656 GPU-dias

ViT alcanza mejor accuracy con ~3x menos compute.
Confirma Dosovitskiy 2020: ViTs son mas eficientes
en compute *cuando hay datos suficientes*.
```

---

## 7. Zero-shot evaluation: como se usa CLIP en inferencia

### 7.1. Procedimiento basico

```text
ZERO-SHOT CLASSIFICATION CON CLIP:
─────────────────────────────────────────────────────

DADO: K clases del dataset destino, 1 imagen x

PASO 1 - Construir K prompts:
  prompts = ["A photo of a " + c for c in classes]

PASO 2 - Encodear textos (1 vez, cached):
  T = [text_encoder(p) for p in prompts]
  T = l2_normalize(T)                     # [K, d_e]

PASO 3 - Encodear imagen:
  i = image_encoder(x)
  i = l2_normalize(i)                     # [d_e]

PASO 4 - Similitudes y argmax:
  scores = (i @ T.T) / tau
  pred = argmax(softmax(scores))
```

### 7.2. Vista alternativa: hypernetwork

Una interpretacion elegante (Lei Ba 2015, Ha 2016): el text encoder es una **hypernetwork** que genera los pesos de un clasificador lineal con inputs L2-normalizados, pesos L2-normalizados, sin bias y con escalado de temperatura. Cada paso de pre-entrenamiento de CLIP es un paso en optimizar un proxy de un clasificador para 32.768 clases definidas via lenguaje.

### 7.3. Prompt engineering

El uso de solo el nombre de la clase tiene problemas:

- **Polisemia**: "boxer" puede ser raza de perro o atleta. "crane" puede ser ave o grua de construccion (ambas estan en clases distintas de ImageNet!).
- **Distribution mismatch**: en WIT los textos son frases completas, no etiquetas sueltas.

Soluciones del paper:

```text
TEMPLATES ESTUDIADOS (Tabla 9 del paper):
─────────────────────────────────────────────────────
- Default:        "A photo of a {label}."           (+1.3% ImageNet)
- Context fine:   "A photo of a {label}, a type of pet."
                  (Oxford-IIIT Pets, +mucho)
- Context food:   "A photo of {label}, a type of food."
                  (Food101)
- OCR:            'A photo of "{label}".'            (con quotes)
- Aircraft:       "a photo of a {label}, a type of aircraft."
- Satellite:      "a satellite photo of a {label}."

ENSEMBLE (en espacio de embeddings, no de probabilidades):
- 80 prompts variando "A photo of a big {label}",
  "a small photo of {label}", etc.
- Promedian los embeddings de texto y los re-normalizan
- Costo de inferencia identico a un solo prompt
- Mejora ~3.5% sobre default en ImageNet
- Combinado con prompt engineering: ~+5% en promedio
  sobre 36 datasets
```

---

## 8. Resultados destacados

### 8.1. Comparacion con Visual N-Grams (Tabla 1)

| Dataset | Visual N-Grams | CLIP zero-shot |
|---|---|---|
| Yahoo (aYahoo) | 72.4 | **98.4** |
| ImageNet | 11.5 | **76.2** |
| SUN | 23.0 | **58.5** |

CLIP no es una mejora *direct comparison* (10x mas datos, 100x mas compute, Transformer-based), pero el delta es enorme.

### 8.2. CLIP zero-shot vs ResNet-50 supervisado linear probe (Figura 5)

Sobre 27 datasets:

- **CLIP gana en 16/27 datasets**.
- Mayores ganancias: Stanford Cars +28.9, Country211 +23.2, Food101 +22.5, Kinetics700 +14.5, SST2 +12.4 (OCR-heavy).
- Mayores perdidas: EuroSAT -37.1, KITTI Distance -34.0, GTSRB -19.5, CLEVRCounts -18.2 (tareas especializadas o sistematicas).

### 8.3. Few-shot anti-intuitivo (Figura 6)

Resultado contraintuitivo: pasar de zero-shot CLIP a 1-shot logistic regression sobre features CLIP **baja** la accuracy. La razon: el clasificador 1-shot sobreajusta al unico ejemplo, mientras que el zero-shot ha sido especificado por una descripcion natural rica. Solo a partir de **4-shot** el linear probe alcanza al zero-shot, y empieza a superarlo claramente con 16-shot.

```text
PERFORMANCE SOBRE 20 DATASETS (Figura 6):
  Zero-Shot CLIP        ────── ★ ~70%
  Linear Probe CLIP 1-shot   ─ 45%
  Linear Probe CLIP 4-shot   ─ 70%  (iguala zero-shot)
  Linear Probe CLIP 16-shot  ─ 75%  (supera zero-shot)
  Best 16-shot otro modelo   ─ ~70% (BiT-M ImageNet-21K)
```

### 8.4. Linear probe sobre 27 datasets (Figura 10)

Comparado con 16 modelos pretrained (SimCLRv2, BYOL, BiT-M, EfficientNet-NS, ViT-21K, ResNet, etc.):

- CLIP-ViT y CLIP-ResNet **dominan la frontera de Pareto** (accuracy vs FLOPs).
- ViT-L/14@336px supera al mejor modelo previo (Noisy Student EfficientNet-L2) en **+2.6% promedio** sobre 27 datasets, *con menos compute*.
- En 21 de 27 datasets CLIP supera a EfficientNet-NS, con mejoras grandes en SST2 (+23.6), Country211 (+22.7), HatefulMemes (+18.8), Stanford Cars (+15.9), GTSRB (+14.7).

### 8.5. Scaling smooth (Figura 9)

Error promedio sobre 39 evaluaciones x 36 datasets es bien modelado por una **trend log-lineal** sobre 44x rango de compute (RN50 a RN50x64). Confirma "scaling laws" en vision multimodal -- mismo fenomeno que GPT-3 en NLP.

---

## 9. Robustez a natural distribution shift (Seccion 3.3)

### 9.1. El problema

Modelos entrenados en ImageNet tienen una "robustness gap": su accuracy cae mucho en distribuciones naturales pero relacionadas (ImageNet-V2, Sketch, ImageNet-A, ObjectNet). Taori et al. 2020 propuso medir **effective robustness**: mejora en out-of-distribution accuracy *mas alla* de lo predicho por la in-distribution accuracy.

### 9.2. CLIP cierra el gap (Figura 13)

Comparacion con un ResNet-101 que tiene la **misma accuracy ImageNet** que zero-shot CLIP (76.2%):

| Dataset | RN101 | ZS CLIP | Delta |
|---|---|---|---|
| ImageNet | 76.2 | 76.2 | 0% |
| ImageNetV2 | 64.3 | 70.1 | +5.8 |
| ImageNet-R | 37.7 | 88.9 | **+51.2** |
| ObjectNet | 32.6 | 72.3 | **+39.7** |
| ImageNet Sketch | 25.2 | 60.2 | **+35.0** |
| ImageNet-A | 2.7 | 77.1 | **+74.4** |

El zero-shot CLIP **reduce el robustness gap hasta 75%**.

### 9.3. Adaptar a ImageNet rompe la robustez (Figura 14)

Detalle interesante: si se hace fine-tune o linear probe de CLIP sobre ImageNet (sube de 76.2 a 85.4), la **robustness promedio cae**. Es decir, ajustarse a la distribucion ImageNet *especificamente* re-introduce el sobreajuste a sus correlaciones espureas. Sugiere que la robustez de CLIP viene del pre-entrenamiento *broad*, no de algun bias inherente de la arquitectura.

---

## 10. Limitaciones (Seccion 6 del paper)

El paper es notablemente honesto al listar fallas:

### 10.1. Tareas fine-grained y abstractas

- Variantes de modelos de auto, especies de flores, tipos de aviones: rendimiento pobre.
- Conteo de objetos (CLEVRCounts): casi random.
- Distancia al auto mas cercano (KITTI): casi random.
- Deteccion de tumores en linfoides (PatchCamelyon): no hay senal en el preentrenamiento.

### 10.2. Generalizacion *truly novel*

CLIP generaliza bien dentro del *distribution* de imagenes naturales, pero falla en datos OOD reales:

- **MNIST**: 88% accuracy; logistic regression sobre pixels crudos lo supera. Razon: casi no hay digitos manuscritos en el preentrenamiento web.
- "CLIP intenta circumvalar el problema entrenando con tanta data variada que todo sea in-distribution. Es una asuncion ingenua, como demuestra MNIST."

### 10.3. No genera, solo elige

CLIP escoge entre clases dadas. No produce captions ni respuestas abiertas. Combinar el contrastive con un objetivo generativo podria unir flexibilidad con eficiencia (anticipa BLIP, CoCa).

### 10.4. Few-shot deficiente

Zero-shot > 1-shot, y se necesita 4-shot para igualar. Los humanos saltan de 54% a 76% con un solo ejemplo. Algo fundamental falta en como CLIP integra ejemplos pocos.

### 10.5. Data efficiency

400M pares x 32 epocas = **12.8B imagenes vistas**. Si se mostrara una imagen por segundo, tomaria **405 anos**. Combinar self-supervision (Henaff 2020, Chen 2020c) o self-training (Lee 2013, Xie 2020) podria mejorar.

### 10.6. Sesgos sociales (Seccion 7)

Resultados *preocupantes* documentados por los autores:

- En FairFace + clases agregadas (animal, gorilla, criminal, suspicious person): **16.5% de imagenes de hombres** se clasifican en categorias relacionadas a crimen vs 9.8% de mujeres. Personas <20 anos: 18% en crime-related y **14% en non-human animal categories** (vs ~12% para 20-60 anos).
- En FairFace race: imagenes 'Black' tienen ~14% de tasa de mis-clasificacion como non-human, vs <8% en otras razas.
- En images of Members of Congress + GCV/Rekognition combined labels: mujeres reciben desproporcionadamente labels de apariencia ('blonde', 'brown hair'), hombres reciben labels de status ('executive', 'doctor').
- Class design es un determinante critico: anadir 'child' como categoria reduce drasticamente la mis-clasificacion de menores en clases criminales/no-humanas.

Los autores son explicitos: estos no son artefactos, son **reflejos de los biases de la web** que CLIP absorbe y *amplifica* via su flexibilidad de prompt.

### 10.7. Surveillance

CLIP es *capaz* en CCTV coarse classification (91.8%) y *moderadamente capaz* en celebrity ID (59.2% sobre 100 clases). Los autores discuten que esto habilita aplicaciones de vigilancia bespoke con cero datos de entrenamiento -- una capacidad nueva que requiere reflexion etica.

### 10.8. Methodological

- Evaluation suite de 27 datasets es *somewhat haphazard* (cherry-picked, en parte co-adaptado al desarrollo de CLIP).
- Querying full validation sets repetidamente puede haber introducido leakage en hiperparametros.
- Cobertura de pre-training data desconocida para investigadores externos.

---

## 11. Sucesores y trabajos relacionados

```text
LINEA DE SUCESORES DIRECTOS:
─────────────────────────────────────────────────────

ALIGN (Jia 2021, Google)         contemporaneo, 1.8B pares ruidosos
  ↓
DeCLIP (Li 2021)                 anade self-supervision intra-modal
                                 + nearest-neighbor supervision
  ↓
LiT (Zhai 2022, Google)          locked image, only train text encoder
                                 sobre encoder pre-entrenado
  ↓
OpenCLIP (Cherti 2023, LAION)    receta abierta, LAION-400M/2B
                                 valida y supera a CLIP en muchos casos
  ↓
SigLIP (Zhai 2023, Google)       reemplaza softmax por sigmoid loss,
                                 robusto a batch size pequeno,
                                 entrena con bs=32 incluso
  ↓
MetaCLIP (Xu 2023, Meta)         documenta receta de curacion
                                 reproduce o supera CLIP openly
  ↓
EVA-CLIP (Sun 2023)              scaling x 5B parametros, MIM init
  ↓
DINOv2 (Oquab 2023, Meta)        no usa texto, pero compite con CLIP
                                 en muchas tareas via self-supervision
                                 escalada
─────────────────────────────────────────────────────

COMPLEMENTOS HACIA GENERATIVO:
- BLIP (Li 2022)         contrastive + caption generation + ITM
- CoCa (Yu 2022)         contrastive + autoregressive
- Flamingo (Alayrac 2022) frozen vision encoder (CLIP-like) + LLM
- LLaVA (Liu 2023)       CLIP encoder + Vicuna LLM, instruction-tuned
─────────────────────────────────────────────────────

GENERACION CONDICIONADA EN TEXTO (downstream):
- DALL-E 2 (Ramesh 2022)        usa CLIP image embeddings como prior
- Stable Diffusion (Rombach 2022) usa CLIP text encoder
                                  para condicionar la difusion
- Imagen (Saharia 2022)         alternativa con T5 (sin CLIP)
- IP-Adapter, ControlNet, etc.  todos asumen CLIP-style features
```

---

## 12. Impacto: la era multimodal abierta

### 12.1. Stable Diffusion y la generacion text-to-image

Stable Diffusion (Rombach et al. 2022) usa el **text encoder de CLIP ViT-L/14** congelado para producir embeddings que condicionan el U-Net del modelo de difusion en el espacio latente. Sin CLIP, no existiria SD ni MidJourney en su forma actual.

### 12.2. DALL-E 2: dos pasadas con CLIP

DALL-E 2 (Ramesh 2022) usa CLIP en dos lugares:
1. Un *prior* aprende a mapear CLIP text embeddings $\to$ CLIP image embeddings.
2. Un *decoder* (difusion) genera imagenes desde CLIP image embeddings.

CLIP cumple aqui el rol de *espacio semantico canonico* compartido entre texto e imagen.

### 12.3. LLM multimodal

Casi todos los multi-modal LLMs (LLaVA, MiniGPT-4, Flamingo, IDEFICS) usan un **encoder visual estilo CLIP** (frozen o parcialmente fine-tuned) como adaptador entre la imagen y el espacio del LLM. CLIP demostro que se podia construir un encoder visual *general purpose* con texto solamente.

### 12.4. Retrieval, busqueda, deduplicacion

- **CLIP retrieval**: dado texto, encuentra imagen relevante en una base con cosine sim. Base de productos como Pinterest visual search, Adobe Stock, etc.
- **Deduplicacion semantica** de datasets: LAION usa CLIP para filtrar pares con similitud baja.
- **Image-image search via texto**: "encuentra imagenes similares a esta pero con un perro en vez de un gato".

---

## 13. Lecciones transferibles

```text
LECCION 1: SUPERVISION DEBIL A ESCALA > SUPERVISION FUERTE LIMITADA
─────────────────────────────────────────────────────
Cuando puedes acceder a 1000x mas datos con una senal
ruidosa pero abierta (texto natural), eso supera datasets
curados pero estrechos. La generalidad es funcion del
tamano y diversidad mas que de la calidad nominal.

LECCION 2: CONTRASTIVE > GENERATIVE PARA REPRESENTATION LEARNING
─────────────────────────────────────────────────────
Predecir la *correspondencia* entre dos modalidades es
un proxy mas eficiente que predecir el contenido exacto.
Aplicable a audio-texto, codigo-doc, retrieval cross-lingual.

LECCION 3: ESPACIO MULTIMODAL COMPARTIDO HABILITA ZERO-SHOT
─────────────────────────────────────────────────────
Si dos modalidades se proyectan a una misma esfera, las
queries de una pueden indexar ejemplos de la otra. Esto
generaliza a ASR, retrieval, robotics (CLIP-RT, VC-1).

LECCION 4: PROMPT ENGINEERING ES PARTE DE LA INTERFAZ
─────────────────────────────────────────────────────
La forma de la consulta importa tanto como el modelo.
Templates contextuales y ensembling sobre prompts dan
mejoras "gratis" comparables a 4x mas compute.

LECCION 5: ROBUSTEZ VIENE DEL PRE-ENTRENAMIENTO BROAD
─────────────────────────────────────────────────────
Modelos entrenados en distribuciones diversas resisten
shifts mucho mejor que modelos entrenados en datos
"limpios" pero estrechos. Adaptar al dataset destino
*degrada* la robustez.

LECCION 6: TEMPERATURA APRENDIBLE
─────────────────────────────────────────────────────
$\tau$ como parametro learnable (en log-space, con clip)
es una mejora barata sobre temperaturas fijas. Patron
reutilizado en SimCLR, BYOL, MoCo, todos los descendientes.

LECCION 7: VIT > RESNET CON DATOS SUFICIENTES
─────────────────────────────────────────────────────
A escala 400M, ViT es ~3x mas eficiente en compute.
La inductive bias de las CNNs deja de pagar dividendos.

LECCION 8: PUBLICAR LIMITACIONES Y SESGOS ES CIENCIA
─────────────────────────────────────────────────────
La Seccion 7 ("Broader Impacts", 6 paginas) sienta un
estandar para papers de modelos generales: documentar
biases sociales con experimentos concretos, no solo
disclaimers genericos.
```

---

## 14. Resumen en una pagina

```text
PROBLEMA:    Modelos de vision dependen de etiquetas curadas (ImageNet,
             JFT). Esto limita generalidad, escalabilidad y zero-shot.
             Visual N-Grams logro 11.5% ZS ImageNet -- muy lejos del SOTA.

SOLUCION:    Pre-entrenar dos encoders (imagen, texto) sobre 400M pares
             (image, text) extraidos de la web (WIT) con un objetivo
             contrastivo simetrico (InfoNCE) sobre similitudes coseno
             escaladas con temperatura aprendible.

ARQUITECTURA:
  - Image encoder: ViT-L/14 (flagship) o ResNets modificadas (RN50x64)
  - Text encoder: Transformer 12L/8H/512w decoder-style, ~63M params
  - Proyecciones lineales a d_e=512, L2 normalize
  - Loss: 1/2 (CE_filas + CE_columnas) sobre matriz N x N de logits

ENTRENAMIENTO:
  - Batch 32.768, Adam decoupled, cosine LR, 32 epocas, mixed precision
  - ViT-L/14: 12 dias en 256 V100s, una epoca extra a 336px

ZERO-SHOT:
  - Prompts: "A photo of a {label}" + variantes contextuales
  - Ensembling: 80 prompts promediados en espacio de embeddings
  - argmax sobre similitud coseno con embeddings de texto cacheados

RESULTADOS:
  - ImageNet ZS: 76.2% (iguala ResNet-50 supervisado)
  - 16/27 datasets: ZS CLIP > linear probe ResNet-50 supervisado
  - Robustez: cierra hasta 75% del gap en ImageNet-R/A/Sketch/V2/ObjNet
  - Linear probe ViT-L/14@336: supera EfficientNet-NS, BiT-M, SimCLRv2
  - Scaling: log-lineal sobre 44x rango de compute

LIMITACIONES:
  - Fine-grained: aviones, autos, flores, texturas (poor)
  - Sistematicas: counting, satellite, traffic signs (near random)
  - OOD verdadero: MNIST 88% (pixel logreg lo supera)
  - Few-shot: zero-shot > 1-shot (anti-intuitivo)
  - Sesgos sociales: race/gender/age clasificados denigrantemente
  - Costo: ~3K-10K GPU-dias

IMPACTO:
  - Stable Diffusion usa text encoder de CLIP
  - DALL-E 2 usa CLIP image embeddings como prior
  - LLaVA, BLIP, Flamingo usan CLIP-style encoders
  - Inicia paradigma "roll your own classifier" via lenguaje
  - Sucesores: ALIGN, OpenCLIP, SigLIP, MetaCLIP, EVA-CLIP

LECCION GENERAL:
  Supervision debil pero amplia (texto web) supera supervision fuerte
  pero estrecha (etiquetas curadas) cuando el escalado es factible.
  El espacio multimodal compartido habilita una nueva interfaz para
  vision por computador: lenguaje natural como linguaje de control.
```
