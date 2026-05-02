---
title: "CLIP (Contrastive Language-Image Pre-training)"
weight: 295
math: true
---

{{< paper-card
    title="Learning Transferable Visual Models From Natural Language Supervision"
    authors="Radford, Kim, Hallacy, Ramesh, Goh, Agarwal, Sastry, Askell, Mishkin, Clark, Krueger, Sutskever"
    year="2021"
    venue="ICML 2021"
    pdf="/papers/clip-radford-2021.pdf"
    arxiv="2103.00020" >}}
Introduce **CLIP** -- el modelo que demostro que aprender representaciones visuales con supervision de lenguaje natural a escala (400M pares imagen-texto) produce features que transfieren *zero-shot* a docenas de tareas de vision por computador. CLIP empareja un encoder de imagen y un encoder de texto en un espacio multimodal compartido via un objetivo contrastivo simetrico, y al hacerlo iguala la accuracy de un ResNet-50 supervisado en ImageNet sin haber visto ni una sola etiqueta de ImageNet durante el entrenamiento.
{{< /paper-card >}}

---

## Contexto

A inicios de 2021, los sistemas SOTA de vision dependian de **datasets cerrados con etiquetas curadas** (ImageNet 1K, JFT-300M con clases predefinidas). Esto introducia tres limitaciones criticas:

- **Costo de supervision**: agregar nuevas categorias requeria etiquetar datos manualmente.
- **Generalidad limitada**: un clasificador entrenado para 1000 clases de ImageNet solo sabia esas 1000 clases.
- **Zero-shot incipiente**: Visual N-Grams (Li et al. 2017) habia logrado solo 11.5% top-1 en ImageNet zero-shot, muy lejos del estado del arte supervisado.

Mientras tanto, en NLP los modelos *task-agnostic* entrenados sobre texto crudo (GPT, BERT, T5) ya transferian sin necesidad de heads especializados. La pregunta natural era: **se puede replicar este paradigma en vision?**

CLIP responde afirmativamente, escalando el contrastive learning imagen-texto (anteriormente probado a pequena escala por ConVIRT, VirTex, ICMLM) a un dataset construido desde la web (WIT, 400M pares).

---

## Ideas principales

### 1. Pares imagen-texto como senal de supervision

En vez de etiquetas one-hot, CLIP usa la **descripcion textual** que acompana naturalmente a la imagen en la web. Esto da supervision *abierta* y *escalable*: cualquier imagen con texto co-ocurrente es senal de entrenamiento.

El dataset **WIT (WebImageText)** se construye buscando 500.000 *queries* (palabras de Wikipedia con frecuencia >=100, bigramas con alto PMI, sinsets de WordNet) y reteniendo hasta 20.000 pares por query, para un total aproximado de **400M pares** -- comparable en word count al dataset usado por GPT-2.

### 2. Doble encoder: imagen y texto

- **Image encoder**: ResNet modificada (RN50, RN101, RN50x4/16/64 con scaling EfficientNet-style) o **Vision Transformer** (ViT-B/32, ViT-B/16, ViT-L/14).
- **Text encoder**: Transformer de 12 capas, 8 cabezas, ancho 512 (~63M parametros). Usa BPE con vocabulario 49.152, secuencia maxima 76 tokens, y self-attention enmascarada (decoder-style) para preservar la opcion de language modeling auxiliar.

Cada encoder produce un vector que se proyecta linealmente al espacio multimodal de dimension $d_e=512$, y se aplica **L2-normalize** para vivir en la esfera unitaria.

### 3. Loss contrastivo simetrico (InfoNCE)

Dado un batch de $N$ pares $(I_i, T_i)$, se calculan los logits de similitud coseno escalada por temperatura $\tau$ aprendible:

$$\text{logits}_{ij} = \frac{\langle \text{enc}_I(I_i), \text{enc}_T(T_j) \rangle}{\tau}$$

Y se aplican **dos** cross-entropies (una por direccion):

$$\mathcal{L} = \frac{1}{2}\big(\mathcal{L}_{i \to t} + \mathcal{L}_{t \to i}\big)$$

donde la diagonal son los $N$ pares correctos y los $N^2 - N$ off-diagonales son negativos. La temperatura $\tau$ se aprende como log-parametro (clip a logits <100 para evitar inestabilidad).

### 4. Zero-shot via prompts

En inferencia, para clasificar entre $K$ clases:

1. Construir prompts: "A photo of a {class}".
2. Pasar cada prompt por el text encoder $\to$ $K$ embeddings de texto.
3. Pasar la imagen por el image encoder $\to$ 1 embedding de imagen.
4. **Argmax** sobre similitudes coseno.

El text encoder actua como una *hypernetwork* que sintetiza un clasificador lineal para cada conjunto de clases descritas en lenguaje natural.

### 5. Prompt engineering y ensembling

Usar solo el nombre de la clase es subobtimo (polisemia, falta de contexto). El paper propone:

- **Templates contextuales**: "A photo of a {label}, a type of pet" sube ~3-5 puntos en Oxford-IIIT Pets.
- **Ensembling de 80 prompts** en el espacio de embeddings $\to$ mejora ImageNet en ~3.5 puntos *gratis* en costo de inferencia (los 80 prompts se promedian una vez).

---

## Resultados experimentales

### Zero-shot ImageNet

| Modelo | Top-1 zero-shot |
|---|---|
| Visual N-Grams (Li 2017) | 11.5% |
| CLIP RN50 | ~59.6% |
| CLIP RN50x64 | 73.6% |
| **CLIP ViT-L/14@336px** | **76.2%** |

CLIP ViT-L/14 **iguala a un ResNet-50 supervisado en ImageNet** sin haber visto ninguna de las 1.28M imagenes etiquetadas.

### Transferencia broad

Sobre 27 datasets:

- CLIP zero-shot **gana** vs linear probe sobre ResNet-50 features en **16/27** datasets.
- Mejoras grandes: Stanford Cars (+28.9), Country211 (+23.2), Food101 (+22.5), Kinetics700 (+14.5).
- CLIP zero-shot iguala o supera un linear probe de **4-shot** en su propio espacio de features.

### Robustez a distribution shift

Comparado con un ResNet-101 con la misma accuracy en ImageNet (76.2%):

| Dataset | RN101 | Zero-Shot CLIP | Delta |
|---|---|---|---|
| ImageNetV2 | 64.3 | 70.1 | +5.8 |
| ImageNet-R | 37.7 | 88.9 | +51.2 |
| ObjectNet | 32.6 | 72.3 | +39.7 |
| ImageNet Sketch | 25.2 | 60.2 | +35.0 |
| ImageNet-A | 2.7 | 77.1 | +74.4 |

CLIP cierra hasta **75% del gap de robustez** que sufren los modelos supervisados ImageNet. La supervision con lenguaje natural produce representaciones mucho menos sobreajustadas a la distribucion de entrenamiento.

---

## Por que importa

### Era multimodal

CLIP es la pieza fundacional de una nueva ola de modelos:

- **DALL-E 2** (Ramesh 2022) usa los embeddings de imagen de CLIP como prior para el difusor.
- **Stable Diffusion** (Rombach 2022) usa el text encoder de CLIP para condicionar la generacion.
- **Flamingo, BLIP, LLaVA** usan encoders al estilo CLIP para integrar vision con LLMs.

### Paradigma zero-shot en vision

CLIP introdujo en mainstream la idea de **clasificadores definidos en lenguaje natural** -- "roll your own classifier" sin reentrenar. Ya no se necesita un dataset etiquetado para una taxonomia nueva: basta describirla.

### Lecciones de scaling

CLIP confirma en vision la ley de escalado observada en NLP: el rendimiento mejora suavemente con compute (log-lineal en GFLOPs) y con tamano de dataset. Tambien valida que **ViT > ResNet** en este regimen (~3x mas eficiente en compute).

---

## Limitaciones

- **Tareas fine-grained**: CLIP es debil en distinguir variantes de aviones (FGVCAircraft -11.3 vs RN50 supervisado), especies de flores (-12.5), texturas (-16.6).
- **Tareas abstractas o sistematicas**: contar objetos (CLEVRCounts -18.2), satellite imagery (EuroSAT -34.0), traffic signs (GTSRB -18.4), distancia al auto mas cercano (KITTI -19.5).
- **OCR rendido vs manuscrito**: rinde bien en SST2 (texto digital) pero solo 88% en MNIST -- el preentrenamiento web no incluye digitos manuscritos.
- **No genera, solo elige**: CLIP debe escoger entre clases dadas en el prompt. No produce captions ni respuestas abiertas.
- **Few-shot anti-intuitivo**: pasar de zero-shot a 1-shot logistico **baja** la accuracy (los humanos suben de 54% a 76%). Las features no integran bien ejemplos pocos.
- **Sesgos sociales**: en FairFace, CLIP zero-shot clasifica desproporcionadamente a hombres negros bajo categorias relacionadas a crimen y a personas <20 anos en categorias no humanas. Crawford-style harms estan presentes y se amplifican con clases mal disenadas.
- **Costo masivo**: ViT-L/14 requiere 256 V100s por 12 dias (~3072 GPU-dias).
- **Data overlap**: el dataset puede contener material con copyright o privado; el filtrado fue minimo.

---

## Notas y enlaces

- Repositorio oficial: [github.com/OpenAI/CLIP](https://github.com/OpenAI/CLIP).
- **OpenCLIP** (Cherti et al. 2023, LAION): re-implementacion abierta entrenada sobre LAION-400M/2B, valida y extiende los resultados.
- **ALIGN** (Jia et al. 2021, Google): contemporaneo, similar a CLIP pero con 1.8B pares ruidosos.
- **SigLIP** (Zhai et al. 2023): reemplaza el softmax por una sigmoid loss pairwise, mas robusta a batch size pequeno.
- **MetaCLIP** (Xu et al. 2023, Meta): documenta y abre la receta de curacion del dataset.
- Lectura recomendada: Sec 1-3 (intro, approach, experimentos) y Sec 6 (limitaciones). El appendix tiene detalles de prompt engineering por dataset y de robustez.

Ver fundamentos: [Aprendizaje Contrastivo](/fundamentos/aprendizaje-contrastivo) -- [Vision Transformer](/fundamentos/vision-transformer) -- [Transformer](/fundamentos/transformer) -- [Clase 14](/clases/clase-14).
