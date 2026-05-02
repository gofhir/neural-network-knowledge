---
title: "Vision Transformer (ViT)"
weight: 296
math: true
---

{{< paper-card
    title="An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
    authors="Dosovitskiy, Beyer, Kolesnikov, Weissenborn, Zhai, Unterthiner, Dehghani, Minderer, Heigold, Gelly, Uszkoreit, Houlsby"
    year="2021"
    venue="ICLR 2021"
    pdf="/papers/vit-dosovitskiy-2021.pdf"
    arxiv="2010.11929" >}}
El paper que llevo el **Transformer al dominio visual sin ninguna convolucion**. Dosovitskiy y colaboradores (Google Research / Brain Team) demuestran que una imagen puede tratarse como una secuencia de **parches de 16x16 pixeles** -- "an image is worth 16x16 words" -- y procesarse con un encoder Transformer estandar tipo BERT. Cuando se pre-entrena a escala suficiente (ImageNet-21k o JFT-300M), ViT iguala o supera a las mejores CNN (BiT, Noisy Student) usando substancialmente menos computo, y abre la puerta a que la vision adopte la misma arquitectura que NLP.
{{< /paper-card >}}

---

## Contexto

Desde AlexNet (2012), la vision por computador estuvo dominada por **CNN**: AlexNet, VGG, ResNet, Inception, EfficientNet, BiT. La conviccion era que los **inductive biases** convolucionales -- localidad, equivarianza a translacion, jerarquia espacial -- eran imprescindibles para procesar imagenes con eficiencia muestral.

Mientras tanto, NLP habia migrado completamente a **Transformers** (Vaswani 2017, BERT, GPT). Hubo intentos de hibridar: CNN con self-attention (Bello 2019), reemplazar convoluciones por atencion local (Ramachandran 2019, Wang 2020), Sparse Transformers (Child 2019), o aplicar atencion full sobre patches diminutos de 2x2 (Cordonnier 2020). Ninguno habia escalado al nivel de las CNN SOTA.

La pregunta del paper es directa: **un Transformer puro, sin convoluciones, sin trucos especificos de vision, puede competir con CNN si se le da datos suficientes?**

---

## Ideas principales

### 1. Patches como tokens

Una imagen $H \times W \times C$ se corta en $N = HW/P^2$ parches de tamano $P \times P$ (tipicamente $P=16$). Cada parche se aplana a un vector de $P^2 C$ dimensiones. Asi, una imagen de 224x224 con $P=16$ produce $N = 196$ tokens.

### 2. Patch embedding (proyeccion lineal)

Los parches aplanados se proyectan linealmente a $D$ dimensiones (latent size del Transformer) mediante una matriz aprendida $E \in \mathbb{R}^{(P^2 C) \times D}$. Equivale a una **convolucion de stride $P$ y kernel $P$**.

### 3. [class] token aprendible

Como en BERT, se prepende un token aprendible $x_{\text{class}}$ a la secuencia de embeddings de parches. Su estado en la salida del encoder $z_L^0$ se usa como representacion de la imagen para clasificacion.

### 4. Positional embedding aprendido (1D)

Se anade un embedding posicional 1D **aprendido** a cada token (incluyendo el [class]). Sorprendentemente, los autores reportan que embeddings 2D-aware no aportan mejora pese a que las imagenes son intrinsecamente 2D -- el modelo aprende solo la estructura espacial.

### 5. Encoder Transformer estandar

Apilan $L$ bloques identicos al original de Vaswani: pre-LN, multi-head self-attention, MLP con GELU, residual connections. **Nada nuevo en la arquitectura interna.** La elegancia esta en la simplicidad: Transformer de NLP, casi out-of-the-box.

### 6. Pre-training masivo + fine-tuning

ViT se pre-entrena supervisado en datasets grandes (ImageNet-21k 14M, JFT-300M 303M) y se hace fine-tune en tareas pequenas (ImageNet-1k, CIFAR, Pets, Flowers, VTAB). Sin pre-entrenamiento masivo, ViT pierde frente a ResNets de tamano comparable.

---

## Resultados clave

| Modelo | Pre-train | ImageNet | CIFAR-100 | Oxford Pets | TPUv3 core-days |
|---|---|---|---|---|---|
| ViT-H/14 | JFT-300M | **88.55%** | **94.55%** | **97.56%** | 2.5k |
| ViT-L/16 | JFT-300M | 87.76% | 93.90% | 97.32% | 0.68k |
| ViT-L/16 | ImageNet-21k | 85.30% | 93.25% | 94.67% | 0.23k |
| BiT-L (ResNet152x4) | JFT-300M | 87.54% | 93.51% | 96.62% | 9.9k |
| Noisy Student (EfficientNet-L2) | JFT + pseudo | 88.4% | -- | -- | 12.3k |

ViT-H/14 fija nuevo SOTA en ImageNet (88.55%) usando **~4x menos TPU-days** que Noisy Student y **~4x menos** que BiT-L.

---

## Trade-off datos vs inductive bias

La grafica clave (Figuras 3 y 4 del paper) cuenta la historia completa:

- Pre-entrenado en **ImageNet (1.3M)**: ViT-Large queda **por debajo** de BiT-ResNet. Las CNN ganan con poco data porque sus biases (locality, translation equivariance) son rentables.
- Pre-entrenado en **ImageNet-21k (14M)**: empate aproximado.
- Pre-entrenado en **JFT-300M (303M)**: ViT **supera** a BiT y la ventaja crece con el tamano del modelo.

El cruce ocurre alrededor de **~100M imagenes**. La leccion: **inductive biases son un trade-off entre velocidad de aprendizaje y techo de performance**. Menos biases requiere mas datos, pero permite mejor performance asintotica.

---

## Por que importa

ViT es el momento en que la vision **adopto la arquitectura de NLP**. Sus consecuencias inmediatas y a largo plazo:

- **DeiT** (Touvron 2021): training data-efficient con destilacion, ViT competitivo solo con ImageNet-1k.
- **Swin Transformer** (Liu 2021): atencion en ventanas con shift, jerarquia multi-escala, backbone para deteccion/segmentacion.
- **MAE** (He 2022): masked autoencoder pretrain self-supervised, ViT-H/14 alcanza 87.8% ImageNet sin etiquetas masivas.
- **ConvNeXt** (Liu 2022): "modernizar las CNN" con trucos de ViT (LN, GELU, depthwise conv grandes) para recuperar terreno.
- **CLIP** (Radford 2021), **DINO** (Caron 2021), **SAM** (Kirillov 2023), **DALL-E 2**, **Stable Diffusion** (encoder visual): todos usan ViT como backbone.
- **Modelos multimodales** (Flamingo, GPT-4V, Gemini, Claude vision): la unificacion de tokens visuales y textuales en una misma secuencia es directamente posible **porque ViT existe**.

---

## Limitaciones

- **Necesita pre-training masivo**: sin JFT-300M (o equivalente self-supervised), ViT pierde frente a CNN bien diseñadas.
- **Costo cuadratico** $O(N^2)$ en numero de patches -- imagenes de alta resolucion con patches pequeños son prohibitivas sin trucos (FlashAttention, ventanas locales tipo Swin, atencion lineal).
- **Sin jerarquia espacial**: a diferencia de CNN o Swin, ViT mantiene resolucion constante a traves de capas. No es ideal para deteccion/segmentacion densa sin adaptaciones.
- **Position embeddings 1D fijos al tamano**: cambiar resolucion en fine-tuning requiere interpolacion 2D ad-hoc.
- **Inestabilidad en training**: requiere warmup largo, weight decay alto, gradient clipping. Trabajos posteriores (LayerScale, registers de Darcet 2024) lo estabilizan.

---

## Notas y enlaces

- Codigo y modelos pre-entrenados oficiales: [github.com/google-research/vision_transformer](https://github.com/google-research/vision_transformer).
- La **Figura 1** muestra la arquitectura completa (parches → linear projection → +pos → encoder → MLP head).
- La **Figura 7** es el analisis interno mas citado: filtros del patch embedding (Gabor-like), similitud de position embeddings (estructura 2D emergente), distancia media de atencion por capa (global desde capas tempranas, a diferencia de CNN).
- Apendice incluye experimento preliminar de **masked patch prediction** (auto-supervisado tipo BERT) -- semilla de lo que MAE explotaria a fondo.
- Follow-ups directos:
  - **Touvron et al. 2021** "DeiT: Training data-efficient image transformers" -- distillation token, sin JFT.
  - **Liu et al. 2021** "Swin Transformer" -- ventanas con shift, backbone jerarquico.
  - **He et al. 2022** "Masked Autoencoders Are Scalable Vision Learners" (MAE).
  - **Liu et al. 2022** "A ConvNet for the 2020s" (ConvNeXt).

Ver fundamentos: [Vision Transformer](/fundamentos/vision-transformer) · [Transformer](/fundamentos/transformer) · [Redes Convolucionales](/fundamentos/redes-convolucionales) · [Clase 14](/clases/clase-14).
