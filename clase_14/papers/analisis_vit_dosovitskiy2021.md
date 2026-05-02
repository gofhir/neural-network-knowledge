# Analisis del Paper: An Image is Worth 16x16 Words — Transformers for Image Recognition at Scale

**Autores**: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby
**Institucion**: Google Research, Brain Team (Berlin, Zurich, Amsterdam)
**Publicado en**: ICLR 2021 (arXiv:2010.11929, Octubre 2020)
**Nota**: 22 paginas (9 paginas de paper + 13 de apendice). Ya con >40,000 citas en Google Scholar.

> PDF descargado en: [vit-dosovitskiy-2021.pdf](vit-dosovitskiy-2021.pdf)

---

## 1. Resumen Ejecutivo

ViT (Vision Transformer) demuestra por primera vez que un **Transformer encoder puro**, sin ninguna convolucion, puede igualar o superar a las mejores CNN en clasificacion de imagenes a gran escala. La receta:

1. Cortar la imagen en parches fijos (16x16 pixeles).
2. Tratar cada parche como un token: aplanar + proyeccion lineal aprendida.
3. Anadir un token [class] aprendible (estilo BERT) y position embeddings 1D aprendidos.
4. Procesar con un Transformer encoder estandar de Vaswani 2017.
5. Pre-entrenar en JFT-300M (303M imagenes etiquetadas), fine-tune en la tarea de interes.

Resultado: **88.55% en ImageNet, 94.55% en CIFAR-100, 99.50% en CIFAR-10, 77.63% en VTAB-19**, todo con menos computo que BiT (ResNet) o Noisy Student (EfficientNet).

El descubrimiento conceptual no es solo el numero, sino el **trade-off datos vs inductive bias**: las CNN ganan con datasets pequeños porque sus biases (locality, translation equivariance) son rentables; ViT, sin esos biases, gana cuando hay suficientes datos para aprenderlos. El cruce ocurre alrededor de ~100M imagenes.

---

## 2. Aporte Central

### 2.1. La idea minimalista

> "An image is worth 16x16 words"

Esta frase resume todo el paper. Si un Transformer puede procesar palabras en NLP, deberia poder procesar parches en vision. **No hay que reinventar la arquitectura**: solo hay que adaptar el input.

```text
NLP:                Texto  →  tokens  →  embeddings  →  Transformer  →  logits
                    "Hola mundo"  →  [Hola, mundo]  →  [v1, v2]  →  ...

Vision (ViT):       Imagen  →  parches  →  patch embeddings  →  Transformer  →  logits
                    [HxWx3]  →  [16x16x3, ...]  →  [v1, ..., vN]  →  ...
```

### 2.2. Decisiones de diseno (lo que NO hicieron)

Los autores deliberadamente **no introducen** mecanismos especificos de vision:

- No hay convoluciones (excepto la proyeccion lineal del parche, que es un caso degenerado de conv).
- No hay pooling jerarquico.
- No hay atencion local restringida a vecindarios.
- No hay position embeddings 2D-aware (probaron varias formas y no aportan).
- No hay augmentation distinta de la basica (aunque importa para training pequeno).

La elegancia esta en la **maxima simplicidad**: probar si la arquitectura NLP funciona out-of-the-box.

---

## 3. Pipeline Detallado con Matematica

### 3.1. Patch extraction

Imagen de entrada: $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$.

Se corta en $N = \dfrac{HW}{P^2}$ parches no superpuestos de tamano $P \times P \times C$. Cada parche se aplana a un vector de $P^2 C$ dimensiones:

$$\mathbf{x}_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$$

Para una imagen tipica de $224 \times 224 \times 3$ con $P = 16$: $N = 196$ parches, cada uno de $768$ dimensiones planas.

### 3.2. Patch embedding

Proyeccion lineal aprendida $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$ que mapea cada parche al espacio latente del Transformer (dimension $D$):

$$\mathbf{x}_p^i E \in \mathbb{R}^{D}, \quad i = 1, \ldots, N$$

Implementacion eficiente: equivale exactamente a aplicar **una convolucion 2D con kernel $P \times P$ y stride $P$**. El paper lo describe como proyeccion lineal para enfatizar que **no hay convolucion en el resto del modelo**.

### 3.3. [class] token + position embedding

Se prepende un token aprendible $\mathbf{x}_{\text{class}} \in \mathbb{R}^{D}$ (analogo a [CLS] en BERT) y se suma un position embedding aprendido $E_{pos} \in \mathbb{R}^{(N+1) \times D}$:

$$\mathbf{z}_0 = [\mathbf{x}_{\text{class}}; \mathbf{x}_p^1 E; \mathbf{x}_p^2 E; \ldots; \mathbf{x}_p^N E] + E_{pos}$$

Esto es la **Ecuacion 1** del paper.

Decision sutil: el position embedding es **1D**, indexando los parches en orden raster (row-major), aunque la imagen sea 2D. Probaron alternativas (2D, sinusoidales, relativas) y no aportan ganancia significativa -- el modelo aprende la estructura 2D solo.

### 3.4. Bloques del Transformer encoder

Para $\ell = 1, \ldots, L$:

$$\mathbf{z}'_\ell = \text{MSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1} \quad (\text{Ec. 2})$$

$$\mathbf{z}_\ell = \text{MLP}(\text{LN}(\mathbf{z}'_\ell)) + \mathbf{z}'_\ell \quad (\text{Ec. 3})$$

Donde:
- $\text{MSA}$ = Multi-head Self-Attention (Vaswani 2017, Apendice A del paper).
- $\text{LN}$ = Layer Normalization, aplicada **antes** de cada bloque (pre-LN, mas estable que post-LN del Transformer original).
- $\text{MLP}$ = dos capas lineales con activacion GELU en el medio.
- Conexiones residuales sumadas despues.

### 3.5. Multi-head self-attention (recordatorio)

Para cada token $\mathbf{z} \in \mathbb{R}^{N \times D}$:

$$[\mathbf{q}, \mathbf{k}, \mathbf{v}] = \mathbf{z} U_{qkv}, \quad U_{qkv} \in \mathbb{R}^{D \times 3D_h}$$

$$A = \text{softmax}\left(\frac{\mathbf{q} \mathbf{k}^\top}{\sqrt{D_h}}\right), \quad A \in \mathbb{R}^{N \times N}$$

$$\text{SA}(\mathbf{z}) = A \mathbf{v}$$

Y para $k$ heads en paralelo:

$$\text{MSA}(\mathbf{z}) = [\text{SA}_1(\mathbf{z}); \ldots; \text{SA}_k(\mathbf{z})] U_{msa}$$

### 3.6. Cabezal de clasificacion

Se toma el estado final del token [class]:

$$\mathbf{y} = \text{LN}(\mathbf{z}_L^0) \quad (\text{Ec. 4})$$

Y se pasa por un MLP head (2 capas con tanh + linear) durante pre-training, o un solo linear durante fine-tuning:

$$\hat{p} = \text{softmax}(\mathbf{y} W_{head} + b_{head})$$

---

## 4. Variantes del Modelo (Tabla 1)

| Modelo | Layers $L$ | Hidden $D$ | Heads | MLP size | Params |
|---|---|---|---|---|---|
| ViT-Base (B) | 12 | 768 | 12 | 3072 | 86M |
| ViT-Large (L) | 24 | 1024 | 16 | 4096 | 307M |
| ViT-Huge (H) | 32 | 1280 | 16 | 5120 | 632M |

Las variantes Base y Large estan tomadas directamente de BERT. Huge es nueva en este paper.

### 4.1. Notacion ViT-X/Y

`ViT-L/16` significa Large con patches de **16x16 pixeles**. La longitud de secuencia es **inversamente proporcional al cuadrado del patch size**:

- ViT-B/32: $N = 7 \times 7 = 49$ tokens. Mas rapido, menos preciso.
- ViT-B/16: $N = 14 \times 14 = 196$ tokens. Estandar.
- ViT-H/14: $N = 16 \times 16 = 256$ tokens. Mas costoso, mas preciso.

Trade-off claro: parches mas pequenos = mas tokens = atencion mas costosa $O(N^2)$, pero mas detalle fino.

---

## 5. Pre-training Datasets

| Dataset | Imagenes | Clases | Notas |
|---|---|---|---|
| ImageNet-1k (ILSVRC-2012) | 1.3M | 1000 | El benchmark clasico |
| ImageNet-21k | 14M | 21k | Superset publico de ImageNet |
| JFT-300M | 303M | 18k | Dataset interno de Google, no publico |

Los autores **deduplican** los datasets de pre-training contra los test sets de las tareas downstream para evitar leakage.

---

## 6. Resultados Experimentales (Tabla 2)

### 6.1. Comparacion con SOTA

| | Ours-JFT (ViT-H/14) | Ours-JFT (ViT-L/16) | Ours-I21k (ViT-L/16) | BiT-L (R152x4) | Noisy Student (EffNet-L2) |
|---|---|---|---|---|---|
| **ImageNet** | **88.55** ± 0.04 | 87.76 ± 0.03 | 85.30 ± 0.02 | 87.54 ± 0.02 | 88.4/88.5 |
| **ImageNet ReaL** | **90.72** ± 0.05 | 90.54 ± 0.03 | 88.62 ± 0.05 | 90.54 | 90.55 |
| **CIFAR-10** | **99.50** ± 0.06 | 99.42 ± 0.03 | 99.15 ± 0.03 | 99.37 ± 0.06 | -- |
| **CIFAR-100** | **94.55** ± 0.04 | 93.90 ± 0.05 | 93.25 ± 0.05 | 93.51 ± 0.08 | -- |
| **Oxford-IIIT Pets** | **97.56** ± 0.03 | 97.32 ± 0.11 | 94.67 ± 0.15 | 96.62 ± 0.23 | -- |
| **Oxford Flowers-102** | 99.68 ± 0.02 | **99.74** ± 0.00 | 99.61 ± 0.02 | 99.63 ± 0.03 | -- |
| **VTAB (19 tasks)** | **77.63** ± 0.23 | 76.28 ± 0.46 | 72.72 ± 0.21 | 76.29 ± 1.70 | -- |
| **TPUv3-core-days** | 2.5k | 0.68k | 0.23k | 9.9k | 12.3k |

Observaciones clave:

- ViT-H/14 fija nuevo SOTA en **6 de 7 benchmarks**.
- ViT-L/16 pre-entrenado en JFT-300M usa **~14x menos computo** que Noisy Student y **~14x menos** que BiT-L, y supera a BiT en todas las tareas.
- ViT-L/16 con solo ImageNet-21k (publico, 14M imagenes) ya supera a BiT-L en CIFAR-10/100 y Pets, con **~43x menos computo** (0.23k vs 9.9k TPUv3-core-days).

### 6.2. Eficiencia computacional vs ResNet (Figura 5)

A presupuesto de computo igual, ViT necesita **2-4x menos exaFLOPs** que ResNet (BiT) para alcanzar la misma transfer accuracy promedio. La pendiente de la curva no satura en el rango probado, sugiriendo que mas escala mejoraria mas.

---

## 7. Trade-off Datos vs Inductive Bias

### 7.1. Figura 3 — Pre-training dataset size

Cuando se pre-entrena en datasets de tamano creciente (ImageNet → ImageNet-21k → JFT-300M):

```text
Accuracy en ImageNet (transfer):

90 │                                          ●── ViT-H/14 (JFT)
   │                                       ●── ViT-L/16
85 │                          ●── BiT (ResNet)
   │                       ●     ●── ViT-B/16
80 │            ●        ●
   │   ●── BiT  ●── ViT-L (cae con poco data!)
75 │      ●
   │── ViT-L/32, ViT-B/32 (peor con poco data)
70 │
   └────────────┬────────────────┬────────────┬─→ Pre-training size
            ImageNet         I21k          JFT-300M
            (1.3M)          (14M)         (303M)
```

Patron critico:
- **Con 1.3M imagenes**: ViT-Large esta **por debajo** de BiT-ResNet. Inductive biases ganan.
- **Con 14M imagenes**: ViT empieza a competir.
- **Con 303M imagenes**: ViT-H supera a todo.

### 7.2. Figura 4 — Few-shot accuracy vs subsets de JFT

Los autores entrenan los mismos modelos en subsets aleatorios de JFT (9M, 30M, 90M, 300M) sin regularizacion adicional para medir el **bias intrinseco** del modelo:

- ViT-B/32 vs ResNet50: ResNet gana en 9M, empata en 30M, ViT gana en 90M+.
- ViT-L/16 vs ResNet152x2 (BiT): ResNet gana hasta ~30M, ViT alcanza en 90M, supera en 300M.

**Leccion**: la convolucion es un **prior util** que ahorra datos. Cuando hay datos suficientes, el prior se convierte en un **techo** y el modelo mas flexible (ViT) gana.

---

## 8. Analisis Interno (Figura 7) — Que aprende ViT

### 8.1. Filtros del patch embedding (Fig 7 izquierda)

Los **primeros 28 componentes principales** de la matriz $E$ (proyeccion lineal del parche) revelan filtros que **se parecen a filtros de Gabor**: detectores de bordes, orientaciones, frecuencias espaciales. **Lo mismo que aprende la primera capa de una CNN.** Sin imponer convoluciones, el modelo descubre filtros tipo conv como representacion de bajo nivel.

### 8.2. Position embeddings (Fig 7 centro)

Pese a que se inicializan **aleatoriamente** y son **1D** (sin estructura 2D explicita), tras el training:

- Parches espacialmente cercanos tienen embeddings con alta similitud coseno.
- Aparece estructura **fila/columna** clara: parches en la misma fila o columna se asemejan.
- Para grids grandes emerge incluso un patron casi sinusoidal.

**Conclusion**: el modelo **aprende la topologia 2D de la imagen solo del gradiente de la tarea**, sin necesidad de inyectarsela como bias arquitectonico. Esto explica por que embeddings 2D-aware no aportan: ViT ya descubre lo que necesita.

### 8.3. Mean attention distance (Fig 7 derecha)

Para cada head en cada capa, los autores computan la **distancia euclidiana media en pixeles** entre un query y los keys ponderados por su atencion. Es analogo al **receptive field** de una CNN.

```text
Mean attention distance (pixeles):

120 │                              ●●●
    │                       ●● ●●●  ●  ← capas profundas: todos los heads
 80 │                ●● ●● ●         ●    atienden globalmente
    │       ●●  ●●●● ●●               
 40 │   ●● ●●  ●     ●               ← mezcla local/global desde temprano
    │● ●     ●                       
  0 │                                ← algunos heads "locales" (CNN-like)
    └──────────────────────────────→
        0    5    10   15   20   Layer
```

Hallazgos:

- En las **capas tempranas** (L=0-5) coexisten heads con atencion **muy local** (10-20 pixeles, similar a un kernel 3x3 de CNN) y heads con atencion **muy global** (80+ pixeles, abarcando toda la imagen).
- En las **capas profundas** (L=15+) **todos los heads** atienden globalmente.
- En el **modelo hibrido** (ResNet + Transformer), la atencion local en capas bajas casi desaparece, sugiriendo que la CNN ya hace ese trabajo.

**Comparacion con CNN**: una CNN solo puede ver un campo receptivo limitado en sus primeras capas (e.g. 3x3, 7x7). ViT puede atender globalmente desde la primera capa. Esto es un **superpoder** de ViT con datos suficientes.

---

## 9. Hybrid Architecture

Como ablation, el paper prueba **ResNet + Transformer**: usar feature maps de un ResNet (intermedios o finales) como secuencia de entrada del Transformer, en lugar de parches crudos.

Resultados (Figura 5):

- En **modelos pequeños** (poca compute): hibrido > ViT puro.
- En **modelos grandes** (mucha compute): hibrido = ViT puro (la diferencia desaparece).

Interpretacion: las features de la CNN sirven como un **shortcut de aprendizaje** cuando el Transformer es chico, pero un Transformer suficientemente grande aprende lo mismo solo. Es una version del trade-off datos vs bias aplicada al espacio de modelos.

---

## 10. Self-Supervision (preliminar, Sec 4.6)

Los autores hacen un **experimento preliminar** de pre-training auto-supervisado: **masked patch prediction**, mimicando MLM de BERT.

- Corromper 50% de patches (80% reemplazo por embedding [mask] aprendido, 10% otro patch random, 10% sin cambio).
- Predecir el **3-bit mean color** del patch original (512 colores).
- ViT-B/16 pre-entrenado asi alcanza **79.9% en ImageNet**: +2% sobre training from scratch, pero **-4% bajo el supervised pre-training en JFT**.

Este experimento es la **semilla directa de MAE** (He et al. 2022), que perfeccionaria la idea con masking del 75%, decoder asimetrico, y reconstruccion de pixeles, llegando a 87.8% sin etiquetas masivas.

---

## 11. Sucesores y Conexiones

### 11.1. Familia ViT (2021-2024)

```text
2017  Transformer (Vaswani)
        │
2020  ─ ViT (Dosovitskiy) ─────────── este paper
        │
2021  ├─ DeiT (Touvron) — distillation, sin JFT
        ├─ Swin (Liu) — ventanas con shift, jerarquia
        ├─ CaiT (Touvron) — class-attention, LayerScale
        ├─ T2T-ViT — tokenization mas elaborada
        ├─ PiT, CrossViT, ViT-Adapter, ...
        │
2022  ├─ MAE (He) — self-supervised, masked autoencoder
        ├─ BEiT (Bao) — discrete VAE tokens (BERT-style)
        ├─ ConvNeXt (Liu) — modernizar CNN para competir
        ├─ DINO (Caron) — self-distillation, features emergentes
        │
2023  ├─ DINOv2 (Oquab) — features universales
        ├─ SAM (Kirillov) — segmentacion universal sobre ViT-H
        ├─ EVA — escalar ViT a 1B+ params
        │
2024  └─ Registers (Darcet) — fix artefactos de atencion
```

### 11.2. Aplicaciones derivadas

- **Multimodal**: CLIP (Radford 2021) usa ViT como encoder visual + Transformer textual con contrastive loss → revolucion zero-shot.
- **Generativa**: Stable Diffusion, DALL-E 2/3, Imagen, Gemini Imagen — todas usan ViT-based encoders/UNets con atencion.
- **Detection/Segmentation**: DETR (que ViT cita), Mask2Former, SAM — Transformers ya no son solo clasificacion.
- **Video**: ViViT, Video Swin, TimeSformer — extender patches a tubelets espacio-temporales.
- **3D**: Point Transformer, NeRF + Transformer, Gaussian Splatting con atencion.

### 11.3. Sucesion arquitectonica

ViT no destrono completamente a las CNN -- ConvNeXt (2022) demostro que CNNs modernizadas (LN, GELU, kernels grandes 7x7) compiten parejo. La leccion mas profunda fue: **muchas decisiones arquitectonicas son intercambiables si la receta de training es buena**. ViT, ConvNeXt, Swin, MaxViT convergen en ~85% ImageNet con design choices distintos.

---

## 12. Lecciones Transferibles

### 12.1. Inductive bias como trade-off

```text
Mas inductive bias                  Menos inductive bias
        │                                       │
        │  CNN              Transformer  ViT    │
        │  ←────────────────────────────────→   │
        │                                       │
        │  Aprende rapido    Aprende lento      │
        │  con poco data     pero llega         │
        │  Techo limitado    mas alto con       │
        │                    mucha data         │
```

Esta dicotomia se generaliza fuera de vision:
- **Robotica**: priors fisicos (rigid body) vs RL puro de pixels.
- **Quimica**: graph neural networks (estructura molecular como bias) vs Transformers sobre SMILES.
- **Audio**: Mel-spectrogram + CNN vs Transformer sobre raw waveform.

La regla empirica de la era 2020+: **si tienes mucha data y compute, prefiere el modelo con menos bias**.

### 12.2. Simplicidad escala mejor

ViT no introduce nada arquitectonicamente nuevo respecto a Vaswani 2017. Su contribucion es **mostrar que la arquitectura mas simple posible funciona si se escala bien**. Esto resuena con la "bitter lesson" de Sutton: a la larga, los metodos generales que escalan con compute ganan a los metodos especializados con priors humanos.

### 12.3. Reutilizacion arquitectonica

Antes de ViT, NLP y vision tenian arquitecturas radicalmente distintas. Despues de ViT, **una sola arquitectura sirve para texto, imagen, audio, video, codigo, proteinas**. Esto:

- Permite **modelos multimodales unificados** (Gemini, GPT-4, Claude).
- Reduce el costo de I+D: una mejora de Transformer (FlashAttention, GQA, KV cache) se beneficia inmediatamente en todos los dominios.
- Concentra el progreso: optimizar Transformers se vuelve la palanca dominante de IA.

### 12.4. Pre-training masivo es la palanca

ViT-L con ImageNet (1.3M) pierde con BiT. ViT-L con JFT-300M **gana**. **Mismo modelo, mismo training loop, mismo numero de parametros**. La unica diferencia es el dataset de pre-training. Esto consolido la era de los **foundation models**: el valor esta cada vez mas en los datos y la escala, no en arquitecturas mas inteligentes.

---

## 13. Resumen en Una Pagina

```text
PROBLEMA:  La vision por computador estaba dominada por CNN.
           ¿Puede un Transformer puro competir, sin convoluciones?

SOLUCION:  Tratar la imagen como una secuencia de parches 16x16.
           Aplicar el Transformer encoder de Vaswani 2017 sin cambios.
           Pre-entrenar a gran escala (JFT-300M) y fine-tune.

RECETA EXACTA:
  1. Imagen 224x224x3  →  196 parches de 16x16x3
  2. Aplanar  →  vectores de 768 dim
  3. Proyeccion lineal aprendida  →  D dim
  4. Prepend [class] token + sum positional embedding 1D
  5. L bloques pre-LN: MSA + MLP, residuales
  6. MLP head sobre el [class] token final

VARIANTES: ViT-B (86M) / L (307M) / H (632M) params
           x patch size /16, /32, /14

RESULTADOS:
  - ViT-H/14 + JFT-300M: 88.55% ImageNet (SOTA 2021)
  - 4-14x menos compute que BiT/Noisy Student
  - SOTA en CIFAR-10 (99.50), CIFAR-100 (94.55), Pets (97.56), VTAB (77.63)

DESCUBRIMIENTO CONCEPTUAL:
  Inductive biases son trade-off:
    - CNN gana con poco data
    - ViT gana con mucho data (cruce ~100M imagenes)

ANALISIS INTERNO:
  - Patch embedding aprende filtros tipo Gabor (como CNN)
  - Position embedding 1D aprende estructura 2D solo
  - Atencion: heads locales y globales coexisten desde capas tempranas
    (a diferencia de CNN, que es estrictamente local en capas bajas)

LIMITACIONES:
  - Necesita pre-training masivo
  - Costo cuadratico en numero de parches
  - Sin jerarquia espacial (problema para detection/segmentation densa)

IMPACTO:
  - DeiT, Swin, MAE, BEiT, DINO, ConvNeXt
  - CLIP, Stable Diffusion, SAM, GPT-4V, Gemini
  - Unifico vision con NLP en una sola arquitectura
  - Hizo posibles los modelos multimodales actuales

LECCION:
  La arquitectura mas simple que escala con compute y datos
  termina ganando. ViT es la "bitter lesson" aplicada a vision.
```
