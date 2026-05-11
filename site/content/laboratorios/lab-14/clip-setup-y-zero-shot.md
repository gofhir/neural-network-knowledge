---
title: "CLIP - setup y zero-shot classification"
weight: 70
math: true
---

La Parte 2 del lab **cambia completamente de paradigma**: en Parte 1 inspeccionamos el mecanismo interno de un modelo de **una sola modalidad** (texto: BETO + bertviz). Ahora pasamos a **CLIP** (Contrastive Language-Image Pre-training, Radford et al. OpenAI 2021), un modelo **multimodal** que vive en un espacio compartido entre texto e imagen. Y el foco se invierte: ya no abrimos el capo, ahora **observamos comportamiento** — clasificacion zero-shot sobre datasets que el modelo nunca vio en entrenamiento.

La leccion clave de esta primera mitad: **embeddings imagen-texto compartidos permiten clasificacion sin fine-tuning**. CLIP no fue entrenado en Food101 ni Stanford Cars — pero puede clasificarlos casi tan bien como modelos especializados, simplemente preguntando "¿que tan parecida es esta imagen a la frase 'A photo of pizza, a type of food.'?"

## ¿Que es CLIP?

CLIP fue entrenado con **400 millones de pares image-text** scrapeados de internet. La arquitectura tiene **dos encoders independientes** que producen embeddings en el **mismo espacio**:

```
       Imagen ──────[ViT-B/32 o ResNet]────► image_features  (512-dim)
                                                  │
                                                  ▼
                                            cosine_sim
                                                  ▲
                                                  │
       Texto ───────[Transformer]──────────► text_features   (512-dim)
```

Ambos encoders producen vectores en el **mismo espacio de 512 dimensiones**. La alineacion se logra durante entrenamiento con **contrastive loss** (InfoNCE): los pares image-text reales se acercan, los pares falsos del mismo batch se alejan.

### La magia: zero-shot classification

CLIP **nunca fue entrenado** en Food101 ni en Stanford Cars. Pero puede clasificarlos asi:

1. Para cada clase del dataset, construye una query: `"A photo of pizza, a type of food."`
2. Codifica las N queries con el text encoder → N vectores de 512 dim
3. Codifica una imagen con el image encoder → 1 vector de 512 dim
4. Calcula la **similitud coseno** entre la imagen y las N queries
5. La query con mayor similitud → prediccion

**No hay fine-tuning, no hay gradient updates, no hay loss function.** Es pura inferencia sobre un modelo pre-entrenado en una tarea completamente distinta.

## Setup del modelo

El notebook empieza instalando CLIP desde el repo de OpenAI *(parte 2, celda 7)*:

```python
!pip install ftfy regex tqdm
!pip install git+https://github.com/openai/CLIP.git
```

| Paquete | Para que |
| --- | --- |
| `ftfy` | Repara texto Unicode mal codificado. CLIP lo usa internamente para limpiar las captions de internet |
| `regex` | Version avanzada de `re` — soporta Unicode property escapes que el tokenizer de CLIP necesita |
| `tqdm` | Barra de progreso |
| `git+https://github.com/openai/CLIP.git` | Instala CLIP directamente desde GitHub porque OpenAI nunca hizo release oficial en PyPI |

> CLIP **NO** esta en HuggingFace `transformers` original (aunque hay forks como `openai/clip-vit-base-patch32`). La libreria original de OpenAI es minimalista — solo 4-5 archivos Python — y hace lo justo para cargar pesos y hacer inferencia. Para fine-tuning serio se usa `open_clip` o el wrapper de HuggingFace.

### Modelos CLIP disponibles

`clip.available_models()` *(parte 2, celda 11)* retorna:

```python
['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64',
 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
```

CLIP fue entrenado con **dos familias de image encoders**:

| Familia | Naming | Que es |
| --- | --- | --- |
| **ResNet** | `RN50`, `RN101`, `RN50x4`, `RN50x16`, `RN50x64` | ResNet clasica modificada (Conv → Attention Pool al final). El `xN` indica que el ancho/profundo esta escalado por un factor N |
| **Vision Transformer** | `ViT-B/32`, `ViT-B/16`, `ViT-L/14`, `ViT-L/14@336px` | Vision Transformer (Dosovitskiy et al. 2020). `B/32` = **B**ase + patches `32×32`. `L/14` = **L**arge con patches `14×14`. `@336px` indica resolucion de input |

### Cual usar: `ViT-B/32`

| Modelo | Params | Embed dim | Imagenes/s (V100) | ImageNet zero-shot Top-1 |
| --- | --- | --- | --- | --- |
| **ViT-B/32** (el del lab) | 151M | 512 | ~600 | 63.2% |
| ViT-B/16 | 150M | 512 | ~150 | 68.6% |
| ViT-L/14 | 428M | 768 | ~40 | 75.5% |
| ViT-L/14@336px | 428M | 768 | ~12 | 76.2% |

**ViT-B/32** es la opcion de compromiso: el peor ImageNet zero-shot accuracy de la familia ViT, pero **3-4× mas rapido** que ViT-B/16 y mucho mas liviano que las L. Para un lab educativo es ideal.

### Como procesa una imagen ViT-B/32

1. Redimensiona la imagen a `224×224`
2. La parte en **49 patches de 32×32** (porque `224/32 = 7`, asi que `7×7 = 49` patches)
3. Cada patch se aplana en un vector y se proyecta a 768-dim
4. Se le antepone un token especial `[CLS]` (similar a BERT, pero para vision)
5. Pasa por 12 capas de Transformer encoder
6. El embedding final del `[CLS]` se proyecta a **512 dimensiones** (espacio compartido con el text encoder)

### Cargar el modelo

```python
model, preprocess = clip.load("ViT-B/32")
model.to(device).eval()
```

Devuelve **dos cosas**: el modelo y un `preprocess` (una transformacion torchvision lista para aplicar a PIL Images). En el caso del notebook ejecutado:

```
Device: cuda
CUDA disponible: True
GPU: Tesla T4

Numero de parametros: 151,277,313
Resolucion de Entrada: 224
Tamano del contexto: 77
Tamano del vocabulario: 49,408
```

Los hyperparametros importantes:

- **`input_resolution = 224`** — todas las imagenes se redimensionan a `224×224`
- **`context_length = 77`** — secuencias de hasta 77 tokens (incluyendo `<sot>` y `<eot>`). En la practica son **~75 tokens de contenido util**
- **`vocab_size = 49408`** — vocabulario BPE compartido con GPT-2

### El pipeline de `preprocess`

```python
preprocess
```

Output:

```
Compose(
    Resize(size=224, interpolation=bicubic, max_size=None, antialias=True)
    CenterCrop(size=(224, 224))
    <function _convert_image_to_rgb at 0x...>
    ToTensor()
    Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
              std=(0.26862954, 0.26130258, 0.27577711))
)
```

| Transformacion | Que hace | Por que importa |
| --- | --- | --- |
| `Resize(224, bicubic)` | Reescala el lado mas corto a 224, manteniendo proporciones. Usa interpolacion bicubica | CLIP fue entrenado con bicubico — usar otra interpolacion degrada los resultados |
| `CenterCrop((224, 224))` | Toma el cuadrado central de 224×224 | Si lo importante esta en una esquina, lo descarta |
| `_convert_image_to_rgb` | Convierte RGBA, escala de grises o paletizada a RGB | CLIP solo acepta RGB |
| `ToTensor()` | PIL Image → tensor PyTorch (C, H, W) con valores [0, 1] | Formato estandar de PyTorch |
| `Normalize(mean, std)` | Resta `mean` y divide por `std` canal por canal | Los valores son las estadisticas del corpus **WIT** (400M pares image-text de CLIP), distintos de ImageNet |

> Si normalizas con stats de ImageNet en lugar de las de CLIP, **degradas los resultados**. Los pesos aprendieron a esperar inputs normalizados con WIT.

## Zero-shot sobre Food101: caso individual

Food101 (Bossard et al. 2014) tiene 101 clases de comidas distintas y 101,000 imagenes (1,000 por clase). CLIP **nunca lo vio** durante entrenamiento.

### Construir las queries con template

```python
query_template = 'A photo of {}, a type of food.'
queries = [query_template.format(label.replace('_', ' ')) for label in food_dataset.classes]
tokenized_queries = clip.tokenize(queries).to(device)
```

Las 101 clases ordenadas alfabeticamente *(parte 2, celda 23 output)*:

```
A photo of apple pie, a type of food.
A photo of baby back ribs, a type of food.
A photo of baklava, a type of food.
A photo of beef carpaccio, a type of food.
A photo of beef tartare, a type of food.
A photo of beet salad, a type of food.
A photo of beignets, a type of food.
A photo of bibimbap, a type of food.
A photo of bread pudding, a type of food.
A photo of breakfast burrito, a type of food.
...
```

### Por que este template especifico

Aqui entra **prompt engineering del lado del text encoder**. El paper de CLIP descubrio que el template afecta MUCHO el rendimiento zero-shot:

| Template | ImageNet zero-shot Top-1 |
| --- | --- |
| `"{label}"` (solo la palabra) | 63.4% |
| `"A photo of {label}"` | 64.2% |
| `"A photo of {label}, a type of {category}"` | 65.0%+ |

El template `"A photo of X, a type of food."` ayuda a CLIP a entender que:
1. Debe esperar **una foto** (no un dibujo, icono, logo)
2. **X es una comida** (no "the word pizza", sino "a pizza")
3. Es una **instancia individual**, no un concepto abstracto

> **Insight clave:** sin un template, "pizza" puede competir con "the word pizza" (imagenes de texto), "a pizza emoji", "a pizza box". El template **acota la distribucion semantica** hacia fotografia real de comida.

### Forward pass: codificar imagen y queries

```python
with torch.no_grad():
    image_features = model.encode_image(image_input).float()
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features = model.encode_text(tokenized_queries).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)
```

**Lineas clave:**

| Linea | Que hace |
| --- | --- |
| `model.encode_image(image_input)` | Forward pass por el Vision Transformer. Output: `(1, 512)` |
| `model.encode_text(tokenized_queries)` | Forward por el text Transformer. Output: `(101, 512)` |
| `/= norm(...)` | **L2-normaliza** los vectores → magnitud 1 |
| `image_features @ text_features.T` | Producto matricial → `(1, 101)`. Cada entrada es la similitud coseno |
| `100.0 * ...` | Multiplica por el **`logit_scale = 100`**. Amplifica las diferencias antes del softmax |
| `.softmax(dim=-1)` | Convierte en probabilidades — suman 1.0 sobre las 101 clases |

### Por que L2-normalizar

La formula de similitud coseno:

$$\text{cos}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \cdot \|\mathbf{b}\|}$$

Si **pre-normalizas** ambos vectores ($\|\mathbf{a}\| = \|\mathbf{b}\| = 1$), entonces:

$$\text{cos}(\mathbf{a}, \mathbf{b}) = \mathbf{a} \cdot \mathbf{b}$$

— el simple producto punto da el coseno directamente. Mas eficiente para hacer el calculo en batch con multiplicacion matricial.

### Por que el factor 100 (`logit_scale`)

Las similitudes coseno estan acotadas en `[-1, 1]`. Sin escalarlas, el softmax produce distribuciones casi uniformes (todas las probabilidades parecidas). CLIP **aprendio** este factor durante entrenamiento — el repo de OpenAI lo expone como `model.logit_scale.exp()` y para CLIP-ViT-B/32 vale **~100**. Es la **temperatura inversa** del softmax.

## Caso individual: hummus

Ejecutando el visualizador top-5 sobre una imagen aleatoria:

![Food101 caso individual hummus](/laboratorios/lab-14/food101-top5-hummus.png)

```
Predicciones Top-5

Clase           Probabilidad
hummus          0.3293   ← Top-1 CORRECTO
garlic_bread    0.1178
cheese_plate    0.0940
foie_gras       0.0719
deviled_eggs    0.0466

Clase verdadera: hummus
```

### Lo que esto demuestra

CLIP **NO** esta haciendo "match exact" como un clasificador convencional. Las 5 predicciones comparten **propiedades visuales abstractas**:

| Pred | Por que tiene sentido |
| --- | --- |
| **hummus** (33%) | El bowl con dip cremoso amarillo-verdoso es inconfundible. Top-1 muy seguro |
| **garlic_bread** (12%) | Las tostadas grandes a la izquierda parecen perfectamente pan de ajo |
| **cheese_plate** (9%) | El **formato del plato** — bowl con dip + tostadas + aceitunas + verduras — es exactamente la disposicion de un cheese plate clasico |
| **foie_gras** (7%) | El dip cremoso amarillento puede confundirse con foie gras (que tambien se sirve untable y con tostadas) |
| **deviled_eggs** (5%) | Huevos rellenos tambien tienen relleno cremoso amarillo |

### Insight conceptual

La representacion visual de CLIP captura **propiedades**, no solo identidades. Por eso funciona zero-shot — puede mapear cualquier descripcion textual ("una imagen amarilla con tostadas") a esas propiedades, sin haber visto explicitamente ese acoplamiento durante entrenamiento.

### Una distancia significativa entre Top-1 y Top-2

`0.33 / 0.12 ≈ 2.7×`. CLIP **no tiene dudas reales** en este caso — `hummus` esta claramente por encima del resto. Si la imagen hubiera sido ambigua, veriamos valores como `0.18 / 0.15 / 0.13 / 0.11 / 0.10` casi parejos.

### Las dimensiones que importan

*(Parte 2, celda 29 output)*:

```
Dim features de la imagen   : torch.Size([1, 512])
Dim features del texto      : torch.Size([101, 512])
Dim matriz de similaridades : torch.Size([1, 101])
```

| Tensor | Shape | Significado |
| --- | --- | --- |
| `image_features` | `(1, 512)` | **1** imagen, espacio compartido de **512 dim** |
| `text_features` | `(101, 512)` | **101** queries (una por clase), **512 dim** |
| `similarity` | `(1, 101)` | **1** imagen × **101** queries → 101 puntajes |

Estas dimensiones son **clave para la Actividad 3** que veremos en la siguiente seccion.

## Lo que viene en la siguiente seccion

Hasta aqui clasificamos **una** imagen. La siguiente seccion **evalua CLIP sobre las 25,250 imagenes del test set de Food101** (Top-1 y Top-5 accuracy) y resuelve la Actividad 3 — donde probamos templates distintos para entender el impacto del **prompt engineering** del lado del text encoder.
