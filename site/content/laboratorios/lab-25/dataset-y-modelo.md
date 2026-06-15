---
title: "Dataset multimodal y arquitectura del modelo"
weight: 2
math: true
---

> **Celdas 24-42 del notebook.** Después de plantear el problema, el lab construye sus tres piezas operativas: la clase `ContentRecommender` que carga un dataset *multimodal* (texto + imagen), el tokenizer BERT que digiere los comentarios, y los dos modelos que se van a comparar — el **multimodal completo** y un **baseline solo-imagen**. El objetivo final no es clasificar usuarios: es obtener un *vector descriptor* aprendido (la salida intermedia) para recomendar contenido afín. Clasificar usuarios es solo el *proxy task* que paga ese descriptor.

## Instalación: el gotcha de las versiones congeladas de 2020

El notebook abre fijando versiones exactas:

```python
!pip install transformers==3.5.1
!pip install torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

Esas son las versiones del entorno de Colab en 2020, cuando se escribió el material. **No instalan en el Colab actual** (Python 3.11/3.12, CUDA 12): `transformers==3.5.1` falla con `metadata-generation-failed` (su `setup.py` ya no es compatible con el resolvedor moderno de pip) y `torch==1.6.0+cu101` da `no matching distribution found` (no hay rueda para esa combinación de Python/CUDA).

**Fix.** Instalar `transformers` sin fijar versión y usar el `torch` ya preinstalado de Colab:

```python
!pip install -q transformers   # toma la 4.x actual
# torch: el que ya trae Colab, no se reinstala
```

> **Nota de compatibilidad — por qué el modelo corre sin parche.** En `transformers` 3.x, `BertModel(...)` devolvía una **tupla** `(last_hidden_state, pooler_output)`. En 4.x devuelve un objeto `BaseModelOutputWithPoolingAndCrossAttentions`. El código del modelo accede al *pooler* con `salida['pooler_output']` — y eso **sigue funcionando** en 4.x, porque `ModelOutput` es subclase de `OrderedDict` y admite indexado por clave. Por eso, pese al salto de tres versiones mayores, la rama de texto del modelo no necesita ninguna modificación.

## `ContentRecommender`: el Dataset multimodal

La clase `ContentRecommender` hereda de `torch.utils.data.Dataset` y, por cada *split*, carga **tres archivos**:

| Archivo | Contenido | Forma |
|---|---|---|
| `imag_*.txt` | Descriptores de imagen pre-extraídos | vectores 4096-d (uno por pin) |
| `text_*.txt` | Comentarios / textos asociados | string por pin |
| `train_users.txt` | Etiqueta de usuario (el *target*) | id de usuario por pin |

La clave es que **la imagen no se procesa en el Dataset**: ya llega como un vector de 4096 dimensiones (más sobre esto abajo). El texto, en cambio, sí se tokeniza al vuelo.

### `sub_set(num_classes)`: recortar y re-mapear

Por los límites de memoria y tiempo de Colab, el dataset completo no se usa entero. El método `sub_set(num_classes)`:

1. **Toma un subconjunto de usuarios al azar** (`num_classes` de ellos).
2. **Re-mapea sus ids originales a `0..num_classes-1`.** Esto es imprescindible: el clasificador necesita clases **contiguas** empezando en 0 (la *cross-entropy* indexa el vector de logits por la etiqueta). Si dejáramos los ids originales — dispersos, p. ej. 173, 4021, 88011 — la capa final tendría que tener tantas salidas como el id máximo.

### `__getitem__`: una muestra multimodal

Cada elemento devuelto es una tripleta multimodal:

```python
def __getitem__(self, idx):
    # 1) texto -> tokenizado con BERT (max_length=100)
    encoded = self.tokenizer(self.texts[idx], max_length=100,
                             padding="max_length", truncation=True,
                             return_tensors="pt")
    # 2) imagen -> vector 4096-d, NO se procesa (ya viene listo)
    image_feat = self.images[idx]
    # 3) target -> id de usuario re-mapeado
    target = self.users[idx]
    return input_ids, attention_mask, image_feat, target
```

> **Vestigio que confunde:** el parámetro `amount_triplet=3` sobrevive de una versión anterior del lab que usaba *triplet loss*. **No se usa** en esta versión — se puede ignorar.

## Instanciación (celda 31)

```python
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
trainset  = ContentRecommender(..., num_classes=10, tokenizer=tokenizer)
testset   = ContentRecommender(..., sub_classes=trainset.sub_classes, tokenizer=tokenizer)

train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
test_loader  = DataLoader(testset,  batch_size=64, shuffle=False)
```

Tres decisiones merecen comentario:

- **`bert-base-uncased` (inglés).** Los comentarios de Pinterest están en inglés, así que el tokenizer es el BERT en inglés *uncased* (sin distinguir mayúsculas). Ver el [fundamento BERT](/fundamentos/bert).
- **`num_classes=10`.** Solo 10 usuarios → el *proxy task* es una clasificación de 10 clases. Es deliberadamente pequeño para que entrene en Colab.
- **`testset` reusa `sub_classes=trainset.sub_classes`.** Esto es **crítico**: garantiza que el test contenga **los mismos 10 usuarios** que el train (con el mismo re-mapeo de ids). Si el test eligiera sus propios 10 usuarios al azar, las etiquetas no significarían lo mismo y la evaluación sería absurda.

`shuffle=True` solo en train (para romper el orden entre épocas); `False` en test (la evaluación es determinista).

**Output real verificado:**

```text
Train: 4000 imágenes | Test: 1000 imágenes
Features por imagen: 4096
Usuarios (clases): 10
→ 400 items/usuario en train, 100 en test
```

## D = 4096: las features fc7 de AlexNet

El número **4096** no es arbitrario: es el tamaño de la penúltima capa totalmente conectada (**fc7**) de **AlexNet** (Krizhevsky et al., 2012), la red que ganó ImageNet y abrió la era del deep learning en visión. Cada imagen de Pinterest se pasó por una AlexNet **pre-entrenada en ImageNet** y se guardó su activación fc7 como descriptor — un caso de manual de [transfer learning](/fundamentos/transfer-learning) por extracción de *features*.

> **Pista forense en el output:** los descriptores **tienen valores negativos**. Como ReLU recorta todo lo negativo a 0, esto confirma que las features se extrajeron **antes de la ReLU** (pre-activación de fc7), no después.

Ver [AlexNet (Krizhevsky 2012)](/papers/alexnet-krizhevsky-2012) y el [fundamento de redes convolucionales](/fundamentos/redes-convolucionales).

**Ejemplo real — usuario 1.** Sus comentarios incluyen `"audrey hepburn"`, `"pencil drawing victoria beckham"`, `"lovely rambling rose"`. Hay coherencia temática (retratos a lápiz de celebridades) pero **imperfecta** — la rosa rompe el patrón. Esa coherencia *parcial* es justamente la premisa del recomendador: si un usuario tuviera gustos perfectamente homogéneos el problema sería trivial; si fueran aleatorios, irresoluble.

## `ModelClass`: el modelo multimodal completo

Es el modelo que combina ambas modalidades. Dos ramas que convergen por **concatenación**:

```text
TEXTO   ──▶ BERT ──▶ pooler (768) ──▶ Linear(768, 64) ──┐
                                                          ├─▶ concat (128)
IMAGEN  ──▶ vector (4096) ─────────▶ Linear(4096, 64) ──┘
                                                          │
                              Linear(128, 32) ─▶ ReLU ─▶ Dropout(0.3)
                                                          │
                                            Linear(32, num_classes) ─▶ logits
```

- **Rama texto:** BERT produce el `pooler_output` de 768-d, proyectado a 64-d por una `Linear`.
- **Rama imagen:** el vector de 4096-d se proyecta a 64-d.
- **Fusión:** se **concatenan** los dos vectores de 64-d → 128-d. (Concatenar, no sumar: conserva ambas modalidades separadas y deja que la capa siguiente aprenda cómo mezclarlas.)
- **Cabeza:** `Linear(128, 32) → ReLU → Dropout(0.3) → Linear(32, num_classes)`.

> **El descriptor es el premio del proxy task.** El parámetro `features=True` hace que el `forward` devuelva el **vector intermedio de 32-d** y se salte el clasificador final. Ese vector de 32-d es el **descriptor aprendido** que después se usa para recomendar (buscando vecinos cercanos en ese espacio). Clasificar los 10 usuarios fue solo la excusa para *aprender* ese espacio de 32 dimensiones.

## `ModelClassImage`: el baseline solo-imagen

El baseline contra el cual se mide el multimodal. **Ignora el texto por completo:**

```text
IMAGEN ──▶ vector (4096) ──▶ Linear(4096, 64) ──▶ ReLU ──▶ Dropout
                                                            │
                                          Linear(64, num_classes) ─▶ logits
```

Recibe `input_ids` y `attention_mask` en su `forward` **solo para tener la misma firma** que `ModelClass` (y reutilizar el mismo `run_epoch`), pero no los usa. Con `features=True` devuelve el vector de **64-d**.

### Los dos modelos, lado a lado

| | `ModelClass` (multimodal) | `ModelClassImage` (baseline) |
|---|---|---|
| **Modalidades** | Texto (BERT) + imagen | Solo imagen |
| **Rama texto** | BERT(768) → Linear(768, 64) | — (ignorada) |
| **Rama imagen** | Linear(4096, 64) | Linear(4096, 64) |
| **Fusión** | Concatenación (128) | — |
| **Cabeza** | Linear(128,32)→ReLU→Drop→Linear(32, C) | ReLU→Drop→Linear(64, C) |
| **Descriptor (`features=True`)** | 32-d | 64-d |
| **Parámetros** | ~110M (domina BERT) | pocos miles |

**La hipótesis del lab:** el modelo multimodal, al sumar la señal del texto a la de la imagen, debe **superar** al baseline solo-imagen en la clasificación de usuarios — y, por extensión, producir descriptores más útiles para recomendar. Si no lo lograra, el texto no aportaría información y la complejidad de BERT estaría de más. Esa comparación es lo que mide la [siguiente página](entrenamiento).

---

**Anterior:** [Planteamiento del problema y datos](planteamiento-y-datos) · **Siguiente:** [Entrenamiento y resultados](entrenamiento)
