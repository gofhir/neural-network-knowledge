---
title: "Dataset Raccoon y DataLoader custom"
weight: 40
math: true
---

Para hacer fine-tuning necesitamos darle al modelo **datos en el formato que espera**. Faster R-CNN durante entrenamiento consume tuplas `(imagen, target)` donde `target` es un dict con keys especificas. Esta pagina cubre el dataset Raccoon, su formato de anotaciones, la clase `RaccoonDataLoader` que parsea las anotaciones, y la funcion `collate` que arma los batches.

## El dataset Raccoon

Repositorio: <https://github.com/bing0037/Raccoon_dataset>

```text
Raccoon_dataset/
├── images/                       (200 archivos .jpg)
├── annotations/                  (200 archivos .xml, formato PASCAL VOC)
├── raccoon_train_data.txt        (160 lineas - train split)
├── raccoon_test_data.txt         (40 lineas - val split)
└── README.md
```

### Formato del archivo .txt

Cada linea tiene **campos separados por espacios**:

```text
<ruta_imagen> <caja_1> <caja_2> ... <caja_N>
```

Donde cada `<caja>` es:

```text
x_min,y_min,x_max,y_max,label
```

(valores separados por **comas**, sin espacios)

### Ejemplo concreto

```text
Raccoon_dataset/images/raccoon-41.jpg 211,78,530,468,0
                                      ↑   ↑  ↑   ↑   ↑
                                      x1  y1 x2  y2  label
```

Una sola caja (un mapache) en (211, 78) a (530, 468). Label `0` (siempre 0 porque solo hay una clase real).

### Imagen con multiples mapaches

```text
Raccoon_dataset/images/raccoon-168.jpg 98,88,374,303,0 173,1,471,309,0
```

Dos cajas (dos mapaches) concatenadas.

### Distribucion del split de train

| Cajas por imagen | Cantidad de imagenes |
| --- | --- |
| 1 mapache | 150 |
| 2 mapaches | 9 |
| 3 mapaches | 1 |
| **Total** | **160 imagenes, 171 mapaches** |

El 94% de las imagenes tienen un solo mapache, pero algunas tienen 2-3. **El DataLoader debe manejar este numero variable**.

## El detalle critico: label `0` -> label `1`

⚠️ El dataset usa **label = 0 para mapache**. Pero en la convencion de `torchvision.models.detection.FasterRCNN`:

- `label = 0` -> **background**
- `label = 1, 2, ..., N-1` -> clases reales

Si pasaramos directamente `label=0` al modelo, le estariamos diciendo que el mapache es **fondo** -> el modelo aprenderia "no hay objeto donde hay mapache" -> entrenamiento roto.

Solucion en el lab: en `__getitem__` del DataLoader, **sumar 1 al label**. Veras este `+1` en una linea especifica del codigo.

## La clase `RaccoonDataLoader`

```python
from torch.utils.data import Dataset

class RaccoonDataLoader(Dataset):
    def __init__(self, data, device):
        self.img_list = []
        self.annotation_list = []
        self.device = device

        file1 = open(data, 'r')
        Lines = file1.readlines()
        for line in Lines:
            tmp_line = line.strip().split()
            self.img_list.append(tmp_line[0])
            img_annotations = []
            for i in range(1, len(tmp_line)):
                tmp_annotation = [int(j) for j in tmp_line[i].split(',')]
                img_annotations.append(tmp_annotation)
            self.annotation_list.append(img_annotations)
        file1.close()

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = Image.open(img_path).convert("RGB")
        img = T.ToTensor()(img)

        annotations_array = np.array(self.annotation_list[index])
        boxes = annotations_array[:, :4]
        labels = annotations_array[:, 4] + 1   # ← el +1 critico

        target = {}
        target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.tensor(labels, dtype=torch.int64)
        target['image_id'] = torch.tensor([index])
        return img, target

    def __len__(self):
        return len(self.img_list)
```

⚠️ **Nota nomenclatural:** la clase se llama `RaccoonDataLoader` pero **tecnicamente es un `Dataset`**, no un `DataLoader`. El `DataLoader` real es `torch.utils.data.DataLoader` que se construye despues.

### El contrato `Dataset` de PyTorch

`torch.utils.data.Dataset` es una clase abstracta. Para crear tu propio dataset, **heredas de ella** y obligatoriamente implementas:

- `__getitem__(self, index)`: retorna UN elemento del dataset dado su indice.
- `__len__(self)`: retorna el tamano total del dataset.

PyTorch luego envuelve tu Dataset en un `DataLoader` que se encarga de: batching, shuffling, multi-processing para cargar datos en paralelo, pin memory para transferencias CPU->GPU.

### `__init__` — parseo del .txt

Parsea linea por linea:

```python
'Raccoon_dataset/images/raccoon-41.jpg 211,78,530,468,0'.split()
# → ['Raccoon_dataset/images/raccoon-41.jpg', '211,78,530,468,0']
```

Despues:

```python
'211,78,530,468,0'.split(',')
# → ['211', '78', '530', '468', '0']
# → [211, 78, 530, 468, 0]  (despues del int())
```

Resultado: `self.img_list` con rutas, `self.annotation_list` con triple-nesting `[imagenes [cajas [coordenadas]]]`.

**Solo carga anotaciones**, no las imagenes. Las imagenes se cargan **on-demand** en `__getitem__`. Esto evita ocupar cientos de MB de RAM con imagenes que tal vez no se usen.

### `__getitem__` — generar un elemento

Cinco pasos:

1. **Cargar imagen** con PIL forzando RGB. El `.convert("RGB")` es defensivo: si la imagen es PNG con alpha (RGBA) o grayscale (L), la convierte a RGB.
2. **Convertir a tensor** con `T.ToTensor()`: permuta a (3, H, W), convierte a float32, normaliza a [0, 1].
3. **Extraer cajas** con numpy slicing: `boxes = annotations_array[:, :4]`.
4. **Extraer labels con offset +1**: `labels = annotations_array[:, 4] + 1`.
5. **Armar target dict** con los tipos correctos:
   - `'boxes'`: float32, shape (K, 4).
   - `'labels'`: int64, shape (K,).
   - `'image_id'`: cualquiera, identificador para tracking.

Retorna `(img, target)`.

⚠️ Para **tu propio dataset con clases distintas**: si las labels empiezan en 0 hay que sumar 1; si ya empiezan en 1 no hay que sumar nada. Esta linea es una de las que hay que ajustar y es **respuesta a la pregunta 1 de la tarea final** del notebook.

### `__len__`

```python
def __len__(self):
    return len(self.img_list)
```

Devuelve 160 para train, 40 para val.

## La funcion `collate`

```python
def collate(batch):
    return tuple(zip(*batch))
```

Solo una linea, pero conceptualmente clave.

### El problema que resuelve

Cuando un `DataLoader` agrupa elementos en un batch, por defecto **intenta apilarlos en tensores con `torch.stack`**. Eso falla aqui por dos razones:

1. **Las imagenes tienen tamanos distintos** -> `stack` requiere shapes identicos -> `RuntimeError`.
2. **Los targets son dicts** con tensores de tamanos variables (1 caja, 2 cajas, 3 cajas) -> `stack` no sabe que hacer.

### La solucion

Le decimos al DataLoader: "no apiles nada, solo agrupame los elementos en listas". `zip(*batch)` hace exactamente eso:

```python
batch = [(img1, t1), (img2, t2), ..., (img6, t6)]
zip(*batch) → [(img1, img2, ..., img6), (t1, t2, ..., t6)]
tuple(zip(*batch)) → ((img1, img2, ..., img6), (t1, t2, ..., t6))
```

Es una "transposicion" de matriz: lista de tuplas -> tupla de listas.

### Por que Faster R-CNN acepta este formato

El `forward()` de `GeneralizedRCNN` esta disenado para aceptar **listas de imagenes y listas de targets**, no tensores apilados. La capa `transform` interna hace el padding necesario para procesarlas en batch.

```python
def forward(self, images, targets=None):
    # images: List[Tensor], cada uno (3, Hi, Wi) con tamanos distintos
    # targets: Optional[List[Dict[str, Tensor]]]
    ...
```

## Instanciacion de Datasets y DataLoaders

```python
train_data = RaccoonDataLoader('Raccoon_dataset/raccoon_train_data.txt', device)
val_data   = RaccoonDataLoader('Raccoon_dataset/raccoon_test_data.txt', device)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=6, shuffle=True, collate_fn=collate)
val_loader   = torch.utils.data.DataLoader(val_data,   batch_size=6, shuffle=True, collate_fn=collate)

num_classes = 2
Category = list(range(num_classes - 1))   # [0]
```

| Variable | Valor | Significado |
| --- | --- | --- |
| `len(train_data)` | 160 | imagenes de train |
| `len(val_data)` | 40 | imagenes de val |
| `batch_size=6` | — | trade-off VRAM vs paralelismo (T4 de Colab aguanta 6) |
| `num_classes=2` | — | 1 raccoon + 1 background |
| `Category=[0]` | — | indices de las clases reales (sin contar background) |

⚠️ Curiosidad: `val_loader` tambien tiene `shuffle=True`. Es inusual (la validacion deberia ser determinista) pero no afecta el accuracy final.

### batch_size=6 — por que no 32 o 64

Faster R-CNN consume **muchisima memoria GPU** por imagen (las imagenes son grandes con lado min 800 px, y el modelo tiene 5 niveles de feature maps). En una T4 de Colab (16 GB) hay que bajar:

| batch_size | VRAM aprox | Quien aguanta |
| --- | --- | --- |
| 1-2 | <6 GB | Cualquier GPU |
| 4-6 | 8-12 GB | T4, V100 |
| 16+ | 24 GB+ | A100, multi-GPU |

Si tu GPU no aguanta `batch_size=6`, baja a 4 o 2. El entrenamiento sera mas lento pero funcional.

## Verificacion visual

```python
img, target = train_data.__getitem__(13)
img = img.detach().cpu().numpy()
img = np.moveaxis(img, 0, 2)
img = (img * 255).astype(np.uint8)
boxes = target['boxes']
for i in range(len(target['boxes'])):
    cv2.rectangle(img, (int(boxes[i][0]), int(boxes[i][1])),
                  (int(boxes[i][2]), int(boxes[i][3])), color=(0, 255, 0), thickness=3)
plt.imshow(img)
plt.xticks([]); plt.yticks([])
plt.show()
print('target:', target)
```

Saca una imagen del dataset (indice 13) y dibuja sus cajas ground-truth.

![Sample del dataset Raccoon con caja ground-truth](/laboratorios/lab-15/raccoon-dataset-sample.jpg)

La caja verde encuadra precisamente al mapache. El `target` impreso al pie de la imagen muestra:

```python
target: {
    'boxes': tensor([[211., 78., 530., 468.]]),
    'labels': tensor([1]),    # ← raccoon es clase 1, no 0
    'image_id': tensor([13])
}
```

Sirve para confirmar empiricamente que:

1. ✅ El parsing del .txt produce coordenadas correctas (la caja encuadra al mapache).
2. ✅ El offset de label funciono (`'labels': tensor([1])`, no `[0]`).
3. ✅ El formato del target es el que espera Faster R-CNN.

Si esta visualizacion falla, no tiene sentido lanzar entrenamiento de 10 minutos para que falle con un error criptico. Esta celda corta ese ciclo.

## Sigue: reemplazar el clasificador

Con los datos listos, el siguiente paso es modificar el modelo. Ver [Reemplazo del clasificador](fine-tuning-setup).
