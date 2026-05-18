---
title: "Stanford 40 Actions Dataset + procesamiento masivo"
weight: 30
---

El experimento cuantitativo del lab compara OpenPifPaf vs. OpenPose como feature extractors para **clasificación de acciones** sobre el [Stanford 40 Actions Dataset](http://vision.stanford.edu/Datasets/40actions.html).

## Por qué clasificación downstream y no PCK

El profesor evita comparar los modelos por **calidad de keypoints directa** (PCK, AP, OKS) y en su lugar mide:

> **Accuracy de un MLP entrenado sobre los keypoints flatten** para clasificar acciones.

La distinción es **conceptualmente importante**:

- **Métricas directas (PCK, AP)**: miden qué tan precisos son los keypoints contra ground-truth anotado. Útil para benchmark académico.
- **Métricas downstream (MLP accuracy)**: miden qué tan **útiles** son los features para una tarea real. Útil para selección de modelo en producción.

El principio: **el mejor modelo de pose es el que produce features más útiles para tu tarea**, no el que dibuja esqueletos más bonitos. Esto se conecta con la regla industrial **"benchmark on your own data"**: ningún leaderboard genérico predice perfectamente cómo se comportará un modelo en tu downstream task.

## El dataset

| Atributo | Valor |
|---|---|
| **Imágenes** | 9,532 |
| **Clases** | 40 acciones humanas |
| **Anotación** | Etiqueta de acción + bounding box de la persona |
| **Año** | 2011 |
| **Paper** | Yao & Fei-Fei, ICCV 2011 |
| **Tipo** | Imágenes fijas (no video) |

Naming convention: `<action>_<NNN>.jpg` — por ejemplo `playing_guitar_042.jpg`. Esto permite extraer la clase con una regex:

```python
FILENAME_SUFFIX_REGEX = r"_[0-9]+\.jpg"
label = re.sub(FILENAME_SUFFIX_REGEX, '', filename)
# 'playing_guitar_042.jpg' → 'playing_guitar'
```

Robusta porque captura "sufijo numérico + .jpg" como **patrón conceptual**, no como posición literal. Si el filename fuera `playing_guitar_v2_042.jpg`, igual extraería `playing_guitar_v2`.

## Descarga + descompresión

```bash
!wget http://vision.stanford.edu/Datasets/Stanford40.zip
!unzip Stanford40.zip -d /content/stanford40
```

- **1.9 GB** descargados desde mirror Stanford Vision Lab (Fei-Fei Li lab).
- Descompresión genera ~9,532 imágenes JPG en `/content/stanford40/JPEGImages/`.
- También vienen splits predefinidos (`ImageSplits/`) y bboxes (`XMLAnnotations/`) que el lab **no usa**.

Esta es la **operación más larga del lab** (~2-4 minutos en Colab típico) y un **punto único de falla**: si el mirror de Stanford cae, el lab queda bloqueado.

## El `Stanford40Dataset` PyTorch

Clase custom que orquesta tres responsabilidades:

1. **Leer imágenes** desde el filesystem.
2. **Correr inferencia** con OpenPose y/o PifPaf sobre cada imagen.
3. **Cachear predicciones** en memoria para el MLP downstream.

```python
class Stanford40Dataset(Dataset):
    def __init__(self, directory, transform=None, pifpaf=None, openpose=None):
        self.directory = directory
        self.transform = transform
        self.pifpaf = pifpaf
        self.openpose = openpose
        self.image_filenames = [f for f in os.listdir(directory)
                                if f.endswith('.jpg')][:MAX_SAMPLES]
        self.pifpaf_predictions = {}
        self.openpose_predictions = {}
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        ...  # PIL.Image.open + convert RGB + return (image, filename)
    
    def process_images(self):
        for i in tqdm(range(len(self))):
            image, filename = self[i]
            if self.pifpaf is not None:
                # ... corre PifPaf y guarda en self.pifpaf_predictions[filename]
            if self.openpose is not None:
                # ... corre OpenPose + openpose_extract_keypoints
                # guarda en self.openpose_predictions[filename]
```

### Diseño elegante

- **Inyección de dependencias** (`pifpaf=processor, openpose=openpose_model`): la misma clase funciona con cualquier par de modelos compatibles.
- **Caché separado por modelo** (`pifpaf_predictions` y `openpose_predictions` son dicts distintos): permite procesar ambos modelos en una sola pasada sobre el dataset.
- **`tqdm` para visibilidad**: una operación de 10-15 minutos sin barra de progreso parece colgada.

### La constante crítica: `MAX_SAMPLES = 2000`

```python
MAX_SAMPLES = 2000
```

**La línea más importante del lab** y la más fácil de pasar por alto. Limita el dataset a 2,000 imágenes (de 9,532) por tres razones:

1. **Tiempo**: correr ambos modelos sobre todas las imágenes tomaría 30-60 minutos en Colab. Sobre 2,000, ~10-15 minutos.
2. **Memoria**: cachear 9,500 predicciones consumiría ~500 MB de RAM solo para los caches.
3. **Estabilidad**: Colab desconecta sesiones largas inactivas.

### El bug pedagógico no documentado

`os.listdir(...)` retorna archivos en orden **probablemente alfabético** en ext4. Las primeras 2000 imágenes incluyen clases que empiezan con A, B, C... Implicación para las clases del subset:

| Clase | Primera letra | ¿Cae en las primeras 2000? |
|---|---|---|
| `climbing` | C | ✅ todas las ~200 imágenes |
| `cutting_vegetables` | C | ✅ todas |
| `playing_guitar` | P | ⚠️ probablemente parcial |
| `riding_a_horse` | R | ❌ probablemente la mayoría se pierde |

**Esto sesga el experimento**. Si `riding_a_horse` tiene 50 muestras en lugar de 200, el clasificador aprende peor en esa clase. El fix profesional sería:

```python
filenames = [f for f in os.listdir(directory) if f.endswith('.jpg')]
random.seed(42)
random.shuffle(filenames)
self.image_filenames = filenames[:MAX_SAMPLES]
```

El lab no lo hace por simplicidad. **Es una limitación reconocible**.

## El subset de 4 clases

```python
SUBSET_LABELS = ['playing_guitar', 'climbing', 'riding_a_horse', 'cutting_vegetables']
```

**Una sola línea, decisión gigante**. Cuatro criterios convergentes:

1. **Distinguibilidad postural**: las 4 acciones tienen **poses muy distintas**. Un MLP simple sobre keypoints debería separarlas. Compara con un subset alternativo "difícil" como `writing_on_book` vs. `reading` (ambas: persona sentada mirando hacia abajo — casi indistinguibles por keypoints solo).

2. **Balance razonable**: las 4 tienen ~150-280 imágenes cada una en el dataset completo. Sin imbalance severo que requiera weighted loss.

3. **Dependencia de pose, no de objeto**: las 4 son distinguibles por **configuración corporal** sin necesitar reconocer el objeto. Compara con `using_a_computer` vs. `reading` — distinción requiere ver el objeto, un MLP sobre keypoints no podría separarlas.

4. **Exclusión deliberada**: `running` y `riding_a_bike` están **reservadas para la actividad evaluable**. Forzando un dominio nuevo en la actividad, el alumno debe **transferir el patrón**, no memorizarlo.

## Las 40 dimensiones del label

```python
DATA_LABELS = list(labels)
LABEL_TO_INDEX = {label: index for index, label in enumerate(DATA_LABELS)}
```

Aunque el experimento usa solo 4 clases, **el one-hot encoding sigue siendo de 40 dimensiones**. Razón implícita: reusar el espacio completo permite explorar otras combinaciones de clases sin cambiar la arquitectura del MLP.

**Costo**: el output del MLP tiene 40 dimensiones, ~5,160 parámetros redundantes. **Beneficio**: la arquitectura es **parametrizable por la lista de subset_labels** — si quisieras experimentar con otras 2, 3, 4, 5 clases, no cambias el MLP, solo el filtro y los gráficos.

## Cross-links

{{< cards >}}
  {{< card link="../demos-tres-librerias" title="Demos sobre la misma imagen" subtitle="Paso previo: intuición visual" icon="academic-cap" >}}
  {{< card link="../clasificador-pifpaf" title="Clasificador MLP con PifPaf" subtitle="Siguiente: entrenamiento real" icon="academic-cap" >}}
  {{< card link="../clasificador-openpose" title="Clasificador MLP con OpenPose" subtitle="A/B test simétrico" icon="academic-cap" >}}
  {{< card link="/fundamentos/pose-estimation" title="Fundamento: Pose Estimation" subtitle="Bottom-up vs. top-down" icon="book-open" >}}
{{< /cards >}}
