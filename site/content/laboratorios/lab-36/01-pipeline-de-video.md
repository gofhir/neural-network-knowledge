---
title: "El pipeline de video"
weight: 1
---

Un video suma una dimensión a la visión: el **tiempo**. Esta primera parte construye el pipeline que convierte videos crudos en tensores que un modelo puede consumir — descarga, preprocesamiento, muestreo temporal y augmentation.

## El dataset: UCF11

**UCF11** (UCF YouTube Action) son videos reales de YouTube con **11 categorías de acciones** (basketball, biking, diving, golf_swing, horse_riding, soccer_juggling, swing, tennis_swing, trampoline_jumping, volleyball_spiking, walking), ~150 videos por clase. Al ser videos "in the wild" tienen condiciones difíciles —cámara temblorosa, iluminación variable, fondos diversos, escala cambiante—, lo que lo hace realista. Es el "hola mundo" del [análisis de video](/fundamentos/analisis-de-video), análogo a lo que MNIST fue para imágenes.

## Preprocesamiento: video → frames en disco

La idea fundamental: **un video es una secuencia de imágenes (frames)**. El preprocesamiento descomprime cada video `.mpg` y guarda **todos** sus frames como `.jpg`:

```python
def vids_to_frames(path_videos, action, path):
    vidcap = cv2.VideoCapture(vid_path)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(f'frame_{count}.jpg', image)   # guarda CADA frame
        success, image = vidcap.read()
        count += 1
```

{{< callout type="info" >}}
**Descomprimir una vez, muestrear muchas veces.** Decodificar video es caro (~8 min para UCF11). Al extraer *todos* los frames a disco, el `VideoDataset` puede luego muestrear rápido —8 frames distintos por época, o cambiar a 4 frames— sin re-decodificar. Separar "decodificar" (lento, una vez) de "muestrear" (rápido, cada vez) es la decisión de ingeniería correcta.
{{< /callout >}}

El split train/val (80/20) se hace **por grupo de videos**, no por video individual: como UCF11 agrupa clips del mismo contexto/actor, poner unos en train y otros en val sería fuga de datos. *(Debilidad menor: el split no es aleatorio — toma los primeros 80% según orden de disco, sin `shuffle`.)*

## El `VideoDataset` y el muestreo temporal

La clase que lee las carpetas de frames y produce los tensores. Su método clave hace el **muestreo temporal** (respuesta a la pregunta 4 de la actividad):

```python
def __getitem__(self, idx):
    frames_elem.sort(key=self.natural_keys)         # orden natural: frame_0,1,2,...,10,11
    idx_frames = np.linspace(0, len(frames_elem)-1, num=self.num_frames, dtype=int)
    # para 90 frames, num_frames=8 -> [0, 12, 25, 38, 50, 63, 76, 89]
    for ind_frame in idx_frames:
        frame = Image.open(...).convert('RGB')
        list_frames.append(frame)
    return self.transform(list_frames), label
```

Es **muestreo temporal uniforme**: en vez de usar todos los frames, elige `num_frames` (8) **equiespaciados** a lo largo del video. Por qué:
- **Eficiencia**: procesar 90 frames por CNN sería 11× más caro, y frames consecutivos son casi idénticos (redundantes).
- **Cobertura**: el espaciado uniforme garantiza cubrir todo el video (inicio, medio, fin), capturando la esencia de la acción.

Detalle importante: el `natural_keys` ordena los frames **numéricamente** (`frame_2` antes de `frame_10`), no alfabéticamente. Sin esto, la secuencia temporal quedaría desordenada.

{{< callout type="info" >}}
**El dataset preserva el orden; el modelo lo tirará.** El `VideoDataset` entrega 8 frames *ordenados*. La pérdida del orden temporal NO ocurre aquí — ocurre después, en el average pooling del modelo. Es una distinción clave para las preguntas 4 (muestreo, que está bien) vs 5 (pooling, que es el problema).
{{< /callout >}}

## Augmentation de video: la coherencia temporal

Las transformaciones tienen el prefijo `Group` porque se aplican **consistentemente a los 8 frames**, no independiente por frame:

```python
class GroupRandomHorizontalFlip(object):
    def __call__(self, img_group):
        v = random.random()
        if v < 0.5:
            return [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]  # voltea TODOS
        else:
            return img_group                                                     # o NINGUNO
```

{{< callout type="warning" >}}
**Si voltearas cada frame por separado, destruirías el movimiento.** Imagina alguien caminando a la derecha: si volteas el frame 1 pero no el 2, el 3 sí... el movimiento sería un caos incoherente. La augmentation de video **debe preservar la coherencia temporal**: o transformas todo el clip igual, o nada. Lo mismo con el crop (la misma región en los 8 frames). Es la diferencia fundamental entre augmentar imágenes y augmentar video.
{{< /callout >}}

El augmentation es **asimétrico**: train usa transformaciones aleatorias (`GroupMultiScaleCrop` + `GroupRandomHorizontalFlip`) para generar variedad y reducir overfitting; val usa deterministas (`GroupCenterCrop`) para evaluar de forma consistente. Con UCF11 chico (~1200 videos) y un ResNet-34 grande, el augmentation es lo que evita el sobreajuste — combinado con el transfer learning de la siguiente parte.
