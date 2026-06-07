---
title: "App 2 · Google Street View"
weight: 5
math: true
---

> **Celdas 62-97 del notebook.** Segunda aplicación: leer texto en ~6000 fotos de [Google Street View de Pittsburgh](/papers/street-view-geolocalization-zamir-2010) y, cruzando con GPS, **mapear geográficamente** dónde aparece cada palabra. Añade una dimensión que la app 1 no tenía: georreferenciación.

## El planteamiento + un problema inesperado

El dataset (UCF) trae **información GPS por posición** (6 fotos por punto). El objetivo final es una función `draw_in_map(keyword)` que marque en un mapa de Pittsburgh dónde se vio una palabra. Pero al mirar el top 40 de palabras aparece algo raro:

| Top "palabras" en la calle | Freq | Diagnóstico |
|---|---|---|
| 54 (**1096**) · 59 (848) · 58 (771) · 53 (511) | miles | ⚠️ no es texto de calle real |
| e9, e8, e4, photos (217), google (38), 02009 | cientos | UI de Google Maps |
| **penn (35), ave (33), stop (37), south (25)** | decenas | texto de calle **real** (enterrado) |

🎯 El número "54" aparece en ~1 de cada 6 imágenes — imposible para texto real. La causa: muchas capturas incluyen el **overlay de la interfaz de Google Maps** (controles, etiquetas, watermark). El modelo lee *todo* el texto, sin distinguir contenido de la foto vs. *chrome* de la app.

![GSV 000002_0 con detecciones sobre el contenido Y sobre el overlay de Maps](/laboratorios/lab-21/gsv-overlay.jpg)

Detalles reveladores del ruido: `02009` es el watermark **"© 2009"** (el © → 0 porque está fuera del charset; fecha el dataset); `photos` (217) y `protos` (97) son el mismo botón leído bien y mal; "Penn Ave" aparece partido en `penn`+`ave` (otra vez detección palabra-por-palabra).

> **Lección transversal:** cuando un resultado es estadísticamente raro, investiga la causa antes de confiar. El ranking de frecuencias es la herramienta de diagnóstico que destapa el ruido sistemático — el equivalente, en datos clínicos, a un campo que parece señal pero es un default del sistema (fechas "1900-01-01", identificadores "11111111-1").

## `get_mask`: filtrado por zonas (celda 79)

Como la UI está en **posición fija**, se puede descartar el texto cuyo centro caiga en ciertas zonas:

```python
import torch as pt

def get_mask(centers, filter_zones):
    mask = pt.zeros(len(centers), dtype=pt.bool, device=centers.device)
    for (x0, y0), (x1, y1) in filter_zones:
        inside_x = pt.logical_and(centers[:,0] > x0, centers[:,0] < x1)
        inside_y = pt.logical_and(centers[:,1] > y0, centers[:,1] < y1)
        mask = pt.logical_or(mask, pt.logical_and(inside_x, inside_y))
    return pt.logical_not(mask)   # True = conservar (NO está en ninguna zona)
```

La lógica es **point-in-rectangle vectorizado**: `centers[:,0] > x0` opera sobre los N centros a la vez (sin loop sobre puntos); el único `for` es sobre las zonas (2-4). `device=centers.device` crea la máscara en el mismo device que los datos.

### `get_mask` — triple framework

{{< tabs >}}
{{< tab name="PyTorch" >}}
```python
import torch

def get_mask(centers, filter_zones):
    mask = torch.zeros(len(centers), dtype=torch.bool, device=centers.device)
    for (x0, y0), (x1, y1) in filter_zones:
        inside = (centers[:,0] > x0) & (centers[:,0] < x1) & \
                 (centers[:,1] > y0) & (centers[:,1] < y1)
        mask = mask | inside
    return ~mask
```
{{< /tab >}}
{{< tab name="TensorFlow" >}}
```python
import tensorflow as tf

def get_mask(centers, filter_zones):
    mask = tf.zeros(tf.shape(centers)[0], dtype=tf.bool)
    for (x0, y0), (x1, y1) in filter_zones:
        inside = (centers[:,0] > x0) & (centers[:,0] < x1) & \
                 (centers[:,1] > y0) & (centers[:,1] < y1)
        mask = mask | inside
    return tf.logical_not(mask)
```
{{< /tab >}}
{{< tab name="JAX" >}}
```python
import jax.numpy as jnp

def get_mask(centers, filter_zones):
    mask = jnp.zeros(centers.shape[0], dtype=bool)
    for (x0, y0), (x1, y1) in filter_zones:
        inside = (centers[:,0] > x0) & (centers[:,0] < x1) & \
                 (centers[:,1] > y0) & (centers[:,1] < y1)
        mask = mask | inside
    return ~mask
```
{{< /tab >}}
{{< /tabs >}}

> Los operadores `&`, `|`, `~` están sobrecargados en los tres frameworks para AND/OR/NOT elemento a elemento. En JAX los arrays son inmutables, pero reasignar el nombre `mask` a un array nuevo en cada iteración está bien.

## Aplicar el filtro (celda 81)

```python
filter_zones = [
    [(0, 0), (95, 145)],        # esquina superior izquierda
    [(95, 10), (500, 50)],      # barra de búsqueda
    [(1190, 0), (1280, 125)]    # esquina superior derecha
]
for image in tqdm(dataset):
    mask = get_mask(image['boxes'].get_centers(), filter_zones)
    image['boxes'] = image['boxes'][mask]
    indices = mask.nonzero().squeeze()
    if indices.shape != ():     # ← maneja el caso de 1 sola palabra
        image['words'] = [image['words'][int(i)] for i in indices]
    else:
        image['words'] = [image['words'][int(indices)]]
```

`boxes.get_centers()` convierte cada caja `(x1,y1,x2,y2)` en su punto medio. Las zonas son **hardcodeadas** (el profe las obtuvo inspeccionando la imagen): solución específica al dataset, no general.

> 🎯 **Gotcha de `.squeeze()`:** `mask.nonzero().squeeze()` colapsa la forma según los datos — si queda **1 sola** palabra, devuelve un escalar 0-D (`shape == ()`), **no iterable**, y un `for` sobre él lanzaría `TypeError`. El `if indices.shape != ()` detecta ese caso. Es un bug clásico de PyTorch/NumPy que solo aparece con ciertos inputs. Alternativa más segura: `nonzero(as_tuple=True)[0]`, que siempre devuelve 1-D.

### El resultado: el ruido desaparece

> Tras filtrar, la palabra más frecuente bajó de **1096 a 46** (~24×). Desaparecieron los números 52-60 y "photos"; el texto real de calle (Penn Ave, stop, food) emergió. El filtrado por zona no elimina el watermark en posición variable ni la basura OCR, pero limpia drásticamente el ranking.

## GPS + el mapa (celdas 85-90)

```python
import scipy.io as sio
coords = sio.loadmat('GPS_Long_Lat_Compass.mat')['GPS_Compass']
min_latitude, min_longitude = coords[:1099].min(axis=0)[:2]
max_latitude, max_longitude = coords[:1099].max(axis=0)[:2]
```

`scipy.io.loadmat` lee el `.mat` (formato MATLAB del dataset UCF). Cada fila es `[lat, lon, compass, …]`. `.min(axis=0)[:2]` calcula el **bounding box geográfico** de las 1099 posiciones (subconjunto de 6594 fotos ÷ 6 vistas/posición). Pittsburgh está en ~40.44° N, -79.99° W.

## La función culminante: `draw_in_map` (celda 92)

```python
def draw_in_map(keyword, dataset, coords, map, min_longitude, max_longitude,
                min_latitude, max_latitude, threshold=80, show_match_image=False):
    latitudes, longitudes = [], []
    for image in dataset:
        for word in image['words']:
            if fuzz.ratio(word.lower(), keyword) > threshold:
                index = image['image_id'] // 6           # foto → posición GPS
                lat, lon = coords[index][:2]
                latitudes.append(lat); longitudes.append(lon)
                if show_match_image:
                    plt.imshow(Image.open(image['file_name'])); plt.show()
                break                                    # un match por imagen
    fig, ax = plt.subplots()
    ax.imshow(map, extent=(min_longitude, max_longitude, min_latitude, max_latitude))
    ax.scatter(longitudes, latitudes, c='red')
    plt.show()
```

La función integra **las tres capas del lab**: OCR (ABCNet) + [fuzzy matching](/fundamentos/fuzzy-string-matching) (Levenshtein) + georreferenciación. Dos detalles clave:

- **`image_id // 6`**: como hay 6 fotos por posición GPS, la división entera mapea la foto a su punto. Depende del `.sort()` que alineó los `image_id` con el orden de `coords`.
- **`extent=(min_lon, max_lon, min_lat, max_lat)`**: le dice a matplotlib que la imagen del mapa ocupa el **rango geográfico** (no píxeles), de modo que el `scatter(longitudes, latitudes)` cae exactamente donde corresponde. Esto es **georreferenciación**: alinear píxeles del mapa ↔ grados geográficos. El orden importa: `x = longitud`, `y = latitud`; confundirlos rotaría/reflejaría el mapa.

![Mapa de Pittsburgh con los puntos donde aparece "university" (concentrados en Oakland)](/laboratorios/lab-21/mapa-university.jpg)

> Probado con `'university'` (threshold 80, `show_match_image=False`), los puntos se agrupan en **Oakland** — el barrio de University of Pittsburgh y Carnegie Mellon. El resultado es interpretable y verificable: la palabra aparece donde están las universidades. Es minería de información geoespacial desde imágenes, lo que hacen sistemas como Google Maps para poblar negocios.

---

**Anterior:** [App 1 · Freiburg Groceries](app-groceries) · **Siguiente:** [Actividades](actividades)
