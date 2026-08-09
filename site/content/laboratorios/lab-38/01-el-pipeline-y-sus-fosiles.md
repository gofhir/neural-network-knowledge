---
title: "El pipeline y sus fósiles"
weight: 1
---

Las primeras 26 celdas son setup: instalar, importar, descargar UCF101 y definir tres utilidades para leer video. Nada aquí ejecuta el modelo. Pero es donde se decide **qué píxeles ve I3D**, y una de estas líneas resulta ser la causa del fallo de la actividad.

El notebook es una adaptación del [tutorial oficial de TensorFlow Hub](https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub), y la adaptación dejó **fósiles**: código que quedó huérfano de las variables que necesitaba y sobrevive por accidentes del lenguaje.

## Los 6,93 GB que se descargan para leer 294 KB

```python
!wget --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
!unrar x '/content/UCF101.rar'
```

El servidor responde `Content-Length: 6932971618` — **6,46 GiB**, 13.320 clips en 101 clases, todos a 320×240 y 25 fps. Se descargan en 2 m 27 s a 45 MB/s. Y todo eso para que la celda 23 abra **un** archivo de 294.566 bytes.

Dos detalles del entorno:

- **`--no-check-certificate`** no es paranoia: el servidor de CRCV arrastra un certificado TLS mal encadenado y `wget` aborta sin ese flag. Es la contraparte shell del `ssl._create_unverified_context()` que el tutorial original tenía y esta versión borró (dejando el `import ssl` huérfano en la celda 7).
- **`unrar`** sí estaba disponible en Colab. UCF101 se publicó en 2012 desde un laboratorio donde el estándar era WinRAR, y RAR es un formato propietario: la especificación de compresión nunca se liberó. Además es un archivo **sólido**, lo que significa que no se puede extraer un solo video sin descomprimir todo lo anterior.

{{< callout type="info" >}}
**Alternativa verificada.** El repositorio `bryanyzhu/tiny-ucf101` —el mismo de donde sale el video de la actividad— publica `tiny-UCF101.zip`: **50,7 MB**, 101 clases con un clip cada una, estructura de carpetas idéntica, y contiene el mismo `v_ApplyEyeMakeup_g01_c01.avi` de 294.566 bytes. Los resultados de las celdas 20 a 35 son idénticos; sólo cambia que `list_ucf_videos` imprime `1 videos` por categoría en vez de `~130`.
{{< /callout >}}

## Fósil 1: la función identidad disfrazada de descargador

```python
_CACHE_DIR = tempfile.mkdtemp()

def fetch_ucf_video(video):
  cache_path = os.path.join(_CACHE_DIR, video)
  if not os.path.exists(cache_path):
    urlpath = request.urljoin(UCF_ROOT, video)          # UCF_ROOT no existe
    data = request.urlopen(urlpath, context=unverified_context).read()   # tampoco
    open(cache_path, "wb").write(data)
  return cache_path
```

`UCF_ROOT` y `unverified_context` **no están definidos en ninguna parte del notebook**. La función debería explotar con `NameError`. No lo hace, y el motivo es una regla de `os.path.join` que rara vez se tiene presente:

> Si algún componente es una ruta absoluta, **todos los anteriores se descartan**.

La celda 23 llama a la función con una ruta absoluta:

```python
>>> os.path.join('/tmp/tmpab3x9k2q', '/content/UCF-101/.../v_ApplyEyeMakeup_g01_c01.avi')
'/content/UCF-101/.../v_ApplyEyeMakeup_g01_c01.avi'      # el cache dir se evapora
```

Como el archivo ya existe (se descomprimió en la celda 11), `os.path.exists` devuelve `True`, el `if` no entra, y las dos variables inexistentes nunca se evalúan. **`fetch_ucf_video` quedó convertida en una función identidad**: recibe una ruta y devuelve la misma ruta. El directorio temporal queda vacío para siempre.

Con un nombre *relativo* —como hacía el tutorial original— el fósil se despierta:

```python
fetch_ucf_video("v_ApplyEyeMakeup_g01_c01.avi")
# NameError: name 'UCF_ROOT' is not defined
```

Por eso el markdown de la celda 21 ("*el beneficio de `fetch_ucf_video` es que no necesitamos descargar todo el conjunto de datos*") es hoy literalmente falso: no descarga nada, y la celda 10 baja el dataset completo.

## Fósil 2: el docstring que promete un `dict` inexistente

`list_ucf_videos` documenta `Returns: dict` y **no tiene ninguna sentencia `return`** — sólo imprime. Así que en la celda 20:

```python
ucf_videos = list_ucf_videos(VIDEO_DIR)     # ucf_videos == None
```

Es inofensivo únicamente porque `ucf_videos` no vuelve a usarse. Un `ucf_videos['Archery']` daría `TypeError: 'NoneType' object is not subscriptable`.

Vale la pena mirar la lista que sí imprime, porque anticipa el problema central del lab: las 101 clases de UCF101 (`ApplyEyeMakeup`, `Typing`, `PlayingGuitar`, ~50 deportes) **no son las 400 de Kinetics-400**. Hay solapamiento parcial, no equivalencia.

## El preprocesamiento: qué ve realmente el modelo

### `crop_center_square` — el 25 % que se tira

```python
def crop_center_square(frame):
  y, x = frame.shape[0:2]
  min_dim = min(y, x)
  start_x = (x // 2) - (min_dim // 2)
  start_y = (y // 2) - (min_dim // 2)
  return frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
```

Con UCF101 (320×240): `min_dim = 240`, `start_x = 40` → se recorta `frame[0:240, 40:280]` y **se descartan 40 px por lado, el 25 % del ancho**.

Se recorta en vez de deformar porque el *squash* directo a 224×224 estiraría los cuerpos horizontalmente, y una CNN entrenada con personas de proporciones reales degrada con eso. El costo es perder los bordes — y en la actividad, con un video de 454 px de ancho, ese costo sube al **44 %**. Ese detalle reaparece en [dónde está la evidencia](../04-donde-esta-la-evidencia).

Conviene notar que el protocolo oficial de I3D es distinto: *"videos are resized preserving aspect ratio so that the smallest dimension is 256 pixels... during test time, the center 224×224 crop is selected"*. El notebook hace **crop primero, resize después**, lo que encoge todo un 12,5 % respecto de lo canónico.

### `load_video` — y la línea que causa el bug

```python
def load_video(path, max_frames=0, resize=(224, 224)):
  ...
      frame = crop_center_square(frame)
      frame = cv2.resize(frame, resize)
      frame = frame[:, :, [2, 1, 0]]          # BGR → RGB
  ...
  return np.array(frames) / 255.0             # ← la causa raíz
```

Tres cosas que merecen atención:

**`max_frames=0` como centinela.** El chequeo `len(frames) == max_frames` ocurre *después* del `append`, así que `len(frames)` vale al menos 1 y nunca iguala a 0: el video se lee entero.

**`frame[:, :, [2, 1, 0]]`** convierte BGR a RGB. OpenCV entrega BGR por herencia de 1999 —el formato BMP de Windows y las cámaras Intel de entonces— y nunca lo cambió. El *fancy indexing* devuelve una **copia**, que es justo lo que se necesita: el buffer de `cap.read()` se reutiliza entre frames, y guardar vistas dejaría todos los frames apuntando al último.

**`np.array(frames) / 255.0`** hace dos cosas, y la segunda es el bug del lab:

1. Promueve a `float64`: 164 × 224 × 224 × 3 × 8 bytes ≈ **197 MB** donde bastaban 99 en `float32`, para que `tf.constant(..., dtype=tf.float32)` lo reconvierta después.
2. Deja el rango en **$[0, 1]$**. El repositorio oficial [`google-deepmind/kinetics-i3d`](https://github.com/google-deepmind/kinetics-i3d) especifica: *"Pixel values are then rescaled between -1 and 1."* La *model card* del módulo de TF Hub documenta la forma del tensor pero **no el rango**, así que la discrepancia no se puede resolver leyendo documentación — hay que medirla. Es exactamente lo que hace [el bug del preproceso](../03-el-bug-del-preproceso).

### `to_gif` y el peso del entregable

```python
converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
imageio.mimsave('./animation.gif', converted_images, fps=25)
return embed.embed_file('./animation.gif')
```

`np.clip` protege contra un `255.0000001` que, al convertir a `uint8`, haría *wrap-around* y pintaría de negro un píxel blanco. `embed_file` codifica el GIF en base64 y lo incrusta como *data URI* dentro del `.ipynb` — decisión correcta, porque el entregable es el notebook ejecutado y un GIF referenciado por ruta aparecería roto al abrirlo en otra máquina. El costo: los tres GIFs de este lab suman **18,1 MB** de base64 sobre un notebook de 18,8 MB.

El `fps=25` está deprecado desde `imageio 2.28` y se traduce internamente a `duration = 40 ms`. Da la casualidad de que funciona exacto: el formato GIF guarda el retardo en **centisegundos**, y 40 ms son 4 cs justos. A 30 fps pedirías 33,3 ms, se redondearía a 3 cs y el GIF correría a 33,3 fps.

---

**Siguiente:** [El vocabulario manda](../02-el-vocabulario-manda) — las 400 etiquetas, `hub.load` y el contraste entre una clase que existe y otra que no.
