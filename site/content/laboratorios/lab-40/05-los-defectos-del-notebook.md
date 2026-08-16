---
title: "05 - Los defectos del notebook"
weight: 50
math: true
---

> El notebook del práctico arrastra siete defectos, y ninguno impide que corra. Esa es justamente la razón para catalogarlos: **todos producen resultados que parecen correctos**, incluido uno que muestra el video equivocado junto a las predicciones y no se nota porque ambos son de la misma clase.

---

## 1. El consejo críptico del principio

La segunda celda del notebook dice:

> **Consejo:** se debe ejecutar primero la descarga del dataset (paso 4)

Sin explicación. Parece una recomendación de eficiencia —la descarga tarda, empecemos por ahí— y en realidad es la única forma de que el notebook funcione.

La celda 6 termina con:

```python
os.chdir("/content/temporal-shift-module")
```

y en Colab los comandos con `!` heredan ese directorio. A partir de ahí:

| Celda | Código | Qué hace realmente |
|---|---|---|
| 27 | `!wget ...UCF101.rar` | guarda en `/content/temporal-shift-module/UCF101.rar` |
| 28 | `!unrar x '/content/UCF101.rar'` | busca en `/content/` → **archivo no encontrado** |
| 29 | `UCF_ROOT = '/content/UCF-101'` | apunta a donde `unrar` tampoco habría extraído |

Tres desajustes encadenados. Si en cambio se ejecuta la sección 4 **antes** del `chdir`, el directorio de trabajo sigue siendo `/content` y las tres rutas calzan por accidente.

El arreglo es hacer explícitas las rutas en lugar de depender del cwd:

```python
!aria2c -x 16 -s 16 --continue=true --check-certificate=false \
    -d /content -o UCF101.rar https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
!unrar x -o+ /content/UCF101.rar /content/
```

El `-d /content` y el `/content/` final fijan destino de descarga y de extracción. De paso, `aria2c` con 16 conexiones baja los **6,93 GB** en unos minutos contra los 15-25 de `wget` en un solo hilo. Y `unrar` **no viene instalado** en Colab (`apt-get install -y unrar`); si el repositorio multiverse no está habilitado, `bsdtar` de libarchive lee RAR4, que es el formato de este archivo.

---

## 2. La función de descarga que no descarga

```python
UCF_ROOT = '/content/UCF-101'
_CACHE_DIR = '/content/data/'

def fetch_ucf_video(video):
  """Fetchs a video and cache into local filesystem."""
  cache_path = os.path.join(_CACHE_DIR, video)
  if not os.path.exists(cache_path):
    urlpath = request.urljoin(UCF_ROOT, video)
    data = request.urlopen(urlpath, context=unverified_context).read()
    open(cache_path, "wb").write(data)
  return cache_path
```

Viene del tutorial de TF Hub para I3D —el mismo del [Laboratorio 38](/laboratorios/lab-38)—, donde `UCF_ROOT` era una URL y la función tenía sentido: `urljoin` armaba la dirección del video y lo descargaba. Alguien cambió `UCF_ROOT` por una ruta local y la llamada quedó así:

```python
video_path = fetch_ucf_video("/content/UCF-101/HighJump/v_HighJump_g01_c02.avi")
```

Con un argumento **absoluto**:

```python
os.path.join('/content/data/', '/content/UCF-101/HighJump/v_HighJump_g01_c02.avi')
# → '/content/UCF-101/HighJump/v_HighJump_g01_c02.avi'
```

`os.path.join` **descarta el primer componente cuando el segundo es absoluto**. Así que `cache_path` termina siendo el argumento, el archivo ya existe porque lo extrajo el RAR, el `if` no entra y la función **devuelve lo que le pasaron**. No descarga, no cachea, no toca `/content/data/`.

Y solo no explota porque el archivo existe: con un video inexistente entraría al `if`, y `request.urljoin('/content/UCF-101', '/content/...')` devuelve una ruta sin esquema que `urlopen` rechaza con `ValueError: unknown url type`.

El origen del problema es que el notebook tiene **dos estrategias de datos superpuestas** —bajar los 6,93 GB completos y bajar un video suelto por HTTP— y la primera hace innecesaria la segunda, pero nadie borró la segunda. La variable `video_path` que produce tampoco se usa: la celda siguiente vuelve a escribir la ruta a mano.

---

## 3. El GIF que muestra otro video

El más entretenido, porque produce una demo convincente por la razón equivocada.

```python
aviFilenamesList = glob.glob('/content/UCF-101/HighJump/*.avi')   # 123 videos
index = 0

to_gif(aviFilenamesList[index])                                    # celda 62
probabilities = outputs[index].reshape(outputs[index].shape[1])    # celda 63
```

El **mismo `index`** indexa dos listas sin ninguna relación:

- `aviFilenamesList` viene de `glob.glob` sobre los **123 videos** de HighJump. `glob` **no ordena**: devuelve lo que da el sistema de archivos.
- `outputs` tiene **un** elemento, la predicción del único video con frames extraídos, `v_HighJump_g01_c02`.

Con altísima probabilidad **el GIF que se muestra no es el video cuyas probabilidades se imprimen**. Como los 123 clips son todos de salto alto, el resultado parece coherente y nadie lo nota. Es un falso positivo pedagógico perfecto: la demo convence precisamente porque el error es invisible cuando ambos videos comparten clase.

Y hay un segundo índice en juego. El orden de `outputs` lo fija `os.listdir` dentro de `_parse_list`:

```python
def _parse_list(self):
    tmp = os.listdir(self.root_path)     # sin ordenar
    ...
```

Al agregar el segundo video de la actividad, `outputs` pasa a tener dos elementos en un orden que no está garantizado. El arreglo es indexar por nombre:

```python
idx_de = {os.path.basename(r.path): i for i, r in enumerate(dataset_test.video_list)}
# {'v_HighJump_g01_c02': 0, 'v_PlayingGuitar_g01_c01': 1}
```

En esta corrida el orden resultó alfabético, pero eso es una coincidencia del sistema de archivos, no una garantía.

---

## 4. La GPU que nunca se usa

```python
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
```

La variable se crea y **no vuelve a aparecer en todo el notebook**. No hay `net.to(device)` ni `data.to(device)`. El modelo se queda en CPU, los tensores también, y la T4 que el notebook pide en su metadata (`"accelerator": "GPU"`) no ejecuta una sola operación.

No cambia los resultados —la predicción es idéntica— pero sí el tiempo. Medido en este lab: **2,1 segundos para dos videos en CPU**, bastante menos de lo que sugeriría el conteo de 33 GFLOPs por video, porque PyTorch paraleliza sobre los hilos disponibles. Con un video la diferencia no se nota; con los 123 de una categoría, sí.

---

## 5. El checkpoint `dense` evaluado con muestreo uniforme

El archivo se llama:

```
TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense.pth
                                                            ^^^^^
```

y la configuración fija:

```python
'dense_sample': False,
```

Los pesos se entrenaron con **muestreo denso** —clips contiguos estilo I3D, 10 por video— y se evalúan con **muestreo uniforme por segmentos**. Las políticas de entrenamiento y evaluación no coinciden.

El notebook no lo menciona, y funciona igual porque Kinetics es tolerante. Es la única inconsistencia del práctico que este laboratorio detectó y **no midió**: bastaría poner `dense_sample=True` y comparar.

---

## 6. Los tres menores

**`os.mkdir` sin `exist_ok`.**

```python
os.mkdir('/content/data/')
os.mkdir('/content/data/videos/')
```

Lanza `FileExistsError` al re-ejecutar, cosa que hay que hacer sí o sí al llegar a la actividad de la sección 9. Se arregla con `os.makedirs(..., exist_ok=True)`.

**El ffmpeg silencioso.**

```python
subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
```

dentro de una función con un `except:` desnudo. Si ffmpeg falla, no hay ningún mensaje: la carpeta queda vacía y el error aparece varias celdas después, disfrazado y mucho peor. Con `num_frames = 0`, `_get_test_indices` calcula `tick = 0`, todos los índices apuntan a `img_00001.jpg` que no existe, y el `while not os.path.exists` de `__getitem__` se pone a buscar otro video al azar. Con un solo video en el directorio, **bucle infinito sin mensaje de error**.

La verificación cuesta dos líneas y conviene tenerla siempre:

```python
n = len(glob.glob(dst + '/img_*.jpg'))
assert n > 0, 'ffmpeg no extrajo nada'
```

**La descarga desperdiciada.** `'pretrain': 'imagenet'` hace que `_prepare_base_model` baje **97,8 MB** de pesos ImageNet de ResNet-50… que la carga del checkpoint sobreescribe por completo. El `<All keys matched successfully>` es la prueba: el 100 % de los pesos vino de Kinetics.

---

## 7. Un defecto de la época, no del autor

Aparte, y verificable: la celda que carga el checkpoint es

```python
checkpoint = torch.load(this_weights)
```

Los primeros bytes del `.pth` son `\x80\x02\x8a\n...`, o sea **pickle protocol 2**: el formato `.tar` legacy anterior a PyTorch 1.6. Desde PyTorch 2.6 el default de `torch.load` es `weights_only=True`, que rechaza ese formato con un `RuntimeError` — el unpickler restringido ni siquiera intenta abrirlo. En el runtime donde se ejecutó este lab la carga funcionó, así que la versión instalada es anterior; en un Colab más nuevo hará falta:

```python
checkpoint = torch.load(this_weights, map_location='cpu', weights_only=False)
```

El `map_location` es higiene aparte: el checkpoint recuerda que sus tensores vivían en `cuda:0` —la GPU del MIT en 2019— y sin él PyTorch intenta recrearlos ahí.

---

## Catálogo

| # | Defecto | Síntoma | Arreglo |
|---|---|---|---|
| 1 | `chdir` contra rutas absolutas | `unrar` no encuentra el archivo | `-d /content` y destino explícito |
| 2 | `fetch_ucf_video` es un no-op | ninguno: devuelve su argumento | borrarla, o usar rutas relativas |
| 3 | `index` compartido entre `glob` y `outputs` | GIF de un video, probabilidades de otro | diccionario nombre → índice |
| 4 | `device` definido y nunca usado | inferencia en CPU con GPU disponible | `net.to(device)` |
| 5 | checkpoint `dense` con `dense_sample=False` | ninguno visible | medirlo |
| 6a | `os.mkdir` sin `exist_ok` | `FileExistsError` al re-ejecutar | `os.makedirs(exist_ok=True)` |
| 6b | ffmpeg silenciado + `except:` desnudo | **bucle infinito** varias celdas después | `assert n_frames > 0` |
| 6c | descarga de ImageNet desperdiciada | 97,8 MB y unos segundos | `pretrain=None` |
| 7 | `torch.load` sin `weights_only=False` | `RuntimeError` en PyTorch ≥ 2.6 | pasar el argumento |

Ninguno cambia las predicciones. Cinco de los nueve **no producen ningún síntoma visible**, y son los que importan: el que muestra un video por otro sobrevivió porque los 123 clips de la carpeta comparten clase, y el del checkpoint `dense` sobrevive porque Kinetics tolera el desajuste. Es el mismo patrón del [Laboratorio 39](/laboratorios/lab-39/01-la-fuga-de-folds), donde una fuga de datos producía un número que coincidía con el paper hasta la segunda cifra decimal.

---

## Ver también

- [01 - El shift desarmado](01-el-shift-desarmado) — la parte del notebook que sí está bien resuelta, incluidos los flags de preproceso derivados de la arquitectura.
- [02 - La varianza intra-clase](02-la-varianza-intra-clase) — por qué concluir desde un solo video es el defecto metodológico que engloba a todos estos.
- [Laboratorio 39 - La fuga de folds](/laboratorios/lab-39/01-la-fuga-de-folds) — el caso extremo de un bug que produce el número correcto.
- [Laboratorio 38](/laboratorios/lab-38) — el bug de preproceso que costó 82 puntos, y que este notebook evita.
