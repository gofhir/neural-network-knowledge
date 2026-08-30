---
title: "Los defectos del notebook"
weight: 5
---

El notebook es un adaptador didáctico sobre el repositorio de investigación original ([mpc001/end-to-end-lipreading](https://github.com/mpc001/end-to-end-lipreading), de Pingchuan Ma, coautor del paper), y arrastra el estilo de 2018: `Variable`, `errno`, sintaxis de PyTorch 0.3. Conviene separar lo que está roto de lo que solo es viejo.

| Categoría | Qué significa |
|---|---|
| **Bug** | El código hace algo distinto de lo que pretende |
| **Frágil** | Depende de versiones o del entorno |
| **Subóptimo** | Hace lo que pretende, con desperdicio |
| **Discrepancia** | El código no es lo que el paper describe |
| **Cosmético** | No afecta nada |

## 1. El bug: `return preds` — variable global filtrada

De todo el notebook, **este es el único defecto de comportamiento demostrable**.

```python
def predict(filename):
  ...
  _, pred = torch.max(F.softmax(output, dim=1).data, 1)   # ← variable local: pred
  pred_str = id2label[int(pred)]
  return preds, pred_str                                   # ← devuelve preds (con s)
```

La variable local se llama `pred`. El `return` entrega `preds`, que no existe en el ámbito de la función. Python la busca en el ámbito global y **la encuentra**: es la que quedó viva del loop de evaluación de la celda anterior.

```python
# en el loop de evaluación, dos celdas más arriba
_, preds = torch.max(F.softmax(outputs, dim=1).data, 1)   # sobrevive al for
```

Así que `predict()` devuelve **la predicción del último clip que procesó el loop de evaluación**, no la del video solicitado.

El efecto es sutil porque la celda que la consume imprime dos cosas:

```python
print('Model prediction: %s [%d]' % (pred_str, int(pred)))
```

- **`pred_str` es correcto** — se calculó con la variable local buena.
- **`int(pred)` es basura** — el índice de otra muestra cualquiera.

La línea se ve consistente y no lo es: la palabra es correcta, el número que la acompaña pertenece a otra predicción.

**Dos agravantes.** Es **no determinista**: el `DataLoader` usa `shuffle=True`, así que "el último clip del loop" cambia en cada corrida y el mismo video devuelve un índice distinto cada vez. Y **depende de un efecto colateral**: sin haber corrido antes el loop de evaluación, la función fallaría con `NameError`.

**La corrección es un carácter:** `return pred, pred_str`. Verificado: para `EXAMPLE/test/EXAMPLE_00001.mp4` la función corregida devuelve el índice **146**, que coincide con la posición de `EXAMPLE` en `label_sorted.txt`.

## 2. Frágil: librosa no decodifica MP4

```python
data = librosa.load(filename, sr=16000)[0][-19456:]
```

En Colab 2026 esto emite **dos warnings por archivo** — más de 5000 líneas para 2500 clips:

```
UserWarning: PySoundFile failed. Trying audioread instead.
FutureWarning: __audioread_load Deprecated as of librosa 0.10.0. It will be removed in librosa 1.0.
```

La cadena de backends es: `PySoundFile` (libsndfile) falla, porque libsndfile maneja WAV, FLAC, OGG y AIFF —formatos de audio puro— pero **un MP4 es un contenedor multimedia** con el audio típicamente en AAC. Cae entonces a `audioread`, que en Colab delega en **ffmpeg** por subproceso, y funciona.

**No es un error: el pipeline completa.** Pero tiene dos costos. El de velocidad —cada llamada levanta un proceso de ffmpeg completo, un `fork/exec` por clip—, y el de futuro: **cuando librosa llegue a 1.0, este notebook se romperá de verdad**.

Se silencia con `warnings.catch_warnings()` alrededor del bloque, preferible a un `filterwarnings('ignore')` global que ocultaría avisos posteriores que sí interesan.

## 3. Frágil: `ffmpeg` sin `-y`

```python
os.system(f"ffmpeg -i {video_filename} -vcodec libx264 {os.path.basename(video_filename)}")
```

El archivo de salida se escribe en el directorio actual con el mismo basename. La segunda vez que se ejecuta la celda con el mismo video, ffmpeg encuentra que el archivo ya existe y pregunta `Overwrite? [y/N]` esperando una respuesta que, con `os.system` en un kernel, puede no llegar nunca.

Y como `os.system` no verifica el código de retorno, si ffmpeg abortó el `open()` siguiente lee **el archivo de la corrida anterior**: el video equivocado junto a la predicción nueva. La corrección son tres caracteres: `-y`, más `-loglevel error` para no llenar el notebook con el banner.

## 4. Frágil: `reload_model` no puede fallar

Analizada en detalle en [la arqueología del checkpoint](03-la-arqueologia-del-checkpoint). En resumen: el filtro `if k in model_dict` descarta silenciosamente toda clave que no coincida, `model_dict.update()` mantiene todas las claves originales de modo que `strict=True` nunca detecta nada, y la función imprime "successfully loaded" pase lo que pase. Un checkpoint con nombres incompatibles daría una accuracy de azar sin una sola excepción.

## 5. Bug latente: `__len__` puede mentir

```python
self.filenames = glob.glob(...)
self.list = {}
for i, x in enumerate(self.filenames):
  target = x.split('/')[-3]
  for j, elem in enumerate(self.data_dir):
    if elem == target:
      self.list[i] = [x]
      self.list[i].append(j)
...
def __len__(self): return len(self.filenames)
```

Si una carpeta tuviera un nombre que **no está** en `label_sorted.txt`, el `if` nunca se cumple y `self.list[i]` jamás se asigna. Pero `__len__` sí lo cuenta. El `DataLoader` pediría ese índice y `__getitem__` reventaría con **`KeyError` a mitad de la evaluación**, después de cientos de clips. El fallo correcto sería al construir el dataset, no al consumirlo.

Además, ese doble bucle es **O(N × 500)**: con 2500 archivos son ~1,25 millones de comparaciones de string, resolubles en O(N) con un diccionario `{palabra: índice}`.

## 6. Bug latente: los paréntesis del `CenterCrop`

```python
x1 = int(round((w - tw))/2.)     # round(8)/2 = 4.0 → 4
#            ↑ el round se aplica a (w−tw), no a la división
x1 = int(round((w - tw)/2.))     # round(4.0) = 4    → 4   ← lo que se pretendía
```

Con 96−88 = 8, par, ambas dan 4 y no cambia nada. Con una diferencia impar —95→88, por ejemplo— la versión escrita da `int(7/2) = 3` (trunca) y la correcta da `round(3.5) = 4`. Un píxel de desalineación, inofensivo hoy y venenoso si alguien cambia el tamaño del recorte.

## 7. Subóptimo: `float64` en todo el camino del video

`arrays / 255.` en `load_video_file` promueve el `uint8` a **`float64`**, porque dividir por un float de Python duplica la precisión. Todo el `CenterCrop` y el `ColorNormalize` se computan en doble precisión, y recién al final del baile de dimensiones aparece `.float()` para bajar a `float32`. Son 2,1 MB por clip en vez de 1,1 MB, y el doble de trabajo aritmético. `np.float32(255)` lo habría evitado.

También: se guardan **3 canales RGB** en disco (~800 KB por clip) cuando el paso siguiente convierte todo a escala de grises. Dos tercios del almacenamiento de video son canales que nadie usa.

## 8. Subóptimo: el resto del inventario

- **`shuffle=True` en el test set** — innecesario, la accuracy no depende del orden. Su efecto colateral real es volver **no determinista** el bug del punto 1.
- **`num_workers=4` con 2 vCPU** — Colab emite el warning explícito. Hay sobresuscripción: los procesos compiten por dos núcleos y agregan overhead de cambio de contexto.
- **`batch_size=1`** — 2500 forwards independientes con la GPU mayormente ociosa entre transferencias.
- **`F.softmax` antes de `torch.max`** — softmax es monótona creciente; el argmax sobre logits es idéntico. Consume tiempo sin cambiar el resultado.
- **`running_loss` y `running_all`** se actualizan y nunca se leen. La línea de la pérdida está comentada, con un `loss.data[0]` que es sintaxis de PyTorch 0.3.
- **`predict()` sin `torch.no_grad()` ni `.eval()`** — construye el grafo de autograd para nada, y funciona solo porque el loop de evaluación dejó los modelos en modo `eval`.
- **`label_id` se calcula y se descarta** en `predict()`. La función conoce la respuesta correcta, no la usa y no la devuelve.
- **`self.list[idx][0] = self.list[idx][0]`** — asignación de una variable a sí misma.
- **`self.clean = 1/7.`** — nunca se usa. Es el vestigio más informativo del notebook: el `7` son las siete condiciones equiprobables de la augmentación de audio del paper — clean, 20, 15, 10, 5, 0 y −5 dB de *babble noise*.
- **`while cap.isOpened()`** — semánticamente incorrecto; `isOpened()` no cambia durante la lectura y quien corta es el `break` sobre `ret`. Si el archivo no se abre, sale con un array vacío sin excepción ni warning.
- **`math`, `random`, `sys`** importados y nunca usados: herencia del repo original, donde sí había augmentación aleatoria.

## 9. Discrepancias con el paper

- **ResNet v1, no v2.** El paper dice usar *"the 34-layer identity mapping version"* — pre-activación, la v2 de He et al. (2016). El código implementa post-activación, la v1 de 2015.
- **`AvgPool2d(2)` sobre un mapa 3×3** descarta la última fila y columna: **se usan 4 de 9 posiciones**. Está congelado en los pesos, así que no es corregible sin reentrenar (ver [Los dos streams](02-los-dos-streams)).
- **La asimetría 512 / 256** entre streams no aparece justificada en el paper.
- **El comentario `# average probability among frames`** describe un promedio de probabilidades cuando el código promedia logits — una media geométrica, no aritmética.

## 10. Cosmético que resultó no serlo

```python
print("reload LSTM model")   # el objeto es un GRU
```

Parecía un fósil de copia-pega. Resultó estar **literalmente en lo cierto**: al inspeccionar el checkpoint aparecen 36 claves de un `lstm.forwardModule1/2` y `lstm.backwardModule1/2` con formas `4 × hidden` — un BiLSTM completo de una versión anterior del código, guardado en el archivo. El print no era un descuido; describía el modelo que estos pesos tuvieron antes.

---

**Anterior:** [El vocabulario y los 29 errores](04-el-vocabulario-y-los-29-errores) · **Siguiente:** [Las tres actividades](06-las-tres-actividades)
