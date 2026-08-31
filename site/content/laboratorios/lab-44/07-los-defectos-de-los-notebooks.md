---
title: "Los defectos de los notebooks"
weight: 7
math: true
---

Los tres notebooks arrastran capas de historia: el primero es un flujo de `RealTimeVoiceCloning` reescrito sobre TorToise (de ahí el `v18` del nombre), y el tercero conserva utilidades pensadas para [FOMM](/papers/fomm-siarohin-2019) que el pipeline actual ya no necesita. Conviene separar lo que está roto de lo que solo es viejo.

| Categoría | Qué significa |
|---|---|
| **Bug** | El código hace algo distinto de lo que pretende |
| **Trampa** | Funciona, pero falla de forma confusa ante un uso razonable |
| **Fósil** | Residuo de una versión anterior; inofensivo |

---

## 1 · Bug — ffmpeg escribiendo sobre su propia entrada

**Notebook 3, utilidad de silencio.**

```bash
!ffmpeg -y ... -i /content/generated2.wav -filter_complex "[1:a][0:a]concat=..." /content/generated2.wav
```

El mismo archivo como entrada y como salida. Con `-y`, ffmpeg trunca el output al abrirlo mientras lo está leyendo. Afortunadamente lo detecta y aborta:

```
Output /content/generated2.wav same as Input #1 - exiting
```

El archivo sobrevive intacto —la salida nunca llega a abrirse— pero **el relleno de silencio final no se aplica**. Lo correcto sería escribir a `generated3.wav`.

{{< concept-alert type="clave" >}}
**Y en este caso conviene que falle.** El comentario del notebook justifica ese relleno *"de tal manera que el audio tenga duración >= que el video, pues sino el FFMPEG dejaría el video del tamaño del más corto"*. Es correcto **en el caso contrario**: cuando el audio es más corto que el video.

Con audio de 7,75 s y video de 8,00 s, agregar 10 s de silencio al final **conserva el tramo degradado** por el padding del segundo segmento. Sin el relleno, `-shortest` lo recorta solo.

Un bug que produce el resultado correcto por accidente sigue siendo un bug — pero conviene saber en qué dirección falla.
{{< /concept-alert >}}

## 2 · Trampa — `concat` con formatos heterogéneos

Misma celda, primer comando:

```bash
-f lavfi -t 0.7 -i anullsrc=channel_layout=stereo:sample_rate=44100 -i /content/generated.wav \
-filter_complex "[0:a][1:a]concat=n=2:v=0:a=1"
```

| | canales | sample rate |
|---|---|---|
| `anullsrc` | estéreo | 44.100 Hz |
| salida de TorToise | mono | 24.000 Hz |

El filtro `concat` exige formato uniforme entre segmentos. ffmpeg inserta un `aresample` implícito y preserva el formato original, como confirma el bitrate reportado:

$$386{,}7\ \text{kbit/s} \approx 24.000 \times 16 \times 1 = 384\ \text{kbps}$$

Funciona, pero depende de un comportamiento no garantizado. La alternativa limpia es una sola pasada, sin concatenar fuentes ni tocar el formato:

```bash
ffmpeg -y -i generated.wav -af "adelay=700:all=1" audio.wav
```

## 3 · Trampa — `-shortest` no corta con `-c:v copy`

**Notebook 3, celda de unión.**

```bash
!ffmpeg -y -i /content/video.mp4 -i /content/audio.wav -c:v copy -c:a aac -shortest /content/final.mp4
```

Medido por stream sobre el resultado:

| | contenedor | video | audio |
|---|---:|---:|---:|
| `final.mp4` | 8,000 s | 7,9375 s (127 frames) | 7,750667 s (183 frames AAC) |
| tras `-t 7.75 -c copy` | 8,000 s | 7,875 s (126 frames) | 7,750667 s |

`-shortest` **sí actuó** —recortó un frame de los 128 generados— pero no puede cortar a la duración exacta del audio: en modo *stream copy* ffmpeg remultiplexa sin decodificar y solo corta en límites de paquete. Un recorte explícito con `-t 7.75` deja 126 frames, no los 124 que corresponderían a 7,75 s exactos.

El desajuste efectivo son **3 frames (0,19 s)** de video sin audio al final.

Contribuye además el padding del encoder AAC, que codifica en frames de 1024 muestras:

$$7{,}75 \times 24000 = 186.000 \to 182 \times 1024 = 186.368\ \text{muestras}$$

Un corte al instante exacto requeriría recodificar (`-c:v libx264`).

## 4 · Trampa — la duración del contenedor miente

Del mismo cuadro: **ningún stream llega a 8,000 s** (el mayor es 7,9375) y sin embargo `format=duration` reporta 8,000 en los dos archivos.

Ese valor sale de la cabecera `mvhd` del MP4, heredada del video de entrada de 128 frames y **no recalculada al remultiplexar**. Es un tropiezo clásico al trabajar con `-c copy`: *el metadato de duración no es el contenido*. Medir por stream (`stream=duration,nb_frames`) da la respuesta real.

## 5 · Bug — el archivo de entrada que ningún notebook crea

**Notebook 3, utilidad de frame rate.**

```bash
!ffmpeg ... -i /content/input_video.mp4 ... /content/input_video2.mp4
```

`/content/input_video.mp4` **no se crea en ninguno de los tres notebooks**. Ejecutar la celda tal cual produce tres `No such file or directory`.

Es un fósil del flujo original con FOMM, donde el video generado salía a una velocidad distinta de la original. Con Wan-Animate el problema no existe porque `--fps` controla las dos puntas. Para dejar la celda ejecutada basta con darle un input:

```python
shutil.copy('/content/video.mp4', '/content/input_video.mp4')
```

## 6 · Trampa — `setpts` solo toca el video

Misma celda:

```bash
-filter:v "setpts=0.33333*PTS"
```

El `:v` significa que el filtro se aplica **solo al stream de video**. Sobre un archivo con audio, la imagen se acelera 3× y el sonido queda intacto: desincronización total. Para audio habría que usar `atempo`, que además está limitado a $[0{,}5, 2{,}0]$ por invocación y hay que encadenar:

```bash
-filter:v "setpts=0.33333*PTS" -filter:a "atempo=2.0,atempo=1.5"
```

En el laboratorio no muerde porque el video generado es mudo, pero es la clase de detalle que rompe en cuanto se reutiliza la utilidad.

*(De paso, el `-acodec aac` de esa celda es **inerte**: pide codificar en AAC una pista de audio que no existe. Mismo tipo de fósil que el `-ab 64000` sobre un WAV del primer notebook.)*

## 7 · Trampa — el par `fps` + `setpts`, bien pensado

No es un defecto, pero suele malinterpretarse:

$$\underbrace{\texttt{fps=90}}_{\text{sube la tasa}} \;\longrightarrow\; \underbrace{\texttt{setpts=0.33333*PTS}}_{\text{acelera } 3\times} \;\longrightarrow\; \frac{90}{3} = 30\ \text{fps efectivos}$$

`setpts` reescribe los timestamps: multiplicarlos por ⅓ hace que el video dure un tercio. Pero acelerar no crea frames — un video de 30 fps acelerado 3× deja **10 frames por segundo de contenido real** y se ve a tirones. Subiendo primero a 90, el filtro `fps` **duplica** frames y al acelerar se vuelve a 30 de material.

El matiz: `fps` duplica, **no interpola**. No inventa movimiento intermedio; para eso haría falta `minterpolate`, que estima flujo óptico y es órdenes de magnitud más caro.

## 8 · Trampa — rutas relativas por todas partes

**Notebook 1.** `tortoise/utils/audio.py`:

```python
def get_voices():
    subs = os.listdir('tortoise/voices')          # RUTA RELATIVA
```

Y en `api.py`, `models_dir='.models'` — también relativa. Ninguna de las dos se resuelve respecto del paquete instalado: dependen del **directorio de trabajo**.

Consecuencias concretas:

- `load_voice('obama')` funciona solo si el cwd es `/content/tortoise-tts`. Un `KeyError: 'obama'` **no significa que falten los clips**: significa que el proceso está en otro directorio.
- Si el runtime se reinicia, el cwd vuelve a `/content` y `TextToSpeech()` **vuelve a descargar los 4,3 GB** en otro lado.

Por eso la celda de instalación usa `%cd` (magic de IPython, que persiste) y no `!cd` (subshell que muere). El segundo notebook tiene el mismo patrón con `os.chdir(root)` + `sys.path.insert(0, ...)`.

## 9 · Bug — el guard que mira el archivo equivocado

**Notebook 1, descarga del audio de referencia.**

```python
if not os.path.exists("/content/obama_voice.mp3"):
    os.system('wget -q ... -O /content/obama_voice.mp3')
    os.system('ffmpeg ... /content/reference_voice.wav')
    display(Audio(...))
```

El guard verifica el **MP3**, pero el archivo que necesitan las celdas siguientes es el **WAV**. Si el `wget` funciona y el `ffmpeg` falla —disco lleno, MP3 corrupto—, en la siguiente ejecución el MP3 ya existe, el bloque se salta entero y `reference_voice.wav` nunca se crea. El error aparece dos celdas después y apunta al lugar equivocado.

Y una molestia menor del mismo bloque: **el `display()` está dentro del `if`**, así que al re-ejecutar la celda no aparece ningún reproductor. Parece que no hizo nada.

## Y el contraste: cómo se hace bien

El mismo problema de "verificar existencia no es verificar integridad" aparece resuelto correctamente en `fetch_preprocess_models.py`:

```python
if target.exists() and target.stat().st_size == spec["size"]:   # tamaño EXACTO
    ...
def verify(key, path):
    if path.stat().st_size != spec["size"]:
        return (f"expected {spec['size']:,} bytes, got {actual:,}. "
                f"Delete it and retry; a partial file will fail inside onnxruntime instead.")
```

Con el porqué documentado: *"Un archivo corto aquí es una descarga truncada o un puntero LFS, y de otro modo aparecería mucho después como un error opaco de parseo de protobuf"*.

Es exactamente el mismo fallo que `download_models()` de TorToise deja pasar, donde un `.pth` truncado produce `PytorchStreamReader failed reading zip archive` varios minutos más tarde. **Dos calidades de ingeniería frente al mismo problema.**

## Un defecto de coordinación entre notebooks

El primer notebook cita en sus créditos el Colab de `RealTimeVoiceCloning` mientras el código clona `tortoise-tts`. Y el bloque de imports arrastra nueve nombres que ningún código posterior usa —`nn`, `F`, `sys`, `widgets`, `clear_output`, `io`, `Path`, `sf`, `load_voices`—, fósiles de cuando el notebook grababa del micrófono con `ipywidgets` y silenciaba logs con `IPython.utils.io`.

No rompe nada. Pero junto con el `v18` del nombre del archivo, cuenta la historia de cuántas veces hubo que parchar esto.

---

**Volver al** [índice del laboratorio](/laboratorios/lab-44).
