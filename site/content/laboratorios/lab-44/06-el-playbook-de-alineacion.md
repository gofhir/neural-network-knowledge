---
title: "El playbook de alineación"
weight: 6
math: true
---

El enunciado del laboratorio advierte que penaliza *"si no consigue alinear el video y la voz generada"*. Es el único criterio de calidad que menciona explícitamente, y el que más fácil se pierde: el modelo anima los labios según **el video de control**, no según el audio. Si el video no dice esa frase, la sincronía no existe.

Hay una forma de resolverlo que no requiere ninguna herramienta adicional.

## Grabar en playback sobre el audio ya generado

El orden natural sería grabar el video y después generar el audio. Invertirlo cambia el problema: **primero se genera el audio y después se graba el video haciendo playback de ese audio**. La sincronía deja de ser aproximada y pasa a ser exacta, porque el video copia el mismo archivo que después se le va a pegar.

Se puede hacer sin salir del notebook: una celda que reproduce el WAV y graba con la webcam **simultáneamente**, con márgenes de 0,7 s antes y después.

```python
import base64, os
from IPython.display import display, Javascript
from google.colab import output

audio_b64 = base64.b64encode(open('/content/generated.wav','rb').read()).decode()

def _guardar(data_url):
    raw = base64.b64decode(data_url.split(',')[1])
    open('/content/assets/mi_video.webm','wb').write(raw)
    return "ok"

output.register_callback('lab44.save_take', _guardar)

display(Javascript('''
(async () => {
  const audio = new Audio('data:audio/wav;base64,''' + audio_b64 + '''');
  const stream = await navigator.mediaDevices.getUserMedia({video:{width:1280,height:720}, audio:false});
  // ... boton, cuenta regresiva, preview ...
  const mr = new MediaRecorder(stream, {mimeType:'video/webm'});
  mr.start();
  await new Promise(r => setTimeout(r, 700));     // margen inicial
  audio.play();
  await new Promise(r => audio.onended = r);      // el audio marca el final
  await new Promise(r => setTimeout(r, 700));     // margen final
  mr.stop();
  // ... subir el blob al kernel ...
})();
'''))
```

Tres detalles que hacen que funcione:

- **`audio: false` en `getUserMedia`.** No se captura sonido: el video se usa solo por su movimiento, y así se evita el eco del parlante. De todos modos el pipeline lo descarta con `-an`.
- **`await audio.onended`** en vez de un temporizador fijo. La duración la marca el propio archivo, sin suponerla.
- **Un botón que dispara la grabación.** Los navegadores bloquean el autoplay de audio sin un gesto del usuario; ejecutar la celda no cuenta como tal.

Y los 0,7 s de margen no son arbitrarios: coinciden exactamente con el silencio que la utilidad del tercer notebook agrega al inicio del audio, así que las dos puntas calzan sin ajuste posterior.

## Calcular los frames desde la duración medida

El costo del denoising escala con los frames, así que conviene pedir los justos. La cuenta es directa, pero **hay que hacerla sobre la duración real del audio, no sobre una estimación**.

Estimar por palabras falla: con la voz de un discurso, el modelo produce **1,65 palabras/s** en vez de las ~2,5 conversacionales, porque [el conditioning captura prosodia](03-los-18-segundos). Y tres candidatos de la misma frase duraron 6,12, 7,05 y 6,87 s — **15 % de dispersión**.

Con la duración medida:

$$\text{frames} = \lceil \text{duración} \times \text{fps} \rceil \qquad 7{,}05\ \text{s} \times 16 = 113$$

## El fps no es el que dice el preset

Los presets del pipeline hablan de *"~3 segundos por segmento a 16 fps"*, pero el `--help` dice otra cosa:

> `--fps FPS  frame rate for BOTH driving-video resampling and the output. Leave unset to use the upstream default (30) consistently; setting only one of the two makes the result play at the wrong speed.`

Y ejecutando sin el flag, el plan de etapas **no lo incluye en ninguna parte**, así que cada etapa cae a su propio default. La corrida lo confirma:

```
[preprocess] driving 3840x2160 @ 24.00 fps -> 33 frames @ 30.00 fps
[vae_decode] wrote outputs/result.mp4 -- 33 frames at 480x480, 30 fps, 1.1s of video
```

Gana el `--help`. **El default es 30 fps**, y las notas de los presets están desactualizadas. Para 7,05 s de audio eso son 212 frames en vez de 113.

Conviene pasarlo explícito, y elegirlo con criterio:

| `--fps` | Frames para 7,05 s | Resampleo desde un origen de 30 fps | Costo relativo |
|---:|---:|---|---:|
| 16 | **113** | submuestreo limpio | **1,00×** |
| 24 | 169 | submuestreo | 1,50× |
| 30 (default) | 212 | ninguno | 1,88× |

Como el flag controla **las dos puntas** —resampleo de la entrada y fps de la salida—, pasarlo es seguro; lo que el `--help` advierte es desbalancearlas. Y 16 fps es el ritmo nativo de Wan.

## No alejarse para mostrar las manos

La tentación, al ver que la imagen de referencia tiene brazos, es retroceder para que entren en cuadro. Es un mal negocio, y el paper permite calcular por qué.

El Face Adapter *"redimensiona las imágenes de rostro a **512 × 512**"* (§3.3). El pipeline recorta el rostro frame a frame —eso es `src_face.mp4`— y lo escala a esa resolución. En un plano medio de 720 px de alto, un rostro ocupa unos 250 px:

$$\text{plano medio: } 250 \to 512 \ (\text{upsample } 2{,}0\times) \qquad \text{alejado: } 150 \to 512 \ (3{,}4\times)$$

Retroceder significa **interpolar píxeles que no existen para alimentar justo la señal que produce la sincronía labial y la expresión**. Se ganan brazos quietos y se pierde nitidez en lo único que un evaluador va a mirar de cerca.

Y hay un segundo argumento: las manos son donde el prior falla. El negative prompt oficial de Wan dedica **tres de sus ~25 términos** a manos —`多余的手指` (dedos de más), `画得不好的手部` (manos mal dibujadas), `手指融合` (dedos fusionados)—. Meterlas en cuadro para que estén *cruzadas y quietas* agrega el riesgo sin obtener movimiento a cambio.

## Hacer coincidir la expresión, no solo el encuadre

Este es el factor que más pesa en el parecido final, y el menos evidente.

![La imagen de referencia usada: retrato oficial recortado a 16:9, con sonrisa amplia](/laboratorios/lab-44/imagen-referencia.jpg)

Esa es la imagen de referencia de la corrida, y el problema está a la vista. Si la referencia tiene una **sonrisa amplia con dientes** y el video de control tiene expresión seria, el modelo debe transformar toda la mitad inferior del rostro: boca, mejillas, líneas de expresión, mandíbula. En esa transformación es donde se pierde la identidad — que es exactamente el riesgo que el paper señala:

> *"comprometen severamente la consistencia de identidad, especialmente en escenarios **cross-identity** con disparidad significativa de forma facial"*

La solución barata no es cambiar la actuación sino **elegir una imagen de referencia cuya expresión se parezca a la del video**. Cuesta cero y elimina el problema de raíz.

## La lista de verificación

Sobre el **video de control**:

| Criterio | Por qué |
|---|---|
| Horizontal 16:9 | El pipeline preserva el aspecto de la *imagen*, pero un video apaisado facilita el encuadre |
| Plano medio: cabeza, cuello y hombros | DWPose necesita esqueleto; solo cara deja muy pocos keypoints |
| Duración con margen sobre el audio | Los frames se recortan después; que sobren no cuesta |
| Quieto ~1 s al inicio | *"La pose se ancla al primer frame"*, y de ahí sale el retargeting de proporciones |
| Fondo liso, sin gente | El negative prompt penaliza `杂乱的背景` y `背景人很多` |
| Luz frontal, sin contraluz | DWPose corre en CPU sobre una silueta oscura y falla |

Sobre la **imagen de referencia**:

| Criterio | Por qué |
|---|---|
| Aspecto igual al que se quiere de salida | §4.3: *"el aspect ratio de salida se ajusta al de la imagen"* |
| **Expresión parecida a la del video** | Minimiza la transformación facial, que es donde se pierde el parecido |
| Encuadre parecido al del video | Si la imagen tiene brazos y el esqueleto no, el modelo decide solo |
| Fondo simple | En Animation Mode, **el fondo del resultado sale de la imagen** |
| Cara frontal, torso visible | El *pose retargeting* estima longitudes óseas desde la referencia |

Y sobre el **texto**: en el mismo registro que la voz de referencia. Un descriptor de discurso solemne diciendo una frase coloquial rápida está fuera de distribución.

## El padding que degrada el final

Un detalle que aparece al final del video y tiene causa exacta:

```
[preprocess] 128 frames -> 149 after padding to whole segments (2 x 77f, 5f overlap)
```

Se pidieron 128 frames, pero dos segmentos de 77 con 5 de solape cubren 149. **Los 21 frames finales son relleno**, no movimiento del video.

Y el DiT denoisea los 77 frames del segmento **como una sola secuencia con atención global**. De esos 77, los últimos 21 —el 27 %— llevan pose inventada, y vía atención contaminan a sus vecinos reales. Sumado al *drift* del encadenamiento, el efecto se observa como una desincronización creciente en el último tramo.

Se evita eligiendo un `--max-frames` que no requiera padding. Con `--frame-num 49` y `refert-num 5` el stride es 44, y los tamaños limpios son 49, 93, **137**. Con 136 frames disponibles, el relleno bajaría de 21 frames a 1.

Y hay una compensación afortunada: como el audio con el silencio inicial dura menos que el video generado, el `-shortest` del tercer notebook **recorta justamente ese tramo final**.

## El resultado de aplicarlo

Con el playbook completo, la corrida de este laboratorio no necesitó ninguna de las utilidades de corrección del tercer notebook:

| | |
|---|---|
| Audio generado | 7,05 s (medido, no estimado) |
| Silencio inicial | 0,70 s → **7,75 s** |
| Frames pedidos | 128 a 16 fps → **8,00 s** de video |
| Diferencia | 0,25 s de margen |

`setpts` no hizo falta. La utilidad de acelerar o ralentizar existe precisamente porque el flujo habitual es el inverso —generar primero un video corto y después estirarlo— y ahí el resultado se ve en cámara lenta.

---

**Siguiente:** [Los defectos de los notebooks](07-los-defectos-de-los-notebooks) — nueve problemas del código, y cuál de ellos conviene que falle.
