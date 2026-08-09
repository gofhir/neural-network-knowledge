---
title: "Transferencia y dominio"
weight: 4
math: true
---

Los anexos del laboratorio están marcados como material de apoyo que *"no se revisa durante la sesión"*, y sin embargo contienen la respuesta más nítida a la pregunta de la Actividad 4. Ejecutados en orden, producen un **gradiente medible**: el mismo modelo preentrenado vale +9 puntos, +34 puntos, o transcribe con cero errores, según cuánto se parezca la tarea a aquello con lo que aprendió.

## El resultado que ordena todo

| Situación | Coincidencia con el preentrenamiento de wav2vec 2.0 | Resultado |
|---|---|---|
| **GTZAN** — música, 22 kHz | ninguna | **+9,31 pp** sobre MFCC (28,62% vs 19,31%) |
| **Speech Commands** — habla en inglés, 16 kHz | idioma y modalidad | **+33,66 pp** sobre MFCC (79,83% vs 46,17%) |
| **VoxPopuli ES** — habla en español | modalidad; otro idioma y acento | ~3 errores en ~80 palabras |
| **LibriSpeech `dev-clean`** | total: corpus, idioma, tasa, estilo | **0,0% WER** |

**El preentrenamiento en habla vale 3,6 veces más cuando la tarea es habla.** Y en el caso de máxima coincidencia —darle un clip del corpus con el que fue entrenado— la transcripción es perfecta palabra por palabra.

Eso responde la pregunta conceptual de la actividad (*"¿esperarían el mismo resultado si la tarea fuera transcribir llamadas a texto?"*) con evidencia y no con intuición: **el valor de un modelo preentrenado es función de la distancia entre su dominio de preentrenamiento y la tarea**, y esa función se puede medir.

## Speech Commands: la comparación limpia

La celda opcional entrena el **mismo modelo** con las dos representaciones sobre 10 palabras habladas (3.000 clips de train, 600 de test, 5 épocas):

```
    MFCC: accuracy en test = 46.17%
 wav2vec: accuracy en test = 79.83%
```

Y es una comparación **mejor que la de GTZAN** en tres aspectos:

| | Speech Commands | GTZAN |
|---|---|---|
| Sample rate | **16 kHz = la tasa nativa de wav2vec** | 22.050 Hz → hay que resamplear |
| Audio que ve wav2vec | el clip completo (1 s) | **la mitad** (15 s de 30) |
| Submuestreo de la secuencia | ninguno | `[::4]` |
| Ventana del MFCC | **400 = 25 ms** (el estándar de ASR) | 8192 = 372 ms |

Las tres concesiones que GTZAN necesitaba desaparecen. Solo persiste la diferencia de parámetros de la GRU (322.610 contra 104.210).

{{< callout type="info" >}}
**Y acá el MFCC sí puede usar los parámetros canónicos** (`win_length=400`, `hop_length=160`, los **10 ms** de la Actividad 1). ¿Por qué acá sí y en GTZAN no? Porque los clips duran **1 segundo**, no 30: la secuencia queda en 101 pasos, manejable para la GRU. **El problema de GTZAN nunca fue el sample rate — era la duración del clip.**

También cambia el presupuesto de optimización: 235 pasos de gradiente (casi los mismos 230 de GTZAN) pero con **`lr=1e-3`, diez veces mayor**. Eso mitiga bastante el problema de saturación, porque el MFCC puede salir de la zona plana más rápido — y aun así pierde por 33 puntos.
{{< /callout >}}

Un detalle de implementación que vale copiar: la función que filtra el dataset usa `ds.get_metadata(i)` para leer la etiqueta **sin decodificar el audio**. Es el mismo principio que `file` en la Parte 1 — leer metadata es órdenes de magnitud más barato que decodificar, y con miles de archivos esa diferencia decide si el preprocesamiento tarda segundos o media hora.

## Los embeddings, vistos de cerca

```python
bundle = torchaudio.pipelines.WAV2VEC2_BASE
feats, _ = w2v.extract_features(y16)
# 12 capas de Transformer
# Ultima capa: (1, 64, 768) -> (batch, frames de ~20 ms, 768 dims)
```

El mismo audio que como MFCC era `(20, 108)` ahora es una secuencia de vectores de 768 dimensiones, uno cada ~20 ms. **No es más compacto: es más informado.** Cada vector salió de un Transformer entrenado con miles de horas de audio sin etiquetar, enmascarando trozos y aprendiendo a predecirlos — el objetivo de BERT trasladado al audio.

Y de las 12 capas, el pipeline de la Parte 2 usa **`feats[6]`, la séptima**. La razón que da el notebook es correcta y vale retenerla: **las capas finales están especializadas en el objetivo de preentrenamiento** (predecir unidades enmascaradas de habla), mientras que **las intermedias transfieren mejor** a tareas distintas. Es un resultado conocido de los estudios de probing sobre wav2vec 2.0, y explica por qué el modelo funciona razonablemente en música: en la capa 7 todavía codifica estructura acústica genérica —timbre, armonicidad, dinámica— y no fonemas.

## Transcripción con CTC

El anexo trae un decodificador CTC completo en seis líneas, y es el mismo algoritmo que ya apareció en el curso para leer texto en imágenes:

```python
def ctc_greedy(em, labels):
    out, prev = [], None
    for i in em.argmax(-1).tolist():
        if i != prev and labels[i] != "-":
            out.append(" " if labels[i] == "|" else labels[i])
        prev = i
    return "".join(out)
```

El modelo emite una distribución sobre caracteres **por cada frame de ~20 ms**. Para decir "hola" produce algo como `h h - o o - l l a a`, y el decodificador hace tres cosas en orden: **argmax** por frame, **colapsar repeticiones consecutivas**, y **eliminar el blank**.

{{< callout type="info" >}}
**El blank no es relleno: es lo que permite las letras dobles.** Sin él, "carro" se colapsaría a "caro". Con blank, el modelo emite `c a r - r o` y las dos erres sobreviven. El `|` cumple otro rol: separador de palabras, que la función convierte en espacio. Ver [CTC](/fundamentos/ctc-loss).
{{< /callout >}}

### En español: VoxPopuli

Con una muestra de habla natural descargada de Wikimedia Commons (41,9 s, español venezolano) y `VOXPOPULI_ASR_BASE_10K_ES`:

> el castellano venezolano es la variedad del idióma español utilizado en venezuela debido a que las instituciones venezolanas son específicas al respecto y se refiere al idioma nacional como castellano existe una preferencia marcada en el uso de esa denominación aunque no se rechaza el término español a su vez dentro del país se habla con distintos acentos y por la **ambigwedad** del término **de alecto** no se catalogan formalmente a nivel académico como tales

El **vocabulario de 35 símbolos** revela el diseño:

```
('-', '|', 'e', 'a', 'o', 's', 'n', 'r', 'i', 'l', 'd', 'c', 't', 'u', 'p', 'm', 'b', 'q',
 'y', 'g', 'v', 'h', 'ó', 'f', 'í', 'á', 'j', 'z', 'ñ', 'é', 'x', 'ú', 'k', 'w', 'ü')
```

- Composición exacta: **2 especiales + 26 letras + 5 acentuadas + ñ + ü = 35**.
- **El orden no es alfabético, es por frecuencia** en el corpus de entrenamiento (`e a o s n r i l d c t u...`, casi calcado de la frecuencia de letras del español).
- **No hay mayúsculas, puntuación ni dígitos** — por eso el texto sale plano. La puntuación es un modelo aparte.

Los errores tienen causas distintas y cada una enseña algo:

| Error | Debería ser | Causa |
|---|---|---|
| `ambigwedad` | ambigüedad | Confundió **ü** con **w**: los dos símbolos **menos frecuentes** del vocabulario (posiciones 34 y 35 de 35). Error de cola larga |
| `de alecto` | de dialecto | Perdió una sílaba. El **greedy decoding** no tiene forma de recuperarla; un beam search con modelo de lenguaje difícilmente aceptaría "alecto" |
| `idióma` | idioma | Tilde de más — y en la corrida sobre 30 s el mismo fragmento dio `idiom` |

{{< callout type="warning" >}}
**Ese último error revela algo importante: la transcripción es global, no frame por frame.** El mismo fragmento de audio, en el segundo ~3, se transcribió `idiom` al procesar 30 segundos y `idióma` al procesar los 41,9 completos. La palabra está lejísimos del punto de corte.

La razón es que wav2vec 2.0 es un **Transformer con self-attention**: la representación de cada frame depende de *toda* la secuencia. Darle 12 segundos más de contexto cambió las activaciones internas de un frame del inicio.

Consecuencia práctica: **cómo trocees el audio afecta el resultado incluso en las partes que no tocaste.** En transcripción de llamadas largas, decidir dónde cortar no es un detalle de implementación — por eso los sistemas serios usan ventanas solapadas y fusionan, en vez de cortar en seco.
{{< /callout >}}

### En inglés: el caso de máxima coincidencia

Con `WAV2VEC2_ASR_BASE_960H` —afinado con las 960 horas de LibriSpeech— sobre el primer utterance de `dev-clean`:

```
REF: MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL
HYP: MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL

WER: 0.0%  (17 palabras de referencia)
```

**Cero errores.** Es el caso donde todo coincide: corpus, idioma, sample rate, estilo de habla. El modelo fue afinado con LibriSpeech y se le dio un clip de LibriSpeech.

{{< callout type="info" >}}
**Dos salvedades que evitan sobreinterpretar ese 0%:**

1. **`dev-clean` es, por construcción, el subconjunto fácil.** La división `clean`/`other` de LibriSpeech no se hizo a mano: los autores corrieron un modelo de referencia sobre todos los hablantes y los ordenaron por tasa de error. Los de menor WER quedaron en `clean`. Cualquier número reportado sobre `dev-clean` es optimista respecto de `dev-other`.
2. **Este clip es el ejemplo más citado del corpus** — el "hola mundo" del reconocimiento de voz, presente en incontables papers y tutoriales. Un 0% acá es el techo, no el promedio.
{{< /callout >}}

Detalles del corpus que explican el comportamiento de estos modelos: los transcripts vienen **en mayúsculas y sin puntuación**, y las abreviaturas están **expandidas** ("MISTER", no "MR."), porque el modelo predice sonidos y "MR." no tiene pronunciación derivable de sus caracteres. Es normalización de texto para ASR, y es la razón por la que los sistemas entrenados con LibriSpeech producen texto plano.

## Los otros datasets del anexo

### UrbanSound8K

8.732 clips de ≤4 s en 10 clases de sonido urbano ([Salamon et al., 2014](/papers/urbansound8k-salamon-2014)). El split de test que trae el `.tar.gz` son 1.747 clips — el 20% exacto del dataset.

**El nombre del archivo codifica la etiqueta.** El formato es `[freesoundID]-[classID]-[occurrenceID]-[sliceID].wav`, así que `62567-5-0-6` es el séptimo recorte del primer evento de la grabación 62567, clase **5 = `engine_idling`**.

Y ahí está el dato que importa para usar bien el dataset: los campos `occurrence` y `slice` significan que **varios clips vienen de la misma grabación original**. Por eso el dataset trae **10 folds predefinidos** y el paper insiste en usarlos tal cual — rehacer el split al azar pondría recortes del mismo audio en train y test. Es la misma fuga que amenaza a GTZAN, pero acá los autores dan la solución armada.

El desbalance medido:

| Clase | n | % | Tipo de sonido |
|---|---|---|---|
| drilling | 223 | 12,8% | continuo |
| *(6 clases más)* | 171-214 | 9,8-12,2% | continuo |
| **gun_shot** | **86** | **4,9%** | **impulsivo** |
| **car_horn** | **74** | **4,2%** | **impulsivo** |

**Las dos clases raras son exactamente las dos impulsivas**, y no por sesgo de recolección: de un motor en ralentí se recortan decenas de slices de 4 segundos, de un bocinazo salen uno o dos. **El desbalance es una consecuencia física de la duración del evento.**

Por qué importa para la métrica: un modelo puede **ignorar por completo las dos clases impulsivas** y aún aspirar al 90,9% de accuracy. Y son justo las clases que más importarían en vigilancia acústica. Es el caso donde hay que mirar **F1 macro o la matriz de confusión**, a diferencia de GTZAN donde las 10 clases estaban parejas.

Un detalle práctico: los clips vienen de Freesound con **sample rates heterogéneos**. Es el problema del batching elevado — no solo largos distintos, también tasas distintas.

### LibriSpeech

Derivado de audiolibros de dominio público ([Panayotov et al., 2015](/papers/librispeech-panayotov-2015)), a 16 kHz. El `dev-clean` que usa el anexo pesa ~340 MB.

Su dato más relevante para este laboratorio: **es el corpus con el que se preentrenó `WAV2VEC2_BASE`** — 960 horas. O sea que todo el arco de la Actividad 4 mide distancia respecto de *este* material.

El `Dataset` devuelve **seis** elementos (waveform, sample rate, transcript, speaker_id, chapter_id, utterance_id); el `*resto` del anexo descarta los tres últimos. Y `SPEAKERS.TXT` trae **ID, género y minutos grabados** por hablante — la tabla que permite auditar el sesgo demográfico de un corpus, la "parte humana" del dato que planteaba la clase. Con ella se puede responder si el balance de género es parejo y si un puñado de personas concentra las horas (en cuyo caso el modelo aprende sus voces).

### MusicNet, y el catálogo

**MusicNet** queda comentado a propósito: ~20 minutos solo entre descarga y descompresión. Lo valioso es su advertencia — sus etiquetas están **alineadas al sample** (cada nota, con su instrumento, ubicada en el tiempo) y **el final de los archivos es silencio sin anotar**. Las etiquetas temporales pueden mentir por omisión.

El catálogo final lista lo que trae `torchaudio.datasets` y un dato de contexto que conviene tener: **torchaudio entró en fase de mantención en 2025**, así que esa lista ya no crece y el punto de encuentro actual para datasets nuevos es Hugging Face. Con una nota práctica sobre **Common Voice**, el dataset más relevante si se necesita español real con metadata demográfica: la clase de torchaudio **ya no descarga sola** — Mozilla exige registro y descarga manual.

## Qué nos llevamos

- **La transferencia se puede medir, y es un gradiente**: +9 pp en música, +34 pp en habla, 0% de WER en el corpus nativo.
- **Las capas intermedias transfieren mejor** que las finales, porque las últimas están especializadas en el objetivo de preentrenamiento.
- **CTC son tres reglas** (argmax, colapsar, quitar el blank) y el blank existe para permitir letras dobles.
- **Un Transformer transcribe globalmente**: cambiar el largo del audio cambia predicciones en partes que no se tocaron.
- **El desbalance de un dataset puede tener causa física**, y entonces no se arregla recolectando más.
- **Los nombres de archivo y la metadata son datos**: leerlos evita decodificar, y a veces revelan la estructura de fugas del dataset.

---

**Ver tambien:** [Lab 37 — hub](/laboratorios/lab-37) · Anterior: [GTZAN: MFCC vs wav2vec](03-gtzan-mfcc-vs-wav2vec) · Papers: [wav2vec 2.0](/papers/wav2vec2-baevski-2020) · [LibriSpeech](/papers/librispeech-panayotov-2015) · [UrbanSound8K](/papers/urbansound8k-salamon-2014) · [Speech Commands](/papers/speech-commands-warden-2018) · [Common Voice](/papers/common-voice-ardila-2020) · Fundamentos: [Datasets de audio](/fundamentos/datasets-de-audio) · [CTC](/fundamentos/ctc-loss).
