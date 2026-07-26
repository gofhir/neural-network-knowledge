---
title: "Datasets de audio"
weight: 124
math: true
---

El salto de los últimos años en audio **no fue de algoritmos: fue de escala de datos**. Los mismos modelos, entrenados con órdenes de magnitud más audio, resolvieron tareas antes inabordables. Por eso, elegir bien el dataset —y entender de dónde sale, qué licencia tiene y a quién representa— es una de las decisiones más importantes de un proyecto de audio. Este fundamento acompaña a la [Clase 37](/clases/clase-37) y recorre el mapa de los datasets de audio: sus tres escalas, los ejes para elegir uno, y las tres grandes familias de problemas que cubren.

---

## 1. Tres familias de problemas de audio

Casi todo problema de audio cae, al menos, en una de estas tres:

- **Música** — géneros, transcripción, separación de fuentes, recomendación. Ej.: Shazam.
- **Habla** — reconocer (ASR), sintetizar (TTS), identificar al hablante (biometría/diarización). Ej.: Siri.
- **Ambiente** — alarmas, máquinas, fauna, ciudad. Ej.: un detector de fallas industriales.

En la práctica **se mezclan**: música que tapa una voz, una máquina que falla bajo el ruido de la fábrica —el audio real casi nunca viene puro.

---

## 2. Tres escalas de datasets

{{< concept-alert type="clave" >}}
Los datasets de audio vienen en **tres tamaños**, y cada uno sirve para una etapa distinta del trabajo.
{{< /concept-alert >}}

**Didácticos** (miles de clips). Para aprender y prototipar sin GPU. Se bajan rápido y alcanzan para un proyecto pequeño:

| Dataset | Qué trae |
|---|---|
| **[GTZAN](/papers/gtzan-tzanetakis-2002)** | 1.000 temas de 30 s, 10 géneros musicales (el del [laboratorio](/laboratorios/lab-37)) |
| **[UrbanSound8K](/papers/urbansound8k-salamon-2014)** | 8.732 clips de ≤4 s, 10 clases urbanas, en **10 folds** |
| **[ESC-50](/papers/esc50-piczak-2015)** | 2.000 clips de 5 s, 50 clases de sonidos ambientales |
| **[SpeechCommands](/papers/speech-commands-warden-2018)** | ~65.000 clips de 1 s, 30 palabras cortas |

**Benchmarks** (cientos/miles de horas). Estándares de investigación:

| Dataset | Qué trae |
|---|---|
| **[LibriSpeech](/papers/librispeech-panayotov-2015)** | ~1.000 h de habla leída en inglés (audiolibros), en FLAC |
| **[MusicNet](/papers/musicnet-thickstun-2017)** | Música clásica con etiquetas alineadas **nota a nota** |
| **[AudioSet](/papers/audioset-gemmeke-2017)** | ~2M clips de YouTube, ontología de ~600 eventos |
| **[FSD50K](/papers/fsd50k-fonseca-2020)** | ~51.000 clips CC, multi-etiqueta, 200 clases |

**Escala web** (cientos de miles de horas). La frontera actual: **[Common Voice](/papers/common-voice-ardila-2020)**, GigaSpeech, Emilia. Para dimensionar: **[Whisper](/papers/whisper-radford-2022)** no se entrenó con ninguno de los anteriores —usó **~680.000 horas de audio web débilmente etiquetado**, y en versiones posteriores subió a ~1M de horas. Ahí está la frontera hoy.

---

## 3. Cómo elegir un dataset: cuatro ejes

Más que los nombres, cuatro ejes deciden si un dataset sirve:

- **Escala vs. limpieza.** Etiquetado a mano (limpio, chico) vs. web débilmente etiquetado (sucio, gigante). La frontera se fue por la escala —pero lo sucio trae sesgos y ruido de etiquetas.
- **Disponibilidad.** ¿Te dan **el audio** (FSD50K, en CC) o solo un **link de YouTube que se cae** (AudioSet)? El *link rot* es un problema real de reproducibilidad.
- **Licencia y privacidad.** Lo mejor suele ser pagado (LDC) o con copyright. La voz es **dato personal**. Lo abierto suele ser Creative Commons o instrumental.
- **Representación / sesgo.** ¿Qué idiomas, acentos y géneros están, y cuáles no?

{{< concept-alert type="advertencia" >}}
El **sesgo** no es un detalle académico: un ASR entrenado solo en inglés "limpio" **falla con los clientes reales**. Y no existe, por ejemplo, un dataset público de llamadas telefónicas chilenas —así que el flujo real es **pre-entrenar con datos públicos y hacer fine-tune con los datos propios** del dominio. La cobertura desigual de idiomas y acentos (el problema que [Common Voice](/papers/common-voice-ardila-2020) intenta mitigar) determina con quién funciona un sistema y con quién no.
{{< /concept-alert >}}

---

## 4. Comprimir o no el dataset

Para entrenar, la decisión de compresión depende de si el cuello de botella es el **disco** o la **CPU**:

- **Sin comprimir (WAV):** exacto y determinista, sin costo de CPU al cargar; pero pesa muchísimo y genera más I/O por época.
- **Comprimido (FLAC/MP3):** mucho menos disco y transferencia; si el cuello es el I/O, más rápido. El costo es decodificar en cada carga.

En la práctica, **decodificar es barato** (~1.000× tiempo real), así que el códec casi nunca es el cuello. Por eso los datasets grandes se distribuyen comprimidos y se decodifican al vuelo: LibriSpeech en FLAC, GigaSpeech en Opus, Common Voice en MP3. La decisión real es la anterior: **¿la tarea tolera pérdida?** ([representación de audio](/fundamentos/representacion-de-audio)).

---

## 5. La parte humana

Un dataset de audio es también un objeto legal y ético. La clase insiste: grabar voz exige **consentimiento** (la voz es biométrica; las llamadas contienen PII); la música casi siempre tiene **copyright**; y hay que vigilar el **sesgo** de cobertura. Las buenas prácticas —anonimizar, minimizar, **documentar el origen**— ahorran problemas legales y, de paso, mejoran el modelo.

---

## 6. Relevancia para salud

En salud, los cuatro ejes se vuelven críticos. La **disponibilidad** y las **licencias** son un cuello de botella severo: los datos clínicos rara vez son abiertos, y la voz de un paciente es dato personal sensible. El **sesgo** puede costar vidas —un modelo de voz clínica que no cubra los acentos, idiomas y perfiles demográficos de los pacientes reales falla justo con los más vulnerables, un problema de **equidad en salud digital**. Y como no suele existir un dataset público del dominio exacto (p. ej., tos de pacientes de un hospital específico), el patrón dominante es el mismo que en el call center: **pre-entrenar con datos públicos** (o con modelos como wav2vec/Whisper) y **hacer fine-tune con los datos propios** de la institución, con todo el cuidado ético que eso exige.

---

## Referencias

- Fundamentos relacionados: [Representación de audio](/fundamentos/representacion-de-audio) · [Data augmentation de audio](/fundamentos/data-augmentation-de-audio) · [Aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado).
- Dominio: [Audio / Voz](/dominios/audio).
