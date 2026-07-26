---
title: "Speech Commands: keyword spotting (2018)"
weight: 411
math: true
---

{{< paper-card
    title="Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition"
    authors="Pete Warden (Google)"
    year="2018"
    venue="arXiv:1804.03209"
    pdf="/papers/speech-commands-warden-2018.pdf" >}}
Speech Commands no propone una arquitectura ni un algoritmo: propone un **dataset estandarizado y un protocolo de evaluación reproducible** para el *keyword spotting* (detección de palabras clave), la tarea que dispara las interfaces de voz ("Hey Google") **on-device**, antes de enviar audio alguno a la nube. Publicado por Pete Warden (Google Brain) con licencia **CC BY 4.0**, la versión 2 documentada en el paper reúne **105.829 grabaciones de 35 palabras**, aportadas por **2.618 hablantes**, cada clip de **un segundo**, PCM lineal de 16 bits, mono, a **16 kHz**. Su tesis es una analogía con **ImageNet**: un benchmark abierto y compartido habilita comparaciones manzana-con-manzana y acelera todo el subcampo. Es, en la práctica, el "MNIST del audio" para keyword spotting, y el dataset didáctico del laboratorio de la [Clase 37](/clases/clase-37) en el anexo "wav2vec 2.0 sobre Speech Commands".
{{< /paper-card >}}

---

## Contexto: keyword spotting on-device y la ausencia de un benchmark abierto

El reconocimiento de voz completo (ASR) transcribe oraciones y, en interfaces comerciales, corre en un servidor. Pero la **detección inicial** del comando de activación no puede correr en la nube: exigiría enviar audio de todos los dispositivos todo el tiempo, algo costosísimo y con enormes riesgos de privacidad. Por eso el módulo que escucha continuamente el micrófono corre **localmente en el teléfono o dispositivo**, y solo cuando detecta un disparo probable comienza la transferencia al servicio web.

Esa arquitectura impone restricciones duras: el modelo debe ser **pequeño** y eficiente en energía (corre sin parar), la mayoría de su entrada es **silencio o ruido** —no habla— así que los falsos positivos deben minimizarse, y buena parte del habla que recibe es **ajena a la interfaz**, así que no debe dispararse ante habla arbitraria. La unidad de reconocimiento es **una palabra corta**, no una oración.

El problema era que ningún dataset existente encajaba. Warden revisa Mozilla Common Voice, LibriSpeech, TIDIGITS y CHiME-5: todos alineados a nivel de oración, o con licencias restrictivas, o en formatos difíciles de decodificar. Ninguno combinaba **licencia abierta y sin fricción**, **palabras aisladas alineadas a nivel de palabra**, **muchos hablantes** y **condiciones ruidosas realistas**. Ese es exactamente el nicho que Speech Commands vino a ocupar.

## Composición: el dataset más el protocolo

La contribución tiene dos mitades inseparables. Primero, **el dataset**: audio crowdsourced de palabras cortas, capturado con micrófonos de teléfono o laptop en entornos ruidosos, con licencia CC BY 4.0. El vocabulario combina tres tipos de contenido —el corazón del diseño: **palabras de control** (los dígitos "zero" a "nine" más "Yes", "No", "Up", "Down", "Left", "Right", "On", "Off", "Stop", "Go"), **palabras auxiliares o desconocidas** ("Bed", "Cat", "Marvin", "Tree"…, algunas elegidas por sonar parecido a las objetivo, como "Tree" frente a "Three") y **silencio/ruido de fondo**. La captura se hizo con una app web open-source (WebAudioAPI) que mostraba palabras en orden aleatorio; el audio volvía en OGG, luego convertido a WAV a 16 kHz.

Segundo, **una metodología de evaluación reproducible**. El punto metodológico clave: el conjunto (train / validation / test) al que pertenece un archivo se decide aplicando una **función de hash estable sobre su nombre**. Para un archivo con nombre $n$, se compara $h(n) \bmod N$ contra umbrales que reflejan los porcentajes deseados de validación y prueba. Como $h$ no depende del total de archivos, **agregar nuevas grabaciones no reasigna las existentes**: los archivos permanecen en el mismo conjunto entre versiones, evitando la contaminación cruzada cuando se prueban modelos viejos sobre datos nuevos. El download incluye `validation_list.txt` y `testing_list.txt`.

Un detalle de limpieza que vale como lección: la herramienta **Extract Loudest Section** estima el volumen sumando las diferencias absolutas de las muestras respecto de cero (escala donde $-32768 \to -1.0$ y $+32767 \to +1.0$); los clips por debajo de 0.004 se descartan por inaudibles, y del resto se extrae el subclip de un segundo con mayor volumen, centrando la palabra.

## Impacto

Speech Commands se convirtió en el **benchmark estándar de facto del keyword spotting** por una combinación de factores diseñados a propósito: licencia sin fricción, formato trivial de procesar, splits reproducibles por hash, baselines y código de referencia listos en el tutorial de TensorFlow, y condiciones realistas de producto. Su bajo costo de entrada —un modelo decente se entrena en minutos— lo volvió el equivalente en audio del MNIST/ImageNet. El baseline reportado para el mejor modelo convolucional (basado en Sainath y Parada 2015) es de **88,2%** de Top-One sobre V2; entrenado en V2 pero evaluado sobre el test de V1 sube a 89,7%, evidencia de que los datos de entrenamiento V2 aportan una mejora sustancial sobre V1 (85,4%). Para streaming, sobre un clip de prueba continuo, el baseline logra 49,0% *matched* con 0,0% de falsos positivos.

## Limitaciones

- **Solo inglés.** Decisión pragmática de control de calidad; Warden abre el código de recolección para otras lenguas, pero el dataset sesga hacia el inglés y, por diseño de privacidad, no documenta la distribución demográfica.
- **Vocabulario minúsculo y palabras aisladas.** Los clips de un segundo excluyen habla continua y palabras largas: no sirve para ASR completo.
- **Etiquetado de un solo revisor.** Cada clip de la revisión manual fue verificado por un único trabajador, dejando posible ruido de etiqueta.
- **Alineación aproximada por volumen.** Extract Loudest Section asume que la palabra es la parte más ruidosa del clip; con ruido de fondo fuerte esa suposición puede fallar.
- **No propone arquitectura.** El baseline modesto es intencional: el paper aporta el terreno de juego, no el jugador estrella.

## Por qué importa para la Clase 37

En la [Clase 37](/clases/clase-37) (Datasets y Herramientas para Audio), Speech Commands es el ejemplo canónico de cómo un dataset abierto, pequeño y bien especificado —con splits reproducibles y baselines— dispara la investigación de todo un subcampo, replicando el rol de ImageNet en visión (ver [datasets de audio](/fundamentos/datasets-de-audio)). La descripción de "65.000 clips de 1 s, 30 palabras" que la clase cita corresponde a la **versión 1** (64.727 utterances); este paper documenta la **versión 2**.

En el **laboratorio**, el anexo "wav2vec 2.0 sobre Speech Commands" cierra el arco histórico del audio. Speech Commands nació (2017-2018) en la era del keyword spotting con **CNN de small footprint** entrenadas desde cero sobre espectrogramas/MFCC, con baseline del orden del 88%. [wav2vec 2.0](/papers/wav2vec2-baevski-2020) (2020) representa el paradigma opuesto: un modelo **auto-supervisado** pre-entrenado sobre miles de horas de audio sin etiquetar, luego afinado sobre un dataset pequeño y etiquetado como este. La **estabilidad del hashing** garantiza que la comparación entre la CNN de 2018 y el wav2vec 2.0 del lab sea justa: ambos ven exactamente los mismos archivos en train, validation y test.

El diseño on-device es directamente pertinente para dispositivos médicos y asistivos con control por voz: un keyword spotter que corre localmente —pequeño, robusto al ruido, con mínimos falsos positivos y baja latencia— permite comandar equipos clínicos o interfaces de accesibilidad sin enviar audio continuo a la nube, preservando la privacidad del paciente por procesamiento local y respondiendo a comandos como "stop" o "up" incluso sin conectividad.
