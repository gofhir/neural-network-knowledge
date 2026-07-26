# Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition*.
- **Autor:** Pete Warden, **Google Brain**, Mountain View, California.
- **Publicación:** preprint arXiv:1804.03209v1 [cs.CL], 9 de abril de 2018. Fechado en abril de 2018.
- **Licencia del dataset:** Creative Commons BY 4.0, lo que permite descargarlo e incorporarlo en tutoriales y scripts sin registro ni permiso previo.
- **Artefacto central:** un dataset de audio de palabras habladas, publicado en dos versiones (v1 en agosto de 2017, v2 documentada en este paper), pensado para entrenar y evaluar sistemas de *keyword spotting* (detección de palabras clave).

El paper no propone una arquitectura nueva de red neuronal ni un algoritmo de aprendizaje: propone un **dataset estandarizado y un protocolo de evaluación reproducible** para una clase acotada de tareas de reconocimiento de voz. Su meta explícita es ofrecer una manera de construir y probar modelos pequeños que detecten cuándo se pronuncia una única palabra, dentro de un conjunto de diez o menos palabras objetivo, con la menor cantidad posible de falsos positivos frente a ruido de fondo o habla no relacionada. Esa tarea se conoce como **keyword spotting**, y es la que dispara la interacción en interfaces de voz ("Hey Google", "Hey Siri") antes de que cualquier audio se envíe a la nube.

La contribución se apoya en una analogía histórica que el propio Warden invoca: así como **ImageNet** democratizó y aceleró la visión por computador al dar un punto de comparación abierto y compartido, un dataset abierto de comandos de voz puede fomentar colaboraciones entre grupos y habilitar **comparaciones manzana-con-manzana** entre distintos enfoques, ayudando a que todo el campo avance. La versión final del dataset consta de **105.829 grabaciones (utterances) de 35 palabras**, aportadas por **2.618 hablantes**, cada clip de un segundo, PCM lineal de 16 bits, mono, a 16 kHz.

Para la **Clase 37 (Datasets y Herramientas para Audio)** este trabajo es el ejemplo canónico de un dataset didáctico de audio: se descarga sin fricción, se procesa en minutos y sostiene el laboratorio de la clase, donde reaparece en el anexo "wav2vec 2.0 sobre Speech Commands". Speech Commands es, en la práctica, el "MNIST del audio" para keyword spotting.

## 2. Contexto: keyword spotting on-device y la ausencia de un benchmark abierto

### 2.1. Por qué el keyword spotting es distinto del ASR completo

El reconocimiento de voz tradicional (ASR, *automatic speech recognition*) transcribe oraciones completas y, en interfaces comerciales, corre en un servidor una vez que ya se detectó el comienzo de una interacción. Ese modelo de servidor solo está limitado por consideraciones comerciales, porque los recursos del cómputo los controla el proveedor de nube.

El problema es que la **detección inicial** del comando de activación no puede correr en la nube: exigiría enviar audio de todos los dispositivos todo el tiempo, lo que sería costosísimo de mantener y aumentaría los riesgos de privacidad. Por eso, la mayoría de las interfaces de voz corren un módulo de reconocimiento **localmente en el teléfono o dispositivo**, escuchando continuamente el micrófono a la espera de la frase disparadora. Recién cuando se detecta un disparo probable comienza la transferencia de audio al servicio web.

Esta arquitectura on-device impone **restricciones duras de recursos**: los procesadores móviles tienen mucha menos capacidad que un servidor, así que el modelo local **debe requerir menos cálculos** para responder en tiempo casi real; la batería limitada obliga a **eficiencia energética** para algo que corre continuamente (los equipos enchufados no sufren batería pero sí restricciones térmicas y programas tipo EnergyStar); y como la latencia de red es variable, un **acuse local inmediato** del comando mejora la experiencia.

De estas restricciones surge un perfil de tarea que Warden contrasta con el ASR de servidor: los modelos deben ser **más pequeños** y eficientes en energía; la mayoría de su entrada es **silencio o ruido**, no habla, así que los falsos positivos deben minimizarse; la mayoría del habla que reciben será **ajena a la interfaz**, así que no deben dispararse ante habla arbitraria; y la unidad de reconocimiento es **una palabra corta**, no una oración.

### 2.2. El vacío que llenaba

El campo del reconocimiento de voz había requerido tradicionalmente los recursos de grandes organizaciones (universidades, corporaciones) con acceso a datasets académicos vía consorcios como el **Linguistic Data Consortium (LDC)** o a datos comerciales propietarios. A medida que la tecnología maduró, la cantidad de gente que quería entrenar modelos creció más allá de esos grupos, pero la disponibilidad de datos no se amplió al mismo ritmo.

Los datasets existentes de habla no encajaban en keyword spotting. El paper revisa varios: **Mozilla Common Voice** (>500 h, 20.000 personas, CC-0; alineado por oración), **LibriSpeech** (1.000 h de habla leída, CC BY 4.0; solo alineación a nivel de oración, más apto para ASR completo), **TIDIGITS** (25.000 secuencias de dígitos, 300 hablantes, sala silenciosa; solo licencia comercial LDC en formato NIST SPHERE difícil de decodificar —fue el dataset de los experimentos iniciales de Warden) y **CHiME-5** (50 h en hogares, WAV 16 kHz, licencia restringida, alineado a nivel de oración).

Ninguno combinaba **licencia abierta y sin fricción**, **palabras aisladas alineadas a nivel de palabra**, **muchos hablantes** y **condiciones ruidosas realistas**. Ese es exactamente el nicho que Speech Commands vino a ocupar.

## 3. Contribución: el dataset más el protocolo

La contribución tiene dos mitades inseparables:

1. **El dataset en sí.** Audio crowdsourced de palabras cortas, capturado en condiciones realistas (micrófonos de teléfono/laptop, entornos ruidosos), con licencia CC BY 4.0 que lo hace trivial de incorporar en tutoriales y utilizable en entornos comerciales sin trabas legales.
2. **Una metodología de evaluación reproducible.** Splits de entrenamiento/validación/prueba definidos por hash del nombre de archivo (estables entre versiones), métricas Top-One y métricas de streaming, y baselines públicas. Sin un protocolo común, las cifras de accuracy de distintos papers no serían comparables.

Warden identifica dos audiencias más allá de los investigadores de ML. La primera son los propios autores de modelos, que pueden demostrar la accuracy de sus arquitecturas con métricas comparables y reproducir baselines entrenando sobre datos idénticos. La segunda, menos obvia, son los **fabricantes de hardware**: usando una tarea pública que refleja de cerca los requisitos de producto, los proveedores de chips pueden demostrar la accuracy y el consumo energético de sus ofertas de manera comparable, habilitando un **co-diseño virtuoso entre machine learning y hardware**.

## 4. Método: construcción del dataset

### 4.1. Requisitos de diseño

Warden fija varias decisiones de partida:

- **Audio realista, no de estudio.** El audio de estudio parecía poco realista: sin ruido de fondo, con micrófonos de alta calidad y en entorno formal. Un modelo exitoso debe lidiar con entornos ruidosos, equipo pobre y habla natural, así que **todas las grabaciones se capturaron con micrófonos de teléfono o laptop** dondequiera que estuvieran los usuarios (con la única salvedad, por privacidad, de grabar solos y sin conversaciones de fondo).
- **Clips de un segundo.** Para simplificar entrenamiento y evaluación, toda grabación se restringió a una **duración estándar de un segundo**. Excluye palabras largas, pero los objetivos de keyword spotting son cortos. Se grabaron **palabras únicas en aislamiento**, no dentro de oraciones, lo que se parece más a la tarea de palabra disparadora y facilita el etiquetado al no ser crucial la alineación.
- **Foco en inglés.** Por pragmatismo, para acotar el alcance y facilitar el control de calidad por nativos. Warden abrió el código de recolección para otras lenguas y confía en transfer learning; sí buscó **la mayor variedad de acentos posible**, consciente del sesgo hacia el inglés estadounidense.
- **Independencia del hablante.** Grabar tantas personas distintas como fuera posible, porque los modelos son mucho más útiles si son **speaker-independent** (personalizar a un individuo es intrusivo). Esto obligaba a un proceso de grabación rápido y fácil.
- **Privacidad por diseño.** No capturar información personalmente identificable: sin género ni etnia, sin inicio de sesión que enlace a datos personales, y con aceptación previa de un acuerdo de uso de datos.

### 4.2. Elección de vocabulario

Warden buscaba un vocabulario limitado (para que la captura fuera liviana) pero con suficiente variedad para ser útil. Eligió **veinte palabras comunes** como núcleo:

- Los **dígitos** "zero" a "nine".
- Diez **palabras de comando** útiles en aplicaciones de IoT o robótica (v1): "Yes", "No", "Up", "Down", "Left", "Right", "On", "Off", "Stop", "Go".
- En la **v2** se añadieron cuatro comandos más: "Backward", "Forward", "Follow", "Learn".

Uno de los problemas más difíciles del keyword spotting es **ignorar habla que no contiene disparadores**, así que se necesitaba un conjunto de palabras que sirviera para poner a prueba esa capacidad. Algunas se eligieron porque **suenan parecido** a palabras objetivo y son buenas pruebas de discernimiento (por ejemplo "Tree", cercana a "Three"); otras se eligieron arbitrariamente como palabras cortas que cubrieran muchos fonemas. La lista de estas palabras **auxiliares/desconocidas** fue: "Bed", "Bird", "Cat", "Dog", "Happy", "House", "Marvin", "Sheila", "Tree", "Wow" (más "Visual", presente en las frecuencias de la v2).

La distinción de tres tipos de contenido es el corazón del diseño: **palabras de control** (los objetivos reales), **palabras auxiliares o desconocidas** (para entrenar la capacidad de rechazar habla no objetivo) y **silencio/ruido de fondo** (para rechazar audio sin habla).

### 4.3. La aplicación de recolección (crowdsourcing)

La captura se hizo con una **aplicación web open-source** que grababa mediante la **WebAudioAPI**, soportada en navegadores de escritorio (Firefox, Chrome) y en Android, pero no en iOS. Warden evaluó apps nativas móviles pero los usuarios eran reacios a instalarlas por privacidad y seguridad; la experiencia web (que pide permiso de micrófono al sitio) tuvo mejor tasa de respuesta.

Flujo de la app: la página inicial pide aceptar **formal y explícitamente** participar (clic en "I Agree" guarda una cookie de sesión sin la cual no aparece la grabación; las subidas usan tokens **CSRF**). Al presionar "Record" se muestra una **palabra aleatoria** durante 1,5 s mientras se graba, con pausa de un segundo entre palabras; el **orden aleatorio** evita cambios de pronunciación por repetición. Las **palabras núcleo se muestran cinco veces** y las auxiliares **una sola vez**, para **135 utterances** por sesión (unos seis minutos). El audio vuelve en **OGG comprimido**, con el **session ID** (aleatorio, no ligado a cuenta) como prefijo; ese ID funciona como **identificador de hablante**, y una cookie impide repetir la sesión para asegurar buena distribución de hablantes.

Para reclutar voluntarios Warden usó redes sociales, y experimentó con **crowdsourcing pagado** para algunas grabaciones, aunque la mayoría viene del sitio abierto, alojado en un dominio de Google (`aiyprojects.withgoogle.com/open_speech_recording`) para dificultar spoofs de permisos de micrófono.

### 4.4. Procesamiento y control de calidad

El pipeline de limpieza tuvo varias capas:

1. **Filtro por tamaño.** Aprovechando OGG, los clips con muy poco audio pesan muy poco; se borraron archivos menores a **5 KB** por ser improbablemente correctos.
2. **Conversión.** Los OGG se convirtieron a **WAV PCM sin comprimir a 16 kHz** con `ffmpeg`; las muestras de otras fuentes se remuestrearon también a 16 kHz.
3. **Extract Loudest Section.** Muchas grabaciones seguían demasiado silenciosas y la palabra estaba mal alineada dentro del clip de 1,5 s. Warden creó una herramienta que estima el volumen sumando las diferencias absolutas de las muestras respecto de cero (escala donde $-32768 \to -1.0$ y $+32767 \to +1.0$) y promediándolas; cualquier clip **por debajo de 0.004** se descartó por inaudible. Luego extrae el **subclip de un segundo con mayor volumen**, lo que tiende a centrar la palabra en el medio.
4. **Revisión manual (crowdsourcing comercial).** Para filtrar palabras incorrectas o ininteligibles, se pidió a los trabajadores **escribir la palabra que oían** dando la lista esperada como ejemplo. Cada clip fue evaluado por **un solo trabajador**, y los que no coincidían con la etiqueta esperada se removieron.

### 4.5. Ruido de fondo y silencio

Un requisito clave del keyword spotting real es distinguir audio con habla de audio sin ella. Para entrenar y probar esa capacidad, se añadieron varios **archivos WAV de un minuto a 16 kHz con distintos tipos de ruido de fondo**. Algunos se grabaron directamente de entornos ruidosos (agua corriendo, maquinaria); otros se generaron matemáticamente con Python (`scipy` + biblioteca `acoustics`), produciendo **ruido blanco y ruido rosa**. Estos archivos se colocaron en una carpeta especial `_background_noise_` en la raíz del archivo. La categoría "Silence" de la evaluación se construye extrayendo clips de un segundo al azar de estos audios de fondo.

### 4.6. Splits reproducibles por hash

El punto metodológico más importante para la reproducibilidad. Los IDs hexadecimales de 16 dígitos de la app se **hashean** a IDs de 8 dígitos (igual para las fuentes de crowdsourcing pagado), removiendo toda conexión a IDs de trabajadores o información identificable. La función de hash es **estable**, así que en futuras versiones los IDs de archivos existentes se mantienen aunque se agreguen hablantes.

El conjunto (train / validation / test) al que pertenece un archivo se decide aplicando una **función de hash sobre su nombre**. El download incluye `validation_list.txt` (archivos para validar durante el entrenamiento, usables con frecuencia para ajustar hiperparámetros) y `testing_list.txt` (archivos que solo deben usarse para medir resultados finales, nunca para entrenar ni validar). La virtud del asignamiento por hash es que **los archivos permanecen en el mismo conjunto entre versiones**, aun cuando el total cambie, evitando la contaminación cruzada de conjuntos al probar modelos viejos sobre datos de prueba más recientes. La implementación en Python está en el código del tutorial de TensorFlow (`input_data.py`).

Formalmente, para un archivo con nombre $n$, el conjunto se asigna comparando $h(n) \bmod N$ contra umbrales que reflejan los porcentajes deseados de validación y prueba, donde $h$ es una función hash estable. Como $h$ no depende del total de archivos, agregar nuevas grabaciones no reasigna las existentes.

### 4.7. Propiedades finales

- **105.829 utterances** de **35 palabras** (Tabla 1 / Figura 1 del paper).
- **2.618 hablantes**, cada uno con ID hexadecimal único de ocho dígitos.
- Cada clip: WAV de **un segundo (o menos)**, PCM lineal de **16 bits**, mono, **16 kHz**.
- Tamaño en disco: **~3,8 GB** sin comprimir; **~2,7 GB** como tar comprimido con gzip.

Las frecuencias por palabra son deliberadamente desbalanceadas: las palabras núcleo/comando tienen del orden de 3.700–4.052 grabaciones (por mostrarse cinco veces por sesión) mientras que las auxiliares rondan 1.500–2.100 (por mostrarse una vez). Por ejemplo "Yes" y "Zero" tienen 4.044 y 4.052; "Follow", "Forward" y "Learn" (agregadas tarde en v2) tienen ~1.550–1.580.

## 5. Evaluación: métricas y baselines

### 5.1. Top-One Error

La métrica más simple: cuántas grabaciones del conjunto de prueba clasifica correctamente el modelo. A diferencia de la clasificación de imágenes tipo ImageNet, **no es obvio cómo ponderar las categorías**, porque el modelo debe además indicar cuándo no hay habla ("Silence") y cuándo se dijo una palabra que no reconoce ("Unknown Word"). Estas categorías de **"mundo abierto"** deben ponderarse según su ocurrencia esperada en una aplicación real.

El estándar del código de ejemplo de TensorFlow define **doce categorías**: las diez palabras "Yes", "No", "Up", "Down", "Left", "Right", "On", "Off", "Stop", "Go", más una etiqueta especial "Unknown Word" y otra "Silence". La prueba usa **igual número de ejemplos por categoría**, así que cada clase representa aproximadamente **8,3%** del total. "Unknown Word" contiene palabras muestreadas al azar de clases fuera del conjunto objetivo, y "Silence" son clips de un segundo extraídos al azar del ruido de fondo. Warden subió un conjunto estándar de archivos de prueba para reproducir la métrica.

El código de entrenamiento acompañante reporta un baseline de **88,2%** de Top-One para el modelo de mayor calidad completamente entrenado (sobre datos V2). Cualitativamente eso es una respuesta razonable pero lejos de perfecta; se espera que sirva como baseline a superar por arquitecturas más sofisticadas.

### 5.2. Métricas de streaming

El Top-One captura una sola dimensión: sus entradas están **alineadas al comienzo de cada utterance**, pero en producción los modelos reciben un **flujo continuo de audio** y no saben cuándo empiezan y terminan las palabras. Además, la ponderación igual de categorías no refleja la distribución real de palabras disparadoras y silencio.

Por eso Warden prueba los modelos contra flujos continuos de audio y los puntúa con múltiples métricas:

- **Matched-percentage:** cuántas palabras fueron correctamente identificadas dentro de una tolerancia temporal dada.
- **Wrong-percentage:** cuántas palabras se distinguieron correctamente como habla (en vez de ruido) pero recibieron la etiqueta de clase equivocada.
- **False-positive percentage:** cuántas palabras se detectaron en partes del audio donde no había habla.

El resultado reportado del baseline V2 sobre el clip de streaming de prueba de 10 minutos fue: **49,0% matched, 46,0% correctamente, 3,0% erróneamente, 0,0% falsos positivos**. Estas métricas exigen más que reconocimiento de plantilla: hay que suprimir reconocimientos repetidos de la misma palabra en ventanas cortas (lógica en `recognize_commands.cc`). La **tolerancia temporal** por defecto para contar un reconocimiento como acierto es de **750 ms**, valor que se ajusta a los requisitos de algunas aplicaciones. El algoritmo está implementado en `test_streaming_accuracy.cc`. Warden también publicó un **archivo de audio de una hora** con utterances en tiempos aleatorios más ruido, con etiquetas de tiempo/verdad, para comparar modelos en streaming.

### 5.3. Evaluaciones históricas (v1 vs v2)

- **v1** (3 de agosto de 2017): **64.727 utterances** de **1.881 hablantes**. El modelo convolucional por defecto del tutorial de TensorFlow (basado en *Convolutional Neural Networks for Small-Footprint Keyword Spotting*, Sainath y Parada 2015) entrenado en V1 dio **85,4%** Top-One en el test de V1.
- **v2** (documentada en este paper): entrenando el mismo modelo en V2 y evaluando en el test de V2 da **88,2%**; entrenado en V2 pero evaluado en el test de V1 da **89,7%**, lo que indica que los datos de entrenamiento V2 son responsables de una **mejora sustancial** sobre V1. La tabla completa: V1-train/V1-test = 85,4%; V2-train/V1-test = 89,7%; V1-train/V2-test = 82,7%; V2-train/V2-test = 88,2%.

### 5.4. Aplicaciones tempranas

La v1 ya había habilitado trabajos que el paper cita como evidencia de utilidad: **CMSIS-NN** (kernels para microcontroladores ARM Cortex-M), **Listening to the World** (combinar el dataset con UrbanSounds para tolerancia al ruido), **Did you Hear That?** (ataques adversariales a interfaces de voz), **Deep Residual Learning for Small-Footprint Keyword Spotting** (ideas de ResNet) y **Raw Waveform-based Audio Classification**.

## 6. Impacto

Speech Commands se convirtió en el **benchmark estándar de facto del keyword spotting** por factores que el paper diseñó deliberadamente: licencia sin fricción (CC BY 4.0, descarga sin registro), formato trivial de procesar (WAV PCM 16 kHz, un segundo, mono), splits reproducibles por hash que hacen las cifras comparables entre papers, baselines y código de referencia listos en el tutorial de TensorFlow, y condiciones realistas que reflejan el caso de producto. Su bajo costo de entrada —se entrena un modelo decente en minutos— lo volvió el equivalente en audio del MNIST/ImageNet. La estabilidad de los splits permitió que la comunidad acumulara resultados comparables durante años, y el énfasis en "small footprint" alineó a investigadores de ML con fabricantes de hardware de borde.

## 7. Limitaciones

El propio paper y su diseño dejan ver varias restricciones:

- **Solo inglés.** Decisión pragmática que Warden reconoce; el dataset sesga hacia el inglés y, por diseño de privacidad, no documenta la distribución demográfica (no recolectó género, etnia ni edad).
- **Vocabulario minúsculo y palabras aisladas.** Los clips de un segundo excluyen palabras largas y todo el habla continua; no sirve para ASR ni para frases. Es, por definición, "vocabulario limitado".
- **Etiquetado de un solo revisor.** Cada clip fue verificado por **un único trabajador**, dejando ruido de etiqueta que múltiples anotadores habrían reducido.
- **Alineación aproximada por volumen.** Extract Loudest Section asume que la palabra es la parte más ruidosa del clip; con ruido de fondo fuerte puede descentrarla o cortarla.
- **Métricas de mundo abierto dependientes de supuestos.** La ponderación igual de las doce categorías (8,3% cada una) es una convención, no la distribución real de disparadores y silencio, algo que el propio paper señala al introducir las métricas de streaming.
- **No propone arquitectura.** El baseline de 88,2% es intencionalmente modesto; el paper aporta el terreno de juego, no el jugador estrella.

## 8. Conexión con la Clase 37 y con el laboratorio

En la **Clase 37 (Datasets y Herramientas para Audio)**, Speech Commands aparece como el dataset didáctico por excelencia: la descripción de "65.000 clips de 1 s, 30 palabras cortas" corresponde a la **versión 1** del dataset (64.727 utterances, ~30 palabras entre comando y auxiliares), mientras que este paper documenta la **versión 2** (105.829 utterances, 35 palabras, 2.618 hablantes). Es el ejemplo canónico de cómo un dataset abierto, pequeño y bien especificado —con splits reproducibles y baselines— dispara la investigación de todo un subcampo, replicando el rol de ImageNet en visión.

En el **laboratorio de la clase**, el anexo "wav2vec 2.0 sobre Speech Commands" cierra el arco histórico del audio. Speech Commands nació (2017-2018) en la era del keyword spotting con **CNN de small footprint** entrenadas desde cero sobre espectrogramas/MFCC, con baseline del orden del 88%. **wav2vec 2.0** (2020) representa el paradigma opuesto: un modelo **auto-supervisado** pre-entrenado sobre miles de horas de audio sin etiquetar, luego afinado sobre un dataset pequeño y etiquetado como este. Usarlo en el lab ilustra dos ideas: que el mismo benchmark de 2018 sigue siendo terreno de comparación válido años después gracias a sus **splits estables**, y que el transfer learning desde representaciones auto-supervisadas suele superar con holgura al baseline convolucional original. La estabilidad del hashing garantiza que la comparación entre la CNN de 2018 y el wav2vec 2.0 del lab sea justa: ambos ven exactamente los mismos archivos en train, validation y test.

**Enlaces internos (sugeridos):**

- Fundamento transversal: [/fundamentos/procesamiento-de-audio](/fundamentos/procesamiento-de-audio) — señales de audio, espectrogramas, MFCC, keyword spotting.
- Clase: [/clases/clase-37](/clases/clase-37) — Datasets y Herramientas para Audio.
- Paper del lab: [/papers/wav2vec2-baevski-2020](/papers/wav2vec2-baevski-2020) — representaciones auto-supervisadas de audio que se afinan sobre Speech Commands.

---

**Nota final — relevancia para salud.** El diseño on-device de Speech Commands es directamente pertinente para dispositivos médicos y asistivos con control por voz. Un modelo de keyword spotting que corre localmente —pequeño, eficiente en energía, capaz de rechazar ruido y habla no relacionada— permite comandar equipos clínicos, sillas de ruedas, camas hospitalarias o interfaces de accesibilidad sin enviar audio continuo a la nube, lo que preserva la **privacidad del paciente por procesamiento local** (el audio ambiental de una sala de hospital, cargado de información sensible, nunca sale del dispositivo). Los mismos requisitos que motivaron el dataset —vocabulario limitado y fiable, mínimos falsos positivos, baja latencia y bajo consumo— son exactamente los que exige un dispositivo asistivo que debe responder de inmediato a comandos como "stop" o "up" incluso sin conectividad, y que no puede depender de un servidor remoto para funciones críticas de un paciente con movilidad reducida.
