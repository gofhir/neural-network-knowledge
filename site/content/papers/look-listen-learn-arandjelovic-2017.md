---
title: "Look, Listen and Learn (2017)"
weight: 315
math: true
---

{{< paper-card
    title="Look, Listen and Learn"
    authors="Relja Arandjelović, Andrew Zisserman"
    year="2017"
    venue="ICCV 2017"
    pdf="/papers/look-listen-learn-arandjelovic-2017.pdf"
    arxiv="1705.08168" >}}
Trabajo fundacional del **aprendizaje autosupervisado audio-visual**. Plantea una pregunta ingenua —¿qué se aprende mirando y escuchando muchos videos sin etiquetar?— y la responde con una tarea pretexto nueva, la **Audio-Visual Correspondence (AVC)**: dado un frame y un clip de audio de 1 segundo, decidir si corresponden. Entrenando **dos torres desde cero** (visión + audio) para resolver AVC emergen, *simultáneamente*, buenas representaciones de imagen Y de sonido —sin una sola etiqueta humana. Las features de audio establecen un **nuevo estado del arte** en clasificación de sonido (ESC-50, DCASE) y las de imagen quedan a la par del SSL de la época en ImageNet. La arquitectura se conoce como **L³-Net** ("Look, Listen and Learn Network").
{{< /paper-card >}}

---

## Contexto: SSL en 2017 y la supervisión "gratis" del video

Hacia 2017 el [aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado) en visión era casi todo **mono-modal**: predicción de contexto espacial, coloreado, puzzles jigsaw, inpainting, o señales temporales en video (verificar el orden de frames). La promesa común era aprender representaciones útiles sin el costo de anotar.

Arandjelović y Zisserman giran el eje hacia lo **multimodal**, con tres motivaciones. Primero, el video con audio es una fuente de supervisión *virtualmente infinita y gratis*. Segundo, es plausible como mecanismo de aprendizaje biológico: un infante podría usar exactamente esta correlación visión-sonido al desarrollar sus sentidos. Tercero, interesa saber *qué* se aprende y *cuán buenas* son las redes resultantes para otras tareas.

El antecedente directo es el grupo del MIT: **SoundNet** (Aytar et al., 2016), que entrena una red de audio destilando conocimiento de redes visuales preentrenadas; *Visually Indicated Sounds* y *Ambient sound provides supervision* (Owens et al., 2016). El denominador común es que **una de las modalidades siempre actúa de maestro fijo** —típicamente una red visual preentrenada en ImageNet/Places. La provocación de este paper es liberar a *ambas* redes y entrenarlas juntas: no solo es viable, sino que **mejora sustancialmente** sobre el esquema maestro-alumno.

## La tarea: Audio-Visual Correspondence (AVC)

La contribución central es una tarea de clasificación binaria de enunciado simple: dado un par formado por **un frame de video** y **un clip de audio de 1 segundo**, decidir si **corresponden o no**. La generación de ejemplos es donde reside la señal gratis:

- **Pares positivos** (corresponden): se muestrea un video al azar, se elige un frame, y se toma un clip de 1 s que **se solapa en el tiempo** con ese frame, del **mismo video**.
- **Pares negativos** (no corresponden): se toman un frame de un video y un clip de audio de **otro video distinto**.

Con igual número de positivos y negativos, el azar acierta 50%. La única forma de superarlo consistentemente es **aprender conceptos semánticos en ambos dominios**: para saber que el frame de un violín no corresponde a un ladrido, la red tiene que haber aprendido qué es un violín *visualmente* y qué es un ladrido *acústicamente* —sin que nadie le dijera nunca que existen esos conceptos. La supervisión emerge de ver muchas veces un violín tocándose junto a su sonido, y casi nunca junto a un ladrido.

AVC es un requisito **más laxo que la sincronía**: no exige alineamiento cuadro a cuadro, solo que *algo* en la imagen correlacione con *algo* en el audio (un auto presente con ruido de motor; una toma exterior con viento). Es una tarea genuinamente difícil incluso para humanos —videos sin restricción y ruidosos, fuentes de sonido fuera de cuadro, narración, música editada—; en tests informales los humanos solo superan a la red por unos pocos puntos cuando se les da un frame aislado y un único segundo de audio.

## Método: dos torres más red de fusión

La L³-Net tiene **tres partes**: subred de visión, subred de audio y red de fusión.

**Subred de visión.** Entrada: imagen a color de 224×224. Estilo VGG —filtros 3×3, max-pooling 2×2 con stride 2— en **cuatro bloques** `conv+conv+pool` que duplican filtros (**64, 128, 256, 512**), cada conv seguida de batch norm y ReLU. Un max-pooling final sobre todas las posiciones espaciales colapsa el mapa en un **vector de 512 dimensiones**.

**Subred de audio.** Entrada: el clip de 1 s convertido en **log-espectrograma**, tratado como imagen en escala de grises de 257×199 (audio resampleado a 48 kHz, ventana de 0.01 s con solape de media ventana → 199 ventanas × 257 bandas de frecuencia, seguido de logaritmo). La arquitectura es **idéntica** a la de visión salvo la entrada 1-D, y produce también un **vector de 512-D**. La simetría entre torres es deliberada: el mismo backbone procesa píxeles y espectrogramas.

**Red de fusión.** Las dos features de 512-D se **concatenan** (1024-D) y atraviesan dos capas fully-connected con ReLU intermedio (tamaño 128-D) y un softmax de 2 vías. Esta **fusión tardía** —cada modalidad se procesa de forma independiente hasta su vector de 512-D, y solo entonces se combinan— es lo que fuerza a *cada* torre a aprender una representación semántica autónoma, y lo que después permite usar cada subred por separado como extractor de features.

**Entrenamiento.** Optimizador Adam, weight decay 10⁻⁵, learning rate 10⁻⁴. Se entrenó en **16 GPUs** con entrenamiento síncrono en TensorFlow (batch efectivo de 256) sobre 400k videos de 10 s, durante **dos días**, viendo **60 millones de pares** frame-audio. Data augmentation estándar de imágenes y, en audio, solo variación de volumen hasta 10%.

## Experimentos y resultados

**Datasets.** *Flickr-SoundNet* (subconjunto de 500k videos de Flickr completamente sin restricción, sin usar tag ni etiqueta para entrenar) es el dataset de transfer. *Kinetics-Sounds* (19k clips de YouTube en **34 clases de acción** audio-visuales: tocar instrumentos, cortar el pasto, tap dancing, reír) tiene etiquetas y sirve para evaluación cuantitativa.

**Tarea AVC.** La L³-Net alcanza **78% en Flickr-SoundNet** y **74% en Kinetics-Sounds** (azar 50%). Comparada de forma justa contra baselines supervisados sobre Kinetics-Sounds, el *supervised pretraining* **empata** a la L³-Net (74%) y la combinación directa rinde peor (65%): la autosupervisión **no pierde** frente a la supervisión.

**Features de audio — clasificación de sonido.** Extrayendo features con la subred de audio (max-pool de `conv4_2` → 6144-D, z-score, SVM lineal one-vs-all):

| Benchmark | L³-Net | SoundNet | Sin AVC | Humano |
|---|---|---|---|---|
| **ESC-50** (50 clases) | **79.3%** | 74.2% | 62.5% | 81.3% |
| **DCASE** (10 escenas) | **93%** | 88% | 85% | — |

En ESC-50 supera a SoundNet por **5.1 puntos** y reduce en **72%** la brecha con el humano; en DCASE recorta el error en **42%**. Notable: **SoundNet usa dos redes visuales supervisadas (ImageNet y Places2) como maestros**, mientras la L³-Net aprende *ambas* redes sin supervisión alguna.

**Features visuales — ImageNet.** Congelando los pesos y entrenando un clasificador lineal sobre `conv4_2`, logra **32.3% top-1**, **a la par** del SSL de la época (Doersch 31.7%, Zhang 32.6%, Noroozi-Favaro 34.7%) y muy por encima de inicialización aleatoria (12.9%). Los autores son honestos: la comparación no es del todo limpia (backbone VGG vs. AlexNet de los rivales; entrena con frames de Flickr —objetos descentrados, motion blur— en vez de imágenes curadas de ImageNet). Medida la *mejora sobre inicialización aleatoria* atribuible a la autosupervisión, la L³-Net **supera a todos los competidores**.

**Qué aprende.** Visualizando activaciones de `pool4`, ambas torres aprenden conceptos semánticos sin supervisión: la visión reconoce guitarras, acordeones, boleras, escenas (cielo —que *no* es un detector de "azul"—, agua, multitud) y distingue grano fino (guitarra acústica vs. bajo eléctrico); el audio distingue entidades, escenas (viento, bajo el agua) y hace clasificación fina ("fingerpicking" vs. "playing bass"). Cuantitativamente, las redes entrenadas tienen unidades de alta preferencia para 10 (visión) y 11 (audio) de las 34 clases, frente a 1 y 1 en redes aleatorias; un t-SNE muestra clustering por clase de acción; y la **NMI** sube de 0.204→0.409 (visión) y 0.219→0.330 (audio). Los heatmaps de `conv4_2` **localizan** la fuente del sonido en el frame —germen directo de la recuperación cross-modal.

## Limitaciones reconocidas

- **Correspondencia, no concurrencia.** AVC solo exige correlación entre un frame y un segundo de audio, no sincronía temporal. La concurrencia es más fuerte, pero requeriría usar *múltiples* frames (video, no un frame aislado).
- **Dificultad y ruido.** En videos sin restricción la tarea es ardua incluso para humanos; la red logra 74–78%, lejos del techo, en parte por lo *local* de la entrada.
- **Comparaciones imperfectas en visión** (backbone y fuente de datos distintos a los rivales) y **confusiones de grano fino** con solo 1 s de audio (saxofón/trombón, tap dancing/pen tapping).

## Impacto

*Look, Listen and Learn* estableció dos cosas que se volvieron estándar: (1) la **correspondencia entre modalidades** es una señal de pretexto suficientemente rica para entrenar *ambas* redes desde cero, sin maestro fijo, mejorando sobre la destilación maestro-alumno; y (2) las features resultantes **transfieren bien** —SOTA en sonido, paridad en ImageNet. Su patrón de **dos torres + fusión tardía** anticipa los modelos cross-modales contrastivos posteriores (CLIP, ConVIRT). Dentro de la obra de los autores abre la saga: [*Objects that Sound*](/papers/objects-that-sound-arandjelovic-2018) (2018) reemplaza la fusión por una comparación de distancia en un espacio embebido común, habilitando explícitamente la **recuperación cross-modal** (encontrar la imagen que produce un sonido y viceversa) y la **localización** del objeto que suena. La idea de "supervisión gratis a partir de la estructura natural de los datos" es el corazón conceptual del SSL moderno.

## Por qué importa para la Clase 28

La [Clase 28](/clases/clase-28) (Aprendizaje Autosupervisado) dedica una sección a la **Correspondencia Audio-Visual** con este ejemplo exacto del par positivo (frame y audio del mismo momento del mismo video) frente al negativo (de videos distintos), y plantea que, una vez aprendido el alineamiento, **se pueden buscar imágenes a partir de audio y audio a partir de imágenes**. Este paper es la base teórica de esa afirmación: la tarea AVC y la simetría de las dos torres producen representaciones de imagen y de sonido en espacios comparables.

## Notas y enlaces

- Preprint: [arxiv.org/abs/1705.08168](https://arxiv.org/abs/1705.08168) (v2, agosto 2017).
- Fundamento transversal: [aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado).
- Dominios: [multimodal](/dominios/multimodal) (donde esta línea audio-visual convive con CLIP, ConVIRT y la familia visión-lenguaje) y [audio](/dominios/audio) (donde vive la clasificación de sonido ESC-50/DCASE que este trabajo lleva al SOTA).
- Continuación: [*Objects that Sound*](/papers/objects-that-sound-arandjelovic-2018) (Arandjelović y Zisserman, 2018).
- Afiliación: DeepMind y VGG (University of Oxford).
</content>
</invoke>
