---
title: "DatasetGAN: Efficient Labeled Data Factory (2021)"
weight: 340
math: true
---

{{< paper-card
    title="DatasetGAN: Efficient Labeled Data Factory with Minimal Human Effort"
    authors="Yuxuan Zhang, Huan Ling, Jun Gao, Kangxue Yin, Jean-Francois Lafleche, Adela Barriuso, Antonio Torralba, Sanja Fidler"
    year="2021"
    venue="CVPR 2021"
    pdf="/papers/datasetgan-zhang-2021.pdf"
    arxiv="2104.06490" >}}
Paper de NVIDIA, Toronto y MIT citado **explícitamente** por la [Clase 29](/clases/clase-29) como ejemplo de *data augmentation* generativa. Su tesis cabe en una frase: un StyleGAN que aprendió a sintetizar imágenes realistas **ya codifica la estructura semántica de los objetos en sus *feature maps* — basta con leerla**. A partir de ahí, DatasetGAN convierte un GAN pre-entrenado en una **fábrica de datos etiquetados**: se anotan a mano apenas 16 a 40 imágenes generadas, se entrena un pequeño intérprete (MLP) sobre las features internas del GAN, y el sistema produce un número **infinito** de pares imagen-anotación sintéticos con etiquetas densas pixel a pixel. Resultado: supera a las líneas base semi-supervisadas por márgenes amplios usando una fracción ínfima del costo de etiquetado.
{{< /paper-card >}}

---

## El problema: etiquetar es el cuello de botella más caro

Las redes profundas de visión son hambrientas de datos, y para tareas densas —segmentación semántica, segmentación de partes, keypoints— el costo no es elegir una etiqueta por imagen sino **anotar cada píxel**. El paper cuantifica lo brutal del esfuerzo: anotar una escena compleja de 50 objetos toma entre 30 y 90 minutos, y crowdsourcear las 10.000 imágenes etiquetadas de sus experimentos al nivel de detalle requerido tomaría más de **3.200 horas (134 días)** —y aun así produciría anotaciones ruidosas.

La comunidad ya tenía salidas parciales, y DatasetGAN se posiciona frente a cada una:

- **Semi-supervisado:** aprovecha un gran conjunto no etiquetado vía pseudo-labels y *consistency regularization*, pero entrena el modelo de segmentación directamente, sin explotar el modelado generativo de las imágenes.
- **Aprendizaje contrastivo:** entrena buenas representaciones auto-supervisadas y luego ajusta con pocas etiquetas; comparte la idea de amortizar etiquetas, pero usa pérdidas contrastivas en lugar de leer el conocimiento de un GAN.
- **Síntesis con gráficos / GANs previos:** generar escenas 3D con motores gráficos, o usar *image-to-image translation* para adaptar un dataset etiquetado a otro dominio. Estos métodos **asumen la existencia de un gran dominio ya etiquetado** que trasladar; DatasetGAN solo necesita un puñado de imágenes anotadas.

## La idea central: el GAN ya "sabe", solo hay que decodificarlo

La intuición que organiza el paper: un modelo generativo entrenado para sintetizar imágenes *altamente* realistas debe haber adquirido conocimiento semántico en su espacio latente —de lo contrario no podría renderizar de forma coherente las distintas partes de un objeto. En [StyleGAN](/papers/stylegan-karras-2019), el código latente tiene dimensiones **desacopladas** (*disentangled*) que controlan propiedades 3D como punto de vista e identidad; interpolar entre dos códigos produce generaciones realistas, señal de que el GAN aprendió a alinear semántica y geométricamente los objetos y sus partes.

De ahí la consecuencia operativa: **si un humano provee la etiqueta correspondiente a un código latente, esa etiqueta puede propagarse a través de todo el espacio latente del GAN**. En vez de etiquetar miles de imágenes independientes, basta anotar unas pocas y aprender la función *features internas → etiqueta de píxel*; esa función generaliza al resto del espacio.

El flujo es de cuatro pasos (Figura 1 del paper):

1. **Anotar muy poco.** Generar pocas imágenes con StyleGAN (16 a 40) y registrar sus feature maps. Un anotador humano las etiqueta a alto detalle.
2. **Entrenar el "Style Interpreter".** Un *ensemble* de pequeños MLP se entrena sobre los vectores de features pixel a pixel para reproducir la etiqueta humana.
3. **Generar infinitos pares.** El intérprete actúa como una **rama de etiquetas** dentro de StyleGAN; muestreando códigos latentes $z$ se obtiene un generador de dataset sin límite.
4. **Entrenar y evaluar.** Entrenar la arquitectura de visión favorita sobre el dataset sintético, y evaluarla en **imágenes reales**.

## Método: feature maps, MLP por píxel y ensemble

DatasetGAN usa StyleGAN como *backbone* generativo, interpretándolo como un **motor de "renderizado"** cuyos códigos latentes son atributos de "gráficos". El generador inyecta los códigos de estilo en bloques de síntesis vía **normalización de instancia adaptativa (AdaIN)**; las features de salida de las $k$ capas AdaIN se denotan $\{S^0, S^1, \dots, S^k\}$.

El **Style Interpreter** es directo: se hace *upsampling* de todos los feature maps AdaIN a la resolución más alta y se concatenan, de modo que **cada píxel tiene su propio vector de features de 5056 dimensiones**. Sobre cada vector se aplica un **MLP de tres capas** que predice la etiqueta del píxel, con pesos compartidos entre todos los píxeles. Un punto conceptual clave: **el GAN queda congelado** —no se retropropagan gradientes al backbone; la imagen sintetizada solo sirve para recolectar la anotación humana. Para segmentación se usa *cross-entropy*; para keypoints, *heatmaps* gaussianos con pérdida L2.

Para amortizar el muestreo aleatorio de vectores (los mapas son enormes) se entrena un **ensemble de $N=10$ clasificadores**: votación por mayoría para segmentación, promedio de heatmaps para keypoints. El ensemble cumple un segundo rol crucial: **medir incertidumbre para filtrar ruido**. StyleGAN falla ocasionalmente, y el score del discriminador *no* detecta esos fallos de forma robusta; en cambio el desacuerdo del ensemble, medido con la **divergencia de Jensen-Shannon** por píxel, sí lo hace. Se filtra el 10% de imágenes más inciertas, lo que sube el mIoU de 44,60 a 45,64; filtrar demasiado reduce diversidad, así que hay un *trade-off*. Cada par imagen-anotación cuesta ~9 s de *forward pass*, y se usan **10k imágenes sintéticas** en la mayoría de experimentos (el rendimiento se satura lentamente).

## Experimentos: supera baselines con una fracción del costo

El esfuerzo de anotación manual fue minúsculo en términos absolutos: **un único anotador experto** con LabelMe etiquetó 16 autos, 16 cabezas, 30 aves, 30 gatos y 40 dormitorios —~5 horas de trabajo por dataset. Curiosamente, hay más etiquetas (polígonos) en una sola imagen que imágenes en el dataset.

**Segmentación de partes.** Se evalúa en Car, Face, Bird, Cat y Bedroom con **DeepLab-V3 (backbone ResNet pre-entrenado en ImageNet)**, idéntico para todos los baselines (solo cambian los datos). Las líneas base: *Transfer-Learning* (finetune desde MS-COCO) y *Semi-Supervised* (Mittal et al., 2019). DatasetGAN gana en **todas** las clases:

| Dataset (mIoU) | Transfer-Learning | Semi-Sup | **DatasetGAN** |
|---|---|---|---|
| ADE-Car-12 | 24,85 | 28,68 | **45,64** |
| Face-34 | 45,77 | 48,17 | **53,46** |

En ADE-Car-12 supera a los baselines fuera-de-dominio por **20,79 y 16,96 puntos**. Más notable aún: lo hace en el setting **fuera de dominio** (entrena solo con datos sintéticos y evalúa en imágenes reales). Comparado con el modelo *fully supervised*, DatasetGAN con **25 anotaciones** iguala a un DeepLab-V3 entrenado con las **2.600 imágenes** completas de ADE-Car-12 —menos del 1% de las etiquetas, **hasta dos órdenes de magnitud** de ahorro.

**Keypoints.** Para mostrar generalidad se predicen heatmaps en Car y Bird: en Car-20, el PCK (th-15) salta de 43,54 (Transfer-Learning) a **79,91** con el mismo presupuesto de anotación.

**Aplicación 3D.** Como demostración aguas abajo, el paper reconstruye **assets 3D animables** de autos a partir de imágenes monoculares: StyleGAN genera vistas múltiples, el Style Interpreter genera etiquetas de partes y keypoints, y una red de *inverse graphics* con *differentiable rendering* produce modelos con parabrisas transparentes, luces emisivas y ruedas riggeadas —el primer resultado de su tipo.

## Limitaciones reconocidas

- **La calidad del GAN limita la de las etiquetas.** Cuando StyleGAN sintetiza mal, la anotación se degrada: el anotador "se quejó" con las aves, cuyas patas sintetizadas eran borrosas e invisibles, lo que degradó el rendimiento en esa parte.
- **Errores en partes finas o sin bordes claros:** arrugas faciales, patas de ave, bigotes de gato, cuello del gato.
- **Costo de generación:** ~9 s por par, con retornos decrecientes más allá de ~10k imágenes.
- **Un StyleGAN por categoría:** el método no es *open-set*; cada clase requiere su generador pre-entrenado.

## Impacto: data augmentation generativa

DatasetGAN reencuadra para qué sirve un GAN: no como fin (generar imágenes bonitas) sino como **infraestructura para fabricar datos etiquetados**. Es *data augmentation* generativa en su forma más ambiciosa, donde lo aumentado no son solo las imágenes sino sus **etiquetas densas**. La lección transferible: cuando un modelo generativo potente ya capturó la estructura de un dominio en sus representaciones internas, etiquetar masivamente desde cero es redundante —conviene *decodificar* lo que el modelo ya sabe a partir de poquísimas anotaciones. La idea reaparece después con modelos de difusión y *foundation models* de segmentación, pero DatasetGAN la cristaliza tempranamente con un mecanismo desarmantemente simple (un MLP por píxel sobre features de StyleGAN).

## Por qué importa para la Clase 29

La [Clase 29 — Modelos Generativos en Visión](/clases/clase-29) cita DatasetGAN **explícitamente** dentro de "Usos en la industria", como el ejemplo canónico de **data augmentation generativa**: entrenar un clasificador o segmentador con pocos datos generando datos sintéticos de alta calidad. El paper materializa esa idea de punta a punta. Sus conexiones con el temario:

- Se apoya en **[StyleGAN](/papers/stylegan-karras-2019)** como backbone: su síntesis de alta calidad y, sobre todo, su **espacio latente desacoplado** son lo que hace que las features internas contengan conocimiento semántico explotable.
- Es heredero directo del marco adversarial de las **[GANs](/papers/gan-goodfellow-2014)**: toda la potencia generativa que DatasetGAN reutiliza viene del entrenamiento adversarial.
- Encaja en los [fundamentos de modelos generativos](/fundamentos/modelos-generativos): el valor de un generador no se agota en muestrear imágenes —sus *representaciones intermedias* son por sí mismas un recurso reutilizable.

**Relevancia para el contexto de salud (pocos datos etiquetados).** El escenario clínico es el caso de uso natural de DatasetGAN: etiquetar imágenes médicas (lesiones, órganos, estructuras) es carísimo, requiere expertos escasos —radiólogos, patólogos— y produce datasets pequeños. El patrón "entrenar un generador de la modalidad → anotar un puñado de casos → decodificar el conocimiento del generador a etiquetas densas → fabricar un dataset sintético grande" es exactamente la palanca propuesta. Las mismas limitaciones se vuelven críticas en salud (las etiquetas heredan la calidad del generador, las estructuras finas o de bordes difusos abundan en imagen médica, se necesita un generador por modalidad), pero la economía del etiquetado que habilita —de miles de anotaciones a decenas— es la que más duele en el dominio clínico.

## Notas y enlaces

- arXiv: https://arxiv.org/abs/2104.06490 (v2, 20 abr 2021)
- Venue: CVPR 2021 (oral)
- Afiliaciones: NVIDIA, University of Toronto, Vector Institute, MIT, University of Waterloo
