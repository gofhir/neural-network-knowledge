---
title: "U-Net: Convolutional Networks for Biomedical Image Segmentation (2015)"
weight: 332
math: true
---

{{< paper-card
    title="U-Net: Convolutional Networks for Biomedical Image Segmentation"
    authors="Olaf Ronneberger, Philipp Fischer, Thomas Brox"
    year="2015"
    venue="MICCAI 2015"
    pdf="/papers/unet-ronneberger-2015.pdf"
    arxiv="1505.04597" >}}
Paper de arquitectura de la Universidad de Freiburg que propone una red convolucional **encoder-decoder simétrica en forma de U** para **segmentación semántica** (una etiqueta de clase por pixel) de imágenes biomédicas. Su pieza decisiva son las **skip connections**, que copian los mapas de características de alta resolución del encoder y los concatenan en el decoder, resolviendo la tensión contexto-vs-localización. Combinada con **data augmentation elástica**, se entrena desde muy pocas imágenes anotadas y ganó los challenges de segmentación de la ISBI. Su impacto es doble: estándar de facto en segmentación de imagen médica y, una década después, **backbone denoiser de los modelos de difusión** (DDPM, Stable Diffusion). Material de la [Clase 29](/clases/clase-29).
{{< /paper-card >}}

---

## Contexto: segmentación biomédica y el problema de los pocos datos

Hacia 2013-2015 las redes convolucionales profundas ya dominaban el reconocimiento visual: AlexNet (2012) había marcado el quiebre entrenando sobre el millón de imágenes de ImageNet, y luego llegaron redes aún más profundas como VGG. Pero el uso *típico* de un convnet era la **clasificación**: una imagen entra, una etiqueta sale. Muchas tareas visuales —y de modo crítico el procesamiento de imágenes biomédicas— requieren además **localización**: una etiqueta por pixel. Y el dato duro que abre el paper es que entrenar redes profundas "requiere muchos miles de muestras anotadas", una cantidad que en el dominio biomédico "habitualmente está fuera de alcance" porque anotar exige expertos (patólogos, biólogos) y mucho tiempo.

El antecedente directo es **Ciresan et al. (2012)**, que entrenó una red en esquema de **ventana deslizante** (*sliding-window*): para etiquetar cada pixel se le daba a la red la región local (*patch*) a su alrededor. Ganó el EM segmentation challenge de ISBI 2012, pero Ronneberger señala dos defectos:

1. **Lentitud y redundancia.** La red corre por separado para cada patch y los patches vecinos se solapan masivamente, recomputando lo mismo una y otra vez.
2. **El compromiso contexto vs. localización.** Patches grandes requieren más capas de *max-pooling*, lo que *reduce* la precisión de localización; patches pequeños permiten ver poco contexto. No se puede tener ambas cosas a la vez.

El otro pilar es la **Fully Convolutional Network (FCN)** de Long, Shelhamer y Darrell (2014), que reemplaza el *pooling* de un convnet contractivo por *upsampling* sucesivo y combina features de alta resolución del camino contractivo con la salida upsampleada. U-Net toma esta idea y la lleva a su forma simétrica, con la diferencia clave de propagar **un gran número de canales de características en el camino de upsampling**, lo que lleva información de contexto hacia las capas de alta resolución y vuelve el camino expansivo "más o menos simétrico" al contractivo: la **forma de U**.

## Contribución central

La aportación se descompone en tres ideas, la primera de las cuales define la arquitectura:

1. **Encoder-decoder simétrico con skip connections.** Un *camino contractivo* (encoder, lado izquierdo de la U) captura **contexto** por downsampling sucesivo, y un *camino expansivo* (decoder, lado derecho) recupera **localización precisa** por upsampling sucesivo. Lo decisivo son las **skip connections**: en cada nivel del decoder, el mapa upsampleado se **concatena con el mapa correspondiente del encoder**. Estas conexiones reinyectan detalle espacial de alta resolución —bordes, contornos finos— que el downsampling había destruido. Resuelven directamente el dilema de Ciresan: el contexto viene por el camino profundo de la U, el detalle por las skip connections laterales.

2. **Data augmentation elástica para el régimen de pocos datos.** Con poquísimos datos anotados, los autores aplican **deformaciones elásticas** aleatorias a las imágenes. Esto enseña a la red invarianza a deformaciones sin verlas en el corpus. El paper lo justifica para el dominio biomédico: la deformación "solía ser la variación más común en tejido" y se simula de forma realista y eficiente.

3. **Weighted loss para separar objetos que se tocan.** Separar células adyacentes de la misma clase es un desafío recurrente. Los autores introducen una **pérdida ponderada** (un mapa de pesos pixel a pixel) que asigna gran peso a los pixeles de fondo que separan células en contacto, forzando a la red a aprender esas finas fronteras.

Una restricción de diseño elegante enmarca todo: la red **no tiene capas totalmente conectadas** y usa solo la parte *válida* de cada convolución (sin padding), de modo que el mapa de segmentación solo contiene pixeles cuyo contexto completo está disponible en la entrada.

## Arquitectura y método

**Camino contractivo (encoder).** Su bloque repetido es: **dos convoluciones 3×3 sin padding** (*valid convolutions*), cada una con **ReLU**, seguidas de **max-pooling 2×2 con stride 2**. En cada paso de downsampling se **duplica el número de canales**. Espacialmente la imagen se encoge mientras la profundidad de canales crece: de 64 canales arriba a 1024 en el fondo de la U.

**Camino expansivo (decoder).** Cada paso consiste en: (1) **upsampling** seguido de una **convolución 2×2** —la *"up-convolution"* o convolución transpuesta— que **reduce a la mitad** los canales; (2) **concatenación con el mapa correspondiente del encoder, recortado** (la skip connection); (3) **dos convoluciones 3×3** con ReLU. El recorte (*cropping*) es necesario porque las convoluciones sin padding pierden pixeles de borde en cada paso, de modo que el mapa del encoder es algo mayor que el del decoder. En la capa final, una **convolución 1×1** mapea cada vector de 64 componentes al número de clases. En total, **23 capas convolucionales**.

**Estrategia overlap-tile.** Como la salida es más chica que la entrada por un borde constante, para segmentar imágenes arbitrariamente grandes la imagen se procesa por baldosas (*tiles*) solapadas; el contexto faltante en los bordes se **extrapola por espejado** (*mirroring*). Esto permite aplicar la red sin que la resolución quede limitada por la memoria de la GPU.

**Entrenamiento.** Usa **SGD** (implementación en Caffe). Por las convoluciones sin padding, los autores prefieren **tiles grandes sobre batches grandes**: reducen el batch a *una sola imagen* y compensan con **momentum alto (0.99)**. La función de energía es un **soft-max pixel a pixel** combinado con **cross-entropy**:

$$E = \sum_{x \in \Omega} w(x) \, \log\big(p_{\ell(x)}(x)\big)$$

donde $\ell(x)$ es la etiqueta verdadera y $w(x)$ un **mapa de pesos** precomputado que compensa la frecuencia desigual de clases y fuerza el aprendizaje de las fronteras de separación:

$$w(x) = w_c(x) + w_0 \cdot \exp\!\left(-\frac{(d_1(x) + d_2(x))^2}{2\sigma^2}\right)$$

con $d_1$ la distancia al borde de la célula más cercana y $d_2$ a la segunda más cercana (usan $w_0 = 10$, $\sigma \approx 5$ pixeles). El efecto es un "valle" de peso alto en la delgada franja de fondo entre células adyacentes, que empuja a la red a no fusionar instancias vecinas. Los pesos se inicializan con la receta de He et al. (gaussiana de desviación $\sqrt{2/N}$) para mantener varianza unitaria por mapa.

**Augmentation en detalle.** Las **deformaciones elásticas aleatorias** son "el concepto clave" para entrenar con pocas imágenes: se generan vectores de desplazamiento aleatorios sobre una grilla gruesa de 3×3 (gaussiana de desviación 10 pixeles) interpolados con bicúbica. El **dropout** al final del camino contractivo agrega augmentation implícita.

## Experimentos

El paper demuestra la U-Net en tres tareas, todas con muy pocos datos de entrenamiento:

- **EM segmentation (ISBI 2012).** Estructuras neuronales en microscopía electrónica del cordón nervioso ventral de larva de *Drosophila*: solo 30 imágenes de 512×512. La U-Net (promediada sobre 7 versiones rotadas), **sin pre- ni post-procesamiento**, logra un *warping error* de **0.000353**, el mejor de la tabla, superando a la sliding-window de Ciresan et al. (0.000420). Los únicos algoritmos con mejor *Rand error* usaban post-procesamiento muy específico del dataset.
- **ISBI cell tracking challenge 2015.** En **PhC-U373** (células de glioblastoma-astrocitoma, 35 imágenes parciales) la U-Net logra **IoU 92%** contra 83% del segundo mejor. En **DIC-HeLa** (20 imágenes parciales) logra **IoU 77.5%** contra apenas 46% del segundo mejor. Estos márgenes son enormes para un benchmark de challenge.

El paper destaca también la **velocidad**: segmentar una imagen 512×512 toma menos de un segundo en una GPU, y el entrenamiento completo es de ~10 horas en una NVidia Titan de 6 GB, gracias a que el overlap-tile elimina la redundancia masiva del sliding-window.

## Limitaciones

- **Las convoluciones sin padding complican el diseño:** obligan al recorte de las skip connections, a elegir el tamaño de tile con cuidado y al espejado de bordes. Variantes posteriores usan *same padding* para simplificar, a costa de algún artefacto de borde.
- **Segmentación semántica, no de instancias nativa:** la separación de instancias que se tocan se logra con el truco de la weighted loss, no con un mecanismo de instancias propiamente dicho.
- **Batch de tamaño 1 y momentum 0.99:** elección forzada por la memoria (tiles grandes), no necesariamente óptima.
- **Validación acotada** al dominio biomédico microscópico y a datasets pequeños; irónicamente, el tiempo demostraría que la arquitectura generaliza muchísimo más allá de lo que el paper se atrevió a reclamar.

## Impacto: de la segmentación médica al corazón de los generativos

La U-Net se volvió **ubicua**, y su impacto es doble.

**Segmentación de imagen médica.** En su nicho original es prácticamente el **estándar de facto**: resonancia magnética, tomografía, histopatología, segmentación de órganos y lesiones. Cualquier pipeline moderno de segmentación de imagen médica casi seguro tiene una U-Net —o un descendiente como nnU-Net, Attention U-Net, U-Net++ o TransUNet— en su núcleo. Es el punto de contacto más directo con el dominio de salud. La idea de skip connections encoder→decoder migró además a innumerables arquitecturas de visión densa.

**Backbone de los modelos de difusión.** El giro que conecta este paper con la [Clase 29](/clases/clase-29) es más profundo. Los [modelos de difusión](/fundamentos/modelos-de-difusion) —[DDPM](/papers/ddpm-ho-2020) (Ho, Jain y Abbeel, 2020) y luego [Latent Diffusion / Stable Diffusion](/papers/latent-diffusion-rombach-2022)— funcionan aprendiendo a *revertir* un proceso que añade ruido gaussiano progresivo a una imagen. La red que aprende ese paso inverso —la que, dado un dato ruidoso $x_t$ y un paso $t$, **predice el ruido** $\epsilon_\theta(x_t, t)$ a quitar— es una **U-Net**.

¿Por qué la U-Net y no otra red? Porque el denoising es, estructuralmente, una tarea *imagen-a-imagen densa*: la salida tiene la misma resolución que la entrada y cada pixel depende tanto del contexto global (qué objeto se está formando) como del detalle local de alta resolución (la textura exacta a reconstruir). Esa es *exactamente* la tensión contexto/localización que las skip connections resuelven: el camino contractivo captura el contexto semántico del paso de difusión y las skip connections reinyectan el detalle fino necesario para predecir el ruido pixel a pixel.

La U-Net de difusión se enriquece respecto a la original —se le añaden bloques residuales, *group normalization*, capas de **atención** en las resoluciones bajas y un *time embedding* que inyecta el paso $t$ en cada bloque— pero el esqueleto es inconfundiblemente la U: encoder que contrae, decoder que expande, skip connections que cosen ambos lados. Un paper de MICCAI sobre microscopía de células neuronales terminó siendo el motor de la generación de imágenes a partir de texto.

## Por qué importa para la Clase 29

La [Clase 29](/clases/clase-29) (Modelos Generativos en Visión) trata los modelos generativos con foco en difusión: el proceso *forward* (añadir ruido gaussiano paso a paso hasta destruir la imagen) y el proceso *reverse* (aprender a quitarlo). El paso inverso necesita una red que mapee una imagen ruidosa a una predicción del ruido manteniendo resolución espacial completa, y esa red es la U-Net de este paper, adaptada. La clase rotula la sección "Aprendiendo el paso inverso: U-Net" precisamente por esto. Estudiar a Ronneberger et al. (2015) antes de [DDPM](/papers/ddpm-ho-2020) permite entender *por qué* el denoiser tiene la forma que tiene, en vez de tomarlo como caja negra: las skip connections no son un detalle de implementación, son la razón de que la red reconstruya detalle fino mientras razona sobre contexto global. La lección transversal es que una buena *primitiva arquitectónica* —el encoder-decoder simétrico con skips— trasciende la tarea para la que fue inventada: nace en segmentación supervisada con pocos datos y reaparece, una década después, como el componente central del paradigma generativo dominante.

## Notas y enlaces

- Preprint: arXiv:1505.04597 (mayo 2015). Venue: MICCAI 2015.
- Afiliación: Computer Science Department y BIOSS Centre, University of Freiburg, Alemania.
- Implementación en Caffe y modelos: `http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net`.
- Relacionados: [modelos de difusión](/fundamentos/modelos-de-difusion), [DDPM (Ho et al., 2020)](/papers/ddpm-ho-2020), [Latent Diffusion (Rombach et al., 2022)](/papers/latent-diffusion-rombach-2022).
