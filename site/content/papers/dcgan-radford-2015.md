---
title: "DCGAN: Deep Convolutional GANs (2015)"
weight: 334
math: true
---

{{< paper-card
    title="Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"
    authors="Alec Radford, Luke Metz, Soumith Chintala"
    year="2015"
    venue="ICLR 2016"
    pdf="/papers/dcgan-radford-2015.pdf"
    arxiv="1511.06434" >}}
El paper que **hizo prácticas las GAN para imágenes**. No introduce un objetivo nuevo —el juego adversarial minimax ya venía de [Goodfellow et al. (2014)](/papers/gan-goodfellow-2014)— sino una **receta de ingeniería arquitectónica**: una familia de redes convolucionales que entrenan de forma estable, bautizada DCGAN. Su aporte va más allá de generar imágenes bonitas: demuestra que las representaciones aprendidas son útiles (el discriminador como extractor de características compite con métodos no supervisados) y que el espacio latente $Z$ tiene estructura semántica navegable, con interpolaciones suaves y **aritmética de vectores** del estilo `word2vec` sobre caras.
{{< /paper-card >}}

---

## Contexto: por qué las GAN no escalaban a imágenes

En 2014 Goodfellow et al. plantearon la GAN como un juego de dos jugadores: un generador $G$ que mapea ruido $z \sim p_z$ a muestras, y un discriminador $D$ que estima si una muestra viene de los datos reales o de $G$. La formulación era elegante y prescindía de la verosimilitud explícita, pero se implementaba con perceptrones multicapa (MLP) y generaba imágenes, en palabras de los autores de DCGAN, "ruidosas e incomprensibles".

El problema no era teórico sino de **estabilidad y arquitectura**. Cuando uno intentaba reemplazar los MLP por las CNN profundas que ya dominaban la visión supervisada, el entrenamiento adversarial colapsaba. La patología más temida es el **mode collapse**, donde "el generador colapsa todas las muestras a un único punto". El equilibrio adversarial es frágil porque $G$ y $D$ se entrenan simultáneamente con objetivos opuestos: si uno gana demasiado rápido, los gradientes del otro se degradan y el aprendizaje se detiene.

La comunidad había buscado rodeos. **LAPGAN** (Denton et al., 2015) generaba la imagen por etapas sobre una pirámide laplaciana, encadenando modelos de baja a alta resolución; funcionaba, pero los objetos salían "tambaleantes". El **VAE** (Kingma & Welling, 2013) producía muestras notoriamente borrosas. Ninguno aprovechaba el generador para tareas supervisadas. El hueco que DCGAN llena es doble: una receta que haga entrenable una sola CNN adversarial de extremo a extremo, y la evidencia de que lo aprendido sirve como representación reutilizable.

## Las pautas arquitectónicas que estabilizan

La contribución que se cita hasta hoy cabe en cinco viñetas, fruto de "exploración extensiva del modelo":

- **Reemplazar todo el *pooling* por convoluciones con *stride*.** Convoluciones con paso (*strided*) en el discriminador y de paso fraccionario (transpuestas) en el generador. La red **aprende su propio submuestreo y sobremuestreo espacial** en lugar de imponer un *max-pooling* determinista (idea de la *all convolutional net*, Springenberg et al., 2014).
- **Usar *batch normalization* en generador y discriminador.** Normaliza la entrada de cada unidad a media cero y varianza unitaria, lo que ataca la mala inicialización y mejora el flujo de gradiente. El paper lo califica de **crítico** para que generadores profundos empiecen a aprender y para prevenir el mode collapse. Con un matiz muy citado: aplicarlo a *todas* las capas causaba oscilación, así que **no se aplica a la salida del generador ni a la entrada del discriminador**.
- **Eliminar las capas *fully-connected* ocultas.** Se conectan directamente las características convolucionales más altas a la entrada/salida. La primera capa del generador toma el ruido $Z$ con una multiplicación matricial y **redimensiona el resultado a un tensor 4-D** que arranca la pila convolucional; la última capa del discriminador se aplana hacia una única salida sigmoide.
- **Activación ReLU en el generador**, salvo la capa de salida que usa **Tanh**. Una activación acotada en $[-1,1]$ permite saturar y cubrir más rápido el espacio de color de la distribución de entrenamiento.
- **Activación LeakyReLU en el discriminador** en todas las capas (pendiente de fuga 0.2), que funciona especialmente bien para mayor resolución.

La lección de diseño es que no son trucos aislados sino un **paquete coherente**: stride para que la geometría espacial sea aprendida y no impuesta; batchnorm para domar el flujo de gradiente del juego adversarial; sin capas densas para mantener profundidad sin explotar parámetros; y ReLU/LeakyReLU + Tanh para acotar y estabilizar las activaciones.

## Arquitectura y entrenamiento

El **generador** canónico (para LSUN) toma una distribución uniforme de **100 dimensiones** ($Z \in \mathbb{R}^{100}$), la proyecta a un tensor 4-D de pequeña extensión espacial pero muchos canales, y aplica **cuatro convoluciones de paso fraccionario** —que el paper aclara se llaman erróneamente "deconvoluciones"— hasta una imagen de **64 × 64** píxeles. Cada convolución transpuesta duplica aproximadamente la resolución mientras reduce los canales: la red va "dibujando" desde un código abstracto hacia píxeles, sin capas fully-connected ni pooling. El **discriminador** es el espejo: convoluciones de stride 2 con LeakyReLU y batchnorm, aplanado y salida sigmoide.

La receta de optimización es deliberadamente explícita, lo que facilitó reproducirla y la volvió folklore de ingeniería:

- **Preprocesamiento:** ninguno, salvo escalar las imágenes al rango de Tanh $[-1, 1]$.
- **Inicialización:** Normal centrada en cero con desviación estándar **0.02**.
- **Optimizador:** **Adam** (no SGD con momentum). La tasa sugerida de 0.001 era demasiado alta; usaron **0.0002**. El momentum $\beta_1 = 0.9$ por defecto causaba oscilación; **reducirlo a 0.5 estabilizó el entrenamiento**.
- **Mini-batch:** 128. **LeakyReLU:** pendiente de fuga 0.2.

## Experimentos

**LSUN bedrooms** (más de 3 millones de ejemplos) es el dataset estrella para mostrar escalado a más datos. Para descartar la mera memorización, exhiben muestras tras **una sola pasada** por el dataset y tras cinco épocas (si memorizara, no podría producir buenas muestras tras una única pasada con tasa de aprendizaje pequeña), reforzado con una deduplicación que removió ~275.000 casi-duplicados. **Faces** (~350.000 recortes de 10K personas) es el dataset de los experimentos de espacio latente. **ImageNet-1k** (recortes de 32 × 32) provee el modelo cuyo discriminador se reutiliza como extractor.

**El discriminador como extractor de características** valida el lado *representation learning*. Entrenan el DCGAN sobre ImageNet-1k, toman las características convolucionales del discriminador de todas las capas, las concatenan en un vector de 28.672 dimensiones y entrenan un **L2-SVM lineal** encima:

- **CIFAR-10:** **82.8%** de precisión, superando todos los métodos basados en K-means (≤82.0%). Como el DCGAN nunca vio CIFAR-10, también demuestra **robustez de dominio**.
- **SVHN** con solo **1000 etiquetas:** **22.48%** de error de test (estado del arte para ese régimen). Control clave: una CNN supervisada *con la misma arquitectura* logra solo 28.87%, lo que prueba que el mérito está en las representaciones no supervisadas y no en la topología.

## Representaciones aprendidas: el espacio latente

Esta es la sección más influyente conceptualmente.

- **Caminar por el espacio latente.** Interpolan entre puntos aleatorios de $Z$ y observan **transiciones suaves**: "un cuarto sin ventana se transforma lentamente en un cuarto con una ventana gigante". La suavidad es evidencia de que el modelo aprendió un *manifold* coherente y no memorizó (lo que daría transiciones abruptas).
- **Filtros del discriminador.** Con *guided backpropagation* muestran que filtros específicos se activan ante partes típicas de un dormitorio (camas, ventanas), frente a un baseline aleatorio sin respuesta semántica.
- **Olvidar dibujar objetos.** Identifican los ~200 mapas de características que codifican "ventana" y los eliminan; el generador **deja de dibujar ventanas** (las reemplaza por puertas o espejos) manteniendo la composición de la escena. Es evidencia de **desenredo** (*disentanglement*) entre representación de escena y de objeto.
- **Aritmética de vectores en caras** —el clímax. Inspirados en el `Rey − Hombre + Mujer ≈ Reina` de word2vec, promedian los vectores $Z$ de tres ejemplares por concepto y operan: "hombre con lentes − hombre sin lentes + mujer sin lentes" produce una mujer con lentes. Construyen además un **vector "giro"** promediando caras mirando a izquierda vs. derecha que, al sumarlo, **rota la pose** de forma fiable. Es la primera demostración de esta estructura lineal emergiendo de forma puramente no supervisada.

## Limitaciones reconocidas

- **Inestabilidad residual:** al entrenar por más tiempo, a veces colapsaban filtros a un único modo oscilante. El mode collapse fue mitigado, no eliminado.
- **Receta empírica, no teoría:** las pautas son condiciones necesarias halladas a mano; el paper no explica teóricamente *por qué* estabilizan.
- **Evaluación cualitativa:** evitan deliberadamente la verosimilitud (mala métrica) y buena parte de la evidencia es visual.
- **Resolución limitada:** 64 × 64 (32 × 32 para ImageNet/CIFAR). Escalar a alta resolución quedaría como problema abierto por años.

## Impacto

DCGAN es uno de los papers más citados en visión generativa porque **hizo las GAN prácticas para imágenes**. Antes, "entrenar una GAN" era un ejercicio frágil de alquimia; después, las cinco pautas se volvieron el *default* sobre el que casi todo el mundo construía, hasta el punto de que "arquitectura DCGAN" se volvió sinónimo de "GAN convolucional que funciona".

Su descendencia es enorme. La línea de progresión de resolución —Progressive Growing y **[StyleGAN](/papers/stylegan-karras-2019)** (Karras et al., 2019)— hereda directamente el esqueleto convolucional generador/discriminador y lleva a fotorrealismo el espacio latente navegable que DCGAN demostró. La aritmética de vectores anticipó toda la literatura de edición de imágenes por manipulación de códigos latentes, y la idea de reutilizar el discriminador como extractor prefiguró el uso de modelos generativos para aprendizaje de representaciones. Incluso los detalles pequeños se volvieron estándar: Adam con $\beta_1 = 0.5$ y lr $= 0.0002$, init Normal con $\sigma = 0.02$, Tanh a la salida.

## Por qué importa para la Clase 29

La [Clase 29](/clases/clase-29) ("Modelos Generativos en Visión") organiza los modelos en familias —GAN, VAE, difusión. DCGAN encaja en dos lugares:

- **Es el "cómo se hace que funcione".** La clase presenta el objetivo adversarial de Goodfellow de forma abstracta (el minimax entre $G$ y $D$). DCGAN lo traduce a una arquitectura concreta y una receta que de verdad converge sobre imágenes. Es el eslabón pedagógico entre "qué optimiza una GAN" y "qué tecleo para entrenar una".
- **El espacio latente conecta con el VAE.** La clase introduce el latente continuo y navegable sobre todo en el [VAE](/fundamentos/modelos-generativos) (donde el prior gaussiano lo hace explícito). DCGAN muestra que **esa misma propiedad emerge en una GAN** sin imponerla. El contraste ilumina el trade-off central: el VAE *garantiza* un latente regular a costa de muestras borrosas; DCGAN *no lo garantiza* pero lo obtiene empíricamente con muestras más nítidas.

## Lecturas relacionadas

- [GAN original (Goodfellow et al., 2014)](/papers/gan-goodfellow-2014) — el objetivo adversarial que DCGAN vuelve práctico.
- [StyleGAN (Karras et al., 2019)](/papers/stylegan-karras-2019) — el descendiente que lleva el esqueleto y el latente navegable de DCGAN al fotorrealismo de alta resolución.
- [Fundamento: Modelos Generativos](/fundamentos/modelos-generativos) — el marco conceptual (GAN vs VAE vs difusión) en el que DCGAN se ubica.
- [Clase 29: Modelos Generativos en Visión](/clases/clase-29) — la clase que este análisis acompaña.
