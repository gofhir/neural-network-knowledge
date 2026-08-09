---
title: "I3D: Quo Vadis, Action Recognition? (2017)"
weight: 405
math: true
---

{{< paper-card
    title="Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset"
    authors="João Carreira, Andrew Zisserman (DeepMind)"
    year="2017"
    venue="CVPR 2017 / arXiv:1705.07750"
    pdf="/papers/i3d-carreira-2017.pdf" >}}
El paper parte de un diagnóstico incómodo: los datasets estándar de acciones —**UCF-101 y HMDB-51**, del orden de 10 000 videos— son tan pequeños que *casi cualquier* arquitectura rinde parecido, haciendo imposible distinguir qué diseño es bueno. La solución llega en dos aportes entrelazados. Primero, re-evaluar el *zoo* de arquitecturas de video a la luz de **Kinetics** (400 clases, ~240 000 videos de entrenamiento), un dataset dos órdenes de magnitud mayor que ordena por mérito real las familias de modelos. Segundo, el modelo **I3D (Two-Stream Inflated 3D ConvNet)**: en vez de diseñar una arquitectura 3D desde cero, se toma una CNN 2D muy profunda y ya probada (**Inception-v1** preentrenada en ImageNet) y se **"inflan"** sus filtros de 2D a 3D ($N\times N \to N\times N\times N$), heredando incluso los pesos de ImageNet. Tras preentrenar en Kinetics y hacer fine-tuning, I3D lleva el estado del arte a **98.0% en UCF-101 y 80.9% en HMDB-51** (reducciones de error del 63% y 35%). Es el pivote de la [Clase 36](/clases/clase-36): establece la receta **"pre-entrenar en Kinetics, transferir"**.
{{< /paper-card >}}

---

## Contexto: el zoo de arquitecturas de video sin un ImageNet propio

Mientras en imágenes las arquitecturas habían madurado rápido con líderes claros (AlexNet → VGG-16 → ResNet), para video **no había una arquitectura ganadora**. Las propuestas divergían en tres ejes: si los operadores usan kernels **2D (basados en imagen)** o **3D (basados en video)**; si la entrada es solo **RGB** o incluye **flujo óptico** precomputado; y, en el caso 2D, cómo se propaga la información entre cuadros (con **LSTM** o por agregación temporal de features). El motivo de fondo de tanta indefinición era la **falta de un "ImageNet de video"**: en imágenes se descubrió que redes profundas entrenadas sobre 1000 categorías servían para otras tareas, y mejorar el backbone se traducía en mejoras aguas abajo; en video la pregunta análoga —¿un dataset grande daría un empujón transferible?— ni siquiera podía plantearse con benchmarks pequeños. **Kinetics** es la respuesta material a esa carencia. El título, tomado de un fotograma de la película *Quo Vadis* (1951), es una metáfora: en un solo cuadro no se sabe si los actores están por besarse o ya lo hicieron —las acciones son ambiguas cuadro a cuadro—, y la pregunta "¿hacia dónde va esto?" se dirige tanto a la escena como al campo.

## Método: inflar una CNN 2D preentrenada a 3D

I3D combina tres ingredientes que hasta entonces se usaban por separado.

**Inflado 2D→3D.** Se parte de una arquitectura 2D exitosa y se **inflan todos los filtros y kernels de pooling** dándoles una dimensión temporal adicional. Como los filtros suelen ser cuadrados, se los vuelve cúbicos:

$$N \times N \;\longrightarrow\; N \times N \times N$$

Con esto, sin re-diseñar la topología, Inception-v1 pasa a operar sobre volúmenes espacio-temporales. Esto rompe el techo histórico de las 3D ConvNets, que eran forzosamente **poco profundas** (hasta 8 capas) porque su alta dimensionalidad de parámetros, sumada a la escasez de datos de video, las excluía del preentrenamiento ImageNet.

**Bootstrapping de pesos: el *boring-video fixed point*.** El inflado da la arquitectura 3D; el gran valor está en **heredar también los parámetros** ImageNet. Una imagen puede convertirse en un video "aburrido" copiándola $N$ veces sobre el eje temporal. Los autores exigen que las activaciones agrupadas sobre ese video aburrido sean **iguales** a las de la imagen original. Gracias a la **linealidad** de la convolución, esto se logra repitiendo el filtro 2D $N$ veces a lo largo del tiempo y reescalándolo:

$$w^{3D}(t) = \frac{1}{N}\, w^{2D}, \qquad t = 1, \dots, N$$

Así la respuesta del filtro sobre el video aburrido es idéntica a la respuesta 2D, y el modelo 3D queda *implícitamente preentrenado en ImageNet* —millones de imágenes etiquetadas de inicialización que ninguna 3D ConvNet desde cero podía aprovechar.

**Two-stream sobre I3D.** Aunque una 3D ConvNet *debería* aprender movimiento del RGB, sigue siendo un cómputo puramente feedforward, mientras que los algoritmos de flujo óptico son en cierto sentido **recurrentes** (optimización iterativa). Por eso los autores encontraron que **seguía siendo valioso** el esquema two-stream: una red I3D sobre RGB y otra sobre **flujo óptico** (computado con TV-L1), entrenadas por separado y promediando sus predicciones en test. Con video a 25 fps, resultó útil **no hacer pooling temporal en las dos primeras capas** ($1\times 3\times 3$, stride 1 en tiempo). El modelo se entrena con **snippets de 64 cuadros** y se testea sobre el video completo. La arquitectura resultante, **Inflated Inception-v1**, tiene 25M parámetros —frente a los 79M de una 3D-ConvNet tipo C3D— y arranca en caliente desde ImageNet.

## Resultados

Entrenando y testeando dentro de cada dataset, **los modelos I3D ganan en todos** con cualquier modalidad: Two-Stream I3D alcanza 93.4% en UCF-101, 66.4% en HMDB-51 y 74.2% en Kinetics, mostrando que **los beneficios del preentrenamiento ImageNet se extienden a las 3D ConvNets**. El preentrenamiento ImageNet sigue ayudando en todos los casos (RGB-I3D 71.1% Top-1 / 89.3% Top-5 en Kinetics). La transferencia es el corazón del argumento: entrenar solo las últimas capas tras Kinetics (régimen *Fixed*) ya rinde mucho mejor que entrenar directamente en los datasets pequeños, y con ImageNet+Kinetics + fine-tuning completo, Two-Stream I3D llega a **98.0% en UCF-101 y 81.2% en HMDB-51** (split 1). Promediando los tres splits, amplía la ventaja a **98.0% y 80.9%**, correspondiente a **reducciones de error del 63% y 35%** sobre el mejor modelo previo (Feichtenhofer et al., 94.6% / 70.3%). La explicación de la transferibilidad superior es la **alta resolución temporal** de I3D (64 cuadros vs. 10 del two-stream clásico): los métodos con entradas ralas se benefician menos porque, desde su perspectiva, los videos no difieren tanto de las imágenes de ImageNet.

## Limitaciones

- **Costo de cómputo.** Los modelos 3D consumen muchos cuadros y requieren batches grandes: se entrenaron sobre **64 GPUs**. I3D es cara de entrenar y evaluar frente a los enfoques 2D+LSTM.
- **Sigue dependiendo del flujo óptico.** Pese a que una 3D ConvNet debería aprender el movimiento del RGB, la mejor configuración **todavía necesita** un stream de flujo óptico externo precomputado (TV-L1), lo que añade un paso costoso fuera de la red y contradice parcialmente la promesa "end-to-end".
- **Exploración de arquitecturas incompleta.** No usaron *action tubes*, atención sobre los actores ni detecciones enlazadas en el tiempo.
- **Transferencia probada solo en una tarea afín.** Se demostró de Kinetics a UCF-101/HMDB-51 (la *misma* tarea con clases distintas); queda abierto si Kinetics ayuda en segmentación de video, detección o cómputo de flujo.

## Por qué importa para la Clase 36

Este paper es el nudo que la [Clase 36](/clases/clase-36) usa para amarrar el [reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones). **I3D es la síntesis** de tres líneas que la clase presenta por separado:

1. **Convolución 3D:** adopta los filtros espacio-temporales de [C3D](/papers/c3d-tran-2015), pero resuelve su defecto histórico —ser poco profunda y entrenada desde cero— vía el inflado desde una CNN 2D madura.
2. **Two-Stream (RGB + flujo óptico):** conserva la intuición de [Simonyan y Zisserman](/papers/two-stream-simonyan-2014) de que un stream de movimiento explícito aporta señal que el RGB no captura fácilmente, y muestra que sigue ayudando incluso sobre una 3D ConvNet profunda.
3. **Preentrenamiento y transferencia:** la lección de ImageNet trasladada al video. El *boring-video fixed point* hereda pesos de imagen, y [Kinetics](/papers/kinetics-kay-2017) provee el preentrenamiento de video que se transfiere a los benchmarks pequeños.

La [Clase 38](/clases/clase-38) retoma este paper como su eje: deriva la condición formal del *boring-video fixed point* en su [profundización](/clases/clase-38/profundizacion), lo implementa y verifica numéricamente en la [práctica](/clases/clase-38/practica), y lo generaliza en el fundamento [Inflado de Convoluciones](/fundamentos/inflado-de-convoluciones). Ahí también se aclara un punto de lectura fácil de confundir: la tabla de resultados que suele reproducirse en clase (93.4% en UCF-101) es la que entrena *dentro* de cada dataset; el famoso **98.0%** requiere pre-entrenar en Kinetics.

La contribución de época no es solo el número: es haber establecido el **paradigma metodológico** que rige el reconocimiento de acciones posterior —*pre-entrenar en Kinetics y transferir*— y haber dado al campo el orden que le faltaba, al re-evaluar cinco familias sobre un backbone común y un dataset grande. En términos de la clase, I3D cierra la evolución "ConvNet+LSTM → C3D → Two-Stream → 3D-Fused → I3D" y abre la era de las arquitecturas de video preentrenadas a gran escala. Para video clínico la receta es directamente aplicable: rara vez hay cientos de miles de videos anotados de endoscopías o gestos quirúrgicos, y **preentrenar en Kinetics y hacer fine-tuning** (o congelar la red en el régimen *Fixed*) transfiere la maquinaria de features espacio-temporales a la tarea médica, recuperando desempeño imposible de obtener entrenando desde cero.
