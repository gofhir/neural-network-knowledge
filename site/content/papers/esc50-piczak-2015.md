---
title: "ESC-50: clasificación de sonido ambiental (2015)"
weight: 413
math: true
---

{{< paper-card
    title="ESC: Dataset for Environmental Sound Classification"
    authors="Karol J. Piczak"
    year="2015"
    venue="ACM Multimedia 2015"
    pdf="/papers/esc50-piczak-2015.pdf" >}}
Piczak liberó la colección **ESC (Environmental Sound Classification)** para paliar un problema concreto del campo: la falta de un benchmark abierto, balanceado y comparable para clasificar **sonidos ambientales** —eventos de audio cotidiano que no son ni habla ni música—. El aporte tiene tres piezas: **ESC-50** (2.000 clips etiquetados de 5 s en 50 clases, 40 por clase, agrupadas en 5 categorías), **ESC-10** (subconjunto más fácil de 10 clases) y **ESC-US** (250.000 clips **sin etiquetar** para preentrenamiento no supervisado), todas construidas desde grabaciones de Freesound. Lo metodológicamente valioso: mide el **techo humano** por crowdsourcing (**≈81,3 %** en ESC-50, **≈95,7 %** en ESC-10) y lo contrasta con baselines clásicos (MFCC + k-NN/RF/SVM) que quedan muy por debajo (**44,3 %** en ESC-50). Esa **brecha humano-máquina** de casi 37 puntos es el argumento del paper: hay margen enorme, y el deep learning es el camino para cerrarlo. Es el dataset didáctico de la [Clase 37](/clases/clase-37).
{{< /paper-card >}}

---

## Contexto: la falta de un benchmark abierto y balanceado

En los años previos, el auge del deep learning había transformado la percepción por máquina —sobre todo en **visión**—, pero en audio el esfuerzo se concentraba casi exclusivamente en **habla y música**. El análisis de sonidos ambientales quedó rezagado, pese a aplicaciones evidentes: vigilancia acústica, audífonos, monitoreo de habitaciones y resúmenes de video.

El diagnóstico de Piczak es que el campo estaba **fragmentado**: la mayoría de los estudios usaban conjuntos pequeños, específicos o (semi)propietarios, muchas veces sin código disponible, lo que hacía la **reproducibilidad** difícil. Lo contrasta directamente con la visión por computador, donde **MNIST** y **CIFAR** funcionaban como estándar de facto. En audio ambiental no había equivalente: solo iniciativas recientes como **UrbanSound** empezaban a cambiar el panorama. De ahí el objetivo declarado: facilitar la investigación abierta mediante un dataset público, una estimación de la exactitud humana sobre él, una comparación con baselines comunes y un notebook Jupyter para replicar todo.

## Composición: ESC-50, ESC-10 y ESC-US

Las tres partes comparten formato de clip corto unificado (5 s, 44,1 kHz, mono, Ogg Vorbis a 192 kbit/s) y provienen de Freesound.

- **ESC-50** — 2.000 grabaciones etiquetadas, perfectamente balanceadas en **50 clases** (40 clips cada una), agrupadas en 5 categorías laxas: sonidos de animales; paisajes naturales y agua; sonidos humanos no vocales; interiores/domésticos; ruidos exteriores/urbanos. La extracción buscó dejar el evento en primer plano, con dificultad deliberadamente mixta: fuentes muy comunes (risa, ladrido), otras muy distintivas (vidrio rompiéndose) y algunas confundibles (helicóptero vs. avión). Su limitación reconocida es el número reducido de clips por clase.
- **ESC-10** — 10 clases seleccionadas del conjunto mayor, concebidas como prueba de concepto y problema más fácil (transitorios percusivos, eventos armónicos, ruido/paisajes estructurados). Fija un listón muy alto: como clasificarlas es trivial para un humano, un modelo debería aspirar a un desempeño casi perfecto.
- **ESC-US** — 250.000 grabaciones **sin anotación verificada**, extraídas de archivos de Freesound etiquetados como *field recording*, pensadas para **preentrenamiento no supervisado**, modelos generativos y regímenes débilmente supervisados. Se apoya solo en la moderación colaborativa de Freesound.

Una decisión metodológica clave hace de ESC un benchmark honesto: los conjuntos etiquetados se organizan en **5 folds de validación cruzada** con la regla de que los clips de un **mismo archivo fuente** quedan siempre en un **único fold**. Sin ese *source filtering*, un modelo podría reconocer características idiosincrásicas de la grabación (fondo, micrófono, sala) en lugar de la clase, inflando la exactitud por fuga de información. Al confinar cada fuente a un fold, la evaluación mide **generalización a fuentes nuevas** y garantiza cifras **comparables** entre trabajos.

## El techo humano y los baselines clásicos

Para estimar el desempeño humano, Piczak recurrió a crowdsourcing (**CrowdFlower**), recolectando ~4.000 juicios por dataset (una docena de clasificaciones por clip en ESC-10 y solo **dos** por clip en ESC-50). Resultado: **95,7 % en ESC-10** y **81,3 % en ESC-50**, con recall por clase muy dispar (desde 34,1 % en ruido de lavadora hasta casi 100 % en bebés llorando y ladridos). El autor proyecta que oyentes entrenados alcanzarían niveles impecables en ESC-10 y del orden del **90 %** en ESC-50.

Los baselines extrajeron **tasa de cruces por cero** y los **primeros 12 MFCC** (con librosa, tramas de 11,6 ms), resumidos por media y desviación estándar, alimentando k-NN, random forest y SVM lineal sobre los 5 folds. Un MFCC resume la envolvente espectral de corto plazo: se agrupa el espectro de potencia en un banco de filtros triangulares espaciados en la **escala mel**

$$m = 2595 \, \log_{10}\!\left(1 + \frac{f}{700}\right)$$

se toma el logaritmo de la energía de cada banda y se aplica una DCT; los primeros coeficientes son los MFCC. La escala mel imita la resolución no lineal del oído en frecuencia.

Los resultados quedan muy por debajo del humano:

| Modelo | ESC-10 | ESC-50 |
|---|---|---|
| k-NN | 66,7 % | 32,2 % |
| SVM (kernel lineal) | 67,5 % | 39,6 % |
| Random forest | **72,7 %** | **44,3 %** |
| **Humano** | **95,7 %** | **81,3 %** |

El random forest fue el mejor en ambos casos; la caída pronunciada del k-NN al pasar a ESC-50 sugiere dependencias entre features más intrincadas que los modelos simples no capturan. El techo humano cumple aquí el rol que en visión cumplían las tasas de error humano en ImageNet: convierte "mejorar la exactitud" en "acercarse a —y superar— al oyente humano".

## Impacto

ESC-50 y ESC-10 se volvieron rápidamente el **benchmark estándar** de la clasificación de sonido ambiental, cumpliendo el objetivo de dar al campo un equivalente abierto a MNIST/CIFAR: tamaño manejable, balance perfecto, folds predefinidos con filtrado por fuente, licencia abierta y distribución trivial. La predicción del autor se cumplió: los sistemas basados en CNN sobre espectrogramas, y luego los modelos preentrenados a gran escala, terminaron **superando el techo humano** de 81,3 %, alcanzando exactitudes por encima del 90 % y del 95 %. ESC-US, por su parte, anticipó la lógica del **preentrenamiento no supervisado/auto-supervisado** en audio que años después se volvería dominante.

## Limitaciones

- **Pocos clips por clase.** 40 grabaciones por clase en ESC-50 es escaso para métodos que aprenden representaciones ricas; es la razón de ser de ESC-US.
- **Selección de clases subjetiva.** Las clases se eligieron según utilidad y distinción percibidas, sin criterio taxonómico formal.
- **Solapamiento de fondo.** Las grabaciones de campo no son estériles; algunos clips conservan ruido o eventos secundarios.
- **Estimación humana informal.** Con ~2 juicios por clip en ESC-50 y participantes no entrenados, es una estimación aproximada, no una medición rigurosa.
- **Baselines deliberadamente débiles.** No buscan ser competitivos; la evaluación de métodos robustos (CNN) se delega a otro trabajo del autor.

## Por qué importa para la Clase 37

La [Clase 37](/clases/clase-37), segunda del bloque de audio, cita ESC-50 como **dataset didáctico** de referencia. Tres ideas conviene internalizar:

1. **Anatomía de un buen benchmark.** ESC-50 ejemplifica qué hace confiable a un dataset —balance estricto, formato unificado, **folds predefinidos con filtrado por fuente**, distribución abierta—, decisiones transferibles a cualquier dominio y centrales entre los [datasets de audio](/fundamentos/datasets-de-audio).
2. **El techo humano como referencia.** Medir y publicar la exactitud humana (≈81 % en ESC-50, ≈95 % en ESC-10) convierte una métrica abstracta en una meta concreta, el mismo principio que guió el progreso en visión.
3. **La brecha que motiva el deep learning.** El salto entre los baselines clásicos (~44 % en ESC-50) y el humano (~81 %) es la justificación empírica de las redes profundas. ESC-50 fue el terreno donde esa promesa se verificó y, con el tiempo, se superó al oyente humano.

La lección tiene valor clínico directo: en el **monitoreo ambiental de pacientes** —detectar tos, caídas, ronquidos o alarmas en una habitación mediante clasificación de sonido— importa saber no solo qué exactitud logra un modelo, sino cuán bien lo haría un clínico atento escuchando lo mismo. Un benchmark abierto, balanceado y con techo humano explícito es tan aplicable a los sonidos de una sala de hospital como a los ladridos y sirenas de ESC-50.
