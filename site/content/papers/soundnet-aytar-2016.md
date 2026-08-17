---
title: "SoundNet: aprender sonido desde video sin etiquetar (2016)"
weight: 458
math: true
---

{{< paper-card
    title="SoundNet: Learning Sound Representations from Unlabeled Video"
    authors="Yusuf Aytar, Carl Vondrick, Antonio Torralba (MIT)"
    year="2016"
    venue="NIPS 2016 / arXiv:1610.09001"
    arxiv="1610.09001"
    pdf="/papers/soundnet-aytar-2016.pdf" >}}
El reconocimiento de sonidos naturales estaba estancado por una razón concreta: no había datasets etiquetados grandes. ESC-50 tiene 2000 clips; DCASE, diez ejemplos de entrenamiento por categoría. SoundNet resuelve el problema por un costado — **no consigue etiquetas de audio, consigue un maestro**. Toma dos millones de videos de Flickr sin anotar, hace que redes visuales ya entrenadas (ImageNet y Places) clasifiquen los fotogramas, y entrena una CNN 1D sobre la **onda cruda** para reproducir esas distribuciones. La sincronía natural entre imagen y sonido hace de puente. El resultado supera el estado del arte por unos 10 puntos en tres benchmarks —**88 % en DCASE, 74,2 % en ESC-50, 92,2 % en ESC-10**— y en inferencia la red no necesita ver nada.
{{< /paper-card >}}

---

## La idea

Tres observaciones encadenadas:

1. El video sin etiquetar se consigue **a escala masiva y gratis**.
2. Los modelos visuales de 2016 ya eran **muy buenos** reconociendo escenas y objetos.
3. En un video, lo que se ve y lo que se oye describen **la misma escena**.

De ahí la formulación como *student-teacher*: la red visual (maestro) predice sobre los fotogramas, la red de sonido (estudiante) tiene que reproducir esas predicciones oyendo solamente la onda. Con $x_i$ la forma de onda, $y_i$ el video correspondiente, $g_k$ las redes visuales y $f_k$ la de audio:

$$\min_\theta \sum_{k=1}^{K}\sum_{i=1}^{N} D_{\mathrm{KL}}\big(g_k(y_i)\;\|\;f_k(x_i;\theta)\big)$$

con $K = 2$ maestros: uno de **objetos** (ImageNet, 1000 clases) y otro de **escenas** (Places, 401), lo que da 1401 salidas.

{{< concept-alert type="clave" >}}
Lo que se transfiere no son etiquetas sino **distribuciones**. El maestro nunca vio una etiqueta de sonido, y muchas de las categorías que interesan —estornudos, por ejemplo— no existen en su vocabulario visual. Aun así el estudiante aprende una representación útil para clasificarlas, porque lo que absorbe es la **estructura semántica** del espacio de escenas, no las clases concretas. Ver [Destilación de Conocimiento](/fundamentos/destilacion-de-conocimiento).
{{< /concept-alert >}}

## La arquitectura

Una red **totalmente convolucional en 1D** sobre la onda a 22 kHz, sin capas densas. La decisión tiene una razón declarada: el sonido varía en duración, y una red convolucional acepta entradas de largo variable. La salida también es convolucional, para producir predicciones **en múltiples instantes del video** en vez de un vector único.

Reconstruyendo la Tabla 1 del paper con la fórmula estándar de convolución, la aritmética cierra en **10 de 11 capas** (la excepción es conv2, donde el paper declara 13 782 y la fórmula da 13 754). Y aparecen dos cosas que la tabla no dice:

**El campo receptivo.** Cuánto audio ve efectivamente una neurona de cada capa:

| capa | conv1 | conv3 | conv5 | pool5 | conv7 | conv8 |
|---|---|---|---|---|---|---|
| campo receptivo | 2,9 ms | 210 ms | 512 ms | 791 ms | 4,13 s | **14,54 s** |

Una neurona de conv8 integra casi **quince segundos** de audio. SoundNet no es un detector de eventos cortos sino un modelo de **escena acústica completa**, y su arquitectura lo dice antes que cualquier experimento.

**Dónde están los parámetros.** De los 14,3 millones totales:

| bloque | parámetros | fracción |
|---|---|---|
| hasta pool5 (las features que se usan) | 247 280 | **1,72 %** |
| conv7 | 2 098 176 | 14,6 % |
| conv8 (capa de salida) | 11 478 393 | **80,0 %** |

{{< concept-alert type="advertencia" >}}
La capa de salida concentra el **80 % de los parámetros** y **se descarta por completo** en la evaluación: el paper usa pool5 como extractor de features. Las representaciones que dan 74,2 % en ESC-50 provienen del **1,72 % del modelo**.

Es el mismo patrón que aparece en [C3D](/papers/c3d-tran-2015), donde 50 de sus 78 millones de parámetros están en fc6 y fc7. La cabeza cara existe para definir el objetivo de entrenamiento, no para usarse.
{{< /concept-alert >}}

## Los datos

Más de dos millones de videos de Flickr, obtenidos consultando etiquetas populares y palabras de diccionario — **más de un año de audio y video continuos**. El único preproceso: convertir a MP3, bajar a 22 kHz, pasar a un solo canal y escalar la onda al rango $[-256, 256]$. Los autores anotan que no hizo falta restar la media porque ya estaba naturalmente cerca de cero.

La elección de Flickr sobre YouTube es deliberada: videos **no editados profesionalmente**, clips cortos de situaciones cotidianas.

## Resultados

| DCASE | Accuracy | | ESC-50 | ESC-10 |
|---|---|---|---|---|
| RG | 69 % | SVM-MFCC | 39,6 % | 67,5 % |
| LTT | 72 % | Autoencoder convolucional | 39,9 % | 74,3 % |
| RNH | 77 % | Random Forest | 44,3 % | 72,7 % |
| Ensemble | 78 % | Piczak ConvNet | 64,5 % | 81,0 % |
| **SoundNet** | **88 %** | **SoundNet** | **74,2 %** | **92,2 %** |
| | | *Humanos* | *81,3 %* | *95,7 %* |

Diez puntos sobre el estado del arte en los tres. En ESC-10 se acerca al desempeño humano, aunque los autores advierten que ese dataset puede ser fácil.

## La ablación, que es lo más instructivo

| Comparación | Configuración | ESC-50 | ESC-10 |
|---|---|---|---|
| **Pérdida** | 8 capas, pérdida $\ell_2$ | **47,8 %** | 81,5 % |
| | 8 capas, pérdida KL | **72,9 %** | 92,2 % |
| **Maestro** | solo ImageNet | 69,5 % | 89,8 % |
| | solo Places | 71,1 % | 89,5 % |
| | ambos | 72,9 % | 92,2 % |
| **Profundidad** | 5 capas, desde cero | **65,0 %** | 82,3 % |
| | 8 capas, desde cero | **51,1 %** | 75,5 % |
| | 5 capas, video sin etiquetar | 66,1 % | 86,8 % |
| | 8 capas, video sin etiquetar | **72,9 %** | 92,2 % |

Tres lecturas:

**KL contra $\ell_2$: 25 puntos de diferencia.** Es la brecha más grande de la tabla y merece atención, porque [Hinton](/papers/distillation-hinton-2015) demostró que a temperatura alta ambas pérdidas son equivalentes. La explicación es que aquí no hay temperatura alta ($T=1$) y, sobre todo, que el $\ell_2$ del paper se aplica a las **salidas** —las probabilidades— y no a los logits. Esa distinción decide el resultado y se desarrolla en la [profundización](/clases/clase-43/profundizacion) de la clase.

**El bloque de profundidad es el argumento central del paper, medido.** Sin transferencia, **pasar de 5 a 8 capas empeora** el resultado (65,0 → 51,1): la red se sobreajusta a un conjunto de entrenamiento de miles de ejemplos. Con transferencia desde video sin etiquetar, la misma profundización **mejora** (66,1 → 72,9). La frase de la clase —*"más datos permiten construir redes más profundas sin sobreajustar"*— no es una intuición: es esta tabla.

**Más maestros es mejor.** Combinar ImageNet y Places supera a cualquiera solo, lo que sugiere, en palabras de los autores, que *"el progreso en comprensión de sonido puede avanzar construyendo modelos de visión más fuertes"*. El estudiante queda acotado por lo que sus maestros saben.

## Qué capa y qué maestro

| Dataset | Modelo | conv4 | conv5 | pool5 | conv6 | conv7 | conv8 |
|---|---|---|---|---|---|---|---|
| DCASE | 8 capas, AlexNet | 84 % | 85 % | 84 % | 83 % | 78 % | 68 % |
| | 8 capas, VGG | 77 % | **88 %** | **88 %** | 87 % | 84 % | 74 % |
| ESC-50 | 8 capas, AlexNet | 66,0 % | 71,2 % | **74,2 %** | 74,0 % | 63,8 % | 45,7 % |
| | 8 capas, VGG | 66,0 % | 69,3 % | 72,9 % | 73,3 % | 59,8 % | 43,7 % |

**pool5 gana en general**, y el rendimiento **cae fuerte** en las últimas capas — conv8 llega a 45,7 % en ESC-50, casi 30 puntos por debajo de pool5. Las capas finales están especializadas en reproducir las 1401 categorías visuales y pierden generalidad, exactamente el patrón que se ve al transferir features en visión.

Sobre el maestro, el paper es honesto: *"los resultados son inconcluyentes"* — VGG gana en DCASE, AlexNet en ESC-50. Esto también explica una discrepancia interna que confunde al leer: la tabla principal reporta 74,2 % (AlexNet, pool5) y la de ablación 72,9 %, porque esta última usa VGG.

## Lo que no funciona tan bien

El experimento multimodal del paper, sobre 44 categorías de un conjunto anotado aparte:

| Features | sonido | visión | visión + sonido |
|---|---|---|---|
| 8 capas, conv7 | 32,4 % | 49,4 % | **51,4 %** |

**La visión sola supera al sonido por 17 puntos**, y agregar sonido a la visión aporta apenas **+2 puntos**. Es un resultado sobrio en un paper por lo demás triunfal: para categorizar escenas, mirar sigue siendo mucho mejor que escuchar. La complementariedad existe pero es modesta — muy distinto del régimen de [E2E-AVSR](/papers/e2e-avsr-petridis-2018), donde la modalidad débil aporta 14 puntos porque la fuerte está degradada por el ruido.

## Por qué importa para la Clase 43

Es el primero de los dos papers de la [Clase 43](/clases/clase-43) y el que instala su tesis: **la supervisión puede venir de otra modalidad**. Su lugar en la genealogía del área es el de precursor de toda la línea audiovisual autosupervisada — [Look, Listen and Learn](/papers/look-listen-learn-arandjelovic-2017) (2017) reemplaza la destilación por correspondencia simétrica, y [AV-HuBERT](/papers/av-hubert-shi-2022) (2022) la lleva al régimen de predicción enmascarada.

Y ofrece una lección transferible más allá del audio: cuando faltan etiquetas, a veces lo que falta no es anotación sino **una segunda vista del mismo fenómeno** que ya esté alineada con la primera.

---

**Ver también:** [Destilación (2015)](/papers/distillation-hinton-2015) · [E2E-AVSR (2018)](/papers/e2e-avsr-petridis-2018) · [Look, Listen and Learn (2017)](/papers/look-listen-learn-arandjelovic-2017) · [VGGish (2017)](/papers/vggish-hershey-2017) · [ESC-50 (2015)](/papers/esc50-piczak-2015) · [Destilación de Conocimiento](/fundamentos/destilacion-de-conocimiento) · [Aprendizaje Audiovisual](/fundamentos/aprendizaje-audiovisual)
