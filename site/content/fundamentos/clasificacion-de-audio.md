---
title: "Clasificación de Audio: tagging, detección de eventos y etiquetas débiles"
weight: 128
math: true
---

"Clasificar audio" nombra al menos cuatro tareas distintas, con datasets, arquitecturas, funciones de pérdida y métricas que no se intercambian. La diferencia entre ellas no está en la señal sino en **la forma de la etiqueta**: si hay una sola o varias, y si vienen con marca de tiempo o no. Confundirlas es la fuente número uno de resultados que parecen buenos y no lo son — un modelo evaluado con la métrica de la tarea equivocada puede reportar 95% de exactitud y ser inservible.

Este fundamento ordena el mapa: las dos dimensiones que generan la taxonomía, la distinción entre etiquetas fuertes y débiles y por qué existe, la función de pérdida que corresponde a cada caso, las métricas y cuándo engañan, y las trampas de evaluación específicas del audio. Es el marco que la [Clase 39](/clases/clase-39) presenta en sus slides 8 a 10 sin nombrarlo con la terminología del campo.

---

## 1. Las dos dimensiones

Toda tarea de clasificación de audio se ubica cruzando dos preguntas independientes:

|  | **Etiqueta global** (todo el clip) | **Etiquetas locales** (con marca de tiempo) |
|---|---|---|
| **Una etiqueta** | Clasificación de escena acústica, identificación de hablante, clasificación de género musical | Segmentación: qué suena en cada instante, sin solapamiento |
| **Múltiples etiquetas** | **Audio tagging** — "hay un perro, una sirena y gente hablando" | **Detección de eventos sonoros (SED)** — "perro de 2.1 a 3.4 s, sirena de 0.0 a 8.0 s" |

Los nombres estándar de las dos celdas de la derecha del eje "múltiples":

- **Audio tagging**: predecir el conjunto de eventos presentes en un clip, sin localizarlos. Es la tarea de [AudioSet](/papers/audioset-gemmeke-2017) y la que resuelve [VGGish](/papers/vggish-hershey-2017).
- **Sound event detection (SED)**: predecir además cuándo empieza y termina cada evento. Es la tarea central de los desafíos DCASE.

{{< concept-alert type="clave" >}}
**Por qué importa usar los nombres del campo.** Buscar "audio classification" en la literatura devuelve las cuatro tareas mezcladas, con cifras que no se pueden comparar entre sí. Buscar "weakly labeled sound event detection" devuelve el problema concreto, sus datasets y sus métricas.

La [Clase 39](/clases/clase-39) presenta esta taxonomía con ejemplos —"John está hablando" contra "John dice: *you* / *must* / *know* / *AI*"— pero sin la terminología, lo que dificulta rastrear la literatura. La correspondencia es directa: *single global-label* es clasificación, *multiple global-labels* es tagging, *multiple local-labels* es SED.
{{< /concept-alert >}}

---

## 2. Etiquetas fuertes y débiles

La distinción entre tagging y SED no es solo de tarea: es de **qué datos existen**.

- Una **etiqueta fuerte** (*strong label*) especifica el evento y su intervalo temporal: `perro, 2.1 s – 3.4 s`.
- Una **etiqueta débil** (*weak label*) solo dice que el evento está presente en alguna parte del clip: `perro`.

La asimetría entre ambas es económica. Anotar débilmente un clip de 10 segundos toma segundos: se escucha y se marcan las casillas. Anotar fuertemente el mismo clip toma minutos, exige revisar el espectrograma y **no converge entre anotadores**: dos personas competentes discrepan sistemáticamente sobre dónde empieza exactamente un evento que aparece gradualmente, y sobre si un ruido de fondo intermitente cuenta como un evento o como parte del ambiente.

Esa discrepancia pone un techo a la calidad alcanzable con datos reales, y es la razón de ser de herramientas como [Scaper](/papers/scaper-salamon-2017): si el paisaje sonoro se **sintetiza** mezclando eventos sobre un fondo, las marcas de tiempo son exactas por construcción, porque quien genera la mezcla sabe con precisión de milisegundo cuándo insertó cada evento.

### Entrenar SED con etiquetas débiles

El caso interesante —y el habitual, porque los datos débiles abundan y los fuertes escasean— es entrenar un modelo que localice eventos disponiendo solo de etiquetas de clip. Es un problema de **aprendizaje de instancias múltiples** (*multiple instance learning*, MIL): el clip es una "bolsa" de segmentos, se sabe que la bolsa es positiva, y hay que inferir cuáles de sus segmentos lo son.

La implementación estándar es un **agregador de pooling** sobre el eje temporal: la red produce una predicción por trama, y una operación de agregación las colapsa a una predicción de clip que es la que recibe la supervisión. La elección del agregador tiene consecuencias:

| Agregador | Comportamiento | Riesgo |
|---|---|---|
| **Max** | La predicción del clip es la trama más segura | Puede activarse por una sola trama espuria; gradiente que llega a una trama por paso |
| **Mean** | Promedio de todas las tramas | Diluye eventos cortos: un evento de 0.2 s en un clip de 10 s aporta el 2% |
| **Atención** | Promedio ponderado con pesos aprendidos | El más usado hoy; los pesos son interpretables como localización |
| **Softmax lineal / exponencial** | Interpolación entre max y mean | Requiere calibrar un hiperparámetro |

{{< concept-alert type="advertencia" >}}
**El pooling que se elige determina qué aprende el modelo, no solo cómo agrega.** Con *mean pooling*, un evento breve dentro de un clip largo produce una señal de entrenamiento tan diluida que el modelo aprende a ignorarlo — y luego el mismo modelo, evaluado con métricas de clip, parece funcionar bien. Con *max pooling*, el modelo puede aprender a detectar un artefacto correlacionado con la etiqueta en lugar del evento.

El síntoma a vigilar: exactitud alta a nivel de clip con localización temporal mala. Si solo se mide lo primero, el problema es invisible.
{{< /concept-alert >}}

---

## 3. La función de pérdida: softmax o sigmoides

La consecuencia más inmediata de la taxonomía, y el error más frecuente al empezar.

**Una etiqueta por clip** → softmax sobre las clases con entropía cruzada categórica. Las probabilidades suman 1, lo que codifica que las clases son mutuamente excluyentes.

$$p_c = \frac{e^{z_c}}{\sum_{c'} e^{z_{c'}}}, \qquad \mathcal{L} = -\log p_{y}$$

**Múltiples etiquetas simultáneas** → una sigmoide independiente por clase con entropía cruzada binaria. Cada clase se decide por separado.

$$p_c = \sigma(z_c), \qquad \mathcal{L} = -\sum_c \big[y_c \log p_c + (1-y_c)\log(1-p_c)\big]$$

{{< concept-alert type="clave" >}}
**Por qué el softmax es incorrecto en audio multi-etiqueta, y por qué esto es más grave en audio que en imágenes.** Un softmax obliga al modelo a **repartir masa de probabilidad** entre eventos que ocurren a la vez: si un perro ladra mientras pasa una sirena, subir la confianza en "perro" fuerza a bajar la de "sirena", aunque ambas sean correctas.

Y en audio la simultaneidad no es un caso raro: es el caso normal. Los sonidos se **suman** en el espectrograma en lugar de ocluirse como los objetos visuales, así que la mezcla de fuentes es la regla y no la excepción. Un dataset urbano típico tiene solapamiento en la mayoría de sus clips. El [slide 43 de la Clase 39](/clases/clase-39/teoria) lo menciona al pasar —"softmax (o sigmoides)"— y es una de sus indicaciones más importantes.
{{< /concept-alert >}}

---

## 4. Las métricas, y cuándo engañan

| Tarea | Métrica estándar | Qué mide |
|---|---|---|
| Clasificación (una etiqueta) | Exactitud, F1 macro | Fracción de aciertos; el F1 macro pondera todas las clases igual |
| Audio tagging | **mAP**, ROC-AUC, **PR-AUC** | Calidad del ranking por clase, promediada |
| SED | **Error rate** y **F1 basado en eventos** o en segmentos | Aciertos con tolerancia temporal |

### ROC-AUC contra PR-AUC

En tagging, el desbalance es extremo: una clase presente en el 0.1% de los clips es normal en AudioSet. Con ese desbalance, **el ROC-AUC engaña**, porque su eje de falsos positivos se normaliza por el número de negativos, que es enorme. Un clasificador puede tener ROC-AUC de 0.95 y una precisión inservible.

El PR-AUC no tiene ese problema: su línea base es la prevalencia de la clase, así que un valor de 0.30 sobre una clase con prevalencia 0.001 es un resultado excelente y se lee como tal. **La regla: reportar los dos, y sospechar cuando solo se reporta el ROC-AUC.** Ver [métricas de ranking](/fundamentos/ranking-metrics).

### F1 por eventos contra F1 por segmentos

En SED hay dos formas de contar un acierto:

- **Por segmentos**: se divide el clip en ventanas (típicamente 1 s) y se compara la predicción con la referencia en cada una. Es indulgente con los límites.
- **Por eventos**: se cuenta un acierto si el evento predicho coincide con uno real dentro de una tolerancia (típicamente 200 ms en el inicio, y una fracción de la duración en el final). Es mucho más exigente.

Las cifras de ambos difieren tanto que compararlas entre sí no tiene sentido. Un sistema con F1 por segmentos de 0.70 puede tener F1 por eventos de 0.35.

---

## 5. Las trampas de evaluación propias del audio

Tres formas de inflar un resultado sin darse cuenta. Las tres son específicas de esta modalidad.

### Fragmentos del mismo archivo repartidos entre train y test

La más común y la más costosa. Los datasets de audio suelen construirse cortando grabaciones largas en fragmentos. Dos fragmentos del mismo archivo original comparten el ambiente acústico, el micrófono, el nivel de ruido de fondo y a menudo la misma fuente sonora. Si uno cae en entrenamiento y otro en test, el modelo puede reconocer la **grabación** en lugar del **evento**.

Por eso [UrbanSound8K](/papers/urbansound8k-salamon-2014) viene con **10 folds preestablecidos** y su documentación insiste en no re-barajarlos: los folds están construidos para que todos los fragmentos de una misma grabación caigan juntos. Re-barajar produce exactitudes por encima del 90% que se desploman con datos nuevos.

{{< concept-alert type="advertencia" >}}
**El síntoma: una exactitud sospechosamente alta que no se replica.** Si un modelo simple da 92% sobre un dataset donde el estado del arte publicado está en 75%, la primera hipótesis debería ser fuga de datos, no genialidad. Verificar de dónde vienen los splits antes de creer cualquier número.

Vale también al revés: si un resultado propio es mucho peor que el publicado, comprobar que el protocolo sea el mismo. Muchas cifras de la literatura de audio usan un solo fold como test y otras el promedio de los diez; no son comparables.
{{< /concept-alert >}}

### Fuga por hablante o por instrumento

El análogo en habla y música. Si el mismo hablante aparece en entrenamiento y test, un modelo de detección de emoción puede estar reconociendo a la persona. Los datasets serios de habla separan por hablante, y los de música por artista o álbum.

### Correlaciones de fondo

Si todas las grabaciones de una clase provienen de la misma fuente —el mismo canal de YouTube, la misma campaña de grabación— el modelo puede aprender la firma del canal. Es la versión sonora del problema clásico de los datasets de imágenes médicas donde el modelo aprende a reconocer el hospital por la marca de agua del equipo.

---

## 6. Las arquitecturas, por tarea

| Tarea | Arquitectura típica hoy | Por qué |
|---|---|---|
| Clasificación de clip | CNN sobre log-mel, o [AST](/papers/ast-gong-2021) | La salida es una etiqueta: se puede colapsar el eje temporal |
| Audio tagging | [VGGish](/papers/vggish-hershey-2017), PANNs, AST, BEATs | Ídem, con sigmoides y pooling con atención |
| SED | [CRNN](/fundamentos/crnn) o Transformer con salida por trama | La salida es densa: no se puede colapsar el tiempo |
| Identificación de hablante | Encoder + pérdida métrica (GE2E, AAM-softmax) | Se necesita un espacio de embeddings, no un clasificador cerrado |

La diferencia estructural está en si la salida es densa. Para clasificación y tagging, todo el eje temporal se colapsa al final y el stride agresivo es la herramienta más barata para crecer el campo receptivo. Para SED, la salida tiene una predicción por trama, así que la reducción de resolución tiene un límite y aparecen las [convoluciones dilatadas](/fundamentos/convoluciones-dilatadas).

La identificación de hablante es un caso aparte: el conjunto de clases es abierto (hay que reconocer personas no vistas durante el entrenamiento), así que no se entrena un clasificador sino un espacio donde la distancia sea informativa. Ver [metric learning](/fundamentos/metric-learning) y [triplet loss](/fundamentos/triplet-loss).

---

## 7. Lista de verificación

Antes de creer un número de clasificación de audio, propio o ajeno:

1. **¿Qué celda de la taxonomía es?** Una etiqueta o varias; global o localizada.
2. **¿La pérdida corresponde?** Softmax solo si las clases son mutuamente excluyentes.
3. **¿La métrica corresponde?** PR-AUC o mAP en multi-etiqueta desbalanceada; F1 por eventos si se afirma localización.
4. **¿De dónde salen los splits?** Si el dataset trae folds oficiales, usarlos. Si no, verificar que no haya fragmentos del mismo archivo, hablante o sesión repartidos.
5. **¿Se midió la localización, o solo el clip?** En SED entrenada con etiquetas débiles, la exactitud de clip puede ser alta con localización mala.
6. **¿El protocolo es el mismo que el de las cifras con las que se compara?** Un fold contra diez folds no son comparables.

---

## Ver también

**Papers:** [AudioSet (2017)](/papers/audioset-gemmeke-2017) · [UrbanSound8K (2014)](/papers/urbansound8k-salamon-2014) · [ESC-50 (2015)](/papers/esc50-piczak-2015) · [FSD50K (2020)](/papers/fsd50k-fonseca-2020) · [VGGish (2017)](/papers/vggish-hershey-2017) · [AST (2021)](/papers/ast-gong-2021) · [Scaper (2017)](/papers/scaper-salamon-2017) · [Deep Learning for Audio Signal Processing (2019)](/papers/dl-audio-purwins-2019).

**Fundamentos:** [Representación de audio](/fundamentos/representacion-de-audio) · [Datasets de audio](/fundamentos/datasets-de-audio) · [MFCC y escala Mel](/fundamentos/mfcc-y-escala-mel) · [CRNN](/fundamentos/crnn) · [Convoluciones dilatadas](/fundamentos/convoluciones-dilatadas) · [Métricas de ranking](/fundamentos/ranking-metrics) · [Metric learning](/fundamentos/metric-learning).

**Clases:** [Clase 39 - Modelos de DL para audio](/clases/clase-39) · [Clase 37 - Datasets y herramientas](/clases/clase-37) · [Dominio: Audio / Voz](/dominios/audio).
