---
title: "Reconocimiento de Hablante"
weight: 130
math: true
---

El **reconocimiento de hablante** es la tarea de determinar *quién* habla a partir de la señal de voz, en oposición al [reconocimiento de voz](/fundamentos/reconocimiento-de-voz), que determina *qué* se dijo. La distinción no es solo de objetivo: los dos problemas quieren cosas opuestas de la representación. El ASR necesita **resolución temporal** —cada fonema en su lugar—; el reconocimiento de hablante necesita **colapsar el enunciado entero** en un vector que descarte el contenido lingüístico y conserve lo invariante de la voz. De esa tensión salen todas las decisiones de diseño del área, y muy especialmente la que resulta decisiva: **cómo se agregan los frames**.

---

## 1. Tres tareas que se confunden

| Tarea | Pregunta | Salida | Conjunto de identidades |
|---|---|---|---|
| **Identificación** | ¿quién de estos $N$ es? | una etiqueta | cerrado, conocido |
| **Verificación** | ¿estos dos audios son de la misma persona? | sí / no | **abierto** |
| **Diarización** | ¿quién habló y cuándo? | segmentos etiquetados | abierto, desconocido |

La distinción entre las dos primeras determina la arquitectura, y es el argumento con que abre la [Clase 41](/clases/clase-41).

**Identificación como clasificación.** Un modelo con $N$ salidas y softmax que da la probabilidad de cada hablante. Funciona, pero tiene un defecto estructural que la clase señala de inmediato: *"our model must be trained entirely for each new speaker"*. Incorporar una persona significa cambiar la capa de salida y reentrenar. En cualquier aplicación real —un banco, un hospital, un asistente— eso es inviable.

**Verificación mediante descriptores.** La alternativa es que el modelo produzca un **vector** por enunciado y comparar vectores:

$$\text{score} = v_1 \cdot v_2$$

Si el puntaje es alto, misma persona; si es bajo, personas distintas. Agregar a alguien no requiere tocar el modelo: se calcula su vector y se guarda. El sistema funciona sobre un **conjunto abierto** de identidades, y puede evaluarse sobre personas que nunca vio.

{{< concept-alert type="clave" >}}
El cambio de clasificador a descriptor es el mismo que ocurre en reconocimiento facial con [FaceNet](/papers/facenet-schroff-2015) y en recuperación de imágenes: en vez de aprender a nombrar clases, se aprende un **espacio métrico** donde la distancia significa identidad. La red se entrena típicamente con una pérdida de clasificación sobre miles de hablantes, y el descriptor se toma de una capa intermedia — la clasificación es el pretexto, el espacio es el producto. Ver [Metric learning](/fundamentos/metric-learning).
{{< /concept-alert >}}

---

## 2. Qué hace difícil el problema

Tres características de la señal, que la clase enumera:

- **Longitud variable.** Los enunciados duran de uno a varios cientos de segundos, y el descriptor debe tener dimensión fija.
- **Contenido irrelevante.** Silencios, ruido ambiente, música, risas y **otras voces**. En habla "in the wild" pueden ser una fracción sustancial del audio.
- **Frecuencia de muestreo común**, que suele asumirse por convención (16 kHz).

Y los objetivos de diseño que se derivan:

1. Representar cada señal con un **vector de dimensión fija**.
2. **Filtrar** las partes irrelevantes.
3. Capturar los **componentes básicos de la voz humana**.

La estrategia estándar tiene dos etapas: **dividir la señal en frames** y calcular descriptores locales, y después **agregar en el tiempo** incluyendo solo lo relevante. La primera etapa es el *frame level*; la segunda, el *utterance level*. Y ahí está el problema central.

---

## 3. El problema de la agregación

Agregar $T$ descriptores de frame en un vector fijo es donde se decide el rendimiento del sistema. La progresión histórica de métodos es una sola pregunta llevada cada vez más lejos: **cuánta estructura se le permite a este paso**.

| Método | Qué calcula | EER en VoxCeleb1 |
|---|---|---|
| **Temporal average pooling** (TAP) | media sobre el tiempo | 4,19 – 10,48 % |
| **Statistics pooling** ([x-vectors](/papers/x-vectors-snyder-2018)) | media **+ desviación estándar** | ~4,70 % |
| **Attentive statistics pooling** | media y std ponderadas por atención aprendida | 3,85 % |
| **NetVLAD / GhostVLAD** ([Xie et al.](/papers/utterance-level-xie-2019)) | residuos respecto de un diccionario aprendido | **3,22 %** |

El salto del promedio a los residuos es grande y está medido de forma limpia: en [Xie et al. (2019)](/papers/utterance-level-xie-2019), **con el mismo backbone, los mismos datos y la misma pérdida**, pasar de TAP a NetVLAD lleva el EER de 10,48 % a 3,57 %.

La razón que dan los autores es precisa. Las features obtenidas con promedio temporal son buenas **separando hablantes distintos** (varianza inter-clase) pero malas **compactando al mismo hablante** (varianza intra-clase): promediar sobre un enunciado con ruido y silencios produce vectores que se mueven mucho de un enunciado a otro de la misma persona. Por eso los sistemas basados en TAP necesitan pérdidas contrastivas con minería de ejemplos difíciles para funcionar, mientras que la agregación por diccionario llega más lejos con softmax simple.

El mecanismo de VLAD y su versión diferenciable está en [Agregación VLAD](/fundamentos/agregacion-vlad).

---

## 4. Cómo se evalúa: del score al umbral

Un sistema de verificación produce un número continuo y hay que decidir un umbral. La clase plantea la pregunta de forma directa: *"¿cuál es el valor límite para determinar qué es bajo y qué es alto?"*.

Sobre un conjunto de pares etiquetados se calculan, **en función del umbral**:

$$\text{TPR} = \frac{TP}{TP + FN}, \qquad \text{FPR} = \frac{FP}{TN + FP}$$

Barriendo el umbral de 0 a 1 y graficando TPR contra FPR se obtiene la **curva ROC**. Cada punto de la curva es una política de decisión distinta: umbral alto significa pocos falsos positivos y muchos falsos negativos, umbral bajo lo contrario.

**El EER** (*equal error rate*) es el punto de la curva donde la tasa de falsos positivos iguala a la de falsos negativos:

$$\text{FPR}(\tau^*) = 1 - \text{TPR}(\tau^*) = \text{EER}$$

Es la métrica estándar del área porque **resume el sistema en un número sin comprometerse con un umbral operativo**. Un EER de 3,22 % significa que existe un umbral en el que el sistema se equivoca en el 3,22 % de los casos en ambas direcciones por igual.

{{< concept-alert type="cuidado" >}}
**El EER es una métrica de comparación, no de operación.** En un despliegue real los dos errores casi nunca cuestan lo mismo. En control de acceso a un sistema clínico, un falso positivo —dejar entrar a quien no es— puede ser inaceptable, mientras que un falso negativo solo obliga a reintentar. El punto de operación debe elegirse sobre la curva ROC según ese costo asimétrico, no en el EER. El área bajo la curva (AUC) resume el sistema completo; el DCF (*detection cost function*), usado en las evaluaciones NIST, pondera explícitamente ambos errores.
{{< /concept-alert >}}

---

## 5. Duración: la variable que se subestima

El rendimiento depende fuertemente de cuánto audio hay disponible. Medido sobre VoxCeleb1 con el mismo modelo:

| Duración | 2 s | 3 s | 4 s | 5 s | 6 s |
|---|---|---|---|---|---|
| EER | 7,97 % | 5,73 % | 4,70 % | 4,10 % | **3,39 %** |

Más del doble de error con 2 segundos que con 6. La explicación es probabilística: en habla no controlada una fracción del audio es ruido, silencio o voz ajena, y **un segmento corto puede caer mayoritariamente en esa fracción**. Al alargarlo, la probabilidad de capturar voz útil del hablante crece.

Tiene una consecuencia práctica: al reportar un EER hay que decir sobre qué duración se midió, o el número no significa nada.

---

## 6. Datasets y protocolo

La familia **VoxCeleb** define el estándar actual:

| | [VoxCeleb1](/papers/voxceleb-nagrani-2017) | [VoxCeleb2](/papers/voxceleb2-chung-2018) |
|---|---|---|
| Hablantes | 1 251 | 6 112 |
| Enunciados | 153 516 | 1 128 246 |
| Nacionalidades | 36 | 145 |

Ambos se construyeron **automáticamente** desde videos de entrevistas de YouTube, usando un pipeline de visión por computador: detección y seguimiento de caras, verificación audiovisual de que la persona en pantalla es la que habla, y verificación de identidad facial. Ningún humano escucha nada, que es lo que permitió la escala.

El protocolo canónico es **entrenar en VoxCeleb2-dev (5 994 hablantes) y evaluar en VoxCeleb1**, que son conjuntos **disjuntos en identidades**. Esa disyunción es lo que hace la evaluación significativa: el modelo nunca vio a las personas del test, así que no puede haber memorizado sus voces — solo puede haber aprendido a representar voces en general.

Hay tres listas de evaluación de dificultad creciente: la original (40 hablantes), **VoxCeleb1-E** (todo VoxCeleb1) y **VoxCeleb1-H**, donde los pares comparten **género y nacionalidad**. La última elimina los dos atajos más obvios, y es donde las diferencias entre métodos se amplían.

---

## 7. Aplicaciones y advertencias

Los usos habituales son control de acceso biométrico, personalización de asistentes de voz, indexación de archivos audiovisuales y **diarización** —saber quién habló en una reunión o una consulta.

En el ámbito clínico el caso más directo es la **diarización de la consulta**: separar la voz del profesional de la del paciente permite estructurar la transcripción, y es un prerrequisito para que el dictado automático produzca una ficha utilizable en lugar de un bloque de texto sin atribuir.

Tres advertencias que conviene enunciar:

- **La voz es dato biométrico.** Un descriptor de hablante identifica a una persona de forma persistente y, a diferencia de una contraseña, no se puede cambiar si se filtra. Los marcos de protección de datos suelen tratarlo como categoría especial.
- **Es suplantable.** La síntesis de voz actual clona un timbre con pocos segundos de referencia. Un sistema de verificación por voz sin **detección de ataques de presentación** (*anti-spoofing*) no es un control de acceso serio.
- **El sesgo demográfico está documentado.** El rendimiento varía según acento, edad y género, y depende de qué poblaciones estén representadas en el entrenamiento. Es la razón por la que la cobertura de 145 nacionalidades de VoxCeleb2 es un aporte y no un dato accesorio.

---

## Lo que se mide al desplegar un sistema real

Cuatro observaciones del [Lab 41](/laboratorios/lab-41), que verifica un sistema entrenado sobre los 37.720 pares de VoxCeleb1 y llega a **3,19 % de EER**:

**1. El score de dos voces no relacionadas no está en cero.** La ReLU previa a la L2-normalización confina los embeddings al ortante positivo, y además el entrenamiento les deja una **dirección común**: la norma de la media global es 0,8088 sobre vectores unitarios, o sea **65,4 % de energía compartida** por todas las grabaciones. Solo el 34,6 % restante distingue hablantes. Consecuencia práctica: los pares de hablantes distintos promedian **0,647** y los del mismo hablante **0,876**, así que el umbral de decisión cae en **0,776** — no cerca de 0. Cualquier intuición sobre «qué es un score alto» tiene que calibrarse con datos, no con la geometría del coseno.

**2. Quitar esa dirección común no ayuda automáticamente.** El centrado (restar la media y renormalizar) es el primer paso del andamiaje estándar de [x-vectors](/papers/x-vectors-snyder-2018), y aquí **no mejora nada**: 3,192 % → 3,266 %, un cambio de 0,58 σ del error de estimación. La razón es que este modelo se entrena con softmax sobre embeddings **ya L2-normalizados**, así que el coseno es la métrica en la que se optimizó; la dirección común es parte de su sistema de coordenadas, no un artefacto a remover. El centrado importa cuando el embedding sale de una capa afín sin normalizar.

**3. Mejorar la separabilidad no es mejorar el error.** El mismo centrado **sube d′ de 3,910 a 4,021 y empeora el EER**. No es contradicción: d′ solo usa medias y varianzas, mientras el EER depende de dónde se cruzan las colas — y la relación $\text{EER} = \Phi(-d'/2)$ solo vale para gaussianas de **igual** varianza, supuesto que aquí no se cumple (0,050 contra 0,066, y tras centrar 0,150 contra 0,176). Con d′ = 4,021 el EER gaussiano sería 2,2 %; el real es 3,27 %.

**4. El solape no es una franja estrecha.** Con el umbral óptimo, el peor par legítimo puntúa 0,670 y el peor impostor 0,860: la **zona de ambigüedad contiene el 33,7 % de todos los pares**. El sistema acierta el 96,8 % porque dentro de esa zona la densidad se inclina hacia el lado correcto, no porque las distribuciones estén separadas. Y el p99 de los impostores (0,802) supera al p5 de los legítimos (0,800).

---

## Referencias

- Fundamentos relacionados: [Agregación VLAD](/fundamentos/agregacion-vlad) · [Reconocimiento de voz](/fundamentos/reconocimiento-de-voz) · [Metric learning](/fundamentos/metric-learning) · [Triplet loss](/fundamentos/triplet-loss) · [Representación de audio](/fundamentos/representacion-de-audio) · [Datasets de audio](/fundamentos/datasets-de-audio).
- Papers: [Utterance-level Aggregation (2019)](/papers/utterance-level-xie-2019) · [x-vectors (2018)](/papers/x-vectors-snyder-2018) · [VoxCeleb (2017)](/papers/voxceleb-nagrani-2017) · [VoxCeleb2 (2018)](/papers/voxceleb2-chung-2018) · [NetVLAD (2016)](/papers/netvlad-arandjelovic-2016) · [GhostVLAD (2018)](/papers/ghostvlad-zhong-2018) · [FaceNet (2015)](/papers/facenet-schroff-2015).
- Clases: [Clase 41](/clases/clase-41).
- Laboratorios: [Lab 41](/laboratorios/lab-41), donde el sistema completo se implementa, se mide y se abre.
- Dominio: [Audio](/dominios/audio).
