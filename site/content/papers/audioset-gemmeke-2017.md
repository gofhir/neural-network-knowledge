---
title: "AudioSet: ontología y dataset de eventos de audio (2017)"
weight: 415
math: true
---

{{< paper-card
    title="Audio Set: An Ontology and Human-Labeled Dataset for Audio Events"
    authors="Jort F. Gemmeke et al. (Google)"
    year="2017"
    venue="ICASSP 2017"
    pdf="/papers/audioset-gemmeke-2017.pdf" >}}
AudioSet es el intento de construir **"el ImageNet del audio"**: un recurso doble que combina (1) una **ontología jerárquica de 632 categorías de eventos sonoros** y (2) un **dataset a gran escala** de segmentos de 10 segundos extraídos de YouTube, etiquetados por humanos y **multi-etiqueta** (promedio **2.7 etiquetas por segmento**). El corpus liberado contiene **1 789 621 segmentos (4971 horas)**, con al menos 100 instancias para 485 de las 632 categorías. Su rasgo más discutido —y el que la [Clase 37](/clases/clase-37) usa como caso de estudio— no es la escala sino el **modo de distribución**: por restricciones de copyright, AudioSet no reparte audio sino **identificadores de YouTube + timestamps + etiquetas**, un diseño que compra la mayor escala del campo al costo del *link rot*. Es el contrapunto de [FSD50K](/papers/fsd50k-fonseca-2020), que reutiliza esta misma ontología pero entrega el audio real.
{{< /paper-card >}}

---

## Contexto: existía ImageNet para imágenes, nada equivalente para audio

El argumento estructural del paper es un paralelo con visión por computador: los "resultados asombrosos" en reconocimiento de imágenes (AlexNet, GoogLeNet, ResNet) descansan sobre **ImageNet** —más de un millón de imágenes con 1000 categorías—, y "nada de esta escala existe para fuentes de sonido". La tesis operativa es que **fue la escala del dataset, no solo la arquitectura, lo que desbloqueó visión**, y que replicar esa escala para audio podría producir un salto análogo.

Los antecedentes que el paper repasa eran todos **de dominio limitado o escala pequeña**: estudios perceptuales sobre 41 o 50 sonidos, taxonomías de ingeniería, datasets como AudioSentiBank (etiquetas a nivel de clip completo, sin garantía de correspondencia) y evaluaciones como **DCASE 2013** (7 sistemas, 16 eventos de oficina, F-measure por debajo de 0.2). A diferencia de todo eso, AudioSet apuesta a considerar **todos los eventos sonoros** en lugar de un dominio acotado.

## Composición: una ontología jerárquica y un corpus verificado por humanos

La contribución es doble e inseparable. **La ontología** es una jerarquía de **632 categorías** con profundidad máxima de **6 niveles**, sembrada a escala web con **patrones de Hearst** (identificando hipónimos de "sonido") sobre una lista inicial de más de 3000 términos, resueltos contra los *machine IDs* (MIDs) de Knowledge Graph y ensamblados manualmente. Sus principios de diseño son marcadamente perceptuales: las categorías deben corresponder a la idea que le viene de inmediato a la mente a un oyente, ser distinguibles por un oyente "típico" (si no, se fusionan) e idealmente distintas **por su sonido solo**. Los 50 nodos de nivel 1 y 2 organizan siete grandes familias: *Human sounds*, *Animal sounds*, *Natural sounds*, *Music*, *Sounds of things*, *Source-ambiguous sounds* y *Channel, environment and background*. Un nodo de nivel 6 es, por ejemplo:

$$\text{Sounds of things} \rightarrow \text{Vehicle} \rightarrow \text{Motor vehicle} \rightarrow \text{Emergency vehicle} \rightarrow \text{Siren} \rightarrow \text{Ambulance (siren)}$$

**El dataset** se construye en dos etapas: **nominación** de candidatos (mitad por un sistema interno de anotación automática restringido a videos con ≥1000 vistas, mitad por búsqueda en metadatos) y **verificación humana**, donde cada segmento de 10 s se presenta con video y audio a tres anotadores que votan "present / not present / unsure" por mayoría. Los anotadores fueron **unánimes en el 76.2%** de las votaciones; los votos 2:1 explican el 23.6%. La distribución es fuertemente desigual: **Music (1 006 882)** y **Speech (893 911)** dominan, mientras la cola larga apenas alcanza ~100 ejemplos. Un **baseline** sobre embeddings de un clasificador profundo (linaje directo de VGGish) obtuvo un **mean Average Precision balanceado de 0.314** y **AUC promedio de 0.959** sobre 485 categorías: la tarea quedaba lejos de resuelta, que era justamente el punto.

## El problema de distribución: links de YouTube, copyright y link rot

Aquí está el rasgo que la Clase 37 destaca. El *Audio Set YouTube Corpus* se distribuye como un **CSV** con identificadores de YouTube, tiempo de inicio, tiempo de fin y etiquetas —no archivos de audio—, porque Google no puede redistribuir contenido de terceros. Junto al corpus se liberaron **features precomputadas** (los embeddings del clasificador base, linaje de VGGish), lo que permite entrenar sin descargar cada video.

Esa arquitectura tiene una consecuencia estructural: **link rot**. Un puntero deja de resolver cuando el video se elimina, se hace privado, se bloquea por región o cae por reclamo de copyright. Con el tiempo, una fracción creciente de los ~1.79 millones de segmentos se vuelve irrecuperable, y no hay dos investigadores que descarguen necesariamente el *mismo* subconjunto. El paper no cuantifica el *link rot* (era un recurso recién liberado), pero la decisión de diseño que lo causa está explícita: audio no redistribuible, solo identificadores. Es el caso de estudio perfecto de la tensión entre **escala** (solo YouTube ofrece millones de horas etiquetables) y **reproducibilidad** (solo el audio en mano garantiza que el dataset no se erosione).

## Impacto

AudioSet se convirtió en el **benchmark de referencia** para clasificación de eventos de audio a gran escala y *audio tagging* multi-etiqueta, cumpliendo su ambición de ser "el ImageNet del audio". Su influencia se materializó en tres direcciones: las embeddings **VGGish** de 128 dimensiones se volvieron una representación estándar *off-the-shelf* (análoga a features de una CNN preentrenada en ImageNet); una generación de modelos entrenados directamente sobre la escala del corpus (PANNs, transformers de audio) reportan su *mAP* sobre la partición de evaluación, la métrica que este paper inauguró con su 0.314; y **la ontología de 632 categorías** se adoptó como vocabulario compartido para etiquetar sonido, más allá del dataset específico.

## Limitaciones

- **Link rot y reproducibilidad.** Distribuir punteros a YouTube en vez de audio erosiona el dataset con el tiempo y hace que distintos equipos entrenen sobre subconjuntos distintos: es la limitación práctica más citada del recurso.
- **Ruido de etiquetas.** El 23.6% de las decisiones se resolvió por mayoría 2:1 y ciertas categorías tienen acuerdo muy bajo (< 0.17); los autores reconocen errores residuales que, dada la escala, quedan como residuo aceptado del diseño.
- **Sesgo de YouTube y de la nominación.** El corpus hereda los sesgos de la plataforma (contenido popular, ≥1000 vistas) y del sistema de nominación, reflejados en el desbalance extremo: "Music" y "Speech" dominan y la cola larga apenas llega a 100 ejemplos.
- **Cobertura incompleta.** Solo **485 de 632** categorías alcanzan 100 instancias; el resto queda excluido o es difícil de poblar.
- **Verificación con video.** El etiquetado se hizo con video visible (solo-audio resultaba demasiado difícil), de modo que las etiquetas verifican que el sonido está presente en un clip donde el anotador *también vio* la fuente.

## Por qué importa para la Clase 37

AudioSet encarna dos de los ejes que organizan la [Clase 37](/clases/clase-37) sobre [datasets de audio](/fundamentos/datasets-de-audio):

- **El eje de la escala web.** AudioSet representa la vía de escala máxima: aprovechar YouTube para reunir ~1.79 millones de clips etiquetados, algo imposible con grabación curada. Esa escala fue lo que hizo viable entrenar modelos de audio "estilo ImageNet".
- **El eje de la disponibilidad.** La clase advierte que a veces un dataset "solo da un link de YouTube que se cae". El contraste con [FSD50K](/papers/fsd50k-fonseca-2020) es didáctico: mientras AudioSet distribuye punteros frágiles a video de terceros, FSD50K —que **reutiliza esta misma ontología**, validando su contribución taxonómica— construye sobre audio de licencia abierta y entrega las formas de onda directamente. La lección de diseño es que **escala y reproducibilidad están en tensión**: la elección de AudioSet compró la mayor escala del campo al costo de la fragilidad.

Para un dominio clínico esto tiene doble filo: la ontología ofrece un marco reutilizable —su rama *Human sounds* ya incluye nodos médicos como "Respiratory sounds" y "Heart sounds, heartbeat" (963 ejemplos)—, pero un corpus distribuido como punteros frágiles se erosiona justo donde la evidencia debe ser auditable, lo que aconseja priorizar la entrega del audio de facto (el modelo FSD50K).
