---
title: "Actividades"
weight: 6
---

> **Celdas 98-101 del notebook.** Las tres preguntas del lab, respondidas con la evidencia empírica recogida a lo largo del recorrido. La Actividad 3 incluye el experimento real de `draw_in_map('food', ..., show_match_image=True)` con sus 15 matches analizados imagen por imagen.

## Pregunta 1 — El modelo solo lee palabras

> *¿Es una limitación de la arquitectura o de los datos? ¿Qué se haría para que prediga frases?*

**Es principalmente una limitación de los datos y del planteamiento de la tarea, no de la arquitectura.** [TotalText](/papers/total-text-chng-2017) está anotado a nivel de palabra: cada instancia de entrenamiento es una palabra con su par de curvas Bézier y su transcripción. El detector aprendió a proponer una instancia por palabra, y el recognizer se entrenó con etiquetas de palabras sueltas (longitud máxima ~25).

La rama de reconocimiento por atención **no tiene impedimento intrínseco** para emitir secuencias largas. La evidencia: en las apps se vio "Uncle Ben's" partido en `uncle`+`bens` y "Penn Ave" en `penn`+`ave` — la segmentación ocurre en la **detección**, no en el reconocimiento. (Matiz arquitectónico menor: la longitud máxima fija de 25 y que dos curvas Bézier describen bien un texto compacto pero no una frase larga y curva.)

**Para predecir frases**, de menor a mayor costo:
1. **Post-procesamiento** sin reentrenar: agrupar las palabras detectadas por proximidad espacial (`pred_boxes`/`beziers`) en orden de lectura.
2. **Fine-tuning** con anotaciones a nivel de línea/frase y mayor longitud de secuencia.
3. **Cambiar de modelo** a uno de reconocimiento a nivel de línea/párrafo.

## Pregunta 2 — Lee alemán sin entrenarse en alemán

> *¿A qué se debe? ¿Podría leer coreano con los mismos pesos?*

**Se debe a que el modelo aprendió formas de glifos del alfabeto latino, no "el idioma inglés".** El recognizer mapea formas visuales → índices de un charset; no modela semántica ni gramática. Inglés y alemán comparten el alfabeto latino, así que `milch`, `reis`, `honig` son combinaciones de glifos que el modelo reconoce visualmente. **Evidencia:** el top 40 de Groceries tuvo decenas de palabras alemanas limpias. **Matiz verificado:** los caracteres específicos del alemán (ä, ö, ü, ß) no están en el charset ASCII y se colapsan a la letra base (`Müsli → musli`, `Äpfel → apfel`).

**No podría leer coreano con los mismos pesos**, por dos razones suficientes:
1. **El charset de salida lo impide.** Solo tiene clases para ASCII (índices 0-94, más 95 = desconocido = 口, 96 = blank). El Hangul (한글) no tiene índice de salida posible → el modelo es físicamente incapaz de emitirlo. El warning CJK del demo (índice 95 = 口) es precisamente ese placeholder de "desconocido".
2. **Los pesos nunca vieron glifos Hangul** — bloques silábicos visualmente nada parecidos a glifos latinos.

> Si la diéresis alemana (caracteres *cercanos* al latino) ya se pierde, el coreano —completamente fuera del charset y del espacio visual aprendido— es imposible. Ver [scene text recognition](/fundamentos/scene-text-recognition).

## Pregunta 3 — `draw_in_map('food', ..., show_match_image=True)`

> *¿Son todas establecimientos de comida? ¿Qué modelo incorporaría para filtrar?*

### a. No, no todas son establecimientos de comida

El experimento devolvió **15 matches**. Revisados imagen por imagen, muestran gran diversidad de contextos para la palabra "food":

| Imagen | Negocio leído | ¿Establecimiento de comida? |
|---|---|---|
| George Aiken's | "Delicious **Prepared Foods**" | ✅ Sí (comida preparada) |
| Food Mart / Rite Aid | "**Food Mart**" junto a farmacia | ⚠️ Tienda de conveniencia |
| Fifth Avenue Place | centro comercial | ⚠️ Mall |
| Shops | "**Food Court**" | ⚠️ Patio de comidas de un mall |
| Almacén | "**DAIRY FOODS**" + camioneta | ❌ Distribuidora de lácteos |

{{< cards >}}
  {{< card image="/laboratorios/lab-21/food-george-aikens.jpg" title="George Aiken's — Prepared Foods" subtitle="Único establecimiento de comida claro" >}}
  {{< card image="/laboratorios/lab-21/food-foodmart-riteaid.jpg" title="Food Mart + Rite Aid" subtitle="Tienda de conveniencia, no restaurante" >}}
  {{< card image="/laboratorios/lab-21/food-foodcourt.jpg" title="Food Court" subtitle="Patio de comidas dentro de un complejo de tiendas" >}}
  {{< card image="/laboratorios/lab-21/food-dairy.jpg" title="DAIRY FOODS" subtitle="Almacén/distribuidora de lácteos — el caso más alejado" >}}
{{< /cards >}}

El OCR detecta "food" sin entender qué objeto la porta. Además, la [búsqueda fuzzy](/fundamentos/fuzzy-string-matching) (threshold 80) hace match con "Food**s**" (ratio ≈89), "Food Mart" y "Food Court", de modo que "food" aparece como subcadena de nombres comerciales muy diversos. La limitación de fondo: el modelo lee texto pero **no clasifica el contexto**.

![Mapa de los matches de food sobre Pittsburgh](/laboratorios/lab-21/food-mapa.jpg)

> Varias imágenes son la misma posición GPS vista en distintos ángulos (6 fotos por punto), por lo que en el mapa caen como puntos cercanos.

### b. Un detector de objetos para filtrar por contexto

**Arquitectura:** un detector de objetos como [Faster R-CNN](/papers/faster-rcnn-ren-2015) (Ren et al., 2015), o un single-stage tipo YOLO/RetinaNet. Para filtrado a nivel de píxel, segmentación semántica/panóptica (Mask R-CNN, DeepLab).

**Datos de entrenamiento:** imágenes de calle con bounding boxes por clase de objeto urbano —vehículos (camión, auto, bus), edificios/fachadas, señales, vallas—. Datasets: **COCO** (ya trae `truck`, `car`, `bus`), Cityscapes, Mapillary Vistas.

**Función:** localizar los vehículos (y portadores móviles) en cada imagen; para cada palabra detectada por ABCNet, si su `pred_box` se solapa (IoU > umbral) con un vehículo, **descartarla**; si cae sobre una fachada, conservarla. Filtra por **contexto del objeto**, no por posición fija — resolviendo lo que `get_mask` no podía: el texto sobre un camión repartidor.

> 🎯 **Conexión con MDM:** la pregunta 3 es el patrón de **pipeline de dos etapas** — una señal barata (el OCR / *blocker*) genera candidatos, y un segundo modelo (el detector de contexto / *scorer*) los filtra. El detector de vehículos cumple el rol del scorer que descarta falsos positivos que la primera etapa no distingue sola.

## Nota de ejecución

El dataset de Street View (Google Drive) estaba bloqueado por **cuota de descarga** ("too many users") al correr el lab. Se recuperó con el workaround estándar: subir el zip al Drive propio (id nuevo, sin cuota), montar y descomprimir en la ruta que el `.pkl` precalculado esperaba (`datasets/GSV/`). Como el `.pkl` ya contenía `words`, `boxes` y `file_name`, no hubo que reprocesar las 6000 imágenes — solo restaurar los archivos físicos para que `show_match_image=True` pudiera abrirlos.

---

**Anterior:** [App 2 · Google Street View](app-streetview) · **Volver al** [índice del lab](../)
