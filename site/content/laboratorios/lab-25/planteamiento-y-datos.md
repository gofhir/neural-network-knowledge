---
title: "Planteamiento del problema y datos"
weight: 1
math: true
---

> **Celdas 0-23 del notebook del Laboratorio 25.** Qué significa recomendar contenido en una red social tipo Pinterest, el **truco del proxy task** que convierte un problema sin etiquetas en uno de clasificación supervisada, por qué el dataset entrega descriptores pre-computados en vez de imágenes, y la Actividad 1 sobre qué datos recolectar para un recomendador.

## El problema: recomendar pares imagen+comentario

El laboratorio aborda un problema concreto de [sistemas de recomendación](/fundamentos/recommender-systems): en una red social tipo **Pinterest**, cada usuario interactúa con **items** que son pares **imagen + comentario** (un *pin*). El objetivo es **recomendar a cada usuario nuevos items** que probablemente le gusten, a partir del contenido con el que ya ha interactuado.

La estrategia elegida es **recomendación basada en contenido (content-based) y multimodal**:

- **Content-based:** se recomiendan items cuyo **contenido** (imagen + texto) es similar al de los items que el usuario consumió. El sistema modela *de qué tratan* los items, no solo *quién interactuó con qué*.
- **Multimodal:** cada item combina dos modalidades —una imagen y un comentario de texto— y el modelo debe fundir ambas en una sola representación.

Esto contrasta con el **filtrado colaborativo** (*collaborative filtering*), que ignora el contenido y se basa solo en la matriz usuario–item de interacciones ("a quienes les gustó X también les gustó Y"). La gran ventaja del enfoque content-based es que **evita el cold-start de items**: un item nuevo, sin ninguna interacción todavía, puede recomendarse de inmediato porque su contenido se codifica igual que el de cualquier item existente. El filtrado colaborativo, en cambio, no sabe qué hacer con un item que nadie ha tocado.

El marco teórico completo —retrieval vs. ranking, content-based vs. collaborative, el problema del cold-start— está en la [clase-25](/clases/clase-25).

## El proxy task: clasificar usuarios para aprender a recomendar

Esta es **la idea central del laboratorio** y conviene detenerse en ella.

El problema real —"¿le gustará este item a este usuario?"— **no tiene etiquetas directas y supervisadas**. No existe un dataset que diga "el usuario A debería ver el item Z pero no el item W". Solo se observa el historial de interacciones pasadas.

El truco consiste en reformular el problema como una **tarea pretexto (*proxy task*)** que **sí** tiene etiquetas: dado un par imagen-texto, **clasificar a qué usuario pertenece**. Esa etiqueta existe gratis —es simplemente el usuario que interactuó con el pin.

¿Por qué funciona? Al forzar al modelo a predecir el usuario dueño de cada item, este aprende un espacio de representaciones donde **el contenido consumido por un mismo usuario queda agrupado** (cerca entre sí) y separado del de otros usuarios. El modelo internaliza así una noción de "qué tipo de contenido le gusta a cada usuario".

El paso clave llega **después** del entrenamiento:

1. Se entrena la red completa (extractor de features + clasificador de usuarios).
2. Se **descarta el clasificador** (la última capa que predice el usuario).
3. Se usan los **features intermedios** —los **descriptores** que produce la red justo antes de clasificar— como representación del item.
4. Se recomienda **por similitud** en ese espacio de descriptores: dado lo que el usuario consumió, se buscan los items más cercanos.

Es exactamente la receta del **metric learning vía tarea pretexto**: se entrena con un objetivo proxy supervisado, pero el producto útil es el **espacio de embeddings**, no el clasificador. La misma intuición que está detrás de la [triplet loss](/fundamentos/triplet-loss) (acercar lo similar, alejar lo distinto en un espacio aprendido) y del esquema de [recuperación de dos torres](/fundamentos/two-tower-retrieval) (codificar usuario e item en un espacio común y recomendar por cercanía).

## El dataset Pinterest: descriptores, no píxeles

Por restricciones de **copyright**, el dataset Pinterest **no entrega las imágenes originales**. En su lugar entrega **descriptores pre-computados**: cada imagen ya fue pasada por una **CNN pre-entrenada en ImageNet**, y lo que se distribuye es el **vector de features** que esa red extrajo (4096 dimensiones por imagen).

Esto es **transfer learning como extractor de features** en estado puro: la CNN aprendió a "ver" en ImageNet (1.2M imágenes, 1000 clases) y sus representaciones intermedias capturan textura, forma y semántica visual genérica que sirve para tareas muy distintas a la clasificación original. Ver el [fundamento de transfer learning](/fundamentos/transfer-learning).

Este planteamiento conecta directamente con [VBPR](/papers/vbpr-he-2016) (He & McAuley, 2016), que fue pionero en usar **features visuales de una CNN** para recomendación de productos, precisamente porque las imágenes capturan señal que la matriz de interacciones no tiene. El laboratorio reproduce esa misma idea: la imagen aporta una señal de contenido que el sistema content-based necesita.

> **"Garbage in, garbage out."** Como los descriptores ya vienen fijos y pre-computados, la calidad del recomendador queda **acotada por la calidad de esos features**. Si la CNN no capturó bien lo relevante de la imagen, ningún modelo posterior podrá recuperarlo. La lección transversal: en recomendación —como en todo ML— los datos y sus representaciones mandan más que la arquitectura.

## Archivos del dataset y gotchas de descarga (celdas de setup)

El dataset se descarga desde **Google Drive** con `gdown` y se descomprime con `unzip`. Los archivos relevantes:

| Archivo | Contenido | ¿Se usa? |
|---|---|---|
| `imag_train.txt` / `imag_val.txt` / `imag_test.txt` | **Descriptores de imagen** (vectores de 4096-d de la CNN), una fila por item | Sí — entrada visual |
| `text_train.txt` / `text_val.txt` / `text_test.txt` | **Comentarios** (texto) asociados a cada item | Sí — entrada textual |
| `train_users.txt` / `val_test_users.txt` | **Etiquetas de usuario** = el *target* del proxy task | Sí — labels |
| `images.rar` | Las **imágenes reales** | **No** — solo referencia; el lab trabaja con los descriptores |

Gotchas verificados al correr el setup:

- **`gdown --id` está DEPRECADO.** En versiones nuevas de `gdown` la bandera `--id` ya no existe; hay que invocar `gdown <ID>` directamente (el ID del archivo de Drive como argumento posicional, sin `--id`).
- **`mv imag_train.txt imag_train2.txt` puede fallar con `cannot stat`.** Si la celda ya se ejecutó antes y el archivo ya fue renombrado, el `mv` no encuentra el origen y arroja un error. Es **inofensivo**: significa que el renombrado ya estaba hecho.
- **Descompresión no interactiva con `!yes | unzip`.** `unzip` pregunta `[y]es/[n]o` cuando un archivo ya existe; en Colab no hay forma de teclear la respuesta. Encauzar `yes` por el pipe responde `y` automáticamente a cada prompt.

## Actividad 1: ¿qué datos recolectar para un recomendador?

La Actividad 1 plantea tres preguntas conceptuales de diseño antes de tocar código. Respuestas razonadas:

**(a) ¿Qué datos recolectarías para construir el recomendador?**

- **Datos de interacción** (la señal más valiosa): *likes*, *saves*/pins, *shares*, comentarios, clicks, y señales implícitas como **dwell time** (tiempo de permanencia) y **scroll depth**. También **impresiones negativas** —items mostrados que el usuario ignoró—, que son cruciales para muestrear negativos realistas.
- **Contenido de los items:** la **imagen** y el **texto** (comentario, título, tags) de cada pin. Es lo que habilita el enfoque content-based.
- **Features de usuario:** demografía, idioma, historial agregado de categorías consumidas.
- **Contexto:** dispositivo, hora del día, sesión actual, ubicación.

**(b) ¿Cómo usarías esos datos?**

- Las **interacciones funcionan como labels** (positivos = lo que el usuario consumió; negativos = impresiones ignoradas o muestreo de items no vistos).
- El **contenido se codifica en embeddings** (imagen y texto) con un enfoque **content-based**, lo que **evita el cold-start de items**: un pin nuevo se recomienda apenas se sube, sin esperar interacciones.
- Arquitectónicamente, un esquema en dos etapas **retrieval → ranking**: primero recuperar cientos de candidatos baratos por similitud de embeddings, luego rankearlos con un modelo más fino. Es la receta del [modelo de YouTube](/papers/youtube-dnn-covington-2016) (Covington et al., 2016).

**(c) ¿Sería útil información externa al sistema?**

Sí. Información externa que aporta valor:

- **Embeddings pre-entrenados** —imagen (ImageNet) y texto (BERT)—, que es **exactamente lo que hace este laboratorio**: importa conocimiento aprendido en corpus enormes ajenos a Pinterest.
- **Knowledge graphs** y catálogos externos (relaciones entre productos, categorías, marcas).
- **Tendencias externas** (qué está de moda, estacionalidad, eventos) para combatir el sesgo de popularidad y refrescar recomendaciones.

Combinar señales diversas —contenido propio + features cruzadas + conocimiento externo— es precisamente la motivación de modelos como [Wide & Deep](/papers/wide-and-deep-cheng-2016) (Cheng et al., 2016), que mezclan memorización de features explícitas con generalización vía embeddings.

---

**Siguiente:** [Dataset y modelo multimodal](dataset-y-modelo)
