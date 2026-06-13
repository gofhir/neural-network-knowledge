---
title: "Modos de fallo: una taxonomía"
weight: 3
---

> **Celdas 25-29 del notebook ("Algunos Errores Vistos en Clases").** Cuatro imágenes, cuatro preguntas, cuatro respuestas equivocadas de BLIP — pero equivocadas de maneras *distintas*. La sección no busca corregir el modelo sino **clasificar cómo falla**, porque cada modo de fallo apunta a una limitación arquitectónica diferente.

Las cuatro celdas comparten exactamente el mismo patrón: descargar una imagen desde una URL, mostrarla con `plt.imshow`, formular una pregunta, llamar a `model.generate(...)` e imprimir el par `❓ pregunta` / `🤖 respuesta`. Lo único que cambia es la imagen y la pregunta. Esa uniformidad es deliberada: aísla la variable interesante, que es **en qué se equivoca el modelo y por qué**.

## La taxonomía de un vistazo

| # | Imagen | Pregunta | Respuesta de BLIP | Modo de fallo | Causa raíz |
|---|--------|----------|-------------------|---------------|------------|
| 26 | perro junto a una silla | *Is the dog in front of the chair?* | **"yes"** | Error **espacial** | El ViT colapsa la geometría 3D en parches 2D; se pierde la profundidad relativa |
| 27 | un gato | *What color is the cat?* | **"multi colored"** | Respuesta **vaga** | "Apuesta segura" estadística que evita comprometerse con un color |
| 28 | un objeto/figura | *What is this?* | **"toy"** | Granularidad **gruesa** | Hiperónimo: categoría amplia que maximiza accuracy promedio |
| 29 | un ornitorrinco | *What kind of animal is this?* | **"monkey"** | **Alucinación** OOD | Clase fuera de distribución proyectada a la conocida más cercana |

Los cuatro son "errores", pero no del mismo tipo. Confundirlos lleva a soluciones equivocadas: el error espacial no se arregla con más datos del mismo tipo, la respuesta vaga no es realmente "incorrecta" en el dataset, y la alucinación es un problema estructural del decoder. Veámoslos uno por uno.

## 1. Error espacial: presencia sí, relación no (perro / silla → "yes")

La pregunta *"Is the dog in front of the chair?"* recibe un **"yes"** confiado. El problema es que el "yes" es correcto solo por casualidad — el modelo no resolvió la relación geométrica, simplemente detectó que **hay un perro y hay una silla** en la escena y emitió la respuesta de mayor frecuencia para preguntas de tipo *"is X in front of Y?"* cuando ambos objetos están presentes.

Esta es la **disociación entre detección y relación**. BLIP es excelente reconociendo *presencia* de objetos (¿hay un perro? sí; ¿hay una silla? sí), porque eso es lo que el encoder visual aprende bien. Pero las **relaciones espaciales** — *in front of*, *behind*, *to the left of*, *on top of* — exigen razonar sobre la geometría tridimensional de la escena, y ahí el pipeline se rompe:

- El **Vision Transformer** ([ViT](/fundamentos/vision-transformer)) trocea la imagen en parches 2D y los procesa como una secuencia. La atención modela co-ocurrencia ("perro y silla aparecen juntos") mucho mejor que orden espacial relativo.
- La **profundidad relativa** (qué objeto está más cerca de la cámara) no está codificada explícitamente en ningún lado. Un parche no sabe si está "delante" o "detrás" de otro en el espacio 3D; solo sabe su posición en la grilla 2D.
- Las preguntas espaciales tienen una respuesta mayoritaria en VQAv2 ("yes" abunda), así que adivinar paga.

> **Nota — reaparición en la Actividad.** Esta misma imagen del perro y la silla vuelve a aparecer en la **pregunta 7 de la Actividad**, justamente para que el estudiante constate de primera mano que el modelo no entiende relaciones espaciales. No es un ejemplo aislado: es un patrón que el curso quiere que internalices.

## 2. Respuesta vaga: el "depende" del VQA (gato → "multi colored")

A *"What color is the cat?"* BLIP responde **"multi colored"**. No es exactamente falso — pocos gatos son de un solo color — pero es una **evasión**. Es el equivalente VQA de responder "depende": una respuesta que minimiza el riesgo de equivocarse al no comprometerse con ningún color concreto.

¿Por qué el modelo prefiere lo vago a lo específico? Porque **"multi colored" tiene alta frecuencia en el dataset** y rara vez está rotundamente equivocada. Si el modelo dijera "orange" y el gato fuera atigrado gris y blanco, sería un error claro; "multi colored" casi nunca lo es. La función de pérdida no penaliza la vaguedad — penaliza estar equivocado —, así que el modelo aprende que la ambigüedad es una estrategia ganadora. Es un fallo de **utilidad**, no de corrección: la respuesta es técnicamente defendible pero informativamente inútil.

## 3. Granularidad gruesa: el hiperónimo seguro (objeto → "toy")

A *"What is this?"* el modelo contesta **"toy"** en vez del objeto específico. "Toy" es un **hiperónimo** — una categoría amplia que engloba muchos objetos concretos. En vez de arriesgarse a nombrar la cosa exacta (que podría errar), el modelo retrocede a la categoría general (que difícilmente erra).

La causa está en el incentivo del entrenamiento. **VQA optimiza accuracy promedio sobre VQAv2**, y en ese régimen las **categorías amplias son apuestas seguras**: si no estás seguro de si es un peluche, una figura de acción o un muñeco, "toy" acierta en los tres casos. Nombrar el objeto específico tiene alto retorno pero alto riesgo; el hiperónimo tiene retorno modesto pero casi sin riesgo. Un modelo entrenado para maximizar el acierto promedio aprende a jugar a lo seguro. El resultado: respuestas correctas pero poco informativas, el mismo síndrome del "multi colored" pero en el eje de la **especificidad** en lugar del color.

## 4. Alucinación: la clase fuera de distribución (ornitorrinco → "monkey") ⭐

Este es el modo de fallo más instructivo. Se muestra un **ornitorrinco** y se pregunta *"What kind of animal is this?"*. BLIP responde, con total aplomo, **"monkey"**.

El ornitorrinco es un caso de manual de **out-of-distribution (OOD)**: es un animal raro, casi ausente de los datos de entrenamiento de BLIP. El modelo nunca aprendió un concepto "ornitorrinco" robusto. Cuando se enfrenta a algo que no conoce, **no se detiene** — proyecta la entrada a la **clase conocida más cercana** según features superficiales: pelaje, cuatro patas, un hocico prominente. Por esa ruta visual, "monkey" gana. El modelo rellena el vacío con su vecino conceptual más próximo.

> **Dato clave — la alucinación depende de la tarea.** Esta **misma imagen del ornitorrinco**, cuando se le pide un *caption* en lugar de responder una pregunta (ver la página de [Image Captioning con BLIP](image-captioning-blip)), produce **"a baby bird is held in a box"**. Es decir: la misma entrada visual ambigua alucina **"monkey"** bajo VQA y **"pájaro bebé en una caja"** bajo captioning. La alucinación no es una propiedad fija de la imagen — **depende de la tarea y del prompt**, porque cambian el prior lingüístico que el decoder aplica sobre la evidencia visual incierta. Este contraste es exactamente lo que explora la **pregunta 5 de la Actividad**.

### Por qué alucinan los VLMs (las cuatro causas)

La alucinación del ornitorrinco no es un accidente; es la consecuencia previsible de cómo está construido un modelo de visión-lenguaje. Hay cuatro causas que se combinan:

1. **Out-of-distribution (OOD).** El concepto simplemente no está (o está malísimamente representado) en el entrenamiento. El modelo no tiene de dónde sacar la respuesta correcta, así que improvisa con lo que tiene cerca.
2. **No existe el botón "no sé".** El decoder autorregresivo **siempre emite un token** en cada paso — está obligado a producir *algo*. No hay una clase "abstención" ni un umbral de confianza que dispare un "no estoy seguro". Generar una respuesta equivocada y abstenerse son, para la arquitectura, el mismo acto: emitir tokens.
3. **El prior lingüístico domina sobre la evidencia visual ambigua.** Cuando la señal visual es incierta, el modelo de lenguaje se inclina hacia lo que es *lingüísticamente probable*: "monkey" es una respuesta común a "what kind of animal", "ornitorrinco" casi nunca lo es. La fluidez del lenguaje sobreescribe la duda visual.
4. **Exposure bias.** El modelo se entrena con **teacher forcing** (siempre ve el token correcto del paso anterior), pero en inferencia **realimenta sus propios tokens**. Si emite un token dudoso, lo toma como verdad para el siguiente paso y construye encima de su propio error, encadenando una respuesta coherente pero falsa.

> El tratamiento completo de estas cuatro causas — con la matemática del decoder, la conexión con teacher forcing y las estrategias de mitigación — está en el fundamento [Vision-Language Models](/fundamentos/vision-language-models). Aquí basta con retener la idea central: **un VLM no falla por ignorancia silenciosa, falla por confianza ruidosa.** Es la misma lección que el [VQA clásico](/papers/pythia-jiang-2018) ya anticipaba sobre las limitaciones de estos sistemas: alta fluidez no implica alta fidelidad.

## GOTCHA de reproducibilidad: las URLs son thumbnails de gstatic.com

Hay un detalle frágil en estas cuatro celdas que conviene saber antes de ejecutarlas. Las imágenes se descargan desde **URLs de thumbnails de `gstatic.com`** (las miniaturas de búsqueda de imágenes de Google). Estas URLs son **inestables**: caducan, rotan o devuelven **404 / 403** sin aviso, porque no están pensadas como hosting permanente.

En Colab esto es especialmente crítico (ver la página de robustez del lab): si el thumbnail muere, la celda revienta en la descarga y nunca llega al `generate`, dando la falsa impresión de que el modelo falló cuando en realidad falló la red. **Sugerencia:** reemplazar esas URLs por enlaces estables — subir las imágenes a un repositorio propio, usar un dataset versionado, o cachearlas localmente — para que la taxonomía de fallos sea reproducible y no dependa de la salud de un CDN de miniaturas.

---

**Anterior:** [VQA como generación](vqa-generacion) · **Siguiente:** [Image Captioning con BLIP](image-captioning-blip)

Vuelve a la teoría en [Clase 23 — VQA e Image Captioning](/clases/clase-23).
