---
title: "Ejercicios Practicos"
weight: 50
math: true
---

> Enunciados literales de las actividades de los notebooks Parte 1 y Parte 2. Las respuestas razonadas estan en [resolucion](resolucion).

El lab tiene **4 actividades** con un total de **11 preguntas**:

- **Actividad 1** (Parte 1, celdas 42-50) — 3 preguntas conceptuales sobre BERT/bertviz
- **Actividad 2** (Parte 1, celdas 51-62) — 4 preguntas sobre el decoder de un Transformer
- **Actividad 3** (Parte 2, celdas 36-45) — 2 preguntas + 2 templates a testear sobre Food101
- **Actividad 4** (Parte 2, celdas 63-71) — 1 ejercicio practico libre + analisis escrito

---

## Actividad 1 — Comparacion entre versiones de BERT

**Enunciado** *(parte 1, celdas 42-43)*:

> Descomente el codigo a continuacion y elija una version de BERT para probar la misma visualizacion que la obtenida anteriormente. Para esto debe definir correctamente la variable `nv_model_version`.
>
> Basta con la ejecucion para la actividad. Debe ser una version distinta a la ya utilizada, `'bert-base-uncased'`.
>
> **Importante:** Por limitaciones del sistema, no elijan modelos muy pesados para la visualizacion pues probablemente se caeran, principalmente eviten los modelos con `large` en el nombre.

Tras la ejecucion, responder **brevemente** las siguientes preguntas:

### Pregunta 1.1

**Enunciado** *(celda 45)*:

> En las visualizaciones de las atenciones que vimos mas arriba. ¿Por que siempre se muestra la misma oracion, tanto la que atiende con la que es atendida?

### Pregunta 1.2

**Enunciado** *(celda 47)*:

> En la visualizacion Neuron View ¿a que corresponden los parametros "Layer" y "Head"?
>
> Parametro Layer:

### Pregunta 1.3

**Enunciado** *(celda 49)*:

> Parametro Head:

---

## Actividad 2 — Atenciones en el Decoder

**Enunciado** *(parte 1, celda 51)*:

> En esta actividad inspeccionaremos la atencion cruzada que realiza el decoder sobre el encoder. Esta atencion nos dara una mejor idea de que esta usando el modelo para generar texto en el decoder.

La configuracion de la red que produjo las atenciones del diagrama *(celda 54)*:

```yaml
enc_layers: 6
dec_layers: 6
heads: 8
```

### Pregunta 2.1

**Enunciado** *(celda 54)*:

> Al generar el grafico anterior, solo lo estamos visualizando las cross attentions para una capa y una "head" del modelo. **¿Cuantos graficos tendremos si es que pudiesemos visualizar todas las cross-attentions?**

### Pregunta 2.2

**Enunciado** *(celda 56)*:

> El decoder del Transformer no solo utiliza este tipo de atenciones, tambien utiliza las del tipo self-attention. Si estuvieramos haciendo decoding de una traduccion en el paso T=5, es decir generando el 5to token de salida (...).
>
> Si visualizaramos las atenciones de self-attention del decoder **¿que dimensiones tendria la matriz mostrada?**

### Pregunta 2.3 (V/F)

**Enunciado** *(celdas 58-59)*:

> Responda si la siguiente afirmacion es verdadera o falsa.
>
> En el decoder de un transformer, como lo vimos en clase, **no es necesario enmascarar "el futuro"**.

### Pregunta 2.4 (V/F)

**Enunciado** *(celdas 61-62)*:

> En un Transformer, el positional encoding se utiliza como **unica** fuente de informacion del orden de la secuencia.

---

## Actividad 3 — Dimensiones de CLIP + prompt engineering

**Enunciado** *(parte 2, celda 36)*:

> Responda las siguientes preguntas tras inspeccionar la celda donde obtuvimos las dimensiones de los features de imagen, texto y la matriz de similaridad.

### Pregunta 3.1a

**Enunciado** *(celda 38)*:

> En la celda en donde obtuvimos las dimensiones de distintos resultados generados por CLIP, vimos que tanto `image_features` como `text_features` terminan en **512**. ¿A que corresponde el valor de la ultima dimension de los features de la imagen y del texto?

### Pregunta 3.1b

**Enunciado** *(celda 40)*:

> ¿Por que la matriz de similaridad es de `1x101`?

### Pregunta 3.2 — Templates alternativos

**Enunciado** *(celdas 41-45)*:

> Sugiera 2 templates para queries distintos al utilizado previamente y testee que resultados obtiene.

El alumno debe:

1. Definir `Q1` y `Q2` (templates con un `{}` que se reemplazara por el label de la clase)
2. Ejecutar `evaluate_model` con cada uno
3. Reportar Top-1 y Top-5 accuracy de ambos

---

## Actividad 4 — Tus propias imagenes con CLIP

**Enunciado** *(parte 2, celda 63)*:

> Prueba con tus propias imagenes. Utiliza el codigo a continuacion para subir 5 imagenes distintas y generar 5 queries para estas, las queries deben ser distintas y debe haber una asociada a cada imagen. No necesariamente deben todas seguir el mismo *template*.
>
> Escribe abajo un pequeno analisis del resultados obtenido.

El alumno debe:

1. Subir 5 imagenes (via `files.upload()` o descarga desde URL)
2. Definir 5 queries (una por imagen)
3. Calcular la matriz de similitud `(5, 5)`
4. Visualizarla con `plt.imshow`
5. Escribir un analisis del resultado en el campo `A`

---

> **Nota:** las respuestas razonadas con justificacion completa estan en [resolucion](resolucion). Los analisis detallados con screenshots reales del Colab estan en las paginas tematicas:
>
> - Actividad 1 → [neuron-view-y-modelos](neuron-view-y-modelos)
> - Actividad 2 → [decoder-cross-attention](decoder-cross-attention)
> - Actividad 3 → [food101-evaluacion-y-templates](food101-evaluacion-y-templates)
> - Actividad 4 → [stanford-cars-limites](stanford-cars-limites)
