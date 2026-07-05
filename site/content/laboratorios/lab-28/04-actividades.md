---
title: "Actividades resueltas (5 de 6)"
weight: 4
---

El práctico pide elegir **5 de las 6** actividades (cada una 1.2 pts). Se saltó la Act 5 (entrenar UDA con las 20.000 etiquetas, ~30+ min de cómputo). Las cinco desarrolladas:

## Actividad 1 — ¿Por qué cambian los ejemplos al aplicar back-translation?

La traducción **no es una función biyectiva**: al traducir EN→FR→EN, el modelo elige entre múltiples formas válidas de expresar el mismo significado. Cambian por: (1) **sinónimos y paráfrasis**, (2) **reordenamiento sintáctico** (el orden válido difiere entre idiomas), (3) el **idioma pivote** (francés) no mapea 1:1 con el inglés — hay ambigüedades léxicas y gramaticales, (4) el **muestreo estocástico** (`do_sample` + `temperature`) introduce variación aleatoria. El resultado preserva el significado global (misma etiqueta) pero cambia la superficie léxica — justo lo que UDA necesita.

## Actividad 2 — ¿Funciona la consistency loss para cada aumentación?

Regla que atraviesa las cinco: **funciona si y solo si la aumentación preserva la etiqueta** (el sentimiento). Si la cambia, forzar consistencia enseña algo incorrecto.

| Aumentación | ¿Funciona? | Razón |
|---|---|---|
| 1. Sinónimo | **SÍ** | preserva el significado y el sentimiento (`great`→`excellent`) |
| 2. Antónimo | **NO** | invierte el sentimiento (`great`→`terrible`) → cambia la etiqueta; enseñaría a ignorar las palabras que la determinan |
| 3. Eliminar una palabra | **SÍ** (en general) | en reseñas largas, quitar una palabra rara vez cambia el sentimiento; similar a un dropout de palabras. Riesgo: eliminar una negación (`not good`→`good`) |
| 4. Reemplazar palabras **muy frecuentes** | **SÍ** | frecuencia alta = baja información (Zipf/TF-IDF): son stopwords sin sentimiento; reemplazarlas preserva la etiqueta e incluso enfoca al modelo en lo informativo |
| 5. Reemplazar palabras **poco frecuentes** | **NO** | frecuencia baja = alta información: las palabras raras cargan el sentimiento (`magnificent`, `atrocious`); reemplazarlas destruye la señal → cambia la etiqueta |

El eje conceptual (casos 4 y 5) es la relación **frecuencia ↔ información** de la ley de Zipf, la misma que se ve en el [lab 16](/laboratorios/lab-16) con TF-IDF.

## Actividad 3 — Filtrar el ruido del back-translation

(Se elige la pregunta 2 de las tres opciones.) Filtro de consistencia semántica en dos niveles:

1. **Similitud semántica:** calcular la similitud coseno entre el embedding del original y el de la back-translation (con un sentence-encoder o BERTScore) y descartar los pares de similitud **demasiado baja** (el BT cambió el significado, como el ejemplo `i=0` del notebook) y también los de similitud casi perfecta (no aportan variación). Quedarse con un rango intermedio.
2. **Round-trip label check:** pasar original y BT por un clasificador de sentimiento (puede ser el mismo modelo) y descartar los pares cuya predicción difiere — son los casos donde la aumentación cambió la etiqueta y rompería la consistency loss.

Complementario: filtrar por la log-probabilidad de la generación del traductor (baja confianza ⟹ traducción más ruidosa).

## Actividad 4 — Back-translation × temperatura (código verificado)

Se aplicó back-translation (EN→FR→EN con MarianMT) a 3 ejemplos propios con 3 temperaturas. Resultados medidos:

| Ejemplo | Temp 0.1 | Temp 1.5 | Temp 3.5 |
|---|---|---|---|
| *This movie was an absolute masterpiece...* | casi idéntico | casi idéntico | limpio |
| *The plot was boring and the acting felt wooden...* | mínimo (`acting`→`act`) | mínimo | **descarrila:** `The site is boring and an actor feels of lifeless wood` |
| *A patient with type 2 diabetes was prescribed metformin to control...* | reformula bien | `control`→**`monitor`** (drift clínico) | limpio |

**¿En qué influye la temperatura?** Controla cuán aleatorio es el muestreo: los logits se dividen por T antes del softmax. Con **T baja (0.1)** la distribución se vuelve puntiaguda → elige la palabra más probable → BT casi idéntico (poca variación). Con **T alta (3.5)** la distribución se aplana → palabras improbables ganan chance → textos diversos pero que se vuelven agramaticales o cambian el significado. Con **T intermedia (1.5)** se logra el balance que UDA busca: paráfrasis que cambian la superficie pero preservan el sentido.

Dos matices observados: (1) el `top_k=10` limita el muestreo a las 10 palabras más probables y actúa como red de seguridad — por eso incluso a T=3.5 las frases cortas quedan intactas; (2) el daño de la temperatura alta es **selectivo**: afecta primero a las frases sintácticamente complejas. En resumen, la temperatura regula el trade-off entre **diversidad** de la aumentación (necesaria para que la consistencia aporte señal) y **fidelidad** al significado (necesaria para no cambiar la etiqueta).

## Actividad 6 — Contrastive learning (SimCLR/MoCo) vs UDA

**Ventaja del preentrenamiento contrastivo sobre UDA:** aprende representaciones **genéricas y reutilizables** sin ninguna etiqueta, que sirven de punto de partida para **múltiples tareas** downstream; se entrena una vez a gran escala y se amortiza el costo. UDA es específico de una tarea (la consistency loss se optimiza junto al clasificador de esa tarea).

**Desventaja del contrastivo frente a UDA:** requiere una fase de preentrenamiento separada y costosa (batches grandes o colas de negativos como en [MoCo](/papers/moco-he-2019), mucho cómputo) y sus representaciones genéricas pueden no estar alineadas con la tarea/dominio objetivo. UDA aprovecha datos no etiquetados **del dominio target** e integra la señal directamente en el fine-tuning, por lo que con muy pocas etiquetas suele adaptarse mejor al dominio específico.

En resumen: **contrastivo = representaciones reutilizables pero caras y genéricas; UDA = adaptación barata y específica a la tarea, pero no reutilizable.**

---

**Volver al** [índice del lab](../) **o seguir a la** [clase 28 (teoría)](/clases/clase-28).
