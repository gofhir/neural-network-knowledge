---
title: "Actividades 1-4 resueltas"
weight: 4
---

El práctico pide elegir **tres** actividades (se corrigen las dos con mejor puntaje). Acá están las cuatro resueltas; las dos conceptuales más fuertes y la de código (1, 2, 4) son la mejor apuesta, con la 3 de respaldo.

## Actividad 1 — Mejores embeddings para cada entrada de la KB

**Consigna:** los tokens ya están separados; cada entrada es una lista de índices. ¿Cómo generar mejores representaciones?

Hoy cada entrada se representa como el **promedio (Bag-of-Words)** de los embeddings de sus tokens, lo que pierde el orden de las palabras y pondera igual a stopwords y entidades. Propuestas, de menor a mayor capacidad:

1. **Ponderar tokens** en vez de promediar uniforme (TF-IDF o atención / self-attention pooling), para que la entidad y la relación dominen sobre las stopwords.
2. **Incorporar el orden** con positional encoding (como en el paper de Memory Networks) o codificando cada entrada con un BiLSTM/GRU — así se distinguen relaciones direccionales (X dirige Y vs. Y dirige X).
3. **Encoder contextual** (Transformer / BERT-BETO) para capturar composición e interacciones entre tokens.
4. **Embeddings preentrenados** (word2vec/GloVe), útil porque la matriz es enorme (186.841 tokens) y muchos aparecen poco.
5. **Calibración / normalización:** usar similitud coseno o una temperatura aprendida en vez del producto punto crudo, que satura el softmax (sobreconfianza observada en [Experimentos](../03-experimentos-y-analisis)).

**Evidencia propia:** la accuracy top-1 (0.58) sube a top-3 (0.74), lo que sugiere que mejores representaciones por entrada (y mejor calibración) recuperarían buena parte de ese margen.

## Actividad 2 — Generalización a entradas nuevas post-entrenamiento

La clave es la separación **paramétrico vs. no-paramétrico**: la KB es externa (no-paramétrica), pero los embeddings de los tokens son paramétricos (se aprenden).

**Caso (a) — entidades NO vistas en entrenamiento → NO (confiablemente).**
Depende de la tokenización. Si cada entidad es un token atómico, una entidad nueva no tiene embedding entrenado: cae en `UNK/<NULL>` o sería aleatorio → el matching por atención y el value devuelto son ruido. Si la entidad se separa en palabras (el preprocesamiento de los `.pkl`), sus subtokens pueden existir y la entrada tendría representación (promedio de esos embeddings), pero de **calidad no garantizada** porque el modelo nunca aprendió esa combinación. En el caso general: **NO**. Es el límite paramétrico — el conocimiento de cómo representar un token vive en los pesos y no se actualiza solo agregando datos a la KB.

**Caso (b) — info nueva con entidades YA vistas → SÍ.**
La KB es **externa y no-paramétrica**: agregar una entrada es solo añadir un par `(key, value)` a la memoria, sin tocar los pesos. Como la nueva información usa solo tokens ya vistos, todos tienen embeddings entrenados y el key addressing + value reading operan igual. Lo único necesario a nivel de implementación es **reindexar** la entrada en el blocker para que sea recuperada como candidato — eso es ingeniería de datos, no reentrenamiento. **Esta es la ventaja central de la memoria externa** frente a un modelo puramente paramétrico (un RNN que guarda el conocimiento en sus pesos): se actualiza el conocimiento sin reentrenar.

## Actividad 3 — KB key-value desde tuplas (sujeto, relación, objeto)

**Diseño:** key = **(sujeto + relación)**, value = **objeto**. Para `(empanada, comida_típica_de, Chile)`: `key = (empanada, comida_típica_de)`, `value = Chile`. Es el esquema del paper KV-MemNN para KBs de tripletas.

**Pregunta 1 — *"¿De qué país es la empanada una comida típica?"* → SÍ.**
La pregunta menciona el sujeto (`empanada`) y la relación, que es lo que compone la key. El key addressing matchea y devuelve el value `Chile`. Correcto.

**Pregunta 2 — *"¿Cuál es una comida típica de Chile?"* → NO (con la tupla original).** Por dos razones:
1. La respuesta esperada (`empanada`) es el **sujeto**, que quedó dentro de la key — y el modelo solo devuelve **values**, nunca partes de la key.
2. La pregunta menciona `Chile`, que está en el value y no en la key → el matching tampoco es directo.

**Solución (a nivel de datos, sin reentrenar):** crear la **tupla inversa** `(Chile, comida_típica_de⁻¹, empanada)`, es decir `key = (Chile, relación_inversa)`, `value = empanada`. Así la pregunta sobre Chile matchea la nueva key y devuelve `empanada`. Esto es lo que el paper llama **"doblar la KB"** con relaciones inversas para permitir recuperación en ambas direcciones.

## Actividad 4 — Preprocesar un ejemplo inventado (código)

**Consigna:** preprocesar la pregunta `who directed doomsday?` (que no está en ningún split) para que el modelo la visualice y conteste bien. La respuesta correcta es darse cuenta de que **no hay que escribir preprocesamiento nuevo** — solo reaplicar las tres funciones que el notebook ya usó para train/dev/test:

```python
# 1) Candidatos de la KB para la pregunta (filtra por token en común, arma keys/values/candidatos)
new_q, new_key, new_value, new_cand, new_a = gen_candidates(new_q_orig, new_a_orig)
# 2) Padding de las keys a largo fijo (MAX_KEY_WORDS)
new_key = pad_all_keys(new_key)
# 3) Largos para las máscaras del modelo
new_q_word_lengths, new_key_num_lengths, new_key_word_lengths, new_cand_lengths = get_data_lengths(
    new_q, new_key, new_cand
)
```

**Dos detalles de orden que importan:** `pad_all_keys` debe ir **antes** de `get_data_lengths` (que hace `k.shape[1]` sobre tensores 2D), y el desempaque de `get_data_lengths` debe respetar el orden `[q_word, key_num, key_word, cand]`.

**Resultado verificado:**

```
Q: who directed doomsday ?
Predicted answer: neil marshall
Correct answer:   neil marshall
```

![Heatmap — Actividad 4, who directed doomsday → neil marshall](/laboratorios/lab-30/actividad4-doomsday-hop2.jpg)

El modelo respondió correctamente una pregunta que **nunca estuvo en el dataset**, demostrando en vivo la tesis de la Actividad 2: como la KB es no-paramétrica y la pregunta usa solo tokens ya vistos (`doomsday`, `neil marshall`, `directed`), el pipeline recupera, direcciona y lee sin reentrenar nada.

---

**Volver al** [índice del lab](../) **o seguir a la** [clase 30 (teoría)](/clases/clase-30).
