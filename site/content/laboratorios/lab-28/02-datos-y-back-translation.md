---
title: "Datos, back-translation y dataloaders"
weight: 2
---

El combustible de UDA son los datos no etiquetados y una aumentación que preserve la etiqueta. Acá se ve por qué IMDB es el dataset ideal, cómo funciona la back-translation, y cómo se arman los batches que mezclan supervisado y no supervisado.

## IMDB: el dataset semi-supervisado

```python
train_df["label"].value_counts()
# unsup    50000   ← SIN etiqueta
# pos      12500   ← positivas
# neg      12500   ← negativas
```

IMDB no solo tiene 25k reseñas etiquetadas (12.5k pos + 12.5k neg), sino **50k sin etiquetar**. Eso lo hace perfecto para SSL: se entrena con **solo 20 etiquetas** de las 25k, apoyándose en las 50k sin etiqueta como fuente de consistencia.

**Filtro de calidad:**
```python
train_df = train_df[(train_df['content'].str.len() >= 500) | (train_df['label'] != 'unsup')]
```
Se lee como *"conserva si el texto es largo (≥500 chars) O si está etiquetado"*. Es decir, el filtro de longitud **solo aplica a los no etiquetados** (quedan 44.972 de 50.000); las etiquetadas se conservan todas. El porqué: la back-translation de textos cortos es ruidosa e inestable, y los textos cortos dan poca señal de consistencia. Las etiquetas son escasas y valiosas — no se filtran.

## Back-translation: la aumentación de UDA

```
"This movie was fantastic"  ──[EN→FR]──►  "Ce film était fantastique"  ──[FR→EN]──►  "This film was great"
        original                              (idioma pivote)                          aumentado
```

Traducir a un idioma pivote y de vuelta produce una **paráfrasis**: cambia palabras y sintaxis pero preserva el significado (misma etiqueta de sentimiento). Es el análogo textual de rotar/recortar una imagen en SimCLR. Es **lento** (un modelo de traducción neuronal por dirección sobre ~70k textos), por eso el lab lo trae pre-computado.

### El back-translation no siempre preserva la etiqueta

El notebook muestra a propósito un caso malo y uno bueno. Y en la [Actividad 4](../04-actividades) se genera en vivo con distintas temperaturas de muestreo:

- **Temperatura baja (0.1):** casi idéntico al original (poca variación).
- **Temperatura media (1.5):** paráfrasis útil — el régimen que UDA quiere. (Ejemplo observado: `control blood glucose` → `monitor blood glucose` — un drift semántico fino.)
- **Temperatura alta (3.5):** descarrila. Ejemplo real medido: `The plot was boring and the acting felt wooden and lifeless` → `The site is boring and an actor feels completely of lifeless wood` (cambia el sujeto, gramática rota).

**Por qué UDA funciona igual** pese a que algunas aumentaciones rompen la etiqueta: (1) la mayoría son buenas y las malas se promedian en el batch; (2) la consistencia se mide sobre la predicción, no sobre la verdad; (3) el [confidence masking / TSA](../03-tres-regimenes-y-analisis) descarta ejemplos donde el modelo no está seguro. UDA es **robusto al ruido de aumentación, no inmune**.

## Los cinco datasets

El truco clave: la clase `Dataset` genérica se usa de dos formas — en la rama supervisada `y` es la etiqueta; en la no supervisada `y` es la **back-translation** (texto, no etiqueta).

| Dataset | `X` | `y` | Para qué |
|---|---|---|---|
| `all_train` (20.000 etiq.) | reseña | label 0/1 | régimen full (techo) |
| `train20` (20 etiq.) | reseña | label 0/1 | régimen low + parte sup. de UDA |
| `unsup_all` (~65.000) | original | **back-translation** | rama de consistencia |
| `val` (5.000) | reseña | label 0/1 | validación |
| `test` | reseña | label 0/1 | test final |

```python
# supervisado: y = etiqueta
Dataset(all_train_data["content"].values, (all_train_data["label"]=="pos").astype(np.longlong))
# no supervisado: y = versión aumentada (¡texto!)
Dataset(unsup_all_train_data["content"].values, unsup_all_train_data["bt"].values)
```

Los datos etiquetados también entran en la rama no supervisada (ignorando su etiqueta, usando su back-translation): para la KL no importa la etiqueta, solo tener pares (original, aumentado). Eso lleva los datos de consistencia a ~65k.

## Tokenización y los dataloaders

```python
tokenize(..., max_length=128, padding=True, truncation=True)   # BERT, 128 tokens por costo
```
El `max_length=128` trunca reseñas largas — es por costo computacional (el comentario del notebook lo admite: "con mayor largo podría tener mejor rendimiento"). Gotcha: si el sentimiento se revela al final de una reseña larga, truncar a 128 lo pierde.

**`unsup_ratio=3` (el balance de UDA):**
```python
unsup_ratio = 3
train_batch_size = 16
unsup_loader:  batch_size = 16*3 = 48    # 48 pares (orig, aum) por batch
```
Cada paso procesa **16 supervisados + 48 no supervisados**. El porqué del ratio >1: con solo 20 etiquetas, la señal supervisada es minúscula; la de consistencia es abundante. Meter 3× más datos no supervisados hace que el gradiente esté **dominado por la consistencia**, que es la que aporta generalización.

**Dos `collate_fn` distintos** materializan la diferencia:
```python
collate_fn:        (tokens, etiquetas)              # supervisado
collate_fn_unsup:  (tokens_orig, tokens_aumentado)  # no supervisado, ambos tokenizados
```

> **Conexión con MDM/FHIR:** la back-translation como "transformación que preserva la etiqueta" es análoga a las reglas de equivalencia en normalización de datos (dos formas de escribir el mismo nombre/dirección que deben matchear). Y como la back-translation, esas reglas fallan en un porcentaje de casos — la solución es la misma: confiar en el agregado + filtrar por confianza, no exigir perfección por caso.

---

**Siguiente:** [Los tres regímenes, TSA y análisis](../03-tres-regimenes-y-analisis).
