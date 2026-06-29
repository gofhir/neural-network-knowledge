---
title: "Arquitectura del KVMemoryReader"
weight: 1
---

El modelo entero vive en una sola clase, `KVMemoryReader`, y se resume en cuatro piezas:

```
__init__   →  embeddings (¡compartidos!) + la transformación de hop
gen_mask   →  utilidad para enmascarar padding
forward    →  codificar → [hop × N] → scorear candidatos
predict    →  envuelve forward y devuelve el índice del candidato ganador
```

Lo notable: todo el modelo tiene **solo dos parámetros entrenables** — una matriz de embeddings gigante (compartida por los cuatro roles) y la transformación lineal entre hops. Sorprendentemente pequeño para lo que logra.

## Pieza 1 — Embeddings compartidos

```python
self.question_embed = nn.Embedding(n_embed, d_embed)   # 186841 x 300
self.key_embed   = self.question_embed
self.value_embed = self.question_embed
self.cand_embed  = self.question_embed
self.hop_linear  = nn.Linear(d_embed, d_embed)         # la matriz R del paper
```

Las cuatro "matrices de embedding" (pregunta, key, value, candidato) son **el mismo objeto**. No hay copias — son referencias al mismo `nn.Embedding`.

**El porqué.** El paper original mantiene matrices distintas (A para keys, C para values...). Acá la separación se logra de otra forma: el truco del **prefijo `1:`** (ver [Construcción de la KB](../02-construccion-kb)) hace que `1:movie` y `movie` sean **filas distintas** de la misma tabla. La misma palabra recibe embeddings diferentes según su rol (key vs. value) porque son IDs de token diferentes. Por eso `n_embed = 186841` es tan grande: es esencialmente el vocabulario duplicado. Una matriz de `186841 × 300` floats ≈ **56M de parámetros** — casi todo el modelo es esta tabla.

## Pieza 2 — Codificación Bag-of-Words

La representación de la pregunta (y de cada key/value) es el **promedio de los embeddings de sus tokens**:

```python
embed_q = self.question_embed(question)              # [B, max_q_len, 300]
q_word_mask = self.gen_mask(max_q_len, q_length)     # [B, max_q_len]  1=real, 0=padding
q_state = torch.bmm(q_word_mask.unsqueeze(1), embed_q)   # suma enmascarada → [B, 1, 300]
q_state /= q_length.view(batch_size, 1, 1)               # promedio
```

El `bmm(máscara, embeddings)` suma solo los tokens reales (la máscara pone 0 en el padding); luego divide por la longitud real → promedio. La misma mecánica codifica cada una de las ~3000 keys del ejemplo a un vector de 300-d (`key_feat`).

**El porqué de Bag-of-Words y no RNN/Transformer.** KV-MemNN (2016) es pre-Transformer. El paper usa BoW + positional encoding simple; acá se simplifica a BoW puro. **Pierde el orden de las palabras** — "X dirigió a Y" e "Y dirigió a X" colapsan al mismo vector. Para WikiMovies funciona porque las preguntas son plantillas donde las entidades pesan más que el orden, pero es la limitación que explica varios de los fallos analizados en [Experimentos](../03-experimentos-y-analisis).

## Pieza 3 — El bucle de hops (el "razonamiento")

```python
for _ in range(self.n_hops):                          # n_hops = 2
    key_similarity = torch.bmm(q_state, key_feat.transpose(1, 2))   # producto punto q·keys
    key_similarity = key_similarity.masked_fill(k_num_mask == 0, float('-inf'))
    attention = F.softmax(key_similarity, dim=-1)     # distribución de atención sobre keys
    self.attentions.append(attention)                 # se guarda para visualizar
    pondered_value = torch.bmm(attention, embed_v)    # value reading: suma ponderada de VALUES
    q_state = torch.add(pondered_value, q_state)      # residual: query += lectura
    q_state = self.hop_linear(q_state)                # R·(query)  para el siguiente hop
```

Los pasos del algoritmo KV-MemNN:

1. **Key addressing** — producto punto entre la query y cada key → similitud.
2. **Mask + softmax** — las keys de padding van a `-inf` (softmax = 0). Se obtiene una distribución de atención sobre los candidatos.
3. **Value reading** — suma de los **values** ponderada por la atención. Detalle clave: la atención se calcula con las **keys**, pero la lectura devuelve los **values** (`embed_v`, no `key_feat`) — exactamente la separación key/value que da nombre al modelo.
4. **Update con residual** — la nueva query es la vieja **más** lo leído, transformada por `R` (`hop_linear`). Esto permite que el hop 2 busque algo distinto que el hop 1.

**El porqué de 2 hops.** Un hop resuelve QA de un salto ("¿quién dirigió X?"). Dos hops permiten composición ("¿quién actuó en la película que dirigió X?"). En la [visualización de atención](../03-experimentos-y-analisis) se ve el efecto: el hop 1 reparte la atención (localiza la zona de la KB), el hop 2 la concentra (refina hacia la respuesta).

## Pieza 4 — Scoring de candidatos (la salida)

```python
cand_similarity = torch.bmm(q_state, embed_c.transpose(1, 2))   # q final · candidatos
cand_similarity = cand_similarity.masked_fill(c_num_mask == 0, float('-inf'))
cand_score = F.log_softmax(cand_similarity, dim=-1)             # log-probabilidades
return cand_score.squeeze(1)                                   # [B, max_n_cands]
```

Tras los hops, la query final se compara con los **embeddings de los candidatos de respuesta** → score por candidato. La salida tiene tamaño `max_n_cands`, **no** `n_embed`: el modelo elige **entre los candidatos recuperados**, no entre todo el vocabulario. Es un problema de **ranking/selección sobre candidatos dinámicos**, no de clasificación de vocabulario fijo — idéntico al patrón retrieve-then-rank de [Dense Retrieval](/fundamentos/dense-retrieval).

`predict` simplemente envuelve `forward` y devuelve el `argmax` sobre candidatos, que se compara contra el índice de la respuesta correcta.

## El flujo completo

```
pregunta ──BoW──► q_state ─┐
                            │  ┌──── hop (×2) ────────────────┐
KB keys ──BoW──► key_feat ──┼─►│ atención(q,keys) → lee values │──► q_state'
KB values ─────► embed_v ───┘  │ q_state = R(q_state + lectura)│
                               └───────────────────────────────┘
                                          │
candidatos ────► embed_c ──► score = q_state'·embed_c ──► argmax ──► respuesta
```

## Conexión con la self-attention

El bloque `atención(q, keys) → lee values` es **literalmente** el mecanismo query/key/value de los Transformers, dos años antes de *Attention is All You Need*. La diferencia: en KV-MemNN la memoria es **externa y explícita** (entradas de una KB), mientras que en self-attention los key/value se derivan de la misma secuencia. Es la razón por la que la clase 30 llama a esta familia "la prehistoria de la self-attention". Ver el [fundamento de self-attention](/fundamentos/self-attention) para la conexión formal.

---

**Siguiente:** [Construcción de la KB y blocking](../02-construccion-kb) — de dónde salen las keys, los values y los candidatos.
