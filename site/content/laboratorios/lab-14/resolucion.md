---
title: "Resolucion"
weight: 60
math: true
---

> Resolucion razonada de las 11 preguntas de las 4 actividades del Laboratorio 14, mas insights consolidados. Los enunciados literales estan en [ejercicios](ejercicios).

---

## Actividad 1 — Comparacion entre versiones de BERT

La actividad pide cambiar `nv_model_version` a una version distinta de `bert-base-uncased` y comparar. Eleccion del estudiante: **`bert-base-multilingual-cased`** (mBERT).

Razones:

- Mismo tamano de parametros (~110M) → no se cae en Colab
- Es **cased** → distingue mayusculas, da pistas para NER
- Esta entrenado en **104 idiomas** → emergen patrones lingüisticos mas universales

El analisis comparativo completo (con screenshots de capas/cabezas relevantes y patrones sintacticos descubiertos como `Alexis → scored`) esta en [neuron-view-y-modelos](neuron-view-y-modelos).

### Pregunta 1.1 — ¿por que la misma oracion a izquierda y derecha?

**Respuesta:**

Porque es **self-attention**: ambas columnas son la misma secuencia (queries vs keys del mismo input). En BERT (encoder-only) la secuencia se mira a si misma. La cross-attention con dos secuencias distintas solo aparece en arquitecturas decoder o encoder-decoder.

### Pregunta 1.2 — parametro Layer

**Respuesta:**

Indica la **capa del encoder Transformer** (0-11 en BERT base). Cada capa es un bloque multi-head self-attention + feedforward. Capas tempranas capturan patrones locales; capas finales integran contexto global.

### Pregunta 1.3 — parametro Head

**Respuesta:**

Indica la **cabeza de atencion** dentro de la capa (0-11 en BERT base). Cada cabeza opera sobre proyecciones de 64 dimensiones (768/12) y aprende un patron distinto: diagonales, sintacticas, no-op, etc. Las 12 cabezas se concatenan al salir del bloque.

---

## Actividad 2 — Atenciones en el Decoder

### Pregunta 2.1 — cuantos graficos de cross-attention

**Configuracion:** `enc_layers: 6, dec_layers: 6, heads: 8`.

**Respuesta:** `48`

**Razonamiento:**

- La **cross-attention solo existe en el decoder**, no en el encoder. El encoder solo tiene self-attention sobre su propio input.
- 6 capas de decoder × 8 cabezas por capa = **48 mapas de cross-attention distintos**.

**Trampa que evitar:** si alguien suma encoder × decoder (6×6×8 = 288), cuenta combinaciones inexistentes. Si alguien suma encoder self-attention (6×8=48) + decoder self-attention (6×8=48) + decoder cross-attention (6×8=48) = 144, eso es el **total** de matrices de atencion del modelo, no solo las cross. El enunciado pide especificamente las cross-attentions.

### Pregunta 2.2 — dimensiones de la self-attention del decoder en T=5

**Respuesta:** `5×5`

**Razonamiento:**

En el paso T=5 de la generacion autoregresiva, el decoder ha procesado los **5 tokens previos** — incluyendo el `<sos>` y los 4 tokens generados. La self-attention con mascara causal opera sobre toda la secuencia procesada:

$$\text{Attention} = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right), \quad Q, K \in \mathbb{R}^{5 \times d_k}$$

La matriz `Q · K^T` tiene shape `(5, 5)`, donde la mitad triangular inferior (mas la diagonal) tiene valores reales y la mitad superior esta enmascarada con `-inf` antes del softmax:

```text
y_1 → atiende a [y_1]
y_2 → atiende a [y_1, y_2]
y_3 → atiende a [y_1, y_2, y_3]
y_4 → atiende a [y_1, y_2, y_3, y_4]
y_5 → atiende a [y_1, y_2, y_3, y_4, y_5]
```

**Nota:** el masking no cambia las dimensiones, solo los valores. La forma sigue siendo **5×5**.

### Pregunta 2.3 — V/F: "no es necesario enmascarar el futuro"

**Respuesta:** `Falso`

**Razonamiento:**

El masking causal en el decoder es **fundamental**. Sin el, dos cosas se rompen:

1. **Durante entrenamiento con teacher forcing**, se le pasa al decoder toda la secuencia target en paralelo. Sin mascara, el decoder veria los tokens futuros y aprenderia el atajo trivial "copia el token siguiente" en lugar de predecirlo.
2. **Coherencia entrenamiento/inferencia.** En inferencia, generas token a token autoregresivamente — el modelo solo tiene tokens previos disponibles. Si entrenas sin mascara, la distribucion de contextos seria muy distinta y el modelo no generalizaria.

> El masking causal es lo que **diferencia un decoder de un encoder bidireccional**. En BERT no hay mascara causal — cada token ve todo. En GPT si la hay — cada token ve solo el pasado.

### Pregunta 2.4 — V/F: "positional encoding como unica fuente de orden"

**Respuesta:** `Verdadero` (en el Transformer clasico de Vaswani 2017).

**Razonamiento:**

El self-attention es **permutation-invariant**: dadas las mismas keys/values, el output no depende del orden en que se presentan los queries. Sin positional encoding, "El perro mordio al hombre" y "El hombre mordio al perro" tendrian las mismas representaciones promediadas — se perderia la sintaxis.

El positional encoding (sinusoidal en el paper original, aprendible en BERT) se **suma** al embedding antes del primer bloque:

$$\text{embedding}_{\text{final}}[i] = \text{word\_embedding}[i] + \text{positional\_encoding}[i]$$

Es **la unica** senal de orden que el modelo tiene en la version clasica.

**Sutilezas (para cultura general):**

- En el **decoder con mascara causal**, la mascara tambien aporta info parcial de orden — un token sabe que solo ve "hacia atras". Tsai et al. 2019 muestran que esto rompe la permutation-invariance sin necesidad de positional encoding. Pero en el **encoder** (sin mascara), el positional encoding es estrictamente la unica fuente.
- Transformers modernos usan alternativas — **RoPE** (LLaMA), **ALiBi** (BLOOM/MPT), **relative position bias** (T5). Cumplen el mismo rol conceptual.

La afirmacion del lab se refiere al Transformer clasico, por lo que la respuesta canonica es **Verdadero**.

---

## Actividad 3 — Dimensiones de CLIP + prompt engineering

### Pregunta 3.1a — ¿a que corresponde la ultima dimension (512)?

**Respuesta:**

Corresponde a la **dimensionalidad del espacio compartido de embeddings de CLIP**. Tanto el image encoder (ViT-B/32) como el text encoder (Transformer) proyectan sus salidas a este espacio comun de 512 dimensiones mediante matrices de proyeccion finales (`visual.proj` y `text_projection`).

Es lo que permite calcular similitud coseno entre imagenes y textos: ambos viven en el mismo espacio vectorial. Para modelos CLIP mas grandes (ViT-L/14) este espacio es de **768 dimensiones**.

### Pregunta 3.1b — ¿por que la matriz de similitud es 1×101?

**Respuesta:**

Porque tenemos **1 imagen y 101 queries** (una por cada clase de Food101). La operacion `image_features @ text_features.T` multiplica `(1, 512)` por `(512, 101)` y produce una matriz `(1, 101)` donde cada entrada `[0, i]` es la similitud coseno entre la imagen y la query `i`. Si pasaramos un batch de N imagenes, la matriz seria `(N, 101)`.

### Pregunta 3.2 — Templates alternativos: Q1 y Q2

**Templates elegidos:**

```python
Q1 = "A photo of {}."
Q2 = "A close-up photo of a plate of {}, a popular dish."
```

**Resultados sobre el test set completo de Food101 (25,250 imagenes):**

| Template | Top-1 | Top-5 | Delta vs baseline |
| --- | --- | --- | --- |
| **Baseline** `'A photo of {}, a type of food.'` | **84.01%** | **97.31%** | — |
| **Q1** `'A photo of {}.'` | 78.41% | 94.93% | **−5.6 / −2.4** |
| **Q2** `'A close-up photo of a plate of {}, a popular dish.'` | 82.49% | 96.88% | **−1.5 / −0.4** |

**Analisis de Q1 — quitar contexto "food" cuesta 5.6 puntos:**

La pista `"a type of food"` aporta **senal real**. Sin ella, CLIP confunde imagenes de comida con otras cosas que tienen el mismo nombre (`"apple pie"` puede activarse en tatuajes, ilustraciones de libros; `"bagel"` tiene significado coloquial). El contexto culinario **acota la distribucion semantica** hacia fotografia real de comida servida.

**Analisis de Q2 — mas detalle NO es mejor:**

Q2 agrega tres pistas (`close-up`, `plate of`, `popular dish`) y sin embargo **rinde 1.5 puntos peor** que el baseline.

| Pista | Por que puede perjudicar |
| --- | --- |
| `close-up photo` | No todas las fotos de Food101 son close-up. Muchas son tomas amplias |
| `plate of` | Muchos platos no estan servidos en plato — sushi en madera, sandwich en mano |
| `popular dish` | Sesga hacia comidas occidentales mainstream. `bibimbap`, `baklava`, `pho` pueden no encajar bien |

> **Leccion clave de prompt engineering:** el template optimo es **lo suficientemente especifico** para anclar el contexto pero **no tan especifico** que excluya casos validos. El template del lab esta **bien balanceado**.

---

## Actividad 4 — Tus propias imagenes

### Setup elegido

5 imagenes de un repo publico de samples de ImageNet (`EliSchwartz/imagenet-sample-images` y `pytorch/hub`):

| # | Imagen | Query |
| --- | --- | --- |
| 1 | Perro samoyedo | `"A photo of a dog"` |
| 2 | Pinzon (brambling) | `"A photo of a bird"` |
| 3 | Ambulancia | `"A photo of an ambulance"` |
| 4 | Pizza de pepperoni | `"A photo of a pizza"` |
| 5 | Limones | `"A photo of a lemon"` |

### Resultado: matriz de similitud 5×5

```text
                          dog  bird  ambul  pizza  lemon
A photo of a dog          0.22  0.20  0.20   0.20   0.20
A photo of a bird         0.20  0.22  0.20   0.20   0.20
A photo of an ambulance   0.19  0.19  0.22   0.20   0.19
A photo of a pizza        0.20  0.20  0.19   0.22   0.19
A photo of a lemon        0.19  0.20  0.19   0.19   0.22
```

**Diagonal correcta para las 5 imagenes** — Top-1 acierta en todas. Pero los valores estan apretados (0.22 vs 0.19-0.20).

### Por que los valores estan apretados

La celda 66 del notebook omite el factor `logit_scale = 100` que si esta en Food101 y Cars:

```python
similarity = (image_features @ text_features.T).softmax(dim=-1).cpu().numpy().T
```

Sin el `100.0 *`, el softmax opera sobre valores coseno entre [-1, 1] y produce distribuciones casi uniformes. Con el factor 100, la diagonal seria ~1.00 y el resto ~0.00.

### Analisis escrito (campo A)

> "Test con 5 imagenes ImageNet distintas (perro samoyedo, pinzon, ambulancia, pizza, limones) y 5 queries 'A photo of a {X}'. Resultados: matriz diagonal con maximo en la diagonal (0.22) y 0.19-0.20 fuera. Top-1 correcto para las 5 imagenes. Conclusion: CLIP separa perfectamente clases visualmente distintas; el contraste de la visualizacion depende de la temperatura del softmax, no solo de la calidad del modelo."

---

## Resumen tabular de respuestas

| Pregunta | Celda | Respuesta |
| --- | --- | --- |
| 1.1 — ¿por que la misma oracion a ambos lados? | P1/46 | Es self-attention; en BERT (encoder-only) la secuencia se mira a si misma |
| 1.2 — parametro Layer | P1/48 | Capa del encoder Transformer (0-11 en BERT base); cada capa es multi-head self-attention + feedforward |
| 1.3 — parametro Head | P1/50 | Cabeza de atencion (0-11 en BERT base); cada una opera sobre 64 dims (768/12) y aprende patrones distintos |
| 2.1 — cantidad de cross-attentions totales | P1/55 | **48** (6 capas decoder × 8 cabezas) |
| 2.2 — dimensiones de self-attention en T=5 | P1/57 | **5×5** (matriz triangular inferior con masking causal) |
| 2.3 — V/F: "no es necesario enmascarar el futuro" | P1/60 | **Falso** |
| 2.4 — V/F: "positional encoding como unica fuente de orden" | P1/62 | **Verdadero** (en el Transformer clasico) |
| 3.1a — ¿que es la dim 512 de los features? | P2/38 | Espacio compartido de embeddings de CLIP (image + text encoders proyectan ahi) |
| 3.1b — ¿por que matriz 1×101? | P2/40 | 1 imagen × 101 queries (una por clase Food101) |
| 3.2 — templates Q1 y Q2 | P2/44-45 | Q1: 78.41/94.93 (−5.6/−2.4). Q2: 82.49/96.88 (−1.5/−0.4) |
| 4 — analisis matriz 5×5 | P2/71 | Top-1 correcto en las 5 imagenes; contraste apretado por falta del factor logit_scale=100 |

---

## Insights consolidados del Lab 14

### Parte 1 — interpretabilidad de BERT

1. **Los patrones de atencion emergen sin supervision** y son estables entre modelos tipo BERT:
   - Sink hacia `[CLS]` en capa 0 — el modelo aun no procesa nada
   - Valle no-op hacia `[SEP]` en capas medias (6-7-8) — "no encontre info, deposito mi atencion aqui"
   - Concentracion en `[CLS]` en capa 8 — consolidacion de la frase
   - Diversidad informativa en capas finales (9-11) — cabezas semanticas con distintos vertices

2. **La atencion no solo "fija", tambien "rechaza".** En Neuron View se ven valores negativos en `q · k`: el modelo aprende activamente que ciertos tokens no son informativos en ciertas cabezas.

3. **Modelos multilingües tienen cabezas mas interpretables que monolingües.** mBERT muestra patrones sintacticos genuinos (Alexis → scored, sujeto-verbo) en capas donde bert-base-uncased solo muestra no-op. La hipotesis (Pires et al. 2019): entrenar en 104 idiomas obliga a aprender abstracciones lingüisticas mas universales.

4. **Si las cabezas no-op son inutiles, se pueden podar.** Voita et al. 2019 muestran que se puede podar hasta el 50% de las cabezas de BERT sin perder casi nada de performance.

### Parte 2 — CLIP zero-shot

1. **Embeddings compartidos imagen-texto permiten clasificacion sin fine-tuning.** En Food101 (101 clases), CLIP-ViT-B/32 zero-shot logra **84% Top-1**, casi alcanzando el ~88-90% de un ResNet50 fine-tuned.

2. **El template importa: prompt engineering del lado del text encoder.** Quitar la pista `"a type of food"` cuesta 5.6 puntos. Mas detalle no es siempre mejor — sobreespecificar puede excluir casos validos.

3. **Zero-shot tiene limites claros.** Stanford Cars (196 clases con distinciones por ano y trim) cae a **58% Top-1** vs 84% en Food101. La razon: las captions de internet raramente tienen el nivel de detalle de "BMW M5 2010 Sedan" — CLIP no aprendio a distinguir esas distinciones sutiles.

4. **La brecha Top-1 vs Top-5 es diagnostica:**
   - Food101: 84% vs 97% (brecha 13) — modelo decidido
   - Stanford Cars: 58% vs 90% (brecha 32) — modelo en el vecindario correcto pero indeciso entre opciones similares

### La idea de fondo del lab

El Transformer original es una arquitectura **encoder-decoder** disenada para traduccion. A partir de esa base surgen las tres familias dominantes hoy:

| Familia | Arquitectura | Direccionalidad | Tareas tipicas | Ejemplos |
| --- | --- | --- | --- | --- |
| **Encoder-only** | Solo encoder | Bidireccional | Clasificacion, NER, QA, NLI | BERT, RoBERTa, mBERT |
| **Decoder-only** | Solo decoder, mascara causal | Autoregresiva | Generacion de texto | GPT, LLaMA, Mistral |
| **Encoder-decoder** | Ambos, con cross-attention | Hibrida | Seq2seq: traduccion, resumen | T5, BART, MarianMT |
| **Multimodal contrastive** | Dual encoder | Producto de espacios | Zero-shot retrieval/classification | CLIP, SigLIP, ALIGN |

Este lab te dio herramientas para inspeccionar la **primera** familia (encoder-only) en profundidad y **observar el comportamiento** de la cuarta (multimodal). Las mismas tecnicas aplican con cambios menores.

### Conexion con la teoria de la clase

- La arquitectura completa del Transformer (encoder + decoder + cross-attention + positional encoding + multi-head) esta en la [teoria de la clase 14](/clases/clase-14/).
- Para fundamentos sobre Q/K/V, softmax escalado y multi-head, ver [fundamentos/transformers](/fundamentos/transformers/).
- Para BERT especifico (MLM, NSP, fine-tuning), ver [fundamentos/bert](/fundamentos/bert/).
