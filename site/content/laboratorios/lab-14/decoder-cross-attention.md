---
title: "Decoder cross-attention (Actividad 2)"
weight: 40
math: true
---

Esta seccion cierra la Parte 1 del lab con un cambio de tema: hasta aqui inspeccionamos un **encoder-only** (BERT). Ahora pasamos a un **Transformer encoder-decoder completo** (la arquitectura original de Vaswani et al. 2017 para traduccion automatica). El lab no implementa nada — son **preguntas conceptuales** que verifican que el alumno entendio:

1. La diferencia entre **self-attention** y **cross-attention**
2. Como cuenta el numero de cabezas/capas/atenciones en una arquitectura encoder-decoder
3. El proposito del **masking causal** en el decoder
4. El rol del **positional encoding** como fuente de orden secuencial

El Transformer del lab tiene la siguiente configuracion:

```
enc_layers: 6
dec_layers: 6
heads: 8
```

— es decir, 6 capas de encoder, 6 capas de decoder, 8 cabezas por capa. Es esencialmente la arquitectura del paper *"Attention Is All You Need"*.

## Los tres tipos de atencion en el decoder

Antes de las preguntas, conviene fijar la anatomia. Un **bloque del decoder** del Transformer tiene **tres** sub-capas, en este orden:

1. **Masked self-attention** — el decoder se atiende a si mismo, pero con **mascara causal** (cada token solo puede ver tokens previos + si mismo)
2. **Cross-attention** — Q viene del decoder, K y V vienen del **output del encoder**
3. **Feedforward** — una MLP token-a-token

```
Output del encoder ──┐
                     │
   y_{<t} ──[mask SA]──[cross-att]──[FFN]── y_t
              ↑               ↑
            Q,K,V          K,V (de encoder)
            (decoder)      Q   (de decoder)
```

| Atencion | Donde aparece | Q viene de | K, V vienen de | Mascara |
| --- | --- | --- | --- | --- |
| **Self-attention** | Encoder (todas las capas) | Encoder | Encoder | Sin mascara |
| **Masked self-attention** | Decoder (primera sub-capa) | Decoder | Decoder | **Mascara causal** |
| **Cross-attention** | Decoder (segunda sub-capa) | Decoder | **Encoder** | Sin mascara causal |

La **cross-attention** es lo que permite que el decoder "consulte" la frase fuente al traducir. Es **el mecanismo descubierto en Bahdanau et al. 2014** que dio origen al boom de atencion (visto en Lab 13 Parte 2) — el Transformer lo reformula con multi-head y dot-product escalado.

> El encoder **no tiene** cross-attention porque no necesita atender a nada externo — solo procesa su propio input.

## Pregunta 1 — ¿cuantos graficos de cross-attention en total?

**Enunciado** *(parte 1, celda 54)*:

> El notebook muestra el diagrama de cross-attention para **una capa y una cabeza** del modelo. ¿Cuantos graficos tendriamos si pudieramos visualizar **todas** las cross-attentions?

### Razonamiento

- La cross-attention **solo existe en el decoder**, no en el encoder
- 6 capas de decoder × 8 cabezas por capa = **48 mapas de cross-attention distintos**

**Respuesta: `48`**

### Trampas posibles

Si alguien suma encoder × decoder (6×6×8 = 288) cuenta combinaciones inexistentes. Si alguien suma encoder self-attention (6×8=48) + decoder self-attention (6×8=48) + decoder cross-attention (6×8=48) = **144** atenciones totales, pero eso es el total general, no solo las cross. El enunciado pide especificamente las cross-attentions, asi que la respuesta es **48**.

## Pregunta 2 — dimensiones de la self-attention del decoder en T=5

**Enunciado** *(parte 1, celda 56)*:

> Si estuvieramos haciendo decoding en el paso T=5, **es decir generando el 5to token de salida**, ¿que dimensiones tendria la matriz de **self-attention del decoder**?

### Razonamiento

Durante la generacion autoregresiva, en el paso T=5 el decoder ha procesado los **5 tokens previos** (incluyendo el `<sos>` y los 4 tokens generados hasta ese momento). La self-attention con mascara causal opera sobre toda la secuencia procesada hasta el momento.

La matriz `Q · K^T` del decoder tiene shape `(seq_len × seq_len)` donde `seq_len = 5`:

$$
\text{Attention} = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right), \quad Q, K \in \mathbb{R}^{5 \times d_k}
$$

La forma de la matriz **es 5×5**, aunque solo la mitad triangular inferior tiene valores reales — la mitad superior esta enmascarada (con `-inf` antes del softmax, lo que tras softmax queda en 0):

```
y_1 → [y_1]                  (atiende a 1)
y_2 → [y_1, y_2]             (atiende a 2)
y_3 → [y_1, y_2, y_3]        (atiende a 3)
y_4 → [y_1, y_2, y_3, y_4]   (atiende a 4)
y_5 → [y_1, y_2, y_3, y_4, y_5] (atiende a 5)
```

El masking **no cambia las dimensiones**, solo los valores de las celdas inalcanzables.

**Respuesta: `5x5`**

### Sutileza

Una interpretacion alternativa: si T=5 se entiende como "los 4 tokens previos sin contar el actual", la matriz seria **4×4**. Ambas son defendibles dependiendo de la convencion del curso. La interpretacion mas comun en cursos y la que normalmente se espera es **5×5**, asumiendo que la secuencia incluye al token actual.

## Pregunta 3 — V/F: "no es necesario enmascarar el futuro"

**Enunciado** *(parte 1, celdas 58-59)*:

> "En el decoder de un Transformer, como lo vimos en clase, **no es necesario** enmascarar 'el futuro'." (V/F)

### Razonamiento

**Falso.** El masking causal es **fundamental** por dos razones tecnicas:

1. **Durante entrenamiento con teacher forcing**, se le pasa al decoder **toda** la secuencia target en paralelo (no token a token). Sin mascara, el decoder vería los tokens futuros y aprenderia el atajo trivial "copia el token siguiente" en lugar de predecirlo. La mascara causal garantiza que la prediccion en posicion `t` solo dependa de las posiciones `1..t-1` (mas la posicion `t` misma).

2. **Coherencia entre entrenamiento e inferencia.** En inferencia, generas token a token autoregresivamente — el modelo solo tiene tokens previos. Si entrenas sin mascara, la distribucion de contextos seria muy distinta y el modelo no generalizaria.

**Respuesta: `Falso`**

> El masking causal es lo que diferencia un **decoder** de un **encoder bidireccional**. En BERT (encoder-only) no hay mascara causal — cada token ve todo. En GPT (decoder-only) si hay mascara causal — cada token ve solo el pasado.

## Pregunta 4 — V/F: positional encoding como unica fuente de orden

**Enunciado** *(parte 1, celdas 61-62)*:

> "En un Transformer, el positional encoding se utiliza como **unica** fuente de informacion del orden de la secuencia." (V/F)

### Razonamiento

**Verdadero** (en el Transformer clasico de Vaswani 2017, que es lo que se ve en clase).

El **self-attention es permutation-invariant**: si das la secuencia `[el, perro, mordio]` o `[mordio, perro, el]`, el output (sin positional encoding) seria **identico**, porque la atencion solo computa dot-products entre Q y K — no tiene noción de orden.

Sin positional encoding:

- "El perro mordio al hombre" y "El hombre mordio al perro" tendrian las mismas representaciones promediadas
- Se perderia completamente la sintaxis

El positional encoding (sinusoidal en el paper original, aprendible en BERT) se **suma** al embedding antes del primer bloque:

$$\text{embedding}_{\text{final}}[i] = \text{word\_embedding}[i] + \text{positional\_encoding}[i]$$

Es **la unica** senal de orden que el modelo tiene en la version clasica.

**Respuesta: `Verdadero`**

### Sutileza extra (para cultura general)

En el decoder con mascara causal, **la mascara tambien aporta informacion parcial de orden** — un token sabe que solo puede "ver hacia atras", lo que implica una posicion relativa. Algunos papers (Tsai et al. 2019) argumentan que esto rompe la permutation-invariance sin necesidad de positional encoding. Pero en el **encoder** del Transformer (sin mascara), el positional encoding es estrictamente la unica fuente.

Tambien, Transformers modernos usan alternativas — **RoPE** (Rotary Positional Embedding, LLaMA), **ALiBi** (BLOOM/MPT), **relative position bias** (T5) — que tambien aportan info de orden y a veces reemplazan al positional encoding aditivo. Pero cumplen el mismo rol conceptual.

La afirmacion del lab dice "como lo vimos en clase" y se refiere al Transformer clasico, por lo que la respuesta canonica es **Verdadero**.

## Resumen de respuestas

| Celda | Pregunta | Respuesta |
| --- | --- | --- |
| 55 | Cantidad de cross-attentions totales | `48` |
| 57 | Dimensiones de self-attention en T=5 | `5x5` |
| 60 | "No es necesario enmascarar el futuro" | `Falso` |
| 62 | "Positional encoding es la unica fuente de orden" | `Verdadero` |

Razonamiento completo en [resolucion](../resolucion).

## La idea de fondo que cierra el lab

El Transformer original de Vaswani es una arquitectura **encoder-decoder** disenada para traduccion: el encoder consume la frase fuente, el decoder genera la frase destino consultando al encoder via cross-attention y manteniendo autoregresividad via masking causal.

A partir de esa arquitectura base surgen tres familias:

1. **Encoder-only** (BERT, RoBERTa, mBERT) — solo el encoder. Bidireccional, ideal para clasificacion, NER, QA, NLI. Lo que inspeccionamos en este lab.
2. **Decoder-only** (GPT, LLaMA, Mistral) — solo el decoder con masking causal. Autoregresivo, ideal para generacion de texto.
3. **Encoder-decoder** (T5, BART, MarianMT) — la arquitectura original completa. Ideal para tareas seq2seq (traduccion, resumen, paraphrase).

El lab te dio herramientas para inspeccionar el primer tipo. Para los otros dos, las mismas tecnicas aplican con minor modifications — `bertviz` soporta GPT-2 con `view='neuron'` y los conceptos de Q/K/V, mascara causal, sink y no-op se trasladan directamente.

## Cierre

Con esta seccion termina el lab Parte 1. Las respuestas concretas a las **7 preguntas conceptuales** (3 de Actividad 1 + 4 de Actividad 2) estan razonadas en [resolucion](../resolucion). Los enunciados literales del notebook estan en [ejercicios](../ejercicios).

Para profundizar en la teoria del Transformer, ver:

- [Clase 14 — Teoria](/clases/clase-14/) — el lecture de Gabriel
- [Fundamento: Transformers](/fundamentos/transformers/) — encoder-decoder, multi-head, positional encoding
- [Fundamento: BERT](/fundamentos/bert/) — MLM, NSP, fine-tuning para tareas downstream
