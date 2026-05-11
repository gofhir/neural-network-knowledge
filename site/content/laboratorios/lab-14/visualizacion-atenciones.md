---
title: "Visualizacion de atenciones (head + model view)"
weight: 20
math: true
---

Esta es la seccion donde **se abre el capo de BERT**. Hasta aqui usabamos el modelo como caja negra para NER. Ahora vamos a ver como distribuye atencion token-a-token, capa-a-capa, cabeza-a-cabeza usando dos vistas de `bertviz`: **Head View** (una capa, todas sus cabezas en colores superpuestos) y **Model View** (las 12 capas × 12 cabezas a la vez como matriz de mini-grillas).

La leccion clave: los patrones que emergen — sink hacia `[CLS]`, no-op masivo hacia `[SEP]`, cabezas diagonales locales, diversidad informativa en capas finales — **no son aleatorios**. Son **propiedades emergentes** del pre-entrenamiento de BERT que aparecen en el mismo orden y forma en distintos modelos tipo BERT. Estan documentados en literatura (Clark et al. 2019, Voita et al. 2019) y son tan estables que justifican la **poda de cabezas** (head pruning) como tecnica de compresion.

## Refresco rapido de la formula de atencion

Para que tenga sentido lo que vamos a ver:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

Para BERT base (que es lo que carga el lab):

- **12 capas** (Transformer encoder blocks)
- **12 cabezas** por capa → **144 patrones de atencion distintos** en total
- Cada cabeza opera sobre proyecciones de **dim = 768/12 = 64**
- Cada cabeza produce una matriz `seq_len × seq_len` con pesos `softmax(...)` sobre los tokens

Por eso es interesante visualizar: cada cabeza aprende a "mirar" patrones distintos — y esos patrones tienen una **estructura emergente** que descubriremos.

## Setup: recuperar los pesos de atencion

Recordatorio de la seccion anterior — el modelo se cargo con `output_attentions=True`, lo que hace que el forward pass devuelva las matrices de atencion. El acceso es:

```python
attention = model(**inputs)[-1]   # tupla de 12 tensores (uno por capa)
                                  # cada tensor shape: (1, 12, seq_len, seq_len)
                                  #                    batch, heads, query, key
```

Despues se construye la lista de tokens via `convert_ids_to_tokens` para que bertviz pueda etiquetar las filas/columnas:

```python
input_id_list = inputs['input_ids'][0].tolist()
tokens = tokenizer.convert_ids_to_tokens(input_id_list)
call_html(view='head')      # inyecta D3.js v3.5.8 al notebook
head_view(attention, tokens)
```

`call_html` es una funcion auxiliar definida en la celda 10 que carga los scripts de D3 desde CDN. Cada vista de bertviz requiere una version distinta:

| Vista | Version de D3 |
| --- | --- |
| `head_view` | 3.5.8 |
| `model_view`, `neuron_view` | 5.7.0 |

Si las visualizaciones salen en blanco en Colab, generalmente es porque `call_html` no se llamo o D3 no cargo.

## Head View — una capa a la vez

### Que se ve

`head_view` renderiza una sola capa, mostrando **las 12 cabezas en colores superpuestos**. La interfaz tiene:

- **Selector de capa (Layer)** arriba: dropdown de 0 a 11
- **Filas a la izquierda**: lista de tokens (queries — "desde quien")
- **Filas a la derecha**: la misma lista de tokens (keys — "hacia quien")
- **Lineas curvas conectando ambas listas**: el peso de atencion. Mas gruesa/oscura = mas atencion
- **12 cuadrados de colores arriba**: cada cuadrado es una cabeza. Doble click aisla esa cabeza (oculta las otras 11)

### Patrones por capa

Inspeccionando la frase `'Eduardo Vargas le metio un gol a Espana en el mundial de Brasil.'` y cambiando el selector de capa:

**Capa 0** — las lineas convergen masivamente hacia `[CLS]` del lado derecho. Casi todas las cabezas atienden ahi.

![Head view Layer 0 - sink CLS](/laboratorios/lab-14/head-view-layer-0-cls-sink.png)

**Capas 6 y 8** — las lineas convergen hacia `[SEP]` del lado derecho. La concentracion es muy alta.

Aislando la cabeza 7 (rosa) de la capa 6 con doble click:

![Head view layer 6 head 7 - no-op SEP](/laboratorios/lab-14/head-view-layer-6-head-7-sep-noop.png)

**Casi todas las lineas terminan en `[SEP]`**. Es una cabeza no-op prototipica. Hay 2-3 excepciones sutiles:

- `[CLS]` → `[CLS]` (autoatencion arriba)
- `Eduardo` izquierda → parece ir a `metio` (potencial conexion sintactica sujeto-verbo)
- `gol` y otros con lineas mas diagonales

### Por que pasan estos patrones — los dos clasicos de BERT

**Attention sink hacia `[CLS]` en capa 0**

En las capas tempranas (0-2), BERT aun no tiene representaciones utiles de los tokens — apenas empezo a procesar el texto. Las cabezas terminan **descartando** la query y atendiendo masivamente a `[CLS]` como una especie de "no atender" disfrazado. Como `[CLS]` aparece **siempre** y esta **siempre en posicion 0**, el modelo aprendio que atender ahi es un buen default cuando no hay info contextual relevante. Es basicamente un "soft no-op".

> **Referencia:** Clark et al. 2019, *"What Does BERT Look At? An Analysis of BERT's Attention"* — seccion 4.1 documenta este patron exacto.

**No-op attention hacia `[SEP]` en capas medias**

A medida que avanzan las capas, `[CLS]` empieza a **acumular informacion** real (porque su embedding final se usa para clasificacion de secuencia). Atender a `[CLS]` deja de ser barato — perturbas su representacion. Entonces el modelo migra el "no-op" a **`[SEP]`**, que es el token mas inocuo: marca fin de oracion y nadie lo usa para tareas down-stream.

`[SEP]` es el token "tranco" — esta al final, no se usa para nada, y nada lo lee. Es el lugar perfecto para "estacionar" atencion que no encontro nada util.

> **Implicacion practica:** Voita et al. 2019, *"Analyzing Multi-Head Self-Attention"*, muestran que **se puede podar hasta el 50% de las cabezas** sin perder casi nada de performance. Las cabezas no-op son las victimas naturales — sus pesos no aportan senal lingüistica.

## Model View — las 144 cabezas simultaneamente

`model_view` muestra una **matriz 12 × 12 de mini-grillas**:

- **Filas** = 12 capas (0 arriba, 11 abajo)
- **Columnas** = 12 cabezas
- **Cada celda** = mini-version del head view, comprimida

Esta vista es la mas util para **detectar patrones globales** porque ves las 144 cabezas a la vez en lugar de cambiar de capa una por una.

### Capas tempranas: caos + diagonales locales (0 a 4)

![Model view capas 0-4](/laboratorios/lab-14/model-view-layers-0-4.png)

**Capa 0** (azul, fila superior): atencion bastante dispersa — patrones caoticos en X, lineas en multiples direcciones. Esto refleja que **las representaciones iniciales no estan contextualizadas** todavia. Hay sub-patron consistente: muchas cabezas tienen conexiones hacia `[CLS]` (esquina superior izquierda visualmente), pero todavia no es absoluto.

**Capa 1** (naranja): aparecen patrones **triangulares** — vertices abajo en muchas cabezas (0-3, 7), sugiriendo que muchos tokens convergen a uno o pocos tokens especificos. Cabezas 9-11 muestran **lineas horizontales paralelas** — patron de cabeza diagonal (look-self).

**Capa 2** (verde): cabezas 6 y 7 muestran **lineas horizontales muy marcadas** — atencion diagonal pura. Cada token query atiende al mismo token key (o al adyacente). Estas son **cabezas "look-self" o "look-shift"** que propagan informacion local sin desperdiciar dimensiones cruzadas.

**Capa 3** (rojo): cabezas 1 y 9 son cabezas diagonales muy claras. Resto de cabezas tiene patrones triangulares con vertices en posiciones distintas.

**Capa 4** (morado): patrones mas difusos, formas como "V" invertidas — sugiere convergencia a tokens centrales. Transicion hacia el "valle no-op" que viene.

> **Insight conceptual:** las cabezas diagonales son el equivalente Transformer de una **convolucion 1×1 o 3×1** sobre la secuencia. Hacen un trabajo muy especifico: propagar informacion local sin desperdiciar capacidad cruzada. Aparecen consistentemente en estas capas.

### Capas medias: el valle no-op (5 a 9)

![Model view capas 5-9](/laboratorios/lab-14/model-view-layers-5-9.png)

**Capa 5** (cafe): mezcla. Cabezas 4 y 8 son diagonales clasicas, cabezas 0-3 son triangulares, cabezas 9-11 ya tienen patrones complejos. Es **transicional** — el modelo esta dejando atras patrones locales y empezando a construir representaciones globales.

**Capa 6** (rosa): triangulos invertidos muy marcados. Cabezas 0-1 con concentracion masiva.

**Capa 7** (gris) — **la capa "no-op" canonica**. Casi todas las 12 cabezas tienen el **mismo patron triangular** convergiendo al vertice inferior derecho. **`[SEP]` recibe casi toda la atencion de casi todas las cabezas.**

Esto confirma que el patron no-op **no es una capa, es un bloque de varias capas medias** (6-7-8). El modelo dedica una franja completa a "almacenar" atenciones no utiles.

**Capa 8** (amarillo): patrones triangulares muy fuertes — pero ahora con **vertice abajo a la izquierda**. Es decir, **todo converge a `[CLS]`** (que esta en posicion 0). Esta capa esta **consolidando informacion en `[CLS]`**, preparando el resumen de la frase.

**Capa 9** (cian): cambio drastico. **Patrones mucho mas variados** — algunas cabezas con lineas horizontales (diagonales), otras con vertices en distintos lugares. Esta capa **NO** colapsa en un solo token. Distintas cabezas atienden a distintos tokens informativos.

### Capas finales: integracion semantica (10-11)

![Model view capas 8-11 (zoom finales)](/laboratorios/lab-14/model-view-layers-8-11.png)

**Capa 10** (azul): cada cabeza tiene un **vertice distinto**. Cabezas 0-3 con vertices en posiciones tempranas, cabezas 7-8 con triangulos muy especificos hacia tokens informativos, cabezas 10-11 con vertices medios. **Senal de cabezas semanticas** — cada una se especializo en atender un token clave distinto. Es la capa que probablemente hace el trabajo real para la cabeza NER final.

**Capa 11** (naranja, la ultima) — el zoo. La capa mas diversa del modelo. Mezcla de:

- Cabezas **diagonales** (cabezas 1, 4, 10 con lineas horizontales) — mantienen info del propio token
- Cabezas con **triangulos densos** (cabezas 7-9) — concentran info en tokens especificos
- Cabezas **complejas** (cabezas 0, 11) — patrones difusos integrando varios tokens

La ultima capa es la que el clasificador NER usa directamente. Cada token-output de la capa 11 pasa por la cabeza lineal `768 → 9 clases`. Para que NER funcione, **cada token debe contener su contexto local + global**. Las cabezas diagonales mantienen el contexto local; las triangulares aportan contexto global. **La mezcla es la clave.**

## El patron canonico de BERT

Resumiendo lo que emergio de Model View:

```text
Capa 0-1:    Caos inicial. Atencion dispersa, leve sink hacia [CLS].
Capa 2-4:    Patrones LOCALES. Cabezas diagonales (look-self, look-right).
Capa 5:      Transicion.
Capa 6-7:    VALLE NO-OP. Casi todo va a [SEP].
Capa 8:      Concentracion en [CLS]. Consolidacion.
Capa 9-10:   DIVERSIDAD INFORMATIVA. Cabezas semanticas con vertices distintos.
Capa 11:     Mezcla zoo. Diagonales + concentradas. Lista para clasificar.
```

Este patron **no es exclusivo de BETO ni de NER fine-tuned**. Aparece en BERT base original y se ha replicado en muchisimos modelos tipo BERT (DistilBERT, mBERT, RoBERTa en menor medida). Es una **propiedad emergente** del pre-entrenamiento con MLM — no algo que el disenador puso a mano.

## Las grandes preguntas que abre esta seccion

1. **¿Por que las cabezas se "estacionan" en `[SEP]`?** Porque MLM solo necesita inferir tokens enmascarados — para muchos tokens la mejor inferencia es local, y la atencion a otros tokens lejanos no aporta nada. El gradiente "empuja" esos pesos hacia un token inocuo. `[SEP]` es la salida menos disruptiva.

2. **¿Estos patrones son universales o dependen de la tarea fine-tuned?** Mas o menos universales. El fine-tuning **ajusta** los patrones (especialmente en las ultimas 2-3 capas) pero el "valle no-op" en medias y el "sink CLS" en tempranas se preservan en casi todos los modelos.

3. **¿Que cabezas SON utiles?** Diagonales en capas tempranas (propagacion local), y diversas en capas finales (integracion semantica). Las medias son mayormente prescindibles.

## Lo que viene en la siguiente seccion

Hasta aqui veiamos los **pesos finales de atencion** (despues del softmax) pero no **como se calculan**. La siguiente seccion abre la cabeza individual y muestra los vectores Q y K dimension por dimension — la Neuron View. Ahi entendemos que el peso de atencion no es magia, es literalmente `softmax(Q · K^T)` y se puede inspeccionar a nivel atomico. Tambien comparamos `bert-base-uncased` (ingles) con `mBERT` (multilingüe) para ver si los patrones cambian o se mantienen.
