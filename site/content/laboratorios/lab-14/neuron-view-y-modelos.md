---
title: "Neuron View + Actividad 1 (mBERT vs uncased)"
weight: 30
math: true
---

Esta seccion es **la vista mas profunda** del Transformer en el lab. Hasta aqui veiamos los **pesos finales** de atencion (las matrices `seq_len × seq_len` que entrega el softmax). Ahora vamos a abrir una sola cabeza y mirar **como se calculan esos pesos** — los vectores Q y K dimension por dimension, su producto elemento-a-elemento, y el dot-product que termina alimentando el softmax.

Despues, la **Actividad 1** del lab pide cambiar la version de BERT y observar si los patrones se mantienen. Comparamos `bert-base-uncased` (ingles, el default de bertviz) contra `bert-base-multilingual-cased` (mBERT, entrenado en 104 idiomas). El hallazgo: **mBERT tiene cabezas con patrones sintacticos genuinos mas interpretables** que el BERT monolingüe.

## Por que cambia de modelo: el subclase parchada de bertviz

La Neuron View necesita **acceso interno** a los vectores Q y K dentro de cada cabeza — no solo a la matriz de atencion final. Por eso bertviz tiene una **subclase parchada** de `BertModel` que expone esos tensores via hooks. El notebook hace el switch *(parte 1, celdas 39-41)*:

```python
from bertviz.transformers_neuron_view import BertModel as VizBertModel, BertTokenizer as VizBertTokenizer
from bertviz.neuron_view import show

sentence_a = "Alexis scored against Brazil in the World Cup."
sentence_b = "Pinilla's shot struck the crossbar in that match."

nv_model_type = 'bert'
nv_model_version = 'bert-base-uncased'

do_lower_case = 'uncased' in nv_model_version
tokenizer = VizBertTokenizer.from_pretrained(nv_model_version, do_lower_case=do_lower_case)
model = VizBertModel.from_pretrained(nv_model_version)
call_html(view='neuron')
show(model, nv_model_type, tokenizer, sentence_a, sentence_b)
```

Notas importantes del switch:

- **Cambia de BETO a `bert-base-uncased`** (ingles). `bertviz.neuron_view` solo soporta ciertos modelos pre-parchados; BETO no esta entre ellos. Por eso las frases de ejemplo cambian a ingles.
- **Es el formato par de frases**: `[CLS] sentence_a [SEP] sentence_b [SEP]`. Esto permite filtrar despues entre atencion intra-frase A, intra-B, cruzada A→B, etc.
- **`do_lower_case=True`**: como el modelo es "uncased", el tokenizer convierte todo a minusculas antes de tokenizar.

## Las 5 columnas de la Neuron View

Al hacer click sobre un token query (digamos "scored") en la columna izquierda, aparecen **5 columnas** a la derecha:

| Columna | Que muestra | Tamano |
| --- | --- | --- |
| **Query q** | Cada fila es el vector Q de un token. La resaltada es la query elegida | seq_len × **64 dims** |
| **Key k** | Cada fila es el vector K de un token | seq_len × 64 dims |
| **q × k (elementwise)** | Producto elemento a elemento entre Q(query) y K(token_i) | seq_len × 64 dims |
| **q · k** | **Suma** de la fila anterior — un escalar por token (el score crudo antes del softmax) | seq_len cuadritos |
| **Softmax** | Aplica softmax sobre los escalares de q·k. La suma de la columna = 1.0 | seq_len cuadritos |

**Convencion de colores**: azul = positivo, naranja = negativo, saturacion = magnitud.

**El calculo paso a paso para una query** (digamos "scored"):

1. Tomas la fila Q de "scored" (resaltada en azul)
2. Tomas la fila K de cada token (incluyendo "scored")
3. Multiplicas dimension por dimension → fila de `q × k`
4. Sumas las 64 dimensiones → escalar en `q · k`
5. Repites para todos los tokens
6. Aplicas softmax sobre los escalares → columna `Softmax`

Cada cuadrito pequeno dentro de las barras horizontales **es UNA dimension** de los 64 del head. El **score final** es la suma. Si dominan los azules → score positivo. Si dominan los naranjas → negativo.

## bert-base-uncased: capa 0, 6, 9 con query "scored"

### Layer 0, Head 0 — ruido inicial

![Neuron view bert-uncased L0 H0 scored](/laboratorios/lab-14/neuron-view-uncased-l0-h0-scored.png)

Observaciones:

- Vectores Q y K **densos y mixtos** — muchas dimensiones azules y naranjas mezcladas. Esto es tipico: los embeddings de BERT no son interpretables dimension por dimension.
- Columna `q × k` se ve **casi igual entre filas**. No hay un token query claramente preferido.
- Columna `q · k` casi **uniforme** — no hay un cuadro mucho mas oscuro que destaque.
- Columna `Softmax` tambien casi uniforme — los pesos quedan repartidos casi igual entre los 11 tokens (~9% cada uno).

**Capa 0, Head 0 NO tiene un patron informativo todavia**. El modelo aun no ha procesado nada — esta mirando embeddings iniciales casi crudos. Es exactamente lo que esperabamos de las capas tempranas en Model View.

### Layer 6, Head 3 — el patron no-op confirmado dim a dim

![Neuron view bert-uncased L6 H3 scored](/laboratorios/lab-14/neuron-view-uncased-l6-h3-scored.png)

Aqui pasa algo mucho mas rico. En la columna `q · k`:

| Token | Color del cuadro | Significado |
| --- | --- | --- |
| [CLS], alexis, scored, against | Azul/oscuro | Score positivo o cercano a 0 |
| **brazil, in, the, world, cup** | **Naranja** | **Score NEGATIVO** |
| . | Oscuro | Cercano a 0 |
| **[SEP]** | **Azul intenso** | Score muy positivo |

**La gran leccion**: por primera vez ves valores **negativos** en `q · k`. No solo el modelo "prefiere" `[SEP]` — **ACTIVAMENTE rechaza** `brazil`, `in`, `the`, `world`, `cup`. El gradiente le ensenó que para procesar "scored", esos tokens son **anti-informativos** en esta cabeza.

Despues del softmax, los scores negativos se exponencian (`exp(num_neg)` es muy pequeno) y quedan **cerca de 0** en `Softmax`. `[SEP]` con su score positivo grande absorbe casi toda la masa.

> **La atencion no es solo "fijarse en algo"** — tambien es **"ignorar activamente otras cosas"**. Esto es lo que las visualizaciones de alto nivel (head_view, model_view) ocultaban. La Neuron View deja ver el mecanismo interno completo.

### Layer 9, Head 7 — sigue siendo no-op

![Neuron view bert-uncased L9 H7 scored](/laboratorios/lab-14/neuron-view-uncased-l9-h7-scored.png)

- `[SEP]` con cuadro azul intenso (~80-90% de la atencion)
- Todos los demas tokens con cuadros oscuros
- Excepcion: `scored` mismo tiene cuadrito levemente mas claro (autoatencion minima)

En `bert-base-uncased`, **la capa 9 sigue mostrando no-op para esta cabeza**. No emerge un patron sintactico claro con esta query.

### La idea profunda

Q y K son proyecciones lineales del embedding del token: `Q = W_q · embedding`, `K = W_k · embedding`. **Las matrices `W_q` y `W_k` se aprenden durante pre-entrenamiento.** El modelo aprende **que dimensiones de Q y K rotar** para producir las afinidades deseadas.

Esto explica por que los Transformers son tan **interpretables matematicamente** pero tan **dificiles de interpretar semanticamente**. Cada decision se hace sobre 64 dimensiones que no corresponden a conceptos legibles para humanos — son combinaciones lineales aprendidas.

## Actividad 1 — cambiar de modelo

El enunciado *(parte 1, celdas 42-43)* pide descomentar el codigo y elegir una version distinta de BERT, "preferentemente no `large` por limitaciones de memoria de Colab". Una opcion valida y rica es `bert-base-multilingual-cased` (mBERT) — mismo tamano de bert-base pero entrenado en 104 idiomas.

```python
nv_model_type = 'bert'
nv_model_version = 'bert-base-multilingual-cased'

do_lower_case = 'uncased' in nv_model_version
tokenizer = VizBertTokenizer.from_pretrained(nv_model_version, do_lower_case=do_lower_case)
model = VizBertModel.from_pretrained(nv_model_version)
call_html(view='neuron')
show(model, nv_model_type, tokenizer, sentence_a, sentence_b)
```

### Diferencias visibles desde la tokenizacion

mBERT es **cased**, asi que preserva mayusculas. La tokenizacion cambia respecto a `bert-base-uncased`:

| Modelo | Tokens (primera frase) |
| --- | --- |
| `bert-base-uncased` | `[CLS] alexis scored against brazil in the world cup . [SEP] pin ##illa ...` |
| `bert-base-multilingual-cased` | `[CLS] Alexis scored against Brazil in the World Cup . [SEP] Pin ##illa ...` |

La capitalizacion es senal fuerte de entidades nombradas — mBERT podria rendir mejor en NER por esa pista visual. Los WordPieces son similares (`Pin` + `##illa`, `cross` + `##bar`), pero los embeddings de tokens capitalizados vs minusculos son **distintos** porque viven en posiciones distintas del vocab.

### mBERT en Layer 0, Head 0

![Neuron view mBERT L0 H0](/laboratorios/lab-14/neuron-view-mbert-l0-h0.png)

Como esperabamos: **atencion dispersa**. Las capas tempranas no muestran patrones claros, **independiente del modelo**. Esto confirma que la "etapa de caos inicial" es **universal en arquitecturas BERT**.

Detalle interesante: aunque la capa 0 es dispersa, ya hay una pequena tendencia a que `Alexis` (izq) se conecte con `Alexis` (der) — **autoatencion fuerte** sobre el nombre. El modelo ya muestra leve preferencia por mirarse a si mismo cuando procesa nombres propios.

### mBERT en Layer 6, Head 3 con query "scored" — patron sintactico real

![Neuron view mBERT L6 H3 scored](/laboratorios/lab-14/neuron-view-mbert-l6-h3-scored.png)

Aqui se quiebra el patron de bert-uncased. Mira la columna `Softmax`:

| Token | Color | Interpretacion |
| --- | --- | --- |
| **against** | **Azul intenso** | **¡Atencion fuerte!** "scored" → "against" |
| scored, in, the, Brazil | Azul medio | Atencion repartida |
| [CLS], Alexis | Oscuro | Poca atencion |
| **Pin, ##illa, ', s, shot, struck, the, cross, ##bar, in, that** | **NARANJA** | **Rechazo activo de la frase B** |

**Dos hallazgos:**

1. "scored" → "against" es **sintactico**. El verbo "scored" busca su complemento mediante la preposicion que lo introduce. Es exactamente la relacion que un parser sintactico esperaria. **Bert-uncased en la misma capa/cabeza iba 100% a `[SEP]`** — mBERT aqui hace trabajo informativo real.

2. **Todos los tokens de la frase B son naranjas (rechazados)**. Esta cabeza aprendio a **mantener separacion entre frases** — util para tareas que distinguen contextos (QA, NLI, similitud).

### mBERT en Layer 9, Head 7 con query "scored"

![Neuron view mBERT L9 H7 scored](/laboratorios/lab-14/neuron-view-mbert-l9-h7-scored.png)

En lugar de irse 100% a `[SEP]` como en bert-uncased, ahora hay **reparto interesante**:

- `[CLS]` alto
- `Alexis`, `scored` medios
- `against` medio-bajo
- **`in`, `the`, `World`, `Cup`, `.`** → **naranja** (rechazo)
- `[SEP]` medio

No hay un unico "ganador no-op". mBERT en esta cabeza **integra informacion** de `[CLS]` + sujeto + verbo + `[SEP]`, y rechaza activamente "palabras de relleno". **Es lo opuesto al patron no-op.**

### mBERT en Layer 9, Head 7 con query "Alexis"

![Neuron view mBERT L9 H7 Alexis](/laboratorios/lab-14/neuron-view-mbert-l9-h7-alexis.png)

- `[CLS]`, `scored`, `[SEP]` → azul medio
- `Alexis` mismo → bajo (poca auto-atencion)
- `in`, `the`, `World`, `Cup` → **naranja** (rechazados)

Cuando "Alexis" busca contexto, **prefiere "scored" (su verbo)** sobre modificadores. Tiene sentido — para saber que hace Alexis, hay que mirar el verbo.

### mBERT en Layer 6, Head 3 con query "Alexis" — la joya

![Neuron view mBERT L6 H3 Alexis - sujeto verbo](/laboratorios/lab-14/neuron-view-mbert-l6-h3-alexis.png)

**Esta es la mejor evidencia de patron sintactico en todo el lab.**

| Token | Atencion |
| --- | --- |
| **Alexis** (auto) | **Azul intenso** ← se atiende a si mismo fuerte |
| **scored** | **Azul intenso** ← **¡SUJETO → VERBO!** |
| against, Brazil, in | Medio |
| the, World, . | Bajo / negativo |
| **Cup** | **Naranja** (rechazado) |
| [CLS], [SEP] | Medio |

**Patron sujeto-verbo nitido**: "Alexis" mira fuertemente a "scored". Esta cabeza aprendio que para entender un sujeto, hay que verlo en relacion con su predicado.

## Conclusion comparativa

| Aspecto | `bert-base-uncased` (ingles) | `bert-base-multilingual-cased` (mBERT) |
| --- | --- | --- |
| Capas 6-7 | Patron no-op masivo hacia [SEP] | Cabezas con relaciones sintacticas |
| Capa 9 | Mayoria va a [SEP] | Reparto informativo |
| Distincion casing | No (`alexis`) | Si (`Alexis`) → pistas para NER |
| Separacion A vs B | No tan clara | Rechazo activo de la otra frase |

### Por que mBERT tiene patrones mas "limpios"

**Hipotesis sostenida en literatura**: porque mBERT esta entrenado en 104 idiomas con vocabularios y gramaticas distintas, **NO puede depender** de heuristicas tan simples como "atiende al token siguiente" o "atiende a `[SEP]`". Tuvo que aprender **patrones lingüisticos mas universales** (sujeto-verbo, sustantivo-modificador, etc.). Eso lo obliga a tener cabezas mas informativas y menos no-op.

> **Estudios formales** (Pires et al. 2019, *"How multilingual is Multilingual BERT?"*) confirman que mBERT aprende **representaciones que transfieren entre lenguajes** porque captura abstracciones sintacticas mas profundas. Lo que se ve aqui en la Neuron View son **esas abstracciones en accion**.

## Implicaciones

1. **No todos los BERTs son iguales en interpretabilidad.** Un modelo multilingüe puede tener cabezas mas legibles que un modelo monolingüe, incluso si ambos tienen el mismo tamano y arquitectura.

2. **El pre-entrenamiento moldea fuertemente los patrones de atencion.** Cambiar el corpus (ingles vs 104 idiomas) cambia cualitativamente los patrones que emergen, sin tocar la arquitectura.

3. **Para tareas downstream donde la interpretabilidad importe** (auditing, fairness, explicabilidad medica), modelos multilingüe pueden ser preferibles a monolingüe — aunque por performance pura el monolingüe gane.

## Cierre de la seccion

Con esta seccion termina el contenido didactico de **inspeccion de atenciones**. Ya tienes el panorama completo:

1. **Tokenizacion y NER** (BETO + WordPiece + displacy) → seccion 1
2. **head_view + model_view** (los 144 patrones de cabezas) → seccion 2
3. **Neuron View y comparacion entre modelos** (Q/K + mBERT vs uncased) → esta seccion

La siguiente y ultima seccion del lab Parte 1 es **conceptual** — preguntas teoricas sobre el decoder de un Transformer encoder-decoder (cross-attention, masking causal, positional encoding).
