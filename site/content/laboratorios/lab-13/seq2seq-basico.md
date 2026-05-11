---
title: "Parte 1 - Seq2Seq basico"
weight: 10
math: true
---

Esta primera parte del laboratorio implementa un **Seq2Seq** estandar — encoder-decoder sin attention — sobre el dataset **SCAN**, donde la tarea es traducir comandos en ingles a secuencias de acciones simbolicas para un robot. La idea es construir el modelo mas simple posible (`embedding` + LSTM bidireccional en el encoder, `LSTMCell` autoregresivo en el decoder) y observar concretamente la limitacion que motiva la Parte 2: el cuello de botella del **context vector** unico de tamano fijo.

El modelo final entrena por **300 epochs** sobre la variante `tasks_simple` de SCAN y satura en torno a **0.91** de accuracy token-level (con padding incluido). Que no llegue a 1.0 a pesar de la simplicidad del dataset es la evidencia empirica del bottleneck.

## Setup del problema (translation)

El notebook usa el dataset **SCAN** *(notebook 1, cell 15)*, una coleccion publica que mapea comandos cortos en ingles a secuencias de **acciones discretas** que un robot deberia ejecutar. La tabla del notebook lista ejemplos como `jump → JUMP` o `jump around right → RTURN JUMP RTURN JUMP RTURN JUMP RTURN JUMP`, pero al inspeccionar el archivo real *(notebook 1, cell 18: `!head tasks_test_simple.txt`)* aparece que **el formato de tokens del split `simple` usa otra convencion** que la del markdown:

| Markdown del notebook | Datos reales en `tasks_*_simple.txt` |
| --- | --- |
| `JUMP` | `I_JUMP` |
| `LTURN` | `I_TURN_LEFT` |
| `RTURN` | `I_TURN_RIGHT` |
| `WALK` | `I_WALK` |
| `LOOK` | `I_LOOK` |
| `RUN` | `I_RUN` |

Es solo una diferencia de naming convention — al modelo no le importa el string concreto, solo le importa que cada token sea un indice estable del vocabulario destino — pero conviene saberlo cuando se inspeccionan outputs cualitativos.

Conceptualmente esto es **el mismo tipo de tarea que la traduccion entre idiomas** que se estudia en la teoria: un mapeo `secuencia → secuencia` donde la salida puede tener largo distinto de la entrada y el orden importa. Lo que cambia es solo el "idioma destino" — en vez de frances, son tokens del vocabulario `{i_jump, i_turn_left, i_turn_right, i_walk, i_run, i_look}`. La ventaja didactica de SCAN sobre un corpus real (En→Fr) es que el vocabulario es minusculo (~14 tokens fuente, 6 tokens destino) y las dependencias entre input y output son **transparentes** ("around right" siempre genera el mismo patron de cuatro `I_TURN_RIGHT I_X`), asi que se puede ver con claridad cuando el modelo aprende a generalizar y cuando solo memoriza.

### Carga, tokenizacion y batching

El bloque de carga *(notebook 1, cells 20-21)* hace tres cosas que vale la pena descomponer porque tienen consecuencias para todo lo demas:

1. **Tokenizer minimalista**: `text.strip().lower().split()`. Pasa todo a minusculas y corta por espacios. Como los tokens destino van todos en mayusculas pero los fuente en minusculas, el `.lower()` no causa colisiones — `i_jump` y `jump` viven en vocabularios distintos (cada `Field` arma el suyo).
2. **`Field` con `unk_token=None`**: el vocabulario es cerrado, no aparecen palabras desconocidas entre train y test. Por defecto cada `Field` agrega `<pad>` con indice `0` (importante para la siguiente seccion).
3. **`BucketIterator`** con `batch_sizes=(128, 128)` y `sort_key=lambda x: len(x.target)`: agrupa ejemplos por largo del target antes de armar batches. Esto **minimiza el padding** dentro de cada batch — frases cortas se agrupan con cortas, largas con largas. Sin esto, batches mixtos rellenarian las frases cortas hasta el largo de la mas larga del batch y desperdiciarian computo.

El padding en este notebook es **dinamico por batch**: cada batch se rellena hasta el largo de la frase mas larga de **ese** batch, no hasta un maximo global. El `BucketIterator` reduce el costo de ese padding agrupando inteligentemente.

### El truco del token `<SOS>`

El notebook **no agrega un token `<SOS>` (start-of-sentence) al vocabulario** via `Field(init_token='<sos>')` como seria estandar. En su lugar usa un hack *(notebook 1, cell 28)*: pasa `start_idx = len(TARGET.vocab)` al decoder, es decir, **el indice siguiente al ultimo del vocabulario**, y le pide al `nn.Embedding` del decoder que tenga `dst_vocab_size = len(TARGET.vocab) + 1` filas. El `<SOS>` queda como una fila extra de la tabla de embedding del decoder, sin un string asociado en el `Vocab` de torchtext. Funciona igual, pero es importante notar la decision para no confundirse leyendo la celda 28.

No hay token `<EOS>`. La generacion del decoder se controla unicamente por el parametro `max_output_length`, que en training se fija al largo del target ground truth del batch *(notebook 1, cell 23)*.

## Arquitectura encoder

La clase `EncoderModule` *(notebook 1, cell 12)* sigue el patron clasico de RNN-encoder:

```python
class EncoderModule(nn.Module):
    def __init__(self, embedding_size, hidden_size, source_vocab_size):
        super().__init__()
        self.embeddings_table = nn.Embedding(source_vocab_size,
                                             embedding_size,
                                             padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=hidden_size,
                            bidirectional=True,
                            batch_first=True)

    def forward(self, src_sentences):
        embedded = self.embeddings_table(src_sentences)
        _, hidden_state = self.lstm(embedded)
        return hidden_state
```

Dos detalles tecnicos que conviene tener claros:

**`padding_idx=0`**: hace que el gradiente del embedding del token `<pad>` sea cero. Sin esa flag, el modelo "aprenderia" un vector con significado para el padding, lo que es ruido puro porque el padding no tiene contenido semantico. Es la convencion estandar en NLP y el motivo por el que el `Field` reserva el indice `0` para `<pad>`.

**Shapes a lo largo del forward**:

| Tensor | Shape | Significado |
| --- | --- | --- |
| `src_sentences` | `(B, L_src)` | Indices enteros del vocabulario fuente, con padding dinamico |
| `embedded` | `(B, L_src, 100)` | Cada indice reemplazado por su vector denso de 100 floats |
| `outputs` (descartado) | `(B, L_src, 2·150)` | Hidden states de cada paso, concatenando forward y backward |
| `hidden_state = (h_n, c_n)` | `(2, B, 150)` cada uno | `h_n[0]` = forward; `h_n[1]` = backward |

La capa **`nn.Embedding` es una tabla lookup**: para cada entero del input devuelve la fila correspondiente de una matriz `(source_vocab_size, embedding_size)`. Los pesos de esa matriz se aprenden durante el entrenamiento junto con todo lo demas — no es una operacion matematica complicada, es indexing.

La **`nn.LSTM` bidireccional** instancia internamente dos LSTMs (una procesa la secuencia de izquierda a derecha, la otra de derecha a izquierda) y concatena sus outputs. Por eso el ultimo hidden state tiene dimension 2 en la primera coordenada: una entrada por direccion. Eso obliga a la transformacion que veremos abajo para conectar al decoder.

El comentario del notebook lo describe como "pretty much the same encoder as in the first part (sentiment analysis)". La diferencia con sentiment analysis es **que se hace con la salida**: en clasificacion solo importa el ultimo hidden state como feature global; aca ese mismo ultimo hidden state cumple el rol de **`context vector`** para condicionar al decoder.

Formalmente, dado el input $x_1, x_2, \ldots, x_T$:

$$h_t = \text{LSTM}(\text{Embedding}(x_t),\; h_{t-1})$$

y al final del barrido obtenemos $h_T$, que codifica — en teoria — toda la oracion fuente en un solo vector de dimension fija. Este $h_T$ es **toda la informacion que el decoder vera del input**: ni los hidden states intermedios $h_1, \ldots, h_{T-1}$ ni los embeddings originales se le pasan. La conexion encoder-decoder es un unico vector.

Esto es exactamente la formulacion de **Sutskever et al. 2014** — el paper canonico de Seq2Seq — donde tambien se usa el ultimo hidden state del LSTM encoder como vector $C$.

## Arquitectura decoder

La clase `DecoderModule` *(notebook 1, cell 14)* es donde aparece la dinamica autoregresiva. El notebook lo explica como pseudocodigo:

```python
predicted_words = []
for word_idx in range(max_output_len):
    prediction = model.predict()
    predicted_words.append(prediction)
return predicted_words
```

Tres decisiones de diseno importantes que el notebook documenta explicitamente:

1. **Conditioning sobre el encoder.** El **ultimo hidden state del encoder se usa como hidden state inicial del decoder**. Es el unico canal por el que la oracion fuente influye en la generacion. Si el encoder no logro comprimir bien la informacion en $h_T$, el decoder no tiene de donde recuperarla.
2. **Generacion autoregresiva token por token con `LSTMCell`** (no `LSTM`).
3. **Token de inicio `<SOS>`.** La primera iteracion no tiene un "token previo predicho", asi que se usa el `<SOS>` (implementado con el hack del indice extra) como input inicial.

### `LSTMCell` vs `LSTM` — por que el decoder usa cell

PyTorch tiene dos APIs distintas para LSTM y la diferencia es crucial entender por que el decoder usa una y el encoder la otra:

| `nn.LSTM` | `nn.LSTMCell` |
| --- | --- |
| Procesa **toda la secuencia** de una vez | Procesa **un solo timestep** |
| Maneja el loop internamente (C++/CUDA, rapido) | El loop lo escribe el usuario en Python |
| Util cuando la secuencia ya existe (encoder) | Util cuando se va **generando** la secuencia (decoder) |

El decoder **no tiene la secuencia destino disponible** durante la generacion — la esta construyendo paso a paso, y la prediccion del paso $t$ es el input del paso $t+1$. Por eso necesita un loop explicito en Python con `LSTMCell` que reciba un solo vector por vez.

### Greedy decoding

En cada paso, despues de calcular los logits sobre el vocabulario, el decoder elige **el indice con mayor score**:

```python
P_t = self.h2o(state[0])              # logits (B, dst_vocab_size)
_, max_indices = P_t.max(dim=1)       # indice ganador por muestra
y_t = self.embeddings_table(max_indices)  # input del proximo paso
```

Esto se llama **greedy decoding** (o `argmax decoding`). Es la estrategia mas simple — siempre toma la opcion localmente optima. Alternativas mas sofisticadas como **beam search** mantienen las top-k secuencias parciales y eligen al final la de mejor probabilidad global. Para SCAN, donde el mapeo input→output es bastante deterministico, greedy suele bastar.

### Logits, no probabilidades

El comentario del notebook llama `P_t` al output de la capa `h2o`, sugiriendo "probability". En realidad **son logits** — scores crudos que pueden ser cualquier float. No se aplica `softmax`. Esto es deliberado por dos razones:

- **Para hacer `argmax` no hace falta softmax**: como `softmax` es monotonica, el indice con mayor score es el mismo antes y despues. Aplicarla seria computo desperdiciado.
- **`F.cross_entropy` espera logits**, no probabilidades. Internamente combina `log_softmax + nll_loss` con mayor estabilidad numerica que aplicar softmax explicito.

Solo se aplicaria softmax si se quisiera samplear (en lugar de argmax) o inspeccionar las probabilidades para visualizacion.

La forma matematica del paso $t$ del decoder, alineada con la teoria:

$$s_t = \text{LSTMCell}(\text{Embedding}(\hat{y}_{t-1}),\; s_{t-1})$$
$$\hat{y}_t = \arg\max \; \text{softmax}(W_{out}\, s_t)$$

con $s_0 = h_T$ (el context vector del encoder, transformado como se explica en la siguiente seccion). Notar que $s_t$ no depende de los hidden states intermedios del encoder — esa es justamente la limitacion que attention va a resolver en la Parte 2.

### El detalle de no usar teacher forcing en parte 1

Mirando el loop del decoder *(notebook 1, cell 14)* hay un detalle clave: **en cada paso el input es siempre la prediccion del paso anterior** (`y_t = embeddings_table(max_indices)`), incluso durante training. **No hay teacher forcing.**

```python
for i in range(max_output_length):
    state = self.lstm_cell(y_t, state)
    P_t = self.h2o(state[0])
    out.append(P_t)
    _, max_indices = P_t.max(dim=1)
    y_t = self.embeddings_table(max_indices)   # ← siempre la prediccion, nunca el ground truth
```

Esto hace que el entrenamiento sea **mas lento y menos estable** que con teacher forcing: si el decoder se equivoca en el paso 3, el paso 4 recibe un input erroneo y el error se propaga. Es parte de la razon por la que el modelo necesita **300 epochs** para saturar y por la que la accuracy se queda en 0.91 en lugar de subir mas alto. La **Parte 3 del laboratorio** retoma exactamente este tema y compara entrenamiento con y sin teacher forcing.

## El puente encoder-decoder

La clase `SeqToSeq` *(notebook 1, cell 27)* junta encoder y decoder y resuelve un problema no trivial: el encoder bidireccional devuelve un hidden state con shape `(2, B, hidden_size)` (una entrada por direccion), pero el `LSTMCell` del decoder espera `(B, hidden_size)`. Hay que combinar las dos direcciones en una sola dimension de features.

El metodo `reshape_enc_states` hace esa transformacion para **`h` y para `c`**:

```python
enc_hidden_state.permute(1, 0, 2).reshape(-1, 2*self.hidden_size)
# (2, B, 150)  →  (B, 2, 150)  →  (B, 300)
```

`permute(1, 0, 2)` mueve la dimension de batch al frente. `reshape(-1, 2*hidden_size)` concatena los 150 floats de la direccion forward con los 150 de la backward, dando un vector de 300 floats por ejemplo. Es por eso que el decoder se instancia con `hidden_size = 2*150 = 300` en la celda 27 — el `LSTMCell` necesita ser dimensionalmente compatible con el estado concatenado del encoder.

Hay decisiones alternativas posibles (sumar ambas direcciones, usar solo la forward, etc). El notebook elige **concatenar** que preserva toda la informacion del encoder a costo de duplicar la dimension del decoder.

## Loop de entrenamiento

Las utilidades de entrenamiento estan en la seccion `Train Utils` *(notebook 1, cells 23-25)*.

### Loss: cross-entropy sobre toda la secuencia aplanada

```python
loss = F.cross_entropy(y_pred.flatten(end_dim=-2),
                       y_gt.view(-1))
```

`F.cross_entropy` espera logits `(N, num_classes)` y targets `(N,)` enteros. Pero nuestros tensores son:

- `y_pred`: `(B, L, dst_vocab_size)` — 3D
- `y_gt`: `(B, L)` — 2D

El truco es **aplanar**:

- `y_pred.flatten(end_dim=-2)`: `(B, L, V) → (B·L, V)`. Aplana todas las dimensiones hasta la anteultima, dejando intacta la de vocabulario.
- `y_gt.view(-1)`: `(B, L) → (B·L,)`. Aplana en un solo vector de enteros.

**Interpretacion**: tratamos cada par `(muestra, posicion temporal)` como **un problema de clasificacion independiente** de `dst_vocab_size` clases. Con batches de 128 y secuencias de hasta 30 tokens, son hasta `128·30 = 3840` problemas de clasificacion fundidos en una sola loss. Esto es estandar en seq2seq.

Formalmente:

$$\mathcal{L} = -\frac{1}{L} \sum_{t=1}^{L} \log P(y_t \mid y_{<t}, x; \theta)$$

que equivale exactamente al objetivo que la teoria escribe como maximizar $\frac{1}{|TS|} \sum \log P(y_i \mid x_i; \theta)$.

### Padding contamina la loss y la accuracy

`F.cross_entropy` se calcula sobre **todos** los tokens del target — incluyendo los `<pad>`. Lo correcto en NLP seria pasar `ignore_index=0` para que el padding no aporte al gradiente. Este notebook no lo hace, lo que tiene dos consecuencias:

1. El modelo aprende a predecir padding (es trivial, siempre es `<pad>` despues del fin de secuencia), lo que **infla la accuracy reportada**.
2. La loss tiene un piso artificial porque incluye terminos triviales del padding.

Ambas observaciones afectan los numeros reportados pero **no la capacidad del modelo** para aprender la tarea real.

### Los tres pasos canonicos de backprop

```python
optimizer.zero_grad()   # 1. resetea gradientes acumulados
loss.backward()         # 2. calcula gradientes (autograd)
optimizer.step()        # 3. actualiza pesos
```

**`zero_grad()` es critico**: PyTorch **acumula gradientes por default** (decision de diseno util para gradient accumulation con batches grandes). Sin reset, la segunda iteracion sumaria los gradientes de la primera a los nuevos, la tercera sumaria todo lo anterior, y el modelo explotaria. **No confundir gradientes con pesos** — `zero_grad` toca solo gradientes, los pesos se modifican con `step()`.

### `torch.no_grad()` en eval

La funcion `eval_one_epoch` *(notebook 1, cell 24)* envuelve todo en `with torch.no_grad()`. Esto **desactiva la construccion del grafo computacional de autograd** — no se guardan las activaciones intermedias necesarias para `backward()`. El efecto es doble: ahorra memoria (potencialmente gigabytes) y acelera el forward.

Es ortogonal a `model.eval()`: este ultimo solo cambia el comportamiento de capas como Dropout y BatchNorm. Para inferencia pura se usan **las dos juntas**.

### Accuracy: token-level con padding

```python
accuracy = (y_pred.argmax(dim=2) == y_gt).float().mean()
```

Compara argmax contra ground truth posicion por posicion y promedia. Como discutimos arriba, esto **incluye los tokens de padding**, lo que da una metrica optimista. Una accuracy "real" en seq2seq seria **sentence-level** (toda la secuencia exactamente correcta) — mas exigente y mas alineada con lo que SCAN suele reportar en la literatura.

## Hiperparametros y configuracion

Confirmados al ejecutar el notebook *(cells 21, 28, 30, 31)*:

| Hiperparametro | Valor |
| --- | --- |
| `embedding_size` (encoder y decoder) | `100` |
| `hidden_size` (por direccion) | `150` |
| `hidden_size` efectivo decoder | `300` (concatenacion bidireccional) |
| `batch_size` (train y test) | `128` |
| `learning_rate` | `0.001` |
| Optimizador | `Adam` (default beta1, beta2) |
| `n_epochs` | `300` |
| Device | `cuda` (GPU T4 en Colab) |

## Resultados modelo base

**Parametros entrenables del modelo**: `789,809`. Aproximadamente:

- LSTM bidireccional del encoder: ~360K parametros (4 puertas internas × `(100·150 + 150·150) · 2` direcciones).
- `LSTMCell` del decoder: ~480K (4 puertas × `(100·300 + 300·300)`).
- Embeddings + capa `h2o`: ~10K (los vocabularios son pequenos).

### Curva de accuracy en eval

![Eval accuracy Seq2Seq base sobre SCAN](/laboratorios/lab-13/eval-acc-seq2seq-base.png)

`plt.plot(history['eval_acc'])` muestra una curva monotonamente creciente con la forma tipica de aprendizaje por gradiente:

| Epoch | Eval accuracy aprox |
| --- | --- |
| 0 (inicio) | ~0.15 |
| 50 | ~0.85 |
| 100 | ~0.88 |
| 150 | ~0.89 |
| 200 | ~0.90 |
| 300 (final) | **~0.91** |

La subida es **muy rapida** las primeras 50 epochs (de 0.15 a 0.85 — el modelo aprende patrones obvios y a predecir padding). Despues entra en un **regimen de refinamiento gradual** y satura cerca de **0.91 token-level**. Aparecen un par de "dips" momentaneos cerca de epochs 120 y 180 — son artefactos del optimizador (un step agresivo de Adam que el modelo recupera rapido). No son overfitting (no hay bajada sostenida).

### Lectura cualitativa del numero

`0.91 token-level con padding incluido` parece alto, pero es enganoso. Algunas consideraciones:

- **El padding infla el numero**. El modelo aprende muy temprano a predecir `<pad>` cuando ya termino la secuencia real, y eso cuenta como acierto. Si una traduccion correcta usa 8 de los 30 tokens del batch, los 22 paddings restantes son aciertos "gratis".
- **Si traducimos a sentence-level**: para una secuencia de 20 tokens, la probabilidad de acertarla **completa** (asumiendo independencia, que es una simplificacion) seria $0.91^{20} \approx 0.15$. El modelo probablemente acierta **menos de la mitad** de las traducciones completas, aunque acierte casi todos los tokens individualmente.
- **SCAN suele reportarse sentence-level en la literatura** (Lake & Baroni 2018). Este notebook reporta token-level por simplicidad.

## Limitacion: el bottleneck

La arquitectura recien descrita tiene una limitacion fundamental que la teoria de la clase 13 enuncia con claridad: **toda la informacion del input tiene que viajar por un unico vector $h_T$ de dimension fija** (en este modelo, 300 floats). Para oraciones cortas no es problema — `jump` cabe sobrado en 300 floats. Pero a medida que la entrada crece, el encoder tiene que comprimir cada vez mas informacion en el mismo numero de coordenadas, y el decoder no tiene forma de "volver a mirar" partes especificas del input cuando esta generando un token particular.

**El plateau en 0.91** que se observa empiricamente es exactamente esa limitacion manifestada como cota superior. Aumentar el numero de epochs mas alla de 300 no mueve el numero significativamente — el modelo ya no esta limitado por el optimizador, esta limitado por la **capacidad de la representacion**.

El ejemplo canonico de la teoria deja la intuicion clara:

```text
Encode: Economic growth has slowed down in recent years.
Decode: La croissance economique a ralenti ces dernieres annees.
```

Cuando el decoder esta generando `croissance` querria mirar a `growth`. Cuando genera `annees` querria mirar a `years`. Pero con un **`context vector`** unico no hay forma de hacer ese alineamiento dinamico — el decoder solo tiene acceso a la representacion comprimida $h_T$ de la oracion completa.

En SCAN el sintoma se observa de forma analoga: comandos compuestos como `jump around right twice` requieren generar secuencias largas de acciones donde cada paso del decoder deberia "atender" a un fragmento distinto del input (`jump` para los `I_JUMP`, `right` para los `I_TURN_RIGHT`, `twice` para repetir el bloque). Un solo $h_T$ tiene que codificar **todo eso en un solo vector** y ademas distribuirlo de forma utilizable a lo largo de todos los pasos del decoder.

La Parte 2 ataca exactamente este cuello de botella reemplazando el context vector fijo por un **context vector adaptativo $c_t$** que en cada paso del decoder es un promedio ponderado de **todos** los hidden states del encoder $h_1, h_2, \ldots, h_T$. Los pesos $\alpha_{t,i}$ los aprende el modelo end-to-end y forman, despues del entrenamiento, un **alineamiento suave** entre input y output que se puede visualizar como un heatmap. Pero ese es el tema de la siguiente pagina.

## Imperfecciones del notebook a tener en cuenta

Cuatro decisiones de simplificacion que el notebook toma — ninguna invalida el aprendizaje, pero conviene listarlas para entender los limites de los resultados reportados:

1. **No usa `ignore_index=0` en `cross_entropy`** → el padding entra a la loss y a la accuracy, contaminando ambas metricas.
2. **Accuracy token-level** (no sentence-level), agravado por incluir padding → el `0.91` reportado sobreestima el desempeno real de traduccion.
3. **No usa teacher forcing en training** → entrenamiento mas lento e inestable. La Parte 3 retoma este punto.
4. **No hay token `<EOS>`** → la generacion se controla por `max_output_length` fijado al largo del ground truth en training. En inferencia real habria que elegir un maximo arbitrario o entrenar un `<EOS>`.
