---
title: "Parte 1 - Seq2Seq basico"
weight: 10
math: true
---

Esta primera parte del laboratorio implementa un **Seq2Seq** estandar — encoder-decoder sin attention — sobre el dataset **SCAN**, donde la tarea es traducir comandos en ingles a secuencias de acciones simbolicas para un robot. La idea es construir el modelo mas simple posible (`embedding` + LSTM en el encoder, LSTM cell autoregresivo en el decoder) y observar concretamente la limitacion que motiva la Parte 2: el cuello de botella del **context vector** unico de tamano fijo.

## Setup del problema (translation)

El notebook usa el dataset **SCAN** *(notebook 1, cell 15)*, una coleccion publica que mapea comandos cortos en ingles a secuencias de **acciones discretas** que un robot deberia ejecutar. Algunos ejemplos del notebook:

| Input (ingles) | Output (acciones) |
| --- | --- |
| `jump` | `JUMP` |
| `jump left` | `LTURN JUMP` |
| `jump around right` | `RTURN JUMP RTURN JUMP RTURN JUMP RTURN JUMP` |

Conceptualmente esto es **el mismo tipo de tarea que la traduccion entre idiomas** que se estudia en la teoria: un mapeo `secuencia → secuencia` donde la salida puede tener largo distinto de la entrada y el orden importa. Lo que cambia es solo el "idioma destino" — en vez de frances, son tokens del vocabulario `{JUMP, LTURN, RTURN, WALK, RUN, ...}`. La ventaja didactica de SCAN sobre un corpus real (En→Fr) es que el vocabulario es minusculo y las dependencias entre input y output son **transparentes** ("around right" siempre genera el mismo patron de cuatro turns), asi que se puede ver con claridad cuando el modelo aprende a generalizar y cuando solo memoriza.

El objetivo concreto de esta parte 1: entrenar un Seq2Seq que aprenda la regla `comando → secuencia de acciones` directamente desde pares de ejemplos, sin reglas escritas a mano. Tamanos exactos del split y del vocabulario `[a confirmar al ejecutar parte 1]`.

## Arquitectura encoder

La clase `Encoder` *(notebook 1, cell 11-12)* sigue el patron clasico de RNN-encoder:

1. Una capa `nn.Embedding` que transforma cada token de entrada (un entero, indice del vocabulario fuente) en un vector denso de dimension `embed_dim`.
2. Una **LSTM bidireccional** que procesa la secuencia de embeddings token por token, produciendo un hidden state $h_t$ en cada paso.

El comentario del notebook lo describe como "pretty much the same encoder as in the first part (sentiment analysis)" — el patron `embedding + biLSTM` es el caballo de batalla de los modelos secuenciales clasicos. La diferencia con sentiment analysis es **que se hace con la salida**: en clasificacion solo importa el ultimo hidden state como feature global; aca ese mismo ultimo hidden state cumple el rol de **`context vector`** para condicionar al decoder.

Formalmente, dado el input $x_1, x_2, \ldots, x_T$:

$$h_t = \text{LSTM}(\text{Embedding}(x_t),\; h_{t-1})$$

y al final del barrido obtenemos $h_T$, que codifica — en teoria — toda la oracion fuente en un solo vector de dimension fija. Este $h_T$ es **toda la informacion que el decoder vera del input**: ni los hidden states intermedios $h_1, \ldots, h_{T-1}$ ni los embeddings originales se le pasan. La conexion encoder-decoder es un unico vector.

Esto es exactamente la formulacion de **Sutskever et al. 2014** — el paper canonico de Seq2Seq — donde tambien se usa el ultimo hidden state del LSTM encoder como vector $C$.

## Arquitectura decoder

La clase `Decoder` *(notebook 1, cell 13-14)* es donde aparece la dinamica autoregresiva. El notebook lo explica como pseudocodigo:

```python
predicted_words = []
for word_idx in range(max_output_len):
    prediction = model.predict()
    predicted_words.append(prediction)
return predicted_words
```

Tres decisiones de diseno importantes que el notebook documenta explicitamente:

1. **Conditioning sobre el encoder.** El **ultimo hidden state del encoder se usa como hidden state inicial del decoder**. Es el unico canal por el que la oracion fuente influye en la generacion. Si el encoder no logro comprimir bien la informacion en $h_T$, el decoder no tiene de donde recuperarla.
2. **Generacion autoregresiva token por token.** El decoder usa una **LSTM cell** (no una LSTM completa que procesa toda la secuencia de una vez), porque cada paso necesita el output del paso anterior. En el paso $t$ el modelo predice un token $\hat{y}_t$, y en el paso $t+1$ ese mismo $\hat{y}_t$ se vuelve el input.
3. **Token de inicio `<SOS>`.** La primera iteracion no tiene un "token previo predicho", asi que se usa un token especial **`<SOS>`** (start-of-sentence) como input inicial. La generacion termina cuando el decoder produce el token **`<EOS>`** o cuando se alcanza un largo maximo `max_output_len`.

La forma matematica del paso $t$ del decoder, alineada con la teoria:

$$s_t = \text{LSTMCell}(\text{Embedding}(\hat{y}_{t-1}),\; s_{t-1})$$
$$\hat{y}_t = \arg\max \; \text{softmax}(W_{out}\, s_t)$$

con $s_0 = h_T$ (el context vector del encoder). Notar que $s_t$ no depende de los hidden states intermedios del encoder — esa es justamente la limitacion que attention va a resolver en la Parte 2.

## Loop de entrenamiento

Las utilidades de entrenamiento estan en la seccion `Train Utils` *(notebook 1, cell 22)* y la integracion completa en `Full Model` *(notebook 1, cell 26)*. Los componentes esperables de un Seq2Seq de este tipo:

- **Loss.** `nn.CrossEntropyLoss` aplicada en cada paso del decoder sobre la distribucion del vocabulario destino. Para una secuencia de salida de largo $L$, el loss total del ejemplo es la suma (o promedio) sobre los $L$ pasos:

$$\mathcal{L} = -\frac{1}{L} \sum_{t=1}^{L} \log P(y_t \mid y_{<t}, x; \theta)$$

  Esto equivale exactamente al objetivo que la teoria escribe como maximizar $\frac{1}{|TS|} \sum \log P(y_i \mid x_i; \theta)$.

- **Optimizador.** Tipicamente Adam con un learning rate moderado *(hiperparametros exactos a confirmar al ejecutar parte 1)*.

- **Teacher forcing.** Durante entrenamiento, en vez de alimentar al decoder con sus propias predicciones (que al inicio son ruido), se le pasan los **tokens ground truth** desplazados un paso. Esto estabiliza el entrenamiento y es estandar en Seq2Seq desde Sutskever 2014. La parte 3 de este lab esta dedicada precisamente a comparar teacher forcing vs sampling.

- **`n_epochs`.** `[a confirmar al ejecutar parte 1]`.

## Resultados modelo base

`[outputs pendientes — se integraran en Fase 2 cuando Roberto ejecute el notebook en Colab]`

Cuando se integren los resultados, esta seccion mostrara:

- **Curva de loss por epoch** (train y validation), para verificar que el modelo efectivamente esta aprendiendo y no esta sobreajustando agresivamente al training set.
- **Ejemplos de traduccion cualitativos** sobre el set de validacion: pares `comando ground truth → secuencia generada por el modelo`. La idea es ver tanto casos donde el modelo acierta exactamente (los comandos cortos como `jump` o `jump left` deberian salir bien) como casos donde falla (probablemente comandos largos con `around` o composiciones anidadas).
- Si aplica, una **metrica agregada** de exact-match accuracy sobre el validation set.

El analisis cualitativo es lo mas valioso aqui: importa menos el numero exacto que el **patron** de errores, porque ese patron motiva directamente la introduccion de attention en la Parte 2.

## Limitacion: el bottleneck

La arquitectura recien descrita tiene una limitacion fundamental que la teoria de la clase 13 enuncia con claridad: **toda la informacion del input tiene que viajar por un unico vector $h_T$ de dimension fija**. Para oraciones cortas no es problema — `jump` cabe sobrado en `hidden_dim` floats. Pero a medida que la entrada crece, el encoder tiene que comprimir cada vez mas informacion en el mismo numero de coordenadas, y el decoder no tiene forma de "volver a mirar" partes especificas del input cuando esta generando un token particular.

El ejemplo canonico de la teoria deja la intuicion clara:

```text
Encode: Economic growth has slowed down in recent years.
Decode: La croissance economique a ralenti ces dernieres annees.
```

Cuando el decoder esta generando `croissance` querria mirar a `growth`. Cuando genera `annees` querria mirar a `years`. Pero con un **`context vector`** unico no hay forma de hacer ese alineamiento dinamico — el decoder solo tiene acceso a la representacion comprimida $h_T$ de la oracion completa.

En SCAN el sintoma se observa de forma analoga: comandos compuestos como `jump around right twice` requieren generar secuencias largas de acciones donde cada paso del decoder deberia "atender" a un fragmento distinto del input (`jump` para los `JUMP`, `right` para los `RTURN`, `twice` para repetir el bloque). Un solo $h_T$ tiene que codificar **todo eso en un solo vector** y ademas distribuirlo de forma utilizable a lo largo de todos los pasos del decoder.

La Parte 2 ataca exactamente este cuello de botella reemplazando el context vector fijo por un **context vector adaptativo $c_t$** que en cada paso del decoder es un promedio ponderado de **todos** los hidden states del encoder $h_1, h_2, \ldots, h_T$. Los pesos $\alpha_{t,i}$ los aprende el modelo end-to-end y forman, despues del entrenamiento, un **alineamiento suave** entre input y output que se puede visualizar como un heatmap. Pero ese es el tema de la siguiente pagina.
