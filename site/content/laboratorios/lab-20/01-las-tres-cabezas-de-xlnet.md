---
title: "Las tres cabezas de XLNet"
weight: 1
math: true
---

El laboratorio abre cargando [XLNet](/papers/xlnet-yang-2019) tres veces seguidas. Las tres celdas parecen la misma con distinto nombre de clase, y esconden la diferencia más instructiva del notebook: **el encoder es idéntico; lo que cambia son unos pocos parámetros al final**.

## Las tres celdas

```python
# [12] backbone
from transformers import XLNetTokenizer, XLNetModel
model = XLNetModel.from_pretrained('xlnet-base-cased', return_dict=True)
outputs = model(**tokenizer("Hello, my dog is cute", return_tensors="pt"))
last_hidden_states = outputs.last_hidden_state

# [13] question answering
from transformers import XLNetForQuestionAnswering
model = XLNetForQuestionAnswering.from_pretrained('xlnet-base-cased', return_dict=True)
outputs = model(input_ids, start_positions=torch.tensor([1]), end_positions=torch.tensor([3]))

# [14] multiple choice
from transformers import XLNetForMultipleChoice
model = XLNetForMultipleChoice.from_pretrained('xlnet-base-cased', return_dict=True)
outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)
```

Los tres cargan `xlnet-base-cased`, o sea **el mismo checkpoint**: los ~110 M de parámetros del encoder son bit a bit los mismos. Lo que cambia es la cabeza que se monta encima, y esa cabeza **no está en el checkpoint** — se inicializa al azar y hay que entrenarla.

Por eso las tres celdas imprimen un aviso de HuggingFace listando pesos sin inicializar. **No es un error.** Es la consecuencia directa de que un modelo pre-entrenado no sepa nada de tu tarea.

## Por qué la cabeza de QA tiene seis submódulos

Aquí está lo que sorprende. Al cargar `XLNetForQuestionAnswering` el aviso lista **seis** submódulos:

```
- start_logits.dense
- end_logits.dense_0, end_logits.LayerNorm, end_logits.dense_1
- answer_class.dense_0, answer_class.dense_1
```

[BERT](/papers/bert-devlin-2018) para la misma tarea tiene **dos vectores** $S, E \in \mathbb{R}^H$ y nada más. ¿Por qué XLNet necesita tres veces esa maquinaria?

**`start_logits`** es la parte simple, y es idéntica a BERT: una `Linear(768, 1)` por token, softmax sobre el párrafo.

$$p(\text{start} = i \mid x) = \frac{\exp(s_i)}{\sum_j \exp(s_j)}, \qquad s_i = W_s h_i + b_s$$

**`end_logits`** es la innovación:

```
end_logits.dense_0:   Linear(2 * hidden_size, hidden_size)   # [h_end, h_start] concatenados
end_logits.LayerNorm: LayerNorm(hidden_size)
end_logits.dense_1:   Linear(hidden_size, 1)
```

{{< concept-alert type="clave" >}}
**El final se condiciona en la representación del inicio.** BERT predice los dos extremos de forma independiente: puede asignar alta probabilidad a un `start` en la posición 40 y a un `end` en la 12, produciendo un span imposible que hay que descartar con reglas externas.

XLNet concatena $[h_{\text{end}}, h_{\text{start}}]$ y pasa el par por una MLP. Es un diseño estilo **R-Net**, no estilo BERT.

Y es exactamente **el mismo argumento que motiva a XLNet entero**. El defecto que ataca frente a BERT es que al enmascarar varios tokens a la vez, BERT los predice de forma independiente y no modela la dependencia entre ellos. La cabeza de QA aplica esa misma crítica al nivel de la tarea: *el final de una respuesta depende de dónde empieza*.
{{< /concept-alert >}}

**`answer_class`** es una cabeza dedicada a decidir si la pregunta **tiene respuesta**, para el escenario de [SQuAD 2.0](/papers/squad2-rajpurkar-2018) — donde un tercio de las preguntas son deliberadamente irrespondibles y el sistema debe abstenerse. BERT resuelve esto con un truco: apuntar el span al token `[CLS]`. XLNet le da un clasificador propio.

## `[CLS]` va al final

El detalle que rompe el reflejo aprendido con BERT:

```
Entrada:  "Hello world"
BERT:     [CLS] Hello world [SEP]        <- CLS al INICIO
XLNet:    ▁Hello ▁world <sep> <cls>      <- CLS al FINAL
```

Quien escriba `last_hidden_state[:, 0, :]` esperando el vector de clasificación **obtiene el primer token del texto**, no el `<cls>`. Hay que usar `[:, -1, :]`.

Lo insidioso es que no falla: devuelve un tensor de la forma correcta, con números plausibles, que representan otra cosa. Es la misma familia de error que [el tokenizador cruzado](02-el-tokenizador-cruzado).

## La dependencia que hay que instalar

```python
!pip install -q transformers==4.56.2
!pip install -q sentencepiece      # <- esta linea no es opcional
```

`XLNetTokenizer` usa un vocabulario Unigram LM construido con [SentencePiece](/papers/sentencepiece-kudo-2018), y ese paquete **no viene entre las dependencias obligatorias de HuggingFace** — es una extensión en C++ que solo necesitan algunos tokenizadores. Sin ella, el fallo es poco informativo:

```
Couldn't instantiate the backend tokenizer
```

El aprendizaje operativo: **la elección de tokenizador arrastra dependencias de sistema**, no solo diferencias de vocabulario.

## La celda de opción múltiple

```python
encoding = tokenizer([[prompt, prompt], [choice0, choice1]], return_tensors='pt', padding=True)
outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)
```

La estructura del input es lo interesante: **la premisa se repite una vez por candidato**. El modelo codifica $n$ pares independientes, produce un escalar por par y aplica softmax sobre ellos.

Esa es la forma canónica de las tareas tipo [SWAG](/papers/swag-zellers-2018), y explica el `unsqueeze(0)`: la dimensión de batch se agrega a mano porque el tensor ya tiene una dimensión de candidatos.

El comentario del propio notebook lo dice sin adornos:

```python
# the linear classifier still needs to be trained
```

La `loss` que devuelve esa celda no significa nada — mide qué tan mal predice un clasificador con pesos aleatorios.

## Lo que las tres celdas enseñan juntas

Un modelo pre-entrenado no es un modelo que resuelve tareas: es **un encoder de texto**. Las cabezas son la parte pequeña, específica y no entrenada — unos miles de parámetros contra ciento diez millones.

Ese reparto es lo que hace viable el paradigma completo: se entrena el encoder una vez, a un costo enorme, y después cada tarea cuesta una cabeza y un fine-tuning corto. El mismo checkpoint sirve para QA, para opción múltiple y para clasificación.

---

**Siguiente:** [El tokenizador cruzado](02-el-tokenizador-cruzado) — qué pasa cuando el tokenizador y el modelo no vienen del mismo checkpoint.
