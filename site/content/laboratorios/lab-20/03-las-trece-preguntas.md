---
title: "Las trece preguntas"
weight: 3
math: true
---

El laboratorio plantea trece preguntas repartidas en cuatro bloques. Las ocho primeras son de código —cargar dos modelos siguiendo la documentación— y las cinco últimas son conceptuales. Esta página recorre qué se pide y, sobre todo, **qué hay que mirar** en cada caso.

## Preguntas 1–4 · RoBERTa

Cargar [RoBERTa](/papers/roberta-liu-2019), su tokenizador, tokenizar *"Hello World!"* y ejecutar el modelo.

```python
from transformers import RobertaTokenizer, RobertaModel
import torch

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base', return_dict=True)

inputs = tokenizer("Hello World!", return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
```

Tres cosas que conviene notar al ejecutarlo:

**No hay `token_type_ids`.** El diccionario que devuelve el tokenizador trae solo `input_ids` y `attention_mask`. Al eliminar Next Sentence Prediction, RoBERTa se quedó sin segmentos A/B que distinguir. Un código escrito para BERT que desempaquete tres claves falla aquí.

**Los tokens especiales son `<s>` y `</s>`**, con IDs **0** y **2** — no `[CLS]`/`[SEP]` con 101/102.

**El vocabulario es Byte-level BPE de 50.265**, heredado de [GPT-2](/papers/gpt-2-radford-2019). No produce `<unk>` nunca: cualquier cadena, en cualquier alfabeto, se representa como bytes.

## Preguntas 5–8 · Un modelo en español

Lo mismo con un modelo en español y la frase *"Hola Mundo!"*. La opción natural es [BETO](/papers/beto-canete-2020):

```python
from transformers import BertTokenizer, BertModel
import torch

CKPT = "dccuchile/bert-base-spanish-wwm-cased"
tokenizer = BertTokenizer.from_pretrained(CKPT)
model = BertModel.from_pretrained(CKPT, return_dict=True)

inputs = tokenizer("Hola Mundo!", return_tensors="pt")
outputs = model(**inputs)
```

Que se cargue con `BertTokenizer` y `BertModel` —y no con clases propias— **es el punto**: BETO replica BERT-base exactamente para funcionar como reemplazo directo de mBERT en cualquier pipeline existente.

El detalle que sorprende al inspeccionar: **`[CLS]` tiene ID 4**, no 101. El nombre del token especial es el mismo; el identificador depende del vocabulario, y el de BETO se construyó desde cero sobre texto en español.

Y el `wwm` del nombre del checkpoint no es decorativo: remite a [Whole Word Masking](/papers/whole-word-masking-cui-2019), el enmascarado por palabra completa en lugar de por subword.

## Pregunta 9 · El tokenizador cruzado

> *¿Qué efecto tendría usar el tokenizador de un modelo para otro, en inferencia, asumiendo que el código ejecuta sin errores?*

**Predicciones sin sentido, producidas en silencio.** El modelo recibe enteros y no puede saber que fueron generados con otro vocabulario: busca la fila correspondiente de su tabla de embeddings y sigue.

Si el identificador máximo del tokenizador A excede el vocabulario del modelo B, sí hay `IndexError`. Si cabe, no hay ninguna señal. La condición es puramente aritmética y está desarrollada en [el tokenizador cruzado](02-el-tokenizador-cruzado), junto con la práctica defensiva: `AutoTokenizer` y `AutoModel` desde un **único identificador de checkpoint**.

## Pregunta 10 · Los límites de un modelo sin instruction tuning

> *Los GPT completan texto según lo que estiman probable. ¿Es esto un impedimento para tareas dirigidas como summarization?*

Las opciones son "sí lo es", "no lo es" y **"es un impedimento en algunos casos, pero no en otros"** — que es la correcta, y la interesante.

GPT-2 **puede** producir resúmenes con el truco de `TL;DR:` al final del texto, y el propio [paper de GPT-2](/papers/gpt-2-radford-2019) lo reporta como capacidad zero-shot. Funciona porque esa cadena aparece en el corpus de entrenamiento seguida de resúmenes reales: el modelo no "entiende la instrucción", **reconoce un patrón textual**.

Y ahí está el límite: sin instruction tuning ni [RLHF](/papers/instructgpt-ouyang-2022), el comportamiento es **inconsistente**. Funciona para las tareas cuyo formato quedó representado en el corpus, y no para las demás. Es exactamente la brecha que InstructGPT vino a cerrar, y por eso ChatGPT aparece al final de la [clase 20](/clases/clase-20) y no al principio.

## Pregunta 11 · `max_length=10000`

> *Se quiere generar varias páginas sobre "New York" con `max_length=10000`. ¿Cuál es el problema? ¿Cómo se resuelve?*

**GPT-2 tiene 1024 posiciones.** Es un límite arquitectónico: la matriz de embeddings posicionales tiene exactamente ese número de filas.

$$\texttt{n\_positions} = 1024 \quad \ll \quad \texttt{max\_length} = 10000$$

No hay nada que el modelo pueda hacer con la posición 1025 — el embedding correspondiente no existe. Según la versión de la librería, el resultado es un error de índice o un truncado silencioso a 1024.

Soluciones concretas:

- **Ventana deslizante:** generar por tramos de ~512 tokens, y usar los últimos tokens generados como prompt del siguiente. La continuidad es local, así que el texto deriva temáticamente en textos largos.
- **Un modelo con contexto mayor.** Es la solución real, y la que la historia tomó: los modelos posteriores ampliaron el contexto por órdenes de magnitud.

Conviene notar que el problema **no es de memoria ni de tiempo**: es que las posiciones más allá de 1024 no están definidas.

## Preguntas 12–13 · Llevarlos al español

> *¿Qué recomendaría para usar BERT y GPT-2 en español, con recursos acotados?*

**BERT: sí, y es directo.** Usar un checkpoint pre-entrenado en español —BETO— en lugar de traducir o de entrenar desde cero. Es tan simple como cambiar el identificador del checkpoint, porque la arquitectura es idéntica. El laboratorio ya lo hizo en las preguntas 5-8.

**GPT-2: sí, pero con menos opciones.** `PlanTL-GOB-ES/gpt2-base-bne` (Biblioteca Nacional de España) o `DeepESP/gpt2-spanish`. Ambos son considerablemente más chicos y están entrenados sobre corpus menores que sus equivalentes en inglés.

{{< concept-alert type="clave" >}}
**La asimetría es el hallazgo de estas dos preguntas.** Para BERT en español hay un ecosistema maduro: BETO, sus variantes destiladas, RoBERTa-base-spanish, modelos de dominio. Para GPT-2 en español las alternativas son pocas y notoriamente más débiles.

La razón es económica. Un encoder de 110 M de parámetros se entrena con recursos que un grupo universitario puede conseguir —BETO se entrenó con TPUs donadas por el programa TFRC de Google—. Un decoder generativo competitivo exige un orden de magnitud más de cómputo y datos, y el retorno académico de publicarlo era menor.

Esa asimetría marcó el ecosistema hispanohablante durante años, y solo empezó a cerrarse cuando aparecieron modelos multilingües grandes que hicieron irrelevante la pregunta.
{{< /concept-alert >}}

## Lo que el conjunto revela

Nueve de las trece preguntas se responden **leyendo lo que se cargó**: qué tokenizador, qué vocabulario, qué identificadores, qué límites de posición. Ninguna requiere entrenar nada.

Es el argumento pedagógico del laboratorio: en la era de los modelos pre-entrenados, buena parte del trabajo consiste en **saber qué hay dentro del checkpoint que acabas de descargar**. Los errores no vienen de la arquitectura —que es la misma en los cinco modelos— sino de las convenciones que la rodean.

---

**Siguiente:** [Fake news y el atajo de Reuters](04-fake-news-y-el-atajo-de-reuters) — el único entrenamiento del laboratorio, y lo que realmente aprende.
