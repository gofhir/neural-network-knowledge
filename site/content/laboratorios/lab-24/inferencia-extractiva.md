---
title: "P1 — Inferencia extractiva y abstención"
weight: 2
math: true
---

> **Celdas 18-25 del notebook `QA_BERT_Spanish.ipynb` (Parte 1).** El pipeline de predicción extractiva de extremo a extremo, el ejemplo de Quito ejecutado en vivo y el mecanismo que es el corazón de SQuAD v2.0: **saber cuándo NO responder**.

## El modelo: BETO ya fine-tuneado en SQuAD-es v2

El notebook ofrece dos caminos con un flag (`use_own_model`): entrenar el lector desde cero, o cargar uno ya listo. En inferencia se toma el segundo (`use_own_model=False`) y se descarga:

```python
model_name = "mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
```

Este modelo es **exactamente el producto que el entrenamiento intentaba fabricar**, pero ya hecho: es **BETO** (`bert-base-spanish-wwm-cased`, el BERT español de la Universidad de Chile) fine-tuneado sobre la traducción al español de SQuAD v2. En vez de pagar las horas de GPU, se reutiliza el checkpoint público.

Los imports son **APIs legacy** de `transformers`, no el pipeline moderno de alto nivel:

```python
from transformers import BertForQuestionAnswering, BertTokenizer
from transformers.data.processors.squad import SquadExample, SquadResult, squad_convert_examples_to_features
from transformers.data.metrics.squad_metrics import compute_predictions_logits
```

> **Gotcha — "restart session if error".** Estas funciones (`squad_convert_examples_to_features`, `compute_predictions_logits`) pertenecen a una era anterior de la librería y su firma ha cambiado entre versiones. Si la importación falla por incompatibilidad, el remedio del notebook es reiniciar el runtime de Colab para forzar una recarga limpia del paquete.

## Hiperparámetros de inferencia

Tres números controlan cómo se convierten los logits del modelo en una respuesta de texto:

| Hiperparámetro | Valor | Qué hace |
|---|---|---|
| `n_best_size` | `1` | Devuelve solo **la mejor** respuesta (no un top-k de candidatos). |
| `max_answer_length` | `30` | Descarta cualquier span de más de 30 tokens — un span absurdamente largo casi nunca es la respuesta. |
| `null_score_diff_threshold` | `0.0` | **El umbral de abstención de v2.0.** |

El `null_score_diff_threshold` es el mecanismo central de esta página. El modelo siempre calcula dos cosas: el score del mejor span de texto y el **score del span nulo** (la posibilidad de que no haya respuesta, anclada en el token `[CLS]`). La decisión es:

$$
\text{predecir "empty"} \quad \Longleftrightarrow \quad \text{score}_\text{null} - \text{score}_\text{mejor span} > \text{threshold}
$$

Con `threshold = 0.0`, el modelo se abstiene en cuanto el span nulo gana al mejor span. Subir el umbral lo hace **más conservador** (responde menos); bajarlo lo vuelve más arriesgado.

> **Gotcha — `do_lower_case=True` sobre un modelo CASED.** La configuración pasa `do_lower_case=True`, pero el checkpoint es `...wwm-cased`, es decir, fue entrenado distinguiendo mayúsculas de minúsculas. Es una **leve inconsistencia**: en teoría bajar todo a minúsculas degradaría la entrada respecto a lo que el modelo espera. En la práctica los resultados de Quito salen correctos, pero conviene tenerlo presente.

## La función `run_prediction`: seis etapas

El corazón de la Parte 1 es una sola función que orquesta todo el pipeline extractivo. Conviene leerla como una tubería de seis etapas:

1. **Envolver la entrada en `SquadExample`.** Cada pregunta + contexto se mete en el objeto que el procesador de SQuAD espera, con un `qas_id` único.
2. **`squad_convert_examples_to_features`.** Tokeniza, parte el contexto en **ventanas deslizantes** (`doc_stride=128`) cuando es más largo que `max_seq_length`, y construye el **offset mapping** que recuerda a qué carácter original corresponde cada token. Sin ese mapeo no se puede recuperar el texto de la respuesta al final.
3. **`DataLoader` con `SequentialSampler`.** En inferencia el orden importa (las ventanas de un mismo documento deben quedar contiguas), así que se recorren secuencialmente, no al azar.
4. **Forward pass.** Con `model.eval()` y `torch.no_grad()` (sin dropout, sin gradientes). La entrada incluye `token_type_ids` — los **segment embeddings** que le dicen a BERT qué tokens son la pregunta (segmento 0) y cuáles el contexto (segmento 1). La salida son dos vectores: `start_logits` y `end_logits`, un puntaje por posición de inicio y de fin.
5. **`SquadResult`.** Empaqueta los logits de cada feature junto a su `unique_id` para poder reensamblar.
6. **`compute_predictions_logits`.** El paso final y más rico: convierte los logits en texto, **reúne las ventanas** del mismo documento, compara el mejor span contra el **span nulo `[CLS]`** (aquí entra el umbral de abstención) y usa el offset mapping para **devolver caracteres del contexto original**, no tokens.

## El ejemplo de Quito (resultado real ejecutado)

El notebook prueba el pipeline sobre un contexto sobre Quito y cuatro preguntas. Estas son las respuestas **reales** que produjo el modelo:

| Pregunta | Respuesta del modelo | Veredicto |
|---|---|---|
| ¿Cuál es la población de Quito? | `2 millones` | ✅ Span mínimo correcto — no "2 millones de habitantes" |
| ¿En qué provincia esta ubicado Quito? | `Pichincha` | ✅ Extracción directa |
| ¿Cuál es la cápital más antigua de Sudamérica? | `Quito` | ✅ Requiere correferencia, tolera el typo |
| ¿Qué tan buena es la comida en Ecuador? | `empty` | ✅ ⭐ **Se abstuvo** (trampa) |

Tres observaciones:

- **Span mínimo.** En "población" el modelo devuelve `2 millones` y se detiene, sin arrastrar "de habitantes". El entrenamiento de SQuAD premia el span más corto que sea suficiente, y el modelo lo aprendió.
- **Correferencia + robustez al typo.** El texto dice algo como "Quito... es la capital más antigua de Sudamérica". La pregunta usa `cápital` (con tilde mal puesta) y aun así el modelo resuelve que el sujeto es **Quito** y lo extrae. Está haciendo una pequeña cadena de correferencia, no un simple match de palabras.
- **La trampa.** La cuarta pregunta — la calidad de la comida en Ecuador — **no tiene respuesta en el contexto**. El modelo lo reconoce y devuelve `empty`.

## Por qué `empty` es el corazón de v2.0

Esa cuarta respuesta es el punto entero de esta página. Un modelo entrenado en **SQuAD v1.1** estaría **obligado a inventar** algún span: siempre devuelve el texto del contexto que "menos mal encaja", aunque la evidencia no exista. **SQuAD v2.0** introdujo las preguntas sin respuesta justamente para enseñarle al modelo el eje que importa: **saber cuándo NO responder**. El span nulo y el `null_score_diff_threshold` son la maquinaria que hace posible esa abstención.

> **El contraste con el lab-23.** En el [lab de BLIP](/laboratorios/lab-23), el modelo multimodal **no tenía forma de abstenerse**: ante una imagen ambigua de un ornitorrinco respondía con confianza algo incorrecto, **alucinando**. Aquí, BETO ante una pregunta sin evidencia se calla. La diferencia no es de tamaño ni de idioma — es de **diseño de la tarea**: v2.0 hornea la opción de no responder en los datos y en el objetivo.

Y este mismo contraste anticipa la **Parte 2 generativa** del lab: un modelo encoder-decoder que *genera* la respuesta token a token **no extrae** spans del contexto y, por lo tanto, **sí alucinará** cuando la evidencia falte. La abstención extractiva de la Parte 1 y la fluidez riesgosa de la Parte 2 son los dos polos del trade-off que define este laboratorio. Para entender cómo se mide formalmente este "saber callar" — Exact Match, F1 y el manejo de los no-answer — ver [métricas de evaluación de QA](/fundamentos/qa-evaluation-metrics).

---

**Anterior:** [Arquitectura: BERT extractivo](arquitectura-extractiva) · **Siguiente:** [Actividades y visualización de atención](actividades-extractivo)
