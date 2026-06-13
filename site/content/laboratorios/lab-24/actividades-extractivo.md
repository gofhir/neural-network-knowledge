---
title: "P1 — Actividad, experimento propio y atención (BertViz)"
weight: 3
math: true
---

> **Celdas 26-41 del notebook `QA_BERT_Spanish.ipynb` (Parte 1).** Tres bloques que cierran el extractivo: (A) un experimento propio sobre un contexto de dominio FHIR — fuera de la distribución de entrenamiento, donde aparece el hallazgo estrella; (B) las tres afirmaciones Verdadero/Falso; y (C) el bonus de visualización de atención con BertViz, con su matiz honesto sobre qué prueba y qué no prueba la atención.

## A. Experimento propio: BETO sobre dominio FHIR

El notebook invita a probar el lector con un contexto y preguntas propias. En vez de un párrafo de Wikipedia, se le dio a BETO un texto técnico **del dominio que conozco** — el estándar de interoperabilidad clínica HL7 FHIR — precisamente para sacarlo de su zona de confort. SQuAD-es es Wikipedia traducida; nada en ese corpus se parece a la jerga de los recursos FHIR.

```python
context = (
    "El estándar HL7 FHIR define recursos para representar información "
    "clínica. El recurso Patient almacena datos demográficos del paciente, "
    "mientras que el recurso Observation registra mediciones como signos "
    "vitales o resultados de laboratorio. FHIR fue publicado por primera "
    "vez en 2014 por la organización Health Level Seven International."
)
```

### Los cuatro resultados reales

| Pregunta | Respuesta del modelo | Veredicto |
|---|---|---|
| ¿Qué recurso almacena los datos demográficos del paciente? | `Patient` | ✅ Extracción directa |
| ¿En qué año se publicó FHIR por primera vez? | `2014` | ✅ Span numérico mínimo |
| ¿Qué organización publicó FHIR? | `empty` | ❌ ⭐ **Falso negativo** — la respuesta **sí** está en el contexto |
| ¿Cuál es el precio de una licencia de FHIR? | `empty` | ✅ Abstención correcta (no está en el contexto) |

Las dos primeras salen perfectas: el modelo extrae `Patient` y `2014` sin problema, incluso siendo vocabulario que jamás vio en entrenamiento. La cuarta es la abstención *deseada* — no hay nada sobre precios en el texto, y BETO se calla, igual que con la comida de Ecuador en la [página anterior](inferencia-extractiva).

**La tercera es el hallazgo.** La respuesta — *"Health Level Seven International"* — está **literalmente escrita** en la última oración del contexto. El modelo, sin embargo, devuelve `empty`. Es un **falso negativo**: se abstuvo cuando había evidencia. Esto no es "saber cuándo no responder"; es **fallar en encontrar lo que estaba ahí**.

### Experimento del umbral (y por qué no era calibración)

La primera hipótesis fue: *quizás el span correcto sí se calculó, pero su score quedó por debajo del span nulo, y basta con ajustar el umbral*. Recordemos la lógica exacta del `null_score_diff_threshold`:

$$
\text{predecir "empty"} \quad \Longleftrightarrow \quad \text{score}_\text{null} - \text{score}_\text{mejor span} > \text{threshold}
$$

Conviene fijar bien el **signo**, porque es contraintuitivo:

- **Bajar** el threshold (hacia negativo) hace que la condición se cumpla más fácil → el modelo **se abstiene MÁS**.
- **Subir** el threshold exige que el span nulo gane por un margen mayor → el modelo **arriesga MÁS respuestas**.

Para intentar recuperar la respuesta de la pregunta 3 había que **subir** el umbral, no bajarlo. Se hizo el experimento... y **la pregunta 3 siguió dando `empty`**. Conclusión limpia: el problema **no era principalmente de calibración**. Si solo fuera que el span correcto perdía por poco contra el nulo, subir el umbral lo habría rescatado. No lo hizo → el span correcto ni siquiera estaba bien posicionado entre los candidatos.

### Reformulación (el span "sucio" que sí funcionó)

El segundo experimento fue cambiar la **forma** de la pregunta, dejando el contexto idéntico:

> *"¿Qué organización publicó FHIR?"* → `empty` ❌
> *"¿Quién publicó FHIR?"* → `organización Health Level Seven International` ✅

Cambiar **qué** por **quién** recuperó la respuesta. Pero fíjate en el span devuelto: incluye el sustantivo común **"organización"** pegado al nombre propio. Es un span **"sucio"** — semánticamente correcto, con los bordes imperfectos.

### Análisis: cuatro causas acumuladas del falso negativo

El fallo de la pregunta 3 no tiene una sola causa; es la suma de cuatro factores que se refuerzan:

1. **Vocabulario OOD (out-of-distribution).** "Health Level Seven International" es un nombre propio técnico que nunca apareció en el entrenamiento. SQuAD-es es Wikipedia traducida; el modelo no tiene representaciones robustas para esta entidad.
2. **`do_lower_case=True` sobre un modelo *cased*.** Como se vio en la página anterior, la config baja todo a minúsculas, pero BETO es `...wwm-cased`. Para nombres propios — donde la mayúscula es la señal de que *esto es una entidad* — bajar a minúsculas **degrada justo la pista** que necesita.
3. **Span largo + estructura compleja.** La respuesta correcta es una entidad multi-token, y la palabra **"organización" aparece tanto en la pregunta como en el contexto**, justo antes del nombre. Eso confunde la localización: el modelo no sabe si el límite del span empieza en "organización" o en "Health".
4. **Confianza bajo el umbral.** Encima de todo lo anterior, el score del candidato quedó por debajo del span nulo — pero, como mostró el experimento, esto era síntoma, no causa raíz.

**La lección.** La "abstención" de un modelo extractivo **no siempre significa "no hay evidencia"**. A veces significa *"no supe mapear esta pregunta a un span"*. Es **fragilidad ante la estructura de la pregunta**, no incertidumbre epistémica genuina. El mismo dato, preguntado con "quién" en vez de "qué", aparece. Eso es preocupante: en producción, un usuario que formula mal la pregunta recibe un `empty` indistinguible de "el dato no existe".

Y el span "sucio" (`organización Health Level Seven International`) conecta directo con la crítica a las métricas: Exact Match daría **0** a esa respuesta por incluir un sustantivo de más, y F1 la penalizaría, aunque para cualquier humano es **correcta**. Es el caso de libro de cómo EM/F1 castigan respuestas semánticamente válidas con bordes imperfectos — ver [métricas de evaluación de QA](/fundamentos/qa-evaluation-metrics).

## B. Verdadero / Falso (celdas con justificación)

Las tres afirmaciones conceptuales que cierran el bloque extractivo:

| Afirmación | Respuesta | Razón |
|---|---|---|
| *The SQuAD is a reading comprehension dataset* | **True** | SQuAD es exactamente eso: un dataset de comprensión lectora (el modelo lee un pasaje y responde preguntas sobre él). |
| *The BERT model is trained from scratch for the QA task* | **False** | BERT se **pre-entrena** de forma genérica (MLM + NSP) sobre texto masivo; para QA solo se hace **fine-tuning** añadiendo los dos vectores de span (inicio $S$ y fin $E$). Es **transfer learning**, no entrenamiento desde cero. |
| *This model generates the answer word by word (generative approach)* | **False** | El modelo de la Parte 1 es **extractivo**: predice **posiciones** (start/end) de un span dentro del contexto, no produce tokens nuevos. Lo generativo "palabra por palabra" es la **Parte 2** (encoder-decoder). |

La segunda y la tercera apuntan al mismo malentendido por dos ángulos: BERT extractivo **no inventa texto** y **no se entrena de cero**. Confundir esto con un enfoque generativo es exactamente lo que separa esta Parte 1 de la Parte 2 del laboratorio.

## C. Bonus: visualización de atención con BertViz (celdas 33-41)

[**BertViz**](https://github.com/jessevig/bertviz) (Jesse Vig, 2019) es una herramienta de visualización interactiva de los pesos de atención de los Transformers. En la *head view* dibuja una línea por cada par de tokens (origen → destino); el **grosor** codifica la magnitud del peso y el **color** identifica la cabeza de atención.

### El detalle clave: se carga `BertModel`, no `BertForQuestionAnswering`

```python
from transformers import BertModel
model = BertModel.from_pretrained(model_name, output_attentions=True)
```

Esto es importante y fácil de pasar por alto: BertViz carga el **encoder base** (`BertModel`), **no** el modelo de QA (`BertForQuestionAnswering`) que tomó las decisiones de span en las páginas anteriores. Por lo tanto, **lo que se visualiza es la atención del encoder**, no el cómputo que produjo las respuestas. Es la misma columna vertebral, pero sin la cabeza de span encima.

La preparación de la entrada:

```python
inputs = tokenizer.encode_plus(pregunta, contexto, return_tensors="pt")
# sentence_b_start = primer índice donde token_type_id == 1
show_head_view(model, tokenizer, inputs, sentence_b_start)
```

El `sentence_b_start` marca dónde termina la pregunta (segmento 0) y empieza el contexto (segmento 1), usando los `token_type_ids` — para que la vista separe visualmente ambos tramos.

### Qué se ve

Sobre el ejemplo de Quito aparecen patrones reconocibles:

- **Cabezas semánticas de alineación**: por ejemplo, el token `provincia` de la pregunta tira líneas fuertes hacia `Pichincha` en el contexto — la cabeza está "alineando" la pregunta con su respuesta candidata.
- **Cabezas nulas / de sumidero**: muchas cabezas concentran su atención en `[SEP]` o `[CLS]`, un *attention sink* bien documentado (cuando una cabeza no tiene nada útil que mirar, "estaciona" el peso en un token especial).
- **Cabezas sintácticas**: alineaciones locales (artículo↔sustantivo, preposición↔objeto).
- Como regla general, las **capas tempranas** capturan relaciones más **locales y sintácticas**, y las **capas profundas** relaciones más **semánticas y de largo alcance**.

> **Gotcha — el notebook explota con secuencias largas.** La atención es $O(n^2)$ en el número de tokens, y BertViz dibuja una línea por par. Con un contexto largo el navegador intenta renderizar decenas de miles de líneas y se **satura** (la celda se cuelga o el tab muere). El remedio: **usar contextos cortos** para la visualización.

### Matiz honesto: la atención no es explicación

La celda del notebook sugiere que la atención muestra "qué tokens contribuyen a la predicción". Conviene tomarlo con pinzas, por **dos** razones:

1. **El debate "Attention is not Explanation".** [Jain & Wallace (2019)](https://arxiv.org/abs/1902.10186) mostraron que se pueden construir distribuciones de atención **alternativas** que dan la misma predicción — es decir, los pesos de atención **no son una explicación fiel** de por qué el modelo decidió lo que decidió. La réplica [Wiegreffe & Pinter (2019), "Attention is not not Explanation"](https://arxiv.org/abs/1908.04626) matiza que *depende de la definición de explicación*, pero ninguno de los dos sostiene que los pesos prueben causalidad.
2. **Aquí ni siquiera es el modelo de QA.** Como se dijo, BertViz carga `BertModel` base, no el `BertForQuestionAnswering` que eligió los spans. Aunque la atención fuera explicación perfecta, sería la del encoder genérico, **no** la de la cabeza de span que produjo `Pichincha` o el falso negativo de FHIR.

La conclusión sana: BertViz es **excelente para construir intuición** sobre cómo el Transformer mueve información entre tokens, pero **no sirve para probar** por qué el modelo de QA respondió lo que respondió. Útil para entender, insuficiente para explicar.

---

**Anterior:** [Inferencia extractiva](inferencia-extractiva) · **Siguiente:** [Arquitectura generativa](arquitectura-generativa)
