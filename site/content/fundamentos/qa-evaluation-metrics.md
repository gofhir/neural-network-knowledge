---
title: "Métricas de Evaluación de QA"
weight: 92
math: true
---

**Question Answering (QA)** es una de las tareas más visibles del NLP moderno: dado un contexto y una pregunta, el sistema debe producir una respuesta. Pero evaluar esa respuesta es engañosamente difícil. A diferencia de la clasificación binaria (donde hay una etiqueta correcta y punto), una misma pregunta admite muchas respuestas correctas con distinta superficie léxica, y según el formato — respuesta cerrada, extractiva sobre un span, o generativa libre — la métrica adecuada cambia por completo.

Este fundamento consolida las métricas canónicas de evaluación de QA que se usan en los benchmarks dominantes — **accuracy** para respuesta cerrada, **Exact Match (EM)** y **token-level F1** para QA extractivo (SQuAD), **Mean Reciprocal Rank (MRR)** y **top-k retrieval accuracy** para passage retrieval, **BLEU/ROUGE** para QA generativo (MS MARCO), el manejo de la **abstención** en SQuAD 2.0, y las métricas semánticas modernas (BERTScore, LLM-as-judge) que intentan cerrar el gap entre lo léxico y lo semántico. Sirve como fundamento transversal de la [Clase 24](/clases/clase-24) del curso IA UC y complementa los fundamentos de [Question Answering](/fundamentos/question-answering) y [Machine Reading Comprehension](/fundamentos/machine-reading-comprehension).

---

## 1. Por qué evaluar QA es difícil

La dificultad central es que **una pregunta no tiene una única respuesta correcta en una única forma superficial**. Considera:

> **Pregunta**: ¿En qué año se fundó la Universidad de Chile?
>
> Respuestas todas correctas: `1842`, `en 1842`, `el año 1842`, `fue fundada en 1842`.

Las cuatro transmiten el mismo hecho, pero su representación textual difiere en longitud, en artículos y en puntuación. Una métrica ingenua de comparación de strings (`prediction == gold`) marcaría tres de ellas como incorrectas. Esto motiva dos ideas que recorren todo el campo:

1. **Normalización**: antes de comparar, se canonicaliza la respuesta (minúsculas, sin puntuación, sin artículos) para que diferencias triviales no penalicen.
2. **Comparación parcial / blanda**: en lugar de exigir coincidencia exacta, se mide *cuánto* se parecen predicción y gold (a nivel de tokens compartidos, de subsecuencia, de embedding semántico).

A esto se suma la distinción de **formato de respuesta**, que determina la familia de métrica:

| Formato | Ejemplo de benchmark | Métrica dominante |
| --- | --- | --- |
| **Respuesta cerrada** (cloze, multiple choice) | LAMBADA, RACE, MMLU, CBT | Accuracy |
| **Extractivo** (span de un pasaje) | SQuAD 1.1, SQuAD 2.0, NewsQA | Exact Match + token F1 |
| **Retrieval / ranking de candidatos** | TREC QA, DPR, Natural Questions (open) | MRR, top-k accuracy |
| **Generativo / abstractivo** (texto libre) | MS MARCO, NarrativeQA, ELI5 | ROUGE-L, BLEU, BERTScore |

{{< concept-alert type="clave" >}}
**No existe una métrica única de QA.** La elección depende del formato de respuesta. Confundir formatos — por ejemplo, aplicar accuracy a una respuesta generativa libre, o BLEU a un cloze — produce números sin sentido. El primer paso de cualquier evaluación de QA es identificar qué tipo de respuesta produce el sistema.
{{< /concept-alert >}}

---

## 2. Accuracy: el caso de respuesta cerrada

Cuando la respuesta es **una de un conjunto finito y discreto de opciones**, QA colapsa a clasificación, y la métrica natural es la **accuracy**: la fracción de preguntas respondidas correctamente.

$$\text{accuracy} = \frac{\text{correct}}{\text{correct} + \text{incorrect}} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = y_i]$$

donde $N$ es el número total de preguntas, $\hat{y}_i$ la respuesta del modelo y $y_i$ la gold answer, y $\mathbb{1}[\cdot]$ es la función indicadora (1 si coincide, 0 si no).

Aplica limpiamente en dos formatos:

- **Cloze (rellenar el hueco)**: "Marie Curie ganó el Premio \_\_\_\_". El modelo predice una palabra de su vocabulario; se compara contra el token gold. Benchmarks: Children's Book Test (CBT), LAMBADA.
- **Multiple choice**: la pregunta viene con $k$ opciones (A, B, C, D) y exactamente una es correcta. Benchmarks: RACE, MMLU, ARC. La accuracy aquí tiene un piso de azar de $1/k$ (25% para 4 opciones).

**Ejemplo numérico**: un modelo responde 850 de 1000 preguntas multiple-choice de 4 opciones correctamente.

$$\text{accuracy} = \frac{850}{1000} = 0.85 = 85\%$$

Como el azar daría 25%, el modelo aporta $85 - 25 = 60$ puntos sobre la línea base aleatoria.

La accuracy es atractiva por su simplicidad e interpretabilidad, pero **solo funciona cuando el espacio de respuestas es discreto y cerrado**. En cuanto la respuesta es un span de texto libre o una oración generada, la coincidencia exacta es demasiado rígida (volvemos al problema de `1842` vs `en 1842`), y necesitamos métricas más finas.

---

## 3. Exact Match (EM)

**Exact Match** es la métrica más estricta para QA extractivo. Vale 1 si la predicción coincide *exactamente* (tras normalización) con **alguna** de las gold answers, y 0 en caso contrario. Se promedia sobre el dataset:

$$\text{EM} = \frac{1}{N} \sum_{i=1}^{N} \max_{g \in G_i} \mathbb{1}\big[\text{norm}(\hat{y}_i) = \text{norm}(g)\big]$$

donde $G_i = \{g_1, \dots, g_m\}$ es el conjunto de gold answers de la pregunta $i$ (en SQuAD, típicamente 3 anotadores humanos), y $\text{norm}(\cdot)$ es la normalización estándar (Sección 5).

**Ejemplo**:

> **Pregunta**: ¿Quién escribió *Cien años de soledad*?
>
> **Gold answers**: `["Gabriel García Márquez", "García Márquez"]`
>
> **Predicción A**: `"García Márquez"` → tras normalizar, coincide con la segunda gold → **EM = 1**.
>
> **Predicción B**: `"el escritor Gabriel García Márquez"` → no coincide exactamente con ninguna → **EM = 0**.
>
> **Predicción C**: `"gabriel garcia marquez"` → tras normalizar (minúsculas, sin tildes no aplica aquí pero sí lowercase) coincide con la primera → **EM = 1**.

{{< concept-alert type="ojo" >}}
La **predicción B** es semánticamente perfecta — incluso *más* informativa que la gold — pero EM la castiga con 0 porque agregó palabras ("el escritor"). Esta rigidez es la principal debilidad de EM: penaliza respuestas correctas pero más verbosas o parcialmente solapadas. Por eso SQuAD nunca reporta EM solo, sino EM **junto con** token-level F1, que da crédito parcial.
{{< /concept-alert >}}

EM es binaria y no perdona: o coincide todo, o nada. Es útil porque es interpretable ("¿qué porcentaje de respuestas clavó exactamente?") y porque correlaciona bien con la percepción humana de "respuesta correcta" en preguntas de respuesta corta y factual. Pero necesita un compañero más blando.

---

## 4. Token-level F1: la métrica estrella de SQuAD

El **F1 a nivel de token** es la métrica que dio fama a SQuAD ([Rajpurkar et al. 2016](/papers/squad-rajpurkar-2016)) y que sigue siendo el estándar de facto en QA extractivo. La idea: tratar predicción y gold answer como **bags of tokens** (bolsas de palabras), y medir cuántos tokens comparten mediante precision, recall y su media armónica.

Sea $\hat{T}$ el multiconjunto de tokens de la predicción normalizada y $T$ el de la gold answer normalizada. El número de tokens compartidos es:

$$\text{shared} = \sum_{w} \min\big(\text{count}_{\hat{T}}(w),\ \text{count}_{T}(w)\big)$$

(el $\min$ implementa el *clipping*: si la predicción repite una palabra más veces de las que aparece en gold, solo cuentan las que están en gold). Entonces:

$$P = \frac{\text{shared}}{|\hat{T}|}, \qquad R = \frac{\text{shared}}{|T|}, \qquad F_1 = 2 \cdot \frac{P \cdot R}{P + R}$$

donde $|\hat{T}|$ es el número de tokens de la predicción y $|T|$ el de la gold.

### Ejemplo numérico paso a paso

> **Pregunta**: ¿Qué causó la caída del puente?
>
> **Gold answer**: `"strong winds and structural fatigue"` → tras normalizar: `[strong, winds, and, structural, fatigue]` → pero "and" no es artículo, se conserva → 5 tokens.
>
> **Predicción**: `"the structural fatigue"` → tras normalizar (quitamos el artículo "the"): `[structural, fatigue]` → 2 tokens.

**Paso 1 — tokens compartidos.** Comparamos las bolsas:

- `structural`: aparece 1 vez en ambas → aporta $\min(1,1) = 1$.
- `fatigue`: aparece 1 vez en ambas → aporta $\min(1,1) = 1$.
- `strong`, `winds`, `and`: solo en gold → aportan 0.

$$\text{shared} = 1 + 1 = 2$$

**Paso 2 — precision.** De los 2 tokens predichos, ambos están en gold:

$$P = \frac{2}{2} = 1.0$$

**Paso 3 — recall.** De los 5 tokens gold, recuperamos 2:

$$R = \frac{2}{5} = 0.4$$

**Paso 4 — F1.**

$$F_1 = 2 \cdot \frac{1.0 \cdot 0.4}{1.0 + 0.4} = 2 \cdot \frac{0.4}{1.4} = \frac{0.8}{1.4} \approx 0.571$$

La predicción es léxicamente *precisa* (no metió palabras de más) pero *incompleta* (le faltaron "strong winds"), y el F1 de 0.571 lo refleja. EM, en cambio, habría dado 0 — ningún crédito por el solapamiento parcial. Esa diferencia es exactamente la razón de ser del token F1.

### Por qué se toma el max sobre las gold answers y se promedia

SQuAD recolecta **3 respuestas humanas** por pregunta (los anotadores no siempre seleccionan el span idéntico). Para no penalizar al modelo por elegir un span válido pero distinto del primer anotador, se computa el F1 contra **cada** gold answer y se toma el **máximo**:

$$F_1^{(i)} = \max_{g \in G_i} F_1\big(\hat{y}_i,\ g\big)$$

Luego se promedia sobre las $N$ preguntas del dataset:

$$F_1^{\text{dataset}} = \frac{1}{N} \sum_{i=1}^{N} F_1^{(i)}$$

El mismo esquema de *max-luego-promedio* aplica a EM. Tomar el max es la forma de reconocer que **cualquiera** de las gold answers es aceptable, así que al modelo se le da crédito por acercarse a la que más le convenga.

{{< concept-alert type="clave" >}}
El reporte oficial de SQuAD es siempre **el par (EM, F1)**. EM es la cota dura ("¿clavó la respuesta exacta?"), F1 la blanda ("¿cuánto solapó?"). En SQuAD 1.1 el desempeño humano fue **EM ≈ 82.3 / F1 ≈ 91.2**; los modelos transformer superaron el F1 humano hacia 2018-2019, lo que motivó SQuAD 2.0 (Sección 8).
{{< /concept-alert >}}

---

## 5. Normalización de respuestas

Tanto EM como F1 dependen críticamente de una etapa previa de **normalización**. SQuAD definió un procedimiento estándar que todo el campo adoptó, implementado en su script oficial de evaluación. Consta de cuatro pasos aplicados en orden:

1. **Lowercase**: convertir todo a minúsculas. `García` y `garcía` deben tratarse igual.
2. **Remove punctuation**: eliminar signos de puntuación. `1842.` → `1842`, `(EE.UU.)` → `eeuu`.
3. **Remove articles**: eliminar los artículos `a`, `an`, `the`. `the structural fatigue` → `structural fatigue`. (En la adaptación al español se removerían `el`, `la`, `los`, `las`, `un`, `una`, etc., aunque el script canónico es para inglés.)
4. **White-space tokenize / fix whitespace**: separar por espacios y colapsar espacios múltiples a uno solo.

En pseudo-código, la función canónica de SQuAD es:

```python
import re, string

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))
```

**Por qué importa**: sin normalización, `"The Beatles"` y `"the beatles"` y `"Beatles"` serían tres respuestas distintas, y EM caería artificialmente. La normalización canonicaliza diferencias *superficiales* que no afectan la corrección, de modo que la métrica mida lo que importa — el contenido — y no el formato. Es una decisión de diseño deliberada y documentada, lo que hace que los resultados de SQuAD sean **reproducibles y comparables** entre papers.

{{< concept-alert type="ojo" >}}
La normalización es un arma de doble filo. Remover artículos ayuda en inglés/español, pero en preguntas donde el artículo *es* la respuesta o cambia el significado (raro, pero existe), puede introducir ruido. Y la remoción de puntuación puede fusionar tokens incorrectamente en dominios técnicos (fórmulas, códigos, fechas con formato). Siempre conviene revisar que el normalizador estándar sea apropiado para tu dominio antes de confiar en los números.
{{< /concept-alert >}}

---

## 6. Mean Reciprocal Rank (MRR) y top-k retrieval accuracy

Cuando el sistema no produce una respuesta única sino una **lista ordenada de candidatos** — pasajes recuperados, respuestas candidatas rankeadas — necesitamos métricas de *ranking*. El caso paradigmático es el **passage retrieval** en QA de dominio abierto (open-domain QA): primero un retriever busca los pasajes más relevantes en un corpus enorme (Wikipedia entera), y luego un reader extrae la respuesta. La calidad del retriever se mide con métricas de ranking.

### Mean Reciprocal Rank (MRR)

El **MRR** premia que el primer resultado *relevante* aparezca lo más arriba posible en la lista. Para cada pregunta $i$, sea $\text{rank}_i$ la posición (1-indexed) del primer resultado correcto. El recíproco $1/\text{rank}_i$ vale 1 si está primero, $1/2$ si está segundo, $1/3$ si está tercero, etc. Se promedia:

$$\text{MRR} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{rank}_i}$$

(si ningún resultado de la lista es correcto, $1/\text{rank}_i$ se toma como 0).

**Ejemplo**: tres preguntas, y el primer pasaje correcto aparece en posiciones 1, 3 y 2 respectivamente.

$$\text{MRR} = \frac{1}{3}\left(\frac{1}{1} + \frac{1}{3} + \frac{1}{2}\right) = \frac{1}{3}\left(1 + 0.333 + 0.5\right) = \frac{1.833}{3} \approx 0.611$$

El MRR castiga fuertemente que el resultado correcto esté hundido en la lista: pasar de la posición 1 a la 2 baja la contribución de 1.0 a 0.5 (una caída de 0.5), mientras que pasar de la 9 a la 10 apenas la mueve de 0.111 a 0.100. Es la métrica adecuada cuando **solo importa encontrar un resultado correcto y cuanto antes mejor** — exactamente el caso de QA factual, donde basta un pasaje que contenga la respuesta.

### Top-k retrieval accuracy

Una métrica más simple y muy usada en retrieval (especialmente en [DPR](/papers/dpr-karpukhin-2020), Dense Passage Retrieval) es la **top-k accuracy**: la fracción de preguntas para las cuales **al menos uno** de los top-k pasajes recuperados contiene la respuesta gold.

$$\text{top-}k\text{ accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\big[\exists\, p \in \text{top-}k_i : \text{answer} \in p\big]$$

DPR reportó, por ejemplo, top-20 y top-100 accuracy sobre Natural Questions: cuántas veces la respuesta correcta está en los primeros 20 o 100 pasajes recuperados. Valores típicos de DPR rondan **top-20 ≈ 78-79%** y **top-100 ≈ 85-86%** en Natural Questions, frente al ~59% top-20 del baseline BM25 — la mejora que justificó el retrieval denso. A diferencia del MRR, top-k accuracy no distingue la posición *dentro* del top-k: solo importa si la respuesta está en el conjunto que se le pasará al reader. Ver el fundamento de [dense retrieval](/fundamentos/dense-retrieval) para el mecanismo de los bi-encoders que producen estos rankings.

---

## 7. BLEU y ROUGE para QA generativo

Cuando la respuesta es **texto libre generado** — no un span copiado del contexto sino una oración compuesta por el modelo — EM y token F1 ya no encajan bien (una buena respuesta generativa rara vez coincide token a token con la referencia). Aquí el campo toma prestadas las métricas de generación de texto: **ROUGE** y **BLEU**.

El benchmark canónico es **MS MARCO** ([Nguyen et al. 2016](/papers/ms-marco-nguyen-2016)), construido a partir de consultas reales de Bing, donde la tarea de QA generativo pide producir una respuesta en lenguaje natural a partir de pasajes recuperados. MS MARCO reporta oficialmente **ROUGE-L** y **BLEU-1** como métricas principales:

- **ROUGE-L** mide la *longest common subsequence* entre respuesta generada y referencia — captura solapamiento de contenido con flexibilidad de orden. Es recall-oriented, apropiado para verificar que la respuesta *cubra* la información de la referencia. Ver el fundamento dedicado de [ROUGE metric](/fundamentos/rouge-metric) para el detalle completo de la familia ROUGE-N/L/W/S.
- **BLEU** mide precision de n-gramas con *brevity penalty* — heredado de traducción automática, premia que lo generado sea correcto a nivel de n-gramas.

**Limitaciones para respuestas cortas**: tanto BLEU como ROUGE fueron diseñadas para textos largos (traducciones, resúmenes). En respuestas cortas de QA (una o dos oraciones) sufren especialmente:

- **Alta varianza**: con pocos tokens, un solo n-grama que no matchea cambia drásticamente el score.
- **BLEU y la brevity penalty**: respuestas correctas pero muy cortas reciben penalización de brevedad injusta.
- **Penalizan paráfrasis válidas**: `"murió en 1991"` vs `"falleció en 1991"` solapan poco a nivel de n-grama aunque sean equivalentes — el mismo problema léxico-vs-semántico que aqueja a estas métricas en summarization.
- **ROUGE-L premia overlap aunque la respuesta sea factualmente errónea** si comparte palabras con la referencia.

{{< concept-alert type="ojo" >}}
Usar ROUGE-L/BLEU en QA generativo es una concesión pragmática, no una solución ideal. Son baratas, deterministas y comparables con literatura previa, pero correlacionan débilmente con la corrección *factual* de una respuesta. Por eso los benchmarks generativos modernos las complementan con métricas semánticas (BERTScore) o evaluación humana / LLM-as-judge (Sección 9).
{{< /concept-alert >}}

---

## 8. Métricas con abstención (SQuAD 2.0)

SQuAD 1.1 tenía un supuesto fuerte: **toda pregunta tiene respuesta en el contexto**. Esto es irreal — un sistema robusto debe saber *cuándo no responder*. [SQuAD 2.0](/papers/squad2-rajpurkar-2018) (Rajpurkar et al. 2018) agregó ~50.000 preguntas **deliberadamente no respondibles**, escritas por anotadores para parecer respondibles pero sin respuesta real en el pasaje. El sistema debe, para esas preguntas, **abstenerse** (predecir "no answer").

### Cómo se puntúa la abstención

La gold answer de una pregunta no respondible es la cadena vacía (el span "no answer"). EM y F1 se computan exactamente igual que antes, pero ahora:

- **Pregunta no respondible + el sistema se abstiene** (predice "") → la predicción vacía coincide con la gold vacía → **EM = 1, F1 = 1**.
- **Pregunta no respondible + el sistema responde algo** → no coincide con la gold vacía → **EM = 0, F1 = 0**.
- **Pregunta respondible + el sistema se abstiene** → **EM = 0, F1 = 0** (falló en responder lo que debía).
- **Pregunta respondible + responde bien** → EM/F1 como en SQuAD 1.1.

Así, la misma fórmula de EM/F1 ahora premia *tanto* responder correctamente *como* callar correctamente. El desempeño humano en SQuAD 2.0 cayó a **EM ≈ 86.8 / F1 ≈ 89.5**, y los modelos de 2016 que dominaban 1.1 se desplomaban en 2.0 — porque no sabían abstenerse.

### El umbral de confianza

¿Cómo decide un modelo abstenerse? Produce, además del mejor span candidato, un **score de "no answer"** (típicamente el logit de que la respuesta esté en la posición especial `[CLS]`, o un score comparativo entre "hay respuesta" y "no hay respuesta"). Sea $s_{\text{best span}}$ el score del mejor span y $s_{\text{null}}$ el score de no-respuesta. El modelo se abstiene si:

$$s_{\text{null}} - s_{\text{best span}} > \tau$$

donde $\tau$ es un **umbral de confianza** calibrado sobre el conjunto de desarrollo para maximizar el F1. Subir $\tau$ hace al modelo más reacio a abstenerse (responde más, arriesga más falsos positivos); bajarlo lo hace más conservador (se abstiene más, arriesga perder preguntas respondibles). El campo reporta a veces curvas de F1 en función de $\tau$, análogas a curvas precision-recall.

---

## 9. Limitaciones generales y métricas modernas

Todas las métricas de las secciones anteriores — accuracy, EM, F1, MRR, ROUGE, BLEU — comparten un pecado original: son **léxicas**, no semánticas. Comparan superficie de tokens, no significado. Esto produce dos errores sistemáticos:

- **Falsos negativos**: respuestas correctas escritas distinto a la gold reciben score bajo. `"un físico nacido en Alemania"` vs `"físico alemán"` solapan poco léxicamente pero son equivalentes.
- **Falsos positivos**: respuestas que comparten palabras con la gold pero son factualmente erróneas reciben crédito.

Las métricas modernas intentan cerrar el gap léxico-semántico:

- **BERTScore** (Zhang et al. 2020): en lugar de matchear tokens exactos, computa similitud coseno entre los *embeddings* contextuales (BERT) de los tokens de predicción y referencia. Reconoce que `murió` y `falleció` tienen embeddings cercanos, dando crédito a paráfrasis. Se reporta como precision/recall/F1 sobre el matching greedy de embeddings.
- **LLM-as-judge**: usar un LLM potente (GPT-4, Claude) como evaluador, pidiéndole que juzgue si la respuesta del sistema es correcta dada la pregunta y la referencia. Captura corrección factual y semántica que ninguna métrica léxica alcanza, a costa de ser caro, no determinista y sensible al prompt. Es el estándar emergente para QA generativo y RAG abierto.
- **Métricas de QA factual / faithfulness**: en QA generativo y RAG, no basta con que la respuesta se parezca a la referencia — debe estar *soportada* por los documentos recuperados. Surgen métricas como *answer faithfulness* (¿la respuesta se deriva de los pasajes?) y *context relevance* (¿los pasajes recuperados son pertinentes?), centrales en frameworks de evaluación de RAG (RAGAS y similares).

{{< concept-alert type="clave" >}}
Pese a todo, **Exact Match y token F1 siguen siendo el estándar de reporte en QA extractivo**, igual que ROUGE sigue siéndolo en summarization. La razón es la misma: son baratas, deterministas, reproducibles y comparables con una década de literatura. La práctica moderna recomendada es **reportar EM/F1 (o ROUGE-L/BLEU si es generativo) + una métrica semántica (BERTScore) + para casos críticos, LLM-as-judge o evaluación humana en muestra.** Ninguna métrica única es suficiente.
{{< /concept-alert >}}

### Evaluación de RAG

El auge de **Retrieval-Augmented Generation (RAG)** — recuperar pasajes y luego generar la respuesta con un LLM — hereda *dos* problemas de evaluación a la vez: el del retriever (Sección 6: MRR, top-k accuracy, recall@k) y el del generador (Secciones 7 y 9: ROUGE, BERTScore, LLM-as-judge, faithfulness). La evaluación de RAG es por tanto *compuesta*: se mide la calidad del retrieval **y** la calidad de la generación condicionada al retrieval, porque un sistema puede fallar en cualquiera de las dos etapas (recuperar mal, o recuperar bien pero generar una respuesta no soportada por lo recuperado).

---

## 10. Resumen ejecutivo

| Métrica | Formato de QA | Qué mide | Fórmula clave |
| --- | --- | --- | --- |
| **Accuracy** | Cerrado (cloze, MC) | Fracción de aciertos exactos | $\frac{\text{correct}}{\text{correct}+\text{incorrect}}$ |
| **Exact Match** | Extractivo | Coincidencia exacta tras normalización | $\max_g \mathbb{1}[\text{norm}(\hat{y})=\text{norm}(g)]$ |
| **Token F1** | Extractivo | Solapamiento parcial de tokens | $2\frac{PR}{P+R}$ sobre bags of tokens |
| **MRR** | Retrieval / ranking | Posición del primer resultado correcto | $\frac{1}{N}\sum \frac{1}{\text{rank}_i}$ |
| **Top-k accuracy** | Retrieval | ¿Respuesta en los top-k pasajes? | $\frac{1}{N}\sum \mathbb{1}[\text{ans} \in \text{top-}k]$ |
| **ROUGE-L / BLEU** | Generativo | Solapamiento de n-gramas / LCS | ver [ROUGE metric](/fundamentos/rouge-metric) |
| **EM/F1 con null** | Extractivo + abstención | Acertar respuesta o callar bien | EM/F1 con gold vacía + umbral $\tau$ |
| **BERTScore / LLM-judge** | Generativo / semántico | Similitud / corrección semántica | cosine sim de embeddings / juicio LLM |

**Reglas prácticas**:

1. Identifica el **formato de respuesta** antes de elegir métrica.
2. Para extractivo, reporta **EM + F1** (estándar SQuAD), con la normalización canónica.
3. Para retrieval, reporta **top-k accuracy** (y MRR si la posición importa).
4. Para generativo, reporta **ROUGE-L/BLEU + BERTScore**, y LLM-as-judge para casos críticos.
5. Recuerda que toda métrica léxica subestima paráfrasis y sobreestima overlap espurio — complementa con semántica.

---

## Recursos relacionados

- [Clase 24](/clases/clase-24) — la clase del curso donde se estudia QA y su evaluación.
- [Question Answering](/fundamentos/question-answering) — fundamento transversal de la tarea de QA.
- [Machine Reading Comprehension](/fundamentos/machine-reading-comprehension) — comprensión lectora y QA extractivo.
- [ROUGE metric](/fundamentos/rouge-metric) — detalle completo de la familia ROUGE usada en QA generativo.
- [Dense Retrieval](/fundamentos/dense-retrieval) — bi-encoders y el retriever cuya calidad mide MRR/top-k.
- [SQuAD (Rajpurkar et al. 2016)](/papers/squad-rajpurkar-2016) — origen de EM + token F1.
- [SQuAD 2.0 (Rajpurkar et al. 2018)](/papers/squad2-rajpurkar-2018) — abstención y preguntas no respondibles.
- [MS MARCO (Nguyen et al. 2016)](/papers/ms-marco-nguyen-2016) — QA generativo evaluado con ROUGE-L y BLEU.
- [DPR (Karpukhin et al. 2020)](/papers/dpr-karpukhin-2020) — top-k retrieval accuracy en dominio abierto.

*Última actualización: 2026-06-07.*
