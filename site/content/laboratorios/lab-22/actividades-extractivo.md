---
title: "P1 — Actividades (1 + Verdadero/Falso)"
weight: 4
---

> **Celdas 38-47 del notebook (Parte 1).** Las preguntas conceptuales que cierran el extractivo, con justificaciones ancladas en lo observado.

## Actividad 1 (celdas 39-41)

### ¿Cuál es el output del modelo para una oración?

Un **único escalar $\hat{Y}_i \in [0,1]$** (salida sigmoide): la **probabilidad de que esa oración pertenezca al resumen**. Mecánicamente: BERT procesa el documento, se toma el vector del `[CLS]` que precede a la oración, el clasificador lo colapsa a ese escalar. La salida completa es `sent_scores` de forma `[batch, num_oraciones]` — **un score por oración, no por palabra**. Resumen extractivo = clasificación binaria por oración. No genera texto.

### ¿Cómo se determina qué oraciones van al resumen final?

Tres pasos (los de la [inferencia](inferencia-extractiva)):
1. **Ranking** de oraciones por score descendente.
2. **Trigram blocking**: agregar una oración solo si no comparte trigrama con las ya elegidas.
3. **Length cap**: detenerse en 3 oraciones.

No hay umbral fijo sobre el score; hay **ranking + cupo fijo + filtro anti-redundancia**.

### Alternativa para seleccionar frases en base al gold standard

⚠️ Esta pregunta es sobre **cómo construir las etiquetas (el oracle)**, no cómo selecciona el modelo. El método estándar es el oracle greedy por ROUGE. Una **alternativa válida**:

> Usar **similitud semántica con embeddings** en vez de solapamiento de n-gramas: para cada oración del documento, calcular la similitud coseno entre su embedding (Sentence-BERT) y el del gold; etiquetar como positivas las de mayor similitud. Captura paráfrasis que ROUGE pierde — atacaría justo el ruido de captions que **observamos**. Es la idea detrás de BERTScore y MatchSum.

Otras alternativas: selección exacta vía ILP (en lugar del greedy sub-óptimo), o MMR explícito (penalizar similitud con lo ya elegido desde la construcción de etiquetas).

## Verdadero / Falso (celdas 43-46)

Las cuatro afirmaciones con su respuesta y razón:

| # | Afirmación | Respuesta | Razón |
|---|---|---|---|
| 43 | Genera texto que no está en la entrada | **Falso** | Es extractivo: solo copia oraciones literales (las predicciones eran idénticas al source) |
| 44 | La entrada es todo el texto a resumir | **Verdadero** | Por diseño, el modelo procesa el **documento completo** de una vez para puntuar cada oración en contexto de las demás (vs. clasificar oraciones aisladas). El límite de 512 tokens es una limitación de implementación, no contradice el principio |
| 45 | Fácil cambiar BERT por LSTM pre-entrenada | **Falso** | La arquitectura depende de los `[CLS]` por oración y los segment embeddings; no hay una LSTM pre-entrenada equivalente para enchufar. El "fácilmente" lo hace falso |
| 46 | No tuvo pre-entrenamiento | **Falso** | BERT está pre-entrenado (MLM + NSP); es la tesis del método. El warmup Noam de 10k steps existe precisamente para no destruir esas representaciones |

### Nota sobre la afirmación 44

Es la más sutil. Tiene dos lecturas:
- **Conceptual/arquitectónica (la intención → Verdadero):** la característica distintiva de BertSum es procesar el documento completo de una vez, para que cada oración se puntúe en contexto de todas las demás.
- **Técnica/literal (Falso):** BERT trunca a 512 tokens, así que en documentos largos la entrada efectiva no es *literalmente* todo el texto.

La pregunta evalúa el **principio de diseño** ("para efectos del modelo"), no el detalle del límite de 512. Contrastada con las otras tres (claramente Falso), la 44 es la "Verdadero" del set.

---

**Anterior:** [inferencia y trigram blocking](inferencia-extractiva) · **Siguiente:** [abstractivo con T5](abstractivo-t5)
