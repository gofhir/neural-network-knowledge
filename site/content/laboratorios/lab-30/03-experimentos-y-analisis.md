---
title: "Experimentos propios y análisis"
weight: 3
---

Más allá del recorrido del notebook, se construyó un mini-estudio de comportamiento del modelo entrenado (~69% accuracy). Cinco experimentos que convergen en una sola tesis.

## Tesis

> **El "31% de error" del modelo es engañoso. Una parte sustancial no son errores de capacidad, sino artefactos de evaluación: ground-truth single-answer sobre preguntas multi-respuesta.**

| # | Experimento | Resultado | Qué reveló |
|---|---|---|---|
| 1 | Visualización de atención | Hop 1 difuso, Hop 2 picudo | El multi-hop reduce entropía: localiza → concentra |
| 2 | Búsqueda de 15 errores | 100% eran preguntas `describe X` | Modo de fallo sistemático, no aleatorio |
| 3 | Top-5 candidatos | Los 5 = facetas válidas; prob=1.0 en el #1 | El modelo recupera bien; falla la métrica + está sobreconfiado |
| 4 | Top-k accuracy | 0.581 / 0.741 / 0.799 | +16 pts top-1→top-3 cuantifica el multi-respuesta |
| 5 | Muestra aleatoria | 0.683 vs 0.581 | El dataset está ordenado por tipo de pregunta |

## Exp. 1 — Visualización de atención: cómo "razona" el modelo

Pregunta: *"can you give a few words describing what kidulthood is about?"*. Respuesta correcta `violence`, predicha `drama` (un fallo — pero `drama` también describe la película).

**Hop 1** — atención **difusa** (0.04–0.19), repartida entre ~10 entradas, todas con `kidulthood`. El modelo localiza la zona de la KB sobre la película:

![Heatmap hop 1 — atención difusa sobre entradas de Kidulthood](/laboratorios/lab-30/viz-describe-hop1.jpg)

**Hop 2** — la atención **colapsa** (~0.7) en una sola entrada tipo `__movie__`. La distribución pasa de entropía alta a baja: la firma del refinamiento multi-hop:

![Heatmap hop 2 — atención concentrada](/laboratorios/lab-30/viz-describe-hop2.jpg)

**Detalle clave:** la predicción final **no sale del hop 2 directamente**. Tras los 2 hops, `q_state` se scorea contra los embeddings de candidatos. Los hops construyen la representación; el output layer decide. La atención es el *proceso de lectura*; el scoring de candidatos es la *decisión*.

## Exp. 2 — Búsqueda de errores: un modo de fallo sistemático

Al listar los primeros 15 errores del modelo, **el 100% eran preguntas `describe X`**. No es ruido aleatorio. Dos categorías:

- **A (13/15) — descripción válida distinta del ground-truth:** `describe old school` → predijo `todd phillips` (director), ground-truth `comedy` (género). Ambas correctas.
- **B (2/15) — "trivial answer":** `describe across the bridge` → predijo el **título mismo** de la película, ground-truth `ken annakin`.

**Por qué `describe` y no `who directed`.** Las preguntas factuales tienen un **token de relación** (`directed`, `genre`, `year`) que ancla la atención al value correcto. `describe X` no especifica qué faceta querés. Sin orden de palabras (BoW) y sin tipo de relación, el modelo solo puede agarrarse de "qué value se parece más", que es ambiguo por construcción.

## Exp. 3 — Top-5 candidatos: el modelo recupera bien

Para cuatro errores `describe`, todo el top-5 resultó ser facetas válidas de la película:

| | #1 | #2 | #3 | #4 | #5 |
|---|---|---|---|---|---|
| old school | todd phillips | luke wilson | old school | **comedy** *(gt)* | 2003 |
| jesse james | western | henry hull | jesse james | 1939 | brian donlevy |
| san francisco | murder | san francisco | barbary coast | **clark gable** *(gt)* | gold |

Dos hallazgos:

1. **Sobreconfianza extrema** — prob=**1.000** en el #1 y ~0 en el resto. El score `q·candidato` sin temperatura satura el softmax. Un modelo bien calibrado para `describe X` debería repartir masa entre facetas. **Es un problema de calibración, no de conocimiento.**
2. **El "trivial answer" acecha** — el título de la película aparece en el top-3 sistemáticamente (entradas `__movie__` con value=título que matchean preguntas que repiten el título).

## Exp. 4 — Top-k accuracy: cuantificar el multi-respuesta

| Métrica (primeros 2000, sin shuffle) | Accuracy | Δ |
|---|---|---|
| Top-1 | 0.581 | — |
| Top-3 | 0.741 | **+0.160** |
| Top-5 | 0.799 | +0.058 |

Permitir solo dos posiciones más recupera **16 puntos**: ~16% de las preguntas tenían el ground-truth en posición #2-#3, con el modelo poniendo arriba otra respuesta válida. El retorno marginal decae (top-3→top-5 solo +5.8 pts), coherente con la sobreconfianza.

## Exp. 5 — El dataset está ordenado

| Evaluación | Top-1 |
|---|---|
| Primeros 2000 (sin shuffle) | 0.581 |
| Muestra aleatoria de 2000 | 0.683 |

+10 pts solo por barajar → el `test_dataset` está **agrupado por tipo de pregunta**, con los `describe X` (los más difíciles) al inicio. El 0.683 aleatorio ≈ 0.69 del test completo confirma la consistencia. **Lección de higiene experimental:** barajar siempre antes de evaluar slices.

## Dos modos de fallo distintos

Un caso de contraste muestra que no todo error es multi-respuesta. Una pregunta factual fácil acierta limpio:

`who directed the film men with brooms?` → **paul gross** ✓

![Heatmap — caso factual acertado](/laboratorios/lab-30/contraste-factual-hop2.jpg)

Pero un título **homónimo** genera un error genuino de desambiguación:

`who directed the film heat?` → predijo **michael mann**, correcta **paul morrissey**

![Heatmap — fallo por entidad homónima](/laboratorios/lab-30/contraste-heat-hop2.jpg)

Existen dos películas *Heat*: la de Michael Mann (1995, famosa) y la de Paul Morrissey (1972). El modelo, con BoW y sin desambiguación de entidad, atendió la entrada de la más prominente. Es un fallo **real** de capacidad — distinto del fallo de métrica de `describe`.

## Conexión con sistemas de matching (MDM, FHIR)

Caso de manual de **ground-truth incompleto** en evaluación de retrieval/matching:

- Si el gold standard marca un único match válido pero existen varios duplicados legítimos, precisión y recall medidos **subestiman** al sistema real.
- **Firma diagnóstica:** la brecha top-1 vs top-k. Si recuperás bien pero el top-1 exact-match es bajo, sospechá multi-validez en el ground-truth antes de culpar al modelo.
- La **sobreconfianza** del softmax sin temperatura es la razón por la que un scorer (p. ej. GBM) necesita **calibración** (Platt / isotónica) antes de fijar umbrales de decisión.

---

**Siguiente:** [Actividades 1-4 resueltas](../04-actividades).
