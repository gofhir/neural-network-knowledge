# Cierre — Análisis de experimentos adicionales (Lab 30: KV-MemNN sobre WikiMovies)

Mini-estudio del comportamiento del modelo KV-MemNN, construido con 5 experimentos
ejecutados sobre el notebook (más allá de las celdas base), usando el checkpoint
pre-entrenado (`best_state.pt`, ~69% accuracy en test).

---

## Resumen de los 5 experimentos

| # | Experimento | Resultado | Qué reveló |
|---|-------------|-----------|------------|
| 1 | Visualización ejemplo 8801 (`describe kidulthood` → violence ✓) | Hop 1 difuso (atención máx 0.19), Hop 2 picudo (0.7) | El multi-hop reduce entropía: localiza → concentra. La interpretabilidad por atención funciona. |
| 2 | Búsqueda de 15 errores | 100% eran preguntas `describe X` | Modo de fallo sistemático, no aleatorio. Un solo tipo de pregunta concentra los errores. |
| 3 | Top-5 candidatos (old school, jesse james, calamari union, san francisco) | Los 5 candidatos = facetas válidas; prob=1.0 en el #1 | El modelo recupera bien (todas las descripciones son ciertas); falla la métrica single-answer + está sobreconfiado. |
| 4 | Top-k accuracy (primeros 2000) | 0.581 / 0.741 / 0.799 | +16 pts de top-1→top-3 cuantifica el problema multi-respuesta. |
| 5 | Muestra aleatoria (2000) | 0.683 (vs 0.581) | El dataset está ordenado por tipo; el accuracy real ≈ 0.69 confirmado. |

---

## Tesis unificada

> **El "31% de error" del modelo es engañoso. Una parte sustancial no son errores de
> capacidad, sino artefactos de evaluación (ground-truth single-answer sobre preguntas
> multi-respuesta).**

Cadena de evidencia:

1. Los errores se concentran en `describe X` (exp. 2), preguntas subespecificadas que no
   indican qué faceta de la película se quiere.
2. Para esas preguntas, el modelo recupera correctamente TODAS las facetas (exp. 3:
   el top-5 entero es ficha técnica válida — director, actores, género, año, temas).
3. El preprocesamiento (celda 31, `first_answer = answer[0]`) fijó arbitrariamente UNA
   faceta como ground-truth; la métrica top-1 exact-match castiga al modelo por elegir
   otra faceta igualmente válida.
4. El salto top-1 → top-3 de +16 puntos (exp. 4) mide directamente cuánta "tasa de error"
   es en realidad multi-validez.

---

## Detalle por experimento

### Exp. 1 — Visualización de atención (ejemplo 8801)
Pregunta: *"can you give a few words describing what kidulthood is about?"* → correcta y
predicha: `violence`.
- **Hop 1:** atención difusa (rango 0.04–0.19) repartida entre ~10 entradas, todas con
  `kidulthood`. El modelo localiza la zona de la KB sobre la película.
- **Hop 2:** atención colapsa (~0.7) en una sola entrada. La distribución pasa de entropía
  alta a baja → firma del refinamiento multi-hop.
- La predicción final NO sale del hop 2 directamente: tras los 2 hops, `q_state` se
  scorea contra los embeddings de candidatos. Los hops construyen la representación;
  el output layer decide.

### Exp. 2 — Búsqueda de errores
Los primeros 15 errores fueron TODOS preguntas `describe X`. Dos categorías:
- **A (13/15):** el modelo predijo una descripción válida distinta del ground-truth
  arbitrario (ej.: old school → "todd phillips", gt "comedy"; jesse james → "western",
  gt "henry king").
- **B (2/15):** "trivial answer" — el modelo devolvió el TÍTULO de la película misma
  (ej.: "describe across the bridge" → predijo "across the bridge", gt "ken annakin").

### Exp. 3 — Top-5 candidatos por score

| Película | #1 | #2 | #3 | #4 | #5 |
|----------|----|----|----|----|----|
| old school | todd phillips (dir) | luke wilson (actor) | old school (título) | comedy (gt) | 2003 (año) |
| jesse james | western (género) | henry hull (actor) | jesse james (título) | 1939 (año) | brian donlevy (actor) |
| calamari union | 1985 (año) | asmo hurula (actor) | calamari union (título) | surreal (tema) | kari heiskanen (actor) |
| san francisco | murder (tema) | san francisco (título) | barbary coast (tema) | clark gable (gt) | gold (tema) |

- Cada candidato del top-5 es una descripción legítima → el modelo recupera bien.
- prob=1.000 en el #1 → sobreconfianza / mala calibración (producto punto sin temperatura
  satura el softmax).
- El título de la película aparece en el top-3 sistemáticamente (sesgo "trivial answer"
  del esquema key/value: entradas `__movie__` con value=título matchean preguntas que
  repiten el título).

### Exp. 4 — Top-k accuracy (primeros 2000, sin shuffle)
| Métrica | Accuracy | Δ |
|---------|----------|---|
| Top-1 | 0.581 | — |
| Top-3 | 0.741 | +0.160 |
| Top-5 | 0.799 | +0.058 |

+16 pts al permitir top-3: ~16% de las preguntas tenían el ground-truth en posición #2–#3.
El retorno marginal decae (top-3→top-5 solo +5.8 pts): el modelo concentra el acierto en
las primeras posiciones, coherente con la sobreconfianza.

### Exp. 5 — Muestra aleatoria
| Evaluación | Top-1 |
|------------|-------|
| Primeros 2000 (sin shuffle) | 0.581 |
| Muestra aleatoria de 2000 | 0.683 |

+10 pts solo por barajar → el `test_dataset` está ordenado por tipo de pregunta, con los
`describe X` (los más difíciles) agrupados al inicio. El 0.683 aleatorio ≈ 0.69 del test
completo confirma la consistencia de la evaluación.

---

## Hallazgos secundarios

- **Sobreconfianza / calibración:** prob=1.0 en el top-1; el softmax sobre `q_state·embed_c`
  sin temperatura no expresa la incertidumbre real entre facetas válidas.
- **Trivial answer:** el título de la película acecha en el top-3 y a veces gana → fuga
  del input a la respuesta.
- **Multi-hop verificado:** la atención difusa→picuda demuestra que los 2 hops refinan
  de verdad, no son decorativos.
- **Dataset no i.i.d. en orden:** agrupado por tipo. Higiene experimental: barajar siempre
  antes de evaluar slices.

---

## Conexión con sistemas de matching / record linkage (MDM, FHIR)

Caso de manual de **ground-truth incompleto** en evaluación de retrieval/matching:
- Si el gold standard marca un único match válido pero existen varios duplicados legítimos,
  precisión y recall medidos subestiman al sistema real.
- Firma diagnóstica: brecha top-1 vs top-k. Si recuperás bien pero el top-1 exact-match es
  bajo, sospechá multi-validez en el ground-truth antes de culpar al modelo.
- La sobreconfianza del softmax sin temperatura es la razón por la que un scorer (p. ej. GBM)
  necesita calibración (Platt / isotónica) antes de fijar umbrales de decisión.
