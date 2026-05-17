---
title: "Porter Stemmer"
weight: 50
math: true
---

{{< paper-card
    title="An algorithm for suffix stripping"
    authors="Porter"
    year="1980"
    venue="Program: Electronic Library and Information Systems 14(3)"
    pdf="/papers/porter-stemmer-1980.pdf" >}}
Describe un algoritmo simple y rápido para reducir palabras inglesas a su raíz (stem) por eliminación iterativa de sufijos. ~400 líneas en BCPL, procesaba 10,000 palabras en 8.1 segundos en un IBM 370/165. Igualó o superó al sistema de Cambridge (~10x más complejo) en la Cranfield 200 collection. **Sigue corriendo en producción en 2026** dentro de Apache Lucene, Elasticsearch, NLTK, y prácticamente todo motor de búsqueda IR que no usa Transformers.
{{< /paper-card >}}

---

## Contexto

A fines de los 70, los stemmers existentes (Lovins 1968 con ~260 reglas, Andrews 1971, Dawson 1974) eran enormes, difíciles de mantener y sensibles al orden de aplicación. Porter buscó algo **simple, rápido y suficientemente bueno** para mejorar recall en IR — sin pretender corrección lingüística.

La tesis filosófica del paper: *"It would not be at all obvious under what circumstances a suffix should be removed, even if we could exactly determine the suffixes of the words by automatic means."* Es decir, no buscaba un stemmer lingüísticamente correcto — buscaba uno que **mejorara recall en IR**, sabiendo que cualquier criterio de "corrección" es discutible.

---

## Ideas principales

### 1. La métrica m (measure)

Toda palabra puede escribirse como:

$$[C](VC)^m[V]$$

donde C es secuencia de consonantes, V de vocales, y `m` es el número de repeticiones del patrón VC. Ejemplos:

| Palabra | m |
|---|---|
| TR, TREE, BY | 0 |
| TROUBLE, OATS, IVY | 1 |
| TROUBLES, PRIVATE, OATEN | 2 |

`m` es **una medida cruda de longitud silábica del stem**. El algoritmo elimina un sufijo solo si el stem que queda tiene suficiente `m`. Esto evita over-stripping.

### 2. Las reglas tipo (condición) S1 → S2

```
(m > 1) EMENT →
```

Significa: si la palabra termina en `EMENT` y el stem antes tiene `m > 1`, eliminar `EMENT`. Por ejemplo `REPLACEMENT → REPLAC` (porque `REPLAC` tiene m=2), pero `PAVEMENT → PAVE` (NO se aplica porque `PAV` tiene m=1).

Las condiciones pueden incluir tests sobre el stem (`*S` termina en S, `*v*` contiene vocal, `*d` termina en doble consonante, `*o` termina en cvc no-WXY) y combinarse con `and`, `or`, `not`.

### 3. Los 5 pasos secuenciales

| Paso | Maneja |
|---|---|
| 1a | Plurales (`-SSES`, `-IES`, `-SS`, `-S`) |
| 1b | Gerundios y participios (`-EED`, `-ED`, `-ING`) con post-procesamiento (`AT→ATE`, `BL→BLE`, doble consonante final) |
| 1c | Y → I después de vocal |
| 2 | Sufijos derivacionales largos (`-ATIONAL→ATE`, `-IZATION→IZE`, `-FULNESS→FUL`, etc.) |
| 3 | Más derivacionales (`-ICATE→IC`, `-ATIVE→`, `-ALIZE→AL`, etc.) |
| 4 | Sufijos derivacionales cortos con `m > 1` (`-AL`, `-ANCE`, `-MENT`, `-ENT`, `-IZE`, etc.) |
| 5a | Eliminar E final si `m > 1`, o si `m=1 and not *o` |
| 5b | Eliminar consonante doble final si termina en L |

En cada paso solo se aplica la regla con el sufijo más largo que matchee.

### 4. Composición de pasos

`GENERALIZATIONS` (14 letras) se reduce a `GENER` (5 letras) en 4 pasos:

```
GENERALIZATIONS
  → Paso 1 (S):           GENERALIZATION
  → Paso 2 (IZATION→IZE):  GENERALIZE
  → Paso 3 (ALIZE→AL):     GENERAL
  → Paso 4 (AL):           GENER
```

---

## Resultados experimentales

Tarea: **Cranfield 200 collection** (200 documentos científicos, 42 queries, evaluación IR estándar de la época).

| Recall | Sistema Cambridge anterior — Precision | Sistema Porter — Precision |
|---|---|---|
| 0 | 57.24 | 58.60 |
| 20 | 52.85 | 53.92 |
| 40 | 42.20 | 39.39 |
| 60 | 32.86 | 33.18 |
| 80 | 27.15 | 27.52 |
| 100 | 24.59 | 25.85 |

**Iguala o supera** al sistema previo en todos los recall levels. La simplicidad **no fue un sacrificio de calidad**.

Reducción del vocabulario: sobre 10,000 palabras, Porter las reduce a **6,370 stems únicos** (-36%). Eso es la **Ley de Heaps** en acción — comprimiendo morfología.

---

## Limitaciones reconocibles

Porter es honesto sobre los errores:

- **Conflations deliberadas pero discutibles**: `RELATE` y `RELATIVITY` se conflate juntas aunque semánticamente son distantes.
- **Spelling changes**: `DECEIVE`/`DECEPTION`, `RESUME`/`RESUMPTION`, `INDEX`/`INDICES` no se conflate.
- **Inconsistencias por design**: `PRELATE` (m=1) preserva `-ATE`, pero `ARCHPRELATE` (m=2) lo elimina. Sin reconocer prefijos.

Solo inglés. No maneja verbos irregulares (`WENT/GO`, `MICE/MOUSE`). Stems no siempre son palabras reales (`relat`, `intellig`, `entri`) — feos para usuario final, fine para BoW.

---

## Por qué importa hoy

- **~15,000 citas en Google Scholar** a mayo de 2026. Entre los 20 papers más citados de NLP de todos los tiempos.
- **Apache Lucene** (motor detrás de Elasticsearch, Solr) usa Porter como stemmer default. Eso significa que **miles de motores de búsqueda corren Porter en producción cada día**, 45 años después de publicado.
- **NLTK** y **scikit-learn** incluyen Porter (`nltk.stem.porter.PorterStemmer`).
- Para producción seria en inglés deberías usar **Snowball/Porter2 (2001)** — el mismo Porter mejorado, también en NLTK como `SnowballStemmer('english')`.
- Es **la baseline obligada** en cualquier benchmark IR. Si tu sistema basado en BERT no supera a BoW + Porter, algo está mal.

En la era de Transformers, **subword tokenization** (BPE, WordPiece, SentencePiece) hace el trabajo de Porter de manera más sofisticada. Pero para pipelines clásicos sin GPU, Porter sigue siendo competitivo por razones de latencia, costo y determinismo.

---

## Notas y enlaces

- El paper es **inusualmente corto** (4 páginas en 2 columnas) y autocontenido — podés leerlo en una tarde y reimplementar el algoritmo.
- Sucesor directo: **Snowball** (Porter 2001) — DSL para escribir stemmers en múltiples idiomas. Incluye `SnowballStemmer('spanish')` para tu trabajo en NLP clínico español.
- Alternativa más agresiva: **Lancaster stemmer** (Paice 1990) — sobre-strippa (`aviation→av`). Casi nadie lo usa.

Ver fundamentos: [Tokenización clásica](/fundamentos/tokenizacion-clasica) · [Bag of Words](/fundamentos/bag-of-words).
