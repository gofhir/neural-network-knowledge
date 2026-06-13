---
title: "Lab 24 - Question Answering: Extractivo (BERT) y Generativo (T5/BART)"
weight: 240
sidebar:
  open: true
---

**Profesor:** Vladimir Araujo · **Basado en:** spark64.com/machine-comprehension
**Fecha:** Junio 2026
**Notebooks origen:** `clase_24/material/Laboratorio/QA_BERT_Spanish.ipynb` (42 celdas) · `QA_EncoderDecoder_Spanish.ipynb` (36 celdas)
**Notebooks ejecutados:** Parte 1 — [lab24-bert.ipynb](/notebooks/lab24-bert.ipynb) · [HTML](/notebooks-html/lab24-bert.html) · Parte 2 — [lab24-encoderdecoder.ipynb](/notebooks/lab24-encoderdecoder.ipynb) · [HTML](/notebooks-html/lab24-encoderdecoder.html)

## Encuadre

Laboratorio en dos partes que recorren los **dos paradigmas de Question Answering** en **español**, replicando para QA el mismo eje extractivo-vs-generativo de [lab-22 (summarization)](/laboratorios/lab-22) y clasificación-vs-generación de [lab-23 (VQA)](/laboratorios/lab-23). La **Parte 1** usa [BERT](/fundamentos/bert) (BETO, el BERT en español de la U. de Chile) para QA **extractivo** sobre SQuAD-es — *localizar* un span literal del contexto. La **Parte 2** usa [T5/BART](/fundamentos/t5-encoder-decoder) (BARTO y T5S, los modelos seq2seq en español de [Araujo et al.](/papers/seq2seq-spanish-araujo-2024)) para QA **generativo** sobre SQAC — *generar* la respuesta token a token.

| | Parte 1 — Extractivo (BERT) | Parte 2 — Generativo (T5/BART) |
|---|---|---|
| Arquitectura | Encoder-only (BETO) | Encoder-decoder (T5S / BARTO) |
| Qué hace con la respuesta | **Localiza** un span | **Genera** texto nuevo |
| Salida | Posiciones (start, end) → fragmento literal | Tokens uno a uno → texto libre |
| Softmax sobre | Posiciones del contexto | Todo el vocabulario |
| Input | `[CLS] Q [SEP] C [SEP]` + segment ids | `"question: Q context: C"` (text-to-text) |
| Dataset | SQuAD-es v2 (**traducido** automáticamente) | SQAC (**nativo** en español) |
| ¿Puede inventar palabras? | Nunca (anclado al texto) | Sí (riesgo de alucinación) |
| Abstención ("no sé") | Sí (span nulo `[CLS]`, SQuAD v2.0) | No la aprende (SQAC sin unanswerable) |
| Métrica | Exact Match / F1 | ROUGE |

El recorrido sigue las secciones de ambos notebooks:

1. **Arquitectura extractiva** (P1): span prediction de BERT (vectores S y E), setup, SQuAD-es traducido, entrenamiento con BETO.
2. **Inferencia extractiva y abstención** (P1): pipeline `run_prediction`, ejemplo de Quito, el `empty` de SQuAD v2.0.
3. **Actividad, experimento propio y atención** (P1): experimento FHIR (falso negativo de abstención), Verdadero/Falso, visualización de atención con BertViz.
4. **Arquitectura generativa** (P2): encoder-decoder, SQAC nativo, pipeline seq2seq con T5S/BARTO.
5. **Inferencia generativa** (P2): ejemplo de Bélgica, la alucinación cuando el modelo no sabe abstenerse.
6. **Actividad y comparación de paradigmas** (P2): experimento FHIR generativo vs extractivo sobre el mismo contexto, Verdadero/Falso.

## Resultados consolidados

### Ejemplo de Quito — extractivo (Parte 1, BETO + SQuAD-es v2)

| Pregunta | Respuesta | Veredicto |
|---|---|---|
| ¿Cuál es la población de Quito? | `2 millones` | ✅ |
| ¿En qué provincia esta ubicado Quito? | `Pichincha` | ✅ |
| ¿Cuál es la cápital más antigua de Sudamérica? | `Quito` | ✅ (correferencia, tolera typo) |
| ¿Qué tan buena es la comida en Ecuador? | `empty` | ✅ **abstención correcta** |

### Ejemplo de Bélgica — generativo (Parte 2, T5S)

| Pregunta | Respuesta | Veredicto |
|---|---|---|
| ¿Cuál es la población de Bélgica? | `11.754.004` | ✅ |
| ¿En qué parte de Europa esta ubicado? | `en el noroeste europeo` | ✅ (reformula, firma generativa) |
| ¿Cuál es la ciudad más poblada? | `amberes` | ✅ (Amberes ≠ Bruselas) |
| ¿Cuál es la cápital de Alemania? | `11.754.004` | ❌ **alucinación** (copió la población) |

### Experimento propio: mismo contexto FHIR, dos paradigmas

| Pregunta | BERT extractivo (P1) | T5 generativo (P2) |
|---|---|---|
| ¿Qué recurso almacena los datos demográficos? | `Patient` ✅ | `el estándar hl7 fhir` ❌ |
| ¿Quién publicó FHIR? | `Health Level Seven International` ✅ | `la organización health level seven international` ✅ |
| Pregunta-trampa sin respuesta | `empty` (se abstuvo) ✅ | alucina ❌ |

→ **Hallazgos propios:** (1) el extractivo se abstuvo de una pregunta *respondible* (falso negativo) por vocabulario técnico fuera de distribución, y solo reformulando *¿Qué organización?* → *¿Quién?* se recuperó la respuesta — ajustar el umbral no bastó; (2) el generativo falló en extracción de entidad precisa (`Patient`) donde el extractivo acertó, y alucinó en la trampa en vez de abstenerse. **Ningún paradigma domina: tienen perfiles de error opuestos.**

## Bloques del lab

{{< cards >}}
  {{< card link="arquitectura-extractiva" title="P1 · Arquitectura: BERT extractivo" subtitle="Span prediction (vectores S y E), setup, SQuAD-es traducido, entrenamiento con BETO" icon="academic-cap" >}}
  {{< card link="inferencia-extractiva" title="P1 · Inferencia extractiva y abstención" subtitle="Pipeline run_prediction, ejemplo de Quito, el empty de SQuAD v2.0" icon="academic-cap" >}}
  {{< card link="actividades-extractivo" title="P1 · Actividad, experimento FHIR y BertViz" subtitle="Falso negativo de abstención, Verdadero/Falso, visualización de atención" icon="academic-cap" >}}
  {{< card link="arquitectura-generativa" title="P2 · Arquitectura: QA generativo (T5/BART)" subtitle="Encoder-decoder, SQAC nativo, pipeline seq2seq con T5S/BARTO" icon="academic-cap" >}}
  {{< card link="inferencia-generativa" title="P2 · Inferencia generativa" subtitle="Ejemplo de Bélgica, la alucinación 11.754.004, cuando el modelo no sabe abstenerse" icon="academic-cap" >}}
  {{< card link="actividades-generativo" title="P2 · Actividad y comparación de paradigmas" subtitle="Generativo vs extractivo sobre el mismo contexto FHIR, Verdadero/Falso" icon="academic-cap" >}}
{{< /cards >}}

## Papers de este lab

{{< cards >}}
  {{< card link="/papers/seq2seq-spanish-araujo-2024" title="Seq2Seq Spanish PLMs (2024)" subtitle="Araujo et al. — BARTO y T5S, los modelos generativos de la Parte 2. El profesor del lab es el primer autor" icon="document-text" >}}
  {{< card link="/papers/squad-rajpurkar-2016" title="SQuAD (2016)" subtitle="Rajpurkar et al. — QA extractivo span-based, EM/F1" icon="document-text" >}}
  {{< card link="/papers/squad2-rajpurkar-2018" title="SQuAD 2.0 (2018)" subtitle="Rajpurkar et al. — preguntas sin respuesta, abstención" icon="document-text" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/question-answering" title="Fundamento: Question Answering" subtitle="Taxonomía, extractivo vs generativo, abstención" icon="book-open" >}}
  {{< card link="/fundamentos/machine-reading-comprehension" title="Fundamento: Machine Reading Comprehension" subtitle="P+Q→A, span prediction, attentive readers, BiDAF" icon="book-open" >}}
  {{< card link="/fundamentos/qa-evaluation-metrics" title="Fundamento: Métricas de QA" subtitle="Exact Match, F1, ROUGE, abstención" icon="book-open" >}}
  {{< card link="/fundamentos/bert" title="Fundamento: BERT" subtitle="Base de la QA extractiva (BETO)" icon="book-open" >}}
  {{< card link="/fundamentos/t5-encoder-decoder" title="Fundamento: T5 y Encoder-Decoder" subtitle="Base de la QA generativa (T5S, BARTO)" icon="book-open" >}}
{{< /cards >}}

## Cross-links

{{< cards >}}
  {{< card link="/clases/clase-24" title="Clase 24 - Teoría" subtitle="Historia de QA, MRC, Stanford Attentive Reader, BiDAF, BERT, Generative QA, métricas" icon="academic-cap" >}}
  {{< card link="/clases/clase-24/profundizacion" title="Profundización" subtitle="Red genérica MRC, BiDAF attention flow, span prediction, EM/F1, MRR" icon="beaker" >}}
  {{< card link="/dominios/texto" title="Dominio: Texto / NLP" subtitle="Timeline NLP: de los attentive readers a los LLM" icon="globe-alt" >}}
  {{< card link="/laboratorios/lab-23" title="Lab 23 - VQA con BLIP (anterior)" subtitle="Clasificación vs generación, alucinación en VLMs" icon="academic-cap" >}}
{{< /cards >}}

---

> **Estado:** Lab completo. Cubre las 42 + 36 celdas de ambos notebooks con 6 páginas temáticas. Incluye experimentos propios sobre dominio FHIR que exhiben los perfiles de error opuestos de cada paradigma: un falso negativo de abstención en el extractivo (recuperable solo reformulando la pregunta, no ajustando el umbral) y una alucinación del generativo ante una pregunta-trampa. El entrenamiento se omitió (opcional, ~1h, y la sesión de Colab se reinició perdiendo el output efímero de `/content`); la inferencia usa modelos ya fine-tuneados del Hub. Notebooks ejecutados en Colab.
