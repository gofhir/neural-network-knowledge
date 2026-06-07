---
title: "Lab 22 - Summarization: Extractivo (BertSum) y Abstractivo (T5)"
weight: 220
sidebar:
  open: true
---

**Profesor:** Felipe del Río · **Ayudante:** Bianca del Solar
**Fecha:** Mayo 2026
**Notebooks origen:** `clase_22/material/Laboratorio/Laboratorio 22 - Summarization - Parte {1,2}.ipynb` (48 + 84 celdas)
**Notebooks ejecutados:** Parte 1 — [lab22-parte-1.ipynb](/notebooks/lab22-parte-1.ipynb) · [HTML](/notebooks-html/lab22-parte-1.html) · Parte 2 — [lab22-parte-2.ipynb](/notebooks/lab22-parte-2.ipynb) · [HTML](/notebooks-html/lab22-parte-2.html)

## Encuadre

Laboratorio en dos mundos que recorren los **dos paradigmas clásicos del resumen automático**. La **Parte 1** usa [BertSum](/papers/bertsum-liu-2019) (BERT + clasificador de oraciones) para resumen **extractivo** — *seleccionar* oraciones literales del documento. La **Parte 2** usa [T5](/papers/t5-raffel-2020) (encoder-decoder seq2seq) para resumen **abstractivo** — *generar* texto nuevo que puede contener palabras no presentes en la fuente. El eje conceptual de todo el lab es esa dicotomía:

| | Parte 1 — Extractivo | Parte 2 — Abstractivo |
|---|---|---|
| Modelo | BertSum (BERT encoder + clasificador) | T5-small (encoder-decoder) |
| Qué hace | **Selecciona** oraciones existentes | **Genera** texto nuevo |
| Salida | Subconjunto del texto original | Tokens generados uno a uno |
| ¿Puede inventar palabras? | Nunca (fiel por construcción) | Sí (riesgo de alucinación) |
| Capa que decide | Clasificador binario por oración | Decoder autoregresivo |
| Métrica | [ROUGE](/fundamentos/rouge-metric) (vía pyrouge/Perl) | ROUGE (vía evaluate/Python) |

El recorrido sigue las actividades del notebook:

1. **Arquitectura extractiva** (P1, celdas 5-30): cómo BERT se modifica para producir representaciones a nivel de oración (`[CLS]` por oración + interval segment embeddings) y la clase `Args` que emula la CLI original.
2. **Estrategia de entrenamiento de BertSum** (análisis): oracle greedy por ROUGE, loss BCE, scheduler Noam con warmup, gradient accumulation, model averaging — por qué está diseñado así.
3. **Inferencia extractiva** (P1, celdas 31-37): dataloader, forward pass, ranking + **trigram blocking**, y análisis de predicciones reales sobre CNN/DailyMail.
4. **Actividades extractivo** (P1, celdas 38-47): Actividad 1 + 4 Verdadero/Falso con justificaciones.
5. **Abstractivo con T5** (P2, celdas 5-15): framework **text-to-text**, span corruption, t5-small, primera generación con beam search.
6. **Generación cualitativa** (P2, celdas 16-27): Actividad 2 sobre una noticia real (DermaSensor/FDA) y párrafos de un libro (*Pride and Prejudice*) — saliencia, alucinación y domain shift.
7. **Parámetros de decodificación** (P2, celdas 28-56): Actividad 3 — `num_beams`, `do_sample`, `top_p`, `temperature` con outputs medidos.
8. **Evaluación ROUGE** (P2, celdas 57-83): Actividad 4 — distribución de largos de CNN/DailyMail, ROUGE promedio real y por qué no se alcanza el paper.

## Resultados consolidados

### Distribución de largos de CNN/DailyMail (test, tokenizer T5)

| | percentil 1 | percentil 5 | **promedio** | percentil 95 | percentil 99 |
|---|---|---|---|---|---|
| **Artículos** | 236.7 | 345.0 | **969.3** | 1974.0 | 2436.0 |
| **Resúmenes** | 32.0 | 44.0 | **79.3** | 135.0 | 174.0 |

→ El artículo promedio (**969 tokens**) supera el límite de contexto de T5 (512, ampliado a 768 en la evaluación). **Más del 90% de los artículos se trunca** → el modelo resume con información incompleta. Los resúmenes humanos, en cambio, son cortos y consistentes (~79 tokens), lo que justifica `min_length=32` / `max_length=135`.

### ROUGE de T5-small sobre el test completo (11.490 artículos, greedy, truncado a 768)

| Métrica | **Medido (t5-small)** | Reportado en el lab (T5) | Ejemplo afortunado individual (20 beams, artículo corto) |
|---|---|---|---|
| ROUGE-1 | **0.3489** | 0.4112 | 0.5833 |
| ROUGE-2 | **0.1311** | 0.1956 | 0.4286 |
| ROUGE-L | **0.2265** | 0.3835 | 0.5556 |

→ El perfil "ROUGE-1 decente, ROUGE-2/L bajos" es típico de un modelo pequeño: acierta *de qué* hablar (palabras), pero no *cómo* estructurarlo (orden y flujo). La brecha con el paper se explica por **tres causas acumuladas**: (1) modelo pequeño — t5-small (60M) vs t5-base/11B; (2) **truncamiento** — 969 promedio vs 768 procesado; (3) **decodificación greedy** (`num_beams=1`) por velocidad en lugar de beam search.

### Efecto de los parámetros de decodificación (noticia COVID, mismo texto)

| Parámetro | Observación medida |
|---|---|
| `num_beams` 5 vs 20 | 5 beams lideró con el hecho central (reapertura); 20 beams lideró con cifras de muertes. **Más beams ≠ mejor resumen** (retornos decrecientes / contraproducentes) |
| `do_sample` False vs True | True trajo alguna cita nueva ("comeback kids"), pero **diversidad modesta**: la distribución de T5 es muy puntiaguda |
| `top_p` 0.95 vs 0.9 | Comparación **contaminada** (también cambió `num_beams` 5→20). Único efecto atribuible al 0.95: un fragmento descolocado ("all of us") |
| `temperature` 0.6 vs 1.5 | **El que más diversidad introdujo**: con 1.5 el modelo parafraseó ("u.s." → "country") y usó citas distintas. La palanca real de diversidad |

## Bloques del lab

{{< cards >}}
  {{< card link="arquitectura-bertsum" title="P1 · Arquitectura extractiva (BertSum)" subtitle="[CLS] por oración + interval segment embeddings, setup del repo, clase Args" icon="academic-cap" >}}
  {{< card link="entrenamiento-bertsum" title="P1 · Estrategia de entrenamiento" subtitle="Oracle greedy por ROUGE, loss BCE, scheduler Noam, gradient accumulation, model averaging" icon="academic-cap" >}}
  {{< card link="inferencia-extractiva" title="P1 · Inferencia y trigram blocking" subtitle="Forward pass, ranking, anti-redundancia y análisis de predicciones reales" icon="academic-cap" >}}
  {{< card link="actividades-extractivo" title="P1 · Actividades (1 + V/F)" subtitle="Output del modelo, selección de oraciones, alternativas al oracle, 4 Verdadero/Falso" icon="academic-cap" >}}
  {{< card link="abstractivo-t5" title="P2 · Abstractivo con T5" subtitle="Encoder-decoder, framework text-to-text, span corruption, primera generación" icon="academic-cap" >}}
  {{< card link="generacion-cualitativa" title="P2 · Generación cualitativa (Act. 2)" subtitle="Noticia DermaSensor/FDA + Pride and Prejudice: saliencia, alucinación, domain shift" icon="academic-cap" >}}
  {{< card link="decodificacion" title="P2 · Parámetros de decodificación (Act. 3)" subtitle="num_beams, do_sample, top_p, temperature con outputs medidos" icon="academic-cap" >}}
  {{< card link="evaluacion-rouge" title="P2 · Evaluación ROUGE (Act. 4)" subtitle="Largos de CNN/DailyMail, ROUGE promedio real, brecha con el paper" icon="academic-cap" >}}
{{< /cards >}}

## Papers de esta clase

{{< cards >}}
  {{< card link="/papers/bertsum-liu-2019" title="Liu (2019) - Fine-tune BERT for Extractive Summarization" subtitle="BertSum: [CLS] por oración, inter-sentence Transformer, oracle greedy, trigram blocking" icon="document-text" >}}
  {{< card link="/papers/t5-raffel-2020" title="Raffel et al. (2020) - T5" subtitle="Framework text-to-text, span corruption, C4, encoder-decoder unificado" icon="document-text" >}}
  {{< card link="/papers/rouge-lin-2004" title="Lin (2004) - ROUGE" subtitle="Métrica recall-oriented: ROUGE-N, ROUGE-L (LCS), familia de variantes" icon="document-text" >}}
  {{< card link="/papers/nucleus-sampling-holtzman-2020" title="Holtzman et al. (2020) - Nucleus Sampling" subtitle="Top-p sampling, beam search degeneration, núcleo dinámico" icon="document-text" >}}
  {{< card link="/papers/bart-lewis-2020" title="Lewis et al. (2020) - BART" subtitle="Denoising autoencoder seq2seq, sucesor abstractivo" icon="document-text" >}}
  {{< card link="/papers/pegasus-zhang-2020" title="Zhang et al. (2020) - PEGASUS" subtitle="Gap sentence generation, pre-entrenamiento específico para resumen" icon="document-text" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/text-summarization" title="Fundamento: Text Summarization" subtitle="Extractivo vs abstractivo, oracle, sesgo LEAD, paradigmas" icon="book-open" >}}
  {{< card link="/fundamentos/decoding-strategies" title="Fundamento: Decoding Strategies" subtitle="Greedy, beam search, top-k, nucleus, temperatura" icon="adjustments" >}}
  {{< card link="/fundamentos/rouge-metric" title="Fundamento: ROUGE" subtitle="Recall-oriented, n-gram overlap, LCS, stemming" icon="variable" >}}
  {{< card link="/fundamentos/t5-encoder-decoder" title="Fundamento: T5 Encoder-Decoder" subtitle="Arquitectura seq2seq, prefijos de tarea, SentencePiece" icon="cube-transparent" >}}
{{< /cards >}}

## Cross-links

{{< cards >}}
  {{< card link="/clases/clase-22" title="Clase 22 - Teoría" subtitle="Modelos de generación: resumen extractivo y abstractivo" icon="academic-cap" >}}
  {{< card link="/clases/clase-22/profundizacion" title="Profundización" subtitle="Span corruption, oracle algoritmo, beam search, nucleus, ROUGE family" icon="academic-cap" >}}
  {{< card link="/dominios/texto" title="Dominio: Texto" subtitle="Timeline NLP: del extractivo clásico a los LLM de resumen" icon="book-open" >}}
  {{< card link="/laboratorios/lab-19" title="Lab 19 - MLOps con BentoML (anterior)" subtitle="Deployment, latencia, concurrencia, compresión JPEG" icon="academic-cap" >}}
{{< /cards >}}

---

> **Estado:** Lab completo. Cubre las 48 + 84 celdas de ambos notebooks con 8 páginas temáticas. Evidencia cuantitativa verificada en outputs reales (ROUGE 0.349/0.131/0.227 sobre 11.490 artículos; distribución de largos; 4 parámetros de decodificación). Incluye análisis crítico de gotchas (bug metodológico en la comparación de top_p, alucinación de atribución en Pride and Prejudice, ejemplo individual no representativo del promedio). Notebooks ejecutados en Colab con T4 GPU.
