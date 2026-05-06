---
title: "Diseño — Sección Dominios, Ola 6 (Datos estructurados)"
date: 2026-05-05
status: aprobado
autor: Roberto Araneda
---

# Diseño — Ola 6: Datos estructurados (cierre)

## Contexto

Ola final del proyecto Dominios. Olas 1-5 entregaron Texto/NLP, Visión, Multimodal, Audio/Voz, Video, Robótica/RL. Esta Ola 6 cierra con **Datos estructurados** — tabular, series temporales, y grafos en una sola página con narrativa unificada.

Tras esta ola, los 7 dominios están completos.

## Material existente

**Sin material específico** de tabular/series/grafos. Todos los hitos serán `minimal`. La decisión deliberada (lección de Olas 3-5): mejor honesto que `covered` overstated.

## Decisiones aprobadas

1. Una página: `dominios/estructurados/_index.md`, patrón idéntico a Audio/Video/Robótica.
2. Cinco eras con narrativa cronológica que mezcla los tres subdominios (tabular + series + grafos).
3. Aproximadamente 18-20 hitos distribuidos a través de las 5 eras.
4. Status mix: 0 deep + 0 covered + todos `minimal`.
5. Implementación en 4 tasks (infraestructura ya existe).
6. Branch: `feat/dominios-ola-6`.

## Estructura de la página

`site/content/dominios/estructurados/_index.md` reemplaza el stub. Mismo molde:

1. Front matter (`title: "Datos estructurados"`, `weight: 7`, `sidebar.open: true`).
2. `# Datos estructurados` (H1).
3. `## El problema central` — 1-2 párrafos.
4. `## Línea de tiempo` con 5 eras y ~19 hitos.
5. 5 subsecciones `## Era N — <nombre> (<rango>)` con Problema heredado / Idea clave / Qué la destronó (eras 1-4) o Qué viene (era 5).
6. `## Estado del arte hoy` (callout).
7. `## Casos de uso reales`.
8. `## Qué viene`.
9. `## Recursos relacionados` (incluyendo cierre como dominio final con enlace al landing y a los 6 dominios anteriores).
10. Pie con fecha.

## El problema central — esbozo

Dos párrafos:

1. **"Datos estructurados" es el cajón de sastre que reúne todo lo que no es texto, imagen, audio o video.** Tres familias principales: **tabular** (filas y columnas, mezcla de tipos numéricos y categóricos — facturas, registros médicos, transacciones), **series temporales** (mediciones ordenadas en el tiempo — sensores, demanda eléctrica, precios), y **grafos** (entidades conectadas por relaciones — redes sociales, moléculas, knowledge graphs). Comparten una propiedad central: la estructura matemática del dato es discreta y heterogénea, no continua y uniforme como en imagen o audio.

2. **Tres tensiones definen el campo:** (1) **¿deep learning o gradient boosting en tabular?** XGBoost (2014) sigue ganando muchos benchmarks tabulares en 2025 — un caso raro donde DL no destrona al método clásico, debate aún vivo; (2) **predicción de horizonte largo en series temporales** — modelos autoregresivos acumulan error en multi-step rollout; modelos globales (entrenados sobre muchas series) generalizan mejor que ajustes por serie; (3) **inductive vs transductive en grafos** — entrenar sobre un grafo fijo (Cora, Pubmed) vs aprender funciones que generalicen a grafos no vistos. Cada era de la disciplina navegó estas tres tensiones de forma distinta.

## Línea de tiempo — eras y hitos

### Era 1 — ML clásico tabular (1990s-2010)

| Hito | Año | Status |
|---|---|---|
| Regresión lineal/logística | clásicos | `minimal` |
| Random Forest (Breiman) | 2001 | `minimal` |
| GBM (Friedman) | 2001 | `minimal` |

### Era 2 — Gradient Boosting domina tabular (2014-2017)

| Hito | Año | Status |
|---|---|---|
| XGBoost (Chen & Guestrin) | 2014 | `minimal` |
| LightGBM (Microsoft) | 2017 | `minimal` |
| CatBoost (Yandex) | 2017 | `minimal` |

### Era 3 — Deep learning para grafos y series (2016-2019)

| Hito | Año | Status |
|---|---|---|
| GCN (Kipf & Welling) | 2017 | `minimal` |
| GraphSAGE (Hamilton) | 2017 | `minimal` |
| GAT (Veličković) | 2018 | `minimal` |
| DeepAR (Salinas / Amazon) | 2017/2019 | `minimal` |
| N-BEATS (Oreshkin) | 2019 | `minimal` |

### Era 4 — Transformers a tabular y series (2019-2022)

| Hito | Año | Status |
|---|---|---|
| TFT — Temporal Fusion Transformer (Lim) | 2019 | `minimal` |
| TabTransformer (Huang et al.) | 2020 | `minimal` |
| FT-Transformer (Gorishniy) | 2021 | `minimal` |
| PatchTST (Nie) | 2022 | `minimal` |

### Era 5 — Foundation models y "XGBoost still rules" (2023-presente)

| Hito | Año | Status |
|---|---|---|
| TabPFN (Hollmann) | 2023 | `minimal` |
| TimeGPT (Nixtla) | 2023 | `minimal` |
| Chronos (Amazon) | 2024 | `minimal` |
| Lag-Llama (ServiceNow) | 2024 | `minimal` |
| TabPFN v2 | 2025 | `minimal` |
| Debate "GBM vs DL" sigue vivo en tabular | 2024-2025 | `minimal` |

**Total: 21 hitos** (3+3+5+4+6).

## Estado del arte hoy — esbozo

Callout con frontier 2024-2025:

- **TabPFN v2** — modelo Bayesian-style preentrenado, supera a XGBoost en datasets pequeños (<10k filas) — pero el debate sigue vivo en datasets grandes.
- **TimeGPT / Chronos / Lag-Llama** — foundation models para forecasting con zero-shot transfer.
- **GraphRAG** (Microsoft) — grafos como contexto para LLMs.
- **GNN modernos** (GIN, Graph Transformers) — estado del arte en química y drug discovery.
- **AutoGluon / H2O AutoML** — pipelines automáticos que combinan GBM, RF y DL.
- **XGBoost / LightGBM / CatBoost** — siguen siendo defaults industriales en Kaggle, fintech, healthtech.

## Casos de uso reales

- **Fraud detection en banca y fintech** — XGBoost domina (millones de transacciones, latencia baja).
- **Forecasting de demanda en retail y energía** — DeepAR, TimeGPT, ensembles con métodos clásicos.
- **Drug discovery y química computacional** — GNN sobre moléculas (AlphaFold para proteínas — diferente paradigma).
- **Recomendación** — DLRM (Meta) y variantes para sistemas de recomendación a escala web.
- **Healthtech** — predicción de readmisión, clasificación de notas clínicas estructuradas, riesgo cardiovascular.
- **Knowledge graphs y enterprise search** — GNN + LLMs para búsqueda semántica corporativa.
- **Series financieras** — trading, riesgo, derivados — combinaciones de GBM, ARIMA, y DL.
- **Manufactura y mantenimiento predictivo** — series temporales de sensores con anomaly detection.

## Qué viene

Las apuestas activas: **foundation models tabular efectivos** (TabPFN extiende a más filas/columnas; ¿nivel de XGBoost en escala?), **forecasting universal** (un modelo que generalice a cualquier dominio sin fine-tuning — TimeGPT, Chronos), **GNN + LLMs** (grafos como contexto enriquecido para razonamiento de modelos generales), **AutoML cada vez más automatizado** (la era del data scientist generalista que selecciona modelos manualmente está terminando), y **el debate DL vs GBM** sigue vivo en tabular — TabPFN v2 (2025) cambió el balance en datasets pequeños, pero XGBoost mantiene su corona en muchos casos de producción. La pregunta abierta: ¿qué tarea dejará de ser "structured data engineering" para volverse "prompt the foundation model"?

## Cierre del proyecto Dominios

Este es el último de los 7 dominios. Tras Ola 6, la sección Dominios cubre:

- Texto / NLP
- Visión
- Audio / Voz
- Video
- Multimodal
- Robótica / RL
- Datos estructurados

Cada uno con problema central, línea de tiempo evolutiva, eras explicadas, SOTA, casos de uso y recursos.

## Plan de implementación (4 tasks)

| Task | Entregable |
|---|---|
| 1 | Front matter + problema central + timeline (5 eras + 21 hitos) |
| 2 | 5 era subsections |
| 3 | SOTA + casos + qué viene + recursos |
| 4 | Build limpio + push + PR |

Sin tocar shortcodes, CSS, menú ni stats.

## Convenciones (heredadas)

- Español con tildes correctas.
- Tono pedagógico-narrativo.
- 800-1500 palabras totales.
- Sin Co-Authored-By en commits.
- `weight: 7` (orden actual del sidebar).

## Riesgos

| Riesgo | Mitigación |
|---|---|
| Datos puntuales (años, autores) | Code reviewer subagent debe verificar especialmente XGBoost (2014 vs 2016 publicación KDD), Random Forest (2001), DeepAR (2017 arXiv vs 2019 publicación) |
| 21 hitos pueden ser demasiados | Aceptable: la era 3 (DL para grafos+series) y era 5 (foundation models + debate) justifican 5-6 hitos cada una |
| El "debate GBM vs DL" como hito separado puede ser editorial | Aceptable: refleja un fenómeno único en este dominio, no presente en otros |
