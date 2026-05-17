---
title: "No Language Left Behind (NLLB-200)"
weight: 240
math: true
---

{{< paper-card
    title="No Language Left Behind: Scaling Human-Centered Machine Translation"
    authors="NLLB Team (Meta AI, UC Berkeley, Johns Hopkins)"
    year="2022"
    venue="arXiv (publicado en Nature 2024)"
    pdf="/papers/nllb-team-2022.pdf"
    arxiv="2207.04672" >}}
Modelo Transformer encoder-decoder con **Sparsely Gated Mixture of Experts (MoE)** que soporta **traducción entre 200 idiomas** (40,602 direcciones), incluyendo decenas de idiomas low-resource históricamente desatendidos. **54.5B parámetros** en la versión completa; variantes distilladas hasta 600M parámetros corren en CPU. **+44% BLEU sobre el state of the art previo**. Open source completo: modelos, datos (FLORES-200), código (fairseq/nllb). Es **el modelo de MT más amplio en cobertura lingüística** que existe.
{{< /paper-card >}}

---

## Contexto

La traducción automática neural recorre este camino: rules-based (1940s-90s) → statistical (1990s-2010s) → RNN/LSTM seq2seq (2010s-2017) → Transformer (Vaswani 2017) → modelos multilingües (M2M-100, 2020) → **NLLB-200 (2022)**.

Hasta 2022, los grandes sistemas multilingües cubrían **~100 idiomas**, todos high-resource o mid-resource. El internet mostraba un sesgo brutal: 63.7% de sitios web en inglés, pero **solo 25.9% de usuarios habla inglés como L1**. Idiomas como catalán, asamés, ligurio, kinyarwanda quedaban fuera de la conversación digital.

NLLB se planteó duplicar la cobertura de 100 a 200 idiomas. El proyecto se enmarca en **Value Sensitive Design** (Friedman & Hendry 2019): empieza con **44 entrevistas a hablantes nativos** de 36 idiomas low-resource antes de decidir arquitectura. Ética antes que ingeniería.

---

## Ideas principales

### 1. FLORES-200 dataset de evaluación

**204 idiomas**, traducidos profesionalmente. Cada frase está en TODOS los idiomas → permite evaluar **40,602 direcciones** (202 × 201).

Códigos BCP-47 con script Unicode explícito: `eng_Latn`, `spa_Latn`, `cmn_Hans`, `arb_Arab`, `kin_Latn`, `quy_Latn`. Distingue scripts del mismo idioma: `srp_Cyrl` (serbio cirílico) vs `srp_Latn` (serbio latino).

### 2. Arquitectura Transformer + Mixture of Experts

Encoder-decoder estándar (Vaswani 2017) con varias modificaciones:

**Tokenización SentencePiece**:
- Vocabulario **256,000 subwords** entrenados sobre 100M sentencias.
- Sampling temperature τ=5 para favorecer low-resource.
- Maneja todos los scripts (latín, cirílico, árabe, devanagari, etíope, chino).

**Source language prefix**: el modelo recibe el código de idioma como prefix en la secuencia. Decisión clave: prefixar con **source** (no target) mejora rendimiento zero-shot.

**Pre-LayerNorm**: norm antes de cada sublayer, más estable que Post-LN para entrenar modelos grandes.

**Sparsely Gated MoE** — el cambio arquitectónico clave. En cada capa Transformer alternada, se reemplaza el FFN denso por **64 expertos paralelos** + un gating network que rutea cada token a sus **top-2 expertos**:

$$\text{MoE}(x_t) = \sum_{e=1}^{E} \mathcal{G}_{t,e} \cdot \text{FFN}_e(x_t)$$

Resultado: modelo de **54B parámetros** pero **solo activa ~1B por token**. Conditional compute mantiene costo de FLOPs comparable a un modelo denso de 1.3B.

### 3. Regularización específica para MoE en low-resource

Vanilla MoE + low-resource = overfitting catastrófico. NLLB introduce tres técnicas:

| Técnica | Mecanismo |
|---|---|
| **Overall dropout** | Dropout uniforme en todo el modelo |
| **MoE Expert Output Masking (EOM)** | Para fracción aleatoria de tokens, enmascarar output del experto antes de combinar |
| **Conditional MoE Routing (CMR)** | Rama densa paralela; gate binario decide MoE pesado vs shared dense |

**EOM con p_drop=0.3, p_eom=0.1** es la mejor configuración global. Mejora +0.6 a +0.9 chrF++ en very low-resource respecto a vanilla MoE.

### 4. Curriculum learning

Para pares low-resource que overfittean en pocos updates, introducen pares **por fases**: empezar con high-resource, agregar progresivamente buckets de low-resource hacia el final del training. Limita updates innecesarios sobre pares pequeños.

### 5. Data mining masivo para low-resource

- **LASER3**: sentence encoder que extrae bitext alineado en 148 idiomas.
- **Language Identification** para 200+ idiomas (necesario para limpiar crawls).
- **stopes**: librería de data mining (corpus monolingüe → bitext alineado).
- **NLLB-Seed**: datos seed profesionales en 39 idiomas low-resource.

### 6. Toxicity-200 y safety

Wordlists para detectar **toxicidad añadida** (added toxicity / hallucinated toxicity) en 200 idiomas. Crítico para deployment: una traducción que invente insultos puede causar daño grave.

---

## Resultados experimentales

Métrica principal: **chrF++** (no solo BLEU). chrF++ usa precision/recall de character n-grams + word n-grams. Más robusto cross-scripts.

> **+44% BLEU relativo sobre el state of the art previo** (M2M-100).

**Distribuciones disponibles**:

| Modelo | Parámetros | Disco | Hardware |
|---|---|---|---|
| `nllb-200-54B` (MoE) | 54.5B | ~120 GB | GPU 80GB+ (A100) |
| `nllb-200-3.3B` (denso) | 3.3B | ~15 GB | GPU 24GB+ |
| `nllb-200-1.3B` (denso) | 1.3B | ~6 GB | GPU 12GB+ |
| `nllb-200-distilled-1.3B` | 1.3B | ~5 GB | GPU 8GB+ |
| **`nllb-200-distilled-600M`** | 600M | **~2.4 GB** | **CPU posible** |

El distilled de 600M es el que usaste en este lab — corre en Colab free tier sin GPU (~10s/frase) o con GPU T4 (~0.5s/frase).

---

## Limitaciones reconocidas

Discutidas explícitamente en el paper:

1. **Cobertura ≠ paridad de calidad**. Un idioma puede estar "soportado" con BLEU 15 mientras inglés-español está a BLEU 45.
2. **Sesgos web heredados**. Corpora minados sobre-representan dominios formales (Wikipedia, gobierno, religión).
3. **Hallucination**: el modelo genera tokens gramaticalmente correctos pero semánticamente disociados. Más frecuente en low-resource.
4. **Janus-faced digital participation** (sección 9.2): traer idiomas low-resource al internet via MT empodera comunidades pero puede acelerar pérdida de matiz cultural.
5. **Energía y huella ambiental**. Entrenar 54B con MoE consume megavatios-hora.

---

## Por qué importa hoy

- **Statement-of-the-art moral** del MT moderno: no se trata solo de hacer más BLEU, sino de **decidir conscientemente qué idiomas merecen calidad humana**.
- **Open source completo**: código (fairseq/nllb), modelos en HuggingFace Hub, datasets en GitHub. Excepcional en una empresa Big Tech.
- **FLORES-200 es el benchmark estándar** post-2022. Cualquier paper de MT multilingüe lo reporta.
- **Wikipedia editors usan NLLB** para generar borradores en idiomas sin artículos.
- **Modelos distilled** (600M, 1.3B) permiten deployment en edge devices con poca RAM.
- **Versión Nature 2024**: peer review consolida el trabajo académicamente.

A 2026, **~3500 citas en Google Scholar** y creciendo rápidamente. El distilled 600M tiene **~5M descargas/mes** en HuggingFace.

---

## Notas y enlaces

- Cómo usar en Python:
  ```python
  from transformers import pipeline
  translator = pipeline("translation",
      model="facebook/nllb-200-distilled-600M",
      src_lang="spa_Latn", tgt_lang="eng_Latn")
  translator("El paciente presenta hipertensión")
  ```
- Lista oficial de idiomas FLORES-200: `github.com/facebookresearch/flores/blob/main/flores200/README.md`.
- Para texto clínico **no está fine-tuneado** específicamente. Traducciones de terminología médica pueden ser literales pero clínicamente subóptimas.
- Sucesor potencial: **SeamlessM4T** (Meta 2023) — multimodal speech+text+translation.

Ver fundamentos: [Sentiment Analysis](/fundamentos/sentiment-analysis) (patrón translate-then-analyze). Ver papers relacionados: [Attention is All You Need](/papers/attention-is-all-you-need-vaswani-2017) (Transformer base) · [Seq2Seq](/papers/seq2seq-sutskever-2014) (encoder-decoder origen) · [VADER](/papers/vader-hutto-gilbert-2014).
