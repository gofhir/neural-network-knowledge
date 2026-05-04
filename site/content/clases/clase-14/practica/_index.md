---
title: "Practica - Construir el Transformer desde 0"
weight: 40
sidebar:
  open: true
---

Mapa de estudio progresivo para entender el Transformer construyendolo a mano en PyTorch. Cada capitulo es un script ejecutable que se acompana de una narrativa pedagogica con preguntas de verificacion. La recomendacion es leer el capitulo, correr el script, leer la salida, y solo avanzar al siguiente cuando el "click" sea solido.

## Filosofia

> La teoria de los Transformers se entiende leyendo papers. La intuicion solo se gana **escribiendo el codigo y mirando los numeros**.

Esta seccion no asume conocimiento previo de PyTorch ni de redes neuronales — solo programacion basica. Cada concepto se construye encima del anterior.

El viaje esta organizado en cinco **Caminos**:

- **Camino 1 (Fases 1-5, capitulos 01-21)**: construir el Transformer desde cero hasta un Mini-LLaMA char-level entrenado en Shakespeare. Cubre Vaswani 2017 → LLaMA 2024 modernizaciones.
- **Camino 2 (Fases 6-7, capitulos 22-29)**: convertir el Mini-LLaMA en un asistente que sigue instrucciones, via SFT + DPO. El stack moderno post-pretraining usado por Llama-3-Instruct, Mistral-Instruct, Zephyr.
- **Camino 2.5 (Fase 8, capitulos 30-37)**: rehacer SFT+DPO con BPE tokenizer bilingue Shakespeare+Quijote. Honesto sobre los tradeoffs char vs BPE.
- **Camino 4 (Fases 9-11, capitulos 38-49)**: Mini-BERT encoder-only con MLM pretraining y fine-tuning a deteccion de idioma EN/ES. El "otro lado" del Transformer.
- **Camino 3 (Fases 12-16, capitulos 50-63)**: interpretabilidad mecanicista — abrir Mini-LLaMA y Mini-BERT con hooks, activation patching, sparse autoencoders. La frontera 2024-2026 inspirada en Anthropic Circuits Thread.

## Capitulos

### Fase 1 — Fundamentos del entrenamiento

{{< cards >}}
  {{< card link="01-embeddings-y-dot-product" title="01 - Embeddings y dot product" subtitle="Vectores como palabras, similitud geometrica, self-attention manual" icon="academic-cap" >}}
  {{< card link="02-cross-entropy" title="02 - Cross-entropy" subtitle="El modelo predice probabilidades, no respuestas. Por que -log(P)" icon="academic-cap" >}}
  {{< card link="02b-self-supervision" title="02b - Self-supervision" subtitle="De donde sale el target sin humanos etiquetando: el truco que hizo posibles los LLMs" icon="academic-cap" >}}
  {{< card link="03-gradient-descent" title="03 - Gradient descent y autograd" subtitle="Como PyTorch ajusta millones de pesos automaticamente" icon="academic-cap" >}}
  {{< card link="04-mini-word2vec" title="04 - Mini Word2Vec" subtitle="Training real: ver embeddings random aprender estructura semantica" icon="academic-cap" >}}
{{< /cards >}}

### Fase 2 — La arquitectura Transformer

{{< cards >}}
  {{< card link="05-qkv-scaled-attention" title="05 - Q/K/V con scaling" subtitle="El primer ladrillo real del Transformer: proyecciones aprendibles, asimetria, scaling sqrt(d_k)" icon="academic-cap" >}}
  {{< card link="06-multi-head-attention" title="06 - Multi-Head Attention" subtitle="h atenciones en paralelo en subespacios distintos. Naive vs eficiente" icon="academic-cap" >}}
  {{< card link="06b-multi-head-internals" title="06b - Multi-Head Internals" subtitle="Demo numerica: ver con tus ojos como las cabezas son slices de la matriz grande" icon="academic-cap" >}}
  {{< card link="07-transformer-block" title="07 - Bloque Transformer" subtitle="Attention + FFN + residual + LayerNorm. La capa completa" icon="academic-cap" >}}
{{< /cards >}}

### Fase 3 — El modelo final

{{< cards >}}
  {{< card link="08-mini-gpt" title="08 - Mini-GPT entrenado en Shakespeare" subtitle="El climax del viaje: un GPT real construido y entrenado por ti" icon="sparkles" >}}
{{< /cards >}}

### Fase 4 — Experimentos sobre el mini-GPT

Una vez tienes el modelo funcionando, ¿que pasa si cambias hyperparametros? 7 experimentos focalizados sobre el mismo modelo.

{{< cards >}}
  {{< card link="09-experimentos-basicos" title="09 - Experimentos basicos" subtitle="Temperatura, prompts variados, modelo micro vs estandar" icon="beaker" >}}
  {{< card link="10-train-longer" title="10 - Entrenar mas tiempo" subtitle="6000 iters: aparecen personajes reales (CLARENCE, BRUTUS)" icon="beaker" >}}
  {{< card link="11-model-xl" title="11 - Modelo XL" subtitle="6x mas grande: aprende personajes reales como MISTRESS OVERDONE" icon="beaker" >}}
  {{< card link="12-dataset-quijote" title="12 - Don Quijote en español" subtitle="Misma arquitectura, distinto idioma. Universalidad del Transformer" icon="beaker" >}}
  {{< card link="13-gelu-vs-relu" title="13 - GELU vs ReLU" subtitle="Por que los modelos modernos usan GELU" icon="beaker" >}}
  {{< card link="14-topk-sampling" title="14 - Top-k sampling" subtitle="Greedy, multinomial, top-k: estrategias de generacion" icon="beaker" >}}
  {{< card link="15-seed-variety" title="15 - Variedad con seeds" subtitle="Por que cada conversacion con un LLM es unica" icon="beaker" >}}
{{< /cards >}}

### Fase 5 — Modernizaciones LLaMA (de Vaswani 2017 a LLaMA 2024)

Tu mini-GPT es la arquitectura de Vaswani 2017. LLaMA (2023) tiene 5 mejoras incrementales que, acumuladas, son la diferencia entre "texto vagamente gramatical" y "asistente utilizable". Cada modernizacion: por que existe, su math, su implementacion. Al final, las 5 combinadas en un Mini-LLaMA.

{{< cards >}}
  {{< card link="16-rmsnorm" title="16 - RMSNorm" subtitle="Reemplaza LayerNorm: mas simple, sin restar la media" icon="cog" >}}
  {{< card link="17-swiglu" title="17 - SwiGLU" subtitle="Reemplaza FFN con ReLU: gating con dos caminos paralelos" icon="cog" >}}
  {{< card link="18-rope" title="18 - RoPE" subtitle="Reemplaza positional embeddings: rotaciones geometricas en Q y K" icon="cog" >}}
  {{< card link="19-gqa" title="19 - GQA" subtitle="Reemplaza MHA: cabezas Q comparten K, V en grupos" icon="cog" >}}
  {{< card link="20-kv-cache" title="20 - KV-cache" subtitle="Generacion eficiente: 10-100x mas rapida que sampling naive" icon="cog" >}}
  {{< card link="21-mini-llama" title="21 - Mini-LLaMA" subtitle="Las 5 modernizaciones combinadas: el estado del arte 2024 (en miniatura)" icon="sparkles" >}}
{{< /cards >}}

---

## Camino 2 — De modelo base a asistente

El Mini-LLaMA del cap 21 predice caracteres al estilo Shakespeare pero **no sigue instrucciones**. Camino 2 lo convierte en un asistente que respeta el formato `INSTR/RESP` y `Q/A`, aplicando el stack moderno post-pretraining: **SFT** (Supervised Fine-Tuning) + **DPO** (Direct Preference Optimization). Es el pipeline que hizo posible Llama-3-Instruct y similares.

### Fase 6 — SFT (Supervised Fine-Tuning)

Fine-tuneamos el base con un dataset sintetico de 4 tareas (reverse, upper, repeat, qa). Lo distintivo: **loss masking** sobre tokens de respuesta — el modelo aprende a generar la respuesta dado el prompt, no a memorizar prompts.

{{< cards >}}
  {{< card link="22-base-model-no-instructions" title="22 - El problema: base model no sigue instrucciones" subtitle="Demo del comportamiento del base — Shakespeare drift puro" icon="academic-cap" >}}
  {{< card link="23-dataset-sft" title="23 - Dataset SFT: 4 tareas sinteticas" subtitle="5000 pares (instruccion, respuesta) char-level vocab-safe" icon="academic-cap" >}}
  {{< card link="24-sft-training" title="24 - SFT training: loss masking" subtitle="El corazon de SFT — solo penalizar tokens de respuesta" icon="cog" >}}
  {{< card link="25-sft-eval" title="25 - Eval SFT: Base vs SFT" subtitle="drift 40% → 0%, repeat/qa al 100%, lectura honesta de limitaciones" icon="chart-bar" >}}
{{< /cards >}}

### Fase 7 — DPO (Direct Preference Optimization)

DPO refina el SFT con preferencias `(chosen, rejected)`. Saltea el reward model + PPO de RLHF clasico via la derivacion de Rafailov 2023. Honesto: en este setting (char-level, beta=0.1), DPO mantuvo el formato (drift 0%) pero degrado accuracy — leccion sobre tradeoffs reales del tuning.

{{< cards >}}
  {{< card link="26-preferencias-bradley-terry" title="26 - Preferencias y Bradley-Terry" subtitle="El modelo de preferencias 1952, demo numerica, RLHF clasico" icon="academic-cap" >}}
  {{< card link="27-dpo-loss" title="27 - DPO loss: la derivacion" subtitle="Forma cerrada policy optima, log-ratios, KL implicito, beta" icon="cog" >}}
  {{< card link="28-dataset-dpo" title="28 - Dataset DPO: chosen + rejected" subtitle="3000 triples mix base-sampled + cross-task" icon="academic-cap" >}}
  {{< card link="29-dpo-training-eval" title="29 - DPO training + eval: cierre Camino 2" subtitle="Loss converge a 0.007 pero accuracy regresa — leccion sobre tradeoffs" icon="sparkles" >}}
{{< /cards >}}

---

## Camino 2.5 — BPE addendum (de char-level a subword)

Camino 2 cerro con un puzzle: SFT funcionaba, pero DPO degradaba accuracy. La hipotesis principal: char-level es subóptimo — sin semantica de palabras, sin compresion de contexto, sin capacidad para tareas open-ended. Camino 2.5 reemplaza el tokenizador por **BPE desde cero** (1000 merges, vocab bilingue Shakespeare+Quijote, 1112 tokens) y rehace SFT+DPO sobre 4 tareas BPE-naturales: qa (memorizacion), repeat (igual que C2), complete-en (continuacion Shakespeare), complete-es (continuacion Quijote).

**Resultado honesto**: BPE no es mejora automatica. Char-level gana qa (69% vs 19%) y repeat (100% vs 77%) — entropia por paso menor lo favorece. PERO BPE habilita complete-* que char-level no puede hacer en absoluto. DPO con beta=0.5 valido parcialmente la hipotesis del cap 29 (β=0.5 degrada menos que β=0.1 pero ambos siguen degradando). El cap 37 cierra con la leccion: char-level es pedagogico, BPE es produccion.

### Fase 8 — BPE + SFT + DPO

{{< cards >}}
  {{< card link="30-bpe-desde-cero" title="30 - BPE desde cero" subtitle="Algoritmo merge frequencies, vocab bilingue 1112 tokens, corpus bias" icon="academic-cap" >}}
  {{< card link="31-pretrain-bpe" title="31 - Pretrain con BPE" subtitle="Mini-LLaMA bilingue, loss 7.18 → 2.68, generacion cross-lingual" icon="cog" >}}
  {{< card link="32-refactor-tokenizer" title="32 - Refactor tokenizer-agnostic" subtitle="CharTokenizer + BPETokenizer mismo interfaz, 11/11 tests" icon="cog" >}}
  {{< card link="33-dataset-sft-bpe" title="33 - Dataset SFT-BPE" subtitle="4 tareas bilingues — qa, repeat, complete-en, complete-es" icon="academic-cap" >}}
  {{< card link="34-sft-bpe" title="34 - SFT con BPE" subtitle="BPE-SFT peor que char-SFT en qa/repeat — leccion honesta" icon="chart-bar" >}}
  {{< card link="35-dataset-dpo-bpe" title="35 - Dataset DPO-BPE" subtitle="3000 triples con rejected linguisticamente ricos" icon="academic-cap" >}}
  {{< card link="36-dpo-bpe" title="36 - DPO-BPE + beta sweep" subtitle="β=0.1 vs β=0.5 — hipotesis cap 29 parcialmente validada" icon="cog" >}}
  {{< card link="37-comparacion-char-vs-bpe" title="37 - Comparacion final char vs BPE" subtitle="Tabla maestra de los 6 modelos — cierre Camino 2.5" icon="sparkles" >}}
{{< /cards >}}

---

## Camino 4 — Mini-BERT (Encoder-only + MLM)

Caminos 1, 2 y 2.5 cubrieron el lado **decoder-only** de la familia Transformer (Mini-GPT y Mini-LLaMA: generan texto auto-regresivamente). Camino 4 cubre el lado **encoder-only**: bidireccionalidad, Masked Language Modeling, y el paradigma "pretrain masivo + fine-tuning ligero" que domino NLP entre 2018 y 2022. Reusa el BPETokenizer del cap 30 extendido con tres special tokens (`[CLS]`, `[SEP]`, `[MASK]`) y construye Mini-BERT (952K params) con MLM pretraining sobre Shakespeare+Quijote y fine-tuning a deteccion de idioma EN/ES (accuracy 0.998).

### Fase 9 — Arquitectura encoder

{{< cards >}}
  {{< card link="38-encoder-vs-decoder" title="38 - Encoder vs Decoder" subtitle="Sin causal mask: cada token ve todos los demas. La diferencia estructural" icon="academic-cap" >}}
  {{< card link="39-positional-embeddings" title="39 - Positional embeddings aprendidos" subtitle="nn.Embedding(max_seq_len, d_model). Comparacion con RoPE de LLaMA" icon="academic-cap" >}}
  {{< card link="40-special-tokens" title="40 - Special tokens [CLS] [SEP] [MASK]" subtitle="Extension del BPE bilingue, encode_bert vs encode" icon="academic-cap" >}}
  {{< card link="41-mini-bert" title="41 - Arquitectura Mini-BERT completa" subtitle="952K params, post-LN, comparacion con Mini-LLaMA" icon="cog" >}}
{{< /cards >}}

### Fase 10 — MLM pretraining

{{< cards >}}
  {{< card link="42-mlm-loss" title="42 - MLM loss (80/10/10)" subtitle="apply_mlm_mask, ignore_index=-100, simetria con SFT loss masking" icon="academic-cap" >}}
  {{< card link="43-mlm-pretraining" title="43 - Pretraining MLM" subtitle="3000 iters Shakespeare+Quijote, loss 7.12 → 4.96" icon="cog" >}}
  {{< card link="44-eval-mlm" title="44 - Eval MLM: predict_mask" subtitle="Top-k sobre [MASK], honesto: BPE bilingue dificulta el MLM" icon="chart-bar" >}}
{{< /cards >}}

### Fase 11 — Fine-tuning EN/ES

{{< cards >}}
  {{< card link="45-cls-head" title="45 - ClassificationHead sobre [CLS]" subtitle="d_model → 2 clases, 258 params, vector [CLS] como resumen" icon="academic-cap" >}}
  {{< card link="46-dataset-lang" title="46 - Dataset EN/ES" subtitle="2000 train + 500 eval con ventanas de 64 tokens, sin leakage" icon="academic-cap" >}}
  {{< card link="47-finetune-bert" title="47 - Fine-tuning con LR=2e-5" subtitle="500 iters, evitar catastrophic forgetting, loss 0.62 → 0.08" icon="cog" >}}
  {{< card link="48-eval-bert" title="48 - Eval: accuracy + attention + PCA" subtitle="Accuracy 0.998, attention pattern del ultimo bloque, PCA de [CLS]" icon="chart-bar" >}}
{{< /cards >}}

### Cierre

{{< cards >}}
  {{< card link="49-comparativa-bert-gpt" title="49 - Comparativa BERT vs GPT" subtitle="Tabla tripartita, historia 2018-2026, Sentence-Transformers, RAG cross-encoders" icon="sparkles" >}}
{{< /cards >}}

---

## Camino 3 — Interpretabilidad mecanicista

Caminos 1-4 construyeron modelos que funcionan. Camino 3 abre la caja: ver QUE hace cada componente, mapear circuitos causales, descomponer la superposition con sparse autoencoders. Construido desde cero (sin TransformerLens) sobre Mini-LLaMA y Mini-BERT — fiel a la pedagogia "you build it, you understand it" del curso. Inspirado en el [Anthropic Circuits Thread](https://transformer-circuits.pub/).

### Fase 12 — Hooks y residual stream (fundacional)

{{< cards >}}
  {{< card link="50-forward-hooks" title="50 - Forward hooks" subtitle="cache_activations con register_forward_hook, context manager para cleanup" icon="academic-cap" >}}
  {{< card link="51-residual-stream" title="51 - Residual stream: la autopista" subtitle="Cada bloque LEE y ESCRIBE delta. Bloque 3 escribe ||delta||/||in||=1.64" icon="academic-cap" >}}
  {{< card link="52-logit-lens" title="52 - Logit lens: capa por capa" subtitle="Honesto: Mini-LLaMA NO predice 'b' tras 'To be or not to' — limitacion escala" icon="chart-bar" >}}
{{< /cards >}}

### Fase 13 — Atencion por dentro

{{< cards >}}
  {{< card link="53-attention-heatmaps" title="53 - Heatmaps de atencion" subtitle="ASCII heatmaps de las 16 cabezas (4 capas x 4 heads)" icon="academic-cap" >}}
  {{< card link="54-previous-token-heads" title="54 - Previous-token heads" subtitle="block.2 head.0 con score 0.547 — top-1 sobre 50 prompts" icon="cog" >}}
  {{< card link="55-induction-heads" title="55 - Induction heads (no emergen)" subtitle="Honesto: top score 0.057 — escala insuficiente vs GPT-2 small" icon="chart-bar" >}}
  {{< card link="56-qk-ov-decomposition" title="56 - QK / OV decomposition" subtitle="La cabeza top-prev-token NO es copy head: ||OV-I||/||I||=1.04" icon="cog" >}}
{{< /cards >}}

### Fase 14 — Causalidad e intervencion

{{< cards >}}
  {{< card link="57-activation-patching" title="57 - Activation patching" subtitle="Del correlacional al causal: flujo del speaker hacia posicion 12" icon="cog" >}}
  {{< card link="58-circuit-discovery" title="58 - Head-level patching" subtitle="Descripcion != causalidad: top prev-token tiene recovery NEGATIVO (-2.7%)" icon="sparkles" >}}
{{< /cards >}}

### Fase 15 — Frontera moderna (SAEs)

{{< cards >}}
  {{< card link="59-superposition" title="59 - Superposition" subtitle="Toy model: 5 features en 2 dim. Cluster colapsado + anti-pareo emerge" icon="academic-cap" >}}
  {{< card link="60-train-sae" title="60 - Entrenar un SAE" subtitle="d_model=128 -> d_features=512, 98.4% var explicada, L0=166" icon="cog" >}}
  {{< card link="61-interpret-sae" title="61 - Interpretar features del SAE" subtitle="47% features monosemanticas (242/512): chars, puntuacion, separadores" icon="chart-bar" >}}
{{< /cards >}}

### Fase 16 — Contraste BERT y cierre

{{< cards >}}
  {{< card link="62-interp-bert" title="62 - Interpretabilidad en Mini-BERT" subtitle="Capa 3 distingue idiomas: EN -> [SEP], ES -> [CLS]. Cosine 0.002 entre [CLS]" icon="academic-cap" >}}
  {{< card link="63-comparativa-interp-frontera" title="63 - Comparativa + frontera 2026" subtitle="Tabla maestra, Anthropic Circuits, SAEs a escala, mech interp para alignment" icon="sparkles" >}}
{{< /cards >}}

---

## Setup

Antes de correr cualquier script:

```bash
cd clase_14/practica
uv venv
uv pip install torch numpy
```

Para correr cualquier capitulo:

```bash
.venv/bin/python 01_dot_product_attention_manual.py
.venv/bin/python 01b_cross_entropy_demo.py
.venv/bin/python 01c_gradient_descent_demo.py
.venv/bin/python 01d_train_embeddings.py
.venv/bin/python 02_qkv_scaled_attention.py
.venv/bin/python 03_multi_head_attention.py
.venv/bin/python 03b_multi_head_internals.py
.venv/bin/python 04_transformer_block.py
.venv/bin/python 05_mini_gpt.py
.venv/bin/python 06_experimentos.py
.venv/bin/python 07_train_longer.py
.venv/bin/python 08_model_xl.py
.venv/bin/python 09_dataset_quijote.py
.venv/bin/python 10_gelu_vs_relu.py
.venv/bin/python 11_topk_sampling.py
.venv/bin/python 12_seed_variety.py
.venv/bin/python 13_mini_llama.py
```

Camino 2 (caps 22-29) requiere haber corrido `13_mini_llama.py` previamente para generar `checkpoints/mini_llama_base.pt`:

```bash
.venv/bin/python 14_show_base_no_instructions.py
.venv/bin/python 15_build_sft_dataset.py
.venv/bin/python 16_train_sft.py
.venv/bin/python 17_eval_sft.py
.venv/bin/python 18_dpo_intro.py
.venv/bin/python 19_dpo_loss_derivation.py
.venv/bin/python 20_build_dpo_dataset.py
.venv/bin/python 21_train_dpo.py
```

Camino 2.5 (caps 30-37) construye un BPETokenizer y reentrena Mini-LLaMA con vocab BPE:

```bash
.venv/bin/python 30_build_bpe.py
.venv/bin/python 31_pretrain_bpe.py
.venv/bin/python 32_tokenizer_refactor_demo.py
.venv/bin/python 33_build_sft_bpe.py
.venv/bin/python 34_train_sft_bpe.py
.venv/bin/python 35_build_dpo_bpe.py
.venv/bin/python 36_train_dpo_bpe.py
.venv/bin/python 37_compare_char_vs_bpe.py
```

Camino 4 (caps 38-49) construye Mini-BERT desde cero — encoder-only con MLM pretraining y fine-tuning a deteccion de idioma. Requiere `data/bpe_tokenizer.json` (cap 30):

```bash
.venv/bin/python 38_encoder_vs_decoder.py
.venv/bin/python 39_positional_embeddings.py
.venv/bin/python 40_special_tokens.py
.venv/bin/python 41_mini_bert.py
.venv/bin/python 42_mlm_loss.py
.venv/bin/python 43_train_bert.py
.venv/bin/python 44_eval_mlm.py
.venv/bin/python 45_cls_head.py
.venv/bin/python 46_dataset_lang.py
.venv/bin/python 47_finetune_bert.py
.venv/bin/python 48_eval_bert.py
```

Camino 3 (caps 50-63) abre Mini-LLaMA y Mini-BERT con tecnicas de interpretabilidad mecanicista. Requiere los checkpoints de Camino 1 (`mini_llama_base.pt`) y Camino 4 (`mini_bert_finetuned.pt`):

```bash
.venv/bin/python 50_forward_hooks.py
.venv/bin/python 51_residual_stream.py
.venv/bin/python 52_logit_lens.py
.venv/bin/python 53_attention_heatmaps.py
.venv/bin/python 54_previous_token_heads.py
.venv/bin/python 55_induction_heads.py
.venv/bin/python 56_qk_ov_decomposition.py
.venv/bin/python 57_activation_patching.py
.venv/bin/python 58_circuit_discovery.py
.venv/bin/python 59_superposition.py
.venv/bin/python 60_train_sae.py
.venv/bin/python 61_interpret_sae.py
.venv/bin/python 62_interp_bert.py
```

Tests unitarios para los helpers (`load_pretrained_mini_llama`, `generate_with_prompt`, `compute_logp_response`, `dpo_loss`, `build_char_maps`):

```bash
.venv/bin/python -m pytest tests/ -v
```

Los scripts viven en `clase_14/practica/` (fuera del sitio Hugo, en el repo principal).

## El camino completo

```text
Camino 1 — Construir el Transformer
  Fase 1: Fundamentos del entrenamiento
    01   vectores, dot product, embeddings, self-attention manual
    02   cross-entropy: -log(P_correcta)
    02b  self-supervision: el dataset es su propio target
    03   gradient descent + autograd
    04   mini-Word2Vec entrenado (training end-to-end)
  Fase 2: La arquitectura
    05   Q/K/V con scaling sqrt(d_k)
    06   multi-head attention
    06b  deep dive numerico de las cabezas
    07   bloque Transformer (FFN + residual + LayerNorm)
  Fase 3: Modelo final
    08   Mini-GPT entrenado en Shakespeare
  Fase 4: Experimentos
    09-15 temperature, scaling, GELU, top-k, datasets
  Fase 5: Modernizaciones LLaMA 2024
    16   RMSNorm  17  SwiGLU  18  RoPE  19  GQA  20  KV-cache
    21   Mini-LLaMA combinando las 5 ← cierre Camino 1

Camino 2 — De modelo base a asistente (SFT + DPO)
  Fase 6: Supervised Fine-Tuning
    22   demo: el base no sigue instrucciones
    23   dataset SFT 4 tareas
    24   SFT training con loss masking
    25   eval Base vs SFT
  Fase 7: Direct Preference Optimization (char-level)
    26   Bradley-Terry preferences
    27   derivacion de la loss DPO
    28   dataset DPO chosen + rejected
    29   DPO training + eval ← cierre Camino 2

Camino 2.5 — BPE addendum (de char-level a subword)
  Fase 8: BPE + SFT + DPO bilingue
    30   BPE desde cero (1112 tokens, Shakespeare+Quijote)
    31   pretrain Mini-LLaMA con BPE
    32   refactor tokenizer-agnostic
    33   dataset SFT-BPE 4 tareas (qa, repeat, complete-en, complete-es)
    34   SFT con BPE + eval comparativo
    35   dataset DPO-BPE 3000 triples
    36   DPO + beta sweep (β=0.1 vs β=0.5)
    37   comparacion final char vs BPE ← cierre Camino 2.5

Camino 4 — Mini-BERT (encoder-only + MLM)
  Fase 9: Arquitectura encoder
    38   encoder vs decoder (sin causal mask)
    39   positional embeddings aprendidos
    40   special tokens [CLS] [SEP] [MASK]
    41   Mini-BERT 952K params (post-LN, MHA, GELU)
  Fase 10: MLM pretraining
    42   MLM loss (80/10/10 masking, ignore_index=-100)
    43   pretrain MLM 3000 iters (Shakespeare+Quijote)
    44   eval MLM: predict_mask top-k
  Fase 11: Fine-tuning EN/ES
    45   ClassificationHead sobre [CLS]
    46   dataset EN/ES (ventanas 64 tokens, sin leakage)
    47   fine-tuning LR=2e-5 (anti-catastrophic forgetting)
    48   eval: accuracy 0.998 + attention + PCA [CLS]
  Cierre
    49   comparativa BERT vs GPT ← cierre Camino 4

Camino 3 — Interpretabilidad mecanicista (caps 50-63)
  Fase 12: Hooks y residual stream
    50-52  forward hooks, residual stream, logit lens
  Fase 13: Atencion por dentro
    53-56  heatmaps, prev-token, induction (no emerge), QK/OV decomp
  Fase 14: Causalidad
    57-58  activation patching, head-level (descripcion != causalidad)
  Fase 15: Frontera SAE
    59-61  superposition, train SAE, interpretar features (47% monosemanticas)
  Fase 16: BERT + cierre
    62-63  interp Mini-BERT, comparativa + frontera 2026 ← cierre Camino 3
```

## Que viene despues (Caminos pendientes)

Despues de cerrar Camino 3 (interpretabilidad), queda un camino para profundizar:

- **Camino 5 — ViT (Vision Transformer)**: llevar el Transformer a imagenes. Patches 16x16 como tokens, [class] token aprendible, entrenar en MNIST/CIFAR, multimodal extensions (CLIP). Las tecnicas de interpretabilidad del Camino 3 se transfieren directamente — solo cambia la modalidad.

Y experimentos mas chicos sobre Camino 2:

- Sweep de `beta` en DPO (0.1, 0.3, 0.5, 1.0): ¿como cambia el tradeoff accuracy vs preferences?
- Early stopping en DPO basado en eval intermedio.
- Limpiar el dataset DPO removiendo cross-task rejected.

Mas alla del Transformer:

- Mamba / SSMs (alternativas modernas).
- RWKV.
- RetNet.
- Mixture of Experts (MoE).

---

**Ver tambien:** [Clase 14 - Teoria](../teoria) · [Clase 14 - Profundizacion](../profundizacion) · [Wiki de investigacion](../wiki) · [Fundamento self-attention](/fundamentos/self-attention) · [Fundamento transformer](/fundamentos/transformer).
