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

El viaje esta organizado en dos **Caminos**:

- **Camino 1 (Fases 1-5, capitulos 01-21)**: construir el Transformer desde cero hasta un Mini-LLaMA char-level entrenado en Shakespeare. Cubre Vaswani 2017 → LLaMA 2024 modernizaciones.
- **Camino 2 (Fases 6-7, capitulos 22-29)**: convertir el Mini-LLaMA en un asistente que sigue instrucciones, via SFT + DPO. El stack moderno post-pretraining usado por Llama-3-Instruct, Mistral-Instruct, Zephyr.

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

Tests unitarios para los helpers (`load_pretrained_mini_llama`, `generate_with_prompt`, `compute_logp_response`, `dpo_loss`, `build_char_maps`):

```bash
.venv/bin/python -m pytest tests/ -v
```

Los scripts viven en `clase_14/practica/` (fuera del sitio Hugo, en el repo principal).

## El camino completo

```text
01  vectores, dot product, embeddings, self-attention manual
02  cross-entropy: -log(P_correcta)
02b self-supervision: el dataset es su propio target
03  gradient descent + autograd
04  mini-Word2Vec entrenado (training end-to-end)
05  Q/K/V con scaling sqrt(d_k)
06  multi-head attention
06b deep dive numerico: ¿se pierde info al dividir en cabezas?
07  bloque Transformer (FFN + residual + LayerNorm)
08  Mini-GPT entrenado en Shakespeare ← el momento "click" final
```

## Que viene despues (Caminos pendientes)

Despues de cerrar Camino 2 (SFT + DPO), quedan varios caminos para profundizar:

- **Camino 3 — Interpretabilidad mecanicista**: abrir el modelo entrenado y ver que hace cada componente. Attention pattern analysis, induction heads, QK/OV decomposition, circuit discovery. Inspirado en los Transformer Circuits de Anthropic.
- **Camino 4 — BERT-style (encoder-only + MLM)**: el "otro paradigma" del Transformer. Masked Language Modeling, fine-tuning para clasificacion / NER / QA, comparacion BERT vs GPT.
- **Camino 5 — ViT (Vision Transformer)**: llevar el Transformer a imagenes. Patches 16x16 como tokens, [class] token aprendible, entrenar en MNIST/CIFAR, multimodal extensions (CLIP).

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
