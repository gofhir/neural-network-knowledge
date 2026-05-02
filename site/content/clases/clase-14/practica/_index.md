---
title: "Practica - Construir el Transformer desde 0"
weight: 40
sidebar:
  open: true
---

Mapa de estudio progresivo para entender el Transformer construyendolo a mano en PyTorch. Cada capitulo es un script ejecutable que se acompana de una narrativa pedagogica con preguntas de verificacion. La recomendacion es leer el capitulo, correr el script, leer la salida, y solo avanzar al siguiente cuando el "click" sea solido.

## Filosofia

> La teoria de los Transformers se entiende leyendo papers. La intuicion solo se gana **escribiendo el codigo y mirando los numeros**.

Esta seccion no asume conocimiento previo de PyTorch ni de redes neuronales — solo programacion basica. Cada concepto se construye encima del anterior. Al terminar los 9 capitulos, vas a haber construido un Transformer completo end-to-end, entrenado en Shakespeare, sin librerias de alto nivel.

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

## Que viene despues (proximos experimentos)

Variantes del mini-GPT para experimentar:

- Texto en español (Don Quijote).
- Mas profundidad / mas heads / mas dim.
- Reemplazar ReLU por GELU.
- Sustituir LayerNorm por RMSNorm (LLaMA).
- Implementar RoPE en lugar de positional embeddings aprendidos.
- Dropout, weight decay, learning rate schedules.

Mas alla del Transformer:

- Mamba / SSMs (alternativas modernas).
- RWKV.
- RetNet.
- Mixture of Experts (MoE).

---

**Ver tambien:** [Clase 14 - Teoria](../teoria) · [Clase 14 - Profundizacion](../profundizacion) · [Wiki de investigacion](../wiki) · [Fundamento self-attention](/fundamentos/self-attention) · [Fundamento transformer](/fundamentos/transformer).
