---
title: "Practica desde 0 - ELMo, BERT, GPT, RLHF mini"
weight: 30
sidebar:
  open: true
---

La clase 20 cubre la transicion del NLP estatico al era de los foundation models: ELMo (contextual), BERT (bidireccional + MLM), GPT (causal + autoregresivo) y el stack RLHF que convirtio modelos base en asistentes. Esta practica los implementa todos en **minima escala** para entender por dentro que los distingue, no solo leer sus papers. Cuando aplica, replicamos el mismo modelo en **triple framework** (PyTorch, TensorFlow y JAX/Flax) para ver como cada uno expresa las mismas ideas. El cierre es un pipeline RLHF de juguete (SFT + Reward Model + PPO) corriendo en una sola maquina, mas un fine-tuning real de BETO sobre informes radiologicos clinicos.

## Caminos

{{< cards >}}
  {{< card link="01-elmo-mini" title="01 - ELMo mini" subtitle="Char-CNN + BiLSTM + biLM en PyTorch desde 0" icon="code" >}}
  {{< card link="02-mlm-encoder-mini" title="02 - MLM encoder mini" subtitle="Encoder bidireccional + Masked LM en PyTorch, TensorFlow y JAX" icon="code" >}}
  {{< card link="03-causal-decoder-mini" title="03 - Decoder causal mini" subtitle="Decoder con causal mask + autoregressive LM en PyTorch, TensorFlow y JAX" icon="code" >}}
  {{< card link="04-fine-tuning-beto" title="04 - Fine-tuning BETO clinico" subtitle="HuggingFace + BETO para clasificacion de informes radiologicos" icon="code" >}}
  {{< card link="05-rlhf-toy" title="05 - RLHF toy pipeline" subtitle="SFT + Reward Model + PPO mini con TRL en una sola maquina" icon="code" >}}
{{< /cards >}}

## Requisitos previos

- [Clase 14 - Transformer desde 0](../../clase-14/practica): self-attention, multi-head, bloque Transformer, decoder causal vs encoder bidireccional.
- [Clase 16 - NLP clasico](../../clase-16/practica): tokenizacion, vocab, OOV, n-gramas — para apreciar que resuelven los embeddings contextuales.
- Python intermedio (clases, context managers, decoradores) y NumPy.
- PyTorch basico (tensores, `nn.Module`, autograd, training loop). Util pero no obligatorio: nociones de TensorFlow/Keras y JAX/Flax — los caminos 02 y 03 los introducen comparativamente.
- GPU **recomendada** pero no obligatoria. Todos los caminos corren en CPU con minutos de tolerancia; con GPU bajan a segundos.

## Tecnologias usadas

| Camino | Stack principal | Frameworks secundarios |
|--------|------------------|------------------------|
| 01 - ELMo mini | PyTorch 2.x | — |
| 02 - MLM encoder mini | PyTorch 2.x | TensorFlow 2.x, JAX + Flax |
| 03 - Decoder causal mini | PyTorch 2.x | TensorFlow 2.x, JAX + Flax |
| 04 - Fine-tuning BETO | `transformers` + `datasets` + PyTorch | — |
| 05 - RLHF toy | `trl` (SFT + Reward + PPO) + `transformers` | `accelerate`, `peft` |

Versiones de referencia: `torch>=2.2`, `tensorflow>=2.15`, `jax>=0.4` con `flax>=0.8`, `transformers>=4.40`, `datasets>=2.18`, `trl>=0.8`, `peft>=0.10`, `accelerate>=0.28`.

## Estructura comun de los caminos

Cada camino sigue el mismo arco pedagogico:

1. **Motivacion**: que problema resuelve este modelo que el anterior no podia.
2. **Setup**: dependencias, dataset minimo, vocab/tokenizer.
3. **Implementacion paso a paso**: cada componente (embedding, attention, head, mascara) en celdas pequenas, con shapes anotadas.
4. **Entrenamiento mini**: pocas iteraciones sobre corpus de juguete para ver la loss bajar y validar que el codigo aprende.
5. **Evaluacion**: metricas adecuadas al modelo (perplexity, accuracy, top-k MLM, BLEU para generacion, reward para RLHF).
6. **Discusion**: que limitaciones tiene la version mini, que cambia a escala, lectura honesta de los resultados.
7. **Siguientes pasos**: pistas para profundizar (papers, escalar a corpus mas grande, conectar con otros caminos del curso).

---

**Ver tambien:** [Clase 20 - Teoria](../teoria) · [Clase 20 - Profundizacion](../profundizacion) · [Clase 14 - Transformer desde 0](../../clase-14/practica) · [Clase 16 - NLP clasico](../../clase-16/practica).
