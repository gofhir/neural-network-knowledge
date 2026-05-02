# Practica Transformer desde 0

Escalones progresivos para entender el Transformer construyendolo a mano en PyTorch.

## Setup

```bash
cd clase_14/practica
uv venv
uv pip install torch numpy
```

Para correr cualquier script:

```bash
.venv/bin/python 01_dot_product_attention_manual.py
```

## Escalones

| Script | Tema | Concepto clave |
|---|---|---|
| `01_dot_product_attention_manual.py` | Embeddings + dot product attention | Self-attention sin Q/K/V: solo `X @ X.T` + softmax + suma ponderada |
| `02_scaled_dot_product_qkv.py` | Q, K, V + scaling | Por que `1/sqrt(d_k)` y proyecciones lineales |
| `03_multi_head_attention.py` | Multi-head | h cabezas en paralelo via reshape |
| `04_transformer_block.py` | Bloque Transformer | Attention + FFN + residual + LayerNorm |
| `05_mini_gpt.py` | Mini-GPT char-level | Decoder-only entrenado en Shakespeare |

Cada script imprime resultados intermedios — el aprendizaje viene de **leer la salida** y entender que esta pasando.
