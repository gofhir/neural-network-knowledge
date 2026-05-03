---
title: "41 - Arquitectura Mini-BERT completa"
weight: 410
math: true
---

## 1. Apertura

Cap 38 mostro por que no hay causal mask en un encoder bidireccional. Cap 39 mostro como funciona el positional embedding aprendido — un `nn.Embedding(max_seq_len, d_model)` sumado al token embedding, en contraste con el RoPE de LLaMA. Cap 40 mostro los special tokens `[CLS]`, `[SEP]` y `[MASK]` y como se integran al vocab BPE.

Ahora todo junto: la arquitectura completa de Mini-BERT, el forward pass de extremo a extremo, el conteo de parametros y una tabla comparativa frente a Mini-LLaMA.

---

## 2. Arquitectura: diagrama textual

```
Input (IDs de tokens: [CLS] w1 w2 ... wN [SEP])
         |
         v
  Token Embedding           nn.Embedding(vocab_size=1115, d_model=128)
         |
         + (suma)
         |
  Pos Embedding             nn.Embedding(max_seq_len=128, d_model=128)
         |
         v
  BERTBlock x 4
  ┌────────────────────────────────────────────────────────────┐
  │   MHA (n_heads=4, sin causal mask — bidireccional)         │
  │   + residual + LayerNorm                                   │
  │   FFN (Linear → GELU → Linear, d_ff=512)                  │
  │   + residual + LayerNorm                                   │
  └────────────────────────────────────────────────────────────┘
         |
         v
  LayerNorm final
         |
         v
  h: (B, T, d_model=128)    — representacion de TODOS los tokens

  h[:, 0, :]  →  vector [CLS]  →  cabeza de clasificacion (fine-tuning)
  h[:, 1:-1, :] →  vectores de tokens  →  cabeza MLM (pretraining)
```

La diferencia estructural mas importante respecto al decoder: no hay mascara triangular inferior en la atencion. Cada posicion puede atender a todas las otras posiciones en ambas direcciones — de ahi el nombre "bidireccional".

---

## 3. Conteo de parametros

Con la configuracion `vocab_size=1115, max_seq_len=128, d_model=128, n_heads=4, n_layers=4, d_ff=512`:

| Componente | Parametros |
|---|---|
| Token Embedding | 1115 × 128 = 142,720 |
| Pos Embedding | 128 × 128 = 16,384 |
| BERTBlock × 4 (MHA + FFN + norms) | ~793,344 |
| LayerNorm final | 256 |
| **Total** | **952,448** |

El script calcula el valor exacto: **952,448 parametros**.

Para comparacion, Mini-LLaMA con configuracion equivalente tenia ~1,072,256 parametros. La diferencia se explica principalmente por:

- LLaMA no tiene pos embedding aprendido (usa RoPE, que no tiene pesos adicionales en la tabla de embedding)
- LLaMA tiene un `lm_head` (Linear de `d_model → vocab_size`) que agrega `128 × 65 = 8,320` params en la version char-level
- La arquitectura GQA de LLaMA tiene matrices W_K y W_V mas pequenas (h_kv=2 en lugar de h_q=4)

Ambos modelos son pedagogicamente equivalentes en magnitud — menos de un millon de parametros, entrenables en un laptop con MPS/CPU en minutos.

---

## 4. El script

`clase_14/practica/41_mini_bert.py`:

```python
"""41_mini_bert.py - Cap 41: forward pass completo de Mini-BERT."""
import torch
from _bpe import BPETokenizer
from _models import MiniBERT, get_device

torch.manual_seed(42)
device = get_device()

tok = BPETokenizer.load("data/bpe_tokenizer.json")
tok.add_special_tokens()
vocab_size = tok.vocab_size  # 1115

cfg = dict(vocab_size=vocab_size, max_seq_len=128,
           d_model=128, n_heads=4, n_layers=4, d_ff=512)
model = MiniBERT(**cfg).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"MiniBERT: {n_params:,} parametros")
print(f"Comparacion: MiniLLaMA tuvo ~1,072,256 params\n")

# Forward pass de ejemplo
sentences = [
    "To be or not to be",
    "En un lugar de la Mancha",
]
print("=== Forward pass ===\n")
for s in sentences:
    ids = torch.tensor([tok.encode_bert(s)], dtype=torch.long, device=device)
    ids = ids[:, :128]  # truncar a max_seq_len
    h = model(ids)
    cls_vec = h[0, 0]  # vector [CLS]
    print(f"Texto:     {s!r}")
    print(f"Tokens:    {ids.shape[1]} (incluyendo [CLS] y [SEP])")
    print(f"h.shape:   {h.shape}  — (batch=1, seq_len, d_model=128)")
    print(f"[CLS] vec: norma={cls_vec.norm().item():.4f}, primeros 4 dims: {cls_vec[:4].tolist()}")
    print()

print("=== Diferencias con Mini-LLaMA ===")
print("""
Mini-LLaMA                Mini-BERT
-----------               ----------
GQA (h_q=4, h_kv=2)      MHA (n_heads=4)
RoPE en Q y K             Learned pos emb (sumado al token emb)
RMSNorm                   LayerNorm
SwiGLU                    GELU
Causal mask               Sin mascara
max_seq_len=256            max_seq_len=128
Genera: next token         Clasifica: [CLS] vector
""")
```

---

## 5. Output del script

```
MiniBERT: 952,448 parametros
Comparacion: MiniLLaMA tuvo ~1,072,256 params

=== Forward pass ===

Texto:     'To be or not to be'
Tokens:    9 (incluyendo [CLS] y [SEP])
h.shape:   torch.Size([1, 9, 128])  — (batch=1, seq_len, d_model=128)
[CLS] vec: norma=11.3137, primeros 4 dims: [-1.0207159519195557, 1.4710266590118408, 2.0132479667663574, -0.6454116702079773]

Texto:     'En un lugar de la Mancha'
Tokens:    16 (incluyendo [CLS] y [SEP])
h.shape:   torch.Size([1, 16, 128])  — (batch=1, seq_len, d_model=128)
[CLS] vec: norma=11.3137, primeros 4 dims: [-0.990801215171814, 1.1741478443145752, 1.8800697326660156, -0.5290235280990601]

=== Diferencias con Mini-LLaMA ===

Mini-LLaMA                Mini-BERT
-----------               ----------
GQA (h_q=4, h_kv=2)      MHA (n_heads=4)
RoPE en Q y K             Learned pos emb (sumado al token emb)
RMSNorm                   LayerNorm
SwiGLU                    GELU
Causal mask               Sin mascara
max_seq_len=256            max_seq_len=128
Genera: next token         Clasifica: [CLS] vector
```

---

## 6. Tabla comparativa: Mini-LLaMA vs Mini-BERT

| Aspecto | Mini-LLaMA | Mini-BERT |
|---|---|---|
| Tipo | Decoder-only | Encoder-only |
| Atencion | GQA (h_q=4, h_kv=2) | MHA (n_heads=4) |
| Pos. encoding | RoPE en Q y K | Learned pos emb (sumado al token emb) |
| Normalizacion | RMSNorm | LayerNorm |
| Activacion FFN | SwiGLU | GELU |
| Mascara | Causal (triangular inferior) | Sin mascara (bidireccional) |
| max_seq_len | 256 | 128 |
| Objetivo | Genera: next token | Clasifica: vector [CLS] |
| Parametros | ~1,072,256 | 952,448 |

La diferencia de mascara es la mas importante arquitecturalmente. LLaMA solo puede ver el pasado (autoregresivo). BERT puede ver el futuro y el pasado desde cualquier posicion — esto es lo que hace que sus representaciones sean contextuales en ambas direcciones y utiles para clasificacion.

---

## 7. Por que el forward retorna (B, T, d_model) completo

Mini-LLaMA en inference tipicamente usa solo el ultimo token del output:

```python
logits = model(x)[:, -1, :]  # solo el ultimo token para next-token prediction
```

Mini-BERT retorna el tensor completo `(B, T, d_model)` — una representacion para cada posicion de la secuencia. Esto no es un detalle de implementacion, es una consecuencia del objetivo de diseno:

**El decoder solo necesita el ultimo token** porque su tarea es predecir el siguiente token dado el prefijo. El estado en la ultima posicion ya "vio" toda la secuencia hasta ese punto gracias a la mascara causal, y a partir de ese estado se proyecta al siguiente token.

**El encoder necesita todos los tokens** porque:

1. **Pretraining MLM:** la tarea es predecir el token original en cada posicion enmascarada. Si la frase tiene 16 tokens y el 15% se enmascara, hay ~2-3 posiciones donde predecir. Necesitas el vector de salida de cada una de esas posiciones para calcular la perdida MLM.

2. **Fine-tuning NER / POS:** tareas como Named Entity Recognition o Part-of-Speech tagging requieren una etiqueta por token. La cabeza de clasificacion se aplica a cada vector `h[:, i, :]` individualmente.

3. **Fine-tuning clasificacion de secuencia:** se usa solo `h[:, 0, :]` (el vector `[CLS]`), pero el resto del tensor existe igualmente — simplemente se ignora.

El costo de retornar el tensor completo es que la memoria escala con $T \times d\_model$, no solo con $d\_model$. Para secuencias largas (BERT base usa max_seq_len=512), esto multiplica el consumo de memoria significativamente en comparacion con un decoder que acumula un solo vector de estado.

---

## 8. Preguntas de verificacion

**1.** El output muestra que ambas frases producen `norma=11.3137` para el vector `[CLS]`. Ambos son modelos sin entrenar (pesos aleatorios, `torch.manual_seed(42)`). iEsperas que esta norma cambie despues del pretraining MLM? iAumenta, disminuye, o no hay una prediccion teorica clara? Razona en terminos de LayerNorm y la inicializacion de pesos.

**2.** "To be or not to be" produce 9 tokens (7 BPE + `[CLS]` + `[SEP]`) y `h.shape = [1, 9, 128]`. Si pasaras un batch de 4 frases con `max_seq_len=128` y hubieras que hacer padding para igualar longitudes, icual seria la forma del tensor de salida? iQue mecanismo evita que el padding afecte los calculos de atencion en las posiciones reales?

**3.** La tabla muestra que Mini-LLaMA usa RoPE (sin pesos adicionales en la tabla de embeddings) y Mini-BERT usa Learned pos emb (`nn.Embedding(128, 128)`). Calcula cuantos parametros extra agrega el positional embedding aprendido respecto a RoPE. Si aumentaras `max_seq_len` de 128 a 512, ien cuanto creceria el conteo de parametros de Mini-BERT? iY el de Mini-LLaMA?
