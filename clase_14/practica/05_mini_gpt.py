"""
Escalon 8 - Mini-GPT entrenado en Shakespeare.

Une TODO lo que hemos construido:
  - Embeddings (escalon 1d)
  - Self-attention con Q/K/V y scaling (escalon 2)
  - Multi-head attention (escalon 6)
  - Bloque Transformer completo (escalon 7)

Le agrega lo unico que faltaba:
  - Positional encoding (aprendido)
  - Causal mask (decoder-only: cada token solo ve el pasado)
  - Output head (proyectar a vocab para next-token prediction)
  - Training loop sobre texto de Shakespeare

Y al correrlo vas a ver al modelo APRENDER A GENERAR TEXTO en vivo.
"""

import math
import os
import urllib.request
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

# Apple Silicon: usar GPU "MPS"; fallback a CPU.
device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Dispositivo: {device}")


# ---------------------------------------------------------------------------
# 1. Cargar el dataset (Shakespeare)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 1: Cargar Shakespeare (tinyshakespeare.txt)")
print("=" * 70)

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
LOCAL = "shakespeare.txt"

if not os.path.exists(LOCAL):
    print(f"\nDescargando {URL}...")
    urllib.request.urlretrieve(URL, LOCAL)
    print(f"Guardado en {LOCAL}.")

with open(LOCAL, "r") as f:
    text = f.read()

print(f"\nTexto cargado: {len(text):,} caracteres")
print(f"\nPrimeros 200 chars:")
print(text[:200])


# ---------------------------------------------------------------------------
# 2. Tokenizador char-level
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 2: Tokenizador a nivel de caracter")
print("=" * 70)

chars = sorted(set(text))
vocab_size = len(chars)
print(f"\nCaracteres unicos: {vocab_size}")
print(f"Vocabulario: {''.join(chars)!r}")

char_to_id = {c: i for i, c in enumerate(chars)}
id_to_char = {i: c for i, c in enumerate(chars)}

def encode(s: str) -> list[int]:
    return [char_to_id[c] for c in s]

def decode(ids) -> str:
    return ''.join(id_to_char[int(i)] for i in ids)

print(f"\nEjemplo: encode('Hello') = {encode('Hello')}")
print(f"         decode({encode('Hello')}) = {decode(encode('Hello'))!r}")

# Encodear todo el texto
data = torch.tensor(encode(text), dtype=torch.long)
print(f"\nTexto encodeado: tensor de shape {data.shape}, dtype {data.dtype}")
print(f"Primeros 50 tokens: {data[:50].tolist()}")
print(f"Decodeado:           {decode(data[:50])!r}")


# ---------------------------------------------------------------------------
# 3. Train/val split
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 3: Split train/val")
print("=" * 70)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"\nTrain: {len(train_data):,} tokens")
print(f"Val:   {len(val_data):,} tokens")


# ---------------------------------------------------------------------------
# 4. Hyperparametros
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 4: Hyperparametros del mini-GPT")
print("=" * 70)

block_size = 64       # context length: cada batch toma ventanas de 64 tokens
batch_size = 32       # ejemplos en paralelo
d_model = 128         # dim de embedding
h = 4                 # cabezas
n_layers = 4          # capas Transformer
d_ff = 4 * d_model    # 512
learning_rate = 3e-4
max_iters = 3000
eval_interval = 500

print(f"\n  vocab_size:    {vocab_size}")
print(f"  block_size:    {block_size}  (context length)")
print(f"  batch_size:    {batch_size}")
print(f"  d_model:       {d_model}")
print(f"  h:             {h}")
print(f"  n_layers:      {n_layers}")
print(f"  d_ff:          {d_ff}")
print(f"  learning_rate: {learning_rate}")
print(f"  max_iters:     {max_iters}")


# ---------------------------------------------------------------------------
# 5. Sampleador de batches
# ---------------------------------------------------------------------------
def get_batch(split: str):
    """
    Toma un batch random de ventanas de tamano block_size.
    Para cada ventana, el target es la misma ventana corrida 1 token a la
    derecha (next-token prediction).
    """
    src = train_data if split == "train" else val_data
    ix = torch.randint(len(src) - block_size, (batch_size,))
    x = torch.stack([src[i:i + block_size] for i in ix])
    y = torch.stack([src[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# 6. Modelo: mini-GPT
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 5: Definir el modelo Mini-GPT")
print("=" * 70)


class CausalMultiHeadAttention(nn.Module):
    """
    Multi-head attention con CAUSAL MASK.
    El causal mask hace que el token i solo pueda atender a los tokens 0..i,
    no al futuro. Es lo que hace al modelo autoregresivo (decoder-only).
    """
    def __init__(self, d_model, h, block_size):
        super().__init__()
        assert d_model % h == 0
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        # Mascara causal: triangular inferior. Se precomputa una vez.
        # Shape: (1, 1, block_size, block_size)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.W_Q(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.h, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        # Aplicar causal mask: posiciones futuras se vuelven -inf
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        head_outputs = weights @ V

        concat = head_outputs.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_O(concat)


class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class TransformerBlock(nn.Module):
    """Pre-norm: x = x + Sublayer(LN(x))."""
    def __init__(self, d_model, h, d_ff, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalMultiHeadAttention(d_model, h, block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, h, n_layers, d_ff, block_size):
        super().__init__()
        self.block_size = block_size

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)

        # Stack de bloques Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, h, d_ff, block_size)
            for _ in range(n_layers)
        ])

        # LayerNorm final + cabeza de salida
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x, targets=None):
        B, T = x.shape

        # Embedding lookup + posicional
        tok = self.token_emb(x)                                   # (B, T, d_model)
        pos = self.pos_emb(torch.arange(T, device=x.device))      # (T, d_model)
        h = tok + pos                                             # broadcasting -> (B, T, d_model)

        # Pasar por las N capas Transformer
        for block in self.blocks:
            h = block(h)

        h = self.ln_final(h)
        logits = self.head(h)                                     # (B, T, vocab_size)

        if targets is None:
            return logits, None

        # Cross-entropy: predecir el siguiente caracter en cada posicion
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """
        Toma un contexto y genera max_new_tokens caracteres autoregresivamente.
        En cada paso, predice el siguiente y lo agrega al contexto.
        """
        for _ in range(max_new_tokens):
            # Cortar el contexto al block_size si es mas largo
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # solo la ultima posicion
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # samplear
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ---------------------------------------------------------------------------
# 7. Crear el modelo
# ---------------------------------------------------------------------------
model = MiniGPT(vocab_size, d_model, h, n_layers, d_ff, block_size).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModelo creado.")
print(f"Parametros: {n_params:,} ({n_params/1e6:.2f} M)")


# ---------------------------------------------------------------------------
# 8. Generar ANTES de entrenar (referencia: basura random)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 6: Generacion ANTES de entrenar (modelo random)")
print("=" * 70)

@torch.no_grad()
def sample(prompt: str = "", max_new_tokens: int = 300) -> str:
    model.eval()
    if prompt:
        ids = encode(prompt)
        ctx = torch.tensor([ids], dtype=torch.long, device=device)
    else:
        ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(ctx, max_new_tokens)
    model.train()
    return decode(out[0].tolist())

print("\n--- Generacion (300 chars) ---")
print(sample("", 300))
print("--- fin ---")
print("\nDeberia ser totalmente random (modelo sin entrenar).")


# ---------------------------------------------------------------------------
# 9. Loop de entrenamiento
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 7: Training loop")
print("=" * 70)
print(f"\nVamos a hacer {max_iters} iteraciones. En el camino vas a ver")
print(f"el loss bajar, y cada {eval_interval} pasos un sample para ver")
print(f"como evoluciona la generacion.\n")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {}
    for split in ["train", "val"]:
        l = []
        for _ in range(20):
            x, y = get_batch(split)
            _, loss = model(x, y)
            l.append(loss.item())
        losses[split] = sum(l) / len(l)
    model.train()
    return losses

start_time = time.time()

for it in range(max_iters):
    # Evaluar y mostrar sample en checkpoints
    if it % eval_interval == 0 or it == max_iters - 1:
        losses = estimate_loss()
        elapsed = time.time() - start_time
        print(f"\n[step {it:4d}] train_loss={losses['train']:.4f}  "
              f"val_loss={losses['val']:.4f}  ({elapsed:.1f}s)")

        if it > 0:  # generar sample (excepto al inicio que ya lo hicimos)
            print(f"--- Sample en step {it} ---")
            print(sample("", 200))
            print("--- fin ---")

    # Forward + backward + step
    x, y = get_batch("train")
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# ---------------------------------------------------------------------------
# 10. Generacion final
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 8: Generacion DESPUES del entrenamiento")
print("=" * 70)

print("\n--- Sample final libre (500 chars) ---")
print(sample("", 500))
print("--- fin ---")

print("\n--- Sample con prompt 'ROMEO:' (300 chars) ---")
print(sample("ROMEO:", 300))
print("--- fin ---")


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("RESUMEN")
print("=" * 70)

elapsed = time.time() - start_time
print(f"""
Lo que acabas de hacer:

1. Cargaste un texto de Shakespeare (~1MB).
2. Tokenizaste a nivel caracter (vocab_size = {vocab_size}).
3. Construiste un Mini-GPT con:
   - Embedding de tokens + positional embedding
   - {n_layers} capas Transformer (cada una: causal MHA + FFN + LayerNorm + residual)
   - Output head (lineal + softmax)
   Total: {n_params:,} parametros ({n_params/1e6:.2f} M)
4. Lo entrenaste con next-token prediction durante {max_iters} iteraciones
   ({elapsed:.1f}s).
5. Generaste texto desde cero, viendo al modelo "aprender" a hablar.

Esto que entrenaste tiene EXACTAMENTE la misma arquitectura que GPT-3
(175 mil millones de parametros) y Claude 3 (varios cientos de
mil millones). La diferencia es solo escala:
  - Vocabulario (caracteres vs tokens BPE de 50k+)
  - d_model (128 vs 12,288)
  - n_layers (4 vs 96)
  - Dataset (1MB Shakespeare vs 570GB de internet)
  - Compute (1 minuto en tu Mac vs meses en miles de GPUs A100)

Pero la ESTRUCTURA es la misma. El "secreto" de los LLMs no es magia —
es esta misma arquitectura, escalada masivamente.

EL VIAJE COMPLETO:
  Escalon 1:  vectores, dot product, softmax, self-attention degenerada
  Escalon 1b: cross-entropy
  Escalon 1c: gradient descent + autograd
  Escalon 1d: mini-Word2Vec entrenado
  Escalon 2:  Q/K/V con scaling
  Escalon 6:  multi-head attention
  Escalon 7:  bloque Transformer completo
  Escalon 8:  Mini-GPT entrenado <- ESTAS AQUI

Ya construiste un Transformer completo, end-to-end, sin librerias de
alto nivel. Ya entiendes los Transformers.
""")

print("=" * 70)
print("FIN escalon 8. FIN del viaje.")
print("=" * 70)
