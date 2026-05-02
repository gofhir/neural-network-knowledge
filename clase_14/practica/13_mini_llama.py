"""
Escalon final: Mini-LLaMA.

Combina TODAS las modernizaciones (RMSNorm + SwiGLU + RoPE + GQA + KV-cache)
en un solo modelo. Entrena en Shakespeare y compara contra el mini-GPT
original en CALIDAD (loss) y VELOCIDAD (generacion con KV-cache).

Es la diferencia entre Vaswani 2017 y LLaMA 2024 — misma arquitectura
base, refinamientos acumulados.
"""

import math
import os
import time
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)
device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ---------------------------------------------------------------------------
# Cargar Shakespeare
# ---------------------------------------------------------------------------
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
LOCAL = "shakespeare.txt"
if not os.path.exists(LOCAL):
    urllib.request.urlretrieve(URL, LOCAL)
with open(LOCAL) as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
char_to_id = {c: i for i, c in enumerate(chars)}
id_to_char = {i: c for i, c in enumerate(chars)}
encode = lambda s: [char_to_id[c] for c in s]
decode = lambda ids: ''.join(id_to_char[int(i)] for i in ids)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


# ---------------------------------------------------------------------------
# Modernizacion 1: RMSNorm
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich 2019)."""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.gamma * x / rms


# ---------------------------------------------------------------------------
# Modernizacion 2: SwiGLU FFN
# ---------------------------------------------------------------------------
class SwiGLU(nn.Module):
    """SwiGLU FFN (Shazeer 2020): tres proyecciones con gating Swish."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ---------------------------------------------------------------------------
# Modernizacion 3: RoPE (Rotary Position Embeddings)
# ---------------------------------------------------------------------------
def precompute_rope(d_k, max_seq_len, theta=10000.0):
    """Precompute cos/sin tables para RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, freqs)  # (max_seq_len, d_k/2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """
    Aplicar rotacion RoPE.
    x:   (B, h, T, d_k)
    cos: (T, d_k/2)
    sin: (T, d_k/2)
    """
    x1, x2 = x.chunk(2, dim=-1)  # split feature dim in half
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, d_k/2) for broadcasting
    sin = sin.unsqueeze(0).unsqueeze(0)
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos
    return torch.cat([rotated_x1, rotated_x2], dim=-1)


# ---------------------------------------------------------------------------
# Modernizacion 4 + 5: GQA con KV-cache
# ---------------------------------------------------------------------------
class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention con RoPE y KV-cache.

    h_q cabezas Q, h_kv cabezas K/V (con h_kv | h_q).
    Cada grupo de h_q/h_kv cabezas Q comparte un par K, V.
    """
    def __init__(self, d_model, h_q, h_kv, max_seq_len):
        super().__init__()
        assert h_q % h_kv == 0
        self.d_model = d_model
        self.h_q = h_q
        self.h_kv = h_kv
        self.d_k = d_model // h_q
        self.group_size = h_q // h_kv
        self.max_seq_len = max_seq_len

        self.W_Q = nn.Linear(d_model, h_q * self.d_k, bias=False)
        self.W_K = nn.Linear(d_model, h_kv * self.d_k, bias=False)
        self.W_V = nn.Linear(d_model, h_kv * self.d_k, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        # Causal mask (precomputada)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask.view(1, 1, max_seq_len, max_seq_len))

        # RoPE precomputado
        cos, sin = precompute_rope(self.d_k, max_seq_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        # KV-cache (lazy init)
        self.cache_k = None
        self.cache_v = None

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, use_cache=False):
        B, T, _ = x.shape
        Q = self.W_Q(x).view(B, T, self.h_q, self.d_k).transpose(1, 2)   # (B, h_q, T, d_k)
        K = self.W_K(x).view(B, T, self.h_kv, self.d_k).transpose(1, 2)  # (B, h_kv, T, d_k)
        V = self.W_V(x).view(B, T, self.h_kv, self.d_k).transpose(1, 2)

        # RoPE: la posicion depende de si tenemos cache o no
        if use_cache and self.cache_k is not None:
            cache_len = self.cache_k.size(2)
            cos = self.rope_cos[cache_len:cache_len + T]
            sin = self.rope_sin[cache_len:cache_len + T]
        else:
            cos = self.rope_cos[:T]
            sin = self.rope_sin[:T]

        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)

        # KV-cache: concatenar al cache previo
        if use_cache:
            if self.cache_k is not None:
                K = torch.cat([self.cache_k, K], dim=2)
                V = torch.cat([self.cache_v, V], dim=2)
            self.cache_k = K
            self.cache_v = V

        # GQA: replicar K, V para que matcheen las h_q cabezas de Q
        K_full = K.repeat_interleave(self.group_size, dim=1)
        V_full = V.repeat_interleave(self.group_size, dim=1)

        # Attention
        scores = Q @ K_full.transpose(-2, -1) / math.sqrt(self.d_k)

        # Causal mask: tomar el slice correcto segun cache_offset
        T_q, T_k = scores.size(-2), scores.size(-1)
        q_start = T_k - T_q  # offset de los queries en la secuencia total
        mask_slice = self.mask[:, :, q_start:q_start + T_q, :T_k]
        scores = scores.masked_fill(mask_slice == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        out = weights @ V_full
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        return self.W_O(out)


# ---------------------------------------------------------------------------
# Bloque LLaMA: pre-norm + GQA + SwiGLU + residuals
# ---------------------------------------------------------------------------
class LLaMABlock(nn.Module):
    def __init__(self, d_model, h_q, h_kv, d_ff, max_seq_len):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, h_q, h_kv, max_seq_len)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x, use_cache=False):
        x = x + self.attn(self.norm1(x), use_cache=use_cache)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Mini-LLaMA: el modelo completo
# ---------------------------------------------------------------------------
class MiniLLaMA(nn.Module):
    """
    Modelo mini-LLaMA con:
      - RMSNorm (en vez de LayerNorm)
      - SwiGLU FFN (en vez de FFN ReLU)
      - RoPE (en vez de positional embeddings aprendidos)
      - GQA (en vez de MHA estandar)
      - KV-cache para generacion eficiente
    """
    def __init__(self, vocab_size, d_model, h_q, h_kv, n_layers, d_ff, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len
        # NO positional embedding! RoPE se encarga de la posicion.
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            LLaMABlock(d_model, h_q, h_kv, d_ff, max_seq_len)
            for _ in range(n_layers)
        ])
        self.norm_final = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def reset_cache(self):
        for block in self.blocks:
            block.attn.reset_cache()

    def forward(self, x, targets=None, use_cache=False):
        h = self.tok_emb(x)  # NO se suma positional embedding
        for block in self.blocks:
            h = block(h, use_cache=use_cache)
        h = self.norm_final(h)
        logits = self.head(h)

        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_cache=True):
        if use_cache:
            self.reset_cache()
            # Prefill: procesar el prompt entero de una vez
            self(idx, use_cache=True)
            # Decode: un token a la vez
            for _ in range(max_new_tokens):
                last = idx[:, -1:]
                logits, _ = self(last, use_cache=True)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('inf')
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_token], dim=1)
        else:
            # Modo naive (sin cache): forward completo cada paso
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.max_seq_len:]
                logits, _ = self(idx_cond, use_cache=False)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('inf')
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_token], dim=1)
        return idx


# ---------------------------------------------------------------------------
# Hyperparametros (mismos que escalon 8 para comparacion)
# ---------------------------------------------------------------------------
print("=" * 72)
print("MINI-LLAMA: todas las modernizaciones combinadas")
print("=" * 72)
print(f"Dispositivo: {device}")

block_size = 64
batch_size = 32
d_model = 128
h_q = 4
h_kv = 2  # GQA con 2 grupos de 2 cabezas Q cada uno
n_layers = 4
d_ff = 384  # SwiGLU con 3 matrices, ~ 4*d_model * 2/3 para mantener params
learning_rate = 3e-4
max_iters = 3000
max_seq_len = 256

# Crear modelo
torch.manual_seed(1337)
model = MiniLLaMA(
    vocab_size=vocab_size,
    d_model=d_model,
    h_q=h_q,
    h_kv=h_kv,
    n_layers=n_layers,
    d_ff=d_ff,
    max_seq_len=max_seq_len,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"\nMini-LLaMA: {n_params:,} parametros ({n_params/1e6:.2f} M)")
print(f"Comparacion con mini-GPT (escalon 8):")
print(f"  Mini-GPT:    816,128 params")
print(f"  Mini-LLaMA:  {n_params:>9,} params")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def get_batch(split):
    src = train_data if split == "train" else val_data
    ix = torch.randint(len(src) - block_size, (batch_size,))
    x = torch.stack([src[i:i + block_size] for i in ix])
    y = torch.stack([src[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


print("\n" + "=" * 72)
print("Training Mini-LLaMA (3000 iteraciones, mismo que escalon 8)")
print("=" * 72)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
start_time = time.time()

for it in range(max_iters):
    x, y = get_batch("train")
    _, loss = model(x, y, use_cache=False)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if it % 500 == 0 or it == max_iters - 1:
        print(f"  step {it:4d}: loss = {loss.item():.4f}  ({time.time()-start_time:.1f}s)")

train_time = time.time() - start_time
print(f"\nTraining tiempo total: {train_time:.1f}s")


# ---------------------------------------------------------------------------
# Generacion con KV-cache
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("Generacion: comparar con vs sin KV-cache")
print("=" * 72)


def sample(prompt, max_new_tokens=200, temperature=0.8, top_k=10, use_cache=True):
    ids = encode(prompt) if prompt else [0]
    ctx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        ctx, max_new_tokens,
        temperature=temperature, top_k=top_k, use_cache=use_cache,
    )
    return decode(out[0].tolist())


# Test de velocidad
gen_tokens = 200
prompt = "ROMEO:\n"

torch.manual_seed(42)
print(f"\nGenerando {gen_tokens} tokens SIN KV-cache (naive):")
t0 = time.time()
text_no_cache = sample(prompt, max_new_tokens=gen_tokens, use_cache=False)
time_no_cache = time.time() - t0
print(f"Tiempo: {time_no_cache:.2f}s")
print(f"\n{text_no_cache}")

torch.manual_seed(42)
print(f"\n\nGenerando {gen_tokens} tokens CON KV-cache:")
t0 = time.time()
text_with_cache = sample(prompt, max_new_tokens=gen_tokens, use_cache=True)
time_with_cache = time.time() - t0
print(f"Tiempo: {time_with_cache:.2f}s")
print(f"\n{text_with_cache}")

speedup = time_no_cache / time_with_cache
print(f"\nSpeedup con KV-cache: {speedup:.2f}x")


# ---------------------------------------------------------------------------
# Generaciones variadas
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("Generaciones variadas (con KV-cache)")
print("=" * 72)

for prompt in ["ROMEO:\n", "JULIET:\n", "HAMLET:\nO, "]:
    print(f"\n--- prompt: {prompt!r} ---")
    torch.manual_seed(42)
    print(sample(prompt, max_new_tokens=200))


# ---------------------------------------------------------------------------
# Resumen comparativo
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("COMPARACION: Mini-GPT (Vaswani 2017) vs Mini-LLaMA (LLaMA 2024)")
print("=" * 72)

print(f"""
                         Mini-GPT          Mini-LLaMA (este script)
                         ----------        --------------------------
LayerNorm                LayerNorm         RMSNorm
FFN                      FFN ReLU          SwiGLU (gate + up + down)
Positional encoding      Aprendido         RoPE (rotaciones)
Attention                MHA (h=4)         GQA (h_q=4, h_kv=2)
Generation               Sin cache         KV-cache
Parametros               816,128           {n_params:,}
Loss final (~3000 iter)  ~1.63             {loss.item():.4f}
Speedup generacion       1x (baseline)     {speedup:.2f}x con KV-cache

Notas:
  - Loss similar significa que las modernizaciones NO degradan calidad.
  - El speedup viene del KV-cache, no de las otras modernizaciones.
  - A escala (BERT-large, GPT-3, LLaMA 70B), las otras modernizaciones SI
    mejoran calidad significativamente.
  - GQA ahorra memoria del KV-cache: en este modelo chico el efecto es
    marginal, pero en LLaMA 2 70B reduce el cache de 10.7GB a 1.3GB por
    secuencia.

Las modernizaciones son INCREMENTALES. Cada una agrega un poquito.
Acumuladas, hacen la diferencia entre Vaswani 2017 (texto vagamente
gramatical) y LLaMA/Claude 2024 (asistentes utilizables).

Has construido la arquitectura completa de un LLM moderno desde cero.
""")

print("=" * 72)
print("FIN. Has llegado al estado del arte 2024 (en miniatura).")
print("=" * 72)
