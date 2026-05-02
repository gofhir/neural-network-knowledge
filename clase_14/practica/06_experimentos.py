"""
Experimentos rapidos sobre mini-GPT.

Tres experimentos focalizados:
  1. Efecto de la TEMPERATURA en sampling (sin retrenar).
  2. PROMPTS variados — como el modelo continua contextos distintos.
  3. Modelo MICRO vs ESTANDAR — capacidad vs velocidad.

Total: ~2-3 minutos en MPS.
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
# Cargar Shakespeare (mismo que escalon 8)
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
# Definicion del modelo (igual que escalon 8, parametrizable)
# ---------------------------------------------------------------------------
class CausalMHA(nn.Module):
    def __init__(self, d_model, h, block_size):
        super().__init__()
        assert d_model % h == 0
        self.d_model, self.h, self.d_k = d_model, h, d_model // h
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.W_Q(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        out = F.softmax(scores, dim=-1) @ V
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_O(out)


class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.l1, self.l2 = nn.Linear(d_model, d_ff), nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.l2(F.relu(self.l1(x)))


class Block(nn.Module):
    def __init__(self, d_model, h, d_ff, block_size):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = CausalMHA(d_model, h, block_size)
        self.ffn = FFN(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, h, n_layers, d_ff, block_size):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, h, d_ff, block_size) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x, targets=None):
        B, T = x.shape
        h = self.tok_emb(x) + self.pos_emb(torch.arange(T, device=x.device))
        for block in self.blocks:
            h = block(h)
        logits = self.head(self.ln_f(h))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_batch(split, block_size, batch_size):
    src = train_data if split == "train" else val_data
    ix = torch.randint(len(src) - block_size, (batch_size,))
    x = torch.stack([src[i:i + block_size] for i in ix])
    y = torch.stack([src[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def train_model(model, max_iters, batch_size, block_size, lr=3e-4, label="Modelo"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== Training {label} ({n_params:,} params) ===")
    start = time.time()
    for it in range(max_iters):
        x, y = get_batch("train", block_size, batch_size)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if it % (max_iters // 5) == 0 or it == max_iters - 1:
            print(f"  step {it:4d}: loss = {loss.item():.4f}  ({time.time()-start:.1f}s)")


def sample(model, prompt, max_new_tokens=200, temperature=1.0):
    model.eval()
    if prompt:
        ids = encode(prompt)
        ctx = torch.tensor([ids], dtype=torch.long, device=device)
    else:
        ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(ctx, max_new_tokens, temperature=temperature)
    model.train()
    return decode(out[0].tolist())


# ===========================================================================
# ENTRENAR DOS MODELOS
# ===========================================================================
print("=" * 72)
print("EXPERIMENTOS RAPIDOS SOBRE MINI-GPT")
print("=" * 72)
print(f"Dispositivo: {device}")
print(f"vocab_size: {vocab_size}")

# Modelo estandar (igual que escalon 8, pero menos iteraciones)
print("\n" + "=" * 72)
print("Modelo ESTANDAR (d_model=128, n_layers=4, h=4)")
print("=" * 72)
torch.manual_seed(1337)
model_std = MiniGPT(vocab_size, d_model=128, h=4, n_layers=4, d_ff=512, block_size=64).to(device)
train_model(model_std, max_iters=2000, batch_size=32, block_size=64, label="ESTANDAR")

# Modelo micro
print("\n" + "=" * 72)
print("Modelo MICRO (d_model=32, n_layers=2, h=2)")
print("=" * 72)
torch.manual_seed(1337)
model_micro = MiniGPT(vocab_size, d_model=32, h=2, n_layers=2, d_ff=128, block_size=64).to(device)
train_model(model_micro, max_iters=2000, batch_size=32, block_size=64, label="MICRO")


# ===========================================================================
# EXPERIMENTO 1: TEMPERATURA
# ===========================================================================
print("\n" + "=" * 72)
print("EXPERIMENTO 1: Efecto de la TEMPERATURA")
print("=" * 72)
print("""
La temperatura modifica como se samplea del softmax:
  temp < 1: distribucion mas pico (mas determinista, repetitivo)
  temp = 1: distribucion natural
  temp > 1: distribucion mas uniforme (mas creativo, mas errores)

Mismo modelo, mismo prompt 'ROMEO:'. Solo cambia la temperatura.
""")

torch.manual_seed(42)  # fijar para comparar
prompt = "ROMEO:"

for temp in [0.5, 1.0, 1.5]:
    torch.manual_seed(42)
    print(f"\n--- temperatura = {temp} ---")
    print(sample(model_std, prompt, max_new_tokens=200, temperature=temp))


# ===========================================================================
# EXPERIMENTO 2: PROMPTS VARIADOS
# ===========================================================================
print("\n" + "=" * 72)
print("EXPERIMENTO 2: Mismo modelo, prompts variados")
print("=" * 72)
print("""
Como el modelo aprendio la estructura de Shakespeare (nombres en mayuscula
seguidos de dialogos), distintos prompts inducen distintos estilos.
""")

prompts = [
    "ROMEO:",
    "JULIET:",
    "To be or not to ",
    "HAMLET:\nO that this too too solid ",
    "Friends, Romans, ",
]

for p in prompts:
    print(f"\n--- prompt: {p!r} ---")
    torch.manual_seed(42)
    print(sample(model_std, p, max_new_tokens=150, temperature=0.8))


# ===========================================================================
# EXPERIMENTO 3: MICRO vs ESTANDAR
# ===========================================================================
print("\n" + "=" * 72)
print("EXPERIMENTO 3: Modelo MICRO vs ESTANDAR (mismo prompt)")
print("=" * 72)
print("""
Mismo prompt para ambos modelos. La diferencia de calidad muestra cuanta
"capacidad" se necesita para aprender la estructura del lenguaje.
""")

prompt = "ROMEO:\n"

print("\n--- MICRO (d_model=32, n_layers=2) — pocas neuronas ---")
torch.manual_seed(42)
print(sample(model_micro, prompt, max_new_tokens=200, temperature=0.8))

print("\n--- ESTANDAR (d_model=128, n_layers=4) — capacidad estandar ---")
torch.manual_seed(42)
print(sample(model_std, prompt, max_new_tokens=200, temperature=0.8))


# ===========================================================================
# RESUMEN
# ===========================================================================
print("\n" + "=" * 72)
print("OBSERVACIONES CLAVE")
print("=" * 72)
n_std = sum(p.numel() for p in model_std.parameters())
n_micro = sum(p.numel() for p in model_micro.parameters())

print(f"""
Modelo MICRO:    {n_micro:>9,} parametros
Modelo ESTANDAR: {n_std:>9,} parametros
Diferencia:      {n_std/n_micro:.1f}x mas grande

EFECTO DE TEMPERATURA:
  temp 0.5 -> texto mas determinista, repetitivo, "seguro"
  temp 1.0 -> balance natural
  temp 1.5 -> mas variedad, mas errores tipograficos, mas creativo

EFECTO DE LOS PROMPTS:
  El modelo aprendio que despues de "X:" viene un parlamento.
  Diferentes nombres -> diferentes estilos de continuacion.

EFECTO DEL TAMAÑO:
  El modelo MICRO produce texto mas "tonto" — palabras mas cortas,
  estructura mas rota. Aun asi aprende lo basico (mayusculas, ":").
  El modelo ESTANDAR captura estructura y vocabulario mas rico.

  En LLMs grandes (GPT-3 con 175B params) este efecto es DRAMATICO:
  emergen capacidades nuevas (few-shot learning, razonamiento) solo
  con escala.
""")

print("=" * 72)
print("FIN experimentos.")
print("=" * 72)
