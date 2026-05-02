"""
Modulo compartido con las clases del Mini-GPT.

Importado por scripts 07-12 (experimentos). Permite no duplicar codigo.
"""

import math
import os
import time
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device():
    return (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )


# ---------------------------------------------------------------------------
# Tokenizer char-level
# ---------------------------------------------------------------------------
class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.char_to_id = {c: i for i, c in enumerate(self.chars)}
        self.id_to_char = {i: c for i, c in enumerate(self.chars)}

    def encode(self, s: str):
        return [self.char_to_id[c] for c in s if c in self.char_to_id]

    def decode(self, ids):
        return ''.join(self.id_to_char[int(i)] for i in ids)


# ---------------------------------------------------------------------------
# Componentes del Transformer
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
    """FFN con activacion configurable (relu o gelu)."""
    def __init__(self, d_model, d_ff, activation="relu"):
        super().__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)
        self.activation = activation

    def forward(self, x):
        h = self.l1(x)
        if self.activation == "relu":
            h = F.relu(h)
        elif self.activation == "gelu":
            h = F.gelu(h)
        else:
            raise ValueError(f"Activacion desconocida: {self.activation}")
        return self.l2(h)


class Block(nn.Module):
    def __init__(self, d_model, h, d_ff, block_size, activation="relu"):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = CausalMHA(d_model, h, block_size)
        self.ffn = FFN(d_model, d_ff, activation=activation)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, h, n_layers, d_ff, block_size, activation="relu"):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, h, d_ff, block_size, activation=activation)
            for _ in range(n_layers)
        ])
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
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k sampling: solo considerar los k tokens mas probables
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ---------------------------------------------------------------------------
# Helpers de training
# ---------------------------------------------------------------------------
def make_batch_fn(train_data, val_data, block_size, batch_size, device):
    def get_batch(split):
        src = train_data if split == "train" else val_data
        ix = torch.randint(len(src) - block_size, (batch_size,))
        x = torch.stack([src[i:i + block_size] for i in ix])
        y = torch.stack([src[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)
    return get_batch


def train(model, get_batch, max_iters, lr=3e-4, log_every=None, label="modelo"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    log_every = log_every or max(1, max_iters // 10)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Training {label}: {n_params:,} params, {max_iters} iters]")
    start = time.time()
    losses = []
    for it in range(max_iters):
        x, y = get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if it % log_every == 0 or it == max_iters - 1:
            print(f"  step {it:5d}: loss = {loss.item():.4f}  ({time.time()-start:.1f}s)")
    return losses


def sample(model, tokenizer, prompt, max_new_tokens=200, temperature=1.0, top_k=None, device="cpu"):
    model.eval()
    if prompt:
        ids = tokenizer.encode(prompt)
        ctx = torch.tensor([ids], dtype=torch.long, device=device)
    else:
        ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(ctx, max_new_tokens, temperature=temperature, top_k=top_k)
    model.train()
    return tokenizer.decode(out[0].tolist())


# ---------------------------------------------------------------------------
# Cargar dataset (default: Shakespeare)
# ---------------------------------------------------------------------------
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_text(local_path: str = "shakespeare.txt", url: str = SHAKESPEARE_URL):
    if not os.path.exists(local_path):
        print(f"Descargando {url}...")
        urllib.request.urlretrieve(url, local_path)
    with open(local_path, "r") as f:
        return f.read()


def prepare_data(text, tokenizer=None, train_split=0.9):
    if tokenizer is None:
        tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(train_split * len(data))
    return tokenizer, data[:n], data[n:]
