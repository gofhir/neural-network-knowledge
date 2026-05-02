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


# ---------------------------------------------------------------------------
# Mini-LLaMA: RMSNorm + SwiGLU + RoPE + GQA + KV-cache
# ---------------------------------------------------------------------------
# NOTE: classes below mirror 13_mini_llama.py — keep in sync.
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich 2019)."""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.gamma * x / rms


class SwiGLU(nn.Module):
    """SwiGLU FFN (Shazeer 2020): tres proyecciones con gating Swish."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def precompute_rope(d_k, max_seq_len, theta=10000.0):
    """Precompute cos/sin tables para RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, freqs)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Aplicar rotacion RoPE. x: (B, h, T, d_k); cos/sin: (T, d_k/2)."""
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos
    return torch.cat([rotated_x1, rotated_x2], dim=-1)


class GroupedQueryAttention(nn.Module):
    """GQA con RoPE y KV-cache. h_q cabezas Q, h_kv cabezas K/V (h_kv | h_q)."""
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

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask.view(1, 1, max_seq_len, max_seq_len))

        cos, sin = precompute_rope(self.d_k, max_seq_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        self.cache_k = None
        self.cache_v = None

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, use_cache=False):
        B, T, _ = x.shape
        Q = self.W_Q(x).view(B, T, self.h_q, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.h_kv, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.h_kv, self.d_k).transpose(1, 2)

        if use_cache and self.cache_k is not None:
            cache_len = self.cache_k.size(2)
            cos = self.rope_cos[cache_len:cache_len + T]
            sin = self.rope_sin[cache_len:cache_len + T]
        else:
            cos = self.rope_cos[:T]
            sin = self.rope_sin[:T]

        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)

        if use_cache:
            if self.cache_k is not None:
                K = torch.cat([self.cache_k, K], dim=2)
                V = torch.cat([self.cache_v, V], dim=2)
            self.cache_k = K
            self.cache_v = V

        K_full = K.repeat_interleave(self.group_size, dim=1)
        V_full = V.repeat_interleave(self.group_size, dim=1)

        scores = Q @ K_full.transpose(-2, -1) / math.sqrt(self.d_k)

        T_q, T_k = scores.size(-2), scores.size(-1)
        q_start = T_k - T_q
        mask_slice = self.mask[:, :, q_start:q_start + T_q, :T_k]
        scores = scores.masked_fill(mask_slice == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        out = weights @ V_full
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        return self.W_O(out)


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


class MiniLLaMA(nn.Module):
    """Mini-LLaMA: RMSNorm + SwiGLU + RoPE + GQA + KV-cache."""
    def __init__(self, vocab_size, d_model, h_q, h_kv, n_layers, d_ff, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len
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
        h = self.tok_emb(x)
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
            self(idx, use_cache=True)
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


def load_pretrained_mini_llama(checkpoint_path, device=None, config=None):
    """Carga Mini-LLaMA desde checkpoint. config dict con keys del constructor."""
    if device is None:
        device = get_device()
    if config is None:
        config = dict(vocab_size=65, max_seq_len=256, d_model=128,
                      h_q=4, h_kv=2, n_layers=4, d_ff=384)
    model = MiniLLaMA(**config)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate_with_prompt(model, prompt, char_to_id, id_to_char, max_new_tokens=50,
                        temperature=1.0, top_k=None, device=None, stop_token="\n"):
    """Genera texto condicionado en prompt char-level. Devuelve prompt + completion."""
    if device is None:
        device = get_device()
    model.eval()
    ids = [char_to_id[c] for c in prompt if c in char_to_id]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        x_cond = x[:, -model.max_seq_len:]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)
        if stop_token is not None and id_to_char.get(next_id.item(), "") == stop_token:
            break
    out_ids = x[0].tolist()
    return "".join(id_to_char.get(i, "") for i in out_ids)


def compute_logp_response(model, prompt_ids, response_ids, device=None):
    """log P(response | prompt) = sum log p_t para tokens de response."""
    if device is None:
        device = get_device()
    full = torch.cat([prompt_ids, response_ids]).to(device).unsqueeze(0)
    inp = full[:, :-1]
    tgt = full[:, 1:]
    logits, _ = model(inp)  # (1, T-1, V)
    logp = torch.log_softmax(logits, dim=-1)
    n_p = prompt_ids.shape[0]
    resp_logits = logp[:, n_p-1:, :]              # (1, R, V)
    resp_targets = tgt[:, n_p-1:].unsqueeze(-1)   # (1, R, 1)
    chosen = resp_logits.gather(-1, resp_targets).squeeze(-1)  # (1, R)
    return chosen.sum()
