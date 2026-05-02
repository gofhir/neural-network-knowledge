"""
Escalon 6 - Multi-Head Attention.

Objetivo: ejecutar h atenciones en paralelo, cada una en un subespacio
distinto. Cada cabeza puede especializarse en un tipo distinto de relacion
(sujeto-verbo, modificador-sustantivo, etc.).

El paso conceptual: una sola cabeza fuerza UNA distribucion de pesos por
token, lo que limita la expresividad. Multi-head ejecuta h cabezas en
paralelo, concatena los resultados, y aplica una proyeccion final.

Implementamos primero la version "naive" (h Linear modules separados,
facil de leer) y luego la "eficiente" (un solo Linear + reshape, el
patron usado en produccion). Verificamos que son equivalentes.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
torch.set_printoptions(precision=3, sci_mode=False)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
print("=" * 70)
print("PASO 1: Setup")
print("=" * 70)

vocab = ["I", "love", "neural", "networks"]
d_model = 8
T = len(vocab)
batch_size = 1

X = torch.randn(batch_size, T, d_model)
print(f"\nVocabulario: {vocab}")
print(f"d_model: {d_model}")
print(f"X shape: {X.shape}")  # (1, 4, 8)


# ---------------------------------------------------------------------------
# Recordar: SelfAttention de una sola cabeza (escalon 2)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 2: Recordatorio de single-head attention")
print("=" * 70)


class SingleHeadAttention(nn.Module):
    """Lo que construimos en escalon 2: una sola cabeza."""
    def __init__(self, d_model):
        super().__init__()
        self.d_k = d_model
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        weights = F.softmax(scores, dim=-1)
        output = weights @ V
        return output, weights


single_head = SingleHeadAttention(d_model)
out_sh, w_sh = single_head(X)
print(f"\nSingle-head output shape: {out_sh.shape}")
print(f"Single-head weights shape: {w_sh.shape}")
print(f"\nMatriz de pesos (UNA distribucion por token):")
print(w_sh[0])
print("""
Cada fila es UNA distribucion sobre los 4 tokens. Si el modelo quiere
atender a "sujeto" Y "objeto" simultaneamente, no puede — el softmax
lo obliga a distribuir entre ambos.
""")


# ---------------------------------------------------------------------------
# Multi-head NAIVE: h Linear modules separados
# ---------------------------------------------------------------------------
print("=" * 70)
print("PASO 3: Multi-head NAIVE (h cabezas separadas, facil de leer)")
print("=" * 70)


class MultiHeadAttentionNaive(nn.Module):
    """
    Version conceptualmente clara: una lista de h SingleHeadAttention.
    Es como ejecutar h independientes en paralelo y concatenar.
    """
    def __init__(self, d_model, h):
        super().__init__()
        assert d_model % h == 0, "d_model debe ser divisible por h"
        self.h = h
        self.d_k = d_model // h

        # h conjuntos de matrices W_Q, W_K, W_V (cada una proyecta
        # de d_model a d_k, no a d_model como en single-head).
        self.heads_W_Q = nn.ModuleList([
            nn.Linear(d_model, self.d_k, bias=False) for _ in range(h)
        ])
        self.heads_W_K = nn.ModuleList([
            nn.Linear(d_model, self.d_k, bias=False) for _ in range(h)
        ])
        self.heads_W_V = nn.ModuleList([
            nn.Linear(d_model, self.d_k, bias=False) for _ in range(h)
        ])

        # Proyeccion final que mezcla las h cabezas.
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        head_outputs = []
        all_weights = []

        # Ejecutar cada cabeza independientemente
        for i in range(self.h):
            Q = self.heads_W_Q[i](x)  # (B, T, d_k)
            K = self.heads_W_K[i](x)
            V = self.heads_W_V[i](x)

            scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
            weights = F.softmax(scores, dim=-1)
            output = weights @ V          # (B, T, d_k)

            head_outputs.append(output)
            all_weights.append(weights)

        # Concatenar las h salidas a lo largo de la dim feature
        concat = torch.cat(head_outputs, dim=-1)  # (B, T, h * d_k) = (B, T, d_model)

        # Proyectar de vuelta
        output = self.W_O(concat)         # (B, T, d_model)

        # Apilar weights para visualizacion: (B, h, T, T)
        all_weights = torch.stack(all_weights, dim=1)

        return output, all_weights


h = 4  # numero de cabezas
mha_naive = MultiHeadAttentionNaive(d_model, h=h)
out_naive, weights_naive = mha_naive(X)

print(f"\nh = {h} cabezas")
print(f"d_k por cabeza: {d_model // h}")
print(f"\nOutput shape: {out_naive.shape}")
print(f"Weights shape: {weights_naive.shape}  (B, h, T, T)")

print(f"\nCada cabeza produce su propia matriz de pesos:")
for head_idx in range(h):
    print(f"\nCabeza {head_idx}:")
    print(weights_naive[0, head_idx])
print("""
Las h cabezas tienen distribuciones distintas. Cada una puede aprender a
atender a un patron de relacion distinto. Esa es la ganancia de expresividad
sobre single-head.
""")


# ---------------------------------------------------------------------------
# Multi-head EFICIENTE: un solo Linear + reshape
# ---------------------------------------------------------------------------
print("=" * 70)
print("PASO 4: Multi-head EFICIENTE (un solo Linear + reshape, produccion)")
print("=" * 70)


class MultiHeadAttention(nn.Module):
    """
    Implementacion eficiente, la usada en BERT/GPT/Vaswani.

    En vez de h Linear modules separados, usamos UN solo Linear que
    proyecta de d_model a d_model = h * d_k en una sola operacion. Luego
    reshape para dividir en h cabezas y permute para que la dim de
    cabezas quede al frente, asi PyTorch puede vectorizar todas las
    cabezas a la vez.

    Matematicamente equivalente a la version naive.
    """
    def __init__(self, d_model, h):
        super().__init__()
        assert d_model % h == 0
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, _ = x.shape

        # Proyectar Q, K, V de una sola vez (a d_model entero)
        Q = self.W_Q(x)  # (B, T, d_model)
        K = self.W_K(x)
        V = self.W_V(x)

        # RESHAPE: dividir d_model = h * d_k en dos dimensiones
        # (B, T, d_model) -> (B, T, h, d_k)
        Q = Q.view(B, T, self.h, self.d_k)
        K = K.view(B, T, self.h, self.d_k)
        V = V.view(B, T, self.h, self.d_k)

        # PERMUTE: poner la dim de cabezas al frente
        # (B, T, h, d_k) -> (B, h, T, d_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Atencion vectorizada en todas las cabezas a la vez
        # scores: (B, h, T, T)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        weights = F.softmax(scores, dim=-1)
        head_outputs = weights @ V        # (B, h, T, d_k)

        # Volver a juntar las cabezas
        # (B, h, T, d_k) -> (B, T, h, d_k) -> (B, T, d_model)
        head_outputs = head_outputs.transpose(1, 2).contiguous()
        concat = head_outputs.view(B, T, self.d_model)

        # Proyeccion final
        output = self.W_O(concat)

        return output, weights


mha_efficient = MultiHeadAttention(d_model, h=h)
out_eff, weights_eff = mha_efficient(X)

print(f"\nOutput shape: {out_eff.shape}")
print(f"Weights shape: {weights_eff.shape}  (B, h, T, T)")
print(f"\nNumero de parametros (naive vs eficiente):")

def count_params(m):
    return sum(p.numel() for p in m.parameters())

print(f"  Naive:     {count_params(mha_naive)} parametros")
print(f"  Eficiente: {count_params(mha_efficient)} parametros")
print("""
Ambas tienen el mismo numero de parametros aprendibles (matematicamente
equivalentes). La diferencia es que la eficiente hace el calculo en una
sola pasada vectorizada, sin loop sobre cabezas. En GPU esto es mucho
mas rapido y es la forma estandar.
""")


# ---------------------------------------------------------------------------
# Verificacion: ambas dan resultados equivalentes (con mismos pesos)
# ---------------------------------------------------------------------------
print("=" * 70)
print("PASO 5: Verificar que naive y eficiente son equivalentes")
print("=" * 70)

# Para que el resultado sea identico, copiamos los pesos de la version naive
# a la eficiente. Esto demuestra que las dos implementaciones son
# matematicamente equivalentes — la diferencia es solo la organizacion del
# computo.

with torch.no_grad():
    # W_Q de la naive: h matrices de shape (d_k, d_model) cada una.
    # W_Q de la eficiente: una matriz de shape (d_model, d_model) que
    # representa las h matrices apiladas.
    W_Q_concat = torch.cat([h_layer.weight for h_layer in mha_naive.heads_W_Q], dim=0)
    W_K_concat = torch.cat([h_layer.weight for h_layer in mha_naive.heads_W_K], dim=0)
    W_V_concat = torch.cat([h_layer.weight for h_layer in mha_naive.heads_W_V], dim=0)

    mha_efficient.W_Q.weight.copy_(W_Q_concat)
    mha_efficient.W_K.weight.copy_(W_K_concat)
    mha_efficient.W_V.weight.copy_(W_V_concat)
    mha_efficient.W_O.weight.copy_(mha_naive.W_O.weight)

# Ahora ambas deberian dar el mismo output
out_naive2, _ = mha_naive(X)
out_eff2, _ = mha_efficient(X)

diff = (out_naive2 - out_eff2).abs().max().item()
print(f"\nMax diferencia entre output naive y eficiente: {diff:.2e}")
print("Si es ~0, ambas implementaciones son matematicamente equivalentes.")


# ---------------------------------------------------------------------------
# Lo que cada cabeza "ve"
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 6: Visualizar lo que cada cabeza atiende")
print("=" * 70)

print("""
En un Transformer entrenado, las cabezas se especializan en distintos
tipos de relaciones (Voita et al. 2019, Clark et al. 2019). Algunas
cabezas atienden a sujeto-verbo, otras a modificadores, otras al token
anterior, otras a tokens lejanos.

En este modelo SIN entrenar las distribuciones son arbitrarias, pero
puedes ver que cada cabeza tiene su propio patron:
""")

with torch.no_grad():
    _, weights = mha_efficient(X)

for head_idx in range(h):
    print(f"\nCabeza {head_idx} (donde mira cada token):")
    for i, word in enumerate(vocab):
        row = weights[0, head_idx, i]
        max_attended = row.argmax().item()
        max_weight = row.max().item()
        print(f"  {word:>9} -> {vocab[max_attended]:>9} (peso {max_weight:.2f})  "
              f"distribucion: {row.tolist()}")


# ---------------------------------------------------------------------------
# Hyperparametros canonicos
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 7: Hyperparametros canonicos del Transformer-base (Vaswani)")
print("=" * 70)

print("""
| Modelo                | d_model | h  | d_k | params/capa |
|-----------------------|---------|----|----|-------------|
| Transformer-base      |    512  |  8 | 64 |       2.6M  |
| Transformer-big       |   1024  | 16 | 64 |      10.5M  |
| BERT-base             |    768  | 12 | 64 |       7.1M  |
| BERT-large            |   1024  | 16 | 64 |      12.6M  |
| GPT-2 small           |    768  | 12 | 64 |       7.1M  |
| GPT-2 large           |   1280  | 20 | 64 |      19.7M  |
| GPT-3                 |  12288  | 96 |128 |     600M+   |

Notar: d_k = 64 es el "default" historico (Vaswani lo eligio asi).
GPT-3 lo subio a 128 porque a mayor escala se justifica.

Numero de parametros por capa de multi-head:
  4 matrices d_model x d_model = 4 * d_model^2
  (W_Q, W_K, W_V, W_O)

Por ejemplo BERT-base: 4 * 768^2 = 2.36M solo para attention.
Mas FFN: 8 * 768^2 = 4.72M.
Total por capa BERT-base: ~7M parametros.
12 capas * 7M = 85M parametros solo en bloques Transformer.
""")


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------
print("=" * 70)
print("RESUMEN")
print("=" * 70)

print("""
Lo que aprendimos:

1. Single-head sufre de "compromiso de softmax": una distribucion no puede
   atender fuertemente a multiples cosas a la vez.
2. Multi-head ejecuta h atenciones en paralelo, cada una en un subespacio
   distinto. Cada cabeza puede especializarse en un tipo de relacion.
3. Conceptualmente: h matrices W_Q^i, W_K^i, W_V^i mas un W_O final.
4. En produccion: un solo W_Q (a d_model = h*d_k) + reshape para dividir
   en cabezas. Vectorizado, sin loop Python.
5. Numero de parametros igual entre naive y eficiente. Diferencia es solo
   la organizacion del computo.

Lo que sigue (escalon 7): el BLOQUE TRANSFORMER COMPLETO.

Multi-head es solo UNA pieza. El bloque Transformer combina:
  1. Multi-head self-attention
  2. Add & Norm (residual + LayerNorm)
  3. Feed-Forward Network (FFN) position-wise
  4. Add & Norm (residual + LayerNorm)

Apilando N de estos bloques, mas embeddings + positional encoding + un
output head, tienes el Transformer completo. Ese es el siguiente capitulo.
""")

print("=" * 70)
print("FIN escalon 6.")
print("=" * 70)
