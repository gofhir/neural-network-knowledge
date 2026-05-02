"""
Escalon 7 - Bloque Transformer completo.

Combina las piezas:
  1. Multi-Head Self-Attention (escalon 6)
  2. Feed-Forward Network (FFN)
  3. Residual connections (suma del input al output de cada sub-bloque)
  4. Layer Normalization (estabilizar training)

Estructura (pre-norm, el estandar moderno):

    x = x + Attention(LayerNorm(x))
    x = x + FFN(LayerNorm(x))

Con esto tenemos UNA capa del encoder. Apilando N de estas, tenemos
el Transformer encoder completo (a la BERT). Es lo que vas a ver en
los ultimos pasos del script.
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
print("=" * 72)
print("PASO 1: Setup")
print("=" * 72)

T = 4
d_model = 16
h = 4
d_k = d_model // h
d_ff = 4 * d_model  # 64 — convencion estandar
batch_size = 1

x = torch.randn(batch_size, T, d_model)
print(f"\nT = {T} tokens")
print(f"d_model = {d_model}")
print(f"h = {h} cabezas, d_k = {d_k} por cabeza")
print(f"d_ff = {d_ff} (capa interna del FFN)")
print(f"\nInput x.shape: {x.shape}")


# ---------------------------------------------------------------------------
# PIEZA 1: Multi-Head Attention (igual que escalon 6)
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("PASO 2: Multi-Head Attention (recordatorio)")
print("=" * 72)


class MultiHeadAttention(nn.Module):
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
        Q = self.W_Q(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.h, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        weights = F.softmax(scores, dim=-1)
        head_outputs = weights @ V

        concat = head_outputs.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_O(concat)


mha = MultiHeadAttention(d_model, h)
out_attn = mha(x)
print(f"\nMHA output shape: {out_attn.shape}")


# ---------------------------------------------------------------------------
# PIEZA 2: FFN — feed-forward position-wise
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("PASO 3: FFN (Feed-Forward Network)")
print("=" * 72)

print("""
Estructura: 2 capas lineales con ReLU (o GELU) entre ellas.
  FFN(x) = max(0, x W_1 + b_1) W_2 + b_2

Donde:
  W_1: d_model -> d_ff   (expande)
  W_2: d_ff -> d_model   (contrae)

La FFN se aplica independientemente a cada token (las matrices son las
mismas para todos los T tokens del batch).

Convencionalmente d_ff = 4 * d_model (Vaswani usa 2048 cuando d_model=512).
""")


class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        h = self.linear1(x)        # (B, T, d_ff)
        h = F.relu(h)              # no linealidad
        out = self.linear2(h)      # (B, T, d_model)
        return out


ffn = FFN(d_model, d_ff)
out_ffn = ffn(x)
print(f"FFN params: {sum(p.numel() for p in ffn.parameters())}")
print(f"FFN input shape:  {x.shape}")
print(f"FFN output shape: {out_ffn.shape}")
print(f"\nNotar: dimensiones de entrada y salida son iguales (d_model = {d_model}).")
print(f"Lo que cambia es el contenido — la FFN procesa cada token internamente.")


# ---------------------------------------------------------------------------
# PIEZA 3: Residual connections
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("PASO 4: Residual connections (suma del input)")
print("=" * 72)

print("""
Patron: y = x + Sublayer(x)

El input se SUMA al output del sub-bloque. Esto:
  1. Permite que el gradiente fluya directo (vanishing gradient solucionado).
  2. Permite al modelo "ignorar" un sub-bloque si no aporta (Sublayer(x) ≈ 0
     -> y ≈ x).

Demo: comparar y SIN residual vs CON residual.
""")

# Sin residual: y = Sublayer(x)
y_no_residual = mha(x)

# Con residual: y = x + Sublayer(x)
y_with_residual = x + mha(x)

print(f"y SIN residual norm:  {y_no_residual.norm():.3f}")
print(f"y CON residual norm:  {y_with_residual.norm():.3f}")
print(f"\n(x + sublayer(x)) preserva la magnitud del input. Es como un")
print(f"'cable expreso' por el que el input atraviesa la red sin atenuarse.")


# ---------------------------------------------------------------------------
# PIEZA 4: LayerNorm
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("PASO 5: LayerNorm")
print("=" * 72)

print("""
Normaliza las activaciones a lo largo de la dim de features (d_model).
Para cada token, calcula su media y desviacion estandar SOBRE sus d_model
dims, luego normaliza:

    LayerNorm(x) = gamma * (x - mu) / sqrt(sigma^2 + eps) + beta

donde gamma, beta son parametros aprendibles (escala y shift).

Diferente de BatchNorm:
  BatchNorm:  normaliza a traves del BATCH (mismo feature, distintos batch).
  LayerNorm:  normaliza a traves de FEATURES (mismo token, distintos features).

LayerNorm es el estandar en Transformers: independiente del batch size,
funciona con secuencias variables, ideal para NLP.
""")

ln = nn.LayerNorm(d_model)
y_normalized = ln(x)

print(f"x[0, 0]: {x[0, 0]}")
print(f"x[0, 0] mean = {x[0, 0].mean().item():.3f}, std = {x[0, 0].std().item():.3f}")
print(f"\nDespues de LayerNorm:")
print(f"y[0, 0]: {y_normalized[0, 0]}")
print(f"y[0, 0] mean = {y_normalized[0, 0].mean().item():.3f}, std = {y_normalized[0, 0].std().item():.3f}")
print(f"\nLa media bajo a ~0 y la std subio a ~1. Eso es lo que hace LayerNorm.")


# ---------------------------------------------------------------------------
# Bloque Transformer completo
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("PASO 6: TransformerBlock — combinando las 4 piezas")
print("=" * 72)

print("""
Pre-norm (estandar moderno):
    x = x + MultiHeadAttention(LayerNorm(x))   # sub-bloque 1
    x = x + FFN(LayerNorm(x))                  # sub-bloque 2

Notar: la LayerNorm se aplica ANTES del sub-bloque (no despues como en
Vaswani original). Esto es mas estable y se usa en GPT-2+, BERT moderno,
LLaMA, etc.
""")


class TransformerBlock(nn.Module):
    def __init__(self, d_model, h, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, h)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff)

    def forward(self, x):
        # Sub-bloque 1: attention con residual y pre-norm
        x = x + self.mha(self.ln1(x))
        # Sub-bloque 2: FFN con residual y pre-norm
        x = x + self.ffn(self.ln2(x))
        return x


block = TransformerBlock(d_model, h, d_ff)
out_block = block(x)

n_params = sum(p.numel() for p in block.parameters())
print(f"\nUn bloque Transformer:")
print(f"  Parametros: {n_params}")
print(f"  Input shape:  {x.shape}")
print(f"  Output shape: {out_block.shape}")
print(f"\nLas dimensiones se preservan: (B, T, d_model) -> (B, T, d_model)")
print(f"Eso permite apilar bloques sin friccion.")


# ---------------------------------------------------------------------------
# Apilar N bloques: encoder Transformer completo
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("PASO 7: Stack de N bloques = Transformer Encoder")
print("=" * 72)

print("""
Un Transformer Encoder es simplemente N bloques apilados. Cada bloque
preserva la dim (B, T, d_model), entonces se pueden encadenar sin
problemas.

Vaswani usa N=6. BERT-base usa N=12. GPT-3 usa N=96. La arquitectura
es la MISMA — solo cambia la profundidad.
""")


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, h, d_ff, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, h, d_ff) for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.ln_final(x)


N = 6  # como Vaswani-base
encoder = TransformerEncoder(d_model, h, d_ff, N)
out_encoder = encoder(x)

n_params_encoder = sum(p.numel() for p in encoder.parameters())
print(f"Encoder con N={N} capas:")
print(f"  Parametros totales: {n_params_encoder}")
print(f"  Por capa: {n_params}")
print(f"  Input shape:  {x.shape}")
print(f"  Output shape: {out_encoder.shape}")


# ---------------------------------------------------------------------------
# Comparacion: 1 capa vs N capas
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("PASO 8: Que cambia al apilar mas capas?")
print("=" * 72)

print("""
Cada capa es una transformacion nueva. Apilar mas capas = transformaciones
mas profundas = representaciones mas abstractas.

Capas tempranas: capturan patrones locales (vocabulario, sintaxis basica).
Capas medias:    capturan estructura sintactica (relaciones gramaticales).
Capas tardias:   capturan semantica abstracta (significado, intencion).

Esto se observa empiricamente en BERT (Tenney et al. 2019, "BERT
Rediscovers the Classical NLP Pipeline").
""")

# Calculemos el norm del output con distintas profundidades
for n_layers in [1, 2, 6, 12]:
    enc = TransformerEncoder(d_model, h, d_ff, n_layers)
    with torch.no_grad():
        out = enc(x)
    print(f"  N={n_layers:2d}: params={sum(p.numel() for p in enc.parameters()):>7}   "
          f"output mean={out.mean().item():.3f}   std={out.std().item():.3f}")


# ---------------------------------------------------------------------------
# Conexion con modelos reales
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("PASO 9: Conexion con modelos reales")
print("=" * 72)

print("""
Lo que acabas de construir es CASI un BERT/GPT completo. Lo unico que
falta es:

  1. Embedding layer (vocab -> d_model). Lo hicimos en escalon 1d.
  2. Positional encoding (sumar info de posicion al embedding).
  3. Output head (d_model -> vocab para next-token prediction o lo que sea).

Comparativa con modelos reales:

| Modelo            | d_model | h  | d_ff | N  | Params totales |
|-------------------|---------|----|----- |----|---------------|
| Nuestro encoder   |     16  |  4 |   64 |  6 |   ~22 K       |
| BERT-base         |    768  | 12 | 3072 | 12 |    110 M      |
| BERT-large        |   1024  | 16 | 4096 | 24 |    340 M      |
| GPT-2 small       |    768  | 12 | 3072 | 12 |    124 M      |
| GPT-2 XL          |   1600  | 25 | 6400 | 48 |    1.5 B      |
| GPT-3             |  12288  | 96 | 49152| 96 |    175 B      |
| LLaMA 2 70B       |   8192  | 64 | 28672| 80 |    70 B       |

La ARQUITECTURA es la misma. Lo que cambia es la escala. Cuando entiendas
nuestro mini-encoder de 22K parametros, entiendes la estructura de GPT-3
de 175B parametros. Solo cambia la magnitud.
""")


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------
print("=" * 72)
print("RESUMEN")
print("=" * 72)

print("""
Lo que aprendimos:

1. FFN: red de 2 capas con ReLU/GELU entre, aplicada independientemente
   a cada token. Procesa cada token "verticalmente" mientras attention
   mezcla "horizontalmente".

2. Residual connections (y = x + Sublayer(x)): solucionan el vanishing
   gradient en redes profundas. Hacen el modelo entrenable con muchas capas.

3. LayerNorm: normaliza activaciones a lo largo de features. Independiente
   del batch, ideal para NLP. Estandar en Transformers.

4. Bloque Transformer: combinacion (pre-norm)
     x = x + MHA(LN(x))
     x = x + FFN(LN(x))

5. Encoder: N bloques apilados. Misma estructura, mas capas = mas profundidad.

Lo que sigue (escalon 8): MINI-GPT entrenado en Shakespeare.

Para llegar al GPT real falta:
  - Embedding + positional encoding (input layer).
  - Causal mask (decoder-only: solo atender al pasado).
  - Output head (proyectar a vocab + softmax + cross-entropy).
  - Training loop con un dataset de texto real.

Y al correrlo, vas a ver al modelo APRENDER A GENERAR texto. Ese es el
"click" final de todo el viaje.
""")

print("=" * 72)
print("FIN escalon 7.")
print("=" * 72)
