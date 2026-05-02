"""
Escalon 2 - Q, K, V aprendibles + scaling por sqrt(d_k).

Objetivo: arreglar las dos limitaciones del escalon 1:
  1. Q = K = V = X era rigido. Ahora Q, K, V son proyecciones lineales
     APRENDIBLES distintas: q = x W_Q, k = x W_K, v = x W_V.
  2. El softmax saturaba con dot products grandes. Ahora dividimos por
     sqrt(d_k) antes del softmax para mantener la varianza controlada.

Resultado: self-attention "real", la operacion central del Transformer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
torch.set_printoptions(precision=3, sci_mode=False)


# ---------------------------------------------------------------------------
# Setup: la misma oracion del escalon 1
# ---------------------------------------------------------------------------
print("=" * 70)
print("PASO 1: Setup (mismo input que escalon 1)")
print("=" * 70)

vocab = ["I", "love", "neural", "networks"]
d_model = 8  # ahora un poco mas grande para que sea mas realista
T = len(vocab)

# Embedding random (en la vida real esto se aprende como en escalon 1d).
embedding = torch.randn(len(vocab), d_model)
X = embedding[torch.arange(T)]  # (T, d_model)

print(f"\nVocabulario: {vocab}")
print(f"d_model:     {d_model}")
print(f"X shape:     {X.shape}")
print(f"X[0] (embedding de 'I'): {X[0]}")


# ---------------------------------------------------------------------------
# Las 3 proyecciones lineales aprendibles
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 2: Proyecciones Q, K, V con nn.Linear")
print("=" * 70)

print("""
nn.Linear(in_dim, out_dim, bias=False) es una capa que aplica:
    salida = entrada @ W.T

donde W es una matriz (out_dim, in_dim). Es UNA matriz aprendible.
Es el ladrillo basico de PyTorch para transformaciones lineales.

Para self-attention necesitamos TRES proyecciones independientes:
  W_Q: proyecta x al espacio "query"
  W_K: proyecta x al espacio "key"
  W_V: proyecta x al espacio "value"

Tipicamente d_k = d_v = d_model (en single-head). En multi-head se divide.
""")

d_k = d_model  # dimension de q, k
d_v = d_model  # dimension de v

W_Q = nn.Linear(d_model, d_k, bias=False)
W_K = nn.Linear(d_model, d_k, bias=False)
W_V = nn.Linear(d_model, d_v, bias=False)

# Verificar que son parametros aprendibles
print(f"W_Q.weight.shape: {W_Q.weight.shape}, requires_grad: {W_Q.weight.requires_grad}")
print(f"W_K.weight.shape: {W_K.weight.shape}, requires_grad: {W_K.weight.requires_grad}")
print(f"W_V.weight.shape: {W_V.weight.shape}, requires_grad: {W_V.weight.requires_grad}")

# Aplicar las proyecciones
Q = W_Q(X)  # (T, d_k)
K = W_K(X)  # (T, d_k)
V = W_V(X)  # (T, d_v)

print(f"\nQ shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")
print(f"\nQ[0] (query de 'I'):  {Q[0]}")
print(f"K[0] (key de 'I'):    {K[0]}")
print(f"V[0] (value de 'I'):  {V[0]}")
print(f"X[0] (embedding):     {X[0]}")
print("\nNotar: Q[0], K[0], V[0] son TRES vectores distintos que provienen del")
print("mismo embedding X[0]. Son las tres 'caras' de la misma palabra.")


# ---------------------------------------------------------------------------
# La matriz de scores con Q y K (no simetrica!)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 3: scores = Q @ K.T (ya NO es simetrica)")
print("=" * 70)

scores = Q @ K.T  # (T, T)
print(f"\nscores shape: {scores.shape}")
print(f"scores:\n{scores}")

# Verificar simetria
diff = (scores - scores.T).abs().max().item()
print(f"\nMax |scores[i,j] - scores[j,i]| = {diff:.3f}")
print("En escalon 1 esto era ~0 (simetrica). Ahora es != 0: la atencion es")
print("DIRECCIONAL. 'I' atendiendo a 'love' puede ser distinto de 'love'")
print("atendiendo a 'I'. El modelo gana flexibilidad.")


# ---------------------------------------------------------------------------
# DEMO VISCERAL: saturacion del softmax con d_k creciente
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 4: Demo del problema de saturacion (sin/con scaling)")
print("=" * 70)

print("""
Vamos a generar Q, K random con varias dimensiones d_k y ver que pasa con
el softmax SIN dividir por sqrt(d_k) vs CON dividir.

Teoria: si q, k tienen componentes iid de varianza 1, entonces q . k tiene
varianza d_k. Mientras d_k crece, el rango tipico de los dot products crece
como sqrt(d_k). Y el softmax, expuesto a numeros muy grandes, satura.
""")

torch.manual_seed(0)
T_demo = 6  # secuencia de 6 tokens para la demo

for d_k_demo in [4, 64, 512]:
    print(f"\n--- d_k = {d_k_demo} ---")
    Q_d = torch.randn(T_demo, d_k_demo)
    K_d = torch.randn(T_demo, d_k_demo)
    scores_d = Q_d @ K_d.T  # (T_demo, T_demo)

    print(f"  Estadisticas de scores SIN scaling:")
    print(f"    media = {scores_d.mean().item():.3f}")
    print(f"    std   = {scores_d.std().item():.3f}    (teoria: sqrt(d_k) = {math.sqrt(d_k_demo):.2f})")

    # Softmax sin scaling
    weights_no_scale = F.softmax(scores_d, dim=-1)
    max_w = weights_no_scale.max(dim=-1).values.mean().item()
    print(f"  Softmax SIN scaling, max peso por fila (promedio): {max_w:.4f}")
    print(f"    Fila 0: {weights_no_scale[0]}")

    # Softmax con scaling por sqrt(d_k)
    scores_scaled = scores_d / math.sqrt(d_k_demo)
    print(f"  Estadisticas de scores CON scaling (/sqrt(d_k)):")
    print(f"    std   = {scores_scaled.std().item():.3f}    (deberia ser ~1)")

    weights_scaled = F.softmax(scores_scaled, dim=-1)
    max_w_s = weights_scaled.max(dim=-1).values.mean().item()
    print(f"  Softmax CON scaling, max peso por fila (promedio): {max_w_s:.4f}")
    print(f"    Fila 0: {weights_scaled[0]}")

print("""
Observa el patron:
  - d_k = 4:   max peso sin scaling ~ 0.5-0.7 (manejable). El scaling ayuda
               poco, pero ya estabiliza a ~0.3-0.4.
  - d_k = 64:  max peso sin scaling ~ 0.85-0.95 (casi one-hot). El scaling
               lo trae de vuelta a ~0.3-0.4.
  - d_k = 512: max peso sin scaling ~ 0.99+ (totalmente saturado, gradiente
               muerto). Con scaling se mantiene a ~0.3-0.4.

CONCLUSION: el factor 1/sqrt(d_k) es lo que hace que el Transformer sea
entrenable a escalas grandes. Sin el, los modelos con d_model > 64 no
podrian aprender nada — el softmax se vuelve argmax y autograd muere.
""")


# ---------------------------------------------------------------------------
# Implementacion como nn.Module
# ---------------------------------------------------------------------------
print("=" * 70)
print("PASO 5: SelfAttention como nn.Module (la forma idiomatic de PyTorch)")
print("=" * 70)

print("""
nn.Module es la clase base de PyTorch para definir capas y modelos. La
forma estandar es:
  - __init__: declarar los parametros aprendibles (capas como nn.Linear).
  - forward: definir el calculo.
PyTorch automaticamente trackea los parametros y permite trainear con autograd.
""")


class SelfAttention(nn.Module):
    """
    Self-attention scaled dot-product de una sola cabeza.

    Recibe X de shape (batch, T, d_model) y devuelve:
      output: (batch, T, d_model)
      attention_weights: (batch, T, T) -- los alfa_ij
    """
    def __init__(self, d_model, d_k=None, d_v=None):
        super().__init__()
        d_k = d_k or d_model
        d_v = d_v or d_model
        self.d_k = d_k

        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_v, bias=False)

    def forward(self, x):
        # x: (batch, T, d_model)
        Q = self.W_Q(x)  # (batch, T, d_k)
        K = self.W_K(x)  # (batch, T, d_k)
        V = self.W_V(x)  # (batch, T, d_v)

        # scaled dot-product
        scores = Q @ K.transpose(-2, -1)             # (batch, T, T)
        scores = scores / math.sqrt(self.d_k)        # scaling
        weights = F.softmax(scores, dim=-1)          # (batch, T, T)
        output = weights @ V                         # (batch, T, d_v)

        return output, weights


# Aplicarlo a nuestra oracion
torch.manual_seed(42)
attention = SelfAttention(d_model=d_model)

# Necesitamos agregar dimension de batch (1 oracion en el batch)
X_batched = X.unsqueeze(0)  # (1, T, d_model)
output, weights = attention(X_batched)

print(f"\nInput X_batched.shape:  {X_batched.shape}")
print(f"Output shape:           {output.shape}")
print(f"Attention weights shape: {weights.shape}")

print(f"\nMatriz de attention weights (cada fila suma 1):")
print(weights[0])  # quitamos la dim de batch para imprimir
print(f"\nSuma por fila: {weights[0].sum(dim=-1)}")


# ---------------------------------------------------------------------------
# Comparacion lado a lado: degenerada (escalon 1) vs Q/K/V scaled
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 6: Comparacion lado a lado")
print("=" * 70)

# Atencion degenerada del escalon 1 (Q=K=V=X, sin scaling)
scores_degen = X @ X.T
weights_degen = F.softmax(scores_degen, dim=-1)

print("\nWeights ESCALON 1 (Q=K=V=X, sin scaling):")
print(weights_degen)
print(f"  Cada fila tiene un valor dominante (~1.0): atencion casi solo a si mismo.")

print("\nWeights ESCALON 2 (Q,K,V aprendibles, con scaling):")
print(weights[0])
print(f"  La distribucion es mas suave: el modelo PUEDE aprender a atender")
print(f"  a otros tokens, no esta forzado a auto-atenderse.")


# ---------------------------------------------------------------------------
# Aprendibilidad: vamos a entrenar 1 paso para ver que los gradientes fluyen
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 7: Verificar que los gradientes fluyen a W_Q, W_K, W_V")
print("=" * 70)

# Target ficticio: queremos que el output sea similar al input (objetivo
# trivial solo para mostrar que autograd funciona).
target = X_batched.clone()
output, _ = attention(X_batched)

loss = F.mse_loss(output, target)
print(f"\nLoss (MSE entre output y target ficticio): {loss.item():.4f}")

# Backprop
loss.backward()

# Verificar que cada matriz tiene gradiente
print(f"\nGradientes despues de loss.backward():")
print(f"  W_Q.weight.grad.shape: {attention.W_Q.weight.grad.shape}")
print(f"  W_Q.weight.grad.norm(): {attention.W_Q.weight.grad.norm().item():.4f}")
print(f"  W_K.weight.grad.norm(): {attention.W_K.weight.grad.norm().item():.4f}")
print(f"  W_V.weight.grad.norm(): {attention.W_V.weight.grad.norm().item():.4f}")
print("\nLos gradientes son distintos de cero -> autograd traceo todo el calculo")
print("(W_Q -> Q -> scores -> softmax -> weights -> output -> loss) y propago")
print("el error hacia atras hasta cada matriz aprendible. Optimizer.step()")
print("ahora podria ajustar las 3 matrices para reducir el loss.")


# ---------------------------------------------------------------------------
# Resumen y limitaciones
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("RESUMEN")
print("=" * 70)

print("""
Lo que se arreglo en este escalon:

  ESCALON 1 (degenerada)              ESCALON 2 (real)
  ----------------------              -----------------
  Q = K = V = X                       Q = X W_Q, K = X W_K, V = X W_V
  Sin parametros aprendibles          3 matrices aprendibles
  Matriz de scores SIMETRICA          Matriz NO simetrica (atencion direccional)
  Sin scaling                         Dividir por sqrt(d_k)
  Softmax saturado en d_k > 64        Softmax controlado para cualquier d_k
  Output ~ input                      Output realmente transformado

Lo que sigue (escalon 3): MULTI-HEAD ATTENTION.

Una sola cabeza de atencion produce UNA distribucion de pesos por token. Eso
es restrictivo: una palabra como "kicked" deberia poder atender a la vez al
sujeto ("Alexis"), a la accion ("kicked") y al objeto ("ball"). Multi-head
ejecuta h atenciones EN PARALELO, cada una en un subespacio distinto, y
concatena los resultados.

Hyperparametros tipicos en Transformer-base (Vaswani 2017):
  d_model = 512, h = 8, d_k = d_v = d_model/h = 64.

Eso lo construimos en el escalon 3.
""")

print("=" * 70)
print("FIN escalon 2.")
print("=" * 70)
