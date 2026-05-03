"""38_encoder_vs_decoder.py - Cap 38: encoder vs decoder.

Visualiza la diferencia entre mascara causal (decoder) y
atencion bidireccional (encoder) sobre la misma frase.
"""
import torch
import torch.nn.functional as F

torch.manual_seed(42)

T = 6  # longitud de secuencia de ejemplo
frase = ["To", "be", "or", "not", "to", "be"]

# === Mascara causal (decoder) ===
causal = torch.tril(torch.ones(T, T)).bool()
print("=== Mascara CAUSAL (decoder) ===")
print("Cada token solo puede atender tokens anteriores (incluyendose):\n")
header = f"{'':>6}" + "".join(f"{w:>6}" for w in frase)
print(header)
for i, wi in enumerate(frase):
    row = f"{wi:>6}" + "".join("  SI  " if causal[i, j] else "  NO  " for j in range(T))
    print(row)

# === Sin mascara (encoder) ===
print("\n=== Atencion BIDIRECCIONAL (encoder) ===")
print("Cada token puede atender a TODOS los tokens:\n")
print(header)
for i, wi in enumerate(frase):
    row = f"{wi:>6}" + "".join("  SI  " for _ in range(T))
    print(row)

# === Scores de atencion reales (un head aleatorio) ===
print("\n=== Scores de atencion encoder (un head) ===")
print("Muestra como 'be' (pos 1) atiende a todos:\n")
Q = torch.randn(T, 16)  # d_k = 16
K = torch.randn(T, 16)
scores = (Q @ K.T) / (16 ** 0.5)
attn_full = F.softmax(scores, dim=-1)
print("Pesos de atencion del token 'be' sobre todos los tokens:")
for j, wj in enumerate(frase):
    print(f"  be → {wj:>4}: {attn_full[1, j]:.3f}")

scores_causal = scores.masked_fill(~causal, float('-inf'))
attn_causal = F.softmax(scores_causal, dim=-1)
print("\nPesos de atencion del token 'not' (decoder, solo ve hasta 'not'):")
for j, wj in enumerate(frase):
    v = attn_causal[3, j]
    print(f"  not → {wj:>4}: {v:.3f}" + (" (bloqueado)" if v == 0 else ""))
