"""39_positional_embeddings.py - Cap 39: learned pos emb vs RoPE.

Muestra como se ven los embeddings de posicion aprendidos
y los compara conceptualmente con RoPE del cap 18.
"""
import torch
import torch.nn as nn
from _models import LearnedPositionalEmbedding

torch.manual_seed(42)

d_model = 128
max_seq_len = 128

emb = LearnedPositionalEmbedding(max_seq_len, d_model)

print("=== Learnable Positional Embeddings ===\n")
print(f"Shape del modulo: nn.Embedding({max_seq_len}, {d_model})")
print(f"Params: {max_seq_len * d_model:,} (uno por posicion × dimension)")
n_params = sum(p.numel() for p in emb.parameters())
print(f"Params totales: {n_params:,}\n")

# Mostrar similitud entre embeddings de posiciones cercanas vs lejanas
weights = emb.embedding.weight.detach()  # (128, 128)

def cos_sim(a, b):
    return (a @ b) / (a.norm() * b.norm())

print("Similaridad coseno entre embeddings de posicion (random init):")
print(f"  pos 0 vs pos 1:  {cos_sim(weights[0], weights[1]):.4f}")
print(f"  pos 0 vs pos 64: {cos_sim(weights[0], weights[64]):.4f}")
print(f"  pos 0 vs pos 127:{cos_sim(weights[0], weights[127]):.4f}")
print("\nNOTA: en random init estos valores son ruido — no tienen significado.")
print("El patron posicional (cercanas mas similares) solo emerge DESPUES del MLM training.")
print("Podemos re-correr este script post-training para ver la diferencia.")

print("\n=== Comparacion con RoPE (cap 18) ===")
print("""
RoPE (Rotary Position Embedding):
  - NO agrega nada a los embeddings de token
  - Rota Q y K en el espacio complejo segun la posicion
  - La similitud posicional emerge del producto punto rotado
  - Ventaja: extrapolacion a secuencias mas largas que el training

Learned Positional Embeddings (BERT):
  - SE SUMA un vector aprendido al embedding de token
  - No hay garantia de extrapolacion
  - Ventaja: mas simple, aprendible de forma directa
  - Limitacion: solo funciona hasta max_seq_len del training
""")

print("=== Forward pass ===")
x = torch.zeros(2, 10, d_model)  # secuencia de zeros
out = emb(x)
print(f"Input:  {x.shape}")
print(f"Output: {out.shape}")
print(f"La diferencia output - input = los embeddings de posicion:")
diff = out - x
for pos in [0, 3, 9]:
    print(f"  pos {pos}: norma = {diff[0, pos].norm():.4f}")
