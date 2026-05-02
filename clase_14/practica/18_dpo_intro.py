"""18_dpo_intro.py - Cap 26: Bradley-Terry numericamente.

Demo: dado un par (y_w, y_l) con rewards r_w, r_l, computar P(y_w succeq y_l).
Sin red neuronal — solo numpy/math. Construye intuicion para la loss DPO del cap 27.
"""
import math

print("=== Bradley-Terry: P(y_w succeq y_l) = sigma(r_w - r_l) ===\n")
sigmoid = lambda z: 1 / (1 + math.exp(-z))

cases = [
    ("preferencia clara",   2.0,  -1.0),
    ("preferencia tibia",   0.5,   0.0),
    ("empate",              1.0,   1.0),
    ("opuesto",            -2.0,   1.0),
]
for label, rw, rl in cases:
    p = sigmoid(rw - rl)
    print(f"{label:<22} r_w={rw:+.1f}  r_l={rl:+.1f}  P(y_w>y_l)={p:.3f}")

print("\n=== Log-likelihood de un dataset de 3 preferencias ===")
prefs = [(2.0, -1.0), (0.5, 0.0), (-2.0, 1.0)]
ll = sum(math.log(sigmoid(rw - rl)) for rw, rl in prefs)
print(f"sum log P(y_w>y_l) = {ll:.4f}")
print("\nMaximizar esta log-likelihood = aprender los rewards.")
print("DPO va mas lejos: parametriza r implicitamente via la policy y ref model.")
