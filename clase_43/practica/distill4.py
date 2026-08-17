"""Seccion 4 corregida: SoundNet aplica L2 sobre las SALIDAS (probabilidades),
no sobre los logits. Esa distincion es la que decide el resultado."""
import numpy as np
from scipy.stats import spearmanr

def softmax(z, T=1.0):
    z = z / T; z = z - z.max(-1, keepdims=True)
    e = np.exp(z); return e / e.sum(-1, keepdims=True)

# ---------------------------------------------------------------- montaje
# Escenario de SoundNet: MUCHAS clases (1401 = 1000 ImageNet + 401 Places) y una
# distribucion del maestro muy concentrada (una imagen de Flickr activa unas pocas).
K, N, D = 400, 3000, 32
rng = np.random.default_rng(7)
X = rng.normal(0, 1, (N, D))
W_true = rng.normal(0, 1, (D, K)) / np.sqrt(D)
logits_true = np.clip(X @ W_true * 6.0, -60, 60)  # escala alta -> maestro concentrado
p_teacher = softmax(logits_true, 1.0)

print(f"Clases: {K}. Masa media en el top-5 del maestro: "
      f"{np.sort(p_teacher, 1)[:, -5:].sum(1).mean():.4f}")
print(f"Probabilidad mediana fuera del top-5: "
      f"{np.median(np.sort(p_teacher,1)[:, :-5]):.2e}\n")

def entrena(loss, T=1.0, epochs=600, lr=0.5):
    W = rng.normal(0, 0.01, (D, K))
    for _ in range(epochs):
        z = np.clip(X @ W, -60, 60)
        q = softmax(z, T)
        if loss == "kl":
            # d/dz KL(p||q) = (q - p)/T ; se multiplica por T^2 como indica Hinton
            g = (q - softmax(logits_true, T)) / T * (T ** 2)
        elif loss == "l2_probs":
            # d/dz ||q - p||^2 : hay que pasar por el jacobiano del softmax
            d = 2 * (q - p_teacher)
            g = q * (d - (d * q).sum(1, keepdims=True))
        elif loss == "l2_logits":
            g = 2 * (z - logits_true) / K
        W -= lr * (X.T @ g) / N
    return np.clip(X @ W, -60, 60)

def top1(z): return (z.argmax(1) == logits_true.argmax(1)).mean()
def rho(z):  return np.mean([spearmanr(a, b).statistic for a, b in zip(z[:200], logits_true[:200])])
def top5(z):
    t5 = np.argsort(logits_true, 1)[:, -5:]
    p5 = np.argsort(z, 1)[:, -5:]
    return np.mean([len(set(a) & set(b)) / 5 for a, b in zip(t5, p5)])

print(f"{'perdida':22s} {'top-1':>8s} {'top-5 solap.':>13s} {'corr. de rango':>16s}")
print("-" * 62)
filas = [("KL, T=1", "kl", 1.0),
         ("KL, T=2", "kl", 2.0),
         ("KL, T=4", "kl", 4.0),
         ("L2 sobre PROBS", "l2_probs", 1.0),
         ("L2 sobre logits", "l2_logits", 1.0)]
res = {}
for nombre, loss, T in filas:
    z = entrena(loss, T)
    res[nombre] = (100*top1(z), 100*top5(z), rho(z))
    print(f"{nombre:22s} {100*top1(z):7.2f}% {100*top5(z):12.2f}% {rho(z):16.4f}")

print(f"""
Lectura: L2 sobre las PROBABILIDADES es la configuracion de SoundNet, y es la que
colapsa. La razon es el jacobiano del softmax: cuando q_j ~= 0, su gradiente
respecto del logit tambien es ~0, asi que la perdida no puede corregir esas
clases aunque esten mal. Con {K} clases y una distribucion concentrada, eso es
casi todo el vector.

L2 sobre LOGITS es otra cosa: no pasa por el softmax, no se satura, y de hecho
funciona bien -- es exactamente el limite de temperatura alta del teorema de
Hinton. Las dos cosas se llaman 'L2' y no son la misma.
""")

# ---------------------------------------------------------------- magnitud del gradiente
print("=" * 62)
print("Por que se satura: magnitud del gradiente por clase")
print("=" * 62)
z0 = np.zeros((1, K))                       # alumno sin entrenar: q uniforme
q0 = softmax(z0, 1.0)
p = p_teacher[:1]
d = 2 * (q0 - p)
g_l2 = q0 * (d - (d * q0).sum(1, keepdims=True))
g_kl = (q0 - p)
orden = np.argsort(-p[0])
print(f"  {'rango':>6s} {'p_maestro':>12s} {'|grad KL|':>12s} {'|grad L2 probs|':>16s} {'razon':>10s}")
for r in [0, 1, 4, 20, 100, 399]:
    j = orden[r]
    print(f"  {r:6d} {p[0, j]:12.3e} {abs(g_kl[0, j]):12.3e} {abs(g_l2[0, j]):16.3e} "
          f"{abs(g_kl[0, j])/max(abs(g_l2[0, j]), 1e-30):10.1f}x")
print(f"\n  El gradiente de L2 sobre probabilidades es ~{K} veces mas chico en todas partes")
print("  (el factor q ~ 1/K del jacobiano), y encima decae donde el maestro tiene senal.")
