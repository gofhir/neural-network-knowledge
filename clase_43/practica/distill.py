"""KL contra L2 en destilacion, y el teorema de Hinton sobre el limite de temperatura alta.
Pregunta que responde: por que SoundNet pierde 25 puntos usando L2 en vez de KL,
si Hinton demostro que a temperatura alta son equivalentes?"""
import numpy as np

rng = np.random.default_rng(0)

def softmax(z, T=1.0):
    z = z / T
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)

# ---------------------------------------------------------------- 1. dark knowledge
print("=" * 78)
print("1. Que informacion hay fuera del argmax (dark knowledge)")
print("=" * 78)
v = np.array([6.0, 2.0, 1.8, -1.0, -3.0])       # logits del maestro
clases = ["perro", "lobo", "zorro", "auto", "silla"]
for T in [1.0, 2.0, 5.0, 10.0]:
    p = softmax(v, T)
    ent = -(p * np.log(p)).sum()
    print(f"  T={T:5.1f}  " + "  ".join(f"{c}={x:.4f}" for c, x in zip(clases, p)) +
          f"   H={ent:.3f} nats")
print("""
  Con T=1 el maestro dice 'perro' y casi nada mas. Al subir T aparece que 'lobo' y
  'zorro' son parecidos entre si y muy distintos de 'auto': esa estructura relativa
  es la que el alumno aprende y las etiquetas duras no contienen.""")

# ---------------------------------------------------------------- 2. teorema de Hinton
print("=" * 78)
print("2. El teorema: a T alta, el gradiente de KL -> el gradiente de L2 sobre logits")
print("=" * 78)
print("""  Hinton (2015), ec. 4:  dC/dz_i  ~=  (z_i - v_i) / (N T^2)
  valido si los logits estan centrados en cero y T >> |logits|.""")

N = 8
v = rng.normal(0, 2.0, N); v -= v.mean()        # maestro, centrado
z = rng.normal(0, 2.0, N); z -= z.mean()        # alumno, centrado

def grad_kl(z, v, T):
    return (softmax(z, T) - softmax(v, T)) / T

def grad_l2_logits(z, v, T):
    return (z - v) / (N * T**2)

print(f"\n  {'T':>8s} {'coseno(grad_KL, grad_L2)':>26s} {'||grad_KL||/||grad_L2||':>26s}")
for T in [1, 2, 5, 10, 25, 50, 100, 1000]:
    g1, g2 = grad_kl(z, v, T), grad_l2_logits(z, v, T)
    cos = g1 @ g2 / (np.linalg.norm(g1) * np.linalg.norm(g2))
    print(f"  {T:8d} {cos:26.6f} {np.linalg.norm(g1)/np.linalg.norm(g2):26.4f}")
print("""
  El coseno -> 1: a T alta las dos perdidas empujan en la MISMA direccion.
  A T=1 no: ahi KL y L2 son objetivos distintos.""")

# ---------------------------------------------------------------- 3. lo que ve cada perdida
print("=" * 78)
print("3. Por que difieren a T=1: donde pone el peso cada perdida")
print("=" * 78)
v = np.array([8.0, 1.0, 0.5, -6.0, -9.0])
p = softmax(v, 1.0)
print("  logits del maestro :", " ".join(f"{x:7.2f}" for x in v))
print("  prob. del maestro  :", " ".join(f"{x:7.4f}" for x in p))
print("""
  L2 sobre logits trata a los cinco por igual: un error de 1 en el logit -9 pesa lo
  mismo que un error de 1 en el logit 8. Pero el logit -9 corresponde a p=0.000004:
  el maestro nunca fue entrenado para calibrarlo, es ruido.

  KL pondera por p: el gradiente respecto de la clase j es proporcional a (q_j - p_j),
  asi que las clases de probabilidad despreciable casi no aportan.""")
w_l2 = np.ones(5) / 5
w_kl = p / p.sum()
print("  peso relativo L2 :", " ".join(f"{x:7.4f}" for x in w_l2))
print("  peso relativo KL :", " ".join(f"{x:7.4f}" for x in w_kl))
print(f"""
  Hinton lo dice explicitamente: 'a temperaturas bajas, la destilacion presta mucha
  menos atencion a los logits mucho mas negativos que el promedio. Esto es
  potencialmente ventajoso porque esos logits estan casi completamente
  no restringidos por la funcion de costo con la que se entreno el modelo grande,
  asi que podrian ser muy ruidosos.'
""")

# ---------------------------------------------------------------- 4. simulacion
print("=" * 78)
print("4. Simulacion: alumno entrenado con KL contra L2, con un maestro ruidoso")
print("=" * 78)
print("""  Montaje: el maestro acierta la clase pero sus logits de cola son ruido puro
  (es lo que pasa con un clasificador de ImageNet aplicado a frames de video de
  Flickr: las 1000 clases de la cola no significan nada). El alumno tiene que
  aprender el ranking correcto.""")

K, Ntr = 12, 4000
rng = np.random.default_rng(3)
# estructura verdadera: 3 grupos de 4 clases; dentro de un grupo las clases se parecen
grupo = np.arange(K) // 4
X = rng.normal(0, 1, (Ntr, 16))
W_true = rng.normal(0, 1, (16, K))
logits_true = X @ W_true
# maestro: logits verdaderos + ruido FUERTE en la cola (clases de baja probabilidad)
p_true = softmax(logits_true, 1.0)
ruido = rng.normal(0, 4.0, (Ntr, K)) * (p_true < 0.05)
logits_teacher = logits_true + ruido

def entrena(loss, T=1.0, epochs=400, lr=0.05):
    W = rng.normal(0, 0.1, (16, K))
    for _ in range(epochs):
        z = X @ W
        if loss == "kl":
            g = (softmax(z, T) - softmax(logits_teacher, T)) / T
        else:  # l2 sobre logits
            g = 2 * (z - logits_teacher) / K
        W -= lr * (X.T @ g) / Ntr
    return X @ W

def top1_acc(z):
    return (z.argmax(1) == logits_true.argmax(1)).mean()

def rank_corr(z):
    # correlacion de Spearman promedio entre el ranking del alumno y el VERDADERO
    from scipy.stats import spearmanr
    return np.mean([spearmanr(a, b).statistic for a, b in zip(z[:300], logits_true[:300])])

for nombre, loss, T in [("KL, T=1", "kl", 1.0), ("KL, T=4", "kl", 4.0), ("L2 sobre logits", "l2", 1.0)]:
    z = entrena(loss, T)
    print(f"  {nombre:18s} top-1 contra la verdad = {100*top1_acc(z):5.2f}%   "
          f"correlacion de rango = {rank_corr(z):.4f}")
print("""
  El alumno de L2 gasta capacidad ajustando el ruido de la cola; el de KL lo ignora
  y se concentra en donde el maestro tiene senal. Es la explicacion de por que
  SoundNet mide 47,8% con L2 y 72,9% con KL sobre ESC-50.""")
