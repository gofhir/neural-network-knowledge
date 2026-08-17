"""Experimentos 4 y 5: hungaro vs codicioso con N objetos, y la patologia de Mahalanobis."""
import numpy as np
from scipy.optimize import linear_sum_assignment
import sys
sys.path.insert(0, '.')
from sort_lab import run_sort, evaluate, iou

rng = np.random.default_rng(7)

# ---------------------------------------------------------- EXP 4
def crowd(n_obj=12, n_frames=50, seed=0, noise=1.0):
    """Multitud densa: objetos que se cruzan en un area chica."""
    r = np.random.default_rng(seed)
    pos = r.uniform(0, 200, (n_obj, 2))
    vel = r.normal(0, 3.0, (n_obj, 2))
    gt, dets = [], []
    for t in range(n_frames):
        g, d = {}, []
        for i in range(n_obj):
            p = pos[i] + vel[i] * t
            b = np.array([p[0], p[1], p[0] + 25.0, p[1] + 50.0])
            g[i] = b
            d.append(b + r.normal(0, noise, 4))
        gt.append(g); dets.append(d)
    return gt, dets

print("=" * 82)
print("EXP 4 — Hungaro contra codicioso en escenas densas (12 objetos, 20 semillas)")
print("=" * 82)
diff = 0
agg = {True: [], False: []}
for seed in range(20):
    gt, dets = crowd(seed=seed)
    for use_h in (True, False):
        m = evaluate(gt, run_sort(dets, use_hungarian=use_h))
        agg[use_h].append((m['MOTA'], m['IDF1'], m['HOTA'], m['IDs']))
    if agg[True][-1] != agg[False][-1]:
        diff += 1
for use_h in (True, False):
    a = np.array(agg[use_h])
    label = "hungaro " if use_h else "codicioso"
    print(f"  {label}: MOTA {a[:,0].mean():6.2f}  IDF1 {a[:,1].mean():6.2f}  "
          f"HOTA {a[:,2].mean():6.2f}  IDs {a[:,3].mean():5.2f}")
print(f"  semillas donde difieren: {diff}/20")

# más denso y con más ruido
print("\n  Con 25 objetos y sigma=6 px:")
agg = {True: [], False: []}
diff = 0
for seed in range(20):
    gt, dets = crowd(n_obj=25, seed=100 + seed, noise=6.0)
    for use_h in (True, False):
        m = evaluate(gt, run_sort(dets, use_hungarian=use_h))
        agg[use_h].append((m['MOTA'], m['IDF1'], m['HOTA'], m['IDs']))
    if agg[True][-1] != agg[False][-1]:
        diff += 1
for use_h in (True, False):
    a = np.array(agg[use_h])
    label = "hungaro " if use_h else "codicioso"
    print(f"  {label}: MOTA {a[:,0].mean():6.2f}  IDF1 {a[:,1].mean():6.2f}  "
          f"HOTA {a[:,2].mean():6.2f}  IDs {a[:,3].mean():5.2f}")
print(f"  semillas donde difieren: {diff}/20")

# ---------------------------------------------------------- EXP 5
print()
print("=" * 82)
print("EXP 5 — La patologia de Mahalanobis: la trayectoria mas incierta gana")
print("=" * 82)
print("""
  Dos trayectorias compiten por UNA deteccion.
  - Track A: visto hace 1 frame  -> covarianza pequena, prediccion a  5 px de la deteccion.
  - Track B: visto hace 25 frames -> covarianza inflada, prediccion a 40 px de la deteccion.
  La respuesta correcta es A: esta 8x mas cerca y su prediccion es confiable.
""")
F = np.array([[1.0, 1.0], [0.0, 1.0]])
Q = np.eye(2) * 1.0

def cov_after(k, p0=1.0):
    P = np.eye(2) * p0
    for _ in range(k):
        P = F @ P @ F.T + Q
    return P[0, 0]

for age_b in (5, 10, 25, 40):
    var_a = cov_after(1); var_b = cov_after(age_b)
    d_a, d_b = 5.0, 40.0
    m_a = d_a**2 / var_a
    m_b = d_b**2 / var_b
    winner = "A (correcto)" if m_a < m_b else "B  <-- ERROR"
    print(f"  edad de B = {age_b:2d} frames | sigma_A={np.sqrt(var_a):5.2f} sigma_B={np.sqrt(var_b):6.2f} | "
          f"Maha_A={m_a:7.2f}  Maha_B={m_b:7.2f} | gana {winner}")

print("""
  La cascada de matching de DeepSORT resuelve primero las trayectorias de menor edad,
  con lo que A toma la deteccion antes de que B pueda competir por ella.
""")

# umbral chi2 y cuantas asociaciones deja pasar
from scipy.stats import chi2
print("=" * 82)
print("EXP 6 — Que tan permisiva es la compuerta chi2 cuando la covarianza crece")
print("=" * 82)
t = chi2.ppf(0.95, 4)
print(f"  umbral t = chi2(0.95, 4) = {t:.4f}")
print("  radio de la region admisible, en pixeles, para covarianza isotropica sigma^2 I:")
for age in (0, 1, 5, 10, 20, 30):
    var = cov_after(age) if age else 1.0
    radio = np.sqrt(t * var)
    print(f"    tras {age:2d} frames sin deteccion: sigma={np.sqrt(var):6.2f} px -> radio admisible {radio:7.2f} px")
print("""
  Tras 30 frames (el A_max de DeepSORT), la compuerta admite cualquier deteccion
  dentro de ~300 px: en una imagen de 1920x1080 eso ya no filtra casi nada.
""")
