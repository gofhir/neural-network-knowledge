import numpy as np
from scipy.stats import chi2
from scipy.optimize import linear_sum_assignment

print("=== 1. Umbral de Mahalanobis ===")
for df in [2, 4, 8]:
    print(f"  chi2_0.95, df={df} = {chi2.ppf(0.95, df):.4f}")

print("\n=== 2. Aritmetica de MOTA (MOT16 test, tabla DeepSORT) ===")
# SORT: FP=8698 FN=63245 IDSW=1423 MOTA=59.8
# DeepSORT: FP=12852 FN=56668 IDSW=781 MOTA=61.4
for name, fp, fn, ids, mota in [("SORT", 8698, 63245, 1423, 0.598),
                                 ("DeepSORT", 12852, 56668, 781, 0.614)]:
    err = fp + fn + ids
    G = err / (1 - mota)
    print(f"  {name}: errores={err}, |gtDet| implicito = {G:.0f}, IDSW aporta {100*ids/err:.2f}% del numerador")

G = 182326  # gtDet de MOT16 test
print(f"\n  Usando |gtDet| = {G}:")
mota_sort = 1 - (8698 + 63245 + 1423) / G
mota_ds = 1 - (12852 + 56668 + 781) / G
print(f"    MOTA SORT     = {100*mota_sort:.2f}  (reportado 59.8)")
print(f"    MOTA DeepSORT = {100*mota_ds:.2f}  (reportado 61.4)")

# Contrafactuales
c1 = 1 - (8698 + 63245 + 781) / G   # SORT con los IDSW de DeepSORT
c2 = 1 - (8698 + 56668 + 781) / G   # DeepSORT sin sus FP extra
print(f"\n  Contrafactual A: SORT pero con 781 IDSW  -> MOTA {100*c1:.2f} (+{100*(c1-mota_sort):.2f} pts)")
print(f"     => arreglar el 45% de los ID switches vale {100*(c1-mota_sort):.2f} puntos de MOTA")
print(f"  Contrafactual B: DeepSORT con los FP de SORT -> MOTA {100*c2:.2f} (+{100*(c2-mota_ds):.2f} pts)")
print(f"     => los 4154 FP extra de A_max=30 cuestan {100*(c2-mota_ds):.2f} puntos")

print("\n=== 3. Mahalanobis premia la incertidumbre ===")
d = np.array([10.0, 0.0])   # detección a 10 px del centro predicho
for sigma in [1.0, 2.0, 5.0, 10.0]:
    S = np.eye(2) * sigma**2
    m = d @ np.linalg.inv(S) @ d
    print(f"  sigma={sigma:5.1f} px -> Mahalanobis^2 = {m:8.2f}")
print("  La MISMA detección, a la MISMA distancia euclidea, se vuelve 100x mas 'cercana'")
print("  cuando la incertidumbre crece de 1 a 10 px.")

print("\n=== 4. Crecimiento de la covarianza durante una oclusion ===")
dt = 1.0
F = np.array([[1, dt], [0, 1]])
Q = np.eye(2) * 1.0
P = np.eye(2) * 1.0
print("  frame | var(pos) | sigma(pos)")
for t in range(1, 31):
    P = F @ P @ F.T + Q
    if t in (1, 2, 5, 10, 20, 30):
        print(f"   {t:4d} | {P[0,0]:8.1f} | {np.sqrt(P[0,0]):8.2f}")

print("\n=== 5. El greedy no es optimo ===")
C = np.array([[1.0, 2.0], [3.0, 100.0]])
r, c = linear_sum_assignment(C)
print(f"  C = {C.tolist()}")
print(f"  Hungaro: filas {r.tolist()} -> cols {c.tolist()}, costo {C[r,c].sum():.0f}")
# greedy
Cg = C.copy(); tot = 0; used_r=set(); used_c=set()
for _ in range(2):
    idx = np.unravel_index(np.argmin(np.where(np.isfinite(Cg), Cg, np.inf)), Cg.shape)
    tot += Cg[idx]; Cg[idx[0],:]=np.inf; Cg[:,idx[1]]=np.inf
print(f"  Greedy: costo {tot:.0f}")
