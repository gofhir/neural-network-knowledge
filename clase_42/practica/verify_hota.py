"""Reconstruir el ejemplo de la Fig. 1 de HOTA y calcular MOTA / IDF1 / HOTA."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

T = 100  # el gt es UNA trayectoria de 100 detecciones, frames 0..99

def make(segments):
    """segments: lista de (inicio, fin) -> dict frame -> prID"""
    pred = {}
    for pid, (a, b) in enumerate(segments):
        for t in range(a, b):
            pred[t] = pid
    return pred

trackers = {
    "A (1 track de 50)":  make([(0, 50)]),
    "B (2 tracks de 35)": make([(0, 35), (35, 70)]),
    "C (4 tracks de 25)": make([(0, 25), (25, 50), (50, 75), (75, 100)]),
}

def mota(pred):
    tp = len(pred); fn = T - tp; fp = 0
    ids = 0
    prev = None
    for t in sorted(pred):
        if prev is not None and pred[t] != prev:
            ids += 1
        prev = pred[t]
    return 1 - (fn + fp + ids) / T, ids

def idf1(pred):
    # gt: una sola trayectoria (gtID=0). pred: varias prIDs.
    pr_tracks = defaultdict(set)
    for t, p in pred.items():
        pr_tracks[p].add(t)
    # matching biyectivo gtTraj <-> prTraj minimizando IDFN+IDFP
    best = None
    for p, frames in pr_tracks.items():
        idtp = len(frames)
        idfn = T - idtp
        idfp = len(pred) - idtp
        score = idtp / (idtp + 0.5 * idfn + 0.5 * idfp)
        if best is None or score > best[0]:
            best = (score, idtp, idfn, idfp)
    return best

def hota(pred):
    tp = len(pred); fn = T - tp; fp = 0
    tot = 0.0
    for t, p in pred.items():
        tpa = sum(1 for tt, pp in pred.items() if pp == p)   # mismo gtID (unico) y mismo prID
        fna = T - tpa                                        # mismo gtID, distinto prID o perdidos
        fpa = 0                                              # mismo prID, distinto gtID: no hay
        tot += tpa / (tpa + fna + fpa)
    deta = tp / (tp + fn + fp)
    assa = tot / tp
    return np.sqrt(deta * assa), deta, assa

print(f"{'Tracker':22s} {'DetA':>6s} {'AssA':>6s} {'MOTA':>7s} {'IDF1':>7s} {'HOTA':>7s}  {'IDSW':>4s}")
print("-" * 68)
for name, pred in trackers.items():
    m, ids = mota(pred)
    f1, idtp, idfn, idfp = idf1(pred)
    h, deta, assa = hota(pred)
    print(f"{name:22s} {100*deta:6.1f} {100*assa:6.1f} {100*m:7.1f} {100*f1:7.1f} {100*h:7.1f}  {ids:4d}")

print("\nPaper (Fig. 1):")
print(f"{'A':22s} {50.0:6.1f} {50.0:6.1f} {50.0:7.1f} {67.0:7.1f} {50.0:7.1f}")
print(f"{'B':22s} {70.0:6.1f} {35.0:6.1f} {69.0:7.1f} {52.0:7.1f} {50.0:7.1f}")
print(f"{'C':22s} {100.0:6.1f} {25.0:6.1f} {97.0:7.1f} {25.0:7.1f} {50.0:7.1f}")

print("\nDetalle IDF1 de B:")
f1, idtp, idfn, idfp = idf1(trackers["B (2 tracks de 35)"])
print(f"  IDTP={idtp}  IDFN={idfn}  IDFP={idfp}  ->  IDF1 = {100*f1:.1f}%")
