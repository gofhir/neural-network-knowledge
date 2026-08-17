"""SORT minimo desde cero + experimentos sobre una escena sintetica controlada."""
import numpy as np
from scipy.optimize import linear_sum_assignment

rng = np.random.default_rng(0)

# ---------------------------------------------------------------- Kalman
class KF:
    """Velocidad constante sobre (u, v, s, r); estado de 7-D como en SORT."""
    def __init__(self, box):
        self.x = np.zeros(7)
        self.x[:4] = box
        self.P = np.eye(7)
        self.P[4:, 4:] *= 1000.0     # velocidades: incertidumbre alta
        self.P *= 10.0
        self.F = np.eye(7)
        for i in range(3):
            self.F[i, 4 + i] = 1.0
        self.H = np.zeros((4, 7)); self.H[:4, :4] = np.eye(4)
        self.Q = np.eye(7) * 0.01; self.Q[4:, 4:] *= 0.01
        self.R = np.eye(4) * 1.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4].copy()

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    def S(self):
        return self.H @ self.P @ self.H.T + self.R


def xysr_to_box(z):
    u, v, s, r = z
    s = max(s, 1.0); r = max(r, 1e-3)
    w = np.sqrt(s * r); h = s / max(w, 1e-6)
    return np.array([u - w / 2, v - h / 2, u + w / 2, v + h / 2])


def box_to_xysr(b):
    w = b[2] - b[0]; h = b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w * h, w / max(h, 1e-6)])


def iou(a, b):
    xx1 = np.maximum(a[0], b[0]); yy1 = np.maximum(a[1], b[1])
    xx2 = np.minimum(a[2], b[2]); yy2 = np.minimum(a[3], b[3])
    w = max(0.0, xx2 - xx1); h = max(0.0, yy2 - yy1)
    inter = w * h
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ---------------------------------------------------------------- tracker
class Track:
    _next = 0
    def __init__(self, box, use_kf=True):
        Track._next += 1
        self.id = Track._next
        self.use_kf = use_kf
        self.kf = KF(box_to_xysr(box))
        self.box = box.copy()
        self.time_since_update = 0
        self.hits = 1

    def predict(self):
        if self.use_kf:
            self.box = xysr_to_box(self.kf.predict())
        self.time_since_update += 1
        return self.box

    def update(self, box):
        self.kf.update(box_to_xysr(box))
        self.box = xysr_to_box(self.kf.x[:4]) if self.use_kf else box.copy()
        self.time_since_update = 0
        self.hits += 1


def run_sort(frames, iou_min=0.3, t_lost=1, use_kf=True, use_hungarian=True):
    """frames: lista de listas de cajas. Devuelve lista de dicts {track_id: box}."""
    Track._next = 0
    tracks, out = [], []
    for dets in frames:
        for t in tracks:
            t.predict()
        matched, un_d = {}, list(range(len(dets)))
        if tracks and dets:
            C = np.zeros((len(tracks), len(dets)))
            for i, t in enumerate(tracks):
                for j, d in enumerate(dets):
                    C[i, j] = -iou(t.box, d)
            if use_hungarian:
                r, c = linear_sum_assignment(C)
                pairs = list(zip(r, c))
            else:  # codicioso
                pairs, ur, uc = [], set(), set()
                order = np.dstack(np.unravel_index(np.argsort(C, axis=None), C.shape))[0]
                for i, j in order:
                    if i not in ur and j not in uc:
                        pairs.append((i, j)); ur.add(i); uc.add(j)
            for i, j in pairs:
                if -C[i, j] >= iou_min:
                    matched[i] = j
        for i, j in matched.items():
            tracks[i].update(dets[j])
            un_d.remove(j)
        for j in un_d:
            tracks.append(Track(dets[j], use_kf=use_kf))
        tracks = [t for t in tracks if t.time_since_update <= t_lost]
        out.append({t.id: t.box.copy() for t in tracks if t.time_since_update == 0})
    return out


# ---------------------------------------------------------------- escena
def scene(n_frames=60, occlude=None, noise=1.0, seed=0):
    """Dos objetos que se cruzan. occlude=(a,b): frames en que el obj 0 no se detecta."""
    r = np.random.default_rng(seed)
    gt, dets = [], []
    for t in range(n_frames):
        g, d = {}, []
        # objeto 0: izquierda -> derecha
        x0 = 10 + 4.0 * t
        # objeto 1: derecha -> izquierda (se cruzan en t=30)
        x1 = 250 - 4.0 * t
        for oid, x in [(0, x0), (1, x1)]:
            b = np.array([x, 100.0, x + 30.0, 180.0])
            g[oid] = b
            if occlude and oid == 0 and occlude[0] <= t < occlude[1]:
                continue
            d.append(b + r.normal(0, noise, 4))
        gt.append(g); dets.append(d)
    return gt, dets


def evaluate(gt, out, thr=0.5):
    """MOTA, ID switches, IDF1 y HOTA sobre la escena sintetica."""
    from collections import defaultdict
    tp = fp = fn = ids = 0
    last = {}
    matches = []          # (frame, gtID, prID)
    for t, (g, o) in enumerate(zip(gt, out)):
        gids, pids = list(g), list(o)
        if gids and pids:
            C = np.zeros((len(gids), len(pids)))
            for i, gi in enumerate(gids):
                for j, pj in enumerate(pids):
                    C[i, j] = -iou(g[gi], o[pj])
            r, c = linear_sum_assignment(C)
            used_g, used_p = set(), set()
            for i, j in zip(r, c):
                if -C[i, j] >= thr:
                    gi, pj = gids[i], pids[j]
                    tp += 1; used_g.add(gi); used_p.add(pj)
                    matches.append((t, gi, pj))
                    if gi in last and last[gi] != pj:
                        ids += 1
                    last[gi] = pj
            fn += len(gids) - len(used_g); fp += len(pids) - len(used_p)
        else:
            fn += len(gids); fp += len(pids)
    n_gt = sum(len(g) for g in gt)
    mota = 1 - (fn + fp + ids) / n_gt

    # IDF1 (matching global gtTraj <-> prTraj)
    cnt = defaultdict(int)
    gt_len = defaultdict(int); pr_len = defaultdict(int)
    for _, gi, pj in matches:
        cnt[(gi, pj)] += 1
    for g in gt:
        for gi in g: gt_len[gi] += 1
    for o in out:
        for pj in o: pr_len[pj] += 1
    G, P = sorted(gt_len), sorted(pr_len)
    if G and P:
        M = np.zeros((len(G), len(P)))
        for i, gi in enumerate(G):
            for j, pj in enumerate(P):
                M[i, j] = -cnt.get((gi, pj), 0)
        r, c = linear_sum_assignment(M)
        idtp = int(-M[r, c].sum())
    else:
        idtp = 0
    idfn = sum(gt_len.values()) - idtp
    idfp = sum(pr_len.values()) - idtp
    idf1 = idtp / (idtp + 0.5*idfn + 0.5*idfp) if idtp else 0.0

    # HOTA (alpha fijo = thr)
    if tp:
        tpa_map = defaultdict(int)
        for _, gi, pj in matches: tpa_map[(gi, pj)] += 1
        gt_tp = defaultdict(int); pr_tp = defaultdict(int)
        for _, gi, pj in matches:
            gt_tp[gi] += 1; pr_tp[pj] += 1
        tot = 0.0
        for _, gi, pj in matches:
            tpa = tpa_map[(gi, pj)]
            fna = gt_len[gi] - tpa
            fpa = pr_len[pj] - tpa
            tot += tpa / (tpa + fna + fpa)
        deta = tp / (tp + fn + fp); assa = tot / tp
        hota = np.sqrt(deta * assa)
    else:
        deta = assa = hota = 0.0
    return dict(MOTA=100*mota, IDF1=100*idf1, HOTA=100*hota, DetA=100*deta,
                AssA=100*assa, IDs=ids, FP=fp, FN=fn, n_tracks=len(pr_len))


print("=" * 78)
print("EXP 1 — Escena limpia (sin oclusion), dos objetos que se cruzan")
print("=" * 78)
gt, dets = scene()
for label, kw in [("SORT completo", {}),
                  ("sin Kalman (caja anterior)", dict(use_kf=False)),
                  ("asociacion codiciosa", dict(use_hungarian=False))]:
    m = evaluate(gt, run_sort(dets, **kw))
    print(f"  {label:28s} MOTA {m['MOTA']:6.2f}  IDF1 {m['IDF1']:6.2f}  HOTA {m['HOTA']:6.2f}  IDs {m['IDs']:3d}  tracks {m['n_tracks']}")

print()
print("=" * 78)
print("EXP 2 — El objeto 0 se ocluye; efecto de T_lost")
print("=" * 78)
for occ in [(28, 31), (28, 36), (28, 46)]:
    gt, dets = scene(occlude=occ)
    dur = occ[1] - occ[0]
    print(f"\n  Oclusion de {dur} frames:")
    for t_lost in [1, 3, 10, 30]:
        m = evaluate(gt, run_sort(dets, t_lost=t_lost))
        print(f"    T_lost={t_lost:2d}  MOTA {m['MOTA']:6.2f}  IDF1 {m['IDF1']:6.2f}  "
              f"HOTA {m['HOTA']:6.2f}  IDs {m['IDs']:2d}  FP {m['FP']:3d}  tracks {m['n_tracks']}")

print()
print("=" * 78)
print("EXP 3 — Ruido de deteccion creciente (sin oclusion)")
print("=" * 78)
for noise in [0.5, 2.0, 5.0, 10.0, 20.0]:
    gt, dets = scene(noise=noise, seed=1)
    m = evaluate(gt, run_sort(dets))
    print(f"  sigma={noise:5.1f} px  MOTA {m['MOTA']:6.2f}  IDF1 {m['IDF1']:6.2f}  "
          f"HOTA {m['HOTA']:6.2f}  IDs {m['IDs']:2d}  FN {m['FN']:3d}  tracks {m['n_tracks']}")
