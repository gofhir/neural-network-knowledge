"""La contribucion de First Order Motion Model, aislada y medida.

Pregunta: cuanto se gana representando el movimiento local con keypoint + jacobiano
(orden 1) en vez de solo el desplazamiento del keypoint (orden 0)?
"""
import numpy as np

rng = np.random.default_rng(0)
H = W = 64
ys, xs = np.mgrid[0:H, 0:W]
grid = np.stack([xs / W - 0.5, ys / H - 0.5], -1)      # (H,W,2) en [-0.5, 0.5]

# ---------------------------------------------------------------- campo verdadero
def campo_real(grid, K, centros, params):
    """Movimiento por partes: cada keypoint controla una region con su propia
    transformacion afin (rotacion + escala + traslacion). Es el modelo de objeto
    articulado que FOMM supone: partes rigidas que se mueven cada una a su modo."""
    d2 = ((grid[..., None, :] - centros[None, None, :, :]) ** 2).sum(-1)   # (H,W,K)
    w = np.exp(-d2 / (2 * 0.15 ** 2))
    w = w / w.sum(-1, keepdims=True)                                      # pesos suaves
    out = np.zeros_like(grid)
    for k in range(K):
        th, sc, tx, ty = params[k]
        A = sc * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        rel = grid - centros[k]
        loc = rel @ A.T + centros[k] + np.array([tx, ty])
        out += w[..., k:k+1] * loc
    return out

def aproxima(grid, centros, valores, jac=None):
    """Reconstruye el campo desde la representacion dispersa.
    orden 0: solo la posicion transformada del keypoint.
    orden 1: + jacobiano local (expansion de Taylor de primer orden)."""
    K = len(centros)
    d2 = ((grid[..., None, :] - centros[None, None, :, :]) ** 2).sum(-1)
    w = np.exp(-d2 / (2 * 0.15 ** 2)); w = w / w.sum(-1, keepdims=True)
    out = np.zeros_like(grid)
    for k in range(K):
        if jac is None:
            loc = np.broadcast_to(valores[k], grid.shape)                  # constante
        else:
            loc = valores[k] + (grid - centros[k]) @ jac[k].T              # afin local
        out = out + w[..., k:k+1] * loc
    return out

print("=" * 76)
print("EXP 1 — Error de reconstruccion del campo de movimiento: orden 0 contra orden 1")
print("=" * 76)
print(f"{'K':>4s} {'orden 0 (solo kp)':>20s} {'orden 1 (kp+jacobiano)':>24s} {'mejora':>10s}")
for K in [2, 4, 6, 10, 16, 24]:
    e0s, e1s = [], []
    for _ in range(30):
        centros = rng.uniform(-0.35, 0.35, (K, 2))
        params = np.stack([rng.normal(0, 0.35, K),        # rotacion
                           rng.normal(1.0, 0.12, K),      # escala
                           rng.normal(0, 0.05, K),        # tx
                           rng.normal(0, 0.05, K)], 1)    # ty
        real = campo_real(grid, K, centros, params)
        # valores y jacobianos EXACTOS en los keypoints (lo que la red estimaria)
        valores, jacs = [], []
        for k in range(K):
            th, sc, tx, ty = params[k]
            A = sc * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            valores.append(centros[k] + np.array([tx, ty]))
            jacs.append(A)
        valores = np.array(valores); jacs = np.array(jacs)
        a0 = aproxima(grid, centros, valores, jac=None)
        a1 = aproxima(grid, centros, valores, jac=jacs)
        e0s.append(np.abs(a0 - real).mean()); e1s.append(np.abs(a1 - real).mean())
    e0, e1 = np.mean(e0s), np.mean(e1s)
    print(f"{K:4d} {e0:20.5f} {e1:24.5f} {e0/e1:9.2f}x")

print("""
  El jacobiano no agrega capacidad expresiva 'en general': agrega exactamente la
  capacidad de representar ROTACION y ESCALA locales. Con pocos keypoints la
  diferencia es grande, porque cada uno tiene que cubrir una region amplia donde
  el movimiento NO es una traslacion pura. Al subir K la brecha se cierra: muchos
  keypoints aproximan la rotacion por partes, con traslaciones distintas.

  Es el argumento del paper: el jacobiano compra con parametros lo que si no habria
  que comprar con mas keypoints -- y mas keypoints es una representacion menos
  compacta y mas dificil de aprender sin supervision.
""")

print("=" * 76)
print("EXP 2 — Cuantos keypoints de orden 0 hacen falta para igualar a 10 de orden 1")
print("=" * 76)
K1 = 10
err_ref = []
for _ in range(30):
    centros = rng.uniform(-0.35, 0.35, (K1, 2))
    params = np.stack([rng.normal(0, 0.35, K1), rng.normal(1.0, 0.12, K1),
                       rng.normal(0, 0.05, K1), rng.normal(0, 0.05, K1)], 1)
    real = campo_real(grid, K1, centros, params)
    valores, jacs = [], []
    for k in range(K1):
        th, sc, tx, ty = params[k]
        A = sc * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        valores.append(centros[k] + np.array([tx, ty])); jacs.append(A)
    err_ref.append(np.abs(aproxima(grid, centros, np.array(valores),
                                   np.array(jacs)) - real).mean())
err_ref = np.mean(err_ref)
print(f"  referencia: {K1} keypoints de orden 1 -> error {err_ref:.5f}")
print(f"  parametros por keypoint: orden 1 = 2 + 4 = 6 ; orden 0 = 2\n")

for K0 in [10, 20, 40, 80, 160]:
    es = []
    for _ in range(20):
        c1 = rng.uniform(-0.35, 0.35, (K1, 2))
        p1 = np.stack([rng.normal(0, 0.35, K1), rng.normal(1.0, 0.12, K1),
                       rng.normal(0, 0.05, K1), rng.normal(0, 0.05, K1)], 1)
        real = campo_real(grid, K1, c1, p1)
        # muestreamos K0 keypoints de orden 0 sobre una grilla regular
        n = int(np.ceil(np.sqrt(K0)))
        gy, gx = np.mgrid[0:n, 0:n]
        c0 = np.stack([gx.ravel(), gy.ravel()], 1)[:K0] / max(n - 1, 1) - 0.5
        c0 = c0 * 0.8
        # el valor exacto del campo en cada uno
        idx = ((c0 + 0.5) * np.array([W - 1, H - 1])).astype(int).clip(0, H - 1)
        v0 = real[idx[:, 1], idx[:, 0]]
        es.append(np.abs(aproxima(grid, c0, v0, jac=None) - real).mean())
    print(f"  {K0:3d} keypoints de orden 0 ({2*K0:4d} params) -> error {np.mean(es):.5f}"
          f"   {'<= referencia' if np.mean(es) <= err_ref else ''}")
print(f"\n  ({K1} keypoints de orden 1 usan {6*K1} parametros)")
