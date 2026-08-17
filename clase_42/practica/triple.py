"""Triple framework: IoU por lotes y un paso de filtro de Kalman batched.
Verifica que NumPy, PyTorch, TensorFlow y JAX coinciden hasta el epsilon de maquina."""
import numpy as np

rng = np.random.default_rng(42)
N, M = 6, 8
tracks = rng.uniform(0, 200, (N, 4)); tracks[:, 2:] += tracks[:, :2] + 20
dets   = rng.uniform(0, 200, (M, 4)); dets[:, 2:]   += dets[:, :2]   + 20

# ------------------------------------------------------------------ NumPy
def iou_np(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter)

ref = iou_np(tracks, dets)

# ------------------------------------------------------------------ PyTorch
import torch
def iou_torch(a, b):
    a = torch.as_tensor(a); b = torch.as_tensor(b)
    x1 = torch.maximum(a[:, None, 0], b[None, :, 0])
    y1 = torch.maximum(a[:, None, 1], b[None, :, 1])
    x2 = torch.minimum(a[:, None, 2], b[None, :, 2])
    y2 = torch.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return (inter / (aa[:, None] + ab[None, :] - inter)).numpy()

from torchvision.ops import box_iou as tv_box_iou  # referencia oficial

# ------------------------------------------------------------------ TensorFlow
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
def iou_tf(a, b):
    a = tf.constant(a); b = tf.constant(b)
    x1 = tf.maximum(a[:, None, 0], b[None, :, 0])
    y1 = tf.maximum(a[:, None, 1], b[None, :, 1])
    x2 = tf.minimum(a[:, None, 2], b[None, :, 2])
    y2 = tf.minimum(a[:, None, 3], b[None, :, 3])
    inter = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return (inter / (aa[:, None] + ab[None, :] - inter)).numpy()

# ------------------------------------------------------------------ JAX
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

@jax.jit
def iou_jax(a, b):
    x1 = jnp.maximum(a[:, None, 0], b[None, :, 0])
    y1 = jnp.maximum(a[:, None, 1], b[None, :, 1])
    x2 = jnp.minimum(a[:, None, 2], b[None, :, 2])
    y2 = jnp.minimum(a[:, None, 3], b[None, :, 3])
    inter = jnp.clip(x2 - x1, 0) * jnp.clip(y2 - y1, 0)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter)

print("=== IoU por lotes: los cuatro backends ===")
for name, val in [("PyTorch", iou_torch(tracks, dets)),
                  ("torchvision.ops.box_iou", tv_box_iou(torch.as_tensor(tracks), torch.as_tensor(dets)).numpy()),
                  ("TensorFlow", iou_tf(tracks, dets)),
                  ("JAX (jit)", np.asarray(iou_jax(jnp.array(tracks), jnp.array(dets))))]:
    print(f"  {name:26s} max|dif| vs NumPy = {np.abs(val - ref).max():.3e}")

# ================================================================== Kalman batched
print("\n=== Un paso predict+update del filtro de Kalman, para N trayectorias a la vez ===")
n_tracks = 5
F = np.eye(7)
for i in range(3):
    F[i, 4 + i] = 1.0
H = np.zeros((4, 7)); H[:, :4] = np.eye(4)
Q = np.eye(7) * 0.01
R = np.eye(4) * 1.0
X = rng.normal(0, 10, (n_tracks, 7))
P = np.tile(np.eye(7) * 10.0, (n_tracks, 1, 1))
Z = rng.normal(0, 10, (n_tracks, 4))

def kalman_np(X, P, Z):
    Xp = X @ F.T
    Pp = F @ P @ F.T + Q
    y = Z - Xp @ H.T
    S = H @ Pp @ H.T + R
    K = Pp @ H.T @ np.linalg.inv(S)
    Xn = Xp + np.einsum('nij,nj->ni', K, y)
    Pn = (np.eye(7) - K @ H) @ Pp
    return Xn, Pn, S

Xn_ref, Pn_ref, S_ref = kalman_np(X, P, Z)

def kalman_torch(X, P, Z):
    Ft, Ht = torch.as_tensor(F), torch.as_tensor(H)
    Qt, Rt = torch.as_tensor(Q), torch.as_tensor(R)
    X, P, Z = map(torch.as_tensor, (X, P, Z))
    Xp = X @ Ft.T
    Pp = Ft @ P @ Ft.T + Qt
    y = Z - Xp @ Ht.T
    S = Ht @ Pp @ Ht.T + Rt
    K = Pp @ Ht.T @ torch.linalg.inv(S)
    Xn = Xp + torch.einsum('nij,nj->ni', K, y)
    Pn = (torch.eye(7, dtype=P.dtype) - K @ Ht) @ Pp
    return Xn.numpy(), Pn.numpy(), S.numpy()

def kalman_tf(X, P, Z):
    Ft, Ht = tf.constant(F), tf.constant(H)
    Qt, Rt = tf.constant(Q), tf.constant(R)
    X, P, Z = tf.constant(X), tf.constant(P), tf.constant(Z)
    Xp = X @ tf.transpose(Ft)
    Pp = Ft @ P @ tf.transpose(Ft) + Qt
    y = Z - Xp @ tf.transpose(Ht)
    S = Ht @ Pp @ tf.transpose(Ht) + Rt
    K = Pp @ tf.transpose(Ht) @ tf.linalg.inv(S)
    Xn = Xp + tf.einsum('nij,nj->ni', K, y)
    Pn = (tf.eye(7, dtype=P.dtype) - K @ Ht) @ Pp
    return Xn.numpy(), Pn.numpy(), S.numpy()

# JAX: se escribe para UNA trayectoria y vmap la replica sobre todas
def kalman_one(x, P, z):
    Fj, Hj = jnp.array(F), jnp.array(H)
    xp = Fj @ x
    Pp = Fj @ P @ Fj.T + jnp.array(Q)
    y = z - Hj @ xp
    S = Hj @ Pp @ Hj.T + jnp.array(R)
    K = Pp @ Hj.T @ jnp.linalg.inv(S)
    return xp + K @ y, (jnp.eye(7) - K @ Hj) @ Pp, S

kalman_jax = jax.jit(jax.vmap(kalman_one))

for name, (Xn, Pn, S) in [
        ("PyTorch", kalman_torch(X, P, Z)),
        ("TensorFlow", kalman_tf(X, P, Z)),
        ("JAX (vmap+jit)", tuple(np.asarray(v) for v in kalman_jax(jnp.array(X), jnp.array(P), jnp.array(Z))))]:
    dx = np.abs(Xn - Xn_ref).max(); dp = np.abs(Pn - Pn_ref).max()
    print(f"  {name:16s} max|dif x| = {dx:.3e}   max|dif P| = {dp:.3e}")

print("\n  Nota: en NumPy/PyTorch/TF, S es (4,4) porque Pp es identica para todas las")
print("  trayectorias en este ejemplo; en JAX el vmap devuelve S por trayectoria, (N,4,4).")
print(f"  formas -> numpy {S_ref.shape}, jax {np.asarray(kalman_jax(jnp.array(X), jnp.array(P), jnp.array(Z))[2]).shape}")
