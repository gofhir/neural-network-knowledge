"""Triple framework para la clase 44: el campo de movimiento de primer orden
(FOMM) y el warping diferenciable. Verifica que los cuatro backends coinciden."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import warnings; warnings.filterwarnings("ignore")

rng = np.random.default_rng(3)
H = W = 32
K = 10
ys, xs = np.mgrid[0:H, 0:W]
grid = np.stack([xs/(W-1) - .5, ys/(H-1) - .5], -1).astype(np.float64)   # (H,W,2)
kp   = rng.uniform(-.35, .35, (K, 2))          # posiciones en la fuente
kp_d = kp + rng.normal(0, .05, (K, 2))         # posiciones en el conductor
jac  = np.eye(2) + rng.normal(0, .15, (K, 2, 2))   # jacobianos por keypoint
sigma = 0.15

# ---------------------------------------------------------------- NumPy
def campo_np(grid, kp, kp_d, jac, sigma):
    """Campo denso por combinacion de aproximaciones locales de primer orden."""
    d = grid[:, :, None, :] - kp_d[None, None, :, :]          # (H,W,K,2)
    w = np.exp(-(d**2).sum(-1) / (2*sigma**2))                # (H,W,K)
    w = w / w.sum(-1, keepdims=True)
    loc = kp[None, None, :, :] + np.einsum('kij,hwkj->hwki', jac, d)
    return (w[..., None] * loc).sum(2)                        # (H,W,2)

ref = campo_np(grid, kp, kp_d, jac, sigma)
print(f"campo de movimiento: {ref.shape},  rango [{ref.min():.4f}, {ref.max():.4f}]")

# ---------------------------------------------------------------- PyTorch
import torch
def campo_torch(grid, kp, kp_d, jac, sigma):
    g, k, kd, J = map(torch.as_tensor, (grid, kp, kp_d, jac))
    d = g[:, :, None, :] - kd[None, None, :, :]
    w = torch.exp(-(d**2).sum(-1) / (2*sigma**2))
    w = w / w.sum(-1, keepdim=True)
    loc = k[None, None, :, :] + torch.einsum('kij,hwkj->hwki', J, d)
    return (w[..., None] * loc).sum(2)

# ---------------------------------------------------------------- TensorFlow
import tensorflow as tf
def campo_tf(grid, kp, kp_d, jac, sigma):
    g, k, kd, J = map(tf.constant, (grid, kp, kp_d, jac))
    d = g[:, :, None, :] - kd[None, None, :, :]
    w = tf.exp(-tf.reduce_sum(d**2, -1) / (2*sigma**2))
    w = w / tf.reduce_sum(w, -1, keepdims=True)
    loc = k[None, None, :, :] + tf.einsum('kij,hwkj->hwki', J, d)
    return tf.reduce_sum(w[..., None] * loc, 2)

# ---------------------------------------------------------------- JAX
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

def campo_local(p, kp, kp_d, jac, sigma):
    """Se escribe para UN pixel; vmap lo aplica a toda la imagen."""
    d = p - kp_d                                    # (K,2)
    w = jnp.exp(-(d**2).sum(-1) / (2*sigma**2))
    w = w / w.sum()
    loc = kp + jnp.einsum('kij,kj->ki', jac, d)
    return (w[:, None] * loc).sum(0)

campo_jax = jax.jit(jax.vmap(jax.vmap(campo_local,
                    in_axes=(0, None, None, None, None)),
                    in_axes=(0, None, None, None, None)))

print("\n=== Campo de movimiento de primer orden ===")
for nombre, val in [
        ("PyTorch",    campo_torch(grid, kp, kp_d, jac, sigma).numpy()),
        ("TensorFlow", campo_tf(grid, kp, kp_d, jac, sigma).numpy()),
        ("JAX (vmap anidado)", np.asarray(campo_jax(jnp.array(grid), jnp.array(kp),
                                jnp.array(kp_d), jnp.array(jac), sigma)))]:
    print(f"  {nombre:22s} max|dif| vs NumPy = {np.abs(val - ref).max():.3e}")

# ================================================================ warping
print("\n=== Warping bilineal diferenciable (back-warping) ===")
img = rng.random((H, W)).astype(np.float64)

def warp_np(img, campo):
    u = (campo[..., 0] + .5) * (W - 1)
    v = (campo[..., 1] + .5) * (H - 1)
    u = np.clip(u, 0, W-1); v = np.clip(v, 0, H-1)
    x0 = np.floor(u).astype(int); y0 = np.floor(v).astype(int)
    x1 = np.minimum(x0+1, W-1);   y1 = np.minimum(y0+1, H-1)
    a = u - x0; b = v - y0
    return (img[y0, x0]*(1-a)*(1-b) + img[y0, x1]*a*(1-b) +
            img[y1, x0]*(1-a)*b     + img[y1, x1]*a*b)

wref = warp_np(img, ref)

it = torch.as_tensor(img)[None, None]
gt = torch.as_tensor(np.stack([ref[..., 0]*2, ref[..., 1]*2], -1))[None]
wt = torch.nn.functional.grid_sample(it, gt, mode="bilinear",
                                     padding_mode="border", align_corners=True)
print(f"  PyTorch grid_sample      max|dif| vs NumPy = "
      f"{np.abs(wt[0,0].numpy() - wref).max():.3e}")
print("""
  grid_sample de PyTorch implementa exactamente esta operacion, y es la razon por
  la que FOMM usa flujo optico HACIA ATRAS: el back-warping se expresa con muestreo
  bilineal, que es diferenciable y tiene kernel optimizado. El forward warping
  exigiria dispersion (scatter) con colisiones, que no lo es.
""")
