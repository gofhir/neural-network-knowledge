"""Triple framework para la clase 43: perdida de destilacion con temperatura
y fusion de log-verosimilitudes. Verifica que los cuatro backends coinciden."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np

rng = np.random.default_rng(5)
B, K = 7, 12
v = rng.normal(0, 3.0, (B, K))     # logits del maestro
z = rng.normal(0, 3.0, (B, K))     # logits del alumno
T = 4.0

# ---------------------------------------------------------------- NumPy
def softmax_np(x, T=1.0):
    x = x / T; x = x - x.max(-1, keepdims=True)
    e = np.exp(x); return e / e.sum(-1, keepdims=True)

def kd_loss_np(z, v, T):
    p = softmax_np(v, T); q = softmax_np(z, T)
    return (p * (np.log(p) - np.log(q))).sum(-1).mean() * T**2

def kd_grad_np(z, v, T):
    return (softmax_np(z, T) - softmax_np(v, T)) / T * T**2 / z.shape[0]

L_ref = kd_loss_np(z, v, T)
G_ref = kd_grad_np(z, v, T)
print(f"NumPy      loss = {L_ref:.10f}")

# ---------------------------------------------------------------- PyTorch
import torch
import torch.nn.functional as F
zt = torch.tensor(z, requires_grad=True); vt = torch.tensor(v)
L = F.kl_div(F.log_softmax(zt / T, -1), F.log_softmax(vt / T, -1),
             reduction="batchmean", log_target=True) * T**2
L.backward()
print(f"PyTorch    loss = {L.item():.10f}   dif = {abs(L.item()-L_ref):.2e}")
print(f"           grad max|dif| = {np.abs(zt.grad.numpy() - G_ref).max():.2e}")

# ---------------------------------------------------------------- TensorFlow
import tensorflow as tf
zf = tf.Variable(z); vf = tf.constant(v)
with tf.GradientTape() as tape:
    p = tf.nn.softmax(vf / T); logq = tf.nn.log_softmax(zf / T)
    Lf = tf.reduce_mean(tf.reduce_sum(p * (tf.math.log(p) - logq), -1)) * T**2
gf = tape.gradient(Lf, zf)
print(f"TensorFlow loss = {Lf.numpy():.10f}   dif = {abs(Lf.numpy()-L_ref):.2e}")
print(f"           grad max|dif| = {np.abs(gf.numpy() - G_ref).max():.2e}")

# ---------------------------------------------------------------- JAX
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

def kd_loss_jax(z, v, T):
    logp = jax.nn.log_softmax(v / T); logq = jax.nn.log_softmax(z / T)
    return jnp.mean(jnp.sum(jnp.exp(logp) * (logp - logq), -1)) * T**2

Lj = kd_loss_jax(jnp.array(z), jnp.array(v), T)
gj = jax.grad(kd_loss_jax)(jnp.array(z), jnp.array(v), T)
print(f"JAX        loss = {float(Lj):.10f}   dif = {abs(float(Lj)-L_ref):.2e}")
print(f"           grad max|dif| = {np.abs(np.asarray(gj) - G_ref).max():.2e}")

# ================================================================ fusion
print()
print("=" * 70)
print("Fusion de log-verosimilitudes: suma contra promedio de probabilidades")
print("=" * 70)
la = rng.normal(0, 5.0, (B, K))     # logits de la rama de audio
lv = rng.normal(0, 2.0, (B, K))     # logits de la rama visual (menos confiada)

def fus_suma_np(la, lv):  return la + lv
def fus_prom_np(la, lv):  return np.log(0.5*softmax_np(la) + 0.5*softmax_np(lv))
ref_suma, ref_prom = fus_suma_np(la, lv), fus_prom_np(la, lv)

lat, lvt = torch.tensor(la), torch.tensor(lv)
t_suma = (lat + lvt).numpy()
t_prom = torch.log(0.5*torch.softmax(lat, -1) + 0.5*torch.softmax(lvt, -1)).numpy()

f_suma = (tf.constant(la) + tf.constant(lv)).numpy()
f_prom = tf.math.log(0.5*tf.nn.softmax(tf.constant(la)) +
                     0.5*tf.nn.softmax(tf.constant(lv))).numpy()

j_suma = np.asarray(jnp.array(la) + jnp.array(lv))
j_prom = np.asarray(jnp.log(0.5*jax.nn.softmax(jnp.array(la)) +
                            0.5*jax.nn.softmax(jnp.array(lv))))

for nombre, s, p in [("PyTorch", t_suma, t_prom), ("TensorFlow", f_suma, f_prom),
                     ("JAX", j_suma, j_prom)]:
    print(f"  {nombre:12s} suma max|dif| = {np.abs(s-ref_suma).max():.2e}   "
          f"promedio max|dif| = {np.abs(p-ref_prom).max():.2e}")

coinciden = (ref_suma.argmax(1) == ref_prom.argmax(1)).mean()
print(f"\n  Las dos fusiones eligen la MISMA clase en {100*coinciden:.1f}% de los casos:")
print("  no son equivalentes, y difieren justo cuando una rama esta mucho mas")
print("  confiada que la otra.")
