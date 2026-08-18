"""El 'informed guess' de la clase, hecho preciso.

La clase dice que la super-resolucion es una 'conjetura informada'. Este script
mide en que sentido exacto: cuanta informacion se destruye, cual es la solucion
optima en MSE, y por que esa solucion optima se ve mal.
"""
import numpy as np
import warnings; warnings.filterwarnings("ignore")

rng = np.random.default_rng(0)

# ---------------------------------------------------------------- 1. cuanto se pierde
print("=" * 78)
print("1. El operador de bajada no es invertible: cuantas HR dan la misma LR")
print("=" * 78)
print("  Con imagenes binarias y promediado por bloques de f x f, cada pixel LR")
print("  solo conserva la SUMA del bloque. Todas las configuraciones con la misma")
print("  suma son indistinguibles.\n")
from math import comb
print(f"  {'factor f':>9s} {'pixeles/bloque':>15s} {'HR posibles':>14s} "
      f"{'valores LR':>11s} {'preimagen media':>17s}")
for f in [2, 3, 4]:
    n = f * f
    total = 2 ** n
    niveles = n + 1
    media = total / niveles
    print(f"  {f:8d}x {n:15d} {total:14d} {niveles:11d} {media:17.1f}")
print("""
  Con factor 4 hay 65 536 parches binarios distintos y solo 17 salidas posibles:
  la bajada es 3855 a 1 en promedio. Ninguna informacion adicional puede
  recuperar cual era el original -- solo un PRIOR sobre que parches son plausibles.""")

# ---------------------------------------------------------------- 2. el optimo en MSE
print("=" * 78)
print("2. La solucion optima en MSE es el PROMEDIO de las candidatas")
print("=" * 78)
f = 4
n = f * f
# distribucion realista: parches con estructura (bordes), no ruido uniforme
def genera_parches(m):
    """Parches 4x4 binarios con un borde vertical u horizontal en posicion aleatoria."""
    out = []
    for _ in range(m):
        p = np.zeros((f, f))
        if rng.random() < 0.5:
            k = rng.integers(1, f); p[:, :k] = 1
        else:
            k = rng.integers(1, f); p[:k, :] = 1
        if rng.random() < 0.5: p = 1 - p
        out.append(p)
    return np.array(out)

parches = genera_parches(20000)
sumas = parches.reshape(-1, n).sum(1)

# para una LR dada (una suma), las HR compatibles y su promedio
objetivo = 8
compat = parches[sumas == objetivo]
promedio = compat.mean(0)
print(f"  Observamos un pixel LR con suma {objetivo} (de {n}).")
print(f"  Parches compatibles en el conjunto: {len(compat)}\n")
print("  Promedio de todas las HR compatibles (= prediccion optima en MSE):")
for fila in promedio:
    print("    " + "  ".join(f"{v:.2f}" for v in fila))

muestra = compat[rng.integers(len(compat))]
print("\n  Una MUESTRA de la posterior (una HR plausible concreta):")
for fila in muestra:
    print("    " + "  ".join(f"{v:.2f}" for v in fila))

mse_prom = ((compat - promedio) ** 2).mean()
mse_muestra = ((compat - muestra) ** 2).mean()
print(f"\n  MSE esperado del promedio  : {mse_prom:.4f}")
print(f"  MSE esperado de la muestra : {mse_muestra:.4f}   ({mse_muestra/mse_prom:.2f}x peor)")
print(f"  Nitidez (varianza espacial) del promedio  : {promedio.var():.4f}")
print(f"  Nitidez (varianza espacial) de la muestra : {muestra.var():.4f}")
print("""
  El promedio gana en MSE por construccion (es la esperanza condicional) y pierde
  en nitidez: sus valores son grises intermedios que NO corresponden a ninguna
  imagen real. La muestra tiene el doble de error cuadratico y es la unica de las
  dos que podria ser una foto.""")

# ---------------------------------------------------------------- 3. distorsion-percepcion
print("=" * 78)
print("3. El intercambio distorsion-percepcion, barrido")
print("=" * 78)
print("  Interpolamos entre el promedio (optimo en MSE) y una muestra de la")
print("  posterior, y medimos las dos cosas a la vez.\n")
print(f"  {'alpha':>6s} {'MSE':>10s} {'PSNR (dB)':>11s} {'nitidez':>10s} "
      f"{'dist. a la distrib. real':>26s}")
var_real = np.array([p.var() for p in compat]).mean()
for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
    pred = (1 - a) * promedio + a * muestra
    mse = ((compat - pred) ** 2).mean()
    psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
    nit = pred.var()
    print(f"  {a:6.2f} {mse:10.4f} {psnr:11.2f} {nit:10.4f} {abs(nit - var_real):26.4f}")
print(f"\n  (la nitidez media de las HR reales es {var_real:.4f})")
print("""
  alpha=0 minimiza el MSE y maximiza el PSNR, y es lo mas lejos que se puede estar
  de la estadistica de las imagenes reales. alpha=1 iguala esa estadistica y tiene
  el PEOR PSNR. No hay un punto que optimice ambas: es un intercambio, no una
  transicion. Blau y Michaeli (2018) lo demostraron formalmente.

  De ahi que las metricas de la literatura de super-resolucion se hayan dividido en
  dos familias -- PSNR/SSIM por un lado, LPIPS/FID por el otro -- y que los modelos
  generativos (GAN, difusion) ganen en la segunda mientras pierden en la primera.
""")

# ---------------------------------------------------------------- 4. alucinacion
print("=" * 78)
print("4. Lo que 'informed guess' significa en la practica: el prior decide")
print("=" * 78)
print("  Dos priors distintos sobre el mismo pixel LR producen dos HR distintas,")
print("  ambas perfectamente consistentes con la observacion.\n")
verticales, horizontales = [], []
for p_ in compat:
    col = np.abs(np.diff(p_, axis=1)).sum()
    fil = np.abs(np.diff(p_, axis=0)).sum()
    if col > 0 and fil == 0: verticales.append(p_)
    elif fil > 0 and col == 0: horizontales.append(p_)
print(f"  compatibles con borde ESTRICTAMENTE vertical  : {len(verticales)}")
print(f"  compatibles con borde ESTRICTAMENTE horizontal: {len(horizontales)}")
print("\n  una reconstruccion bajo prior 'bordes verticales':")
for fila in verticales[0]: print("    " + "  ".join(f"{v:.0f}" for v in fila))
print("\n  una reconstruccion bajo prior 'bordes horizontales':")
for fila in horizontales[0]: print("    " + "  ".join(f"{v:.0f}" for v in fila))
print(f"\n  las dos bajan al MISMO pixel LR: suma {verticales[0].sum():.0f} y {horizontales[0].sum():.0f}")
print("""
  Las dos son consistentes con lo observado y son distintas. El modelo no recupera
  informacion: la aporta. Por eso 'super-resolucion' no es una operacion forense y
  no debe usarse como si lo fuera -- un rostro o una patente 'recuperados' de un
  video de vigilancia son lo que el prior del modelo considera probable, no lo que
  habia en la escena.
""")
