"""Fusion audiovisual bajo ruido: reproducir la forma de la Fig. 3 de Petridis et al. (2018).
Pregunta: por que la fusion aporta 0,3 puntos en audio limpio y 14,1 a -5 dB?"""
import numpy as np

rng = np.random.default_rng(11)
K, N, D = 40, 20000, 24          # clases, muestras, dimension por modalidad

# Cada clase tiene un prototipo distinto en cada modalidad.
proto_a = rng.normal(0, 1, (K, D))
proto_v = rng.normal(0, 1, (K, D))
y = rng.integers(0, K, N)

# El canal VISUAL es intrinsecamente mas ambiguo: los visemas colapsan fonemas
# distintos ('p','b','m' se ven igual). Lo modelamos haciendo que grupos de 2 clases
# compartan casi el mismo prototipo visual.
for k in range(0, K, 2):
    proto_v[k + 1] = proto_v[k] + rng.normal(0, 0.35, D)

sigma_v = 1.0                     # ruido visual, FIJO (no depende del SNR acustico)

def genera(snr_db):
    sigma_a = 3.2 * 10 ** (-snr_db / 20)    # ruido acustico segun SNR
    Xa = proto_a[y] + rng.normal(0, sigma_a, (N, D))
    Xv = proto_v[y] + rng.normal(0, sigma_v, (N, D))
    return Xa, Xv

def clasifica_gauss(X, proto, sigma):
    """Log-verosimilitud por clase bajo ruido gaussiano isotropico -> logits."""
    d2 = ((X[:, None, :] - proto[None, :, :]) ** 2).sum(-1)
    return -d2 / (2 * sigma ** 2)

def acc(logits): return (logits.argmax(1) == y).mean()

def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

print("Montaje: 40 clases. El canal visual tiene pares de clases casi identicos")
print("(el analogo de los visemas: 'p'/'b'/'m' se ven igual en los labios).")
print("El ruido visual es FIJO; solo el acustico cambia con el SNR.\n")

print(f"{'SNR (dB)':>9s} {'A':>8s} {'V':>8s} {'AV tardia':>11s} {'AV optima':>11s} "
      f"{'AV-A':>8s}")
print("-" * 62)
filas = []
for snr in [-5, 0, 5, 10, 15, 20]:
    Xa, Xv = genera(snr)
    sigma_a = 3.2 * 10 ** (-snr / 20)
    la = clasifica_gauss(Xa, proto_a, sigma_a)
    lv = clasifica_gauss(Xv, proto_v, sigma_v)
    a, v = acc(la), acc(lv)
    # fusion tardia: promediar las probabilidades de cada rama
    av_late = acc(np.log(0.5 * softmax(la) + 0.5 * softmax(lv) + 1e-300))
    # fusion optima: sumar log-verosimilitudes (las dos observaciones son
    # condicionalmente independientes dada la clase)
    av_opt = acc(la + lv)
    filas.append((snr, 100*a, 100*v, 100*av_late, 100*av_opt))
    print(f"{snr:9d} {100*a:7.2f}% {100*v:7.2f}% {100*av_late:10.2f}% "
          f"{100*av_opt:10.2f}% {100*(av_opt-a):+7.2f}")

print(f"""
Tres cosas que reproduce el montaje, y que son el argumento del paper:

1. La columna V es CONSTANTE ({filas[0][2]:.1f}% a -5 dB, {filas[-1][2]:.1f}% a 20 dB).
   El ruido acustico no toca el canal visual. En la Fig. 3 del paper es
   literalmente una linea horizontal, y esa es toda la razon de ser de la fusion.

2. La ganancia de la fusion CRECE al bajar el SNR:
   {filas[-1][4]-filas[-1][1]:+.2f} puntos a 20 dB, {filas[0][4]-filas[0][1]:+.2f} a -5 dB.
   El paper mide +0,3 en limpio y +14,1 a -5 dB: la misma forma.

3. En audio limpio la fusion aporta poco PORQUE NO HAY NADA QUE APORTAR: el audio
   ya esta cerca del techo. La modalidad debil solo ayuda donde la fuerte falla.
""")

# ---------------------------------------------------------------- techo del visual
print("=" * 62)
print("Por que el canal visual tiene un techo que no depende del ruido")
print("=" * 62)
Xa, Xv = genera(20)
lv = clasifica_gauss(Xv, proto_v, sigma_v)
pred = lv.argmax(1)
mismo_par = (pred // 2) == (y // 2)
print(f"  exactitud visual exacta        : {100*(pred == y).mean():.2f}%")
print(f"  exactitud 'acierta el par'     : {100*mismo_par.mean():.2f}%")
print(f"  errores que caen DENTRO del par: {100*(mismo_par & (pred != y)).sum()/max((pred != y).sum(),1):.1f}%")
print("""
  Casi todo el error visual es confusion dentro del par ambiguo. Ese techo no baja
  con mas datos ni con una red mas grande: la informacion no esta en la imagen.
  Es el analogo de los homofonos visuales de LRW ('America' contra 'American').""")

# ---------------------------------------------------------------- cuando estorba
print("=" * 62)
print("Cuando la fusion ESTORBA: pesos fijos contra un canal roto")
print("=" * 62)
print("  Si una modalidad se degrada mas alla de lo previsto y la fusion no lo sabe,")
print("  el promedio de probabilidades arrastra a la buena.\n")
snr = 20
Xa, _ = genera(snr)
la = clasifica_gauss(Xa, proto_a, 3.2 * 10 ** (-snr / 20))
print(f"  {'sigma visual':>13s} {'V':>8s} {'A':>8s} {'AV tardia':>11s} {'delta':>8s}")
for sv in [1.0, 2.0, 4.0, 8.0, 16.0]:
    Xv_bad = proto_v[y] + rng.normal(0, sv, (N, D))
    lv_bad = clasifica_gauss(Xv_bad, proto_v, 1.0)      # el modelo CREE que sigma=1
    av = acc(np.log(0.5 * softmax(la) + 0.5 * softmax(lv_bad) + 1e-300))
    print(f"  {sv:13.1f} {100*acc(lv_bad):7.2f}% {100*acc(la):7.2f}% {100*av:10.2f}% "
          f"{100*(av - acc(la)):+7.2f}")
print("""
  Con el canal visual degradado y el peso fijo en 0,5, la fusion queda POR DEBAJO
  del audio solo. El paper entrena la BGRU de fusion sobre datos con ruido inyectado
  entre -5 y 20 dB justamente para que aprenda a ponderar segun la condicion, en vez
  de promediar a ciegas.""")
