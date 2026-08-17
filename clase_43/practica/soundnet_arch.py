"""Verificar la Tabla 1 de SoundNet: dimensiones, campo receptivo y parametros."""

# Tabla 1 del paper (8 capas). (nombre, filtros, kernel, stride, padding, dim_declarada)
L8 = [
    ("conv1",   16,  64, 2, 32, 220050),
    ("pool1",   16,   8, 1,  0,  27506),   # pool con stride segun tabla
    ("conv2",   32,  32, 2, 16,  13782),
    ("pool2",   32,   8, 1,  0,   1722),
    ("conv3",   64,  16, 2,  8,    862),
    ("conv4",  128,   8, 2,  4,    432),
    ("conv5",  256,   4, 2,  2,    217),
    ("pool5",  256,   4, 1,  0,     54),
    ("conv6",  512,   4, 2,  2,     28),
    ("conv7", 1024,   4, 2,  2,     15),
    ("conv8", 1401,   8, 2,  0,      4),
]

SR = 22050
DUR = 20  # s -> el paper usa clips; 220050 muestras = 9.98 s a 22050 Hz
print(f"Entrada declarada en conv1: 220 050 muestras")
print(f"  a 22 050 Hz  ->  {220050/SR:.2f} s de audio\n")

print("== Propagacion de dimensiones (formula out = floor((in + 2p - k)/s) + 1) ==")
print(f"{'capa':8s} {'k':>4s} {'s':>3s} {'p':>3s} {'declarado':>10s} {'calculado':>10s}  {'':4s}")

# El paper declara la dim DE SALIDA de cada capa. Partimos de la entrada de conv1.
# La dim de entrada a conv1 no se declara; la inferimos desde su salida.
def out_dim(n, k, s, p):
    return (n + 2*p - k) // s + 1

# inferir entrada de conv1 tal que salida = 220050 con k=64,s=2,p=32
# 220050 = (n + 64 - 64)//2 + 1  ->  n = 2*(220050-1) = 440098
n_in = 2 * (220050 - 1)
print(f"  entrada inferida a conv1: {n_in} muestras = {n_in/SR:.2f} s\n")

n = n_in
ok = 0
for (name, f, k, s, p, decl) in L8:
    # los pool de la tabla tienen stride 1 declarado, lo que no reduce; el paper
    # usa maxpool con stride igual al kernel. Probamos ambas lecturas.
    calc_s1 = out_dim(n, k, s, p)
    calc_sk = out_dim(n, k, k, p)
    if name.startswith("pool"):
        pick = calc_sk
        nota = f"(stride=k={k}; con stride={s} daria {calc_s1})"
    else:
        pick = calc_s1
        nota = ""
    flag = "OK " if pick == decl else "!= "
    ok += (pick == decl)
    print(f"{name:8s} {k:4d} {s:3d} {p:3d} {decl:10d} {pick:10d}  {flag}{nota}")
    n = pick

print(f"\n  coinciden {ok}/{len(L8)} capas")

print("\n== Campo receptivo acumulado (cuanto audio ve una neurona de cada capa) ==")
rf, jump = 1, 1
print(f"{'capa':8s} {'RF (muestras)':>14s} {'RF (ms)':>10s}")
for (name, f, k, s, p, decl) in L8:
    ks = k
    ss = k if name.startswith("pool") else s
    rf = rf + (ks - 1) * jump
    jump = jump * ss
    print(f"{name:8s} {rf:14d} {1000*rf/SR:10.1f}")

print("\n== Parametros por capa ==")
prev_f = 1
total = 0
for (name, f, k, s, p, decl) in L8:
    if name.startswith("pool"):
        print(f"{name:8s} {'-':>12s}")
        continue
    par = prev_f * f * k + f          # pesos + bias
    total += par
    print(f"{name:8s} {par:12,d}   ({prev_f} -> {f}, k={k})")
    prev_f = f
print(f"{'TOTAL':8s} {total:12,d}")
print(f"  conv8 sola: {256*0+1024*1401*8+1401:,d} ... ver desglose arriba")
