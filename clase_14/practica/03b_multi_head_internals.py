"""
Multi-Head Internals: ver con tus ojos lo que pasa numericamente.

Usamos d_model=4, h=2, d_k=2 (muy chiquito) e imprimimos TODO. Cada
numero del calculo se ve. La idea es que no quede ninguna duda sobre
que pasa "de verdad" cuando dividimos en cabezas.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_printoptions(precision=3, sci_mode=False, linewidth=120)


# ---------------------------------------------------------------------------
# Setup tiny
# ---------------------------------------------------------------------------
print("=" * 72)
print("SETUP")
print("=" * 72)

T = 3        # 3 tokens
d_model = 4  # cada token tiene 4 dimensiones
h = 2        # 2 cabezas
d_k = d_model // h  # = 2 dimensiones por cabeza

# Embeddings INVENTADOS (no random) para que veas la transformacion clara.
x = torch.tensor([[
    [1.0, 0.0, 0.0, 0.0],   # token 0: vector "puro" en dim 0
    [0.0, 1.0, 0.0, 0.0],   # token 1: vector "puro" en dim 1
    [0.5, 0.5, 0.5, 0.5],   # token 2: equiparte
]])  # shape: (1, 3, 4) = (batch, T, d_model)

print(f"\nT = {T}, d_model = {d_model}, h = {h}, d_k = {d_k}")
print(f"\nx (input embeddings) shape: {x.shape}")
print(f"x:")
print(x[0])
print("\nLeer asi: cada FILA es un token con 4 dims.")
print("  token 0 = [1, 0, 0, 0]")
print("  token 1 = [0, 1, 0, 0]")
print("  token 2 = [0.5, 0.5, 0.5, 0.5]")


# ---------------------------------------------------------------------------
# La matriz W_Q grande
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("PASO 1: La matriz W_Q completa (4x4 = 16 numeros)")
print("=" * 72)

W_Q = nn.Linear(d_model, d_model, bias=False)
print(f"\nW_Q.weight tiene shape {W_Q.weight.shape}")
print("W_Q.weight (la matriz interna que PyTorch guarda):")
print(W_Q.weight.data)

print("""
IMPORTANTE: PyTorch guarda W como (out, in). La operacion es:
  output[i] = sum_j input[j] * W[i, j]

Para nuestro proposito, la "columna conceptual" del output es la "fila"
de W.weight. Cuando dividamos en cabezas, vamos a "dividir las filas"
de W.weight en grupos de d_k = 2.

Vamos a interpretar W asi:
  Filas 0-1 (primera mitad) -> cabeza 0
  Filas 2-3 (segunda mitad) -> cabeza 1
""")


# ---------------------------------------------------------------------------
# Aplicar W_Q al input completo
# ---------------------------------------------------------------------------
print("=" * 72)
print("PASO 2: Q = x @ W_Q.T (sin reshape todavia)")
print("=" * 72)

Q = W_Q(x)  # shape: (1, T, d_model) = (1, 3, 4)
print(f"\nQ shape: {Q.shape}")
print(f"Q (sin reshape):")
print(Q[0])

print("""
Cada fila de Q es la "query" del token correspondiente, en 4 dimensiones.
Q[0] es la query del token 0. Q[1] del token 1. Q[2] del token 2.

VERIFICACION manual: Q[0, 0] (primera dim de la query del token 0) deberia
ser sum_j x[0,j] * W[0, j].
""")

# Verificacion manual del primer numero
manual_q00 = sum(x[0, 0, j].item() * W_Q.weight[0, j].item() for j in range(d_model))
print(f"Calculo manual: Q[0, 0] = sum_j x[0,0,j] * W_Q[0,j] = {manual_q00:.4f}")
print(f"Valor de PyTorch: Q[0, 0] = {Q[0, 0, 0].item():.4f}")
print(f"Coinciden: {abs(manual_q00 - Q[0, 0, 0].item()) < 1e-5}")
print("\nEs decir: Q[0,0] uso TODOS los valores de x[0,0,:] y los combino")
print("con la PRIMERA fila de W_Q.weight.")


# ---------------------------------------------------------------------------
# Reshape: aqui aparecen las cabezas
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("PASO 3: Reshape de Q para dividir en cabezas")
print("=" * 72)

print(f"\nAntes:    Q.shape = {Q.shape} = (B, T, d_model)")

Q_reshaped = Q.view(1, T, h, d_k)
print(f"Despues:  Q_reshaped.shape = {Q_reshaped.shape} = (B, T, h, d_k)")

print("\nLA CLAVE: el reshape NO copia datos. Solo reinterpreta el tensor.")
print("\nQ original (cada token tiene 4 numeros):")
print(Q[0])
print("\nQ_reshaped (cada token tiene 2 cabezas, cada una con 2 numeros):")
print(Q_reshaped[0])

print("""
Notar: los MISMOS 12 numeros se ven distinto.
  Q[0, 0] = [a, b, c, d] (4 numeros)
  Q_reshaped[0, 0] = [[a, b], [c, d]] (2 cabezas con 2 numeros cada una)

  -> cabeza 0 del token 0: [a, b]   <- los primeros 2 numeros de Q[0,0]
  -> cabeza 1 del token 0: [c, d]   <- los ultimos 2 numeros de Q[0,0]
""")


# ---------------------------------------------------------------------------
# Permute: poner la dim de cabezas al frente
# ---------------------------------------------------------------------------
print("=" * 72)
print("PASO 4: Transpose para vectorizar las cabezas")
print("=" * 72)

Q_heads = Q_reshaped.transpose(1, 2)  # (B, h, T, d_k)
print(f"\nAntes:    Q_reshaped.shape = {Q_reshaped.shape} = (B, T, h, d_k)")
print(f"Despues:  Q_heads.shape = {Q_heads.shape} = (B, h, T, d_k)")

print("\nQ_heads[0, 0] = matriz (T, d_k) de la CABEZA 0 (con los T tokens):")
print(Q_heads[0, 0])
print("\nQ_heads[0, 1] = matriz (T, d_k) de la CABEZA 1:")
print(Q_heads[0, 1])

print("""
Ahora cada cabeza tiene su propio "tensor independiente" de Q. Cada
fila es un token, cada cabeza ve los T tokens en su subespacio d_k.
""")


# ---------------------------------------------------------------------------
# Verificacion: la cabeza 0 usa las "filas 0-1" de W_Q
# ---------------------------------------------------------------------------
print("=" * 72)
print("PASO 5: Verificacion - la cabeza 0 usa filas 0-1 de W_Q")
print("=" * 72)

# Hagamos el calculo de la cabeza 0 a mano, usando solo filas 0-1 de W_Q
W_Q_cabeza0 = W_Q.weight[0:2, :]  # shape (2, 4)
W_Q_cabeza1 = W_Q.weight[2:4, :]  # shape (2, 4)

print(f"\nW_Q_cabeza0 (filas 0-1 de W_Q.weight): shape {W_Q_cabeza0.shape}")
print(W_Q_cabeza0)
print(f"\nW_Q_cabeza1 (filas 2-3 de W_Q.weight): shape {W_Q_cabeza1.shape}")
print(W_Q_cabeza1)

# Calcular Q de la cabeza 0 manualmente: x @ W_Q_cabeza0.T
Q_cabeza0_manual = x[0] @ W_Q_cabeza0.T  # (T, 2)
Q_cabeza1_manual = x[0] @ W_Q_cabeza1.T  # (T, 2)

print(f"\nQ_cabeza0_manual = x @ W_Q_cabeza0.T:")
print(Q_cabeza0_manual)
print(f"\nQ obtenido por reshape (Q_heads[0, 0]):")
print(Q_heads[0, 0])

diff0 = (Q_cabeza0_manual - Q_heads[0, 0]).abs().max().item()
diff1 = (Q_cabeza1_manual - Q_heads[0, 1]).abs().max().item()
print(f"\nMax diferencia cabeza 0: {diff0:.2e}")
print(f"Max diferencia cabeza 1: {diff1:.2e}")
print("\nLas dos formas dan EXACTAMENTE lo mismo. La cabeza 0 efectivamente")
print("usa solo las filas 0-1 de W_Q.weight, la cabeza 1 las filas 2-3.")
print("El reshape implementa esa division SIN copiar datos.")


# ---------------------------------------------------------------------------
# Cada cabeza ve el x completo
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("PASO 6: Verificacion - cada cabeza VE el x completo")
print("=" * 72)

print("""
Mira W_Q_cabeza0:
""")
print(W_Q_cabeza0)
print("""
Cada fila tiene 4 numeros (porque opera sobre las 4 dims de x). La
cabeza 0 NO ve solo "las primeras 2 dims de x" — lee TODAS las 4 dims
para producir cada uno de sus 2 outputs.

Ejemplo: para producir el primer numero de Q de la cabeza 0 para token 0:
  Q_cabeza0[0, 0] = x[0,0,0] * W[0,0] + x[0,0,1] * W[0,1]
                  + x[0,0,2] * W[0,2] + x[0,0,3] * W[0,3]
                  ^^^^^^^^^^   ^^^^^^^   ^^^^^^^^   ^^^^^^^
                  4 contribuciones distintas, una por cada dim de x

La "compresion" es del OUTPUT (de 4 dims a 2 dims por cabeza), no del INPUT.
""")

manual_val = sum(
    x[0, 0, j].item() * W_Q_cabeza0[0, j].item()
    for j in range(d_model)
)
print(f"Calculo manual paso a paso:")
print(f"  x[0,0,0] * W[0,0] = {x[0,0,0].item():.3f} * {W_Q_cabeza0[0,0].item():.3f} = {x[0,0,0].item()*W_Q_cabeza0[0,0].item():.4f}")
print(f"  x[0,0,1] * W[0,1] = {x[0,0,1].item():.3f} * {W_Q_cabeza0[0,1].item():.3f} = {x[0,0,1].item()*W_Q_cabeza0[0,1].item():.4f}")
print(f"  x[0,0,2] * W[0,2] = {x[0,0,2].item():.3f} * {W_Q_cabeza0[0,2].item():.3f} = {x[0,0,2].item()*W_Q_cabeza0[0,2].item():.4f}")
print(f"  x[0,0,3] * W[0,3] = {x[0,0,3].item():.3f} * {W_Q_cabeza0[0,3].item():.3f} = {x[0,0,3].item()*W_Q_cabeza0[0,3].item():.4f}")
print(f"  SUMA = {manual_val:.4f}")
print(f"PyTorch dio:  {Q_heads[0, 0, 0, 0].item():.4f}")
print(f"Coinciden: {abs(manual_val - Q_heads[0, 0, 0, 0].item()) < 1e-5}")


# ---------------------------------------------------------------------------
# Cada cabeza tiene su propia atencion
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("PASO 7: Cada cabeza calcula su propia atencion (independiente)")
print("=" * 72)

W_K = nn.Linear(d_model, d_model, bias=False)
W_V = nn.Linear(d_model, d_model, bias=False)

K = W_K(x).view(1, T, h, d_k).transpose(1, 2)  # (1, h, T, d_k)
V = W_V(x).view(1, T, h, d_k).transpose(1, 2)

# scores: (B, h, T, T) — UNA matriz de scores por cabeza
scores = Q_heads @ K.transpose(-2, -1) / math.sqrt(d_k)
print(f"\nscores shape: {scores.shape}  (B, h, T, T) = (1, 2, 3, 3)")

print("\nMatriz de scores de la CABEZA 0 (3x3):")
print(scores[0, 0])

print("\nMatriz de scores de la CABEZA 1 (3x3):")
print(scores[0, 1])

print("\nNo son iguales. Cada cabeza tiene una geometria distinta de comparacion.")

weights = F.softmax(scores, dim=-1)
print(f"\nWeights cabeza 0 (despues de softmax, cada fila suma 1):")
print(weights[0, 0])
print(f"\nWeights cabeza 1:")
print(weights[0, 1])

print("\nCada cabeza tiene su propia distribucion de pesos. Esa es la ganancia")
print("de multi-head: dos formas distintas de mezclar la oracion en paralelo.")


# ---------------------------------------------------------------------------
# Output: concatenar las cabezas
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("PASO 8: Concatenar outputs de las cabezas")
print("=" * 72)

head_outputs = weights @ V   # (1, h, T, d_k)
print(f"\nhead_outputs shape: {head_outputs.shape}")

print(f"\nOutput de la cabeza 0 (T tokens, cada uno con d_k=2 dims):")
print(head_outputs[0, 0])
print(f"\nOutput de la cabeza 1:")
print(head_outputs[0, 1])

# Concatenar (volver a juntar)
concat = head_outputs.transpose(1, 2).contiguous().view(1, T, d_model)
print(f"\nDespues de concatenar (volver a B, T, d_model):")
print(f"concat shape: {concat.shape}")
print(concat[0])

print("""
Concatenar = poner las salidas de cada cabeza una al lado de la otra:
  token 0 final = [cabeza0_token0 (2 dims) | cabeza1_token0 (2 dims)] = 4 dims
  token 1 final = [cabeza0_token1 (2 dims) | cabeza1_token1 (2 dims)] = 4 dims
  token 2 final = [cabeza0_token2 (2 dims) | cabeza1_token2 (2 dims)] = 4 dims

Las 4 dims del output reconstruyen el d_model original. Mismas dimensiones
que la entrada. PERO los numeros son combinaciones de DOS atenciones distintas.
""")


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------
print("=" * 72)
print("RESUMEN")
print("=" * 72)

print("""
LO QUE ACABAS DE VER:

1. W_Q es UNA matriz de 4x4 (16 numeros). NO hay 2 matrices distintas
   para las 2 cabezas — solo una grande.

2. Cuando aplicas W_Q a x, sale una Q de 4 dims por token. Esos 4 numeros
   se pueden VER como "2 cabezas de 2 dims cada una".

3. La cabeza 0 efectivamente usa solo las filas 0-1 de W_Q.weight.
   La cabeza 1 usa solo las filas 2-3.
   Esto es matematicamente identico a tener 2 matrices separadas de 2x4.

4. PERO cada cabeza, dentro de su slice, sigue leyendo TODAS las 4 dims
   del input x. La compresion es del output (4 -> 2 por cabeza), no del input.

5. Cada cabeza calcula su propia matriz de scores y su propia softmax.
   Resultado: 2 distribuciones de pesos distintas, que mezclan los tokens
   de 2 maneras distintas.

6. Al final, los 2 outputs se concatenan y se proyectan con W_O.

LA INFORMACION DEL INPUT NO SE PIERDE — se PROYECTA a 2 subespacios
distintos en paralelo, cada uno con su propia mezcla. Las matrices grandes
W_Q, W_K, W_V actuan como "h matrices chicas pegadas" que el reshape
"separa" sin copiar datos.
""")

print("=" * 72)
print("FIN.")
print("=" * 72)
