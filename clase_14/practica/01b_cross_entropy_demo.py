"""
Demo cross-entropy: 3 modelos hipoteticos prediciendo la misma respuesta.

Objetivo: ver con tus ojos como -log(P) se traduce en numeros de error
distintos segun que tan seguro este el modelo de la respuesta correcta.

NO estamos entrenando nada. Solo simulamos las salidas que daria cada
modelo (sus probabilidades) y calculamos el loss que tendria.
"""

import math
import torch
import torch.nn.functional as F

torch.set_printoptions(precision=4, sci_mode=False)


# ---------------------------------------------------------------------------
# Setup: vocabulario y respuesta correcta
# ---------------------------------------------------------------------------
print("=" * 70)
print("SETUP")
print("=" * 70)

vocab = ["azul", "oscuro", "verde", "limpio", "hermoso"]
vocab_size = len(vocab)
target_id = 0  # "azul" es el ID 0

print(f"\nVocabulario:        {vocab}")
print(f"Pregunta al modelo: 'El cielo es ___'")
print(f"Respuesta correcta: '{vocab[target_id]}' (ID {target_id})")


# ---------------------------------------------------------------------------
# 3 modelos hipoteticos. Cada uno produce probabilidades sobre el vocab.
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 1: Las predicciones de 3 modelos hipoteticos")
print("=" * 70)

modelo_A = torch.tensor([0.95, 0.02, 0.01, 0.01, 0.01])  # bueno
modelo_B = torch.tensor([0.30, 0.30, 0.20, 0.10, 0.10])  # mediocre
modelo_C = torch.tensor([0.01, 0.50, 0.20, 0.20, 0.09])  # malo

# Validacion: cada distribucion debe sumar 1 (es una probabilidad valida)
for nombre, probs in [("A", modelo_A), ("B", modelo_B), ("C", modelo_C)]:
    suma = probs.sum().item()
    assert abs(suma - 1.0) < 1e-5, f"{nombre} no suma 1: {suma}"

print(f"\n{'palabra':<10}  modelo_A    modelo_B    modelo_C")
print("-" * 50)
for i, palabra in enumerate(vocab):
    marker = "  <-- correcta" if i == target_id else ""
    print(f"{palabra:<10}  {modelo_A[i].item():.4f}      "
          f"{modelo_B[i].item():.4f}      {modelo_C[i].item():.4f}{marker}")

print("\nNotar:")
print("  Modelo A le da 95% a 'azul' -> esta muy seguro de la respuesta correcta.")
print("  Modelo B le da 30% a 'azul' -> esta dudoso.")
print("  Modelo C le da  1% a 'azul' -> esta convencido de algo equivocado.")


# ---------------------------------------------------------------------------
# Calculo manual: -log(P) de la respuesta correcta
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 2: Calcular el loss manualmente con -log(P_correcta)")
print("=" * 70)

p_A = modelo_A[target_id].item()
p_B = modelo_B[target_id].item()
p_C = modelo_C[target_id].item()

loss_A = -math.log(p_A)
loss_B = -math.log(p_B)
loss_C = -math.log(p_C)

print(f"\nModelo A: P(azul) = {p_A:.4f}    loss = -log({p_A:.4f}) = {loss_A:.4f}")
print(f"Modelo B: P(azul) = {p_B:.4f}    loss = -log({p_B:.4f}) = {loss_B:.4f}")
print(f"Modelo C: P(azul) = {p_C:.4f}    loss = -log({p_C:.4f}) = {loss_C:.4f}")

print(f"\nRatio C/A: {loss_C / loss_A:.1f}x mas castigo para el modelo malo.")
print(f"Ratio C/B: {loss_C / loss_B:.1f}x mas que el mediocre.")


# ---------------------------------------------------------------------------
# Verificacion con F.cross_entropy (la funcion oficial de PyTorch)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 3: Verificar con F.cross_entropy de PyTorch")
print("=" * 70)

# F.cross_entropy espera "logits" (numeros crudos sin softmax), NO probabilidades.
# Para convertir probs a logits aplicamos log: logit = log(prob).
# (Internamente cross_entropy hace softmax + log + negar de manera estable
# numericamente; aqui pasamos log(probs) para que el softmax devuelva las probs
# originales y el resultado coincida con nuestro calculo manual.)
target = torch.tensor([target_id])  # batch de tamano 1

for nombre, probs, loss_manual in [
    ("A", modelo_A, loss_A),
    ("B", modelo_B, loss_B),
    ("C", modelo_C, loss_C),
]:
    logits = torch.log(probs).unsqueeze(0)  # shape: (1, vocab_size)
    loss_pytorch = F.cross_entropy(logits, target).item()
    coincide = "OK" if abs(loss_pytorch - loss_manual) < 1e-4 else "MISMATCH"
    print(f"Modelo {nombre}: loss manual = {loss_manual:.4f}, "
          f"F.cross_entropy = {loss_pytorch:.4f}  [{coincide}]")

print("\nLos numeros coinciden: F.cross_entropy hace exactamente lo mismo que")
print("calcular -log(P_correcta) a mano. La funcion es solo una version robusta")
print("y vectorizada del mismo concepto.")


# ---------------------------------------------------------------------------
# Tabla panoramica: como cambia el loss para distintos valores de P
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 4: Tabla panoramica de -log(P)")
print("=" * 70)

print(f"\n{'P(correcta)':>12}  {'-log(P) = loss':>15}  interpretacion")
print("-" * 60)
casos = [1.00, 0.99, 0.95, 0.90, 0.50, 0.30, 0.10, 0.01, 0.001, 0.0001]
for p in casos:
    if p == 1.0:
        loss_val = 0.0
        interp = "perfecto, sin castigo"
    elif p >= 0.9:
        loss_val = -math.log(p)
        interp = "muy bien"
    elif p >= 0.5:
        loss_val = -math.log(p)
        interp = "ok pero mejorable"
    elif p >= 0.1:
        loss_val = -math.log(p)
        interp = "mal"
    elif p >= 0.001:
        loss_val = -math.log(p)
        interp = "muy mal"
    else:
        loss_val = -math.log(p)
        interp = "catastrofico"
    print(f"{p:>12.4f}  {loss_val:>15.4f}  {interp}")

print("\nPropiedades clave de la curva:")
print("  - Cuando P = 1.0 (perfecto), loss = 0 (sin castigo).")
print("  - Cuando P -> 0 (terrible), loss -> infinito.")
print("  - El crecimiento NO es lineal: cae rapido al inicio y se dispara")
print("    al acercarse a 0. Eso hace que el modelo aprenda fuerte cuando")
print("    esta muy equivocado, y suavemente cuando ya esta cerca.")


# ---------------------------------------------------------------------------
# Comparacion -log(P) vs (1 - P) para ver por que -log es mejor
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 5: Por que -log(P) y no (1 - P)?")
print("=" * 70)

print(f"\n{'P':>8}  {'1 - P':>10}  {'-log(P)':>10}")
print("-" * 35)
for p in [0.9, 0.5, 0.1, 0.01, 0.001]:
    print(f"{p:>8.4f}  {1-p:>10.4f}  {-math.log(p):>10.4f}")

print("""
Comparacion:
  - Modelo con P=0.1 (malo) vs P=0.001 (terrible, 100x peor):
      Con (1-P):     loss = 0.9000 vs 0.9990 -> casi identicos
      Con -log(P):   loss = 2.3026 vs 6.9078 -> 3x mas castigo

  Con -log el modelo "ve" la diferencia entre malo y terrible. Esa
  diferencia se traduce en gradientes mas fuertes, y por lo tanto
  ajustes mas grandes a los pesos. Por eso cross-entropy es la
  funcion de perdida estandar en clasificacion.
""")

print("=" * 70)
print("FIN demo cross-entropy.")
print("=" * 70)
