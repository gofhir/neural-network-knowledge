"""
Ejemplo 1 — MSE vs Cross-Entropy: cuando usar cual

Objetivo: ver con numeros reales por que MSE no funciona bien
          para clasificacion, y por que Cross-Entropy si.

Ejecutar:
  docker run --rm clase8 python -u ejemplos/pytorch/01_mse_vs_crossentropy.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SEPARATOR = "=" * 60

# =============================================
# 1. MSE para REGRESION (funciona bien)
# =============================================
print(SEPARATOR)
print("1. MSE para REGRESION (funciona bien)")
print(SEPARATOR)

print("""
  MSE mide que tan LEJOS esta la prediccion del valor real.
  Perfecto para predecir numeros continuos.
""")

# Predecir precios de casas
predictions = torch.tensor([200000.0, 350000.0, 150000.0])
real_values = torch.tensor([210000.0, 340000.0, 180000.0])

loss_fn = nn.MSELoss()
loss = loss_fn(predictions, real_values)

print(f"  Predicciones: {predictions.tolist()}")
print(f"  Valores reales: {real_values.tolist()}")
print(f"  Diferencias: {(predictions - real_values).tolist()}")
print(f"  MSE Loss: {loss.item():,.0f}")
print(f"")
print(f"  Calculo manual:")
for i in range(3):
    diff = predictions[i] - real_values[i]
    print(f"    Casa {i+1}: ({predictions[i]:,.0f} - {real_values[i]:,.0f})² = {diff**2:,.0f}")
mse_manual = ((predictions - real_values) ** 2).mean()
print(f"    Promedio: {mse_manual.item():,.0f}")

# =============================================
# 2. MSE para CLASIFICACION (funciona MAL)
# =============================================
print(f"\n{SEPARATOR}")
print("2. MSE para CLASIFICACION (funciona MAL)")
print(SEPARATOR)

print("""
  Si usamos MSE para clasificar digitos (0-9), el modelo
  trata las clases como NUMEROS con orden y distancia.
  Pero "gato" no esta "entre" "perro" y "pajaro".
""")

# Modelo que siempre predice "0"
always_0 = torch.zeros(10)  # predice 0 para las 10 clases
real_classes = torch.arange(10, dtype=torch.float32)  # clases 0-9

mse_predict_0 = ((always_0 - real_classes) ** 2).mean()

# Modelo que siempre predice "5" (la del medio)
always_5 = torch.full((10,), 5.0)
mse_predict_5 = ((always_5 - real_classes) ** 2).mean()

print(f"  Si el modelo SIEMPRE predice clase 0:")
print(f"    Errores²: {((always_0 - real_classes) ** 2).tolist()}")
print(f"    MSE = {mse_predict_0.item():.1f}")
print(f"")
print(f"  Si el modelo SIEMPRE predice clase 5:")
print(f"    Errores²: {((always_5 - real_classes) ** 2).tolist()}")
print(f"    MSE = {mse_predict_5.item():.1f}  ← MENOR!")
print(f"")
print(f"  MSE prefiere predecir 5 (el valor del MEDIO)")
print(f"  porque esta mas 'cerca' de todo numericamente.")
print(f"  Pero eso no tiene sentido: la clase 5 no es")
print(f"  'mejor' que la clase 0 para todos los datos!")

# =============================================
# 3. Cross-Entropy para CLASIFICACION (funciona bien)
# =============================================
print(f"\n{SEPARATOR}")
print("3. Cross-Entropy para CLASIFICACION (funciona bien)")
print(SEPARATOR)

print("""
  Cross-Entropy mide que tanta PROBABILIDAD le da el modelo
  a la clase CORRECTA. No le importa el "orden" de las clases.
""")

# Ejemplo: la clase correcta es la 3
real_label = torch.tensor([3])

# Modelo que predice bien (alta probabilidad en clase 3)
logits_good = torch.tensor([[0.1, 0.2, 0.1, 8.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
probs_good = F.softmax(logits_good, dim=1)

# Modelo que predice mal (alta probabilidad en clase 7)
logits_bad = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 8.5, 0.1, 0.1]])
probs_bad = F.softmax(logits_bad, dim=1)

loss_fn_ce = nn.CrossEntropyLoss()
loss_good = loss_fn_ce(logits_good, real_label)
loss_bad = loss_fn_ce(logits_bad, real_label)

print(f"  Clase correcta: {real_label.item()}")
print(f"")
print(f"  Modelo bueno (confia en clase 3):")
print(f"    Prob clase 3: {probs_good[0, 3].item():.4f}")
print(f"    CrossEntropy: {loss_good.item():.4f}  ← BAJO (bien!)")
print(f"")
print(f"  Modelo malo (confia en clase 7):")
print(f"    Prob clase 3: {probs_bad[0, 3].item():.4f}")
print(f"    CrossEntropy: {loss_bad.item():.4f}  ← ALTO (mal!)")
print(f"")
print(f"  A Cross-Entropy no le importa que la clase 7 esta")
print(f"  'cerca' de la 3 numericamente. Solo le importa")
print(f"  cuanta probabilidad le dio a la clase CORRECTA.")

# =============================================
# 4. Softmax: de logits a probabilidades
# =============================================
print(f"\n{SEPARATOR}")
print("4. Softmax: de logits a probabilidades")
print(SEPARATOR)

logits = torch.tensor([-58.0, 18.3, 0.008, 0.935, -0.156, -88.72, 0.01, 10.24, 3.333, 2.5])

print(f"  Logits (salida cruda de la red):")
for i, v in enumerate(logits):
    print(f"    Clase {i}: {v:8.3f}")

probs = F.softmax(logits, dim=0)

print(f"\n  Despues de Softmax (probabilidades):")
for i, v in enumerate(probs):
    bar = "#" * int(v * 50)
    print(f"    Clase {i}: {v:.6f}  {bar}")

print(f"\n  Suma: {probs.sum().item():.4f} (siempre suma 1.0)")
print(f"\n  Softmax convierte numeros crudos en probabilidades.")
print(f"  El valor mas alto (18.3) concentra casi toda la probabilidad.")
print(f"  nn.CrossEntropyLoss hace Softmax + Cross-Entropy internamente.")

# =============================================
# Resumen
# =============================================
print(f"\n{SEPARATOR}")
print("RESUMEN")
print(SEPARATOR)
print(f"""
  Tarea                     Loss Function               PyTorch
  ─────                     ─────────────               ───────
  Clasificar en N clases    CrossEntropyLoss             nn.CrossEntropyLoss()
  Si o no (binario)         Binary Cross-Entropy         nn.BCEWithLogitsLoss()
  Predecir un numero        MSE                          nn.MSELoss()
  Predecir numero robusto   L1 / MAE                     nn.L1Loss()

  NUNCA usar MSE para clasificacion.
  NUNCA usar Cross-Entropy para regresion.
""")
