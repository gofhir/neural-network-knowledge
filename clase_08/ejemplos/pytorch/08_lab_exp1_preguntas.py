"""
Lab Experimento 1 — Preguntas: ¿MSE o Cross-Entropy?

Analiza las 6 preguntas del laboratorio con ejemplos concretos.

Ejecutar:
  docker run --rm clase8 python -u ejemplos_c8/pytorch/08_lab_exp1_preguntas.py
"""
import torch
import torch.nn as nn

SEPARATOR = "=" * 60

print(SEPARATOR)
print("PREGUNTAS: ¿MSE o Cross-Entropy?")
print(SEPARATOR)

print("""
  Regla de decision:

  ┌─────────────────────────────────────────────────────┐
  │ ¿Que tipo de valor predice el modelo?               │
  │                                                     │
  │ Un NUMERO continuo    → MSE (nn.MSELoss)            │
  │   (precio, temp, score, coordenadas)                │
  │                                                     │
  │ Una CATEGORIA         → CrossEntropy                │
  │   (gato/perro, digito 0-9, tipo)                    │
  │                                                     │
  │ Una PROBABILIDAD      → BCE (nn.BCEWithLogitsLoss)  │
  │   (¿compra? ¿enfermo?)                              │
  └─────────────────────────────────────────────────────┘
""")

# =============================================
# Pregunta 1
# =============================================
print(SEPARATOR)
print("PREGUNTA 1")
print(SEPARATOR)
print("""
  "Un sistema que estima la PROBABILIDAD de que un cliente
   compre pasajes en avion dentro de los proximos 30 dias"

  Respuesta: Cross-Entropy (BCE)

  ¿Por que?
    - El output es una probabilidad de un evento BINARIO
    - Compra (1) o no compra (0)
    - BCE esta diseñada para producir probabilidades calibradas
""")

# Demostrar con codigo
model = nn.Sequential(nn.Linear(10, 1))  # features del cliente → 1 salida
loss_bce = nn.BCEWithLogitsLoss()

# Simular: cliente que SI compra
features = torch.randn(1, 10)
label = torch.tensor([1.0])  # si compra
pred = model(features)
loss = loss_bce(pred.squeeze(0), label)
prob = torch.sigmoid(pred).item()
print(f"  Ejemplo:")
print(f"    Prediccion (logit): {pred.item():.4f}")
print(f"    Probabilidad:       {prob:.1%}")
print(f"    Label real:         compra (1)")
print(f"    BCE Loss:           {loss.item():.4f}")

# =============================================
# Pregunta 2
# =============================================
print(f"\n{SEPARATOR}")
print("PREGUNTA 2")
print(SEPARATOR)
print("""
  "Un sistema que predice un score CONTINUO entre -1 y 1
   de que tan positivo o negativo es un comentario"

  Respuesta: MSE

  ¿Por que?
    - El output es un VALOR CONTINUO (no una categoria)
    - -1.0 = muy negativo, 0.0 = neutro, +1.0 = muy positivo
    - Hay un ORDEN y DISTANCIA entre los valores
    - Predecir 0.8 cuando es 0.9 es MEJOR que predecir -0.5
""")

model = nn.Sequential(nn.Linear(10, 1), nn.Tanh())  # Tanh produce valores [-1, 1]
loss_mse = nn.MSELoss()

features = torch.randn(1, 10)
label = torch.tensor([0.8])  # comentario positivo
pred = model(features).squeeze()
loss = loss_mse(pred, label)
print(f"  Ejemplo:")
print(f"    Score predicho:   {pred.item():.4f}")
print(f"    Score real:       {label.item():.1f} (positivo)")
print(f"    MSE Loss:         {loss.item():.4f}")
print(f"    Error:            {abs(pred.item() - label.item()):.4f}")
print(f"""
  NOTA: si el problema fuera clasificar en 3 clases
  (positivo / neutro / negativo), se usaria Cross-Entropy.
  La diferencia es CONTINUO (MSE) vs CATEGORICO (CE).
""")

# =============================================
# Pregunta 3
# =============================================
print(SEPARATOR)
print("PREGUNTA 3")
print(SEPARATOR)
print("""
  "Un sistema que estima el PRECIO del Dolar
   a partir de tweets de Donald Trump"

  Respuesta: MSE

  ¿Por que?
    - El precio es un VALOR CONTINUO ($800, $823.50, $850...)
    - Hay orden y distancia: $823 esta mas cerca de $825 que de $900
    - MSE mide exactamente la distancia al valor real
""")

model = nn.Sequential(nn.Linear(10, 1))
features = torch.randn(1, 10)
label = torch.tensor([825.0])  # precio real
pred = model(features).squeeze()
loss = loss_mse(pred, label)
print(f"  Ejemplo:")
print(f"    Precio predicho:  ${pred.item():.2f}")
print(f"    Precio real:      ${label.item():.2f}")
print(f"    MSE Loss:         {loss.item():.2f}")

# =============================================
# Pregunta 4
# =============================================
print(f"\n{SEPARATOR}")
print("PREGUNTA 4")
print(SEPARATOR)
print("""
  "Un sistema que predice la CANTIDAD de alumnos
   que asistiran (entero entre 1 y 40)"

  Respuesta: MSE

  ¿Por que?
    - Aunque es DISCRETO (entero), tiene ORDEN numerico
    - 15 alumnos es MAS que 10 y MENOS que 20
    - Predecir 14 cuando son 15 es MEJOR que predecir 5
""")

# Comparar MSE vs CrossEntropy para este caso
real = 15

print(f"  ¿Que pasa si el real es {real} alumnos?\n")
print(f"  Con MSE (respeta distancia):")
for pred_val in [14, 10, 5, 30]:
    error = (pred_val - real) ** 2
    print(f"    Predecir {pred_val:2d}: MSE = ({pred_val}-{real})² = {error:4d}  "
          f"{'← chico, casi acerto!' if error < 5 else ''}")

print(f"\n  Con Cross-Entropy (40 clases, no respeta distancia):")
print(f"    Predecir 14: tan 'incorrecto' como predecir 1")
print(f"    → NO aprovecha que 14 esta CERCA de 15")

# =============================================
# Pregunta 5
# =============================================
print(f"\n{SEPARATOR}")
print("PREGUNTA 5")
print(SEPARATOR)
print("""
  "Predecir para cada pixel el color RGB
   (3 valores entre 0 y 255)"

  Respuesta: MSE

  ¿Por que?
    - Cada canal (R, G, B) es un valor CONTINUO de 0 a 255
    - Rojo=200 es mas rojo que Rojo=100 (hay orden)
    - Si el real es R=200 y predices R=195, es CASI correcto
""")

# Simular prediccion de 1 pixel
pred_rgb = torch.tensor([195.0, 102.0, 48.0])  # prediccion
real_rgb = torch.tensor([200.0, 100.0, 50.0])  # real

loss = nn.MSELoss()(pred_rgb, real_rgb)
print(f"  Pixel predicho: R={pred_rgb[0]:.0f}, G={pred_rgb[1]:.0f}, B={pred_rgb[2]:.0f}")
print(f"  Pixel real:     R={real_rgb[0]:.0f}, G={real_rgb[1]:.0f}, B={real_rgb[2]:.0f}")
print(f"  MSE Loss: {loss.item():.2f}")
print(f"  → Error chico, los colores son MUY parecidos visualmente")

# =============================================
# Pregunta 6
# =============================================
print(f"\n{SEPARATOR}")
print("PREGUNTA 6")
print(SEPARATOR)
print("""
  "Aproximar una funcion continua altamente no lineal"

  Respuesta: MSE

  ¿Por que?
    - El output es un valor continuo: f(x) → y
    - MSE mide que tan lejos esta f_predicha(x) de f_real(x)
    - Es el caso clasico de REGRESION
""")

# Simular: aproximar f(x) = sin(x)
x = torch.linspace(0, 6.28, 20)
y_real = torch.sin(x)

# Red simple
model = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(500):
    pred = model(x.unsqueeze(1)).squeeze()
    loss = nn.MSELoss()(pred, y_real)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

pred_final = model(x.unsqueeze(1)).squeeze().detach()
print(f"  Aproximando f(x) = sin(x):")
print(f"  MSE final: {loss.item():.6f}")
print(f"\n  {'x':>6s}  {'sin(x)':>8s}  {'pred':>8s}  {'error':>8s}")
for i in range(0, 20, 4):
    err = abs(y_real[i].item() - pred_final[i].item())
    print(f"  {x[i].item():6.2f}  {y_real[i].item():+8.4f}  {pred_final[i].item():+8.4f}  {err:8.4f}")

# =============================================
# Resumen
# =============================================
print(f"\n{SEPARATOR}")
print("RESUMEN DE LAS 6 PREGUNTAS")
print(SEPARATOR)
print(f"""
  Pregunta  Respuesta         Razon
  ────────  ─────────         ─────
  1         BCE               Probabilidad de evento binario
  2         MSE               Score continuo [-1, 1]
  3         MSE               Valor continuo (precio)
  4         MSE               Entero con orden y distancia
  5         MSE               Valores continuos (RGB 0-255)
  6         MSE               Regresion de funcion continua

  5 de 6 usan MSE. Solo la primera usa BCE.
  → MSE es para NUMEROS (continuos o discretos con orden).
  → Cross-Entropy es para CATEGORIAS (sin orden numerico).
""")
