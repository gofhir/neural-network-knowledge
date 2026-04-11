"""
Ejemplo 9 — Red neuronal con LayerNorm (PyTorch)

Objetivo: ver como se usa LayerNorm dentro de una red.
          LayerNorm es el que se usa en Transformers (GPT, BERT, etc.)
          A diferencia de BatchNorm, no cambia entre train/eval.

Ejecutar:
  docker run --rm clase7-pytorch python -u ejemplos/pytorch/09_red_con_layernorm_pytorch.py
"""
import torch
import torch.nn as nn


class NetworkWith_LN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.ln1 = nn.LayerNorm(64)      # normaliza los 64 features
        self.fc2 = nn.Linear(64, 32)
        self.ln2 = nn.LayerNorm(32)      # normaliza los 32 features
        self.fc3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Capa 1: Linear -> LayerNorm -> ReLU
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.relu(x)

        # Capa 2: Linear -> LayerNorm -> ReLU
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.relu(x)

        # Salida (sin normalizar)
        x = self.fc3(x)
        return x


model = NetworkWith_LN()

# =============================================
# Paso 1: Forward pass
# =============================================
print("=" * 55)
print("Paso 1: Forward pass con LayerNorm")
print("=" * 55)

torch.manual_seed(42)
x = torch.randn(8, 10) * 5  # 8 muestras, 10 features

print(f"\nEntrada: {x.shape}")

# Pasar capa por capa
h = x
for name, layer in [("fc1", model.fc1), ("ln1", model.ln1), ("relu", model.relu),
                     ("fc2", model.fc2), ("ln2", model.ln2), ("relu", model.relu),
                     ("fc3", model.fc3)]:
    h = layer(h)
    marker = "  <- normalizado!" if "ln" in name else ""
    print(f"  Despues de {name}: media={h.mean().item():+8.4f}, std={h.std().item():.4f}{marker}")

# =============================================
# Paso 2: LayerNorm NO cambia entre train/eval
# =============================================
print(f"\n{'='*55}")
print("Paso 2: train vs eval (no cambia)")
print("=" * 55)

sample = torch.randn(1, 10)

model.train()
output_train = model(sample)

model.eval()
output_eval = model(sample)

print(f"\n  .train() salida: {[round(v, 4) for v in output_train.squeeze().tolist()]}")
print(f"  .eval()  salida: {[round(v, 4) for v in output_eval.squeeze().tolist()]}")
print(f"  Son iguales? {torch.allclose(output_train, output_eval)}")
print(f"\n  -> Con LayerNorm, .train() y .eval() dan el MISMO resultado.")
print(f"     No hay running stats que cambien el comportamiento.")

# =============================================
# Paso 3: Funciona con batch_size=1
# =============================================
print(f"\n{'='*55}")
print("Paso 3: Funciona con batch_size=1")
print("=" * 55)

single_sample = torch.randn(1, 10)  # UNA sola muestra

# LayerNorm funciona bien
model_ln = NetworkWith_LN()
output_ln = model_ln(single_sample)
print(f"\n  LayerNorm con batch=1: {[round(v, 4) for v in output_ln.squeeze().tolist()]}")
print(f"  -> Funciona perfecto!")

# BatchNorm falla con batch=1
print(f"\n  BatchNorm con batch=1:")
try:
    bn_layer = nn.BatchNorm1d(10)
    bn_layer(single_sample)
    print(f"  -> Funciono (raro)")
except Exception as e:
    print(f"  -> ERROR: {e}")
    print(f"     BatchNorm necesita al menos 2 muestras para calcular estadisticas")

# =============================================
# Paso 4: Comparar parametros
# =============================================
print(f"\n{'='*55}")
print("Paso 4: Parametros de LayerNorm")
print("=" * 55)

print(f"\n  ln1 (64 features):")
print(f"    gamma (weight): {model.ln1.weight.shape} = {model.ln1.weight.numel()} parametros")
print(f"    beta  (bias):   {model.ln1.bias.shape} = {model.ln1.bias.numel()} parametros")
print(f"    NO tiene running_mean ni running_var")
print(f"")
print(f"  -> Mismos parametros que BatchNorm (gamma y beta por feature)")
print(f"     pero sin running stats. Mas simple.")

# =============================================
# Resumen
# =============================================
print(f"\n{'='*55}")
print("Resumen: cuando usar LayerNorm vs BatchNorm")
print("=" * 55)
print(f"""
  LayerNorm:
    - Transformers, RNNs, secuencias (NLP)
    - Cuando el batch puede ser 1
    - Cuando quieres que train y eval sean iguales
    - Orden: Linear -> LayerNorm -> ReLU

  BatchNorm:
    - CNNs, imagenes
    - Cuando tienes batches grandes (>= 32)
    - Orden: Linear/Conv -> BatchNorm -> ReLU
""")
