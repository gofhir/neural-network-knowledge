"""
Ejemplo 5 — BatchNorm: entrenamiento vs inferencia (PyTorch)

Objetivo: entender que BatchNorm se comporta DISTINTO en entrenamiento
          y en inferencia. En entrenamiento usa las stats del batch actual.
          En inferencia usa las stats ACUMULADAS (running mean/var).

Ejecutar:
  docker run --rm clase7-pytorch python -u ejemplos/05_batchnorm_train_vs_eval_pytorch.py
"""
import torch
import torch.nn as nn

bn = nn.BatchNorm1d(num_features=1)

# =============================================
# Paso 1: Entrenamiento — pasar varios batches
# =============================================
print("--- Paso 1: ENTRENAMIENTO (model.train) ---")
print("Pasamos 3 batches. En cada uno, BN:")
print("  - Normaliza usando la media/var del batch actual")
print("  - Actualiza el running_mean y running_var acumulados\n")

bn.train()  # Activar modo entrenamiento

batches = [
    torch.tensor([[10.0], [12.0], [8.0], [14.0]]),   # batch 1: media=11, var=5.33
    torch.tensor([[20.0], [22.0], [18.0], [24.0]]),   # batch 2: media=21, var=5.33
    torch.tensor([[15.0], [17.0], [13.0], [19.0]]),   # batch 3: media=16, var=5.33
]

for i, batch in enumerate(batches):
    batch_mean = batch.mean().item()
    batch_var = batch.var(correction=0).item()

    result = bn(batch)

    print(f"  Batch {i+1}: valores={batch.squeeze().tolist()}")
    print(f"    Media del batch:     {batch_mean:.2f}")
    print(f"    Var del batch:       {batch_var:.2f}")
    print(f"    running_mean:        {bn.running_mean.item():.4f}")
    print(f"    running_var:         {bn.running_var.item():.4f}")
    print(f"    Salida normalizada:  {[round(v, 4) for v in result.squeeze().tolist()]}")
    print()

# =============================================
# Paso 2: Inferencia — una sola muestra
# =============================================
print("--- Paso 2: INFERENCIA (model.eval) ---")
print("Ahora usamos el modelo para predecir.")
print("Solo tenemos UNA muestra, no hay batch.\n")

bn.eval()  # Activar modo inferencia

sample = torch.tensor([[16.0]])
result = bn(sample)

running_mean = bn.running_mean.item()
running_var = bn.running_var.item()

print(f"  Muestra: {sample.item()}")
print(f"  Running mean (acumulada): {running_mean:.4f}")
print(f"  Running var (acumulada):  {running_var:.4f}")
print(f"  Resultado: {result.item():.4f}")
print(f"  Calculo: ({sample.item()} - {running_mean:.4f}) / sqrt({running_var:.4f}) = {result.item():.4f}")

# =============================================
# Paso 3: Que pasa si OLVIDAS model.eval()?
# =============================================
print(f"\n--- Paso 3: Que pasa si OLVIDAS .eval()? ---")

bn.train()  # "olvidamos" poner eval

# La misma muestra, pero 3 veces en batches distintos
sample1 = torch.tensor([[16.0], [16.0]])  # batch de 2 iguales
sample2 = torch.tensor([[16.0], [100.0]])  # batch con un valor extremo

result1 = bn(sample1)
result2 = bn(sample2)

print(f"  Con .train() activado:")
print(f"    Batch [16, 16]:  resultado para 16 = {result1[0].item():.4f}")
print(f"    Batch [16, 100]: resultado para 16 = {result2[0].item():.4f}")
print(f"  -> El MISMO valor (16) da resultados DISTINTOS!")
print(f"     Porque en .train(), BN depende de QUE OTRAS muestras hay en el batch.")
print(f"")
print(f"  LECCION: siempre llamar model.eval() antes de inferencia.")
