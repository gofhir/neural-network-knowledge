"""
Ejemplo 14 — Entrenar una red neuronal PASO A PASO (MNIST)

Objetivo: entrenar una red que aprenda a reconocer digitos escritos a mano.
          Cada paso esta explicado en detalle para entender que pasa.

Genera graficos en /app/output/:
  - 12_arquitectura_red.png
  - 13_entrenamiento_loss.png
  - 14_predicciones_correctas.png
  - 15_predicciones_incorrectas.png
  - 16_confusion_matrix.png

Ejecutar:
  docker run --rm -v $(pwd)/output:/app/output clase7-pytorch \
    python -u ejemplos/transformaciones/14_entrenar_mnist_paso_a_paso.py
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

OUTPUT = "/app/output"

# ╔══════════════════════════════════════════════════════════╗
# ║  PASO 1: PREPARAR LOS DATOS                            ║
# ╚══════════════════════════════════════════════════════════╝
print("=" * 60)
print("PASO 1: PREPARAR LOS DATOS")
print("=" * 60)

print("""
  Necesitamos dos conjuntos de datos:

  - ENTRENAMIENTO (train): la red aprende con estos datos.
    Son como los ejercicios de practica para una prueba.

  - PRUEBA (test): evaluamos si la red aprendio de verdad.
    Son como la prueba final. La red NUNCA los vio antes.
""")

# transforms.ToTensor() hace dos cosas:
# 1. Convierte la imagen PIL a tensor
# 2. Normaliza los pixeles de [0, 255] a [0.0, 1.0]
transform = transforms.ToTensor()

# Descargar MNIST
print("  Descargando MNIST...")
train_dataset = datasets.MNIST(root='/tmp/data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='/tmp/data', train=False, download=True, transform=transform)

print(f"\n  Datos de entrenamiento: {len(train_dataset)} imagenes")
print(f"  Datos de prueba:        {len(test_dataset)} imagenes")

# Veamos una imagen
image, label = train_dataset[0]
print(f"\n  Una imagen:")
print(f"    Shape:    {image.shape}  → (1 canal, 28 alto, 28 ancho)")
print(f"    Valores:  {image.min():.2f} a {image.max():.2f}  → ya normalizado [0, 1]")
print(f"    Etiqueta: {label}  → es el digito '{label}'")

# DataLoader: agrupa las imagenes en BATCHES
# La red no procesa de a 1 imagen, procesa de a 64 juntas (mas rapido)
batch_size = 64

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
    # shuffle=True: mezcla las imagenes en cada epoca
    # para que la red no memorice el orden
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)

print(f"\n  Batch size: {batch_size}")
print(f"  Batches de entrenamiento: {len(train_loader)}")
print(f"    → {len(train_dataset)} imagenes / {batch_size} por batch ≈ {len(train_loader)} batches")

# Veamos un batch
batch_images, batch_labels = next(iter(train_loader))
print(f"\n  Un batch:")
print(f"    Imagenes: {batch_images.shape}  → 64 imagenes de 1x28x28")
print(f"    Labels:   {batch_labels.shape}  → 64 etiquetas")
print(f"    Labels:   {batch_labels[:10].tolist()}...  → los primeros 10 digitos")


# ╔══════════════════════════════════════════════════════════╗
# ║  PASO 2: CONSTRUIR LA RED NEURONAL                     ║
# ╚══════════════════════════════════════════════════════════╝
print(f"\n{'=' * 60}")
print("PASO 2: CONSTRUIR LA RED NEURONAL")
print("=" * 60)

print("""
  La red transforma pixeles en predicciones:

  Entrada: imagen 28x28 = 784 pixeles
     ↓
  Flatten: aplana la imagen a un vector de 784 numeros
     ↓
  Linear(784 → 256): primera capa, reduce de 784 a 256
     ↓
  BatchNorm(256): normaliza las 256 activaciones
     ↓
  ReLU: activacion (convierte negativos en 0)
     ↓
  Dropout(0.3): apaga 30% de neuronas (regularizacion)
     ↓
  Linear(256 → 128): segunda capa, reduce de 256 a 128
     ↓
  BatchNorm(128): normaliza las 128 activaciones
     ↓
  ReLU: activacion
     ↓
  Dropout(0.3): apaga 30% de neuronas
     ↓
  Linear(128 → 10): capa final, 10 salidas (una por digito)
     ↓
  Salida: 10 numeros (el mas alto es la prediccion)
""")


class DigitRecognizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # Aplanar imagen 1x28x28 → 784
            nn.Flatten(),

            # Capa 1: 784 → 256
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),    # normalizar (de la clase 7!)
            nn.ReLU(),              # activacion (de la clase 7!)
            nn.Dropout(0.3),        # regularizacion (de la clase 7!)

            # Capa 2: 256 → 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Capa de salida: 128 → 10
            # NO ponemos BatchNorm, ReLU ni Dropout aqui
            # porque queremos los valores "crudos" para la prediccion
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.network(x)


model = DigitRecognizer()

# Contar parametros
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"  Modelo creado:")
print(f"    Parametros totales:    {total_params:,}")
print(f"    Parametros trainable:  {trainable_params:,}")
print(f"    (son los numeros que la red va a APRENDER)")

# Probar que funciona con un batch
test_output = model(batch_images)
print(f"\n  Prueba rapida:")
print(f"    Entrada:  {batch_images.shape}")
print(f"    Salida:   {test_output.shape}  → 64 predicciones, 10 clases cada una")
print(f"    Ejemplo:  {[round(v, 2) for v in test_output[0].tolist()]}")
print(f"    → Son numeros ALEATORIOS porque la red no ha entrenado!")

# Guardar grafico de la arquitectura
fig, ax = plt.subplots(figsize=(8, 10))
ax.axis('off')

layers_info = [
    ("Imagen\n28×28×1", "lightcyan", "784 valores"),
    ("Flatten", "lightyellow", "784 → 784"),
    ("Linear + BatchNorm\n+ ReLU + Dropout", "lightblue", "784 → 256"),
    ("Linear + BatchNorm\n+ ReLU + Dropout", "lightblue", "256 → 128"),
    ("Linear\n(salida)", "lightsalmon", "128 → 10"),
    ("Prediccion\n10 clases", "lightgreen", "dígitos 0-9"),
]

for i, (name, color, desc) in enumerate(layers_info):
    y = 0.85 - i * 0.15
    rect = plt.Rectangle((0.15, y - 0.04), 0.5, 0.08, facecolor=color,
                          edgecolor='black', linewidth=1.5, transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(0.4, y, name, transform=ax.transAxes, ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(0.72, y, desc, transform=ax.transAxes, ha='left', va='center',
            fontsize=10, color='gray')
    if i < len(layers_info) - 1:
        ax.annotate('', xy=(0.4, y - 0.05), xytext=(0.4, y - 0.09),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

ax.set_title("Arquitectura de la red\nDigitRecognizer", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT}/12_arquitectura_red.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Guardado: {OUTPUT}/12_arquitectura_red.png")


# ╔══════════════════════════════════════════════════════════╗
# ║  PASO 3: DEFINIR COMO MEDIR EL ERROR                   ║
# ╚══════════════════════════════════════════════════════════╝
print(f"\n{'=' * 60}")
print("PASO 3: DEFINIR COMO MEDIR EL ERROR (loss function)")
print("=" * 60)

print("""
  Necesitamos una funcion que diga "que tan MAL esta la prediccion".
  Se llama LOSS FUNCTION (funcion de perdida).

  Para clasificacion se usa CrossEntropyLoss:
    - Si la red predice bien → loss BAJO (cercano a 0)
    - Si la red predice mal  → loss ALTO

  Ejemplo:
    Etiqueta real: 3
    Red predice:   [0.1, 0.1, 0.1, 0.9, 0.1, ...]  → dice "3" → loss bajo ✓
    Red predice:   [0.1, 0.1, 0.1, 0.1, 0.9, ...]  → dice "4" → loss alto ✗
""")

loss_fn = nn.CrossEntropyLoss()

# Ejemplo con nuestro batch
outputs = model(batch_images)
loss = loss_fn(outputs, batch_labels)
print(f"  loss = CrossEntropyLoss(prediccion, etiqueta_real)")
print(f"  loss actual = {loss.item():.4f}  (alto, porque la red no ha aprendido)")


# ╔══════════════════════════════════════════════════════════╗
# ║  PASO 4: DEFINIR COMO ACTUALIZAR LOS PESOS             ║
# ╚══════════════════════════════════════════════════════════╝
print(f"\n{'=' * 60}")
print("PASO 4: DEFINIR EL OPTIMIZADOR")
print("=" * 60)

print("""
  El optimizador decide COMO ajustar los pesos para reducir el error.

  Adam es el optimizador mas popular. Necesita un parametro:
    - learning_rate (lr): que tan grandes son los ajustes
      - Muy alto (0.1):  ajustes grandes → se pasa de largo, no converge
      - Muy bajo (0.00001): ajustes chicos → aprende muy lento
      - Justo (0.001):  el default de Adam, funciona bien casi siempre
""")

learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print(f"  Optimizador: Adam")
print(f"  Learning rate: {learning_rate}")
print(f"  Parametros a optimizar: {trainable_params:,}")


# ╔══════════════════════════════════════════════════════════╗
# ║  PASO 5: ENTRENAR!                                     ║
# ╚══════════════════════════════════════════════════════════╝
print(f"\n{'=' * 60}")
print("PASO 5: ENTRENAR (el loop de entrenamiento)")
print("=" * 60)

print("""
  Entrenar = repetir estos 4 sub-pasos miles de veces:

  Para cada batch de imagenes:
    1. FORWARD:  pasar imagenes por la red → obtener prediccion
    2. LOSS:     comparar prediccion con la respuesta correcta
    3. BACKWARD: calcular como ajustar cada peso (gradientes)
    4. UPDATE:   aplicar los ajustes a los pesos

  Una EPOCA = pasar por TODAS las imagenes una vez.
  Normalmente se entrena varias epocas.
""")

num_epochs = 5
history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

for epoch in range(num_epochs):
    # ─── ENTRENAMIENTO ───
    model.train()  # activa Dropout y BatchNorm en modo train
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):

        # Sub-paso 1: FORWARD (pasar datos por la red)
        predictions = model(images)

        # Sub-paso 2: LOSS (medir el error)
        loss = loss_fn(predictions, labels)

        # Sub-paso 3: BACKWARD (calcular gradientes)
        optimizer.zero_grad()  # limpiar gradientes anteriores
        loss.backward()        # calcular gradientes nuevos

        # Sub-paso 4: UPDATE (ajustar pesos)
        optimizer.step()       # aplicar los gradientes

        # Registrar estadisticas
        running_loss += loss.item()
        _, predicted = predictions.max(1)  # el indice del valor mas alto
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Mostrar progreso cada 200 batches
        if batch_idx % 200 == 0:
            print(f"    Epoca {epoch+1}/{num_epochs}, "
                  f"Batch {batch_idx}/{len(train_loader)}, "
                  f"Loss: {loss.item():.4f}")

    train_loss = running_loss / len(train_loader)
    train_acc = 100.0 * correct / total

    # ─── EVALUACION (en datos de prueba) ───
    model.eval()  # desactiva Dropout, BatchNorm usa running stats
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # no necesitamos gradientes para evaluar
        for images, labels in test_loader:
            predictions = model(images)
            loss = loss_fn(predictions, labels)
            test_loss += loss.item()
            _, predicted = predictions.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = test_loss / len(test_loader)
    test_acc = 100.0 * correct / total

    # Guardar en historial
    history['train_loss'].append(train_loss)
    history['test_loss'].append(test_loss)
    history['train_acc'].append(train_acc)
    history['test_acc'].append(test_acc)

    print(f"\n  ► Epoca {epoch+1}/{num_epochs} completada:")
    print(f"    Train Loss: {train_loss:.4f}  |  Train Accuracy: {train_acc:.1f}%")
    print(f"    Test Loss:  {test_loss:.4f}  |  Test Accuracy:  {test_acc:.1f}%")
    print()

# Grafico de entrenamiento
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

epochs_range = range(1, num_epochs + 1)

ax1.plot(epochs_range, history['train_loss'], 'o-', label='Train Loss', color='steelblue')
ax1.plot(epochs_range, history['test_loss'], 'o-', label='Test Loss', color='coral')
ax1.set_xlabel('Epoca')
ax1.set_ylabel('Loss (error)')
ax1.set_title('Loss por epoca\n(mas bajo = mejor)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_range, history['train_acc'], 'o-', label='Train Accuracy', color='steelblue')
ax2.plot(epochs_range, history['test_acc'], 'o-', label='Test Accuracy', color='coral')
ax2.set_xlabel('Epoca')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Precision por epoca\n(mas alto = mejor)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(90, 100)

plt.tight_layout()
plt.savefig(f"{OUTPUT}/13_entrenamiento_loss.png", dpi=150)
plt.close()
print(f"  Guardado: {OUTPUT}/13_entrenamiento_loss.png")


# ╔══════════════════════════════════════════════════════════╗
# ║  PASO 6: USAR LA RED ENTRENADA (PREDECIR)              ║
# ╚══════════════════════════════════════════════════════════╝
print(f"\n{'=' * 60}")
print("PASO 6: USAR LA RED PARA PREDECIR")
print("=" * 60)

print("""
  La red ya entreno. Ahora le pasamos imagenes que NUNCA VIO
  y vemos si las clasifica correctamente.
""")

model.eval()  # CRITICO: poner en modo evaluacion

# Tomar imagenes del test set (nunca vistas)
test_images = []
test_labels = []
test_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        predictions = model(images)
        _, predicted = predictions.max(1)
        # Obtener probabilidades con softmax
        probs = torch.softmax(predictions, dim=1)

        for i in range(len(images)):
            test_images.append(images[i])
            test_labels.append(labels[i].item())
            test_preds.append({
                'predicted': predicted[i].item(),
                'confidence': probs[i][predicted[i]].item(),
                'probs': probs[i].tolist(),
            })

# Mostrar algunas predicciones
print(f"  Predicciones en imagenes NUNCA VISTAS:\n")
for i in range(10):
    real = test_labels[i]
    pred = test_preds[i]['predicted']
    conf = test_preds[i]['confidence']
    check = "✓" if real == pred else "✗"
    print(f"    Imagen {i+1}: Real={real}, Prediccion={pred}, "
          f"Confianza={conf:.1%}  {check}")

# ─── Graficos de predicciones correctas ───
fig, axes = plt.subplots(2, 5, figsize=(14, 6))

# Buscar 10 predicciones correctas
correct_indices = [i for i in range(len(test_labels))
                   if test_labels[i] == test_preds[i]['predicted']][:10]

for idx, i in enumerate(correct_indices):
    ax = axes[idx // 5, idx % 5]
    img = test_images[i].squeeze().numpy()
    ax.imshow(img, cmap='gray')
    pred = test_preds[i]['predicted']
    conf = test_preds[i]['confidence']
    ax.set_title(f"Pred: {pred} ({conf:.0%})", fontsize=11, color='green')
    ax.axis('off')

plt.suptitle("Predicciones CORRECTAS (imagenes nunca vistas)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT}/14_predicciones_correctas.png", dpi=150)
plt.close()
print(f"\n  Guardado: {OUTPUT}/14_predicciones_correctas.png")

# ─── Graficos de predicciones incorrectas ───
wrong_indices = [i for i in range(len(test_labels))
                 if test_labels[i] != test_preds[i]['predicted']]

if len(wrong_indices) > 0:
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    n_wrong = min(10, len(wrong_indices))

    for idx in range(10):
        ax = axes[idx // 5, idx % 5]
        if idx < n_wrong:
            i = wrong_indices[idx]
            img = test_images[i].squeeze().numpy()
            ax.imshow(img, cmap='gray')
            real = test_labels[i]
            pred = test_preds[i]['predicted']
            conf = test_preds[i]['confidence']
            ax.set_title(f"Real: {real}, Pred: {pred} ({conf:.0%})",
                        fontsize=10, color='red')
        ax.axis('off')

    plt.suptitle(f"Predicciones INCORRECTAS ({len(wrong_indices)} de {len(test_labels)} total)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/15_predicciones_incorrectas.png", dpi=150)
    plt.close()
    print(f"  Guardado: {OUTPUT}/15_predicciones_incorrectas.png")
    print(f"\n  Errores: {len(wrong_indices)} de {len(test_labels)} "
          f"({len(wrong_indices)/len(test_labels)*100:.1f}%)")
    print(f"  → Muchos errores son digitos que HASTA UN HUMANO confundiria!")

# ─── Matriz de confusion ───
from collections import Counter
confusion = np.zeros((10, 10), dtype=int)
for i in range(len(test_labels)):
    confusion[test_labels[i]][test_preds[i]['predicted']] += 1

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(confusion, cmap='Blues')

for i in range(10):
    for j in range(10):
        color = 'white' if confusion[i, j] > confusion.max() * 0.5 else 'black'
        ax.text(j, i, str(confusion[i, j]), ha='center', va='center',
                fontsize=9, color=color)

ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xlabel("Prediccion de la red", fontsize=12)
ax.set_ylabel("Digito real", fontsize=12)
ax.set_title("Matriz de confusion\n(diagonal = predicciones correctas)", fontsize=13)
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(f"{OUTPUT}/16_confusion_matrix.png", dpi=150)
plt.close()
print(f"  Guardado: {OUTPUT}/16_confusion_matrix.png")


# ╔══════════════════════════════════════════════════════════╗
# ║  PASO 7: GUARDAR EL MODELO                             ║
# ╚══════════════════════════════════════════════════════════╝
print(f"\n{'=' * 60}")
print("PASO 7: GUARDAR EL MODELO ENTRENADO")
print("=" * 60)

torch.save(model.state_dict(), f"{OUTPUT}/modelo_mnist.pth")
print(f"\n  Modelo guardado en: {OUTPUT}/modelo_mnist.pth")
print(f"  Tamaño: {sum(p.numel() for p in model.parameters()) * 4 / 1024:.0f} KB")

print(f"""
  Para usar el modelo despues:

  # Cargar
  model = DigitRecognizer()
  model.load_state_dict(torch.load('modelo_mnist.pth'))
  model.eval()

  # Predecir
  imagen = transforms.ToTensor()(mi_imagen)   # tu imagen
  imagen = imagen.unsqueeze(0)                 # agregar dim de batch
  prediccion = model(imagen)                   # pasar por la red
  digito = prediccion.argmax().item()          # el digito predicho
  print(f"Es un {{digito}}")
""")


# ╔══════════════════════════════════════════════════════════╗
# ║  RESUMEN FINAL                                          ║
# ╚══════════════════════════════════════════════════════════╝
print(f"{'=' * 60}")
print("RESUMEN: Todo el proceso completo")
print("=" * 60)
print(f"""
  1. DATOS:       Cargar imagenes + normalizar [0,1] + armar batches
  2. RED:         Definir capas (Linear, BatchNorm, ReLU, Dropout)
  3. LOSS:        CrossEntropyLoss (mide que tan mal predice)
  4. OPTIMIZADOR: Adam (decide como ajustar los pesos)
  5. ENTRENAR:    Repetir por cada batch:
                    forward → loss → backward → update
  6. EVALUAR:     Probar con datos nunca vistos
  7. GUARDAR:     torch.save() para usar despues

  Resultado final:
    Train Accuracy: {history['train_acc'][-1]:.1f}%
    Test Accuracy:  {history['test_acc'][-1]:.1f}%
    (en {num_epochs} epocas de entrenamiento)

  Graficos guardados en {OUTPUT}/
""")
