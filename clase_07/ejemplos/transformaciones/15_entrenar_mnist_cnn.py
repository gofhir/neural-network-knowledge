"""
Ejemplo 15 — Entrenar MNIST con CNN (Red Convolucional)

Objetivo: comparar la red "plana" (Flatten + Linear) con una CNN
          que ENTIENDE la estructura 2D de la imagen.

Genera graficos en /app/output/:
  - 17_cnn_filtros_explicacion.png
  - 18_cnn_feature_maps.png
  - 19_cnn_vs_linear_comparacion.png

Ejecutar:
  docker run --rm -v $(pwd)/output:/app/output clase7-pytorch \
    python -u ejemplos/transformaciones/15_entrenar_mnist_cnn.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

OUTPUT = "/app/output"

# Cargar datos
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='/tmp/data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='/tmp/data', train=False, download=True, transform=transform)

batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ╔══════════════════════════════════════════════════════════╗
# ║  PASO 1: QUE ES UNA CONVOLUCION                        ║
# ╚══════════════════════════════════════════════════════════╝
print("=" * 60)
print("PASO 1: QUE ES UNA CONVOLUCION")
print("=" * 60)

print("""
  Una convolucion es un FILTRO que recorre la imagen
  mirando zonas pequenas (por ejemplo 3x3 pixeles).

  Imagina que pones una lupa de 3x3 sobre la imagen
  y la vas moviendo pixel por pixel:

    Imagen:              Filtro 3x3:          Resultado:
    ┌─────────────┐      ┌─────────┐
    │ 0  0  0  0  │      │-1  0  1 │         El filtro MULTIPLICA
    │ 0 [1  1] 0  │  x   │-1  0  1 │   =     cada pixel por su peso
    │ 0 [0  1] 0  │      │-1  0  1 │         y SUMA todo.
    │ 0  0  0  0  │      └─────────┘
    └─────────────┘                          Si el resultado es alto,
                                             hay un patron aqui!

  Distintos filtros detectan distintas cosas:
    Filtro 1: bordes verticales   |
    Filtro 2: bordes horizontales ─
    Filtro 3: esquinas            ┘
    Filtro 4: texturas            ░
""")

# Demostrar con un ejemplo real
image, label = train_dataset[0]  # primer digito
img = image.squeeze()  # quitar dim de canal: (28, 28)

# Definir filtros manuales para demostrar
filter_vertical = torch.tensor([[-1, 0, 1],
                                 [-1, 0, 1],
                                 [-1, 0, 1]], dtype=torch.float32)

filter_horizontal = torch.tensor([[-1, -1, -1],
                                   [ 0,  0,  0],
                                   [ 1,  1,  1]], dtype=torch.float32)

# Aplicar filtros manualmente
# Conv2d espera (batch, canal, alto, ancho)
img_4d = image.unsqueeze(0)  # (1, 1, 28, 28)

# Crear capas conv con filtros manuales
conv_v = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
conv_h = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

with torch.no_grad():
    conv_v.weight[0, 0] = filter_vertical
    conv_h.weight[0, 0] = filter_horizontal

edges_v = conv_v(img_4d).squeeze().detach()
edges_h = conv_h(img_4d).squeeze().detach()

# Grafico
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Fila 1: la imagen y los filtros
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title(f"Imagen original\n(digito '{label}')", fontsize=11)
axes[0, 0].axis('off')

axes[0, 1].imshow(filter_vertical, cmap='RdBu', vmin=-1, vmax=1)
axes[0, 1].set_title("Filtro: bordes\nverticales", fontsize=11)
for i in range(3):
    for j in range(3):
        axes[0, 1].text(j, i, f"{filter_vertical[i,j]:.0f}",
                        ha='center', va='center', fontsize=14, fontweight='bold')
axes[0, 1].axis('off')

axes[0, 2].imshow(filter_horizontal, cmap='RdBu', vmin=-1, vmax=1)
axes[0, 2].set_title("Filtro: bordes\nhorizontales", fontsize=11)
for i in range(3):
    for j in range(3):
        axes[0, 2].text(j, i, f"{filter_horizontal[i,j]:.0f}",
                        ha='center', va='center', fontsize=14, fontweight='bold')
axes[0, 2].axis('off')

# Fila 2: resultados
axes[1, 0].imshow(img, cmap='gray')
axes[1, 0].set_title("Original", fontsize=11)
axes[1, 0].axis('off')

axes[1, 1].imshow(edges_v, cmap='RdBu')
axes[1, 1].set_title("Resultado: bordes\nverticales detectados", fontsize=11)
axes[1, 1].axis('off')

axes[1, 2].imshow(edges_h, cmap='RdBu')
axes[1, 2].set_title("Resultado: bordes\nhorizontales detectados", fontsize=11)
axes[1, 2].axis('off')

plt.suptitle("Convolucion: un filtro 3x3 recorre la imagen detectando patrones",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT}/17_cnn_filtros_explicacion.png", dpi=150)
plt.close()
print(f"  Guardado: {OUTPUT}/17_cnn_filtros_explicacion.png")


# ╔══════════════════════════════════════════════════════════╗
# ║  PASO 2: CONSTRUIR LA CNN                              ║
# ╚══════════════════════════════════════════════════════════╝
print(f"\n{'=' * 60}")
print("PASO 2: CONSTRUIR LA CNN")
print("=" * 60)

print("""
  CNN = Capas convolucionales + capas lineales al final

  Red "plana" (ejemplo 14):       CNN (este ejemplo):
  ──────────────────────          ──────────────────────
  Flatten (pierde 2D)             Conv2d (mantiene 2D)
  Linear 784→256                  Conv2d (detecta patrones)
  Linear 256→128                  Flatten (solo al final)
  Linear 128→10                   Linear → 10 clases

  La CNN primero ENTIENDE la imagen (bordes, formas)
  y solo al final aplana para clasificar.
""")


class CNN_DigitRecognizer(nn.Module):
    def __init__(self):
        super().__init__()

        # --- Parte convolucional: entiende la estructura 2D ---

        # Conv2d(entrada, salida, filtro)
        # 1 canal de entrada (blanco/negro)
        # 16 filtros de 3x3 → detecta 16 patrones distintos
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # BatchNorm para imagenes (2D)

        # 16 canales de entrada, 32 filtros
        # → detecta 32 patrones mas complejos
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # MaxPool: reduce el tamano a la mitad (28→14→7)
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout para regularizacion
        self.dropout_conv = nn.Dropout2d(0.25)
        self.dropout_fc = nn.Dropout(0.5)

        # --- Parte lineal: clasifica ---
        # Despues de 2 pools: 28→14→7, con 32 canales: 32*7*7 = 1568
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Capa conv 1: Conv → BatchNorm → ReLU → Pool
        x = self.conv1(x)       # (batch, 1, 28, 28) → (batch, 16, 28, 28)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)        # (batch, 16, 28, 28) → (batch, 16, 14, 14)
        x = self.dropout_conv(x)

        # Capa conv 2: Conv → BatchNorm → ReLU → Pool
        x = self.conv2(x)       # (batch, 16, 14, 14) → (batch, 32, 14, 14)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)        # (batch, 32, 14, 14) → (batch, 32, 7, 7)
        x = self.dropout_conv(x)

        # Aplanar (solo aqui, despues de extraer patrones)
        x = x.view(x.size(0), -1)  # (batch, 32, 7, 7) → (batch, 1568)

        # Capas lineales para clasificar
        x = self.fc1(x)         # (batch, 1568) → (batch, 128)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)         # (batch, 128) → (batch, 10)
        return x


cnn_model = CNN_DigitRecognizer()

# Mostrar como cambian las dimensiones
print("  Arquitectura CNN (dimensiones paso a paso):\n")
sample = torch.randn(1, 1, 28, 28)  # 1 imagen

x = sample
layers_demo = [
    ("Entrada", None),
    ("Conv2d(1→16, 3x3)", cnn_model.conv1),
    ("BatchNorm2d(16)", cnn_model.bn1),
    ("ReLU", None),
    ("MaxPool2d(2)", cnn_model.pool),
    ("Conv2d(16→32, 3x3)", cnn_model.conv2),
    ("BatchNorm2d(32)", cnn_model.bn2),
    ("ReLU", None),
    ("MaxPool2d(2)", cnn_model.pool),
    ("Flatten", None),
    ("Linear(1568→128)", cnn_model.fc1),
    ("Linear(128→10)", cnn_model.fc2),
]

with torch.no_grad():
    for name, layer in layers_demo:
        if name == "Entrada":
            pass
        elif name == "ReLU":
            x = F.relu(x)
        elif name == "Flatten":
            x = x.view(x.size(0), -1)
        else:
            x = layer(x)
        print(f"    {name:25s} → shape: {tuple(x.shape)}")

total_params_cnn = sum(p.numel() for p in cnn_model.parameters())
print(f"\n  Total parametros CNN: {total_params_cnn:,}")


# ╔══════════════════════════════════════════════════════════╗
# ║  PASO 3: VER QUE DETECTAN LOS FILTROS                  ║
# ╚══════════════════════════════════════════════════════════╝
print(f"\n{'=' * 60}")
print("PASO 3: VISUALIZAR QUE VE LA CNN (antes de entrenar)")
print("=" * 60)

# Feature maps de la primera capa
cnn_model.eval()
with torch.no_grad():
    # Pasar una imagen por la primera capa conv
    first_conv_output = cnn_model.conv1(img_4d)
    first_conv_output = F.relu(first_conv_output)

print(f"\n  Imagen de entrada: {img_4d.shape}")
print(f"  Despues de conv1:  {first_conv_output.shape}")
print(f"    → 16 'feature maps' (uno por cada filtro)")
print(f"    → Cada feature map muestra DONDE se activa ese filtro")

# Graficar los 16 feature maps
fig, axes = plt.subplots(2, 9, figsize=(16, 4))

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title("Original", fontsize=9)
axes[0, 0].axis('off')
axes[1, 0].axis('off')

for i in range(16):
    row = i // 8
    col = (i % 8) + 1
    feature_map = first_conv_output[0, i].numpy()
    axes[row, col].imshow(feature_map, cmap='viridis')
    axes[row, col].set_title(f"Filtro {i+1}", fontsize=8)
    axes[row, col].axis('off')

plt.suptitle("Feature maps: cada filtro detecta un patron distinto\n(antes de entrenar = aleatorio)",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT}/18_cnn_feature_maps.png", dpi=150)
plt.close()
print(f"  Guardado: {OUTPUT}/18_cnn_feature_maps.png")


# ╔══════════════════════════════════════════════════════════╗
# ║  PASO 4: ENTRENAR LA CNN                               ║
# ╚══════════════════════════════════════════════════════════╝
print(f"\n{'=' * 60}")
print("PASO 4: ENTRENAR LA CNN")
print("=" * 60)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)

num_epochs = 5
history_cnn = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

for epoch in range(num_epochs):
    # Entrenamiento
    cnn_model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        predictions = cnn_model(images)
        loss = loss_fn(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = predictions.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % 200 == 0:
            print(f"    Epoca {epoch+1}/{num_epochs}, "
                  f"Batch {batch_idx}/{len(train_loader)}, "
                  f"Loss: {loss.item():.4f}")

    train_loss = running_loss / len(train_loader)
    train_acc = 100.0 * correct / total

    # Evaluacion
    cnn_model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            predictions = cnn_model(images)
            loss = loss_fn(predictions, labels)
            test_loss += loss.item()
            _, predicted = predictions.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = test_loss / len(test_loader)
    test_acc = 100.0 * correct / total

    history_cnn['train_loss'].append(train_loss)
    history_cnn['test_loss'].append(test_loss)
    history_cnn['train_acc'].append(train_acc)
    history_cnn['test_acc'].append(test_acc)

    print(f"\n  ► Epoca {epoch+1}/{num_epochs}:")
    print(f"    Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.1f}%")
    print(f"    Test Loss:  {test_loss:.4f}  |  Test Acc:  {test_acc:.1f}%\n")


# ╔══════════════════════════════════════════════════════════╗
# ║  PASO 5: COMPARAR CNN vs RED PLANA                     ║
# ╚══════════════════════════════════════════════════════════╝
print(f"{'=' * 60}")
print("PASO 5: COMPARACION CNN vs RED PLANA")
print("=" * 60)

# Entrenar red plana para comparar
class LinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.network(x)

linear_model = LinearNet()
optimizer_linear = torch.optim.Adam(linear_model.parameters(), lr=0.001)
history_linear = {'train_acc': [], 'test_acc': []}

print("\n  Entrenando red plana (Flatten + Linear) para comparar...")
for epoch in range(num_epochs):
    linear_model.train()
    correct = 0
    total = 0
    for images, labels in train_loader:
        pred = linear_model(images)
        loss = loss_fn(pred, labels)
        optimizer_linear.zero_grad()
        loss.backward()
        optimizer_linear.step()
        _, predicted = pred.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    train_acc = 100.0 * correct / total

    linear_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            pred = linear_model(images)
            _, predicted = pred.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    test_acc = 100.0 * correct / total
    history_linear['train_acc'].append(train_acc)
    history_linear['test_acc'].append(test_acc)
    print(f"    Epoca {epoch+1}: Train={train_acc:.1f}%, Test={test_acc:.1f}%")

# Grafico comparativo
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

epochs_range = range(1, num_epochs + 1)

# Accuracy
ax1.plot(epochs_range, history_cnn['test_acc'], 'o-', label='CNN', color='steelblue', linewidth=2)
ax1.plot(epochs_range, history_linear['test_acc'], 's--', label='Red plana (Linear)', color='coral', linewidth=2)
ax1.set_xlabel('Epoca', fontsize=12)
ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
ax1.set_title('Precision en datos de prueba', fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(95, 100)

# Tabla comparativa
ax2.axis('off')
total_params_linear = sum(p.numel() for p in linear_model.parameters())

table_data = [
    ["", "Red Plana\n(Flatten+Linear)", "CNN\n(Conv2d+Linear)"],
    ["Arquitectura", "Flatten → Linear\n→ Linear → Linear", "Conv → Conv\n→ Flatten → Linear"],
    ["Entiende 2D?", "NO\n(aplana todo)", "SI\n(filtros 3x3)"],
    ["Parametros", f"{total_params_linear:,}", f"{total_params_cnn:,}"],
    ["Test Accuracy", f"{history_linear['test_acc'][-1]:.1f}%", f"{history_cnn['test_acc'][-1]:.1f}%"],
    ["Ideal para", "Datos tabulares\n(CSV, tablas)", "Imagenes\n(fotos, escaneos)"],
]

table = ax2.table(cellText=table_data, cellLoc='center', loc='center',
                  colWidths=[0.25, 0.35, 0.35])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.2)

# Colorear header
for j in range(3):
    table[0, j].set_facecolor('#4472C4')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Colorear fila de accuracy
for j in range(3):
    table[4, j].set_facecolor('#E2EFDA')

ax2.set_title("Comparacion", fontsize=13)

plt.suptitle("CNN vs Red Plana en MNIST", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT}/19_cnn_vs_linear_comparacion.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Guardado: {OUTPUT}/19_cnn_vs_linear_comparacion.png")

# ╔══════════════════════════════════════════════════════════╗
# ║  PASO 6: FEATURE MAPS DESPUES DE ENTRENAR              ║
# ╚══════════════════════════════════════════════════════════╝
print(f"\n{'=' * 60}")
print("PASO 6: QUE APRENDIO LA CNN (feature maps entrenados)")
print("=" * 60)

cnn_model.eval()
with torch.no_grad():
    conv1_out = F.relu(cnn_model.bn1(cnn_model.conv1(img_4d)))
    conv2_out = F.relu(cnn_model.bn2(cnn_model.conv2(cnn_model.pool(conv1_out))))

fig, axes = plt.subplots(3, 9, figsize=(16, 6))

# Original
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title("Original", fontsize=9)
axes[0, 0].axis('off')
for i in range(1, 9):
    axes[0, i].axis('off')

# Conv1: 16 filtros (mostramos 8)
for i in range(8):
    axes[1, i].imshow(conv1_out[0, i].numpy(), cmap='viridis')
    axes[1, i].set_title(f"Conv1-{i+1}", fontsize=8)
    axes[1, i].axis('off')
axes[1, 8].axis('off')

# Conv2: 32 filtros (mostramos 8)
for i in range(8):
    axes[2, i].imshow(conv2_out[0, i].numpy(), cmap='viridis')
    axes[2, i].set_title(f"Conv2-{i+1}", fontsize=8)
    axes[2, i].axis('off')
axes[2, 8].axis('off')

plt.suptitle("Feature maps DESPUES de entrenar\nConv1: bordes simples → Conv2: patrones complejos",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT}/18_cnn_feature_maps.png", dpi=150)
plt.close()
print(f"  Guardado: {OUTPUT}/18_cnn_feature_maps.png (actualizado con red entrenada)")

# ╔══════════════════════════════════════════════════════════╗
# ║  RESUMEN                                                ║
# ╚══════════════════════════════════════════════════════════╝
print(f"\n{'=' * 60}")
print("RESUMEN")
print("=" * 60)
print(f"""
  Red Plana (ejemplo 14):
    Flatten → Linear → Linear → Linear
    Test Accuracy: {history_linear['test_acc'][-1]:.1f}%
    Problema: destruye la estructura 2D de la imagen

  CNN (este ejemplo):
    Conv2d → Conv2d → Flatten → Linear
    Test Accuracy: {history_cnn['test_acc'][-1]:.1f}%
    Ventaja: entiende que pixeles vecinos estan relacionados

  La CNN es mejor porque:
    1. Los FILTROS detectan patrones locales (bordes, curvas)
    2. El POOLING reduce el tamano (28→14→7) manteniendo lo importante
    3. Las capas profundas combinan patrones simples en complejos:
       Capa 1: "hay un borde aqui"
       Capa 2: "hay una curva aqui" (combinacion de bordes)
       Clasificador: "bordes + curvas en esta forma = digito 3"

  En MNIST la diferencia es chica (~1%).
  En imagenes reales (fotos) la diferencia es ENORME (~60%).

  Graficos guardados en {OUTPUT}/
""")
