"""
Ejemplo 11 — Como se transforma una IMAGEN en numeros

Objetivo: ver paso a paso como una imagen (foto) se convierte
          en un tensor de numeros que la red puede procesar.

Genera graficos en /app/output/:
  - 01_imagen_original.png
  - 02_imagen_pixeles_zoom.png
  - 03_imagen_canales_rgb.png
  - 04_imagen_normalizada.png

Ejecutar:
  docker run --rm -v $(pwd)/output:/app/output clase7-pytorch \
    python -u ejemplos/transformaciones/11_transformar_imagen.py
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # no necesita pantalla
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

OUTPUT = "/app/output"

# =============================================
# 1. Descargar una imagen real (MNIST)
# =============================================
print("=" * 60)
print("PASO 1: Obtener una imagen")
print("=" * 60)

mnist = datasets.MNIST(root='/tmp/data', train=True, download=True)

# Tomar el digito 3 (buscamos uno)
for i in range(len(mnist)):
    img, label = mnist[i]
    if label == 3:
        break

print(f"\n  Tipo original: {type(img)}")
print(f"  Es una imagen PIL (libreria de imagenes de Python)")
print(f"  Etiqueta: {label} (es el digito '{label}')")

# Guardar imagen original
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.imshow(img, cmap='gray')
ax.set_title(f"Imagen original - Digito '{label}'", fontsize=14)
ax.axis('off')
plt.tight_layout()
plt.savefig(f"{OUTPUT}/01_imagen_original.png", dpi=100)
plt.close()
print(f"  Guardado: {OUTPUT}/01_imagen_original.png")

# =============================================
# 2. Convertir a array de numeros (pixeles)
# =============================================
print(f"\n{'=' * 60}")
print("PASO 2: Convertir a numeros (pixeles)")
print("=" * 60)

# Convertir PIL → numpy array
pixels = np.array(img)

print(f"\n  np.array(imagen)")
print(f"  Shape: {pixels.shape} → {pixels.shape[0]} filas x {pixels.shape[1]} columnas")
print(f"  Tipo de dato: {pixels.dtype}")
print(f"  Rango de valores: {pixels.min()} a {pixels.max()}")
print(f"  (0 = negro puro, 255 = blanco puro)")

# Mostrar una zona de pixeles con sus valores numericos
print(f"\n  Pixeles de la zona central (filas 10-17, columnas 10-17):")
print(f"  Cada numero es la 'claridad' de ese pixel (0-255):\n")
zona = pixels[10:18, 10:18]
for row in zona:
    print("  " + "  ".join(f"{v:3d}" for v in row))

# Guardar grafico con zoom a los pixeles
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.imshow(pixels, cmap='gray')
ax1.set_title("Imagen completa (28x28)")
ax1.axis('off')

# Zoom con valores numericos
zona_grande = pixels[8:20, 8:20]
ax2.imshow(zona_grande, cmap='gray', interpolation='nearest')
ax2.set_title("Zoom (cada cuadrado = 1 pixel)")
for i in range(zona_grande.shape[0]):
    for j in range(zona_grande.shape[1]):
        color = 'white' if zona_grande[i, j] < 128 else 'black'
        ax2.text(j, i, str(zona_grande[i, j]),
                ha='center', va='center', fontsize=7, color=color)
ax2.axis('off')

plt.tight_layout()
plt.savefig(f"{OUTPUT}/02_imagen_pixeles_zoom.png", dpi=150)
plt.close()
print(f"\n  Guardado: {OUTPUT}/02_imagen_pixeles_zoom.png")

# =============================================
# 3. Imagen a color (RGB = 3 canales)
# =============================================
print(f"\n{'=' * 60}")
print("PASO 3: Imagenes a color (3 canales RGB)")
print("=" * 60)

print(f"""
  MNIST es blanco y negro (1 canal).
  Una foto a color tiene 3 canales: Rojo, Verde, Azul (RGB).

  Foto B/N:   shape = (28, 28)      → 1 grilla de pixeles
  Foto color:  shape = (3, 224, 224)  → 3 grillas (R, G, B)
""")

# Crear una imagen a color sintetica para mostrar los canales
np.random.seed(42)
# Simular una foto pequeña con colores
color_img = np.zeros((8, 8, 3), dtype=np.uint8)
color_img[:, :4, 0] = 200    # mitad izquierda roja
color_img[:, 4:, 2] = 200    # mitad derecha azul
color_img[2:6, 2:6, 1] = 200  # cuadrado verde al centro

fig, axes = plt.subplots(1, 4, figsize=(14, 3))

axes[0].imshow(color_img)
axes[0].set_title("Imagen color\n(R + G + B combinados)")
axes[0].axis('off')

titles = ['Canal Rojo (R)', 'Canal Verde (G)', 'Canal Azul (B)']
cmaps = ['Reds', 'Greens', 'Blues']
for i in range(3):
    axes[i+1].imshow(color_img[:, :, i], cmap=cmaps[i], vmin=0, vmax=255)
    axes[i+1].set_title(f"{titles[i]}\nshape = (8, 8)")
    axes[i+1].axis('off')

plt.suptitle("Una imagen a color son 3 'capas' de numeros", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT}/03_imagen_canales_rgb.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Guardado: {OUTPUT}/03_imagen_canales_rgb.png")

print(f"\n  Canal Rojo:")
print(f"  {color_img[:, :, 0]}")
print(f"\n  Cada canal es una grilla de numeros independiente.")
print(f"  La red ve 3 grillas, no 'colores'.")

# =============================================
# 4. Normalizar para la red (0-255 → 0-1)
# =============================================
print(f"\n{'=' * 60}")
print("PASO 4: Normalizar para la red")
print("=" * 60)

print(f"""
  Los pixeles van de 0 a 255. Pero las redes funcionan mejor
  con valores pequenos. Se normaliza dividiendo por 255:

  pixel_normalizado = pixel / 255
  0   → 0.0  (negro)
  128 → 0.5  (gris)
  255 → 1.0  (blanco)
""")

# Convertir a tensor de PyTorch normalizado
transform = transforms.ToTensor()  # hace PIL → tensor [0-1]
tensor = transform(img)

print(f"  transforms.ToTensor() hace 2 cosas:")
print(f"    1. Divide por 255 (normaliza a [0, 1])")
print(f"    2. Reordena a (canales, alto, ancho)")
print(f"")
print(f"  Antes:  numpy array  shape = {pixels.shape}    rango = [{pixels.min()}, {pixels.max()}]")
print(f"  Despues: tensor      shape = {tuple(tensor.shape)}  rango = [{tensor.min():.2f}, {tensor.max():.2f}]")

# Mostrar comparacion
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.imshow(pixels, cmap='gray', vmin=0, vmax=255)
ax1.set_title(f"Antes: valores 0-255\nshape = {pixels.shape}")
ax1.axis('off')

ax2.imshow(tensor.squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
ax2.set_title(f"Despues: valores 0.0-1.0\nshape = {tuple(tensor.shape)}")
ax2.axis('off')

plt.suptitle("Normalizar: dividir por 255", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUTPUT}/04_imagen_normalizada.png", dpi=150)
plt.close()
print(f"\n  Guardado: {OUTPUT}/04_imagen_normalizada.png")

# =============================================
# 5. Armar un batch y pasarlo por la red
# =============================================
print(f"\n{'=' * 60}")
print("PASO 5: Armar un batch y pasarlo por la red")
print("=" * 60)

# Cargar varias imagenes
tensors = []
labels = []
for i in range(8):
    img_i, label_i = mnist[i]
    tensors.append(transform(img_i))
    labels.append(label_i)

batch = torch.stack(tensors)  # (8, 1, 28, 28)
print(f"\n  8 imagenes apiladas en un batch:")
print(f"  Shape: {batch.shape}")
print(f"    → batch={batch.shape[0]}, canales={batch.shape[1]}, alto={batch.shape[2]}, ancho={batch.shape[3]}")
print(f"  Etiquetas: {labels}")

import torch.nn as nn
model = nn.Sequential(
    nn.Flatten(),           # (8, 1, 28, 28) → (8, 784)
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),     # 10 clases (digitos 0-9)
)

output = model(batch)
print(f"\n  Dentro de la red:")
print(f"    Entrada: {batch.shape}  →  Flatten  →  (8, 784)  →  Linear  →  Salida: {output.shape}")
print(f"    784 = 1 x 28 x 28 (todos los pixeles en una fila)")

# =============================================
# RESUMEN
# =============================================
print(f"\n{'=' * 60}")
print("RESUMEN: Imagen → Tensor")
print("=" * 60)
print(f"""
  Foto (PIL/JPEG/PNG)
    ↓  np.array()
  Array de pixeles [0-255], shape (28, 28)
    ↓  / 255.0
  Array normalizado [0.0-1.0], shape (28, 28)
    ↓  agregar canal
  Tensor shape (1, 28, 28)
    ↓  apilar en batch
  Batch shape (32, 1, 28, 28)
    ↓  pasar a la red
  Salida shape (32, 10) → una prediccion por imagen

  Graficos guardados en {OUTPUT}/
""")
