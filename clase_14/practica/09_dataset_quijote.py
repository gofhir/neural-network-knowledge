"""
Experimento 3: cambiar el dataset.

En lugar de Shakespeare, usamos Don Quijote (Project Gutenberg, primera
parte). Vemos al modelo aprender ESPAÑOL desde cero, con el mismo modelo
y misma cantidad de training.
"""

import os
import urllib.request
import torch
from _models import (
    MiniGPT, CharTokenizer, get_device, prepare_data,
    make_batch_fn, train, sample,
)

torch.manual_seed(1337)
device = get_device()
print(f"Dispositivo: {device}")

# Descargar Don Quijote
QUIJOTE_URL = "https://www.gutenberg.org/cache/epub/2000/pg2000.txt"
LOCAL = "quijote.txt"
if not os.path.exists(LOCAL):
    print(f"Descargando Don Quijote desde Project Gutenberg...")
    urllib.request.urlretrieve(QUIJOTE_URL, LOCAL)

with open(LOCAL, "r", encoding="utf-8") as f:
    text = f.read()

# Limpieza basica: quitar el header de Gutenberg si existe
if "*** START" in text:
    text = text.split("*** START", 1)[1].split("\n", 1)[1]
if "*** END" in text:
    text = text.split("*** END", 1)[0]

# Tomar solo los primeros 500K chars para que sea comparable con tinyshakespeare
text = text[:500_000]
print(f"\nTexto cargado: {len(text):,} chars")
print(f"Primeros 200 chars:\n{text[:200]}")

tokenizer = CharTokenizer(text)
print(f"\nVocab size: {tokenizer.vocab_size}")
print(f"Caracteres: {''.join(tokenizer.chars)!r}")

_, train_data, val_data = prepare_data(text, tokenizer=tokenizer)

# Mismo modelo estandar
config = dict(d_model=128, h=4, n_layers=4, d_ff=512, block_size=64)
model = MiniGPT(vocab_size=tokenizer.vocab_size, **config).to(device)
get_batch = make_batch_fn(train_data, val_data, block_size=64, batch_size=32, device=device)

print("\n" + "=" * 70)
print("Training en Don Quijote: 3000 iteraciones")
print("=" * 70)
train(model, get_batch, max_iters=3000, label="modelo en Quijote")

# Generaciones
print("\n" + "=" * 70)
print("Generaciones en español")
print("=" * 70)

for prompt in ["", "En un lugar de ", "Don Quijote ", "Sancho dijo:\n"]:
    print(f"\n--- prompt: {prompt!r} ---")
    print(sample(model, tokenizer, prompt, max_new_tokens=300, temperature=0.8, device=device))

print("\n" + "=" * 70)
print("La MISMA arquitectura aprendio ESPAÑOL desde cero, sin que cambiemos")
print("una sola linea de codigo del modelo. La unica diferencia: el dataset.")
print("Eso muestra la universalidad del Transformer: solo cambia el input.")
print("=" * 70)
