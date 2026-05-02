"""
Experimento 1: efecto de entrenar mas tiempo.

Mismo modelo, distintos numeros de iteraciones. Vemos como evoluciona la
calidad del texto.
"""

import torch
from _models import (
    MiniGPT, CharTokenizer, get_device, load_text, prepare_data,
    make_batch_fn, train, sample,
)

torch.manual_seed(1337)
device = get_device()
print(f"Dispositivo: {device}")

# Datos
text = load_text()
tokenizer, train_data, val_data = prepare_data(text)

# Modelo y batch
config = dict(d_model=128, h=4, n_layers=4, d_ff=512, block_size=64)
model = MiniGPT(vocab_size=tokenizer.vocab_size, **config).to(device)
get_batch = make_batch_fn(train_data, val_data, block_size=64, batch_size=32, device=device)

# Entrenar mucho
print("\n" + "=" * 70)
print("Training: 6000 iteraciones (3x mas que el escalon 8 estandar)")
print("=" * 70)
train(model, get_batch, max_iters=6000, label="modelo entrenado 6000 iters")

# Generar
print("\n" + "=" * 70)
print("Generaciones (200 chars cada una)")
print("=" * 70)

for prompt in ["ROMEO:\n", "HAMLET:\n", "JULIET:\nO Romeo, "]:
    print(f"\n--- prompt: {prompt!r} ---")
    print(sample(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, device=device))

print("\n" + "=" * 70)
print("Compara con el escalon 8 (3000 iters): el texto ahora es mas coherente,")
print("con menos 'palabras inventadas' y mejor estructura de dialogo.")
print("=" * 70)
