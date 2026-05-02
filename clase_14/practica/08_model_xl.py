"""
Experimento 2: modelo XL (mas grande).

d_model=256, n_layers=6, h=8 -> ~3M parametros (vs 0.82M del estandar).
Vemos si el modelo "mas grande" produce texto notablemente mejor.
"""

import torch
from _models import (
    MiniGPT, CharTokenizer, get_device, load_text, prepare_data,
    make_batch_fn, train, sample,
)

torch.manual_seed(1337)
device = get_device()
print(f"Dispositivo: {device}")

text = load_text()
tokenizer, train_data, val_data = prepare_data(text)

# Modelo XL
config = dict(d_model=256, h=8, n_layers=6, d_ff=1024, block_size=64)
model = MiniGPT(vocab_size=tokenizer.vocab_size, **config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModelo XL: {n_params:,} parametros ({n_params/1e6:.2f} M)")
print(f"Config: {config}")

get_batch = make_batch_fn(train_data, val_data, block_size=64, batch_size=32, device=device)

# Entrenar
print("\n" + "=" * 70)
print("Training XL: 3000 iteraciones")
print("=" * 70)
train(model, get_batch, max_iters=3000, label="modelo XL")

# Generaciones
print("\n" + "=" * 70)
print("Generaciones (300 chars)")
print("=" * 70)

for prompt in ["ROMEO:\n", "JULIET:\n", "What light through "]:
    print(f"\n--- prompt: {prompt!r} ---")
    print(sample(model, tokenizer, prompt, max_new_tokens=300, temperature=0.8, device=device))

print("\n" + "=" * 70)
print("3.5x mas parametros que el estandar, en 3000 iters. La calidad")
print("deberia ser visiblemente mejor: vocabulario mas rico, estructura")
print("mas coherente, menos palabras inventadas.")
print("=" * 70)
