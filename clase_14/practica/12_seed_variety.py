"""
Experimento 6: variedad con distintas semillas.

Mismo modelo, mismo prompt, distintas semillas aleatorias en sampling.
Muestra que el modelo NO es deterministico (excepto por la semilla):
genera continuaciones distintas cada vez.

Es lo que hace que cada conversacion con ChatGPT o Claude sea unica:
mismo prompt, distinta seed, distinta respuesta.
"""

import torch
from _models import (
    MiniGPT, get_device, load_text, prepare_data,
    make_batch_fn, train, sample,
)

torch.manual_seed(1337)
device = get_device()
print(f"Dispositivo: {device}")

text = load_text()
tokenizer, train_data, val_data = prepare_data(text)

config = dict(d_model=128, h=4, n_layers=4, d_ff=512, block_size=64)
model = MiniGPT(vocab_size=tokenizer.vocab_size, **config).to(device)
get_batch = make_batch_fn(train_data, val_data, block_size=64, batch_size=32, device=device)

print("\nEntrenando modelo (3000 iters)...")
train(model, get_batch, max_iters=3000, label="modelo")

# Generar con distintas semillas
print("\n" + "=" * 70)
print("5 generaciones del mismo prompt, distintas semillas")
print("=" * 70)

prompt = "ROMEO:\n"

for seed in [0, 42, 100, 1234, 9999]:
    print(f"\n--- seed = {seed} ---")
    torch.manual_seed(seed)
    print(sample(
        model, tokenizer, prompt,
        max_new_tokens=200, temperature=0.8, top_k=10,
        device=device,
    ))

print("\n" + "=" * 70)
print("Cada seed produce una continuacion distinta. Las 5 son 'validas'")
print("(coherentes con la distribucion aprendida), pero distintas en")
print("contenido. Esa es la base de la VARIEDAD en LLMs: el sampling")
print("aleatorio del softmax produce respuestas distintas para el mismo")
print("prompt.")
print("=" * 70)
