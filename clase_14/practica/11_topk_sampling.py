"""
Experimento 5: Top-k sampling.

Comparacion de estrategias de sampling:
  - greedy (siempre el mas probable): repetitivo, sin variedad
  - multinomial libre: puede caer en tokens raros (errores)
  - top-k (k=5, 20): solo samplea de los k mas probables
                    balance entre diversidad y calidad

Top-k es lo que usan ChatGPT, Claude, Gemini en produccion.
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

# Comparar estrategias de sampling
print("\n" + "=" * 70)
print("Comparacion de estrategias de sampling")
print("=" * 70)

prompt = "ROMEO:\n"

print(f"\n=== Greedy (top_k=1, temperature=1.0) ===")
print("Siempre elige el token mas probable. Sin variedad. Suele caer en loops.")
torch.manual_seed(42)
print(sample(model, tokenizer, prompt, max_new_tokens=200, temperature=1.0, top_k=1, device=device))

print(f"\n=== Multinomial libre (top_k=None, temperature=1.0) ===")
print("Samplea de toda la distribucion. Maxima variedad. A veces tokens raros.")
torch.manual_seed(42)
print(sample(model, tokenizer, prompt, max_new_tokens=200, temperature=1.0, top_k=None, device=device))

print(f"\n=== Top-k=5 (temperature=1.0) ===")
print("Solo de los 5 tokens mas probables. Balance: diverso pero coherente.")
torch.manual_seed(42)
print(sample(model, tokenizer, prompt, max_new_tokens=200, temperature=1.0, top_k=5, device=device))

print(f"\n=== Top-k=20 (temperature=1.0) ===")
print("Solo de los 20 mas probables. Mas variedad que k=5.")
torch.manual_seed(42)
print(sample(model, tokenizer, prompt, max_new_tokens=200, temperature=1.0, top_k=20, device=device))

print(f"\n=== Top-k=5 + temperature=0.8 (settings comunes en LLMs) ===")
torch.manual_seed(42)
print(sample(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, top_k=5, device=device))

print("\n" + "=" * 70)
print("ChatGPT/Claude usan combinaciones de temperature + top_k + top_p (nucleus")
print("sampling). top_k=40 + temperature=0.7 es un setting comun. La idea:")
print("controlar diversidad sin permitir tokens muy improbables.")
print("=" * 70)
