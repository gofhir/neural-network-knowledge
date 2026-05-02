"""
Experimento 4: GELU vs ReLU.

Vaswani 2017 uso ReLU. Modelos modernos (BERT, GPT-2+, LLaMA) usan GELU.
Aqui entrenamos dos modelos identicos excepto por la activacion del FFN
y comparamos.

GELU(x) = x * Phi(x) donde Phi es la CDF de la normal estandar.
Es "ReLU suavizada": diferenciable en cero, no apaga completamente las
neuronas con activacion negativa.
"""

import torch
from _models import (
    MiniGPT, get_device, load_text, prepare_data,
    make_batch_fn, train, sample,
)

device = get_device()
print(f"Dispositivo: {device}")

text = load_text()
tokenizer, train_data, val_data = prepare_data(text)

config = dict(d_model=128, h=4, n_layers=4, d_ff=512, block_size=64)
get_batch = make_batch_fn(train_data, val_data, block_size=64, batch_size=32, device=device)


def train_and_eval(activation):
    print("\n" + "=" * 70)
    print(f"Training con activacion: {activation.upper()}")
    print("=" * 70)
    torch.manual_seed(1337)  # misma semilla -> misma init -> comparable
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size, activation=activation, **config,
    ).to(device)
    losses = train(model, get_batch, max_iters=3000, label=f"FFN {activation}")
    final_loss = sum(losses[-50:]) / 50  # promedio de los ultimos 50 steps
    return model, final_loss


model_relu, loss_relu = train_and_eval("relu")
model_gelu, loss_gelu = train_and_eval("gelu")

print("\n" + "=" * 70)
print("COMPARACION FINAL")
print("=" * 70)
print(f"\nLoss promedio (ultimos 50 steps):")
print(f"  ReLU: {loss_relu:.4f}")
print(f"  GELU: {loss_gelu:.4f}")
print(f"  Diferencia: {(loss_relu - loss_gelu):+.4f}  (negativo = GELU mejor)")

# Generaciones
print("\n" + "=" * 70)
print("Generaciones lado a lado (mismo prompt y semilla)")
print("=" * 70)

for prompt in ["ROMEO:\n", "JULIET:\n"]:
    print(f"\n--- prompt: {prompt!r} ---")
    print(f"\nReLU:")
    torch.manual_seed(42)
    print(sample(model_relu, tokenizer, prompt, max_new_tokens=200, temperature=0.8, device=device))
    print(f"\nGELU:")
    torch.manual_seed(42)
    print(sample(model_gelu, tokenizer, prompt, max_new_tokens=200, temperature=0.8, device=device))

print("\n" + "=" * 70)
print("GELU tiende a converger un poco mejor en modelos pequeños. La diferencia")
print("se amplifica con escala. Por eso BERT, GPT-2, GPT-3, LLaMA usan GELU")
print("(o variantes como SwiGLU).")
print("=" * 70)
