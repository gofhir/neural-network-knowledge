"""57_activation_patching.py - Cap 57: activation patching para causalidad."""
import torch
from _models import load_pretrained_mini_llama, get_device, CharTokenizer, load_text
from _interp import cache_activations, patch_activation

torch.manual_seed(1337)
device = get_device()

text = load_text("shakespeare.txt")
tok = CharTokenizer(text)

model = load_pretrained_mini_llama("checkpoints/mini_llama_base.pt", device=device,
                                   config=dict(vocab_size=tok.vocab_size, max_seq_len=256,
                                               d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384))

# Pareja clean/corrupted: misma estructura, distinto speaker
clean_prompt = "BRUTUS:\nI am "
corrupted_prompt = "BIANCA:\nI am "

# Asegurar misma longitud
assert len(clean_prompt) == len(corrupted_prompt)
clean_ids = torch.tensor([tok.encode(clean_prompt)], dtype=torch.long, device=device)
corrupted_ids = torch.tensor([tok.encode(corrupted_prompt)], dtype=torch.long, device=device)
T = clean_ids.shape[1]

print(f"Clean:     {clean_prompt!r}")
print(f"Corrupted: {corrupted_prompt!r}")
print(f"T = {T} tokens\n")

# Run clean: capturar activaciones de todos los bloques
clean_points = [f"blocks.{i}" for i in range(4)]
with cache_activations(model, clean_points) as clean_cache:
    with torch.no_grad():
        clean_logits, _ = model(clean_ids)

# Run corrupted: medir prediccion sin patch
with torch.no_grad():
    corrupted_logits, _ = model(corrupted_ids)

# Identificar el target: token donde la prediccion DIFIERE
clean_pred = clean_logits[0, -1].argmax().item()
corrupted_pred = corrupted_logits[0, -1].argmax().item()
print(f"Clean prediction:     {tok.id_to_char[clean_pred]!r} (id={clean_pred})")
print(f"Corrupted prediction: {tok.id_to_char[corrupted_pred]!r} (id={corrupted_pred})")

if clean_pred == corrupted_pred:
    print("\nADVERTENCIA: clean y corrupted predicen lo mismo. Patching no informativo.")
    print("Buscando la mayor diferencia en logits...")
    diff = (clean_logits[0, -1] - corrupted_logits[0, -1])
    target_id = diff.argmax().item()
    print(f"Token con mayor diff (clean - corrupted): {tok.id_to_char[target_id]!r}")
else:
    target_id = clean_pred

# Logit del target en clean y corrupted
clean_logit_target = clean_logits[0, -1, target_id].item()
corrupted_logit_target = corrupted_logits[0, -1, target_id].item()
diff = clean_logit_target - corrupted_logit_target
print(f"\nLogit del target {tok.id_to_char[target_id]!r} en:")
print(f"  clean:     {clean_logit_target:+.3f}")
print(f"  corrupted: {corrupted_logit_target:+.3f}")
print(f"  diff (clean - corrupted) = {diff:+.3f}\n")


def patching_score(layer, pos):
    """Patchea el output de blocks.{layer} en posicion pos del run corrupted con el de clean.
    Retorna % de recuperacion del logit del target."""
    name = f"blocks.{layer}"
    clean_act = clean_cache[name][:, pos:pos + 1, :]  # (1, 1, d_model)
    patched_logits = patch_activation(model, corrupted_ids, {name: (pos, clean_act)})
    patched_target = patched_logits[0, -1, target_id].item()
    if abs(diff) < 1e-6:
        return 0.0
    recovery = (patched_target - corrupted_logit_target) / diff
    return recovery * 100  # %


print("=== Activation patching: % de recovery por (layer, posicion) ===")
print("Cells > 50% son causales para la prediccion clean.\n")
prompt_chars = [c.replace("\n", "\\n") for c in clean_prompt]
print(f"        " + "".join(f"{c:>6}" for c in prompt_chars))
for layer in range(4):
    row = ""
    for pos in range(T):
        score = patching_score(layer, pos)
        row += f"  {score:>+5.0f}"
    print(f"block.{layer} {row}")

print("\nCells con recovery > 30%:")
for layer in range(4):
    for pos in range(T):
        score = patching_score(layer, pos)
        if score > 30:
            print(f"  block.{layer} pos.{pos} (token={prompt_chars[pos]!r})  recovery={score:+.1f}%")
