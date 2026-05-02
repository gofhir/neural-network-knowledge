"""17_eval_sft.py - Cap 25: eval comparativa Base vs SFT."""
import torch
from _models import load_pretrained_mini_llama
from _eval import build_char_maps, eval_exact_match, eval_qualitative, eval_drift
from _bpe import CharTokenizer

torch.manual_seed(1337)

text = open("shakespeare.txt").read()
c2i, i2c = build_char_maps(text)
tokenizer = CharTokenizer(c2i, i2c)

print("=== Eval Base vs SFT ===\n")
results = {}
for name, ckpt in [
    ("base", "checkpoints/mini_llama_base.pt"),
    ("sft",  "checkpoints/mini_llama_sft.pt"),
]:
    print(f"--- Evaluando {name} ---")
    model = load_pretrained_mini_llama(ckpt)
    em = eval_exact_match(model, "data/sft_eval.jsonl", tokenizer, n_per_task=200)
    results[name] = em
    print(f"exact_match: {em}\n")

print("=== Tabla comparativa ===")
print(f"{'task':<12}{'base':<10}{'sft':<10}")
for task in ["reverse", "upper", "repeat", "qa"]:
    b = results["base"].get(task, 0.0)
    s = results["sft"].get(task, 0.0)
    print(f"{task:<12}{b:<10.3f}{s:<10.3f}")

print("\n=== Eval cualitativo (SFT) ===")
prompts = [
    "INSTR: reverse 'house'\nRESP: ",
    "INSTR: upper 'world'\nRESP: ",
    "Q: who wrote Hamlet?\nA: ",
]
sft_model = load_pretrained_mini_llama("checkpoints/mini_llama_sft.pt")
qual = eval_qualitative(sft_model, prompts, tokenizer, n_samples=3)
for p, samples in qual.items():
    print(f"\nPrompt: {p!r}")
    for i, s in enumerate(samples):
        completion = s[len(p):].rstrip()
        print(f"  [{i}] {completion!r}")

print("\n=== Drift score (Shakespeare-style markers) ===")
ambiguous = ["INSTR: capitalize 'cat'\nRESP: ", "Q: what is 2+2?\nA: "]
for name, ckpt in [
    ("base", "checkpoints/mini_llama_base.pt"),
    ("sft",  "checkpoints/mini_llama_sft.pt"),
]:
    m = load_pretrained_mini_llama(ckpt)
    drift = eval_drift(m, ambiguous, tokenizer)
    print(f"{name}: drift = {drift:.3f}")
