"""Eval harness compartido para SFT y DPO."""
import json
import torch
from _models import generate_with_prompt


def build_char_maps(text):
    chars = sorted(set(text))
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for i, c in enumerate(chars)}
    return c2i, i2c


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def eval_exact_match(model, dataset_jsonl, char_to_id, id_to_char,
                     n_per_task=200, max_new_tokens=20, device=None, temperature=0.1):
    """Por cada tarea, generar respuesta y comparar exact match."""
    examples = load_jsonl(dataset_jsonl)
    by_task = {}
    for ex in examples:
        by_task.setdefault(ex["task"], []).append(ex)
    results = {}
    for task, items in by_task.items():
        sample = items[:n_per_task]
        correct = 0
        for ex in sample:
            full = generate_with_prompt(
                model, ex["prompt"], char_to_id, id_to_char,
                max_new_tokens=max_new_tokens, temperature=temperature,
                top_k=10, device=device, stop_token="\n",
            )
            generated = full[len(ex["prompt"]):].rstrip("\n")
            expected = ex["response"].rstrip("\n")
            if generated == expected:
                correct += 1
        results[task] = correct / len(sample)
    return results


def eval_qualitative(model, prompts, char_to_id, id_to_char,
                     n_samples=3, temperature=0.8, device=None):
    out = {}
    for p in prompts:
        out[p] = [
            generate_with_prompt(model, p, char_to_id, id_to_char,
                                 max_new_tokens=30, temperature=temperature,
                                 top_k=10, device=device, stop_token="\n")
            for _ in range(n_samples)
        ]
    return out


def eval_drift(model, ambiguous_prompts, char_to_id, id_to_char, device=None):
    """Heurística: % de samples que contienen palabras Shakespeare-style."""
    shakespeare_markers = ["thou", "thee", "thy", "hath", "doth", "ye", "O ", "wilt"]
    drift_count = 0
    total = 0
    for p in ambiguous_prompts:
        for _ in range(5):
            s = generate_with_prompt(model, p, char_to_id, id_to_char,
                                     max_new_tokens=40, temperature=0.8,
                                     top_k=20, device=device, stop_token="\n")
            comp = s[len(p):].lower()
            if any(m.lower() in comp for m in shakespeare_markers):
                drift_count += 1
            total += 1
    return drift_count / total if total else 0.0
