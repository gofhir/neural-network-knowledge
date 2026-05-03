"""46_dataset_lang.py - Cap 46: dataset EN/ES para deteccion de idioma."""
import json, random, torch
from pathlib import Path
from _bpe import BPETokenizer

SEED = 246
random.seed(SEED); torch.manual_seed(SEED)

tok = BPETokenizer.load("data/bpe_tokenizer.json")
tok.add_special_tokens()

WINDOW = 64  # tokens por ejemplo (sin [CLS][SEP])

en_text = Path("shakespeare.txt").read_text(encoding="utf-8")
es_text = Path("quijote.txt").read_text(encoding="utf-8")
en_tokens = tok.encode(en_text)
es_tokens = tok.encode(es_text)

def sample_windows(tokens, n, label, split_offset=0):
    rng = random.Random(SEED + label + split_offset)
    examples = []
    for _ in range(n):
        start = rng.randint(0, len(tokens) - WINDOW - 1)
        window = tokens[start:start + WINDOW]
        full = [tok.cls_id] + window + [tok.sep_id]
        examples.append({"ids": full, "label": label})
    return examples

Path("data").mkdir(exist_ok=True)

for offset, (split, n_each, fout) in enumerate([
    ("train", 1000, "data/lang_train.jsonl"),
    ("eval",   250, "data/lang_eval.jsonl"),
]):
    examples = sample_windows(en_tokens, n_each, 0, split_offset=offset * 100) + \
               sample_windows(es_tokens, n_each, 1, split_offset=offset * 100)
    random.shuffle(examples)
    with open(fout, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"[{split}] {len(examples)} ejemplos ({n_each} EN + {n_each} ES) -> {fout}")

print("\nEjemplos del train set:")
with open("data/lang_train.jsonl") as f:
    for line in list(f)[:2]:
        ex = json.loads(line)
        decoded = tok.decode(ex["ids"])
        lang = "EN" if ex["label"] == 0 else "ES"
        print(f"  [{lang}] {decoded[:60]!r}...")
