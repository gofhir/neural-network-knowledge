"""33_build_sft_bpe.py - Cap 33: dataset SFT-BPE 4 tareas bilingues.

Tareas: qa (bilingue), repeat (word numerals), complete-en, complete-es.
Genera 5000 pares (4000 train + 1000 eval) en data/sft_bpe_*.jsonl.
"""
import json, random
from pathlib import Path
from _bpe import BPETokenizer

SFT_BPE_SEED = 142
EVAL_BPE_SEED = 1242

tok = BPETokenizer.load("data/bpe_tokenizer.json")
en_text = Path("shakespeare.txt").read_text(encoding="utf-8")
es_text = Path("quijote.txt").read_text(encoding="utf-8")

LOWERCASE = "abcdefghijklmnopqrstuvwxyz"
NUM_WORDS = {2: "two", 3: "three", 4: "four"}

QA_FACTS = [
    ("Q: who wrote Hamlet?\nA: ", "Shakespeare\n"),
    ("Q: who wrote Macbeth?\nA: ", "Shakespeare\n"),
    ("Q: who wrote Don Quijote?\nA: ", "Cervantes\n"),
    ("Q: who wrote Romeo and Juliet?\nA: ", "Shakespeare\n"),
    ("Q: who wrote King Lear?\nA: ", "Shakespeare\n"),
    ("Q: what is the capital of France?\nA: ", "Paris\n"),
    ("Q: what is the capital of Spain?\nA: ", "Madrid\n"),
    ("Q: what is the capital of Italy?\nA: ", "Rome\n"),
    ("Q: what is the capital of England?\nA: ", "London\n"),
    ("Q: what is two plus two?\nA: ", "four\n"),
    ("Q: what is three plus three?\nA: ", "six\n"),
    ("Q: what is five minus two?\nA: ", "three\n"),
    ("Q: quien escribio Don Quijote?\nA: ", "Cervantes\n"),
    ("Q: quien escribio Hamlet?\nA: ", "Shakespeare\n"),
    ("Q: cual es la capital de Francia?\nA: ", "Paris\n"),
    ("Q: cual es la capital de Espana?\nA: ", "Madrid\n"),
    ("Q: cual es la capital de Italia?\nA: ", "Roma\n"),
    ("Q: cuanto es dos mas dos?\nA: ", "cuatro\n"),
    ("Q: cuanto es tres mas tres?\nA: ", "seis\n"),
]

def gen_qa(rng):
    p, r = rng.choice(QA_FACTS)
    return {"prompt": p, "response": r, "task": "qa"}

def gen_repeat(rng):
    c = rng.choice(LOWERCASE)
    n = rng.choice([2, 3, 4])
    return {"prompt": f"INSTR: repeat '{c}' {NUM_WORDS[n]}\nRESP: ",
            "response": f"{c * n}\n", "task": "repeat"}

def extract_complete(text, lang_tag, rng, n, min_len=25, max_len=65):
    lines = [l.strip() for l in text.split("\n")
             if min_len <= len(l.strip()) <= max_len]
    rng.shuffle(lines)
    examples = []
    for line in lines:
        words = line.split()
        if len(words) < 4:
            continue
        target = words[-1]
        context = " ".join(words[:-1])
        prompt = f"{lang_tag}: '{context}'\nNEXT: "
        response = f"{target}\n"
        # Verificar que todos los chars estan en vocab BPE
        if all(c in tok.vocab for c in prompt + response):
            examples.append({"prompt": prompt, "response": response,
                             "task": f"complete-{lang_tag.lower()}"})
        if len(examples) >= n:
            break
    return examples

def vocab_ok(ex):
    return all(c in tok.vocab for c in ex["prompt"] + ex["response"])

def main():
    Path("data").mkdir(exist_ok=True)
    for split, n_each, n_complete, fout, seed in [
        ("train", 1000, 1000, "data/sft_bpe_dataset.jsonl", SFT_BPE_SEED),
        ("eval",  250,  250,  "data/sft_bpe_eval.jsonl",    EVAL_BPE_SEED),
    ]:
        rng = random.Random(seed)
        examples = []
        for _ in range(n_each): examples.append(gen_qa(rng))
        for _ in range(n_each): examples.append(gen_repeat(rng))
        examples += extract_complete(en_text, "EN", rng, n_complete)
        examples += extract_complete(es_text, "ES", rng, n_complete)

        before = len(examples)
        examples = [ex for ex in examples if vocab_ok(ex)]
        dropped = before - len(examples)

        per_task = {}
        for ex in examples:
            per_task[ex["task"]] = per_task.get(ex["task"], 0) + 1

        with open(fout, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        print(f"[{split}] kept={len(examples)} dropped={dropped} dist={per_task}")

if __name__ == "__main__":
    main()
