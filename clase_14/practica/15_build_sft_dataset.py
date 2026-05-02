"""15_build_sft_dataset.py - Cap 23: dataset SFT char-level.

4 tareas deterministas con vocab subset Shakespeare. Genera 5000 pares
(4000 train + 1000 eval) en data/sft_dataset.jsonl + data/sft_eval.jsonl.
"""
import json
import random
from pathlib import Path

SFT_SEED = 42
EVAL_SEED = 4242

LOWERCASE = "abcdefghijklmnopqrstuvwxyz"


def gen_word(rng, min_len=2, max_len=6):
    n = rng.randint(min_len, max_len)
    return "".join(rng.choices(LOWERCASE, k=n))


def gen_reverse(rng):
    w = gen_word(rng)
    return {"prompt": f"INSTR: reverse '{w}'\nRESP: ",
            "response": f"{w[::-1]}\n", "task": "reverse"}


def gen_upper(rng):
    w = gen_word(rng)
    return {"prompt": f"INSTR: upper '{w}'\nRESP: ",
            "response": f"{w.upper()}\n", "task": "upper"}


NUM_WORDS = {2: "two", 3: "three", 4: "four"}


def gen_repeat(rng):
    c = rng.choice(LOWERCASE)
    n = rng.choice([2, 3, 4])
    return {"prompt": f"INSTR: repeat '{c}' {NUM_WORDS[n]}\nRESP: ",
            "response": f"{c * n}\n", "task": "repeat"}


# QA_FACTS: facts curados, vocab-safe (solo chars del Shakespeare vocab)
QA_FACTS = [
    ("who wrote Hamlet?", "Shakespeare"),
    ("who wrote Macbeth?", "Shakespeare"),
    ("who wrote Othello?", "Shakespeare"),
    ("who wrote King Lear?", "Shakespeare"),
    ("who wrote Romeo and Juliet?", "Shakespeare"),
    ("who wrote Don Quijote?", "Cervantes"),
    ("who wrote La Galatea?", "Cervantes"),
    ("who wrote La Iliada?", "Homer"),
    ("who wrote La Odisea?", "Homer"),
    ("who wrote Faust?", "Goethe"),
    ("who wrote The Divine Comedy?", "Dante"),
    ("who wrote War and Peace?", "Tolstoy"),
    ("who wrote Crime and Punishment?", "Dostoyevsky"),
    ("who wrote Les Miserables?", "Hugo"),
    ("who wrote Madame Bovary?", "Flaubert"),
    ("what is the capital of France?", "Paris"),
    ("what is the capital of Spain?", "Madrid"),
    ("what is the capital of Italy?", "Rome"),
    ("what is the capital of Germany?", "Berlin"),
    ("what is the capital of England?", "London"),
    ("what is the capital of Portugal?", "Lisbon"),
    ("what is the capital of Greece?", "Athens"),
    ("what is the capital of Russia?", "Moscow"),
    ("what is the capital of Japan?", "Tokyo"),
    ("what is the capital of China?", "Beijing"),
    ("what is two plus two?", "four"),
    ("what is three plus three?", "six"),
    ("what is five plus five?", "ten"),
    ("what is ten minus three?", "seven"),
    ("what is two times three?", "six"),
]


def gen_qa(rng):
    q, a = rng.choice(QA_FACTS)
    return {"prompt": f"Q: {q}\nA: ", "response": f"{a}\n", "task": "qa"}


def vocab_filter_ok(ex, vocab_chars):
    return all(c in vocab_chars for c in ex["prompt"] + ex["response"])


def main():
    text = Path("shakespeare.txt").read_text()
    vocab = set(text)
    print(f"Vocab base: {len(vocab)} chars")

    rng = random.Random(SFT_SEED)
    eval_rng = random.Random(EVAL_SEED)

    Path("data").mkdir(exist_ok=True)

    for split, n_per_task, n_qa, fout, r in [
        ("train", 1000, 1000, "data/sft_dataset.jsonl", rng),
        ("eval",  250,  250,  "data/sft_eval.jsonl",    eval_rng),
    ]:
        examples = []
        for _ in range(n_per_task):
            examples.append(gen_reverse(r))
        for _ in range(n_per_task):
            examples.append(gen_upper(r))
        for _ in range(n_per_task):
            examples.append(gen_repeat(r))
        for _ in range(n_qa):
            examples.append(gen_qa(r))

        before = len(examples)
        examples = [ex for ex in examples if vocab_filter_ok(ex, vocab)]
        dropped = before - len(examples)

        with open(fout, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        per_task = {}
        for ex in examples:
            per_task[ex["task"]] = per_task.get(ex["task"], 0) + 1
        print(f"[{split}] kept={len(examples)} dropped={dropped} dist={per_task}")


if __name__ == "__main__":
    main()
