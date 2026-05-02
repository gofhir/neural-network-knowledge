---
title: "23 - Dataset SFT: 4 tareas sinteticas"
weight: 230
math: true
---

En el [cap 22](../22-base-model-no-instructions) vimos que el base model ignora instrucciones — recibe `INSTR: reverse 'cat'\nRESP: ` y devuelve Shakespeare-ish. Para fine-tunearlo necesitamos un dataset de pares (instruccion, respuesta). No tenemos uno a mano, asi que lo construimos sintetico: 4 tareas, 5000 ejemplos totales (4000 train + 1000 eval), todo char-level y compatible con el vocab de Shakespeare. Este capitulo describe el dataset; en el [cap 24](../24-sft-training) lo usamos para entrenar.

---

## 1. Las 4 tareas

Elegimos 4 tareas que cubren dos pedagogias distintas: **funciones determinis ticas** (3 de ellas) y **memorizacion de hechos** (la cuarta). El detalle por tarea:

| Tarea | Plantilla prompt | Generador | Vocab que requiere |
|---|---|---|---|
| reverse | `INSTR: reverse 'WORD'\nRESP: ` | `WORD[::-1]` | lowercase + comillas |
| upper | `INSTR: upper 'WORD'\nRESP: ` | `WORD.upper()` | lowercase + UPPERCASE |
| repeat | `INSTR: repeat 'X' N\nRESP: ` | `X * N`, N en {two, three, four} | lowercase + word numerals |
| qa | `Q: PREGUNTA?\nA: ` | tabla de 30 facts curados | depende del fact |

`reverse`, `upper` y `repeat` son funciones puras de su input. `qa` no — es una tabla fija de 30 pares pregunta/respuesta sobre autores y capitales (ver el script abajo).

---

## 2. La trampa del vocab — un detalle real

Cuando empece a generar el dataset, el filtro de vocab elimino la mitad de los ejemplos `repeat`. Investigue: el Shakespeare vocab tiene exactamente UN digito (`3`, que aparece una sola vez en el corpus en algun numero de escena o linea) y ningun otro. Los caracteres `2` y `4` simplemente **no estan** en el alfabeto del modelo. Cuando generaba `INSTR: repeat 'a' 2\nRESP: `, el filtro lo descartaba porque `2` no era encodeable.

La solucion fue trivial pero ilustrativa: usar palabras en vez de digitos. `repeat 'a' two`, `repeat 'a' three`, `repeat 'a' four`. Asi todos los chars del prompt y de la respuesta ya estan en el vocab y nada se cae al filtrar.

Es una decision pequeña pero ilustra algo real: cuando construyes un dataset para un modelo char-level, tienes que respetar **su** vocab. Esto no aparece con tokenizadores BPE (que pueden inventar tokens nuevos) pero aqui es ineludible — si el char no esta en el vocab, el modelo no lo puede ni leer ni escribir. Es el tipo de friccion que solo aparece cuando uno construye el sistema de punta a punta.

---

## 3. El script

El script completo, `15_build_sft_dataset.py`:

```python
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
```

Lectura rapida: cuatro generadores `gen_*` que devuelven dicts `{prompt, response, task}`, una lista `QA_FACTS` curada manualmente, un filtro de vocab que descarta cualquier ejemplo con chars fuera del alfabeto Shakespeare, y un loop principal que genera train + eval con seeds distintas.

---

## 4. La distribucion

Corrimos el script y produjo:

```
Vocab base: 65 chars
[train] kept=4000 dropped=0 dist={'reverse': 1000, 'upper': 1000, 'repeat': 1000, 'qa': 1000}
[eval]  kept=1000 dropped=0 dist={'reverse': 250, 'upper': 250, 'repeat': 250, 'qa': 250}
```

`dropped=0` confirma que la adaptacion del vocab funciono — ningun ejemplo se perdio al filtrar. Las 4 tareas estan **balanceadas** a 1000/1000/1000/1000 en train y 250/250/250/250 en eval. Eval usa una seed distinta (`4242`) para que los ejemplos no se solapen con los de train; con palabras de 2-6 caracteres aleatorias, la chance de colision real es despreciable.

---

## 5. Ejemplos concretos

Asi se ve el dataset, una linea JSONL por ejemplo, una tarea de cada tipo:

```jsonl
{"prompt": "INSTR: reverse 'ah'\nRESP: ", "response": "ha\n", "task": "reverse"}
{"prompt": "INSTR: upper 'pnwunl'\nRESP: ", "response": "PNWUNL\n", "task": "upper"}
{"prompt": "INSTR: repeat 'd' two\nRESP: ", "response": "dd\n", "task": "repeat"}
{"prompt": "Q: what is the capital of Germany?\nA: ", "response": "Berlin\n", "task": "qa"}
```

Notar el `\n` al final de cada `response`. Es importante: durante entrenamiento, el `\n` actua como token de fin de respuesta, y al inferir el modelo aprende a parar ahi. Sin ese delimitador, el modelo no sabe cuando dejar de generar.

---

## 6. Q&A: la memorizacion deliberada

`qa` es distinta a las otras 3 tareas. `reverse`, `upper` y `repeat` son funciones determinis ticas que el modelo puede **aprender** — hay un mapeo input→output sistematico, y con suficientes ejemplos el modelo extrae la regla. Si entrenamos `reverse` con miles de palabras distintas, el modelo aprende a invertir cualquier string, incluyendo strings que nunca vio.

`qa` es **memorizacion** pura. Hay 30 facts, 1000 ejemplos en train, y cada fact aparece en promedio ~33 veces. No hay regla que aprender — `Hamlet` no tiene una "funcion" que devuelva `Shakespeare`, simplemente esta en una tabla. El modelo memoriza la tabla.

Eso es el punto. SFT puede memorizar tablas de hechos, y eso es exactamente lo que hace ChatGPT cuando "recuerda" que la capital de Francia es Paris o que Shakespeare escribio Hamlet. No hay razonamiento — hay co-ocurrencia memorizada en parametros. Tenerlo claro en este dataset chico nos da intuicion sobre que es y que no es la "memoria" de un LLM.

---

## 7. Lo que viene

En el [cap 24](../24-sft-training) entrenamos el SFT loop con **loss masking** sobre los tokens de respuesta. Es la parte tecnicamente mas sutil del Camino 2: solo penalizamos al modelo por generar la respuesta, no por copiar el prompt. Sin ese masking, el loss se domina por la reproduccion del prompt y el modelo aprende a copiar `INSTR: reverse 'cat'\nRESP: ` mejor que a generar `tac\n`.

Volver al [hub de practica](..) o a la [Clase 14](../..).
