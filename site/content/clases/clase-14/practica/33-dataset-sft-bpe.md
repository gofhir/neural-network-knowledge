---
title: "33 - Dataset SFT-BPE: 4 tareas bilingues"
weight: 330
math: true
---

El BPE-base (cap 31) genera Shakespeare drift, igual que el char-base en [cap 22]({{< relref "22-base-model-no-instructions.md" >}}). La diferencia llega con SFT. Antes de entrenar necesitamos el dataset — 4 tareas que aprovechan el nuevo tokenizador. En cap 34 veremos que pasa cuando lo entrenamos.

---

## 1. Las 4 tareas — tabla y razonamiento

| Tarea | Plantilla prompt | Generador | n |
|---|---|---|---|
| `qa` | `Q: PREGUNTA?\nA: ` | ~19 facts bilingues (EN + ES) | 1000 |
| `repeat` | `INSTR: repeat 'X' N\nRESP: ` | X char, N en {two,three,four} | 1000 |
| `complete-en` | `EN: 'FRASE'\nNEXT: ` | ultima palabra de linea Shakespeare | 1000 |
| `complete-es` | `ES: 'FRASE'\nNEXT: ` | ultima palabra de linea Quijote | 1000 |

Las cuatro tareas suman 5000 pares (4000 train + 1000 eval). Cada tarea aporta exactamente la misma cantidad de ejemplos para que ninguna domine el fine-tuning.

### Relacion con Camino 2 (char-level)

`qa` y `repeat` son identicas en estructura a las del dataset char-level (cap 23) — sirven como comparacion directa: si el modelo BPE-SFT aprende estas tareas igual de bien que el char-level, el tokenizador no introduce regresion. Si aprende mejor, el vocabulario mas rico esta ayudando.

`complete-en` y `complete-es` son **nuevas** — no existen en Camino 2. El motivo es tecnico: estas tareas requieren predecir una **palabra real** dada la mitad de una frase. Con char-level, predecir `withal,` de `We do instate and widow you` son **7 predicciones char independientes**, sin nocion de que esos 7 caracteres forman una sola palabra. Con BPE, `withal,` puede ser 1-2 tokens. El modelo aprende la semantica de la palabra completa en un solo paso de prediccion. Eso es exactamente lo que habilita BPE.

---

## 2. Por que `complete-*` es una tarea "LLM real"

Las tareas de completacion de texto (text completion) son el objetivo de pretraining de todos los LLMs grandes. GPT-4 fue preentrenado exactamente con "dado el contexto, predice el siguiente token". Ese entrenamiento da al modelo conocimiento del mundo: sabe que `Paris` sigue a `la capital de Francia es`, que `Hamlet` sigue a `who wrote`, que `withal` puede seguir a `We do instate and widow you` porque lo vio miles de veces en texto shakespeareano.

SFT convierte ese modelo de completacion en uno de instruccion: en vez de predecir el siguiente token de texto libre, aprende que cuando el prompt tiene una estructura especifica (`EN: '...' NEXT: `) debe generar la siguiente palabra especifica. Aqui hacemos lo mismo a escala miniatura: el BPE-base aprendio a predecir el siguiente token durante pretrain (cap 31), SFT le ensena que cuando el prompt dice `EN: '...' NEXT: ` debe generar la siguiente palabra, no Shakespeare drift.

La diferencia con `qa` y `repeat` es que `complete-*` depende del conocimiento adquirido en pretrain. El modelo tiene que "recordar" que despues de ese contexto Shakespeare viene esa palabra especifica. `qa` y `repeat` son patrones simples que se aprenden desde cero en SFT. `complete-*` aprovecha la representacion que BPE construyo durante pretrain sobre los dos corpus.

---

## 3. El dataset bilingue — por que EN y ES

Quijote y Shakespeare son los dos corpus que usamos para el pretrain BPE. El modelo ya vio ambos idiomas durante pretraining — tiene representacion de palabras EN y ES en el mismo espacio de embeddings. Si solo pusieramos tareas EN, estariamos sub-utilizando el bilingualismo del modelo.

Con `complete-es`, el SFT aprovecha que el BPE-base ya "sabe" espanol (aunque no haya aprendido a seguir instrucciones en el). El tokenizador BPE fue entrenado sobre ambos corpus, por lo que palabras como `gente`, `consejo`, `farsantes` son tokens reconocidos. El modelo no las aprende por primera vez en SFT — las reconoce del pretrain y SFT le ensena a generarlas en el contexto correcto.

El dataset QA tambien es bilingue: incluye preguntas en ingles (`Q: who wrote Hamlet?`) y en espanol (`Q: quien escribio Hamlet?`). El modelo tiene que aprender que `Shakespeare\n` es la respuesta correcta en ambos casos, aunque el prompt este en idiomas distintos. Esto seria imposible con un tokenizador solo ingles — con BPE bilingue es natural.

---

## 4. El script

`clase_14/practica/33_build_sft_bpe.py`:

```python
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
```

---

## 5. Output del script

```text
[train] kept=4000 dropped=0 dist={'qa': 1000, 'repeat': 1000, 'complete-en': 1000, 'complete-es': 1000}
[eval] kept=1000 dropped=0 dist={'qa': 250, 'repeat': 250, 'complete-en': 250, 'complete-es': 250}
```

`dropped=0` en ambos splits. El filtro `vocab_ok` descarta ejemplos cuyo prompt o response contiene caracteres fuera del vocab BPE. Como el BPE fue entrenado sobre los mismos corpus (Shakespeare + Quijote), todos los caracteres que aparecen en lineas de esos textos estan cubiertos. Si se usara un corpus nuevo con caracteres especiales o Unicode fuera del rango de entrenamiento, `dropped` seria mayor que cero.

---

## 6. Ejemplos concretos

Los cuatro ejemplos a continuacion son datos reales del dataset (uno por tarea):

```jsonl
{"prompt": "EN: 'We do instate and widow you'\nNEXT: ", "response": "withal,\n", "task": "complete-en"}
{"prompt": "ES: 'mi consejo, que es que nunca se tome con farsantes, que es'\nNEXT: ", "response": "gente\n", "task": "complete-es"}
{"prompt": "Q: who wrote Romeo and Juliet?\nA: ", "response": "Shakespeare\n", "task": "qa"}
{"prompt": "INSTR: repeat 'd' two\nRESP: ", "response": "dd\n", "task": "repeat"}
```

**complete-en:** `We do instate and widow you` → `withal,` — una linea real de Shakespeare. El modelo necesita saber que `withal` sigue a ese contexto especifico. En char-level, eso requiere predecir siete chars independientes: `w`, `i`, `t`, `h`, `a`, `l`, `,`. Con BPE, `withal,` puede ser un token o dos — una sola decision de prediccion que captura la palabra completa.

**complete-es:** `mi consejo, que es que nunca se tome con farsantes, que es` → `gente` — una linea real del Quijote. El modelo aprovecha que `gente` es un token BPE reconocido del pretrain sobre el corpus espanol. La frase es larga (62 chars) pero cabe dentro del rango `[25, 65]` del extractor.

**qa:** `Q: who wrote Romeo and Juliet?\nA: ` → `Shakespeare\n` — identico al dataset char-level de cap 23. La comparacion entre el accuracy de este modelo y el char-level en esta tarea es directa.

**repeat:** `INSTR: repeat 'd' two\nRESP: ` → `dd\n` — identico en estructura al dataset char-level. Los numerales word-form (`two`, `three`, `four`) en vez de digitos (`2`, `3`, `4`) evitan el riesgo de OOV: el vocab BPE puede no tener el digito `2` como token independiente (puede estar fusionado en bigrams o no aparecer), pero `two` es una palabra comun en ingles que seguramente tiene token propio.

---

## 7. Preguntas de verificacion

**1. Por que `complete-en` y `complete-es` son mas utiles con BPE que con char-level?**

Con BPE, la siguiente palabra es 1-2 tokens: el modelo aprende semantica de palabras completas en una sola prediccion. Con char-level, predecir `withal` son 7 predicciones char independientes sin ninguna nocion de que esos caracteres forman una unidad lexica. El modelo char-level puede aprender a copiar el patron si lo ve suficientes veces, pero no generaliza — cada nueva palabra es 5-8 predicciones char independientes. Con BPE, la generalizacion es al nivel de palabra, no de caracter.

**2. Que pasa si un caracter del dataset no esta en el vocab BPE?**

La funcion `vocab_ok` verifica que cada caracter del prompt y la response este en `tok.vocab` (el conjunto de caracteres y merges del BPE). Si encuentra un caracter fuera del vocab, descarta el ejemplo (`dropped` incrementa). Aqui `dropped=0` porque el BPE fue entrenado exactamente sobre Shakespeare y Quijote — cubre todos sus caracteres. Si se agregara un tercer corpus con, por ejemplo, letras acentuadas no cubiertas, algunos ejemplos se descartarian.

**3. Por que usar numerales word-form (`two`/`three`/`four`) en repeat en vez de digitos (`2`/`3`/`4`)?**

El mismo motivo que en Camino 2 (cap 23): el vocab BPE puede no tener el digito `2` o `4` como token independiente. Los digitos suelen aparecer en el texto en contextos como `1623`, `II`, `Act 2` — el BPE puede haberlos fusionado en tokens de varios caracteres o pueden no aparecer solos con frecuencia suficiente para tener token propio. Las word forms `two`, `three`, `four` son palabras comunes en ingles con tokens propios garantizados en un BPE entrenado sobre texto literario ingles. El riesgo de OOV con digitos es real; con word forms es cero.

---

## 8. Lo que viene

En cap 34 entrenamos el SFT-BPE sobre este dataset. Veremos si BPE da mejor accuracy que char-level en las tareas compartidas (`qa`, `repeat`) y si los modelos BPE pueden aprender `complete-en` y `complete-es` — tareas que char-level no puede abordar de forma natural.

Volver al [hub de practica](..) o a la [Clase 14](../..).
