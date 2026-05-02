---
title: "35 - Dataset DPO-BPE: rejected linguisticamente ricos"
weight: 350
math: true
---

## 1. Apertura

En el [capitulo 28]({{< relref "28-dataset-dpo" >}}) (Camino 2, char-level) construimos un dataset DPO con la misma receta que vamos a usar aqui: 1500 triples base-sampled + 1500 cross-task = 3000 triples. La diferencia clave esta en **la calidad de los rejected**.

El char-level base-model del Camino 2 generaba chars aleatorios localmente coherentes — fragmentos como `"alast the king, there is be doth"` que parecen ingles pero no terminan de cuajar en gramatica ni semantica. El BPE-base (Camino 2.5) genera **palabras reales, oraciones completas, formato literario**: los rejected son fragmentos de Shakespeare/Quijote autenticos pero incorrectos para el prompt dado. Esa diferencia da una senal de preferencia mucho mas limpia y mas dura para el optimizador DPO.

En este capitulo construimos el dataset; en cap 36 entrenamos la policy DPO sobre el (con dos valores de beta para validar la hipotesis del cap 29).

---

## 2. La estructura del dataset

El diseno es identico al cap 28 — solo cambia el tokenizador. Tres elementos por triple: `prompt`, `chosen`, `rejected`. Dos fuentes complementarias de rejected:

- **Base-sampled (1500 triples)**: para cada prompt SFT, sampleamos del Mini-LLaMA BPE-base (pre-SFT, cap 31). El base genera Shakespeare-like sin seguir el formato INSTR/RESP o Q/A. `chosen` = respuesta correcta del SFT (cap 33), `rejected` = output del base.
- **Cross-task (1500 triples)**: para un prompt de tarea A, `chosen` = respuesta correcta de A, `rejected` = respuesta correcta de OTRA tarea B (sampleada del mismo SFT). Aprende: "no respondas con el formato de otra tarea".

Total 3000 triples balanceados 50/50. Mismo `chosen` siempre — la respuesta SFT correcta. Lo que cambia entre fuentes es como se elige el `rejected`.

---

## 3. La diferencia clave — rejected linguisticamente ricos

Esta es la seccion central del capitulo. Comparemos lado a lado un triple de cada nivel:

**Char-level rejected (cap 28, Camino 2):**

```
prompt:   "INSTR: reverse 'oz'\nRESP: "
chosen:   'zo\n'
rejected: 'I may, nothing? she,\n'        <- chars coherentes, mucha incoherencia local
```

**BPE-level rejected (cap 35, Camino 2.5):**

```
prompt:   "INSTR: repeat 'y' two\nRESP: "
chosen:   'yy\n'
rejected: 'let them not mine honour, as thou hast thou arty.\n\nKING \n'   <- oracion COMPLETA
```

El BPE-base genera oraciones reales con sujetos, verbos, gramatica. `'let them not mine honour, as thou hast thou arty'` es Shakespeare fragmentado pero con palabras enteras y estructura sintactica coherente. El char-base solo generaba secuencias que parecian palabras pero sin gramatica clara — el modelo char-level sabia que `'th'` y `'ing'` son n-gramas frecuentes, pero no sabia componer una frase entera.

**Por que importa para DPO**. La loss DPO entrena la policy a preferir `chosen` sobre `rejected`. Si el rejected es facil de distinguir (gibberish, chars aleatorios), la policy aprende rapido pero la senal es debil — solo aprende **"esto es texto, esto no"**. Si el rejected es DIFICIL (texto fluido pero del prompt equivocado), la policy aprende algo mas sutil — **"sigue ESTE formato especifico, no aquel"**.

Ese es el escenario realista de DPO en produccion: ambos `chosen` y `rejected` son linguisticamente plausibles, y la preferencia es la unica senal. Con BPE estamos mas cerca de ese escenario que con char-level. La hipotesis es que tambien deberia exigir un `beta` mas alto para que la policy se aleje del referencia con suficiente fuerza — lo verificamos en el cap 36.

---

## 4. El script

`clase_14/practica/35_build_dpo_bpe.py`:

```python
"""35_build_dpo_bpe.py - Cap 35: dataset DPO-BPE mix (base-sampled + cross-task).

3000 triples = 1500 base-sampled (del BPE-base) + 1500 cross-task.
"""
import json, random, torch
from pathlib import Path
from _bpe import BPETokenizer
from _models import load_pretrained_mini_llama, generate_with_prompt
from _eval import load_jsonl

DPO_BPE_SEED = 143
torch.manual_seed(DPO_BPE_SEED)
random.seed(DPO_BPE_SEED)

tok = BPETokenizer.load("data/bpe_tokenizer.json")
vocab = set(tok.vocab.keys())
vocab_size = tok.vocab_size
cfg = dict(vocab_size=vocab_size, max_seq_len=256,
           d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384)

print("Cargando BPE base model...")
base_model = load_pretrained_mini_llama("checkpoints/mini_llama_bpe_base.pt", config=cfg)

sft = load_jsonl("data/sft_bpe_dataset.jsonl")
rng = random.Random(DPO_BPE_SEED)
rng.shuffle(sft)

triples = []

print("Generando 1500 triples base-sampled (esto toma ~5-7 min)...")
for i, ex in enumerate(sft[:1500]):
    rej_full = generate_with_prompt(base_model, ex["prompt"], tok,
                                    max_new_tokens=20, temperature=0.8,
                                    top_k=10, stop_token="\n")
    rejected = rej_full[len(ex["prompt"]):]
    if not rejected.endswith("\n"):
        rejected += "\n"
    if rejected == ex["response"]:
        continue
    triples.append({"prompt": ex["prompt"], "chosen": ex["response"],
                    "rejected": rejected, "source": "base"})
    if (i + 1) % 200 == 0:
        print(f"  base-sampled: {i+1}/1500, aceptados: {len(triples)}", flush=True)

print(f"\nBase-sampled: {len(triples)} triples")

print("Generando 1500 triples cross-task...")
by_task = {}
for ex in sft:
    by_task.setdefault(ex["task"], []).append(ex)

base_count = len(triples)
for ex in sft[1500:3000]:
    other_tasks = [t for t in by_task if t != ex["task"]]
    other_task = rng.choice(other_tasks)
    other_ex = rng.choice(by_task[other_task])
    if other_ex["response"] == ex["response"]:
        continue
    triples.append({"prompt": ex["prompt"], "chosen": ex["response"],
                    "rejected": other_ex["response"], "source": "cross"})

print(f"Cross-task: {len(triples) - base_count} triples")

def vocab_ok(t):
    return all(c in vocab for c in t["prompt"] + t["chosen"] + t["rejected"])

before = len(triples)
triples = [t for t in triples if vocab_ok(t)]
print(f"Filtered by vocab: {before - len(triples)} dropped")

with open("data/dpo_bpe_dataset.jsonl", "w", encoding="utf-8") as f:
    for t in triples:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")

by_source = {}
for t in triples:
    by_source[t["source"]] = by_source.get(t["source"], 0) + 1
print(f"\nTotal: {len(triples)}  by_source: {by_source}")
```

Notar tres detalles:

- **Seed `DPO_BPE_SEED = 143`** distinta del cap 28 (43) — el sampling del BPE-base es propio de este capitulo y queremos reproducibilidad independiente.
- **`max_new_tokens=20`**: en BPE 20 tokens son ~50-80 caracteres aprox. (depende del merge), suficiente para que el rejected tenga forma de oracion sin descontrolarse en longitud.
- **`vocab_ok`**: filtro de seguridad. El prompt y respuesta vienen del SFT-BPE (cap 33), que se construyo con este mismo tokenizador, asi que el filtro deberia descartar cero.

---

## 5. Output literal de la corrida

```
Cargando BPE base model...
Generando 1500 triples base-sampled (esto toma ~5-7 min)...
  base-sampled: 200/1500, aceptados: 200
  base-sampled: 400/1500, aceptados: 400
  base-sampled: 600/1500, aceptados: 600
  base-sampled: 800/1500, aceptados: 800
  base-sampled: 1000/1500, aceptados: 1000
  base-sampled: 1200/1500, aceptados: 1200
  base-sampled: 1400/1500, aceptados: 1400

Base-sampled: 1500 triples
Generando 1500 triples cross-task...
Cross-task: 1500 triples
Filtered by vocab: 0 dropped

Total: 3000  by_source: {'base': 1500, 'cross': 1500}
```

3000 triples balanceados 50/50, 0 dropped por chosen-igual-rejected (en BPE el base genera Shakespeare casi siempre, no factoides), 0 dropped por vocab (esperado — mismo tokenizador para todo).

---

## 6. Ejemplos concretos analizados

Cuatro spot-checks reales tomados de `data/dpo_bpe_dataset.jsonl`:

```
[base]   prompt:   "INSTR: repeat 'y' two\nRESP: "
         chosen:   'yy\n'
         rejected: 'let them not mine honour, as thou hast thou arty.\n\nKING \n'

[base]   prompt:   'Q: what is the capital of Spain?\nA: '
         chosen:   'Madrid\n'
         rejected: "find that the king Petruchio's hands.\n\nGRU\n"

[cross]  prompt:   "ES: 'libertad de Don Gregorio, y de otros'\nNEXT: "
         chosen:   'sucesos\n'
         rejected: 'nn\n'

[cross]  prompt:   "INSTR: repeat 'q' four\nRESP: "
         chosen:   'qqqq\n'
         rejected: 'mi\n'
```

Analisis ejemplo por ejemplo:

- **`repeat 'y' two`** (base). `chosen='yy'`, `rejected=` Shakespeare libre con marca de personaje (`KING `). Senal de DPO: **"no me des prosa con personajes y honor; dame `yy`"**. El rejected es linguisticamente impecable — gramatica correcta, vocabulario shakespeariano — pero totalmente fuera del formato de la tarea repeat.
- **`Q: capital of Spain?`** (base). `chosen='Madrid'`, `rejected="find that the king Petruchio's hands."`. El BPE-base sabe Shakespeare (Petruchio es de *The Taming of the Shrew*) pero no sabe geografia factual. DPO aprende: **"no me hables de Petruchio, dime Madrid"**. Es una senal mas dificil que en char-level — el rejected es una oracion completa, gramatical, con un nombre propio real. La policy tiene que aprender a discriminar formato (Q/A factual vs continuacion narrativa) sobre algo que linguisticamente compite con el chosen.
- **`ES: '...y de otros' NEXT:`** (cross). `chosen='sucesos'` (palabra real del Quijote — la continuacion correcta del fragmento), `rejected='nn'` (output tipico de la tarea repeat). Cross-task: **"no me des chars repetidos cuando te pido la siguiente palabra del Quijote"**.
- **`repeat 'q' four`** (cross). `chosen='qqqq'`, `rejected='mi'` (output de la tarea Q&A — quizas el final de "mi" como respuesta corta a una pregunta espanola). Cross-task: **"no me des un fragmento de palabra cuando te pido cuatro `q`s"**.

Estos cuatro casos cubren los dos tipos de senal que ensena DPO: **(1) seguir formato vs no seguirlo** (los dos base-sampled) y **(2) seguir LA tarea correcta vs otra tarea valida** (los dos cross-task).

---

## 7. Por que descartamos `chosen == rejected`

El script tiene una guarda: si `rejected == ex["response"]` durante el sampling base, descartamos el triple. La razon es matematica — la loss DPO depende del log-ratio entre la policy y el referencia evaluados sobre `chosen` menos los mismos sobre `rejected`. Si `chosen == rejected`, los dos terminos son identicos y el log-ratio es exactamente cero. La sigmoide queda en `0.5`, el gradiente es trivial, y el triple no aporta senal al optimizador (solo diluye el batch).

En char-level (cap 28) era posible: prompts faciles tipo `repeat 'a' one` -> `a` el base podia adivinarlos. En BPE-level es practicamente imposible — el BPE-base genera Shakespeare ~99% del tiempo, nunca produce respuestas correctas a tareas. Por eso aqui: **0 dropped**. Mantenemos el filtro por consistencia con cap 28 y porque cuesta nada.

---

## 8. Preguntas de verificacion

1. **Por que los rejected del BPE-base son mas "ricos" que los del char-base?** Porque el BPE-base genera tokens que ya son palabras o subpalabras enteras (`'thou'`, `'hast'`, `'KING'`), entonces compone oraciones gramaticales completas. El char-base solo generaba caracteres uno por uno — sabia que ciertos n-gramas eran frecuentes pero no podia mantener gramatica de oracion entera. La diferencia es estructural: BPE da rejected linguisticamente plausibles, char-level da rejected localmente coherentes pero globalmente fragmentados.

2. **Que ensena un triple cross-task?** Que cada formato de prompt corresponde a un formato de respuesta especifico. El triple `(repeat 'q' four, qqqq, mi)` ensena: cuando ves un prompt INSTR/RESP de repeat, no respondas con un fragmento corto de Q&A — respeta el formato de la tarea pedida. La policy aprende a no mezclar tareas.

3. **Por que filtrar `chosen == rejected`?** Porque en la loss DPO los terminos `log pi(chosen) - log pi(rejected)` y `log pi_ref(chosen) - log pi_ref(rejected)` se cancelan exactamente cuando los strings son iguales. El log-ratio queda en cero, la sigmoide en 0.5, y el ejemplo no aporta gradiente. Es ruido en el batch.

---

## 9. Lo que viene

En cap 36 entrenamos DPO sobre estos 3000 triples con DOS valores de `beta` (0.1 y 0.5) — uno bajo y uno alto — para validar la hipotesis del [cap 29]({{< relref "29-dpo-training-eval" >}}) de que `beta=0.1` fue demasiado bajo para que la policy se separara del referencia. Con rejected linguisticamente mas duros (este capitulo), la separacion necesaria es mayor, y esperamos que `beta=0.5` resulte claramente superior.
