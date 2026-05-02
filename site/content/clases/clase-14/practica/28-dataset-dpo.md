---
title: "28 - Dataset DPO: chosen + rejected"
weight: 280
math: true
---

## Apertura

En el [capitulo 27]({{< relref "27-dpo-loss" >}}) derivamos la loss DPO y vimos que requiere triples `(prompt, chosen, rejected)`. Ahora viene la pregunta practica: **como conseguimos esos datos sin un equipo de etiquetadores humanos?**

La respuesta corta: generamos las preferencias algoritmicamente, mezclando dos fuentes complementarias de "rejected". En este capitulo construimos el dataset; en el capitulo 29 entrenamos la policy DPO sobre el.

## La estrategia: mix de dos fuentes

Combinamos dos fuentes de rejected, cada una enseña algo distinto al modelo:

- **Base-sampled (1500 triples)**: para cada prompt, sampleamos del Mini-LLaMA BASE (pre-SFT). El base model tiende a generar Shakespeare drift porque solo vio el corpus de Shakespeare. Esto enseña a la policy: *no decaer al estilo del corpus pre-SFT*.

- **Cross-task (1500 triples)**: para cada prompt de tarea A, usamos como rejected una respuesta de OTRA tarea B (extraida del mismo dataset SFT). Esto enseña: *seguir LA INSTRUCCION dada, no otra*.

El `chosen` siempre es la respuesta correcta del dataset SFT (capitulo 23).

## Por que NO solo human labels (RLHF clasico)

RLHF clasico requiere humanos comparando respuestas: "A es mejor que B". Es caro, lento, y dificil de escalar. Para un curso (o cualquier proyecto pequeño) es directamente impracticable.

DPO funciona con CUALQUIER fuente de preferencias, incluyendo synthetic. Aqui usamos preferencias generadas algoritmicamente: lo correcto vs lo incorrecto. Es una forma valida de DPO — el paper original de Rafailov et al. demuestra que el metodo funciona sobre HH-RLHF (Human Helpful-Harmless), pero tambien con preferencias sinteticas para tareas estructuradas, que es exactamente nuestro caso (reverse, repeat, Q&A factual).

## El script

```python
"""20_build_dpo_dataset.py - Cap 28: dataset DPO mix (base-sampled + cross-task).

3000 triples = 1500 base-sampled + 1500 cross-task.
"""
import json
import random
import torch
from pathlib import Path
from _models import load_pretrained_mini_llama, generate_with_prompt
from _eval import build_char_maps, load_jsonl

DPO_SEED = 43
torch.manual_seed(DPO_SEED)
random.seed(DPO_SEED)

text = Path("shakespeare.txt").read_text()
c2i, i2c = build_char_maps(text)
vocab = set(c2i)

print("Cargando base model para sampling...")
base_model = load_pretrained_mini_llama("checkpoints/mini_llama_base.pt")

sft = load_jsonl("data/sft_dataset.jsonl")
rng = random.Random(DPO_SEED)
rng.shuffle(sft)

triples = []

# (1) Base-sampled: rejected = output del base model
print("Generando 1500 triples base-sampled (esto toma ~1-2 min)...")
for i, ex in enumerate(sft[:1500]):
    rejected_full = generate_with_prompt(
        base_model, ex["prompt"], c2i, i2c,
        max_new_tokens=20, temperature=0.8, top_k=10, stop_token="\n",
    )
    rejected = rejected_full[len(ex["prompt"]):]
    if not rejected.endswith("\n"):
        rejected += "\n"
    if rejected == ex["response"]:
        continue  # base acerto por casualidad — descartar
    triples.append({
        "prompt": ex["prompt"],
        "chosen": ex["response"],
        "rejected": rejected,
        "source": "base",
    })
    if (i + 1) % 200 == 0:
        print(f"  base-sampled: {i+1}/1500 procesados, {len(triples)} aceptados")

print(f"\nBase-sampled: {len(triples)} triples\n")

# (2) Cross-task: rejected = respuesta de OTRA tarea
print("Generando 1500 triples cross-task...")
by_task = {}
for ex in sft:
    by_task.setdefault(ex["task"], []).append(ex)

base_count = len(triples)
for ex in sft[1500:3000]:
    other_tasks = [t for t in by_task if t != ex["task"]]
    other_task = rng.choice(other_tasks)
    other_ex = rng.choice(by_task[other_task])
    rejected = other_ex["response"]
    if rejected == ex["response"]:
        continue
    triples.append({
        "prompt": ex["prompt"],
        "chosen": ex["response"],
        "rejected": rejected,
        "source": "cross",
    })

print(f"Cross-task: {len(triples) - base_count} triples\n")

# vocab filter
def vocab_ok(t):
    return all(c in vocab for c in t["prompt"] + t["chosen"] + t["rejected"])

before = len(triples)
triples = [t for t in triples if vocab_ok(t)]
dropped_vocab = before - len(triples)
print(f"Filtered by vocab: {dropped_vocab} dropped")

with open("data/dpo_dataset.jsonl", "w") as f:
    for t in triples:
        f.write(json.dumps(t) + "\n")

by_source = {}
for t in triples:
    by_source[t["source"]] = by_source.get(t["source"], 0) + 1
print(f"\nTotal: {len(triples)}  by_source: {by_source}")
```

Notar dos detalles del codigo:

- **Seed fija (`DPO_SEED = 43`)**: tanto `torch` como el `random.Random` local. Asi el sampling del base model y la eleccion de tareas cruzadas son reproducibles.
- **`max_new_tokens=20, stop_token="\n"`**: las respuestas SFT son cortas (numeros, palabras, secuencias de chars). Limitamos para que el rejected del base sea comparable en longitud al chosen.

## Output literal de la corrida

```
Cargando base model para sampling...
Generando 1500 triples base-sampled (esto toma ~1-2 min)...
  base-sampled: 200/1500 procesados, 200 aceptados
  base-sampled: 400/1500 procesados, 400 aceptados
  base-sampled: 600/1500 procesados, 600 aceptados
  base-sampled: 800/1500 procesados, 800 aceptados
  base-sampled: 1000/1500 procesados, 1000 aceptados
  base-sampled: 1200/1500 procesados, 1200 aceptados
  base-sampled: 1400/1500 procesados, 1400 aceptados

Base-sampled: 1500 triples

Generando 1500 triples cross-task...
Cross-task: 1500 triples

Filtered by vocab: 0 dropped

Total: 3000  by_source: {'base': 1500, 'cross': 1500}
```

## Ejemplos concretos — base-sampled

Asi se ve un triple base-sampled:

```jsonl
{"prompt": "INSTR: reverse 'oz'\nRESP: ", "chosen": "zo\n", "rejected": "I may, nothing? she,\n", "source": "base"}
{"prompt": "Q: what is five plus five?\nA: ", "chosen": "ten\n", "rejected": "bear the hang it: su\n", "source": "base"}
{"prompt": "INSTR: reverse 'buwnlz'\nRESP: ", "chosen": "zlnwub\n", "rejected": "you do must were sor\n", "source": "base"}
```

Analisis:

- **`reverse 'oz'`**: chosen = `zo`, rejected = `I may, nothing? she,` — el base no entendio el formato, genero Shakespeare-ish.
- **`Q: ... five plus five?`**: chosen = `ten`, rejected = `bear the hang it: su` — claramente NO una respuesta numerica, es prosa fragmentada.
- **`reverse 'buwnlz'`**: chosen = `zlnwub`, rejected = `you do must were sor` — otro caso de drift al ingles arcaico.

Estos ejemplos enseñan a la policy: **"cuando ves un prompt INSTR/RESP o Q/A, NO emitas Shakespeare. Emite el formato corto y especifico"**.

## Ejemplos concretos — cross-task

Y asi se ve un triple cross-task:

```jsonl
{"prompt": "INSTR: reverse 'mbbyxo'\nRESP: ", "chosen": "oxybbm\n", "rejected": "Shakespeare\n", "source": "cross"}
{"prompt": "Q: who wrote Crime and Punishment?\nA: ", "chosen": "Dostoyevsky\n", "rejected": "xxxx\n", "source": "cross"}
{"prompt": "INSTR: repeat 'b' two\nRESP: ", "chosen": "bb\n", "rejected": "IZ\n", "source": "cross"}
```

Analisis:

- **`reverse 'mbbyxo'`**: chosen = `oxybbm`, rejected = `Shakespeare` (que seria la respuesta a una pregunta de Q&A literario). El modelo aprende: "no respondas Shakespeare a un prompt de reverse".
- **`Q: ... Crime and Punishment?`**: chosen = `Dostoyevsky`, rejected = `xxxx` (output tipico de un repeat task). Aprende: "no des una secuencia de chars repetidos cuando te preguntan un autor".
- **`repeat 'b' two`**: chosen = `bb`, rejected = `IZ` (parece output de otro repeat o reverse). Aprende: "respeta el caracter solicitado, no improvises".

Estos ejemplos enseñan: **"cada formato de prompt corresponde a un formato de respuesta especifico. No mezcles tareas"**.

## Filtros y descartes

**Filtro 1 — `chosen == rejected`**. Descartamos triples donde el rejected coincide con el chosen. En base-sampled puede pasar si el base model adivina la respuesta correcta por casualidad (raro pero posible para prompts faciles tipo `repeat 'a' one` -> `a`). En cross-task puede pasar si dos tareas distintas tienen la misma respuesta corta (ej. ambos contestan `a`). Si `chosen == rejected` el log-ratio es exactamente 0 y el ejemplo no aporta señal de gradiente. En esta corrida: **0 dropped en ese filtro** dentro del flujo principal (los descartes implicitos durante sampling se reflejan en el conteo final que dio 1500 + 1500 = 3000 limpios).

**Filtro 2 — vocab**. Todos los chars de `prompt + chosen + rejected` deben estar en el vocab del modelo (65 chars del Shakespeare corpus). En esta corrida: **0 dropped** — esperado, porque el dataset SFT y el base model usan exactamente el mismo vocab, asi que los rejected del base solo pueden contener chars del vocab por construccion, y los rejected cross-task vienen del mismo SFT.

## Composicion final

```
Total: 3000  by_source: {'base': 1500, 'cross': 1500}
```

3000 triples balanceados 50/50 entre los dos tipos. ~333 KB en disco. Versionado al repo en `data/dpo_dataset.jsonl` para reproducibilidad — cualquiera puede correr el entrenamiento DPO sin tener que regenerar las preferencias.

## Preguntas de verificacion

1. **Por que mezclar base-sampled y cross-task en vez de usar solo uno?** Cada fuente enseña algo distinto. Base-sampled enseña "no decaigas al corpus pre-SFT" — combate el drift estilistico. Cross-task enseña "sigue LA instruccion, no otra" — combate la confusion de formato. Usar solo uno deja el otro problema sin presion explicita.

2. **Que esperarias si solo usaramos cross-task?** El modelo aprenderia a no mezclar tareas, pero podria seguir generando Shakespeare drift en prompts ambiguos o cuando la cabeza de attention se distrae, porque ese fallo nunca aparece como rejected y por tanto no recibe penalizacion.

3. **Que pasa si el base model OCASIONALMENTE adivina correcto el chosen?** Lo descartamos. Un triple donde `chosen == rejected` da log-ratio 0 (los logp del numerador y denominador se cancelan), la sigmoide queda en 0.5, y el gradiente en ese ejemplo es trivial. No aporta señal y diluye el batch — mejor saltearlo.

## Lo que viene

En el capitulo 29 entrenamos la policy DPO sobre estos 3000 triples y comparamos contra el SFT cuantitativamente: format-rate, accuracy por tarea, y win-rate side-by-side.
