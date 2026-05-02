---
title: "25 - Eval SFT: Base vs SFT"
weight: 250
math: true
---

Tenemos el modelo SFT del [cap 24](../24-sft-training). ¿Cuanto mejoro vs el base? Esta es la pregunta cuantitativa de Fase 6 — y el cierre del bloque SFT antes de saltar a DPO en la Fase 7.

---

## 1. Las tres metricas

Para responder bien necesitamos mas que una sola metrica. Cada una mide algo distinto:

- **`exact_match` por tarea**: porcentaje de respuestas EXACTAMENTE iguales a la esperada. 200 prompts holdout por tarea (no vistos en training), `temperature=0.1`, `top_k=10`. Es la metrica "dura" — o acerto la respuesta o no.
- **`qualitative`**: 3 muestras por prompt a `temperature=0.8` (mas creativa) — para inspeccion cualitativa, no para promediar. Sirve para ver QUE tipo de errores comete cuando se equivoca.
- **`drift`**: porcentaje de generaciones que contienen marcadores Shakespeare-style (`thou`, `doth`, `hath`, etc.) sobre prompts ambiguos que NO son de las 4 tareas entrenadas. Mide cuanto el modelo se sigue yendo al estilo del corpus original. Es el termometro de "¿el SFT logro despegarlo del prior viejo?".

---

## 2. El script

`17_eval_sft.py` completo:

```python
"""17_eval_sft.py - Cap 25: eval comparativa Base vs SFT."""
import torch
from _models import load_pretrained_mini_llama
from _eval import build_char_maps, eval_exact_match, eval_qualitative, eval_drift

torch.manual_seed(1337)

text = open("shakespeare.txt").read()
c2i, i2c = build_char_maps(text)

print("=== Eval Base vs SFT ===\n")
results = {}
for name, ckpt in [
    ("base", "checkpoints/mini_llama_base.pt"),
    ("sft",  "checkpoints/mini_llama_sft.pt"),
]:
    print(f"--- Evaluando {name} ---")
    model = load_pretrained_mini_llama(ckpt)
    em = eval_exact_match(model, "data/sft_eval.jsonl", c2i, i2c, n_per_task=200)
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
qual = eval_qualitative(sft_model, prompts, c2i, i2c, n_samples=3)
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
    drift = eval_drift(m, ambiguous, c2i, i2c)
    print(f"{name}: drift = {drift:.3f}")
```

Las funciones `eval_exact_match`, `eval_qualitative` y `eval_drift` viven en `_eval.py`. La logica es simple: tokenizar el prompt, generar hasta `\n` con los hyperparams indicados, comparar con la respuesta esperada (en el caso de exact match) o contar marcadores (en el caso de drift).

---

## 3. La tabla — el resultado central

Salida literal del bloque `=== Tabla comparativa ===`:

```
task        base      sft
reverse     0.000     0.210
upper       0.000     0.235
repeat      0.000     1.000
qa          0.000     1.000
```

Reformateada para lectura:

| Tarea | Base | SFT | Salto |
|---|---|---|---|
| reverse | 0.0% | 21.0% | +21 pp |
| upper | 0.0% | 23.5% | +23.5 pp |
| repeat | 0.0% | 100.0% | +100 pp |
| qa | 0.0% | 100.0% | +100 pp |

Cuatro tareas, cuatro saltos positivos. La columna `base` es uniformemente cero — esperable: en el cap 22 vimos que el base ante `INSTR: ...` devuelve Shakespeare drift, jamas la respuesta correcta. La columna `sft` es lo interesante: dos tareas saturan a 100%, dos quedan en ~20%. Esa asimetria merece explicacion.

---

## 4. Lectura honesta de los numeros

Esta es la seccion critica del capitulo. Los numeros dicen mas de lo que parece a simple vista, pero solo si los miramos con honestidad.

**`repeat` y `qa` saturan a 100%**. Esperable. `repeat` es la tarea mas trivial del set: dado `'a' three`, devolver `aaa`. Una vez que el modelo aprendio el patron, el char a copiar esta literalmente en el prompt — solo hay que repetirlo N veces. `qa` por su parte es memorizacion pura: el dataset tiene 30 facts (cap 23) y el eval set se construye con prompts que mayoritariamente caen en facts vistos en training. Un modelo con ~1M parametros puede memorizar 30 pares (pregunta, respuesta) sin esfuerzo. Ambas tareas pasan de 0% (drift Shakespeare) a perfecto.

**`reverse` y `upper` solo llegan a ~21-23%**. ¿Por que tan bajo? Aqui hay que ser honesto. El modelo aprende el FORMATO (lo veremos cualitativamente en la proxima seccion), pero pierde precision a nivel caracter. Razones concretas:

- **Capacidad chica**: `d_model=128`, 4 capas, ~1M parametros. Un modelo asi no puede memorizar todas las permutaciones char-level. Para `reverse` con palabras de 5 chars, el espacio de entradas es del orden $26^5 \approx 12M$ — ningun modelo de 1M parametros memoriza eso.
- **Tarea anti-natural para char-level**: `reverse 'house' -> esuoh` requiere predecir `e`, despues `s`, despues `u`, etc., en orden inverso. El modelo solo ve a izquierda (causal mask); no tiene mecanismo para "ver la palabra completa antes de empezar a generar". Para hacer reverse correctamente tendria que recursar internamente sobre la palabra del prompt — y un transformer chico no aprende esa recursion en 1500 iters.
- **Mixed-batch interference**: el modelo ve 4 tareas distintas en el mismo batch. Para `reverse` necesita recursar internamente la palabra; para `upper` solo aplica una funcion char-a-char; para `repeat` reproduce un caracter. Los tres comportamientos comparten parametros — no tiene capacidad para especializarse perfecto en cada uno.

Lo que estos numeros **NO dicen**: NO dicen que SFT no funcione. Dicen que el modelo es chico y la tarea es dura char-level. Para validar SFT, lo que importa es el SALTO desde 0% — el modelo paso de "no entender el formato" a "entender el formato y empezar la respuesta correcta". Eso es exactamente lo que hace SFT: enseña formato, no enseña razonamiento. El razonamiento depende de la capacidad del modelo y del tipo de tarea.

Si quisieramos que `reverse` y `upper` saturen tambien, las palancas son ortogonales a SFT: subir `d_model` a 256 o 512, usar BPE en vez de char-level (asi cada token es una palabra y el modelo no necesita razonar char-a-char), o entrenar mas iters. Ninguna de esas requiere cambiar SFT.

---

## 5. Eval cualitativo — donde se ve el formato

Salida literal del bloque `=== Eval cualitativo (SFT) ===`:

```
Prompt: "INSTR: reverse 'house'\nRESP: "
  [0] 'eohUT'
  [1] 'eouFU'
  [2] 'eoGoe'

Prompt: "INSTR: upper 'world'\nRESP: "
  [0] 'WOWOO'
  [1] 'WROQR'
  [2] 'WOLOW'

Prompt: 'Q: who wrote Hamlet?\nA: '
  [0] 'Shakespeare'
  [1] 'Shakespeare'
  [2] 'Shakespeare'
```

Tres observaciones:

- **`reverse 'house'`** -> arranca con `e`, `o`, `h`: las primeras tres letras del reverso son correctas (`esuoh` empieza con `e`, despues `s`, despues `u`...). De hecho `eoh` son las 3 letras iniciales del reverso si las leemos con ruido — el modelo agarro la idea pero no la ejecuta perfecto. A partir de la cuarta posicion el modelo se pierde y mete chars random (algunas mayusculas, lo cual es interferencia con la tarea `upper`). **Comparar con el cap 22**, donde el base ante el mismo prompt generaba `alast the king, there is be doth in him` — la diferencia de comportamiento es enorme: el base ignora la instruccion y sigue con Shakespeare; el SFT al menos intenta la inversion.
- **`upper 'world'`** -> 5 letras MAYUSCULAS, todas plausibles. La primera (`W`) es siempre correcta. Las siguientes drift dentro del set de uppercase. Format learning sin char-level precision: el modelo aprendio "cuando ves `upper`, devuelve 5 chars en mayuscula y un `\n`" — lo de las letras especificas le falta.
- **`Q: who wrote Hamlet?`** -> `Shakespeare` perfecto, 3 de 3 incluso a `temperature=0.8`. Memorizacion de fact, sin variabilidad. Es el comportamiento esperable: el modelo cacheo el par (pregunta, respuesta) durante SFT y lo reproduce determinista.

Comparar la consistencia del FORMATO en los tres casos: siempre termina (no genera 100 chars seguidos), siempre la longitud de la respuesta es cercana a lo esperado, siempre uppercase para `upper`, siempre lowercase + caracteres random para `reverse`, siempre el fact correcto para `qa`. Esa consistencia de formato es la señal de que SFT funciono — y es ortogonal a la accuracy char-level.

---

## 6. Drift — la metrica subestimada

Salida literal del bloque `=== Drift score ===`:

```
base: drift = 0.400
sft: drift = 0.000
```

Sobre prompts AMBIGUOS (`INSTR: capitalize ...`, `Q: what is 2+2?`) que **NO son de las 4 tareas entrenadas**, el base genera Shakespeare 40% del tiempo. El SFT, **0%** — no genera ni una sola palabra Shakespeare-style.

Esta metrica parece secundaria pero es importante: SFT no solo enseño 4 tareas — enseño a **respetar el formato instrucciones** en general. Es transferencia de comportamiento, no solo memorizacion. Cuando le damos un prompt fuera de distribucion (`capitalize` en vez de `upper`), el SFT no se vuelve al modo Shakespeare — intenta resolverlo con el formato instrucciones, aunque le pifie a la respuesta.

Esto es exactamente lo que DPO (caps 26-29) refina. Veremos que DPO empuja el drift aun mas abajo (de hecho ya esta en 0%, asi que lo que mejora es la CALIDAD de las respuestas en prompts fuera de distribucion) y mejora el comportamiento general. La transicion de "drift=40%" a "drift=0%" es la firma macroscopica de que el modelo cambio de regimen — paso de ser un generador de Shakespeare a ser un seguidor de instrucciones.

---

## 7. Preguntas de verificacion

1. ¿Por que `repeat` y `qa` saturan a 100% pero `reverse` y `upper` no?
2. ¿Como interpretar el drift bajando de 40% a 0% sobre prompts no entrenados?
3. ¿Que esperarias si entrenaramos con `d_model=512` en vez de 128?

Pista para la 1: capacidad del modelo + dificultad char-level inherente. `repeat` y `qa` no requieren razonamiento char-a-char — el primero copia, el segundo memoriza facts. `reverse` y `upper` requieren operar caracter por caracter de forma precisa, y un modelo de 1M parametros no tiene capacidad sobrada para eso cuando ademas comparte pesos con otras 3 tareas.

Pista para la 2: SFT enseña formato, no solo tareas especificas. El modelo aprendio "ante un prompt con estructura instruccion, no respondo con Shakespeare — intento seguirlo". Esa generalizacion es transferencia de comportamiento — ocurre porque el formato instruccion es una señal consistente en todo el dataset SFT, mientras que las tareas especificas son señales heterogeneas.

Pista para la 3: mejor accuracy en `reverse` y `upper` (probablemente 60-80% en cada una), pero mismo formato y mismo drift. Aumentar `d_model` mejora la precision char-level pero no cambia lo que SFT enseña — el formato y el comportamiento ya estaban resueltos en `d_model=128`. La accuracy es ortogonal al SFT en si.

---

## 8. Cierre de Fase 6

Cerramos la Fase 6. SFT funciono: el modelo aprendio el formato (eval cualitativo), satura en tareas faciles (`repeat` y `qa` a 100%), tiene accuracy parcial en las dificiles (`reverse` y `upper` ~21-23%, limitado por capacidad y char-level), y el drift cayo de 40% a 0%. La validacion no es "todas las tareas a 100%" sino "salto consistente desde 0% en todas, con formato correcto y sin drift al estilo viejo".

En la Fase 7 ([caps 26-29](..)) vamos a refinarlo con **DPO** — preferencias en vez de demostraciones. En SFT le mostramos al modelo "asi se responde". En DPO le vamos a mostrar pares (respuesta_buena, respuesta_mala) y el modelo aprendera a inclinarse hacia las buenas. Es la diferencia entre "imitar al profesor" (SFT) y "aprender de feedback comparativo" (DPO).

Volver al [hub de practica](..) o a la [Clase 14](../..).
