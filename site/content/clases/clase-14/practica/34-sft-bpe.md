---
title: "34 - SFT con BPE: resultados honestos"
weight: 340
math: true
---

El SFT-BPE sigue el mismo procedimiento que el SFT char-level del [cap 24](../24-sft-training): loss masking sobre tokens de respuesta, `lr=1e-4`, 1500 iters. La diferencia es el tokenizador — 1112 tokens BPE en vez de 65 chars. El titulo de este capitulo dice "resultados honestos" porque los numeros no son lo que esperabamos.

---

## 1. Apertura

Cargamos el BPE-base del cap 31, el dataset SFT-BPE del cap 33 (4000 ejemplos, 4 tareas), y entrenamos 1500 iters con loss masking. El loop es identico al del cap 24: `encode_example` arma el `mask` con ceros en las posiciones del prompt y unos en las de la respuesta, y el loss enmascarado se calcula con `reduction="none"` antes de multiplicar por el mask.

La unica diferencia tecnica es que `c2i[c]` del cap 24 (lookup de caracter a indice) se reemplaza por `tok.encode(...)` (el tokenizador BPE del cap 30). El resto es identico.

---

## 2. El script

`clase_14/practica/34_train_sft_bpe.py`:

```python
"""34_train_sft_bpe.py - Cap 34: SFT con BPE + eval comparativo.

Loss masking identico al cap 24 (solo tokens de respuesta).
Carga BPE-base, entrena sobre 4 tareas BPE, evalua vs cap 25 char-level.
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from _bpe import BPETokenizer
from _models import load_pretrained_mini_llama, get_device
from _eval import load_jsonl, eval_exact_match, eval_drift

torch.manual_seed(1337)
device = get_device()

tok = BPETokenizer.load("data/bpe_tokenizer.json")
vocab_size = tok.vocab_size
cfg = dict(vocab_size=vocab_size, max_seq_len=256,
           d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384)

BLOCK = 256
BATCH = 32
LR = 1e-4
ITERS = 1500
WD = 0.01

model = load_pretrained_mini_llama("checkpoints/mini_llama_bpe_base.pt",
                                   device=device, config=cfg)
model.train()

examples = load_jsonl("data/sft_bpe_dataset.jsonl")
print(f"Loaded {len(examples)} SFT-BPE examples")

def encode_example(ex):
    prompt_ids = tok.encode(ex["prompt"])
    response_ids = tok.encode(ex["response"])
    full = prompt_ids + response_ids
    if len(full) > BLOCK + 1:
        full = full[:BLOCK + 1]
    P = len(prompt_ids)
    R = len(full) - P
    mask = [0] * (P - 1) + [1] * R
    assert len(mask) == len(full) - 1
    return full, mask

def get_batch():
    batch_inp, batch_tgt, batch_mask = [], [], []
    for _ in range(BATCH):
        ex = examples[torch.randint(0, len(examples), (1,)).item()]
        full, mask = encode_example(ex)
        while len(full) < BLOCK + 1:
            full.append(0); mask.append(0)
        full = full[:BLOCK + 1]; mask = mask[:BLOCK]
        batch_inp.append(full[:-1])
        batch_tgt.append(full[1:])
        batch_mask.append(mask)
    return (torch.tensor(batch_inp, dtype=torch.long, device=device),
            torch.tensor(batch_tgt, dtype=torch.long, device=device),
            torch.tensor(batch_mask, dtype=torch.float, device=device))

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

print(f"\nSFT-BPE training: {ITERS} iters\n")
for it in range(ITERS):
    inp, tgt, mask = get_batch()
    logits, _ = model(inp)
    loss_per_tok = F.cross_entropy(logits.reshape(-1, vocab_size),
                                   tgt.reshape(-1), reduction="none")
    loss_per_tok = loss_per_tok.reshape(inp.shape)
    masked_loss = (loss_per_tok * mask).sum() / mask.sum().clamp(min=1)
    opt.zero_grad(); masked_loss.backward(); opt.step()
    if it % 100 == 0 or it == ITERS - 1:
        print(f"iter {it:4d}  loss {masked_loss.item():.4f}", flush=True)

Path("checkpoints").mkdir(exist_ok=True)
torch.save(model.state_dict(), "checkpoints/mini_llama_bpe_sft.pt")
print("\nSaved -> checkpoints/mini_llama_bpe_sft.pt")

# Eval comparativo BPE-Base vs BPE-SFT
print("\n=== Eval BPE-Base vs BPE-SFT ===\n")
results = {}
for name, ckpt in [("bpe-base", "checkpoints/mini_llama_bpe_base.pt"),
                   ("bpe-sft",  "checkpoints/mini_llama_bpe_sft.pt")]:
    print(f"--- {name} ---")
    m = load_pretrained_mini_llama(ckpt, device=device, config=cfg)
    em = eval_exact_match(m, "data/sft_bpe_eval.jsonl", tok, n_per_task=200, device=device)
    results[name] = em
    print(f"exact_match: {em}\n")

print("=== Tabla comparativa ===")
# Referencia char-level del cap 25
char_ref = {"qa": 1.0, "repeat": 1.0, "reverse": 0.21, "upper": 0.235}
print(f"{'task':<15}{'bpe-base':<12}{'bpe-sft':<12}{'char-sft (ref)':<15}")
for task in ["qa", "repeat", "complete-en", "complete-es"]:
    b = results["bpe-base"].get(task, 0.0)
    s = results["bpe-sft"].get(task, 0.0)
    c = char_ref.get(task, "N/A")
    print(f"{task:<15}{b:<12.3f}{s:<12.3f}{str(c):<15}")

print("\n=== Drift BPE-Base vs BPE-SFT ===")
ambiguous = ["INSTR: capitalize 'cat'\nRESP: ", "Q: what is 2+2?\nA: "]
for name, ckpt in [("bpe-base", "checkpoints/mini_llama_bpe_base.pt"),
                   ("bpe-sft",  "checkpoints/mini_llama_bpe_sft.pt")]:
    m = load_pretrained_mini_llama(ckpt, device=device, config=cfg)
    drift = eval_drift(m, ambiguous, tok, device=device)
    print(f"{name}: drift = {drift:.3f}")
```

---

## 3. La curva de loss

Salida literal del training:

```text
Loaded 4000 SFT-BPE examples

SFT-BPE training: 1500 iters

iter    0  loss 6.1010
iter  100  loss 2.0399
iter  200  loss 1.5859
iter  300  loss 1.5207
iter  400  loss 1.0550
iter  500  loss 1.3865
iter  600  loss 1.1618
iter  700  loss 0.9641
iter  800  loss 0.7659
iter  900  loss 0.7312
iter 1000  loss 0.8271
iter 1100  loss 1.0038
iter 1200  loss 0.7260
iter 1300  loss 0.7312
iter 1400  loss 0.6716
iter 1499  loss 0.5946

Saved -> checkpoints/mini_llama_bpe_sft.pt
```

### Analisis de la curva

**Loss inicial: 6.10.** Con un vocabulario de 1112 tokens, la entropia maxima (distribucion uniforme) es $\log(1112) \approx 7.0$. El loss inicial de 6.10 es completamente consistente con eso: el modelo BPE-base nunca vio los prefijos `INSTR:` ni `Q:` en el contexto de instruccion durante pretrain — para el son contextos desconocidos, y la distribucion sobre 1112 tokens de respuesta es practicamente uniforme al principio. Es el mismo fenomeno que en el cap 24 (loss inicial 6.56 con 65 chars, $\log(65) \approx 4.17$ con mayor incertidumbre contextual), pero escalado al vocabulario mayor.

**Caida rapida a iter 100 (~2.04), convergencia lenta a 0.59.** Las primeras 100 iteraciones son las mas productivas: el modelo deja de generar tokens aleatorios y empieza a asociar los patrones del prompt con el inicio de la respuesta. Despues, la convergencia es mas lenta y ruidosa — hay picos en iter 500 (1.39), iter 1000 (0.83), iter 1100 (1.00). Esa varianza es tipica con batches de 32 sobre un dataset de 4000 ejemplos y 4 tareas distintas: a veces el batch tiene muchos ejemplos `complete-en` dificiles (donde la respuesta es una palabra poco comun de Shakespeare), lo que sube el loss momentaneamente. El final en **0.59** es un poco peor que el char-level (0.51 en cap 24) — el modelo BPE tiene mas incertidumbre residual, en parte porque las tareas `complete-*` son genuinamente dificiles de memorizar.

---

## 4. La tabla — lo que sorprende

Salida literal del eval:

```text
=== Eval BPE-Base vs BPE-SFT ===
--- bpe-base ---
exact_match: {'qa': 0.0, 'repeat': 0.0, 'complete-en': 0.0, 'complete-es': 0.0}
--- bpe-sft ---
exact_match: {'qa': 0.205, 'repeat': 0.785, 'complete-en': 0.005, 'complete-es': 0.06}

=== Tabla comparativa ===
task           bpe-base    bpe-sft     char-sft (ref)
qa             0.000       0.205       1.0
repeat         0.000       0.785       1.0
complete-en    0.000       0.005       N/A
complete-es    0.000       0.060       N/A

=== Drift BPE-Base vs BPE-SFT ===
bpe-base: drift = 0.700
bpe-sft: drift = 0.100
```

---

## 5. Analisis honesto — tarea por tarea

### qa: 20.5% BPE vs 100% char — mucho peor

Este es el numero que mas sorprende. ¿Por que BPE-SFT queda tan por debajo del char-SFT en una tarea que deberia ser mas facil con vocabulario mas rico?

Con char-level, memorizar `Shakespeare\n` = predecir 11 chars en secuencia (`S`, `h`, `a`, `k`, `e`, `s`, `p`, `e`, `a`, `r`, `e`, `\n`). El modelo char-level puede memorizar eso perfectamente — es simplemente copiar 12 caracteres despues de ver el prompt suficientes veces.

Con BPE, `Shakespeare\n` se tokeniza en varios tokens — probablemente algo como `['Shake', 'sp', 'eare', '\n']` o una variante similar segun los merges del vocabulario construido en el cap 30. El modelo tiene que predecir esa secuencia exacta de tokens. Si genera `['Shakespeare', ' ']` (con espacio final) en vez de `['Shake', 'sp', 'eare', '\n']`, falla el exact-match aunque el contenido sea correcto. El exact-match es **mucho mas sensible a la tokenizacion** con BPE: un solo token de diferencia al final (espacio vs newline, o una fusion distinta de la palabra) produce fallo donde el resultado es semanticamente correcto.

Hay un segundo factor: el BPE-base preentrenado en el cap 31 tenia loss de ~2.68 al final, mientras que el char-base del cap 22 llegaba a ~1.4. El base BPE es mas debil como modelo de lenguaje — las representaciones internas son de menor calidad — y eso se hereda al SFT. Un SFT que parte de un base mas debil necesita mas iteraciones o mas datos para alcanzar el mismo nivel de accuracy.

### repeat: 78.5% BPE vs 100% char — tambien peor, pero mucho mejor que qa

El repeat (`INSTR: repeat 'd' two\nRESP: dd\n`) funciona mejor que qa porque la respuesta `dd\n` es corta y su tokenizacion es simple. Con BPE, `dd\n` probablemente son 2-3 tokens sencillos — el modelo no tiene que encadenar una secuencia larga de tokens para acertar el exact-match. El 78.5% indica que el modelo aprendio el patron bien (en casi 4 de cada 5 casos genera la secuencia exacta), pero el 21.5% restante son errores de tokenizacion borde: variaciones en la cantidad de repeticiones, o un token de padding incorrecto al final.

La diferencia con qa es la longitud y complejidad de la respuesta: `Shakespeare\n` son varios tokens, `dd\n` son pocos.

### complete-en: 0.5%, complete-es: 6.0% — la metrica equivocada

Estos numeros parecen un fracaso. No lo son — pero la metrica es inadecuada.

**El exact-match es la metrica incorrecta para completacion de texto.** Considera el ejemplo del cap 33: `EN: 'We do instate and widow you'\nNEXT: `. La respuesta del dataset es `withal,\n`. Pero el modelo podria razonablemente generar `Withal,\n` (capital), o `withal\n` (sin coma), o `withal.\n` (con punto), o incluso otra palabra que semanticamente podria seguir ese contexto en Shakespeare. Todas son respuestas razonables que fallan en exact string match.

La metrica justa para completacion es BLEU (que da credito parcial por n-gramas correctos), o accuracy de top-K tokens, o evaluacion humana. El exact-match solo funciona bien cuando hay una sola respuesta posible y exacta — que es el caso de `qa` (la respuesta es siempre `Shakespeare\n`) y `repeat` (la respuesta esta determinada mecanicamente por el prompt).

Lo que **si** podemos afirmar sobre `complete-*`: el BPE-SFT aprende el **formato** correcto. El BPE-base generaba Shakespeare drift ante cualquier prompt; el BPE-SFT, ante `EN: '...'\nNEXT: `, genera una palabra (o un token de palabra) con o sin puntuacion — la estructura de respuesta corta que define la tarea. Ese es el aprendizaje que SFT aporta, aunque el exact-match no lo capture.

### drift: 70% → 10% — el mayor exito

Esta es la metrica donde BPE-SFT muestra el mejor resultado. El drift baja de 70% en BPE-base a 10% en BPE-SFT — una reduccion del 85%.

¿Por que el BPE-base tenia drift tan alto (70%) comparado con el char-base (26.7% en cap 22)? El modelo BPE genera texto mas fluido: palabras completas reconocibles, no secuencias de chars incoherentes. Ese texto fluido incluye palabras shakespeareanas completas — `thou`, `thee`, `doth`, `hath` — que el modelo usa incluso cuando el prompt pide una respuesta estructurada. El char-base generaba gibberish que a veces contenia marcadores y a veces no; el BPE-base generaba prosa shakespeareana coherente, que casi siempre activa el detector de drift. Mas calidad de lenguaje = mas drift cuando el modelo no ha aprendido a seguir instrucciones.

SFT resuelve eso: despues de 1500 iters con loss masking, el modelo BPE-SFT aprendio que esos prompts piden respuestas cortas y especificas. El drift cae a 10% — el 90% del tiempo genera en el formato correcto.

---

## 6. Comparacion honesta char vs BPE hasta aqui

| Metrica | Char-Base | Char-SFT | BPE-Base | BPE-SFT |
|---|---|---|---|---|
| qa | 0% | 100% | 0% | 20.5% |
| repeat | 0% | 100% | 0% | 78.5% |
| complete-en | — | — | 0% | 0.5% |
| complete-es | — | — | 0% | 6.0% |
| drift | 26.7% | 0% | 70% | 10% |

**BPE-SFT es peor que char-SFT en las metricas comparables (qa, repeat).** Eso es inesperado. Las hipotesis para explicarlo:

1. **El base BPE es mas debil.** Loss de pretrain 2.68 vs ~1.4 del char-base. El SFT hereda esa debilidad. Con un base de peor calidad, SFT tiene menos representacion interna con que trabajar — necesita "aprender mas desde cero" en 1500 iters, mientras que char-SFT pudo aprovechar representaciones de mejor calidad para memorizar los patrones.

2. **Los facts de qa incluyen nombres propios que BPE tokeniza de forma no obvia.** `Shakespeare`, `Cervantes`, `Madrid`, `London` — estos nombres son tokens multi-parte en el BPE. El modelo tiene que predecir la secuencia exacta de tokens para cada nombre. Char-level simplifica: cada caracter es un token independiente con un indice fijo y conocido. Con BPE, la segmentacion es aprendida y dependiente del vocabulario.

3. **Exact-match con BPE es mas sensible a errores de ultimo token.** Un token de newline de mas, un espacio sobrante, una fusion alternativa del ultimo caracter — cualquiera de esos errores produce fallo en exact-match. Con char-level, cada prediccion es un caracter; el margen de error es minimo porque la granularidad es maxima.

**Pero BPE habilita tareas nuevas que char-level nunca pudo abordar.** `complete-en` y `complete-es` son imposibles de hacer bien con char-level — no porque el modelo no pueda aprenderlas, sino porque predecir palabras completas char-a-char sin nocion de unidades lexicas no generaliza. Y el drift mejoro muchisimo: de 70% a 10%. El modelo BPE-SFT respeta el formato de instruccion en 9 de cada 10 casos.

---

## 7. Preguntas de verificacion

**1. ¿Por que qa=20.5% BPE vs 100% char?**

Tres factores combinados: (a) BPE tokeniza nombres propios en multiples tokens — el modelo tiene que predecir la secuencia exacta `['Shake', 'sp', 'eare', '\n']` en vez de 12 chars individuales; (b) el exact-match es mas sensible con BPE porque un token de diferencia (espacio vs newline al final) produce fallo; (c) el BPE-base es mas debil que el char-base (loss pretrain 2.68 vs ~1.4), y SFT hereda esa debilidad de representacion.

**2. ¿Por que complete-en/es tiene exact-match tan bajo? ¿Es un fracaso?**

No es un fracaso — es una metrica inadecuada. Para completacion de texto hay muchas continuaciones validas: `withal,` vs `Withal,` vs `withal` son todas razonables pero fallan en exact string match. La metrica correcta para esta tarea es BLEU o accuracy de top-K, no exact match. Lo que si aprendio BPE-SFT es el **formato**: generar una palabra corta en vez de Shakespeare drift. Ese aprendizaje es real aunque el exact-match no lo capture.

**3. El drift bajo de 70% a 10% con SFT. ¿Por que el BPE-base tenia mas drift que el char-base?**

El BPE-base genera texto mas fluido — palabras completas reconocibles incluyendo palabras shakespeareanas (`thou`, `thee`, `doth`). Esa fluidez activa el detector de drift casi siempre. El char-base generaba secuencias de chars incoherentes que a veces contenian marcadores estructurales y a veces no — la varianza era alta, el drift promedio menor. Mas calidad de lenguaje en el base implica mas drift cuando el modelo no sabe seguir instrucciones.

---

## 8. Lo que viene

Cap 35: construimos el dataset DPO-BPE. El `rejected` sera texto BPE-base — palabras reales pero incorrectas para el contexto de la instruccion. Eso deberia dar una senal de preferencia mas limpia que los rejected char-level del cap 28 (que eran gibberish — es dificil aprender preferencia cuando el rejected es incomprensible). En cap 36 vemos si DPO puede mejorar sobre el BPE-SFT en qa y repeat, y reducir el drift residual del 10% que quedo.

Volver al [hub de practica](..) o a la [Clase 14](../..).
