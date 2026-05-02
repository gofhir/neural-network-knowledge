---
title: "29 - DPO training + eval: cierre Camino 2"
weight: 290
math: true
---

## Apertura

Cerramos Camino 2. Tenemos las tres piezas listas:

- El dataset DPO con 3000 triples `(prompt, chosen, rejected)` del [capitulo 28]({{< relref "clases/clase-14/practica/28-dataset-dpo" >}}).
- La loss DPO derivada y testeada en el [capitulo 27]({{< relref "clases/clase-14/practica/27-dpo-loss" >}}).
- Los helpers `dpo_loss`, `load_pretrained_mini_llama`, `eval_exact_match` y `eval_drift` viviendo en `clase_14/practica/_models.py` y `_eval.py`.

Ahora corremos el training y evaluamos comparativamente Base vs SFT vs DPO.

El resultado es **menos limpio de lo que esperabamos** — y esa es la leccion del capitulo. DPO en este setting concreto **bajo la accuracy en todas las tareas** vs el SFT, aunque mantuvo el formato (drift = 0). Vamos a verlo paso a paso, sin maquillaje, y discutir por que pasa esto y que se puede hacer al respecto.

Si quieres volver al hub de Camino 2 para ver la secuencia completa, ahi esta el [indice de practica]({{< relref "clases/clase-14/practica" >}}).

## Setup del training

El esquema de DPO requiere dos copias del modelo:

- `policy` = copia del checkpoint SFT, **trainable**. Es la que vamos a actualizar.
- `ref` = copia del checkpoint SFT, **frozen** (`requires_grad_(False)`, `eval()`). Es la que define el "anchor" — la policy original sobre la que medimos el cambio.

La loss DPO es:

$$
\mathcal{L}\_{\text{DPO}} = - \log \sigma\!\left( \beta \cdot \big[ \log \pi\_\theta(y\_w \mid x) - \log \pi\_\theta(y\_l \mid x) - \log \pi\_{\text{ref}}(y\_w \mid x) + \log \pi\_{\text{ref}}(y\_l \mid x) \big] \right)
$$

donde $y_w$ es la respuesta `chosen` y $y_l$ la `rejected`. La loss empuja a la policy a asignar mas probabilidad relativa a `chosen` sobre `rejected`, **medido como cambio respecto al ref**.

Hyperparams elegidos:

| Hyperparam | Valor | Por que |
|---|---|---|
| `LR` | `5e-5` | Mas chico que SFT (`3e-4`). DPO es mas sensible — la policy ya esta cerca de un buen punto. |
| `BETA` | `0.1` | Estandar de la literatura (DPO paper, Zephyr, Llama-3). KL implicito permisivo. |
| `ITERS` | `1000` | Suficiente para ver convergencia. |
| `BATCH` | `16` | **Mas chico que SFT (32)** — cada step DPO hace 4 forward passes (policy chosen, policy rejected, ref chosen, ref rejected) en vez de 1. |
| `WD` | `0.01` | AdamW estandar. |

Ese `BATCH=16` es el detalle de implementacion que mas duele en memoria. Cada iteracion arma un mini-lote, y por cada triple del lote llama a `dpo_loss(policy, ref, prompt, chosen, rejected)`, que internamente hace dos forward por modelo. Con `BATCH=16` ya estamos haciendo 64 forwards por iter — mas que eso saturaba la maquina.

## El script

`clase_14/practica/21_train_dpo.py`:

```python
"""21_train_dpo.py - Cap 29: DPO training + eval comparativa final.

Carga policy y ref desde SFT checkpoint, entrena DPO con 3000 triples,
evalua Base vs SFT vs DPO, y mide drift sobre prompts ambiguos.
"""
import torch
from pathlib import Path
from _models import load_pretrained_mini_llama, dpo_loss
from _eval import build_char_maps, eval_exact_match, eval_drift, load_jsonl

torch.manual_seed(1337)

# Hyperparams (ver tabla design doc)
LR = 5e-5
BETA = 0.1
ITERS = 1000
BATCH = 16
WD = 0.01

text = Path("shakespeare.txt").read_text()
c2i, i2c = build_char_maps(text)

print("Cargando policy y ref desde SFT checkpoint...")
policy = load_pretrained_mini_llama("checkpoints/mini_llama_sft.pt")
ref    = load_pretrained_mini_llama("checkpoints/mini_llama_sft.pt")
for p in ref.parameters():
    p.requires_grad_(False)
ref.eval()
policy.train()

triples = load_jsonl("data/dpo_dataset.jsonl")
print(f"Loaded {len(triples)} DPO triples\n")

def encode(s):
    return torch.tensor([c2i[c] for c in s], dtype=torch.long)

def get_batch_loss():
    losses = []
    for _ in range(BATCH):
        idx = torch.randint(0, len(triples), (1,)).item()
        t = triples[idx]
        l = dpo_loss(policy, ref, encode(t["prompt"]), encode(t["chosen"]),
                     encode(t["rejected"]), beta=BETA)
        losses.append(l)
    return torch.stack(losses).mean()

opt = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=WD)

print(f"DPO training: ITERS={ITERS} BATCH={BATCH} LR={LR} BETA={BETA}\n")
for it in range(ITERS):
    loss = get_batch_loss()
    opt.zero_grad()
    loss.backward()
    opt.step()
    if it % 50 == 0 or it == ITERS - 1:
        print(f"iter {it:4d}  loss {loss.item():.4f}", flush=True)

torch.save(policy.state_dict(), "checkpoints/mini_llama_dpo.pt")
print("\nSaved -> checkpoints/mini_llama_dpo.pt\n")

# === Eval comparativa Base vs SFT vs DPO ===
print("=== Eval comparativa Base vs SFT vs DPO ===\n")
results = {}
for name, ckpt in [
    ("base", "checkpoints/mini_llama_base.pt"),
    ("sft",  "checkpoints/mini_llama_sft.pt"),
    ("dpo",  "checkpoints/mini_llama_dpo.pt"),
]:
    print(f"--- Evaluando {name} ---")
    m = load_pretrained_mini_llama(ckpt)
    em = eval_exact_match(m, "data/sft_eval.jsonl", c2i, i2c, n_per_task=200)
    results[name] = em
    print(f"exact_match: {em}\n")

print("=== Tabla comparativa final ===")
print(f"{'task':<12}{'base':<10}{'sft':<10}{'dpo':<10}")
for task in ["reverse", "upper", "repeat", "qa"]:
    b = results["base"].get(task, 0.0)
    s = results["sft"].get(task, 0.0)
    d = results["dpo"].get(task, 0.0)
    print(f"{task:<12}{b:<10.3f}{s:<10.3f}{d:<10.3f}")

# === Drift en prompts ambiguos (OOD) ===
print("\n=== Drift en prompts ambiguos (OOD) ===")
ambiguous = [
    "INSTR: capitalize 'cat'\nRESP: ",
    "INSTR: revrse 'dog'\nRESP: ",
    "Q: what is the meaning of life?\nA: ",
]
for name, ckpt in [
    ("base", "checkpoints/mini_llama_base.pt"),
    ("sft",  "checkpoints/mini_llama_sft.pt"),
    ("dpo",  "checkpoints/mini_llama_dpo.pt"),
]:
    m = load_pretrained_mini_llama(ckpt)
    drift = eval_drift(m, ambiguous, c2i, i2c)
    print(f"{name}: drift = {drift:.3f}")
```

## La curva de loss

Salida literal del training:

```
DPO training: ITERS=1000 BATCH=16 LR=5e-05 BETA=0.1

iter    0  loss 0.6931
iter   50  loss 0.2184
iter  100  loss 0.1207
iter  150  loss 0.1881
iter  200  loss 0.1369
iter  250  loss 0.0829
iter  300  loss 0.0750
iter  350  loss 0.0685
iter  400  loss 0.0702
iter  450  loss 0.0721
iter  500  loss 0.0444
iter  550  loss 0.0357
iter  600  loss 0.0483
iter  650  loss 0.0457
iter  700  loss 0.0192
iter  750  loss 0.0254
iter  800  loss 0.0380
iter  850  loss 0.0357
iter  900  loss 0.0298
iter  950  loss 0.0367
iter  999  loss 0.0070

Saved -> checkpoints/mini_llama_dpo.pt
```

Tres observaciones:

**Loss inicial = 0.6931.** Exactamente $-\log(0.5) \approx 0.6931$, como predijimos en el capitulo 27. En el iter 0, `policy = ref` (mismos pesos), entonces los logratios son cero, el sigmoide da 0.5, y la loss es $-\log 0.5$. Esto confirma que la implementacion arranca exactamente donde la teoria dice.

**Bajada agresiva en los primeros 50 iters.** De 0.69 a 0.22 — la policy aprende muy rapido a separar `chosen` de `rejected`. Despues sigue bajando, mas lento, hasta 0.007 al final. El logaritmo del logratio crece monotonamente y la loss converge cerca de cero.

**Convergencia muy fuerte.** Una loss de 0.007 significa que $\sigma(\beta \cdot \text{logratios}) \approx 0.993$ — la policy esta **casi 100% confiada** en preferir chosen sobre rejected en cada triple del lote. Demasiado fuerte? Lo veremos al evaluar.

## Eval comparativa — el resultado honesto

Resultado literal de la corrida final:

```
task        base      sft       dpo
reverse     0.000     0.210     0.155
upper       0.000     0.235     0.125
repeat      0.000     1.000     0.760
qa          0.000     1.000     0.840

base: drift = 0.267
sft: drift = 0.000
dpo: drift = 0.000
```

Reformateado como tabla con el delta DPO vs SFT:

| Tarea | Base | SFT | DPO | Cambio DPO vs SFT |
|---|---|---|---|---|
| reverse | 0.0% | 21.0% | 15.5% | **-5.5 pp** |
| upper | 0.0% | 23.5% | 12.5% | **-11.0 pp** |
| repeat | 0.0% | 100.0% | 76.0% | **-24.0 pp** |
| qa | 0.0% | 100.0% | 84.0% | **-16.0 pp** |

Drift (sobre prompts ambiguos OOD):

| Modelo | Drift |
|---|---|
| Base | 0.267 |
| SFT | 0.000 |
| DPO | 0.000 |

DPO bajo la accuracy en **las cuatro tareas**, entre 5.5 y 24 puntos porcentuales. El drift se mantuvo en cero. Esto es el resultado real, sin filtros.

## ¿Que paso? — analisis honesto

Esta es la seccion mas importante del capitulo. No vamos a inventar excusas. Vamos a discutir cuatro hipotesis con humildad cientifica.

**El hecho.** DPO bajo la accuracy en TODAS las tareas. La loss converge a 0.007 (casi cero) pero el exact-match empeoro entre 5.5 y 24 puntos porcentuales. El drift se mantuvo en 0% — DPO **no destruyo lo que SFT aprendio sobre formato**, pero deteriorito la precision a nivel de caracter. La policy sigue diciendo "voy a responder en formato `RESP:` o `A:`", pero los chars que produce son menos exactos.

### Hipotesis 1: over-optimization sobre las preferencias

Loss = 0.007 indica que la policy esta **muy confiada** de preferir chosen sobre rejected en casi todos los triples. Para lograr esto, necesita hacer $\log \pi_\theta(\text{chosen})$ alto y $\log \pi_\theta(\text{rejected})$ bajo. Pero la policy puede lograr esto **deformando otras zonas de su distribucion** que no son ni chosen ni rejected.

Cuando despues le pedimos generar `tac` token a token, las distribuciones de cada char fueron afectadas en formas inesperadas — incluso si globalmente la policy prefiere `tac` sobre `CAT`, el camino char-por-char hasta llegar a `tac` puede haberse vuelto menos probable que antes. La preferencia se mide a nivel de **secuencia completa** (logprob sumado), pero la generacion greedy/argmax decide **char por char**.

Es el mismo fenomeno que el reward hacking de RLHF en miniatura: optimizar agresivamente una proxy puede degradar la metrica que en realidad importa.

### Hipotesis 2: beta demasiado bajo

`beta=0.1` es valor estandar de la literatura, pero en este setup concreto puede estar aplicando un KL implicito **demasiado permisivo**. La policy se aleja agresivamente del ref. Con `beta=0.5` o `1.0`, la policy estaria mas pegada al SFT — preservaria mejor la accuracy original a cambio de menor diferenciacion preference.

Esto es el **tradeoff clasico de DPO**: beta alto = anclado al SFT, conservador, poca señal preference. Beta bajo = libre para alejarse, fuerte señal preference, pero riesgo de degradar lo que el SFT logro. En entornos industriales (Llama-3-Instruct, Mistral-7B-Instruct) los equipos hacen sweeps sobre beta y eligen segun una eval suite. Aqui usamos el valor por defecto y pagamos el precio.

### Hipotesis 3: cross-task rejected es ruidoso

El 50% de nuestros 3000 triples tiene `rejected` = respuesta de **otra tarea**. Por ejemplo:

- Triple A: `INSTR: reverse 'cat'` → chosen=`tac`, rejected=`Shakespeare wrote Hamlet` (de `qa`).
- Triple B: `Q: who wrote Hamlet?` → chosen=`Shakespeare wrote Hamlet`, rejected=`xxxx` (de `repeat`).

La policy aprende: "no generes Shakespeare cuando el prompt dice reverse". Bien. Pero al mismo tiempo, en el otro triple, aprende: "preferir Shakespeare sobre xxxx en qa". Cada char individual de `Shakespeare` (la `S` mayuscula, las `e/a/k` del medio) recibe **señales conflictivas** segun el contexto. Es posible que estas señales contradictorias, agregadas sobre 1000 iters, perturben las distribuciones char-level de formas que degradan la generacion.

La fix natural: usar **solo base-sampled** rejected (la primera mitad del dataset, donde `rejected` viene del modelo base sobre el mismo prompt). Eso elimina el cross-task noise.

### Hipotesis 4: iters demasiado

Loss bajo de 0.69 a 0.22 en los primeros **50 iters**. Despues siguio bajando 0.22 a 0.007 en los 950 iters restantes. Es decir, **el 70% de la señal se aprendio en el 5% del training**. El resto fue refinamiento.

Tal vez **early stopping en iter 100 o 200** (loss ~0.12) hubiera dado mejores resultados — la policy hubiera capturado la señal de preferencia sin sobreajustar las distribuciones token-level. Esta es la regularizacion mas barata que existe: parar antes.

### Lo que esto SI dice

DPO no es magia. Es una tecnica con tradeoffs reales. En literatura industrial (Llama-3-Instruct, Mistral, Zephyr, Tulu) DPO se ajusta finamente — beta, iters, dataset quality, todos importan. Mostrar esto pedagogicamente, viendo el fallo, es mas honesto que pretender que DPO siempre mejora todo.

### Lo que esto NO dice

NO dice que DPO no funcione. NO dice que la implementacion este rota — el helper `dpo_loss` paso el test del capitulo 27 y la loss converge al optimo matematico ($\to 0$). Dice que **DPO necesita ajuste empirico** y que en un setting controlado y pequeño como el nuestro, los tradeoffs son mas visibles que en un modelo grande con datasets curados.

## El drift sobrevive

Punto positivo, importante:

- Drift permanecio en **0.000** vs SFT (tambien 0.000) y mucho mejor que el base (0.267).
- A pesar de la regression en accuracy, DPO **no destruyo el formato aprendido por SFT**. Sigue respetando la convencion `INSTR:/RESP:` y `Q:/A:`.
- La policy se quedo en el espacio "asistente que sigue formato" — solo se desplazo dentro de ese espacio, no salio.

Esto importa porque significa que el `ref` model **si esta cumpliendo su rol regularizador** a nivel macro. La policy no colapso a Shakespeare-drift, no empezo a generar texto incoherente, no rompio la estructura. La degradacion fue dentro del manifold "asistente formateado", lo cual es exactamente el tipo de degradacion que es recuperable con un ajuste de hyperparams.

## Que harias diferente — proximos pasos honestos

Tres cosas concretas que mejorarian este experimento:

1. **Tunear beta.** Probar `beta=0.3, 0.5, 1.0`. La literatura sugiere que beta mas alto preserva mejor el SFT a cambio de menor differentiation. Un sweep simple sobre 3-4 valores te da una curva de tradeoff y un punto optimo empirico.
2. **Early stopping con eval cada N iters.** Monitorear `eval_exact_match` cada 100 iters, y parar cuando deje de mejorar (o empiece a degradar). En este corrida, casi seguro hubieramos parado en iter 100-200.
3. **Limpiar el dataset.** Quitar los 1500 triples cross-task y usar solo los 1500 base-sampled. Eliminamos las señales conflictivas char-level y vemos si la regression desaparece.

Estos experimentos los dejamos como ejercicio o como bonus de Camino 2. Lo importante para cerrar este Camino es que **viste como DPO funciona end-to-end** — derivacion, dataset, training, eval — y los tradeoffs reales que aparecen al implementarlo.

## Cierre Camino 2

Camino 2 termina aqui. Resumen de lo que viste:

**Fase 6 — SFT desde cero (capitulos 22-25).** Mini-LLaMA paso de Shakespeare-drift al formato instruccion. Drift cayo de 40% a 0%. `repeat` y `qa` subieron de 0% a 100% exact-match. `reverse` y `upper`, las dos tareas que requieren manipulacion char-level real, llegaron a 21-23%. Aprendimos que SFT es un cambio de **distribucion** sobre tokens, no un truco — el modelo aprendio a "ser asistente que sigue formato" reentrenando con triples `(INSTR, RESP)` y `(Q, A)`.

**Fase 7 — DPO desde cero (capitulos 26-29).** Bradley-Terry para preferencias, derivacion de la loss DPO desde RLHF (capitulo 27), construccion del dataset mix base-sampled + cross-task (capitulo 28), training y eval comparativa (este capitulo). Resultado honesto: DPO mantuvo el formato (drift = 0) pero **degrado la accuracy** en este setting concreto — leccion sobre tradeoffs reales y la importancia de tunear hyperparams empiricamente.

El stack completo: **pretrain → SFT → DPO**. Es el pipeline de Llama-3-Instruct, Mistral-Instruct, Zephyr y Tulu — y ahora lo construiste pieza por pieza, sin ocultar las partes feas.

Caminos siguientes (referencia en la memoria del proyecto):

- **Camino 3**: interpretabilidad mecanicista (Anthropic Transformer Circuits, induction heads, sparse autoencoders).
- **Camino 4**: BERT (encoder-only + masked language modeling).
- **Camino 5**: ViT (Vision Transformer).

Y como bonus inmediato, retomable cuando quieras: **re-entrenar Camino 2 con `beta=0.5` y comparar**. Seria tu primer experimento real de hyperparam search sobre tu propio modelo. Si la accuracy DPO se acerca o supera al SFT, validas la hipotesis 2.

## Preguntas finales

1. **¿Por que la loss bajo a 0.007 pero la accuracy se DEGRADO?**
   La loss DPO mide preferencia a nivel de **secuencia completa** (logprob sumado). La accuracy exact-match mide generacion **char por char**. Optimizar agresivamente la primera puede deformar las distribuciones char-level de formas que rompen la segunda — es over-optimization sobre la proxy.

2. **¿Cual de las cuatro hipotesis te parece mas probable?**
   Sin respuesta canonica. Probablemente una mezcla de (1) over-optimization y (4) iters demasiado. La forma de saberlo es correr los experimentos. Te invito a pensarlo y, cuando tengas tiempo, probar.

3. **¿Para que sirve DPO entonces si baja la accuracy?**
   Para mantener el **formato** sin colapsar — el drift sobrevive en 0%. Y en datasets reales con preferencias bien curadas y hyperparams ajustados, DPO si mejora — es lo que esta detras de los modelos `-Instruct` y `-DPO` que usas todos los dias. Aqui mostramos los tradeoffs en miniatura, en un entorno donde podes ver cada componente.
