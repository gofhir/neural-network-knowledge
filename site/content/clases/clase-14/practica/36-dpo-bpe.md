---
title: "36 - DPO-BPE: beta sweep y validacion de hipotesis cap 29"
weight: 360
math: true
---

## 1. Apertura

El [capitulo 29]({{< relref "29-dpo-training-eval" >}}) cerro el Camino 2 (char-level) con un puzzle: SFT funcionaba bien, pero DPO **degradaba** la accuracy en casi todas las tareas. En el cierre de ese capitulo propusimos cuatro hipotesis para explicar la regresion. La principal — la mas probable, la mas falsable — fue **"`beta=0.1` demasiado bajo"**: la policy se alejo de la referencia con tanta libertad que no solo aprendio a discriminar `chosen` de `rejected`, sino que rompio el comportamiento aprendido en SFT.

Aqui validamos esa hipotesis empiricamente. Tomamos el BPE-SFT del [cap 34]({{< relref "34-sft-bpe" >}}) como starting point (policy y ref ambos cargados desde alli), corremos DPO sobre los 3000 triples del [cap 35]({{< relref "35-dataset-dpo-bpe" >}}), y barremos dos betas: `beta=0.1` (el del cap 29) y `beta=0.5` (cinco veces mas restrictivo). Si la hipotesis es correcta, `beta=0.5` deberia preservar mucho mas la accuracy del SFT. Si fuera incorrecta, ambos betas degradarian igual.

Spoiler honesto: la hipotesis se valida **parcialmente**. `beta=0.5` preserva mucho mejor que `beta=0.1`, pero **DPO sigue degradando vs SFT puro** incluso con el beta alto. Las otras tres hipotesis del cap 29 (over-optimization, ruido cross-task, demasiados iters) probablemente tambien aplican.

---

## 2. El experimento

Setup minimalista para que la unica variable libre sea `beta`:

| Hyperparam     | Valor                                  |
|----------------|----------------------------------------|
| Policy init    | `mini_llama_bpe_sft.pt` (cap 34)       |
| Reference init | `mini_llama_bpe_sft.pt` (frozen)       |
| Dataset        | 3000 triples DPO-BPE (cap 35)          |
| Iters          | 1000                                   |
| Batch          | 16                                     |
| LR             | 5e-5                                   |
| Weight decay   | 0.01                                   |
| Optimizer      | AdamW                                  |
| Beta sweep     | {0.1, 0.5}                             |

La referencia se inicializa identica a la policy (`bpe-sft`) y se congela (`requires_grad_(False)`). Esa es la version cap 35-36 del [KL implicito]({{< relref "kl-implicito" >}}): la policy se mide contra una copia exacta de si misma al inicio, y el termino de regularizacion en la loss DPO penaliza alejarse demasiado.

Dos checkpoints de salida:

- `mini_llama_bpe_dpo_b01.pt` (beta=0.1)
- `mini_llama_bpe_dpo_b05.pt` (beta=0.5)

Y al final, eval comparativa de los tres modelos: BPE-SFT (baseline), DPO-b01, DPO-b05 — sobre las mismas 200 ejemplos por tarea del eval set BPE.

---

## 3. El script

`clase_14/practica/36_train_dpo_bpe.py`:

```python
"""36_train_dpo_bpe.py - Cap 36: DPO-BPE + beta sweep.

Prueba beta=0.1 y beta=0.5 para validar hipotesis del cap 29
(DPO char-level degrado con beta=0.1 demasiado bajo).
"""
import torch
from pathlib import Path
from _bpe import BPETokenizer
from _models import load_pretrained_mini_llama, dpo_loss, get_device
from _eval import load_jsonl, eval_exact_match, eval_drift

torch.manual_seed(1337)
device = get_device()

tok = BPETokenizer.load("data/bpe_tokenizer.json")
vocab_size = tok.vocab_size
cfg = dict(vocab_size=vocab_size, max_seq_len=256,
           d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384)

ITERS = 1000
BATCH = 16
LR = 5e-5
WD = 0.01

triples = load_jsonl("data/dpo_bpe_dataset.jsonl")
print(f"Loaded {len(triples)} DPO-BPE triples\n")

def encode(s):
    return torch.tensor([tok.vocab.get(c, 0) for c in s], dtype=torch.long)

def run_dpo(beta, out_ckpt):
    print(f"=== DPO-BPE beta={beta} ===")
    policy = load_pretrained_mini_llama("checkpoints/mini_llama_bpe_sft.pt",
                                        device=device, config=cfg)
    ref    = load_pretrained_mini_llama("checkpoints/mini_llama_bpe_sft.pt",
                                        device=device, config=cfg)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval(); policy.train()

    opt = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=WD)

    for it in range(ITERS):
        losses = []
        for _ in range(BATCH):
            t = triples[torch.randint(0, len(triples), (1,)).item()]
            p_ids = encode(t["prompt"])
            c_ids = encode(t["chosen"])
            r_ids = encode(t["rejected"])
            l = dpo_loss(policy, ref, p_ids, c_ids, r_ids, beta=beta, device=device)
            losses.append(l)
        loss = torch.stack(losses).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 50 == 0 or it == ITERS - 1:
            print(f"  iter {it:4d}  loss {loss.item():.4f}", flush=True)

    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(policy.state_dict(), out_ckpt)
    print(f"  Saved -> {out_ckpt}\n")
    return policy

for beta, ckpt in [(0.1, "checkpoints/mini_llama_bpe_dpo_b01.pt"),
                   (0.5, "checkpoints/mini_llama_bpe_dpo_b05.pt")]:
    run_dpo(beta, ckpt)

# Eval comparativa final
print("=== Eval final: BPE-SFT vs DPO-b01 vs DPO-b05 ===\n")
results = {}
for name, ckpt in [("bpe-sft",   "checkpoints/mini_llama_bpe_sft.pt"),
                   ("dpo-b01",   "checkpoints/mini_llama_bpe_dpo_b01.pt"),
                   ("dpo-b05",   "checkpoints/mini_llama_bpe_dpo_b05.pt")]:
    m = load_pretrained_mini_llama(ckpt, device=device, config=cfg)
    em = eval_exact_match(m, "data/sft_bpe_eval.jsonl", tok, n_per_task=200, device=device)
    results[name] = em
    print(f"{name}: {em}")

print("\n=== Tabla final ===")
print(f"{'task':<15}{'bpe-sft':<12}{'dpo-b01':<12}{'dpo-b05':<12}")
for task in ["qa", "repeat", "complete-en", "complete-es"]:
    s  = results["bpe-sft"].get(task, 0.0)
    d1 = results["dpo-b01"].get(task, 0.0)
    d5 = results["dpo-b05"].get(task, 0.0)
    print(f"{task:<15}{s:<12.3f}{d1:<12.3f}{d5:<12.3f}")

print("\n=== Drift BPE-SFT vs DPO-b01 vs DPO-b05 ===")
ambiguous = ["INSTR: capitalize 'cat'\nRESP: ", "Q: what is 2+2?\nA: "]
for name, ckpt in [("bpe-sft",  "checkpoints/mini_llama_bpe_sft.pt"),
                   ("dpo-b01",  "checkpoints/mini_llama_bpe_dpo_b01.pt"),
                   ("dpo-b05",  "checkpoints/mini_llama_bpe_dpo_b05.pt")]:
    m = load_pretrained_mini_llama(ckpt, device=device, config=cfg)
    drift = eval_drift(m, ambiguous, tok, device=device)
    print(f"{name}: drift = {drift:.3f}")
```

---

## 4. Loss curves — ambas convergen, una mas oscilante

El output literal del entrenamiento:

```
Loaded 3000 DPO-BPE triples

=== DPO-BPE beta=0.1 ===
  iter    0  loss 0.6931
  iter   50  loss 0.1945
  iter  100  loss 0.1376
  iter  150  loss 0.0884
  iter  200  loss 0.0344
  iter  250  loss 0.0173
  iter  300  loss 0.1546
  iter  350  loss 0.0934
  iter  400  loss 0.0117
  iter  450  loss 0.0111
  iter  500  loss 0.0600
  iter  550  loss 0.0373
  iter  600  loss 0.0426
  iter  650  loss 0.0348
  iter  700  loss 0.0210
  iter  750  loss 0.0403
  iter  800  loss 0.0361
  iter  850  loss 0.0149
  iter  900  loss 0.0082
  iter  950  loss 0.0306
  iter  999  loss 0.0008
  Saved -> checkpoints/mini_llama_bpe_dpo_b01.pt

=== DPO-BPE beta=0.5 ===
  iter    0  loss 0.6931
  iter   50  loss 0.0159
  iter  100  loss 0.3706
  iter  150  loss 0.0940
  iter  200  loss 0.0155
  iter  250  loss 0.0026
  iter  300  loss 0.0025
  iter  350  loss 0.0583
  iter  400  loss 0.0142
  iter  450  loss 0.0147
  iter  500  loss 0.0007
  iter  550  loss 0.0221
  iter  600  loss 0.0278
  iter  650  loss 0.0010
  iter  700  loss 0.4095
  iter  750  loss 0.0074
  iter  800  loss 0.0029
  iter  850  loss 0.0595
  iter  900  loss 0.0031
  iter  950  loss 0.0030
  iter  999  loss 0.0007
  Saved -> checkpoints/mini_llama_bpe_dpo_b05.pt
```

Lectura. Ambas curvas parten en el valor teorico de la [Bradley-Terry]({{< relref "bradley-terry" >}}) sin separacion: $-\log \sigma(0) = \log 2 \approx 0.6931$. Ambas convergen a valores casi nulos (~0.0007–0.001) hacia el final, lo que significa que la policy aprende a discriminar `chosen` de `rejected` con altisima confianza ($\sigma(\text{margin}) \to 1$).

`beta=0.5` baja mas rapido al inicio (loss 0.016 en iter 50 vs 0.19 con `beta=0.1`) — coherente: un `beta` mayor amplifica el margin observado por sigmoid. Pero es **mas oscilante**: spikes a 0.37 en iter 100, 0.41 en iter 700. Eso es esperable — la regularizacion implicita es mas dura, asi que cualquier batch con triples ruidosos empuja la loss arriba antes de volver a bajar.

Ambas terminan esencialmente saturadas: la policy ya distingue perfectamente las preferencias del dataset. La pregunta pasa a ser **a costa de que** se aprendio esa discriminacion.

---

## 5. La tabla final — el resultado honesto

```
=== Eval final: BPE-SFT vs DPO-b01 vs DPO-b05 ===

bpe-sft: {'qa': 0.205, 'repeat': 0.785, 'complete-en': 0.005, 'complete-es': 0.065}
dpo-b01: {'qa': 0.205, 'repeat': 0.375, 'complete-en': 0.005, 'complete-es': 0.04}
dpo-b05: {'qa': 0.205, 'repeat': 0.69, 'complete-en': 0.005, 'complete-es': 0.04}

=== Tabla final ===
task           bpe-sft     dpo-b01     dpo-b05
qa             0.205       0.205       0.205
repeat         0.785       0.375       0.690
complete-en    0.005       0.005       0.005
complete-es    0.065       0.040       0.040

=== Drift BPE-SFT vs DPO-b01 vs DPO-b05 ===
bpe-sft: drift = 0.100
dpo-b01: drift = 0.000
dpo-b05: drift = 0.000
```

Cuatro lecturas separadas, una por fila.

**`qa` identico (20.5% en los tres modelos)**. DPO no mueve la aguja en factoid recall. Lectura: los `rejected` de qa son fragmentos de Shakespeare/Quijote, ya muy distantes del `chosen` correcto (`"Madrid\n"`, `"Shakespeare\n"`). La policy ya los rechazaba implicitamente despues del SFT, y el gradient DPO sobre ellos es practicamente cero — no hay mucho que aprender. La accuracy queda intacta porque el optimizador casi no la toca.

**`repeat`: 78.5% → 37.5% (β=0.1) → 69.0% (β=0.5)**. Aqui esta la **validacion parcial de la hipotesis cap 29**. Con `beta=0.1` la accuracy de repeat **se desploma 41 puntos porcentuales** — exactamente el patron del cap 29 char-level. Con `beta=0.5` la caida es de solo 9.5 pp. **Un beta cinco veces mayor preservo cuatro veces mas accuracy**: la hipotesis fue correcta en su direccion. Una regularizacion mas dura mantiene a la policy mas cerca del SFT y evita que el optimizador rompa el comportamiento aprendido.

**`complete-en` y `complete-es` esencialmente cero en los tres**. SFT-BPE ya marcaba 0.5% / 6.5%; DPO no mejora estas tareas. El motivo no es DPO sino la metrica: completacion abierta evaluada con exact-match es brutalmente exigente, y ningun modelo del tamaño Mini-LLaMA acierta el continuation literal con frecuencia. DPO sobre este eval simplemente no tiene espacio donde mejorar.

**`drift` mejora con DPO: 10% → 0% en ambos betas**. El unico punto donde DPO ayuda. Sobre prompts ambiguos (`"INSTR: capitalize 'cat'\nRESP: "`, `"Q: what is 2+2?\nA: "`), el SFT a veces todavia generaba continuaciones tipo Shakespeare (10% drift). DPO refuerza el formato instruido y elimina esa tendencia residual a deslizarse al estilo del dataset de pretrain. Ambos betas igual de efectivos aqui — la senal de drift en los `rejected` (formato tipo libro) es lo bastante fuerte para que ambos converjan.

---

## 6. Validacion de la hipotesis cap 29

Resumen ejecutivo: **si, `beta=0.5` preservo claramente mejor que `beta=0.1`** — la hipotesis principal del cap 29 fue **parcialmente validada**. El KL implicito mas alto evita que la policy se aleje demasiado del SFT, y eso traduce en menos perdida de accuracy en la tarea sensible (repeat).

Lo "parcial" importa. Aunque `beta=0.5` es mejor que `beta=0.1`, **sigue degradando** vs SFT puro: 78.5 → 69.0 en repeat (-9.5 pp). Es decir, no hay un valor de `beta` que evite por completo la regresion en este setup. Las otras tres hipotesis del cap 29 probablemente tambien aplican:

- **Over-optimization**: 1000 iters × 16 batch sobre 3000 triples = ~5.3 epochs efectivos. Con loss casi cero al final, el optimizador estuvo gran parte del tiempo memorizando el dataset, no descubriendo estructura util.
- **Ruido cross-task**: la mitad de los triples son "responde con formato A no con formato B". Esa senal puede entrar en conflicto con prompts donde el formato correcto no esta perfectamente determinado.
- **Demasiados iters**: el loss colapso a 0.001 en iter 500-600; los siguientes 400 iters fueron sobreajuste al ruido.

Combinacion realista: `beta` correcto **mitigaria** la degradacion pero no la elimina. Para eliminarla, habria que reducir iters, mejorar la calidad de las preferencias, o ambas.

---

## 7. Que ensena esto sobre DPO

Tres parrafos que conviene fijar.

**DPO es una herramienta para alinear comportamiento, no para subir accuracy.** El unico eje donde DPO mejora el modelo es `drift` — es decir, "no te deslices al estilo del dataset de pretrain cuando el prompt es ambiguo". Eso es alineamiento de formato. En todas las tareas con respuesta verificable (qa, repeat, complete-*), DPO o no afecta o degrada. Esto coincide con la intuicion del [DPO paper original]({{< relref "dpo" >}}): DPO no le ensena al modelo cosas nuevas, lo afina sobre comportamientos ya presentes en SFT.

**En produccion, DPO se usa con preferencias humanas curadas**. Llama-3-Instruct, Claude, GPT-4 — todos pasan por una fase tipo DPO/RLHF, pero con datasets de decenas de miles de pares evaluados por anotadores humanos sobre criterios cualitativos sutiles (utilidad, tono, seguridad, factualidad). Aqui usamos preferencias **sinteticas** (base-sampled + cross-task): la senal es ruidosa, los rejected son obviamente malos en muchos casos, y no hay nada de la riqueza de un juicio humano. Que aun asi DPO mejore drift y deje el resto estable es un resultado razonable para preferencias sinteticas.

**El stack moderno tiene tres pasos por buena razon**: pretrain (capacidad bruta — saber que existen las palabras), SFT (formato y comportamiento — saber responder a un prompt instruccional), DPO/RLHF (alineamiento fino — preferencias cualitativas). DPO **no reemplaza** SFT: lo refina. Si saltaras de pretrain directo a DPO, no tendrias el comportamiento base que DPO necesita ajustar — la policy no sabria responder en formato instruccional para empezar. Cap 22 mostro que el base-model es esencialmente inutil sin SFT; este capitulo muestra que SFT sin DPO ya hace casi todo el trabajo.

---

## 8. Preguntas de verificacion

1. **¿Por que `beta=0.5` degrada menos que `beta=0.1`?** Porque el [KL implicito]({{< relref "kl-implicito" >}}) es mas restrictivo. La loss DPO se puede leer como "maximiza el margin chosen-rejected con un termino de regularizacion KL hacia la referencia escalado por `1/beta`". Cuando `beta` sube, el peso de la regularizacion sube tambien, y la policy no puede alejarse mucho del SFT inicial. Eso preserva el comportamiento aprendido a costa de margins menos extremos — pero los margins ya saturan en ambos betas, asi que el costo es minimo y el beneficio (no romper el SFT) es grande.

2. **¿Por que DPO no mejora `qa` pero ayuda con `drift`?** El gradient DPO empuja `log P(chosen)` hacia arriba y `log P(rejected)` hacia abajo. En `qa`, los rejected son fragmentos largos de Shakespeare totalmente distintos al chosen `"Madrid\n"`; las distribuciones casi no se solapan, asi que el gradient sobre ellos es ya casi cero — la policy SFT ya los rechaza. En `drift`, los rejected suprimidos son completaciones Shakespeare-like sutiles que SI competian probabilisticamente con la respuesta correcta cuando el prompt era ambiguo; ahi el gradient DPO si empuja la masa de probabilidad de la policy y se nota.

3. **¿Si DPO mejorara TODO dramaticamente, que querria decir?** Casi seguro **over-fitting al dataset de preferencias**. DPO con preferencias bien curadas mejora marginalmente — uno o dos puntos en MMLU, un poco mas en utilidad subjetiva. Si vieras saltos de 30 pp en accuracy, lo razonable es sospechar que el modelo memorizo el formato exacto del dataset DPO y la metrica esta detectando esa memorizacion (data leakage entre train y eval, o triples que comparten estructura con el eval). En aprendizaje real con humanos, el techo es bajo y se llega despacio.

---

## 9. Lo que viene

El [cap 37]({{< relref "37-comparacion-char-vs-bpe" >}}) cierra el Camino 2.5 con una tabla maestra que pone los seis modelos lado a lado: `char-base`, `char-sft`, `char-dpo`, `bpe-base`, `bpe-sft`, `bpe-dpo-b05`. Es la comparativa final char-level vs BPE-level — y la respuesta a la pregunta que nunca aparece en blogs: **¿cuando conviene cada tokenizacion?**
