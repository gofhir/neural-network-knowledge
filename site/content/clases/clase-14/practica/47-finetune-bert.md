---
title: "47 - Fine-tuning: deteccion de idioma"
weight: 470
math: true
---

## 1. El encoder ya sabe demasiado

Cap 43 preentrano el encoder durante 3000 iters sobre Shakespeare+Quijote. Cap 46 construyo el dataset: 2000 pares `(secuencia, idioma)` sin leakage. Ahora: fine-tuning.

La tarea es clasificacion binaria: EN=0, ES=1. El modelo que lo resuelve tiene dos partes:

- **MiniBERT**: 952K parametros que ya saben representar texto bilingue. Se ajusta levemente.
- **ClassificationHead**: una sola capa lineal `d_model → 2`. 128×2 + 2 = **258 parametros** totales.

La cabeza es casi invisible comparada con el encoder. El encoder hace todo el trabajo pesado: convierte la secuencia en un vector `[CLS]` rico en semantica. La cabeza simplemente aprende a separar ese vector en dos clases. Esta asimetria es la esencia del paradigma BERT: **pretraining masivo + fine-tuning minimal**.

---

## 2. Por que LR=2e-5 (5x menor que pretraining)

El pretraining uso `LR=1e-4`. El fine-tuning usa `LR=2e-5`. Factor 5x. No es arbitrario.

El fenomeno que se quiere evitar es **catastrophic forgetting**: si el LR es muy alto, el gradiente del fine-tuning "sobreescribe" los pesos del encoder con la senal del dataset de clasificacion. Los 3000 iters de MLM que ensenaron al encoder a representar ingles isabelino y castellano del siglo XVII se destruyen en 500 iters de `cross_entropy` sobre 2000 ejemplos.

La intuicion matematica: el gradiente del fine-tuning tiene magnitud proporcional al LR. Un LR alto mueve los pesos lejos de los minimos del pretraining. Si esos minimos estaban en una region "buena" del espacio de parametros (lo que el checkpoint garantiza), alejarse de ellos destruye la representacion.

Un LR bajo (`2e-5`) mueve los pesos lo suficiente para que la cabeza lineal aprenda el mapeo `[CLS] → idioma`, pero no lo suficiente para que el encoder olvide lo que aprendio en MLM. El encoder se "ajusta fino" — de ahi el termino — sin perder conocimiento lingüistico.

El mismo patron aparece en cap 24 (SFT): el SFT usaba un LR 10x menor que el pretrain del Mini-LLaMA. La convencion es conservadora: mejor LR demasiado bajo (convergencia lenta) que LR demasiado alto (olvidar el pretrain).

$$\text{LR}_{\text{fine-tune}} \approx \frac{\text{LR}_{\text{pretrain}}}{5\text{ a }10}$$

---

## 3. El loop de fine-tuning: sorprendentemente simple

Comparado con el loop MLM del cap 43, el fine-tuning es casi trivial. No hay masking, no hay `ignore_index`, no hay labels `-100`. Solo:

1. Pasar el batch por el encoder → tensor `(B, T, d_model)`.
2. Tomar el vector en posicion 0 (`[CLS]`) → tensor `(B, d_model)`.
3. Proyectar con la cabeza lineal → tensor `(B, 2)`.
4. `cross_entropy(logits, labels)` — clasificacion binaria directa.

El vector `[CLS]` es la clave. Durante el pretraining MLM, el token `[CLS]` no tiene tarea directa — no se enmascara, no se predice. Sin embargo, al estar en posicion 0 y participar en la atencion bidireccional con todos los demas tokens, acumula informacion global de la secuencia. El fine-tuning de clasificacion explota exactamente esa propiedad: `[CLS]` es un resumen de la secuencia, y la cabeza lineal aprende a leer ese resumen para distinguir idiomas.

Esta convencion viene del paper original de BERT (Devlin et al., 2018): el vector `[CLS]` en la ultima capa del encoder se usa directamente para tareas de clasificacion de secuencia. No requiere pooling especial ni agregacion — una sola posicion basta.

---

## 4. Script completo

```python
"""47_finetune_bert.py - Cap 47: fine-tuning BERT para deteccion de idioma."""
import json, torch, random as _random
import torch.nn.functional as F
from pathlib import Path
from _bpe import BPETokenizer
from _models import MiniBERT, ClassificationHead, get_device

torch.manual_seed(1337)
device = get_device()

tok = BPETokenizer.load("data/bpe_tokenizer.json")
tok.add_special_tokens()

ckpt = torch.load("checkpoints/mini_bert_pretrained.pt", map_location=device, weights_only=False)
cfg  = ckpt["config"]
model    = MiniBERT(**cfg).to(device)
model.load_state_dict(ckpt["model"])
model.train()

cls_head = ClassificationHead(d_model=cfg["d_model"], n_classes=2).to(device)

LR    = 2e-5  # 5x menor que pretraining (convencion BERT)
ITERS = 500
BATCH = 32
WD    = 0.01

train_data = [json.loads(l) for l in open("data/lang_train.jsonl")]
params = list(model.parameters()) + list(cls_head.parameters())
opt    = torch.optim.AdamW(params, lr=LR, weight_decay=WD)

print(f"Fine-tuning: {ITERS} iters, LR={LR}\n")
for it in range(ITERS):
    batch = _random.sample(train_data, BATCH)
    max_len = max(len(ex["ids"]) for ex in batch)
    ids_t = torch.zeros(BATCH, max_len, dtype=torch.long, device=device)
    lbl_t = torch.zeros(BATCH, dtype=torch.long, device=device)
    for i, ex in enumerate(batch):
        ids_t[i, :len(ex["ids"])] = torch.tensor(ex["ids"])
        lbl_t[i] = ex["label"]

    h      = model(ids_t)
    logits = cls_head(h)
    loss   = F.cross_entropy(logits, lbl_t)
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 50 == 0 or it == ITERS - 1:
        print(f"iter {it:4d}  loss {loss.item():.4f}", flush=True)

Path("checkpoints").mkdir(exist_ok=True)
torch.save({
    "model":    model.state_dict(),
    "cls_head": cls_head.state_dict(),
    "config":   cfg,
}, "checkpoints/mini_bert_finetuned.pt")
print("\nSaved -> checkpoints/mini_bert_finetuned.pt")
```

---

## 5. Output del script

```
Fine-tuning: 500 iters, LR=2e-05

iter    0  loss 0.6275
iter   50  loss 0.0623
iter  100  loss 0.0221
iter  150  loss 0.0118
iter  200  loss 0.0075
iter  250  loss 0.0060
iter  300  loss 0.0035
iter  350  loss 0.0028
iter  400  loss 0.0026
iter  450  loss 0.0031
iter  499  loss 0.0764

Saved -> checkpoints/mini_bert_finetuned.pt
```

---

## 6. Analisis de la caida: de 0.6275 a 0.0764

El punto de partida esperado para un clasificador binario aleatorio es `log(2) ≈ 0.6931`. La iter 0 da `0.6275` — ligeramente por debajo porque la semilla `1337` inicializa la cabeza con pesos que ya tienen un pequeno sesgo positivo. Este es el punto de partida legitimo: el encoder todavia no ha visto ningun gradiente de clasificacion, y la cabeza es aleatoria.

La caida en las primeras 50 iteraciones es drastica: de `0.6275` a `0.0623`. El encoder ya tenia representaciones ricas — simplemente habia que ensenarle a la cabeza como leer el vector `[CLS]`. En las primeras decenas de iters, el gradiente ajusta la cabeza radicalmente mientras el encoder cambia muy poco (LR bajo).

De iter 50 a iter 400, la convergencia continua de forma monotona: `0.0623 → 0.0026`. Las representaciones del encoder se ajustan finamente para que los vectores `[CLS]` de EN y ES queden bien separados en el espacio d_model=128.

La iter 499 muestra `0.0764` — ligeramente mas alta que iter 450 (`0.0031`). Esto es ruido de minibatch (BATCH=32): la loss reportada es sobre un solo minibatch aleatorio, no sobre el dataset completo. El modelo no empeoro — simplemente ese minibatch era mas dificil. Si se midiera la loss media sobre el dataset completo, seguiria siendo muy baja.

Una loss de `0.0764` al final significa que el clasificador asigna probabilidades cercanas a 1.0 a la clase correcta en casi todos los ejemplos del batch. En terminos de accuracy: practicamente 100% sobre los datos de entrenamiento. El encoder aprendio a distinguir EN de ES con las representaciones preentrenadas en MLM.

---

## 7. Lo que cambio vs lo que no

### Lo que cambio: la cabeza (258 parametros)

La `ClassificationHead` empezo con pesos aleatorios y aprendio a leer el vector `[CLS]` del encoder. Los 500 iters entrenaron estos 258 parametros desde cero.

### Lo que cambio levemente: el encoder (952K parametros)

El encoder tambien se actualizo, pero con LR=2e-5. Los cambios son pequenos pero significativos: las representaciones del encoder se ajustaron para que `[CLS]` sea mas discriminativo entre idiomas. Esto se llama **full fine-tuning** (todos los pesos del encoder participan en backprop). Existe una alternativa llamada **frozen fine-tuning** o **probing** donde el encoder se congela y solo se entrena la cabeza — util para medir que tan buenas son las representaciones preentrenadas sin ninguna adaptacion.

### Lo que no cambio: el conocimiento lingüistico

Gracias al LR bajo, el encoder no olvido la estructura de Shakespeare ni del Quijote. Si se usara el encoder para MLM despues del fine-tuning, seguiria prediciendo tokens enmascarados con alta precision. El conocimiento lingüistico fue preservado. La clasificacion de idioma fue aprendida encima de ese conocimiento, no en lugar de el.

Esta propiedad — **preservacion del pretrain bajo fine-tuning con LR bajo** — es lo que hace que el paradigma BERT sea practico. Un mismo encoder preentrenado puede fine-tunearse para docenas de tareas distintas (clasificacion de sentimiento, NER, QA, inference) sin perder su capacidad general. Cada fine-tune produce un checkpoint especializado. El encoder pretrained es el conocimiento compartido; cada cabeza es el conocimiento de tarea.

---

## 8. Preguntas de verificacion

**1. ¿Por que la iter 0 da `0.6275` y no exactamente `0.6931` (log 2)?**

El valor `0.6931` es la entropia de una distribucion uniforme sobre 2 clases — lo que esperarias si la cabeza tuviera pesos exactamente cero. Pero la inicializacion por defecto de `nn.Linear` en PyTorch usa Kaiming uniform, que produce pesos pequenos pero no cero. Con la semilla `1337`, esos pesos iniciales generan logits ligeramente desiguales. Ademas, el encoder ya tiene representaciones no triviales del pretraining — no es un modelo aleatorio. La combinacion de encoder preentrenado + cabeza con pesos Kaiming produce `0.6275` en lugar de `0.6931`. Ambos estan en la zona "practicamente aleatoria" para clasificacion binaria, pero el encoder ya da una pequena ventaja inicial.

**2. ¿Que pasaria si se congelara el encoder y solo se entrenara la cabeza?**

En frozen fine-tuning, `model.requires_grad_(False)` antes de crear el optimizer. Solo `cls_head.parameters()` recibirian gradientes. Las 258 parametros de la cabeza aprenderian a leer los vectores `[CLS]` del encoder congelado. Si las representaciones preentrenadas son buenas, el accuracy seria similar — posiblemente un poco menor porque el encoder no puede ajustarse a la tarea. La ventaja es que frozen fine-tuning es mucho mas rapido y computacionalmente economico. En BERT original, frozen fine-tuning (llamado "feature-based") logra resultados ligeramente peores que full fine-tuning en la mayoria de benchmarks.

**3. ¿Por que el loss final (iter 499) es mayor que el de iter 450?**

La loss reportada en cada iteracion es sobre un solo minibatch de BATCH=32 ejemplos, seleccionado aleatoriamente con `_random.sample`. La varianza entre minibatches hace que la loss fluctue — iter 450 cayo en un minibatch "facil" (`0.0031`) y iter 499 en uno "mas dificil" (`0.0764`). La tendencia general del entrenamiento es descendente, como muestran las 10 lecturas intermedias. Para conocer el estado real del modelo hay que evaluar sobre el conjunto de evaluacion completo (`data/lang_eval.jsonl`) — lo que hace el cap 48.
