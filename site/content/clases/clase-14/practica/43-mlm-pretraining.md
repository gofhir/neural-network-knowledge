---
title: "43 - MLM Pretraining"
weight: 430
math: true
---

## 1. Apertura: el mismo loop, nuevas piezas

Cap 42 definio el objetivo: cross-entropy sobre tokens enmascarados, `ignore_index=-100`, split 80/10/10. Ahora entrenamos.

Los mismos 3000 iters, el mismo loop de training que conoces del cap 08 (Mini-GPT) y cap 31 (BPE pretrain). Lo nuevo: `get_batch()` agrega `[CLS]`/`[SEP]` a cada ventana y aplica masking 80/10/10 antes de enviar al modelo. El encoder bidireccional MiniBERT ve el corpus completo — Shakespeare + Quijote — y aprende a predecir tokens en contexto.

---

## 2. get_batch(): ventanas + tokens especiales + masking

La funcion `get_batch()` tiene tres etapas:

**Etapa 1 — ventanas aleatorias**: se muestrean `BATCH=32` indices al azar y se extraen ventanas de `BLOCK=64` tokens del corpus tokenizado. El resultado es un tensor `(32, 64)`.

**Etapa 2 — formato BERT**: se prepende una columna `[CLS]` y se agrega una columna `[SEP]`, dando `input_ids` de forma `(32, 66)`. BERT siempre recibe secuencias con esos tokens en los extremos — el vector de `[CLS]` acumulara la representacion global de la secuencia.

**Etapa 3 — MLM masking**: se llama a `apply_mlm_mask` del cap 42 con `mask_prob=0.15`. Devuelve dos tensores:
- `masked_ids`: la secuencia con ~15% de tokens reemplazados (80% por `[MASK]`, 10% aleatorio, 10% sin cambio)
- `labels`: tensor con `token_id` en las posiciones enmascaradas y `-100` en el resto

El modelo ve `masked_ids` y solo calcula loss sobre las posiciones con `label != -100`.

```python
def get_batch():
    ix = torch.randint(0, len(data) - BLOCK, (BATCH,))
    windows = torch.stack([data[i:i+BLOCK] for i in ix])  # (B, 64)
    cls_col = torch.full((BATCH, 1), tok.cls_id, dtype=torch.long)
    sep_col = torch.full((BATCH, 1), tok.sep_id, dtype=torch.long)
    input_ids = torch.cat([cls_col, windows, sep_col], dim=1)  # (B, 66)
    masked_ids, labels = apply_mlm_mask(
        input_ids.clone(), mask_prob=0.15,
        mask_id=tok.mask_id, vocab_size=vocab_size,
        special_ids=(tok.cls_id, tok.sep_id, tok.mask_id)
    )
    return masked_ids.to(device), labels.to(device)
```

---

## 3. El loop de training — igual que siempre

El loop es identico al del cap 08 y cap 31: forward, loss, backward, step. La unica diferencia es que ahora hay dos modulos con parametros conjuntos: `MiniBERT` (el encoder) y `MLMHead` (la cabeza de prediccion).

```python
for it in range(ITERS):
    masked_ids, labels = get_batch()
    h      = model(masked_ids)          # encoder: (B, 66, 128)
    logits = mlm_head(h)                # head: (B, 66, vocab_size)
    loss   = F.cross_entropy(logits.view(-1, vocab_size),
                              labels.view(-1), ignore_index=-100)
    opt.zero_grad(); loss.backward(); opt.step()
```

El flujo completo:
1. `model(masked_ids)` — el encoder bidireccional procesa la secuencia con atencion completa (sin mascara causal)
2. `mlm_head(h)` — proyeccion lineal de `d_model=128` a `vocab_size=1115` para cada posicion
3. `F.cross_entropy(..., ignore_index=-100)` — loss solo sobre posiciones enmascaradas (~15% de 66 = ~10 tokens por secuencia)
4. backward + step — los gradientes fluyen a traves del `MLMHead` y de todos los `BERTBlock` del encoder

El optimizador `AdamW` con `lr=1e-4` y `weight_decay=0.01` aplica regularizacion L2 a todos los parametros — importante en encoders donde el overfitting es mas facil que en decoders autorregresivos.

---

## 4. Script completo

```python
"""43_train_bert.py - Cap 43: MLM pretraining de Mini-BERT."""
import torch
import torch.nn.functional as F
from pathlib import Path
from _bpe import BPETokenizer
from _models import MiniBERT, MLMHead, get_device
from _bert_utils import apply_mlm_mask

torch.manual_seed(1337)
device = get_device()

tok = BPETokenizer.load("data/bpe_tokenizer.json")
tok.add_special_tokens()
vocab_size = tok.vocab_size

en = Path("shakespeare.txt").read_text(encoding="utf-8")
es = Path("quijote.txt").read_text(encoding="utf-8")
corpus = en + "\n" + es
data = torch.tensor(tok.encode(corpus), dtype=torch.long)
print(f"Corpus: {len(data):,} tokens")

BLOCK = 64   # longitud de secuencia (sin [CLS][SEP] la ventana real es 62)
BATCH = 32
LR    = 1e-4
ITERS = 3000
WD    = 0.01

model    = MiniBERT(vocab_size=vocab_size, max_seq_len=BLOCK+2,
                    d_model=128, n_heads=4, n_layers=4, d_ff=512).to(device)
mlm_head = MLMHead(d_model=128, vocab_size=vocab_size).to(device)

params = list(model.parameters()) + list(mlm_head.parameters())
opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)

n_params = sum(p.numel() for p in params)
print(f"Params: {n_params:,}\n")

def get_batch():
    """Muestrea ventanas aleatorias y las formatea como BERT input."""
    ix = torch.randint(0, len(data) - BLOCK, (BATCH,))
    windows = torch.stack([data[i:i+BLOCK] for i in ix])  # (B, 64)
    # Agregar [CLS] al inicio y [SEP] al final
    cls_col = torch.full((BATCH, 1), tok.cls_id, dtype=torch.long)
    sep_col = torch.full((BATCH, 1), tok.sep_id, dtype=torch.long)
    input_ids = torch.cat([cls_col, windows, sep_col], dim=1)  # (B, 66)
    masked_ids, labels = apply_mlm_mask(
        input_ids.clone(), mask_prob=0.15,
        mask_id=tok.mask_id, vocab_size=vocab_size,
        special_ids=(tok.cls_id, tok.sep_id, tok.mask_id)
    )
    return masked_ids.to(device), labels.to(device)

print(f"MLM pretraining: {ITERS} iters\n")
for it in range(ITERS):
    masked_ids, labels = get_batch()
    h      = model(masked_ids)
    logits = mlm_head(h)
    loss   = F.cross_entropy(logits.view(-1, vocab_size),
                              labels.view(-1), ignore_index=-100)
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 300 == 0 or it == ITERS - 1:
        print(f"iter {it:4d}  loss {loss.item():.4f}", flush=True)

Path("checkpoints").mkdir(exist_ok=True)
torch.save({
    "model": model.state_dict(),
    "mlm_head": mlm_head.state_dict(),
    "config": dict(vocab_size=vocab_size, max_seq_len=BLOCK+2,
                   d_model=128, n_heads=4, n_layers=4, d_ff=512),
}, "checkpoints/mini_bert_pretrained.pt")
print("\nSaved -> checkpoints/mini_bert_pretrained.pt")
```

---

## 5. Output del training

```
Corpus: 1,606,374 tokens
Params: 1,088,347

MLM pretraining: 3000 iters

iter    0  loss 7.1173
iter  300  loss 5.3326
iter  600  loss 5.5521
iter  900  loss 5.2230
iter 1200  loss 5.2286
iter 1500  loss 5.1774
iter 1800  loss 5.1349
iter 2100  loss 5.0464
iter 2400  loss 4.9918
iter 2700  loss 4.9614
iter 2999  loss 4.9573

Saved -> checkpoints/mini_bert_pretrained.pt
```

---

## 6. Analisis de la curva de loss

**Inicio: 7.1173** — el modelo tiene pesos aleatorios. La loss inicial es ligeramente superior a $\log(1115) \approx 7.02$, lo mismo que observamos en el cap 42 con el modelo sin entrenar. Consistente.

**Caida rapida en iter 300: 5.33** — una caida de ~1.8 en las primeras 300 iteraciones. El modelo aprende rapido que algunos tokens son mas frecuentes que otros. La entropia de una distribucion sesgada es menor que la de la distribucion uniforme — el modelo ya sabe "predecir lo comun".

**Convergencia lenta 600-2999: 5.55 → 4.96** — despues del dropout inicial de entropia, la curva se aplana. Hay incluso un repunte en iter 600 (5.55 > 5.33). Esto es normal: el optimizer explora el espacio de parametros y a veces da pasos hacia arriba en el paisaje de loss antes de encontrar un camino mas profundo.

**Final: 4.9573** — la loss convergio alrededor de ~5.0.

**Comparacion con CLM (cap 31):**

| Modelo | Objetivo | Loss iter 0 | Loss iter 3000 | Caida total |
|--------|----------|-------------|----------------|-------------|
| Mini-GPT BPE (cap 31) | CLM (next-token) | ~6.56 | ~1.0-2.0 | ~4.5-5.5 |
| Mini-BERT (cap 43) | MLM (masked token) | 7.12 | 4.96 | ~2.2 |

MLM es **mas dificil** que CLM en este contexto, por dos razones:

1. **Objetivo mas ruidoso**: el decoder CLM predice el siguiente token dado todo lo anterior — hay una correlacion fuerte y direccional. El encoder MLM predice tokens en cualquier posicion con contexto bidireccional, pero solo el 15% de posiciones contribuye a la loss — la senal de gradiente es escasa por iteracion.

2. **Sin ventaja left-to-right**: el decoder sabe que si predice correctamente en posicion $t$, puede usar esa prediccion como contexto para $t+1$. El encoder no tiene esa "trampa" — debe resolver cada posicion enmascarada de forma independiente con contexto bidireccional fijo.

3. **Corpus pequeño, modelo sin pretraining externo**: BERT-base real se preentrenar sobre BookCorpus + Wikipedia (3.3 mil millones de tokens). Nuestro Mini-BERT ve 1.6 millones de tokens con 1M parametros. La comparacion justa no es con BERT-base sino con el tamano relativo del experimento.

La loss final de ~4.96 indica que el modelo ha aprendido estructura estadistica del corpus — es mejor que azar (7.02) pero lejos de memorizacion perfecta. Eso es exactamente lo que queremos para un pretrained encoder: representaciones generalizables, no overfitting.

---

## 7. Que aprendio el modelo

El checkpoint `mini_bert_pretrained.pt` contiene un encoder que:

**Ha visto el corpus completo bidirecionalmente**: a diferencia del decoder que solo mira hacia la izquierda, cada capa de atencion en MiniBERT accede a todos los tokens de la secuencia al mismo tiempo. En 3000 iteraciones con batches de 32 secuencias de 66 tokens, el modelo ha procesado ~6.3 millones de pares (posicion, contexto).

**Puede predecir tokens en contexto**: la tarea MLM obliga al modelo a construir representaciones que capturen el significado de cada token en funcion de sus vecinos. Un token como "be" en "to be or not to be" recibe una representacion diferente de "be" en "to be continued" — el contexto bidireccional diferencia los usos.

**Base para fine-tuning (caps 45-47)**: el encoder preentrenado es ahora un extractor de caracteristicas. Para tareas downstream:
- **Clasificacion de texto**: se toma el vector `[CLS]` (posicion 0) y se pasa por una `ClassificationHead` lineal. Solo se necesitan pocas iteraciones de fine-tuning con datos etiquetados.
- **NER / clasificacion de tokens**: se usan los vectores de cada posicion (no solo `[CLS]`).
- **QA**: el encoder procesa la pregunta + contexto concatenados con `[SEP]` entre ellos, y predice las posiciones de inicio y fin de la respuesta.

En todos estos casos, el pretrained encoder aporta representaciones contextuales ricas que el fine-tuning adapta al task especifico con mucho menos datos que entrenar desde cero.

---

## 8. Preguntas de verificacion

1. **La loss MLM final (~4.96) es mayor que la loss CLM final del cap 31 (~1.5-2.0). Significa eso que Mini-BERT es un modelo peor?** No — las dos losses no son comparables directamente. CLM predice el siguiente token dado todo el contexto izquierdo, una tarea con dependencias fuertes y seriales que facilita el aprendizaje. MLM predice tokens enmascarados con 15% de cobertura — la senal de gradiente es mas escasa por iteracion. Ademas, CLM puede overfit facilmente porque los tokens de training se predicen en orden secuencial; MLM tiene un efecto regularizador natural porque la mascara es aleatoria en cada batch.

2. **El repunte de loss en iter 600 (5.55 > 5.33 del iter 300): es un bug?** No — es una fluctuacion normal del SGD estocastico. Cada batch muestrea ventanas diferentes y aplica masking diferente. La loss reportada es la de un solo batch, no el promedio sobre el dataset. Fluctuaciones de +/- 0.3 son esperables. Lo relevante es la tendencia descendente a largo plazo (7.12 → 4.96).

3. **Por que el checkpoint guarda `mlm_head` junto con `model` si en fine-tuning se reemplaza la cabeza?** Durante pretraining, el `mlm_head` es necesario para calcular la loss MLM y entrenar el encoder. En fine-tuning, se descarta `mlm_head` y se conecta una nueva cabeza (clasificacion, NER, QA). Guardar ambos en el checkpoint es util para: (a) resumir pretraining si se interrumpe, (b) calcular metricas de evaluacion MLM despues del fine-tuning para comparar, (c) depuracion. El campo `config` en el checkpoint permite reconstruir la arquitectura sin hardcodear parametros.
