---
title: "42 - MLM Loss: el objetivo simetrico al SFT"
weight: 420
math: true
---

## 1. Apertura: la misma idea, diferente contexto

Cap 24 te enseno el SFT: `loss_mask=1` sobre los tokens de la RESPUESTA, `loss_mask=0` sobre el prompt. La cross-entropy solo backpropaga donde el modelo debe aprender — el resto se ignora.

MLM es exactamente lo mismo, pero en un encoder bidireccional. En vez de un vector binario `loss_mask`, BERT usa `ignore_index=-100`: las posiciones que no fueron enmascaradas tienen `label=-100` y la loss las ignora por completo. El modelo solo predice donde hay una posicion enmascarada — el resto no cuenta.

Son la misma idea con distinto vocabulario:

| Concepto | SFT (decoder) | MLM (encoder) |
|---|---|---|
| Objetivo | predecir tokens de respuesta | predecir tokens enmascarados |
| Ignorar posiciones | `loss_mask=0` | `labels=-100` |
| Penalizar | solo tokens de respuesta | solo tokens enmascarados (~15%) |
| PyTorch | multiplicar loss por mask | `F.cross_entropy(..., ignore_index=-100)` |
| Fraccion activa | ~50% del batch (respuesta) | ~15% de los tokens |

La simetria es exacta: ambos son cross-entropy con una mascara que selecciona "aqui aprendo, alli no".

---

## 2. El split 80/10/10 de BERT

El paper original de BERT no simplemente reemplaza el 15% de tokens con `[MASK]`. Usa un split de tres casos para cada token seleccionado:

- **80% de las veces**: reemplazar con `[MASK]` — el caso obvio
- **10% de las veces**: reemplazar con un token ALEATORIO del vocabulario
- **10% de las veces**: dejar el token SIN CAMBIAR

Por que los tres casos? La razon es pedagogicamente fundamental: **evitar que el modelo aprenda a atender solo los tokens `[MASK]`**.

Si el 100% de los tokens enmascarados se convirtieran en `[MASK]`, el modelo tenderia a ignorar todos los tokens no-MASK durante el forward pass — sabe que la respuesta siempre viene de una posicion marcada con `[MASK]`. El 10% aleatorio y el 10% sin cambio obligan al modelo a evaluar CADA token con atencion: cualquier posicion podria ser la que hay que predecir.

Esto es lo que da a BERT sus representaciones contextuales ricas — el encoder no puede hacer trampa, debe procesar todo el contexto para predecir cualquier posicion.

En `_bert_utils.py`:

```python
if r < 0.80:
    masked_ids[b, t] = mask_id        # [MASK]
elif r < 0.90:
    masked_ids[b, t] = random_token   # token aleatorio
# else: dejar sin cambio (el 10% final)
```

---

## 3. La simetria SFT vs MLM explicada

Comparacion directa del codigo PyTorch:

**SFT (cap 24) — decoder causal:**
```python
# loss_mask: 1 donde response, 0 donde prompt
loss = F.cross_entropy(logits.view(-1, V), targets.view(-1), reduction='none')
loss = (loss * loss_mask.view(-1)).mean()
```

**MLM (cap 42) — encoder bidireccional:**
```python
# labels: token_id donde enmascarado, -100 donde no se predice
loss = F.cross_entropy(
    logits.view(-1, vocab_size),
    labels_dev.view(-1),
    ignore_index=-100
)
```

Ambos calculan cross-entropy sobre el vocabulario completo en cada posicion. Ambos aplican una mascara que silencia las posiciones que no deben contribuir al gradiente. La diferencia es la implementacion tecnica: SFT multiplica manualmente por cero, MLM usa `ignore_index` que hace lo mismo internamente en PyTorch.

---

## 4. Script completo

```python
"""42_mlm_loss.py - Cap 42: MLM masking + 80/10/10 split."""
import torch
import torch.nn.functional as F
from _bpe import BPETokenizer
from _models import MiniBERT, MLMHead, get_device
from _bert_utils import apply_mlm_mask

torch.manual_seed(42)
device = get_device()

tok = BPETokenizer.load("data/bpe_tokenizer.json")
tok.add_special_tokens()
vocab_size = tok.vocab_size

model = MiniBERT(vocab_size=vocab_size, max_seq_len=128,
                 d_model=128, n_heads=4, n_layers=4, d_ff=512).to(device)
mlm_head = MLMHead(d_model=128, vocab_size=vocab_size).to(device)

sentence = "To be or not to be that is the question"
ids = torch.tensor([tok.encode_bert(sentence)], dtype=torch.long)
print(f"Tokens originales ({ids.shape[1]}):")
print(f"  {[tok.id_to_token[i] for i in ids[0].tolist()]}\n")

masked_ids, labels = apply_mlm_mask(ids.clone(), mask_prob=0.15,
                                     mask_id=tok.mask_id, vocab_size=vocab_size)

print("Despues de MLM masking (15%, split 80/10/10):")
for pos, (orig, masked, label) in enumerate(
        zip(ids[0].tolist(), masked_ids[0].tolist(), labels[0].tolist())):
    if label != -100:
        orig_tok   = tok.id_to_token.get(orig, "?")
        masked_tok = tok.id_to_token.get(masked, "?")
        print(f"  pos {pos:2d}: '{orig_tok}' -> '{masked_tok}'  (label={label}, predict='{orig_tok}')")

n_masked = (labels != -100).sum().item()
print(f"\nTokens enmascarados: {n_masked}/{ids.shape[1]} = {n_masked/ids.shape[1]:.1%}")

# Calcular la loss MLM
masked_ids_dev = masked_ids.to(device)
labels_dev     = labels.to(device)
h = model(masked_ids_dev)
logits = mlm_head(h)  # (1, T, vocab_size)

loss = F.cross_entropy(
    logits.view(-1, vocab_size),
    labels_dev.view(-1),
    ignore_index=-100  # ignorar posiciones no enmascaradas
)
print(f"\nMLM loss (modelo random): {loss.item():.4f}")
print(f"Esperado ~log({vocab_size}) = {torch.tensor(vocab_size).float().log().item():.4f}")
print("\nNota: la loss MLM usa ignore_index=-100, igual que SFT usaba loss_mask=0.")
print("Son la misma idea: solo backpropagar donde importa.")
```

---

## 5. Output del script

```
Tokens originales (14):
  ['[CLS]', 'T', 'o ', 'be ', 'or ', 'not ', 'to be ', 'that ', 'is ', 'the ', 'qu', 'es', 'tion', '[SEP]']

Despues de MLM masking (15%, split 80/10/10):
  pos  9: 'the ' -> '[MASK]'  (label=161, predict='the ')
  pos 11: 'es' -> 'F'  (label=134, predict='es')

Tokens enmascarados: 2/14 = 14.3%

MLM loss (modelo random): 7.0601
Esperado ~log(1115) = 7.0166

Nota: la loss MLM usa ignore_index=-100, igual que SFT usaba loss_mask=0.
Son la misma idea: solo backpropagar donde importa.
```

---

## 6. Analisis del output

**Los 14 tokens**: el tokenizador BPE ha fusionado secuencias comunes. "to be " es un solo token (pos 6), "the " es otro token (pos 9). El `[CLS]` y `[SEP]` se ignoran para el masking — `apply_mlm_mask` los excluye por ser special tokens.

**Los 2 tokens enmascarados (14.3% de 14)**:
- `pos 9: 'the ' -> '[MASK]'` — caso 80%: el token fue reemplazado por `[MASK]`. El modelo recibe `[MASK]` en esa posicion y debe predecir que el token original era 'the ' (label=161).
- `pos 11: 'es' -> 'F'` — caso 10% aleatorio: el token fue reemplazado por un token aleatorio ('F'). El modelo ve 'F' pero debe predecir 'es' (label=134). Este caso ilustra la robustez que BERT aprende: ve una señal incorrecta y aun asi debe reconstruir el token correcto.

**La loss MLM: 7.0601 vs log(1115) = 7.0166**

El modelo tiene pesos aleatorios (random init). Para 1115 tokens en el vocabulario, la probabilidad de adivinar correctamente por azar es 1/1115. La cross-entropy de una distribucion uniforme sobre 1115 clases es $\log(1115) \approx 7.0166$.

La loss observada (7.0601) es ligeramente mayor que el ideal uniforme — esto es normal para pesos aleatorios porque la distribucion de logits no es perfectamente uniforme. El entrenamiento en cap 43 reducira esta loss hacia cero sobre el conjunto de entrenamiento (con riesgo de overfitting si no se regulariza).

---

## 7. El 10% keep: robustez como objetivo de diseno

El 10% de tokens que se mantienen sin cambio es la pieza mas sutil del diseno MLM. Su efecto:

**Sin el 10% keep**: el modelo aprende que si ve un token normal en una posicion, esa posicion NO es un target — puede ignorarla en el calculo de prediccion. El modelo se vuelve perezoso y eficiente de forma incorrecta.

**Con el 10% keep**: el modelo nunca sabe si el token que ve en cualquier posicion ya fue reemplazado o es el token original correcto. Debe mantener atencion activa sobre TODAS las posiciones. Esto lo obliga a construir representaciones contextuales densas — el hallazgo central que hace a BERT util para downstream tasks.

La consecuencia practica: cuando usamos BERT para clasificacion, el vector `[CLS]` contiene informacion contextual de toda la secuencia porque el modelo aprendio a no ignorar ninguna posicion durante el pretraining.

---

## 8. Preguntas de verificacion

1. **Por que `ignore_index=-100` en `F.cross_entropy` y no simplemente multiplicar la loss por un vector de ceros?** Ambos logran el mismo resultado matematico. La diferencia es que `ignore_index` es mas eficiente (PyTorch lo implementa internamente sin crear tensores extra) y mas legible. Para SFT se uso la multiplicacion explicita para mostrar la mecanica; para MLM usamos `ignore_index` que es el patron estandar de Hugging Face.

2. **Si la loss MLM inicial es ~7.06 y la esperada para distribucion uniforme es 7.02, por que no coinciden exactamente?** Los pesos aleatorios de `nn.Linear` siguen una distribucion normal, no uniforme. Los logits resultantes no estan perfectamente distribuidos — algunos tokens tendran logits ligeramente mas altos que otros por azar. La cross-entropy de una distribucion no-uniforme siempre es >= la de la uniforme (principio de maxima entropia).

3. **En la secuencia de 14 tokens, `[CLS]` y `[SEP]` no fueron enmascarados. Por que es importante esta decision de diseno?** `[CLS]` es el token especial cuyo vector se usa para clasificacion en downstream tasks — si se enmascarara durante pretraining, el modelo no aprenderia a usarlo consistentemente como agregador de informacion global. `[SEP]` marca los limites de segmentos en pares de oraciones (tarea NSP de BERT) — enmascararlo introduciria ruido en esa tarea estructural.
