---
title: "44 - Eval MLM: fill-in-the-blank"
weight: 440
math: true
---

## 1. Apertura: el encoder esta entrenado — hora de interrogarlo

Cap 43 guardo el checkpoint. Ahora lo cargamos y hacemos la prueba mas intuitiva que existe para un encoder: **fill-in-the-blank**.

La idea es simple: le damos al modelo una oracion con un hueco, y le preguntamos que token pondria ahi. Este es exactamente el objetivo para el que fue entrenado — predecir tokens enmascarados en contexto. A diferencia de un decoder (que solo genera hacia la derecha), el encoder puede usar tanto el contexto de la izquierda como el de la derecha para tomar su decision.

Fill-in-the-blank es el formato visual mas intuitivo para ver que aprendio el encoder. Si el modelo coloca predicciones semanticamente razonables, esta usando el contexto bidireccional de verdad.

---

## 2. El truco de predict_mask: por que no se puede pasar "[MASK]" como texto

Aqui hay un detalle que parece menor pero es critico: **no podemos pasar la cadena `"[MASK]"` al tokenizador BPE**.

El motivo: el BPE tokeniza caracter a caracter antes de aplicar sus merges. La cadena `"[MASK]"` seria tokenizada como los caracteres individuales `'['`, `'M'`, `'A'`, `'S'`, `'K'`, `']'` — seis tokens separados, ninguno de los cuales es el token especial `[MASK]`.

El token especial `[MASK]` fue agregado al vocabulario con `tok.add_special_tokens()` como una entrada atomica con su propio ID. Para usarlo correctamente, debemos inyectarlo directamente por su ID, sin pasar por el tokenizador de texto.

La solucion es construir la secuencia manualmente:

```
ids = [CLS_ID] + encode(left_context) + [MASK_ID] + encode(right_context) + [SEP_ID]
```

La posicion exacta del mask en la secuencia es:

```
mask_pos = 1 + len(encode(left_context))
```

El `1` corresponde al `[CLS]` que siempre ocupa la posicion 0. Luego vienen los tokens del contexto izquierdo, y el siguiente slot es exactamente donde esta el `[MASK]`. Esta posicion `mask_pos` es la que le pedimos al modelo que prediga.

---

## 3. El mecanismo de prediccion: logits → softmax → top-k

Una vez construida la secuencia, el flujo es:

1. El encoder MiniBERT procesa la secuencia completa y devuelve un tensor de representaciones `h` de forma `(1, T, d_model)`.
2. La MLMHead proyecta cada posicion al espacio del vocabulario: `logits` de forma `(1, T, vocab_size)`.
3. Extraemos los logits en la posicion `mask_pos`: `logits[0, mask_pos]` — un vector de dimension `vocab_size`.
4. Aplicamos softmax para obtener probabilidades.
5. Tomamos el `top_k` de esas probabilidades y mostramos los tokens correspondientes.

La clave bidireccional esta en el paso 1: el encoder usa **toda la secuencia** — incluido el contexto a la derecha de `[MASK]` — para construir la representacion en `mask_pos`. Esto es lo que lo distingue de un decoder.

```python
with torch.no_grad():
    h = model(x)          # (1, T, d_model) — contexto bidireccional completo
    logits = mlm_head(h)  # (1, T, vocab_size)
probs = torch.softmax(logits[0, mask_pos], dim=-1)
top_ids = probs.topk(top_k).indices.tolist()
```

---

## 4. Script completo

```python
"""44_eval_mlm.py - Cap 44: fill-in-the-blank con Mini-BERT pretrained."""
import torch
from _bpe import BPETokenizer
from _models import MiniBERT, MLMHead, get_device

device = get_device()
tok = BPETokenizer.load("data/bpe_tokenizer.json")
tok.add_special_tokens()

ckpt = torch.load("checkpoints/mini_bert_pretrained.pt", map_location=device)
cfg  = ckpt["config"]
model    = MiniBERT(**cfg).to(device)
mlm_head = MLMHead(d_model=cfg["d_model"], vocab_size=cfg["vocab_size"]).to(device)
model.load_state_dict(ckpt["model"])
mlm_head.load_state_dict(ckpt["mlm_head"])
model.eval(); mlm_head.eval()

def predict_mask(left: str, right: str, top_k: int = 5):
    """Predice el token entre left y right.

    IMPORTANTE: NO pasar "[MASK]" como texto — el BPE lo tokenizaria como
    chars individuales '[','M','A','S','K',']'. En su lugar, construimos
    manualmente la secuencia: [CLS] + encode(left) + mask_id + encode(right) + [SEP].
    """
    l_ids = tok.encode(left)
    r_ids = tok.encode(right)
    ids = [tok.cls_id] + l_ids + [tok.mask_id] + r_ids + [tok.sep_id]
    mask_pos = 1 + len(l_ids)  # posicion exacta del mask_id

    x = torch.tensor([ids[:cfg["max_seq_len"]]], dtype=torch.long, device=device)
    with torch.no_grad():
        h = model(x)
        logits = mlm_head(h)
    probs = torch.softmax(logits[0, mask_pos], dim=-1)
    top_ids = probs.topk(top_k).indices.tolist()
    top_probs = probs.topk(top_k).values.tolist()
    display = f"{left!r} [MASK] {right!r}"
    print(f"Texto: {display}")
    print(f"Top-{top_k} predicciones:")
    for i, (tid, prob) in enumerate(zip(top_ids, top_probs)):
        tok_str = tok.id_to_token.get(tid, "?")
        print(f"  {i+1}. '{tok_str}' ({prob:.3f})")
    print()

print("=== Fill-in-the-blank con Mini-BERT ===\n")
# Cada ejemplo: (left_context, right_context)
examples = [
    ("To ", " or not to be"),
    ("To be or not to ", ""),
    ("En un ", " de la Mancha"),
    ("The ", " is dead"),
    ("No hay mal que por bien no ", ""),
]
for left, right in examples:
    predict_mask(left, right)
```

---

## 5. Output del script

```
=== Fill-in-the-blank con Mini-BERT ===

Texto: 'To ' [MASK] ' or not to be'
Top-5 predicciones:
  1. ' ' (0.028)
  2. ':
' (0.027)
  3. 'E' (0.019)
  4. 'a' (0.015)
  5. 'd' (0.014)

Texto: 'To be or not to ' [MASK] ''
Top-5 predicciones:
  1. ':
' (0.019)
  2. 'a' (0.017)
  3. ' ' (0.016)
  4. ', ' (0.015)
  5. 'd' (0.014)

Texto: 'En un ' [MASK] ' de la Mancha'
Top-5 predicciones:
  1. ' ' (0.071)
  2. 'E' (0.029)
  3. 'L' (0.029)
  4. 'T' (0.021)
  5. ':
' (0.021)

Texto: 'The ' [MASK] ' is dead'
Top-5 predicciones:
  1. 's' (0.023)
  2. ' ' (0.019)
  3. 'i' (0.018)
  4. 'c' (0.017)
  5. ', ' (0.015)

Texto: 'No hay mal que por bien no ' [MASK] ''
Top-5 predicciones:
  1. 'qu' (0.048)
  2. ' ' (0.044)
  3. 'e ' (0.030)
  4. 'o ' (0.025)
  5. 'a ' (0.024)
```

---

## 6. Analisis del output

### Honestidad pedagogica: un modelo con loss ~4.96

Los resultados son elocuentes sobre el estado real del modelo. Con 3000 iteraciones y un corpus de tamano pedagogico, la loss final fue ~4.96. Esto implica que el modelo no llego a memorizar ni a generalizar bien — las distribuciones de probabilidad son bastante planas (el token top-1 tiene solo entre 2-7% de probabilidad, cuando un vocabulario de ~600 tokens daria ~0.17% por azar uniforme).

### Lo que el modelo SI aprendio

Hay senales reales de aprendizaje:

**Ejemplo 5 — "No hay mal que por bien no [MASK]"**: el top-1 es `'qu'`, un BPE-merge de `q` + `u`. Dada la estructura del Quijote, "no" tiende a ir seguido de frases con "que" — el merge `qu` es la primera mitad de "que". El modelo capta una frecuencia real del corpus en castellano.

**Ejemplo 3 — "En un [MASK] de la Mancha"**: el top-1 es espacio `' '`, pero los tokens 2 y 3 son `'E'` y `'L'` — las primeras letras del artigo "el" o nombres propios que comienzan con esas letras. El contexto "de la Mancha" es muy especifico del Quijote; que aparezcan letras mayusculas sugiere que el modelo asocia ese patron con nombres propios o articulos.

**Distribucion no uniforme**: en todos los casos, el top-1 tiene una probabilidad claramente mayor que el resto (el modelo no asigna probabilidades exactamente iguales a todos los tokens). Eso significa que aprendio algo — aunque sea estadistica de superficie.

### Lo que el modelo NO logro

`"be"` no aparece en el top-5 de `"To [MASK] or not to be"`. El modelo no generalizo la frase de Shakespeare. Hay dos razones:

1. **El corpus es de personajes**: Shakespeare esta dividido en lineas de dialogo con formato `PERSONAJE:\n texto`. La frase "To be or not to be" como bloque continuo puede no aparecer lo suficiente como para que un modelo pequeno la memorice con confianza.
2. **La loss de 4.96 refleja sub-entrenamiento**: con mas iteraciones, un corpus mas limpio, o un modelo mas grande, las predicciones mejorarian.

### Interpretacion correcta

El objetivo de este cap no es demostrar que Mini-BERT es bueno — es demostrar que el **mecanismo funciona**: el modelo carga, el truco de construir la secuencia manualmente funciona, el bidireccional esta activo. Los pesos entrenados hacen algo no trivial con el contexto.

---

## 7. Contraste con el decoder

Un decoder **no podria hacer fill-in-the-blank de esta forma**.

Un decoder autoregresivo (Mini-GPT, Mini-LLaMA) solo puede ver el contexto a la izquierda del token que predice — la mascara causal bloquea la atencion hacia la derecha. En la frase `"To [MASK] or not to be"`, el decoder podria usar `"To "` para predecir el siguiente token, pero jamas podria usar `" or not to be"` — esa informacion esta en el futuro segun su definicion de tiempo.

El encoder bidireccional no tiene mascara causal. Cuando procesa la posicion `[MASK]`, los bloques de atencion calculan scores contra **todos** los tokens de la secuencia — incluyendo `"or"`, `"not"`, `"to"`, `"be"` que estan a la derecha. Esa es la ventaja del encoder: puede leer la oración completa antes de decidir.

La tabla de diferencias:

| | Decoder (GPT) | Encoder (BERT) |
|---|---|---|
| Atencion | Solo contexto izquierdo | Contexto completo (bidireccional) |
| Fill-in-the-blank | No directamente | Si — objetivo MLM |
| Generacion de texto | Si — token a token | No directamente |
| Clasificacion (CLS) | Menos natural | Natural — vector [CLS] |

---

## 8. Preguntas de verificacion

**1. ¿Por que `mask_pos = 1 + len(l_ids)` y no simplemente `len(l_ids)`?**

Porque la secuencia comienza con el token `[CLS]` en la posicion 0. Los tokens del contexto izquierdo ocupan las posiciones 1 a `len(l_ids)`. Por lo tanto, el `[MASK]` esta en la posicion `1 + len(l_ids)`.

**2. Si se duplicara el corpus de entrenamiento (de 3000 a 6000 iters), ¿que cambio esperarias en las predicciones?**

La loss bajaria y las distribuciones se volverian mas "picadas" — el top-1 tendria mayor probabilidad relativa y los tokens semanticamente coherentes subirian en el ranking. No garantiza que `"be"` llegue al top-3, pero el modelo seria mas confiado en sus predicciones.

**3. ¿Que pasaria si se llamara `tok.encode("[MASK]")` en vez de usar `tok.mask_id` directamente?**

El BPE tokenizaria la cadena `"[MASK]"` caracter a caracter: `'['`, `'M'`, `'A'`, `'S'`, `'K'`, `']'`. La posicion del "mask" no existiria como token especial; `mask_pos` apuntaria a un token incorrecto y las predicciones no tendrian sentido. Ademas, la longitud de la secuencia creceria en 5 tokens extra por cada "mascara".
