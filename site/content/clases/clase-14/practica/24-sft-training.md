---
title: "24 - SFT training: loss masking"
weight: 240
math: true
---

Tenemos el dataset ([cap 23](../23-dataset-sft)) y el base model ([cap 22](../22-base-model-no-instructions)). Ahora hacemos el fine-tune. Lo distintivo vs el pretrain del Camino 1 son TRES cosas: cargar pesos, dataset distinto, y la pieza critica — **loss masking**.

---

## 1. Que cambia vs pretrain

Mismo modelo (Mini-LLaMA del cap 21), mismo loss base (cross-entropy), mismo optimizador (AdamW). Tres diferencias concretas:

- **Cargar pesos**: usamos `load_pretrained_mini_llama` para iniciar desde el base. No partimos de cero — partimos del modelo que ya hablaba Shakespeare.
- **Dataset distinto**: pares (instruccion, respuesta) en vez de Shakespeare crudo. Cada ejemplo del dataset SFT (ver cap 23) es un dict `{prompt, response, task}`.
- **Loss masking**: solo penalizamos los tokens de la respuesta, no los del prompt. Es la pieza tecnicamente sutil — la que voy a desarrollar a continuacion.

---

## 2. Loss masking: el corazon de SFT

Esta es la idea central. Sin entenderla, SFT no funciona — el modelo aprende lo que no debe aprender.

Imagina el ejemplo `INSTR: reverse 'cat'\nRESP: tac\n`. Tokenizado char-level es:

```
['I','N','S','T','R',':',' ','r','e','v','e','r','s','e',' ',"'","c","a","t","'","\n",'R','E','S','P',':',' ','t','a','c','\n']
```

Son 31 caracteres. Los primeros 27 son el **prompt** (`INSTR: reverse 'cat'\nRESP: `). Los ultimos 4 son la **respuesta** (`tac\n`).

Durante training, el modelo predice cada token a partir del anterior — es el mismo loop autoregresivo del pretrain. **Si penalizaramos al modelo por predecir mal cualquier posicion**, le pediriamos que aprenda a generar el prompt entero. Pero el prompt cambia segun la pregunta del usuario: a veces es `INSTR: reverse 'dog'`, a veces `Q: what is the capital of Spain?`. No queremos que aprenda a generar prompts — queremos que aprenda a generar la **respuesta** dado el prompt.

La solucion es enmascarar el loss. Construimos un vector `mask` de la misma forma que el target, con `0` en las posiciones del prompt y `1` en las posiciones de la respuesta. Multiplicamos el loss por token elemento a elemento por ese mask antes de promediar:

```
posicion:  0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20  21 22 23 24 25 26  27 28 29 30
token:     I N S T R : _ r e v e  r  s  e  _  '  c  a  t  '  \n  R  E  S  P  :  _   t  a  c  \n
mask:      0 0 0 0 0 0 0 0 0 0 0  0  0  0  0  0  0  0  0  0  0   0  0  0  0  0  0   1  1  1  1
                                          (prompt: mask=0)                              (response: mask=1)
```

El mask vale 1 solo en las posiciones de respuesta (28..31). El optimizador solo recibe gradiente por errores en la respuesta — los aciertos o errores en el prompt no contribuyen al loss y por lo tanto no mueven los pesos.

Si **NO** masqueamos, el modelo aprende a copiar el prompt — lo cual ya hace casi gratis (la mayor parte del prompt es predecible: visto `INSTR: re`, predecir `v` no es dificil cuando el dataset esta lleno de `INSTR: reverse`). El loss baja porque el modelo memoriza prompts, pero NO mejora en generar respuestas. Es el bug clasico que la primera iteracion de SFT en cualquier framework siempre tiene — uno mira la curva de loss bajando y celebra, pero al evaluar el modelo el output sigue siendo basura.

Otra forma de verlo: sin mask, el loss promedio se domina por los tokens de prompt (son ~27 de 31 en este ejemplo, ~87% del peso total). Los pocos tokens de respuesta apenas mueven el gradiente. Con mask, el 100% del gradiente proviene de los tokens que realmente importan.

En el script (siguiente seccion), las dos lineas clave son:

```python
loss_per_tok = F.cross_entropy(..., reduction="none")  # (B, T)
masked_loss = (loss_per_tok * mask).sum() / mask.sum().clamp(min=1)
```

`reduction="none"` es lo que nos permite tener loss por token. `mask.sum().clamp(min=1)` evita division por cero si un batch tuviera mask vacio.

---

## 3. Hyperparams: por que cambiamos

Comparativa frente al pretrain del Camino 1:

| Param | Pretrain (Camino 1) | SFT (Camino 2) | Por que |
|---|---|---|---|
| `learning_rate` | 3e-4 | 1e-4 | 10x menor — convencion: SFT no debe destruir el conocimiento previo |
| `max_iters` | 3000 | 1500 | Dataset chico, no hace falta tanto |
| `batch_size` | 32 | 32 | igual |
| `weight_decay` | 0.01 | 0.01 | igual |
| `warmup` | si | no | dataset chico, evitamos ruido inicial |

El cambio clave es `lr`. Con un lr alto, los gradientes del SFT pueden mover demasiado los pesos preentrenados — el modelo "olvida" lo que sabia (catastrophic forgetting) antes de aprender lo nuevo. Con `lr=1e-4` los pasos son chicos y el conocimiento previo se preserva mientras el modelo se adapta al formato nuevo.

---

## 4. El script

`16_train_sft.py` completo:

```python
"""16_train_sft.py - Cap 24: SFT loop con loss masking.

Carga Mini-LLaMA base + fine-tune con loss enmascarada (solo response tokens cuentan).
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from _models import load_pretrained_mini_llama, get_device
from _eval import build_char_maps, load_jsonl

torch.manual_seed(1337)
device = get_device()

# Hiperparametros (ver tabla design doc)
BLOCK = 64
BATCH = 32
LR = 1e-4
ITERS = 1500
WD = 0.01

text = Path("shakespeare.txt").read_text()
c2i, i2c = build_char_maps(text)
vocab_size = len(c2i)

model = load_pretrained_mini_llama("checkpoints/mini_llama_base.pt", device=device)
model.train()

examples = load_jsonl("data/sft_dataset.jsonl")
print(f"Loaded {len(examples)} SFT examples")


def encode_example(ex):
    """Devuelve (full_ids, mask) donde mask alinea con tgt = full[1:]."""
    prompt_ids = [c2i[c] for c in ex["prompt"]]
    response_ids = [c2i[c] for c in ex["response"]]
    full = prompt_ids + response_ids
    if len(full) > BLOCK + 1:
        full = full[: BLOCK + 1]
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
            full.append(0)
            mask.append(0)
        full = full[: BLOCK + 1]
        mask = mask[:BLOCK]
        inp = full[:-1]
        tgt = full[1:]
        batch_inp.append(inp)
        batch_tgt.append(tgt)
        batch_mask.append(mask)
    return (
        torch.tensor(batch_inp, dtype=torch.long, device=device),
        torch.tensor(batch_tgt, dtype=torch.long, device=device),
        torch.tensor(batch_mask, dtype=torch.float, device=device),
    )


opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

for it in range(ITERS):
    inp, tgt, mask = get_batch()
    logits, _ = model(inp)  # (B, T, V)
    loss_per_tok = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        tgt.reshape(-1),
        reduction="none",
    )
    loss_per_tok = loss_per_tok.reshape(inp.shape)
    masked_loss = (loss_per_tok * mask).sum() / mask.sum().clamp(min=1)

    opt.zero_grad()
    masked_loss.backward()
    opt.step()

    if it % 100 == 0 or it == ITERS - 1:
        print(f"iter {it:4d}  loss {masked_loss.item():.4f}")

torch.save(model.state_dict(), "checkpoints/mini_llama_sft.pt")
print("\nSaved -> checkpoints/mini_llama_sft.pt")
```

Lectura rapida: `encode_example` arma el `mask` (ceros para prompt, unos para response, alineado con `tgt = full[1:]`), `get_batch` arma batches con padding, y el loop principal hace el masked loss en las dos lineas que mencione antes. Todo lo demas es estandar.

---

## 5. La curva de loss — y un fenomeno interesante

Salida literal al correr `16_train_sft.py`:

```
Loaded 4000 SFT examples
iter    0  loss 6.5598
iter  100  loss 1.8977
iter  200  loss 1.9237
iter  300  loss 1.9724
iter  400  loss 1.9899
iter  500  loss 1.3258
iter  600  loss 1.3672
iter  700  loss 1.3161
iter  800  loss 1.1584
iter  900  loss 0.9780
iter 1000  loss 0.7718
iter 1100  loss 0.7688
iter 1200  loss 0.4409
iter 1300  loss 0.5072
iter 1400  loss 0.4579
iter 1499  loss 0.5053

Saved -> checkpoints/mini_llama_sft.pt
```

Loss arranca en **6.56**. Eso es ALTO — mucho mas que los ~2-3 que veiamos en pretrain. Es esperable: el base model nunca vio el prefijo `INSTR:` ni `RESP:` durante pretrain. Para el son tokens "random" al principio. Cuando le pedimos predecir el primer caracter de la respuesta, no tiene idea — la entropia condicional es practicamente uniforme sobre el vocab (el `\log(65) \approx 4.17`, mas el factor de incertidumbre adicional por el contexto desconocido empuja el numero todavia mas arriba).

Algo interesante: la loss baja a ~1.9 muy rapido (iter 100), se mantiene **plana en ~1.9 entre iters 100 y 400**, y despues vuelve a bajar. ¿Que pasa? Mi interpretacion: el modelo esta **desaprendiendo el prior puro de Shakespeare** antes de poder lockearse al formato SFT. Es como un equilibrio inestable — primero olvida el viejo proyecto (donde "doth" y "thou" eran probables), despues aprende el nuevo (donde el primer caracter de la respuesta tiene una distribucion completamente distinta segun la tarea). Durante esa meseta el modelo esta en transicion: ya no es el base, todavia no es el SFT.

Para iter 1500 estamos en **0.51** — aprox 13x menos que el inicio. Convergencia limpia. El modelo ahora asigna probabilidad alta a los tokens de respuesta correctos dado el prompt. Eso es lo que SFT optimiza — y eso es lo que medimos cuantitativamente en el cap 25.

---

## 6. Smoke test del modelo entrenado

Cargamos el checkpoint recien guardado y le damos los mismos cuatro prompts del cap 22:

```
=== Smoke test SFT model ===
prompt: "INSTR: reverse 'cat'\nRESP: "
output: 'tcc\n'

prompt: "INSTR: upper 'hello'\nRESP: "
output: 'HLOPE\n'

prompt: "INSTR: repeat 'a' three\nRESP: "
output: 'aaa\n'

prompt: 'Q: who wrote Hamlet?\nA: '
output: 'Shakespeare\n'
```

Cuatro pruebas. **Q: who wrote Hamlet? -> Shakespeare** es perfecto, **repeat 'a' three -> aaa** tambien. **upper 'hello' -> HLOPE** es casi: cinco letras mayusculas que casi son las correctas (el modelo aprendio el FORMATO — devolver una secuencia de mayusculas terminada en `\n` —, le falta refinar las letras especificas; esperable para un char-level con d_model=128). **reverse 'cat' -> tcc** es similar: empieza con la `t` correcta pero pierde la secuencia.

Lo importante NO es la accuracy perfecta. Es que el modelo **dejo de generar Shakespeare drift** y empezo a respetar el formato `INSTR -> respuesta concisa`. Compara con el cap 22, donde ante `INSTR: reverse 'cat'\nRESP: ` el base devolvia `alast the king, there is be doth in him.`. Ahora devuelve `tcc\n`. Mal contenido, formato correcto. Esa es la diferencia que SFT enseña.

Para mejorar accuracy: mas iteraciones, mas datos, modelo mas grande (d_model=256 o 512 en vez de 128), o usar un tokenizador BPE en vez de char-level. Pero el contraste antes/despues — Shakespeare-drift vs respuestas concisas con formato — es lo que importa pedagogicamente. La accuracy es ortogonal: la consigues escalando.

---

## 7. Preguntas de verificacion

1. ¿Por que la loss arranca en 6.56 y no en ~3-4 como el pretrain?
2. ¿Que pasaria si NO masquearamos el prompt en el loss?
3. ¿Por que `lr=1e-4` y no `3e-4` como en el pretrain del Camino 1?

Pista para la 1: el base model nunca vio los tokens `INSTR:` ni `RESP:` durante pretrain — son "sorpresa" al principio del SFT, y la entropia condicional del modelo sobre ellos es practicamente uniforme sobre el vocab.

Pista para la 2: el modelo aprenderia a memorizar prompts en vez de respuestas. La curva de loss bajaria igual (porque la mayor parte del peso esta en los tokens de prompt, que son predecibles) pero el modelo no mejoraria en generar respuestas correctas.

Pista para la 3: con SFT no queremos destruir el conocimiento del base — pasos pequeños evitan el reset catastrofico. Un `lr` alto es apropiado cuando partis de pesos aleatorios, pero ya tenemos pesos buenos y queremos ajustarlos, no reescribirlos.

---

## 8. Lo que viene

En el [cap 25](../25-eval-sft) evaluamos cuantitativamente: corremos el modelo SFT contra el eval set (1000 ejemplos no vistos durante training) y comparamos exact-match con el base. Veremos numeros concretos del salto de accuracy, desglosados por tarea (reverse, upper, repeat, qa).

Volver al [hub de practica](..) o a la [Clase 14](../..).
