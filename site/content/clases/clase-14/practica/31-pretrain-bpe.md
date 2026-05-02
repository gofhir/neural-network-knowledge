---
title: "31 - Pretrain con BPE: nuevo base model bilingue"
weight: 310
math: true
---

Tenemos el tokenizer (cap 30). Ahora construimos el modelo desde cero con ese vocab de 1112 tokens. El pretrain es identico al de Camino 1 — embedding table mas grande, mismo Transformer, mismo objetivo next-token prediction. La unica diferencia real es que el texto de entrada llega como secuencias de subwords en vez de secuencias de caracteres individuales, y eso cambia todo lo que el modelo puede aprender en esas 3000 iteraciones.

---

## 1. Diferencia vs char-level

La tabla siguiente compara los dos modelos en sus dimensiones clave.

| | Char-level (Camino 1) | BPE (Camino 2.5) |
|---|---|---|
| vocab_size | 65 | 1112 |
| embedding params | 65×128 = 8,320 | 1112×128 = 142,336 |
| total params | ~1.06M | ~1.07M |
| tokens en corpus | ~1.1M chars | 1,606,374 tokens |
| loss inicial | ~3.5 | ~7.2 |
| loss final | ~1.4 | ~2.7 |

El total de parametros es casi identico (~1.07M) porque el modelo es el mismo Transformer de 4 capas con d_model=128. Lo que cambia es la embedding table: 65×128 en char-level contra 1112×128 en BPE. La diferencia es pequena en terminos absolutos porque las capas Transformer dominan el conteo.

La loss inicial es mas alta en BPE (7.17 vs ~3.5) porque hay mas posibilidades: la incertidumbre inicial de un modelo aleatorio es $\log(\text{vocab\_size})$.

$$\log(1112) \approx 7.0 \qquad \log(65) \approx 4.2$$

Al arrancar el modelo predice uniforme sobre el vocab — con 1112 tokens eso es mucha mas incertidumbre que con 65. La loss final (~2.7 vs ~1.4) se ve peor en BPE en terminos absolutos, pero en bits-per-char los dos modelos convergen a niveles comparables. El BPE esta prediciendo tokens que representan varias letras a la vez, entonces la loss-por-token mas alta no implica peor calidad de modelado.

---

## 2. El script

`clase_14/practica/31_pretrain_bpe.py`:

```python
"""31_pretrain_bpe.py - Cap 31: pretrain Mini-LLaMA con vocab BPE.

Carga BPETokenizer (1112 tokens), entrena Mini-LLaMA sobre Shakespeare+Quijote
tokenizado, guarda mini_llama_bpe_base.pt.
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from _bpe import BPETokenizer
from _models import MiniLLaMA, get_device, generate_with_prompt

torch.manual_seed(1337)
device = get_device()

print("Cargando BPETokenizer...")
tok = BPETokenizer.load("data/bpe_tokenizer.json")
vocab_size = tok.vocab_size
print(f"vocab_size={vocab_size}")

print("Tokenizando corpus bilingue...")
en = Path("shakespeare.txt").read_text(encoding="utf-8")
es = Path("quijote.txt").read_text(encoding="utf-8")
corpus = en + "\n" + es
data = torch.tensor(tok.encode(corpus), dtype=torch.long)
print(f"Tokens totales: {len(data):,}")

# Hyperparams (igual que char-level salvo vocab_size)
BLOCK = 256
BATCH = 32
LR = 3e-4
ITERS = 3000
WD = 0.01

model = MiniLLaMA(vocab_size=vocab_size, max_seq_len=BLOCK,
                  d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384)
model.to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Params: {n_params:,}")

def get_batch():
    ix = torch.randint(0, len(data) - BLOCK, (BATCH,))
    x = torch.stack([data[i:i+BLOCK] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+BLOCK+1] for i in ix]).to(device)
    return x, y

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

print(f"\nPretrain BPE: {ITERS} iters\n")
for it in range(ITERS):
    x, y = get_batch()
    logits, loss = model(x, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if it % 300 == 0 or it == ITERS - 1:
        print(f"iter {it:4d}  loss {loss.item():.4f}", flush=True)

Path("checkpoints").mkdir(exist_ok=True)
torch.save(model.state_dict(), "checkpoints/mini_llama_bpe_base.pt")
print("\nSaved -> checkpoints/mini_llama_bpe_base.pt")

# Sample de generacion
print("\n=== Sample generacion BPE-base ===")
for prompt in ["To be or not", "En un lugar"]:
    out = generate_with_prompt(model, prompt, tok, max_new_tokens=30,
                               temperature=0.8, top_k=10, device=device,
                               stop_token=None)
    print(f"Prompt: {prompt!r}")
    print(f"Output: {out!r}")
    print()
```

---

## 3. Curva de loss

```text
Cargando BPETokenizer...
vocab_size=1112
Tokenizando corpus bilingue...
Tokens totales: 1,606,374
Params: 1,072,256

Pretrain BPE: 3000 iters

iter    0  loss 7.1770
iter  300  loss 4.0647
iter  600  loss 3.4620
iter  900  loss 3.1666
iter 1200  loss 3.1002
iter 1500  loss 2.8574
iter 1800  loss 2.9465
iter 2100  loss 2.5693
iter 2400  loss 2.5673
iter 2700  loss 2.5241
iter 2999  loss 2.6799

Saved -> checkpoints/mini_llama_bpe_base.pt

=== Sample generacion BPE-base ===
Prompt: 'To be or not'
Output: 'To be or notor\nque la historia y la fuerza de las aventuras de los c'

Prompt: 'En un lugar'
Output: 'En un lugar\ndecir los ojos decís de aquellos míos que en la ca'
```

La curva baja de 7.17 a 2.68 en 3000 iteraciones, con una ligera oscilacion alrededor de iter 1800 (2.9465, subida momentanea vs 2.8574 en iter 1500). Eso es ruido de minibatch normal — con BATCH=32 y datos estocasticos, la loss no monotona es esperable.

---

## 4. Los samples de generacion — el fenomeno cross-lingual

El output mas llamativo del capitulo no es la loss — es la mezcla de idiomas en la generacion.

Prompt `'To be or not'` continua con `'notor\nque la historia y la fuerza de las aventuras de los c'`. El modelo empieza en ingles y en cuestion de tokens cambia al espanol. Prompt `'En un lugar'` arranca en espanol pero el espanol que sigue ("decir los ojos decis de aquellos mios") tiene sabor quijotesco. Ambos son correctos en terminos de coherencia local — las palabras que siguen son palabras que el modelo vio en el corpus — pero el idioma puede cambiar en cualquier momento.

Esto es esperado y honesto. El mini-LLaMA pretrained es un modelo de lenguaje bilingue sin ningun concepto de idioma como categoria. Solo predice el siguiente token mas probable dado el contexto. Si el contexto "To be or not" tiene en el corpus vecindad con texto ingles que luego alterna con espanol (porque la concatenacion es Shakespeare seguido de Quijote), el modelo aprende esa transicion como un patron probable. No hay nada "incorrecto" en esto — el modelo hace exactamente lo que el objetivo next-token prediction le pide.

En produccion (Llama-3 multilingue, mT5, mBART) se agrega un language tag especial — `[ES]`, `[EN]`, o el equivalente segun el sistema — al principio de cada segmento de texto. El modelo aprende entonces a condicionarse en ese tag: dado `[EN] To be or not`, la distribucion de probabilidad sobre el siguiente token esta sesgada hacia tokens del ingles. Dado `[ES] En un lugar`, hacia el espanol. No lo hacemos aqui para mantener la complejidad manejable, y porque la leccion principal de Camino 2.5 no es multilingualismo sino SFT+DPO con tokens semanticamente ricos.

La generacion tambien muestra algo positivo: el modelo produce palabras reales de ambos idiomas, no gibberish. Contraste esto con el char-level (caps 8-11): el char-level generaba caracteres con coherencia ortografica pero sin semantica de palabras clara — habia que leerlo con esfuerzo para ver si estaba "diciendo algo". El BPE-base, incluso sin fine-tuning, genera secuencias donde cada token es una palabra o subword reconocible. Esto es la ventaja de la granularidad semantica que BPE otorga.

---

## 5. Preguntas de verificacion

1. **¿Por que la loss inicial del BPE (7.17) es mayor que la del char-level (~3.5)?**
   Con mas opciones en el vocab, la incertidumbre inicial es mayor. Un modelo aleatorio asigna probabilidad uniforme sobre el vocab: la cross-entropy esperada es $-\log(1/V) = \log(V)$. Para V=1112 eso es $\log(1112) \approx 7.0$. Para V=65 es $\log(65) \approx 4.2$. El vocab mas grande implica mas posibilidades por predecir, entonces la loss de partida es mayor.

2. **El modelo mezcla ES y EN libremente. ¿Como evitarias esto sin cambiar la arquitectura?**
   Agregando un language tag especial al inicio de cada documento en el corpus de pretrain. Por ejemplo, cada bloque Shakespeare arranca con `[EN]` y cada bloque Quijote con `[ES]`. Asi el modelo aprende que dado `[EN]` la distribucion de next-token esta concentrada en tokens del ingles, y viceversa. En inferencia, el prompt incluye el tag deseado y el modelo respeta el idioma. Esta tecnica la usan mT5, mBART y la familia Llama multilingue.

3. **El pretrain tomo varios minutos vs ~30s en char-level. ¿A que se debe la diferencia?**
   Hay dos factores. Primero, tokenizar el corpus completo (3.2M chars) con BPE toma tiempo porque `encode` aplica todos los merges aprendidos en orden secuencial sobre cada caracter. El char-level no necesita este paso — cada caracter ya es un ID directo via el diccionario. Segundo, el corpus BPE produce 1,606,374 tokens para el mismo texto que char-level representaria en ~1.1M caracteres (el ratio de compresion BPE es ~0.5, entonces hay mas tokens BPE que chars en este corpus bilingue). Mas tokens en el dataset implica mas variedad de batches disponibles, pero la tokenizacion inicial es el cuello de botella principal.

---

## 6. Lo que viene

En cap 32 hacemos un demo rapido de que el refactor tokenizer-agnostic funciona para ambos modelos — la misma funcion `generate_with_prompt` acepta `CharTokenizer` o `BPETokenizer` sin modificacion. En cap 33 construimos el dataset SFT con las 4 tareas BPE-naturales, aprovechando que el BPE tiene tokens de palabras completas para tareas como conteo de palabras y deteccion de idioma.

Volver al [hub de practica](..) o a la [Clase 14](../..).
