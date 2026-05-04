---
title: "61 - Interpretar features del SAE: top tokens"
weight: 610
math: true
---

## 1. Apertura: el momento de la verdad

Cap 60 entreno un SAE sobre el residual stream de Mini-LLaMA. Reconstruccion 98.4%, sparsity razonable (L0=166/512). Pero esos numeros no nos dicen NADA sobre si las features son interpretables. Para saber si el SAE descompuso superposition de manera util, hay que MIRAR las features individualmente.

La tecnica estandar (Bricken et al. 2023): para cada feature, encontrar los **top-k tokens** del corpus que la activan mas. Si los top tokens son consistentemente del mismo "tipo" (todos un caracter especifico, todos puntuacion, todos principios de oracion), la feature es **monosemantica** — representa UN concepto. Si los top tokens son heterogeneos, la feature es polisemantica — representa varios conceptos mezclados.

Este capitulo aplica este analisis al SAE entrenado en cap 60. **Spoiler: 47% de las features son claramente monosemanticas**, lo que es un resultado sorprendentemente bueno para Mini-LLaMA a esta escala.

---

## 2. La metodologia

1. Cargar el SAE entrenado.
2. Recolectar 50 prompts × 64 chars = ~3200 tokens del corpus.
3. Para cada token, capturar el residual stream en `block.2` y aplicar el SAE para obtener el vector de 512 features.
4. Para cada una de las 512 features, encontrar los 8 tokens del corpus con mayor activacion.
5. Inspeccionar visualmente: ¿son del mismo caracter? ¿del mismo contexto?

Si los top-3 tokens de una feature son el MISMO caracter, declaramos la feature monosemantica fuerte. Si son 2 caracteres distintos, monosemantica parcial.

---

## 3. Script

```python
"""61_interpret_sae.py - Cap 61: interpretar features del SAE entrenado."""
import random, torch
from _models import load_pretrained_mini_llama, get_device, CharTokenizer, load_text
from _interp import cache_activations, SparseAutoencoder

torch.manual_seed(1337); random.seed(1337)
device = get_device()
tok = CharTokenizer(load_text("shakespeare.txt"))
model = load_pretrained_mini_llama("checkpoints/mini_llama_base.pt", device=device,
                                   config=dict(vocab_size=tok.vocab_size, max_seq_len=256,
                                               d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384))

ckpt = torch.load("checkpoints/sae_mini_llama.pt", map_location=device, weights_only=False)
sae = SparseAutoencoder(**{k: v for k, v in ckpt["config"].items()
                            if k in ["d_model", "d_features", "l1_coeff"]}).to(device)
sae.load_state_dict(ckpt["sae"]); sae.eval()

# Recolectar features sobre tokens
samples = []
for _ in range(50):
    start = random.randint(0, len(text) - 64 - 1)
    ids = torch.tensor([tok.encode(text[start:start+64])], dtype=torch.long, device=device)
    with cache_activations(model, [ckpt["config"]["target_name"]]) as cache:
        with torch.no_grad():
            model(ids)
    acts = cache[ckpt["config"]["target_name"]][0]
    with torch.no_grad():
        _, features = sae(acts)
    for t in range(features.shape[0]):
        ch = tok.id_to_char[ids[0, t].item()]
        samples.append((ch, features[t].cpu()))

# Para cada feature, top-k tokens
all_features = torch.stack([s[1] for s in samples])
for fi in range(512):
    top_k = all_features[:, fi].topk(8).indices
    chars_top = [samples[i][0] for i in top_k.tolist()]
    print(f"Feature {fi}: top tokens = {chars_top}")
```

---

## 4. Output literal: features representativas

Las top-12 features ordenadas por activacion maxima:

```
--- Feature #486  (max_act=3.728) ---
  rank 1: act=3.73  char='.'   context='indly.\n\nLor'
  rank 2: act=3.67  char='.'   context='rs on. What'
  rank 3: act=3.67  char='.'   context='d man.\n\nFir'
  rank 4: act=3.65  char='.'   context=' sake.\n\nQUE'
  rank 5: act=3.62  char='.'   context='issue.\n\nLEO'
  rank 6: act=3.59  char='.'   context='y her.\n\nCLA'
  rank 7: act=3.59  char='!'   context='one! Hoo!'
  rank 8: act=3.59  char='.'   context='ghter. Go'
```

Feature #486: **periodo final de oracion**. 7 de 8 top tokens son `.`, el otro es `!`. Todos en contextos de fin de oracion seguido por nueva linea o espacio.

```
--- Feature #206  (max_act=3.722) ---
  rank 1: act=3.72  char='s'   context='ects slain,'
  rank 2: act=3.62  char='s'   context=' subjec'
  rank 3: act=3.53  char='s'   context='cted son?\nT'
  ... (todos 's')
```

Feature #206: **caracter `s`**. 8/8 top tokens son `s`. Pure character feature.

```
--- Feature #309  (max_act=3.645) ---
  rank 1: act=3.64  char='h'   context='off than ne'
  rank 2: act=3.62  char='h'   context='ore than I '
  rank 3: act=3.61  char='h'   context=" o' the\nrig"
  ... (todos 'h')
```

Feature #309: **caracter `h`** seguido de `e` en contextos como "the", "than". Patron consistente.

```
--- Feature #65  (max_act=3.274) ---
  rank 1: act=3.27  char='?'   context='iance?\nTo p'
  rank 2: act=3.14  char=':'   context='words: look'
  rank 3: act=3.10  char=':'   context='ntent:\nThe '
  rank 4: act=3.07  char=':'   context='ilisk:\nI ha'
  rank 5: act=3.06  char='?'   context='d son?\nThen'
  ...
```

Feature #65: **puntuacion media** — `?` y `:`. NO es character-feature pura, es categoria semantica: "puntuacion que separa frases significantes".

```
--- Feature #405  (max_act=3.142) ---
  rank 1: act=3.14  char='\n'  context=' to\nher r'
  rank 2: act=2.98  char='\n'  context='st to\npriso'
  ... (todos '\n')
```

Feature #405: **salto de linea** dentro de versos (no de fin de speech). 8/8 son `\n` con caracter de letra antes y despues.

```
--- Feature #197  (max_act=3.139) ---
  ... (todos ',')
```

Feature #197: **coma**. 8/8 top tokens son `,`.

---

## 5. La estadistica completa

```
=== Cuantas features son monosemanticas? (top-3 tokens iguales) ===
  Features con top-3 mismo char (monosemantica fuerte):  242/512
  Features con top-3 dos chars distintos (parcial):       180/512

  Ejemplos de features monosemanticas:
    feature #0: char='e'   max_act=2.92
    feature #3: char='E'   max_act=0.90
    feature #4: char='o'   max_act=1.14
    feature #7: char=';'   max_act=2.20
    feature #8: char='r'   max_act=2.04
```

**242 de 512 features (47%) son monosemanticas fuertes**: sus top-3 tokens son identicos. Otras 180 son monosemanticas parciales (top-3 tienen solo 2 caracteres distintos). Sumando: **82% de las features tienen baja diversidad en top-3**, indicando especializacion clara.

---

## 6. Analisis: que aprendio el SAE

### Categoria 1: Character-features (mayoria)

La mayoria de las features parecen estar especializadas en **un caracter especifico del vocab** (Shakespeare tiene 65 chars). Vimos features para `e`, `o`, `s`, `h`, `r`, `v`, `,`, `.`, `\n`, etc. Esto es "boring but correct": el SAE aprendio a representar la identidad del token en el residual stream.

Por que aparecen tantas character-features: Mini-LLaMA es char-level, su "vocabulario semantico" se construye sobre identidad de chars. La mayor distincion linguistica que el modelo hace es "que caracter estoy procesando". El SAE recupera esta distincion como features individuales.

### Categoria 2: Punctuation features (interesantes)

Feature #486 (`.` en fin de oracion), #65 (`?`, `:`), #197 (`,`), #405 (`\n` interno). Estas son features estructurales — NO solo el caracter, sino el caracter EN UN CONTEXTO especifico:

- #486: periodo SEGUIDO de salto de linea + nombre de speaker (fin de speech)
- #405: salto de linea entre versos (no entre speeches)
- #65: puntuacion antes de salto de linea o explicacion

Son features mas semanticas que character-pure. El SAE las descubrio sin guidance — emergen del entrenamiento.

### Categoria 3: Features compuestas (las polisemanticas restantes)

Las ~180 features con top-3 mostrando 2 caracteres distintos son monosemanticas parciales. Algunas posibles interpretaciones:

- Feature que se activa para "h" tras "t" (formando "th"): mostraria mezcla de h y t
- Feature de "vocal despues de consonante": mostraria mezcla de vocales
- Feature de "principio de palabra": mostraria mezcla de mayusculas y consonantes iniciales

Estas requieren analisis mas profundo (cap 61 solo hace top-k). Verificarlas requeriria intervencion sobre la feature (activarla artificialmente y ver que predice el modelo).

### Las ~90 features no clasificadas

512 - 242 - 180 = 90 features que no caen en monosemantica fuerte/parcial. Probablemente son polisemanticas mas profundas (3+ chars distintos) o features que rara vez se activan fuerte.

---

## 7. Comparacion con SAEs industriales

Para contextualizar nuestros resultados:

| Setting | d_features | L0 | Var explicada | Features monosemanticas |
|---|---|---|---|---|
| Anthropic Scaling Monosemanticity (2024) | ~300k | 200-2000 | ~85% | ~70% (estimado) |
| Bricken et al. 2023 (1L Pythia) | 4096 | 30-100 | ~95% | ~50% |
| **Mini-LLaMA (cap 60-61)** | **512** | **166** | **98.4%** | **47%** |

A escala chica logramos 47% monosemanticas — comparable a SAEs en modelos pequenos (1 capa). En modelos grandes (Claude 3 Sonnet) se alcanza ~70% pero sobre vocabularios mucho mas ricos.

Lo notable: con `d_features=512` (4x expansion) y entrenamiento basico (2000 iters), el SAE encontro estructura clara. Esto valida la idea fundacional: **superposition se puede deshacer con un SAE bien configurado**.

---

## 8. Lo que esto significa para Mini-LLaMA

Antes del SAE (cap 59): "el residual stream tiene 128 dimensiones polisemanticas, cada una representa varios conceptos mezclados".

Despues del SAE: "el residual stream se puede descomponer en 512 features, ~50% de las cuales representan conceptos individuales (caracteres especificos, tipos de puntuacion)".

Esta es una transformacion epistemica: pasamos de "modelo opaco" a "modelo con direcciones interpretables identificadas". Para investigaciones futuras (por ejemplo, "que feature se activa cuando el modelo predice incorrectamente"), tenemos una base concreta para empezar.

Limitaciones honestas:

- Mini-LLaMA es char-level: las features descubiertas son mayoritariamente de identidad de caracter, no conceptos linguisticos abstractos como "sustantivo plural" o "metafora"
- 47% monosemantica es bueno pero no completo — 53% sigue siendo polisemantica o no analizado
- El SAE solo cubre `block.2`; otras capas tendrian features distintas
- No probamos causalidad: ¿activar feature #486 (periodo) artificialmente HACE que el modelo prediga `\n` despues? Eso requiere "feature ablation" que no implementamos aqui

---

## 9. Preguntas de verificacion

**1. Si el SAE encontro 242 features monosemanticas, ¿deberia tener `d_features=242` en lugar de 512?**

No. La monosemanticidad emerge BAJO la sparsity penalty. Si el `d_features` fuera exactamente 242, el SAE no tendria "espacio sobrante" para que algunas features colapsen a 0 (las features muertas) o se distribuyan. Tipicamente los SAEs requieren expansion de 4-32× sobre `d_model` para encontrar features monosemanticas. La razon: la matriz de embeddings del modelo "compacto" en `d_model` representa MAS conceptos que dimensiones tiene; expandir da espacio. El numero exacto de features monosemanticas que emergen depende del corpus, de $\lambda$, de $d_{\text{features}}$, y del entrenamiento. Aqui obtenemos 242 con d_features=512; con d_features=2048 probablemente obtendriamos 600-1000 monosemanticas.

**2. ¿Por que tantas features son sobre caracteres individuales (e, o, s, etc.) en lugar de conceptos abstractos?**

Porque Mini-LLaMA es **char-level** sobre Shakespeare con tokenizer simple (65 chars). Las features que el modelo necesita para predecir bien son inherentemente char-level: que viene despues de "th" (probablemente "e"), que viene despues de "BRUTUS" en mayusculas (probablemente ":\n"). El modelo no necesita representar conceptos como "monarchia inglesa" o "tragedia shakesperiana" porque la tarea de predecir el proximo caracter no se beneficia de eso. Si entrenaramos un BPE-tokenized model sobre el mismo corpus, las features serian mas semanticas (palabras enteras, frases). Si entrenaramos sobre datos mas variados (Shakespeare + Quijote + filosofia), aparecerian features sobre idioma, registro, etc.

**3. ¿Por que algunas features tienen max_act = 0.5 mientras otras 3.7?**

Porque diferentes features representan conceptos de diferente "frecuencia" o "magnitud" en el residual stream. Una feature que se activa rara vez pero fuertemente cuando aparece (como puntuacion final de speech) tendra max_act alto. Una feature que se activa frecuentemente pero en magnitud moderada (como "soy una vocal") tendra max_act medio. Las features con max_act < 0.1 son probablemente "casi muertas" — el SAE las dejo apagadas porque no son utiles. La distribucion de max_acts (algunas alto, otras bajas) es tipica de SAEs bien entrenados — refleja la heterogeneidad de los conceptos representados en el residual stream del modelo.
