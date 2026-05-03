---
title: "48 - Eval: accuracy + attention + PCA [CLS]"
weight: 480
math: true
---

## 1. Tres metricas para entender que aprendio el encoder

Despues del fine-tuning del cap 47, el checkpoint `mini_bert_finetuned.pt` existe. Pero "guardado" no es lo mismo que "entendido". Este capitulo aplica tres metricas complementarias sobre el mismo modelo:

- **Accuracy** (cuantitativo): cuantos ejemplos del conjunto de evaluacion clasifica bien. Responde "¿funciona?".
- **Attention patterns** (cualitativo): a que posiciones de la secuencia presta atencion el token `[CLS]` en el ultimo bloque. Responde "¿como mira el encoder el texto?".
- **PCA de [CLS]** (geometrico): los vectores `[CLS]` proyectados en 2D — ¿se separan EN y ES en el espacio de representaciones? Responde "¿que geometria aprendio el encoder?".

Cada metrica ilumina un aspecto distinto. El accuracy puede ser alto incluso si la geometria es confusa; la geometria puede ser clara incluso si el accuracy tiene casos borde. Usar las tres juntas da una imagen mas completa que cualquiera de ellas por separado.

---

## 2. Accuracy: 99.8% con solo 258 parametros de cabeza

El conjunto de evaluacion `data/lang_eval.jsonl` tiene 500 ejemplos: 250 EN y 250 ES, sin ningun solapamiento con el conjunto de entrenamiento de 2000 ejemplos.

```
Accuracy EN/ES: 0.998 (499/500)
```

El encoder clasifica correctamente 499 de 500 ejemplos. Un solo error en 500 — un 0.2% de tasa de error. Para una tarea de clasificacion binaria (dos clases) esto es cuasi-perfecto.

Lo notable no es el numero en si — distinguir ingles de castellano es una tarea relativamente facil si las representaciones son ricas. Lo notable es la **asimetria de parametros**: el encoder tiene 952K parametros entrenados durante 3000 iters de MLM sobre Shakespeare+Quijote. La cabeza de clasificacion tiene 258 parametros (`d_model × 2 + 2 = 128×2 + 2`) entrenados en 500 iters de fine-tuning.

Esa proporcion — 952K parametros encodificando estructura lingüistica, 258 parametros aprendiendo a leer un vector — explica por que el paradigma BERT funciona: el pretraining masivo crea representaciones suficientemente ricas para que una cabeza minimal logre alta precision en la tarea downstream. Si las representaciones del encoder fueran malas (por ejemplo, sin pretraining MLM), 258 parametros no alcanzarian para clasificar correctamente. El encoder hace el trabajo pesado; la cabeza simplemente aprende a leer el resultado.

El unico error podria ser un ejemplo con vocabulario ambiguo, mezcla de idiomas (anglicismos en espanol, palabras latinas en ingles), o simplemente una secuencia con tokens OOV (fuera del vocabulario BPE) que el encoder no puede distinguir bien.

---

## 3. Attention patterns: lo que [CLS] mira en "To be or not to be"

El segundo bloque del script registra los pesos de atencion del ultimo `BERTBlock` usando un forward hook sobre `model.blocks[-1].attn` (el `nn.MultiheadAttention`). El hook captura la segunda componente del output — los pesos promediados sobre las `n_heads=4` cabezas — y los almacena para inspeccion.

El texto de prueba es `"To be or not to be that is the question"`. Tras tokenizacion BPE + `[CLS]`/`[SEP]`, la secuencia tiene 14 tokens:

```
Attention pattern ultimo bloque (desde [CLS], sobre todos los tokens):
   0      [CLS]: 0.007 
   1          T: 0.014 
   2         o : 0.036 =
   3        be : 0.026 =
   4        or : 0.027 =
   5       not : 0.115 ====
   6     to be : 0.026 =
   7      that : 0.014 
   8        is : 0.068 ==
   9       the : 0.011 
  10         qu: 0.032 =
  11         es: 0.026 =
  12       tion: 0.019 
  13      [SEP]: 0.584 =======================
```

Hay tres observaciones importantes:

**El `[SEP]` domina (0.584)**. El token `[SEP]` recibe mas del 58% de la atencion del `[CLS]`. Este patron — `[CLS]` atendiendo fuertemente al `[SEP]` — aparece en modelos BERT reales y se interpreta de dos maneras no excluyentes. Primera: `[SEP]` actua como un "sumidero" de atencion — un token especial donde se pueden "depositar" pesos de atencion que no van a ningun lugar informativo. Segunda: `[SEP]` es el fin de secuencia, y el modelo aprendio que el vector `[CLS]` se construye mirando primero al marcador de limite antes que al contenido. En ambos casos, la atencion al `[SEP]` no es un fallo — es un comportamiento emergente del entrenamiento bidireccional.

**`not` (0.115) e `is` (0.068) son los tokens de contenido mas atendidos**. El `[CLS]` presta mas atencion a palabras funcionales con carga semantica fuerte en ingles. `not` es un marcador de negacion sin equivalente directo en espanol (donde "no" se posiciona diferente en la oracion). `is` es la copula inglesa. El modelo, entrenado para distinguir EN de ES, ha aprendido que ciertas palabras funcionales son marcadores de idioma mas fiables que el vocabulario de contenido.

**El `[CLS]` se atiende a si mismo poco (0.007)**. El token `[CLS]` en posicion 0 asigna muy poca atencion a su propia posicion. Esto tiene sentido: `[CLS]` no tiene contenido semantico propio — su representacion se construye integrando informacion de los demas tokens, no de si mismo.

La visualizacion con barras `=` mapea cada peso al rango $[0, 40]$ caracteres: `int(weight * 40)` barras. La barra de `[SEP]` tiene 23 caracteres (`int(0.584 * 40) = 23`) y la de `not` tiene 4 (`int(0.115 * 40) = 4`). Esta escala no es logaritmica, por lo que diferencias pequenas entre tokens de contenido son visualmente planas — la barra de `[SEP]` eclipsa todo lo demas.

---

## 4. PCA de [CLS]: geometria de la separacion EN vs ES

La tercera metrica examina la geometria del espacio de representaciones. Se toman los 50 primeros ejemplos del eval set (25 EN + 25 ES), se extrae el vector `[CLS]` de cada uno (dimension 128), y se proyectan en 2D usando PCA (`torch.pca_lowrank`).

```
=== PCA de embeddings [CLS] (EN vs ES) ===
EN centroide: (7.94, 0.11)
ES centroide: (-11.92, -0.17)
Distancia entre centroides: 19.862
(>2.0 = separacion clara, <1.0 = mezclados)
```

La distancia entre centroides es **19.862** — muy por encima del umbral de "separacion clara" de 2.0. Los vectores `[CLS]` EN se agrupan alrededor de `(7.94, 0.11)` en el espacio PCA, y los vectores ES alrededor de `(-11.92, -0.17)`. La separacion es casi exclusivamente sobre el primer componente principal (PC1): EN en positivo, ES en negativo.

El segundo componente principal (PC2) tiene valores casi cero para ambos centroides: `0.11` vs `-0.17`. Esto sugiere que la estructura mas importante en el espacio `[CLS]` es unidimensional — el encoder encontro una direccion principal que separa los dos idiomas. El PC1 actua como un "eje de idioma".

Esta separacion de casi 20 unidades en el espacio PCA no es accidental. Durante el fine-tuning, la funcion de perdida `cross_entropy(cls_head(h), label)` presiona al encoder para que los vectores `[CLS]` de EN y ES sean maximalmente separables por el hiperplano de la cabeza lineal. El gradiente ajusta las representaciones hasta que esa separacion en el espacio d_model=128 queda codificada en la primera dimension PCA.

Una distancia de 19.862 indica que el modelo tiene margen amplio: no esta apenas distinguiendo los dos grupos, sino colocandolos en regiones muy distintas del espacio de representaciones. Esto se traduce en robustez: ejemplos con vocabulario ambiguo o tokens OOV pueden estar en posiciones intermedias y aun asi clasificarse correctamente porque el umbral de decision esta lejos de los puntos intermedios.

---

## 5. Script completo

```python
"""48_eval_bert.py - Cap 48: accuracy + attention patterns + PCA [CLS]."""
import json, torch
import torch.nn.functional as F
from _models import MiniBERT, ClassificationHead, get_device
from _bpe import BPETokenizer

device = get_device()

ckpt = torch.load("checkpoints/mini_bert_finetuned.pt", map_location=device, weights_only=False)
cfg  = ckpt["config"]
model    = MiniBERT(**cfg).to(device)
cls_head = ClassificationHead(d_model=cfg["d_model"], n_classes=2).to(device)
model.load_state_dict(ckpt["model"])
cls_head.load_state_dict(ckpt["cls_head"])
model.eval(); cls_head.eval()

tok = BPETokenizer.load("data/bpe_tokenizer.json")
tok.add_special_tokens()

# === Accuracy en eval set ===
with open("data/lang_eval.jsonl") as f:
    eval_data = [json.loads(l) for l in f]
correct = 0
for ex in eval_data:
    ids = torch.tensor([ex["ids"]], dtype=torch.long, device=device)
    with torch.no_grad():
        h = model(ids); logits = cls_head(h)
    pred = logits.argmax(dim=-1).item()
    if pred == ex["label"]: correct += 1
acc = correct / len(eval_data)
print(f"Accuracy EN/ES: {acc:.3f} ({correct}/{len(eval_data)})\n")

# === Attention patterns ===
attention_weights = {}
def hook_fn(module, input, output):
    if isinstance(output, tuple) and len(output) == 2 and output[1] is not None:
        attention_weights["last"] = output[1].detach().cpu()

handle = model.blocks[-1].attn.register_forward_hook(hook_fn)

example_en = "To be or not to be that is the question"
ids_en = torch.tensor([tok.encode_bert(example_en)[:cfg["max_seq_len"]]],
                       dtype=torch.long, device=device)
with torch.no_grad():
    h = model(ids_en)
handle.remove()

tokens_list = [tok.id_to_token.get(i, "?") for i in ids_en[0].tolist()]
attn = attention_weights.get("last")
if attn is not None:
    print("Attention pattern ultimo bloque (desde [CLS], sobre todos los tokens):")
    cls_attn = attn[0, 0, :].tolist()
    for i, (tok_str, weight) in enumerate(zip(tokens_list, cls_attn)):
        bar = "=" * int(weight * 40)
        print(f"  {i:2d} {tok_str:>10}: {weight:.3f} {bar}")
else:
    print("(attention weights no disponibles)")

# === PCA de [CLS] vectors ===
print("\n=== PCA de embeddings [CLS] (EN vs ES) ===")
en_vecs, es_vecs = [], []
for ex in eval_data[:50]:
    ids = torch.tensor([ex["ids"][:cfg["max_seq_len"]]], dtype=torch.long, device=device)
    with torch.no_grad():
        h = model(ids)
    cls_vec = h[0, 0].cpu()
    if ex["label"] == 0: en_vecs.append(cls_vec)
    else:                es_vecs.append(cls_vec)

all_vecs = torch.stack(en_vecs + es_vecs)
mean = all_vecs.mean(0)
centered = all_vecs - mean
U, S, V = torch.pca_lowrank(centered, q=2)
proj = centered @ V  # (N, 2)
n_en = len(en_vecs)
en_proj = proj[:n_en]; es_proj = proj[n_en:]
print(f"EN centroide: ({en_proj[:, 0].mean():.2f}, {en_proj[:, 1].mean():.2f})")
print(f"ES centroide: ({es_proj[:, 0].mean():.2f}, {es_proj[:, 1].mean():.2f})")
dist = ((en_proj.mean(0) - es_proj.mean(0)).norm()).item()
print(f"Distancia entre centroides: {dist:.3f}")
print("(>2.0 = separacion clara, <1.0 = mezclados)")
```

---

## 6. Output literal del script

```
Accuracy EN/ES: 0.998 (499/500)

Attention pattern ultimo bloque (desde [CLS], sobre todos los tokens):
   0      [CLS]: 0.007 
   1          T: 0.014 
   2         o : 0.036 =
   3        be : 0.026 =
   4        or : 0.027 =
   5       not : 0.115 ====
   6     to be : 0.021 
   7      that : 0.014 
   8        is : 0.068 ==
   9       the : 0.011 
  10         qu: 0.032 =
  11         es: 0.026 =
  12       tion: 0.019 
  13      [SEP]: 0.584 =======================

=== PCA de embeddings [CLS] (EN vs ES) ===
EN centroide: (7.94, 0.11)
ES centroide: (-11.92, -0.17)
Distancia entre centroides: 19.862
(>2.0 = separacion clara, <1.0 = mezclados)
```

---

## 7. Contraste con cap 45: antes vs despues del fine-tuning

El cap 45 mostro los logits del encoder preentrenado sin fine-tuning:

```
Texto                                    Idioma   Logit EN   Logit ES
To be or not to be                       EN         -0.072      0.043
The king is dead                         EN          0.021     -0.038
En un lugar de la Mancha                 ES         -0.015      0.082
No hay mal que por bien no venga         ES          0.061     -0.024
```

Los logits eran casi simetricos alrededor de cero — la cabeza lineal era aleatoria (no entrenada) y los vectores `[CLS]` del encoder preentrenado no estaban orientados para distinguir idiomas. Un clasificador con esos logits tendria accuracy cercano al 50% — azar.

Despues del fine-tuning (cap 48), la situacion es radicalmente distinta:
- Accuracy: 99.8% (499/500) frente al ~50% esperado con cabeza aleatoria.
- Geometria: distancia entre centroides de 19.862 frente a una geometria mezclada donde EN y ES ocupan el mismo espacio.

Lo que cambio entre cap 45 y cap 48 no fue el encoder en su totalidad — los 952K parametros del encoder MLM siguieron siendo la base. Lo que cambio fue la **orientacion** de los vectores `[CLS]` en el espacio d_model=128: el fine-tuning con LR=2e-5 presiono suavemente esos vectores para que EN y ES quedaran en regiones opuestas del espacio. Los 258 parametros de la cabeza aprendieron exactamente donde trazar el hiperplano de separacion.

Este es el nucleo del paradigma BERT: el pretraining crea un espacio de representaciones rico pero no orientado para ninguna tarea especifica. El fine-tuning orienta ese espacio para la tarea concreta con minimas modificaciones. El resultado es un modelo que combina conocimiento lingüistico general (del pretraining) con precision en la tarea especifica (del fine-tuning).

---

## 8. Preguntas de verificacion

**1. ¿Por que `[SEP]` recibe tanto mas atencion que cualquier token de contenido?**

El token `[SEP]` es un "token sumidero" (sink token) — un lugar donde el mecanismo de atencion puede depositar masa de probabilidad que no necesita ir a ningun token informativo. Durante el entrenamiento bidireccional, `[CLS]` aprende a construir su representacion mirando selectivamente al contenido relevante, pero la atencion debe sumar 1 (softmax). Los pesos "sobrantes" — masa que no aporta informacion util — se acumulan en tokens especiales como `[SEP]`. Este patron es consistente con observaciones en BERT real: ciertos tokens especiales actuan como repositorios de atencion difusa. No es un problema — el vector `[CLS]` final integra informacion de forma no trivial a traves de multiples capas, y el patron de atencion del ultimo bloque es solo la ultima capa de ese proceso.

**2. Si la distancia entre centroides PCA es 19.862, ¿por que hay un error de clasificacion en 500 ejemplos?**

La distancia de 19.862 es entre los centroides de los grupos — el promedio de los vectores EN vs ES. Un centroide lejano no garantiza que todos los puntos individuales esten bien separados: puede haber ejemplos que son outliers respecto a su grupo y quedan en la region del otro grupo. El unico error (1/500) probablemente corresponde a un ejemplo con vocabulario BPE muy ambiguo (por ejemplo, palabras que existen en ambos idiomas, o una secuencia muy corta donde `[SEP]` domina completamente la representacion). La alta distancia media es compatible con un error individual.

**3. ¿Que significaria una distancia PCA menor de 1.0 entre centroides?**

Una distancia menor de 1.0 indicaria que los vectores `[CLS]` de EN y ES estan mezclados en el espacio de representaciones — no hay una direccion clara que separe los idiomas. Esto podria ocurrir si: (a) el fine-tuning no convergio (LR demasiado bajo o muy pocas iters), (b) el dataset de entrenamiento tiene mucho ruido o mezcla de idiomas, (c) el encoder preentrenado no vio suficiente texto bilingue para desarrollar representaciones que distingan los idiomas, o (d) la arquitectura es demasiado pequena para capturar la diferencia. Una distancia entre 1.0 y 2.0 indicaria separacion parcial — el clasificador probablemente funciona pero con menor confianza en los casos borde.
