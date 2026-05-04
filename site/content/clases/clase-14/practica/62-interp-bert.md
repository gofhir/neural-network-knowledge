---
title: "62 - Interpretabilidad en Mini-BERT: bidireccional vs causal"
weight: 620
math: true
---

## 1. Apertura: el contraste con todo lo anterior

Caps 50-61 aplicaron tecnicas de interpretabilidad sobre Mini-LLaMA — un modelo decoder-only con atencion causal. Este capitulo aplica las mismas tecnicas a **Mini-BERT** (Camino 4), un encoder-only con atencion bidireccional. ¿Que cambia?

**Lo que NO cambia**:

- Forward hooks funcionan igual
- El residual stream sigue siendo la autopista
- QK/OV decomposition se aplica igual
- Activation patching funciona
- Sparse autoencoders se entrenan igual

**Lo que SI cambia**:

- **Sin causal mask**: la matriz de atencion es densa, no triangular. Cada token atiende a TODOS los demas (anteriores y posteriores).
- **No hay induction heads**: el patron `[A][B] ... [A] -> [B]` no aplica — no hay generacion auto-regresiva. En cambio aparecen patrones de **agregacion hacia [CLS]** y **pooling hacia [SEP]**.
- **El "output" es el vector [CLS]**, no logits sobre vocab. La interpretabilidad se centra en como las cabezas escriben informacion al [CLS].

Vamos a aplicar attention pattern analysis al Mini-BERT fine-tuneado para deteccion de idioma EN/ES (cap 47), y comparar con lo que vimos en Mini-LLaMA.

---

## 2. El experimento

**Modelo**: `mini_bert_finetuned.pt` del cap 47. 4 capas, 4 heads, accuracy 99.8% sobre EN/ES.

**Prompts**:

- EN: `"to be or not to be that is the question"`
- ES: `"ser o no ser esa es la cuestion"`

**Que medimos**: por cada capa,

1. **Atencion hacia `[CLS]`**: media de la columna 0 de la matriz de atencion (cuanto atienden TODOS los tokens al CLS)
2. **Atencion hacia `[SEP]`**: media de la columna del SEP (idem para SEP)
3. **Atencion diagonal (self)**: media de la diagonal — cuanto se autoatiende cada token

Estos tres scores nos dicen el "patron dominante" de cada capa.

**Tambien comparamos los vectores [CLS]** (post-norm-final) entre EN y ES via cosine similarity. Si capturan idioma, deberian ser muy distintos.

---

## 3. Script

```python
"""62_interp_bert.py - Cap 62: interpretabilidad sobre Mini-BERT (encoder bidireccional)."""
import torch
from _models import MiniBERT, ClassificationHead, get_device
from _bpe import BPETokenizer

device = get_device()
ckpt = torch.load("checkpoints/mini_bert_finetuned.pt", map_location=device, weights_only=False)
cfg = ckpt["config"]
model = MiniBERT(**cfg).to(device); model.load_state_dict(ckpt["model"]); model.eval()
cls_head = ClassificationHead(d_model=cfg["d_model"], n_classes=2).to(device)
cls_head.load_state_dict(ckpt["cls_head"]); cls_head.eval()

tok = BPETokenizer.load("data/bpe_tokenizer.json")
tok.add_special_tokens()

# Hook captura attention weights (output[1] de nn.MultiheadAttention)
attention_caches = {}
def make_attn_hook(layer):
    def hook(module, inputs, output):
        if isinstance(output, tuple) and output[1] is not None:
            attention_caches[layer] = output[1].detach().cpu()
    return hook

def get_attention(prompt_text):
    handles = [model.blocks[i].attn.register_forward_hook(make_attn_hook(i))
               for i in range(cfg['n_layers'])]
    ids = tok.encode_bert(prompt_text)
    ids_t = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        h = model(ids_t)
        logits = cls_head(h)
    for hand in handles:
        hand.remove()
    return ids, h, logits, dict(attention_caches)

en_ids, en_h, en_logits, en_attn = get_attention("to be or not to be that is the question")
es_ids, es_h, es_logits, es_attn = get_attention("ser o no ser esa es la cuestion")

# Para cada capa, computar media de atencion hacia [CLS], [SEP], diagonal
def attention_to_special_tokens(attn, ids, tok):
    a = attn[0] if attn.dim() == 3 else attn
    cls_attn = a[:, ids.index(tok.cls_id)].mean().item()
    sep_pos = max(i for i, x in enumerate(ids) if x == tok.sep_id)
    sep_attn = a[:, sep_pos].mean().item()
    self_diag = torch.tensor([a[i, i] for i in range(a.shape[0])]).mean().item()
    return cls_attn, sep_attn, self_diag
```

---

## 4. Output literal

```
Mini-BERT: 4 capas, 4 heads, max_seq_len=66

PROMPT EN: 'to be or not to be that is the question'
Prediccion: EN (logit=3.076)
Tokens (n=12): ['[CLS]', 'to be ', 'or ', 'not ', 'to be ', 'that ', 'is ',
                'the ', 'qu', 'es', 'tion', '[SEP]']

PROMPT ES: 'ser o no ser esa es la cuestion'
Prediccion: ES (logit=4.154)
Tokens (n=17): ['[CLS]', 's', 'er ', 'o ', 'no ', 's', 'er ', 'es', 'a ',
                'es ', 'la', ' ', 'c', 'u', 'es', 'tion', '[SEP]']

=== Patrones de atencion: hacia [CLS], [SEP] y diagonal (self) ===

capa     EN: [CLS]   EN: [SEP]   EN: self     ES: [CLS]   ES: [SEP]   ES: self
---------------------------------------------------------------------------
block.0        0.074       0.095      0.086         0.057       0.054      0.051
block.1        0.067       0.117      0.084         0.057       0.073      0.058
block.2        0.053       0.082      0.085         0.055       0.067      0.048
block.3        0.031       0.377      0.061         0.101       0.032      0.045

=== Vector [CLS] (pos 0) en EN vs ES ===
||CLS_EN|| = 12.861
||CLS_ES|| = 13.269
Distancia: 18.457
Cosine similarity: 0.002
```

---

## 5. Analisis 1: capas 0-2 son similares entre EN y ES

Mirando los scores en bloques 0, 1, 2:

| Capa | EN: [CLS] | ES: [CLS] | EN: [SEP] | ES: [SEP] | EN: self | ES: self |
|---|---|---|---|---|---|---|
| 0 | 0.074 | 0.057 | 0.095 | 0.054 | 0.086 | 0.051 |
| 1 | 0.067 | 0.057 | 0.117 | 0.073 | 0.084 | 0.058 |
| 2 | 0.053 | 0.055 | 0.082 | 0.067 | 0.085 | 0.048 |

Las primeras tres capas tratan EN y ES de forma similar. Los scores son del orden de 0.05-0.12 en todas las metricas. NO hay especializacion clara por idioma en estas capas.

Esto es coherente con la nocion estandar de BERT: las capas tempranas-medias hacen procesamiento generico (sintaxis, embeddings contextualizados); las capas finales se especializan por tarea.

---

## 6. Analisis 2: la capa 3 es la que distingue

```
block.3        0.031       0.377      0.061         0.101       0.032      0.045
```

**Diferencia dramatica:**

- **EN: atencion al [SEP] = 0.377** (37.7%!)
- **ES: atencion al [SEP] = 0.032** (3.2%)

Para prompts en ingles, la capa 3 hace que TODOS los tokens atiendan masivamente al `[SEP]`. Para prompts en espanol, NO. Esto es una diferencia 12x entre lenguajes.

Y al reves para [CLS]:

- **EN: atencion al [CLS] = 0.031** (3.1%)
- **ES: atencion al [CLS] = 0.101** (10.1%)

ES atiende mas al [CLS] que EN. Diferencia 3x.

Interpretacion: la capa 3 implementa el **pooling final** que extrae la informacion de idioma. Lo hace de forma asimetrica:

- En EN: agrega informacion al residual stream del [SEP] (marcador de final). El [SEP] termina con una representacion enriquecida.
- En ES: agrega informacion al [CLS] directamente.

**¿Por que asimetrico?** El cls_head (la cabeza de clasificacion) lee SOLO el vector [CLS] (posicion 0). Para clasificar, la informacion debe llegar al [CLS]. Si EN deposita informacion en [SEP], ¿como la recupera el head?

La respuesta es que el [CLS] tambien atiende — y posiciones tempranas del prompt (que estan cerca del [CLS] en la atencion bidireccional) leen informacion del [SEP] y la traen al [CLS]. La distincion EN/ES emerge de patrones COMPUESTOS de atencion entre [CLS], el body del prompt, y [SEP].

Este patron seria **imposible en un decoder causal**: en GPT, el [SEP] (final del prompt) no puede atender al [CLS] (inicio del prompt) porque atencion causal solo va hacia atras. En BERT bidireccional, el [SEP] SI puede atender al [CLS], creando estos patrones de "pooling distribuido".

---

## 7. Analisis 3: los vectores [CLS] son ortogonales entre EN y ES

```
||CLS_EN|| = 12.861
||CLS_ES|| = 13.269
Distancia: 18.457
Cosine similarity: 0.002
```

**Cosine similarity = 0.002 — practicamente cero**. Los vectores [CLS] de EN y ES son casi perfectamente ortogonales en `d_model=128` espacio. Esto explica el accuracy 99.8% del modelo: el espacio de [CLS] vectors ESTA particionado en dos regiones casi ortogonales (una para EN, otra para ES), y la cabeza lineal traza un hiperplano de separacion entre ellas.

Distancia entre los dos vectores: 18.46. Cada vector tiene norma ~13. Una distancia 18 es mas grande que las dos normas individuales — confirma que los vectores estan en regiones del espacio bien separadas.

Esto es el resultado del fine-tuning de cap 47: el LR=2e-5 sobre 500 iters fue suficiente para que el encoder + el cls_head aprendieran a producir vectores [CLS] distinguibles por idioma.

---

## 8. Comparacion con Mini-LLaMA

Lo que NO se ve en Mini-BERT:

- **No hay induction heads**: el experimento del cap 55 sobre BERT daria scores aun mas bajos (no aplica el patron).
- **No hay flujo causal hacia "el ultimo token"**: en un decoder, la prediccion ocurre en posicion T-1; en BERT, ocurre en posicion 0 ([CLS]). El "lugar de cristalizacion" es distinto.

Lo que SI se ve en Mini-BERT:

- **Pooling especializado en [SEP]**: cabezas que recogen informacion global hacia tokens estructurales.
- **Aggregation hacia [CLS]**: el modelo "deposita" la prediccion final en [CLS].
- **Asimetria entre lenguajes**: la capa final tiene comportamientos distintos segun idioma.

Comparativo:

| Aspecto | Mini-LLaMA (decoder) | Mini-BERT (encoder) |
|---|---|---|
| Atencion | Causal (triangular) | Bidireccional (densa) |
| Cabezas canonicas | Previous-token, induction | [CLS]-aggregation, [SEP]-pooling |
| Cristalizacion de prediccion | Ultimo token (posicion T-1) | Primer token ([CLS]) |
| Output | Logits sobre vocab | Vector [CLS] -> head |
| Flujo causal | Hacia T-1 | Hacia 0 ([CLS]) |
| Activation patching usable | Si | Si (mismo principio) |
| SAE entrenable | Si | Si |

Las **tecnicas** son las mismas. Lo que cambia son los **patrones** que emergen — porque la arquitectura subyacente determina que tipos de algoritmos puede aprender el modelo.

---

## 9. Lo que esto significa pedagogicamente

Camino 3 cubrio interpretabilidad principalmente sobre Mini-LLaMA. Cap 62 valida que las tecnicas TRANSFIEREN a otra arquitectura — el toolkit no es especifico a decoders.

Aplicaciones futuras: las mismas herramientas (`cache_activations`, `qk_circuit`, `patch_activation`, `SparseAutoencoder`) se aplican a:

- Vision Transformers (ViT, Camino 5)
- Modelos multimodales (CLIP)
- Variantes de BERT (RoBERTa, DeBERTa)
- Encoder-decoder (T5, BART)

Lo unico que cambia es la **interpretacion de los resultados**. Las cabezas previous-token y induction son features de decoders causales. En encoders bidireccionales aparecen otras features estructurales (pooling, aggregation, syntax). En ViT aparecen features espaciales (atencion local vs global, atencion a patches centrales vs perifericos).

---

## 10. Preguntas de verificacion

**1. ¿Por que la capa 3 de Mini-BERT atiende fuertemente al [SEP] en EN pero no en ES?**

Esto es un patron aprendido durante el fine-tuning. El modelo no fue programado para atender al [SEP] — descubrio durante el entrenamiento que esa estrategia funciona para clasificar EN. Posibles razones:

- En el corpus de Shakespeare (de donde vienen los prompts EN), ciertos patrones ortograficos (capitalizacion, palabras como "the", "thou") aparecen frecuentemente. El [SEP] al final agrega estos como "marcador de fin" para el resto de la secuencia. La capa 3 lo explota.
- En espanol, el mismo trabajo se hace via [CLS]. Es una eleccion arbitraria de arquitectura aprendida — el modelo encontro DOS mecanismos distintos que funcionan, uno por idioma. No hay razon a priori para que sea asi; otro fine-tune con semilla distinta probablemente produciria un patron diferente.

**2. ¿La diferencia EN vs ES en capa 3 es causal o solo correlacional?**

Por si misma, la observacion del patron es correlacional. Para verificar causalidad necesitariamos activation patching: si patcheamos la activacion del [SEP] en capa 3 con la version "neutral" (de un prompt random), ¿cae el accuracy? Si si, el patron es causal. Cap 57 mostro como hacer esto en Mini-LLaMA; el mismo procedimiento funciona en BERT. Esta verificacion no la hicimos en este capitulo, pero seria un siguiente experimento natural.

**3. ¿Por que el cosine similarity entre [CLS] de EN y ES es ~0?**

Porque el fine-tuning con cross-entropy loss sobre 2 clases empuja los vectores [CLS] hacia regiones del espacio que la cabeza lineal `(d_model -> 2)` puede separar facilmente. La cabeza lineal traza un hiperplano; para que ese hiperplano clasifique con accuracy 99.8%, los vectores [CLS] de EN deben caer en un lado y los de ES en el otro. La forma mas simple de lograr esto es que los vectores sean "casi ortogonales" — un eje del espacio representa "ingles", otro "espanol", y los vectores se alinean con uno u otro. Esto es una propiedad de aprendizaje supervisado: el feature que el modelo necesita (idioma) se vuelve dominante en el espacio de [CLS].
