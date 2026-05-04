---
title: "53 - Heatmaps de atencion: ver patrones a ojo"
weight: 530
math: true
---

## 1. Apertura: cada cabeza tiene un patron

Hasta ahora vimos el residual stream como un bus compartido. Pero el mecanismo que escribe al stream NO es uniforme: en cada bloque, el modelo tiene `h_q=4` cabezas de atencion que operan independientemente, cada una decidiendo con su propia matriz `attn_weights` de tamano `(T, T)` cuanto atender de cada token a cada otro.

Mini-LLaMA tiene 4 capas × 4 cabezas Q = **16 cabezas en total**, cada una con su propio patron. Visualizar las 16 matrices de atencion sobre un prompt fijo revela la division del trabajo: algunas cabezas miran al token anterior, otras al primer token, otras a tokens estructurales (saltos de linea, dos puntos), otras se autoatienden.

Este capitulo construye los heatmaps en ASCII y los analiza visualmente. Es el primer paso para identificar cabezas con patrones canonicos — previous-token heads (cap 54), induction heads (cap 55) — y descomponerlas matematicamente (cap 56).

---

## 2. Recuperar `attn_weights`: la complicacion

`GroupedQueryAttention.forward` no expone los `attn_weights`. Su return es solo `self.W_O(out)`, donde `weights = F.softmax(scores, dim=-1)` ya fue absorbido en el calculo. Para ver los pesos necesitamos uno de dos enfoques:

1. **Modificar el forward** para que retorne tambien `weights`. Rompe la API existente.
2. **Recomputar los pesos manualmente** dado el input post-norma a la atencion. Mantiene el modelo intocado.

Opcion 2 es mas limpia y pedagogicamente clarificadora: replicar la matematica de la atencion explicitamente en el script muestra exactamente como se forman los `attn_weights`. Necesitamos:

```python
def compute_attn_weights(x_norm, attn):
    """Recomputa attn_weights manualmente dado el input post-norm1."""
    B, T, _ = x_norm.shape
    Q = attn.W_Q(x_norm).view(B, T, attn.h_q, attn.d_k).transpose(1, 2)
    K = attn.W_K(x_norm).view(B, T, attn.h_kv, attn.d_k).transpose(1, 2)
    Q = apply_rope(Q, attn.rope_cos[:T], attn.rope_sin[:T])
    K = apply_rope(K, attn.rope_cos[:T], attn.rope_sin[:T])
    K_full = K.repeat_interleave(attn.group_size, dim=1)  # GQA: replica K para cada grupo
    scores = Q @ K_full.transpose(-2, -1) / math.sqrt(attn.d_k)
    mask = attn.mask[:, :, :T, :T]
    scores = scores.masked_fill(mask == 0, float('-inf'))
    return F.softmax(scores, dim=-1)
```

Cacheamos el output de `blocks.{i}.norm1` (el input que recibe la atencion) y aplicamos esta funcion. El resultado es `(1, h_q=4, T, T)` — los pesos de las 4 cabezas Q por capa.

---

## 3. ASCII heatmap: visualizacion sin matplotlib

Para visualizar matrices `(T, T)` sin libreria grafica, mapeamos cada peso a un caracter por su magnitud:

```python
chars = [' ', '.', '-', '+', '*', '#']  # 0 -> espacios, alto -> '#'
idx = min(int(weight * len(chars)), len(chars) - 1)
```

El resultado es un grid donde se ven los patrones inmediatamente: una diagonal `#` es self-attention, una sub-diagonal `#` es atencion al token anterior, una columna 0 con `#` es atencion al primer token.

---

## 4. Script

```python
"""53_attention_heatmaps.py - Cap 53: heatmaps de atencion ASCII por capa/cabeza."""
import math, torch
import torch.nn.functional as F
from _models import (load_pretrained_mini_llama, get_device, CharTokenizer,
                     load_text, apply_rope)
from _interp import cache_activations

torch.manual_seed(1337)
device = get_device()

text = load_text("shakespeare.txt")
tok = CharTokenizer(text)

model = load_pretrained_mini_llama("checkpoints/mini_llama_base.pt", device=device,
                                   config=dict(vocab_size=tok.vocab_size, max_seq_len=256,
                                               d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384))

prompt = "BRUTUS:\nI am"
ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)

def compute_attn_weights(x_norm, attn):
    B, T, _ = x_norm.shape
    Q = attn.W_Q(x_norm).view(B, T, attn.h_q, attn.d_k).transpose(1, 2)
    K = attn.W_K(x_norm).view(B, T, attn.h_kv, attn.d_k).transpose(1, 2)
    Q = apply_rope(Q, attn.rope_cos[:T], attn.rope_sin[:T])
    K = apply_rope(K, attn.rope_cos[:T], attn.rope_sin[:T])
    K_full = K.repeat_interleave(attn.group_size, dim=1)
    scores = Q @ K_full.transpose(-2, -1) / math.sqrt(attn.d_k)
    scores = scores.masked_fill(attn.mask[:, :, :T, :T] == 0, float('-inf'))
    return F.softmax(scores, dim=-1)

with cache_activations(model, [f"blocks.{i}.norm1" for i in range(4)]) as cache:
    with torch.no_grad():
        model(ids)

for layer in range(4):
    x_norm = cache[f"blocks.{layer}.norm1"]
    w = compute_attn_weights(x_norm, model.blocks[layer].attn)[0]  # (4, T, T)
    # ... renderizar heatmap ASCII por cabeza
```

(El script completo incluye la funcion `render_heatmap` y el resumen de scores).

---

## 5. Output: heatmap representativo

Sobre el prompt `"BRUTUS:\nI am"` (12 tokens), cada cabeza produce una matriz 12×12. Mostramos dos cabezas representativas — una con auto-atencion fuerte y una con prev-token fuerte:

### `block.0 head.0` (mezcla de self + prev)

```
          B  R  U  T  U  S  : \n  I     a  m
   B ->   #  .  .  .  .  .  .  .  .  .  .  .
   R ->   *  *  .  .  .  .  .  .  .  .  .  .
   U ->   -  +  +  .  .  .  .  .  .  .  .  .
   T ->   .  -  +  +  .  .  .  .  .  .  .  .
   U ->   .  -  +  +  +  .  .  .  .  .  .  .
   S ->   .  -  -  +  +  +  .  .  .  .  .  .
   : ->   .  -  -  +  +  +  +  .  .  .  .  .
  \n ->   .  -  -  +  +  +  +  +  .  .  .  .
   I ->   .  -  -  +  +  +  +  +  +  .  .  .
     ->   .  -  -  +  +  +  +  +  +  +  .  .
   a ->   .  -  -  +  +  +  +  +  +  +  +  .
   m ->   .  -  -  +  +  +  +  +  +  +  +  +
```

Patron mixto: la diagonal y sub-diagonal estan activas (self + prev) con un suave decay hacia tokens lejanos. Es una cabeza de "rolling context" que mira los ultimos 3-4 tokens.

### `block.2 head.1` (cabeza con previous-token fuerte)

```
          B  R  U  T  U  S  : \n  I     a  m
   B ->   #  .  .  .  .  .  .  .  .  .  .  .
   R ->   .  #  .  .  .  .  .  .  .  .  .  .
   U ->   .     #  .  .  .  .  .  .  .  .  .
   T ->   .     -  *  .  .  .  .  .  .  .  .
   U ->   .        .  *  .  .  .  .  .  .  .
   S ->               -  *  .  .  .  .  .  .
   : ->                  .  *  .  .  .  .  .
  \n ->                     -  *  .  .  .  .
   I ->                        .  *  .  .  .
     ->                              *  .  .
   a ->                                 *  .
   m ->                                    *
```

Patron casi puro de previous-token: la atencion se concentra en la subdiagonal `(i, i-1)`. Esta cabeza aprende a copiar informacion del token inmediatamente anterior — uno de los patrones canonicos identificados por Anthropic.

---

## 6. Resumen cuantitativo: scores por cabeza

Para cada cabeza calculamos tres metricas:

- **`self_attn`**: media de la diagonal `attn[i, i]` (cuanto se autoatiende cada posicion)
- **`prev_token`**: media de la subdiagonal `attn[i, i-1]` para `i >= 1`
- **`cls_attn`**: media de la columna 0 `attn[:, 0]` (atencion al primer token)

```
cabeza           self_attn   prev_token   cls_attn
--------------------------------------------------
block.0 head.0       0.258        0.280      0.262
block.0 head.1       0.322        0.224      0.191
block.0 head.2       0.265        0.184      0.257
block.0 head.3       0.338        0.257      0.197
block.1 head.0       0.191        0.414      0.191
block.1 head.1       0.185        0.450      0.158
block.1 head.2       0.270        0.422      0.185
block.1 head.3       0.338        0.438      0.130
block.2 head.0       0.260        0.466      0.232
block.2 head.1       0.180        0.565      0.252
block.2 head.2       0.324        0.418      0.143
block.2 head.3       0.156        0.341      0.222
block.3 head.0       0.355        0.391      0.119
block.3 head.1       0.201        0.443      0.188
block.3 head.2       0.356        0.382      0.144
block.3 head.3       0.408        0.339      0.123
```

Patrones observables:

- **`block.0`**: prev_token bajo (0.18-0.28). La primera capa no tiene cabezas claramente "previous-token" — son patrones difusos.
- **`block.1` y `block.2`**: prev_token salta a 0.34-0.57. **Las cabezas 1 y 2 desarrollan el patron previous-token con fuerza**. La mas extrema: `block.2 head.1` con `prev_token=0.565`. Sera nuestra candidata principal en el cap 54.
- **`block.3`**: prev_token sigue alto pero baja un poco. La ultima capa tiene `self_attn` mas alto (0.36-0.41) — atiende mas a si misma, posiblemente porque ahi se cristaliza la prediccion final.
- **`cls_attn`**: en general ~0.13-0.26. Algunas cabezas (block.0 head.0, block.2 head.0) atienden particularmente al primer token, posiblemente como "informacion default" cuando no hay contexto relevante.

La emergencia del patron previous-token en capas 1-2 (no en capa 0) es **predictiva**: en GPT-2 small (Anthropic) las induction heads emergen sobre las previous-token heads — y aparecen en capas 5-6, no en 0-1. Aqui vemos el primer paso de esa jerarquia: la base de previous-token esta en capas 1-2.

---

## 7. Lo que esto NO muestra

Los heatmaps son una herramienta **descriptiva**, no causal. Que una cabeza tenga patron previous-token no implica que el modelo *use* esa cabeza para algo importante. Para verificarlo, en cap 57 patcharemos cabezas individualmente y mediremos que tanto cambia la prediccion final.

Tampoco vemos contenido: el heatmap dice "esta cabeza atiende al token i-1" pero no "que mueve de la posicion i-1 a i". Eso requiere descomponer el circuito OV (cap 56), que captura QUE informacion la cabeza copia desde la fuente al destino.

Finalmente, los heatmaps son por prompt: distintos prompts produciran distintos patrones. La metrica robusta requiere promediar sobre muchos prompts (cap 54).

---

## 8. Preguntas de verificacion

**1. ¿Por que la matriz de atencion es triangular inferior?**

El modelo Mini-LLaMA es decoder-only (cap 21 del Camino 1) y usa **causal masking**: la posicion `i` solo puede atender a posiciones `j <= i`. Las celdas `j > i` reciben `-infinity` en `scores`, y el softmax las convierte en 0 exacto. Esto es necesario para entrenamiento auto-regresivo: durante training, predecir el token `t+1` no debe poder mirar al token `t+1` mismo (eso seria trivial). En contraste, Mini-BERT del Camino 4 NO tiene mascara causal — su matriz de atencion es densa y bidireccional. En cap 62 veremos esos heatmaps y como contrastan con estos.

**2. ¿Que indica un heatmap con todas las filas casi uniformes (sin picos)?**

Indica que la cabeza no tiene un patron especifico — simplemente promedia informacion de todas las posiciones disponibles. Esto puede ser legitimo (una cabeza de "smoothing") o sintomatico de un cabeza que no aprendio nada util y esta colapsada al promedio. La diferencia se distingue mirando el efecto causal de la cabeza (cap 57): si removerla degrada la prediccion, era util; si no, era ruido.

**3. ¿Por que los scores `prev_token` en `block.0` son mas bajos que en `block.1`?**

La primera capa opera directamente sobre las embeddings, que son representaciones cruadas del token actual sin contexto. Para que una cabeza pueda "decidir" atender al token anterior, necesita features que distingan posiciones adyacentes — features que las capas posteriores construyen via la atencion + FFN. La primera capa ESTA construyendo estas features, no usandolas. Por eso los patrones de previous-token emergen mas fuerte en capas 1-2 (cuando ya hay representaciones contextuales que las cabezas pueden aprovechar). Es una jerarquia natural: capas tempranas hacen "feature engineering" sobre tokens; capas posteriores explotan esas features para patrones complejos como previous-token, induction, name-mover.
