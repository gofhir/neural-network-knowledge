---
title: "54 - Previous-token heads: el patron mas simple"
weight: 540
math: true
---

## 1. Apertura: la cabeza mas simple del Transformer

En cap 53 vimos heatmaps de las 16 cabezas de Mini-LLaMA y observamos que algunas tenian patrones reconocibles a ojo: una sub-diagonal fuerte significa que cada posicion atiende mayoritariamente al token anterior. Esa familia de cabezas — las **previous-token heads** — son los patrones mas simples y frecuentes en Transformers entrenados.

¿Por que el modelo aprende a copiar del anterior? Porque el contexto local es predictivo. Despues de "T" probablemente viene "h" (formando "Th"), despues de "Th" probablemente viene "e" (formando "The"). Una cabeza que sistematicamente "lee" el caracter anterior y lo usa para condicionar la prediccion captura este principio basico de modelado de secuencia.

Anthropic (Olsson et al. 2022) identifico previous-token heads como un patron emergente robusto en GPT-2, sirviendo de **base** para construcciones mas complejas como las induction heads (cap 55). Aqui vamos a:

1. Cuantificar el patron via `previous_token_score` sobre 50 prompts diferentes
2. Identificar la(s) cabeza(s) "mas previous-token" del modelo
3. Visualizar el patron de la cabeza top sobre un prompt concreto

---

## 2. La metrica: `previous_token_score`

Definida en `_interp.py`:

```python
def previous_token_score(attn):
    """Score [0, 1]: cuanto atiende cada posicion i a la i-1."""
    T = attn.shape[0]
    diag = torch.tensor([attn[i, i - 1].item() for i in range(1, T)])
    return diag.mean().item()
```

Es la media de la sub-diagonal `attn[i, i-1]`. Una cabeza que perfectamente atiende al anterior tiene `attn[i, i-1] = 1.0` para todo `i >= 1`, dando score = 1.0. Una cabeza que atiende uniformemente da score = 1/T (chico para T grande). El test del cap 5 verifico ambos extremos.

Para una metrica robusta evaluamos sobre **50 prompts aleatorios** de 24 caracteres extraidos de Shakespeare. Esto captura el patron promedio de la cabeza, no un comportamiento especifico de un prompt particular.

---

## 3. Script

```python
"""54_previous_token_heads.py - Cap 54: identificar cabezas previous-token."""
import math, random, torch
import torch.nn.functional as F
from _models import (load_pretrained_mini_llama, get_device, CharTokenizer,
                     load_text, apply_rope)
from _interp import cache_activations, previous_token_score

torch.manual_seed(1337); random.seed(1337)
device = get_device()

text = load_text("shakespeare.txt")
tok = CharTokenizer(text)
model = load_pretrained_mini_llama("checkpoints/mini_llama_base.pt", device=device,
                                   config=dict(vocab_size=tok.vocab_size, max_seq_len=256,
                                               d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384))

def compute_attn_weights(x_norm, attn):
    # ... (igual que cap 53)
    ...

N_PROMPTS = 50
WIN = 24
prompts = [text[start:start + WIN] for start in
           [random.randint(0, len(text) - WIN - 1) for _ in range(N_PROMPTS)]]

sum_scores = torch.zeros(4, 4)
for prompt in prompts:
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    with cache_activations(model, [f"blocks.{i}.norm1" for i in range(4)]) as cache:
        with torch.no_grad():
            model(ids)
    for layer in range(4):
        w = compute_attn_weights(cache[f"blocks.{layer}.norm1"],
                                 model.blocks[layer].attn)[0]
        for head in range(4):
            sum_scores[layer, head] += previous_token_score(w[head].cpu())

avg_scores = sum_scores / N_PROMPTS
# ... ordenar y mostrar
```

---

## 4. Output literal

```
Promediando previous_token_score sobre 50 prompts de 24 chars

=== Tabla: previous_token_score promedio por cabeza ===

cabeza                score
----------------------------
block.0 head.0       0.141
block.0 head.1       0.144
block.0 head.2       0.176
block.0 head.3       0.236
block.1 head.0       0.479
block.1 head.1       0.400
block.1 head.2       0.467
block.1 head.3       0.407
block.2 head.0       0.547
block.2 head.1       0.500
block.2 head.2       0.431
block.2 head.3       0.209
block.3 head.0       0.457
block.3 head.1       0.505
block.3 head.2       0.410
block.3 head.3       0.427

=== Top-5 cabezas con mayor previous-token score ===

  rank 1: block.2 head.0  score=0.547
  rank 2: block.3 head.1  score=0.505
  rank 3: block.2 head.1  score=0.500
  rank 4: block.1 head.0  score=0.479
  rank 5: block.1 head.2  score=0.467

=== Bottom-3 cabezas (menor previous-token score) ===

  rank -1: block.0 head.2  score=0.176
  rank -2: block.0 head.1  score=0.144
  rank -3: block.0 head.0  score=0.141

=== Heatmap de la cabeza top-1 sobre prompt 'BRUTUS:\nI am' ===
        B  R  U  T  U  S  : \n  I     a  m
   B   #                                 
   R   #                                 
   U   +  -                              
   T      -  .                           
   U      .  .  -                        
   S      -     .                        
   :                     #               
  \n                     #               
   I                     #               
                               #         
   a                                 #   
   m                                 #
```

---

## 5. Analisis: la jerarquia previous-token emerge en capas 1-2

### El patron es claro: capa 0 < capas 1-3

El bottom-3 son las TRES cabezas de `block.0` (excepto la 3): scores 0.14-0.18. La capa 0 NO tiene cabezas previous-token — sus cabezas operan sobre embeddings crudas y aun no hay representaciones contextuales que faciliten "atender al anterior".

A partir de `block.1`, los scores saltan a 0.40-0.55. **Las cabezas de las capas 1-3 si aprendieron a atender al token anterior**, todas con scores > 2x los de capa 0.

La **cabeza top: `block.2 head.0` con score 0.547**. Sobre 50 prompts diferentes, esta cabeza atiende al token anterior en promedio 54.7% del peso de su softmax. Eso es un patron claro y robusto.

### El heatmap honesto: NO es prev-token puro

Aqui esta la honestidad pedagogica: aunque `block.2 head.0` es la mas previous-token del modelo (score 0.547), su heatmap visual sobre `"BRUTUS:\nI am"` muestra un patron **mas complejo**:

```
   B   #                                 (attiende a si misma — no hay prev)
   R   #                                 (attiende a B = prev) ✓
   U   +  -                              (mezcla: B + R, R es prev)
   T      -  .                           (mezcla)
   ...
   :                     #               (atencion al tokens estructurales)
  \n                     #               (atiende a algun token estructural)
```

Las posiciones **al inicio de palabra** (despues de `:` o `\n`) muestran patrones DIFERENTES — atienden a los tokens estructurales mismos, no al anterior inmediato. Esto es interesante: la cabeza tiene **dos modos**:

1. Modo "previous-token" durante texto continuo
2. Modo "atencion a separadores" cuando hay puntuacion / saltos de linea

Esa heterogeneidad es por que el score global es 0.55 y no 0.95: la cabeza es mayormente prev-token pero no exclusivamente.

### Comparacion con GPT-2 small (Anthropic 2022)

Anthropic encontro en GPT-2 small (12 capas, d_model=768) varias cabezas con scores >0.9 — patrones casi puros. La diferencia con Mini-LLaMA es de escala:

- **GPT-2 small**: 144 cabezas. Sobra capacidad para que algunas se especialicen 100%.
- **Mini-LLaMA**: 16 cabezas. Cada cabeza tiene que "hacer mas cosas" — mezclar previous-token con otros patrones.

A escala chica, las cabezas son **polisemanticas** (similar a la polisemanticidad de neuronas que veremos en cap 59). Para encontrar previous-token heads "puras" necesitas modelos grandes.

---

## 6. ¿Por que esta jerarquia importa?

Las previous-token heads son **el insumo** de las induction heads (cap 55). El argumento de Anthropic es:

```
Capa N:    previous-token head escribe en residual stream:
           "el token anterior fue X"

Capa N+k:  induction head LEE esa info y la usa para
           "completar el patron" en repeticiones
```

Una induction head necesita que ALGUNA capa antes le diga "el anterior fue X". Si no hay previous-token heads en capas tempranas, no pueden formarse induction heads en capas posteriores.

Mini-LLaMA tiene `block.1` y `block.2` con cabezas previous-token fuertes. **Si hay induction heads, deben estar en `block.2` o `block.3`** (despues de las prev-token heads). Esa es la prediccion teorica que probamos en el siguiente capitulo.

---

## 7. Lo que esta tabla NO dice

- **No mide importancia causal**: que `block.2 head.0` tenga score alto NO implica que el modelo *use* esa cabeza para tareas. Para verificarlo: cap 57 (activation patching).
- **No mide especializacion**: una cabeza con score 0.55 puede tener tambien score alto en otras metricas (induction, name-mover, etc.). No es exclusiva.
- **No considera contenido**: solo mide DONDE atiende, no QUE informacion mueve. Para eso: cap 56 (QK/OV).

Esta es la naturaleza de los heatmaps y scores: descripcion, no causalidad. La interpretabilidad mecanicista necesita las dos cosas — primero descubrir que cabezas tienen patrones; despues verificar que esos patrones son causales para tareas concretas.

---

## 8. Preguntas de verificacion

**1. ¿Por que el score de `block.0` es tan bajo (0.14-0.24)?**

La capa 0 opera directamente sobre las embeddings de los tokens, que son representaciones cruadas sin contexto. Para que una cabeza pueda decidir "atender al anterior", necesita features que distingan "soy posicion i" vs "soy posicion i+1" — y esas features las construye el modelo via la atencion + FFN de capas previas. La capa 0 ESTA construyendo esas features (con RoPE rotando Q y K segun posicion), pero aun no las tiene listas para EXPLOTARLAS en patrones complejos como previous-token. Por eso el patron emerge en capa 1: cuando ya hay una capa de procesamiento previo que provee features posicionales utilizables.

**2. Si Mini-LLaMA tiene cabezas con score 0.55 maximo, ¿como se ven cabezas con score 0.95 en GPT-2?**

En GPT-2 small, la matriz de atencion de una previous-token head tipica tiene practicamente 1.0 en `attn[i, i-1]` y 0 en todas las otras celdas — una sub-diagonal limpia. El modelo grande tiene capacidad para que una cabeza se "especialice" 100% en este patron, mientras que en Mini-LLaMA la cabeza tiene que mezclar previous-token con otros comportamientos por escasez de capacidad. La regla general: a mas parametros (mas cabezas, mas capas, mas d_model), mayor probabilidad de cabezas especializadas con patrones puros.

**3. ¿Que significa que el promedio sobre 50 prompts sea estable?**

Si las cabezas tuvieran patrones aleatorios o dependientes del contenido especifico del prompt, los scores fluctuarian mucho prompt-a-prompt y el promedio seria ruidoso (alta varianza). Que el promedio converja a valores claros (0.14 para capa 0, 0.40-0.55 para capas 1-3) indica que los patrones SON robustos: una cabeza que tiende a atender al anterior lo hace consistentemente sobre cualquier prompt de Shakespeare. Esa estabilidad es lo que justifica llamarlas "cabezas previous-token" — son una propiedad estructural del modelo, no un accidente de un prompt particular.
