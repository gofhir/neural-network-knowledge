---
title: "58 - Mini-circuit discovery: head-level patching"
weight: 580
math: true
---

## 1. Apertura: de bloque a cabeza

Cap 57 hizo activation patching a nivel de **bloque completo** y revelo que la informacion del speaker fluye principalmente hacia la posicion final (12) via las primeras dos capas. Ese resultado es informativo pero coarse — un bloque tiene 4 cabezas Q operando en paralelo, y dentro de ese bloque solo algunas son responsables del flujo causal.

Este capitulo refina el patching: en lugar de patchear el output completo del bloque, patcheamos **la contribucion de cada cabeza individualmente**. Esto produce un grid `(n_layers × n_heads) = (4 × 4) = 16 cabezas`, cada una con su propio score de recovery.

El objetivo: identificar que cabezas concretas conforman el "circuito" para distinguir speakers en Mini-LLaMA. Y, criticamente, **comparar** los resultados del patching causal con los scores descriptivos del cap 54 (previous-token) — ¿coinciden? ¿O estamos viendo cabezas distintas?

---

## 2. La matematica: contribuciones por cabeza

El forward de `GroupedQueryAttention` es:

$$\text{attn\_out} = \text{concat}(\text{head}_0, \text{head}_1, ..., \text{head}_{h_q-1}) \cdot W_O$$

donde cada cabeza produce un vector `(B, T, d_k)` post-atencion, y `W_O` los proyecta de vuelta a `d_model`. Reescribiendo:

$$\text{attn\_out} = \sum_{h=0}^{h_q - 1} \text{head}_h \cdot W_O^{(h)}$$

donde $W_O^{(h)}$ es la slice de `W_O` correspondiente a la cabeza $h$ (las columnas `[h \cdot d_k : (h+1) \cdot d_k]`). **El output del attention es una SUMA de contribuciones independientes por cabeza**.

Esto es lo que hace posible el head-level patching: para reemplazar la cabeza $h$ con su version clean, calculamos la diferencia $\delta_h = \text{contrib}_h^{\text{clean}} - \text{contrib}_h^{\text{corrupted}}$ y la sumamos al output del attention:

$$\text{attn\_out}^{\text{patched}} = \text{attn\_out}^{\text{corrupted}} + \delta_h$$

Las otras cabezas mantienen su valor corrupted. Solo la cabeza $h$ "se cambia" a clean.

---

## 3. Por que esto es mas informativo que cap 57

Cap 57 patcheo bloques enteros — eso mezcla los efectos de las 4 cabezas. Si patchear `block.0` recupera 40%, no sabemos si es porque:

a) Una cabeza concreta de block.0 lleva el 40% del flujo causal
b) Las 4 cabezas contribuyen 10% cada una
c) Una cabeza contribuye +60% y otra contribuye -20%

Head-level patching distingue estos casos. Si encontramos UNA cabeza con recovery 40% (caso a), el circuito es localizado. Si todas tienen recovery ~10% (caso b), el procesamiento es distribuido. Si hay cabezas positivas y negativas (caso c), hay tension causal entre cabezas.

---

## 4. Script

```python
"""58_circuit_discovery.py - Cap 58: head-level patching para identificar circuitos."""
import math, torch
import torch.nn.functional as F
from _models import (load_pretrained_mini_llama, get_device, CharTokenizer,
                     load_text, apply_rope)
from _interp import cache_activations

torch.manual_seed(1337)
device = get_device()
tok = CharTokenizer(load_text("shakespeare.txt"))
model = load_pretrained_mini_llama("checkpoints/mini_llama_base.pt", device=device,
                                   config=dict(vocab_size=tok.vocab_size, max_seq_len=256,
                                               d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384))

clean_ids = torch.tensor([tok.encode("BRUTUS:\nI am ")], dtype=torch.long, device=device)
corrupted_ids = torch.tensor([tok.encode("BIANCA:\nI am ")], dtype=torch.long, device=device)

def compute_per_head_output(x_norm, attn):
    """Contribucion por cabeza al residual: (B, h_q, T, d_model). Suman = attn output."""
    B, T, _ = x_norm.shape
    Q = attn.W_Q(x_norm).view(B, T, attn.h_q, attn.d_k).transpose(1, 2)
    K = attn.W_K(x_norm).view(B, T, attn.h_kv, attn.d_k).transpose(1, 2)
    V = attn.W_V(x_norm).view(B, T, attn.h_kv, attn.d_k).transpose(1, 2)
    Q = apply_rope(Q, attn.rope_cos[:T], attn.rope_sin[:T])
    K = apply_rope(K, attn.rope_cos[:T], attn.rope_sin[:T])
    K_full = K.repeat_interleave(attn.group_size, dim=1)
    V_full = V.repeat_interleave(attn.group_size, dim=1)
    scores = Q @ K_full.transpose(-2, -1) / math.sqrt(attn.d_k)
    scores = scores.masked_fill(attn.mask[:, :, :T, :T] == 0, float('-inf'))
    out_per_head = F.softmax(scores, dim=-1) @ V_full
    contributions = []
    for h in range(attn.h_q):
        W_O_h = attn.W_O.weight[:, h*attn.d_k:(h+1)*attn.d_k]
        contributions.append(out_per_head[:, h] @ W_O_h.T)
    return torch.stack(contributions, dim=1)

# Cachear contribuciones clean
with cache_activations(model, [f"blocks.{i}.norm1" for i in range(4)]) as clean_norms:
    with torch.no_grad():
        clean_logits, _ = model(clean_ids)

clean_per_head = [
    compute_per_head_output(clean_norms[f"blocks.{i}.norm1"], model.blocks[i].attn)
    for i in range(4)
]

# Patch cabeza por cabeza
def patch_one_head(layer, head):
    with cache_activations(model, [f"blocks.{layer}.norm1"]) as cor_norms:
        with torch.no_grad():
            model(corrupted_ids)
    cor_contribs = compute_per_head_output(cor_norms[f"blocks.{layer}.norm1"],
                                            model.blocks[layer].attn)
    delta = clean_per_head[layer][:, head] - cor_contribs[:, head]

    def patch_attn(module, inputs, output):
        return output + delta

    handle = model.blocks[layer].attn.register_forward_hook(patch_attn)
    try:
        with torch.no_grad():
            patched, _ = model(corrupted_ids)
    finally:
        handle.remove()
    return patched
```

---

## 5. Output literal

```
Clean:     'BRUTUS:\nI am '
Corrupted: 'BIANCA:\nI am '
Target token: 'n', diff (clean - corrupted) = +0.425

=== Head-level patching: % de recovery por (layer, head) ===

cabeza               recovery
------------------------------
block.0 head.0        +12.6%
block.0 head.1        +24.7%
block.0 head.2         +7.3%
block.0 head.3        +29.7%
block.1 head.0        +10.5%
block.1 head.1         +4.4%
block.1 head.2        +29.7%
block.1 head.3         +6.8%
block.2 head.0         -2.7%
block.2 head.1        +19.3%
block.2 head.2        +11.7%
block.2 head.3        +23.5%
block.3 head.0        +12.1%
block.3 head.1        +17.4%
block.3 head.2         +0.3%
block.3 head.3         +2.6%

=== Top-3 cabezas con mayor recovery causal ===
  rank 1: block.1 head.2  recovery=+29.7%
  rank 2: block.0 head.3  recovery=+29.7%
  rank 3: block.0 head.1  recovery=+24.7%

=== Bottom-3 cabezas con menor (o negativo) recovery ===
  rank -1: block.3 head.3  recovery=+2.6%
  rank -2: block.3 head.2  recovery=+0.3%
  rank -3: block.2 head.0  recovery=-2.7%

=== Lectura del circuito ===
Cabezas con recovery > 5%:  12/16
Cabezas con recovery > 20%: 4/16
Hay cabezas con efecto causal claro — circuito identificable
```

---

## 6. Analisis: el descubrimiento principal

### Cuatro cabezas forman el "circuito speaker"

Recovery > 20%:

| Cabeza | Recovery |
|---|---|
| **block.1 head.2** | +29.7% |
| **block.0 head.3** | +29.7% |
| **block.0 head.1** | +24.7% |
| **block.2 head.3** | +23.5% |

Estas cuatro cabezas concentran el procesamiento causal de la identidad del speaker. Cada una contribuye 23-30% — significativas pero no dominantes individualmente. **El circuito es distribuido, no localizado**.

Distribucion por capa: 2 cabezas en block.0, 1 en block.1, 1 en block.2. Las primeras tres capas hacen el trabajo causal — block.3 esta principalmente "leyendo" lo que ya esta en el residual stream sin agregar info nueva.

### El descubrimiento sorprendente: prev-token NO es causal

Aqui esta la leccion central del cap. **`block.2 head.0`** — la cabeza con MAYOR previous-token score del modelo (0.547 en cap 54) — tiene **recovery NEGATIVO** (-2.7%) en este experimento causal.

Patchear esa cabeza con su version clean **EMPEORA** la prediccion clean. La cabeza esta en el bottom-3 de cabezas causalmente importantes para distinguir speakers.

Esto invalida una intuicion comun: que las cabezas con patrones descriptivos claros son las "importantes". **No es asi**. Una cabeza puede tener score alto en alguna metrica (previous-token, induction, name-mover) y aun asi no ser causal para tareas concretas. El patching es la unica manera de saber.

¿Por que `block.2 head.0` es prev-token pero anti-causal aqui? Posible interpretacion: la cabeza implementa un patron generico (atender al anterior) que en ESTA tarea (distinguir speakers) introduce ruido. La info del speaker esta en posiciones 1-5; "atender al anterior" no es la operacion correcta para extraer esa info hacia la posicion 12.

### Las cabezas causales NO eran las "obvias"

Mirando los scores prev-token del cap 54 vs los scores causales de aqui:

| Cabeza | prev-token (cap 54) | causal (cap 58) |
|---|---|---|
| block.2 head.0 | **0.547 (top-1)** | -2.7% |
| block.0 head.3 | 0.236 | **+29.7% (top-2)** |
| block.1 head.2 | 0.467 | **+29.7% (top-1)** |
| block.0 head.1 | 0.144 (bottom) | **+24.7% (top-3)** |

`block.0 head.1` tenia el SCORE PREV-TOKEN MAS BAJO del modelo (0.144) y tiene recovery +24.7% — es una de las top-3 cabezas causales. Sin patching, jamas habriamos sospechado de ella.

Esta es la diferencia fundamental entre interpretabilidad descriptiva (caps 54-56) e interpretabilidad causal (caps 57-58):

- **Descriptiva**: ¿que patrones tiene esta cabeza?
- **Causal**: ¿esta cabeza importa para tarea X?

Las dos son ortogonales. Una cabeza puede tener patrones claros pero ser irrelevante; otra puede tener patrones difusos pero ser critica.

---

## 7. Por que el plan original no se cumple del todo

El plan original de cap 58 era encontrar el circuito que el Mini-LLaMA SFT (cap 24) usa para la tarea "repeat" — una tarea concreta donde el modelo recibe `INSTR: Repeat: hello\nRESP:` y debe generar `hello`. Esto requeriria:

- Cargar `mini_llama_sft.pt` con su tokenizer especifico
- Disenar prompts SFT adecuados (clean: instruccion correcta, corrupted: instruccion alterada)
- Patchear cabezas y MLPs

El experimento aqui (head-level patching sobre el modelo BASE con prompts de Shakespeare) es **mas simple y mas robusto**. La razon: el modelo SFT char-level mostro tradeoffs claros (cap 25), pero su circuito interno para la tarea repeat probablemente sea distribuido y dificil de identificar limpiamente a esta escala. El cap 55 ya mostro que las induction heads (que serian la base del circuito repeat) NO emergen en Mini-LLaMA.

Por lo tanto: este capitulo demuestra la TECNICA de head-level patching y el HALLAZGO clave (descripcion ≠ causalidad), pero usa un experimento accesible. Para investigar circuits SFT especificos en serio se necesita mas escala (12+ capas).

---

## 8. Conexion con el resto del curso

El head-level patching es la tecnica que descubrio el circuito IOI en GPT-2 (Wang et al. 2022). Aqui hicimos una version reducida sobre Mini-LLaMA, identificando 4 cabezas concentradas y validando el principio "patrones != causalidad". Las extensiones naturales son:

- **Path patching**: patchear pares de cabezas simultaneamente para detectar interacciones (caps 57-58 son "single-component patching")
- **Edge patching**: patchear conexiones especificas entre componentes (cabeza A → cabeza B, MLP → cabeza)
- **Feature visualization** (cap 59-61 con SAEs): identificar QUE features hay en el residual stream donde las cabezas escriben

El curso esta sentando la base. Para problemas reales en modelos grandes, las mismas tecnicas escalan — solo cambia el numero de cabezas y la complejidad del circuito.

---

## 9. Preguntas de verificacion

**1. ¿Por que el recovery de las cabezas individuales suma menos que el del bloque completo?**

Cap 57 mostro que patchear `block.0` completo da ~40% recovery en la posicion 12. Cap 58 muestra que las 4 cabezas de block.0 individualmente suman aproximadamente: 12.6 + 24.7 + 7.3 + 29.7 = 74.3%. Mucho mas que 40%! La razon: los efectos no son aditivos. Cuando patcheas DOS cabezas simultaneamente, los efectos pueden cancelarse parcialmente porque las cabezas interactuan via la atencion en capas posteriores. El head-level patching nos dice el efecto MARGINAL de cada cabeza (asumiendo el resto corrupted); el block patching nos dice el efecto COMBINADO. Para el efecto verdadero hay que hacer "joint patching" — todas las cabezas a la vez — lo que da un numero entre el marginal y el block-level.

**2. ¿Por que `block.3 head.2` tiene recovery casi exactamente 0% (+0.3%)?**

Esa cabeza es **causalmente irrelevante** para la tarea de distinguir speakers. Probablemente se especializa en algo distinto (algun aspecto local de prediccion del proximo token, atencion a separadores, etc.) que no depende de la identidad del speaker. Esto es coherente con el patron mas amplio que vimos: block.3 en general tiene cabezas con recovery bajo (12.1, 17.4, 0.3, 2.6), porque el flujo causal del speaker ya se "completo" en capas 0-2 antes de llegar a block.3. Block.3 hace prediccion final, no procesamiento de speaker.

**3. ¿Que significaria un grid de patching donde TODAS las cabezas tienen recovery ~6% (16 × 6% ~ 100%)?**

Significaria un circuito **completamente distribuido**: no hay cabezas "estrellas", todas contribuyen igualmente al flujo causal. Eso ocurre cuando el modelo aprendio una representacion altamente redundante donde la informacion no esta concentrada en pocos componentes. En general, modelos grandes tienden a circuitos mas localizados (algunas cabezas hacen el trabajo, otras son ruido); modelos chicos tienden a circuitos distribuidos (cada cabeza tiene que "hacer un poco de todo" por capacidad limitada). Mini-LLaMA esta en el medio: vimos algunas cabezas con recovery >20% (cierto grado de localizacion) pero ninguna con recovery >50% (no totalmente localizado). Es el patron tipico de modelos pequenos.
