---
title: "56 - QK / OV decomposition: matematica de las cabezas"
weight: 560
math: true
---

## 1. Apertura: separar "a quien atender" de "que mover"

Cap 54 identifico que `block.2 head.0` es la cabeza con mayor previous-token score. Pero **eso solo nos dice donde mira la cabeza** — no que informacion mueve. Una cabeza puede atender a la posicion correcta pero modificar lo que copia hasta hacerlo irreconocible. Para entender que hace una cabeza, necesitamos descomponerla en sus dos circuitos matematicos:

**QK circuit** ($W_Q W_K^T$): determina **a que tokens atender** dada una query. Es la matriz que el modelo usa para decidir el patron de atencion.

**OV circuit** ($W_V W_O$): determina **que escribir al residual stream** dado un token fuente. Es la matriz que mueve informacion de la posicion atendida al residual stream del query.

Esta descomposicion (Elhage et al. 2021) es la herramienta matematica fundamental de la interpretabilidad mecanicista. Permite responder preguntas como:

- ¿Esta cabeza COPIA literalmente lo que atiende, o lo TRANSFORMA?
- ¿Que pares (query_token, key_token) tienen mayor afinidad?
- ¿Que tokens "se escriben" al stream cuando esta cabeza atiende a un token X?

---

## 2. Las matrices: shapes para Mini-LLaMA

Mini-LLaMA usa GQA con `h_q=4`, `h_kv=2`, `d_k=32`. Esto significa:

- `W_Q`: shape `(h_q * d_k, d_model) = (128, 128)`. Cada row de 32 dimensiones es una "rebanada" para una cabeza Q.
- `W_K`: shape `(h_kv * d_k, d_model) = (64, 128)`. Solo 2 grupos K/V comparten K — head 0,1 usan W_K[0:32], head 2,3 usan W_K[32:64].
- `W_V`: igual a W_K.
- `W_O`: shape `(d_model, h_q * d_k) = (128, 128)`. Cada column de 32 dim es la rebanada de output para una cabeza Q.

Para descomponer **`block.2 head.0`**:

```python
HEAD = 0
kv_group = HEAD // group_size  # 0 // 2 = 0

W_Q = attn.W_Q.weight[HEAD*d_k:(HEAD+1)*d_k, :].T          # (128, 32)
W_K = attn.W_K.weight[kv_group*d_k:(kv_group+1)*d_k, :].T  # (128, 32)
W_V = attn.W_V.weight[kv_group*d_k:(kv_group+1)*d_k, :].T  # (128, 32)
W_O = attn.W_O.weight[:, HEAD*d_k:(HEAD+1)*d_k].T          # (32, 128)
```

Los circuitos:

$$QK = W_Q W_K^T \quad \text{shape: } (d_{\text{model}}, d_{\text{model}})$$
$$OV = W_V W_O \quad \text{shape: } (d_{\text{model}}, d_{\text{model}})$$

Ambos son matrices `128 × 128` pero **rank ≤ 32** (limitado por `d_k`). Esta limitacion de rank es importante: significa que cada cabeza solo puede capturar 32 patrones independientes de query-key o de movimiento de informacion.

---

## 3. Aplicar a embeddings: la "tabla de afinidad" sobre el vocab

Una vez descompuesta la cabeza, queremos saber: **¿como interactuan los tokens del vocab con esta cabeza?**

La idea es proyectar `QK` y `OV` al espacio del vocabulario via la matriz de embeddings `E` (shape `(vocab, d_model) = (65, 128)` para char-level Shakespeare):

$$QK_{\text{emb}} = E \cdot QK \cdot E^T \quad \text{shape: } (65, 65)$$

Cada celda `(i, j)` de `QK_emb` es el "score" que tiene un query token `i` atendiendo a un key token `j` SOLO via esta cabeza, ignorando el contexto. Tokens que aparecen como `(query, key)` con score alto son los preferidos por esta cabeza para "mirar".

Analogamente:

$$OV_{\text{emb}} = E \cdot OV \cdot E^T$$

Cada celda `(i, j)` es cuanto el residual stream "se mueve hacia la direccion del token j" cuando la cabeza atiende a un token con embedding cercano a `i`. Si la cabeza es de COPIA pura, `OV_emb` es proporcional a la matriz de Gram de embeddings — los tokens "se copian a si mismos".

---

## 4. Script

```python
"""56_qk_ov_decomposition.py - Cap 56: descomposicion QK y OV de la cabeza top."""
import torch
from _models import load_pretrained_mini_llama, get_device, CharTokenizer, load_text
from _interp import qk_circuit, ov_circuit

torch.manual_seed(1337)
device = get_device()
text = load_text("shakespeare.txt")
tok = CharTokenizer(text)
model = load_pretrained_mini_llama("checkpoints/mini_llama_base.pt", device=device,
                                   config=dict(vocab_size=tok.vocab_size, max_seq_len=256,
                                               d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384))

LAYER, HEAD = 2, 0
attn = model.blocks[LAYER].attn
d_k = attn.d_k
kv_group = HEAD // attn.group_size

W_Q = attn.W_Q.weight[HEAD*d_k:(HEAD+1)*d_k, :].T
W_K = attn.W_K.weight[kv_group*d_k:(kv_group+1)*d_k, :].T
W_V = attn.W_V.weight[kv_group*d_k:(kv_group+1)*d_k, :].T
W_O = attn.W_O.weight[:, HEAD*d_k:(HEAD+1)*d_k].T

QK = qk_circuit(W_Q, W_K).cpu()
OV = ov_circuit(W_V, W_O).cpu()

# QK / OV aplicados a embeddings
E = model.tok_emb.weight.detach().cpu()
QK_emb = E @ QK @ E.T
OV_emb = E @ OV @ E.T

# Top pares
qk_no_diag = QK_emb.clone(); qk_no_diag.fill_diagonal_(float('-inf'))
top_qk = qk_no_diag.flatten().topk(5).indices
for fi in top_qk.tolist():
    q, k = fi // QK_emb.shape[1], fi % QK_emb.shape[1]
    print(f"query={tok.id_to_char[q]!r} -> key={tok.id_to_char[k]!r}  score={QK_emb[q, k]:.3f}")
```

---

## 5. Output literal

```
=== Descomposicion de block.2 head.0 (top previous-token) ===

d_model=128, d_k=32, kv_group=0
W_Q shape: (128, 32)
W_K shape: (128, 32)
W_V shape: (128, 32)
W_O shape: (32, 128)

QK circuit shape: (128, 128)
OV circuit shape: (128, 128)

=== Estadisticas del QK circuit ===
||QK||_F (Frobenius)  = 3.698
Rank efectivo (>1e-4) = 32
Top-5 singular values = [2.32, 1.20, 1.07, 0.87, 0.76]

=== Estadisticas del OV circuit ===
||OV||_F (Frobenius)  = 2.584
Rank efectivo (>1e-4) = 32
Top-5 singular values = [0.93, 0.84, 0.72, 0.69, 0.62]

=== Test: ¿OV se parece a la identidad? (copy circuit test) ===
||OV - I||_F / ||I||_F = 1.035
Si ~0: OV es matriz identidad (copy puro)
Si ~1: OV difiere completamente de identidad

=== QK aplicado a embeddings: ¿que tokens prefiere atender esta cabeza? ===
Top-5 pares (query, key) con mayor score (excluyendo diagonal):
  query='\n'  -> key='w'   score=23.528
  query='\n'  -> key='K'   score=21.999
  query='\n'  -> key=':'   score=21.394
  query='\n'  -> key=','   score=20.963
  query=' '   -> key=':'   score=19.364

=== OV aplicado a embeddings: ¿que tokens copia esta cabeza? ===
Top-5 pares (input_token -> escribe_token) con mayor score:
  input='\n'  -> output='$'   score=13.027
  input='\n'  -> output='K'   score=12.216
  input='\n'  -> output='D'   score=12.125
  input='\n'  -> output='3'   score=12.109
  input='\n'  -> output=':'   score=11.780
```

---

## 6. Analisis: la "previous-token head" NO es una copy head

### El test de identidad falla

`||OV - I||_F / ||I||_F = 1.035` — la matriz OV se parece a identidad casi como una matriz aleatoria de la misma magnitud. **NO es una copy head**. Si lo fuera, esperariamos que cuando atiende al token anterior, simplemente escribiera ese token al stream del query. Lo que hace es mas complejo: **transforma la informacion** del token atendido antes de escribirla.

### El QK favorece structure tokens (`\n`, ` `, `:`)

Los pares con mayor score en `QK_emb` son consistentes:

- `query=\n -> key=w/K/:/,`: cuando la query es un salto de linea, la cabeza prefiere atender a tokens estructurales o consonantes prominentes.
- `query=' ' -> key=':'`: espacios atienden a dos puntos.

Esto sugiere que la cabeza es **structure-aware**: tiene patrones especificos para tokens que delimitan estructura (saltos de linea, espacios, puntuacion). El previous-token score alto del cap 54 es un epifenómeno: en texto continuo, cada token (que suele ser una letra) atiende al anterior — pero el patron dominante de la cabeza es manejar la estructura.

### El OV transforma `\n` en otros tokens estructurales

Los pares de OV muestran que cuando la cabeza atiende a `\n`, escribe al stream componentes hacia tokens como `$`, `K`, `D`, `3`, `:`. Estos NO son copias literales — son una transformacion. La cabeza esta sumando al stream del query un vector que apunta hacia "regiones del espacio" asociadas a varios tokens distintos.

Esto es coherente con la naturaleza polisematica de cabezas en modelos chicos (cap 54 seccion 5): una cabeza tiene que codificar varios "concepts" en sus 32 dimensiones, asi que el OV no es nunca una copia limpia — es una mezcla.

### Por que ESTA cabeza tiene previous-token score alto

Si la cabeza es structure-aware y NO es copy, ¿por que aparecio top-1 en previous-token? Porque en texto continuo (la mayoria del corpus), los tokens contiguos son letras/espacios. La cabeza atiende a tokens estructurales cuando los hay, y a "el token anterior" cuando no hay estructura. Promediado sobre todo el corpus, eso da un score de previous-token alto sin ser primariamente una previous-token head.

Esta es una leccion importante: **scores de patrones miden CO-OCURRENCIA, no causalidad**. La cabeza puede tener score alto en X sin que X sea su funcion principal. La descomposicion QK/OV revela el comportamiento estructural verdadero.

---

## 7. Por que el rank es exactamente 32

La matriz QK = $W_Q W_K^T$ tiene shape (128, 128) pero rank limitado por `min(rank(W_Q), rank(W_K)) ≤ d_k = 32`. Lo mismo para OV. Esto es estructural: cada cabeza de atencion solo puede capturar **32 dimensiones independientes** de patrones.

Practicamente: si el modelo necesita distinguir 100 patrones distintos, una cabeza no alcanza. Necesita varias cabezas en paralelo (`h_q=4` da 128 dimensiones potenciales en total) o composicion entre capas (las cabezas de capa N usan info escrita por capas anteriores).

Esta limitacion **es por que existe multi-head attention**: con una sola cabeza grande (`d_k=128`), tendrias rank 128 pero sin la modularidad de cabezas separadas. Multi-head fragmenta el espacio en 4 cabezas de rank 32 cada una, permitiendo que cada una se especialice.

Los singular values del QK aqui son `[2.32, 1.20, 1.07, 0.87, 0.76, ...]`. Decay relativamente lento — la cabeza usa muchas de sus 32 dimensiones. Comparativamente, las induction heads canonicas en GPT-2 small tienen singular values muy concentrados en las primeras 2-3 dimensiones (un patron bajo rango), reflejando que su funcion es estrecha.

---

## 8. Lo que falta para una analisis completo

QK/OV decomposition aqui se queda en lo descriptivo. Para llegar a circuitos completos hace falta:

- **Composicion entre cabezas**: una induction head no opera en aislamiento — depende de previous-token heads via composicion `OV_prev_token @ QK_induction`. Eso requiere multiplicar circuitos de capas distintas.
- **Composicion con MLPs**: las FFN tambien escriben al residual stream. Los circuitos completos incluyen MLPs.
- **Path patching**: para verificar que un circuito hipotetico es causal, no solo correlacional. Eso es cap 57.

Para Mini-LLaMA, llegar a circuitos completos esta limitado por la falta de patrones puros (vimos en cap 55 que no hay induction heads). Los caps 57-58 abordan el problema desde activation patching: en lugar de descomponer matrices, hacer intervenciones causales.

---

## 9. Preguntas de verificacion

**1. ¿Por que la cabeza con previous-token score alto NO es una copy head?**

Porque previous-token score solo mide DONDE atiende la cabeza (a la posicion i-1), no QUE informacion mueve. Una cabeza puede atender al token anterior y luego transformar drasticamente esa informacion antes de escribirla al residual stream — su OV circuit dicta que. En `block.2 head.0` vimos que OV no es identidad (||OV - I||/||I|| = 1.04), por lo tanto la cabeza atiende al anterior pero NO copia su valor literalmente. Distinguir QK (donde mira) de OV (que mueve) es el punto central de la decomposition.

**2. ¿Que esperariamos ver en el QK de una induction head pura?**

Una induction head ideal en un prompt `[A][B] ... [A]?` debe atender desde la segunda `[A]` hacia la posicion `i+1` (donde estaba `[B]` la primera vez). Su QK circuit deberia tener afinidad alta para pares donde el query es `[A]` y la key es la EMBEDDING POSICIONAL de `[B]` (no el token `[B]` mismo — eso seria una "matching" head, distinta). En la practica, las induction heads canonicas dependen de **composicion**: las previous-token heads escriben al stream "soy la posicion despues de [A]", y la induction head matchea su query (codificando "estoy en la posicion despues de la primera [A]") contra esas keys. La QK de la induction sola no muestra el patron — solo se ve cuando se compone con el OV de la previous-token de capas anteriores.

**3. ¿Por que el rank de QK y OV son ambos 32 (igual a d_k)?**

QK = $W_Q W_K^T$. $W_Q$ tiene shape `(d_model, d_k) = (128, 32)`, asi que rank(W_Q) ≤ 32. Idem $W_K$. Por la propiedad de rank de productos: rank(AB) ≤ min(rank(A), rank(B)). Por lo tanto rank(QK) ≤ 32. Empiricamente, rank(QK) = 32 porque W_Q y W_K son matrices "generic" entrenadas (no tienen filas linealmente dependientes). Lo mismo para OV. Esta limitacion de rank ES por que se usa multi-head: cada cabeza tiene capacidad estrecha (32 patrones), pero `h_q` cabezas en paralelo y multiples capas componen para producir comportamiento complejo.
