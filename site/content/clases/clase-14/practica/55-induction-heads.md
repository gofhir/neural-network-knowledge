---
title: "55 - Induction heads: el descubrimiento que NO emerge a esta escala"
weight: 550
math: true
---

## 1. Apertura: el patron `[A][B] ... [A] -> [B]`

Las **induction heads** son uno de los descubrimientos centrales de la interpretabilidad mecanicista. Anthropic (Olsson et al. 2022) las identifico en GPT-2 y mostro que son el sustrato de **in-context learning** — la capacidad de los LLMs de aprender patrones del contexto sin entrenamiento.

El patron es simple. Dado el contexto:

```
... [A] [B] ... [A]
```

donde `[A][B]` aparecio una vez antes en la secuencia, una induction head atiende desde la **segunda** ocurrencia de `[A]` hacia la posicion **inmediatamente despues** de la primera ocurrencia de `[A]` — es decir, hacia donde estaba `[B]` la primera vez. Esto le permite al modelo predecir que probablemente viene `[B]` otra vez.

```
ABCD ... A ?
         |
         v
       miro (atencion) hacia donde estaba B la primera vez
       -> predigo B
```

Es un mecanismo elegante de "completar patrones" emergente. Las induction heads aparecen tipicamente en capas **2-6** de modelos de tamano mediano, y dependen estructuralmente de las previous-token heads (cap 54) que les dan la informacion sobre "que token vino antes de cual".

Este capitulo busca induction heads en Mini-LLaMA. **Spoiler: no las vamos a encontrar.** Pero el resultado honesto es pedagogicamente importante.

---

## 2. La metrica: `induction_score`

Definida en `_interp.py`:

```python
def induction_score(attn, ids):
    """Score de induccion: para token repetido en posicion j,
    cuanto atiende a la posicion i+1 (donde estaba el siguiente token la primera vez)."""
    T = attn.shape[0]
    scores = []
    for j in range(2, T):
        tok = ids[j]
        for i in range(j - 1):
            if ids[i] == tok and i + 1 < j:
                scores.append(attn[j, i + 1].item())
                break
    return sum(scores) / len(scores) if scores else 0.0
```

Para cada token repetido en posicion `j`, busca su primera aparicion en `i`, y mide cuanto la cabeza atiende desde `j` hacia `i+1` (el "siguiente token" de la primera ocurrencia). Si la cabeza es induction, ese peso sera alto.

---

## 3. Los prompts: secuencias repetidas

Para que la metrica sea informativa, el prompt debe TENER repeticion. Generamos secuencias del estilo `XYZ...XYZ` — un segmento aleatorio de 12 caracteres concatenado con su copia exacta. Cada caracter aparece dos veces, en posiciones separadas por exactamente 12 chars.

Una cabeza de induccion ideal sobre estos prompts atenderia desde la posicion 13 (segunda X) hacia la posicion 1 (donde estaba Y la primera vez).

Promediamos sobre **30 prompts** distintos para estabilizar la metrica.

---

## 4. Script

```python
"""55_induction_heads.py - Cap 55: induction heads sobre prompts repetidos."""
import math, random, torch
import torch.nn.functional as F
from _models import (load_pretrained_mini_llama, get_device, CharTokenizer,
                     load_text, apply_rope)
from _interp import cache_activations, induction_score

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

N_PROMPTS = 30
SEG_LEN = 12
vocab_chars = list(tok.id_to_char.values())

prompts_ids = []
for _ in range(N_PROMPTS):
    seg = [random.choice(vocab_chars) for _ in range(SEG_LEN)]
    full = seg + seg
    ids = tok.encode("".join(full))[:2 * SEG_LEN]
    prompts_ids.append(torch.tensor(ids, dtype=torch.long))

sum_scores = torch.zeros(4, 4)
for ids_t in prompts_ids:
    ids = ids_t.unsqueeze(0).to(device)
    with cache_activations(model, [f"blocks.{i}.norm1" for i in range(4)]) as cache:
        with torch.no_grad():
            model(ids)
    for layer in range(4):
        w = compute_attn_weights(cache[f"blocks.{layer}.norm1"],
                                 model.blocks[layer].attn)[0]
        for head in range(4):
            sum_scores[layer, head] += induction_score(w[head].cpu(), ids_t)
avg_scores = sum_scores / len(prompts_ids)
```

---

## 5. Output literal

```
Generados 30 prompts repetidos de longitud 24

Ejemplo: 'hIdkhalNhApmhIdkhalNhApm'

=== Tabla: induction_score promedio por cabeza ===

cabeza                score
----------------------------
block.0 head.0       0.057
block.0 head.1       0.052
block.0 head.2       0.051
block.0 head.3       0.045
block.1 head.0       0.015
block.1 head.1       0.012
block.1 head.2       0.019
block.1 head.3       0.011
block.2 head.0       0.007
block.2 head.1       0.012
block.2 head.2       0.011
block.2 head.3       0.034
block.3 head.0       0.007
block.3 head.1       0.008
block.3 head.2       0.011
block.3 head.3       0.011

=== Top-5 cabezas con mayor induction score ===

  rank 1: block.0 head.0  score=0.057
  rank 2: block.0 head.1  score=0.052
  rank 3: block.0 head.2  score=0.051
  rank 4: block.0 head.3  score=0.045
  rank 5: block.2 head.3  score=0.034

=== Honestidad: lectura de los resultados ===
  Cabeza top con score 0.057 < 0.3: NO hay induction heads claras
  Limitacion de escala: Anthropic encontro induction en GPT-2 small (12 capas)
  Mini-LLaMA tiene 4 capas — posiblemente insuficiente para induction emergente
```

---

## 6. Analisis honesto: por que NO emergen induction heads

### El score top es 0.057 — basicamente ruido

Para ser una induction head, una cabeza necesitaria score >0.3 (umbral pragmatico) o >0.5 (umbral de Anthropic). El score maximo en Mini-LLaMA es **0.057** — mas de un orden de magnitud bajo. Comparativamente, las previous-token heads del cap 54 tenian scores 0.4-0.55 (claros y robustos).

La distribucion de scores tampoco muestra estructura interesante:

- `block.0`: scores 0.04-0.06 (todos similares — ruido uniforme)
- `block.1` a `block.3`: scores 0.007-0.034 (aun mas bajos)

Que los scores mas altos esten en `block.0` (donde no esperariamos induction) confirma que NO hay senal — la cabeza top esta atendiendo difusamente y por azar capta algun peso en posiciones "post-anterior". No es un patron real.

### Tres razones esperables

**1. Profundidad insuficiente.** Las induction heads dependen estructuralmente de las previous-token heads — necesitan que ALGUNA capa anterior les diga "el token previo fue X". Anthropic encontro que las induction heads emergen en capas 2-6 de GPT-2 small (12 capas), DESPUES de previous-token heads en capas 0-2. Mini-LLaMA tiene solo **4 capas**: las previous-token heads emergen en capas 1-2 (cap 54), dejando solo capas 2-3 para induction. Eso puede ser insuficiente para el patron de procesamiento de dos pasos que requiere induction.

**2. Capacidad insuficiente por cabeza.** Mini-LLaMA tiene 4 cabezas Q por capa con `d_k = d_model/h_q = 32`. Cada cabeza opera en un subespacio de 32 dimensiones. Las induction heads requieren codificar:
- Q: "que esta buscando esta posicion"
- K: "que info tiene este token sobre lo que vino antes"

A 32 dim/cabeza, las cabezas estan ocupadas con tareas mas basicas (previous-token, self-attention, atencion a separadores). No queda capacidad para el doble paso de induccion.

**3. Datos de entrenamiento no enfatizan repeticion.** El corpus de entrenamiento (Shakespeare) tiene patrones lingüisticos pero no patrones de repeticion estricta del estilo `XYZ XYZ`. El modelo no tiene presion evolutiva para desarrollar mecanismos de copia exacta — esos emergen mas claramente en corpus que enfatizan repeticion (codigo, tablas, secuencias estructuradas).

### Lo que esto significa

NO encontrar induction heads en Mini-LLaMA NO invalida la teoria de Anthropic. El descubrimiento original era sobre modelos de cierta escala minima — GPT-2 small ya es ~115M parametros, ~1500x mas grande que Mini-LLaMA (~890K). Las induction heads son una **propiedad emergente** que requiere escala suficiente.

Pedagogicamente, el resultado nulo es informativo:

- Confirma que la metrica esta bien implementada (los scores son discriminantes en cap 54)
- Demuestra que la interpretabilidad mecanicista REQUIERE escala apropiada
- Ilustra el principio "ausencia de evidencia no es evidencia de ausencia" — Mini-LLaMA quizas tiene proto-circuitos de induccion no detectables a esta escala

---

## 7. ¿Que tendriamos que hacer para encontrarlas?

Para detectar induction heads en un modelo construido en este curso:

**Opcion 1: escalar Mini-LLaMA.** Subir a 8-12 capas, d_model=256-512, entrenar mas tiempo. Probable que aparezcan induction parciales. Costo: dias de entrenamiento.

**Opcion 2: cargar GPT-2 small.** Usar transformers de Hugging Face para cargar GPT-2 (124M params) y aplicar las mismas tecnicas. Costo: depender de un modelo externo, romper la consistencia "build it yourself" del curso.

**Opcion 3: usar TransformerLens.** La libreria de Neel Nanda tiene Mini-models pre-entrenados para experimentos de mech interp. `solu-2l`, `attn-only-2l`, etc. son explicitamente disenados para mostrar induction. Costo: mismo que opcion 2.

Para el curso, la decision sensata es: documentar honestamente la limitacion en Mini-LLaMA y mencionar que las tecnicas SI funcionan en modelos de escala apropiada. Cap 63 va a abordar este punto cuando comparemos Mini-LLaMA con GPT-2 small.

---

## 8. Lo que SI vale el ejercicio

Aunque no encontramos induction heads, el cap valida varias cosas:

- **La metrica `induction_score` esta correctamente implementada**: el test del cap 5 confirmo que un patron sintetico de induccion da score alto. La metrica funciona.
- **El framework de busqueda funciona**: cachear activaciones, recomputar attn_weights, agregar sobre prompts. Este pipeline es el mismo que se usa en circuitos reales — solo necesita un modelo mas grande.
- **El resultado nulo es honesto**: muchos papers de interpretabilidad reportan resultados positivos. Mostrar UN resultado nulo en el contexto adecuado refuerza el rigor — no todo se encuentra en todos los modelos.

En el cap 57 haremos activation patching para verificar el efecto causal de las cabezas con scores altos en CAP 54 (previous-token). Y en cap 58 buscaremos circuitos en Mini-LLaMA SFT para una tarea concreta (la tarea "repeat" del cap 24). Ambos son experimentos donde Mini-LLaMA SI es lo suficientemente capaz para mostrar resultados positivos.

---

## 9. Preguntas de verificacion

**1. ¿Por que las induction heads dependen de las previous-token heads?**

Una induction head, al recibir el token `[A]` repetido en posicion `j`, necesita "saber" donde estaba `[A]` antes y especificamente que vino DESPUES de esa primera `[A]`. Para construir esa query, necesita features posicionales — informacion sobre "el token anterior a este fue X" o "este es el token que viene despues de Y". Esa informacion NO esta en las embeddings crudas — se construye via las previous-token heads de capas anteriores, que escriben al residual stream cosas como "soy la posicion despues de A" o "el token anterior a mi fue B". La induction head LEE esa info via su query y la matchea contra las keys del contexto. Por eso necesita capas previas que ya hayan resuelto previous-token; por eso emerge tipicamente en capas 2-6.

**2. ¿Si entrenara Mini-LLaMA por 100x mas iteraciones aparecerian induction heads?**

Posiblemente — pero probablemente no claramente. La escala (4 capas, 4 cabezas, d_model=128) es lo limitante mas que el numero de pasos de entrenamiento. Mas entrenamiento podria mejorar la calidad de las cabezas existentes (previous-token mas nitidas) pero el modelo sigue sin tener la "anchura" estructural para aprender el procesamiento de dos pasos que requiere induccion. Una analogia: aunque entrenes un perceptron con un millon de pasos, no aprende XOR — necesita una capa oculta. Aqui, mas profundidad o mas cabezas serian la "capa oculta" que falta.

**3. ¿Como sabemos que el resultado nulo no es un bug en la metrica?**

Tres validaciones:

a) El test unitario del cap 5 verifico la metrica con un patron sintetico de induccion: cuando la atencion concentra en la posicion correcta, el score es alto (>0.5). El codigo funciona.

b) Los scores de previous-token heads (cap 54) son **discriminantes**: capa 0 da scores bajos (~0.14), capas 1-3 dan scores altos (~0.40-0.55). Si el problema fuera de medicion, todas las cabezas darian scores similares. La discriminacion en cap 54 prueba que la metrica de atencion es sensible.

c) Los scores de induccion son **uniformemente bajos** (0.005-0.057) sin estructura. Si hubiera induction emergente parcial, esperariamos ver al menos una cabeza con score significativamente mas alto que las demas. La uniformidad sugiere ausencia de senal, no medicion defectuosa.
