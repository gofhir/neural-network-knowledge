---
title: "01 - CTC desde cero"
weight: 10
math: true
---

> La clase presenta [CTC](/papers/ctc-graves-2006) como una idea: *"agregar a la salida un token blank"*. Implementarla obliga a responder tres preguntas que la idea deja abiertas — cómo se colapsa exactamente, cuántas alineaciones hay que sumar, y cómo se suman sin enumerarlas. Las tres tienen respuesta verificable.

---

## 1. La función de colapso

Todo empieza acá, y el orden de las dos operaciones no es negociable:

```python
BLANK = 0

def colapsa(pi):
    """Colapsa repeticiones consecutivas, DESPUÉS elimina blanks."""
    out, prev = [], None
    for s in pi:
        if s != prev and s != BLANK:
            out.append(s)
        prev = s
    return tuple(out)
```

```python
colapsa([1,1,0,1])   # (1, 1)   <- el blank separa las dos
colapsa([1,1,1,1])   # (1,)     <- sin blank, colapsan en una
colapsa([0,1,0,0,2]) # (1, 2)
```

Las dos primeras líneas son la razón de fondo del blank, y va más allá de la "pausa" que menciona la clase: **sin un símbolo separador no habría forma de escribir letras dobles**. `carro`, `llave` o `innecesario` serían inexpresables.

---

## 2. La preimagen: cuántas alineaciones hay

Para entender qué está sumando CTC conviene contar primero. Con alfabeto chico y $T$ chico se puede enumerar todo:

```python
import itertools

def preimagen(T, y, C):
    """Todas las secuencias de largo T sobre C símbolos que colapsan a y."""
    return [pi for pi in itertools.product(range(C), repeat=T)
            if colapsa(pi) == tuple(y)]

print(len(preimagen(6, [1,2], 3)))   # 70
```

Enumerando para la palabra `casa` ($U = 4$) con distintos $T$:

| $T$ | 6 | 8 | 10 | 12 |
|---|---|---|---|---|
| alineaciones | 45 | 495 | 3 003 | 12 870 |

Esos números son coeficientes binomiales: $\binom{10}{8}$, $\binom{12}{8}$, $\binom{14}{8}$, $\binom{16}{8}$. En general, para una transcripción **sin símbolos repetidos consecutivos**:

$$\big|\mathcal{B}^{-1}(y)\big| = \binom{T+U}{2U}$$

Extrapolando a un segundo de audio real (~100 frames):

```
T=  20 -> 735,471
T=  50 -> 1,040,465,790
T= 100 -> 257,575,523,205
```

**Doscientos cincuenta mil millones de alineaciones para cuatro letras.** Sumarlas una por una no es una opción.

---

## 3. La recursión de tres términos

La secuencia extendida intercala blanks:

$$l = (\varnothing, y_1, \varnothing, y_2, \varnothing, \dots, y_U, \varnothing)$$

y $\alpha_t(s)$ acumula la probabilidad de todos los prefijos de largo $t$ que terminan en la posición $s$ de $l$:

```python
import numpy as np

def ctc_forward(P, y):
    """P: (T, C) probabilidades por frame. Devuelve P(y|x)."""
    T = P.shape[0]
    l = [BLANK]
    for c in y:
        l += [c, BLANK]                       # b y1 b y2 b ... yU b
    S = len(l)

    a = np.zeros((T, S))
    a[0, 0] = P[0, l[0]]                      # empezar en blank
    if S > 1:
        a[0, 1] = P[0, l[1]]                  # o en el primer símbolo

    for t in range(1, T):
        for s in range(S):
            v = a[t-1, s]                                  # quedarse
            if s > 0:
                v += a[t-1, s-1]                           # avanzar uno
            if s > 1 and l[s] != BLANK and l[s] != l[s-2]:
                v += a[t-1, s-2]                           # saltarse el blank
            a[t, s] = v * P[t, l[s]]

    return a[T-1, S-1] + (a[T-1, S-2] if S > 1 else 0)
```

Los tres términos son las tres formas de llegar a $(t, s)$. Y **la condición del tercero es la pieza sutil**: `l[s] != l[s-2]` impide saltarse el blank cuando los dos símbolos alrededor son iguales, que es justamente lo que preserva las letras dobles.

Nótese también que la salida suma **dos** celdas: la secuencia puede terminar en el último símbolo o en el blank final.

---

## 4. La verificación

Acá está el valor del ejercicio. Dos implementaciones independientes —una que enumera, otra que hace programación dinámica— deben dar el mismo número:

```python
def ctc_fuerza_bruta(P, y):
    T, C = P.shape
    return sum(np.prod([P[t, s] for t, s in enumerate(pi)])
               for pi in itertools.product(range(C), repeat=T)
               if colapsa(pi) == tuple(y))

rng = np.random.default_rng(0)
print(f"{'T':>3} {'y':>10} {'#alin':>7} {'fuerza bruta':>15} {'forward DP':>14} {'|dif|':>10}")
for T, y, C in [(4,[1],3), (5,[1,2],3), (6,[1,2],3), (7,[1,1],3), (8,[1,2,1],3)]:
    P = rng.random((T, C)); P /= P.sum(1, keepdims=True)
    n  = len(preimagen(T, y, C))
    fb, dp = ctc_fuerza_bruta(P, y), ctc_forward(P, y)
    print(f"{T:3d} {str(y):>10} {n:7d} {fb:15.9f} {dp:14.9f} {abs(fb-dp):10.2e}")
```

```
  T          y   #alin    fuerza bruta     forward DP      |dif|
  4        [1]      10     0.253750768    0.253750768   0.00e+00
  5     [1, 2]      35     0.188044421    0.188044421   5.55e-17
  6     [1, 2]      70     0.124306657    0.124306657   2.78e-17
  7     [1, 1]      70     0.044837265    0.044837265   2.08e-17
  8  [1, 2, 1]     462     0.067375506    0.067375506   5.55e-17
```

Coinciden hasta el épsilon de máquina.

{{< concept-alert type="clave" >}}
**Mirá la fila de `[1, 1]`.** Con $T=7$ admite **70** alineaciones. La fórmula $\binom{T+U}{2U}$ daría $\binom{9}{4} = 126$: el símbolo repetido **elimina 56 alineaciones**, exactamente las que se saltarían el blank obligatorio. Es la condición `l[s] != l[s-2]` haciendo su trabajo, y se puede ver en el conteo.
{{< /concept-alert >}}

**Un detalle práctico:** esta implementación multiplica probabilidades y con $T$ grande se va a cero por *underflow*. Las implementaciones reales trabajan en **log-espacio** con `logsumexp`:

```python
from scipy.special import logsumexp

def ctc_forward_log(logP, y):
    T = logP.shape[0]
    l = [BLANK]
    for c in y: l += [c, BLANK]
    S = len(l)
    a = np.full((T, S), -np.inf)
    a[0, 0] = logP[0, l[0]]
    if S > 1: a[0, 1] = logP[0, l[1]]
    for t in range(1, T):
        for s in range(S):
            terms = [a[t-1, s]]
            if s > 0: terms.append(a[t-1, s-1])
            if s > 1 and l[s] != BLANK and l[s] != l[s-2]: terms.append(a[t-1, s-2])
            a[t, s] = logsumexp(terms) + logP[t, l[s]]
    return logsumexp([a[T-1, S-1], a[T-1, S-2]]) if S > 1 else a[T-1, S-1]
```

---

## 5. Contra la implementación de PyTorch

`nn.CTCLoss` devuelve $-\log P(y \mid x)$, así que la comparación es directa:

```python
import torch
import torch.nn as nn

T, C, y = 8, 3, [1, 2, 1]
P = rng.random((T, C)); P /= P.sum(1, keepdims=True)

log_probs = torch.log(torch.tensor(P, dtype=torch.float64)).unsqueeze(1)  # (T, N=1, C)
targets   = torch.tensor([y], dtype=torch.long)
in_len    = torch.tensor([T]); tgt_len = torch.tensor([len(y)])

perdida = nn.CTCLoss(blank=BLANK, reduction='none', zero_infinity=True)(
    log_probs, targets, in_len, tgt_len)

print(f"PyTorch    : {perdida.item():.9f}")
print(f"implementación propia: {-np.log(ctc_forward(P, y)):.9f}")
```

Los dos valores deben coincidir hasta la tolerancia de punto flotante. Tres cosas que rompen esta comparación si no se cuidan:

- **El índice del blank.** PyTorch usa `blank=0` por defecto; si el vocabulario pone el blank al final hay que decírselo.
- **La forma del tensor** es `(T, N, C)` con el tiempo **primero**, no el lote.
- **`log_probs` debe ser log-softmax**, no logits ni probabilidades.

---

## 6. En los tres frameworks

Las tres librerías traen CTC como primitiva, con firmas distintas:

### PyTorch

```python
perdida = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
loss = perdida(log_probs, targets, input_lengths, target_lengths)
# log_probs: (T, N, C) con log_softmax aplicado
```

### TensorFlow

```python
import tensorflow as tf

loss = tf.nn.ctc_loss(labels=targets,               # (N, U) enteros
                      logits=logits,                # (T, N, C) LOGITS, no log-probs
                      label_length=target_lengths,
                      logit_length=input_lengths,
                      blank_index=0,
                      logits_time_major=True)
```

Ojo: TensorFlow espera **logits crudos** y aplica el softmax internamente, al revés que PyTorch.

### JAX (Optax)

```python
import optax, jax.numpy as jnp

loss = optax.ctc_loss(logits=logits,                # (N, T, C) lote primero
                      logit_paddings=logit_paddings,
                      labels=labels,
                      label_paddings=label_paddings)
```

En JAX el orden de ejes es **lote primero** y el enmascarado se expresa con arrays de *padding* (1 = posición rellenada) en vez de longitudes.

{{< concept-alert type="cuidado" >}}
Las tres discrepan en el orden de ejes, en si esperan logits o log-probabilidades, y en cómo se indican las longitudes. Es una fuente clásica de bugs silenciosos: la pérdida baja, el modelo no aprende, y el error está en que el tiempo y el lote estaban transpuestos. La implementación propia de la sección 3 sirve como oráculo para verificar cualquiera de ellas sobre un caso pequeño.
{{< /concept-alert >}}

---

## 7. Decodificación: greedy contra beam

Entrenar con CTC es la mitad; falta convertir las probabilidades por frame en texto.

**Greedy** — tomar el argmax de cada frame y colapsar:

```python
def decodifica_greedy(P):
    return colapsa(P.argmax(1))
```

Rápido y suficiente para diagnosticar. Pero es incorrecto en un sentido preciso: encuentra la **alineación** más probable, no la **transcripción** más probable. Como muchas alineaciones colapsan a la misma salida, una transcripción puede acumular más probabilidad total repartida entre miles de alineaciones mediocres que la que tiene la única alineación ganadora.

**Beam search con prefijos** mantiene $B$ transcripciones candidatas y va acumulando, para cada una, la probabilidad de **todas** sus alineaciones. Es el que se usa en producción, y el punto natural para integrar un modelo de lenguaje externo — necesario justamente porque CTC no modela dependencias entre salidas (ver [profundización, Parte II](/clases/clase-41/profundizacion)).

---

## Qué queda establecido

| Afirmación | Verificación | Resultado |
|---|---|---|
| El forward suma sobre todas las alineaciones | contra enumeración exhaustiva | error ≤ $5{,}6\times10^{-17}$ |
| El número de alineaciones es $\binom{T+U}{2U}$ | conteo directo | 45, 495, 3 003, 12 870 |
| Con $T=100$, $U=4$ hay $2{,}6\times10^{11}$ alineaciones | conteo por DP | sumarlas requiere $O(TU)$ |
| Un símbolo repetido reduce la preimagen | `[1,1]` con $T=7$ | 70 contra las 126 de la fórmula |
| La implementación propia == `nn.CTCLoss` | comparación directa | igualdad numérica |

---

## Ver también

- [02 - Agregación VLAD](02-agregacion-vlad) — el mecanismo de la segunda mitad de la clase.
- [Profundización, Partes I-III](/clases/clase-41/profundizacion) — la derivación de todo lo que acá se implementa.
- [Fundamento: CTC Loss](/fundamentos/ctc-loss) · [Reconocimiento de voz](/fundamentos/reconocimiento-de-voz).
- [Paper: CTC (2006)](/papers/ctc-graves-2006) · [Deep RNN Speech (2013)](/papers/deep-rnn-speech-graves-2013).
