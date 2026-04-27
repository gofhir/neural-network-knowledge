---
title: "Modelos: RNN, LSTM, BiLSTM"
weight: 20
math: true
---

El laboratorio implementa **tres arquitecturas recurrentes** comparables, todas con la misma interfaz: reciben una entrada one-hot por step, mantienen un estado oculto y emiten una distribución sobre las 18 categorías.

---

## 1. RNN vanilla (Elman cell)

### La idea: una red con memoria

Una red neuronal estándar es como un alumno con amnesia: cada vez que le muestras algo lo procesa **sin acordarse de lo anterior**. Para clasificar un apellido eso no sirve — necesitas que cuando lea la `'t'` final de `"Sato"`, recuerde que antes vio `'S'`, `'a'`, `'t'`, `'o'`.

Una **RNN** soluciona esto agregando una **memoria** llamada **estado oculto** (`hidden state`, $h$). Es un vector de números que se va actualizando letra por letra y que resume todo lo que la red vio hasta ese punto:

```
Step 1: lee 'S' →  h_1 = [0.2, -0.5, 0.8, ...]   ← memoria después de S
Step 2: lee 'a' →  h_2 = [0.4, -0.1, 0.3, ...]   ← memoria después de Sa
Step 3: lee 't' →  h_3 = [0.7,  0.2, 0.6, ...]   ← memoria después de Sat
Step 4: lee 'o' →  h_4 = [0.9,  0.1, 0.4, ...]   ← memoria después de Sato
                                                   ↑ esta memoria se usa para clasificar
```

Esa es **la única idea**. Todo lo demás implementa exactamente esto.

### La fórmula de Elman

$$h_t = \tanh(W_{xh} \, x_t + W_{hh} \, h_{t-1} + b)$$
$$y_t = W_{ho} \, h_t + b_o$$

Tres transformaciones lineales:

| Matriz | Qué transforma | Tamaños (lab, $H$=147) |
|--------|----------------|------------------------|
| $W_{xh}$ (`x2h`) | letra one-hot → contribución al estado oculto | `Linear(57, 147)` |
| $W_{hh}$ (`h2h`) | memoria previa → contribución al nuevo estado | `Linear(147, 147)` |
| $W_{ho}$ (`h2o`) | estado oculto → predicción de clase | `Linear(147, 18)` |

**¿Por qué `tanh`?** Sin función no lineal, la red colapsaría a una única transformación lineal (composición de lineales = lineal). `tanh` aplasta la salida al rango $[-1, 1]$, lo que mantiene el estado oculto **acotado** y evita que crezca sin control en cada paso temporal.

### Diagrama de un timestep

```
       x_t (letra one-hot, 57)              h_{t-1} (memoria previa, 147)
            │                                       │
            ▼                                       ▼
       x2h: Linear(57, 147)               h2h: Linear(147, 147)
            │                                       │
            └──────────────► (+) ◄──────────────────┘
                              │
                              ▼
                            tanh
                              │
                              ▼
                        h_t (memoria nueva, 147) ──────────► (al siguiente step)
                              │
                              ▼
                       h2o: Linear(147, 18)
                              │
                              ▼
                        y_t (logits, 18)
```

### Implementación

La separación en `RNNCell` (un paso temporal) + `RNN` (wrapper que inicializa el estado y expone el `forward`) es **didáctica**: la celda contiene la fórmula matemática pura, el wrapper la infraestructura. En código de producción usarías directamente la API de alto nivel (`nn.RNN`, `tf.keras.layers.SimpleRNN`, `flax.linen.SimpleCell`).

{{< tabs >}}
{{< tab name="PyTorch" >}}
```python
import torch
import torch.nn as nn

class RNNCell(nn.Module):
    """Elman RNN cell — un paso temporal."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.x2h = nn.Linear(input_size, hidden_size)   # W_xh
        self.h2h = nn.Linear(hidden_size, hidden_size)  # W_hh
        self.h2o = nn.Linear(hidden_size, output_size)  # W_ho

    def forward(self, x, old_hidden):
        hidden = torch.tanh(self.x2h(x) + self.h2h(old_hidden))
        output = self.h2o(hidden)
        return output, hidden


class RNN(nn.Module):
    """Wrapper: inicializa el estado oculto y delega al cell."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.bidirectional = False
        self.hidden_size = hidden_size
        self.cell = RNNCell(input_size, hidden_size, output_size)

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

    def forward(self, x, hidden):
        return self.cell(x, hidden)
```
{{< /tab >}}

{{< tab name="TensorFlow" >}}
```python
import tensorflow as tf

class RNNCell(tf.keras.Model):
    """Elman RNN cell — un paso temporal."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.x2h = tf.keras.layers.Dense(hidden_size)   # W_xh
        self.h2h = tf.keras.layers.Dense(hidden_size)   # W_hh
        self.h2o = tf.keras.layers.Dense(output_size)   # W_ho

    def call(self, x, old_hidden):
        hidden = tf.tanh(self.x2h(x) + self.h2h(old_hidden))
        output = self.h2o(hidden)
        return output, hidden


class RNN(tf.keras.Model):
    """Wrapper: inicializa el estado oculto y delega al cell."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.bidirectional = False
        self.hidden_size = hidden_size
        self.cell = RNNCell(input_size, hidden_size, output_size)

    def init_hidden(self):
        return tf.zeros((1, self.hidden_size))

    def call(self, x, hidden):
        return self.cell(x, hidden)
```
{{< /tab >}}

{{< tab name="JAX" >}}
```python
import jax
import jax.numpy as jnp
from flax import linen as nn

class RNNCell(nn.Module):
    """Elman RNN cell — un paso temporal."""
    hidden_size: int
    output_size: int

    @nn.compact
    def __call__(self, x, old_hidden):
        h_x = nn.Dense(self.hidden_size, name='x2h')(x)
        h_h = nn.Dense(self.hidden_size, name='h2h')(old_hidden)
        hidden = jnp.tanh(h_x + h_h)
        output = nn.Dense(self.output_size, name='h2o')(hidden)
        return output, hidden


class RNN(nn.Module):
    """Wrapper: inicializa el estado oculto y delega al cell."""
    hidden_size: int
    output_size: int
    bidirectional: bool = False

    def setup(self):
        self.cell = RNNCell(hidden_size=self.hidden_size,
                            output_size=self.output_size)

    def init_hidden(self):
        return jnp.zeros((1, self.hidden_size))

    def __call__(self, x, hidden):
        return self.cell(x, hidden)
```

> **Nota JAX/Flax:** los módulos son **stateless** — el `hidden` viaja siempre como argumento explícito (no se guarda en `self`). Esto es lo que permite a JAX hacer `jit`/`vmap`/`grad` sobre el modelo entero.
{{< /tab >}}
{{< /tabs >}}

### Análisis del forward, línea por línea

```python
hidden = torch.tanh(self.x2h(input) + self.h2h(old_hidden))
                    └─ contribución ─┘   └─ contribución ─┘
                       de la letra        de la memoria previa
```

1. **`self.x2h(input)`** → toma la letra (vector one-hot de 57) y la transforma en un vector de 147 que representa **qué aporta esa letra**.
2. **`self.h2h(old_hidden)`** → toma la memoria previa (147) y la transforma en otro vector de 147 que representa **qué se conserva del pasado**.
3. **Suma + `tanh`** → fusiona ambas contribuciones en la nueva memoria, acotada a $[-1, 1]$.

```python
output = self.h2o(hidden)
```

4. Proyecta la memoria nueva (147) a **18 logits** (uno por idioma). El loop de entrenamiento **descarta los outputs intermedios** y solo usa el del último carácter para calcular el loss.

```python
return output, hidden
```

5. Devuelve ambos: `output` para la predicción, `hidden` para alimentar el siguiente step.

### Por qué inicializar `hidden` en ceros

```python
def init_hidden(self):
    return torch.zeros(1, self.hidden_size)
```

Antes de leer ninguna letra, la red **no tiene memoria de nada** — el vector arranca en cero. En el step 1, ese vector de ceros pasa por `h2h` (que también lo deja en cero salvo por el bias) y la red queda dependiendo solo de `x2h(primera_letra) + bias`. A partir del step 2, la memoria empieza a acumular información real.

El `1` es la dimensión de batch (siempre 1 en este lab — un apellido a la vez).

### Conteo de parámetros (input=57, hidden=H, output=18)

| Capa | Pesos | Bias | Total |
|------|-------|------|-------|
| `x2h` | $57 \cdot H$ | $H$ | $58H$ |
| `h2h` | $H \cdot H$ | $H$ | $H^2 + H$ |
| `h2o` | $H \cdot 18$ | $18$ | $18H + 18$ |

Para `H=147`: $58(147) + 147^2 + 147 + 18(147) + 18 = 8\,526 + 21\,609 + 147 + 2\,646 + 18 = $ **~32 946 parámetros**.

---

## 2. LSTM (unidireccional)

### El problema que LSTM resuelve

La RNN vanilla tiene un defecto **fundamental**: durante el backpropagation through time, el gradiente de la última pérdida tiene que viajar hacia atrás por toda la secuencia, y en cada paso se multiplica por una matriz $W_{hh}^T \cdot \text{diag}(\tanh'(h))$. Si el mayor valor singular de esa matriz es menor que 1, **el gradiente se contrae exponencialmente**:

$$\left\| \frac{\partial L}{\partial h_t} \right\| \approx \alpha^{T-t} \cdot \left\| \frac{\partial L}{\partial h_T} \right\|, \quad \text{con } \alpha < 1$$

A los 10–20 timesteps, el gradiente es prácticamente cero. La red **no puede aprender dependencias largas** — para clasificar `"Schwarzenegger"`, la información de la `'S'` inicial nunca llega de vuelta a actualizar los pesos cuando la pérdida se calcula al final.

Este es el famoso **problema del vanishing gradient**, identificado por Hochreiter (1991) y Bengio et al. (1994). Las RNN vanilla son matemáticamente elegantes pero prácticamente inútiles para secuencias largas.

### La idea de Hochreiter & Schmidhuber (1997): una autopista para el gradiente

> *"The cell state is kind of like a conveyor belt. It runs straight down the entire chain, with only some minor linear interactions. It's very easy for information to just flow along it unchanged."* — [Christopher Olah, *Understanding LSTMs* (2015)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

LSTM agrega un **segundo canal de información** que viaja en paralelo al estado oculto $h_t$: el **estado celular** $c_t$ ("cell state"). Mientras $h_t$ es lo que la red emite hacia afuera, $c_t$ es la **memoria de largo plazo interna** que viaja a través del tiempo con operaciones casi lineales:

$$c_t = f_t \odot c_{t-1} + i_t \odot g_t$$

Sólo dos operaciones: una multiplicación elemento a elemento ($\odot$) y una suma. **Sin** activaciones no lineales aplastando el gradiente. Por eso $\partial c_t / \partial c_{t-1} \approx f_t$ — un factor cercano a 1 si la compuerta de olvido decide "mantener". El gradiente puede viajar 100 timesteps sin desaparecer.

### Las 4 compuertas: qué hace cada una

LSTM no reemplaza el `tanh` con otro `tanh`. Reemplaza la operación entera con **4 mini-redes** que aprenden a controlar el flujo de información. Cada compuerta es una `Linear + sigmoid` (o `tanh` para el candidato), y todas reciben los mismos inputs: $x_t$ (entrada actual) y $h_{t-1}$ (memoria emitida en el step anterior).

| Compuerta | Símbolo | Activación | Función |
|-----------|---------|------------|---------|
| **Forget gate** | $f_t$ | sigmoid → $[0,1]$ | "¿Qué borro de la memoria de largo plazo?" |
| **Input gate** | $i_t$ | sigmoid → $[0,1]$ | "¿Qué tan importante es lo nuevo que voy a agregar?" |
| **Candidate cell** | $g_t$ (también $\tilde{c}_t$) | tanh → $[-1,1]$ | "¿Qué información nueva propongo agregar?" |
| **Output gate** | $o_t$ | sigmoid → $[0,1]$ | "¿Qué parte de la memoria muestro hacia afuera?" |

Las **sigmoides funcionan como llaves de paso** (un valor cercano a 0 cierra, cercano a 1 abre). El **tanh del candidato** propone valores positivos y negativos (puede sumar o restar a la memoria).

### Las ecuaciones completas

$$\begin{aligned}
f_t &= \sigma(W_{xf} x_t + W_{hf} h_{t-1} + b_f) \quad &\text{(qué olvidar)} \\
i_t &= \sigma(W_{xi} x_t + W_{hi} h_{t-1} + b_i) \quad &\text{(cuánto del candidato)} \\
g_t &= \tanh(W_{xg} x_t + W_{hg} h_{t-1} + b_g) \quad &\text{(candidato)} \\
o_t &= \sigma(W_{xo} x_t + W_{ho} h_{t-1} + b_o) \quad &\text{(qué emitir)} \\
\\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t \quad &\text{(actualizar la cinta)} \\
h_t &= o_t \odot \tanh(c_t) \quad &\text{(emitir hacia afuera)}
\end{aligned}$$

**Lectura paso a paso:**

1. **Decide qué olvidar.** $f_t$ mira lo nuevo ($x_t$) y la memoria emitida ($h_{t-1}$) y produce un vector con valores entre 0 y 1. Donde es 0, se borra ese slot de $c_{t-1}$. Donde es 1, se conserva intacto.
   > *"A 1 represents 'completely keep this' while a 0 represents 'completely get rid of this.'"* — Olah

2. **Decide qué agregar.** En paralelo, $g_t$ propone un vector candidato (qué info nueva entra), e $i_t$ decide cuánto de ese candidato dejar pasar (compuerta multiplicativa).

3. **Actualiza la cinta.** $c_t = f_t \odot c_{t-1} + i_t \odot g_t$ — borra lo que decidió olvidar y suma lo nuevo (escalado por la importancia).

4. **Emite hacia afuera.** $h_t = o_t \odot \tanh(c_t)$ — pasa la cinta por `tanh` (acota a $[-1,1]$) y la filtra con $o_t$ (decide qué partes hacer visibles al siguiente módulo y al output).

### Por qué LSTM resuelve vanishing gradient (la prueba rápida)

Derivando $c_t = f_t \odot c_{t-1} + i_t \odot g_t$ con respecto a $c_{t-1}$:

$$\frac{\partial c_t}{\partial c_{t-1}} = f_t$$

Si $f_t \approx 1$ (la red aprendió que esa información es importante), el gradiente atraviesa el timestep **sin atenuarse**. Multiplicar por 1 muchas veces no contrae nada. Comparado con la RNN vanilla donde $\partial h_t / \partial h_{t-1}$ depende de $W_{hh}^T \cdot \tanh'$ — un producto siempre menor que 1 que decae exponencialmente.

> Esta es la razón profunda por la que LSTM destrona a la RNN vanilla en cualquier tarea con dependencias largas. No es "una mejora incremental" — es un **cambio arquitectónico** que arregla un defecto matemático.

### Diagrama del flujo completo

```
                             c_{t-1} ───────────►(×)──────►(+)─────────► c_t (a t+1)
                                                  ▲         ▲                │
                                                  │         │                │
                                                  │         │              tanh
                                                  │         │                │
                                                  │       (×)               (×)◄── o_t
                                                  │      ▲   ▲               │
                                                  f_t   i_t  g_t             ▼
                                                  ▲      ▲   ▲              h_t (a t+1 y a output)
                                                  │      │   │
                                                  σ      σ  tanh
                                                  ▲      ▲   ▲
                                                  └──────┴───┴───── concat(x_t, h_{t-1})
```

Las 4 compuertas (en la base) se computan en paralelo desde `concat(x_t, h_{t-1})`. Luego $f_t$ borra de la cinta, $i_t \odot g_t$ agrega a la cinta, y $o_t$ filtra qué se muestra. El **estado celular $c_t$ es la línea horizontal de arriba** — la "autopista" que viaja sin obstrucciones graves.

### El estado oculto es ahora una **tupla** `(h, c)`

A diferencia de la RNN vanilla que mantenía un solo vector, LSTM mantiene **dos vectores** que viajan juntos:

- `h` (hidden state): lo que la celda **emite** hacia afuera y al siguiente módulo. Es lo que se usa para clasificar.
- `c` (cell state): la **memoria interna** que viaja por la "autopista". Nunca se muestra directamente, sólo se filtra a través de $o_t \odot \tanh(c_t)$ para producir $h$.

Por eso `init_hidden()` devuelve dos tensores (en el lab los dos en cero al inicio), y `nn.LSTMCell` consume y devuelve la tupla `(h, c)`.

### Implementación

El lab usa la API de alto nivel (`nn.LSTMCell`, `tf.keras.layers.LSTMCell`, `flax.linen.OptimizedLSTMCell`) que internamente computa las 4 compuertas. Se podría implementar manualmente con 4 capas `Linear`, pero las versiones del framework están optimizadas (compuertas computadas como una sola multiplicación de matriz grande, kernels CUDA fusionados, etc.).

{{< tabs >}}
{{< tab name="PyTorch" >}}
```python
import torch
import torch.nn as nn

class LSTM(nn.Module):
    """LSTM unidireccional para clasificacion de secuencias."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.bidirectional = False
        self.hidden_size = hidden_size
        self.cell = nn.LSTMCell(input_size, hidden_size)   # 4 compuertas internas
        self.h2o = nn.Linear(hidden_size, output_size)

    def init_hidden(self):
        # Tupla (h_0, c_0): hidden state + cell state, ambos en cero
        return (torch.zeros(1, self.hidden_size),
                torch.zeros(1, self.hidden_size))

    def forward(self, x, hidden):
        # hidden = (h_t, c_t) — la celda recibe y devuelve la tupla
        hidden = self.cell(x, hidden)
        output = self.h2o(hidden[0])    # clasifica desde h_t (no desde c_t)
        return output, hidden
```
{{< /tab >}}

{{< tab name="TensorFlow" >}}
```python
import tensorflow as tf

class LSTM(tf.keras.Model):
    """LSTM unidireccional para clasificacion de secuencias."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.bidirectional = False
        self.hidden_size = hidden_size
        self.cell = tf.keras.layers.LSTMCell(hidden_size)   # 4 compuertas internas
        self.h2o = tf.keras.layers.Dense(output_size)

    def init_hidden(self):
        # Lista [h_0, c_0]: convencion de Keras es lista, no tupla
        return [tf.zeros((1, self.hidden_size)),
                tf.zeros((1, self.hidden_size))]

    def call(self, x, hidden):
        # LSTMCell.call(inputs, states) -> (output, new_states)
        # output == new_states[0] (h_t); new_states = [h_t, c_t]
        _, new_states = self.cell(x, states=hidden)
        output = self.h2o(new_states[0])
        return output, new_states
```

> **Nota TF/Keras:** las APIs `tf.keras.layers.LSTMCell` esperan `states` como una **lista** `[h, c]` (no tupla). El `output` que devuelve la celda es exactamente `new_states[0]` (es decir $h_t$) — Keras lo expone duplicado por compatibilidad con la API de capas Dense.
{{< /tab >}}

{{< tab name="JAX" >}}
```python
import jax
import jax.numpy as jnp
from flax import linen as nn

class LSTM(nn.Module):
    """LSTM unidireccional para clasificacion de secuencias."""
    hidden_size: int
    output_size: int
    bidirectional: bool = False

    def setup(self):
        self.cell = nn.OptimizedLSTMCell(features=self.hidden_size)
        self.h2o = nn.Dense(self.output_size)

    def init_hidden(self, rng):
        # Flax requiere PRNGKey + input_shape para inicializar el estado
        # Devuelve tupla (c_0, h_0) — convencion Flax: cell state primero
        return self.cell.initialize_carry(rng, (1, self.hidden_size))

    def __call__(self, x, hidden):
        # cell(carry, x) -> (new_carry, output)
        # carry == (c_t, h_t); output == h_t
        new_hidden, h_t = self.cell(hidden, x)
        output = self.h2o(h_t)
        return output, new_hidden
```

> **Nota JAX/Flax:** Flax invierte la convención de orden — el "carry" es `(c, h)` con cell state **primero**, opuesto a PyTorch que usa `(h, c)`. Además el método se llama `initialize_carry` y requiere una `PRNGKey` (JAX no tiene aleatoriedad implícita; toda fuente de aleatoriedad es explícita).
{{< /tab >}}
{{< /tabs >}}

### Análisis del forward de LSTM (PyTorch, línea por línea)

```python
hidden = self.cell(x, hidden)
```

Esta única línea ejecuta **internamente** las 6 ecuaciones del LSTM:

1. Computa $f_t, i_t, g_t, o_t$ desde $x_t$ y $h_{t-1}$ (4 multiplicaciones lineales + activaciones)
2. Actualiza $c_t = f_t \odot c_{t-1} + i_t \odot g_t$
3. Computa $h_t = o_t \odot \tanh(c_t)$
4. Devuelve la nueva tupla $(h_t, c_t)$

`nn.LSTMCell` lo hace de forma optimizada: las 4 transformaciones lineales se computan como **una sola matmul** con una matriz `(4H, input+H)`, luego se separa el resultado en cuartos. Más rápido que 4 matmuls separadas.

```python
output = self.h2o(hidden[0])
```

Toma sólo `h_t` (el primer elemento de la tupla, el estado oculto emitido) y lo proyecta a 18 logits. **No usa $c_t$ directamente** — la cell state es memoria interna, no se expone al clasificador.

### Conteo de parámetros: por qué LSTM tiene ~4× los parámetros de una RNN al mismo $H$

`nn.LSTMCell(input_size=57, hidden_size=H)` mantiene **4 conjuntos** de pesos (uno por compuerta), que internamente PyTorch fusiona en dos matrices grandes:

- `weight_ih`: shape `(4H, 57)` → $4 \cdot 57 \cdot H = 228H$ pesos
- `weight_hh`: shape `(4H, H)` → $4 \cdot H \cdot H = 4H^2$ pesos
- `bias_ih`: shape `(4H,)` → $4H$
- `bias_hh`: shape `(4H,)` → $4H$ (PyTorch tiene 2 bias por convención CuDNN)

**Total LSTMCell:** $4 \cdot (57H + H^2 + 2H)$

Más la capa de salida `h2o = Linear(H, 18)`: $18H + 18$.

**Para `H=64`:**
$$4(57 \cdot 64 + 64^2 + 2 \cdot 64) + 18 \cdot 64 + 18$$
$$= 4(3\,648 + 4\,096 + 128) + 1\,152 + 18$$
$$= 4 \cdot 7\,872 + 1\,170 = 31\,488 + 1\,170 = \mathbf{32\,658 \text{ parámetros}}$$

**Casi idéntico a la RNN vanilla con $H=147$** (~32 946). La elección de $H=64$ para LSTM no es arbitraria — los autores del lab calcularon $H$ tal que LSTM tuviera el mismo orden de parámetros que la RNN vanilla, para que la **comparación entre arquitecturas sea justa** (ver Actividad 1).

---

## 3. BiLSTM

### El problema que BiLSTM resuelve

La RNN y la LSTM unidireccionales tienen una limitación que no tiene nada que ver con el vanishing gradient: **sólo pueden mirar hacia atrás**. Cuando procesan el carácter 'a' en `"Sato"`, la red sólo ha visto `'S', 'a'` — no sabe que vienen `'t', 'o'` después. Para muchas tareas eso es problemático:

- **Clasificación de apellidos:** el sufijo es a menudo lo más informativo (`-ov` ruso, `-ez` español, `-ic` croata, `-ski` polaco). Una LSTM unidireccional tiene que esperar hasta el final para "ver" el sufijo, y entonces ya cargó con toda la representación intermedia que quizás no conviene.
- **POS tagging / NER:** para etiquetar la palabra del medio en *"the bank approved the loan"*, te ayuda mirar tanto *"the"* (a la izquierda) como *"approved"* (a la derecha) — la palabra siguiente desambigua el sentido de "bank".
- **Comprensión de lectura:** entender el rol de un token requiere contexto antes y después.

> **Insight clave:** mientras tengas la **secuencia completa disponible al momento de la inferencia** (como aquí: el apellido entero), no hay razón para limitarte a leer en una sola dirección.

### La idea de Schuster & Paliwal (1997)

Una **BiLSTM** corre **dos LSTM independientes en paralelo**:

- **LSTM forward** $\overrightarrow{\text{LSTM}}$: lee la secuencia de izquierda a derecha. En el step $t$ produce $\overrightarrow{h}_t$ que resume *"todo lo visto hasta el carácter $t$"*.
- **LSTM backward** $\overleftarrow{\text{LSTM}}$: lee la secuencia de derecha a izquierda. En el step $t$ produce $\overleftarrow{h}_t$ que resume *"todo lo visto desde el carácter $T$ hasta el $t$ (en sentido inverso)"*.

En cada timestep, los dos hidden states se **concatenan** para formar una representación enriquecida con contexto **bidireccional**:

$$h_t^{\text{bi}} = [\overrightarrow{h}_t \, ; \, \overleftarrow{h}_t] \in \mathbb{R}^{2H}$$

Para clasificación de secuencia (many-to-one), se usa la concatenación al **final del recorrido** de cada dirección:

- $\overrightarrow{h}_T$: la forward LSTM ya leyó todo, su estado final resume el apellido completo de izq a der.
- $\overleftarrow{h}_T$: la backward LSTM ya recorrió todo en reversa, su estado final resume el apellido de der a izq.

### Por qué esto ayuda concretamente

Considera `"Schwarzenegger"` clasificado como **alemán**. La forward LSTM al final tiene la `'r'` final fresca y los caracteres iniciales borrosos por la cantidad de pasos. La backward LSTM al final tiene la `'S'` inicial fresca y los caracteres finales borrosos. **Concatenarlas** da una representación donde **ambos extremos están bien representados**, y la red puede aprender que `"Sch"` al inicio + `"egger"` al final son señales fuertes de alemán.

### Cuándo NO usar BiLSTM

BiLSTM **no funciona** para tareas online/streaming donde no tienes el futuro al momento de predecir:

- Reconocimiento de voz en vivo (necesitas predecir mientras llega el audio)
- Generación autoregresiva de texto (estás creando el futuro, no lo conoces)
- Trading algorítmico con datos en tiempo real

Para esos casos quedas con LSTM unidireccional o arquitecturas causales (Transformers con máscara causal).

### Las ecuaciones

Definiendo dos LSTM independientes (cada una con sus propios pesos):

$$\overrightarrow{h}_t, \overrightarrow{c}_t = \overrightarrow{\text{LSTM}}(x_t, \overrightarrow{h}_{t-1}, \overrightarrow{c}_{t-1})$$
$$\overleftarrow{h}_t, \overleftarrow{c}_t = \overleftarrow{\text{LSTM}}(x_{T-t+1}, \overleftarrow{h}_{t-1}, \overleftarrow{c}_{t-1})$$
$$y = W_{ho} \, [\overrightarrow{h}_T \, ; \, \overleftarrow{h}_T] + b_o$$

Nota que **cada LSTM tiene 4 compuertas con sus propios pesos** — no hay nada compartido entre forward y backward. Por eso el conteo de parámetros se duplica respecto a LSTM unidireccional (ver más abajo).

### Diagrama del flujo

```
              x_1   x_2   x_3   x_4   x_5    (secuencia: S, a, t, o, !)
               │     │     │     │     │
   forward:    ▼     ▼     ▼     ▼     ▼
              [→] → [→] → [→] → [→] → [→]    h_fwd_5 ◄── contexto izq → der completo
               h     h     h     h     h
               c     c     c     c     c
                                      ▲
                                      │
              x_5   x_4   x_3   x_2   x_1    (secuencia invertida: !, o, t, a, S)
               │     │     │     │     │
  backward:    ▼     ▼     ▼     ▼     ▼
              [←] → [←] → [←] → [←] → [←]    h_bwd_5 ◄── contexto der → izq completo
               h     h     h     h     h
               c     c     c     c     c

                          ┌──────────────┐
                          │  CONCAT 2H   │ = [h_fwd_5 ; h_bwd_5]
                          └──────┬───────┘
                                 │
                                 ▼
                            Linear(2H, 18)
                                 │
                                 ▼
                          logits (18 clases)
```

### El truco didáctico del lab: pasar (forward, backward) al `forward()`

La forma "natural" de implementar BiLSTM sería:

```python
# Pseudocódigo idiomático (no es lo del lab)
h_fwd_final, _ = forward_lstm(secuencia)
h_bwd_final, _ = backward_lstm(reversed(secuencia))
output = h2o(concat([h_fwd_final, h_bwd_final]))
```

Pero el lab adopta una convención **diferente** para integrar bien con el loop de entrenamiento existente: en cada iteración del loop, le pasa al `forward` una **tupla de dos caracteres** — el i-ésimo desde el inicio y el i-ésimo desde el final:

```python
# En el loop de entrenamiento (entrenamiento.md):
for i in range(line_tensor.size()[0]):
    if model.bidirectional:
        output, hidden = model(
            (line_tensor[i],                           # carácter i desde el inicio
             line_tensor[line_tensor.size()[0]-i-1]),  # carácter i desde el final
            hidden,
        )
```

Así, en el step $i=0$ la forward LSTM ve `'S'` y la backward LSTM ve `'!'` (último). En el step $i=1$ la forward ve `'a'` y la backward ve `'o'` (penúltimo). Y así sucesivamente. **Cuando el loop termina, la backward LSTM ha procesado el apellido completo en reversa.**

Esto permite **reusar el mismo loop** que las RNN/LSTM unidireccionales — sólo cambia la firma del input. Es ingenioso pedagógicamente pero atípico en producción.

### Implementación

{{< tabs >}}
{{< tab name="PyTorch" >}}
```python
import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    """LSTM bidireccional — dos LSTMs paralelas en sentidos opuestos."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.bidirectional = True
        self.hidden_size = hidden_size
        self.forward_cell  = nn.LSTMCell(input_size, hidden_size)
        self.backward_cell = nn.LSTMCell(input_size, hidden_size)
        self.h2o = nn.Linear(2 * hidden_size, output_size)   # input doble: concat fwd+bwd

    def init_hidden(self):
        # ((h_fwd_0, c_fwd_0), (h_bwd_0, c_bwd_0)) — todos en cero
        return ((torch.zeros(1, self.hidden_size), torch.zeros(1, self.hidden_size)),
                (torch.zeros(1, self.hidden_size), torch.zeros(1, self.hidden_size)))

    def forward(self, x, hidden):
        # x = (carac_forward, carac_backward) — el loop entrega las dos direcciones
        new_fwd = self.forward_cell(x[0], hidden[0])
        new_bwd = self.backward_cell(x[1], hidden[1])

        h_fwd = new_fwd[0]   # estado oculto forward (descarta cell state)
        h_bwd = new_bwd[0]   # estado oculto backward

        output = self.h2o(torch.cat((h_fwd, h_bwd), dim=1))
        return output, (new_fwd, new_bwd)
```

> **Alternativa idiomática en PyTorch:** `nn.LSTM(input_size, hidden_size, bidirectional=True)` ya implementa BiLSTM internamente y es más eficiente (kernels CUDA optimizados). El lab usa la versión manual para que veas explícitamente las dos celdas.
{{< /tab >}}

{{< tab name="TensorFlow" >}}
```python
import tensorflow as tf

class BiLSTM(tf.keras.Model):
    """LSTM bidireccional — dos LSTMs paralelas en sentidos opuestos."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.bidirectional = True
        self.hidden_size = hidden_size
        self.forward_cell  = tf.keras.layers.LSTMCell(hidden_size)
        self.backward_cell = tf.keras.layers.LSTMCell(hidden_size)
        self.h2o = tf.keras.layers.Dense(output_size)   # input doble: concat fwd+bwd

    def init_hidden(self):
        return ([tf.zeros((1, self.hidden_size)), tf.zeros((1, self.hidden_size))],
                [tf.zeros((1, self.hidden_size)), tf.zeros((1, self.hidden_size))])

    def call(self, x, hidden):
        # x = (carac_forward, carac_backward)
        _, new_fwd = self.forward_cell(x[0], states=hidden[0])
        _, new_bwd = self.backward_cell(x[1], states=hidden[1])

        h_fwd = new_fwd[0]
        h_bwd = new_bwd[0]

        output = self.h2o(tf.concat([h_fwd, h_bwd], axis=1))
        return output, (new_fwd, new_bwd)
```

> **Alternativa idiomática en TF/Keras:** `tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size))` envuelve cualquier capa recurrente y la convierte en bidireccional automáticamente. Es la forma estándar de uso en producción.
{{< /tab >}}

{{< tab name="JAX" >}}
```python
import jax
import jax.numpy as jnp
from flax import linen as nn

class BiLSTM(nn.Module):
    """LSTM bidireccional — dos LSTMs paralelas en sentidos opuestos."""
    hidden_size: int
    output_size: int
    bidirectional: bool = True

    def setup(self):
        self.forward_cell  = nn.OptimizedLSTMCell(features=self.hidden_size)
        self.backward_cell = nn.OptimizedLSTMCell(features=self.hidden_size)
        self.h2o = nn.Dense(self.output_size)

    def init_hidden(self, rng):
        # Flax: cada cell expone initialize_carry; necesitamos 2 RNGs (uno por dirección)
        rng_fwd, rng_bwd = jax.random.split(rng)
        carry_fwd = self.forward_cell.initialize_carry(rng_fwd, (1, self.hidden_size))
        carry_bwd = self.backward_cell.initialize_carry(rng_bwd, (1, self.hidden_size))
        return (carry_fwd, carry_bwd)

    def __call__(self, x, hidden):
        # x = (carac_forward, carac_backward)
        new_carry_fwd, h_fwd = self.forward_cell(hidden[0], x[0])
        new_carry_bwd, h_bwd = self.backward_cell(hidden[1], x[1])

        output = self.h2o(jnp.concatenate([h_fwd, h_bwd], axis=1))
        return output, (new_carry_fwd, new_carry_bwd)
```

> **Alternativa idiomática en JAX/Flax:** `nn.Bidirectional(forward_rnn=..., backward_rnn=...)` envuelve dos `nn.RNN` en una sola capa que recorre la secuencia completa en ambos sentidos. Es el equivalente al wrapper de Keras.
{{< /tab >}}
{{< /tabs >}}

### Análisis del forward, línea por línea

```python
new_fwd = self.forward_cell(x[0], hidden[0])
```

La LSTM forward procesa el i-ésimo carácter **desde el inicio** (`x[0]`) y actualiza su tupla `(h_fwd, c_fwd)`. Es exactamente una llamada a LSTMCell — todo lo visto en la sección anterior aplica idéntico.

```python
new_bwd = self.backward_cell(x[1], hidden[1])
```

La LSTM backward procesa el i-ésimo carácter **desde el final** (`x[1]`). En el step $i=0$ esto es el último carácter del apellido; en el step $i=1$ el penúltimo; etc. **Es una segunda LSTM completamente independiente** — pesos propios, estado oculto propio.

```python
output = self.h2o(torch.cat((h_fwd, h_bwd), dim=1))
```

Concatena los dos hidden states (cada uno `(1, H)`) en un único vector `(1, 2H)` y lo proyecta a 18 logits. La capa de salida tiene **input doble** (`2H`) precisamente porque recibe la concatenación.

> **Nota sutil:** este `output` se computa en cada step del loop, pero **sólo el del último step se usa para el loss**. En ese step final, `h_fwd` resume todo el apellido leído de izq a der, y `h_bwd` resume todo el apellido leído de der a izq — la información completa de ambos sentidos.

### Conteo de parámetros: por qué BiLSTM tiene ~2× los parámetros de LSTM al mismo $H$

Tres componentes:

- **2× `nn.LSTMCell(57, H)`** — dos celdas LSTM completamente independientes, no comparten nada:
$$2 \cdot 4 \cdot (57H + H^2 + 2H) = 8 \cdot (57H + H^2 + 2H)$$

- **`h2o = Linear(2H, 18)`** — la capa de salida tiene input `2H` por la concatenación:
$$18 \cdot 2H + 18 = 36H + 18$$

**Para `H=40`:**
$$8 \cdot (57 \cdot 40 + 40^2 + 2 \cdot 40) + 36 \cdot 40 + 18$$
$$= 8 \cdot (2\,280 + 1\,600 + 80) + 1\,440 + 18$$
$$= 8 \cdot 3\,960 + 1\,458 = 31\,680 + 1\,458 = \mathbf{33\,138 \text{ parámetros}}$$

De nuevo **~33K**, en línea con RNN vanilla ($H=147$ → ~33K) y LSTM ($H=64$ → ~33K). La elección $H=40$ no es arbitraria: con `8 \cdot (\cdot)` en el conteo, $H$ tiene que ser pequeño para mantener el orden. Otra vez, el lab fija $H$ por modelo para hacer la **comparación justa entre arquitecturas**.

### Resumen visual: las 3 arquitecturas lado a lado

```
              Secuencia "Sato"
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
    [RNN]        [LSTM]      [BiLSTM]
   H=147         H=64         H=40
   ~33K          ~33K         ~33K
       │            │            │
       │            │            ├── forward LSTM (lee Sato →)
       │            │            └── backward LSTM (lee otaS ←)
       │            │            │
       │            │       concat (h_fwd, h_bwd) → 2H
       │            │            │
       ▼            ▼            ▼
   h final     (h, c) final   2H combinado
       │            │            │
       └────────────┼────────────┘
                    ▼
            Linear → 18 logits → CrossEntropy
```

Las 3 arquitecturas convergen en la misma capa de salida (`Linear` a 18 clases), pero llegan a esa proyección con representaciones cada vez más ricas:

| Arquitectura | Memoria | Direccionalidad |
|--------------|---------|----------------|
| RNN | un vector $h_t$ | unidireccional |
| LSTM | dos vectores $(h_t, c_t)$ con autopista | unidireccional |
| BiLSTM | cuatro vectores (dos pares $(h, c)$, uno por dirección) | bidireccional |

---

## Resumen comparativo

| Modelo | $H$ | Parámetros (~) | Lectura |
|--------|-----|----------------|---------|
| RNN vanilla | 147 | 32.9K | izquierda → derecha |
| LSTM | 64 | 32.7K | izquierda → derecha + estado celular |
| BiLSTM | 40 | 33.1K | bidireccional + estado celular |

La función helper para calcular esto en cualquier modelo:

```python
def num_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

---

## Verificación del forward pass (sanity check)

Antes de invertir 5–10 minutos entrenando con 200 000 iteraciones, conviene confirmar que las clases compilan, los shapes cuadran y el autograd está vivo. El notebook hace esto con dos celdas mínimas que ejecutan **un solo paso temporal** sobre la letra `'A'`.

### Celda 1: forward sobre `letterToTensor('A')`

```python
input = letterToTensor('A')                      # (1, 57)
print(input.shape)
hidden = rnn.init_hidden()                       # (1, 32) en cero

output, next_hidden = rnn(input, hidden)
print(output, output.shape, next_hidden, next_hidden.shape)
```

**Output esperado** (los números varían por la inicialización aleatoria de `nn.Linear`):

```
torch.Size([1, 57])
tensor([[ 0.01, -0.04,  0.17, ...,  0.04]], grad_fn=<AddmmBackward0>)
torch.Size([1, 18])
tensor([[ 0.02, -0.05, ...,  0.06]], grad_fn=<TanhBackward0>)
torch.Size([1, 32])
```

**Lo que pasa internamente** en `rnn(input, hidden)`:

```
1. self.x2h(input)        → Linear(57, 32) sobre (1, 57)  →  (1, 32)
2. self.h2h(old_hidden)   → Linear(32, 32) sobre (1, 32)  →  (1, 32)
3. suma + tanh            → (1, 32) acotado a [-1, 1]      ← nuevo hidden
4. self.h2o(hidden)       → Linear(32, 18) sobre (1, 32)  →  (1, 18)  ← output
```

### Celda 2: forma equivalente con `lineToTensor('Albert')[0]`

```python
input = lineToTensor('Albert')                   # (6, 1, 57)
print(input.shape)
hidden = rnn.init_hidden()

output, next_hidden = rnn(input[0], hidden)      # solo la primera letra
print(output)
```

**Output esperado:**

```
torch.Size([6, 1, 57])
tensor([[ 0.0097,  0.1105, -0.1662, -0.1557,  0.0832,  0.1177, -0.1692, -0.0050,
          0.0166, -0.1240,  0.0395,  0.1003, -0.1207, -0.1688, -0.1632, -0.1630,
         -0.2446,  0.2767]], grad_fn=<AddmmBackward0>)
```

`lineToTensor('Albert')[0]` es **idéntico** a `letterToTensor('A')` — son dos formas de obtener el mismo tensor `(1, 57)` para la primera letra. Ambas celdas demuestran lo mismo: un único paso temporal del forward.

### Cuatro cosas que confirmar en el output

| Verificación | Cómo se ve | Por qué importa |
|--------------|-----------|-----------------|
| ✅ `output` shape `(1, 18)` | 18 valores en el tensor | Confirma que `h2o` está bien dimensionada (`output_size = n_categories = 18`) |
| ✅ `next_hidden` shape `(1, H)` | `(1, 32)` con `n_hidden=32` | El hidden state nuevo tiene la dimensión correcta |
| ✅ Valores pequeños y centrados en 0 | `[-0.25, +0.28]` aprox. | Esperado: pesos aleatorios + casi todo el input es 0 → logits sin estructura |
| ✅ `grad_fn=<AddmmBackward0>` | Aparece en cada tensor | Autograd está rastreando el grafo — `loss.backward()` podrá calcular gradientes |

### "Predicción" antes de entrenar

Si forzaras una predicción ahora con `argmax`:

```python
predicted_class = output.argmax(dim=1).item()
print(all_categories[predicted_class])
```

El logit más alto en el ejemplo es `0.2767` en la posición 17 (la última). Eso te daría la clase `all_categories[17]` — pero **no significa nada**. La red no aprendió eso de la `'A'`; es ruido aleatorio de la inicialización. Reinicia todo y el "ganador" cambia.

### Para procesar un apellido completo

Esta celda procesa **un solo carácter**. Para clasificar la palabra entera necesitas iterar alimentando el `hidden` recursivamente:

```python
hidden = rnn.init_hidden()
input_tensor = lineToTensor('Albert')

for i in range(input_tensor.size(0)):       # 0..5
    output, hidden = rnn(input_tensor[i], hidden)

# El último 'output' (después de la 't') es el que se usa para clasificar
prediccion = output.argmax(dim=1).item()
print(all_categories[prediccion])
```

Eso es exactamente lo que hace la función `train()` que veremos en [Entrenamiento](../entrenamiento). Esta celda existe sólo para confirmar que la mecánica de un paso funciona antes de armar el loop completo.

> **¿Por qué `n_hidden=32` aquí y no 147?** El `n_hidden=32` que aparece en esta celda es arbitrario — sólo necesita un valor razonable para hacer el sanity check. Los valores reales del experimento (`147` para RNN, `64` para LSTM, `40` para BiLSTM) viven en las celdas de entrenamiento más abajo, calibrados para que las 3 arquitecturas tengan ~33K parámetros y la comparación sea justa.
