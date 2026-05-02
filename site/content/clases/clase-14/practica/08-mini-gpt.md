---
title: "08 - Mini-GPT entrenado en Shakespeare"
weight: 80
math: true
---

Llegamos. Este es el final del viaje. Despues de ocho escalones, vamos a juntar absolutamente todo lo que construimos — embeddings, dot product, softmax, cross-entropy, autograd, gradient descent, Q/K/V con scaling, multi-head, bloque Transformer — y le vamos a agregar las pocas piezas que faltan para tener un GPT real, en miniatura. Vamos a entrenarlo en texto de Shakespeare. Y vas a verlo aprender a hablar.

El script que acompana este capitulo es `clase_14/practica/05_mini_gpt.py`. Te recomiendo leerlo en paralelo con el texto, correrlo en tu maquina (idealmente con MPS o CUDA), y mirar como evoluciona el texto generado en pantalla mientras el modelo entrena. Ese momento — verlo pasar de basura random a Shakespeare reconocible en menos de un minuto — es el "click" que hace que todo lo aprendido se sienta real.

---

## 1. Donde estamos: el cierre del viaje

Antes de meternos en el codigo, mira hacia atras. Esto es lo que construiste, escalon por escalon:

- **Escalon 01** — vectores, dot product, softmax, self-attention degenerada con $Q = K = V = X$.
- **Escalon 02** — cross-entropy, la funcion de perdida que castiga estar confiado y equivocado.
- **Escalon 02b** — self-supervision, la idea revolucionaria: el texto se etiqueta a si mismo.
- **Escalon 03** — gradient descent + autograd, el motor que aprende.
- **Escalon 04** — mini-Word2Vec entrenado: el primer modelo que de verdad aprendio algo.
- **Escalon 05** — Q/K/V con scaling: el primer ladrillo real del Transformer.
- **Escalon 06** — multi-head attention: el modelo gana cabezas paralelas.
- **Escalon 06b** — multi-head internals, las shapes y el split por cabezas.
- **Escalon 07** — el bloque Transformer completo, con residuales, LayerNorm y FFN.
- **Escalon 08** — **Mini-GPT entrenado en Shakespeare** ← estas aqui.

Cada escalon agregaba una pieza pequena. Hoy, en este capitulo, ensamblamos todo. No hay teoria nueva mayor, salvo una pieza pequena: la **mascara causal**. Lo demas es composicion: tomamos los componentes que ya hicimos, los apilamos, los conectamos a un dataset real, y le damos `optimizer.step()` durante unos miles de iteraciones.

{{< concept-alert type="recordar" >}}
Este escalon no introduce conceptos nuevos significativos. Su valor esta en **ver todo lo aprendido funcionando junto**. Vas a reconocer cada pieza del codigo. Si te da la sensacion de "todo encaja", ese es el objetivo: querias llegar exactamente a este momento.
{{< /concept-alert >}}

---

## 2. Lo que falta para llegar a un GPT real

Tenemos casi todo. Que falta? Cinco cosas, todas pequenas:

1. **Positional encoding.** Self-attention sin ayuda no sabe el orden de los tokens — para ella, "perro muerde hombre" y "hombre muerde perro" son la misma bolsa. Necesitamos inyectar informacion de posicion. Usaremos un embedding de posiciones aprendido: una tabla con una fila por cada posicion en la ventana de contexto.

2. **Causal mask.** Para que el modelo pueda **generar** texto autoregresivamente — un token a la vez, basado solo en lo anterior — el token en la posicion $i$ tiene que poder ver solo las posiciones $0, 1, \dots, i$ y nunca el futuro. Eso lo logramos con una mascara triangular en la matriz de atencion.

3. **Output head.** Despues de las $N$ capas Transformer, el modelo tiene una representacion vectorial por token de dimension $d_{model}$. Para predecir el siguiente caracter necesitamos una capa lineal final que proyecte de $d_{model}$ a $\text{vocab\_size}$. Eso da los logits sobre todos los caracteres posibles.

4. **Training loop con texto real.** Bajar Shakespeare, tokenizar a nivel caracter, samplear ventanas, calcular cross-entropy, llamar `backward`, llamar `step`. Ya lo vimos en el escalon 04 con Word2Vec; aqui es la misma estructura pero con un modelo mucho mas grande.

5. **Generacion autoregresiva.** Una vez entrenado, hay que poder pedirle "dame 500 caracteres". El modelo predice el siguiente, lo concatena al contexto, predice el siguiente, repite. Eso es lo que hace `model.generate(...)` en cualquier LLM moderno.

De estas cinco, la unica realmente nueva es la **causal mask**. Las demas son aplicaciones directas de cosas que ya conoces. Vamos a ella primero.

---

## 3. Causal mask: la pieza nueva

La self-attention que construimos en los escalones 05-07 es bidireccional: cada token podia atender a todos los demas, incluyendo los del futuro. Eso esta bien para tareas como clasificacion o reconstruccion (lo que hace BERT), pero rompe completamente la posibilidad de **generar texto un token a la vez**.

Si en entrenamiento le mostramos al modelo la oracion "Romeo, oh Romeo" entera y le preguntamos "que viene despues de la primera 'R'?", queremos que la respuesta sea "o" (la siguiente letra). Pero si la atencion es bidireccional, el modelo en la posicion 0 puede mirar todas las letras posteriores, "ver" la respuesta y aprender a copiarla. No esta aprendiendo a predecir, esta aprendiendo a hacer trampa.

La **causal mask** soluciona esto a nivel de la matriz de atencion. La idea: forzar a que cada posicion solo pueda atender al pasado.

### 3.1 La idea visual

Pensemoslo en una secuencia de cuatro tokens. Sin mascara, cada token puede atender a todos:

```
        atendiendo a:  0    1    2    3
posicion 0  ->         si   si   si   si
posicion 1  ->         si   si   si   si
posicion 2  ->         si   si   si   si
posicion 3  ->         si   si   si   si
```

Con causal mask, queda triangular:

```
        atendiendo a:  0    1    2    3
posicion 0  ->         si   no   no   no
posicion 1  ->         si   si   no   no
posicion 2  ->         si   si   si   no
posicion 3  ->         si   si   si   si
```

Es decir, el token en posicion $i$ ve $\{0, 1, \dots, i\}$ y nada mas.

### 3.2 Como se implementa: triangular + masked_fill

La mascara es una matriz triangular inferior de unos. En PyTorch:

```python
mask = torch.tril(torch.ones(T, T))
# T = 4:
# tensor([[1., 0., 0., 0.],
#         [1., 1., 0., 0.],
#         [1., 1., 1., 0.],
#         [1., 1., 1., 1.]])
```

Esa matriz tiene un `1` donde el modelo PUEDE atender y `0` donde NO debe.

El truco para aplicarla esta en el momento exacto: **antes del softmax, despues del scaling**. Tomamos los scores, y donde la mascara es `0` reemplazamos por $-\infty$:

```python
scores = scores.masked_fill(mask == 0, float('-inf'))
weights = F.softmax(scores, dim=-1)
```

Por que $-\infty$ y no $0$? Porque dentro del softmax, $e^{-\infty} = 0$. Es decir: las posiciones futuras reciben score $-\infty$, $e^{-\infty}$ las saca a 0, el softmax las normaliza fuera, y la fila resultante distribuye masa **solo entre el pasado y el presente**, sumando 1. Si en cambio pusieramos $0$, $e^0 = 1$ daria peso no nulo a esas posiciones — mascara rota.

### 3.3 El codigo completo

Asi queda la atencion con mascara causal, version del script:

```python
class CausalMultiHeadAttention(nn.Module):
    """
    Multi-head attention con CAUSAL MASK.
    El causal mask hace que el token i solo pueda atender a los tokens 0..i,
    no al futuro. Es lo que hace al modelo autoregresivo (decoder-only).
    """
    def __init__(self, d_model, h, block_size):
        super().__init__()
        assert d_model % h == 0
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        # Mascara causal: triangular inferior. Se precomputa una vez.
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.W_Q(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.h, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        head_outputs = weights @ V

        concat = head_outputs.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_O(concat)
```

Tres detalles importantes:

- `register_buffer` registra el tensor de mascara como parte del modulo (se mueve con `.to(device)`) **pero no como parametro aprendible** (autograd no lo trackea, el optimizer no lo actualiza). Es exactamente lo que queremos: la mascara es estructura, no aprendizaje.
- La mascara se precomputa **una sola vez** al construir el modulo, con shape `(1, 1, block_size, block_size)`. En cada forward pass se "recorta" a `(1, 1, T, T)` para el largo actual de la secuencia. Eso evita reconstruirla en cada paso.
- El `1, 1` al frente es para hacer broadcasting limpio contra `scores` que tiene shape `(batch, h, T, T)`. La misma mascara se aplica a todos los batches y todas las cabezas.

{{< concept-alert type="clave" >}}
La causal mask es la diferencia entre un modelo "encoder" (BERT, ve todo el contexto) y un modelo "decoder-only" (GPT, ve solo el pasado). Es una linea de codigo. Pero define toda la familia de los LLMs generativos.
{{< /concept-alert >}}

---

## 4. El dataset: Shakespeare

Para entrenar el modelo necesitamos texto. Mucho? No tanto. Para un modelo chico, ~1MB es suficiente para que pase cosas interesantes.

El dataset clasico para experimentos de char-level language modeling es `tinyshakespeare.txt`, popularizado por Andrej Karpathy en su blog "The Unreasonable Effectiveness of Recurrent Neural Networks" (2015). Son las obras completas de Shakespeare concatenadas en un solo archivo de texto plano: 1.1 MB, ~1.1 millones de caracteres, 65 caracteres unicos.

```python
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
```

El script lo descarga la primera vez y lo cachea localmente.

### 4.1 Tokenizador char-level

GPT-3 y Claude usan tokenizadores BPE (Byte Pair Encoding) o variantes, que parten el texto en sub-palabras. Eso permite que un vocabulario de ~50k tokens cubra todo el ingles eficientemente. Para nuestro mini-GPT eso es overkill: usamos algo mucho mas simple — **char-level**: cada caracter es un token.

```python
chars = sorted(set(text))
vocab_size = len(chars)  # 65

char_to_id = {c: i for i, c in enumerate(chars)}
id_to_char = {i: c for i, c in enumerate(chars)}

def encode(s: str) -> list[int]:
    return [char_to_id[c] for c in s]

def decode(ids) -> str:
    return ''.join(id_to_char[int(i)] for i in ids)
```

Esos 65 caracteres son: las letras minusculas y mayusculas, los digitos, los signos de puntuacion comunes, espacio, salto de linea. Cada caracter es ahora un id entero.

Por que char-level? Tres razones pedagogicas:

- **Vocabulario chiquito** (65). El embedding y la cabeza de salida son super baratos. Un mini-GPT entrena en minutos en un Mac.
- **El modelo "ve" cada letra individualmente.** Cuando aprende a generar "Romeo", esta aprendiendo la distribucion conjunta de las letras: que despues de "Rom" suele venir "e", que despues de "Rome" suele venir "o", etc. Es brutalmente concreto.
- **Hay menos ayuda externa.** Un tokenizador BPE le da gratis al modelo el conocimiento de que "Romeo" es una palabra. En char-level, el modelo tiene que aprender eso solo, a partir de los datos.

### 4.2 Encoder de todo el texto

```python
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape)
# torch.Size([1115394])
```

Mas de un millon de tokens (= caracteres). De ahi vamos a samplear ventanas para entrenar.

### 4.3 Train/val split

90/10:

```python
n = int(0.9 * len(data))
train_data = data[:n]   # ~1.0M tokens
val_data = data[n:]     # ~110K tokens
```

El val split solo se usa para medir generalizacion. El modelo nunca lo ve durante entrenamiento.

---

## 5. La arquitectura completa: MiniGPT

Aqui es donde todo se junta. La clase `MiniGPT` toma:

- `vocab_size`: 65 (caracteres unicos).
- `d_model`: 128 (dimension de los embeddings y de cada token a lo largo de la red).
- `h`: 4 (cabezas de atencion).
- `n_layers`: 4 (numero de bloques Transformer apilados).
- `d_ff`: 512 (dimension interna del FFN, tipicamente $4 \times d_{model}$).
- `block_size`: 64 (longitud maxima de contexto que el modelo ve de una vez).

```python
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, h, n_layers, d_ff, block_size):
        super().__init__()
        self.block_size = block_size

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)

        # Stack de bloques Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, h, d_ff, block_size)
            for _ in range(n_layers)
        ])

        # LayerNorm final + cabeza de salida
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x, targets=None):
        B, T = x.shape

        # Embedding lookup + posicional
        tok = self.token_emb(x)                                   # (B, T, d_model)
        pos = self.pos_emb(torch.arange(T, device=x.device))      # (T, d_model)
        h = tok + pos                                             # (B, T, d_model)

        # Pasar por las N capas Transformer
        for block in self.blocks:
            h = block(h)

        h = self.ln_final(h)
        logits = self.head(h)                                     # (B, T, vocab_size)

        if targets is None:
            return logits, None

        # Cross-entropy: predecir el siguiente caracter en cada posicion
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
        return logits, loss
```

### 5.1 Las dos tablas de embedding

Hay dos `nn.Embedding`. Cada una es esencialmente una **tabla de lookup**: una matriz donde la fila $i$ es el embedding del id $i$.

- `token_emb`: tabla de shape `(vocab_size, d_model) = (65, 128)`. Para cada caracter unico, un vector de 128 dimensiones.
- `pos_emb`: tabla de shape `(block_size, d_model) = (64, 128)`. Para cada posicion en la ventana, un vector de 128 dimensiones.

Ambas se inicializan random y se aprenden durante el entrenamiento. La de tokens va a aprender que letras "se parecen" entre si (en algun sentido implicito de la tarea). La de posiciones va a aprender la geometria de "estar al principio vs al final de la ventana".

### 5.2 La suma: tok + pos

```python
h = tok + pos
```

Este es el momento donde se inyecta la informacion de posicion. `tok` tiene shape `(B, T, d_model)`, `pos` tiene shape `(T, d_model)`. PyTorch hace broadcasting y suma en cada posicion el embedding del token con el embedding de su posicion absoluta.

Por que sumar y no concatenar? Porque conceptualmente el embedding final debe seguir viviendo en $\mathbb{R}^{d_{model}}$. Sumar mantiene la dimension. Conceptualmente, suma significa: "el embedding del token, desplazado un poquito segun su posicion".

Hay variantes mas modernas (RoPE en LLaMA, ALiBi en otros) que codifican posicion de formas mas elegantes. Para el mini-GPT, este positional embedding aprendido es suficiente y es exactamente lo que usaba GPT-2.

### 5.3 El stack de bloques

```python
for block in self.blocks:
    h = block(h)
```

Cuatro bloques Transformer apilados. Cada bloque internamente hace pre-norm + causal multi-head attention + residual + pre-norm + FFN + residual. Eso lo construimos en el escalon 07.

A medida que pasa por los bloques, la representacion `h` se va "refinando". Cada bloque agrega una capa de procesamiento contextual.

### 5.4 La cabeza de salida

```python
self.ln_final = nn.LayerNorm(d_model)
self.head = nn.Linear(d_model, vocab_size, bias=False)
```

Despues de los bloques, `h` tiene shape `(B, T, d_model)`. Para cada posicion $t$ queremos un score sobre los 65 caracteres posibles: cual es el caracter mas probable como **siguiente** despues de la posicion $t$.

La proyeccion `head: d_model -> vocab_size` produce eso. El output `logits` tiene shape `(B, T, vocab_size)`. En cada posicion, 65 numeros. Cuando los pases por softmax, te dan la distribucion sobre el siguiente caracter.

LayerNorm antes de la cabeza es estandar en GPT-2/3. Estabiliza la salida.

### 5.5 El loss

```python
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    targets.view(-1)
)
```

Cross-entropy clasica. `logits` se aplana de `(B, T, V)` a `(B*T, V)`, y `targets` de `(B, T)` a `(B*T,)`. Para cada uno de los $B \cdot T$ pares (logits, target), cross-entropy mide cuanto se equivoca el modelo. La media es el loss.

Cada caracter del texto es un ejemplo de entrenamiento independiente. En un batch de `B=32` ventanas de `T=64`, eso son 2048 predicciones por iteracion.

### 5.6 Hyperparametros y conteo de parametros

Resumen:

```
vocab_size:    65
block_size:    64
batch_size:    32
d_model:       128
h:             4
n_layers:      4
d_ff:          512
learning_rate: 3e-4
max_iters:     3000
```

El modelo total tiene **0.82 M parametros (816,128)**. Para que tengas referencia: GPT-2 small tiene 124M, GPT-3 tiene 175B. Tu modelo es 1/150 de GPT-2 small, 1/215000 de GPT-3. Y aun asi, vas a ver cosas.

---

## 6. La generacion autoregresiva

Una vez entrenado, queremos generar texto. La idea es la mas pura definicion de "modelo de lenguaje": dado un contexto, predecir el siguiente token, agregarlo al contexto, repetir.

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0):
    for _ in range(max_new_tokens):
        # Cortar el contexto al block_size si es mas largo
        idx_cond = idx[:, -self.block_size:]
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / temperature  # solo la ultima posicion
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

Linea por linea:

- `idx_cond = idx[:, -self.block_size:]`: el modelo solo puede ver `block_size = 64` tokens hacia atras. Si ya generamos mas, recortamos. Esa es la "ventana deslizante".
- `logits, _ = self(idx_cond)`: forward pass. Devuelve logits para todas las posiciones de la ventana.
- `logits = logits[:, -1, :] / temperature`: solo nos interesa la **ultima** posicion. Es la que predice el siguiente token. La temperatura es un escalar que afila/suaviza la distribucion (T < 1 = mas determinista, T > 1 = mas creativo). Default 1.
- `probs = F.softmax(logits, dim=-1)`: convierte logits en distribucion de probabilidad sobre los 65 caracteres.
- `idx_next = torch.multinomial(probs, num_samples=1)`: **sampling**. En lugar de tomar el argmax (que daria texto deterministico y aburrido), sampleamos un caracter de acuerdo a la distribucion. Eso da variedad.
- `idx = torch.cat((idx, idx_next), dim=1)`: agregamos el nuevo caracter al contexto.

Repetir `max_new_tokens` veces. Cada iteracion es un forward pass completo del modelo.

{{< concept-alert type="clave" >}}
La generacion autoregresiva es el algoritmo mas simple del mundo: predecir-uno, concatenar, repetir. Asi funciona ChatGPT, Claude, Gemini, Llama. La unica diferencia es la escala del modelo. **El procedimiento es exactamente este**.
{{< /concept-alert >}}

---

## 7. Training loop

El loop es minusculo. Esto es todo:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for it in range(max_iters):
    if it % eval_interval == 0 or it == max_iters - 1:
        losses = estimate_loss()
        print(f"[step {it}] train_loss={losses['train']:.4f} val_loss={losses['val']:.4f}")
        if it > 0:
            print(sample("", 200))

    # Forward + backward + step
    x, y = get_batch("train")
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Cinco pasos por iteracion:

1. **Tomar batch**: `get_batch("train")` samplea 32 ventanas random de 64 caracteres del texto de Shakespeare. Para cada una, `y` es la misma ventana corrida un caracter a la derecha (target = next token).
2. **Forward**: pasar el batch por el modelo, obtener logits y loss.
3. **Zero grad**: limpiar gradientes acumulados de la iteracion anterior.
4. **Backward**: `loss.backward()` calcula gradientes para todos los parametros via autograd.
5. **Step**: `optimizer.step()` actualiza los pesos en la direccion que reduce el loss.

`AdamW` es Adam con weight decay desacoplado. Es el optimizer estandar para Transformers desde 2017. Learning rate `3e-4` es la "magia de Karpathy" — funciona como punto de partida razonable para casi cualquier Transformer chico.

Cada 500 pasos imprimimos el loss y un sample del modelo. Esa seccion siguiente es donde ocurre la magia.

---

## 8. EL CLICK MOMENT: la evolucion del texto generado

Esta es la parte donde se nota que algo profundo pasa. Vamos a ver, paso a paso, como evoluciona el texto generado por el modelo durante el entrenamiento. Este no es un ejemplo inventado: son los outputs reales que produce el script en `step` 0, 500, 1000, 1500, 2000, 2500 y 2999.

### Step 0 (loss 4.29) — random total

Antes de cualquier entrenamiento, el modelo es ruido inicializado con `torch.randn`. Los pesos no codifican nada. Generemos 200 caracteres:

```
Uoas&3;K?YEf-fEcWkPwQNRe.
OOuUfZWiewKy:Q$c-UkcECOIieeg abZqg
RtOKYhMtVcAO:DXOHjdng&WofyOAvrFjtKyLVDL
Cc;QeVfCcfYUSWNVcUvwfsjp
```

Basura completa. Mayusculas y minusculas mezcladas, simbolos al azar, sin estructura. El loss inicial de 4.29 esta cerca del baseline teorico $-\log(1/65) = 4.17$ — es decir, el modelo predice esencialmente uniforme sobre los 65 caracteres.

### Step 500 (loss 2.28) — aprendio el formato

Despues de solo 500 iteraciones (~5 segundos en MPS):

```
OLMVONCUSO

ICETher:
KEENCENT:
SI mamy thavend thm Hoe, doVof that,
```

Epico. **Mira lo que paso**: el modelo aprendio que en Shakespeare los nombres de personajes van en MAYUSCULA seguidos de dos puntos y salto de linea. No le dijimos nada. **Lo dedujo de los datos**. "OLMVONCUSO" no es un nombre real, "ICETher" tampoco — pero la **estructura** de "palabra mayuscula + dos puntos + texto" la captó.

Tambien aprendio que las minusculas vienen en bloques (palabras), separadas por espacios. Aparece "thavend", "doVof" — palabras inventadas pero con la geometria correcta del ingles (vocales y consonantes alternando).

### Step 1000 (loss 2.03)

Otros 500 pasos:

```
Ak, peque forcith him not thim hornordy-pried.

MUCINTIA:
herly-hu he now puadilersst:
```

**Aparecen palabras casi inglesas**. "him", "not", "now" son palabras reales. "forcith", "hornordy", "puadilersst" no, pero su morfologia es plausible: terminaciones en "-ith", "-ed", "-st" que recuerdan a ingles antiguo. El modelo capto que Shakespeare tiene mucho ingles arcaico.

### Step 1500 (loss 1.88)

```
To mord's of he pitighteer.

LEO:
That son.
```

Frases coherentes empiezan a emerger. "To mord's of he" tiene gramatica casi correcta. "That son." es una oracion completa, gramaticalmente impecable. Y ya aparecen contracciones con apostrofe — "mord's" — algo que el modelo tuvo que aprender de los datos: que despues de ciertas letras puede venir un apostrofe seguido de "s".

### Step 2000 (loss 1.74)

```
I'll plapely contuous
The mone reased of murredainter'd weren-o's thide.
```

**"I'll"** — primera contraccion bien formada. El modelo aprendio que "I" + apostrofe + "ll" es comun. "plapely", "contuous" no son palabras, pero la cadencia de la frase es Shakespeare. Hay verbos compuestos con prefijos ("contuous"), participios pasados ("reased", "murredainter'd"). Las oraciones tienen sujeto y verbo.

### Step 2500 (loss 1.66)

El modelo sigue mejorando. Loss baja, perplexity baja, las palabras inventadas son cada vez mas cortas (mas memorizables) y la estructura mas limpia.

### Step 2999 (loss 1.63) — texto Shakespeare-ish

```
To the fao well at condents
We thinking to 'that darkimter act againgts aff,
```

Texto reconocible como Shakespeare en su forma. "To the [palabra] well at [palabra]" tiene la estructura tipica del verso isabelino. "We thinking to" es casi gramatical. "'that" con apostrofe inicial es un detalle propio de obras teatrales.

### Step 2999 con prompt "ROMEO:"

Ahora le damos un contexto especifico — el nombre de un personaje real de Shakespeare:

```
ROMEO:
Sweet you, all hour hath more and conforth on
these we live us the fersest with him.

ADWARD IV:
Will how the give love isme to is withoulds,
```

**Esto es escalofriante**. El modelo:

- Mantuvo el formato: nombre en mayuscula seguido de dos puntos y dialogo en la siguiente linea.
- Genero un soliloquio en estilo Shakespeare: "Sweet you, all hour hath more and conforth on these we live us..." — tiene la cadencia y el vocabulario.
- Uso vocabulario shakespeariano: "hath", "thee" (no aparece aqui pero si en otros samples), "Sweet you" como vocativo.
- **Cambio de personaje**: despues del soliloquio de "Romeo", introdujo "ADWARD IV:" — un personaje historico de Shakespeare (Edward IV, rey en Henry VI). Tipograficamente lo armo mal ("ADWARD" en lugar de "EDWARD"), pero la idea — "despues de un personaje hablando, viene otro personaje hablando" — la tiene clarisima.

Todo esto en menos de un minuto de entrenamiento, con 0.82 M parametros, en una Mac.

---

## 9. La curva de loss

Resumiendo numericamente la corrida:

```
step    0:  4.29   (basicamente -log(1/65) = 4.17, peor)
step  500:  2.28
step 1000:  2.03
step 1500:  1.88
step 2000:  1.74
step 2500:  1.66
step 2999:  1.63
```

De **4.29 a 1.63 en 31 segundos** (en Apple Silicon MPS). Cada bajada de ~0.7 puntos en cross-entropy equivale, aproximadamente, a "duplicar la habilidad predictiva" del modelo, porque cross-entropy esta en escala logaritmica:

$$
\text{perplexity} = e^{\text{loss}}
$$

| step | loss | perplexity (caracteres efectivos) |
|------|------|-----------------------------------|
| 0    | 4.29 | 73.0 |
| 500  | 2.28 | 9.8  |
| 1000 | 2.03 | 7.6  |
| 2000 | 1.74 | 5.7  |
| 2999 | 1.63 | 5.1  |

La perplexity baja de 73 a 5.1. Significa que el modelo entrenado, cuando ve un contexto, esta **efectivamente eligiendo entre ~5 caracteres con probabilidad razonable**, en lugar de los 65 iniciales. Pasa de no saber nada a tener intuicion fuerte sobre que viene despues.

---

## 10. Lo que el modelo aprendio SIN supervision explicita

Esta es la parte que vale la pena meditar. Nadie le dijo al modelo:

- "Los nombres de personajes van en mayuscula."
- "Despues del nombre va dos puntos y salto de linea."
- "Las contracciones se escriben con apostrofe."
- "ROMEO es un personaje, EDWARD IV es otro."
- "El ingles tiene cierta distribucion de vocales y consonantes."
- "Las palabras tienen entre 3 y 10 caracteres tipicamente."
- "Los espacios separan palabras."
- "Despues de un punto suele venir mayuscula."

**Todo eso lo dedujo** el modelo a partir del unico objetivo "predecir el siguiente caracter dado el pasado". Cross-entropy + 3000 iteraciones de gradient descent + 1MB de texto de Shakespeare = patrones destilados.

Eso es **self-supervision en accion**. Sin etiquetas humanas. Sin reglas explicitas. Sin alguien diciendo "este nombre va con mayuscula". El modelo destila los patrones del texto solo, porque para predecir bien el proximo caracter tiene que internalizar todas esas regularidades.

{{< concept-alert type="recordar" >}}
La self-supervision (escalon 02b) es la idea mas importante del deep learning moderno. El **texto se etiqueta a si mismo**: cada caracter es la etiqueta del fragmento anterior. Eso convierte cualquier corpus de texto en un dataset supervisado infinito. Es la razon por la que GPT-3 puede entrenarse con 570 GB de internet sin ninguna anotacion humana.
{{< /concept-alert >}}

---

## 11. La conexion con modelos reales

Vamos a poner los numeros lado a lado para que veas exactamente **cuanto** pequeno es tu modelo y cuanto grande es la diferencia con los reales.

| Modelo          | $d_{model}$ | $n_{layers}$ | $h$  | params totales | dataset            |
|-----------------|-------------|--------------|------|----------------|--------------------|
| Tu mini-GPT     | 128         | 4            | 4    | 0.82 M         | 1 MB Shakespeare   |
| GPT-2 small     | 768         | 12           | 12   | 124 M          | 40 GB WebText      |
| GPT-2 XL        | 1600        | 48           | 25   | 1.5 B          | 40 GB WebText      |
| GPT-3           | 12288       | 96           | 96   | 175 B          | 570 GB internet    |
| Claude 3 Opus   | ~10k        | ~80          | ~96  | ~400 B         | ~varios TB         |
| LLaMA-3 70B     | 8192        | 80           | 64   | 70 B           | ~15 T tokens       |

(Los numeros de Claude son aproximados; Anthropic no los publica oficialmente.)

**MISMA ARQUITECTURA, distinta escala.** Todos comparten:

- Token embedding + positional encoding (tu mini-GPT lo tiene aprendido; LLaMA usa RoPE; GPT-2 usa aprendido como tu).
- Stack de bloques Transformer decoder-only con causal mask (identico).
- Multi-head self-attention (identica, salvo variantes mas modernas como GQA).
- FFN con activacion no lineal (tu usas ReLU; los modernos usan GELU o SwiGLU).
- LayerNorm (tu usas estandar; los modernos a veces RMSNorm).
- Output head lineal a vocab (identica).
- Entrenamiento con cross-entropy next-token + AdamW (identica).
- Generacion autoregresiva con sampling (identica).

Tu mini-GPT tiene **215,000 veces menos parametros** que GPT-3. Fue entrenado con **570,000 veces menos data**. Pero la **estructura es identica**.

> El "secreto" de los LLMs no es magia — es esta misma arquitectura, escalada masivamente. Si entiendes tu mini-GPT, entiendes la columna vertebral de GPT-3, Claude, LLaMA. Lo que sobra son detalles de ingenieria de scale.

---

## 12. Los Transformers en otros dominios

(Sidebar reflexivo. Cierra el viaje con perspectiva amplia.)

El Transformer no es solo para texto. La misma arquitectura — token embeddings + atencion + FFN + residuales — funciona en muchisimo mas:

- **Vision** (ViT, DeiT, Swin Transformer): la imagen se parte en parches de 16x16 pixels, cada parche se embebe como si fuera un token, y un Transformer normal procesa la "secuencia de parches". Resultado: state-of-the-art en clasificacion, deteccion, segmentacion. Reemplazaron a las CNN en muchos benchmarks.
- **Audio** (Whisper, MusicGen, AudioGen): el audio se convierte en un espectrograma o en tokens discretos (con un VQ-VAE), y un Transformer los procesa como secuencia. Whisper transcribe 99 idiomas con un solo modelo. MusicGen genera musica a partir de prompts de texto.
- **Generacion de imagenes** (DALL-E, Stable Diffusion, Midjourney): usan CLIP — que es un Transformer text encoder — para entender el prompt, y luego un U-Net o un Diffusion Transformer para generar pixeles. La parte de "lenguaje" es Transformer puro.
- **Biologia molecular** (AlphaFold 2/3, ESM): los aminoacidos de una proteina se tratan como tokens. Un Transformer aprende a predecir la estructura 3D. AlphaFold le dio a Demis Hassabis y John Jumper el premio **Nobel de Quimica 2024**. Si, un Transformer ganando un Nobel.
- **Codigo** (Copilot, Codex, Cursor, Claude Code): el codigo es texto. Cualquier LLM lo procesa. Asi se construyeron asistentes que escriben codigo a niveles cercanos a senior engineers.
- **Robotica** (RT-2, PaLM-E): combinan imagenes + texto + acciones del robot, todo como tokens. El Transformer aprende politicas que generalizan a tareas nuevas.
- **Multimodal** (GPT-4V, Gemini, Claude 3 con vision): un solo modelo procesa imagenes + texto + audio. Internamente todo se convierte en una unica secuencia de tokens, y un Transformer la procesa.

Mientras puedas convertir tu dominio en una **secuencia de tokens** (texto, parches, espectrogramas, aminoacidos, acciones), el Transformer funciona. Es la arquitectura mas universal jamas inventada en deep learning. Algunos hablan ya del Transformer como "la ley de Moore del aprendizaje automatico": la pieza estructural que sostiene todo el progreso de la decada.

---

## 13. La filosofia de fondo: matematicas viejas, hardware moderno

Una observacion para guardar. Las matematicas que usamos en el Transformer **no se inventaron para IA**. Son enormemente anteriores:

- **Algebra lineal** (matrices, productos punto, transposiciones): formalizada en el 1800s. Cayley, Sylvester, Grassmann.
- **Cross-entropy**: Claude Shannon, "A Mathematical Theory of Communication", 1948. Originalmente para teoria de la informacion en telecomunicaciones.
- **Softmax**: distribuciones de Boltzmann en mecanica estadistica, 1860s. La forma exacta es la misma.
- **Gradient descent**: Augustin-Louis Cauchy, 1847. Uno de los grandes matematicos del siglo XIX, mucho antes de que existiera el concepto de "computadora".

Las computadoras (GPUs) tampoco se inventaron para IA. Empezaron a fines de los 90s como aceleradores de graficos 3D para videojuegos. Su arquitectura masivamente paralela — miles de nucleos haciendo multiplicaciones de matrices en paralelo — era para renderizar poligonos. Que casualmente (o no) es exactamente la operacion que un Transformer necesita.

**El Transformer surge en la interseccion**: matematicas viejas, hardware moderno paralelizable, y una idea (atencion) que en 2014-2017 hizo click. Las arquitecturas que sobreviven en deep learning son las que se "casan bien" con la geometria del silicio. Las RNN no podian — son secuenciales, cada paso depende del anterior, no paralelizan. Los Transformers si — toda la atencion es matrices, todo se hace en paralelo.

> Si la GPU no hubiera evolucionado para juegos 3D durante 20 anos, no habria hardware para entrenar Transformers. Y sin Transformers, no habria ChatGPT. La cadena causal pasa por una decada de avances en pixeles.

Es facil mirar los LLMs de hoy y pensar que son magia. No lo son. Son una receta de cinco piezas (atencion, residuales, normalizacion, embedding, FFN), apilada $N$ veces, entrenada con un objetivo simple (predecir el siguiente token), sobre una cantidad enorme de texto, en hardware que casualmente acelera matrices. Cada pieza individual es matematicas del siglo XIX o computer science del siglo XX. La sintesis es del siglo XXI.

---

## 14. El cierre

Mira hacia atras. Hace ocho capitulos no sabias que era un vector. O sabias, pero no te habia hecho click la idea de "vector como representacion semantica". Acabas de:

- **Construir un Transformer end-to-end desde cero**, sin librerias de alto nivel como Hugging Face. Con `nn.Linear`, `nn.LayerNorm`, `F.softmax` y `F.cross_entropy`. Nada mas.
- **Entrenarlo en datos reales**, no en juguetes. Shakespeare. ~1 millon de caracteres.
- **Verlo aprender a "hablar"** en vivo, paso a paso, desde basura random hasta texto reconocible como Shakespeare en menos de un minuto.

Si abres ahora el codigo de GPT-2 (`huggingface/transformers` lo tiene), **vas a reconocer cada pieza**. La clase `GPT2Model` tiene token embedding, position embedding, una `ModuleList` de bloques, un LayerNorm final, una cabeza lineal a vocab. **Es lo mismo que tu mini-GPT.** Solo que con 124M parametros en lugar de 0.82M.

La arquitectura de PaLM, LLaMA, Claude — todas comparten la estructura que construiste. Las variaciones son menores: cambiar ReLU por SwiGLU, LayerNorm por RMSNorm, positional encoding aprendido por RoPE. El esqueleto es identico.

> **Ya entiendes los Transformers.** No conceptualmente — operacionalmente. Sabes que pasa en cada linea, sabes que shapes tienen los tensores, sabes por que cada pieza esta donde esta. Sabes leer un paper de arquitectura y mapearlo a codigo. Sabes debuggear cuando algo no entrena.

Eso no se aprende leyendo. Se aprende construyendo. Y lo construiste.

---

## 15. Que viene despues: variantes y experimentos

Si quieres seguir jugando, hay un universo enorme de variaciones del mini-GPT que puedes explorar tu solo. La estructura del codigo te lo permite — cambia una linea, ve que pasa. Algunas ideas:

**Cambios al mini-GPT actual:**

- **Texto en español**: bajate Don Quijote completo de Project Gutenberg (~2 MB) y entrenalo. El vocabulario char-level absorbe perfectamente las tildes y la "ñ". Vas a ver al modelo aprender cervantes.
- **Mas profundidad / mas heads / mas dim**: sube `n_layers` a 8, `d_model` a 256, `h` a 8. Verifica que sigue entrenando. Mide cuanto baja el loss final con mas capacidad. Hay un punto de retornos decrecientes — encuentralo.
- **Reemplazar ReLU por GELU**: cambia `F.relu` por `F.gelu` en el FFN. GELU es lo que usa GPT-2 y suele dar un 1-2% de mejora. Mide en tu mini si se nota.
- **Sustituir LayerNorm por RMSNorm**: lo que usa LLaMA. Es mas barato computacionalmente (omite el centrado por la media). Implementacion: $\text{RMSNorm}(x) = x / \text{RMS}(x) \cdot g$, una linea.
- **Implementar RoPE**: positional embeddings rotatorios. Lo que usa LLaMA. Mas elegante que el embedding aprendido, generaliza mejor a longitudes mas grandes que las vistas en entrenamiento.
- **Dropout** en attention y FFN, **weight decay** mas agresivo, **learning rate schedule** (warmup + cosine decay): esto es ingenieria de entrenamiento. Cada uno aporta unos puntos.

**Mas alla del Transformer (frontera 2024-2026):**

- **Mamba / SSMs (State Space Models)**: arquitectura alternativa, lineal en la longitud de la secuencia (vs cuadratica del Transformer). Compite cabeza a cabeza con Transformers en benchmarks recientes. Albert Gu, 2023-2024.
- **RWKV**: hibrido entre RNN y Transformer. Costo lineal en inferencia, paralelizable en entrenamiento. Open source, comunidad activa.
- **RetNet**: "Retentive Network". Microsoft, 2023. Otra alternativa lineal.
- **Mixture of Experts (MoE)**: en lugar de una FFN, hay $N$ FFNs ("expertos") y un router que decide cual usar para cada token. Asi es como GPT-4 y Mixtral aumentan capacidad sin proporcionalmente aumentar costo. La arquitectura "gigante pero esparsa" del momento.
- **Diffusion Transformers (DiT)**: combinan diffusion models con Transformer backbones. Lo que esta detras de Sora, Stable Diffusion 3. La pelea por el video generativo se esta peleando aqui.

Pero todas estas — y todas las que vendran — son variaciones sobre el tema que acabas de aprender. La intuicion central — secuencias de tokens, atencion entre ellos, capas residuales — sigue siendo la guia. Si entiendes el Transformer, todas las otras arquitecturas son comparaciones contra una referencia que ya tienes.

---

## 16. El final del final

Acabas de pasar por uno de los caminos pedagogicos mas densos posibles: de "que es un vector" a "construi mi propio GPT y lo entrene en Shakespeare". En ocho capitulos. Sin saltos. Sin cajas negras. Sin librerias magicas. Cada pieza, vista por dentro.

El campo va a seguir evolucionando — Mamba, MoE, lo que venga. Pero el Transformer va a estar ahi, y vas a poder leer cualquier paper de arquitectura porque tienes el lenguaje. "Multi-head causal attention", "pre-norm with residual", "FFN with $4d$ expansion", "cross-entropy on next-token prediction". No son palabras en un paper que tienes que aceptar — son cosas que escribiste tu, en codigo, y viste correr.

Eso no se desaprende.

Cierra el editor, deja al modelo seguir entrenando un rato mas si quieres, y escribe algo nuevo con el. Tu mini-GPT ya genero su primer dialogo de Romeo. El siguiente paso es tuyo.

---

## Codigo y referencias

Codigo completo: `clase_14/practica/05_mini_gpt.py`

Referencias clave:

- Vaswani et al., **"Attention Is All You Need"** (2017) — el paper original del Transformer.
- Karpathy, **"The Unreasonable Effectiveness of Recurrent Neural Networks"** (2015) — donde aparecio tinyshakespeare.
- Karpathy, **"nanoGPT"** y **"Let's build GPT: from scratch, in code, spelled out"** (YouTube, 2023) — la inspiracion directa de este escalon.
- Radford et al., **"Language Models are Unsupervised Multitask Learners"** (2019) — paper de GPT-2.
- Brown et al., **"Language Models are Few-Shot Learners"** (2020) — paper de GPT-3.

Volver al [hub de practica](..) o a la [Clase 14](../..).

**Fin del viaje.**
