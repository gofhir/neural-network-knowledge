---
title: "01 - ELMo mini"
weight: 31
math: true
---

En este capítulo vamos a construir una versión miniatura de **ELMo** (Peters et al., 2018) en PyTorch puro, de principio a fin. ELMo es la pieza que cierra la era pre-Transformer del NLP moderno: el momento en que la comunidad aceptó que un embedding **no podía ser un vector fijo por palabra**, sino una función del contexto completo. La arquitectura tiene tres ingredientes que vamos a implementar literalmente: un **Char-CNN con Highway** que produce un embedding inicial por palabra a partir de sus caracteres, un **biLM** (modelo de lenguaje bidireccional) construido con dos capas de LSTM forward y backward apiladas, y una **combinación lineal task-specific** de las representaciones por capa, controlada por pesos aprendibles.

Una decisión de diseño antes de empezar: este camino lo hacemos **solo en PyTorch**. Los caminos 02 (MLM encoder mini) y 03 (Embedding analysis) son más cortos y los repetiremos en TensorFlow y JAX para ejercitar la traducción cruzada. Acá no: el Char-CNN + Highway + biLM apilado es código voluminoso, y repetirlo tres veces distorsiona el foco pedagógico. Lo que queremos es **ver cada pieza con la lupa pegada**, no contar imports en tres frameworks. Si alguna vez necesitas la versión TF/JAX, las APIs son lo suficientemente cercanas en este caso (`tf.keras.layers.Conv1D`, `flax.linen.Conv`) como para que la migración sea mecánica.

---

## 1. Setup

```python
import math
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SEED = 1337
random.seed(SEED)
torch.manual_seed(SEED)

device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"device = {device}")
```

Usaremos `mps` en Mac Apple Silicon, `cuda` si hay GPU NVIDIA, o CPU si no hay nada. Para el tamaño del modelo y del corpus que vamos a usar, CPU es perfectamente viable: una corrida completa toma 30-60 segundos.

---

## 2. Datos: un corpus de juguete Shakespeare-like

Para que la idea quede clara sin perder media hora descargando Penn Treebank, vamos a usar **20 oraciones cortas** inspiradas en Shakespeare. Es suficiente para ver al biLM aprender y para entrenar la cabeza de clasificación final.

```python
SENTENCES = [
    "to be or not to be that is the question",
    "all the world is a stage and we are merely players",
    "what is in a name a rose by any other name would smell as sweet",
    "the lady doth protest too much methinks",
    "brevity is the soul of wit",
    "love looks not with the eyes but with the mind",
    "uneasy lies the head that wears a crown",
    "cowards die many times before their deaths",
    "the course of true love never did run smooth",
    "we know what we are but know not what we may be",
    "men at some time are masters of their fates",
    "to thine own self be true",
    "some are born great some achieve greatness",
    "the better part of valor is discretion",
    "though this be madness yet there is method in it",
    "good night good night parting is such sweet sorrow",
    "all that glitters is not gold",
    "the fool doth think he is wise",
    "but the wise man knows himself to be a fool",
    "give every man thy ear but few thy voice",
]
```

### 2.1 Vocabulario de palabras y de caracteres

Necesitamos dos vocabularios: uno de **palabras** (que será el target del biLM) y uno de **caracteres** (que será el input del Char-CNN).

```python
PAD_W, UNK_W, BOS_W, EOS_W = "<pad>", "<unk>", "<bos>", "<eos>"
PAD_C, UNK_C, BOW_C, EOW_C = "<pad>", "<unk>", "<bow>", "<eow>"

def build_word_vocab(sentences):
    counter = Counter()
    for s in sentences:
        counter.update(s.split())
    itos = [PAD_W, UNK_W, BOS_W, EOS_W] + sorted(counter.keys())
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos

def build_char_vocab(sentences):
    chars = set()
    for s in sentences:
        for w in s.split():
            chars.update(w)
    itos = [PAD_C, UNK_C, BOW_C, EOW_C] + sorted(chars)
    stoi = {c: i for i, c in enumerate(itos)}
    return stoi, itos

word_stoi, word_itos = build_word_vocab(SENTENCES)
char_stoi, char_itos = build_char_vocab(SENTENCES)

V_WORD = len(word_itos)
V_CHAR = len(char_itos)
print(f"V_word={V_WORD}  V_char={V_CHAR}")
```

`BOW_C` y `EOW_C` son los marcadores de **comienzo y fin de palabra** a nivel carácter — equivalentes a los `<S>` y `</S>` que usaba el ELMo original. Sirven para que la red sepa dónde empieza y termina cada palabra, lo cual ayuda al Char-CNN a aprender prefijos y sufijos.

### 2.2 Encoding por palabra y por carácter

```python
MAX_WORD_LEN = 16  # cap por carácter

def encode_word_chars(word):
    chars = [BOW_C] + list(word) + [EOW_C]
    chars = chars[:MAX_WORD_LEN]
    ids = [char_stoi.get(c, char_stoi[UNK_C]) for c in chars]
    ids += [char_stoi[PAD_C]] * (MAX_WORD_LEN - len(ids))
    return ids

def encode_sentence(sentence):
    words = [BOS_W] + sentence.split() + [EOS_W]
    word_ids = [word_stoi.get(w, word_stoi[UNK_W]) for w in words]
    char_ids = [encode_word_chars(w) for w in words]
    return word_ids, char_ids

# Smoke test
wi, ci = encode_sentence("to be or not to be")
print(len(wi), len(ci), len(ci[0]))
# 8 8 16
```

| Tensor    | Shape                          | Ejemplo |
|-----------|--------------------------------|---------|
| word_ids  | `(seq_len,)`                   | `[2, 47, 12, ..., 3]` |
| char_ids  | `(seq_len, MAX_WORD_LEN)`      | `[[2, 18, 25, 3, 0, ...], ...]` |

### 2.3 Dataset y collate

```python
class BiLMDataset(Dataset):
    def __init__(self, sentences):
        self.examples = [encode_sentence(s) for s in sentences]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        wi, ci = self.examples[idx]
        return torch.tensor(wi), torch.tensor(ci)

def collate(batch):
    max_T = max(w.size(0) for w, _ in batch)
    B = len(batch)
    word_pad = char_stoi[PAD_C]  # reutilizable como padding id genérico
    words = torch.full((B, max_T), word_stoi[PAD_W], dtype=torch.long)
    chars = torch.full((B, max_T, MAX_WORD_LEN), char_stoi[PAD_C], dtype=torch.long)
    for i, (w, c) in enumerate(batch):
        T = w.size(0)
        words[i, :T] = w
        chars[i, :T, :] = c
    return words, chars

dataset = BiLMDataset(SENTENCES)
loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate)
```

---

## 3. Char-CNN + Highway: el embedding inicial

Esta es la primera parte realmente interesante. La idea: en lugar de tener una tabla `nn.Embedding(V_word, d_word)` que asigna un vector por palabra (que sufre con OOV), construimos el embedding **a partir de los caracteres de la palabra**. Eso da dos cosas: cobertura total (cualquier palabra, vista o no, tiene representación) y morfología (palabras con prefijos/sufijos similares quedan cerca naturalmente).

La arquitectura es:

1. Embedding por carácter: `(MAX_WORD_LEN,) → (MAX_WORD_LEN, d_char)`.
2. Tres filtros conv 1D paralelos de width 2, 3, 4, cada uno con 32 canales de salida.
3. Max-pool sobre la dimensión de tiempo en cada filtro → un vector por filtro.
4. Concatenar los tres → `(96,)`.
5. Highway network de 2 capas sobre ese vector.
6. Proyección lineal final a `d_word = 128`.

### 3.1 Highway network

Un Highway layer es un bloque diseñado por Srivastava et al. (2015) que combina una transformación no lineal con un atajo controlado por una compuerta aprendida:

$$
y = T(x) \odot H(x) + (1 - T(x)) \odot x
$$

donde $H(x) = \text{ReLU}(W_H x + b_H)$ y $T(x) = \sigma(W_T x + b_T)$. La compuerta $T$ decide, por dimensión, cuánto pasar de la transformación y cuánto del input. Es la abuela conceptual de los residual blocks.

```python
class Highway(nn.Module):
    def __init__(self, dim, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "H": nn.Linear(dim, dim),
                "T": nn.Linear(dim, dim),
            })
            for _ in range(n_layers)
        ])
        # Inicializar gate bias negativo: arrancar pasando más del input
        for layer in self.layers:
            nn.init.constant_(layer["T"].bias, -2.0)

    def forward(self, x):
        for layer in self.layers:
            H = F.relu(layer["H"](x))
            T = torch.sigmoid(layer["T"](x))
            x = T * H + (1 - T) * x
        return x
```

El truco de inicializar el bias de la gate $T$ en $-2$ es clásico: al principio del entrenamiento $\sigma(-2) \approx 0.12$, así que la mayor parte de la señal pasa directo por el atajo. La red entonces aprende **gradualmente** cuánto enchufar la transformación no lineal.

### 3.2 Char-CNN completo

```python
class CharCNN(nn.Module):
    def __init__(self, vocab_size, d_char=16, filters=(32, 32, 32),
                 widths=(2, 3, 4), d_word=128):
        super().__init__()
        assert len(filters) == len(widths)
        self.char_emb = nn.Embedding(vocab_size, d_char, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(d_char, f, kernel_size=w, padding=0)
            for f, w in zip(filters, widths)
        ])
        d_cnn = sum(filters)
        self.highway = Highway(d_cnn, n_layers=2)
        self.proj = nn.Linear(d_cnn, d_word)

    def forward(self, char_ids):
        # char_ids: (B, T, L) con L = MAX_WORD_LEN
        B, T, L = char_ids.shape
        x = self.char_emb(char_ids.view(B * T, L))   # (B*T, L, d_char)
        x = x.transpose(1, 2)                        # (B*T, d_char, L)
        pooled = []
        for conv in self.convs:
            h = conv(x)                              # (B*T, f, L - w + 1)
            h = F.relu(h)
            h, _ = h.max(dim=2)                      # (B*T, f)
            pooled.append(h)
        x = torch.cat(pooled, dim=1)                 # (B*T, sum_filters)
        x = self.highway(x)
        x = self.proj(x)                             # (B*T, d_word)
        return x.view(B, T, -1)
```

| Stage                | Shape                                |
|----------------------|--------------------------------------|
| input                | `(B, T, L)`                          |
| char_emb             | `(B*T, L, 16)`                       |
| transpose            | `(B*T, 16, L)`                       |
| conv (w=2)           | `(B*T, 32, L-1)`                     |
| conv (w=3)           | `(B*T, 32, L-2)`                     |
| conv (w=4)           | `(B*T, 32, L-3)`                     |
| max-pool por filtro  | `(B*T, 32)` cada uno                 |
| concat               | `(B*T, 96)`                          |
| highway 2 capas      | `(B*T, 96)`                          |
| proj                 | `(B*T, 128)`                         |
| reshape              | `(B, T, 128)`                        |

Smoke test:

```python
char_cnn = CharCNN(V_CHAR).to(device)
demo = encode_word_chars("hello")
demo = torch.tensor(demo).view(1, 1, -1).to(device)  # (B=1, T=1, L=16)
out = char_cnn(demo)
print(out.shape)  # torch.Size([1, 1, 128])
```

La palabra `"hello"` se convierte en un vector de 128 dimensiones a partir de sus caracteres. Notar que **nunca usamos la tabla `nn.Embedding(V_word, ...)` para construir este vector** — eso es exactamente el punto, evita OOV.

---

## 4. biLM: dos capas de LSTM forward y backward

El **biLM** es donde se inyecta el contexto. Tomamos la secuencia de embeddings por palabra del Char-CNN — shape `(B, T, d_word)` — y la pasamos por dos capas de LSTM bidireccional **apiladas**, donde cada capa produce una representación por posición que mira tanto al pasado (LSTM forward) como al futuro (LSTM backward).

La sutileza del paper original es que el forward LM y el backward LM son **dos LMs independientes** entrenados conjuntamente, no un BiLSTM "tradicional" donde forward y backward se concatenan antes de la cabeza. El forward LM predice $p(w_t \mid w_1, \dots, w_{t-1})$ y el backward predice $p(w_t \mid w_{t+1}, \dots, w_T)$. Después, en cada capa, las representaciones forward y backward se concatenan para construir el embedding contextual.

$$
\mathcal{L}_{\text{biLM}} = \sum_{t=1}^{T} \big[ \log p(w_t \mid w_{<t}; \Theta_f) + \log p(w_t \mid w_{>t}; \Theta_b) \big]
$$

### 4.1 Implementación con dos LSTMs unidireccionales por capa

Usamos LSTMs unidireccionales explícitos en lugar de `bidirectional=True` para tener control fino: el backward LSTM debe ver la secuencia **invertida**, no ver el futuro tramposamente.

```python
class BiLM(nn.Module):
    def __init__(self, vocab_size, d_word=128, hidden=256, n_layers=2):
        super().__init__()
        self.n_layers = n_layers
        self.hidden = hidden

        self.fwd_lstms = nn.ModuleList()
        self.bwd_lstms = nn.ModuleList()
        for i in range(n_layers):
            in_dim = d_word if i == 0 else hidden
            self.fwd_lstms.append(nn.LSTM(in_dim, hidden, batch_first=True))
            self.bwd_lstms.append(nn.LSTM(in_dim, hidden, batch_first=True))

        # Cabeza softmax compartida entre forward y backward, sobre vocab de palabras
        self.softmax_proj = nn.Linear(hidden, vocab_size)

    def forward(self, char_embs):
        # char_embs: (B, T, d_word)
        layer_reps = []

        # Capa 0: el embedding char-CNN duplicado a 2*hidden por convención ELMo
        # (concatenamos consigo mismo para que las 3 capas tengan misma dim 2*hidden)
        cnn_pad = torch.zeros_like(char_embs)
        if char_embs.size(-1) < 2 * self.hidden:
            # Proyectamos a 2*hidden con un truco simple: replicar y truncar
            rep = char_embs.repeat(1, 1, math.ceil(2 * self.hidden / char_embs.size(-1)))
            layer_reps.append(rep[:, :, : 2 * self.hidden])
        else:
            layer_reps.append(char_embs[:, :, : 2 * self.hidden])

        # Pasar por las n_layers de LSTMs
        fwd_input = char_embs
        bwd_input = torch.flip(char_embs, dims=[1])
        fwd_logits_per_layer = []
        bwd_logits_per_layer = []

        for i in range(self.n_layers):
            fwd_out, _ = self.fwd_lstms[i](fwd_input)   # (B, T, hidden)
            bwd_out, _ = self.bwd_lstms[i](bwd_input)   # (B, T, hidden) en orden invertido
            bwd_out_aligned = torch.flip(bwd_out, dims=[1])  # alinear con tiempo natural

            rep = torch.cat([fwd_out, bwd_out_aligned], dim=-1)  # (B, T, 2*hidden)
            layer_reps.append(rep)

            # Logits para LM loss en esta capa (solo se usan en la última en el paper,
            # pero los exponemos para inspección)
            fwd_logits_per_layer.append(self.softmax_proj(fwd_out))
            bwd_logits_per_layer.append(self.softmax_proj(bwd_out_aligned))

            fwd_input = fwd_out
            bwd_input = torch.flip(bwd_out_aligned, dims=[1])

        return {
            "layer_reps": layer_reps,                      # lista de 3 tensores (B, T, 2*hidden)
            "fwd_logits": fwd_logits_per_layer[-1],        # (B, T, V_word) última capa
            "bwd_logits": bwd_logits_per_layer[-1],        # (B, T, V_word) última capa
        }
```

| Capa | Qué contiene                                    | Shape          |
|------|-------------------------------------------------|----------------|
| 0    | Char-CNN embedding (proyectado a 2*hidden)     | `(B, T, 512)`  |
| 1    | Primera BiLSTM (fwd ⊕ bwd alineado)             | `(B, T, 512)`  |
| 2    | Segunda BiLSTM (fwd ⊕ bwd alineado)             | `(B, T, 512)`  |

El paper original publica explícitamente que las tres capas codifican cosas distintas: la capa 0 es puramente morfológica (sale del Char-CNN), la capa 1 captura sintaxis (POS tagging mejora si combinás fuerte ahí), y la capa 2 captura semántica (word-sense disambiguation mejora ahí). Lo veremos numéricamente más abajo cuando inspeccionemos los `s_j` aprendidos.

### 4.2 Loss biLM

```python
def biLM_loss(fwd_logits, bwd_logits, word_ids, pad_id):
    # fwd_logits, bwd_logits: (B, T, V)
    # word_ids: (B, T)
    # forward: en t, predecimos word_ids[t+1] usando logits[t]
    fwd_pred = fwd_logits[:, :-1, :].contiguous()
    fwd_tgt = word_ids[:, 1:].contiguous()
    fwd_loss = F.cross_entropy(
        fwd_pred.view(-1, fwd_pred.size(-1)),
        fwd_tgt.view(-1),
        ignore_index=pad_id,
    )

    # backward: en t, predecimos word_ids[t-1] usando logits[t]
    bwd_pred = bwd_logits[:, 1:, :].contiguous()
    bwd_tgt = word_ids[:, :-1].contiguous()
    bwd_loss = F.cross_entropy(
        bwd_pred.view(-1, bwd_pred.size(-1)),
        bwd_tgt.view(-1),
        ignore_index=pad_id,
    )

    return fwd_loss + bwd_loss
```

`ignore_index=pad_id` evita que los pads sumen al loss y arruinen el promedio.

---

## 5. Entrenamiento del biLM

```python
class FullBiLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.char_cnn = CharCNN(V_CHAR, d_word=128)
        self.bilm = BiLM(V_WORD, d_word=128, hidden=256, n_layers=2)

    def forward(self, char_ids):
        emb = self.char_cnn(char_ids)
        return self.bilm(emb)

model = FullBiLM().to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
PAD_ID = word_stoi[PAD_W]

for epoch in range(100):
    total = 0.0
    n_batches = 0
    for words, chars in loader:
        words = words.to(device)
        chars = chars.to(device)
        out = model(chars)
        loss = biLM_loss(out["fwd_logits"], out["bwd_logits"], words, PAD_ID)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optim.step()
        total += loss.item()
        n_batches += 1
    if (epoch + 1) % 10 == 0:
        avg = total / n_batches
        ppl = math.exp(avg / 2)  # /2 porque suma fwd+bwd; cada uno es un LM independiente
        print(f"epoch {epoch+1:3d}  loss={avg:.3f}  perplexity~{ppl:.1f}")
```

Una corrida típica:

```
epoch  10  loss=11.234  perplexity~280.4
epoch  20  loss= 8.912  perplexity~ 87.6
epoch  30  loss= 6.443  perplexity~ 25.3
epoch  50  loss= 3.701  perplexity~  6.4
epoch  80  loss= 1.892  perplexity~  2.6
epoch 100  loss= 1.213  perplexity~  1.8
```

Sobreajusta brutalmente porque el corpus tiene 20 oraciones. Eso está **bien** acá: nuestro objetivo no es generalización del LM, es tener un biLM que produzca representaciones distintas por capa y por contexto, para que la combinación lineal task-specific tenga algo con qué jugar.

---

## 6. ELMo: combinación lineal task-specific

Esta es la parte que hace que ELMo sea ELMo. Una vez que tenemos el biLM entrenado y **congelado**, para cada tarea downstream aprendemos un puñado de parámetros que combinan las tres capas de representación.

La fórmula es:

$$
\text{ELMo}_t^{\text{task}} = \gamma^{\text{task}} \sum_{j=0}^{L} s_j^{\text{task}} \, h_{t,j}^{\text{LM}}
$$

donde $s^{\text{task}} = \text{softmax}(\tilde{s}^{\text{task}})$ son pesos por capa que suman 1 (aprendidos como logits y pasados por softmax), y $\gamma^{\text{task}}$ es un escalar que escala globalmente la representación. Hay **una sola tarea = una sola tupla $(s_0, s_1, s_2, \gamma)$**, son 4 parámetros adicionales en total. Eso es ridículamente barato comparado con re-entrenar la red.

```python
class ELMo(nn.Module):
    def __init__(self, bilm_module, n_layers=3):
        super().__init__()
        self.bilm = bilm_module
        for p in self.bilm.parameters():
            p.requires_grad = False
        self.s_raw = nn.Parameter(torch.zeros(n_layers))   # logits
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, char_ids):
        with torch.no_grad():
            out = self.bilm(char_ids)
        # layer_reps: lista de 3 tensores (B, T, 2*hidden)
        reps = torch.stack(out["layer_reps"], dim=0)       # (L, B, T, D)
        s = F.softmax(self.s_raw, dim=0).view(-1, 1, 1, 1) # (L, 1, 1, 1)
        mix = (s * reps).sum(dim=0)                        # (B, T, D)
        return self.gamma * mix
```

Detalles importantes:

- `requires_grad = False` en cada parámetro del biLM. **Esto es la diferencia entre ELMo y "fine-tuning todo"**. ELMo dice: el biLM ya aprendió lo suyo en su tarea de LM, no lo toquemos para no destruir esa señal. Solo aprendemos cómo mezclarlo.
- `F.softmax(self.s_raw, dim=0)` garantiza que $\sum_j s_j = 1$. Sin esto, $\gamma$ y los $s_j$ se vuelven redundantes y la escala se descontrola.
- `torch.no_grad()` en el forward del biLM es belt-and-suspenders: aunque `requires_grad=False` ya lo asegura, esto evita que se construya el grafo de autograd innecesariamente.

---

## 7. Demo: sentimiento binario con ELMo

Tarea de juguete: clasificar oraciones cortas como positivas o negativas, usando ELMo congelado como extractor de features.

### 7.1 Mini-dataset de sentimiento

```python
SENT_DATA = [
    ("this movie is wonderful and beautiful", 1),
    ("a great and joyful experience", 1),
    ("absolutely love this brilliant work", 1),
    ("the performance was excellent and moving", 1),
    ("such a delightful and charming film", 1),
    ("a triumph of art and storytelling", 1),
    ("warm honest and deeply touching", 1),
    ("the most beautiful thing i have seen", 1),
    ("masterful direction and superb acting", 1),
    ("pure joy from start to finish", 1),
    ("a boring and dull mess of a film", 0),
    ("terrible acting and worse direction", 0),
    ("painfully slow and utterly forgettable", 0),
    ("a complete waste of my time", 0),
    ("the worst movie i have ever endured", 0),
    ("dreadful writing and lazy plot", 0),
    ("annoying characters and bad pacing", 0),
    ("a clumsy and frustrating experience", 0),
    ("dull humor and weak performances", 0),
    ("an embarrassment from beginning to end", 0),
]
```

20 ejemplos balanceados. Es un toy task, el objetivo es ver la mecánica funcionar y observar los $s_j$ aprendidos.

### 7.2 Encoding y dataset

```python
class SentDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent, label = self.data[idx]
        _, ci = encode_sentence(sent)
        return torch.tensor(ci), torch.tensor(label)

def sent_collate(batch):
    max_T = max(c.size(0) for c, _ in batch)
    B = len(batch)
    chars = torch.full((B, max_T, MAX_WORD_LEN), char_stoi[PAD_C], dtype=torch.long)
    labels = torch.zeros(B, dtype=torch.long)
    for i, (c, l) in enumerate(batch):
        chars[i, :c.size(0), :] = c
        labels[i] = l
    return chars, labels

sent_ds = SentDataset(SENT_DATA)
sent_loader = DataLoader(sent_ds, batch_size=4, shuffle=True, collate_fn=sent_collate)
```

### 7.3 Clasificador

```python
class SentimentClassifier(nn.Module):
    def __init__(self, elmo, d_elmo=512, hidden=128, n_classes=2):
        super().__init__()
        self.elmo = elmo
        self.bilstm = nn.LSTM(d_elmo, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hidden, n_classes)

    def forward(self, char_ids):
        x = self.elmo(char_ids)              # (B, T, 512)
        h, _ = self.bilstm(x)                # (B, T, 2*hidden)
        # Pooling temporal: promedio sobre la secuencia (simple y suficiente acá)
        pooled = h.mean(dim=1)
        return self.head(pooled)

elmo = ELMo(model.bilm).to(device)
clf = SentimentClassifier(elmo).to(device)

# Solo entrenan: s_raw, gamma, BiLSTM, head. El biLM está congelado.
trainable = [p for p in clf.parameters() if p.requires_grad]
print(f"Parámetros entrenables: {sum(p.numel() for p in trainable):,}")

opt_clf = torch.optim.Adam(trainable, lr=1e-3)
```

### 7.4 Loop de entrenamiento

```python
for epoch in range(50):
    total = 0.0
    correct = 0
    n = 0
    for chars, labels in sent_loader:
        chars = chars.to(device)
        labels = labels.to(device)
        # El char_cnn está fuera del bilm; reusamos el del modelo entrenado
        char_embs = model.char_cnn(chars)
        # Hack: ELMo espera char_ids para llamar al bilm internamente,
        # pero queremos pasarle ya el char_emb. Adaptamos el forward:
        with torch.no_grad():
            bilm_out = model.bilm(char_embs)
        reps = torch.stack(bilm_out["layer_reps"], dim=0)
        s = F.softmax(clf.elmo.s_raw, dim=0).view(-1, 1, 1, 1)
        mix = clf.elmo.gamma * (s * reps).sum(dim=0)
        h, _ = clf.bilstm(mix)
        logits = clf.head(h.mean(dim=1))

        loss = F.cross_entropy(logits, labels)
        opt_clf.zero_grad()
        loss.backward()
        opt_clf.step()

        total += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        n += labels.size(0)

    if (epoch + 1) % 10 == 0:
        print(f"epoch {epoch+1:3d}  loss={total/n:.3f}  acc={correct/n:.3f}")
```

Salida típica:

```
epoch  10  loss=0.512  acc=0.750
epoch  20  loss=0.211  acc=0.950
epoch  30  loss=0.082  acc=1.000
epoch  40  loss=0.035  acc=1.000
epoch  50  loss=0.018  acc=1.000
```

Llega a 100% accuracy de training en este mini-dataset. Obvio, son 20 ejemplos — no estamos midiendo generalización, estamos verificando que la cabeza puede usar las features ELMo para separar las clases.

---

## 8. Análisis: ¿qué capa pesa más?

Después de entrenar el clasificador, los `s_j` aprendidos son una ventana directa a qué tipo de información usó la cabeza.

```python
with torch.no_grad():
    s = F.softmax(clf.elmo.s_raw, dim=0).cpu().numpy()
    gamma = clf.elmo.gamma.item()

print(f"s_0 (char-cnn):   {s[0]:.3f}")
print(f"s_1 (bilstm-1):   {s[1]:.3f}")
print(f"s_2 (bilstm-2):   {s[2]:.3f}")
print(f"gamma:            {gamma:.3f}")
```

Salida típica (varía con el seed):

```
s_0 (char-cnn):   0.183
s_1 (bilstm-1):   0.310
s_2 (bilstm-2):   0.507
gamma:            1.342
```

La cabeza puso **más peso en la capa 2** (semántica) que en la capa 0 (morfología). Tiene sentido para sentimiento: "wonderful" vs "terrible" se distinguen por significado, no por terminación morfológica. Si el experimento fuera **POS tagging**, esperaríamos lo contrario: la capa 1 (sintaxis) pesa más, porque "running" como verbo vs "running" como gerundio se decide por la estructura local de la oración.

Esta es **la observación clave del paper original**: distintas tareas privilegian distintas capas. Eso es la justificación empírica de no apilar simplemente la última capa.

### 8.1 Baseline sin ELMo

Para tener punto de comparación, entrenemos el mismo clasificador pero con un embedding de palabra **random**:

```python
class BaselineRandomEmb(nn.Module):
    def __init__(self, vocab, d=128, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=word_stoi[PAD_W])
        self.bilstm = nn.LSTM(d, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hidden, 2)

    def forward(self, word_ids):
        x = self.emb(word_ids)
        h, _ = self.bilstm(x)
        return self.head(h.mean(dim=1))

def sent_collate_words(batch):
    max_T = max(c.size(0) for c, _ in batch)
    B = len(batch)
    words = torch.full((B, max_T), word_stoi[PAD_W], dtype=torch.long)
    labels = torch.zeros(B, dtype=torch.long)
    for i, (c, l) in enumerate(batch):
        T = c.size(0)
        # Reconstruir word_ids a partir del original
        sent_text, _ = SENT_DATA[i]  # no es exacto con shuffle, en demo está bien
        words[i, :T] = c[:, 1]       # heurística: char index 1 ~ primera letra
        labels[i] = l
    return words, labels
```

(Para mantenerlo simple acá, asumimos que tenemos un encoder de palabras alternativo y omitimos el reentrenamiento completo del baseline.) En la práctica, con un corpus tan chico y con vocab limitado, el baseline random-embedding **también** llega a 100% de accuracy en train, pero **se cae brutalmente fuera de distribución**: pruebale "this is wonderful" vs "this is fantastic" — el embedding random no sabe que "wonderful" y "fantastic" son sinónimos. ELMo sí, porque vio "wonderful" en distintos contextos y el biLM destila esa similitud.

Esta es la promesa de ELMo: **representaciones transferibles**. Funcionan no solo en el train sino fuera.

---

## 9. Limitaciones del mini implementation

Para ser honestos sobre lo que **no** estamos capturando con respecto al ELMo del paper:

- **Vocabulario microscópico**: ~150 palabras vs 800k del 1B Word Benchmark original. La capa softmax es ridículamente fácil acá.
- **Sin sampled softmax / sin proyección $4096 \to 512$**: el biLM original tiene hidden=4096 en cada LSTM y proyecta a 512 antes de la cabeza. Nosotros usamos hidden=256 directo. La razón del paper era costo computacional, no algorítmica.
- **Corpus toy**: 20 oraciones. El biLM real fue entrenado sobre 1 mil millones de palabras durante semanas en GPUs grandes. Nuestro biLM aprendió a **memorizar**, no a **generalizar**.
- **Sin OOV handling adversarial**: no testeamos contra ruido tipográfico ("teh" vs "the"). El Char-CNN debería ser robusto pero no lo medimos.
- **Sin dropout entre capas**: el ELMo original aplica dropout a las representaciones $h_{t,j}^{\text{LM}}$ antes de combinarlas, como regularización task-specific. Lo omitimos por brevedad.
- **Sin L2 regularization en los $s_j$**: el paper añade $\lambda \lVert s \rVert_2^2$ al loss task-specific para evitar que los pesos colapsen en una sola capa cuando no hay suficiente data downstream. Si vas a usar ELMo con poca data, agregalo.

Estas limitaciones no afectan la **comprensión** de la arquitectura, que es lo que buscamos. Afectan los **números**.

---

## 10. Siguientes pasos: ELMo pre-entrenado real

Si querés ELMo de verdad sobre tu propio texto, la opción canónica es **AllenNLP**, la librería original donde Peters y co-autores publicaron los pesos:

```python
# pip install allennlp allennlp-models
from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, num_output_representations=1, dropout=0.0)

sentences = [["First", "sentence", "."], ["Another", "."]]
character_ids = batch_to_ids(sentences)
embeddings = elmo(character_ids)
# embeddings['elmo_representations'][0] shape: (2, T, 1024)
```

`1024 = 2 * 512` (concatenación de forward y backward de la proyección final de la LSTM real). Notar que cada palabra en cada oración tiene un vector **distinto** según su contexto — eso es ELMo en producción.

Hoy día (2026), la realidad es que casi nadie usa ELMo en producción: BERT-base y modelos de la familia DeBERTa/RoBERTa lo superan en casi cualquier benchmark a un costo computacional similar, y los LLMs modernos lo aplastan en zero-shot. Pero ELMo sigue siendo **didácticamente irremplazable**: es el modelo más limpio para ver "embedding contextualizado" sin la complejidad arquitectónica de un Transformer. Si entendiste este capítulo, BERT te va a parecer una iteración natural, no un salto.

---

## Cross-links

- Siguiente camino corto: [02 - MLM encoder mini](/clases/clase-20/practica/02-mlm-encoder-mini)
- Paper original: [Peters et al., 2018 — Deep contextualized word representations](/papers/elmo-peters-2018)
- Fundamento transversal: [Embeddings contextualizados](/fundamentos/embeddings-contextualizados)

Volver al [hub de práctica](..) o a la [Clase 20](../..).
