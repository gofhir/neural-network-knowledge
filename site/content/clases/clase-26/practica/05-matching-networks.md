---
title: "05 - Matching Networks"
weight: 35
math: true
---

En este capítulo vamos a implementar **Matching Networks** (Vinyals et al., 2016) desde cero, en PyTorch puro. Es uno de los trabajos seminales del meta-aprendizaje moderno, y su idea es deceptivamente simple: en lugar de entrenar un clasificador paramétrico que destila lentamente las clases dentro de sus pesos, construimos un **clasificador no-paramétrico** que, dado un puñado de ejemplos etiquetados (el *support set*), predice la etiqueta de un ejemplo nuevo como una **suma ponderada por atención** sobre las etiquetas del support. La fórmula central cabe en una línea:

$$
\hat{y} = \sum_{i=1}^{k} a(\hat{x}, x_i)\, y_i,
$$

y esa línea —un softmax de similitudes que pondera "valores" (las etiquetas)— resultará ser, en retrospectiva, **exactamente la operación de atención de los Transformers**. Por eso este capítulo es a la vez una receta de few-shot learning y el puente conceptual hacia la atención key-value y hacia el in-context learning de los LLMs.

Una decisión de diseño antes de empezar: este camino lo hacemos **solo en PyTorch**. El sampler episódico, el encoder convolucional y el attention kernel forman un sistema acoplado que conviene ver con la lupa pegada en un solo framework, no contar imports en tres. La migración a TensorFlow/JAX de cada pieza (un `Conv2d`, un `cosine_similarity`, un `softmax`) es mecánica y no aporta nada pedagógico nuevo acá.

---

## 1. La idea: el support set ES el clasificador

En un clasificador estándar, las clases viven en los pesos: hay una capa `Linear(d, n_clases)` cuya matriz $W$ contiene un vector prototipo aprendido por clase. Cambiar de clases significa reescribir $W$ con descenso de gradiente — lento y con olvido catastrófico.

Matching Networks invierte esto. Las clases no viven en los pesos: viven **en los datos del support set**. El modelo aprende un único objeto reutilizable, una **función de embedding**, y la decisión se construye sobre la marcha comparando el query contra los ejemplos del support. Formalmente, MN aprende un mapeo $S \to c_S(\cdot)$: dado un support set $S=\{(x_i,y_i)\}_{i=1}^k$, produce un clasificador $c_S(\hat{x})$ que define una distribución sobre las etiquetas posibles. La forma más simple es:

$$
\hat{y} = \sum_{i=1}^{k} a(\hat{x}, x_i)\, y_i.
$$

Si las $y_i$ están codificadas one-hot y los pesos de atención $a(\hat{x}, x_i)$ suman 1, entonces $\hat{y}$ es **directamente** una distribución de probabilidad sobre clases: el peso total que cae sobre los ejemplos de la clase $c$ es la probabilidad de clase $c$.

El attention kernel propuesto es un **softmax sobre la similitud coseno** entre embeddings:

$$
a(\hat{x}, x_i) = \frac{\exp\big(c(f(\hat{x}),\, g(x_i))\big)}{\sum_{j=1}^{k} \exp\big(c(f(\hat{x}),\, g(x_j))\big)},
\qquad
c(u, v) = \frac{u^\top v}{\lVert u\rVert\,\lVert v\rVert}.
$$

Donde $f$ y $g$ son funciones de embedding (acá $f=g$, una CNN compartida). Tres lecturas iluminan la fórmula:

1. **kNN suave.** En vez del voto duro de los $k$ vecinos más cercanos, cada vecino vota con peso $\propto \exp(c)$. Con similitudes muy grandes el softmax se concentra en el vecino más cercano (kNN con $k=1$); con similitudes chicas reparte uniforme.
2. **Estimador de densidad por kernel (KDE).** Cada $x_i$ deposita densidad alrededor de su etiqueta; $a$ es el kernel. La Ecuación 1 **subsume** tanto kNN como KDE como casos particulares.
3. **Memoria asociativa.** $a$ es atención y las $y_i$ son "memorias" ligadas a sus $x_i$. El modelo "apunta" al ejemplo del support que mejor matchea y recupera su etiqueta. A diferencia de la memoria atencional paramétrica, esta es **no-paramétrica**: cuando el support crece, crece la memoria, sin tocar un solo peso.

El uso de **coseno** y no de distancia euclídea es deliberado: normaliza la magnitud de los embeddings, de modo que la similitud depende solo de la dirección. Eso estabiliza el softmax y evita que features de norma grande dominen el voto. (Prototypical Networks revisaría esta elección más adelante; lo veremos en la comparación final.)

---

## 2. El sampler N-way K-shot (reuso del camino 01)

Matching Networks no se entrena con minibatches al uso, sino con **episodios** que reproducen exactamente la tarea de evaluación: elegimos $N$ clases, damos $K$ ejemplos por clase como support, y un batch disjunto de queries de esas mismas clases. Este es el sampler que ya construimos en el camino 01; lo replicamos acá de forma autocontenida para que el capítulo corra solo.

Usaremos un dataset sintético de "caracteres" (inspirado en Omniglot) generado proceduralmente, para no depender de descargas. Cada "clase" es un patrón base al que añadimos ruido y pequeñas transformaciones, de modo que ejemplos de la misma clase se parezcan y ejemplos de clases distintas no.

```python
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 1337
random.seed(SEED)
torch.manual_seed(SEED)

device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"device = {device}")

IMG = 28          # imágenes 28x28
N_CLASSES = 60    # universo total de clases sintéticas
N_PER_CLASS = 20  # ejemplos por clase (estilo Omniglot)


def make_dataset(n_classes=N_CLASSES, n_per=N_PER_CLASS, img=IMG):
    """Genera n_classes patrones base 28x28 y n_per variantes ruidosas de cada uno."""
    g = torch.Generator().manual_seed(SEED)
    bases = torch.randn(n_classes, 1, img, img, generator=g)
    # suavizamos el patrón base con un blur para que tenga estructura espacial
    blur = torch.ones(1, 1, 3, 3) / 9.0
    bases = F.conv2d(bases, blur, padding=1)
    data, labels = [], []
    for c in range(n_classes):
        for _ in range(n_per):
            noise = 0.6 * torch.randn(1, img, img, generator=g)
            shift = random.randint(-2, 2)
            x = torch.roll(bases[c], shifts=shift, dims=2) + noise
            data.append(x)
            labels.append(c)
    return torch.stack(data), torch.tensor(labels)


X, Y = make_dataset()
print(X.shape, Y.shape)   # torch.Size([1200, 1, 28, 28]) torch.Size([1200])

# Split disjunto de CLASES: train ve clases 0..39, test ve 40..59 (nunca vistas)
TRAIN_CLASSES = list(range(0, 40))
TEST_CLASSES = list(range(40, 60))
```

El punto crítico es el **split por clases, no por ejemplos**: las clases de test (40 a 59) jamás aparecen en entrenamiento. Eso es lo que hace que estemos midiendo few-shot de verdad — la red nunca vio esas clases y debe clasificarlas solo con el support del episodio.

El sampler de episodios:

```python
def sample_episode(classes, n_way, k_shot, q_query, X, Y):
    """Construye un episodio N-way K-shot.

    Devuelve:
      support_x: (n_way*k_shot, 1, 28, 28)
      support_y: (n_way*k_shot,)   etiquetas en [0, n_way)  -- relabel local
      query_x:   (n_way*q_query, 1, 28, 28)
      query_y:   (n_way*q_query,)
    """
    chosen = random.sample(classes, n_way)
    sx, sy, qx, qy = [], [], [], []
    for local_label, c in enumerate(chosen):
        idx = (Y == c).nonzero(as_tuple=True)[0].tolist()
        random.shuffle(idx)
        sup = idx[:k_shot]
        qry = idx[k_shot:k_shot + q_query]
        for i in sup:
            sx.append(X[i]); sy.append(local_label)
        for i in qry:
            qx.append(X[i]); qy.append(local_label)
    return (
        torch.stack(sx), torch.tensor(sy),
        torch.stack(qx), torch.tensor(qy),
    )


# Smoke test: 5-way 1-shot, 15 queries por clase
sx, sy, qx, qy = sample_episode(TRAIN_CLASSES, n_way=5, k_shot=1, q_query=15, X=X, Y=Y)
print(sx.shape, sy.shape, qx.shape, qy.shape)
# torch.Size([5, 1, 28, 28]) torch.Size([5]) torch.Size([75, 1, 28, 28]) torch.Size([75])
```

Notar el **relabel local**: dentro del episodio las clases reales (digamos 42, 7, 13, …) se renumeran a `0..n_way-1`. La red nunca usa el ID global de la clase — solo aprende a comparar. Eso es lo que permite que clasifique clases nuevas: la identidad de la clase está en el support, no en una salida fija.

| Tensor      | Shape                          | Significado |
|-------------|--------------------------------|-------------|
| `support_x` | `(N·K, 1, 28, 28)`             | $N$ clases × $K$ shots de imágenes |
| `support_y` | `(N·K,)`                       | etiquetas locales en `[0, N)` |
| `query_x`   | `(N·Q, 1, 28, 28)`             | $Q$ queries por clase |
| `query_y`   | `(N·Q,)`                       | etiquetas locales a predecir |

---

## 3. El encoder convolucional compartido

El embedding $f=g$ es la **Conv-4** que el paper popularizó y que se volvió el backbone estándar de los benchmarks de few-shot: cuatro bloques idénticos de conv $3\times3$ con 64 filtros, batch norm, ReLU y max-pool $2\times2$. Con entrada $28\times28$, cada pool divide el lado entre 2: $28 \to 14 \to 7 \to 3 \to 1$, y al final tenemos un feature map $1\times1\times64$ que aplanamos a un vector de 64 dimensiones.

```python
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class ConvEncoder(nn.Module):
    """Conv-4: el encoder compartido f = g. Mapea (B,1,28,28) -> (B, 64)."""
    def __init__(self, in_ch=1, hidden=64, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(in_ch, hidden),    # 28 -> 14
            conv_block(hidden, hidden),   # 14 -> 7
            conv_block(hidden, hidden),   # 7  -> 3
            conv_block(hidden, out_dim),  # 3  -> 1
        )

    def forward(self, x):
        h = self.net(x)              # (B, 64, 1, 1)
        return h.flatten(start_dim=1)  # (B, 64)


enc = ConvEncoder().to(device)
print(enc(sx.to(device)).shape)   # torch.Size([5, 64])
```

| Stage          | Shape           |
|----------------|-----------------|
| input          | `(B, 1, 28, 28)`|
| bloque 1       | `(B, 64, 14, 14)`|
| bloque 2       | `(B, 64, 7, 7)` |
| bloque 3       | `(B, 64, 3, 3)` |
| bloque 4       | `(B, 64, 1, 1)` |
| flatten        | `(B, 64)`       |

Que $f$ y $g$ **compartan pesos** es lo habitual en MN para Omniglot: un solo encoder embebe tanto el query como cada elemento del support. Mantenerlos separados es posible (y necesario con FCE, ver más abajo), pero compartirlos reduce parámetros y funciona bien en tareas no demasiado difíciles.

---

## 4. El attention kernel: coseno + softmax + voto sobre labels

Acá está el corazón del modelo. Recibimos los embeddings del query y del support, calculamos la similitud coseno query-vs-support, la pasamos por un softmax para obtener pesos que suman 1, y multiplicamos esos pesos por los labels del support codificados one-hot. El resultado es directamente $P(y \mid \hat{x}, S)$.

```python
def matching_logprobs(query_emb, support_emb, support_y, n_way):
    """Predicción no-paramétrica de Matching Networks.

    query_emb:   (Nq, d)
    support_emb: (Ns, d)
    support_y:   (Ns,) en [0, n_way)
    Devuelve log-probabilidades (Nq, n_way).
    """
    # 1) Normalizar para que el producto punto SEA la similitud coseno
    q = F.normalize(query_emb, p=2, dim=1)      # (Nq, d)
    s = F.normalize(support_emb, p=2, dim=1)    # (Ns, d)

    # 2) Coseno de cada query contra cada elemento del support
    cos = q @ s.t()                             # (Nq, Ns)  en [-1, 1]

    # 3) Softmax sobre el support -> pesos de atención que suman 1
    attn = F.softmax(cos, dim=1)                # (Nq, Ns)

    # 4) One-hot de los labels del support: (Ns, n_way)
    onehot = F.one_hot(support_y, num_classes=n_way).float()  # (Ns, n_way)

    # 5) Voto ponderado: pesos @ one-hot -> probabilidad por clase
    probs = attn @ onehot                       # (Nq, n_way), suma 1 por fila
    return torch.log(probs + 1e-8)              # log-probs para NLL estable
```

Cada paso, con sus shapes:

| Paso | Operación | Shape de salida |
|------|-----------|-----------------|
| 1 | `normalize(query)`, `normalize(support)` | `(Nq, d)`, `(Ns, d)` |
| 2 | `q @ s.t()` (coseno) | `(Nq, Ns)` |
| 3 | `softmax(dim=1)` (atención) | `(Nq, Ns)` |
| 4 | `one_hot(support_y)` | `(Ns, n_way)` |
| 5 | `attn @ onehot` (voto) | `(Nq, n_way)` |

El paso 5 es la implementación literal de $\hat{y}=\sum_i a(\hat{x},x_i)\,y_i$: el producto matriz `attn @ onehot` **agrupa** el peso de atención por clase. Si tres ejemplos del support pertenecen a la clase 2, sus tres pesos de atención se suman en la columna 2 del resultado. Cuando $K>1$ (varios shots por clase), esto reparte la responsabilidad de cada clase entre sus $K$ ejemplos de forma natural — sin promediar embeddings, a diferencia de Prototypical (lo discutiremos al final).

Un detalle de estabilidad: normalizamos **antes** de multiplicar, de modo que `q @ s.t()` es exactamente el coseno. No usamos `F.cosine_similarity` con broadcasting porque para `(Nq, Ns)` pares la versión matricial es mucho más eficiente y más clara.

---

## 5. Entrenamiento episódico

El objetivo de entrenamiento es maximizar la log-verosimilitud de las etiquetas del query condicionadas al support, promediando sobre episodios:

$$
\theta = \arg\max_\theta\; \mathbb{E}_{L\sim T}\Big[\, \mathbb{E}_{S,\,B \sim L}\big[\textstyle\sum_{(x,y)\in B} \log P_\theta(y\mid x, S)\big]\Big].
$$

En código, cada paso de entrenamiento es un episodio completo: muestreamos, embebemos support y query con el **mismo** encoder, calculamos las log-probs no-paramétricas y aplicamos NLL contra las etiquetas verdaderas del query.

```python
class MatchingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ConvEncoder()   # f = g compartido

    def forward(self, support_x, support_y, query_x, n_way):
        support_emb = self.encoder(support_x)   # (Ns, d)
        query_emb = self.encoder(query_x)       # (Nq, d)
        return matching_logprobs(query_emb, support_emb, support_y, n_way)


def train(model, n_way=5, k_shot=1, q_query=15, episodes=2000, lr=1e-3):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    running_acc = 0.0
    for ep in range(1, episodes + 1):
        sx, sy, qx, qy = sample_episode(TRAIN_CLASSES, n_way, k_shot, q_query, X, Y)
        sx, sy = sx.to(device), sy.to(device)
        qx, qy = qx.to(device), qy.to(device)

        logp = model(sx, sy, qx, n_way)          # (Nq, n_way)
        loss = F.nll_loss(logp, qy)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        acc = (logp.argmax(dim=1) == qy).float().mean().item()
        running_acc = 0.99 * running_acc + 0.01 * acc
        if ep % 200 == 0:
            print(f"ep {ep:4d}  loss={loss.item():.3f}  acc(ema)={running_acc:.3f}")
    return model


model = MatchingNet()
train(model, n_way=5, k_shot=1, q_query=15, episodes=2000)
```

Una corrida típica (los números exactos varían con el seed y el ruido del dataset sintético):

```
ep  200  loss=1.205  acc(ema)=0.512
ep  400  loss=0.842  acc(ema)=0.701
ep  600  loss=0.611  acc(ema)=0.803
ep  800  loss=0.494  acc(ema)=0.851
ep 1000  loss=0.402  acc(ema)=0.881
ep 1500  loss=0.318  acc(ema)=0.912
ep 2000  loss=0.281  acc(ema)=0.927
```

Notar que **no hay capa de clasificación final entrenable**: el optimizador solo toca el encoder. Todo el "saber clasificar" emerge de aprender un embedding tal que el voto por atención sobre el support dé la respuesta correcta. Esto es meta-aprendizaje en estado puro: el modelo aprende *a usar* un support set, no a memorizar clases.

---

## 6. Full Context Embeddings (FCE)

En la forma vanilla, $g(x_i)$ embebe cada elemento del support **miópicamente**: ignorando a los demás elementos de $S$. Pero la decisión final está condicionada a todo el support. Hay una inconsistencia: si dos elementos del support son muy parecidos, podría convenir embeberlos de forma que se **separen** para discriminar mejor; y el modo de embeber el query debería poder depender del support entero.

**Full Context Embeddings (FCE)** resuelve ambas cosas haciendo que los embeddings tomen $S$ como entrada:

- **$g(x_i, S)$ — biLSTM sobre el support.** Se trata al support como secuencia y se lo recorre con un LSTM bidireccional, sumando una skip connection a las features crudas: $g(x_i, S) = \overrightarrow{h}_i + \overleftarrow{h}_i + g'(x_i)$.
- **$f(\hat{x}, S)$ — LSTM con atención (read/process/write).** El query se refina en $K$ pasos, atendiendo sobre todo el support embebido en cada paso (esto es el bloque "Process" de *Order Matters*, Vinyals et al. 2015).

El hallazgo empírico del paper es matizado: **FCE no ayudó en Omniglot** (tarea fácil, se omitió de la tabla) pero **sí ayudó en miniImageNet** (~2 puntos porcentuales), que es lo bastante difícil para que condicionar al contexto valga la pena. Es complejidad con beneficio condicional.

Dejamos la versión vanilla como la principal y mostramos FCE-$g$ (el biLSTM sobre el support) como **extensión opcional**, que es la parte más sencilla y la que más se entiende:

```python
class FCESupportEncoder(nn.Module):
    """g(x_i, S): refina los embeddings del support con un biLSTM + skip.
    Extensión opcional sobre la Conv-4. d debe ser par para sumar ambas direcciones."""
    def __init__(self, encoder, d=64):
        super().__init__()
        self.encoder = encoder
        self.lstm = nn.LSTM(d, d // 2, batch_first=True, bidirectional=True)

    def forward(self, support_x):
        g_raw = self.encoder(support_x)          # (Ns, d)
        seq = g_raw.unsqueeze(0)                 # (1, Ns, d) -- el support como UNA secuencia
        h, _ = self.lstm(seq)                    # (1, Ns, d)  (d//2 * 2 direcciones)
        return (h.squeeze(0) + g_raw)            # skip connection: contexto + features crudas
```

El costo de FCE: introduce LSTMs, hace al cómputo **secuencial** sobre el support (cuando conceptualmente $S$ es un conjunto sin orden), y multiplica el costo de inferencia. En tareas como nuestro dataset sintético o Omniglot, la versión vanilla ya satura — no vale la pena pagar FCE. Lo dejamos documentado para cuando la tarea sea fine-grained y difícil.

---

## 7. La conexión profunda con Transformers (la joya)

Acá está lo que hace que este capítulo merezca su lugar en una clase de meta-aprendizaje que mira hacia los LLMs. Reescribamos el attention kernel de Matching Networks al lado de la atención de *Attention Is All You Need* (2017):

$$
\underbrace{\hat{y} = \sum_{i} \operatorname{softmax}_i\big(c(f(\hat{x}), g(x_i))\big)\, y_i}_{\text{Matching Networks, 2016}}
\qquad\Longleftrightarrow\qquad
\underbrace{\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\!\Big(\tfrac{QK^\top}{\sqrt{d}}\Big)\,V}_{\text{Transformer, 2017}}
$$

El mapeo es **literal**, término a término:

| Matching Networks (2016) | Transformer (2017) | Rol |
|--------------------------|--------------------|-----|
| $f(\hat{x})$ — embedding del query | $Q$ — query | qué estoy buscando |
| $g(x_i)$ — embedding del support $i$ | $K$ — key | contra qué comparo |
| $y_i$ — etiqueta one-hot del support $i$ | $V$ — value | qué recupero si matchea |
| $c(\cdot,\cdot)$ coseno | $\tfrac{QK^\top}{\sqrt{d}}$ producto punto escalado | la métrica de similitud |
| $\operatorname{softmax}$ sobre el support | $\operatorname{softmax}$ sobre las posiciones | normalización a pesos |
| suma ponderada de labels | suma ponderada de values | la agregación |

Matching Networks **es** una operación de **cross-attention sobre una memoria etiquetada no-paramétrica**. La diferencia con el self-attention de un Transformer es solo qué juega el rol de los values: en MN son las etiquetas one-hot (clasificación); en un Transformer son representaciones aprendidas que se propagan por la red. La maquinaria —comparar un query contra un conjunto de keys, normalizar las similitudes con softmax, agregar los values ponderadamente— es idéntica.

Y la idea va más allá de la coincidencia matemática. El principio de fondo es: **clasificar (o computar) es atender sobre un conjunto de ejemplos de referencia y agregar sus valores**. Ese principio reaparece, escalado y sin etiquetas explícitas, en el **in-context learning** de los LLMs. Cuando le das a un LLM unos ejemplos en el prompt (`entrada → salida`, `entrada → salida`, …) y luego una entrada nueva, el modelo "aprende" de esos ejemplos **sin actualizar un solo peso**: los ejemplos del prompt son el support set, la entrada nueva es el query, y la atención del Transformer hace el voto ponderado. Es exactamente el espíritu no-paramétrico de Matching Networks, ocho años antes de que se volviera el modo dominante de usar modelos de lenguaje.

Dicho de otro modo: si entendiste por qué `attn @ onehot` clasifica un dígito a partir de cinco ejemplos sin reentrenar nada, ya entendiste la mecánica esencial por la que GPT clasifica reseñas a partir de tres ejemplos en su prompt. El few-shot prompting de los LLMs es Matching Networks con values aprendidos y un support set que vive en la ventana de contexto.

---

## 8. Evaluación 5-way 1-shot / 5-shot con intervalos

La práctica estándar en few-shot es reportar la accuracy promedio sobre muchos episodios de test (clases **nunca vistas**), con su intervalo de confianza al 95%. Un solo episodio tiene altísima varianza; lo que importa es la media sobre cientos de tareas.

```python
@torch.no_grad()
def evaluate(model, n_way=5, k_shot=1, q_query=15, episodes=600):
    model.eval()
    accs = []
    for _ in range(episodes):
        sx, sy, qx, qy = sample_episode(TEST_CLASSES, n_way, k_shot, q_query, X, Y)
        sx, sy = sx.to(device), sy.to(device)
        qx, qy = qx.to(device), qy.to(device)
        logp = model(sx, sy, qx, n_way)
        acc = (logp.argmax(dim=1) == qy).float().mean().item()
        accs.append(acc)
    accs = torch.tensor(accs)
    mean = accs.mean().item()
    # IC 95% ~= 1.96 * desviacion_estandar / sqrt(n_episodios)
    ci95 = 1.96 * accs.std().item() / math.sqrt(len(accs))
    return mean, ci95


for k in (1, 5):
    m, ci = evaluate(model, n_way=5, k_shot=k, q_query=15, episodes=600)
    print(f"5-way {k}-shot: {100*m:.1f}% ± {100*ci:.1f}%")
```

Salida típica:

```
5-way 1-shot: 86.4% ± 0.8%
5-way 5-shot: 94.1% ± 0.5%
```

Dos lecturas. Primero, **5-shot supera a 1-shot**: con cinco ejemplos por clase el voto por atención es más robusto al ruido de un solo ejemplo atípico. Segundo, fíjate que **el modelo se entrenó en 1-shot pero evaluamos también en 5-shot sin reentrenar** — porque MN es no-paramétrico, cambiar el número de shots es solo cambiar el tamaño del support, no la arquitectura. (Aun así, lo limpio es entrenar en el mismo régimen en que evaluarás; ver gotchas.) El intervalo de confianza estrecho sobre 600 episodios nos dice que la media es confiable, no un golpe de suerte de un episodio.

---

## 9. Gotchas

Tres trampas que valen oro al implementar Matching Networks:

**Estabilidad del softmax de cosenos.** El coseno está acotado en $[-1, 1]$, así que las diferencias de logits **antes** del softmax son chicas (a lo sumo 2). Eso hace que la atención sea relativamente "blanda": incluso el mejor match recibe un peso modesto, y la señal de gradiente puede ser débil al inicio. No es un bug, es la física de la fórmula. Por eso normalizamos los embeddings explícitamente con `F.normalize` antes del producto punto — si te olvidas y multiplicas vectores sin normalizar, ya no estás computando coseno sino producto punto crudo, y los embeddings de norma grande dominan el voto.

**Temperatura.** La magnitud efectiva del coseno actúa como una **temperatura** del softmax. El paper original no introduce temperatura explícita, pero trabajos posteriores de few-shot añadieron un factor $\tau$: $a(\hat{x},x_i) \propto \exp(\tau \cdot c(f(\hat{x}), g(x_i)))$. Con $\tau$ grande el softmax se endurece hacia un argmax (kNN con $k=1$); con $\tau$ chico se ablanda hacia un promedio uniforme (KDE de banda ancha). Si tu modelo no aprende, una temperatura aprendible (`self.tau = nn.Parameter(torch.tensor(10.0))`) suele desbloquear el entrenamiento, porque le da al modelo control sobre cuán "decisivo" es el voto. Es el ajuste más rentable que puedes hacer sobre la versión vanilla.

**Costo cuadrático sobre el support.** La atención compara el query contra **cada** elemento del support: el costo es $O(N_q \cdot N_s \cdot d)$. Para support sets chicos (5-way 5-shot = 25 ejemplos) es trivial, pero **no escala** a support grande — y con FCE es peor, porque el attLSTM hace $K$ reads sobre todo $S$ y el biLSTM lo procesa secuencialmente. Esta es exactamente la razón por la que, en sistemas reales de matching (recuperación, MDM de pacientes), no se atiende sobre toda la base: se usa un *blocker* que reduce los candidatos por similitud aproximada antes de aplicar el scorer caro.

---

## 10. Limitaciones y comparación con Prototypical Networks

Matching Networks atiende sobre **todos** los puntos individuales del support. Su descendiente directo, **Prototypical Networks** (Snell et al., 2017, ver camino 02), simplifica el modelo promediando los embeddings de cada clase en un único **prototipo** $c_n = \frac{1}{|S_n|}\sum_i g(x_i)$ y clasificando por **distancia euclídea** al prototipo más cercano:

| Aspecto | Matching Networks | Prototypical Networks |
|---------|-------------------|------------------------|
| Sobre qué se atiende | cada punto del support individualmente | un centroide promedio por clase |
| Métrica | coseno | distancia euclídea (cuadrada) |
| Costo de inferencia | $O(N_q \cdot N_s)$ comparaciones | $O(N_q \cdot N)$ ($N \ll N_s$ con $K>1$) |
| FCE | opcional, ayuda en tareas difíciles | no usa (más simple) |
| Comportamiento con $K>1$ | reparte voto entre los $K$ ejemplos | colapsa los $K$ en un centroide |

La intuición de la diferencia: con $K=1$ (un solo shot por clase) ambos son casi equivalentes — el prototipo de una clase con un solo ejemplo *es* ese ejemplo. La divergencia aparece con $K>1$. MN deja que cada ejemplo vote por separado, lo que es más expresivo (puede capturar multimodalidad dentro de una clase) pero más sensible a outliers y más caro. Prototypical promedia, lo que regulariza (un outlier se diluye en el centroide), simplifica el cómputo y, empíricamente, **suele ganar** en miniImageNet — una crítica implícita a la complejidad del coseno + FCE de MN.

Las limitaciones que el propio paper reconoce, y que conviene tener presentes:

- **Costo computacional creciente con el support** (gotcha 3): no escala a memorias grandes.
- **FCE complica la arquitectura** con beneficio solo condicional (Sección 6).
- **Degradación bajo shift de distribución de tareas.** En el experimento $L_{dogs}$ del paper, MN *empeora* al pasar de clases dispares (en train) a razas de perro muy similares (en test). La lección es el principio rector del paper: **"test and train conditions must match"** — si entrenas muestreando tareas fáciles/heterogéneas pero evalúas en tareas fine-grained, el modelo sufre. En salud esto es directo: un clasificador few-shot entrenado sobre casos heterogéneos puede fallar al distinguir dos lesiones de aspecto casi idéntico. La distribución de tareas de entrenamiento debe reflejar la dificultad real del despliegue.

A pesar de estas limitaciones, Matching Networks sigue siendo **didácticamente irremplazable**: es el modelo más limpio para ver que "clasificar es atender sobre ejemplos de referencia y agregar sus valores", y ese insight es la semilla de los Transformers y del in-context learning que dominan hoy.

---

## Cross-links

- Camino anterior: [04 - Siamese y verificación](/clases/clase-26/practica/04-siamese-verificacion)
- Paper original: [Matching Networks (Vinyals et al., 2016)](/papers/matching-networks-vinyals-2016)
- Fundamento transversal: [Metric Learning](/fundamentos/metric-learning)
- Fundamento transversal: [Self-Attention](/fundamentos/self-attention)
- Fundamento transversal: [In-Context Learning](/fundamentos/in-context-learning)
- Teoría de la clase: [Clase 26 — Meta-aprendizaje (teoría)](../teoria)

Volver al [hub de práctica](..) o a la [Clase 26](../..).
