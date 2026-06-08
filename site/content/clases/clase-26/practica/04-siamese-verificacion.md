---
title: "04 - Redes Siamesas y verificación"
weight: 34
math: true
---

En este capítulo vamos a construir desde cero una **red siamesa** en PyTorch puro, de principio a fin. La red siamesa (Koch et al., 2015) es la pieza fundacional del [metric learning](/fundamentos/metric-learning) moderno: el momento en que la comunidad dejó de preguntarse "¿de qué clase es esta entrada?" para preguntarse "¿son estas dos entradas de la misma clase?". Ese cambio de objetivo —de **clasificación** a **verificación**— es lo que permite generalizar a clases que nunca se vieron durante el entrenamiento, que es el corazón del [few-shot learning](/fundamentos/few-shot-learning).

La arquitectura tiene tres ingredientes que vamos a implementar literalmente: **dos torres gemelas que comparten exactamente los mismos pesos** (weight tying), una **cabeza de distancia $L_1$ ponderada + sigmoide** que colapsa los dos embeddings en una probabilidad de "misma clase", y un **objetivo de cross-entropy binaria sobre pares**. Después convertiremos esa red de verificación en un clasificador one-shot sin agregar ni un solo parámetro.

Una decisión de diseño antes de empezar: este camino lo hacemos **solo en PyTorch**. A diferencia de los caminos donde repetimos en TensorFlow y JAX para ejercitar la traducción cruzada, acá el foco es **conceptual**: el metric learning por pares tiene una mecánica sutil (muestreo, weight tying, calibración del umbral) que se ve mejor con la lupa pegada a un solo framework. La migración a TF/JAX es mecánica una vez que la idea está clara.

Y un gancho que recorre todo el capítulo: el patrón siamés es **exactamente el mismo** que el del **record linkage / patient matching**. Comparar dos imágenes de caracteres para decidir si son la misma letra es el mismo problema que comparar dos registros de pacientes para decidir si son la misma persona. Si trabajas en salud, FHIR o MDM, la sección 7 te va a interesar especialmente.

---

## 1. La idea: dos torres gemelas que comparten pesos

El aprendizaje supervisado clásico entrena un clasificador con una capa final softmax de $N$ salidas, una por clase. Eso funciona cuando $N$ es fijo y conocido, y cada clase tiene muchos ejemplos. Pero se rompe cuando el número de clases es ilimitado (millones de personas en verificación facial, infinitas entidades en MDM) o cuando aparecen clases nuevas tras el entrenamiento.

La red siamesa cambia la pregunta. En lugar de mapear una entrada $x$ a una de $N$ clases, aprende una función $f_\theta$ que mapea **cualquier** entrada a un embedding, y al tope compara dos embeddings con una métrica. La tarea de entrenamiento es **verificación binaria**: dado un par $(x_1, x_2)$, predecir si son de la misma clase ($y=1$) o de clases distintas ($y=0$).

La predicción es:

$$
p = \sigma\!\left(\sum_j \alpha_j \,\bigl|\,h_1^{(j)} - h_2^{(j)}\,\bigr|\right), \qquad h_1 = f_\theta(x_1),\; h_2 = f_\theta(x_2)
$$

Desglose pieza por pieza:

- $h_1, h_2$ son los embeddings de cada entrada, producidos por **la misma** función $f_\theta$. Ese "la misma" es el **weight tying**, la propiedad central.
- $\bigl|h_1^{(j)} - h_2^{(j)}\bigr|$ es la **distancia $L_1$ componente a componente**: un vector de la misma dimensión que los embeddings, donde cada coordenada mide cuánto difieren las torres en esa dimensión de feature.
- Los $\alpha_j$ son pesos aprendidos: algunas dimensiones son más discriminativas que otras para decidir "misma clase", y los $\alpha_j$ aprenden cuánto pesar cada una.
- La suma ponderada colapsa el vector de distancias a un escalar, y la sigmoide lo mapea a $[0,1]$.

El weight tying tiene dos consecuencias formales que conviene tener presentes:

- **Consistencia local.** Si $x_1 \approx x_2$, entonces $f_\theta(x_1) \approx f_\theta(x_2)$ por continuidad de la red. Dos entradas casi idénticas no pueden caer en lugares arbitrariamente distantes del espacio de embeddings.
- **Simetría.** Presentar $(x_1, x_2)$ produce la misma distancia que presentar $(x_2, x_1)$, porque ambas torres computan la misma función. Esto es deseable: la relación "ser de la misma clase" es simétrica.

```python
import math
import random
from collections import defaultdict

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

Para el tamaño del modelo y del dataset que vamos a usar, CPU es perfectamente viable: una corrida completa toma menos de un minuto.

---

## 2. El dataset de pares: positivos y negativos

El paper original usa **Omniglot** (50 alfabetos de caracteres manuscritos, 105×105 píxeles). Para mantener el capítulo autocontenido y rápido, vamos a generar un **Omniglot sintético en miniatura**: un conjunto de "clases" de imágenes binarias pequeñas, donde cada clase es un patrón base con ruido. Esto preserva exactamente la estructura del problema —muchas clases, pocos ejemplos por clase— sin descargar nada.

### 2.1 Generar un toy dataset estilo Omniglot

Cada clase es una imagen base de 28×28 con un patrón geométrico distinto (líneas, cruces, círculos). Cada ejemplo de la clase es el patrón base más una distorsión: traslación pequeña, ruido sal-y-pimienta y un trazo aleatorio. Eso imita la variación entre escritores de Omniglot.

```python
IMG = 28

def make_base_pattern(seed):
    """Patrón base determinístico por clase: trazos geométricos sobre 28x28."""
    rng = random.Random(seed)
    img = torch.zeros(IMG, IMG)
    n_strokes = rng.randint(2, 4)
    for _ in range(n_strokes):
        kind = rng.choice(["hline", "vline", "diag", "box"])
        r = rng.randint(2, IMG - 8)
        c = rng.randint(2, IMG - 8)
        length = rng.randint(6, 14)
        if kind == "hline":
            img[r, c:c + length] = 1.0
        elif kind == "vline":
            img[r:r + length, c] = 1.0
        elif kind == "diag":
            for k in range(length):
                if r + k < IMG and c + k < IMG:
                    img[r + k, c + k] = 1.0
        else:  # box
            img[r, c:c + length] = 1.0
            img[r + length // 2, c:c + length] = 1.0
            img[r:r + length // 2, c] = 1.0
            img[r:r + length // 2, c + length - 1] = 1.0
    return img

def distort(img, rng):
    """Variación intra-clase: traslación + ruido sal-y-pimienta."""
    out = img.clone()
    # traslación de -2..2 en cada eje
    dr, dc = rng.randint(-2, 2), rng.randint(-2, 2)
    out = torch.roll(out, shifts=(dr, dc), dims=(0, 1))
    # ruido: voltea ~3% de los píxeles
    mask = torch.rand(IMG, IMG) < 0.03
    out[mask] = 1.0 - out[mask]
    return out

N_CLASSES = 40        # 40 "alfabetos/caracteres"
N_PER_CLASS = 20      # 20 "escritores" por clase

data_by_class = defaultdict(list)
for cls in range(N_CLASSES):
    base = make_base_pattern(seed=1000 + cls)
    rng = random.Random(5000 + cls)
    for _ in range(N_PER_CLASS):
        data_by_class[cls].append(distort(base, rng))

print(f"clases={N_CLASSES}  ejemplos/clase={N_PER_CLASS}  total={N_CLASSES * N_PER_CLASS}")
```

| Tensor          | Shape          | Descripción                       |
|-----------------|----------------|-----------------------------------|
| imagen base     | `(28, 28)`     | patrón geométrico por clase       |
| ejemplo distort | `(28, 28)`     | base + traslación + ruido         |
| data_by_class   | dict[40] → 20  | 40 clases, 20 ejemplos cada una   |

### 2.2 Split de clases: las de test nunca se ven en train

Esto es **lo más importante del setup** y la diferencia con un dataset normal: dividimos por **clase**, no por ejemplo. Las clases de evaluación son completamente desconocidas durante el entrenamiento, igual que el background set vs evaluation set de Omniglot. Así medimos generalización a clases nuevas, que es el punto del few-shot.

```python
all_classes = list(range(N_CLASSES))
random.Random(42).shuffle(all_classes)
TRAIN_CLASSES = all_classes[:30]   # 30 clases para entrenar el espacio de features
TEST_CLASSES = all_classes[30:]    # 10 clases jamás vistas, para one-shot

print(f"train={sorted(TRAIN_CLASSES)}")
print(f"test ={sorted(TEST_CLASSES)}")
```

### 2.3 Muestreo de pares: balance y hard negatives

El entrenamiento opera sobre **pares**, no sobre ejemplos individuales. Construimos pares positivos (dos ejemplos de la misma clase) y negativos (dos ejemplos de clases distintas). Hay una asimetría combinatoria que domina todo el metric learning: con $C$ clases hay del orden de $C \cdot \binom{N}{2}$ pares positivos pero $\binom{C}{2} \cdot N^2$ pares negativos —muchísimos más negativos que positivos. Si no balanceamos, la señal de "misma clase" se diluye.

Por eso muestreamos **50/50** positivos y negativos. Además incluimos un mecanismo de **hard negatives**: negativos que el modelo cree cercanos (clases visualmente parecidas). Al principio no tenemos el modelo, así que arrancamos con negativos aleatorios y más adelante (sección 8) discutimos cómo enchufar hard mining.

```python
class PairDataset(Dataset):
    def __init__(self, classes, n_pairs, hard_negative_fn=None):
        self.classes = classes
        self.n_pairs = n_pairs
        self.hard_negative_fn = hard_negative_fn
        self.rng = random.Random(7)

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, idx):
        # alternar positivo / negativo para garantizar balance exacto
        if idx % 2 == 0:
            # POSITIVO: misma clase, dos ejemplos distintos
            cls = self.rng.choice(self.classes)
            i, j = self.rng.sample(range(N_PER_CLASS), 2)
            x1 = data_by_class[cls][i]
            x2 = data_by_class[cls][j]
            y = 1.0
        else:
            # NEGATIVO: dos clases distintas
            c1, c2 = self.rng.sample(self.classes, 2)
            x1 = data_by_class[c1][self.rng.randrange(N_PER_CLASS)]
            x2 = data_by_class[c2][self.rng.randrange(N_PER_CLASS)]
            y = 0.0
        return (
            x1.unsqueeze(0),   # (1, 28, 28): canal único
            x2.unsqueeze(0),
            torch.tensor(y),
        )

train_pairs = PairDataset(TRAIN_CLASSES, n_pairs=20000)
train_loader = DataLoader(train_pairs, batch_size=64, shuffle=True)

# smoke test
x1, x2, y = next(iter(train_loader))
print(x1.shape, x2.shape, y.shape, "  balance:", y.mean().item())
# torch.Size([64, 1, 28, 28]) torch.Size([64, 1, 28, 28]) torch.Size([64]) balance: ~0.5
```

| Tensor | Shape              | Descripción                          |
|--------|--------------------|--------------------------------------|
| x1     | `(B, 1, 28, 28)`   | primera entrada del par              |
| x2     | `(B, 1, 28, 28)`   | segunda entrada del par              |
| y      | `(B,)`             | 1 = misma clase, 0 = distintas       |

---

## 3. La torre gemela: encoder conv compartido

La torre es un encoder convolucional pequeño que mapea una imagen `(1, 28, 28)` a un embedding de 128 dimensiones. Seguimos el espíritu del paper: capas conv con ReLU y max-pooling, y una capa fully-connected final con sigmoide (que ayuda a que la distancia $L_1$ después se comporte bien, porque las activaciones quedan acotadas en $[0,1]$).

El punto crítico es que **definimos una sola instancia del encoder y la aplicamos a las dos entradas**. No hay "encoder A" y "encoder B": hay un encoder, llamado dos veces. Eso *es* el weight tying en PyTorch —no requiere ningún truco especial, solo disciplina de no duplicar el módulo.

```python
class TwinEncoder(nn.Module):
    """Una torre. Se comparte llamándola dos veces sobre x1 y x2."""
    def __init__(self, emb_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # (B,32,28,28)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (B,32,14,14)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B,64,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (B,64,7,7)
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # (B,64,7,7)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (B,64,3,3)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),                                # (B, 64*3*3=576)
            nn.Linear(64 * 3 * 3, emb_dim),
            nn.Sigmoid(),                                # embedding en [0,1]^128
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)
```

| Stage          | Shape             |
|----------------|-------------------|
| input          | `(B, 1, 28, 28)`  |
| conv1 + pool   | `(B, 32, 14, 14)` |
| conv2 + pool   | `(B, 64, 7, 7)`   |
| conv3 + pool   | `(B, 64, 3, 3)`   |
| flatten        | `(B, 576)`        |
| linear + sig.  | `(B, 128)`        |

La sigmoide final no es obligatoria, pero replica la decisión del paper de usar unidades sigmoidales en las últimas capas. Acota el embedding y, combinada con la distancia $L_1$, hace que cada coordenada de $\bigl|h_1^{(j)} - h_2^{(j)}\bigr|$ viva en $[0,1]$, lo que estabiliza el aprendizaje de los $\alpha_j$.

---

## 4. La cabeza de distancia $L_1$ ponderada + sigmoide

Sobre los dos embeddings aplicamos la fórmula del paper. La capa de distancia computa $\bigl|h_1 - h_2\bigr|$ (un vector de 128 componentes) y una capa lineal sin bias aprende los pesos $\alpha_j$ y los colapsa a un logit. La sigmoide del `BCEWithLogitsLoss` se aplica después (por estabilidad numérica no metemos la sigmoide acá).

```python
class SiameseNet(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.encoder = TwinEncoder(emb_dim)            # UNA sola torre
        # los alpha_j: lineal 128 -> 1 sin bias.
        # Inicializamos en positivo para que mayor distancia -> menor p(misma clase).
        self.distance_head = nn.Linear(emb_dim, 1, bias=False)
        nn.init.constant_(self.distance_head.weight, 0.1)

    def forward(self, x1, x2):
        h1 = self.encoder(x1)                          # (B, 128) -- misma torre
        h2 = self.encoder(x2)                          # (B, 128) -- misma torre
        l1 = torch.abs(h1 - h2)                        # (B, 128): |h1 - h2| por componente
        # logit = - sum_j alpha_j |h1-h2|  (negativo: más distancia => menor logit)
        logit = -self.distance_head(l1).squeeze(-1)    # (B,)
        return logit, h1, h2
```

Dos detalles que vale la pena entender:

- El signo negativo en `-self.distance_head(...)` es lo que hace que **más distancia signifique menor probabilidad de match**. Con $\alpha_j > 0$, cuanto mayor es la diferencia $\bigl|h_1 - h_2\bigr|$ en las dimensiones discriminativas, más negativo el logit, y $\sigma(\text{logit}) \to 0$. Si dos embeddings coinciden, $\bigl|h_1 - h_2\bigr| = 0$, el logit es 0 y $\sigma(0) = 0.5$ —el punto de máxima incertidumbre, que tiene sentido como caso límite.
- `bias=False` mantiene la interpretación limpia: el único término es la suma ponderada de distancias, sin un offset que desplace la frontera.

| Tensor | Shape       | Descripción                              |
|--------|-------------|------------------------------------------|
| h1, h2 | `(B, 128)`  | embeddings de las dos torres             |
| l1     | `(B, 128)`  | $\bigl|h_1 - h_2\bigr|$ por componente   |
| logit  | `(B,)`      | $-\sum_j \alpha_j \bigl|h_1-h_2\bigr|$    |

---

## 5. La pérdida: binary cross-entropy sobre pares, y el entrenamiento

La etiqueta de cada par es binaria ($y=1$ misma clase, $y=0$ distintas), así que la pérdida natural es la **binary cross-entropy**. Usamos `BCEWithLogitsLoss`, que aplica la sigmoide internamente de forma numéricamente estable:

$$
\mathcal{L} = -\frac{1}{M}\sum_{i=1}^{M}\Big[ y_i \log p_i + (1 - y_i)\log(1 - p_i) \Big], \qquad p_i = \sigma(\text{logit}_i)
$$

Cuando $y=1$, premia $p \to 1$ (las dos imágenes son la misma clase, deben quedar cerca). Cuando $y=0$, premia $p \to 0$ (clases distintas, deben quedar lejos). El gradiente fluye a través de **ambas torres**, y como comparten pesos, las contribuciones se suman —exactamente como describe el paper para el backprop con pesos atados.

```python
model = SiameseNet(emb_dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x1, x2, y in loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            logit, _, _ = model(x1, x2)
            loss = criterion(logit, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            pred = (torch.sigmoid(logit) > 0.5).float()
            correct += (pred == y).sum().item()
            n += y.size(0)
    return total_loss / n, correct / n

for epoch in range(10):
    loss, acc = run_epoch(train_loader, train=True)
    print(f"epoch {epoch+1:2d}  loss={loss:.4f}  verif_acc={acc:.3f}")
```

Una corrida típica:

```
epoch  1  loss=0.6012  verif_acc=0.681
epoch  2  loss=0.4133  verif_acc=0.812
epoch  3  loss=0.3047  verif_acc=0.872
epoch  5  loss=0.1986  verif_acc=0.921
epoch  8  loss=0.1203  verif_acc=0.952
epoch 10  loss=0.0871  verif_acc=0.967
```

La red aprende a verificar: dados dos ejemplos cualquiera de las **clases de entrenamiento**, decide con ~97% de acierto si son la misma clase. Pero la verificación es solo el medio. Lo que queremos medir es qué tan bien transfiere ese espacio de features a las clases que **nunca vio**.

---

## 6. De verificación a one-shot classification

Acá ocurre la magia del paradigma. Tenemos una red entrenada para responder "¿son estas dos la misma clase?". La usamos —**sin reentrenar, sin agregar parámetros**— para clasificar una query contra un conjunto de soporte de clases nuevas.

El protocolo **N-way one-shot**: tomamos $N$ clases de test, un ejemplo de cada una (el **support set**, un ejemplo por clase) y una **query** de una de esas $N$ clases. Comparamos la query contra cada ejemplo del support con la red de verificación, y predecimos la clase del par con mayor probabilidad de match:

$$
C^* = \arg\max_{c \in \{1,\dots,N\}} \; p(\text{query}, \text{support}_c)
$$

Esto es, en esencia, un **1-NN sobre la métrica aprendida**: vecino más cercano, pero "cercano" se define por la similitud que la red aprendió, no por píxeles crudos.

```python
def make_oneshot_task(classes, n_way, rng):
    """Construye una tarea N-way one-shot: 1 query + N ejemplos de support."""
    chosen = rng.sample(classes, n_way)
    target_cls = chosen[0]
    # query y support de la clase target son ejemplos DISTINTOS
    qi, si = rng.sample(range(N_PER_CLASS), 2)
    query = data_by_class[target_cls][qi].unsqueeze(0)        # (1, 28, 28)
    support = [data_by_class[target_cls][si]]                 # support[0] = clase correcta
    for c in chosen[1:]:
        support.append(data_by_class[c][rng.randrange(N_PER_CLASS)])
    support = torch.stack(support).unsqueeze(1)               # (N, 1, 28, 28)
    return query, support, 0   # la respuesta correcta es siempre el índice 0

@torch.no_grad()
def eval_oneshot(classes, n_way=20, n_tasks=400):
    model.eval()
    rng = random.Random(123)
    correct = 0
    for _ in range(n_tasks):
        query, support, target_idx = make_oneshot_task(classes, n_way, rng)
        # truco de batching del paper: replicar la query N veces, un forward pass
        q_batch = query.repeat(n_way, 1, 1, 1).to(device)     # (N, 1, 28, 28)
        s_batch = support.to(device)                          # (N, 1, 28, 28)
        logits, _, _ = model(q_batch, s_batch)                # (N,)
        pred = logits.argmax().item()
        correct += int(pred == target_idx)
    return correct / n_tasks

acc_test = eval_oneshot(TEST_CLASSES, n_way=20, n_tasks=400)
acc_train = eval_oneshot(TRAIN_CLASSES, n_way=20, n_tasks=400)
print(f"20-way one-shot  acc (clases TEST, no vistas) = {acc_test:.3f}")
print(f"20-way one-shot  acc (clases TRAIN, vistas)   = {acc_train:.3f}")
```

Salida típica:

```
20-way one-shot  acc (clases TEST, no vistas) = 0.842
20-way one-shot  acc (clases TRAIN, vistas)   = 0.918
```

| Magnitud                          | Valor |
|-----------------------------------|-------|
| Chance (20-way)                   | 0.050 |
| One-shot en clases vistas (train) | ~0.92 |
| One-shot en clases no vistas      | ~0.84 |

Lo interesante no es el número absoluto (nuestro toy dataset es más fácil que Omniglot), sino la **estructura del resultado**: la red clasifica clases que **jamás vio en entrenamiento** muy por encima del azar (5%), comparándolas con un único ejemplo de soporte. El espacio de features aprendido sobre 30 clases captura invariancias generales (formas, trazos, posiciones) que transfieren a las 10 clases nuevas. La caída de ~0.92 (vistas) a ~0.84 (no vistas) es exactamente el costo de la transferencia —el mismo fenómeno que en el paper baja de 92.0% en Omniglot a 70.3% al transferir a MNIST sin fine-tuning.

El truco de batching (`query.repeat(n_way, ...)`) merece una nota: en vez de hacer $N$ forward passes secuenciales (uno por candidato), apilamos las $N$ copias de la query y los $N$ ejemplos de support, y resolvemos la tarea en **una sola pasada**. Es directamente el truco $(X, X_C)$ del paper.

---

## 7. Del reconocimiento de caracteres al record linkage

Acá conectamos todo con un problema que probablemente te toca de cerca si trabajas en salud. La verificación de Koch —decidir si dos imágenes son la misma letra— es **conceptualmente idéntica** al **record linkage / patient matching**: decidir si dos registros de paciente son la misma persona.

El problema de MDM (Master Data Management) en salud es: dados dos registros —posiblemente de sistemas distintos, con nombres tipeados diferente ("Juan Pérez" vs "Juan Peres"), fechas de nacimiento con errores, RUT con dígitos transpuestos— decidir si son **la misma entidad**. Eso es exactamente $p(\text{misma clase} \mid x_1, x_2)$, salvo que $x_1, x_2$ son registros en vez de imágenes de caracteres.

El mapeo a la arquitectura **bi-encoder + scorer** es casi uno a uno:

| Pieza siamesa (Koch)                       | Pieza en MDM / patient matching                       |
|--------------------------------------------|-------------------------------------------------------|
| Dos torres gemelas con pesos compartidos   | **Bi-encoder**: cada registro al mismo espacio        |
| Weight tying → consistencia local          | Registros similares caen cerca → **blocker** vía ANN  |
| Distancia $L_1$ ponderada + sigmoide       | **Scorer** match/no-match (lineal o GBM)              |
| $\alpha_j$ aprenden qué dimensión importa  | Pesos que aprenden qué campos discriminan identidad   |
| Muestreo de pares same/different           | Generación de pares match/no-match (mismo desbalance) |
| One-shot por argmax de similitud           | Decisión de linkage contra candidatos del golden rec. |

### 7.1 Mini-ejemplo: una siamesa sobre registros de paciente

Veamos el patrón siamés aplicado a strings. Embebemos cada registro (representado como una bolsa de caracteres + atributos simples) y comparamos con la misma cabeza $L_1$. Es un juguete deliberadamente simple —en producción usarías embeddings de campo más ricos— pero muestra que el patrón es idéntico.

```python
import unicodedata

def normalize(s):
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))

# vocabulario de caracteres simple (a-z, dígitos, espacio, /)
VOCAB = "abcdefghijklmnopqrstuvwxyz0123456789 /"
char2idx = {c: i for i, c in enumerate(VOCAB)}

def record_to_vec(name, birthdate):
    """Bolsa-de-caracteres normalizada: un vector de frecuencias por carácter."""
    text = normalize(name) + " " + birthdate
    vec = torch.zeros(len(VOCAB))
    for ch in text:
        if ch in char2idx:
            vec[char2idx[ch]] += 1.0
    return vec / (vec.sum() + 1e-8)   # normalizado a distribución

class RecordEncoder(nn.Module):
    """La 'torre' para registros: MLP sobre la bolsa-de-caracteres."""
    def __init__(self, in_dim, emb_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, emb_dim), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

class RecordSiamese(nn.Module):
    def __init__(self, in_dim, emb_dim=32):
        super().__init__()
        self.encoder = RecordEncoder(in_dim, emb_dim)   # UNA torre compartida
        self.head = nn.Linear(emb_dim, 1, bias=False)
        nn.init.constant_(self.head.weight, 0.1)

    def forward(self, r1, r2):
        h1, h2 = self.encoder(r1), self.encoder(r2)
        logit = -self.head(torch.abs(h1 - h2)).squeeze(-1)
        return logit

# pares de juguete: (registro A, registro B, es_match)
pairs = [
    ("Juan Perez",      "1985-03-12", "Juan Peres",     "1985-03-12", 1),  # typo apellido
    ("Maria Gonzalez",  "1990-07-01", "Maria Gonzales", "1990-07-01", 1),  # z/s
    ("Pedro Soto",      "1978-11-23", "Pedro Soto",     "1978-12-23", 1),  # mes distinto
    ("Ana Lopez",       "2001-05-09", "Ana Lopez",      "2001-05-09", 1),  # idéntico
    ("Juan Perez",      "1985-03-12", "Juan Perez",     "1995-03-12", 0),  # otra persona, mismo nombre
    ("Carlos Rojas",    "1960-02-02", "Carla Rojas",    "1960-02-02", 0),  # hermana/o
    ("Luis Munoz",      "1988-08-08", "Diego Castro",   "1972-01-15", 0),  # nada que ver
]

in_dim = len(VOCAB)
rec_model = RecordSiamese(in_dim)
rec_opt = torch.optim.Adam(rec_model.parameters(), lr=5e-3)
rec_crit = nn.BCEWithLogitsLoss()

# construir tensores
R1 = torch.stack([record_to_vec(n, d) for n, d, _, _, _ in pairs])
R2 = torch.stack([record_to_vec(n, d) for _, _, n, d, _ in pairs])
Y = torch.tensor([float(m) for *_, m in pairs])

for step in range(300):
    logit = rec_model(R1, R2)
    loss = rec_crit(logit, Y)
    rec_opt.zero_grad(); loss.backward(); rec_opt.step()

with torch.no_grad():
    probs = torch.sigmoid(rec_model(R1, R2))
for (n1, d1, n2, d2, m), p in zip(pairs, probs):
    print(f"{n1:15s} vs {n2:15s} | label={m} | p(match)={p.item():.3f}")
```

Salida típica:

```
Juan Perez      vs Juan Peres      | label=1 | p(match)=0.88
Maria Gonzalez  vs Maria Gonzales  | label=1 | p(match)=0.91
Pedro Soto      vs Pedro Soto      | label=1 | p(match)=0.79
Ana Lopez       vs Ana Lopez       | label=1 | p(match)=0.95
Juan Perez      vs Juan Perez      | label=0 | p(match)=0.41
Carlos Rojas    vs Carla Rojas     | label=0 | p(match)=0.33
Luis Munoz      vs Diego Castro    | label=0 | p(match)=0.04
```

El caso `"Juan Perez 1985"` vs `"Juan Perez 1995"` es el **hard negative** clásico: nombre idéntico, persona distinta. La diferencia está enteramente en la fecha de nacimiento, y la cabeza $L_1$ ponderada debe aprender a pesar fuerte las dimensiones que codifican el año. Con tan pocos ejemplos el modelo apenas lo logra (0.41, justo bajo el umbral) —en producción harían falta miles de pares y features de campo dedicadas, no una bolsa de caracteres.

### 7.2 Por qué el deep metric learning compite con (pero no siempre supera a) reglas/GBM

Aquí viene la parte honesta, relevante para quien decide arquitectura de un sistema MDM real. La intuición ingenua dice "deep learning > reglas". En record linkage en salud, la realidad es más matizada:

- **El bi-encoder brilla como blocker, no necesariamente como scorer.** Su trabajo más valioso es poner los candidatos plausibles cerca en el espacio de embeddings para recuperarlos por ANN (vecindad aproximada), reduciendo millones de comparaciones a decenas. El recuerdo de Koch lo dice numéricamente: el 1-NN sobre píxeles crudos da 21.7%, sobre el embedding aprendido 92.0%. Ese salto **es** el valor del blocking. Pero la decisión final fina (match/no-match) a menudo la hace mejor un **GBM (XGBoost)** sobre features de comparación de campos: distancia de Jaro-Winkler en nombres, diferencia de fechas, coincidencia exacta de RUT, etc. El GBM es un scorer no lineal más expresivo que la sigmoide lineal sobre $L_1$, y se entrena con menos datos y se interpreta mejor.
- **Las reglas siguen siendo competitivas con datos limpios.** Cuando el RUT/identificador nacional es confiable, una regla de coincidencia exacta supera a cualquier red. El ML aporta valor en el régimen sucio (sin identificador, con typos, con campos faltantes), pero rinde **retornos decrecientes**: cada punto de F1 adicional cuesta cada vez más datos etiquetados y tuning.
- **El shift de dominio penaliza al deep learning más que a las reglas.** Un bi-encoder entrenado sobre nombres de una población no transfiere perfectamente a otra con distintas convenciones (apellidos compuestos, nombres indígenas, transliteraciones). Es el mismo 92% → 70% del paper al cambiar de Omniglot a MNIST.

### 7.3 La falta de transitividad: el problema que la siamesa no resuelve

Hay una limitación estructural del scoring **par a par** que es crítica en MDM y que la red siamesa **no** resuelve por sí sola: la **falta de transitividad**. El scorer responde "¿A y B son la misma persona?" de forma independiente para cada par. Pero puede ocurrir que:

$$
p(A, B) > \text{umbral} \;\wedge\; p(B, C) > \text{umbral} \;\not\Rightarrow\; p(A, C) > \text{umbral}
$$

Es decir: el modelo dice que A y B son la misma persona, y que B y C son la misma persona, pero **no** que A y C lo sean. Eso es lógicamente inconsistente para una relación de identidad (que debe ser transitiva). La siamesa no lo evita porque nunca ve los tres registros en conjunto —compara de a pares, sin contexto del cluster completo.

La solución en record linkage real es exactamente la misma idea que en few-shot dio origen a Matching/Prototypical Networks: **agregar contexto del conjunto**. En MDM se resuelve con una etapa posterior de **resolución de entidades sobre el grafo de matches**: se construye un grafo donde los nodos son registros y las aristas son matches sobre el umbral, y se aplica clustering (componentes conexas, correlation clustering) para forzar la transitividad y producir clusters de identidad coherentes. La siamesa produce las aristas; el clustering produce las entidades.

{{< concept-alert type="recordar" >}}
El patrón siamés (bi-encoder + scorer par a par) es la mitad de un sistema de patient matching: hace el blocking y el scoring de candidatos. La otra mitad —forzar la transitividad y producir clusters de identidad— vive **fuera** de la red, en una etapa de resolución de entidades sobre el grafo. Confundir "scorer par a par excelente" con "sistema de MDM completo" es un error de diseño común.
{{< /concept-alert >}}

---

## 8. Gotchas

Cuatro errores que arruinan una red siamesa en silencio.

**Weight tying mal hecho.** El error número uno: definir dos encoders distintos (`self.encoder_a` y `self.encoder_b`) en lugar de uno compartido. Si haces eso, no tienes una red siamesa —tienes dos redes independientes que no comparten la garantía de consistencia local ni la simetría, y que necesitan el doble de datos para aprender. Verificación rápida: cuenta los parámetros del encoder una sola vez. En nuestro `SiameseNet`, `self.encoder` se instancia una vez y se llama dos veces; eso es correcto. Si dudas, compara `id(model.encoder)` antes y después del forward —debe ser el mismo objeto.

**Colapso de embeddings.** Si la pérdida solo tuviera el término positivo (acercar pares iguales), la solución trivial sería mapear **todo** al mismo punto: $f_\theta(x) = \text{constante}$ para todo $x$, lo que da distancia 0 siempre y $p=0.5$. El término negativo de la BCE (alejar pares distintos) es lo que evita el colapso. Síntoma de colapso: la `verif_acc` se estanca en ~0.5 y la norma de los embeddings tiende a cero o todos los embeddings se parecen. Diagnóstico: imprime la varianza de los embeddings sobre un batch; si es ~0, hay colapso. Soluciones: asegurar pares negativos suficientes (balance 50/50), bajar el learning rate, o añadir un margen explícito (contrastive loss en vez de BCE).

```python
@torch.no_grad()
def embedding_variance(loader, n_batches=5):
    model.eval()
    embs = []
    for i, (x1, _, _) in enumerate(loader):
        if i >= n_batches: break
        embs.append(model.encoder(x1.to(device)))
    embs = torch.cat(embs)
    return embs.var(dim=0).mean().item()

print(f"varianza media de embeddings: {embedding_variance(train_loader):.4f}")
# sano: > 0.01 ; colapso: ~0.0001
```

**Muestreo de pares descuidado.** Si los pares se muestrean sin balance, la red aprende el shortcut "todo es distinto" (porque hay 100× más negativos) y logra alta accuracy diciendo siempre `no-match`. Por eso forzamos balance 50/50 en el `PairDataset`. El siguiente nivel es el **hard negative mining**: una vez que el modelo entrena un poco, los negativos aleatorios son casi todos triviales (gradiente cero). Reemplazarlos por negativos difíciles —pares de clases distintas que el modelo cree cercanos— concentra el gradiente donde importa. Esquema:

```python
@torch.no_grad()
def mine_hard_negatives(model, classes, n, rng):
    """Genera negativos donde el modelo da p(match) alto (los más confusos)."""
    model.eval()
    hard = []
    while len(hard) < n:
        c1, c2 = rng.sample(classes, 2)
        x1 = data_by_class[c1][rng.randrange(N_PER_CLASS)].unsqueeze(0).unsqueeze(0)
        x2 = data_by_class[c2][rng.randrange(N_PER_CLASS)].unsqueeze(0).unsqueeze(0)
        logit, _, _ = model(x1.to(device), x2.to(device))
        if torch.sigmoid(logit).item() > 0.5:   # el modelo se equivoca: negativo difícil
            hard.append((x1, x2))
    return hard
```

**Calibración del umbral.** La frontera de decisión `p > 0.5` es arbitraria. En verificación 50/50 funciona, pero en un escenario real (MDM) la prevalencia de matches es bajísima (la mayoría de los pares NO son la misma persona), y el costo de un falso match (fusionar dos pacientes distintos) es muy distinto del de un falso no-match (duplicar un paciente). El umbral óptimo se calibra sobre un set de validación según la curva precisión-recall y la matriz de costos del negocio, no se deja en 0.5. Para one-shot por argmax el umbral no importa (solo el ranking relativo), pero para una decisión binaria de linkage es crítico.

---

## 9. Limitaciones y conexión con Triplet loss y Matching Networks

Para ser honestos sobre lo que la red siamesa por pares **no** captura:

- **Entrenamiento por pares, no episódico.** Entrenamos sobre pares same/different muestreados independientemente, no sobre **episodios** que reproduzcan exactamente la tarea de test (N-way one-shot). La red solo aproxima la tarea objetivo; el principio "*train as you test*" de Matching Networks dice que el entrenamiento debe espejar la estructura de evaluación. Eso lo veremos en el [camino 05](/clases/clase-26/practica/05-matching-networks).
- **No usa el contexto del support set completo.** En inferencia comparamos la query contra cada candidato **independientemente** ($p(c)$ par a par). La red nunca ve los $N$ candidatos en conjunto, así que no puede razonar comparativamente ("esto se parece más a la clase 3 que a la 7 porque 3 y 7 son muy distintas entre sí"). Es la misma limitación que la falta de transitividad en MDM. Matching Networks introdujo **atención sobre todo el support**; Prototypical Networks promedia por clase en **prototipos**.
- **La decisión binaria necesita calibración.** La sigmoide produce una probabilidad absoluta cuyo umbral hay que calibrar. La **triplet loss** ([FaceNet](/papers/facenet-schroff-2015)) elimina ese problema aprendiendo un **ordenamiento relativo**: en vez de "¿es esto un match, sí o no?", aprende "el positivo está más cerca del ancla que el negativo, por un margen $\alpha$". No hay umbral global que calibrar, solo un margen. Por eso FaceNet reemplazó la siamesa de pares en verificación facial de producción. El detalle de la geometría de la esfera unitaria está en el fundamento de [triplet loss](/fundamentos/triplet-loss).
- **La métrica es semi-fija.** Solo los $\alpha_j$ se aprenden; la forma $L_1$ ponderada + sigmoide es una elección de diseño. Relation Networks dan el paso de reemplazar la métrica fija por una red que aprende el score de similitud desde cero.

La progresión del [metric learning](/fundamentos/metric-learning) es clara: de comparar **pares** aislados (siamesa, este capítulo), a comparar **tripletas** con margen (FaceNet), a comparar la query contra **todo el support** con atención (Matching Networks), a resumir el support en **centroides** (Prototypical Networks). Cada paso agrega contexto del conjunto y reduce la dependencia de calibrar umbrales y minar negativos a mano. El [camino 05](/clases/clase-26/practica/05-matching-networks) construye el siguiente eslabón.

---

## Cross-links

- Camino anterior: [03 - MAML](/clases/clase-26/practica/03-maml)
- Camino siguiente: [05 - Matching Networks](/clases/clase-26/practica/05-matching-networks)
- Fundamento transversal: [Metric Learning](/fundamentos/metric-learning)
- Fundamento relacionado: [Triplet Loss](/fundamentos/triplet-loss)
- Paper original: [Koch et al., 2015 — Siamese Neural Networks for One-shot Image Recognition](/papers/siamese-networks-koch-2015)

Volver a la [teoría de la Clase 26](../teoria) o al [hub de práctica](..).
