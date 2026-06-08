---
title: "01 - Episodios N-way K-shot"
weight: 31
math: true
---

En este capítulo vamos a construir, de principio a fin y en PyTorch puro, la **infraestructura de datos del meta-aprendizaje few-shot**: el episodio N-way K-shot. Antes de que exista ningún algoritmo concreto — Prototypical Networks, Matching Networks, MAML — hace falta una maquinaria que sepa **muestrear tareas**, no ejemplos. Esa maquinaria es lo que vamos a escribir acá. El `EpisodicSampler` que armemos en este capítulo es la base que reusan literalmente todos los caminos siguientes de esta clase: cambiará el modelo, cambiará la regla de adaptación, pero la forma del lote — un support set y un query set con etiquetas relativas — es siempre la misma.

Una decisión de diseño antes de empezar: este camino lo hacemos **solo en PyTorch**, sin la versión paralela en TensorFlow y JAX que sí hacemos en otros lugares del curso. La razón es que esto es **código de datos**, no de modelo: muestreo, indexado, `torch.stack`, padding de etiquetas. No hay capas, ni autograd, ni una arquitectura que valga la pena traducir tres veces. El bucle de muestreo es esencialmente NumPy con tensores; portarlo a `tf.data` o a un `jax` pipeline es mecánico y no enseña nada nuevo sobre meta-learning. Lo que sí importa — la **lógica** del episodio — es idéntica en cualquier framework. Acá la queremos ver con la lupa pegada.

---

## 1. Setup

```python
import math
import random
from collections import defaultdict

import numpy as np
import torch

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"device = {device}")
```

Para el muestreo de episodios la GPU es casi irrelevante: lo pesado vendrá después, cuando un modelo procese el episodio. El muestreo en sí es indexado de tensores en CPU. Fijamos las tres semillas — `random`, `numpy` y `torch` — porque vamos a usar las tres fuentes de aleatoriedad: `random.sample` para elegir clases, los generadores de NumPy para sintetizar datos, y `torch` para barajar dentro de las clases. Reproducibilidad total es crítica acá: un bug en el muestreo es invisible si no podés volver al mismo episodio dos veces.

---

## 2. El concepto de tarea (episodio): por qué muestreamos tareas y no ejemplos

En el aprendizaje supervisado clásico, la unidad de muestreo es el **ejemplo**: tomás un minibatch de imágenes con sus etiquetas absolutas, calculás la pérdida, retropropagás. El conjunto de clases es fijo y conocido de antemano: si entrenás un clasificador de 1000 clases de ImageNet, esas 1000 clases son las mismas en train y en test.

El meta-aprendizaje few-shot rompe esa suposición. El objetivo ya no es **aprender una tarea**, sino **aprender a aprender tareas nuevas a partir de poquísimos ejemplos**. Para entrenar esa capacidad, la unidad de muestreo deja de ser el ejemplo y pasa a ser la **tarea**, también llamada **episodio**. Cada episodio es un mini-problema de clasificación autocontenido:

- Se eligen **N clases** al azar de un reservorio grande de clases (el parámetro **way**).
- De cada una de esas N clases se toman **K ejemplos etiquetados** que forman el **support set** (el parámetro **shot**).
- De las mismas N clases se toman **Q ejemplos adicionales, disjuntos del support**, que forman el **query set**.

El modelo ve el support set (los "K ejemplos por clase" con etiqueta) y debe clasificar correctamente el query set. La gracia es que las N clases **cambian en cada episodio**, y en la fase de meta-test son clases que el modelo **nunca vio durante el meta-entrenamiento**. Así, lo que el modelo aprende no es "cómo se ve un gato", sino "cómo usar K ejemplos etiquetados de N clases arbitrarias para clasificar ejemplos nuevos de esas mismas clases". Esa es la idea central de Matching Networks (Vinyals et al., 2016) y la formulación que heredan casi todos los métodos posteriores: **las condiciones de entrenamiento deben imitar las de test** — si en test vas a tener 5 clases con 1 ejemplo cada una, entrená con episodios 5-way 1-shot.

El diagrama de un episodio 3-way 2-shot con 2 queries por clase se ve así:

```
Reservorio de clases:  [A] [B] [C] [D] [E] [F] [G] ...   (muchas clases)
                         |       |           |
        muestrear N=3:  [B] ----[D]---------[F]

Para cada clase elegida, tomar K+Q ejemplos disjuntos:

   clase B  ->  support: b1 b2   query: b3 b4
   clase D  ->  support: d1 d2   query: d3 d4
   clase F  ->  support: f1 f2   query: f3 f4

Etiquetas RELATIVAS al episodio:  B->0   D->1   F->2

  support_set = { (b1,0)(b2,0)  (d1,1)(d2,1)  (f1,2)(f2,2) }   -> N*K = 6
  query_set   = { (b3,0)(b4,0)  (d3,1)(d4,1)  (f3,2)(f4,2) }   -> N*Q = 6
```

Dos cosas para fijar desde ya, porque son fuente de errores sutiles:

1. **Las etiquetas son relativas al episodio, no absolutas.** Dentro del episodio, la clase B es "0", la D es "1" y la F es "2". En el episodio siguiente esos mismos índices 0, 1, 2 corresponderán a otras tres clases del reservorio. El modelo nunca aprende "la clase B"; aprende "la clase 0 del episodio actual, definida por su support set".
2. **Support y query son disjuntos.** Los ejemplos del query jamás aparecen en el support del mismo episodio. Si se mezclan, el modelo puede "memorizar" la respuesta y el accuracy reportado es una mentira.

---

## 3. Un dataset de juguete reproducible

Para construir y depurar el sampler necesitamos un reservorio de clases con ejemplos. Mostramos **dos vías**: una sintética que corre siempre (blobs gaussianos en 2D, una clase = un cluster), y la canónica de la literatura (Omniglot vía `torchvision`), que requiere descarga. La sintética es la que usaremos para los smoke tests porque es instantánea, determinista y permite visualizar el episodio en un plano.

### 3.1 Vía sintética: blobs gaussianos (siempre corre)

La idea: cada **clase** es una gaussiana 2D con su propio centro $\mu_c$ y una covarianza isotrópica chica. Un **ejemplo** de la clase $c$ es una muestra $x \sim \mathcal{N}(\mu_c, \sigma^2 I)$. Generamos muchas clases (digamos 50) para tener un reservorio del que muestrear, exactamente como en few-shot real donde hay cientos o miles de clases base.

```python
class GaussianBlobDataset:
    """Reservorio sintético: cada clase es una gaussiana 2D propia.

    Expone exactamente lo que el sampler necesita:
      - self.by_class: dict {label_absoluto: tensor (n_per_class, D)}
      - self.classes:  lista de labels absolutos disponibles
    """
    def __init__(self, n_classes=50, n_per_class=30, dim=2,
                 spread=8.0, sigma=0.6, seed=0):
        g = np.random.default_rng(seed)
        # Centros de clase repartidos en una grilla ruidosa para que no colapsen
        centers = g.uniform(-spread, spread, size=(n_classes, dim))
        self.by_class = {}
        for c in range(n_classes):
            pts = centers[c] + sigma * g.standard_normal((n_per_class, dim))
            self.by_class[c] = torch.tensor(pts, dtype=torch.float32)
        self.classes = list(self.by_class.keys())
        self.dim = dim

    def get_examples(self, label, idxs):
        return self.by_class[label][idxs]


pool = GaussianBlobDataset(n_classes=50, n_per_class=30, dim=2, seed=0)
print(f"clases en el reservorio: {len(pool.classes)}")
print(f"ejemplos por clase:      {pool.by_class[0].shape[0]}")
print(f"shape de una clase:      {tuple(pool.by_class[0].shape)}")
# clases en el reservorio: 50
# ejemplos por clase:      30
# shape de una clase:      (30, 2)
```

La estructura que expone — un diccionario `by_class` que va de **etiqueta absoluta** a un tensor `(n_per_class, dim)` — es deliberadamente el contrato mínimo que el sampler necesita. Cualquier dataset real (Omniglot, miniImageNet, un dataset clínico de imágenes raras) se puede envolver en esta misma interfaz: "dame la lista de clases" y "dame los ejemplos `idxs` de la clase `c`".

### 3.2 Vía canónica: Omniglot (requiere descarga)

Omniglot — el "MNIST transpuesto" de Lake et al. (2015) — es el banco de pruebas histórico del few-shot: 1623 caracteres de 50 alfabetos, con apenas **20 ejemplos por carácter**. Esa escasez por clase es justamente la condición few-shot. `torchvision` lo trae:

```python
# Requiere conexión a internet la primera vez.
from torchvision import transforms
from torchvision.datasets import Omniglot

tf = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),          # (1, 28, 28) en [0,1]
])

omni = Omniglot(root="./data", background=True, download=True, transform=tf)

# Reagrupar por clase para exponer el mismo contrato que el dataset sintético.
by_class = defaultdict(list)
for img, label in omni:
    by_class[label].append(img)
by_class = {c: torch.stack(v) for c, v in by_class.items()}
print(f"clases Omniglot: {len(by_class)}  | ejemplos/clase: {next(iter(by_class.values())).shape[0]}")
# clases Omniglot: 964  | ejemplos/clase: 20
```

El detalle relevante: una vez reagrupado, `by_class` tiene **exactamente la misma forma** que el `pool.by_class` sintético — un dict de etiqueta a tensor `(n_per_class, ...)`. Lo único que cambia es la forma de cada ejemplo: `(1, 28, 28)` en vez de `(2,)`. El sampler no necesita saber nada de eso; trabaja con índices. De aquí en adelante usamos el reservorio sintético para que todo corra sin descargas, pero todo aplica idéntico a Omniglot.

---

## 4. El EpisodicSampler

Esta es la pieza central del capítulo. Le pasamos un reservorio (cualquier objeto con `classes` y `by_class`) y los hiperparámetros del episodio — N (way), K (shot), Q (query) — y nos devuelve, episodio tras episodio, los cuatro tensores `support_x, support_y, query_x, query_y` con las etiquetas ya **relativas** y los conjuntos **disjuntos**.

```python
class EpisodicSampler:
    """Muestrea episodios N-way K-shot con Q queries por clase.

    Garantías:
      - N clases distintas elegidas al azar del reservorio.
      - K+Q ejemplos disjuntos por clase (support y query no se solapan).
      - Etiquetas RELATIVAS 0..N-1, reasignadas en cada episodio.
    """
    def __init__(self, pool, n_way, k_shot, q_query):
        self.pool = pool
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

        need = k_shot + q_query
        # Solo clases con suficientes ejemplos para K+Q sin reposición.
        self.eligible = [c for c in pool.classes
                         if pool.by_class[c].shape[0] >= need]
        if len(self.eligible) < n_way:
            raise ValueError(
                f"Se necesitan >= {n_way} clases con >= {need} ejemplos; "
                f"hay {len(self.eligible)}."
            )

    def sample_episode(self):
        # 1) Elegir N clases distintas (sin reposición) del reservorio.
        episode_classes = random.sample(self.eligible, self.n_way)

        support_x, support_y = [], []
        query_x, query_y = [], []

        # 2) Para cada clase, asignar una etiqueta RELATIVA y partir K+Q.
        for rel_label, abs_label in enumerate(episode_classes):
            n_avail = self.pool.by_class[abs_label].shape[0]
            # Permutación de los índices de ESTA clase, luego cortar.
            perm = torch.randperm(n_avail)
            chosen = perm[: self.k_shot + self.q_query]
            sup_idx = chosen[: self.k_shot]            # primeros K -> support
            qry_idx = chosen[self.k_shot:]             # siguientes Q -> query (disjunto)

            support_x.append(self.pool.get_examples(abs_label, sup_idx))
            query_x.append(self.pool.get_examples(abs_label, qry_idx))
            support_y.append(torch.full((self.k_shot,), rel_label, dtype=torch.long))
            query_y.append(torch.full((self.q_query,), rel_label, dtype=torch.long))

        # 3) Apilar. Orden: clase por clase (no barajado aún).
        support_x = torch.cat(support_x, dim=0)        # (N*K, *feat)
        support_y = torch.cat(support_y, dim=0)        # (N*K,)
        query_x = torch.cat(query_x, dim=0)            # (N*Q, *feat)
        query_y = torch.cat(query_y, dim=0)            # (N*Q,)

        # 4) Barajar el query (buena higiene: que el orden no filtre la clase).
        qperm = torch.randperm(query_x.shape[0])
        query_x, query_y = query_x[qperm], query_y[qperm]

        return support_x, support_y, query_x, query_y
```

Repasemos las cuatro decisiones, porque cada una corresponde a una garantía:

- **Paso 1 — `random.sample`** elige N clases **sin reposición**: nunca aparece la misma clase dos veces en un episodio. Si usáramos `random.choices` (con reposición) tendríamos dos etiquetas relativas distintas apuntando a la misma clase absoluta, lo que es un episodio degenerado.
- **Paso 2 — `enumerate`** es donde nacen las **etiquetas relativas**. `rel_label` va 0, 1, ..., N-1 en el orden en que salieron las clases. La etiqueta absoluta `abs_label` solo se usa para ir a buscar los datos; el modelo jamás la ve.
- **El `torch.randperm` por clase + corte `[:K]` y `[K:K+Q]`** es lo que garantiza **disjunción** support/query: los K del support y los Q del query salen de la misma permutación, de tramos que no se solapan. Imposible que un ejemplo caiga en ambos.
- **Paso 4 — barajar el query** no es estrictamente necesario para la corrección (las etiquetas viajan con los datos), pero evita que cualquier código aguas abajo se apoye accidentalmente en el orden "todas las de clase 0, luego todas las de clase 1". Es higiene defensiva.

Probemos un episodio 5-way 1-shot con 15 queries por clase — la configuración clásica de los papers de few-shot:

```python
sampler = EpisodicSampler(pool, n_way=5, k_shot=1, q_query=15)
sx, sy, qx, qy = sampler.sample_episode()

print("support_x:", tuple(sx.shape))
print("support_y:", tuple(sy.shape), "->", sy.tolist())
print("query_x:  ", tuple(qx.shape))
print("query_y:  ", tuple(qy.shape))
print("labels relativos únicos:", sorted(set(qy.tolist())))
# support_x: (5, 2)
# support_y: (5,) -> [0, 1, 2, 3, 4]
# query_x:   (75, 2)
# query_y:   (75,)
# labels relativos únicos: [0, 1, 2, 3, 4]
```

La tabla de shapes, que conviene tener memorizada porque aparece en cada algoritmo de la clase:

| Tensor      | Shape           | Para N-way K-shot Q-query | Ejemplo 5-way 1-shot 15-query |
|-------------|-----------------|---------------------------|-------------------------------|
| `support_x` | `(N*K, *feat)`  | features de los ejemplos etiquetados | `(5, 2)` |
| `support_y` | `(N*K,)`        | etiquetas relativas 0..N-1          | `(5,)`  |
| `query_x`   | `(N*Q, *feat)`  | features de los ejemplos a clasificar | `(75, 2)` |
| `query_y`   | `(N*Q,)`        | etiquetas relativas 0..N-1 (verdad)   | `(75,)` |

Donde `*feat` es la forma de un ejemplo: `(2,)` para los blobs, `(1, 28, 28)` para Omniglot, `(3, 84, 84)` para miniImageNet. El sampler es agnóstico a esa forma — solo concatena en la dimensión 0.

### 4.1 Verificación de disjunción

Vale la pena escribir un test explícito que confirme que support y query nunca comparten un ejemplo. Como los blobs no tienen IDs persistentes, hagamos el test sobre el sampler mismo, comprobando que los índices elegidos por clase no se solapan. Reescribimos brevemente el corte para inspeccionarlo:

```python
def check_disjoint(pool, n_way, k_shot, q_query, trials=1000):
    for _ in range(trials):
        c = random.choice(pool.classes)
        n_avail = pool.by_class[c].shape[0]
        perm = torch.randperm(n_avail)
        chosen = perm[: k_shot + q_query]
        sup = set(chosen[:k_shot].tolist())
        qry = set(chosen[k_shot:].tolist())
        assert sup.isdisjoint(qry), "fuga support/query!"
    print(f"OK: {trials} cortes sin solapamiento support/query")

check_disjoint(pool, n_way=5, k_shot=1, q_query=15)
# OK: 1000 cortes sin solapamiento support/query
```

El test pasa porque la disjunción es estructural: viene de cortar una **única** permutación en dos tramos. Es el tipo de invariante que conviene blindar con un assert, porque cuando se rompe (típicamente al refactorizar para hacer batching de episodios) el síntoma es un accuracy sospechosamente alto, no un crash.

---

## 5. Visualizar un episodio 5-way 1-shot

Como nuestros ejemplos viven en 2D, podemos dibujar un episodio completo y *ver* la estructura del problema few-shot. Es uno de los pocos casos donde el dataset de juguete paga doble: lo que en Omniglot serían imágenes, acá son puntos en un plano.

```python
import matplotlib.pyplot as plt

sx, sy, qx, qy = sampler.sample_episode()

fig, ax = plt.subplots(figsize=(6, 6))
cmap = plt.cm.tab10
for rel in range(sampler.n_way):
    # Query de esta clase: puntos chicos, semitransparentes.
    qm = qy == rel
    ax.scatter(qx[qm, 0], qx[qm, 1], s=20, alpha=0.35,
               color=cmap(rel), label=f"clase {rel} (query)")
    # Support de esta clase: estrella grande, el "prototipo etiquetado".
    sm = sy == rel
    ax.scatter(sx[sm, 0], sx[sm, 1], s=300, marker="*",
               edgecolor="black", color=cmap(rel), zorder=3)

ax.set_title("Episodio 5-way 1-shot (estrellas = support, puntos = query)")
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.show()
```

Qué se vería: cinco nubes de puntos bien separadas, cada una de un color (las 5 clases del episodio, elegidas al azar de las 50 del reservorio). En cada nube hay **una sola estrella grande** — ese es el único ejemplo de support (K=1) — rodeada de unos 15 puntos chicos semitransparentes, que son el query a clasificar. La separación clara de las nubes anticipa por qué un método de distancias como Prototypical Networks va a funcionar bien acá: para clasificar un punto de query basta con asignarlo a la estrella más cercana. Si volvés a correr `sample_episode()` y redibujás, las cinco nubes serán **otras cinco clases** del reservorio, con otros colores y otras posiciones. Esa variabilidad episodio a episodio es exactamente lo que el meta-entrenamiento explota.

---

## 6. El loop de meta-entrenamiento genérico (andamiaje)

Con el sampler listo, podemos escribir el esqueleto del meta-entrenamiento. Lo importante de este capítulo es que el esqueleto es **independiente del algoritmo**: Prototypical Networks, MAML, Matching Networks y Relation Networks comparten esta misma estructura de tres pasos por episodio. Lo que cambia es **qué** ocurre dentro de "adaptar" y "evaluar". Acá dejamos esos pasos como funciones abstractas; los caminos siguientes de la clase los rellenan.

La estructura conceptual del meta-entrenamiento — el "outer loop" sobre episodios, con un "inner loop" de adaptación adentro — es:

$$
\theta^{\star} = \arg\min_{\theta} \; \mathbb{E}_{\mathcal{T} \sim p(\mathcal{T})} \Big[ \mathcal{L}_{\mathcal{T}}^{\text{query}}\big( f_{\theta}, \; S_{\mathcal{T}}, \; Q_{\mathcal{T}} \big) \Big]
$$

En palabras: buscamos parámetros $\theta$ que, **promediados sobre la distribución de tareas** $p(\mathcal{T})$, minimicen la pérdida en el query set $Q_{\mathcal{T}}$ después de haber usado el support set $S_{\mathcal{T}}$ para adaptarse a la tarea. La esperanza sobre $\mathcal{T}$ es lo que el sampler aproxima por muestreo: cada episodio es una muestra de $p(\mathcal{T})$.

```python
def inner_adapt(model, support_x, support_y):
    """Usa el support para condicionar/adaptar el modelo a ESTA tarea.

    Lo que devuelve depende del algoritmo:
      - Prototypical Networks: prototipos (media de embeddings por clase).
      - Matching Networks:     el support embebido para la attention.
      - MAML:                  parámetros adaptados tras unos pasos de SGD interno.
    Acá es un placeholder.
    """
    raise NotImplementedError("lo rellena cada camino de la clase")


def query_loss_and_acc(model, adapted, query_x, query_y):
    """Clasifica el query con el modelo adaptado y devuelve (loss, acc)."""
    raise NotImplementedError("lo rellena cada camino de la clase")


def meta_train(model, sampler, n_episodes=10_000, lr=1e-3, log_every=500):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    running_acc = []
    for episode in range(n_episodes):
        # ---- muestrear una TAREA (no un minibatch de ejemplos) ----
        sx, sy, qx, qy = sampler.sample_episode()
        sx, sy = sx.to(device), sy.to(device)
        qx, qy = qx.to(device), qy.to(device)

        # ---- inner: adaptar al support ----
        adapted = inner_adapt(model, sx, sy)

        # ---- outer: evaluar en el query y actualizar theta ----
        loss, acc = query_loss_and_acc(model, adapted, qx, qy)
        optim.zero_grad()
        loss.backward()
        optim.step()

        running_acc.append(acc)
        if (episode + 1) % log_every == 0:
            window = running_acc[-log_every:]
            print(f"episodio {episode+1:6d}  "
                  f"loss={loss.item():.3f}  "
                  f"acc(últimos {log_every})={sum(window)/len(window):.3f}")
    return model
```

Tres observaciones sobre el andamiaje:

1. **El `optim.step()` es el "outer loop".** Actualiza los parámetros $\theta$ que persisten **entre** tareas. En Prototypical Networks ese $\theta$ es el encoder; en MAML es la inicialización de la que parte cada adaptación interna.
2. **El "inner loop" vive dentro de `inner_adapt`.** En métodos métricos (Prototypical, Matching) no hay realmente un loop — la "adaptación" es calcular prototipos o embeber el support, sin gradientes internos. En MAML sí hay un loop de SGD sobre el support, y ahí la cosa se pone interesante con la doble derivada.
3. **Se evalúa en el query, no en el support.** El gradiente del outer loop viene de la pérdida sobre el query. Esto es lo que fuerza al modelo a aprender una adaptación que **generalice** del support al query, en vez de memorizar el support.

Notar que `meta_train` ya está completo y correcto en su andamiaje: lo único `NotImplementedError` son las dos funciones que dependen del algoritmo. Esa separación limpia es justamente la que permite reusar el sampler y el loop en todos los caminos siguientes.

---

## 7. Gotchas

El muestreo episódico tiene varias trampas que producen bugs silenciosos — código que corre sin error pero reporta métricas falsas. Las más comunes:

- **Etiquetas relativas vs absolutas.** El error número uno del principiante es pasarle al modelo las etiquetas absolutas del reservorio (que pueden ir de 0 a 49, o ser strings de Omniglot). La cabeza de clasificación de un episodio N-way tiene **exactamente N salidas**, indexadas 0..N-1. Si le pasás un label absoluto de 37 a una `cross_entropy` con 5 clases, o crashea o (peor) indexa basura. La reasignación con `enumerate` en el sampler es lo que blinda esto: **toda etiqueta que sale del sampler está en [0, N)**.

- **Fuga de información support/query.** Si un ejemplo aparece tanto en el support como en el query del mismo episodio, el modelo puede "verlo" durante la adaptación y luego "reconocerlo" en el query. El accuracy se infla. Nuestra disjunción por permutación lo previene, pero el bug reaparece apenas alguien hace caching de episodios mal, o reusa índices entre support y query al optimizar. Por eso el assert de la sección 4.1 vale su peso en oro.

- **Balance de clases.** Por construcción el episodio está **perfectamente balanceado**: exactamente K support y Q query por clase. Eso simplifica las métricas — un baseline aleatorio en N-way da exactamente $1/N$ de accuracy, sin la ambigüedad del baseline en datasets desbalanceados. Pero también significa que un episodio few-shot **no** te entrena para distribuciones de clase desbalanceadas; si tu problema real lo es, eso es un gap entre el setup de entrenamiento y el despliegue.

- **Por qué se reportan intervalos de confianza sobre muchos episodios.** Un solo episodio 5-way 1-shot evalúa sobre apenas $N \cdot Q = 75$ queries, y depende fuertemente de qué 5 clases tocaron y qué ejemplo de support cayó. La varianza entre episodios es enorme. Por eso la literatura **nunca** reporta el accuracy de un episodio: se promedia sobre **600 a 10 000 episodios de test** y se reporta media $\pm$ intervalo de confianza del 95 %:

$$
\overline{\text{acc}} \pm 1.96 \cdot \frac{\sigma}{\sqrt{M}}
$$

donde $M$ es el número de episodios de test y $\sigma$ la desviación estándar del accuracy por episodio. Comparar dos métodos sin esos intervalos es comparar ruido: diferencias de 1-2 puntos suelen estar dentro del intervalo y no significan nada.

```python
def evaluate(model, sampler, n_episodes=600):
    """Esqueleto de evaluación con intervalo de confianza al 95%."""
    accs = []
    for _ in range(n_episodes):
        sx, sy, qx, qy = sampler.sample_episode()
        # adapted = inner_adapt(model, sx.to(device), sy.to(device))
        # _, acc = query_loss_and_acc(model, adapted, qx.to(device), qy.to(device))
        # accs.append(acc)
        pass
    accs = np.array(accs) if accs else np.zeros(1)
    mean = accs.mean()
    ci95 = 1.96 * accs.std(ddof=1) / math.sqrt(len(accs)) if len(accs) > 1 else 0.0
    print(f"accuracy = {mean:.4f} ± {ci95:.4f}  (sobre {n_episodes} episodios)")
    return mean, ci95
```

- **Episodios de test con clases nunca vistas.** El split de few-shot no es por ejemplo sino **por clase**: las clases del meta-train, meta-val y meta-test son **disjuntas**. Si una clase aparece tanto en train como en test, estás midiendo memorización, no la capacidad de aprender clases nuevas. En Omniglot eso se respeta con los splits `background` (train) y `evaluation` (test), que son alfabetos distintos. En nuestro reservorio sintético habría que partir las 50 clases en, por ejemplo, 30 de train y 20 de test, y construir **dos samplers** sobre subconjuntos disjuntos de `pool.classes`.

---

## 8. Limitaciones del toy setup y siguiente paso

Para ser honestos sobre lo que **no** captura este andamiaje:

- **Las clases gaussianas son linealmente separables.** Cinco blobs bien apartados son un problema trivial: hasta un clasificador de distancia al centroide con un encoder identidad lo resuelve. El few-shot real (Omniglot, miniImageNet) requiere que el modelo **aprenda un encoder** que mapee imágenes a un espacio donde las clases nuevas sean separables. Nuestro toy elimina justo esa dificultad — a propósito, para aislar la mecánica del muestreo.

- **No partimos train/test por clase.** En el código de arriba un mismo `pool` alimenta entrenamiento y evaluación. Para un experimento honesto hay que construir dos samplers sobre subconjuntos disjuntos de clases, como se explicó en el último gotcha. Lo omitimos para mantener el foco en el sampler.

- **No hay batching de episodios.** Procesamos un episodio por paso. Las implementaciones de producción muestrean un **meta-batch** de varios episodios y promedian sus gradientes, lo que estabiliza el outer loop. Agregarlo es envolver el sampler en un bucle y apilar una dimensión más adelante de `(N*K, ...)`.

- **No medimos costo de muestreo.** Con 50 clases el `random.sample` es instantáneo; con millones de clases o ejemplos en disco, el muestreo episódico necesita índices precomputados y carga perezosa para no volverse el cuello de botella.

El siguiente capítulo toma exactamente este `EpisodicSampler` y rellena las dos funciones abstractas — `inner_adapt` y `query_loss_and_acc` — con el primer algoritmo concreto: **Prototypical Networks**, donde la adaptación es calcular el centroide (prototipo) de cada clase en el espacio de embeddings y la clasificación es asignar cada query al prototipo más cercano. Toda la infraestructura de datos que armamos acá se reusa sin cambios.

---

## Cross-links

- Siguiente camino: [02 - Prototypical Networks](/clases/clase-26/practica/02-prototypical-net)
- Teoría de la clase: [Meta-aprendizaje](../teoria)
- Fundamento transversal: [Few-shot learning](/fundamentos/few-shot-learning)
- Paper de referencia: [Vinyals et al., 2016 — Matching Networks for One Shot Learning](/papers/matching-networks-vinyals-2016)

Volver al [hub de práctica](..) o a la [Clase 26](../..).
