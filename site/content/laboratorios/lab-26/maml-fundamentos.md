---
title: "MAML: optimización binivel en código"
weight: 2
math: true
---

> **El corazón de la Parte 1 del lab.** Implementamos MAML (Model-Agnostic Meta-Learning, [Finn et al. 2017](/papers/maml-finn-2017)) usando la librería [learn2learn](/papers/learn2learn-arnold-2020) sobre dos *benchmarks* few-shot: **Omniglot** (caracteres manuscritos, fácil) y **Mini-ImageNet** (fotos RGB, difícil). Esta página desmonta las cuatro piezas del algoritmo —muestreo episódico, bucle interno (`fast_adapt_MAML`), bucle externo (`run_MAML`) y la intuición binivel— y explica el QUÉ, el PORQUÉ y el CÓMO de cada decisión. La configuración base del lab es **4-way 1-shot** (`WAYS=4`, `SHOTS=1`).

## 1. Encuadre: aprender a aprender con dos niveles

MAML no resuelve una tarea: aprende una **inicialización de pesos** $\theta$ desde la cual cualquier tarea nueva se resuelve con poquísimos ejemplos y poquísimos pasos de gradiente. Esa es la promesa del [meta-aprendizaje](/fundamentos/meta-aprendizaje) y del [few-shot learning](/fundamentos/few-shot-learning): no memorizar, sino dejar el modelo *a un empujón de distancia* de la solución de cada tarea.

El lab lo prueba sobre dos datasets que cubren los extremos de dificultad:

| Dataset | Entrada | Dificultad | Por qué |
|---|---|---|---|
| **Omniglot** | 1×28×28 (gris) | Fácil | Caracteres binarios, formas muy distintas entre clases, casi sin variación intra-clase. |
| **Mini-ImageNet** | 3×84×84 (RGB) | Difícil | Fotos reales: variación intra-clase enorme, fondos distractores, color y textura. |

Ambos comparten la misma estructura **N-way K-shot**: cada *tarea* presenta $N$ clases con $K$ ejemplos etiquetados, y el modelo debe clasificar ejemplos nuevos de esas mismas $N$ clases. Con `WAYS=4, SHOTS=1` el modelo ve **4 clases, 1 ejemplo cada una**, y debe generalizar. La gracia es que las clases cambian en cada tarea: el modelo nunca puede memorizar un clasificador fijo de 4 salidas.

## 2. El muestreo episódico: cómo nace una tarea

La maquinaria de tareas viene de learn2learn:

```python
tasksets = l2l.vision.benchmarks.get_tasksets(
    'omniglot',
    train_ways=WAYS,        # 4 clases por tarea
    train_samples=2*SHOTS,  # 2 ejemplos por clase -> mitad support, mitad query
    test_ways=WAYS,
    test_samples=2*SHOTS,
    num_tasks=10000,        # universo de tareas distintas
    root='~/data',
)
```

El detalle clave es `train_samples=2*SHOTS`. Pedimos **el doble** de ejemplos por clase porque cada tarea se parte en dos mitades con roles distintos:

- **Support set** (adaptación): los $K$ ejemplos con los que el modelo *aprende* la tarea en el bucle interno.
- **Query set** (evaluación): otros $K$ ejemplos, no vistos en la adaptación, con los que *medimos* si la adaptación funcionó.

Esta separación es el equivalente few-shot del clásico train/test, pero **dentro de una sola tarea**. Si evaluáramos sobre el mismo support con el que adaptamos, mediríamos memorización, no generalización.

### El truco par/impar para el split

Cada `.sample()` devuelve un lote ya barajado de $2 \cdot K \cdot N$ ejemplos. El lab los separa con una máscara booleana basada en índices pares:

```python
adaptation_indices = np.zeros(data.size(0), dtype=bool)
adaptation_indices[np.arange(shots*ways) * 2] = True  # 0, 2, 4, ... -> support
evaluation_indices = ~adaptation_indices              # 1, 3, 5, ... -> query
```

`np.arange(shots*ways)*2` produce `[0, 2, 4, ...]`: los índices pares van a *support*, los impares a *query*. Como learn2learn entrega los ejemplos intercalados por clase, este patrón par/impar reparte **una mitad de cada clase a cada conjunto** de forma balanceada, sin sesgar el split hacia ninguna clase.

> **Splits con clases disjuntas.** Los conjuntos `train`, `validation` y `test` de los tasksets no comparten clases: las clases que el modelo ve en meta-entrenamiento nunca aparecen en meta-testing. Esto es lo que hace honesto el experimento few-shot: en el test, el modelo enfrenta categorías literalmente nuevas. Si las clases se solaparan, mediríamos transferencia trivial, no aprendizaje few-shot real.

### Cómo se ven las tareas

Una tarea de Omniglot, support (lo que el modelo usa para adaptarse) y query (lo que debe clasificar):

![Tarea Omniglot - support](/laboratorios/lab-26/omniglot-task-adapt.jpg)

![Tarea Omniglot - query](/laboratorios/lab-26/omniglot-task-eval.jpg)

Las formas son binarias y muy distintas entre sí: discriminar 4 caracteres manuscritos con un ejemplo basta porque la señal es nítida y la variación dentro de cada clase es mínima.

La misma estructura en Mini-ImageNet revela por qué es mucho más duro:

![Tarea Mini-ImageNet - support](/laboratorios/lab-26/miniimagenet-task-adapt.jpg)

![Tarea Mini-ImageNet - query](/laboratorios/lab-26/miniimagenet-task-eval.jpg)

Aquí la **variación intra-clase** es brutal. Un mismo objeto puede aparecer en el support con una pose y en el query con otra completamente distinta —un tazón boca abajo en la adaptación, boca arriba en la evaluación—, además de fondos distractores, iluminación variable y oclusiones. Con un solo ejemplo de support, el modelo tiene que inferir la categoría sin haber visto esa apariencia concreta. Por eso el accuracy de Mini-ImageNet 1-shot ronda valores mucho más modestos que el de Omniglot: la tarea es intrínsecamente ambigua.

## 3. `fast_adapt_MAML`: el bucle interno

El bucle interno toma una tarea, adapta los pesos sobre el support y mide en el query:

```python
def fast_adapt_MAML(batch, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # split support / query con el truco par-impar
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways) * 2] = True
    evaluation_indices = ~adaptation_indices
    adaptation_data,  adaptation_labels  = data[adaptation_indices],  labels[adaptation_indices]
    evaluation_data,  evaluation_labels  = data[evaluation_indices],  labels[evaluation_indices]

    # --- adaptacion (theta -> theta') sobre el SUPPORT ---
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        train_error /= len(adaptation_data)   # (ver gotcha)
        learner.adapt(train_error)            # actualiza pesos EN el grafo

    # --- evaluacion sobre el QUERY ---
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy
```

La línea cargada de significado es `learner.adapt(train_error)`. A diferencia de un `optimizer.step()` normal, **no rompe el grafo de cómputo**: actualiza los pesos de $\theta$ a $\theta'$ pero mantiene la dependencia diferenciable entre ambos. Esto es esencial, porque el `valid_error` que devolvemos —calculado con $\theta'$— se va a derivar **respecto de $\theta$** en el bucle externo. Sin esa conexión preservada, no habría meta-gradiente.

Formalmente, un paso de adaptación es un paso de descenso de gradiente diferenciable:

$$
\theta' = \theta - \alpha \, \nabla_\theta \, \mathcal{L}_{\text{support}}(\theta)
$$

donde $\alpha$ es la tasa de aprendizaje interna (`fast_lr`). Con `adaptation_steps>1` se encadenan varios de estos pasos, y el grafo crece para que todos sean diferenciables hacia atrás.

> **Gotcha: la doble normalización.** La pérdida del lab usa `reduction="mean"` por defecto (promedia sobre los ejemplos), y aun así el código hace `train_error /= len(adaptation_data)`. Eso aplica una división extra por $N$, así que el gradiente queda escalado en $1/N$ respecto del esperado. En la práctica es **inofensivo**: solo reescala la magnitud efectiva de `fast_lr` por una constante, y como `fast_lr=0.5` es agresivo de entrada, la adaptación sigue funcionando. Pero es el tipo de detalle que conviene notar: si alguien reusa `fast_adapt_MAML` con otra loss o con `reduction="sum"`, la escala cambia y el `fast_lr` deja de ser comparable.

## 4. `run_MAML`: el bucle externo

El bucle externo es donde MAML *aprende a aprender*. Maneja **dos tasas de aprendizaje** con propósitos opuestos:

| LR | Valor | Dónde | Rol |
|---|---|---|---|
| `fast_lr` | **0.5** | `learner.adapt()` (interno) | Agresivo: en 1-5 pasos debe mover el modelo lo suficiente para resolver UNA tarea, que luego se descarta. |
| `meta_lr` | **0.003** | `opt.step()` de Adam (externo) | Suave: ajusta la inicialización $\theta$ que se **conserva** entre tareas. Pasos grandes la desestabilizarían. |

```python
def run_MAML(tasksets, fast_lr=0.5, meta_lr=0.003,
             adaptation_steps=1, meta_batch_size=32,
             num_iterations=..., first_order=True, ways=WAYS, shots=SHOTS):
    model = ...
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=first_order)
    opt  = optim.Adam(maml.parameters(), meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

    for iteration in range(num_iterations):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0

        for task in range(meta_batch_size):
            learner = maml.clone()                  # parte SIEMPRE desde theta limpio
            batch = tasksets.train.sample()
            ev_err, ev_acc = fast_adapt_MAML(batch, learner, loss,
                                             adaptation_steps, shots, ways, device)
            ev_err.backward()                       # ACUMULA meta-gradiente en theta
            meta_train_error    += ev_err.item()
            meta_train_accuracy += ev_acc.item()

        # promediar el gradiente acumulado sobre el meta-batch
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()                                  # UN solo paso mueve theta
```

Tres mecanismos merecen subrayarse:

**`first_order=True` activa FOMAML.** El meta-gradiente exacto de MAML requiere derivar a través del paso de adaptación, lo que produce un término de **segundo orden** (la matriz Hessiana de la pérdida de support). Calcularlo es caro en cómputo y memoria. FOMAML (First-Order MAML) lo **ignora**: aproxima el meta-gradiente tratando $\theta'$ como si no dependiera de $\theta$ para el término de orden superior. El paper original ya reportó que el accuracy apenas cae, mientras la velocidad casi se duplica y el consumo de VRAM baja —de ahí que el lab lo use por defecto.

**`maml.clone()` antes de cada tarea.** Cada tarea del meta-batch debe partir desde **el mismo $\theta$ limpio**. `clone()` crea una copia diferenciable del modelo sobre la que se aplica la adaptación, sin tocar el $\theta$ compartido. Si adaptáramos directamente sobre `maml`, la tarea 2 partiría desde el $\theta'$ de la tarea 1: las tareas se contaminarían entre sí y el meta-objetivo dejaría de tener sentido.

**Acumulación + promedio + un solo `step()`.** Cada `ev_err.backward()` **suma** su contribución al `.grad` de los parámetros (PyTorch acumula gradientes por defecto). Tras las 32 tareas del meta-batch, se promedia (`p.grad.mul_(1/32)`) y se da **un único** `opt.step()`. El meta-gradiente resultante apunta en la dirección que mejora la inicialización *en promedio sobre muchas tareas distintas*:

$$
\theta \leftarrow \theta - \beta \, \frac{1}{B} \sum_{i=1}^{B} \nabla_\theta \, \mathcal{L}^{\text{query}}_{\,\mathcal{T}_i}\big(\theta'_i\big)
$$

con $\beta=$ `meta_lr` y $B=$ `meta_batch_size`. Nótese que la pérdida que se deriva es la del **query** evaluada con los pesos **adaptados** $\theta'_i$, pero el gradiente se toma respecto de $\theta$. Eso es exactamente "mejorar el punto de partida para que la adaptación funcione".

**Meta-testing.** Tras entrenar, se evalúa sobre **2000 tareas** de `tasksets.test` —clases nunca vistas—. El protocolo por tarea es idéntico: `clone()`, adaptar sobre el support con `fast_lr`, medir en el query. Se reporta el accuracy promedio. Aquí ya no hay `opt.step()`: $\theta$ está congelado; solo medimos qué tan buen punto de partida resultó.

## 5. La intuición binivel

La estructura de dos niveles es lo que distingue a MAML de un entrenamiento normal. Vale la pena hacerla explícita:

- **Bucle interno (rápido, efímero).** Aprende UNA tarea con `fast_lr` grande y luego **descarta** $\theta'$. No nos importa retener lo aprendido de esa tarea concreta; solo nos importa *cuánto mejoró* tras adaptarse.
- **Bucle externo (lento, persistente).** Mejora la **inicialización** $\theta$ con `meta_lr` pequeño. Eso es lo único que sobrevive entre tareas: el punto de partida.

Esto es [optimización binivel](/fundamentos/optimizacion-binivel): un problema de optimización (externo, sobre $\theta$) cuya función objetivo depende de la solución de otro problema de optimización (interno, sobre $\theta'$).

### Un mini-ejemplo numérico

Imagina un parámetro escalar $\theta$ y dos tareas cuyos óptimos están en posiciones distintas: la tarea A tiene su mínimo en $+10$ y la tarea B en $-4$. ¿Dónde debería colocar MAML la inicialización?

- **No** en $+10$: desde ahí, un paso de gradiente hacia la tarea B tendría que recorrer 14 unidades.
- **No** en $-4$: el problema simétrico para la tarea A.
- **Sí** en un punto de compromiso, digamos $\approx 3$: desde ahí, **un solo paso** de adaptación con `fast_lr` acerca razonablemente a $+10$ *o* a $-4$, según qué tarea toque.

MAML no busca resolver ninguna tarea en particular: busca el $\theta$ desde el cual **un puñado de pasos de gradiente** lleva a la solución de *cualquiera* de las tareas. El meta-gradiente del bucle externo es precisamente la fuerza que arrastra $\theta$ hacia ese punto de compromiso, promediando sobre el meta-batch.

Y aquí cierra el círculo con `clone()`: el experimento mental solo tiene sentido si **cada tarea parte desde el mismo $\theta$**. Si la tarea B partiera del $\theta'$ ya movido hacia $+10$ por la tarea A, el "punto de compromiso" que aprendiéramos estaría sesgado por el orden de las tareas. `clone()` garantiza que las 32 tareas del meta-batch midan, todas, la calidad del *mismo* punto de partida limpio.

## Enlaces

- **Papers:** [MAML (Finn 2017)](/papers/maml-finn-2017) · [learn2learn (Arnold 2020)](/papers/learn2learn-arnold-2020) · [Optimization as a Model for Few-Shot Learning (Ravi 2017)](/papers/ravi-optimization-fewshot-2017)
- **Fundamentos:** [Optimización binivel](/fundamentos/optimizacion-binivel) · [Meta-aprendizaje](/fundamentos/meta-aprendizaje) · [Few-shot learning](/fundamentos/few-shot-learning)
- **Experimentos:** [Experimentos con MAML (Act. 1-3)](experimentos-maml)
- **Clase:** [Clase 26 - Meta-aprendizaje](/clases/clase-26)
