---
title: "Prototypical Networks: clasificar por distancia"
weight: 4
math: true
---

> **Parte 2 del notebook.** Después de MAML, el lab cambia de paradigma. Prototypical Networks (Snell et al., 2017) no adapta el modelo por gradiente en test: aprende un **espacio de embeddings** donde cada clase se resume en un **prototipo** —el centroide de los embeddings de su *support set*— y clasifica cada *query* asignándola al prototipo más cercano. El setup de datos es el mismo de la Parte 1 (Omniglot, Mini-ImageNet, `learn2learn`), pero la mecánica interna no podría ser más distinta: aquí no hay `clone()`, ni bucle interno, ni segundo orden. Solo un encoder, un promedio y una distancia.

## Encuadre: metric-based frente a optimization-based

MAML pertenece a la familia **optimization-based**: aprende una inicialización $\theta$ de parámetros tal que, partiendo de ahí, **pocos pasos de descenso de gradiente** sobre el *support set* alcanzan una buena solución para la tarea nueva. La adaptación en test es, literalmente, entrenar un poco.

Prototypical Networks pertenece a la familia **metric-based** (basada en métrica). La idea es la opuesta: en lugar de aprender *cómo adaptarse rápido*, aprende un **espacio de representación** donde clasificar es trivial —basta medir distancias—. En test **no se ajusta ningún parámetro**: se calcula un embedding por cada ejemplo, se promedian los del *support* para obtener un prototipo por clase, y cada *query* se etiqueta con la clase de su prototipo más cercano. Toda la inteligencia vive en el encoder, entrenado de antemano.

Esa diferencia tiene consecuencias prácticas enormes —cero adaptación en test, sin segundo orden, independencia del número de clases— que iremos desplegando. La conexión conceptual es directa con [Matching Networks (Vinyals 2016)](/papers/matching-networks-vinyals-2016) y con las [redes siamesas (Koch 2015)](/papers/siamese-networks-koch-2015): las tres aprenden a *comparar* en vez de a *clasificar* con una cabeza fija. Ver el [fundamento de metric learning](/fundamentos/metric-learning).

## `pairwise_distances_logits(a, b)`: la distancia como logit

El primer ingrediente es la función que convierte distancias en logits. Recibe dos conjuntos de embeddings —`a` son las *queries* $(N, D)$ y `b` los prototipos $(M, D)$— y devuelve, por cada par, la **distancia euclidiana al cuadrado negada**:

```python
def pairwise_distances_logits(a, b):
    n = a.shape[0]
    m = b.shape[0]
    logits = -((a.unsqueeze(1).expand(n, m, -1) -
                b.unsqueeze(0).expand(n, m, -1)) ** 2).sum(dim=2)
    return logits
```

El truco es el *broadcasting*. `a.unsqueeze(1)` pasa de $(N, D)$ a $(N, 1, D)$ y `b.unsqueeze(0)` de $(M, D)$ a $(1, M, D)$; al restarlos PyTorch los expande a $(N, M, D)$, restando **cada query contra cada prototipo**. Se eleva al cuadrado elemento a elemento y se suma sobre la dimensión $D$ (`dim=2`), dejando una matriz $(N, M)$ donde la entrada $(i, j)$ es

$$
\text{logit}_{ij} = -\lVert a_i - b_j \rVert_2^2 .
$$

El signo negativo es clave: **mientras más cerca está la query del prototipo, mayor es el logit**. Esos $M$ logits por query van directos a una *cross-entropy* (que aplica `softmax` internamente), de modo que el prototipo más cercano recibe la probabilidad más alta. En otras palabras, la regla "gana el prototipo más cercano" emerge sola al minimizar la *cross-entropy* sobre estos logits.

> **¿Por qué euclidiana y no coseno?** Snell et al. demostraron que usar la **distancia euclidiana al cuadrado** no es un capricho. Esa distancia es una *divergencia de Bregman*, y para esa familia el **promedio aritmético** del *support* es exactamente el estimador que minimiza la distancia total a sus puntos. Es decir: tomar el centroide como prototipo es *óptimo* precisamente bajo la métrica euclidiana. Con distancia coseno esa garantía se pierde, y empíricamente el paper reporta peor desempeño. La elección de la métrica y la del prototipo (la media) están acopladas teóricamente.

## `fast_adapt_protonet`: el nombre engaña

La función conserva el nombre `fast_adapt` por simetría con la Parte 1 (MAML), pero **aquí no hay adaptación por gradiente**. No clona el modelo, no corre pasos internos, no toca parámetros. Lo que hace es: codificar el *batch*, separar *support* de *query*, **promediar los embeddings de support para formar prototipos**, y medir distancias.

```python
def fast_adapt_protonet(model, batch, ways, shots, metric=None, device=None):
    if metric is None:
        metric = pairwise_distances_logits
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # 1) Ordenar por etiqueta: tras esto, el reshape agrupará por clase
    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)

    # 2) Codificar TODO el batch con el encoder
    embeddings = model(data)

    # 3) Separar support / query con el truco par-impar
    support_indices = np.zeros(data.size(0), dtype=bool)
    support_indices[np.arange(shots * ways) * 2] = True   # posiciones pares = support
    query_indices = torch.from_numpy(~support_indices)    # el resto = query
    support_indices = torch.from_numpy(support_indices)

    support = embeddings[support_indices]

    # 4) EL CORAZÓN: un prototipo por clase = media de sus embeddings de support
    support = support.reshape(ways, shots, -1).mean(dim=1)

    query = embeddings[query_indices]
    labels = labels[query_indices].long()

    # 5) Distancias -> logits -> cross-entropy
    logits = metric(query, support)
    loss = F.cross_entropy(logits, labels)
    acc = accuracy(logits, labels)
    return loss, acc
```

Tres detalles merecen lupa:

- **`torch.sort(labels)` no es cosmético.** El `reshape(ways, shots, -1)` solo agrupa correctamente los embeddings por clase si las muestras vienen **contiguas por etiqueta**. Ordenar primero garantiza que las primeras `shots` filas pertenezcan a la clase 0, las siguientes `shots` a la clase 1, y así. Sin el `sort`, el `reshape` mezclaría clases en cada fila y los prototipos saldrían sin sentido.
- **El encoder se aplica una sola vez a todo el batch.** `model(data)` codifica *support* y *query* juntos; recién después se separan por índice. Es más eficiente que pasarlos por separado y, como el encoder es el mismo, no hay diferencia de resultado.
- **El truco par-impar.** Cada tarea trae `2 * shots` muestras por clase (`train_samples=2 * shots` en la construcción del *taskset*). Las posiciones pares (`np.arange(shots*ways)*2`) se marcan como *support*; las impares quedan como *query*. Así cada clase aporta `shots` ejemplos al prototipo y `shots` ejemplos a evaluar.

El paso 4 es, literalmente, todo Prototypical Networks. El prototipo de la clase $k$ es

$$
c_k = \frac{1}{|S_k|} \sum_{(x_i, y_i) \in S_k} f_\phi(x_i),
$$

donde $f_\phi$ es el encoder y $S_k$ el *support* de la clase $k$. Y la probabilidad de que una query $x$ sea de la clase $k$ es

$$
p_\phi(y = k \mid x) = \frac{\exp\!\left(-\lVert f_\phi(x) - c_k \rVert^2\right)}{\sum_{k'} \exp\!\left(-\lVert f_\phi(x) - c_{k'} \rVert^2\right)},
$$

que es exactamente el `softmax` de los logits que produce `pairwise_distances_logits`.

## `run_Protonet`: entrenamiento normal, no binivel

Aquí se ve la mayor diferencia estructural con MAML. El entrenamiento de Prototypical Networks es **supervisado plano**: un optimizador, un `loss.backward()`, un `optimizer.step()` por tarea. No hay optimización binivel (inner/outer loop), no hay segundo orden, no hay `clone()`.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

for epoch in range(1, num_epochs + 1):
    model.train()
    for task in range(num_train_tasks_per_epoch):       # 200 tareas / época
        batch = tasksets.train.sample()
        loss, acc = fast_adapt_protonet(model, batch, ways, shots, device=device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()                                # SGD directo, una vez por tarea
    lr_scheduler.step()                                 # ↓ LR a la mitad cada 20 épocas

    model.eval()
    for task in range(num_val_tasks_per_epoch):
        batch = tasksets.validation.sample()
        loss, acc = fast_adapt_protonet(model, batch, ways, shots, device=device)
```

Los parámetros del esquema:

| Elemento | Valor | Rol |
|---|---|---|
| Optimizador | `Adam(lr=0.001)` | Un solo optimizador sobre todo el encoder |
| Scheduler | `StepLR(step_size=20, gamma=0.5)` | Multiplica el LR por 0.5 cada 20 épocas |
| Tareas por época | `num_train_tasks_per_epoch=200` | Cada una = 1 paso de gradiente |
| `backward`/`step` | Una vez por tarea | SGD plano, sin bucle interno |
| `clone()` / 2º orden | **Ninguno** | No existe optimización binivel |
| Meta-testing | 2000 tareas | Solo `forward` + distancias, sin entrenar |

El `StepLR` baja el *learning rate* a la mitad cada 20 épocas; en una corrida de 80 épocas eso lo divide por $2^4 = 16$, de modo que la segunda mitad del entrenamiento es **refinamiento fino**.

> **¿Dónde está el "meta" entonces?** No está en la estructura del bucle —que es un entrenamiento corriente— sino en el **objetivo**. Cada tarea es un episodio *few-shot* nuevo con clases potencialmente distintas, y la *loss* premia al encoder por producir embeddings cuyo **promedio** clasifica bien clases que no ha visto. El encoder no aprende a reconocer las clases del *train*: aprende a construir un espacio donde *cualquier* puñado de ejemplos define un buen prototipo. Ese es el sabor meta, escondido en la función de pérdida, no en la maquinaria.

## El modelo es un encoder, no un clasificador

Este es el punto que más se malinterpreta. En MAML el modelo termina en una capa `Linear(..., ways)` que produce un logit por clase. En Prototypical Networks **el modelo no tiene cabeza de clasificación**: es puro extractor de *features*, y su salida es un **vector de embedding**.

```python
model = l2l.vision.models.OmniglotFC(28 ** 2, 50)  # 50 = DIMENSIÓN DEL EMBEDDING
```

El `50` **no es WAYS**. Es la dimensión $D$ del espacio de embeddings. El modelo recibe una imagen de Omniglot ($28 \times 28 = 784$ píxeles) y devuelve un vector de 50 dimensiones. La clasificación ocurre *fuera* del modelo, en `pairwise_distances_logits`, comparando ese vector con los prototipos.

La consecuencia es enorme:

> **El modelo no depende de WAYS.** Como no hay capa de salida atada al número de clases, **puedes entrenar con un WAYS y evaluar con otro sin reinstanciar nada**. Un encoder entrenado en episodios de 5 clases clasifica perfectamente episodios de 20 clases: solo cambia cuántos prototipos calculas. En MAML esto es **imposible** sin cirugía: su `Linear(..., ways)` tiene forma fija, así que cambiar WAYS obliga a crear un modelo nuevo y reentrenar. Por eso el lab puede barrer `ways = 2, 4, 8` (Actividad 5) reusando la misma arquitectura.

Lo mismo aplica con `MiniImagenetCNN(50)` o `ResNet12(output_size=50)` en las actividades de Mini-ImageNet: el argumento numérico es siempre la dimensión del embedding, nunca el número de clases.

## MAML frente a Prototypical Networks, lado a lado

| | **MAML** (optimization-based) | **Prototypical** (metric-based) |
|---|---|---|
| **Paradigma** | Aprender a optimizar | Aprender a comparar |
| **Qué aprende** | Inicialización $\theta$ de parámetros | Espacio de embeddings $f_\phi$ |
| **Adaptación en test** | Pocos pasos de gradiente sobre support | **Ninguna**: solo calcula distancias |
| **Estructura de entrenamiento** | Binivel (inner + outer loop) | Supervisado normal (un solo loop) |
| **`clone()` / 2º orden** | Sí (segundo orden o aprox. FOMAML) | **No** |
| **Costo en test** | Forward + backward (varios pasos) | Solo forward + distancias |
| **Salida del modelo** | WAYS logits (cabeza fija) | Vector de embedding ($D$ dims) |
| **¿Depende de WAYS?** | Sí (capa de salida atada) | **No** (mismo encoder para cualquier WAYS) |

En el lab, esta simplicidad estructural se traduce en resultados parejos o mejores: con una CNN como encoder, Prototypical iguala al mejor MAML afinado (≈0.877 en Omniglot 4-way 1-shot) a las 40 épocas y lo supera (≈0.934) a las 80, sin segundo orden ni búsqueda de hiperparámetros. Ver los [experimentos con Prototypical (Act. 4-6)](experimentos-prototypical).

## Conexión FHIR / record linkage

Para quien viene de *record linkage* clínico, Prototypical Networks describe casi literalmente la arquitectura de un **bi-encoder para *patient matching***:

- **El encoder $f_\phi$** mapea un registro de paciente (nombre, RUT, fecha de nacimiento, dirección, atributos FHIR) a un vector en un espacio de embeddings —exactamente lo que hace `model(data)`—.
- **El prototipo $c_k$** —centroide de los embeddings— representa a una **entidad-paciente** consolidada: el "golden record" como promedio de todas sus apariciones conocidas.
- **Clasificar por distancia** es *blocking* y *matching* por cercanía: un registro entrante se asigna a la entidad cuyo prototipo está más próximo en el espacio, o se marca como entidad nueva si ninguno está suficientemente cerca.

Es decir, Prototypical Networks **es** el paradigma de un *blocker* de MDM (Master Data Management): no compara pares de registros con reglas, sino que proyecta todo a un espacio donde la proximidad geométrica *es* la evidencia de identidad. La elección euclidiana del paper se traduce aquí en que el centroide de las variantes de un paciente es el mejor representante de su identidad —una propiedad deseable cuando se consolidan múltiples fuentes con ruido—. La diferencia respecto a un clasificador supervisado tradicional es la misma que en *few-shot*: el sistema generaliza a pacientes nunca vistos sin reentrenar, porque aprendió a *comparar* en lugar de a *enumerar* identidades. Ver el [fundamento de few-shot learning](/fundamentos/few-shot-learning).

## Enlaces

- **Papers:** [Prototypical Networks (Snell 2017)](/papers/prototypical-networks-snell-2017) · [Matching Networks (Vinyals 2016)](/papers/matching-networks-vinyals-2016) · [Siamese Networks (Koch 2015)](/papers/siamese-networks-koch-2015) · [learn2learn (Arnold 2020)](/papers/learn2learn-arnold-2020)
- **Fundamentos:** [Metric Learning](/fundamentos/metric-learning) · [Meta-aprendizaje](/fundamentos/meta-aprendizaje) · [Few-shot learning](/fundamentos/few-shot-learning)
- **Experimentos:** [Experimentos con Prototypical (Act. 4-6)](experimentos-prototypical)
- **Clase:** [Clase 26 - Meta-aprendizaje](/clases/clase-26)
