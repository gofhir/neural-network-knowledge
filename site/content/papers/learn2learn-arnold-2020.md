---
title: "learn2learn (librería de meta-learning)"
weight: 268
math: true
---

{{< paper-card
    title="learn2learn: A Library for Meta-Learning Research"
    authors="Sébastien M. R. Arnold, Praateek Mahajan, Debajyoti Datta, Ian Bunner, Konstantinos Saitas Zarkias"
    year="2020"
    venue="arXiv"
    pdf="/papers/learn2learn-arnold-2020.pdf"
    arxiv="2008.12284" >}}
Este no es un paper de un *algoritmo* nuevo: es un **paper de software**, un *tools paper*. Su tesis es que hacia 2020 el meta-aprendizaje no estaba frenado por falta de ideas sino por dos problemas de ingeniería: **prototipado** propenso a errores —porque muchos métodos requieren computar *gradientes de pasos de optimización*, no de funciones— e **irreproducibilidad** —la comunidad llegó a perder los splits originales de mini-ImageNet de Vinyals et al. (2016). `learn2learn`, construida sobre PyTorch (con Cython en las rutas calientes de datos), ataca ambos con dos capas: rutinas de bajo nivel de **optimización diferenciable** para prototipar, e implementaciones y benchmarks de alto nivel para reproducir. Su patrón idiomático `clone()`/`adapt()` se volvió una forma estándar de enseñar MAML, y es la base sobre la que se construye todo el Laboratorio 26.
{{< /paper-card >}}

---

## El problema

Entre MAML (Finn et al., 2017) y este paper (2020) el meta-aprendizaje pasó de nicho a subcampo en plena ebullición, con tres familias canónicas ya consolidadas —basada en optimización (MAML, Meta-SGD), métrica (Matching/Prototypical Networks) y basada en modelos/memoria (MANN, RL²). Lo que el paper diagnostica no es escasez de ideas sino **fragmentación de implementaciones**: cada laboratorio reimplementaba MAML, cada paper de meta-RL usaba su propia función de recompensa sobre el mismo ambiente, y los datasets estándar derivaban silenciosamente. El ejemplo emblemático: la comunidad perdió los splits originales de mini-ImageNet de Vinyals et al. (2016), obligando a trabajos posteriores a replicarlos "lo mejor que pudieran" — variabilidad oculta que contamina toda comparación posterior. En palabras del paper, los investigadores "gastan cantidades desmesuradas de tiempo implementando software en vez de entender y desarrollar ideas nuevas".

¿Por qué es tan propenso a errores prototipar meta-aprendizaje? Porque muchos métodos requieren **computar gradientes de algoritmos, no de funciones** — por ejemplo, el gradiente de un paso de optimización. Esto es posible en PyTorch, TensorFlow o JAX, pero —cita el paper— "extenuante y propenso a errores". La tesis de ingeniería: muchos algoritmos diferenciables se pueden implementar con cambios menores *si se cuenta con las abstracciones correctas*. La librería provee esas abstracciones.

El paper se diferencia explícitamente de dos librerías contemporáneas. **`higher`** (Grefenstette et al., 2019) trata el modelo como un grafo de cómputo simbólico con parametrización *stateless*, tan expresiva como `learn2learn.optim` pero que obliga al investigador a razonar sobre qué conjunto de parámetros está activo en cada punto; además ignora la reproducibilidad. **`Torchmeta`** (Deleu et al., 2019) unifica datasets de few-shot vision, pero soportar uno nuevo exige escribir una *clase puente* aun si ya está en formato PyTorch estándar, y su wrapper algorítmico no es compatible con la mayoría de capas ni con módulos custom. La conclusión comparativa: al momento de escritura, ninguna de las dos rivales soportaba meta-descent ni meta-RL, mientras `learn2learn` ofrece una solución más general.

---

## La arquitectura de la librería

La contribución es una **arquitectura de software en dos niveles** que separa limpiamente las dos capas del problema:

1. **Rutinas de bajo nivel para prototipar:** optimización diferenciable en `learn2learn.optim`, y muestreo de tareas vía `TaskDataset`/`TaskTransform` (few-shot) y `MetaEnv`/`gym` (meta-RL).
2. **Implementaciones y benchmarks de alto nivel para reproducir:** wrappers probados (`MAML`, `GBML` con Meta-SGD/Meta-Curvature/Meta-KFO) y benchmarks estandarizados (`learn2learn.vision.benchmarks`, `learn2learn.gym`) con ejemplos que reproducen experimentos publicados *exactamente*.

La idea que une todo: las implementaciones de alto nivel se *construyen sobre* las rutinas de bajo nivel. Un investigador puede usar `l2l.algorithms.MAML` como caja negra para reproducir, o bajar a `l2l.optim` para prototipar una variante propia, *sin cambiar de librería ni de estilo de código*. Esa continuidad sin fricción es la aportación de ingeniería que la diferencia.

### `learn2learn.optim` — optimización diferenciable (el corazón)

Es el módulo conceptualmente más importante, porque es donde vive el mecanismo que hace posible MAML y toda su familia. El problema: cómo expresar una actualización de optimización de modo que **el propio acto de actualizar siga siendo diferenciable**, para poder retropropagar a través de él. El paper lo ilustra con un paso del optimizador Kronecker-factored:

```python
learned_update = l2l.optim.ParameterUpdate(
    model.parameters(),
    l2l.optim.KroneckerTransform(l2l.nn.KroneckerLinear),
)
clone = l2l.clone_module(model)          # torch.clone() para nn.Modules
updates = learned_update(                # API similar a torch.autograd.grad
    loss(clone(X), y), clone.parameters(), create_graph=True,
)
l2l.update_module(clone, updates)        # actualización in-place y diferenciable
loss(clone(X), y).backward()             # gradientes w.r.t. parámetros pre-update
```

Las piezas: **`ParameterUpdate`** computa el gradiente de una pérdida y lo pasa por un *gradient transform* (un módulo que mapea gradientes a actualizaciones), con API deliberadamente análoga a `torch.autograd.grad` para no imponer un paradigma nuevo. **`clone_module`** crea una copia diferenciable del modelo ("`torch.clone()` para nn.Modules"). **`update_module`** aplica la actualización *in-place* de modo que la actualización misma queda registrada en el grafo de autograd. El `.backward()` final retropropaga hasta los parámetros pre-update del modelo *y* hasta los del transform (el optimizador aprendido). El paper subraya el ahorro: lograr lo mismo con PyTorch puro requiere **10× más líneas de código**.

### `learn2learn.algorithms` — wrappers de alto nivel

Los wrappers "envuelven módulos de PyTorch para extenderlos con funcionalidades de meta-aprendizaje". La demostración más elegante: **Meta-SGD, Meta-Curvature y Meta-KFO son el mismo wrapper `GBML` con distinto transform**:

```python
meta_sgd       = l2l.algorithms.GBML(model, l2l.optim.ModuleTransform(l2l.nn.Scale))
meta_curvature = l2l.algorithms.GBML(model, l2l.optim.MetaCurvatureTransform)
meta_kfo       = l2l.algorithms.GBML(model,
                    l2l.optim.KroneckerTransform(l2l.nn.KroneckerLinear),
                    adapt_transform=True)
```

Tres algoritmos publicados distintos colapsan en una sola clase parametrizada que difiere solo en *cómo se transforman los gradientes de fast-adaptation*. **`l2l.algorithms.MAML`** —el que el Lab 26 usa directamente— es un caso particular de este patrón donde el transform es la identidad escalada por el learning rate del inner loop: envuelve un módulo y expone `.clone()` (copia adaptable que preserva el grafo) y `.adapt(loss)` (un paso del inner loop sobre el clone).

### `learn2learn.data` — `TaskDataset` y transforms composables

Para prototipar dominios de few-shot, la librería ofrece `TaskDataset`, que muestrea "tareas pequeñas" a partir de un dataset grande aplicando una serie de **`TaskTransforms`** composables:

```python
dataset = l2l.data.MetaDataset(MyDataset())   # cualquier dataset PyTorch
transforms = [
    l2l.data.transforms.NWays(dataset, n=5),
    l2l.data.transforms.KShots(dataset, k=1),
    l2l.data.transforms.LoadData(dataset),
    lambda task: [(random_rotation(x), y) for x, y in task],  # augmentación custom
]
taskset = l2l.data.TaskDataset(dataset, transforms, num_tasks=20000)
```

`MetaDataset` envuelve *cualquier* dataset PyTorch sin clase puente —lo que lo distingue de Torchmeta— indexándolo por etiqueta para muestrear por clase eficientemente (de ahí el uso de Cython). Los transforms se aplican en orden: `NWays` selecciona $N$ clases, `KShots` selecciona $K$ ejemplos por clase, `LoadData` materializa los tensores, y el `lambda` aplica una augmentación arbitraria. La lección de diseño: el setting episódico $N$-way $K$-shot **no está hardcodeado, sino que emerge de componer transforms**, reteniendo compatibilidad con cualquier dataset PyTorch (incluido texto y audio vía `l2l.text`).

### El mecanismo: cómo `clone()` preserva el grafo para el meta-gradiente de 2º orden

Esta es la parte que separa entender la librería de leer su README. El meta-gradiente de MAML "involves a gradient through a gradient": para retropropagar el meta-objetivo hasta los parámetros pre-update $\theta$, el grafo del *paso de adaptación* (inner loop) debe permanecer intacto.

**El problema con PyTorch puro.** Un `nn.Module` mantiene sus parámetros como hojas (leaf tensors) del grafo de autograd. Cuando uno hace `optimizer.step()`, la actualización ocurre dentro de `torch.no_grad()` e *in-place* sobre los `.data`, **rompiendo deliberadamente el grafo** —porque en entrenamiento normal no queremos retropropagar a través de la actualización del optimizador. MAML necesita lo contrario: el parámetro adaptado

$$
\theta' = \theta - \alpha \nabla_\theta \mathcal{L}(f_\theta)
$$

debe seguir siendo un *nodo no-hoja* del grafo, con $\theta$ como ancestro, para que al hacer `.backward()` sobre la pérdida post-update aparezca el término Hessiano que caracteriza el segundo orden:

$$
\nabla_\theta \mathcal{L}(f_{\theta'}) = \big(I - \alpha \nabla_\theta^2 \mathcal{L}(f_\theta)\big)\, \nabla_{\theta'} \mathcal{L}(f_{\theta'}).
$$

**Qué hace `clone_module`/`MAML.clone()`.** En vez de copiar los `.data` (lo que crearía hojas desconectadas), reconstruye el módulo de modo que sus parámetros sean *referencias a los tensores del modelo dentro del grafo*: el clone comparte el grafo computacional con el padre. Al computar el gradiente con `create_graph=True` y aplicar `update_module` —que reasigna $\theta' = \theta - \alpha g$ *sin* `torch.no_grad()`—, el resultado $\theta'$ queda registrado en el grafo con $\theta$ y $g$ como ancestros. La bandera `create_graph=True` ordena a autograd construir el grafo *del cálculo del gradiente mismo*, habilitando la derivada de segundo orden; sin ella, el meta-gradiente colapsa a la aproximación de primer orden (FOMAML).

---

## Decisiones de diseño

- **Diferenciabilidad de alto orden como ciudadano de primera clase.** Toda la capa de bajo nivel se construye "estrechamente alrededor del motor de autodiferenciación de PyTorch". La consecuencia: cualquier algoritmo expresable como "optimización cuyo grafo permanece diferenciable" cabe en la librería — no solo MAML, sino meta-descent e hypergradients.
- **Unificación vía composición, no vía herencia.** Tanto los algoritmos (GBML parametrizado por un transform) como los dominios (TaskDataset parametrizado por una lista de transforms) usan composición de objetos pequeños en vez de jerarquías de clases. Tres algoritmos distintos = un GBML + tres transforms; el setting $N$-way $K$-shot = composición de `NWays` + `KShots` + `LoadData`. Menos superficie de código donde alojar bugs.
- **Compatibilidad universal con PyTorch.** Decisión deliberada frente a Torchmeta: `TaskDataset` acepta *cualquier* dataset PyTorch sin clase puente, y `l2l.optim` maneja *todos* los `Module`, incluidos los custom. El precio de admisión es cero refactor del código existente.
- **Eficiencia donde duele.** PyTorch para autograd y álgebra lineal; **Cython** para el manejo de datos (el muestreo por clase de `MetaDataset`, cuello de botella del loop episódico). Se optimiza solo la ruta caliente.
- **Reproducibilidad como objetivo de diseño, no afterthought.** Los ejemplos que reproducen experimentos publicados validan la corrección de la implementación, sirven de documentación ejecutable y llenan la necesidad de reproducciones estandarizadas — y de paso bootstrapean experimentación nueva alrededor de cada método.

---

## Impacto

Al ser un *tools paper*, no reporta tablas con números propios: reporta **cobertura y fidelidad**. En few-shot vision cubre Omniglot, mini-ImageNet, CIFAR-FS y FC100 con settings 5-way 1-shot, 5-way 5-shot y 20-way 5-shot, incluyendo las etapas de procesamiento correctas (normalización, rotación, cropping). En algoritmos reproduce MAML, ANIL, Meta-SGD, Meta-Curvature, Meta-KFO, Hypergradient descent y ProMP "exactamente". En meta-RL (`learn2learn.gym`) abarca desde navegación 2D de partícula hasta control robótico (half-cheetah, ant) más wrappers para **MetaWorld** (50 tareas de manipulación), con utilidades como `AsyncVectorEnv` para paralelizar la recolección de episodios.

El paper declara la librería bajo licencia MIT, ya usada en la investigación diaria de sus autores, con el objetivo de construir comunidad alrededor de software estandarizado. En el ecosistema, `learn2learn` se volvió una de las librerías de referencia para meta-aprendizaje en PyTorch junto a `higher` y `Torchmeta`, ampliamente usada en cursos, tutoriales y papers de few-shot. Su patrón `clone()`/`adapt()` se consolidó como forma idiomática de *enseñar* MAML, porque hace visible el ciclo inner/outer sin enterrar al estudiante en la mecánica de `create_graph`. Sus límites, leídos entre líneas: atadura total a PyTorch (sin backend TensorFlow ni JAX), foco en familias de meta-aprendizaje *ya conocidas* (lo que no encaja en "optimización diferenciable" o "muestreo episódico" requiere trabajo fuera del marco), y ausencia de benchmarks de rendimiento más allá del claim cualitativo de "10× menos líneas".

---

## Conexión con el Laboratorio 26

El Lab 26 no solo *menciona* esta librería: **se construye enteramente sobre ella**. Pieza por pieza:

- **Muestreo de tareas episódicas $N$-way $K$-shot.** El lab usa `l2l.data.MetaDataset` para envolver Omniglot/Mini-ImageNet y compone los transforms `NWays(n=...)`, `KShots(k=...)`, `LoadData` y `RemapLabels` dentro de `l2l.data.TaskDataset` — exactamente el patrón del paper. Cada transform tiene su razón: `NWays` elige las $N$ clases del episodio, `KShots` toma $K$ ejemplos por clase (support + query, donde el lab típicamente pide `2*K` y parte en mitades), `LoadData` materializa los tensores, y `RemapLabels` reindexa las etiquetas globales (p. ej. clases 412, 87, 1003…) a etiquetas locales $0..N{-}1$ *por episodio*, requisito para que la cross-entropy de $N$ clases tenga sentido. La lección central: el setting episódico **emerge de componer transforms**, no está hardcodeado.

- **Carga de datasets.** `l2l.vision.benchmarks.get_tasksets('omniglot', ...)` entrega directamente un *namedtuple* con `train`/`validation`/`test`, cada uno un `TaskDataset` ya particionado — esto resuelve de raíz el problema de los splits perdidos de mini-ImageNet: hay *un* pipeline canónico. El lab puede usar ese nivel o bajar a `l2l.vision.datasets` (descarga y preprocesado) según cuánto control quiera el estudiante.

- **Envoltura del modelo para MAML.** El lab hace `maml = l2l.algorithms.MAML(model, lr=inner_lr, first_order=...)`. Por cada tarea del meta-batch:

  ```python
  learner = maml.clone()                 # copia diferenciable; comparte grafo con maml
  for step in range(adaptation_steps):   # inner loop sobre el support set
      support_error = loss(learner(support_X), support_y)
      learner.adapt(support_error)       # θ' = θ − α∇θ L; queda en el grafo
  query_error = loss(learner(query_X), query_y)   # evaluación en el query set
  query_error.backward()                 # acumula meta-gradiente en maml.parameters()
  ```

  Aquí el mecanismo de `clone()` se vuelve tangible: **es lo que permite que el `backward()` del query error llegue hasta los pesos originales** y materialice el "gradiente a través del gradiente" de MAML. Con `first_order=False` (default) aparece el segundo orden con su término Hessiano; con `first_order=True` se obtiene FOMAML, más barato y casi tan bueno. El estudiante escribe el ciclo conceptual —clonar → adaptar en support → evaluar en query → meta-actualizar— *sin tocar nunca* `create_graph` ni `torch.no_grad()`: la librería encapsula la fontanería "exótica y propensa a errores" que el paper identifica como causa raíz del problema de prototipado.

- **Modelos provistos.** El lab usa las arquitecturas listas de `l2l.vision.models`: `OmniglotFC` (MLP) y `OmniglotCNN` para Omniglot, `MiniImagenetCNN` y `ResNet12` para Mini-ImageNet — los backbones canónicos de la literatura few-shot, ya cableados con las dimensiones y normalizaciones correctas.

- **Puente hacia salud.** La compatibilidad de `TaskDataset` con cualquier dataset PyTorch (incluido texto/tabular vía `l2l.text`) conecta el ejercicio canónico de visión con los dominios reales del estudiante: tratar cada patología rara, cada sitio/escáner o cada subpoblación como una "tarea" $\mathcal{T}_i$ y meta-entrenar una inicialización que se adapte con pocos ejemplos. La librería hace ese salto barato porque no exige reescribir el pipeline de datos — solo componer transforms distintos sobre el mismo `TaskDataset`.

---

## Notas y enlaces

Fundamentos: [Meta-aprendizaje](/fundamentos/meta-aprendizaje) - [Optimización binivel](/fundamentos/optimizacion-binivel) - [Few-shot learning](/fundamentos/few-shot-learning)

Papers relacionados: [MAML (Finn 2017)](/papers/maml-finn-2017) - [Prototypical Networks (Snell 2017)](/papers/prototypical-networks-snell-2017) - [Optimization as a Model for Few-Shot Learning (Ravi 2017)](/papers/ravi-optimization-fewshot-2017)

Laboratorio: [Lab 26 - Meta-aprendizaje](/laboratorios/lab-26)

Clase: [Clase 26 - Meta-aprendizaje](/clases/clase-26)
