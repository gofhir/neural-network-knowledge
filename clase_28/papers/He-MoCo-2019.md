# Momentum Contrast for Unsupervised Visual Representation Learning (MoCo) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Momentum Contrast for Unsupervised Visual Representation Learning*.
- **Autores:** Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick (Facebook AI Research, FAIR).
- **Venue:** CVPR 2020 (Conference on Computer Vision and Pattern Recognition).
- **Año:** 2019 (preprint) / 2020 (publicación). **Preprint:** arXiv:1911.05722v3 (23 mar 2020), [arxiv.org/abs/1911.05722](https://arxiv.org/abs/1911.05722).
- **Código:** [github.com/facebookresearch/moco](https://github.com/facebookresearch/moco).

MoCo (Momentum Contrast) es uno de los trabajos que **cierra la brecha entre el preentrenamiento no supervisado y el supervisado** en visión por computador. Su tesis, simple de enunciar y de gran impacto, parte de mirar el aprendizaje contrastivo bajo una sola lente: como una tarea de **búsqueda en diccionario** (*dictionary look-up*). Bajo esa lente, un *query* codificado `q` debe parecerse a su clave positiva `k+` (otra vista de la misma imagen) y diferenciarse de un conjunto de claves negativas. El problema que el paper identifica con claridad quirúrgica es que, para aprender buenas representaciones, ese diccionario de claves debería ser a la vez **(i) grande** —para muestrear densamente el espacio visual continuo y de alta dimensión— y **(ii) consistente** —las claves deberían estar codificadas por un encoder igual o muy similar, para que sus comparaciones contra el query tengan sentido—. Los mecanismos previos sacrificaban uno de los dos: el *end-to-end* era consistente pero pequeño (atado al *batch size*), y el *memory bank* era grande pero inconsistente (claves codificadas por encoders de toda la época anterior).

La contribución de MoCo es un mecanismo que logra *ambas* propiedades simultáneamente, con dos ideas que encajan: **una cola FIFO** (*queue*) de embeddings de mini-batches previos que actúa como diccionario grande de negativos, desacoplado del *batch size*; y **un momentum encoder** para las claves, actualizado por media móvil exponencial de los pesos del query encoder, lo que mantiene la consistencia del diccionario pese a que evoluciona durante el entrenamiento. El resultado experimental es histórico: bajo el protocolo lineal de ImageNet MoCo es competitivo (60.6% top-1 con ResNet-50), pero, más importante, **las representaciones transfieren a tareas downstream de detección y segmentación igualando o superando al preentrenamiento supervisado de ImageNet en 7 tareas** sobre PASCAL VOC, COCO y otros datasets, a veces por márgenes no triviales. Para la Clase 28 (Aprendizaje Autosupervisado) MoCo es el paradigma contrastivo eficiente en memoria: la clase tiene un slide dedicado que muestra el encoder, el momentum encoder y la cola FIFO de `K` negativos, y explica por qué este diseño requiere menos memoria que SimCLR.

## 2. Contexto histórico: del éxito en NLP a la búsqueda de diccionarios visuales

El paper abre con un contraste que motiva todo el trabajo. El aprendizaje no supervisado de representaciones es altamente exitoso en procesamiento de lenguaje natural —como demuestran GPT (Radford et al., 2018, 2019) y BERT (Devlin et al., 2019)—, pero hacia 2019 el preentrenamiento **supervisado** seguía dominando en visión, donde los métodos no supervisados generalmente quedaban rezagados. He et al. proponen una explicación elegante: la diferencia está en los **espacios de señal** respectivos. El lenguaje tiene un espacio de señal *discreto* (palabras, sub-word units) sobre el cual se pueden construir diccionarios tokenizados, base natural del aprendizaje no supervisado. La visión, en cambio, opera sobre una señal cruda *continua y de alta dimensión* que no está estructurada para la comunicación humana (a diferencia de las palabras), de modo que el propio acto de "construir el diccionario" es parte del problema.

Varios estudios recientes (Wu et al. 2018; van den Oord et al. 2018 — CPC; Hjelm et al. 2019; Bachman et al. 2019; Tian et al. 2019 — CMC; Ye et al. 2019) ya presentaban resultados prometedores con enfoques relacionados a la **pérdida contrastiva** (Hadsell, Chopra, LeCun, 2006). Aunque motivados de forma diversa, el paper los unifica conceptualmente: todos pueden pensarse como métodos que **construyen diccionarios dinámicos**. Las "claves" (tokens) del diccionario se muestrean de los datos (imágenes o parches) y se representan con una red encoder; el aprendizaje no supervisado entrena ese encoder para realizar la búsqueda en diccionario. Es justamente esta abstracción —ver todo el campo del aprendizaje contrastivo como *dictionary look-up*— la que le permite a MoCo diagnosticar la tensión tamaño-vs-consistencia y proponer una solución dirigida a ella.

El paper se posiciona también frente a la taxonomía estándar del aprendizaje autosupervisado, que distingue dos aspectos: las **pretext tasks** (la tarea sustituta que se resuelve, p.ej. recuperar el input corrupto en un denoising autoencoder, ordenar parches, colorizar, predecir rotaciones, clustering de features) y las **funciones de pérdida** (que pueden estudiarse independientemente de la pretext task). MoCo declara explícitamente que su foco es la **función de pérdida y el mecanismo del diccionario**, no el diseño de una pretext task nueva: usa la tarea simple de *instance discrimination* (Wu et al. 2018) —dos vistas aumentadas de la misma imagen son un par positivo— precisamente para aislar la contribución del mecanismo.

## 3. El problema central: tamaño vs. consistencia del diccionario

Esta es la sección conceptual que vale la pena destrozar en detalle, porque es donde vive la justificación de cada decisión de diseño. La hipótesis del paper es que se pueden aprender buenas features con un **diccionario grande que cubra un conjunto rico de negativos**, mientras el encoder de las claves se mantenga **lo más consistente posible** pese a que evoluciona durante el entrenamiento. El paper compara tres mecanismos (su Figura 2), todos sobre la misma pretext task e InfoNCE, de modo que la comparación es puramente sobre el mecanismo:

- **(a) End-to-end.** Las representaciones de query y de clave se computan con encoders que se actualizan *end-to-end* por retropropagación. Es el mecanismo más natural (lo usan CPC, CMC, AMDIM, entre otros) y es perfectamente **consistente**: las claves del batch actual se codifican con el mismo conjunto de pesos. Pero el **tamaño del diccionario está acoplado al tamaño del mini-batch**, limitado por la memoria de la GPU. En la práctica, la máquina de gama alta del paper (8 GPUs Volta de 32GB) alcanza a lo sumo un batch de 1024; y empujar el batch más allá topa con el problema abierto de la optimización con mini-batches grandes (Goyal et al. 2017), que requiere la regla de escalamiento lineal del learning rate y, aun así, es cuestionable que el truco extrapole a `K` mucho mayores.

- **(b) Memory bank** (Wu et al. 2018). Un banco de memoria almacena las representaciones de *todas* las muestras del dataset; el diccionario de cada mini-batch se muestrea aleatoriamente del banco sin retropropagación, lo que **sí permite un diccionario grande**. El precio es la **inconsistencia**: la representación de una muestra en el banco se actualizó la última vez que esa muestra se vio, de modo que las claves muestreadas provienen de encoders de *múltiples pasos distintos a lo largo de la época pasada*, y son por tanto poco consistentes entre sí. (El memory bank de Wu et al. aplica un momentum, pero sobre las *representaciones de la misma muestra*, no sobre el encoder; es un mecanismo distinto e irrelevante para MoCo.)

El diagnóstico es entonces nítido: end-to-end = consistente pero pequeño; memory bank = grande pero inconsistente. **Ninguno logra las dos propiedades a la vez.** MoCo se diseña como la respuesta directa a esa tensión.

## 4. Contribución: la cola FIFO y el momentum encoder

### 4.1. El diccionario como cola (queue) — desacoplar tamaño y batch

El núcleo de MoCo es mantener el diccionario como una **cola FIFO de muestras de datos**. Esto permite **reutilizar las claves codificadas de los mini-batches inmediatamente precedentes**. La introducción de la cola **desacopla el tamaño del diccionario del tamaño del mini-batch**: el tamaño de la cola `K` puede ser mucho mayor que un batch típico y se fija de forma flexible e independiente como hiperparámetro (en los experimentos principales, `K = 65536`).

Las muestras del diccionario se reemplazan progresivamente: el mini-batch actual se *encola* (enqueue) y el mini-batch más antiguo se *desencola* (dequeue). El diccionario siempre representa un subconjunto muestreado de todos los datos, y el coste computacional extra de mantenerlo es manejable —solo se almacenan los embeddings, no se recodifican—. He et al. notan además un beneficio sutil: *remover el batch más antiguo es beneficioso*, porque sus claves codificadas son las más desactualizadas y, por tanto, las menos consistentes con las más nuevas. La cola es, de hecho, lo que hace grande al diccionario sin pagar memoria de gradientes.

### 4.2. El momentum encoder — recuperar la consistencia

La cola resuelve el tamaño, pero crea un problema nuevo: vuelve **intratable actualizar el encoder de claves por retropropagación**, porque el gradiente debería propagarse a todas las muestras de la cola (que vienen de batches pasados). Una solución ingenua —copiar el encoder de claves `f_k` desde el encoder de queries `f_q`, ignorando ese gradiente— da malos resultados en los experimentos. El paper hipotetiza que ese fracaso se debe a que un encoder de claves que **cambia rápidamente** reduce la consistencia de las representaciones de las claves en la cola (que fueron codificadas por versiones distintas del encoder).

La solución es el **momentum update**. Denotando `θ_k` los parámetros de `f_k` y `θ_q` los de `f_q`, la actualización es:

```
θ_k ← m · θ_k + (1 − m) · θ_q          (Ec. 2)
```

donde `m ∈ [0,1)` es el coeficiente de momentum. **Solo `θ_q` se actualiza por retropropagación**; `θ_k` se arrastra como media móvil exponencial de `θ_q`. La consecuencia clave: `θ_k` evoluciona **mucho más suavemente** que `θ_q`. Así, aunque las claves de la cola fueron codificadas por encoders de distintos mini-batches, la *diferencia entre esos encoders se mantiene pequeña* porque todos son versiones suavizadas y cercanas entre sí. Esto es lo que restaura la consistencia que la cola amenazaba.

El paper enfatiza que el valor del momentum importa muchísimo: un **`m` relativamente grande (0.999, su default) funciona mucho mejor que uno pequeño (0.9)**. La tabla de ablación lo cuantifica con `K = 4096`:

| momentum `m` | 0 | 0.9 | 0.99 | 0.999 | 0.9999 |
|---|---|---|---|---|---|
| accuracy (%) | falla | 55.2 | 57.8 | 59.0 | 58.9 |

Con `m = 0` (sin momentum, copiar directo el encoder) el *training loss oscila y no converge*. El régimen útil es `m ∈ [0.99, 0.9999]`. Esto confirma empíricamente la motivación: **un encoder de claves que progresa lentamente es el núcleo que permite explotar la cola**. Es un resultado contraintuitivo —uno esperaría que el encoder de claves debiera seguir de cerca al de queries— y es precisamente la observación no trivial del paper.

### 4.3. InfoNCE — la pérdida que une todo

La pérdida es **InfoNCE** (van den Oord et al. 2018), una pérdida contrastiva donde la similitud se mide por producto punto:

```
            exp(q·k+ / τ)
L_q = − log ─────────────────────          (Ec. 1)
            Σ_{i=0}^{K} exp(q·k_i / τ)
```

La suma corre sobre una clave positiva y `K` negativas (las de la cola). Intuitivamente, es la log-loss de un clasificador softmax de `(K+1)` clases que intenta clasificar `q` como `k+`. La temperatura `τ = 0.07`. El pseudocódigo del paper (Algoritmo 1, estilo PyTorch) es revelador de la simplicidad del mecanismo:

```python
f_k.params = f_q.params                     # inicializar
for x in loader:                            # mini-batch de N muestras
    x_q, x_k = aug(x), aug(x)               # dos vistas aumentadas
    q = f_q.forward(x_q)                    # queries: NxC
    k = f_k.forward(x_k).detach()           # keys: NxC, SIN gradiente
    l_pos = bmm(q.view(N,1,C), k.view(N,C,1))    # logits positivos: Nx1
    l_neg = mm(q.view(N,C), queue.view(C,K))     # logits negativos: NxK
    logits = cat([l_pos, l_neg], dim=1)          # Nx(1+K)
    labels = zeros(N)                            # el positivo es el 0-ésimo
    loss = CrossEntropyLoss(logits / t, labels)
    loss.backward(); update(f_q.params)          # SGD solo en query encoder
    f_k.params = m*f_k.params + (1-m)*f_q.params # momentum update
    enqueue(queue, k); dequeue(queue)            # actualizar diccionario
```

El `.detach()` sobre las claves materializa el punto central: **el gradiente fluye solo por el query encoder**; el key encoder nunca se entrena por backprop, solo por la Ec. 2.

### 4.4. Detalles técnicos: Shuffling BN

Un detalle de ingeniería que el paper considera esencial. Ambos encoders usan Batch Normalization (BN) como en el ResNet estándar. He et al. descubrieron que usar BN ingenuamente *impide aprender buenas representaciones*: el modelo "hace trampa" en la pretext task y encuentra una solución de baja pérdida fácilmente, porque la comunicación intra-batch que introduce BN **filtra información** (las estadísticas del sub-batch sirven de "firma" para identificar en qué sub-batch está la clave positiva). La solución es **Shuffling BN**: al entrenar con múltiples GPUs, antes de distribuir el mini-batch entre GPUs para el key encoder `f_k` se *baraja el orden de las muestras* (y se des-baraja tras codificar); el orden para `f_q` no se altera. Esto garantiza que las estadísticas de batch usadas para un query y su clave positiva provengan de subconjuntos distintos, eliminando la fuga. La ablación del apéndice muestra que sin Shuffling BN hay sobreajuste evidente a la pretext task (la accuracy de entrenamiento sube a >99.9% mientras la validación kNN cae).

## 5. Experimentos

### 5.1. Clasificación lineal en ImageNet

El protocolo estándar: preentrenar sin supervisión en ImageNet-1M (IN-1M, ~1.28M imágenes), congelar las features y entrenar un clasificador lineal supervisado (una capa fully-connected + softmax) sobre las features de global average pooling. Hallazgos:

- **Los tres mecanismos se benefician de un `K` mayor** —lo que sustenta la motivación del diccionario grande—. Pero MoCo escala mejor: el end-to-end topa en `K ≈ 1024` por memoria, y el memory bank, aun pudiendo crecer, queda **2.6% por debajo de MoCo** (58.0% vs 60.6%), confirmando la hipótesis de inconsistencia.
- **MoCo con ResNet-50 (R50) alcanza 60.6% top-1**, mejor que todos los competidores de tamaño de modelo similar (~24M parámetros), y *sin* diseños arquitectónicos especiales (sin parches, sin campos receptivos custom, sin combinar dos redes). Con modelos más anchos: R50w2× → 65.4%, R50w4× → **68.6%**, RX50 (ResNeXt) → 63.9%.

### 5.2. Transferencia a detección y segmentación — el hito

El objetivo real del aprendizaje no supervisado es preentrenar features transferibles. Para una comparación justa, MoCo se fine-tunea con los **mismos hiperparámetros que el contrapunto supervisado de ImageNet** (lo que en principio desfavorece a MoCo), usando dos prerequisitos que el paper discute: *normalización* (fine-tunear con BN sincronizada en vez de congelarla, para calibrar las distribuciones de features que difieren del supervisado) y *schedules* controlados (1×/2× en vez de los 6×–9× de "Rethinking ImageNet pre-training").

- **PASCAL VOC (Faster R-CNN).** Con el backbone R50-C4, MoCo IN-1M supera al supervisado (+0.9 AP50, +3.7 AP, +4.9 AP75) y MoCo IG-1B lo amplía. En la comparación contra métodos previos (Tabla 4), **ningún método previo alcanzaba a su contrapunto supervisado en AP50; MoCo es el primero que lo supera**, con ganancias de hasta +5.2 AP y +9.0 AP75 en las métricas más estrictas.
- **COCO (Mask R-CNN, detección + segmentación de instancias).** Con el schedule 2×, MoCo supera al supervisado en *todas* las métricas en ambos backbones (FPN y C4).
- **Más tareas downstream:** keypoint detection en COCO (MoCo supera; el supervisado no aventaja ni siquiera a la inicialización aleatoria), dense pose en COCO (MoCo supera sustancialmente, +3.7 AP75), LVIS instance segmentation, y semantic/instance segmentation en Cityscapes.

**Resumen del hito:** MoCo **supera a su contrapunto supervisado de ImageNet en 7 tareas** de detección/segmentación (detección VOC/COCO, segmentación de instancias COCO/LVIS, keypoints COCO, dense pose COCO, segmentación semántica Cityscapes), está a la par en segmentación de instancias Cityscapes, y solo queda por debajo en segmentación semántica VOC (un caso negativo reconocido). El paper concluye que "ha cerrado en gran medida la brecha entre el aprendizaje de representaciones no supervisado y supervisado en muchas tareas de visión".

### 5.3. Escala billion-image: IG-1B

MoCo se preentrena también en Instagram-1B (IG-1B, ~940M imágenes públicas), un dataset relativamente *no curado* y de distribución de cola larga (mundo real), frente al bien balanceado IN-1M. MoCo IG-1B es consistentemente mejor que MoCo IN-1M en casi todas las tareas, demostrando que el mecanismo funciona en escenarios de mil millones de imágenes —algo intratable para un memory bank, lo que subraya la **eficiencia de memoria** de la cola—. La mejora de IN-1M a IG-1B es notable pero relativamente pequeña, lo que sugiere que la pretext task simple no explota plenamente los datos a gran escala.

### 5.4. MoCo v1 vs MoCo v2

El paper menciona "MoCo v2" (Chen et al. 2020), una extensión de una versión preliminar de este manuscrito que, con cambios pequeños tomados de SimCLR —una **cabeza de proyección MLP** y **augmentación de datos más fuerte** (blur)— sube de 60.6% a **71.1%** con R50, mostrando la generalidad y robustez del framework MoCo.

| Variante | top-1 ImageNet (R50, protocolo lineal) | cambios clave |
|---|---|---|
| **MoCo v1** | 60.2 / 60.6 | cola FIFO + momentum encoder, head lineal |
| **MoCo v2** | **71.1** | + proyección MLP + augmentación fuerte (blur) |

(El 60.2 es el valor reportado en la presentación de la Clase 28; el 60.6 es el de la Tabla 1 del paper para R50. La diferencia corresponde a variantes de reporte del mismo modelo base.)

## 6. Limitaciones reconocidas

El propio paper es modesto y señala varias:

- **La pretext task simple no aprovecha los datos a gran escala.** La mejora de IN-1M a IG-1B es "consistentemente notable pero relativamente pequeña", lo que sugiere que los datos a mayor escala "no se explotan plenamente". El paper conjetura que una pretext task más avanzada (p.ej. masked auto-encoding, anticipando MAE) podría mejorar esto.
- **Casos negativos de transferencia.** En segmentación semántica de VOC MoCo queda por debajo del supervisado (al menos −0.8 punto), un caso negativo explícito. La accuracy de transferencia depende de la estructura del detector (la ventaja es mayor con backbone C4 que con FPN), una interacción que "ha estado velada en el pasado".
- **Dependencia de Shuffling BN.** El buen funcionamiento depende de un truco de ingeniería de BN multi-GPU; sin él el modelo sobreajusta a la pretext task. Esto ata el método a un setup de entrenamiento distribuido.
- **MoCo es un mecanismo, no una pretext task.** Es deliberadamente agnóstico a la pretext task; el paper no explora factores ortogonales (como pretexts mejores) que podrían subir más la accuracy —lo que efectivamente hizo MoCo v2 después.

## 7. Impacto

MoCo, junto con SimCLR (Chen et al. 2020), marca el momento en que el **aprendizaje contrastivo se vuelve el paradigma dominante del autosupervisado en visión** y en que el preentrenamiento no supervisado se convierte en una alternativa viable —no solo competitiva sino a veces superior— al preentrenamiento supervisado de ImageNet para tareas downstream. Tres ideas de MoCo quedaron como infraestructura del campo: (1) ver el aprendizaje contrastivo como *dictionary look-up*; (2) desacoplar el número de negativos del *batch size* mediante una cola, lo que lo hace **eficiente en memoria** (a diferencia de SimCLR, que necesita batches gigantes para tener muchos negativos); y (3) el **momentum encoder / EMA de pesos**, una técnica que reapareció en BYOL, DINO y muchos métodos de auto-distilación posteriores. El linaje MoCo v1 → v2 → v3 trazó además la transición de ResNet a Vision Transformers en el autosupervisado contrastivo.

## 8. Conexión con la Clase 28 (Aprendizaje Autosupervisado)

La Clase 28 dedica un slide a MoCo (He et al. 2019) que dibuja exactamente los tres componentes diseccionados arriba: el **encoder** (que produce el query `q` y por el que fluye el gradiente), el **momentum encoder** (que produce las claves, actualizado por EMA `θ_k ← m·θ_k + (1−m)·θ_q`, sin gradiente) y la **cola FIFO de `K` negativos** (los embeddings de batches previos). El punto pedagógico que la clase subraya —y que este análisis fundamenta— es **por qué MoCo requiere menos memoria que SimCLR**: SimCLR obtiene sus negativos del propio mini-batch, de modo que para tener muchos negativos necesita batches enormes (miles de imágenes simultáneas en memoria, con gradientes); MoCo, en cambio, mantiene los negativos en una cola de embeddings *ya computados y sin gradiente*, así que puede tener `K = 65536` negativos con un batch ordinario de 256. La cola es memoria barata (vectores de 128-D); el batch grande de SimCLR es memoria cara (activaciones + gradientes de toda la red).

Esto sitúa a MoCo dentro del recorrido del autosupervisado en visión que la clase construye: de las *pretext tasks* manuales —colorización ([Zhang et al. 2016](Zhang-Colorization-2016.md)), context prediction ([Doersch et al. 2015](Doersch-ContextPrediction-2015.md)), jigsaw, rotaciones ([Gidaris et al. 2018](Gidaris-RotNet-2018.md))— hacia el **aprendizaje contrastivo** basado en *instance discrimination* ([Ye et al. 2019](Ye-InvariantSpreading-2019.md)) y su consolidación en MoCo y SimCLR. La comparación canónica de la clase es MoCo (cola + momentum, eficiente en memoria) frente a SimCLR (batch grande + cabeza MLP + augmentación fuerte), y la convergencia de ambos en MoCo v2.

- Fundamentos transversales: [Aprendizaje contrastivo](/fundamentos/aprendizaje-contrastivo), [Aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado).
- Clase: [Clase 28 — Aprendizaje Autosupervisado](/clases/clase-28).
- Paper hermano: [SimCLR (Chen et al. 2020)](/papers/simclr-chen-2020) — el otro pilar contrastivo de 2020, con el que MoCo se compara directamente y del que MoCo v2 toma la cabeza MLP y la augmentación fuerte.
