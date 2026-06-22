---
title: "MoCo: Momentum Contrast (2019)"
weight: 321
math: true
---

{{< paper-card
    title="Momentum Contrast for Unsupervised Visual Representation Learning"
    authors="Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick"
    year="2020"
    venue="CVPR 2020"
    pdf="/papers/moco-he-2019.pdf"
    arxiv="1911.05722" >}}
Trabajo de Facebook AI Research (FAIR) que **cierra la brecha entre el preentrenamiento no supervisado y el supervisado** en visión. MoCo (Momentum Contrast) mira todo el [aprendizaje contrastivo](/fundamentos/aprendizaje-contrastivo) bajo una sola lente —una **búsqueda en diccionario**— y propone dos ideas que encajan: una **cola FIFO** de negativos que desacopla el tamaño del diccionario del *batch size*, y un **momentum encoder** actualizado por media móvil exponencial que mantiene la consistencia de las claves. El resultado es histórico: las representaciones no supervisadas **igualan o superan al preentrenamiento supervisado de ImageNet** al transferir a detección y segmentación en 7 tareas, y con mucha menos memoria que SimCLR.
{{< /paper-card >}}

---

## Contexto: del éxito en NLP a los diccionarios visuales

El paper abre con un contraste que motiva todo. El aprendizaje no supervisado de representaciones es muy exitoso en NLP —GPT, BERT—, pero hacia 2019 el preentrenamiento **supervisado** seguía dominando en visión. He et al. ofrecen una explicación elegante: la diferencia está en los **espacios de señal**. El lenguaje es *discreto* (palabras, sub-word units) y permite construir diccionarios tokenizados naturalmente. La visión opera sobre una señal cruda *continua y de alta dimensión* que no está estructurada para la comunicación humana, de modo que "construir el diccionario" es parte del problema.

Bajo esa lente, varios métodos contrastivos previos (CPC, CMC, instance discrimination) se unifican conceptualmente: todos **construyen diccionarios dinámicos**. Las "claves" del diccionario se muestrean de los datos y se representan con un encoder; el aprendizaje no supervisado entrena ese encoder para realizar la búsqueda. MoCo declara explícitamente que su foco es la **función de pérdida y el mecanismo del diccionario**, no una pretext task nueva: usa la tarea simple de *instance discrimination* (dos vistas aumentadas de la misma imagen son un par positivo) justamente para aislar la contribución del mecanismo.

## El problema central: tamaño vs. consistencia

La hipótesis del paper es que se pueden aprender buenas features con un **diccionario grande** que cubra muchos negativos, siempre que el encoder de las claves se mantenga **lo más consistente posible** pese a que evoluciona durante el entrenamiento. Un diccionario ideal debería ser:

- **Grande** — para muestrear densamente el espacio visual continuo y de alta dimensión.
- **Consistente** — las claves deberían codificarse con un encoder igual o muy similar, para que las comparaciones contra el query tengan sentido.

El paper compara tres mecanismos, todos con la misma pretext task e InfoNCE, de modo que la diferencia es puramente el mecanismo:

- **(a) End-to-end.** Query y clave se computan con encoders actualizados por retropropagación. Es perfectamente **consistente** (las claves del batch usan los mismos pesos), pero el **tamaño del diccionario queda acoplado al mini-batch**, limitado por la memoria de GPU (a lo sumo ~1024 en la máquina del paper, 8 GPUs Volta de 32GB).
- **(b) Memory bank.** Almacena las representaciones de *todas* las muestras del dataset y muestrea el diccionario sin retropropagación, lo que **sí permite un diccionario grande**. El precio es la **inconsistencia**: cada muestra del banco se actualizó la última vez que se vio, así que las claves vienen de encoders de pasos muy distintos a lo largo de la época pasada.

El diagnóstico es nítido: end-to-end = consistente pero pequeño; memory bank = grande pero inconsistente. **Ninguno logra las dos propiedades a la vez.** MoCo se diseña como respuesta directa a esa tensión.

## La cola FIFO: desacoplar tamaño y batch

El núcleo de MoCo es mantener el diccionario como una **cola FIFO** de muestras de datos. Esto permite **reutilizar las claves codificadas de los mini-batches inmediatamente precedentes** y **desacopla el tamaño del diccionario del tamaño del mini-batch**: la cola `K` puede ser mucho mayor que un batch típico y se fija como hiperparámetro (en los experimentos principales, `K = 65536`).

Las muestras se reemplazan progresivamente: el mini-batch actual se *encola* (enqueue) y el más antiguo se *desencola* (dequeue). Solo se almacenan los embeddings —no se recodifican—, así que el coste extra es manejable. He et al. notan un beneficio sutil: *remover el batch más antiguo es beneficioso*, porque sus claves son las más desactualizadas y, por tanto, las menos consistentes con las nuevas. La cola es exactamente lo que hace grande al diccionario sin pagar memoria de gradientes.

## El momentum encoder: recuperar la consistencia

La cola resuelve el tamaño, pero crea un problema nuevo: vuelve **intratable actualizar el key encoder por retropropagación**, porque el gradiente debería propagarse a todas las muestras de la cola (de batches pasados). Una solución ingenua —copiar el key encoder desde el query encoder ignorando ese gradiente— da malos resultados. La hipótesis es que un encoder de claves que **cambia rápidamente** reduce la consistencia de las claves ya guardadas en la cola.

La solución es el **momentum update**. Denotando `θ_k` los parámetros del key encoder `f_k` y `θ_q` los del query encoder `f_q`:

$$\theta_k \leftarrow m \cdot \theta_k + (1 - m) \cdot \theta_q$$

donde `m ∈ [0,1)` es el coeficiente de momentum. **Solo `θ_q` se actualiza por retropropagación**; `θ_k` se arrastra como media móvil exponencial (EMA) de `θ_q`. La consecuencia clave: `θ_k` evoluciona **mucho más suavemente** que `θ_q`. Aunque las claves de la cola fueron codificadas por encoders de distintos mini-batches, la diferencia entre esos encoders se mantiene pequeña porque todos son versiones suavizadas y cercanas. Eso restaura la consistencia que la cola amenazaba.

El valor del momentum importa muchísimo: un **`m` grande (0.999, el default) funciona mucho mejor que uno pequeño (0.9)**. La ablación con `K = 4096` lo cuantifica:

| momentum `m` | 0 | 0.9 | 0.99 | 0.999 | 0.9999 |
|---|---|---|---|---|---|
| accuracy (%) | falla | 55.2 | 57.8 | **59.0** | 58.9 |

Con `m = 0` (copiar directo el encoder) el training loss oscila y no converge. El régimen útil es `m ∈ [0.99, 0.9999]`. Esto confirma la motivación: **un encoder de claves que progresa lentamente es el núcleo que permite explotar la cola** —un resultado contraintuitivo, porque uno esperaría que el key encoder debiera seguir de cerca al de queries.

## InfoNCE: la pérdida que une todo

La pérdida es **InfoNCE** (van den Oord et al. 2018), una pérdida contrastiva donde la similitud se mide por producto punto:

$$\mathcal{L}_q = -\log \frac{\exp(q \cdot k_+ / \tau)}{\sum_{i=0}^{K} \exp(q \cdot k_i / \tau)}$$

La suma corre sobre una clave positiva y `K` negativas (las de la cola). Intuitivamente es la log-loss de un clasificador softmax de `(K+1)` clases que intenta clasificar `q` como `k+`, con temperatura `τ = 0.07`. El mecanismo, en pseudocódigo estilo PyTorch:

```python
f_k.params = f_q.params                          # inicializar
for x in loader:                                 # mini-batch de N muestras
    x_q, x_k = aug(x), aug(x)                    # dos vistas aumentadas
    q = f_q.forward(x_q)                         # queries: NxC
    k = f_k.forward(x_k).detach()               # keys: NxC, SIN gradiente
    l_pos = bmm(q.view(N,1,C), k.view(N,C,1))    # logits positivos: Nx1
    l_neg = mm(q.view(N,C), queue.view(C,K))     # logits negativos: NxK
    logits = cat([l_pos, l_neg], dim=1)          # Nx(1+K)
    labels = zeros(N)                            # el positivo es el 0-esimo
    loss = CrossEntropyLoss(logits / t, labels)
    loss.backward(); update(f_q.params)          # SGD solo en query encoder
    f_k.params = m*f_k.params + (1-m)*f_q.params # momentum update
    enqueue(queue, k); dequeue(queue)            # actualizar diccionario
```

El `.detach()` sobre las claves materializa el punto central: **el gradiente fluye solo por el query encoder**; el key encoder nunca se entrena por backprop, solo por la EMA.

Un detalle de ingeniería esencial es **Shuffling BN**. La Batch Normalization ingenua *impide aprender buenas representaciones*: el modelo "hace trampa" porque las estadísticas del sub-batch sirven de "firma" para identificar dónde está la clave positiva. La solución es barajar el orden de las muestras entre GPUs antes de codificarlas con `f_k` (y des-barajar después), eliminando esa fuga de información.

## El hito: transferencia a detección y segmentación

Bajo el protocolo lineal de ImageNet, MoCo con ResNet-50 alcanza **60.6% top-1**, mejor que todos los competidores de tamaño de modelo similar y *sin* diseños arquitectónicos especiales. Los tres mecanismos mejoran con `K` mayor —lo que sustenta la motivación del diccionario grande—, pero el end-to-end topa por memoria y el memory bank queda **2.6% por debajo de MoCo** (58.0% vs 60.6%), confirmando la hipótesis de inconsistencia.

Pero el resultado verdaderamente histórico es la **transferencia**. Fine-tuneando con los **mismos hiperparámetros que el contrapunto supervisado de ImageNet** (lo que en principio desfavorece a MoCo):

- **PASCAL VOC (Faster R-CNN):** MoCo IN-1M supera al supervisado (+0.9 AP50, +3.7 AP, +4.9 AP75). **Ningún método previo alcanzaba a su contrapunto supervisado en AP50; MoCo es el primero que lo supera.**
- **COCO (Mask R-CNN):** con schedule 2×, MoCo supera al supervisado en *todas* las métricas en ambos backbones (FPN y C4).
- **Más tareas:** keypoint detection y dense pose en COCO, LVIS instance segmentation, segmentación semántica/de instancias en Cityscapes.

En total, MoCo **supera a su contrapunto supervisado de ImageNet en 7 tareas** de detección/segmentación, está a la par en una y solo queda por debajo en segmentación semántica de VOC. El paper concluye que "ha cerrado en gran medida la brecha entre el aprendizaje de representaciones no supervisado y supervisado en muchas tareas de visión". MoCo también escala a **Instagram-1B** (~940M imágenes, no curadas), algo intratable para un memory bank, lo que subraya su **eficiencia de memoria**.

## MoCo v1 vs MoCo v2

[MoCo v2](/papers/moco-v2-chen-2020) (Chen et al. 2020) extiende el framework con dos cambios pequeños tomados de [SimCLR](/papers/simclr-chen-2020) —una **cabeza de proyección MLP** y **augmentación de datos más fuerte** (blur)— y sube de 60.6% a **71.1%**, mostrando la generalidad del diseño:

| Variante | top-1 ImageNet (R50, protocolo lineal) | cambios clave |
|---|---|---|
| **MoCo v1** | 60.2 / 60.6 | cola FIFO + momentum encoder, head lineal |
| **MoCo v2** | **71.1** | + proyección MLP + augmentación fuerte (blur) |

## Limitaciones reconocidas

- **La pretext task simple no aprovecha los datos a gran escala.** La mejora de IN-1M a IG-1B es notable pero pequeña; el paper conjetura que una pretext task más avanzada (anticipando MAE) lo mejoraría.
- **Casos negativos de transferencia.** En segmentación semántica de VOC MoCo queda por debajo del supervisado, y la ventaja depende de la estructura del detector (mayor con C4 que con FPN).
- **Dependencia de Shuffling BN.** El buen funcionamiento ata el método a un setup de entrenamiento distribuido multi-GPU.
- **MoCo es un mecanismo, no una pretext task.** Es deliberadamente agnóstico a la pretext task, dejando margen que MoCo v2 explotó después.

## Por qué importa para la Clase 28

La [Clase 28 — Aprendizaje Autosupervisado](/clases/clase-28) dedica un slide a MoCo que dibuja exactamente sus tres componentes: el **encoder** (produce el query `q`, por él fluye el gradiente), el **momentum encoder** (produce las claves, actualizado por EMA `θ_k ← m·θ_k + (1−m)·θ_q`, sin gradiente) y la **cola FIFO de `K` negativos** (embeddings de batches previos). El punto pedagógico clave es **por qué MoCo requiere menos memoria que SimCLR**: SimCLR obtiene sus negativos del propio mini-batch, así que para tener muchos necesita batches enormes (miles de imágenes con gradientes en memoria); MoCo mantiene los negativos en una cola de embeddings *ya computados y sin gradiente*, logrando `K = 65536` negativos con un batch ordinario de 256. La cola es memoria barata (vectores de 128-D); el batch grande de SimCLR es memoria cara (activaciones + gradientes de toda la red).

MoCo se sitúa en el recorrido del [aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado) en visión que la clase construye: de las *pretext tasks* manuales (colorización, context prediction, jigsaw, rotaciones) hacia el [aprendizaje contrastivo](/fundamentos/aprendizaje-contrastivo) basado en *instance discrimination* y su consolidación en MoCo y SimCLR. Tres ideas de MoCo quedaron como infraestructura del campo: ver el contrastive como *dictionary look-up*, desacoplar los negativos del *batch size* mediante una cola, y el **momentum encoder / EMA de pesos** —técnica que reapareció en BYOL, DINO y muchos métodos de auto-distilación posteriores.

## Enlaces

- Clase: [Clase 28 — Aprendizaje Autosupervisado](/clases/clase-28)
- Fundamentos: [Aprendizaje contrastivo](/fundamentos/aprendizaje-contrastivo) · [Aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado)
- Paper hermano: [SimCLR (Chen et al. 2020)](/papers/simclr-chen-2020) — el otro pilar contrastivo de 2020
- Extensión: [MoCo v2 (Chen et al. 2020)](/papers/moco-v2-chen-2020) — toma de SimCLR la cabeza MLP y la augmentación fuerte
- arXiv: [1911.05722](https://arxiv.org/abs/1911.05722) · Código: [github.com/facebookresearch/moco](https://github.com/facebookresearch/moco)
