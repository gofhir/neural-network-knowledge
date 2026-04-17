---
title: "Transferable Features (Yosinski)"
weight: 180
math: true
---

{{< paper-card
    title="How transferable are features in deep neural networks?"
    authors="Yosinski, Clune, Bengio, Lipson"
    year="2014"
    venue="NeurIPS 2014"
    pdf="/papers/transferable-features-yosinski-2014.pdf"
    arxiv="1411.1792" >}}
Cuantifica **layer-by-layer** cuan transferibles son las representaciones aprendidas por redes convolucionales. Distingue dos mecanismos distintos que degradan la transferibilidad: **especificidad de features** y **fragile co-adaptation**. Muestra que incluso tareas muy diferentes pueden beneficiarse de transferir pesos, y que **transferir + fine-tune mejora la generalizacion incluso en datasets grandes** -- un resultado sorprendente que influyo toda la practica moderna de transfer learning.
{{< /paper-card >}}

---

## Contexto

En 2014 era ya conocido empiricamente que los filtros de la primera capa de cualquier CNN entrenada en imagenes naturales **se parecen** a filtros Gabor y blobs de color -- independiente del dataset o incluso del objetivo de entrenamiento (supervisado, no supervisado, density learning). La ultima capa, en cambio, claramente depende de la tarea. **Donde ocurre la transicion entre features generales y especificos?** Este paper responde cuantitativamente esa pregunta.

---

## Ideas principales

### 1. Setup experimental

Los autores disenaron experimentos controlados en ImageNet:

1. Divide las 1000 clases de ImageNet en dos subconjuntos de 500 clases (A y B), ~645K ejemplos cada uno.
2. Entrena dos redes de 8 capas: `baseA` en A, `baseB` en B.
3. Para cada $n \in \{1, 2, \ldots, 7\}$, construye dos variantes:
   - **Selffer BnB**: primeras $n$ capas copiadas y **frozen** de `baseB`; entrenar capas $n+1$ a $8$ en B.
   - **Transfer AnB**: primeras $n$ capas copiadas y **frozen** de `baseA`; entrenar capas $n+1$ a $8$ en B.
4. Crear versiones **fine-tuned** (no frozen): **BnB+** y **AnB+**.

Adicionalmente, construyeron un split **disimilar**: A = clases hechas por humanos (551), B = clases naturales (449), para estudiar el efecto de distancia entre tareas.

### 2. Cinco hallazgos clave

#### Hallazgo 1: Degradacion al cortar capas medias (selffer BnB)

Sorprendentemente, incluso **copiar capas de la misma red** y congelarlas degrada el rendimiento cuando se corta en capas 3-5. Esto revela un fenomeno que llamaron **"fragile co-adaptation"**: las features de capas adyacentes se coordinan de maneras complejas durante el training conjunto, y congelar parte rompe esa coordinacion que no puede recuperarse entrenando solo las capas superiores.

#### Hallazgo 2: Fine-tuning recupera la co-adaptation

Los modelos BnB+ (fine-tuned) no sufren la caida. Permitir que las capas frozen se ajusten recupera la colaboracion perdida. **Lesson**: si vas a transferir a una tarea similar, mejor fine-tune que feature extraction.

#### Hallazgo 3: Generalidad decae con la profundidad

Para transferencia **AnB** (entre tareas diferentes):

- Capas 1-2: transfieren casi perfectamente, sin perdida.
- Capa 3: leve caida (~1-2%).
- Capas 4-5: caida significativa (~3-5%).
- Capas 6-7: caida importante (~5-10%).

La transicion no es abrupta; es **gradual a lo largo de la red**.

#### Hallazgo 4: Transfer + fine-tune mejora generalizacion incluso en datasets grandes

Este es **el resultado mas sorprendente**. Incluso cuando el target dataset es tan grande como el source, usar pesos preentrenados (y fine-tunear) da una mejora de **~1.6-2.1% absoluto en accuracy** sobre entrenar desde cero, **despues de 450k iteraciones de fine-tuning**.

El efecto **persiste**: no es solo una "cabeza de arranque" que se pierde. Los autores hipotetizan que el conocimiento del dataset source deja una huella indeleble en el paisaje de optimizacion, guiando al modelo a una mejor region del espacio de parametros.

#### Hallazgo 5: Similitud de tareas importa, pero transferir siempre ayuda

Con el split disimilar (man-made vs natural), la transferibilidad cae mas rapido con la profundidad. Pero incluso asi, **transferir siempre supera a pesos aleatorios**. Es decir, aunque la tarea sea muy distinta, los features aprendidos son mejores puntos de partida que la inicializacion estandar.

### 3. Dos mecanismos distintos

El paper **separa rigurosamente** dos efectos que antes se confundian:

| Efecto | Causa | Dominio |
|---|---|---|
| **Fragile co-adaptation** | Features adyacentes aprenden a colaborar; romper la colaboracion requiere mucho reentrenamiento | Capas 3-5 (medio de la red) |
| **Representation specificity** | Features se especializan a la tarea source | Capas 6-7 (altas) |

Este marco distingue por que **fine-tuning ayuda diferente** en cada capa:

- En capas medias, fine-tune restaura la co-adaptacion.
- En capas altas, fine-tune especializa las features a la nueva tarea.

---

## Implicaciones practicas

{{< concept-alert type="clave" >}}
**Reglas derivadas del paper para transfer learning**:

1. **Siempre transferir cuando existan pesos preentrenados**, incluso si el target es grande. Free lunch.
2. **Fine-tune es mejor que feature extraction** cuando hay datos suficientes, por co-adaptation.
3. **No cortar en el medio** (capas 3-5) si se va a usar feature extraction, por fragile co-adaptation.
4. **Si source y target son muy diferentes**, transferir solo las primeras capas (1-3).
5. **Si el target es pequeno**, fine-tune toda la red con lr muy bajo y discriminative learning rates.
{{< /concept-alert >}}

---

## Por que importa hoy

- Es el **analisis de referencia** para cualquier discusion sobre transferibilidad en vision.
- Influyo directamente en la adopcion masiva de ImageNet pretraining como default en vision aplicada.
- El concepto de **fragile co-adaptation** explica por que **layer-wise training** y **progressive unfreezing** son efectivos (ULMFiT, Howard & Ruder 2018).
- Las ideas generalizan a NLP (BERT, GPT): la penultima capa tambien contiene features mas tarea-especificas que la primera.
- Precursor conceptual de **foundation models**: si transferir + fine-tune siempre ayuda, escalar masivamente el pretraining es la direccion natural.

---

## Limitaciones

- Experimentos solo en ImageNet; la generalizacion a dominios muy diferentes (medical, satelite, text) requiere investigacion adicional.
- La red usada (AlexNet-like, 8 capas) es pequena para estandares modernos; ResNets y Transformers pueden tener patrones distintos.
- No cubre **scale mismatch**: transferir de red pequena a red grande o viceversa.
- Los experimentos son costosos (~9.5 dias por run en GPU), limitando el numero de replicas.

---

## Notas y enlaces

- La **Figura 2** es una de las mas citadas en transfer learning: muestra las curvas de accuracy vs layer $n$ para las cuatro condiciones.
- Codigo y parametros: [yosinski.com/transfer](http://yosinski.com/transfer)
- Follow-ups destacables:
  - **Oquab et al. 2014** "Learning and Transferring Mid-Level Image Representations using Convolutional Neural Networks" (CVPR).
  - **Razavian et al. 2014** "CNN Features off-the-shelf: an Astounding Baseline for Recognition".
  - **Huh et al. 2016** "What makes ImageNet good for transfer learning?".

Ver fundamentos: [Transfer Learning](/fundamentos/transfer-learning) · [Foundation Models](/fundamentos/foundation-models).
