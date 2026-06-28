---
title: "Experimentos con MAML: hiperparámetros y dificultad"
weight: 3
math: true
---

> **Actividades 1, 2 y 3 del laboratorio.** Tres experimentos encadenados con MAML que responden tres preguntas distintas: ¿qué palanca optimiza un MAML que va lento (Omniglot)?, ¿qué hace *intrínsecamente* difícil un problema few-shot (WAYS vs. SHOTS)?, y ¿qué pasa cuando el cuello de botella deja de ser la optimización y pasa a ser la información (Mini-ImageNet)? La conclusión transversal: **la naturaleza del dato decide qué hiperparámetro importa.**

MAML (Model-Agnostic Meta-Learning) entrena un conjunto de pesos iniciales $\theta$ tales que, partiendo de ahí, **pocos pasos de descenso de gradiente** sobre una tarea nueva (el *inner loop*) bastan para resolverla. El *outer loop* ajusta $\theta$ midiendo qué tan bien quedó el modelo *después* de adaptarse. Los hiperparámetros clave son por eso de dos niveles:

- **`fast_lr`** (inner) — el learning rate de la adaptación a cada tarea.
- **`adaptation_steps`** (inner) — cuántos pasos de gradiente se dan al adaptarse.
- **`meta_lr`** (outer) — el learning rate con que se actualiza la inicialización $\theta$.
- **`meta_batch_size`** (outer) — cuántas tareas se promedian por cada paso del meta-gradiente.
- **`num_iterations`** (outer) — cuántos pasos de meta-entrenamiento en total.

Detalle del algoritmo en [/fundamentos/optimizacion-binivel](/fundamentos/optimizacion-binivel) y en el paper original [MAML (Finn 2017)](/papers/maml-finn-2017).

## Actividad 1 — Optimizar MAML en Omniglot (4-way 1-shot)

El punto de partida es un MAML que apenas supera el azar útil: con un clasificador *fully-connected*, 50 iteraciones de meta-entrenamiento y un solo paso de adaptación, llega a **0.699** de Meta Test Accuracy en 4-way 1-shot (azar = 0.25). La actividad es un *ablation*: mover una palanca a la vez y medir su efecto real.

| Config | Train | Valid | Test |
|---|---|---|---|
| FC · 50 iters · steps=1 · b=32 *(baseline)* | 0.602 | 0.727 | 0.699 |
| CNN · 50 · steps=1 · b=32 | 0.625 | 0.719 | 0.706 |
| CNN · 150 · steps=1 · b=32 | 0.844 | 0.742 | 0.749 |
| CNN · 150 · steps=5 · b=32 | 0.875 | 0.773 | 0.760 |
| **CNN · 400 · steps=5 · b=64 · meta_lr=0.005 · fast_lr=0.3** | **0.859** | **0.852** | **0.877** |

**¿Por qué cambiar FC → CNN casi no movió la aguja (+0.007)?** Intuitivamente una CNN debería arrasar en imágenes. Pero a 50 iteraciones la CNN ni siquiera terminó de despegar: tiene **más parámetros** que la FC y, por tanto, necesita *más pasos de meta-entrenamiento* para que su inicialización $\theta$ se vuelva útil. A presupuesto fijo de iteraciones, la mayor capacidad de la CNN es un cheque que todavía no se puede cobrar. La arquitectura correcta sin entrenamiento suficiente no rinde.

**La palanca dominante fue `num_iterations` (50 → 150 dio +0.043).** Esto confirma el diagnóstico anterior: el modelo estaba *subentrenado*, no mal diseñado. Al darle 3× más pasos de outer loop, la inicialización $\theta$ llega a un punto desde el cual la adaptación a tareas nuevas funciona. La mejora no vino de un cambio cualitativo, sino de **volumen de meta-entrenamiento**.

**`adaptation_steps` 1 → 5 aportó poco (+0.011) a 5× el costo.** Más pasos de inner loop refinan la adaptación a cada tarea, pero en Omniglot (caracteres simples, alta separabilidad) un buen punto inicial ya queda cerca de la solución: un paso casi alcanza. El costo, en cambio, es lineal — cinco pasos de gradiente por tarea, más el backprop a través de ellos. Es una palanca de **rendimientos decrecientes** en este dataset.

**El overfitting y cómo se eliminó.** A `batch=32` apareció una brecha train − valid de \~0.10 (config CNN·150·steps=5: train 0.875 vs. valid 0.773): el meta-gradiente, estimado sobre pocas tareas por paso, era ruidoso y el modelo se ajustaba a las idiosincrasias de los lotes. La solución no fue regularizar el modelo, sino **estabilizar la optimización**: subir `meta_batch_size` a 64 (promediar el meta-gradiente sobre más tareas reduce su varianza) y bajar `fast_lr` a 0.3 (adaptaciones internas menos agresivas, que no sobre-ajustan al *support set* de cada tarea). El resultado fue **0.877** con la brecha cerrada — train 0.859 $\approx$ valid 0.852.

> **Gotcha — el overfitting en meta-learning no se ataca como en supervisado clásico.** Acá la cura fue *estabilizar el estimador del meta-gradiente* (más tareas por paso, inner-LR más suave), no añadir dropout o weight decay. El "ruido" venía de estimar el outer gradient con pocas tareas, no de un modelo demasiado expresivo.

![MAML Omniglot mejor config](/laboratorios/lab-26/maml-omniglot-best.jpg)

**Balance.** De **0.699 a 0.877** (+25% relativo). La ganancia salió del volumen de meta-entrenamiento, y el overfitting resultó controlable estabilizando la optimización. El techo canónico de MAML en Omniglot 4-way 1-shot (\~0.93–0.98) exige del orden de **60.000 iteraciones**, fuera del alcance de una sesión de Colab — pero la tendencia ya quedó probada: la curva sube con iteraciones, no se estanca.

## Actividad 2 — Variar WAYS y SHOTS en Omniglot

La segunda actividad cambia de pregunta: ya no *cómo* entrenar mejor, sino *qué hace difícil* a un problema few-shot. Se mueven dos ejes de la tarea misma:

- **WAYS** — cuántas clases hay que distinguir en cada tarea (N-way).
- **SHOTS** — cuántos ejemplos etiquetados se dan por clase para adaptarse (K-shot).

Comparar accuracies crudos a distinto número de WAYS es injusto, porque el azar cambia: 2-way tiene 50% de azar y 8-way solo 12.5%. Por eso usamos la **accuracy normalizada**, que descuenta el azar y mide cuánto del margen disponible sobre el azar realmente captura el modelo:

$$\text{Test normalizado} = \frac{\text{acc} - \text{azar}}{1 - \text{azar}}$$

**Eje WAYS (fijando SHOTS = 1):**

| WAYS | Test acc | Azar | Normalizado |
|---|---|---|---|
| 2-way | 0.932 | 0.500 | 0.864 |
| 4-way | 0.797 | 0.250 | 0.729 |
| 8-way | 0.657 | 0.125 | 0.608 |

**Eje SHOTS (fijando WAYS = 4):**

| SHOTS | Test acc |
|---|---|
| 1-shot | 0.797 |
| 5-shot | 0.943 |

**Más WAYS = más difícil, y no es solo el azar.** El accuracy cae monotónicamente al subir WAYS (0.932 → 0.797 → 0.657), lo cual era esperable: más clases, más formas de equivocarse. Lo revelador es que la caída **persiste tras normalizar** (0.864 → 0.729 → 0.608). Si la única dificultad fuera el azar, el accuracy normalizado se mantendría constante. Que baje significa que hay una **dificultad intrínseca** real: con más clases, las fronteras de decisión son más finas y la inicialización $\theta$ debe servir para discriminaciones más sutiles a partir de la misma cantidad de evidencia.

**Más SHOTS = más fácil (+0.146).** Pasar de 1 a 5 ejemplos por clase sube el accuracy de 0.797 a 0.943. Cada *shot* adicional es evidencia con la que el inner loop adapta el modelo: el *support set* deja de ser un punto aislado por clase y pasa a definir una región, lo que hace la adaptación mucho más robusta.

**Los dos ejes son fuerzas opuestas de magnitud comparable, y se compensan.** Observa el cruce:

$$\underbrace{\text{2-way, 1-shot} = 0.932}_{\text{pocas clases, poca evidencia}} \;\approx\; \underbrace{\text{4-way, 5-shot} = 0.943}_{\text{más clases, más evidencia}}$$

La dificultad extra de **duplicar las clases** se "paga" exactamente con **más shots**. Esto sugiere que el verdadero parámetro de dificultad del few-shot no es WAYS ni SHOTS por separado, sino su relación: la **razón evidencia/clases**. A más ejemplos por clase que hay que separar, más fácil; el problema es la información disponible *por frontera de decisión*.

> **Matiz — los problemas con más WAYS convergen más lento.** Con presupuesto fijo de iteraciones, el 8-way queda algo *subentrenado* frente al 2-way: necesita más pasos para que $\theta$ aprenda discriminaciones más finas. Parte del gap del 8-way está, por tanto, inflado por entrenamiento incompleto, no solo por dificultad intrínseca.

## Actividad 3 — MAML en Mini-ImageNet: información vs. optimización

Mini-ImageNet sube la apuesta: imágenes naturales a color, con fondo, iluminación y poses variables — mucho más difíciles que los caracteres binarios de Omniglot. El baseline 4-way 1-shot con 60 iteraciones da **Test 0.299**, prácticamente el azar (25%).

**El fenómeno que confunde: el error baja pero el accuracy no se mueve.** Durante el entrenamiento, la cross-entropy cae de \~2.5 a \~0.4, lo que normalmente sería una excelente señal. Sin embargo el accuracy queda plano en \~0.30. La explicación: el modelo aprende a dejar de estar **"confiado y equivocado"** — la pérdida penaliza fuerte las predicciones seguras pero incorrectas, así que el optimizador primero corrige la *calibración* (baja la confianza) antes de mejorar la *discriminación*. La pérdida mejora porque las probabilidades se vuelven más prudentes, pero el `argmax` —que es lo que mide el accuracy— sigue cayendo casi al azar.

![MAML Mini-ImageNet baseline 4w1s](/laboratorios/lab-26/maml-miniimagenet-base.jpg)

> **Gotcha — una pérdida que baja no garantiza un modelo que discrimina.** Cross-entropy y accuracy no son intercambiables: la primera es sensible a la confianza, el segundo solo al ganador. Vigilar *ambas* curvas evita celebrar un entrenamiento que en realidad solo se está calibrando.

**Ablation factorial 2×2** (todas con 150 iteraciones y argumentos afinados; `batch=32` por límite de memoria):

| Config | Test | Azar | Normalizado |
|---|---|---|---|
| 4-way 1-shot | 0.324 | 0.250 | 0.099 |
| 4-way 5-shot | 0.491 | 0.250 | 0.321 |
| 2-way 1-shot | 0.547 | 0.500 | 0.094 |
| 2-way 5-shot | 0.710 | 0.500 | 0.420 |

**SHOTS es la palanca dominante, siempre.** Pasar de 1 a 5 shots aporta entre **+0.222 y +0.326** de accuracy normalizada en ambos niveles de WAYS. En imágenes naturales un solo ejemplo por clase es radicalmente insuficiente: una sola foto no captura la variación intra-clase (un perro en distintas poses, fondos, escalas). Cinco ejemplos empiezan a delinear la clase como concepto, no como instancia.

**WAYS revela una interacción, no un efecto principal limpio.** Reducir de 4 a 2 WAYS da resultados aparentemente contradictorios:

- Con **1 shot**: el normalizado *baja* −0.005 (0.099 → 0.094) — esencialmente cero.
- Con **5 shots**: el normalizado *sube* +0.099 (0.321 → 0.420) — ganancia real.

El −0.005 con 1 shot es un **espejismo del azar**: en 2-way 1-shot el accuracy crudo sube a 0.547, pero como el azar también saltó a 50%, una vez normalizado no hay ganancia genuina — el modelo no aprendió a discriminar mejor, solo se benefició de tener menos opciones donde fallar. La conclusión es una **interacción**: *reducir clases solo aporta poder real cuando hay evidencia que explotar*. Sin shots suficientes, tener menos clases no ayuda porque el problema no era cuántas clases, sino que no había información para caracterizar ninguna.

**El overfitting también se elimina en la esquina rica.** Brecha train − valid por config: 4w1s **0.086**, 4w5s **0.095**, 2w1s **0.078**, 2w5s **0.007**. La combinación con más evidencia y menos clases (2w5s) prácticamente borra el overfitting — más shots dan al inner loop un *support set* representativo, y menos clases simplifican la frontera, de modo que el modelo generaliza en vez de memorizar el lote.

![MAML Mini-ImageNet 4w5s](/laboratorios/lab-26/maml-miniimagenet-5shot.jpg)

**La lección clave — el cuello de botella cambió de naturaleza.** En Omniglot (Actividad 1) la palanca que mandaba era `num_iterations`: el problema era de **optimización**, había que entrenar más. En Mini-ImageNet la palanca que manda es SHOTS: el problema es de **información**, falta evidencia. No hay cantidad de iteraciones que compense un solo ejemplo por clase cuando la clase es visualmente compleja. **La naturaleza del dato determina qué palanca importa** — caracteres simples se resuelven entrenando más; imágenes naturales se resuelven con más evidencia por clase. Diagnosticar *cuál* de las dos limita es el primer paso para gastar el presupuesto de cómputo donde rinde.

## Enlaces

- Fundamentos: [Few-shot learning](/fundamentos/few-shot-learning) · [Optimización binivel](/fundamentos/optimizacion-binivel) · [Meta-aprendizaje](/fundamentos/meta-aprendizaje)
- Páginas: [MAML: fundamentos](maml-fundamentos) · [Experimentos con Prototypical](experimentos-prototypical) · [Comparación y teoría](comparacion-y-teoria)
- Papers: [MAML (Finn 2017)](/papers/maml-finn-2017) · [Optimization as a Model for Few-Shot Learning (Ravi 2017)](/papers/ravi-optimization-fewshot-2017)
