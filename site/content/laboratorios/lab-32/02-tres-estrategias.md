---
title: "Las tres estrategias"
weight: 2
---

Sobre una secuencia fija de 3 tareas (T0 original + T1/T2 permutadas con `seed=1,2`), se comparan tres estrategias. Todas parten del **mismo modelo fresco** (pesos aleatorios) — rigor experimental: si cada una partiera de un modelo distinto, no serían comparables. El esqueleto de evaluación es común: tras entrenar cada tarea, se mide el accuracy en **todas** las tareas y se promedia (la métrica **Average Accuracy** de la literatura).

## Naive — el baseline

Entrenar tarea tras tarea con SGD puro, sin ninguna defensa. Es la versión sistematizada de la demostración de olvido, medida sobre 3 tareas.

Matriz de accuracy medida (negrita = tarea recién aprendida):

| Después de entrenar ↓ | T0 | T1 | T2 | Avg |
|---|---|---|---|---|
| Tarea 0 | **94.73%** | 6.27% | 12.95% | 37.98% |
| Tarea 1 | 21.43% | **74.61%** | 10.94% | 35.66% |
| Tarea 2 | 23.64% | 28.47% | **79.78%** | 43.96% |

Los tres fenómenos que revela:

1. **La diagonal siempre gana** (94→75→80): la tarea recién entrenada rinde bien. Aprender nunca fue el problema.
2. **El olvido es inmediato y masivo**: T0 cae 73 puntos en cuanto llega T1, y **nunca se recupera** (21→24%) porque nada le recuerda a la red que T0 existió.
3. **El olvido es acumulativo y encadenado**: T1 estaba en 74.61% y tras entrenar T2 cayó a 28.47%. Cada tarea nueva erosiona *todas* las anteriores. Con 10 tareas, las primeras quedarían en ruinas.

**El promedio engaña**: sube (38→36→44%) no porque haya menos olvido, sino porque la tarea recién aprendida infla el promedio. La señal real de olvido está en las **columnas** (T0: 94→24, –70 pts).

## Rehearsal — replay de datos viejos

Si el problema es que olvidas los datos viejos, guárdalos y vuelve a mostrárselos. Se mantiene un **buffer** de ejemplos de tareas pasadas y se **mezclan** con los datos nuevos en cada batch.

```python
num_past_elem = 1000   # ejemplos guardados POR tarea pasada (1.67% de 60k)

for id, task in enumerate(tasks):
    (x_train, y_train), _ = task
    for i in range(id):                                   # tareas anteriores
        (px, py), _ = tasks[i]
        idx = random.sample(range(len(py)), num_past_elem)
        x_train = np.concatenate((x_train, px[idx]))       # añade 1000 viejos
        y_train = np.concatenate((y_train, py[idx]))
    x_train, y_train = shuffle_in_unison([x_train, y_train], 0)  # ← clave
    for epoch in range(1, num_epoch):
        train(model, device, x_train, y_train, optimizer, epoch)
```

Es una estrategia a nivel de **datos**, no de modelo: no cambia ni la red, ni la pérdida, ni el optimizador — solo el contenido del dataset. Por eso es tan fácil de implementar y tan popular como baseline fuerte.

**Por qué `shuffle_in_unison` es imprescindible.** La función `train` recorre los datos en orden secuencial por batches de 256, sin barajar. Sin shuffle, al concatenar `[nuevos, viejos]` los batches quedarían "puros" (256 imágenes de una sola tarea) y el gradiente de cada batch tiraría hacia una sola distribución. Barajar mezcla viejos y nuevos **dentro de cada batch**, de modo que cada actualización ve ambas tareas a la vez. El "in unison" garantiza que imágenes y etiquetas se barajen con la **misma** permutación (usa `get_state`/`set_state` del RNG para reproducir la permutación exacta en ambos arrays).

Matriz medida (buffer = 1.000):

| Después de entrenar ↓ | T0 | T1 | T2 | Avg |
|---|---|---|---|---|
| Tarea 0 | 94.53% | 6.02% | 11.54% | 37.36% |
| Tarea 1 | 78.82% | 83.34% | 8.64% | 56.93% |
| Tarea 2 | **80.32%** | **43.94%** | 85.80% | **70.02%** |

Frente a Naive (44%), el Avg final salta a **70%** (+26 pts). Y la trayectoria **sube monótonamente** (37→57→70) en vez de estancarse — la firma de una estrategia que funciona: acumula conocimiento en vez de reemplazarlo.

**El hallazgo contraintuitivo:** T0 se retuvo (80%) mucho mejor que T1 (44%), pese al mismo buffer. La razón estructural es la **frecuencia de rehearsal**: T0 se rehearsó **dos veces** (al entrenar T1 y al entrenar T2), mientras T1 solo **una** (al entrenar T2). En este esquema, cuanto más antigua la tarea, más veces se refuerza — al revés del olvido natural. (Contribuye también que T0 es MNIST original, cuya estructura espacial la CNN retiene más fácil que una permutación arbitraria.)

## EWC — proteger los pesos importantes

[EWC](/fundamentos/aprendizaje-continuo) (Elastic Weight Consolidation, Kirkpatrick et al. 2017) resuelve el mismo dilema a nivel de **pesos**, no de datos. Su gran ventaja: **no guarda ni un solo dato** — solo estadísticas de los pesos (~2× el tamaño del modelo por tarea). Ideal cuando la privacidad importa (datos clínicos).

**La intuición geométrica.** Tras aprender la Tarea A, los pesos están en el fondo de un valle de la superficie de pérdida. Algunas direcciones son empinadas (mover ahí dispara la pérdida de A → peso importante) y otras planas (mover no afecta A → peso libre). EWC dice: al aprender B, muévete libre por las direcciones planas, resiste las empinadas.

La pérdida total suma la tarea nueva más un resorte cuadrático por cada peso importante:

$$L = L_B(\theta) + \sum_i \frac{\lambda}{2} F_i (\theta_i - \theta_{A,i}^*)^2$$

- $(\theta_i - \theta_{A,i}^*)^2$ — cuánto se alejó el peso de su valor bueno para A (resorte elástico).
- $F_i$ — la rigidez del resorte = importancia del peso para A.
- $\lambda$ — la fuerza global de todos los resortes (el mando estabilidad↔plasticidad).

**La aproximación clave: $F_i = g_i^2$.** $F$ es la matriz de información de Fisher = curvatura de la log-verosimilitud en el óptimo (coincide con el Hessiano). Calcular el Hessiano completo (21.840² entradas) es inviable, así que EWC hace dos aproximaciones: (1) solo la **diagonal** (21.840 números), y (2) el resultado estadístico de que, en el óptimo, $\mathbb{E}[g_i^2]$ = información de Fisher. La intuición: si mover un peso produce gradientes grandes, es importante; si los gradientes son ~0, es libre. **El gradiente al cuadrado te dice gratis qué pesos importan.**

El cálculo de Fisher (tras cada tarea) hace forward+backward pero **sin `optimizer.step()`** — solo mide gradientes:

```python
def on_task_update(task_id, x_mem, t_mem, bs=256):
  model.eval()                                    # desactiva dropout: Fisher limpio
  fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
  N = 0
  for start in range(0, len(t_mem), bs):
      ...
      F.cross_entropy(model(x), y).backward()      # llena p.grad, NO actualiza pesos
      for name, p in model.named_parameters():
          fisher[name] += (p.grad.detach()**2) * bsz   # ← F_i = g_i²
      N += bsz
  fisher_dict[task_id] = {n: fisher[n] / N for n in fisher}         # E[g²]
  optpar_dict[task_id] = {n: p.detach().clone() for n, p in ...}    # θ* congelado
```

Y la penalización se añade en el entrenamiento (aquí sí con el cuadrado):

```python
loss = F.cross_entropy(output, y)
for task in range(task_id):
    for name, param in model.named_parameters():
        fisher = fisher_dict[task][name]
        optpar = optpar_dict[task][name]
        loss += (fisher * (optpar - param).pow(2)).sum() * ewc_lambda
```

Nótese que este bloque **no toca ningún dato viejo** — solo lee `fisher_dict` y `optpar_dict`. Esa es la promesa de EWC hecha código.

**Con λ=0.7 (el valor por defecto del notebook), EWC casi no protege:** T0=20%, T1=37%, Avg=46% — apenas 2 puntos sobre Naive. Es demasiado débil: la penalización es despreciable frente a la cross-entropy. El paper original usa λ ~400-40.000. El barrido de λ (en [las actividades](03-actividades-y-resultados)) muestra que con λ=10.000 EWC llega a 58.98%, y que pasarse (λ≥100.000) lo hace **colapsar a azar** por divergencia numérica.

## Comparación

Las tres trayectorias del Avg ACC a lo largo de las tareas (EWC aquí con λ=0.7, subrepresentado):

![Comparación de las tres estrategias: Naive plano, Rehearsal y EWC subiendo](/laboratorios/lab-32/comparacion-estrategias.png)

Naive se mantiene plano; Rehearsal es la que más sube. Las tres arrancan igual en la Tarea 1 (no hay pasado que proteger). El orden final — Rehearsal > EWC > Naive — y el porqué de cada trade-off se analizan en las [actividades](03-actividades-y-resultados).
