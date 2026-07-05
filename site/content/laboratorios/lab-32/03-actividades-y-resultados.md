---
title: "Las 4 actividades resueltas"
weight: 3
---

Las cuatro actividades del práctico, resueltas con experimentos propios y todos los números medidos en el notebook ejecutado (GPU T4). El ruido de inicialización entre corridas idénticas es de **~1.6 puntos** — la vara para juzgar si una diferencia es señal o ruido.

## Actividad 1 — ¿Afecta el orden? ¿Cómo escala con más tareas?

Se corrió Naive con la secuencia normal `[0,1,2]` y la invertida `[2,1,0]`, y luego con 2, 3 y 5 tareas.

**P1 — Orden:**

| Orden | Avg ACC final |
|---|---|
| Normal `[0,1,2]` | 42.06% |
| Invertido `[2,1,0]` | 39.95% |
| Diferencia | **2.11 pts** |

La diferencia (2.11 pts) es del orden del ruido de init (~1.6 pts) → **el orden no afecta** el promedio. En Permuted MNIST las tareas son simétricas en dificultad e independientes, así que bajo Naive el patrón "solo sobrevive la última, el resto en ruinas" es invariante al orden. Lo que cambia es *cuál* tarea se recuerda, no *cuánto* en promedio. (En un benchmark con tareas de dificultad desigual, el orden sí importaría vía efectos de currículo.)

**P2 — Escalabilidad:**

| num_tasks | Avg ACC final |
|---|---|
| 2 | 42.68% |
| 3 | 39.56% |
| 5 | **29.24%** |

Descenso monótono: de 2→5 tareas se pierden ~13 puntos. Bajo Naive el modelo retiene solo la última tarea (~75-80%) y las N–1 anteriores caen a un residuo bajo (~15-20%); el promedio tiende a ese residuo conforme crece N. **El aprendizaje secuencial ingenuo no escala** — la motivación central del campo.

## Actividad 2 — Trade-off buffer ↔ accuracy ↔ memoria (Rehearsal)

Barrido del tamaño del buffer, midiendo accuracy y memoria del buffer:

| Buffer/tarea | Memoria | Avg ACC | Eficiencia |
|---|---|---|---|
| 0 (=Naive) | 0.00 MB | 42.00% | — |
| 100 | 0.63 MB | 45.85% | 6.1 pts/MB |
| 500 | 3.14 MB | 65.09% | **7.7 pts/MB** ⭐ |
| 1.000 | 6.29 MB | 71.99% | 2.2 pts/MB |
| 2.000 | 12.58 MB | 73.07% | 0.17 pts/MB |
| 5.000 | 31.44 MB | 84.51% | 0.61 pts/MB |
| 10.000 | 62.88 MB | 86.25% | 0.055 pts/MB |

![Dos curvas: accuracy vs tamaño del buffer, y accuracy vs memoria, ambas cóncavas](/laboratorios/lab-32/act2-tradeoff-buffer-memoria.png)

Curva cóncava con **rendimientos decrecientes**:

- **El "codo" está en ~500-1.000.** El salto brutal es 100→500 (+19 puntos por solo 2.5 MB): el modelo pasa de "casi no recordar" a "recordar razonablemente". Zona de máxima eficiencia (7.7 pts/MB).
- **Después de 1.000, saturación.** De 1.000 (72%) a 10.000 (86%) solo ganas 14 puntos a costa de 10× más memoria. Te acercas al techo del oráculo (joint training, ~90%). Pasar de 84.5%→86.25% cuesta 31 MB extra (0.055 pts/MB, 140× menos eficiente que el codo).
- **La no-monotonía** (2.000 apenas sube sobre 1.000, luego 5.000 salta) es ruido de corrida única, no una ley. La tendencia es sólida; los puntos individuales tienen ±ruido.

**P3 — ¿Por qué shuffle antes de entrenar?** Porque `train` recorre los datos secuencialmente en batches de 256 sin barajar. Sin shuffle, al concatenar `[nuevos, viejos]` los batches quedan "puros" y el gradiente de cada batch tira hacia una sola distribución, reproduciendo el olvido dentro de la propia época. `shuffle_in_unison` mezcla ambas tareas dentro de cada batch, y el "in unison" preserva la correspondencia imagen-etiqueta.

## Actividad 3 — Curva de λ (EWC) y comparación de memoria

Barrido de λ en 8 órdenes de magnitud:

| λ | Avg ACC | Régimen |
|---|---|---|
| 0.7 | 47.22% | Plástico — no protege (≈Naive) |
| 10 | 38.62% | Plástico |
| 100 | 53.63% | Empieza a proteger |
| 1.000 | 56.39% | Subiendo |
| **10.000** | **58.98%** | **Óptimo** ⭐ |
| 100.000 | 9.80% | 💥 Colapso (azar) |
| 1.000.000 | 9.80% | 💥 Colapso |
| 10.000.000 | 9.80% | 💥 Colapso |

![Curva de accuracy vs lambda en escala log: sube hasta 10000 y colapsa a azar después](/laboratorios/lab-32/act3-curva-lambda.png)

La **firma canónica del dilema estabilidad-plasticidad**, en tres regímenes:

1. **Zona plástica (λ ≤ 10): ~40-47%.** Resortes blandos; la penalización es despreciable → olvido, como Naive. Aquí vivía el λ=0.7 por defecto.
2. **Zona óptima (λ ≈ 100-10.000): pico de 58.98%.** Rigidez justa: protege los pesos viejos sin ahogar el aprendizaje nuevo. +15 puntos sobre Naive, **sin guardar ni un dato**.
3. **Colapso (λ ≥ 100.000): 9.8% = azar en TODAS las tareas.** Y aquí el detalle sutil: si fuera por rigidez ("no puede aprender lo nuevo") verías T0 alta y T1/T2 en azar → ~38%. Pero es azar en todo, **incluida T0**: es **divergencia numérica**. Con λ enorme, el gradiente de la penalización explota, el paso de SGD sobredispara, los pesos revientan a NaN/basura y el modelo queda destruido. Es exactamente lo que el parámetro `fisher_clip` (declarado en `on_task_update` pero **nunca implementado**) debía prevenir recortando los valores de Fisher.

**Comparación de memoria (P2):**

| | Accuracy | Memoria | Escala con |
|---|---|---|---|
| **EWC** (λ=10⁴, óptimo) | 58.98% | **0.52 MB** | tamaño del **modelo** |
| **Rehearsal** (500/tarea) | 65.09% | 3.14 MB | **datos** × tareas |

EWC guarda Fisher + θ* por tarea (2 × 21.840 params × 3 tareas × 4 bytes = 0.52 MB). Rehearsal(500) guarda 500 imágenes por tarea pasada (3.14 MB). **Rehearsal gana en accuracy (+6 pts), EWC gana en memoria (6× menos).** La ventaja estructural de EWC: su memoria depende del tamaño del modelo, no de los datos, y no retiene datos crudos (privacidad). Rehearsal escala con datos × tareas y puede volverse inviable con datasets grandes.

## Actividad 4 — Síntesis comparativa

| Método | Trayectoria | Mecanismo | Costo | Final |
|---|---|---|---|---|
| **Naive** | Plana ~44% | Ninguno (SGD puro) | 0 | Olvida todo menos la última |
| **EWC** | Sube a ~59% | Penaliza mover pesos importantes (Fisher) | ~0.5 MB (2× modelo) | Protege sin datos |
| **Rehearsal** | Sube a ~70% | Reentrena con buffer de datos viejos | ~3 MB (crece con datos) | Mejor accuracy, más memoria |

**Cuándo conviene cada uno:**

- **Rehearsal** cuando puedes almacenar datos legalmente y con RAM de sobra, las tareas son muy distintas entre sí, y el número de tareas es moderado. El baseline más fuerte cuando la memoria no es problema.
- **EWC** cuando **no puedes guardar datos** (privacidad clínica, GDPR/HIPAA), o cuando el modelo es chico pero los datos enormes (escala con el modelo, no con los datos), o cuando las tareas comparten estructura. El caso de matching de pacientes / MDM cae aquí: no querrías retener registros clínicos en un buffer.
- **Naive** solo cuando no importan las tareas viejas, o cuando en realidad haces joint training disfrazado (todos los datos siempre disponibles). Es el control negativo.

**El matiz de "aprender para el futuro":** EWC tiene un riesgo a largo plazo — con muchas tareas, cada vez más pesos quedan consolidados (Fisher acumulado alto) y la red se vuelve **progresivamente rígida**, perdiendo capacidad de aprender (*capacity saturation*). Rehearsal no sufre esto pero paga en memoria creciente. Ninguno es universalmente superior: es un trade-off de tres vías entre **accuracy, memoria y plasticidad futura**.
