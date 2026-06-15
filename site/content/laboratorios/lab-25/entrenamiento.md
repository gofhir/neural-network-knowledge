---
title: "Entrenamiento: multimodal vs baseline"
weight: 3
math: true
---

> **El experimento central del lab (celdas 34-52).** Entrenamos dos modelos sobre el mismo *proxy task* —clasificar a qué usuario pertenece un item— y comparamos. El multimodal (texto + imagen) contra el baseline solo-imagen. La pregunta: *¿aporta algo el texto?* Spoiler: sí, ~15 puntos de accuracy. Esta página desmonta las funciones de entrenamiento, los resultados reales ejecutados, y resuelve la Actividad 2.

## 1. Las funciones de entrenamiento (`run_epoch`, `run_training`)

El bucle es PyTorch estándar, sin trucos exóticos. Lo interesante está en las decisiones de diseño:

```python
criterion = nn.CrossEntropyLoss()          # clasificación = el proxy task
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer)   # baja LR cuando la val loss se estanca
```

| Pieza | Elección | Por qué |
|---|---|---|
| Loss | `CrossEntropyLoss` | El proxy es **clasificar item → 1 de 10 usuarios**. Es un problema multiclase. |
| Optimizador | Adam, `lr=1e-4` | LR conservador; estamos fine-tuneando un BERT pre-entrenado y no queremos dañarlo. |
| Scheduler | `ReduceLROnPlateau` | Reactivo, no programado: cuando la **val loss** deja de bajar, recorta el LR. Se invoca con `scheduler.step(val_loss)` al cerrar la época de validación. |

**Valida ANTES de entrenar.** La primera medición ("pre", época 0) es el modelo recién inicializado: con 10 clases balanceadas, el azar es ~10%, y eso es justo lo que se observa (9.47% multimodal, 8.30% baseline). Es el *sanity check* que confirma que el accuracy posterior es aprendizaje real y no un artefacto.

**El forward recibe 4 entradas.** El modelo multimodal toma `input_ids`, `attention_mask`, `token_type_ids` (las tres del texto, vía BERT) más el descriptor de imagen, y los fusiona. Las predicciones salen de un `argmax` sobre el logit de 10 usuarios:

```python
preds = torch.argmax(outputs, dim=1)   # ¿a cuál de los 10 usuarios?
```

### El bug sutil (inofensivo)

En el control de fase aparece:

```python
elif phase == 'val' or 'test':   # ⚠️ siempre True
```

La expresión `phase == 'val' or 'test'` se evalúa como `(phase == 'val') or ('test')`, y `'test'` es una cadena no vacía → **siempre verdadera**. Lo correcto sería `phase in ('val', 'test')`. En este lab no rompe nada porque solo se usan las fases `'train'` y `'val'`, pero es el tipo de error que en otro código metería un *test set* dentro de la rama de validación sin avisar.

## 2. ¿Por qué CrossEntropy y no triplet loss?

El lab menciona una alternativa: en lugar de clasificar, podríamos usar [triplet loss](/fundamentos/triplet-loss) —acercar en el espacio de embeddings los items que comparte un mismo usuario y alejar los de usuarios distintos. Conceptualmente encaja mejor con la tarea final (recomendar por similitud).

El problema es **operativo**: triplet loss exige generar tripletas (ancla, positivo, negativo), y muestrearlas bien (*hard negative mining*) es costoso. Por eso quedó el vestigio `amount_triplet` en el código: una idea contemplada y abandonada en favor del proxy de clasificación, más barato de entrenar. El lab *entrena clasificando* aunque *evalúa con ranking* — tensión que retomamos en la Actividad 2.

## 3. Resultados reales — modelo multimodal (`ModelClass`, celda 44)

Ejecución de 5 épocas (texto BERT + descriptor de imagen):

| Época | Train acc | Val acc | Val loss |
|---|---|---|---|
| pre (azar) | — | 9.47% | 2.38 |
| 1 | — | 51.46% | — |
| 2 | — | 61.43% | — |
| 3 | — | 65.82% | — |
| 4 | — | 69.63% | — |
| 5 | — | **71.48%** | **0.87** |

La val loss cae monótonamente de 2.38 a 0.87 y el accuracy sube en todas las épocas sin señales de saturación: a la época 5 **todavía estaba mejorando**. Con más épocas probablemente seguiría subiendo.

## 4. Resultados reales — baseline solo-imagen (`ModelClassImage`, celda 45)

El mismo entrenamiento, pero el modelo solo ve el descriptor de imagen (sin texto):

| Época | Val acc |
|---|---|
| pre (azar) | 8.30% |
| 1 | 46.39% |
| 2 | 52.64% |
| 3 | 56.25% |
| 4 | **58.30%** |
| 5 | 56.84% ↓ |

Aquí pasa algo distinto: el accuracy **satura cerca de 58%** y en la época 5 **cae** (58.30 → 56.84) mientras el train acc sigue subiendo. Eso es el inicio clásico del **overfitting**: el modelo empieza a memorizar el train a costa de la validación.

## 5. Comparación y hallazgos

| Modelo | Val acc final | Comportamiento |
|---|---|---|
| **Multimodal** (texto + imagen) | **71.48%** | Sigue subiendo, sin saturar |
| Baseline (solo imagen) | 56.84% | Satura ~58%, overfitting en ep. 5 |
| **Diferencia** | **+14.6 pts** | El texto aporta señal complementaria |

Tres conclusiones:

1. **El texto aporta ~15 puntos de accuracy.** Confirma la hipótesis del lab: la descripción textual del item lleva información que la imagen sola no captura (marca, material, contexto de uso, palabras que el usuario asocia).
2. **El baseline overfittea; el multimodal no (todavía).** El baseline ya pide **early stopping** en la época 4. El multimodal tiene más capacidad de señal y aún no toca su techo.
3. La brecha no es solo "más parámetros": es **más información**. Dos modalidades describen al item desde ángulos distintos y se complementan.

## 6. El patrón Val Acc > Train Acc (no es un error)

En **todas** las épocas el val acc supera al train acc. Esto desconcierta —intuitivamente esperamos lo contrario— pero **no es overfitting ni un bug**. Hay dos causas que se suman:

- **(a) Dropout activo en train, apagado en val.** Con `Dropout(0.3)`, durante el entrenamiento se "mutila" el 30% de las activaciones en cada paso → el modelo predice con una versión degradada de sí mismo. En validación se llama `model.eval()`, el dropout se desactiva y el modelo opera al 100% de su capacidad. El val acc se mide con un modelo *más fuerte*.
- **(b) El train acc es un promedio acumulado de la época.** Se calcula sobre todos los batches, **incluyendo los primeros**, cuando el modelo aún estaba malo dentro de esa misma época. El val acc se mide **al final**, con los pesos ya mejorados. El train acc arrastra el lastre de los batches tempranos.

> **Cómo se vería el overfitting de verdad:** train acc $\gg$ val acc (el modelo memoriza el train y falla en datos nuevos). Eso es lo que insinúa el **baseline** en la época 5. El multimodal, con val $>$ train estable y ambas curvas subiendo, está sano.

## 7. El reporte de claves UNEXPECTED al cargar BERT

Al cargar `bert-base-uncased`, HuggingFace imprime un aviso de claves **UNEXPECTED**:

```text
Some weights of the model checkpoint were not used:
  - cls.predictions.transform.dense.weight ...
  - cls.seq_relationship.weight ...
```

Son las **cabezas de pre-entrenamiento** de BERT:

- `cls.predictions.*` → cabeza de **MLM** (Masked Language Modeling).
- `cls.seq_relationship.*` → cabeza de **NSP** (Next Sentence Prediction).

Como solo usamos el **encoder** (`BertModel`, los embeddings contextuales) y no las tareas de pre-entrenamiento, esas cabezas no tienen dónde encajar y se descartan. El aviso es **totalmente inofensivo**: indica que cargamos el cuerpo del modelo y dejamos fuera el andamiaje de pre-entrenamiento que no necesitamos.

## 8. Actividad 2 (resuelta): limitaciones, mejoras y data augmentation

### Limitaciones del enfoque

- **Lim 1 — Clasificador de usuarios fijos → cold-start.** El modelo aprende a clasificar entre **10 usuarios concretos**. Si llega un usuario nuevo, no existe una clase para él: hay que reentrenar. Es el problema de *cold-start* de usuario. La solución de la industria es un **two-tower retrieval** ([Yi et al. 2019](/papers/two-tower-yi-2019)): una torre de usuario y otra de item proyectan a un espacio común, y un usuario nuevo se representa por sus features, sin necesidad de una clase dedicada.
- **Lim 2 — Descriptores de imagen congelados de ImageNet.** Las features visuales vienen de una CNN pre-entrenada en ImageNet y **no se ajustan al dominio** (recomendación de productos). Una red entrenada para distinguir "perro vs gato" no necesariamente captura lo que hace deseable un producto. **VBPR** ([He & McAuley 2016](/papers/vbpr-he-2016)) ya advertía esto: integra features visuales en el ranking, pero idealmente se las afina al objetivo de recomendación, no se dejan congeladas.

### Mejoras con temas del diplomado

- **Pérdida de ranking en vez de clasificación.** El lab **entrena clasificando** (CrossEntropy) pero **evalúa con ranking** (nDCG en la página siguiente). Hay un *mismatch* objetivo/evaluación. Lo coherente sería entrenar con una pérdida de ranking como **BPR** ([Rendle et al. 2009](/papers/bpr-rendle-2009)), que optimiza directamente que el item relevante quede por encima del irrelevante —exactamente lo que mide el nDCG.
- **Early stopping.** Visto el overfitting del baseline en la época 5, detener el entrenamiento cuando la val loss deja de mejorar evitaría degradar el modelo (el `ReduceLROnPlateau` mitiga, pero no detiene).

### Data augmentation por modalidad

| Modalidad | Técnica aplicable | Por qué |
|---|---|---|
| **Texto** | Back-translation, reemplazo por sinónimos | Genera descripciones equivalentes que preservan el significado. |
| **Imagen** | **Ruido gaussiano / mixup en el espacio de features** | Aquí está el gotcha: solo tenemos **descriptores** (vectores), no los píxeles (omitidos por copyright). Las técnicas clásicas de visión —flips, crops, rotaciones— **no aplican** porque operan sobre la imagen cruda. Hay que augmentar en el **espacio de features** ya extraído. |

> El punto fino de la Actividad 2: con descriptores en vez de píxeles, el augmentation de imagen cambia de naturaleza. No se aumenta la imagen, se aumenta su *embedding*.

---

**Anterior:** [Dataset multimodal y arquitectura](dataset-y-modelo) · **Siguiente:** [Recomendación por similitud](recomendacion)
