---
title: "Transfer Learning y Finetuning"
weight: 20
math: true
---

## La idea de transfer learning

Entrenar un modelo desde cero requiere **mucha** información: ResNet18 tiene ~11M de parámetros, y aprenderlos todos desde inicialización aleatoria con solo 5,687 imágenes (lo que tiene flowers train) es prácticamente imposible. El modelo base de la sección anterior alcanzó 42.58% en test después de 5 épocas — saturando rápidamente cerca de 50% en train, sin haber aprendido representaciones útiles.

**Transfer learning** resuelve este problema reutilizando los pesos de un modelo entrenado previamente sobre un dataset masivo (típicamente ImageNet, con 1.2M imágenes y 1000 clases). La intuición es que las primeras capas de una CNN aprenden features genéricas (bordes, texturas, patrones de bajo nivel) que son útiles para **cualquier** tarea de visión, no solo para las clases de ImageNet.

Hay dos modalidades clásicas:

| Modalidad | Cómo se hace | Cuándo usarla |
|---|---|---|
| **Feature extraction** | Congelar todo el backbone y entrenar solo la cabeza nueva | Dataset muy chico, o features ya muy alineadas con la tarea |
| **Finetuning** | Entrenar todo el modelo (backbone + cabeza) con LR bajo | Dataset mediano-grande, o features que necesitan especialización |

Este laboratorio aplica **finetuning completo**: ningún parámetro queda congelado, pero se parte de pesos de ImageNet en vez de inicialización aleatoria.

## Implementación

```python
ft_model = models.resnet18(pretrained=True)
ft_model.fc = nn.Linear(in_features=512, out_features=n_classes, bias=True)
```

Las dos líneas hacen lo siguiente:

1. **`models.resnet18(pretrained=True)`**: descarga el ResNet18 de torchvision con los pesos entrenados sobre ImageNet (~45 MB). El modelo viene con capa final `fc` de 1000 outputs (las 1000 clases de ImageNet).
2. **`ft_model.fc = nn.Linear(512, n_classes)`**: reemplaza esa capa final por una nueva `Linear(512, 102)` para las 102 clases de flowers. Esta capa nueva se inicializa aleatoriamente — todo lo demás conserva los pesos de ImageNet.

El feature extractor (todas las capas convolucionales) entra al entrenamiento ya sabiendo "ver" estructuras visuales generales. Solo el clasificador final empieza desde cero.

## Hiperparámetros del finetuning

Mismo setup que los modelos anteriores: Adam (`lr=5e-4`), `batch_size=128`, 5 épocas, sin augmentation (transforms determinísticos: resize 256 + center crop 224 + normalize ImageNet). El modelo no necesita augmentation porque ya parte de un punto excelente.

> **Nota sobre el LR**: para finetuning completo de un modelo preentrenado, la práctica recomendada es **bajar el LR 10x** respecto al entrenamiento from scratch. ResNet original usa `lr=0.1`, así que un LR de finetuning canónico sería ~`0.01`. Aquí se usa `5e-4` (que es lo que usaba el baseline) — funciona bien por la combinación con Adam, que adapta el LR efectivo por parámetro. Con un LR demasiado alto se corre el riesgo de "olvidar" lo aprendido en ImageNet (catastrophic forgetting).

## Resultados modelo finetuning

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 0 (val inicial) | — | — | 4.888 | 0.47% |
| 1 | 1.724 | 63.98% | 0.760 | 80.47% |
| 2 | 0.258 | 94.53% | 0.402 | 87.11% |
| 3 | 0.094 | 97.33% | 0.321 | 87.97% |
| 4 | 0.071 | 97.52% | 0.293 | 88.36% |
| 5 | **0.037** | **98.23%** | **0.274** | **89.14%** |

**Test final:** Loss = 0.300, Acc = **92.66%**.

![Loss finetuning](/laboratorios/lab-12/loss-ft.png)
![Accuracy finetuning](/laboratorios/lab-12/acc-ft.png)

### Observaciones clave de las curvas

- **Val acc inicial de 0.47%**: el modelo recién creado clasifica básicamente al azar entre 102 clases (random ≈ 0.98%, así que está incluso un poco peor por la cabeza recién inicializada). Esto confirma que el conocimiento de ImageNet **no transfiere automáticamente** la respuesta — solo provee features útiles que el clasificador final debe aprender a usar.
- **Salto enorme en epoch 1**: train sube a 63.98% y val a 80.47% en una sola pasada por el dataset. Esto sería imposible desde inicialización aleatoria. Las features preentrenadas son tan informativas que con pocas iteraciones el clasificador final ya separa clases.
- **Val supera a train en epoch 1** (80.47% > 63.98%): asimetría temporal. Train se mide promediando sobre el batch (incluyendo los primeros pasos donde el modelo todavía estaba aleatorio), mientras que val se mide al final de la época con un modelo ya muy mejorado.
- **Sobreajuste moderado desde epoch 3**: train llega a 97-98%, pero val se estanca en 88-89% (gap ~9pp). Sería buen lugar para introducir augmentation y reducir el gap.
- **Test > val** (92.66% vs 89.14%): suerte del split. Sin leak ni nada raro.

## Predicción cualitativa con finetuning (sample 166)

```text
Ground Truth: bougainvillea

Top-5 Predictions (modelo ft) — ¡ACERTÓ!
  1. bougainvillea     (p=0.8566)  ✅
  2. mallow            (p=0.0467)
  3. pink primrose     (p=0.0121)
  4. pelargonium       (p=0.0073)
  5. sweet pea         (p=0.0067)
```

**Cambios cualitativos respecto al base:**

| Aspecto | Base | Finetuning |
|---|---|---|
| Top-1 | lotus (incorrecto) | **bougainvillea (correcto)** |
| Confianza top-1 | 0.167 | **0.857** |
| Suma top-5 | ~48% | **~93%** |
| Tipo de error | confunde por color | predicción dominante correcta |
| Alternativas top-5 | otras flores moradas | flores morfológicamente similares (no solo cromáticamente) |

Esto valida que las features de ImageNet enseñaron al modelo a **ver estructura**, no solo color. Las 4 alternativas top-2..5 (mallow, pink primrose, pelargonium, sweet pea) son flores que comparten morfología con bougainvillea (4-5 pétalos, simetría radial similar) — el modelo está confundido entre vecinas reales en el espacio de features, no entre cualquier flor que sea morada.

## Comparación final — los cuatro escenarios de Actividad I

| Modelo | Train Acc | Val Acc | Test Acc | Test Loss | Mejora vs base |
|---|---|---|---|---|---|
| **Base** (from scratch) | 49.91% | 40.55% | 42.58% | 2.171 | — |
| **Aug** (rot+crop+flip h.) | 49.46% | 39.92% | 44.45% | 2.139 | +1.87pp |
| **Ex** (vertical flip) | 48.65% | 39.45% | 40.55% | 2.196 | −2.03pp ⚠️ |
| **Finetuning** (ImageNet) | **98.23%** | **89.14%** | **92.66%** | **0.300** | **+50.08pp** ⭐ |

La diferencia es brutal:

- Augmentation aporta **+1.87pp** sobre el baseline. Es una mejora real pero modesta.
- Vertical flip (Ejercicio I) **degrada** el modelo en 2.03pp porque introduce una invarianza que **no existe** en el dominio.
- Finetuning aporta **+50.08pp**. No es una mejora marginal — es un cambio de régimen. Pasamos de un modelo casi inútil (42% en test) a un clasificador profesional (92.66%).

## Por qué finetuning gana tan dramáticamente

Tres razones técnicas:

1. **Inicialización informada vs aleatoria**. Los pesos de ImageNet están en una región del espacio de parámetros que ya extrae bordes, texturas y patrones de bajo nivel. Inicialización aleatoria parte de un mínimo terrible. Con solo ~5,700 imágenes, el modelo from-scratch nunca puede compensar el déficit.
2. **Capas tempranas son universales**. Las primeras capas convolucionales aprenden "detectores de Gabor" (orientaciones, frecuencias espaciales) que son **idénticos** entre tareas. El paper de [Yosinski et al. 2014](/papers/transferable-features-yosinski-2014) lo demostró cuantitativamente: las primeras capas son intercambiables entre datasets.
3. **Las clases de ImageNet incluyen muchas plantas y flores**. El feature extractor ya "vio" pétalos y hojas durante el preentrenamiento. La transferencia al dominio flowers es prácticamente directa.

## Cuándo NO usar finetuning

Aunque parezca el método dominante, finetuning tiene fronteras:

- **Dominio muy distinto al preentrenamiento**: imágenes médicas (radiografías), satelitales o microscópicas necesitan adaptación más profunda. ImageNet aprendió sobre fotos cotidianas.
- **Constraints de cómputo o latencia**: el modelo grande puede no caber en producción. Feature extraction (congelar el backbone) genera modelos más compactos.
- **Catastrophic forgetting**: si el LR es muy alto durante el finetuning, el modelo puede "olvidar" lo aprendido en ImageNet y converger a una solución peor que el baseline.

## Conclusión

**Para datasets pequeños o medianos en visión, finetuning desde ImageNet es casi siempre la primera elección.** El costo de cómputo es el mismo que entrenar from-scratch (los modelos preentrenados son del mismo tamaño), pero el beneficio en performance es de un orden de magnitud. La augmentation entonces juega un rol secundario: ayudar al modelo finetuned a generalizar mejor cuando el gap train-val empieza a aparecer (epoch 3+ en este experimento).

La Actividad II llevará esta misma lógica al dominio de **texto**, donde el preentrenamiento de modelos como BERT genera saltos aún más dramáticos.
