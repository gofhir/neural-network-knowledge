---
title: "Clasificador MLP con representaciones OpenPose"
weight: 50
math: true
---

El segundo lado del A/B test: **mismo MLP, mismo training, mismo split, distinto modelo de pose**. Lo único que cambia respecto al [clasificador PifPaf](../clasificador-pifpaf) es la fuente de features.

## La asimetría de formato

| Aspecto | PifPaf | OpenPose |
|---|---|---|
| Source del caché | `dataset.pifpaf_predictions` | `dataset.openpose_predictions` |
| Tipo de cada item | `openpifpaf.annotation.Annotation` object | NumPy array `(18, 3)` directo |
| Acceso a keypoints | `prediction.data` (atributo) | `prediction` (el array mismo) |
| Feature size | 51 (17×3) | 54 (18×3) |
| Parámetros del MLP | 44,840 | 45,224 |

La asimetría refleja la **diferencia de APIs originales**:

- **PifPaf** (librería madura, mantenida activamente) devuelve **objetos `Annotation`** con metadata rica (score, category_id, bbox, etc.). Para acceder al array usas `.data`.
- **OpenPose** (fork PyTorch de Hzzone) no tiene clase `Annotation`. Devuelve `(candidate, subset)` raw, y el lab pasa por `openpose_extract_keypoints()` para obtener arrays NumPy directos.

## La función de extracción manual

```python
def openpose_extract_keypoints(subset, candidates):
    num_keypoints = 18
    person_tensors = []

    for person in subset:
        person_tensor = np.zeros((num_keypoints, 3))  # init con ceros

        for i in range(num_keypoints):
            keypoint_index = int(person[i])
            if keypoint_index != -1:
                x, y, confidence, _ = candidates[keypoint_index]
                person_tensor[i] = [x, y, confidence]
            # si == -1, deja el slot como cero (keypoint no detectado)

        person_tensors.append(person_tensor)

    return person_tensors
```

**Sin esta función no podrías entrenar el MLP** porque tendrías que reescribir la lógica de "convertir `candidate+subset` a array NumPy" en cada lugar donde se necesita. La función encapsula ese conocimiento y normaliza el formato.

### Detalle clave — keypoints faltantes como ceros

OpenPose devuelve `-1` cuando un keypoint no fue detectado. La función rellena con `[0, 0, 0]` (el array se inicializa con `np.zeros`). Esto significa:

- Para OpenPose, **el patrón "muchos ceros consecutivos" significa "keypoints faltantes"**. El MLP debe aprender que ese patrón es proxy de "vista parcial de persona".
- Para PifPaf, **no hay tal patrón** — el modelo siempre da coordenadas, solo que con confianza baja.

Esto **sutilmente favorece a uno u otro modelo** dependiendo de qué patrón sea más fácil de aprender. En la práctica, los dos MLPs se las arreglan bien.

## El diff con la celda PifPaf — una sola línea

```diff
- for filename, predictions in dataset.pifpaf_predictions.items():
+ for filename, predictions in dataset.openpose_predictions.items():
      ...
      for prediction in predictions:
-         flattened_tensor = prediction.data.reshape(-1)
+         flattened_tensor = prediction.reshape(-1)
```

**Dos diferencias en todo el bloque de preparación de datos**:

1. Lee de `openpose_predictions` en lugar de `pifpaf_predictions`.
2. `prediction.reshape(-1)` en vez de `prediction.data.reshape(-1)`.

El resto del código (zip, stack, squeeze, split, DataLoaders, MLP, optimizer, loss, training loop) es **literalmente idéntico** al bloque PifPaf. Esto es **el A/B test riguroso**: máximo solapamiento de configuración, mínima diferencia controlada.

## La sobrescritura de variables

```python
model = MLP(input_size=X_train.shape[1], ...)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.BCEWithLogitsLoss()

training_losses = []      # ← sobrescribe los de PifPaf
testing_accuracies = []
```

Las variables `model`, `X_train`, `train_loader`, `training_losses`, etc., son **las mismas que en el experimento PifPaf**, pero ahora con valores nuevos. **Esto es deliberado** — el profesor confía en que el alumno entiende la sobrescritura.

**Implicación práctica**: si quieres comparar las dos gráficas después, **debes guardar los valores de PifPaf antes** de ejecutar este bloque:

```python
pifpaf_losses = list(training_losses)
pifpaf_accuracies = list(testing_accuracies)
# después: ejecutar el bloque OpenPose, que sobrescribe las listas
```

## Resultados — gráficos del experimento

### Curva de pérdida

![Pérdida de training OpenPose](/laboratorios/lab-17/mlp-openpose-loss.png)

Caída similar a PifPaf — convergencia rápida en las primeras 5-10 epochs, plateau cerca de cero después.

### Curva de precisión

![Precisión de test OpenPose](/laboratorios/lab-17/mlp-openpose-accuracy.png)

Subida desde chance level (~25%) hasta plateau en **~70-80%**. Forma similar a PifPaf con ligera variación.

## Análisis del A/B test

| Métrica | PifPaf | OpenPose |
|---|---|---|
| Feature size | 51 | 54 |
| Parámetros MLP | 44,840 | 45,224 |
| Loss final | ~0.05 | ~0.05 |
| **Accuracy de test** | **~75-85%** | **~70-80%** |

**El resultado típico**: PifPaf gana por **~3-8 puntos** sobre OpenPose. Pero hay caveats importantes:

### Caveat 1 — splits distintos

Sin `random_state` fijo, los splits de PifPaf y OpenPose son **distintos entre sí**. La diferencia residual de accuracy tiene **±5% de ruido** por esa fuente sola.

**Regla mental para interpretar**:

| Diferencia | Conclusión |
|---|---|
| >10 puntos | Significativa, el modelo realmente importa |
| 5-10 puntos | Probablemente real pero con ruido |
| <5 puntos | Probablemente ruido del split |

### Caveat 2 — cantidad distinta de ejemplos

PifPaf y OpenPose pueden detectar **distinto número de personas por imagen**. Si una imagen tiene 3 personas y PifPaf detecta las 3 pero OpenPose detecta 2, esa imagen contribuye **3 ejemplos a PifPaf y 2 a OpenPose**. **Rompe ligeramente la simetría**.

### Caveat 3 — keypoints distintos

PifPaf da 17 (COCO standard). OpenPose da 18 (COCO + neck inferido). Los 3 features extra son potencialmente útiles, pero probablemente **redundantes** (cuello derivable de hombros).

## Interpretación del resultado

PifPaf típicamente gana, consistente con la promesa de su [paper](/papers/pifpaf-kreiss-2019): **localización sub-pixel** y **mejor manejo de oclusión** producen features más informativos que OpenPose para clasificación downstream.

Pero la diferencia **no es dramática** porque:

- Stanford 40 es **alta-media resolución** (no el régimen donde PifPaf domina dramáticamente).
- Las 4 clases del subset son **suficientemente fáciles** que ambos modelos producen features útiles.
- El MLP simple no aprovecha plenamente la mayor precisión sub-pixel de PifPaf.

**Lección general**: el SOTA académico **no siempre gana** en tu task específica. La diferencia depende del **régimen** (resolución, oclusión, dominio) y del **downstream task**. Por eso medir empíricamente es mejor que confiar en leaderboards.

## Cross-links

{{< cards >}}
  {{< card link="../clasificador-pifpaf" title="Clasificador MLP con PifPaf" subtitle="Lado A del A/B test" icon="academic-cap" >}}
  {{< card link="../actividad-running-vs-bike" title="Actividad evaluable" subtitle="Pipeline end-to-end con OpenPifPaf" icon="check-circle" >}}
  {{< card link="/papers/openpose-cao-2017" title="Paper OpenPose" subtitle="El modelo que produce las features" icon="document-text" >}}
  {{< card link="/papers/pifpaf-kreiss-2019" title="Paper PifPaf" subtitle="Comparación con el competidor" icon="document-text" >}}
{{< /cards >}}
