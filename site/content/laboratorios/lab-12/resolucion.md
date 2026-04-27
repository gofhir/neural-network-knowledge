---
title: "Resolución del Laboratorio"
weight: 50
math: true
---

Resolución completa de los 3 ejercicios prácticos del laboratorio.

{{< notebook-viewer src="/notebooks-html/lab12.html" >}}

---

## Ejercicio I — Vertical Flip Transform

### Implementación

```python
class VerticalFlipTransform:
    def __init__(self):
        pass

    def __call__(self, img):
        # Refleja la imagen verticalmente (lo que arriba va abajo)
        return img.transpose(Image.FLIP_TOP_BOTTOM)
```

Implementación mínima usando `PIL.Image.transpose` con la constante `Image.FLIP_TOP_BOTTOM`. La transformación se aplica con `RandomApply(p=0.5)` durante el entrenamiento.

### Visualización de la transformación

![Vertical flip aplicado a samples](/laboratorios/lab-12/aug-vertical-flip.png)

Las flores aparecen efectivamente "boca abajo" — los pétalos hacia arriba pasan a estar hacia abajo. Visualmente la imagen sigue siendo "una flor", pero su orientación natural se ha invertido.

### Resultados del modelo `ex_model`

Con la nueva transformación durante el entrenamiento:

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 0 (val inicial) | — | — | 5.742 | 1.25% |
| 1 | 3.492 | 17.71% | 3.341 | 18.36% |
| 2 | 2.653 | 31.70% | 2.756 | 30.31% |
| 3 | 2.329 | 38.18% | 2.690 | 30.08% (dip) |
| 4 | 2.102 | 42.86% | 2.458 | 34.92% |
| 5 | **1.843** | **48.65%** | **2.227** | **39.45%** |

**Test final:** Loss = 2.196, Acc = **40.55%**.

![Loss ex_model](/laboratorios/lab-12/loss-ex.png)
![Accuracy ex_model](/laboratorios/lab-12/acc-ex.png)

### Comparación con baseline y modelo aug

| Métrica | Modelo base | Modelo aug | **Modelo ex (V flip)** |
|---|---|---|---|
| Train Acc | 49.91% | 49.46% | 48.65% |
| Val Acc | 40.55% | 39.92% | 39.45% |
| **Test Acc** | **42.58%** | **44.45%** | **40.55%** |

El modelo entrenado con vertical flip obtuvo **el peor rendimiento en test**:

- **vs modelo base**: −2.03pp (42.58% → 40.55%)
- **vs modelo aug**: −3.90pp (44.45% → 40.55%)

### Análisis

El resultado confirma que **no toda transformación de data augmentation ayuda** — algunas pueden ser contraproducentes. La razón es semántica: vertical flip introduce una invarianza que **no existe** en el dominio. En el mundo real:

- Las fotos de flores se toman típicamente **con la flor hacia arriba** (mostrando la corola, los pétalos abriéndose hacia el cielo).
- Una flor "boca abajo" es virtualmente inexistente en val/test.
- Al exponer el modelo a flores invertidas durante el train, el modelo gasta capacidad aprendiendo features para una orientación que **nunca va a ver** en evaluación.

Comparando con horizontal flip (que sí ayuda en el modelo aug, +1.87pp): horizontal flip preserva la simetría natural del objeto (una flor reflejada horizontalmente sigue siendo plausible), mientras que vertical flip rompe esa estructura.

**Lección general:** la augmentación debe **respetar las invarianzas del dominio**. Si el dominio tiene una orientación canónica (texto, dígitos, flores en pose natural, edificios verticales), introducir vertical flip degrada el modelo en lugar de mejorarlo. Las invarianzas válidas dependen de la tarea — un dataset de imágenes satelitales sí podría beneficiarse de vertical flip, porque desde arriba no hay orientación canónica.

### Predicción cualitativa para sample 166

```text
Ground Truth: bougainvillea

Top-5 Predictions (modelo ex_model)
  1. lotus              (p=0.1313)
  2. lenten rose        (p=0.1133)
  3. sweet pea          (p=0.0782)
  4. cyclamen           (p=0.0744)
  5. siam tulip         (p=0.0689)
```

Los 3 modelos sin finetuning (base, aug, ex) predicen `lotus` como top-1 → confirma que el sesgo dominante es **cromático**, y ninguna combinación de transformaciones espaciales (horizontal flip, vertical flip, crop, rotation) lo soluciona. Para romper ese sesgo se necesitarían transformaciones de color: `ColorJitter`, `Grayscale`, `RandomEqualize`.

---

## Ejercicio II — Finetuning sobre flowers

### Parte 1 — Explicación de la línea clave

```python
ft_model.fc = nn.Linear(in_features=512, out_features=n_classes, bias=True)
```

Esta línea **reemplaza la capa fully-connected final** (`fc`) del modelo ResNet18 preentrenado por una nueva capa lineal adaptada a la tarea actual. La razón es que el modelo descargado con `pretrained=True` viene entrenado sobre **ImageNet**, que tiene 1000 clases — por lo tanto su capa `fc` original tiene `out_features=1000`. Como nuestro problema (clasificación de flores) tiene `n_classes=102`, esa capa no sirve y se reemplaza por una `Linear(512, 102)`.

Las **512 features de entrada** corresponden a la salida del backbone convolucional de ResNet18 (después del `avgpool`). El backbone se mantiene intacto con los pesos de ImageNet, mientras que la nueva capa final se inicializa aleatoriamente. Durante el entrenamiento, ambos se actualizan simultáneamente con backpropagation: el backbone afina sus features para flowers, y la cabeza nueva aprende a mapear esas features a las 102 clases.

Si **no** se reemplazara la capa, el `forward` lanzaría un error dimensional al calcular la pérdida (logits de 1000 vs labels de 102). Si se reemplazara pero no se cargaran los pesos preentrenados, no habría transfer learning — equivaldría al modelo base from-scratch.

### Parte 2 — Tabla comparativa de los tres escenarios

Resultados después de 5 épocas de entrenamiento, misma arquitectura ResNet18, mismo optimizador (Adam con `lr=5e-4`), mismo `batch_size=128` y misma función de pérdida (`CrossEntropyLoss`):

| Escenario | Train Loss | Train Acc | Val Loss | Val Acc | Test Loss | Test Acc |
|---|---|---|---|---|---|---|
| **Modelo base** (from scratch) | 1.819 | 49.91% | 2.166 | 40.55% | 2.171 | 42.58% |
| **Data augmentation** (rot ±5°, crop, flip h.) | 1.833 | 49.46% | 2.175 | 39.92% | 2.139 | 44.45% |
| **Finetuning** (ImageNet pretrained) | **0.037** | **98.23%** | **0.274** | **89.14%** | **0.300** | **92.66%** |

#### ¿Cuál escenario logró el mejor rendimiento?

El **finetuning** desde pesos preentrenados en ImageNet ganó por un margen abrumador:

- **+50.08pp** sobre el modelo base (42.58% → 92.66%)
- **+48.21pp** sobre el modelo con data augmentation (44.45% → 92.66%)
- Test loss de **0.300** vs **2.171** del baseline (~7× menor)

#### ¿Por qué?

Tres razones técnicas:

1. **Inicialización informada vs aleatoria.** Los pesos de ImageNet ponen al modelo en una región del espacio de parámetros que ya extrae features genéricas (bordes, texturas, patrones). Inicialización aleatoria parte de un mínimo terrible, y con solo ~5,700 imágenes el modelo nunca puede compensar el déficit en 5 épocas.

2. **Capas tempranas son universales.** Como mostró Yosinski et al. (2014), las primeras capas convolucionales aprenden features que son intercambiables entre datasets de visión. ImageNet aprendió esas features sobre 1.2M imágenes — no hay forma de replicar esa señal con un dataset 200× más chico.

3. **Las clases de ImageNet incluyen plantas y flores.** La transferencia al dominio flowers es prácticamente directa: el feature extractor ya "vio" pétalos y hojas en preentrenamiento.

#### Relación con lo visto en clase

Los resultados validan empíricamente las dos técnicas vistas:

- **Data augmentation** funciona pero su efecto es **modesto** (+1.87pp). Es una herramienta de **regularización**: reduce overfitting al exponer al modelo a variantes plausibles de los samples de train. No aporta información nueva — solo ayuda a que la información existente se generalice mejor.
- **Transfer learning / finetuning** transfiere **conocimiento real** de un modelo entrenado en un dataset masivo. El efecto es de **otro orden de magnitud** (+50pp) porque resuelve el problema fundamental: la falta de datos para aprender features visuales desde cero.

La combinación natural es: **finetuning como prerequisito + augmentation como refinamiento**. En este lab vemos finetuning sin aug y aug sin finetuning, pero en la práctica conviene aplicar ambos. La augmentation sería especialmente útil para reducir el gap train-val que aparece desde epoch 3 (97-98% en train vs 88-89% en val).

---

## Ejercicio III — Learning rate y BERT

### Resumen comparativo (output del notebook)

```text
======================================================================
  RESUMEN COMPARATIVO — Ejercicio III
======================================================================

LR                    Acc     F1 micro     F1 macro
--------------------------------------------------
2e-5 (def)          93.31       0.7708       0.4120  ← ft_model anterior
5e-06             91.6318       0.6588       0.3377
5e-05             90.7950       0.7778       0.4178
0.0001            90.3766       0.7290       0.3884
```

| LR | Acc (%) | F1 micro | F1 macro | Test Loss |
|---|---|---|---|---|
| 5e-6 | 91.63 | 0.6588 | 0.3377 | 0.103 |
| **2e-5** (default) | **93.31** | 0.7708 | 0.4120 | 0.065 |
| **5e-5** | 90.79 | **0.7778** ⭐ | **0.4178** ⭐ | **0.048** ⭐ |
| 1e-4 | 90.38 | 0.7290 | 0.3884 | 0.070 |

### Análisis del entrenamiento — curvas por LR

#### `lr = 5e-6` (muy bajo)

![Loss lr=5e-6](/laboratorios/lab-12/loss-lr5e-6.png)
![Accuracy lr=5e-6](/laboratorios/lab-12/acc-lr5e-6.png)

Descenso muy lento y suave. La pérdida sigue bajando en epoch 5 sin haber convergido — el modelo está **subentrenado** dado el presupuesto de 5 épocas. La accuracy de train arranca en 83% (vs 93-95% con los otros LRs en epoch 1) y converge gradualmente. Si tuviéramos 15-20 épocas probablemente alcanzaría niveles similares al `2e-5`, pero con presupuesto fijo no aprovecha el preentrenamiento.

#### `lr = 5e-5` (alto-pero-razonable)

![Loss lr=5e-5](/laboratorios/lab-12/loss-lr5e-5.png)
![Accuracy lr=5e-5](/laboratorios/lab-12/acc-lr5e-5.png)

Caída brusca de la pérdida en epoch 1 (val ~0.09 vs ~0.13 del default), luego convergencia suave y estable. Sin oscilaciones. Es el comportamiento ideal para finetuning: aprovechar al máximo el preentrenamiento aprendiendo rápido en epoch 1 y refinando los detalles en las siguientes.

#### `lr = 1e-4` (muy alto)

![Loss lr=1e-4](/laboratorios/lab-12/loss-lr1e-4.png)
![Accuracy lr=1e-4](/laboratorios/lab-12/acc-lr1e-4.png)

Comportamiento sorprendentemente estable, sin spikes de loss ni divergencia. Convergencia incluso más rápida que con `5e-5` (epoch 2-3). Train y val loss casi idénticos al final → **no hay catastrophic forgetting visible**. Esto fue una sorpresa pedagógica: la hipótesis era que `1e-4` rompería el preentrenamiento, pero Adam (con su LR adaptativo por parámetro) y el dataset chico hicieron que el modelo aguantara.

### Análisis de F1 macro — patrón en U invertida

```text
F1 macro
   ↑
   │              ⭐ 5e-5 (0.4178)
   │      ●            ●
   │ ●                   
   │                  ● 1e-4 (0.3884)
   │ ● 2e-5 (0.4120)
   │5e-6 (0.3377)
   └──────────────────→ LR (escala log)
   5e-6   2e-5   5e-5   1e-4
```

El F1 macro sigue una **curva en U invertida**: sube de `5e-6` a `5e-5` y luego baja en `1e-4`. El óptimo en este experimento es **`5e-5`**, no el `2e-5` del enunciado del lab. Este patrón es consistente con la literatura de finetuning de transformers ([Sun et al. 2019](https://arxiv.org/abs/1905.05583), [Mosbach et al. 2021](https://arxiv.org/abs/2006.04884)), que recomienda un rango `[2e-5, 5e-5]` para BERT.

### Análisis de matrices de confusión

#### `lr = 5e-6`

![Matrices de confusión lr=5e-6](/laboratorios/lab-12/cm-lr5e-6.png)

| Label | TP rate | vs default (2e-5) |
|---|---|---|
| toxic | 50.0% | −12.5pp |
| severe_toxic | 0% | igual (degenerado) |
| obscene | 80.0% | igual |
| threat | 0% | igual (degenerado) |
| insult | 33.3% | **−50pp** ← colapso |
| identity_hate | 0% | igual (degenerado) |

LR demasiado bajo → recall de `insult` colapsa a 33% (vs 83% con default). Confirma subentrenamiento.

#### `lr = 5e-5`

![Matrices de confusión lr=5e-5](/laboratorios/lab-12/cm-lr5e-5.png)

| Label | TP rate | vs default (2e-5) |
|---|---|---|
| toxic | **79.2%** | **+16.7pp** ⚡ |
| severe_toxic | 0% | igual (degenerado) |
| obscene | **86.7%** | +6.7pp |
| threat | 0% | igual (degenerado) |
| insult | 83.3% | igual |
| identity_hate | 0% | igual (degenerado) |

LR óptimo → mejor recall en `toxic` y `obscene`. Las 3 clases minoritarias siguen degeneradas (problema de datos, no de LR).

#### `lr = 1e-4`

![Matrices de confusión lr=1e-4](/laboratorios/lab-12/cm-lr1e-4.png)

| Label | TP rate | vs lr=5e-5 |
|---|---|---|
| toxic | 75.0% | −4pp |
| severe_toxic | 0% | igual |
| obscene | 80.0% | −7pp |
| threat | 0% | igual |
| insult | 75.0% | −8pp |
| identity_hate | 0% | igual |

LR muy alto → degradación sutil en las 3 clases mayoritarias por **overshoot** (pasos demasiado grandes alrededor del mínimo de cada clase minoritaria). No es catastrophic forgetting — es pérdida de precisión fina.

### Observación crítica — el problema persiste

**Tres clases mantienen 0% de recall en TODAS las configuraciones** (`severe_toxic`, `threat`, `identity_hate`). Esto demuestra que **el LR no resuelve el problema fundamental**: el desbalance extremo del dataset (con `subset_size=0.01`, esas clases tienen menos de 15 positivos en train). Soluciones reales serían:

1. **`pos_weight` en `BCEWithLogitsLoss`** — ponderar positivos por la inversa de su frecuencia.
2. **Focal Loss** ([Lin et al. 2017](https://arxiv.org/abs/1708.02002)) — penalizar fuerte los falsos negativos en minoritarias.
3. **Oversampling** de positivos minoritarios (`WeightedRandomSampler`).
4. **Threshold tuning por clase** — no usar 0.5 como umbral universal.
5. **Más datos** — subir `subset_size` de 0.01 a 0.05+ multiplica los positivos.

### Trade-off Acc vs F1 — por qué 5e-5 baja accuracy pero sube F1

| Métrica | `2e-5` | `5e-5` | Δ |
|---|---|---|---|
| Acc (exact-match) | **93.31%** | 90.79% | −2.5pp |
| F1 micro | 0.7708 | **0.7778** | +0.007 |
| F1 macro | 0.4120 | **0.4178** | +0.006 |

Aparente paradoja: el LR mejor en F1 es peor en accuracy. La explicación es el **trade-off precision/recall**:

- Con `5e-5` el modelo predice **más positivos** en `toxic` (TP rate 79% vs 62% del default).
- Esto introduce algunos **falsos positivos**: comentarios cuyas 6 etiquetas eran "no, no, no..." ahora se predicen mal en alguna.
- En **exact-match accuracy** (que requiere que las 6 predicciones sean correctas) cualquier FP rompe el match → la métrica baja.
- En **F1** se valora el balance: subir TP es positivo, FP es negativo, pero el balance neto mejora.

**F1 es la métrica honesta** en problemas multi-label con desbalance. Accuracy exact-match es engañosa: un modelo que predice "todo no" obtendría ~98% en una clase con 2% de positivos, pero F1=0 lo delata como degenerado.

### Pregunta 1 — ¿Con qué learning rate utilizaría?

> **Usaría `lr = 5e-5`.** Es el que obtuvo el mejor **F1 micro (0.7778)** y **F1 macro (0.4178)** en test, además de la menor *test loss* (0.0484). Aunque su exact-match accuracy (90.79%) es 2.5pp menor que con `2e-5`, en un problema multi-label con clases desbalanceadas el F1 refleja mejor la capacidad real del modelo de detectar las clases minoritarias (ej. `toxic` sube de 62.5% → 79.2% de recall). Con `5e-6` el modelo queda subentrenado y con `1e-4` aparece *overshoot* que degrada las clases minoritarias.

### Pregunta 2 — Propuesta de data augmentation para texto

> **Propondría back-translation aplicada selectivamente a las clases minoritarias** (`severe_toxic`, `threat`, `identity_hate`). Para cada comentario positivo de esas clases, se traduce a otro idioma (por ejemplo inglés → francés → inglés, o → alemán → inglés) usando un modelo de traducción automática. El texto resultante preserva el significado y las etiquetas, pero cambia el vocabulario y la estructura sintáctica, lo que aumenta la diversidad léxica que ve BERT durante el fine-tuning.
>
> Esto ataca directamente el problema real que detectamos en el Ejercicio III: las tres clases con 0% de recall están subrepresentadas, y ningún `learning rate` lo arregla. Multiplicar 3-5x los positivos minoritarios vía back-translation balancearía el dataset sin sintetizar etiquetas falsas (a diferencia de `synonym replacement` ingenuo, que puede romper el contenido tóxico al reemplazar justo el insulto). Como bonus, es una forma de regularización: el modelo aprende que la toxicidad no depende de palabras exactas sino del contenido semántico.

---

## Conclusiones generales del laboratorio

1. **Transfer learning domina sobre data augmentation** cuando el dataset target es chico. En imágenes el salto es +50pp; en texto pasamos de F1=0 a F1=0.77.

2. **No toda augmentación ayuda** — debe respetar las invarianzas del dominio. Vertical flip degradó el modelo de flowers porque las flores boca abajo no son una variante natural.

3. **El learning rate óptimo de finetuning sigue una curva en U invertida** — `5e-5` ganó en F1 sobre el default `2e-5`. Demasiado bajo subentrena; demasiado alto introduce overshoot.

4. **Accuracy exact-match es engañosa en problemas multi-label desbalanceados**. F1 macro es la métrica honesta porque expone clases degeneradas.

5. **El LR no resuelve problemas de datos**. Las 3 clases minoritarias del Jigsaw quedaron en 0% de recall en todas las configuraciones — solucionarlo requiere `pos_weight`, focal loss, oversampling o más datos, no más experimentación con LRs.
