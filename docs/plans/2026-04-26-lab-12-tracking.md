# Lab 12 — Tracking colaborativo (Colab + local)

**Fecha inicio:** 2026-04-26
**Notebook origen:** `clase_12/material/Laboratorio/Laboratorio 12 - Data Augmentation, Transferencia de conocimiento y Finetuning.ipynb`
**Notebook resolución:** `clase_12/material/Laboratorio/Practico_12_RAE.ipynb`

## Modalidad

- Roberto ejecuta en Colab y entrega outputs/screenshots
- Claude integra outputs en el RAE notebook + site content `site/content/laboratorios/lab-12/`

## Estructura del lab

- **Actividad I — Imágenes** (flowers + ResNet18)
  - Modelo base
  - Data augmentation (visualizaciones + entrenamiento)
  - **Ejercicio I**: VerticalFlipTransform
  - Finetuning desde ImageNet
  - **Ejercicio II.1**: Explicación FC layer
  - **Ejercicio II.2**: Tabla comparativa base / aug / ft en train, val, test
- **Actividad II — Texto** (Jigsaw + BERT)
  - BERT desde cero
  - Finetuning BERT preentrenado
  - **Ejercicio III**: 3 learning rates + 2 preguntas (mejor LR + propuesta data augmentation texto)

## Checklist de avance

### Actividad I

- [ ] Setup + dataset flowers descargado
- [ ] Modelo base entrenado (5 epochs) — outputs: history + test_performance + show_prediction
- [ ] Visualización de transforms (flips, crops, rotations, affine, custom)
- [ ] Modelo aug entrenado (5 epochs) — outputs
- [ ] **Ejercicio I**: VerticalFlipTransform implementado y probado
- [ ] Modelo finetuning entrenado (5 epochs) — outputs
- [ ] **Ejercicio II.1**: Respuesta redactada
- [ ] **Ejercicio II.2**: Tabla comparativa generada

### Actividad II

- [ ] Setup + dataset Jigsaw descargado
- [ ] Modelo BERT base entrenado — outputs (acc, F1 micro, F1 macro, confusion matrix)
- [ ] Modelo BERT finetuning entrenado — outputs
- [ ] **Ejercicio III**: 3 learning rates entrenados (sugerencia: 1e-5, 5e-5, 1e-4)
- [ ] **Ejercicio III**: Pregunta 1 (qué LR elegir) respondida
- [ ] **Ejercicio III**: Pregunta 2 (data augmentation para texto) respondida

### Site content

- [ ] `site/content/laboratorios/lab-12/_index.md`
- [ ] `site/content/laboratorios/lab-12/data-augmentation.md`
- [ ] `site/content/laboratorios/lab-12/transfer-learning.md`
- [ ] `site/content/laboratorios/lab-12/bert-finetuning.md`
- [ ] `site/content/laboratorios/lab-12/ejercicios.md`
- [ ] `site/content/laboratorios/lab-12/resolucion.md`

### Static assets

- [ ] Curvas de pérdida (base, aug, ft) — Actividad I
- [ ] Matrices de confusión BERT base / ft — Actividad II
- [ ] Loss curves Ejercicio III (3 LRs)
- [ ] `site/static/notebooks/lab12.ipynb` y `lab12.html`

## Decisiones

- **Learning rates Ejercicio III:** por confirmar (sugerencia: `1e-5`, `5e-5`, `1e-4`)
- **subset_size BERT:** por confirmar (default `0.01` puede ser muy pequeño para resultados estables)
- **n_epochs:** mantener `5` por defecto salvo que necesitemos ajustar por tiempo de Colab

## Hallazgos / Insights del dataset

- **Dataset flowers = Oxford 102 Flowers** (NO el dataset pequeño de 5 clases)
  - 8,189 imágenes totales → split 5,687 / 1,224 / 1,278 (train / val / test, ~70/15/15)
  - 102 especies → ~80 imágenes promedio por clase
  - Random baseline ≈ 0.98%
  - Hace que el contraste base vs finetuning sea dramático (ideal para el lab)
- **GPU Colab T4** confirmada (`device(type='cuda')`)
- **Tamaño imágenes**: variable (necesitará Resize a 224×224 para ResNet)

## Resultados modelo base (Actividad I)

**Setup:** ResNet18 from scratch, Adam lr=5e-4, batch=128, 5 epochs, normalización ImageNet, Resize 224×224.

### Tabla por epoch

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
| --- | --- | --- | --- | --- |
| 0 (val inicial) | — | — | 5.281 | 1.72% |
| 1 | 3.541 | 16.94% | 3.703 | 14.14% |
| 2 | 2.694 | 30.92% | 3.120 | 22.58% |
| 3 | 2.345 | 37.41% | 2.590 | 32.19% |
| 4 | 2.074 | 43.91% | 2.464 | 34.84% |
| 5 | **1.819** | **49.91%** | **2.166** | **40.55%** |

### Test final

- Test Loss: **2.171**
- Test Acc: **42.58%**
- Val/test gap ≈ 0 (val bien calibrado)

### Assets pendientes (a guardar manualmente)

- [ ] `loss-base.png` — curva loss train vs val
- [ ] `acc-base.png` — curva acc train vs val

### show_prediction sample 166 (modelo base)

- **Ground Truth**: bougainvillea
- **Top-5** (modelo NO acertó):
  1. lotus (p=0.1670)
  2. columbine (p=0.1366)
  3. purple coneflower (p=0.0795)
  4. snapdragon (p=0.0515)
  5. sword lily (p=0.0464)
- Sum top-5 ≈ 48% → modelo muy disperso
- Patrón: las 5 predichas comparten paleta de color con bougainvillea (morados/rosados) → aprendió color pero no estructura
- Comparar después con modelo aug y modelo finetuning para esta misma sample

## Resultados modelo aug (Actividad I)

**Setup:** ResNet18 from scratch (mismo arquitectura), Adam lr=5e-4, batch=128, 5 epochs.
**Transforms train**: rotation(±5°) → resize 256 → random crop 224 → flip(p=0.5) → ToTensor + Normalize ImageNet.
**Transforms val/test**: resize 256 → center crop 224 → ToTensor + Normalize.

### Tabla por epoch (aug)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
| --- | --- | --- | --- | --- |
| 0 (val inicial) | — | — | 5.238 | 0.86% |
| 1 | 3.515 | 16.91% | 3.783 | 15.86% |
| 2 | 2.656 | 31.65% | 2.850 | 26.95% |
| 3 | 2.306 | 37.62% | 2.864 | 25.47% (dip) |
| 4 | 2.044 | 44.46% | 2.195 | 39.14% |
| 5 | **1.833** | **49.46%** | **2.175** | **39.92%** |

### Test final (aug)

- Test Loss: **2.139** (mejor que base 2.171)
- Test Acc: **44.45%** (mejor que base 42.58% por +1.87pp)

### Comparación base vs aug

| Métrica | Base | Aug | Δ |
| --- | --- | --- | --- |
| Train Acc | 49.91% | 49.46% | −0.45pp |
| Val Acc | 40.55% | 39.92% | −0.63pp |
| **Test Acc** | **42.58%** | **44.45%** | **+1.87pp** |

### Observaciones

- Augmentation suave (rot ±5°, flip, crop) → train acc casi igual
- Test mejor que val (44.45 vs 39.92) → patrón curioso, posible varianza del split
- Val tiene "dip" en epoch 3 (25.47% tras 26.95%) → típico con augmentation
- Mejora modesta (+1.87pp test) pero consistente; con más epochs sería mayor

### Assets pendientes (modelo aug)

- [ ] `loss-aug.png`
- [ ] `acc-aug.png`

### show_prediction sample 166 (modelo aug)

- **Ground Truth**: bougainvillea
- **Top-5** (modelo aug NO acertó):
  1. lotus (p=0.2476) — más confiado que base (0.1670)
  2. sweet pea (p=0.0673)
  3. pink primrose (p=0.0489)
  4. mexican aster (p=0.0447)
  5. columbine (p=0.0442)
- Patrón: aug reforzó el sesgo "lotus" — más confianza en la respuesta incorrecta
- Insight: las transforms (flip/crop/rotation ±5°) preservan color → reforzaron la bias por paleta morada/rosa
- Para romper el color bias necesitarías ColorJitter, grayscale, etc. (no usado en este lab)

## Resultados Ejercicio I — modelo ex (VerticalFlipTransform)

**Setup:** `VerticalFlipTransform` custom con `PIL.Image.FLIP_TOP_BOTTOM`, aplicada con `RandomApply(p=0.5)`. Resto idéntico al base (Resize 224, sin otras augmentations).

### Tabla por epoch (ex_model)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
| --- | --- | --- | --- | --- |
| 0 (val inicial) | — | — | 5.742 | 1.25% |
| 1 | 3.492 | 17.71% | 3.341 | 18.36% |
| 2 | 2.653 | 31.70% | 2.756 | 30.31% |
| 3 | 2.329 | 38.18% | 2.690 | 30.08% (dip) |
| 4 | 2.102 | 42.86% | 2.458 | 34.92% |
| 5 | **1.843** | **48.65%** | **2.227** | **39.45%** |

### Test final (ex)

- Test Loss: 2.196 (peor que base 2.171 y aug 2.139)
- Test Acc: **40.55%** (peor que base 42.58% por −2.03pp, peor que aug 44.45% por −3.90pp)

### Comparación final 3 modelos (sin finetuning)

| Modelo | Train Acc | Val Acc | Test Acc | Test Loss |
| --- | --- | --- | --- | --- |
| Base | 49.91% | 40.55% | 42.58% | 2.171 |
| Aug (rot+crop+flip h.) | 49.46% | 39.92% | **44.45%** | **2.139** |
| Ex (V flip) | 48.65% | 39.45% | 40.55% | 2.196 |

### show_prediction sample 166 (ex_model)

- **Ground Truth**: bougainvillea
- **Top-5** (NO acertó):
  1. lotus (p=0.1313)
  2. lenten rose (p=0.1133)
  3. sweet pea (p=0.0782)
  4. cyclamen (p=0.0744)
  5. siam tulip (p=0.0689)
- Los 3 modelos predicen lotus como top-1 → sesgo de color confirmado

### Conclusión Ejercicio I

Vertical flip degrada el modelo porque las flores "boca abajo" no aparecen en val/test → el modelo gasta capacidad aprendiendo patrones espurios. Confirma que **data augmentation debe respetar las invarianzas naturales del dominio**.

## Resultados modelo finetuning (Actividad I)

**Setup:** ResNet18 con pesos `ResNet18_Weights.IMAGENET1K_V1` (ImageNet). Capa `fc` reemplazada por `nn.Linear(512, 102)`. Adam lr=5e-4, batch=128, 5 epochs. Transforms idénticos al modelo base (Resize 224 + Normalize ImageNet, sin augmentation).

### Tabla por epoch (ft)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
| --- | --- | --- | --- | --- |
| 0 (val inicial) | — | — | 4.888 | 0.47% |
| 1 | 1.724 | 63.98% | 0.760 | 80.47% |
| 2 | 0.258 | 94.53% | 0.402 | 87.11% |
| 3 | 0.094 | 97.33% | 0.321 | 87.97% |
| 4 | 0.071 | 97.52% | 0.293 | 88.36% |
| 5 | **0.037** | **98.23%** | **0.274** | **89.14%** |

### Test final (ft)

- Test Loss: **0.300** (¡vs base 2.171!)
- Test Acc: **92.66%** (+50.08pp vs base 42.58%, +48.21pp vs aug 44.45%)

### Comparación FINAL 4 modelos

| Modelo | Train Acc | Val Acc | Test Acc | Test Loss |
| --- | --- | --- | --- | --- |
| Base | 49.91% | 40.55% | 42.58% | 2.171 |
| Aug | 49.46% | 39.92% | 44.45% | 2.139 |
| Ex (V flip) | 48.65% | 39.45% | 40.55% | 2.196 |
| **Finetuning** | **98.23%** | **89.14%** | **92.66%** | **0.300** |

### show_prediction sample 166 (ft_model)

- **Ground Truth**: bougainvillea
- **Top-5** (¡ACERTÓ!):
  1. **bougainvillea (p=0.8566)** ✅
  2. mallow (p=0.0467)
  3. pink primrose (p=0.0121)
  4. pelargonium (p=0.0073)
  5. sweet pea (p=0.0067)
- Suma top-5 ≈ 93% → distribución muy concentrada, modelo confiado
- Las otras 4 alternativas son morfológicamente similares (no solo cromáticamente)
- Confirma que ImageNet enseñó al modelo a "ver estructura"

### Observaciones importantes

- **Val > Train en epoch 1** (80.47% > 63.98%): asimetría por la velocidad del aprendizaje en finetuning (los primeros batches de la epoch arrastran train hacia abajo, val se mide al final con modelo ya muy mejorado)
- **Sobreajuste moderado** desde epoch 3: train 97-98%, val estancada en 88-89% (gap ~9pp)
- **Test > Val** (92.66% vs 89.14%): suerte del split, no hay leak

### Assets pendientes (modelo ft)

- [ ] `loss-ft.png`
- [ ] `acc-ft.png`

## Actividad II — Setup Jigsaw (cells 84-106)

### Dataset Jigsaw (Wikipedia toxic comments)

- 159,571 comentarios totales (Kaggle Jigsaw Multilingual Toxic Comment)
- 6 labels binarias **multi-label** (un comentario puede tener varias o ninguna)
- Estructura: `[toxic, severe_toxic, obscene, threat, insult, identity_hate]`

### Proporción positivos (verificada celda 97)

| Label | % positivos | En train (1117) |
| --- | --- | --- |
| toxic | 9.58% | ~107 |
| severe_toxic | 1.00% | ~11 |
| obscene | 5.29% | ~59 |
| **threat** | **0.30%** | **~3-4** 🚨 |
| insult | 4.94% | ~55 |
| identity_hate | 0.88% | ~10 |

### Hiperparámetros (celda 99)

- `subset_size = 0.01` → 1596 ejemplos totales
- `max_len = 200` (tokens BERT por comentario)
- `batch_size = 16` (BERT consume mucha VRAM)
- `n_epochs = 5`
- `learning_rate = 1e-5` (50× menor que ResNet, BERT necesita lr bajo)
- `tokenizer = bert-base-uncased`

### Splits

- Train: 1,117 (70%)
- Val: 240 (15%)
- Test: 239 (15%)

### Implicación crítica del subset pequeño

- threat tiene solo ~3-4 ejemplos positivos en train → F1 será probablemente 0
- severe_toxic e identity_hate tienen ~10 cada uno → F1 muy ruidoso
- Solo toxic, obscene, insult tienen ejemplos suficientes para aprender bien
- Pendiente decisión: subir `subset_size` a 0.05 si Ejercicio III necesita más estabilidad

## Resultados BERT base (Actividad II — sin preentrenamiento)

**Setup:** `BERTClass(pretrained=False)` — arquitectura BERT-base con pesos aleatorios. Adam lr=1e-5, batch=16, 5 epochs, BCEWithLogitsLoss.

### Tabla por epoch (BERT base)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
| --- | --- | --- | --- | --- |
| 0 (val inicial) | — | — | 0.722 | 34.58% |
| 1 | 0.185 | 95.27% | 0.120 | 97.36% |
| 2 | 0.146 | 96.12% | 0.116 | 97.36% |
| 3 | 0.143 | 96.12% | 0.115 | 97.36% |
| 4 | 0.141 | 96.12% | 0.116 | 97.36% |
| 5 | 0.140 | 96.12% | 0.108 | 97.36% |

### Test final BERT base

- Test Loss: **0.143**
- **Test Acc (exact match): 89.96%** (engañosa por desbalance)
- **Test F1 micro: 0.0000** 🚨
- **Test F1 macro: 0.0000** 🚨

### Matrices de confusión BERT base

Las 6 matrices son **idénticas**: el modelo predice "no" para TODAS las labels en TODOS los ejemplos.

```text
            Predicted
            non-X     X
non-X       1.0000    0.0000
X           1.0000    0.0000
```

→ Clasificador degenerado: solo aprendió la mayoría (predicción constante "no").

### show_prediction sample 566 (BERT base)

- Comment: "and i admit that i'm to sensitive"
- Real labels: todas "no"
- Predicciones: todas "no" con probabilidades 0.01-0.08
- "Acierta" pero solo por la regla degenerada

### Conclusión BERT base

Confirma la moraleja del lab: BERT desde cero con dataset pequeño y desbalanceado **converge a la solución trivial** "siempre predecir mayoría". Accuracy alta (89.96%) es engañosa. F1=0 expone que no aprendió nada útil. El preentrenamiento es esencial.

### Assets pendientes (BERT base)

- [ ] `loss-bert-base.png`
- [ ] `acc-bert-base.png`
- [ ] `cm-bert-base.png` (matrices de confusión 6 labels)

## Resultados BERT finetuning (Actividad II — preentrenado)

**Setup:** `BERTClass(pretrained=True)` — pesos de bert-base-uncased preentrenados (~440 MB). Adam lr=**2e-5** (subido de 1e-5 del base, default canónico para finetuning BERT). batch=16, 5 epochs, BCEWithLogitsLoss.

### Tabla por epoch (BERT ft)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
| --- | --- | --- | --- | --- |
| 0 (val inicial) | — | — | 0.667 | 60.21% |
| 1 | 0.297 | 93.50% | 0.132 | 97.36% |
| 2 | 0.125 | 96.15% | 0.080 | 97.85% |
| 3 | 0.074 | 97.68% | 0.071 | 98.19% |
| 4 | 0.057 | 98.38% | 0.066 | 98.12% |
| 5 | **0.046** | **98.42%** | **0.067** | **98.33%** |

### Test final BERT ft

- Test Loss: **0.065** (vs base 0.143, -55%)
- Test Acc (exact match): **93.31%** (vs base 89.96%, +3.4pp)
- **F1 micro: 0.7708** (vs base 0.0000, +0.77 ⚡)
- **F1 macro: 0.4120** (vs base 0.0000, +0.41 ⚡)

### Matrices de confusión BERT ft

| Label | TN | TP | Veredicto |
| --- | --- | --- | --- |
| toxic | 0.9953 | **0.6250** | Detecta 5/8 |
| severe_toxic | 1.0000 | 0.0000 | Degenerado (~11 train ej.) |
| obscene | 1.0000 | **0.8000** | Casi perfecto |
| threat | 1.0000 | 0.0000 | Degenerado (~3 train ej.) |
| insult | 0.9912 | **0.8333** | Excelente |
| identity_hate | 1.0000 | 0.0000 | Degenerado (~10 train ej.) |

→ Las 3 clases con más ejemplos (toxic, obscene, insult) son detectadas. Las 3 minoritarias siguen sin predicción positiva.

### show_prediction sample 566 (BERT ft)

- Comment: "and i admit that i'm to sensitive"
- Real labels: todas "no"
- Predicciones: todas "no" con probabilidades **0.007-0.014** (vs base 0.008-0.083)
- Mejor calibrado: más certeza en decisiones correctas

### Comparación final BERT base vs ft

| Métrica | Base | Ft | Δ |
| --- | --- | --- | --- |
| Test Acc | 89.96% | 93.31% | +3.4pp |
| F1 micro | 0.0000 | **0.7708** | +0.77 |
| F1 macro | 0.0000 | **0.4120** | +0.41 |

### Conclusión Actividad II

El preentrenamiento de BERT es esencial. BERT desde cero con 1117 ejemplos NO aprende (F1=0); BERT preentrenado y finetuned con los mismos datos alcanza F1 micro 0.77. Las clases minoritarias (threat ~3 ejemplos) siguen sin predicción → falta de datos, no debilidad del modelo.

### Assets pendientes (BERT ft)

- [ ] `loss-bert-ft.png`
- [ ] `acc-bert-ft.png`
- [ ] `cm-bert-ft.png` (matrices de confusión 6 labels)

### Notas

- Resultado mejor de lo esperado: subestimé ResNet18 from scratch
- Anticipo finetuning: ~75-90% (mejora significativa pero no tan dramática como con baseline más débil)
- Overfitting incipiente (gap train-val ~9pp en epoch 5) → motiva data augmentation
