---
title: "BERT Finetuning sobre Jigsaw"
weight: 30
math: true
---

## El problema — Jigsaw Toxic Comments

La Actividad II aplica transfer learning al dominio de **texto**, usando el dataset [Jigsaw Multilingual Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). Son comentarios de Wikipedia anotados con 6 etiquetas binarias **multi-label** (un comentario puede tener varias o ninguna):

```text
[toxic, severe_toxic, obscene, threat, insult, identity_hate]
```

Es un problema **multi-label** (no multi-clase): cada comentario puede tener cualquier subconjunto de las 6 etiquetas activadas. Esto cambia tres cosas respecto a clasificación de imágenes:

1. **La capa de salida** tiene 6 logits (uno por etiqueta) sin softmax — cada uno es independiente.
2. **La función de pérdida** es `BCEWithLogitsLoss` (binary cross-entropy aplicada a cada logit), no `CrossEntropyLoss`.
3. **Las métricas** deben separarse por etiqueta o agregarse de forma multi-label (F1 micro/macro), no exact-match.

## El desbalance — la dificultad real del dataset

El dataset completo (159,571 comentarios) tiene una distribución muy desbalanceada:

| Label | % positivos | En train (n=1,117 con `subset_size=0.01`) |
|---|---|---|
| `toxic` | 9.58% | ~107 |
| `severe_toxic` | 1.00% | ~11 |
| `obscene` | 5.29% | ~59 |
| `threat` | **0.30%** | **~3-4** 🚨 |
| `insult` | 4.94% | ~55 |
| `identity_hate` | 0.88% | ~10 |

Con `subset_size=0.01` (que es lo que usa el lab para que entrene rápido en Colab), tres clases (`severe_toxic`, `threat`, `identity_hate`) tienen **menos de 15 ejemplos positivos** en train. Es virtualmente imposible que el modelo aprenda a detectarlas: si predice "todo no" para esas tres clases, ya acierta el 99% del tiempo en cada una. Cualquier intento de aprenderlas requiere un esfuerzo desproporcionado del optimizador.

**Esta limitación va a dominar todos los resultados** que veremos en Actividad II y Ejercicio III.

### Splits

```text
Train:  1,117  (70%)
Val:      240  (15%)
Test:     239  (15%)
```

### Hiperparámetros del lab

```python
subset_size  = 0.01      # 1,596 ejemplos totales
max_len      = 200       # tokens BERT por comentario
batch_size   = 16        # BERT consume mucha VRAM
n_epochs     = 5
learning_rate = 1e-5     # 50× menor que ResNet — BERT necesita LR bajo
tokenizer    = bert-base-uncased
```

## La arquitectura — `BERTClass`

```python
class BERTClass(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        else:
            config = PretrainedConfig.from_pretrained('bert-base-uncased')
            self.l1 = transformers.BertModel(config)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 6)

    def forward(self, ids, mask, token_type_ids):
        output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids).pooler_output
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output
```

Tres componentes:

- **`l1`**: BERT-base-uncased (12 capas transformer, 768 dim, 110M parámetros). El flag `pretrained` decide si se cargan los pesos de Hugging Face o se inicializa aleatoriamente con la misma arquitectura.
- **`l2`**: Dropout 0.3 sobre el pooler output (representación del token `[CLS]`).
- **`l3`**: capa final `Linear(768, 6)` que produce 6 logits, uno por etiqueta. Sin sigmoid — `BCEWithLogitsLoss` lo aplica internamente para mejor estabilidad numérica.

El método `forward` toma los tensores que produce el tokenizer (`input_ids`, `attention_mask`, `token_type_ids`) y devuelve los logits crudos.

## Modelo BERT desde cero (`pretrained=False`)

Primero entrenamos BERT con **arquitectura idéntica pero pesos aleatorios** — para ver qué tan importante es el preentrenamiento.

### Resultados BERT base

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 0 (val inicial) | — | — | 0.722 | 34.58% |
| 1 | 0.185 | 95.27% | 0.120 | 97.36% |
| 2 | 0.146 | 96.12% | 0.116 | 97.36% |
| 3 | 0.143 | 96.12% | 0.115 | 97.36% |
| 4 | 0.141 | 96.12% | 0.116 | 97.36% |
| 5 | 0.140 | 96.12% | 0.108 | 97.36% |

![Loss BERT base](/laboratorios/lab-12/loss-bert-base.png)
![Accuracy BERT base](/laboratorios/lab-12/acc-bert-base.png)

```text
Test loss: 0.143
Test acc (exact match): 89.96%   ← engañoso
F1 micro: 0.0000  🚨
F1 macro: 0.0000  🚨
```

### Las matrices de confusión revelan el desastre

![Matrices de confusión BERT base](/laboratorios/lab-12/cm-bert-base.png)

Las **6 matrices son idénticas**: el modelo predice "no" para **todas** las etiquetas en **todos** los ejemplos. Es un clasificador degenerado: la solución trivial "siempre predecir mayoría".

| Label | TN rate | TP rate |
|---|---|---|
| toxic | 1.0000 | 0.0000 |
| severe_toxic | 1.0000 | 0.0000 |
| obscene | 1.0000 | 0.0000 |
| threat | 1.0000 | 0.0000 |
| insult | 1.0000 | 0.0000 |
| identity_hate | 1.0000 | 0.0000 |

### Predicción cualitativa (sample 566)

```text
Comment: "and i admit that i'm to sensitive"
Real labels: todas no

Task             Pred           Real
toxic            no (p=0.083)   no
severe_toxic     no (p=0.012)   no
obscene          no (p=0.052)   no
threat           no (p=0.008)   no
insult           no (p=0.048)   no
identity_hate    no (p=0.012)   no
```

"Acierta" la predicción, pero solo porque el comentario es benigno y el modelo predice "no" para **todo**. Las probabilidades 0.008 - 0.083 muestran un modelo apenas entrenado, sin información discriminativa.

### La accuracy 89.96% es engañosa

89.96% suena alto, pero es **exact-match accuracy**: el modelo "acierta" cuando predice los 6 labels correctamente. Como la mayoría de los comentarios tienen los 6 labels en cero, predecir "todo no" acierta automáticamente esos casos. Las métricas honestas son:

- **F1 micro = 0**: ni un solo TP en todo el test set sumando las 6 etiquetas.
- **F1 macro = 0**: idem, promediando por etiqueta.

**Conclusión BERT base**: BERT desde cero con 1,117 ejemplos no aprende nada. Confirma que el preentrenamiento es esencial para transformers grandes.

## Modelo BERT preentrenado + finetuning (`pretrained=True`)

Ahora se carga BERT con los pesos de Hugging Face (~440 MB, descargados en epoch 1) y se entrena con un LR ligeramente más alto (`2e-5` en lugar de `1e-5`, que es el default canónico para finetuning de BERT según el paper original).

### Resultados BERT ft

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 0 (val inicial) | — | — | 0.667 | 60.21% |
| 1 | 0.297 | 93.50% | 0.132 | 97.36% |
| 2 | 0.125 | 96.15% | 0.080 | 97.85% |
| 3 | 0.074 | 97.68% | 0.071 | 98.19% |
| 4 | 0.057 | 98.38% | 0.066 | 98.12% |
| 5 | **0.046** | **98.42%** | **0.067** | **98.33%** |

![Loss BERT ft](/laboratorios/lab-12/loss-bert-ft.png)
![Accuracy BERT ft](/laboratorios/lab-12/acc-bert-ft.png)

```text
Test loss: 0.065      (vs base 0.143, −55%)
Test acc (exact match): 93.31%   (vs base 89.96%, +3.4pp)
F1 micro: 0.7708      ⚡  (vs base 0.0000)
F1 macro: 0.4120      ⚡  (vs base 0.0000)
```

### Matrices de confusión BERT ft

![Matrices de confusión BERT ft](/laboratorios/lab-12/cm-bert-ft.png)

| Label | TN rate | TP rate | Veredicto |
|---|---|---|---|
| `toxic` | 0.9953 | **0.6250** | Detecta 5/8 positivos |
| `severe_toxic` | 1.0000 | 0.0000 | Degenerado (~11 train ej.) |
| `obscene` | 1.0000 | **0.8000** | Casi perfecto |
| `threat` | 1.0000 | 0.0000 | Degenerado (~3 train ej.) |
| `insult` | 0.9912 | **0.8333** | Excelente |
| `identity_hate` | 1.0000 | 0.0000 | Degenerado (~10 train ej.) |

Las **3 clases con suficientes ejemplos** (`toxic`, `obscene`, `insult`) son detectadas con altas tasas de TP. Las **3 clases minoritarias** (`severe_toxic`, `threat`, `identity_hate`) siguen degeneradas — el modelo no logra superar la barrera del desbalance ni con preentrenamiento.

### Predicción cualitativa con ft (sample 566)

```text
Comment: "and i admit that i'm to sensitive"
Real labels: todas no

Task             Pred           Real
toxic            no (p=0.014)   no
severe_toxic     no (p=0.008)   no
obscene          no (p=0.008)   no
threat           no (p=0.007)   no
insult           no (p=0.008)   no
identity_hate    no (p=0.009)   no
```

Probabilidades 0.007 - 0.014: **mejor calibrado** que el base (0.008 - 0.083). El modelo finetuned está **más confiado** en sus predicciones correctas.

## Comparación final BERT

| Métrica | Base | Finetuning | Δ |
|---|---|---|---|
| Test Loss | 0.143 | 0.065 | −55% |
| Test Acc (exact) | 89.96% | 93.31% | +3.4pp |
| **F1 micro** | **0.0000** | **0.7708** | **+0.77** ⚡ |
| **F1 macro** | **0.0000** | **0.4120** | **+0.41** ⚡ |
| Etiquetas detectadas | 0/6 | 3/6 | +3 |

El salto de F1 macro de **0** a **0.41** confirma que BERT preentrenado **realmente aprendió a discriminar**, mientras que BERT desde cero solo encontró la solución trivial.

## Por qué este patrón se parece al de imágenes

El paralelo con Actividad I es directo:

| Imágenes (ResNet18) | Texto (BERT) |
|---|---|
| Modelo from-scratch: 42.58% test acc | BERT from-scratch: F1=0 |
| Finetuning: 92.66% test acc (+50pp) | BERT ft: F1=0.77 (+0.77) |
| ImageNet aporta features visuales generales | BERT preentrenado aporta representaciones lingüísticas |
| Capas iniciales detectan bordes/texturas | Capas iniciales capturan sintaxis y morfología |
| Capas tardías especializan a la tarea | Capas tardías especializan al dominio (toxicidad) |

La lección es la misma: **para transformers (visuales o textuales), el preentrenamiento NO es opcional cuando el dataset target es chico**. El modelo necesita una inicialización que ya entienda el medio (píxeles, tokens) antes de poder aprender la tarea.

## Las clases que NO se aprendieron — el problema real

Tres clases quedaron en 0% de recall **incluso con finetuning**:

- `severe_toxic` (~11 ejemplos en train)
- `threat` (~3-4 ejemplos en train)
- `identity_hate` (~10 ejemplos en train)

**Esto NO es un problema del LR ni del preentrenamiento.** Es un problema de **datos**. El gradiente que viene de 3-4 ejemplos positivos es insuficiente para escapar del mínimo trivial "predice cero". Soluciones reales:

1. **`pos_weight` en `BCEWithLogitsLoss`** — pondera positivos por la inversa de su frecuencia, multiplicando su contribución al gradiente.
2. **Focal Loss** ([Lin et al. 2017](https://arxiv.org/abs/1708.02002)) — penaliza fuertemente los falsos negativos en minoritarias.
3. **Oversampling** de positivos minoritarios (`WeightedRandomSampler`).
4. **Threshold tuning por clase** — no usar 0.5 como umbral universal de decisión, ajustarlo en validation por clase.
5. **Más datos** — la solución más obvia: subir `subset_size` de 0.01 a 0.05+ multiplica los positivos.

El Ejercicio III explora qué pasa al variar el LR — confirmaremos que **ningún LR resuelve este problema**.
