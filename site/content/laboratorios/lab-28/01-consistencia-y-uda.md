---
title: "Pérdida de consistencia y el método UDA"
weight: 1
---

UDA (Unsupervised Data Augmentation) es **aprendizaje semi-supervisado por consistencia**. Combina dos ramas: una supervisada clásica sobre las pocas etiquetas, y una no supervisada que fuerza al modelo a ser **consistente** ante aumentaciones de los datos sin etiqueta.

## Las dos ramas

```
        RAMA SUPERVISADA                    RAMA NO SUPERVISADA
   x_labeled ──► M ──► P(y|x)          x_unlab ─────► M ──► P(y|x_unlab)   [target, stop_gradient]
                  │                     x̂_unlab ────► M ──► P(y|x̂_unlab)   [se ajusta hacia el target]
          cross-entropy(y)                          │
                                          KL( P(y|x̂_unlab) || P(y|x_unlab) )
```

**Loss total** = `sup_loss` (cross-entropy sobre 20 etiquetas) + `consistency_loss` (KL sobre datos no etiquetados).

La idea que lo habilita: hay transformaciones que **no cambian la etiqueta**. Si cambio una palabra por su sinónimo, una reseña positiva sigue siendo positiva. Entonces se puede forzar que el modelo prediga lo mismo para un dato y su versión aumentada, **sin conocer la etiqueta real** — la consistencia es señal de entrenamiento gratis, disponible en abundancia.

## La pérdida de consistencia: KL-divergencia

El modelo debería predecir distribuciones parecidas para un dato y su aumentación, pero al principio no lo hace:

$$P(y \mid x_{unsup}) \neq P(y \mid \hat{x}_{unsup})$$

Se usa la **KL-divergencia** (mide cuán distintas son dos distribuciones, vale 0 si son idénticas) para acercarlas:

$$KL\big(P(y \mid \hat{x}_{unsup})\ \|\ P(y \mid x_{unsup})\big)$$

En el código:
```python
kl = nn.KLDivLoss(reduction="none")
unsup_loss = kl((bt_output.logits / tau).log_softmax(-1), src_probs)
```

**Gotcha del orden de argumentos:** `KLDivLoss` de PyTorch exige `kl(log_probs, probs)` — el 1er argumento son **log-probabilidades** (del aumentado, `.log_softmax`) y el 2º **probabilidades** (del original, `.softmax`). Invertirlos calcula la KL en la dirección equivocada. El notebook lo comenta explícitamente.

## El `stop_gradient`: por qué es imprescindible

Ambas ramas pasan por **el mismo modelo M**. La predicción del **original** se usa como **target fijo** (pseudo-etiqueta suave) y se congela con `stop_gradient`; solo se ajusta la predicción del **aumentado**:

```python
with torch.no_grad():                          # ← el stop_gradient
    src_output = model(src_input_ids, ...)     # predicción del ORIGINAL (target, sin gradiente)
bt_output = model(bt_input_ids, ...)           # predicción del AUMENTADO (se ajusta)
```

**Por qué es imprescindible — el colapso trivial.** Si el gradiente fluyera por **ambas** ramas, el optimizador minimizaría la KL de la forma más fácil: hacer que ambas predicciones sean **idénticas y triviales** (p. ej. `[0.5, 0.5]` para todo), logrando `KL=0` sin aprender nada.

```
Sin stop_gradient (mal):
   original:  [0.7, 0.3]      el gradiente empuja AMBAS al medio
   aumentado: [0.4, 0.6]  →   ambas colapsan a [0.5, 0.5]  → KL=0 (pero no aprendió nada)

Con stop_gradient (bien):
   original:  [0.7, 0.3]  ← CONGELADO (target)
   aumentado: [0.4, 0.6]  →  solo esta se mueve → [0.7, 0.3]  → KL=0 (aprendió la invarianza)
```

Con el freno, `KL=0` solo se logra cuando el aumentado iguala al original confiable — que es lo que queremos. El original es el "maestro" porque, al no estar distorsionado, da una predicción más limpia; es la lógica del **teacher-student / pseudo-labeling**.

## `tau` (temperatura / sharpening)

```python
unsup_loss = kl((bt_output.logits / tau).log_softmax(-1), src_probs)
```

`tau` divide los logits antes del softmax. Con `tau=1.0` no hace nada; `tau<1` afila la distribución (más confiada), `tau>1` la suaviza. En UDA se usa para controlar cuán "duras" son las predicciones que se igualan. Es el mismo mecanismo de temperatura que en la generación de texto (ver [back-translation](../02-datos-y-back-translation)) pero aplicado al target de consistencia.

## La loss combinada

```python
loss = sup_loss + unsup_loss     # sin un λ explícito: se suman 1:1
```

No hay un peso `λ` entre las dos ramas. El balance efectivo lo dan (a) el `unsup_ratio=3` (3× más datos no supervisados por batch) y (b) el [TSA](../03-tres-regimenes-y-analisis) que apaga la señal supervisada al inicio. Al principio `sup_loss≈0` (todo enmascarado por TSA) → domina la consistencia; al final la supervisada vuelve.

## Dónde encaja en la clase 28

La [clase 28](/clases/clase-28) cubre el [aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado). UDA es el primo [semi-supervisado](/fundamentos/aprendizaje-semi-supervisado): comparte con [SimCLR/MoCo](/fundamentos/aprendizaje-contrastivo) la filosofía de **invarianza a aumentaciones** (dos vistas del mismo dato deben coincidir), pero aplicada al **espacio de predicciones** (KL sobre probabilidades) en vez del espacio de embeddings (InfoNCE contrastivo). Además, BERT ya es un modelo autosupervisado (Masked LM), así que UDA es **SSL sobre SSL**: fine-tuning semi-supervisado de un modelo pre-entrenado sin etiquetas.

---

**Siguiente:** [Datos, back-translation y dataloaders](../02-datos-y-back-translation).
