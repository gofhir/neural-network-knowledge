---
title: "Los tres regímenes, TSA y análisis"
weight: 3
---

El lab compara tres formas de entrenar el mismo BERT, para aislar el aporte de UDA. Antes, la pieza que hace posible entrenar con 20 etiquetas sin colapsar: TSA.

## TSA (Training Signal Annealing): el freno anti-overfitting

Con 20 etiquetas, BERT (110M parámetros) memoriza y se sobreconfía en pocos pasos. TSA lo evita: cuando el modelo predice un ejemplo etiquetado con **demasiada confianza**, multiplica su pérdida por **0** (lo enmascara). Y el umbral de "demasiada confianza" **crece durante el entrenamiento**:

```
Inicio:  umbral = 0.5  → enmascara casi todos los supervisados
Mitad:   umbral = 0.75 → enmascara solo los muy confiados
Final:   umbral = 1.0  → no enmascara a nadie
```

![Schedule lineal de TSA: el umbral sube de 0.5 a 1.0](/laboratorios/lab-28/schedule-tsa.png)

**Por qué funciona:** al principio el modelo no debería estar seguro de nada; cualquier ejemplo donde ya está muy confiado es síntoma de memorización → se descarta. Esto **frena la señal supervisada** deliberadamente ("annealing" = enfriar gradualmente), forzando al entrenamiento a apoyarse en la [consistencia](../01-consistencia-y-uda) en las etapas tempranas. Luego se relaja el freno.

En el código, el truco `(-sup_loss).exp()` recupera la probabilidad de la clase correcta sin recalcular el softmax (porque `sup_loss = -log(p)` ⟹ `exp(-sup_loss) = p`):
```python
larger_than_threshold = (-sup_loss).exp() > tsa_thresh    # ¿prob de la clase correcta > umbral?
loss_mask = 1 - larger_than_threshold.float()             # 1=conservar, 0=enmascarar
sup_loss = (sup_loss * loss_mask).sum() / torch.max(loss_mask.sum(), torch_one)  # guard anti /0
```
El `torch.max(loss_mask.sum(), 1)` evita división por cero cuando **todos** los ejemplos quedan enmascarados (frecuente al inicio) — sin ese guard, `0/0 = NaN` y el entrenamiento explota.

## Los tres regímenes (resultados en test)

| Régimen | Etiquetas | Datos no etiq. | **Test Acc** | **Test Loss** |
|---|---|---|---|---|
| Full | 20.000 | — | **87.65%** | 0.4300 |
| Low | 20 | — | **60.58%** | 2.1453 |
| **UDA** | **20** | **~65.000** | **85.06%** | **0.3443** |

### Régimen 1 — Full (el techo)

BERT con 20.000 etiquetas, sin UDA. Converge sin drama a ~88%.

![Curva de validación del régimen full: salto rápido a ~88%](/laboratorios/lab-28/curva-full.png)

La curva salta de **50% (azar) a 84% en 250 pasos** — el poder del transfer learning: BERT ya "entiende" el lenguaje por su pre-entrenamiento, solo aprende la tarea. Luego mesetea en ~88%. Hay overfitting **sutil**: la val_loss sube levemente (0.32→0.38) mientras el train_loss cae a ~0.005 (memoriza el train), pero el accuracy se sostiene porque 20k etiquetas anclan el modelo.

### Régimen 2 — Low (el piso)

Mismo modelo, **solo 20 etiquetas**, sin UDA.

![Curva de validación del régimen low: estancado ~55%](/laboratorios/lab-28/curva-low.png)

Overfitting **catastrófico**: accuracy clavado en ~55-60% (apenas sobre el azar), y la val_loss **explota de 0.71 a 2.84** mientras el train_loss cae a 0.0002. El modelo memoriza perfectamente las 20 reseñas y se equivoca con altísima confianza en datos nuevos. Es sobreconfianza pura: predice la clase incorrecta con probabilidad ~0.95, y la cross-entropy castiga eso exponencialmente. **Reducir las etiquetas 1000× (20k→20) tira el accuracy de 88% a ~60%, pese a usar BERT pre-entrenado.**

### Régimen 3 — UDA (el rescate)

Las mismas 20 etiquetas + consistencia sobre 65k datos no etiquetados.

![Curva de validación de UDA: arranque lento, meseta, despegue](/laboratorios/lab-28/curva-uda.png)

Tres fases, cualitativamente distinta de las otras dos:
1. **Arranque lento (0-750): ~52-59%.** TSA enmascara casi toda la señal supervisada; el modelo depende de la consistencia, señal débil e indirecta.
2. **Meseta (1000-1750): ~59%.** El "le cuesta aprender" del notebook.
3. **Despegue (2000-5000): 59% → 82%.** TSA se relaja, la señal supervisada vuelve sobre representaciones ya buenas, y sube sostenido **sin explotar**.

## La tesis del lab

```
Brecha full − low  = 87.65 − 60.58 = 27.1 puntos
UDA − low          = 85.06 − 60.58 = 24.5 puntos recuperados  →  90.5% de la brecha cerrada
```

Con las **mismas 20 etiquetas** que dieron 60.58%, UDA llega a **85.06%** — a solo **2.6 puntos** del modelo con 1000× más etiquetas. Los datos no etiquetados (baratos) sustituyen casi por completo a las etiquetas (caras).

## El hallazgo fino: UDA está mejor calibrado que el full

| | Test Acc | Test Loss |
|---|---|---|
| Full | 87.65% | 0.4300 |
| **UDA** | 85.06% | **0.3443** ← menor |
| Low | 60.58% | 2.1453 ← explota |

UDA tiene **menor loss que el full** pese a menor accuracy: sus probabilidades son más honestas (mejor calibración). El full se sobreajusta un poco (val_loss sube); UDA, gracias a TSA (frena memorización) + consistencia (predicciones estables ante perturbaciones), acierta con confianza realista.

> **Conexión con MDM/FHIR:** por esto interesa la regularización por consistencia en un scorer. Un modelo con menos accuracy pero mejor calibrado (loss 0.34 vs 0.43) es más útil en producción, porque sus umbrales de decisión son confiables. Un scorer sobreconfiado (como el régimen low) es peligroso: dice "99% match" cuando se equivoca. La sobreconfianza del low es la razón por la que un GBM necesita calibración (Platt/isotónica) antes de fijar umbrales.

---

**Siguiente:** [Actividades resueltas](../04-actividades).
