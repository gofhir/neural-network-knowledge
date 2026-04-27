---
title: "Ejercicios Prácticos"
weight: 40
math: true
---

## Ejercicio I — Vertical Flip Transform

**Enunciado:**

> Cree su propia transformación, esta debe recibir una imagen y devolverla **reflejada verticalmente**. Luego visualice su transformación y pruebe entrenar un modelo utilizándola. Comente los resultados.

Pasos:

1. Implementar una clase `VerticalFlipTransform` que sea callable y devuelva la imagen volteada sobre el eje horizontal (lo que arriba, abajo).
2. Aplicarla con `transforms.RandomApply([VerticalFlipTransform()], p=0.5)` durante el entrenamiento.
3. Entrenar un modelo `ex_model` con la misma arquitectura y hiperparámetros del baseline, usando la nueva transformación en train.
4. Comparar test accuracy del nuevo modelo con el baseline y con el modelo aug (rotation + crop + horizontal flip).

*Pista:* `PIL.Image` tiene un método `.transpose(Image.FLIP_TOP_BOTTOM)` que hace exactamente esto.

---

## Ejercicio II — Finetuning sobre flowers

### Parte 1 — Explicación de la línea clave

**Enunciado:**

> Explique brevemente la siguiente línea de código:
>
> ```python
> ft_model.fc = nn.Linear(in_features=512, out_features=n_classes, bias=True)
> ```

Se espera una explicación corta (1-2 párrafos) de qué hace esa línea, por qué es necesaria y qué pasaría si se omitiera.

### Parte 2 — Tabla comparativa de tres escenarios

**Enunciado:**

> Haga una tabla resumen en donde muestre el rendimiento obtenido en los **tres escenarios** (modelo base, modelo con data augmentation, modelo con finetuning) en train, val y test. Comente brevemente:
>
> 1. ¿Cuál escenario logró el mejor rendimiento? ¿Por qué?
> 2. ¿Qué relación tiene esto con lo que vimos en clase sobre transfer learning y data augmentation?

---

## Ejercicio III — Learning rate y BERT

**Enunciado:**

> Cree un modelo nuevo para realizar finetuning y pruébelo con **tres learning rates distintos** al que utilizamos por defecto (`2e-5`).
>
> 1. Analice brevemente el entrenamiento de los modelos (gráficos con las curvas de pérdida y accuracy por época), qué diferencias ve en estos.
> 2. Analice brevemente el rendimiento en el set de test medido según la métrica **Macro F1-Score**, además tome en consideración las matrices de confusión en su análisis.

### Preguntas adicionales

> **Pregunta 1.** En base a los modelos entrenados, ¿con qué *learning rate* utilizaría para entrenar un modelo? Justifique muy brevemente.
>
> **Pregunta 2.** Proponga una forma de *data augmentation* que crea que ayudaría para este problema (clasificación multi-label de comentarios tóxicos).

### LRs sugeridos

El enunciado pide tres LRs distintos al default `2e-5`. Selección utilizada en este lab:

| LR | Razón pedagógica |
|---|---|
| `5e-6` | LR muy bajo — para mostrar subentrenamiento con presupuesto fijo de 5 epochs |
| `5e-5` | LR alto-pero-razonable — el "óptimo agresivo" del paper original de BERT |
| `1e-4` | LR muy alto — para poner a prueba estabilidad / catastrophic forgetting |

Los tres LRs cubren un rango logarítmico amplio (3 órdenes de magnitud) en torno al default `2e-5`, lo que permite identificar el comportamiento en U invertida típico del fine-tuning de transformers.

---

## Anexo — ¿Cómo congelar capas del modelo?

Un patrón común en transfer learning es **congelar** capas del backbone para entrenar solo la cabeza (feature extraction) o solo las últimas capas (partial finetuning). Esto se hace seteando `requires_grad = False` en los parámetros que no se quieren entrenar:

```python
# Congelar TODAS las capas del modelo
for param in model.parameters():
    param.requires_grad = False

# Descongelar solo la capa final
model.fc.requires_grad = True
```

Para este lab no es estrictamente necesario congelar nada (el finetuning completo funciona bien), pero es una herramienta importante para cuando el dataset es muy chico o se quiere reducir cómputo.
