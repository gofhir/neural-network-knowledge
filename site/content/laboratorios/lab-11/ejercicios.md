---
title: "Ejercicios Practicos"
weight: 40
math: true
---

## Actividad 1 — Cantidad de parámetros de los 3 modelos

**Enunciado:** Mencione la dimensionalidad oculta de los 3 modelos ejecutados en este laboratorio, calcule su cantidad de parámetros y explique por qué **no sería justo** utilizar dimensionalidad oculta pareja para todos los modelos.

Dimensiones del lab:

| Modelo | $n\_hidden$ |
|--------|-------------|
| RNN vanilla (Elman) | 147 |
| LSTM | 64 |
| BiLSTM | 40 |

Use `num_trainable_parameters(model)` para calcular los conteos:

```python
def num_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

---

## Actividad 2 — Clasificador con capa oculta + ReLU

**Enunciado:** Agregue otra capa lineal con una ReLU al clasificador del modelo de **LSTM (no bidireccional)** como se muestra en la figura del enunciado.

Pruebe variantes de **50, 150 y 300 neuronas** en la capa oculta.

- Grafique la pérdida para las 3 y comente.
- Calcule la cantidad de parámetros de los tres modelos y comente.

*Consejo: debe tener dos capas `Linear`, y se necesita una activación entre ellas.*

Es decir, reemplazar el `h2o = nn.Linear(hidden_size, output_size)` actual por:

```
h_lstm  ─►  Linear(hidden_size, mlp_hidden)  ─►  ReLU  ─►  Linear(mlp_hidden, output_size)  ─►  logits
```

con `mlp_hidden ∈ {50, 150, 300}`.
