# Actividad 1 — Mejora de MAML sobre Omniglot (4-way 1-shot)

> Práctico Meta-Aprendizaje (Parte 1) · Diplomado IA UC · Prof. Pablo Messina
> Análisis del experimento de optimización de hiperparámetros de `run_MAML`.

## 1. Objetivo

El enunciado pide mejorar el desempeño de MAML modificando los **argumentos** de
`run_MAML` (`adaptation_steps`, `meta_lr`, `fast_lr`, `meta_batch_size`,
`num_iterations`, modelo), **sin alterar WAYS ni SHOTS**. En lugar de tantear
cambios al azar, diseñamos un **ablation experimental**: modificar una palanca a la
vez para medir su aporte aislado, y solo al final combinarlas con fundamento.

## 2. Línea base

| Modelo | Iters | adapt_steps | batch | meta_lr | fast_lr | **Test Acc** |
|---|---|---|---|---|---|---|
| OmniglotFC | 50 | 1 | 32 | 0.003 | 0.5 | **0.699** |

Las curvas mostraban error decreciente y accuracy creciente **aún sin aplanarse** a
la iteración 50 → señal de **subentrenamiento**, no de techo de capacidad.

## 3. Diseño experimental y resultados

| # | Config | Train | Valid | **Test** | Brecha train−valid |
|---|---|---|---|---|---|
| 0 | FC · 50 · steps1 · b32 | 0.602 | 0.727 | **0.699** | — |
| 1 | CNN · 50 · steps1 · b32 | 0.625 | 0.719 | **0.706** | ~0 |
| 2 | CNN · 150 · steps1 · b32 | 0.844 | 0.742 | **0.749** | 0.10 |
| 3 | CNN · 150 · steps5 · b32 | 0.875 | 0.773 | **0.760** | 0.10 |
| 4 | **CNN · 400 · steps5 · b64 · meta_lr0.005 · fast_lr0.3** | 0.859 | 0.852 | **0.877** | ~0 |

**Mejora total: 0.699 → 0.877 = +0.178 (+25 % relativo).**

## 4. Análisis por palanca

**4.1 Modelo (FC → CNN), a iteraciones fijas — Δ ≈ +0.007 (nulo).**
Cambiar a una arquitectura convolucional, manteniendo 50 iteraciones, **no mejoró**
el resultado. La CNN tiene más parámetros y por tanto necesita más meta-iteraciones
para converger; a 50 iteraciones su ventaja queda "escondida" y rinde igual que el
FC. **Conclusión: la capacidad del modelo no sirve sin entrenamiento suficiente.**

**4.2 Iteraciones (50 → 150) — Δ ≈ +0.043 (la palanca dominante).**
Al triplicar las iteraciones sobre la CNN, el test saltó de 0.706 a 0.749. Esto
confirmó que el **cuello de botella real era el volumen de meta-entrenamiento**, no
la arquitectura. Es coherente con que el benchmark canónico de Omniglot (>0.98)
entrena del orden de decenas de miles de iteraciones.

**4.3 Pasos de adaptación (1 → 5) — Δ ≈ +0.011 (marginal, costoso).**
Más pasos de adaptación interna por tarea mejoran la señal del meta-gradiente, pero
el aporte fue pequeño y a un costo de ~5× en el bucle interno. Buen costo/beneficio
bajo: el cómputo rinde más invertido en iteraciones.

**4.4 Estabilización (batch 32 → 64, fast_lr 0.5 → 0.3) + más iteraciones — Δ ≈ +0.117.**
La configuración final combinó la palanca ganadora (400 iteraciones) con dos ajustes
de estabilización. Resultado: el mejor test (0.877) **y la eliminación del
overfitting** (ver sección 5).

## 5. Hallazgo central: el overfitting y su control

A `meta_batch_size=32` (configs 2 y 3) apareció una **brecha train−valid de ~0.10**:
el modelo aprendía mejor las clases del split de entrenamiento que las de
validación/test. En meta-learning esto es **overfitting a las clases del pool de
tareas**, no a imágenes individuales.

La solución, lograda **solo con argumentos permitidos**:

- **`meta_batch_size` 32 → 64:** promediar el meta-gradiente sobre el doble de
  tareas reduce su varianza; un gradiente estable no se especializa en tareas
  concretas del pool.
- **`fast_lr` 0.5 → 0.3:** con 5 pasos de adaptación, un learning rate interno menor
  evita la sobre-adaptación agresiva a cada support set.

Efecto medido: la brecha cayó de 0.10 a ~0.007 (train 0.859 ≈ valid 0.852), con
curvas de train y validación superpuestas durante toda la corrida — **entrenamiento
sano**. El test (0.877) incluso superó la validación, dentro del ruido de muestreo.

> Nota metodológica: `run_MAML` reporta el modelo de la **última** iteración (no el
> mejor según validación). Sin early stopping, una corrida que sobreajuste
> subreportaría su pico. En la config final esto no afecta porque train ≈ valid
> hasta el final.

## 6. Conclusiones

1. **El factor de mayor impacto fue el número de iteraciones de meta-entrenamiento.**
   Es la palanca que más recorrido tenía y la que produjo los saltos relevantes.
2. **Las palancas interactúan:** la arquitectura CNN solo aporta cuando hay
   iteraciones suficientes para explotarla. Aisladamente no mejoró nada.
3. **El overfitting es controlable con hiperparámetros del meta-gradiente**
   (`meta_batch_size`, `fast_lr`), sin necesidad de regularización explícita ni de
   modificar la función — basta estabilizar la optimización.
4. **Límite estructural:** hacia las 400 iteraciones las curvas entran en
   rendimientos decrecientes (~0.85-0.88). Acercarse al 0.98 de la literatura
   requeriría MAML de segundo orden (`first_order=False`) y un orden de magnitud más
   de iteraciones, fuera del alcance de cómputo de este laboratorio.

## 7. Configuración final recomendada

```python
model = l2l.vision.models.OmniglotCNN(WAYS)
model.to(device)
run_MAML(dataset="omniglot", model=model, device=device,
         ways=WAYS, shots=SHOTS,
         num_iterations=400, adaptation_steps=5,
         meta_batch_size=64, meta_lr=0.005, fast_lr=0.3)
# Meta Test Accuracy ≈ 0.877, sin overfitting (train ≈ valid)
```
