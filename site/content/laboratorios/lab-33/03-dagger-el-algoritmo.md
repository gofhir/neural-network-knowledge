---
title: "DAGGER: el algoritmo y el loop"
weight: 3
---

Aquí está el corazón del lab. Con el entorno, la red y el experto listos ([parte anterior](02-pipeline-atari-modelo-experto)), se ejecuta el bucle de DAGGER. El detalle sorprendente: la diferencia entre Behaviour Cloning y DAGGER **es una sola línea de código**.

## El algoritmo, formalmente

DAGGER = **Da**taset **Agg**regation. La idea en una frase: *iterar entre desplegar la política actual para recolectar estados, y pedirle al experto que los etiquete, agregando todo a un dataset que crece*.

```
Inicializar D ← ∅                          (dataset vacío)
Inicializar π̂₁ ← política cualquiera        (estudiante inicial)

para i = 1, 2, …, N:
    1. πᵢ = βᵢ·π* + (1-βᵢ)·π̂ᵢ               ← política de MUESTREO (mezcla experto/estudiante)
    2. Desplegar πᵢ →                        recolectar estados {s}
    3. Para cada estado s:                   consultar al experto a* = π*(s)
    4. Dᵢ = {(s, π*(s))}                     ← etiquetar con el experto
    5. D ← D ∪ Dᵢ                            ← AGREGAR al dataset acumulado
    6. Entrenar π̂ᵢ₊₁ sobre TODO D            ← re-entrenar el estudiante
```

Los tres ingredientes que definen el método:

1. **Quién visita los estados (paso 2).** En BC, el experto. En DAGGER, una política que involucra al **estudiante** → lo expone a *sus propios* estados de error.
2. **Quién etiqueta (paso 3-4).** Siempre el **experto**. Aunque el estudiante conduzca, la etiqueta correcta la da el experto ($a^* = \pi^*(s)$). El estudiante nunca aprende de sus propias acciones — aprende *qué habría hecho el experto* en el estado al que él llegó.
3. **Agregación (paso 5).** El dataset **crece**, no se reemplaza → el estudiante no olvida lo aprendido mientras incorpora estados nuevos.

### El parámetro β

La política de muestreo $\pi_i = \beta_i\pi^* + (1-\beta_i)\hat\pi_i$ es una **mezcla estocástica**: con probabilidad $\beta_i$ actúa el experto, con $(1-\beta_i)$ el estudiante. El calendario típico: $\beta_1 = 1$ (primera iteración = puro experto = BC) y $\beta_i \to 0$. Es un **currículum**: de "el experto te lleva de la mano" a "conduces tú y el experto te corrige". Empezar con el experto evita que el estudiante random genere trayectorias basura al inicio.

## Fase 1: Behaviour Cloning (β=1, el experto conduce)

La primera fase recolecta 10.000 pasos jugados **por el experto** y entrena al estudiante sobre ellos:

```python
with torch.no_grad():
    for _ in range(num_rollouts):                                  # 10.000 pasos
        state_tensor = torch.FloatTensor(np.array([state])).to(DEVICE)
        expert_action = get_action_from_policy(expert_model, state_tensor)
        state, _, terminated, truncated, _ = env.step(expert_action.item())  # el EXPERTO conduce
        observations.append(state_tensor.squeeze(0).cpu())
        actions.append(expert_action.cpu())                        # etiqueta = experto
        if terminated or truncated: state, _ = env.reset()

# entrenar 5 épocas sobre estos datos
train_student_with_eval(student_model, torch.stack(observations),
                        torch.stack(actions).squeeze(), optimizer, env, epochs=5)
```

Esto es **exactamente Behaviour Cloning**: el dataset son estados de la distribución experta $d_{\pi^*}$.

El entrenamiento reduce la imitación a clasificación:

```python
loss = F.cross_entropy(student_actions, expert_actions)
```

`F.cross_entropy` aplica **softmax** a los logits (el softmax que faltaba en la red) y penaliza cuando el estudiante asigna baja probabilidad a la acción del experto. Minimizarla empuja al estudiante a *decir lo que el experto diría*. Se usa cross-entropy y no MSE porque las acciones son **categóricas** (NOOP/FIRE/RIGHT/LEFT no tienen orden ni distancia).

{{< callout type="warning" >}}
**El buffer NO crece indefinidamente.** El notebook recorta a los últimos 20.000 ejemplos: `observations = observations[-max_train_samples:]`. Esto es una **ventana deslizante** tipo replay buffer, no la agregación pura de Ross et al. Ahorra memoria y da más peso a estados recientes, pero se aleja de la garantía teórica (que asumía dataset creciente). Es DAGGER *con replay buffer acotado*, no DAGGER estricto.
{{< /callout >}}

## Fases 2–19: DAGGER puro (β=0, el estudiante conduce)

El loop principal repite 18 veces un ciclo **casi idéntico** a la fase 1, con **una sola diferencia**:

```python
for iteration in range(2, 20):
    state, _ = env.reset()
    with torch.no_grad():
        for _ in range(num_rollouts):
            state_tensor = torch.FloatTensor(np.array([state])).to(DEVICE)
            student_action = get_action_from_policy(student_model, state_tensor)  # ← NUEVO
            expert_action  = get_action_from_policy(expert_model,  state_tensor)
            state, _, terminated, truncated, _ = env.step(student_action.item())  # ← el ESTUDIANTE conduce
            observations.append(state_tensor.squeeze(0).cpu())
            actions.append(expert_action.cpu())                                   # ← etiqueta = experto
            if terminated or truncated: state, _ = env.reset()
    observations = observations[-max_train_samples:]
    actions = actions[-max_train_samples:]
    train_student_with_eval(student_model, torch.stack(observations),
                            torch.stack(actions).squeeze(), optimizer, env, epochs=5)
```

### La diferencia de una línea que lo cambia todo

Compara con la fase 1:

- **Fase 1 (BC):** `env.step(expert_action)` → el **experto** conduce.
- **Fases 2+ (DAGGER):** `env.step(student_action)` → el **estudiante** conduce, pero se guarda `expert_action` como etiqueta.

Eso es todo. Se calculan **ambas** acciones en cada paso, pero se usan para cosas distintas: la del estudiante determina *a dónde va* (genera la trayectoria), la del experto determina *qué se aprende* (la supervisión). Esta disociación —actuar con una política, aprender de otra— es el núcleo técnico de DAGGER.

Como el estudiante conduce, visita **sus propios estados de error** (la distribución $d_{\hat\pi}$) — precisamente los que BC nunca veía. Y el experto los etiqueta: *"cuando llegues a este estado raro donde tú solo caes, esto es lo que yo haría para recuperarme"*.

{{< callout type="info" >}}
**Cómo se maneja la discrepancia experto-estudiante (pregunta 5 de la tarea).** Donde ambos coinciden, no hay nada que corregir. Donde **difieren** (la discrepancia), el entorno avanza con la acción del estudiante (vive las consecuencias) y se guarda `(estado, expert_action)`. Al re-entrenar, la cross-entropy empuja al estudiante hacia el experto en ese estado. La discrepancia no se resuelve en el instante: se **acumula como dato** y se corrige en la siguiente ronda, iteración tras iteración.
{{< /callout >}}

## Simplificaciones respecto al DAGGER canónico

Esta implementación toma dos atajos válidos y comunes, que conviene reconocer:

1. **β es binario, no gradual.** Salta de β=1 (fase 1, puro experto) a β=0 (fases 2-19, puro estudiante), sin el calendario suave $\beta_i \to 0$. Funciona porque la fase BC deja al estudiante *razonablemente competente* antes de soltarlo — un currículum de dos escalones en vez de una rampa.
2. **Buffer acotado (20k), no agregación pura.** Como el estudiante mejora, los estados que visita cambian; el buffer deslizante "sigue" al estudiante actual. Deseable (no reentrena sobre errores ya superados), pero sin garantía de recordar estados que dejó de visitar.

## Medir el progreso: `evaluate_model`

Cada época, el estudiante se evalúa jugando 50 partidas y se registra la **mediana** de recompensa:

```python
return np.median(rewards)
```

Se usa **mediana y no media** por robustez: los scores de Atari tienen alta varianza y colas pesadas; una racha afortunada inflaría el promedio. La mediana refleja el rendimiento *típico* y hace la curva de aprendizaje menos ruidosa. Cada llamada a `train_student_with_eval` (5 épocas) añade 5 puntos a la curva → 5 (fase BC) + 18×5 = **95 puntos** en total.

{{< callout type="info" >}}
**Loss y score no se mueven juntos.** La cross-entropy mide *qué tan bien copia al experto en el dataset*; el score, *qué tan bien juega*. Pueden divergir — y en este lab divergen dramáticamente. Esa brecha es justo lo que la siguiente parte mide con los resultados reales.
{{< /callout >}}
