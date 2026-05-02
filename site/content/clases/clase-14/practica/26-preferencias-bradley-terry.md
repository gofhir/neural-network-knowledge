---
title: "26 - Preferencias y Bradley-Terry"
weight: 260
math: true
---

Cerramos Fase 6 con un Mini-LLaMA SFT que sigue 4 instrucciones. Algunas mejor (`repeat`, `qa` al 100%), otras parciales (`reverse`, `upper` al 21-23%). Y el formato general aprendido: drift de 40% a 0%. ¿Para que mas necesitamos?

Fase 7 introduce **DPO** — Direct Preference Optimization. La idea: si SFT te enseña _que respuesta dar_, DPO te enseña _que respuesta es mejor entre dos_. Antes de tocar la loss DPO ([cap 27](../27-dpo-loss-derivation)), construimos la intuicion con preferencias y el modelo Bradley-Terry. Volvemos del [cap 25](../25-sft-eval) que cerro SFT.

---

## 1. La pregunta motivadora

Imagina dos respuestas a la misma pregunta:

- **A**: `Shakespeare wrote Hamlet.` (concisa, correcta)
- **B**: `O hark, the Bard didst pen the Prince of Denmark, methinks.` (correcta pero pomposa)

Ambas son `factually correct`. SFT no distingue: si ambas estuvieran en el dataset, el modelo aprende a generar cualquiera con probabilidad proporcional a su frecuencia. Pero **una es preferida sobre la otra** en la mayoria de contextos. ¿Como enseñamos eso al modelo?

La respuesta clasica: dale al modelo pares `(y_w, y_l)` — winner y loser — y enseñale a empujar la probabilidad de `y_w` hacia arriba y la de `y_l` hacia abajo. Pero antes de hacerlo concretamente con una loss, necesitamos un modelo matematico de _que significa preferir_. Ese modelo es Bradley-Terry.

---

## 2. Bradley-Terry — la historia

En 1952, Ralph A. Bradley y Milton E. Terry estudiaron `paired comparisons` en deportes, psicometria y testing de productos. Su modelo, hoy estandar en la literatura de preferencias:

$$
P(y_w \succeq y_l) = \sigma(r(y_w) - r(y_l))
$$

Donde:

- `y_w` = winner (la respuesta preferida)
- `y_l` = loser (la respuesta rechazada)
- `r(y)` = reward escalar asociado a `y` (mas alto = mejor)
- `sigma` = funcion sigmoide, `sigma(z) = 1 / (1 + e^{-z})`

La probabilidad de que se prefiera `y_w` sobre `y_l` depende SOLO de la diferencia de rewards. Si `r_w >> r_l`, la sigmoide tiende a 1. Si `r_w == r_l`, da exactamente 0.5 (empate). Si `r_w << r_l`, tiende a 0. La sigmoide aplasta los extremos y queda lineal cerca de cero — lo cual es deseable: si los rewards estan cerca, una pequeña diferencia mueve la probabilidad de manera notable; si los rewards estan muy separados, probabilidades adicionales casi no cambian.

¿Por que la sigmoide y no otra funcion? Porque mapea cualquier diferencia real a `(0, 1)`, es monotona, derivable en todas partes, y sus asintotas son justamente lo que esperamos: certeza total cuando un reward domina. Es la eleccion natural para preferencias binarias.

---

## 3. El script — demo numerica

`18_dpo_intro.py` completo:

```python
"""18_dpo_intro.py - Cap 26: Bradley-Terry numericamente.

Demo: dado un par (y_w, y_l) con rewards r_w, r_l, computar P(y_w succeq y_l).
Sin red neuronal — solo numpy/math. Construye intuicion para la loss DPO del cap 27.
"""
import math

print("=== Bradley-Terry: P(y_w succeq y_l) = sigma(r_w - r_l) ===\n")
sigmoid = lambda z: 1 / (1 + math.exp(-z))

cases = [
    ("preferencia clara",   2.0,  -1.0),
    ("preferencia tibia",   0.5,   0.0),
    ("empate",              1.0,   1.0),
    ("opuesto",            -2.0,   1.0),
]
for label, rw, rl in cases:
    p = sigmoid(rw - rl)
    print(f"{label:<22} r_w={rw:+.1f}  r_l={rl:+.1f}  P(y_w>y_l)={p:.3f}")

print("\n=== Log-likelihood de un dataset de 3 preferencias ===")
prefs = [(2.0, -1.0), (0.5, 0.0), (-2.0, 1.0)]
ll = sum(math.log(sigmoid(rw - rl)) for rw, rl in prefs)
print(f"sum log P(y_w>y_l) = {ll:.4f}")
print("\nMaximizar esta log-likelihood = aprender los rewards.")
print("DPO va mas lejos: parametriza r implicitamente via la policy y ref model.")
```

Sin red neuronal, sin gradiente, sin GPU. Cuatro casos sinteticos con rewards puestos a mano y un mini-dataset de tres preferencias para computar la log-likelihood. La idea es ver _que predice Bradley-Terry_ antes de meterse con como aprenderlo.

---

## 4. Output literal

```
=== Bradley-Terry: P(y_w succeq y_l) = sigma(r_w - r_l) ===

preferencia clara      r_w=+2.0  r_l=-1.0  P(y_w>y_l)=0.953
preferencia tibia      r_w=+0.5  r_l=+0.0  P(y_w>y_l)=0.622
empate                 r_w=+1.0  r_l=+1.0  P(y_w>y_l)=0.500
opuesto                r_w=-2.0  r_l=+1.0  P(y_w>y_l)=0.047

=== Log-likelihood de un dataset de 3 preferencias ===
sum log P(y_w>y_l) = -3.5713
```

---

## 5. Lectura del output

**Caso 1 (preferencia clara, +3 de diferencia)**: P=0.953. Casi certeza de que `y_w` se prefiere. La sigmoide con argumento `+3` ya esta cerca de su asintota superior.

**Caso 2 (preferencia tibia, +0.5)**: P=0.622. El reward apenas inclina la balanza. La sigmoide cerca de cero es casi lineal, asi que `0.5` mueve la probabilidad a `~0.62` — preferencia debil, pero existe.

**Caso 3 (empate)**: P=0.500. Si los rewards son iguales, Bradley-Terry no tiene preferencia. `sigma(0) = 0.5` exactamente. Es el punto de indecision del modelo.

**Caso 4 (opuesto, `r_w` < `r_l`)**: P=0.047. Aqui le pasamos al modelo una contradiccion: la etiqueta dice que `y_w` es preferida, pero los rewards dicen que `y_l` es mejor (`r_w = -2`, `r_l = +1`). La sigmoide con argumento `-3` aplasta a `~0.05`. Es decir: bajo estos rewards, el modelo predice que la preferencia humana es muy poco probable.

La log-likelihood total `-3.5713` es negativa porque el caso 4 tiene los rewards al reves de la preferencia etiquetada. Maximizar la log-likelihood significaria ajustar los rewards para que el caso 4 sea consistente — subir `r_w` o bajar `r_l`. Esto es exactamente lo que hace un reward model entrenado: encontrar valores de `r` que reproduzcan las preferencias humanas observadas.

---

## 6. RLHF clasico vs DPO — el camino corto

En RLHF clasico (`Reinforcement Learning from Human Feedback`, popularizado por InstructGPT en 2022), el flujo tiene tres fases:

1. **SFT**: entrenar el modelo con demostraciones (lo que hicimos en caps 23-25).
2. **Reward model**: recolectar pares de preferencias humanas `(x, y_w, y_l)` y entrenar `r_phi(x, y)` — una red neuronal que predice rewards desde texto. La loss es la log-likelihood Bradley-Terry: `-log sigma(r_phi(x, y_w) - r_phi(x, y_l))`.
3. **PPO**: usar `r_phi` como recompensa en un loop de Proximal Policy Optimization para empujar la policy `pi_theta` a generar respuestas con reward alto, regularizando con un termino KL contra una `pi_ref` para no irse muy lejos de SFT.

DPO (Rafailov et al. 2023, paper [arxiv:2305.18290](https://arxiv.org/abs/2305.18290)) demostro que el paso 2 es innecesario. Hay una manera de entrenar la policy DIRECTAMENTE desde las preferencias, saltandose el reward model y el PPO. La derivacion es elegante: en [cap 27](../27-dpo-loss-derivation) la vemos paso a paso.

La consecuencia practica: tres fases (SFT + RM + PPO) se reducen a dos (SFT + DPO). Menos hiperparametros, menos infraestructura, mas estabilidad.

---

## 7. La intuicion clave para cap 27

DPO entrena la policy `pi_theta` para que SU log-probability ratio (vs un reference model `pi_ref`, tipicamente la policy SFT congelada) reproduzca los rewards Bradley-Terry. La formula final de la loss es:

$$
\mathcal{L}_{DPO} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{ref}(y_l \mid x)}\right)
$$

Comparala con Bradley-Terry: misma estructura — `-log sigma(diferencia)`. La diferencia es que en lugar de rewards explicitos `r(y_w) - r(y_l)`, ahora hay una diferencia de log-ratios entre la policy y la referencia, escalada por un hiperparametro `beta`. Es como decir: el reward _implicito_ de cada respuesta es `beta * log(pi_theta / pi_ref)`. Maximizar la log-likelihood Bradley-Terry sobre esos rewards implicitos es lo que minimiza esta loss.

En [cap 27](../27-dpo-loss-derivation) derivamos por que esta formula. En [cap 28](../28-dpo-dataset) construimos el dataset de pares. En [cap 29](../29-dpo-training) entrenamos.

---

## 8. Preguntas de verificacion

1. ¿Por que la sigmoide y no otra funcion para mapear diferencias de reward a probabilidades?
2. Si dos respuestas tienen `r_w = r_l`, ¿que dice Bradley-Terry?
3. ¿Que ventaja practica da DPO sobre RLHF clasico?

Pista para la 1: la sigmoide es monotona, derivable en todas partes, mapea `R` a `(0, 1)`, y sus asintotas son `0` y `1` — exactamente las propiedades que pedimos a una funcion que convierte diferencias arbitrarias en probabilidades. Ademas, su derivada es maxima en cero, lo cual la hace bien comportada para optimizacion.

Pista para la 2: `sigma(0) = 0.5` exactamente. Empate — el modelo no tiene preferencia. Esto es coherente: si dos respuestas tienen el mismo reward, no hay razon para preferir una sobre la otra.

Pista para la 3: DPO se salta el reward model y el PPO. Una sola fase de training (DPO) en vez de dos (RM + PPO). Menos infraestructura, menos hiperparametros que tunear, mas estable porque no hay un reward model ruidoso entre la policy y la señal de preferencia.

---

Volver al [hub de practica](..) o a la [Clase 14](../..).
