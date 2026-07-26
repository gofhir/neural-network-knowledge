---
title: "Resultados y las 5 preguntas"
weight: 4
---

El notebook ejecutado en Colab (GPU T4) entrega uno de los contrastes más nítidos posibles: **experto = 10, Behaviour Cloning puro = 0, DAGGER = 5**. Esta parte lee esos números y consolida las respuestas de la tarea.

## Los resultados reales

| Métrica | Valor medido |
|---|---|
| Experto (techo, mediana 50 episodios) | **10.00** |
| Estudiante tras BC puro (fase 1) | **0.00** en las 5 épocas |
| Estudiante final (tras 18 fases DAGGER) | **5.00** |

### El hallazgo estrella: BC puro fracasó por completo

Mira la fase 1 (Behaviour Cloning puro):

```
[Epoch 1] Loss: 1.324764 | Student Eval Score: 0.00
[Epoch 2] Loss: 1.286523 | Student Eval Score: 0.00
[Epoch 3] Loss: 1.239310 | Student Eval Score: 0.00
[Epoch 4] Loss: 1.201963 | Student Eval Score: 0.00
[Epoch 5] Loss: 1.167832 | Student Eval Score: 0.00
```

**La loss baja (1.32 → 1.17) pero el score es 0.00 clavado.** Es la demostración empírica más contundente posible del [covariate shift](01-imitacion-y-covariate-shift): el estudiante *aprende a imitar al experto en el dataset* (la loss baja → clasifica mejor), pero **cuando lo sueltan a jugar, saca cero** — falla apenas se sale de la distribución del experto. BC puro, aquí, no produjo un jugador funcional en absoluto. No se quedó corto: fracasó del todo.

### DAGGER rescató el aprendizaje: 0 → 5

En cuanto el estudiante empieza a conducir (fase 2+), el score despega y sube —ruidosamente— hasta estabilizarse cerca de 5:

![Curva de aprendizaje: Student vs Expert. La curva azul del estudiante parte en 0 durante la fase BC (épocas 0-5), luego sube ruidosamente hasta ~5 durante las fases DAGGER, con picos de 6, siempre bajo la línea roja del experto en 10](/laboratorios/lab-33/curva-student-vs-expert.png)

La curva cuenta la historia completa en tres regiones:

1. **Meseta en 0 (épocas 0–5, fase BC):** el estudiante no juega. La firma del covariate shift.
2. **Segunda subida (épocas 6–95, fases DAGGER):** la curva despega y trepa a ~5. **Esta es la demostración empírica de que DAGGER funciona** — al entrenar sobre los estados que el estudiante visita, cierra la brecha que BC no podía.
3. **Estancamiento en ~50% del experto:** oscila entre 4 y 6, con picos de 6, sin alcanzar la línea roja (10).

La diferencia entre "BC = 0" y "DAGGER = 5" es, literalmente, todo el valor del algoritmo hecho número.

### Tres matices que los datos reales revelan

- **La curva es MUY ruidosa.** Oscila entre 2 y 6 constantemente. Es el ruido intrínseco de RL — evaluar una política sobre 50 episodios-vida sigue teniendo alta varianza.
- **El estudiante se estancó en ~50% del experto.** 19 fases no bastaron para igualarlo; con este buffer acotado, el techo del método ronda 5–6.
- **Confirmación del shape 5D.** El output `Observation size: torch.Size([4, 84, 84, 1])` confirma el `1` sobrante que activa el parche `if x.dim() == 5: squeeze(-1)` en cada forward. No era paranoia del autor: era necesario.

## La tarea: las 5 respuestas

### 1. [1.5 pts] Una **ventaja** de DAGGER frente a Behaviour Cloning

DAGGER corrige el **covariate shift**. En BC el estudiante se entrena solo sobre los estados que visita el *experto*; nunca ve los estados de error a los que él mismo llega, no aprende a recuperarse y los errores se **acumulan** ($O(\epsilon T^2)$). DAGGER recolecta los estados que visita el *estudiante* y le pide al experto que los **etiquete**, enseñándole a recuperarse y reduciendo la cota a $O(\epsilon T)$ (lineal).

> **Evidencia:** BC puro dejó al estudiante en **0.00** pese a que la loss bajaba; DAGGER lo llevó a **5.00**. La ventaja fue la diferencia entre un agente que no juega y uno que juega a la mitad del experto.

### 2. [1.5 pts] Una **desventaja** de DAGGER frente a Behaviour Cloning

DAGGER exige el **experto disponible online durante todo el entrenamiento** — no basta un dataset grabado offline. En cada iteración hay que consultarlo por los estados nuevos que genera el estudiante. Si el experto es un humano (caso Tesla), etiquetar cada estado nuevo es **caro, lento e incluso inseguro** (los estados de error son los peligrosos). BC solo necesita demostraciones grabadas una vez.

> **Costo secundario visible en el lab:** DAGGER consulta *dos* políticas por paso (estudiante + experto), duplicando los forward passes de la recolección.

### 3. [1 pt] ¿Por qué la salida del experto son números reales?

Porque el experto es un **DQN entrenado con Q-learning**: su salida son **Q-values**, $Q(s,a)$ = la recompensa futura acumulada esperada de cada acción. Un retorno esperado es un número real sin acotar (no restringido a $[0,1]$ ni suma 1), no una probabilidad. El experto elige por argmax del Q-value.

### 4. [1 pt] ¿Por qué la salida del estudiante son probabilidades?

Porque el estudiante se entrena como **clasificador** con `cross_entropy`, que aplica **softmax** a los logits → los convierte en una distribución $P(a\mid s)$ sobre las 4 acciones ($p_i \geq 0$, $\sum_i p_i = 1$). Modela *con qué probabilidad el experto tomaría cada acción*.

> **Precisión que demuestra dominio:** la salida cruda del estudiante también son logits reales; el softmax vive dentro de la cross-entropy. En inferencia, el argmax sobre logits ≡ argmax sobre probabilidades (softmax es monótono). El estudiante "produce probabilidades" por su *objetivo de entrenamiento*, no porque la salida esté literalmente normalizada.

### 5. [1 pt] ¿Cómo se maneja la discrepancia experto-estudiante en DAGGER?

Convirtiéndola en **señal de entrenamiento**. En cada paso se calculan ambas acciones. El entorno avanza con la del **estudiante** (así vive las consecuencias y llega a estados que el experto evitaría), pero se guarda el par **(estado, acción del experto)**. Al re-entrenar, la cross-entropy empuja al estudiante hacia la acción del experto en esos estados. La discrepancia no se corrige en el instante: se **acumula como dato** y se reduce ronda tras ronda, hasta que ambos tienden a coincidir.

## Limitaciones: por qué imitar no basta

El material complementario del notebook pone la imitación en su lugar, y **los resultados reales confirman sus dos limitaciones**:

- **"Está limitado al profesor."** El estudiante alcanzó 5, el experto era 10 — no lo superó. La imitación, por construcción, **no puede exceder al experto**.
- **"Difícilmente generaliza fuera del training set."** El fracaso de BC puro (score 0) es exactamente esto: fuera de los estados del experto, no generalizaba. DAGGER lo mitiga ampliando el training set, pero aun así se estancó.

La **ventaja** es que imitar es **más rápido que RL**: cada acción del experto es una señal supervisada directa, evitando la exploración costosa y el problema de asignación de crédito temporal (que sí enfrentaste en el [lab 31, DQN sobre CartPole](/laboratorios/lab-31)).

### El caso AlphaStar: la síntesis

StarCraft II tiene un espacio de acciones gigantesco (~$10^{26}$) y recompensas sparse → RL puro desde cero es inviable, e imitación pura nunca sería sobrehumana. AlphaStar (DeepMind, 2019) usó **ambas en secuencia**:

1. **Pre-entrenar con imitación** sobre partidas humanas → aprende lo básico rápido (lo que hiciste en este lab).
2. **Continuar con RL** (self-play) → supera a los humanos, rompe la limitación del techo.

Esta receta —**imitación para arrancar, RL para superar**— es el paradigma dominante: aparece en [AlphaGo](/clases/clase-33), en [RLHF](/fundamentos/rlhf) (los LLMs se pre-entrenan imitando texto humano, luego se refinan con PPO), y en Tesla. La moraleja del lab: la imitación es un punto de partida brillante y rápido, pero con techo (el profesor) y generalización frágil. Para ir más allá, hay que combinarla con [aprendizaje reforzado](/fundamentos/aprendizaje-reforzado).
