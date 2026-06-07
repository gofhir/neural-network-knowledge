---
title: "MAML (Model-Agnostic Meta-Learning)"
weight: 261
math: true
---

{{< paper-card
    title="Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"
    authors="Chelsea Finn, Pieter Abbeel, Sergey Levine"
    year="2017"
    venue="ICML 2017"
    pdf="/papers/maml-finn-2017.pdf"
    arxiv="1703.03400" >}}
Uno de los papers fundacionales del meta-aprendizaje moderno. Su tesis es deceptivamente simple: en vez de aprender una *regla de actualización* o una *arquitectura especializada* para aprender rápido, aprendamos directamente **una inicialización de pesos $\theta$** tal que, partiendo de ella, unos pocos pasos de descenso de gradiente sobre una tarea nueva produzcan buena generalización. La frase del abstract lo resume: el método "trains the model to be easy to fine-tune". *Model-agnostic* es literal: no agrega parámetros para el meta-aprendizaje, no impone arquitectura, y sirve para clasificación, regresión y RL con cambios mínimos. Resultados de cabecera: **98.7%** en Omniglot 5-way 1-shot y **48.70%** en MiniImagenet 5-way 1-shot, igualando o superando al estado del arte de 2017 con menos parámetros.
{{< /paper-card >}}

---

## El problema

En 2017 el few-shot learning convivía con tres familias dominantes, y MAML se define en contraste con cada una.

La primera son los **meta-learners basados en gradiente aprendido**: una red (típicamente un LSTM) que *produce las actualizaciones de pesos* del learner. El trabajo más cercano en espíritu es Ravi & Larochelle (2017), que aprende tanto la inicialización como el optimizador mediante un meta-learner LSTM. La crítica de MAML: estos métodos expanden el número de parámetros aprendidos (el optimizador en sí) y atan el procedimiento a una arquitectura concreta.

La segunda son los **métodos métricos / no paramétricos** —Siamese Networks, Matching Networks, Prototypical Networks—, que aprenden un espacio de embeddings donde la clasificación se hace por comparación. Dieron algunos de los mejores resultados en clasificación, pero "are difficult to directly extend to other problems, such as reinforcement learning": un espacio métrico no tiene traducción natural a una política de control.

La tercera son los **modelos con memoria aumentada / RNN que ingieren datasets** (MANN, RL²): tratan la adaptación como la dinámica de una RNN condicionada por el conjunto de soporte. Son más generales (sirven para RL), pero exigen modelo recurrente y la adaptación está limitada por la longitud del rollout.

A esto se suma la práctica estándar de **pretraining + fine-tuning**: entrenar en un dataset grande y afinar para la tarea nueva. El problema —que MAML expone con su baseline "pretraining on all tasks"— es que ese pretraining optimiza para el *promedio* de las tareas, no para la *adaptabilidad*. Cuando tareas distintas exigen salidas contradictorias para la misma entrada (dos senos de fase opuesta), la red aprende a emitir el promedio y queda atrapada en una región desde la cual pocos pasos de gradiente no recuperan ninguna tarea individual.

La pregunta que motiva el paper: ¿se puede tener la **generalidad** de los métodos RNN/memoria, la **simplicidad** del fine-tuning con gradiente y el **rendimiento** de los métodos métricos, sin pagar parámetros extra ni arquitectura restringida? La respuesta de MAML es desplazar el aprendizaje al lugar más barato y universal posible: la *inicialización*.

---

## La idea central (aprender una inicialización adaptable)

La intuición es que **algunas representaciones internas son más transferibles que otras**. Una red podría aprender features aplicables a *todas* las tareas de la distribución $p(\mathcal{T})$ en vez de a una sola. Como el modelo se afinará con una regla *basada en gradiente* sobre la tarea nueva, se busca aprender $\theta$ tal que **esa misma regla de gradiente progrese rápido** en tareas de $p(\mathcal{T})$, sin sobreajustar.

Formulado de otro modo: se buscan parámetros **sensibles a cambios en la tarea**, de modo que pequeños cambios en $\theta$ —en la dirección del gradiente de cualquier tarea— produzcan grandes mejoras en esa pérdida. "When the sensitivity is high, small local changes to the parameters can lead to large improvements in the task loss".

**Intuición geométrica.** Imaginemos el espacio de parámetros. Cada tarea $\mathcal{T}_i$ tiene su propio óptimo $\theta_i^*$. El pretraining convencional busca un punto que minimice la pérdida promedio sobre todas las tareas; pero ese punto puede estar lejos de *todos* los óptimos individuales y, peor, en una región donde el gradiente no apunta hacia ninguno. MAML busca un $\theta$ desde el cual **un solo paso de gradiente $\nabla\mathcal{L}_i$ caiga cerca de cada $\theta_i^*$**. No es el centroide de los óptimos: es el punto de máxima *adaptabilidad direccional*.

La única suposición sobre el modelo: que esté parametrizado por un vector $\theta$ y que la pérdida sea suficientemente suave en $\theta$ para usar técnicas basadas en gradiente.

---

## El algoritmo: inner loop y outer loop

MAML tiene dos niveles de optimización anidados.

**Inner loop (adaptación).** Al adaptarse a una tarea $\mathcal{T}_i$, los parámetros $\theta$ se convierten en $\theta_i'$ mediante uno (o más) pasos de descenso de gradiente. Con un paso:

$$
\theta_i' = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_\theta)
$$

donde $\alpha$ es el step size del inner loop (fijo o meta-aprendido).

**Outer loop (meta-objetivo).** Los parámetros se entrenan optimizando el desempeño del modelo *ya adaptado* $f_{\theta_i'}$, respecto a $\theta$, a través de las tareas:

$$
\min_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}\big(f_{\theta_i'}\big) = \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}\big(f_{\theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_\theta)}\big)
$$

El punto sutil: **la meta-optimización se hace sobre $\theta$, pero el objetivo se evalúa con los parámetros actualizados $\theta_i'$**. La meta-actualización vía SGD:

$$
\theta \leftarrow \theta - \beta \nabla_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}\big(f_{\theta_i'}\big)
$$

con $\beta$ el meta step size. El esqueleto del algoritmo:

```text
inicializar θ aleatoriamente
while not done:
  muestrear batch de tareas T_i ~ p(T)
  for all T_i:
    evaluar ∇θ L_{T_i}(f_θ) con K ejemplos del support set D
    θ'_i = θ − α ∇θ L_{T_i}(f_θ)
    muestrear query set D'_i de T_i para el meta-update
  θ ← θ − β ∇θ Σ_i L_{T_i}(f_{θ'_i}) usando cada D'_i
```

La estructura clave es la **separación support/query**: el support $D$ (los $K$ ejemplos) se usa para *adaptar*; el query $D'_i$, disjunto, se usa para evaluar la pérdida del meta-objetivo. Si ambos fueran el mismo conjunto, MAML podría aprender un $\theta$ que sobreajusta trivialmente esos $K$ puntos en un paso. Al exigir que la pérdida post-update se mida en un conjunto disjunto, se fuerza a que la adaptación *generalice* dentro de la tarea — la misma lógica de un set de validación, internalizada en el bucle de entrenamiento.

El framework es general porque la tarea se define de forma amplia: en aprendizaje supervisado i.i.d. la longitud de episodio es $H=1$ (entrada → salida); en RL, $H>1$ y la tarea es un MDP completo. La misma maquinaria de adaptación sirve para ambos.

---

## La derivada de segundo orden y FOMAML

Aquí aparece la complejidad computacional real: la meta-actualización "involves a gradient through a gradient". Con un paso de inner loop, derivar $\mathcal{L}_{\mathcal{T}_i}(f_{\theta_i'})$ donde $\theta_i' = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_\theta)$ requiere la regla de la cadena:

$$
\nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_{\theta_i'}) = \big(I - \alpha \nabla_\theta^2 \mathcal{L}_{\mathcal{T}_i}(f_\theta)\big)\, \nabla_{\theta_i'} \mathcal{L}_{\mathcal{T}_i}(f_{\theta_i'})
$$

El término $\frac{\partial \theta_i'}{\partial \theta} = I - \alpha \nabla_\theta^2 \mathcal{L}_{\mathcal{T}_i}$ contiene el **Hessiano** de la pérdida del inner loop. Computacionalmente exige "an additional backward pass through $f$ to compute Hessian-vector products", soportado por librerías con diferenciación automática.

**El truco first-order (FOMAML).** Si se *ignora* el término Hessiano —se aproxima $I - \alpha \nabla^2 \approx I$—, el meta-gradiente se reduce a evaluar simplemente el gradiente de la pérdida en los parámetros post-actualización:

$$
\nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_{\theta_i'}) \approx \nabla_{\theta_i'} \mathcal{L}_{\mathcal{T}_i}(f_{\theta_i'})
$$

El hallazgo sorprendente: FOMAML rinde "nearly the same" que MAML completo. En MiniImagenet 5-way: **48.07 ± 1.75%** (FOMAML) vs **48.70 ± 1.84%** (MAML) en 1-shot, y **63.15 ± 0.91%** vs **63.11 ± 0.92%** en 5-shot — estadísticamente indistinguibles. La explicación: las redes ReLU son "locally almost linear" (Goodfellow et al., 2015), así que las segundas derivadas son cercanas a cero y $I - \alpha\nabla^2 \approx I$ es buena aproximación. La ganancia práctica: eliminar los Hessian-vector products produjo un **speed-up de ~33%**.

Conceptualmente, el factor $(I - \alpha \nabla_\theta^2 \mathcal{L}_{\mathcal{T}_i})$ es el Jacobiano de la transformación $\theta \mapsto \theta_i'$: precondiciona la dirección de mejora según la curvatura local. FOMAML renuncia a ese precondicionamiento y se conforma con apuntar $\theta$ hacia donde la pérdida post-update es baja. Que ambos coincidan es una afirmación fuerte sobre la geometría ReLU, no una trivialidad: con activaciones suaves o muchos pasos de inner loop, la brecha reaparece. En RL, el meta-gradiente exacto implicaría hasta *terceras* derivadas, así que el paper usa **diferencias finitas** para los Hessian-vector products de TRPO.

---

## Resultados (Omniglot, MiniImagenet, regresión sinusoidal, RL)

El paper plantea tres preguntas: ¿permite MAML aprendizaje rápido? ¿sirve en múltiples dominios? ¿puede seguir mejorando con más pasos?

**Regresión sinusoidal.** Cada tarea regresa una onda seno con amplitud en $[0.1, 5.0]$ y fase en $[0, \pi]$; pérdida MSE; MLP de 2 capas ocultas de 40 con ReLU. El MSE 5-shot a 1/5/10 pasos de gradiente:

| Método | 1 paso | 5 pasos | 10 pasos |
|---|---|---|---|
| pretrain on all tasks | 2.41 | 2.23 | 2.19 |
| **MAML** | **0.67** | **0.38** | **0.35** |

MAML mejora un orden de magnitud. Resultado crucial: cuando los $K$ puntos están todos en *una mitad* del rango de entrada, MAML aún infiere amplitud y fase en la otra mitad — ha aprendido la **estructura periódica** de la onda. Y sigue mejorando con más pasos pese a haber sido entrenado para máximo desempeño tras *un* paso.

**Clasificación (Omniglot y MiniImagenet).** Arquitectura siguiendo a Vinyals et al. (2016): 4 módulos de convoluciones 3×3 con 64 filtros, batch norm, ReLU y max-pooling. Intervalos de confianza al 95%:

| Omniglot | 5-way 1-shot | 5-way 5-shot | 20-way 1-shot | 20-way 5-shot |
|---|---|---|---|---|
| matching nets (2016) | 98.1% | 98.9% | 93.8% | 98.5% |
| **MAML** | **98.7 ± 0.4%** | **99.9 ± 0.1%** | **95.8 ± 0.3%** | **98.9 ± 0.2%** |

| MiniImagenet | 5-way 1-shot | 5-way 5-shot |
|---|---|---|
| matching nets (2016) | 43.56 ± 0.84% | 55.31 ± 0.73% |
| meta-learner LSTM (2017) | 43.44 ± 0.77% | 60.60 ± 0.71% |
| **MAML** | **48.70 ± 1.84%** | **63.11 ± 0.92%** |

MAML supera (estrechamente en Omniglot, con margen claro en MiniImagenet) a métodos diseñados *específicamente* para clasificación, sin introducir parámetros extra.

**RL (navegación 2D y locomoción).** Política: MLP de 2 capas ocultas de 100 con ReLU; inner loop con REINFORCE, meta-optimización con TRPO. En half-cheetah (forward/backward), retorno promedio:

| pasos de grad | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| context vector | −40.49 | −44.08 | −38.27 | −42.50 |
| **MAML** | −50.69 | **293.19** | **313.48** | **315.65** |

MAML salta de −50.69 (sin adaptar) a 293.19 tras *un solo* paso de policy gradient; el baseline no despega. Que la misma receta inner/outer absorba pérdidas tan distintas como una MSE de un seno y una recompensa acumulada de un robot simulado es la justificación empírica más fuerte del adjetivo "model-agnostic".

Un hallazgo lateral elocuente: promediar en el *espacio de parámetros* (500 regresores entrenados por separado y promediados) rinde MSE 2.91, peor que el pretraining — confirmando que MAML aprende algo más sofisticado que el vector de parámetros óptimo medio. El promedio de óptimos en un paisaje no convexo cae entre cuencas; MAML elige específicamente un punto cuya vecindad bajo un paso de gradiente intersecta las cuencas de todas las tareas.

---

## Por qué importa hoy

MAML detonó una línea de investigación entera y dejó dos lentes para entender su mecanismo: **aprendizaje de features** (entrenar para que pocos pasos basten equivale a construir una representación interna ampliamente reutilizable, afinando sobre todo las capas superiores) y **sensibilidad** (deja $\theta$ en una región donde pequeños cambios producen grandes mejoras).

Las extensiones más relevantes:

- **Reptile** (Nichol et al., 2018): aproximación de primer orden aún más simple, mueve $\theta$ hacia los pesos resultantes de SGD por tarea ($\theta \leftarrow \theta + \epsilon(\theta_i' - \theta)$), sin support/query separados.
- **ANIL — Almost No Inner Loop** (Raghu et al., 2020): demuestra que el inner loop solo necesita adaptar la *última capa*; las features apenas cambian. Confirma la hipótesis de **feature reuse** del propio MAML.
- **Meta-SGD** (Li et al., 2017): meta-aprende también el vector de step sizes $\alpha$ por parámetro.
- **iMAML** (Rajeswaran et al., 2019): usa el teorema de la función implícita para computar el meta-gradiente sin desplegar el grafo del inner loop, desacoplando el costo del número de pasos.
- **MAML++ / LEO** (2018–2019): atacan la inestabilidad práctica catalogando patologías de entrenamiento o meta-aprendiendo en un espacio latente de baja dimensión.

Conceptualmente, MAML estableció el meta-aprendizaje **basado en optimización** como una de las tres familias canónicas, junto a la métrica (Prototypical/Matching Networks) y la basada en modelos/memoria. Y, en retrospectiva, "una inicialización que se adapta con pocos pasos de gradiente" es la misma intuición que sustenta el éxito del **fine-tuning eficiente de modelos fundacionales**: un punto de partida bien elegido (un LLM o un ViT preentrenado) que se especializa con poquísimos ejemplos. MAML formalizó esa propiedad años antes de que se volviera el paradigma dominante.

Las limitaciones que la literatura confirmó: costo de segundo orden, costo de muestreo on-policy en RL (cada paso de adaptación exige rollouts nuevos), inestabilidad del bi-nivel anidado, y fuerte sensibilidad a hiperparámetros — la generalidad "model-agnostic" no significa "hyperparameter-free".

---

## Conexión con la Clase 26

MAML es la piedra angular del **meta-aprendizaje basado en gradiente** y ocupa el lugar del método "puente": combina la generalidad de los enfoques RNN/memoria con la simplicidad operativa del fine-tuning. Es el contraste pedagógico ideal frente a los métodos métricos (Prototypical Networks, Matching Networks) y frente a los meta-learners aprendidos (Ravi & Larochelle). La distinción que vale internalizar: *no aprende a aprender produciendo una regla nueva, sino preparando un punto de partida desde el cual la regla de siempre (SGD) aprende rápido*.

**Relevancia para salud.** El cuello de botella en imagenología médica casi nunca es el cómputo: es la **escasez de datos anotados** para condiciones raras, protocolos nuevos o subpoblaciones específicas. Meta-entrenar sobre el conjunto de tareas frecuentes (cada patología u órgano como una "tarea") produce una inicialización que se adapta a la entidad rara con $K$ ejemplos sin sobreajustar — la propiedad demostrada en el experimento sinusoidal y en Omniglot 1-shot. Lo mismo aplica a la adaptación entre sitios o escáneres (domain shift), tratando cada sitio como una tarea. Y la virtud de "merely produces a weight initialization" es operativamente atractiva: la adaptación es un fine-tuning estándar, auditable y reproducible, sin componentes meta-aprendidos opacos que compliquen la validación regulatoria. La advertencia, también del paper: la inestabilidad y la sensibilidad a hiperparámetros exigen validación cuidadosa antes de cualquier uso clínico.

---

## Notas y enlaces

Ver fundamentos: [Meta-aprendizaje](/fundamentos/meta-aprendizaje) - [Optimización binivel](/fundamentos/optimizacion-binivel) - [Few-shot learning](/fundamentos/few-shot-learning) - [Transfer learning](/fundamentos/transfer-learning).

Papers relacionados: [Prototypical Networks (Snell 2017)](/papers/prototypical-networks-snell-2017) - [Matching Networks (Vinyals 2016)](/papers/matching-networks-vinyals-2016) - [Meta-Learning Survey (Hospedales 2020)](/papers/meta-learning-survey-hospedales-2020) - [MetaSeg (Vyas 2025)](/papers/metaseg-vyas-2025).

Clase: Ver [Clase 26 -- Meta-aprendizaje](/clases/clase-26).
