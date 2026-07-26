---
title: "¿El RL incentiva el razonamiento más allá del modelo base? (2025)"
weight: 391
math: true
---

{{< paper-card
    title="Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?"
    authors="Yang Yue et al. (Tsinghua)"
    year="2025"
    venue="arXiv:2504.13837"
    pdf="/papers/rl-reasoning-yue-2025.pdf" >}}
Este paper es la **nota de precaución** frente al entusiasmo por el RL con recompensa verificable (RLVR) tras [DeepSeek-R1](/papers/deepseek-r1-2025). La creencia dominante era que, como el RL de Atari o Go descubre estrategias nuevas por exploración, el RLVR permitiría a los LLMs adquirir capacidades de razonamiento **genuinamente nuevas**. Yue et al. la ponen a prueba con un instrumento distinto: **pass@k evaluado a valores grandes de $k$**. El hallazgo es sobrio y contraintuitivo: en $k$ pequeño el modelo RL supera al base, pero al crecer $k$ las **curvas se cruzan** y el base lo iguala y supera. El RLVR **estrecha** la distribución hacia caminos que el base ya podía generar —reordena y afila el muestreo— pero **no expande la frontera** de razonamiento. La [destilación](/papers/deepseek-r1-2025), en cambio, sí. Es la nota de precaución de la [Clase 34](/clases/clase-34) (slide 36) sobre los límites del [test-time compute](/fundamentos/test-time-compute) inducido por RL.
{{< /paper-card >}}

---

## Contexto: el hype de RLVR después de R1

El motor del salto reciente de los "modelos de razonamiento" —o1, DeepSeek-R1, Kimi-1.5— fue el RLVR a gran escala. La receta es de una simplicidad seductora: partir de un modelo base y optimizarlo por RL con una **recompensa verificable automáticamente**. Sea $\pi_\theta$ un LLM que genera $y$ condicionado a un prompt $x$, y un verificador determinista $V$ que devuelve $r = V(x,y) \in \{0,1\}$; el objetivo es la política que maximiza la recompensa esperada:

$$J(\theta) = \mathbb{E}_{x\sim D}\,\mathbb{E}_{y\sim \pi_\theta(\cdot\mid x)}\,[\,r\,].$$

El razonamiento por analogía que alimentó el optimismo es explícito: en el RL de juegos los agentes descubren estrategias nuevas por auto-mejora (AlphaGo Zero, DQN). Se asumió que el RLVR haría lo mismo con el lenguaje. La pregunta que da título al paper: **¿el RLVR genuinamente hace adquirir capacidades nuevas, o simplemente reutiliza patrones que ya viven en el modelo base?**

## Método / Contribución

La contribución clave es **cambiar el instrumento de medición**. La evaluación habitual —greedy o promedio de *nucleus sampling*— refleja el caso promedio y **subestima el potencial** del modelo. Para medir la **frontera de capacidad** —los problemas que el modelo *puede potencialmente* resolver— los autores extienden **pass@k** a todas las tareas verificables: dado un problema, se muestrean $k$ salidas y pass@k vale $1$ si **al menos una** pasa la verificación. Usan el estimador insesgado de Chen et al. (2021):

$$\text{pass@}k := \mathbb{E}_{x_i\sim D}\left[\,1 - \frac{\binom{n-c_i}{k}}{\binom{n}{k}}\,\right].$$

La lógica es limpia: si el RL realmente **expandiera** la capacidad, el modelo RL debería resolver *más problemas* que el base a cualquier $k$ —en particular resolver problemas que el base no puede ni con muchísimos intentos. Esta misma métrica es la que [Large Language Monkeys](/papers/large-language-monkeys-brown-2024) explota como palanca de inferencia: escalar $k$ revela cobertura latente del modelo base.

El estudio es deliberadamente amplio para blindar la conclusión: cubre **matemática** (Qwen2.5, LLaMA-3.1, con contrapartes zero-RL vía GRPO; GSM8K, MATH500, AIME24, AMC23), **código** (LiveCodeBench, HumanEval+, MBPP+) y **razonamiento visual** (Qwen2.5-VL en MathVista/MathVision). Dos salvaguardas atienden aciertos espurios: inspección manual de las CoT en problemas difíciles (los aciertos del base **no son suerte**, son razonamiento real: 24 de 25 con CoT válida en GSM8K) y, en código, el compilador con tests hace pass@k intrínsecamente confiable.

## Resultados

- **El cruce de curvas.** En $k$ pequeño el RL va por delante (mejor eficiencia de muestreo). Pero las curvas del base son **más empinadas**: a $k$ de decenas o centenas, el base **supera** al RL en todos los benchmarks y familias. En Minerva (32B) el base supera al RL en ~9% en $k=128$. Con Oat-Zero y DAPO el RL arranca ~30% por encima a $k$ bajo pero es **finalmente sobrepasado**.
- **La frontera se estrecha con el entrenamiento.** A medida que avanza el RL, pass@1 sube de 26.1 a 42.5, pero **pass@256 disminuye**: la cobertura de problemas resolubles se contrae. El RL **compra** exactitud promedio al costo de estrechar el rango de problemas resolubles.
- **El RL crea más problemas irresolubles.** El análisis de distribución muestra que RLVR aumenta la frecuencia de exactitudes cercanas a $1.0$ pero también **aumenta la de exactitud exactamente $0$**. La mejora del promedio no viene de resolver problemas nuevos, sino de muestrear con más eficiencia los que el base ya resolvía.
- **Cobertura: el RL es casi un subconjunto del base.** En AIME24 ($k=1024$), el caso "el base falla pero el RL resuelve" es **0.0%**; en MATH500 apenas **1.0%**. En cambio "el base resuelve pero el RL falla" es 13.3% y 3.6%.
- **Perplejidad.** Las respuestas del modelo RL tienen, bajo el base, la perplejidad de las que el base ya generaría con **alta probabilidad**: el RLVR **afila la distribución dentro del prior del base** en lugar de expandirse fuera de él.
- **Todos los algoritmos rinden parecido.** PPO, GRPO, Reinforce++, RLOO, ReMax y DAPO dan una brecha de eficiencia de muestreo casi idéntica (>40 puntos): ninguno explota el potencial completo del base.

## Limitaciones

- **Modelos propietarios fuera de alcance** (el base de o1 no es accesible; R1-Zero sin API pública era impráctico de evaluar a pass@k).
- **El experimento cerca de la frontera es preliminar** (un solo modelo, Magistral-Medium).
- **Preguntas abiertas de escala:** queda por ver si escalar el cómputo de RL hasta presupuestos de pre-entrenamiento superaría al base.
- **pass@k mide potencial, no utilidad práctica:** en producción rara vez se muestrean 256 respuestas, y ahí la eficiencia del RL sí importa. El paper no niega el valor práctico del RLVR; niega que equivalga a **expandir la capacidad**.

## Por qué importa para la Clase 34

Este paper es el **contrapeso deliberado** al entusiasmo. En la [Clase 34](/clases/clase-34), Amenábar lo trae (slide 36) para matizar: la evidencia apunta a que el RL **no le enseña comportamientos nuevos** al modelo —ni la verificación, ni la autorreflexión, ni el "momento ajá"— sino que esos patrones fueron aprendidos en el pre-entrenamiento y el RL simplemente los **muestrea con mayor probabilidad**. La clase presenta primero el arco optimista de [DeepSeek-R1](/papers/deepseek-r1-2025) y luego usa a Yue et al. como el freno: distinguir entre "el modelo aprendió a razonar mejor" y "el modelo aprendió a *mostrar* con más frecuencia el razonamiento que ya tenía".

La imagen mental correcta no es la del RL clásico que *explora* y descubre jugadas nunca vistas, sino la de una **reponderación** de una distribución preexistente: el pre-entrenamiento dota al modelo de un vasto repertorio latente; el RLVR sube la probabilidad de los patrones recompensados y baja la de los demás, sacrificando diversidad y con ella cobertura a $k$ alto. **RLVR y destilación son fundamentalmente distintos**: la destilación inyecta patrones nuevos de un maestro más fuerte y su curva pass@k queda por encima del base a *todo* $k$ —sí supera la frontera. La lección para el [test-time compute](/fundamentos/test-time-compute): escalar el muestreo (pass@k, [Large Language Monkeys](/papers/large-language-monkeys-brown-2024)) revela lo que el base ya puede; el RL afina cuál de esas respuestas emerge primero, pero no amplía el conjunto de lo posible.
