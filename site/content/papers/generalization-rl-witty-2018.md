---
title: "Measuring and Characterizing Generalization in Deep RL (2018)"
weight: 378
math: true
---

{{< paper-card
    title="Measuring and Characterizing Generalization in Deep Reinforcement Learning"
    authors="Sam Witty, Jun Ki Lee, Emma Tosch, Akanksha Atrey, Michael Littman, David Jensen (UMass Amherst)"
    year="2018"
    venue="arXiv:1812.02868"
    pdf="/papers/generalization-rl-witty-2018.pdf" >}}
Hacia 2018 era cómodo suponer que un agente de RL profundo que juega Atari a nivel sobrehumano "entiende" su entorno y ha construido una representación generalizada. Witty et al. muestran que esa impresión puede estar profundamente equivocada. Su aporte no es solo **medir** que un agente falla —una métrica agregada—, sino ofrecer un aparato para **caracterizar dónde y cómo** falla un agente aparentemente competente. Formalizan tres grados de generalización (repetición, **interpolación** off-policy y **extrapolación** a estados no alcanzables) y, sobre un simulador parametrizable del juego Amidar, demuestran que una DQN de vanguardia toma decisiones pésimas ante cambios mínimos, no adversariales y semánticamente plausibles: rellenar **un solo segmento de línea** puede reducir su recompensa en un **orden de magnitud**. Es la slide 14 que abre la sección de generalización de la [Clase 33](/clases/clase-33).
{{< /paper-card >}}

---

## Contexto: por qué evaluar generalización en RL es engañoso

En aprendizaje supervisado la receta es canónica: separar entrenamiento de prueba y evaluar sobre datos no vistos. Witty et al. argumentan que trasladar esa receta al RL es engañoso por dos razones estructurales. **Primera:** los datos de entrenamiento dependen de la política del agente —la experiencia acumulada está determinada por la propia $\pi$ mientras evoluciona, así que omitir un subconjunto de la experiencia solo evalúa el uso de datos ya recolectados e ignora el efecto de la exploración. **Segunda:** la inmensidad del espacio de estados vuelve casi seguro encontrar estados nuevos al desplegar.

La respuesta es reformular la pregunta: definir la generalización como propiedad del **agente entrenado tratado como entidad autónoma**, agnóstica a los datos que encontró. Para saber cómo se comportará en zonas del espacio de estados que debería saber manejar, hay que exponerlo a estados que **nunca pudo haber observado**. El trabajo se distingue de los [ataques adversariales](/papers/overfitting-rl-zhang-2018): aquí las intervenciones no son adversariales y operan sobre el **estado latente** (la semántica real del juego), no sobre la percepción del agente. Es la base del [problema de generalización en RL](/fundamentos/generalizacion-en-rl).

## Contribución: partición formal del espacio de estados

Sobre el MDP estándar $\langle S, A, T, R \rangle$, y con $\alpha, \delta, \beta$ pequeños, se define $S_{reachable}$ como los estados que el agente encuentra con probabilidad mayor que $\alpha$ bajo alguna política. De ahí, tres grados de generalización en correspondencia con la **interpolación** y **extrapolación** de las tareas de predicción:

- **Repetición ($G_R$):** desempeño alto sobre los estados **on-policy** $S_{on}$ (los que el agente visita ejecutando su propia $\pi$). Es la generalización más débil: lo que se obtiene evaluando en el mismo entorno de entrenamiento.
- **Interpolación ($G_I$):** desempeño alto sobre los estados **off-policy** $S_{off} = S_{reachable} \setminus S_{on}$, que el agente *podría* encontrar bajo otra política pero no visita bajo la suya.
- **Extrapolación ($G_E$):** desempeño alto sobre los estados **no alcanzables** $S_{unreachable} = S \setminus S_{reachable}$: estados válidos según $T$ pero que ninguna política produce desde los estados iniciales.

Un cuerpo enorme de trabajo usa implícitamente $G_R$ como criterio, la capacidad más débil. El error de generalización se resume en dos mediciones: el **error de estimación de valor** $\text{VEE}_\pi(s) = \hat{v}(s) - v_\pi(s)$ y la **recompensa total acumulada** $\text{TAR}_\pi(s) = \mathbb{E}_\pi[\sum_{k=1}^{\infty} R(s_{t+k}) \mid s_t = s, a_t = a]$, normalizada contra **agentes de referencia** entrenados en cada escenario.

## Método: INTERVENIDAR y las intervenciones controladas

El eje empírico es Intervenidar, una implementación **completamente parametrizada** del Atari 2600 Amidar (tipo Pac-Man). Al ser parametrizable, permite manipular el estado latente —posiciones, existencia y relleno de los 88 segmentos de línea, comportamiento de enemigos— sin tocar el código fuente. Todos los agentes usan la arquitectura de vanguardia: **dueling network**, pérdida **double Q** y **prioritized experience replay** (hiperparámetros de Mnih et al., OpenAI Baselines).

Los **estados off-policy** se generan por estocasticidad (*k off-policy actions*, $k \in \{10,20\}$), *human starts* y *agent swaps*. Los **estados no alcanzables** requieren intervenir el estado latente antes del inicio de la partida: enemy removal, enemy shift, add line segment, fill line segments (FLS) y player random start (PRS). La virtud decisiva es el **control**: variar un único componente latente permite aislar la causa de la fragilidad, evitando el confundimiento de los estados off-policy (donde varios componentes cambian a la vez). Es esto lo que permite informar *dónde* fallan los agentes, no solo *que* fallan.

## Resultados: dónde y cómo fallan los agentes

1. **Generalización pobre.** La DQN completamente entrenada es **excepcionalmente frágil** a cambios pequeños. Los casos más flagrantes son FLS y PRS: la inspección visual muestra que el agente **permanece predominantemente inmóvil**, terminando el episodio sin recorrer un solo segmento. Es un modo de fallo cualitativo y concreto, no una caída abstracta de un número.
2. **Distancia en la representación anti-correlacionada con generalización.** Cuanto más lejos está el estado de prueba de lo visto (en el espacio de la última capa), peor generaliza. El agente **no "reconoce"** los estados donde peor generaliza. Además VEE y TAR están fuertemente anti-correlacionadas: el modelo **siempre sobreestima** el valor de los estados off-policy y no alcanzables.
3. **Volumen, capacidad y exploración con efectos menores y contraintuitivos.** Más entrenamiento aumenta la TAR de control pero tiempos *más cortos* generalizan algo mejor; **reducir la capacidad a veces mejora** la generalización; y los *exploring starts* (30–50 acciones aleatorias al inicio) fueron lo que **más ayudó** —el agente con 50 casi duplica su recompensa en human starts y en toda condición de extrapolación.
4. **Generalizar ≠ transferir representaciones.** Los agentes que mejor generalizan no acumulan más recompensa tras congelar las convolucionales y reentrenar con un protocolo alternativo de enemigos, lo que **contradice** el patrón conocido en visión por computador.

## Limitaciones

El propio marco reconoce sus fronteras. La metodología de estados no alcanzables **exige un simulador parametrizable** con acceso al estado latente; Intervenidar es un caso de estudio, no una herramienta general. Los estados off-policy sufren de **confundimiento** (varios componentes varían a la vez), y generar estados que *de verdad* difieran de la política inspeccionada no es trivial. La caracterización es intrínsecamente **empírica y específica del dominio**, sin garantías teóricas de transferencia. El trabajo se concentra en métodos **basados en valores** (DQN); no se extiende automáticamente a policy-gradient. Y los autores no ofrecen una solución a la fragilidad, sino un aparato para **diagnosticarla**.

## Por qué importa para la Clase 33

En la narrativa de la [Clase 33](/clases/clase-33) este paper **abre el argumento** de que el RL profundo sobreajusta y que la generalización debe medirse con cuidado. Antes de discutir aprendizaje por imitación o RL inverso —donde el objetivo es recuperar o copiar un comportamiento experto—, conviene entender que un agente que se ve competente puede no haber aprendido nada transferible: su "competencia" puede ser $G_R$ pura, memorización de la trayectoria de entrenamiento disfrazada de comprensión. El aporte duradero es doble: **(a)** la distinción **interpolación (off-policy) vs. extrapolación (no alcanzable)**, que da un vocabulario preciso para hablar de *qué tipo* de generalización se está midiendo; y **(b)** la insistencia en **caracterizar** los modos de fallo en lugar de reportar un único número agregado.

Su relación con [Zhang et al. (2018)](/papers/overfitting-rl-zhang-2018) es directa y citada: ambos enmarcan la generalización como el problema de evitar el sobreajuste a un entorno particular. Zhang et al. documentan que las DQN memorizan; Witty et al. complementan ese diagnóstico con una **partición formal de estados** e intervenciones controladas sobre el estado latente. La relación con [Cobbe et al. (2019)](/papers/quantifying-generalization-cobbe-2019) es de continuidad: donde Witty et al. aportan **profundidad diagnóstica** —caracterizar dónde y cómo se rompe un único agente mediante intervenciones quirúrgicas—, Cobbe et al. aportan **amplitud estadística** —cuantificar la brecha promediando sobre distribuciones de niveles procedimentales. Juntos definen los dos ejes de la evaluación moderna de la [generalización en RL](/fundamentos/generalizacion-en-rl): caracterización controlada y cuantificación a escala.
