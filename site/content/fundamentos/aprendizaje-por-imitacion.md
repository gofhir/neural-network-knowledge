---
title: "Aprendizaje por Imitación"
weight: 111
math: true
---

El **aprendizaje por imitación** (imitation learning) es la estrategia de enseñarle a un agente a resolver una tarea **mostrándole cómo la resuelve un experto**, en lugar de dejarlo aprender por ensayo y error como en el [aprendizaje reforzado](/fundamentos/aprendizaje-reforzado). La idea es tan antigua como intuitiva —así aprenden los humanos a conducir, cocinar o operar— y tiene una ventaja enorme sobre el RL puro: es **rápido** y **seguro**, porque el agente no necesita explorar acciones peligrosas ni esperar recompensas diferidas. Este fundamento acompaña a la [Clase 33](/clases/clase-33), donde la imitación aparece como una alternativa directa al [aprendizaje reforzado inverso](/fundamentos/aprendizaje-reforzado-inverso): en vez de inferir la recompensa y *después* resolver el RL, ¿por qué no aprender la política directamente copiando al experto?

---

## 1. La idea central: convertir el control en aprendizaje supervisado

El [IRL](/fundamentos/aprendizaje-reforzado-inverso) usa demostraciones para aprender la política del experto $\pi^*(a\mid s)$ en **dos pasos**: primero infiere una recompensa $R$ a partir de las demostraciones, y luego resuelve un problema de RL para obtener $\pi^*$ bajo esa $R$. La pregunta natural que plantea la clase es: *si ya tenemos las demostraciones, ¿no sería más fácil aprender $\pi^*(a\mid s)$ directamente imitando al experto?*

La respuesta es el **behavioral cloning** (clonación de comportamiento). El procedimiento es sorprendentemente simple:

1. Dejamos que expertos resuelvan la tarea y **registramos qué acción ejecutan en cada estado**.
2. Con esos pares $(s, a)$ construimos un **conjunto de entrenamiento supervisado**: la entrada es el estado, la etiqueta es la acción del experto.
3. Entrenamos una política $\pi_\theta(a\mid s)$ como un **clasificador** (acciones discretas) o **regresor** (acciones continuas) estándar, minimizando la pérdida entre la acción predicha y la del experto.

$$
\min_\theta \; \mathbb{E}_{(s,a)\sim \mathcal{D}_{\text{experto}}} \big[\, \mathcal{L}(\pi_\theta(s),\, a)\,\big].
$$

Con esto, todo el arsenal del aprendizaje supervisado —descenso de gradiente, redes profundas, regularización— se aplica al problema del control. Una vez entrenada, $\pi_\theta(a\mid s)$ **debería generalizar** a situaciones nuevas, igual que un clasificador de imágenes generaliza a fotos que no vio en entrenamiento.

{{< concept-alert type="clave" >}}
El behavioral cloning **reduce el control secuencial a un problema de clasificación supervisada**: estado → acción. Esta es su gran fortaleza (simplicidad, velocidad, herramientas maduras) y también la raíz de su gran debilidad, porque el control **no es** clasificación i.i.d.: las decisiones de la política determinan qué estados verá después, algo que el aprendizaje supervisado estándar no contempla.
{{< /concept-alert >}}

---

## 2. Por qué funciona (bastante bien)

En la práctica, el aprendizaje por imitación **funciona bastante bien**. El ejemplo más espectacular que cita la clase es **Gato** (Reed et al., 2022), un único transformer de 1.2 mil millones de parámetros —con los *mismos* pesos— que resuelve 604 tareas distintas: jugar Atari, subtitular imágenes, conversar y apilar bloques con un brazo robótico real. Gato es **behavioral cloning a escala masiva**: convierte texto, imágenes, propiocepción y acciones en una única secuencia de *tokens* y se entrena de forma autorregresiva para predecir el siguiente token, imitando demostraciones de expertos. No usa la recompensa para mejorar; solo aprende a copiar.

La razón de fondo por la que la imitación es atractiva es doble:

- **Velocidad de aprendizaje.** El agente no explora a ciegas: recibe directamente ejemplos de buen comportamiento. Donde el RL puede necesitar millones de episodios de ensayo y error, la imitación aprende de un dataset fijo en una sola pasada supervisada.
- **Seguridad.** No hace falta que el agente ejecute acciones peligrosas durante el entrenamiento (un requisito prohibitivo en robótica, conducción autónoma o medicina).

---

## 3. Por qué falla: distribution shift y compounding error

La clase es honesta: la imitación *"tampoco funciona excelente"*. El aprendizaje por imitación tiende a fallar por **dos motivos principales**, y ambos vienen del mismo defecto conceptual: **el control no es un problema i.i.d.**

El problema es el **cambio de distribución** (*distribution shift*). El clasificador se entrena sobre los estados que **visita el experto**, pero en el momento de actuar, la política visita los estados que **ella misma genera**. Apenas comete un pequeño error, aterriza en un estado ligeramente distinto de los que vio en entrenamiento; ahí su predicción es un poco peor, lo que la lleva a un estado *aún más* raro, y así sucesivamente. Los errores **se acumulan** (*compounding error*): el agente se aleja progresivamente de la distribución de datos del experto hacia una región donde nunca fue entrenado y no sabe qué hacer.

Ross et al. (2011) formalizaron esta intuición con una cota devastadora. Si la política comete errores con probabilidad $\epsilon$ sobre la distribución de estados del experto, entonces sobre un horizonte de $T$ pasos su costo total crece como

$$
\mathcal{O}(T^2 \epsilon),
$$

es decir, **cuadráticamente en el horizonte**, no linealmente. Un error pequeño por paso se amplifica de forma catastrófica en tareas largas. Este es el resultado teórico que explica los dos modos de fallo que muestra la clase.

{{< concept-alert type="advertencia" >}}
La lección central: en imitación pura hay que **procurar que $\pi_\theta(a\mid s)$ se mantenga dentro de su zona de entrenamiento**. En cuanto la política se sale de la distribución de estados que vio el experto, no hay garantía alguna sobre su comportamiento. El aprendizaje supervisado asume que train y test vienen de la misma distribución; en control, la propia política rompe ese supuesto.
{{< /concept-alert >}}

---

## 4. DAgger: cerrar el bucle con el experto

La solución más influyente al distribution shift es **DAgger** (Dataset Aggregation, Ross et al. 2011), el algoritmo que implementa el laboratorio de la Clase 33 sobre Atari Breakout. La idea es **recolectar datos de entrenamiento precisamente en los estados que la política visita**, no solo en los del experto:

1. Entrena una política inicial $\hat\pi_1$ por behavioral cloning sobre las demostraciones del experto.
2. **Ejecuta la política actual** en el ambiente y recolecta los estados que efectivamente visita.
3. **Consulta al experto** qué acción tomaría *en esos estados visitados* (aunque la política ya no esté imitándolo perfectamente).
4. **Agrega** esos nuevos pares $(s, a_{\text{experto}})$ al dataset acumulado.
5. **Reentrena** la política sobre el dataset agregado y repite.

Al recolectar etiquetas expertas sobre la distribución de estados de la *propia política*, DAgger cierra el bucle y elimina el distribution shift. Su análisis como **aprendizaje online sin arrepentimiento** (no-regret online learning, vía Follow-The-Leader) garantiza que el error crece solo **linealmente** $\mathcal{O}(T\epsilon)$ en vez de cuadráticamente. En la práctica, DAgger convierte una política que se estrella en una que iguala al experto tras pocas iteraciones. Su costo: requiere un **experto interactivo consultable durante el entrenamiento**, algo no siempre disponible (aunque en el lab el "experto" es un DQN previamente entrenado, siempre disponible para consulta).

{{< concept-alert type="recordar" >}}
DAgger puede entenderse como **active learning dirigido al control**: en lugar de etiquetar datos al azar, se etiquetan exactamente los estados donde la política se está equivocando —los que ella misma alcanza—, que son los más informativos para corregir su trayectoria.
{{< /concept-alert >}}

---

## 5. RL, IRL e imitación: cuándo usar cada uno

La Clase 33 cierra con una comparación directa, cuyo caso de estudio es **AlphaGo Zero** (Silver et al., 2017): un sistema que, sin usar *ninguna* partida humana, aprende Go solo por auto-juego (RL puro) y **supera** a todas las versiones anteriores de AlphaGo que sí partían de imitación de partidas humanas. Esto ilustra el trade-off fundamental:

| | Aprendizaje reforzado | Aprendizaje por imitación |
|---|---|---|
| **Demostraciones** | No las necesita ✓ | Las requiere ✗ |
| **Techo de desempeño** | Puede superar a los humanos ✓ | Rara vez supera al maestro ✗ |
| **Velocidad** | Aprende muy lento ✗ | Aprende una buena política rápido ✓ |
| **Seguridad** | Explora acciones inseguras ✗ | No requiere exploración peligrosa ✓ |

La síntesis de la clase: el **RL** aprende de forma autónoma y puede alcanzar desempeño sobrehumano, pero es lento y peligroso durante el entrenamiento; la **imitación** aprende rápido y seguro, pero no suele superar a su maestro y necesita demostraciones. Por eso muchos sistemas reales **combinan ambos**: arrancar con imitación (para partir de una política razonable rápido y seguro) y luego refinar con RL (para superar el techo humano). AlphaGo 2016 hizo exactamente esto —*supervised learning* de partidas humanas seguido de RL por auto-juego— antes de que AlphaGo Zero demostrara que, en un dominio con modelo perfecto del mundo, el RL puro basta y sobra.

---

## 6. Relevancia para MDM y record linkage

Para quien construye un sistema de **matching de pacientes**, el aprendizaje por imitación es el marco más directo: los *data stewards* generan a diario miles de decisiones de match/no-match que son, literalmente, un dataset supervisado (par de registros → decisión experta). Entrenar un clasificador sobre esos pares es **behavioral cloning**. Pero la lección de la Clase 33 es la advertencia crítica: **el distribution shift también acecha aquí**. Un modelo entrenado con los casos que los stewards revisaron manualmente —típicamente los ambiguos o los que el sistema marcó— fallará sobre la distribución de casos que el sistema en producción realmente encuentra, que es distinta. La estrategia estilo DAgger es la respuesta correcta: hacer que el sistema procese datos reales, identificar los casos donde *duda* o *se equivoca*, y pedir a un steward que los etiquete para reentrenar. Es active learning dirigido a los estados que el sistema visita —exactamente lo que evita que el error se acumule.

---

## Referencias

- Ross, S., Gordon, G. & Bagnell, J.A. (2011). *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (DAgger). AISTATS.
- Reed, S. et al. (2022). *A Generalist Agent* (Gato). TMLR / arXiv:2205.06175.
- Silver, D. et al. (2017). *Mastering the Game of Go without Human Knowledge* (AlphaGo Zero). Nature.
- Pomerleau, D. (1991). *Efficient Training of Artificial Neural Networks for Autonomous Navigation* (ALVINN) — origen histórico del behavioral cloning.
