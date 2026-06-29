---
title: "GEM: Gradient Episodic Memory (2017)"
weight: 358
math: true
---

{{< paper-card
    title="Gradient Episodic Memory for Continual Learning"
    authors="David Lopez-Paz, Marc'Aurelio Ranzato"
    year="2017"
    venue="NeurIPS 2017"
    pdf="/papers/gem-lopez-paz-2017.pdf"
    arxiv="1706.08840" >}}
Paper fundacional del **aprendizaje continuo** moderno y el representante canónico de los **métodos basados en memoria** de la [Clase 32](/clases/clase-32). Hace dos aportes que el campo adoptó casi universalmente. Primero, formaliza el problema con tres **métricas** —ACC, BWT (*backward transfer*) y FWT (*forward transfer*)— que miden no solo el rendimiento sino la transferencia de conocimiento entre tareas. Segundo, propone GEM: mantiene una **memoria episódica $M_k$ por cada tarea $k$** y usa los **gradientes** de esa memoria como **restricciones** que proyectan el gradiente actual para que no aumente la pérdida en tareas pasadas, resolviendo un pequeño programa cuadrático (QP). El resultado más llamativo es que GEM no solo evita el olvido, sino que habilita **transferencia positiva hacia atrás** (mejorar tareas viejas al aprender nuevas), algo que EWC e iCaRL no lograban por diseño.
{{< /paper-card >}}

---

## Contexto: aprender de una secuencia de tareas

El aprendizaje supervisado clásico asume que cada ejemplo $(x_i, y_i)$ es una muestra iid de una distribución fija $P$ y minimiza el riesgo empírico (ERM) con múltiples pasadas sobre el set de entrenamiento. El aprendizaje humano no es así: observamos los datos como una secuencia ordenada, rara vez vemos el mismo ejemplo dos veces, memorizamos solo unos pocos datos y la secuencia abarca tareas distintas. Aplicar ERM ingenuamente a ese régimen produce el **olvido catastrófico** (*catastrophic forgetting*, McCloskey y Cohen 1989): el modelo olvida cómo resolver tareas pasadas tras exponerse a tareas nuevas.

La contribución de contexto de GEM es **endurecer el setting experimental** para acercarlo a lo humano. La literatura previa (Progressive Nets, EWC, iCaRL) trabajaba en un régimen cómodo: pocas tareas, muchos ejemplos por tarea, varias pasadas y una sola métrica. GEM propone lo contrario: **muchas tareas, pocos ejemplos por tarea, una sola pasada (cada ejemplo se ve una vez)** y métricas que capturan transferencia y olvido. Formalmente, el aprendiz observa un *continuum* de datos como tripletas $(x_i, t_i, y_i)$, donde $t_i$ es un **descriptor de tarea**. Los datos no son iid: cuando las tareas cambian, se observa toda la secuencia de ejemplos de la tarea nueva antes de pasar a la siguiente (se asume iid solo localmente, dentro de cada tarea).

GEM se ubica dentro de la taxonomía de tres familias de remedios contra el olvido que estructura la Clase 32:

1. **Métodos de arquitectura:** congelar capas y clonar/afinar otras, o estructuras modulares (Progressive Networks, PathNet). Limitación: difíciles de escalar por la explosión combinatoria de composiciones de módulos.
2. **Métodos de regularización ("memoria sináptica"):** EWC y Synaptic Intelligence penalizan cambios en los parámetros importantes para tareas anteriores. Un solo modelo, objetivo de entrenamiento modificado.
3. **Métodos de memoria episódica:** almacenan ejemplos de tareas previas. GEM pertenece a esta familia, pero introduce un giro decisivo sobre cómo usarlos.

## El dilema de cómo usar la memoria

Tener una memoria $M_k$ por tarea abre la pregunta central: **¿qué hago con esos ejemplos guardados?** El paper descarta explícitamente las opciones malas:

- **Re-entrenar con la memoria** (minimizar la pérdida actual junto con la pérdida en $M_k$). Como $|M_k|$ es pequeño, esto lleva a **sobreajustar los pocos ejemplos guardados** en vez de la tarea original.
- **Destilar para mantener predicciones invariantes** (como iCaRL/LwF). Esto **vuelve imposible la transferencia positiva hacia atrás**, porque congela el rendimiento pasado en su valor actual en lugar de permitir que mejore.

La innovación de GEM es usar las pérdidas de memoria $\ell(f_\theta, M_k)$ **no como términos del objetivo, sino como restricciones de desigualdad**: no permitir que aumenten, pero sí permitir que disminuyan. Esto evita el olvido (no aumentan) y a la vez **deja la puerta abierta a que el rendimiento pasado mejore** (pueden disminuir), habilitando BWT positivo.

## Las tres métricas: ACC, BWT, FWT

El protocolo construye una **matriz $R \in \mathbb{R}^{T\times T}$**, donde $R_{i,j}$ es la accuracy de test en la tarea $t_j$ tras terminar de aprender la tarea $t_i$. Sea $\bar{b}$ el vector de accuracies de cada tarea en la inicialización aleatoria. Se definen:

$$\text{ACC} = \frac{1}{T}\sum_{i=1}^{T} R_{T,i} \qquad \text{(accuracy promedio al final)}$$

$$\text{BWT} = \frac{1}{T-1}\sum_{i=1}^{T-1}\left(R_{T,i} - R_{i,i}\right) \qquad \text{(transferencia hacia atrás)}$$

$$\text{FWT} = \frac{1}{T-1}\sum_{i=2}^{T}\left(R_{i-1,i} - \bar{b}_i\right) \qquad \text{(transferencia hacia adelante)}$$

- **BWT** mide cómo aprender una tarea posterior afecta una tarea previa. BWT positivo = aprender lo nuevo mejoró lo viejo; BWT muy negativo *es* la definición operativa de olvido catastrófico.
- **FWT** mide cómo aprender una tarea afecta una tarea *futura* aún no vista; FWT positivo equivale a una forma de *zero-shot* y solo es posible explotando estructura en los descriptores de tarea.
- Cuanto más grandes, mejor. Entre dos modelos con ACC similar, se prefiere el de mayor BWT y FWT.

Estas tres métricas se volvieron el **estándar de facto** para evaluar aprendizaje continuo, y por sí solas justifican gran parte del impacto del paper.

## Método: el QP de proyección de gradientes

**La memoria episódica.** El aprendiz tiene un presupuesto total de $M$ localizaciones de memoria. Si el número de tareas $T$ es conocido, se asignan $m = M/T$ memorias por tarea; si es desconocido, $m$ se reduce gradualmente. Por simplicidad la memoria se llena con los **últimos $m$ ejemplos de cada tarea**.

**El problema con restricciones.** Al observar la tripleta $(x, t, y)$, GEM resuelve:

$$\min_\theta\; \ell(f_\theta(x, t), y) \quad \text{sujeto a}\quad \ell(f_\theta, M_k) \le \ell(f_\theta^{t-1}, M_k)\;\; \text{para todo } k < t$$

donde $f_\theta^{t-1}$ es el predictor al final del aprendizaje de la tarea $t-1$. En palabras: minimizar la pérdida en el ejemplo actual **sin que la pérdida en la memoria de ninguna tarea pasada supere el valor que tenía** al terminar esa tarea.

**Dos observaciones que lo hacen eficiente.** Primero, no hace falta guardar predictores viejos: basta garantizar que la pérdida pasada no suba tras cada actualización. Segundo, asumiendo **linealidad local** (válida en pasos pequeños) y memoria representativa, un aumento de pérdida en una tarea pasada se diagnostica por el **ángulo** entre el gradiente de esa tarea y la actualización propuesta. Las restricciones se reformulan como **producto interno no negativo** entre el gradiente actual $g$ y cada gradiente de memoria $g_k$:

$$\langle g, g_k\rangle \ge 0,\quad \text{para todo } k < t$$

Si todas se cumplen, $g$ es improbable que aumente la pérdida pasada y se aplica directamente. Si una o más se violan, hay olvido potencial y se **proyecta**.

**La proyección como QP.** Se busca el gradiente $\tilde{g}$ más cercano a $g$ (en norma $\ell_2$) que satisfaga todas las restricciones:

$$\min_{\tilde{g}}\; \tfrac{1}{2}\|g - \tilde{g}\|_2^2 \quad \text{sujeto a}\quad \langle\tilde{g}, g_k\rangle \ge 0\;\; \text{para todo } k < t$$

El primal tiene $p$ variables (el número de **parámetros de la red**, hasta millones). La maniobra crucial es pasar al **dual**: con $G = (g_1, \dots, g_{t-1})$ la matriz de gradientes de memoria, el dual es un QP en **solo $t-1$ variables, el número de tareas observadas** ($t-1 \ll p$). Resuelto para $v^\star$, el gradiente proyectado se recupera como $\tilde{g} = G^\top v^\star + g$ y el paso de SGD es $\theta \leftarrow \theta - \alpha\tilde{g}$. Geométricamente: proyectar el gradiente al cono donde no daña ninguna tarea pasada.

## A-GEM: la versión eficiente

El cuello de botella de GEM —reconocido por el propio paper— es que **cada iteración requiere un backward pass por cada tarea previa** para computar todos los $g_k$, más resolver el QP. El trabajo de seguimiento **A-GEM** (*Averaged GEM*, Chaudhry et al., ICLR 2019) simplifica esto: en vez de una restricción por tarea, **promedia los gradientes de memoria en una sola restricción** sobre un lote muestreado. La proyección se reduce a una fórmula cerrada (sin QP iterativo): si $\langle g, g_{\text{ref}}\rangle < 0$, se proyecta $\tilde{g} = g - \frac{g^\top g_{\text{ref}}}{g_{\text{ref}}^\top g_{\text{ref}}} g_{\text{ref}}$. A-GEM conserva la idea esencial —gradientes de memoria como restricción— a una fracción del costo, y se volvió la variante de referencia en la práctica.

## Experimentos

Tres benchmarks con $T = 20$ tareas: **MNIST Permutations** (cada tarea permuta los píxeles, distribuciones no relacionadas), **MNIST Rotations** (rotación fija por tarea) e **Incremental CIFAR-100** (subconjuntos disjuntos de clases). Cada ejemplo se ve **una sola vez**. Baselines: *single* (un predictor para todo), *independent* (uno por tarea), *multimodal*, EWC e iCaRL. Sobre MNIST permutations:

| Modelo | ACC | BWT | FWT |
|---|---|---|---|
| single | 0.6018 | -0.1980 | 0.0093 |
| EWC | 0.6185 | -0.1653 | 0.0054 |
| multimodal | 0.7561 | -0.0275 | 0.0059 |
| **GEM** | **0.8260** | **+0.0247** | 0.0088 |

GEM logra ACC 0.826 con **BWT positivo (+0.0247)**: no solo no olvida, sino que aprender tareas posteriores *mejoró* en promedio las previas. EWC, en cambio, sufre BWT -0.165, señal clara de olvido. Tres hallazgos adicionales: (1) GEM es **más rápido que EWC** en CPU, porque optimiza sobre $T=20$ variables en el dual y no sobre $p \approx 10^6$ parámetros; (2) en CIFAR-100 la ACC final de GEM **crece monótonamente con el tamaño de memoria** y **supera a iCaRL en todo el rango**; (3) con múltiples pasadas, los métodos sin memoria agravan el olvido, mientras GEM **iguala la cota oráculo** de un modelo entrenado con datos iid barajados de todas las tareas.

## Limitaciones reconocidas

- **No aprovecha descriptores de tarea estructurados:** al usar descriptores enteros, GEM no explota relaciones semánticas entre tareas y por eso no logra FWT positivo significativo (zero-shot).
- **Gestión de memoria ingenua:** guarda los últimos $m$ ejemplos por tarea, sin coresets; y la memoria total crece (o $m$ decrece) con el número de tareas, problema de escalabilidad en horizontes largos.
- **Costo computacional:** un backward pass por tarea pasada en cada iteración más el QP. Es exactamente lo que A-GEM resuelve después.
- **Asume el descriptor de tarea en test** (escenario *task-incremental*); los escenarios *domain-* y *class-incremental* son más exigentes.

## Por qué importa para la Clase 32

La [Clase 32](/clases/clase-32) (Olvido Catastrófico) organiza los remedios en tres familias —arquitectura, regularización y **memoria**— y GEM es el ejemplar destacado de la última. El mapeo es directo: la "memoria $M_k$ para cada tarea $k$" es literalmente la memoria episódica de GEM; el énfasis de la clase en **usar los datos en memoria para modificar los gradientes** es la esencia del método, que **no re-entrena** (sobreajustaría) ni **destila** (impediría BWT positivo) sino que convierte los gradientes de memoria en **restricciones geométricas** sobre la dirección del paso de SGD. El contraste con [EWC](/fundamentos/aprendizaje-continuo) es limpio: "proteger pesos" (regularización, BWT -0.165) frente a "proteger pérdidas pasadas" (memoria episódica, BWT +0.025). Un BWT muy negativo *es* el olvido catastrófico medido con las métricas de GEM.

GEM abrió además la subfamilia de **proyección de gradientes** (A-GEM, OGD, GPM), distinta tanto del *replay* puro como de la regularización, y demostró empíricamente que la **transferencia positiva hacia atrás es alcanzable**.

## Notas y enlaces

- Marco general (escenarios task/domain/class-incremental, dilema estabilidad-plasticidad, taxonomía completa): [/fundamentos/aprendizaje-continuo](/fundamentos/aprendizaje-continuo).
- Método de memoria rival basado en *replay* + destilación: [iCaRL](/papers/icarl-rebuffi-2017).
- arXiv:1706.08840 (v1 jun 2017). Código: github.com/facebookresearch/GradientEpisodicMemory.
- Autores: David Lopez-Paz y Marc'Aurelio Ranzato (Facebook AI Research). NeurIPS 2017, Long Beach.
