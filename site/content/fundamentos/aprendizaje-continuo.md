---
title: "Aprendizaje Continuo y Olvido Catastrófico"
weight: 109
math: true
---

El **aprendizaje continuo** (continual learning, también lifelong learning) estudia cómo un modelo puede aprender de un flujo de datos que llega **a lo largo del tiempo** —tareas, clases o dominios que aparecen en secuencia— sin reentrenar desde cero ni acumular todos los datos pasados. Es el régimen natural de cualquier modelo en producción: un clasificador médico al que se le incorporan patologías nuevas, un detector de fraude que enfrenta esquemas que mutan, un recomendador que ve productos que no existían el mes anterior. El obstáculo que define el campo tiene nombre propio desde fines de los años 80: el **olvido catastrófico** (catastrophic forgetting, McCloskey y Cohen 1989; French 1999). Este fundamento recorre por qué olvidan las redes, las tres grandes familias de soluciones —regularización, memoria/replay y arquitectura—, cómo se miden, y por qué, pese a tres décadas de trabajo, sigue sin estar resuelto. Es el fundamento núcleo de la [Clase 32](/clases/clase-32).

---

## 1. El problema del olvido catastrófico

Una red neuronal entrenada secuencialmente en una serie de tareas **pierde de forma abrupta** —no gradual— el desempeño sobre las tareas previas a medida que aprende las nuevas. La causa mecánica es directa: la red comparte un único conjunto de parámetros entre todas las tareas, y el descenso de gradiente que optimiza la tarea B **sobreescribe** los pesos que eran importantes para la tarea A. El gradiente de B no "sabe" nada de A; nada le impide destruir la solución de A para minimizar la pérdida de B. El resultado visible: al cambiar de la tarea A a la B, el desempeño en B sube empinadamente mientras el de A cae en picada, y el deterioro empeora con cada tarea añadida.

El núcleo del problema es el **dilema estabilidad-plasticidad** (stability-plasticity dilemma), tomado de la neurociencia. Un sistema de aprendizaje necesita dos propiedades en tensión: **plasticidad** suficiente para adquirir conocimiento nuevo, y **estabilidad** suficiente para no destruir el conocimiento viejo. Demasiada plasticidad y todo se sobreescribe (olvido catastrófico); demasiada estabilidad y la red no aprende nada nuevo (rigidez). Todo el campo del aprendizaje continuo puede leerse como distintas formas de negociar este dilema.

{{< concept-alert type="clave" >}}
El olvido catastrófico no es un capricho de las redes: es la **consecuencia directa de optimizar solo la tarea presente** sobre parámetros compartidos. Si el único objetivo es minimizar la pérdida de los datos de hoy, el gradiente moverá libremente los pesos que ayer resolvían otra cosa. Mitigarlo siempre implica introducir, de algún modo, **información sobre el pasado** en el proceso de optimización: ya sea protegiendo pesos, repitiendo datos o aislando capacidad.
{{< /concept-alert >}}

Por qué importa en producción: el paradigma cómodo del **multitask learning** —entrelazar (interleave) los datos de todas las tareas y optimizarlas conjuntamente— elimina el olvido por construcción, pero exige tener **todos los datos disponibles simultáneamente**. En el mundo real los datos llegan en el tiempo, el acceso a los datos viejos puede estar restringido (privacidad, costo de almacenamiento, datos propietarios o ya no registrados), y reentrenar el modelo completo cada vez es caro. El aprendizaje continuo es precisamente el estudio de qué hacer cuando no se puede recurrir al multitask ingenuo.

---

## 2. ¿Qué hacer cuando llegan datos nuevos?

Antes de los métodos sofisticados, conviene entender las tres respuestas ingenuas a "llegaron datos nuevos, ¿qué hago?", porque cada familia avanzada es una forma de comprar las ventajas de una sin pagar sus costos.

- **Reentrenar con todo** (joint / multitask training). Se guardan todos los datos históricos y se reentrena el modelo desde cero con el conjunto completo. Es la **cota superior** de desempeño y no sufre olvido. Pero el almacenamiento crece sin límite, el tiempo de entrenamiento crece con cada lote nuevo, y **requiere acceso permanente a todos los datos viejos** —imposible bajo restricciones de privacidad o cuando los datos legacy se perdieron.

- **Entrenar solo con los datos nuevos** (fine-tuning secuencial). Se toma el modelo actual y se lo ajusta solo con lo nuevo. Es barato en espacio y rápido, pero es exactamente la receta del **olvido catastrófico**: es la cota inferior contra la que se mide todo lo demás.

- **Un modelo nuevo por lote/tarea**. Se entrena un modelo independiente para cada conjunto de datos. No hay olvido (cada modelo conserva su tarea) pero el almacenamiento crece linealmente con el número de tareas, **no hay transferencia** de conocimiento entre tareas, y en inferencia hay que saber qué modelo usar.

| Estrategia | Espacio | Tiempo | Acceso a datos viejos | Transferencia | Olvido |
|---|---|---|---|---|---|
| Reentrenar con todo | Crece (todos los datos) | Crece con cada lote | Requiere todos | Sí (conjunta) | Nulo |
| Solo datos nuevos | Constante | Bajo | No requiere | Limitada | **Catastrófico** |
| Un modelo por tarea | Crece (un modelo/tarea) | Constante por tarea | No requiere | **Nula** | Nulo |

La pregunta que organiza el campo: ¿cómo lograr el **no-olvido** de "reentrenar con todo" y la **transferencia** que da compartir representaciones, con el **footprint acotado** de "solo datos nuevos"? Ninguna familia lo consigue del todo; cada una sacrifica algo distinto.

---

## 3. Los tres escenarios (van de Ven y Tolias)

No todos los problemas de aprendizaje continuo tienen la misma dificultad, y el factor que más la determina **no es el dataset, sino qué información sobre la identidad de la tarea está disponible en test**. La taxonomía canónica de [van de Ven y Tolias (2019)](/papers/three-scenarios-van-de-ven-2019) deriva de una sola pregunta operativa —¿se entrega el *task-ID* en test?, y si no, ¿hay que inferirlo?— tres escenarios de dificultad creciente. Se volvió el estándar de facto: cualquier paper moderno especifica bajo cuál evalúa.

- **Task-Incremental Learning (Task-IL).** El modelo **siempre sabe qué tarea debe resolver** en test. Es el más fácil: permite componentes específicos por tarea, típicamente una capa de salida *multi-head* (una cabeza por tarea). Ejemplo en split MNIST: "dado que esta es la tarea X, ¿es la primera o la segunda clase?".

- **Domain-Incremental Learning (Domain-IL).** El *task-ID* **no está disponible**, pero el modelo solo necesita resolver la tarea actual, no identificar cuál es. La estructura de las tareas es siempre la misma; cambia la distribución de entrada (mismo problema, distinto dominio). Ejemplo: un agente que debe sobrevivir en distintos entornos sin necesidad de saber en cuál está.

- **Class-Incremental Learning (Class-IL).** El **más difícil**. El modelo debe resolver todas las tareas vistas **y además inferir con cuál está siendo confrontado**: distinguir entre todas las clases acumuladas sin que nadie le diga a qué grupo pertenece la entrada. Ejemplo: "tarea desconocida, ¿qué dígito es?" eligiendo entre las 10 clases 0–9, habiéndolas aprendido de a dos. Es el escenario de máxima relevancia práctica: un clasificador médico en inferencia no sabe a qué "tarea" pertenece el caso, debe elegir entre todas las patologías que aprendió.

El hallazgo empírico que da fuerza a la taxonomía —y que adelantamos aquí porque organiza el resto del fundamento— es que en **Class-IL los métodos de regularización colapsan al nivel del azar** mientras solo el replay funciona. La dificultad no es académica: define qué familia de métodos es viable.

---

## 4. Familia 1 — Regularización

**Idea:** usar la red completa para todas las tareas, pero **penalizar el cambio de los pesos importantes** para las tareas previas. En vez de congelar nada de forma dura, se añade a la pérdida un término que actúa como un resorte elástico: cada peso queda anclado a su valor anterior con una fuerza proporcional a cuán importante fue. Pesos cruciales para el pasado quedan casi rígidos; pesos irrelevantes quedan libres para reaprender.

**EWC — Elastic Weight Consolidation** ([Kirkpatrick et al. 2017](/papers/ewc-kirkpatrick-2017)). El método fundacional de la familia, con inspiración neurocientífica directa (la consolidación sináptica que protege el conocimiento en el neocórtex mamífero). Al entrenar la tarea B, EWC minimiza la pérdida de B más una penalización cuadrática que jala cada peso hacia su óptimo en A, ponderada por la **diagonal de la matriz de información de Fisher** $F_i$:

$$
\mathcal{L}(\theta) = \mathcal{L}_B(\theta) + \sum_i \frac{\lambda}{2}\, F_i\,(\theta_i - \theta^*_{A,i})^2.
$$

La diagonal de Fisher estima la importancia de cada peso (aproxima la curvatura de la pérdida) y se calcula solo con derivadas de primer orden, lo que la hace barata. Tiene además una lectura bayesiana limpia: **el posterior de la tarea A se convierte en el prior para aprender B**. Su debilidad: la aproximación diagonal ignora las correlaciones entre pesos, y la estimación puntual subestima la incertidumbre.

**Synaptic Intelligence (SI)** ([Zenke et al. 2017](/papers/synaptic-intelligence-zenke-2017)). El hermano gemelo de EWC. Comparte exactamente la misma forma de penalización cuadrática $\sum_k \Omega_k (\theta_k - \tilde\theta_k)^2$, pero estima la importancia de **otra fuente**: en vez de calcularla post-hoc en el mínimo (como la Fisher de EWC), la acumula **online a lo largo de toda la trayectoria de entrenamiento**, integrando cuánto contribuyó cada peso a reducir la pérdida —sin pases de backpropagation adicionales. La lección clave: misma penalización, distinta fuente de importancia (Fisher post-hoc vs. integral de camino online).

**LwF — Learning without Forgetting** ([Li y Hoiem 2016](/papers/lwf-li-2016)). Una variante *funcional*: en vez de regularizar los parámetros, regulariza la **salida**. Antes de entrenar la tarea nueva, registra las predicciones del modelo viejo sobre los datos nuevos; durante el entrenamiento, una pérdida de **destilación de conocimiento** (la misma de la Clase 20) obliga a que esas salidas se mantengan estables mientras una cross-entropy aprende lo nuevo. Su gran virtud: no necesita un solo dato de las tareas viejas. Su límite: depende de que los datos nuevos activen representaciones útiles de las viejas (falla bajo *domain shift* fuerte).

La familia de regularización tiene un **footprint constante** (no guarda datos ni crece) y es la más respetuosa de la privacidad. Su talón de Aquiles aparece en la sección 7.

---

## 5. Familia 2 — Memoria / Replay

**Idea:** guardar un pequeño conjunto de ejemplos de tareas pasadas y **reproducirlos** (replay / rehearsal) al entrenar las nuevas, recreando aproximadamente el régimen multitask. Es la familia inspirada en los *complementary learning systems* del cerebro (el hipocampo repasa experiencias para consolidarlas en el neocórtex).

**Experience Replay.** La forma más simple: mantener un *buffer* de ejemplos viejos y mezclarlos en cada mini-batch con los datos nuevos. Funciona sorprendentemente bien, pero reentrenar directamente sobre pocos ejemplos guardados tiende a **sobreajustarlos**.

**GEM — Gradient Episodic Memory** ([Lopez-Paz y Ranzato 2017](/papers/gem-lopez-paz-2017)). El giro elegante de la familia: usa la memoria $M_k$ de cada tarea **no para reentrenar, sino como restricciones de desigualdad sobre el gradiente**. En cada paso proyecta el gradiente propuesto para que no forme un ángulo obtuso con los gradientes de ninguna tarea pasada, garantizando que la actualización **no aumente la pérdida** en ellas. Si el gradiente ya cumple $\langle g, g_k\rangle \ge 0$ para todo $k$, se aplica tal cual; si no, se resuelve un programa cuadrático pequeño (en el dual, sobre $t-1$ variables, no sobre los millones de parámetros) para hallar el gradiente factible más cercano. Como las restricciones permiten que la pérdida pasada *baje*, GEM habilita incluso **transferencia positiva hacia atrás** (aprender lo nuevo mejora lo viejo). Su sucesor eficiente, A-GEM (2019), promedia todos los gradientes de memoria en una sola restricción con fórmula cerrada.

**iCaRL — Incremental Classifier and Representation Learning** ([Rebuffi et al. 2017](/papers/icarl-rebuffi-2017)). El *baseline* de facto del class-incremental, y el método que **fusiona memoria y regularización** en un solo sistema con tres componentes: (1) un conjunto de **exemplars** por clase, seleccionados por *herding* (greedy, de modo que su media de features aproxime la media de la clase) bajo un presupuesto de memoria fijo $K$; (2) una pérdida que combina **clasificación + destilación** estilo LwF al incorporar clases nuevas; y (3) una regla de clasificación por **nearest-mean-of-exemplars (NME)** que reemplaza al softmax. La NME es la clave de su robustez: el prototipo $\mu_y = \frac{1}{|P_y|}\sum_{p\in P_y}\varphi(p)$ es función directa de la representación $\varphi$, así que cuando $\varphi$ cambia al aprender clases nuevas, los prototipos se mueven con ella —sin pesos rezagados que provoquen olvido descontrolado.

La familia de replay logra el mejor desempeño en los escenarios difíciles, al costo de **almacenar datos** (problemático bajo privacidad) y de una memoria que crece o se reduce con el número de tareas.

---

## 6. Familia 3 — Arquitectura

**Idea:** **aislar los parámetros de cada tarea** (parameter isolation), de modo que aprender una nueva no toque los pesos de las viejas. El olvido se vuelve **imposible por construcción**, no se mitiga: si los pesos viejos no cambian, no hay nada que sobreescribir.

**Progressive Neural Networks** ([Rusu et al. 2016](/papers/progressive-nets-rusu-2016)). El método arquitectónico seminal. Por cada tarea nueva instancia una **columna** de red, **congela** las columnas anteriores (de ahí la inmunidad total al olvido) y las conecta a la nueva mediante **conexiones laterales** que permiten transferir features. El precio: el número de parámetros **crece** (linealmente en unidades, cuadráticamente en parámetros laterales) y no escala a muchas tareas.

**PiggyBack** ([Mallya et al. 2018](/papers/piggyback-mallya-2018)). Resuelve el crecimiento de Progressive Nets: toma una red preentrenada y **congelada**, y para cada tarea aprende solo una **máscara binaria por peso** (un bit que enciende o apaga cada conexión). Los pesos base nunca se tocan, así que cero olvido; cada tarea cuesta solo ~1 bit por parámetro (un ahorro de 32–64× frente a guardar un modelo completo). La máscara binaria, aunque no diferenciable, se entrena con un *straight-through estimator*.

**SupSup — Supermasks in Superposition** ([Wortsman et al. 2020](/papers/supsup-wortsman-2020)). Generaliza PiggyBack en dos ejes. Primero, las máscaras se aprenden sobre una red **aleatoria fija** (vía la *lottery ticket hypothesis*: dentro de una red al azar ya existen subredes que resuelven la tarea), así que basta guardar la semilla. Segundo —y crucial—, **infiere el task-ID cuando no se provee**: superpone todas las máscaras ponderadas por coeficientes $\alpha_i$ y, por gradiente, busca la combinación que **minimiza la entropía** de la salida (la máscara correcta produce la predicción más confiada). Rompe así la atadura de PiggyBack/PackNet/HAT al task-ID conocido.

**HAT — Hard Attention to the Task** ([Serra et al. 2018](/papers/hat-serra-2018)). Aprende, por tarea y vía backpropagation, **máscaras de atención casi binarias sobre las unidades** de cada capa (no sobre pesos: las de peso se derivan). Un *embedding* de tarea pasa por un *gate* sigmoide cuyo factor de escala se va recociendo (*annealing*) hasta volver las máscaras duras. Las máscaras acumuladas modulan el gradiente: los pesos importantes para tareas previas quedan congelados. El factor $s_{\max}$ es literalmente una perilla del dilema estabilidad-plasticidad.

**L2P — Learning to Prompt** ([Wang et al. 2022](/papers/l2p-wang-2022)). El método moderno que cierra el campo. Parte de un **Transformer preentrenado y congelado** y guarda el conocimiento de cada tarea en un pequeño **pool de prompts** (vectores aprendibles, ~0.1% de parámetros extra). Un mecanismo *query-key* selecciona, **por instancia y sin conocer la tarea**, qué prompts anteponer a la secuencia de entrada. Sintetiza dos hilos del curso —prompting (Clase 20) y Transformers— y supera a los métodos de replay **incluso sin buffer de memoria**, funcionando en class-incremental y hasta en escenarios sin fronteras de tarea.

Esta familia ofrece no-olvido garantizado, pero la mayoría (salvo SupSup y L2P) **requiere el task-ID en inferencia**, lo que las ata al escenario Task-IL.

---

## 7. Comparación de las tres familias

| | **Regularización** | **Memoria / Replay** | **Arquitectura** |
|---|---|---|---|
| Qué guarda | Importancia de pesos + valores ancla | Exemplars o gradientes de memoria | Máscaras / columnas / prompts por tarea |
| ¿Requiere task-ID en test? | No (Domain-IL) | No | Sí (salvo SupSup, L2P) |
| ¿Crece con las tareas? | No (footprint constante) | Sí (memoria crece o $m$ baja) | Sí (Progressive); casi no (máscaras/prompts) |
| Privacidad | Buena (no guarda datos) | Mala (guarda datos crudos) | Buena (no guarda datos) |
| Olvido | Mitigado | Mitigado (fuerte en Class-IL) | Nulo por construcción |
| Escenario donde brilla | Task-IL, Domain-IL | Class-IL | Task-IL (todos con L2P) |
| Ejemplos | EWC, SI, LwF | GEM, iCaRL, Experience Replay | Progressive, PiggyBack, SupSup, HAT, L2P |

El **hallazgo central** del campo, demostrado por [van de Ven y Tolias](/papers/three-scenarios-van-de-ven-2019) sobre split y permuted MNIST: en **Task-IL todos los métodos funcionan** (>98%), pero en **Class-IL los métodos de regularización colapsan al nivel del azar** (~20% = 1/5, igual que el fine-tuning ingenuo), mientras solo el **replay** supera el 90%.

| Método (split MNIST) | Task-IL | Domain-IL | Class-IL |
|---|---|---|---|
| Fine-tuning (cota inferior) | 87.2% | 59.2% | 19.9% |
| EWC | 98.6% | 64.0% | **20.0%** |
| SI | 99.1% | 65.4% | **20.0%** |
| Replay generativo (DGR+distill) | 99.6% | 96.8% | **91.8%** |
| iCaRL | — | — | **94.6%** |

La lectura es nítida: **la regularización no basta cuando hay que inferir la tarea** (porque penalizar el cambio de pesos no ayuda a distinguir entre clases nunca vistas juntas), y en ese régimen —el más realista— el **replay es necesario**. Este resultado reorientó buena parte de la investigación posterior hacia los métodos de memoria.

---

## 8. Métricas

Evaluar aprendizaje continuo requiere más que una accuracy final: hay que medir **transferencia y olvido**. El marco estándar lo fijó [GEM](/papers/gem-lopez-paz-2017) construyendo una matriz $R \in \mathbb{R}^{T\times T}$, donde $R_{i,j}$ es la accuracy de test en la tarea $j$ después de terminar de aprender la tarea $i$.

$$
\text{ACC} = \frac{1}{T}\sum_{i=1}^{T} R_{T,i}, \qquad
\text{BWT} = \frac{1}{T-1}\sum_{i=1}^{T-1}\left(R_{T,i} - R_{i,i}\right), \qquad
\text{FWT} = \frac{1}{T-1}\sum_{i=2}^{T}\left(R_{i-1,i} - \bar b_i\right).
$$

- **ACC** (average accuracy): accuracy promedio sobre todas las tareas al final del entrenamiento. Es el número titular.
- **BWT** (backward transfer): cuánto afectó aprender tareas posteriores al desempeño en una tarea previa. **BWT muy negativo es la definición operativa del olvido catastrófico**; BWT positivo significa que aprender lo nuevo *mejoró* lo viejo (lo que GEM logra, +0.025, frente a EWC −0.165 en MNIST permutado).
- **FWT** (forward transfer): cuánto ayuda aprender una tarea al desempeño en una tarea *futura* aún no vista; FWT positivo es una forma de *zero-shot* y solo se consigue explotando estructura en los descriptores de tarea.

Entre dos modelos con ACC similar, se prefiere el de mayor BWT y FWT. En el escenario class-incremental se usa además la **average incremental accuracy** (promedio de accuracies tras cada lote de clases, protocolo de iCaRL), y las **matrices de confusión** sirven de diagnóstico: el fine-tuning colapsa hacia el último lote (olvido puro), la representación congelada sesga hacia el primero (rigidez), e iCaRL produce una matriz homogénea (balance).

---

## 9. Conexión con el curso y panorama

El aprendizaje continuo no vive aislado: reutiliza piezas vistas a lo largo del curso. La **destilación de conocimiento** de la Clase 20 reaparece como mecanismo de regularización en LwF e iCaRL (preservar las salidas del modelo viejo como sustituto barato de los datos viejos). El **prompting** y los **Transformers preentrenados** —también de la Clase 20 y del módulo de NLP— son la base de L2P, que reformula el problema entero: en vez de proteger pesos, guarda el conocimiento *fuera* de los pesos y lo recupera por contenido. La **atención** se recicla en HAT, ya no para ponderar tokens sino para particionar la capacidad de la red entre tareas. Y EWC nació en el grupo del DQN de DeepMind, probándose en **aprendizaje por refuerzo** sobre Atari secuencial —el mismo linaje que el RLHF del curso.

El panorama, sin embargo, es honesto: **el aprendizaje continuo no es un problema resuelto**. El survey crítico de [Mundt et al. (2020)](/papers/continual-survey-mundt-2020) lo argumenta con fuerza: el campo se estrechó a "monitorear el olvido sobre benchmarks secuencializados" (MNIST/CIFAR partidos) bajo una **suposición de mundo cerrado**, ignorando lecciones de *active learning* (qué datos consultar) y *open set recognition* (cómo rechazar lo desconocido). Su evidencia empírica muestra que factores rutinariamente ignorados —el balanceo de mini-batches (~5% de brecha), la estrategia de selección de exemplars, la robustez a corrupciones de entrada, y el **orden de las tareas** (~10% de brecha)— cambian las conclusiones. La moraleja metodológica: importa tanto *cómo se evalúa* el olvido como *qué algoritmo* se usa, y ningún método domina en todos los regímenes.

**Resumen.** El olvido catastrófico surge de optimizar la tarea presente sobre pesos compartidos; combatirlo exige inyectar información del pasado vía una de tres familias —**regularización** (protege pesos importantes: EWC, SI, LwF), **memoria/replay** (repite datos o restringe gradientes: GEM, iCaRL), **arquitectura** (aísla parámetros: Progressive Nets, PiggyBack, SupSup, HAT, L2P)—. La dificultad la fijan los tres escenarios de van de Ven y Tolias, y el hallazgo duro es que en el más realista (class-incremental) la regularización falla y el replay es necesario. Se mide con ACC/BWT/FWT, y sigue abierto: la era de los modelos preentrenados (L2P) lo reformula, pero la robustez en el mundo abierto y la evaluación honesta quedan como frontera.

---

## Para profundizar

- [Three scenarios for continual learning (van de Ven y Tolias 2019)](/papers/three-scenarios-van-de-ven-2019) — la taxonomía Task/Domain/Class-IL y el colapso de la regularización en Class-IL.
- [Overcoming Catastrophic Forgetting (EWC, Kirkpatrick et al. 2017)](/papers/ewc-kirkpatrick-2017) — el método de regularización fundacional, importancia por Fisher.
- [Continual Learning Through Synaptic Intelligence (Zenke et al. 2017)](/papers/synaptic-intelligence-zenke-2017) — importancia online por integral de camino.
- [Learning without Forgetting (Li y Hoiem 2016)](/papers/lwf-li-2016) — regularización funcional vía destilación.
- [Gradient Episodic Memory (Lopez-Paz y Ranzato 2017)](/papers/gem-lopez-paz-2017) — gradientes de memoria como restricciones; métricas ACC/BWT/FWT.
- [iCaRL (Rebuffi et al. 2017)](/papers/icarl-rebuffi-2017) — exemplars + destilación + NME, el baseline del class-incremental.
- [Progressive Neural Networks (Rusu et al. 2016)](/papers/progressive-nets-rusu-2016) — columnas congeladas con conexiones laterales.
- [Piggyback (Mallya et al. 2018)](/papers/piggyback-mallya-2018) — máscaras binarias sobre una red congelada.
- [Supermasks in Superposition (Wortsman et al. 2020)](/papers/supsup-wortsman-2020) — supermáscaras sobre red aleatoria + inferencia de tarea por entropía.
- [Hard Attention to the Task (Serra et al. 2018)](/papers/hat-serra-2018) — atención dura por tarea a nivel de unidad.
- [Learning to Prompt for Continual Learning (Wang et al. 2022)](/papers/l2p-wang-2022) — pool de prompts sobre Transformer congelado.
- [A Wholistic View of Continual Learning (Mundt et al. 2020)](/papers/continual-survey-mundt-2020) — survey crítico: el panorama del campo y la crítica al mundo cerrado.

**Recurso del curso:** [Clase 32 — Olvido Catastrófico y Aprendizaje Continuo](/clases/clase-32)
