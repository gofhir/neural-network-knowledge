# Gradient Episodic Memory for Continual Learning — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Gradient Episodic Memory for Continual Learning* (GEM).
- **Autores:** David Lopez-Paz y Marc'Aurelio Ranzato (Facebook Artificial Intelligence Research, FAIR).
- **Venue:** *31st Conference on Neural Information Processing Systems* (NeurIPS / NIPS 2017), Long Beach, CA, USA.
- **Preprint:** arXiv:1706.08840 (v1 jun 2017; la versión consultada es v6, 13 sep 2022). [arxiv.org/abs/1706.08840](https://arxiv.org/abs/1706.08840).
- **Código:** [github.com/facebookresearch/GradientEpisodicMemory](https://github.com/facebookresearch/GradientEpisodicMemory).

GEM es uno de los papers fundacionales del **aprendizaje continuo** (*continual learning*, también *lifelong learning*) moderno, y es el representante canónico de la familia de **métodos basados en memoria** en la Clase 32 del curso. Su tesis arranca de un contraste: el aprendizaje supervisado clásico asume que cada ejemplo $(x_i, y_i)$ es una muestra iid de una distribución fija $P$, y minimiza el riesgo empírico (ERM) con múltiples pasadas sobre el set de entrenamiento. El aprendizaje humano no es así: observamos los datos como una secuencia ordenada, rara vez vemos el mismo ejemplo dos veces, memorizamos solo unos pocos datos, y la secuencia abarca tareas distintas. Aplicar ERM ingenuamente a ese régimen produce el **olvido catastrófico** (*catastrophic forgetting*, McCloskey y Cohen 1989): el modelo olvida cómo resolver tareas pasadas tras exponerse a tareas nuevas.

El paper hace **dos aportes que el campo adoptó casi universalmente**. Primero, formaliza el problema con un conjunto de **métricas** —ACC (accuracy promedio), BWT (*backward transfer*) y FWT (*forward transfer*)— que miden no solo el rendimiento sino la transferencia de conocimiento entre tareas. Segundo, propone GEM, un modelo que mantiene una **memoria episódica $M_k$ por cada tarea $k$** y la usa de una forma sutil: en lugar de re-entrenar con esos ejemplos (que llevaría a sobreajuste a la memoria), **usa los gradientes de la memoria como restricciones de desigualdad** que proyectan el gradiente de la tarea actual para que no aumente la pérdida en las tareas pasadas. Esta proyección se resuelve con un **programa cuadrático (QP)** pequeño. La consecuencia más llamativa: GEM no solo evita el olvido, sino que permite **transferencia positiva hacia atrás** (mejorar tareas viejas al aprender tareas nuevas), algo que los métodos rivales de la época (EWC, iCaRL) no lograban por diseño.

Para la Clase 32 (Olvido Catastrófico) esto importa porque GEM es exactamente el método que la clase presenta cuando habla de "memoria $M_k$ para cada tarea $k$" y de **usar los datos en memoria para modificar los gradientes**, en contraste con los métodos de regularización (EWC, SI) que penalizan cambios en parámetros importantes, y con los de arquitectura (Progressive Networks, PackNet) que asignan capacidad nueva por tarea. Ver el fundamento transversal en [/fundamentos/aprendizaje-continuo](/fundamentos/aprendizaje-continuo) y el hub de la clase en [/clases/clase-32](/clases/clase-32).

## 2. Contexto: aprendizaje continuo y la familia de métodos basados en memoria

El paper sitúa el aprendizaje continuo como el estudio de aprender a través de una **secuencia de tareas**, donde el aprendiz debe retener conocimiento de las tareas pasadas y aprovecharlo para adquirir habilidades nuevas más rápido. Esta tradición (Ring 1994; Thrun 1994, 1996, 1998; Thrun y Pratt 2012) había producido implementaciones y análisis teóricos, pero estos últimos restringidos a modelos lineales.

La contribución de contexto de GEM es **endurecer el setting experimental** para acercarlo a lo "humano". La literatura previa (Rusu et al. 2016 Progressive Nets; Kirkpatrick et al. 2017 EWC; Rebuffi et al. 2017 iCaRL) trabajaba en un régimen cómodo: pocas tareas, muchos ejemplos por tarea, varias pasadas por tarea, y una sola métrica (accuracy promedio). GEM propone lo contrario: **muchas tareas, pocos ejemplos por tarea, una sola pasada (cada ejemplo se ve una vez), y métricas que capturan transferencia y olvido**. Formalmente, el aprendiz observa un *continuum* de datos como tripletas $(x_i, t_i, y_i)$, donde $t_i \in \mathcal{T}$ es un **descriptor de tarea**. Los datos no son iid respecto de ninguna distribución fija sobre $(x, t, y)$: cuando las tareas cambian, se observa toda una secuencia de ejemplos de la tarea nueva antes de pasar a la siguiente. El paper asume que el continuum es **localmente iid** (dentro de cada tarea sí vale la iid).

¿Dónde encaja GEM dentro del taxón de métodos contra el olvido? El paper distingue tres grandes familias, que son también la espina dorsal de la Clase 32:

1. **Métodos de arquitectura / modulares:** congelar capas tempranas y clonar/afinar capas tardías (Oquab et al. 2014), o estructuras modulares con primitivas compartidas (Rusu et al. 2016; Fernando et al. 2017 PathNet). Limitación reconocida: difíciles de escalar por el número combinatorio de composiciones de módulos.
2. **Métodos de regularización ("memoria sináptica"):** EWC (Kirkpatrick et al. 2017) y Synaptic Intelligence (Zenke et al. 2017) ajustan las tasas de aprendizaje o penalizan cambios en los parámetros importantes para tareas anteriores. Un solo modelo, objetivo de entrenamiento modificado.
3. **Métodos de "memoria episódica":** almacenan ejemplos de tareas previas y los **reproducen** (*replay*) para mantener invariantes las predicciones, típicamente mediante destilación (Li y Hoiem 2016 LwF; Rebuffi et al. 2017 iCaRL; Jung et al. 2016; Rannen Triki et al. 2017).

GEM pertenece a la tercera familia —usa memoria episódica— pero introduce un giro decisivo sobre cómo usarla.

### 2.1. El dilema de cómo usar la memoria

Tener una memoria episódica $M_k$ por tarea abre la pregunta central de la familia basada en memoria: **¿qué hago con esos ejemplos guardados?** El paper razona explícitamente sobre las opciones malas antes de presentar la suya:

- **Opción ingenua — re-entrenar con la memoria.** Minimizar la pérdida en el ejemplo actual *junto con* la pérdida en la memoria $\ell(f_\theta, M_k) = \frac{1}{|M_k|}\sum_{(x_i,k,y_i)\in M_k}\ell(f_\theta(x_i,k), y_i)$ (ecuación 5 del paper). Problema: como $|M_k|$ es pequeño, esto lleva a **sobreajustar los pocos ejemplos guardados** en memoria, en vez de a la tarea original.
- **Opción por destilación — mantener predicciones invariantes.** Como en iCaRL/LwF, forzar que las predicciones en las tareas pasadas no cambien. Problema: esto **vuelve imposible la transferencia positiva hacia atrás**, porque congela el rendimiento pasado en su valor actual en lugar de permitir que mejore.

La innovación de GEM es usar las pérdidas de memoria $\ell(f_\theta, M_k)$ **no como términos del objetivo, sino como restricciones de desigualdad**: no permitir que aumenten, pero sí permitir que disminuyan. Esto evita el olvido (no aumentan) y a la vez **deja la puerta abierta a que el rendimiento pasado mejore** (pueden disminuir), habilitando BWT positivo.

## 3. Contribución central

### 3.1. Las tres métricas: ACC, BWT, FWT

El protocolo de evaluación construye una **matriz $R \in \mathbb{R}^{T\times T}$**, donde $R_{i,j}$ es la accuracy de test en la tarea $t_j$ después de haber terminado de aprender la tarea $t_i$. Sea $\bar{b}$ el vector de accuracies de test de cada tarea en la inicialización aleatoria. Se definen:

$$\text{ACC} = \frac{1}{T}\sum_{i=1}^{T} R_{T,i} \qquad \text{(accuracy promedio al final)}$$

$$\text{BWT} = \frac{1}{T-1}\sum_{i=1}^{T-1}\left(R_{T,i} - R_{i,i}\right) \qquad \text{(transferencia hacia atrás)}$$

$$\text{FWT} = \frac{1}{T-1}\sum_{i=2}^{T}\left(R_{i-1,i} - \bar{b}_i\right) \qquad \text{(transferencia hacia adelante)}$$

Interpretación:

- **BWT** mide cómo aprender una tarea posterior afecta el rendimiento en una tarea previa. BWT positivo = aprender lo nuevo mejoró lo viejo. BWT muy negativo = **olvido catastrófico** (es literalmente la definición operativa de "forgetting" en el paper).
- **FWT** mide cómo aprender una tarea afecta el rendimiento en una tarea *futura* aún no vista; FWT positivo equivale a una forma de *zero-shot* y solo es posible explotando estructura en los descriptores de tarea.
- Cuanto más grandes, mejor. Entre dos modelos con ACC similar, se prefiere el de mayor BWT y FWT.

Estas tres métricas se volvieron el **estándar de facto** para evaluar aprendizaje continuo, y por sí solas justifican gran parte del impacto del paper.

### 3.2. La idea de GEM en una frase

Mantener una memoria episódica $M_k$ por tarea $k$, y en cada paso de entrenamiento **proyectar el gradiente propuesto** para que no forme un ángulo obtuso con los gradientes de ninguna tarea pasada, garantizando así que la actualización no aumente la pérdida en ninguna de ellas. Cuando el gradiente ya cumple esa condición, se aplica tal cual; cuando la viola, se busca el gradiente más cercano (en norma $\ell_2$) que sí la cumpla, resolviendo un programa cuadrático.

## 4. Método: el programa cuadrático de proyección de gradientes

### 4.1. La memoria episódica

El aprendiz tiene un presupuesto total de $M$ localizaciones de memoria. Si el número de tareas $T$ es conocido, se asignan $m = M/T$ memorias por tarea; si es desconocido, se reduce $m$ gradualmente al observar tareas nuevas (estrategia de Rebuffi et al. 2017). Por simplicidad el paper puebla la memoria con los **últimos $m$ ejemplos de cada tarea**, aunque admite que estrategias mejores (como construir un *coreset* por tarea) son posibles.

### 4.2. El problema con restricciones

Al observar la tripleta $(x, t, y)$, GEM resuelve:

$$\min_\theta\; \ell(f_\theta(x, t), y) \quad \text{sujeto a}\quad \ell(f_\theta, M_k) \le \ell(f_\theta^{t-1}, M_k)\;\; \text{para todo } k < t \tag{6}$$

donde $f_\theta^{t-1}$ es el estado del predictor al final del aprendizaje de la tarea $t-1$. En palabras: minimizar la pérdida en el ejemplo actual **sin que la pérdida en la memoria de ninguna tarea pasada supere el valor que tenía** al terminar esa tarea.

### 4.3. Dos observaciones clave que lo hacen eficiente

1. **No hace falta guardar predictores viejos $f_\theta^{t-1}$**, basta con garantizar que la pérdida en tareas previas no suba tras cada actualización $g$.
2. Asumiendo **linealidad local** (válida en pasos de optimización pequeños) y que la memoria es representativa, los aumentos de pérdida en tareas pasadas se pueden diagnosticar por el **ángulo entre el gradiente de la tarea pasada y la actualización propuesta**.

Esto reformula las restricciones (6) como condiciones de **producto interno no negativo** entre el gradiente actual $g = \partial\ell(f_\theta(x,t),y)/\partial\theta$ y cada gradiente de memoria $g_k = \partial\ell(f_\theta, M_k)/\partial\theta$:

$$\langle g, g_k\rangle \ge 0,\quad \text{para todo } k < t \tag{7}$$

Si todas se cumplen, la actualización $g$ es improbable que aumente la pérdida en tareas pasadas, y se aplica directamente. Si una o más se violan, hay al menos una tarea pasada que sufriría olvido, y entonces se proyecta.

### 4.4. La proyección como QP

Cuando hay violaciones, se busca el gradiente $\tilde{g}$ más cercano a $g$ (en norma $\ell_2$ al cuadrado) que satisfaga todas las restricciones:

$$\min_{\tilde{g}}\; \tfrac{1}{2}\|g - \tilde{g}\|_2^2 \quad \text{sujeto a}\quad \langle\tilde{g}, g_k\rangle \ge 0\;\; \text{para todo } k < t \tag{8}$$

El primal de este QP tiene $p$ variables (el número de **parámetros de la red**, que puede ser de millones —por ejemplo $p = 1\,109\,240$ en la red de CIFAR-100). Resolverlo así sería prohibitivo. La maniobra crucial es pasar al **dual**: escribiendo $G = (g_1, \dots, g_{t-1})$ (la matriz de gradientes de memoria), el dual queda

$$\min_v\; \tfrac{1}{2}v^\top G G^\top v + g^\top G^\top v \quad \text{sujeto a}\quad v \ge 0 \tag{11}$$

que es un QP en **solo $t-1$ variables, el número de tareas observadas hasta ahora** ($t-1 \ll p$). Una vez resuelto para $v^\star$, el gradiente proyectado se recupera como $\tilde{g} = G^\top v^\star + g$, y el paso de SGD es $\theta \leftarrow \theta - \alpha\tilde{g}$. En la práctica, sumar una pequeña constante $\gamma \ge 0$ a $v^\star$ sesga la proyección hacia actualizaciones que favorecen la transferencia hacia atrás beneficiosa.

El Algoritmo 1 del paper resume el ciclo: por cada ejemplo, añadir a $M_t$, calcular $g$ y los $g_k$ de las tareas previas, proyectar vía (11), y actualizar $\theta$; tras cada tarea, evaluar y llenar la fila correspondiente de $R$.

### 4.5. Interpretación de "compresión causal"

El paper ofrece una lectura conceptual: GEM aprende el **subconjunto de correlaciones comunes a un conjunto de distribuciones (tareas)**. Esto es deseable en inferencia causal, donde las predicciones causales son invariantes a través de entornos (Peters et al. 2016) y proveen la representación más comprimida de un conjunto de distribuciones. De hecho, en MNIST GEM puede predecir sin usar el descriptor de tarea.

### 4.6. A-GEM (Averaged GEM): la versión eficiente posterior

El cuello de botella de GEM —reconocido por el propio paper— es que **cada iteración requiere un backward pass por cada tarea previa** para computar todos los $g_k$, más resolver el QP. Un trabajo de seguimiento muy citado, **A-GEM** (Chaudhry, Ranzato, Rohrbach y Elhoseiny, *Efficient Lifelong Learning with A-GEM*, ICLR 2019), simplifica esto drásticamente: en lugar de mantener una restricción por tarea, **promedia los gradientes de memoria en una sola restricción** sobre un lote muestreado de la memoria total. La proyección se reduce entonces a una fórmula cerrada (sin QP iterativo): si $\langle g, g_{\text{ref}}\rangle < 0$, se proyecta $\tilde{g} = g - \frac{g^\top g_{\text{ref}}}{g_{\text{ref}}^\top g_{\text{ref}}} g_{\text{ref}}$. A-GEM conserva la idea esencial de GEM (gradientes de memoria como restricción) a una fracción del costo de cómputo y memoria, y se volvió la variante de referencia en la práctica. *(A-GEM es contexto del ecosistema posterior a este paper, no parte del texto de 2017.)*

## 5. Experimentos

### 5.1. Datasets y arquitecturas

Tres variantes con $T = 20$ tareas cada una:

- **MNIST Permutations** (Kirkpatrick et al. 2017): cada tarea aplica una permutación fija de píxeles. Las distribuciones de entrada son **no relacionadas** entre tareas.
- **MNIST Rotations:** cada tarea rota los dígitos un ángulo fijo entre 0 y 180 grados.
- **Incremental CIFAR-100** (Rebuffi et al. 2017): cada tarea introduce un subconjunto disjunto de $100/T$ clases. La entrada es similar entre tareas pero la salida difiere.

En MNIST cada tarea tiene 1000 ejemplos de 10 clases; en CIFAR-100, 2500 ejemplos de 5 clases. Arquitecturas: MLP con dos capas ocultas de 100 ReLU para MNIST; un ResNet18 reducido (un tercio de los feature maps) con un clasificador lineal por tarea para CIFAR-100 (forma simple de aprovechar el descriptor). Todo con SGD plano, mini-batches de 10, e hiperparámetros por grid-search. Cada ejemplo se ve **una sola vez**.

### 5.2. Baselines comparados

(1) un **single** predictor entrenado a través de todas las tareas; (2) **independent**, un predictor por tarea (con $T$ veces menos unidades); (3) **multimodal**, una capa de entrada dedicada por tarea (solo MNIST); (4) **EWC** (regularización); (5) **iCaRL** (clasificador nearest-exemplar con memoria episódica, solo CIFAR-100).

### 5.3. Resultados principales

GEM **iguala o supera** a todos los rivales, **minimizando el olvido (BWT cercano a cero o positivo) y con FWT despreciable o positivo**. Cifras clave de las matrices completas (Apéndice B), MNIST permutations:

| Modelo | ACC | BWT | FWT |
|---|---|---|---|
| single | 0.6018 | -0.1980 | 0.0093 |
| independent | 0.4523 | 0.0000 | 0.0000 |
| multimodal | 0.7561 | -0.0275 | 0.0059 |
| EWC | 0.6185 | -0.1653 | 0.0054 |
| **GEM** | **0.8260** | **+0.0247** | 0.0088 |

GEM logra ACC 0.826 con **BWT positivo (+0.0247)**: no solo no olvida, sino que aprender tareas posteriores *mejoró* en promedio las tareas previas. EWC, en cambio, sufre BWT muy negativo (-0.165), señal de olvido. La curva de la accuracy en la primera tarea a lo largo del continuum (Figura 1, derecha) muestra que GEM exhibe olvido mínimo y BWT positivo también en CIFAR-100.

**Eficiencia (Tabla 1, tiempo de CPU en MNIST):** GEM (77 s permutations / 135 s rotations) es **más rápido que EWC** (179 / 169 s), porque optimiza sobre $T = 20$ variables en el dual en vez de sobre $p \approx 10^6$ parámetros. El costo residual de GEM es calcular los gradientes de tareas previas en cada iteración.

**Tamaño de memoria (Tabla 2, CIFAR-100):** la ACC final de GEM **crece monótonamente con el tamaño de memoria** (0.487 con 200 → 0.654 con 5120) y **supera a iCaRL en todo el rango** (iCaRL: 0.436 → 0.508). Que sea monótona elimina la necesidad de afinar finamente este hiperparámetro.

**Número de pasadas (Tabla 3, MNIST rotations):** múltiples pasadas por tarea *agravan* el olvido catastrófico en métodos sin memoria. Con 5 épocas por tarea, "single" cae a ACC 0.43 / BWT -0.40, mientras GEM se mantiene en 0.89 / -0.02. Más aún: comparando GEM contra la **cota superior oráculo** (single entrenado con datos iid barajados de todas las tareas, ACC 0.83-0.89 con BWT ≈ 0), **GEM iguala esa cota** mientras minimiza el BWT negativo. Es un resultado fuerte: el aprendizaje continuo con restricciones de gradiente alcanza el rendimiento del multi-task learning iid.

## 6. Limitaciones reconocidas

El paper es explícito sobre tres puntos de mejora:

1. **No aprovecha descriptores de tarea estructurados.** Al usar descriptores enteros, GEM no puede explotar relaciones semánticas entre tareas y por eso no logra FWT positivo significativo (zero-shot). Descriptores ricos (p. ej. texto que describa la tarea) quedan como trabajo futuro.
2. **Gestión de memoria ingenua.** Guarda los últimos $m$ ejemplos por tarea; no investiga estrategias avanzadas como construir coresets (Lucic et al. 2017). Y, transversal a la familia basada en memoria: **la memoria total crece (o $m$ se reduce) con el número de tareas**, lo que es un problema de escalabilidad en horizontes largos.
3. **Costo computacional del QP y de los gradientes por tarea.** Cada iteración requiere **un backward pass por tarea pasada** más la resolución del QP dual. Aunque el QP es barato ($t-1$ variables), el cómputo de los $g_k$ escala con el número de tareas. (Este es precisamente el problema que A-GEM resuelve después, §4.6.)

Otra limitación implícita del setting: GEM asume disponibilidad del **descriptor de tarea $t$ en test** (saber a qué tarea pertenece el ejemplo a clasificar), lo que corresponde al escenario *task-incremental*; los escenarios más duros *domain-* y *class-incremental* (Van de Ven y Tolias 2019) son más exigentes.

## 7. Impacto

GEM es uno de los trabajos seminales que **estructuraron el campo del aprendizaje continuo**. Sus dos legados perduran:

- **Las métricas ACC/BWT/FWT** se volvieron el lenguaje estándar para reportar resultados de aprendizaje continuo; prácticamente todo paper posterior las usa o las extiende.
- **La idea de gradientes de memoria como restricciones** abrió la subfamilia de métodos de *gradient projection* (A-GEM, OGD — Orthogonal Gradient Descent, GPM — Gradient Projection Memory), distinta tanto del *replay* puro como de la regularización. La intuición geométrica —proyectar el gradiente al cono donde no daña tareas pasadas— es pedagógicamente limpia y reaparece en múltiples variantes.

GEM también demostró empíricamente que la **transferencia positiva hacia atrás es alcanzable**, refutando la idea de que aprender lo nuevo necesariamente degrada lo viejo, y conectó el aprendizaje continuo con la inferencia causal (correlaciones invariantes entre entornos).

## 8. Conexión con la Clase 32 (Olvido Catastrófico)

La Clase 32 organiza los remedios contra el olvido catastrófico en las tres familias de la §2 —arquitectura, regularización y **memoria**— y GEM es el ejemplar destacado de la familia basada en memoria. El mapeo con el material de la clase es directo:

- **"Memoria $M_k$ para cada tarea $k$"** (slide de la clase): es literalmente la memoria episódica de GEM (§4.1). La clase usa GEM para ilustrar qué significa guardar ejemplos de tareas pasadas.
- **Usar los datos en memoria para modificar los gradientes** (el punto que la clase enfatiza para diferenciar a GEM del *replay* ingenuo): es la esencia del §4.2-4.4. La distinción pedagógica clave que la clase transmite es que GEM **no re-entrena** con la memoria (eso sobreajustaría) ni **destila** para congelar predicciones (eso impediría BWT positivo), sino que convierte los gradientes de memoria en **restricciones geométricas** sobre la dirección del paso de SGD.
- **Contraste con EWC** (la otra estrella de la clase, familia regularización): la clase puede contraponer ambos con las cifras de la §5.3 —EWC penaliza cambios en parámetros importantes (memoria sináptica) y sufre BWT -0.165, mientras GEM restringe gradientes (memoria episódica) y alcanza BWT +0.025, además de ser más rápido—. Es un contraste limpio entre "proteger pesos" y "proteger pérdidas pasadas".
- **Las métricas ACC/BWT/FWT** (§3.1) son las que la clase usa para cuantificar el olvido: un BWT muy negativo *es* el olvido catastrófico medido.

Para profundizar en el marco general (escenarios task/domain/class-incremental, *stability-plasticity dilemma*, taxonomía de métodos) ver [/fundamentos/aprendizaje-continuo](/fundamentos/aprendizaje-continuo); para el recorrido completo de la clase con el resto de los métodos (iCaRL, LwF, Progressive Networks, HAT, SupSup, L2P) ver [/clases/clase-32](/clases/clase-32).
