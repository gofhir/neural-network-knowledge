---
title: "L2P: Learning to Prompt for Continual Learning (2022)"
weight: 365
math: true
---

{{< paper-card
    title="Learning to Prompt for Continual Learning"
    authors="Zifeng Wang, Zizhao Zhang, Chen-Yu Lee, Han Zhang, Ruoxi Sun, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer Dy, Tomas Pfister"
    year="2022"
    venue="CVPR 2022"
    pdf="/papers/l2p-wang-2022.pdf"
    arxiv="2112.08654" >}}
Paper bisagra del aprendizaje continuo: en vez de proteger los pesos de un modelo que se reentrena, **L2P** congela un Transformer pre-entrenado y guarda el conocimiento de cada tarea en un **pool de prompts** —vectores aprendibles diminutos— recuperables por contenido. Un mecanismo de selección *query-key* instance-wise elige qué prompts usar para cada entrada **sin conocer la identidad de la tarea**, los concatena como tokens al inicio de la secuencia, y solo entrena prompts y clasificador (~0.1% de parámetros extra). Supera a métodos con buffer de memoria **incluso sin guardar ningún ejemplo del pasado**, y abrió la era *prompt-based* del aprendizaje continuo (DualPrompt, CODA-Prompt). Es el método con que cierra la [Clase 32](/clases/clase-32).
{{< /paper-card >}}

---

## Contexto: olvido catastrófico en la era de los Transformers

El **olvido catastrófico** (catastrophic forgetting), documentado desde McCloskey & Cohen (1989), es el talón de Aquiles del aprendizaje secuencial: cuando una red entrena sobre una tarea nueva, al ajustar sus pesos para resolverla destruye las representaciones de las tareas previas. Durante décadas el campo consolidó tres familias de soluciones, todas con limitaciones críticas:

- **Regularización** (EWC, Synaptic Intelligence, LwF): penalizan cambios en los pesos importantes para tareas pasadas. No guardan datos, pero rinden mal en escenarios desafiantes.
- **Rehearsal / repaso** (ER, iCaRL, DER++, Co²L): mantienen un *buffer* de ejemplos viejos para mezclarlos al entrenar. Alcanzaban el SOTA, pero se degradan con buffers pequeños y son inviables cuando la privacidad prohíbe almacenar datos (escenarios clínicos, financieros).
- **Arquitectura** (Progressive Networks, PackNet, SupSup): asignan subredes o máscaras por tarea. La mayoría requiere conocer la **identidad de la tarea en test**, suposición poco realista, y suelen duplicar el número de parámetros.

L2P se apoya en dos cambios de era. Primero, la disponibilidad de **Transformers pre-entrenados potentes** (ViT-B/16 en ImageNet, ver [ViT](/papers/vit-dosovitskiy-2021)), cuyas representaciones genéricas son tan buenas que ya no hace falta reajustar el backbone. Segundo, la maduración del **prompt-based learning** en NLP (Prompt Tuning, Prefix Tuning, AutoPrompt): en vez de adaptar los pesos de un modelo de lenguaje congelado, basta diseñar *prompts* —tokens aprendibles prepended a la entrada— que instruyen al modelo para resolver la tarea. Un prompt captura conocimiento específico con muchísimos menos parámetros que Adapters o LoRA.

La analogía que el paper invoca es la teoría de **Sistemas de Aprendizaje Complementarios** (CLS): el cerebro logra aprendizaje continuo combinando el hipocampo (aprendizaje rápido, memoria episódica) con el neocórtex (memoria de largo plazo). En L2P el *pool de prompts* hace de aprendizaje rápido y el *backbone congelado* de memoria de largo plazo. Un dato para dimensionar la economía del método: su mayor espacio de prompts ocupa **menos que una sola imagen de 224×224**.

## El obstáculo: por qué prompting no se aplica trivialmente

Antes de su solución, el paper explica por qué el prompting de NLP no se traslada directamente. Hay dos formas ingenuas, y ambas fallan:

1. **Un prompt distinto por tarea.** En test seguimos necesitando saber a qué tarea pertenece la entrada para escoger el prompt correcto —vuelve el requisito de identidad de tarea que queríamos eliminar—, y prompts independientes impiden compartir conocimiento entre tareas similares.
2. **Un único prompt compartido.** Habilita compartir conocimiento, pero como se reentrena secuencialmente, sufre el olvido catastrófico igual que cualquier parámetro reajustado (confirmado en la ablación del paper).

El objetivo ideal: aprender un sistema que **comparta conocimiento cuando las tareas son similares y lo mantenga aislado cuando no lo son**, decidiéndolo automáticamente y por instancia. Esa tensión es exactamente lo que resuelve el pool de prompts con selección query-key.

## Contribución central: el pool de prompts

L2P introduce un espacio de memoria *key-value* llamado **prompt pool**, optimizado junto con la pérdida supervisada, que decopla conocimiento *task-invariant* (compartido) y *task-specific* (aislado):

$$P = \{P_1, P_2, \cdots, P_M\}, \quad M = \text{número total de prompts}$$

Cada $P_j \in \mathbb{R}^{L_p \times D}$ es un prompt con longitud de token $L_p$ y la misma dimensión $D$ que los embeddings de entrada. Para una entrada $x$ con embedding $x_e = f_e(x)$, se seleccionan $N$ índices del pool y se concatenan al inicio:

$$x_p = [P_{s_1}; \cdots; P_{s_N}; x_e], \quad 1 \le N \le M$$

donde `;` es concatenación a lo largo de la dimensión de tokens. La clave es que **los prompts son libres de componerse**: entradas similares tienden a compartir más prompts, y viceversa, habilitando una compartición de conocimiento de grano fino a nivel de instancia.

## Método: selección query-key y predicción

**Backbone congelado y query.** El backbone es un ViT $f = f_r \circ f_e$ (sin cabeza de clasificación) que **permanece congelado** durante todo el aprendizaje continuo, preservando su generalidad. La query usa el propio modelo como extractor de features: $q(x) = f(x)[0,:]$ toma el vector del token `[class]`. Crucialmente, $q$ es **determinista y sin parámetros aprendibles** —eso es lo que permite operar sin conocer la identidad de la tarea.

**Selección instance-wise.** Cada prompt se asocia, como *valor*, a una **clave aprendible** $k_i \in \mathbb{R}^{D_k}$:

$$\{(k_1, P_1), \cdots, (k_M, P_M)\}, \quad K = \{k_i\}_{i=1}^M$$

La propia instancia decide qué prompts elegir mediante el matching query-key. Con $\gamma$ como función de puntuación (los autores usan **distancia coseno**), se buscan las top-$N$ claves:

$$K_x = \underset{\{s_i\}_{i=1}^N \subseteq [1,M]}{\arg\min} \sum_{i=1}^N \gamma(q(x), k_{s_i})$$

Este diseño key-value **decopla el aprendizaje del mecanismo de query del aprendizaje de los prompts** —la ablación lo demuestra crítico. Y como la consulta es instance-wise, todo el marco es **task-agnostic**: funciona sin fronteras claras de tarea en entrenamiento ni identidad de tarea en test. Opcionalmente, una tabla de frecuencia $H_t$ penaliza los prompts muy usados para diversificar la selección entre tareas no relacionadas (solo en entrenamiento).

**Objetivo y predicción.** Tras seleccionar los $N$ prompts, el embedding adaptado pasa por el resto del modelo congelado y el clasificador. La pérdida end-to-end minimiza:

$$\min_{P, K, \phi} \; \mathcal{L}\big(g_\phi(f_r^{avg}(x_p)), y\big) + \lambda \sum_{K_x} \gamma(q(x), k_{s_i})$$

El primer término es la cross-entropy de clasificación; el segundo es una *pérdida surrogate de matching* que acerca las claves seleccionadas a la query, entrenando el mecanismo de selección a apuntar a los prompts correctos ($\lambda = 0.5$, poco sensible). La predicción promedia los vectores ocultos de las posiciones de prompt antes del clasificador. **Solo se actualizan $P$, $K$ y $\phi$** —el backbone queda intacto.

## Experimentos

L2P se evalúa sobre un ViT-B/16 pre-entrenado en cuatro benchmarks que cubren los tres escenarios, con métricas **Average Accuracy** (mayor mejor) y **Forgetting** (menor mejor). Se reportan dos variantes: **L2P** (sin buffer) y **L2P-R** (con buffer, para comparación justa con rehearsal). Los prompts añaden apenas 0.05–0.11% de parámetros.

| Benchmark | Escenario | L2P (sin buffer) | Mejor baseline |
|---|---|---|---|
| Split CIFAR-100 | class-incremental | **83.83%** acc / 7.63% forget | LwF 60.69% / 27.77% |
| 5-datasets | class-incremental | **81.14%** acc / 4.64% forget | rehearsal, superados |
| CORe50 | domain-incremental | **78.33%** acc | LwF 75.45% |
| Gaussian CIFAR-100 | task-agnostic | **88.34%** acc | DER++ (con buffer) 85.24% |

Los titulares: L2P **sin buffer supera a casi todos los métodos rehearsal con buffer pequeño**; en el escenario task-agnostic (el más desafiante, sin fronteras de tarea) supera incluso a rehearsal *con* memoria; y frente a métodos de arquitectura, su brecha al upper-bound es de 7.02 puntos contra 40.27 de DualNet y 52.07 de SupSup.

**Ablaciones** (en 5-datasets): cada componente importa. Quitar el pool (usar un solo prompt) hunde la accuracy a 51.96% con 26.60% de forgetting —confirma que un prompt único sufre olvido severo. Quitar la clave aprendible baja a 58.33%; quitar la diversificación, a 62.26%; el modelo completo, 81.14%. Los histogramas de selección muestran el comportamiento esperado: en Split CIFAR-100 las tareas comparten prompts (alta similitud), mientras en 5-datasets cada tarea favorece prompts específicos.

## Limitaciones reconocidas

- **Demostrado solo en visión:** todos los experimentos son clasificación de imágenes, aunque el método no asume modalidad.
- **Requiere un modelo pre-entrenado basado en secuencias:** el mecanismo de prepender tokens es nativo de Transformers; generalizar a ConvNets no es trivial.
- **El benchmark task-agnostic es sintético:** el Gaussian scheduled CIFAR-100 es artificial; el paper pide benchmarks más realistas.
- **Sesgos heredados:** cualquier sesgo del backbone congelado se arrastra al proceso continuo; los autores recomiendan auditar el modelo base.

## Impacto: el nacimiento del continual learning prompt-based

L2P **inauguró una línea de investigación completa**. Demostrar que un backbone congelado más prompts aprendibles podía batir a métodos rehearsal *sin guardar datos* reformuló el campo y disparó una familia dominante:

- **DualPrompt** (Wang et al., ECCV 2022), de los mismos autores, separa los prompts en *G-Prompts* (conocimiento general compartido) y *E-Prompts* (conocimiento específico de tarea), insertándolos en distintas capas —refinando la dicotomía task-invariant/task-specific que L2P introdujo.
- **CODA-Prompt** (Smith et al., CVPR 2023) reemplaza la selección discreta top-$N$ por una **combinación ponderada y diferenciable** de componentes vía atención, eliminando la no-diferenciabilidad del argmin de L2P.
- La línea conecta con el auge del **parameter-efficient fine-tuning** (PEFT): en L2P, la eficiencia de parámetros (~0.1%) no es solo ahorro de cómputo, sino el mecanismo mismo que evita el olvido (al no tocar el backbone).

El método sigue siendo referencia obligada y baseline estándar en cualquier paper de aprendizaje continuo con modelos pre-entrenados posterior a 2022.

## Por qué importa para la Clase 32

La [Clase 32](/clases/clase-32) recorre el olvido catastrófico y sus soluciones clásicas —regularización (EWC, SI), rehearsal (replay, iCaRL), arquitectura (Progressive Networks, PackNet)— y **cierra con L2P** como el método moderno que sintetiza dos hilos del curso: el prompting (Clase 20, prompts aprendibles que instruyen modelos congelados) y los [Transformers pre-entrenados](/papers/vit-dosovitskiy-2021) como backbone. El flujo que las slides presentan es el del método: **query** (features del token `[class]`) → **selección query-key** (coseno contra las claves del pool, top-$N$, instance-wise y sin saber la tarea) → **prompts como tokens** (concatenados al inicio) → **predicción** (el Transformer congelado procesa la secuencia extendida y el clasificador predice).

La lección central, por contraste con los métodos clásicos: donde EWC penaliza el cambio de pesos y el replay guarda ejemplos, L2P simplemente *no toca el backbone* y guarda el conocimiento en prompts diminutos recuperables por contenido. Esto profundiza el fundamento de [aprendizaje continuo](/fundamentos/aprendizaje-continuo) y conecta con el [aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado), que produce los backbones pre-entrenados de los que L2P depende.

## Notas y enlaces

- arXiv: https://arxiv.org/abs/2112.08654 (v2, 21 mar 2022)
- Código oficial (JAX/Flax): https://github.com/google-research/l2p
- Venue: CVPR 2022. Afiliaciones: Google Cloud AI / Google Research y Northeastern University.
