---
title: "HAT: Hard Attention to the Task (2018)"
weight: 364
math: true
---

{{< paper-card
    title="Overcoming Catastrophic Forgetting with Hard Attention to the Task"
    authors="Joan Serrà, Dídac Surís, Marius Miron, Alexandros Karatzoglou"
    year="2018"
    venue="ICML 2018"
    pdf="/papers/hat-serra-2018.pdf"
    arxiv="1801.01423" >}}
HAT (Hard Attention to the Task) es el representante canónico de los **métodos de arquitectura** contra el olvido catastrófico que ve la [Clase 32](/clases/clase-32). Su idea: aprender —tarea a tarea y por backpropagation— **máscaras de atención casi binarias sobre las unidades de cada capa**, condicionadas por un *embedding* de tarea. Esas máscaras se acumulan y se usan para **modular el gradiente**, congelando los pesos importantes de tareas previas mientras dejan libre el resto de la capacidad. Con solo dos hiperparámetros interpretables, HAT reduce el olvido entre un **45% y un 80%** respecto a los mejores métodos de la época.
{{< /paper-card >}}

---

## El problema: olvido catastrófico

El **olvido catastrófico** (McCloskey & Cohen, 1989) es la tendencia de una red neuronal a borrar lo aprendido en una tarea cuando se entrena en tareas posteriores. Es el obstáculo central del [aprendizaje continuo](/fundamentos/aprendizaje-continuo) (*lifelong learning*): un modelo debería absorber una secuencia de tareas sin re-procesar los datos antiguos. El campo se organiza en tres familias de soluciones:

- **Rehearsal / memoria:** almacenar y re-procesar instancias previas, o sintetizarlas con redes generativas. Implica alguna forma de aprendizaje concurrente.
- **Regularización estructural (*suave*):** EWC (Fisher, post-entrenamiento), SI (durante el entrenamiento). Penalizan mover pesos importantes con un término en el *loss*, pero el peso aún puede cambiar: solo paga un costo.
- **Métodos de arquitectura:** dedicar sub-partes de la red a cada tarea. Aquí viven PNN (una columna por tarea, parámetros sin tope), PathNet (algoritmo genético sobre *paths*), PackNet (máscara binaria sobre pesos por poda heurística) y **HAT**.

El *trade-off* transversal es **estabilidad vs. plasticidad**: mantener los pesos importantes (estabilidad) sin bloquear el aprendizaje de lo nuevo (plasticidad). HAT controla ese balance de forma *aprendida* y con grano fino, a nivel de unidad.

## La idea central de HAT

La tesis es elegante: si supiéramos *qué unidades* de cada capa son cruciales para cada tarea, podríamos congelar sus pesos asociados (estabilidad) y dejar las restantes libres para el futuro (plasticidad). HAT no impone esa partición a mano: la **aprende junto con la red** vía backpropagation. Sus piezas conceptuales:

1. **Atención por unidad, no por peso.** A diferencia de PackNet (máscara sobre pesos) o PNN/PathNet (sobre columnas/módulos), HAT enmascara las **unidades** (neuronas o filtros convolucionales) de cada capa. La máscara sobre los pesos se *deriva automáticamente* de las máscaras de unidades. Estructura liviana: un vector por capa, no una matriz.
2. **Condicionada por un *embedding* de tarea.** Cada tarea $t$ tiene su propio *embedding* $e^t_l$ por capa; un *gate* sigmoide lo convierte en el vector de atención. Es la **identidad de la tarea** (su *task ID*) la que activa o desactiva unidades.
3. **Máscaras casi binarias.** Inspirándose en McCulloch & Pitts (1943), la atención $a^t_l \to \{0,1\}$ actúa como sinapsis inhibitoria, creando y destruyendo *paths* a través de las capas —pero sin pre-asignar tamaño ni número de módulos.
4. **Protección del gradiente.** Las máscaras acumuladas de tareas previas **modulan el gradiente** de la tarea actual: los pesos importantes quedan congelados. Es regularización *dura*, no *suave* como EWC/SI.
5. **Penalización de capacidad.** Un término de esparsidad incentiva a cada tarea a usar el mínimo de unidades, reservando capacidad para el futuro.

## El método paso a paso

### Máscara de atención (forward)

La salida de las unidades de la capa $l$ se multiplica elemento a elemento por la máscara: $h'_l = a^t_l \odot h_l$. La máscara **no es una distribución de probabilidad**: es un *embedding* de tarea gateado,

$$a^t_l = \sigma(s\, e^t_l),$$

donde $\sigma$ es la sigmoide y $s>0$ un **parámetro de escala**. La última capa tiene una máscara binaria fija (salida *multi-head*: un cabezal por tarea).

### Annealing del gating

Una función escalón daría máscaras perfectamente binarias, pero no es diferenciable. La solución es usar la sigmoide escalada como **pseudo-escalón diferenciable**: con $s \to \infty$ tiende al escalón duro $\{0,1\}$; con $s \to 0$ todas las unidades quedan igualmente activas ($1/2$). El truco es **recocer (annealing) $s$** dentro de cada época, empezando bajo (máxima plasticidad para explorar) y subiéndolo linealmente por *batch*:

$$s = \frac{1}{s_{\max}} + \left(s_{\max} - \frac{1}{s_{\max}}\right)\frac{b-1}{B-1}.$$

En *test* se fija $s = s_{\max} \gg 1$ y las máscaras son efectivamente binarias. El hiperparámetro $s_{\max}$ **es la perilla estabilidad/plasticidad**: cerca de 1 el *gate* es una sigmoide normal (mucha plasticidad, la red puede olvidar); muy grande, un escalón (mucha estabilidad, pesos congelados).

### Modulación del gradiente (backward)

HAT **acumula** las atenciones con el máximo elemento a elemento, preservando cualquier unidad que haya importado en *alguna* tarea previa:

$$a^{\le t}_l = \max\!\left(a^t_l,\, a^{\le t-1}_l\right).$$

El gradiente del peso que conecta la unidad $j$ de la capa $l-1$ con la unidad $i$ de la capa $l$ se modula con el reverso del **mínimo** de la atención acumulada de ambas:

$$g'_{l,ij} = \left[1 - \min\!\left(a^{\le t}_{l,i},\, a^{\le t}_{l-1,j}\right)\right] g_{l,ij}.$$

Intuición: un peso solo se protege si **tanto** la unidad de entrada **como** la de salida fueron importantes antes. Si ambas tienen atención $\to 1$, el factor $\to 0$ y el gradiente se anula: el peso queda congelado. Así, la máscara *sobre pesos* se deriva automáticamente de las máscaras *sobre unidades*. (Para corregir que los *embeddings* casi no se movían por culpa del annealing, los autores añaden una compensación del gradiente del *embedding* y *clamps* numéricos.)

### Regularización de capacidad

Como las unidades dedicadas a una tarea quedan ocupadas, hay que reservar espacio promoviendo esparsidad:

$$L' = L(y,\hat y) + c\, R(A^t, A^{<t}),$$

donde $R$ es una **L1 ponderada y normalizada** sobre las atenciones de la tarea actual. Las unidades ya usadas antes ($a^{<t}\to 1$) reciben peso $\approx 0$ y quedan excluidas de la penalización —se incentiva su reutilización. El hiperparámetro $c \ge 0$ es la **constante de compresibilidad**: a mayor $c$, red más esparsa.

## Resultados

Los autores critican los *setups* clásicos (permuted/split MNIST) por sesgar conclusiones y proponen un protocolo más exigente: **secuencias aleatorias de 8 datasets** distintos (CIFAR10/100, FaceScrub, FashionMNIST, NotMNIST, MNIST, SVHN, TrafficSigns), arquitectura tipo AlexNet, 10 repeticiones con 10 semillas. La métrica es el **forgetting ratio** $\rho$, normalizado entre clasificador aleatorio ($\approx -1$) y multitarea conjunto ($\approx 0$).

| Setup | HAT | Mejor baseline | Reducción del olvido |
|---|---|---|---|
| Secuencia 8 tareas ($t=2$) | $\rho = -0.02$ | EWC $-0.08$ | **75%** |
| Secuencia 8 tareas ($t=8$) | $\rho = -0.06$ | PNN $-0.11$ | **45%** |
| Incremental class (CIFAR10/100) | $\rho = -0.09$ | EWC $-0.18$ | **55%** |
| Permuted MNIST | $98.6\%$ | SI $97.1\%$ | **52%** (error) |
| Split MNIST | $99.0\%$ | conceptor $94.9\%$ | **80%** (error) |

HAT supera consistentemente a los 11 *baselines* con menor varianza (robustez frente a secuencias, *splits* e inicializaciones). PathNet y PNN, por construcción, nunca olvidan, pero pierden plasticidad al pre-asignar pesos. HAT tiene solo **dos hiperparámetros** con interpretación directa ($s_{\max}$ para estabilidad/plasticidad, $c$ para compacidad; por defecto $s_{\max}=400$, $c=0.75$), permite **monitorear** el uso de capacidad por capa, y como subproducto sirve para **poda y compresión** (comprime la red al 1–21% de su tamaño, aprendiendo la poda vía backpropagation).

## Limitaciones

- **Requiere el *task ID* en inferencia.** La limitación más relevante: HAT necesita saber a qué tarea pertenece cada entrada para elegir su máscara y cabezal. Lo ubica en el escenario *task-incremental* (con *task ID* conocido), no en el más difícil *class-incremental*. En muchas aplicaciones reales el *task ID* no está disponible en *test*.
- **Capacidad finita.** La red base es de tamaño fijo; con suficientes tareas el espacio libre se agota y la plasticidad cae. La compresión adaptativa lo mitiga, no lo elimina.
- **Salida *multi-head* fija.** La última capa es binaria *hard-coded* por tarea, coherente con el supuesto de *task ID* pero limitante.
- **Evaluación acotada a clasificación de imágenes.** El protocolo, aunque riguroso, no demuestra extensión a otros dominios.

## Por qué importa para la Clase 32

La [Clase 32](/clases/clase-32) presenta HAT como el representante de los **métodos de arquitectura** frente a la regularización (EWC, SI) y el rehearsal/memoria. Su valor pedagógico es triple:

1. **Hace tangible el *trade-off* estabilidad–plasticidad.** El parámetro $s_{\max}$ es literalmente una perilla entre "recordar todo" y "aprender rápido lo nuevo".
2. **Contrasta regularización *dura* vs. *suave*.** HAT anula el gradiente de los pesos protegidos; EWC/SI solo lo penalizan. Comparar ambos sobre la misma métrica es el análisis central de la clase.
3. **Muestra el grano fino de la atención por unidad.** HAT *aprende* qué unidades importan, sin pre-asignar columnas (PNN) ni módulos (PathNet) a ciegas. Conecta el concepto de "atención" (ya familiar del módulo NLP) con un uso nuevo: particionar la capacidad de la red entre tareas.

Para el marco general, ver el fundamento de [aprendizaje continuo](/fundamentos/aprendizaje-continuo). Un método de arquitectura emparentado —máscaras binarias sobre los pesos de una red preentrenada y congelada— es [Piggyback (Mallya et al., 2018)](/papers/piggyback-mallya-2018), que comparte con HAT la idea de enmascarar para evitar el olvido pero parte de pesos fijos en lugar de aprenderlos.

## Notas y enlaces

- arXiv: [arxiv.org/abs/1801.01423](https://arxiv.org/abs/1801.01423) (v3, may 2018).
- Código: [github.com/joansj/hat](https://github.com/joansj/hat) (PyTorch).
- Venue: *Proceedings of the 35th ICML* (2018), Estocolmo, PMLR 80.
- Afiliación: Telefónica Research, Barcelona.
