---
title: "Profundizacion - Math del meta-aprendizaje"
weight: 20
math: true
---

> Esta pagina complementa la [teoria de la clase 26](/clases/clase-26/teoria) con derivaciones formales. Cuatro bloques: **Parte I** formaliza el meta-aprendizaje como optimizacion bi-nivel. **Parte II** deriva paso a paso el meta-gradiente de MAML y su Hessiano, mas las aproximaciones FOMAML y Reptile. **Parte III** desarrolla la matematica de los metodos no-parametricos y el resultado de Bregman que justifica la distancia euclidiana en Prototypical Networks. **Parte IV** formaliza el acceso a memoria de MANN.

---

## Parte I — El meta-aprendizaje como optimizacion bi-nivel

### I.1 El problema de aprendizaje estandar

En aprendizaje supervisado clasico, dado un dataset $\mathcal{D}=\{(x_n,y_n)\}_{n=1}^{N}$ y un modelo $f_\theta$, resolvemos

$$
\theta^* = \arg\min_\theta \mathcal{L}(\mathcal{D};\theta,\omega)
$$

donde $\omega$ engloba **todo lo que no es $\theta$ pero condiciona el aprendizaje**: la arquitectura, el optimizador, el learning rate, la inicializacion, la funcion de perdida. En el paradigma clasico $\omega$ se fija a mano (con conocimiento experto o busqueda de hiperparametros). El meta-aprendizaje convierte $\omega$ — el **meta-knowledge** — en una variable a optimizar.

### I.2 La formulacion bi-nivel

Siguiendo a [Hospedales et al.](/papers/meta-learning-survey-hospedales-2020), el meta-aprendizaje se escribe como un problema de dos niveles:

$$
\omega^* = \arg\min_\omega \sum_{i=1}^{M} \mathcal{L}^{meta}\Big(\theta^{*(i)}(\omega),\,\omega,\,\mathcal{D}^{val\,(i)}\Big)
\tag{nivel externo}
$$
$$
\text{s.t.}\quad \theta^{*(i)}(\omega) = \arg\min_\theta \mathcal{L}^{task}\Big(\theta,\,\omega,\,\mathcal{D}^{tr\,(i)}\Big)
\tag{nivel interno}
$$

La estructura es la de un **juego de Stackelberg**: el nivel externo (lider) elige $\omega$ anticipando que el nivel interno (seguidor) respondera con su mejor $\theta^{*(i)}(\omega)$. La asimetria es esencial — el seguidor optimiza *dado* $\omega$, y el lider optimiza *sabiendo como responde* el seguidor.

{{< concept-alert type="clave" >}}
La dependencia critica es $\theta^{*(i)}(\omega)$: la solucion del problema interno es una **funcion** de $\omega$. Optimizar el nivel externo exige diferenciar a traves de esa funcion, es decir, **a traves del proceso de optimizacion interno**. Ahi nace toda la dificultad (y la elegancia) del meta-aprendizaje basado en gradientes.
{{< /concept-alert >}}

### I.3 La vista de distribucion de tareas

Equivalentemente, asumimos una distribucion de tareas $p(\mathcal{T})$. Cada tarea $\mathcal{T}_i \sim p(\mathcal{T})$ aporta un par $(\mathcal{D}^{tr}_i, \mathcal{D}^{ts}_i)$ (support y query). El meta-objetivo es el riesgo esperado de adaptacion:

$$
\min_\omega \; \mathbb{E}_{\mathcal{T}_i \sim p(\mathcal{T})}\Big[\, \mathcal{L}^{meta}\big(\theta^{*}_i(\omega),\,\mathcal{D}^{ts}_i\big)\,\Big]
$$

En meta-test, las tareas provienen de la **misma distribucion** pero con clases disjuntas. La hipotesis de generalizacion del meta-aprendizaje es mas fuerte que la del aprendizaje clasico: no asumimos i.i.d. de *ejemplos*, sino i.i.d. de *tareas*. Cuando esa hipotesis se rompe (meta-shift), aparece el **meta-overfitting** — el modelo memoriza la familia de tareas de meta-train y no generaliza a tareas nuevas.

---

## Parte II — El meta-gradiente de MAML

### II.1 Especializacion de MAML

[MAML](/papers/maml-finn-2017) elige $\omega = \theta_0$, una **inicializacion** de los parametros, y define el nivel interno como **$k$ pasos de descenso de gradiente**. Para un solo paso ($k=1$):

$$
\phi_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}\big(f_\theta, \mathcal{D}^{tr}_i\big)
$$

y el nivel externo evalua los parametros adaptados $\phi_i$ sobre el query:

$$
\min_\theta \; \sum_{i} \mathcal{L}_{\mathcal{T}_i}\big(f_{\phi_i}, \mathcal{D}^{ts}_i\big)
= \min_\theta \; \sum_{i} \mathcal{L}_{\mathcal{T}_i}\Big(f_{\theta - \alpha\nabla_\theta\mathcal{L}_{\mathcal{T}_i}(f_\theta,\mathcal{D}^{tr}_i)},\; \mathcal{D}^{ts}_i\Big)
$$

### II.2 Derivacion del meta-gradiente

Queremos $\nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_{\phi_i})$. Aplicamos la **regla de la cadena**, recordando que $\phi_i$ depende de $\theta$:

$$
\nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_{\phi_i})
= \frac{\partial \phi_i}{\partial \theta}^{\!\top} \, \nabla_{\phi_i}\mathcal{L}_{\mathcal{T}_i}(f_{\phi_i})
$$

El primer factor es el **Jacobiano de la adaptacion**. Derivando $\phi_i = \theta - \alpha\nabla_\theta\mathcal{L}_{\mathcal{T}_i}(f_\theta)$ respecto de $\theta$:

$$
\frac{\partial \phi_i}{\partial \theta}
= I - \alpha \, \nabla^2_\theta \mathcal{L}_{\mathcal{T}_i}\big(f_\theta, \mathcal{D}^{tr}_i\big)
$$

donde $\nabla^2_\theta\mathcal{L}$ es la matriz **Hessiana** de la perdida del support. Sustituyendo:

$$
\boxed{\;\nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_{\phi_i})
= \Big(I - \alpha\,\nabla^2_\theta\mathcal{L}_{\mathcal{T}_i}(f_\theta,\mathcal{D}^{tr}_i)\Big)\, \nabla_{\phi_i}\mathcal{L}_{\mathcal{T}_i}(f_{\phi_i},\mathcal{D}^{ts}_i)\;}
$$

{{< concept-alert type="recordar" >}}
El meta-gradiente contiene un termino de **segundo orden**: el Hessiano $\nabla^2_\theta\mathcal{L}$. No se materializa la matriz completa (seria $|\theta|\times|\theta|$); se computa el **producto Hessiano-vector** $\nabla^2_\theta\mathcal{L}\cdot v$ via diferenciacion automatica de modo reverso, con costo comparable a un gradiente.
{{< /concept-alert >}}

### II.3 El meta-update

Con un mini-batch de tareas, el paso del nivel externo es:

$$
\theta \leftarrow \theta - \beta \sum_i \Big(I - \alpha\,\nabla^2_\theta\mathcal{L}_{\mathcal{T}_i}(f_\theta)\Big)\,\nabla_{\phi_i}\mathcal{L}_{\mathcal{T}_i}(f_{\phi_i})
$$

con $\alpha$ el learning rate **interno** (adaptacion) y $\beta$ el learning rate **externo** (meta). Notese que $\beta$ actua sobre $\theta$, no sobre $\phi_i$: actualizamos la inicializacion, no los parametros adaptados.

### II.4 Generalizacion a $k$ pasos

Con $k$ pasos internos $\theta = \phi^{(0)} \to \phi^{(1)} \to \dots \to \phi^{(k)}$, el Jacobiano se vuelve un **producto de $k$ Jacobianos**:

$$
\frac{\partial \phi^{(k)}}{\partial \theta} = \prod_{j=0}^{k-1}\Big(I - \alpha\,\nabla^2_{\phi^{(j)}}\mathcal{L}(f_{\phi^{(j)}})\Big)
$$

Esto exige **desenrollar** (unroll) el grafo de computo de los $k$ pasos y retropropagar a traves de el — costoso en memoria, que crece con $k$.

### II.5 FOMAML: la aproximacion de primer orden

**First-Order MAML** ignora el termino Hessiano, aproximando $\frac{\partial\phi_i}{\partial\theta}\approx I$:

$$
\nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_{\phi_i}) \;\approx\; \nabla_{\phi_i}\mathcal{L}_{\mathcal{T}_i}(f_{\phi_i})
$$

Es decir, se evalua el gradiente de la perdida del query **en los parametros adaptados** $\phi_i$ y se aplica directamente a $\theta$. El paper reporta que FOMAML alcanza casi el mismo rendimiento (48.07% vs 48.70% en MiniImagenet 5-way 1-shot) con ~33% menos de computo. La justificacion empirica: cerca de un minimo local, las derivadas de segundo orden son pequeñas (las activaciones ReLU son localmente lineales, asi que el Hessiano es casi nulo).

### II.6 Reptile

**Reptile** (Nichol et al.) prescinde por completo de la distincion support/query y del Hessiano. Para cada tarea: ejecuta $k$ pasos de SGD desde $\theta$ obteniendo $\phi_i^{(k)}$, y luego mueve $\theta$ hacia $\phi_i^{(k)}$:

$$
\theta \leftarrow \theta + \epsilon\,\big(\phi_i^{(k)} - \theta\big)
$$

Sorprendentemente funciona. La expansion de Taylor del update de Reptile revela que, en esperanza, **maximiza el producto interno entre gradientes de distintos minibatches de la misma tarea**, lo que empuja hacia inicializaciones cuyos gradientes estan alineados — la misma intuicion que MAML, sin computar segundas derivadas.

| Metodo | Segundo orden | Memoria | Costo | Calidad |
| --- | --- | --- | --- | --- |
| MAML | si (Hessiano) | desenrolla $k$ pasos | alto | referencia |
| FOMAML | no (aprox. $I$) | baja | ~33% menos | casi igual |
| Reptile | no | baja | bajo | competitivo |
| iMAML | implicito (sin desenrollar) | $O(1)$ en $k$ | medio | igual a MAML |

**iMAML** usa el **teorema de la funcion implicita** para obtener el meta-gradiente sin desenrollar el inner loop, asumiendo que el inner loop converge a un optimo con regularizacion. Detalle completo en [Optimizacion bi-nivel](/fundamentos/optimizacion-binivel).

---

## Parte III — Metodos metric-based

### III.1 Matching Networks: kernel de atencion

[Matching Networks](/papers/matching-networks-vinyals-2016) predice con una suma ponderada de los labels del support set:

$$
P(\hat{y}\mid \hat{x}, S) = \sum_{i=1}^{k} a(\hat{x}, x_i)\, y_i, \qquad
a(\hat{x}, x_i) = \frac{\exp\big(c(f(\hat{x}), g(x_i))\big)}{\sum_{j=1}^{k}\exp\big(c(f(\hat{x}), g(x_j))\big)}
$$

donde $c(\cdot,\cdot)$ es la similitud coseno y $y_i$ esta en codificacion one-hot. Es un **clasificador no-parametrico**: no hay una capa de salida con pesos por clase; el support set *es* el clasificador. Conecta dos clasicos:

- Si $a$ fuera una indicadora del vecino mas cercano, recuperariamos **1-NN**.
- Si $a$ fuera un kernel de densidad, recuperariamos **kernel density estimation**.

Matching Networks aprende el kernel (via los embeddings $f,g$) en lugar de fijarlo.

### III.2 La equivalencia con la atencion de Transformers

Reescribiendo con la notacion de atencion: $Q = f(\hat{x})$ (query), $K = \{g(x_i)\}$ (keys), $V = \{y_i\}$ (values):

$$
\hat{y} = \text{softmax}\big(Q K^\top\big)\, V
$$

Es **exactamente** una capa de atencion key-value. Matching Networks (2016) prefiguro el mecanismo central de los Transformers (2017) y, conceptualmente, el *in-context learning* de los LLMs: el contexto (support set / prompt) actua como una memoria sobre la que se atiende para predecir. Ver [Self-Attention](/fundamentos/self-attention) e [In-Context Learning](/fundamentos/in-context-learning).

### III.3 Prototypical Networks y la justificacion de Bregman

[Prototypical Networks](/papers/prototypical-networks-snell-2017) define el prototipo de la clase $k$ como el centroide de los embeddings del support:

$$
c_k = \frac{1}{|S_k|}\sum_{(x_i,y_i)\in S_k} f_\phi(x_i)
$$

y clasifica con softmax sobre **distancias negativas**:

$$
p_\phi(y=k\mid x) = \frac{\exp\big(-d(f_\phi(x), c_k)\big)}{\sum_{k'}\exp\big(-d(f_\phi(x), c_{k'})\big)}
$$

minimizando $J = -\log p_\phi(y=k\mid x)$ sobre episodios.

**¿Por que euclidiana y no coseno?** El resultado teorico clave: cuando $d$ es una **divergencia de Bregman** (la distancia euclidiana al cuadrado lo es), el clasificador prototipico es equivalente a una **estimacion de densidad por mezcla** con densidades de la familia exponencial, y el prototipo (la media) es el estimador que minimiza la divergencia esperada. La distancia coseno **no** es una divergencia de Bregman, por lo que rompe esa coherencia. Empiricamente la brecha es grande: ~17 puntos a favor de euclidiana en 5-shot.

Desarrollemoslo. Con $d(z,c) = \|z-c\|^2$, expandimos el logito:

$$
-\|f_\phi(x)-c_k\|^2 = -\,f_\phi(x)^\top f_\phi(x) + 2\,c_k^\top f_\phi(x) - c_k^\top c_k
$$

El primer termino $-f_\phi(x)^\top f_\phi(x)$ es **comun a todas las clases** $k$, asi que se cancela en el softmax. Quedan terminos lineales en $f_\phi(x)$:

$$
-\|f_\phi(x)-c_k\|^2 = \underbrace{2 c_k^\top}_{w_k^\top} f_\phi(x) \underbrace{- c_k^\top c_k}_{b_k} + \text{const}
$$

Es decir, **el clasificador prototipico con distancia euclidiana es un clasificador lineal** $w_k^\top z + b_k$ en el espacio de embeddings, con $w_k = 2c_k$ y $b_k = -\|c_k\|^2$. Esta linealidad escondida explica por que es tan estable y por que entrenar con mas *way* del que se usara en test mejora la calidad del espacio aprendido.

### III.4 Relacion con Matching Networks

En el caso **1-shot**, cada prototipo $c_k$ es el unico ejemplo de su clase, asi que Prototypical Networks y Matching Networks (con coseno reemplazado por euclidiana) coinciden. Para $K>1$, Matching Networks atiende sobre **todos** los puntos del support, mientras Prototypical Networks los **promedia** en un centroide. Promediar es mas simple, mas barato y resulta igual o mejor en la practica.

---

## Parte IV — MANN: acceso a memoria

### IV.1 Lectura por contenido

[MANN](/papers/mann-santoro-2016) usa un controlador (LSTM) que interactua con una matriz de memoria externa $M_t \in \mathbb{R}^{N\times m}$ ($N$ slots de dimension $m$). En el paso $t$, el controlador emite una clave $k_t$. El peso de lectura del slot $i$ es un softmax de similitudes coseno:

$$
w_t^r(i) = \frac{\exp\big(K(k_t, M_t(i))\big)}{\sum_j \exp\big(K(k_t, M_t(j))\big)}, \qquad
K(u,v) = \frac{u\cdot v}{\|u\|\,\|v\|}
$$

y la lectura es la combinacion ponderada de slots:

$$
r_t = \sum_{i} w_t^r(i)\, M_t(i)
$$

Esto es, de nuevo, **atencion sobre una memoria** — la misma estructura matematica que Matching Networks y los Transformers.

### IV.2 Escritura: LRUA

El modulo **Least Recently Used Access** decide donde escribir combinando un *usage weight* $w_t^u$ (cuanto se ha usado cada slot) y un *least-used weight* $w_t^{lu}$ (indicadora de los slots menos usados). El peso de escritura interpola, mediante una compuerta $\sigma(\alpha)$, entre escribir en el slot **leido mas recientemente** y en el slot **menos usado**:

$$
w_t^w = \sigma(\alpha)\, w_{t-1}^r + \big(1-\sigma(\alpha)\big)\, w_{t-1}^{lu}
$$

y la memoria se actualiza de forma aditiva:

$$
M_t(i) = M_{t-1}(i) + w_t^w(i)\, k_t
$$

El *usage weight* se actualiza con decaimiento $\gamma$ acumulando lecturas y escrituras: $w_t^u = \gamma\,w_{t-1}^u + w_t^r + w_t^w$. La gracia de LRUA frente al *location-based addressing* de las Neural Turing Machines: para one-shot, queremos escribir info nueva en slots libres (poco usados) sin pisar lo recien aprendido, y poder sobrescribir lo recien leido cuando corresponde. Es un esquema de memoria puramente por contenido, mas adecuado al meta-aprendizaje episodico. Detalle en [Memory-Augmented Networks](/fundamentos/memory-augmented-networks).

---

## Sintesis matematica

| Concepto | Ecuacion central |
| --- | --- |
| Meta-objetivo bi-nivel | $\omega^*=\arg\min_\omega \sum_i \mathcal{L}^{meta}(\theta^{*(i)}(\omega),\mathcal{D}^{val}_i)$ |
| Adaptacion MAML | $\phi_i=\theta-\alpha\nabla_\theta\mathcal{L}(\theta,\mathcal{D}^{tr}_i)$ |
| Meta-gradiente | $(I-\alpha\nabla^2_\theta\mathcal{L})\nabla_{\phi_i}\mathcal{L}(f_{\phi_i})$ |
| Matching Networks | $\hat{y}=\sum_i \text{softmax}(c(f(\hat{x}),g(x_i)))\,y_i$ |
| Prototypical (lineal escondido) | $-\|f(x)-c_k\|^2 = 2c_k^\top f(x)-\|c_k\|^2+\text{const}$ |
| Lectura de memoria | $r_t=\sum_i \text{softmax}(K(k_t,M_t(i)))\,M_t(i)$ |

El hilo conductor: **atencion sobre una coleccion** (support set o memoria) aparece en Matching Networks, MANN y los Transformers; **diferenciacion a traves de la optimizacion** define MAML. Dos ideas matematicas, cinco algoritmos.

---

**Ver tambien:** [Teoria de la clase 26](/clases/clase-26/teoria) · [Practica desde 0](/clases/clase-26/practica) · Fundamentos: [Optimizacion bi-nivel](/fundamentos/optimizacion-binivel) · [Metric Learning](/fundamentos/metric-learning) · [Memory-Augmented Networks](/fundamentos/memory-augmented-networks) · Papers: [MAML](/papers/maml-finn-2017) · [Prototypical Networks](/papers/prototypical-networks-snell-2017) · [Matching Networks](/papers/matching-networks-vinyals-2016).
