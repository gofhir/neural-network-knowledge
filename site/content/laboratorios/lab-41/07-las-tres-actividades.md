---
title: "07 - Las tres actividades"
weight: 70
math: true
---

> Las tres preguntas de selección múltiple del práctico, con su justificación en las slides y en el código, y el análisis de por qué cada distractor falla. Las tres preguntan por cosas que el recorrido del notebook ya había verificado.

---

## Actividad 1

> **¿Cuál es la principal desventaja de usar un modelo de clasificación para la tarea de reconocimiento de hablante?**

### ✅ **«Cada vez que agreguemos una nueva persona, tendremos que reentrenar el modelo»**

**Es literal en las slides.** La clase construye la pregunta en cuatro pasos (slides 4–7) y la responde con signo de exclamación:

> *"We can model it as a classifier, which calculates the probability of each speaker based on the current input signal. But, how can we train this model? How can we incorporate new people? → **Our model must be trained entirely for each new speaker!**"*

Y en el código está materializado: `dense_1` es una `Linear(512, 8631)` cuya dimensión de salida **está clavada** al número de identidades de entrenamiento. Añadir un hablante exige una fila más, y esa fila no existe hasta que se reentrena.

Por eso la arquitectura **descarta esa capa en inferencia** —los 4.419.072 parámetros que son el 36,6 % del modelo y nunca se ejecutan— y usa el embedding. Los 40 hablantes de VoxCeleb1-test, **que el modelo nunca vio**, se verifican sin tocar un solo peso. La slide lo subraya: *"VoxCeleb1 and VoxCeleb2 are completely disjoint!"*.

Es la diferencia entre representar la identidad por **índice de clase** (que no generaliza) y por **posición en el espacio** (que sí).

### Por qué los distractores no son la respuesta principal

| Distractor | Por qué no |
|---|---|
| «El etiquetado de los datos es extremadamente costoso» | Cierto en general, pero **no distingue** clasificación de embeddings: el modelo de este lab *también* se entrenó con 8.631 identidades etiquetadas. No es una desventaja *de la clasificación*. |
| «Existirán tantas clases como personas, lo que implica una altísima dimensionalidad de salida» | Es un **síntoma, no la causa**, y es manejable: 8.631 salidas son 4,4 M de parámetros en un modelo de 12 M. Lo fatal no es que la capa sea grande, sino que **su tamaño dependa del número de personas**. |
| «Los modelos de clasificación no tienen buen rendimiento en este tipo de tareas» | **Falso, y el lab lo desmiente**: el clasificador es exactamente lo que se entrena. El paper obtiene 3,22 % de EER con una pérdida de clasificación. Lo que se descarta es la *cabeza*, no el *método de entrenamiento*. |
| «No tienen ninguna desventaja con respecto al visto en clases» | Contradice explícitamente la slide 7. |

> **La distinción que hay detrás** está en el [fundamento de reconocimiento de hablante](/fundamentos/reconocimiento-de-hablante): identificación (1:N, conjunto cerrado) contra verificación (1:1, conjunto abierto). El archivo de pares del lab implementa la segunda, y por eso nunca nombra a nadie: solo compara.

---

## Actividad 2

> **¿Por qué se obtienen N descriptores por cada señal de audio en el procesamiento de VLAD?**

### ✅ **«Porque sirve para mantener información de la ubicación (localidad) desde donde cada descriptor fue extraído»**

La clave está en el nombre: **V**ector of **L**ocally **A**ggregated **D**escriptors. La «L» es esto.

La slide 12 lo plantea como estrategia de diseño explícita:

> *"**Divide each audio signal into frames (local descriptors)**. Aggregate features across time, including only relevant components."*

Y la slide 19 dibuja los `v₁₁, v₁₂, …, v₁N` como los N descriptores locales de **una sola** señal.

**El sentido profundo:** si se colapsara el audio en **un** descriptor antes de agregar, no habría nada que agregar — y eso es precisamente TAP, el promedio temporal, con su **10,48 % de EER contra 3,22 %**. Los N descriptores locales son la materia prima que permite medir *cómo se distribuyen* los residuos respecto a los centroides, y esa distribución es lo que vale 2,9× en error.

En este modelo, los N vienen del eje temporal que el backbone conserva deliberadamente (uno cada 160 ms) mientras colapsa la frecuencia de 257 a 1 — la asimetría analizada en [El Thin ResNet](02-el-thin-resnet-y-la-errata).

> **Con un matiz que el lab permite medir**, y que vale tener presente al responder: esos «descriptores locales» tienen un **campo receptivo de 1,84 segundos** y un **91,3 % de solape** entre vecinos. La localidad es relativa —local frente al enunciado completo, no local en sentido acústico—, pero sigue siendo localidad: cada descriptor está anclado a una posición temporal, y es esa multiplicidad de posiciones lo que da estructura al conjunto que VLAD agrega.

### Por qué los distractores no

| Distractor | Por qué no |
|---|---|
| «Porque debemos generar tantos descriptores como centroides» | **Confunde N con K.** La *salida* de VLAD tiene K = 8 vectores (uno por centroide); la *entrada* tiene N descriptores locales (~102 para un audio típico de VoxCeleb). La pregunta es por los N. Es el distractor más plausible y el que exige leer la fórmula: la `Σ` corre sobre `i = 1…N`, y el índice `k` es otro. |
| «Porque necesitamos separar las frecuencias en slots temporales» | Describe mal el proceso: el backbone hace **lo contrario** — colapsa la frecuencia (257 → 1) y conserva el tiempo. |
| «Para mantener un tamaño de descriptores relativamente bajo, pero sin perder demasiada información» | Eso describe la **reducción de dimensionalidad** (`block_2`, 4.096 → 512), que es otra etapa y ocurre *después* de la agregación. El paper la justifica exactamente así: *"To keep computational and memory requirements low"*. Pero no es el porqué de los N. |

---

## Actividad 3

> **¿Cómo se adapta la función de pertenencia $a_k(x_i)$ para poder ser aprendida end-to-end?**

### ✅ **«Se convierte en una red neuronal con una función softmax a la salida»**

Es la fórmula de la slide 28:

$$\bar{a}_k(x_i) = \frac{e^{\,w_k x_i + b_k}}{\sum_{k'} e^{\,w_{k'} x_i + b_{k'}}}$$

Y es literalmente lo que hace el código, en dos piezas:

```python
self.vlad_conv = nn.Conv2d(512, 10, (7,1), bias=True)   # calcula w_k·x_i + b_k
...
A = exp_cluster_score / exp_cluster_score.sum(dim=-1, keepdim=True)   # el softmax
```

**El porqué:** VLAD clásico usa `argmin_k ‖x_i − c_k‖²`, que **no es derivable** — gradiente cero en casi todas partes, inexistente en las fronteras. Sin gradiente no se puede aprender nada por descenso, y ese es todo el problema que [NetVLAD](/papers/netvlad-arandjelovic-2016) resuelve. La clase lo formula como pregunta antes de responderla: *"In this process we need to find $x_i$, $c_k$ y $a_k(x_i)$. **Is it possible to learn all of them end-to-end?**"*

Y hay una capa más de profundidad que el lab permite verificar: el softmax de distancias negativas es **algebraicamente idéntico** a un softmax lineal, porque el término `−α‖x_i‖²` no depende de `k` y se cancela. Comprobado con error ≤ 10⁻⁴ (precisión de `float32`) para tres valores de α. Eso es lo que permite tratar `w_k` y `b_k` como parámetros **libres**, desacoplados de `c_k` — y es de donde sale la expresividad extra sobre k-means. Está desarrollado en [NetVLAD desarmado](03-netvlad-desarmado).

### Por qué los distractores no

| Distractor | Por qué no |
|---|---|
| «…una función **sigmoidal** a la salida» | Una sigmoide da pertenencias **independientes** por cluster, que no suman 1. Se perdería la **competencia** entre clusters — y esa competencia es justamente lo que hace funcionar a los ghost clusters, que ganan masa **restándola** a los reales. Sin softmax no hay presupuesto que repartir, y GhostVLAD no tendría mecanismo. |
| «Una función que permite obtener la diferencia entre cada descriptor y el centroide más cercano» | Eso es el **residuo** `x_i − c_k`, el *otro* factor de la fórmula. La pertenencia es el peso `ā_k`, no la diferencia. Y «el más cercano» es precisamente el `argmin` que había que eliminar. |
| «No se utiliza, ya que se considera que cada vector pertenece a todos los centroides» | Sería asignación uniforme `ā_k = 1/K`. Medido: equivale a α → 0, con entropía **2,302 de un máximo de 2,303** — colapsa `V` a `x̄ − c̄` y destruye toda la estructura. Pertenecer a todos *por igual* no es lo mismo que pertenecer a todos *con pesos*. |
| «No sufre alteraciones con respecto a la definición original» | Contradice la slide 28, que dice *"is replaced by soft assignment"*. |

---

## Lo que las tres preguntas trazan juntas

Puestas en orden, las tres actividades recorren el argumento completo de la segunda mitad de la [clase 41](/clases/clase-41):

1. **Por qué no clasificar** → porque el conjunto de personas es abierto y una capa de salida es cerrada. *Solución: producir un descriptor.*
2. **Por qué N descriptores locales** → porque un descriptor único no deja nada que agregar, y la agregación es donde está la información. *Solución: conservar el eje temporal.*
3. **Por qué softmax** → porque el `argmin` que agrega bien no es derivable. *Solución: hacerlo suave y aprenderlo todo.*

Cada respuesta habilita la siguiente pregunta, y las tres juntas son la ruta de VLAD 2010 a GhostVLAD 2018. El práctico las plantea como trivia de selección múltiple; leídas en secuencia, son el índice del método.

---

**Anterior:** [Los defectos del notebook](06-los-defectos-del-notebook) · **Volver al** [índice del lab](../)
