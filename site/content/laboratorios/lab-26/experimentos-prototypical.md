---
title: "Experimentos con Prototypical: robustez y límites"
weight: 5
math: true
---

> **Actividades 4, 5 y 6 del laboratorio.** Después de construir Prototypical Networks, toca exprimirlo: mejorar el encoder, variar el número de clases (WAYS) y de ejemplos por clase (SHOTS), y por último saltar de Omniglot al mucho más difícil Mini-ImageNet. Tres actividades, tres lecciones sobre **dónde** metric-based brilla y **dónde** se queda corto. Todos los números son los que medimos realmente en el notebook.

## El recordatorio de qué hace ProtoNet

Antes de leer las tablas conviene tener fresca la mecánica, porque explica casi todos los hallazgos. Prototypical Networks no aprende un clasificador con pesos por clase: aprende un **encoder** $f_\phi$ que mapea cada imagen a un vector de embedding. Para cada tarea few-shot calcula un **prototipo** por clase como el promedio de los embeddings de su *support set*:

$$c_k = \frac{1}{|S_k|} \sum_{(x_i, y_i) \in S_k} f_\phi(x_i)$$

y clasifica un *query* $x$ por la clase cuyo prototipo está más cerca (distancia euclidiana al cuadrado):

$$p_\phi(y=k \mid x) = \frac{\exp(-\lVert f_\phi(x) - c_k \rVert^2)}{\sum_{k'} \exp(-\lVert f_\phi(x) - c_{k'} \rVert^2)}$$

Dos consecuencias que vamos a usar a cada rato: (1) la única pieza entrenable es el encoder, que se optimiza con **SGD directo** sobre la pérdida de clasificación de los queries, sin meta-gradiente de segundo orden como MAML; (2) agregar clases **no agrega parámetros** — solo agrega prototipos en el mismo espacio de embeddings. Esa segunda propiedad es la que hace a ProtoNet robusto al número de clases, como veremos en la Actividad 5.

## Actividad 4 — Mejorar el encoder en Omniglot (4-way 1-shot)

La pregunta es directa: con la misma tarea 4-way 1-shot sobre Omniglot, ¿cuánto gana el encoder al pasar de un MLP a una CNN, y al entrenar más?

| Encoder | Épocas | Train | Valid | Test |
|---|---|---|---|---|
| `OmniglotFC(28², 50)` | 40 | 0.675 | 0.694 | 0.676 |
| `OmniglotCNN(64)` | 40 | 0.879 | 0.894 | 0.877 |
| `OmniglotCNN(64)` | 80 | 0.939 | 0.933 | **0.934** |

**La CNN da +0.201 de un golpe** (0.676 → 0.877 en test), y aquí hay un contraste importante con MAML. En el laboratorio de MAML, cambiar el MLP por la CNN *por sí solo* no ayudaba: la CNN quedaba **subentrenada** porque el meta-gradiente la actualizaba de forma indirecta y lenta. En ProtoNet pasa lo contrario, y la razón es estructural: el encoder se entrena con **SGD directo** sobre la pérdida de los queries. Con 40 épocas × 200 pasos por época son **8000 pasos de gradiente reales** sobre los pesos convolucionales, mucho más eficientes que el gradiente de segundo orden que MAML reparte entre el bucle interno y el externo. ProtoNet "alimenta" mejor a la CNN, así que la capacidad extra de la convolución se traduce de inmediato en accuracy.

**Doblar las épocas (40 → 80) refina hasta 0.934.** La ganancia marginal cae (de +0.201 a +0.057), porque las curvas de entrenamiento ya se están aplanando y el `lr_scheduler` ha ido recortando el learning rate — al final del entrenamiento el LR está dividido por **16** respecto al inicial, así que cada paso mueve los pesos muy poco. Es el régimen de afinamiento fino, no de aprendizaje grueso.

Un detalle sano: **no hay overfitting en ninguna fila**. Train, valid y test van prácticamente pegados (0.939 / 0.933 / 0.934 en la mejor config). Omniglot es un dataset "amable" — caracteres manuscritos limpios, alta separabilidad — y el encoder no tiene incentivo para memorizar.

### La comparación que importa: ProtoNet vs. MAML en Omniglot

El mejor resultado de MAML en este mismo 4-way 1-shot fue **0.877**, conseguido con **400 iteraciones** y **cinco hiperparámetros afinados** (learning rate interno, externo, número de pasos del bucle interno, tamaño del meta-batch, etc.). ProtoNet **iguala ese 0.877 con solo 40 épocas** y lo **supera con 80 épocas (0.934)**, siendo estructuralmente mucho más simple: no hay bucle interno, no hay segundo orden, no hay learning rate de adaptación que afinar. Esto confirma la tesis de Snell *et al.* (2017): en few-shot de imágenes, lo **metric-based iguala o supera a lo optimization-based con menos maquinaria**. Cuando una métrica simple en el espacio de embeddings basta, no hace falta aprender *cómo aprender* con gradientes anidados.

![ProtoNet Omniglot mejor (80 épocas)](/laboratorios/lab-26/protonet-omniglot-best.jpg)

## Actividad 5 — Variar WAYS y SHOTS en Omniglot

Aquí movemos las dos perillas de la tarea, manteniendo fijo el encoder ganador (`OmniglotCNN(64)`, 40 épocas). Para comparar de forma justa entre configuraciones con distinto azar, reportamos también el **accuracy normalizado**:

$$\text{norm} = \frac{\text{acc} - \text{azar}}{1 - \text{azar}}, \qquad \text{azar} = \frac{1}{\text{WAYS}}$$

El normalizado responde "¿qué fracción del margen sobre el azar logró capturar el modelo?". Un 2-way con 0.50 de azar y un 8-way con 0.125 de azar no son comparables en crudo; el normalizado los pone en la misma escala.

**Eje WAYS (con SHOTS = 1):**

| WAYS | Azar | Test crudo | Test normalizado |
|---|---|---|---|
| 2 | 0.500 | 0.892 | 0.784 |
| 4 | 0.250 | 0.896 | 0.861 |
| 8 | 0.125 | 0.890 | 0.874 |

**Eje SHOTS (con WAYS = 4):**

| SHOTS | Azar | Test crudo | Test normalizado |
|---|---|---|---|
| 1 | 0.250 | 0.896 | 0.861 |
| 5 | 0.250 | **0.977** | 0.970 |

### El hallazgo espectacular: el accuracy crudo es CONSTANTE

Mira la columna "Test crudo" del eje WAYS: **0.892, 0.896, 0.890**. Pasar de distinguir 2 clases a distinguir 8 **no degrada el accuracy crudo** — se mantiene clavado en ~0.89. ProtoNet es **robusto al número de clases**.

El contraste con MAML es brutal. En la Actividad 2 (MAML sobre Omniglot), el accuracy crudo **se desploma** al subir WAYS: **0.932 → 0.797 → 0.657** (2-way, 4-way, 8-way). ¿Por qué la diferencia? Es exactamente la consecuencia (2) del recordatorio inicial:

- **MAML tiene una capa clasificadora de tamaño WAYS.** Más clases significan más pesos en la cabeza, más fronteras de decisión que aprender, y todo eso debe emerger del bucle interno con poquísimos ejemplos. La tarea de 8-way es genuinamente más difícil de *adaptar*.
- **ProtoNet solo agrega prototipos en el espacio de embeddings.** No añade ni un parámetro. Si el encoder aprendió a separar bien las clases, entonces distinguir 8 prototipos bien dispersos es **casi tan fácil** como distinguir 2: el query cae cerca del centroide correcto y lejos de los demás, independientemente de cuántos centroides haya.

Esta es una **ventaja fundamental del enfoque metric-based: escalabilidad al número de clases**. Y no es un detalle académico — es directamente relevante para *patient matching*, donde el "número de clases" es, en el límite, el número de entidades distintas (millones de pacientes). Un clasificador con cabeza por clase no escala ahí; un comparador en espacio de embeddings sí. Es la misma intuición que hay detrás de usar embeddings + distancia para *record linkage* en lugar de un clasificador multiclase gigante.

### SHOTS: el prototipo mejora cuando promedia más

En el eje SHOTS la mejora es nítida: **0.896 → 0.977** al pasar de 1 a 5 ejemplos por clase. La explicación es la definición misma del prototipo. Con 1-shot, el prototipo $c_k$ **es** el único embedding disponible — una muestra ruidosa del centroide verdadero de la clase. Con 5-shot, $c_k$ es el **promedio de 5 embeddings**, y promediar reduce la varianza del estimador del centroide (en el caso ideal, por un factor $\sqrt{5}$). Un prototipo mejor estimado cae más cerca del centro real de su clase, las fronteras quedan más limpias, y el accuracy sube hacia 0.977. ProtoNet aprovecha cada shot extra de forma casi gratuita, porque "usar más datos" es literalmente "promediar más términos en una media".

### Un matiz honesto sobre la comparación con MAML

La comparación de WAYS contra MAML **no es 100% justa**: el experimento de MAML 8-way usó 150 iteraciones y quedó algo subentrenado, lo que infla artificialmente su caída. Si MAML hubiera entrenado más, el desplome sería menos dramático. Pero la **robustez interna de ProtoNet sí es genuina**: dentro del propio ProtoNet, con el mismo encoder y las mismas épocas, el crudo no se mueve de 2 a 8 clases. Esa constancia no depende de ninguna comparación externa.

## Actividad 6 — Prototypical en Mini-ImageNet: las tres palancas

Mini-ImageNet es otro mundo: imágenes naturales a color, alta variabilidad intra-clase, fondos complejos. El baseline 4-way 1-shot con `MiniImagenetCNN(50)` a 40 épocas da **Test 0.377** (normalizado 0.169). Es bajo en absoluto, pero ya **supera al mejor MAML 4w1s (0.324)** en el mismo dataset — de nuevo metric-based gana a optimization-based. Y aparece algo que en Omniglot no existía: **overfitting** (train > valid), inducido por la dificultad del dataset.

| Config | Test | Azar | Normalizado | Brecha train − valid |
|---|---|---|---|---|
| 4w1s `MiniImagenetCNN(50)` | 0.377 | 25% | 0.169 | 0.044 |
| 4w5s `MiniImagenetCNN(50)` | **0.632** | 25% | **0.509** | 0.126 |
| 2w1s `MiniImagenetCNN(50)` | 0.539 | 50% | 0.078 | 0.018 |
| 4w1s `ResNet12` | 0.362 | 25% | 0.150 | 0.053 |

Las tres palancas que se exploran son las mismas que en clase, pero ahora sobre un encoder que ya no separa tan bien las clases. Eso cambia todo.

### Palanca 1 — SHOTS↑: la ganadora (con letra chica)

Subir de 1 a 5 shots es la jugada que más rinde: el normalizado **se triplica** (0.169 → 0.509) y el crudo salta de 0.377 a 0.632. Es la misma física de la Actividad 5: promediar 5 embeddings da un prototipo mucho mejor estimado, y eso ayuda **aún más** cuando los embeddings son ruidosos, porque hay más ruido que cancelar. Pero la letra chica importa: la **brecha train − valid casi se triplica** (0.044 → 0.126). Más shots por tarea hacen al modelo más capaz de ajustarse a las clases de entrenamiento, y por tanto **agrava el overfitting**. La palanca que más sube el rendimiento es también la que más erosiona la generalización.

### Palanca 2 — WAYS↓: un espejismo del azar

Bajar de 4-way a 2-way **parece** mejorar: el crudo sube de 0.377 a 0.539. Pero es una ilusión. El azar del 2-way es 0.50, así que el normalizado **baja** a 0.078 (contra 0.169 del 4-way). El modelo de 2 clases acierta más en crudo simplemente porque adivinar al azar ya le da 50%; descontado el azar, **captura menos margen real** que el de 4 clases.

Y aquí está lo más interesante: esto es **lo contrario de lo que vimos en Omniglot**, donde el normalizado *subía* con más WAYS (0.784 → 0.874). ¿Cómo puede ser robusto al WAYS en un dataset y un espejismo en el otro? La respuesta unifica las dos actividades:

> La robustez al número de clases **depende de la calidad de los embeddings**. Embeddings buenos (Omniglot) → prototipos bien separados → distinguir muchas clases es fácil → robustez genuina. Embeddings pobres (Mini-ImageNet 1-shot) → prototipos solapados → el accuracy crudo solo refleja el azar → "robustez" aparente que se desvanece al normalizar.

ProtoNet no es mágicamente escalable; es escalable **cuando el encoder hace bien su trabajo**. El metric-based hereda la calidad del espacio de representación que aprende, y en 1-shot sobre imágenes naturales ese espacio todavía no está lo bastante separado.

### Palanca 3 — Más capacidad (ResNet12): no mejora

Cambiar la CNN por un `ResNet12` —un encoder con muchísima más capacidad— da **0.362 en test, peor que el 0.377 de la CNN simple**. Más parámetros, peor resultado. La razón es el mismo patrón que vimos con la CNN en MAML (Actividad 1): **capacidad extra sin más épocas, sin data augmentation y sin regularización adicional no rinde**. Un ResNet12 necesita un presupuesto de entrenamiento (y trucos anti-overfitting) acorde a su tamaño; lanzarlo con la misma receta que la CNN pequeña solo le da más formas de no generalizar. La capacidad es una palanca **condicional**: ayuda si la acompañas de los recursos que exige, y estorba si no.

![ProtoNet Mini-ImageNet 4w5s](/laboratorios/lab-26/protonet-miniimagenet-5shot.jpg)

### La tensión que deja Mini-ImageNet

Las tres palancas dibujan un compromiso incómodo:

- **SHOTS↑** sube el rendimiento real (la única que sube el normalizado de verdad) **pero agrava el overfitting**.
- **WAYS↓** *mitiga* el overfitting (la brecha cae a 0.018, la menor de todas) **pero no aporta poder real** — su mejora es puro azar.
- **Más capacidad** no ayuda sin el presupuesto que exige.

No hay una palanca gratis. La que mejora generaliza peor, la que generaliza mejor no mejora, y la de fuerza bruta no rinde. Es el retrato honesto de few-shot en un dominio difícil: con pocos datos y embeddings imperfectos, cada decisión de diseño cobra un precio en otra dimensión. La salida real —que excede este lab— pasa por mejorar el **espacio de embeddings** (pre-entrenamiento, augmentation, regularización), porque de su calidad cuelgan todas las demás propiedades, incluida la robustez al WAYS.

## La síntesis de las tres actividades

| Pregunta | Omniglot | Mini-ImageNet |
|---|---|---|
| ¿La CNN ayuda? | Sí, +0.201 inmediato | Sí, pero ResNet12 no |
| ¿Robusto al WAYS? | Sí (crudo constante ~0.89) | No (espejismo del azar) |
| ¿Más SHOTS ayudan? | Sí (0.896 → 0.977) | Sí (0.377 → 0.632), pero overfit |
| ¿Supera a MAML? | Sí (0.934 vs 0.877) | Sí (0.377 vs 0.324) |
| ¿Overfitting? | No | Sí |

ProtoNet sale del laboratorio como un método **más simple y a menudo más fuerte que MAML** en few-shot de imágenes, cuya virtud central —escalar al número de clases— está **condicionada a la calidad de sus embeddings**. Donde el encoder separa bien (Omniglot), es robusto y eficiente; donde no (Mini-ImageNet 1-shot), su robustez es aparente y todas las palancas entran en tensión. La lección que se lleva uno a problemas reales como *patient matching* es doble: el enfoque metric-based es el correcto para espacios de muchas entidades, pero **lo que decide su éxito es el espacio de representación**, no el clasificador.

## Enlaces

- **Fundamentos:** [Metric Learning](/fundamentos/metric-learning) · [Few-shot learning](/fundamentos/few-shot-learning)
- **Páginas:** [Prototypical: fundamentos](prototypical-fundamentos) · [Experimentos con MAML](experimentos-maml) · [Comparación y teoría](comparacion-y-teoria)
- **Papers:** [Prototypical Networks (Snell 2017)](/papers/prototypical-networks-snell-2017)
