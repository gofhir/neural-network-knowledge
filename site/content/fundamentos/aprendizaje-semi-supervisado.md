---
title: "Aprendizaje Semi-Supervisado y Consistency Training"
weight: 104
math: true
---

El **aprendizaje semi-supervisado** (*semi-supervised learning*, SSL) ataca uno de los cuellos de botella más caros del machine learning aplicado: **etiquetar datos**. Recolectar ejemplos crudos —imágenes, textos, registros clínicos— es barato y abundante; ponerles una etiqueta confiable requiere personas, tiempo y, en dominios como salud, expertos. El SSL parte de esa asimetría: dispone de un **pequeño conjunto etiquetado** y un **gran conjunto sin etiquetar**, y busca exprimir el segundo para mejorar lo que el primero, por sí solo, no alcanza. Este fundamento recorre las ideas clásicas del paradigma, profundiza en la **regularización por consistencia** —la familia que dominó los benchmarks hacia 2019— y aterriza en **UDA** ([Xie et al. 2019](/papers/uda-xie-2019)), el paper que sostiene el [Laboratorio 28](/laboratorios/lab-28) y articula la [Clase 28](/clases/clase-28).

---

## 1. Qué es el aprendizaje semi-supervisado

Formalmente, el SSL dispone de dos fuentes de datos: un conjunto etiquetado $\mathcal{L} = \{(x_i, y_i)\}_{i=1}^{n}$ y un conjunto no etiquetado $\mathcal{U} = \{x_j\}_{j=1}^{m}$, casi siempre con $m \gg n$ (a veces miles de veces mayor). El objetivo es aprender un clasificador $p_\theta(y \mid x)$ que aproveche **ambos**, en vez de descartar $\mathcal{U}$ como hace el aprendizaje supervisado puro.

La razón por la que esto puede funcionar no es magia: descansa en **supuestos sobre la estructura de los datos**. Sin alguna suposición que conecte la distribución $p(x)$ de los datos sin etiquetar con la distribución condicional $p(y \mid x)$ que queremos aprender, los datos no etiquetados son inútiles. Los tres supuestos clásicos del SSL son:

- **Suavidad (*smoothness*):** si dos puntos están cerca en una región densa del espacio, sus etiquetas deberían ser iguales.
- **Cluster:** los datos forman grupos, y los puntos de un mismo cluster tienden a compartir etiqueta; por tanto, las fronteras de decisión deberían pasar por regiones de **baja densidad**, no atravesar clusters.
- **Variedad (*manifold*):** los datos viven en una variedad de dimensión mucho menor que el espacio ambiente, y la etiqueta varía suavemente sobre esa variedad.

Conviene ubicar el SSL en un espectro según cuánta supervisión usa. No son compartimentos estancos sino un continuo: a medida que las etiquetas escasean, el modelo debe extraer cada vez más señal de la estructura de los propios datos.

| Paradigma | Etiquetas | Idea central |
|---|---|---|
| Supervisado | Todas | Aprender $p(y\mid x)$ directo de pares $(x,y)$. |
| **Semi-supervisado** | Pocas + mucho $\mathcal{U}$ | Pocos labels guían; $\mathcal{U}$ refina la frontera. |
| Autosupervisado | Ninguna (pretexto) | Inventar una tarea pretexto a partir de los propios datos. |
| No supervisado | Ninguna | Estructura pura: clustering, densidad, dimensionalidad. |

El SSL importa porque el coste de etiquetar es, en la práctica, la barrera dominante para desplegar aprendizaje profundo en dominios reales. Una imagen médica anotada por un radiólogo, un texto clínico codificado por un experto o un par de registros validados como coincidencia por un especialista en *record linkage* pueden costar minutos de trabajo cualificado cada uno; reunir decenas de miles de ejemplos así es prohibitivo. Mientras tanto, los datos crudos —escáneres sin anotar, notas clínicas sin codificar, millones de registros sin emparejar— se acumulan casi gratis. El aprendizaje profundo supervisado tradicional **descarta** todo ese material por no estar etiquetado; el SSL lo recupera. La promesa cuantitativa es lo que vuelve atractivo el paradigma: como veremos en la sección 4, en algunos benchmarks 20 etiquetas más datos sin anotar igualan a 25.000 etiquetas, una reducción de tres órdenes de magnitud en el esfuerzo de anotación.

{{< concept-alert type="clave" >}}
El aprendizaje semi-supervisado **no inventa información**: amplifica la poca que hay. Los pocos ejemplos etiquetados anclan el significado de las clases; el enorme conjunto sin etiquetar moldea la **geometría de la frontera de decisión** empujándola hacia regiones de baja densidad. Si los supuestos de suavidad/cluster no se cumplen, los datos sin etiquetar no ayudan —e incluso pueden dañar.
{{< /concept-alert >}}

---

## 2. Las grandes ideas clásicas

Antes de la regularización por consistencia moderna, tres familias de técnicas concentraban casi todo el SSL. No son excluyentes; los métodos actuales las combinan.

El **self-training** (o **pseudo-labeling**) es la más antigua e intuitiva. Se entrena un modelo con el poco $\mathcal{L}$ disponible, se usa para **predecir etiquetas** sobre $\mathcal{U}$, y las predicciones más confiadas se agregan al conjunto de entrenamiento como si fueran verdaderas (*pseudo-etiquetas*). Se reentrena y se repite. Es simple y agnóstico al modelo, pero arrastra un peligro estructural: el **error de confirmación** (*confirmation bias*). Si el modelo se equivoca con confianza, esa pseudo-etiqueta errónea se refuerza a sí misma en la siguiente ronda.

La **minimización de entropía** (*entropy minimization*, Grandvalet y Bengio 2005) formaliza el supuesto de cluster: añade un término a la pérdida que **penaliza la incertidumbre** del modelo sobre los datos no etiquetados, empujando $p_\theta(y\mid x)$ hacia distribuciones puntiagudas (de baja entropía). El efecto neto es alejar la frontera de decisión de las regiones densas. Es un ingrediente, no un método completo: suele combinarse con los otros dos.

La **regularización por consistencia** (*consistency regularization*) es la idea que dominó 2017–2020 y la protagonista de este fundamento (sección 3). En una frase: el modelo debe **predecir lo mismo** para un dato y para una versión ligeramente perturbada de ese dato.

| Familia | Mecanismo | Riesgo / límite |
|---|---|---|
| Self-training / pseudo-labeling | Etiquetar $\mathcal{U}$ con el propio modelo y reentrenar | Error de confirmación; refuerza sus propios errores |
| Entropy minimization | Penalizar la entropía de $p_\theta(y\mid x)$ sobre $\mathcal{U}$ | Por sí solo no basta; colapso si no hay otra señal |
| Consistency regularization | Forzar invarianza a perturbaciones de la entrada | Depende críticamente de qué perturbación se use |

---

## 3. Consistency training en detalle

La regularización por consistencia traduce el supuesto de suavidad en una pérdida concreta. La intuición es geométrica: *un buen modelo debería ser robusto a pequeños cambios en su entrada*. Si perturbo levemente un ejemplo de una manera que **no cambia su clase**, la predicción del modelo tampoco debería cambiar.

El esqueleto que comparten todos estos métodos tiene dos pasos. Dada una entrada sin etiquetar $x$:

1. Computar la predicción del modelo sobre el original, $p_\theta(y \mid x)$, y sobre una versión perturbada, $p_\theta(y \mid \hat{x})$, donde $\hat{x}$ se obtiene aplicando una transformación que preserva la etiqueta.
2. Minimizar una **divergencia** entre ambas distribuciones de salida. La elección canónica es la **divergencia de Kullback-Leibler**:

$$
\mathcal{L}_{\text{cons}} = \mathbb{E}_{x \sim \mathcal{U}} \; D_{\mathrm{KL}}\!\big(p_{\tilde\theta}(y \mid x) \,\big\|\, p_\theta(y \mid \hat{x})\big), \qquad D_{\mathrm{KL}}(p \,\|\, q) = \sum_{y} p(y)\,\log\frac{p(y)}{q(y)}.
$$

Dos detalles que parecen menores pero son decisivos. Primero, la predicción del original $p_{\tilde\theta}(y\mid x)$ se trata como un **objetivo fijo**: se aplica *stop-gradient* sobre $\tilde\theta$ (una copia congelada de $\theta$), de modo que el gradiente fluye solo por la rama del ejemplo aumentado, que debe **imitar** la del original. Sin esto, el modelo podría minimizar la divergencia degenerando hacia predicciones constantes para todo. Segundo, como la entropía del objetivo es constante respecto a $\theta$ bajo el stop-gradient, minimizar la cross-entropy entre ambas ramas **equivale** a minimizar la KL —por eso ambas formulaciones aparecen indistintamente en la literatura.

Visto desde otro ángulo, la pérdida de consistencia **propaga las etiquetas** por vecindad: si $x$ y $\hat{x}$ están conectados por una transformación de bajo riesgo y uno de ellos está cerca de un ejemplo etiquetado, el conocimiento de la etiqueta fluye a través de esa conexión hacia el resto de la región. La frontera de decisión se ve forzada a no atravesar el "puente" entre $x$ y sus versiones transformadas, y termina asentándose en regiones de baja densidad —exactamente lo que pide el supuesto de cluster de la sección 1—.

En la práctica, la consistencia se combina con la pérdida supervisada en un único objetivo, ponderando la consistencia con un coeficiente $\lambda$ que suele crecer gradualmente durante el entrenamiento (un *ramp-up*): al principio el modelo aún no predice nada sensato sobre $\mathcal{U}$, así que forzar consistencia sobre predicciones basura solo introduce ruido; a medida que el modelo madura con los pocos labels, se le da más peso a la regularización no supervisada. Hasta 2018, la pregunta que distinguía a cada método era **dónde y cómo** generar la perturbación $\hat{x}$, y el catálogo era amplio: ruido gaussiano aditivo (Pseudo-ensemble), *dropout* (Π-Model), recortes y volteos aleatorios (Temporal Ensembling), ruido **adversarial** que apunta a la dirección de máxima sensibilidad del modelo (VAT, Miyato et al. 2018), o consistencia entre dos copias del modelo en el espacio de parámetros (Mean Teacher). El supuesto tácito común —y el que UDA derriba— era que cualquier perturbación pequeña servía por igual.

---

## 4. UDA: la calidad de la augmentación es el cuello de botella

Como vimos en la sección 3, hasta 2018 los métodos de consistencia se diferenciaban por la perturbación que usaban —ruido gaussiano, *dropout*, adversarial, recortes—, todos bajo el supuesto de que cualquier perturbación pequeña servía. **UDA** (*Unsupervised Data Augmentation*, [Xie et al. 2019](/papers/uda-xie-2019)) demuele esa premisa: lo que limita el rendimiento **no es el algoritmo de propagación, sino la calidad del ruido**.

La tesis es de una simplicidad desarmante. El aprendizaje supervisado ya había descubierto **augmentaciones de datos** mucho mejores que el ruido aleatorio —RandAugment y Cutout en visión, back-translation en NLP—. UDA propone usar **esas mismas augmentaciones de alta calidad como fuente de perturbación** en la pérdida de consistencia, aplicadas sin etiquetas sobre $\mathcal{U}$. La hipótesis —verificada empíricamente— es que la mejor augmentación en el régimen supervisado es también la mejor en el semi-supervisado. El objetivo combinado de UDA suma la cross-entropy supervisada habitual sobre $\mathcal{L}$ y la consistencia (KL) sobre $\mathcal{U}$, ponderada por $\lambda$:

$$
\min_\theta \; \mathbb{E}_{(x,y)\sim\mathcal{L}}\big[-\log p_\theta(y\mid x)\big] \;+\; \lambda\, \mathbb{E}_{x\sim\mathcal{U}}\,\mathbb{E}_{\hat{x}\sim q(\hat{x}\mid x)}\Big[ D_{\mathrm{KL}}\big(p_{\tilde\theta}(y\mid x)\,\|\,p_\theta(y\mid\hat{x})\big)\Big].
$$

Aquí $q(\hat{x}\mid x)$ es la transformación de augmentación —no ruido gaussiano—. Tres razones explican por qué las augmentaciones avanzadas baten al ruido simple: son **ruido válido** (generan ejemplos realistas que preservan la etiqueta), **ruido diverso** (permiten cambios grandes sin alterar la clase, mejorando la eficiencia muestral) y **sesgos inductivos dirigidos** (codifican la invarianza específica que la tarea necesita: a rotaciones, a paráfrasis, a iluminación).

Las augmentaciones por dominio son:

- **RandAugment** (imágenes): muestrea uniformemente de 15 transformaciones de imagen (rotación, contraste, *shear*, *solarize*, etc.) con magnitud aleatoria. No requiere búsqueda de política ni etiquetas.
- **Back-translation** (texto): traduce $x$ a otro idioma y de vuelta, obteniendo una **paráfrasis**. Usa muestreo con temperatura $0.7$–$0.9$ para equilibrar el *trade-off* diversidad–validez (temperatura $0$ degenera en paráfrasis idénticas; temperatura $1$ produce texto diverso pero ilegible).
- **Reemplazo por TF-IDF** (clasificación de tópicos): sustituye palabras poco informativas (bajo TF-IDF) y conserva las decisivas (alto TF-IDF).

Para que esto funcione en regímenes con **poquísimas etiquetas**, UDA añade tres técnicas auxiliares que el Lab 28 expone directamente:

- **TSA (*Training Signal Annealing*):** con 20 labels y millones de ejemplos sin etiquetar, el modelo **sobreajusta** los pocos labels antes de aprovechar $\mathcal{U}$. TSA libera la señal supervisada gradualmente: si la probabilidad predicha de la clase correcta supera un umbral $\eta_t$ que crece de $1/K$ a $1$, ese ejemplo se **remueve** de la pérdida supervisada, evitando el sobre-entrenamiento de los ejemplos fáciles.
- **Confidence masking:** la consistencia se computa **solo** sobre ejemplos no etiquetados donde el modelo ya está seguro (probabilidad máxima $> \beta$), para no propagar predicciones ruidosas.
- **Sharpening:** la distribución objetivo del original se **afila** con una temperatura softmax baja $\tau < 1$, acercándola a *one-hot* y reforzando predicciones confiadas (en línea con la minimización de entropía de la sección 2).

Los resultados son contundentes. En **IMDb con solo 20 ejemplos etiquetados**, UDA alcanza **4.20%** de error, superando al modelo supervisado entrenado con las **25.000** etiquetas completas (1.250× más datos). En CIFAR-10 con 4.000 labels iguala al supervisado con 50.000; en SVHN con 1.000 labels logra 2.23%; en ImageNet al 10% de etiquetas sube el top-1 de 58.84 a **68.78**. Y a diferencia de muchos métodos de SSL que solo brillan con pocos datos, UDA **escala al régimen de muchos datos**: con el 100% de las etiquetas de ImageNet más 1.3M de imágenes externas filtradas por dominio, todavía mejora el top-1 de 78.43 a 79.05.

### El grafo de augmentación: por qué 20 etiquetas bastan

UDA ofrece una justificación teórica elegante de por qué tan pocas etiquetas pueden bastar. Bajo tres supuestos sobre la augmentación —que sea **in-domain** (los aumentados son ejemplos plausibles), **label-preserving** ($f^*(x)=f^*(\hat{x})$) y **reversible** (si $q(\hat{x}\mid x)>0$ entonces $q(x\mid\hat{x})>0$)— se construye un grafo donde cada ejemplo es un nodo y hay una arista entre $x$ y $\hat{x}$ cuando la augmentación los conecta. Como la augmentación preserva la etiqueta, los ejemplos de clases distintas viven en **componentes desconectados** distintos del grafo.

La intuición clave: basta **un solo ejemplo etiquetado por componente** para propagar su etiqueta al resto del componente recorriéndolo vía augmentación. La augmentación supervisada solo alcanza a los **vecinos directos** del nodo etiquetado; la consistencia no supervisada de UDA **recorre el componente completo**. De aquí se sigue una consecuencia que cierra el círculo con la tesis empírica: augmentaciones **más diversas** generan más aristas, mejor conectividad y, por tanto, **menos componentes** —y menos componentes significa menos etiquetas necesarias—. La diversidad de RandAugment y back-translation no es cosmética: es exactamente lo que reduce el número de componentes y vuelve suficientes 20 etiquetas. Esto explica también por qué el *trade-off* diversidad–validez de la temperatura en back-translation es tan delicado: demasiada validez (temperatura baja) deja el grafo fragmentado; demasiada diversidad (temperatura alta) crea aristas que rompen el supuesto *label-preserving* y conectan clases distintas.

---

## 5. Relación con el aprendizaje contrastivo

La regularización por consistencia y el [aprendizaje contrastivo](/fundamentos/aprendizaje-contrastivo) comparten una misma raíz conceptual: **imponer invarianza a transformaciones que preservan el contenido**. En ambos, dos vistas aumentadas del mismo dato deberían "verse igual" para el modelo —en consistency training, igual *predicción de clase*; en contrastivo, *representaciones cercanas* en el espacio latente—. Las augmentaciones (RandAugment, recortes, color jitter) son, de hecho, las mismas.

La diferencia clave está en cómo evitan el **colapso** (que el modelo prediga lo mismo para todo y minimice trivialmente la pérdida):

- El **aprendizaje contrastivo necesita negativos**: además de acercar las dos vistas del mismo ejemplo (positivos), debe **alejar** las vistas de ejemplos distintos (negativos), como en SimCLR o MoCo. El contraste positivo/negativo es lo que impide el colapso.
- El **consistency training no usa negativos**: lo que lo salva del colapso es la **señal supervisada** (la cross-entropy sobre $\mathcal{L}$, que ancla las clases) combinada con el *stop-gradient* sobre el objetivo. La invarianza se impone solo entre original y aumentado, sin comparar contra otros ejemplos.

Ambos son, además, primos del [aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado): las transformaciones que el SSL usa para regularizar son las mismas que el autosupervisado usa para construir tareas pretexto. UDA es precisamente la **bisagra** entre los tres paradigmas: muestra que las invarianzas descubiertas para el supervisado son exactamente las que se necesitan para explotar datos sin etiquetar.

---

## 6. La familia posterior

UDA, junto con **MixMatch** (Berthelot et al. 2019, trabajo paralelo), inauguró la era moderna de la regularización por consistencia. Los métodos que siguieron son refinamientos de la misma receta:

- **Mean Teacher** (Tarvainen y Valpola 2017, anterior pero influyente): el "profesor" que genera el objetivo no es una copia congelada instantánea, sino un **promedio móvil exponencial (EMA)** de los pesos del estudiante. El objetivo es más estable y suaviza el error de confirmación.
- **MixMatch** (2019): combina pseudo-etiquetado, *sharpening* y la interpolación **MixUp** entre ejemplos etiquetados y no etiquetados en una sola pérdida unificada.
- **FixMatch** (Sohn et al. 2020): la síntesis minimalista de UDA y MixMatch que se volvió estándar. Genera una pseudo-etiqueta a partir de una augmentación **débil** del dato; si la confianza supera un umbral, fuerza consistencia entre esa pseudo-etiqueta y una augmentación **fuerte** del mismo dato. "Augmentación débil → pseudo-etiqueta confiada → consistencia con augmentación fuerte" es el patrón que UDA pavimentó.

La complejidad de comparar tantos métodos motivó **USB** (*Unified Semi-supervised learning Benchmark*, Wang et al. 2022), una librería que implementa una docena de algoritmos de SSL (UDA, FixMatch, MixMatch, Mean Teacher y más) sobre visión, texto y audio con protocolos de evaluación homogéneos —el punto de referencia actual para reproducir y comparar SSL.

---

## 7. Conexión con la Clase 28 y el Laboratorio 28

La [Clase 28](/clases/clase-28) dedica su **Sección 2 —"Autosupervisión para potenciar el aprendizaje supervisado"** íntegramente a esta idea. El mensaje de fondo es que las transformaciones que el autosupervisado usa para crear tareas pretexto (rotaciones, recortes, paráfrasis) sirven **también** para regularizar el aprendizaje supervisado con datos sin etiquetar: el SSL no compite con lo supervisado, lo **potencia**. La clase usa UDA y su tabla de ImageNet (10% de etiquetas) como ilustración de que la consistencia rinde incluso con etiquetas escasas, y la divergencia KL entre la predicción del dato y la de su versión transformada como objetivo central.

El [Laboratorio 28](/laboratorios/lab-28) (`Practico_Autosupervision_UDA`) es la implementación práctica de esta receta. El estudiante ensambla la **pérdida combinada** (cross-entropy supervisada + KL de consistencia ponderada por $\lambda$), genera las **augmentaciones** (RandAugment en visión, back-translation en texto) y experimenta con las técnicas auxiliares —**TSA**, **confidence masking**, **sharpening**— que vuelven viable el régimen de poquísimas etiquetas. Comprender *por qué* cada pieza está donde está —por qué el *stop-gradient*, por qué afilar, por qué enmascarar por confianza— es la diferencia entre ejecutar el notebook y entender el mecanismo.

{{< concept-alert type="resumen" >}}
El **aprendizaje semi-supervisado** explota un gran conjunto sin etiquetar para mejorar lo que pocos labels permiten, apoyándose en los supuestos de suavidad y cluster. La **regularización por consistencia** los traduce en una pérdida: el modelo debe predecir lo mismo (minimizar la KL) para un dato y su versión perturbada. **UDA** muestra que la palanca no es el algoritmo sino la **calidad de la augmentación** —RandAugment, back-translation— y con TSA + masking + sharpening logra 4.20% de error en IMDb con solo 20 etiquetas. Comparte con el aprendizaje contrastivo la invarianza a transformaciones, pero no necesita negativos: la señal supervisada y el stop-gradient lo salvan del colapso.
{{< /concept-alert >}}

---

## Para profundizar

- [Unsupervised Data Augmentation for Consistency Training (Xie et al. 2019)](/papers/uda-xie-2019) — el paper del Lab 28: augmentaciones de calidad como fuente de ruido en consistencia, TSA, confidence masking, sharpening.

**Fundamentos relacionados:** [Aprendizaje Autosupervisado](/fundamentos/aprendizaje-autosupervisado) · [Aprendizaje Contrastivo](/fundamentos/aprendizaje-contrastivo) · [Data Augmentation](/fundamentos/data-augmentation) · [Clase 28](/clases/clase-28) · [Laboratorio 28](/laboratorios/lab-28)
