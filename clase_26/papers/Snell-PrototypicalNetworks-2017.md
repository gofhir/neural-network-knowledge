# Prototypical Networks for Few-shot Learning — Análisis interno exhaustivo

## 1. Metadata y resumen ejecutivo

- **Título:** *Prototypical Networks for Few-shot Learning*
- **Autores:** Jake Snell (University of Toronto), Kevin Swersky (Twitter), Richard S. Zemel (University of Toronto, Vector Institute)
- **Venue:** NeurIPS 2017 (Advances in Neural Information Processing Systems).
- **arXiv:** 1703.05175v2, 19 de junio de 2017.
- **Subcampo:** few-shot / zero-shot classification, metric learning, meta-learning.

**Resumen ejecutivo.** El paper propone *prototypical networks* (en adelante ProtoNets), un modelo para clasificación con pocos ejemplos (*few-shot classification*), donde el clasificador debe generalizar a clases que no aparecieron en el entrenamiento, disponiendo apenas de un puñado de ejemplos por clase nueva. La idea es deliberadamente simple: aprender un espacio de *embedding* mediante una red neuronal $f_\phi$ tal que los puntos de una clase se agrupen alrededor de un único punto representativo, el **prototipo**, calculado como la media (centroide) de los *embeddings* de los ejemplos de soporte de esa clase. Un punto de consulta nuevo se clasifica buscando el prototipo más cercano según una distancia en ese espacio. La clasificación se formaliza como un *softmax* sobre las distancias negativas a los prototipos.

La tesis central tiene tres patas:

1. **Sesgo inductivo simple.** Bajo escasez extrema de datos, un sesgo inductivo simple —"una clase es un centroide"— evita el sobreajuste mejor que las arquitecturas complejas con atención o meta-aprendizaje basado en LSTM.
2. **La distancia importa, y la teoría dice cuál.** Usar **distancia euclidiana cuadrada** (en lugar de la similitud coseno, que era la norma en *matching networks*) mejora sustancialmente los resultados. Y no es un truco empírico: con divergencias de Bregman, el clasificador prototípico equivale a una **estimación de densidad por mezcla** (*mixture density estimation*) con densidades de la familia exponencial. El coseno no es una divergencia de Bregman, así que rompe esa equivalencia.
3. **Resultados estado del arte con un modelo más barato.** ProtoNets alcanza, o supera, a *matching networks* (Vinyals et al. 2016) y al *meta-learner LSTM* (Ravi & Larochelle 2017) en Omniglot y miniImageNet, y establece estado del arte en zero-shot sobre CUB-200, todo con menos parámetros y sin componentes auxiliares (sin *fully-conditional embedding*, sin LSTM meta-aprendiz).

Números clave que ancla el resto del análisis: en miniImageNet 5-way 1-shot ProtoNets logra **49.42 ± 0.78%** y en 5-way 5-shot **68.20 ± 0.66%** con euclidiana; en Omniglot 20-way 1-shot **96.0%**; en CUB zero-shot 50-way **54.6%**. Cada cifra se desarrolla en la sección 10.

---

## 2. Contexto: Matching Networks y el deseo de un modelo más simple

El problema de *few-shot classification* es viejo en la literatura (Miller et al. 2000; Lake et al. 2011; Koch 2015), pero el paper se sitúa explícitamente como respuesta a dos trabajos recientes y muy influyentes que habían empujado el estado del arte:

**Matching Networks (Vinyals et al. 2016).** Introdujo dos ideas que se volvieron estándar. Primera, un clasificador que es esencialmente un **vecino más cercano ponderado** dentro de un espacio de *embedding* aprendido: para un punto de consulta $x$, la predicción es una suma ponderada de las etiquetas de soporte, donde los pesos vienen de un mecanismo de **atención** $a(x, x_i)$ —típicamente un *softmax* sobre similitudes coseno entre el *embedding* de la consulta y el de cada punto de soporte:

$$
p(y \mid x, S) = \sum_{i} a(x, x_i)\, y_i, \qquad a(x, x_i) = \frac{\exp\!\big(\cos(f(x), g(x_i))\big)}{\sum_{j} \exp\!\big(\cos(f(x), g(x_j))\big)}.
$$

Segunda, y quizás más duradera, el **entrenamiento episódico**: en vez de batches ordinarios, se muestrean *episodios* que imitan la tarea de test —se submuestrean clases y, dentro de ellas, puntos de soporte y consulta— de modo que la distribución de entrenamiento coincida con la de evaluación. Vinyals lo resume como "test and train conditions must match". Matching networks añadía además refinamientos costosos: *full context embeddings* (FCE), donde el *embedding* de cada punto depende del resto del episodio vía un LSTM bidireccional, y la opción de desacoplar las funciones de *embedding* de soporte ($g$) y consulta ($f$).

**Meta-Learner LSTM (Ravi & Larochelle 2017).** Llevó el episodio aún más lejos hacia el meta-aprendizaje: entrenó un LSTM para *producir las actualizaciones* de un clasificador dado un episodio, aprovechando que la dinámica de una celda LSTM y un paso de descenso por gradiente se pueden escribir de la misma forma. En vez de entrenar un solo modelo sobre múltiples episodios, el meta-aprendiz aprende a *entrenar un modelo a medida* para cada episodio.

El diagnóstico de Snell et al. es que ambos enfoques atacan el sobreajuste con maquinaria sofisticada (atención sobre todo el soporte, LSTM que aprende a optimizar), cuando el problema fundamental —escasez de datos— pide lo contrario: **un sesgo inductivo fuerte y simple**. La apuesta del paper es que con la decisión de diseño correcta (la media como prototipo + la distancia correcta) se puede igualar o superar a esos modelos sin su complejidad. Es, en esencia, una navaja de Occam aplicada al few-shot.

El argumento de sobreajuste merece énfasis porque es el motor de todo el diseño. Un enfoque ingenuo —reentrenar la red sobre las pocas muestras de las clases nuevas— sobreajustaría catastróficamente: con 1 o 5 ejemplos por clase, los grados de libertad de una red profunda superan ampliamente la información disponible. La literatura ya había mostrado que los humanos hacen *one-shot classification* con alta precisión, lo que sugiere que el truco no es más capacidad, sino el sesgo inductivo correcto. ProtoNets traslada toda la capacidad de aprendizaje a la fase de *meta-entrenamiento* (donde sí hay muchos datos, en las clases base) y deja la fase de *adaptación a clases nuevas* sin parámetros que ajustar: el clasificador nuevo es función cerrada del soporte (la media). Así el few-shot deja de ser un problema de optimización en datos escasos y pasa a ser un problema de *recuperación geométrica* en un espacio bien construido.

---

## 3. La idea central: el prototipo como centroide del soporte

**Notación.** Se da un conjunto de soporte $S = \{(x_1, y_1), \dots, (x_N, y_N)\}$ de $N$ ejemplos etiquetados, con $x_i \in \mathbb{R}^D$ y $y_i \in \{1, \dots, K\}$. $S_k$ denota el subconjunto de ejemplos con etiqueta $k$.

ProtoNets calcula una representación de $M$ dimensiones, $c_k \in \mathbb{R}^M$ —el **prototipo**— para cada clase, a través de una función de *embedding* $f_\phi : \mathbb{R}^D \to \mathbb{R}^M$ con parámetros aprendibles $\phi$ (una CNN). El prototipo es la **media vectorial** de los puntos de soporte embebidos de su clase:

$$
c_k = \frac{1}{|S_k|} \sum_{(x_i, y_i) \in S_k} f_\phi(x_i). \tag{1}
$$

Esta es la ecuación que define todo el método. Conceptualmente: cada clase queda resumida en un solo vector, su centroide en el espacio aprendido. La consecuencia práctica es enorme para *inferencia*: a diferencia de matching networks (que necesita conservar y comparar contra *todos* los puntos de soporte), ProtoNets solo necesita conservar $K$ prototipos. La representación de una clase es **concisa e independiente del número de ejemplos** de soporte. Esto importa en producción: el costo de predicción no crece con el tamaño del soporte.

El nombre "prototipo" remite a la psicología cognitiva (teoría de prototipos de categorías): una categoría se representa por su ejemplar central idealizado. La hipótesis del modelo es que *existe un espacio de embedding donde esta abstracción es válida*, y la red se entrena para fabricar ese espacio.

**Una observación sobre el "shot" y la varianza del prototipo.** El promedio de la Ecuación (1) también tiene un efecto estadístico de reducción de ruido: con $|S_k|$ ejemplos, el centroide es un estimador de la media de la clase con varianza que decae como $1/|S_k|$. Por eso el salto de 1-shot a 5-shot mejora consistentemente la precisión en todos los experimentos: el prototipo de 5-shot es una estimación mucho más estable del "centro" de la clase que el de 1-shot (donde el prototipo es un solo punto ruidoso). Esta intuición frecuentista se vuelve formal en la sección 5, donde el prototipo aparece como el estimador de máxima verosimilitud de la media de una densidad de la familia exponencial.

---

## 4. La clasificación: softmax sobre distancias negativas

Dada una función de distancia $d : \mathbb{R}^M \times \mathbb{R}^M \to [0, +\infty)$, ProtoNets produce una distribución sobre clases para un punto de consulta $x$ mediante un *softmax* sobre las **distancias negativas** a los prototipos:

$$
p_\phi(y = k \mid x) = \frac{\exp\!\big(-d(f_\phi(x), c_k)\big)}{\sum_{k'} \exp\!\big(-d(f_\phi(x), c_{k'})\big)}. \tag{2}
$$

La intuición es directa: cuanto más cerca está $f_\phi(x)$ del prototipo $c_k$, menor es $d$, mayor es $-d$, mayor la probabilidad de la clase $k$. El *softmax* convierte el conjunto de distancias en una distribución de probabilidad bien definida.

**Entrenamiento.** Se minimiza la log-verosimilitud negativa de la clase verdadera $k$ vía SGD:

$$
J(\phi) = -\log p_\phi(y = k \mid x).
$$

El detalle crucial es que los episodios de entrenamiento *replican* la tarea de test. La pseudocódigo del **Algorithm 1** lo formaliza:

- $N_C \le K$ clases por episodio (el "way"), seleccionadas aleatoriamente: `V ← RANDOM_SAMPLE({1,...,K}, N_C)`.
- $N_S$ ejemplos de soporte por clase (el "shot"): `S_k ← RANDOM_SAMPLE(D_{V_k}, N_S)`.
- $N_Q$ ejemplos de consulta por clase, tomados de lo que sobra: `Q_k ← RANDOM_SAMPLE(D_{V_k} \ S_k, N_Q)`.
- Se calcula cada prototipo $c_k$ promediando los soportes embebidos.
- La pérdida acumula, sobre todas las clases y consultas del episodio:

$$
J \leftarrow J + \frac{1}{N_C N_Q}\Big[\, d(f_\phi(x), c_k) + \log \sum_{k'} \exp\big(-d(f_\phi(x), c_{k'})\big)\Big].
$$

Este término es exactamente $-\log p_\phi(y=k\mid x)$ reescrito: el primer sumando es la distancia a la clase correcta (que queremos minimizar), el segundo es el *log-sum-exp* normalizador. Por episodio, los gradientes fluyen a través tanto de $f_\phi(x)$ (consulta) como de los $c_k$ (que dependen de $f_\phi$ aplicado a los soportes), de modo que la red aprende a colocar consultas cerca de sus centroides y lejos de los ajenos. Todo es entrenable end-to-end con SGD ordinario —no hay fase de partición ni meta-optimizador separado.

**Nota sobre el doble rol de $f_\phi$ y el flujo de gradiente.** Un punto sutil pero importante: la misma red $f_\phi$ embebe los soportes (para formar $c_k$) y la consulta. Esto significa que cuando se minimiza la distancia $\|f_\phi(x) - c_k\|^2$, el gradiente empuja *simultáneamente* la representación de la consulta hacia el centroide y el centroide (vía los soportes) hacia la consulta. La red no aprende a memorizar posiciones absolutas, sino a estructurar el espacio para que la regla "consulta cerca de su centroide" se cumpla sobre episodios *nuevos y aleatorios*. La aleatoriedad de la composición episódica (clases distintas cada vez) es lo que fuerza la generalización: el espacio debe servir para particiones de clase que nunca se vieron juntas. Comparado con un clasificador estándar de $K$ salidas fijas, aquí no hay una capa de clasificación con pesos por clase; la "capa de clasificación" se reconstruye en cada episodio a partir de los datos (de ahí su naturaleza no paramétrica, sección 13).

---

## 5. La distancia: por qué euclidiana (Bregman) supera al coseno

Esta es la contribución teórica más original del paper, y la que lo distingue de un simple "centroide + vecino más cercano".

### 5.1 Divergencias de Bregman

Una **divergencia de Bregman regular** $d_\varphi$ se define como:

$$
d_\varphi(z, z') = \varphi(z) - \varphi(z') - (z - z')^\top \nabla\varphi(z'), \tag{3}
$$

donde $\varphi$ es una función diferenciable, estrictamente convexa, de tipo Legendre. Geométricamente, $d_\varphi(z, z')$ es la diferencia entre $\varphi(z)$ y su aproximación lineal de primer orden alrededor de $z'$ (el "gap" entre la curva convexa y su tangente). Ejemplos: la **distancia euclidiana cuadrada** $\|z - z'\|^2$ (con $\varphi(z) = \|z\|^2$) y la **distancia de Mahalanobis**.

### 5.2 El prototipo (media) es óptimo bajo Bregman

Banerjee et al. (2005, "Clustering with Bregman divergences") probaron un resultado clave: para *cualquier* divergencia de Bregman, el representante de un cluster que minimiza la distancia total a sus puntos asignados **es la media del cluster**. Es decir,

$$
\arg\min_{c} \sum_{z \in \text{cluster}} d_\varphi(z, c) = \frac{1}{|\text{cluster}|}\sum_{z} z.
$$

Por tanto, calcular el prototipo como media (Ecuación 1) **no es arbitrario**: produce el representante óptimo del soporte de cada clase, *siempre que la distancia sea de Bregman*. Si uno eligiera el coseno (que no es de Bregman), la media dejaría de ser el minimizador y la coherencia entre cómo se construye el prototipo y cómo se mide la distancia se rompe.

### 5.3 Equivalencia con estimación de densidad por mezcla

El resultado más profundo conecta ProtoNets con modelos generativos. Cualquier distribución de la familia exponencial regular $p_\psi(z \mid \theta)$, con parámetros $\theta$ y función cumulante $\psi$, se puede escribir en términos de una divergencia de Bregman únicamente determinada (Banerjee et al.):

$$
p_\psi(z \mid \theta) = \exp\{z^\top \theta - \psi(\theta) - g_\psi(z)\} = \exp\{-d_\varphi(z, \mu(\theta)) - g_\varphi(z)\}. \tag{4}
$$

Consideremos un modelo de mezcla de la familia exponencial con parámetros $\Gamma = \{\theta_k, \pi_k\}_{k=1}^{K}$:

$$
p(z \mid \Gamma) = \sum_{k=1}^{K} \pi_k\, p_\psi(z \mid \theta_k) = \sum_{k=1}^{K} \pi_k \exp\big(-d_\varphi(z, \mu(\theta_k)) - g_\varphi(z)\big). \tag{5}
$$

La inferencia de la asignación de cluster $y$ para un punto no etiquetado $z$ es:

$$
p(y = k \mid z) = \frac{\pi_k \exp\big(-d_\varphi(z, \mu(\theta_k))\big)}{\sum_{k'} \pi_{k'} \exp\big(-d_\varphi(z, \mu(\theta_{k'}))\big)}. \tag{6}
$$

El término $g_\varphi(z)$ se cancela porque aparece igual en numerador y denominador. Ahora el remate: **para una mezcla equiponderada** (todas las $\pi_k$ iguales) **con un cluster por clase**, la Ecuación (6) es *idéntica* a la predicción de ProtoNets (Ecuación 2), con la identificación $f_\phi(x) = z$ y $c_k = \mu(\theta_k)$.

La conclusión es elegante: **ProtoNets realiza implícitamente estimación de densidad por mezcla de la familia exponencial** en el espacio de *embedding*, y la divergencia $d_\varphi$ elegida *especifica las suposiciones de modelado sobre la distribución condicional de clase*. La euclidiana cuadrada corresponde a **gaussianas esféricas** (covarianza isotrópica). El coseno no corresponde a ninguna densidad de la familia exponencial vía esta construcción —no es Bregman— y por eso, conjeturan los autores, funciona peor: rompe la interpretación probabilística que hace coherente al método.

Esto es lo que eleva ProtoNets de "heurística razonable" a "modelo principiado".

### 5.4 Por qué el coseno falla, en concreto

Vale la pena hacer explícito el mecanismo del fracaso del coseno, porque es el resultado empírico más citado del paper (brecha de ~17 puntos en miniImageNet 5-shot, sección 10.3). La distancia coseno solo depende de la *dirección* de los vectores, no de su norma: $\cos(u, v) = \frac{u^\top v}{\|u\|\,\|v\|}$. Bajo coseno, todos los puntos se proyectan implícitamente sobre la hiperesfera unitaria. Pero la media aritmética de la Ecuación (1) **no vive en esa hiperesfera**: el promedio de varios vectores unitarios tiene norma menor que 1 y apunta en una dirección que no minimiza la distancia coseno total a los puntos (el minimizador coseno sería la dirección media normalizada, una cantidad distinta). Hay, pues, una *incoherencia interna*: se construye el prototipo según un criterio (media euclidiana) y se mide según otro (ángulo). Con euclidiana cuadrada esa incoherencia desaparece porque la media *es* el minimizador, y la geometría del clustering, de la distancia y de la densidad gaussiana esférica encajan. Por eso el efecto es más pronunciado en ProtoNets que en matching networks: matching networks no promedia (compara contra puntos individuales), así que sufre menos la incoherencia media-vs-ángulo.

---

## 6. Relación con Matching Networks

ProtoNets y matching networks **coinciden exactamente en el caso 1-shot** y divergen para $K > 1$ shots:

- **1-shot.** Con un solo punto de soporte por clase, $c_k = f_\phi(x_k)$: el prototipo *es* ese único ejemplo. El *softmax* sobre distancias a prototipos colapsa al vecino más cercano ponderado de matching networks. Son el mismo modelo. (Por eso en la Figura 2 del paper las barras 1-shot de ambos métodos son idénticas, y en las tablas aparecen fusionados como "MATCHING NETS / PROTONETS" en 1-shot.)

- **K-shot ($K>1$).** Aquí está la diferencia conceptual. Matching networks aplica **atención sobre todos los puntos de soporte individuales** (vecino más cercano ponderado sobre $N$ puntos). ProtoNets primero **promedia** los soportes de cada clase en un prototipo y luego compara contra los $K$ prototipos. Cuando se usa euclidiana cuadrada, ProtoNets resulta ser un **clasificador lineal** (ver sección 7); matching networks sigue siendo un clasificador no paramétrico sobre todos los puntos.

**Lo que ProtoNets descarta deliberadamente.** Vinyals et al. proponían extensiones: desacoplar los *embeddings* de soporte y consulta, y la FCE (*fully-conditional embedding*), un segundo nivel de *embedding* que toma en cuenta los puntos específicos del episodio vía un LSTM bidireccional. Snell et al. observan que la FCE (i) aumenta el número de parámetros aprendibles y (ii) **impone un orden arbitrario** sobre el conjunto de soporte (un conjunto no tiene orden natural, pero un LSTM lo procesa secuencialmente). ProtoNets prescinde de todo esto y usa **el mismo encoder** para soporte y consulta. El argumento es que, con tan pocos datos, un sesgo inductivo simple basta: no hace falta aprender un *embedding* a medida por episodio.

**Sobre múltiples prototipos por clase.** ¿Y si se usaran varios prototipos por clase (para clases multimodales)? Eso requeriría un esquema de partición de los puntos de soporte dentro de cada clase, desacoplado de las actualizaciones de pesos (como en Mensink et al. 2013 y Rippel et al. 2016, que usan k-means en pre-proceso). ProtoNets evita ese paso: un prototipo por clase, aprendible con descenso por gradiente ordinario.

---

## 7. Reinterpretación como modelo lineal; relación con clustering

### 7.1 El clasificador lineal escondido

Cuando $d$ es la euclidiana cuadrada, $d(z, z') = \|z - z'\|^2$, el modelo de la Ecuación (2) equivale a un **modelo lineal** en el espacio de *embedding*, con una parametrización particular (resultado heredado de Mensink et al.). Expandiendo el exponente:

$$
-\|f_\phi(x) - c_k\|^2 = -f_\phi(x)^\top f_\phi(x) + 2c_k^\top f_\phi(x) - c_k^\top c_k. \tag{7}
$$

El primer término, $-f_\phi(x)^\top f_\phi(x)$, **es constante respecto de la clase $k$**, así que se cancela en el *softmax* (numerador y denominador lo comparten). Los términos restantes se escriben como un modelo lineal:

$$
2c_k^\top f_\phi(x) - c_k^\top c_k = w_k^\top f_\phi(x) + b_k, \quad \text{con } w_k = 2c_k,\ b_k = -c_k^\top c_k. \tag{8}
$$

Es decir, el clasificador es lineal en $f_\phi(x)$: pesos $w_k = 2c_k$ y sesgo $b_k = -\|c_k\|^2$. Esto podría sonar como una limitación ("¿solo un clasificador lineal?"), pero los autores lo defienden: **toda la no linealidad necesaria se aprende dentro de $f_\phi$**. Es exactamente lo que hacen las redes modernas de clasificación (AlexNet, GoogLeNet): un *backbone* no lineal seguido de una capa lineal *softmax*. La elección de euclidiana cuadrada corresponde, además, a densidades gaussianas esféricas (sección 5).

### 7.2 Clustering y nearest class mean

ProtoNets se ubica entre dos tradiciones:

- **Clustering duro.** El cálculo del prototipo es exactamente *clustering duro* del soporte: un cluster por clase, cada punto asignado al de su etiqueta, y el representante es la media —óptima bajo Bregman (sección 5.2).
- **Nearest Class Mean (Mensink et al. 2013).** Representa cada clase por la media de sus ejemplos para incorporar clases nuevas sin reentrenar. La diferencia: Mensink usa *embedding lineal* y supone *muchos* ejemplos por clase nueva; ProtoNets usa *embedding no lineal* aprendido por red neuronal y *entrenamiento episódico* para el régimen few-shot.

Frente a NCA no lineal (Salakhutdinov & Hinton 2007), el parecido es grande —red neuronal + *softmax* sobre distancias euclidianas— pero la distinción clave es que ProtoNets forma el *softmax* **directamente sobre clases** (vía prototipos), no sobre puntos individuales. Esto da la representación concisa por clase y evita almacenar todo el soporte en inferencia.

---

## 8. Protocolo episódico: el efecto del "way"

El entrenamiento episódico (heredado de matching networks) es central, pero ProtoNets hace un hallazgo empírico contraintuitivo y muy citado:

**Conviene entrenar con MÁS "way" (más clases por episodio) que las que se usarán en test.** Si en test se hará clasificación 5-way, *no* conviene entrenar con episodios 5-way; conviene entrenar con 20-way o 30-way. La conjetura: la mayor dificultad de la tarea 20-way **fuerza a la red a tomar decisiones más finas en el espacio de embedding**, lo que mejora la generalización. Es una forma de currículo difícil que regulariza.

Los datos lo confirman. En miniImageNet (Tabla 6, apéndice), 5-way 1-shot test:
- entrenar con way=5 → **46.14%**; con way=10 → 48.27%; con way=15 → 48.60%; con way=20 → 48.57%; con way=30 → **49.42%**.

La mejora de way=5 a way=30 es de ~3.3 puntos absolutos, gratis, solo cambiando la composición de episodios.

**Sobre el "shot".** A diferencia del "way", para el número de soportes ("shot") los autores encuentran que **lo mejor es igualar shot de entrenamiento y de test**. La asimetría es interesante: subir el "way" ayuda, pero el "shot" debe coincidir. En Tabla 6, para 5-shot test, entrenar con 5-shot rinde mucho mejor (68.20% con way=20) que entrenar con 1-shot (65.04% con way=20). Y hay un punto de retornos decrecientes / reversión en el "way" para 5-shot: el óptimo está alrededor de way=20, y subir a way=30 lo degrada (66.79%).

Estos detalles —reportados en las Tablas 4, 5 y 6 del apéndice— convirtieron el "ajuste del way" en una práctica estándar para reproducir resultados few-shot.

**Por qué subir el "way" regulariza (lectura más profunda).** El *softmax* normaliza sobre las clases del episodio. Con 5-way, el modelo solo necesita separar la clase correcta de otras 4; con 20-way o 30-way, debe separarla de 19 o 29 distractores simultáneamente. El término de log-sum-exp en la pérdida penaliza la cercanía a *cualquier* prototipo ajeno, así que más clases por episodio significa más restricciones de margen por paso de gradiente. El espacio de embedding resultante tiene que ser globalmente más fino, no solo localmente discriminativo. Es el mismo principio que el muestreo de negativos difíciles en aprendizaje contrastivo: más negativos por consulta endurece la tarea y mejora la representación. El límite —ver la reversión en 5-shot con way=30— sugiere que hay un punto donde la tarea se vuelve tan difícil que el "shot" disponible (la estimación del prototipo) ya no alcanza para resolverla con fiabilidad, y el entrenamiento se degrada.

---

## 9. Variante zero-shot: meta-datos de clase como prototipo

ProtoNets se extiende a **zero-shot learning** (cero ejemplos de la clase nueva) con un cambio mínimo y elegante. En zero-shot, en vez de un conjunto de soporte, cada clase trae un **vector de meta-datos** $v_k$ (atributos que describen la clase: color, forma, patrones de plumaje, etc.; predefinidos o aprendidos de texto crudo).

La modificación: en vez de promediar soportes, se define el prototipo como un *embedding* de los meta-datos vía una **segunda función de embedding** $g_\vartheta$:

$$
c_k = g_\vartheta(v_k).
$$

El resto del modelo es idéntico: *softmax* sobre distancias del *embedding* de la imagen $f_\phi(x)$ a los prototipos $c_k$. Como la imagen y el meta-dato vienen de **dominios distintos**, hay dos funciones de *embedding* separadas ($f_\phi$ para imágenes, $g_\vartheta$ para atributos) que mapean a un **espacio compartido**. Empíricamente ayuda **fijar el embedding del prototipo $g$ a norma unitaria** (no así el de la consulta $f$), para alinear las escalas entre dominios. La Figura 1(b) ilustra el caso: tres vectores de meta-datos $v_1, v_2, v_3$ se embeben en prototipos $c_1, c_2, c_3$, y la imagen $x$ se clasifica por proximidad.

---

## 10. Experimentos

### 10.1 Omniglot (few-shot)

Omniglot (Lake et al. 2011): 1623 caracteres manuscritos de 50 alfabetos, 20 ejemplos por carácter (cada uno dibujado por una persona distinta). Pre-proceso siguiendo a Vinyals: imágenes en escala de grises redimensionadas a $28 \times 28$, clases aumentadas con rotaciones de múltiplos de 90°. División: 1200 caracteres + rotaciones para entrenar (**4,800 clases** en total), el resto para test.

**Arquitectura del encoder** (la que se volvió el "backbone canónico" del few-shot, conv-4): cuatro bloques convolucionales, cada uno con convolución $3 \times 3$ de **64 filtros** + batch normalization + ReLU + max-pooling $2 \times 2$. Sobre imágenes $28 \times 28$ esto produce un espacio de salida de **64 dimensiones**. Mismo encoder para soporte y consulta. Optimización: SGD con Adam, learning rate inicial $10^{-3}$, reducido a la mitad cada 2000 episodios. Sin regularización salvo batch norm. Entrenamiento con 60 clases y 5 consultas por clase por episodio. Accuracy promediada sobre **1000 episodios de test**.

Resultados (Tabla 1), euclidiana, sin fine-tune:

| Configuración | ProtoNets | Matching Nets (cosine) | Neural Statistician |
|---|---|---|---|
| 5-way 1-shot | **98.8%** | 98.1% | 98.1% |
| 5-way 5-shot | **99.7%** | 98.9% | 99.5% |
| 20-way 1-shot | **96.0%** | 93.8% | 93.2% |
| 20-way 5-shot | **98.9%** | 98.5% | 98.1% |

ProtoNets es estado del arte en todas las configuraciones. La ventaja es más notoria en 20-way (tarea más difícil): +2.2 puntos sobre matching networks en 20-way 1-shot.

### 10.2 miniImageNet (few-shot)

miniImageNet (Vinyals et al. 2016): subconjunto de ILSVRC-12, 60,000 imágenes a color de $84 \times 84$, 100 clases con 600 ejemplos cada una. Se usan los *splits* de Ravi & Larochelle (64 train / 16 validación / 20 test) para comparar directamente. El mismo encoder de 4 bloques aquí produce un espacio de **1600 dimensiones** (imágenes mayores). Entrenamiento: **30-way para 1-shot, 20-way para 5-shot**, igualando shot train/test, 15 consultas por clase. Accuracy sobre **600 episodios de test** con intervalos de confianza al 95%.

Resultados (Tabla 2), 5-way, sin fine-tune:

| Modelo | Dist. | 1-shot | 5-shot |
|---|---|---|---|
| Baseline Nearest Neighbors | Cosine | 28.86 ± 0.54% | 49.79 ± 0.79% |
| Matching Networks | Cosine | 43.40 ± 0.78% | 51.09 ± 0.71% |
| Matching Networks FCE | Cosine | 43.56 ± 0.84% | 55.31 ± 0.73% |
| Meta-Learner LSTM | — | 43.44 ± 0.77% | 60.60 ± 0.71% |
| **Prototypical Networks** | **Euclid.** | **49.42 ± 0.78%** | **68.20 ± 0.66%** |

ProtoNets gana **por amplio margen**: ~6 puntos sobre el mejor baseline en 1-shot, y ~7.6 puntos sobre el Meta-Learner LSTM en 5-shot. Que un modelo más simple supere al meta-aprendiz LSTM es el golpe retórico del paper.

### 10.3 Ablación euclidiana vs coseno y efecto del way

La Figura 2 y la Tabla 5 (apéndice) aíslan dos factores con una implementación propia de matching networks que comparte arquitectura. Datos de la Tabla 5 (miniImageNet, 5-way test):

- **Euclidiana >> coseno.** Para ProtoNets 5-shot, way=20: coseno **51.48%** vs euclidiana **68.20%** — una brecha de ~17 puntos. El efecto es aún más marcado en ProtoNets que en matching networks, justamente porque la media (cómo se construye el prototipo) está naturalmente alineada con la euclidiana y *no* con el coseno (que no es Bregman).
- **20-way > 5-way.** Para 1-shot (donde Matching=Proto): coseno 5-way 38.82% vs coseno 20-way 43.63%; euclidiana 5-way 46.61% vs euclidiana 20-way 49.17%.

La Tabla 6 da el barrido fino del "way" (sección 8).

### 10.4 CUB-200 (zero-shot)

Caltech-UCSD Birds 200-2011 (CUB): 11,788 imágenes de 200 especies de aves. *Splits*: 100 train / 50 validación / 50 test. Features de imagen: 1,024 dimensiones extraídas con **GoogLeNet** sobre cinco recortes (medio, cuatro esquinas) de la imagen original y su espejo horizontal (en test, solo el recorte central). Meta-datos: vectores de atributos continuos de **312 dimensiones** (color, forma, patrones). Se aprende un **mapeo lineal simple** sobre ambos (features de imagen 1024-d y atributos 312-d) a un espacio compartido de 1,024 dimensiones, normalizando los prototipos (atributos embebidos) a norma unitaria. Episodios de 50 clases, 10 consultas por clase. SGD/Adam, lr fijo $10^{-4}$, weight decay $10^{-5}$, early stopping.

Resultados (Tabla 3), 50-way 0-shot:

| Modelo | Features | Accuracy |
|---|---|---|
| ALE | Fisher | 26.9% |
| SJE | AlexNet | 40.3% |
| Sample Clustering | AlexNet | 44.3% |
| SJE | GoogLeNet | 50.1% |
| DS-SJE | GoogLeNet | 50.4% |
| DA-SJE | GoogLeNet | 50.9% |
| **Proto. Nets** | **GoogLeNet** | **54.6%** |

Estado del arte por margen amplio (+3.7 puntos sobre DA-SJE) entre métodos que usan atributos como meta-datos. Demuestra que el método es general: funciona aun cuando los puntos (imágenes) y las clases (atributos) viven en dominios distintos.

### 10.5 Visualización t-SNE

El apéndice incluye una visualización t-SNE (Figura 3) de los *embeddings* aprendidos sobre un subconjunto del alfabeto Tengwar (clases de test). Aun cuando los caracteres son variaciones menores entre sí, la red agrupa los manuscritos ajustadamente alrededor de los prototipos de clase (marcados en negro), con unos pocos mal clasificados resaltados en rojo. Es evidencia cualitativa de que el espacio aprendido tiene la geometría que la teoría supone (clusters compactos por clase).

---

## 11. Por qué importa: simplicidad + estado del arte

El impacto de ProtoNets no se explica solo por las cifras, sino por la **relación coste/beneficio**:

1. **Simplicidad como característica, no como compromiso.** Una sola ecuación (media + softmax sobre distancias) iguala o supera arquitecturas con atención, LSTM bidireccionales y meta-optimizadores. En el régimen de datos escasos, el sesgo inductivo fuerte gana. Es una lección que trasciende el few-shot.
2. **Eficiencia en inferencia.** $K$ prototipos en vez de todo el soporte. Costo de predicción independiente del tamaño del soporte. Atractivo para producción.
3. **Principios, no truco.** La conexión con divergencias de Bregman y estimación de densidad por mezcla da una *justificación teórica* a por qué la media y por qué la euclidiana. Convierte una intuición en un modelo derivable.
4. **Baseline de facto.** ProtoNets se volvió la línea base obligatoria del *metric-based few-shot learning*. El encoder conv-4 (4 bloques de 64 filtros) y el protocolo de evaluación (5-way 1-shot / 5-shot, 600/1000 episodios, IC 95%) que el paper consolidó son hoy estándar de la literatura. Casi todo paper posterior de few-shot (Relation Networks, TADAM, métodos transductivos, etc.) se compara contra ProtoNets.

---

## 12. Limitaciones

El paper es honesto sobre sus supuestos; varios se volvieron blanco de trabajo posterior:

1. **Clusters unimodales.** Un prototipo por clase asume que cada clase forma un único *blob* compacto (gaussiana esférica) en el espacio de embedding. Clases multimodales (p. ej. "perro" con razas muy distintas) violan ese supuesto. El propio paper menciona que múltiples prototipos requerirían partición; trabajo posterior (Infinite Mixture Prototypes, etc.) abordó esto.
2. **Embedding fijo, no adaptado por tarea.** El encoder $f_\phi$ queda **congelado tras el entrenamiento**. No hay adaptación específica al episodio (a diferencia de la FCE de matching networks o del meta-aprendiz LSTM). Es deliberado —de ahí la simplicidad— pero significa que el mismo espacio debe servir para todas las tareas de test. Si una tarea nueva requiere atender dimensiones que el espacio aprendido ignora, ProtoNets no puede reajustarse. (TADAM y métodos de modulación condicionada a la tarea atacaron esto.)
3. **Mezcla equiponderada.** La equivalencia con mixture density estimation supone $\pi_k$ iguales (clases balanceadas). Bajo desbalance de clases en el soporte, el supuesto se quiebra.
4. **Solo euclidiana en la práctica.** Aunque el marco admite cualquier divergencia de Bregman, los autores reportan que aprender una varianza por dimensión por clase (Mahalanobis aprendida) **no dio ganancias empíricas**, conjeturando que el encoder ya tiene suficiente flexibilidad. La promesa teórica de divergencias más ricas quedó como trabajo futuro sin frutos inmediatos.
5. **Shift de dominio.** Si las clases de test provienen de una distribución muy distinta de las de entrenamiento (cambio de dominio, p. ej. entrenar en imágenes naturales y testear en imágenes médicas), el embedding fijo puede no transferir. El método supone que train y test comparten la estructura de bajo nivel del espacio.

---

## 13. Legado y conexión con la Clase 26 (métodos no-paramétricos) y salud

**Lugar en la Clase 26 (métodos no-paramétricos).** ProtoNets es un ejemplar limpio de método **no paramétrico** en el sentido relevante para el aprendizaje few-shot: el clasificador para una tarea nueva **no tiene parámetros propios entrenados sobre esa tarea**; se construye sobre la marcha a partir de los datos (los prototipos *son* función directa del soporte, no pesos aprendidos por gradiente sobre la clase nueva). Esto lo emparenta con k-NN y con kernel methods: la "memoria" del modelo son los ejemplos (resumidos en centroides), no un vector de pesos por clase. Encaja en la familia de **metric-based meta-learning**: aprender una métrica/espacio donde la regla de decisión no paramétrica (vecino al prototipo) funcione. Es el contrapunto perfecto a los métodos *optimization-based* (Meta-Learner LSTM, y más tarde MAML), que sí adaptan parámetros por tarea. La dicotomía métrico vs optimización es uno de los ejes organizadores del few-shot que la clase explora.

**Relevancia para salud y oncología (FALP).** El few-shot importa precisamente donde los datos etiquetados son escasos y la cola de clases es larga —el caso de las **patologías raras** y de subtipos poco frecuentes:

- **Clasificación de patologías raras.** Para un cáncer raro o un subtipo histológico con apenas un puñado de casos etiquetados, reentrenar un clasificador profundo sobreajustaría. ProtoNets permitiría aprender un espacio de embedding sobre las patologías frecuentes (abundantes) y luego clasificar una clase rara con 1–5 ejemplos calculando su centroide. La representación concisa por clase (un prototipo) es operacionalmente cómoda: agregar un subtipo nuevo es calcular una media, sin reentrenar.
- **Zero-shot por meta-datos clínicos.** La variante zero-shot mapea directamente a escenarios donde una condición se describe por **atributos estructurados** (en el mundo FHIR: codificaciones, características clínicas, descriptores fenotípicos) en vez de imágenes etiquetadas. Embeber esos atributos como prototipo y comparar contra el embedding del caso es un patrón aplicable a triage o sugerencia de codificación.
- **Cautela honesta.** Las limitaciones de la sección 12 pesan en salud: las clases médicas suelen ser **multimodales** (una patología con presentaciones heterogéneas viola el supuesto unimodal), y el **shift de dominio** entre centros/equipos de imagen es la regla, no la excepción. El embedding fijo de ProtoNets puede no transferir entre instituciones sin recalibración. Sirve como baseline fuerte y como pieza de un sistema (p. ej., como *blocker*/recuperador), pero no como clasificador final autónomo en escenarios de seguridad.

**Legado técnico.** ProtoNets cristalizó el paradigma metric-based, fijó el protocolo de evaluación y el backbone conv-4, e inspiró una familia: Relation Networks (aprende la métrica con una red en vez de fijar euclidiana), TADAM (condiciona el embedding a la tarea, atacando la limitación del embedding fijo), métodos transductivos e Infinite Mixture Prototypes (rompe el supuesto unimodal con varios prototipos por clase). Su combinación de simplicidad, fundamento teórico y resultados lo mantiene, casi una década después, como la primera línea base que cualquier trabajo de few-shot reporta.
