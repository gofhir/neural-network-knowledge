---
title: "Siamese Networks (One-shot)"
weight: 265
math: true
---

{{< paper-card
    title="Siamese Neural Networks for One-shot Image Recognition"
    authors="Gregory Koch, Richard Zemel, Ruslan Salakhutdinov"
    year="2015"
    venue="ICML Deep Learning Workshop 2015"
    pdf="/papers/siamese-networks-koch-2015.pdf" >}}
En vez de entrenar un clasificador de $N$ clases fijas, se entrena una red con **dos torres gemelas que comparten pesos** para responder una pregunta más simple: *¿estas dos imágenes son de la misma clase?* Una vez aprendida esa función de similitud sobre clases conocidas, se reutiliza —sin reentrenar— para clasificar clases nunca vistas con un solo ejemplo. En Omniglot 20-way one-shot alcanza **92.0%**, frente a **95.5%** de humanos y **95.2%** de HBPL. Es uno de los papers fundacionales del *deep metric learning* y del *few-shot learning*.
{{< /paper-card >}}

---

## El problema

El aprendizaje supervisado clásico necesita muchos ejemplos por clase y un conjunto de clases conocido de antemano. Cuando aparece una clase nueva, hay que reentrenar el modelo: caro, lento, e imposible si solo se tienen uno o dos ejemplos. Los humanos no funcionan así: ante un símbolo nuevo entienden el concepto al instante y reconocen variaciones futuras del mismo.

El **one-shot learning** lleva esto al extremo: se observa **un único ejemplo** de cada clase posible antes de tener que predecir la clase de una instancia de test. (Conviene no confundirlo con el *zero-shot learning*, donde no se ve ningún ejemplo de las clases objetivo y hay que apoyarse en descripciones semánticas auxiliares.)

El estado del arte previo era **HBPL** (Hierarchical Bayesian Program Learning, Lake et al.): un modelo generativo que descompone cada carácter en trazos y busca una explicación estructural de los píxeles. Potente —llega a 95.2%— pero carga mucho conocimiento de dominio (sabe que los caracteres se dibujan con trazos) y su inferencia es costosa. La pregunta de Koch et al. es: ¿puede un modelo **discriminativo y agnóstico al dominio**, que no sabe nada sobre trazos, acercarse a ese resultado solo aprendiendo features transferibles?

## La idea: dos torres gemelas que comparten pesos

La estrategia tiene dos pasos:

1. **Entrenar** una red para discriminar pares *same / different* (misma clase / clases distintas). Esta tarea se llama **verificación**.
2. **Generalizar** a clases nuevas reutilizando los features aprendidos, vía esa misma verificación.

La red toma **dos** entradas $x_1$ y $x_2$. Cada una pasa por una de dos torres convolucionales **gemelas**, que computan exactamente la misma función $f_\theta$ porque comparten los mismos pesos (*weight tying*). Esto fuerza dos propiedades deseables:

- **Consistencia local.** Si $x_1 \approx x_2$, entonces $f_\theta(x_1) \approx f_\theta(x_2)$: dos imágenes muy parecidas no pueden caer en lugares muy distintos del espacio de features.
- **Simetría.** Presentar $(x_1, x_2)$ produce la misma métrica que $(x_2, x_1)$, lo cual es correcto porque "ser de la misma clase" es una relación simétrica.

Cada torre mapea su entrada a un **embedding** $h_1 = f_\theta(x_1)$ y $h_2 = f_\theta(x_2)$ (en el modelo grande, un vector de 4096 unidades tras una capa fully-connected). En el tope, una **capa de distancia** compara los dos embeddings y una **sigmoide** produce la probabilidad de "misma clase":

$$
p = \sigma\!\left(\sum_j \alpha_j \,\bigl|\,h^{(j)}_{1} - h^{(j)}_{2}\,\bigr|\right)
$$

Desglose:

- $\bigl|h^{(j)}_1 - h^{(j)}_2\bigr|$ es la **distancia $L_1$ componente a componente**: el valor absoluto de la diferencia, dimensión a dimensión. Produce un vector (no un escalar) que mide cuánto difieren las gemelas en cada dimensión de feature.
- Los $\alpha_j$ son **parámetros aprendidos** que ponderan cada componente: algunas dimensiones son más discriminativas que otras para decidir "misma clase", y los $\alpha_j$ aprenden ese peso.
- La suma ponderada colapsa el vector a un escalar y la sigmoide $\sigma$ lo mapea a $[0,1]$.

La elección de $L_1$ (Manhattan) en vez de $L_2$ (euclidiana) es deliberada: $L_1$ componente a componente preserva información por dimensión *antes* de ponderar, mientras que $L_2$ ya habría colapsado todo a un escalar. La salida ya en $[0,1]$ hace que la **cross-entropy binaria** sea el objetivo natural:

$$
\mathcal{L} = y\,\log p + (1 - y)\,\log(1 - p) + \boldsymbol{\lambda}^T |\mathbf{w}|^2
$$

donde $y \in \{0,1\}$ es la etiqueta del par y $\boldsymbol{\lambda}^T|\mathbf{w}|^2$ es regularización $L_2$ con un coeficiente definido **por capa**.

Lo elegante: el conocimiento de dominio no se inyecta a mano (como los trazos de HBPL), sino que **emerge** de aprender qué features hacen que dos imágenes sean iguales o distintas. Esos features —curvas, intersecciones, terminaciones de trazo— se transfieren a alfabetos nunca vistos.

## De verificación a one-shot classification

Una vez entrenada la red para verificar, se la usa directamente en one-shot **sin reentrenar**. Sea $\mathbf{x}$ una imagen de test a clasificar en una de $C$ categorías, y sea $\{x_c\}_{c=1}^{C}$ el **support set**: un ejemplo por cada categoría posible. Se consulta la red con cada par $(\mathbf{x}, x_c)$, obteniendo $p(c)$, la probabilidad de que sean de la misma clase, y se elige:

$$
C^* = \arg\max_c \; p(c)
$$

Es decir: comparar la query contra cada ejemplo del support set y quedarse con la clase del par más similar. Esto convierte una red de verificación binaria en un clasificador multiclase **sin parámetros extra ni reentrenamiento**, definido enteramente por la métrica aprendida y el support set.

En la práctica esto es **un único forward pass batched**: se apilan $C$ copias de $\mathbf{x}$ en una matriz $X$ y los $x_c$ en otra matriz $X_C$, y se procesa el minibatch de tamaño $C$ de una sola vez. No hay $C$ pasadas secuenciales.

Visto así, el one-shot por argmax es esencialmente **un 1-NN sobre un espacio de embeddings aprendido**: vecino más cercano, pero según una métrica entrenada en vez de la distancia euclidiana sobre píxeles crudos.

## Detalles de entrenamiento

**Muestreo de pares.** El entrenamiento opera sobre **pares**, no sobre ejemplos individuales. Se construyeron tres conjuntos de 30k, 90k y 150k pares *same / different* muestreados al azar, usando el 60% de los datos para entrenar (30 de 50 alfabetos, 12 de 20 escritores). Se fijó un **número uniforme de ejemplos por alfabeto** para que los alfabetos grandes (hasta 40 caracteres) no dominaran sobre los pequeños (~15). Hay muchísimos más pares *different* posibles que *same*, así que balancear es necesario para que la señal de "misma clase" no se diluya —el mismo desbalance que años después motivó el *hard negative mining*.

**Data augmentation afín.** Se generaron **8 transformaciones afines** por ejemplo, parametrizadas como $T = (\theta, \rho_x, \rho_y, s_x, s_y, t_x, t_y)$ con rotación $\theta \in [-10°, 10°]$, cizalla $\rho \in [-0.3, 0.3]$, escala $s \in [0.8, 1.2]$ y traslación $t \in [-2, 2]$, cada componente incluido con probabilidad 0.5. Los datasets aumentados pasaron a 270k / 810k / 1.350.000 ejemplos efectivos.

**Optimización.** Backpropagation estándar; por el weight tying, el gradiente respecto de cada peso es la **suma** de las contribuciones de ambas torres. Minibatch de 128. Cada capa tiene su propio learning rate $\eta_j$, momentum $\mu_j$ y regularización $\lambda_j$. El learning rate decae 1% por época ($\eta^{(T)}_j = 0.99\,\eta^{(T-1)}_j$); el momentum arranca en 0.5 y crece linealmente hasta su valor por capa.

**Criterio de parada.** Hasta 200 épocas, con *early stopping* sobre el error de **one-shot validation** (320 tareas one-shot generadas al azar del conjunto de validación), no sobre el error de verificación. Es decir, el criterio de parada usa la tarea objetivo como proxy.

**Búsqueda de hiperparámetros.** Optimización bayesiana (Whetlab) sobre rangos *layer-wise*: $\eta_j \in [10^{-4}, 10^{-1}]$, $\mu_j \in [0, 1]$, $\lambda_j \in [0, 0.1]$, filtros de 3×3 a 20×20, 16 a 256 filtros por capa, y 128 a 4096 unidades fully-connected. El objetivo a maximizar fue la accuracy de one-shot validation.

## Resultados en Omniglot

**Omniglot** es el benchmark estándar de few-shot en caracteres manuscritos: 50 alfabetos (del latín y coreano a conjuntos ficticios como Klingon), imágenes binarias de 105×105, cada carácter dibujado **una sola vez** por cada uno de 20 escritores. Se lo llama el "MNIST transpuesto": muchísimas clases con poquísimos ejemplos cada una. Lake lo divide en un *background set* de 40 alfabetos (desarrollo) y un *evaluation set* de 10 alfabetos (nunca vistos en entrenamiento).

**Verificación.** Con 150k ejemplos y distorsiones afines ×8 se alcanza **93.42%** de accuracy. Dos lecciones: más datos ayudan pero con retornos decrecientes (91.54% → 91.63% al pasar de 90k a 150k sin distorsiones), y las distorsiones afines aportan ~1.3–1.8 puntos de forma consistente, más valor que duplicar los datos crudos.

**20-way one-shot (within-alphabet).** Se eligen 20 caracteres de un mismo alfabeto del evaluation set y dos escritores; cada carácter del primero se compara contra los 20 del segundo. Que los 20 distractores sean del mismo alfabeto (mismo estilo visual) lo hace difícil. Sobre 400 trials:

| Método | Test accuracy |
|---|---|
| Humans | 95.5 |
| Hierarchical Bayesian Program Learning (HBPL) | 95.2 |
| **Convolutional Siamese Net** | **92.0** |
| Affine model | 81.8 |
| Hierarchical Deep | 65.2 |
| Deep Boltzmann Machine | 62.0 |
| Siamese Net (no convolucional) | 58.3 |
| Simple Stroke | 35.2 |
| 1-Nearest Neighbor | 21.7 |

Con **92.0%**, la siamesa convolucional supera a todo excepto HBPL (95.2%), que a su vez queda apenas por debajo de los humanos (95.5%). Dos contrastes son reveladores: el salto sobre 1-NN crudo (21.7%) muestra que comparar píxeles en el espacio original es casi inútil mientras que comparar en el espacio de features aprendido es casi tan bueno como un humano; y la brecha entre la versión convolucional (92.0%) y la fully-connected (58.3%) confirma que la convolución —connectividad local, compartición de parámetros, invariancia a traslación— es esencial.

**Transferencia a MNIST.** Tratando los 10 dígitos como un "alfabeto" y evaluando 10-way one-shot **sin ningún fine-tuning**, la red entrenada en Omniglot logra **70.3%** frente a 26.5% de 1-NN. Cae respecto al 92% de Omniglot por el cambio de dominio, pero generaliza razonablemente a un dataset jamás visto: una demostración temprana de transfer learning de features de similitud entre dominios.

## Por qué importa hoy

La contribución no es una técnica matemática nueva —la siamesa existía desde Bromley y LeCun (1993) para verificación de firmas, y la cross-entropy es estándar— sino la **demostración empírica** de tres cosas:

1. Aprender una función de similitud sobre clases conocidas produce features que **transfieren a clases desconocidas** sin reentrenar.
2. Un enfoque **discriminativo y agnóstico al dominio** puede competir con métodos generativos cargados de priors (HBPL) y acercarse al rendimiento humano.
3. La **verificación es una tarea proxy poderosa** para el one-shot: optimizar "¿son iguales estas dos?" induce features que sirven para "¿cuál de estas $C$ es?".

Este paper estableció el **deep metric learning** como el enfoque dominante para one-shot y few-shot, y es el antecedente directo de una línea entera:

- **FaceNet (Schroff et al., 2015):** reemplaza los pares por **triplets** (anchor, positivo, negativo) con margen y semi-hard mining, resolviendo la calibración de la decisión binaria y llevando la verificación facial a producción.
- **Matching Networks (Vinyals et al., 2016):** entrenamiento **episódico** end-to-end (*train as you test*) con atención sobre el support set completo.
- **Prototypical Networks (Snell et al., 2017):** representa cada clase por el **centroide** de los embeddings de su support set y clasifica por distancia al prototipo más cercano.

De ahí en adelante, todo el metric learning moderno —contrastive loss, InfoNCE, SimCLR/MoCo, y los *bi-encoders / dual-encoders* para retrieval semántico y *dense retrieval*— desciende de la misma idea: **dos torres que comparten pesos, embeddings comparables por una métrica**.

Las limitaciones del paper marcan precisamente lo que vino después: entrena por pares en vez de episodios (corregido por Matching/Prototypical con el principio *train as you test*); compara la query contra cada candidato de forma **independiente**, sin razonar sobre el support set completo (resuelto con atención y prototipos); y deja la **métrica fija** tras el entrenamiento, sin adaptación al support set en test time (lo que MAML abordaría con meta-aprendizaje del optimizador).

## Conexión con la Clase 26 y con patient matching / record linkage

En la **Clase 26** (métodos no-paramétricos y meta-aprendizaje), la red siamesa es el puente perfecto entre el k-NN clásico y el deep learning: el one-shot por argmax de similitud *es* un 1-NN, pero sobre un espacio de embeddings aprendido discriminativamente. La distancia entre el 21.7% del 1-NN crudo y el 92.0% de la siamesa es, exactamente, el valor de aprender la métrica en vez de usar la euclidiana sobre píxeles.

Para **record linkage y patient matching** (MDM en salud) el mapeo es casi uno a uno. El problema de MDM es: dados dos registros de paciente —de sistemas distintos, con nombres tipeados diferente, fechas con errores, RUT con dígitos transpuestos— decidir si son **la misma persona** o personas distintas. Eso es exactamente la **tarea de verificación**, $p(\text{misma entidad} \mid x_1, x_2)$, salvo que $x_1, x_2$ son registros en vez de imágenes:

- **Las dos torres con pesos compartidos** = un **bi-encoder**: cada registro se mapea al mismo espacio de embeddings, y el weight tying garantiza que registros similares caigan cerca —la propiedad que permite usar el bi-encoder como **blocker** (recuperar candidatos por vecindad ANN antes de scorear).
- **La distancia $L_1$ ponderada + sigmoide** = la cabeza de scoring "match / no-match". Un scorer GBM (p. ej. XGBoost) sobre features de comparación de campos es simplemente una versión no lineal del mismo principio $\sum_j \alpha_j |h_1^{(j)} - h_2^{(j)}|$: una combinación ponderada de distancias componente a componente donde los pesos aprenden qué campos son más discriminativos para la identidad.
- **El muestreo de pares same/different** = la generación de pares match/no-match, con el mismo desbalance (muchísimos más no-match) que se ataca con blocking y muestreo de hard negatives (dos hermanos, padre e hijo con mismo apellido).
- **El one-shot por argmax** = la decisión de linkage: comparar un registro entrante contra los candidatos del golden record y elegir (o crear) la entidad de máxima probabilidad. El truco de batching ($X$, $X_C$ en un solo forward pass) es directamente aplicable a scorear un registro contra todos los candidatos de su bloque.
- **La caída cross-domain (92% → 70.3% al pasar a MNIST sin fine-tuning)** cuantifica una promesa y una advertencia: features de similitud transfieren entre dominios, pero con degradación. Un bi-encoder entrenado en una población no transfiere perfecto a otra con distintas convenciones de nombres —respaldo empírico para el argumento de los **retornos decrecientes del ML en MDM**.

La limitación de la **verificación par a par sin contexto del cluster** es también la limitación de diseño en MDM: un scorer par a par no resuelve transitividad (si A=B y B=C, entonces A=C), lo que en record linkage se aborda con clustering / resolución de entidades sobre el grafo de matches —análogo a cómo Matching y Prototypical Networks agregaron contexto del support set.

## Notas y enlaces

- Paper sin identificador arXiv (publicado en el ICML 2015 Deep Learning Workshop). PDF local: `/papers/siamese-networks-koch-2015.pdf`.
- Fundamentos: [meta-aprendizaje](/fundamentos/meta-aprendizaje), [metric learning](/fundamentos/metric-learning), [triplet loss](/fundamentos/triplet-loss), [few-shot learning](/fundamentos/few-shot-learning).
- Papers relacionados: [FaceNet (Schroff 2015)](/papers/facenet-schroff-2015), [Matching Networks (Vinyals 2016)](/papers/matching-networks-vinyals-2016), [Prototypical Networks (Snell 2017)](/papers/prototypical-networks-snell-2017).
- Ver [Clase 26 -- Meta-aprendizaje](/clases/clase-26).
