---
title: "Invariant and Spreading Instance Feature (2019)"
weight: 319
math: true
---

{{< paper-card
    title="Unsupervised Embedding Learning via Invariant and Spreading Instance Feature"
    authors="Mang Ye, Xu Zhang, Pong C. Yuen, Shih-Fu Chang"
    year="2019"
    venue="CVPR 2019"
    pdf="/papers/invariant-spreading-ye-2019.pdf"
    arxiv="1904.03436" >}}
Paper de CVPR 2019 (Hong Kong Baptist University + Columbia) que **destila y nombra** la intuición que un año más tarde se convertiría en el aprendizaje contrastivo canónico. Aprende *embeddings* sin etiquetas exigiendo dos propiedades duales: que el rasgo sea **invariante** a la aumentación de la misma imagen (positivo) y que se **disperse** respecto de las demás instancias del batch (negativos, **sin memory bank**). Lo hace con un softmax sobre las similitudes coseno de los rasgos de instancia reales, que ejerce *hard negative mining* implícito y converge muy rápido. Es el **puente conceptual hacia SimCLR y MoCo**, y la diapositiva de la [Clase 28](/clases/clase-28) que introduce el aprendizaje contrastivo lo cita literalmente.
{{< /paper-card >}}

---

## Contexto

El **aprendizaje de *embeddings* no supervisado** busca una función $f_\theta(\cdot)$ que mapee imágenes a un espacio de baja dimensión donde la cercanía de los vectores refleje similitud visual, **sin etiquetas humanas**. La distinción que el paper hace desde la primera página es fina pero crucial: el *feature learning* general (autoencoders, GANs, tareas de pretexto) aprende una representación intermedia que luego se afina con datos etiquetados, pero esa representación "puede no preservar la similitud visual" y se desploma en tareas directas como la búsqueda por vecino más cercano (kNN). El aprendizaje de *embeddings* exige esa propiedad de similitud **directamente** en el espacio aprendido.

La tesis central nace de una analogía con el caso supervisado: un buen *embedding* satisface **(1) concentración positiva** —rasgos de la misma categoría cercanos entre sí— y **(2) separación negativa** —rasgos de categorías distintas tan separados como sea posible—. Sin etiquetas, el paper *aproxima* ambas con **supervisión a nivel de instancia**: cada imagen es su propia clase. El positivo se construye con aumentación de datos (la misma instancia, transformada, debe dar un rasgo invariante); los negativos, tratando las **otras instancias del mini-batch** como negativos aproximados, lo que fuerza una propiedad de dispersión (*spread-out*).

### De la discriminación de instancias al contrastive learning

- **Exemplar CNN** (Dosovitskiy et al., 2014): la idea fundacional de tratar cada imagen como una clase. Pero la matriz de pesos del clasificador $W \in \mathbb{R}^{n \times d}$ crece linealmente con el número de imágenes (millones de columnas) y limita la eficiencia.
- **Instance Discrimination / NCE** (Wu et al., CVPR 2018): elimina esos pesos y monta un ***memory bank*** que almacena el rasgo $v_i$ de cada instancia. El problema que Ye et al. identifican: $v_i$ **solo se actualiza una vez por época**, mientras la red cambia en cada iteración; comparar el rasgo en tiempo real con un $v_i$ desactualizado "entorpece el entrenamiento".

Ye et al. resuelven esto optimizando **directamente sobre el rasgo de instancia real**, sin pesos de clasificador ni memory bank. Es exactamente la receta que SimCLR (Chen et al., 2020) escalaría con batches enormes, cabeza de proyección no lineal y la pérdida NT-Xent, y que MoCo (He et al., 2020) refinaría con una cola y un *momentum encoder*.

## La dualidad invariante + spreading

Esta es la aportación conceptual para el curso. Con todos los rasgos $\ell_2$-normalizados ($\|f_i\|_2 = 1$):

- **Invariante (positivo).** Las dos vistas aumentadas $x_i$ y $\hat{x}_i = T(x_i)$ de la misma instancia deben producir rasgos cercanos. Maximizar $\exp(f_i^T \hat{f}_i / \tau)$ equivale a aumentar la similitud coseno entre $f_i$ y $\hat{f}_i$: un rasgo **invariante a la aumentación**. Aproxima la *concentración positiva* supervisada.
- **Spreading (negativo).** Como los datos no etiquetados están muy desbalanceados, un pequeño lote de instancias al azar puede tratarse aproximadamente como negativos de cada instancia. Minimizar $\exp(f_k^T \hat{f}_i / \tau)$ para $k \neq i$ separa $\hat{f}_i$ de las demás; sumando sobre todo el batch, las instancias se ven forzadas a separarse entre sí. Aproxima la *separación negativa* supervisada.

El paper es honesto: tratar todas las demás instancias del batch como negativos introduce **falsos negativos** ocasionales (instancias de la misma clase semántica), pero la evidencia muestra que el beneficio domina. Es el mismo *trade-off* que SimCLR aceptaría después.

## Método: softmax sobre el rasgo de instancia real

En vez de plantear el aprendizaje como una **clasificación multiclase** ($n$ clases, una por instancia — el enfoque de Exemplar/NCE), el paper lo resuelve como **clasificación binaria vía máxima verosimilitud**. Para la instancia $x_i$, su versión aumentada $\hat{x}_i$ *debe* clasificarse en $i$ y las otras instancias *no*.

La probabilidad de que $\hat{x}_i$ se reconozca como la instancia $i$ es:

$$P(i \mid \hat{x}_i) = \frac{\exp(f_i^T \hat{f}_i / \tau)}{\sum_{k=1}^{m} \exp(f_k^T \hat{f}_i / \tau)}$$

y la de que $x_j$ ($j \neq i$) se reconozca como $i$:

$$P(i \mid x_j) = \frac{\exp(f_i^T f_j / \tau)}{\sum_{k=1}^{m} \exp(f_k^T f_j / \tau)}, \quad j \neq i$$

Asumiendo independencia, la pérdida es la log-verosimilitud negativa sobre el batch:

$$J = -\sum_i \log P(i \mid \hat{x}_i) - \sum_i \sum_{j \neq i} \log\big(1 - P(i \mid x_j)\big)$$

El primer término empuja la **invariancia** (alinea cada instancia con su aumentación); el segundo empuja la **dispersión** (separa las instancias entre sí). La temperatura $\tau$ (0.1 en los experimentos) controla la concentración de la distribución.

**Hard negative mining implícito.** La función softmax aprovecha las relaciones entre todas las instancias muestreadas y **pondera más los negativos difíciles** (los más parecidos al query), sin necesidad de una estrategia explícita de muestreo de tripletas. El paper atribuye a esta naturaleza de *hard mining* la superioridad del método frente a la pérdida *triplet*.

**Red siamesa sin memory bank.** Se implementa con una red siamesa de dos ramas que comparten pesos: $m$ instancias entran por la primera rama y sus versiones aumentadas por la segunda; ambas pasan por el backbone CNN, una capa FC y normalización $\ell_2$. Para cada muestra hay **un positivo aumentado** y **$2N-2$ negativos**, todos del propio mini-batch —**sin memory bank**—. Esta es la diferencia operativa clave con NCE: los negativos son rasgos *frescos*, calculados en la misma pasada, no memorizados y desactualizados.

## Resultados

**Categorías vistas (CIFAR-10, STL-10).** Con ResNet18, *embedding* de 128-d, $\tau=0.1$ y un clasificador kNN ponderado: en **CIFAR-10** alcanza **83.6 %** de precisión kNN, superando a Exemplar (74.5 %), NPSoftmax (80.8 %) y NCE (80.4 %). En STL-10 escala bien con más datos no etiquetados (81.6 % kNN con 105K imágenes).

**Convergencia rápida.** Uno de los resultados más vistosos: el método alcanza 60 % de precisión kNN en CIFAR-10 en **solo 2 épocas**, mientras NCE necesita 25 y Exemplar 45. La velocidad se debe a optimizar directamente sobre el rasgo real, en vez de sobre pesos de clasificador o un memory bank desactualizado.

**Categorías no vistas (CUB200, Stanford Online Product, Car196).** Con backbone Inception-V1 y Recall@K, es el claro ganador entre los no supervisados (Car196 R@1 = 41.3 % vs. 37.5 % de NCE) y resulta competitivo incluso con algunos métodos supervisados en CUB200. Entrenando **desde cero** sobre Stanford Online Product sigue ganando (R@1 = 39.7 %), donde MOM —que mina etiquetas— fracasa por depender de una buena inicialización.

**Ablaciones.** (1) *Invariancia*: quitar la aumentación derrumba la precisión de 83.6 % a **37.4 %**; entre las operaciones, **RandomResizedCrop es la que más aporta** (cae a 56.2 % sin ella), un orden que SimCLR confirmaría después. (2) *Dispersión*: usar solo el 50 % de negativos **difíciles** mantiene la precisión casi intacta (83.2 %), mientras usar solo los **fáciles** la desploma a 57.5 % —corroborando el *hard mining* implícito del softmax—.

## Limitaciones

- **El supuesto de negativos es ruidoso:** los falsos negativos del batch persisten y motivarían líneas posteriores (BYOL/SimSiam, que prescinden de negativos explícitos).
- **Dependencia del tamaño de batch:** la cantidad y diversidad de negativos está acotada por el batch. El paper usa batches modestos (128); SimCLR mostraría que escalar a miles ayuda, y MoCo desacoplaría el número de negativos del batch con una cola.
- **Sensibilidad a las aumentaciones:** sin ellas el método colapsa, y no hay una búsqueda exhaustiva del espacio de transformaciones.
- **Escala:** experimentos en datasets pequeños/medianos y backbones modestos; sin validación en ImageNet a gran escala con transfer a detección/segmentación —el banco de pruebas que consagraría a MoCo y SimCLR—.

## Por qué importa para la Clase 28

El valor histórico de este paper no está en sus tablas (modestas en retrospectiva por usar datasets pequeños) sino en haber **destilado y nombrado** la intuición del contrastive learning canónico. Tres ideas que aquí ya están completas:

1. **Positivo por aumentación de la misma instancia, sin etiquetas.** La invariancia como señal de supervisión propia.
2. **Negativos del mismo mini-batch, sin memory bank.** Optimizar sobre rasgos frescos y reales —el esquema que [SimCLR](/papers/simclr-chen-2020) adoptaría y que [MoCo](/papers/moco-he-2019) refinaría con su cola—.
3. **Softmax sobre similitudes coseno con temperatura $\tau$.** La forma funcional —invariante en el numerador, spreading en el denominador— es la columna vertebral de NT-Xent/InfoNCE.

La frase que la [Clase 28](/clases/clase-28) cita —que la representación de una imagen debe estar más cerca de sí misma transformada que de otra imagen distinta— es precisamente la dualidad **invariante + spreading** que este paper formaliza. Por eso ocupa el lugar de **antecedente directo** en la narrativa del curso: es el eslabón entre la discriminación de instancias con memory bank (Wu et al., 2018) y la explosión del contrastive learning de 2020. Quien entiende este paper entiende *por qué* SimCLR funciona, antes de ver el truco de escala que lo hizo famoso.

Conviene leerlo junto con el fundamento de [aprendizaje contrastivo](/fundamentos/aprendizaje-contrastivo), que sistematiza la familia de pérdidas (InfoNCE, NT-Xent, triplet) y el rol de positivos/negativos, y con el de [aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado), que ubica el contrastive learning dentro del panorama de tareas de pretexto. Para un trasfondo en *record linkage* y *patient matching*, hay una lectura transversal directa: la dualidad invariante+spreading es la misma geometría que se busca en un *embedding* de pares de registros —acercar las variantes de una misma entidad (mismo paciente con datos ruidosos) y separar entidades distintas—, con el mismo *hard mining* implícito y la misma advertencia sobre falsos negativos.

## Notas y enlaces

- arXiv: https://arxiv.org/abs/1904.03436
- Código (PyTorch): https://github.com/mangye16/Unsupervised_Embedding_Learning
- Venue: CVPR 2019 (IEEE/CVF Conference on Computer Vision and Pattern Recognition).
