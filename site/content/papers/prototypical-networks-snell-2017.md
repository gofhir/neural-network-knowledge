---
title: "Prototypical Networks (Few-shot)"
weight: 264
math: true
---

{{< paper-card
    title="Prototypical Networks for Few-shot Learning"
    authors="Jake Snell, Kevin Swersky, Richard Zemel"
    year="2017"
    venue="NeurIPS 2017"
    pdf="/papers/prototypical-networks-snell-2017.pdf"
    arxiv="1703.05175" >}}
Propone **prototypical networks** (ProtoNets), un modelo de clasificacion *few-shot* deliberadamente simple: aprender un espacio de *embedding* donde cada clase se resume en un **prototipo** —el centroide de sus ejemplos de soporte— y clasificar por proximidad. La clasificacion es un *softmax* sobre **distancias euclidianas negativas** a los prototipos. La eleccion de distancia no es un truco: con divergencias de Bregman, ProtoNets equivale a una **estimacion de densidad por mezcla** de la familia exponencial. Iguala o supera a *matching networks* y al *meta-learner LSTM* con menos parametros: **49.42% / 68.20%** en miniImageNet 5-way 1/5-shot, **96.0%** en Omniglot 20-way 1-shot y **54.6%** en CUB zero-shot 50-way.
{{< /paper-card >}}

---

## El problema

La clasificacion *few-shot* exige que un modelo generalice a clases que nunca vio en entrenamiento, disponiendo apenas de uno o unos pocos ejemplos etiquetados por clase nueva. El enfoque ingenuo —reentrenar la red sobre esas pocas muestras— sobreajusta de forma catastrofica: con 1 o 5 ejemplos por clase, los grados de libertad de una red profunda superan ampliamente la informacion disponible.

Hacia 2017 dos trabajos habian empujado el estado del arte atacando este sobreajuste con maquinaria sofisticada. **Matching Networks** (Vinyals et al. 2016) usaba atencion sobre todo el conjunto de soporte y *full context embeddings* (FCE) via un LSTM bidireccional. El **Meta-Learner LSTM** (Ravi & Larochelle 2017) entrenaba un LSTM para *producir las actualizaciones* de un clasificador a medida por episodio.

El diagnostico de Snell et al. es que el problema fundamental —escasez de datos— pide lo contrario a esa complejidad: **un sesgo inductivo fuerte y simple**. La apuesta del paper es que, con la decision de diseno correcta (la media como prototipo y la distancia correcta), se iguala o supera a esos modelos sin su maquinaria. Es una navaja de Occam aplicada al *few-shot*: toda la capacidad de aprendizaje se concentra en la fase de *meta-entrenamiento* (donde si hay muchos datos, en las clases base), y la adaptacion a clases nuevas queda sin parametros que ajustar. El *few-shot* deja de ser un problema de optimizacion con datos escasos y pasa a ser uno de **recuperacion geometrica** en un espacio bien construido.

## La idea central: prototipos como centroides

Se da un conjunto de soporte $S = \{(x_1, y_1), \dots, (x_N, y_N)\}$ con $x_i \in \mathbb{R}^D$ y etiquetas $y_i \in \{1, \dots, K\}$; $S_k$ es el subconjunto de ejemplos de la clase $k$. ProtoNets computa, mediante una funcion de *embedding* $f_\phi : \mathbb{R}^D \to \mathbb{R}^M$ con parametros aprendibles $\phi$ (una CNN), un **prototipo** $c_k \in \mathbb{R}^M$ por clase: la media de los soportes embebidos de esa clase.

$$
c_k = \frac{1}{|S_k|} \sum_{(x_i, y_i) \in S_k} f_\phi(x_i).
$$

Esta es la ecuacion que define el metodo. Cada clase queda resumida en un solo vector, su centroide en el espacio aprendido. La consecuencia practica es enorme: a diferencia de matching networks (que debe conservar y comparar contra *todos* los puntos de soporte), ProtoNets solo guarda $K$ prototipos. La representacion de una clase es **concisa e independiente del numero de ejemplos** de soporte, asi que el costo de inferencia no crece con el tamano del soporte.

Dada una distancia $d : \mathbb{R}^M \times \mathbb{R}^M \to [0, +\infty)$, un punto de consulta $x$ se clasifica con un *softmax* sobre las **distancias negativas** a los prototipos:

$$
p_\phi(y = k \mid x) = \frac{\exp\!\big(-d(f_\phi(x), c_k)\big)}{\sum_{k'} \exp\!\big(-d(f_\phi(x), c_{k'})\big)}.
$$

Cuanto mas cerca esta $f_\phi(x)$ del prototipo $c_k$, mayor es la probabilidad de la clase $k$. El entrenamiento minimiza la log-verosimilitud negativa de la clase verdadera, $J(\phi) = -\log p_\phi(y = k \mid x)$, via SGD. Como la *misma* red $f_\phi$ embebe soportes y consultas, el gradiente empuja simultaneamente la consulta hacia su centroide y al centroide hacia la consulta; no hay capa de clasificacion con pesos por clase, sino una "capa" que se reconstruye en cada episodio a partir de los datos.

Un detalle estadistico clave: el promedio reduce ruido. Con $|S_k|$ ejemplos, el centroide estima la media de la clase con varianza que decae como $1/|S_k|$. Por eso el salto de 1-shot a 5-shot mejora consistentemente la precision: el prototipo de 5-shot es una estimacion mucho mas estable del "centro" de la clase que el de 1-shot (un solo punto ruidoso).

## Por que la distancia euclidiana (Bregman) supera al coseno

Esta es la contribucion teorica mas original del paper, y la que distingue a ProtoNets de un simple "centroide + vecino mas cercano".

Una **divergencia de Bregman** $d_\varphi$ se define a partir de una funcion estrictamente convexa $\varphi$ como

$$
d_\varphi(z, z') = \varphi(z) - \varphi(z') - (z - z')^\top \nabla\varphi(z'),
$$

es decir, el "gap" entre $\varphi(z)$ y su aproximacion lineal de primer orden alrededor de $z'$. La **distancia euclidiana cuadrada** $\|z - z'\|^2$ (con $\varphi(z) = \|z\|^2$) y la de Mahalanobis son casos particulares.

Banerjee et al. (2005) probaron un resultado decisivo: para *cualquier* divergencia de Bregman, el representante de un cluster que minimiza la distancia total a sus puntos **es la media del cluster**,

$$
\arg\min_{c} \sum_{z} d_\varphi(z, c) = \frac{1}{|\text{cluster}|}\sum_{z} z.
$$

Por tanto, calcular el prototipo como media **no es arbitrario**: produce el representante optimo del soporte, *siempre que la distancia sea de Bregman*. Mas aun, toda densidad de la familia exponencial regular se escribe en terminos de una divergencia de Bregman univocamente determinada. Para una **mezcla equiponderada con un cluster por clase**, la inferencia de la asignacion de cluster

$$
p(y = k \mid z) = \frac{\exp\big(-d_\varphi(z, \mu(\theta_k))\big)}{\sum_{k'} \exp\big(-d_\varphi(z, \mu(\theta_{k'}))\big)}
$$

resulta **identica** a la prediccion de ProtoNets, con $f_\phi(x) = z$ y $c_k = \mu(\theta_k)$. La conclusion es elegante: **ProtoNets realiza implicitamente estimacion de densidad por mezcla de la familia exponencial** en el espacio de *embedding*, y la divergencia elegida especifica las suposiciones sobre la distribucion condicional de clase. La euclidiana cuadrada corresponde a **gaussianas esfericas**.

El coseno, en cambio, **no es una divergencia de Bregman**, asi que rompe esa interpretacion. El mecanismo del fracaso es concreto: $\cos(u, v) = \frac{u^\top v}{\|u\|\,\|v\|}$ solo depende de la *direccion*, proyectando todo sobre la hiperesfera unitaria. Pero la media aritmetica **no vive en esa hiperesfera** (el promedio de vectores unitarios tiene norma menor que 1). Se construye el prototipo segun un criterio (media euclidiana) y se mide segun otro (angulo): una incoherencia interna. Con euclidiana cuadrada esa incoherencia desaparece porque la media *es* el minimizador, y la geometria del clustering, de la distancia y de la gaussiana esferica encajan. Empiricamente la brecha es de **~17 puntos** en miniImageNet 5-shot.

Como nota tecnica, bajo euclidiana cuadrada el modelo es en realidad **lineal** en $f_\phi(x)$: expandiendo $-\|f_\phi(x) - c_k\|^2$, el termino $-f_\phi(x)^\top f_\phi(x)$ es constante en $k$ y se cancela en el *softmax*, dejando pesos $w_k = 2c_k$ y sesgo $b_k = -\|c_k\|^2$. Esto no es una limitacion: toda la no linealidad necesaria vive dentro de $f_\phi$, igual que en un clasificador estandar (backbone no lineal + capa lineal *softmax*).

## Relacion con Matching Networks

ProtoNets y matching networks **coinciden exactamente en el caso 1-shot**: con un solo soporte por clase, $c_k = f_\phi(x_k)$, el prototipo *es* ese unico ejemplo y el *softmax* sobre distancias colapsa al vecino mas cercano ponderado de matching networks. Son el mismo modelo (por eso las tablas los fusionan en 1-shot).

La divergencia aparece para $K > 1$ shots. Matching networks aplica **atencion sobre todos los puntos de soporte individuales**; ProtoNets primero **promedia** los soportes de cada clase en un prototipo y luego compara contra los $K$ prototipos. Es mas simple y mas barato en inferencia.

Lo que ProtoNets **descarta deliberadamente** es la maquinaria de matching networks: la FCE (un segundo nivel de *embedding* que toma en cuenta el episodio via un LSTM bidireccional) y el desacople de los encoders de soporte y consulta. Snell et al. observan que la FCE (i) aumenta los parametros y (ii) **impone un orden arbitrario** sobre el conjunto de soporte (un conjunto no tiene orden, pero un LSTM lo procesa secuencialmente). ProtoNets usa **el mismo encoder** para soporte y consulta, y un solo prototipo por clase. El argumento: con tan pocos datos, un sesgo inductivo simple basta; no hace falta un *embedding* a medida por episodio.

## El protocolo episodico

El entrenamiento episodico (heredado de matching networks) replica la tarea de test: en cada episodio se muestrean $N_C$ clases (el "way"), $N_S$ soportes por clase (el "shot") y $N_Q$ consultas, se forman los prototipos y se acumula la perdida

$$
J \leftarrow J + \frac{1}{N_C N_Q}\Big[\, d(f_\phi(x), c_k) + \log \sum_{k'} \exp\big(-d(f_\phi(x), c_{k'})\big)\Big],
$$

que es exactamente $-\log p_\phi(y=k\mid x)$. La aleatoriedad de la composicion (clases distintas cada vez) fuerza al espacio a servir para particiones nunca vistas juntas, lo que produce la generalizacion.

Aqui aparece un hallazgo contraintuitivo y muy citado: **conviene entrenar con MAS "way" que el de test**. Si en test se hara 5-way, conviene entrenar con 20-way o 30-way. La mayor dificultad fuerza decisiones mas finas en el espacio de embedding: el *log-sum-exp* penaliza la cercania a *cualquier* prototipo ajeno, asi que mas clases por episodio implica mas restricciones de margen por paso de gradiente —el mismo principio que el muestreo de negativos dificiles en aprendizaje contrastivo. En miniImageNet 5-way 1-shot, entrenar con way=5 da **46.14%** y con way=30 da **49.42%**: ~3.3 puntos gratis, solo cambiando la composicion de episodios.

Para el "shot", en cambio, lo mejor es **igualar shot de entrenamiento y de test**. La asimetria es interesante: subir el "way" ayuda, pero el "shot" debe coincidir. Y hay un punto de reversion: para 5-shot, el "way" optimo ronda 20 y subir a 30 lo degrada, porque la tarea se vuelve tan dificil que la estimacion del prototipo ya no alcanza.

## Resultados

**Omniglot** (1623 caracteres manuscritos, 20 ejemplos cada uno; encoder conv-4: cuatro bloques de conv $3\times3$ con 64 filtros + BatchNorm + ReLU + max-pool). Euclidiana, sin fine-tune:

| Configuracion | ProtoNets | Matching Nets (cosine) | Neural Statistician |
|---|---|---|---|
| 5-way 1-shot | **98.8%** | 98.1% | 98.1% |
| 5-way 5-shot | **99.7%** | 98.9% | 99.5% |
| 20-way 1-shot | **96.0%** | 93.8% | 93.2% |
| 20-way 5-shot | **98.9%** | 98.5% | 98.1% |

Estado del arte en todas las configuraciones; la ventaja es mayor en 20-way (la tarea mas dificil).

**miniImageNet** (subconjunto de ILSVRC-12, 100 clases x 600 imagenes a color de $84\times84$; splits de Ravi & Larochelle). 5-way, sin fine-tune:

| Modelo | Dist. | 1-shot | 5-shot |
|---|---|---|---|
| Baseline Nearest Neighbors | Cosine | 28.86 ± 0.54% | 49.79 ± 0.79% |
| Matching Networks | Cosine | 43.40 ± 0.78% | 51.09 ± 0.71% |
| Matching Networks FCE | Cosine | 43.56 ± 0.84% | 55.31 ± 0.73% |
| Meta-Learner LSTM | — | 43.44 ± 0.77% | 60.60 ± 0.71% |
| **Prototypical Networks** | **Euclid.** | **49.42 ± 0.78%** | **68.20 ± 0.66%** |

ProtoNets gana por amplio margen: ~6 puntos sobre el mejor baseline en 1-shot y ~7.6 puntos sobre el Meta-Learner LSTM en 5-shot. Que un modelo mas simple supere al meta-aprendiz LSTM es el golpe retorico del paper. La ablacion confirma la teoria: en 5-shot way=20, el coseno da **51.48%** frente a **68.20%** de la euclidiana (brecha de ~17 puntos).

**CUB-200 (zero-shot).** La variante zero-shot reemplaza el promedio de soportes por un *embedding* de los **meta-datos de clase**: $c_k = g_\vartheta(v_k)$, donde $v_k$ es un vector de atributos (312 dimensiones) embebido por una segunda funcion a un espacio compartido con las imagenes (features GoogLeNet de 1024-d), con los prototipos normalizados a norma unitaria. En 50-way 0-shot ProtoNets logra **54.6%**, estado del arte por +3.7 puntos sobre DA-SJE entre metodos basados en atributos. Demuestra que el metodo es general aun cuando puntos (imagenes) y clases (atributos) viven en dominios distintos.

## Por que importa hoy

El impacto de ProtoNets no se explica solo por las cifras sino por su relacion coste/beneficio:

1. **Simplicidad como caracteristica, no como compromiso.** Una sola ecuacion (media + *softmax* sobre distancias) iguala o supera arquitecturas con atencion, LSTM bidireccionales y meta-optimizadores. En el regimen de datos escasos, el sesgo inductivo fuerte gana.
2. **Eficiencia en inferencia.** $K$ prototipos en vez de todo el soporte; costo de prediccion independiente del tamano del soporte.
3. **Principios, no truco.** La conexion con divergencias de Bregman y estimacion de densidad por mezcla justifica *por que* la media y *por que* la euclidiana.
4. **Baseline de facto.** ProtoNets fijo el encoder conv-4 y el protocolo de evaluacion (5-way 1/5-shot, 600/1000 episodios, IC 95%) que hoy son estandar. Casi todo paper posterior de *few-shot* (Relation Networks, TADAM, metodos transductivos, Infinite Mixture Prototypes) se compara contra el.

Las limitaciones tambien marcaron la agenda posterior: el supuesto **unimodal** (un prototipo por clase asume un solo *blob* gaussiano, lo que falla en clases multimodales), el **embedding fijo** no adaptado por tarea (atacado por TADAM y la modulacion condicionada), la suposicion de **mezcla equiponderada** (clases balanceadas) y el **shift de dominio** entre entrenamiento y test.

## Conexion con la Clase 26

ProtoNets es un ejemplar limpio de metodo **no parametrico** en el sentido relevante para el *few-shot*: el clasificador para una tarea nueva no tiene parametros propios entrenados sobre esa tarea; se construye sobre la marcha a partir de los datos (los prototipos *son* funcion directa del soporte). Esto lo emparenta con k-NN y los kernel methods: la "memoria" del modelo son los ejemplos resumidos en centroides, no un vector de pesos por clase.

Encaja en la familia de **metric-based meta-learning** —aprender una metrica/espacio donde la regla de decision no parametrica (vecino al prototipo) funcione— y es el contrapunto perfecto a los metodos **optimization-based** (Meta-Learner LSTM y, mas tarde, MAML), que si adaptan parametros por tarea. La dicotomia metrico vs optimizacion es uno de los ejes organizadores del *few-shot* que la clase explora. En salud y oncologia el atractivo es directo: aprender un espacio sobre patologias frecuentes y clasificar una **patologia rara** con 1-5 ejemplos calculando su centroide, sin reentrenar; con la cautela de que las clases medicas suelen ser multimodales y el shift de dominio entre centros es la regla, no la excepcion.

## Notas y enlaces

**Fundamentos:** [meta-aprendizaje](/fundamentos/meta-aprendizaje), [metric learning](/fundamentos/metric-learning), [few-shot learning](/fundamentos/few-shot-learning), [aprendizaje contrastivo](/fundamentos/aprendizaje-contrastivo).

**Papers relacionados:** [Matching Networks](/papers/matching-networks-vinyals-2016), [Siamese Networks](/papers/siamese-networks-koch-2015), [MAML](/papers/maml-finn-2017).

**Clase:** Ver [Clase 26 -- Meta-aprendizaje](/clases/clase-26).
