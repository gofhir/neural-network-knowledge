---
title: "Profundizacion - Aprendizaje Autosupervisado"
weight: 20
math: true
---

> Esta pagina complementa la [teoria de la clase 28](/clases/clase-28/teoria) con derivaciones formales del aprendizaje autosupervisado (SSL). Seis bloques: **Parte I** analiza el autoencoder y por que la perdida $L_2$ produce reconstrucciones borrosas, con la conexion lineal autoencoder = PCA. **Parte II** deriva la perdida contrastiva InfoNCE / NT-Xent y su cota inferior sobre la informacion mutua. **Parte III** contrasta formalmente SimCLR y MoCo (batch de negativos vs cola FIFO + momentum encoder). **Parte IV** formaliza el masked modeling de [MAE](/papers/mae-he-2022). **Parte V** desarrolla el consistency training de [UDA](/papers/uda-xie-2019). **Parte VI** trata la rotacion como clasificacion ([RotNet](/papers/rotnet-gidaris-2018)) y la discriminacion de instancias.

---

## Parte I — Autoencoders, reconstruccion y el sesgo del promedio

### I.1 El autoencoder y el cuello de botella

Un **autoencoder** es un par de funciones $f_\theta$ (encoder) y $g_\phi$ (decoder) entrenadas para reconstruir su propia entrada. Dado un ejemplo $x \in \mathbb{R}^D$, se computa un codigo latente $z = f_\theta(x) \in \mathbb{R}^d$ y una reconstruccion $\hat{x} = g_\phi(z)$. El objetivo minimiza el riesgo de reconstruccion:

$$
\min_{\theta,\phi}\; \mathbb{E}_{x \sim p(x)}\big[\, \ell\big(x,\, g_\phi(f_\theta(x))\big) \,\big]
$$

La pieza esencial es el **cuello de botella** (*bottleneck*): cuando $d \ll D$, la red no puede copiar la entrada y se ve forzada a comprimirla, descubriendo estructura de bajo rango. Sin restriccion ($d \ge D$) la identidad $g_\phi \circ f_\theta = \mathrm{id}$ es solucion trivial y nada se aprende; de ahi que el SSL generativo introduzca corrupciones (denoising, masking, inpainting) que impiden el atajo aun con codigos grandes. Como nota [Pathak et al.](/papers/context-encoders-pathak-2016), un autoencoder ordinario "probablemente solo comprime el contenido de la imagen sin aprender una representacion semanticamente significativa": el bottleneck obliga a comprimir, pero no garantiza *semantica*.

### I.2 Por que $L_2$ predice el promedio condicional y produce blur

La perdida cuadratica (MSE) es la opcion por defecto:

$$
\ell_{L_2}(x,\hat{x}) = \lVert x - \hat{x} \rVert_2^2 = \sum_{i=1}^{D}(x_i - \hat{x}_i)^2
$$

El punto clave —central a la motivacion de [Context Encoders](/papers/context-encoders-pathak-2016)— es que la reconstruccion optima bajo $L_2$ es el **promedio condicional**. Consideremos un solo pixel (o region) faltante $y$ que debemos predecir a partir del contexto $c$. El predictor $\mu(c)$ que minimiza el error cuadratico esperado resuelve:

$$
\mu^\star(c) = \arg\min_{\mu}\; \mathbb{E}_{y \sim p(y\mid c)}\big[(y - \mu)^2\big]
$$

Expandiendo y derivando respecto de $\mu$:

$$
\frac{\partial}{\partial \mu}\, \mathbb{E}\big[(y-\mu)^2\big]
= \frac{\partial}{\partial \mu}\Big(\mathbb{E}[y^2] - 2\mu\,\mathbb{E}[y] + \mu^2\Big)
= -2\,\mathbb{E}[y] + 2\mu = 0
$$

$$
\boxed{\;\mu^\star(c) = \mathbb{E}[\,y \mid c\,]\;}
$$

La solucion es la **media condicional**. Cuando la tarea es **multimodal** —hay varias formas igualmente plausibles de rellenar un hueco, como notan Pathak et al.— la media de varios modos nitidos distintos es una imagen *promediada* y por tanto **borrosa**. El $L_2$ "prefiere una solucion borrosa sobre texturas precisas" porque eso minimiza el error medio por pixel, aunque ninguna imagen real luzca asi. Este es el argumento estadistico que justifica el termino adversarial GAN que Context Encoders suma al $L_2$: el discriminador escoge *un* modo nitido en vez del promedio.

### I.3 Comparacion con $L_1$

La perdida absoluta tiene un sesgo distinto:

$$
\ell_{L_1}(x,\hat{x}) = \lVert x - \hat{x} \rVert_1 = \sum_{i=1}^{D}|x_i - \hat{x}_i|
$$

Repitiendo el argumento, el minimizador de $\mathbb{E}_{y}[\,|y-\mu|\,]$ no es la media sino la **mediana condicional** $\mathrm{med}(y\mid c)$. La mediana es robusta a outliers y, a diferencia de la media, *si* es un valor que la distribucion puede tomar; por eso $L_1$ tiende a producir bordes algo mas nitidos que $L_2$. En la practica las diferencias dependen del dominio: Pathak et al. "experimentaron con L1 y L2 y no encontraron diferencia significativa" para inpainting, mientras que en super-resolucion y traduccion imagen-a-imagen $L_1$ suele preferirse por su nitidez. En ambos casos el problema de fondo persiste —regresion a un estadistico puntual— y solo se resuelve modelando la *distribucion* completa (GAN, difusion, o targets discretos).

### I.4 El autoencoder lineal es PCA

Un resultado clasico ancla el bottleneck a un metodo conocido. Sea un autoencoder **lineal sin no-linealidades**: $z = W_e\, x$ con $W_e \in \mathbb{R}^{d\times D}$ y $\hat{x} = W_d\, z$ con $W_d \in \mathbb{R}^{D\times d}$. Asumiendo datos centrados ($\mathbb{E}[x]=0$), el objetivo es:

$$
\min_{W_e, W_d}\; \mathbb{E}\big[\, \lVert x - W_d W_e\, x \rVert_2^2 \,\big]
$$

Sea $M = W_d W_e \in \mathbb{R}^{D\times D}$ una matriz de rango a lo sumo $d$. El problema es buscar la mejor aproximacion de rango $d$ a la identidad bajo la covarianza $\Sigma = \mathbb{E}[xx^\top]$. El **teorema de Eckart–Young** garantiza que el optimo se obtiene proyectando sobre los $d$ vectores propios principales de $\Sigma$: si $\Sigma = U\Lambda U^\top$ con autovalores ordenados $\lambda_1 \ge \dots \ge \lambda_D$, entonces

$$
W_d W_e = U_d U_d^\top, \qquad U_d = [\,u_1, \dots, u_d\,]
$$

es decir, **el autoencoder lineal optimo proyecta sobre el subespacio principal de PCA** (Baldi & Hornik, 1989). El codigo $z$ recupera las $d$ componentes principales (salvo una rotacion arbitraria $A$ invertible, ya que $W_d A^{-1}\, A W_e$ deja $M$ invariante). La leccion conceptual: lo que da poder a los autoencoders profundos no es el bottleneck por si solo —que sin no-linealidad es solo PCA— sino la composicion de no-linealidades, que permite descubrir variedades curvas en lugar de subespacios planos.

---

## Parte II — La perdida contrastiva: InfoNCE y NT-Xent

### II.1 La formulacion NT-Xent de SimCLR

[SimCLR](/papers/simclr-chen-2020) define la tarea como: dado un minibatch de $N$ ejemplos, generar $2N$ vistas aumentadas y, para cada par positivo $(i,j)$ —dos vistas del mismo ejemplo—, identificar a $j$ entre las demas $2N-1$ vistas. Sea la **similitud coseno**

$$
\mathrm{sim}(u,v) = \frac{u^\top v}{\lVert u\rVert\,\lVert v\rVert}
$$

entre los embeddings proyectados $z = g(f(\tilde{x}))$. La perdida **NT-Xent** (*Normalized Temperature-scaled Cross Entropy*) para el par positivo $(i,j)$ es:

$$
\ell_{i,j} = -\log \frac{\exp\!\big(\mathrm{sim}(z_i,z_j)/\tau\big)}{\sum_{k=1}^{2N}\mathbb{1}_{[k\neq i]}\,\exp\!\big(\mathrm{sim}(z_i,z_k)/\tau\big)}
$$

donde $\mathbb{1}_{[k\neq i]}\in\{0,1\}$ excluye el propio termino $z_i$ y $\tau>0$ es la **temperatura**. La perdida total promedia sobre todos los pares positivos en *ambos sentidos* $(i,j)$ y $(j,i)$:

$$
\mathcal{L} = \frac{1}{2N}\sum_{k=1}^{N}\big[\,\ell_{2k-1,\,2k} + \ell_{2k,\,2k-1}\,\big]
$$

### II.2 Por que es "clasificar el positivo entre negativos"

La estructura de $\ell_{i,j}$ es exactamente la de una **entropia cruzada softmax** sobre $2N-1$ clases candidatas, donde la "clase correcta" es el positivo $j$ y todos los $z_k$ ($k\neq i,j$) son negativos. Escribiendo $s_k = \mathrm{sim}(z_i,z_k)/\tau$ como logits, $\ell_{i,j} = -\log \mathrm{softmax}(s)_j$. La red minimiza esta perdida empujando $z_i$ y $z_j$ juntos (numerador grande) y separando $z_i$ de todos los negativos (denominador chico). Por eso la cantidad de negativos importa tanto: con batch $N=8192$ cada positivo enfrenta $2(N-1)=16382$ negativos.

El **gradiente** revela una propiedad de *hard negative mining* automatico. Para un negativo $z_k$, el coeficiente que lo aleja es proporcional a su probabilidad softmax $p_{ik} = \mathrm{softmax}(s)_k$:

$$
\frac{\partial \ell_{i,j}}{\partial \,\mathrm{sim}(z_i,z_k)} = \frac{1}{\tau}\,p_{ik}, \qquad k\neq j
$$

Los negativos *mas similares* a $z_i$ (los mas dificiles, $p_{ik}$ grande) reciben el gradiente mas fuerte. Esto explica por que NT-Xent supera a perdidas de margen o logisticas que requieren *semi-hard negative mining* manual: la ponderacion por dificultad esta incorporada en el softmax.

### II.3 El rol de la temperatura $\tau$

La temperatura controla la concentracion del softmax. Para $\tau \to 0$ la distribucion se vuelve casi one-hot: domina el negativo mas duro y el gradiente se concentra alli (sensible al ruido, gradientes grandes). Para $\tau$ grande la distribucion se aplana y todos los negativos contribuyen casi por igual, perdiendo el efecto de *hard negative mining*. SimCLR encuentra que temperaturas demasiado altas ($\tau=1,10,100$) degradan el top-1, y la normalizacion $\ell_2$ es indispensable: sin ella la *precision de la tarea contrastiva* sube pero la calidad de la representacion empeora. MoCo usa $\tau=0.07$ por defecto. Intuitivamente, $\tau$ fija el radio de la "zona de influencia" alrededor de cada ancla en la hiperesfera unitaria donde viven los embeddings normalizados.

### II.4 InfoNCE como cota inferior de la informacion mutua

La forma generica, **InfoNCE** (van den Oord et al., 2018), usa producto punto como score $f(q,k)=\exp(q^\top k/\tau)$ sobre una clave positiva $k^+$ y $K$ negativas:

$$
\mathcal{L}_q = -\log \frac{\exp(q^\top k^+/\tau)}{\sum_{i=0}^{K}\exp(q^\top k_i/\tau)}
$$

El resultado teorico central es que **minimizar InfoNCE maximiza una cota inferior de la informacion mutua** $I(X;Y)$ entre las dos vistas. Para un conjunto de $K+1$ candidatos con un positivo muestreado de $p(y\mid x)$ y $K$ negativos de $p(y)$, el clasificador optimo asigna densidad proporcional a la razon $\tfrac{p(y\mid x)}{p(y)}$. Sustituyendo en la perdida esperada se obtiene:

$$
\mathcal{L}_{\text{NCE}} \;\ge\; \log(K+1) - I(X;Y)
\quad\Longleftrightarrow\quad
I(X;Y) \;\ge\; \log(K+1) - \mathcal{L}_{\text{NCE}}
$$

Dos consecuencias: (i) la cota mejora con **mas negativos** $K$ —otra razon formal para batches grandes o colas largas—; (ii) la cota esta acotada por $\log(K+1)$, por lo que con pocos negativos es floja. Conviene matizar, como hace el propio SimCLR citando a Tschannen et al. (2019): no esta claro si el exito se debe a la maximizacion de informacion mutua o "a la forma especifica de la perdida contrastiva". La cota es una motivacion, no una explicacion completa.

---

## Parte III — SimCLR vs MoCo, formalmente

Ambos optimizan la misma perdida InfoNCE; difieren en **de donde salen los negativos** y **como se codifican las claves**.

### III.1 SimCLR: negativos del batch

En SimCLR los $2N-2$ negativos de cada ancla provienen del **mismo minibatch**, codificados por la red *actual* $f_\theta$ con gradiente activo. Esto da dos propiedades: las claves son perfectamente **consistentes** (todas vienen del mismo $\theta$) pero su numero esta **atado al tamano del batch**. Para tener muchos negativos hay que usar batches enormes (4096–8192), lo que exige el optimizador LARS y decenas de cores de TPU. El numero de negativos $K = 2N-2$ y el costo de memoria/computo crecen *linealmente* con $N$.

### III.2 MoCo: cola FIFO + momentum encoder

[MoCo](/papers/moco-he-2019) **desacopla** el numero de negativos del batch mediante un diccionario implementado como **cola FIFO** de tamano $K$ (p. ej. $K=65536 \gg N$). Cada iteracion: el batch actual se *encola* y el batch mas antiguo se *desencola*. Solo se almacenan los embeddings (no se recomputan), asi que mantener miles de negativos cuesta poco.

El problema que crea la cola: las claves fueron codificadas por **versiones distintas y pasadas** del encoder, por lo que backprop a traves de toda la cola es intratable. Copiar ingenuamente $\theta_k \leftarrow \theta_q$ falla (la perdida oscila y no converge). La solucion es el **momentum encoder** con media movil exponencial (EMA):

$$
\boxed{\;\theta_k \leftarrow m\,\theta_k + (1-m)\,\theta_q\;}, \qquad m\in[0,1)
$$

donde **solo $\theta_q$ se actualiza por backprop**; $\theta_k$ se arrastra suavemente. Para ver por que la EMA da consistencia, desenrollemos la recursion. Si $\theta_q^{(t)}$ es la trayectoria del encoder de queries, entonces

$$
\theta_k^{(t)} = (1-m)\sum_{i=0}^{t-1} m^{\,i}\, \theta_q^{(t-i)} \;+\; m^{t}\,\theta_k^{(0)}
$$

es un promedio ponderado exponencialmente del pasado de $\theta_q$, con escala de tiempo efectiva $\sim 1/(1-m)$. Con $m=0.999$ esa escala es $\sim 1000$ iteraciones: $\theta_k$ cambia *mil veces mas lento* que $\theta_q$. La diferencia entre dos claves codificadas con pocas iteraciones de separacion es por tanto minuscula —de ahi la **consistencia** de las keys en la cola—. La ablacion de MoCo lo confirma: $m=0$ falla, $m=0.9$ da 55.2%, $m=0.999$ da 59.0%.

### III.3 El trade-off memoria / consistencia

| Aspecto | SimCLR | MoCo |
|---|---|---|
| Fuente de negativos | el propio batch ($K=2N-2$) | cola FIFO ($K$ independiente de $N$) |
| Consistencia de claves | maxima (mismo $\theta$) | alta gracias a la EMA con $m\to 1$ |
| Tamano efectivo de diccionario | atado al batch (memoria de gradientes) | desacoplado (solo embeddings) |
| Costo dominante | batch enorme + TPUs | mantener la cola (barato) |
| Encoder de claves | $f_\theta$ con gradiente | $f_{\theta_k}$ por momentum, sin gradiente |

El compromiso central: la cola hace el diccionario **grande y barato** (memoria), pero introduce **inconsistencia** entre claves de distintas iteraciones; la EMA recupera la consistencia a cambio de actualizar las claves lentamente. SimCLR evita la inconsistencia por construccion pero paga el costo en memoria del batch. Ambos comparten un detalle de ingenieria por la misma razon —evitar que Batch Normalization filtre la "firma" del sub-batch—: MoCo usa **Shuffling BN** entre GPUs y SimCLR usa **Global BN** agregando estadisticas sobre todos los dispositivos.

---

## Parte IV — Masked modeling: el MAE

### IV.1 La tarea de reconstruccion de parches

[MAE](/papers/mae-he-2022) divide la imagen en parches regulares no solapados, oculta un subconjunto aleatorio y reconstruye los parches faltantes. Sea $\mathcal{P}=\{p_1,\dots,p_n\}$ el conjunto de parches y $\mathcal{M}\subset\mathcal{P}$ el subconjunto enmascarado con razon $\rho = |\mathcal{M}|/n$. El encoder ve solo los visibles $\mathcal{V}=\mathcal{P}\setminus\mathcal{M}$ y el decoder reconstruye $\hat{p}$ para los parches de $\mathcal{M}$.

### IV.2 Por que mascara alta (75%) hace la tarea no trivial

La razon de enmascaramiento de MAE ($\rho\approx 0.75$) es muchisimo mas alta que el 15% de BERT. El argumento es de **redundancia espacial**: las imagenes naturales tienen fuerte correlacion local, asi que con poca mascara un parche faltante se recupera por simple interpolacion de sus vecinos —tarea de bajo nivel que no exige semantica—. Al ocultar tres cuartas partes con muestreo uniforme, la interpolacion local deja de bastar y el modelo debe razonar sobre la *gestalt* de objetos y escenas: inferir contenido a partir de evidencia lejana y escasa. Empiricamente, en *linear probing* la precision sube de forma sostenida hasta el punto dulce (54.6% al 10% de mascara vs 73.5% al 75%). La mascara alta cumple ademas el rol regularizador que en el contrastivo cumplen las augmentaciones fuertes.

### IV.3 El encoder asimetrico y su ahorro

El diseno clave es **asimetrico**: el encoder (un ViT) procesa *solo* los parches visibles —los enmascarados se eliminan, no se reemplazan por tokens de mascara— y un decoder ligero recibe el conjunto completo (visibles codificados + tokens de mascara compartidos, todos con embeddings posicionales). El ahorro proviene de la complejidad **cuadratica** de la autoatencion en el numero de tokens: si el encoder procesa solo una fraccion $(1-\rho)$ de los $n$ tokens, el costo de atencion cae de $O(n^2)$ a

$$
O\big(((1-\rho)\,n)^2\big) = (1-\rho)^2\, O(n^2)
$$

Con $\rho=0.75$ el factor es $(0.25)^2 = 1/16$ en la atencion del encoder. En la practica MAE reporta $3.3\times$ menos FLOPs y un *speedup* de pared de $2.8\times$ (hasta $4.1\times$ con encoder grande). Crucialmente, no usar tokens de mascara en el encoder evita la **brecha de distribucion** entre preentrenamiento (entrada con muchas mascaras) y despliegue (imagenes integras): meter tokens de mascara en el encoder empeora el linear probing en 14 puntos.

### IV.4 La perdida: MSE sobre parches ocultos normalizados

La perdida es el **error cuadratico medio computado solo sobre los parches enmascarados** (computarlo sobre todos baja ~0.5% la precision). Para cada parche oculto $p\in\mathcal{M}$ con vector de pixeles ground-truth, MAE usa la variante de **pixeles normalizados por parche**: se calcula media $\mu_p$ y desviacion $\sigma_p$ de cada parche y se normaliza el target

$$
\tilde{p}_i = \frac{p_i - \mu_p}{\sigma_p + \epsilon}, \qquad
\mathcal{L}_{\text{MAE}} = \frac{1}{|\mathcal{M}|}\sum_{p\in\mathcal{M}}\frac{1}{|p|}\sum_{i\in p}\big(\hat{p}_i - \tilde{p}_i\big)^2
$$

La normalizacion por parche realza el contraste local y enfatiza componentes de alta frecuencia, mejorando la representacion (85.4% vs 84.9% en fine-tuning). Notese que sigue siendo regresion $L_2$ —y por tanto sujeta al sesgo del promedio condicional de la Parte I.2—; lo que rescata a MAE es que la tarea es lo bastante dura como para que la representacion *latente* sea rica aunque el pixel reconstruido sea algo borroso. He et al. son explicitos: las imagenes son "luz registrada" sin descomposicion semantica analoga a las palabras.

---

## Parte V — Consistency training: UDA

### V.1 La perdida de consistencia con stop-gradient

El [consistency training](/fundamentos/aprendizaje-semi-supervisado) parte de un postulado: un buen clasificador debe ser **invariante a perturbaciones que preserven la etiqueta**. Si $\hat{x}\sim q(\hat{x}\mid x)$ es una version aumentada de $x$, sus predicciones deben coincidir. [UDA](/papers/uda-xie-2019) penaliza la divergencia entre la prediccion del original (objetivo) y la del aumentado:

$$
\mathcal{L}_{\text{cons}} = \mathbb{E}_{x\sim p_U(x)}\,\mathbb{E}_{\hat{x}\sim q(\hat{x}\mid x)}\Big[\,\mathrm{KL}\big(p_{\tilde\theta}(y\mid x)\,\big\|\,p_\theta(y\mid \hat{x})\big)\Big]
$$

donde $\tilde\theta$ es una **copia congelada (stop-gradient)** de $\theta$: el gradiente *no* fluye por el objetivo, que actua como un "profesor" instantaneo. Esto es esencial —heredado de VAT— para evitar la solucion degenerada en que el modelo colapsa todas las predicciones a una distribucion constante para minimizar trivialmente la consistencia.

### V.2 Derivacion: por que CE equivale a KL con stop-gradient

La divergencia KL entre la distribucion objetivo $p_{\tilde\theta}(\cdot\mid x)$ (fija) y la del aumentado $p_\theta(\cdot\mid\hat{x})$ se descompone:

$$
\mathrm{KL}\big(p_{\tilde\theta}\,\|\,p_\theta\big)
= \sum_{y} p_{\tilde\theta}(y\mid x)\,\log\frac{p_{\tilde\theta}(y\mid x)}{p_\theta(y\mid\hat{x})}
= \underbrace{-\sum_{y} p_{\tilde\theta}(y\mid x)\,\log p_\theta(y\mid\hat{x})}_{\text{cross-entropy } H(p_{\tilde\theta},\,p_\theta)}
\;\underbrace{-\;\Big(-\sum_{y} p_{\tilde\theta}\log p_{\tilde\theta}\Big)}_{-H(p_{\tilde\theta})}
$$

El segundo termino es la **entropia del objetivo** $H(p_{\tilde\theta}(\cdot\mid x))$. Como $\tilde\theta$ esta congelado por el stop-gradient, $H(p_{\tilde\theta})$ es **constante respecto de $\theta$** y desaparece al derivar:

$$
\nabla_\theta\,\mathrm{KL}\big(p_{\tilde\theta}\,\|\,p_\theta\big) = \nabla_\theta\, H(p_{\tilde\theta},\,p_\theta) = -\nabla_\theta \sum_{y} p_{\tilde\theta}(y\mid x)\,\log p_\theta(y\mid\hat{x})
$$

Por eso UDA puede escribir la consistencia como **cross-entropy** y aun asi describirla como "minimizar la KL entre la prediccion del dato y su version transformada": con el target fijo, optimizar CE u optimizar KL es identico en gradiente.

### V.3 La perdida total y las tres tecnicas de control

El objetivo combina la supervisada (sobre las pocas etiquetas) con la consistencia (sobre los muchos no etiquetados), ponderada por $\lambda$ (UDA fija $\lambda=1$):

$$
\min_\theta\; \mathcal{J}(\theta) = \underbrace{\mathbb{E}_{x\sim p_L}\big[-\log p_\theta(y^\star\mid x)\big]}_{\text{supervisada}} + \lambda\,\mathcal{L}_{\text{cons}}
$$

Tres tecnicas hacen viable el regimen de poquisimas etiquetas:

**Training Signal Annealing (TSA).** Con 20 etiquetas y millones de ejemplos sin etiquetar, el modelo *sobreajusta* los pocos labels mientras aun *subajusta* el resto. TSA remueve de la perdida supervisada los ejemplos donde la confianza supera un umbral creciente:

$$
\text{descartar } (x,y^\star) \;\text{ si }\; p_\theta(y^\star\mid x) > \eta_t,
\qquad \eta_t = \alpha_t\,(1 - \tfrac{1}{K}) + \tfrac{1}{K}
$$

con $\alpha_t$ creciendo de 0 a 1 segun un cronograma (exp, lineal o log) y $K$ el numero de clases. El umbral pasa de $1/K$ (azar) a 1, liberando la senal supervisada gradualmente y actuando como techo contra el sobreentrenamiento de ejemplos faciles.

**Confidence masking.** La consistencia se computa solo donde el modelo esta seguro: si $\max_y p_{\tilde\theta}(y\mid x) > \beta$ (p. ej. $\beta=0.8$ en CIFAR-10), evitando propagar consistencia desde predicciones ruidosas.

**Sharpening con temperatura.** Como regularizar hacia baja entropia ayuda (Grandvalet & Bengio, 2005), el objetivo se *afila* con una temperatura $\tau<1$ sobre los logits $z$ del original:

$$
p^{(\text{sharp})}_{\tilde\theta}(y\mid x) = \frac{\exp(z_y/\tau)}{\sum_{y'}\exp(z_{y'}/\tau)}, \qquad \tau=0.4
$$

Una temperatura $<1$ vuelve el target mas puntiagudo (cercano a one-hot), reforzando predicciones confiadas —el efecto opuesto al $\tau>1$ que aplana el softmax—.

---

## Parte VI — Rotacion como clasificacion e instance discrimination

### VI.1 RotNet: cross-entropy sobre 4 clases

[RotNet](/papers/rotnet-gidaris-2018) define la *pretext task* como clasificar la rotacion $y\in\{0°,90°,180°,270°\}$ (K=4) aplicada a una imagen. Sea $g(X\mid y)$ la operacion que rota $X$ en la clase $y$ y $F^y(\cdot\mid\theta)$ la probabilidad predicha para la rotacion $y$. El objetivo promedia la cross-entropy sobre las cuatro rotaciones de cada imagen:

$$
\min_\theta \; \frac{1}{N}\sum_{i=1}^{N}\mathrm{loss}(X_i,\theta),
\qquad
\mathrm{loss}(X_i,\theta) = -\frac{1}{K}\sum_{y=1}^{K}\log F^{y}\big(g(X_i\mid y)\,\big|\,\theta\big)
$$

Las cuatro rotaciones se implementan con *flip* y *transpose* exactos (sin interpolacion), de modo que no dejan artefactos de bajo nivel que la red pudiera explotar como atajo.

### VI.2 Por que predecir rotacion fuerza semantica

El argumento de **orientacion canonica**: para decidir en cuantos grados se roto una imagen, la red *necesariamente* debe reconocer y localizar los objetos, identificar su tipo y sus partes (ojos, narices, colas) y conocer la orientacion erguida con que esos objetos suelen aparecer en fotografias capturadas por humanos. No hay atajo de bajo nivel que resuelva la rotacion sin entender la escena —"alguien que no conoce los conceptos de los objetos no puede reconocer la rotacion que se les aplico"—. La tarea esta **bien definida** (*well-posed*) precisamente porque existe una orientacion canonica; pierde sentido en objetos rotacionalmente simetricos o en dominios sin "arriba" definido (imagenes aereas, microscopia, escaneos medicos), una limitacion directa de este sesgo. La eleccion de K=4 es optima: 8 rotaciones (multiplos de 45°) introducen interpolacion y artefactos, y bajan la precision.

### VI.3 Instance discrimination y el softmax sobre instancias

Una linea complementaria —antecedente directo de SimCLR/MoCo, formalizada por [Ye et al.](/papers/aprendizaje-contrastivo)— trata **cada imagen como su propia clase**. El embedding debe ser *invariante* (la misma instancia aumentada da el mismo rasgo) y *spread-out* (instancias distintas se separan). La probabilidad de reconocer una vista $f_j$ como la instancia $i$ es un softmax sobre similitudes coseno entre rasgos $\ell_2$-normalizados:

$$
P(i\mid f_j) = \frac{\exp\!\big(v_i^\top f_j/\tau\big)}{\sum_{k=1}^{n}\exp\!\big(v_k^\top f_j/\tau\big)}
$$

donde $v_i$ es el rasgo memorizado de la instancia $i$ y $\tau$ la temperatura. Esta es la misma estructura del NT-Xent (Parte II): un clasificador softmax cuya "etiqueta" es la identidad de la instancia. Wu et al. (2018) lo implementaron con un *memory bank* que guarda cada $v_i$ y se actualiza una vez por epoca; Ye et al. notaron que esa desactualizacion entorpece el entrenamiento y propusieron usar el batch en tiempo real. SimCLR reconoceria despues esta receta —positivo por aumentacion, negativos del batch, softmax coseno con temperatura— como el nucleo del aprendizaje contrastivo moderno, cerrando el circulo con la Parte II.

---

> **Sintesis.** El SSL convierte estructura de los datos en supervision. Tres familias y su matematica: **(i) generativa** (autoencoder, Context Encoders, [MAE](/papers/mae-he-2022)) — reconstruir lo que falta, con el sesgo del promedio condicional bajo $L_2$ como limitacion central; **(ii) contrastiva** ([SimCLR](/papers/simclr-chen-2020), [MoCo](/papers/moco-he-2019)) — clasificar el positivo entre negativos via InfoNCE/NT-Xent, cota inferior de la informacion mutua, con el eje memoria-vs-consistencia separando ambos metodos; **(iii) predictiva/consistency** ([RotNet](/papers/rotnet-gidaris-2018), [UDA](/papers/uda-xie-2019)) — pretext tasks que fuerzan semantica e invarianza a perturbaciones via KL con stop-gradient. Fundamentos relacionados: [aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado), [aprendizaje contrastivo](/fundamentos/aprendizaje-contrastivo), [aprendizaje semi-supervisado](/fundamentos/aprendizaje-semi-supervisado).
