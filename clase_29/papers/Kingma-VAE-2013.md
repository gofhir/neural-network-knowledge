# Auto-Encoding Variational Bayes (VAE) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Auto-Encoding Variational Bayes*.
- **Autores:** Diederik P. Kingma y Max Welling, ambos del Machine Learning Group de la Universiteit van Amsterdam.
- **Venue:** *International Conference on Learning Representations* (ICLR) 2014.
- **Año / preprint:** Primera versión en arXiv el 20 de diciembre de 2013; arXiv:1312.6114 ([arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)).
- **Abreviaturas que introduce:** **SGVB** (Stochastic Gradient Variational Bayes, el estimador), **AEVB** (Auto-Encoding Variational Bayes, el algoritmo) y, como caso particular cuando el modelo de reconocimiento es una red neuronal, el **VAE** (Variational Auto-Encoder).

El paper resuelve una pregunta de inferencia bayesiana que parece muy abstracta —"¿cómo hacer inferencia y aprendizaje eficientes en modelos dirigidos con variables latentes continuas cuyo posterior es intratable, y con datasets grandes?"— y termina entregando uno de los modelos generativos más influyentes de la década. Es importante separar las dos lecturas. La lectura *de fondo* (la que da el título) es estadística: un nuevo estimador del *lower bound* variacional, diferenciable y de baja varianza, optimizable con gradiente estocástico estándar. La lectura *que la comunidad adoptó* es arquitectónica: cuando el modelo de inferencia se implementa con una red neuronal, "llegamos al variational auto-encoder", un autoencoder reformulado como **modelo generativo probabilístico**.

Las dos contribuciones que el propio abstract enumera son: (1) una **reparametrización del lower bound variacional** que produce un estimador que se optimiza directamente con métodos de gradiente estocástico estándar; y (2) para datasets i.i.d. con variables latentes continuas por dato, mostrar que la inferencia del posterior se vuelve especialmente eficiente al ajustar un **modelo de inferencia aproximado** (también llamado *recognition model*) al posterior intratable usando ese estimador. La frase clave del cierre de la introducción condensa todo: "cuando se usa una red neuronal para el modelo de reconocimiento, llegamos al variational auto-encoder".

Para la Clase 29 (Modelos Generativos en Visión) esto importa porque el VAE es **la primera familia generativa con espacio latente continuo** que la clase presenta en la transición Autoencoders → VAE. El autoencoder clásico aprende un espacio latente comprimido pero no probabilístico: puede reconstruir, no generar de forma principiada. El VAE añade el marco probabilístico —un *prior* sobre el latente y un posterior aproximado gaussiano— que convierte ese espacio latente en algo *muestreable*, habilitando generación e interpolación suave. Y, mirando hacia adelante en la propia clase, el VAE es además el componente que **comprime al espacio latente** en Latent Diffusion / Stable Diffusion: el autoencoder del Stable Diffusion es exactamente un VAE (con regularización adicional) cuyo encoder lleva la imagen a un latente sobre el cual opera el modelo de difusión.

## 2. Contexto: inferencia variacional y autoencoders no generativos

### 2.1. El problema bayesiano de fondo

Considérese un dataset $X = \{x^{(i)}\}_{i=1}^N$ de $N$ muestras i.i.d. de una variable $x$ (continua o discreta). El paper postula que los datos se generan por un proceso de dos pasos: primero se muestrea una variable latente continua **no observada** $z^{(i)} \sim p_{\theta^*}(z)$ desde un *prior*; luego se genera $x^{(i)} \sim p_{\theta^*}(x \mid z)$ desde una *verosimilitud condicional*. Tanto el prior $p_\theta(z)$ como la verosimilitud $p_\theta(x\mid z)$ pertenecen a familias paramétricas diferenciables casi en todas partes respecto de $\theta$ y de $z$. El problema es que casi todo está oculto: ni los parámetros verdaderos $\theta^*$ ni los valores de las latentes $z^{(i)}$ son conocidos.

La dificultad central tiene dos caras que el paper nombra explícitamente:

- **Intratabilidad.** La verosimilitud marginal $p_\theta(x) = \int p_\theta(z)\,p_\theta(x\mid z)\,dz$ no se puede evaluar ni diferenciar, porque la integral sobre todas las latentes es intratable. En consecuencia el posterior verdadero $p_\theta(z\mid x) = p_\theta(x\mid z)\,p_\theta(z)/p_\theta(x)$ también es intratable —lo que descarta el algoritmo EM— y las integrales que requeriría un esquema de *mean-field* variacional razonable también lo son. El paper subraya que estas intratabilidades son comunes y aparecen apenas la verosimilitud $p_\theta(x\mid z)$ es moderadamente complicada, por ejemplo "una red neuronal con una capa oculta no lineal".
- **Datasets grandes.** Hay tanta data que la optimización por *batch* es demasiado costosa; se quieren actualizaciones por minibatch o incluso por dato. Las soluciones basadas en muestreo como Monte Carlo EM serían en general demasiado lentas, porque implican un costoso *loop* de muestreo por cada dato.

El enfoque variacional bayesiano (VB) clásico ataca esto optimizando una aproximación al posterior intratable. Pero el *mean-field* estándar requiere soluciones analíticas de esperanzas respecto del posterior aproximado, que son igualmente intratables en el caso general. Ese es el muro que el paper rompe.

### 2.2. El modelo de reconocimiento como encoder/decoder probabilístico

La pieza que destraba todo es introducir un **modelo de reconocimiento** $q_\phi(z\mid x)$: una aproximación al posterior verdadero intratable $p_\theta(z\mid x)$. A diferencia del mean-field, no se asume que sea factorizable ni que sus parámetros provengan de una esperanza en forma cerrada; en su lugar, se aprenden los parámetros $\phi$ del modelo de reconocimiento **conjuntamente** con los parámetros generativos $\theta$. La Figura 1 del paper lo dibuja como un modelo gráfico dirigido: líneas sólidas para el modelo generativo $p_\theta(z)\,p_\theta(x\mid z)$, líneas punteadas para la aproximación variacional $q_\phi(z\mid x)$.

Aquí entra la reinterpretación que conecta con los autoencoders. Desde la teoría de codificación, las latentes $z$ son una *representación latente* o *código*. Por eso el paper rebautiza:

- $q_\phi(z\mid x)$ como **encoder probabilístico**: dado un dato $x$, produce una *distribución* (por ejemplo gaussiana) sobre los posibles códigos $z$ de los que $x$ podría haberse generado.
- $p_\theta(x\mid z)$ como **decoder probabilístico**: dado un código $z$, produce una *distribución* sobre los posibles valores de $x$ correspondientes.

Esta es la diferencia conceptual con el **autoencoder no generativo** clásico. Un autoencoder tradicional (incluyendo variantes regularizadas como *denoising*, *contractive* o *sparse*) mapea $x$ a un código determinista y lo reconstruye, minimizando un error de reconstrucción. El paper cita el resultado conocido (Vincent et al., 2010; principio infomax de Linsker, 1989) de que entrenar autoencoders no regularizados equivale a maximizar una cota inferior de la información mutua entre la entrada $X$ y la representación $Z$. Pero —y esto es central— "es bien sabido que este criterio de reconstrucción en sí mismo no basta para aprender representaciones útiles": de ahí la necesidad de regularizaciones ad hoc con hiperparámetros de ajuste. El VAE elimina esa nuisance: su término de regularización **lo dicta la cota variacional**, no un hiperparámetro inventado. El autoencoder se vuelve, por construcción, un modelo probabilístico.

## 3. Contribución central

La contribución se puede formular en tres niveles encajados:

1. **El estimador SGVB.** Una reformulación del lower bound variacional que, vía reparametrización, produce un estimador Monte Carlo **diferenciable y de baja varianza** del bound y de sus derivadas respecto de $\theta$ y $\phi$. Funciona para casi cualquier modelo con variables latentes continuas y se optimiza con ascenso de gradiente estocástico estándar (SGD, Adagrad).
2. **El algoritmo AEVB.** Para el caso i.i.d. con latentes por dato, usar el estimador SGVB para ajustar un modelo de reconocimiento $q_\phi(z\mid x)$ que permite inferencia aproximada del posterior por *ancestral sampling* simple, evitando esquemas iterativos caros (MCMC) por cada dato.
3. **El VAE.** El caso particular en que encoder y decoder son redes neuronales (MLPs), con prior gaussiano isotrópico y posterior aproximado gaussiano diagonal. Es la materialización práctica que la comunidad de visión adoptó.

La reformulación del autoencoder como modelo generativo probabilístico es lo que lo hace generativo: una vez entrenado, se puede **descartar el encoder, muestrear $z \sim p(z)$ del prior, y pasarlo por el decoder** para generar datos nuevos que se parecen a los reales. El autoencoder clásico no puede hacer esto porque su espacio latente no tiene una distribución conocida de la cual muestrear.

## 4. Método

### 4.1. Derivación del ELBO (Evidence Lower Bound)

La verosimilitud marginal del dataset es suma de las marginales por dato: $\log p_\theta(x^{(1)}, \dots, x^{(N)}) = \sum_i \log p_\theta(x^{(i)})$. Cada término se reescribe exactamente (ecuación 1 del paper) como:

$$\log p_\theta(x^{(i)}) = D_{\mathrm{KL}}\big(q_\phi(z\mid x^{(i)}) \,\|\, p_\theta(z\mid x^{(i)})\big) + \mathcal{L}(\theta, \phi; x^{(i)})$$

El primer término del lado derecho es la divergencia KL del posterior aproximado respecto del verdadero. Como la KL es **no negativa**, el segundo término $\mathcal{L}(\theta,\phi;x^{(i)})$ es una *cota inferior* (lower bound) sobre la log-verosimilitud marginal de ese dato —el famoso **ELBO**:

$$\log p_\theta(x^{(i)}) \geq \mathcal{L}(\theta,\phi;x^{(i)}) = \mathbb{E}_{q_\phi(z\mid x)}\big[-\log q_\phi(z\mid x) + \log p_\theta(x, z)\big]$$

Maximizar $\mathcal{L}$ respecto de $\theta$ y $\phi$ logra dos cosas a la vez: empuja la log-verosimilitud hacia arriba *y* aprieta la KL entre posterior aproximado y verdadero (porque la brecha entre $\log p_\theta(x)$ y $\mathcal{L}$ es exactamente esa KL). El ELBO admite una segunda forma (ecuación 3), que es la que da la lectura "autoencoder":

$$\mathcal{L}(\theta,\phi;x^{(i)}) = -\,D_{\mathrm{KL}}\big(q_\phi(z\mid x^{(i)}) \,\|\, p_\theta(z)\big) + \mathbb{E}_{q_\phi(z\mid x^{(i)})}\big[\log p_\theta(x^{(i)}\mid z)\big]$$

Esta es la ecuación que hay que retener:

$$\boxed{\;\text{ELBO} = \underbrace{\mathbb{E}_{q_\phi(z\mid x)}\big[\log p_\theta(x\mid z)\big]}_{\text{reconstrucción}} - \underbrace{D_{\mathrm{KL}}\big(q_\phi(z\mid x) \,\|\, p_\theta(z)\big)}_{\text{regularización}}\;}$$

El primer término es el **error de reconstrucción negativo esperado**: cuán bien el decoder reconstruye $x$ a partir de un $z$ muestreado del encoder. El segundo es la **KL del posterior aproximado respecto del prior**, que actúa como regularizador empujando a $q_\phi(z\mid x)$ a parecerse al prior $p_\theta(z)$. El paper enfatiza que este término de regularización *emerge del bound*, no es un hiperparámetro añadido a mano como en los autoencoders regularizados.

### 4.2. El problema del gradiente y por qué hace falta la reparametrización

Se quiere diferenciar y optimizar $\mathcal{L}$ respecto de $\theta$ **y** $\phi$. Respecto de $\theta$ no hay problema. Respecto de $\phi$ sí: la esperanza está tomada bajo $q_\phi(z\mid x)$, que *depende de $\phi$*, de modo que no se puede simplemente intercambiar gradiente y esperanza. El estimador Monte Carlo ingenuo para este tipo de problema —el llamado *score function* o REINFORCE— es

$$\nabla_\phi\, \mathbb{E}_{q_\phi(z)}[f(z)] = \mathbb{E}_{q_\phi(z)}\big[f(z)\,\nabla_\phi \log q_\phi(z)\big]$$

y el paper lo descarta de inmediato: "exhibe muy alta varianza y es impráctico para nuestros propósitos". Ese es el obstáculo concreto que la reparametrización resuelve.

### 4.3. El reparameterization trick

La idea (sección 2.4 del paper) es deceptivamente simple. En vez de muestrear $z \sim q_\phi(z\mid x)$ directamente —operación no diferenciable respecto de $\phi$ porque el muestreo es estocástico—, se expresa $z$ como una **transformación determinista y diferenciable** de una variable de ruido auxiliar $\epsilon$ con distribución marginal independiente $p(\epsilon)$:

$$z = g_\phi(\epsilon, x), \qquad \epsilon \sim p(\epsilon)$$

Como $\epsilon$ no depende de $\phi$, toda la dependencia de $\phi$ queda dentro de la función determinista $g_\phi$, **a través de la cual sí se puede retropropagar**. La esperanza se reescribe:

$$\mathbb{E}_{q_\phi(z\mid x)}[f(z)] = \mathbb{E}_{p(\epsilon)}\big[f(g_\phi(\epsilon, x))\big] \approx \frac{1}{L}\sum_{l=1}^{L} f\big(g_\phi(\epsilon^{(l)}, x)\big), \quad \epsilon^{(l)} \sim p(\epsilon)$$

El caso gaussiano univariado es el ejemplo canónico y el que se usa en el VAE. Si $z \sim \mathcal{N}(\mu, \sigma^2)$, una reparametrización válida es:

$$\boxed{\; z = \mu + \sigma \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I) \;}$$

donde $\odot$ es el producto elemento a elemento. El encoder ya no emite una muestra: emite los **parámetros** $\mu$ y $\sigma$ de la gaussiana; el ruido $\epsilon$ se inyecta aparte, y el muestreo de $z$ se convierte en una operación aritmética diferenciable. Esto es exactamente lo que permite que el gradiente del ELBO fluya desde la pérdida de reconstrucción, a través de $z$, hasta los pesos del encoder $\phi$ — habilitando el entrenamiento *end-to-end* con backpropagation estándar. El paper enumera tres estrategias generales para encontrar $g_\phi$ y $p(\epsilon)$: (1) CDF inversa tratable (exponencial, Cauchy, logística, etc.); (2) familias *location-scale* —donde $g(\cdot) = \text{location} + \text{scale}\cdot\epsilon$— que incluyen la gaussiana, Laplace, Student-t, uniforme; (3) composición (log-normal, gamma, Dirichlet, beta, etc.).

### 4.4. El estimador SGVB y el algoritmo AEVB

Aplicando la reparametrización al ELBO se obtiene el estimador SGVB. El paper da dos versiones. La genérica (eq. 6) estima la esperanza completa por muestreo. La segunda y preferida (eq. 7), válida cuando la KL contra el prior se puede integrar **analíticamente**, deja solo el término de reconstrucción para estimar por muestreo y tiene menor varianza:

$$\widetilde{\mathcal{L}}^B(\theta,\phi;x^{(i)}) = -\,D_{\mathrm{KL}}\big(q_\phi(z\mid x^{(i)}) \,\|\, p_\theta(z)\big) + \frac{1}{L}\sum_{l=1}^{L} \log p_\theta\big(x^{(i)}\mid z^{(i,l)}\big)$$

con $z^{(i,l)} = g_\phi(\epsilon^{(i,l)}, x^{(i)})$ y $\epsilon^{(l)} \sim p(\epsilon)$. El **algoritmo AEVB** (Algoritmo 1) es entonces un *loop* minibatch ordinario: inicializar $\theta, \phi$; repetir —tomar un minibatch de $M$ datos, muestrear ruido $\epsilon$, calcular el gradiente $\nabla_{\theta,\phi}\widetilde{\mathcal{L}}^M$, actualizar con SGD/Adagrad— hasta convergencia. Un hallazgo práctico notable: el número de muestras $L$ por dato puede fijarse en **$L = 1$** siempre que el minibatch sea suficientemente grande, p. ej. $M = 100$. Es decir, una sola muestra de ruido por dato basta, lo que hace el VAE tan barato como un autoencoder normal.

### 4.5. El VAE concreto: prior gaussiano, KL en forma cerrada

En el ejemplo del VAE (sección 3) el prior sobre las latentes es la gaussiana isotrópica centrada $p_\theta(z) = \mathcal{N}(z; 0, I)$ —que notablemente *no tiene parámetros*— y la verosimilitud $p_\theta(x\mid z)$ es una gaussiana multivariada (para datos reales) o una Bernoulli (para datos binarios) cuyos parámetros se computan desde $z$ con un MLP. El posterior aproximado se elige gaussiano con covarianza diagonal: $\log q_\phi(z\mid x^{(i)}) = \log \mathcal{N}(z; \mu^{(i)}, \sigma^{2(i)} I)$, donde la media $\mu^{(i)}$ y la desviación $\sigma^{(i)}$ son salidas del MLP encoder.

Cuando prior y posterior aproximado son ambos gaussianos, la KL se integra analíticamente (apéndice B), dando el estimador cerrado para un dato:

$$\mathcal{L}(\theta,\phi;x^{(i)}) \approx \frac{1}{2}\sum_{j=1}^{J}\Big(1 + \log\big((\sigma_j^{(i)})^2\big) - (\mu_j^{(i)})^2 - (\sigma_j^{(i)})^2\Big) + \frac{1}{L}\sum_{l=1}^{L}\log p_\theta\big(x^{(i)}\mid z^{(i,l)}\big)$$

con $z^{(i,l)} = \mu^{(i)} + \sigma^{(i)} \odot \epsilon^{(l)}$ y $\epsilon^{(l)} \sim \mathcal{N}(0, I)$, y $J$ la dimensionalidad de $z$. El primer sumando es el negativo de la KL en forma cerrada; el segundo, el término de reconstrucción (un MLP Bernoulli con log-verosimilitud cross-entropy, o un MLP gaussiano, según el tipo de dato). El decoder Bernoulli (apéndice C.1) usa $\log p(x\mid z) = \sum_i x_i \log y_i + (1-x_i)\log(1-y_i)$ con $y = f_\sigma(W_2 \tanh(W_1 z + b_1) + b_2)$ — exactamente la cross-entropy binaria, motivo por el cual el VAE de MNIST se entrena con esa pérdida.

## 5. Experimentos

El paper entrena modelos generativos de imágenes sobre **MNIST** y el dataset de **Frey Faces** (rostros continuos), comparando algoritmos en términos del lower bound variacional y de la verosimilitud marginal estimada.

- **Configuración.** Encoder y decoder con igual número de unidades ocultas: 500 para MNIST, 200 para Frey Faces (menos, para evitar overfitting por ser un dataset más chico). Frey Faces es continuo, así que el decoder es gaussiano con medias restringidas a $(0,1)$ vía sigmoide. Parámetros inicializados desde $\mathcal{N}(0, 0.01)$, optimizados conjuntamente con Adagrad (global stepsize elegido de $\{0.01, 0.02, 0.1\}$), minibatches $M=100$, $L=1$. El *weight decay* corresponde a un prior $p(\theta) = \mathcal{N}(0, I)$, de modo que la optimización equivale a estimación MAP aproximada.
- **Lower bound (Figura 2).** Se compara AEVB contra el algoritmo **wake-sleep** (Hinton et al., 1995) para distintas dimensionalidades del espacio latente $N_z \in \{3, 5, 10, 20, 200\}$ en MNIST y $\{2,5,10,20\}$ en Frey Faces. AEVB **converge considerablemente más rápido y alcanza una mejor solución en todos los experimentos**. Un resultado contraintuitivo destacado: *más variables latentes no produce más overfitting* —incluso latentes superfluas no dañan—, lo que se explica por el **efecto regularizador del lower bound** (el término KL contra el prior penaliza usar capacidad latente innecesaria). El cómputo tomó unos 20–40 minutos por millón de muestras de entrenamiento en una CPU Intel Xeon a ~40 GFLOPS efectivos.
- **Verosimilitud marginal (Figura 3).** Para espacios latentes de baja dimensión (3 latentes, 100 unidades ocultas) se estima la verosimilitud marginal con un estimador MCMC (HMC). Se compara AEVB, wake-sleep y **Monte Carlo EM (MCEM)** con sampler Hybrid Monte Carlo, para tamaños de entrenamiento chico ($N=1000$) y grande ($N=50000$). MCEM no es online y no escala al MNIST completo; AEVB y wake-sleep sí. AEVB converge rápido en ambos regímenes.
- **Visualización del manifold latente (apéndice A, Figuras 4–5).** Con un espacio latente 2D, recorriendo coordenadas linealmente espaciadas transformadas por la CDF inversa gaussiana y pasándolas por el decoder $p_\theta(x\mid z)$, se obtienen manifolds 2D suaves de MNIST y Frey Faces: los dígitos y los rostros varían **continuamente** al desplazarse por el latente. Esta visualización es la demostración visual de que el espacio latente del VAE es continuo e interpolable —justo la propiedad que la Clase 29 destaca como ventaja sobre el autoencoder clásico.

## 6. Trabajo relacionado

El paper se sitúa respecto de varias líneas. El **wake-sleep** (Hinton et al., 1995) es el único otro método online aplicable a la misma clase general de modelos de latentes continuas y, como AEVB, usa un modelo de reconocimiento; su desventaja es que optimiza *dos* objetivos concurrentes que juntos no corresponden a optimizar (una cota de) la verosimilitud marginal —aunque tiene la ventaja de aplicar también a latentes discretas. La conexión autoencoder–modelo generativo lineal-gaussiano es antigua: Roweis (1998) mostró que **PCA** es la solución ML de un caso especial lineal-gaussiano. Vincent et al. (2010) conectaron autoencoders no regularizados con el principio infomax. De forma muy relevante, **Rezende, Mohamed y Wierstra (2014)** —*Stochastic Backpropagation and Variational Inference in Deep Latent Gaussian Models*— hacen la misma conexión entre autoencoders, modelos dirigidos y inferencia variacional usando esencialmente el mismo reparameterization trick, desarrollado *independientemente* y de forma simultánea. Ese par de papers (Kingma-Welling y Rezende et al.) es el origen conjunto del VAE.

## 7. Limitaciones

El paper original es modesto en autocrítica, pero las limitaciones que la comunidad identificó después —y que son las que la Clase 29 debe nombrar— son:

- **Imágenes borrosas.** La pérdida de reconstrucción gaussiana (un decoder $\mathcal{N}(x; \mu, \sigma^2 I)$ equivale a un error cuadrático medio entre la imagen y su media reconstruida) penaliza el promedio sobre modos plausibles. Frente a la incertidumbre, el decoder gaussiano *promedia* en vez de comprometerse con un detalle nítido, produciendo el característico aspecto **borroso/difuminado** de las muestras del VAE. Es una consecuencia directa de la forma de la verosimilitud, no un bug de implementación. Esta es la razón por la cual, en visión, las GANs (más nítidas) y luego la difusión desplazaron al VAE puro como generador final.
- **Posterior collapse.** Cuando el decoder es muy expresivo (p. ej. un decoder autorregresivo potente), el modelo puede ignorar el latente: $q_\phi(z\mid x)$ colapsa al prior $p(z)$ —la KL se anula— y el decoder reconstruye sin usar $z$. El latente deja de codificar información útil. Mitigaciones posteriores (KL annealing, $\beta$-VAE, free bits) atacan este fenómeno.
- **El bound es solo una cota.** Se optimiza el ELBO, no la log-verosimilitud directa; la brecha es la KL entre posterior aproximado y verdadero, que con un posterior gaussiano diagonal puede ser grande si el posterior real es multimodal o correlacionado. Familias variacionales más ricas (normalizing flows) reducen esta brecha.

## 8. Impacto

El VAE se convirtió en una de las dos columnas (junto a las GANs de Goodfellow et al., 2014) sobre las que se construyó el modelado generativo profundo. Es **la base de los modelos de variables latentes** profundos: prácticamente toda la línea posterior —VAEs jerárquicos, $\beta$-VAE (disentanglement), VQ-VAE (latentes discretos, base de DALL·E 1), CVAE (condicionales)— parte de la formulación de este paper. El reparameterization trick, además, trascendió el VAE: es la técnica estándar para retropropagar a través de cualquier muestreo de una distribución reparametrizable, usada en inferencia variacional amortizada, *stochastic* RL y modelos bayesianos profundos en general.

Pero el impacto más concreto para la Clase 29 es su rol en **Latent Diffusion / Stable Diffusion**. La difusión opera de forma costosísima en el espacio de píxeles. Rombach et al. (2022) movieron el proceso de difusión a un **espacio latente comprimido** producido precisamente por un **VAE** (un autoencoder con regularización KL o VQ): el encoder del VAE lleva una imagen de $512\times512\times3$ a un latente de, digamos, $64\times64\times4$; la difusión genera en ese latente, mucho más barato; y el decoder del VAE reconstruye la imagen final. El VAE de 2013 es, literalmente, el componente que hace tratable a Stable Diffusion. El defecto del VAE puro (reconstrucciones algo borrosas) deja de importar tanto porque su trabajo no es *generar* sino *comprimir/descomprimir* fielmente; la generación creativa la hace el modelo de difusión sobre el latente.

## 9. Conexión con la Clase 29 (Modelos Generativos en Visión)

La Clase 29 presenta los modelos generativos como una progresión, y el VAE ocupa una posición bisagra precisa:

- **Autoencoder → VAE como primera familia generativa.** La clase arranca con el autoencoder clásico: comprime $x$ a un código latente $z$ y reconstruye. Su limitación pedagógica es que el espacio latente no es muestreable —no sabemos de qué distribución sacar un $z$ nuevo— así que *no genera*. El VAE corrige esto con dos ingredientes que este paper aporta: (1) un **prior** $p(z) = \mathcal{N}(0, I)$ que da una distribución conocida desde la cual muestrear; y (2) un **encoder probabilístico** $q_\phi(z\mid x)$ regularizado vía KL para acercarse a ese prior. El resultado es un espacio latente **continuo y muestreable**, que permite tanto generar (muestrear del prior → decodear) como **interpolar** suavemente entre dos puntos —la propiedad que la clase ilustra con el manifold 2D de las Figuras 4–5 del paper.
- **El marco probabilístico es lo que añade la capacidad generativa.** La clase enfatiza que el salto autoencoder → VAE no es arquitectónico (ambos tienen encoder y decoder) sino *probabilístico*: el ELBO = reconstrucción − KL convierte un compresor en un modelo generativo principiado. El reparameterization trick es el detalle técnico que hace ese marco entrenable con backprop.
- **El VAE como compresor de Latent/Stable Diffusion.** Cuando la clase llega a los modelos de difusión, el VAE reaparece —ya no como generador final, sino como el **autoencoder que define el espacio latente** sobre el cual la difusión trabaja. Esto cierra el arco: el primer modelo generativo de la clase termina siendo un componente del último.

Enlaces internos: [/fundamentos/modelos-generativos](/fundamentos/modelos-generativos) · [/clases/clase-29](/clases/clase-29).
