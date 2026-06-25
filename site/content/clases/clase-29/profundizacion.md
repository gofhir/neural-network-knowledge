---
title: "Profundizacion - Modelos Generativos en Visión"
weight: 20
math: true
---

> Esta pagina complementa la [teoria de la clase 29](/clases/clase-29/teoria) con las derivaciones formales de las cuatro grandes familias generativas y de su metrica de evaluacion. **Parte I** deriva el ELBO del VAE, el reparameterization trick y la KL gaussiana en forma cerrada. **Parte II** desarrolla el objetivo minimax de las GAN, el discriminador optimo y la conexion con la divergencia de Jensen-Shannon. **Parte III** construye la difusion (DDPM): proceso forward, marginal cerrada, ELBO y su simplificacion a $L_\text{simple}$. **Parte IV** muestra que score matching y DDPM son la misma cosa via la dinamica de Langevin. **Parte V** lleva la difusion al espacio latente y deriva la cross-attention y la classifier-free guidance. **Parte VI** deriva la Frechet Inception Distance entre gaussianas.

---

## Parte I — VAE y el ELBO

### I.1 El problema de la verosimilitud marginal intratable

Postulamos un modelo de variable latente: un dato $x$ se genera muestreando primero un latente $z \sim p(z)$ de un *prior* y luego $x \sim p_\theta(x\mid z)$ de una verosimilitud condicional ([Kingma & Welling, 2013](/papers/vae-kingma-2013)). La cantidad que querriamos maximizar es la **verosimilitud marginal**:

$$
p_\theta(x) = \int p_\theta(x\mid z)\,p(z)\,dz
$$

Esa integral sobre todo el espacio latente es intratable apenas $p_\theta(x\mid z)$ es una red neuronal. Como consecuencia, el posterior verdadero $p_\theta(z\mid x) = p_\theta(x\mid z)\,p(z)/p_\theta(x)$ tambien es intratable: no podemos hacer EM ni inferencia exacta. La salida es introducir un **posterior aproximado** $q_\phi(z\mid x)$ —el *encoder*— y optimizar una cota inferior.

### I.2 Derivacion del ELBO

Partimos de la identidad exacta que se obtiene insertando $q_\phi$ y aplicando la definicion de KL. Para cualquier $q_\phi(z\mid x)$:

$$
\log p_\theta(x)
= \log \int p_\theta(x,z)\,dz
= \log \int q_\phi(z\mid x)\,\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\,dz
$$

Por la desigualdad de Jensen, como $\log$ es concava y $q_\phi$ es una densidad de probabilidad,

$$
\log p_\theta(x) \;\geq\; \mathbb{E}_{q_\phi(z\mid x)}\!\left[\log \frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right] \;=:\; \mathcal{L}(\theta,\phi;x)
$$

Esta cota inferior es el **ELBO** (*Evidence Lower BOund*). Expandiendo $p_\theta(x,z) = p_\theta(x\mid z)\,p(z)$ y separando el logaritmo del cociente:

$$
\mathcal{L}
= \mathbb{E}_{q_\phi(z\mid x)}\!\big[\log p_\theta(x\mid z)\big]
+ \mathbb{E}_{q_\phi(z\mid x)}\!\left[\log \frac{p(z)}{q_\phi(z\mid x)}\right]
$$

El segundo termino es, por definicion, el negativo de la KL del posterior aproximado contra el prior. Llegamos a la forma canonica:

$$
\boxed{\;\log p_\theta(x) \;\geq\; \mathcal{L} = \underbrace{\mathbb{E}_{q_\phi(z\mid x)}\!\big[\log p_\theta(x\mid z)\big]}_{\text{reconstruccion}} - \underbrace{D_{\mathrm{KL}}\!\big(q_\phi(z\mid x)\,\|\,p(z)\big)}_{\text{regularizacion}}\;}
$$

La **brecha** entre $\log p_\theta(x)$ y el ELBO es exactamente $D_{\mathrm{KL}}\big(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\big) \geq 0$. Maximizar el ELBO logra dos cosas a la vez: empuja la log-verosimilitud hacia arriba *y* aprieta el posterior aproximado contra el verdadero.

{{< concept-alert type="clave" >}}
El termino de regularizacion **emerge de la cota variacional**, no es un hiperparametro inventado como en los autoencoders regularizados (denoising, contractive, sparse). Esa es la diferencia conceptual entre un autoencoder y un VAE: el VAE es un modelo probabilistico principiado, no un compresor con un termino ad hoc.
{{< /concept-alert >}}

### I.3 El problema del gradiente y el reparameterization trick

Queremos diferenciar $\mathcal{L}$ respecto de $\theta$ **y** $\phi$. Respecto de $\theta$ no hay problema. Respecto de $\phi$ si: la esperanza esta tomada bajo $q_\phi(z\mid x)$, que *depende de $\phi$*, asi que no podemos intercambiar gradiente y esperanza. El estimador ingenuo tipo REINFORCE,

$$
\nabla_\phi\, \mathbb{E}_{q_\phi(z)}[f(z)] = \mathbb{E}_{q_\phi(z)}\big[f(z)\,\nabla_\phi \log q_\phi(z)\big],
$$

tiene varianza tan alta que es impractico. El **reparameterization trick** resuelve esto expresando $z$ como una transformacion determinista y diferenciable de una variable de ruido auxiliar $\epsilon$ cuya distribucion **no depende de $\phi$**:

$$
z = g_\phi(\epsilon, x), \qquad \epsilon \sim p(\epsilon)
$$

Para el caso gaussiano del VAE, con $q_\phi(z\mid x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x)\,I)$:

$$
\boxed{\; z = \mu + \sigma \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I) \;}
$$

donde $\odot$ es el producto elemento a elemento. Ahora la esperanza se reescribe como una esperanza sobre $p(\epsilon)$, que es fija:

$$
\mathbb{E}_{q_\phi(z\mid x)}[f(z)] = \mathbb{E}_{p(\epsilon)}\big[f(\mu + \sigma\odot\epsilon)\big]
$$

y el gradiente entra dentro de la esperanza sin problema. **Por que permite backprop:** el muestreo estocastico queda aislado en $\epsilon$, que es una hoja del grafo de computo sin parametros; toda la dependencia de $\phi$ vive dentro de la funcion determinista $g_\phi$, a traves de la cual el gradiente fluye sin obstaculos. El encoder ya no emite una muestra sino los **parametros** $\mu, \sigma$; el muestreo de $z$ se convierte en una operacion aritmetica diferenciable, y el VAE se entrena end-to-end con la misma backpropagation que cualquier red.

### I.4 La KL gaussiana en forma cerrada

Cuando prior y posterior aproximado son ambos gaussianos —$p(z) = \mathcal{N}(0, I)$ y $q_\phi(z\mid x) = \mathcal{N}(\mu, \sigma^2 I)$ diagonal en dimension $J$— la KL se integra analiticamente. Para una sola dimension $j$, usando $D_{\mathrm{KL}}\big(\mathcal{N}(\mu,\sigma^2)\,\|\,\mathcal{N}(0,1)\big) = \tfrac12\big(\mu^2 + \sigma^2 - \log\sigma^2 - 1\big)$, y sumando sobre las $J$ dimensiones independientes:

$$
D_{\mathrm{KL}}\!\big(q_\phi(z\mid x)\,\|\,p(z)\big)
= -\frac{1}{2}\sum_{j=1}^{J}\Big(1 + \log\big(\sigma_j^2\big) - \mu_j^2 - \sigma_j^2\Big)
$$

Esto es lo que hace el termino de regularizacion barato y de baja varianza: no se estima por Monte Carlo, se calcula de forma exacta. El ELBO completo para un dato queda:

$$
\mathcal{L}(\theta,\phi;x) \approx \frac{1}{2}\sum_{j=1}^{J}\Big(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\Big) + \frac{1}{L}\sum_{l=1}^{L}\log p_\theta\big(x\mid z^{(l)}\big)
$$

con $z^{(l)} = \mu + \sigma\odot\epsilon^{(l)}$. En la practica basta $L = 1$ muestra por dato si el minibatch es suficientemente grande.

### I.5 Por que el VAE genera imagenes borrosas

Para datos continuos el decoder es gaussiano: $p_\theta(x\mid z) = \mathcal{N}\big(x; \mu_\theta(z), \sigma^2 I\big)$. Su log-verosimilitud es

$$
\log p_\theta(x\mid z) = -\frac{1}{2\sigma^2}\,\lVert x - \mu_\theta(z)\rVert^2 + \text{const}
$$

es decir, **un error cuadratico medio** entre la imagen y su media reconstruida. El optimo de un MSE bajo incertidumbre es el **promedio condicional** $\mathbb{E}[x\mid z]$. Cuando un mismo $z$ es compatible con varios detalles finos plausibles (texturas, bordes), el decoder gaussiano no se compromete con ninguno: promedia sobre todos los modos, y ese promedio de detalles incompatibles es precisamente una imagen **borrosa**. Es una consecuencia matematica directa de la forma de la verosimilitud, no un defecto de implementacion. Esta es la razon de fondo por la cual las GAN (Parte II) y luego la difusion (Parte III), que no minimizan un MSE pixel a pixel, producen muestras mucho mas nitidas.

---

## Parte II — GAN y el objetivo minimax

### II.1 El juego minimax

Las [GAN](/papers/gan-goodfellow-2014) cambian de estrategia: en vez de modelar la densidad, aprenden a *generar muestras* enfrentando un generador $G$ (que mapea ruido $z\sim p_z$ a imagenes $G(z)$) contra un discriminador $D$ (que estima la probabilidad de que su entrada sea real). El objetivo es un juego de dos jugadores con funcion de valor:

$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_\text{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

$D$ trepa la funcion (quiere $D(x)\to 1$ en reales y $D(G(z))\to 0$ en falsos); $G$ la baja (quiere que $D(G(z))\to 1$, es decir, enganar a $D$). El equilibrio del juego es lo que produce el aprendizaje.

### II.2 El discriminador optimo

Fijemos $G$ y busquemos el $D$ que maximiza $V$. Reescribiendo el segundo termino como una integral sobre el espacio de datos via el cambio de variable $x = G(z)$ (que induce la densidad $p_g$):

$$
V(D, G) = \int_x \Big( p_\text{data}(x)\log D(x) + p_g(x)\log\big(1 - D(x)\big)\Big)\,dx
$$

Para cada $x$ fijo, el integrando tiene la forma $a\log y + b\log(1-y)$ con $a = p_\text{data}(x)$, $b = p_g(x)$, $y = D(x)$. Derivando respecto de $y$ e igualando a cero:

$$
\frac{a}{y} - \frac{b}{1-y} = 0 \;\Longrightarrow\; a(1-y) = by \;\Longrightarrow\; y = \frac{a}{a+b}
$$

Por tanto el **discriminador optimo** es:

$$
\boxed{\; D^*_G(x) = \frac{p_\text{data}(x)}{p_\text{data}(x) + p_g(x)} \;}
$$

Cuando $p_g = p_\text{data}$ en todas partes, $D^*_G(x) = \tfrac12$: el discriminador optimo no puede distinguir nada.

### II.3 El optimo global minimiza la divergencia de Jensen-Shannon

Sustituyendo $D^*_G$ de vuelta en $V$ obtenemos el criterio "virtual" $C(G) = \max_D V(D,G)$:

$$
C(G) = \mathbb{E}_{x\sim p_\text{data}}\!\left[\log \frac{p_\text{data}}{p_\text{data} + p_g}\right] + \mathbb{E}_{x\sim p_g}\!\left[\log \frac{p_g}{p_\text{data} + p_g}\right]
$$

El truco algebraico es sumar y restar $\log 2$ dentro de cada esperanza, equivalente a restar $\mathbb{E}_{p_\text{data}}[-\log 2] + \mathbb{E}_{p_g}[-\log 2] = -\log 4$:

$$
C(G) = -\log 4 + D_{\mathrm{KL}}\!\left(p_\text{data}\,\Big\|\,\frac{p_\text{data} + p_g}{2}\right) + D_{\mathrm{KL}}\!\left(p_g\,\Big\|\,\frac{p_\text{data} + p_g}{2}\right)
$$

La suma de esas dos KL respecto del promedio $M = (p_\text{data}+p_g)/2$ es, por definicion, dos veces la **divergencia de Jensen-Shannon**:

$$
\boxed{\; C(G) = -\log 4 + 2\cdot \mathrm{JSD}(p_\text{data}\,\|\,p_g) \;}
$$

Como $\mathrm{JSD} \geq 0$ y vale cero solo cuando $p_g = p_\text{data}$, el **minimo global** es $C^* = -\log 4$, alcanzado unicamente cuando el generador replica perfectamente la distribucion de los datos. Entrenar una GAN equivale, en el limite ideal, a minimizar la JSD entre la distribucion del modelo y la de los datos.

{{< concept-alert type="recordar" >}}
Que la JSD se *sature* (se vuelva constante en $\log 2$) cuando los soportes de $p_\text{data}$ y $p_g$ no se solapan es la semilla de la inestabilidad del entrenamiento GAN: cuando el generador esta lejos, el gradiente de la JSD se desvanece. Es exactamente lo que Wasserstein GAN ataco reemplazando la JSD por la distancia de Wasserstein.
{{< /concept-alert >}}

### II.4 El problema de gradientes y el truco non-saturating

El generador se entrenaria, segun la funcion de valor original, minimizando $\log(1 - D(G(z)))$. El problema aparece **temprano** en el entrenamiento: cuando $G$ es pobre, $D$ rechaza sus muestras con altisima confianza, $D(G(z)) \approx 0$, y entonces

$$
\frac{\partial}{\partial \theta_g}\log\big(1 - D(G(z))\big) = \frac{-D'(G(z))}{1 - D(G(z))} \approx 0
$$

el gradiente **se satura** justo cuando $G$ mas necesita aprender. La solucion practica es entrenar $G$ para **maximizar** $\log D(G(z))$ en lugar de minimizar $\log(1 - D(G(z)))$:

$$
\mathcal{L}_G^\text{ns} = -\mathbb{E}_{z\sim p_z}\big[\log D(G(z))\big]
$$

Ambas formas tienen el **mismo punto fijo** de la dinamica (el mismo equilibrio), pero la version *non-saturating* provee gradientes fuertes cuando $D(G(z))\approx 0$: $\frac{\partial}{\partial\theta_g}\log D(G(z)) = D'(G(z))/D(G(z))$ se dispara cuando el denominador es pequeno. Casi todas las implementaciones reales usan esta variante; sin ella, las GAN simplemente no entrenan en la practica.

---

## Parte III — Difusion (DDPM)

### III.1 El proceso forward

Los [modelos de difusion](/papers/ddpm-ho-2020) definen un **proceso forward** fijo (sin parametros) que destruye gradualmente la senal anadiendo ruido gaussiano segun una *variance schedule* $\beta_1,\dots,\beta_T$:

$$
q(x_t\mid x_{t-1}) = \mathcal{N}\big(x_t;\; \sqrt{1-\beta_t}\,x_{t-1},\; \beta_t I\big)
$$

Cada paso encoge el estado anterior por $\sqrt{1-\beta_t}$ y le suma ruido de varianza $\beta_t$. Tras $T$ pasos, $x_T$ es practicamente ruido gaussiano puro $\mathcal{N}(0, I)$.

### III.2 La marginal en forma cerrada

La propiedad que hace tratable el entrenamiento es que podemos saltar a un paso $t$ arbitrario **sin recorrer la cadena**. Definiendo $\alpha_t := 1-\beta_t$ y $\bar\alpha_t := \prod_{s=1}^{t}\alpha_s$, y componiendo dos pasos gaussianos (la suma de dos gaussianas independientes es gaussiana con varianzas que se suman):

$$
x_t = \sqrt{\alpha_t}\,x_{t-1} + \sqrt{1-\alpha_t}\,\epsilon_{t-1}
= \sqrt{\alpha_t\alpha_{t-1}}\,x_{t-2} + \sqrt{1-\alpha_t\alpha_{t-1}}\,\bar\epsilon + \dots
$$

Iterando hasta $x_0$, los terminos de ruido se combinan en uno solo y se obtiene la **marginal cerrada**:

$$
\boxed{\; q(x_t\mid x_0) = \mathcal{N}\big(x_t;\; \sqrt{\bar\alpha_t}\,x_0,\; (1-\bar\alpha_t)\,I\big) \;}
\quad\Longleftrightarrow\quad
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\;\; \epsilon\sim\mathcal{N}(0,I)
$$

El factor $\bar\alpha_t$ interpola: con $\bar\alpha_t\to 1$ (t pequeno) $x_t\approx x_0$; con $\bar\alpha_t\to 0$ (t grande) $x_t$ es casi ruido puro. Esta formula permite, en entrenamiento, muestrear un $t$ uniforme y saltar directamente a $x_t$ desde $x_0$ —la diferencia entre un entrenamiento factible y uno prohibitivo.

### III.3 La posterior tratable y el ELBO de difusion

El proceso reverse es una cadena de Markov aprendida $p_\theta(x_{t-1}\mid x_t) = \mathcal{N}\big(x_{t-1}; \mu_\theta(x_t,t), \Sigma_\theta\big)$ que arranca de $p(x_T)=\mathcal{N}(0,I)$. El blanco que intenta imitar es la **posterior del forward condicionada en $x_0$**, que por Bayes tambien es gaussiana y tratable:

$$
q(x_{t-1}\mid x_t, x_0) = \mathcal{N}\big(x_{t-1}; \tilde\mu_t(x_t,x_0), \tilde\beta_t I\big),
\qquad
\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t
$$

El entrenamiento maximiza el ELBO de la log-verosimilitud, que tras Rao-Blackwellizacion se descompone en una suma de KL **entre gaussianas** (calculables en forma cerrada):

$$
\mathbb{E}_q\Big[\underbrace{D_{\mathrm{KL}}\big(q(x_T\mid x_0)\,\|\,p(x_T)\big)}_{L_T} + \sum_{t>1}\underbrace{D_{\mathrm{KL}}\big(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\big)}_{L_{t-1}} - \underbrace{\log p_\theta(x_0\mid x_1)}_{L_0}\Big]
$$

$L_T$ es constante (el forward no tiene parametros). El grueso del aprendizaje vive en los $L_{t-1}$, que con varianza fija $\Sigma_\theta = \sigma_t^2 I$ se reducen a un error cuadratico entre medias:

$$
L_{t-1} = \mathbb{E}_q\!\left[\frac{1}{2\sigma_t^2}\,\lVert \tilde\mu_t(x_t,x_0) - \mu_\theta(x_t,t)\rVert^2\right] + C
$$

### III.4 La reparametrizacion $\epsilon$ y $L_\text{simple}$

El paso astuto de DDPM: en vez de predecir $\tilde\mu_t$ directamente, se reparametriza $x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\epsilon)/\sqrt{\bar\alpha_t}$ de la marginal cerrada y se sustituye en $\tilde\mu_t$. Tras el algebra, la media optima se expresa en terminos de $x_t$ y del ruido $\epsilon$, de modo que basta con que la **red prediga el ruido** $\epsilon_\theta(x_t,t)$:

$$
\mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t)\Big)
$$

Sustituyendo en $L_{t-1}$, el termino del ELBO colapsa a un error cuadratico ponderado entre el ruido real y el predicho. DDPM descubre que **descartar el peso** $\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}$ (fijarlo a 1) mejora la calidad de muestra y simplifica todo:

$$
\boxed{\; L_\text{simple}(\theta) = \mathbb{E}_{t,x_0,\epsilon}\Big[\big\lVert \epsilon - \epsilon_\theta\big(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\; t\big)\big\rVert^2\Big] \;}
$$

con $t\sim\text{Uniform}(\{1,\dots,T\})$. Una simple perdida MSE: "adivina que ruido le eche a esta imagen". El reweighting implicito infrapondera los $t$ pequenos (denoising trivial) y concentra la capacidad en los $t$ grandes (denoising dificil).

### III.5 El sampling reverse

Generar una imagen es partir de $x_T\sim\mathcal{N}(0,I)$ e iterar de $T$ a $1$ aplicando la media reconstruida mas una pizca de ruido fresco:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t)\Big) + \sigma_t z,
\qquad z\sim\mathcal{N}(0,I)\;\;(z=0 \text{ si } t=1)
$$

Cada paso resta una fraccion del ruido predicho y reinyecta ruido —un proceso que, como veremos en la Parte IV, es exactamente dinamica de Langevin con $\epsilon_\theta$ como gradiente aprendido de la densidad.

---

## Parte IV — Score matching y Langevin

### IV.1 El score: el gradiente del log-densidad

La perspectiva [score-based](/papers/score-based-song-2019) cambia el objeto que se modela. En vez de aprender la densidad $p(x)$, se aprende el **score**:

$$
s(x) = \nabla_x \log p(x)
$$

un campo vectorial que en cada punto apunta hacia donde la log-densidad crece mas rapido. Su ventaja decisiva: **esquiva la constante de normalizacion**. Para $p(x) = \tilde p(x)/Z$,

$$
\nabla_x \log p(x) = \nabla_x \log \tilde p(x) - \underbrace{\nabla_x \log Z}_{=\,0} = \nabla_x \log \tilde p(x)
$$

porque $Z$ no depende de $x$. La red $s_\theta(x)$ no necesita parametrizar una densidad normalizada; basta con que produzca un campo vectorial $\mathbb{R}^D \to \mathbb{R}^D$.

### IV.2 La conexion score ↔ ε

El *denoising score matching* entrena la red para estimar el score de la distribucion **perturbada** por un kernel gaussiano $q_\sigma(\tilde x\mid x) = \mathcal{N}(\tilde x; x, \sigma^2 I)$. El score de ese kernel tiene forma cerrada:

$$
\nabla_{\tilde x}\log q_\sigma(\tilde x\mid x) = -\frac{\tilde x - x}{\sigma^2} = -\frac{\epsilon}{\sigma}
$$

donde $\tilde x = x + \sigma\epsilon$ con $\epsilon\sim\mathcal{N}(0,I)$. Comparando con la parametrizacion $\epsilon$ de DDPM (Parte III), vemos que predecir el ruido y predecir el score son **el mismo objetivo hasta un factor de escala**:

$$
\boxed{\; s_\theta(x) = -\frac{\epsilon_\theta(x)}{\sigma} \;}
$$

La red $\epsilon_\theta$ de DDPM y la red $s_\theta$ de NCSN aprenden la misma informacion. Esta equivalencia es la que Song et al. (2021) formalizaron mostrando que DDPM (la SDE *variance preserving*) y score-based (la SDE *variance exploding*) son **dos discretizaciones del mismo proceso continuo** descrito por una ecuacion diferencial estocastica.

### IV.3 Dinamica de Langevin recocida

Conociendo el score, se puede generar con la **dinamica de Langevin**, un MCMC que solo necesita el gradiente del log-densidad. Dado un paso $\eta > 0$:

$$
\tilde x_{k} = \tilde x_{k-1} + \frac{\eta}{2}\,\nabla_x \log p(\tilde x_{k-1}) + \sqrt{\eta}\,z_k,
\qquad z_k\sim\mathcal{N}(0,I)
$$

Es un descenso de gradiente sobre la log-densidad con ruido inyectado que evita colapsar al modo; cuando $\eta\to 0$ y $K\to\infty$, $\tilde x_K$ es una muestra exacta de $p(x)$. El score ingenuo falla en dos puntos: en variedades de baja dimension esta indefinido, y en regiones de baja densidad no hay datos para estimarlo. La solucion es la **Langevin recocida** (*annealed*): se construye una secuencia de niveles de ruido $\sigma_1 > \sigma_2 > \cdots > \sigma_L$, se entrena una sola red $s_\theta(x,\sigma)$ condicionada en $\sigma$, y se muestrea empezando en el ruido alto $\sigma_1$ (donde el campo es suave y bien definido) usando las muestras finales de cada nivel como inicializacion del siguiente, bajando gradualmente hasta $\sigma_L$. Es *simulated annealing* aplicado al muestreo. El recorrido $\sigma_1\to\sigma_L$ es exactamente la trayectoria temporal $t=T\to 0$ del reverse de DDPM: **son dos vistas del mismo proceso de invertir el ruido paso a paso.**

---

## Parte V — Latent Diffusion y cross-attention

### V.1 Difusion en el espacio latente

La [difusion latente](/papers/latent-diffusion-rombach-2022) ataca el costo de operar en pixeles. Primero se entrena (una sola vez, por separado) un autoencoder: un encoder $\mathcal{E}$ lleva la imagen $x\in\mathbb{R}^{H\times W\times 3}$ a un latente $z = \mathcal{E}(x)\in\mathbb{R}^{h\times w\times c}$ con factor de downsampling $f = H/h$, y un decoder reconstruye $\tilde x = \mathcal{D}(z)$. Ese autoencoder es esencialmente un VAE (con regularizacion KL leve) entrenado con perdida perceptual (LPIPS) y un objetivo adversarial PatchGAN para evitar el desenfoque del MSE puro (Parte I.5). El modelo de difusion se entrena entonces **en el latente**, no en pixeles:

$$
L_\text{LDM} = \mathbb{E}_{\mathcal{E}(x),\,\epsilon\sim\mathcal{N}(0,I),\,t}\Big[\big\lVert \epsilon - \epsilon_\theta(z_t, t)\big\rVert^2\Big]
$$

con $z_t = \sqrt{\bar\alpha_t}\,\mathcal{E}(x) + \sqrt{1-\bar\alpha_t}\,\epsilon$. Es el mismo $L_\text{simple}$ de DDPM (Parte III.4) con un cambio de espacio. Como el latente ya abstrajo los detalles imperceptibles de alta frecuencia, el modelo deja de gastar capacidad en la fase de **compresion perceptual** y se concentra en la **compresion semantica**, lo que reduce drasticamente el costo de entrenamiento e inferencia. El punto dulce empirico es $f\in\{4,8\}$.

### V.2 Condicionamiento por cross-attention

Para modelar la distribucion condicional $p(z\mid y)$ —con $y$ texto, clases o layouts— se aumenta la U-Net con capas de **cross-attention**. Un encoder de dominio $\tau_\theta$ proyecta la condicion $y$ (por ejemplo un prompt) a una representacion $\tau_\theta(y)\in\mathbb{R}^{M\times d_\tau}$ (en Stable Diffusion, el encoder de texto de CLIP). Esa representacion se inyecta en las capas intermedias de la U-Net:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V
$$

donde —y esta es la clave— las *queries* vienen del estado interno de la U-Net y las *keys/values* del condicionamiento:

$$
Q = W_Q\,\varphi_i(z_t), \qquad K = W_K\,\tau_\theta(y), \qquad V = W_V\,\tau_\theta(y)
$$

El producto $QK^\top$ mide, para cada posicion espacial del latente (cada query), cuanto atiende a cada token del texto (cada key); el softmax convierte eso en pesos que mezclan los *values* (la informacion textual). Asi cada region de la imagen en construccion "lee" las partes del prompt que le conciernen. La elegancia es que $\tau_\theta$ puede ser cualquier experto de dominio: cambiar la modalidad de la condicion es cambiar el encoder, no la arquitectura de difusion.

### V.3 Classifier-free guidance

Sin clasificador externo, la [classifier-free guidance](/papers/classifier-free-guidance-ho-2022) controla el trade-off fidelidad/diversidad. Se entrena **una sola red** que aprende simultaneamente el modelo condicional $\epsilon_\theta(z_t, c)$ y el no-condicional $\epsilon_\theta(z_t) = \epsilon_\theta(z_t, \varnothing)$, simplemente haciendo *dropout* del condicionamiento (poniendo $c\leftarrow\varnothing$, un token nulo) con probabilidad $p_\text{uncond}\approx 0.1$. En sampling se combina linealmente:

$$
\boxed{\; \tilde\epsilon = \epsilon_\text{uncond} + w\,(\epsilon_\text{cond} - \epsilon_\text{uncond}) \;}
$$

(equivale a $\tilde\epsilon = (1+w')\epsilon_\text{cond} - w'\,\epsilon_\text{uncond}$ con $w = w'+1$; el `guidance_scale` de `diffusers` es este $w$). La direccion $(\epsilon_\text{cond} - \epsilon_\text{uncond})$ es **lo que el texto aporta** sobre la generacion a ciegas; multiplicarla por $w$ la **extrapola**, empujando la muestra hacia donde la condicion $c$ "importa". Subir $w$ aumenta la fidelidad al prompt a costa de la diversidad (y, si es muy alto, satura los colores). Notar que $\tilde\epsilon$ no contiene gradiente de clasificador alguno: es guia puramente generativa.

{{< concept-alert type="recordar" >}}
Cada paso de sampling con guia evalua la U-Net **dos veces** (condicional + no-condicional). Por eso generar con guia cuesta el doble que sin guia. El *negative prompt* de Stable Diffusion es una generalizacion: en lugar de restar el score del token nulo, se resta el score de un prompt "negativo", empujando la imagen *lejos* de ese concepto.
{{< /concept-alert >}}

---

## Parte VI — Evaluacion: la Frechet Inception Distance

### VI.1 De los momentos a la distancia entre gaussianas

Para evaluar un modelo generativo de imagenes ([Heusel et al., 2017](/papers/fid-heusel-2017)) querriamos medir una distancia entre la distribucion real $p_\text{data}$ y la generada $p_g$. El defecto del Inception Score previo era que **nunca comparaba contra los datos reales**: solo auditaba la confianza de clasificacion y la variedad de lo generado. La FID corrige esto.

La construccion parte de un principio de teoria de la medida: dos distribuciones quedan caracterizadas por sus **momentos**. La FID hace dos elecciones. Primero, en vez de momentos de los pixeles crudos, usa los *features* de la **ultima capa de pooling de Inception-v3** (un vector de 2048 dimensiones), que captura semantica de vision. Segundo, conserva solo los **dos primeros momentos** (media y covarianza) y asume que esos features siguen una gaussiana multidimensional —justificado porque la gaussiana es la distribucion de **maxima entropia** dados una media y una covarianza.

### VI.2 La formula

Bajo el supuesto gaussiano, la distancia entre las dos distribuciones de features es la **distancia de Frechet** (equivalente a la **Wasserstein-2**) entre $\mathcal{N}(\mu_r, \Sigma_r)$ (real) y $\mathcal{N}(\mu_g, \Sigma_g)$ (generada), que admite forma cerrada:

$$
\boxed{\; \mathrm{FID} = \lVert \mu_r - \mu_g \rVert_2^2 + \mathrm{Tr}\!\Big(\Sigma_r + \Sigma_g - 2\big(\Sigma_r \Sigma_g\big)^{1/2}\Big) \;}
$$

### VI.3 Que captura cada termino

El **primer termino** $\lVert \mu_r - \mu_g\rVert^2$ penaliza diferencias en el centro de la distribucion de features: mide si el modelo genera, en promedio, el "tipo" de imagen correcto. El **segundo termino** compara la estructura de covarianza —la forma y la diversidad de la distribucion. El termino $(\Sigma_r\Sigma_g)^{1/2}$ es la raiz matricial del producto de covarianzas; la traza completa se anula solo cuando $\Sigma_r = \Sigma_g$.

Esto le da a la FID tres propiedades:

| Propiedad | Mecanismo |
| --- | --- |
| **Detecta mode collapse** | Si $G$ colapsa a pocos modos, $\Sigma_g \ll \Sigma_r$ (poca diversidad) y el termino de traza dispara la FID, aunque el IS quede alto. |
| **Mejora con el realismo** | A mas realismo y diversidad, los features generados se acercan a los reales y la FID baja monotonamente con la calidad visual. |
| **Detecta overfitting** | Si la FID diverge o sube durante el entrenamiento, es senal de inestabilidad o sobreajuste. |

**Menor FID = mejor**; FID $= 0$ significa distribuciones de features identicas, el ideal inalcanzable. El protocolo canonico usa 50.000 imagenes generadas (la covarianza estimada con pocas muestras esta sesgada, asi que comparar FID con distinto $N$ es invalido). La FID se convirtio en la metrica estandar de toda tabla de GANs y modelos de difusion posterior a 2017, y es la vara comun con la que la Clase 29 ordena el progreso de cada familia generativa.

---

## Sintesis: cuatro familias, un eje de medida

| Familia | Objeto que aprende | Objetivo | Muestreo | Trade-off |
| --- | --- | --- | --- | --- |
| **VAE** | densidad via $q_\phi(z\mid x)$ | ELBO (recon. − KL) | 1 paso (decoder) | rapido, alta cobertura, **borroso** |
| **GAN** | generador implicito | minimax $\to$ JSD | 1 paso (generador) | rapido, **nitido**, mode collapse |
| **Difusion** | ruido $\epsilon_\theta$ / score $s_\theta$ | $L_\text{simple}$ (MSE) | $T$ pasos (reverse) | **lento**, nitido y alta cobertura |
| **Latent Diffusion** | $\epsilon_\theta$ en $z=\mathcal{E}(x)$ | $L_\text{LDM}$ + cross-attn | $T$ pasos (en latente) | eficiente, controlable (CFG) |

La difusion gano porque combina la **alta calidad** de las GAN con la **alta cobertura** de los modelos de verosimilitud (sin mode collapse), pagando solo el precio del muestreo lento —precio que Latent Diffusion mitiga moviendose al latente. La **FID** es el instrumento transversal que hace cuantificable esa narrativa de progreso.

---

**Enlaces internos:**

- Fundamentos: [Modelos Generativos](/fundamentos/modelos-generativos) · [Modelos de difusion](/fundamentos/modelos-de-difusion)
- Papers: [VAE (Kingma & Welling, 2013)](/papers/vae-kingma-2013) · [GAN (Goodfellow et al., 2014)](/papers/gan-goodfellow-2014) · [DDPM (Ho et al., 2020)](/papers/ddpm-ho-2020) · [Score-based (Song & Ermon, 2019)](/papers/score-based-song-2019) · [Latent Diffusion (Rombach et al., 2022)](/papers/latent-diffusion-rombach-2022) · [Classifier-Free Guidance (Ho & Salimans, 2022)](/papers/classifier-free-guidance-ho-2022) · [FID (Heusel et al., 2017)](/papers/fid-heusel-2017)
- Clase: [Clase 29 — Modelos Generativos en Visión](/clases/clase-29) · [Teoria](/clases/clase-29/teoria)
