---
title: "VAE: Auto-Encoding Variational Bayes (2013)"
weight: 328
math: true
---

{{< paper-card
    title="Auto-Encoding Variational Bayes"
    authors="Diederik P. Kingma, Max Welling"
    year="2013"
    venue="ICLR 2014"
    pdf="/papers/vae-kingma-2013.pdf"
    arxiv="1312.6114" >}}
El paper que introdujo el **Variational Autoencoder (VAE)**, uno de los dos pilares del modelado generativo profundo junto a las [GANs](/papers/gan-goodfellow-2014). Reformula el autoencoder como un **modelo generativo probabilístico**: un encoder $q_\phi(z\mid x)$ que mapea cada dato a una *distribución* sobre el código latente, y un decoder $p_\theta(x\mid z)$ que reconstruye desde ese código. El truco técnico que lo hace entrenable de punta a punta con backpropagation es el **reparameterization trick**. Su objetivo es el **ELBO = reconstrucción − KL**. Hoy el VAE es el componente que comprime al espacio latente sobre el cual opera [Stable Diffusion](/papers/latent-diffusion-rombach-2022).
{{< /paper-card >}}

---

## Contexto

El problema que ataca el paper parece muy abstracto: cómo hacer inferencia y aprendizaje eficientes en modelos dirigidos con **variables latentes continuas** cuyo posterior es intratable, y con datasets grandes. La respuesta resultó ser uno de los modelos generativos más influyentes de la década.

El planteamiento bayesiano es el siguiente. Se asume que cada dato se genera en dos pasos: primero se muestrea una variable latente $z \sim p_\theta(z)$ desde un *prior*; luego se genera $x \sim p_\theta(x\mid z)$ desde una *verosimilitud condicional*. El problema es que casi todo está oculto. La **verosimilitud marginal** $p_\theta(x) = \int p_\theta(z)\,p_\theta(x\mid z)\,dz$ es intratable porque la integral sobre todas las latentes no se puede evaluar; en consecuencia el **posterior verdadero** $p_\theta(z\mid x)$ también lo es, lo que descarta el algoritmo EM. Y como los datasets son grandes, los métodos de muestreo tipo Monte Carlo EM serían demasiado lentos. Estas intratabilidades aparecen apenas la verosimilitud $p_\theta(x\mid z)$ es moderadamente complicada, por ejemplo una red neuronal con una capa oculta no lineal.

El paper introduce tres niveles de contribución encajados, con tres siglas: el **SGVB** (Stochastic Gradient Variational Bayes, el estimador diferenciable y de baja varianza del lower bound), el **AEVB** (Auto-Encoding Variational Bayes, el algoritmo de entrenamiento por minibatch), y el **VAE** (Variational Autoencoder, el caso particular en que encoder y decoder son redes neuronales). La comunidad de visión adoptó sobre todo el tercero.

## De autoencoder clásico a autoencoder probabilístico

Para entender el aporte conviene contrastar con el autoencoder clásico. Un autoencoder tradicional mapea $x$ a un código *determinista* y lo reconstruye minimizando un error de reconstrucción. Puede comprimir, pero **no puede generar** de forma principiada: su espacio latente no tiene una distribución conocida de la cual muestrear un código nuevo. Además, "es bien sabido que este criterio de reconstrucción en sí mismo no basta para aprender representaciones útiles", de ahí que las variantes regularizadas (denoising, contractive, sparse) añadan términos ad hoc con hiperparámetros inventados.

El VAE introduce un **modelo de reconocimiento** $q_\phi(z\mid x)$ —una aproximación al posterior intratable— y reinterpreta ambas piezas desde la teoría de codificación, donde las latentes $z$ son un *código*:

- $q_\phi(z\mid x)$ es el **encoder probabilístico**: dado $x$, produce una *distribución* (gaussiana) sobre los códigos $z$ de los que $x$ podría haberse generado.
- $p_\theta(x\mid z)$ es el **decoder probabilístico**: dado un código $z$, produce una distribución sobre los $x$ correspondientes.

Los parámetros del encoder $\phi$ y del decoder $\theta$ se aprenden **conjuntamente**. La diferencia clave frente al autoencoder regularizado: su término de regularización **lo dicta la cota variacional**, no un hiperparámetro a mano. El autoencoder se vuelve, por construcción, un modelo generativo del cual se puede muestrear.

## El ELBO

El corazón matemático es el **ELBO** (Evidence Lower Bound). La log-verosimilitud de un dato se descompone exactamente como:

$$\log p_\theta(x) = D_{\mathrm{KL}}\big(q_\phi(z\mid x) \,\|\, p_\theta(z\mid x)\big) + \mathcal{L}(\theta, \phi; x)$$

Como la divergencia KL es **no negativa**, el término $\mathcal{L}$ es una *cota inferior* sobre la log-verosimilitud. Reescrito en la forma que da la lectura "autoencoder":

$$\boxed{\;\text{ELBO} = \underbrace{\mathbb{E}_{q_\phi(z\mid x)}\big[\log p_\theta(x\mid z)\big]}_{\text{reconstrucción}} - \underbrace{D_{\mathrm{KL}}\big(q_\phi(z\mid x) \,\|\, p_\theta(z)\big)}_{\text{regularización}}\;}$$

El primer término es el **error de reconstrucción negativo esperado**: cuán bien el decoder reconstruye $x$ desde un $z$ muestreado del encoder. El segundo es la **KL del posterior aproximado respecto del prior**, que empuja a $q_\phi(z\mid x)$ a parecerse al prior $p(z)$. Maximizar el ELBO logra dos cosas a la vez: sube la log-verosimilitud *y* aprieta la brecha entre posterior aproximado y verdadero, porque esa brecha es exactamente la KL del primer término de la descomposición.

## El reparameterization trick

Hay un obstáculo concreto al optimizar el ELBO respecto de $\phi$: la esperanza se toma bajo $q_\phi(z\mid x)$, que *depende de $\phi$*, de modo que no se puede intercambiar gradiente y esperanza. El estimador ingenuo de tipo *score function* (REINFORCE) "exhibe muy alta varianza y es impráctico". 

La solución es el **reparameterization trick**. En vez de muestrear $z \sim q_\phi(z\mid x)$ directamente —operación estocástica, no diferenciable respecto de $\phi$—, se expresa $z$ como una **transformación determinista y diferenciable** de una variable de ruido auxiliar $\epsilon$ cuya distribución no depende de $\phi$:

$$z = g_\phi(\epsilon, x), \qquad \epsilon \sim p(\epsilon)$$

El caso gaussiano es el canónico y el que usa el VAE. Si $z \sim \mathcal{N}(\mu, \sigma^2)$:

$$\boxed{\; z = \mu + \sigma \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I) \;}$$

donde $\odot$ es el producto elemento a elemento. El encoder ya no emite una muestra: emite los **parámetros** $\mu$ y $\sigma$ de la gaussiana; el ruido $\epsilon$ se inyecta aparte, y muestrear $z$ pasa a ser una operación aritmética diferenciable. Así el gradiente del ELBO fluye desde la pérdida de reconstrucción, a través de $z$, hasta los pesos del encoder — habilitando el entrenamiento *end-to-end* con backpropagation estándar.

## El VAE concreto

En el VAE el prior es la gaussiana isotrópica $p_\theta(z) = \mathcal{N}(z; 0, I)$ —que notablemente *no tiene parámetros*— y el posterior aproximado se elige gaussiano con covarianza diagonal, cuya media $\mu$ y desviación $\sigma$ son salidas de un MLP encoder. Cuando prior y posterior son ambos gaussianos, la **KL se integra analíticamente**, dando un estimador cerrado:

$$\mathcal{L}(\theta,\phi;x) \approx \tfrac{1}{2}\sum_{j=1}^{J}\Big(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\Big) + \frac{1}{L}\sum_{l=1}^{L}\log p_\theta\big(x\mid z^{(l)}\big)$$

con $z^{(l)} = \mu + \sigma \odot \epsilon^{(l)}$. El primer sumando es la KL en forma cerrada; el segundo, la reconstrucción. Para datos binarios (MNIST) el decoder es Bernoulli y la reconstrucción es exactamente la **cross-entropy binaria**; para datos continuos el decoder es gaussiano. Un hallazgo práctico: basta con **$L = 1$** muestra de ruido por dato si el minibatch es suficientemente grande ($M = 100$), lo que hace el VAE tan barato como un autoencoder normal.

## Experimentos

El paper entrena modelos generativos de imágenes sobre **MNIST** y **Frey Faces** (rostros continuos), comparando algoritmos por el lower bound variacional y la verosimilitud marginal estimada.

- **Convergencia.** AEVB se compara contra el algoritmo **wake-sleep** (Hinton et al., 1995) para distintas dimensionalidades del latente $N_z \in \{3, 5, 10, 20, 200\}$. AEVB **converge mucho más rápido y alcanza mejor solución en todos los experimentos**.
- **Más latentes no produce overfitting.** Un resultado contraintuitivo: añadir variables latentes superfluas no daña el desempeño, gracias al **efecto regularizador del término KL**, que penaliza usar capacidad latente innecesaria.
- **Verosimilitud marginal.** Frente a Monte Carlo EM (que no es online y no escala al MNIST completo), AEVB converge rápido en regímenes de datos chico ($N=1000$) y grande ($N=50000$).
- **Manifold latente continuo.** Con un latente 2D, recorriendo coordenadas linealmente espaciadas y pasándolas por el decoder, se obtienen manifolds suaves donde los dígitos y rostros varían **continuamente**. Es la demostración visual de que el espacio latente del VAE es continuo e interpolable, justo la propiedad que lo distingue del autoencoder clásico.

## Limitaciones

El paper original es modesto en autocrítica, pero la comunidad identificó después tres limitaciones centrales:

- **Imágenes borrosas.** El decoder gaussiano equivale a un error cuadrático medio entre la imagen y su media reconstruida; frente a la incertidumbre, *promedia* sobre modos plausibles en vez de comprometerse con detalle nítido, produciendo el característico aspecto **difuminado** de las muestras del VAE. Es consecuencia directa de la forma de la verosimilitud. Por eso las GANs y luego la difusión desplazaron al VAE puro como generador final en visión.
- **Posterior collapse.** Cuando el decoder es muy expresivo (p. ej. autorregresivo potente), el modelo puede ignorar el latente: $q_\phi(z\mid x)$ colapsa al prior, la KL se anula y $z$ deja de codificar información útil. Mitigaciones posteriores: KL annealing, $\beta$-VAE, free bits.
- **El bound es solo una cota.** Se optimiza el ELBO, no la log-verosimilitud directa; la brecha es la KL entre posterior aproximado y verdadero, que con un posterior gaussiano diagonal puede ser grande si el real es multimodal. Familias variacionales más ricas (normalizing flows) reducen esta brecha.

## Impacto

El VAE se volvió una de las dos columnas del modelado generativo profundo, junto a las [GANs](/papers/gan-goodfellow-2014). Es la **base de los modelos de variables latentes** profundos: VAEs jerárquicos, $\beta$-VAE (disentanglement), [VQ-VAE](/papers/vq-vae-oord-2017) (latentes discretos, base de DALL·E 1) y CVAE (condicionales) parten todos de esta formulación. El **reparameterization trick** trascendió al VAE y es hoy la técnica estándar para retropropagar a través de cualquier muestreo reparametrizable, usada en inferencia variacional amortizada, RL estocástico y modelos bayesianos profundos.

Pero el impacto más concreto es su rol en **Latent Diffusion / Stable Diffusion**. La difusión es costosísima en el espacio de píxeles. [Rombach et al. (2022)](/papers/latent-diffusion-rombach-2022) movieron el proceso a un **espacio latente comprimido** producido precisamente por un **VAE** (con regularización KL o VQ): el encoder lleva una imagen de $512\times512\times3$ a un latente de, digamos, $64\times64\times4$; la difusión genera en ese latente mucho más barato; y el decoder reconstruye la imagen final. El VAE de 2013 es, literalmente, el componente que hace tratable a Stable Diffusion. Y su defecto —reconstrucciones algo borrosas— deja de importar tanto, porque su trabajo no es *generar* sino *comprimir/descomprimir* fielmente; la generación creativa la hace la difusión sobre el latente.

## Por qué importa para la Clase 29

La [Clase 29](/clases/clase-29) presenta los [modelos generativos](/fundamentos/modelos-generativos) como una progresión, y el VAE ocupa una posición bisagra precisa:

- **Autoencoder → VAE como primera familia generativa.** La clase arranca con el autoencoder clásico, que comprime pero no genera porque su latente no es muestreable. El VAE corrige esto con dos ingredientes de este paper: un **prior** $p(z) = \mathcal{N}(0, I)$ que da una distribución conocida desde la cual muestrear, y un **encoder probabilístico** regularizado vía KL para acercarse a ese prior. El resultado es un espacio latente **continuo y muestreable** que permite generar (muestrear del prior → decodear) e **interpolar** suavemente.
- **El marco probabilístico añade la capacidad generativa.** El salto autoencoder → VAE no es arquitectónico (ambos tienen encoder y decoder) sino *probabilístico*: el ELBO = reconstrucción − KL convierte un compresor en un modelo generativo principiado, y el reparameterization trick lo hace entrenable.
- **El VAE como compresor de Stable Diffusion.** Cuando la clase llega a la difusión, el VAE reaparece —ya no como generador final, sino como el autoencoder que define el espacio latente sobre el cual la difusión trabaja. Esto cierra el arco: el primer modelo generativo de la clase termina siendo un componente del último.

## Notas y enlaces

- arXiv: https://arxiv.org/abs/1312.6114 (primera versión, 20 de diciembre de 2013)
- Venue: International Conference on Learning Representations (ICLR) 2014.
- Afiliación: Machine Learning Group, Universiteit van Amsterdam.
- Trabajo simultáneo e independiente: Rezende, Mohamed y Wierstra (2014), *Stochastic Backpropagation and Variational Inference in Deep Latent Gaussian Models* — origen conjunto del VAE con el mismo reparameterization trick.
