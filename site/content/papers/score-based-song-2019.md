---
title: "Score-Based Generative Modeling / NCSN (2019)"
weight: 337
math: true
---

{{< paper-card
    title="Generative Modeling by Estimating Gradients of the Data Distribution"
    authors="Yang Song, Stefano Ermon"
    year="2019"
    venue="NeurIPS 2019"
    pdf="/papers/score-based-song-2019.pdf"
    arxiv="1907.05600" >}}
Paper de Stanford que inaugura una familia distinta de modelos generativos: en vez de modelar la densidad $p(x)$, modela el **score** $\nabla_x \log p(x)$ —el campo vectorial que apunta hacia donde la log-densidad crece más rápido—. El score se estima con **denoising score matching** a múltiples niveles de ruido (la **Noise Conditional Score Network, NCSN**) y se generan muestras con **dinámica de Langevin recocida (annealed)**. El truco que hace que todo funcione es perturbar los datos con ruido gaussiano a varias escalas, lo que resuelve los dos obstáculos del score ingenuo (variedades de baja dimensión y regiones de baja densidad). Logra un **inception score de 8.87 en CIFAR-10** sin entrenamiento adversarial. Es, junto con [DDPM](/papers/ddpm-ho-2020), uno de los dos pilares de la difusión moderna.
{{< /paper-card >}}

---

## Contexto: la otra estirpe de la difusión

Hacia 2019 el campo generativo estaba dominado por dos enfoques con limitaciones intrínsecas. Los **métodos basados en verosimilitud** (autorregresivos como PixelCNN, flows normalizadores, VAEs, EBMs) usan la log-verosimilitud como objetivo, pero pagan un precio: o necesitan arquitecturas especializadas que construyan una densidad *normalizada* (imponiendo invertibilidad y jacobianos tratables), o recurren a surrogates como el ELBO de los VAEs o contrastive divergence en los EBMs. Los **GANs** evitan modelar la densidad usando entrenamiento adversarial, pero su juego min-max es inestable y su objetivo no sirve para evaluar ni comparar modelos cuantitativamente.

En este panorama, Song y Ermon recuperan una idea con raíces más antiguas: el **score matching** de Hyvärinen (2005), diseñado para aprender modelos estadísticos no normalizados. La clave conceptual es que **el score esquiva la constante de normalización**. Para una densidad $p(x) = \tilde{p}(x)/Z$, el gradiente del logaritmo elimina $Z$:

$$\nabla_x \log p(x) = \nabla_x \log \tilde{p}(x) - \nabla_x \log Z = \nabla_x \log \tilde{p}(x)$$

porque $Z$ no depende de $x$. Esto libera al modelo de la camisa de fuerza que sufren los métodos de verosimilitud: no hace falta que la red parametrice una densidad normalizada, basta con que produzca un campo vectorial $\mathbb{R}^D \to \mathbb{R}^D$.

El paper también se sitúa frente a la **Nonequilibrium Thermodynamics** de Sohl-Dickstein et al. (2015) —que prescribe un proceso de difusión que transforma datos en ruido y aprende a revertirlo, la semilla de DDPM— y la critica por no escalar (requiere miles de pasos simulados en entrenamiento). La distinción estructural que NCSN reivindica es doble: **no muestrea de una cadena de Markov durante el entrenamiento**, y entrenamiento y muestreo están **desacoplados** (cualquier estimador de score combinable con cualquier muestreador basado en gradientes). En retrospectiva, esta es la otra estirpe de la difusión: mientras DDPM la deriva desde la termodinámica y la verosimilitud variacional, NCSN la deriva desde el score y el muestreo de Langevin.

## Estimar el score: score matching

El objetivo básico de score matching minimiza el error cuadrático esperado entre el score estimado y el verdadero, $\frac{1}{2}\mathbb{E}_{p_{\text{data}}}[\lVert s_\theta(x) - \nabla_x \log p_{\text{data}}(x)\rVert_2^2]$. El problema es que $\nabla_x \log p_{\text{data}}(x)$ es desconocido. Hyvärinen demostró que este objetivo equivale —hasta una constante— a uno que solo depende de la red y de los datos, $\mathbb{E}_{p_{\text{data}}}[\text{tr}(\nabla_x s_\theta(x)) + \frac{1}{2}\lVert s_\theta(x)\rVert_2^2]$. Pero la traza del jacobiano no escala a redes profundas. El paper adopta dos variantes:

- **Denoising score matching** (Vincent, 2011): esquiva por completo la traza. Perturba $x$ con un kernel de ruido $q_\sigma(\tilde{x}\mid x)$ y entrena la red para estimar el score de la distribución *perturbada* $q_\sigma(\tilde{x})$, cuyo score del kernel tiene forma cerrada. Es el objetivo principal de NCSN porque encaja naturalmente con datos perturbados. La advertencia: el óptimo aproxima $\nabla_x \log p_{\text{data}}(x)$ solo cuando el ruido es pequeño.
- **Sliced score matching** (Song et al., 2019): aproxima la traza con proyecciones aleatorias. Estima el score de los datos *sin* perturbar, pero cuesta unas cuatro veces más.

## Los dos obstáculos: variedad y baja densidad

El paper dedica una sección a *por qué* la aplicación ingenua falla, con dos diagnósticos experimentales.

**La hipótesis de la variedad.** Los datos reales se concentran en variedades de baja dimensión embebidas en el espacio ambiente. Bajo esta hipótesis, el score $\nabla_x \log p_{\text{data}}(x)$ —un gradiente del espacio ambiente— está **indefinido** cuando $x$ está confinado a la variedad. Empíricamente, al entrenar con CIFAR-10 crudo la pérdida fluctúa irregularmente; al perturbar con un ruido minúsculo $\mathcal{N}(0, 0.0001)$ —imperceptible al ojo— la pérdida converge limpiamente.

**Regiones de baja densidad.** Donde $p_{\text{data}} \approx 0$ casi no hay muestras de entrenamiento, así que el score matching no tiene evidencia para estimar el score con precisión. Peor aún es el **mezclado lento de Langevin**: cuando dos modos están separados por una región de baja densidad, la dinámica no recupera los pesos relativos de los modos. La razón es elegante: para una mezcla $\pi p_1(x) + (1-\pi)p_2(x)$ con soportes disjuntos, dentro de cada componente el score *no depende de $\pi$* —el peso de la mezcla se cancela al tomar el gradiente del logaritmo—, así que las muestras de Langevin ignoran $\pi$.

## Generar: dinámica de Langevin recocida

La **dinámica de Langevin** produce muestras de $p(x)$ usando solo el score. Dado un tamaño de paso $\epsilon > 0$ y una inicialización $\tilde{x}_0 \sim \pi(x)$, itera:

$$\tilde{x}_t = \tilde{x}_{t-1} + \frac{\epsilon}{2}\nabla_x \log p(\tilde{x}_{t-1}) + \sqrt{\epsilon}\, z_t, \quad z_t \sim \mathcal{N}(0, I)$$

Cuando $\epsilon \to 0$ y $T \to \infty$, $\tilde{x}_T$ es una muestra exacta de $p(x)$. Es un descenso de gradiente sobre la log-densidad con ruido inyectado que evita colapsar al modo.

La **Langevin recocida** es la receta de muestreo de NCSN. Se construye una secuencia geométrica de niveles de ruido $\sigma_1 > \sigma_2 > \cdots > \sigma_L$ y **una sola red $s_\theta(x, \sigma)$ condicionada en $\sigma$** estima simultáneamente los scores de todas las distribuciones perturbadas. El muestreo arranca desde ruido usando los scores del nivel más alto $\sigma_1$ (donde el campo es suave y bien definido), corre $T$ pasos de Langevin, y usa las muestras finales como inicialización para $\sigma_2$, reduciendo el tamaño de paso $\alpha_i = \epsilon \cdot \sigma_i^2/\sigma_L^2$, y así hasta $\sigma_L$ —tan pequeño que la distribución perturbada es casi idéntica a los datos reales—. Es **simulated annealing** aplicado al muestreo: cada nivel entrega buenas inicializaciones al siguiente, transfiriendo los beneficios del ruido alto a los niveles bajos. En el experimento de la mezcla de gaussianas la Langevin estándar falla en recuperar los pesos de los modos mientras que la recocida los recupera fielmente.

El hilo que une todo: el ruido no es un mal necesario, es el ingrediente que *hace bien definido y tratable* el problema. Perturbar con gaussiano garantiza soporte en todo $\mathbb{R}^D$ (mata el problema de la variedad), llena las regiones de baja densidad (da señal de entrenamiento) y crea un puente de distribuciones intermedias que acelera el mezclado entre modos.

## Entrenamiento de la NCSN

Con el kernel gaussiano $q_\sigma(\tilde{x}\mid x) = \mathcal{N}(\tilde{x}\mid x, \sigma^2 I)$, el score del kernel es $-(\tilde{x}-x)/\sigma^2$. Para un $\sigma$ dado, el objetivo de denoising score matching es:

$$\ell(\theta;\sigma) = \frac{1}{2}\mathbb{E}_{p_{\text{data}}}\mathbb{E}_{\tilde{x}\sim\mathcal{N}(x,\sigma^2 I)}\left\lVert s_\theta(\tilde{x},\sigma) + \frac{\tilde{x}-x}{\sigma^2}\right\rVert_2^2$$

El objetivo unificado combina todos los niveles, $\mathcal{L}(\theta) = \frac{1}{L}\sum_{i=1}^L \lambda(\sigma_i)\ell(\theta;\sigma_i)$. Como cerca del óptimo se observa $\lVert s_\theta(x,\sigma)\rVert_2 \propto 1/\sigma$, se elige **$\lambda(\sigma) = \sigma^2$**, lo que deja todos los términos del mismo orden de magnitud. La arquitectura combina **U-Net** (RefineNet) con **convoluciones dilatadas** y una variante de **conditional instance normalization** para inyectar la condición $\sigma_i$. No requiere ninguna restricción de normalización de la densidad.

## Experimentos

NCSN se evalúa en MNIST, CelebA y CIFAR-10, con $L = 10$ niveles geométricos ($\sigma_1 = 1$, $\sigma_{10} = 0.01$), $T = 100$ pasos por nivel, inicializando desde ruido uniforme.

- **CIFAR-10.** Como modelo **incondicional**, NCSN logra un **inception score de 8.87 ± 0.12**, estado del arte de la época —superando a ProgressiveGAN (8.80), SNGAN (8.22) e incluso a varios modelos *condicionales*— y un **FID de 25.32**, comparable con SNGAN. Todo sin entrenamiento adversarial, sin MCMC durante el entrenamiento y sin arquitecturas especiales.
- **Ablación reveladora.** Un baseline con un solo nivel de ruido ($\sigma = 0.01$) y Langevin estándar —sin recocido ni múltiples niveles— **fracasa por completo**: el ruido pequeño basta para esquivar el problema de la variedad, pero no para dar señal del score en regiones de baja densidad. Esto confirma que *ambas* innovaciones (múltiples niveles + Langevin recocido) son necesarias.
- **Inpainting.** Con una modificación simple que reinyecta los píxeles conocidos en cada paso, NCSN produce inpaintings diversos y coherentes en CelebA y CIFAR-10, manejando oclusiones arbitrarias —a diferencia de PixelCNN, que solo imputa en orden raster—.

## Limitaciones

- **Muestreo lento.** La Langevin recocida requiere $L \times T$ evaluaciones de red (en imágenes, $10 \times 100 = 1000$), un orden de magnitud más lento que una sola pasada de un GAN. Es la limitación de muestreo iterativo que caracteriza a toda la difusión.
- **Sensibilidad a los hiperparámetros de ruido.** La elección de la secuencia $\{\sigma_i\}$ (rango, número de niveles, escala) es delicada y específica del dominio; trabajos posteriores del propio Song (NCSNv2, 2020) abordarían las recetas de escalamiento a resoluciones más altas.
- **Arquitectura atada a imágenes.** Explota arquitecturas de segmentación (U-Net, convoluciones dilatadas) propias de visión.
- **Aproximación de Langevin.** Para $\epsilon > 0$ y $T < \infty$ se omite la corrección de Metropolis-Hastings, asumiendo error despreciable —una aproximación no garantizada en general—.

## Impacto: la fundación de la difusión moderna

NCSN es, junto con [DDPM](/papers/ddpm-ho-2020) (Ho et al., 2020), uno de los **dos pilares sobre los que se construyó la difusión moderna**. Ambos papers llegaron al mismo destino por caminos distintos: DDPM desde la termodinámica de no equilibrio y la verosimilitud variacional; NCSN desde el score matching y el muestreo de Langevin. La intuición compartida —perturbar datos con ruido a múltiples niveles y aprender a revertir esa perturbación paso a paso— es idéntica.

La síntesis llegó en 2021, cuando **Song et al.** ("Score-Based Generative Modeling through Stochastic Differential Equations", ICLR 2021) demostraron que **score-based y DDPM son dos discretizaciones de un mismo proceso continuo descrito por una ecuación diferencial estocástica (SDE)**. En ese marco, el forward process de DDPM (la SDE "VP", *variance preserving*) y el de NCSN (la SDE "VE", *variance exploding*) son casos particulares de una misma familia, y ambos modelos aprenden el score de la distribución perturbada en el tiempo. Esta unificación es la base teórica de casi toda la difusión posterior: classifier guidance, classifier-free guidance, los *probability flow ODEs* que aceleran el muestreo, y los modelos latentes como Stable Diffusion. La idea de que **el score es el objeto central que la red aprende** —y que generar es integrar una SDE/ODE de reversión guiada por ese score— nace aquí.

## Conexión con la Clase 29

La [Clase 29](/clases/clase-29) enseña la difusión principalmente a través de **DDPM**: el forward process que agrega ruido gaussiano paso a paso y el reverse process que aprende a invertirlo. NCSN es la **perspectiva "score" exactamente del mismo proceso**, y profundiza la idea central de la clase de tres maneras.

- **Es la otra vista del mismo modelo.** Donde DDPM predice el ruido $\epsilon$ añadido en cada paso, NCSN predice el **score** de la distribución perturbada. Pero ambos son la misma información: el score de un kernel gaussiano $\mathcal{N}(x, \sigma^2 I)$ es $-(\tilde{x}-x)/\sigma^2 = -\epsilon/\sigma$, así que **predecir el ruido y predecir el score son objetivos equivalentes hasta un factor de escala**. Es la equivalencia que Song et al. (2021) formalizaron con las SDEs.
- **Profundiza el "aprender a invertir el ruido paso a paso".** El núcleo pedagógico de la clase se ve aquí desde el muestreo: la Langevin recocida *es* el reverse process, descendiendo el gradiente de la log-densidad nivel a nivel de ruido, recorriendo la misma trayectoria temporal $t = T \to 0$ que el reverse process de DDPM.
- **Justifica por qué la difusión necesita múltiples niveles de ruido.** La sección de obstáculos (variedad + baja densidad) explica *por qué* un solo nivel no basta —algo que en DDPM aparece como dado (el schedule $\beta_t$) pero que aquí se motiva desde primeros principios—. La ablación del baseline de un solo nivel es la prueba experimental.

Para profundizar, ver el paper hermano [DDPM (Ho et al., 2020)](/papers/ddpm-ho-2020), el fundamento transversal de [modelos de difusión](/fundamentos/modelos-de-difusion) y el panorama general en [modelos generativos](/fundamentos/modelos-generativos). El contexto completo de la sesión está en [Clase 29 — Modelos Generativos en Visión](/clases/clase-29).
