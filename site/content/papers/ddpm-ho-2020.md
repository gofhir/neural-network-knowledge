---
title: "DDPM: Denoising Diffusion Probabilistic Models (2020)"
weight: 330
math: true
---

{{< paper-card
    title="Denoising Diffusion Probabilistic Models"
    authors="Jonathan Ho, Ajay Jain, Pieter Abbeel"
    year="2020"
    venue="NeurIPS 2020"
    pdf="/papers/ddpm-ho-2020.pdf"
    arxiv="2006.11239" >}}
Paper de UC Berkeley que detonó la era de la difusión. No inventa los modelos de difusión (eso fue Sohl-Dickstein et al. 2015), pero **demuestra por primera vez que pueden generar imágenes de alta calidad**, comparables o mejores que las GANs de la época: FID de **3.17** en CIFAR-10 incondicional. La idea es definir un proceso *forward* fijo que convierte datos en ruido gaussiano gradualmente, y aprender el proceso *reverse* que lo invierte paso a paso. La clave técnica es reparametrizar el modelo para que la red **prediga el ruido $\epsilon$** que se añadió, con una pérdida cuadrática desarmadoramente simple, $L_\text{simple}=\|\epsilon-\epsilon_\theta(x_t,t)\|^2$. Es el cimiento directo de Stable Diffusion, DALL·E 2 e Imagen. Material de la [Clase 29](/clases/clase-29).
{{< /paper-card >}}

---

## Contexto

El antecedente directo es Sohl-Dickstein et al. (ICML 2015), que tomó prestada la intuición de la **termodinámica de no-equilibrio**: un sistema estructurado de baja entropía (la distribución de datos) puede llevarse hacia un estado de equilibrio simple y de alta entropía (ruido gaussiano) mediante un proceso de difusión que añade ruido infinitesimal en muchos pasos. Si cada paso es suficientemente pequeño, el proceso inverso —de ruido a datos— tiene la *misma forma funcional* (también gaussiano), y por tanto puede aprenderse. Esa simetría forward/reverse para pasos pequeños es la observación de Feller que sostiene todo el método.

El trabajo de 2015 era bello pero **nunca produjo imágenes competitivas**. Durante cinco años los modelos de difusión fueron una curiosidad teórica mientras la generación de imágenes la dominaban las GANs (StyleGAN, BigGAN), los autorregresivos (PixelCNN, Sparse Transformer), los *flows* y los VAEs. En paralelo, una segunda línea —los modelos basados en *score matching* de Song & Ermon (2019, NCSN)— mostraba que estimar el gradiente de la densidad ($\nabla_x \log p(x)$, el "score") sobre múltiples escalas de ruido y muestrear con dinámica de Langevin recocida producía imágenes comparables a GANs. DDPM es **el punto donde estas dos líneas convergen**: Ho et al. demuestran que su difusión con predicción de ruido es matemáticamente equivalente al score matching de NCSN.

El renacimiento no vino de una idea nueva sino de **ingeniería de la parametrización y la pérdida**: predecir el ruido en lugar de la media, una pérdida MSE simple sin pesos, una *variance schedule* lineal pequeña, $T=1000$ pasos, y una U-Net con atención y embeddings de tiempo.

## El proceso forward: convertir datos en ruido

El **proceso forward** (o de difusión) es la posterior $q(x_{1:T}|x_0)$. A diferencia de un VAE, **no tiene parámetros aprendibles**: es una cadena de Markov fija que añade ruido gaussiano según una *variance schedule* $\beta_1,\dots,\beta_T$:

$$q(x_t|x_{t-1}) := \mathcal{N}\big(x_t;\; \sqrt{1-\beta_t}\,x_{t-1},\; \beta_t I\big)$$

Cada paso escala el dato anterior por $\sqrt{1-\beta_t}$ (lo encoge hacia el origen) y le suma ruido de varianza $\beta_t$. Tras $T$ pasos, $x_T$ es prácticamente ruido gaussiano puro $\mathcal{N}(0,I)$, sin información del dato original.

La **propiedad que lo hace tratable** —central para el entrenamiento eficiente— es que se puede muestrear $x_t$ en un paso $t$ arbitrario **en forma cerrada**, sin recorrer la cadena. Definiendo $\alpha_t := 1-\beta_t$ y $\bar\alpha_t := \prod_{s=1}^{t}\alpha_s$:

$$q(x_t|x_0) = \mathcal{N}\big(x_t;\; \sqrt{\bar\alpha_t}\,x_0,\; (1-\bar\alpha_t)I\big) \quad\Longrightarrow\quad x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\;\; \epsilon\sim\mathcal{N}(0,I)$$

El parámetro $\bar\alpha_t$ controla una interpolación: con $\bar\alpha_t\to 1$ (t pequeño) $x_t$ es casi $x_0$; con $\bar\alpha_t\to 0$ (t grande) $x_t$ es casi ruido puro. Esta fórmula permite, durante el entrenamiento, **muestrear un $t$ uniforme al azar y saltar directamente a $x_t$** sin simular la cadena — la diferencia entre un entrenamiento factible y uno prohibitivo.

## El proceso reverse: aprender a invertir paso a paso

El **proceso reverse** es la conjunta $p_\theta(x_{0:T})$, una cadena de Markov con transiciones gaussianas aprendidas que arranca de $p(x_T)=\mathcal{N}(0,I)$:

$$p_\theta(x_{t-1}|x_t) := \mathcal{N}\big(x_{t-1};\; \mu_\theta(x_t,t),\; \Sigma_\theta(x_t,t)\big)$$

Que las transiciones inversas sean gaussianas se justifica por la observación de Feller: con $\beta_t$ pequeños, forward y reverse comparten forma funcional. Generar una imagen consiste en muestrear $x_T\sim\mathcal{N}(0,I)$ y aplicar $T$ transiciones inversas hasta $x_0$.

El entrenamiento optimiza el *bound* variacional sobre la log-verosimilitud negativa, reescrito (vía Rao-Blackwellización) en una forma de varianza reducida donde todas las divergencias KL son **comparaciones entre gaussianas** calculables en forma cerrada. Esto es posible porque durante el entrenamiento $x_0$ está disponible, lo que hace tratable la posterior $q(x_{t-1}|x_t,x_0)$ —la "respuesta correcta" que el modelo intenta imitar—. El término del prior $L_T$ es constante (forward sin parámetros) y se ignora; el grueso del aprendizaje vive en las transiciones inversas.

Para la varianza, DDPM elige **no aprenderla**: la fija a constantes dependientes del tiempo $\sigma_t^2 I$ (con $\sigma_t^2=\beta_t$ o $\tilde\beta_t$, resultados similares). Aprender una $\Sigma$ diagonal volvía el entrenamiento inestable — algo que *Improved DDPM* (Nichol & Dhariwal 2021) revisaría después con éxito.

## La clave: predecir el ruido $\epsilon$

Aquí está el corazón del paper. La parametrización **más directa** sería que la red prediga la media $\tilde\mu_t$ de la posterior. Pero Ho et al. dan un paso astuto: sustituyen $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ en la fórmula de $\tilde\mu_t$, revelando que **basta con que la red prediga $\epsilon$** y la media inversa se reconstruye analíticamente:

$$\mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t)\Big)$$

La red $\epsilon_\theta$ no aprende a "limpiar" hacia la media; aprende a **identificar el ruido presente** en $x_t$. Como $x_t$ ya es entrada de la red, predecir $\epsilon$ y predecir $\mu$ son matemáticamente intercambiables, pero empíricamente **predecir $\epsilon$ es muy superior** cuando se combina con la pérdida simplificada.

Sustituyendo esta parametrización, el término del *bound* colapsa a la forma exacta del *denoising score matching* de NCSN sobre múltiples escalas de ruido. Así, **optimizar algo que parece score matching equivale a inferencia variacional para ajustar un muestreador tipo Langevin** — la equivalencia que el paper considera una de sus contribuciones primarias.

El paso final es pragmático. El *bound* completo lleva un coeficiente complicado delante de cada término; Ho et al. encuentran que **descartarlo** (fijarlo a 1) mejora la calidad de muestra y simplifica la implementación:

$$L_\text{simple}(\theta) := \mathbb{E}_{t,x_0,\epsilon}\Big[\big\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\,t)\big\|^2\Big]$$

con $t$ uniforme en $\{1,\dots,T\}$. Es un simple MSE entre el ruido real y el predicho. El *reweighting* implícito **infrapondera los $t$ pequeños** (denoising trivial) y concentra la red en los $t$ grandes (denoising difícil). La ablación confirma que esto es lo que lleva FID de 13.51 (bound verdadero) a **3.17** ($L_\text{simple}$). El precio: la verosimilitud empeora ligeramente (≤3.75 bits/dim), confirmando que $L_\text{simple}$ prioriza percepción sobre compresión exacta.

## La red $\epsilon_\theta$: una U-Net

La red que predice el ruido es una [U-Net](/papers/unet-ronneberger-2015) (Ronneberger et al. 2015), la arquitectura encoder-decoder con *skip connections* nacida para segmentación biomédica. DDPM usa un *backbone* tipo PixelCNN++ con **group normalization**. Detalles clave:

- **Parámetros compartidos en el tiempo:** una sola red sirve para los $T=1000$ pasos. El paso $t$ se comunica mediante **embeddings sinusoidales** estilo Transformer, inyectados en cada bloque residual, para que la red adapte su comportamiento al nivel de ruido.
- **Auto-atención** en la resolución de $16\times16$, que captura dependencias de largo alcance que la convolución pura no alcanza.

Su estructura multi-escala con *skips* es ideal para operar a la vez sobre estructura global (qué objeto hay) y detalle local (texturas), y se ha mantenido como la columna vertebral de los modelos de difusión hasta hoy.

El bucle de **entrenamiento** es la traducción directa de $L_\text{simple}$: muestrear una imagen real $x_0$, un paso $t$ uniforme y un ruido $\epsilon$, y dar un paso de gradiente sobre $\|\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\,t)\|^2$. El **muestreo** es iterativo de $T$ a $1$: partir de $x_T\sim\mathcal{N}(0,I)$ y aplicar
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\Big) + \sigma_t z$$
restando una fracción del ruido predicho y añadiendo una pizca de ruido fresco (excepto el último paso). En la generación progresiva, **las características de gran escala aparecen primero y los detalles finos al final**.

## Resultados

Con $T=1000$ y una *variance schedule* **lineal** de $\beta_1=10^{-4}$ a $\beta_T=0.02$:

- **CIFAR-10 incondicional:** Inception Score 9.46, **FID = 3.17**. Supera a la mayoría de la literatura, incluyendo modelos *condicionales por clase*, y queda solo detrás de StyleGAN2+ADA. Versus rivales directos: NCSN 25.32, SNGAN 21.7, EBM 38.2.
- **LSUN $256\times256$:** FID 7.89 (Church), 4.90 (Bedroom), calidad similar a ProgressiveGAN.
- **CelebA-HQ $256\times256$:** muestras de alta fidelidad e **interpolaciones** suaves en espacio latente (codificar dos imágenes a $t=500$, interpolar y decodificar).
- **Ablación:** predecir $\epsilon$ con $L_\text{simple}$ da FID 3.17, frente a 13.51 con el *bound* verdadero — el salto cuantitativo que valida la tesis.

Un matiz honesto: las *log-likelihoods* (≤3.75 bits/dim) no compiten con los autorregresivos (Sparse Transformer 2.80). Más de la mitad del *codelength* describe distorsiones imperceptibles, lo que el paper reencuadra elegantemente: los modelos de difusión son **excelentes compresores con pérdida**.

## Limitaciones

- **Muestreo lento — la limitación dominante.** Generar una imagen requiere **$T$ pasos secuenciales** (1000), cada uno una evaluación completa de la U-Net. No se puede paralelizar en el tiempo (cada $x_{t-1}$ depende de $x_t$). Frente a una GAN que genera en un solo *forward pass*, DDPM es órdenes de magnitud más lento en inferencia. Esto motivó casi toda la investigación posterior de aceleración: DDIM, destilación progresiva, solvers de EDO, modelos de consistencia.
- **Verosimilitudes no competitivas** con los modelos autorregresivos.
- **Varianza inversa no aprendida:** aprenderla desestabiliza el entrenamiento; el paper la fija.

El *trade-off* explica el dominio posterior de la difusión: combina la **alta calidad** de las GANs con la **alta cobertura de la distribución** de los modelos de verosimilitud (sin *mode collapse*), pagando solo el precio de un muestreo lento —el único costo que la investigación posterior atacó.

## Impacto

DDPM es, sin exageración, **el paper que detonó la era de la difusión**. En menos de dos años: Nichol & Dhariwal (2021) mejoraron la schedule (cosine) y aprendieron la varianza; Dhariwal & Nichol (2021) demostraron que la difusión con *classifier guidance* **supera a las GANs** en ImageNet; Ho & Salimans (2022) introdujeron *classifier-free guidance*, hoy ubicuo; Rombach et al. (2022) movieron la difusión a un espacio latente comprimido ([Latent Diffusion](/papers/latent-diffusion-rombach-2022) → Stable Diffusion), haciéndola viable a alta resolución en hardware de consumo. DALL·E 2, Imagen, Midjourney y prácticamente todo generador texto-a-imagen comercial descansan sobre la maquinaria de DDPM: forward gaussiano, predicción de $\epsilon$, U-Net con embedding de tiempo, muestreo iterativo.

El legado conceptual es igual de importante: DDPM **reconcilió** las dos tradiciones que llevaban años separadas —la difusión variacional de Sohl-Dickstein y el [score matching](/papers/score-based-song-2019) de Song & Ermon— mostrando que eran la misma cosa vista desde dos ángulos. Esa unificación abrió el marco continuo de las *ecuaciones diferenciales estocásticas* que subsume a ambas.

## Por qué importa para la Clase 29

La [Clase 29](/clases/clase-29) ("Modelos Generativos en Visión") presenta los modelos de difusión con una frase que es el resumen exacto de DDPM: *"convertir datos en ruido gradualmente y aprender a invertir paso a paso"*, con la U-Net como el aprendiz del paso inverso. Cada pieza del paper mapea a la narrativa de la clase: el forward $q(x_t|x_{t-1})$ que no se aprende, el reverse $p_\theta(x_{t-1}|x_t)$ que parte de ruido y reconstruye, y la $\epsilon$-prediction como el punto donde un estudiante entiende por qué la difusión es entrenable: el objetivo se reduce a "adivina qué ruido le eché a esta imagen".

## Notas y enlaces

- Fundamento específico: [/fundamentos/modelos-de-difusion](/fundamentos/modelos-de-difusion) — forward/reverse, $\epsilon$-prediction, score matching, schedules.
- Fundamento transversal: [/fundamentos/modelos-generativos](/fundamentos/modelos-generativos) — el marco común de VAE/GAN/difusión y sus trade-offs.
- Paper relacionado: [/papers/unet-ronneberger-2015](/papers/unet-ronneberger-2015) — la U-Net que DDPM adopta como $\epsilon_\theta$.
- Paper relacionado: [/papers/latent-diffusion-rombach-2022](/papers/latent-diffusion-rombach-2022) — la difusión en espacio latente (Stable Diffusion).
- Paper relacionado: [/papers/score-based-song-2019](/papers/score-based-song-2019) — el score matching que DDPM demuestra equivalente.
- Código: [github.com/hojonathanho/diffusion](https://github.com/hojonathanho/diffusion) (TensorFlow, Cloud TPUs).
- Preprint: [arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239).
