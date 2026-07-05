---
title: "Modelos de Difusión"
weight: 106
math: true
---

Un **modelo de difusión** (diffusion model) es una familia de modelos generativos que aprende a sintetizar datos —imágenes, audio, moléculas— **invirtiendo un proceso de corrupción por ruido**. La idea, en una frase, es engañosamente simple: si destruir información es fácil (basta sumar ruido gaussiano gradualmente hasta que una foto se convierta en niebla estadística), entonces el problema difícil —generar— se puede atacar enseñándole a una red a **deshacer ese ruido un poquito a la vez**. Encadenando miles de pequeños pasos de "limpieza" se llega de ruido puro a una muestra nítida. Esta receta, latente desde 2015 pero hecha práctica por [DDPM (Ho et al., 2020)](/papers/ddpm-ho-2020), es hoy el motor de Stable Diffusion, DALL·E 2, Imagen y Midjourney. Este fundamento desarrolla la difusión **en detalle**: el proceso forward, el proceso reverse aprendido, la parametrización del ruido, la U-Net como denoiser, la perspectiva score-based, el condicionamiento, la difusión latente y el trade-off de velocidad. Para el panorama comparado de VAE/GAN/difusión —dónde encaja cada familia— está el fundamento de [modelos generativos](/fundamentos/modelos-generativos); aquí entramos en la maquinaria.

---

## 1. La intuición: destruir es fácil, reconstruir es lo que se aprende

El truco conceptual de la difusión es separar dos procesos. El primero, **forward**, es una receta fija y conocida que toma un dato real $x_0$ y le añade ruido gaussiano en muchos pasos pequeños, hasta que tras $T$ pasos queda $x_T$, indistinguible de ruido blanco $\mathcal{N}(0, I)$. No hay nada que aprender aquí: es un "profesor" que define la tarea. El segundo, **reverse**, es el proceso que sí se aprende: una red neuronal que, dado un dato ruidoso $x_t$ y el nivel de ruido $t$, estima cómo dar un paso hacia atrás —hacia un $x_{t-1}$ algo más limpio—. Generar una muestra nueva consiste en arrancar de ruido puro $x_T \sim \mathcal{N}(0,I)$ y aplicar el paso reverse aprendido $T$ veces, descendiendo de $t=T$ a $t=0$.

¿Por qué esta descomposición funciona donde fallaron enfoques más directos? Porque **cada paso individual es trivial de modelar**. Pedirle a una red que mapee ruido a una imagen completa de una sola vez (como hace una GAN) es una tarea difícil y propensa a inestabilidad. Pedirle que quite *una pizca* de ruido a una imagen casi-limpia es una regresión sencilla y estable. La difusión cambia un problema de generación intratable por miles de problemas de denoising fáciles, encadenados. La raíz teórica de por qué los pasos deben ser muchos y pequeños es una observación de Feller (1949): cuando el ruido añadido por paso es infinitesimal, el paso inverso tiene **la misma forma funcional gaussiana** que el paso forward, y por tanto se puede aprender con una gaussiana. Esa simetría forward/reverse para pasos pequeños sostiene todo el método.

{{< concept-alert type="clave" >}}
La difusión no aprende a generar de la nada: aprende a **invertir un proceso de corrupción conocido**. El forward (añadir ruido) es fijo y sin parámetros; el reverse (quitar ruido) es lo que la red aprende. Generar = empezar en ruido y aplicar el denoiser aprendido paso a paso, de $t=T$ a $t=0$.
{{< /concept-alert >}}

---

## 2. El proceso forward: una cadena de Markov que añade ruido

El proceso forward $q(x_{1:T} \mid x_0)$ es una **cadena de Markov** fija que añade ruido gaussiano según una *variance schedule* predefinida $\beta_1, \dots, \beta_T$ (valores pequeños crecientes, por ejemplo de $10^{-4}$ a $0.02$ en DDPM). Cada transición es:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\big(x_t;\; \sqrt{1-\beta_t}\,x_{t-1},\; \beta_t I\big).
$$

Cada paso hace dos cosas: escala el estado anterior por $\sqrt{1-\beta_t}$ (lo encoge ligeramente hacia el origen, evitando que la varianza explote) y le suma ruido gaussiano de varianza $\beta_t$. Tras $T$ pasos, toda la estructura del dato original se ha disuelto y $x_T \approx \mathcal{N}(0, I)$.

La propiedad que vuelve **tratable** el entrenamiento —y sin la cual la difusión sería computacionalmente prohibitiva— es que el forward admite muestrear $x_t$ en un paso arbitrario $t$ **en forma cerrada**, sin recorrer los $t$ pasos intermedios. Definiendo $\alpha_t := 1-\beta_t$ y el producto acumulado $\bar\alpha_t := \prod_{s=1}^{t}\alpha_s$:

$$
q(x_t \mid x_0) = \mathcal{N}\big(x_t;\; \sqrt{\bar\alpha_t}\,x_0,\; (1-\bar\alpha_t)I\big),
$$

lo que se puede escribir directamente con el truco de reparametrización como:

$$
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon, \qquad \epsilon \sim \mathcal{N}(0, I).
$$

El coeficiente $\bar\alpha_t$ es el corazón de esta fórmula: controla una **interpolación** entre dato y ruido. Con $\bar\alpha_t \to 1$ (para $t$ pequeño), $x_t$ es casi $x_0$; con $\bar\alpha_t \to 0$ (para $t$ grande), $x_t$ es casi ruido puro. Esto significa que durante el entrenamiento podemos **elegir un $t$ uniforme al azar y saltar directamente a $x_t$** desde cualquier imagen $x_0$, sin simular la cadena. Esa es la diferencia entre un entrenamiento factible y uno imposible.

Una segunda cantidad clave es la **posterior del forward condicionada en $x_0$**, $q(x_{t-1}\mid x_t, x_0)$, que también es gaussiana y tiene forma cerrada. Es la "respuesta correcta" que el modelo intentará imitar: la distribución exacta del paso reverse, *si conociéramos* el $x_0$ original. Durante el entrenamiento $x_0$ está disponible, así que esta posterior es calculable y sirve de blanco — es lo que hace tratable el objetivo, igual que el conocer $x_0$ hace tratable el ELBO de un VAE.

---

## 3. El proceso reverse aprendido: predecir el ruido

El proceso reverse $p_\theta(x_{0:T})$ es una cadena de Markov de transiciones gaussianas **aprendidas**, que arranca de $p(x_T) = \mathcal{N}(0, I)$ y reconstruye el dato:

$$
p_\theta(x_{t-1}\mid x_t) = \mathcal{N}\big(x_{t-1};\; \mu_\theta(x_t, t),\; \Sigma_\theta(x_t, t)\big).
$$

Se entrena maximizando una cota variacional sobre la log-verosimilitud, que se descompone término a término en divergencias KL entre **gaussianas conocidas** —comparando la transición aprendida $p_\theta(x_{t-1}\mid x_t)$ contra la posterior verdadera $q(x_{t-1}\mid x_t, x_0)$—. DDPM toma dos decisiones de simplificación: fija la varianza $\Sigma_\theta = \sigma_t^2 I$ a constantes (aprenderla desestabilizaba el entrenamiento), y reparametriza la media de una forma astuta.

Aquí está el corazón de DDPM. La parametrización más directa sería que la red prediga la media $\mu_\theta$ de la transición. Pero sustituyendo $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ en la fórmula de la posterior, la media se reexpresa en términos de $x_t$ y del **ruido $\epsilon$** que se añadió. Esto revela que basta con que la red **prediga el ruido**, y la media se reconstruye analíticamente:

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t)\Big).
$$

La red $\epsilon_\theta(x_t, t)$ no aprende a "limpiar hacia una media"; aprende a **identificar el ruido presente** en $x_t$. Como $x_t$ ya es entrada de la red, predecir $\epsilon$ y predecir $\mu$ son matemáticamente intercambiables, pero empíricamente predecir $\epsilon$ es muy superior. Sustituyendo esta parametrización en la cota y **descartando los pesos** delante de cada término (un cambio pragmático que mejora la calidad de muestra), el objetivo colapsa a un simple error cuadrático medio:

$$
L_\text{simple}(\theta) = \mathbb{E}_{t,\,x_0,\,\epsilon}\Big[\big\|\epsilon - \epsilon_\theta\big(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\; t\big)\big\|^2\Big],
$$

con $t$ uniforme en $\{1,\dots,T\}$. Esta pérdida es desarmadoramente simple para lo que logra: el entrenamiento se reduce a "toma una imagen, échale ruido de nivel $t$ al azar, y haz que la red adivine qué ruido echaste". El detalle clave en [DDPM](/papers/ddpm-ho-2020) es que descartar los pesos infrapondera los $t$ pequeños (donde denoising es trivial) y concentra la capacidad en los $t$ grandes (donde es difícil) — ese reweighting implícito es lo que lleva el FID de 13.5 a 3.17 en CIFAR-10.

{{< concept-alert type="importante" >}}
La parametrización canónica de DDPM no predice la imagen ni la media: predice el **ruido $\epsilon$**. La pérdida $L_\text{simple} = \mathbb{E}\,\|\epsilon - \epsilon_\theta(x_t, t)\|^2$ es un MSE entre el ruido real y el predicho. Esta elección, junto con descartar los pesos del bound variacional, es la decisión de ingeniería que volvió la difusión competitiva con las GANs.
{{< /concept-alert >}}

---

## 4. La U-Net como denoiser

¿Qué arquitectura es $\epsilon_\theta$? Una **U-Net** ([Ronneberger et al., 2015](/papers/unet-ronneberger-2015)), la red encoder-decoder con *skip connections* nacida para segmentación biomédica. La elección no es accidental: tres propiedades la hacen ideal para denoising.

**Multi-escala.** La U-Net comprime la entrada por una rama descendente (encoder) hasta un cuello de botella de baja resolución, y la reconstruye por una rama ascendente (decoder). Esto le permite razonar simultáneamente sobre **estructura global** (qué objeto hay, su composición) en las capas profundas de baja resolución, y **detalle local** (texturas, bordes) en las capas superficiales de alta resolución. Quitar ruido bien exige justo esa doble vista: saber qué se está reconstruyendo y cómo deben verse sus texturas.

**Skip connections.** Las conexiones que saltan del encoder al decoder a cada resolución reinyectan los detalles de alta frecuencia que el cuello de botella habría perdido. Esto es esencial: la salida de $\epsilon_\theta$ tiene la **misma forma espacial** que la entrada (es un mapa de ruido del tamaño de la imagen), y los *skips* permiten que esa salida conserve el detalle fino del input.

**Embedding de timestep.** Una sola red sirve para los $T=1000$ niveles de ruido. ¿Cómo sabe en qué paso está? El timestep $t$ se codifica con un **embedding sinusoidal** estilo Transformer y se inyecta en cada bloque residual de la U-Net. Esto deja que la misma red adapte su comportamiento según el nivel de ruido: con mucho ruido se concentra en recuperar estructura global; con poco, en pulir texturas. DDPM añade además **auto-atención** en la resolución $16\times16$ para capturar dependencias de largo alcance que la convolución pura no alcanza. Esta arquitectura —U-Net + embedding de tiempo + atención— se ha mantenido como la columna vertebral de la difusión hasta hoy.

---

## 5. La perspectiva score-based: la otra cara de la misma moneda

Existe una segunda derivación de la difusión, desarrollada en paralelo por [Song y Ermon (2019)](/papers/score-based-song-2019), que llega al mismo lugar por un camino distinto y revela qué está aprendiendo la red en un sentido más profundo. En vez de modelar la densidad $p(x)$ —que obliga a calcular una constante de normalización intratable— se aprende el **score**: el gradiente del logaritmo de la densidad,

$$
s_\theta(x) \approx \nabla_x \log p(x).
$$

El score es un campo vectorial que en cada punto apunta hacia donde la densidad de los datos crece más rápido. Si uno conoce ese campo, puede *navegar* hacia las regiones de alta densidad y generar muestras, sin conocer nunca la constante de normalización (que se cancela al tomar el gradiente del log). Se estima con **denoising score matching**: perturbar los datos con ruido gaussiano y entrenar la red a estimar el score de la distribución perturbada — un objetivo que, notablemente, no requiere conocer $p(x)$.

El muestreo se hace con **dinámica de Langevin recocida** (*annealed Langevin*): se arranca de ruido y se itera $x \leftarrow x + \frac{\epsilon}{2}\nabla_x \log p(x) + \sqrt{\epsilon}\,z$ —un descenso de gradiente sobre la log-densidad con ruido inyectado— recorriendo niveles de ruido de alto a bajo. El "recocido" (empezar en ruido alto, donde el campo es suave, y bajar gradualmente) resuelve dos patologías del score ingenuo: que está indefinido sobre variedades de baja dimensión, y que carece de señal en regiones de baja densidad. Esto explica *por qué* la difusión necesita una secuencia completa de niveles de ruido y no uno solo.

La conexión profunda: el score de un kernel gaussiano $\mathcal{N}(x_0, \sigma^2 I)$ es $-(x_t - x_0)/\sigma^2 = -\epsilon/\sigma$. Es decir, **predecir el ruido $\epsilon$ (DDPM) y predecir el score $\nabla_x \log p$ (NCSN) son el mismo objetivo hasta un factor de escala**. La $\epsilon_\theta$ de DDPM y la $s_\theta$ de Song-Ermon aprenden lo mismo. Song et al. (2021) formalizaron esto con **ecuaciones diferenciales estocásticas (SDE)**: el forward de DDPM (la SDE "variance preserving") y el de NCSN (la SDE "variance exploding") son dos discretizaciones de una misma familia continua, y muestrear es resolver la SDE/ODE de reversión guiada por el score. Dos comunidades que trabajaban por separado resultaron estar haciendo lo mismo.

{{< concept-alert type="nota" >}}
DDPM y score-based no son dos modelos: son **dos vistas del mismo proceso**. "Predecir el ruido" y "estimar el gradiente de la log-densidad" coinciden hasta un factor $1/\sigma$. El marco de SDEs (Song 2021) las unifica: el forward es una SDE que añade ruido, el reverse es la SDE de reversión que la red resuelve usando el score aprendido.
{{< /concept-alert >}}

---

## 6. Condicionamiento y guidance: hacer que el texto mande

Un modelo de difusión incondicional genera muestras de la distribución de datos, pero no se le puede pedir "un gato astronauta". Para eso hace falta **condicionamiento**: pasarle a la red una señal $c$ (una clase, un texto codificado) además de $x_t$ y $t$, de modo que aprenda $\epsilon_\theta(x_t, t, c)$. Pero condicionar a secas resulta débil: el modelo tiende a ignorar parcialmente $c$. La técnica que lo arregla es la **guidance**.

La primera versión, **classifier guidance** (Dhariwal & Nichol, 2021), mezcla el score del modelo con el gradiente de un clasificador externo $p(c \mid x_t)$, empujando la generación hacia donde el clasificador asigna alta probabilidad a la clase deseada. Funciona, pero tiene tres problemas: hay que entrenar un clasificador extra, ese clasificador debe entrenarse sobre **imágenes ruidosas** (no sirve uno preentrenado estándar), y dar pasos en la dirección del gradiente del clasificador se parece sospechosamente a un ataque adversarial.

La solución que se impuso es **classifier-free guidance** ([Ho & Salimans, 2022](/papers/classifier-free-guidance-ho-2022)), que elimina el clasificador por completo. La receta tiene dos cambios de una línea. En **entrenamiento**: se entrena una sola red que aprende a la vez el modelo condicional $\epsilon_\theta(x_t, c)$ y el incondicional $\epsilon_\theta(x_t)$, simplemente reemplazando $c$ por un **token nulo** $\varnothing$ con cierta probabilidad (un *dropout* del condicionamiento, típicamente del 10-20%). En **muestreo**: se combinan ambas estimaciones linealmente, extrapolando en la dirección que va de lo incondicional a lo condicional:

$$
\tilde\epsilon_\theta(x_t, c) = \epsilon_\theta(x_t) + s\,\big(\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t)\big).
$$

El factor $s$ es la **guidance scale**. El término $\big(\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t)\big)$ es exactamente "lo que el texto aporta" sobre la generación a ciegas, y multiplicarlo por $s>1$ lo amplifica. Por eso subir la guidance scale hace que la imagen obedezca cada vez más al prompt. El trade-off es monótono: más guidance da más fidelidad al prompt pero **menos diversidad** (y, si se exagera, colores saturados y artefactos). Es la perilla más usada de la generación de imágenes: el `guidance_scale` que se pasa a la pipeline de `diffusers` (típicamente 7-12 en Stable Diffusion) **es literalmente este $s$**. El precio computacional: cada paso de muestreo evalúa la red **dos veces**, una condicional y una incondicional. El truco del *negative prompt* es una generalización directa: restar el score de un prompt "negativo" empuja la imagen *lejos* de ese concepto.

---

## 7. Latent diffusion: difundir en el espacio comprimido

La difusión en píxeles es cara: opera en un espacio de altísima dimensión (una imagen de $512\times512\times3$ tiene ~786k dimensiones) y desperdicia capacidad modelando detalles imperceptibles de alta frecuencia. Entrenar un modelo potente costaba cientos de días de GPU. La solución de **Latent Diffusion** ([Rombach et al., 2022](/papers/latent-diffusion-rombach-2022)) —el paper que dio origen a Stable Diffusion— es **mover la difusión a un espacio latente comprimido**.

La arquitectura tiene dos fases. **Fase 1**: un autoencoder (encoder $\mathcal{E}$ + decoder $\mathcal{D}$, entrenado con pérdida perceptual LPIPS y un objetivo adversarial PatchGAN, regularizado al estilo VAE) comprime la imagen $x$ a un latente $z = \mathcal{E}(x)$ de mucha menor dimensión (factor de downsampling $f \in \{4, 8\}$ es el punto dulce), perceptualmente equivalente pero abstrayendo el detalle de alta frecuencia. Este autoencoder se entrena **una sola vez** y se congela. **Fase 2**: el modelo de difusión —el mismo DDPM, con la misma $L_\text{simple}$ y la misma U-Net— se entrena en el **espacio latente** en vez de en píxeles:

$$
L_\text{LDM} = \mathbb{E}_{\mathcal{E}(x),\,\epsilon,\,t}\big[\|\epsilon - \epsilon_\theta(z_t, t)\|_2^2\big].
$$

Como el latente conserva su estructura espacial 2D, la U-Net convolucional sigue siendo el backbone natural. Al generar, se difunde en el latente y se decodifica el resultado a píxeles con **una sola pasada** por $\mathcal{D}$. La reducción de cómputo es lo que permitió que Stable Diffusion corriera en una GPU de consumo con ~8-10 GB de VRAM, democratizando la generación. La segunda contribución del paper es el **condicionamiento por cross-attention**: un encoder de dominio $\tau_\theta$ (el encoder de texto de CLIP en Stable Diffusion) proyecta el prompt, y la U-Net atiende a esa representación vía capas de cross-attention, donde las *queries* vienen del estado visual y las *keys/values* del texto. El flujo completo de Stable Diffusion es: prompt → CLIP → ruido latente → denoising condicionado por cross-attention (con classifier-free guidance) → decoder VAE → imagen.

---

## 8. El trade-off: calidad y cobertura altas, pero muestreo lento

Para situar la difusión frente a sus rivales conviene el marco del **trilema del aprendizaje generativo** ([Xiao et al., 2021](/papers/diffusion-gan-xiao-2021)): un modelo generativo querría a la vez (1) alta calidad de muestras, (2) muestreo rápido y (3) buena cobertura de modos/diversidad, pero casi siempre sacrifica uno por los otros dos.

| Familia | Velocidad de muestreo | Calidad de muestra | Cobertura / diversidad |
|---|---|---|---|
| **GAN** | Rápida (un *forward pass*) | Alta | **Baja** (riesgo de *mode collapse*) |
| **VAE** | Rápida | **Media** (muestras borrosas) | Alta |
| **Difusión** | **Lenta** (miles de pasos) | Alta | Alta |

La difusión combina la **alta calidad** de las GANs con la **alta cobertura** de los modelos de verosimilitud (sin el *mode collapse* que aqueja a las GANs), pero paga un único precio brutal: el **muestreo es lentísimo**. Generar una imagen requiere $T$ (hasta 1000) pasos secuenciales, cada uno una evaluación completa de la U-Net, y no se pueden paralelizar porque cada $x_{t-1}$ depende de $x_t$. Frente a una GAN que genera en una pasada, la difusión es órdenes de magnitud más lenta en inferencia. Ese único vértice débil —la velocidad— concentró casi toda la investigación posterior:

- **DDIM** (Song et al., 2021) reformula el muestreo como un proceso determinista no-Markoviano que produce muestras de calidad comparable con **muchos menos pasos** (50 en vez de 1000), saltándose pasos sin reentrenar. Es el sampler por defecto de muchas pipelines.
- **Denoising diffusion GANs** ([Xiao et al., 2021](/papers/diffusion-gan-xiao-2021)) atacan la raíz: la lentitud viene del supuesto gaussiano sobre el paso de denoising, válido solo para pasos infinitesimales. Si se quieren pasos *grandes* (pocos pasos), la distribución verdadera se vuelve **multimodal**, y la modelan con un GAN condicional por paso. Resultado: ~4 pasos en vez de 1000, ~2000× más rápido, manteniendo calidad y cobertura.
- Otras líneas: destilación progresiva, solvers de ODE de alto orden, y modelos de consistencia, todas apuntando al mismo vértice.

{{< concept-alert type="recordar" >}}
La difusión gana en calidad y diversidad, y pierde en velocidad. Esa lentitud no es accidental: viene del supuesto gaussiano del paso de denoising, válido solo con pasos infinitesimales (Feller, 1949), de ahí los miles de pasos. Toda la investigación de aceleración (DDIM, diffusion-GANs, destilación) ataca *ese único vértice*, dejando intactas las dos ventajas.
{{< /concept-alert >}}

---

## 9. Conexión con la Clase 29 y resumen

La [Clase 29 — Modelos Generativos en Visión](/clases/clase-29) presenta la difusión como el destino del arco del módulo: tras VAEs y GANs, es la familia que termina dominando la generación de imágenes. El resumen de la clase —"convertir datos en ruido gradualmente y aprender a invertir paso a paso, con la U-Net como aprendiz del paso inverso"— es exactamente la maquinaria desarrollada aquí. El laboratorio usa Stable Diffusion vía `diffusers`, donde el `guidance_scale` es el $s$ de la §6 y el VAE/U-Net/CLIP son las piezas de la §7.

Los puntos a retener:

- **Forward** (§2): cadena de Markov fija que añade ruido gaussiano; la fórmula cerrada $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ permite saltar a cualquier $t$ y hace tratable el entrenamiento.
- **Reverse** (§3): la red aprende a invertir el ruido; la parametrización clave de [DDPM](/papers/ddpm-ho-2020) es **predecir el ruido $\epsilon$** con la pérdida $L_\text{simple}$.
- **U-Net** (§4): denoiser multi-escala con *skips* y embedding de timestep; columna vertebral de toda la difusión.
- **Score-based** (§5): [DDPM y NCSN](/papers/score-based-song-2019) son la misma cosa vista como predicción de ruido vs. estimación del score; unificadas por SDEs.
- **Guidance** (§6): [classifier-free guidance](/papers/classifier-free-guidance-ho-2022) hace que el prompt mande, con la guidance scale como perilla fidelidad/diversidad.
- **Latent diffusion** (§7): [difundir en el latente comprimido](/papers/latent-diffusion-rombach-2022) abarata todo y da Stable Diffusion.
- **Trade-off** (§8): calidad y cobertura altas, muestreo lento; [DDIM y diffusion-GANs](/papers/diffusion-gan-xiao-2021) aceleran.

---

## Para profundizar

- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](/papers/ddpm-ho-2020) — el acta de nacimiento práctica de la difusión: $\epsilon$-prediction, $L_\text{simple}$, U-Net con embedding de tiempo.
- [Generative Modeling by Estimating Gradients of the Data Distribution (Song & Ermon, 2019)](/papers/score-based-song-2019) — la perspectiva score-based: denoising score matching y Langevin recocido.
- [Classifier-Free Diffusion Guidance (Ho & Salimans, 2022)](/papers/classifier-free-guidance-ho-2022) — la guidance scale sin clasificador, el motor del condicionamiento por texto.
- [High-Resolution Image Synthesis with Latent Diffusion Models (Rombach et al., 2022)](/papers/latent-diffusion-rombach-2022) — difusión en el latente + cross-attention = Stable Diffusion.
- [Tackling the Generative Learning Trilemma with Denoising Diffusion GANs (Xiao et al., 2021)](/papers/diffusion-gan-xiao-2021) — el trilema velocidad/calidad/diversidad y el muestreo en pocos pasos.
- [U-Net (Ronneberger et al., 2015)](/papers/unet-ronneberger-2015) — la arquitectura encoder-decoder que sirve de denoiser $\epsilon_\theta$.

**Laboratorio:** [Lab 29 — Stable Diffusion con diffusers](/laboratorios/lab-29) manipula en la práctica los pasos de denoising, los schedulers y la guidance scale, y explora Img2Img/Inpainting/ControlNet.

**Fundamentos relacionados:** [Modelos Generativos](/fundamentos/modelos-generativos) · [Clase 29 — Modelos Generativos en Visión](/clases/clase-29)
