# High-Resolution Image Synthesis with Latent Diffusion Models — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *High-Resolution Image Synthesis with Latent Diffusion Models*.
- **Autores:** Robin Rombach (Ludwig Maximilian University of Munich e IWR, Heidelberg University), Andreas Blattmann (íd.), Dominik Lorenz (íd.), Patrick Esser (Runway ML) y Björn Ommer (íd.). Los dos primeros autores contribuyeron por igual. El grupo de Ommer (el *CompVis Group*, antes en Heidelberg, luego en Múnich) es el mismo que había publicado VQGAN ("Taming Transformers", 2020).
- **Venue:** CVPR 2022 (*Conference on Computer Vision and Pattern Recognition*).
- **Año:** 2021–2022. **Preprint:** arXiv:2112.10752v2 (versión 2 del 13 abr 2022; v1 de diciembre 2021), [arxiv.org/abs/2112.10752](https://arxiv.org/abs/2112.10752).
- **Código y pesos:** [github.com/CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion) — modelos pre-entrenados de difusión latente y de autoencoding liberados públicamente.

Este es **el paper fundacional de Stable Diffusion**, el modelo generativo de imágenes más usado del mundo. La tesis es simultáneamente sencilla de enunciar y profunda en consecuencias: los modelos de difusión (DDPM y descendientes) producen imágenes de calidad estado del arte, pero **operan directamente en el espacio de píxeles** —un espacio de altísima dimensión—, lo que los vuelve carísimos de entrenar (cientos de días de GPU; "150–1000 días-V100" para los modelos más potentes según Dhariwal & Nichol) y lentos de samplear (un modelo grande tarda ~5 días en producir 50 000 muestras en una sola A100). El paper propone mover la difusión a un **espacio latente comprimido** aprendido por un autoencoder pre-entrenado. Como ese espacio es perceptualmente equivalente a la imagen pero de mucha menor dimensión, el modelo de difusión deja de gastar capacidad y cómputo modelando detalles imperceptibles de alta frecuencia y se concentra en la estructura semántica. El resultado es una reducción drástica del costo de entrenamiento e inferencia *sin* degradar la calidad — y, en varias tareas, mejorándola.

La segunda contribución, igual de consecuente, es un **mecanismo de condicionamiento de propósito general basado en cross-attention** inyectado en la U-Net. Esto convierte al modelo de difusión en un generador flexible que puede condicionarse en texto, clases, layouts (bounding boxes), mapas semánticos o imágenes de baja resolución — todo con la misma arquitectura, sin diseños específicos por tarea. Es la combinación de estas dos ideas —**difusión en el latente** + **condicionamiento por cross-attention**— la que define a los *Latent Diffusion Models* (LDM) y, escalada con un encoder de texto, da origen a Stable Diffusion.

Para la **Clase 29 (Modelos Generativos en Visión)** este paper es el destino del arco completo del curso: VAE da el autoencoder de compresión, VQGAN da la versión adversarial-perceptual de ese autoencoder, DDPM da el proceso de difusión, la U-Net da el backbone, classifier-free guidance da el control, y CLIP (en Stable Diffusion) da el encoder de texto. Las slides finales de la clase lo presentan literalmente como "VAE + U-Net con cross-attention + Text encoder (CLIP)", con el flujo prompt → CLIP → ruido latente → denoising condicionado → decoder VAE → imagen.

## 2. Contexto histórico: el muro de cómputo de la difusión en píxeles

Hacia 2021 la síntesis de imágenes de alta resolución estaba dominada por dos familias, cada una con su talón de Aquiles. Las **GAN** (Goodfellow et al., 2014; BigGAN, StyleGAN) samplean rápido y con buena calidad perceptual, pero son difíciles de optimizar, sufren inestabilidades y *mode collapse*, y "no escalan fácilmente a modelar distribuciones complejas, multimodales" — quedan confinadas a datos de variabilidad relativamente limitada (rostros, una clase). Los **modelos basados en verosimilitud** (VAE, flows, modelos autoregresivos) tienen mejor estimación de densidad y optimización más estable, pero los autoregresivos como los *transformers* de DALL-E o VQGAN+Transformer requieren miles de millones de parámetros y un sampleo secuencial costoso, mientras que VAE y flows no alcanzan la calidad de las GAN.

Los **modelos de difusión** (DM), introducidos por Sohl-Dickstein et al. (2015) y vueltos prácticos por DDPM (Ho et al., 2020) y *Score-based SDEs* (Song et al., 2020), aparecieron como la síntesis ganadora: al descomponer la generación en una secuencia de *denoising autoencoders*, logran estado del arte en calidad de muestra (Dhariwal & Nichol, "Diffusion Models Beat GANs", 2021) y, al ser basados en verosimilitud, evitan el *mode collapse* y las inestabilidades de las GAN sin necesitar miles de millones de parámetros, gracias al fuerte *parameter sharing* del proceso de denoising. Tienen además una ventaja decisiva: su formulación admite un mecanismo de guía (*guidance*) para controlar la generación sin reentrenar, y modelos incondicionales pueden aplicarse a inpainting, colorización o síntesis basada en trazos.

El problema, central para este paper, es el **costo**. Los DM son modelos basados en verosimilitud con comportamiento *mode-covering*: tienden a gastar cantidades excesivas de capacidad —y por tanto de cómputo— modelando detalles imperceptibles de los datos. El objetivo variacional reponderado de DDPM mitiga esto submuestreando los pasos iniciales de denoising, pero los DM siguen siendo costosos: **entrenar** requiere evaluaciones repetidas de la red y cálculo de gradientes en el espacio de alta dimensión de imágenes RGB, e **inferir** exige correr la misma arquitectura secuencialmente por decenas o cientos de pasos (25–1000). Esto tiene dos consecuencias: entrenar un DM potente solo está al alcance de una fracción pequeña del campo (dejando además una huella de carbono enorme), y evaluar uno ya entrenado es caro en tiempo y memoria. Reducir esa carga sin sacrificar calidad era, en palabras del paper, "clave para mejorar la accesibilidad" — democratizar la síntesis de alta resolución.

El punto de partida del método es un análisis del *trade-off* tasa-distorsión de un DM ya entrenado (Fig. 2 del paper). El aprendizaje de cualquier modelo basado en verosimilitud puede dividirse en dos etapas: una primera de **compresión perceptual**, que elimina detalles de alta frecuencia pero aprende poca variación semántica; y una segunda de **compresión semántica**, donde el modelo generativo aprende la composición conceptual de los datos. Los DM gastan gradientes y pasos de red en *toda* la primera etapa, sobre cada píxel, lo que es desperdicio computacional. La idea: encontrar primero un espacio perceptualmente equivalente pero computacionalmente más conveniente, y entrenar la difusión ahí.

## 3. Contribución central

El paper hace, según su propia enumeración, seis contribuciones. Las esenciales:

1. **Escalabilidad graciosa frente a transformers puros.** A diferencia de los enfoques puramente basados en transformers (DALL-E, VQGAN+Transformer), el método de LDM escala más suavemente a datos de mayor dimensión gracias a su backbone convolucional. Esto permite (a) trabajar en un nivel de compresión que da reconstrucciones más fieles y detalladas que trabajos previos, y (b) aplicar el modelo a síntesis de alta resolución de imágenes de megapíxeles.

2. **Rendimiento competitivo a costo mucho menor.** LDM logra rendimiento competitivo o estado del arte en múltiples tareas (síntesis incondicional, inpainting, super-resolución estocástica) y datasets, *bajando significativamente los costos de cómputo* tanto de entrenamiento como —de forma notable— de inferencia, frente a los enfoques de difusión en píxeles.

3. **Separación limpia compresión/generación.** A diferencia de trabajos previos que aprenden *conjuntamente* un autoencoder y un *prior* score-based (LSGM, Vahdat et al.), el enfoque de LDM **no requiere un balanceo delicado** entre capacidad de reconstrucción y capacidad generativa. El autoencoder se entrena una vez, por separado, asegurando reconstrucciones extremadamente fieles y requiriendo muy poca regularización del espacio latente.

4. **Sampleo convolucional para tareas densamente condicionadas.** Para super-resolución, inpainting y síntesis semántica, el modelo puede aplicarse de forma convolucional y generar imágenes grandes y consistentes de ~1024² px.

5. **Mecanismo de condicionamiento de propósito general por cross-attention** (la pieza que abre el multimodal — ver §4.3).

6. **Liberación de modelos pre-entrenados** de difusión latente y de autoencoding, reutilizables para tareas más allá del entrenamiento de DM.

La idea de diseño que une todo es la **separación explícita de la fase de compresión y la fase generativa**. Como el autoencoder universal se entrena una sola vez, puede reutilizarse para múltiples entrenamientos de DM o para tareas completamente distintas — exactamente lo que ocurrió en la práctica: el mismo VAE de Stable Diffusion sirvió de cimiento a innumerables modelos derivados.

## 4. Método

### 4.1. Fase 1 — compresión perceptual con un autoencoder adversarial-perceptual

Dada una imagen $x \in \mathbb{R}^{H \times W \times 3}$ en RGB, un **encoder** $\mathcal{E}$ la mapea a una representación latente $z = \mathcal{E}(x)$, y un **decoder** $\mathcal{D}$ reconstruye $\tilde{x} = \mathcal{D}(z) = \mathcal{D}(\mathcal{E}(x))$, con $z \in \mathbb{R}^{h \times w \times c}$. El encoder *downsamplea* la imagen por un **factor $f = H/h = W/w$**, donde se investigan factores $f = 2^m$.

La clave de la fidelidad está en *cómo* se entrena este autoencoder. No se usa una pérdida pixel-wise (L2/L1) sola, porque esas pérdidas producen reconstrucciones borrosas. En su lugar, el autoencoder —basado en el trabajo previo de VQGAN (Esser et al.)— se entrena con una combinación de:

- una **pérdida perceptual** (LPIPS, Zhang et al. 2018), que compara características profundas en vez de píxeles crudos; y
- un **objetivo adversarial basado en parches** (un discriminador PatchGAN), que fuerza el realismo local y confina las reconstrucciones a la variedad (*manifold*) de imágenes naturales, evitando el desenfoque.

Para evitar latentes de varianza arbitrariamente alta, se exploran **dos regularizaciones**:

- **KL-reg.**: una penalización KL leve hacia una normal estándar sobre el latente aprendido, "similar a un VAE" (de ahí que en Stable Diffusion se hable directamente del "VAE").
- **VQ-reg.**: una capa de cuantización vectorial (VQ) dentro del decoder. Este modelo se interpreta como un VQGAN pero con la capa de cuantización absorbida por el decoder.

A diferencia de trabajos previos (VQGAN, DALL-E) que dependían de una compresión espacial *agresiva* y de un ordenamiento 1D del latente para modelarlo autoregresivamente —ignorando su estructura espacial—, aquí el DM posterior está diseñado para trabajar con la **estructura bidimensional** del latente $z$. Esto permite usar tasas de compresión relativamente *suaves* y obtener reconstrucciones muy buenas, preservando los detalles de $x$ mejor que los enfoques anteriores. Empíricamente, los factores $f \in \{4, 8\}$ resultan ser el punto dulce (la Fig. 1 muestra que con $f=4$ el LDM alcanza R-FID de 0.58 y PSNR 27.4 frente a 32.01 / 22.8 de DALL-E con $f=8$ y 4.98 / 19.9 de VQGAN con $f=16$).

### 4.2. Fase 2 — difusión en el espacio latente

Un modelo de difusión aprende $p(x)$ denoising gradualmente una variable normal, lo que equivale a aprender el proceso inverso de una cadena de Markov fija de longitud $T$. Se interpreta como una secuencia de *denoising autoencoders* $\epsilon_\theta(x_t, t)$, $t=1\dots T$, entrenados para predecir el ruido de una versión ruidosa $x_t$ de la entrada. El objetivo reponderado de DDPM es:

$$L_{DM} = \mathbb{E}_{x, \epsilon \sim \mathcal{N}(0,1), t}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|_2^2\right]$$

La innovación de LDM es **trasladar este objetivo al latente**. Con $\mathcal{E}$ y $\mathcal{D}$ ya entrenados y congelados, se tiene acceso a un espacio de baja dimensión donde los detalles imperceptibles de alta frecuencia ya están abstraídos. El objetivo se vuelve:

$$L_{LDM} = \mathbb{E}_{\mathcal{E}(x), \epsilon \sim \mathcal{N}(0,1), t}\left[\|\epsilon - \epsilon_\theta(z_t, t)\|_2^2\right]$$

El backbone $\epsilon_\theta(\circ, t)$ es una **U-Net condicionada en el tiempo** (Ronneberger et al., 2015). Dos consecuencias prácticas: como el proceso *forward* es fijo, $z_t$ se obtiene eficientemente de $\mathcal{E}$ durante el entrenamiento; y las muestras de $p(z)$ se decodifican a imagen con **una sola pasada** por $\mathcal{D}$. La U-Net, construida principalmente con convoluciones 2D, aprovecha el sesgo inductivo apropiado para datos con estructura espacial — y como el latente conserva esa estructura 2D, ese sesgo sigue siendo útil. Esto es lo que elimina la necesidad de la compresión agresiva que exigían los enfoques autoregresivos.

El paper enumera explícitamente las ventajas de salir del espacio de píxeles: (i) al trabajar en un espacio de baja dimensión, los DM son computacionalmente mucho más eficientes porque el sampleo ocurre en dimensión reducida; (ii) se explota el sesgo inductivo que los DM heredan de su arquitectura U-Net, particularmente efectivo para datos con estructura espacial, lo que alivia la necesidad de niveles de compresión agresivos que degradan la calidad (como exigían VQGAN o DALL-E); y (iii) se obtienen modelos de compresión de propósito general cuyo espacio latente puede usarse para entrenar múltiples modelos generativos y para otras aplicaciones aguas abajo, como la síntesis guiada por CLIP de una sola imagen. Esta reutilización del autoencoder universal —entrenado una sola vez— es lo que hace eficiente la exploración de un gran número de modelos de difusión para tareas distintas de imagen-a-imagen y texto-a-imagen.

Es importante subrayar el contraste con el trabajo previo que motivó la separación de fases. LSGM (Vahdat et al.) aprendía el autoencoder y el *prior* score-based *conjuntamente*, lo que obliga a un ponderado delicado entre reconstrucción y capacidad generativa; y los enfoques autoregresivos sobre latentes discretos (VQGAN, DALL-E) imponían un orden 1D arbitrario sobre $z$ para modelarlo, ignorando su estructura inherente. LDM evita ambos problemas: la primera etapa se fija de antemano, garantizando reconstrucciones fieles con regularización mínima, y el latente conserva su naturaleza 2D para que la difusión convolucional lo modele de forma natural.

### 4.3. Mecanismo de condicionamiento por cross-attention

Para modelar distribuciones condicionales $p(z \mid y)$ —donde $y$ puede ser texto, mapas semánticos, layouts u otra imagen—, se aumenta la U-Net con **capas de cross-attention** (Vaswani et al., 2017). El condicionamiento se procesa así:

1. Un **encoder específico de dominio** $\tau_\theta$ proyecta la condición $y$ a una representación intermedia $\tau_\theta(y) \in \mathbb{R}^{M \times d_\tau}$. Para texto, $\tau_\theta$ es un transformer (en este paper, con tokenizador BERT; en Stable Diffusion, el encoder de texto de **CLIP**).
2. Esa representación se inyecta en las capas intermedias de la U-Net mediante cross-attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) \cdot V$$

donde las *queries* vienen del estado interno de la U-Net, $Q = W_Q^{(i)} \cdot \varphi_i(z_t)$, y las *keys* y *values* vienen de la condición, $K = W_K^{(i)} \cdot \tau_\theta(y)$, $V = W_V^{(i)} \cdot \tau_\theta(y)$, con matrices de proyección aprendibles.

El objetivo condicional se vuelve $L_{LDM} = \mathbb{E}_{\mathcal{E}(x), y, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(z_t, t, \tau_\theta(y))\|_2^2\right]$, donde $\tau_\theta$ y $\epsilon_\theta$ se optimizan conjuntamente. La elegancia es que $\tau_\theta$ puede ser cualquier *experto de dominio*: cambiar la modalidad de la condición es cambiar el encoder, no la arquitectura de difusión. Para condiciones **espacialmente alineadas** (mapas semánticos, imágenes de baja resolución para super-resolución), el paper usa además un mecanismo más simple: **concatenar** la condición *downsampleada* al input de la U-Net (Fig. 3 muestra ambos: concatenación vs. cross-attention).

### 4.4. Los factores de downsampling y el balance de compresión

El análisis empírico (§4.1 del paper) barre $f \in \{1, 2, 4, 8, 16, 32\}$ (LDM-$f$, donde LDM-1 es difusión en píxeles). El hallazgo central:

- $f$ **pequeño** (LDM-1, LDM-2): entrenamiento lento, porque se deja casi toda la compresión perceptual al modelo de difusión (desperdicio de cómputo).
- $f$ **demasiado grande** (LDM-32): la fidelidad se estanca tras pocos pasos, porque una primera etapa demasiado comprimida pierde información y limita la calidad alcanzable.
- $f \in \{4, \dots, 16\}$, y en particular **LDM-4 y LDM-8**: el punto óptimo entre eficiencia y resultados perceptualmente fieles. Tras 2M pasos en ImageNet hay una brecha de FID de 38 puntos entre la difusión en píxeles (LDM-1) y LDM-8.

## 5. Experimentos

El paper fija recursos computacionales (una sola NVIDIA A100) para comparaciones limpias y demuestra LDM en una batería amplia de tareas.

**Generación incondicional.** En CelebA-HQ 256², LDM-4 establece un nuevo estado del arte con **FID 5.11**, superando modelos basados en verosimilitud previos y a las GAN, e incluso a LSGM (donde el latente y la difusión se entrenan conjuntamente). En FFHQ logra FID 4.98, en LSUN-Churches 4.02 y en LSUN-Bedrooms 2.95 (cercano a ADM pero con la mitad de parámetros y 4× menos recursos de entrenamiento). LDM mejora consistentemente sobre las GAN en *Precision* y *Recall*, confirmando la ventaja del entrenamiento *mode-covering* basado en verosimilitud sobre el adversarial.

**Síntesis condicional por clase (ImageNet).** LDM-4 con classifier-free guidance (LDM-4-G) alcanza **FID 3.60** con 400M parámetros, superando a ADM-G (FID 4.59, 608M parámetros) — estado del arte con menos cómputo y menos parámetros.

**Texto-a-imagen (el camino a Stable Diffusion).** Se entrena un LDM-KL-8 de **1.45B parámetros** condicionado en prompts de lenguaje sobre **LAION-400M** (400 millones de pares imagen-texto filtrados por CLIP), con $\tau_\theta$ implementado como transformer. En MS-COCO 256², con 250 pasos DDIM y classifier-free guidance, LDM-KL-8-G logra FID 12.63, **a la par de los métodos AR y de difusión más recientes** de la época (GLIDE de 6B parámetros, Make-A-Scene de 4B) pero con sustancialmente menos parámetros. La Fig. 5 muestra prompts arbitrarios ("A street sign that reads 'Latent Diffusion'", "An illustration of a slightly conscious neural network") generados con calidad notable.

**Layout-to-image y síntesis semántica.** Gracias a la cross-attention y a la concatenación, el modelo se entrena para sintetizar a partir de layouts (bounding boxes en COCO/OpenImages) y mapas semánticos de paisajes. Entrenado a 256² pero aplicado de forma convolucional, generaliza a resoluciones de megapíxel (Fig. 9: paisajes a 512×1024).

**Super-resolución.** Condicionando por concatenación de la imagen de baja resolución, LDM-SR (×4, siguiendo el protocolo de SR3) **supera a SR3 en FID** con bastante menos parámetros (169M vs. 625M). Un estudio con humanos confirma la preferencia por LDM-SR; PSNR/SSIM favorecen modelos más borrosos, lo que el paper nota que no se alinea con la percepción humana.

**Inpainting.** Establece un nuevo estado del arte en Places, superando a LaMa (que usa Fast Fourier Convolutions especializadas) en FID. Crucialmente, mide la **eficiencia**: entre difusión en píxeles (LDM-1) y difusión latente (LDM-4) hay un *speed-up* de al menos **2.7×** mejorando el FID en al menos 1.6×. El estudio de eficiencia (Tab. 6 del paper) cuantifica el contraste: LDM-1 sin primera etapa entrega 0.11 muestras/s de *throughput* de entrenamiento y tarda 20.66 horas por época, mientras que LDM-4 (VQ, sin atención) entrega 0.35 muestras/s y solo 6.66 horas por época, con mejor FID. El paper observa además que las reconstrucciones del LDM producen resultados *diversos* (a diferencia de LaMa, que tiende a recuperar una imagen promedio), lo que se refleja en un LPIPS algo mayor pero en una preferencia humana superior. Tras entrenar un modelo grande (387M parámetros) en el latente VQ sin atención y hacer *fine-tuning* a 512², se fija un nuevo estado del arte de FID en inpainting. Un estudio con humanos favorece los resultados de LDM sobre los de LaMa tanto en preferencia frente al *ground truth* como en comparación directa entre muestras generadas.

**Eficiencia global.** Comparando velocidad de sampleo con el sampler DDIM contra FID (Fig. 7), LDM-{4-8} no solo logran FID mucho menores que LDM-1 (difusión en píxeles) sino que aumentan significativamente el *throughput* de muestras. Este es el argumento central del paper: **misma o mejor calidad, fracción del cómputo.**

## 6. Limitaciones reconocidas

El paper es explícito en su sección de limitaciones e impacto social:

- **Sampleo todavía secuencial.** Aunque LDM reduce drásticamente el costo frente a la difusión en píxeles, su proceso de sampleo secuencial **sigue siendo más lento que el de las GAN** (que generan en una pasada). La difusión paga su estabilidad y cobertura de modos con latencia de inferencia.
- **El autoencoder como cuello de botella de precisión.** Cuando se requiere alta precisión a nivel de píxel, la capacidad de reconstrucción del autoencoder ($f=4$) puede ser un límite, aun cuando la pérdida de calidad sea pequeña. El paper asume que sus modelos de super-resolución ya están algo limitados por esto. Es decir, *nada* puede recuperarse mejor de lo que el decoder es capaz de reconstruir.
- **Impacto social (doble filo).** Los modelos generativos de medios son un arma de doble filo: democratizan el acceso a la tecnología pero también facilitan crear y diseminar datos manipulados, desinformación y spam — en particular *deep fakes*, que afectan desproporcionadamente a las mujeres. Los modelos generativos pueden además **revelar datos de entrenamiento** (preocupante con información sensible recolectada sin consentimiento), y tienden a reproducir o exacerbar **sesgos** presentes en los datos. El paper señala que en qué medida su enfoque de dos etapas (que combina entrenamiento adversarial con objetivo de verosimilitud) tergiversa los datos sigue siendo una pregunta de investigación abierta.

## 7. Impacto: Stable Diffusion y la democratización de la síntesis

El impacto de este paper es difícil de exagerar. La combinación que describe —**VAE de compresión + U-Net con cross-attention + encoder de texto**— es, casi literalmente, la arquitectura de **Stable Diffusion**, lanzado en agosto de 2022 por Stability AI, CompVis y Runway ML (con los mismos autores Rombach, Blattmann, Esser, Ommer en el centro). La diferencia operativa de Stable Diffusion respecto al LDM-KL-8 del paper es principalmente de escala y de encoder de texto: Stable Diffusion v1 usa el **encoder de texto de CLIP** (Radford et al., 2021) como $\tau_\theta$, en lugar del transformer con tokenizador BERT, y se entrena sobre subconjuntos de LAION-5B.

Por qué LDM fue el modelo que "democratizó" la generación: la reducción de cómputo es exactamente lo que permitió que el modelo **corriera en GPU de consumidor**. Mientras DALL-E 2 o Imagen permanecían tras APIs cerradas en servidores de Google/OpenAI, Stable Diffusion —pesos abiertos, inferencia en una GPU con ~8–10 GB de VRAM gracias a la difusión en latente— desató un ecosistema masivo: fine-tuning (DreamBooth, Textual Inversion), control espacial (ControlNet), interfaces (Automatic1111, ComfyUI), y una explosión de modelos derivados. Es, con amplio margen, el modelo generativo de imágenes más usado del mundo, y la familia LDM (SDXL, Stable Diffusion 3) sigue siendo el caballo de batalla open-source. El paper también fundó conceptualmente la difusión latente en otras modalidades: video (Stable Video Diffusion), audio y 3D adoptaron el mismo principio de "comprimir primero, difundir después".

## 8. Conexión con la Clase 29 (Modelos Generativos en Visión)

La Clase 29 recorre el arco completo de la generación en visión —GAN, VAE, autoregresivos, difusión— y dedica sus slides finales precisamente a Latent Diffusion (abril 2022) y Stable Diffusion. Este paper es, literalmente, el punto de convergencia de casi todos los demás papers de la clase. El mapeo pieza por pieza:

- **El autoencoder de compresión = VAE + VQGAN.** La fase 1 de LDM es un autoencoder cuya regularización KL lo hace "similar a un VAE" ([Kingma & Welling, 2013](/papers/kingma-vae-2013)) y cuyo entrenamiento adversarial-perceptual lo hace un VQGAN ([Oord et al., VQ-VAE 2017](/papers/oord-vqvae-2017) más el discriminador PatchGAN de [Goodfellow et al., GAN 2014](/papers/goodfellow-gan-2014)). El curso construye estos cimientos antes de llegar a LDM; aquí se ve para qué servían.
- **El proceso de difusión = DDPM.** El objetivo $L_{LDM}$ es el objetivo reponderado de [Ho et al., DDPM 2020](/papers/ho-ddpm-2020), aplicado en el latente en vez de en píxeles. Entender DDPM es prerrequisito directo: LDM *es* DDPM con un cambio de espacio.
- **El backbone = U-Net.** El predictor de ruido $\epsilon_\theta$ es una U-Net condicionada en el tiempo, la arquitectura de [Ronneberger et al., 2015](/papers/unet-ronneberger-2015), originalmente diseñada para segmentación biomédica y reutilizada aquí por su sesgo inductivo para datos espaciales.
- **El encoder de texto = CLIP.** El $\tau_\theta$ que procesa los prompts en Stable Diffusion es el encoder de texto de [Radford et al., CLIP 2021](/papers/clip-radford-2021). La cross-attention entre las *queries* visuales de la U-Net y las *keys/values* textuales de CLIP es el corazón del condicionamiento texto-a-imagen.
- **El control = classifier-free guidance.** Los mejores resultados (LDM-4-G, LDM-KL-8-G) usan la guía sin clasificador, que el curso cubre en paralelo y que es lo que hace que el modelo "obedezca" el prompt con fuerza ajustable.

El **flujo de trabajo** que la clase presenta es exactamente el de inferencia de Stable Diffusion: el *prompt* se codifica con **CLIP** → se inicializa **ruido latente** gaussiano en el espacio del VAE → la **U-Net** hace *denoising condicionado* por la cross-attention con el texto, paso a paso (DDIM) → el **decoder del VAE** transforma el latente final en una imagen. Comprender este paper es comprender por qué cada uno de esos bloques está ahí y de qué paper anterior proviene — la diferencia entre saber usar la API de Stable Diffusion y entender la máquina por dentro.

Conexión transversal con los fundamentos del dominio: este paper es el clímax de [modelos de difusión](/fundamentos/modelos-de-difusion) y, más ampliamente, de [modelos generativos](/fundamentos/modelos-generativos), donde difusión latente cierra la tensión histórica entre calidad (GAN), estabilidad/cobertura (verosimilitud) y costo (autoregresivos en píxeles). Material de la clase en [/clases/clase-29](/clases/clase-29).
