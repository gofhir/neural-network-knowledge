---
title: "Latent Diffusion / Stable Diffusion (2022)"
weight: 331
math: true
---

{{< paper-card
    title="High-Resolution Image Synthesis with Latent Diffusion Models"
    authors="Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer"
    year="2022"
    venue="CVPR 2022"
    pdf="/papers/latent-diffusion-rombach-2022.pdf"
    arxiv="2112.10752" >}}
El paper fundacional de **Stable Diffusion**. Su tesis: los modelos de difusión producen imágenes de calidad estado del arte pero **operan en el espacio de píxeles**, lo que los vuelve carísimos (cientos de días-GPU para entrenar, ~5 días en una A100 para 50.000 muestras). La solución es mover la difusión a un **espacio latente comprimido** aprendido por un autoencoder, perceptualmente equivalente a la imagen pero de mucha menor dimensión. El modelo deja de gastar cómputo modelando detalles imperceptibles y se concentra en la estructura semántica: misma o mejor calidad, una fracción del costo. La segunda contribución es un **condicionamiento por cross-attention** que inyecta texto (vía CLIP), clases, layouts o mapas semánticos en la U-Net con una sola arquitectura. Difusión en el latente + cross-attention = los *Latent Diffusion Models* (LDM) que, escalados con un encoder de texto, dan origen a Stable Diffusion.
{{< /paper-card >}}

---

## Contexto: el muro de cómputo de la difusión en píxeles

Hacia 2021, la síntesis de imágenes de alta resolución estaba dominada por dos familias con talones de Aquiles opuestos. Las **GAN** ([Goodfellow et al., 2014](/papers/goodfellow-gan-2014); BigGAN, StyleGAN) samplean rápido y con buena calidad, pero son inestables, sufren *mode collapse* y no escalan a distribuciones complejas y multimodales. Los **modelos basados en verosimilitud** ([VAE](/papers/vae-kingma-2013), flows, autoregresivos) estiman densidad mejor y optimizan de forma estable, pero los autoregresivos (DALL-E, VQGAN+Transformer) exigen miles de millones de parámetros y sampleo secuencial costoso.

Los **modelos de difusión** ([DDPM, Ho et al. 2020](/papers/ddpm-ho-2020); score-based SDEs) emergieron como la síntesis ganadora: descomponen la generación en una secuencia de *denoising autoencoders*, logran calidad estado del arte (Dhariwal & Nichol, "Diffusion Models Beat GANs", 2021) y, al ser basados en verosimilitud, evitan el *mode collapse* sin necesitar miles de millones de parámetros. Admiten además guía (*guidance*) para controlar la generación sin reentrenar.

El problema es el **costo**. Los DM son modelos *mode-covering*: tienden a gastar capacidad —y cómputo— modelando detalles imperceptibles de alta frecuencia. Entrenar requiere gradientes en el espacio de alta dimensión de imágenes RGB, e inferir exige correr la red secuencialmente por decenas o cientos de pasos (25–1000). Entrenar un DM potente solo estaba al alcance de una fracción del campo. El punto de partida del método es un análisis del *trade-off* tasa-distorsión: el aprendizaje de cualquier modelo de verosimilitud se divide en una etapa de **compresión perceptual** (elimina alta frecuencia, aprende poca semántica) y una de **compresión semántica** (composición conceptual). Los DM gastan gradientes en toda la primera etapa, sobre cada píxel — desperdicio. La idea: encontrar un espacio perceptualmente equivalente pero computacionalmente conveniente, y entrenar la difusión ahí.

## Método

El diseño separa explícitamente la **fase de compresión** y la **fase generativa**. A diferencia de LSGM (Vahdat et al.), que aprendía autoencoder y *prior* score-based conjuntamente y obligaba a un balanceo delicado, aquí el autoencoder se entrena una sola vez, por separado, y puede reutilizarse para múltiples modelos de difusión o tareas distintas.

### Fase 1 — compresión perceptual con autoencoder adversarial-perceptual

Dada una imagen $x \in \mathbb{R}^{H \times W \times 3}$, un **encoder** $\mathcal{E}$ la mapea a un latente $z = \mathcal{E}(x) \in \mathbb{R}^{h \times w \times c}$, y un **decoder** $\mathcal{D}$ reconstruye $\tilde{x} = \mathcal{D}(z)$. El encoder *downsamplea* por un factor $f = H/h = W/w$, con $f = 2^m$.

La fidelidad depende de *cómo* se entrena este autoencoder. No se usa una pérdida pixel-wise sola (produce reconstrucciones borrosas). En su lugar, basándose en VQGAN (Esser et al.), se combina:

- una **pérdida perceptual** (LPIPS), que compara características profundas en vez de píxeles crudos; y
- un **objetivo adversarial basado en parches** (discriminador PatchGAN), que fuerza el realismo local y confina las reconstrucciones a la variedad de imágenes naturales, evitando el desenfoque.

Para evitar latentes de varianza arbitraria se exploran dos regularizaciones: **KL-reg.** (penalización KL leve hacia una normal estándar, "similar a un VAE" — de ahí que en Stable Diffusion se hable del "VAE") y **VQ-reg.** (cuantización vectorial absorbida por el decoder, interpretable como un VQGAN). A diferencia de VQGAN/DALL-E, que dependían de compresión agresiva y un orden 1D del latente, aquí el DM trabaja con la **estructura bidimensional** de $z$, lo que permite compresiones suaves y reconstrucciones fieles. Empíricamente, $f \in \{4, 8\}$ es el punto dulce.

### Fase 2 — difusión en el espacio latente

Un modelo de difusión aprende $p(x)$ denoising gradualmente una variable normal, interpretado como una secuencia de *denoising autoencoders* $\epsilon_\theta(x_t, t)$ entrenados para predecir el ruido de una versión ruidosa $x_t$. El objetivo reponderado de DDPM es:

$$L_{DM} = \mathbb{E}_{x, \epsilon \sim \mathcal{N}(0,1), t}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|_2^2\right]$$

La innovación de LDM es **trasladar este objetivo al latente**. Con $\mathcal{E}$ y $\mathcal{D}$ congelados, se tiene un espacio de baja dimensión donde la alta frecuencia ya está abstraída:

$$L_{LDM} = \mathbb{E}_{\mathcal{E}(x), \epsilon \sim \mathcal{N}(0,1), t}\left[\|\epsilon - \epsilon_\theta(z_t, t)\|_2^2\right]$$

El backbone $\epsilon_\theta$ es una **U-Net condicionada en el tiempo** ([Ronneberger et al., 2015](/papers/unet-ronneberger-2015)). Como el proceso *forward* es fijo, $z_t$ se obtiene eficientemente de $\mathcal{E}$ durante el entrenamiento, y las muestras de $p(z)$ se decodifican a imagen con **una sola pasada** por $\mathcal{D}$. Las ventajas de salir de los píxeles: (i) el sampleo ocurre en dimensión reducida, mucho más eficiente; (ii) se explota el sesgo inductivo convolucional de la U-Net, útil porque el latente conserva su estructura 2D, lo que elimina la necesidad de la compresión agresiva de los enfoques autoregresivos; (iii) el autoencoder es de propósito general y reutilizable.

### Condicionamiento por cross-attention

Para modelar distribuciones condicionales $p(z \mid y)$ —donde $y$ puede ser texto, mapas semánticos, layouts u otra imagen—, se aumenta la U-Net con **capas de cross-attention** (Vaswani et al., 2017):

1. Un **encoder específico de dominio** $\tau_\theta$ proyecta la condición $y$ a una representación intermedia $\tau_\theta(y) \in \mathbb{R}^{M \times d_\tau}$. Para texto, $\tau_\theta$ es un transformer (en el paper, con tokenizador BERT; en Stable Diffusion, el encoder de texto de [**CLIP**](/papers/clip-radford-2021)).
2. Esa representación se inyecta en las capas intermedias de la U-Net:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) \cdot V$$

donde las *queries* vienen del estado interno de la U-Net, $Q = W_Q^{(i)} \cdot \varphi_i(z_t)$, y las *keys/values* de la condición, $K = W_K^{(i)} \cdot \tau_\theta(y)$, $V = W_V^{(i)} \cdot \tau_\theta(y)$. El objetivo se vuelve $L_{LDM} = \mathbb{E}_{\mathcal{E}(x), y, \epsilon, t}[\|\epsilon - \epsilon_\theta(z_t, t, \tau_\theta(y))\|_2^2]$, con $\tau_\theta$ y $\epsilon_\theta$ optimizados conjuntamente. Cambiar la modalidad de la condición es cambiar el encoder, no la arquitectura de difusión. Para condiciones espacialmente alineadas (mapas semánticos, baja resolución para super-res), se concatena la condición *downsampleada* al input de la U-Net.

### Los factores de downsampling

El barrido $f \in \{1, 2, 4, 8, 16, 32\}$ revela el balance: $f$ pequeño (LDM-1 = difusión en píxeles) entrena lento porque deja toda la compresión perceptual al DM; $f$ demasiado grande (LDM-32) pierde información y limita la calidad alcanzable; **LDM-4 y LDM-8** son el óptimo. Tras 2M pasos en ImageNet hay una brecha de FID de 38 puntos entre LDM-1 y LDM-8.

## Experimentos

Fijando recursos (una sola NVIDIA A100) para comparaciones limpias:

- **Generación incondicional.** LDM-4 establece estado del arte en CelebA-HQ 256² (**FID 5.11**), superando modelos de verosimilitud previos, GAN e incluso LSGM. FFHQ 4.98, LSUN-Churches 4.02, LSUN-Bedrooms 2.95 (cercano a ADM con la mitad de parámetros y 4× menos recursos). Mejora consistente en *Precision* y *Recall* sobre las GAN.
- **Síntesis condicional por clase (ImageNet).** LDM-4-G con [classifier-free guidance](/papers/classifier-free-guidance-ho-2022) alcanza **FID 3.60** con 400M parámetros, superando a ADM-G (FID 4.59, 608M).
- **Texto-a-imagen (el camino a Stable Diffusion).** Un LDM-KL-8 de **1.45B parámetros** condicionado en prompts sobre **LAION-400M**, con $\tau_\theta$ transformer. En MS-COCO 256² con 250 pasos DDIM y guía, FID 12.63, a la par de GLIDE (6B) y Make-A-Scene (4B) con sustancialmente menos parámetros. Genera prompts arbitrarios ("A street sign that reads 'Latent Diffusion'") con calidad notable.
- **Layout-to-image y síntesis semántica.** Entrenado a 256² pero aplicado convolucionalmente, generaliza a megapíxeles (paisajes a 512×1024).
- **Super-resolución.** LDM-SR (×4) **supera a SR3 en FID** con menos parámetros (169M vs. 625M); estudio humano confirma la preferencia.
- **Inpainting.** Nuevo estado del arte en Places, superando a LaMa. La medición de **eficiencia** es el corazón del argumento: entre difusión en píxeles (LDM-1) y latente (LDM-4) hay un *speed-up* de al menos **2.7×** mejorando el FID en al menos 1.6× (LDM-1 entrega 0.11 muestras/s y 20.66 h/época; LDM-4 entrega 0.35 muestras/s y 6.66 h/época, con mejor FID).

El mensaje transversal: **misma o mejor calidad, fracción del cómputo.**

## Limitaciones reconocidas

- **Sampleo todavía secuencial.** Aunque LDM reduce drásticamente el costo, su sampleo secuencial sigue siendo más lento que el de las GAN (que generan en una pasada). La difusión paga su estabilidad y cobertura de modos con latencia de inferencia.
- **El autoencoder como cuello de botella.** Cuando se requiere alta precisión a nivel de píxel, la capacidad de reconstrucción del autoencoder ($f=4$) puede ser un límite: nada puede recuperarse mejor de lo que el decoder es capaz de reconstruir.
- **Impacto social (doble filo).** Democratiza el acceso pero facilita *deep fakes* y desinformación, puede revelar datos de entrenamiento sensibles y tiende a reproducir sesgos de los datos.

## Impacto: Stable Diffusion y la democratización de la síntesis

La combinación que describe —**VAE de compresión + U-Net con cross-attention + encoder de texto**— es, casi literalmente, la arquitectura de **Stable Diffusion**, lanzado en agosto de 2022 por Stability AI, CompVis y Runway ML (con los mismos autores en el centro). La diferencia operativa frente al LDM-KL-8 del paper es de escala y de encoder de texto: Stable Diffusion v1 usa el encoder de texto de [CLIP](/papers/clip-radford-2021) como $\tau_\theta$ (en lugar del transformer con tokenizador BERT) y se entrena sobre subconjuntos de LAION-5B.

Por qué fue el modelo que "democratizó" la generación: la reducción de cómputo es exactamente lo que permitió que corriera en **GPU de consumidor** (~8–10 GB de VRAM). Mientras DALL-E 2 e Imagen permanecían tras APIs cerradas, Stable Diffusion —pesos abiertos, inferencia en latente— desató un ecosistema masivo: fine-tuning (DreamBooth, Textual Inversion), control espacial (ControlNet), interfaces (Automatic1111, ComfyUI) y una explosión de derivados. La familia LDM (SDXL, Stable Diffusion 3) sigue siendo el caballo de batalla open-source, y el principio de "comprimir primero, difundir después" se extendió a video (Stable Video Diffusion), audio y 3D.

## El flujo de inferencia de Stable Diffusion

La [Clase 29](/clases/clase-29) presenta literalmente el flujo de inferencia, donde cada bloque proviene de un paper anterior del curso:

1. El **prompt** se codifica con [**CLIP**](/papers/clip-radford-2021) → embeddings de texto.
2. Se inicializa **ruido latente** gaussiano en el espacio del VAE.
3. La [**U-Net**](/papers/unet-ronneberger-2015) hace *denoising condicionado* por la cross-attention con el texto, paso a paso (DDIM), guiada por [classifier-free guidance](/papers/classifier-free-guidance-ho-2022).
4. El **decoder del VAE** transforma el latente final en una imagen.

El mapeo pieza por pieza muestra a LDM como el punto de convergencia del curso: el autoencoder de compresión es [VAE](/papers/vae-kingma-2013) + VQGAN; el proceso de difusión es [DDPM](/papers/ddpm-ho-2020) aplicado en el latente; el backbone es la [U-Net](/papers/unet-ronneberger-2015); el encoder de texto es [CLIP](/papers/clip-radford-2021); y el control es [classifier-free guidance](/papers/classifier-free-guidance-ho-2022). Comprender este paper es entender por qué cada bloque está ahí — la diferencia entre usar la API de Stable Diffusion y entender la máquina por dentro.

Transversalmente, este paper es el clímax de [modelos de difusión](/fundamentos/modelos-de-difusion) y, más ampliamente, de [modelos generativos](/fundamentos/modelos-generativos), donde la difusión latente cierra la tensión histórica entre calidad (GAN), estabilidad/cobertura (verosimilitud) y costo (autoregresivos en píxeles). Material de la clase en [/clases/clase-29](/clases/clase-29).

El [Laboratorio 29](/laboratorios/lab-29) usa esta arquitectura en la práctica: manipula Stable Diffusion (la implementación de LDM) con la librería `diffusers` — el efecto de `num_inference_steps`, los noise schedulers, la `guidance_scale`, y los modos de condicionamiento Img2Img/Inpainting/ControlNet — y su cuestionario responde qué es la difusión latente y sus ventajas.
