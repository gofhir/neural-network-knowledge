---
title: "VQ-VAE: Neural Discrete Representation Learning (2017)"
weight: 336
math: true
---

{{< paper-card
    title="Neural Discrete Representation Learning"
    authors="Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu"
    year="2017"
    venue="NeurIPS 2017"
    pdf="/papers/vq-vae-oord-2017.pdf"
    arxiv="1711.00937" >}}
Paper de DeepMind que introdujo el **VQ-VAE** (*Vector Quantised-Variational AutoEncoder*), el primer autoencoder variacional con **latentes discretos** que iguala el desempeño de sus contrapartes continuas en log-verosimilitud. La idea central es importar la **cuantización vectorial** al cuello de botella del autoencoder: el encoder emite un vector continuo que se reemplaza por el embedding más cercano de un **codebook** aprendido (asignación *nearest-neighbor* 1-de-K), y ese embedding discreto alimenta al decoder. Para retropropagar a través de la cuantización no diferenciable usa el **straight-through estimator**, y entrena el diccionario con un par de pérdidas auxiliares (*codebook loss* + *commitment loss*). Tras entrenar el autoencoder, ajusta un **prior autorregresivo** (PixelCNN para imagen, WaveNet para audio) sobre los códigos para poder generar. Evita el **posterior collapse** y resultó el cimiento del que descienden VQ-GAN, DALL-E y el autoencoder de Latent/Stable Diffusion.
{{< /paper-card >}}

---

## Contexto

Hacia 2017 el modelado generativo de imágenes, audio y video ya producía muestras impresionantes (GANs, PixelCNN, WaveNet), pero las representaciones latentes aprendidas seguían siendo poco útiles. El punto de partida es el [VAE](/papers/vae-kingma-2013) (Kingma & Welling, 2013), con encoder $q(z|x)$, prior $p(z)$ y decoder $p(x|z)$, donde posteriors y priors **gaussianos** habilitan el truco de reparametrización y gradientes de baja varianza. Esta elección continua domina el campo aun cuando la modalidad subyacente es discreta —lenguaje, fonemas, símbolos.

El villano del paper es el **posterior collapse**. Al emparejar un VAE con un decoder autorregresivo potente, el modelo aprende a ignorar $z$: el decoder modela $x$ por sí solo y los latentes quedan inertes, vaciando de sentido la representación que se quería aprovechar. Los autores también motivan los latentes discretos por razones conceptuales: muchas modalidades de interés son intrínsecamente simbólicas y encajan naturalmente con razonamiento, planificación y aprendizaje predictivo.

Entrenar latentes discretos había sido difícil. Las alternativas previas —**NVIL** (estimador de una muestra), **VIMCO** (objetivo multi-muestra), **Concrete / Gumbel-softmax** (reparametrización continua con temperatura anelada)— no cerraban la brecha con los VAE continuos y se evaluaban en datasets pequeños. La línea de compresión con redes (cuantización escalar, *soft-to-hard*) tampoco servía desde cero: el decoder siempre invertía la relajación continua, de modo que no ocurría cuantización real.

## Método: la arquitectura VQ-VAE

### Cuantización vectorial del latente

Se define un espacio de embeddings $e \in \mathbb{R}^{K \times D}$: un **codebook** con $K$ entradas (el tamaño del "vocabulario" discreto) y dimensión $D$ por entrada. El encoder produce un vector continuo $z_e(x)$ y el latente discreto se obtiene buscando el embedding **más cercano** en $L_2$:

$$z_q(x) = e_k, \quad k = \arg\min_j \|z_e(x) - e_j\|_2$$

El posterior categórico $q(z=k|x)$ es **determinista** y *one-hot* (1 para el vecino más cercano, 0 en otro caso). El *forward pass* es un autoencoder regular con una no-linealidad que mapea cada latente a uno de $K$ vectores (mapeo 1-de-K). En la práctica no se extrae un único $z$, sino **campos de latentes**: 1D para habla, **2D para imágenes** (p.ej. una grilla $32 \times 32$ en ImageNet), 3D para video.

Como marco probabilístico se interpreta como un VAE en el que, al ser el posterior determinista y asumirse un **prior uniforme**, la divergencia KL es **constante e igual a $\log K$**: desaparece como término entrenable del ELBO y puede ignorarse respecto al encoder.

### Straight-through estimator y la pérdida de tres términos

La operación $\arg\min$ **no tiene gradiente**. El paper la sortea con el **straight-through estimator** (Bengio et al., 2013): en el *forward* se pasa $z_q(x)$ al decoder; en el *backward* el gradiente se **copia sin alterar** desde la entrada del decoder $z_q(x)$ hacia la salida del encoder $z_e(x)$. Como ambos viven en el mismo espacio $D$-dimensional, ese gradiente le indica al encoder cómo mover su salida para reducir el error de reconstrucción (eventualmente discretizándose a un embedding distinto).

El truco tiene una consecuencia: los embeddings $e_i$ **no reciben gradiente** del término de reconstrucción y necesitan entrenarse aparte. La pérdida total tiene **tres componentes**:

$$L = \underbrace{\log p(x \mid z_q(x))}_{\text{reconstrucción}} + \underbrace{\|\,\text{sg}[z_e(x)] - e\,\|_2^2}_{\text{codebook loss}} + \underbrace{\beta \,\|\,z_e(x) - \text{sg}[e]\,\|_2^2}_{\text{commitment loss}}$$

donde $\text{sg}[\cdot]$ es el operador *stopgradient* (identidad en el *forward*, derivada cero en el *backward*). El reparto es limpio:

- **Reconstrucción:** optimiza al decoder y, vía straight-through, al encoder.
- **Codebook loss:** mueve los embeddings hacia las salidas del encoder. Es exactamente el objetivo de la **cuantización vectorial / k-means**; solo actualiza el diccionario.
- **Commitment loss:** obliga al encoder a "comprometerse" con un embedding y evita que su salida crezca sin límite.

El método es **robusto a $\beta$** (resultados estables para $\beta \in [0.1, 2.0]$; se usa $\beta = 0.25$). Como variante (Apéndice A.1), el codebook puede actualizarse por **medias móviles exponenciales** (EMA, $\gamma = 0.99$) en lugar de la *codebook loss* —una versión *online* de k-means que se volvería el estándar posterior por su estabilidad.

### El prior autorregresivo

Durante el entrenamiento del VQ-VAE el prior $p(z)$ se mantiene **uniforme**. *Después*, se ajusta una distribución **autorregresiva** sobre los códigos discretos —**PixelCNN** para imágenes, **WaveNet** para audio crudo— para poder generar $x$ por *ancestral sampling*. Esta receta en dos etapas —comprimir a un código discreto, luego modelar la distribución de los códigos con un autorregresivo potente— es la que define el paradigma: el autorregresivo ya no modela píxeles ruidosos y correlacionados, sino una grilla pequeña de símbolos que captura la **estructura global**.

## Experimentos

**Imágenes vs. continuo (CIFAR10).** Con la misma arquitectura estándar, en *bits/dim*: VAE continuo **4.51**, VQ-VAE **4.67**, VIMCO **5.14**. El VQ-VAE es el primer modelo de latentes discretos que reta a los VAE continuos.

**Imágenes (ImageNet, DeepMind Lab).** Comprime imágenes de $128 \times 128 \times 3$ a un espacio discreto $32 \times 32 \times 1$ con $K=512$ —una **reducción de ~42.6x en bits**— con reconstrucciones solo ligeramente más borrosas. Una PixelCNN entrenada sobre ese espacio genera muestras coherentes. El experimento anti-colapso más demostrativo, en DeepMind Lab, comprime una imagen completa a **solo 3 latentes** (~27 bits, menos que un float32): un montaje que normalmente rompe a los VAE, pero el VQ-VAE **usa los latentes de forma significativa**, conservando la disposición de la sala mientras la PixelCNN genera las texturas proceduralmente.

**Audio (VCTK, LibriSpeech).** Comprime el *waveform* 64x. La reconstrucción tiene el **mismo contenido** (texto) pero distinta prosodia: un espacio de alto nivel invariante a lo de bajo nivel. Muestreando el prior incondicionalmente aparecen **palabras y frases claras** (vs. el balbuceo de un WaveNet incondicional); mapeando los 128 latentes a 41 fonemas la exactitud es **49.3%** (vs. 7.2% aleatorio). Permite además **conversión de hablante** sin supervisión.

**Video (DeepMind Lab).** Condicionado en acciones, genera 10 cuadros **enteramente en el espacio latente** (sin pasar por píxeles), decodificando solo al final —cuadros coherentes con la acción y sin degradación visual.

## Limitaciones reconocidas

- **Reconstrucción borrosa con MSE.** Los autores reconocen que una pérdida perceptual / adversarial (GAN) mejoraría la nitidez —el hueco exacto que VQ-GAN cerraría después.
- **Prior entrenado por separado.** Ajustar el prior conjuntamente con el VQ-VAE podría reforzar resultados; queda a futuro.
- **El soft-to-hard no funcionó desde cero**, de ahí la elección del straight-through duro.
- **Aproximación de la verosimilitud** vía el supuesto MAP de que el decoder no asigna masa a $z \neq z_q(x)$.
- **El significado de cada código depende del contexto** (bi/tri-gramas), por lo que el mapeo simple a fonemas subestima la información codificada.

## Impacto

El VQ-VAE inauguró el paradigma de **"comprimir a un código discreto + modelar los códigos con un autorregresivo potente"** que domina buena parte de la generación moderna:

- **VQ-VAE-2** (Razavi et al., 2019) escaló la idea a jerarquías multi-escala, con fidelidad competitiva con GANs.
- **VQ-GAN** (Esser et al., 2021) reemplazó la pérdida MSE por una **perceptual + adversarial** sobre el autoencoder discreto y modeló los códigos con un Transformer, habilitando síntesis de alta resolución.
- **DALL-E** (Ramesh et al., 2021) tokenizó imágenes con un autoencoder discreto (dVAE) y modeló texto + imagen con un Transformer autorregresivo: el VQ-VAE es el ancestro directo de los "tokens de imagen".
- **Latent / Stable Diffusion** ([Rombach et al., 2022](/papers/latent-diffusion-rombach-2022)) movió la difusión a un **espacio latente comprimido** por un autoencoder, ofreciendo explícitamente una **regularización VQ** (*VQ-reg*) heredada directamente del codebook de este paper.

En síntesis, la pregunta que el VQ-VAE respondió —cómo aprender un espacio latente discreto, comprimido y semánticamente útil sin que el decoder lo ignore— resultó ser el cimiento de la generación a gran escala de la era 2019–2022.

## Por qué importa para la Clase 29

La [Clase 29](/clases/clase-29) ("Modelos Generativos en Visión") introduce autoencoders y VAE; el VQ-VAE aporta la **variante discreta** del cuello de botella latente, con varios usos pedagógicos:

- **Del VAE continuo al VQ-VAE.** Misma arquitectura encoder–bottleneck–decoder, distinta naturaleza del *bottleneck*: latente discreto vía codebook + nearest-neighbor + straight-through, y de paso la solución canónica al **posterior collapse**.
- **El latente comprimido como puente hacia Stable Diffusion.** La lección transversal es que la generación de alta resolución no ocurre en píxeles sino en un **espacio latente compacto**; el VQ-VAE es su primera demostración limpia y el origen del autoencoder de [Latent Diffusion](/papers/latent-diffusion-rombach-2022).
- **El codebook como tokenización de imágenes.** El salto de "píxeles" a "tokens discretos" que habilitan VQ-GAN y DALL-E nace aquí: la bisagra entre los modelos de verosimilitud explícita ([VAE](/papers/vae-kingma-2013)) y los pipelines basados en Transformers/difusión.

## Notas y enlaces

- arXiv: https://arxiv.org/abs/1711.00937
- Muestras de audio/video: https://avdnoord.github.io/homepage/vqvae/
- Venue: 31st Conference on Neural Information Processing Systems (NeurIPS 2017), Long Beach, CA.
- Enlaces internos: [Variational Autoencoder (Kingma & Welling, 2013)](/papers/vae-kingma-2013) · [Latent Diffusion (Rombach et al., 2022)](/papers/latent-diffusion-rombach-2022) · [Fundamento: Modelos Generativos](/fundamentos/modelos-generativos) · [Clase 29](/clases/clase-29).
