# Neural Discrete Representation Learning (VQ-VAE) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Neural Discrete Representation Learning*.
- **Autores:** Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu (los tres en DeepMind).
- **Venue:** 31st Conference on Neural Information Processing Systems (**NeurIPS / NIPS 2017**), Long Beach, CA, USA.
- **Año:** 2017. **Preprint:** arXiv:1711.00937 (v2, 30 may 2018), [arxiv.org/abs/1711.00937](https://arxiv.org/abs/1711.00937).
- **Muestras de audio/video:** [avdnoord.github.io/homepage/vqvae](https://avdnoord.github.io/homepage/vqvae/).

El paper propone el **VQ-VAE** (*Vector Quantised-Variational AutoEncoder*), un modelo generativo que aprende representaciones latentes **discretas** sin supervisión. Difiere del VAE clásico en dos puntos que el abstract subraya: el encoder produce **códigos discretos** en vez de continuos, y el **prior se aprende** en lugar de mantenerse estático. La idea central es importar la **cuantización vectorial (VQ)** —un algoritmo clásico de *dictionary learning*— al cuello de botella del autoencoder: el encoder emite un vector continuo que se reemplaza por el embedding más cercano de un **codebook** aprendido, y ese embedding discreto alimenta al decoder.

La tesis tiene dos motivaciones. La primera es **conceptual**: muchas modalidades de interés son intrínsecamente discretas o se describen bien con símbolos —el lenguaje es discreto, el habla se representa como secuencias de símbolos, las imágenes se describen concisamente con lenguaje— y las representaciones discretas encajan naturalmente con razonamiento, planificación y aprendizaje predictivo. La segunda es **práctica**: el VQ-VAE evita el **"posterior collapse"** (colapso del posterior), la patología por la cual, cuando un VAE se combina con un decoder autorregresivo potente (p.ej. PixelCNN), los latentes son ignorados porque el decoder por sí solo modela los datos. El resultado declarado es notable: es el **primer modelo de VAE con latentes discretos que iguala el desempeño de sus contrapartes continuas** en log-verosimilitud, manteniendo la flexibilidad de las distribuciones discretas.

Para la **Clase 29 (Modelos Generativos en Visión)** esto importa porque el VQ-VAE es la pieza que conecta los autoencoders/VAE con la generación moderna a gran escala: el espacio latente comprimido y discreto que introduce es exactamente el ingrediente que más tarde explotan VQ-GAN, DALL-E y el autoencoder (opcionalmente VQ-regularizado) de Latent/Stable Diffusion. Entender este paper es entender *por qué* la generación de imágenes de alta resolución migró del espacio de píxeles a un espacio latente compacto.

## 2. Contexto histórico: VAE continuo, posterior collapse y el deseo de latentes discretos

Hacia 2017, el modelado generativo de imágenes, audio y video ya producía muestras impresionantes (GANs, PixelCNN/PixelRNN, WaveNet, Video Pixel Networks), pero la utilidad de las representaciones aprendidas de forma no supervisada seguía siendo limitada. El objetivo que el paper se fija es preciso: un modelo que **conserve las características importantes de los datos en su espacio latente** mientras optimiza la verosimilitud.

El punto de partida es el **VAE** (Kingma & Welling, 2013; Rezende et al., 2014). Un VAE consta de tres piezas: un encoder que parametriza un posterior $q(z|x)$, un prior $p(z)$, y un decoder $p(x|z)$. La formulación estándar asume posteriors y priors **gaussianos con covarianza diagonal**, lo que habilita el *truco de reparametrización* gaussiano y, con él, gradientes de baja varianza. Esta elección continua domina el campo incluso cuando la modalidad subyacente es discreta.

El **problema del posterior collapse** es el villano del paper. Trabajos previos (Chen et al., 2016, "Variational lossy autoencoder") habían sugerido que los mejores modelos generativos por log-verosimilitud son aquellos *sin latentes* pero con un decoder potente. La consecuencia práctica es que, al emparejar un VAE con un decoder autorregresivo fuerte, el modelo "aprende" a ignorar $z$: el decoder modela $x$ por sí solo y los latentes quedan inertes. Esto vacía de sentido a la representación, que es justo lo que uno quería aprovechar.

**¿Por qué latentes discretos y por qué eran difíciles?** Los latentes discretos son un encaje más natural para lenguaje, habla y razonamiento, pero entrenarlos había sido un dolor de cabeza. El paper revisa las alternativas existentes y por qué no cerraban la brecha con los VAE continuos:

- **NVIL** (Mnih & Gregor, 2014): estimador de una sola muestra del *lower bound*, con técnicas de reducción de varianza.
- **VIMCO** (Mnih & Rezende, 2016): objetivo multi-muestra que acelera la convergencia usando varias muestras de la red de inferencia.
- **Concrete / Gumbel-softmax** (Maddison et al., 2016; Jang et al., 2016): una reparametrización *continua* con una temperatura que se *anela* (anneal) durante el entrenamiento para converger a una distribución discreta en el límite. Al inicio los gradientes tienen baja varianza pero son sesgados; al final, alta varianza pero insesgados.

Ninguno de estos métodos cerraba la brecha de desempeño con los VAE de latentes continuos (que gozan de la baja varianza del truco gaussiano), y la mayoría se evaluaba en datasets pequeños (MNIST) con latentes de baja dimensionalidad. El paper también se distancia de la línea de **compresión de imágenes con redes neuronales**: Theis et al. (2017) usan cuantización escalar; Agustsson et al. (2017) proponen una relajación continua de la cuantización vectorial anelada hacia un *clustering* duro, pero entrenan primero un autoencoder y aplican VQ después. Los autores reportan que, en sus experimentos, ese enfoque *soft-to-hard* desde cero **no funcionó**: el decoder siempre lograba invertir la relajación continua, de modo que no ocurría cuantización real.

## 3. Contribución central

La contribución es el **VQ-VAE**, resumida por los propios autores en cuatro puntos:

1. **Un modelo con latentes discretos que es simple, no sufre posterior collapse y no tiene problemas de varianza** en el gradiente (a diferencia de NVIL/VIMCO/Gumbel-softmax).
2. **Iguala a sus contrapartes continuas en log-verosimilitud** —el primer modelo de latentes discretos que lo consigue.
3. **Muestras coherentes y de alta calidad** al emparejarlo con un prior potente, en imágenes, video y habla.
4. **Aprendizaje de lenguaje a partir de habla cruda sin supervisión** y **conversión de hablante** no supervisada.

El mecanismo se sostiene sobre cuatro decisiones técnicas que conviene enumerar porque son el núcleo del aporte:

- **Codebook de embeddings discretos.** Un espacio de embeddings $e \in \mathbb{R}^{K \times D}$, con $K$ entradas (el tamaño del "vocabulario" discreto) y dimensión $D$ por entrada.
- **Asignación por vecino más cercano (*nearest neighbour*).** El encoder produce $z_e(x)$ y el latente discreto se obtiene buscando el embedding más cercano en $L_2$.
- **Straight-through estimator** para retropropagar a través de la operación de cuantización, que no es diferenciable.
- **Prior autorregresivo aprendido** (PixelCNN para imágenes, WaveNet para audio) ajustado *después* del entrenamiento del autoencoder, sobre los códigos discretos, para poder generar.

## 4. Método: la arquitectura VQ-VAE en detalle

### 4.1. Variables latentes discretas y la cuantización por vecino más cercano

Se define el espacio de embeddings $e \in \mathbb{R}^{K \times D}$, con $K$ vectores $e_i \in \mathbb{R}^{D}$, $i \in \{1, \dots, K\}$. El modelo toma una entrada $x$, la pasa por un encoder que produce $z_e(x)$, y luego calcula el latente discreto por **búsqueda de vecino más cercano** en el codebook compartido. El posterior categórico $q(z|x)$ es **determinista** y *one-hot*:

$$q(z = k \mid x) = \begin{cases} 1 & \text{si } k = \arg\min_j \|z_e(x) - e_j\|_2 \\ 0 & \text{en otro caso} \end{cases}$$

La entrada al decoder es el embedding correspondiente:

$$z_q(x) = e_k, \quad \text{donde } k = \arg\min_j \|z_e(x) - e_j\|_2$$

Se puede ver este *forward pass* como un autoencoder regular con una no-linealidad particular que mapea los latentes a uno de $K$ vectores de embedding (un mapeo *1-de-K*). El conjunto de parámetros del modelo es la unión de los del encoder, los del decoder y el propio espacio de embeddings $e$.

Aunque se describe con una sola variable $z$ por simplicidad, en la práctica se extraen **campos de latentes**: 1D para habla, **2D para imágenes** y 3D para video. Por ejemplo, en ImageNet se usa un campo de $32 \times 32$ latentes; en CIFAR10, $8 \times 8 \times 10$.

**El marco probabilístico.** El modelo se interpreta como un VAE en el que se puede acotar $\log p(x)$ con el ELBO. Como el posterior $q(z=k|x)$ es determinista y se asume un **prior uniforme** sobre $z$, la divergencia KL es **constante e igual a $\log K$**: desaparece como término entrenable del ELBO y puede ignorarse durante el entrenamiento respecto a los parámetros del encoder.

### 4.2. Aprendizaje: straight-through estimator y la pérdida de tres términos

La operación $\arg\min$ de la cuantización **no tiene gradiente real**. El paper aproxima el gradiente con el **straight-through estimator** (Bengio et al., 2013): durante el *forward pass* se pasa $z_q(x)$ al decoder, y durante el *backward pass* el gradiente $\nabla_z L$ se **copia sin alterar** desde la entrada del decoder $z_q(x)$ hacia la salida del encoder $z_e(x)$. Como ambos viven en el mismo espacio $D$-dimensional, ese gradiente contiene información útil sobre cómo el encoder debe cambiar su salida para reducir el error de reconstrucción. La Figura 1 lo ilustra: el gradiente puede empujar la salida del encoder de modo que en el siguiente *forward pass* se discretice a un embedding distinto.

El truco del straight-through tiene una consecuencia: como el gradiente "salta" por encima de la cuantización, **los embeddings $e_i$ no reciben gradiente** del término de reconstrucción. Para entrenarlos hace falta un término aparte. La función de pérdida total tiene **tres componentes**:

$$L = \underbrace{\log p(x \mid z_q(x))}_{\text{reconstrucción}} + \underbrace{\|\,\text{sg}[z_e(x)] - e\,\|_2^2}_{\text{codebook loss}} + \underbrace{\beta \,\|\,z_e(x) - \text{sg}[e]\,\|_2^2}_{\text{commitment loss}}$$

donde $\text{sg}[\cdot]$ es el operador *stopgradient* (identidad en el *forward*, derivada cero en el *backward*; deja su operando como una constante no actualizable). El reparto de responsabilidades es limpio:

- **Reconstrucción:** optimiza al decoder y, vía el straight-through, al encoder.
- **Codebook loss** ($\|\text{sg}[z_e(x)] - e\|_2^2$): mueve los embeddings $e_i$ hacia las salidas del encoder. Es exactamente el objetivo de la **cuantización vectorial (VQ)** / k-means. Solo actualiza el diccionario.
- **Commitment loss** ($\beta\|z_e(x) - \text{sg}[e]\|_2^2$): obliga al encoder a "comprometerse" con un embedding y evita que su salida crezca sin límite (el volumen del espacio de embeddings es adimensional, así que podría crecer arbitrariamente si los embeddings no se entrenan tan rápido como el encoder).

El método resulta **robusto a $\beta$**: los resultados no varían para $\beta \in [0.1, 2.0]$; se usa $\beta = 0.25$ en todos los experimentos. Cuando hay $N$ latentes, los términos de k-means y commitment se promedian sobre los $N$.

**Alternativa con EMA (Apéndice A.1).** En vez de aprender el codebook con el término de pérdida, se pueden actualizar las entradas del diccionario como **medias móviles exponenciales (EMA)** de las salidas del encoder asignadas a cada entrada —una versión *online* de la actualización de k-means apta para minibatches, con factor $\gamma = 0.99$. No se usó en los experimentos del paper, pero se volvería el método estándar en implementaciones posteriores por su estabilidad.

### 4.3. El prior autorregresivo aprendido

Durante el entrenamiento del VQ-VAE el prior $p(z)$ se mantiene **constante y uniforme**. *Después* de entrenar el autoencoder, se ajusta una **distribución autorregresiva** sobre los códigos discretos $z$, de modo que se pueda generar $x$ por *ancestral sampling*. Para imágenes se usa una **PixelCNN** sobre los latentes discretos; para audio crudo, una **WaveNet**. Entrenar el prior y el VQ-VAE **conjuntamente** —que podría fortalecer los resultados— se deja como trabajo futuro.

Esta separación en dos etapas (primero comprimir a un código discreto, luego modelar la distribución de los códigos con un autorregresivo potente) es la receta que define el paradigma: el autorregresivo ya no modela píxeles ruidosos y correlacionados, sino una grilla pequeña de símbolos que captura la **estructura global**.

## 5. Experimentos

### 5.1. Comparación con variables continuas (CIFAR10)

Con la misma arquitectura estándar de VAE sobre CIFAR10, variando la capacidad latente, se comparan VAE continuo, VQ-VAE y VIMCO. Resultados en *bits/dim* (cotas inferiores): **VAE 4.51, VQ-VAE 4.67, VIMCO 5.14**. El VAE continuo es comparable al 4.54 reportado para un VAE convolucional profundo. La conclusión: el VQ-VAE es el **primer modelo de latentes discretos que reta el desempeño de los VAE continuos**, obteniendo buenas reconstrucciones con la representación comprimida y simbólica.

### 5.2. Imágenes (ImageNet, DeepMind Lab)

Se modelan imágenes de $128 \times 128 \times 3$ comprimiéndolas a un espacio discreto $z = 32 \times 32 \times 1$ con $K = 512$ —una **reducción de ~42.6x en bits** vía un $p(x|z)$ puramente deconvolucional. Las reconstrucciones se ven solo **ligeramente más borrosas** que los originales pese a la fuerte reducción de dimensionalidad (los autores sugieren que una pérdida perceptual tipo GAN, en vez de MSE sobre píxeles, mejoraría esto; lo dejan a futuro). Entrenando luego una PixelCNN sobre el espacio latente $32 \times 32 \times 1$, las muestras decodificadas (zorro, ballena gris, oso, mariposa, arrecife, etc.) son coherentes.

El experimento más demostrativo del anti-colapso usa **DeepMind Lab**: un segundo VQ-VAE con decoder PixelCNN encima del espacio latente del primero, usando **solo 3 latentes** (cada uno con $K=512$) para modelar la imagen completa —es decir, comprimiendo a $3 \times 9 = 27$ bits, menos que un float32. Este montaje normalmente *rompe* a los VAE por posterior collapse, pero el VQ-VAE **usa los latentes de forma significativa**: la reconstrucción no es perfecta (no puede serlo a 27 bits), pero conserva la disposición de la sala, las paredes cercanas y las texturas, que la PixelCNN **genera proceduralmente** en vez de almacenar los valores de píxel.

### 5.3. Audio (VCTK, LibriSpeech)

Con una arquitectura de encoder/decoder estilo WaveNet sobre habla cruda (VCTK, 109 hablantes), el VQ-VAE comprime el *waveform* a un espacio latente **64x más pequeño**. Tres hallazgos:

- **Abstracción del contenido:** la reconstrucción tiene el **mismo contenido** (mismo texto) pero distinta forma de onda y prosodia. Sin supervisión lingüística, el modelo aprendió un espacio de alto nivel invariante a detalles de bajo nivel que codifica solo el contenido del habla.
- **Modelo de lenguaje fonémico no supervisado:** entrenando el prior (sobre 460 hablantes, resolución 128x menor) y muestreando incondicionalmente, las muestras contienen **palabras y fragmentos de frases claros** —frente al "balbuceo" de un WaveNet incondicional. Al mapear cada uno de los 128 valores latentes a uno de 41 fonemas, la **exactitud es 49.3%** (vs. 7.2% de un latente aleatorio): los códigos discretos son descriptores de alto nivel estrechamente relacionados con fonemas.
- **Conversión de hablante:** extrayendo los latentes de un hablante y reconstruyendo con el *id* de otro, se transfiere la voz manteniendo el contenido —prueba de que la representación factoriza la información específica del hablante.

### 5.4. Video (DeepMind Lab)

Se entrena un modelo generativo condicionado en una secuencia de acciones. Dadas 6 cuadros iniciales, se generan 10 cuadros **enteramente en el espacio latente** ($z_t$), sin pasar por píxeles, y solo al final se decodifican a imágenes. El VQ-VAE puede así "imaginar" secuencias largas en el espacio latente, generando cuadros coherentes con la acción ("avanzar", "girar a la derecha") sin degradación visual y manteniendo la geometría local correcta.

## 6. Limitaciones reconocidas

- **Reconstrucción borrosa con MSE.** Las reconstrucciones de imagen son algo más borrosas que los originales; los autores reconocen que una pérdida perceptual / adversarial (GAN) mejoraría la nitidez y lo dejan a trabajo futuro. (Este es precisamente el hueco que VQ-GAN cerraría después.)
- **Prior entrenado por separado.** El prior autorregresivo se ajusta *después* del VQ-VAE; entrenar ambos conjuntamente podría reforzar los resultados, pero queda como investigación futura.
- **El soft-to-hard no funcionó desde cero.** El enfoque de relajación continua de la VQ (anelada) no logró entrenar desde cero porque el decoder invertía la relajación; de ahí la elección del straight-through duro.
- **Aproximación de la verosimilitud.** La evaluación de $\log p(x) \approx \log p(x|z_q(x))p(z_q(x))$ descansa en el supuesto MAP de que el decoder converge a no asignar masa a $z \neq z_q(x)$; es una aproximación (acotada por debajo vía Jensen).
- **El significado del código depende del contexto.** Los pares encoder/decoder pueden hacer que el significado de cada latente dependa de los anteriores (bi/tri-gramas), de modo que el mapeo simple a fonemas subestima la información codificada.

## 7. Impacto

El VQ-VAE inauguró el paradigma de **"comprimir a un código discreto + modelar los códigos con un autorregresivo potente"** que domina buena parte de la generación moderna:

- **VQ-VAE-2** (Razavi et al., 2019) escaló la idea a jerarquías multi-escala de códigos, produciendo imágenes de alta fidelidad competitivas con GANs.
- **VQ-GAN** (Esser et al., 2021) reemplazó la pérdida MSE por una **pérdida perceptual + adversarial** sobre el autoencoder discreto —cerrando justo la limitación de §6— y modeló los códigos con un Transformer, habilitando síntesis de alta resolución.
- **DALL-E** (Ramesh et al., 2021) usó un autoencoder discreto (dVAE, con relajación Gumbel) para tokenizar imágenes y un Transformer autorregresivo sobre tokens de texto e imagen: el VQ-VAE es el ancestro directo de la idea de "tokens de imagen".
- **Latent / Stable Diffusion** (Rombach et al., 2022) movió la difusión del espacio de píxeles a un **espacio latente comprimido** por un autoencoder; Rombach et al. ofrecen explícitamente una **regularización VQ** (*VQ-reg*) como una de las opciones de ese autoencoder, heredando directamente el codebook del VQ-VAE.

En síntesis, la pregunta que el VQ-VAE respondió —cómo aprender un espacio latente discreto, comprimido y semánticamente útil sin que el decoder lo ignore— resultó ser el cimiento sobre el que se construyó la generación a gran escala de la era 2019–2022.

## 8. Conexión con la Clase 29 (Modelos Generativos en Visión)

El VQ-VAE complementa de forma natural la sección de **autoencoders y VAE** de la clase, aportando la **variante discreta** del cuello de botella latente:

- **Del VAE continuo al VQ-VAE.** La clase introduce el VAE (Kingma & Welling, 2013) con su latente continuo gaussiano y el truco de reparametrización. El VQ-VAE muestra el camino alternativo —latente discreto vía codebook + nearest neighbour + straight-through— y, de paso, da la solución canónica al **posterior collapse** que aparece al combinar VAEs con decoders potentes. Es el contraste pedagógico ideal: *misma arquitectura encoder-bottleneck-decoder, distinta naturaleza del bottleneck*.

- **El latente comprimido como puente hacia Stable Diffusion.** La lección transversal de la clase es que la generación de alta resolución no ocurre en el espacio de píxeles sino en un **espacio latente compacto**. El VQ-VAE es la primera demostración limpia de ese espacio (reducción de ~42.6x en imágenes; generación de video puramente en latentes). **Rombach et al. (2022)** explotan exactamente esa idea, y ofrecen la **VQ-reg** —derivada directa del codebook de este paper— como una opción de su autoencoder. Quien entienda el VQ-VAE entiende *por qué* Latent Diffusion funciona y de dónde viene su autoencoder.

- **El codebook como tokenización de imágenes.** El salto de "píxeles" a "tokens discretos de imagen" que habilitan VQ-GAN y DALL-E nace aquí. Para una clase de visión generativa, el VQ-VAE es la bisagra entre los modelos de verosimilitud explícita (VAE) y los pipelines basados en Transformers/difusión.

**Enlaces internos del curso:**

- [Variational Autoencoder (Kingma & Welling, 2013)](/papers/kingma-vae-2013) — el VAE continuo del que el VQ-VAE es la variante discreta.
- [High-Resolution Image Synthesis with Latent Diffusion Models (Rombach et al., 2022)](/papers/rombach-latentdiffusion-2022) — usa un autoencoder latente con opción de regularización VQ heredada de este paper.
- [Fundamento: Modelos Generativos](/fundamentos/modelos-generativos) — marco general (VAE, GAN, autorregresivos, difusión) en el que encaja el VQ-VAE.
- [Clase 29 — Modelos Generativos en Visión](/clases/clase-29) — hub de la clase.
