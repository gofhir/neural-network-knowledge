# Denoising Diffusion Probabilistic Models — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Denoising Diffusion Probabilistic Models* (DDPM).
- **Autores:** Jonathan Ho, Ajay Jain, Pieter Abbeel — los tres en UC Berkeley.
- **Venue:** 34th Conference on Neural Information Processing Systems (**NeurIPS 2020**), Vancouver, Canadá.
- **Preprint:** arXiv:2006.11239v2 (16 dic 2020), [arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239).
- **Código:** [github.com/hojonathanho/diffusion](https://github.com/hojonathanho/diffusion) (TensorFlow, entrenado en Cloud TPUs vía TFRC).
- **Resultado titular:** en **CIFAR-10 incondicional**, *Inception Score* de 9.46 y **FID de 3.17** — estado del arte en el momento, mejor que la mayoría de los modelos publicados *incluidos los condicionales por clase*.

Este es uno de los papers más influyentes de la última década en visión generativa, pero conviene leerlo con precisión sobre qué reclama y qué no. DDPM **no inventa** los modelos de difusión: esos vienen de Sohl-Dickstein et al. (2015), "Deep unsupervised learning using nonequilibrium thermodynamics". Lo que hace Ho et al. es **demostrar por primera vez que los modelos de difusión pueden generar muestras de alta calidad**, comparables o mejores que las de las GANs de la época, y lo logran mediante una combinación específica de decisiones de parametrización que el paper presenta como su contribución central. La frase clave de la introducción es modesta y exacta: "to the best of our knowledge, there has been no demonstration that they are capable of generating high quality samples. We show that diffusion models actually are capable of generating high quality samples".

La idea es un modelo de variable latente de la forma $p_\theta(x_0) := \int p_\theta(x_{0:T})\,dx_{1:T}$, donde los latentes $x_1,\dots,x_T$ tienen **la misma dimensionalidad que el dato** $x_0$ (a diferencia de un VAE, donde el latente suele ser de menor dimensión). El modelo se define como una **cadena de Markov parametrizada** entrenada por inferencia variacional para producir muestras que igualen los datos tras un tiempo finito. Las transiciones de esa cadena se aprenden para **revertir** un proceso de difusión: una cadena de Markov fija que añade ruido gaussiano gradualmente hasta destruir la señal. Cuando el ruido se añade en cantidades pequeñas, basta con que las transiciones de muestreo sean también gaussianas condicionales, lo que habilita una parametrización por red neuronal "particularmente simple".

La segunda contribución, que el paper considera "una de nuestras contribuciones primarias", es teórica: establece una **equivalencia explícita** entre una cierta parametrización de los modelos de difusión y el *denoising score matching* sobre múltiples niveles de ruido (Vincent 2011; Song & Ermon 2019), con dinámica de Langevin recocida (*annealed Langevin*) durante el muestreo. Esa equivalencia justifica el objetivo simplificado que da los mejores resultados.

Un matiz honesto que el paper reconoce: pese a la calidad de muestreo, sus *log-likelihoods* no son competitivos con otros modelos basados en verosimilitud (Sparse Transformer logra 2.80 bits/dim; DDPM, ≤3.70). Más de la mitad de la longitud de código sin pérdida se gasta describiendo detalles imperceptibles de la imagen — lo que el paper reinterpreta elegantemente diciendo que los modelos de difusión tienen un sesgo inductivo que los hace **excelentes compresores con pérdida**.

## 2. Contexto histórico: de la termodinámica de no-equilibrio (2015) al renacimiento de 2020

El antecedente directo es Sohl-Dickstein, Weiss, Maheswaranathan y Ganguli (ICML 2015). La intuición física de aquel trabajo —de la que DDPM hereda todo el andamiaje matemático— viene de la **termodinámica de no-equilibrio**: un sistema en un estado estructurado de baja entropía (una distribución de datos complicada) se puede llevar de forma controlada hacia un estado de equilibrio simple y de alta entropía (ruido gaussiano isotrópico) mediante un proceso de difusión que añade ruido infinitesimal en muchos pasos. Si cada paso es suficientemente pequeño, el proceso inverso —de ruido a datos— tiene la *misma forma funcional* (también gaussiano), y por tanto puede aprenderse. Esa simetría forward/reverse para pasos pequeños es la observación de Feller que sostiene todo el método.

El trabajo de 2015 era matemáticamente correcto y conceptualmente bello, pero **nunca produjo imágenes que compitieran**. Durante cinco años los modelos de difusión fueron una curiosidad teórica mientras el campo de la generación de imágenes lo dominaban las GANs (Goodfellow 2014; BigGAN, StyleGAN), los modelos autorregresivos (PixelCNN, PixelRNN, Sparse Transformer), los *flows* (Glow, RealNVP) y los VAEs. Paralelamente, una segunda línea —los modelos basados en *score matching* de Song & Ermon (2019, NCSN)— mostraba que estimar el gradiente de la densidad ($\nabla_x \log p(x)$, el "score") sobre múltiples escalas de ruido y muestrear con dinámica de Langevin recocida producía imágenes comparables a GANs. DDPM es, en cierto sentido, **el punto donde estas dos líneas convergen**: Ho et al. demuestran que su difusión con $\epsilon$-prediction es matemáticamente equivalente al score matching de NCSN, unificando dos comunidades que trabajaban en paralelo.

El renacimiento, entonces, no fue por una idea nueva sino por **ingeniería de la parametrización y de la pérdida**: predecir el ruido en lugar de la media, usar una pérdida cuadrática simple sin pesos, una *variance schedule* lineal pequeña, $T=1000$ pasos, una U-Net con atención y embeddings sinusoidales de tiempo. El impacto fue inmediato: DDPM es el cimiento técnico directo de *Improved DDPM* (Nichol & Dhariwal 2021), *Guided Diffusion* / *Diffusion Models Beat GANs* (Dhariwal & Nichol 2021), *Classifier-Free Guidance* (Ho & Salimans 2022 — del mismo primer autor), *Latent Diffusion* / Stable Diffusion (Rombach et al. 2022), DALL·E 2, Imagen y, en la práctica, toda la generación moderna de imágenes por difusión.

## 3. Contribución central

La aportación de DDPM se puede descomponer en cuatro piezas que actúan juntas:

1. **La demostración empírica.** Por primera vez, un modelo de difusión alcanza calidad de muestra de estado del arte (FID 3.17 en CIFAR-10), validando una clase de modelos que llevaba cinco años inerte.

2. **La parametrización por predicción de ruido ($\epsilon$-prediction).** En vez de que la red prediga la media $\mu_\theta(x_t,t)$ de la transición inversa (lo más directo), DDPM la reparametriza para que **prediga el ruido $\epsilon$** que se añadió. Esta es la decisión que más impacto tiene en la calidad y la que conecta con score matching.

3. **El objetivo simplificado $L_\text{simple}$.** Una pérdida cuadrática sin pesos sobre el ruido predicho, que descarta la ponderación del *bound* variacional original. Es más simple de implementar y produce *mejores* muestras (aunque peores verosimilitudes).

4. **La conexión con score matching y Langevin.** La prueba de que entrenar con $L_\text{simple}$ y $\epsilon$-prediction equivale a *denoising score matching* sobre múltiples escalas de ruido, y que el muestreo iterativo se parece a dinámica de Langevin con $\epsilon_\theta$ como gradiente aprendido de la densidad.

La unidad de diseño es que las cuatro se refuerzan: la parametrización (2) habilita el objetivo (3), y ambas son justificadas por la equivalencia teórica (4), que a su vez explica *por qué* funciona tan bien empíricamente (1).

## 4. El proceso forward: convertir datos en ruido gradualmente

El **proceso forward** (o proceso de difusión) es la posterior aproximada $q(x_{1:T}|x_0)$. A diferencia de un VAE, **no tiene parámetros aprendibles**: está fijado como una cadena de Markov que añade ruido gaussiano según una *variance schedule* $\beta_1,\dots,\beta_T$:

$$q(x_{1:T}|x_0) := \prod_{t=1}^{T} q(x_t|x_{t-1}), \qquad q(x_t|x_{t-1}) := \mathcal{N}\big(x_t;\; \sqrt{1-\beta_t}\,x_{t-1},\; \beta_t I\big)$$

Cada paso escala el dato anterior por $\sqrt{1-\beta_t}$ (lo encoge ligeramente hacia el origen) y le suma ruido gaussiano de varianza $\beta_t$. Tras $T$ pasos con la schedule adecuada, $x_T$ es prácticamente ruido gaussiano puro $\mathcal{N}(0,I)$, sin información del dato original.

La **propiedad que lo hace tratable** —y que es central para el entrenamiento eficiente— es que el forward admite muestrear $x_t$ en un paso $t$ arbitrario **en forma cerrada**, sin recorrer los $t$ pasos. Definiendo $\alpha_t := 1-\beta_t$ y $\bar\alpha_t := \prod_{s=1}^{t}\alpha_s$:

$$q(x_t|x_0) = \mathcal{N}\big(x_t;\; \sqrt{\bar\alpha_t}\,x_0,\; (1-\bar\alpha_t)I\big)$$

Esto significa que se puede escribir directamente $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ con $\epsilon\sim\mathcal{N}(0,I)$. El parámetro $\bar\alpha_t$ controla una interpolación: con $\bar\alpha_t \to 1$ (t pequeño) $x_t$ es casi $x_0$; con $\bar\alpha_t \to 0$ (t grande) $x_t$ es casi ruido puro. Esta fórmula cerrada es lo que permite, durante el entrenamiento, **muestrear un $t$ uniforme al azar y saltar directamente a $x_t$** desde $x_0$ sin simular la cadena — la diferencia entre un entrenamiento factible y uno prohibitivo.

Otra cantidad clave: la **posterior del forward condicionada en $x_0$**, $q(x_{t-1}|x_t,x_0)$, también es gaussiana y tratable:

$$q(x_{t-1}|x_t,x_0) = \mathcal{N}(x_{t-1};\; \tilde\mu_t(x_t,x_0),\; \tilde\beta_t I)$$

con $\tilde\mu_t(x_t,x_0) = \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t$ y $\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$. Esta posterior es la "respuesta correcta" que el modelo intenta imitar — el blanco contra el cual se compara la transición aprendida mediante divergencia KL. El detalle a notar es que esta posterior solo es tratable *porque está condicionada en $x_0$*: la posterior marginal $q(x_{t-1}|x_t)$ sin condicionar es intratable (requiere integrar sobre todos los datos), pero durante el entrenamiento $x_0$ está disponible, así que la KL del *bound* compara dos gaussianas conocidas. Es la misma idea que hace tratable el ELBO de un VAE, llevada a una cadena de muchos pasos.

## 5. El proceso reverse: aprender a invertir paso a paso

El **proceso reverse** es la distribución conjunta $p_\theta(x_{0:T})$, una cadena de Markov con transiciones gaussianas aprendidas que arranca de $p(x_T)=\mathcal{N}(0,I)$:

$$p_\theta(x_{0:T}) := p(x_T)\prod_{t=1}^{T} p_\theta(x_{t-1}|x_t), \qquad p_\theta(x_{t-1}|x_t) := \mathcal{N}\big(x_{t-1};\; \mu_\theta(x_t,t),\; \Sigma_\theta(x_t,t)\big)$$

La justificación de que las transiciones inversas sean gaussianas es la observación de Feller: cuando los $\beta_t$ son pequeños, forward y reverse comparten forma funcional. Generar una imagen consiste en muestrear $x_T\sim\mathcal{N}(0,I)$ y aplicar $T$ transiciones inversas hasta llegar a $x_0$.

### 5.1. El objetivo variacional y su descomposición

El entrenamiento optimiza el habitual *bound* variacional sobre la log-verosimilitud negativa, que DDPM reescribe (vía Rao-Blackwellización) en una forma de varianza reducida, término a término:

$$\mathbb{E}_q\Big[\underbrace{D_{KL}(q(x_T|x_0)\,\|\,p(x_T))}_{L_T} + \sum_{t>1}\underbrace{D_{KL}(q(x_{t-1}|x_t,x_0)\,\|\,p_\theta(x_{t-1}|x_t))}_{L_{t-1}} - \underbrace{\log p_\theta(x_0|x_1)}_{L_0}\Big]$$

La gracia de esta forma es que todas las KL son **comparaciones entre gaussianas**, calculables en forma cerrada en lugar de estimadas con Monte Carlo de alta varianza. Los tres bloques se tratan por separado:

- **$L_T$ (forward y prior):** como los $\beta_t$ están fijos, el forward no tiene parámetros y $L_T$ es una **constante durante el entrenamiento**, así que se ignora. En sus experimentos $L_T\approx 10^{-5}$ bits/dim, es decir, $x_T$ es indistinguible del prior gaussiano.
- **$L_{1:T-1}$ (transiciones inversas):** el grueso del aprendizaje. Aquí vive la parametrización del ruido (§6).
- **$L_0$ (decoder discreto):** el último paso. Los datos de imagen son enteros en $\{0,\dots,255\}$ escalados a $[-1,1]$; el término $L_0$ usa un decoder discreto independiente derivado de la gaussiana $\mathcal{N}(x_0;\mu_\theta(x_1,1),\sigma_1^2 I)$, integrando la densidad sobre los *bins* de cada nivel de intensidad. Esto garantiza que el *bound* sea una longitud de código sin pérdida de datos discretos, sin necesidad de añadir ruido de dequantización.

### 5.2. La varianza fija

Para $\Sigma_\theta(x_t,t)$, DDPM elige **no aprenderla**: la fija a constantes dependientes del tiempo $\sigma_t^2 I$. Experimentalmente, $\sigma_t^2=\beta_t$ y $\sigma_t^2=\tilde\beta_t$ dan resultados similares (son los dos extremos: óptimo para $x_0\sim\mathcal{N}(0,I)$ y para $x_0$ determinista). La ablación (Tabla 2) muestra que **aprender una $\Sigma$ diagonal vuelve el entrenamiento inestable** y empeora las muestras — un hallazgo que *Improved DDPM* revisaría después, logrando aprenderla con éxito. Por ahora, varianza fija es la decisión ganadora por simplicidad y estabilidad.

## 6. La simplificación clave: predecir el ruido $\epsilon$ en lugar de la media

Aquí está el corazón del paper. Con $p_\theta(x_{t-1}|x_t)=\mathcal{N}(x_{t-1};\mu_\theta(x_t,t),\sigma_t^2 I)$, el término $L_{t-1}$ se reduce a un error cuadrático entre la media verdadera de la posterior y la media predicha:

$$L_{t-1} = \mathbb{E}_q\Big[\frac{1}{2\sigma_t^2}\,\|\tilde\mu_t(x_t,x_0) - \mu_\theta(x_t,t)\|^2\Big] + C$$

La parametrización **más directa** sería que la red prediga $\tilde\mu_t$ directamente. Pero Ho et al. dan un paso astuto: reparametrizan $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ y sustituyen en la fórmula de $\tilde\mu_t$. Tras el álgebra, $\tilde\mu_t$ se expresa en términos de $x_t$ y del ruido $\epsilon$. Esto revela que **basta con que la red prediga $\epsilon$**, y la media inversa se reconstruye analíticamente:

$$\mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t)\Big)$$

Donde $\epsilon_\theta$ es la red neuronal —un aproximador de funciones— **entrenada para predecir el ruido $\epsilon$ que se añadió a $x_0$ para producir $x_t$**. La red no aprende a "limpiar" hacia la media; aprende a *identificar el ruido presente*. Como $x_t$ ya es entrada de la red, predecir $\epsilon$ y predecir $\mu$ son matemáticamente intercambiables, pero empíricamente **predecir $\epsilon$ es muy superior** cuando se combina con la pérdida simplificada.

Sustituyendo esta parametrización en $L_{t-1}$, el término del *bound* colapsa a:

$$\mathbb{E}_{x_0,\epsilon}\Big[\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}\,\big\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\,t)\big\|^2\Big]$$

que es exactamente la forma del *denoising score matching* de NCSN sobre múltiples escalas de ruido. Y como esta expresión es (un término de) el *bound* variacional del proceso inverso tipo Langevin, **optimizar algo que parece score matching equivale a inferencia variacional para ajustar un muestreador tipo Langevin**. Esta es la equivalencia que el paper considera contribución primaria.

### 6.1. $L_\text{simple}$: tirar los pesos

El paso final es pragmático. El *bound* completo lleva el coeficiente $\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}$ delante de cada término. Ho et al. encuentran que **descartar ese peso** —fijarlo a 1— mejora la calidad de muestra y simplifica la implementación:

$$L_\text{simple}(\theta) := \mathbb{E}_{t,x_0,\epsilon}\Big[\big\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\,t)\big\|^2\Big]$$

con $t$ uniforme en $\{1,\dots,T\}$. Es una pérdida MSE entre el ruido real y el predicho — desarmadoramente simple para lo que logra. El efecto del *reweighting* implícito es que $L_\text{simple}$ **infrapondera los términos de $t$ pequeño** (donde el ruido es mínimo y la tarea de denoising trivial) y deja que la red se concentre en los $t$ grandes (denoising difícil). La ablación (Tabla 2) confirma que este reweighting es lo que lleva FID de 13.51 (bound verdadero, varianza fija) a 3.17 ($L_\text{simple}$). El precio: la verosimilitud empeora ligeramente (≤3.70 → ≤3.75 bits/dim), confirmando que $L_\text{simple}$ es un *bound* variacional ponderado que prioriza percepción sobre compresión exacta.

## 7. Método: U-Net, entrenamiento y muestreo

### 7.1. La red $\epsilon_\theta$: una U-Net

La red que predice el ruido es una **U-Net** (Ronneberger, Fischer & Brox 2015), la misma arquitectura encoder-decoder con *skip connections* nacida para segmentación biomédica. DDPM usa un *backbone* tipo PixelCNN++ sin enmascarar, con **group normalization** en todo el modelo. Detalles arquitectónicos clave:

- **Parámetros compartidos a través del tiempo:** una sola red sirve para todos los $T=1000$ pasos. El paso $t$ se le comunica mediante **embeddings sinusoidales de posición** estilo Transformer (Vaswani et al. 2017), inyectados en cada bloque residual. Esto deja que la misma red adapte su comportamiento según el nivel de ruido.
- **Auto-atención** en la resolución de *feature map* de $16\times16$ (capas *non-local*, Wang et al. 2018), que captura dependencias de largo alcance que la convolución pura no alcanza.

La elección de la U-Net no es accidental: su estructura multi-escala con *skips* es ideal para una tarea que opera simultáneamente sobre estructura global (qué objeto hay) y detalle local (texturas), y se ha mantenido como la columna vertebral de los modelos de difusión hasta hoy.

### 7.2. Entrenamiento (Algoritmo 1)

El bucle de entrenamiento es notablemente compacto, y es la traducción directa de $L_\text{simple}$:

1. Muestrear $x_0\sim q(x_0)$ (una imagen real).
2. Muestrear $t\sim\text{Uniform}(\{1,\dots,T\})$.
3. Muestrear $\epsilon\sim\mathcal{N}(0,I)$.
4. Dar un paso de descenso de gradiente sobre $\nabla_\theta\,\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\,t)\|^2$.
5. Repetir hasta converger.

La fórmula cerrada de $q(x_t|x_0)$ (§4) es lo que permite el salto directo a $x_t$ en el paso 4: no se simula la cadena. Se optimizan términos aleatorios del *bound* con SGD, una forma de estimación estocástica del objetivo.

### 7.3. Muestreo (Algoritmo 2)

El muestreo es iterativo y va de $T$ a $1$:

1. $x_T\sim\mathcal{N}(0,I)$.
2. Para $t=T,\dots,1$: muestrear $z\sim\mathcal{N}(0,I)$ (con $z=0$ si $t=1$) y calcular
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\Big) + \sigma_t z$$
3. Devolver $x_0$.

Este procedimiento "se parece a la dinámica de Langevin con $\epsilon_\theta$ como gradiente aprendido de la densidad de datos". Cada paso resta una fracción del ruido predicho y añade una pizca de ruido fresco (excepto el último). DDPM también muestra que se puede **predecir progresivamente** $\hat{x}_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t))/\sqrt{\bar\alpha_t}$ en cualquier paso intermedio: en la generación progresiva, **las características de gran escala aparecen primero y los detalles finos al final** — un eco de la *conceptual compression*.

## 8. Experimentos y resultados

DDPM fija $T=1000$ y una *variance schedule* **lineal** de $\beta_1=10^{-4}$ a $\beta_T=0.02$ (constantes pequeñas relativas a datos en $[-1,1]$).

- **CIFAR-10 incondicional (Tabla 1):** IS = 9.46, **FID = 3.17** (respecto al *training set*, práctica estándar; 5.24 respecto al *test set*). El FID de 3.17 supera a la mayoría de la literatura, incluyendo modelos *condicionales por clase*, y queda solo detrás de StyleGAN2+ADA (FID 2.67 condicional / 3.26 incondicional). Versus los rivales directos: NCSN 25.32, SNGAN 21.7, EBM 38.2. *NLL*: ≤3.75 bits/dim con $L_\text{simple}$ — no competitivo con Sparse Transformer (2.80), confirmando la disociación entre calidad de muestra y verosimilitud.
- **LSUN $256\times256$ (Tabla 3):** FID 7.89 (Church), **4.90** (Bedroom, modelo grande), 19.75 (Cat) — calidad similar a ProgressiveGAN, aunque por debajo de StyleGAN/StyleGAN2.
- **CelebA-HQ $256\times256$:** muestras de alta fidelidad (Figura 1), e **interpolaciones** suaves en espacio latente (Figura 8): codificando dos imágenes con $q$ a $t=500$, interpolando linealmente los latentes y decodificando con el reverse, se obtienen transiciones plausibles que varían pose, tono de piel, peinado y expresión.
- **Ablación (Tabla 2):** la tabla más informativa del paper. Predecir $\tilde\mu$ con MSE sin pesos es inestable (entradas en blanco). Aprender $\Sigma$ diagonal es inestable. Predecir $\tilde\mu$ con el *bound* verdadero y $\Sigma$ fija da FID 13.22. Predecir $\epsilon$ con $\Sigma$ fija y *bound* verdadero da 13.51. Y predecir $\epsilon$ con **$L_\text{simple}$ da 3.17** — el salto cuantitativo que valida la tesis del paper.

## 9. Limitaciones reconocidas

- **Muestreo lento — la limitación dominante.** Generar una imagen requiere **$T$ pasos secuenciales** (1000 en el paper), cada uno una evaluación completa de la U-Net. No se puede paralelizar a lo largo del tiempo porque cada $x_{t-1}$ depende de $x_t$. Frente a una GAN que genera en un solo *forward pass*, DDPM es órdenes de magnitud más lento en inferencia. Esta es la limitación que motivó casi toda la investigación posterior de aceleración (DDIM de Song et al. 2021, destilación progresiva, solvers de EDO, modelos de consistencia). El propio paper lo nota: "las difusiones gaussianas pueden acortarse para muestreo rápido o alargarse para más expresividad".
- **Verosimilitudes no competitivas.** Como ya se dijo, ≤3.75 bits/dim queda lejos de los modelos autorregresivos. Más de la mitad del *codelength* describe distorsiones imperceptibles (la *rate* es 1.78 bits/dim y la *distortion* 1.97, RMSE 0.95 sobre 255). DDPM reencuadra esto como una virtud (compresión con pérdida), pero como modelo de densidad pura es subóptimo.
- **Varianza inversa no aprendida.** Aprender $\Sigma_\theta$ desestabiliza el entrenamiento; el paper la fija. *Improved DDPM* mostraría después que sí se puede aprender con un objetivo híbrido, mejorando log-likelihood y permitiendo menos pasos.
- **Compresión progresiva solo conceptual.** El esquema de compresión con pérdida progresiva (Algoritmos 3 y 4) depende de *minimal random coding*, intratable en alta dimensión; es prueba de concepto, no sistema práctico.

## 10. Impacto: el cimiento de la generación moderna de imágenes

DDPM es, sin exageración, **el paper que detonó la era de la difusión**. Tras él, en menos de dos años: Nichol & Dhariwal (2021) mejoraron la schedule (cosine) y aprendieron la varianza; Dhariwal & Nichol (2021) demostraron que la difusión con *classifier guidance* **supera a las GANs** en ImageNet; Ho & Salimans (2022, el mismo primer autor) introdujeron *classifier-free guidance*, hoy ubicuo; Rombach et al. (2022) movieron la difusión a un espacio latente comprimido (*Latent Diffusion* → Stable Diffusion), haciéndola viable a alta resolución en hardware de consumo. DALL·E 2, Imagen, Midjourney y prácticamente todo generador texto-a-imagen comercial descansan sobre la maquinaria de DDPM: forward gaussiano, $\epsilon$-prediction, U-Net con embedding de tiempo, muestreo iterativo. La afirmación de que "esperamos investigar su utilidad en otras modalidades" se cumplió con creces: audio, video, 3D, moléculas, política en RL.

El legado conceptual es igual de importante: DDPM **reconcilió** las dos tradiciones que llevaban años separadas —los modelos de difusión variacionales de Sohl-Dickstein y el score matching de Song & Ermon— mostrando que eran la misma cosa vista desde dos ángulos. Esa unificación abrió el marco continuo de las *ecuaciones diferenciales estocásticas* (Song et al. 2021), que subsume a ambos.

## 11. Comparación de paradigmas generativos

El paper se sitúa en un ecosistema de familias generativas con perfiles de *trade-off* distintos. La Clase 29 las contrasta así:

| Familia | Velocidad de muestreo | Calidad de muestra | Cobertura de la distribución (diversidad) |
| --- | --- | --- | --- |
| **GAN** | Rápida (un *forward pass*) | Alta | Baja (riesgo de *mode collapse*) |
| **VAE** | Rápida | Media (muestras borrosas) | Alta |
| **Difusión (DDPM)** | **Lenta** (miles de pasos) | **Alta** | **Alta** |

La posición de la difusión en esta tabla explica su dominio: combina la **alta calidad** de las GANs con la **alta cobertura de la distribución** de los modelos de verosimilitud (sin el *mode collapse* que aqueja a las GANs), a cambio de pagar el precio de un **muestreo lento**. Ese único costo —la lentitud— es exactamente lo que la investigación posterior atacó, dejando intactas las dos ventajas.

## 12. Conexión con la Clase 29 (Modelos Generativos en Visión)

La Clase 29 presenta los modelos de difusión con una frase que es el resumen exacto de DDPM: *"convertir datos en ruido gradualmente y aprender a invertir paso a paso"*, con la **U-Net como el aprendiz del paso inverso**. Cada pieza del paper mapea a la narrativa de la clase:

- **"Convertir datos en ruido gradualmente"** es el proceso forward $q(x_t|x_{t-1})$ (§4): la cadena fija que añade ruido gaussiano según la *variance schedule* $\beta_t$, hasta que $x_T$ es ruido puro. La clase enfatiza que este proceso *no se aprende* — es el "profesor" que define la tarea.
- **"Aprender a invertir paso a paso"** es el proceso reverse $p_\theta(x_{t-1}|x_t)$ (§5): la cadena de Markov aprendida que parte de $\mathcal{N}(0,I)$ y reconstruye el dato. "Paso a paso" es literal: $T=1000$ transiciones secuenciales (de ahí la lentitud de la tabla de §11).
- **"La U-Net como el aprendiz del paso inverso"** es $\epsilon_\theta$ (§7.1): la red que, dado $x_t$ y el paso $t$ (vía embedding sinusoidal), **predice el ruido $\epsilon$** presente. La clase usa la U-Net porque su estructura multi-escala con *skip connections* —heredada de la segmentación biomédica de Ronneberger 2015— es ideal para operar sobre estructura global y detalle local a la vez.
- **La simplificación central de la clase** —que la red no predice la imagen ni la media, sino *el ruido*— es la $\epsilon$-prediction de §6, con su pérdida MSE $L_\text{simple} = \mathbb{E}\|\epsilon - \epsilon_\theta(x_t,t)\|^2$. Es el punto donde un estudiante entiende por qué la difusión es entrenable: el objetivo se reduce a "adivina qué ruido le eché a esta imagen".

Esta clase pertenece al módulo de **modelos generativos en visión**, donde DDPM ocupa el lugar de bisagra: tras VAEs (Kingma & Welling 2013) y GANs (Goodfellow 2014, StyleGAN, DCGAN), la difusión es la familia que termina dominando, y este paper es su acta de nacimiento práctica.

**Enlaces internos:**

- Fundamento transversal: [/fundamentos/modelos-generativos](/fundamentos/modelos-generativos) — el marco común de VAE/GAN/difusión y sus trade-offs.
- Fundamento específico: [/fundamentos/modelos-de-difusion](/fundamentos/modelos-de-difusion) — forward/reverse, $\epsilon$-prediction, score matching, schedules.
- Clase: [/clases/clase-29](/clases/clase-29) — Modelos Generativos en Visión.
- Paper relacionado: [/papers/unet-ronneberger-2015](/papers/unet-ronneberger-2015) — la arquitectura U-Net que DDPM adopta como $\epsilon_\theta$.
