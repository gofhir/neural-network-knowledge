# Tackling the Generative Learning Trilemma with Denoising Diffusion GANs — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Tackling the Generative Learning Trilemma with Denoising Diffusion GANs*.
- **Autores:** Zhisheng Xiao (The University of Chicago; trabajo realizado durante una pasantía en NVIDIA), Karsten Kreis (NVIDIA), Arash Vahdat (NVIDIA).
- **Venue:** ICLR 2022 (publicado como *conference paper*).
- **Año:** 2021 (preprint). **Preprint:** arXiv:2112.07804v2 (4 abr 2022), [arxiv.org/abs/2112.07804](https://arxiv.org/abs/2112.07804).
- **Sitio / código:** [nvlabs.github.io/denoising-diffusion-gan](https://nvlabs.github.io/denoising-diffusion-gan).

Este paper introduce un marco conceptual —el **trilema del aprendizaje generativo** (*generative learning trilemma*)— y propone un modelo, los **denoising diffusion GANs**, para atacarlo. La tesis se puede resumir en una frase: los modelos generativos profundos rara vez satisfacen a la vez tres requisitos clave —(1) **alta calidad** de muestras, (2) **muestreo rápido** y computacionalmente barato, y (3) **buena cobertura de modos** y diversidad—; casi siempre sacrifican uno por los otros dos. Los autores nombran ese compromiso de tres vías como el trilema y muestran, con una figura-resumen (Fig. 1), cómo cada familia clásica falla en uno de los vértices: **las GANs** generan muestras de alta calidad rápidamente pero tienen pobre cobertura de modos; **los VAEs y normalizing flows** cubren los modos fielmente pero sufren de baja calidad; y **los modelos de difusión** logran a la vez alta calidad y buena cobertura, pero su muestreo es lentísimo (miles de evaluaciones de red por muestra), lo que impide aplicarlos en muchos escenarios del mundo real.

La contribución central es un diagnóstico seguido de un remedio. El **diagnóstico**: la lentitud de la difusión se debe, de raíz, al **supuesto gaussiano** sobre la distribución de denoising $p_\theta(x_{t-1}\mid x_t)$, que solo es válido en el límite de pasos infinitesimalmente pequeños. El **remedio**: si se quiere dar pasos de denoising *grandes* (y por tanto reducir el número total de pasos), la distribución verdadera deja de ser gaussiana y se vuelve **multimodal**, por lo que hay que modelarla con una distribución expresiva. Los autores la modelan con un **GAN condicional** por paso, obteniendo un modelo que necesita tan solo **~4 pasos** (en lugar de 1000+) y que resulta **~2000× más rápido** que el muestreo predictor-corrector de Song et al. (2021c) en CIFAR-10, manteniendo calidad y cobertura competitivas.

Para la Clase 29 (Modelos Generativos en Visión) este paper es estructural: la clase **cita explícitamente este trabajo** en la slide "Generative learning trilemma" (enlazando a `https://arxiv.org/pdf/2112.07804`), y la tabla comparativa VAE / GAN / Difusión sobre los ejes velocidad / calidad / distribución que aparece en la clase **es justamente el trilema** que este paper articula. Es decir, no es un paper periférico: es el que aporta el *marco de comparación* con el que la clase ordena toda la familia de modelos generativos.

## 2. Contexto histórico: el estado de los modelos generativos hacia 2021 y la formulación del trilema

En la década previa a este paper se desarrolló una gran variedad de modelos generativos profundos para imágenes, audio, nubes de puntos y grafos. Pero —argumentan los autores— ninguno satisfacía simultáneamente los tres requisitos que el mundo real suele exigir. El paper lo plantea con ejemplos concretos: la síntesis de imágenes se ha enfocado mayormente en alta calidad, pero la cobertura de modos y la diversidad importan para **representar mejor a las minorías y reducir los impactos sociales negativos** de los modelos generativos; y aplicaciones como la edición interactiva de imágenes o la síntesis de voz en tiempo real **exigen muestreo rápido**. El reto impuesto por esos tres requisitos a la vez es lo que bautizan como *generative learning trilemma*.

El mapa del trilema (Fig. 1) ubica cada familia:

- **GANs** (Goodfellow et al., 2014; Brock et al., 2018): muestras de alta calidad y rápidas, pero **pobre cobertura de modos** (mode collapse documentado por Salimans et al., 2016; Zhao et al., 2018).
- **VAEs** (Kingma & Welling, 2014) y **normalizing flows** (Dinh et al., 2016; Kingma & Dhariwal, 2018): cubren los modos fielmente (alta verosimilitud) pero **baja calidad** de muestras.
- **Modelos de difusión** (Sohl-Dickstein et al., 2015; Ho et al., 2020; Song et al., 2021c): emergieron como modelos muy potentes, **batiendo a las GANs** en calidad de imagen (Dhariwal & Nichol, 2021) y con buena cobertura de modos (alta verosimilitud); su talón de Aquiles es que muestrear requiere **miles de evaluaciones de red**, haciéndolos caros en la práctica.

El paper se posiciona, entonces, como un intento de reformular la difusión "específicamente para muestreo rápido sin perder cobertura de modos ni calidad". El insight clave —que da origen a todo el método— es una observación clásica de la teoría de procesos estocásticos: el supuesto de que la distribución de denoising puede aproximarse por una gaussiana **solo se sostiene en el límite infinitesimal de pasos pequeños** (Sohl-Dickstein et al., 2015; Feller, 1949). De ahí la necesidad de cientos o miles de pasos en el proceso reverso. Cuando el proceso reverso usa pasos grandes (pocos pasos), se necesita una distribución **no-gaussiana y multimodal**. Intuitivamente, en síntesis de imágenes esa multimodalidad surge del hecho de que **múltiples imágenes limpias plausibles pueden corresponder a la misma imagen ruidosa**.

## 3. Contribución central

Los autores resumen tres contribuciones:

1. **Diagnóstico:** atribuyen el muestreo lento de la difusión al supuesto gaussiano sobre la distribución de denoising, y proponen emplear en su lugar **distribuciones de denoising complejas y multimodales**.
2. **Modelo:** proponen los **denoising diffusion GANs**, un modelo de difusión cuyo **proceso reverso está parametrizado por GANs condicionales** —uno por paso de denoising.
3. **Validación:** mediante evaluaciones cuidadosas, demuestran *varios órdenes de magnitud* de aceleración respecto a los modelos de difusión actuales, tanto en generación como en edición de imágenes, superando el trilema "en gran medida" y haciendo —según afirman— que los modelos de difusión sean **por primera vez aplicables a escenarios interactivos del mundo real a bajo costo computacional**.

La idea de diseño que une todo: en lugar de reducir el tamaño de paso para que la gaussiana sea válida (lo que exige muchos pasos), se mantiene el paso grande y se **reemplaza la gaussiana por un generador GAN condicional** capaz de capturar la multimodalidad de la distribución verdadera. El número de pasos baja de ~1000 a $T \le 8$ (en la práctica $T=4$ es el óptimo).

## 4. Trasfondo: el proceso de difusión y el supuesto gaussiano

En un modelo de difusión hay un **proceso forward** que añade ruido gaussiano a los datos $x_0 \sim q(x_0)$ en $T$ pasos según un *schedule* de varianzas $\beta_t$ predefinido:

$$q(x_{1:T}\mid x_0) = \prod_{t\ge 1} q(x_t\mid x_{t-1}), \qquad q(x_t\mid x_{t-1}) = \mathcal{N}\!\big(x_t;\sqrt{1-\beta_t}\,x_{t-1},\,\beta_t I\big).$$

El **proceso reverso** (denoising) se define como una cadena que parte de ruido $p(x_T)$ y va limpiando:

$$p_\theta(x_{0:T}) = p(x_T)\prod_{t\ge 1} p_\theta(x_{t-1}\mid x_t), \qquad p_\theta(x_{t-1}\mid x_t) = \mathcal{N}\!\big(x_{t-1};\mu_\theta(x_t,t),\,\sigma_t^2 I\big).$$

El entrenamiento maximiza el ELBO, que equivale a hacer *matching* entre la distribución de denoising verdadera $q(x_{t-1}\mid x_t)$ y la parametrizada $p_\theta(x_{t-1}\mid x_t)$ vía divergencia KL. Aquí están los **dos supuestos clave** de la difusión estándar que el paper interroga:

1. La distribución de denoising $p_\theta(x_{t-1}\mid x_t)$ se modela como **gaussiana**.
2. El número de pasos $T$ se asume del **orden de cientos a miles**.

El segundo supuesto es consecuencia del primero. Usando la regla de Bayes, $q(x_{t-1}\mid x_t) \propto q(x_t\mid x_{t-1})\,q(x_{t-1})$, los autores muestran que la distribución de denoising verdadera toma forma gaussiana solo en dos situaciones: (a) en el **límite de $\beta_t$ infinitesimal**, donde el producto de Bayes está dominado por $q(x_t\mid x_{t-1})$ (que es gaussiana) y el reverso hereda esa forma (Feller, 1949); o (b) si la **marginal de datos $q(x_t)$ es gaussiana**. La primera condición obliga a usar miles de pasos con $\beta_t$ pequeño. La segunda fue la apuesta de LSGM (Vahdat et al., 2021), que usa un encoder VAE para acercar los datos a una gaussiana, pero —señala el paper— transformar perfectamente los datos a gaussiano es difícil, por lo que LSGM aún requiere decenas a cientos de pasos en datasets complejos.

La Fig. 2 del paper ilustra el punto visualmente con una distribución de datos 1D multimodal: para un paso pequeño ($q(x_4\mid x_5)$) la denoising verdadera es casi gaussiana, pero **a medida que el paso crece se vuelve cada vez más compleja y multimodal**. Cuando ninguna de las dos condiciones se cumple (paso grande + datos no-gaussianos), no hay garantía alguna de que la gaussiana sea válida.

## 5. Método: modelar el denoising con GANs condicionales

### 5.1. El planteamiento adversarial

El objetivo es reducir $T$. Como las GANs condicionales han demostrado modelar distribuciones condicionales complejas en el dominio de imagen (Mirza & Osindero, 2014; Isola et al., 2017), los autores las adoptan para aproximar $q(x_{t-1}\mid x_t)$. El proceso forward se monta igual que en la difusión estándar, pero ahora con $T$ **pequeño** ($T \le 8$) y cada paso con $\beta_t$ **grande**. El entrenamiento hace *matching* entre el generador $p_\theta(x_{t-1}\mid x_t)$ y $q(x_{t-1}\mid x_t)$ minimizando una divergencia adversarial $D_{\text{adv}}$ por paso:

$$\min_\theta \sum_{t\ge 1} \mathbb{E}_{q(x_t)}\big[D_{\text{adv}}\big(q(x_{t-1}\mid x_t)\,\|\,p_\theta(x_{t-1}\mid x_t)\big)\big].$$

$D_{\text{adv}}$ puede ser distancia de Wasserstein, divergencia de Jensen-Shannon o una f-divergencia, según el setup adversarial. El paper usa **GANs no saturantes** (Goodfellow et al., 2014) —las mismas de StyleGAN—, en cuyo caso $D_{\text{adv}}$ es una instancia de f-divergencia llamada **softened reverse KL** (Shannon et al., 2020), distinta de la KL forward que usa la difusión estándar. Se entrena un **discriminador dependiente del tiempo** $D_\phi(x_{t-1}, x_t, t)$ que decide si $x_{t-1}$ es una versión "denoised" plausible de $x_t$, contrastando muestras falsas del generador contra muestras reales.

### 5.2. Parametrización del modelo implícito de denoising

Una pieza elegante del método: en vez de que el generador prediga $x_{t-1}$ directamente, se parametriza de forma análoga a DDPM. Primero el generador predice una estimación de la imagen limpia $x_0$, y luego $x_{t-1}$ se muestrea de la **posterior gaussiana $q(x_{t-1}\mid x_t, x_0)$** —que *siempre* tiene forma gaussiana cerrada, independiente del tamaño de paso:

$$p_\theta(x_{t-1}\mid x_t) := \int p_\theta(x_0\mid x_t)\,q(x_{t-1}\mid x_t, x_0)\,dx_0 = \int p(z)\,q\big(x_{t-1}\mid x_t,\,x_0 = G_\theta(x_t, z, t)\big)\,dz,$$

donde $G_\theta(x_t, z, t)$ es el generador GAN que recibe la imagen ruidosa $x_t$ y una variable latente $z \sim \mathcal{N}(0,I)$. Las ventajas:

- **Hereda el sesgo inductivo de DDPM:** la estructura de red se puede tomar prestada de DDPM. La diferencia crucial es que en DDPM $x_0$ se predice de forma **determinista** a partir de $x_t$, mientras que aquí $x_0$ lo produce el generador con la **variable latente aleatoria $z$**. Es exactamente esa $z$ la que permite que $p_\theta(x_{t-1}\mid x_t)$ sea **multimodal y compleja**, en contraste con la denoising unimodal de DDPM. La ablación confirma que **quitar $z$ degrada severamente la calidad** (FID 3.75 → 20.6).
- **Facilita la predicción:** como para distintos $t$ la perturbación de $x_t$ es muy distinta, predecir $x_{t-1}$ directamente con una sola red sería difícil; aquí el generador solo predice $x_0$ (sin perturbar) y la perturbación se re-añade vía la posterior.

### 5.3. Ventaja sobre un generador "one-shot"

¿Por qué no entrenar simplemente una GAN tradicional que genere en un solo paso? El paper argumenta —y verifica empíricamente— que descomponer la generación en varios pasos de denoising condicionados en $x_t$ aporta dos beneficios: (1) cada paso es **relativamente simple de modelar** gracias al fuerte condicionamiento en $x_t$, evitando la inestabilidad y el mode collapse de generar de una distribución compleja en un solo disparo; y (2) el proceso de difusión **suaviza la distribución de datos** (Lyu, 2012), haciendo que el discriminador sea **menos propenso a sobreajustar** (a diferencia de una GAN que solo ve muestras limpias). Resultado esperado y confirmado: mejor estabilidad de entrenamiento y mejor cobertura de modos.

## 6. Experimentos

### 6.1. Superando el trilema en CIFAR-10

Los autores adoptan la arquitectura **NCSN++** (U-Net, de Song et al., 2021c) para el generador, con la variable latente $z$ controlando las capas de normalización (reemplazan group normalization por *adaptive group normalization*). Evalúan los tres ejes del trilema: **fidelidad** (FID e Inception Score), **diversidad** (improved recall de Kynkäänniemi et al., 2019) y **tiempo de muestreo** (número de evaluaciones de función, NFE, y tiempo de reloj en una V100).

El resultado central (Tabla 1, generación incondicional en CIFAR-10) con $T=4$:

| Modelo | IS↑ | FID↓ | Recall↑ | NFE↓ | Tiempo (s)↓ |
|---|---|---|---|---|---|
| **Denoising Diffusion GAN (ours), T=4** | 9.63 | 3.75 | 0.57 | **4** | **0.21** |
| DDPM (Ho et al., 2020) | 9.46 | 3.21 | 0.57 | 1000 | 80.5 |
| Score SDE (VE) (Song et al., 2021c) | 9.89 | 2.20 | 0.59 | 2000 | 423.2 |
| LSGM (Vahdat et al., 2021) | 9.87 | 2.10 | 0.61 | 147 | 44.5 |
| StyleGAN2 w/ ADA (Karras et al., 2020a) | 9.83 | 2.92 | **0.49** | 1 | 0.04 |
| SNGAN (Miyato et al., 2018) | 8.22 | 21.7 | **0.44** | 1 | — |

Lecturas clave:

- **Velocidad:** ~**2000× más rápido** que el muestreo predictor-corrector de Score SDE (0.21 s vs 423.2 s) y ~**20× más rápido** que FastDDPM. La Fig. 4 (FID vs tiempo de muestreo, variando el número de pasos) muestra que el modelo domina a la difusión previa en el trade-off calidad/tiempo.
- **Calidad:** FID 3.75 competitivo entre los mejores modelos. Solo StyleGAN2 con ADA es ligeramente mejor en calidad pura.
- **Diversidad:** las GANs tienen recall **por debajo de 0.5** (pobre cobertura); el modelo logra 0.57, superior a varias likelihood-based y competitivo con la difusión. Es decir, **gana en los tres ejes a la vez**, que es justo el punto del trilema.

### 6.2. Ablaciones

- **Número de pasos $T$:** $T=1$ (equivalente a una GAN incondicional, porque $x_t$ casi no informa sobre $x_0$) da resultados muy pobres con baja diversidad (recall 0.19). $T=4$ es el óptimo (FID 3.75, recall 0.57); $T=8$ degrada ligeramente, hipótesis: se necesitaría más capacidad para acomodar más pasos (un GAN condicional por paso).
- **Difusión como data augmentation:** entrenar una GAN one-shot con la difusión forward solo como augmentación da resultados muy inferiores → el modelo **no es equivalente** a aumentar datos antes del discriminador.
- **Parametrización:** producir $x_0$ y muestrear de la posterior supera por amplio margen a (a) predecir $x_{t-1}$ directamente (*direct denoising*) y (b) predecir el ruido $\epsilon$ (*noise generation*, lo más cercano a la difusión clásica).
- **Importancia de la latente $z$:** quitarla convierte el denoising en unimodal y degrada FID a 20.6 → confirma que la **multimodalidad es esencial**.

### 6.3. Cobertura de modos y alta resolución

- **25-Gaussians (toy 2D):** la GAN vanilla colapsa modos; WGAN-GP mejora cobertura pero con calidad limitada; el modelo cubre **todos los modos** con alta calidad usando 4 pasos, mientras DDPM necesita ~500 pasos para igualar calidad.
- **StackedMNIST (1000 modos):** cubre los **1000 modos** con el **menor KL (0.071)**, superando a GANs diseñadas específicamente para cobertura (PacGAN, PresGAN) y a StyleGAN2.
- **Alta resolución (256×256):** competitivo en CelebA-HQ (FID 7.64) y **superando a DDPM e ImageBART** en LSUN Church (FID 5.25 vs 7.89 de DDPM).
- **Síntesis basada en trazos (stroke-based):** aplicado a la tarea de SDEdit (Meng et al., 2021b), genera muestras realistas y diversas que preservan el trazo, con un **speedup de ~1100×** (0.16 s vs 181 s por imagen a 256 px), confirmando viabilidad en edición interactiva.

## 7. Limitaciones reconocidas

- **Capacidad vs número de pasos:** el modelo necesita un GAN condicional por paso, y aumentar $T$ más allá de 4 degrada el rendimiento salvo que se aumente la capacidad. El óptimo $T=4$ es un punto delicado, no un parámetro libremente escalable.
- **Herencia de los problemas de las GANs:** aunque el condicionamiento por pasos y el suavizado mitigan la inestabilidad y el mode collapse, el modelo sigue entrenándose con pérdida adversarial; la estabilidad de entrenamiento se discute en apéndice y no es trivial.
- **Calidad pura aún por debajo del tope:** algunos modelos de difusión con muchos pasos (Score SDE, LSGM) y StyleGAN2 con ADA obtienen mejor FID/IS en CIFAR-10; el aporte del modelo es el *balance* de los tres ejes, no liderar el de calidad aislado.
- **VAE condicional descartado empíricamente:** los autores reportan en nota al pie que probaron un VAE condicional como $p_\theta(x_{t-1}\mid x_t)$ y dio resultados consistentemente pobres; dejan la exploración de otros generadores condicionales expresivos para trabajo futuro. Es decir, la elección del GAN es empírica, no demostrada óptima.

## 8. Impacto

El paper afirma ser, hasta donde los autores saben, **el primer modelo que reduce el costo de muestreo de la difusión a un punto que permite aplicarla a escenarios reales de forma barata**. En su *statement* de ética subrayan que la cobertura de modos y la diversidad son requisitos clave para **reducir sesgos en modelos generativos y mejorar la representación de minorías**, de modo que abaratar la difusión (que ya tenía buena diversidad) ayuda a difundir esos beneficios. El trabajo se inscribe en la oleada 2021–2022 de métodos para acelerar la difusión (destilación, DDIM, schedules no-Markovianos, mejores solvers de SDE), pero se diferencia al cambiar la *naturaleza* de la distribución de denoising en vez de solo afinar el muestreo de una gaussiana. La idea de "una GAN por paso de denoising" influyó en la línea de investigación que combina modelos adversariales y de difusión para muestreo en pocos pasos.

## 9. Conexión con la Clase 29 (Modelos Generativos en Visión)

Este paper es el **andamiaje conceptual** de la clase, no una referencia secundaria:

- **La slide del trilema.** La Clase 29 incluye una slide titulada "Generative learning trilemma" que **enlaza directamente a este paper** (`https://arxiv.org/pdf/2112.07804`). El triángulo con los vértices *High Quality Samples* / *Fast Sampling* / *Mode Coverery & Diversity* que la clase muestra es literalmente la Fig. 1 de Xiao et al.

- **La tabla comparativa VAE / GAN / Difusión es el trilema.** La clase compara las tres familias sobre los ejes **velocidad / calidad / distribución (cobertura)**. Esa tabla operacionaliza exactamente el diagnóstico del paper: GANs rápidas y de calidad pero con mala cobertura; VAEs con buena cobertura pero baja calidad; difusión con calidad y cobertura pero lenta. Entender el paper es entender *por qué* esa tabla está construida así y *qué* casilla intenta llenar cada modelo. El paper es la fuente que articula el marco con el que la clase ordena el panorama.

- **Bisagra entre GANs y difusión.** La clase recorre las GANs (ver [Goodfellow et al., 2014](/papers/goodfellow-gan-2014)) y los modelos de difusión (ver [Ho et al., 2020 — DDPM](/papers/ho-ddpm-2020)). Este paper es precisamente el que **une ambos mundos**: usa el andamiaje de difusión (proceso forward/reverso, parametrización vía $x_0$ y posterior gaussiana, heredada de DDPM) pero reemplaza la gaussiana del denoising por un generador adversarial. Es el ejemplo canónico de que las familias generativas no son compartimentos estancos.

- **Por qué la difusión es lenta, explicado en su raíz.** Para el [fundamento de modelos de difusión](/fundamentos/modelos-de-difusion), este paper aporta la explicación *fundamental* (no solo empírica) de la lentitud: el supuesto gaussiano del denoising solo vale con pasos infinitesimales (Feller, 1949), de ahí los miles de pasos. Esa es la pieza teórica que conecta el algoritmo de DDPM con su costo, y que la clase puede usar para motivar todas las técnicas de muestreo acelerado.

- **Encaje en el panorama de modelos generativos.** Para el [fundamento de modelos generativos](/fundamentos/modelos-generativos), el trilema es el criterio de comparación transversal: cualquier modelo nuevo (VAE, GAN, flow, difusión, o híbridos como este) puede situarse en el triángulo según qué vértice sacrifica. Es la lente con la que la [Clase 29](/clases/clase-29) evalúa toda la familia.
