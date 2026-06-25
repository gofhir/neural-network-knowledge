# GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID / TTUR) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium*.
- **Autores:** Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler y Sepp Hochreiter — todos del **LIT AI Lab & Institute of Bioinformatics, Johannes Kepler University Linz** (Austria). Hochreiter es el coinventor de las LSTM.
- **Venue:** NeurIPS 2017 (31st Conference on Neural Information Processing Systems), Long Beach, CA.
- **Año:** 2017. **Preprint:** arXiv:1706.08500 (v6, 12 ene 2018), [arxiv.org/abs/1706.08500](https://arxiv.org/abs/1706.08500).

Este paper es recordado casi universalmente por **una sola de sus dos contribuciones** —la métrica FID—, pero su título y el grueso de su matemática giran en torno a la otra: una regla de entrenamiento (TTUR) con **prueba de convergencia**. Conviene leerlo entendiendo que son dos aportes técnicamente independientes empaquetados juntos porque ambos atacan la misma frustración de la época: entrenar GANs era un arte negro sin garantías de convergencia y sin una forma confiable de medir si un modelo era mejor que otro.

El punto de partida es la naturaleza adversarial del entrenamiento GAN. Como lo plantea Goodfellow et al. (2014), aprender un GAN es un **juego** entre el generador G —que fabrica datos sintéticos a partir de ruido— y el discriminador D —que intenta separar lo sintético de lo real—. La solución del juego es un **equilibrio de Nash**, no el mínimo de una única función de pérdida. Y ahí está el problema que el paper nombra en su primera página: "*since training GANs is a game and its solution is a Nash equilibrium, gradient descent may fail to converge*". El descenso de gradiente es un optimizador local que persigue el mínimo de una función; aplicado a un juego de suma no nula puede oscilar indefinidamente sin converger. Solo se pueden alcanzar **equilibrios de Nash locales**: puntos del espacio de parámetros donde ni G ni D pueden bajar su pérdida respectiva moviéndose unilateralmente.

Las dos contribuciones del paper:

1. **TTUR (two time-scale update rule):** usar **tasas de aprendizaje distintas** para D y G, y demostrar —vía teoría de aproximación estocástica— que bajo supuestos suaves el entrenamiento **converge a un equilibrio de Nash local estacionario**. Además prueban que Adam, el optimizador de facto, se comporta como una "bola pesada con fricción" (Heavy Ball with Friction) y por eso prefiere mínimos planos y evita el *mode collapse*.
2. **FID (Fréchet Inception Distance):** una métrica para evaluar la calidad de imágenes generadas que compara las **distribuciones de features** (extraídas de una red Inception) de imágenes reales contra generadas, modelándolas como gaussianas y midiendo la distancia de Fréchet entre ellas. Es más consistente con el juicio humano y con la degradación de la imagen que el Inception Score previo.

Para la Clase 29 (Modelos Generativos en Visión) este paper es el ancla de la pregunta "**¿cómo evalúo la calidad de lo que genera mi modelo?**". Las slides de la clase enseñan FID como *la* métrica estándar (menor = mejor, FID = 0 ideal, detecta overfitting/diversidad/realismo) y comparan modelos sobre ImageNet y FFHQ. Entender este paper es entender de dónde sale ese número que aparece en absolutamente toda tabla de GANs y modelos de difusión posteriores a 2017.

## 2. Contexto: el problema de evaluar modelos generativos

El objetivo del aprendizaje generativo es que el modelo produzca datos que **coincidan con la distribución observada**. En principio, entonces, cualquier distancia entre la probabilidad de los datos reales $p_w(\cdot)$ y la del modelo $p(\cdot)$ sirve como medida de desempeño. El problema, como dice el paper citando a Theis et al., es que "*defining appropriate performance measures for generative models is difficult*".

Las opciones disponibles en 2017 eran insatisfactorias:

- **Verosimilitud (likelihood):** la medida más conocida, estimable por *annealed importance sampling*. Pero depende fuertemente de los supuestos de ruido sobre los datos reales y puede estar **dominada por muestras individuales**. Una imagen atípica puede arruinar la estimación. Además, para GANs la verosimilitud no es ni siquiera computable de forma directa: el generador es un mapeo determinista de ruido a imagen, sin densidad explícita tratable. Esa es justamente la razón de ser de los GANs —aprender modelos para los que la máxima verosimilitud es inviable—, lo que vuelve circular usar verosimilitud para evaluarlos.
- **Estimación de densidad:** también tiene desventajas conocidas.

### 2.1. El Inception Score y sus defectos

La mejor medida práctica de la época era el **Inception Score (IS)** de Salimans et al. (2016), que correlaciona con el juicio humano. Funciona así: se pasan las muestras generadas por un modelo **Inception preentrenado en ImageNet** y se mira la distribución de etiquetas predichas. La intuición tiene dos partes:

- Las imágenes con objetos significativos deben tener **baja entropía de etiquetas** $p(y|X)$ — es decir, el clasificador está seguro de qué objeto es, la imagen pertenece claramente a pocas clases.
- A través de todas las imágenes generadas, la distribución marginal de etiquetas $p(y)$ debe tener **alta entropía** — es decir, el modelo genera variedad, no colapsa a una sola clase.

El IS combina ambas en $\exp\!\big(\mathbb{E}_X\, \mathrm{KL}(p(y|X)\,\|\,p(y))\big)$. Tiene además una cota superior limpia: para $m$ muestras y $K$ clases, el IS está acotado por $m$ (cuando $m \le K$ y cada muestra cae en una clase distinta clasificada con probabilidad 1).

El **defecto central** que el paper le señala es contundente: el Inception Score "*does not use the statistics of real world samples and compare it to the statistics of synthetic samples*". El IS solo mira las muestras generadas a través del clasificador; **nunca compara contra los datos reales**. Un modelo podría generar una imagen perfectamente clasificable por cada clase de ImageNet —IS alto— sin que esas imágenes se parezcan en nada, estadísticamente, a las imágenes reales del dataset objetivo. El IS premia "objetividad y variedad de clases", no "fidelidad a la distribución real". El paper lo demuestra empíricamente (Fig. A8): ante degradaciones crecientes de la imagen, el IS "*fluctúa, se mantiene plano o, en el peor caso, decrece*", mientras que la métrica que proponen crece monótonamente con la degradación, como debe hacerlo una buena medida.

## 3. Contribución 1 — TTUR: convergencia a un equilibrio de Nash local

### 3.1. La idea: dos escalas de tiempo

Consideremos un discriminador $D(\cdot; w)$ con parámetros $w$ y un generador $G(\cdot; \theta)$ con parámetros $\theta$. El entrenamiento usa un gradiente estocástico $\tilde{g}(\theta, w)$ de la pérdida del discriminador $L_D$ y un gradiente estocástico $\tilde{h}(\theta, w)$ de la pérdida del generador $L_G$. Importante: el marco **no se restringe a GANs min-max**; vale para cualquier GAN donde $L_D$ no esté necesariamente atada a $L_G$ (incluye Wasserstein GAN). Los gradientes son estocásticos porque usan mini-batches de $m$ muestras reales y $m$ sintéticas, de modo que $\tilde{g} = g + M^{(w)}$ y $\tilde{h} = h + M^{(\theta)}$, con $M$ ruido aleatorio (la diferencia entre el gradiente del mini-batch y el gradiente verdadero).

La regla TTUR usa **tasas de aprendizaje separadas** $b(n)$ para el discriminador y $a(n)$ para el generador:

$$w_{n+1} = w_n + b(n)\,\big(g(\theta_n, w_n) + M_n^{(w)}\big), \qquad \theta_{n+1} = \theta_n + a(n)\,\big(h(\theta_n, w_n) + M_n^{(\theta)}\big).$$

La premisa de diseño, en palabras del paper: el discriminador converge a un mínimo local **cuando el generador está fijo**; si el generador cambia suficientemente despacio, el discriminador igual converge porque las perturbaciones que le impone el generador son pequeñas. Típicamente el generador es la actualización **lenta** (tasa pequeña, escala de tiempo lenta) y el discriminador la **rápida**. Esto formaliza una intuición empírica de la época: las implementaciones que funcionaban entrenaban el discriminador más agresivamente —WGAN hacía 5 pasos de D por cada paso de G—. TTUR logra el mismo efecto con tasas distintas y **una sola actualización de cada red por iteración**, lo que además mejora el desempeño: el discriminador "aprende patrones nuevos antes de que se transfieran al generador", mientras que un generador demasiado rápido empuja al discriminador a regiones nuevas sin que este capture la información acumulada.

### 3.2. La prueba de convergencia

El paper se apoya en la **teoría de aproximación estocástica de dos escalas de tiempo**, cuyo resultado seminal es de Borkar (1997). Bajo los supuestos:

- **(A1)** Los gradientes $h$ y $g$ son Lipschitz. (Consecuencia práctica curiosa: redes con activaciones suaves como **ELU** cumplen, pero **redes ReLU no**, por la no diferenciabilidad en el origen.)
- **(A2)** Condiciones tipo Robbins-Monro sobre las tasas: $\sum a(n) = \infty$, $\sum a^2(n) < \infty$, ídem para $b(n)$, y crucialmente $a(n) = o(b(n))$ — la tasa lenta debe ser asintóticamente despreciable frente a la rápida. **Esta es la condición que define las "dos escalas de tiempo".**
- **(A3)** Los errores de gradiente estocástico son secuencias de diferencias de martingala con segundo momento acotado — se cumple en el entrenamiento por mini-batches.
- **(A4)** Para cada $\theta$, la ODE del discriminador tiene un atractor localmente asintóticamente estable $\lambda(\theta)$, y la ODE del generador (con $w = \lambda(\theta)$) tiene un atractor $\theta^*$. Es decir: D converge a un mínimo para G fijo, y G converge a un mínimo para ese D.
- **(A5)** Los iterados están acotados.

**Teorema 1 (Borkar):** bajo estos supuestos, las actualizaciones convergen casi seguramente a $(\theta^*, \lambda(\theta^*))$.

La solución $(\theta^*, \lambda(\theta^*))$ es un **equilibrio de Nash local estacionario**, porque tanto $\theta^*$ como $\lambda(\theta^*)$ son atractores localmente estables con $g(\theta^*, \lambda(\theta^*)) = 0$ y $h(\theta^*, \lambda(\theta^*)) = 0$ — ninguna de las dos redes puede mejorar moviéndose sola. La idea técnica de la prueba es usar **ODEs perturbadas $(T,\delta)$** (Hirsch 1989): eventualmente llega un instante en que la perturbación de la actualización lenta es lo bastante pequeña (acotada por $\delta$) como para que la rápida converja. Con escalas de tiempo **iguales** solo se puede probar que los iterados revisitan un entorno de la solución infinitas veces, entorno que puede ser muy grande — de ahí la oscilación que se observa empíricamente.

### 3.3. Adam como Heavy Ball with Friction

El segundo resultado teórico, más sutil. Adam (Kingma & Ba) se usa para evitar el *mode collapse* —la patología donde el generador concentra toda la masa de probabilidad en unos pocos modos, perdiendo la variedad del mundo real—. El paper demuestra (**Teorema 2**) que, para segundos momentos del gradiente estacionarios, Adam sigue la ecuación diferencial de la **bola pesada con fricción**:

$$\ddot{\theta}_t + a(t)\,\dot{\theta}_t + \nabla f(\theta_t) = 0.$$

La interpretación física es elegante: el promediado exponencial de gradientes pasados de Adam equivale a una **velocidad** (inercia) que vuelve al generador resistente a ser empujado a regiones pequeñas. Como una bola con masa, **sobrepasa los mínimos locales pequeños** —los que corresponden a mode collapse— y se asienta en **mínimos planos**, que generalizan mejor. La normalización por el segundo momento $\sqrt{v}$ se interpreta como introducir gravedad. Y como Adam admite una función de Lyapunov $E(t) = \frac{1}{2}|\dot{\theta}(t)|^2 + f(\theta(t))$ con $\dot{E}(t) = -a|\dot{\theta}(t)|^2 < 0$, la maquinaria de ODEs perturbadas se extiende a Adam: **GANs entrenados con TTUR y Adam convergen** a un equilibrio de Nash local.

## 4. Contribución 2 — FID: la Fréchet Inception Distance

### 4.1. Derivación

La construcción de FID parte de un principio de teoría de la medida: dos distribuciones $p$ y $p_w$ son iguales (salvo conjunto de medida nula) si y solo si $\int p(\cdot)f(x)\,dx = \int p_w(\cdot)f(x)\,dx$ para una base $f(\cdot)$ que abarque el espacio de funciones. Estas igualdades de esperanzas describen las distribuciones por sus **momentos** (con $f$ polinomios de los datos $x$).

El salto conceptual de FID: en vez de usar polinomios de los píxeles crudos $x$, **se reemplaza $x$ por la capa de codificación (coding layer) de un modelo Inception**, obteniendo features relevantes para visión. Por razones prácticas se consideran solo los dos primeros momentos: **media y covarianza**. Y aquí entra el supuesto gaussiano: la gaussiana es la **distribución de máxima entropía** dados una media y una covarianza, así que se asume que las unidades de codificación siguen una gaussiana multidimensional. La diferencia entre dos gaussianas se mide con la **distancia de Fréchet** (también conocida como **distancia de Wasserstein-2**).

La fórmula —el corazón del paper para la posteridad— es:

$$\mathrm{FID} = d^2\big((m, C), (m_w, C_w)\big) = \lVert m - m_w \rVert_2^2 + \mathrm{Tr}\!\left(C + C_w - 2\,(C\,C_w)^{1/2}\right),$$

donde $(m_w, C_w)$ son media y covarianza de los features Inception de las imágenes **reales**, y $(m, C)$ los de las **generadas**. El primer término penaliza diferencias en la media (el "centro" de la distribución de features); el segundo, vía la traza y el término $(C C_w)^{1/2}$, penaliza diferencias en la **estructura de covarianza** (la "forma" y diversidad de la distribución). **Menor FID = mejor**; FID = 0 significa distribuciones de features idénticas, el ideal inalcanzable.

### 4.2. Por qué FID es mejor que el Inception Score

La ventaja decisiva sobre el IS: **FID sí compara contra los datos reales**. Calcula $(m_w, C_w)$ sobre las imágenes reales y los contrasta con $(m, C)$ de las generadas. Mientras el IS solo audita la "confianza de clasificación + variedad" de lo generado, FID mide cuán cerca está la **distribución completa de features** del modelo respecto de la real. Esto le da tres propiedades que la Clase 29 destaca:

- **Detecta mode collapse:** si el generador colapsa a pocos modos, la covarianza $C$ de sus features será mucho menor (menos diversidad) que $C_w$, y el término de traza dispara el FID. Un IS podría incluso quedar alto si esos pocos modos se clasifican con confianza.
- **Mejora con el realismo:** a más realismo y diversidad, los features generados se acercan a los reales y el FID baja.
- **Sensible a la degradación:** la Fig. A9–A12 del paper muestra imágenes de DCGAN/WGAN-GP en CelebA con FIDs de 500, 300, 133, 100, 45, 13 y 3 — la calidad visual mejora monótonamente a medida que baja el FID, lo que valida que el número se corresponde con el juicio humano.

### 4.3. Validación empírica de la métrica

El paper valida que FID se comporta como una buena distancia sometiendo imágenes de CelebA a **seis tipos de degradación** crecientes (Fig. 3) y verificando que FID **crece monótonamente** con cada una:

1. **Ruido gaussiano** ($(1-\alpha)X + \alpha N$).
2. **Desenfoque gaussiano** (convolución con kernel gaussiano).
3. **Rectángulos negros** implantados en posiciones aleatorias.
4. **Swirl** (efecto remolino sobre regiones de la imagen).
5. **Ruido sal y pimienta** (píxeles puestos a negro/blanco al azar).
6. **Contaminación con ImageNet** (un porcentaje $\alpha$ de las imágenes CelebA se reemplaza por imágenes de ImageNet).

En los seis casos FID sube de forma limpia y monótona; el IS (medido como "Inception Distance", $\mathrm{IND} = m - \mathrm{IS}$, para hacerlo comparable como distancia) fluctúa o se aplana. Esta es la evidencia directa de que FID es la mejor métrica.

## 5. Detalles de cómputo del FID

El procedimiento canónico —el que define el FID que se reporta hasta hoy— es:

- Se propagan **todas** las imágenes del dataset de entrenamiento por un **Inception-v3 preentrenado**, usando la **última capa de pooling** como capa de codificación (un vector de 2048 dimensiones). Se calculan $m_w$ y $C_w$.
- Para el modelo, se **generan 50.000 imágenes**, se propagan por Inception-v3 y se calculan $m$ y $C$.
- Se evalúa con la fórmula. En los experimentos se mide FID cada 1.000 actualizaciones de mini-batch para DCGAN y cada 5.000 iteraciones para WGAN-GP.

El uso de FID también sirve como criterio de **selección de modelo y de parada temprana**: se detiene el entrenamiento cuando el FID del mejor modelo deja de bajar. El paper observa además que para algunos modelos el **FID diverge o empieza a subir** a partir de cierto punto (Fig. 5) — una señal de inestabilidad o sobreajuste que el FID hace visible.

## 6. Experimentos de TTUR

El paper compara TTUR contra el entrenamiento de una sola escala de tiempo (que abrevia "orig"), repitiendo cada configuración 8 veces (imágenes) o 10 veces (lenguaje) y reportando media, mínimo y máximo del FID. Resultados de la Tabla 1:

- **Toy data (saddle point):** sobre $f(x,y) = (1+x^2)(100-y^2)$ con un punto silla en $(0,0)$, las actualizaciones de una sola escala con tasa grande divergen y con tasa pequeña convergen lento; **TTUR converge más rápido** y va directo al punto silla.
- **DCGAN sobre imágenes** (CelebA, CIFAR-10, SVHN, LSUN Bedrooms): "orig" es más rápido al inicio, pero **TTUR alcanza siempre un FID más bajo y es más estable**, con mucha menor varianza entre corridas. En CelebA y LSUN **todas** las corridas de una sola escala divergieron. Mejores FIDs: CelebA 12.5 (TTUR) vs 21.4 (orig); SVHN 12.5 vs 21.4; LSUN 57.5 vs 70.4.
- **WGAN-GP sobre imágenes** (CIFAR-10, LSUN): donde el código original entrena D cinco veces por paso de G, TTUR lo entrena **una sola vez** y usa una tasa más alta para D porque TTUR estabiliza. CIFAR-10 24.8 (TTUR) vs 29.3 (orig); LSUN **9.5 vs 20.5** — una mejora grande.
- **WGAN-GP sobre lenguaje** (One Billion Word Benchmark): como FID solo aplica a imágenes, se mide con la **divergencia de Jensen-Shannon (JSD)** normalizada sobre estadísticas de 4-gramas y 6-gramas. TTUR gana en ambas (4-gram 0.35 vs 0.38; 6-gram 0.74 vs 0.77); la mejora en 6-gramas indica que TTUR aprende a generar pseudo-palabras más sutiles, más parecidas a palabras reales.

## 7. Limitaciones

- **Sesgo de la red Inception.** FID hereda los sesgos del Inception-v3 preentrenado en ImageNet. Los features están optimizados para discriminar las 1000 clases de ImageNet (objetos cotidianos), de modo que la métrica es más sensible a lo que ImageNet considera relevante y puede ser ciega a atributos importantes en dominios alejados de ImageNet (imágenes médicas, satelitales, rostros con artefactos sutiles). Comparar FIDs entre datasets de naturaleza muy distinta no es significativo.
- **Sensibilidad al número de muestras.** La estimación de la covarianza $C$ desde un número finito de muestras está **sesgada**: con pocas muestras el FID se sobrestima. Por eso el paper fija un protocolo de 50.000 imágenes generadas; comparar FIDs computados con distinto $N$ es inválido. Esta dependencia del tamaño muestral es una limitación práctica reconocida del estimador.
- **El supuesto gaussiano.** FID asume que los features Inception siguen una gaussiana multidimensional —justificado por máxima entropía dados dos momentos, pero las distribuciones reales de activaciones no son gaussianas—. FID solo captura los dos primeros momentos; diferencias de orden superior entre distribuciones le son invisibles.
- **No aplica fuera de imágenes.** FID requiere una red de visión; para texto el propio paper recurre a JSD. La métrica no es agnóstica al dominio.
- **TTUR requiere búsqueda de tasas.** Aunque TTUR garantiza convergencia, las tasas prácticas deben optimizarse por experimento, equilibrando "lo bastante pequeñas para converger" con "lo bastante grandes para aprender rápido".

## 8. Impacto

La asimetría entre las dos contribuciones es notable. La prueba de convergencia de TTUR es matemáticamente la pieza central del paper, y TTUR (tasas distintas para D y G) sigue siendo práctica común. Pero el **legado dominante es FID**: se convirtió en **la métrica estándar de facto para evaluar modelos generativos de imágenes**, primero para GANs (StyleGAN, BigGAN, ProGAN) y luego para toda la familia de modelos de difusión (DDPM, Latent Diffusion/Stable Diffusion, modelos basados en score). Prácticamente ninguna tabla de resultados de generación de imágenes posterior a 2017 omite el FID. Variantes posteriores —**FID con menos sesgo de muestreo, KID (Kernel Inception Distance), CLIP-FID** que reemplaza Inception por un encoder CLIP— nacen todas como respuesta a las limitaciones enumeradas arriba, lo que confirma cuán central se volvió la idea original: comparar distribuciones de features de una red auxiliar mediante una distancia entre gaussianas.

## 9. Conexión con la Clase 29 (Modelos Generativos en Visión)

La Clase 29 dedica una sección explícita a "**Cómo evalúo la calidad: Fréchet Inception Distance (FID)**", y este paper es su fuente primaria. Mapeo directo:

- **"Compara distribuciones con una red auxiliar."** Es exactamente §4.1: las slides explican que FID pasa imágenes reales y generadas por una red preentrenada (Inception) y compara las distribuciones de sus features modelándolas como gaussianas. El "red auxiliar" de la clase es el Inception-v3 cuya última capa de pooling provee el espacio de 2048 features.
- **"Menor = mejor, FID = 0 ideal."** Es la naturaleza de distancia de la fórmula: FID = 0 cuando $(m, C) = (m_w, C_w)$, es decir distribuciones de features idénticas. La clase enseña esta semántica directamente desde la ecuación $\lVert m - m_w \rVert^2 + \mathrm{Tr}(\cdots)$.
- **"Detecta overfitting / diversidad / realismo."** Conecta con §4.2: el término de covarianza detecta mode collapse (poca diversidad → $C \ll C_w$ → FID alto); la divergencia del FID durante el entrenamiento (Fig. 5 del paper) es justamente la señal de overfitting que la clase menciona; y la batería de degradaciones (Fig. 3) valida que FID baja con el realismo.
- **Comparaciones FID en ImageNet / FFHQ.** Las slides muestran tablas de FID sobre ImageNet y **FFHQ** (Flickr-Faces-HQ, el dataset de rostros de StyleGAN). FFHQ es posterior a este paper, pero la métrica con la que se reportan esos números es precisamente la definida aquí — la clase usa FID como vara común para ordenar GANs y modelos de difusión.

Dentro del recorrido de la Clase 29 —que va de VAE y GAN, pasando por DCGAN, StyleGAN y VQ-VAE, hasta los modelos de difusión (DDPM, score-based, Latent Diffusion)—, este paper ocupa el rol de **infraestructura de evaluación transversal**: es el instrumento de medición que permite afirmar que un modelo generativo es mejor que otro. Sin FID, la narrativa de progreso de la clase (cada arquitectura nueva baja el FID respecto de la anterior) no tendría un eje cuantitativo. TTUR, por su parte, aporta el otro mensaje de fondo de la clase sobre GANs: que su entrenamiento es un juego inestable, y que estabilizarlo —vía tasas separadas, Adam, gradiente penalizado de WGAN-GP— es la mitad del trabajo de hacerlos funcionar.

Material relacionado: fundamento [Modelos Generativos](/fundamentos/modelos-generativos), clase [Clase 29 — Modelos Generativos en Visión](/clases/clase-29).
