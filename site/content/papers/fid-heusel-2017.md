---
title: "FID / TTUR: GANs y la Fréchet Inception Distance (2017)"
weight: 333
math: true
---

{{< paper-card
    title="GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"
    authors="Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter"
    year="2017"
    venue="NeurIPS 2017"
    pdf="/papers/fid-heusel-2017.pdf"
    arxiv="1706.08500" >}}
Paper del LIT AI Lab de la Johannes Kepler University Linz (con Sepp Hochreiter, coinventor de las LSTM) que empaqueta **dos contribuciones independientes**. La primera, que da título al trabajo, es **TTUR** (*two time-scale update rule*): usar tasas de aprendizaje distintas para el generador y el discriminador de un GAN, con una prueba de que el entrenamiento **converge a un equilibrio de Nash local**. La segunda, por la que se recuerda casi universalmente, es la **FID** (*Fréchet Inception Distance*): una métrica que compara las distribuciones de features de imágenes reales y generadas modelándolas como gaussianas y midiendo la distancia de Fréchet entre ellas. La FID se convirtió en **la métrica estándar de facto** para evaluar GANs y modelos de difusión. Menor = mejor.
{{< /paper-card >}}

---

## Contexto: el problema de evaluar lo que genera un modelo

El objetivo del aprendizaje generativo es producir datos que **coincidan con la distribución observada**. En principio cualquier distancia entre la distribución real y la del modelo serviría como medida de calidad, pero en la práctica esto es difícil. En 2017 las opciones eran insatisfactorias:

- **Verosimilitud (likelihood):** depende fuertemente de los supuestos de ruido y puede estar dominada por muestras individuales atípicas. Peor aún, para un GAN ni siquiera es computable: el generador es un mapeo determinista de ruido a imagen, sin densidad explícita tratable. Usar verosimilitud para evaluar modelos cuya razón de ser es evitarla resulta circular.
- **Inception Score (IS)** de Salimans et al. (2016): la mejor métrica práctica de la época. Pasa las muestras generadas por una red **Inception preentrenada en ImageNet** y combina dos intuiciones: baja entropía de etiquetas por imagen (el clasificador está seguro de qué objeto ve) y alta entropía marginal de etiquetas (el modelo genera variedad de clases), en la fórmula $\exp\!\big(\mathbb{E}_X\,\mathrm{KL}(p(y\mid X)\,\|\,p(y))\big)$.

El **defecto central** del IS, que motiva todo el paper: "*no usa las estadísticas de muestras del mundo real para compararlas con las sintéticas*". El IS solo audita lo generado a través del clasificador; **nunca compara contra los datos reales**. Un modelo podría generar una imagen perfectamente clasificable por cada clase de ImageNet (IS alto) sin que esas imágenes se parezcan en nada, estadísticamente, al dataset objetivo. Premia objetividad y variedad de clases, no fidelidad a la distribución real.

## Contribución 1 — TTUR: convergencia a un equilibrio de Nash local

Entrenar un GAN es un **juego** entre el generador $G$ (que fabrica datos desde ruido) y el discriminador $D$ (que separa real de sintético). Su solución es un **equilibrio de Nash**, no el mínimo de una sola función de pérdida. El descenso de gradiente persigue mínimos; aplicado a un juego de suma no nula puede oscilar indefinidamente sin converger. Lo máximo alcanzable son **equilibrios de Nash locales**: puntos donde ni $G$ ni $D$ pueden bajar su pérdida moviéndose unilateralmente.

La idea de **TTUR** es usar **tasas de aprendizaje separadas** $b(n)$ para el discriminador y $a(n)$ para el generador:

$$w_{n+1} = w_n + b(n)\,\big(g(\theta_n, w_n) + M_n^{(w)}\big), \qquad \theta_{n+1} = \theta_n + a(n)\,\big(h(\theta_n, w_n) + M_n^{(\theta)}\big),$$

donde $M$ es el ruido del gradiente estocástico de mini-batches. La premisa: el discriminador converge a un mínimo cuando el generador está fijo; si el generador cambia despacio (escala de tiempo **lenta**) mientras el discriminador es rápido, las perturbaciones que sufre $D$ son pequeñas y converge igual. Esto formaliza una intuición de la época: las implementaciones que funcionaban entrenaban $D$ más agresivamente (WGAN hacía 5 pasos de $D$ por cada paso de $G$). TTUR logra el mismo efecto con tasas distintas y **una sola actualización de cada red por iteración**.

La prueba se apoya en la **teoría de aproximación estocástica de dos escalas de tiempo** (Borkar, 1997). Bajo supuestos suaves —gradientes Lipschitz (lo cumplen activaciones suaves como ELU, no las ReLU), condiciones tipo Robbins-Monro sobre las tasas con la clave $a(n) = o(b(n))$ (la lenta asintóticamente despreciable frente a la rápida), y atractores localmente estables para cada red— el **Teorema 1** garantiza que las actualizaciones convergen casi seguramente a $(\theta^*, \lambda(\theta^*))$, un **equilibrio de Nash local estacionario**.

Un segundo resultado más sutil (**Teorema 2**): el optimizador Adam se comporta como una **bola pesada con fricción** (*Heavy Ball with Friction*), siguiendo la ecuación $\ddot{\theta}_t + a(t)\,\dot{\theta}_t + \nabla f(\theta_t) = 0$. El promediado de gradientes pasados actúa como **inercia**: como una bola con masa, sobrepasa los mínimos locales pequeños —los que corresponden a **mode collapse**— y se asienta en mínimos planos que generalizan mejor. Esto explica por qué Adam es bueno evitando el colapso de modos.

## Contribución 2 — FID: la Fréchet Inception Distance

El salto conceptual de la FID parte de un principio de teoría de la medida: dos distribuciones quedan descritas por sus **momentos**. En vez de usar polinomios de los píxeles crudos, se reemplaza la imagen $x$ por la **capa de codificación de una red Inception**, obteniendo features relevantes para visión. Por practicidad se consideran solo los **dos primeros momentos** (media y covarianza), y se asume que esos features siguen una **gaussiana multidimensional** —justificado porque la gaussiana es la distribución de máxima entropía dados media y covarianza—. La diferencia entre dos gaussianas se mide con la **distancia de Fréchet** (equivalente a la **distancia de Wasserstein-2**):

$$\mathrm{FID} = \lVert m - m_w \rVert_2^2 + \mathrm{Tr}\!\left(C + C_w - 2\,(C\,C_w)^{1/2}\right),$$

donde $(m_w, C_w)$ son la media y covarianza de los features Inception de las imágenes **reales** y $(m, C)$ los de las **generadas**. El primer término penaliza diferencias en la media (el "centro" de la distribución de features); el segundo, vía la traza, penaliza diferencias en la **estructura de covarianza** (la "forma" y diversidad). **Menor FID = mejor**; FID = 0 significa distribuciones idénticas, el ideal inalcanzable.

### Por qué supera al Inception Score

La ventaja decisiva: **FID sí compara contra los datos reales**. Mientras el IS solo audita la confianza de clasificación más la variedad de lo generado, FID mide cuán cerca está la **distribución completa de features** del modelo respecto de la real. Esto le da tres propiedades:

- **Detecta mode collapse.** Si el generador colapsa a pocos modos, la covarianza $C$ de sus features será mucho menor (menos diversidad) que $C_w$, y el término de traza dispara la FID. El IS podría incluso quedar alto si esos pocos modos se clasifican con confianza.
- **Mejora con el realismo.** A más realismo y diversidad, los features generados se acercan a los reales y la FID baja.
- **Sensible a la degradación.** El paper somete imágenes de CelebA a seis tipos de degradación crecientes (ruido gaussiano, desenfoque, rectángulos negros, swirl, sal y pimienta, contaminación con ImageNet) y verifica que la FID **crece monótonamente** en los seis casos. El IS, en cambio, "fluctúa, se mantiene plano o, en el peor caso, decrece" —exactamente lo contrario de lo que debe hacer una buena medida.

### Cómputo canónico

El procedimiento que define la FID que se reporta hasta hoy: se propagan **todas** las imágenes reales por un **Inception-v3 preentrenado**, usando la última capa de pooling (un vector de **2.048 dimensiones**) para calcular $m_w$ y $C_w$; se **generan 50.000 imágenes** del modelo y se calculan $m$ y $C$; se evalúa la fórmula. La FID sirve también como criterio de **selección de modelo y parada temprana**: cuando deja de bajar (o empieza a divergir), señala inestabilidad o sobreajuste.

## Resultados de TTUR

Repitiendo cada configuración 8-10 veces y reportando media/mínimo/máximo de la FID, TTUR alcanza siempre una FID más baja y es **más estable** que el entrenamiento de una sola escala:

- **DCGAN** (CelebA, CIFAR-10, SVHN, LSUN): TTUR mejora la FID y reduce drásticamente la varianza entre corridas; en CelebA y LSUN **todas** las corridas de una sola escala divergieron.
- **WGAN-GP** (CIFAR-10, LSUN): donde el código original entrena $D$ cinco veces por paso de $G$, TTUR lo hace **una sola vez** con tasa más alta para $D$ y mejora mucho (LSUN 9.5 vs 20.5).
- **Lenguaje** (One Billion Word): como FID solo aplica a imágenes, aquí se mide con divergencia de Jensen-Shannon sobre 4- y 6-gramas; TTUR gana en ambas.

## Limitaciones

- **Sesgo de la red Inception.** La FID hereda los sesgos del Inception-v3 preentrenado en ImageNet: sus features están optimizados para las 1000 clases de objetos cotidianos, de modo que puede ser ciega a atributos importantes en dominios alejados (imágenes médicas, satelitales). Comparar FIDs entre datasets de naturaleza muy distinta no es significativo.
- **Sensibilidad al número de muestras.** La estimación de la covarianza $C$ desde un número finito de muestras está **sesgada**: con pocas muestras la FID se sobrestima. Por eso el protocolo fija 50.000 imágenes generadas; comparar FIDs computados con distinto $N$ es inválido.
- **El supuesto gaussiano.** La FID solo captura los dos primeros momentos; diferencias de orden superior entre distribuciones le son invisibles.
- **No aplica fuera de imágenes.** Requiere una red de visión; para texto el propio paper recurre a JSD. La métrica no es agnóstica al dominio.
- **TTUR requiere búsqueda de tasas.** Aunque garantiza convergencia, las tasas prácticas deben optimizarse por experimento.

## Impacto

La asimetría entre las dos contribuciones es notable. TTUR (tasas distintas para $D$ y $G$) sigue siendo práctica común, pero el **legado dominante es la FID**: se volvió la métrica estándar para evaluar modelos generativos de imágenes, primero para GANs ([StyleGAN](/papers/stylegan-karras-2019), BigGAN, ProGAN) y luego para toda la familia de modelos de difusión (DDPM, Latent Diffusion / Stable Diffusion). Prácticamente ninguna tabla de generación de imágenes posterior a 2017 omite la FID. Variantes posteriores —**KID** (Kernel Inception Distance) y **CLIP-FID** (que reemplaza Inception por un encoder CLIP)— nacen como respuesta a las limitaciones de arriba, lo que confirma cuán central se volvió la idea original: comparar distribuciones de features de una red auxiliar mediante una distancia entre gaussianas.

## Por qué importa para la Clase 29

La [Clase 29](/clases/clase-29) ("Modelos Generativos en Visión") dedica una sección a **"¿cómo evalúo la calidad de lo que genera mi modelo?"**, y este paper es su fuente primaria. Mapeo directo:

- **"Compara distribuciones con una red auxiliar."** Es la derivación de la FID: pasa imágenes reales y generadas por Inception-v3 y compara las distribuciones de features (vector de 2.048-d de la última capa de pooling) modelándolas como gaussianas.
- **"Menor = mejor, FID = 0 ideal."** Es la naturaleza de distancia de la fórmula $\lVert m - m_w \rVert^2 + \mathrm{Tr}(\cdots)$: vale 0 cuando las distribuciones de features son idénticas.
- **"Detecta overfitting / diversidad / realismo."** El término de covarianza detecta mode collapse; la divergencia de la FID durante el entrenamiento señala overfitting; la batería de degradaciones valida que baja con el realismo.

Dentro del recorrido de la clase —de VAE y GAN, pasando por DCGAN, StyleGAN y VQ-VAE, hasta los modelos de difusión— este paper es la **infraestructura de evaluación transversal**: el instrumento que permite afirmar cuantitativamente que un modelo generativo es mejor que otro. Sin FID, la narrativa de progreso (cada arquitectura nueva baja la FID respecto de la anterior) no tendría eje numérico. TTUR aporta el otro mensaje de fondo sobre GANs: que su entrenamiento es un juego inestable, y que estabilizarlo es la mitad del trabajo de hacerlos funcionar.

## Material relacionado

- Fundamento: [Modelos Generativos](/fundamentos/modelos-generativos)
- Clase: [Clase 29 — Modelos Generativos en Visión](/clases/clase-29)
- Papers: [GAN — Goodfellow et al. (2014)](/papers/gan-goodfellow-2014), [StyleGAN — Karras et al. (2019)](/papers/stylegan-karras-2019)
