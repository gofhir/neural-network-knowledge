---
title: "GAN: Generative Adversarial Nets (2014)"
weight: 329
math: true
---

{{< paper-card
    title="Generative Adversarial Nets"
    authors="Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio"
    year="2014"
    venue="NeurIPS 2014"
    pdf="/papers/gan-goodfellow-2014.pdf"
    arxiv="1406.2661" >}}
El paper que fundó las **Redes Generativas Adversarias (GANs)**, uno de los trabajos más influyentes del aprendizaje profundo generativo. Su tesis es de una elegancia desarmante: en vez de estimar explícitamente una densidad de probabilidad —tarea que tropieza una y otra vez con cómputos intratables— se aprende a *generar muestras* enfrentando dos redes en un juego. Un **generador G** intenta capturar la distribución de los datos y un **discriminador D** estima la probabilidad de que una muestra sea real en vez de venir de G; G se entrena para *maximizar la probabilidad de que D se equivoque*. El paper formaliza esto como un **juego minimax de dos jugadores** y demuestra que existe una solución única en la que G recupera exactamente la distribución de los datos y D vale 1/2 en todas partes. Todo se entrena con **retropropagación pura**, sin cadenas de Markov ni inferencia aproximada.
{{< /paper-card >}}

---

## Contexto

Hacia 2014 el deep learning cosechaba sus mayores éxitos en **modelos discriminativos** —mapear una imagen o un audio a una etiqueta— apoyándose en retropropagación, dropout y unidades lineales por tramos (ReLU, maxout). Los **modelos generativos profundos**, en cambio, tenían mucho menos impacto, y el paper diagnostica por qué: la dificultad de aproximar los muchos *cómputos probabilísticos intratables* que surgen en la estimación por máxima verosimilitud.

Ese problema de la **verosimilitud intratable** es el villano de fondo. Los modelos clásicos de la época estaban atascados, cada uno por su versión del mismo obstáculo: las máquinas de Boltzmann (RBM, DBM) requieren una **función de partición** —y su gradiente— intratable, estimable solo por MCMC con problemas de *mixing*; las redes de creencia profunda (DBN) heredan las dificultades de los modelos dirigidos y no dirigidos a la vez; criterios como *score matching* y *noise-contrastive estimation* exigen una densidad especificada analíticamente salvo una constante. La línea que traza el paper es nítida: **el framework adversarial no requiere cadenas de Markov para muestrear**, ni durante el entrenamiento ni durante la generación, lo que además le permite aprovechar bien las unidades lineales por tramos. Sus primos directos son los [VAE](/papers/vae-kingma-2013) de Kingma y Welling, la otra gran familia generativa del mismo año.

La analogía canónica del paper son los **falsificadores contra la policía**: el generador es un equipo que produce moneda falsa e intenta usarla sin ser detectado, mientras el discriminador es la policía que intenta cazar la falsificación. La competencia empuja a ambos a mejorar hasta que las falsificaciones son indistinguibles de lo genuino.

## El juego minimax

Para aprender la distribución $p_g$ del generador sobre los datos $x$, se define un ruido de entrada con prior $p_z(z)$ y un mapeo diferenciable al espacio de datos $G(z; \theta_g)$ representado por un MLP. Un segundo MLP, $D(x; \theta_d)$, emite un escalar: $D(x)$ es la probabilidad de que $x$ sea real y no provenga de $p_g$. Se entrena $D$ para **maximizar** la probabilidad de asignar la etiqueta correcta tanto a los datos reales como a las muestras de $G$, y simultáneamente $G$ para que $D$ se equivoque. Formalmente, $D$ y $G$ juegan el siguiente **juego minimax** con función de valor $V(G, D)$:

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]. \quad (1)$$

Conviene leerla con calma porque es el corazón del paper:

- El **primer término** $\mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)]$ premia a $D$ por dar probabilidad alta (cercana a 1) a los datos reales. $D$ quiere maximizarlo.
- El **segundo término** $\mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$ premia a $D$ por dar probabilidad baja (cercana a 0) a las falsificaciones $G(z)$. $D$ también lo maximiza, pero $G$ quiere **minimizarlo**: busca que $D(G(z))$ sea alto, que $D$ se *crea* la falsificación.
- El operador $\min_G \max_D$ codifica el conflicto: $D$ trepa la función de valor, $G$ la baja, y el equilibrio del juego es lo que produce el aprendizaje.

## El óptimo de D y la divergencia de Jensen-Shannon

El paper demuestra (Proposición 1) que, para un $G$ fijo, el **discriminador óptimo** es

$$D^*_G(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}. \quad (2)$$

La prueba es directa: el integrando $a\log y + b\log(1-y)$ alcanza su máximo en $[0,1]$ en $\frac{a}{a+b}$; sustituyendo $a = p_{\text{data}}(x)$ y $b = p_g(x)$ sale (2). Entrenar $D$ equivale, así, a estimar la probabilidad condicional de que $x$ sea real.

Sustituyendo $D^*_G$ de vuelta en el juego se obtiene un criterio virtual $C(G) = \max_D V(G,D)$. El **Teorema 1** establece que su mínimo global se alcanza *si y solo si* $p_g = p_{\text{data}}$, y ahí vale $-\log 4$. La belleza está en el álgebra que lo conecta con una divergencia conocida: restando $-\log 4$ se reescribe como una suma de dos divergencias de Kullback-Leibler respecto al promedio $\frac{p_{\text{data}} + p_g}{2}$, que es exactamente —por definición— la **divergencia de Jensen-Shannon** (JSD):

$$C(G) = -\log(4) + 2 \cdot \text{JSD}(p_{\text{data}} \,\|\, p_g). \quad (6)$$

Como la JSD es no negativa y solo cero cuando las distribuciones son iguales, queda demostrado que $C^* = -\log 4$ es el mínimo global y que la única solución es $p_g = p_{\text{data}}$: **el generador replicando perfectamente el proceso generador de los datos**. Este es el resultado conceptual central del paper: *entrenar una GAN equivale, en el límite ideal, a minimizar la divergencia de Jensen-Shannon entre el modelo y los datos.* Que la JSD se sature cuando los soportes no se solapan es, además, la semilla de la inestabilidad que la literatura posterior (Wasserstein GAN) vendría a atacar.

## Entrenamiento alternado y el truco de $-\log D(G(z))$

El óptimo anterior vive en el límite no paramétrico (capacidad infinita). En la práctica el juego se implementa de forma **iterativa y numérica**, con dos decisiones de ingeniería que definen cómo se entrena una GAN de verdad.

**Entrenamiento alternado (Algoritmo 1).** Optimizar $D$ hasta el óptimo en cada bucle interno sería prohibitivo y llevaría a sobreajuste. En vez de eso se **alternan** $k$ pasos de $D$ con *un* paso de $G$, manteniendo $D$ cerca de su óptimo mientras $G$ cambie despacio. En los experimentos se usó $k = 1$. Por iteración: (1) muestrear un minibatch de ruidos $z$ y otro de datos reales $x$, y **ascender** el gradiente del discriminador $\nabla_{\theta_d} \frac{1}{m}\sum_i [\log D(x^{(i)}) + \log(1 - D(G(z^{(i)})))]$; (2) muestrear ruidos nuevos y **descender** el gradiente del generador $\nabla_{\theta_g} \frac{1}{m}\sum_i \log(1 - D(G(z^{(i)})))$. Nótese que $D$ *asciende* y $G$ *desciende* sobre la misma función de valor: el signo opuesto es el juego minimax hecho código. Las actualizaciones usaron *momentum*.

**El truco $-\log D(G(z))$ (non-saturating loss).** Hay un problema práctico con la ecuación (1): temprano en el entrenamiento, cuando $G$ es pobre, $D$ rechaza sus muestras con altísima confianza y $\log(1 - D(G(z)))$ **se satura** —su gradiente respecto a $\theta_g$ se vuelve casi nulo justo cuando $G$ más necesita aprender. La solución: en vez de entrenar $G$ para minimizar $\log(1 - D(G(z)))$, entrenarlo para **maximizar $\log D(G(z))$**. Esto da el *mismo punto fijo* en la dinámica, pero **gradientes mucho más fuertes al inicio**. Este truco es lo que hace que las GANs entrenen en absoluto; casi todas las implementaciones reales lo usan, aunque la teoría se escriba con la forma saturante.

El paper también prueba la convergencia (Proposición 2): si $G$ y $D$ tienen capacidad suficiente y $D$ alcanza su óptimo dado $G$ en cada paso, $p_g$ converge a $p_{\text{data}}$, porque $V$ es convexa en $p_g$ con óptimo global único. La salvedad honesta: en la práctica se optimiza $\theta_g$ —no $p_g$ directamente— y el MLP introduce múltiples puntos críticos, así que las garantías no aplican estrictamente; pero "el excelente desempeño de los MLPs sugiere que son un modelo razonable a pesar de la falta de garantías teóricas".

## Experimentos

Se entrenaron redes adversarias en **MNIST** (dígitos), la **Toronto Face Database (TFD)** y **CIFAR-10**. El generador mezcló activaciones ReLU y sigmoides; el discriminador usó **maxout** y dropout. Como una GAN puede *muestrear* pero no *evaluar* la densidad, se estimó la log-verosimilitud del test ajustando una **ventana de Parzen gaussiana** a las muestras generadas: en MNIST las adversarias obtuvieron $225 \pm 2$ (frente a DBN $138$, Deep GSN $214$) y en TFD $2057 \pm 26$ (competitivo con el mejor). El propio paper reconoce que este estimador tiene **varianza alta y funciona mal en alta dimensión**, pero era el mejor disponible.

Dos detalles cualitativos honestos: las muestras mostradas son extracciones aleatorias *justas* (no cherry-picked) y se acompañan del ejemplo de entrenamiento más cercano para demostrar que el modelo **no memorizó**; y a diferencia de los métodos basados en MCMC, son muestras *reales* de la distribución del modelo, descorrelacionadas porque no dependen del mixing de una cadena. Interpolar linealmente en el espacio $z$ produce transiciones suaves, evidencia temprana de que el **latente aprendido tiene estructura semántica**.

## Limitaciones reconocidas

- **No hay representación explícita de $p_g(x)$.** No se puede preguntar "¿qué probabilidad le asigna el modelo a esta imagen?", lo que obligó al rodeo de la ventana de Parzen.
- **$D$ debe estar bien sincronizado con $G$.** Si $G$ se entrena demasiado sin actualizar $D$, colapsa muchos valores de $z$ al mismo $x$ —el **"escenario Helvetica"**, que la literatura posterior bautizaría como **mode collapse**: la GAN genera unas pocas muestras convincentes pero no cubre toda la variedad de los datos.
- A esto la comunidad sumaría pronto la **inestabilidad del entrenamiento**: el equilibrio minimax es delicado, los gradientes oscilan y no hay curva de pérdida monótona que indique progreso.

Las **ventajas** compensan: nunca se necesitan cadenas de Markov, solo backprop; no hay inferencia durante el aprendizaje; y, a diferencia de MCMC (que requiere distribuciones borrosas para mezclar entre modos), las adversarias pueden representar distribuciones **muy nítidas**, razón de fondo por la que las GANs producirían imágenes más nítidas que los VAE.

## Impacto

Las GANs **revolucionaron la generación de imágenes**. La idea desató miles de variantes —[DCGAN](/papers/dcgan-radford-2015) (convolucional, estabilizó el entrenamiento), Conditional GAN, CycleGAN, Pix2Pix, [StyleGAN](/papers/stylegan-karras-2019) (rostros foto-realistas con control de estilo), Progressive GAN, BigGAN, Wasserstein GAN (que reemplazó la JSD por la distancia de Wasserstein para atacar justo la inestabilidad y el mode collapse diagnosticados aquí)— y dominaron la síntesis de imágenes de alta fidelidad durante buena parte de la segunda mitad de la década de 2010. Yann LeCun describió las GANs como "la idea más interesante en machine learning de los últimos diez años".

La derrota relativa frente a los **modelos de difusión** (a partir de ~2021) no las jubiló: el esquema adversarial sobrevive *dentro* de los sistemas modernos. En Stable Diffusion, la difusión opera en un espacio latente comprimido por un autoencoder, y ese autoencoder se entrena con una **pérdida adversarial** (un discriminador estilo PatchGAN) precisamente para que las reconstrucciones tengan la nitidez que el método de Parzen ya anticipaba como fortaleza de lo adversarial. El juego de falsificadores contra policía de 2014 sigue afilando las imágenes de los generativos de hoy.

## Por qué importa para la Clase 29

La [Clase 29](/clases/clase-29) organiza los modelos generativos en familias, y las GANs son la **segunda familia** tras los autoencoders/VAE. El mapeo es casi uno a uno:

- **El juego adversarial G vs D.** La intuición que la clase verbaliza —el discriminador dice "voy a detectar tu falsedad", el generador responde "te voy a engañar"— es exactamente la dinámica falsificadores-policía y el operador $\min_G \max_D$ de la ecuación (1).
- **La pérdida con $z$, $x$, $G$, $D$.** Es $V(D,G) = \mathbb{E}_x[\log D(x)] + \mathbb{E}_z[\log(1 - D(G(z)))]$, y para $G$ se entrena con el truco $-\log D(G(z))$.
- **La tabla comparativa.** Clasifica las GANs como de **muestreo rápido** (una pasada forward, sin cadenas de Markov), **calidad alta** (distribuciones nítidas) pero **cobertura baja** —el *mode collapse* / "escenario Helvetica". Los tres veredictos salen directamente de este paper.
- **GAN dentro de Stable Diffusion.** El discriminador adversarial mantiene nítidas las reconstrucciones del autoencoder sobre el que opera la difusión: el puente entre la segunda familia (GANs) y la dominante hoy (difusión).

Para el panorama completo —VAE, flujos normalizadores, autorregresivos y difusión— ver el fundamento transversal [Modelos Generativos](/fundamentos/modelos-generativos) y el hub de la [Clase 29](/clases/clase-29).
