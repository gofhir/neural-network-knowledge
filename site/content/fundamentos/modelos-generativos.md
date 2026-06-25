---
title: "Modelos Generativos"
weight: 105
math: true
---

Un **modelo generativo** es un modelo que aprende a producir datos nuevos —imágenes, audio, texto, moléculas— que se parecen a los de un conjunto de entrenamiento. La diferencia con casi todo lo que el curso vio antes es de raíz: un clasificador o un detector aprenden a *responder preguntas sobre* los datos, mientras que un modelo generativo aprende a *fabricar* datos. Esa capacidad de síntesis —que hace cinco años era una curiosidad académica que producía rostros borrosos de 64×64 píxeles— es hoy la maquinaria detrás de Stable Diffusion, DALL·E, Midjourney y toda la generación de imágenes por *prompt* de texto. Este fundamento recorre las familias que hicieron posible ese salto: autoencoders variacionales (VAE), redes generativas adversarias (GAN), modelos de difusión y sus combinaciones, con el hilo conductor del **trilema generativo** —calidad, velocidad, cobertura— que explica por qué ninguna familia gana en todo y por qué la difusión latente terminó dominando. Es el fundamento núcleo de la [Clase 29](/clases/clase-29).

---

## 1. Qué es un modelo generativo: aprender $p(x)$

Formalmente, un modelo generativo aprende —explícita o implícitamente— la **distribución de probabilidad de los datos** $p(x)$, donde $x$ es una muestra del dominio (un vector de píxeles, una forma de onda, una secuencia de tokens). Una vez aprendida esa distribución, el modelo puede hacer tres cosas que un clasificador no puede:

- **Muestrear** ($x \sim p(x)$): generar ejemplos nuevos que nunca estuvieron en el dataset pero que pertenecen verosímilmente a la misma distribución.
- **Reconstruir**: comprimir un dato a una representación compacta y recuperarlo (el rol de los autoencoders).
- **Interpolar**: moverse suavemente entre dos puntos del espacio aprendido, generando una transición coherente (por ejemplo, una cara que envejece gradualmente).

El contraste con los **modelos discriminativos** es la clave conceptual. Un modelo discriminativo aprende la distribución condicional $p(y \mid x)$ —dada una imagen $x$, ¿qué etiqueta $y$ le corresponde?—. Le basta con trazar fronteras entre clases; no necesita entender cómo se ve un gato por dentro, solo qué lo separa de un perro. Un modelo generativo, en cambio, debe aprender la estructura completa de los datos: para muestrear un gato plausible tiene que haber capturado texturas, poses, iluminación y proporciones. Por eso generar es, en general, **mucho más difícil** que clasificar.

{{< concept-alert type="clave" >}}
**Discriminativo vs. generativo en una frase.** El discriminativo aprende $p(y \mid x)$ y responde "¿qué es esto?"; el generativo aprende $p(x)$ (o $p(x \mid y)$) y responde "¿cómo se ve algo así?". Clasificar requiere distinguir; generar requiere comprender la distribución entera.
{{< /concept-alert >}}

Una distinción técnica organiza toda la taxonomía que sigue: **densidad explícita vs. implícita**. Algunos modelos definen y evalúan $p(x)$ de forma explícita —exacta (modelos autoregresivos, *normalizing flows*) o aproximada vía una cota (VAE, difusión)—, lo que permite preguntar "¿qué probabilidad le asigna el modelo a esta imagen?". Otros, los de **densidad implícita**, nunca representan $p(x)$ directamente: solo aprenden a *muestrear* de ella. Las GAN son el caso paradigmático: el generador es un mapeo determinista de ruido a imagen, sin densidad tratable —no se puede evaluar la verosimilitud de una muestra, solo producirla—. Esta diferencia no es académica: determina cómo se entrena cada familia, cómo se evalúa y qué patologías sufre.

El obstáculo histórico que todas estas familias debieron sortear es la **verosimilitud intratable**. La verosimilitud marginal $p_\theta(x) = \int p_\theta(z)\,p_\theta(x \mid z)\,dz$ —la integral sobre todas las variables latentes posibles— no se puede evaluar ni diferenciar apenas la verosimilitud condicional $p_\theta(x \mid z)$ es moderadamente complicada (por ejemplo, una red neuronal). Cada familia inventa un rodeo distinto para esta integral: el VAE la acota con el ELBO, la GAN la elude por completo aprendiendo a muestrear sin densidad, la difusión la descompone en una cadena de pasos pequeños tratables, y los autoregresivos la factorizan en condicionales univariados. Entender que todas atacan *el mismo* problema —cómo modelar una distribución de altísima dimensión sin poder calcular su normalización— es la clave para ver las familias como respuestas alternativas a una sola pregunta.

---

## 2. La taxonomía de familias y el trilema generativo

Las familias generativas profundas se reparten en cinco grandes grupos. Tres son protagonistas de la Clase 29 —VAE, GAN, difusión— y dos aparecen como contexto: los **autoregresivos** (PixelCNN, los Transformers que generan token a token; modelan $p(x) = \prod_i p(x_i \mid x_{<i})$ con verosimilitud exacta pero muestreo secuencial lento) y los **normalizing flows** (transformaciones invertibles que mapean ruido a datos con verosimilitud exacta, pero arquitectónicamente restrictivos).

El criterio de comparación más útil es el **trilema del aprendizaje generativo**, articulado por [Xiao et al. (2021)](/papers/diffusion-gan-xiao-2021): un buen modelo generativo querría satisfacer **tres** requisitos a la vez, pero las familias clásicas casi siempre sacrifican uno por los otros dos.

1. **Calidad** de muestras (alta fidelidad, nitidez, realismo).
2. **Velocidad** de muestreo (cuántas evaluaciones de red por imagen).
3. **Cobertura** de la distribución (diversidad; modelar *todos* los modos de los datos, no solo unos pocos).

| Familia | Velocidad de muestreo | Calidad | Cobertura (diversidad) | Densidad | Vértice sacrificado |
|---|---|---|---|---|---|
| [**VAE**](/papers/vae-kingma-2013) | Rápida (1 pasada) | Media (borrosa) | Alta | Explícita (ELBO) | calidad |
| [**GAN**](/papers/gan-goodfellow-2014) | Rápida (1 pasada) | Alta | Baja (*mode collapse*) | Implícita | cobertura |
| **Autoregresivo** | Lenta (secuencial) | Alta | Alta | Explícita exacta | velocidad |
| **Normalizing flow** | Rápida | Media | Alta | Explícita exacta | calidad |
| [**Difusión**](/fundamentos/modelos-de-difusion) | **Lenta** (cientos/miles de pasos) | **Alta** | **Alta** | Explícita (cota) | velocidad |

La lectura del trilema es que la **difusión** logró lo que parecía imposible —alta calidad *y* alta cobertura simultáneas— a cambio de un único defecto, la lentitud de muestreo; y que casi toda la investigación posterior a 2020 se dedicó a atacar precisamente ese defecto. Las GAN, en el otro extremo, son rapidísimas y nítidas pero pagan con *mode collapse* (cobertura baja). Los VAE cubren bien la distribución pero generan muestras borrosas. Las secciones siguientes recorren cada familia.

---

## 3. VAE: el autoencoder probabilístico

El **autoencoder variacional** (VAE) de [Kingma y Welling (2013)](/papers/vae-kingma-2013) es la primera familia generativa con **espacio latente continuo**. Su punto de partida es el autoencoder clásico: un encoder comprime $x$ a un código latente $z$, un decoder lo reconstruye. El problema del autoencoder ordinario es que su espacio latente **no es muestreable**: no sabemos de qué distribución sacar un $z$ nuevo, así que comprime y reconstruye pero *no genera*.

El VAE corrige esto con un marco probabilístico. El encoder no produce un código fijo sino una **distribución** $q_\phi(z \mid x)$ (gaussiana, con media y varianza); se impone un **prior** $p(z) = \mathcal{N}(0, I)$ del cual sí se puede muestrear; y el entrenamiento maximiza el **ELBO** (Evidence Lower Bound), una cota inferior de $\log p(x)$:

$$
\text{ELBO} = \underbrace{\mathbb{E}_{q_\phi(z\mid x)}\big[\log p_\theta(x\mid z)\big]}_{\text{reconstrucción}} \;-\; \underbrace{D_{\mathrm{KL}}\big(q_\phi(z\mid x)\,\|\,p(z)\big)}_{\text{regularización}}.
$$

El primer término premia reconstruir bien $x$; el segundo, una divergencia KL, empuja el posterior aproximado hacia el prior, regularizando el latente para que sea continuo y muestreable. La elegancia es que ese término de regularización **emerge de la matemática**, no es un hiperparámetro inventado a mano. Para entrenar con *backpropagation* a través del muestreo estocástico de $z$, el VAE usa el **truco de reparametrización**:

$$
z = \mu + \sigma \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I),
$$

que saca la aleatoriedad fuera del camino del gradiente: el encoder emite $\mu$ y $\sigma$ (deterministas, derivables), y el ruido $\epsilon$ se inyecta aparte. Esto hace el muestreo diferenciable y el VAE entrenable de extremo a extremo.

El resultado es un espacio latente **continuo e interpolable**: se puede generar muestreando $z \sim p(z)$ y decodificando, o interpolar suavemente entre dos códigos para producir transiciones plausibles. Su talón de Aquiles es la **borrosidad**: la verosimilitud gaussiana del decoder equivale a un error cuadrático medio, y ante la incertidumbre ese MSE promedia sobre todos los detalles plausibles en vez de comprometerse con uno nítido —no es un bug de implementación, sino una consecuencia directa de la forma de la verosimilitud—. Por eso, en visión, las GAN y luego la difusión desplazaron al VAE puro como *generador* final —pero el VAE sobrevive como **compresor** en la difusión latente (sección 6).

Otra patología propia del VAE es el **posterior collapse**: cuando el decoder es muy expresivo (por ejemplo, uno autoregresivo potente), el modelo aprende a ignorar el latente —el posterior $q_\phi(z\mid x)$ colapsa al prior, la KL se anula— y el decoder reconstruye sin usar $z$. El latente deja de codificar información útil, vaciando de sentido la representación. Mitigaciones como el *KL annealing*, el $\beta$-VAE o el *free bits* atacan este fenómeno; el VQ-VAE lo elude de raíz con su latente discreto.

Una variante decisiva es el **VQ-VAE** de [van den Oord et al. (2017)](/papers/vq-vae-oord-2017), que reemplaza el latente continuo por uno **discreto**: el encoder emite un vector que se cuantiza al embedding más cercano de un *codebook* aprendido. El VQ-VAE evita el *posterior collapse* (que los latentes sean ignorados por un decoder potente) e inauguró el paradigma "comprimir a códigos discretos + modelar los códigos con un autoregresivo", ancestro directo de VQ-GAN, DALL·E y la regularización VQ del autoencoder de Stable Diffusion.

---

## 4. GANs: el juego adversarial

Las **redes generativas adversarias** (GAN) de [Goodfellow et al. (2014)](/papers/gan-goodfellow-2014) atacan el problema desde un ángulo opuesto: en vez de definir y evaluar $p(x)$ explícitamente —tarea plagada de integrales intratables—, aprenden a *muestrear* enfrentando dos redes en un juego. Un **generador** $G$ toma ruido $z$ y produce una imagen $G(z)$; un **discriminador** $D$ recibe o bien un dato real $x$ o bien la falsificación $G(z)$ y estima la probabilidad de que sea real. La analogía canónica es **falsificadores contra policía**: el generador fabrica billetes falsos cada vez más convincentes, el discriminador afina su detección, y la competencia empuja a ambos a mejorar.

El juego es un **minimax de dos jugadores** sobre la función de valor:

$$
\min_G \max_D \; V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))].
$$

$D$ quiere maximizar (asignar 1 a lo real, 0 a lo falso); $G$ quiere minimizar (que $D$ se crea sus falsificaciones). El paper demuestra que, en el límite ideal, el equilibrio recupera exactamente $p_g = p_{\text{data}}$ —entrenar una GAN equivale a minimizar la divergencia de Jensen-Shannon entre la distribución del modelo y la de los datos—. En la práctica se usa la *non-saturating loss* (entrenar $G$ para maximizar $\log D(G(z))$) porque da gradientes más fuertes temprano.

Las dos patologías que definen a las GAN salen de aquí. La primera es el **mode collapse**: $G$ colapsa muchos valores de $z$ a la misma salida, generando unas pocas muestras convincentes pero perdiendo diversidad —el vértice de cobertura del trilema—. La segunda es la **inestabilidad de entrenamiento**: el equilibrio del juego es delicado, no hay una curva de pérdida monótona que indique progreso. A cambio, las GAN producen imágenes **muy nítidas** (mejores que el VAE) y muestrean en una sola pasada (rápidas).

Dos descendientes son obligatorios. **DCGAN** ([Radford et al., 2015](/papers/dcgan-radford-2015)) es el paper de *ingeniería* que hizo las GAN prácticas con imágenes: antes de él, entrenar una GAN convolucional era un ejercicio frágil que solía colapsar o producir ruido. DCGAN aportó una receta convolucional —convoluciones con *stride* en vez de *pooling* (la red aprende su propio sub/sobre-muestreo), *batch normalization*, ReLU en el generador y LeakyReLU en el discriminador, Tanh a la salida, Adam con $\beta_1=0.5$ y lr$=0.0002$— que estabilizó el entrenamiento de extremo a extremo. De paso demostró que el espacio latente $Z$ tiene estructura semántica: interpolaciones suaves entre puntos de $Z$ y aritmética de vectores tipo `rey − hombre + mujer ≈ reina`, pero sobre caras. **StyleGAN** ([Karras et al., 2019](/papers/stylegan-karras-2019)) llevó las GAN a la cumbre de calidad y, sobre todo, de **control**. En vez de inyectar $z$ por la capa de entrada, una red de mapeo de 8 capas lo proyecta a un espacio intermedio **desenredado** $\mathcal{W}$ —que, a diferencia de $\mathcal{Z}$, no está obligado a seguir la densidad de los datos y puede "desenrollarse"—, y el estilo se inyecta por resolución vía normalización de instancia adaptativa (*AdaIN*), complementado con ruido por-píxel para el detalle estocástico. El efecto emergente, que *no se programa*, es el control jerárquico por escala: las resoluciones gruesas controlan pose e identidad, las medias los rasgos faciales, las finas el color y la microestructura. thispersondoesnotexist.com —caras hiperrealistas de personas inexistentes— es StyleGAN, y marca el techo de calidad que las GAN alcanzaron antes de que la difusión disputara el liderazgo.

---

## 5. Modelos de difusión: convertir datos en ruido y aprender a invertir

Los **modelos de difusión**, vueltos prácticos por DDPM de [Ho et al. (2020)](/papers/ddpm-ho-2020), son la familia que terminó dominando la generación de imágenes. La idea, prestada de la termodinámica de no-equilibrio, tiene dos procesos:

- **Proceso forward** (fijo, sin parámetros): añade ruido gaussiano a una imagen real $x_0$ de forma gradual, en $T$ pasos, según una *variance schedule* $\beta_t$, hasta que $x_T$ es ruido puro $\mathcal{N}(0, I)$. Este proceso *no se aprende*: es el "profesor" que define la tarea.
- **Proceso reverse** (aprendido): una red neuronal aprende a **invertir** ese ruido paso a paso, partiendo de ruido gaussiano y reconstruyendo una imagen.

La simplificación central —el motivo por el que la difusión es entrenable— es que la red **no predice la imagen ni la media de la transición, sino el ruido $\epsilon$** que se añadió. El objetivo colapsa a una pérdida MSE desarmantemente simple:

$$
L_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\Big[\big\|\,\epsilon - \epsilon_\theta(x_t, t)\,\big\|^2\Big],
$$

es decir, "adivina qué ruido le eché a esta imagen en el paso $t$". La red que predice el ruido es una **U-Net** —la misma arquitectura encoder-decoder con *skip connections* nacida para segmentación biomédica— condicionada en el tiempo $t$ vía *embeddings* sinusoidales. Su estructura multi-escala es ideal para operar simultáneamente sobre estructura global y detalle local.

Una conexión teórica profunda explica por qué $\epsilon$-prediction funciona tan bien: predecir el ruido equivale a estimar el **gradiente de la densidad** $\nabla_x \log p(x)$ —el "score"— sobre múltiples niveles de ruido, y el muestreo iterativo se parece a una **dinámica de Langevin** que sube por ese gradiente hacia regiones de alta probabilidad. DDPM unificó así dos tradiciones que llevaban años separadas: los modelos de difusión variacionales y el *score matching*. Cada paso de muestreo resta una fracción del ruido predicho y añade una pizca de ruido fresco; las características de gran escala aparecen primero y los detalles finos al final.

El perfil de la difusión en el trilema es excelente en dos ejes: **calidad alta** (DDPM logró FID 3.17 en CIFAR-10, batiendo a la mayoría de las GAN, incluso a varias condicionales por clase) y **cobertura alta** (al ser un modelo basado en verosimilitud con comportamiento *mode-covering*, no sufre el *mode collapse* de las GAN). El precio es el tercer eje: el **muestreo es lento**, porque requiere $T$ pasos secuenciales (1000 en el paper original), cada uno una evaluación completa de la U-Net, y no se puede paralelizar en el tiempo porque cada $x_{t-1}$ depende de $x_t$. Frente a una GAN que genera en una sola pasada, la difusión es órdenes de magnitud más lenta en inferencia —el único defecto que motivó casi toda la investigación posterior (DDIM, destilación, modelos de consistencia)—. El detalle matemático completo —forward/reverse, $\epsilon$-prediction, *score matching*, *schedules*— vive en el fundamento dedicado de [modelos de difusión](/fundamentos/modelos-de-difusion).

---

## 6. Latent / Stable Diffusion: la síntesis ganadora

La difusión en el espacio de píxeles es carísima: operar sobre tensores de $512 \times 512 \times 3$ por cientos de pasos cuesta cientos de días-GPU de entrenamiento. **Latent Diffusion** de [Rombach et al. (2022)](/papers/latent-diffusion-rombach-2022) —el paper fundacional de Stable Diffusion— resuelve esto combinando tres piezas que el curso ya vio por separado:

1. **VAE para comprimir** (sección 3): un autoencoder lleva la imagen de $512 \times 512 \times 3$ a un latente comprimido de, digamos, $64 \times 64 \times 4$, perceptualmente equivalente pero de mucha menor dimensión. Aquí el defecto de borrosidad del VAE deja de importar: su trabajo no es *generar* sino comprimir/descomprimir fielmente.
2. **Difusión en el latente** (sección 5): el proceso de difusión —el mismo $\epsilon$-prediction de DDPM con su U-Net— opera *dentro* del espacio latente comprimido, no en píxeles. Como los detalles imperceptibles de alta frecuencia ya están abstraídos, el modelo gasta su capacidad solo en estructura semántica.
3. **Cross-attention para condicionar**: un encoder de texto —el de [CLIP](/papers/clip-radford-2021) en Stable Diffusion— transforma el *prompt* en *embeddings* que se inyectan en la U-Net por *cross-attention*. Las *queries* vienen del estado visual de la U-Net, las *keys/values* del texto. Es el mecanismo que hace que el modelo "obedezca" la descripción.

El flujo de inferencia de Stable Diffusion es, entonces: *prompt* → **CLIP** lo codifica → se inicializa ruido latente gaussiano → la **U-Net** hace *denoising condicionado* por la *cross-attention* con el texto, paso a paso → el **decoder del VAE** transforma el latente final en imagen. Esta reducción de cómputo es exactamente lo que permitió que el modelo corriera en GPU de consumidor (~8–10 GB de VRAM), democratizando la generación: mientras DALL·E 2 e Imagen quedaban tras APIs cerradas, Stable Diffusion liberó sus pesos y desató un ecosistema masivo. Es, con amplio margen, el modelo generativo de imágenes más usado del mundo.

---

## 7. Evaluación: FID e Inception Score

¿Cómo se mide objetivamente que un modelo generativo es mejor que otro, si "se ve bonito" es subjetivo y la verosimilitud no es ni siquiera computable para las GAN? La respuesta dominante es la **Fréchet Inception Distance (FID)** de [Heusel et al. (2017)](/papers/fid-heusel-2017), hoy presente en prácticamente toda tabla de resultados generativos.

La idea: pasar las imágenes **reales** y las **generadas** por una red Inception-v3 preentrenada, extraer sus *features* (un vector de 2048 dimensiones de la última capa de *pooling*), modelar cada conjunto como una gaussiana multidimensional, y medir la distancia de Fréchet (Wasserstein-2) entre ambas:

$$
\mathrm{FID} = \lVert m - m_w \rVert_2^2 + \mathrm{Tr}\!\left(C + C_w - 2\,(C\,C_w)^{1/2}\right),
$$

donde $(m_w, C_w)$ son media y covarianza de los *features* reales y $(m, C)$ los de los generados. **Menor FID = mejor**; FID = 0 es el ideal inalcanzable (distribuciones de *features* idénticas). La ventaja decisiva sobre el **Inception Score** (IS) previo —que solo audita confianza de clasificación y variedad de clases de lo generado— es que el FID **sí compara contra los datos reales**. Por eso detecta el *mode collapse* (si el generador colapsa modos, su covarianza $C$ es mucho menor que $C_w$ y el FID se dispara), mejora monótonamente con el realismo, y se valida con degradaciones crecientes. Sus límites: hereda el sesgo de ImageNet del Inception-v3, es sensible al número de muestras (el protocolo estándar usa 50.000), y asume gaussianidad de los *features*.

{{< concept-alert type="clave" >}}
**Por qué FID y no verosimilitud.** Las GAN no tienen densidad explícita: no se puede evaluar $p(x)$ de una muestra. El FID rodea el problema comparando *distribuciones de features* de una red auxiliar, no densidades. Eso lo vuelve una vara común para ordenar GAN, VAE y difusión bajo la misma métrica —el eje cuantitativo de la narrativa de progreso de la Clase 29.
{{< /concept-alert >}}

---

## 8. El trilema generativo en detalle

Vale la pena volver al **trilema** ([Xiao et al., 2021](/papers/diffusion-gan-xiao-2021)) con la perspectiva completa, porque es la lente con la que se ordena todo el panorama. Ningún modelo clásico satisface calidad + velocidad + cobertura a la vez:

- Las **GAN** dan calidad y velocidad, pero sacrifican cobertura (*mode collapse*).
- Los **VAE y flows** dan cobertura y velocidad, pero sacrifican calidad (borrosidad).
- La **difusión** da calidad y cobertura, pero sacrifica velocidad (cientos/miles de pasos).

El diagnóstico de fondo del paper es elegante: la lentitud de la difusión viene del **supuesto gaussiano** sobre la distribución de *denoising* $p_\theta(x_{t-1} \mid x_t)$, que solo es válido en el límite de pasos infinitesimalmente pequeños (de ahí los miles de pasos). Si uno quiere dar pasos *grandes* —y reducir el número total—, la distribución verdadera deja de ser gaussiana y se vuelve **multimodal** (múltiples imágenes limpias plausibles corresponden a la misma imagen ruidosa). El remedio de los **denoising diffusion GANs** es modelar ese paso multimodal con un GAN condicional, logrando generar en ~4 pasos en vez de 1000 (~2000× más rápido) sin perder calidad ni cobertura —un híbrido que ataca el trilema combinando dos familias—. Es el ejemplo canónico de que las familias generativas no son compartimentos estancos: se cruzan para cubrir el vértice que a cada una le falta.

---

## 9. Aplicaciones

Los modelos generativos dejaron de ser generadores de imágenes bonitas para volverse infraestructura útil. Cuatro usos destacan, varios con relevancia médica directa:

- **Data augmentation generativa.** Cuando etiquetar es carísimo, un generador puede *fabricar* datos etiquetados. **DatasetGAN** ([Zhang et al., 2021](/papers/datasetgan-zhang-2021)) lo lleva al extremo: anotando a mano unas **16 a 40 imágenes** generadas por StyleGAN y entrenando un pequeño intérprete sobre sus *feature maps*, convierte el GAN en una fábrica de pares imagen-segmentación, igualando a métodos supervisados que usan hasta 100× más datos anotados. El patrón —entrenar un generador de la modalidad, anotar un puñado de casos, decodificar el conocimiento del generador a etiquetas densas— es exactamente la palanca que más duele en el dominio clínico, donde segmentar lesiones u órganos exige expertos escasos y produce datasets pequeños.
- **Detección de anomalías.** Un VAE entrenado solo con datos *normales* reconstruye mal lo anómalo: un error de reconstrucción alto —o una baja verosimilitud bajo el modelo— señala una muestra fuera de distribución. Como el modelo nunca vio el patrón patológico, no logra comprimirlo y recuperarlo fielmente, y esa discrepancia se vuelve la señal de alerta. Útil para control de calidad industrial y para flaggear hallazgos atípicos en imagen médica sin necesidad de etiquetar las anomalías de antemano.
- **Arte y diseño.** Las GAN (StyleGAN y derivados) habilitaron generación de rostros, transferencia de estilo y edición de atributos por manipulación del latente.
- **Restauración y edición.** La difusión brilla en *inpainting* (rellenar regiones faltantes), super-resolución, colorización y edición guiada por texto, donde su cobertura genera resultados *diversos* en vez de la imagen promedio.

---

## 10. Conexión con el curso y resumen

Los modelos generativos no aparecen de la nada: reutilizan piezas que el curso construyó antes. El **autoencoder** —encoder, *bottleneck*, decoder— es el esqueleto del VAE y del compresor de Stable Diffusion. **CLIP** (Clase 23, multimodal) es el encoder de texto que condiciona la difusión por *cross-attention*, la misma *cross-attention* de los Transformers. La **U-Net** (de la segmentación biomédica) es el predictor de ruido de la difusión. Y el **espacio latente** continuo e interpolable, que el VAE introduce de forma principiada, es el sustrato sobre el que opera toda la generación moderna.

En síntesis: un modelo generativo aprende $p(x)$ para muestrear, reconstruir e interpolar, en contraste con el $p(y \mid x)$ de los discriminativos. El **VAE** aporta el espacio latente continuo (con borrosidad como precio); la **GAN** aporta nitidez vía el juego adversarial (con *mode collapse* como precio); la **difusión** aporta calidad y cobertura simultáneas (con lentitud como precio); y **Latent/Stable Diffusion** combina VAE + difusión + cross-attention con CLIP para convertirse en el modelo dominante. El **trilema** —calidad, velocidad, cobertura— explica por qué ninguna familia gana en todo y por qué los híbridos son el futuro, y el **FID** es la vara común que ordena a todas.

---

## Para profundizar

- [Auto-Encoding Variational Bayes (Kingma y Welling, 2013)](/papers/vae-kingma-2013) — el VAE, ELBO, reparametrización y espacio latente continuo.
- [Neural Discrete Representation Learning — VQ-VAE (van den Oord et al., 2017)](/papers/vq-vae-oord-2017) — la variante discreta y ancestro del autoencoder de Stable Diffusion.
- [Generative Adversarial Nets (Goodfellow et al., 2014)](/papers/gan-goodfellow-2014) — el juego adversarial minimax G vs D.
- [DCGAN (Radford et al., 2015)](/papers/dcgan-radford-2015) — la receta convolucional que hizo las GAN prácticas.
- [StyleGAN (Karras et al., 2019)](/papers/stylegan-karras-2019) — control jerárquico de estilo y la cumbre de calidad de las GAN.
- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](/papers/ddpm-ho-2020) — la difusión hecha práctica: forward/reverse, $\epsilon$-prediction.
- [High-Resolution Image Synthesis with Latent Diffusion Models (Rombach et al., 2022)](/papers/latent-diffusion-rombach-2022) — el paper de Stable Diffusion: difusión en latente + cross-attention.
- [Learning Transferable Visual Models — CLIP (Radford et al., 2021)](/papers/clip-radford-2021) — el encoder de texto que condiciona la difusión.
- [GANs Trained by a Two Time-Scale Update Rule — FID (Heusel et al., 2017)](/papers/fid-heusel-2017) — la métrica estándar de evaluación generativa.
- [Tackling the Generative Learning Trilemma with Denoising Diffusion GANs (Xiao et al., 2021)](/papers/diffusion-gan-xiao-2021) — el marco del trilema y un híbrido GAN-difusión.
- [DatasetGAN (Zhang et al., 2021)](/papers/datasetgan-zhang-2021) — data augmentation generativa con poquísimas etiquetas.

**Fundamentos relacionados:** [Modelos de Difusión](/fundamentos/modelos-de-difusion) · [Aprendizaje Contrastivo](/fundamentos/aprendizaje-contrastivo) · [Foundation Models](/fundamentos/foundation-models) · [Clase 29](/clases/clase-29)
