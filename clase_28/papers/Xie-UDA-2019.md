# Unsupervised Data Augmentation for Consistency Training — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Unsupervised Data Augmentation for Consistency Training* (abreviado **UDA**).
- **Autores:** Qizhe Xie (Google Research, Brain Team / Carnegie Mellon University), Zihang Dai (Google Brain / CMU), Eduard Hovy (CMU), Minh-Thang Luong (Google Brain), Quoc V. Le (Google Brain).
- **Venue:** 34th Conference on Neural Information Processing Systems (**NeurIPS 2020**), Vancouver, Canadá.
- **Año / preprint:** primer preprint en abril de 2019 (arXiv:1904.12848v1); versión final v6, 5 de noviembre de 2020. De ahí la doble datación habitual "Xie et al. 2019 / 2020".
- **Código:** [github.com/google-research/uda](https://github.com/google-research/uda), incluyendo el sistema de back-translation y los checkpoints de traducción liberados.

UDA es uno de los trabajos canónicos del **aprendizaje semi-supervisado moderno** (SSL) y de la familia de la **regularización por consistencia** (*consistency training*). Su tesis es de una simplicidad desarmante: en el SSL basado en consistencia, lo que limita el rendimiento no es el algoritmo de propagación de etiquetas sino **la calidad del ruido** que se inyecta a los ejemplos sin etiquetar. Los métodos previos (Π-Model, Mean Teacher, VAT) perturbaban los ejemplos con ruido aditivo gaussiano, dropout o ruido adversarial — perturbaciones *locales* y a menudo poco realistas. UDA propone sustituir ese ruido por **augmentaciones de datos de última generación, específicas del dominio** —**RandAugment** en imágenes, **back-translation** en texto, reemplazo por **TF-IDF** en clasificación de tópicos— y forzar al modelo a predecir lo mismo para el ejemplo original y para su versión aumentada, minimizando una divergencia (cross-entropy / KL) entre ambas distribuciones de salida.

El resultado más citado es contundente: en **IMDb**, con solo **20 ejemplos etiquetados**, UDA alcanza una tasa de error de **4.20%**, superando al modelo supervisado estado del arte entrenado con **25.000** etiquetas (1.250× más datos). En **CIFAR-10** con 250 etiquetas baja a **5.43%** de error; en **SVHN** con 250 etiquetas, **2.72%**. Y a diferencia de muchos métodos de SSL que solo funcionan en el régimen de pocos datos, UDA **escala**: en **ImageNet con 10% de etiquetas** sube la precisión top-1 de 58.84 a **68.78**, y con el 100% de etiquetas más 1.3M de ejemplos extra sin etiquetar la mejora de 78.43 a **79.05**.

Para el **Laboratorio 28** (Práctico de Autosupervisión / `Practico_Autosupervision_UDA`) este es **el** paper: el laboratorio implementa y experimenta directamente con la receta de UDA. Entender este documento es entender qué hace el lab por dentro y por qué cada técnica auxiliar (TSA, confidence masking, sharpening) está donde está.

## 2. Contexto: aprendizaje semi-supervisado y regularización por consistencia

La debilidad estructural del aprendizaje profundo, según abre el paper, es que **típicamente requiere muchísimos datos etiquetados** para funcionar bien. Etiquetar es caro; recolectar datos crudos sin etiquetar es barato y abundante. El **aprendizaje semi-supervisado** (SSL; Chapelle et al., 2006) es uno de los paradigmas más prometedores para cerrar esa brecha: un pequeño conjunto etiquetado $p_L(x)$ más un gran conjunto sin etiquetar $p_U(x)$.

Dentro del SSL, hacia 2018–2019 la familia que mejor funcionaba en los benchmarks era la **regularización por consistencia**. Su intuición es geométrica y se puede enunciar en una frase: *un buen modelo debería ser robusto a pequeños cambios en la entrada o en los estados ocultos*. Si perturbo levemente un ejemplo, la predicción no debería cambiar. Formalmente, los métodos de consistencia comparten un esqueleto:

1. Dada una entrada $x$, computar la distribución de salida $p_\theta(y \mid x)$ y una versión ruidosa $p_\theta(y \mid x, \epsilon)$ inyectando un pequeño ruido $\epsilon$ (en la entrada o en los estados ocultos).
2. Minimizar una métrica de divergencia entre ambas distribuciones, $D\big(p_\theta(y \mid x)\,\|\,p_\theta(y \mid x, \epsilon)\big)$.

Esto **fuerza al modelo a ser insensible al ruido** $\epsilon$ y, por tanto, más suave (*smoother*) respecto a cambios en el espacio de entrada. Desde otra óptica, minimizar la pérdida de consistencia **propaga gradualmente la información de las etiquetas** desde los ejemplos etiquetados hacia los no etiquetados: si dos ejemplos están conectados por una transformación de bajo ruido, comparten etiqueta, y el conocimiento fluye por esa vecindad.

Lo que distinguía a los métodos previos entre sí era **cómo y dónde** se inyectaba el ruido. El catálogo típico: **ruido gaussiano aditivo** (Pseudo-ensemble, Bachman et al. 2014), **dropout**, **ruido adversarial** (VAT, Miyato et al. 2018, que aproxima la dirección de máxima sensibilidad del modelo), **recortes y volteos aleatorios** (Π-Model, Temporal Ensembling), o consistencia en el **espacio de parámetros** (Mean Teacher, fast-SWA). El punto ciego que UDA identifica: **nadie se había preguntado seriamente por la *calidad* —la "forma"— de la operación de ruido $q$**. Se asumía que cualquier perturbación pequeña servía. UDA demuestra que no: hay perturbaciones mucho mejores que otras.

## 3. Contribución central: el ruido de calidad importa

La observación seminal de UDA conecta dos literaturas que vivían separadas. Por un lado, el aprendizaje **supervisado** había hecho enormes progresos en **augmentación de datos**: AutoAugment, RandAugment y Cutout en visión; back-translation en NLP; SpecAugment en voz. Por otro, el **SSL por consistencia** seguía usando ruido primitivo. La hipótesis de UDA: **las augmentaciones que funcionan mejor en supervisado también funcionarán mejor como fuente de ruido en consistencia**. Y lo verifican empíricamente: existe una **correlación positiva fuerte** entre la efectividad de una augmentación en el régimen supervisado y su efectividad en el semi-supervisado (Tablas 1 y 2 del paper).

La receta, entonces, es directa: **sustituir las operaciones de ruido simples por métodos de augmentación de datos de alta calidad** dentro del mismo marco de consistencia. Para enfatizar este cambio bautizan al método **Unsupervised Data Augmentation**: augmentación de datos aplicada *sin etiquetas*, sobre el conjunto $p_U(x)$.

El paper articula **tres razones** por las cuales las augmentaciones avanzadas baten al ruido simple:

- **Ruido válido (*valid noise*).** Las augmentaciones que rinden en supervisado generan ejemplos *realistas* que **preservan la etiqueta** verdadera del original. Por tanto es seguro forzar la consistencia entre la predicción del ejemplo no etiquetado y la de su versión aumentada: ambos deben tener la misma respuesta correcta. El ruido gaussiano, en cambio, puede empujar el ejemplo fuera de la variedad de datos realistas.
- **Ruido diverso (*diverse noise*).** Las augmentaciones avanzadas pueden hacer **modificaciones grandes** a la entrada sin cambiar la etiqueta (una paráfrasis completa, una imagen muy transformada), mientras que el ruido gaussiano solo hace cambios locales. Forzar consistencia sobre un conjunto diverso de aumentaciones mejora drásticamente la **eficiencia muestral**: cada ejemplo no etiquetado enseña más.
- **Sesgos inductivos dirigidos (*targeted inductive biases*).** Cada tarea requiere sesgos inductivos distintos. Una augmentación que funciona bien en supervisado esencialmente **codifica el sesgo inductivo faltante** para esa tarea (invariancia a rotaciones, a paráfrasis, a cambios de iluminación), y lo transfiere al régimen no etiquetado.

Las contribuciones que el paper enumera: (1) las augmentaciones SOTA del supervisado son una fuente *superior* de ruido bajo consistencia; (2) UDA iguala o supera al supervisado puro que usa órdenes de magnitud más etiquetas; (3) UDA es **complementario al transfer learning** (se combina con BERT) y funciona en el **régimen de muchos datos** (ImageNet); (4) un **análisis teórico** que explica por qué.

## 4. Método: la pérdida combinada y las augmentaciones

### 4.1. El objetivo de entrenamiento

UDA entrena con una pérdida que **suma** dos términos (Figura 1 del paper): una pérdida supervisada estándar sobre los pocos ejemplos etiquetados, y una pérdida de consistencia no supervisada sobre el gran conjunto sin etiquetar, ponderada por $\lambda$:

$$
\min_\theta\ \mathcal{J}(\theta) = \underbrace{\mathbb{E}_{x_1 \sim p_L(x)}\big[-\log p_\theta(f^*(x_1)\mid x_1)\big]}_{\text{cross-entropy supervisada}} + \lambda\, \underbrace{\mathbb{E}_{x_2 \sim p_U(x)}\,\mathbb{E}_{\hat{x}\sim q(\hat{x}\mid x_2)}\Big[\mathrm{CE}\big(p_{\tilde\theta}(y\mid x_2)\,\|\,p_\theta(y\mid \hat{x})\big)\Big]}_{\text{consistencia no supervisada}}
$$

Las piezas, una por una:

- $f^*$ es el clasificador perfecto que se quiere aprender; el primer término es la log-verosimilitud negativa habitual sobre el conjunto etiquetado.
- $q(\hat{x}\mid x)$ es la **transformación de augmentación**: dada una imagen o texto $x$, produce una versión aumentada $\hat{x}$. Esta es la sustitución clave: donde antes había ruido gaussiano, ahora hay RandAugment o back-translation.
- $p_{\tilde\theta}(y\mid x_2)$ es la predicción sobre el ejemplo **original** (no aumentado), y $p_\theta(y\mid\hat{x})$ la predicción sobre el ejemplo **aumentado**. La consistencia exige que ambas coincidan.
- $\tilde\theta$ es una **copia congelada (*stop-gradient*) de los parámetros actuales** $\theta$: el gradiente **no** se propaga a través de $\tilde\theta$. Esto, heredado de VAT, evita una solución degenerada y trata la predicción del original como un *objetivo* fijo (un "profesor" instantáneo) que el aumentado debe imitar.
- Aunque se usa **cross-entropy** en la formulación, equivale a minimizar la **divergencia KL** entre la distribución del original y la del aumentado (la entropía del objetivo es constante respecto a $\theta$ por el stop-gradient). Por eso en la Clase 28 se describe como "minimizar la KL entre la predicción del dato y su versión transformada".

En la práctica, en cada iteración se computa la pérdida supervisada sobre un minibatch etiquetado y la de consistencia sobre un minibatch sin etiquetar (con un **batch size mayor** para la consistencia, dado que los datos no etiquetados son abundantes), y se suman. Se fija $\lambda = 1$ en la mayoría de experimentos. En visión, a los ejemplos etiquetados se les aplican augmentaciones simples (recorte y volteo); para minimizar la discrepancia entre entrenamiento supervisado y predicción sobre no etiquetados, esas mismas augmentaciones simples se aplican al original $x$ al computar $p_{\tilde\theta}(y\mid x)$.

### 4.2. Estrategias de augmentación por tarea

- **RandAugment (clasificación de imágenes).** Inspirado en AutoAugment (Cubuk et al. 2018), pero sin búsqueda costosa de políticas. RandAugment muestrea **uniformemente** de un conjunto de **15 transformaciones** de la Python Image Library —Invert, Cutout, Sharpness, AutoContrast, Posterize, ShearX/Y, TranslateX/Y, Rotate, Equalize, Contrast, Color, Solarize, Brightness— eligiendo una magnitud aleatoria en $[1,10)$ con probabilidad fija de 0.5. Es más simple que AutoAugment y **no necesita datos etiquetados** para buscar la política óptima, lo que lo hace ideal para el régimen no supervisado.

- **Back-translation (clasificación de texto).** Para aumentar un texto $x$ en idioma A, se lo traduce a un idioma B y luego de vuelta a A, obteniendo una **paráfrasis** $\hat{x}$ (Sennrich et al. 2015; Edunov et al. 2018). El paper usa modelos de traducción **WMT'14 inglés↔francés** en ambas direcciones. Un detalle crítico: para maximizar **diversidad** usan **muestreo aleatorio con temperatura ajustable** en lugar de *beam search*. Con temperatura 0 el muestreo degenera en *greedy* y produce paráfrasis idénticas (válidas pero no diversas); con temperatura 1 produce paráfrasis muy diversas pero apenas legibles (diversas pero no válidas). El punto óptimo está en **0.7–0.9**. Esto materializa el **trade-off diversidad–validez** que el paper discute en el apéndice.

- **Reemplazo de palabras por TF-IDF (clasificación de tópicos).** Back-translation preserva la semántica global pero no controla *qué* palabras se conservan. En tareas de tópico (p. ej. DBPedia) ciertas palabras clave son decisivas. Por eso UDA propone reemplazar palabras con **bajo TF-IDF** (poco informativas) por otras palabras poco informativas, **conservando las de alto TF-IDF**. La probabilidad de reemplazo se hace negativamente correlacionada con el score TF-IDF de cada palabra.

### 4.3. Técnicas de entrenamiento adicionales

Estas técnicas son tan importantes como la pérdida misma para que UDA funcione en regímenes extremos; el Lab 28 las expone directamente.

- **Training Signal Annealing (TSA).** Resuelve el problema central del régimen de poquísimas etiquetas: cuando hay 20 etiquetas y millones de ejemplos sin etiquetar, **el modelo sobreajusta rápido los pocos labels mientras aún subajusta el resto**. TSA libera *gradualmente* la "señal de entrenamiento" supervisada: en el paso $t$, si la probabilidad predicha para la categoría correcta $p_\theta(y^*\mid x)$ **supera** un umbral $\eta_t$, ese ejemplo se **remueve** de la pérdida supervisada. El umbral crece de $1/K$ ($K$ = número de clases) hasta 1 según un cronograma, actuando como **techo** que impide sobre-entrenar ejemplos fáciles. Hay tres cronogramas: **exp-schedule** (libera la señal sobre todo al final; ideal cuando el modelo tiende a sobreajustar, p. ej. pocas etiquetas), **linear-schedule** y **log-schedule** (libera la señal temprano; ideal cuando hay muchas etiquetas o regularización fuerte). Se define $\eta_t = \alpha_t(1 - 1/K) + 1/K$.

- **Confidence-based masking (enmascaramiento por confianza).** La pérdida de consistencia se computa **solo** sobre los ejemplos sin etiquetar en los que el modelo está **suficientemente seguro**: solo si la probabilidad máxima entre categorías supera un umbral $\beta$ (0.8 para CIFAR-10/SVHN, 0.5 para ImageNet). Esto evita propagar consistencia desde predicciones ruidosas o incorrectas, que solo añadirían ruido al entrenamiento.

- **Sharpening (afilado de predicciones).** Como regularizar hacia baja entropía es beneficioso (Grandvalet & Bengio 2005), UDA **afila** la distribución objetivo del ejemplo original usando una **temperatura Softmax baja** $\tau$ (0.4 para CIFAR-10/SVHN/ImageNet): $p^{(\text{sharp})}_{\tilde\theta}(y\mid x) = \frac{\exp(z_y/\tau)}{\sum_{y'}\exp(z_{y'}/\tau)}$. Una temperatura $<1$ vuelve el objetivo más "puntiagudo" (más cercano a one-hot), reforzando predicciones confiadas. Combinados, masking y sharpening forman el objetivo de consistencia efectivo sobre cada minibatch.

- **Domain-relevance data filtering (filtrado por relevancia de dominio).** Para poder usar datos no etiquetados *fuera de dominio* (mucho más fáciles de recolectar) sin sufrir el desajuste de distribución de clases, UDA usa el modelo base entrenado en el dominio para inferir etiquetas sobre un gran conjunto externo y **selecciona los ejemplos de mayor confianza por categoría**. Así filtran 1.3M de imágenes de JFT para el experimento de ImageNet al 100%.

## 5. Análisis teórico: el grafo de augmentación

El paper ofrece una justificación teórica elegante. Bajo tres supuestos simplificadores sobre la augmentación —**in-domain** ($p_U(\hat{x})>0$ para los aumentados), **label-preserving** ($f^*(x)=f^*(\hat{x})$) y **reversible** (si $q(\hat{x}\mid x)>0$ entonces $q(x\mid\hat{x})>0$)— se construye un grafo $G_{p_U}$ donde cada nodo es un ejemplo y existe una arista entre $x$ y $\hat{x}$ si la augmentación los conecta. Como la augmentación preserva la etiqueta, ejemplos de clases distintas viven en **componentes (subgrafos) desconectados** distintos.

La intuición clave: para una clasificación de $N$ categorías, basta con **un solo ejemplo etiquetado por componente** para propagar la etiqueta al resto del componente recorriéndolo vía la augmentación. La augmentación supervisada solo propaga la etiqueta a los **vecinos directos** del nodo etiquetado; la augmentación **no supervisada (UDA) recorre el componente completo**. Más aún, el **número de componentes** acota inferiormente la cantidad mínima de etiquetas necesarias para aprender un clasificador perfecto. El **Teorema 1** lo formaliza: la probabilidad de no poder inferir la etiqueta de un ejemplo de test, dados $m$ ejemplos etiquetados, es $P(A)=\sum_i P_i(1-P_i)^m$, y $m = O(k/\epsilon)$ etiquetas garantizan un error $O(\epsilon)$, donde $k$ es el número de componentes.

La consecuencia conecta de vuelta con la tesis empírica: **mejores augmentaciones (más diversas) generan más aristas, mejor conectividad y por tanto menos componentes** $k$ — y menos componentes significa que se necesitan menos etiquetas. La diversidad de RandAugment / back-translation no es un detalle cosmético: es lo que reduce $k$ y lo que hace que 20 etiquetas basten.

## 6. Experimentos: números reales

### 6.1. Correlación supervisado ↔ semi-supervisado (Tablas 1 y 2)

Sobre **CIFAR-10**, comparando augmentaciones en supervisado (50k labels) vs. semi-supervisado (4k labels): Crop & flip da 5.36 / 10.94, Cutout da 4.42 / 5.43, y **RandAugment da 4.23 / 4.32**. Sobre **Yelp-5** (texto), en supervisado (650k) vs. semi-supervisado (2.5k): el método "7" (sin augmentación) da 38.36 / 50.80, Switchout 37.24 / 43.38, y **back-translation 36.71 / 41.35**. En ambos dominios, **la mejor augmentación supervisada es la mejor semi-supervisada** — validando la hipótesis fundacional.

### 6.2. Benchmarks de visión: CIFAR-10 y SVHN (Tabla 3, Figura 4)

Con Wide-ResNet-28-2 y etiquetas variables, UDA supera consistentemente a **VAT** y a **MixMatch** (trabajo paralelo). La diferencia con VAT es reveladora: como solo cambia el *proceso de ruido*, demuestra que las augmentaciones realistas baten al ruido adversarial (que introduce artefactos de alta frecuencia inexistentes en imágenes reales). Con **4k labels en CIFAR-10**, UDA logra **4.32%** de error (WRN-28-2), bajando a **3.7%** (Shake-Shake) y **2.7%** (PyramidNet+ShakeDrop) — **igualando** el rendimiento totalmente supervisado entrenado con 50.000 ejemplos (10× más). En **SVHN con 1k labels**: **2.23%**, igualando al supervisado con 73.257 ejemplos.

### 6.3. Clasificación de texto y combinación con BERT (Tabla 4)

UDA se evalúa con cuatro inicializaciones: Transformer aleatorio, BERT-base, BERT-large y BERT-large fine-tuneado en datos de dominio. El resultado emblemático: **IMDb con 20 labels**. Sin UDA, BERT-large da 11.72% de error; **con UDA, 4.78%**; con BERT fine-tuneado e iniciado, UDA baja de **6.50 a 4.20%** — superando la SOTA pre-BERT entrenada con las 25.000 etiquetas completas (4.32%). En Yelp-2 y Amazon-2 (binarios) con 20 labels UDA es igualmente competitivo. La lección: **UDA es complementario al transfer learning**, no redundante. En las tareas de 5 categorías (Yelp-5, Amazon-5) persiste una brecha respecto al supervisado completo — son intrínsecamente más difíciles, y el paper lo señala como margen futuro.

### 6.4. Escalabilidad en ImageNet (Tabla 5)

Con ResNet-50: en el setting de **10% de etiquetas** (resto in-domain sin etiquetar), UDA sube la precisión top-1/top-5 de 58.84/80.56 (RandAugment supervisado) a **68.78/88.80**. En el setting de **100% de etiquetas** más 1.3M de imágenes externas de JFT filtradas por dominio, sube de 78.43/94.37 a **79.05/94.49**. UDA no solo funciona con pocos datos: **escala al régimen de muchos datos y aprovecha datos fuera de dominio**.

## 7. Limitaciones reconocidas

- **Brecha en tareas multiclase difíciles.** En clasificación de sentimiento de 5 categorías persiste una diferencia clara entre UDA (con 500 labels/clase) y BERT con el conjunto supervisado completo. El paper lo reconoce como problema abierto.
- **Dependencia de augmentaciones de calidad.** Toda la ventaja de UDA descansa en disponer de una augmentación fuerte, válida y diversa *para el dominio*. En dominios sin una buena augmentación conocida (datos tabulares, señales muy específicas), la receta pierde su palanca principal.
- **Costo de la back-translation.** Requiere modelos de traducción entrenados (aunque el paper argumenta que la tarea de traducción es distinta de la de clasificación y no usa etiquetas de la tarea destino). Es un componente pesado de preprocesado.
- **Supuestos teóricos idealizados.** El análisis del grafo asume augmentaciones in-domain, label-preserving y reversibles — supuestos que en la práctica solo se cumplen aproximadamente (de ahí el trade-off diversidad–validez que el propio paper debe gestionar con la temperatura).
- **Sensibilidad a hiperparámetros del régimen extremo.** TSA (qué cronograma), $\beta$ (umbral de confianza) y $\tau$ (temperatura de sharpening) requieren ajuste por dataset; sin ellos, el régimen de 20 labels no converge bien.

## 8. Impacto: la regularización por consistencia moderna

UDA, junto con MixMatch (trabajo paralelo de 2019), marca el inicio de la **era moderna de la regularización por consistencia** y pavimenta el camino directo a **FixMatch** (Sohn et al. 2020), que sintetiza UDA y MixMatch en una receta minimalista (augmentación débil → pseudo-etiqueta con umbral → consistencia con augmentación fuerte). El patrón "augmentación fuerte como ruido + pseudo-etiqueta confiada del original + consistencia" se volvió el estándar del SSL de la generación 2020. La idea de que **la calidad de la augmentación es el cuello de botella, no el algoritmo**, reorientó el campo.

El otro legado es la **bisagra entre supervisado, semi-supervisado y autosupervisado**: UDA muestra que las invariancias descubiertas para el supervisado (RandAugment) son exactamente las que se necesitan para explotar datos sin etiquetar — la misma intuición de "predecir igual para un dato y su transformación" que sostiene el aprendizaje contrastivo (SimCLR, MoCo). En el dominio de salud y de matching de registros, donde las etiquetas curadas son caras pero los datos crudos abundan, esta es exactamente la palanca relevante.

## 9. Conexión con la Clase 28 y el Laboratorio 28

La **Clase 28 (Aprendizaje Autosupervisado)** dedica su **Sección 2** —"Autosupervisión para potenciar el aprendizaje supervisado"— íntegramente a UDA (Xie et al. 2019). El hilo conceptual que la clase desarrolla es justamente el de este paper:

- **La intuición rectora:** "el modelo debe predecir de forma similar para un dato y para su versión transformada". La clase la presenta como puente entre lo autosupervisado y lo supervisado: las transformaciones que usamos para crear tareas pretexto (rotaciones, recortes, paráfrasis) sirven también para regularizar con datos sin etiquetar.
- **La divergencia KL** entre la predicción del dato original y la del aumentado como objetivo de consistencia — el segundo término de la pérdida de la §4.1, con su stop-gradient sobre el objetivo.
- **La tabla de ImageNet** (Tabla 5 de este análisis): el caso de 10% de etiquetas que la clase usa para ilustrar que la autosupervisión/consistencia rinde incluso con etiquetas escasas, y la mención de **back-translation** como la augmentación análoga para el dominio de texto.

El **Laboratorio 28** (`Practico_Autosupervision_UDA`) es la implementación práctica de esta receta. El estudiante experimenta con: el ensamblaje de la **pérdida combinada** (cross-entropy supervisada + KL de consistencia ponderada por $\lambda$); la generación de **augmentaciones** (RandAugment en visión, back-translation en texto); y las técnicas auxiliares —**TSA**, **confidence masking**, **sharpening**— que vuelven viable el régimen de poquísimas etiquetas. Entender por qué cada pieza está donde está (por qué el stop-gradient, por qué afilar, por qué enmascarar por confianza, por qué TSA en el régimen de 20 labels) es la diferencia entre ejecutar el notebook y comprender el mecanismo — exactamente la lección que este paper, base del laboratorio, transmite.

Enlaces internos del curso: fundamento [/fundamentos/aprendizaje-autosupervisado](/fundamentos/aprendizaje-autosupervisado), clase [/clases/clase-28](/clases/clase-28), laboratorio [/laboratorios/lab-28](/laboratorios/lab-28).
