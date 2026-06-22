---
title: "UDA: Unsupervised Data Augmentation for Consistency Training (2019)"
weight: 324
math: true
---

{{< paper-card
    title="Unsupervised Data Augmentation for Consistency Training"
    authors="Qizhe Xie, Zihang Dai, Eduard Hovy, Minh-Thang Luong, Quoc V. Le"
    year="2020"
    venue="NeurIPS 2020"
    pdf="/papers/uda-xie-2019.pdf"
    arxiv="1904.12848" >}}
El paper que sostiene el **Laboratorio 28**. UDA es uno de los trabajos canónicos del **aprendizaje semi-supervisado moderno** y de la **regularización por consistencia** (*consistency training*). Su tesis es desarmante: lo que limita el rendimiento del semi-supervisado por consistencia no es el algoritmo, sino **la calidad del ruido** que se inyecta a los ejemplos sin etiquetar. La propuesta: reemplazar el ruido primitivo (gaussiano, dropout, adversarial) por **augmentaciones de datos de última generación** —**RandAugment** en imágenes, **back-translation** en texto— y forzar al modelo a predecir lo mismo para el ejemplo original y para su versión aumentada, minimizando la **divergencia KL** entre ambas distribuciones. El resultado emblemático: en **IMDb con solo 20 ejemplos etiquetados**, UDA alcanza 4.20% de error, superando al supervisado estado del arte entrenado con 25.000 etiquetas.
{{< /paper-card >}}

---

## Contexto: semi-supervisado y regularización por consistencia

La debilidad estructural del aprendizaje profundo es que típicamente requiere **muchísimos datos etiquetados**. Etiquetar es caro; recolectar datos crudos sin etiquetar es barato y abundante. El **aprendizaje semi-supervisado** (SSL) busca cerrar esa brecha combinando un pequeño conjunto etiquetado $p_L(x)$ con un gran conjunto sin etiquetar $p_U(x)$.

Hacia 2018-2019, la familia de SSL que mejor funcionaba era la **regularización por consistencia**. Su intuición es geométrica: *un buen modelo debería ser robusto a pequeños cambios en la entrada*. Si perturbo levemente un ejemplo, la predicción no debería cambiar. Los métodos de consistencia comparten un esqueleto:

1. Dada una entrada $x$, computar la distribución de salida $p_\theta(y \mid x)$ y una versión ruidosa $p_\theta(y \mid x, \epsilon)$ inyectando un pequeño ruido $\epsilon$.
2. Minimizar una divergencia entre ambas, $D\big(p_\theta(y \mid x)\,\|\,p_\theta(y \mid x, \epsilon)\big)$.

Esto **fuerza al modelo a ser insensible al ruido** y, de paso, **propaga la información de las etiquetas** desde los ejemplos etiquetados hacia los no etiquetados: si dos ejemplos están conectados por una transformación de bajo ruido, comparten etiqueta y el conocimiento fluye por esa vecindad. Lo que distinguía a los métodos previos (Π-Model, Mean Teacher, VAT) era **cómo y dónde** se inyectaba el ruido: gaussiano aditivo, dropout, ruido adversarial, recortes y volteos. El punto ciego que UDA identifica: **nadie se había preguntado por la *calidad* —la "forma"— del ruido**. Se asumía que cualquier perturbación pequeña servía. UDA demuestra que no.

## Contribución central: el ruido de calidad importa

UDA conecta dos literaturas que vivían separadas. Por un lado, el aprendizaje **supervisado** había hecho enormes progresos en **augmentación de datos** (AutoAugment, RandAugment, Cutout en visión; back-translation en texto). Por otro, el **SSL por consistencia** seguía usando ruido primitivo. La hipótesis de UDA: **las augmentaciones que funcionan mejor en supervisado funcionarán mejor también como fuente de ruido en consistencia**. Lo verifican empíricamente: existe una **correlación positiva fuerte** entre la efectividad de una augmentación en el régimen supervisado y en el semi-supervisado.

La receta, entonces, es directa: **sustituir el ruido simple por augmentaciones de alta calidad** dentro del mismo marco de consistencia, aplicadas *sin etiquetas* sobre $p_U(x)$ —de ahí el nombre **Unsupervised Data Augmentation**. El paper articula **tres razones** por las que las augmentaciones avanzadas baten al ruido simple:

- **Ruido válido.** Las buenas augmentaciones generan ejemplos *realistas* que **preservan la etiqueta** verdadera. Es seguro forzar consistencia: original y aumentado deben tener la misma respuesta correcta. El ruido gaussiano puede empujar el ejemplo fuera de la variedad de datos realistas.
- **Ruido diverso.** Las augmentaciones avanzadas hacen **modificaciones grandes** sin cambiar la etiqueta (una paráfrasis completa, una imagen muy transformada), mientras el ruido gaussiano solo hace cambios locales. Más diversidad mejora drásticamente la **eficiencia muestral**: cada ejemplo no etiquetado enseña más.
- **Sesgos inductivos dirigidos.** Cada tarea requiere invariancias distintas (a rotaciones, a paráfrasis, a iluminación). Una augmentación que rinde en supervisado **codifica el sesgo inductivo faltante** y lo transfiere al régimen no etiquetado.

## Método: la pérdida combinada

UDA entrena con una pérdida que **suma** dos términos: una cross-entropy supervisada estándar sobre los pocos ejemplos etiquetados, y una pérdida de consistencia no supervisada sobre el gran conjunto sin etiquetar, ponderada por $\lambda$:

$$
\min_\theta\ \mathcal{J}(\theta) = \underbrace{\mathbb{E}_{x_1 \sim p_L(x)}\big[-\log p_\theta(f^*(x_1)\mid x_1)\big]}_{\text{cross-entropy supervisada}} + \lambda\, \underbrace{\mathbb{E}_{x_2 \sim p_U(x)}\,\mathbb{E}_{\hat{x}\sim q(\hat{x}\mid x_2)}\Big[\mathrm{CE}\big(p_{\tilde\theta}(y\mid x_2)\,\|\,p_\theta(y\mid \hat{x})\big)\Big]}_{\text{consistencia no supervisada}}
$$

Las piezas clave:

- $q(\hat{x}\mid x)$ es la **transformación de augmentación**: donde antes había ruido gaussiano, ahora hay RandAugment o back-translation. Esta es la sustitución central.
- $p_{\tilde\theta}(y\mid x_2)$ es la predicción sobre el ejemplo **original** y $p_\theta(y\mid\hat{x})$ la del **aumentado**; la consistencia exige que coincidan.
- $\tilde\theta$ es una **copia congelada (*stop-gradient*)** de los parámetros actuales: el gradiente **no** se propaga por $\tilde\theta$. Esto evita una solución degenerada y trata la predicción del original como un *objetivo* fijo —un "profesor" instantáneo— que el aumentado debe imitar.
- Aunque la fórmula usa cross-entropy, equivale a **minimizar la divergencia KL** entre la distribución del original y la del aumentado (la entropía del objetivo es constante por el stop-gradient). De ahí la descripción de la Clase 28: "minimizar la KL entre la predicción del dato y su versión transformada".

En la práctica se fija $\lambda = 1$ y se usa un **batch mayor** para la consistencia, dado que los datos sin etiquetar abundan.

### Augmentaciones por tarea

- **RandAugment (imágenes).** Inspirado en AutoAugment pero sin búsqueda costosa de políticas. Muestrea **uniformemente** de un conjunto de **15 transformaciones** (Cutout, Sharpness, AutoContrast, ShearX/Y, TranslateX/Y, Rotate, Equalize, Contrast, Color, Solarize, Brightness…) con magnitud aleatoria. No necesita datos etiquetados para buscar la política óptima, lo que lo vuelve ideal para el régimen no supervisado.
- **Back-translation (texto).** Para aumentar un texto en idioma A se lo traduce a B y de vuelta a A, obteniendo una **paráfrasis**. El paper usa modelos WMT'14 inglés↔francés. Detalle crítico: para maximizar **diversidad** usan **muestreo con temperatura** (no *beam search*). Con temperatura 0 las paráfrasis son idénticas (válidas pero no diversas); con temperatura 1 son muy diversas pero apenas legibles (diversas pero no válidas). El punto óptimo está en **0.7-0.9**: este es el **trade-off diversidad-validez**.
- **Reemplazo por TF-IDF (tópicos).** Para clasificación de tópicos, donde ciertas palabras clave son decisivas, UDA reemplaza palabras de **bajo TF-IDF** (poco informativas) **conservando las de alto TF-IDF**.

### Técnicas de entrenamiento adicionales

Estas técnicas son tan importantes como la pérdida misma para que UDA funcione en regímenes extremos; el Lab 28 las expone directamente.

- **Training Signal Annealing (TSA).** Resuelve el problema del régimen de poquísimas etiquetas: con 20 labels y millones de ejemplos sin etiquetar, el modelo **sobreajusta rápido los pocos labels mientras subajusta el resto**. TSA libera *gradualmente* la señal supervisada: si la probabilidad predicha para la clase correcta supera un umbral $\eta_t$, ese ejemplo se **remueve** de la pérdida supervisada. El umbral crece de $1/K$ ($K$ = número de clases) hasta 1, actuando como **techo** que impide sobre-entrenar ejemplos fáciles. Hay tres cronogramas: **exp** (libera tarde, ideal con pocas etiquetas), **linear** y **log** (liberan temprano, ideal con muchas etiquetas).
- **Confidence-based masking.** La consistencia se computa **solo** sobre los ejemplos sin etiquetar donde el modelo está suficientemente seguro: solo si la probabilidad máxima supera un umbral $\beta$ (0.8 en CIFAR-10/SVHN, 0.5 en ImageNet). Evita propagar consistencia desde predicciones ruidosas.
- **Sharpening.** Como regularizar hacia baja entropía es beneficioso, UDA **afila** la distribución objetivo del original con una **temperatura Softmax baja** $\tau$ (0.4): vuelve el objetivo más "puntiagudo" (cercano a one-hot), reforzando predicciones confiadas.
- **Domain-relevance filtering.** Para usar datos sin etiquetar *fuera de dominio*, el modelo base infiere etiquetas sobre un gran conjunto externo y **selecciona los de mayor confianza por categoría** (así filtran 1.3M imágenes de JFT para ImageNet al 100%).

## Análisis teórico: el grafo de augmentación

Bajo tres supuestos —augmentación **in-domain**, **label-preserving** y **reversible**— se construye un grafo donde cada nodo es un ejemplo y hay una arista entre $x$ y $\hat{x}$ si la augmentación los conecta. Como preserva la etiqueta, ejemplos de clases distintas viven en **componentes desconectados**. La intuición clave: basta **un solo ejemplo etiquetado por componente** para propagar la etiqueta a todo el componente recorriéndolo vía augmentación —la supervisión sola solo alcanza a los vecinos directos; UDA recorre el componente completo. El **número de componentes** acota inferiormente la cantidad mínima de etiquetas necesarias. La consecuencia conecta con la tesis empírica: **mejores augmentaciones (más diversas) generan más aristas, mejor conectividad y menos componentes** —y menos componentes significa menos etiquetas. La diversidad de RandAugment / back-translation es lo que hace que 20 etiquetas basten.

## Experimentos: números reales

- **Correlación supervisado ↔ semi-supervisado.** En CIFAR-10 y Yelp-5, **la mejor augmentación supervisada es la mejor semi-supervisada** —validando la hipótesis fundacional.
- **Visión (CIFAR-10, SVHN).** UDA supera consistentemente a **VAT** y **MixMatch**. Como solo cambia el proceso de ruido frente a VAT, demuestra que las augmentaciones realistas baten al ruido adversarial. Con **4k labels en CIFAR-10**: 4.32% de error (WRN-28-2), bajando a 2.7% (PyramidNet+ShakeDrop) —**igualando** al supervisado con 50.000 ejemplos. En **SVHN con 1k labels**: 2.23%, igualando al supervisado con 73.257 ejemplos.
- **Texto + BERT.** El resultado emblemático: **IMDb con 20 labels**. BERT-large solo da 11.72% de error; **con UDA, 4.78%**; con BERT fine-tuneado en dominio, UDA baja de 6.50 a **4.20%** —superando la SOTA pre-BERT entrenada con las 25.000 etiquetas completas (4.32%). Lección: **UDA es complementario al transfer learning**, no redundante.
- **Escalabilidad en ImageNet.** Con ResNet-50 y **10% de etiquetas**, UDA sube top-1 de 58.84 a **68.78**. Con **100% de etiquetas** más 1.3M imágenes externas filtradas, sube de 78.43 a **79.05**. UDA no solo funciona con pocos datos: **escala al régimen de muchos datos** y aprovecha datos fuera de dominio.

## Limitaciones reconocidas

- **Brecha en multiclase difícil.** En sentimiento de 5 categorías persiste una diferencia clara respecto al supervisado completo —problema abierto.
- **Dependencia de augmentaciones de calidad.** Toda la ventaja descansa en disponer de una augmentación fuerte, válida y diversa para el dominio. En dominios sin una buena augmentación conocida (datos tabulares, señales muy específicas), la receta pierde su palanca.
- **Costo de la back-translation.** Requiere modelos de traducción entrenados; es un preprocesado pesado.
- **Sensibilidad a hiperparámetros.** TSA (qué cronograma), $\beta$ y $\tau$ requieren ajuste por dataset; sin ellos, el régimen de 20 labels no converge bien.

## Impacto

UDA, junto con MixMatch (trabajo paralelo de 2019), marca el inicio de la **era moderna de la regularización por consistencia** y pavimenta el camino directo a **FixMatch** (Sohn et al. 2020), que sintetiza ambos en una receta minimalista: augmentación débil → pseudo-etiqueta con umbral → consistencia con augmentación fuerte. La idea de que **la calidad de la augmentación es el cuello de botella, no el algoritmo**, reorientó el SSL de la generación 2020.

El otro legado es la **bisagra entre supervisado, semi-supervisado y autosupervisado**: las invariancias descubiertas para el supervisado (RandAugment) son exactamente las que se necesitan para explotar datos sin etiquetar —la misma intuición de "predecir igual para un dato y su transformación" que sostiene el aprendizaje contrastivo de [SimCLR](/papers/simclr-chen-2020). En salud y matching de registros, donde las etiquetas curadas son caras pero los datos crudos abundan, esta es exactamente la palanca relevante.

## Por qué importa para el Laboratorio 28

La [Clase 28](/clases/clase-28) ("Aprendizaje Autosupervisado") dedica su sección sobre "autosupervisión para potenciar el aprendizaje supervisado" íntegramente a UDA. El hilo conceptual es el de este paper: la intuición rectora ("el modelo debe predecir de forma similar para un dato y su versión transformada"), la **divergencia KL** como objetivo de consistencia, y la tabla de ImageNet al 10% como ilustración de que la consistencia rinde con etiquetas escasas.

El **Laboratorio 28** es la implementación práctica de esta receta. El estudiante experimenta con: el ensamblaje de la **pérdida combinada** (cross-entropy supervisada + KL de consistencia ponderada por $\lambda$), la generación de **augmentaciones** (RandAugment en visión, back-translation en texto) y las técnicas auxiliares —**TSA**, **confidence masking**, **sharpening**— que vuelven viable el régimen de poquísimas etiquetas. Entender por qué cada pieza está donde está —por qué el stop-gradient, por qué afilar, por qué enmascarar por confianza, por qué TSA con 20 labels— es la diferencia entre ejecutar el notebook y comprender el mecanismo. Este paper es **la base del lab-28**.

## Notas y enlaces

- Fundamento transversal: [/fundamentos/aprendizaje-autosupervisado](/fundamentos/aprendizaje-autosupervisado)
- Clase: [/clases/clase-28](/clases/clase-28)
- Laboratorio: [/laboratorios/lab-28](/laboratorios/lab-28)
- Paper hermano (aprendizaje contrastivo): [/papers/simclr-chen-2020](/papers/simclr-chen-2020)
- Código: [github.com/google-research/uda](https://github.com/google-research/uda)
- Preprint arXiv:1904.12848 (abril 2019); versión final NeurIPS 2020.
