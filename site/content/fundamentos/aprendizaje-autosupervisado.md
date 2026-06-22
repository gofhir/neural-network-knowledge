---
title: "Aprendizaje Autosupervisado (SSL)"
weight: 103
math: true
---

El **aprendizaje autosupervisado** (self-supervised learning, SSL) es el paradigma en el que el modelo aprende representaciones útiles **sin etiquetas humanas**, porque el *output* objetivo se genera automáticamente a partir del propio dato. La idea es desconcertantemente simple y profundamente fértil: si escondemos una parte de un dato y le pedimos al modelo que la reconstruya, o si le aplicamos una transformación conocida y le pedimos que la identifique, hemos fabricado una tarea supervisada **cuya etiqueta sale gratis de la estructura de los datos**. No hay anotador humano: la "respuesta correcta" la conoce el sistema porque él mismo la ocultó. Resolver bien esa tarea artificial —llamada *pretext task*— solo es posible si el modelo aprende algo genuino sobre el contenido de la señal, y esa representación aprendida es lo que luego transferimos a las tareas que de verdad nos importan. Este fundamento recorre el qué, el porqué, la taxonomía y la evaluación del SSL, el núcleo conceptual de la [Clase 28](/clases/clase-28).

---

## 1. Qué es y por qué importa

En el aprendizaje **supervisado** clásico, cada ejemplo $x$ viene acompañado de una etiqueta $y$ producida por un humano, y entrenamos $f_\theta$ para predecir $y$ desde $x$. En el aprendizaje **autosupervisado**, no hay $y$ humano: definimos una función automática $y = T(x)$ que genera el objetivo a partir del propio dato. Por ejemplo, tomamos una imagen, le quitamos el centro ($x' = $ imagen agujereada) y el objetivo es el parche removido ($y = $ centro original); o rotamos la imagen un ángulo aleatorio y el objetivo es el ángulo. El modelo entrena con un objetivo supervisado ordinario, pero **la supervisión es sintética y autogenerada**.

La motivación es triple, y nace de los problemas reales del etiquetado:

- **Costo.** Etiquetar a mano millones de ejemplos es caro y lento. ImageNet costó años-persona de anotación; en dominios especializados el costo se dispara porque la etiqueta requiere un experto (ver §7).
- **Consistencia.** Las etiquetas humanas son ruidosas e inconsistentes: distintos anotadores discrepan, los criterios cambian, las taxonomías envejecen.
- **Detalle.** Una etiqueta de clase ("perro") comprime una imagen riquísima en un único símbolo, descartando casi toda la información. El SSL, al pedir reconstruir la señal o sus relaciones internas, **obliga a usar mucha más estructura** que un puñado de clases.

El argumento filosófico que recorre la Clase 28 es que **los humanos y los animales aprenden la mayor parte de lo que saben sin etiquetas explícitas**. Un niño no necesita millones de pares (imagen, clase) para entender que una silla es una silla: aprende observando el mundo, prediciendo lo que viene, llenando lo que falta. Geoffrey Hinton lo planteaba como un problema de capacidad: los modelos modernos tienen **más parámetros que datos etiquetados disponibles**, así que la única forma de alimentar ese apetito es extraer señal de los datos no etiquetados, que son virtualmente infinitos. Yann LeCun lo popularizó con su metáfora del pastel: *"si la inteligencia es un pastel, el aprendizaje no supervisado/autosupervisado es el bizcocho, el aprendizaje supervisado es el glaseado y el aprendizaje por refuerzo es la cereza"*. La mayor parte del aprendizaje —el grueso del pastel— debería provenir de observar el mundo sin que nadie nos diga la respuesta.

Conviene precisar la terminología, porque a veces se confunde. El SSL es un caso particular de **aprendizaje no supervisado** (no usa etiquetas humanas), pero con un giro: en vez de buscar estructura difusa (clustering, reducción de dimensión), define una **tarea supervisada artificial** y la entrena con las mismas herramientas que el aprendizaje supervisado (una pérdida, un objetivo claro, descenso de gradiente). De ahí el "auto": el sistema **se supervisa a sí mismo**. La diferencia con el aprendizaje **semi-supervisado** (§6) es que el semi-supervisado sí usa unas pocas etiquetas humanas y las complementa con datos sin etiqueta, mientras que el SSL puro no usa ninguna en la fase de preentrenamiento. En la práctica, el flujo dominante es **SSL para preentrenar + un poco de supervisión para afinar**, combinando lo mejor de ambos mundos.

Históricamente, el SSL pasó de ser una curiosidad a dominar el campo. En NLP la transición fue temprana y total: word2vec (2013) y luego BERT y GPT (2018-2020) demostraron que preentrenar sin etiquetas sobre corpus gigantes producía representaciones que destrozaban a los modelos entrenados desde cero, y hoy **ningún modelo de lenguaje serio se entrena sin una fase autosupervisada masiva**. En visión el camino fue más lento: las pretext tasks artesanales de 2015-2018 (inpainting, rotación, posicionamiento) cerraban parte de la brecha pero no la igualaban; recién con el contrastivo (SimCLR, MoCo, 2020) y luego con MAE (2022) el SSL visual alcanzó y superó al preentrenamiento supervisado. Esa convergencia de NLP y visión hacia el mismo paradigma —enmascarar y predecir, o contrastar vistas— es lo que hace del SSL el tema unificador de la era moderna del deep learning.

{{< concept-alert type="clave" >}}
La definición operacional del SSL: **el output objetivo se deriva automáticamente del dato de entrada**, sin intervención humana. Escondemos, transformamos o relacionamos partes de la señal y pedimos al modelo predecirlas. Resolver esa tarea artificial *fuerza* a aprender representaciones semánticas transferibles. El SSL no es un truco para ahorrar etiquetas: es la apuesta de que la mayor parte del conocimiento se aprende de la estructura de los datos, no de las etiquetas.
{{< /concept-alert >}}

---

## 2. La taxonomía de pretext tasks

Una *pretext task* (tarea pretexto) es la tarea artificial que diseñamos para que el modelo, al resolverla, aprenda buenas representaciones. La creatividad del campo está casi toda aquí: cómo inventar una tarea cuya solución sea imposible sin entender la señal. Las familias se organizan en cuatro grandes grupos.

| Familia | Pretext task | Ejemplo canónico | Qué fuerza a aprender |
|---|---|---|---|
| **Predicción / generativos** | reconstruir parte oculta de la señal | autoencoders, inpainting, colorización, masked modeling | estructura semántica para "llenar" lo ausente |
| **Por transformaciones** | identificar una transformación aplicada | rotación, posición relativa, orden temporal | orientación canónica, partes, configuración espacial/temporal |
| **Contrastivos** | acercar vistas del mismo dato, alejar las de otros | instance discrimination, SimCLR, MoCo | invarianza a augmentaciones, similitud semántica |
| **Multimodales** | alinear señales de dos modalidades | audio-visual, imagen-texto | correspondencia entre modalidades alineadas por la naturaleza |

### (a) Predicción / generativos

Aquí el objetivo es **reconstruir** parte del dato. El antecedente conceptual son los **autoencoders** (§4). El **inpainting** —rellenar una región removida— es el ejemplo canónico de auto-predicción en imágenes: los [Context Encoders (Pathak et al. 2016)](/papers/context-encoders-pathak-2016) entrenan una CNN encoder-decoder para regresar a los píxeles de un agujero grande, lo que obliga a "conjurar una ventana entera de la nada" y, por tanto, a entender la escena. La **colorización** —[Colorful Image Colorization (Zhang et al. 2016)](/papers/colorization-zhang-2016)— predice los canales de color $ab$ a partir del canal de luminancia $L$; al ser el color multimodal (una manzana puede ser roja o verde), el paper lo formula como **clasificación sobre bins de color** en vez de regresión, evitando el colapso al promedio gris. El **masked modeling** es la encarnación moderna: enmascarar una porción de la entrada y predecirla. En texto, es el **masked language modeling** de BERT (predecir palabras ocultas) y el **next-token prediction** de GPT (predecir la siguiente palabra) — ver [Clase 20](/clases/clase-20). En visión, su equivalente directo es el [Masked Autoencoder (He et al. 2022)](/papers/mae-he-2022), que enmascara un 75% de los parches y reconstruye los píxeles faltantes (§5).

### (b) Por transformaciones

En vez de reconstruir, el modelo **identifica una transformación conocida** aplicada al dato. La **rotación** —[RotNet (Gidaris et al. 2018)](/papers/rotnet-gidaris-2018)— aplica una de cuatro rotaciones $\{0°, 90°, 180°, 270°\}$ y pide clasificarla: para acertar, la red debe reconocer los objetos y su orientación canónica ("erguida"), porque no hay atajo de bajo nivel que delate el ángulo. El **posicionamiento relativo** —[Context Prediction (Doersch et al. 2015)](/papers/context-prediction-doersch-2015)— toma un parche central y un vecino, y pide clasificar cuál de las 8 posiciones relativas ocupa el vecino; es el "word2vec de los píxeles" y el origen de toda la familia espacial (jigsaw puzzles). El **orden temporal** —[Shuffle and Learn (Misra et al. 2016)](/papers/shuffle-and-learn-misra-2016)— toma fotogramas de un video y pide verificar si están en orden temporal correcto o desordenados, explotando la flecha del tiempo como señal gratuita y aprendiendo, entre otras cosas, pose humana.

### (c) Contrastivos

La familia que **eclipsó a las pretext tasks artesanales** hacia 2020. La idea: la representación de un dato debe parecerse a la de **otra vista del mismo dato** (positivo) y diferenciarse de la de **datos distintos** (negativos). Su formulación temprana está en [Invariant and Spreading Instance Feature (Ye et al. 2019)](/papers/invariant-spreading-ye-2019), que enuncia el ADN del campo: cada imagen es su propia clase (*instance discrimination*), su versión augmentada es el positivo y las otras del batch son negativos. El objetivo se concreta en la pérdida **InfoNCE**, que para un par positivo $(i, j)$ entre $2N$ vistas del batch es:

$$
\ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]}\exp(\text{sim}(z_i, z_k)/\tau)},
$$

donde $\text{sim}(\cdot,\cdot)$ es la similitud coseno y $\tau$ la temperatura. Es, en esencia, un clasificador que debe elegir el positivo verdadero entre todos los negativos del batch. [SimCLR (Chen et al. 2020)](/papers/simclr-chen-2020) lo escaló a ImageNet componiendo augmentaciones fuertes, una cabeza de proyección y batches enormes, igualando a un ResNet-50 supervisado (76,5% top-1). [MoCo (He et al. 2019)](/papers/moco-he-2019) resolvió la dependencia del batch gigante con una **cola FIFO** de negativos (desacoplada del tamaño de batch) y un **momentum encoder** para las claves, manteniendo un diccionario a la vez grande y consistente. El mecanismo detallado (InfoNCE, temperatura, augmentaciones, negativos duros) está en el [fundamento de aprendizaje contrastivo](/fundamentos/aprendizaje-contrastivo).

### (d) Multimodales

La señal de supervisión proviene de la **correspondencia entre dos modalidades** que la naturaleza ya alinea. Es quizás la idea más fértil del SSL, porque la segunda modalidad aporta una semántica que ninguna augmentación intra-modal puede inventar. La correspondencia **audio-visual** —[Look, Listen and Learn (Arandjelović y Zisserman 2017)](/papers/look-listen-learn-arandjelovic-2017)— entrena dos redes (visión y audio) **desde cero** para decidir si un fotograma y un clip de audio corresponden al mismo momento; la co-ocurrencia (el piano que suena cuando se mueven los dedos) tiene una causa común, y entrenar ambas redes juntas —en vez de fijar una como maestro— hace emerger buenas features en las dos modalidades (estado del arte en clasificación de sonido en ESC-50 y DCASE). La correspondencia **imagen-texto** es hoy la más influyente: [ConVIRT (Zhang et al. 2020)](/papers/convirt-zhang-2020) alinea imágenes médicas con sus reportes mediante contraste bidireccional InfoNCE entre dos torres, y [CLIP (Radford et al. 2021)](/papers/clip-radford-2021) —descrito por sus propios autores como "una versión simplificada de ConVIRT"— lo escaló a 400M de pares web, dando lugar a la clasificación *zero-shot* por texto. [VisualBERT (Li et al. 2019)](/papers/visualbert-li-2019) lleva el masked language modeling al territorio multimodal: en un único stack Transformer mete las regiones de la imagen como tokens adicionales y predice palabras enmascaradas usando también el contexto visual, un puente limpio entre la autosupervisión en lenguaje (clase 20) y la multimodal (clase 23). El patrón común de la familia multimodal —dos encoders, un espacio compartido, una pérdida contrastiva o de enmascaramiento— es hoy la receta base de los modelos de visión-lenguaje.

---

## 3. Cómo se evalúa el SSL

Como el SSL no produce directamente predicciones de la tarea final, su calidad se mide por **lo bien que transfiere la representación aprendida**. Hay cuatro protocolos estándar, de menos a más intervención:

- **Linear probing.** Se **congela** el encoder preentrenado y se entrena solo una capa lineal encima sobre la tarea destino. Mide la calidad *pura* de los features: si una representación es linealmente separable por clases, es porque ya organizó la semántica. Es exigente y muy usado, pero no lo es todo (MAE, por ejemplo, brilla bajo fine-tuning aunque su linear probing sea modesto, porque sus features son fuertes pero no-lineales).
- **Fine-tuning.** Se **descongela** todo el modelo y se afina con las etiquetas de la tarea destino. Refleja el uso práctico; casi siempre supera al linear probing.
- **Transfer learning.** Se transfiere el encoder a otra tarea/dominio distinto del de preentrenamiento (clasificación → detección, segmentación), midiendo la generalidad de la representación. Es el [transfer learning](/fundamentos/transfer-learning) de toda la vida, con un preentrenamiento sin etiquetas.
- **Semi-supervisado (few labels).** Se afina el encoder con **muy pocas etiquetas** (1%, 10%). Aquí el SSL muestra su valor más dramático: si con el 1% de las etiquetas iguala a un modelo supervisado con el 100%, el ahorro de anotación es de dos órdenes de magnitud.

La tabla histórica que la Clase 28 usa para mostrar que **el pretexto funciona** viene de Context Encoders, y reporta el fine-tuning a clasificación en **PASCAL VOC** según el preentrenamiento del encoder (AlexNet):

| Preentrenamiento | Supervisión | VOC clasificación |
|---|---|---|
| Inicialización aleatoria | ninguna | 53,3 |
| Autoencoder | ninguna (reconstruye entrada) | 53,8 |
| Inpainting (Context Encoder) | ninguna (predice agujero) | 56,5 |
| Posicionamiento relativo (Doersch) | ninguna (posición) | 65,3 |
| Colorización | ninguna (predice color) | 65,9 |
| ImageNet supervisado | 1000 etiquetas de clase | 79,9 |

La lectura es la médula de la clase: cada pretexto **cierra una porción de la brecha** entre el azar (53,3) y la supervisión completa de ImageNet (79,9), **sin usar una sola etiqueta**. El autoencoder simple apenas supera al azar (porque puede copiar píxeles sin entender), mientras que el inpainting, el posicionamiento y la colorización suben sostenidamente a medida que la tarea exige más semántica. Es la evidencia cuantitativa de que un pretexto bien diseñado induce representaciones útiles. (Nota: los números de SSL "puro" de 2016 hoy están ampliamente superados por los métodos contrastivos y por MAE, que llegan a igualar o superar al supervisado.)

---

## 4. Autoencoders: el punto de partida

El **autoencoder** es la pieza fundacional de la familia generativa y el lugar natural para empezar. Su arquitectura es **encoder-decoder**: un encoder $f_\theta$ comprime la entrada $x$ a una representación latente $z = f_\theta(x)$ de dimensión reducida —el **cuello de botella** (bottleneck)—, y un decoder $g_\phi$ intenta reconstruir la entrada original $\hat{x} = g_\phi(z)$. Se entrena minimizando el error de reconstrucción:

$$
\mathcal{L}(x) = \lVert x - g_\phi(f_\theta(x)) \rVert_2^2 .
$$

El cuello de botella es lo que impide la solución trivial: si $z$ tuviera la misma dimensión que $x$, el modelo podría aprender la identidad y copiar la entrada sin entender nada. Al forzar $\dim(z) \ll \dim(x)$, la red debe **comprimir** la información, reteniendo lo esencial. El problema, central en la Clase 28, es que un autoencoder ordinario "probablemente solo comprime el contenido sin aprender una representación semánticamente significativa": la tarea de copiar no exige semántica, y por eso su transferencia es pobre (53,8 en la tabla, casi como el azar).

Hay además un problema con la **pérdida**. Las pérdidas $L_2$ (cuadrática) y $L_1$ (absoluta) son **conservadoras**: cuando hay varias reconstrucciones plausibles (el problema es *multimodal*), minimizar el error cuadrático medio empuja a **predecir el promedio** de todas ellas, y el promedio de varias imágenes nítidas distintas es una imagen **borrosa**. Por eso el inpainting con solo $L_2$ produce resultados desenfocados, y Context Encoders añade una **pérdida adversarial** (GAN) que escoge un modo nítido en lugar del promedio. La lección —"predecir la media segura borronea"— reaparece en todo el modelado generativo y motiva muchas decisiones de diseño en SSL.

El **denoising autoencoder** (Vincent et al. 2008) es el puente al SSL moderno: corrompe la entrada (con ruido) y pide reconstruir el original limpio. Si llevamos esa corrupción al extremo espacial —un agujero grande— obtenemos el Context Encoder; si la llevamos a enmascarar el 75% de los parches, obtenemos el MAE. El masked modeling es, en esencia, **un denoising autoencoder cuya corrupción es un enmascaramiento masivo**.

---

## 5. SSL generativo (masked) vs. contrastivo

Hacia 2020 el SSL en visión se había bifurcado en dos grandes familias, con trade-offs claros.

El **SSL generativo (masked)** reconstruye lo que falta. Su exponente moderno es el [Masked Autoencoder (MAE)](/papers/mae-he-2022), que materializa tres decisiones acopladas: (1) **enmascarar muy alto** (≈75% de los parches), porque las imágenes son espacialmente redundantes y una máscara pequeña se resuelve copiando vecinos sin entender nada; (2) un **encoder asimétrico** que procesa solo los parches visibles (≈25%), reduciendo el cómputo 3× o más por la complejidad cuadrática de la atención; y (3) un **decoder ligero** que reconstruye píxeles y se descarta tras el preentrenamiento. MAE *revive* el denoising autoencoder llevándolo al Vision Transformer ([ViT](/clases/clase-23) lo hizo posible al tratar la imagen como secuencia de tokens-parche), y demuestra que el SSL generativo puede igualar o superar al contrastivo: un ViT-Huge alcanza **87,8% top-1 en ImageNet-1K**. Rasgo distintivo: **funciona casi sin augmentaciones**, porque el enmascaramiento aleatorio ya genera una vista nueva por iteración.

El **SSL contrastivo** no reconstruye: aprende organizando el espacio de embeddings para que las vistas del mismo dato se acerquen y las de datos distintos se alejen (SimCLR, MoCo). Su fortaleza histórica fue la **separabilidad lineal** de sus features (mejor linear probing que MAE), a costa de una **fuerte dependencia de las augmentaciones** y de batches o memorias grandes de negativos.

Un matiz importante que MAE expone es que **el linear probing no es la única ni la mejor métrica de calidad**. Las representaciones de MAE son menos *linealmente* separables que las contrastivas, pero son rasgos no-lineales más fuertes: basta afinar unos pocos bloques Transformer para que MAE salte de un linear probing modesto a superar al contrastivo bajo fine-tuning, y a transferir mejor a tareas densas como detección (COCO) y segmentación (ADE20K), donde supera al preentrenamiento supervisado. Esto sugiere que dos métodos pueden parecer muy distintos bajo un protocolo y converger bajo otro, y que la elección entre generativo y contrastivo depende tanto de la tarea final como del protocolo de evaluación.

| Aspecto | Generativo (masked) | Contrastivo |
|---|---|---|
| Objetivo | reconstruir lo oculto | acercar/alejar embeddings |
| Augmentaciones | ligeras (basta el masking) | fuertes y críticas |
| Negativos | no necesita | sí (batch grande o cola FIFO) |
| Linear probing | modesto | fuerte |
| Fine-tuning / transfer | muy fuerte | fuerte |
| Ejemplo | MAE, BERT MLM | SimCLR, MoCo |

No son rivales excluyentes sino dos caminos hacia la misma meta —representaciones sin etiquetas—, y la frontera entre ambos se ha vuelto difusa con métodos híbridos y predictivos en el espacio latente.

---

## 6. SSL para potenciar lo supervisado

El SSL no solo sirve para preentrenar de cero: también **amplifica** un poco de supervisión. El **aprendizaje semi-supervisado** combina pocas etiquetas con muchos datos no etiquetados, y la familia dominante hoy es el **consistency training** (regularización por consistencia): el modelo debe **predecir lo mismo** para un ejemplo no etiquetado y para una versión perturbada de él, minimizando una divergencia entre ambas salidas. Es una forma de autosupervisión sobre los datos sin etiqueta: la "etiqueta" es la consistencia con uno mismo bajo perturbación.

El trabajo canónico es [UDA — Unsupervised Data Augmentation (Xie et al. 2019)](/papers/uda-xie-2019), cuya tesis es que **lo que limita el consistency training no es el algoritmo sino la calidad del ruido** inyectado. En vez de ruido gaussiano o dropout —perturbaciones locales y poco realistas—, UDA usa **augmentaciones de última generación específicas del dominio** —RandAugment en imágenes, back-translation en texto, reemplazo por TF-IDF en clasificación de tópicos— y fuerza la consistencia entre el original y su versión aumentada minimizando una divergencia entre ambas distribuciones de salida. El resultado es espectacular: en IMDb, con **solo 20 ejemplos etiquetados**, alcanza 4,20% de error, superando a un modelo supervisado entrenado con 25.000 etiquetas (1.250× más datos). Y a diferencia de muchos métodos que solo funcionan con pocos datos, UDA **escala**: en ImageNet con 10% de etiquetas sube la precisión de 58,8 a 68,8. El detalle de la pérdida y de las técnicas auxiliares (TSA, confidence masking, sharpening) está en el [fundamento de aprendizaje semi-supervisado](/fundamentos/aprendizaje-semi-supervisado). La conexión con el SSL es directa: el consistency training es autosupervisión sobre los ejemplos sin etiqueta, y por eso vive en la misma familia conceptual.

---

## 7. Dónde el SSL importa MÁS

El SSL ahorra costo en cualquier dominio, pero hay escenarios donde **habilita lo que de otro modo sería inviable**. El caso paradigmático es la **medicina**. Etiquetar una radiografía no lo puede hacer un anotador genérico: requiere el tiempo de un **radiólogo certificado**, lo que hace que los datasets médicos sean órdenes de magnitud más pequeños que ImageNet. Peor aún, las imágenes médicas tienen **altísima similitud inter-clase** —una radiografía sana y una con cardiomegalia se parecen mucho más entre sí que un gato y un avión—, por lo que el SSL contrastivo solo-imagen (SimCLR, MoCo) apenas ayuda: las augmentaciones no añaden la señal semántica que falta.

[ConVIRT (Zhang et al. 2020)](/papers/convirt-zhang-2020) resuelve esto con una idea brillante: el segundo extremo del contraste deja de ser otra vista de la imagen y pasa a ser **el texto del reporte que el radiólogo ya escribió gratis** en el flujo clínico. Alineando imagen y reporte con contraste bidireccional InfoNCE, ConVIRT iguala o supera a una inicialización ImageNet usando **solo el 10% de las etiquetas** (y en tres de cuatro tareas, bajo linear probing, basta el **1%**). Para un dominio donde cada etiqueta cuesta el tiempo de un especialista, esto no es un número de benchmark: es la diferencia entre un proyecto viable y uno imposible. La lección, que es también el mensaje de cierre de la clase, generaliza a cualquier dominio sin labels (sensores, datos científicos, registros): **hay que tener creatividad para diseñar el pretexto según el dominio**, reutilizando cualquier estructura o modalidad emparejada que los datos ya traigan gratis (texto descriptivo, metadata, co-ocurrencia temporal o multimodal). El SSL es, en el fondo, el arte de inventar la pregunta cuya respuesta el dato ya conoce.

---

## 8. Conexión con el curso

El SSL no es un tema aislado: es el hilo que une buena parte del curso bajo una sola idea —aprender de la estructura de los datos antes que de las etiquetas—.

1. **Transfer learning.** El SSL es preentrenamiento *sin etiquetas*; lo que se transfiere y cómo (linear probing, fine-tuning) es el clásico [transfer learning](/fundamentos/transfer-learning), solo que el origen del encoder ya no es ImageNet supervisado sino una pretext task.
2. **Aprendizaje contrastivo.** La familia (c) de pretextos tiene su propio fundamento dedicado: el [aprendizaje contrastivo](/fundamentos/aprendizaje-contrastivo) con InfoNCE, temperatura y negativos, base de SimCLR y MoCo.
3. **BERT y GPT (Clase 20).** El masked language modeling (BERT) y el next-token prediction (GPT) son SSL generativo sobre texto; fueron la prueba de que el preentrenamiento autosupervisado escala, y la inspiración directa de MAE en visión. Ver [Clase 20](/clases/clase-20).
4. **ViT y CLIP (Clase 23).** El Vision Transformer hizo posible el masked modeling en imágenes (MAE), y CLIP llevó el SSL multimodal imagen-texto a escala web, heredando la receta de ConVIRT. Ver [Clase 23](/clases/clase-23).
5. **Semi-supervisado.** El SSL potencia la supervisión escasa vía consistency training; UDA es el caso del [aprendizaje semi-supervisado](/fundamentos/aprendizaje-semi-supervisado) y del laboratorio de la clase.

---

## Para profundizar

- [Context Encoders (Pathak et al. 2016)](/papers/context-encoders-pathak-2016) — inpainting como pretexto generativo; el "word2vec de los píxeles" y la pérdida L2 + adversarial.
- [Colorful Image Colorization (Zhang et al. 2016)](/papers/colorization-zhang-2016) — predecir color como clasificación multimodal; cross-channel encoder.
- [Context Prediction (Doersch et al. 2015)](/papers/context-prediction-doersch-2015) — posición relativa de parches; el origen de los pretextos espaciales y del combate a los atajos.
- [RotNet (Gidaris et al. 2018)](/papers/rotnet-gidaris-2018) — predecir la rotación; simplicidad sin atajos de bajo nivel.
- [Shuffle and Learn (Misra et al. 2016)](/papers/shuffle-and-learn-misra-2016) — orden temporal de fotogramas como señal gratuita en video.
- [Invariant and Spreading Instance Feature (Ye et al. 2019)](/papers/invariant-spreading-ye-2019) — el ADN del contrastivo: invarianza positiva + dispersión negativa.
- [SimCLR (Chen et al. 2020)](/papers/simclr-chen-2020) — contrastivo simple a escala ImageNet; el poder de la composición.
- [MoCo (He et al. 2019)](/papers/moco-he-2019) — diccionario grande y consistente con cola FIFO y momentum encoder.
- [MAE (He et al. 2022)](/papers/mae-he-2022) — masked autoencoding moderno; revive el denoising autoencoder sobre ViT.
- [UDA (Xie et al. 2019)](/papers/uda-xie-2019) — consistency training con augmentaciones fuertes; SSL al servicio del semi-supervisado.
- [ConVIRT (Zhang et al. 2020)](/papers/convirt-zhang-2020) — SSL multimodal imagen-texto en medicina; precursor de CLIP.
- [Look, Listen and Learn (Arandjelović y Zisserman 2017)](/papers/look-listen-learn-arandjelovic-2017) — correspondencia audio-visual como pretexto multimodal.

**Fundamentos relacionados:** [Aprendizaje Contrastivo](/fundamentos/aprendizaje-contrastivo) · [Aprendizaje Semi-Supervisado](/fundamentos/aprendizaje-semi-supervisado) · [Transfer Learning](/fundamentos/transfer-learning) · [Metric Learning](/fundamentos/metric-learning) · [Clase 28 — Aprendizaje Autosupervisado](/clases/clase-28)
