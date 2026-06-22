# Masked Autoencoders Are Scalable Vision Learners — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Masked Autoencoders Are Scalable Vision Learners*.
- **Autores:** Kaiming He, Xinlei Chen (contribución técnica equivalente, marcados con ∗), Saining Xie, Yanghao Li, Piotr Dollár y Ross Girshick. Kaiming He figura además como *project lead* (†).
- **Afiliación:** Facebook AI Research (FAIR).
- **Venue:** CVPR 2022 (publicado como *highlight* / oral). **Preprint:** arXiv:2111.06377v3 (19 dic 2021), [arxiv.org/abs/2111.06377](https://arxiv.org/abs/2111.06377).

La tesis del paper cabe en una frase, que es además el título: **los autoencoders enmascarados (MAE) son aprendices visuales escalables y autosupervisados**. La receta es deliberadamente simple — *enmascarar parches aleatorios de la imagen de entrada y reconstruir los píxeles faltantes* — y descansa sobre dos decisiones de diseño acopladas. Primera: una **arquitectura encoder-decoder asimétrica**, donde el encoder opera *solo* sobre el subconjunto visible de parches (sin tokens de máscara) y un decoder ligero reconstruye la imagen original a partir de la representación latente más los tokens de máscara. Segunda: **enmascarar una proporción muy alta de la imagen, por ejemplo 75%**, lo que produce una tarea autosupervisada no trivial y significativa. Acoplar ambas ideas permite entrenar modelos grandes de manera eficiente y efectiva: acelera el entrenamiento 3× o más, reduce la memoria, y mejora la precisión. El resultado emblemático: un **ViT-Huge vanilla alcanza 87.8% de top-1 en ImageNet-1K**, la mejor precisión entre métodos que usan *solo* datos de ImageNet-1K, y el preentrenamiento MAE transfiere mejor que el preentrenamiento supervisado en detección, segmentación de instancias y segmentación semántica.

Para la Clase 28 (Aprendizaje Autosupervisado) este paper es la pieza que cierra el arco que abre con los **autoencoders**: MAE *revive* la idea del *denoising autoencoder* (Vincent et al., 2008/2010) y la lleva al régimen moderno de los Transformers de visión, demostrando que el SSL generativo —reconstruir lo que falta— puede competir con (y superar a) el SSL contrastivo que dominaba la visión hacia 2020. Es el equivalente visual de lo que BERT fue para NLP, y por eso aparece como slide propio en la clase.

## 2. Contexto histórico: el masked language modeling de BERT y por qué no cruzaba a visión

El punto de partida conceptual es explícito en la introducción: el apetito de datos del deep learning se resolvió en NLP mediante **preentrenamiento autosupervisado**. Las soluciones —modelado de lenguaje autorregresivo en GPT (Radford et al.) y *masked autoencoding* en BERT (Devlin et al., 2019)— son conceptualmente simples: *remueven una porción de los datos y aprenden a predecir el contenido removido*. Esa receta habilitó modelos de lenguaje generalizables con más de cien mil millones de parámetros (GPT-3, Brown et al., 2020).

La idea del masked autoencoder, como forma más general de *denoising autoencoder*, es natural y aplicable también en visión — de hecho trabajos cercanos en visión (los *stacked denoising autoencoders* de Vincent y el *Context Encoder* de Pathak) *precedieron* a BERT. Sin embargo, **pese al enorme interés tras el éxito de BERT, el progreso de los métodos de autoencoding en visión quedó rezagado respecto a NLP**. El paper se hace la pregunta central — *¿qué hace que el masked autoencoding sea diferente entre visión y lenguaje?* — y la responde desde tres perspectivas:

1. **Las arquitecturas eran distintas.** Durante la última década la visión estuvo dominada por **redes convolucionales**, que operan sobre grillas regulares. No es directo integrar "indicadores" como *tokens de máscara* o *embeddings posicionales* dentro de una convolución: el kernel convolucional desliza sobre un mapa denso y no sabe cómo representar "aquí falta un parche". Esa brecha arquitectónica es justamente lo que hacía difícil manejar parches ocultos en CNNs. **La llegada del Vision Transformer (ViT; Dosovitskiy et al., 2021) elimina ese obstáculo**: al tratar la imagen como una secuencia de parches-tokens, agregar tokens de máscara y embeddings posicionales se vuelve tan natural como en el Transformer de texto. ViT es, literalmente, el habilitador que permite trasladar la receta de BERT a visión.

2. **La densidad de información es distinta.** El lenguaje es una señal generada por humanos, altamente semántica y densa en información: predecir unas pocas palabras faltantes por oración ya induce comprensión lingüística sofisticada. Las imágenes, en cambio, son señales naturales con **fuerte redundancia espacial** — un parche faltante se recupera de los vecinos con poco entendimiento de alto nivel sobre partes, objetos y escenas. La consecuencia de diseño es la idea central de MAE: para superar esa redundancia y forzar el aprendizaje de rasgos útiles, hay que **enmascarar una porción muy alta de parches aleatorios**, lo que reduce drásticamente la redundancia y crea una tarea que exige comprensión holística más allá de las estadísticas de bajo nivel.

3. **El decoder juega un rol distinto.** En visión el decoder reconstruye píxeles, cuya salida tiene un nivel semántico *más bajo* que las tareas de reconocimiento; en lenguaje el decoder predice palabras faltantes, ricas en semántica. En BERT el decoder puede ser trivial (un MLP), pero el paper encuentra que **para imágenes el diseño del decoder determina el nivel semántico de las representaciones latentes aprendidas**.

## 3. Contribución central

MAE materializa esas tres observaciones en un método simple, efectivo y escalable. Sus contribuciones acopladas son:

1. **Tasa de enmascaramiento muy alta (≈75%) con muestreo aleatorio uniforme.** Frente al 15% típico de BERT y al 20–50% de trabajos previos de masked image encoding (iGPT, ViT, BEiT), MAE oculta tres cuartas partes de los parches. Esto elimina la redundancia, impide resolver la tarea por simple extrapolación de vecinos y obliga al modelo a razonar sobre la *gestalt* de objetos y escenas (Figuras 2–4). La distribución uniforme evita un sesgo hacia el centro de la imagen.

2. **Encoder asimétrico que procesa solo los parches visibles.** El encoder es un ViT estándar pero aplicado *únicamente* al subconjunto visible (≈25%); los parches enmascarados se *eliminan*, no se reemplazan por tokens de máscara. Esto permite entrenar encoders muy grandes con una fracción del cómputo y la memoria, porque el costo de la autoatención es cuadrático en el número de tokens: pasar de 100% a 25% de tokens es un ahorro enorme.

3. **Decoder ligero que reconstruye en píxeles.** Los tokens de máscara —vectores compartidos y aprendidos que marcan dónde falta un parche— se introducen *después* del encoder, y el conjunto completo (parches codificados + tokens de máscara, todos con embeddings posicionales) lo procesa un decoder pequeño y poco profundo. El decoder solo se usa en el preentrenamiento y luego se descarta; para reconocimiento solo se conserva el encoder, aplicado a imágenes íntegras.

La combinación produce un escenario *win-win*: la alta tasa de máscara optimiza la precisión *y* permite que el encoder procese poco, reduciendo el tiempo de preentrenamiento 3× o más. Un cuarto rasgo distintivo: **MAE funciona bien sin augmentaciones intensas** — el rol regularizador que en el aprendizaje contrastivo cumplen las augmentaciones fuertes, aquí lo cumple el enmascaramiento aleatorio, que genera una vista nueva por iteración.

## 4. Método: arquitectura, masking y target en píxeles

**Masking.** Siguiendo a ViT, la imagen se divide en parches regulares no solapados. Se muestrea un subconjunto sin reemplazo bajo distribución uniforme ("random sampling") y se eliminan los demás. La entrada altamente dispersa que resulta es lo que crea la oportunidad de un encoder eficiente.

**Encoder.** Un ViT que embebe los parches visibles mediante proyección lineal con embeddings posicionales sumados, y los procesa con bloques Transformer. Crucialmente **no usa tokens de máscara**: solo ve parches reales, lo que evita una brecha entre preentrenamiento (entrada con muchos tokens de máscara) y despliegue (imágenes íntegras sin máscara).

**Decoder.** Recibe el conjunto completo: parches codificados visibles + tokens de máscara, todos con embeddings posicionales (sin ellos los tokens de máscara no tendrían información de ubicación). Es deliberadamente ligero — el decoder por defecto tiene 8 bloques y ancho 512, con <10% del cómputo por token respecto al encoder. Su arquitectura es independiente del encoder porque solo vive durante el preentrenamiento.

**Target de reconstrucción.** MAE predice los valores de píxel de cada parche enmascarado. La última capa del decoder es una proyección lineal con tantos canales como píxeles tiene un parche, y la pérdida es el **error cuadrático medio (MSE) computado solo sobre los parches enmascarados** (como en BERT; computar la pérdida sobre todos los píxeles baja ~0.5% la precisión). Una variante importante usa **píxeles normalizados por parche** como target (se calcula media y desviación de cada parche y se normaliza): esto realza el contraste local, enfatiza componentes de alta frecuencia y mejora la calidad de la representación (85.4% vs 84.9% en fine-tuning).

**Implementación simple.** No requiere operaciones dispersas especializadas: se generan tokens para todos los parches, se *barajan* aleatoriamente, se descarta la cola según la tasa de máscara (equivalente a muestrear sin reemplazo), se codifica el subconjunto, se anexan tokens de máscara, se *desbaraja* para alinear con los targets y se aplica el decoder. El overhead de barajar/desbarajar es despreciable.

## 5. Experimentos: ImageNet, escalabilidad y transferencia

El preentrenamiento es autosupervisado sobre ImageNet-1K (IN1K); luego se evalúa con *fine-tuning* extremo-a-extremo o *linear probing*.

**Tasa de enmascaramiento (Figura 5).** Las tasas óptimas son sorprendentemente altas: 75% es buena tanto para linear probing como para fine-tuning. Para linear probing la precisión sube de forma sostenida hasta el punto dulce (brecha de ~20%: 54.6% al 10% de máscara vs 73.5% al 75%); para fine-tuning un rango amplio (40–80%) funciona bien, y todo supera el entrenamiento desde cero (82.5%).

**Token de máscara y eficiencia (Tablas 1c y 2).** Meter tokens de máscara en el encoder *empeora* el linear probing en 14 puntos (por la brecha preentrenamiento/despliegue) y multiplica el cómputo. Sacarlos reduce los FLOPs 3.3× y da un *speedup* de pared de 2.8×, que sube a 3.5–4.1× con decoder de 1 bloque y/o encoder más grande (ViT-H). Entrenar ViT-L con tokens de máscara toma 42.4 h vs 15.4 h sin ellos (mismas 800 épocas, 128 núcleos TPU-v3).

**Decoder (Tablas 1a, 1b).** Un decoder profundo ayuda al linear probing (hasta +8%) porque las últimas capas se especializan en reconstrucción y dejan la latente en un nivel más abstracto; para fine-tuning el decoder es poco influyente — incluso un decoder de un solo bloque rinde 84.8%.

**Augmentación (Tabla 1e).** MAE rinde bien con solo *cropping* (fijo o aleatorio) e incluso de forma decente *sin ninguna augmentación* (solo center-crop). Esto contrasta dramáticamente con el contrastivo, que depende fuertemente de la augmentación (BYOL y SimCLR pierden 13% y 28% solo con cropping). El rol regularizador lo asume el enmascaramiento aleatorio.

**Estrategia de muestreo (Tabla 1f, Figura 6).** El muestreo aleatorio supera al *block-wise* (que degrada al 75%) y al *grid-wise* (más fácil, reconstrucción más nítida pero peor representación).

**Schedule (Figura 7).** La precisión mejora de forma sostenida con entrenamientos largos; el linear probing no satura ni a 1600 épocas — a diferencia de MoCo v3, que satura a 300 épocas. El encoder MAE solo ve 25% de parches por época, frente al 200%+ (dos o más vistas) del contrastivo.

**Resultados principales (Tabla 3, Figura 8).** Con preentrenamiento solo en IN1K: ViT-B 83.6%, ViT-L 85.9%, ViT-H 86.9%, y ViT-H a 448 px **87.8%** — superando el mejor previo (87.1%) en el competitivo benchmark sin datos externos. MAE escala mejor que el supervisado y sigue una tendencia análoga al preentrenamiento supervisado en JFT-300M de ViT, pero usando ~mil veces menos imágenes. Es más preciso, simple y rápido (3.5× por época) que BEiT, que predice tokens y requiere un dVAE preentrenado en 250M imágenes.

**Fine-tuning parcial (Figura 9).** Ajustar un solo bloque Transformer salta de 73.5% (linear probing) a 81.0%; ajustar 4–6 bloques se acerca al fine-tuning completo. Las representaciones MAE son menos *linealmente* separables que las de MoCo v3 pero son rasgos no-lineales más fuertes (gap de 2.6% a favor de MAE al ajustar 4 bloques). Esto cuestiona el linear probing como única métrica de calidad.

**Transferencia.** En **COCO** (Mask R-CNN con backbone ViT adaptado a FPN) MAE supera al supervisado en todas las configuraciones: +2.4 APbox con ViT-B (50.3 vs 47.9) y +4.0 con ViT-L (53.3 vs 49.3), igualando o superando a BEiT, y con MoCo v3 apenas a la par del supervisado. En **ADE20K** (UperNet) mejora +3.7 mIoU sobre el supervisado con ViT-L y supera a BEiT. En clasificación (iNaturalist, Places) muestra fuerte escalamiento y supera a métodos preentrenados con miles de millones de imágenes (Places205/365). Los **píxeles normalizados igualan a los tokens dVAE** (Tabla 7, diferencia estadísticamente insignificante en IN1K/COCO/ADE20K): la tokenización no es necesaria para transferir.

**Linear probing y robustez (Apéndices B y C).** Bajo linear probing puro, MAE-ViT-L logra 75.8%, muy por encima de los métodos basados en máscara previos —iGPT-XL (6.8 mil M de parámetros) llega a 72.0% y BEiT-ViT-L cae a 52.1% al reconstruir con su tokenizador— aunque por debajo del contrastivo MoCo v3 (77.6%). En **robustez** (Tabla 13) los mismos modelos fine-tuneados, sin reentrenamiento especializado, exhiben escalamiento fuerte: en IN-Adversarial el ViT-H pasa de 33.1% (supervisado) a 68.2% (MAE), una mejora de 35 puntos; en IN-Rendition e IN-Sketch supera el estado del arte previo por márgenes amplios. La precisión sube de forma consistente con el tamaño del modelo y de la imagen.

**Detalles de implementación que importan (Apéndice A).** El preentrenamiento usa AdamW (lr base 1.5e-4, weight decay 0.05, batch 4096, β₂=0.95, 40 épocas de warmup, cosine decay) y como única augmentación RandomResizedCrop — sin color jittering, drop path ni gradient clip. El fine-tuning usa *layer-wise lr decay* 0.75 (siguiendo BEiT) y regularización fuerte (RandAug, mixup, cutmix, label smoothing). El linear probing, en cambio, requiere una receta opuesta: la regularización es perjudicial, así que se desactivan mixup/cutmix/drop-path y weight decay, y se añade un BatchNorm sin transformación afín para calibrar magnitudes. Un dato revelador del Apéndice A.2: entrenar ViT-L/H supervisado *desde cero* en IN1K es inestable (pérdidas NaN frecuentes) y exige una receta cuidadosa con EMA; el preentrenamiento MAE elude esa fragilidad y solo necesita 50 épocas de fine-tuning (vs 200 desde cero) para superarlo.

## 6. Limitaciones reconocidas y matices

- **El target son píxeles, no entidades semánticas.** El paper es honesto: las imágenes son "luz registrada" sin descomposición semántica análoga a las palabras; MAE reconstruye píxeles, que no son entidades semánticas. Que aun así infiera reconstrucciones holísticas y plausibles (Figura 4) lo atribuyen a una rica representación oculta — una hipótesis, no una prueba.
- **Menor separabilidad lineal.** Bajo el protocolo de linear probing, MAE (75.8% con ViT-L) supera a los métodos basados en máscara previos pero **queda por debajo del contrastivo** (MoCo v3, 77.6%). Su fortaleza aparece con fine-tuning o ajuste parcial, no con un clasificador lineal sobre rasgos congelados.
- **Impacto social.** El modelo predice contenido según estadísticas aprendidas del dataset, por lo que **reflejará sus sesgos** y puede generar contenido inexistente — algo a considerar al construir aplicaciones generativas sobre él.

## 7. Impacto: revivir el denoising autoencoder y fundar el SSL generativo en visión

La conclusión del paper es programática: *los algoritmos simples que escalan bien son el núcleo del deep learning*. En NLP, métodos autosupervisados simples habilitaron el escalamiento exponencial de modelos; MAE muestra que un **autoencoder —método autosupervisado simple, emparentado con las técnicas de NLP— ofrece beneficios escalables también en visión**, sugiriendo que el SSL visual puede recorrer una trayectoria análoga a la de NLP.

El aporte histórico tiene tres capas. Primero, **revive los denoising autoencoders**: la idea de corromper la entrada y reconstruir el original, latente desde Vincent et al. (2008), encuentra en el ViT el sustrato que le faltaba y se vuelve estado del arte. Segundo, **vuelve competitivo el SSL generativo frente al contrastivo**: hasta 2020 la visión autosupervisada estaba dominada por métodos contrastivos (SimCLR, MoCo, BYOL) que dependían de augmentaciones cuidadosas y pares positivos/negativos; MAE demuestra que reconstruir-lo-que-falta, sin augmentaciones intensas ni vistas múltiples, transfiere mejor a tareas densas. Tercero, **se convierte en base del SSL moderno en visión**: MAE inspiró toda una familia de masked image modeling (SimMIM, data2vec, MaskFeat, las variantes de video y multimodales) y consolidó el preentrenamiento generativo como camino estándar para escalar ViTs.

## 8. Conexión con la Clase 28 (Aprendizaje Autosupervisado)

La Clase 28 dedica un slide a MAE (He et al., 2022) que resume con precisión la tesis del paper, y conviene mapear cada afirmación del slide a su evidencia en el texto:

- **"Reviviendo ideas de Denoising Autoencoder."** El paper lo declara literalmente: MAE es "una forma de denoising autoencoding" emparentada con los *stacked denoising autoencoders* de Vincent. La clase usa esto para cerrar el arco que abre con los **autoencoders** al inicio de la sesión: MAE es el autoencoder llevado a su forma autosupervisada moderna, donde la "corrupción" es el enmascaramiento masivo de parches.

- **"Tarea más adecuada a Transformer."** El traslado de la receta de BERT a visión solo fue posible gracias a ViT (ver [ViT, Dosovitskiy et al. 2021](/papers/vit-dosovitskiy-2021)), que trata la imagen como secuencia de parches-token y permite insertar tokens de máscara y embeddings posicionales de forma natural — algo que el slide subraya y que el paper identifica como la primera de las tres diferencias visión/lenguaje.

- **"No trivial lidiar con parches ocultos en CNN."** Es la brecha arquitectónica de la §2: las convoluciones operan sobre grillas densas y no integran fácilmente indicadores de "parche faltante". Por eso los intentos previos en CNN (Context Encoder) no alcanzaban la simplicidad ni la escalabilidad de MAE.

- **"Ahorra cómputo procesando solo lo visible."** El encoder asimétrico que solo ve el ~25% visible reduce FLOPs 3.3× y da speedups de pared de 2.8–4.1× (Tablas 1c y 2), aprovechando la complejidad cuadrática de la autoatención.

- **"Funciona sin augmentaciones tan intensas."** La Tabla 1e muestra que MAE rinde decentemente incluso sin augmentación, porque el enmascaramiento aleatorio genera una vista nueva por iteración — el contraste exacto con el aprendizaje contrastivo que la clase quiere destacar.

En el mapa del curso, MAE conecta hacia atrás con los **autoencoders** (la primera idea de la clase, de la que MAE es la encarnación autosupervisada y escalable) y con **ViT** ([/papers/vit-dosovitskiy-2021](/papers/vit-dosovitskiy-2021)), su habilitador arquitectónico. Para profundizar en el marco conceptual del preentrenamiento sin etiquetas y su lugar entre los paradigmas de SSL (generativo vs contrastivo), ver el fundamento de [aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado) y el hub de la sesión en [Clase 28](/clases/clase-28).
