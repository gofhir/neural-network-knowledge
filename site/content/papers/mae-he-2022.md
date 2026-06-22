---
title: "MAE: Masked Autoencoders Are Scalable Vision Learners (2022)"
weight: 326
math: true
---

{{< paper-card
    title="Masked Autoencoders Are Scalable Vision Learners"
    authors="Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick"
    year="2022"
    venue="CVPR 2022"
    pdf="/papers/mae-he-2022.pdf"
    arxiv="2111.06377" >}}
Paper de Facebook AI Research (FAIR) que traslada a la visión la receta del *masked language modeling* de BERT: **enmascarar parches aleatorios de una imagen y reconstruir los píxeles faltantes**. Sus dos ideas acopladas son un **encoder asimétrico** que procesa solo el ~25% de parches visibles (gran ahorro de cómputo) y un **decoder ligero** que reconstruye los píxeles a partir de la latente más tokens de máscara. La clave: enmascarar una proporción muy alta (≈75%) convierte la tarea en algo no trivial. El resultado emblemático es un **ViT-Huge vanilla con 87.8% de top-1 en ImageNet-1K**, la mejor precisión usando solo datos de IN1K, y transferencia superior al preentrenamiento supervisado. MAE *revive* los denoising autoencoders y demuestra que el SSL generativo compite con (y supera a) el contrastivo. Es la pieza que cierra el arco de los autoencoders en la [Clase 28](/clases/clase-28).
{{< /paper-card >}}

---

## Contexto: por qué BERT no cruzaba a visión

El punto de partida es explícito: el apetito de datos del deep learning se resolvió en NLP mediante **preentrenamiento autosupervisado**. Tanto el modelado de lenguaje autorregresivo de GPT como el *masked autoencoding* de BERT (Devlin et al., 2019) son conceptualmente simples — *remueven una porción de los datos y aprenden a predecir el contenido removido* — y habilitaron modelos de más de cien mil millones de parámetros.

La idea del masked autoencoder, como forma general de *denoising autoencoder*, es natural también en visión; de hecho los *stacked denoising autoencoders* de Vincent (2008) y el [Context Encoder de Pathak](/papers/context-encoders-pathak-2016) (2016) *precedieron* a BERT. Sin embargo, pese al enorme interés tras el éxito de BERT, el autoencoding en visión quedó rezagado. El paper se pregunta *¿qué hace al masked autoencoding diferente entre visión y lenguaje?* y responde desde tres ángulos:

1. **Las arquitecturas eran distintas.** La visión estuvo dominada por **redes convolucionales**, que operan sobre grillas densas y no integran fácilmente indicadores como *tokens de máscara* o embeddings posicionales: el kernel desliza sobre un mapa denso y no sabe representar "aquí falta un parche". La llegada del [Vision Transformer (ViT)](/papers/vit-dosovitskiy-2021) elimina ese obstáculo: al tratar la imagen como secuencia de parches-token, agregar máscaras y posiciones es tan natural como en texto. ViT es, literalmente, el habilitador.

2. **La densidad de información es distinta.** El lenguaje es una señal humana, semántica y densa: predecir unas pocas palabras ya induce comprensión sofisticada. Las imágenes son señales naturales con **fuerte redundancia espacial** — un parche faltante se recupera de los vecinos con poco entendimiento de alto nivel. La consecuencia de diseño es la idea central de MAE: **enmascarar una porción muy alta de parches** reduce la redundancia y crea una tarea que exige comprensión holística más allá de las estadísticas de bajo nivel.

3. **El decoder juega un rol distinto.** En visión el decoder reconstruye píxeles, de nivel semántico *más bajo* que el reconocimiento; en lenguaje predice palabras ricas en semántica. En BERT el decoder puede ser trivial (un MLP), pero en imágenes su diseño determina el nivel semántico de las representaciones aprendidas.

## Contribución central: tres decisiones acopladas

1. **Tasa de enmascaramiento muy alta (≈75%) con muestreo aleatorio uniforme.** Frente al 15% típico de BERT y al 20–50% de trabajos previos (iGPT, ViT, BEiT), MAE oculta tres cuartas partes de los parches. Esto elimina la redundancia, impide resolver la tarea por extrapolación de vecinos y obliga a razonar sobre la *gestalt* de objetos y escenas. El muestreo uniforme evita un sesgo hacia el centro de la imagen.

2. **Encoder asimétrico que procesa solo los parches visibles.** El encoder es un ViT estándar aplicado *únicamente* al subconjunto visible (≈25%); los parches enmascarados se *eliminan*, no se reemplazan por tokens de máscara. Como el costo de la autoatención es cuadrático en el número de tokens, pasar de 100% a 25% es un ahorro enorme: permite entrenar encoders muy grandes con una fracción del cómputo y la memoria.

3. **Decoder ligero que reconstruye en píxeles.** Los tokens de máscara —vectores compartidos y aprendidos que marcan dónde falta un parche— se introducen *después* del encoder; el conjunto completo (parches codificados + máscaras, todos con embeddings posicionales) lo procesa un decoder pequeño y poco profundo. El decoder solo vive en el preentrenamiento y luego se descarta: para reconocimiento se conserva solo el encoder, aplicado a imágenes íntegras.

La combinación es *win-win*: la alta tasa de máscara optimiza la precisión *y* deja al encoder procesando poco, reduciendo el preentrenamiento 3× o más. Un cuarto rasgo distintivo: **MAE funciona sin augmentaciones intensas** — el rol regularizador que en el contrastivo cumplen las augmentaciones fuertes, aquí lo cumple el enmascaramiento aleatorio, que genera una vista nueva por iteración.

## Método: arquitectura, masking y target en píxeles

**Masking.** Siguiendo a ViT, la imagen se divide en parches regulares no solapados. Se muestrea un subconjunto sin reemplazo bajo distribución uniforme y se eliminan los demás. La entrada dispersa resultante es lo que habilita un encoder eficiente.

**Encoder.** Un ViT que embebe los parches visibles por proyección lineal con embeddings posicionales y los procesa con bloques Transformer. Crucialmente **no usa tokens de máscara**: solo ve parches reales, lo que evita una brecha entre preentrenamiento (entrada con muchas máscaras) y despliegue (imágenes íntegras).

**Decoder.** Recibe el conjunto completo —parches codificados visibles + tokens de máscara, todos con embeddings posicionales (sin ellos las máscaras no tendrían ubicación)—. Es deliberadamente ligero: por defecto 8 bloques y ancho 512, con <10% del cómputo por token del encoder. Su arquitectura es independiente del encoder porque solo existe durante el preentrenamiento.

**Target de reconstrucción.** MAE predice los valores de píxel de cada parche enmascarado. La última capa del decoder es una proyección lineal con tantos canales como píxeles tiene un parche, y la pérdida es el **error cuadrático medio (MSE) computado solo sobre los parches enmascarados** (como en BERT):

$$\mathcal{L} = \frac{1}{|\mathcal{M}|}\sum_{i \in \mathcal{M}} \lVert \hat{x}_i - x_i \rVert^2$$

Una variante importante usa **píxeles normalizados por parche** (se calcula media y desviación de cada parche y se normaliza): realza el contraste local, enfatiza alta frecuencia y mejora la representación (85.4% vs 84.9% en fine-tuning).

**Implementación simple.** No requiere operaciones dispersas: se generan tokens para todos los parches, se *barajan* aleatoriamente, se descarta la cola según la tasa de máscara (equivalente a muestrear sin reemplazo), se codifica el subconjunto, se anexan tokens de máscara, se *desbaraja* para alinear con los targets y se aplica el decoder. El overhead de barajar/desbarajar es despreciable.

## Experimentos: ImageNet, escalabilidad y transferencia

El preentrenamiento es autosupervisado sobre ImageNet-1K (IN1K); luego se evalúa con *fine-tuning* extremo-a-extremo o *linear probing*.

**Tasa de enmascaramiento.** Las tasas óptimas son sorprendentemente altas: 75% sirve tanto para linear probing como para fine-tuning. Para linear probing la precisión sube de 54.6% (10% de máscara) a 73.5% (75%); para fine-tuning un rango amplio (40–80%) funciona y todo supera el entrenamiento desde cero (82.5%).

**Token de máscara y eficiencia.** Meter tokens de máscara en el encoder *empeora* el linear probing en 14 puntos (por la brecha preentrenamiento/despliegue) y multiplica el cómputo. Sacarlos reduce los FLOPs 3.3× y da un *speedup* de pared de 2.8×, que sube a 3.5–4.1× con decoder de 1 bloque o encoder ViT-H. Entrenar ViT-L con máscaras toma 42.4 h vs 15.4 h sin ellas (mismas 800 épocas).

**Decoder.** Un decoder profundo ayuda al linear probing (hasta +8%) porque sus últimas capas se especializan en reconstrucción y dejan la latente más abstracta; para fine-tuning es poco influyente — incluso un decoder de un solo bloque rinde 84.8%.

**Augmentación.** MAE rinde bien con solo *cropping* e incluso de forma decente *sin ninguna augmentación* (solo center-crop). Contrasta con el contrastivo, que depende fuertemente de ella (BYOL y SimCLR pierden 13% y 28% solo con cropping). El rol regularizador lo asume el enmascaramiento aleatorio.

**Schedule.** La precisión mejora de forma sostenida con entrenamientos largos; el linear probing no satura ni a 1600 épocas, a diferencia de MoCo v3 (satura a 300). El encoder MAE solo ve 25% de parches por época, frente al 200%+ de las dos o más vistas del contrastivo.

**Resultados principales.** Con preentrenamiento solo en IN1K: ViT-B 83.6%, ViT-L 85.9%, ViT-H 86.9%, y ViT-H a 448 px **87.8%** — superando el mejor previo (87.1%) en el benchmark competitivo sin datos externos. MAE escala mejor que el supervisado y sigue una tendencia análoga al preentrenamiento en JFT-300M de ViT usando ~mil veces menos imágenes. Es más preciso, simple y rápido (3.5× por época) que BEiT, que predice tokens y requiere un dVAE preentrenado en 250M imágenes.

**Transferencia.** En **COCO** (Mask R-CNN con backbone ViT) MAE supera al supervisado en todas las configuraciones: +2.4 AP con ViT-B y +4.0 con ViT-L. En **ADE20K** (UperNet) mejora +3.7 mIoU sobre el supervisado con ViT-L y supera a BEiT. En clasificación (iNaturalist, Places) muestra fuerte escalamiento. Los **píxeles normalizados igualan a los tokens dVAE** en transferencia: la tokenización no es necesaria.

**Fine-tuning parcial y robustez.** Ajustar un solo bloque Transformer salta de 73.5% (linear probing) a 81.0%; las representaciones MAE son menos *linealmente* separables que las de MoCo v3 pero son rasgos no-lineales más fuertes, lo que cuestiona el linear probing como única métrica. En robustez, sin reentrenamiento especializado, el ViT-H pasa de 33.1% (supervisado) a 68.2% (MAE) en IN-Adversarial, una mejora de 35 puntos.

## Limitaciones reconocidas

- **El target son píxeles, no entidades semánticas.** El paper es honesto: las imágenes son "luz registrada" sin descomposición semántica análoga a las palabras. Que MAE infiera reconstrucciones holísticas plausibles se atribuye a una rica representación oculta — una hipótesis, no una prueba.
- **Menor separabilidad lineal.** Bajo linear probing puro MAE (75.8% con ViT-L) supera a los métodos basados en máscara previos pero queda por debajo del contrastivo MoCo v3 (77.6%); su fortaleza aparece con fine-tuning o ajuste parcial.
- **Impacto social.** El modelo predice según estadísticas del dataset, por lo que reflejará sus sesgos y puede generar contenido inexistente.

## Impacto: revivir el denoising autoencoder y fundar el SSL generativo en visión

La conclusión es programática: *los algoritmos simples que escalan bien son el núcleo del deep learning*. El aporte histórico tiene tres capas. Primero, **revive los denoising autoencoders**: la idea de corromper la entrada y reconstruir el original, latente desde Vincent et al. (2008), encuentra en el ViT el sustrato que le faltaba y se vuelve estado del arte. Segundo, **vuelve competitivo el SSL generativo frente al contrastivo**: hasta 2020 la visión autosupervisada estaba dominada por métodos contrastivos (SimCLR, MoCo, BYOL) que dependían de augmentaciones cuidadosas y pares positivos/negativos; MAE muestra que reconstruir-lo-que-falta, sin augmentaciones intensas ni vistas múltiples, transfiere mejor a tareas densas. Tercero, **funda el SSL moderno en visión**: MAE inspiró toda una familia de masked image modeling (SimMIM, data2vec, MaskFeat) y se extendió a video (ver [MAE para video, Feichtenhofer et al. 2022](/papers/mae-video-feichtenhofer-2022)), consolidando el preentrenamiento generativo como camino estándar para escalar ViTs.

## Por qué importa para la Clase 28

La [Clase 28](/clases/clase-28) ("Aprendizaje Autosupervisado") dedica un slide a MAE que resume la tesis del paper:

- **"Reviviendo ideas de Denoising Autoencoder."** El paper lo declara literalmente: MAE es "una forma de denoising autoencoding". Es el autoencoder llevado a su forma autosupervisada moderna, donde la "corrupción" es el enmascaramiento masivo de parches — cerrando el arco que abre la clase con los autoencoders.
- **"Tarea más adecuada a Transformer."** El traslado de la receta de BERT a visión solo fue posible gracias a [ViT](/papers/vit-dosovitskiy-2021), que trata la imagen como secuencia de parches-token e inserta máscaras y posiciones de forma natural.
- **"No trivial lidiar con parches ocultos en CNN."** Es la brecha arquitectónica: las convoluciones no integran fácilmente indicadores de "parche faltante", por eso los intentos previos en CNN no alcanzaban la simplicidad ni la escalabilidad de MAE.
- **"Ahorra cómputo procesando solo lo visible."** El encoder asimétrico que solo ve el ~25% visible reduce FLOPs 3.3× y da speedups de 2.8–4.1×, aprovechando la complejidad cuadrática de la autoatención.
- **"Funciona sin augmentaciones tan intensas."** MAE rinde decentemente incluso sin augmentación porque el enmascaramiento aleatorio genera una vista nueva por iteración — el contraste exacto con el contrastivo que la clase quiere destacar.

Para el marco conceptual del preentrenamiento sin etiquetas y su lugar entre los paradigmas de SSL (generativo vs contrastivo), ver el fundamento de [aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado) y el hub de la sesión en [Clase 28](/clases/clase-28).
