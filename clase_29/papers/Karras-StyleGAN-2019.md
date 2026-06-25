# A Style-Based Generator Architecture for Generative Adversarial Networks (StyleGAN) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *A Style-Based Generator Architecture for Generative Adversarial Networks*.
- **Autores:** Tero Karras, Samuli Laine, Timo Aila (los tres de **NVIDIA**).
- **Venue:** CVPR 2019. **Preprint:** arXiv:1812.04948v3 (29 mar 2019), [arxiv.org/abs/1812.04948](https://arxiv.org/abs/1812.04948).
- **Código y dataset:** implementación oficial en TensorFlow ([github.com/NVlabs/stylegan](https://github.com/NVlabs/stylegan)); dataset FFHQ ([github.com/NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset)). Redes pre-entrenadas y video de acompañamiento liberados.

La tesis del paper es de **arquitectura del generador**, no de pérdida ni de entrenamiento adversarial. Los autores son explícitos en que **no modifican el discriminador ni la función de pérdida**, de modo que su trabajo es "ortogonal" a la discusión en curso sobre pérdidas de GAN, regularización e hiperparámetros. El problema que diagnostican: aun cuando las GANs progresivas ya producían imágenes de alta resolución y calidad, los generadores seguían operando como **cajas negras**. No se entendía el origen de los rasgos estocásticos, las propiedades del espacio latente eran pobremente comprendidas, y las interpolaciones latentes que se mostraban de costumbre no daban ninguna forma cuantitativa de comparar generadores entre sí. En resumen: **calidad alta, pero control nulo y comprensión nula**.

La propuesta, tomada prestada de la literatura de *style transfer*, rediseña el generador para exponer ejes de control nuevos. En vez de inyectar el código latente $z$ por la capa de entrada, StyleGAN lo transforma con una **red de mapeo** $f: \mathcal{Z} \to \mathcal{W}$ que produce un código intermedio $w$; la síntesis empieza desde un **tensor constante aprendido** y $w$ controla la imagen en cada resolución vía **normalización de instancia adaptativa (AdaIN)**, complementada con **ruido por-píxel** para el detalle estocástico. El resultado es triple: (1) mejora el estado del arte en métricas de calidad de distribución (FID), (2) demuestra mejores propiedades de interpolación, y (3) **desenreda automáticamente y sin supervisión** los factores de variación, separando atributos de alto nivel (pose, identidad) de la variación estocástica (pecas, pelo). Para cuantificar el desenredo, los autores introducen dos métricas nuevas — *perceptual path length* y *separabilidad lineal* — aplicables a cualquier generador. Finalmente, aportan **FFHQ (Flickr-Faces-HQ)**, un dataset de caras de mucha mayor calidad y variación que los existentes.

Para la Clase 29 (Modelos Generativos en Visión) StyleGAN importa porque representa la **cumbre de calidad de las GANs**: en la comparación de la clase aparece en la fila de "GANs — calidad alta", y FFHQ es justamente el dataset sobre el que se reportan los FID comparativos del módulo generativo.

## 2. Contexto histórico: de la GAN original al control jerárquico

El linaje es directo. La GAN original (ver [`/papers/goodfellow-gan-2014`](/papers/goodfellow-gan-2014)) estableció el juego minimax generador-discriminador pero sufría de inestabilidad y baja resolución. DCGAN (ver [`/papers/dcgan-radford-2015`](/papers/dcgan-radford-2015)) aportó la receta convolucional —generador transpuesto, BatchNorm, sin capas densas— que estabilizó el entrenamiento y permitió aritmética semántica en el espacio latente. La línea de NVIDIA culminó en **Progressive Growing of GANs** (Karras et al., 2017), que entrenaba primero a baja resolución (4×4) y añadía capas progresivamente hasta 1024×1024, logrando por primera vez caras de alta resolución convincentes.

StyleGAN parte exactamente de ahí: su **configuración base (A) es el setup de Progressive GAN**, del cual hereda las redes y todos los hiperparámetros salvo donde se indique. El punto de partida del paper no es "¿cómo generar imágenes mejores?" sino "¿cómo generar imágenes con el mismo nivel de calidad pero **entendiendo y controlando** lo que pasa dentro del generador?". Las GANs tradicionales alimentan el latente solo por la capa de entrada; toda la estructura jerárquica de la imagen —de la pose a los poros de la piel— tenía que emerger implícitamente de esa única inyección. StyleGAN rompe ese cuello de botella distribuyendo el control a cada resolución.

El préstamo conceptual viene del *style transfer* feedforward y de la traducción imagen-a-imagen no supervisada, donde se había establecido que las **estadísticas espacialmente invariantes** (matriz de Gram, media y varianza por canal) codifican de forma fiable el *estilo* de una imagen, mientras que las características que varían espacialmente codifican la *instancia* concreta. StyleGAN traslada esa dicotomía al generador: el estilo (vía AdaIN, global por canal) controla aspectos coherentes de toda la imagen; el ruido (por-píxel, local) controla la variación estocástica.

## 3. Contribución central

La aportación es una **arquitectura de generador basada en estilos** con cuatro componentes acoplados:

1. **Red de mapeo $f: \mathcal{Z} \to \mathcal{W}$** que produce un **espacio latente intermedio $\mathcal{W}$ desenredado**. A diferencia de $\mathcal{Z}$, que debe seguir la densidad de probabilidad de los datos de entrenamiento (y por tanto arrastra un enredo inevitable), $\mathcal{W}$ está libre de esa restricción y puede "desenrollarse".
2. **Inyección de estilo vía AdaIN en cada resolución**: transformaciones afines aprendidas especializan $w$ en estilos $y = (y_s, y_b)$ que escalan y sesgan las activaciones normalizadas, capa por capa.
3. **Ruido por-píxel** para el detalle estocástico: imágenes de ruido gaussiano no correlacionado, una por capa, escaladas por factores aprendidos y sumadas tras cada convolución.
4. **Style mixing (regularización de mezcla)**: durante el entrenamiento, un porcentaje de imágenes se genera mezclando dos códigos latentes en un punto de cruce aleatorio de la red, lo que decorrelaciona estilos vecinos y habilita control fino.

El efecto emergente más citado es el **control jerárquico de atributos por escala**: copiar estilos de resoluciones gruesas (4²–8²) transfiere pose, forma de cara y peinado general; de resoluciones medias (16²–32²), rasgos faciales más pequeños y ojos abiertos/cerrados; de resoluciones finas (64²–1024²), principalmente el esquema de color y la microestructura. Esta separación **no se programa**: surge de la arquitectura.

Como subproducto, dos métricas nuevas de desenredo (§7) y un dataset (§5).

## 4. Método: la tubería del generador basado en estilos

### 4.1. Red de mapeo y espacio $\mathcal{W}$

Dado $z \in \mathcal{Z}$ (normalizado), un **MLP de 8 capas** produce $w \in \mathcal{W}$. Ambos espacios tienen dimensión 512. La intuición está en la Figura 6 del paper: si el conjunto de entrenamiento carece de cierta combinación de factores (p.ej. "hombres de pelo largo"), el mapeo de $\mathcal{Z}$ al espacio de características debe **curvarse** para que esa combinación prohibida desaparezca —porque $\mathcal{Z}$ está obligado a respetar la densidad de los datos. El mapeo aprendido $f$ a $\mathcal{W}$ puede "deshacer" buena parte de ese curvado, porque $\mathcal{W}$ **no** tiene que seguir ninguna distribución fija: su densidad de muestreo la induce el propio $f(z)$. La hipótesis del paper es que hay presión durante el entrenamiento para que $\mathcal{W}$ se vuelva más lineal y menos enredado, porque generar imágenes realistas es más fácil desde una representación desenredada.

### 4.2. AdaIN — inyección de estilo capa por capa

La síntesis no recibe $z$ por una capa de entrada: empieza desde un **tensor constante aprendido de 4×4×512**. El estilo entra exclusivamente vía AdaIN:

$$\text{AdaIN}(x_i, y) = y_{s,i}\,\frac{x_i - \mu(x_i)}{\sigma(x_i)} + y_{b,i}$$

Cada mapa de características $x_i$ se normaliza por separado a media cero y varianza unitaria, y luego se escala y sesga con las componentes escalares del estilo $y$ derivado de $w$ por una transformación afín. La dimensión de $y$ es el doble del número de mapas de esa capa. El detalle clave de **localización**: como AdaIN primero normaliza y *después* aplica el estilo, las estadísticas que dicta el estilo no dependen de las estadísticas originales; por tanto **cada estilo controla una sola convolución antes de ser sobreescrito por el siguiente AdaIN**. Esto es lo que hace que modificar un subconjunto de estilos afecte solo a ciertos aspectos de la imagen.

La red de síntesis $g$ tiene 18 capas (dos por resolución, de 4² a 1024²); la salida pasa a RGB con una convolución 1×1. El generador tiene 26,2M parámetros (vs. 23,1M del tradicional).

### 4.3. Ruido por-píxel y style mixing

El ruido se inyecta tras cada convolución, antes de la no linealidad, como imágenes de un canal de ruido gaussiano broadcasteadas con factores de escala aprendidos por característica. El **style mixing** corre dos latentes $z_1, z_2$ por el mapeo, obtiene $w_1, w_2$, y aplica $w_1$ antes del punto de cruce y $w_2$ después. Esto impide que la red asuma que estilos adyacentes están correlacionados.

## 5. El dataset FFHQ (Flickr-Faces-HQ)

FFHQ es una contribución de peso por sí misma. Consta de **70.000 imágenes de caras humanas de alta calidad a resolución 1024²**. Frente a CelebA-HQ, ofrece **mucha más variación** en edad, etnia y fondo de imagen, y mejor cobertura de accesorios (gafas, gafas de sol, sombreros). Las imágenes se rastrearon de Flickr (heredando los sesgos de ese sitio), bajo licencias permisivas; se alinearon y recortaron automáticamente, se podaron con filtros automáticos, y finalmente se usó Mechanical Turk para eliminar estatuas, pinturas y fotos-de-fotos ocasionales. FFHQ se volvió el **benchmark estándar de facto** para síntesis de caras y aparece en la comparación FID de la Clase 29.

## 6. Experimentos: calidad de imagen (FID)

El paper construye los resultados de forma incremental, configuración por configuración, midiendo **Fréchet Inception Distance (FID)** (ver [`/papers/heusel-fid-2017`](/papers/heusel-fid-2017)) sobre 50.000 imágenes (menor es mejor). La Tabla 1, en FFHQ:

| Config | Descripción | CelebA-HQ | FFHQ |
| --- | --- | --- | --- |
| A | Progressive GAN base | 7,79 | 8,04 |
| B | + tuning (up/down bilineal, entrenamiento largo) | 6,11 | 5,25 |
| C | + red de mapeo y estilos (AdaIN) | 5,34 | 4,85 |
| D | + quitar la entrada tradicional | 5,07 | 4,88 |
| E | + ruido | 5,06 | 4,42 |
| F | + style mixing | 5,17 | 4,40 |

Dos observaciones del paper merecen destacarse. Primero, al añadir mapeo + AdaIN (C), **la red deja de beneficiarse de alimentar el latente a la primera convolución**: por eso (D) elimina la capa de entrada tradicional y arranca del constante aprendido sin perder calidad —un resultado que los autores califican de "notable". Segundo, el generador basado en estilos (E) mejora el FID casi un **20%** sobre el tradicional (B). Para CelebA-HQ se usó WGAN-GP; para FFHQ, WGAN-GP en (A) y pérdida no saturante con regularización $R_1$ en (B–F). Todos los FID se calculan **sin** truncation trick; este último solo se usa con fines ilustrativos.

## 7. Estudios de desenredo: dos métricas nuevas

Las métricas previas de desenredo requerían un encoder imagen→latente que la GAN base no tiene. Los autores proponen dos que **no requieren encoder ni factores conocidos**, computables para cualquier dataset y generador:

- **Perceptual path length (longitud de camino perceptual).** Mide cuán drásticamente cambia la imagen al interpolar en el espacio latente, usando una distancia perceptual basada en *embeddings* de VGG16 (LPIPS) sobre segmentos pequeños de la trayectoria ($\epsilon = 10^{-4}$). En $\mathcal{Z}$ se usa interpolación esférica (slerp); en $\mathcal{W}$, lineal (lerp), porque $\mathcal{W}$ no está normalizado. Un espacio menos curvado da transiciones perceptualmente más suaves. La Tabla 3 (FFHQ) muestra que la longitud de camino completa es **sustancialmente menor** para el generador basado en estilos con ruido (config E: 200,5 en $\mathcal{W}$ vs. 412,0 del tradicional en $\mathcal{Z}$), indicando que $\mathcal{W}$ es perceptualmente más lineal.
- **Separabilidad lineal.** Si $\mathcal{W}$ está desenredado, debe ser posible separar atributos binarios (p.ej. hombre/mujer) con un hiperplano. Se entrenan clasificadores auxiliares para 40 atributos de CelebA, se generan y etiquetan 200.000 imágenes, se descarta la mitad menos confiable, y se ajusta un SVM lineal por atributo; el score final es $\exp\!\big(\sum_i H(Y_i|X_i)\big)$ sobre la entropía condicional. Resultado (Tabla 3): el generador tradicional en $\mathcal{Z}$ da separabilidad **10,78**; el basado en estilos en $\mathcal{W}$ baja a **3,54–3,79** — mucho mejor separable, es decir, menos enredado.

La Tabla 4 añade un hallazgo de diseño: **profundizar la red de mapeo mejora FID, separabilidad y path length**, tanto en generadores tradicionales como basados en estilos. Curiosamente, anteponer una red de mapeo a un generador tradicional **empeora drásticamente la separabilidad en $\mathcal{Z}$ pero la mejora en $\mathcal{W}$**, confirmando la tesis: $\mathcal{Z}$ puede estar arbitrariamente enredado, y un espacio intermedio que no tenga que seguir la distribución de los datos ayuda incluso a la arquitectura tradicional.

## 8. Truncation trick y variación estocástica

El **truncation trick** opera en $\mathcal{W}$: se computa el centro de masa $\bar{w} = \mathbb{E}_{z}[f(z)]$ (una "cara promedio" en FFHQ) y se escala la desviación de cada $w$ como $w' = \bar{w} + \psi(w - \bar{w})$ con $\psi < 1$. Reducir $\psi \to 0$ converge a la cara media; con $\psi$ negativo aparece la "anti-cara". Notablemente, **la truncación se puede aplicar selectivamente solo a baja resolución**, dejando intacto el detalle de alta resolución. Funciona de forma fiable en $\mathcal{W}$ sin tocar la pérdida.

Sobre la **variación estocástica** (Figuras 4 y 5): distintas realizaciones del ruido cambian la colocación de cabellos individuales, los poros, el vello, pero dejan **intactos identidad y pose**. El ruido fino (64²–1024²) trae rizos finos y poros; el grueso (4²–32²), rizado de pelo a gran escala. La hipótesis del paper: en cualquier punto del generador hay presión por introducir contenido nuevo cuanto antes, y como hay ruido fresco disponible en cada capa, no hay incentivo para generar los efectos estocásticos desde activaciones previas — de ahí la localización del efecto.

## 9. Limitaciones

El paper de 2019 es notablemente positivo sobre su propia arquitectura ("el generador GAN tradicional es en todo sentido inferior a un diseño basado en estilos"), pero hay límites que el propio texto o el contexto posterior dejan ver:

- **Artefactos tipo "blob".** Aunque no es el foco de este paper, la arquitectura de StyleGAN produce **artefactos característicos en forma de gota (blob)** en las imágenes generadas, atribuidos posteriormente al diseño del AdaIN (la normalización destruye información de magnitud relativa entre características que la red recupera creando un pico de gran amplitud). Estos artefactos se **diagnosticaron y corrigieron en StyleGAN2** (Karras et al., 2020), que reemplazó AdaIN por modulación/demodulación de pesos.
- **Costo de la truncación.** La truncación mejora la calidad media a costa de variación, y solo un subconjunto de redes es amenable a ella (como ya observaba Brock et al.).
- **Tensión calidad–desenredo en el tiempo.** La Figura 9 muestra que conforme el FID sigue bajando tarde en el entrenamiento, la path length sube ligeramente — la mejora de calidad viene "al costo de una representación más enredada", una tensión que el paper deja abierta.
- **Sesgos del dataset.** FFHQ hereda los sesgos de Flickr; el paper lo reconoce explícitamente.
- **Solo arquitectura del generador.** No se reportan tiempos ni se toca el discriminador; la contribución es deliberadamente acotada.

## 10. Impacto y adopción

StyleGAN fue uno de los resultados generativos más influyentes de su época. Su impacto mediático más visible fue **[thispersondoesnotexist.com](https://thispersondoesnotexist.com)**, que mostraba caras hiperrealistas de personas inexistentes generadas por StyleGAN, llevando la calidad de las GANs al conocimiento del público general. Más allá del espectáculo, sus aportes técnicos perduraron: el **espacio latente $\mathcal{W}$ (y su extensión $\mathcal{W}+$)** se volvió el sustrato estándar para *edición semántica* de imágenes y *GAN inversion*; el **control de estilo por escala** habilitó toda una línea de manipulación de atributos; y FFHQ es hoy un benchmark de referencia. La familia continuó con **StyleGAN2** (corrigiendo los blobs y la "phase artifact" del progressive growing), **StyleGAN2-ADA** (entrenamiento con pocos datos vía augmentación adaptativa) y **StyleGAN3** (equivarianza a traslación/rotación, eliminando el "texture sticking"). En la narrativa de la Clase 29, StyleGAN marca el techo de calidad que las GANs alcanzaron antes de que los modelos de difusión (DDPM, Latent Diffusion) disputaran el liderazgo en calidad y diversidad de muestras.

## 11. Conexión con la Clase 29 (Modelos Generativos en Visión)

StyleGAN es la pieza que ancla la fila de "GANs — calidad alta" en la tabla comparativa de la clase. Su rol pedagógico es triple:

- **Cierra el arco de las GANs.** La clase recorre la genealogía GAN original → DCGAN → progressive growing → StyleGAN. Aquí culmina la idea de que **la arquitectura del generador, no solo la pérdida**, determina qué tan controlable e interpretable es el modelo. El contraste con la entrada única del latente en la GAN original (ver [`/papers/goodfellow-gan-2014`](/papers/goodfellow-gan-2014)) y con la receta convolucional de DCGAN (ver [`/papers/dcgan-radford-2015`](/papers/dcgan-radford-2015)) hace tangible cuánto se gana distribuyendo el control por escala.
- **FFHQ y FID en la comparación de la clase.** La métrica con la que la clase compara familias generativas es **FID** (ver [`/papers/heusel-fid-2017`](/papers/heusel-fid-2017)), y el dataset sobre el que se reportan esas cifras de caras es justamente FFHQ, introducido por este paper. Entender que el FID de StyleGAN en FFHQ ronda 4,40 da la referencia contra la cual se leen los modelos de difusión posteriores del módulo.
- **Desenredo y control como eje transversal.** Las dos métricas que introduce —path length y separabilidad— y la noción de espacio latente desenredado conectan con el tema de fondo del [`/fundamentos/modelos-generativos`](/fundamentos/modelos-generativos): no basta generar muestras realistas, importa qué tan **estructurado y manipulable** es el espacio latente. StyleGAN es el ejemplo canónico de un modelo cuyo latente se puede recorrer semánticamente.

Para profundizar en el recorrido completo del módulo generativo, ver [`/clases/clase-29`](/clases/clase-29).
