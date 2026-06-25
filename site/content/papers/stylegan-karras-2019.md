---
title: "StyleGAN: A Style-Based Generator (2019)"
weight: 335
math: true
---

{{< paper-card
    title="A Style-Based Generator Architecture for Generative Adversarial Networks"
    authors="Tero Karras, Samuli Laine, Timo Aila"
    year="2019"
    venue="CVPR 2019"
    pdf="/papers/stylegan-karras-2019.pdf"
    arxiv="1812.04948" >}}
Paper de NVIDIA que rediseña el **generador** de una GAN —sin tocar discriminador ni pérdida— para volverlo controlable e interpretable. En vez de inyectar el latente $z$ por la capa de entrada, lo transforma con una **red de mapeo** $f:\mathcal{Z}\to\mathcal{W}$ que produce un espacio intermedio **desenredado** $\mathcal{W}$; la síntesis arranca desde un tensor constante aprendido y $w$ controla cada resolución vía **AdaIN**, con **ruido por-píxel** para el detalle estocástico. El resultado: mejor FID, control jerárquico de atributos por escala, dos métricas nuevas de desenredo y el dataset **FFHQ**. Es la cumbre de calidad de las GANs, popularizada por *thispersondoesnotexist.com*. Ancla la familia generativa en la [Clase 29](/clases/clase-29).
{{< /paper-card >}}

---

## Contexto

El linaje es directo. La [GAN original](/papers/gan-goodfellow-2014) estableció el juego minimax generador-discriminador pero sufría inestabilidad y baja resolución. [DCGAN](/papers/dcgan-radford-2015) aportó la receta convolucional —generador transpuesto, BatchNorm, sin capas densas— que estabilizó el entrenamiento y habilitó aritmética semántica en el espacio latente. La línea de NVIDIA culminó en **Progressive Growing of GANs** (Karras et al., 2017), que entrenaba primero a baja resolución (4×4) y añadía capas progresivamente hasta 1024×1024, logrando por primera vez caras de alta resolución convincentes.

StyleGAN parte exactamente de ahí: su configuración base es el setup de Progressive GAN, del que hereda redes e hiperparámetros. El diagnóstico de los autores no es "¿cómo generar mejores imágenes?" sino "¿cómo generarlas con el mismo nivel de calidad pero **entendiendo y controlando** lo que pasa dentro del generador?". Las GANs tradicionales alimentan el latente solo por la capa de entrada; toda la jerarquía de la imagen —de la pose a los poros— debía emerger implícitamente de esa única inyección. El generador seguía siendo una **caja negra**: calidad alta, control nulo, comprensión nula.

El préstamo conceptual viene del *style transfer*, donde se había establecido que las **estadísticas espacialmente invariantes** (media y varianza por canal) codifican el *estilo* de una imagen, mientras las características que varían espacialmente codifican la *instancia* concreta. StyleGAN traslada esa dicotomía al generador: el estilo (global, por canal) controla aspectos coherentes de toda la imagen; el ruido (local, por-píxel) controla la variación estocástica.

## Contribución central

La aportación es una **arquitectura de generador basada en estilos** con cuatro piezas acopladas:

1. **Red de mapeo $f:\mathcal{Z}\to\mathcal{W}$** que produce un espacio latente intermedio **desenredado**. A diferencia de $\mathcal{Z}$, que debe seguir la densidad de probabilidad de los datos de entrenamiento (y arrastra un enredo inevitable), $\mathcal{W}$ no tiene esa restricción y puede "desenrollarse".
2. **Inyección de estilo vía AdaIN en cada resolución**: transformaciones afines aprendidas especializan $w$ en estilos que escalan y sesgan las activaciones normalizadas, capa por capa.
3. **Ruido por-píxel** para el detalle estocástico: imágenes de ruido gaussiano no correlacionado, una por capa, escaladas por factores aprendidos y sumadas tras cada convolución.
4. **Style mixing** (regularización de mezcla): durante el entrenamiento, una fracción de imágenes se genera mezclando dos latentes en un punto de cruce aleatorio, lo que decorrelaciona estilos vecinos.

El efecto emergente más citado es el **control jerárquico de atributos por escala**, que no se programa: surge de la arquitectura.

## La red de mapeo y el espacio $\mathcal{W}$

Dado $z\in\mathcal{Z}$ (normalizado), un **MLP de 8 capas** produce $w\in\mathcal{W}$; ambos espacios tienen dimensión 512. La intuición: si el conjunto de entrenamiento carece de cierta combinación de factores (p.ej. "hombres de pelo largo"), el mapeo desde $\mathcal{Z}$ debe **curvarse** para que esa combinación prohibida desaparezca, porque $\mathcal{Z}$ está obligado a respetar la densidad de los datos. El mapeo aprendido $f$ a $\mathcal{W}$ puede deshacer buena parte de ese curvado, porque $\mathcal{W}$ **no** tiene que seguir ninguna distribución fija: su densidad de muestreo la induce el propio $f(z)$. La hipótesis es que hay presión durante el entrenamiento para que $\mathcal{W}$ se vuelva más lineal y menos enredado, porque generar imágenes realistas es más fácil desde una representación desenredada.

## AdaIN: inyección de estilo capa por capa

La síntesis no recibe $z$ por una capa de entrada: empieza desde un **tensor constante aprendido de 4×4×512**. El estilo entra exclusivamente vía normalización de instancia adaptativa:

$$\text{AdaIN}(x_i, y) = y_{s,i}\,\frac{x_i - \mu(x_i)}{\sigma(x_i)} + y_{b,i}$$

Cada mapa de características $x_i$ se normaliza por separado a media cero y varianza unitaria, y luego se escala ($y_{s,i}$) y sesga ($y_{b,i}$) con las componentes del estilo $y$ derivado de $w$ por una transformación afín. El detalle clave de **localización**: como AdaIN primero normaliza y *después* aplica el estilo, las estadísticas que dicta el estilo no dependen de las originales; por tanto **cada estilo controla una sola convolución antes de ser sobreescrito por el siguiente AdaIN**. Esto es lo que hace que modificar un subconjunto de estilos afecte solo a ciertos aspectos de la imagen.

La red de síntesis tiene 18 capas (dos por resolución, de 4² a 1024²); la salida pasa a RGB con una convolución 1×1. El generador suma 26,2M parámetros (vs. 23,1M del tradicional).

## Ruido por-píxel y style mixing

El **ruido** se inyecta tras cada convolución, antes de la no linealidad, como imágenes de un canal de ruido gaussiano escaladas por factores aprendidos por característica. Distintas realizaciones del ruido cambian la colocación de cabellos individuales, poros y vello, pero dejan **intactos identidad y pose**: el ruido fino (64²–1024²) trae rizos finos y poros; el grueso (4²–32²), rizado de pelo a gran escala.

El **style mixing** corre dos latentes $z_1, z_2$ por el mapeo, obtiene $w_1, w_2$, y aplica $w_1$ antes del punto de cruce y $w_2$ después. Impide que la red asuma que estilos adyacentes están correlacionados. Su uso en *inferencia* revela el control por escala: copiar estilos de resoluciones **gruesas** (4²–8²) transfiere pose, forma de cara y peinado general; de **medias** (16²–32²), rasgos faciales menores y ojos abiertos/cerrados; de **finas** (64²–1024²), principalmente el esquema de color y la microestructura.

## El dataset FFHQ

FFHQ (Flickr-Faces-HQ) es una contribución de peso por sí misma: **70.000 imágenes de caras humanas a resolución 1024²**. Frente a CelebA-HQ ofrece **mucha más variación** en edad, etnia y fondo, y mejor cobertura de accesorios (gafas, sombreros). Las imágenes se rastrearon de Flickr bajo licencias permisivas, se alinearon y recortaron automáticamente, se podaron con filtros y finalmente se limpiaron con Mechanical Turk (quitando estatuas, pinturas y fotos-de-fotos). FFHQ se volvió el **benchmark estándar de facto** para síntesis de caras.

## Resultados: calidad de imagen (FID)

El paper construye los resultados de forma incremental, configuración por configuración, midiendo [FID](/papers/fid-heusel-2017) sobre 50.000 imágenes (menor es mejor):

| Config | Descripción | FFHQ |
|---|---|---|
| A | Progressive GAN base | 8,04 |
| B | + tuning (up/down bilineal, entrenamiento largo) | 5,25 |
| C | + red de mapeo y estilos (AdaIN) | 4,85 |
| D | + quitar la entrada tradicional | 4,88 |
| E | + ruido | 4,42 |
| F | + style mixing | 4,40 |

Dos observaciones clave. Primero, al añadir mapeo + AdaIN (C), la red **deja de beneficiarse de alimentar el latente a la primera convolución**: por eso (D) elimina la capa de entrada tradicional y arranca del constante aprendido sin perder calidad —un resultado que los autores califican de notable. Segundo, el generador basado en estilos mejora el FID casi un **20%** sobre el tradicional. Todos los FID se calculan **sin** truncation trick.

## Métricas de desenredo

Las métricas previas requerían un encoder imagen→latente que la GAN base no tiene. Los autores proponen dos que **no requieren encoder ni factores conocidos**:

- **Perceptual path length (longitud de camino perceptual).** Mide cuán drásticamente cambia la imagen al interpolar en el espacio latente, usando distancia perceptual (LPIPS sobre embeddings de VGG16) sobre segmentos pequeños de la trayectoria. Un espacio menos curvado da transiciones perceptualmente más suaves. La longitud de camino completa es **sustancialmente menor** en el generador basado en estilos (200,5 en $\mathcal{W}$ vs. 412,0 del tradicional en $\mathcal{Z}$): $\mathcal{W}$ es perceptualmente más lineal.
- **Separabilidad lineal.** Si $\mathcal{W}$ está desenredado, debe poderse separar atributos binarios (p.ej. hombre/mujer) con un hiperplano. Se entrenan clasificadores auxiliares para 40 atributos de CelebA, se etiquetan 200.000 imágenes y se ajusta un SVM lineal por atributo. El generador tradicional en $\mathcal{Z}$ da separabilidad **10,78**; el basado en estilos en $\mathcal{W}$ baja a **3,54–3,79** —mucho mejor separable, es decir, menos enredado.

Profundizar la red de mapeo mejora FID, separabilidad y path length en ambas arquitecturas. Anteponer un mapeo a un generador tradicional **empeora la separabilidad en $\mathcal{Z}$ pero la mejora en $\mathcal{W}$**, confirmando la tesis.

## Truncation trick

El truncation trick opera en $\mathcal{W}$: se computa el centro de masa $\bar{w}=\mathbb{E}_z[f(z)]$ (una "cara promedio" en FFHQ) y se escala la desviación de cada $w$ como $w'=\bar{w}+\psi(w-\bar{w})$ con $\psi<1$. Reducir $\psi\to 0$ converge a la cara media. Notablemente, la truncación se puede aplicar **selectivamente solo a baja resolución**, dejando intacto el detalle de alta resolución, y funciona de forma fiable sin tocar la pérdida.

## Limitaciones

- **Artefactos tipo "blob".** La arquitectura produce artefactos característicos en forma de gota, atribuidos al diseño de AdaIN: la normalización destruye información de magnitud relativa entre características y la red la recupera creando un pico de gran amplitud. Estos artefactos se **diagnosticaron y corrigieron en StyleGAN2** (Karras et al., 2020), que reemplazó AdaIN por modulación/demodulación de pesos.
- **Tensión calidad–desenredo.** Conforme el FID sigue bajando tarde en el entrenamiento, la path length sube ligeramente: la mejora de calidad viene "al costo de una representación más enredada", una tensión que el paper deja abierta.
- **Costo de la truncación.** Mejora la calidad media a costa de variación, y solo un subconjunto de redes es amenable a ella.
- **Sesgos del dataset.** FFHQ hereda los sesgos de Flickr; el paper lo reconoce explícitamente.
- **Solo arquitectura del generador.** No se toca el discriminador; la contribución es deliberadamente acotada.

## Impacto

StyleGAN fue uno de los resultados generativos más influyentes de su época. Su impacto mediático más visible fue **thispersondoesnotexist.com**, que mostraba caras hiperrealistas de personas inexistentes, llevando la calidad de las GANs al conocimiento del público general. Más allá del espectáculo, sus aportes técnicos perduraron: el espacio latente $\mathcal{W}$ (y su extensión $\mathcal{W}+$) se volvió el sustrato estándar para *edición semántica* de imágenes y *GAN inversion*; el control de estilo por escala habilitó toda una línea de manipulación de atributos; y FFHQ es hoy un benchmark de referencia. El desenredo latente que StyleGAN ofrece resultó además clave para trabajos posteriores que explotan los embeddings de la GAN como fuente de etiquetas, como [DatasetGAN](/papers/datasetgan-zhang-2021). La familia continuó con **StyleGAN2** (corrigiendo los blobs), **StyleGAN2-ADA** (entrenamiento con pocos datos) y **StyleGAN3** (equivarianza a traslación/rotación).

## Por qué importa para la Clase 29

StyleGAN ancla la fila de "GANs — calidad alta" en la tabla comparativa de la [Clase 29](/clases/clase-29). Su rol pedagógico es triple:

- **Cierra el arco de las GANs.** La clase recorre la genealogía [GAN original](/papers/gan-goodfellow-2014) → [DCGAN](/papers/dcgan-radford-2015) → progressive growing → StyleGAN. Aquí culmina la idea de que **la arquitectura del generador, no solo la pérdida**, determina qué tan controlable e interpretable es el modelo.
- **FFHQ y FID en la comparación de la clase.** La métrica con que la clase compara familias generativas es [FID](/papers/fid-heusel-2017), y el dataset sobre el que se reportan las cifras de caras es FFHQ. El FID de StyleGAN en FFHQ ronda 4,40, la referencia contra la cual se leen los modelos de difusión posteriores del módulo.
- **Desenredo y control como eje transversal.** Las dos métricas que introduce —path length y separabilidad— y la noción de espacio latente desenredado conectan con el tema de fondo de [modelos generativos](/fundamentos/modelos-generativos): no basta generar muestras realistas, importa qué tan **estructurado y manipulable** es el espacio latente. StyleGAN es el ejemplo canónico de un modelo cuyo latente se puede recorrer semánticamente.
