---
title: "First Order Motion Model for Image Animation (2019)"
weight: 464
math: true
---

{{< paper-card
    title="First Order Motion Model for Image Animation"
    authors="Aliaksandr Siarohin, Stéphane Lathuilière, Sergey Tulyakov, Elisa Ricci, Nicu Sebe (Universidad de Trento / Télécom Paris / Snap Inc.)"
    year="2019"
    venue="NeurIPS 2019 / arXiv:2003.00196"
    arxiv="2003.00196"
    pdf="/papers/fomm-siarohin-2019.pdf" >}}
Anima una **imagen fija** con el movimiento de un video, sin usar ninguna anotación ni conocimiento previo sobre el objeto que anima. Entrenado sobre videos de una categoría —rostros, cuerpos, caricaturas—, funciona después sobre cualquier objeto de esa clase. Su idea central está en el nombre: el campo de movimiento se aproxima por su **expansión de Taylor de primer orden** alrededor de un conjunto de puntos clave **aprendidos sin supervisión**, lo que agrega a cada punto un jacobiano —una transformación afín local— capaz de expresar rotación y escala, no solo desplazamiento. Un generador con **máscara de oclusión** decide qué se puede deformar desde la fuente y qué hay que inventar. Es el paper que la [Clase 44](/clases/clase-44) usa para su laboratorio.
{{< /paper-card >}}

---

## El problema

Animar una imagen consiste en generar un video donde el objeto de una imagen *fuente* se mueve como el objeto de un video *conductor*. La dificultad de fondo es que **no existe supervisión directa**: no hay pares de videos con objetos distintos moviéndose de forma idéntica.

La salida es auto-supervisarse. Durante el entrenamiento se toman **dos cuadros del mismo video** y se pide reconstruir uno a partir del otro más una representación latente del movimiento. En inferencia, se aplica el mismo modelo a una imagen fuente y a los cuadros de un video distinto. Es exactamente lo que la clase describe como *"entrenar para regenerar el video"* y luego *"reemplazar el frame inicial"*.

## El método

**Marco de referencia abstracto.** El campo de movimiento $\mathcal{T}_{S\leftarrow D}$ de un cuadro conductor a la imagen fuente se estima pasando por un marco de referencia $R$ que se cancela algebraicamente y **nunca se calcula**. Esto permite procesar $S$ y $D$ de forma independiente — necesario, porque en test provienen de videos que pueden verse muy distintos.

**Puntos clave sin supervisión.** Un codificador-decodificador predice $K$ puntos clave por cuadro. No son *landmarks* faciales predefinidos: la red los descubre sola, y actúan como **cuello de botella** que fuerza una representación compacta del movimiento. FOMM usa $K = 10$.

**La aproximación de primer orden.** Alrededor de cada punto clave $p_k$, la transformación se aproxima por Taylor:

$$\mathcal{T}_{X\leftarrow R}(p) = \mathcal{T}_{X\leftarrow R}(p_k) + \left.\frac{d}{dp}\mathcal{T}_{X\leftarrow R}(p)\right|_{p=p_k}(p - p_k) + o(\lVert p - p_k\rVert)$$

Es decir, el detector emite **la posición del punto y el jacobiano** en ese punto. El jacobiano es una matriz $2\times 2$ que codifica la transformación afín local — rotación, escala y cizalla.

**Flujo denso y oclusión.** Una segunda red combina las aproximaciones locales en un campo denso $\hat{\mathcal{T}}_{S\leftarrow D}$ y produce además una **máscara de oclusión** $\hat{O}_{S\leftarrow D}$ que indica qué partes del resultado pueden obtenerse deformando la fuente y cuáles hay que **inpaintar**, es decir, inferir del contexto.

**Generación.** El generador deforma los mapas de características de la fuente según el campo denso y rellena lo ocluido.

Se usa **flujo óptico hacia atrás**, y el paper explica por qué: el *back-warping* se implementa de forma diferenciable y eficiente con muestreo bilineal.

{{< concept-alert type="clave" >}}
La máscara de oclusión es la pieza que la clase omite y sin la cual el método no funcionaría. Cuando una cabeza gira, aparece una oreja que **no estaba en la imagen fuente**: ningún campo de deformación puede producirla, porque esos píxeles no existen en la entrada. La máscara le dice al generador *"esta región no la deformes, invéntala"*, y separa así los dos problemas —transportar lo que existe y generar lo que falta— en vez de pedirle a un solo mecanismo que resuelva ambos.
{{< /concept-alert >}}

## Cuánto vale el jacobiano

La contribución central se puede aislar y medir. Aproximando campos de movimiento suaves arbitrarios con una representación de $K$ puntos, con y sin jacobiano, ajustando los parámetros por mínimos cuadrados (**medido**):

| $K$ | error, solo posición | error, posición + jacobiano | mejora |
|---|---|---|---|
| 4 | 0,02879 | 0,01689 | 1,71× |
| 8 | 0,02102 | 0,00899 | 2,34× |
| 10 | 0,01873 | 0,00635 | **2,95×** |
| 16 | 0,01241 | 0,00247 | 5,02× |
| 24 | 0,00786 | 0,00074 | 10,64× |

Y sobre el caso que el jacobiano existe para resolver —una rotación pura, con 8 puntos (**medido**)—:

| rotación | error, solo posición | error, con jacobiano |
|---|---|---|
| 2° | 0,000493 | 2,6 × 10⁻¹⁶ |
| 10° | 0,002181 | 1,3 × 10⁻¹⁵ |
| 40° | 0,008623 | 4,4 × 10⁻¹⁵ |

Con jacobiano el error es **cero numérico** en todos los casos: una rotación es afín y la representación de primer orden la expresa exactamente. Sin él, el error crece linealmente con el ángulo.

{{< concept-alert type="recordar" >}}
Hay un matiz que el experimento revela y que hace más interesante el argumento. **A igual presupuesto de parámetros**, muchos puntos sin jacobiano aproximan un campo arbitrario tan bien o mejor que pocos puntos con jacobiano (medido: con 48 y 72 parámetros gana la representación de orden 0).

Entonces, ¿por qué usar jacobianos? Porque el presupuesto que importa no son los parámetros del campo sino **el número de puntos clave**: cada uno es una parte del objeto que la red debe descubrir **sin supervisión** y seguir de forma consistente entre cuadros. Sostener 10 puntos coherentes es factible; sostener 96 no lo es.

El jacobiano compra precisión sin pagar en número de partes que hay que descubrir. Es una decisión sobre la dificultad del **aprendizaje**, no sobre la capacidad de representación.
{{< /concept-alert >}}

## Entrenamiento

Pérdida de reconstrucción perceptual sobre el cuadro regenerado, más una **pérdida de equivarianza** que exige que los puntos clave detectados se transformen de forma consistente cuando se aplica una deformación conocida a la imagen — el mecanismo que evita que los puntos degeneren, dado que nada externo dice dónde deberían estar. El paper extiende esa pérdida, habitual para posiciones, de modo que también restrinja los jacobianos.

Se evalúa sobre VoxCeleb (rostros), UvA-Nemo, Tai-Chi-HD (cuerpos completos) y BAIR (robótica), superando a los métodos previos —X2Face y Monkey-Net— en todas las categorías. La clase menciona VoxCeleb y anota correctamente que *"en realidad también usan datasets de videos a cuerpo completo"*.

## Limitaciones

- **El movimiento no puede ser demasiado grande.** La aproximación de primer orden vale en un entorno de cada punto clave; si la pose del video conductor difiere mucho de la de la imagen fuente, deja de valer y aparecen deformaciones.
- **Requiere que fuente y conductor sean de la misma categoría.** Un modelo entrenado con rostros no anima un cuerpo completo.
- **La identidad puede filtrarse** desde el video conductor: si la forma del rostro conductor es muy distinta, parte de esa geometría contamina el resultado.
- **No hay modelo 3D ni de iluminación.** Todo ocurre en el plano de la imagen, así que las sombras y los reflejos no se recalculan.

## Por qué importa para la Clase 44

Es el paper del laboratorio y el ejemplo con el que la clase cierra el diplomado. Su valor didáctico está en que **casi todos sus componentes ya aparecieron en el curso**: puntos clave y su seguimiento ([Clase 17](/clases/clase-17), [Clase 42](/clases/clase-42)), flujo óptico y deformación ([Clase 36](/clases/clase-36)), codificador-decodificador con generador adversarial ([Clase 29](/clases/clase-29)), auto-supervisión a partir de la estructura del video ([Clase 28](/clases/clase-28), [Clase 43](/clases/clase-43)). Es el sentido preciso de la diapositiva de apertura: *"aplicaciones sorprendentes usando las cosas que ya conocemos"*.

Vale hacer una precisión de vocabulario que la clase no hace: **FOMM es animación de imágenes (*reenactment*), no *face swap***. No reemplaza un rostro dentro de un video existente; anima una foto con el movimiento de otro video, y la salida conserva identidad, fondo y encuadre de la **foto**. La distinción importa porque cambia qué se necesita —una sola imagen en vez de miles— y qué rastros deja. Ver [Síntesis de Medios](/fundamentos/sintesis-de-medios).

---

**Ver también:** [FaceForensics++ (2019)](/papers/faceforensics-rossler-2019) · [SV2TTS (2018)](/papers/sv2tts-jia-2018) · [Síntesis de Medios](/fundamentos/sintesis-de-medios) · [Clase 29 - Modelos Generativos](/clases/clase-29) · [Clase 44 — Práctica](/clases/clase-44/practica)
