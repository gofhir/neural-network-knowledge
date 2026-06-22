---
title: "Objects that Sound (2018)"
weight: 316
math: true
---

{{< paper-card
    title="Objects that Sound"
    authors="Relja Arandjelović, Andrew Zisserman"
    year="2018"
    venue="ECCV 2018"
    pdf="/papers/objects-that-sound-arandjelovic-2018.pdf"
    arxiv="1712.06651" >}}
Continuación directa de [Look, Listen and Learn](/papers/look-listen-learn-arandjelovic-2017): los mismos autores (DeepMind/VGG) reutilizan la **tarea de correspondencia audio-visual (AVC)** —decidir si un *frame* y 1 segundo de audio provienen del mismo instante de video— como única señal de supervisión, gratuita, sin etiquetas humanas. Con ella resuelven dos problemas nuevos: (1) aprender **embeddings de audio e imagen alineados en un espacio común** (AVE-Net) para recuperación cross-modal, y (2) **localizar en la imagen el objeto que produce el sonido** (AVOL-Net), sin ninguna etiqueta de ubicación ni de identidad. Es [autosupervisión](/fundamentos/aprendizaje-autosupervisado) multimodal en estado puro.
{{< /paper-card >}}

---

## Contexto

Hacia 2016–2018 estalló el interés por el aprendizaje cross-modal entre imagen y audio (SoundNet, Aytar et al. 2016; Owens et al. 2016). El combustible fue un recurso casi ilimitado: los videos de YouTube entregan, simultáneamente, un *stream* visual y uno de audio sincronizado, y esa coocurrencia sirve para entrenar redes profundas **sin anotaciones**.

El paper destaca por qué el audio es un compañero distinto del texto. Aunque ambos son secuenciales, el texto está mucho más cerca de una anotación semántica: el concepto "perro" ya está disponible como palabra, y el problema se reduce a aterrizarlo en una región. Con audio, en cambio, la semántica es indirecta —el concepto "perro" no está en la señal cruda, hay que extraerlo con una ConvNet—, lo que vuelve el problema más parecido a clasificar imágenes y, por ello, interesante y difícil.

*Objects that Sound* es la **continuación directa de Look, Listen and Learn** (L3-Net, ICCV 2017), que introdujo la tarea AVC. Pero la L3-Net fusiona las dos modalidades por **concatenación** seguida de capas *fully connected*, y solo después calcula el *score* de correspondencia. Ese diseño tiene dos consecuencias que este trabajo ataca: los embeddings de audio y visión **no quedan alineados** en ningún espacio común (inservibles para recuperación cross-modal), y la red nunca fue pensada para responder *dónde* está el objeto que suena. La solución es rediseñar la arquitectura conservando intacta la señal AVC.

## La tarea AVC como objetivo autosupervisado

La entrada es un par (*frame*, 1 s de audio). El **positivo** se genera muestreando un video al azar, eligiendo un *frame* y tomando 1 segundo de audio con ese *frame* en su punto medio. El **negativo** toma *frame* y audio de **videos distintos**. La red predice corresponde / no corresponde con entropía cruzada binaria.

Por qué AVC fuerza semántica: la única manera de resolver la tarea es **clasificar conceptos en ambas modalidades** y juzgar si concuerdan. Además, la red visual ve un **solo *frame***, de modo que no puede hacer trampa explotando información de movimiento; debe entender el contenido estático.

## AVE-Net: alinear embeddings por distancia euclidiana

La imagen (224×224×3) y el audio (1 s a 48 kHz, convertido a log-espectrograma tratado como imagen 257×200) pasan por dos *subnetworks*. Cada una produce un **embedding de 128-D normalizado en L2**. Se calcula la **distancia euclidiana** entre ambos vectores, y ese **escalar único** pasa por una FC diminuta que lo calibra antes del softmax (su sesgo aprende el umbral de distancia sobre el cual el par se declara *no correspondiente*).

La clave es ese **cuello de botella de información**: como lo único que decide la correspondencia es la distancia entre los embeddings, la red está **obligada a alinear ambas modalidades en el mismo espacio**. El contraste con L3-Net es directo: aquella concatenaba las *features* y dejaba que las FC produjeran el *score*, sin nada que forzara alineamiento. La AVE-Net **mueve las FC dentro de cada *subnetwork*** y optimiza las *features* para recuperación. Respecto al *metric learning* clásico (*contrastive loss*, Chopra et al. 2005), tiene dos ventajas: es **libre de hiperparámetros** (no hay margen que ajustar) y produce explícitamente la salida corresponde-o-no, comparable con L3-Net.

## AVOL-Net: del embedding global al mapa espacial

Aquí está la "pequeña modificación de arquitectura" que habilita la localización. En lugar de un embedding único de la imagen entera, el objetivo es **hallar las regiones que explican el sonido**, dejando el resto como fondo. Se formula bajo **Multiple Instance Learning (MIL)**. Los cambios respecto de la AVE-Net son quirúrgicos:

- La *subnetwork* de visión **no hace *pooling* global**; sigue operando a resolución **14×14**. Las FC de visión se convierten en convoluciones 1×1 (*fully convolutional*).
- Se **elimina la normalización de *features*** para que el fondo pueda apagarse (respuesta baja).
- Se calcula el producto escalar entre cada uno de los **14×14 descriptores visuales de 128-D** y el **único descriptor de audio de 128-D**, produciendo un **mapa de similitud de 14×14**.
- Una convolución 1×1 diminuta calibra los *scores*, seguida de una **sigmoide**, dando un *score* de correspondencia **por posición espacial**.
- Un ***max-pooling* sobre todas las posiciones** entrega el *score* final, con el que se entrena la tarea AVC.

La lógica MIL: para pares correspondientes, una región debe responder alto (y localizar el objeto); para pares desemparejados, el máximo debe ser bajo, dejando todo el mapa apagado. En esencia, **la representación de audio actúa como un filtro de atención que busca parches relevantes** —el paper lo llama "atención infinitamente dura"—. A diferencia de los *heatmaps* de L3-Net (que dependían solo de la imagen), aquí la salida **depende del sonido**: cambia el audio, cambia la región resaltada.

> Las variantes con múltiples *frames* y flujo óptico suben la *accuracy* de AVC (~85% vs 81.9%) pero **no mejoran la recuperación**: con movimiento disponible, la red resuelve AVC explotando correlaciones de bajo nivel y pierde el incentivo de aprender embeddings semánticos. Por eso los experimentos principales usan un solo *frame*.

## Experimentos y resultados

**Dataset: AudioSet-Instruments.** Clips de 10 s de YouTube filtrados a instrumentos musicales, canto y herramientas: **110 clases**, con **263k / 30k / 4.3k** clips en *train* / *val* / *test*. **No se usa ninguna etiqueta para entrenar**; las etiquetas solo sirven para evaluar.

**Recuperación (nDCG@30).** En la propia tarea AVC, la AVE-Net logra **81.9%**, batiendo a L3-Net (80.8%). Pero AVC es solo un *proxy*; lo que importa es la recuperación:

| Método | im-im | im-aud | aud-im | aud-aud |
|---|---|---|---|---|
| Azar | .407 | .407 | .407 | .407 |
| L3-Net | .567 | .418 | .385 | .653 |
| L3-Net + CCA | .578 | .531 | .560 | .649 |
| VGG16-ImageNet (supervisado) | .600 | – | – | – |
| **AVE-Net** | **.604** | **.561** | **.587** | **.665** |

Lecturas clave: (1) en **cross-modal** (im-aud, aud-im) la AVE-Net bate a todos; las *features* crudas de L3-Net dan recuperación cross-modal **a nivel de azar** (.418, .385), confirmando que no están alineadas, y alinearlas con CCA *post hoc* ayuda pero entrenar la alineación directamente es mejor. (2) En **intra-modal** la AVE-Net incluso **supera levemente a VGG16-ImageNet** (.604 vs .600), pese a que esta fue entrenada de forma supervisada y a que la AVE-Net **nunca vio pares de la misma modalidad** —funciona por **transitividad**: la imagen de un violín está cerca del sonido de un violín, que a su vez está cerca de otras imágenes de violines—.

**Localización del objeto que suena.** La AVOL-Net iguala la *accuracy* de AVC de la AVE-Net (cambiar al esquema MIL no cuesta capacidad semántica) y localiza un rango amplio de objetos —teclados, tambores, arpas, guitarras, violines, bocas cantando— bajo *clutter*, variaciones de escala e iluminación, e incluso **múltiples objetos** a la vez.

La preocupación obvia —que la red solo detecte el objeto *saliente* ignorando el sonido— se **refuta** con un experimento ingenioso de pares desemparejados: ante una imagen de violín, reproducir sonido de tambores deja el mapa **vacío**; reproducir otro violín lo resalta. Y de forma decisiva, ante una imagen con **piano y flauta**, reproducir flauta resalta la flauta y reproducir piano resalta el piano. La localización depende genuinamente del sonido. Cuantitativamente: un *baseline* que siempre predice el centro logra **57.2%**; la AVOL-Net alcanza **81.7%**.

**La *cautionary tale*.** El muestreo ingenuo de negativos abre un atajo: el audio positivo siempre tiene su punto medio alineado a un *frame* (múltiplo de 0.04 s a 25 fps), el negativo no, así que existe una diferencia estadística de bajo nivel (artefactos de codificación MPEG o *resampling*) que la red aprende a explotar **en vez de semántica**. El efecto es aleccionador: **sin** prevención, la AVE-Net logra una *accuracy* artificialmente alta de **87.6%**, pero su recuperación es **1–2% peor**. **Con** prevención (muestrear el negativo también en múltiplos de 0.04 s), la *accuracy* baja a 81.9% pero las representaciones mejoran. Es decir, **mejor desempeño en la tarea *pretext* puede significar peores representaciones reales**.

## Limitaciones reconocidas

- **El movimiento no ayuda a la recuperación:** ofrece un atajo de bajo nivel que desincentiva aprender *features* semánticas.
- **Localización parcial:** al ser no supervisada, la AVOL-Net puede enfocar solo partes discriminativas (la interfaz manos-teclado) más que el objeto completo, y enfrenta la ambigüedad de *qué* hace el sonido. Gramófonos o radios (sonidos arbitrarios) quedan sin resolver.
- **Mono por diseño:** no usa información multicanal —la calibración del *rig* es desconocida en YouTube y los métodos multi-micrófono son sensibles a ruido—; la meta es semántica, no localizar "haciendo trampa".
- **Mejora futura propuesta:** reemplazar el *max-pooling* por **atención suave explícita**.

## Por qué importa

*Objects that Sound* es un hito en la línea que va del par contrastivo simple hacia la autosupervisión multimodal moderna. Mostró que una sola señal gratuita —la coocurrencia de imagen y sonido en video— basta no solo para aprender representaciones, sino para habilitar dos capacidades de alto nivel (recuperación y localización) que normalmente exigirían anotación costosa. La idea de **alinear modalidades en un espacio común mediante un cuello de botella de distancia** anticipa los métodos contrastivos cross-modal que dominarían después (la familia CLIP en imagen-texto, ConVIRT en imagen-texto clínico). Y el patrón de **localización por correspondencia espacial sin etiquetas** —tratar una modalidad como filtro de atención sobre la rejilla de la otra— reaparece en *grounding* visual y segmentación de vocabulario abierto.

## Conexión con la Clase 28

La [Clase 28](/clases/clase-28) (Aprendizaje Autosupervisado) incluye el slide "Correspondencia Audio-Visual", que afirma que "modificando un poco la arquitectura podemos saber en qué parte está el objeto que produce el sonido". Este paper es la fuente de esa afirmación y permite desempacarla:

- **El "modificar un poco" es literal y mínimo:** se parte de la AVE-Net y se hacen tres cambios (no hacer *pooling* global, convertir las FC en convoluciones 1×1, quitar la normalización). El resto queda igual, y de ese cambio menor **emerge gratis la localización**.
- **Por qué encaja en autosupervisión:** toda la potencia viene de la tarea *pretext* AVC, cuyas etiquetas se construyen del propio video. No hay etiquetas de clase, de ubicación ni de identidad —el caso paradigmático de [autosupervisión](/fundamentos/aprendizaje-autosupervisado) aplicado al setting multimodal—.
- **La *cautionary tale* es material de clase:** ilustra que el desempeño en la tarea *pretext* no es el objetivo, y que un atajo de bajo nivel puede inflar esa métrica mientras arruina las representaciones —el mismo fenómeno que motiva las aumentaciones agresivas en SimCLR/MoCo—.
- **Paper hermano:** para entender por qué la AVE-Net rediseña la fusión, conviene leer primero [Look, Listen and Learn](/papers/look-listen-learn-arandjelovic-2017).

Esta línea de trabajo vive en el [dominio multimodal](/dominios/multimodal), junto a CLIP, ConVIRT y la familia de modelos visión-lenguaje.
