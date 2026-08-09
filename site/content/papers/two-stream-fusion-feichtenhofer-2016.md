---
title: "Two-Stream Fusion: dónde y cómo fusionar (2016)"
weight: 421
math: true
---

{{< paper-card
    title="Convolutional Two-Stream Network Fusion for Video Action Recognition"
    authors="Christoph Feichtenhofer, Axel Pinz, Andrew Zisserman (TU Graz, Oxford VGG)"
    year="2016"
    venue="CVPR 2016 / arXiv:1604.06573"
    arxiv="1604.06573"
    pdf="/papers/two-stream-fusion-feichtenhofer-2016.pdf" >}}
Dos años después de [Two-Stream](/papers/two-stream-simonyan-2014), y desde el mismo grupo (Zisserman firma ambos, y los autores agradecen las discusiones con Simonyan), este paper hace la pregunta que la arquitectura original dejó abierta: **si en lugar de promediar los softmax fusionamos en medio de la red, ¿qué pasa?** La respuesta no es una arquitectura nueva presentada como fait accompli, sino un estudio de ablación disciplinado sobre tres ejes ortogonales —**cómo** fusionar (suma, máximo, concatenación, convolución, bilineal), **dónde** fusionar (ReLU2 a ReLU5, capas totalmente conectadas, combinaciones) y **cómo fusionar temporalmente** (pooling 2D, pooling 3D, convolución 3D seguida de pooling 3D)— del que después se ensambla la red final. Las tres conclusiones: se puede fusionar en una capa convolucional **sin pérdida de desempeño y con casi la mitad de los parámetros** (97.58M contra 181.42M); el punto óptimo es la **última capa convolucional**, no antes; y hacer *pooling* de features convolucionales sobre vecindarios espacio-temporales mejora todavía más. En números: **92.5% en UCF-101 y 65.4% en HMDB-51** con VGG-16 en ambas corrientes, contra 88.0% / 59.4% del two-stream original, y **93.5% / 69.2%** combinado con iDT, el mejor resultado publicado en 2016. Para la [Clase 38](/clases/clase-38) el paper tiene un lugar preciso: es la familia **(d) 3D-Fused Two-Stream** de la comparativa canónica de [I3D](/papers/i3d-carreira-2017), la caja que el diagrama muestra pero no explica.
{{< /paper-card >}}

---

## Contexto: la fusión tardía deja algo sobre la mesa

El [two-stream original](/papers/two-stream-simonyan-2014) descompone el video en dos señales explícitas: fotogramas RGB para la apariencia y pilas de $L=10$ campos de [flujo óptico](/fundamentos/flujo-optico) horizontal y vertical para el movimiento. Cada señal alimenta una ConvNet independiente, entrenada para clasificar la acción **por su cuenta**, y la predicción final es el promedio de los scores de softmax. El diseño fue brillante por una razón concreta: permitió heredar el preentrenamiento de ImageNet en lugar de aprender el movimiento desde píxeles crudos.

Feichtenhofer et al. identifican dos defectos estructurales. El primero conviene decirlo sin eufemismos: **la red nunca aprende "qué se mueve dónde"**. Si la fusión ocurre en el softmax, lo único que se combina son dos distribuciones sobre 101 clases. Nada en la arquitectura permite que un parámetro entrenable represente la conjunción *"movimiento periódico de mano EN la ubicación de la boca"*.

El ejemplo del paper es el más limpio posible: **distinguir "cepillarse los dientes" de "cepillarse el pelo"**. En ambos casos hay una mano moviéndose periódicamente en algún lugar. La corriente temporal reconoce ese movimiento pero no puede decir *dónde* en términos semánticos; la espacial reconoce la ubicación (dientes o pelo) pero no el movimiento. Solo su combinación **en la misma posición espacial** discrimina la acción. El argumento constructivo: si distintos canales de la red espacial responden a distintas áreas faciales y algún canal de la temporal responde a ese tipo de campo de movimiento, al apilar los canales los filtros de las capas siguientes **pueden aprender la correspondencia entre los canales apropiados**, exactamente como pesos de un filtro de convolución.

El segundo defecto es de escala temporal: la corriente espacial ve un fotograma y la temporal una ventana de $L$ fotogramas adyacentes. El two-stream original mitigaba esto haciendo *pooling* de predicciones sobre 25 muestras del video, pero promediar 25 predicciones independientes **no modela evolución temporal**: no es lo mismo que aprender que "dibujar una flecha, tensar el arco, disparar" es una secuencia con orden. Es la misma debilidad que la [Clase 36](/clases/clase-36) señala en el [reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones).

> El contexto de 2016 era además incómodo: los mejores números seguían obteniéndose al combinar ConvNets con codificaciones Fisher Vector de features artesanales. Parte de la explicación es de escala: UCF-101 tiene 100 ejemplos por clase contra los 1000 de ImageNet.

---

## Cómo fusionar: los operadores

Formalmente, el operador es $f: \mathbf{x}^a, \mathbf{x}^b \to \mathbf{y}$ con $\mathbf{x}^a \in \mathbb{R}^{H \times W \times D}$ y $\mathbf{x}^b \in \mathbb{R}^{H' \times W' \times D'}$. Todos los operadores actúan índice por índice sobre $(i,j)$, lo que impone una **restricción dura de correspondencia espacial**: $H = H'$ y $W = W'$. Si los mapas no están registrados píxel a píxel, el operador combina features de lugares distintos de la imagen y toda la premisa del "qué se mueve dónde" se desmorona.

**Suma.** $y^{\text{sum}}_{i,j,d} = x^a_{i,j,d} + x^b_{i,j,d}$, salida en $\mathbb{R}^{H \times W \times D}$. El punto conceptual: **la numeración de los canales es arbitraria**, así que la suma impone una correspondencia $d$-a-$d$ igualmente arbitraria. No hay razón para que el canal 37 de la red espacial se empareje con el 37 de la temporal. El entrenamiento puede *explotar* esa correspondencia ajustando los filtros previos, pero no puede *elegirla*.

**Máximo.** $y^{\text{max}}_{i,j,d} = \max\{x^a_{i,j,d}, x^b_{i,j,d}\}$. Misma arbitrariedad, más la desventaja de descartar información: en cada posición y canal una de las dos corrientes se pierde por completo.

**Concatenación.** Apila los mapas a lo largo de los canales en las mismas posiciones, con $\mathbf{y} \in \mathbb{R}^{H \times W \times 2D}$. **No define ninguna correspondencia** y deja que las capas siguientes la aprendan. El costo: sin reducción de dimensionalidad la capa siguiente recibe $2D$ canales, y como esa capa es FC6, **duplica los parámetros de la primera capa densa**.

**Convolución (la ganadora).** Concatena y luego convoluciona con un banco $\mathbf{f} \in \mathbb{R}^{1 \times 1 \times 2D \times D}$ y sesgos $b \in \mathbb{R}^D$:

$$\mathbf{y}^{\text{conv}} = \mathbf{y}^{\text{cat}} * \mathbf{f} + b$$

El filtro actúa **puramente en la dimensión de canales, sin extensión espacial**: reduce la dimensionalidad por dos y modela combinaciones ponderadas de $\mathbf{x}^a$ y $\mathbf{x}^b$ en la misma posición de píxel. El punto teórico elegante: **contiene a la suma como caso particular**. Si $\mathbf{f}$ se aprende como la concatenación de dos matrices identidad, el canal $i$ de una red se combina solo con el canal $i$ de la otra, vía suma. Es **la generalización aprendible de la suma**.

**Bilineal.** Producto externo por píxel seguido de suma sobre las posiciones:

$$\mathbf{y}^{\text{bil}} = \sum_{i=1}^{H} \sum_{j=1}^{W} \mathbf{x}^{a\top}_{i,j} \, \mathbf{x}^{b}_{i,j} \in \mathbb{R}^{D^2}$$

Captura **interacciones multiplicativas**: cada canal de una red se combina con cada canal de la otra, así que la correspondencia es completa y no arbitraria. Las desventajas son severas: la dimensionalidad explota a $D^2$ (262144 para $D=512$) y **toda la información espacial queda marginalizada** por la suma sobre $i,j$. En la práctica se aplica en ReLU5, se eliminan las capas densas y se clasifica con SVMs lineales sobre features normalizados.

Resultados con dos VGG-M-2048, fusión en la salida de **ReLU5** (después de la rectificación, que en experimentos preliminares dio mejor que la salida no rectificada de conv5), una sola torre después, UCF-101 split 1:

| Fusión | Capa | Accuracy | #capas | #parámetros |
|---|---|---|---|---|
| Suma (reportado en two-stream original) | Softmax | 85.6% | 16 | 181.42M |
| Suma (reimplementación de los autores) | Softmax | 85.94% | 16 | 181.42M |
| Máximo | ReLU5 | 82.70% | 13 | 97.31M |
| Concatenación | ReLU5 | 83.53% | 13 | 172.81M |
| Bilineal | ReLU5 | 85.05% | 10 | 6.61M + SVM |
| Suma | ReLU5 | 85.20% | 13 | 97.31M |
| **Convolución** | **ReLU5** | **85.96%** | **14** | **97.58M** |

Máximo y concatenación quedan considerablemente atrás, y la convolución en ReLU5 **iguala exactamente al promedio de softmax con la mitad de los parámetros**, pagando apenas 0.27M extra sobre la suma por el filtro $1\times1\times1024\times512$.

La decisión de implementación que el paper marca como crítica es la **inicialización con matrices identidad**, para que la capa *empiece* sumando las dos corrientes. Y acá viene el dato honesto: la identidad rinde **85.96% contra 85.59%** con ruido gaussiano, es decir prácticamente igual, **pero converge mucho más rápido**. La lectura tiene dos filos: la identidad arranca en un punto ya funcional y solo refina, pero que el óptimo aleatorio llegue casi al mismo lugar sugiere que **simplemente sumar los mapas ya es una buena técnica de fusión**, y que aprender una combinación inicializada al azar no lleva a resultados significativamente distintos.

> Gotcha del registro espacial: al fusionar un VGG-16 en ReLU5\_3 con un VGG-M en ReLU5 las resoluciones no coinciden ($14\times14$ contra $13\times13$), así que hubo que **rellenar la salida menor con una fila y una columna de ceros**. Una arquitectura heterogénea exige un parche de padding para que el índice $(i,j)$ signifique lo mismo en ambos tensores.

---

## Dónde fusionar

Con la convolución fijada como operador e inicializada con identidades, el paper barre la capa de fusión (UCF-101 split 1, dos VGG-M):

| Capa(s) de fusión | Accuracy | #capas | #parámetros |
|---|---|---|---|
| ReLU2 | 82.25% | 11 | 91.90M |
| ReLU3 | 83.43% | 12 | 93.08M |
| ReLU4 | 82.55% | 13 | 95.48M |
| **ReLU5** | **85.96%** | **14** | **97.57M** |
| **ReLU5 + FC8** | **86.04%** | **17** | **181.68M** |
| ReLU3 + ReLU5 + FC6 | 81.55% | 17 | 190.06M |

**Fusionar temprano degrada, y degrada mucho.** ReLU2, ReLU3 y ReLU4 quedan entre 2.5 y 3.7 puntos por debajo de ReLU5, y ni siquiera de forma monótona (ReLU4 es *peor* que ReLU3). La intuición: en capas bajas los features de las dos corrientes son de naturalezas demasiado distintas —bordes y texturas de apariencia contra gradientes de campos de flujo— para que una combinación lineal canal a canal signifique algo; y fusionar temprano **destruye una jerarquía completa**, porque desde el punto de fusión hay una sola torre. El óptimo es la **última capa convolucional**, donde los features son ya altamente informativos pero todavía conservan información gruesa de localización. En las capas densas todos los operadores rinden peor, con el mismo orden relativo, y entre ellas FC8 es mejor que FC7 y FC6. La explicación de los autores: en ReLU5 **todavía existen las correspondencias espaciales**; en las FC ya fueron colapsadas.

El trade-off de parámetros es la parte más útil para un ingeniero. **ReLU5 + FC8 gana 0.08 puntos y cuesta 84M de parámetros más** (181.68M contra 97.57M), porque mantiene **ambas torres**: una convertida en híbrida espacio-temporal y la otra puramente espacial. Consigue el registro píxel a píxel pero no ahorra nada. Y hay un detalle contraintuitivo: fusionar en capas convolucionales tempranas **no ahorra tanto como uno esperaría** (91.90M contra 97.57M), porque **la abrumadora mayoría de los parámetros de [VGG](/papers/vggnet-simonyan-2014) vive en FC6/FC7/FC8**; truncar una torre en ReLU2 o en ReLU5 elimina aproximadamente el mismo bloque denso. Ahorrar 5.7M sacrificando 3.7 puntos no tiene sentido. Fusionar en múltiples capas tampoco: ReLU3 + ReLU5 + FC6 es el peor resultado (81.55%) con la mayor cantidad de parámetros. Más fusión no es mejor fusión.

---

## Fusión temporal: 2D pooling, 3D pooling y 3D conv

La entrada de la capa de fusión temporal es $\mathbf{x} \in \mathbb{R}^{H \times W \times T \times D}$, apilando mapas espaciales sobre $t = 1 \ldots T$. Las tres opciones evaluadas:

- **2D pooling**: ignora el tiempo, hace *pooling* espacial por muestra y promedia predicciones después. Es lo que hace el two-stream original.
- **3D pooling**: max-pooling sobre un cubo $3\times3\times3$ (espacial × temporal) sobre los canales correspondientes apilados, sin *pooling* entre canales distintos. Da invarianza a pequeños desplazamientos de los features en el tiempo.
- **3D conv + 3D pooling**: primero convoluciona el tensor 4D con filtros $\mathbf{f} \in \mathbb{R}^{W'' \times H'' \times T'' \times D \times D'}$ y **después** aplica el *pooling* 3D.

Por qué el orden importa: el *pooling* 3D da invarianza pero es un operador fijo y sin parámetros, capaz de registrar la **presencia** de un feature en algún lugar del vecindario, no su **evolución**. La convolución 3D aprende combinaciones ponderadas en un vecindario espacio-temporal local, y el paper nombra los dos casos canónicos: **ponderar centralmente la muestra temporal central**, o **diferenciar en el tiempo o en el espacio**. Ese segundo caso es la clave: un filtro que aproxima $\partial/\partial t$ detecta *cambio* de feature, no presencia. Es la diferencia entre "hubo movimiento periódico en algún momento" y "el movimiento se aceleró y luego se detuvo". Hacer el *pooling* después preserva esa capacidad y solo entonces le agrega invarianza posicional.

El muestreo temporal crea dos escalas deliberadas: la **fina** en la entrada de la corriente temporal ($t \pm L/2$ con $L=10$), que captura primitivas de movimiento como "el trazado de una flecha", y la **gruesa** en la capa de fusión, que recibe $T$ chunks separados por $\tau$ fotogramas y las pone en contexto: "dibujar una flecha, tensar el arco, disparar". El campo receptivo temporal total es $T \times L$, y $\tau < L$ produce entradas solapadas mientras $\tau \ge L$ produce features no solapados.

Resultados (VGG-16 espacial + VGG-M temporal, split 1):

| Fusión | Pooling | Capas | UCF-101 | HMDB-51 |
|---|---|---|---|---|
| 2D Conv | 2D | ReLU5 + | 89.35% | 56.93% |
| 2D Conv | 3D | ReLU5 + | 89.64% | 57.58% |
| **3D Conv** | **3D** | **ReLU5 +** | **90.40%** | **58.63%** |

El argumento se construye incrementalmente: pasar de 2D a 3D *pooling* gana (+0.29 UCF, +0.65 HMDB) y añadir el filtro 3D gana otro salto (+0.76 UCF, +1.05 HMDB). Que HMDB gane más es consistente con su naturaleza: es un dataset donde el contexto de escena ayuda menos y la dinámica importa más.

La arquitectura final fusiona **desde la corriente temporal hacia la espacial en ReLU5**, con 3D Conv seguida de 3D pooling, convirtiendo la espacial en espacio-temporal; **no trunca la corriente temporal** y usa las pérdidas de ambas para entrenar, promediando sus predicciones en test (eso es el "+" de la tabla). El kernel de fusión es $3 \times 3 \times 3 \times 1024 \times 512$ con $T=5$: los 1024 canales de entrada salen de concatenar los ReLU5 de ambas corrientes ($512+512$) y los 512 de salida coinciden con la entrada de FC6.

Dos detalles de [inicialización](/fundamentos/transfer-learning) que valen su peso en oro. Las activaciones de la ConvNet temporal en la última capa convolucional son **aproximadamente 3 veces menores** que las de la espacial, así que la identidad temporal de $\mathbf{f}$ se inicializa con un **factor 3 más alto** para que la fusión no quede dominada por la apariencia (la parte espacio-temporal usa una gaussiana $3\times3\times3$ con $\sigma=1$). Y **no se fusiona en la capa de predicción durante el entrenamiento**, porque sesgaría la pérdida hacia la arquitectura temporal: la espacio-temporal necesita más tiempo para adaptarse a los features fusionados.

---

## Resultados

Promedios sobre los tres splits estándar:

| Método | UCF-101 | HMDB-51 |
|---|---|---|
| Spatiotemporal ConvNet (Karpathy et al.) | 65.4% | — |
| [LRCN](/papers/lrcn-donahue-2015) (Donahue et al.) | 82.9% | — |
| Composite LSTM (Srivastava et al.) | 84.3% | 44.0% |
| [C3D](/papers/c3d-tran-2015) (Tran et al.) | 85.2% | — |
| [Two-Stream ConvNet](/papers/two-stream-simonyan-2014) (VGG-M) | 88.0% | 59.4% |
| Factorized ConvNet (Sun et al.) | 88.1% | 59.1% |
| Two-Stream Conv Pooling (Ng et al.) | 88.2% | — |
| Two-Stream ConvNet (VGG-16, Wang et al.) | 91.4% | 58.5% |
| Two-Stream ConvNet (VGG-16, reimplementación de los autores) | 91.7% | 58.7% |
| **Este paper (S: VGG-16, T: VGG-M)** | **90.8%** | **62.1%** |
| **Este paper (S y T: VGG-16, una torre tras fusión)** | **91.8%** | **64.6%** |
| **Este paper (S y T: VGG-16, dos torres)** | **92.5%** | **65.4%** |

Contra el two-stream original la mejora es de **~3 puntos en ambos datasets** con backbone mixto, y de **4.5 (UCF) y 6.0 (HMDB)** con VGG-16 en las dos corrientes. Contra Two-Stream Conv Pooling de Ng et al. —que hace *pooling* temporal sobre 120 fotogramas y llega a 88.2%, u 88.6% con LSTM— el 92.5% se obtiene con una huella temporal mucho menor. Y el dato de ingeniería: **una sola torre tras la fusión da 91.8% contra 92.5% con dos**; 0.7 puntos es el precio de la simplicidad.

La comparación de las corrientes entrenadas por separado (split 1) esconde el hallazgo más citado del paper:

| Corriente | UCF-101 VGG-M-2048 | UCF-101 VGG-16 | HMDB-51 VGG-M-2048 | HMDB-51 VGG-16 |
|---|---|---|---|---|
| Espacial | 74.22% | 82.61% | 36.77% | 47.06% |
| Temporal | 82.34% | 86.25% | 51.50% | 55.23% |
| Fusión tardía (promedio) | 85.94% | 90.62% | 54.90% | 58.17% |

La **asimetría de preentrenamiento** es brutal: la corriente espacial gana **+8.11 (UCF) y +10.29 (HMDB)** al pasar de VGG-M a VGG-16, mientras la temporal gana menos de la mitad (**+3.91 y +3.73**). La corriente de apariencia hereda directamente el beneficio de las arquitecturas de imagen y de ImageNet; la de flujo, mucho menos. De hecho, la temporal **también se inicializa desde ImageNet**, pero con una justificación reveladora: lo hacen porque acelera el entrenamiento sin pérdida de desempeño respecto de entrenar desde cero. ImageNet le da a la corriente temporal **convergencia, no precisión**.

Combinado con iDT:

| Método | UCF-101 | HMDB-51 |
|---|---|---|
| iDT + Fisher Vector de alta dimensión (Peng et al.) | 87.9% | 61.1% |
| C3D + iDT | 90.4% | — |
| TDD + iDT (Wang et al.) | 91.5% | 65.9% |
| **Este paper + iDT (S: VGG-16, T: VGG-M)** | **92.5%** | **67.3%** |
| **Este paper + iDT (S y T: VGG-16)** | **93.5%** | **69.2%** |

El resultado intelectualmente más interesante es justamente este: **sigue habiendo una mejora sustancial al sumar features artesanales** (+1.0 en UCF y +3.8 en HMDB sobre el mejor modelo puro). Los autores lo llaman "intrigante" y sospechan que la diferencia podría desaparecer con muchísimos más datos, pero que por ahora indica dónde debe apuntar la investigación futura. Un año más tarde, Kinetics confirmó la sospecha.

---

## Limitaciones

- **Sigue dependiendo del flujo óptico precomputado.** Se calcula antes del entrenamiento con Brox et al. o TV-L1, se almacena como JPEG y se recortan los desplazamientos mayores a 20 píxeles: preprocesamiento costoso fuera de la red, almacenamiento no trivial y un pipeline no entrenable de punta a punta, con pérdida de información por clipping y compresión. La arquitectura *fusiona* mejor, pero no elimina la dependencia.
- **La fusión sigue siendo tardía en términos temporales.** Espacialmente el paper mueve la fusión de la capa 16 a la 14, un avance real. Pero el modelado temporal ocurre **solo en la capa alta de fusión**: las torres procesan cada chunk de forma independiente hasta ReLU5. Es "features 2D por snippet, luego mezcla temporal", no una red espacio-temporal completa.
- **Ventana temporal limitada.** El campo receptivo temporal es $T \times L$ con $T=5$ y $L=10$: entre **15 y 50 fotogramas** según el $\tau$ muestreado, o entre 0.6 y 2 segundos a 25 fps. El propio ejemplo del arco y la flecha apenas cabe.
- **La fusión temprana no queda descartada, solo no lograda.** Que ReLU2/3/4 degraden no prueba que fusionar temprano sea intrínsecamente malo: prueba que **este operador** (convolución $1\times1$ con inicialización identidad, truncando una torre) no funciona temprano.
- **Backpropagation parcial.** Solo se retropropaga hasta la capa de fusión inyectada, porque la retropropagación completa no dio mejoras. Consecuencia: las corrientes no se **co-adaptan** de verdad, y la promesa de "aprender correspondencias que minimizan una pérdida conjunta" se cumple en la capa de fusión, no en toda la jerarquía.
- **Datasets demasiado pequeños o demasiado ruidosos.** El paper cierra con esa advertencia y con razón: las ablaciones críticas se corrieron sobre **un solo split de UCF-101** y las diferencias que deciden el diseño son del orden de 0.1 a 0.8 puntos. Los autores piden explícitamente tratar algunas conclusiones con cautela. Es el diagnóstico que I3D convertiría en su premisa.

---

## Por qué importa hoy

Su lugar en el mapa es preciso. La [Clase 38](/clases/clase-38) abre con la figura de "Overview" de [I3D](/papers/i3d-carreira-2017), que reordena el zoo de arquitecturas de video en cinco familias: (a) LSTM, (b) 3D-ConvNet, (c) Two-Stream, (d) 3D-Fused Two-Stream, (e) Two-Stream I3D. **Este paper es la (d)**, y en la tabla de la [teoría de la clase](/clases/clase-38/teoria) es la fila de **39M de parámetros con 83.2 / 85.8 / 89.3 en UCF-101** (RGB / flujo / RGB+flujo). La reimplementación de I3D es una traducción fiel: 5 fotogramas RGB muestreados cada 10 más sus snippets de flujo, grillas de features de $5\times7\times7$, una convolución 3D de $3\times3\times3$ con 512 canales, un max-pooling 3D de $3\times3\times3$ y una capa densa. La clase lo muestra como una caja en un diagrama; acá está lo que hay dentro.

La familia (e), Two-Stream I3D, la supera con **25M de parámetros**, y el *por qué* es el diagnóstico exacto de la limitación de este trabajo: 3D-Fused tiene features 2D en toda la jerarquía y solo mezcla en el tiempo al final. La convolución 3D de fusión de Feichtenhofer es, en retrospectiva, **una única capa espacio-temporal encima de una pila enteramente espacial**. I3D infla toda la pila.

Y ese es el punto de contacto más importante con el eje de la clase. La asimetría de preentrenamiento documentada arriba —espacial +8.11/+10.29 contra temporal +3.91/+3.73, ImageNet dándole a la corriente temporal convergencia y no precisión, activaciones 3× menores que obligan al hack del factor 3— es un problema que este paper mide sin poder resolver. Peor aún: la **capa de fusión 3D no puede heredar nada**; son identidades para los canales y una gaussiana para la parte $3\times3\times3$. Cada parámetro nuevo destinado a modelar tiempo hay que aprenderlo desde cero con 100 videos por clase, de ahí el énfasis obsesivo en aumentación (dropout 0.85, jitter de aspect-ratio de ±25%, muestreo aleatorio de $\tau \in [1,10]$ como aumentación de escala temporal) y la advertencia de que entrenar 3D ConvNets es todavía más propenso al sobreajuste que fusionar two-stream ConvNets. El [inflado de convoluciones](/fundamentos/inflado-de-convoluciones) de I3D, con su *boring-video fixed point*, es exactamente lo que hace desaparecer esa asimetría: permite que **toda** la red 3D arranque con pesos de ImageNet y elimina la necesidad de confinar el modelado temporal a una capa delgada al final. Feichtenhofer tenía la arquitectura; le faltaba Kinetics.

El linaje más directo, sin embargo, va del mismo primer autor. En [SlowFast](/papers/slowfast-feichtenhofer-2019) (2019) las dos vías ya no son RGB y flujo óptico sino **dos frecuencias de muestreo del mismo RGB**: una vía *Slow* de baja tasa de fotogramas y alta capacidad para semántica espacial, y una vía *Fast* de alta tasa y baja capacidad para movimiento. Lo que las une son **conexiones laterales que fusionan la vía Fast en la Slow en múltiples etapas de la jerarquía**: la descendiente directa de esta idea, con la misma convicción de que dos torres deben intercambiar información en capas intermedias y no solo en el softmax, y heredando incluso el requisito de correspondencia espacial (por eso las conexiones laterales deben reconciliar formas de tensor). Y resuelve la dependencia del flujo prescindiendo de él: la vía Fast *es* el detector de movimiento aprendido.

> Tres lecciones transferibles para quien monte un clasificador multimodal con datos escasos. **Dónde fusionar importa más que cómo**: mover la fusión de la última capa a la penúltima ganó más que elegir entre suma, máximo o convolución, y fusionar demasiado temprano costó 3.7 puntos. **La fusión intermedia es un ahorro de parámetros, no solo una mejora de precisión**: truncar una torre eliminó casi la mitad de los parámetros a igualdad de accuracy, porque los parámetros viven en las capas densas. Y **la inicialización identidad de un módulo de fusión es una técnica general**: hacer que un módulo nuevo empiece siendo la operación trivial y solo pueda mejorar desde ahí añade capacidad sin desestabilizar un modelo ya entrenado, el mismo principio que subyace a las conexiones residuales, a los adapters y a LoRA.

---

## Notas y enlaces

- Código original en `github.com/feichtenhofer/twostreamfusion`, implementado en MatConvNet. Sin batch normalization; fine-tuning con batch de 96 videos, learning rate inicial $10^{-3}$ (o $5\times10^{-4}$ para VGG-16) reducido 10× cuando la accuracy de validación se satura. Backbones: [VGG-M-2048 y VGG-16](/papers/vggnet-simonyan-2014), ambos preentrenados en ImageNet ([transfer learning](/fundamentos/transfer-learning)).
- Antecedente directo: [Two-Stream (Simonyan y Zisserman, 2014)](/papers/two-stream-simonyan-2014), cuya sección de limitaciones ya anticipaba este trabajo. La [Clase 36](/clases/clase-36) construye el arco desde la 2D CNN por fotograma hasta el [flujo óptico](/fundamentos/flujo-optico) y el [reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones).
- Contraparte de la familia (b): [C3D (Tran et al., 2015)](/papers/c3d-tran-2015), que aprende movimiento con filtros espacio-temporales desde el RGB crudo. Este paper es el primer híbrido serio entre (b) y (c): conserva la entrada de flujo de (c) y le añade el operador de (b), pero solo en la capa de fusión.
- Continuación: [I3D (Carreira y Zisserman, 2017)](/papers/i3d-carreira-2017) y el [inflado de convoluciones](/fundamentos/inflado-de-convoluciones). Descendiente conceptual: [SlowFast (Feichtenhofer et al., 2019)](/papers/slowfast-feichtenhofer-2019). Alternativa contemporánea al problema de la escala temporal: [TSN (Wang et al., 2016)](/papers/tsn-wang-2016), que ataca el largo alcance con muestreo disperso y consenso de segmentos en lugar de convoluciones 3D.
- El propio grupo extendió el catálogo de operadores en *Spatiotemporal Residual Networks* (NIPS 2016) y *Spatiotemporal Multiplier Networks* (CVPR 2017), donde conexiones residuales entre corrientes reemplazan la capa de fusión única.
