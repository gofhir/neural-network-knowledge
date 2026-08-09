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
Dos años después de [Two-Stream](/papers/two-stream-simonyan-2014) y desde el mismo grupo (Zisserman firma ambos), este paper hace la pregunta que la arquitectura original dejó abierta: **si en lugar de promediar los softmax fusionamos en medio de la red, ¿qué pasa?** La respuesta es un estudio de ablación sobre tres ejes —**cómo** fusionar, **dónde** y **cómo fusionar temporalmente**— del que después se ensambla la red final. Las tres conclusiones: se puede fusionar en una capa convolucional **sin pérdida de desempeño y con casi la mitad de los parámetros** (97.58M contra 181.42M); el punto óptimo es la **última capa convolucional**, no antes; y el *pooling* sobre vecindarios espacio-temporales mejora todavía más. En números: **92.5% en UCF-101 y 65.4% en HMDB-51** con VGG-16 en ambas corrientes, contra 88.0% / 59.4% del original, y **93.5% / 69.2%** con iDT, el mejor resultado publicado en 2016. Para la [Clase 38](/clases/clase-38) el paper es la familia **(d) 3D-Fused Two-Stream** de la comparativa canónica de [I3D](/papers/i3d-carreira-2017): la caja que el diagrama muestra pero no explica.
{{< /paper-card >}}

---

## Contexto: la fusión tardía deja algo sobre la mesa

El [two-stream original](/papers/two-stream-simonyan-2014) descompone el video en fotogramas RGB y pilas de $L=10$ campos de [flujo óptico](/fundamentos/flujo-optico); cada corriente se entrena **por su cuenta** y la predicción final promedia los softmax. El primero de los dos defectos que señala este paper conviene decirlo sin eufemismos: **la red nunca aprende "qué se mueve dónde"**. Fusionar en el softmax combina dos distribuciones sobre 101 clases; nada permite que un parámetro entrenable represente la conjunción *"movimiento periódico de mano EN la ubicación de la boca"*.

El ejemplo del paper es el más limpio posible: **distinguir "cepillarse los dientes" de "cepillarse el pelo"**. En ambos hay una mano moviéndose periódicamente en algún lugar. La corriente temporal reconoce ese movimiento pero no puede decir *dónde* en términos semánticos; la espacial reconoce la ubicación pero no el movimiento. Solo su combinación **en la misma posición espacial** discrimina la acción. Al apilar los canales, en cambio, los filtros siguientes **pueden aprender la correspondencia entre los canales apropiados** —los de las áreas faciales con el del campo de movimiento— como pesos de una convolución.

El segundo defecto es de escala temporal: promediar 25 predicciones independientes **no modela evolución**. No es lo mismo que aprender que "dibujar una flecha, tensar el arco, disparar" es una secuencia con orden. Es la debilidad que la [Clase 36](/clases/clase-36) señala en el [reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones).

> El contexto de 2016 era incómodo: los mejores números seguían saliendo de combinar ConvNets con Fisher Vectors artesanales. Parte de la explicación es de escala: UCF-101 tiene 100 ejemplos por clase contra los 1000 de ImageNet.

---

## Cómo fusionar: los operadores

Cada operador toma $\mathbf{x}^a, \mathbf{x}^b \in \mathbb{R}^{H \times W \times D}$ y actúa índice por índice sobre $(i,j)$: eso impone una **restricción dura de correspondencia espacial**, porque sin registro píxel a píxel se combinan features de lugares distintos.

| Operador | Definición | Salida | Correspondencia entre canales |
|---|---|---|---|
| Suma | $y_{i,j,d} = x^a_{i,j,d} + x^b_{i,j,d}$ | $\mathbb{R}^{H\times W\times D}$ | **Arbitraria**, fija $d$-a-$d$ |
| Máximo | $y_{i,j,d} = \max\{x^a_{i,j,d}, x^b_{i,j,d}\}$ | $\mathbb{R}^{H\times W\times D}$ | Arbitraria, y descarta una corriente por posición |
| Concatenación | apila los canales | $\mathbb{R}^{H\times W\times 2D}$ | **Ninguna**; la aprenden las capas siguientes |
| Convolución | $\mathbf{y}^{\text{cat}} * \mathbf{f} + b$, con $\mathbf{f}\in\mathbb{R}^{1\times1\times2D\times D}$ | $\mathbb{R}^{H\times W\times D}$ | **Aprendida** en la misma posición de píxel |
| Bilineal | $\sum_{i,j} \mathbf{x}^{a\top}_{i,j}\mathbf{x}^{b}_{i,j}$ | $\mathbb{R}^{D^2}$ | **Completa** y multiplicativa, pero sin información espacial |

La arbitrariedad de la suma merece énfasis: **la numeración de los canales no significa nada**, así que no hay razón para emparejar el canal 37 de la red espacial con el 37 de la temporal; el entrenamiento puede *explotar* esa correspondencia, pero no *elegirla*. La concatenación tampoco elige y paga un costo estructural: la capa siguiente recibe $2D$ canales y, siendo FC6, **duplica los parámetros de la primera capa densa**. La bilineal sí la resuelve, pero explota a $D^2$ (262144 para $D=512$) y **marginaliza la información espacial**.

La convolución gana por una razón elegante: su filtro actúa **puramente en la dimensión de canales** y **contiene a la suma como caso particular** —si $\mathbf{f}$ se aprende como dos matrices identidad concatenadas, el canal $i$ de una red se combina solo con el canal $i$ de la otra, vía suma. Es **la generalización aprendible de la suma**.

Resultados con dos VGG-M-2048, fusión en la salida de **ReLU5** (mejor que la salida no rectificada de conv5), una sola torre después, UCF-101 split 1:

| Fusión | Capa | Accuracy | #capas | #parámetros |
|---|---|---|---|---|
| Suma (reportado en two-stream original) | Softmax | 85.6% | 16 | 181.42M |
| Suma (reimplementación de los autores) | Softmax | 85.94% | 16 | 181.42M |
| Máximo | ReLU5 | 82.70% | 13 | 97.31M |
| Concatenación | ReLU5 | 83.53% | 13 | 172.81M |
| Bilineal | ReLU5 | 85.05% | 10 | 6.61M + SVM |
| Suma | ReLU5 | 85.20% | 13 | 97.31M |
| **Convolución** | **ReLU5** | **85.96%** | **14** | **97.58M** |

La convolución en ReLU5 **iguala al promedio de softmax con la mitad de los parámetros**, pagando 0.27M extra sobre la suma. La decisión crítica es la **inicialización con matrices identidad**, para que la capa *empiece* sumando. Y acá viene el dato honesto: la identidad rinde **85.96% contra 85.59%** con ruido gaussiano —prácticamente igual— **pero converge mucho más rápido**. Que el óptimo aleatorio llegue casi al mismo lugar sugiere que **sumar los mapas ya es casi óptimo**.

> Gotcha del registro: al fusionar un VGG-16 en ReLU5\_3 con un VGG-M en ReLU5 las resoluciones no coinciden ($14\times14$ contra $13\times13$), así que hubo que **rellenar la salida menor con una fila y una columna de ceros** para que el índice $(i,j)$ signifique lo mismo en ambos tensores.

---

## Dónde fusionar

| Capa(s) de fusión | Accuracy | #capas | #parámetros |
|---|---|---|---|
| ReLU2 | 82.25% | 11 | 91.90M |
| ReLU3 | 83.43% | 12 | 93.08M |
| ReLU4 | 82.55% | 13 | 95.48M |
| **ReLU5** | **85.96%** | **14** | **97.57M** |
| **ReLU5 + FC8** | **86.04%** | **17** | **181.68M** |
| ReLU3 + ReLU5 + FC6 | 81.55% | 17 | 190.06M |

**Fusionar temprano degrada, y degrada mucho:** ReLU2, ReLU3 y ReLU4 quedan entre 2.5 y 3.7 puntos por debajo de ReLU5, y ni siquiera de forma monótona (ReLU4 es *peor* que ReLU3). En capas bajas los features de las dos corrientes son demasiado distintos —bordes y texturas contra gradientes de campos de flujo— para que una combinación lineal canal a canal signifique algo, y fusionar temprano **destruye una jerarquía completa** porque desde ahí hay una sola torre. En las capas densas todos los operadores rinden peor, y entre ellas FC8 supera a FC7 y FC6: en ReLU5 **todavía existen las correspondencias espaciales**, en las FC ya fueron colapsadas.

El trade-off de parámetros es la parte más útil para un ingeniero. **ReLU5 + FC8 gana 0.08 puntos y cuesta 84M de parámetros más**, porque mantiene **ambas torres**. Y hay un detalle contraintuitivo: fusionar temprano **no ahorra tanto como uno esperaría** (91.90M contra 97.57M), porque **la mayoría de los parámetros de [VGG](/papers/vggnet-simonyan-2014) vive en FC6/FC7/FC8** y truncar una torre en ReLU2 o en ReLU5 borra casi el mismo bloque denso. Fusionar en varias capas tampoco ayuda: ReLU3 + ReLU5 + FC6 es el peor resultado con la mayor cantidad de parámetros. Más fusión no es mejor fusión.

---

## Fusión temporal: 2D pooling, 3D pooling y 3D conv

La entrada es $\mathbf{x} \in \mathbb{R}^{H \times W \times T \times D}$, apilando mapas espaciales sobre $t = 1 \ldots T$. Tres opciones: **2D pooling** (ignora el tiempo y promedia predicciones después, lo que hace el two-stream original); **3D pooling** (max-pooling sobre un cubo $3\times3\times3$ espacial × temporal); y **3D conv + 3D pooling**, que primero convoluciona el tensor 4D con filtros $\mathbf{f} \in \mathbb{R}^{W'' \times H'' \times T'' \times D \times D'}$ y **después** aplica el *pooling*.

Por qué el orden importa: el *pooling* 3D da invarianza pero es un operador fijo y sin parámetros, capaz de registrar la **presencia** de un feature en el vecindario, no su **evolución**. La convolución 3D aprende combinaciones ponderadas en un vecindario espacio-temporal local, y el paper nombra dos casos canónicos: **ponderar centralmente la muestra temporal central** o **diferenciar en el tiempo o en el espacio**. El segundo es la clave: un filtro que aproxima $\partial/\partial t$ detecta *cambio* de feature, no presencia —la diferencia entre "hubo movimiento periódico" y "el movimiento se aceleró y luego se detuvo". Hacer el *pooling* después preserva esa capacidad y solo entonces agrega invarianza posicional.

El muestreo crea dos escalas: la **fina** en la entrada de la corriente temporal ($t \pm L/2$, con $L=10$), que captura primitivas como "el trazado de una flecha", y la **gruesa** en la capa de fusión, que recibe $T$ chunks separados por $\tau$ fotogramas. El campo receptivo temporal total es $T \times L$.

| Fusión | Pooling | Capas | UCF-101 | HMDB-51 |
|---|---|---|---|---|
| 2D Conv | 2D | ReLU5 + | 89.35% | 56.93% |
| 2D Conv | 3D | ReLU5 + | 89.64% | 57.58% |
| **3D Conv** | **3D** | **ReLU5 +** | **90.40%** | **58.63%** |

(VGG-16 espacial + VGG-M temporal, split 1.) Pasar de 2D a 3D *pooling* gana +0.29 UCF y +0.65 HMDB; añadir el filtro 3D gana otro salto de +0.76 y +1.05. Que HMDB gane más es consistente: ahí el contexto de escena ayuda menos y la dinámica importa más.

La arquitectura final fusiona **desde la corriente temporal hacia la espacial en ReLU5**, con 3D Conv seguida de 3D pooling; **no trunca la corriente temporal** y usa las pérdidas de ambas (eso es el "+" de la tabla). El kernel es $3 \times 3 \times 3 \times 1024 \times 512$ con $T=5$: los 1024 canales de entrada salen de concatenar los ReLU5 de ambas corrientes y los 512 de salida coinciden con la entrada de FC6.

Dos detalles de [inicialización](/fundamentos/transfer-learning) que valen su peso en oro. Las activaciones temporales en la última capa convolucional son **unas 3 veces menores** que las espaciales, así que la identidad temporal de $\mathbf{f}$ se inicializa con un **factor 3 más alto** para que la fusión no quede dominada por la apariencia. Y **no se fusiona en la capa de predicción durante el entrenamiento**: sesgaría la pérdida hacia la arquitectura temporal, porque la espacio-temporal necesita más tiempo para adaptarse a los features fusionados.

---

## Resultados

Promedios sobre los tres splits estándar:

| Método | UCF-101 | HMDB-51 |
|---|---|---|
| Spatiotemporal ConvNet (Karpathy et al.) | 65.4% | — |
| [LRCN](/papers/lrcn-donahue-2015) (Donahue et al.) | 82.9% | — |
| [C3D](/papers/c3d-tran-2015) (Tran et al.) | 85.2% | — |
| [Two-Stream ConvNet](/papers/two-stream-simonyan-2014) (VGG-M) | 88.0% | 59.4% |
| Two-Stream Conv Pooling (Ng et al.) | 88.2% | — |
| Two-Stream ConvNet (VGG-16, Wang et al.) | 91.4% | 58.5% |
| Two-Stream ConvNet (VGG-16, reimplementado por los autores) | 91.7% | 58.7% |
| **Este paper (S: VGG-16, T: VGG-M)** | **90.8%** | **62.1%** |
| **Este paper (S y T: VGG-16, una torre tras fusión)** | **91.8%** | **64.6%** |
| **Este paper (S y T: VGG-16, dos torres)** | **92.5%** | **65.4%** |

Contra el two-stream original la mejora es de **~3 puntos en ambos datasets** con backbone mixto, y de **4.5 (UCF) y 6.0 (HMDB)** con VGG-16 en las dos corrientes. Contra Two-Stream Conv Pooling de Ng et al. —*pooling* temporal sobre 120 fotogramas, 88.2%— el 92.5% se obtiene con una huella temporal mucho menor. Y el dato de ingeniería: **una sola torre tras la fusión da 91.8% contra 92.5% con dos**; 0.7 puntos es el precio de la simplicidad.

Las corrientes entrenadas por separado (split 1) esconden el hallazgo más citado:

| Corriente | UCF VGG-M-2048 | UCF VGG-16 | HMDB VGG-M-2048 | HMDB VGG-16 |
|---|---|---|---|---|
| Espacial | 74.22% | 82.61% | 36.77% | 47.06% |
| Temporal | 82.34% | 86.25% | 51.50% | 55.23% |
| Fusión tardía (promedio) | 85.94% | 90.62% | 54.90% | 58.17% |

La **asimetría de preentrenamiento** es brutal: la corriente espacial gana **+8.11 (UCF) y +10.29 (HMDB)** al pasar de VGG-M a VGG-16, mientras la temporal gana menos de la mitad (**+3.91 y +3.73**). La de apariencia hereda el beneficio de las arquitecturas de imagen; la de flujo, mucho menos. La temporal **también se inicializa desde ImageNet**, con una justificación reveladora: acelera el entrenamiento sin mejorar el desempeño frente a entrenar desde cero. ImageNet le da **convergencia, no precisión**.

| Método | UCF-101 | HMDB-51 |
|---|---|---|
| iDT + Fisher Vector de alta dimensión (Peng et al.) | 87.9% | 61.1% |
| C3D + iDT | 90.4% | — |
| TDD + iDT (Wang et al.) | 91.5% | 65.9% |
| **Este paper + iDT (S: VGG-16, T: VGG-M)** | **92.5%** | **67.3%** |
| **Este paper + iDT (S y T: VGG-16)** | **93.5%** | **69.2%** |

Lo más interesante es esto último: **sumar features artesanales todavía mejora de forma sustancial** (+1.0 en UCF y +3.8 en HMDB sobre el mejor modelo puro). Los autores lo llaman "intrigante" y sospechan que la diferencia desaparecería con muchísimos más datos. Un año más tarde, Kinetics lo confirmó.

---

## Limitaciones

- **Sigue dependiendo del flujo óptico precomputado**, con Brox et al. o TV-L1, almacenado como JPEG y con clipping a 20 píxeles: preprocesamiento costoso fuera de la red, pipeline no entrenable de punta a punta y pérdida de información. La arquitectura *fusiona* mejor, pero no elimina la dependencia.
- **La fusión sigue siendo tardía en lo temporal.** El modelado del tiempo ocurre **solo en la capa alta de fusión**: las torres procesan cada chunk de forma independiente hasta ReLU5. Es "features 2D por snippet, luego mezcla temporal", no una red espacio-temporal completa.
- **Ventana temporal limitada.** El campo receptivo es $T \times L$ con $T=5$ y $L=10$: entre **15 y 50 fotogramas**, o 0.6 a 2 segundos a 25 fps. El propio ejemplo del arco y la flecha apenas cabe.
- **La fusión temprana no queda descartada, solo no lograda.** Que ReLU2/3/4 degraden prueba que **este operador** (convolución $1\times1$ inicializada en identidad, truncando una torre) no funciona temprano, no que fusionar temprano sea malo en sí. Y como solo se retropropaga hasta la capa de fusión, las corrientes tampoco se **co-adaptan** de verdad.
- **Datasets demasiado pequeños o ruidosos.** Las ablaciones críticas se corrieron sobre **un solo split de UCF-101** y las diferencias que deciden el diseño son de 0.1 a 0.8 puntos; los autores piden tratar algunas conclusiones con cautela. Es el diagnóstico que I3D convertiría en su premisa.

---

## Por qué importa hoy

La [Clase 38](/clases/clase-38) abre con la figura de "Overview" de [I3D](/papers/i3d-carreira-2017), que reordena el zoo de arquitecturas de video en cinco familias: (a) LSTM, (b) 3D-ConvNet, (c) Two-Stream, (d) 3D-Fused Two-Stream, (e) Two-Stream I3D. **Este paper es la (d)**, y en la tabla de la [teoría de la clase](/clases/clase-38/teoria) es la fila de **39M de parámetros con 83.2 / 85.8 / 89.3 en UCF-101** (RGB / flujo / RGB+flujo). La reimplementación de I3D es fiel: 5 fotogramas RGB muestreados cada 10 más sus snippets de flujo, grillas de $5\times7\times7$, una convolución 3D de $3\times3\times3$ con 512 canales, max-pooling 3D y una capa densa. La clase lo muestra como una caja en un diagrama; acá está lo que hay dentro.

La familia (e), Two-Stream I3D, la supera con **25M de parámetros**, y el *por qué* es el diagnóstico de la limitación de este trabajo: 3D-Fused tiene features 2D en toda la jerarquía y solo mezcla en el tiempo al final. Su convolución 3D de fusión es, en retrospectiva, **una única capa espacio-temporal encima de una pila enteramente espacial**; I3D infla toda la pila.

Ahí está el punto de contacto más importante con el eje de la clase. La asimetría de preentrenamiento —espacial +8.11/+10.29 contra temporal +3.91/+3.73, ImageNet dándole a la corriente temporal convergencia y no precisión, activaciones 3× menores que obligan al hack del factor 3— es un problema que el paper mide sin poder resolver. Peor aún: la **capa de fusión 3D no puede heredar nada**. Cada parámetro destinado a modelar tiempo hay que aprenderlo desde cero con 100 videos por clase, de ahí el énfasis obsesivo en aumentación (dropout 0.85, jitter de aspect-ratio de ±25%, $\tau \in [1,10]$ aleatorio) y la advertencia sobre el sobreajuste de las 3D ConvNets. El [inflado de convoluciones](/fundamentos/inflado-de-convoluciones), con su *boring-video fixed point*, es exactamente lo que hace desaparecer esa asimetría: permite que **toda** la red 3D arranque con pesos de ImageNet y elimina la necesidad de confinar el modelado temporal a una capa delgada al final. Feichtenhofer tenía la arquitectura; le faltaba Kinetics.

El linaje más directo va del mismo primer autor. En [SlowFast](/papers/slowfast-feichtenhofer-2019) (2019) las dos vías ya no son RGB y flujo sino **dos frecuencias de muestreo del mismo RGB**: una *Slow* de baja tasa y alta capacidad para semántica espacial, y una *Fast* de alta tasa y baja capacidad para movimiento. Lo que las une son **conexiones laterales que fusionan la vía Fast en la Slow en múltiples etapas de la jerarquía**: la descendiente directa de esta idea de fusión intermedia. Y elimina la dependencia del flujo prescindiendo de él: la vía Fast *es* el detector de movimiento aprendido.

> Tres lecciones transferibles para un clasificador multimodal con datos escasos. **Dónde fusionar importa más que cómo**: mover la fusión de la última capa a la penúltima ganó más que elegir el operador. **La fusión intermedia ahorra parámetros**, no solo mejora precisión. Y **la inicialización identidad de un módulo de fusión es una técnica general**: hacer que un módulo nuevo empiece siendo la operación trivial añade capacidad sin desestabilizar un modelo ya entrenado, el mismo principio de las conexiones residuales, los adapters y LoRA.

---

## Notas y enlaces

- Código en `github.com/feichtenhofer/twostreamfusion` (MatConvNet). Sin batch normalization; batch de 96 videos y learning rate $10^{-3}$ ($5\times10^{-4}$ para VGG-16) reducido 10× al saturarse la validación. Backbones: [VGG-M-2048 y VGG-16](/papers/vggnet-simonyan-2014) ([transfer learning](/fundamentos/transfer-learning) desde ImageNet).
- Antecedente directo: [Two-Stream (2014)](/papers/two-stream-simonyan-2014), cuyas limitaciones ya anticipaban este trabajo; la [Clase 36](/clases/clase-36) construye el arco desde la 2D CNN por fotograma hasta el [flujo óptico](/fundamentos/flujo-optico) y el [reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones).
- Contraparte de la familia (b): [C3D (2015)](/papers/c3d-tran-2015), que aprende movimiento con filtros espacio-temporales desde el RGB crudo. Este paper es el primer híbrido serio entre (b) y (c): conserva la entrada de flujo y le añade el operador de (b) solo en la capa de fusión.
- Continuación: [I3D (2017)](/papers/i3d-carreira-2017) y el [inflado de convoluciones](/fundamentos/inflado-de-convoluciones). Descendiente conceptual: [SlowFast (2019)](/papers/slowfast-feichtenhofer-2019). Alternativa contemporánea a la escala temporal: [TSN (2016)](/papers/tsn-wang-2016), con muestreo disperso y consenso de segmentos en lugar de convoluciones 3D. El grupo extendió además el catálogo de operadores en *Spatiotemporal Residual Networks* (NIPS 2016) y *Spatiotemporal Multiplier Networks* (CVPR 2017).
